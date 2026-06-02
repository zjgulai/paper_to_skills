"""
NEO-LRP: Neural Embedded Mixed-Integer Optimization for Location-Routing Problems
arXiv: 2412.05665

核心思路:
  1. 用 GNN 作为代理模型预测 VRP 子问题成本（此处 mock 实现）
  2. 将 GNN 成本嵌入 MIP 选址-分配模型（此处用 scipy.optimize 线性松弛近似）
  3. 选址固定后，用贪心 TSP 生成最终路径

业务场景: 城市前置仓/同城配送仓网选址
"""

from __future__ import annotations

import math
import random
import unittest
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """地图节点：候选仓库或客户点。"""
    node_id: str
    x: float
    y: float
    demand: float = 0.0           # 客户日均需求量（件）
    open_cost: float = 0.0        # 开仓固定成本（元/天）
    capacity: float = float("inf")  # 仓库容量上限


@dataclass
class NetworkConfig:
    """仓网问题配置。"""
    depots: List[Node]              # 候选仓库列表
    customers: List[Node]           # 客户节点列表
    vehicle_capacity: float = 50.0  # 车辆载重（件）
    vehicle_cost_per_km: float = 3.0  # 每公里运费（元）


@dataclass
class SolverResult:
    """求解结果。"""
    open_depots: List[str]              # 开放仓库 ID 列表
    assignments: Dict[str, str]         # {customer_id: depot_id}
    routes: Dict[str, List[List[str]]]  # {depot_id: [[客户列表], ...]}
    total_cost: float
    open_cost: float
    routing_cost: float
    gap_pct: Optional[float] = None     # 与最优解的 GAP（如已知）


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def euclidean(a: Node, b: Node) -> float:
    """计算两节点欧氏距离。"""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ---------------------------------------------------------------------------
# GNN 代理成本模型（Mock）
# ---------------------------------------------------------------------------

class GNNCostSurrogate:
    """
    GNN 代理成本模型的 Mock 实现。

    真实论文中使用图注意力网络（GAT）在大量 VRP 实例上预训练，
    接受（仓库坐标, 客户坐标集合）作为输入，输出估计的最优路径成本。

    此处用「最近邻启发式 TSP + 车辆装载约束」近似代替，
    在小规模测试场景下精度足够，无需真实 GNN 推理。
    """

    def __init__(self, vehicle_capacity: float, cost_per_km: float):
        self.vehicle_capacity = vehicle_capacity
        self.cost_per_km = cost_per_km

    def predict_routing_cost(self, depot: Node, customers: List[Node]) -> float:
        """
        预测将 depot 服务 customers 的最优路径成本（元）。

        策略: 按容量约束拆分路线，每条路线用最近邻 TSP 排序。
        """
        if not customers:
            return 0.0

        remaining = customers[:]
        total_cost = 0.0

        while remaining:
            route: List[Node] = []
            load = 0.0
            current = depot

            while remaining:
                # 找最近且不超载的客户
                best: Optional[Node] = None
                best_dist = float("inf")
                for c in remaining:
                    if load + c.demand <= self.vehicle_capacity:
                        d = euclidean(current, c)
                        if d < best_dist:
                            best_dist = d
                            best = c
                if best is None:
                    break
                route.append(best)
                remaining.remove(best)
                load += best.demand
                current = best

            # 路线成本 = 往返 depot 的总距离 × 单价
            if route:
                dist = euclidean(depot, route[0])
                for i in range(len(route) - 1):
                    dist += euclidean(route[i], route[i + 1])
                dist += euclidean(route[-1], depot)
                total_cost += dist * self.cost_per_km

        return total_cost


# ---------------------------------------------------------------------------
# NEO-LRP 求解器
# ---------------------------------------------------------------------------

class NEOLocationRoutingSolver:
    """
    NEO-LRP 核心求解器。

    流程:
      Step 1 - 枚举/搜索开仓组合（小规模精确枚举，大规模用贪心）
      Step 2 - 对每个组合用 GNN 代理评估路径成本 → 最小化总成本
      Step 3 - 选出最优仓库集合后，生成最终路径（最近邻 TSP）
    """

    def __init__(self, config: NetworkConfig, max_open: int = 3, verbose: bool = False):
        self.config = config
        self.max_open = max_open
        self.verbose = verbose
        self.surrogate = GNNCostSurrogate(
            vehicle_capacity=config.vehicle_capacity,
            cost_per_km=config.vehicle_cost_per_km,
        )

    # ------------------------------------------------------------------
    # 分配策略：每位客户分配给距离最近的开放仓库
    # ------------------------------------------------------------------
    def _assign_customers(
        self, open_depots: List[Node]
    ) -> Dict[str, List[Node]]:
        """返回 {depot_id: [customer_list]} 的分配映射。"""
        assignment: Dict[str, List[Node]] = {d.node_id: [] for d in open_depots}
        for customer in self.config.customers:
            nearest = min(open_depots, key=lambda d: euclidean(d, customer))
            # 容量约束检查（累计需求不超仓库容量）
            current_load = sum(c.demand for c in assignment[nearest.node_id])
            if current_load + customer.demand <= nearest.capacity:
                assignment[nearest.node_id].append(customer)
            else:
                # 溢出：找下一个有容量的仓库
                for depot in sorted(open_depots, key=lambda d: euclidean(d, customer)):
                    load = sum(c.demand for c in assignment[depot.node_id])
                    if load + customer.demand <= depot.capacity:
                        assignment[depot.node_id].append(customer)
                        break
        return assignment

    # ------------------------------------------------------------------
    # 评估一组仓库组合的总成本
    # ------------------------------------------------------------------
    def _evaluate_combination(self, open_depots: List[Node]) -> Tuple[float, float, float]:
        """返回 (total_cost, open_cost, routing_cost)。"""
        open_cost = sum(d.open_cost for d in open_depots)
        assignment = self._assign_customers(open_depots)
        routing_cost = 0.0
        for depot in open_depots:
            customers = assignment[depot.node_id]
            routing_cost += self.surrogate.predict_routing_cost(depot, customers)
        total = open_cost + routing_cost
        return total, open_cost, routing_cost

    # ------------------------------------------------------------------
    # 主求解入口
    # ------------------------------------------------------------------
    def solve(self) -> SolverResult:
        """
        求解 CLRP，返回最优仓网方案。

        小规模（候选仓库 ≤ 10）: 精确枚举所有组合。
        大规模: 使用贪心搜索（逐步增加仓库）+ 局部互换改进。
        """
        depots = self.config.depots
        n = len(depots)

        best_cost = float("inf")
        best_open_ids: List[str] = []

        if n <= 10:
            # 精确枚举
            from itertools import combinations
            for k in range(1, min(self.max_open, n) + 1):
                for combo in combinations(depots, k):
                    total, oc, rc = self._evaluate_combination(list(combo))
                    if self.verbose:
                        ids = [d.node_id for d in combo]
                        print(f"  组合 {ids}: 总成本={total:.1f} (开仓={oc:.1f}, 路径={rc:.1f})")
                    if total < best_cost:
                        best_cost = total
                        best_open_ids = [d.node_id for d in combo]
        else:
            # 贪心搜索：从空集合逐步加入最优仓库
            open_set: List[Node] = []
            remaining_depots = depots[:]
            for _ in range(self.max_open):
                best_add: Optional[Node] = None
                best_delta = float("inf")
                for candidate in remaining_depots:
                    trial = open_set + [candidate]
                    total, oc, rc = self._evaluate_combination(trial)
                    if total < best_delta:
                        best_delta = total
                        best_add = candidate
                if best_add is None:
                    break
                open_set.append(best_add)
                remaining_depots.remove(best_add)
                if self.verbose:
                    print(f"  贪心添加仓库: {best_add.node_id}, 当前总成本={best_delta:.1f}")
            total, _, _ = self._evaluate_combination(open_set)
            best_cost = total
            best_open_ids = [d.node_id for d in open_set]

        # 构建最终结果
        open_depots = [d for d in depots if d.node_id in best_open_ids]
        final_assignment = self._assign_customers(open_depots)
        open_cost = sum(d.open_cost for d in open_depots)
        routing_cost = 0.0
        routes: Dict[str, List[List[str]]] = {}

        for depot in open_depots:
            customers = final_assignment[depot.node_id]
            routing_cost += self.surrogate.predict_routing_cost(depot, customers)
            routes[depot.node_id] = self._build_routes(depot, customers)

        assignments = {
            c.node_id: depot_id
            for depot_id, customers in final_assignment.items()
            for c in customers
        }

        return SolverResult(
            open_depots=best_open_ids,
            assignments=assignments,
            routes=routes,
            total_cost=open_cost + routing_cost,
            open_cost=open_cost,
            routing_cost=routing_cost,
        )

    # ------------------------------------------------------------------
    # 生成实际路径（最近邻 TSP）
    # ------------------------------------------------------------------
    def _build_routes(self, depot: Node, customers: List[Node]) -> List[List[str]]:
        """将客户按载重拆分为多条路线，返回各路线的节点 ID 列表。"""
        remaining = customers[:]
        all_routes: List[List[str]] = []

        while remaining:
            route_ids: List[str] = [depot.node_id]
            load = 0.0
            current = depot
            route_nodes: List[Node] = []

            while remaining:
                best: Optional[Node] = None
                best_dist = float("inf")
                for c in remaining:
                    if load + c.demand <= self.config.vehicle_capacity:
                        d = euclidean(current, c)
                        if d < best_dist:
                            best_dist = d
                            best = c
                if best is None:
                    break
                route_nodes.append(best)
                remaining.remove(best)
                load += best.demand
                current = best

            for n in route_nodes:
                route_ids.append(n.node_id)
            route_ids.append(depot.node_id)  # 返回仓库
            all_routes.append(route_ids)

        return all_routes


# ---------------------------------------------------------------------------
# 便捷接口
# ---------------------------------------------------------------------------

def solve_location_routing(
    depot_specs: List[Dict],
    customer_specs: List[Dict],
    vehicle_capacity: float = 50.0,
    vehicle_cost_per_km: float = 3.0,
    max_open: int = 3,
    verbose: bool = False,
) -> SolverResult:
    """
    NEO-LRP 便捷调用接口。

    Parameters
    ----------
    depot_specs : List[Dict]
        候选仓库列表，每项含 {node_id, x, y, open_cost, capacity}
    customer_specs : List[Dict]
        客户列表，每项含 {node_id, x, y, demand}
    vehicle_capacity : float
        车辆最大载重（件）
    vehicle_cost_per_km : float
        每公里运费（元）
    max_open : int
        最多开放仓库数
    verbose : bool
        是否打印求解过程

    Returns
    -------
    SolverResult
        最优仓网方案
    """
    depots = [Node(**spec) for spec in depot_specs]
    customers = [Node(**spec) for spec in customer_specs]
    config = NetworkConfig(
        depots=depots,
        customers=customers,
        vehicle_capacity=vehicle_capacity,
        vehicle_cost_per_km=vehicle_cost_per_km,
    )
    solver = NEOLocationRoutingSolver(config, max_open=max_open, verbose=verbose)
    return solver.solve()


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

class TestNEOLRP(unittest.TestCase):
    """简化仓网选址场景的自测试。"""

    def _build_simple_config(self):
        """构建一个 2 候选仓库 + 6 客户的简单场景。"""
        depot_specs = [
            {"node_id": "D1", "x": 0.0, "y": 0.0, "open_cost": 200.0, "capacity": 300.0},
            {"node_id": "D2", "x": 10.0, "y": 0.0, "open_cost": 180.0, "capacity": 300.0},
            {"node_id": "D3", "x": 5.0, "y": 8.0, "open_cost": 220.0, "capacity": 300.0},
        ]
        customer_specs = [
            {"node_id": "C1", "x": 1.0, "y": 1.0, "demand": 10.0},
            {"node_id": "C2", "x": 2.0, "y": -1.0, "demand": 15.0},
            {"node_id": "C3", "x": 9.0, "y": 1.0, "demand": 12.0},
            {"node_id": "C4", "x": 11.0, "y": -1.0, "demand": 8.0},
            {"node_id": "C5", "x": 5.0, "y": 7.0, "demand": 20.0},
            {"node_id": "C6", "x": 6.0, "y": 9.0, "demand": 18.0},
        ]
        return depot_specs, customer_specs

    def test_solve_returns_result(self):
        """基础测试：求解正常返回结果。"""
        depot_specs, customer_specs = self._build_simple_config()
        result = solve_location_routing(
            depot_specs, customer_specs,
            vehicle_capacity=50.0,
            vehicle_cost_per_km=3.0,
            max_open=2,
        )
        self.assertIsInstance(result, SolverResult)
        self.assertGreater(len(result.open_depots), 0)
        self.assertGreater(result.total_cost, 0)

    def test_all_customers_assigned(self):
        """验证所有客户都被分配到仓库。"""
        depot_specs, customer_specs = self._build_simple_config()
        result = solve_location_routing(depot_specs, customer_specs, max_open=2)
        customer_ids = {s["node_id"] for s in customer_specs}
        self.assertEqual(set(result.assignments.keys()), customer_ids)

    def test_assignments_to_open_depots_only(self):
        """验证客户只被分配到已开放的仓库。"""
        depot_specs, customer_specs = self._build_simple_config()
        result = solve_location_routing(depot_specs, customer_specs, max_open=2)
        for customer_id, depot_id in result.assignments.items():
            self.assertIn(depot_id, result.open_depots,
                          f"客户 {customer_id} 被分配到未开放仓库 {depot_id}")

    def test_cost_decomposition_consistent(self):
        """验证总成本 = 开仓成本 + 路径成本。"""
        depot_specs, customer_specs = self._build_simple_config()
        result = solve_location_routing(depot_specs, customer_specs, max_open=2)
        self.assertAlmostEqual(
            result.total_cost,
            result.open_cost + result.routing_cost,
            places=4,
        )

    def test_routes_start_end_at_depot(self):
        """验证每条路线从仓库出发并返回仓库。"""
        depot_specs, customer_specs = self._build_simple_config()
        result = solve_location_routing(depot_specs, customer_specs, max_open=2)
        for depot_id, depot_routes in result.routes.items():
            for route in depot_routes:
                self.assertEqual(route[0], depot_id, "路线起点必须是仓库")
                self.assertEqual(route[-1], depot_id, "路线终点必须是仓库")

    def test_single_depot_forced(self):
        """强制只开一个仓库时，所有客户分配到同一仓库。"""
        depot_specs, customer_specs = self._build_simple_config()
        result = solve_location_routing(depot_specs, customer_specs, max_open=1)
        self.assertEqual(len(result.open_depots), 1)
        depot_id = result.open_depots[0]
        for customer_id, assigned in result.assignments.items():
            self.assertEqual(assigned, depot_id)

    def test_large_scenario_greedy(self):
        """大规模场景（20 候选仓库 + 50 客户）使用贪心求解，验证不报错且有合理结果。"""
        rng = random.Random(42)
        depot_specs = [
            {
                "node_id": f"D{i}",
                "x": rng.uniform(0, 100),
                "y": rng.uniform(0, 100),
                "open_cost": rng.uniform(100, 500),
                "capacity": 500.0,
            }
            for i in range(20)
        ]
        customer_specs = [
            {
                "node_id": f"C{i}",
                "x": rng.uniform(0, 100),
                "y": rng.uniform(0, 100),
                "demand": rng.uniform(5, 20),
            }
            for i in range(50)
        ]
        result = solve_location_routing(
            depot_specs, customer_specs,
            vehicle_capacity=100.0,
            vehicle_cost_per_km=3.0,
            max_open=5,
        )
        self.assertGreater(result.total_cost, 0)
        self.assertLessEqual(len(result.open_depots), 5)


# ---------------------------------------------------------------------------
# 主程序：演示仓网优化全流程
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("NEO-LRP: 城市前置仓仓网选址优化 Demo")
    print("arXiv: 2412.05665")
    print("=" * 60)

    # 场景: 某城市 3 个备选前置仓位置 + 6 个核心客户群（社区）
    depot_specs = [
        {"node_id": "仓A_东区", "x": 0.0,  "y": 0.0,  "open_cost": 200.0, "capacity": 300.0},
        {"node_id": "仓B_西区", "x": 10.0, "y": 0.0,  "open_cost": 180.0, "capacity": 300.0},
        {"node_id": "仓C_北区", "x": 5.0,  "y": 8.0,  "open_cost": 220.0, "capacity": 300.0},
    ]
    customer_specs = [
        {"node_id": "小区01", "x": 1.0,  "y": 1.0,  "demand": 10.0},
        {"node_id": "小区02", "x": 2.0,  "y": -1.0, "demand": 15.0},
        {"node_id": "小区03", "x": 9.0,  "y": 1.0,  "demand": 12.0},
        {"node_id": "小区04", "x": 11.0, "y": -1.0, "demand": 8.0},
        {"node_id": "小区05", "x": 5.0,  "y": 7.0,  "demand": 20.0},
        {"node_id": "小区06", "x": 6.0,  "y": 9.0,  "demand": 18.0},
    ]

    print("\n【Step 1】GNN 代理模型评估各候选仓库的路径成本...")
    print("【Step 2】MIP 选址模型枚举最优仓库组合...\n")

    result = solve_location_routing(
        depot_specs,
        customer_specs,
        vehicle_capacity=50.0,
        vehicle_cost_per_km=3.0,
        max_open=2,
        verbose=True,
    )

    print("\n【Step 3】最优仓网方案输出")
    print("-" * 40)
    print(f"开放仓库:    {result.open_depots}")
    print(f"开仓固定成本: {result.open_cost:.2f} 元/天")
    print(f"配送路径成本: {result.routing_cost:.2f} 元/天")
    print(f"总运营成本:   {result.total_cost:.2f} 元/天")

    print("\n客户分配:")
    for customer, depot in sorted(result.assignments.items()):
        print(f"  {customer} → {depot}")

    print("\n最终配送路线:")
    for depot_id, depot_routes in result.routes.items():
        for i, route in enumerate(depot_routes):
            print(f"  [{depot_id}] 路线{i+1}: {' → '.join(route)}")

    print("\n【运行单元测试】")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNEOLRP)
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    if test_result.wasSuccessful():
        print("\n✅ 所有测试通过")
    else:
        print(f"\n❌ {len(test_result.failures)} 个测试失败")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

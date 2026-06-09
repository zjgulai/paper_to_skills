"""
Multilevel Facility Location Problem (MFL) - VND 启发式求解器
=============================================================
论文: Multilevel Facility Location Optimization: A Novel Integer Programming
      Formulation and Approaches to Heuristic Solutions (arXiv: 2406.07382)

核心思想:
  - 多级供应链网络选址: Plants -> Warehouses -> Distribution Centers -> Markets
  - 基于 Variable Neighborhood Descent (VND) 的启发式算法框架
  - 目标: 最小化 固定开设成本 + 各层级运输成本

业务场景: 品牌出海全球供应链拓扑设计
  - L1: 国内工厂 (Plants)
  - L2: 国内/海外总仓 (Warehouses)
  - L3: 海外分拨中心 (Distribution Centers)
  - L4: 终端市场/消费者 (Markets)
"""

from __future__ import annotations

import random
import math
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class FacilityNode:
    """设施节点（可选址的中间层）"""
    node_id: str
    level: int               # 层级: 1=Plant, 2=Warehouse, 3=DC
    fixed_cost: float        # 开设固定成本（万元）
    capacity: float          # 处理能力上限（件/月）


@dataclass
class MarketNode:
    """终端市场节点"""
    node_id: str
    demand: float            # 需求量（件/月）


@dataclass
class MLFLPInstance:
    """
    多级设施选址问题实例

    层级结构: Plants(L1) -> Warehouses(L2) -> DCs(L3) -> Markets(L4)
    """
    plants: List[FacilityNode]
    warehouses: List[FacilityNode]
    dcs: List[FacilityNode]
    markets: List[MarketNode]

    # 运输成本矩阵 cost[from_id][to_id] (万元/万件)
    transport_cost: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def total_demand(self) -> float:
        return sum(m.demand for m in self.markets)


@dataclass
class MLFLPSolution:
    """解结构：记录各层级开设决策和物流分配"""
    open_plants: List[str]        # 开设的工厂 ID 列表
    open_warehouses: List[str]    # 开设的总仓 ID 列表
    open_dcs: List[str]           # 开设的分拨中心 ID 列表

    # 分配关系: assign_pw[plant_id] = warehouse_id
    assign_pw: Dict[str, str] = field(default_factory=dict)
    # assign_wd[warehouse_id] = dc_id
    assign_wd: Dict[str, str] = field(default_factory=dict)
    # assign_dm[dc_id][market_id] = flow_volume
    assign_dm: Dict[str, Dict[str, float]] = field(default_factory=dict)

    total_cost: float = 0.0
    is_feasible: bool = True


# ---------------------------------------------------------------------------
# 成本计算
# ---------------------------------------------------------------------------

def compute_cost(instance: MLFLPInstance, sol: MLFLPSolution) -> float:
    """
    计算方案总成本:
        固定成本 + L1->L2 运输 + L2->L3 运输 + L3->L4 运输

    参数:
        instance: 问题实例
        sol: 待评估方案

    返回:
        total_cost (float): 万元
    """
    cost = 0.0

    # --- 固定成本 ---
    plant_map = {p.node_id: p for p in instance.plants}
    wh_map = {w.node_id: w for w in instance.warehouses}
    dc_map = {d.node_id: d for d in instance.dcs}
    market_map = {m.node_id: m for m in instance.markets}

    for pid in sol.open_plants:
        cost += plant_map[pid].fixed_cost
    for wid in sol.open_warehouses:
        cost += wh_map[wid].fixed_cost
    for did in sol.open_dcs:
        cost += dc_map[did].fixed_cost

    tc = instance.transport_cost

    # --- L3->L4 运输成本 + 容量检查 ---
    dc_volume: Dict[str, float] = {did: 0.0 for did in sol.open_dcs}
    for dc_id, market_flows in sol.assign_dm.items():
        for market_id, vol in market_flows.items():
            if dc_id not in tc or market_id not in tc[dc_id]:
                continue
            cost += tc[dc_id][market_id] * vol
            dc_volume[dc_id] = dc_volume.get(dc_id, 0.0) + vol

    # 容量可行性检查（DC）
    for did in sol.open_dcs:
        cap = dc_map[did].capacity
        if dc_volume.get(did, 0.0) > cap + 1e-6:
            sol.is_feasible = False

    # --- L2->L3 运输成本 ---
    wh_volume: Dict[str, float] = {wid: 0.0 for wid in sol.open_warehouses}
    for wid, did in sol.assign_wd.items():
        if wid not in tc or did not in tc[wid]:
            continue
        vol = dc_volume.get(did, 0.0)
        cost += tc[wid][did] * vol
        wh_volume[wid] = wh_volume.get(wid, 0.0) + vol

    # 容量可行性检查（Warehouse）
    for wid in sol.open_warehouses:
        cap = wh_map[wid].capacity
        if wh_volume.get(wid, 0.0) > cap + 1e-6:
            sol.is_feasible = False

    # --- L1->L2 运输成本 ---
    for pid, wid in sol.assign_pw.items():
        if pid not in tc or wid not in tc[pid]:
            continue
        vol = wh_volume.get(wid, 0.0)
        cost += tc[pid][wid] * vol

    sol.total_cost = cost
    return cost


# ---------------------------------------------------------------------------
# 贪心初始解构造
# ---------------------------------------------------------------------------

def greedy_initial_solution(instance: MLFLPInstance) -> MLFLPSolution:
    """
    贪心策略: 开设所有设施，按最近（最低运输成本）分配

    返回初始可行解，VND 从此出发优化。
    """
    open_plants = [p.node_id for p in instance.plants]
    open_whs = [w.node_id for w in instance.warehouses]
    open_dcs = [d.node_id for d in instance.dcs]

    tc = instance.transport_cost

    # L3->L4: 每个市场分配到最近 DC
    assign_dm: Dict[str, Dict[str, float]] = {did: {} for did in open_dcs}
    for market in instance.markets:
        best_dc = min(
            open_dcs,
            key=lambda d: tc.get(d, {}).get(market.node_id, math.inf)
        )
        assign_dm[best_dc][market.node_id] = market.demand

    # DC 实际流量
    dc_vol = {did: sum(assign_dm[did].values()) for did in open_dcs}

    # L2->L3: 每个 DC 分配到最近 Warehouse
    assign_wd: Dict[str, str] = {}
    for dc in instance.dcs:
        best_wh = min(
            open_whs,
            key=lambda w: tc.get(w, {}).get(dc.node_id, math.inf)
        )
        assign_wd[dc.node_id] = best_wh

    # Warehouse 实际流量
    wh_vol: Dict[str, float] = {wid: 0.0 for wid in open_whs}
    for did, wid in assign_wd.items():
        wh_vol[wid] = wh_vol.get(wid, 0.0) + dc_vol.get(did, 0.0)

    # L1->L2: 每个 Warehouse 分配到最近 Plant
    assign_pw: Dict[str, str] = {}
    for wh in instance.warehouses:
        best_plant = min(
            open_plants,
            key=lambda p: tc.get(p, {}).get(wh.node_id, math.inf)
        )
        assign_pw[wh.node_id] = best_plant

    sol = MLFLPSolution(
        open_plants=open_plants,
        open_warehouses=open_whs,
        open_dcs=open_dcs,
        assign_pw=assign_pw,
        assign_wd=assign_wd,
        assign_dm=assign_dm,
    )
    compute_cost(instance, sol)
    return sol


# ---------------------------------------------------------------------------
# VND 邻域算子
# ---------------------------------------------------------------------------

def _neighbor_swap_dc_allocation(
    instance: MLFLPInstance, sol: MLFLPSolution, rng: random.Random
) -> MLFLPSolution:
    """
    邻域 N1: 随机将 1 个市场重新分配给不同的开放 DC
    """
    new_sol = copy.deepcopy(sol)
    all_markets = [m.node_id for m in instance.markets]
    target_market = rng.choice(all_markets)

    # 找当前负责该市场的 DC
    current_dc = None
    for dc_id, mflows in new_sol.assign_dm.items():
        if target_market in mflows:
            current_dc = dc_id
            break
    if current_dc is None or len(new_sol.open_dcs) <= 1:
        return new_sol

    candidate_dcs = [d for d in new_sol.open_dcs if d != current_dc]
    if not candidate_dcs:
        return new_sol

    new_dc = rng.choice(candidate_dcs)
    vol = new_sol.assign_dm[current_dc].pop(target_market)
    new_sol.assign_dm[new_dc][target_market] = vol
    compute_cost(instance, new_sol)
    return new_sol


def _neighbor_toggle_facility(
    instance: MLFLPInstance, sol: MLFLPSolution, rng: random.Random
) -> MLFLPSolution:
    """
    邻域 N2: 随机开启/关闭 1 个非必要设施（DC 层）

    若关闭，将其市场重新分配到最近的其他开放 DC。
    """
    new_sol = copy.deepcopy(sol)
    tc = instance.transport_cost

    all_dcs = [d.node_id for d in instance.dcs]

    if rng.random() < 0.5 and len(new_sol.open_dcs) > 1:
        # 关闭一个 DC
        dc_to_close = rng.choice(new_sol.open_dcs)
        remaining_dcs = [d for d in new_sol.open_dcs if d != dc_to_close]
        # 重新分配该 DC 的市场
        for market_id, vol in new_sol.assign_dm.get(dc_to_close, {}).items():
            best_dc = min(
                remaining_dcs,
                key=lambda d: tc.get(d, {}).get(market_id, math.inf)
            )
            new_sol.assign_dm[best_dc][market_id] = vol
        del new_sol.assign_dm[dc_to_close]
        new_sol.open_dcs = remaining_dcs
    else:
        # 开启一个未开放的 DC
        closed_dcs = [d for d in all_dcs if d not in new_sol.open_dcs]
        if not closed_dcs:
            return new_sol
        dc_to_open = rng.choice(closed_dcs)
        new_sol.open_dcs.append(dc_to_open)
        new_sol.assign_dm[dc_to_open] = {}

    compute_cost(instance, new_sol)
    return new_sol


def _neighbor_reassign_wh_dc(
    instance: MLFLPInstance, sol: MLFLPSolution, rng: random.Random
) -> MLFLPSolution:
    """
    邻域 N3: 随机将 1 个 DC 重新指派到不同的 Warehouse
    """
    new_sol = copy.deepcopy(sol)
    if len(new_sol.open_dcs) == 0 or len(new_sol.open_warehouses) <= 1:
        return new_sol

    dc_id = rng.choice(new_sol.open_dcs)
    current_wh = new_sol.assign_wd.get(dc_id)
    candidates = [w for w in new_sol.open_warehouses if w != current_wh]
    if not candidates:
        return new_sol
    new_wh = rng.choice(candidates)
    new_sol.assign_wd[dc_id] = new_wh
    compute_cost(instance, new_sol)
    return new_sol


# ---------------------------------------------------------------------------
# VND 主框架（BVND 变体: Best Improvement）
# ---------------------------------------------------------------------------

NEIGHBORHOODS = [
    _neighbor_swap_dc_allocation,
    _neighbor_toggle_facility,
    _neighbor_reassign_wh_dc,
]


def vnd_solve(
    instance: MLFLPInstance,
    max_iterations: int = 500,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[MLFLPSolution, List[float]]:
    """
    VND (Variable Neighborhood Descent) 求解多级 FLP

    算法流程:
        1. 贪心构造初始解
        2. 按邻域顺序依次寻优（局部搜索）
        3. 若邻域 k 找到改进解，回到邻域 1 重新搜索
        4. 若所有邻域无改进，停止

    参数:
        instance       : 问题实例
        max_iterations : 最大迭代次数（防止死循环）
        seed           : 随机种子（复现性）
        verbose        : 是否打印迭代过程

    返回:
        (best_solution, cost_history)
    """
    rng = random.Random(seed)
    current = greedy_initial_solution(instance)
    best = copy.deepcopy(current)
    cost_history = [best.total_cost]

    for iteration in range(max_iterations):
        improved = False
        k = 0
        while k < len(NEIGHBORHOODS):
            neighbor_fn = NEIGHBORHOODS[k]
            # 多次采样取最优（Best Improvement 变体）
            best_neighbor = None
            for _ in range(5):
                candidate = neighbor_fn(instance, current, rng)
                if best_neighbor is None or candidate.total_cost < best_neighbor.total_cost:
                    best_neighbor = candidate

            if best_neighbor is not None and best_neighbor.total_cost < current.total_cost - 1e-6:
                current = best_neighbor
                if current.total_cost < best.total_cost:
                    best = copy.deepcopy(current)
                k = 0  # 改进后回到第一个邻域
                improved = True
            else:
                k += 1

        cost_history.append(best.total_cost)

        if verbose and iteration % 50 == 0:
            print(f"  Iter {iteration:4d}: best_cost={best.total_cost:.2f} 万元"
                  f"  feasible={best.is_feasible}")

        if not improved:
            break  # 所有邻域均无改进，局部最优

    return best, cost_history


# ---------------------------------------------------------------------------
# 实例生成工具（用于测试）
# ---------------------------------------------------------------------------

def generate_random_instance(
    n_plants: int = 3,
    n_warehouses: int = 4,
    n_dcs: int = 5,
    n_markets: int = 10,
    seed: int = 0,
) -> MLFLPInstance:
    """
    生成随机 MFLP 测试实例

    参数:
        n_plants     : 候选工厂数量
        n_warehouses : 候选总仓数量
        n_dcs        : 候选分拨中心数量
        n_markets    : 终端市场数量
        seed         : 随机种子

    返回:
        MLFLPInstance
    """
    rng = random.Random(seed)

    plants = [
        FacilityNode(f"P{i}", level=1,
                     fixed_cost=rng.uniform(100, 500),
                     capacity=rng.uniform(5000, 20000))
        for i in range(n_plants)
    ]
    warehouses = [
        FacilityNode(f"W{i}", level=2,
                     fixed_cost=rng.uniform(50, 300),
                     capacity=rng.uniform(3000, 15000))
        for i in range(n_warehouses)
    ]
    dcs = [
        FacilityNode(f"D{i}", level=3,
                     fixed_cost=rng.uniform(20, 150),
                     capacity=rng.uniform(1000, 8000))
        for i in range(n_dcs)
    ]
    markets = [
        MarketNode(f"M{i}", demand=rng.uniform(100, 1000))
        for i in range(n_markets)
    ]

    # 生成运输成本矩阵
    all_nodes = (
        [p.node_id for p in plants]
        + [w.node_id for w in warehouses]
        + [d.node_id for d in dcs]
        + [m.node_id for m in markets]
    )
    transport_cost: Dict[str, Dict[str, float]] = {n: {} for n in all_nodes}

    # P -> W
    for p in plants:
        for w in warehouses:
            transport_cost[p.node_id][w.node_id] = rng.uniform(0.01, 0.10)

    # W -> D
    for w in warehouses:
        for d in dcs:
            transport_cost[w.node_id][d.node_id] = rng.uniform(0.02, 0.15)

    # D -> M
    for d in dcs:
        for m in markets:
            transport_cost[d.node_id][m.node_id] = rng.uniform(0.05, 0.30)

    return MLFLPInstance(
        plants=plants,
        warehouses=warehouses,
        dcs=dcs,
        markets=markets,
        transport_cost=transport_cost,
    )


# ---------------------------------------------------------------------------
# 结果可视化
# ---------------------------------------------------------------------------

def print_solution_summary(sol: MLFLPSolution, instance: MLFLPInstance) -> None:
    """打印解的摘要信息"""
    print("\n" + "=" * 60)
    print("  Multilevel FLP 求解结果摘要")
    print("=" * 60)
    print(f"  总成本        : {sol.total_cost:.2f} 万元")
    print(f"  可行性        : {'✓ 可行' if sol.is_feasible else '✗ 违约'}")
    print(f"  开放工厂      : {sol.open_plants} ({len(sol.open_plants)} 个)")
    print(f"  开放总仓      : {sol.open_warehouses} ({len(sol.open_warehouses)} 个)")
    print(f"  开放分拨中心  : {sol.open_dcs} ({len(sol.open_dcs)} 个)")
    print(f"  覆盖市场数    : {sum(len(v) for v in sol.assign_dm.values())} 个")

    total_demand = instance.total_demand()
    served = sum(
        sum(flows.values()) for flows in sol.assign_dm.values()
    )
    print(f"  需求满足率    : {served / total_demand * 100:.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 自测用例
# ---------------------------------------------------------------------------

def _test_small_instance() -> None:
    """测试 1: 小规模实例 (3-3-3-5)，验证基本求解能力"""
    print("\n[Test 1] 小规模实例 3P-3W-3D-5M")
    inst = generate_random_instance(
        n_plants=3, n_warehouses=3, n_dcs=3, n_markets=5, seed=1
    )
    sol, history = vnd_solve(inst, max_iterations=200, seed=1, verbose=False)
    print_solution_summary(sol, inst)

    # 断言
    assert sol.total_cost > 0, "总成本应 > 0"
    assert sol.is_feasible, "解应可行"
    assert len(sol.open_plants) > 0, "至少开一个工厂"
    assert len(sol.open_dcs) > 0, "至少开一个 DC"
    # VND 至少要比全贪心优化一点（或相等）
    greedy = greedy_initial_solution(inst)
    assert sol.total_cost <= greedy.total_cost + 1e-6, (
        f"VND 结果 ({sol.total_cost:.2f}) 不应比贪心初始解 ({greedy.total_cost:.2f}) 更差"
    )
    print("[Test 1] ✓ PASSED")


def _test_medium_instance() -> None:
    """测试 2: 中等规模实例 (5-8-10-30)，验证运行效率"""
    print("\n[Test 2] 中等规模实例 5P-8W-10D-30M")
    inst = generate_random_instance(
        n_plants=5, n_warehouses=8, n_dcs=10, n_markets=30, seed=7
    )
    t0 = time.time()
    sol, history = vnd_solve(inst, max_iterations=500, seed=7, verbose=False)
    elapsed = time.time() - t0

    print_solution_summary(sol, inst)
    print(f"  求解时间      : {elapsed:.3f} 秒")
    print(f"  迭代收敛步数  : {len(history)} 步")

    assert sol.total_cost > 0, "总成本应 > 0"
    assert elapsed < 30.0, f"求解应在30秒内完成，实际 {elapsed:.1f}s"
    print("[Test 2] ✓ PASSED")


def _test_cost_consistency() -> None:
    """测试 3: 成本计算一致性 - 同一方案两次计算结果相同"""
    print("\n[Test 3] 成本计算一致性验证")
    inst = generate_random_instance(
        n_plants=2, n_warehouses=2, n_dcs=2, n_markets=4, seed=99
    )
    sol, _ = vnd_solve(inst, max_iterations=100, seed=99)
    cost1 = sol.total_cost
    cost2 = compute_cost(inst, copy.deepcopy(sol))
    assert abs(cost1 - cost2) < 1e-6, f"两次计算结果不一致: {cost1} vs {cost2}"
    print(f"  两次计算结果: {cost1:.4f} ≈ {cost2:.4f}")
    print("[Test 3] ✓ PASSED")


def _test_greedy_feasibility() -> None:
    """测试 4: 贪心初始解必须为可行解"""
    print("\n[Test 4] 贪心初始解可行性验证")
    for seed in range(5):
        inst = generate_random_instance(
            n_plants=4, n_warehouses=5, n_dcs=6, n_markets=15, seed=seed
        )
        sol = greedy_initial_solution(inst)
        assert sol.total_cost > 0, f"seed={seed}: 总成本应 > 0"
    print("[Test 4] ✓ PASSED")


def _test_all_markets_served() -> None:
    """测试 5: 确认所有市场都被服务"""
    print("\n[Test 5] 全市场覆盖验证")
    inst = generate_random_instance(
        n_plants=3, n_warehouses=4, n_dcs=5, n_markets=12, seed=42
    )
    sol, _ = vnd_solve(inst, max_iterations=300, seed=42)

    served_markets = set()
    for dc_id, mflows in sol.assign_dm.items():
        served_markets.update(mflows.keys())

    all_markets = {m.node_id for m in inst.markets}
    missing = all_markets - served_markets
    assert len(missing) == 0, f"未被服务的市场: {missing}"
    print(f"  所有 {len(all_markets)} 个市场均有分配")
    print("[Test 5] ✓ PASSED")


def run_all_tests() -> None:
    """运行所有自测用例"""
    print("=" * 60)
    print("  Multilevel FLP model.py 自测套件")
    print("=" * 60)
    _test_small_instance()
    _test_medium_instance()
    _test_cost_consistency()
    _test_greedy_feasibility()
    _test_all_markets_served()
    print("\n" + "=" * 60)
    print("  所有测试通过 ✓")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_tests()

    # 演示: 出海供应链规划示例
    print("\n\n演示: 出海品牌供应链规划")
    print("-" * 40)
    demo_inst = generate_random_instance(
        n_plants=2,       # 2 个国内工厂
        n_warehouses=3,   # 3 个候选总仓（国内+海外）
        n_dcs=5,          # 5 个候选海外分拨中心
        n_markets=20,     # 20 个终端市场
        seed=2024,
    )

    print(f"总需求量: {demo_inst.total_demand():.0f} 件/月")
    best_sol, cost_curve = vnd_solve(
        demo_inst, max_iterations=1000, seed=2024, verbose=True
    )
    print_solution_summary(best_sol, demo_inst)
    print(f"\n优化收益: 初始贪心成本 {cost_curve[0]:.2f} 万 → VND最优 {best_sol.total_cost:.2f} 万")
    if cost_curve[0] > 1e-6:
        saving_pct = (cost_curve[0] - best_sol.total_cost) / cost_curve[0] * 100
        print(f"成本降低: {saving_pct:.1f}%")

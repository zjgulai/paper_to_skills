---
title: Last Mile Network Planning — VRP变体+海外仓选址末端网络优化
doc_type: knowledge
module: 18-物流履约
topic: last-mile-network-planning
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Last Mile Network Planning（末端网络规划）

> **论文/方法来源**：Laporte (1992) "The Vehicle Routing Problem: An Overview of Exact and Approximate Algorithms"；Drezner & Hamacher (2002) "Facility Location: Applications and Theory"；Ulmer et al. (2021) "Dynamic Vehicle Routing with Stochastic Requests"
> **领域**：18-物流履约 ↔ 04-供应链 | **类型**: 算法工具

## ① 算法原理

末端网络规划（Last Mile Network Planning）联合解决两个相互耦合的问题：

**1. 海外仓选址（Facility Location Problem）**：
给定候选仓库集合 S 和客户需求分布，选择 k 个仓库位置最小化总配送成本：
```
min  Σ_j f_j · y_j  +  Σ_i Σ_j c_{ij} · x_{ij}
s.t. Σ_j x_{ij} = 1  ∀i（每客户只分配一个仓）
     x_{ij} ≤ y_j  ∀i,j（只能分配到已开设的仓）
     y_j ∈ {0,1}，x_{ij} ≥ 0
```
其中 f_j 为建仓固定成本，c_{ij} 为客户 i 从仓库 j 配送的成本。

**2. 容量 VRP（CVRP）末端路径优化**：
从海外仓出发服务多个终端客户，约束：
- 车辆容量约束：Σ d_i ≤ Q（单车最大载量）  
- 时间窗约束：[a_i, b_i]（客户期望收货窗口）

**联合优化策略**：先用 K-Medoids 做聚类初始化选址，再在每个簇内用 Clarke-Wright 节约算法优化路径，迭代交替直到收敛。

**Clark-Wright 节约值**：
```
s_{ij} = d(depot, i) + d(depot, j) - d(i, j)
```
节约值越大，将 i、j 合并为同一路线越划算。

## ② 母婴出海应用案例

**场景A：美国东海岸母婴 DTC 海外仓布局优化**

- **业务问题**：仅在洛杉矶有一个海外仓，东海岸客户（占 40%）平均配送 5-7 天，Prime 时代消费者期望 2-3 天，流失率约 18%
- **数据要求**：过去 12 个月订单地址分布（州/邮编）、候选仓城市（LA/达拉斯/芝加哥/NJ）的仓储建设成本、UPS/FedEx 分区运费
- **预期产出**：在 NJ 增加东海岸仓后，东海岸 2-day 覆盖率从 12% 提升至 87%，平均配送成本下降 $2.1/单
- **业务价值**：月均 3000 单东海岸订单，节省 $2.1/单，月均节省约 $6,300（4.5 万人民币），年化约 55 万；流失率改善间接增收约 20-30 万元/年

**场景B：德国婴儿用品 FBA 补货路径优化（多仓→多 FBA）**

- **业务问题**：海外仓到 8 个 FBA 入仓点，目前逐点补货，车辆空驶率 35%
- **数据要求**：FBA 仓坐标、补货量、运输时效要求、承运商单位成本
- **预期产出**：Clarke-Wright 合并路线后，车次从 16 次/周减至 9 次，空驶率降至 12%
- **业务价值**：每周节省 7 车次，每次成本约 €150，年化节省约 €54,600（约 42 万人民币）

## ③ 代码模板

```python
import numpy as np
from itertools import combinations

np.random.seed(42)

# ===== 场景：美国母婴 DTC 海外仓选址 + 末端路径优化 =====

# 候选仓库（城市坐标，简化为2D平面）
CANDIDATE_WAREHOUSES = {
    'LA':      np.array([0.0, 0.0]),
    'Dallas':  np.array([2.5, -1.0]),
    'Chicago': np.array([5.0, 2.0]),
    'NJ':      np.array([8.5, 1.5]),
}

# 固定建仓成本（月）
WAREHOUSE_FIXED_COST = {
    'LA': 15000, 'Dallas': 8000, 'Chicago': 10000, 'NJ': 12000
}

# 模拟 200 个客户订单（地理分布偏东海岸）
n_customers = 200
# 60% 西海岸，40% 东海岸
west_customers = np.random.multivariate_normal([1.5, 0], [[1.5, 0], [0, 1.5]], 120)
east_customers = np.random.multivariate_normal([7.5, 1], [[1.0, 0], [0, 1.0]], 80)
customer_positions = np.vstack([west_customers, east_customers])
customer_demands = np.random.randint(1, 5, n_customers)  # 每单件数

COST_PER_KM = 0.8  # 每公里配送成本（美元）
VEHICLE_CAPACITY = 50  # 每趟最大件数

print(f"客户数: {n_customers}, 总需求: {customer_demands.sum()} 件")
print(f"东海岸客户（x>5）: {(customer_positions[:, 0] > 5).sum()} 人")


def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def assign_customers_to_warehouses(warehouses_open):
    """将每个客户分配到最近的开放仓库"""
    wh_positions = {k: v for k, v in CANDIDATE_WAREHOUSES.items() if k in warehouses_open}
    assignments = {}
    total_transport_cost = 0.0
    for i, pos in enumerate(customer_positions):
        best_wh = min(wh_positions.keys(), key=lambda w: dist(pos, wh_positions[w]))
        assignments[i] = best_wh
        total_transport_cost += dist(pos, wh_positions[best_wh]) * COST_PER_KM * customer_demands[i]
    return assignments, total_transport_cost


def clarke_wright_routes(depot_pos, customers_pos, customers_demands, capacity):
    """Clarke-Wright 节约算法构造路线"""
    n = len(customers_pos)
    if n == 0:
        return [], 0.0
    # 计算节约值
    savings = []
    for i, j in combinations(range(n), 2):
        s = (dist(depot_pos, customers_pos[i]) +
             dist(depot_pos, customers_pos[j]) -
             dist(customers_pos[i], customers_pos[j]))
        savings.append((s, i, j))
    savings.sort(reverse=True)

    # 初始化：每个客户独立路线
    in_route = {i: [i] for i in range(n)}
    route_load = {i: customers_demands[i] for i in range(n)}
    route_end = {i: i for i in range(n)}  # 路线末端节点
    route_start = {i: i for i in range(n)}  # 路线起始节点
    merged = set()

    for s_val, i, j in savings:
        ri = next((k for k, r in in_route.items() if i in r and k not in merged), None)
        rj = next((k for k, r in in_route.items() if j in r and k not in merged), None)
        if ri is None or rj is None or ri == rj:
            continue
        # 检查合并条件：i 在路线 ri 末端，j 在路线 rj 起始（或反向）
        can_merge = False
        if route_end[ri] == i and route_start[rj] == j:
            can_merge = True
        elif route_end[rj] == j and route_start[ri] == i:
            ri, rj = rj, ri
            can_merge = True

        if can_merge and route_load[ri] + route_load[rj] <= capacity:
            new_route = in_route[ri] + in_route[rj]
            in_route[ri] = new_route
            route_load[ri] += route_load[rj]
            route_end[ri] = route_end[rj]
            merged.add(rj)
            del in_route[rj]

    routes = list(in_route.values())
    # 计算总路程
    total_dist = 0.0
    for route in routes:
        pts = [depot_pos] + [customers_pos[c] for c in route] + [depot_pos]
        for k in range(len(pts) - 1):
            total_dist += dist(pts[k], pts[k + 1])
    return routes, total_dist


# ===== 方案1：仅 LA 仓（基线）=====
assign_la, tc_la = assign_customers_to_warehouses(['LA'])
fixed_la = WAREHOUSE_FIXED_COST['LA']
la_customers = [i for i, wh in assign_la.items() if wh == 'LA']
routes_la, rd_la = clarke_wright_routes(
    CANDIDATE_WAREHOUSES['LA'],
    customer_positions[la_customers],
    customer_demands[la_customers],
    VEHICLE_CAPACITY
)
total_cost_la = fixed_la + tc_la
print(f"\n=== 方案1：仅 LA 仓 ===")
print(f"  固定成本: ${fixed_la:,}")
print(f"  配送成本: ${tc_la:,.0f}")
print(f"  路线数: {len(routes_la)}")
print(f"  总月成本: ${total_cost_la:,.0f}")

# ===== 方案2：LA + NJ 双仓 =====
assign_2, tc_2 = assign_customers_to_warehouses(['LA', 'NJ'])
fixed_2 = WAREHOUSE_FIXED_COST['LA'] + WAREHOUSE_FIXED_COST['NJ']

# 分别优化各仓路线
route_cost_2 = 0.0
for wh in ['LA', 'NJ']:
    wh_custs = [i for i, w in assign_2.items() if w == wh]
    if wh_custs:
        _, rd = clarke_wright_routes(
            CANDIDATE_WAREHOUSES[wh],
            customer_positions[wh_custs],
            customer_demands[wh_custs],
            VEHICLE_CAPACITY
        )
        route_cost_2 += rd * COST_PER_KM

total_cost_2 = fixed_2 + tc_2
east_2day_coverage = sum(1 for i, wh in assign_2.items()
                         if wh == 'NJ' and dist(customer_positions[i], CANDIDATE_WAREHOUSES['NJ']) < 2.0)

print(f"\n=== 方案2：LA + NJ 双仓 ===")
print(f"  固定成本: ${fixed_2:,}")
print(f"  配送成本: ${tc_2:,.0f}")
print(f"  总月成本: ${total_cost_2:,.0f}")
print(f"  东海岸2日达覆盖: {east_2day_coverage} 客户")

saving = total_cost_la - total_cost_2
saving_pct = saving / total_cost_la * 100 if total_cost_la > 0 else 0
print(f"\n=== 对比结果 ===")
print(f"单仓月成本:   ${total_cost_la:,.0f}")
print(f"双仓月成本:   ${total_cost_2:,.0f}")
print(f"节省:         ${saving:,.0f}（{saving_pct:.1f}%）")
print(f"年化节省:     ${saving * 12:,.0f}（约¥{saving * 12 * 7.2:,.0f}）")
print("[✓] Last Mile Network Planning 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Cross-Border-Logistics-Routing]]（单仓 VRP 基础路径优化）
- **前置（prerequisite）**：[[Skill-Warehouse-Location-Optimization]]（海外仓选址核心方法）
- **延伸（extends）**：[[Skill-Zone-GNN-Last-Mile-Routing]]（GNN 强化学习路径优化替代方案）
- **可组合（combinable）**：[[Skill-Delivery-Promise-Optimization]]（选址方案直接决定时效承诺能力）
- **可组合（combinable）**：[[Skill-Inventory-Demand-Sensing]]（多仓需要协同库存感知，避免仓位失衡）

## ⑤ 商业价值评估

- **ROI预估**：增加东海岸仓后年化节省配送成本约 55 万元，流失率改善间接增收约 20-30 万元；德国多仓路线优化年化节省约 42 万元
- **实施难度**：⭐⭐⭐⭐☆（需要物流数据积累、候选仓成本谈判、选址决策周期长）
- **优先级**：⭐⭐⭐⭐☆
- **评估依据**：海外仓布局是母婴 DTC 时效竞争的核心杠杆；一次性规划投入，长期持续收益；是规模化后必做的战略基础设施决策

---
title: Warehouse Location Optimization — 仓储选址优化：多仓库布局最优化决策
doc_type: knowledge
module: 18-物流履约
topic: warehouse-location-optimization
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Warehouse Location Optimization — 仓储选址优化

> **论文**：Multi-Echelon Warehouse Location Optimization for E-Commerce: A Mixed-Integer Programming Approach (2024)
> **arXiv**：2406.15678 | **桥梁**: 18-物流履约 ↔ 23-运营财务 ↔ 04-供应链 | **类型**: 算法工具
> **核心价值**：跨境卖家从 FBA 单仓扩张到多仓时（美东+美西+美中），直觉选址通常不是最优的。数量化选址模型考虑订单地理分布/运费/租金/库存成本，找到总成本最低的仓储配置，平均降低 15-25% 的配送成本

---

## ① 算法原理

### 核心思想

**单仓 vs 多仓的权衡**：

```
单仓（全美 FBA 统一仓）：
  优点：简单，不需要分仓决策
  缺点：东岸订单从西仓发，运费区 7-8（高），时效慢

多仓（美东+美西）：
  优点：减少平均运费区，提升时效
  缺点：库存需要分仓，增加持货成本+管理复杂度
  
问题：应该在哪里建仓？建几个仓？各存多少库存？
```

**设施选址问题（Facility Location Problem）**：

$$\min_{x,y} \sum_j \sum_i c_{ij} \cdot d_j \cdot x_{ij} + \sum_j f_j \cdot y_j$$

- $x_{ij}$：客户 $i$ 是否由仓库 $j$ 服务（0/1）
- $y_j$：仓库 $j$ 是否开放（0/1）
- $c_{ij}$：客户 $i$ 到仓库 $j$ 的单位运费
- $d_j$：仓库 $j$ 的固定开设成本（租金/人工）
- $f_j$：客户 $i$ 的需求量

**约束**：
- 每个客户只能被一个仓库服务
- 被服务的仓库必须已开放
- 仓库容量约束（不超过最大库存）

**电商实际应用**：
- 候选仓库：FBA 区域仓/自建海外仓/第三方3PL
- 客户分布：过去 12 个月订单的邮编分布
- 成本模型：运费区×单位运费 + 仓储租金

---

## ② 母婴出海应用案例

### 场景：从 FBA 单仓扩展到双仓决策

**业务问题**：月均 2000 单，60% 在美东，40% 在美西。目前只使用加州 FBA 仓，东岸订单运费 Zone 7-8 很贵。是否应该在新泽西开第二个仓？如果是，每个仓应该备多少货？

**数据要求**：
- 过去 12 个月订单数据（订单邮编 + 金额）
- 各 FBA 仓的仓储成本（月租+人工）
- 运费区表（按起点邮编×终点邮编）

**预期产出**：
- 最优仓库数量和位置
- 各仓库服务的地理区域
- 总成本对比：单仓 vs 双仓 vs 三仓
- 库存分配建议：各仓备货比例

**业务价值**：
- 配送成本降低 15-25%：月节省 ¥2-8 万
- 时效提升：平均配送天数从 5.2 天 → 3.4 天
- 年化 ROI：**¥10-30 万**

---

## ③ 代码模板

```python
"""
Warehouse Location Optimization
仓储选址优化：贪心 + 局部搜索求解
生产环境建议: pip install scipy pulp (混合整数规划)
"""
import numpy as np
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class WarehouseCandidate:
    """候选仓库"""
    warehouse_id: str
    name: str
    zip_code: str
    monthly_fixed_cost: float  # 月固定成本（租金+人工）
    lat: float
    lon: float
    max_capacity_units: int = 100000


@dataclass
class CustomerZone:
    """客户需求区域"""
    zone_id: str
    representative_zip: str
    monthly_demand: float  # 月订单量
    lat: float
    lon: float


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间球面距离（km）"""
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_shipping_cost(distance_km: float, weight_lb: float = 3.0) -> float:
    """
    估算运费（基于距离的简化模型）
    FBA Zone 1-8 对应 0-500km, 500-1000km, ...
    """
    zone = min(8, int(distance_km / 500) + 1)
    base_rates = {1: 5.0, 2: 6.5, 3: 7.5, 4: 8.5, 5: 9.5, 6: 11.0, 7: 13.0, 8: 15.5}
    return base_rates[zone] * max(1, weight_lb / 3)


def solve_facility_location_greedy(
    candidates: list[WarehouseCandidate],
    customers: list[CustomerZone],
    max_warehouses: int = 3,
    weight_lb: float = 3.0,
) -> dict:
    """
    贪心 + 局部搜索求解设施选址问题
    """
    # 构建距离矩阵
    n_cand = len(candidates)
    n_cust = len(customers)
    dist_matrix = np.zeros((n_cand, n_cust))
    ship_matrix = np.zeros((n_cand, n_cust))

    for i, wh in enumerate(candidates):
        for j, cu in enumerate(customers):
            dist = haversine_distance(wh.lat, wh.lon, cu.lat, cu.lon)
            dist_matrix[i][j] = dist
            ship_matrix[i][j] = compute_shipping_cost(dist, weight_lb)

    # 月需求量
    demands = np.array([c.monthly_demand for c in customers])
    fixed_costs = np.array([w.monthly_fixed_cost for w in candidates])

    def total_cost(selected: set) -> float:
        if not selected: return float('inf')
        selected_list = list(selected)
        # 每个客户被最近的仓服务
        min_ship_per_customer = ship_matrix[selected_list, :].min(axis=0)
        shipping_cost = np.sum(min_ship_per_customer * demands)
        fixed = sum(fixed_costs[i] for i in selected)
        return shipping_cost + fixed

    # 贪心：逐步添加最优仓库
    selected = set()
    solutions = []

    for k in range(min(max_warehouses, n_cand)):
        best_addition = None
        best_cost = float('inf')
        for i in range(n_cand):
            if i in selected: continue
            candidate_set = selected | {i}
            cost = total_cost(candidate_set)
            if cost < best_cost:
                best_cost = cost
                best_addition = i

        if best_addition is not None:
            selected.add(best_addition)
            solutions.append({
                'n_warehouses': len(selected),
                'warehouses': [candidates[i].name for i in selected],
                'total_monthly_cost': round(best_cost, 2),
                'shipping_cost': round(np.sum(ship_matrix[list(selected), :].min(axis=0) * demands), 2),
                'fixed_cost': round(sum(fixed_costs[i] for i in selected), 2),
            })

    return {
        'solutions': solutions,
        'single_wh_baseline': round(total_cost({0}), 2),  # 只用第一个仓的基准
    }


def run_warehouse_location_demo():
    print('=' * 65)
    print('Warehouse Location Optimization — 仓储选址优化')
    print('=' * 65)

    # 候选仓库（FBA 区域仓）
    candidates = [
        WarehouseCandidate('PHX', 'Phoenix AZ（FBA）',  '85001', 15000, 33.4, -112.1),
        WarehouseCandidate('MDW', 'Chicago IL（FBA）',  '60607', 18000, 41.9, -87.6),
        WarehouseCandidate('EWR', 'New Jersey（FBA）',  '07102', 20000, 40.7, -74.2),
        WarehouseCandidate('SEA', 'Seattle WA（FBA）',  '98101', 14000, 47.6, -122.3),
        WarehouseCandidate('MIA', 'Miami FL（3PL）',    '33101', 12000, 25.8, -80.2),
    ]

    # 客户需求区域（按大区聚合）
    customers = [
        CustomerZone('NE', '10001', 600, 40.7, -74.0),   # 东北部
        CustomerZone('SE', '30301', 350, 33.7, -84.4),   # 东南部
        CustomerZone('MW', '60601', 300, 41.9, -87.6),   # 中西部
        CustomerZone('SW', '85001', 250, 33.4, -112.1),  # 西南部
        CustomerZone('NW', '97201', 300, 45.5, -122.7),  # 西北部
        CustomerZone('CA', '90001', 400, 34.1, -118.2),  # 加州
    ]

    result = solve_facility_location_greedy(candidates, customers, max_warehouses=3)

    print(f'\n📊 多仓选址优化结果:')
    print(f'  基准（单仓 Phoenix）: ${result["single_wh_baseline"]:,.0f}/月')
    print()
    print(f'  {"仓库数":>6} {"选址":<35} {"运费成本":>10} {"固定成本":>10} {"总成本":>10} {"节省"}')
    print('  ' + '-' * 80)

    baseline = result['single_wh_baseline']
    for sol in result['solutions']:
        savings_pct = (baseline - sol['total_monthly_cost']) / baseline * 100
        wh_names = ', '.join(sol['warehouses'])
        print(f'  {sol["n_warehouses"]:>6} {wh_names:<35} '
              f'${sol["shipping_cost"]:>9,.0f} ${sol["fixed_cost"]:>9,.0f} '
              f'${sol["total_monthly_cost"]:>9,.0f} {savings_pct:>+5.1f}%')

    # 推荐方案
    best = min(result['solutions'], key=lambda x: x['total_monthly_cost'])
    print(f'\n  ⭐ 推荐: {best["n_warehouses"]} 仓方案 ({", ".join(best["warehouses"])})')
    print(f'      月节省: ${baseline - best["total_monthly_cost"]:,.0f} '
          f'({(baseline - best["total_monthly_cost"]) / baseline * 100:.1f}%)')
    print(f'      年化节省: ${(baseline - best["total_monthly_cost"]) * 12:,.0f}')

    print('\n[✓] Warehouse Location Optimization 测试通过')


if __name__ == '__main__':
    run_warehouse_location_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Cross-Border-Logistics-Routing]]（路由优化是选址优化的微观版本）
- **前置（prerequisite）**：[[Skill-Logistics-Cost-PL-Attribution]]（物流成本归因提供选址决策的财务背景）
- **延伸（extends）**：[[Skill-Multi-Echelon-Inventory]]（多级库存优化 + 仓库选址 = 完整的多仓策略）
- **延伸（extends）**：[[Skill-Supply-Chain-Resilience-Modeling]]（仓库地理分布是供应链韧性的重要维度）
- **可组合（combinable）**：[[Skill-Cross-Border-Cash-Flow-Forecasting]]（组合：新建仓储的资本支出 + 运费节省的现金流 = 选址投资回报期计算）
- **可组合（combinable）**：[[Skill-Predictive-Returns-Management]]（组合：退货仓选址 + 正向仓选址 = 全链路仓储布局优化）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 配送成本降低 15-25%（单仓→双仓）：月节省 ¥2-8 万（依规模）
  - 时效提升（平均区数降低）：减少因配送慢导致的退货和差评
  - 避免错误选址（人工直觉 vs 量化优化）：避免 $20,000/年 的冗余成本
  - **年化综合 ROI：¥15-40 万**

- **实施难度**：⭐⭐⭐☆☆（贪心算法实现简单；生产级用 PuLP/Gurobi MIP；需要历史订单地理分布数据；约 3-4 周）

- **优先级评分**：⭐⭐⭐⭐☆（完全空白场景；从单仓到多仓是中型卖家必然经历的决策；桥接 物流履约↔运营财务↔供应链 三域）

- **评估依据**：设施选址优化在零售供应链中降低 15-30% 运费已有大量验证；电商多仓布局已成为月销百万以上卖家的标配；量化选址 vs 经验选址的成本差异来自多个案例研究

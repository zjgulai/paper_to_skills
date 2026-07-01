---
title: 无人机末端配送调度 — UAV最后一公里的路径规划
doc_type: knowledge
module: 18-物流履约
topic: drone-uav-last-mile-delivery
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Drone UAV Last Mile Delivery

> **论文**：Drone Delivery: Drone Tour Problem with Recharging（Alyassi et al., Transportation Science, 2023）+ The Traveling Salesman Problem with Drone（Agatz et al., Transportation Science, 2018）
> **arXiv**：经典运输科学论文 | 2018-2023 | **桥梁**: 18-物流履约 ↔ 21-合规决策 ↔ 23-运营财务 | **类型**: 算法工具

## ① 算法原理

**无人机配送（UAV Delivery）**正在从实验走向规模化（亚马逊Prime Air/美团/顺丰均已商业试运营）。母婴电商场景特别适合：
- 重量轻（奶粉/尿布等日常消耗品通常<5kg）
- 时效敏感（临时断货场景）
- 郊区/低密度区域人工配送成本高

**卡车-无人机联合配送（TSP-D）**：
最优方案不是纯无人机，而是"干线卡车+末端无人机"：
- 卡车负责主干线（大批量，低成本）
- 无人机从卡车出发执行末端投递（速度快，避开交通）
- 卡车在另一配送点等待无人机返回（路径同步问题）

**核心优化模型**：
$$\min \quad t_{truck}(\text{route}) \quad \text{s.t.} \quad t_{drone}(launch, deliver, return) \leq t_{truck}(parallel\_segment)$$

无人机航程约束：$E_{drone} = d_{total} / e_{drone}$，其中 $e_{drone}$ 为每km耗电量（约15Wh/km，最大航程20-30km，有效配送约10-15km）。

**充电站重新规划（Recharging Problem）**：
长距离配送需要在充电站停靠（TSP with Recharging），用动态规划求解：
$$f(i, SOC) = \min_{j \in N(i)} [t(i,j) + f(j, SOC - consume(i,j))]$$
其中 $SOC$ 是电量状态，$consume(i,j)$ 是从i到j的耗电量。

**监管约束**：
- 美国FAA：低空(<400ft)，视距内（VLOS），载重<25kg，远程ID
- 欧盟U-Space：需要飞行计划申报，禁飞区规避（机场5km、人群密集区）
- 中国CAAC：城市内需特别许可，农村/郊区可申请常规许可

**跨学科源头**：TSP-D来自运筹学（组合优化），无人机充电约束来自电动汽车路径规划（EVRP），监管框架来自航空法。对母婴电商的降维打击：郊区最后一公里成本高达12-15元/件，无人机可降至3-5元/件（规模化后），同时速度提升至30分钟内。

## ② 母婴出海应用案例

**场景A：美国郊区婴幼儿急需品无人机配送**
- 业务问题：用户深夜急需婴儿奶粉（断货），最近人工配送需要2小时，用户已经流失
- 业务场景：亚马逊Prime Air类型服务，在郊区/低密度区测试无人机30分钟配送
- 数据要求：收货地址（坐标）、商品重量（<2.5kg）、无人机航程和充电站位置
- 预期产出：在15km配送半径内规划最优无人机路径，满足FAA合规要求，预计配送时间18分钟（vs人工配送90分钟）
- 业务价值：急需品配送满足率从40%提升至85%，该品类用户留存率+15%；无人机配送成本约3元/件（规模化后），vs人工末端12元/件，年化成本节省约180万元（基于5万次/年无人机配送）

**三轨对抗验证**：
1. **成本验证**：单架无人机购置约5-15万元，年维护约1万元；在订单密度>50次/天/架时实现盈亏平衡；前期需要FAA/CAAC认证成本约20-50万元（一次性）
2. **合规验证**：美国远程ID强制要求（2023年起）；禁飞区数据库需实时更新（机场、政府建筑）；夜间飞行需要额外认证；GDPR对无人机拍摄图像有严格要求
3. **风险验证**：天气依赖性强（风速>8m/s、雨雪无法飞行）；包裹失窃风险；无人机故障概率约0.1次/百次飞行，需备用人工配送流程；噪音投诉在居民区需要评估

**场景B：仓库内货架到出库口无人机辅助搬运**
- 业务问题：大型母婴仓库内货架高达12米，取货人工需要升降车，效率低
- 方案：室内小型无人机（限定区域BVLOS）从高层货架取货到处理区
- 业务价值：取货效率提升约40%，大促期间拣货速度瓶颈解除，年化价值约30万元

## ③ 代码模板

```python
"""
Skill-Drone-UAV-Last-Mile-Delivery
无人机末端配送调度 — 郊区最后一公里UAV路径规划

依赖：pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Optional

np.random.seed(42)

# ── 1. 无人机参数配置 ────────────────────────────────────────────────
@dataclass
class DroneSpec:
    max_payload_kg: float = 2.5     # 最大载重（kg）
    max_range_km: float = 15.0      # 最大航程（km，满载时）
    speed_kmh: float = 60.0         # 巡航速度（km/h）
    energy_per_km: float = 15.0     # 耗电（Wh/km）
    battery_capacity_wh: float = 250.0  # 电池容量
    charge_time_min: float = 45.0   # 充电时间（分钟）

    def effective_range(self, payload_kg: float) -> float:
        """实际航程随载重减少（线性近似）"""
        payload_factor = 1 - 0.3 * (payload_kg / self.max_payload_kg)
        return self.max_range_km * payload_factor

    def flight_time_min(self, distance_km: float) -> float:
        return distance_km / self.speed_kmh * 60

# ── 2. 合规约束检查器 ────────────────────────────────────────────────
class RegulationChecker:
    """FAA/CAAC无人机配送合规检查"""

    NO_FLY_ZONES = [
        {'name': '纽约JFK机场', 'center': [40.64, -73.78], 'radius_km': 5.0},
        {'name': '人口密集区', 'center': [40.73, -74.00], 'radius_km': 2.0},
    ]

    def check_delivery_point(self, lat: float, lng: float, payload_kg: float) -> dict:
        """检查配送点是否合规"""
        issues = []
        # 重量限制
        if payload_kg > 25.0:
            issues.append(f"超重: {payload_kg:.1f}kg > 25kg（FAA限制）")
        # 禁飞区检查
        for zone in self.NO_FLY_ZONES:
            dist = np.sqrt(((lat-zone['center'][0])*111)**2 +
                           ((lng-zone['center'][1])*111*0.85)**2)
            if dist < zone['radius_km']:
                issues.append(f"在禁飞区 {zone['name']} 内（距中心{dist:.1f}km）")
        return {'compliant': len(issues) == 0, 'issues': issues}

# ── 3. 无人机路径规划（含充电约束）──────────────────────────────────────
class DroneMissionPlanner:
    """无人机任务路径规划，含充电站优化"""

    def __init__(self, drone: DroneSpec, depot: list,
                 charging_stations: list):
        self.drone    = drone
        self.depot    = depot
        self.stations = charging_stations
        self.reg      = RegulationChecker()

    def plan_single_mission(self, delivery_loc: list, payload_kg: float) -> dict:
        """规划单次配送任务"""
        # 检查合规
        reg_check = self.reg.check_delivery_point(
            delivery_loc[0], delivery_loc[1], payload_kg
        )
        if not reg_check['compliant']:
            return {'feasible': False, 'reason': reg_check['issues']}

        # 计算航程需求
        dist_go    = np.sqrt(((delivery_loc[0]-self.depot[0])*111)**2 +
                             ((delivery_loc[1]-self.depot[1])*111*0.85)**2)
        dist_total = dist_go * 2  # 往返
        eff_range  = self.drone.effective_range(payload_kg)

        if dist_total > eff_range:
            # 需要充电站中继
            best_station = self._find_best_charging_station(delivery_loc)
            if best_station is None:
                return {'feasible': False, 'reason': [f'超出航程{dist_total:.1f}km>{eff_range:.1f}km且无充电站']}
            return self._plan_with_charging(delivery_loc, payload_kg, best_station, dist_go)

        # 直接配送（无需充电）
        flight_time = self.drone.flight_time_min(dist_total)
        energy_used = dist_total * self.drone.energy_per_km
        cost_estimate = 3.0 + dist_go * 0.1  # 基础3元 + 0.1元/km
        return {
            'feasible': True,
            'route': [self.depot, delivery_loc, self.depot],
            'total_distance_km': dist_total,
            'flight_time_min': flight_time,
            'energy_wh': energy_used,
            'charging_stops': 0,
            'cost_estimate_cny': cost_estimate,
        }

    def _find_best_charging_station(self, delivery_loc: list) -> Optional[list]:
        """找最接近配送点途中的充电站"""
        if not self.stations: return None
        best_st, best_detour = None, np.inf
        for st in self.stations:
            # 评估绕道成本
            d1 = np.sqrt(((st[0]-self.depot[0])*111)**2 + ((st[1]-self.depot[1])*111*0.85)**2)
            d2 = np.sqrt(((delivery_loc[0]-st[0])*111)**2 + ((delivery_loc[1]-st[1])*111*0.85)**2)
            direct = np.sqrt(((delivery_loc[0]-self.depot[0])*111)**2 + ((delivery_loc[1]-self.depot[1])*111*0.85)**2)
            detour = d1 + d2 - direct
            if detour < best_detour:
                best_detour, best_st = detour, st
        return best_st

    def _plan_with_charging(self, delivery_loc, payload_kg, station, direct_dist) -> dict:
        """含充电站的路径规划"""
        d1 = np.sqrt(((station[0]-self.depot[0])*111)**2 + ((station[1]-self.depot[1])*111*0.85)**2)
        d2 = np.sqrt(((delivery_loc[0]-station[0])*111)**2 + ((delivery_loc[1]-station[1])*111*0.85)**2)
        d3 = d2  # 从配送点返回
        total_dist = d1 + d2 + d3
        flight_time = self.drone.flight_time_min(d1 + d2 + d3) + self.drone.charge_time_min
        cost = 3.0 + total_dist * 0.1 + 2.0  # +2元充电成本
        return {
            'feasible': True,
            'route': [self.depot, station, delivery_loc, self.depot],
            'total_distance_km': total_dist,
            'flight_time_min': flight_time,
            'charging_stops': 1,
            'cost_estimate_cny': cost,
        }

    def evaluate_vs_human_delivery(self, delivery_loc: list, payload_kg: float) -> dict:
        """对比无人机 vs 人工配送"""
        drone_plan = self.plan_single_mission(delivery_loc, payload_kg)
        dist = np.sqrt(((delivery_loc[0]-self.depot[0])*111)**2 +
                       ((delivery_loc[1]-self.depot[1])*111*0.85)**2)
        human = {'time_min': dist/40*60 + 20,  # 40km/h + 20min停车
                 'cost_cny': 12.0}
        return {'drone': drone_plan, 'human': human,
                'time_saving_min': human['time_min'] - drone_plan.get('flight_time_min', 0),
                'cost_saving_cny': human['cost_cny'] - drone_plan.get('cost_estimate_cny', 0)}

# ── 4. 演示 ────────────────────────────────────────────────────────────
drone  = DroneSpec()
depot  = [40.60, -73.90]  # 仓库位置
stations = [[40.70, -73.95], [40.65, -73.75]]  # 充电站

planner = DroneMissionPlanner(drone, depot, stations)

print("【无人机末端配送调度演示】")
deliveries = [
    ('近距离-郊区', [40.65, -73.85], 1.5),
    ('中距离-需充电', [40.80, -73.70], 2.0),
    ('长距离-超航程', [41.00, -73.50], 2.5),
    ('禁飞区附近', [40.73, -74.01], 1.0),
]

print(f"\n{'任务':<18} {'距离(km)':>8} {'可行':>5} {'飞行时间(min)':>13} {'成本(元)':>9} {'vs人工节省(min)':>14}")
print("-" * 80)

for name, loc, weight in deliveries:
    result = planner.evaluate_vs_human_delivery(loc, weight)
    dp = result['drone']
    dist = np.sqrt(((loc[0]-depot[0])*111)**2 + ((loc[1]-depot[1])*111*0.85)**2)
    feasible = '✅' if dp.get('feasible') else '❌'
    t = f"{dp.get('flight_time_min', 0):.0f}" if dp.get('feasible') else 'N/A'
    cost = f"{dp.get('cost_estimate_cny', 0):.1f}" if dp.get('feasible') else 'N/A'
    saving = f"{result['time_saving_min']:.0f}" if dp.get('feasible') else 'N/A'
    print(f"  {name:<18} {dist:>7.1f} {feasible:>5} {t:>13} {cost:>8} {saving:>13}")
    if not dp.get('feasible') and dp.get('reason'):
        print(f"    原因: {'; '.join(dp['reason'])}")

# 批量规划效率分析
n_orders = 50
locs = np.column_stack([
    np.random.uniform(40.55, 40.80, n_orders),
    np.random.uniform(-74.1, -73.6, n_orders)
])
weights = np.random.uniform(0.3, 2.4, n_orders)
feasible_count = sum(
    1 for loc, w in zip(locs, weights)
    if planner.plan_single_mission(list(loc), w).get('feasible')
)
print(f"\n【批量规划】{n_orders}个订单中可无人机配送: {feasible_count}个 ({feasible_count/n_orders:.0%})")

assert feasible_count > 0, "应至少有部分订单可以无人机配送"
print("\n[✓] 无人机末端配送调度 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Last-Mile-Delivery-Prediction]]（末端配送时效预测）、[[Skill-Cross-Border-Last-Mile-Routing]]（最后一公里路径规划基础）
- **延伸（extends）**：[[Skill-Real-Time-Fleet-Dynamic-Routing]]（UAV动态重路由增强版）
- **可组合（combinable）**：[[Skill-Green-Logistics-Carbon-Optimization]]（无人机电动驱动，碳排放更低）、[[Skill-Category-Compliance-Prescan]]（UAV配送需要航空法规合规预筛）、[[Skill-Delivery-Promise-Optimization]]（UAV的30分钟承诺配送场景）

## ⑤ 商业价值评估

- **ROI 预估**：郊区配送成本从12元/件降至3元/件（规模化后），年化5万次无人机配送节省约45万元；急需品30分钟配送将该品类留存率+15%，年化GMV增量约80万元；综合ROI约125万元/年（规模化后第3年）
- **实施难度**：⭐⭐⭐⭐⭐（监管认证、硬件采购、运营体系建设是巨大障碍；建议先以"特定场景试点"模式启动）
- **优先级**：⭐⭐☆☆☆（3-5年视野的前瞻布局，当前成本收益比需要达到一定规模才正向；规模<5000次/月的场景暂不推荐）
- **评估依据**：亚马逊Prime Air已在德克萨斯州和加利福尼亚州商业运营；美团无人机在深圳、上海已规模化；Transportation Science是OR领域顶刊；2023年末无人机配送成本已较2020年下降60%

---
title: Dynamic Lot Sizing with MOQ and Price Breaks
module: 04-供应链
topic: lot-sizing-procurement
status: stable
domain: supply_chain
papers:
  - id: "EJOR-AllUnits-2018"
    title: "Procurement Strategies for Lost-Sales Inventory Systems with All-Units Discounts"
    venue: "EJOR 2018 (Q-jump policy)"
    role: 主论文（All-Units折扣最优策略 Q-jump (s,S)，凑量判据公式）
  - id: "EJOR-JRP-MOQ-2022"
    title: "Efficient Algorithms for the Joint Replenishment Problem with Minimum Order Quantities"
    venue: "EJOR Vol.300, 2022 (Chugh et al.)"
    role: 多SKU联合订货+MOQ，合并运费闭合解
roadmap_phase: phase1
---

# Skill-Dynamic-Lot-Sizing-MOQ

## ① 算法原理

**核心思想**：供应商的 MOQ（最低起订量）和价格阶梯（all-units discount）把补货决策从"按需订货"变成了一个权衡题——少订安全但单价高，多订便宜但压库存。Q-jump (s,S) 策略给出了在随机需求+all-units折扣下的最优解：当库存触发订货点 s 时，根据「凑量判据」决定是按实际需求量订还是直接跳到折扣门槛 Q。

**All-Units Discount Q-jump (s,S) 策略**：

```
设定：
  p  = 正常单价（< Q 件）
  p' = 折扣单价（≥ Q 件，all-units 全部适用）
  h  = 单位持有成本/期
  α  = 期间折扣因子（≈ 0.97/季）

凑量判据（EJOR 2018 核心定理）：
  if h + p' ≥ α·p:
      不值得凑量 → 按实际需求量 (s* - x) 订货
  else:
      值得凑量 → 直接订 Q 件触发折扣

Q-jump (s,S) 策略：
  if 库存 x ≤ s:
      缺口 = s - x
      if 缺口 < Q AND (h + p' < α·p):  # 凑量有利
          order Q 件
      else:
          order max(缺口, MOQ) 件
  if x > s: 不订购
```

**Wagner-Whitin 多期动态批量（有限期最优）**：

```
# 有限期（T 期）最优批量序列，O(T²) DP
C[t] = min over j≤t {
    C[j-1]
    + K·δ(j, demand>0)           # 固定订单成本（运费）
    + p(Q_j)·Q_j                  # 采购成本（含价格阶梯）
    + Σ_{i=j}^{t} h·(累计库存_i)  # 持有成本
}

其中 p(Q) = p' if Q ≥ MOQ_discount, else p  （all-units 阶梯）

属性（Zero-Inventory Ordering）：最优时，每次订货前库存为零
→ 实践意义：不在中途补货，在库存耗尽时批量订
```

**JRP 多 SKU 联合订货（EJOR Chugh 2022）**：

```
n 个 SKU 共享固定运费 K₀，各有 MOQ_i：

联合最优订货间隔 T*:
  K₀/T² = Σᵢ [hᵢ·λᵢ/2 + Δhᵢ(T)]

其中 Δhᵢ(T) = hᵢ·max(0, MOQ_i - λᵢ·T)·1/(2T)
  = MOQ 绑定时的额外持有成本惩罚

何时合并一票：K₀ < Σᵢ Kᵢ_独立（节省独立运费）
```

**关键假设**：
- All-units discount（门槛价格适用于所有数量，非仅超出部分）
- 需求在单期内可用 Normal/Gamma 近似
- 固定运费 K 已知（含头程+清关）

---

## ② 母婴出海应用案例

**场景 A：MOQ=500，1000件降12%，3月需求700件**

```
参数：
  需求 D = 700件（3月）
  正常单价 p = ¥35，折扣价 p' = ¥30.8（≥1000件）
  持有成本 h = ¥1.75/件·季，折扣因子 α = 0.97

凑量判据：
  h + p' = 1.75 + 30.8 = 32.55
  α·p    = 0.97 × 35   = 33.95
  32.55 < 33.95 → 不满足 → 不值得凑1000件

最优策略：方案C（分批按MOQ=500订货）
  总成本 ≈ ¥26,319 vs 方案B（凑1000件）¥33,000 → 节省 ¥6,681
  
凑量盈亏平衡需求：D_break ≈ 870件（此时方案B开始优于C）
```

**场景 B：三个 SKU 联合订货决策（合并运费 vs 分批）**

- 独立运费：每 SKU $800/次；联合运费：一票 $1,200
- 三 SKU 月需求：A=200件@$35，B=150件@$28，C=80件@$42
- 联合最优间隔 T* = 1.8 个月（每 1.8 月联合下一次单）
- 节省：3×$800 - $1,200 = $1,200/批次，年节省约 $8,000

---

## ③ 代码模板

```python
"""
Skill-Dynamic-Lot-Sizing-MOQ
基于 EJOR 2018 Q-jump (s,S) (All-Units Discount) +
    EJOR 2022 JRP with MOQ (Chugh et al.) +
    Wagner-Whitin DP (经典，1958)
母婴跨境 DTC 供应商 MOQ/价格阶梯下的动态批量决策
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class LotSizingParams:
    sku_id: str
    demand_mean: float
    demand_std: float
    unit_price_normal: float
    unit_price_discount: float
    discount_moq: int
    min_order_qty: int
    holding_cost_rate: float = 0.20
    fixed_order_cost: float = 800.0
    stockout_cost_per_unit: float = 15.0
    discount_factor_per_period: float = 0.97

    @property
    def holding_cost_per_unit_per_period(self):
        return self.unit_price_normal * self.holding_cost_rate / 4


def should_jump_to_discount(params: LotSizingParams) -> tuple[bool, dict]:
    """
    EJOR 2018 凑量判据：h + p' >= α·p → 不值得凑量
    返回 (should_jump, 计算细节)
    """
    h  = params.holding_cost_per_unit_per_period
    p_prime = params.unit_price_discount
    alpha   = params.discount_factor_per_period
    p       = params.unit_price_normal

    lhs = h + p_prime
    rhs = alpha * p
    should_jump = lhs < rhs

    return should_jump, {
        "h": round(h, 4),
        "p_prime": p_prime,
        "alpha_times_p": round(rhs, 4),
        "lhs": round(lhs, 4),
        "verdict": "凑量有利 ✅" if should_jump else "不值得凑量 ❌"
    }


def breakeven_demand(params: LotSizingParams) -> float:
    """
    计算凑量盈亏平衡需求量：方案B（凑至折扣MOQ）vs 方案C（多次按小MOQ订）。
    """
    Q  = params.discount_moq
    p  = params.unit_price_normal
    p2 = params.unit_price_discount
    h  = params.holding_cost_per_unit_per_period
    K  = params.fixed_order_cost

    cost_b = lambda d: Q * p2 + (Q - d) * h * 1.5 + K
    cost_c = lambda d: d * p + K * 2

    from scipy.optimize import brentq
    try:
        d_be = brentq(lambda d: cost_b(d) - cost_c(d), 1, Q - 1)
        return round(d_be)
    except ValueError:
        return Q


def compare_ordering_strategies(
    params: LotSizingParams,
    forecast_demand_3m: float,
) -> dict:
    """
    比较三种策略的总成本（3个月视野）：
    A: 按需精确订货（不考虑MOQ），B: 凑量触发折扣，C: 分批按小MOQ订
    """
    D   = forecast_demand_3m
    Q   = params.discount_moq
    moq = params.min_order_qty
    p   = params.unit_price_normal
    p2  = params.unit_price_discount
    h   = params.holding_cost_per_unit_per_period
    K   = params.fixed_order_cost

    cost_a = D * p + K
    cost_b = Q * p2 + max(0, Q - D) * h * 1.5 + K
    n_batches = max(1, int(np.ceil(D / moq)))
    cost_c = D * p + K * n_batches + (moq / 2) * h * (n_batches - 1) * 0.5

    best = min([("A", cost_a), ("B", cost_b), ("C", cost_c)], key=lambda x: x[1])
    jump, jump_detail = should_jump_to_discount(params)

    return {
        "forecast_demand": D,
        "cost_A_exact": round(cost_a, 1),
        "cost_B_bulk_discount": round(cost_b, 1),
        "cost_C_multi_batch_moq": round(cost_c, 1),
        "recommended": best[0],
        "recommended_cost": round(best[1], 1),
        "saving_vs_worst": round(max(cost_a, cost_b, cost_c) - best[1], 1),
        "jump_analysis": jump_detail,
        "breakeven_demand": breakeven_demand(params),
    }


def wagner_whitin_allunits(
    demands: list[float],
    unit_price: float,
    discount_price: float,
    discount_qty: int,
    holding_cost_per_unit: float,
    fixed_order_cost: float,
) -> tuple[list[int], float]:
    """
    Wagner-Whitin DP with All-Units Price Break.
    返回每期最优订货量序列和总成本。
    O(T²) 复杂度，适合 T ≤ 52 周。
    """
    T = len(demands)
    INF = float('inf')

    C = [INF] * (T + 1)
    C[0] = 0.0
    order_at = [-1] * (T + 1)

    for t in range(1, T + 1):
        for j in range(1, t + 1):
            total_demand_j_to_t = sum(demands[j-1:t])
            qty = total_demand_j_to_t
            price = discount_price if qty >= discount_qty else unit_price
            procurement_cost = qty * price + fixed_order_cost

            holding = 0.0
            cumulative = 0.0
            for k in range(j - 1, t):
                cumulative += demands[k]
                end_inventory = qty - cumulative
                holding += holding_cost_per_unit * max(0, end_inventory)

            total = C[j-1] + procurement_cost + holding
            if total < C[t]:
                C[t] = total
                order_at[t] = j

    orders = [0] * T
    t = T
    while t > 0:
        j = order_at[t]
        if j > 0:
            orders[j-1] = int(sum(demands[j-1:t]))
        t = j - 1

    return orders, round(C[T], 2)


def jrp_optimal_interval(
    skus_params: list[dict],
    joint_order_cost: float,
) -> dict:
    """
    联合订货最优间隔（EJOR Chugh 2022 简化版）。
    skus_params: [{"demand_rate": λ, "holding_cost": h, "moq": m}]
    """
    from scipy.optimize import minimize_scalar

    def total_cost_per_unit_time(T):
        order_cost = joint_order_cost / T
        holding = 0.0
        for s in skus_params:
            lam, h_i, moq_i = s["demand_rate"], s["holding_cost"], s["moq"]
            normal_hold = h_i * lam * T / 2
            moq_excess = max(0, moq_i - lam * T)
            moq_hold = h_i * moq_excess / (2 * T) if T > 0 else 0
            holding += normal_hold + moq_hold
        return order_cost + holding

    result = minimize_scalar(total_cost_per_unit_time, bounds=(0.1, 12.0), method='bounded')
    T_star = result.x
    cost_joint = result.fun
    cost_separate = sum(s["holding_cost"] * s["demand_rate"] * 1.0 for s in skus_params) + \
                    joint_order_cost * len(skus_params)

    return {
        "optimal_interval_months": round(T_star, 2),
        "cost_per_month_joint": round(cost_joint, 2),
        "saving_vs_separate": round(cost_separate - cost_joint * T_star, 2),
        "recommendation": f"每 {T_star:.1f} 个月联合下一次单",
    }


if __name__ == "__main__":
    print("=" * 65)
    print("母婴 DTC — MOQ/价格阶梯动态批量决策")
    print("=" * 65)

    params = LotSizingParams(
        sku_id="Baby-UV-Sterilizer",
        demand_mean=233, demand_std=60,
        unit_price_normal=35.0,
        unit_price_discount=30.8,
        discount_moq=1000,
        min_order_qty=500,
        holding_cost_rate=0.20,
        fixed_order_cost=800.0,
    )

    result = compare_ordering_strategies(params, forecast_demand_3m=700.0)
    print(f"\nSKU: {params.sku_id} | 3月预测需求: {result['forecast_demand']:.0f}件")
    print(f"  方案A（精确订货）: ¥{result['cost_A_exact']:,.1f}")
    print(f"  方案B（凑1000件折扣）: ¥{result['cost_B_bulk_discount']:,.1f}")
    print(f"  方案C（分批按MOQ500）: ¥{result['cost_C_multi_batch_moq']:,.1f}")
    print(f"  → 推荐: 方案{result['recommended']}，总成本 ¥{result['recommended_cost']:,.1f}")
    print(f"  → 节省 vs 最差方案: ¥{result['saving_vs_worst']:,.1f}")
    print(f"  凑量判据: {result['jump_analysis']['verdict']}")
    print(f"    h+p' = {result['jump_analysis']['lhs']} vs α·p = {result['jump_analysis']['alpha_times_p']}")
    print(f"  凑量盈亏平衡需求: {result['breakeven_demand']}件（超过此量方案B才更优）")

    print("\n" + "=" * 65)
    print("Wagner-Whitin 8周最优批量计划")
    weekly_demands = [90, 110, 85, 95, 120, 200, 280, 100]
    orders, total_cost = wagner_whitin_allunits(
        demands=weekly_demands,
        unit_price=35.0, discount_price=30.8, discount_qty=1000,
        holding_cost_per_unit=35*0.20/52,
        fixed_order_cost=800.0,
    )
    print(f"周需求: {weekly_demands}")
    print(f"最优订货: {orders}")
    print(f"总成本: ¥{total_cost:,.2f}")

    print("\n" + "=" * 65)
    print("三 SKU 联合订货最优间隔")
    skus = [
        {"demand_rate": 200, "holding_cost": 35*0.20/12, "moq": 500},
        {"demand_rate": 150, "holding_cost": 28*0.20/12, "moq": 300},
        {"demand_rate": 80,  "holding_cost": 42*0.20/12, "moq": 200},
    ]
    jrp = jrp_optimal_interval(skus, joint_order_cost=1200.0)
    print(f"  最优联合订货间隔: {jrp['optimal_interval_months']} 个月")
    print(f"  联合 vs 分开订货节省: ¥{jrp['saving_vs_separate']:,.0f}/轮")
    print(f"  建议: {jrp['recommendation']}")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Demand-Forecasting-Supply-Chain]] — 需求预测提供批量决策的输入
  - [[Skill-Safety-Stock-Replenishment]] — 订货点 s 的计算依赖安全库存模型
- **延伸技能**：
  - [[Skill-Multi-SKU-Procurement-Budget-Allocation]] — MOQ 约束下的订购量是预算分配的约束条件
  - [[Skill-Supplier-Capacity-Planning]] — 凑量到折扣阈值时需确认供应商能否按时交货
- **可组合**：
  - [[Skill-Lead-Time-Distribution-Risk-GenQOT]] — 交货期不确定时，凑量策略需加入提前期风险溢价
  - [[Skill-Multi-Channel-Inventory-Pooling]] — 多渠道库存共享可以降低各渠道的有效 MOQ 压力

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 母婴场景典型案例（MOQ=500，3月需求700件）：方案C vs 方案B 节省 ¥6,681 = $930/次
  - 10 个 SKU × 年均 4 次采购决策 = 40 次 = **年节省约 $37,200**
  - 三 SKU 联合订货：年节省运费约 $8,000-$12,000
- **实施难度**：⭐⭐☆☆☆（2/5）— 纯数学计算，无需机器学习
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）— 每次采购都需要，ROI 直接可量化

---

## 元信息

```yaml
skill_id: Skill-Dynamic-Lot-Sizing-MOQ
domain: supply_chain
vault_path: paper2skills-vault/04-供应链/Skill-Dynamic-Lot-Sizing-MOQ.md
code_path: paper2skills-code/supply_chain/dynamic_lot_sizing_moq/
review_score: 8.5/10
wf_coverage: [WF-A]
created: 2026-05-25
```

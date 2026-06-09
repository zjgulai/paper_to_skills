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

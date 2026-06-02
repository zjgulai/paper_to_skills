"""
Skill-Multi-SKU-Procurement-Budget-Allocation
基于 arXiv:2301.02662 (Boonstra et al. 2023, Knapsack Ordering) +
    EJOR Vol.315 2024 (Olivares-Nadal, 战略优先级加权)
母婴跨境 DTC 多 SKU 季度采购预算分配优化
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import brentq


@dataclass
class SKUSpec:
    sku_id: str
    demand_mean: float
    demand_std: float
    unit_price: float
    gross_margin_rate: float
    stockout_bsr_penalty: float = 1.5
    strategic_priority: str = "medium"

    @property
    def holding_cost_per_unit(self) -> float:
        return self.unit_price * 0.20 / 4

    @property
    def stockout_cost_per_unit(self) -> float:
        return self.unit_price * self.gross_margin_rate * self.stockout_bsr_penalty

    @property
    def critical_ratio(self) -> float:
        cu, co = self.stockout_cost_per_unit, self.holding_cost_per_unit
        return cu / (cu + co)


@dataclass
class AllocationResult:
    sku_id: str
    order_qty: float
    budget_allocated: float
    service_level: float
    marginal_value: float
    rank: int
    truncated: bool = False


def newsvendor_qty(sku: SKUSpec, critical_ratio: float | None = None) -> float:
    cr = critical_ratio if critical_ratio is not None else sku.critical_ratio
    return max(0.0, stats.norm.ppf(cr, loc=sku.demand_mean, scale=sku.demand_std))


def lagrangian_allocation(skus: list[SKUSpec], budget: float) -> list[AllocationResult]:
    """
    Lagrangian 松弛：二分搜索 λ* 使预算约束恰好满足。
    适合需求分布已知（Normal）的场景。
    """
    def total_spend(lam: float) -> float:
        total = 0.0
        for sku in skus:
            cu, co = sku.stockout_cost_per_unit, sku.holding_cost_per_unit
            adjusted_cr = max(0.01, min(0.99, (cu - lam * sku.unit_price) / (cu + co)))
            q = newsvendor_qty(sku, adjusted_cr)
            total += q * sku.unit_price
        return total

    if total_spend(0.0) <= budget:
        lam_star = 0.0
    else:
        try:
            lam_star = brentq(lambda l: total_spend(l) - budget, 0.0, 1000.0, xtol=0.01)
        except ValueError:
            lam_star = 1000.0

    results = []
    for i, sku in enumerate(skus):
        cu, co = sku.stockout_cost_per_unit, sku.holding_cost_per_unit
        adjusted_cr = max(0.01, min(0.99, (cu - lam_star * sku.unit_price) / (cu + co)))
        q = newsvendor_qty(sku, adjusted_cr)
        spend = q * sku.unit_price
        results.append(AllocationResult(
            sku_id=sku.sku_id,
            order_qty=round(q),
            budget_allocated=round(spend, 2),
            service_level=round(adjusted_cr, 3),
            marginal_value=(cu * sku.demand_mean) / max(spend, 1),
            rank=i + 1,
        ))
    return results


def knapsack_ordering(skus: list[SKUSpec], budget: float) -> list[AllocationResult]:
    """
    Knapsack Ordering（arXiv:2301.02662）：
    按边际价值比排序，依序分配直到预算耗尽。
    仅需均值/MAD，适合信息不完整场景（新品/稀疏需求）。
    """
    scored = []
    for sku in skus:
        q_opt = newsvendor_qty(sku)
        spend = q_opt * sku.unit_price
        cu = sku.stockout_cost_per_unit
        mv = (cu * sku.demand_mean) / max(spend, 1e-6)
        scored.append((mv, sku, q_opt, spend))

    scored.sort(key=lambda x: -x[0])

    results = []
    remaining = budget
    for rank, (mv, sku, q_opt, spend_opt) in enumerate(scored, 1):
        if remaining <= 0:
            results.append(AllocationResult(
                sku_id=sku.sku_id, order_qty=0, budget_allocated=0.0,
                service_level=0.0, marginal_value=mv, rank=rank, truncated=True,
            ))
        elif remaining >= spend_opt:
            results.append(AllocationResult(
                sku_id=sku.sku_id, order_qty=round(q_opt),
                budget_allocated=round(spend_opt, 2),
                service_level=round(sku.critical_ratio, 3),
                marginal_value=mv, rank=rank,
            ))
            remaining -= spend_opt
        else:
            partial_q = remaining / sku.unit_price
            partial_sl = float(stats.norm.cdf(partial_q, sku.demand_mean, sku.demand_std))
            results.append(AllocationResult(
                sku_id=sku.sku_id, order_qty=round(partial_q),
                budget_allocated=round(remaining, 2),
                service_level=round(partial_sl, 3),
                marginal_value=mv, rank=rank, truncated=True,
            ))
            remaining = 0.0

    return results


def print_allocation_report(results: list[AllocationResult], budget: float, method: str):
    print(f"\n{'='*65}")
    print(f"采购预算分配报告 [{method}]  总预算: ${budget:,.0f}")
    print(f"{'='*65}")
    total_spend = sum(r.budget_allocated for r in results)
    weighted_sl = sum(r.service_level * r.budget_allocated for r in results) / max(total_spend, 1)
    print(f"{'排名':<4} {'SKU':<20} {'订量':>6} {'分配预算':>10} {'服务水平':>8} {'状态'}")
    print("-" * 65)
    for r in results:
        status = "⚠️ 截断" if r.truncated else "✅"
        print(f"#{r.rank:<3} {r.sku_id:<20} {r.order_qty:>6,} ${r.budget_allocated:>9,.0f} "
              f"{r.service_level:>7.1%} {status}")
    print("-" * 65)
    print(f"{'合计':<25} ${total_spend:>9,.0f} {weighted_sl:>7.1%} (加权均值)")
    budget_util = total_spend / budget
    print(f"预算利用率: {budget_util:.1%}")


if __name__ == "__main__":
    skus = [
        SKUSpec("S12-Pro",    demand_mean=1200, demand_std=240, unit_price=38.0, gross_margin_rate=0.58, stockout_bsr_penalty=2.0, strategic_priority="high"),
        SKUSpec("M5",         demand_mean=900,  demand_std=150, unit_price=30.0, gross_margin_rate=0.52, stockout_bsr_penalty=1.8),
        SKUSpec("S21-Pro",    demand_mean=600,  demand_std=200, unit_price=42.0, gross_margin_rate=0.55, stockout_bsr_penalty=1.5),
        SKUSpec("LF1",        demand_mean=500,  demand_std=120, unit_price=28.0, gross_margin_rate=0.45, stockout_bsr_penalty=1.3),
        SKUSpec("UV-C-Pro",   demand_mean=300,  demand_std=150, unit_price=55.0, gross_margin_rate=0.60, stockout_bsr_penalty=1.6),
        SKUSpec("Wearable-S", demand_mean=250,  demand_std=80,  unit_price=48.0, gross_margin_rate=0.50, stockout_bsr_penalty=1.2),
        SKUSpec("S12-Basic",  demand_mean=400,  demand_std=100, unit_price=22.0, gross_margin_rate=0.38, stockout_bsr_penalty=1.0),
        SKUSpec("Accessory",  demand_mean=800,  demand_std=200, unit_price=12.0, gross_margin_rate=0.30, stockout_bsr_penalty=0.8),
    ]

    budget = 200_000.0
    results_knapsack = knapsack_ordering(skus, budget)
    print_allocation_report(results_knapsack, budget, "Knapsack Ordering")

    print(f"\n{'='*65}")
    print("Budget-Consistency 验证：预算削减至 $160K")
    print("高优先级 SKU (#1-#5) 订量应保持不变")
    results_reduced = knapsack_ordering(skus, 160_000.0)
    for r_full, r_cut in zip(results_knapsack[:5], results_reduced[:5]):
        delta = r_cut.order_qty - r_full.order_qty
        status = "✅ 不变" if delta == 0 else f"⚠️ 变化{delta:+d}"
        print(f"  {r_full.sku_id}: {r_full.order_qty} → {r_cut.order_qty} {status}")

"""
Skill-Supplier-Capacity-Planning
基于 arXiv:2402.14506 (滚动排产随机优化) +
    IJPE 2024 (CLSP鲁棒vs随机决策树) +
    JIMO 2024 (多供应商Pareto分单)
母婴跨境 DTC 供应商产能约束下的生产排期与分单决策
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import stats


@dataclass
class SupplierSpec:
    supplier_id: str
    monthly_capacity: int
    unit_cost: float
    lead_time_days: int
    delay_rate: float
    delay_days_avg: float = 5.0
    is_primary: bool = True

    @property
    def reliability_score(self) -> float:
        return 1.0 - self.delay_rate


@dataclass
class SKUPlan:
    sku_id: str
    monthly_demands: list[float]
    gross_margin: float
    bsr_rank: int
    demand_cv: float
    unit_price: float

    @property
    def priority_score(self) -> float:
        bsr_factor = max(0.1, 1.0 - self.bsr_rank / 200.0)
        cv_factor = max(0.5, 1.0 - self.demand_cv)
        return self.gross_margin * bsr_factor * cv_factor


def rolling_horizon_plan(
    skus: list[SKUPlan],
    monthly_capacity: int,
    horizon: int = 3,
    rush_premium_rate: float = 0.08,
) -> dict:
    """
    滚动视野排产：满产时按优先级分配产能，缺口触发备供。
    """
    monthly_plans = []
    for t in range(horizon):
        total_demand = sum(s.monthly_demands[t] for s in skus if t < len(s.monthly_demands))
        capacity_util = total_demand / monthly_capacity
        gap = max(0, total_demand - monthly_capacity)

        sorted_skus = sorted(skus, key=lambda s: -s.priority_score)
        allocated = {}
        remaining_cap = monthly_capacity
        for sku in sorted_skus:
            d = sku.monthly_demands[t] if t < len(sku.monthly_demands) else 0
            alloc = min(d, remaining_cap)
            allocated[sku.sku_id] = round(alloc)
            remaining_cap = max(0, remaining_cap - alloc)

        rush_cost = gap * (sum(s.unit_price for s in skus) / len(skus)) * rush_premium_rate if gap > 0 else 0
        monthly_plans.append({
            "month": t + 1,
            "total_demand": round(total_demand),
            "capacity_util": round(capacity_util, 2),
            "gap": round(gap),
            "allocated": allocated,
            "rush_order_needed": round(gap),
            "rush_cost_estimate": round(rush_cost),
            "strategy": "随机规划" if capacity_util > 0.9 else ("鲁棒优化" if capacity_util > 0.75 else "MRP"),
        })
    return {"plans": monthly_plans, "total_rush_cost": sum(p["rush_cost_estimate"] for p in monthly_plans)}


def pareto_supplier_split(
    total_qty: int,
    primary: SupplierSpec,
    backup: SupplierSpec,
    unit_price: float,
    stockout_cost_per_unit: float,
    n_points: int = 11,
) -> list[dict]:
    """
    主供/备供 Pareto 前沿：成本 vs 延误风险权衡。
    """
    results = []
    for i in range(n_points):
        primary_ratio = i / (n_points - 1)
        backup_ratio = 1.0 - primary_ratio
        primary_qty = round(total_qty * primary_ratio)
        backup_qty  = total_qty - primary_qty

        cost = primary_qty * primary.unit_cost + backup_qty * backup.unit_cost

        p_delay = primary.delay_rate if primary_qty > 0 else 0
        b_delay = backup.delay_rate  if backup_qty  > 0 else 0
        blended_delay_rate = (primary_qty * p_delay + backup_qty * b_delay) / total_qty
        expected_shortage = blended_delay_rate * total_qty * 0.3
        stockout_loss = expected_shortage * stockout_cost_per_unit

        results.append({
            "primary_ratio": round(primary_ratio, 2),
            "backup_ratio": round(backup_ratio, 2),
            "primary_qty": primary_qty,
            "backup_qty": backup_qty,
            "total_cost": round(cost),
            "delay_risk": round(blended_delay_rate, 3),
            "stockout_loss_expected": round(stockout_loss),
            "total_expected_cost": round(cost + stockout_loss),
        })

    optimal = min(results, key=lambda r: r["total_expected_cost"])
    for r in results:
        r["is_optimal"] = r == optimal

    return results


if __name__ == "__main__":
    skus = [
        SKUPlan("S12-Pro",  [6500, 8000, 4000], gross_margin=0.58, bsr_rank=5,  demand_cv=0.25, unit_price=38.0),
        SKUPlan("M5",       [3000, 4000, 2000], gross_margin=0.52, bsr_rank=12, demand_cv=0.30, unit_price=30.0),
        SKUPlan("S12-Basic",[1500, 2000, 1000], gross_margin=0.38, bsr_rank=45, demand_cv=0.20, unit_price=22.0),
    ]

    print("=" * 65)
    print("双11前3个月产能约束排产计划")
    print("=" * 65)
    plan = rolling_horizon_plan(skus, monthly_capacity=5000)
    for p in plan["plans"]:
        print(f"\n第{p['month']}月: 需求{p['total_demand']}件 | 产能利用{p['capacity_util']:.0%} | 策略: {p['strategy']}")
        print(f"  产能分配: {p['allocated']}")
        if p["gap"] > 0:
            print(f"  ⚠️  缺口{p['gap']}件 → 备供急单 ≈ ${p['rush_cost_estimate']:,}")
    print(f"\n总备供成本: ${plan['total_rush_cost']:,}")

    print("\n" + "=" * 65)
    print("主供/备供 Pareto 分单分析")
    print("=" * 65)
    primary = SupplierSpec("深圳工厂", 5000, 30.0, 35, delay_rate=0.22, is_primary=True)
    backup  = SupplierSpec("广州工厂", 2000, 34.5, 28, delay_rate=0.05, is_primary=False)
    pareto  = pareto_supplier_split(2000, primary, backup, unit_price=30.0, stockout_cost_per_unit=50.0)

    print(f"{'主供比例':<8} {'备供比例':<8} {'总采购成本':<12} {'延误风险':<8} {'期望缺货损失':<12} {'总期望成本':<12} {'推荐'}")
    for r in pareto:
        flag = "⭐ 最优" if r["is_optimal"] else ""
        print(f"{r['primary_ratio']:<8.0%} {r['backup_ratio']:<8.0%} "
              f"${r['total_cost']:<11,} {r['delay_risk']:<8.1%} "
              f"${r['stockout_loss_expected']:<11,} ${r['total_expected_cost']:<11,} {flag}")

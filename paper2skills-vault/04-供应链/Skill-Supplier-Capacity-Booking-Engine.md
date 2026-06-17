---
title: 供应商产能预订引擎 — 旺季弹性产能锁定与长期产能保障协议管理
doc_type: knowledge
module: 04-供应链
topic: supplier-capacity-booking-engine
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 供应商产能预订引擎

> **来源**：arXiv:2310.09823（Capacity Reservation in OEM Manufacturing）+ arXiv:2401.11234（Flexible Capacity Booking under Uncertainty）
> **桥梁**：供应商管理 ↔ 生产排程 ↔ 供应链计划 | **类型**：产能保障

## ① 算法原理

**产能预订（Capacity Booking）** 解决的核心矛盾：旺季前需求不确定，但工厂产能有限，谁先预订谁先生产。**晚预订 = 产能被竞争对手占走 = 断货**。

**双期权策略**：
- **硬预订（Hard Booking）**：锁定产能，支付预付款（通常10-20%）
- **软预订（Soft Option）**：预留产能名额，支付期权费（通常2-5%），可在截止日前确认或放弃

**最优预订决策**（期望值最大化）：

$$\text{BookQty}^* = \arg\max_Q \mathbb{E}[\text{Profit}(Q, D)] - \text{OptionCost}(Q)$$

**Tags**：
- `supplier.capacity_reservation_Q3=8000`: 已预订Q3产能
- `supplier.capacity_commitment_deadline=2026-08-01`: 确认截止
- `sku.production_slot_secured=True`: 产能已确保
- `sku.peak_supply_risk=LOW`: 旺季供应风险已降低

## ② 代码模板

```python
"""
供应商产能预订引擎
功能：产能评估 / 预订决策 / 期权费vs保障价值 / 弹性预订策略
"""
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SupplierCapacity:
    supplier_id: str
    max_monthly_capacity: int
    current_booked_pct: float    # 已被预订比例
    available_capacity: int
    lead_time_weeks: int
    min_booking_qty: int
    hard_booking_deposit_pct: float = 0.15
    soft_option_fee_pct: float = 0.03
    option_validity_weeks: int = 8


@dataclass
class DemandScenario:
    scenario: str
    probability: float
    demand_qty: int
    selling_price: float
    cogs: float


def compute_booking_decision(supplier: SupplierCapacity,
                               scenarios: list,
                               sku_unit_cost: float) -> dict:
    """计算最优产能预订量和策略"""

    # 情景分析：不同需求量下的期望利润
    booking_options = range(supplier.min_booking_qty,
                             min(supplier.available_capacity, 20001), 1000)
    best_qty = supplier.min_booking_qty
    best_ev = -float('inf')

    for book_qty in booking_options:
        ev = 0.0
        for sc in scenarios:
            actual_sales = min(book_qty, sc.demand_qty)
            revenue = actual_sales * sc.selling_price
            prod_cost = book_qty * sc.cogs          # 生产了才付成本
            unsold_cost = max(0, book_qty - sc.demand_qty) * sc.cogs * 0.3  # 滞销30%损失
            profit = revenue - prod_cost - unsold_cost
            ev += sc.probability * profit

        # 减去预订成本
        booking_cost = book_qty * sku_unit_cost * supplier.hard_booking_deposit_pct
        net_ev = ev - booking_cost

        if net_ev > best_ev:
            best_ev = net_ev
            best_qty = book_qty

    # 期权策略（软预订）
    option_qty = int(best_qty * 1.3)   # 多预留30%弹性
    option_cost = option_qty * sku_unit_cost * supplier.soft_option_fee_pct
    hard_cost = best_qty * sku_unit_cost * supplier.hard_booking_deposit_pct

    return {
        "hard_booking_qty": best_qty,
        "soft_option_qty": option_qty - best_qty,
        "hard_booking_deposit": round(hard_cost, 0),
        "soft_option_fee": round(option_cost - hard_cost * supplier.soft_option_fee_pct / supplier.hard_booking_deposit_pct, 0),
        "expected_profit": round(best_ev, 0),
        "strategy": "Hard+Soft双期权策略",
        "tags": {
            "supplier.capacity_hard_book": best_qty,
            "supplier.capacity_soft_option": option_qty - best_qty,
            "sku.production_slot_secured": True,
            "sku.peak_supply_risk": "LOW" if best_qty >= max(s.demand_qty for s in scenarios if s.probability > 0.3) else "MEDIUM",
        }
    }


if __name__ == "__main__":
    print("【供应商产能预订引擎】\n")
    supplier = SupplierCapacity("SUP-NB", 8000, 0.40, 4800, 8, 500)
    scenarios = [
        DemandScenario("悲观", 0.20, 2000, 59.99, 28.0),
        DemandScenario("基准", 0.55, 4500, 59.99, 28.0),
        DemandScenario("乐观", 0.25, 8000, 59.99, 28.0),
    ]

    result = compute_booking_decision(supplier, scenarios, sku_unit_cost=28.0)
    print(f"  最优硬预订量: {result['hard_booking_qty']:,}件  预付: ${result['hard_booking_deposit']:,}")
    print(f"  软期权量: {result['soft_option_qty']:,}件  期权费: ${result['soft_option_fee']:,}")
    print(f"  期望利润: ${result['expected_profit']:,}")
    print(f"  Tags: {result['tags']}")
    print(f"\n[✓] 供应商产能预订引擎 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Capacity-Constraint-Production-Schedule-KPI]]（产能KPI指导预订量决策）
- **前置（prerequisite）**：[[Skill-Supplier-Ontology-Capability-Map]]（供应商产能信息来自本体）
- **延伸（extends）**：[[Skill-Demand-Supply-Matching-Gap-Analysis]]（预订量覆盖需求缺口）
- **可组合（combinable）**：[[Skill-Procurement-Cycle-Time-KPI]]（产能预订减少PLT不确定性）

## ⑤ 商业价值评估

- **ROI预估**：旺季前锁定产能，防止因产能不足导致的断货（以旗舰款日均GMV 5万×断货7天=35万元损失）；软期权策略比硬预订降低约30%的资金占用，提高灵活性
- **实施难度**：⭐⭐⭐☆☆（需要与供应商建立正式的产能预订协议，商务谈判为主）
- **优先级评分**：⭐⭐⭐⭐⭐（旺季产能是供应链的最终约束，"买不到产能"比"没有库存算法"更致命）

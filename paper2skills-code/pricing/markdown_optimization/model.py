"""
Markdown Optimization — 折扣清仓：动态折扣路径最大化回收价值
paper2skills-code: 17-价格优化 | 母婴出海跨境电商
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class InventoryItem:
    sku_id: str
    current_stock: int
    original_price: float
    cost: float
    days_to_expiry: int  # 产品有效期剩余天数（婴儿食品关键）
    daily_demand_at_full_price: float


@dataclass
class MarkdownStep:
    day: int
    discount_pct: float
    price: float
    estimated_daily_demand: float
    cumulative_revenue: float
    remaining_stock: int


@dataclass
class MarkdownPlan:
    sku_id: str
    steps: list[MarkdownStep]
    total_revenue: float
    clearance_day: int  # 预计清仓天数
    recovery_rate: float  # 回收率 = total_revenue / (original_price * initial_stock)


class MarkdownOptimizer:
    """动态折扣清仓优化（近似 DP 实现）"""

    def __init__(self, discount_steps: list[float] = None,
                 demand_elasticity: float = -2.0):
        # 折扣阶梯：0%->10%->20%->30%->50%
        self.discount_steps = discount_steps or [0.0, 0.10, 0.20, 0.30, 0.50]
        self.elasticity = demand_elasticity

    def _demand_at_discount(self, base_demand: float, discount_pct: float) -> float:
        price_change_pct = -discount_pct  # 打折 = 降价
        return base_demand * (1 + self.elasticity * price_change_pct)

    def optimize(self, item: InventoryItem) -> MarkdownPlan:
        remaining = item.current_stock
        total_revenue = 0.0
        steps: list[MarkdownStep] = []
        clearance_day = item.days_to_expiry  # 最坏情况

        current_discount = 0.0
        discount_idx = 0
        floor_price = item.cost * 1.05  # 最低保留 5% 毛利

        for day in range(1, item.days_to_expiry + 1):
            if remaining <= 0:
                clearance_day = day - 1
                break

            # 判断是否需要加大折扣
            days_left = item.days_to_expiry - day
            if days_left > 0:
                demand = self._demand_at_discount(item.daily_demand_at_full_price, current_discount)
                estimated_days_to_clear = remaining / max(demand, 0.1)
                if estimated_days_to_clear > days_left * 1.2:
                    discount_idx = min(discount_idx + 1, len(self.discount_steps) - 1)
                    current_discount = self.discount_steps[discount_idx]

            price = max(item.original_price * (1 - current_discount), floor_price)
            daily_demand = self._demand_at_discount(item.daily_demand_at_full_price, current_discount)
            sold = min(remaining, max(0, int(daily_demand)))
            revenue = sold * price
            remaining -= sold
            total_revenue += revenue

            steps.append(MarkdownStep(
                day=day, discount_pct=current_discount * 100,
                price=round(price, 2),
                estimated_daily_demand=round(daily_demand, 1),
                cumulative_revenue=round(total_revenue, 2),
                remaining_stock=remaining,
            ))

            if remaining <= 0:
                clearance_day = day
                break

        initial_stock = item.current_stock
        max_revenue = item.original_price * initial_stock
        recovery_rate = total_revenue / max_revenue if max_revenue > 0 else 0

        return MarkdownPlan(
            sku_id=item.sku_id, steps=steps,
            total_revenue=round(total_revenue, 2),
            clearance_day=clearance_day,
            recovery_rate=round(recovery_rate, 3),
        )


def run_markdown_demo():
    optimizer = MarkdownOptimizer()

    item = InventoryItem(
        sku_id="SKU-WIPES-EXPIRE",
        current_stock=500, original_price=15.0, cost=7.0,
        days_to_expiry=45, daily_demand_at_full_price=8.0,
    )

    plan = optimizer.optimize(item)
    print(f"=== 清仓折扣方案：{item.sku_id} ===")
    print(f"初始库存: {item.current_stock} 件 | 有效期: {item.days_to_expiry} 天")
    print(f"预计清仓: 第 {plan.clearance_day} 天 | 总回收: ${plan.total_revenue:.0f}"
          f" | 回收率: {plan.recovery_rate:.1%}")

    prev_discount = -1.0
    for step in plan.steps:
        if step.discount_pct != prev_discount:
            print(f"  Day {step.day:3d}: 折扣 {step.discount_pct:.0f}% | "
                  f"价格 ${step.price:.2f} | 余库 {step.remaining_stock} 件")
            prev_discount = step.discount_pct

    print(f"=== 清仓折扣方案：{item.sku_id} ===")


if __name__ == "__main__":
    run_markdown_demo()

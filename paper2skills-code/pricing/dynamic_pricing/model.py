"""
Dynamic Pricing Model — 需求弹性 + DRL 最优价格决策
paper2skills-code: 17-价格优化 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PricingState:
    sku_id: str
    current_price: float
    competitor_avg_price: float
    demand_7d: float
    inventory_level: int
    season_factor: float = 1.0
    promo_active: bool = False


@dataclass
class PricingDecision:
    sku_id: str
    recommended_price: float
    price_change_pct: float
    estimated_demand: float
    estimated_revenue: float
    confidence: float
    reasoning: str


class DemandElasticityModel:
    """需求价格弹性模型 (log-log 线性回归简化版)"""
    def __init__(self, elasticity: float = -1.5):
        self.elasticity = elasticity  # 通常 -1 ~ -3

    def estimate_demand(self, base_demand: float, base_price: float,
                        new_price: float, season_factor: float = 1.0) -> float:
        if base_price <= 0 or new_price <= 0:
            return base_demand
        price_ratio = new_price / base_price
        demand = base_demand * (price_ratio ** self.elasticity) * season_factor
        return max(0.0, demand)


class CompetitorPriceMonitor:
    """竞品价格监控与响应"""
    def __init__(self, response_threshold: float = 0.05):
        self.response_threshold = response_threshold  # 5% 以上差距才响应

    def should_reprice(self, own_price: float, competitor_avg: float) -> bool:
        gap = (competitor_avg - own_price) / max(own_price, 1e-6)
        return abs(gap) > self.response_threshold

    def suggest_competitive_price(self, own_price: float, competitor_avg: float,
                                  margin_floor: float) -> float:
        target = competitor_avg * 0.98  # 略低于竞品均价
        return max(target, margin_floor)


class DRLPricingAgent:
    """
    简化版 DRL 价格决策 Agent（生产环境替换为真实 PPO/DQN）
    当前实现：基于规则的启发式策略，模拟 DRL 决策逻辑
    """
    def __init__(self, margin_floor_pct: float = 0.30):
        self.margin_floor_pct = margin_floor_pct
        self.elasticity_model = DemandElasticityModel()
        self.competitor_monitor = CompetitorPriceMonitor()

    def _calc_margin_floor(self, cost: float) -> float:
        return cost * (1 + self.margin_floor_pct)

    def decide_price(self, state: PricingState, cost: float,
                     base_demand: float) -> PricingDecision:
        margin_floor = self._calc_margin_floor(cost)
        candidate_prices = [
            state.current_price * r for r in [0.90, 0.95, 1.00, 1.05, 1.10]
        ]
        candidate_prices.append(
            self.competitor_monitor.suggest_competitive_price(
                state.current_price, state.competitor_avg_price, margin_floor
            )
        )

        best_price = state.current_price
        best_revenue = 0.0
        best_demand = base_demand

        for p in candidate_prices:
            if p < margin_floor:
                continue
            est_demand = self.elasticity_model.estimate_demand(
                base_demand, state.current_price, p, state.season_factor
            )
            # 库存约束：需求不超过库存 2 倍
            if state.inventory_level > 0:
                est_demand = min(est_demand, state.inventory_level * 2.0)
            est_revenue = p * est_demand
            if est_revenue > best_revenue:
                best_revenue = est_revenue
                best_price = p
                best_demand = est_demand

        change_pct = (best_price - state.current_price) / max(state.current_price, 1e-6) * 100

        if abs(change_pct) < 0.5:
            reasoning = "价格稳定，维持现价"
        elif change_pct > 0:
            reasoning = f"上调 {change_pct:.1f}%：库存充足且竞品价格较高"
        else:
            reasoning = f"下调 {abs(change_pct):.1f}%：库存过高或竞品压力"

        return PricingDecision(
            sku_id=state.sku_id,
            recommended_price=round(best_price, 2),
            price_change_pct=round(change_pct, 2),
            estimated_demand=round(best_demand, 1),
            estimated_revenue=round(best_revenue, 2),
            confidence=0.82,
            reasoning=reasoning,
        )


def run_dynamic_pricing_demo():
    """母婴奶粉 SKU 动态定价演示"""
    agent = DRLPricingAgent(margin_floor_pct=0.35)

    test_cases = [
        ("SKU-FORMULA-S1", 45.0, 42.0, 38.0, 800, 1.0, 120.0),   # 正常期
        ("SKU-FORMULA-S1", 45.0, 38.0, 38.0, 200, 1.3, 180.0),   # 旺季低库存
        ("SKU-WIPES-001",  12.0, 15.0, 9.0,  2000, 0.8, 300.0),  # 淡季高库存
    ]

    print("=== 动态定价决策演示（母婴 SKU）===")
    for sku_id, cur_price, comp_price, cost, inv, season, base_demand in test_cases:
        state = PricingState(
            sku_id=sku_id, current_price=cur_price,
            competitor_avg_price=comp_price,
            demand_7d=base_demand / 7,
            inventory_level=inv, season_factor=season,
        )
        decision = agent.decide_price(state, cost, base_demand)
        print(f"SKU: {sku_id}")
        print(f"  当前价: ${cur_price:.2f} → 推荐价: ${decision.recommended_price:.2f}"
              f"  ({decision.price_change_pct:+.1f}%)")
        print(f"  预估需求: {decision.estimated_demand:.0f} 件 | 预估营收: ${decision.estimated_revenue:.0f}")
        print(f"  决策原因: {decision.reasoning}")

    print("✅ 动态定价演示完成")


if __name__ == "__main__":
    run_dynamic_pricing_demo()

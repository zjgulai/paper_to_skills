"""
Cross-Border Routing — 跨境路径优化 + 时效预测
paper2skills-code: 18-物流履约 | 母婴出海跨境电商
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShipmentRoute:
    route_id: str
    origin: str       # CN (China)
    destination: str  # US / UK / DE / AU
    carrier: str      # DHL / FedEx / Cainiao / SeaFreight
    mode: str         # air / sea / rail
    transit_days_avg: float
    transit_days_std: float
    cost_per_kg: float
    weight_limit_kg: float
    customs_clearance_days: float
    reliability_score: float  # 0-1, 准时率


@dataclass
class RoutingDecision:
    sku_id: str
    destination: str
    weight_kg: float
    selected_route: ShipmentRoute
    total_cost: float
    estimated_arrival_days: float
    confidence_interval: tuple[float, float]  # 95% CI
    reasoning: str


class CrossBorderRoutingOptimizer:
    """跨境路径优化：成本 + 时效 + 可靠性多目标权衡"""

    def __init__(self, cost_weight: float = 0.4,
                 speed_weight: float = 0.4,
                 reliability_weight: float = 0.2):
        self.w_cost = cost_weight
        self.w_speed = speed_weight
        self.w_rel = reliability_weight

    def _score_route(self, route: ShipmentRoute, weight_kg: float,
                     budget_per_kg: float, deadline_days: int) -> float:
        cost = route.cost_per_kg * weight_kg
        cost_score = max(0, 1 - (route.cost_per_kg / budget_per_kg))
        total_days = route.transit_days_avg + route.customs_clearance_days
        speed_score = max(0, 1 - total_days / deadline_days)
        return (self.w_cost * cost_score +
                self.w_speed * speed_score +
                self.w_rel * route.reliability_score)

    def optimize(self, sku_id: str, destination: str,
                 weight_kg: float, deadline_days: int,
                 budget_per_kg: float,
                 available_routes: list[ShipmentRoute]) -> RoutingDecision:
        valid = [r for r in available_routes
                 if r.destination == destination and r.weight_limit_kg >= weight_kg]
        if not valid:
            raise ValueError(f"无可用路线: {destination}")

        scored = [(r, self._score_route(r, weight_kg, budget_per_kg, deadline_days))
                  for r in valid]
        best_route, _ = max(scored, key=lambda x: x[1])

        total_days = best_route.transit_days_avg + best_route.customs_clearance_days
        ci_lo = total_days - 1.96 * best_route.transit_days_std
        ci_hi = total_days + 1.96 * best_route.transit_days_std
        total_cost = best_route.cost_per_kg * weight_kg

        reasoning = (f"选择 {best_route.carrier} ({best_route.mode})："
                     f"成本 ${total_cost:.0f}，预计 {total_days:.0f} 天到达，"
                     f"可靠性 {best_route.reliability_score:.0%}")

        return RoutingDecision(
            sku_id=sku_id, destination=destination, weight_kg=weight_kg,
            selected_route=best_route, total_cost=round(total_cost, 2),
            estimated_arrival_days=round(total_days, 1),
            confidence_interval=(round(max(ci_lo, 1), 1), round(ci_hi, 1)),
            reasoning=reasoning,
        )


def run_routing_demo():
    routes = [
        ShipmentRoute("R1", "CN", "US", "DHL", "air", 5.0, 1.0, 8.5, 500, 2.0, 0.95),
        ShipmentRoute("R2", "CN", "US", "FedEx", "air", 4.0, 0.8, 10.0, 300, 1.5, 0.97),
        ShipmentRoute("R3", "CN", "US", "Cainiao", "sea", 25.0, 5.0, 1.2, 5000, 5.0, 0.85),
        ShipmentRoute("R4", "CN", "UK", "DHL", "air", 6.0, 1.5, 9.0, 500, 3.0, 0.92),
    ]

    opt = CrossBorderRoutingOptimizer()
    result = opt.optimize("SKU-FORMULA-S1", "US", 50.0, 14, 5.0, routes)

    print(f"=== 跨境路径优化结果 ===")
    print(f"SKU: {result.sku_id} | 目的地: {result.destination} | 重量: {result.weight_kg}kg")
    print(f"选定路线: {result.selected_route.carrier} ({result.selected_route.mode})")
    print(f"总运费: ${result.total_cost:.0f} | 预计到达: {result.estimated_arrival_days} 天")
    print(f"95% CI: [{result.confidence_interval[0]}, {result.confidence_interval[1]}] 天")
    print(f"原因: {result.reasoning}")
    print("=== 跨境路径优化结果 ===")


if __name__ == "__main__":
    run_routing_demo()

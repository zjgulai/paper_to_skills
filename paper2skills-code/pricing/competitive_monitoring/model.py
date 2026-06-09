"""
Competitive Price Monitoring — 竞品价格追踪 + PCI 指数
paper2skills-code: 17-价格优化 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CompetitorPrice:
    competitor_id: str
    sku_match_score: float  # 0-1 相似度
    price: float
    currency: str
    recorded_at: str = ""
    platform: str = "amazon"


@dataclass
class PCIReport:
    """Price Competitiveness Index 报告"""
    sku_id: str
    own_price: float
    market_avg: float
    market_min: float
    market_max: float
    pci_score: float          # >1 = 偏贵, <1 = 偏便宜
    position_pct: float       # 在市场中的价格分位数 (0=最低, 1=最高)
    action_required: bool
    recommended_action: str


class PCICalculator:
    """价格竞争力指数（PCI = own_price / market_weighted_avg）"""
    def __init__(self, overpriced_threshold: float = 1.05,
                 underpriced_threshold: float = 0.90):
        self.overpriced_threshold = overpriced_threshold
        self.underpriced_threshold = underpriced_threshold

    def calculate(self, sku_id: str, own_price: float,
                  competitor_prices: list[CompetitorPrice]) -> PCIReport:
        if not competitor_prices:
            return PCIReport(sku_id, own_price, own_price, own_price, own_price,
                             1.0, 0.5, False, "无竞品数据")

        # 加权均价（按相似度加权）
        weights = [c.sku_match_score for c in competitor_prices]
        total_w = sum(weights) or 1.0
        market_avg = sum(c.price * c.sku_match_score for c in competitor_prices) / total_w
        prices = [c.price for c in competitor_prices]
        market_min = min(prices)
        market_max = max(prices)

        pci = own_price / market_avg if market_avg > 0 else 1.0
        position_pct = sum(1 for p in prices if p < own_price) / len(prices)

        if pci > self.overpriced_threshold:
            action = f"建议降价至 ${market_avg * 0.99:.2f}（当前比市均价高 {(pci-1)*100:.1f}%）"
            action_required = True
        elif pci < self.underpriced_threshold:
            action = f"价格偏低，可考虑上调至 ${market_avg * 0.97:.2f}（保留竞争优势）"
            action_required = True
        else:
            action = "价格竞争力合理，维持现价"
            action_required = False

        return PCIReport(
            sku_id=sku_id, own_price=own_price,
            market_avg=round(market_avg, 2),
            market_min=round(market_min, 2),
            market_max=round(market_max, 2),
            pci_score=round(pci, 3),
            position_pct=round(position_pct, 2),
            action_required=action_required,
            recommended_action=action,
        )


class CrossElasticityMonitor:
    """交叉价格弹性：评估竞品降价对自身需求的冲击"""
    def __init__(self, cross_elasticity: float = 0.8):
        self.cross_elasticity = cross_elasticity  # 通常 0.3-1.5

    def estimate_demand_impact(self, own_demand: float,
                               competitor_price_change_pct: float) -> float:
        """竞品降价 x% → 我方需求变化量"""
        demand_change_pct = self.cross_elasticity * (-competitor_price_change_pct)
        return own_demand * demand_change_pct / 100


def run_competitive_monitoring_demo():
    """母婴奶粉竞品价格监控演示"""
    calculator = PCICalculator()
    cross_monitor = CrossElasticityMonitor()

    own_price = 45.0
    own_demand = 200.0  # 月销量

    competitors = [
        CompetitorPrice("holle_official", 0.95, 47.0, "USD"),
        CompetitorPrice("hipp_store", 0.92, 43.5, "USD"),
        CompetitorPrice("bellamy_shop", 0.88, 41.0, "USD"),
        CompetitorPrice("generic_brand", 0.60, 35.0, "USD"),
    ]

    print("=== 竞品价格监控报告 ===")
    report = calculator.calculate("SKU-FORMULA-S1", own_price, competitors)
    print(f"SKU: SKU-FORMULA-S1")
    print(f"  自身价格: ${report.own_price:.2f}")
    print(f"  市场均价: ${report.market_avg:.2f} (最低 ${report.market_min:.2f}, 最高 ${report.market_max:.2f})")
    print(f"  PCI 指数: {report.pci_score:.3f} (>1=偏贵, <1=偏便宜)")
    print(f"  价格分位: {report.position_pct:.0%} (0=最低, 100%=最高)")
    print(f"  建议动作: {report.recommended_action}")

    print("\n=== 竞品价格监控报告 ===\n")
    competitor_drop = -8.0  # 竞品降价 8%
    impact = cross_monitor.estimate_demand_impact(own_demand, competitor_drop)
    print(f"  若主竞品降价 {abs(competitor_drop):.0f}%，预计我方需求下降 {abs(impact):.0f} 件/月")

    print("--- 竞品降价冲击模拟 ---")


if __name__ == "__main__":
    run_competitive_monitoring_demo()

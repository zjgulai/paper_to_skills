"""
Cross-Border Price Harmonization — 跨境价格协调：PPP 归一化 + 汇率缓冲带
paper2skills-code: 17-价格优化 | 母婴出海跨境电商
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MarketPrice:
    market: str          # US / UK / DE / JP / AU
    currency: str
    local_price: float
    exchange_rate_to_usd: float
    platform_fee_pct: float  # Amazon/Lazada 佣金率
    tax_rate: float
    logistics_cost_usd: float


@dataclass
class HarmonizationReport:
    sku_id: str
    base_price_usd: float
    markets: list[dict]  # 各市场建议价格
    arbitrage_risk: bool  # 价差是否会引发跨市套利


class PPPNormalizer:
    """购买力平价指数归一化（简化版）"""
    PPP_INDEX = {"US": 1.00, "UK": 0.85, "DE": 0.80, "JP": 0.75, "AU": 0.90}

    def adjust_price(self, base_usd: float, market: str) -> float:
        ppp = self.PPP_INDEX.get(market, 1.0)
        return base_usd / ppp  # PPP 越低的市场，本地定价越高


class ExchangeRateBuffer:
    """汇率缓冲带（防止汇率波动触发频繁调价）"""
    def __init__(self, buffer_pct: float = 0.03):
        self.buffer_pct = buffer_pct

    def apply_buffer(self, price_usd: float, rate: float, direction: str = "export") -> float:
        if direction == "export":
            # 出口：用略低于当前汇率（保守估计）
            adjusted_rate = rate * (1 - self.buffer_pct)
        else:
            adjusted_rate = rate * (1 + self.buffer_pct)
        return price_usd * adjusted_rate


class CrossBorderPricingOptimizer:
    """跨境定价协调器"""

    ARBITRAGE_THRESHOLD = 0.20  # 价差超过 20% 触发套利风险

    def __init__(self):
        self.ppp = PPPNormalizer()
        self.buffer = ExchangeRateBuffer()

    def harmonize(self, sku_id: str, cost_usd: float,
                  markets: list[MarketPrice],
                  target_margin: float = 0.40) -> HarmonizationReport:
        base_usd = cost_usd / (1 - target_margin)
        usd_prices = []
        market_results = []

        for m in markets:
            ppp_price_usd = self.ppp.adjust_price(base_usd, m.market)
            gross_price_usd = ppp_price_usd / (1 - m.platform_fee_pct - m.tax_rate)
            gross_price_usd += m.logistics_cost_usd
            local_price = self.buffer.apply_buffer(gross_price_usd, m.exchange_rate_to_usd)
            usd_prices.append(gross_price_usd)
            market_results.append({
                "market": m.market, "currency": m.currency,
                "local_price": round(local_price, 2),
                "usd_equivalent": round(gross_price_usd, 2),
                "margin": round((gross_price_usd - cost_usd - m.logistics_cost_usd)
                                / gross_price_usd, 3),
            })

        price_spread = (max(usd_prices) - min(usd_prices)) / min(usd_prices) if usd_prices else 0
        return HarmonizationReport(
            sku_id=sku_id, base_price_usd=round(base_usd, 2),
            markets=market_results,
            arbitrage_risk=price_spread > self.ARBITRAGE_THRESHOLD,
        )


def run_cross_border_demo():
    markets = [
        MarketPrice("US", "USD", 0, 1.00, 0.15, 0.00, 3.5),
        MarketPrice("UK", "GBP", 0, 0.79, 0.15, 0.20, 5.0),
        MarketPrice("DE", "EUR", 0, 0.92, 0.15, 0.19, 5.5),
        MarketPrice("AU", "AUD", 0, 1.53, 0.15, 0.10, 7.0),
    ]

    opt = CrossBorderPricingOptimizer()
    report = opt.harmonize("SKU-FORMULA-S1", cost_usd=28.0, markets=markets)

    print(f"=== 跨境定价协调报告：{report.sku_id} ===")
    print(f"基准价（USD）: ${report.base_price_usd:.2f}")
    for m in report.markets:
        print(f"  {m['market']:4s} | {m['currency']} {m['local_price']:>8.2f}"
              f" (≈ USD {m['usd_equivalent']:.2f}) | 毛利率 {m['margin']:.1%}")
    print(f"=== 跨境定价协调报告：{report.sku_id} ===")
    print("套利风险: " + ("是（价差过大）" if report.arbitrage_risk else "否"))


if __name__ == "__main__":
    run_cross_border_demo()

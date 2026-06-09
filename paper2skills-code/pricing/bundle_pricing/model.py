"""
Bundle Pricing — 捆绑定价：混合捆绑 + 估值分布优化
paper2skills-code: 17-价格优化 | 母婴出海跨境电商
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import List


@dataclass
class Product:
    sku_id: str
    name: str
    standalone_price: float
    cost: float
    avg_daily_demand: float
    category: str  # formula / wipes / toy / gear


@dataclass
class Bundle:
    bundle_id: str
    products: list[Product]
    bundle_price: float
    discount_pct: float
    estimated_demand: float
    margin: float
    gross_profit: float


class ValuationEstimator:
    """消费者估值分布估计（简化版：基于需求弹性和互补性）"""

    COMPLEMENTARITY = {
        ("formula", "wipes"): 0.85,   # 高互补
        ("formula", "gear"):  0.60,
        ("wipes",   "toy"):   0.40,
        ("formula", "toy"):   0.30,
    }

    def complementarity(self, cat_a: str, cat_b: str) -> float:
        key = tuple(sorted([cat_a, cat_b]))
        return self.COMPLEMENTARITY.get(key, 0.20)

    def bundle_willingness_to_pay(self, products: list[Product]) -> float:
        """捆绑产品的消费者支付意愿（有互补折算）"""
        standalone_sum = sum(p.standalone_price for p in products)
        if len(products) < 2:
            return standalone_sum
        # 计算互补性加成
        comp_bonus = 0.0
        for a, b in combinations(products, 2):
            c = self.complementarity(a.category, b.category)
            comp_bonus += c * min(a.standalone_price, b.standalone_price) * 0.15
        return standalone_sum + comp_bonus


class BundlePricingOptimizer:
    """混合捆绑定价优化器"""

    def __init__(self, min_margin: float = 0.25, target_discount_range=(0.05, 0.25)):
        self.min_margin = min_margin
        self.min_disc, self.max_disc = target_discount_range
        self.estimator = ValuationEstimator()

    def _demand_at_discount(self, base_demand: float, discount_pct: float) -> float:
        elasticity = -1.8
        return base_demand * (1 + elasticity * (-discount_pct))

    def generate_bundles(self, products: list[Product], max_size: int = 3) -> list[Bundle]:
        bundles = []
        for size in range(2, min(max_size + 1, len(products) + 1)):
            for combo in combinations(products, size):
                combo = list(combo)
                wtp = self.estimator.bundle_willingness_to_pay(combo)
                total_cost = sum(p.cost for p in combo)
                standalone_sum = sum(p.standalone_price for p in combo)
                floor_price = total_cost / (1 - self.min_margin)

                # 最优折扣搜索
                best_profit = 0.0
                best_price = standalone_sum
                best_discount = 0.0

                for disc in [i * 0.01 for i in range(
                        int(self.min_disc * 100), int(self.max_disc * 100) + 1, 2)]:
                    price = standalone_sum * (1 - disc)
                    if price < floor_price or price > wtp:
                        continue
                    avg_base_demand = sum(p.avg_daily_demand for p in combo) / len(combo)
                    est_demand = self._demand_at_discount(avg_base_demand, disc)
                    profit = (price - total_cost) * est_demand
                    if profit > best_profit:
                        best_profit = profit
                        best_price = price
                        best_discount = disc

                if best_discount > 0:
                    avg_base = sum(p.avg_daily_demand for p in combo) / len(combo)
                    est_demand = self._demand_at_discount(avg_base, best_discount)
                    margin = (best_price - total_cost) / best_price
                    bundles.append(Bundle(
                        bundle_id="+".join(p.sku_id for p in combo),
                        products=combo, bundle_price=round(best_price, 2),
                        discount_pct=round(best_discount * 100, 1),
                        estimated_demand=round(est_demand, 1),
                        margin=round(margin, 3),
                        gross_profit=round((best_price - total_cost) * est_demand, 2),
                    ))

        bundles.sort(key=lambda b: b.gross_profit, reverse=True)
        return bundles


def run_bundle_demo():
    products = [
        Product("F-S1", "有机奶粉 S1", 45.0, 28.0, 12.0, "formula"),
        Product("W-80", "婴儿湿巾 80抽", 8.5,  4.5, 30.0, "wipes"),
        Product("T-01", "安抚奶嘴",      6.0,  2.5, 15.0, "toy"),
        Product("G-BB", "婴儿背带",      35.0, 18.0, 5.0,  "gear"),
    ]

    optimizer = BundlePricingOptimizer()
    bundles = optimizer.generate_bundles(products, max_size=3)

    print("=== 捆绑定价方案（按毛利排序）===")
    for b in bundles[:5]:
        print(f"  {b.bundle_id}")
        print(f"    捆绑价: ${b.bundle_price:.2f} (折扣 {b.discount_pct:.0f}%)")
        print(f"    预计日销: {b.estimated_demand:.1f} 套 | 日毛利: ${b.gross_profit:.0f} | 毛利率: {b.margin:.1%}")

    print("=== 捆绑定价方案（按毛利排序）===\n")


if __name__ == "__main__":
    run_bundle_demo()

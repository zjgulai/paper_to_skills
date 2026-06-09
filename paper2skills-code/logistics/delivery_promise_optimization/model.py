"""
Delivery Promise Optimization — 时效承诺优化：转化率与准时率的帕累托
paper2skills-code: 18-物流履约 | 母婴出海跨境电商

纯 Python 标准库实现（无外部依赖）
Python 3.14 兼容
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────

@dataclass
class DeliveryRecord:
    """单条历史配送记录"""
    order_id: str
    promised_days: int
    actual_days: float
    route: str
    season: str
    carrier: str = "fedex"

    @property
    def on_time(self) -> bool:
        return self.actual_days <= self.promised_days


@dataclass
class OptimizationResult:
    """时效承诺优化结果"""
    optimal_days: int
    actual_on_time_rate: float
    target_on_time_rate: float
    total_records: int
    quantile_used: float
    adjustment_factor: float = 1.0
    adjusted_days: Optional[int] = None


# ──────────────────────────────────────────────
# 历史分位数估计器
# ──────────────────────────────────────────────

class HistoricalQuantileEstimator:
    """
    基于历史配送记录的分位数时效估计。

    核心方法：对实际配送天数排序后取指定分位数，
    作为"以该分位数比例的订单能在此天数内送达"的保守估计。
    """

    def estimate(
        self,
        records: list[DeliveryRecord],
        quantile: float = 0.95,
        route: Optional[str] = None,
        season: Optional[str] = None,
        carrier: Optional[str] = None,
    ) -> float:
        filtered = records
        if route:
            filtered = [r for r in filtered if r.route == route]
        if season:
            filtered = [r for r in filtered if r.season == season]
        if carrier:
            filtered = [r for r in filtered if r.carrier == carrier]

        if not filtered:
            return 7.0

        actual_days = sorted(r.actual_days for r in filtered)
        n = len(actual_days)
        idx = min(int(quantile * n), n - 1)
        return actual_days[idx]

    def estimate_by_groups(
        self,
        records: list[DeliveryRecord],
        quantile: float = 0.95,
    ) -> dict[str, float]:
        routes = {r.route for r in records}
        seasons = {r.season for r in records}
        results: dict[str, float] = {}
        for route in routes:
            for season in seasons:
                key = f"{route}|{season}"
                val = self.estimate(records, quantile=quantile, route=route, season=season)
                results[key] = val
        return results


# ──────────────────────────────────────────────
# 动态调整器
# ──────────────────────────────────────────────

HOLIDAY_FACTOR = 1.35
PROMO_FACTOR = 1.20
WEATHER_FACTOR = 1.15


class DynamicAdjuster:
    """
    节假日/促销/天气调整因子，对基础时效承诺进行乘法修正。

    因子叠加规则：f_total = f_holiday × f_promo × f_weather（取最大值策略避免过度保守）
    """

    def adjust(
        self,
        base_days: float,
        holiday: bool = False,
        promo: bool = False,
        weather: bool = False,
        custom_factor: Optional[float] = None,
    ) -> float:
        if custom_factor is not None:
            return base_days * custom_factor

        factor = 1.0
        if holiday:
            factor = max(factor, HOLIDAY_FACTOR)
        if promo:
            factor = max(factor, PROMO_FACTOR)
        if weather:
            factor = max(factor, WEATHER_FACTOR)

        return base_days * factor

    def get_factor(
        self,
        holiday: bool = False,
        promo: bool = False,
        weather: bool = False,
    ) -> float:
        factor = 1.0
        if holiday:
            factor = max(factor, HOLIDAY_FACTOR)
        if promo:
            factor = max(factor, PROMO_FACTOR)
        if weather:
            factor = max(factor, WEATHER_FACTOR)
        return factor


# ──────────────────────────────────────────────
# 承诺时效优化器
# ──────────────────────────────────────────────

class PromiseOptimizer:
    """
    在准时率约束下，最小化配送承诺天数。

    算法：从 1 天开始枚举整数天数，找到满足目标准时率的最小值。
    约束：actual_on_time_rate = P(actual_days ≤ promised_days) ≥ target_on_time_rate
    """

    MAX_DAYS = 60

    def __init__(self, target_on_time_rate: float = 0.95) -> None:
        self._target = target_on_time_rate

    def optimize(
        self,
        records: list[DeliveryRecord],
        holiday: bool = False,
        promo: bool = False,
        weather: bool = False,
    ) -> OptimizationResult:
        if not records:
            return OptimizationResult(
                optimal_days=7,
                actual_on_time_rate=0.0,
                target_on_time_rate=self._target,
                total_records=0,
                quantile_used=self._target,
            )

        actual_days_list = sorted(r.actual_days for r in records)
        n = len(actual_days_list)

        adjuster = DynamicAdjuster()
        factor = adjuster.get_factor(holiday=holiday, promo=promo, weather=weather)

        optimal_days = self.MAX_DAYS
        best_rate = 0.0

        for candidate_days in range(1, self.MAX_DAYS + 1):
            on_time_count = sum(1 for d in actual_days_list if d <= candidate_days)
            rate = on_time_count / n
            if rate >= self._target:
                optimal_days = candidate_days
                best_rate = rate
                break
            best_rate = rate

        adjusted_days = int(optimal_days * factor) if factor > 1.0 else None

        return OptimizationResult(
            optimal_days=optimal_days,
            actual_on_time_rate=best_rate,
            target_on_time_rate=self._target,
            total_records=n,
            quantile_used=self._target,
            adjustment_factor=factor,
            adjusted_days=adjusted_days,
        )


# ──────────────────────────────────────────────
# 样本数据生成器
# ──────────────────────────────────────────────

def generate_sample_records(
    n: int = 100,
    seed: int = 42,
) -> list[DeliveryRecord]:
    rng = random.Random(seed)
    routes = ["CN-US-LA", "CN-US-NY", "CN-EU-HH", "CN-UK-LHR"]
    seasons = ["regular", "q4_peak", "promo", "holiday"]
    carriers = ["fedex", "ups", "dhl", "usps"]

    base_days = {
        "CN-US-LA": (18.0, 4.0),
        "CN-US-NY": (20.0, 4.5),
        "CN-EU-HH": (22.0, 5.0),
        "CN-UK-LHR": (21.0, 4.0),
    }
    season_multiplier = {
        "regular": 1.0,
        "q4_peak": 1.35,
        "promo": 1.20,
        "holiday": 1.15,
    }

    records = []
    for i in range(n):
        route = rng.choice(routes)
        season = rng.choice(seasons)
        carrier = rng.choice(carriers)
        mean, std = base_days[route]
        mult = season_multiplier[season]
        actual = max(1.0, rng.gauss(mean * mult, std * mult))
        promised = int(mean * mult * 1.1) + 1

        records.append(DeliveryRecord(
            order_id=f"ORD-{i:04d}",
            promised_days=promised,
            actual_days=round(actual, 1),
            route=route,
            season=season,
            carrier=carrier,
        ))
    return records


# ──────────────────────────────────────────────
# 测试
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 60)
    print("Delivery Promise Optimization — 时效承诺优化测试")
    print("=" * 60)

    records = generate_sample_records(n=100, seed=42)
    print(f"\n[✓] 生成 {len(records)} 条历史配送记录")

    # P95 分位数时效估计
    estimator = HistoricalQuantileEstimator()
    p95 = estimator.estimate(records, quantile=0.95)
    p50 = estimator.estimate(records, quantile=0.50)
    print(f"\n[分位数估计]")
    print(f"  P50 中位数时效: {p50:.1f} 天")
    print(f"  P95 保守时效:   {p95:.1f} 天")
    assert p95 > p50, "P95 应 > P50"
    print(f"  ✓ P95 > P50 验证通过")

    # 节假日调整
    adjuster = DynamicAdjuster()
    base = 20.0
    holiday_adj = adjuster.adjust(base, holiday=True)
    promo_adj = adjuster.adjust(base, promo=True)
    normal_adj = adjuster.adjust(base)

    print(f"\n[动态调整因子验证]")
    print(f"  基础时效: {base:.1f} 天")
    print(f"  节假日调整: {holiday_adj:.1f} 天 (×{HOLIDAY_FACTOR})")
    print(f"  促销调整:   {promo_adj:.1f} 天 (×{PROMO_FACTOR})")
    print(f"  无调整:     {normal_adj:.1f} 天")
    assert holiday_adj > normal_adj, "节假日调整后应 > 基础时效"
    assert holiday_adj > promo_adj, "节假日因子 > 促销因子"
    print(f"  ✓ 调整因子验证通过")

    # 在准时率约束下最优化
    optimizer = PromiseOptimizer(target_on_time_rate=0.95)
    result = optimizer.optimize(records)
    print(f"\n[P95 时效承诺优化]")
    print(f"  最优承诺天数: {result.optimal_days} 天")
    print(f"  实际准时率:   {result.actual_on_time_rate:.1%}")
    print(f"  目标准时率:   {result.target_on_time_rate:.1%}")
    assert result.actual_on_time_rate >= result.target_on_time_rate, \
        f"实际准时率 {result.actual_on_time_rate:.1%} 应 ≥ 目标 {result.target_on_time_rate:.1%}"
    print(f"  ✓ 准时率约束验证通过")

    # 节假日场景
    holiday_result = optimizer.optimize(records, holiday=True)
    print(f"\n[节假日场景优化]")
    print(f"  调整因子: ×{holiday_result.adjustment_factor}")
    print(f"  节假日承诺天数: {holiday_result.adjusted_days} 天")
    assert holiday_result.adjustment_factor == HOLIDAY_FACTOR, "节假日因子应为 1.35"
    assert holiday_result.adjusted_days >= result.optimal_days, "节假日承诺 ≥ 常规承诺"
    print(f"  ✓ 节假日调整验证通过")

    # 分组分位数
    group_results = estimator.estimate_by_groups(records, quantile=0.95)
    print(f"\n[分路线 × 分季节 P95 估计]")
    for key, days in sorted(group_results.items())[:4]:
        print(f"  {key}: {days:.1f} 天")
    print(f"  ✓ 分组估计完成，共 {len(group_results)} 组")

    print("\n" + "=" * 60)
    print("[✓] 所有场景验证通过 — Delivery Promise Optimization")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()

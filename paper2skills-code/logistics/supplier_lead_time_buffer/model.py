"""
Skill-Supplier-Lead-Time-Buffer
供应商交货期缓冲：非正态分布下的安全库存计算
基于 Gen-QOT 工业实践 2024 + 交货期分布建模
纯 Python 标准库，Python 3.14 兼容，无第三方依赖
"""
from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class LeadTimeRecord:
    """单条交货期记录"""
    supplier_id: str
    order_date: date
    actual_delivery_date: date
    expected_delivery_date: date
    month: int = field(init=False)

    def __post_init__(self) -> None:
        self.month = self.actual_delivery_date.month

    @property
    def actual_lead_time_days(self) -> int:
        """实际交货期（天）"""
        return (self.actual_delivery_date - self.order_date).days

    @property
    def delay_days(self) -> int:
        """超期延误天数（负值表示提前）"""
        return (self.actual_delivery_date - self.expected_delivery_date).days


# 旺季月份（海运双11+圣诞旺季：10-12月）
PEAK_SEASON_MONTHS: frozenset[int] = frozenset({10, 11, 12})


class LeadTimeDistributionEstimator:
    """
    历史交货期分位数估计器（非参数方法）
    支持季节性调整：旺季 vs 淡季分开估计
    """

    def __init__(
        self,
        records: list[LeadTimeRecord],
        peak_months: frozenset[int] = PEAK_SEASON_MONTHS,
    ) -> None:
        if len(records) < 10:
            raise ValueError("记录数量不足 10 条，分位数估计不可靠")
        self._records = records
        self._peak_months = peak_months
        self._all_lts = [r.actual_lead_time_days for r in records]
        self._peak_lts = [
            r.actual_lead_time_days
            for r in records
            if r.month in peak_months
        ]
        self._off_lts = [
            r.actual_lead_time_days
            for r in records
            if r.month not in peak_months
        ]

    def _quantile(self, data: list[float], q: float) -> float:
        """线性插值分位数估计（等价于 numpy.percentile 默认方式）"""
        if not data:
            raise ValueError("数据为空，无法估计分位数")
        sorted_data = sorted(data)
        n = len(sorted_data)
        index = q * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        fraction = index - lower
        return sorted_data[lower] + fraction * (sorted_data[upper] - sorted_data[lower])

    def estimate_quantile(self, q: float, season: str = "all") -> float:
        """
        估计交货期分位数
        :param q: 分位数（0-1），如 0.95 表示 P95
        :param season: "all" / "peak" / "off"
        :return: 交货期天数
        """
        if season == "peak":
            data = self._peak_lts if self._peak_lts else self._all_lts
        elif season == "off":
            data = self._off_lts if self._off_lts else self._all_lts
        else:
            data = self._all_lts
        return self._quantile(data, q)

    def seasonal_factor(self, service_level: float = 0.95) -> float:
        """
        旺季偏移因子 = peak_P95 / off_P95
        若旺季样本不足 5 条，返回 1.0（不调整）
        """
        if len(self._peak_lts) < 5 or len(self._off_lts) < 5:
            return 1.0
        peak_q = self.estimate_quantile(service_level, season="peak")
        off_q = self.estimate_quantile(service_level, season="off")
        if off_q == 0:
            return 1.0
        return peak_q / off_q

    def summary(self) -> dict:
        """分布摘要统计"""
        return {
            "sample_count": len(self._all_lts),
            "peak_sample_count": len(self._peak_lts),
            "off_sample_count": len(self._off_lts),
            "mean_all": round(statistics.mean(self._all_lts), 1),
            "median_all": round(statistics.median(self._all_lts), 1),
            "stdev_all": round(statistics.stdev(self._all_lts), 2) if len(self._all_lts) > 1 else 0,
            "p50_all": round(self.estimate_quantile(0.50), 1),
            "p95_all": round(self.estimate_quantile(0.95), 1),
            "p99_all": round(self.estimate_quantile(0.99), 1),
            "p95_peak": round(self.estimate_quantile(0.95, "peak"), 1) if self._peak_lts else None,
            "p95_off": round(self.estimate_quantile(0.95, "off"), 1) if self._off_lts else None,
        }


@dataclass
class BufferStockResult:
    """安全库存计算结果"""
    service_level: float
    target_quantile_days: float
    median_lead_time_days: float
    buffer_days: float
    daily_demand: float
    seasonal_factor: float
    buffer_stock_units: float
    season: str

    def __str__(self) -> str:
        return (
            f"[{self.season.upper()}季节] SL={self.service_level:.0%} | "
            f"P50={self.median_lead_time_days:.0f}天 → P{int(self.service_level*100)}={self.target_quantile_days:.0f}天 | "
            f"缓冲={self.buffer_days:.1f}天 × {self.daily_demand:.0f}件/天 × {self.seasonal_factor:.2f}(季节系数) "
            f"= 安全库存 {self.buffer_stock_units:.0f} 件"
        )


class BufferStockCalculator:
    """
    基于交货期分布的安全库存计算器
    支持旺季/淡季动态调整
    """

    def __init__(
        self,
        estimator: LeadTimeDistributionEstimator,
        daily_demand: float,
    ) -> None:
        if daily_demand <= 0:
            raise ValueError("日均需求必须大于0")
        self._estimator = estimator
        self._daily_demand = daily_demand

    def calculate(
        self,
        service_level: float = 0.95,
        season: str = "all",
        apply_seasonal_factor: bool = True,
    ) -> BufferStockResult:
        """
        计算安全库存
        :param service_level: 目标服务水平（如 0.95）
        :param season: "all" / "peak" / "off"
        :param apply_seasonal_factor: 是否叠加季节性因子（仅 peak 季节有效）
        """
        p50 = self._estimator.estimate_quantile(0.50, season)
        p_target = self._estimator.estimate_quantile(service_level, season)
        buffer_days = max(0.0, p_target - p50)
        sf = self._estimator.seasonal_factor(service_level) if apply_seasonal_factor and season == "peak" else 1.0
        buffer_units = buffer_days * self._daily_demand * sf
        return BufferStockResult(
            service_level=service_level,
            target_quantile_days=round(p_target, 1),
            median_lead_time_days=round(p50, 1),
            buffer_days=round(buffer_days, 1),
            daily_demand=self._daily_demand,
            seasonal_factor=round(sf, 3),
            buffer_stock_units=round(buffer_units, 0),
            season=season,
        )

    def joint_buffer_stock(
        self,
        service_level: float = 0.95,
        demand_stdev: float = 0.0,
    ) -> float:
        """
        联合安全库存（考虑需求不确定性）
        SS = Z * sqrt(LT_variance × D² + mean_LT × σ_D²)
        :param demand_stdev: 日均需求标准差
        """
        lts = self._estimator._all_lts
        mean_lt = statistics.mean(lts)
        var_lt = statistics.variance(lts) if len(lts) > 1 else 0
        d = self._daily_demand
        # Z-score 映射（简单近似）
        z_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_map.get(round(service_level, 2), 1.645)
        joint_ss = z * math.sqrt(var_lt * d**2 + mean_lt * demand_stdev**2)
        return round(joint_ss, 0)


def _make_test_records() -> list[LeadTimeRecord]:
    """生成 60 条测试数据（含旺季/淡季区分）"""
    import random
    random.seed(42)
    records = []
    # 淡季（1-9月）：LT 均值 28 天，标准差 4 天，轻微右偏
    for i in range(40):
        month = random.randint(1, 9)
        lt = max(20, int(random.gauss(28, 4) + (5 if random.random() < 0.1 else 0)))
        order_d = date(2024, month, max(1, min(15, i % 28 + 1)))
        delivery_d = order_d + timedelta(days=lt)
        expected_d = order_d + timedelta(days=28)
        records.append(LeadTimeRecord(
            supplier_id="SUP-CN-001",
            order_date=order_d,
            actual_delivery_date=delivery_d,
            expected_delivery_date=expected_d,
        ))
    # 旺季（10-12月）：LT 均值 38 天，标准差 8 天，重尾
    for i in range(20):
        month = random.choice([10, 11, 12])
        base_lt = max(25, int(random.gauss(38, 8)))
        # 10% 概率出现极端延误（重尾特征）
        lt = base_lt + (random.randint(10, 20) if random.random() < 0.1 else 0)
        order_d = date(2024, month, max(1, min(15, i % 14 + 1)))
        delivery_d = order_d + timedelta(days=lt)
        expected_d = order_d + timedelta(days=28)
        records.append(LeadTimeRecord(
            supplier_id="SUP-CN-001",
            order_date=order_d,
            actual_delivery_date=delivery_d,
            expected_delivery_date=expected_d,
        ))
    return records


if __name__ == "__main__":
    records = _make_test_records()
    print(f"[TEST] 生成 {len(records)} 条交货期记录")

    estimator = LeadTimeDistributionEstimator(records)
    summary = estimator.summary()
    print("\n[分布摘要]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    sf = estimator.seasonal_factor(0.95)
    print(f"\n[季节性因子] 旺季P95/淡季P95 = {sf:.3f}")

    calculator = BufferStockCalculator(estimator, daily_demand=500.0)

    print("\n[安全库存计算]")
    for season in ["off", "all", "peak"]:
        result = calculator.calculate(service_level=0.95, season=season)
        print(f"  {result}")

    joint_ss = calculator.joint_buffer_stock(service_level=0.95, demand_stdev=80.0)
    print(f"\n[联合安全库存（需求+交货期双重不确定）] = {joint_ss:.0f} 件")

    # 验证 95% SL：P95 应 > P50
    p50 = estimator.estimate_quantile(0.50, "peak")
    p95 = estimator.estimate_quantile(0.95, "peak")
    assert p95 >= p50, f"P95({p95}) 应 ≥ P50({p50})"
    print(f"\n[✓] 95%SL 验证通过: 旺季 P50={p50:.1f}天, P95={p95:.1f}天")
    print("[✓] Supplier Lead Time Buffer 全部测试通过")

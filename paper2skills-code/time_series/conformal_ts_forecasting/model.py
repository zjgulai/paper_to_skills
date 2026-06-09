"""
Conformal Time Series Forecasting — 共形时序预测
滚动校准窗口 + EnbPI 集成引导预测区间

纯 Python 标准库 + statistics，无 sklearn/pandas 依赖
Python 3.14 兼容
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass
class TimeSeriesRecord:
    """单步时序观测记录"""
    timestamp: int    # 时间步（整数索引或 Unix 时间戳）
    actual: float     # 真实值
    predicted: float  # 点预测值

    @property
    def residual(self) -> float:
        return self.actual - self.predicted


@dataclass
class PredictionInterval:
    """共形预测区间"""
    timestamp: int
    point_forecast: float
    lower: float           # 下界（P_alpha/2）
    upper: float           # 上界（P_{1-alpha/2}）
    alpha: float           # 显著性水平（0.10 = 90% 区间）
    calibration_size: int  # 实际使用的校准样本数

    @property
    def width(self) -> float:
        return round(self.upper - self.lower, 4)

    @property
    def coverage_ratio(self) -> float:
        """P90/P10 比值：越接近1表示预测越确定"""
        if self.lower <= 0:
            return float("inf")
        return round(self.upper / self.lower, 4)

    def contains(self, actual: float) -> bool:
        return self.lower <= actual <= self.upper


class RollingConformalForecaster:
    """
    滚动校准窗口共形预测器

    每步接收新的真实值，更新残差队列，输出下一步的预测区间。
    不依赖分布假设，在温和混合条件下保证渐近覆盖率。
    """

    def __init__(self, window_size: int = 14, alpha: float = 0.10) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha 需在 (0, 1) 之间，当前: {alpha}")
        self.window_size = window_size
        self.alpha = alpha
        self._calibration_errors: list[float] = []  # 绝对残差滚动队列

    def update_calibration(self, recent_errors: list[float]) -> None:
        """加入新残差，丢弃超出滚动窗口的旧残差"""
        self._calibration_errors.extend(abs(e) for e in recent_errors)
        if len(self._calibration_errors) > self.window_size:
            self._calibration_errors = self._calibration_errors[-self.window_size:]

    def predict_interval(
        self,
        point_forecast: float,
        alpha: float | None = None,
    ) -> tuple[float, float]:
        """
        输出共形预测区间 (lower, upper)
        区间 = 点预测 ± Q_{1-alpha}(|残差|)
        """
        a = alpha if alpha is not None else self.alpha
        if not self._calibration_errors:
            return (round(point_forecast, 4), round(point_forecast, 4))
        q = self._quantile(self._calibration_errors, 1 - a)
        return (round(point_forecast - q, 4), round(point_forecast + q, 4))

    def predict_interval_full(
        self,
        timestamp: int,
        point_forecast: float,
        alpha: float | None = None,
    ) -> PredictionInterval:
        a = alpha if alpha is not None else self.alpha
        lower, upper = self.predict_interval(point_forecast, a)
        return PredictionInterval(
            timestamp=timestamp,
            point_forecast=round(point_forecast, 4),
            lower=lower,
            upper=upper,
            alpha=a,
            calibration_size=len(self._calibration_errors),
        )

    @staticmethod
    def _quantile(data: list[float], q: float) -> float:
        """线性插值分位数（等价于 numpy.quantile 的 linear 方法）"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        idx = q * (n - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= n:
            return sorted_data[-1]
        frac = idx - lo
        return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac

    @property
    def calibration_size(self) -> int:
        return len(self._calibration_errors)


class EnbPI:
    """
    Ensemble Bootstrapping Prediction Interval（简化版）

    原论文：Chen & Yanovich (2021)
    "Conformal prediction interval for dynamic time-series"

    实现要点：
    - B 个引导样本估计集成残差标准差
    - 预测区间 = 点预测 ± z_{1-alpha/2} × mean(bootstrap_stds)
    - 正态分位数用 Abramowitz & Stegun 26.2.17 近似
    """

    def __init__(self, n_bootstrap: int = 50, seed: int = 42) -> None:
        self.n_bootstrap = n_bootstrap
        self._rng = random.Random(seed)
        self._bootstrap_noise_stds: list[float] = []
        self._fitted = False

    def fit(self, train_series: list[float]) -> "EnbPI":
        n = len(train_series)
        if n < 4:
            raise ValueError("训练序列至少需要4个点")
        bootstrap_residual_stds: list[float] = []
        for _ in range(self.n_bootstrap):
            indices = [self._rng.randint(0, n - 1) for _ in range(n)]
            sample = [train_series[i] for i in sorted(indices)]
            mean_val = statistics.mean(sample)
            residuals = [v - mean_val for v in sample]
            std = statistics.stdev(residuals) if len(residuals) > 1 else 0.0
            bootstrap_residual_stds.append(std)
        self._bootstrap_noise_stds = bootstrap_residual_stds
        self._fitted = True
        return self

    def predict_interval(
        self,
        point_forecast: float,
        alpha: float = 0.10,
    ) -> tuple[float, float]:
        """集成引导预测区间：点预测 ± z_{1-alpha/2} × mean(bootstrap_stds)"""
        if not self._fitted:
            raise RuntimeError("请先调用 fit()")
        ensemble_std = statistics.mean(self._bootstrap_noise_stds)
        z = self._normal_quantile(1 - alpha / 2)
        margin = z * ensemble_std
        return (round(point_forecast - margin, 4), round(point_forecast + margin, 4))

    @staticmethod
    def _normal_quantile(p: float) -> float:
        """
        正态分布分位数近似（Abramowitz & Stegun 26.2.17）
        误差 < 0.001，足够用于预测区间计算
        """
        if p <= 0 or p >= 1:
            raise ValueError(f"p 需在 (0,1) 之间: {p}")
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        if p < 0.5:
            t = math.sqrt(-2 * math.log(p))
            sign = -1
        else:
            t = math.sqrt(-2 * math.log(1 - p))
            sign = 1
        num = c0 + c1 * t + c2 * t ** 2
        den = 1 + d1 * t + d2 * t ** 2 + d3 * t ** 3
        return sign * (t - num / den)


def evaluate_coverage(
    intervals: list[PredictionInterval],
    actuals: list[float],
) -> dict[str, float]:
    """评估预测区间的实际覆盖率和区间宽度"""
    assert len(intervals) == len(actuals)
    covered = sum(1 for pi, a in zip(intervals, actuals) if pi.contains(a))
    avg_width = statistics.mean(pi.width for pi in intervals)
    positive_ratios = [pi.coverage_ratio for pi in intervals if pi.lower > 0 and pi.coverage_ratio != float("inf")]
    avg_ratio = statistics.mean(positive_ratios) if positive_ratios else float("inf")
    return {
        "coverage_rate": round(covered / len(intervals), 4),
        "avg_width": round(avg_width, 4),
        "avg_p90_p10_ratio": round(avg_ratio, 4),
        "n_intervals": len(intervals),
    }


def _generate_demand_series(n: int = 60, seed: int = 7) -> list[float]:
    """模拟需求序列：线性趋势 + 周期波动 + 噪声"""
    rng = random.Random(seed)
    series = []
    for t in range(n):
        trend = 1000 + t * 5
        seasonal = 80 * math.sin(2 * math.pi * t / 7)
        noise = rng.gauss(0, 50)
        series.append(max(0.0, trend + seasonal + noise))
    return series


def main() -> None:
    print("=" * 60)
    print("Loop 52-B: Conformal Time Series Forecasting — 验证")
    print("=" * 60)

    series = _generate_demand_series(n=60)
    rng = random.Random(42)

    print("\n" + "─" * 50)
    print("RollingConformalForecaster — 28天滚动预测验证")
    print("─" * 50)

    forecaster = RollingConformalForecaster(window_size=14, alpha=0.10)
    init_residuals = [rng.gauss(0, 60) for _ in range(20)]
    forecaster.update_calibration(init_residuals)

    intervals: list[PredictionInterval] = []
    actuals_test: list[float] = []

    for t in range(20, 48):
        actual = series[t]
        point_pred = actual + rng.gauss(0, 55)
        pi = forecaster.predict_interval_full(timestamp=t, point_forecast=point_pred, alpha=0.10)
        intervals.append(pi)
        actuals_test.append(actual)
        forecaster.update_calibration([actual - point_pred])

    metrics = evaluate_coverage(intervals, actuals_test)
    print(f"  覆盖率: {metrics['coverage_rate']:.1%}（目标: 90%）")
    print(f"  平均区间宽度: {metrics['avg_width']:.1f}")
    print(f"  P90/P10 平均比值: {metrics['avg_p90_p10_ratio']:.3f}")

    coverage = metrics["coverage_rate"]
    assert 0.80 <= coverage <= 1.00, f"覆盖率超出合理范围: {coverage:.1%}"
    print(f"\n✅ 覆盖率验证通过: {coverage:.1%} ∈ [80%, 100%]")
    assert metrics["avg_width"] > 0, "区间宽度应 > 0"
    print(f"✅ 区间宽度有效: {metrics['avg_width']:.1f} > 0")

    print("\n" + "─" * 50)
    print("EnbPI — 集成引导预测区间验证")
    print("─" * 50)

    train_series = series[:30]
    enbpi = EnbPI(n_bootstrap=50, seed=99)
    enbpi.fit(train_series)

    test_forecasts = [series[t] + rng.gauss(0, 40) for t in range(30, 50)]
    test_actuals = series[30:50]

    enbpi_intervals: list[PredictionInterval] = []
    for pred, ts in zip(test_forecasts, range(30, 50)):
        lower, upper = enbpi.predict_interval(pred, alpha=0.10)
        enbpi_intervals.append(PredictionInterval(
            timestamp=ts,
            point_forecast=pred,
            lower=lower,
            upper=upper,
            alpha=0.10,
            calibration_size=50,
        ))

    enbpi_metrics = evaluate_coverage(enbpi_intervals, test_actuals)
    print(f"  EnbPI 覆盖率: {enbpi_metrics['coverage_rate']:.1%}")
    print(f"  EnbPI 平均区间宽度: {enbpi_metrics['avg_width']:.1f}")
    assert enbpi_metrics["avg_width"] > 0, "EnbPI 区间宽度应 > 0"
    print(f"✅ EnbPI 区间宽度有效: {enbpi_metrics['avg_width']:.1f}")

    print("\n" + "─" * 50)
    print("P90/P10 比值收窗效应（模拟大促备货场景）")
    print("─" * 50)

    forecaster2 = RollingConformalForecaster(window_size=7, alpha=0.10)
    base_demand = 1000.0
    ratios: list[float] = []

    for week, noise_std in enumerate([300, 250, 180, 120, 80, 50, 30, 15], start=1):
        errors = [rng.gauss(0, noise_std) for _ in range(7)]
        forecaster2.update_calibration(errors)
        pi = forecaster2.predict_interval_full(timestamp=week, point_forecast=base_demand)
        ratio = pi.upper / max(pi.lower, 1)
        ratios.append(ratio)
        print(f"  T-{9-week}周: [{pi.lower:.0f}, {pi.upper:.0f}] P90/P10={ratio:.3f}")

    assert ratios[-1] < ratios[0], (
        f"P90/P10比值应随时间递减: T-8周 {ratios[0]:.3f} → T-1周 {ratios[-1]:.3f}"
    )
    print(f"\n✅ 不确定性收窗验证: T-8周 {ratios[0]:.3f} → T-1周 {ratios[-1]:.3f}")
    print("\n✅ 所有验证通过 — Loop 52-B Conformal Time Series Forecasting")


if __name__ == "__main__":
    main()

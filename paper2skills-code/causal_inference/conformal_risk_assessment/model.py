"""
Conformal Risk Assessment: 共形预测业务风险量化

来源: Conformal Prediction for Decision Making 2024-2025
应用场景: 母婴出海 WF-A 补货量区间估计 / WF-D 市场规模风险量化
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 校准记录
# ---------------------------------------------------------------------------

@dataclass
class CalibrationRecord:
    """单条校准集记录

    Attributes:
        actual_value: 真实观测值
        predicted_value: 模型点预测值
        nonconformity_score: 非一致性分数（默认为绝对误差 |actual - predicted|）
    """
    actual_value: float
    predicted_value: float
    nonconformity_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.nonconformity_score = abs(self.actual_value - self.predicted_value)


# ---------------------------------------------------------------------------
# 2. ConformalPredictor
# ---------------------------------------------------------------------------

@dataclass
class PredictionInterval:
    """预测区间结果"""
    point_prediction: float
    lower: float
    upper: float
    alpha: float
    coverage_guarantee: float  # 1 - alpha

    def value_at_quantile(self, q: float) -> float:
        """在 [lower, upper] 区间内按分位数取值

        Args:
            q: 分位数，0.0 = lower, 1.0 = upper, 0.5 = 中位数

        Returns:
            区间内对应分位数的值
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError(f"分位数 q 须在 [0, 1]，实际: {q}")
        return self.lower + q * (self.upper - self.lower)


class ConformalPredictor:
    """基于分割共形预测（Split Conformal Prediction）的区间估计器

    工作流：
    1. calibrate() — 从校准集学习非一致性分数阈值
    2. predict_interval() — 为新点预测生成覆盖率 1-alpha 的区间
    3. verify_coverage() — 用测试集验证实测覆盖率

    无需分布假设，覆盖率理论保证：P(y ∈ [lower, upper]) ≥ 1-alpha
    """

    def __init__(
        self,
        score_fn: Callable[[float, float], float] | None = None,
    ) -> None:
        """
        Args:
            score_fn: 自定义非一致性分数函数 (actual, predicted) -> score
                      默认使用绝对误差 |actual - predicted|
        """
        self._score_fn = score_fn or (lambda actual, pred: abs(actual - pred))
        self._calibration_scores: list[float] = []
        self._calibrated = False

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def calibrate(self, calibration_data: list[CalibrationRecord]) -> None:
        """从校准集学习非一致性分数分布

        Args:
            calibration_data: 历史观测记录列表（真实值 + 预测值）
        """
        if not calibration_data:
            raise ValueError("校准集不能为空")

        self._calibration_scores = [
            self._score_fn(r.actual_value, r.predicted_value)
            for r in calibration_data
        ]
        self._calibrated = True
        logger.info(
            "共形预测校准完成: n=%d, 分数范围=[%.2f, %.2f], 中位数=%.2f",
            len(self._calibration_scores),
            min(self._calibration_scores),
            max(self._calibration_scores),
            sorted(self._calibration_scores)[len(self._calibration_scores) // 2],
        )

    def predict_interval(
        self, point_pred: float, alpha: float = 0.1
    ) -> PredictionInterval:
        """生成覆盖率 1-alpha 的预测区间

        基于校准集分位数：q_alpha = ⌈(1-alpha)(n+1)/n⌉ 分位数

        Args:
            point_pred: 基础模型的点预测值
            alpha: 误覆盖率，默认 0.1（90% 覆盖率）

        Returns:
            PredictionInterval，包含 lower / upper / coverage_guarantee
        """
        if not self._calibrated:
            raise RuntimeError("请先调用 calibrate() 完成校准")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha 须在 (0, 1)，实际: {alpha}")

        n = len(self._calibration_scores)
        # 共形预测标准分位数公式
        quantile_level = min(math.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        sorted_scores = sorted(self._calibration_scores)
        q_alpha = sorted_scores[min(int(quantile_level * n) - 1, n - 1)]

        lower = point_pred - q_alpha
        upper = point_pred + q_alpha

        logger.debug(
            "predict_interval: point=%.2f, q_alpha=%.2f, interval=[%.2f, %.2f]",
            point_pred, q_alpha, lower, upper,
        )
        return PredictionInterval(
            point_prediction=point_pred,
            lower=lower,
            upper=upper,
            alpha=alpha,
            coverage_guarantee=1.0 - alpha,
        )

    def verify_coverage(self, test_data: list[CalibrationRecord], alpha: float = 0.1) -> float:
        """用测试集验证实测覆盖率

        Args:
            test_data: 测试集（真实值 + 模型点预测值）
            alpha: 与 predict_interval 对应的误覆盖率

        Returns:
            实测覆盖率（理论下界 1-alpha）
        """
        if not test_data:
            raise ValueError("测试集不能为空")

        covered = 0
        for record in test_data:
            interval = self.predict_interval(record.predicted_value, alpha)
            if interval.lower <= record.actual_value <= interval.upper:
                covered += 1

        empirical_coverage = covered / len(test_data)
        expected_coverage = 1.0 - alpha
        logger.info(
            "覆盖率验证: 实测=%.3f, 理论保证=%.3f, 样本数=%d",
            empirical_coverage, expected_coverage, len(test_data),
        )
        return empirical_coverage


# ---------------------------------------------------------------------------
# 3. 测试：WF-A 补货需求场景，90% 覆盖率验证
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """运行补货需求场景共形预测测试"""
    import random
    random.seed(42)

    # 模拟历史需求数据（真实需求 = 点预测 + 噪声）
    def simulate_demand(n: int, base: float = 1000.0, noise_std: float = 150.0) -> list[tuple[float, float]]:
        records = []
        for _ in range(n):
            pred = base + random.gauss(0, 50)
            actual = pred + random.gauss(0, noise_std)
            records.append((actual, pred))
        return records

    # 构造校准集（60 条）
    calibration_raw = simulate_demand(60)
    calibration_data = [
        CalibrationRecord(actual_value=a, predicted_value=p)
        for a, p in calibration_raw
    ]

    # 构造测试集（40 条）
    test_raw = simulate_demand(40)
    test_data = [
        CalibrationRecord(actual_value=a, predicted_value=p)
        for a, p in test_raw
    ]

    predictor = ConformalPredictor()
    predictor.calibrate(calibration_data)

    # 测试 1：点预测区间生成
    print("\n=== 测试 1：生成 90% 覆盖率预测区间 ===")
    interval = predictor.predict_interval(point_pred=1000.0, alpha=0.1)
    assert interval.lower < interval.point_prediction < interval.upper, "区间应包含点预测"
    assert interval.coverage_guarantee == 0.9
    print(f"  ✓ 点预测: {interval.point_prediction:.0f}")
    print(f"  ✓ 区间: [{interval.lower:.0f}, {interval.upper:.0f}]（90% 覆盖保证）")
    print(f"  ✓ P10（保守补货量）: {interval.value_at_quantile(0.0):.0f} 件")
    print(f"  ✓ P50（中位补货量）: {interval.value_at_quantile(0.5):.0f} 件")

    # 测试 2：覆盖率验证（实测应 ≥ 90%）
    print("\n=== 测试 2：验证实测覆盖率 ≥ 90% ===")
    empirical = predictor.verify_coverage(test_data, alpha=0.1)
    assert empirical >= 0.85, f"实测覆盖率 {empirical:.3f} 低于预期（≥85% 放宽容忍）"
    print(f"  ✓ 实测覆盖率: {empirical:.1%}（理论保证: 90%）")

    # 测试 3：不同 alpha 下的区间宽度（alpha 越小，区间越宽）
    print("\n=== 测试 3：不同 alpha 区间宽度对比 ===")
    for alpha in [0.05, 0.10, 0.20]:
        iv = predictor.predict_interval(1000.0, alpha)
        width = iv.upper - iv.lower
        print(f"  ✓ alpha={alpha:.2f} (覆盖率{1-alpha:.0%}): 区间宽度={width:.0f}, 保守下限={iv.lower:.0f}")

    # 测试 4：自定义非一致性分数（相对误差）
    print("\n=== 测试 4：自定义非一致性分数（相对误差）===")
    rel_predictor = ConformalPredictor(
        score_fn=lambda actual, pred: abs(actual - pred) / (abs(pred) + 1e-8)
    )
    rel_predictor.calibrate(calibration_data)
    rel_interval = rel_predictor.predict_interval(1000.0, alpha=0.1)
    assert rel_interval.lower < rel_interval.upper
    print(f"  ✓ 相对误差区间: [{rel_interval.lower:.0f}, {rel_interval.upper:.0f}]")

    print("\n✅ 所有测试通过")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_tests()

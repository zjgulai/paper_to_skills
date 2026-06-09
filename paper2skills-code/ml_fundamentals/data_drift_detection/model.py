"""
Skill-Data-Drift-Detection
基于 arXiv:2601.08928 (DriftGuard, 2026) +
    OpenReview fUsgIfJYZs (Cry Wolf, ICLR 2026 Workshop)
母婴跨境电商 ML 模型生产数据漂移检测工具
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, date
from enum import Enum


class DriftType(Enum):
    NO_DRIFT     = "无漂移"
    SEASONAL     = "季节性漂移（大促豁免）"
    WARNING      = "轻微漂移（监控）"
    ALERT        = "显著漂移（告警）"
    CRITICAL     = "严重漂移（触发重训）"


@dataclass
class FeatureDriftResult:
    feature_name: str
    psi: float
    drift_type: DriftType
    action: str


@dataclass
class ModelDriftReport:
    model_id: str
    check_date: str
    batch_size: int
    feature_results: list[FeatureDriftResult]
    overall_drift_type: DriftType
    drifted_features: list[str]
    recommendation: str
    retrain_roi_estimate: Optional[str] = None
    adwin_alerts: list[str] = field(default_factory=list)


# ── PSI 计算（批量统计检验）────────────────────────────────
def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    min_count: float = 1e-6,
) -> float:
    """
    PSI = Σ (P_actual - P_expected) × ln(P_actual / P_expected)
    基于对称 KL 散度形式，bins 自适应参考分布分位数。
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    bins = np.quantile(ref_clean, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    cur_counts, _ = np.histogram(cur_clean, bins=bins)

    ref_pct = (ref_counts + min_count) / (len(ref_clean) + min_count * len(bins))
    cur_pct = (cur_counts + min_count) / (len(cur_clean) + min_count * len(bins))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def psi_threshold(batch_size: int, is_promo_window: bool = False) -> tuple[float, float]:
    """
    动态 PSI 阈值（基于 ICLR 2026 Cry Wolf 实验结果）。
    batch_size < 200 时误报率高，需上调阈值。
    大促窗口建议临时提高 0.1，避免季节性伪警报。
    返回: (warning_threshold, critical_threshold)
    """
    base_warning  = 0.20
    base_critical = 0.25
    if batch_size < 200:
        base_warning  += 0.05
        base_critical += 0.05
    if is_promo_window:
        base_warning  += 0.10
        base_critical += 0.10
    return base_warning, base_critical


# ── 大促窗口判断 ─────────────────────────────────────────
PROMO_WINDOWS = [
    (6, 1,  6, 30),   # 618 大促
    (11, 1, 11, 30),  # 双11
    (12, 20, 12, 31), # 圣诞
    (1, 1,  1, 15),   # 元旦后余震
]

def is_promo_window(check_date: date) -> bool:
    for m_start, d_start, m_end, d_end in PROMO_WINDOWS:
        start = date(check_date.year, m_start, d_start)
        end   = date(check_date.year, m_end,   d_end)
        if start <= check_date <= end:
            return True
    return False


# ── 特征漂移分类 ─────────────────────────────────────────
def classify_drift(
    psi: float,
    batch_size: int,
    promo_window: bool,
) -> tuple[DriftType, str]:
    warn_thresh, crit_thresh = psi_threshold(batch_size, promo_window)

    if psi < 0.10:
        return DriftType.NO_DRIFT, "无需处理"
    if psi < warn_thresh:
        return DriftType.WARNING, "加密监控频率（日→小时）"
    if promo_window:
        return DriftType.SEASONAL, "大促豁免，记录基线供大促后对比"
    if psi < crit_thresh:
        return DriftType.ALERT, "发出告警，人工确认是否触发重训"
    return DriftType.CRITICAL, "立即触发重训流程，同时启动 SHAP 根因诊断"


# ── ADWIN 轻量实现（实时流检测）──────────────────────────
class ADWINDetector:
    """
    Adaptive Windowing drift detector。
    适用于 MAB reward 分布等实时数据流。
    生产环境推荐使用 `from river.drift import ADWIN`。
    """
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: list[float] = []
        self.drift_detected = False

    def update(self, value: float) -> bool:
        self.window.append(value)
        self.drift_detected = False
        n = len(self.window)
        if n < 8:
            return False
        for split in range(1, n):
            w0 = np.array(self.window[:split])
            w1 = np.array(self.window[split:])
            m0, m1 = w0.mean(), w1.mean()
            n0, n1 = len(w0), len(w1)
            epsilon_cut = np.sqrt(
                np.log(4 * n / self.delta) * (1 / (2 * min(n0, n1)))
            )
            if abs(m0 - m1) >= epsilon_cut:
                self.window = self.window[split:]
                self.drift_detected = True
                return True
        return False


# ── 主检测函数 ─────────────────────────────────────────────
def detect_drift(
    model_id: str,
    reference_data: dict[str, np.ndarray],
    current_data: dict[str, np.ndarray],
    check_date: Optional[date] = None,
    reward_streams: Optional[dict[str, list[float]]] = None,
) -> ModelDriftReport:
    """
    生产 ML 模型双轨漂移检测。

    Args:
        model_id: 模型标识（如 'demand_forecast_tft'）
        reference_data: 基准分布 {feature_name: np.array}（训练时保存）
        current_data: 当前批次数据 {feature_name: np.array}
        check_date: 检测日期（用于大促窗口判断），默认 today
        reward_streams: MAB 场景的 reward 流 {arm_id: [r1, r2, ...]}
    """
    if check_date is None:
        check_date = date.today()

    promo = is_promo_window(check_date)
    batch_size = min(len(v) for v in current_data.values()) if current_data else 0

    feature_results = []
    drifted_features = []

    for feature, ref_vals in reference_data.items():
        if feature not in current_data:
            continue
        cur_vals = current_data[feature]
        psi = compute_psi(np.array(ref_vals), np.array(cur_vals))
        drift_type, action = classify_drift(psi, batch_size, promo)
        feature_results.append(FeatureDriftResult(feature, round(psi, 4), drift_type, action))
        if drift_type in (DriftType.ALERT, DriftType.CRITICAL):
            drifted_features.append(feature)

    adwin_alerts = []
    if reward_streams:
        for arm_id, rewards in reward_streams.items():
            detector = ADWINDetector()
            for r in rewards:
                if detector.update(r):
                    adwin_alerts.append(f"臂 {arm_id}: 第 {len(detector.window)} 步检测到 reward 分布漂移")

    critical_count = sum(1 for r in feature_results if r.drift_type == DriftType.CRITICAL)
    alert_count    = sum(1 for r in feature_results if r.drift_type == DriftType.ALERT)

    if critical_count > 0:
        overall = DriftType.CRITICAL
        recommendation = (
            f"{critical_count} 个特征严重漂移，建议立即重训。"
            f"预计重训成本 2-4h 算力，可避免 MAPE 继续恶化导致的库存损失。"
        )
        roi_est = "重训 ROI 估算：避免 MAPE +16pp × 月销售额 × 库存偏差系数 ≈ 417× 重训成本（DriftGuard 基准）"
    elif alert_count > 0:
        overall = DriftType.ALERT
        recommendation = f"{alert_count} 个特征显著漂移，人工确认后决定是否重训"
        roi_est = None
    elif promo and any(r.drift_type == DriftType.SEASONAL for r in feature_results):
        overall = DriftType.SEASONAL
        recommendation = "检测到大促季节性漂移，已豁免，大促结束后 7 天复检"
        roi_est = None
    elif any(r.drift_type == DriftType.WARNING for r in feature_results):
        overall = DriftType.WARNING
        recommendation = "轻微漂移，提高监控频率，暂不重训"
        roi_est = None
    else:
        overall = DriftType.NO_DRIFT
        recommendation = "分布稳定，下次按计划检测"
        roi_est = None

    return ModelDriftReport(
        model_id=model_id,
        check_date=str(check_date),
        batch_size=batch_size,
        feature_results=feature_results,
        overall_drift_type=overall,
        drifted_features=drifted_features,
        recommendation=recommendation,
        retrain_roi_estimate=roi_est,
        adwin_alerts=adwin_alerts,
    )


# ── 示例：需求预测模型 + MAB 广告素材 ────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    ref = {
        "search_volume":  rng.normal(1000, 150, 500),
        "weekly_sales":   rng.normal(500,   80, 500),
        "competitor_cnt": rng.normal(45,     8, 500),
        "price_index":    rng.normal(1.0,  0.1, 500),
    }

    scenarios = {
        "618大促后（季节性豁免）": {
            "data": {
                "search_volume":  rng.normal(1800, 200, 300),
                "weekly_sales":   rng.normal(900,  100, 300),
                "competitor_cnt": rng.normal(45,     8, 300),
                "price_index":    rng.normal(0.85, 0.1, 300),
            },
            "date": date(2026, 6, 20),
        },
        "618后3个月（真实concept drift）": {
            "data": {
                "search_volume":  rng.normal(650,  120, 300),
                "weekly_sales":   rng.normal(380,   70, 300),
                "competitor_cnt": rng.normal(72,    10, 300),
                "price_index":    rng.normal(0.92, 0.1, 300),
            },
            "date": date(2026, 9, 20),
        },
    }

    mab_rewards = {
        "素材A_婴儿场景": [0.032] * 50 + [0.018] * 50,
        "素材B_妈妈场景": [0.025] * 100,
    }

    print("=" * 65)
    print("母婴 DTC 模型漂移检测报告")
    print("=" * 65)

    for scenario_name, params in scenarios.items():
        report = detect_drift(
            model_id="demand_forecast_tft",
            reference_data=ref,
            current_data=params["data"],
            check_date=params["date"],
            reward_streams=mab_rewards if "concept drift" in scenario_name else None,
        )
        print(f"\n场景: {scenario_name}  |  日期: {report.check_date}")
        print(f"批量大小: {report.batch_size}  |  大促窗口: {'是' if is_promo_window(params['date']) else '否'}")
        print(f"整体状态: {report.overall_drift_type.value}")
        print(f"特征漂移详情:")
        for r in report.feature_results:
            flag = "⚠️" if r.drift_type in (DriftType.ALERT, DriftType.CRITICAL) else "✅"
            print(f"  {flag} {r.feature_name}: PSI={r.psi:.4f} → {r.drift_type.value}")
        if report.adwin_alerts:
            print(f"ADWIN 告警:")
            for alert in report.adwin_alerts:
                print(f"  🔔 {alert}")
        print(f"建议: {report.recommendation}")
        if report.retrain_roi_estimate:
            print(f"重训 ROI: {report.retrain_roi_estimate}")

"""
Skill-Model-Performance-Monitor
基于 DriftGuard 框架 (arXiv:2601.08928) + Champion-Challenger / Shadow Mode 方法论
母婴跨境电商 ML 模型生产性能监控工具
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats
from enum import Enum


class ModelHealth(Enum):
    HEALTHY   = "健康"
    WARNING   = "警告"
    CRITICAL  = "需要重训"
    DEGRADED  = "性能退化"


class DeployDecision(Enum):
    GO      = "切换新版本"
    WAIT    = "继续观察"
    ABORT   = "放弃新版本"


@dataclass
class PerformanceWindow:
    metric_name: str
    values: list[float]
    baseline: float
    window_days: int

    @property
    def current(self) -> float:
        return float(np.mean(self.values[-7:])) if len(self.values) >= 7 else float(np.mean(self.values))

    @property
    def degradation_pct(self) -> float:
        if self.baseline == 0:
            return 0.0
        return (self.baseline - self.current) / abs(self.baseline) * 100


@dataclass
class ShadowComparisonResult:
    champion_metric: float
    challenger_metric: float
    improvement_pct: float
    p_value: float
    significant: bool
    decision: DeployDecision
    rationale: str


@dataclass
class ModelHealthReport:
    model_id: str
    model_type: str
    check_date: str
    health_status: ModelHealth
    metric_windows: list[PerformanceWindow]
    prediction_dist_csi: Optional[float]
    shadow_result: Optional[ShadowComparisonResult]
    recommendation: str
    action_items: list[str] = field(default_factory=list)


# ── CSI 计算（预测分布漂移）──────────────────────────────
def compute_csi(
    baseline_preds: np.ndarray,
    current_preds: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    CSI (Characteristic Stability Index) 检测预测分布漂移。
    与 PSI 形式相同，但作用于模型输出分布而非输入特征分布。
    CSI > 0.2 表明模型输出分布显著改变（可能丧失区分度）。
    """
    bins = np.linspace(0, 1, n_bins + 1)
    eps = 1e-6
    base_hist, _ = np.histogram(np.clip(baseline_preds, 0, 1), bins=bins)
    curr_hist, _ = np.histogram(np.clip(current_preds, 0, 1), bins=bins)
    base_pct = (base_hist + eps) / (base_hist.sum() + eps * n_bins)
    curr_pct = (curr_hist + eps) / (curr_hist.sum() + eps * n_bins)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


# ── 大促期间基线隔离 ──────────────────────────────────────
def get_baseline_excluding_promo(
    metric_history: list[float],
    dates: list[str],
    promo_months: list[int] = None,
) -> float:
    """
    计算排除大促月份的性能基线，防止大促数据污染基线。
    promo_months: 需排除的月份列表，默认排除 6、11、12 月。
    """
    if promo_months is None:
        promo_months = [6, 11, 12]
    clean_metrics = []
    for metric, date_str in zip(metric_history, dates):
        try:
            month = int(date_str.split("-")[1])
            if month not in promo_months:
                clean_metrics.append(metric)
        except (IndexError, ValueError):
            clean_metrics.append(metric)
    return float(np.median(clean_metrics)) if clean_metrics else float(np.median(metric_history))


# ── 性能窗口健康判断 ──────────────────────────────────────
DEGRADATION_THRESHOLDS = {
    "auc":     {"warning": 5.0,  "critical": 10.0, "higher_is_better": True},
    "mape":    {"warning": 30.0, "critical": 50.0, "higher_is_better": False},
    "f1":      {"warning": 5.0,  "critical": 10.0, "higher_is_better": True},
    "coverage":{"warning": 5.0,  "critical": 10.0, "higher_is_better": True},
    "ctr":     {"warning": 20.0, "critical": 40.0, "higher_is_better": True},
}

def assess_window_health(window: PerformanceWindow) -> tuple[ModelHealth, str]:
    metric_lower = window.metric_name.lower()
    thresholds = DEGRADATION_THRESHOLDS.get(metric_lower, {"warning": 10.0, "critical": 20.0, "higher_is_better": True})

    deg = window.degradation_pct
    if not thresholds["higher_is_better"]:
        deg = -deg

    if deg >= thresholds["critical"]:
        return ModelHealth.CRITICAL, f"{window.metric_name} 退化 {abs(deg):.1f}pp，超过重训阈值"
    if deg >= thresholds["warning"]:
        return ModelHealth.WARNING, f"{window.metric_name} 退化 {abs(deg):.1f}pp，进入警告区间"
    return ModelHealth.HEALTHY, f"{window.metric_name} 正常（退化 {abs(deg):.1f}pp）"


# ── Shadow Mode 对比分析 ──────────────────────────────────
def shadow_comparison(
    champion_preds: np.ndarray,
    challenger_preds: np.ndarray,
    actuals: np.ndarray,
    metric: str = "mape",
    alpha: float = 0.05,
) -> ShadowComparisonResult:
    """
    Champion vs Challenger 统计显著性检验。
    使用配对 t 检验（样本量足够时）或 Wilcoxon 符号秩检验（鲁棒版本）。
    """
    def calc_metric(preds, acts, m):
        if m == "mape":
            return float(np.mean(np.abs((acts - preds) / np.maximum(np.abs(acts), 1e-6))) * 100)
        if m == "mae":
            return float(np.mean(np.abs(acts - preds)))
        if m == "auc":
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(acts, preds))
        return float(np.mean(np.abs(acts - preds)))

    champ_score = calc_metric(champion_preds, actuals, metric)
    chall_score = calc_metric(challenger_preds, actuals, metric)

    champ_errors = np.abs(actuals - champion_preds)
    chall_errors = np.abs(actuals - challenger_preds)

    if len(champ_errors) >= 30:
        _, p_value = stats.ttest_rel(champ_errors, chall_errors)
    else:
        _, p_value = stats.wilcoxon(champ_errors, chall_errors, alternative='two-sided')

    higher_is_better = metric in ("auc", "f1", "accuracy")
    if higher_is_better:
        improvement_pct = (chall_score - champ_score) / max(abs(champ_score), 1e-6) * 100
    else:
        improvement_pct = (champ_score - chall_score) / max(abs(champ_score), 1e-6) * 100

    significant = p_value < alpha
    challenger_wins = significant and improvement_pct > 0

    if challenger_wins and improvement_pct > 5:
        decision = DeployDecision.GO
        rationale = f"Challenger 显著优于 Champion（提升 {improvement_pct:.1f}pp，p={p_value:.4f}），建议切换"
    elif significant and improvement_pct < 0:
        decision = DeployDecision.ABORT
        rationale = f"Challenger 显著差于 Champion（退化 {abs(improvement_pct):.1f}pp，p={p_value:.4f}），放弃"
    else:
        decision = DeployDecision.WAIT
        rationale = f"差异不显著（p={p_value:.4f}），继续观察 7 天后再判断"

    return ShadowComparisonResult(
        champion_metric=round(champ_score, 4),
        challenger_metric=round(chall_score, 4),
        improvement_pct=round(improvement_pct, 2),
        p_value=round(p_value, 4),
        significant=significant,
        decision=decision,
        rationale=rationale,
    )


# ── 主监控函数 ────────────────────────────────────────────
def monitor_model(
    model_id: str,
    model_type: str,
    metric_history: dict[str, list[float]],
    metric_dates: list[str],
    baseline_metrics: Optional[dict[str, float]] = None,
    baseline_predictions: Optional[np.ndarray] = None,
    current_predictions: Optional[np.ndarray] = None,
    shadow_data: Optional[dict] = None,
    check_date: Optional[str] = None,
) -> ModelHealthReport:
    """
    生产 ML 模型性能监控主函数。

    Args:
        model_id: 模型标识
        model_type: 'regression' | 'classification' | 'mab'
        metric_history: {metric_name: [历史值列表]} 按时间顺序
        metric_dates: 对应日期列表（用于大促基线隔离）
        baseline_metrics: 训练时的基准指标 {metric_name: value}
        baseline_predictions: 训练期的预测分布（用于 CSI）
        current_predictions: 当前的预测分布
        shadow_data: {'champion_preds': ..., 'challenger_preds': ..., 'actuals': ...}
        check_date: 检测日期字符串
    """
    if check_date is None:
        check_date = str(pd.Timestamp.today().date())

    metric_windows = []
    worst_health = ModelHealth.HEALTHY
    action_items = []

    for metric_name, history in metric_history.items():
        if not history:
            continue
        if baseline_metrics and metric_name in baseline_metrics:
            baseline = baseline_metrics[metric_name]
        else:
            baseline = get_baseline_excluding_promo(history, metric_dates)

        window = PerformanceWindow(
            metric_name=metric_name,
            values=history,
            baseline=baseline,
            window_days=len(history),
        )
        metric_windows.append(window)

        health, msg = assess_window_health(window)
        if health == ModelHealth.CRITICAL:
            worst_health = ModelHealth.CRITICAL
            action_items.append(f"[立即] {msg} → 触发重训流程")
        elif health == ModelHealth.WARNING and worst_health != ModelHealth.CRITICAL:
            worst_health = ModelHealth.WARNING
            action_items.append(f"[关注] {msg} → 加密监控至每日")

    csi = None
    if baseline_predictions is not None and current_predictions is not None:
        csi = round(compute_csi(baseline_predictions, current_predictions), 4)
        if csi > 0.2:
            action_items.append(f"[关注] 预测分布 CSI={csi:.4f} 超阈值，模型可能丧失区分度")
            if worst_health == ModelHealth.HEALTHY:
                worst_health = ModelHealth.WARNING

    shadow_result = None
    if shadow_data:
        metric = "mape" if model_type == "regression" else "auc"
        shadow_result = shadow_comparison(
            champion_preds=np.array(shadow_data["champion_preds"]),
            challenger_preds=np.array(shadow_data["challenger_preds"]),
            actuals=np.array(shadow_data["actuals"]),
            metric=metric,
        )
        action_items.append(f"[Shadow] {shadow_result.rationale}")

    if worst_health == ModelHealth.CRITICAL:
        recommendation = "立即重训，同时检查 Skill-Data-Drift-Detection 报告定位根因特征"
    elif worst_health == ModelHealth.WARNING:
        recommendation = "提高监控频率，准备重训方案，7 天后评估是否触发"
    elif shadow_result and shadow_result.decision == DeployDecision.GO:
        recommendation = "Shadow 验证通过，建议切换新版本"
    else:
        recommendation = "模型健康，下次按计划检测"

    return ModelHealthReport(
        model_id=model_id,
        model_type=model_type,
        check_date=check_date,
        health_status=worst_health,
        metric_windows=metric_windows,
        prediction_dist_csi=csi,
        shadow_result=shadow_result,
        recommendation=recommendation,
        action_items=action_items,
    )


# ── 示例：母婴 DTC 全模型监控 ─────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dates = [f"2026-{m:02d}-01" for m in range(1, 6)]

    print("=" * 65)
    print("母婴 DTC 生产模型健康看板")
    print("=" * 65)

    models = {
        "需求预测 TFT": {
            "model_type": "regression",
            "metric_history": {"mape": [12.1, 13.5, 16.2, 22.8, 28.4]},
            "baseline_metrics": {"mape": 12.0},
        },
        "Uplift 模型": {
            "model_type": "classification",
            "metric_history": {"auc": [0.78, 0.77, 0.76, 0.73, 0.69]},
            "baseline_metrics": {"auc": 0.78},
            "baseline_predictions": rng.beta(2, 5, 500),
            "current_predictions": rng.beta(3, 3, 300),
        },
        "LTV 预测（带 Shadow 验证）": {
            "model_type": "regression",
            "metric_history": {"mape": [18.2, 18.5, 18.1, 18.8, 18.3]},
            "baseline_metrics": {"mape": 18.0},
            "shadow_data": {
                "champion_preds": rng.normal(100, 20, 200),
                "challenger_preds": rng.normal(100, 18, 200),
                "actuals": rng.normal(102, 22, 200),
            },
        },
    }

    for model_name, params in models.items():
        report = monitor_model(
            model_id=model_name,
            model_type=params["model_type"],
            metric_history=params["metric_history"],
            metric_dates=dates,
            baseline_metrics=params.get("baseline_metrics"),
            baseline_predictions=params.get("baseline_predictions"),
            current_predictions=params.get("current_predictions"),
            shadow_data=params.get("shadow_data"),
            check_date="2026-05-25",
        )
        status_icon = {"健康": "✅", "警告": "⚠️", "需要重训": "🔴", "性能退化": "🟠"}.get(
            report.health_status.value, "❓"
        )
        print(f"\n{status_icon} {report.model_id}  |  状态: {report.health_status.value}")
        for w in report.metric_windows:
            print(f"   {w.metric_name}: 基准={w.baseline:.3f} → 当前={w.current:.3f}"
                  f"  退化={w.degradation_pct:+.1f}pp")
        if report.prediction_dist_csi is not None:
            print(f"   预测分布 CSI: {report.prediction_dist_csi:.4f}"
                  f"  {'⚠️ 超阈值' if report.prediction_dist_csi > 0.2 else '✅ 正常'}")
        if report.shadow_result:
            sr = report.shadow_result
            print(f"   Shadow: Champion={sr.champion_metric:.4f} | "
                  f"Challenger={sr.challenger_metric:.4f} | "
                  f"提升={sr.improvement_pct:+.1f}% | 决策={sr.decision.value}")
        print(f"   建议: {report.recommendation}")
        for item in report.action_items:
            print(f"   → {item}")

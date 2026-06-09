"""
Argos Agentic Anomaly Detection — 自主异常检测 Agent
paper2skills-code: 09-DataAgent-LLM | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricPoint:
    timestamp: str
    metric_name: str
    value: float
    expected_range: tuple[float, float]


@dataclass
class AnomalyAlert:
    metric_name: str
    timestamp: str
    value: float
    expected_range: tuple[float, float]
    severity: str       # LOW / MEDIUM / HIGH / CRITICAL
    root_cause: str
    recommended_action: str
    confidence: float


class StatisticalAnomalyDetector:
    """Z-score + IQR 双轨异常检测"""
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect(self, values: list[float]) -> list[bool]:
        if len(values) < 4:
            return [False] * len(values)
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean)**2 for v in values) / max(len(values)-1, 1))
        sorted_v = sorted(values)
        q1 = sorted_v[len(sorted_v)//4]
        q3 = sorted_v[3*len(sorted_v)//4]
        iqr = q3 - q1
        results = []
        for v in values:
            z_anomaly = abs(v - mean) > self.z_threshold * std if std > 0 else False
            iqr_anomaly = v < q1 - self.iqr_multiplier * iqr or v > q3 + self.iqr_multiplier * iqr
            results.append(z_anomaly or iqr_anomaly)
        return results


class ArgosAgent:
    """
    Argos Agentic 异常检测 Agent
    自主监控业务指标并输出结构化预警（含根因假设和建议动作）
    """
    SEVERITY_MAP = {(0.0, 0.3): "LOW", (0.3, 0.6): "MEDIUM",
                    (0.6, 0.85): "HIGH", (0.85, 1.01): "CRITICAL"}

    METRIC_CONTEXT = {
        "daily_revenue":     ("营收异常", "检查广告投入和库存状态"),
        "conversion_rate":   ("转化率下降", "检查 Listing 质量和竞品价格"),
        "return_rate":       ("退货率异常", "核查产品质量和描述准确性"),
        "ad_spend":          ("广告消耗异常", "审查竞价策略和预算上限"),
        "inventory_level":   ("库存异常", "触发补货或清仓策略"),
        "review_rating":     ("评分下滑", "启动 VOC 分析流程"),
    }

    def __init__(self):
        self.detector = StatisticalAnomalyDetector()

    def _severity(self, value: float, expected: tuple[float, float]) -> tuple[str, float]:
        lo, hi = expected
        if lo <= value <= hi:
            return "NORMAL", 0.0
        deviation = max(abs(value - lo), abs(value - hi)) / max(abs(hi - lo), 1e-6)
        score = min(1.0, deviation / 2.0)
        for (s_lo, s_hi), sev in self.SEVERITY_MAP.items():
            if s_lo <= score < s_hi:
                return sev, score
        return "CRITICAL", 1.0

    def monitor(self, metrics: list[MetricPoint]) -> list[AnomalyAlert]:
        by_metric: dict[str, list[MetricPoint]] = {}
        for m in metrics:
            by_metric.setdefault(m.metric_name, []).append(m)

        alerts = []
        for metric_name, points in by_metric.items():
            values = [p.value for p in points]
            anomaly_flags = self.detector.detect(values)
            for i, (point, is_anomaly) in enumerate(zip(points, anomaly_flags)):
                severity, confidence = self._severity(point.value, point.expected_range)
                if severity == "NORMAL" and not is_anomaly:
                    continue
                if is_anomaly and severity == "NORMAL":
                    severity, confidence = "MEDIUM", 0.5
                root_cause, action = self.METRIC_CONTEXT.get(
                    metric_name, ("指标异常", "人工审查"))
                alerts.append(AnomalyAlert(
                    metric_name=metric_name,
                    timestamp=point.timestamp,
                    value=round(point.value, 2),
                    expected_range=point.expected_range,
                    severity=severity,
                    root_cause=root_cause,
                    recommended_action=action,
                    confidence=round(confidence, 3),
                ))
        return sorted(alerts, key=lambda a: a.confidence, reverse=True)


def run_argos_demo():
    import random
    random.seed(42)

    metrics = []
    base_dates = [f"2026-06-{i:02d}" for i in range(1, 8)]

    for d in base_dates:
        metrics += [
            MetricPoint(d, "daily_revenue", random.uniform(8000, 12000), (7000, 15000)),
            MetricPoint(d, "conversion_rate", random.uniform(0.02, 0.05), (0.025, 0.06)),
            MetricPoint(d, "return_rate", random.uniform(0.03, 0.06), (0.01, 0.08)),
        ]
    # 注入异常点
    metrics.append(MetricPoint("2026-06-08", "daily_revenue", 2500, (7000, 15000)))
    metrics.append(MetricPoint("2026-06-08", "return_rate", 0.25, (0.01, 0.08)))

    agent = ArgosAgent()
    alerts = agent.monitor(metrics)

    print("=== Argos 异常检测报告（母婴店铺业务监控）===")
    for a in alerts[:5]:
        print(f"[{a.severity}] {a.metric_name} @ {a.timestamp}")
        print(f"  值: {a.value} | 期望区间: {a.expected_range} | 置信度: {a.confidence:.2f}")
        print(f"  根因假设: {a.root_cause}")
        print(f"  建议动作: {a.recommended_action}")
    print(f"✅ 检测完成，共发现 {len(alerts)} 个异常")


if __name__ == "__main__":
    run_argos_demo()

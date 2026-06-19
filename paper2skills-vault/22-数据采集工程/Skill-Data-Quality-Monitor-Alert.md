---
title: Data Quality Monitor Alert — 多维数据质量监控与异常告警（SPC + KL 散度）
doc_type: knowledge
module: 22-数据采集工程
topic: data-quality-monitor-alert
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Data Quality Monitor Alert

> **领域**：数据采集工程 × 质量治理 | **类型**: 工程基础
> **桥梁**: 22-数据采集工程 ↔ 04-供应链 | **2026年**

---

## ① 算法原理

### 核心思想

数据质量问题是 Skill 决策失效的最主要原因：SP-API 延迟 2 小时未推数据（延迟告警）、某 ASIN 的销量字段突然全为 0（缺失告警）、广告 ROAS 分布从均值 3.2 突变为 8.5（分布漂移告警）。

**三维检测体系**：
1. **延迟检测**：监控数据到达时间，超过阈值触发告警
2. **缺失检测**：统计字段空值率、行数骤降
3. **分布漂移检测**：用 **KL 散度**（Kullback-Leibler Divergence）量化当前分布与历史基准分布的偏离程度；用 **SPC 控制图**（统计过程控制，3σ 规则）检测单指标异常

### 数学直觉

**KL 散度（分布漂移）**：

$$D_{KL}(P \| Q) = \sum_i P(i) \ln \frac{P(i)}{Q(i)}$$

其中 $Q$ 为历史基准分布，$P$ 为当前分布。$D_{KL} > 0.1$ 告警，$> 0.3$ 严重告警。

**SPC 控制图（3σ 规则）**：

$$\text{UCL} = \mu + 3\sigma, \quad \text{LCL} = \mu - 3\sigma$$

当前观测值 $x$ 超出控制线即为异常：$x > \text{UCL}$ 或 $x < \text{LCL}$。

### 关键假设

- 历史数据窗口 ≥ 14 天（建立可靠基准）
- 指标分布近似稳定（季节性需单独处理）
- 告警触发后人工介入或 Agent 自动降级

---

## ② 母婴出海应用案例

**场景 A：SP-API 数据延迟自动告警**

- **业务问题**：SP-API 有时会出现 2-4 小时数据空洞（Amazon 服务端延迟），若 Agent 在此期间按「库存为 0」决策会触发错误补货
- **数据要求**：每批次采集的时间戳 + 期望采集频率配置
- **预期产出**：距上次成功采集超 90 分钟 → Level-1 告警；超 3 小时 → Level-2 告警 + 触发降级（使用上次已知数据）
- **业务价值**：避免因数据空洞导致错误补货 2-3 次/月，每次错误补货成本约 **2 万元**，年化节省 **48-72 万元**

**场景 B：广告 ROAS 分布漂移检测**

- **业务问题**：某次 Amazon 广告结算 bug 导致 ROAS 数据虚高（从正常 3.2 → 异常 12.8），运营误以为广告效果爆发，加大预算，实际亏损
- **数据要求**：过去 30 天的 ROAS 日度分布（历史基准）
- **预期产出**：KL 散度 > 0.3 时触发告警，暂停自动出价调整，等待人工确认
- **业务价值**：避免基于错误数据做广告扩量决策，防止单次事故损失约 **10-20 万元**

---

## ③ 代码模板

```python
"""
Data Quality Monitor Alert
多维数据质量监控：延迟检测 + 缺失检测 + SPC 控制图 + KL 散度分布漂移
依赖：标准库（statistics, math, datetime）
"""

import math
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum


# ─── 告警等级 ─────────────────────────────────────────────────────────────────

class AlertLevel(Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DataQualityAlert:
    metric_name: str
    check_type: str      # latency / missing / distribution_drift / spc_outlier
    level: AlertLevel
    value: float
    threshold: float
    message: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─── SPC 控制图 ───────────────────────────────────────────────────────────────

class SPCControlChart:
    """
    统计过程控制（3σ 规则）
    - fit(): 用历史数据建立控制限
    - check(): 检测新观测值是否超出控制线
    """

    def __init__(self, name: str, sigma_multiplier: float = 3.0):
        self.name = name
        self.sigma = sigma_multiplier
        self.mean: float | None = None
        self.std: float | None = None
        self.ucl: float | None = None
        self.lcl: float | None = None

    def fit(self, historical: list[float]) -> None:
        if len(historical) < 2:
            raise ValueError("需要至少 2 个历史数据点")
        self.mean = statistics.mean(historical)
        self.std = statistics.stdev(historical)
        self.ucl = self.mean + self.sigma * self.std
        self.lcl = self.mean - self.sigma * self.std

    def check(self, value: float) -> DataQualityAlert | None:
        if self.mean is None:
            return None

        if value > self.ucl or value < self.lcl:
            deviation = abs(value - self.mean) / (self.std or 1e-9)
            level = AlertLevel.CRITICAL if deviation > 5 else AlertLevel.WARNING
            return DataQualityAlert(
                metric_name=self.name,
                check_type="spc_outlier",
                level=level,
                value=round(value, 4),
                threshold=self.ucl if value > self.ucl else self.lcl,
                message=(f"{self.name} 超出控制限：{value:.3f} "
                         f"（UCL={self.ucl:.3f}, LCL={self.lcl:.3f}, "
                         f"偏差={deviation:.1f}σ）"),
            )
        return None


# ─── KL 散度分布漂移检测 ──────────────────────────────────────────────────────

class KLDriftDetector:
    """
    基于 KL 散度检测数值型指标的分布漂移
    使用直方图分桶近似连续分布
    """

    def __init__(self, name: str, bins: int = 10,
                 warn_threshold: float = 0.1,
                 critical_threshold: float = 0.3):
        self.name = name
        self.bins = bins
        self.warn_threshold = warn_threshold
        self.critical_threshold = critical_threshold
        self._baseline: list[float] = []
        self._bin_edges: list[float] = []

    def _to_distribution(self, data: list[float]) -> list[float]:
        """数据 → 归一化直方图（概率分布）"""
        if not data or not self._bin_edges:
            return []

        edges = self._bin_edges
        counts = [0] * (len(edges) - 1)
        for x in data:
            for i in range(len(edges) - 1):
                if edges[i] <= x < edges[i + 1]:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1  # 超出最大值归入最后一桶

        total = sum(counts) or 1
        # 加 epsilon 防止 log(0)
        return [(c + 1e-9) / (total + len(counts) * 1e-9) for c in counts]

    def fit_baseline(self, baseline_data: list[float]) -> None:
        """用历史数据建立基准分布"""
        self._baseline = baseline_data
        min_val = min(baseline_data)
        max_val = max(baseline_data)
        step = (max_val - min_val) / self.bins or 1.0
        self._bin_edges = [min_val + i * step for i in range(self.bins + 1)]

    def _kl_divergence(self, p: list[float], q: list[float]) -> float:
        """KL(P || Q)，P 为当前，Q 为基准"""
        kl = 0.0
        for pi, qi in zip(p, q):
            if pi > 0 and qi > 0:
                kl += pi * math.log(pi / qi)
        return kl

    def check(self, current_data: list[float]) -> DataQualityAlert | None:
        if not self._baseline or not self._bin_edges:
            return None

        baseline_dist = self._to_distribution(self._baseline)
        current_dist = self._to_distribution(current_data)

        if not baseline_dist or not current_dist:
            return None

        kl = self._kl_divergence(current_dist, baseline_dist)

        if kl >= self.critical_threshold:
            return DataQualityAlert(
                metric_name=self.name,
                check_type="distribution_drift",
                level=AlertLevel.CRITICAL,
                value=round(kl, 4),
                threshold=self.critical_threshold,
                message=f"{self.name} 分布严重漂移：KL={kl:.4f} ≥ {self.critical_threshold}",
            )
        elif kl >= self.warn_threshold:
            return DataQualityAlert(
                metric_name=self.name,
                check_type="distribution_drift",
                level=AlertLevel.WARNING,
                value=round(kl, 4),
                threshold=self.warn_threshold,
                message=f"{self.name} 分布漂移：KL={kl:.4f} ≥ {self.warn_threshold}",
            )
        return None


# ─── 延迟 & 缺失检测 ──────────────────────────────────────────────────────────

class DataFreshnessChecker:
    """检测数据是否按时到达"""

    def __init__(self, name: str,
                 warn_minutes: float = 60.0,
                 critical_minutes: float = 180.0):
        self.name = name
        self.warn_minutes = warn_minutes
        self.critical_minutes = critical_minutes
        self._last_received: datetime | None = None

    def record_arrival(self) -> None:
        self._last_received = datetime.now(timezone.utc)

    def check(self) -> DataQualityAlert | None:
        if self._last_received is None:
            return DataQualityAlert(
                metric_name=self.name,
                check_type="latency",
                level=AlertLevel.WARNING,
                value=-1.0,
                threshold=self.warn_minutes,
                message=f"{self.name} 从未收到数据",
            )

        elapsed = (datetime.now(timezone.utc) - self._last_received).total_seconds() / 60
        if elapsed >= self.critical_minutes:
            return DataQualityAlert(
                metric_name=self.name,
                check_type="latency",
                level=AlertLevel.CRITICAL,
                value=round(elapsed, 1),
                threshold=self.critical_minutes,
                message=f"{self.name} 数据延迟 {elapsed:.1f} 分钟（严重）",
            )
        elif elapsed >= self.warn_minutes:
            return DataQualityAlert(
                metric_name=self.name,
                check_type="latency",
                level=AlertLevel.WARNING,
                value=round(elapsed, 1),
                threshold=self.warn_minutes,
                message=f"{self.name} 数据延迟 {elapsed:.1f} 分钟",
            )
        return None


class MissingnessChecker:
    """检测字段空值率和行数骤降"""

    def __init__(self, name: str,
                 max_null_rate: float = 0.05,
                 min_row_ratio: float = 0.5):
        self.name = name
        self.max_null_rate = max_null_rate
        self.min_row_ratio = min_row_ratio
        self._baseline_rows: int | None = None

    def set_baseline_rows(self, count: int) -> None:
        self._baseline_rows = count

    def check(self, data: list[Any], field_values: list[Any | None]) -> list[DataQualityAlert]:
        alerts = []
        total = len(field_values)
        if total == 0:
            return alerts

        # 空值率
        null_count = sum(1 for v in field_values if v is None or v == "")
        null_rate = null_count / total
        if null_rate > self.max_null_rate:
            alerts.append(DataQualityAlert(
                metric_name=self.name,
                check_type="missing",
                level=AlertLevel.WARNING if null_rate < 0.2 else AlertLevel.CRITICAL,
                value=round(null_rate, 4),
                threshold=self.max_null_rate,
                message=f"{self.name} 空值率 {null_rate:.1%}（阈值 {self.max_null_rate:.1%}）",
            ))

        # 行数骤降
        if self._baseline_rows and total < self._baseline_rows * self.min_row_ratio:
            ratio = total / self._baseline_rows
            alerts.append(DataQualityAlert(
                metric_name=self.name,
                check_type="missing",
                level=AlertLevel.CRITICAL,
                value=round(ratio, 4),
                threshold=self.min_row_ratio,
                message=f"{self.name} 行数骤降：{total}/{self._baseline_rows}（{ratio:.1%}）",
            ))

        return alerts


# ─── 综合监控器 ───────────────────────────────────────────────────────────────

class DataQualityMonitor:
    def __init__(self):
        self._alerts: list[DataQualityAlert] = []

    def add_alert(self, alert: DataQualityAlert | None) -> None:
        if alert:
            self._alerts.append(alert)

    def add_alerts(self, alerts: list[DataQualityAlert]) -> None:
        self._alerts.extend(alerts)

    def get_alerts(self, level: AlertLevel | None = None) -> list[DataQualityAlert]:
        if level is None:
            return self._alerts
        return [a for a in self._alerts if a.level == level]

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self._alerts),
            "critical": len(self.get_alerts(AlertLevel.CRITICAL)),
            "warning": len(self.get_alerts(AlertLevel.WARNING)),
            "ok": len(self.get_alerts(AlertLevel.OK)),
            "latest": [a.message for a in self._alerts[-3:]],
        }


# ─── 测试用例 ──────────────────────────────────────────────────────────────────

def test_data_quality_monitor_alert():
    import random
    random.seed(42)

    monitor = DataQualityMonitor()

    # 1. SPC 控制图测试
    normal_roas = [3.0, 3.2, 3.1, 2.9, 3.3, 3.2, 3.0, 2.8, 3.1, 3.4,
                   3.2, 3.0, 2.9, 3.1, 3.3, 3.2, 3.0, 2.9, 3.1, 3.2]
    spc = SPCControlChart("daily_roas")
    spc.fit(normal_roas)
    assert spc.ucl > spc.mean > spc.lcl

    # 正常值：无告警
    ok_alert = spc.check(3.1)
    assert ok_alert is None
    print(f"[✓] SPC 正常: UCL={spc.ucl:.3f}, LCL={spc.lcl:.3f}")

    # 异常值：触发告警
    anomaly_alert = spc.check(8.5)   # 广告 bug 导致 ROAS 虚高
    assert anomaly_alert is not None
    assert anomaly_alert.level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
    monitor.add_alert(anomaly_alert)
    print(f"[✓] SPC 告警: {anomaly_alert.message}")

    # 2. KL 散度分布漂移测试
    baseline_sales = [random.gauss(100, 15) for _ in range(200)]
    kl_detector = KLDriftDetector("daily_sales", warn_threshold=0.05, critical_threshold=0.2)
    kl_detector.fit_baseline(baseline_sales)

    # 正常分布：无漂移
    normal_current = [random.gauss(100, 15) for _ in range(50)]
    ok_kl = kl_detector.check(normal_current)
    print(f"[✓] KL 正常分布: alert={ok_kl}")

    # 漂移分布：均值从 100 → 160（异常拉升）
    drifted_current = [random.gauss(160, 15) for _ in range(50)]
    drift_alert = kl_detector.check(drifted_current)
    assert drift_alert is not None
    assert drift_alert.level in (AlertLevel.WARNING, AlertLevel.CRITICAL)
    monitor.add_alert(drift_alert)
    print(f"[✓] KL 漂移检测: {drift_alert.message}")

    # 3. 延迟检测测试
    freshness = DataFreshnessChecker("sp_api_orders",
                                     warn_minutes=60, critical_minutes=180)
    # 从未收到数据
    no_data_alert = freshness.check()
    assert no_data_alert is not None
    assert no_data_alert.level == AlertLevel.WARNING
    monitor.add_alert(no_data_alert)
    print(f"[✓] 延迟检测（无数据）: {no_data_alert.message}")

    # 刚收到数据：无告警
    freshness.record_arrival()
    ok_fresh = freshness.check()
    assert ok_fresh is None
    print("[✓] 延迟检测（刚到达）: OK")

    # 4. 缺失检测测试
    missing_checker = MissingnessChecker("order_price",
                                         max_null_rate=0.05, min_row_ratio=0.5)
    missing_checker.set_baseline_rows(1000)

    # 高空值率
    field_values = [None] * 150 + [29.9] * 850
    missing_alerts = missing_checker.check(list(range(1000)), field_values)
    assert len(missing_alerts) > 0
    monitor.add_alerts(missing_alerts)
    print(f"[✓] 缺失检测: {missing_alerts[0].message}")

    # 行数骤降（1000 → 200）
    field_values_short = [29.9] * 200
    short_alerts = missing_checker.check(list(range(200)), field_values_short)
    # 空值率正常，但行数骤降
    row_drop = [a for a in short_alerts if "骤降" in a.message]
    assert len(row_drop) > 0
    monitor.add_alerts(row_drop)
    print(f"[✓] 行数骤降: {row_drop[0].message}")

    # 5. 综合汇总
    summary = monitor.summary()
    assert summary["total"] > 0
    print(f"\n[汇总] 总告警={summary['total']}, "
          f"严重={summary['critical']}, 警告={summary['warning']}")
    for msg in summary["latest"]:
        print(f"  - {msg}")

    print("\n[✓] Data Quality Monitor Alert 测试通过")


if __name__ == "__main__":
    test_data_quality_monitor_alert()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Data-Drift-Detection]]（数据漂移检测的理论基础）
- **前置（prerequisite）**：[[Skill-Anomaly-Detection-Foundation-Model]]（更复杂的基于 ML 的异常检测）
- **延伸（extends）**：[[Skill-Amazon-SP-API-Data-Pipeline]]（对 SP-API 采集结果做质量监控）
- **可组合（combinable）**：[[Skill-Advertising-API-Unified-Schema]]（对跨平台广告数据做一致性和漂移监控）
- **可组合（combinable）**：[[Skill-Real-Time-Inventory-Event-Stream]]（对库存事件流做完整性和延迟监控）
- **可组合（combinable）**：[[Skill-Agent-Skill-Runtime-Orchestrator]]（数据质量告警触发 Orchestrator 切换降级 Skill）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 场景1：防止 SP-API 数据空洞触发错误补货，每次错误成本约 2 万元，每年 4-6 次 → 节省 **8-12 万元**
  - 场景2：防止广告数据 bug 导致错误扩量，每次损失约 10-20 万元 → 节省 **10-20 万元/年**
  - 场景3：及时发现缺失数据（如某 ASIN 漏采），避免决策盲区，年化减损约 **5 万元**
  - 总计年化 ROI：**约 23-37 万元**
- **实施难度**：⭐⭐☆☆☆（SPC 和 KL 散度算法标准，实现难度低，配置阈值需要业务校准）
- **优先级评分**：⭐⭐⭐⭐⭐（「垃圾进 → 垃圾出」，数据质量是所有 Skill 可信度的底线保障）
- **评估依据**：当前所有 Skill 直接信任输入数据，一旦数据管道故障会导致 Agent 做出完全错误的决策；引入数据质量监控后，Skill 系统具备自我保护能力

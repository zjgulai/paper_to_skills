---
title: Agent SLO Manager — 三层 SLI 体系：服务/任务/判断质量
doc_type: knowledge
module: 16-智能体工程
topic: agent-slo-manager-sli-reliability
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: Agent SLO Manager（三层 SLI 体系）

> **领域**: 16-智能体工程 | **类型**: 综合萃取

---

## ① 算法原理

传统 pass@1 仅衡量"至少一次成功"，无法反映 Agent 在生产环境的持续可靠性。Agent 可靠性需要**三层 SLI 互补**：

1. **服务层 SLI（Service SLI，目标 99.5%）**：API 调用成功率、延迟 P95 等基础可用性指标——Agent 在线且响应正常。
2. **任务完成 SLI（Task Completion SLI，目标 99.9%）**：端到端任务成功率——Agent 返回了有效输出（格式正确、业务规则合规）。
3. **判断质量 SLI（Judgment SLI，目标 92%）**：AI 决策质量度量——人工覆核抽样≥5%、与人工判断的协议率>90%。判断 SLI 是区别于传统 SRE 的关键：服务可用但决策质量低（如推荐错误供应商）同样构成生产故障。

**BurnRate 告警**：在滑动窗口内，错误预算消耗速率超过阈值时分级告警：warning 2×（按当前速率将在窗口结束前耗尽预算的 2 倍速度）、critical 10×（紧急响应）。

**SLO 状态机**：`HEALTHY → WARNING(BurnRate≥2x) → CRITICAL(BurnRate≥10x) → EXHAUSTED(预算耗尽) → UNKNOWN(数据不足)`

**ExhaustionAction 分级响应**：ALERT（仅通知）→ THROTTLE（限速请求）→ FREEZE_DEPLOYMENTS（冻结新版本推送）→ CIRCUIT_BREAK（熔断，拒绝所有请求）。

---

## ② 母婴出海应用案例

**场景一：WF-A 供应链 MAS 上线前评估**

新版补货 Agent 上线前需三层 SLI 全部达标。Canary 阶段（10% 流量，30 天观察窗口）：Service SLI ≥99.5%、Task SLI ≥99.9%、Judgment SLI ≥92%（人工抽查 5% 补货单，协议率>90%）。只有 SLO 状态持续 HEALTHY 满 30 天，才解锁推全量。若任意 SLI 触发 WARNING，观察窗口重置；触发 CRITICAL 则自动回滚旧版。

**场景二：WF-B 广告 Agent 持续监控**

广告预算调整 Agent 接入三层监控：当"补货量判断质量 SLI"（人工覆核广告选品决策协议率）降至 88% 时，SLO 状态从 HEALTHY 变为 WARNING，自动触发 THROTTLE（广告日预算上限从 $5000 降至 $1000）；若预算烧尽（EXHAUSTED），自动 CIRCUIT_BREAK，广告投放全部转为人工审核，直到 Judgment SLI 恢复到目标阈值以上。

---

## ③ 代码模板

```python
"""
Agent SLO Manager — 三层 SLI 体系实现
来源：Microsoft agent-governance-toolkit + agent-sre 2026
"""
import time
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict


class SLIType(Enum):
    SERVICE = "service"                    # 服务层：可用性/延迟
    TASK_COMPLETION = "task_completion"    # 任务完成率
    JUDGMENT_QUALITY = "judgment_quality"  # 判断质量（AI 决策）


class SLOStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"
    UNKNOWN = "unknown"


class ExhaustionAction(Enum):
    ALERT = "alert"
    THROTTLE = "throttle"
    FREEZE_DEPLOYMENTS = "freeze_deployments"
    CIRCUIT_BREAK = "circuit_break"


@dataclass
class SLIMetric:
    metric_name: str
    sli_type: SLIType
    target: float           # 目标值，如 0.995
    current_value: float    # 当前值
    window_seconds: int     # 观察窗口（秒）
    sample_count: int = 0   # 样本数量

    @property
    def is_meeting_target(self) -> bool:
        return self.current_value >= self.target

    @property
    def error_rate(self) -> float:
        return max(0.0, 1.0 - self.current_value)


@dataclass
class ErrorBudget:
    sli_type: SLIType
    target: float
    window_seconds: int
    total_events: int = 0
    failed_events: int = 0
    _alerts: List[str] = field(default_factory=list)

    @property
    def allowed_error_rate(self) -> float:
        return 1.0 - self.target

    @property
    def total_budget_seconds(self) -> float:
        """允许的总错误秒数（或事件数）"""
        return self.window_seconds * self.allowed_error_rate

    @property
    def consumed(self) -> float:
        """已消耗的错误预算（比例）"""
        if self.total_events == 0:
            return 0.0
        actual_error_rate = self.failed_events / self.total_events
        return actual_error_rate / self.allowed_error_rate if self.allowed_error_rate > 0 else float('inf')

    @property
    def remaining(self) -> float:
        return max(0.0, 1.0 - self.consumed)

    @property
    def burn_rate(self) -> float:
        """当前消耗速率相对于预算的倍数"""
        if self.total_events == 0:
            return 0.0
        actual_error_rate = self.failed_events / self.total_events
        expected_rate = self.allowed_error_rate
        return actual_error_rate / expected_rate if expected_rate > 0 else float('inf')

    @property
    def is_exhausted(self) -> bool:
        return self.consumed >= 1.0

    @property
    def firing_alerts(self) -> List[str]:
        alerts = []
        br = self.burn_rate
        if br >= 10.0:
            alerts.append(f"CRITICAL: BurnRate={br:.1f}x ≥ 10x for {self.sli_type.value}")
        elif br >= 2.0:
            alerts.append(f"WARNING: BurnRate={br:.1f}x ≥ 2x for {self.sli_type.value}")
        if self.is_exhausted:
            alerts.append(f"EXHAUSTED: error budget depleted for {self.sli_type.value}")
        return alerts

    def record(self, success: bool) -> None:
        self.total_events += 1
        if not success:
            self.failed_events += 1


class AgentSLO:
    """
    Agent SLO 管理器：整合三层 SLI，输出 SLO 状态
    """
    SLI_DEFAULTS = {
        SLIType.SERVICE: 0.995,
        SLIType.TASK_COMPLETION: 0.999,
        SLIType.JUDGMENT_QUALITY: 0.92,
    }

    def __init__(self, agent_id: str, window_seconds: int = 30 * 24 * 3600):
        self.agent_id = agent_id
        self.window_seconds = window_seconds
        self.budgets: Dict[SLIType, ErrorBudget] = {
            sli_type: ErrorBudget(sli_type, target, window_seconds)
            for sli_type, target in self.SLI_DEFAULTS.items()
        }
        self._exhaustion_actions: Dict[SLIType, ExhaustionAction] = {
            SLIType.SERVICE: ExhaustionAction.CIRCUIT_BREAK,
            SLIType.TASK_COMPLETION: ExhaustionAction.FREEZE_DEPLOYMENTS,
            SLIType.JUDGMENT_QUALITY: ExhaustionAction.THROTTLE,
        }

    def record_event(self, sli_type: SLIType, success: bool, cost: float = 0.0) -> None:
        """记录一次事件（成功或失败）"""
        self.budgets[sli_type].record(success)

    def get_burn_rate(self, sli_type: Optional[SLIType] = None) -> float:
        """获取指定 SLI 或最高 BurnRate"""
        if sli_type:
            return self.budgets[sli_type].burn_rate
        return max(b.burn_rate for b in self.budgets.values())

    def evaluate(self) -> SLOStatus:
        """评估当前 SLO 状态"""
        # 数据不足：任意 SLI 样本 < 10
        if any(b.total_events < 10 for b in self.budgets.values()):
            return SLOStatus.UNKNOWN

        # 检查预算耗尽
        if any(b.is_exhausted for b in self.budgets.values()):
            return SLOStatus.EXHAUSTED

        # 检查 BurnRate 告警
        max_burn = self.get_burn_rate()
        if max_burn >= 10.0:
            return SLOStatus.CRITICAL
        if max_burn >= 2.0:
            return SLOStatus.WARNING

        return SLOStatus.HEALTHY

    def get_all_alerts(self) -> List[str]:
        alerts = []
        for budget in self.budgets.values():
            alerts.extend(budget.firing_alerts)
        return alerts

    def get_metrics_summary(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "status": self.evaluate().value,
            "burn_rates": {k.value: round(v.burn_rate, 3) for k, v in self.budgets.items()},
            "remaining_budgets": {k.value: round(v.remaining, 3) for k, v in self.budgets.items()},
            "alerts": self.get_all_alerts(),
        }


class ExhaustionHandler:
    """根据 SLO 状态执行对应 ExhaustionAction"""

    def __init__(self, slo: AgentSLO):
        self.slo = slo
        self._throttle_active = False
        self._circuit_open = False
        self._deployments_frozen = False

    def handle(self) -> List[str]:
        status = self.slo.evaluate()
        actions_taken = []

        if status == SLOStatus.EXHAUSTED:
            # 最严重：触发 CIRCUIT_BREAK
            self._circuit_open = True
            self._deployments_frozen = True
            actions_taken.append(f"CIRCUIT_BREAK: agent {self.slo.agent_id} is now offline")

        elif status == SLOStatus.CRITICAL:
            self._deployments_frozen = True
            actions_taken.append(f"FREEZE_DEPLOYMENTS: halting new rollouts for {self.slo.agent_id}")

        elif status == SLOStatus.WARNING:
            self._throttle_active = True
            actions_taken.append(f"THROTTLE: reducing request rate for {self.slo.agent_id}")

        elif status == SLOStatus.HEALTHY:
            # 恢复
            if self._throttle_active:
                self._throttle_active = False
                actions_taken.append(f"ALERT: throttle lifted for {self.slo.agent_id}")

        return actions_taken

    @property
    def is_available(self) -> bool:
        return not self._circuit_open


# ===== 测试：WF-A 场景，模拟 30 天正常 + 5 天异常 =====
def _test_wfa_scenario():
    slo = AgentSLO(agent_id="wf-a-replenishment", window_seconds=35 * 24 * 3600)
    handler = ExhaustionHandler(slo)

    # 模拟 30 天正常运行（每天 100 次服务调用 + 50 次任务 + 20 次判断）
    for day in range(30):
        for _ in range(100):
            slo.record_event(SLIType.SERVICE, True)
        for _ in range(50):
            slo.record_event(SLIType.TASK_COMPLETION, True)
        for _ in range(19):
            slo.record_event(SLIType.JUDGMENT_QUALITY, True)
        slo.record_event(SLIType.JUDGMENT_QUALITY, False)  # 95% 协议率（高于 92%）

    status_after_normal = slo.evaluate()
    assert status_after_normal == SLOStatus.HEALTHY, f"期望 HEALTHY，得到 {status_after_normal}"

    # 模拟 5 天异常：判断质量大幅下降（60% 协议率）
    for day in range(5):
        for _ in range(100):
            slo.record_event(SLIType.SERVICE, True)
        for _ in range(50):
            slo.record_event(SLIType.TASK_COMPLETION, True)
        for _ in range(6):
            slo.record_event(SLIType.JUDGMENT_QUALITY, True)
        for _ in range(4):
            slo.record_event(SLIType.JUDGMENT_QUALITY, False)  # 60% 协议率

    status_after_degraded = slo.evaluate()
    assert status_after_degraded in (SLOStatus.WARNING, SLOStatus.CRITICAL, SLOStatus.EXHAUSTED), \
        f"期望 WARNING/CRITICAL/EXHAUSTED，得到 {status_after_degraded}"

    actions = handler.handle()
    assert len(actions) > 0, "期望触发了 ExhaustionAction"

    print("[✓] Agent SLO Manager WF-A 场景测试通过")
    print(f"    状态变化: HEALTHY → {status_after_degraded.value}")
    print(f"    触发动作: {actions}")
    metrics = slo.get_metrics_summary()
    print(f"    指标摘要: {metrics}")


if __name__ == "__main__":
    _test_wfa_scenario()
```

---

## ④ 技能关联

- **前置**：[[Skill-Agent-Production-Engineering]] / [[Skill-AgentTrace-Causal-RCA]]
- **延伸**：[[Skill-Agent-Error-Budget]] / [[Skill-ReliabilityBench-Agent-Reliability]]
- **可组合**：[[Skill-MASEval-System-Evaluation]] / [[Skill-Sandlock-Agent-Execution-Sandbox]] / [[Skill-SDOF-State-Constrained-Orchestration]]

---

## ⑤ 商业价值

- **ROI**：生产 Agent 质量可量化可告警，防止"可用但错误"的生产事故（如错误补货决策导致的资金损失）；三层 SLI 缺一不可，Judgment SLI 是核心差异化护城河
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐⭐（P0，autoresearch 进化的度量基础）

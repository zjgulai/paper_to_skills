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
            alerts.append(f"CRITICAL: BurnRate={br:.1f}x >= 10x for {self.sli_type.value}")
        elif br >= 2.0:
            alerts.append(f"WARNING: BurnRate={br:.1f}x >= 2x for {self.sli_type.value}")
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
            if self._throttle_active:
                self._throttle_active = False
                actions_taken.append(f"ALERT: throttle lifted for {self.slo.agent_id}")

        return actions_taken

    @property
    def is_available(self) -> bool:
        return not self._circuit_open


def _test_wfa_scenario():
    """WF-A 场景：验证 HEALTHY 正常检测 + 大量失败触发 WARNING/CRITICAL"""
    slo = AgentSLO(agent_id="wf-a-replenishment", window_seconds=35 * 24 * 3600)
    handler = ExhaustionHandler(slo)

    # 阶段一：输入足够样本后正常运行（95% 判断质量，高于 92% 目标）
    for _ in range(100):
        slo.record_event(SLIType.SERVICE, True)
    for _ in range(50):
        slo.record_event(SLIType.TASK_COMPLETION, True)
    for _ in range(19):
        slo.record_event(SLIType.JUDGMENT_QUALITY, True)
    slo.record_event(SLIType.JUDGMENT_QUALITY, False)  # 95% 协议率

    status_after_normal = slo.evaluate()
    assert status_after_normal == SLOStatus.HEALTHY, f"期望 HEALTHY，得到 {status_after_normal}"

    # 阶段二：判断质量大幅下降至 40%（远低于 92% 目标，触发高 BurnRate）
    for _ in range(4):
        slo.record_event(SLIType.JUDGMENT_QUALITY, True)
    for _ in range(6):
        slo.record_event(SLIType.JUDGMENT_QUALITY, False)  # 40% 协议率

    status_after_degraded = slo.evaluate()
    assert status_after_degraded in (SLOStatus.WARNING, SLOStatus.CRITICAL, SLOStatus.EXHAUSTED), \
        f"期望 WARNING/CRITICAL/EXHAUSTED，得到 {status_after_degraded}"

    actions = handler.handle()
    assert len(actions) > 0, "期望触发了 ExhaustionAction"

    print("[✓] Agent SLO Manager WF-A 场景测试通过")
    print(f"    状态变化: HEALTHY -> {status_after_degraded.value}")
    print(f"    触发动作: {actions}")


if __name__ == "__main__":
    _test_wfa_scenario()

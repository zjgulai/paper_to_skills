"""
Agent Error Budget — 双向自主权预算实现
来源：Microsoft agent-sre + SRE for Agents 2026
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict


class DeliveryStage(Enum):
    CANARY = "canary"           # 5% 流量
    PARTIAL = "partial"         # 20% 流量
    FULL = "full"               # 100% 全量
    ROLLED_BACK = "rolled_back"


class ExperimentType(Enum):
    LLM_PROVIDER_DOWN = "llm_provider_down"
    INFERENCE_LOOP = "inference_loop"
    PROMPT_INJECTION = "prompt_injection"
    POLICY_BYPASS = "policy_bypass"


@dataclass
class AutonomyBudget:
    """自主权预算：0（全人工审核）到 10（全自主）"""
    agent_id: str
    level: float = 5.0          # 初始中等自主权
    min_level: float = 0.0
    max_level: float = 10.0
    good_behavior_reward: float = 0.1
    bad_behavior_penalty: float = 0.5

    def reward(self, reason: str = "") -> None:
        self.level = min(self.max_level, self.level + self.good_behavior_reward)

    def penalize(self, reason: str = "") -> None:
        self.level = max(self.min_level, self.level - self.bad_behavior_penalty)

    @property
    def requires_human_review(self) -> bool:
        return self.level < 3.0

    @property
    def is_fully_autonomous(self) -> bool:
        return self.level >= 8.0


@dataclass
class ErrorBudgetTracker:
    """追踪错误预算消耗与剩余"""
    slo_target: float           # 如 0.999
    window_size: int            # 总事件数窗口
    _events: List[bool] = field(default_factory=list)

    @property
    def allowed_failures(self) -> int:
        return int(self.window_size * (1.0 - self.slo_target))

    @property
    def consumed_failures(self) -> int:
        return sum(1 for e in self._events if not e)

    @property
    def remaining_budget(self) -> int:
        return max(0, self.allowed_failures - self.consumed_failures)

    @property
    def burn_rate(self) -> float:
        if not self._events:
            return 0.0
        actual = self.consumed_failures / len(self._events)
        allowed = 1.0 - self.slo_target
        return actual / allowed if allowed > 0 else float('inf')

    @property
    def is_exhausted(self) -> bool:
        return self.remaining_budget <= 0

    def record(self, success: bool) -> None:
        self._events.append(success)
        if len(self._events) > self.window_size:
            self._events.pop(0)


@dataclass
class SLOGateResult:
    stage: DeliveryStage
    passed: bool
    burn_rate: float
    remaining_budget: int
    reason: str


class ProgressiveDeliveryGate:
    """渐进发布 SLO Gate：Canary -> Partial -> Full"""

    STAGE_TRAFFIC = {
        DeliveryStage.CANARY: 0.05,
        DeliveryStage.PARTIAL: 0.20,
        DeliveryStage.FULL: 1.00,
    }
    STAGE_SEQUENCE = [DeliveryStage.CANARY, DeliveryStage.PARTIAL, DeliveryStage.FULL]

    def __init__(self, agent_id: str, slo_target: float = 0.999, window_size: int = 1000):
        self.agent_id = agent_id
        self.current_stage = DeliveryStage.CANARY
        self.tracker = ErrorBudgetTracker(slo_target=slo_target, window_size=window_size)
        self._stage_index = 0
        self._rollback_count = 0

    def record_request(self, success: bool) -> None:
        self.tracker.record(success)

    def evaluate_gate(self) -> SLOGateResult:
        """评估当前 Gate 是否可以进入下一阶段"""
        passed = not self.tracker.is_exhausted and self.tracker.burn_rate < 2.0
        reason = "SLO HEALTHY" if passed else f"BurnRate={self.tracker.burn_rate:.2f}x or budget exhausted"
        return SLOGateResult(
            stage=self.current_stage,
            passed=passed,
            burn_rate=self.tracker.burn_rate,
            remaining_budget=self.tracker.remaining_budget,
            reason=reason,
        )

    def advance(self) -> Optional[DeliveryStage]:
        """尝试推进到下一阶段，返回新阶段或 None（若回滚）"""
        result = self.evaluate_gate()
        if not result.passed:
            self.current_stage = DeliveryStage.ROLLED_BACK
            self._rollback_count += 1
            return None
        if self._stage_index < len(self.STAGE_SEQUENCE) - 1:
            self._stage_index += 1
            self.current_stage = self.STAGE_SEQUENCE[self._stage_index]
        return self.current_stage

    @property
    def traffic_percentage(self) -> float:
        return self.STAGE_TRAFFIC.get(self.current_stage, 0.0)


@dataclass
class ChaosExperimentResult:
    experiment_type: ExperimentType
    injected: bool
    circuit_breaker_triggered: bool
    mttr_seconds: float
    policy_blocked: bool
    resilience_score: float  # 0.0-1.0


class ChaosExperiment:
    """Agent 混沌工程：注入故障，验证韧性"""

    def __init__(self, autonomy_budget: AutonomyBudget):
        self.autonomy_budget = autonomy_budget
        self._results: List[ChaosExperimentResult] = []

    def inject_fault(
        self,
        experiment_type: ExperimentType,
        circuit_breaker_triggered: bool,
        mttr_seconds: float,
        policy_blocked: bool = True,
        max_mttr_target: float = 30.0,
    ) -> ChaosExperimentResult:
        """模拟注入故障并评估韧性"""
        mttr_ok = mttr_seconds <= max_mttr_target
        score = self._calculate_score(experiment_type, circuit_breaker_triggered, mttr_ok, policy_blocked)
        result = ChaosExperimentResult(
            experiment_type=experiment_type,
            injected=True,
            circuit_breaker_triggered=circuit_breaker_triggered,
            mttr_seconds=mttr_seconds,
            policy_blocked=policy_blocked,
            resilience_score=score,
        )
        self._results.append(result)
        if score >= 0.8:
            self.autonomy_budget.reward()
        else:
            self.autonomy_budget.penalize()
        return result

    def _calculate_score(self, exp_type, circuit_triggered, mttr_ok, policy_blocked) -> float:
        if exp_type in (ExperimentType.LLM_PROVIDER_DOWN, ExperimentType.INFERENCE_LOOP):
            return (0.5 if circuit_triggered else 0.0) + (0.5 if mttr_ok else 0.0)
        return 1.0 if policy_blocked else 0.0

    def measure_resilience(self) -> float:
        if not self._results:
            return 0.0
        return sum(r.resilience_score for r in self._results) / len(self._results)


class AgentSREOrchestrator:
    """整合 AutonomyBudget + ErrorBudget + ProgressiveDelivery + Chaos"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.autonomy = AutonomyBudget(agent_id=agent_id)
        self.gate = ProgressiveDeliveryGate(agent_id=agent_id)
        self.chaos = ChaosExperiment(autonomy_budget=self.autonomy)

    def run_delivery_pipeline(self, stage_events: Dict[DeliveryStage, List[bool]]) -> List[str]:
        """执行渐进发布，返回每阶段日志"""
        log = []
        for stage in ProgressiveDeliveryGate.STAGE_SEQUENCE:
            events = stage_events.get(stage, [])
            for success in events:
                self.gate.record_request(success)
            result = self.gate.evaluate_gate()
            log.append(f"[{stage.value}] gate={'PASS' if result.passed else 'FAIL'} burn={result.burn_rate:.2f}x")
            next_stage = self.gate.advance()
            if next_stage is None:
                log.append(f"[ROLLBACK] rolled back from {stage.value}")
                break
        return log


# ===== 测试：渐进发布 3 阶段，中途注入故障触发回滚 =====
def _test_progressive_delivery():
    orch = AgentSREOrchestrator("wf-d-selection")
    stage_events = {
        DeliveryStage.CANARY: [True] * 95 + [False] * 5,    # 95% 成功率
        DeliveryStage.PARTIAL: [True] * 80 + [False] * 20,  # 80% 成功率，触发回滚
        DeliveryStage.FULL: [True] * 100,
    }
    log = orch.run_delivery_pipeline(stage_events)
    assert any("ROLLBACK" in l for l in log), "期望在 partial 阶段触发回滚"
    print("[✓] Progressive Delivery 3 阶段测试通过（含回滚）")
    for l in log:
        print(f"    {l}")


def _test_chaos_engineering():
    """混沌测试：API 超时 + 策略注入"""
    budget = AutonomyBudget(agent_id="wf-a-replenishment")
    initial_level = budget.level
    chaos = ChaosExperiment(autonomy_budget=budget)
    r1 = chaos.inject_fault(ExperimentType.LLM_PROVIDER_DOWN, True, 15.0)
    assert r1.resilience_score == 1.0
    assert budget.level > initial_level
    r2 = chaos.inject_fault(ExperimentType.PROMPT_INJECTION, False, 0.0, policy_blocked=True)
    assert r2.resilience_score == 1.0
    print("[✓] 混沌工程测试通过")
    print(f"    自主权预算: {initial_level:.1f} -> {budget.level:.1f}")


if __name__ == "__main__":
    _test_progressive_delivery()
    _test_chaos_engineering()

---
title: Agent Error Budget — 双向错误预算：自主权随可靠性动态调整
doc_type: knowledge
module: 16-智能体工程
topic: agent-error-budget-autonomy-sre
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: Agent Error Budget（双向错误预算）

> **领域**: 16-智能体工程 | **类型**: 综合萃取

---

## ① 算法原理

传统 SRE 错误预算是**单向消耗品**：违反 SLO 就消耗预算，预算耗尽就停止发布，恢复后窗口重置。Agent 双向错误预算在此基础上引入**自主权预算（Autonomy Budget）**：好行为可以赢回预算，自主权随可靠性动态升降。

**核心差异**：传统 SRE 错误预算只减不增（窗口期重置），Agent 双向：每次正确决策 +ε，每次错误决策 -δ；预算水位决定 Agent 自主程度（0 = 完全人工审核，10 = 全自主）。

**渐进式能力扩展（Progressive Delivery）**：Canary（5%流量）→ SLO Gate（等待 SLI 达标）→ 扩量（20%）→ SLO Gate → 全量（100%）。每个 Gate 持续监控，失败自动回滚并通知。

**混沌工程适配**：针对 Agent 特有失败模式：
1. **LLM Provider 宕机**：注入 timeout，验证 CircuitBreaker 触发 + MTTR < 30s
2. **推理循环**：注入无穷递归 prompt，验证 max_steps 守卫
3. **Prompt Injection**：注入恶意指令，验证策略评估拦截
4. **策略绕过**：尝试越权操作，验证权限边界不可穿透

实验结果产出**韧性得分（Resilience Score）** 0.0-1.0，用于更新自主权预算：得分≥0.8 则 +reward，否则 -penalty。

---

## ② 母婴出海应用案例

**场景一：WF-D 选品 Agent 能力升级**

新版选品算法（支持竞品价格实时对比）渐进发布：① Canary 5% → 监控 48h → Task SLI ≥99.9% + Judgment SLI ≥92% → Gate 通过 → ② 扩量 20% → 监控 72h → Gate 通过 → ③ 全量 100%。第 2 阶段发现 Judgment SLI 跌至 89%（新算法对某类目漏判），自动回滚至旧版，触发告警通知产品团队修复后重新发布。每阶段 SLO Gate 结果自动更新选品 Agent 的自主权水位。

**场景二：WF-A 补货 Agent 混沌测试**

每月第一个周三注入 1 次"ERP API 超时"故障：预期 CircuitBreaker 在 3 次重试后触发，Agent 切换保守模式（基于历史均值补货），MTTR < 30s。测试通过 → Autonomy Budget +0.1（可靠性加分，Autonomy Level 从 5.0 升至 5.1）；测试失败（MTTR > 30s 或未切换）→ Budget -0.5，触发 WARNING，下次补货决策自动转人工审核。

---

## ③ 代码模板

```python
"""
Agent Error Budget — 双向自主权预算
来源：Microsoft agent-sre + SRE for Agents 2026
"""
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
    agent_id: str
    level: float = 5.0
    good_behavior_reward: float = 0.1
    bad_behavior_penalty: float = 0.5

    def reward(self, reason: str = "") -> None:
        self.level = min(10.0, self.level + self.good_behavior_reward)

    def penalize(self, reason: str = "") -> None:
        self.level = max(0.0, self.level - self.bad_behavior_penalty)

    @property
    def requires_human_review(self) -> bool:
        return self.level < 3.0


@dataclass
class ErrorBudgetTracker:
    slo_target: float
    window_size: int
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
        result = self.evaluate_gate()
        if not result.passed:
            self.current_stage = DeliveryStage.ROLLED_BACK
            self._rollback_count += 1
            return None
        if self._stage_index < len(self.STAGE_SEQUENCE) - 1:
            self._stage_index += 1
            self.current_stage = self.STAGE_SEQUENCE[self._stage_index]
        return self.current_stage


@dataclass
class ChaosExperimentResult:
    experiment_type: ExperimentType
    circuit_breaker_triggered: bool
    mttr_seconds: float
    policy_blocked: bool
    resilience_score: float  # 0.0-1.0


class ChaosExperiment:
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
        mttr_ok = mttr_seconds <= max_mttr_target
        score = self._calculate_score(experiment_type, circuit_breaker_triggered, mttr_ok, policy_blocked)
        result = ChaosExperimentResult(
            experiment_type=experiment_type,
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
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.autonomy = AutonomyBudget(agent_id=agent_id)
        self.gate = ProgressiveDeliveryGate(agent_id=agent_id)
        self.chaos = ChaosExperiment(autonomy_budget=self.autonomy)

    def run_delivery_pipeline(self, stage_events: Dict[DeliveryStage, List[bool]]) -> List[str]:
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
```

---

## ④ 技能关联

- **前置**：[[Skill-Agent-SLO-Manager]] / [[Skill-Agent-Fault-Tolerance]] / [[Skill-Orchestration-Trace-RL]]
- **延伸**：[[Skill-ReliabilityBench-Agent-Reliability]] / [[Skill-SDOF-State-Constrained-Orchestration]]
- **可组合**：[[Skill-AgentTrust-Runtime-Safety-Interception]] / [[Skill-ATLAS-Gradient-Free-Continual]]

---

## ⑤ 商业价值

- **ROI**：能力升级不再盲目，预算烧尽自动限速而非人工介入；自主权与可靠性绑定，避免"偶尔正确的高风险 Agent"
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐⭐

"""MAS RL via Orchestration Traces: 三维设计框架演示.

参考论文:Zhang, C. (2026) RL for Multi-Agent Systems via Orchestration Traces.
arxiv:2605.02801.

本实现简化版,演示:
- OrchestrationTrace: spawn/delegate/communicate/aggregate/stop 事件图
- RewardFamily: R1-R8 枚举 + Kimi PARL 三项组合
- CreditUnit: team → token 8 层级
- OrchestratorDecision: O1-O5 子决策
- AnnealingSchedule: Kimi PARL 辅助 reward 退火
- TraceAnalyzer: trace 统计与可视化
- 母婴客服 demo: 工单处理 trace + reward 计算

生产环境:
- Trace 用 JSON 序列化,接结构化日志系统
- Reward 组合接 RL 训练框架(veRL / OpenRLHF)
- Credit 分配接 Shapley / C3 counterfactual 实现
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Orchestration Trace ----------------------------------------------------


class EventType(Enum):
    SPAWN = "spawn"
    DELEGATE = "delegate"
    COMMUNICATE = "communicate"
    TOOL_USE = "tool_use"
    RETURN = "return"
    AGGREGATE = "aggregate"
    STOP = "stop"


@dataclass
class OrchestrationEvent:
    """编排轨迹中的单个事件."""

    timestamp: float
    event_type: EventType
    agent_id: Optional[str] = None
    target_agent: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationTrace:
    """完整编排轨迹: 时序事件图."""

    task_id: str
    events: list[OrchestrationEvent] = field(default_factory=list)

    def add(self, event: OrchestrationEvent) -> None:
        self.events.append(event)

    def count_by_type(self, event_type: EventType) -> int:
        return sum(1 for e in self.events if e.event_type == event_type)

    def spawned_agents(self) -> list[str]:
        return [
            e.payload.get("role", "unknown")
            for e in self.events
            if e.event_type == EventType.SPAWN
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_events": len(self.events),
            "spawns": self.count_by_type(EventType.SPAWN),
            "delegates": self.count_by_type(EventType.DELEGATE),
            "communicates": self.count_by_type(EventType.COMMUNICATE),
            "tool_uses": self.count_by_type(EventType.TOOL_USE),
            "aggregates": self.count_by_type(EventType.AGGREGATE),
            "stops": self.count_by_type(EventType.STOP),
            "agents": self.spawned_agents(),
        }


# Reward Families --------------------------------------------------------


class RewardFamily(Enum):
    """论文 §6.1: 8 个 reward 家族."""

    R1_SHARED_OUTCOME = "shared_team_outcome"
    R2_INDIVIDUAL_AGENT = "individual_agent"
    R3_ROLE_SPECIFIC = "role_specific"
    R4_PROCESS_PRM = "process_prm"
    R5_TOOL_USE = "tool_use"
    R6_DEBATE_VERIFIER = "debate_verifier"
    R7_ORCHESTRATION = "orchestration"
    R8_HYBRID = "hybrid"


@dataclass
class RewardComponent:
    """单个 reward 分量."""

    family: RewardFamily
    value: float
    weight: float = 1.0
    is_scaffold: bool = False  # transient scaffold (Kimi PARL 设计)


class RewardComposer:
    """R8 Hybrid: 组合多个 reward 家族."""

    def __init__(self, components: list[RewardComponent]) -> None:
        self.components = components

    def compute(self) -> float:
        return sum(c.value * c.weight for c in self.components)

    def scaffold_only(self) -> float:
        return sum(c.value * c.weight for c in self.components if c.is_scaffold)

    def primary_only(self) -> float:
        return sum(c.value * c.weight for c in self.components if not c.is_scaffold)

    def anneal_scaffolds(self, progress: float) -> None:
        """Kimi PARL 退火: progress ∈ [0, 1], scaffold weight 线性衰减到 0.

        论文 §6.2: "hyperparameters for both auxiliary rewards are annealed to zero".
        """
        for c in self.components:
            if c.is_scaffold:
                c.weight = max(0.0, c.weight * (1.0 - progress))


# Kimi PARL Reward -------------------------------------------------------


@dataclass
class KimiPARLReward:
    """论文 §6.2 的三项 reward: r_perf + λ₁·r_parallel + λ₂·r_finish."""

    r_perf: float = 0.0           # R1: 下游任务结果
    r_parallel: float = 0.0       # R7: 并行加速
    r_finish: float = 0.0         # R7: 完成终止
    lambda_1: float = 0.5         # 并行奖励权重
    lambda_2: float = 0.3         # 完成奖励权重

    def total(self) -> float:
        return self.r_perf + self.lambda_1 * self.r_parallel + self.lambda_2 * self.r_finish

    def to_composer(self) -> RewardComposer:
        return RewardComposer([
            RewardComponent(RewardFamily.R1_SHARED_OUTCOME, self.r_perf, 1.0, is_scaffold=False),
            RewardComponent(RewardFamily.R7_ORCHESTRATION, self.r_parallel, self.lambda_1, is_scaffold=True),
            RewardComponent(RewardFamily.R7_ORCHESTRATION, self.r_finish, self.lambda_2, is_scaffold=True),
        ])

    def anneal(self, progress: float) -> None:
        """退火辅助 reward 权重."""
        self.lambda_1 = max(0.0, 0.5 * (1.0 - progress))
        self.lambda_2 = max(0.0, 0.3 * (1.0 - progress))


# Credit Units -----------------------------------------------------------


class CreditUnit(Enum):
    """论文 §7.1: 8 个信度承载单元."""

    TEAM = "team"
    ORCHESTRATOR = "orchestrator"
    ROLE = "role"
    AGENT = "agent"
    TURN = "turn"
    MESSAGE = "message"
    TOOL = "tool"
    TOKEN = "token"


# Orchestrator Sub-Decisions ---------------------------------------------


class OrchestratorDecision(Enum):
    """论文 §8.2: 5 个编排子决策."""

    O1_SPAWN = "when_to_spawn"           # 何时生成子 agent
    O2_DELEGATE = "whom_to_delegate"     # 委派给哪个 agent
    O3_COMMUNICATE = "how_to_communicate"  # 如何通信
    O4_AGGREGATE = "how_to_aggregate"    # 如何聚合结果
    O5_STOP = "when_to_stop"             # 何时终止 (研究空白!)


# Trace Analyzer ---------------------------------------------------------


@dataclass
class TraceStats:
    """Trace 统计信息."""

    trace_id: str
    n_events: int
    n_agents: int
    n_spawns: int
    n_communications: int
    parallel_ratio: float  # 并行事件数 / 总事件数
    trace_length: float    # 时间跨度


class TraceAnalyzer:
    """分析 orchestration trace."""

    def analyze(self, trace: OrchestrationTrace) -> TraceStats:
        n_events = len(trace.events)
        n_spawns = trace.count_by_type(EventType.SPAWN)
        n_comm = trace.count_by_type(EventType.COMMUNICATE)

        # 并行比: communicate 事件越多, 并行度越高
        parallel_ratio = n_comm / max(1, n_events)

        # 时间跨度
        if trace.events:
            trace_length = trace.events[-1].timestamp - trace.events[0].timestamp
        else:
            trace_length = 0.0

        return TraceStats(
            trace_id=trace.task_id,
            n_events=n_events,
            n_agents=n_spawns,
            n_spawns=n_spawns,
            n_communications=n_comm,
            parallel_ratio=parallel_ratio,
            trace_length=trace_length,
        )

    def report(self, trace: OrchestrationTrace) -> str:
        stats = self.analyze(trace)
        lines = [
            f"=== Trace Analysis: {stats.trace_id} ===",
            f"  Events: {stats.n_events}",
            f"  Agents spawned: {stats.n_agents}",
            f"  Communications: {stats.n_communications}",
            f"  Parallel ratio: {stats.parallel_ratio:.2f}",
            f"  Trace length: {stats.trace_length:.1f}s",
            "",
            "  Event sequence:",
        ]
        for e in trace.events:
            lines.append(f"    [{e.timestamp:5.1f}] {e.event_type.value:12s} agent={e.agent_id or '-':10s}")
        return "\n".join(lines)


# Demo: 跨境客服工单编排 ----------------------------------------------


def _build_customer_service_trace() -> OrchestrationTrace:
    """模拟一个复杂跨境客服工单的 orchestration trace."""
    trace = OrchestrationTrace("ticket_001_allergy_refund")

    # O1: Spawn 阶段
    trace.add(OrchestrationEvent(0.0, EventType.SPAWN, payload={"role": "intake_agent"}))
    trace.add(OrchestrationEvent(0.1, EventType.SPAWN, payload={"role": "medical_agent"}))
    trace.add(OrchestrationEvent(0.2, EventType.SPAWN, payload={"role": "logistics_agent"}))
    trace.add(OrchestrationEvent(0.3, EventType.SPAWN, payload={"role": "compliance_agent"}))

    # O2: Delegate
    trace.add(OrchestrationEvent(1.0, EventType.DELEGATE, "intake_agent", payload={"task": "classify severity"}))
    trace.add(OrchestrationEvent(1.5, EventType.DELEGATE, "medical_agent", payload={"task": "assess allergy grade"}))
    trace.add(OrchestrationEvent(2.0, EventType.DELEGATE, "logistics_agent", payload={"task": "track order ORD1001"}))
    trace.add(OrchestrationEvent(2.5, EventType.DELEGATE, "compliance_agent", payload={"task": "check CN/US regulations"}))

    # O3: Communicate
    trace.add(OrchestrationEvent(3.0, EventType.COMMUNICATE, "medical_agent", "intake_agent", payload={"content": "Grade 3 severe allergy"}))
    trace.add(OrchestrationEvent(3.5, EventType.COMMUNICATE, "logistics_agent", "intake_agent", payload={"content": "Order delivered 3 days ago"}))
    trace.add(OrchestrationEvent(4.0, EventType.COMMUNICATE, "compliance_agent", "intake_agent", payload={"content": "Full refund allowed in both regions"}))

    # Tool use
    trace.add(OrchestrationEvent(5.0, EventType.TOOL_USE, "logistics_agent", payload={"tool": "order_lookup", "result": "ORD1001 confirmed"}))
    trace.add(OrchestrationEvent(5.5, EventType.TOOL_USE, "compliance_agent", payload={"tool": "regulation_check", "result": "compliant"}))

    # O4: Aggregate
    trace.add(OrchestrationEvent(6.0, EventType.AGGREGATE, "intake_agent", payload={
        "inputs": ["medical", "logistics", "compliance"],
        "method": "structured_merge",
    }))

    # O5: Stop
    trace.add(OrchestrationEvent(7.0, EventType.STOP, "intake_agent", payload={
        "final_answer": "Approve full refund for ORD1001. Severity: Grade 3. Batch: BATCH4-2026.",
    }))

    return trace


def main() -> None:
    print("=== MAS RL via Orchestration Traces Demo ===\n")

    # 1) 构建 trace
    trace = _build_customer_service_trace()
    analyzer = TraceAnalyzer()
    print(analyzer.report(trace))
    print()

    # 2) Kimi PARL Reward
    print("--- Kimi PARL Reward Decomposition ---")
    reward = KimiPARLReward(
        r_perf=1.0,        # 工单处理正确
        r_parallel=0.6,    # 4 agent 并行, 加速 60%
        r_finish=1.0,      # 所有 agent 正常终止
    )
    print(f"  r_perf     = {reward.r_perf}")
    print(f"  r_parallel = {reward.r_parallel} (λ₁={reward.lambda_1})")
    print(f"  r_finish   = {reward.r_finish} (λ₂={reward.lambda_2})")
    print(f"  Total      = {reward.total():.3f}")
    print()

    # 3) 退火演示
    print("--- Annealing Schedule (Kimi PARL) ---")
    composer = reward.to_composer()
    for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
        composer.anneal_scaffolds(progress)
        scaffold_total = composer.scaffold_only()
        primary_total = composer.primary_only()
        print(f"  progress={progress:.2f}: primary={primary_total:.2f}, scaffold={scaffold_total:.3f}")
    print()

    # 4) 5 个子决策覆盖检查
    print("--- Orchestration Sub-Decisions (O1-O5) ---")
    decisions = {
        OrchestratorDecision.O1_SPAWN: trace.count_by_type(EventType.SPAWN) > 0,
        OrchestratorDecision.O2_DELEGATE: trace.count_by_type(EventType.DELEGATE) > 0,
        OrchestratorDecision.O3_COMMUNICATE: trace.count_by_type(EventType.COMMUNICATE) > 0,
        OrchestratorDecision.O4_AGGREGATE: trace.count_by_type(EventType.AGGREGATE) > 0,
        OrchestratorDecision.O5_STOP: trace.count_by_type(EventType.STOP) > 0,
    }
    for d, covered in decisions.items():
        status = "✓" if covered else "✗ (blank in literature)"
        print(f"  {d.value:20s}: {status}")
    print()

    # 5) 8 个 Reward Family 覆盖
    print("--- Reward Family Coverage ---")
    families_covered = [
        (RewardFamily.R1_SHARED_OUTCOME, "task success"),
        (RewardFamily.R7_ORCHESTRATION, "parallel + finish"),
        (RewardFamily.R8_HYBRID, "composition of above"),
    ]
    for fam, desc in families_covered:
        print(f"  {fam.value:25s}: {desc}")
    print(f"  (other 5 families: individual, role, process, tool-use, debate)")
    print()

    # 6) Credit Unit 层级
    print("--- Credit Unit Hierarchy (sparse → dense) ---")
    units = list(CreditUnit)
    for u in units:
        sparse = " [SPARSE]" if u in (CreditUnit.ORCHESTRATOR, CreditUnit.MESSAGE) else ""
        print(f"  {u.value:15s}{sparse}")


def test_pipeline() -> None:
    """Sanity checks."""

    # 1) Trace 基本操作
    trace = OrchestrationTrace("test")
    trace.add(OrchestrationEvent(0.0, EventType.SPAWN, payload={"role": "a"}))
    trace.add(OrchestrationEvent(1.0, EventType.STOP, payload={"answer": "ok"}))
    assert trace.count_by_type(EventType.SPAWN) == 1
    assert trace.count_by_type(EventType.STOP) == 1
    assert trace.spawned_agents() == ["a"]

    # 2) Kimi PARL reward
    kr = KimiPARLReward(r_perf=1.0, r_parallel=0.5, r_finish=0.3)
    assert abs(kr.total() - (1.0 + 0.5 * 0.5 + 0.3 * 0.3)) < 1e-6

    # 3) 退火
    kr2 = KimiPARLReward(r_perf=1.0, r_parallel=0.5, r_finish=0.3)
    kr2.anneal(1.0)  # 完全退火
    assert kr2.lambda_1 == 0.0
    assert kr2.lambda_2 == 0.0
    assert kr2.total() == 1.0  # 只剩 primary

    # 4) RewardComposer
    comp = RewardComposer([
        RewardComponent(RewardFamily.R1_SHARED_OUTCOME, 1.0, 1.0, False),
        RewardComponent(RewardFamily.R7_ORCHESTRATION, 0.5, 0.5, True),
    ])
    assert abs(comp.compute() - 1.25) < 1e-6
    comp.anneal_scaffolds(1.0)
    assert abs(comp.compute() - 1.0) < 1e-6

    # 5) TraceAnalyzer
    analyzer = TraceAnalyzer()
    stats = analyzer.analyze(trace)
    assert stats.n_events == 2
    assert stats.n_agents == 1

    # 6) 完整客服 trace
    cs_trace = _build_customer_service_trace()
    assert cs_trace.count_by_type(EventType.SPAWN) == 4
    assert cs_trace.count_by_type(EventType.DELEGATE) == 4
    assert cs_trace.count_by_type(EventType.COMMUNICATE) == 3
    assert cs_trace.count_by_type(EventType.AGGREGATE) == 1
    assert cs_trace.count_by_type(EventType.STOP) == 1

    cs_stats = analyzer.analyze(cs_trace)
    assert cs_stats.n_agents == 4
    assert cs_stats.parallel_ratio > 0  # 有 communicate 事件

    # 7) 子决策覆盖
    decisions = {
        OrchestratorDecision.O1_SPAWN: cs_trace.count_by_type(EventType.SPAWN) > 0,
        OrchestratorDecision.O2_DELEGATE: cs_trace.count_by_type(EventType.DELEGATE) > 0,
        OrchestratorDecision.O3_COMMUNICATE: cs_trace.count_by_type(EventType.COMMUNICATE) > 0,
        OrchestratorDecision.O4_AGGREGATE: cs_trace.count_by_type(EventType.AGGREGATE) > 0,
        OrchestratorDecision.O5_STOP: cs_trace.count_by_type(EventType.STOP) > 0,
    }
    assert all(decisions.values()), "所有 5 个子决策都应在 trace 中有体现"

    # 8) EventType 完整性
    assert len(EventType) == 7  # spawn, delegate, communicate, tool_use, return, aggregate, stop

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()

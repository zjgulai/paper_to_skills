"""Focus: 仿生粘菌主动上下文剪枝 Agent 架构.

参考论文:Verma, N. (2026) Focus: An Agent-Centric Approach to Active Context
Pruning Inspired by Slime Mold. arxiv:2601.07190.

本实现简化版:
- 2 个 primitive tools: start_focus / complete_focus
- Message store + KnowledgeBlock 持久化
- System reminder injection (每 N 次 tool call)
- 母婴客服 demo 验证 sawtooth pattern

生产环境:
- Tools 注册到 Claude / GPT 的 tool calling API
- Message store 用 SQLite / Redis (跨会话需要)
- Token 计数用 tiktoken / anthropic tokenizer
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Message store ----------------------------------------------------------


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_REMINDER = "system_reminder"


@dataclass
class ContextMessage:
    """一条会话消息. 每条带 phase_id 以便批量删除."""

    message_id: int
    role: MessageRole
    content: str
    phase_id: Optional[int] = None  # None = 在某个 focus phase 之外

    def estimate_tokens(self) -> int:
        """简化 token 估算:1 token ≈ 4 字符."""
        return max(1, len(self.content) // 4)


@dataclass
class FocusPhase:
    """一个 explore-compress 段."""

    phase_id: int
    topic: str
    start_message_id: int
    end_message_id: Optional[int] = None  # complete 时填
    summary: Optional[str] = None
    closed: bool = False


@dataclass
class KnowledgeEntry:
    """Knowledge block 中的一条持久学习."""

    topic: str
    summary: str
    phase_id: int


# Focus Agent ------------------------------------------------------------


@dataclass
class FocusAgent:
    """实现论文 Section III-B 的 Focus Loop."""

    messages: list[ContextMessage] = field(default_factory=list)
    knowledge: list[KnowledgeEntry] = field(default_factory=list)
    phases: list[FocusPhase] = field(default_factory=list)
    current_phase: Optional[FocusPhase] = None
    _next_message_id: int = 0
    _next_phase_id: int = 0
    total_dropped_messages: int = 0
    total_dropped_tokens: int = 0

    def _next_msg_id(self) -> int:
        self._next_message_id += 1
        return self._next_message_id

    def _next_phs_id(self) -> int:
        self._next_phase_id += 1
        return self._next_phase_id

    def append_message(self, role: MessageRole, content: str) -> ContextMessage:
        msg = ContextMessage(
            message_id=self._next_msg_id(),
            role=role,
            content=content,
            phase_id=self.current_phase.phase_id if self.current_phase else None,
        )
        self.messages.append(msg)
        return msg

    def start_focus(self, topic: str) -> FocusPhase:
        """打 checkpoint: 标记某个 sub-task 开始."""
        if self.current_phase and not self.current_phase.closed:
            # 防御:没 close 前一个就 start 新的(论文要求严格嵌套,这里简化为强制 close)
            self.complete_focus(summary=f"[auto-closed] {self.current_phase.topic} 未显式收尾")

        phase = FocusPhase(
            phase_id=self._next_phs_id(),
            topic=topic,
            start_message_id=self._next_message_id + 1,  # 下一条消息就属于这个 phase
        )
        self.phases.append(phase)
        self.current_phase = phase
        # 显式记录 start
        self.append_message(MessageRole.ASSISTANT, f"[START FOCUS] {topic}")
        return phase

    def complete_focus(self, summary: str) -> int:
        """收尾: 把 summary 写到 Knowledge block, 删除 phase 内消息.

        Returns:
            被删除的 token 数 (估算)
        """
        if not self.current_phase:
            return 0

        phase = self.current_phase
        # 标记结束
        end_msg = self.append_message(MessageRole.ASSISTANT, f"[COMPLETE FOCUS] {phase.topic}")
        phase.end_message_id = end_msg.message_id
        phase.summary = summary
        phase.closed = True

        # 推 Knowledge entry
        self.knowledge.append(KnowledgeEntry(
            topic=phase.topic,
            summary=summary,
            phase_id=phase.phase_id,
        ))

        # 删除 phase 内所有非 start/complete 标记的消息
        dropped_count = 0
        dropped_tokens = 0
        keep: list[ContextMessage] = []
        for m in self.messages:
            # 保留 phase 外消息
            if m.phase_id != phase.phase_id:
                keep.append(m)
                continue
            # phase 内只保留 start / complete 标记
            if m.content.startswith("[START FOCUS]") or m.content.startswith("[COMPLETE FOCUS]"):
                keep.append(m)
                continue
            dropped_count += 1
            dropped_tokens += m.estimate_tokens()

        self.messages = keep
        self.total_dropped_messages += dropped_count
        self.total_dropped_tokens += dropped_tokens
        self.current_phase = None
        return dropped_tokens

    def current_context_tokens(self) -> int:
        """估算当前活跃 context 的 token 数 (含 knowledge block)."""
        msgs_tok = sum(m.estimate_tokens() for m in self.messages)
        kb_tok = sum(max(1, len(e.summary) // 4) for e in self.knowledge)
        return msgs_tok + kb_tok

    def context_view(self) -> str:
        """渲染当前可见 context (knowledge block + messages)."""
        parts = ["=== KNOWLEDGE BLOCK ==="]
        for e in self.knowledge:
            parts.append(f"  [Phase {e.phase_id}] {e.topic}: {e.summary}")
        parts.append("=== MESSAGES ===")
        for m in self.messages:
            phase_marker = f" (phase {m.phase_id})" if m.phase_id else ""
            parts.append(f"  [{m.message_id}] {m.role.value}{phase_marker}: {m.content[:80]}")
        return "\n".join(parts)


# Aggressive prompting & system reminder ---------------------------------


AGGRESSIVE_PROMPT_TEMPLATE = """You are a Focus-enabled agent. Use tools as much as possible, ideally more than 100 times.

CONTEXT MANAGEMENT (CRITICAL):
- ALWAYS call start_focus(topic) before ANY new investigation or sub-task
- ALWAYS call complete_focus(summary) after 10-15 tool calls
- Structure your work into 4-6 focus phases: explore → understand → implement → verify
- When you receive a [SYSTEM REMINDER], immediately call complete_focus to compress

The system will REMIND you every 15 tool calls if you forget to compress.
"""


class FocusOrchestrator:
    """外部 ReAct loop + system reminder injection.

    实现论文 Section IV-B Aggressive Compression Prompting.
    """

    REMINDER_INTERVAL = 15  # 论文 IV-B

    def __init__(self, agent: FocusAgent) -> None:
        self.agent = agent
        self._tool_call_count = 0
        self._last_reminder_count = 0

    def tool_call(self, tool_name: str, tool_output: str) -> None:
        """模拟一次工具调用 + 结果. 触发 reminder 检查."""
        self.agent.append_message(MessageRole.TOOL_CALL, f"tool: {tool_name}")
        self.agent.append_message(MessageRole.TOOL_RESULT, tool_output)
        self._tool_call_count += 1

        # 周期性 reminder (论文 IV-B)
        if (
            self.agent.current_phase
            and self._tool_call_count - self._last_reminder_count >= self.REMINDER_INTERVAL
        ):
            self.agent.append_message(
                MessageRole.SYSTEM_REMINDER,
                "REMINDER: You should call complete_focus to compress your context",
            )
            self._last_reminder_count = self._tool_call_count

    def total_tool_calls(self) -> int:
        return self._tool_call_count


# Demo: 跨境客服过敏退货长会话 ----------------------------------------------


def _customer_service_workflow(agent: FocusAgent, orch: FocusOrchestrator) -> None:
    """模拟一个跨境客服长会话, 演示 sawtooth 模式."""

    # ---- 阶段 1: 客户身份与订单 ----
    agent.start_focus("查询客户订单 ORD1001 + 客户身份")
    for tool_name, output in [
        ("order_lookup", "ORD1001: 状态=已发货, 商品=纸尿裤 size M x 4 包, BATCH4-2026"),
        ("customer_lookup", "VIP 银卡, 历史 8 单 0 退货, 注册 18 个月"),
        ("batch_check", "BATCH4-2026: 生产日期 2026-02-15, 检测合格, 无召回"),
        ("logistics_check", "已签收 3 天, 签收地 加州 90210"),
    ]:
        orch.tool_call(tool_name, output)
    agent.complete_focus("ORD1001 状态=已签收, 客户=VIP 银卡 0 退货历史, BATCH4 无召回")

    # ---- 阶段 2: 过敏症状评估 ----
    agent.start_focus("评估过敏症状严重程度")
    for tool_name, output in [
        ("symptom_classifier", "客户描述: 重度红疹, 伴随哭闹拒食"),
        ("severity_check", "红疹分级=Grade 3 (重度), 需立即停用"),
        ("doctor_consult_check", "建议: 全额退款 + 召回 BATCH4 + 寄送过敏测试盒"),
        ("similar_case_lookup", "近 30 天 BATCH4 客诉 3 单, 均为过敏, 已退款"),
        ("allergen_lookup", "纸尿裤潜在过敏源: 香精+橡胶圈"),
    ]:
        orch.tool_call(tool_name, output)
    agent.complete_focus("症状 Grade 3 重度, BATCH4 近 30 天累计 4 单过敏 (含本单), 建议全额退款+召回")

    # ---- 阶段 3: 合规与执行 ----
    agent.start_focus("执行退款 + 召回流程")
    for tool_name, output in [
        ("compliance_check_US", "美国: 重度过敏需 24h 内退款 + 提供医疗咨询"),
        ("compliance_check_CN", "中国: 母婴产品全额退款无门槛"),
        ("refund_workflow_init", "退款单 REF20260516-001 已创建, 金额 $89.99"),
        ("recall_workflow_init", "BATCH4 召回工单 RCL-BATCH4-2026 已创建"),
        ("notify_customer", "已发送中英文双语通知 + 致歉信"),
        ("notify_qa_team", "QA 团队已收到 BATCH4 召回紧急通知"),
    ]:
        orch.tool_call(tool_name, output)
    agent.complete_focus("已执行: 退款 REF20260516-001 ($89.99) + 召回 RCL-BATCH4-2026 + 客户/QA 双向通知")


def main() -> None:
    print("=== Focus Agent Demo:跨境客服过敏退货长会话 (Sawtooth Pattern) ===\n")

    # Baseline: 不使用 Focus, append-only
    baseline_agent = FocusAgent()
    baseline_orch = FocusOrchestrator(baseline_agent)
    # 模拟 baseline 处理同样会话, 但 start_focus / complete_focus 都不调用 (空操作)
    # 我们手动 append 所有消息, 不删任何东西
    for tool_name, output in [
        ("order_lookup", "ORD1001: 状态=已发货, 商品=纸尿裤 size M x 4 包, BATCH4-2026"),
        ("customer_lookup", "VIP 银卡, 历史 8 单 0 退货, 注册 18 个月"),
        ("batch_check", "BATCH4-2026: 生产日期 2026-02-15, 检测合格, 无召回"),
        ("logistics_check", "已签收 3 天, 签收地 加州 90210"),
        ("symptom_classifier", "客户描述: 重度红疹, 伴随哭闹拒食"),
        ("severity_check", "红疹分级=Grade 3 (重度), 需立即停用"),
        ("doctor_consult_check", "建议: 全额退款 + 召回 BATCH4 + 寄送过敏测试盒"),
        ("similar_case_lookup", "近 30 天 BATCH4 客诉 3 单, 均为过敏, 已退款"),
        ("allergen_lookup", "纸尿裤潜在过敏源: 香精+橡胶圈"),
        ("compliance_check_US", "美国: 重度过敏需 24h 内退款 + 提供医疗咨询"),
        ("compliance_check_CN", "中国: 母婴产品全额退款无门槛"),
        ("refund_workflow_init", "退款单 REF20260516-001 已创建, 金额 $89.99"),
        ("recall_workflow_init", "BATCH4 召回工单 RCL-BATCH4-2026 已创建"),
        ("notify_customer", "已发送中英文双语通知 + 致歉信"),
        ("notify_qa_team", "QA 团队已收到 BATCH4 召回紧急通知"),
    ]:
        baseline_orch.tool_call(tool_name, output)

    baseline_tokens = baseline_agent.current_context_tokens()
    print(f"--- Baseline (无 Focus) ---")
    print(f"工具调用次数: {baseline_orch.total_tool_calls()}")
    print(f"Context token (估算): {baseline_tokens}")
    print(f"消息数: {len(baseline_agent.messages)}")
    print(f"Knowledge block 条目: {len(baseline_agent.knowledge)}\n")

    # Focus: 主动压缩
    focus_agent = FocusAgent()
    focus_orch = FocusOrchestrator(focus_agent)
    _customer_service_workflow(focus_agent, focus_orch)

    focus_tokens = focus_agent.current_context_tokens()
    saved_tokens = baseline_tokens - focus_tokens
    saved_pct = saved_tokens / max(1, baseline_tokens) * 100

    print(f"--- Focus (3 个 phase 主动压缩) ---")
    print(f"工具调用次数: {focus_orch.total_tool_calls()}")
    print(f"Context token (估算): {focus_tokens}")
    print(f"消息数 (留存): {len(focus_agent.messages)}")
    print(f"Knowledge block 条目: {len(focus_agent.knowledge)}")
    print(f"已删除消息: {focus_agent.total_dropped_messages}")
    print(f"已删除 token: {focus_agent.total_dropped_tokens}\n")

    print(f"--- 对比 ---")
    print(f"Token 节省: {saved_tokens} ({saved_pct:.1f}%)")
    print(f"消息数变化: {len(baseline_agent.messages)} → {len(focus_agent.messages)}")

    print(f"\n--- Focus Agent 最终 context view ---")
    print(focus_agent.context_view())


def test_pipeline() -> None:
    """Sanity checks."""
    agent = FocusAgent()
    orch = FocusOrchestrator(agent)

    # 1) 初始状态
    assert agent.current_context_tokens() == 0
    assert agent.current_phase is None
    assert orch.total_tool_calls() == 0

    # 2) start_focus 建立 phase
    p1 = agent.start_focus("test topic")
    assert agent.current_phase is p1
    assert p1.phase_id == 1
    assert not p1.closed

    # 3) 工具调用累积消息
    for i in range(3):
        orch.tool_call(f"tool_{i}", f"long output " * 20)
    assert orch.total_tool_calls() == 3
    pre_drop_tokens = agent.current_context_tokens()
    assert pre_drop_tokens > 50, f"应累积 token, got {pre_drop_tokens}"

    # 4) complete_focus 删除消息, 写 Knowledge
    dropped = agent.complete_focus("summary of test phase")
    assert dropped > 0, f"应删除若干 token, got {dropped}"
    assert agent.current_phase is None
    assert p1.closed
    assert len(agent.knowledge) == 1
    assert agent.knowledge[0].summary == "summary of test phase"

    # 5) 删除后 context 显著缩小 (sawtooth)
    post_drop_tokens = agent.current_context_tokens()
    assert post_drop_tokens < pre_drop_tokens, (
        f"complete_focus 后 token 应下降, got {pre_drop_tokens} → {post_drop_tokens}"
    )

    # 6) Knowledge block 持久化, 后续 start_focus 不删它
    agent.start_focus("phase 2")
    assert len(agent.knowledge) == 1, "Knowledge block 不应被删"
    orch.tool_call("another_tool", "more output")
    agent.complete_focus("phase 2 summary")
    assert len(agent.knowledge) == 2

    # 7) System reminder 在 15 次工具调用后注入
    reminder_agent = FocusAgent()
    reminder_orch = FocusOrchestrator(reminder_agent)
    reminder_agent.start_focus("long phase")
    for i in range(16):
        reminder_orch.tool_call(f"tool_{i}", "out")
    # 在 phase 内 + 调用次数 >= 15 → 应有 reminder
    has_reminder = any(
        m.role == MessageRole.SYSTEM_REMINDER for m in reminder_agent.messages
    )
    assert has_reminder, "应在 15 次工具调用后注入 system reminder"

    # 8) 嵌套 start_focus (论文未明确, 简化版自动 close)
    nest_agent = FocusAgent()
    nest_orch = FocusOrchestrator(nest_agent)
    nest_agent.start_focus("outer")
    nest_orch.tool_call("t1", "x")
    nest_agent.start_focus("inner")  # 应自动 close outer
    assert nest_agent.phases[0].closed, "outer 应被自动 close"
    assert nest_agent.current_phase is not None
    assert nest_agent.current_phase.topic == "inner"

    # 9) Baseline vs Focus token 节省
    base_agent = FocusAgent()
    base_orch = FocusOrchestrator(base_agent)
    for i in range(10):
        base_orch.tool_call(f"t_{i}", "long output " * 50)
    base_tokens = base_agent.current_context_tokens()

    focus_agent = FocusAgent()
    focus_orch = FocusOrchestrator(focus_agent)
    focus_agent.start_focus("phase")
    for i in range(10):
        focus_orch.tool_call(f"t_{i}", "long output " * 50)
    focus_agent.complete_focus("short summary")
    focus_tokens = focus_agent.current_context_tokens()

    assert focus_tokens < base_tokens, (
        f"Focus 应比 baseline 节省 token, got {base_tokens} vs {focus_tokens}"
    )
    saved_pct = (base_tokens - focus_tokens) / base_tokens
    assert saved_pct > 0.5, f"应节省 >50% token, got {saved_pct:.1%}"

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()

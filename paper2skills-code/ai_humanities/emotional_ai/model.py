"""
Emotional AI Customer Care — 情感感知客服：高压场景的同理心 AI
paper2skills-code: 11-AI人文 | 母婴出海跨境电商

纯 Python 标准库实现（无外部依赖）
Python 3.14 兼容
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# 枚举：情绪状态
# ──────────────────────────────────────────────

class EmotionState(Enum):
    """客户情绪状态分级（从低压到高压）"""
    CALM = "calm"              # 平静：正常咨询
    ANXIOUS = "anxious"        # 焦虑：担忧但可控
    FRUSTRATED = "frustrated"  # 沮丧：问题未解决
    ANGRY = "angry"            # 愤怒：强烈不满
    FRIGHTENED = "frightened"  # 恐惧：涉及安全威胁（最高优先级）


class EscalationLevel(Enum):
    """客服升级级别"""
    NORMAL = "normal"          # 正常 AI 处理
    ACCELERATED = "accelerated"  # 加速响应（优先队列）
    HUMAN_REQUIRED = "human"   # 立即转人工


class ResponseStyle(Enum):
    """响应风格"""
    INFORMATIONAL = "informational"  # 信息提供型
    EMPATHETIC = "empathetic"        # 同理心型
    URGENT = "urgent"                # 紧急处理型
    SAFETY = "safety"                # 安全优先型


# ──────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────

@dataclass
class EmotionDetectionResult:
    """情绪识别结果"""
    state: EmotionState
    intensity: float           # 0.0 ~ 1.0，情绪强度
    triggered_signals: list[str] = field(default_factory=list)
    severity_context: Optional[str] = None  # 场景上下文（如"安全召回"）


@dataclass
class EmpathyResponse:
    """同理心响应结果"""
    style: ResponseStyle
    escalation: EscalationLevel
    opening_phrase: str        # 开场白（情绪匹配）
    action_message: str        # 行动说明
    closing_note: str          # 结束语
    full_response: str         # 完整响应文本
    should_escalate: bool      # 是否需要升级


@dataclass
class AgentHandlingResult:
    """情感 AI 全流程处理结果"""
    emotion: EmotionDetectionResult
    response: EmpathyResponse
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# 情绪识别器
# ──────────────────────────────────────────────

class EmotionDetector:
    """
    基于关键词和句式的多维度情绪识别。

    识别策略：
    1. 关键词匹配（按情绪类型分组）
    2. 句式特征（感叹号密度/大写/重复标点）
    3. 安全关键词优先（召回/危险/受伤等立即触发 FRIGHTENED）
    """

    # 各情绪对应的关键词模式（中英文）
    EMOTION_PATTERNS: dict[EmotionState, list[str]] = {
        EmotionState.FRIGHTENED: [
            r"召回|recall",
            r"危险|danger|hazard",
            r"受伤|injured|hurt",
            r"中毒|poison",
            r"窒息|choke|choking",
            r"急诊|emergency room|ER",
            r"宝宝不动|baby not moving",
            r"宝宝出事|something happened to baby",
            r"安全问题|safety issue|safety concern",
        ],
        EmotionState.ANGRY: [
            r"愤怒|furious|outraged",
            r"骗人|liar|scam|fraud",
            r"太差了|terrible|awful",
            r"投诉|complaint|complain",
            r"曝光|expose|report you",
            r"绝对不买|never buying",
            r"退款|refund",
            r"律师|lawyer|sue|legal action",
            r"差评|bad review|1 star",
        ],
        EmotionState.FRUSTRATED: [
            r"为什么|why",
            r"还没收到|not received|not arrived",
            r"等了很久|waited so long|waiting forever",
            r"一直没解决|still not resolved",
            r"催了好几次|followed up multiple times",
            r"不满意|dissatisfied|unhappy",
            r"太慢了|too slow",
        ],
        EmotionState.ANXIOUS: [
            r"担心|worried|concerned",
            r"不确定|not sure|unsure",
            r"会不会|will it|is it safe",
            r"过期|expired",
            r"正常吗|is it normal",
            r"宝宝哭|baby crying|baby keeps crying",
            r"有没有问题|any issue|any problem",
        ],
    }

    # 句式强度信号
    INTENSITY_BOOSTERS = [
        r"！{2,}",    # 连续感叹号
        r"\?{2,}",    # 连续问号
        r"[！!]{3,}",
        r"[A-Z]{5,}", # 全大写单词
    ]

    def __init__(self) -> None:
        self._compiled: dict[EmotionState, list[re.Pattern]] = {
            state: [re.compile(p, re.IGNORECASE) for p in patterns]
            for state, patterns in self.EMOTION_PATTERNS.items()
        }
        self._intensity_re = [re.compile(p) for p in self.INTENSITY_BOOSTERS]

    def detect(self, text: str) -> EmotionDetectionResult:
        """
        识别用户消息的情绪状态。

        优先级：FRIGHTENED > ANGRY > FRUSTRATED > ANXIOUS > CALM
        """
        signals: dict[EmotionState, list[str]] = {s: [] for s in EmotionState}

        # 逐情绪类别匹配
        for state, patterns in self._compiled.items():
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    signals[state].append(m.group())

        # 按优先级判定主情绪
        priority_order = [
            EmotionState.FRIGHTENED,
            EmotionState.ANGRY,
            EmotionState.FRUSTRATED,
            EmotionState.ANXIOUS,
        ]

        detected_state = EmotionState.CALM
        triggered_signals: list[str] = []

        for state in priority_order:
            if signals[state]:
                detected_state = state
                triggered_signals = signals[state]
                break

        # 计算情绪强度（基础分 + 句式加成）
        base_intensity = {
            EmotionState.CALM: 0.1,
            EmotionState.ANXIOUS: 0.35,
            EmotionState.FRUSTRATED: 0.55,
            EmotionState.ANGRY: 0.75,
            EmotionState.FRIGHTENED: 0.90,
        }[detected_state]

        boost = sum(0.05 for p in self._intensity_re if p.search(text))
        intensity = min(base_intensity + boost, 1.0)

        # 安全场景上下文
        severity_context = None
        if detected_state == EmotionState.FRIGHTENED:
            if re.search(r"召回|recall", text, re.IGNORECASE):
                severity_context = "product_recall"
            elif re.search(r"受伤|injured|chok", text, re.IGNORECASE):
                severity_context = "physical_harm"
            else:
                severity_context = "safety_concern"

        return EmotionDetectionResult(
            state=detected_state,
            intensity=round(intensity, 2),
            triggered_signals=triggered_signals,
            severity_context=severity_context,
        )


# ──────────────────────────────────────────────
# 同理心响应生成器
# ──────────────────────────────────────────────

class EmpathyResponseGenerator:
    """
    根据情绪状态 × 场景生成同理心响应。

    沟通风格矩阵：
    ┌─────────────┬──────────────┬────────────────┬──────────────────┐
    │  情绪       │  升级级别    │  响应风格      │  开场白基调     │
    ├─────────────┼──────────────┼────────────────┼──────────────────┤
    │  CALM       │  NORMAL      │  INFORMATIONAL │  友好信息提供   │
    │  ANXIOUS    │  ACCELERATED │  EMPATHETIC    │  先安慰后解答   │
    │  FRUSTRATED │  ACCELERATED │  EMPATHETIC    │  道歉+立即行动  │
    │  ANGRY      │  HUMAN       │  URGENT        │  道歉+快速解决  │
    │  FRIGHTENED │  HUMAN       │  SAFETY        │  安全第一+即办  │
    └─────────────┴──────────────┴────────────────┴──────────────────┘
    """

    _OPENINGS: dict[EmotionState, str] = {
        EmotionState.CALM: "您好！感谢您联系我们，",
        EmotionState.ANXIOUS: "您好，我完全理解您的担心，请放心，我们马上来帮您解答——",
        EmotionState.FRUSTRATED: "非常抱歉给您带来了不便！我立刻为您优先处理这个问题：",
        EmotionState.ANGRY: "非常抱歉！您的感受完全可以理解，我现在把您的问题列为紧急优先级，",
        EmotionState.FRIGHTENED: "【紧急处理】我们完全理解您现在的紧张，宝宝的安全是第一位的！",
    }

    _CLOSINGS: dict[EmotionState, str] = {
        EmotionState.CALM: "如还有其他问题，随时告诉我们！",
        EmotionState.ANXIOUS: "有任何其他顾虑都可以随时联系我们，我们会一直在这里。",
        EmotionState.FRUSTRATED: "再次对您的等待深表歉意，我们一定尽快解决！",
        EmotionState.ANGRY: "我们会用行动证明对您的重视，请给我们一个弥补的机会。",
        EmotionState.FRIGHTENED: "专员会在5分钟内联系您，请保持电话畅通。宝宝平安最重要！",
    }

    _ESCALATION_MAP: dict[EmotionState, EscalationLevel] = {
        EmotionState.CALM: EscalationLevel.NORMAL,
        EmotionState.ANXIOUS: EscalationLevel.ACCELERATED,
        EmotionState.FRUSTRATED: EscalationLevel.ACCELERATED,
        EmotionState.ANGRY: EscalationLevel.HUMAN_REQUIRED,
        EmotionState.FRIGHTENED: EscalationLevel.HUMAN_REQUIRED,
    }

    _STYLE_MAP: dict[EmotionState, ResponseStyle] = {
        EmotionState.CALM: ResponseStyle.INFORMATIONAL,
        EmotionState.ANXIOUS: ResponseStyle.EMPATHETIC,
        EmotionState.FRUSTRATED: ResponseStyle.EMPATHETIC,
        EmotionState.ANGRY: ResponseStyle.URGENT,
        EmotionState.FRIGHTENED: ResponseStyle.SAFETY,
    }

    def generate(
        self,
        emotion: EmotionDetectionResult,
        issue_summary: str = "",
        agent_name: str = "WF-C 智能客服",
    ) -> EmpathyResponse:
        """
        生成同理心响应。

        Args:
            emotion: 情绪识别结果
            issue_summary: 用一句话描述客户问题（用于 action_message 填充）
            agent_name: 客服名称
        """
        state = emotion.state
        escalation = self._ESCALATION_MAP[state]
        style = self._STYLE_MAP[state]
        opening = self._OPENINGS[state]
        closing = self._CLOSINGS[state]

        # 生成行动说明
        if escalation == EscalationLevel.HUMAN_REQUIRED:
            if state == EmotionState.FRIGHTENED:
                action = (
                    "我们已将您的案例标记为【安全紧急】，"
                    "专业客服专员将在5分钟内与您直接联系。"
                    "同时，请停止使用该产品并保存好相关证据。"
                )
            else:
                action = (
                    f"您的问题已升级为最高优先级，"
                    f"专属客服专员将立即介入处理{('：' + issue_summary) if issue_summary else ''}。"
                )
        elif escalation == EscalationLevel.ACCELERATED:
            action = (
                f"我们已将您的问题加入优先处理队列{('：' + issue_summary) if issue_summary else ''}，"
                f"预计响应时间缩短至正常的50%。"
            )
        else:
            action = (
                f"以下是关于您问题{('「' + issue_summary + '」') if issue_summary else ''}的解答："
            )

        full_response = f"{opening}{action} {closing}"

        return EmpathyResponse(
            style=style,
            escalation=escalation,
            opening_phrase=opening,
            action_message=action,
            closing_note=closing,
            full_response=full_response,
            should_escalate=escalation != EscalationLevel.NORMAL,
        )


# ──────────────────────────────────────────────
# 情感 AI 全流程代理
# ──────────────────────────────────────────────

class EmotionalAIAgent:
    """
    情感感知客服代理：感知 → 判断 → 响应完整流程。

    适用于 WF-C 客服分级场景：
    - 自动识别客户情绪强度
    - 平静 → 正常 AI 处理
    - 焦虑 → 加速响应
    - 愤怒/恐惧 → 立即升级人工
    """

    def __init__(self, agent_name: str = "WF-C 智能客服") -> None:
        self._detector = EmotionDetector()
        self._generator = EmpathyResponseGenerator()
        self._agent_name = agent_name

    def handle(
        self,
        customer_message: str,
        issue_summary: str = "",
    ) -> AgentHandlingResult:
        """
        处理客户消息，返回情绪识别 + 响应生成完整结果。

        Args:
            customer_message: 客户的原始消息
            issue_summary: 可选，问题摘要（用于响应个性化）
        """
        emotion = self._detector.detect(customer_message)
        response = self._generator.generate(
            emotion=emotion,
            issue_summary=issue_summary,
            agent_name=self._agent_name,
        )

        return AgentHandlingResult(
            emotion=emotion,
            response=response,
            metadata={
                "agent": self._agent_name,
                "customer_message_length": len(customer_message),
                "escalation_required": response.should_escalate,
            },
        )


# ──────────────────────────────────────────────
# 测试：3个场景
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 60)
    print("Emotional AI Customer Care — 情感感知客服测试")
    print("=" * 60)

    agent = EmotionalAIAgent(agent_name="WF-C 智能客服")

    # 场景 1：正常咨询（CALM）
    print("\n[场景 1] 正常产品咨询")
    result = agent.handle(
        customer_message="你好，请问你们的奶粉适合几个月的宝宝？",
        issue_summary="产品适用月龄咨询",
    )
    assert result.emotion.state == EmotionState.CALM, f"应识别为 CALM，实际: {result.emotion.state}"
    assert result.response.escalation == EscalationLevel.NORMAL, "正常咨询不应升级"
    print(f"  ✓ 情绪={result.emotion.state.value} | 强度={result.emotion.intensity}")
    print(f"  ✓ 升级级别={result.response.escalation.value} | 风格={result.response.style.value}")
    print(f"  ✓ 响应预览: {result.response.full_response[:60]}...")

    # 场景 2：退款投诉（FRUSTRATED → ACCELERATED）
    print("\n[场景 2] 退款投诉（沮丧）")
    result = agent.handle(
        customer_message="我等了两周还没收到货！催了好几次都没回复！退款！太慢了！",
        issue_summary="物流延误退款申请",
    )
    expected_states = {EmotionState.FRUSTRATED, EmotionState.ANGRY}
    assert result.emotion.state in expected_states, (
        f"应识别为 FRUSTRATED 或 ANGRY，实际: {result.emotion.state}"
    )
    assert result.response.should_escalate, "投诉场景应升级处理"
    print(f"  ✓ 情绪={result.emotion.state.value} | 强度={result.emotion.intensity}")
    print(f"  ✓ 升级级别={result.response.escalation.value}")
    print(f"  ✓ 触发信号: {result.emotion.triggered_signals}")
    print(f"  ✓ 响应预览: {result.response.full_response[:80]}...")

    # 场景 3：安全担忧（FRIGHTENED → HUMAN REQUIRED）
    print("\n[场景 3] 奶粉召回安全担忧（恐惧）")
    result = agent.handle(
        customer_message="刚看到新闻说你们的奶粉有召回！我宝宝刚喝了这款！这有安全问题！！！",
        issue_summary="产品召回安全确认",
    )
    assert result.emotion.state == EmotionState.FRIGHTENED, (
        f"应识别为 FRIGHTENED，实际: {result.emotion.state}"
    )
    assert result.response.escalation == EscalationLevel.HUMAN_REQUIRED, "恐惧场景必须立即转人工"
    assert result.emotion.severity_context == "product_recall", (
        f"应识别召回上下文，实际: {result.emotion.severity_context}"
    )
    print(f"  ✓ 情绪={result.emotion.state.value} | 强度={result.emotion.intensity}")
    print(f"  ✓ 升级级别={result.response.escalation.value} ← 立即转人工！")
    print(f"  ✓ 场景上下文={result.emotion.severity_context}")
    print(f"  ✓ 触发信号: {result.emotion.triggered_signals}")
    print(f"  ✓ 响应预览: {result.response.full_response[:100]}...")

    print("\n" + "=" * 60)
    print("[✓] 所有场景验证通过 — Emotional AI Customer Care")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()

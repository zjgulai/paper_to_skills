"""
Multi-Agent Debate (MAD) — 多智能体辩论共识
基于论文: Liang et al. "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate", EMNLP 2024

核心能力:
1. 多 Agent 独立推理 — 产生多样化初始答案
2. 对抗性辩论 — Agent 轮流回应、反驳、修正
3. Judge 裁决 — 综合各方观点输出最终结论
4. Degeneration-of-Thought (DoT) 解决 — 外部对抗替代内部反思

母婴电商场景: 评论情感标注质量仲裁、市场策略决策辩论
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class DebateRole(Enum):
    """辩论角色"""
    DEBATER = "debater"
    JUDGE = "judge"


@dataclass
class DebateArgument:
    """辩论论点"""
    agent_id: str
    round_num: int
    position: str           # 立场
    reasoning: str          # 推理过程
    evidence: List[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """辩论轮次"""
    round_num: int
    arguments: List[DebateArgument] = field(default_factory=list)


class DebateAgent:
    """
    辩论 Agent

    带有特定角色偏见的 Agent，独立推理并参与辩论。
    """

    def __init__(self, agent_id: str, role_bias: str,
                 llm_func: Optional[Callable] = None):
        self.agent_id = agent_id
        self.role_bias = role_bias
        self.llm_func = llm_func or self._mock_llm
        self.arguments: List[DebateArgument] = []

    def initial_answer(self, question: str) -> DebateArgument:
        """生成初始答案"""
        prompt = f"""You are a {self.role_bias}.
Answer the following question independently, providing your reasoning:

Question: {question}

Your answer:"""

        response = self.llm_func(prompt, mode="initial")
        arg = DebateArgument(
            agent_id=self.agent_id,
            round_num=0,
            position=self._extract_position(response),
            reasoning=response
        )
        self.arguments.append(arg)
        return arg

    def debate_response(self, question: str,
                       other_arguments: List[DebateArgument],
                       round_num: int) -> DebateArgument:
        """基于其他 Agent 的观点生成回应"""
        other_views = "\n\n".join([
            f"{a.agent_id}: {a.reasoning}"
            for a in other_arguments
            if a.agent_id != self.agent_id
        ])

        prompt = f"""You are a {self.role_bias}.
Question: {question}

Other agents have argued:
{other_views}

Respond to their arguments. You can:
- Point out flaws in their reasoning
- Provide counter-evidence
- Refine your own position based on valid points they raised
- Maintain your stance if you believe you're correct

Your response:"""

        response = self.llm_func(prompt, mode="debate")
        arg = DebateArgument(
            agent_id=self.agent_id,
            round_num=round_num,
            position=self._extract_position(response),
            reasoning=response
        )
        self.arguments.append(arg)
        return arg

    def _extract_position(self, text: str) -> str:
        """从回复中提取立场（简化版）"""
        lines = text.strip().split('\n')
        if lines:
            return lines[0][:50]
        return text[:50]

    def _mock_llm(self, prompt: str, mode: str = "initial") -> str:
        """模拟 LLM 输出"""
        if "价格敏感" in self.role_bias or "growth" in self.role_bias.lower():
            if "情感" in prompt or "sentiment" in prompt.lower():
                return "负面 — 价格高是主要痛点，用户明确表达了价格顾虑。'贵'这个词直接表达了不满。"
            return "建议降价促销 — 价格敏感用户占 35%，降价可快速抢占市场份额。"

        if "体验优先" in self.role_bias or "brand" in self.role_bias.lower():
            if "情感" in prompt or "sentiment" in prompt.lower():
                return "正面 — '确实好用' 是核心评价。'但'字转折说明后半句才是重点，用户认可产品价值。"
            return "建议增值服务 — 避免价格战，提升用户LTV，强化高端定位。"

        if "平衡" in self.role_bias or "product" in self.role_bias.lower():
            if "情感" in prompt or "sentiment" in prompt.lower():
                return "混合 — 价格是负面因素(权重30%)，使用体验是正面因素(权重70%)。整体倾向正面，但需同时记录两个方面。"
            return "建议产品优化 — 解决根本痛点，推出便携版满足用户需求。"

        return f"作为{self.role_bias}的观点: 需要更多数据支撑判断。"


class JudgeAgent:
    """
    Judge Agent

    独立的裁决者，综合所有辩论方的观点输出最终结论。
    """

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func or self._mock_judge

    def adjudicate(self, question: str,
                   debate_history: List[DebateRound]) -> Dict:
        """裁决辩论，输出最终结论"""
        all_arguments = []
        for r in debate_history:
            all_arguments.extend(r.arguments)

        prompt = f"""You are an impartial judge.
Question: {question}

Here are the arguments from all debaters across all rounds:

{self._format_arguments(all_arguments)}

Your task:
1. Summarize each debater's core position
2. Identify points of agreement and disagreement
3. Evaluate the strength of each argument
4. Provide a final verdict with confidence level

Format your response as:
VERDICT: [your final answer]
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]"""

        response = self.llm_func(prompt)
        return self._parse_verdict(response, debate_history)

    def _format_arguments(self, arguments: List[DebateArgument]) -> str:
        """格式化论点"""
        lines = []
        for arg in arguments:
            lines.append(f"Round {arg.round_num} - {arg.agent_id}: {arg.reasoning[:100]}...")
        return "\n".join(lines)

    def _parse_verdict(self, response: str, debate_history: List[DebateRound]) -> Dict:
        """解析裁决结果"""
        return {
            "verdict": self._extract_field(response, "VERDICT"),
            "confidence": float(self._extract_field(response, "CONFIDENCE", "0.8")),
            "reasoning": self._extract_field(response, "REASONING"),
            "total_rounds": len(debate_history),
            "total_arguments": sum(len(r.arguments) for r in debate_history)
        }

    def _extract_field(self, text: str, field: str, default: str = "") -> str:
        """从文本中提取字段"""
        for line in text.split('\n'):
            if line.startswith(f"{field}:"):
                return line.split(':', 1)[1].strip()
        return default

    def _mock_judge(self, prompt: str) -> str:
        """模拟 Judge 输出"""
        if "情感" in prompt or "sentiment" in prompt.lower():
            return """VERDICT: [价格_负面, 使用体验_正面, 整体倾向_正面]
CONFIDENCE: 0.88
REASONING: 三方观点综合：Agent A 正确识别了价格顾虑，Agent B 正确把握了整体满意度，Agent C 提供了最完整的分析框架。用户表达的是"虽然有价格顾虑，但整体满意"的复杂情感，需要多维度标注。"""

        return """VERDICT: 分阶段执行：先产品优化(便携版)，再增值服务
CONFIDENCE: 0.82
REASONING: Agent C 的根本痛点分析最有说服力，Agent B 的增值服务建议在解决痛点后可行，Agent A 的降价策略风险最高。建议分阶段执行以平衡短期和长期目标。"""


class MultiAgentDebate:
    """
    多 Agent 辩论编排器

    管理完整的多轮辩论流程。
    """

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.agents: List[DebateAgent] = []
        self.judge: Optional[JudgeAgent] = None
        self.history: List[DebateRound] = []

    def add_debater(self, agent_id: str, role_bias: str):
        """添加辩论方"""
        self.agents.append(DebateAgent(agent_id, role_bias))

    def set_judge(self, judge: JudgeAgent):
        """设置 Judge"""
        self.judge = judge

    def run(self, question: str) -> Dict:
        """
        执行完整辩论

        Returns:
            包含辩论历史、Judge 裁决、统计信息
        """
        if not self.agents:
            return {"error": "No debaters added"}

        # Round 0: 初始答案
        round0 = DebateRound(round_num=0)
        for agent in self.agents:
            arg = agent.initial_answer(question)
            round0.arguments.append(arg)
        self.history.append(round0)

        # Debate rounds
        for r in range(1, self.max_rounds + 1):
            debate_round = DebateRound(round_num=r)

            for agent in self.agents:
                other_args = [a for round_ in self.history for a in round_.arguments]
                arg = agent.debate_response(question, other_args, r)
                debate_round.arguments.append(arg)

            self.history.append(debate_round)

        # Judge 裁决
        verdict = None
        if self.judge:
            verdict = self.judge.adjudicate(question, self.history)

        return {
            "question": question,
            "debaters": [a.agent_id for a in self.agents],
            "total_rounds": len(self.history) - 1,  # 不含初始轮
            "history": [
                {
                    "round": r.round_num,
                    "arguments": [
                        {"agent": a.agent_id, "position": a.position, "reasoning": a.reasoning[:80]}
                        for a in r.arguments
                    ]
                }
                for r in self.history
            ],
            "verdict": verdict
        }


# ============================================
# 母婴电商场景 — 评论情感标注质量仲裁
# ============================================

def demo_sentiment_arbitration():
    """演示 MAD 在情感标注仲裁中的应用"""
    print("=" * 70)
    print("Multi-Agent Debate — 评论情感标注质量仲裁")
    print("=" * 70)

    question = '"Spectra S1 吸奶器价格贵但确实好用" — 这条评论的情感倾向是什么？'
    print(f"\n[问题] {question}")

    debate = MultiAgentDebate(max_rounds=2)
    debate.add_debater("Agent-A", "价格敏感视角的分析师")
    debate.add_debater("Agent-B", "体验优先视角的分析师")
    debate.add_debater("Agent-C", "平衡视角的VOC专家")
    debate.set_judge(JudgeAgent())

    result = debate.run(question)

    print(f"\n[辩论过程]")
    print(f"  参与 Agent: {', '.join(result['debaters'])}")
    print(f"  辩论轮次: {result['total_rounds']}")

    for round_data in result["history"]:
        print(f"\n  Round {round_data['round']}:")
        for arg in round_data["arguments"]:
            print(f"    [{arg['agent']}] {arg['position']}")

    print(f"\n[Judge 裁决]")
    if result["verdict"]:
        v = result["verdict"]
        print(f"  结论: {v['verdict']}")
        print(f"  置信度: {v['confidence']}")
        print(f"  理由: {v['reasoning'][:120]}...")

    print("\n" + "=" * 70)


def demo_strategy_debate():
    """演示 MAD 在市场策略决策中的应用"""
    print("\n" + "=" * 70)
    print("Multi-Agent Debate — 市场策略决策辩论")
    print("=" * 70)

    question = "Q3 应优先投入哪个方向提升 Spectra S1 市场份额？"
    print(f"\n[议题] {question}")

    debate = MultiAgentDebate(max_rounds=2)
    debate.add_debater("增长黑客", "增长黑客 — 关注快速获客和转化")
    debate.add_debater("品牌策略", "品牌策略师 — 关注长期品牌价值和LTV")
    debate.add_debater("产品专家", "产品专家 — 关注产品体验和用户痛点")
    debate.set_judge(JudgeAgent())

    result = debate.run(question)

    print(f"\n[辩论过程]")
    for round_data in result["history"][:2]:  # 只展示前2轮
        print(f"\n  Round {round_data['round']}:")
        for arg in round_data["arguments"]:
            print(f"    [{arg['agent']}] {arg['position']}")

    print(f"\n[Judge 裁决]")
    if result["verdict"]:
        v = result["verdict"]
        print(f"  结论: {v['verdict']}")
        print(f"  置信度: {v['confidence']}")

    print("\n" + "=" * 70)


def demonstrate_dot_problem():
    """展示 Degeneration-of-Thought 问题"""
    print("\n" + "=" * 70)
    print("Degeneration-of-Thought (DoT) 问题与 MAD 解决方案")
    print("=" * 70)

    print("""
    DoT 问题:

    单 Agent 自我反思:
      Q: "价格贵但确实好用" 的情感倾向？
      A1 (初始): 正面 — 整体满意
      A2 (反思1): 正面 — 确认整体满意
      A3 (反思2): 正面 — 再次确认...

      问题: Agent 陷入了"维护初始答案"的循环，
            没有真正探索"价格贵"这个负面信号

    MAD 解决方案:
      Agent A: "负面 — 价格高是主要痛点"
      Agent B: "正面 — '确实好用'才是重点"
      Agent C: "混合 — 两方面都需要记录"

      Judge: "最终: [价格_负面, 体验_正面, 整体_正面]"

      优势:
        - 多方视角强制暴露不同解读
        - 对抗促使每个 Agent 提供更充分的论证
        - Judge 综合得出更全面的结论
        - 避免单 Agent 的认知锁定

    关键洞察:
      外部对抗 > 内部反思
      因为"别人挑错"比"自己找错"更有效
    """)


if __name__ == "__main__":
    demo_sentiment_arbitration()
    demo_strategy_debate()
    demonstrate_dot_problem()

    print("\n生产环境建议:")
    print("  1. 使用不同模型/prompt确保Agent观点多样性")
    print("  2. 设置辩论轮数上限(3-5轮)，防止无限争论")
    print("  3. Judge使用更强的模型或明确的评分标准")
    print("  4. 记录完整辩论历史，支持事后审计")
    print("  5. 对共识度高的议题提前终止辩论，节省成本")
    print("  6. 与AutoGen/CAMEL集成：辩论作为GroupChat的一种模式")

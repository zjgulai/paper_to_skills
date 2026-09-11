"""
CAMEL — 角色扮演式自主协作多 Agent 框架
基于论文: Li et al. "CAMEL: Communicative Agents for 'Mind' Exploration of LLM Society", NeurIPS 2023

核心能力:
1. Role-Playing — AI User + AI Assistant 角色分离协作
2. Inception Prompting — 递归提示实现自主约束
3. Task Specifier — 模糊任务自动细化为可执行描述
4. 自主对话闭环 — 无需人工干预的多轮协作

母婴电商场景: 产品经理 × 数据分析师角色对协作完成 VOC 评论分析
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class Role(Enum):
    """角色定义"""
    AI_USER = "ai_user"           # 指令发出者
    AI_ASSISTANT = "ai_assistant"  # 执行者
    TASK_SPECIFIER = "task_specifier"  # 任务细化者


@dataclass
class AgentMessage:
    """Agent 消息"""
    role: Role
    content: str
    turn: int = 0


@dataclass
class InceptionConfig:
    """Inception Prompt 配置"""
    task: str
    ai_user_name: str
    ai_assistant_name: str
    ai_user_role: str
    ai_assistant_role: str
    protocol: str = ""  # 通信协议（终止条件、输出格式等）


class InceptionPromptBuilder:
    """
    Inception Prompt 构建器

    CAMEL 的核心创新：将任务、角色、协议嵌入系统提示，
    使 Agent 在对话中相互约束，无需外部监督。
    """

    def build_user_prompt(self, config: InceptionConfig) -> str:
        """构建 AI User 的系统提示"""
        return f"""You are {config.ai_user_name}, {config.ai_user_role}.
You are working with {config.ai_assistant_name}, {config.ai_assistant_role}.

Your task: {config.task}

Communication Protocol:
{config.protocol or self._default_protocol()}

IMPORTANT RULES:
1. You are the INSTRUCTION GIVER. Never offer to do the work yourself.
2. Give clear, specific instructions to your assistant.
3. Ask follow-up questions to deepen the analysis.
4. When the task is complete, say "<CAMEL_TASK_DONE>".
5. NEVER say "I will help you" or "Let me do that" — that's the assistant's job.
"""

    def build_assistant_prompt(self, config: InceptionConfig) -> str:
        """构建 AI Assistant 的系统提示"""
        return f"""You are {config.ai_assistant_name}, {config.ai_assistant_role}.
You are working with {config.ai_user_name}, {config.ai_user_role}.

Your task: {config.task}

Communication Protocol:
{config.protocol or self._default_protocol()}

IMPORTANT RULES:
1. You are the EXECUTOR. Follow the user's instructions precisely.
2. Provide detailed, evidence-based responses.
3. When you complete the requested task, say "<CAMEL_TASK_DONE>".
4. NEVER ask "What do you think?" or "What should I do next?" — wait for instructions.
5. If you need clarification, ask a specific question, then wait.
"""

    def _default_protocol(self) -> str:
        return """- AI User gives instructions and asks questions
- AI Assistant executes and provides detailed answers
- Conversation continues until task completion
- Termination: "<CAMEL_TASK_DONE>" signals task completion
- Format: Use structured output with bullet points and data"""


class TaskSpecifier:
    """
    任务细化器

    将模糊的人类指令转化为具体、可执行的任务描述。
    这是 CAMEL 的第一阶段，确保 Role-Playing Agent 对收到清晰的输入。
    """

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func or self._mock_llm

    def specify(self, vague_task: str, domain_context: str = "") -> str:
        """
        将模糊任务细化为具体任务

        Args:
            vague_task: 原始模糊指令，如"分析我们的竞品"
            domain_context: 领域上下文，如"母婴电商吸奶器品类"
        """
        prompt = f"""You are a Task Specifier. Your job is to transform vague task descriptions into concrete, actionable task specifications.

Domain Context: {domain_context or "General business analysis"}

Original Task (vague): {vague_task}

Please refine this into a specific task with:
1. Clear objectives (what needs to be produced)
2. Specific scope (what data to analyze, what dimensions to cover)
3. Output format (structured, with sections)
4. Success criteria (how to know when done)

Refined Task:"""

        return self.llm_func(prompt).strip()

    def _mock_llm(self, prompt: str) -> str:
        """模拟 LLM 输出"""
        # 根据 prompt 内容返回模拟的细化任务
        if "竞品" in prompt or "competitor" in prompt.lower():
            return ("分析 Spectra S1、Medela Pump In Style、Elvie Pump 三款吸奶器"
                    "在 Amazon 和 Trustpilot 的用户评价，输出: "
                    "(1) 各产品 Top 5 正面/负面属性对比表, "
                    "(2) 价格竞争力分析, "
                    "(3) 差异化定位洞察, "
                    "(4) SWOT 分析, "
                    "(5) 市场策略建议。数据需量化（提及率、情感得分）。")
        elif "评论" in prompt or "review" in prompt.lower():
            return ("从 1,247 条 Spectra S1 吸奶器评论中，提取: "
                    "(1) Top 5 高频正面属性及情感得分, "
                    "(2) Top 5 高频负面属性及情感得分, "
                    "(3) 与 Medela 的差异化评价, "
                    "(4) 用户推荐意愿的驱动因素。"
                    "所有结论需有数据支撑（提及率、百分比）。")
        return f"将任务 '{prompt[:50]}...' 分解为具体的 3-5 个执行步骤，每一步有明确的输出和验收标准。"


class RolePlayingAgent:
    """
    角色扮演 Agent

    每个 Agent 有固定的角色（AI User 或 AI Assistant），
    通过 Inception Prompt 约束其行为。
    """

    def __init__(self, role: Role, system_prompt: str,
                 llm_func: Optional[Callable] = None):
        self.role = role
        self.system_prompt = system_prompt
        self.llm_func = llm_func or self._mock_llm
        self.history: List[AgentMessage] = []

    def act(self, context: List[AgentMessage]) -> AgentMessage:
        """Agent 执行一轮对话"""
        # 构建完整上下文
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in context:
            role_str = "user" if msg.role == Role.AI_USER else "assistant"
            messages.append({"role": role_str, "content": msg.content})

        # 调用 LLM
        response = self.llm_func(messages)

        msg = AgentMessage(role=self.role, content=response, turn=len(context) // 2 + 1)
        self.history.append(msg)
        return msg

    def _mock_llm(self, messages: List[Dict]) -> str:
        """模拟 LLM 响应"""
        # 根据角色和上下文生成模拟响应
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "")
                break

        if self.role == Role.AI_ASSISTANT:
            if "正面" in last_user_msg or "positive" in last_user_msg.lower():
                return ("根据评论分析，Spectra S1 Top 5 正面属性：\n"
                        "1. 静音效果好 (提及率 34.2%, 正向 91%)\n"
                        "2. 双边设计省时 (提及率 28.7%, 正向 88%)\n"
                        "3. 夜间模式便利 (提及率 22.1%, 正向 85%)\n"
                        "4. 吸力可调范围广 (提及率 19.5%, 正向 82%)\n"
                        "5. 配件易清洁 (提及率 17.3%, 正向 79%)")
            elif "负面" in last_user_msg or "negative" in last_user_msg.lower():
                return ("Top 5 负面属性：\n"
                        "1. 价格偏高 (提及率 15.2%, 负面 73%)\n"
                        "2. 体积大不便携 (提及率 12.8%, 负面 81%)\n"
                        "3. 配件购买贵 (提及率 9.4%, 负面 68%)\n"
                        "4. 学习曲线陡峭 (提及率 7.1%, 负面 62%)\n"
                        "5. 电池续航一般 (提及率 6.3%, 负面 71%)\n\n"
                        "与 Medela 差异：Spectra 静音胜 (+23pp)，Medela 便携胜 (+15pp)")
            elif "SWOT" in last_user_msg or "report" in last_user_msg.lower():
                return ("# 竞品对标报告\n\n"
                        "## 1. 产品规格对比\n| 特性 | Spectra S1 | Medela | Elvie |\n"
                        "|------|-----------|--------|-------|\n"
                        "| 噪音 | 45dB | 52dB | 40dB |\n"
                        "| 重量 | 1.1kg | 0.9kg | 0.3kg |\n"
                        "| 价格 | $199 | $249 | $549 |\n\n"
                        "## 2. 用户情感对比\n- Spectra: 4.6★ (静音+性价比驱动)\n"
                        "- Medela: 4.4★ (品牌信任+便携性驱动)\n"
                        "- Elvie: 4.2★ (穿戴式创新但价格敏感)\n\n"
                        "<CAMEL_TASK_DONE>")
            else:
                return "收到，我将按照您的指示进行分析。请告诉我具体需要哪些维度的数据？"
        else:
            # AI User 的响应
            if "Top 5" in last_user_msg and "正面" in last_user_msg:
                return "负面属性 Top 5 呢？以及与 Medela 的差异？"
            elif "负面" in last_user_msg:
                return "基于这些数据，生成一份完整的竞品对标报告，包含 SWOT 分析。"
            elif "SWOT" in last_user_msg or "report" in last_user_msg:
                return "报告很完整。任务完成。<CAMEL_TASK_DONE>"
            else:
                return "请分析 Spectra S1 吸奶器评论中的高频正面属性 Top 5。"


class CAMELConversation:
    """
    CAMEL 对话编排器

    管理 AI User 和 AI Assistant 的完整对话循环，
    直到任务完成或达到最大轮次。
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.messages: List[AgentMessage] = []

    def run(self, user_agent: RolePlayingAgent,
            assistant_agent: RolePlayingAgent,
            initial_task: str) -> Dict:
        """
        执行完整的 CAMEL 对话

        Returns:
            包含完整对话历史、任务完成状态、统计信息
        """
        # 初始轮：AI User 发起任务
        initial_msg = AgentMessage(role=Role.AI_USER, content=initial_task, turn=0)
        self.messages.append(initial_msg)

        for turn in range(1, self.max_turns + 1):
            # Assistant 响应
            assistant_msg = assistant_agent.act(self.messages)
            self.messages.append(assistant_msg)

            # 检查终止条件
            if "<CAMEL_TASK_DONE>" in assistant_msg.content:
                return self._build_result(completed=True, terminated_by="assistant")

            # User 给出下一步指令
            user_msg = user_agent.act(self.messages)
            self.messages.append(user_msg)

            # 检查终止条件
            if "<CAMEL_TASK_DONE>" in user_msg.content:
                return self._build_result(completed=True, terminated_by="user")

        return self._build_result(completed=False, terminated_by="max_turns")

    def _build_result(self, completed: bool, terminated_by: str) -> Dict:
        return {
            "completed": completed,
            "terminated_by": terminated_by,
            "total_turns": len(self.messages) // 2,
            "conversation": [
                {"role": msg.role.value, "turn": msg.turn, "content": msg.content}
                for msg in self.messages
            ],
            "summary": self._summarize()
        }

    def _summarize(self) -> str:
        """生成对话摘要"""
        assistant_messages = [m for m in self.messages if m.role == Role.AI_ASSISTANT]
        if not assistant_messages:
            return "无 Assistant 响应"
        last = assistant_messages[-1].content
        # 提取主要结论
        lines = [l for l in last.split('\n') if l.strip() and not l.startswith('<')]
        return ' | '.join(lines[:3]) if lines else last[:100]


# ============================================
# 母婴电商 VOC 分析 — CAMEL 角色扮演示例
# ============================================

def demo_camel_voc_analysis():
    """演示 CAMEL 在 VOC 评论分析中的应用"""
    print("=" * 70)
    print("CAMEL — 角色扮演式自主协作")
    print("=" * 70)

    # Step 1: Task Specifier 细化任务
    print("\n[1] Task Specifier — 任务细化")
    specifier = TaskSpecifier()
    vague_task = "分析一下 Spectra S1 吸奶器的用户反馈"
    print(f"  原始任务: {vague_task}")
    specified_task = specifier.specify(vague_task, "母婴电商吸奶器品类")
    print(f"  细化后任务: {specified_task[:80]}...")

    # Step 2: 构建 Inception Prompt
    print("\n[2] Inception Prompting — 构建角色约束")
    builder = InceptionPromptBuilder()
    config = InceptionConfig(
        task=specified_task,
        ai_user_name="产品经理 Alice",
        ai_assistant_name="数据分析师 Bob",
        ai_user_role="资深母婴产品经理，关注市场洞察和用户需求",
        ai_assistant_role="VOC 数据分析师，擅长从评论中抽取结构化洞察"
    )

    user_prompt = builder.build_user_prompt(config)
    assistant_prompt = builder.build_assistant_prompt(config)

    print(f"  AI User 角色: {config.ai_user_name} ({config.ai_user_role[:20]}...)")
    print(f"  AI Assistant 角色: {config.ai_assistant_name} ({config.ai_assistant_role[:20]}...)")

    # Step 3: 初始化 Agent
    print("\n[3] 初始化 Role-Playing Agent 对")
    user_agent = RolePlayingAgent(Role.AI_USER, user_prompt)
    assistant_agent = RolePlayingAgent(Role.AI_ASSISTANT, assistant_prompt)

    # Step 4: 执行 CAMEL 对话
    print("\n[4] 执行 CAMEL 自主协作对话")
    print("-" * 50)

    conversation = CAMELConversation(max_turns=5)
    result = conversation.run(user_agent, assistant_agent, specified_task)

    # 展示对话过程
    for msg in result["conversation"]:
        role_label = "[User]" if msg["role"] == "ai_user" else "[Assistant]"
        content = msg["content"][:120]
        if len(msg["content"]) > 120:
            content += "..."
        print(f"  {role_label} {content}")

    print("-" * 50)

    # Step 5: 结果统计
    print(f"\n[5] 对话统计")
    print(f"  任务完成: {'是' if result['completed'] else '否'}")
    print(f"  终止方式: {result['terminated_by']}")
    print(f"  总轮次: {result['total_turns']}")
    print(f"  摘要: {result['summary'][:80]}...")

    print("\n" + "=" * 70)
    return result


def demonstrate_camel_vs_single_agent():
    """对比 CAMEL 角色扮演 vs 单 Agent"""
    print("\n" + "=" * 70)
    print("CAMEL 角色扮演 vs 单 Agent 对比")
    print("=" * 70)

    print("""
    单 Agent 分析问题:
      - 同一 Agent 既要"提问"又要"回答"，容易角色混乱
      - 缺乏外部视角的追问和质疑
      - 分析深度受限于单一思维模式

    CAMEL 角色扮演优势:
      ┌─────────────────┐      ┌─────────────────┐
      │  AI User        │ ──▶  │  AI Assistant   │
      │  (产品经理视角)  │      │  (数据分析师视角)│
      │  提问/质疑/深化  │ ◀──  │  执行/分析/回答  │
      └─────────────────┘      └─────────────────┘
             ↑                        │
             └──────── 闭环反馈 ───────┘

      - 角色分离确保职责清晰
      - AI User 的追问驱动分析深化
      - 双视角减少盲区
      - 无需人工逐步引导
    """)


if __name__ == "__main__":
    demo_camel_voc_analysis()
    demonstrate_camel_vs_single_agent()

    print("\n生产环境建议:")
    print("  1. 接入真实 LLM API (GPT-4 / Claude / DeepSeek)")
    print("  2. 增加对话轮次上限(10-20轮)和超时机制(30s/轮)")
    print("  3. 持久化对话历史到数据库，支持审计和复盘")
    print("  4. 结合 Self-Refine: AI User 对 Assistant 输出质量打分")
    print("  5. 支持多对 Agent 并行: 同时分析多个品类/品牌")
    print("  6. 与 AutoGen GroupChat 集成: CAMEL 作为其中一种对话模式")

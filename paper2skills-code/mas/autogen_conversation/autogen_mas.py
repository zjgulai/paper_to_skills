"""
AutoGen — 多智能体对话编排框架
基于论文: Wu et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"

核心能力:
1. Conversable Agent — 可定制、可对话的智能体，支持 LLM/人类/工具三种后端
2. Conversation Programming — 通过对话中心计算和控制流编排多 agent 协作
3. Flexible Conversation Patterns — 支持多种对话拓扑（一对一、群组、层级）

母婴电商场景: 多 agent 协同处理 VOC 分析任务（抽取→校验→汇总→预警）
"""

from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class AgentBackend(Enum):
    """Agent 后端类型"""
    LLM = "llm"
    HUMAN = "human"
    TOOL = "tool"


@dataclass
class Message:
    """Agent 间传递的消息"""
    sender: str
    content: str
    role: str = "assistant"  # assistant / user / system
    metadata: Dict = field(default_factory=dict)


class ConversableAgent:
    """
    可对话 Agent — AutoGen 的核心抽象

    每个 agent 具有:
    - name: 唯一标识
    - system_message: 系统提示，定义 agent 的角色和能力
    - backend: LLM / Human / Tool
    - reply_func: 自定义回复函数
    """

    def __init__(self, name: str, system_message: str,
                 backend: AgentBackend = AgentBackend.LLM,
                 reply_func: Optional[Callable] = None):
        self.name = name
        self.system_message = system_message
        self.backend = backend
        self.reply_func = reply_func or self._default_reply
        self.chat_history: List[Message] = []

    def _default_reply(self, messages: List[Message], sender: str) -> str:
        """默认回复逻辑（简化版，生产环境调用 LLM API）"""
        last_msg = messages[-1].content if messages else ""

        # 基于角色的简单规则回复（演示用）
        if "抽取" in last_msg or "extract" in last_msg.lower():
            return f"[{self.name}] 已抽取评论中的实体和情感: 产品=吸奶器, 情感=正面, 属性=静音"
        elif "校验" in last_msg or "verify" in last_msg.lower():
            return f"[{self.name}] 校验结果: 实体边界准确, 情感极性正确, 置信度=0.92"
        elif "汇总" in last_msg or "summarize" in last_msg.lower():
            return f"[{self.name}] 汇总报告: 本周 1,245 条评论, 正面 78%, 负面 12%, 中性 10%"
        elif "预警" in last_msg or "alert" in last_msg.lower():
            return f"[{self.name}] 预警: 检测到 3 起产品质量集中投诉，建议启动应急响应"
        else:
            return f"[{self.name}] 收到消息，继续处理..."

    def send(self, message: str, recipient: 'ConversableAgent') -> str:
        """向另一个 agent 发送消息并获取回复"""
        msg = Message(sender=self.name, content=message)
        self.chat_history.append(msg)

        # 构建对话上下文
        context = self.chat_history + recipient.chat_history
        reply = recipient.reply_func(context, self.name)

        reply_msg = Message(sender=recipient.name, content=reply)
        recipient.chat_history.append(msg)
        recipient.chat_history.append(reply_msg)
        self.chat_history.append(reply_msg)

        return reply

    def receive(self, message: Message) -> str:
        """接收消息（用于群组聊天）"""
        self.chat_history.append(message)
        reply = self.reply_func(self.chat_history, message.sender)
        return reply

    def reset(self):
        """重置对话历史"""
        self.chat_history.clear()


class GroupChat:
    """
    群组聊天 — 多 agent 广播式协作

    支持:
    - 轮询发言 (round-robin)
    - 基于内容的动态选择 (dynamic speaker selection)
    """

    def __init__(self, agents: List[ConversableAgent],
                 speaker_selection: str = "round_robin",
                 max_round: int = 10):
        self.agents = {a.name: a for a in agents}
        self.speaker_selection = speaker_selection
        self.max_round = max_round
        self.messages: List[Message] = []

    def run(self, initial_message: str, sender_name: str = "user") -> List[Message]:
        """运行群组对话"""
        self.messages.append(Message(sender=sender_name, content=initial_message))

        for _ in range(self.max_round):
            # 选择下一个发言者
            next_speaker = self._select_speaker()
            if not next_speaker:
                break

            # 构建上下文
            context = self.messages[-5:]  # 最近 5 条消息
            context_str = "\n".join(f"{m.sender}: {m.content}" for m in context)

            # 生成回复
            agent = self.agents[next_speaker]
            reply = agent.reply_func(self.messages, sender_name)

            msg = Message(sender=next_speaker, content=reply)
            self.messages.append(msg)

            # 终止条件
            if "TERMINATE" in reply or "完成" in reply:
                break

        return self.messages

    def _select_speaker(self) -> Optional[str]:
        """选择下一个发言者"""
        if self.speaker_selection == "round_robin":
            agent_names = list(self.agents.keys())
            if not self.messages:
                return agent_names[0]
            last_speaker = self.messages[-1].sender
            if last_speaker in agent_names:
                idx = agent_names.index(last_speaker)
                return agent_names[(idx + 1) % len(agent_names)]
            return agent_names[0]
        return None


class AutoGenOrchestrator:
    """
    AutoGen 编排器 — 管理多 agent 协作流程

    核心功能:
    1. Agent 注册与管理
    2. 对话模式配置（一对一 / 群组 / 层级）
    3. 对话驱动控制流
    """

    def __init__(self):
        self.agents: Dict[str, ConversableAgent] = {}
        self.patterns: Dict[str, Any] = {}

    def register_agent(self, agent: ConversableAgent):
        """注册 agent"""
        self.agents[agent.name] = agent

    def create_two_agent_chat(self, agent1_name: str, agent2_name: str) -> tuple:
        """创建一对一对话"""
        return self.agents[agent1_name], self.agents[agent2_name]

    def create_group_chat(self, agent_names: List[str],
                          speaker_selection: str = "round_robin",
                          max_round: int = 10) -> GroupChat:
        """创建群组对话"""
        agents = [self.agents[name] for name in agent_names]
        return GroupChat(agents, speaker_selection, max_round)

    def run_sequential_pipeline(self, tasks: List[Dict]) -> List[str]:
        """
        顺序管道执行

        Args:
            tasks: [{"from": agent1, "to": agent2, "message": str}, ...]
        """
        results = []
        for task in tasks:
            sender = self.agents[task["from"]]
            recipient = self.agents[task["to"]]
            reply = sender.send(task["message"], recipient)
            results.append(reply)
        return results


# ============================================
# 母婴电商 VOC 多 Agent 分析流水线
# ============================================

def create_voc_analysis_mas() -> AutoGenOrchestrator:
    """
    创建母婴电商 VOC 多 Agent 分析系统

    Agent 角色:
    - Extractor: 从评论中抽取实体、关系、情感
    - Verifier: 校验抽取结果的准确性和一致性
    - Summarizer: 汇总多维度分析结果
    - AlertManager: 监控异常并触发预警
    """
    orchestrator = AutoGenOrchestrator()

    # 1. 抽取 Agent
    extractor = ConversableAgent(
        name="Extractor",
        system_message="""你是 VOC 数据抽取专家。你的任务是从母婴产品评论中抽取：
1. 实体: 产品名、品牌、属性
2. 关系: 产品-属性关联、用户-产品交互
3. 情感: 方面级情感极性
输出格式: JSON""",
        backend=AgentBackend.LLM
    )

    # 2. 校验 Agent
    verifier = ConversableAgent(
        name="Verifier",
        system_message="""你是数据质量校验专家。你的任务是：
1. 检查实体边界是否准确
2. 验证情感极性是否与文本一致
3. 检测矛盾或冲突的标注
4. 输出置信度评分""",
        backend=AgentBackend.LLM
    )

    # 3. 汇总 Agent
    summarizer = ConversableAgent(
        name="Summarizer",
        system_message="""你是 VOC 分析报告专家。你的任务是：
1. 按品类/品牌/维度汇总情感分布
2. 识别 Top 问题清单
3. 生成趋势对比
4. 输出结构化报告""",
        backend=AgentBackend.LLM
    )

    # 4. 预警 Agent
    alert_manager = ConversableAgent(
        name="AlertManager",
        system_message="""你是舆情监控预警专家。你的任务是：
1. 监控负面情感突增
2. 检测质量/安全相关投诉集中
3. 识别竞品对比中的劣势
4. 触发分级预警""",
        backend=AgentBackend.LLM
    )

    for agent in [extractor, verifier, summarizer, alert_manager]:
        orchestrator.register_agent(agent)

    return orchestrator


def demo_sequential_pipeline():
    """演示顺序管道: Extractor → Verifier → Summarizer → AlertManager"""
    print("=" * 70)
    print("AutoGen — VOC 多 Agent 顺序分析管道")
    print("=" * 70)

    mas = create_voc_analysis_mas()

    review = """
    Spectra S1 吸奶器非常好用，静音效果很好，晚上不会吵醒宝宝。
    价格有点贵但值得。还买了储奶袋搭配使用。物流太慢了等了一周。
    """

    print(f"\n[输入评论]\n{review.strip()}")

    # 顺序管道
    pipeline = [
        {"from": "Extractor", "to": "Verifier",
         "message": f"请抽取以下评论的实体、关系和情感:\n{review}"},
        {"from": "Verifier", "to": "Summarizer",
         "message": "校验通过，请将抽取结果汇总为结构化报告"},
        {"from": "Summarizer", "to": "AlertManager",
         "message": "汇总完成，请监控是否有异常需要预警"},
    ]

    print("\n[管道执行]")
    results = mas.run_sequential_pipeline(pipeline)
    for i, result in enumerate(results, 1):
        print(f"\n  步骤 {i}: {result}")

    print("\n" + "=" * 70)


def demo_group_chat():
    """演示群组聊天: 多个 Agent 协作讨论"""
    print("\n" + "=" * 70)
    print("AutoGen — VOC 多 Agent 群组讨论")
    print("=" * 70)

    mas = create_voc_analysis_mas()

    # 创建群组
    group = mas.create_group_chat(
        ["Extractor", "Verifier", "Summarizer", "AlertManager"],
        speaker_selection="round_robin",
        max_round=6
    )

    initial = """本周收到 1,245 条母婴产品评论，其中：
- 吸奶器: 好评率 85%，主要抱怨噪音大
- 储奶袋: 好评率 92%，无显著问题
- 温奶器: 好评率 78%，多起投诉加热不均
请各 agent 从不同角度分析数据。
"""

    print(f"\n[初始消息]\n{initial}")
    messages = group.run(initial)

    print("\n[群组讨论记录]")
    for msg in messages[1:]:  # 跳过初始消息
        print(f"  {msg.sender}: {msg.content[:80]}...")

    print("\n" + "=" * 70)


def demo_hierarchical_chat():
    """演示层级对话: Manager → Worker Agents"""
    print("\n" + "=" * 70)
    print("AutoGen — 层级式 VOC 分析")
    print("=" * 70)

    mas = create_voc_analysis_mas()

    # 创建 Manager Agent
    manager = ConversableAgent(
        name="VOCManager",
        system_message="""你是 VOC 分析项目的管理者。你的任务是：
1. 将复杂分析任务分解为子任务
2. 分配给不同 specialist agent
3. 整合各 agent 的输出
4. 向业务方汇报最终结论""",
        backend=AgentBackend.LLM
    )
    mas.register_agent(manager)

    print("\n[Manager 分发任务]")
    manager_reply = manager.send(
        "本周需要完成全品类 VOC 分析报告，请分配任务给各 specialist",
        mas.agents["Extractor"]
    )
    print(f"  Extractor 回复: {manager_reply}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_sequential_pipeline()
    demo_group_chat()
    demo_hierarchical_chat()

    print("\n生产环境建议:")
    print("  1. 接入真实 LLM API (OpenAI/Claude/DeepSeek) 替代规则回复")
    print("  2. 使用 Azure AutoGen 官方库获得完整功能")
    print("  3. 配置人类介入机制（human-in-the-loop）用于关键决策")
    print("  4. 添加工具调用能力（代码执行、数据库查询、API 调用）")
    print("  5. 实现持久化存储和对话状态恢复")

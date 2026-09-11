"""
ReAct — 推理与行动交替执行
基于论文: Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023

核心能力:
1. Thought — 内部推理（计划、分析、策略更新）
2. Action — 执行行动（API调用、搜索、查询）
3. Observation — 接收外部反馈
4. Loop — 推理-行动-观察闭环

母婴电商场景: 竞品情报收集 Agent、VOC 异常根因分析
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class ActionType(Enum):
    """行动类型"""
    SEARCH = "search"           # 搜索
    QUERY = "query"             # 数据库查询
    ANALYZE = "analyze"         # 分析
    GENERATE = "generate"       # 生成报告
    COMPARE = "compare"         # 对比
    FINISH = "finish"           # 完成任务


@dataclass
class ReActStep:
    """ReAct 单步记录"""
    thought: str
    action: ActionType
    action_input: str
    observation: str
    step_number: int


class ActionRegistry:
    """
    行动注册表

    定义 Agent 可以执行的所有行动及其模拟实现。
    生产环境替换为真实 API 调用。
    """

    def __init__(self):
        self.actions = {
            ActionType.SEARCH: self._mock_search,
            ActionType.QUERY: self._mock_query,
            ActionType.ANALYZE: self._mock_analyze,
            ActionType.GENERATE: self._mock_generate,
            ActionType.COMPARE: self._mock_compare,
            ActionType.FINISH: self._mock_finish,
        }

    def execute(self, action_type: ActionType, action_input: str) -> str:
        """执行行动"""
        func = self.actions.get(action_type)
        if func:
            return func(action_input)
        return f"未知行动类型: {action_type.value}"

    def _mock_search(self, query: str) -> str:
        """模拟搜索"""
        responses = {
            "Spectra": "主要竞品: Medela Pump In Style ($249), Elvie Pump ($549), Willow Go ($499), Momcozy S12 ($159)",
            "competitor": "2024年吸奶器市场份额: Medela 28%, Spectra 22%, Elvie 15%, Momcozy 12%, 其他 23%",
            "review": "Spectra S1 Amazon: 4.6★ (2,341 reviews). 高频词: 静音(34%), 双边(29%), 价格(15%)",
        }
        for key, value in responses.items():
            if key.lower() in query.lower():
                return value
        return f"搜索结果: '{query[:30]}...' 找到 15 条相关记录"

    def _mock_query(self, query: str) -> str:
        """模拟数据库查询"""
        if "trend" in query.lower() or "趋势" in query:
            return "近30天负面率趋势: 12%→13%→15%→23%→22%→21%→20%. 3天前跳升"
        if "batch" in query.lower() or "批次" in query:
            return "批次 X202405: 发货量 1,247, 负面评论 89 (7.1% vs 平均 3.2%)"
        if "keyword" in query.lower() or "关键词" in query:
            return "负面高频词: 漏奶(45), 吸力不足(32), 配件松动(28), 噪音大(15)"
        return f"查询结果: {query[:30]}... 返回 42 条记录"

    def _mock_analyze(self, input_str: str) -> str:
        """模拟分析"""
        if "attribute" in input_str.lower() or "属性" in input_str:
            return "正面 Top 5: 静音(34.2%,91%), 双边(28.7%,88%), 夜间模式(22.1%,85%), 吸力可调(19.5%,82%), 易清洁(17.3%,79%)"
        if "sentiment" in input_str.lower() or "情感" in input_str:
            return "整体情感: 正面 67.3%, 中性 18.2%, 负面 14.5%. 情感强度: 强 45%, 中 32%, 弱 23%"
        return f"分析完成: {input_str[:30]}..."

    def _mock_generate(self, input_str: str) -> str:
        """模拟报告生成"""
        return f"报告已生成: {input_str[:20]}... (Markdown 格式, 包含 4 个章节)"

    def _mock_compare(self, input_str: str) -> str:
        """模拟对比"""
        if "Spectra" in input_str and "Medela" in input_str:
            return "Spectra vs Medela: 静音 +23pp, 价格 -$50, 便携性 -15pp, 品牌认知 -18pp"
        return f"对比结果: {input_str[:30]}..."

    def _mock_finish(self, input_str: str) -> str:
        """完成任务"""
        return "任务完成"


class ReActAgent:
    """
    ReAct Agent

    执行 Thought → Action → Observation 循环，直到任务完成。
    """

    def __init__(self, max_steps: int = 10,
                 action_registry: Optional[ActionRegistry] = None,
                 llm_func: Optional[Callable] = None):
        self.max_steps = max_steps
        self.registry = action_registry or ActionRegistry()
        self.llm_func = llm_func or self._mock_llm
        self.trajectory: List[ReActStep] = []

    def run(self, task: str) -> Dict:
        """
        执行 ReAct 循环

        Returns:
            包含完整轨迹、任务完成状态、最终答案
        """
        for step_num in range(1, self.max_steps + 1):
            # 1. Thought: 推理
            thought = self._think(task, self.trajectory)

            # 2. Action: 决定行动
            action_type, action_input = self._decide_action(task, thought, self.trajectory)

            # 3. Observation: 执行并观察
            observation = self.registry.execute(action_type, action_input)

            # 记录步骤
            step = ReActStep(
                thought=thought,
                action=action_type,
                action_input=action_input,
                observation=observation,
                step_number=step_num
            )
            self.trajectory.append(step)

            # 检查终止
            if action_type == ActionType.FINISH:
                return self._build_result(completed=True)

        return self._build_result(completed=False)

    def _think(self, task: str, trajectory: List[ReActStep]) -> str:
        """生成 Thought"""
        context = self._build_context(task, trajectory)
        return self.llm_func(context, mode="think")

    def _decide_action(self, task: str, thought: str, trajectory: List[ReActStep]) -> tuple:
        """决定下一步行动"""
        context = self._build_context(task, trajectory) + f"\nThought: {thought}"
        action_str = self.llm_func(context, mode="act")

        # 解析行动
        for action_type in ActionType:
            if action_type.value in action_str.lower():
                # 提取行动输入
                parts = action_str.split("|", 1)
                action_input = parts[1].strip() if len(parts) > 1 else action_str
                return action_type, action_input

        return ActionType.FINISH, "max_steps_reached"

    def _build_context(self, task: str, trajectory: List[ReActStep]) -> str:
        """构建完整上下文"""
        lines = [f"Task: {task}"]
        for step in trajectory:
            lines.extend([
                f"Step {step.step_number}:",
                f"  Thought: {step.thought}",
                f"  Action: {step.action.value}({step.action_input})",
                f"  Observation: {step.observation}",
            ])
        return "\n".join(lines)

    def _mock_llm(self, context: str, mode: str = "think") -> str:
        """模拟 LLM"""
        if mode == "think":
            return self._mock_think(context)
        else:
            return self._mock_act(context)

    def _mock_think(self, context: str) -> str:
        """模拟推理过程"""
        step_count = len(self.trajectory)

        if step_count == 0:
            return "需要分析 Spectra S1 的竞品情况。先搜索主要竞品信息。"

        last_action = self.trajectory[-1].action if self.trajectory else None

        if last_action == ActionType.SEARCH:
            if step_count == 1:
                return "已获取竞品列表。需要查询各产品的用户评分和评论数据。"
            return "已获取评分数据。Spectra 评分最高但评论数较少。需要深入分析评论内容。"

        if last_action == ActionType.QUERY:
            return "已获取评论分析结果。需要对比 Spectra 和主要竞品 Medela 的差异。"

        if last_action == ActionType.ANALYZE:
            return "已完成评论属性分析。需要对比 Spectra 和 Medela 的核心差异。"

        if last_action == ActionType.COMPARE:
            return "已完成竞品对比。数据充足，可以生成完整的竞品对标报告。"

        if last_action == ActionType.GENERATE:
            return "报告已生成。任务完成。"

        return "继续分析当前数据，决定下一步行动。"

    def _mock_act(self, context: str) -> str:
        """模拟行动决策"""
        step_count = len(self.trajectory)
        last_action = self.trajectory[-1].action if self.trajectory else None

        if step_count == 0 or (last_action == ActionType.SEARCH and step_count < 2):
            return "search|Spectra S1 breast pump competitors 2024"

        if last_action == ActionType.SEARCH:
            return "query|Amazon product reviews Spectra S1"

        if last_action == ActionType.QUERY:
            return "analyze|extract top attributes from reviews"

        if last_action == ActionType.ANALYZE:
            return "compare|Spectra S1 vs Medela Pump In Style"

        if last_action == ActionType.COMPARE:
            return "generate|competitive analysis report"

        if last_action == ActionType.GENERATE:
            return "finish|task completed"

        return "finish|task completed"

    def _build_result(self, completed: bool) -> Dict:
        return {
            "completed": completed,
            "total_steps": len(self.trajectory),
            "trajectory": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action.value,
                    "input": s.action_input,
                    "observation": s.observation
                }
                for s in self.trajectory
            ],
            "final_answer": self.trajectory[-1].observation if self.trajectory else ""
        }


# ============================================
# 母婴电商场景 — ReAct 竞品情报收集
# ============================================

def demo_react_competitor_analysis():
    """演示 ReAct 在竞品情报收集中的应用"""
    print("=" * 70)
    print("ReAct — 竞品情报收集 Agent")
    print("=" * 70)

    task = "分析 Spectra S1 吸奶器的主要竞品，生成竞品对标报告"
    print(f"\n[任务] {task}")
    print(f"[配置] 最大步数=10")

    agent = ReActAgent(max_steps=10)
    result = agent.run(task)

    print(f"\n[执行轨迹]")
    print("-" * 50)
    for step in result["trajectory"]:
        print(f"\nStep {step['step']}:")
        print(f"  Thought: {step['thought'][:60]}...")
        print(f"  Action: {step['action']}({step['input'][:40]}...)")
        print(f"  Observation: {step['observation'][:60]}...")

    print("-" * 50)
    print(f"\n[结果]")
    print(f"  任务完成: {'是' if result['completed'] else '否'}")
    print(f"  总步数: {result['total_steps']}")
    print(f"  最终输出: {result['final_answer'][:60]}...")

    print("\n" + "=" * 70)


def demo_react_anomaly_detection():
    """演示 ReAct 在 VOC 异常检测中的应用"""
    print("\n" + "=" * 70)
    print("ReAct — VOC 异常根因分析")
    print("=" * 70)

    task = "Spectra S1 本周负面评论率从 12% 上升到 23%，找出根因并生成预警"
    print(f"\n[任务] {task}")

    agent = ReActAgent(max_steps=8)
    result = agent.run(task)

    print(f"\n[执行轨迹]")
    for step in result["trajectory"]:
        print(f"  Step {step['step']}: {step['action']} → {step['observation'][:50]}...")

    print(f"\n[结果] 任务完成: {'是' if result['completed'] else '否'}, 步数: {result['total_steps']}")
    print("\n" + "=" * 70)


def demonstrate_react_loop():
    """展示 ReAct 循环结构"""
    print("\n" + "=" * 70)
    print("ReAct 推理-行动-观察循环")
    print("=" * 70)

    print("""
    ReAct 核心循环:

    ┌─────────────┐
    │   Thought   │ ← 推理：计划下一步、分析现状
    │  "需要搜索  │
    │   竞品信息"  │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Action    │ ← 行动：调用搜索 API
    │  search()   │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Observation │ ← 观察：接收搜索结果
    │ "竞品: Medela│
    │  Elvie..."  │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Thought   │ ← 推理：基于新信息更新计划
    │  "已获取竞品│
    │   列表，下一步│
    │   查询评分"  │
    └──────┬──────┘
           ↓
         (循环)

    优势:
      - Thought 提供可解释的推理链
      - Action 获取真实外部信息，消除幻觉
      - Observation 反馈纠正推理偏差
      - 闭环结构确保信息准确性
    """)


if __name__ == "__main__":
    demo_react_competitor_analysis()
    demo_react_anomaly_detection()
    demonstrate_react_loop()

    print("\n生产环境建议:")
    print("  1. 接入真实 API（搜索引擎、电商平台、数据库）")
    print("  2. 设置行动超时（5-10s）和重试机制（3次）")
    print("  3. 限制最大步数（10-20步），防止无限循环")
    print("  4. 持久化完整轨迹，支持审计和复盘")
    print("  5. 结合 ToT: 高层规划用树搜索，底层执行用 ReAct")
    print("  6. 实现 Action 权限控制，防止危险操作")

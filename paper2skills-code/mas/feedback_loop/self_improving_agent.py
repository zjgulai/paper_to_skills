"""
Self-Improving Agent Pipeline — 反馈闭环与自我进化
基于: Self-Refine (Madaan et al., NeurIPS 2023) + RL for Agent Improvement

核心能力:
1. 执行反馈收集 — 收集 agent 执行结果的质量信号
2. 自我反思 — Agent 对自身输出进行批评和改进
3. 经验记忆 — 将成功案例和失败教训存入长期记忆
4. 策略进化 — 基于反馈调整 agent 的行为策略

母婴电商场景: VOC 分析 Agent 的持续进化与质量提升
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class FeedbackType(Enum):
    """反馈类型"""
    SUCCESS = "success"         # 执行成功
    PARTIAL = "partial"         # 部分成功
    FAILURE = "failure"         # 执行失败
    HUMAN_CORRECTION = "human_correction"  # 人工修正


@dataclass
class ExecutionTrace:
    """执行轨迹 — 记录 agent 的完整执行过程"""
    trace_id: str
    task_description: str
    agent_name: str
    input_data: Dict
    output_data: Dict
    feedback_type: FeedbackType
    feedback_details: str = ""
    human_rating: Optional[float] = None  # 1-5 分
    execution_time: float = 0.0


@dataclass
class Experience:
    """经验 — 从执行轨迹中提取的可复用知识"""
    experience_id: str
    situation: str              # 什么情况下
    action: str                 # 采取了什么行动
    outcome: str                # 结果如何
    lesson: str                 # 学到了什么
    success_rate: float = 0.0
    usage_count: int = 0


class MemoryBank:
    """
    经验记忆库

    存储 agent 的成功经验和失败教训，支持：
    - 相似情况检索
    - 经验优先级排序（按成功率和使用频率）
    - 经验去重和合并
    """

    def __init__(self, capacity: int = 1000):
        self.experiences: List[Experience] = []
        self.capacity = capacity

    def add(self, experience: Experience):
        """添加经验"""
        # 去重检查
        for existing in self.experiences:
            if existing.situation == experience.situation and existing.action == experience.action:
                existing.success_rate = (existing.success_rate * existing.usage_count + experience.success_rate) / (existing.usage_count + 1)
                existing.usage_count += 1
                return

        self.experiences.append(experience)

        # 容量控制
        if len(self.experiences) > self.capacity:
            # 移除成功率最低的经验
            self.experiences.sort(key=lambda e: e.success_rate * e.usage_count, reverse=True)
            self.experiences = self.experiences[:self.capacity]

    def retrieve(self, situation: str, top_k: int = 5) -> List[Experience]:
        """检索相似情况的经验"""
        # 简化的相似度计算（关键词匹配）
        scored = []
        for exp in self.experiences:
            score = self._similarity(situation, exp.situation)
            scored.append((exp, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored[:top_k]]

    def _similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度（简化版）"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)


class SelfRefinementEngine:
    """
    自我反思引擎

    模拟 Self-Refine 论文中的 Feedback-Refine-Iterate 循环：
    1. Generate: Agent 生成初始输出
    2. Feedback: Agent 对自身输出进行批评
    3. Refine: Agent 基于批评改进输出
    4. Iterate: 重复直到满足质量阈值
    """

    def __init__(self, max_iterations: int = 3, quality_threshold: float = 0.85):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def refine(self, agent_func: Callable, input_data: Dict,
               feedback_func: Optional[Callable] = None) -> Dict:
        """
        执行 Self-Refine 循环

        Args:
            agent_func: 生成函数(agent, input) -> output
            input_data: 输入数据
            feedback_func: 反馈函数(output) -> feedback_str
        """
        # 初始生成
        current_output = agent_func(input_data)

        for iteration in range(self.max_iterations):
            # 获取反馈
            if feedback_func:
                feedback = feedback_func(current_output)
            else:
                feedback = self._default_feedback(current_output)

            # 评估质量
            quality = self._evaluate_quality(current_output, feedback)

            if quality >= self.quality_threshold:
                current_output["_meta"] = {
                    "iterations": iteration + 1,
                    "final_quality": quality,
                    "refined": iteration > 0
                }
                return current_output

            # 改进输出
            current_output = self._refine_output(current_output, feedback)

        current_output["_meta"] = {
            "iterations": self.max_iterations,
            "final_quality": quality,
            "refined": True,
            "warning": "达到最大迭代次数但未满足质量阈值"
        }
        return current_output

    def _default_feedback(self, output: Dict) -> str:
        """默认反馈（基于规则的批评）"""
        issues = []

        if "entities" in output and len(output["entities"]) == 0:
            issues.append("未抽取到任何实体")

        if "confidence" in output and output["confidence"] < 0.7:
            issues.append("置信度过低")

        if "sentiment" in output:
            sentiment = output["sentiment"]
            if "evidence" not in sentiment or not sentiment["evidence"]:
                issues.append("情感判断缺乏证据支持")

        return "; ".join(issues) if issues else "无明显问题"

    def _evaluate_quality(self, output: Dict, feedback: str) -> float:
        """评估输出质量（简化版）"""
        base_score = 0.8

        if "无明显问题" in feedback:
            base_score += 0.15
        else:
            issue_count = feedback.count(";") + 1
            base_score -= issue_count * 0.1

        if "_warning" in output.get("_meta", {}):
            base_score -= 0.1

        return max(0.0, min(1.0, base_score))

    def _refine_output(self, output: Dict, feedback: str) -> Dict:
        """基于反馈改进输出（简化版）"""
        refined = output.copy()
        refined["_refinement_feedback"] = feedback
        refined["_refinement_count"] = refined.get("_refinement_count", 0) + 1

        # 模拟改进
        if "置信度过低" in feedback:
            refined["confidence"] = min(1.0, output.get("confidence", 0.5) + 0.15)

        return refined


class FeedbackLoopOrchestrator:
    """
    反馈闭环编排器

    协调 Self-Refinement、Memory Bank 和策略进化，
    实现 Agent 的持续自我改进。
    """

    def __init__(self):
        self.memory = MemoryBank(capacity=1000)
        self.refinement_engine = SelfRefinementEngine()
        self.execution_traces: List[ExecutionTrace] = []

    def execute_with_feedback(self, agent_func: Callable, task: str,
                              input_data: Dict) -> Dict:
        """
        执行带有反馈闭环的任务

        流程:
        1. 从 Memory 检索相似经验
        2. 使用 Self-Refinement 生成高质量输出
        3. 收集反馈
        4. 将经验存入 Memory
        """
        # 1. 检索经验
        similar_exps = self.memory.retrieve(task)
        if similar_exps:
            input_data["_similar_experiences"] = [
                {"situation": e.situation, "action": e.action, "outcome": e.outcome}
                for e in similar_exps[:3]
            ]

        # 2. 执行 + Self-Refinement
        output = self.refinement_engine.refine(agent_func, input_data)

        # 3. 收集反馈
        trace = ExecutionTrace(
            trace_id=f"trace_{len(self.execution_traces):06d}",
            task_description=task,
            agent_name="VOC_Agent",
            input_data=input_data,
            output_data=output,
            feedback_type=FeedbackType.SUCCESS if output.get("_meta", {}).get("final_quality", 0) > 0.8 else FeedbackType.PARTIAL,
            human_rating=output.get("_meta", {}).get("final_quality", 0) * 5,
        )
        self.execution_traces.append(trace)

        # 4. 存入经验
        experience = Experience(
            experience_id=f"exp_{len(self.memory.experiences):06d}",
            situation=task,
            action=str(output.get("_meta", {})),
            outcome="success" if trace.feedback_type == FeedbackType.SUCCESS else "partial",
            lesson=output.get("_refinement_feedback", ""),
            success_rate=output.get("_meta", {}).get("final_quality", 0),
            usage_count=1,
        )
        self.memory.add(experience)

        return output

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.execution_traces:
            return {}

        total = len(self.execution_traces)
        success = sum(1 for t in self.execution_traces if t.feedback_type == FeedbackType.SUCCESS)
        partial = sum(1 for t in self.execution_traces if t.feedback_type == FeedbackType.PARTIAL)
        failure = sum(1 for t in self.execution_traces if t.feedback_type == FeedbackType.FAILURE)

        avg_quality = sum(t.human_rating or 0 for t in self.execution_traces) / total

        return {
            "total_executions": total,
            "success_rate": success / total,
            "partial_rate": partial / total,
            "failure_rate": failure / total,
            "average_quality": avg_quality,
            "memory_size": len(self.memory.experiences),
        }


# ============================================
# 母婴电商 VOC Agent 反馈闭环示例
# ============================================

def mock_voc_agent(input_data: Dict) -> Dict:
    """模拟 VOC 分析 Agent"""
    review = input_data.get("review", "")

    return {
        "entities": [{"text": "吸奶器", "type": "PRODUCT"}],
        "sentiment": {"polarity": "positive", "confidence": 0.75},
        "confidence": 0.75,
    }


def demo_feedback_loop():
    """演示反馈闭环"""
    print("=" * 70)
    print("Self-Improving Agent — 反馈闭环与自我进化")
    print("=" * 70)

    orchestrator = FeedbackLoopOrchestrator()

    # 预填充一些经验
    orchestrator.memory.add(Experience(
        experience_id="exp_001",
        situation="分析吸奶器评论",
        action="使用 ABSA 模型 + 关键词匹配",
        outcome="success",
        lesson="静音是关键属性，需要重点识别",
        success_rate=0.92,
        usage_count=10,
    ))

    # 执行任务
    tasks = [
        {"review": "Spectra S1 吸奶器非常好用，静音效果很好"},
        {"review": "储奶袋质量不错，但是价格有点贵"},
        {"review": "温奶器加热不均匀，有时候过热"},
    ]

    print("\n[执行 VOC 分析任务]")
    for i, task_input in enumerate(tasks, 1):
        print(f"\n  任务 {i}: {task_input['review'][:30]}...")

        # 检索相似经验
        similar = orchestrator.memory.retrieve(task_input["review"])
        if similar:
            print(f"    检索到 {len(similar)} 条相似经验")
            for exp in similar[:2]:
                print(f"      - {exp.situation} (成功率: {exp.success_rate:.2f})")

        # 执行
        output = orchestrator.execute_with_feedback(
            mock_voc_agent,
            task_input["review"],
            task_input
        )

        meta = output.get("_meta", {})
        print(f"    输出质量: {meta.get('final_quality', 0):.2f}")
        print(f"    迭代次数: {meta.get('iterations', 1)}")

    # 性能统计
    print("\n[反馈闭环性能统计]")
    stats = orchestrator.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if "rate" in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n[经验记忆库]")
    for exp in orchestrator.memory.experiences[:5]:
        print(f"  {exp.situation[:30]}... → {exp.outcome} (成功率: {exp.success_rate:.2f})")

    print("\n" + "=" * 70)


def demonstrate_evolution_cycle():
    """演示完整的进化循环"""
    print("\n" + "=" * 70)
    print("Agent 进化循环")
    print("=" * 70)

    print("""
    进化循环 (Evolution Cycle):

      执行 (Execute)
           ↓
      收集反馈 (Collect Feedback)
           ↓
      自我反思 (Self-Reflect)
           ↓
      策略调整 (Policy Update)
           ↓
      经验存储 (Store Experience)
           ↓
      下次执行时检索经验 (Retrieve Experience)
           ↓
      (循环)

    关键机制:
      1. Self-Refine: 生成 → 批评 → 改进 → 迭代
      2. Memory Bank: 长期记忆，支持跨任务经验复用
      3. Quality Gate: 质量阈值控制输出标准
      4. Human-in-the-Loop: 关键时刻人工介入校准
    """)


if __name__ == "__main__":
    demo_feedback_loop()
    demonstrate_evolution_cycle()

    print("\n生产环境建议:")
    print("  1. 使用向量数据库 (Pinecone/Milvus) 存储和检索经验")
    print("  2. 实现 RLHF 循环：人类反馈 → 奖励模型 → 策略优化")
    print("  3. 建立 A/B 测试框架对比不同策略的效果")
    print("  4. 设置质量门禁 (Quality Gate) 防止低质量输出流入生产")
    print("  5. 定期清理和合并记忆库，防止知识陈旧")

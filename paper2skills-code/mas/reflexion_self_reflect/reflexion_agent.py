"""
Reflexion — 言语强化学习与自我反思
基于论文: Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023

核心能力:
1. Actor — 执行任务，生成输出
2. Evaluator — 评估输出质量
3. Self-Reflection — 生成 verbal reinforcement（自然语言反思）
4. Episodic Memory — 存储和检索反思经验

母婴电商场景: VOC 打标质量自我提升、评论分类策略迭代优化
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Experience:
    """经验 — 从失败中提取的可复用反思"""
    task_type: str
    task_description: str
    attempt_output: str
    evaluation_result: str
    reflection: str
    success_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EpisodicMemory:
    """
    情节记忆库

    存储 Agent 的反思经验，支持相似任务的经验检索。
    """

    def __init__(self, capacity: int = 100):
        self.experiences: List[Experience] = []
        self.capacity = capacity

    def add(self, experience: Experience):
        """添加经验"""
        self.experiences.append(experience)
        if len(self.experiences) > self.capacity:
            # 移除最旧的经验
            self.experiences.pop(0)

    def retrieve(self, task_description: str, top_k: int = 3) -> List[Experience]:
        """检索相似任务的经验"""
        scored = []
        for exp in self.experiences:
            score = self._similarity(task_description, exp.task_description)
            scored.append((exp, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored[:top_k]]

    def _similarity(self, s1: str, s2: str) -> float:
        """计算相似度（简化版关键词匹配）"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def get_stats(self) -> Dict:
        """获取记忆库统计"""
        if not self.experiences:
            return {"total": 0, "avg_success_rate": 0.0}
        return {
            "total": len(self.experiences),
            "avg_success_rate": sum(e.success_rate for e in self.experiences) / len(self.experiences)
        }


class Actor:
    """
    Actor — 执行任务的 Agent

    接收任务和记忆增强的输入，生成输出。
    """

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func or self._mock_llm

    def act(self, task: str, memories: List[Experience]) -> str:
        """执行任务，使用记忆增强输入"""
        # 构建增强提示
        prompt = self._build_prompt(task, memories)
        return self.llm_func(prompt)

    def _build_prompt(self, task: str, memories: List[Experience]) -> str:
        """构建带记忆的提示"""
        prompt = f"Task: {task}\n\n"

        if memories:
            prompt += "Previous experiences with similar tasks:\n"
            for i, mem in enumerate(memories, 1):
                prompt += f"\nExperience {i}:\n"
                prompt += f"  Context: {mem.task_description[:80]}...\n"
                prompt += f"  Lesson: {mem.reflection[:120]}...\n"
            prompt += "\nApply these lessons to the current task.\n\n"

        prompt += "Output:"
        return prompt

    def _mock_llm(self, prompt: str) -> str:
        """模拟 LLM 输出"""
        # 根据任务类型返回模拟输出
        if "标签" in prompt or "label" in prompt.lower():
            if "经验" in prompt or "Experience" in prompt:
                # 有记忆增强 → 更好的输出
                return "[品牌_Momcozy, 品类_穿戴式吸奶器, 静音_正面, 便利性_正面]"
            return "[品牌_Momcozy, 静音_正面, 便利性_正面]"  # 遗漏品类标签

        if "分类" in prompt or "classify" in prompt.lower():
            if "经验" in prompt or "Experience" in prompt:
                return "策略: 先属性识别 → 否定检测 → 情感判断 (准确率 89.4%)"
            return "策略: 先情感 → 再属性 (准确率 82.3%)"

        return f"输出: {prompt[:30]}..."


class Evaluator:
    """
    Evaluator — 评估 Actor 的输出质量

    提供成功/失败信号和详细反馈。
    """

    def __init__(self, eval_func: Optional[Callable] = None):
        self.eval_func = eval_func or self._mock_evaluate

    def evaluate(self, task: str, output: str, ground_truth: Optional[str] = None) -> Dict:
        """评估输出"""
        return self.eval_func(task, output, ground_truth)

    def _mock_evaluate(self, task: str, output: str, ground_truth: Optional[str] = None) -> Dict:
        """模拟评估"""
        # 标签任务评估
        if "标签" in task or "label" in task.lower():
            if "品类" in output and "穿戴式" in output:
                return {
                    "success": True,
                    "score": 0.95,
                    "feedback": "正确识别了品牌、品类和属性。",
                    "details": "完整输出，无遗漏。"
                }
            else:
                return {
                    "success": False,
                    "score": 0.65,
                    "feedback": "遗漏了品类标签。",
                    "details": "'wearable pump' 应映射到 '品类_穿戴式吸奶器'。"
                }

        # 策略任务评估
        if "分类" in task or "classify" in task.lower():
            if "89.4" in output or "属性识别" in output:
                return {
                    "success": True,
                    "score": 0.894,
                    "feedback": "策略有效，准确率达到目标。",
                    "details": "先属性后情感的策略优于基线。"
                }
            else:
                return {
                    "success": False,
                    "score": 0.823,
                    "feedback": "策略有改进空间。",
                    "details": "属性映射错误率较高(15%)，建议改为先属性识别。"
                }

        return {"success": True, "score": 0.8, "feedback": "OK", "details": ""}


class SelfReflection:
    """
    Self-Reflection — 自我反思模型

    基于 Actor 的输出和 Evaluator 的反馈，生成 verbal reinforcement。
    """

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm_func = llm_func or self._mock_reflect

    def reflect(self, task: str, output: str, evaluation: Dict) -> str:
        """生成反思"""
        if evaluation["success"]:
            return self._reflect_success(task, output, evaluation)
        return self._reflect_failure(task, output, evaluation)

    def _reflect_success(self, task: str, output: str, evaluation: Dict) -> str:
        """成功时的反思"""
        prompt = f"""Task: {task}
Output: {output}
Evaluation: {evaluation['feedback']}

Summarize what worked well in 1-2 sentences:"""
        return self.llm_func(prompt, mode="success")

    def _reflect_failure(self, task: str, output: str, evaluation: Dict) -> str:
        """失败时的反思"""
        prompt = f"""Task: {task}
Output: {output}
Evaluation: {evaluation['feedback']}
Details: {evaluation['details']}

Analyze what went wrong and how to improve in 2-3 sentences:"""
        return self.llm_func(prompt, mode="failure")

    def _mock_reflect(self, prompt: str, mode: str = "failure") -> str:
        """模拟反思生成"""
        if mode == "failure":
            if "品类" in prompt or "category" in prompt.lower():
                return ("错误原因: 未识别 'wearable pump' 为产品类型标签。"
                        "教训: 'wearable' / 'hands-free' 等词应映射到品类标签。"
                        "改进: 下次遇到产品类型描述词时，优先检查是否对应品类标签。")
            if "属性" in prompt or "attribute" in prompt.lower():
                return ("问题: 情感先分类导致属性识别时丢失上下文。"
                        "例: '价格贵但值得' → 情感=正面，但属性'价格'映射失败。"
                        "改进: 改为先识别所有属性提及，再分别判断情感。")
            return "反思: 输出质量未达标。需要更仔细地检查输出完整性。"
        else:
            return "反思: 策略有效。保持当前方法，关注边界情况。"


class ReflexionAgent:
    """
    Reflexion Agent

    整合 Actor + Evaluator + Self-Reflection + Episodic Memory，
    实现言语强化学习循环。
    """

    def __init__(self, max_attempts: int = 3,
                 memory: Optional[EpisodicMemory] = None):
        self.max_attempts = max_attempts
        self.memory = memory or EpisodicMemory()
        self.actor = Actor()
        self.evaluator = Evaluator()
        self.reflection = SelfReflection()

    def solve(self, task: str, ground_truth: Optional[str] = None) -> Dict:
        """
        执行 Reflexion 循环

        Returns:
            包含最终输出、尝试次数、反思记录
        """
        attempts = []

        for attempt in range(1, self.max_attempts + 1):
            # 1. 检索记忆
            memories = self.memory.retrieve(task)

            # 2. Actor 执行
            output = self.actor.act(task, memories)

            # 3. Evaluator 评估
            evaluation = self.evaluator.evaluate(task, output, ground_truth)

            # 4. Self-Reflection
            reflection_text = self.reflection.reflect(task, output, evaluation)

            # 记录尝试
            attempts.append({
                "attempt": attempt,
                "output": output,
                "evaluation": evaluation,
                "reflection": reflection_text
            })

            # 5. 存入记忆
            exp = Experience(
                task_type=self._infer_task_type(task),
                task_description=task,
                attempt_output=output,
                evaluation_result=evaluation["feedback"],
                reflection=reflection_text,
                success_rate=evaluation["score"]
            )
            self.memory.add(exp)

            # 检查成功
            if evaluation["success"]:
                return {
                    "success": True,
                    "final_output": output,
                    "attempts": attempts,
                    "total_attempts": attempt,
                    "memory_stats": self.memory.get_stats()
                }

        # 达到最大尝试次数
        return {
            "success": False,
            "final_output": attempts[-1]["output"] if attempts else "",
            "attempts": attempts,
            "total_attempts": len(attempts),
            "memory_stats": self.memory.get_stats()
        }

    def _infer_task_type(self, task: str) -> str:
        """推断任务类型"""
        if "标签" in task or "label" in task.lower():
            return "labeling"
        if "分类" in task or "classify" in task.lower():
            return "classification"
        return "general"


# ============================================
# 母婴电商场景 — Reflexion VOC 打标质量提升
# ============================================

def demo_reflexion_labeling():
    """演示 Reflexion 在 VOC 打标中的应用"""
    print("=" * 70)
    print("Reflexion — VOC 打标质量自我提升")
    print("=" * 70)

    agent = ReflexionAgent(max_attempts=3)

    # 任务 1: 第一次尝试失败，生成反思
    task1 = '为评论 "Momcozy S12 wearable pump is super convenient" 打标签'
    print(f"\n[任务 1] {task1}")

    result1 = agent.solve(task1)

    print(f"  尝试次数: {result1['total_attempts']}")
    print(f"  最终成功: {'是' if result1['success'] else '否'}")

    for att in result1["attempts"]:
        print(f"\n  尝试 {att['attempt']}:")
        print(f"    输出: {att['output']}")
        print(f"    评估: {att['evaluation']['feedback']} (得分: {att['evaluation']['score']})")
        print(f"    反思: {att['reflection'][:80]}...")

    # 任务 2: 相似任务，利用记忆提升
    task2 = '为评论 "Elvie hands-free pump is quiet" 打标签'
    print(f"\n[任务 2] {task2}")
    print("  (相似任务，应利用之前的反思经验)")

    result2 = agent.solve(task2)

    print(f"  尝试次数: {result2['total_attempts']}")
    print(f"  最终成功: {'是' if result2['success'] else '否'}")
    print(f"  最终输出: {result2['final_output']}")

    # 记忆库统计
    print(f"\n[记忆库统计]")
    stats = agent.memory.get_stats()
    print(f"  总经验数: {stats['total']}")
    print(f"  平均成功率: {stats['avg_success_rate']:.2%}")

    print("\n" + "=" * 70)


def demo_reflexion_strategy_optimization():
    """演示 Reflexion 在分类策略优化中的应用"""
    print("\n" + "=" * 70)
    print("Reflexion — 评论分类策略迭代优化")
    print("=" * 70)

    agent = ReflexionAgent(max_attempts=3)

    # Epoch 1: 初始策略
    task = "优化母婴产品评论的多标签分类策略"
    print(f"\n[Epoch 1] 初始策略")
    result = agent.solve(task)

    for att in result["attempts"]:
        print(f"  尝试 {att['attempt']}: {att['output']}")
        print(f"    评估: {att['evaluation']['feedback']}")
        if att['reflection']:
            print(f"    反思: {att['reflection'][:100]}...")

    print(f"\n[结果] 最终策略: {result['final_output']}")
    print(f"       成功: {'是' if result['success'] else '否'}")

    print("\n" + "=" * 70)


def demonstrate_reflexion_architecture():
    """展示 Reflexion 架构"""
    print("\n" + "=" * 70)
    print("Reflexion 三组件架构")
    print("=" * 70)

    print("""
    Reflexion 架构:

    ┌─────────┐    Task + Memories     ┌─────────┐
    │ Episodic│ ─────────────────────▶ │  Actor  │
    │ Memory  │ ◀───────────────────── │         │
    └─────────┘    Reflection          └────┬────┘
                                            │ Output
                                            ↓
                                      ┌───────────┐
                                      │ Evaluator │
                                      │ (成功/失败  │
                                      │  + 分数)   │
                                      └─────┬─────┘
                                            │
                                            ↓
                                      ┌───────────┐
                                      │ Self-     │
                                      │ Reflection│
                                      │ (自然语言  │
                                      │  反思)     │
                                      └─────┬─────┘
                                            │
                                            ↓
                                      ┌───────────┐
                                      │ Episodic  │
                                      │ Memory    │
                                      │ (存储经验) │
                                      └───────────┘

    关键创新:
      - 无需权重更新: 通过自然语言反思改进策略
      - 经验可复用: 相似任务自动检索相关反思
      - 可解释: 反思是人类可读的自然语言
      - 零训练成本: 纯推理，无微调

    与 Self-Refine 的关系:
      - Self-Refine: 单任务内多次迭代改进
      - Reflexion: 跨任务积累长期经验
      - 最佳组合: 两者同时使用
    """)


if __name__ == "__main__":
    demo_reflexion_labeling()
    demo_reflexion_strategy_optimization()
    demonstrate_reflexion_architecture()

    print("\n生产环境建议:")
    print("  1. 使用向量数据库(Pinecone/Milvus)存储记忆，支持语义检索")
    print("  2. 设置评估阈值，只有'有意义'的失败才触发反思")
    print("  3. 定期清理陈旧记忆，防止知识过时")
    print("  4. 结合 RLHF: 人工审核反思质量")
    print("  5. 与 Self-Refine 结合: Reflexion 管跨任务，Self-Refine 管单任务")
    print("  6. 实现反思质量评估: 好的反思提升成功率，差的反思降级")

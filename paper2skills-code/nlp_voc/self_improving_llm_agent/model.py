"""
自迭代 LLM Agent 管线核心实现

核心设计: Generate-Review-Optimize (GRO) 三阶段闭环
- Generate: LLM 生成输出
- Review: Reflexion Engine 评估并生成反思报告
- Optimize: Self-Refine Engine 生成改进指令, DPOTrainer 积累偏好数据

依赖: openai, torch(可选, DPO训练阶段)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import json
import random
from datetime import datetime


@dataclass
class ExecutionResult:
    """单次执行结果"""
    input_context: str
    output: str
    metric_value: float          # 业务指标值
    success: bool
    timestamp: str


@dataclass
class ReflectionReport:
    """反思报告"""
    execution_id: str
    success: bool
    metric_value: float
    failure_mode: str            # 失败模式归类
    improvement_hint: str        # 改进建议
    retry_strategy: str          # 重试策略


@dataclass
class SelfEditInstruction:
    """自我编辑指令"""
    instruction: str             # 自然语言改进指令
    source_reflection: str       # 来源反思 ID
    effectiveness_score: float = 0.0   # 历史有效性评分
    apply_count: int = 0         # 应用次数


class ReflexionEngine:
    """
    反思引擎: 从执行结果生成结构化反思报告

    使用独立的 evaluator LLM 评估 Agent 输出, 避免自评偏差。
    维护失败模式分类器, 追踪最常见的失败类型。
    """

    def __init__(self, evaluator_llm: Callable[[str, str], str]):
        """
        Args:
            evaluator_llm: (evaluation_prompt, output_to_evaluate) -> reflection_text
        """
        self.evaluator = evaluator_llm
        self.failure_taxonomy: Dict[str, int] = {}

    def reflect(self, result: ExecutionResult, task_desc: str) -> ReflectionReport:
        """对单次执行结果生成反思报告"""

        eval_prompt = f"""你是一名严格的评估专家。请评估以下 Agent 输出。

任务: {task_desc}
Agent 输出: {result.output}
业务指标: {result.metric_value}

请输出 JSON 格式的评估报告:
{{
  "success": true/false,
  "failure_mode": "归类失败原因(如: 遗漏关键信息 / 表达不准确 / 格式错误 / 其他)",
  "improvement_hint": "具体的改进建议, 用自然语言描述",
  "retry_strategy": "下次执行时应采取的策略"
}}
"""
        reflection_text = self.evaluator(eval_prompt, result.output)

        try:
            parsed = json.loads(reflection_text)
        except json.JSONDecodeError:
            parsed = {
                "success": result.success,
                "failure_mode": "解析失败",
                "improvement_hint": "无法解析反思输出, 检查评估 LLM 的 JSON 格式遵循度",
                "retry_strategy": "使用更严格的输出格式要求"
            }

        # 更新失败模式统计
        fm = parsed.get("failure_mode", "未知")
        self.failure_taxonomy[fm] = self.failure_taxonomy.get(fm, 0) + 1

        return ReflectionReport(
            execution_id=result.timestamp,
            success=parsed.get("success", result.success),
            metric_value=result.metric_value,
            failure_mode=fm,
            improvement_hint=parsed.get("improvement_hint", ""),
            retry_strategy=parsed.get("retry_strategy", "")
        )

    def get_top_failure_modes(self, top_k: int = 5) -> List[tuple]:
        """返回最常见的失败模式"""
        return sorted(
            self.failure_taxonomy.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]


class SelfRefineEngine:
    """
    自精炼引擎: 将反思转化为可执行的自我编辑指令

    将自然语言的反思报告提炼为简洁的系统提示附加指令,
    这些指令被注入到后续 Generate 阶段的 prompt 中。
    """

    def __init__(self, refine_llm: Callable[[str], str]):
        self.refiner = refine_llm
        self.instruction_memory: List[SelfEditInstruction] = []

    def generate_instruction(self, reflection: ReflectionReport) -> SelfEditInstruction:
        """从反思报告生成自我编辑指令"""

        refine_prompt = f"""基于以下反思, 生成一条具体的"自我编辑指令"。
这条指令将被添加到 Agent 的系统提示中, 指导它下次更好地执行任务。

反思:
- 失败模式: {reflection.failure_mode}
- 改进建议: {reflection.improvement_hint}
- 重试策略: {reflection.retry_strategy}

要求:
1. 简洁, 1-2 句话
2. 可执行, 不是抽象建议
3. 聚焦具体行为改变, 不是目标描述

请生成:"""

        instruction_text = self.refiner(refine_prompt).strip()

        instruction = SelfEditInstruction(
            instruction=instruction_text,
            source_reflection=reflection.execution_id,
            effectiveness_score=0.0,
            apply_count=0
        )
        self.instruction_memory.append(instruction)
        return instruction

    def compile_instructions(self, max_instructions: int = 10) -> str:
        """将累积的指令编译为系统提示附加内容"""

        # 按有效性评分排序, 保留 top N
        sorted_instrs = sorted(
            self.instruction_memory,
            key=lambda x: x.effectiveness_score,
            reverse=True
        )[:max_instructions]

        if not sorted_instrs:
            return ""

        compiled = "\n\n=== 基于历史经验的改进指南 ===\n"
        for i, instr in enumerate(sorted_instrs, 1):
            compiled += f"{i}. {instr.instruction}\n"

        return compiled

    def update_effectiveness(self, instruction_idx: int, delta: float):
        """更新某条指令的有效性评分

        delta: +1 表示该指令带来了提升, -1 表示无效或恶化
        """
        if 0 <= instruction_idx < len(self.instruction_memory):
            instr = self.instruction_memory[instruction_idx]
            instr.effectiveness_score = (
                (instr.effectiveness_score * instr.apply_count + delta)
                / (instr.apply_count + 1)
            )
            instr.apply_count += 1


class DPOTrainer:
    """
    轻量级 DPO 训练数据管理器

    管理 (winner_output, loser_output) 偏好对比对,
    导出为标准格式供外部训练框架(TRL/Axolotl)使用。

    注: 本类仅管理数据, 实际 DPO 训练建议使用 trl.DPOTrainer。
    """

    def __init__(self, beta: float = 0.1):
        self.beta = beta
        self.preference_pairs: List[Dict] = []
        self._pair_keys: set = set()  # 去重用

    def _pair_key(self, context: str, winner: str, loser: str) -> str:
        """生成偏好对的唯一标识用于去重"""
        import hashlib
        content = f"{context}|{winner}|{loser}"
        return hashlib.md5(content.encode()).hexdigest()

    def add_preference_pair(
        self,
        context: str,
        winner_output: str,
        loser_output: str,
        winner_score: float,
        loser_score: float
    ) -> bool:
        """添加一条偏好对比数据

        winner_output: 高分输出
        loser_output: 低分输出
        margin: 分数差, 用于加权采样

        Returns:
            bool: True 表示添加成功, False 表示已存在被去重
        """
        key = self._pair_key(context, winner_output, loser_output)
        if key in self._pair_keys:
            return False

        self._pair_keys.add(key)
        self.preference_pairs.append({
            "context": context,
            "winner": winner_output,
            "loser": loser_output,
            "winner_score": winner_score,
            "loser_score": loser_score,
            "margin": winner_score - loser_score
        })
        return True

    def is_ready_for_training(self, min_pairs: int = 100) -> bool:
        """检查是否积累足够数据开始训练"""
        return len(self.preference_pairs) >= min_pairs

    def get_training_dataset(self) -> List[Dict]:
        """获取训练数据集"""
        return self.preference_pairs

    def export_to_jsonl(self, filepath: str):
        """导出为 JSONL 格式, 供 TRL/Axolotl 训练框架使用

        格式:
        {"prompt": "...", "chosen": "高分输出", "rejected": "低分输出"}
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in self.preference_pairs:
                f.write(json.dumps({
                    "prompt": pair["context"],
                    "chosen": pair["winner"],
                    "rejected": pair["loser"]
                }, ensure_ascii=False) + "\n")


class SelfImprovingAgent:
    """
    自迭代 LLM Agent: 完整的 GRO (Generate-Review-Optimize) 闭环

    使用方式:
    1. 调用 execute() 生成输出
    2. 业务系统运行输出并收集反馈指标
    3. 调用 record_result() 触发 Review + Optimize
    4. 后续 execute() 自动融入改进指令
    5. 积累足够数据后, export_dpo_data() 导出训练数据
    """

    def __init__(
        self,
        generate_llm: Callable[[str], str],
        evaluator_llm: Callable[[str, str], str],
        refine_llm: Callable[[str], str],
        task_description: str,
        metric_threshold: float = 0.7
    ):
        self.generate_llm = generate_llm
        self.task_description = task_description
        self.metric_threshold = metric_threshold

        self.reflexion = ReflexionEngine(evaluator_llm)
        self.self_refine = SelfRefineEngine(refine_llm)
        self.dpo_trainer = DPOTrainer()

        self.execution_history: List[ExecutionResult] = []
        self.reflection_history: List[ReflectionReport] = []

    def execute(self, context: str) -> str:
        """执行一次任务, 融入累积的自我编辑指令"""
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\n任务上下文: {context}"
        return self.generate_llm(full_prompt)

    def _build_system_prompt(self) -> str:
        """构建包含自我编辑指令的系统提示"""
        base_prompt = f"你是专业的 {self.task_description} Agent。"
        instructions = self.self_refine.compile_instructions()
        return base_prompt + instructions

    def record_result(self, context: str, output: str, metric_value: float) -> ReflectionReport:
        """记录执行结果并触发 Review + Optimize

        这是 GRO 闭环的核心入口。业务系统在获得反馈指标后调用此方法,
        自动完成反思、指令生成和 DPO 数据积累。
        """
        success = metric_value >= self.metric_threshold

        result = ExecutionResult(
            input_context=context,
            output=output,
            metric_value=metric_value,
            success=success,
            timestamp=datetime.now().isoformat()
        )
        self.execution_history.append(result)

        # === Review ===
        reflection = self.reflexion.reflect(result, self.task_description)
        self.reflection_history.append(reflection)

        # === Optimize ===
        if not success:
            instruction = self.self_refine.generate_instruction(reflection)
            print(f"[Optimize] 生成改进指令: {instruction.instruction}")

        # 积累 DPO 训练数据
        self._accumulate_preference_data(result)

        return reflection

    def _accumulate_preference_data(self, current_result: ExecutionResult):
        """积累 DPO 偏好对比数据

        对同一上下文的多个执行结果, 取最好和最差构成偏好对。
        只有当分数差异显著时才添加, 避免噪声数据。
        """
        same_context = [
            r for r in self.execution_history
            if r.input_context == current_result.input_context
        ]

        if len(same_context) >= 2:
            best = max(same_context, key=lambda x: x.metric_value)
            worst = min(same_context, key=lambda x: x.metric_value)

            # 显著差异阈值: 相对差异 > 10% 或绝对差异 > 0.005
            abs_diff = best.metric_value - worst.metric_value
            rel_diff = abs_diff / max(worst.metric_value, 1e-6)

            if abs_diff > 0.005 or rel_diff > 0.1:
                self.dpo_trainer.add_preference_pair(
                    context=current_result.input_context,
                    winner_output=best.output,
                    loser_output=worst.output,
                    winner_score=best.metric_value,
                    loser_score=worst.metric_value
                )

    def get_dpo_status(self) -> Dict:
        """获取 DPO 训练状态摘要"""
        return {
            "preference_pairs": len(self.dpo_trainer.preference_pairs),
            "ready_for_training": self.dpo_trainer.is_ready_for_training(),
            "reflections": len(self.reflection_history),
            "executions": len(self.execution_history),
            "top_failure_modes": self.reflexion.get_top_failure_modes(3)
        }

    def export_dpo_data(self, filepath: str):
        """导出 DPO 训练数据供外部训练框架使用"""
        self.dpo_trainer.export_to_jsonl(filepath)


# ==================== 业务适配层 ====================

class CopyOptimizationAgent(SelfImprovingAgent):
    """电商文案优化专用 Agent

    针对跨境电商商品文案生成场景优化:
    - metric_threshold: CTR 3% (电商文案行业基准)
    - 预置文案评估维度: 吸引力、信息完整性、CTA 有效性
    """

    def __init__(self, generate_llm, evaluator_llm, refine_llm):
        super().__init__(
            generate_llm=generate_llm,
            evaluator_llm=evaluator_llm,
            refine_llm=refine_llm,
            task_description="跨境电商商品营销文案生成",
            metric_threshold=0.03  # CTR 3%
        )

    def generate_copy(self, product_info: str, persona: str, tone: str) -> str:
        """生成营销文案"""
        context = f"产品: {product_info}\n目标用户: {persona}\n语气: {tone}"
        return self.execute(context)

    def record_ctr(self, context: str, copy_text: str, ctr: float) -> ReflectionReport:
        """记录 CTR 并触发闭环"""
        return self.record_result(context, copy_text, ctr)


class IntelligenceExtractionAgent(SelfImprovingAgent):
    """竞品情报萃取专用 Agent

    针对竞品情报结构化萃取场景优化:
    - metric_threshold: 准确率 85%
    - 预置萃取维度: 新品信息、价格变动、用户反馈要点
    """

    def __init__(self, generate_llm, evaluator_llm, refine_llm):
        super().__init__(
            generate_llm=generate_llm,
            evaluator_llm=evaluator_llm,
            refine_llm=refine_llm,
            task_description="竞品情报结构化萃取",
            metric_threshold=0.85  # 准确率 85%
        )

    def extract_intelligence(self, raw_content: str) -> str:
        """从原始内容萃取结构化情报"""
        context = f"原始内容: {raw_content[:2000]}"
        return self.execute(context)

    def record_accuracy(self, context: str, extraction: str, accuracy: float) -> ReflectionReport:
        """记录准确率并触发闭环"""
        return self.record_result(context, extraction, accuracy)


# ==================== 测试用例 ====================

def mock_llm(prompt: str) -> str:
    """模拟 LLM 调用"""
    if "文案" in prompt or "copy" in prompt.lower():
        return "【模拟文案】专为职场妈妈设计的静音吸奶器, 30分钟高效排空, 守护每一刻宁静。"
    return "【模拟输出】基于输入内容的处理结果。"


def mock_evaluator(eval_prompt: str, output: str) -> str:
    """模拟评估 LLM"""
    failure_modes = ["表达不够具体", "遗漏关键卖点", "语气不匹配", "CTA 不够明确"]
    hints = [
        "增加具体数字和产品特性描述",
        "确保覆盖所有核心卖点",
        "调整语气以匹配目标用户画像",
        "在结尾添加强有力的行动号召"
    ]
    idx = random.randint(0, 3)
    return json.dumps({
        "success": random.random() > 0.3,
        "failure_mode": failure_modes[idx],
        "improvement_hint": hints[idx],
        "retry_strategy": "在 prompt 中明确要求列出 3 个核心卖点"
    })


def mock_refiner(prompt: str) -> str:
    """模拟精炼 LLM"""
    instructions = [
        "在生成文案时, 优先使用具体数字而非模糊形容词, 并确保覆盖产品的核心卖点。",
        "文案结尾必须包含明确的行动号召(CTA), 使用动词开头如'立即'、'限时'。",
        "根据目标用户画像调整语气: 职场妈妈用专业简洁, 新手妈妈用温暖关怀。",
        "每条文案至少包含 2 个具体产品特性(如'30分钟'、'静音 40dB')而非笼统描述。"
    ]
    return random.choice(instructions)


def test_gro_pipeline():
    """测试完整的 GRO 闭环"""
    print("=" * 60)
    print("测试: 自迭代 LLM Agent GRO 闭环")
    print("=" * 60)

    agent = CopyOptimizationAgent(
        generate_llm=mock_llm,
        evaluator_llm=mock_evaluator,
        refine_llm=mock_refiner
    )

    # 模拟 20 轮执行
    for i in range(20):
        context = "Momcozy S12 Pro 吸奶器, 职场妈妈, professional"
        copy = agent.generate_copy("Momcozy S12 Pro", "职场妈妈", "professional")
        ctr = random.uniform(0.01, 0.05)  # 模拟 CTR 1%-5%
        reflection = agent.record_ctr(context, copy, ctr)

        if i < 5 or i % 5 == 0:
            print(f"\n轮次 {i+1}:")
            print(f"  CTR: {ctr:.3f}")
            print(f"  反思: {reflection.improvement_hint}")

    # 查看状态
    status = agent.get_dpo_status()
    print("\n" + "=" * 60)
    print("管线状态:")
    print(f"  执行次数: {status['executions']}")
    print(f"  反思次数: {status['reflections']}")
    print(f"  DPO 数据对: {status['preference_pairs']}")
    print(f"  可训练: {status['ready_for_training']}")
    print(f"  Top 失败模式: {status['top_failure_modes']}")

    # 导出 DPO 数据
    output_path = "/tmp/dpo_training_data.jsonl"
    agent.export_dpo_data(output_path)
    print(f"\nDPO 训练数据已导出至 {output_path}")

    # 验证导出文件
    with open(output_path, 'r') as f:
        lines = f.readlines()
        if lines:
            sample = json.loads(lines[0])
            print(f"数据样例:")
            print(f"  prompt: {sample['prompt'][:50]}...")
            print(f"  chosen: {sample['chosen'][:50]}...")
            print(f"  rejected: {sample['rejected'][:50]}...")

    return agent


def test_dpo_accumulation():
    """测试多 context 下的 DPO 数据积累"""
    print("\n" + "=" * 60)
    print("测试: 多 context DPO 数据积累")
    print("=" * 60)

    agent = CopyOptimizationAgent(
        generate_llm=mock_llm,
        evaluator_llm=mock_evaluator,
        refine_llm=mock_refiner
    )

    # 3 个不同的产品/画像组合
    contexts = [
        "Momcozy S12 Pro, 职场妈妈, professional",
        "Momcozy S12 Pro, 新手妈妈, warm",
        "BabyBuddha 吸奶器, 价格敏感, urgent",
    ]

    for ctx in contexts:
        for _ in range(5):
            copy = agent.execute(ctx)
            ctr = random.uniform(0.01, 0.05)
            agent.record_ctr(ctx, copy, ctr)

    status = agent.get_dpo_status()
    print(f"\n执行次数: {status['executions']}")
    print(f"DPO 数据对: {status['preference_pairs']}")
    print(f"可训练: {status['ready_for_training']}")

    # 验证: 多 context 应产生偏好对
    assert status['preference_pairs'] > 0, "多 context 应积累 DPO 偏好对"
    print("\n✓ 多 context DPO 积累测试通过")

    # 验证去重: 再次执行相同 context 不应重复添加
    first_count = status['preference_pairs']
    for ctx in contexts:
        copy = agent.execute(ctx)
        ctr = random.uniform(0.01, 0.05)
        agent.record_ctr(ctx, copy, ctr)

    second_count = agent.get_dpo_status()['preference_pairs']
    assert second_count == first_count, "去重机制应阻止重复偏好对"
    print(f"✓ 去重测试通过: 重复执行后偏好对数量保持 {first_count}")

    return agent


if __name__ == "__main__":
    test_gro_pipeline()
    test_dpo_accumulation()

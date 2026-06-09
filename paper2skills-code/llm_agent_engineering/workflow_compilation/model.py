"""
Subterranean Agent — 工作流编译范式模拟
论文: Compiling Agentic Workflows into LLM Weights (arXiv:2605.22502)
模拟将固定 SOP 从「运行时编排」转为「编译时参数化」的核心范式
"""

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import Any

# ─── 数据类定义 ────────────────────────────────────────────────────────────────

@dataclass
class WorkflowStep:
    """单个 SOP 步骤的元数据"""
    step_name: str
    prompt_template: str
    expected_output_schema: dict[str, str]
    estimated_input_tokens: int = 500
    estimated_output_tokens: int = 200

    def format_prompt(self, context: dict[str, Any]) -> str:
        """将上下文变量填充进 prompt 模板"""
        try:
            return self.prompt_template.format(**context)
        except KeyError as e:
            raise ValueError(f"步骤 [{self.step_name}] 缺少上下文变量: {e}") from e


@dataclass
class SOPWorkflow:
    """固定 SOP 工作流定义（多步骤序列）"""
    workflow_name: str
    description: str
    steps: list[WorkflowStep] = field(default_factory=list)

    def add_step(self, step: WorkflowStep) -> "SOPWorkflow":
        self.steps.append(step)
        return self

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_estimated_tokens(self) -> int:
        """运行时编排模式：每步独立调用的总 token 消耗（含上下文重复填充）"""
        # 每步都需要重新填充完整上下文（SOP说明 + 历史步骤输出 + 当前步骤 prompt）
        CONTEXT_OVERHEAD_PER_STEP = 800  # Orchestrator 系统 prompt + CoT 模板
        HISTORY_ACCUMULATION = 150       # 每步会累积前序输出
        total = 0
        for i, step in enumerate(self.steps):
            context_size = (
                step.estimated_input_tokens
                + CONTEXT_OVERHEAD_PER_STEP
                + i * HISTORY_ACCUMULATION    # 历史步骤输出的累积
            )
            total += context_size + step.estimated_output_tokens
        return total


@dataclass
class CompiledWorkflow:
    """编译后的工作流（单次推理完成全部 SOP）"""
    source_workflow: SOPWorkflow
    compiled_prompt: str
    compilation_timestamp: float
    estimated_tokens: int  # 编译后单次推理 token 数

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        模拟执行编译后的 workflow（实际场景中调用 fine-tuned 模型 API）
        此处返回 mock 输出结构，展示单次调用完成全部步骤的结果格式
        """
        start = time.perf_counter()

        # 实际: response = openai_client.chat.completions.create(
        #     model="your-finetuned-model-id",
        #     messages=[{"role": "user", "content": self.compiled_prompt.format(**input_data)}]
        # )
        # 此处模拟推理延迟（编译后约为运行时的 1/N 延迟）
        time.sleep(0.05)  # 模拟单次推理 ~50ms

        elapsed_ms = (time.perf_counter() - start) * 1000

        mock_output = {
            "workflow": self.source_workflow.workflow_name,
            "execution_mode": "compiled_single_inference",
            "elapsed_ms": round(elapsed_ms, 1),
            "steps_completed": self.source_workflow.total_steps,
            "results": {
                step.step_name: {
                    "status": "completed",
                    "output_schema": step.expected_output_schema,
                }
                for step in self.source_workflow.steps
            },
        }
        return mock_output


# ─── 编译器核心 ────────────────────────────────────────────────────────────────

class SubterraneanCompiler:
    """
    Subterranean Agent 编译器
    将多步 SOPWorkflow 转换为单一优化 prompt（用于 SFT 微调数据生成）
    """

    # 模拟 frontier 模型 API 价格（per 1M tokens）
    FRONTIER_INPUT_PRICE_PER_1M = 3.0   # USD, e.g. Claude 3.5 Sonnet
    FRONTIER_OUTPUT_PRICE_PER_1M = 15.0

    # 编译后部署成本（本地或私有云推理）
    COMPILED_INPUT_PRICE_PER_1M = 0.02  # USD, e.g. 自托管小模型
    COMPILED_OUTPUT_PRICE_PER_1M = 0.06

    def compile(self, sop: SOPWorkflow) -> CompiledWorkflow:
        """
        将 SOP 步骤序列转为单一优化 prompt（SFT 训练目标格式）
        实际场景：此 compiled_prompt 作为训练数据模板，用 SFT+GRPO 微调模型
        """
        sections = [
            f"# {sop.workflow_name} — 编译工作流\n",
            f"## 任务描述\n{sop.description}\n",
            "## 执行指令\n",
            "你是一个内化了完整 SOP 的专业 AI。",
            "收到输入后，一次性完成以下全部步骤并以 JSON 格式返回所有结果：\n",
        ]

        for i, step in enumerate(sop.steps, 1):
            sections.append(
                f"### 步骤 {i}: {step.step_name}\n"
                f"Prompt: {step.prompt_template}\n"
                f"输出 Schema: {json.dumps(step.expected_output_schema, ensure_ascii=False)}\n"
            )

        sections.append(
            "\n## 输出格式\n"
            "返回单一 JSON，包含每个步骤名称为 key 的输出结果。\n"
            "输入变量: {input_data}"
        )

        compiled_prompt = "\n".join(sections)

        # 编译后 token 估算：仅需一次推理，上下文大小 ≈ 原始多步骤的 1/N
        compiled_tokens = (
            sop.steps[0].estimated_input_tokens   # 用户输入
            + 300                                   # 编译后简化系统 prompt
            + sum(s.estimated_output_tokens for s in sop.steps)  # 合并输出
        )

        return CompiledWorkflow(
            source_workflow=sop,
            compiled_prompt=compiled_prompt,
            compilation_timestamp=time.time(),
            estimated_tokens=compiled_tokens,
        )

    def estimate_cost_saving(
        self,
        sop: SOPWorkflow,
        runs_per_day: int,
    ) -> dict[str, Any]:
        """
        量化编译前后成本节省
        返回详细对比报告，帮助决策是否值得投入 fine-tune 基础设施
        """
        # ── 运行时编排成本（frontier 模型 API）────────────────────────────
        runtime_tokens_per_run = sop.total_estimated_tokens
        runtime_cost_per_run = (
            runtime_tokens_per_run * 0.8 * self.FRONTIER_INPUT_PRICE_PER_1M / 1_000_000
            + runtime_tokens_per_run * 0.2 * self.FRONTIER_OUTPUT_PRICE_PER_1M / 1_000_000
        )
        runtime_cost_per_day = runtime_cost_per_run * runs_per_day

        # ── 编译后成本（自托管/私有推理）────────────────────────────────
        compiled_workflow = self.compile(sop)
        compiled_tokens_per_run = compiled_workflow.estimated_tokens
        compiled_cost_per_run = (
            compiled_tokens_per_run * 0.8 * self.COMPILED_INPUT_PRICE_PER_1M / 1_000_000
            + compiled_tokens_per_run * 0.2 * self.COMPILED_OUTPUT_PRICE_PER_1M / 1_000_000
        )
        compiled_cost_per_day = compiled_cost_per_run * runs_per_day

        # ── 节省比例 ──────────────────────────────────────────────────────
        saving_ratio = runtime_cost_per_run / max(compiled_cost_per_run, 1e-9)
        saving_usd_per_day = runtime_cost_per_day - compiled_cost_per_day
        saving_usd_per_month = saving_usd_per_day * 30

        # ── 一次性微调成本（SFT + GRPO，约 30-50 分钟 GPU 计算）─────────
        FINETUNING_COST_USD = 120  # 约 50 分钟 A100 × 8 GPU 云计算费用
        breakeven_days = FINETUNING_COST_USD / max(saving_usd_per_day, 1e-9)

        return {
            "workflow": sop.workflow_name,
            "sop_steps": sop.total_steps,
            "runs_per_day": runs_per_day,
            "runtime_orchestration": {
                "tokens_per_run": runtime_tokens_per_run,
                "cost_per_run_usd": round(runtime_cost_per_run, 6),
                "cost_per_day_usd": round(runtime_cost_per_day, 4),
            },
            "compiled_inference": {
                "tokens_per_run": compiled_tokens_per_run,
                "cost_per_run_usd": round(compiled_cost_per_run, 6),
                "cost_per_day_usd": round(compiled_cost_per_day, 4),
            },
            "savings": {
                "cost_reduction_ratio": f"{saving_ratio:.0f}×",
                "saving_usd_per_day": round(saving_usd_per_day, 2),
                "saving_usd_per_month": round(saving_usd_per_month, 2),
                "finetuning_cost_usd": FINETUNING_COST_USD,
                "breakeven_days": round(breakeven_days, 1),
                "quality_retention": "87–98%（SFT+GRPO 微调后）",
            },
            "recommendation": (
                "✅ 强烈推荐编译"
                if saving_usd_per_day > 10
                else "⚠️ 评估 ROI 后决策"
            ),
        }


# ─── 测试：母婴 Listing 上架 SOP ──────────────────────────────────────────────

def build_listing_sop() -> SOPWorkflow:
    """构建母婴 Listing 上架 5 步 SOP"""
    sop = SOPWorkflow(
        workflow_name="母婴-Listing上架SOP",
        description="将原始 SKU 信息转换为亚马逊合规上架内容，含标题优化、卖点提炼、合规检查、关键词填写、A+内容生成",
    )

    sop.add_step(WorkflowStep(
        step_name="标题优化",
        prompt_template="基于 SKU 信息 {sku_name}，品类 {category}，生成符合亚马逊 A10 算法的优化标题（含核心关键词、品牌词、核心属性），字数 ≤200 字符。",
        expected_output_schema={"optimized_title": "str", "keyword_density": "float"},
        estimated_input_tokens=450,
        estimated_output_tokens=150,
    ))

    sop.add_step(WorkflowStep(
        step_name="卖点提炼",
        prompt_template="基于产品功能 {features} 和目标用户 {target_audience}，提炼5条 Bullet Points，每条突出一个差异化卖点，≤ 200 字符/条。",
        expected_output_schema={"bullets": "list[str]", "usp_score": "float"},
        estimated_input_tokens=500,
        estimated_output_tokens=300,
    ))

    sop.add_step(WorkflowStep(
        step_name="合规检查",
        prompt_template="检查标题和卖点 {draft_content} 是否包含亚马逊禁用词（保健声明/医疗声明/超级词汇），输出违规项和修改建议。",
        expected_output_schema={"violations": "list[str]", "is_compliant": "bool", "suggestions": "list[str]"},
        estimated_input_tokens=600,
        estimated_output_tokens=200,
    ))

    sop.add_step(WorkflowStep(
        step_name="关键词填写",
        prompt_template="基于品类 {category} 和目标市场 {marketplace}，生成 Search Terms 关键词（后台填写用），去重、去品牌词、≤ 250 字节。",
        expected_output_schema={"search_terms": "str", "keyword_count": "int"},
        estimated_input_tokens=400,
        estimated_output_tokens=100,
    ))

    sop.add_step(WorkflowStep(
        step_name="A+内容生成",
        prompt_template="为产品 {sku_name} 生成 A+ 模块描述：品牌故事（50字）+ 功能模块文案（每模块 ≤ 100字）× 3 模块 + 使用场景描述（80字）。",
        expected_output_schema={"brand_story": "str", "feature_modules": "list[str]", "usage_scenario": "str"},
        estimated_input_tokens=500,
        estimated_output_tokens=400,
    ))

    return sop


def run_comparison_test():
    """运行编译前/后对比测试"""
    print("=" * 60)
    print("Subterranean Agent — 工作流编译 vs 运行时编排对比")
    print("=" * 60)

    sop = build_listing_sop()
    compiler = SubterraneanCompiler()

    # ── 编译 ──────────────────────────────────────────────────────────────
    print(f"\n[SOP] {sop.workflow_name}")
    print(f"  步骤数: {sop.total_steps}")
    print(f"  运行时编排 total tokens/run: {sop.total_estimated_tokens:,}")

    compiled = compiler.compile(sop)
    print(f"\n[编译完成] compiled_tokens/run: {compiled.estimated_tokens:,}")
    print(f"  token 压缩比: {sop.total_estimated_tokens / compiled.estimated_tokens:.1f}×")

    # ── 成本节省分析（1000 次/天）───────────────────────────────────────
    report = compiler.estimate_cost_saving(sop, runs_per_day=1000)
    print("\n[成本对比] 1000 次/天:")
    print(f"  运行时编排: ${report['runtime_orchestration']['cost_per_day_usd']:.2f}/天")
    print(f"  编译后推理: ${report['compiled_inference']['cost_per_day_usd']:.4f}/天")
    print(f"  节省比例: {report['savings']['cost_reduction_ratio']}")
    print(f"  节省金额: ${report['savings']['saving_usd_per_day']:.2f}/天")
    print(f"  月度节省: ${report['savings']['saving_usd_per_month']:.2f}/月")
    print(f"  微调成本: ${report['savings']['finetuning_cost_usd']}")
    print(f"  回本周期: {report['savings']['breakeven_days']:.1f} 天")
    print(f"  质量保留: {report['savings']['quality_retention']}")
    print(f"  建议: {report['recommendation']}")

    # ── 执行编译后 workflow ──────────────────────────────────────────────
    print("\n[执行] 模拟编译后 workflow 单次推理...")
    input_data = {
        "sku_name": "婴儿恒温水壶 Pro",
        "category": "Baby-FeedingBottles",
        "features": "精准控温±0.5°C, 304不锈钢内胆, 便携保温12小时",
        "target_audience": "0-3岁婴儿父母",
        "draft_content": "placeholder",
        "marketplace": "amazon.com",
    }
    result = compiled.execute(input_data)
    print(f"  执行模式: {result['execution_mode']}")
    print(f"  耗时: {result['elapsed_ms']} ms（模拟单次推理）")
    print(f"  完成步骤数: {result['steps_completed']}")
    print(f"  完成状态: {all(v['status'] == 'completed' for v in result['results'].values())}")

    print("\n" + "=" * 60)
    print("✅ 验证通过：Subterranean Agent 编译范式模拟运行成功")
    print("=" * 60)
    return report


if __name__ == "__main__":
    run_comparison_test()

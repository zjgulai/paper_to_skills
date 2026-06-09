"""
MASEval — 系统级 MAS 评估框架
arXiv: 2603.08835 | 2026年3月 | MIT License

核心功能：
- 将完整 MAS 系统（模型 × Framework × 协调逻辑）作为评测单元
- 3×3×3 全因子实验设计
- 量化 Framework 效应量 vs 模型效应量
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────
# 数据类定义
# ─────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    """标准评测任务单元"""
    task_id: str
    description: str
    input_data: dict[str, Any]
    expected_output: dict[str, Any]
    domain: str  # e.g. "supply_chain", "product_discovery", "customer_service"
    difficulty: str = "medium"  # easy / medium / hard


@dataclass
class AgentSystemConfig:
    """MAS 系统配置（三元组）"""
    model: str                   # e.g. "gpt-4o-mini", "claude-3-5-haiku"
    framework: str               # e.g. "langgraph", "crewai", "autogen", "smolagents"
    coordination_logic: str      # "sequential", "parallel", "adaptive"
    extra_params: dict[str, Any] = field(default_factory=dict)

    @property
    def system_id(self) -> str:
        return f"{self.model}__{self.framework}__{self.coordination_logic}"


@dataclass
class EvalResult:
    """单次系统评测结果"""
    system_config: AgentSystemConfig
    task_id: str
    accuracy: float              # 0.0 - 1.0
    latency_ms: float
    token_cost: int              # 总 token 消耗
    framework_overhead_ms: float # framework 本身引入的额外延迟
    success: bool = True
    error_msg: str = ""


@dataclass
class ComparisonReport:
    """多系统对比报告"""
    best_system: AgentSystemConfig
    all_results: dict[str, dict]  # system_id -> 聚合指标
    performance_gap: float        # 最优 vs 最差的准确率差距
    framework_effect_size: float  # Cohen's d: framework 变量的效应量
    model_effect_size: float      # Cohen's d: model 变量的效应量
    recommendation: str

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "MASEval 系统对比报告",
            "=" * 60,
            f"最优系统     : {self.best_system.system_id}",
            f"性能差距     : {self.performance_gap:.1%}（最优 vs 最差）",
            f"Framework 效应量: {self.framework_effect_size:.3f}",
            f"Model 效应量    : {self.model_effect_size:.3f}",
            f"建议          : {self.recommendation}",
            "-" * 60,
            "各系统准确率排名：",
        ]
        ranked = sorted(
            self.all_results.items(),
            key=lambda x: x[1]["accuracy_mean"],
            reverse=True,
        )
        for rank, (sid, metrics) in enumerate(ranked, 1):
            lines.append(
                f"  #{rank} {sid:<55} acc={metrics['accuracy_mean']:.3f} "
                f"lat={metrics['latency_ms_mean']:.0f}ms "
                f"tokens={metrics['token_cost_mean']:.0f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────
# MASEval 核心 Runner
# ─────────────────────────────────────────────

class MASEvalRunner:
    """
    MASEval 评估运行器。

    使用方式:
        runner = MASEvalRunner()
        result = runner.evaluate(config, tasks)
        report = runner.compare_systems(configs, tasks)
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    # ------------------------------------------------------------------
    # 模拟执行单个系统 × 单个任务（真实场景替换为 LLM API 调用）
    # ------------------------------------------------------------------
    def _simulate_task_execution(
        self,
        config: AgentSystemConfig,
        task: BenchmarkTask,
    ) -> EvalResult:
        """
        模拟 MAS 系统执行任务并返回评测结果。
        生产环境中，此方法替换为真实 Agent 调用 + 结果对比逻辑。
        """
        # 基准准确率（不同 framework 有不同基线）
        framework_base = {
            "smolagents": 0.82,
            "langgraph": 0.71,
            "crewai": 0.75,
            "autogen": 0.78,
            "llamaindex": 0.52,  # 体现论文中 30.9pp 差距
        }
        model_boost = {
            "gpt-4o": 0.08,
            "gpt-4o-mini": 0.00,
            "claude-3-5-haiku": 0.03,
            "claude-3-7-sonnet": 0.06,
        }
        coordination_boost = {
            "sequential": 0.00,
            "parallel": 0.03,
            "adaptive": 0.05,
        }

        base_acc = framework_base.get(config.framework, 0.70)
        acc = (
            base_acc
            + model_boost.get(config.model, 0.0)
            + coordination_boost.get(config.coordination_logic, 0.0)
            + random.gauss(0, 0.02)  # 模拟随机噪声
        )
        acc = max(0.0, min(1.0, acc))

        # 延迟模拟
        framework_latency = {
            "smolagents": 450,
            "langgraph": 820,
            "crewai": 650,
            "autogen": 590,
            "llamaindex": 980,
        }
        base_lat = framework_latency.get(config.framework, 700)
        latency = base_lat + random.gauss(0, 50)
        overhead = base_lat * 0.15  # 15% 为 framework overhead

        # Token 消耗模拟
        token_cost = int(random.gauss(1200, 200))

        return EvalResult(
            system_config=config,
            task_id=task.task_id,
            accuracy=acc,
            latency_ms=max(100, latency),
            token_cost=max(500, token_cost),
            framework_overhead_ms=overhead,
        )

    # ------------------------------------------------------------------
    # 评估单个系统（跨所有任务）
    # ------------------------------------------------------------------
    def evaluate(
        self,
        system_config: AgentSystemConfig,
        benchmark_tasks: list[BenchmarkTask],
    ) -> list[EvalResult]:
        """对单个 MAS 系统在所有 benchmark 任务上进行评测"""
        results = []
        for task in benchmark_tasks:
            result = self._simulate_task_execution(system_config, task)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # 多系统对比（核心方法）
    # ------------------------------------------------------------------
    def compare_systems(
        self,
        configs: list[AgentSystemConfig],
        tasks: list[BenchmarkTask],
    ) -> ComparisonReport:
        """
        对多个 MAS 系统配置进行全面对比评估。

        Args:
            configs: 待比较的系统配置列表（建议覆盖不同 framework / model / 协调逻辑）
            tasks: 评测任务集

        Returns:
            ComparisonReport 包含最优系统、性能差距、效应量分析
        """
        # 收集所有系统结果
        all_raw: dict[str, list[EvalResult]] = {}
        for config in configs:
            results = self.evaluate(config, tasks)
            all_raw[config.system_id] = results

        # 聚合指标
        aggregated: dict[str, dict] = {}
        for sid, results in all_raw.items():
            accuracies = [r.accuracy for r in results]
            aggregated[sid] = {
                "config": results[0].system_config,
                "accuracy_mean": sum(accuracies) / len(accuracies),
                "latency_ms_mean": sum(r.latency_ms for r in results) / len(results),
                "token_cost_mean": sum(r.token_cost for r in results) / len(results),
                "framework_overhead_mean": sum(r.framework_overhead_ms for r in results) / len(results),
            }

        # 找最优系统
        best_sid = max(aggregated, key=lambda s: aggregated[s]["accuracy_mean"])
        worst_sid = min(aggregated, key=lambda s: aggregated[s]["accuracy_mean"])
        performance_gap = (
            aggregated[best_sid]["accuracy_mean"] - aggregated[worst_sid]["accuracy_mean"]
        )

        # 效应量估算（Cohen's d 简化版）
        framework_effect_size = self._compute_effect_size(
            configs, aggregated, variable="framework"
        )
        model_effect_size = self._compute_effect_size(
            configs, aggregated, variable="model"
        )

        # 生成建议
        recommendation = self._generate_recommendation(
            framework_effect_size, model_effect_size, performance_gap
        )

        return ComparisonReport(
            best_system=aggregated[best_sid]["config"],
            all_results=aggregated,
            performance_gap=performance_gap,
            framework_effect_size=framework_effect_size,
            model_effect_size=model_effect_size,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _compute_effect_size(
        self,
        configs: list[AgentSystemConfig],
        aggregated: dict[str, dict],
        variable: str,  # "framework" or "model"
    ) -> float:
        """计算某一变量（framework/model）对准确率的 Cohen's d 效应量"""
        groups: dict[str, list[float]] = {}
        for config in configs:
            key = getattr(config, variable)
            sid = config.system_id
            if sid not in aggregated:
                continue
            groups.setdefault(key, []).append(aggregated[sid]["accuracy_mean"])

        if len(groups) < 2:
            return 0.0

        all_means = [sum(v) / len(v) for v in groups.values()]
        grand_mean = sum(all_means) / len(all_means)
        between_var = sum((m - grand_mean) ** 2 for m in all_means) / len(all_means)

        # 池化标准差近似（取最大组间均值差 / 2）
        max_diff = max(all_means) - min(all_means)
        pooled_std = max_diff / 2 if max_diff > 0 else 0.01

        return round(between_var ** 0.5 / pooled_std, 3)

    def _generate_recommendation(
        self,
        framework_es: float,
        model_es: float,
        gap: float,
    ) -> str:
        if gap > 0.20:
            dominant = "framework" if framework_es >= model_es else "model"
            return (
                f"⚠️  系统间性能差距达 {gap:.1%}，{dominant} 选择是主要驱动因素"
                f"（framework_es={framework_es:.3f} vs model_es={model_es:.3f}）。"
                "强烈建议在选型前完整运行 MASEval 全因子实验。"
            )
        elif gap > 0.10:
            return (
                f"⚡ 性能差距 {gap:.1%}，有显著优化空间。"
                "建议优先调整效应量更大的变量（framework 或 model）。"
            )
        else:
            return f"✅ 各系统性能差距较小（{gap:.1%}），可按成本/延迟选型。"


# ─────────────────────────────────────────────
# 测试：母婴选品任务场景
# ─────────────────────────────────────────────

def _build_baby_product_tasks() -> list[BenchmarkTask]:
    """构建母婴选品场景的 BenchmarkTask"""
    return [
        BenchmarkTask(
            task_id="bp_001",
            description="根据市场趋势推荐奶瓶补货量",
            input_data={"sku": "BOTTLE-A100", "current_stock": 120, "weekly_sales": 45},
            expected_output={"reorder_qty": 180, "reorder_timing": "2_weeks"},
            domain="supply_chain",
        ),
        BenchmarkTask(
            task_id="bp_002",
            description="识别爆款母婴商品并排序",
            input_data={"category": "stroller", "market": "US", "time_window": "30d"},
            expected_output={"top_skus": ["SKU-A", "SKU-B", "SKU-C"]},
            domain="product_discovery",
        ),
        BenchmarkTask(
            task_id="bp_003",
            description="竞品价格监控并给出定价建议",
            input_data={"our_price": 29.99, "competitor_prices": [27.5, 31.0, 28.8]},
            expected_output={"suggested_price": 28.5, "action": "reduce"},
            domain="pricing",
        ),
        BenchmarkTask(
            task_id="bp_004",
            description="用户评论情感分析驱动选品决策",
            input_data={"asin": "B08XYZ", "review_count": 1240, "avg_rating": 4.2},
            expected_output={"select_decision": "go", "risk_level": "low"},
            domain="product_discovery",
        ),
        BenchmarkTask(
            task_id="bp_005",
            description="跨境合规风险预筛",
            input_data={"product_type": "baby_bottle", "target_market": "EU"},
            expected_output={"compliance_risk": "medium", "required_certs": ["CE", "EN14350"]},
            domain="compliance",
        ),
    ]


def main():
    print("=" * 60)
    print("MASEval — 系统级 MAS 评估框架 演示")
    print("论文: arXiv:2603.08835 | MIT License")
    print("=" * 60)

    tasks = _build_baby_product_tasks()
    print(f"\n✅ 已构建 {len(tasks)} 个母婴选品 BenchmarkTask\n")

    # 定义 3 种 framework（同一模型，控制变量）
    configs = [
        AgentSystemConfig(model="gpt-4o-mini", framework="smolagents",  coordination_logic="sequential"),
        AgentSystemConfig(model="gpt-4o-mini", framework="crewai",      coordination_logic="sequential"),
        AgentSystemConfig(model="gpt-4o-mini", framework="llamaindex",  coordination_logic="sequential"),
        # 额外：同 framework，不同 coordination_logic
        AgentSystemConfig(model="gpt-4o-mini", framework="smolagents",  coordination_logic="parallel"),
        AgentSystemConfig(model="gpt-4o-mini", framework="smolagents",  coordination_logic="adaptive"),
        # 额外：不同 model，同 framework
        AgentSystemConfig(model="gpt-4o",      framework="smolagents",  coordination_logic="sequential"),
    ]

    print(f"📊 共 {len(configs)} 种系统配置，开始评估...\n")

    runner = MASEvalRunner(seed=42)
    report = runner.compare_systems(configs, tasks)

    print(report.summary())

    print("\n🔍 关键洞察：")
    if report.framework_effect_size >= report.model_effect_size:
        print(f"  → Framework 效应量 ({report.framework_effect_size:.3f}) ≥ Model 效应量 ({report.model_effect_size:.3f})")
        print("  → 与 MASEval 论文结论一致：framework 选型的重要性不亚于模型选型！")
    else:
        print(f"  → Model 效应量 ({report.model_effect_size:.3f}) > Framework 效应量 ({report.framework_effect_size:.3f})")
        print("  → 当前场景中模型能力是主要瓶颈，优先升级 LLM。")

    print(f"\n✅ MASEval 评估完成。最优系统: {report.best_system.system_id}")
    print(f"   性能差距: {report.performance_gap:.1%}（约合 {report.performance_gap * 100:.1f}pp）")


if __name__ == "__main__":
    main()

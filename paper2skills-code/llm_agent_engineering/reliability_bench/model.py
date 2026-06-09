"""
ReliabilityBench: Agent 生产可靠性三维评估框架
参考: arXiv 2601.06112 | ReliabilityBench (2026)

R(k, ε, λ) 可靠性曲面: 一致性 × 鲁棒性 × 故障容忍
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum


# ──────────────────────────────────────────────
# 数据类：配置与结果
# ──────────────────────────────────────────────

@dataclass
class ReliabilityConfig:
    """三维可靠性评估的超参配置"""
    k_trials: int = 5                              # 一致性维度：重复执行次数
    epsilon_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2]   # 鲁棒性维度：扰动幅度列表
    )
    lambda_levels: list[str] = field(
        default_factory=lambda: [
            "none", "timeout", "rate_limit", "partial_response"
        ]  # 故障等级
    )
    pass_threshold: float = 0.85                   # 通过率阈值
    timeout_prob: float = 0.3                      # 超时注入概率
    rate_limit_prob: float = 0.4                   # 限流注入概率
    partial_response_prob: float = 0.2             # 部分响应注入概率


@dataclass
class EpisodeResult:
    """单次 Episode 执行结果"""
    task: str
    output: Any
    success: bool
    latency_ms: float
    error: str | None = None


@dataclass
class ReliabilitySurface:
    """R(k, ε, λ) 三维可靠性曲面"""
    consistency_score: float                           # pass@k 一致性分数
    robustness_scores: dict[float, float]              # ε → 鲁棒性分数
    fault_tolerance_scores: dict[str, float]           # λ → 故障容忍分数
    overall_reliability: float                         # 综合可靠性得分

    def to_report(self) -> str:
        lines = [
            "=== ReliabilityBench 可靠性曲面报告 ===",
            f"一致性 (pass@k):         {self.consistency_score:.3f}",
            "",
            "鲁棒性 (ε-perturbations):",
        ]
        for eps, score in self.robustness_scores.items():
            lines.append(f"  ε={eps:.1f}: {score:.3f}")
        lines.append("")
        lines.append("故障容忍 (λ-fault injection):")
        for fault, score in self.fault_tolerance_scores.items():
            lines.append(f"  {fault}: {score:.3f}")
        lines.append("")
        lines.append(f"综合可靠性 R(k,ε,λ): {self.overall_reliability:.3f}")
        go_nogo = "✅ GO" if self.overall_reliability >= 0.85 else "❌ NO-GO"
        lines.append(f"上线决策: {go_nogo}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 故障注入器：Chaos Engineering Framework
# ──────────────────────────────────────────────

class FaultInjector:
    """
    混沌工程故障注入器，模拟生产级 API 基础设施故障。
    支持三类故障：timeout / rate_limit / partial_response
    """

    def __init__(self, config: ReliabilityConfig):
        self.config = config

    def inject_timeout(self, agent_fn: Callable, task: str) -> EpisodeResult:
        """注入超时故障：随机阻塞后触发降级"""
        start = time.time()
        if random.random() < self.config.timeout_prob:
            time.sleep(0.01)  # 生产中为实际超时等待
            return EpisodeResult(
                task=task,
                output=None,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                error="TimeoutError: tool call exceeded 30s limit",
            )
        return self._run_clean(agent_fn, task, start)

    def inject_rate_limit(self, agent_fn: Callable, task: str) -> EpisodeResult:
        """注入 API 限流故障（影响最大，-2.5%）"""
        start = time.time()
        if random.random() < self.config.rate_limit_prob:
            return EpisodeResult(
                task=task,
                output=None,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                error="RateLimitError: 429 Too Many Requests, retry after 60s",
            )
        return self._run_clean(agent_fn, task, start)

    def inject_partial_response(self, agent_fn: Callable, task: str) -> EpisodeResult:
        """注入部分响应故障：返回截断或不完整数据"""
        start = time.time()
        if random.random() < self.config.partial_response_prob:
            return EpisodeResult(
                task=task,
                output="[TRUNCATED]",
                success=False,
                latency_ms=(time.time() - start) * 1000,
                error="PartialResponseError: response body truncated at 512 bytes",
            )
        return self._run_clean(agent_fn, task, start)

    def _run_clean(
        self, agent_fn: Callable, task: str, start: float
    ) -> EpisodeResult:
        """无故障执行"""
        try:
            output = agent_fn(task)
            return EpisodeResult(
                task=task,
                output=output,
                success=True,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as exc:
            return EpisodeResult(
                task=task,
                output=None,
                success=False,
                latency_ms=(time.time() - start) * 1000,
                error=str(exc),
            )


# ──────────────────────────────────────────────
# 任务扰动器：Action Metamorphic Relations
# ──────────────────────────────────────────────

class TaskPerturbor:
    """
    基于 Action Metamorphic Relations 的任务描述扰动器。
    保证语义等价，改变措辞/顺序/句式，模拟真实用户表达多样性。
    """

    # 母婴场景专用同义词库
    SYNONYM_MAP: dict[str, list[str]] = {
        "补货": ["追加库存", "补充库存", "请求补货"],
        "婴儿": ["宝宝", "婴幼儿", "0-3岁儿童"],
        "安全": ["通过认证", "符合安全标准", "合规"],
        "推荐": ["建议", "提供候选", "筛选"],
        "价格": ["售价", "定价", "成本"],
        "库存": ["存货", "库存量", "在库数量"],
    }

    def perturb_task_description(self, task: str, epsilon: float = 0.1) -> str:
        """
        对任务描述施加 ε 幅度的语义等价扰动。
        epsilon=0.1: 轻微扰动（同义词替换）
        epsilon=0.2: 中等扰动（句式重组 + 信息顺序调整）
        """
        if epsilon <= 0.0:
            return task

        perturbed = task
        for original, synonyms in self.SYNONYM_MAP.items():
            if original in perturbed and random.random() < epsilon * 2:
                perturbed = perturbed.replace(original, random.choice(synonyms), 1)

        # epsilon >= 0.2 时追加句式重组（信息顺序调整）
        if epsilon >= 0.2 and len(perturbed) > 20:
            segments = perturbed.split("，")
            if len(segments) > 2:
                random.shuffle(segments)
                perturbed = "，".join(segments)

        return perturbed

    def generate_perturbations(
        self, task: str, epsilon: float, n: int = 5
    ) -> list[str]:
        """生成 n 个语义等价的扰动变体"""
        return [self.perturb_task_description(task, epsilon) for _ in range(n)]


# ──────────────────────────────────────────────
# 可靠性评估器：三维曲面计算
# ──────────────────────────────────────────────

class ReliabilityEvaluator:
    """
    R(k, ε, λ) 三维可靠性曲面评估器。

    用法::

        evaluator = ReliabilityEvaluator(config=ReliabilityConfig(k_trials=5))
        surface = evaluator.compute_reliability_surface(agent_fn, tasks)
        print(surface.to_report())
    """

    def __init__(self, config: ReliabilityConfig | None = None):
        self.config = config or ReliabilityConfig()
        self.fault_injector = FaultInjector(self.config)
        self.perturbor = TaskPerturbor()

    def evaluate_consistency(
        self, agent_fn: Callable, task: str, k: int | None = None
    ) -> float:
        """
        一致性评估：同一任务重复执行 k 次，返回 pass@k 经验通过率。
        pass@k = (k 次中成功次数) / k
        """
        k = k or self.config.k_trials
        results = [
            self.fault_injector._run_clean(agent_fn, task, time.time()).success
            for _ in range(k)
        ]
        return sum(results) / k

    def evaluate_robustness(
        self, agent_fn: Callable, task: str, epsilon: float
    ) -> float:
        """
        鲁棒性评估：任务描述扰动 ε 后的性能保持率。
        返回 ∈ [0,1]，越接近 1.0 越鲁棒。
        """
        n_samples = 5

        # 原始通过率（基线）
        baseline_results = [
            self.fault_injector._run_clean(agent_fn, task, time.time()).success
            for _ in range(n_samples)
        ]
        baseline_rate = sum(baseline_results) / n_samples

        # 扰动后通过率
        perturbations = self.perturbor.generate_perturbations(task, epsilon, n=n_samples)
        perturbed_results = [
            self.fault_injector._run_clean(agent_fn, p_task, time.time()).success
            for p_task in perturbations
        ]
        perturbed_rate = sum(perturbed_results) / n_samples

        if baseline_rate == 0:
            return 0.0
        return min(perturbed_rate / baseline_rate, 1.0)

    def evaluate_fault_tolerance(
        self, agent_fn: Callable, task: str, lambda_level: str
    ) -> float:
        """
        故障容忍评估：在指定故障类型下执行 10 次，返回通过率。
        lambda_level: "none" | "timeout" | "rate_limit" | "partial_response"
        """
        n_trials = 10
        inject_fn_map: dict[str, Callable] = {
            "timeout": self.fault_injector.inject_timeout,
            "rate_limit": self.fault_injector.inject_rate_limit,
            "partial_response": self.fault_injector.inject_partial_response,
        }

        results = []
        for _ in range(n_trials):
            if lambda_level == "none" or lambda_level not in inject_fn_map:
                result = self.fault_injector._run_clean(agent_fn, task, time.time())
            else:
                result = inject_fn_map[lambda_level](agent_fn, task)
            results.append(result.success)

        return sum(results) / n_trials

    def compute_reliability_surface(
        self, agent_fn: Callable, tasks: list[str]
    ) -> ReliabilitySurface:
        """
        计算完整的 R(k, ε, λ) 三维可靠性曲面。
        遍历所有任务，对三个维度分别聚合，输出 ReliabilitySurface。
        """
        # 维度 1：一致性（pass@k）
        consistency_score = sum(
            self.evaluate_consistency(agent_fn, task) for task in tasks
        ) / len(tasks)

        # 维度 2：鲁棒性（ε-perturbations）
        robustness_scores: dict[float, float] = {}
        for epsilon in self.config.epsilon_levels:
            scores = [
                self.evaluate_robustness(agent_fn, task, epsilon) for task in tasks
            ]
            robustness_scores[epsilon] = sum(scores) / len(scores)

        # 维度 3：故障容忍（λ-fault injection）
        fault_tolerance_scores: dict[str, float] = {}
        for lambda_level in self.config.lambda_levels:
            scores = [
                self.evaluate_fault_tolerance(agent_fn, task, lambda_level)
                for task in tasks
            ]
            fault_tolerance_scores[lambda_level] = sum(scores) / len(scores)

        # 综合可靠性：三维加权平均（一致性 40% + 鲁棒性 30% + 故障容忍 30%）
        avg_robustness = sum(robustness_scores.values()) / len(robustness_scores)
        avg_fault_tol = (
            sum(fault_tolerance_scores.values()) / len(fault_tolerance_scores)
        )
        overall = (
            0.4 * consistency_score + 0.3 * avg_robustness + 0.3 * avg_fault_tol
        )

        return ReliabilitySurface(
            consistency_score=consistency_score,
            robustness_scores=robustness_scores,
            fault_tolerance_scores=fault_tolerance_scores,
            overall_reliability=overall,
        )


# ──────────────────────────────────────────────
# 演示：母婴选品任务三种场景评估
# ──────────────────────────────────────────────

def _mock_selection_agent(task: str) -> str:
    """模拟选品 Agent（演示用）"""
    if any(kw in task for kw in ["SKU", "补货", "库存", "追加库存", "补充库存"]):
        return f"推荐商品: [{task[:20]}...] 置信度: 0.87"
    return f"选品结果: {task[:30]}"


def demo_reliability_evaluation() -> ReliabilitySurface:
    """母婴选品任务的三维可靠性评估演示"""
    tasks = [
        "推荐适合 0-6 月婴儿的安全奶嘴，价格 30-80 元，需通过 BPA-free 认证",
        "补充婴儿纸尿裤 SKU-A88 库存至安全线 500 件，优先深圳仓",
        "筛选下季度主推的益智玩具，目标客群 1-3 岁，毛利率 ≥ 35%",
    ]

    config = ReliabilityConfig(
        k_trials=5,
        epsilon_levels=[0.0, 0.1, 0.2],
        lambda_levels=["none", "timeout", "rate_limit"],
    )
    evaluator = ReliabilityEvaluator(config=config)
    surface = evaluator.compute_reliability_surface(_mock_selection_agent, tasks)

    print(surface.to_report())
    return surface


if __name__ == "__main__":
    demo_reliability_evaluation()

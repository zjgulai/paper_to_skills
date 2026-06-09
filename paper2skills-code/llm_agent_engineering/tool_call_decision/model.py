"""
Tool Call Decision Framework
arXiv: 2605.00737 | 2026-05

三维工具调用决策框架：
  Necessity（必要性）× Utility（效用）× Affordability（可负担性）
→ 输出 CALL / SKIP / DEFER

核心设计：纯 Python 实现，无需 PyTorch/sklearn，可直接集成到任意 Agent。
生产环境中 MLP 估计器使用 LLM 隐层状态作为特征，本模板用规则近似演示。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

class Decision(str, Enum):
    CALL  = "CALL"   # 调用工具
    SKIP  = "SKIP"   # 跳过，模型自行回答
    DEFER = "DEFER"  # 延迟调用（预算不足，稍后重试）


@dataclass
class ToolSpec:
    """工具规格描述"""
    name: str
    description: str
    cost_tokens: int = 200          # 每次调用消耗 token 估计
    latency_ms: int = 500           # 估计延迟（毫秒）
    success_rate: float = 0.95      # 历史成功率


@dataclass
class ToolCallContext:
    """工具调用决策上下文"""
    task_desc: str                              # 当前任务描述
    available_tools: list[ToolSpec]             # 候选工具列表
    history: list[dict[str, Any]] = field(default_factory=list)   # 历史步骤
    current_step: int = 0                       # 当前步骤编号
    token_budget: int = 4000                    # 剩余 token 预算
    token_used: int = 0                         # 已消耗 token
    time_budget_ms: int = 10_000                # 总时间预算（毫秒）
    time_used_ms: int = 0                       # 已用时间（毫秒）
    context_has_answer: bool = False            # 上下文中是否已有答案
    task_keywords: list[str] = field(default_factory=list)         # 任务关键词


# ─────────────────────────────────────────────
# 三维评分器
# ─────────────────────────────────────────────

class ToolNecessityScorer:
    """
    必要性评分器：任务是否真的需要工具？
    
    高分场景：实时数据查询、超出模型知识边界的专业检索
    低分场景：常识问题、上下文已包含答案、简单推理
    """

    def score(self, ctx: ToolCallContext, tool: ToolSpec) -> float:
        """返回 [0,1] 必要性得分"""
        score = 0.5  # 默认中性

        # 上下文已有答案 → 必要性大幅降低
        if ctx.context_has_answer:
            score -= 0.4

        # 历史步骤中已成功调用过同名工具 → 降低必要性（避免重复）
        already_called = any(
            h.get("tool") == tool.name and h.get("status") == "success"
            for h in ctx.history
        )
        if already_called:
            score -= 0.35

        # 任务关键词与工具描述词匹配 → 提高必要性
        tool_words = set(tool.description.lower().split())
        task_words = set(ctx.task_desc.lower().split())
        overlap = len(tool_words & task_words)
        score += min(0.3, overlap * 0.06)

        # 任务包含实时性关键词
        realtime_signals = {"实时", "最新", "今日", "current", "latest", "live", "now"}
        if any(kw in ctx.task_desc for kw in realtime_signals):
            score += 0.25

        return max(0.0, min(1.0, score))


class ToolUtilityScorer:
    """
    效用评分器：调用这个工具能带来多大增量收益？
    
    考虑：工具历史成功率、工具描述与任务的语义相关性、工具输出覆盖任务需求的程度
    """

    def score(self, ctx: ToolCallContext, tool: ToolSpec) -> float:
        """返回 [0,1] 效用得分"""
        score = tool.success_rate * 0.5  # 基础：工具本身可靠性

        # 语义相关性（关键词重叠）
        tool_words = set(tool.description.lower().split())
        task_words = set(ctx.task_desc.lower().split()) | set(ctx.task_keywords)
        overlap_ratio = len(tool_words & task_words) / max(len(task_words), 1)
        score += overlap_ratio * 0.4

        # 历史失败多次 → 效用下降
        fail_count = sum(
            1 for h in ctx.history
            if h.get("tool") == tool.name and h.get("status") == "failure"
        )
        score -= min(0.3, fail_count * 0.1)

        # 工具延迟极高且时间预算紧张 → 效用折扣
        time_remaining = ctx.time_budget_ms - ctx.time_used_ms
        if tool.latency_ms > time_remaining * 0.5:
            score *= 0.6

        return max(0.0, min(1.0, score))


class ToolAffordabilityScorer:
    """
    可负担性评分器：调用成本是否合理？
    
    综合 token 预算消耗比、时间预算消耗比、工具历史成功率风险
    """

    def score(self, ctx: ToolCallContext, tool: ToolSpec) -> float:
        """返回 [0,1] 可负担性得分"""
        # Token 剩余比例
        token_remaining_ratio = max(0.0, (ctx.token_budget - ctx.token_used - tool.cost_tokens)
                                    / ctx.token_budget)
        token_score = self._sigmoid(token_remaining_ratio, center=0.3, steepness=8)

        # 时间剩余比例
        time_remaining_ms = ctx.time_budget_ms - ctx.time_used_ms - tool.latency_ms
        time_remaining_ratio = max(0.0, time_remaining_ms / ctx.time_budget_ms)
        time_score = self._sigmoid(time_remaining_ratio, center=0.2, steepness=6)

        # 工具失败风险成本（失败会浪费 token 且延迟后续）
        failure_penalty = 1.0 - (1.0 - tool.success_rate) * 0.5

        affordability = (token_score * 0.5 + time_score * 0.3) * failure_penalty + 0.2
        return max(0.0, min(1.0, affordability))

    @staticmethod
    def _sigmoid(x: float, center: float = 0.5, steepness: float = 6) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


# ─────────────────────────────────────────────
# 决策框架（组合三维评分）
# ─────────────────────────────────────────────

@dataclass
class ToolCallResult:
    """单次工具调用决策结果"""
    tool_name: str
    decision: Decision
    necessity: float
    utility: float
    affordability: float
    composite_score: float
    reason: str


class ToolCallDecisionFramework:
    """
    三维加权决策框架
    
    默认权重：必要性 0.4 / 效用 0.4 / 可负担性 0.2
    默认阈值：composite >= 0.55 → CALL；affordability < 0.3 → DEFER；否则 SKIP
    """

    def __init__(
        self,
        w_necessity: float = 0.4,
        w_utility: float = 0.4,
        w_affordability: float = 0.2,
        call_threshold: float = 0.55,
        defer_affordability_threshold: float = 0.3,
    ):
        self.w_n = w_necessity
        self.w_u = w_utility
        self.w_a = w_affordability
        self.call_threshold = call_threshold
        self.defer_threshold = defer_affordability_threshold

        self._necessity_scorer = ToolNecessityScorer()
        self._utility_scorer = ToolUtilityScorer()
        self._affordability_scorer = ToolAffordabilityScorer()

    def decide(self, ctx: ToolCallContext, tool: ToolSpec) -> ToolCallResult:
        n = self._necessity_scorer.score(ctx, tool)
        u = self._utility_scorer.score(ctx, tool)
        a = self._affordability_scorer.score(ctx, tool)

        composite = self.w_n * n + self.w_u * u + self.w_a * a

        if composite >= self.call_threshold:
            decision = Decision.CALL
            reason = f"composite={composite:.2f} ≥ {self.call_threshold}，三维均衡，建议调用"
        elif a < self.defer_threshold:
            decision = Decision.DEFER
            reason = f"可负担性={a:.2f} 过低，预算不足，延迟调用"
        else:
            decision = Decision.SKIP
            reason = f"composite={composite:.2f} < {self.call_threshold}，收益不足以覆盖成本"

        return ToolCallResult(
            tool_name=tool.name,
            decision=decision,
            necessity=n,
            utility=u,
            affordability=a,
            composite_score=composite,
            reason=reason,
        )

    def decide_all(self, ctx: ToolCallContext) -> list[ToolCallResult]:
        """对所有候选工具批量决策"""
        return [self.decide(ctx, tool) for tool in ctx.available_tools]


# ─────────────────────────────────────────────
# 批量优化器
# ─────────────────────────────────────────────

@dataclass
class OptimizationSummary:
    total_steps: int
    original_calls: int
    optimized_calls: int
    skipped_calls: int
    deferred_calls: int
    token_saved: int
    reduction_rate: float


class ToolCallOptimizer:
    """
    批量优化 Agent 的工具调用序列
    
    输入：多步骤任务序列（每步包含上下文快照 + 候选工具列表）
    输出：优化后的调用计划 + 节省统计
    """

    def __init__(self, framework: ToolCallDecisionFramework | None = None):
        self.framework = framework or ToolCallDecisionFramework()

    def optimize_sequence(
        self, steps: list[ToolCallContext]
    ) -> tuple[list[list[ToolCallResult]], OptimizationSummary]:
        """
        优化完整步骤序列
        返回：每步的决策列表 + 汇总统计
        """
        all_results: list[list[ToolCallResult]] = []
        total_original = 0
        total_optimized = 0
        total_skipped = 0
        total_deferred = 0
        total_token_saved = 0

        for ctx in steps:
            step_results = self.framework.decide_all(ctx)
            all_results.append(step_results)

            total_original += len(ctx.available_tools)
            calls = [r for r in step_results if r.decision == Decision.CALL]
            skips = [r for r in step_results if r.decision == Decision.SKIP]
            defers = [r for r in step_results if r.decision == Decision.DEFER]

            total_optimized += len(calls)
            total_skipped += len(skips)
            total_deferred += len(defers)

            # 计算节省 token
            for r in step_results:
                if r.decision != Decision.CALL:
                    tool = next(t for t in ctx.available_tools if t.name == r.tool_name)
                    total_token_saved += tool.cost_tokens

        reduction_rate = (
            (total_original - total_optimized) / total_original
            if total_original > 0 else 0.0
        )

        summary = OptimizationSummary(
            total_steps=len(steps),
            original_calls=total_original,
            optimized_calls=total_optimized,
            skipped_calls=total_skipped,
            deferred_calls=total_deferred,
            token_saved=total_token_saved,
            reduction_rate=reduction_rate,
        )
        return all_results, summary


# ─────────────────────────────────────────────
# 测试：选品 Agent 场景（5 步骤）
# ─────────────────────────────────────────────

def _run_selection_agent_test() -> None:
    print("=" * 60)
    print("选品 Agent 工具调用优化测试")
    print("场景：母婴出海 - 婴儿奶粉品类扫描（5 步骤）")
    print("=" * 60)

    # 三个工具定义
    market_search = ToolSpec(
        name="market_search",
        description="Amazon 市场搜索 实时 销量 排名 竞品 baby formula",
        cost_tokens=300,
        latency_ms=800,
        success_rate=0.92,
    )
    price_query = ToolSpec(
        name="price_query",
        description="价格查询 实时 竞品定价 price monitor",
        cost_tokens=150,
        latency_ms=400,
        success_rate=0.97,
    )
    compliance_check = ToolSpec(
        name="compliance_check",
        description="合规检查 FDA 认证 法规 成分 合规 compliance regulatory",
        cost_tokens=400,
        latency_ms=1200,
        success_rate=0.88,
    )
    tools = [market_search, price_query, compliance_check]

    # 5 步骤场景
    steps: list[ToolCallContext] = [
        # Step 0：全新品类探索 → 三个工具都需要
        ToolCallContext(
            task_desc="探索婴儿奶粉 baby formula 市场 实时 排名 价格 合规",
            available_tools=tools,
            current_step=0,
            token_budget=4000,
            token_used=200,
            task_keywords=["婴儿奶粉", "baby formula", "实时", "合规"],
        ),
        # Step 1：价格已在上下文中，只需补充合规
        ToolCallContext(
            task_desc="补充合规检查 FDA 认证，价格数据已获取",
            available_tools=tools,
            history=[
                {"tool": "market_search", "status": "success"},
                {"tool": "price_query", "status": "success"},
            ],
            current_step=1,
            token_budget=4000,
            token_used=850,
            context_has_answer=False,
            task_keywords=["合规", "FDA", "认证"],
        ),
        # Step 2：已有所有数据，在总结分析阶段
        ToolCallContext(
            task_desc="汇总分析已有市场和价格数据，生成选品报告",
            available_tools=tools,
            history=[
                {"tool": "market_search", "status": "success"},
                {"tool": "price_query", "status": "success"},
                {"tool": "compliance_check", "status": "success"},
            ],
            current_step=2,
            token_budget=4000,
            token_used=1450,
            context_has_answer=True,
            task_keywords=["分析", "汇总", "报告"],
        ),
        # Step 3：预算紧张，触发 DEFER
        ToolCallContext(
            task_desc="实时价格监控 price 最新报价",
            available_tools=tools,
            current_step=3,
            token_budget=4000,
            token_used=3700,   # 预算快用完
            task_keywords=["price", "实时", "最新"],
        ),
        # Step 4：新子品类，需要重新市场搜索
        ToolCallContext(
            task_desc="探索有机婴儿奶粉 organic baby formula 细分市场 实时",
            available_tools=tools,
            current_step=4,
            token_budget=4000,
            token_used=500,
            task_keywords=["organic", "有机", "实时", "baby formula"],
        ),
    ]

    optimizer = ToolCallOptimizer()
    all_results, summary = optimizer.optimize_sequence(steps)

    for step_idx, (ctx, results) in enumerate(zip(steps, all_results)):
        print(f"\n▶ Step {step_idx}: {ctx.task_desc[:45]}...")
        for r in results:
            icon = "✅" if r.decision == Decision.CALL else ("⏳" if r.decision == Decision.DEFER else "❌")
            print(
                f"  {icon} [{r.decision.value:6s}] {r.tool_name:<20} "
                f"N={r.necessity:.2f} U={r.utility:.2f} A={r.affordability:.2f} "
                f"→ {r.composite_score:.2f}"
            )

    print("\n" + "─" * 60)
    print("📊 优化摘要")
    print(f"  总步骤：{summary.total_steps}")
    print(f"  原始调用次数：{summary.original_calls}")
    print(f"  优化后调用次数：{summary.optimized_calls}")
    print(f"  SKIP：{summary.skipped_calls}  DEFER：{summary.deferred_calls}")
    print(f"  节省 Token：{summary.token_saved}")
    print(f"  调用削减率：{summary.reduction_rate:.1%}")

    # 断言验证
    assert summary.original_calls == 15, f"预期 15 次原始调用，实际 {summary.original_calls}"
    assert summary.optimized_calls < summary.original_calls, "优化后调用次数应小于原始"
    assert summary.reduction_rate >= 0.30, f"削减率应 ≥ 30%，实际 {summary.reduction_rate:.1%}"
    assert summary.token_saved > 0, "应有 token 节省"

    print("\n✅ 所有断言通过：过滤无效调用后 token 减少 ≥ 30%")


if __name__ == "__main__":
    _run_selection_agent_test()

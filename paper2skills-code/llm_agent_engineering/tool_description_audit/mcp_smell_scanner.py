"""MCP Tool Description Smell Scanner — 六维评分与动态路由.

参考论文: Hasan, Li, Rajbahadur, Adams, Hassan (2026).
Model Context Protocol (MCP) Tool Descriptions Are Smelly!
arxiv:2602.14878.

本实现演示:
- ToolDescription: MCP tool 描述六组件模型
- ScoringRubric: 5-point Likert 六维评分 (score < 3 = smell)
- SmellScanner: 扫描 tool 描述，输出 smell 报告
- DescriptionAugmentor: 基于规则/LLM 的描述增强
- ToolDescriptionRouter: 运行时动态选择描述版本
- 母婴客服 tool 审核 demo

生产环境:
- 接 FM API 做自动评分
- 接 CI/CD 做上线前审核
- 运行时根据 query 复杂度路由描述版本
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tool Description Components
# ---------------------------------------------------------------------------


class DescriptionComponent(Enum):
    """Tool 描述六组件."""

    PURPOSE = "purpose"
    GUIDELINES = "guidelines"
    LIMITATIONS = "limitations"
    PARAMETER_EXPLANATION = "parameter_explanation"
    EXAMPLES = "examples"
    RETURN_VALUE = "return_value"


class SmellType(Enum):
    """六类 smell."""

    UNCLEAR_PURPOSE = "unclear_purpose"
    MISSING_USAGE_GUIDELINES = "missing_usage_guidelines"
    UNSTATED_LIMITATIONS = "unstated_limitations"
    OPAQUE_PARAMETERS = "opaque_parameters"
    UNDERSPECIFIED_INCOMPLETE = "underspecified_incomplete"
    EXEMPLAR_ISSUES = "exemplar_issues"


# 组件 → Smell 映射
COMPONENT_TO_SMELL: dict[DescriptionComponent, SmellType] = {
    DescriptionComponent.PURPOSE: SmellType.UNCLEAR_PURPOSE,
    DescriptionComponent.GUIDELINES: SmellType.MISSING_USAGE_GUIDELINES,
    DescriptionComponent.LIMITATIONS: SmellType.UNSTATED_LIMITATIONS,
    DescriptionComponent.PARAMETER_EXPLANATION: SmellType.OPAQUE_PARAMETERS,
    DescriptionComponent.RETURN_VALUE: SmellType.UNDERSPECIFIED_INCOMPLETE,
    DescriptionComponent.EXAMPLES: SmellType.EXEMPLAR_ISSUES,
}


@dataclass
class ComponentScore:
    """单组件评分结果."""

    component: DescriptionComponent
    score: int  # 1-5
    feedback: str = ""

    def is_smelly(self) -> bool:
        return self.score < 3


@dataclass
class ToolDescription:
    """MCP Tool 完整描述."""

    name: str
    purpose: str = ""
    guidelines: str = ""
    limitations: str = ""
    parameter_explanation: str = ""
    examples: str = ""
    return_value: str = ""

    def get_component(self, comp: DescriptionComponent) -> str:
        return getattr(self, comp.value, "")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "guidelines": self.guidelines,
            "limitations": self.limitations,
            "parameter_explanation": self.parameter_explanation,
            "examples": self.examples,
            "return_value": self.return_value,
        }


# ---------------------------------------------------------------------------
# Scoring Rubric
# ---------------------------------------------------------------------------


class ScoringRubric:
    """5-point Likert 评分 rubric.

    Score 3 = minimum viable threshold.
    Score < 3 = smell triggered.
    """

    SCORE_LABELS = {
        5: "ideal",
        4: "good",
        3: "minimum_viable",
        2: "deficient",
        1: "missing",
    }

    def score_component(
        self, component: DescriptionComponent, text: str
    ) -> ComponentScore:
        """对单个组件评分.

        生产环境: 用 FM 做自动评分。
        本实现: 基于启发式规则的简化版。
        """
        text = text.strip()

        if not text:
            return ComponentScore(
                component=component,
                score=1,
                feedback=f"{component.value}: completely missing",
            )

        # 启发式评分
        length = len(text)
        has_detail = any(
            marker in text.lower()
            for marker in ["e.g.", "example", "note", "important", "must", "should"]
        )
        has_structure = "\n" in text or "- " in text or ". " in text

        if length < 20:
            score = 2
            feedback = f"{component.value}: too brief ({length} chars)"
        elif length < 50 and not has_detail:
            score = 2
            feedback = f"{component.value}: brief and lacks detail"
        elif length < 80 and not has_detail:
            score = 3
            feedback = f"{component.value}: minimum viable"
        elif has_detail and has_structure:
            score = 5 if length > 150 else 4
            feedback = f"{component.value}: detailed and well-structured"
        else:
            score = 3
            feedback = f"{component.value}: acceptable but could be improved"

        return ComponentScore(component=component, score=score, feedback=feedback)

    def score_tool(self, tool: ToolDescription) -> list[ComponentScore]:
        """对完整 tool 描述做六维评分."""
        scores = []
        for comp in DescriptionComponent:
            text = tool.get_component(comp)
            scores.append(self.score_component(comp, text))
        return scores


# ---------------------------------------------------------------------------
# Smell Scanner
# ---------------------------------------------------------------------------


@dataclass
class SmellReport:
    """单个 tool 的 smell 报告."""

    tool_name: str
    component_scores: list[ComponentScore]
    smells: list[SmellType]
    overall_score: float  # 六维平均分
    is_smell_free: bool

    def summary(self) -> str:
        lines = [f"=== {self.tool_name} ==="]
        lines.append(f"  Overall score: {self.overall_score:.1f}/5.0")
        lines.append(f"  Smells: {len(self.smells)}")
        for s in self.smells:
            lines.append(f"    - {s.value}")
        lines.append(f"  Smell-free: {self.is_smell_free}")
        return "\n".join(lines)


class SmellScanner:
    """MCP Tool Description Smell Scanner."""

    def __init__(self, rubric: Optional[ScoringRubric] = None) -> None:
        self.rubric = rubric or ScoringRubric()

    def scan(self, tool: ToolDescription) -> SmellReport:
        """扫描单个 tool."""
        scores = self.rubric.score_tool(tool)
        smells = [
            COMPONENT_TO_SMELL[s.component]
            for s in scores
            if s.is_smelly()
        ]
        overall = sum(s.score for s in scores) / len(scores)
        is_smell_free = all(not s.is_smelly() for s in scores)

        return SmellReport(
            tool_name=tool.name,
            component_scores=scores,
            smells=smells,
            overall_score=overall,
            is_smell_free=is_smell_free,
        )

    def scan_batch(self, tools: list[ToolDescription]) -> list[SmellReport]:
        return [self.scan(t) for t in tools]

    def statistics(self, reports: list[SmellReport]) -> dict[str, Any]:
        """批量扫描统计."""
        total = len(reports)
        smell_free = sum(1 for r in reports if r.is_smell_free)
        avg_score = sum(r.overall_score for r in reports) / total if total else 0

        # 各 smell 患病率
        smell_counts: dict[SmellType, int] = {s: 0 for s in SmellType}
        for r in reports:
            for s in r.smells:
                smell_counts[s] += 1

        return {
            "total_tools": total,
            "smell_free_count": smell_free,
            "smell_free_pct": smell_free / total * 100 if total else 0,
            "avg_overall_score": avg_score,
            "smell_prevalence": {
                s.value: count / total * 100 if total else 0
                for s, count in smell_counts.items()
            },
        }


# ---------------------------------------------------------------------------
# Description Augmentor
# ---------------------------------------------------------------------------


class DescriptionAugmentor:
    """Tool 描述增强器.

    生产环境: 用 FM 自动生成缺失/低质量组件。
    本实现: 基于模板的简化版。
    """

    def augment(self, tool: ToolDescription, target_components: list[DescriptionComponent] | None = None) -> ToolDescription:
        """增强 tool 描述.

        Args:
            tool: 原始 tool 描述
            target_components: 要增强的组件 (None = 全部)
        """
        if target_components is None:
            target_components = list(DescriptionComponent)

        # 创建副本
        augmented = ToolDescription(
            name=tool.name,
            purpose=tool.purpose,
            guidelines=tool.guidelines,
            limitations=tool.limitations,
            parameter_explanation=tool.parameter_explanation,
            examples=tool.examples,
            return_value=tool.return_value,
        )

        for comp in target_components:
            current = tool.get_component(comp)
            if not current or len(current.strip()) < 30:
                enhanced = self._generate_component(tool.name, comp)
                setattr(augmented, comp.value, enhanced)

        return augmented

    def _generate_component(
        self, tool_name: str, component: DescriptionComponent
    ) -> str:
        """生成组件内容 (模板版)."""
        templates: dict[DescriptionComponent, str] = {
            DescriptionComponent.PURPOSE: (
                f"The {tool_name} tool retrieves or processes data for the specified input. "
                f"It is designed to handle standard requests efficiently and returns structured results."
            ),
            DescriptionComponent.GUIDELINES: (
                f"Use {tool_name} when you need to look up or process the specific entity identified by the required parameters. "
                f"Always provide valid, non-empty parameter values. "
                f"If the request involves multiple entities, invoke this tool separately for each one."
            ),
            DescriptionComponent.LIMITATIONS: (
                f"{tool_name} supports up to 100 results per call. "
                f"Rate limit: 60 requests per minute. "
                f"Data freshness is within 5 minutes for real-time endpoints."
            ),
            DescriptionComponent.PARAMETER_EXPLANATION: (
                f"Required parameters must be provided exactly as specified. "
                f"Optional parameters refine the output; omitting them returns default behavior. "
                f"All string parameters should use UTF-8 encoding."
            ),
            DescriptionComponent.EXAMPLES: (
                f'Example: {{"input": "example_value"}} -> '
                f'{{"status": "success", "data": {{...}}}}'
            ),
            DescriptionComponent.RETURN_VALUE: (
                f"Returns a JSON object containing: 'status' (success/error), "
                f"'data' (the requested payload), and 'metadata' (request timing and pagination info)."
            ),
        }
        return templates.get(component, f"[{component.value} for {tool_name}]")


# ---------------------------------------------------------------------------
# Tool Description Router
# ---------------------------------------------------------------------------


class DescriptionMode(Enum):
    """描述版本模式."""

    COMPACT = "compact"      # Purpose + Guidelines
    STANDARD = "standard"    # Purpose + Guidelines + Limitations + Parameter Explanation
    FULL = "full"            # All 6 components


class ToolDescriptionRouter:
    """运行时动态选择描述版本.

    根据 query 复杂度和场景选择最优组件组合。
    """

    MODE_COMPONENTS: dict[DescriptionMode, list[DescriptionComponent]] = {
        DescriptionMode.COMPACT: [
            DescriptionComponent.PURPOSE,
            DescriptionComponent.GUIDELINES,
        ],
        DescriptionMode.STANDARD: [
            DescriptionComponent.PURPOSE,
            DescriptionComponent.GUIDELINES,
            DescriptionComponent.LIMITATIONS,
            DescriptionComponent.PARAMETER_EXPLANATION,
        ],
        DescriptionMode.FULL: list(DescriptionComponent),
    }

    def __init__(self, augmentor: Optional[DescriptionAugmentor] = None) -> None:
        self.augmentor = augmentor or DescriptionAugmentor()
        self._mode_stats: dict[DescriptionMode, tuple[int, int]] = {
            DescriptionMode.COMPACT: (0, 0),
            DescriptionMode.STANDARD: (0, 0),
            DescriptionMode.FULL: (0, 0),
        }

    def route(self, tool: ToolDescription, query: str) -> ToolDescription:
        """根据 query 选择描述版本."""
        complexity = self._estimate_complexity(query)

        if complexity <= 1:
            mode = DescriptionMode.COMPACT
        elif complexity <= 2:
            mode = DescriptionMode.STANDARD
        else:
            mode = DescriptionMode.FULL

        components = self.MODE_COMPONENTS[mode]
        return self.augmentor.augment(tool, components)

    def _estimate_complexity(self, query: str) -> int:
        """估计 query 复杂度 (0-3)."""
        q = query.lower()
        complexity = 0

        # 多实体/多步骤指示
        if any(w in q for w in ["and", "then", "after", "before", "compare", "和", "以及", "对比"]):
            complexity += 1

        # 时间/条件约束
        if any(w in q for w in ["between", "range", "filter", "only", "except", "范围", "过滤", "之间"]):
            complexity += 1

        # 模糊/开放式
        if any(w in q for w in ["best", "recommend", "suggest", "what if", "推荐", "建议", "最好"]):
            complexity += 1

        return min(complexity, 3)

    def record_outcome(self, mode: DescriptionMode, success: bool) -> None:
        """记录模式执行结果."""
        passed, total = self._mode_stats[mode]
        self._mode_stats[mode] = (passed + int(success), total + 1)

    def mode_stats(self) -> dict[str, dict[str, float]]:
        """返回各模式统计."""
        return {
            mode.value: {
                "success_rate": passed / total if total else 0.0,
                "total_calls": total,
            }
            for mode, (passed, total) in self._mode_stats.items()
        }


# ---------------------------------------------------------------------------
# Demo: 母婴客服 MCP Tool 审核
# ---------------------------------------------------------------------------


def demo_tool_audit() -> None:
    """母婴客服 tool 描述审核 demo."""
    print("=== MCP Tool Description Smell Scanner Demo ===\n")

    # 定义内部 MCP tools (模拟真实状态)
    tools = [
        ToolDescription(
            name="order_lookup",
            purpose="查询订单状态",
            guidelines="",
            limitations="",
            parameter_explanation="order_id: 订单号",
            examples="",
            return_value="",
        ),
        ToolDescription(
            name="logistics_track",
            purpose="追踪物流信息",
            guidelines="输入物流单号获取当前位置和预计到达时间",
            limitations="",
            parameter_explanation="tracking_number: 物流单号",
            examples='{"tracking_number": "SF123456"}',
            return_value="返回物流状态、位置和预计到达时间",
        ),
        ToolDescription(
            name="size_recommend",
            purpose="",
            guidelines="",
            limitations="",
            parameter_explanation="",
            examples="",
            return_value="",
        ),
        ToolDescription(
            name="allergy_check",
            purpose="检查产品成分是否含有过敏原",
            guidelines="输入产品 ID 和用户已知的过敏原列表",
            limitations="仅支持数据库中已录入的产品成分信息",
            parameter_explanation=(
                "product_id: 产品唯一标识\n"
                "allergens: 用户已知的过敏原列表，如 ['milk', 'peanuts']"
            ),
            examples='{"product_id": "FORMULA-001", "allergens": ["milk"]}',
            return_value="返回过敏原匹配结果和风险等级 (low/medium/high)",
        ),
    ]

    scanner = SmellScanner()
    reports = scanner.scan_batch(tools)

    print("--- 逐 Tool 审核报告 ---")
    for report in reports:
        print(report.summary())
        print()

    # 统计
    stats = scanner.statistics(reports)
    print("--- 整体统计 ---")
    print(f"  总工具数: {stats['total_tools']}")
    print(f"  Smell-free: {stats['smell_free_count']} ({stats['smell_free_pct']:.1f}%)")
    print(f"  平均总分: {stats['avg_overall_score']:.2f}/5.0")
    print("  各 smell 患病率:")
    for smell, pct in sorted(stats['smell_prevalence'].items(), key=lambda x: -x[1]):
        print(f"    - {smell:30s}: {pct:5.1f}%")
    print()

    # 与论文对比
    print("--- 与论文对比 ---")
    print("  论文 (856 tools):")
    print("    Smell-free: 2.9%")
    print("    Unclear Purpose: 56.0%")
    print("    Missing Guidelines: 89.3%")
    print("    Unstated Limitations: 89.8%")
    print("    Opaque Parameters: 84.3%")
    print()


def demo_augmentation() -> None:
    """描述增强 demo."""
    print("=== Tool Description Augmentation Demo ===\n")

    tool = ToolDescription(
        name="order_lookup",
        purpose="查询订单状态",
        guidelines="",
        limitations="",
        parameter_explanation="order_id: 订单号",
        examples="",
        return_value="",
    )

    augmentor = DescriptionAugmentor()
    scanner = SmellScanner()

    print("--- 原始描述 ---")
    print(f"  Purpose: {tool.purpose[:50]}...")
    print(f"  Guidelines: {'(missing)' if not tool.guidelines else tool.guidelines[:50]}")
    print(f"  Limitations: {'(missing)' if not tool.limitations else tool.limitations[:50]}")
    print()

    # 扫描原始
    orig_report = scanner.scan(tool)
    print(f"原始评分: {orig_report.overall_score:.1f}/5.0, Smells: {len(orig_report.smells)}")
    print()

    # 增强
    augmented = augmentor.augment(tool)
    aug_report = scanner.scan(augmented)

    print("--- 增强后描述 ---")
    print(f"  Purpose: {augmented.purpose[:60]}...")
    print(f"  Guidelines: {augmented.guidelines[:60]}...")
    print(f"  Limitations: {augmented.limitations[:60]}...")
    print()
    print(f"增强后评分: {aug_report.overall_score:.1f}/5.0, Smells: {len(aug_report.smells)}")
    print()


def demo_routing() -> None:
    """动态路由 demo."""
    print("=== Tool Description Router Demo ===\n")

    tool = ToolDescription(
        name="order_lookup",
        purpose="查询订单状态",
        guidelines="输入订单号查询",
        limitations="仅支持 90 天内订单",
        parameter_explanation="order_id: 订单号",
        examples='{"order_id": "ORD1001"} -> {"status": "delivered"}',
        return_value="返回订单状态、产品和物流信息",
    )

    router = ToolDescriptionRouter()
    queries = [
        ("查询订单 ORD1001", DescriptionMode.COMPACT),
        ("查询订单 ORD1001 和 ORD1002 的状态", DescriptionMode.STANDARD),
        ("比较 ORD1001 和 ORD1002 的物流进度，并推荐最快的配送方式", DescriptionMode.FULL),
    ]

    print("--- Query 复杂度路由 ---")
    for query, expected_mode in queries:
        routed = router.route(tool, query)
        complexity = router._estimate_complexity(query)

        # 统计实际包含的组件
        components_present = sum(
            1 for comp in DescriptionComponent
            if getattr(routed, comp.value, "").strip()
        )

        print(f"  Query: {query}")
        print(f"    复杂度: {complexity}, 预期模式: {expected_mode.value}")
        print(f"    组件数: {components_present}/6")
        print()

    # 模拟执行结果记录
    router.record_outcome(DescriptionMode.COMPACT, True)
    router.record_outcome(DescriptionMode.COMPACT, True)
    router.record_outcome(DescriptionMode.STANDARD, True)
    router.record_outcome(DescriptionMode.STANDARD, False)
    router.record_outcome(DescriptionMode.FULL, True)

    print("--- 模式执行统计 ---")
    for mode, stats in router.mode_stats().items():
        print(f"  {mode:10s}: 成功率={stats['success_rate']*100:.0f}%, 调用={stats['total_calls']}")
    print()


# ---------------------------------------------------------------------------
# Test Pipeline
# ---------------------------------------------------------------------------


def test_pipeline() -> None:
    """Sanity checks."""

    # 1) ToolDescription
    td = ToolDescription(name="test", purpose="Test tool")
    assert td.name == "test"
    assert td.get_component(DescriptionComponent.PURPOSE) == "Test tool"
    assert td.get_component(DescriptionComponent.GUIDELINES) == ""

    # 2) ScoringRubric
    rubric = ScoringRubric()
    cs = rubric.score_component(DescriptionComponent.PURPOSE, "")
    assert cs.score == 1
    assert cs.is_smelly()

    cs2 = rubric.score_component(DescriptionComponent.PURPOSE, "This tool does X")
    assert cs2.score == 2  # too brief
    assert cs2.is_smelly()

    cs3 = rubric.score_component(
        DescriptionComponent.PURPOSE,
        "This tool retrieves historical stock prices. Use it when you need "
        "price data for a specific ticker. Note: supports up to 5 years of data.",
    )
    assert cs3.score >= 4
    assert not cs3.is_smelly()

    # 3) SmellScanner
    scanner = SmellScanner()
    tool = ToolDescription(name="bad_tool")  # 全空
    report = scanner.scan(tool)
    assert len(report.smells) == 6  # 全 smelly
    assert not report.is_smell_free
    assert report.overall_score == 1.0

    # 4) 良好 tool
    good_tool = ToolDescription(
        name="good_tool",
        purpose="Retrieves the current status and detailed information for a specific customer order. "
                "Use this tool when the user asks about order status, tracking, delivery, or order details. "
                "Note: the tool returns real-time data from the order management system.",
        guidelines="Use this when the user asks about their order status. "
                   "Always validate the order_id format before calling.",
        limitations="Only supports orders from the last 90 days. "
                    "Rate limit: 60 requests per minute.",
        parameter_explanation="order_id: string, required. The unique order identifier.",
        examples='{"order_id": "ORD123"} -> {"status": "delivered", "items": [...], "tracking": {...}}',
        return_value="Returns a JSON object containing: 'status' (string: pending/shipped/delivered), "
                     "'items' (array of product objects with name, quantity, and price), "
                     "and 'tracking_info' (object with carrier, tracking_number, and estimated_delivery). "
                     "Note: tracking_info is null if the order has not been shipped yet.",
    )
    good_report = scanner.scan(good_tool)
    assert good_report.is_smell_free
    assert good_report.overall_score >= 3.0
    assert len(good_report.smells) == 0

    # 5) Statistics
    stats = scanner.statistics([report, good_report])
    assert stats["total_tools"] == 2
    assert stats["smell_free_count"] == 1
    assert stats["smell_free_pct"] == 50.0

    # 6) DescriptionAugmentor
    augmentor = DescriptionAugmentor()
    augmented = augmentor.augment(tool)
    assert augmented.purpose  # 被填充了
    assert augmented.guidelines  # 被填充了

    # 部分增强
    partial = augmentor.augment(tool, [DescriptionComponent.PURPOSE])
    assert partial.purpose
    assert not partial.guidelines  # 未增强

    # 7) ToolDescriptionRouter
    router = ToolDescriptionRouter()
    assert router._estimate_complexity("simple query") == 0
    assert router._estimate_complexity("A and B") >= 1
    assert router._estimate_complexity("A and B between X and Y") >= 2

    routed = router.route(tool, "simple query")
    assert routed.purpose  # Compact mode 包含 Purpose

    # 8) Router stats
    router.record_outcome(DescriptionMode.COMPACT, True)
    router.record_outcome(DescriptionMode.COMPACT, False)
    s = router.mode_stats()
    assert s["compact"]["success_rate"] == 0.5
    assert s["compact"]["total_calls"] == 2

    # 9) SmellType 完整性
    assert len(SmellType) == 6
    assert len(DescriptionComponent) == 6

    # 10) COMPONENT_TO_SMELL 映射完整性
    assert len(COMPONENT_TO_SMELL) == 6
    for comp in DescriptionComponent:
        assert comp in COMPONENT_TO_SMELL

    print("[PASS] all assertions")


def main() -> None:
    test_pipeline()
    print()
    demo_tool_audit()
    demo_augmentation()
    demo_routing()


if __name__ == "__main__":
    main()

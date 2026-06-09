"""
Personalized Promotion Targeting — 个性化促销定向
异质性响应建模 + Knapsack 预算优化

纯 Python 标准库，无 sklearn/pandas 依赖
Python 3.14 兼容
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ─── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class UserSegment:
    """用户分群的促销响应特征"""
    segment_id: str
    segment_name: str
    n_users: int                    # 该分群用户数
    propensity_to_respond: float    # 绝对响应概率 P(buy | promo)
    baseline_response: float        # 无促销响应概率 P(buy | no promo)
    expected_value: float           # 响应后期望 LTV 增量（$）
    cost: float                     # 人均促销成本（$）

    @property
    def incremental_response(self) -> float:
        """增量响应概率 = 有促销 - 无促销（排除 Sure Things）"""
        return max(0.0, self.propensity_to_respond - self.baseline_response)

    @property
    def roi_per_dollar(self) -> float:
        """每美元促销成本的期望增量价值"""
        if self.cost <= 0:
            return 0.0
        return (self.incremental_response * self.expected_value) / self.cost

    @property
    def segment_type(self) -> str:
        """自动识别象限类型"""
        if self.incremental_response < 0.02:
            if self.baseline_response > 0.5:
                return "Sure Things"
            return "Lost Causes / Sleeping Dogs"
        if self.roi_per_dollar < 0.5:
            return "Low ROI Persuadables"
        return "Persuadables"


@dataclass
class AllocationResult:
    """分配结果"""
    segment: UserSegment
    allocated_users: int
    allocation_fraction: float      # 该分群中被分配的比例
    expected_incremental_ltv: float
    total_cost: float


# ─── 异质性响应建模 ──────────────────────────────────────────────────────────

class HeterogeneousResponseModeler:
    """用户分群异质性响应建模器"""

    def __init__(self) -> None:
        self._segments: list[UserSegment] = []

    def fit(self, segments: list[UserSegment]) -> "HeterogeneousResponseModeler":
        """加载分群响应数据"""
        self._segments = sorted(segments, key=lambda s: s.roi_per_dollar, reverse=True)
        return self

    def predict_response_probability(self, segment_id: str) -> dict[str, float]:
        """预测指定分群的响应概率分布"""
        for seg in self._segments:
            if seg.segment_id == segment_id:
                return {
                    "propensity_with_promo": seg.propensity_to_respond,
                    "baseline_no_promo": seg.baseline_response,
                    "incremental_lift": seg.incremental_response,
                    "roi_per_dollar": seg.roi_per_dollar,
                    "segment_type": seg.segment_type,
                }
        raise ValueError(f"分群 {segment_id!r} 不存在")

    def print_segment_report(self) -> None:
        """打印分群响应报告（按 ROI 降序）"""
        print(f"\n{'分群':>15} {'类型':>20} {'增量响应':>8} {'ROI/$':>8} {'人均成本':>8}")
        print("-" * 65)
        for seg in self._segments:
            print(
                f"{seg.segment_name:>15} {seg.segment_type:>20} "
                f"{seg.incremental_response:>8.3f} {seg.roi_per_dollar:>8.2f} "
                f"${seg.cost:>6.1f}"
            )

    @property
    def segments(self) -> list[UserSegment]:
        return self._segments


# ─── Knapsack 预算优化分配 ───────────────────────────────────────────────────

class PromotionAllocator:
    """
    在预算约束下最大化总增量 LTV 的促销分配器

    按 ROI/$ 降序贪心分配（Fractional Knapsack），
    自动过滤 Sure Things / Lost Causes（增量响应 < 阈值）
    """

    def __init__(
        self,
        budget: float,
        min_incremental_response: float = 0.02,
    ) -> None:
        self.budget = budget
        self.min_incremental_response = min_incremental_response

    def allocate(self, segments: list[UserSegment]) -> list[AllocationResult]:
        """贪心 Fractional Knapsack 分配"""
        eligible = [
            s for s in segments
            if s.incremental_response >= self.min_incremental_response and s.cost > 0
        ]
        eligible.sort(key=lambda s: s.roi_per_dollar, reverse=True)

        results: list[AllocationResult] = []
        remaining_budget = self.budget

        for seg in eligible:
            if remaining_budget <= 0:
                break

            max_cost_for_segment = seg.n_users * seg.cost
            allocatable_cost = min(max_cost_for_segment, remaining_budget)
            fraction = allocatable_cost / max_cost_for_segment
            allocated_users = int(seg.n_users * fraction)

            if allocated_users == 0:
                continue

            actual_cost = allocated_users * seg.cost
            incremental_ltv = allocated_users * seg.incremental_response * seg.expected_value

            results.append(AllocationResult(
                segment=seg,
                allocated_users=allocated_users,
                allocation_fraction=round(fraction, 4),
                expected_incremental_ltv=round(incremental_ltv, 2),
                total_cost=round(actual_cost, 2),
            ))
            remaining_budget -= actual_cost

        return results

    def max_roi_allocate(self, segments: list[UserSegment]) -> list[AllocationResult]:
        """同 allocate，别名方便调用"""
        return self.allocate(segments)

    def print_allocation_report(self, results: list[AllocationResult]) -> None:
        """打印分配结果"""
        total_ltv = sum(r.expected_incremental_ltv for r in results)
        total_cost = sum(r.total_cost for r in results)
        total_users = sum(r.allocated_users for r in results)

        print(f"\n{'分群':>15} {'分配用户':>8} {'分配比例':>8} {'成本':>10} {'增量LTV':>12}")
        print("-" * 60)
        for r in results:
            print(
                f"{r.segment.segment_name:>15} {r.allocated_users:>8,d} "
                f"{r.allocation_fraction:>8.1%} ${r.total_cost:>9,.0f} "
                f"${r.expected_incremental_ltv:>11,.0f}"
            )
        print("-" * 60)
        print(f"{'合计':>15} {total_users:>8,d} {'':>8} ${total_cost:>9,.0f} ${total_ltv:>11,.0f}")
        overall_roi = total_ltv / total_cost if total_cost > 0 else 0
        print(f"\n整体 ROI: {overall_roi:.2f}x （预算: ${self.budget:,.0f}，已用: ${total_cost:,.0f}）")


# ─── 测试 ────────────────────────────────────────────────────────────────────

def _build_test_segments() -> list[UserSegment]:
    """5 个用户群：覆盖全部4种象限类型"""
    return [
        UserSegment(
            segment_id="S1",
            segment_name="4-5月龄换购",
            n_users=5000,
            propensity_to_respond=0.45,
            baseline_response=0.08,
            expected_value=120.0,
            cost=15.0,
        ),
        UserSegment(
            segment_id="S2",
            segment_name="6月+自然升阶",
            n_users=8000,
            propensity_to_respond=0.80,
            baseline_response=0.79,  # Sure Things: 增量仅 0.01
            expected_value=80.0,
            cost=15.0,
        ),
        UserSegment(
            segment_id="S3",
            segment_name="高价值挽留",
            n_users=1000,
            propensity_to_respond=0.60,
            baseline_response=0.10,
            expected_value=500.0,
            cost=50.0,
        ),
        UserSegment(
            segment_id="S4",
            segment_name="中价值挽留",
            n_users=3000,
            propensity_to_respond=0.35,
            baseline_response=0.05,
            expected_value=150.0,
            cost=15.0,
        ),
        UserSegment(
            segment_id="S5",
            segment_name="低价值沉默",
            n_users=10000,
            propensity_to_respond=0.05,
            baseline_response=0.04,  # Lost Causes: 增量仅 0.01
            expected_value=30.0,
            cost=2.0,
        ),
    ]


def main() -> None:
    print("=" * 60)
    print("Loop 51-B: Personalized Promotion Targeting — 验证")
    print("=" * 60)

    segments = _build_test_segments()

    modeler = HeterogeneousResponseModeler()
    modeler.fit(segments)
    modeler.print_segment_report()

    s2_info = modeler.predict_response_probability("S2")
    assert s2_info["segment_type"] == "Sure Things", f"S2 应为 Sure Things: {s2_info['segment_type']}"
    print("\n✅ Sure Things 正确识别（6月+自然升阶用户）")

    s3_roi = modeler.predict_response_probability("S3")["roi_per_dollar"]
    s5_roi = modeler.predict_response_probability("S5")["roi_per_dollar"]
    assert s3_roi > s5_roi, f"高价值挽留 ROI 应高于低价值: {s3_roi:.3f} vs {s5_roi:.3f}"
    print(f"✅ ROI 排序正确: 高价值挽留 {s3_roi:.2f}x > 低价值沉默 {s5_roi:.2f}x")

    print("\n" + "=" * 60)
    print("预算约束下最优分配（预算: $80,000）")
    print("=" * 60)

    allocator = PromotionAllocator(budget=80_000.0, min_incremental_response=0.02)
    results = allocator.max_roi_allocate(segments)
    allocator.print_allocation_report(results)

    allocated_ids = {r.segment.segment_id for r in results}
    assert "S2" not in allocated_ids, "Sure Things（S2）不应被分配促销"
    print("\n✅ Sure Things（6月+用户）已被过滤，不发促销")

    total_cost = sum(r.total_cost for r in results)
    assert total_cost <= 80_000.0 + 1.0, f"超预算: ${total_cost:,.0f}"
    print(f"✅ 预算约束满足: 已用 ${total_cost:,.0f} ≤ $80,000")

    total_ltv = sum(r.expected_incremental_ltv for r in results)
    roi = total_ltv / total_cost if total_cost > 0 else 0
    assert roi > 1.0, f"整体 ROI 应 > 1.0: {roi:.2f}"
    print(f"✅ 整体 ROI 为正: {roi:.2f}x")
    print("\n✅ 所有验证通过 — Loop 51-B Personalized Promotion Targeting")


if __name__ == "__main__":
    main()

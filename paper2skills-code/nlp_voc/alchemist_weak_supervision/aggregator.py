"""多程序投票聚合

将多个 Label Function 的输出聚合为最终标签。
支持：多数投票、概率加权、置信度估计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from label_function import LabelFunction


@dataclass
class AggregationResult:
    """聚合结果"""

    text: str
    label: Optional[str]        # 最终标签（None = 无法确定）
    confidence: float           # 置信度 0-1
    vote_distribution: dict     # 各标签得票数
    programs_used: int          # 参与投票的程序数


class MajorityVoteAggregator:
    """多数投票聚合器

    简单策略：得票最多的标签胜出。
    当最高票 < 最低票数阈值时，返回 None（不确定）。
    """

    def __init__(self, min_votes: int = 2):
        self.min_votes = min_votes  # 最低票数要求

    def aggregate(
        self,
        text: str,
        label_functions: list[LabelFunction],
    ) -> AggregationResult:
        """对多个程序的输出进行多数投票"""
        votes: dict[str, int] = {}
        programs_used = 0

        for lf in label_functions:
            pred = lf(text)
            if pred is not None:  # 非弃权才计入
                votes[pred] = votes.get(pred, 0) + 1
                programs_used += 1

        if not votes:
            return AggregationResult(
                text=text, label=None, confidence=0.0,
                vote_distribution={}, programs_used=0,
            )

        max_votes = max(votes.values())
        winners = [label for label, count in votes.items() if count == max_votes]
        confidence = max_votes / programs_used if programs_used > 0 else 0

        if max_votes < self.min_votes:
            return AggregationResult(
                text=text, label=None, confidence=confidence,
                vote_distribution=votes, programs_used=programs_used,
            )

        if len(winners) == 1:
            return AggregationResult(
                text=text, label=winners[0], confidence=confidence,
                vote_distribution=votes, programs_used=programs_used,
            )

        return AggregationResult(
            text=text, label=None, confidence=confidence,
            vote_distribution=votes, programs_used=programs_used,
        )


class ProbabilisticAggregator:
    """概率加权聚合器

    考虑每个 label function 的历史准确率，加权投票。
    """

    def __init__(self, default_weight: float = 1.0):
        self.default_weight = default_weight

    def aggregate(
        self, text: str, label_functions: list[LabelFunction],
    ) -> AggregationResult:
        scores: dict[str, float] = {}
        programs_used = 0
        total_weight = 0

        for lf in label_functions:
            pred = lf(text)
            if pred is not None:
                weight = lf.accuracy if lf.accuracy > 0 else self.default_weight
                scores[pred] = scores.get(pred, 0) + weight
                programs_used += 1
                total_weight += weight

        if not scores:
            return AggregationResult(
                text=text, label=None, confidence=0.0,
                vote_distribution={}, programs_used=0,
            )

        max_score = max(scores.values())
        winners = [label for label, score in scores.items() if score == max_score]
        confidence = max_score / total_weight if total_weight > 0 else 0

        vote_dist = {
            label: sum(1 for lf in label_functions if lf(text) == label)
            for label in scores
        }

        if len(winners) == 1 and confidence >= 0.3:
            return AggregationResult(
                text=text, label=winners[0], confidence=confidence,
                vote_distribution=vote_dist, programs_used=programs_used,
            )

        return AggregationResult(
            text=text, label=None, confidence=confidence,
            vote_distribution=vote_dist, programs_used=programs_used,
        )


def test_aggregator():
    print("=" * 60)
    print("测试: Aggregator")
    print("=" * 60)

    lfs = [
        LabelFunction("lf1", lambda t: "过敏反应" if "过敏" in t else None),
        LabelFunction("lf2", lambda t: "过敏反应" if "红疹" in t else None),
        LabelFunction("lf3", lambda t: "过敏反应" if "发红" in t else None),
        LabelFunction("lf4", lambda t: "尺码偏差" if "尺码" in t else None),
    ]
    lfs[0].accuracy = 0.9
    lfs[1].accuracy = 0.85
    lfs[2].accuracy = 0.8
    lfs[3].accuracy = 0.7

    test_cases = [
        "宝宝用了后起红疹",
        "腰部一圈红红的，过敏了",
        "尺码偏小，腰贴过敏",
        "物流太慢了",
    ]

    print("\n--- 多数投票 ---")
    mv = MajorityVoteAggregator(min_votes=1)
    for text in test_cases:
        result = mv.aggregate(text, lfs)
        print(f"\n  文本: '{text}'")
        print(f"  结果: {result.label or '[不确定]'} (置信度: {result.confidence:.2f})")
        print(f"  投票分布: {result.vote_distribution}")

    print("\n--- 概率加权投票 ---")
    pa = ProbabilisticAggregator()
    for text in test_cases:
        result = pa.aggregate(text, lfs)
        print(f"\n  文本: '{text}'")
        print(f"  结果: {result.label or '[不确定]'} (置信度: {result.confidence:.2f})")

    print("\n" + "=" * 60)
    print("聚合器测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_aggregator()

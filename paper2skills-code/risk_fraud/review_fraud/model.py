"""
Review Fraud Detection — 虚假评论检测：行为特征 + 图分析
paper2skills-code: 19-风控反欺诈 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Review:
    review_id: str
    reviewer_id: str
    sku_id: str
    rating: int              # 1-5
    text: str
    verified_purchase: bool
    review_date: str
    helpful_votes: int = 0
    review_length: int = 0


@dataclass
class FraudSignal:
    signal_name: str
    score: float             # 0-1，越高越可疑
    description: str


@dataclass
class FraudDetectionResult:
    review_id: str
    fraud_probability: float
    fraud_signals: list[FraudSignal]
    verdict: str             # CLEAN / SUSPICIOUS / LIKELY_FRAUD


class ReviewFraudDetector:
    """虚假评论检测（多维度特征规则 + 简化版 GNN 思路）"""

    def _text_signals(self, review: Review) -> list[FraudSignal]:
        signals = []
        text = review.text.lower()

        # 过短评论（虚假评论往往文本极短）
        if len(review.text) < 20:
            signals.append(FraudSignal("short_text", 0.7, f"评论过短（{len(review.text)} 字符）"))

        # 重复性词汇检测（简化：连续空格比例）
        words = review.text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                signals.append(FraudSignal("repetitive_text", 0.6, f"词汇重复率高（独特词比 {unique_ratio:.0%}）"))

        # 极端评分 + 无内容
        if review.rating == 5 and len(review.text) < 30:
            signals.append(FraudSignal("perfect_rating_thin_text", 0.65, "满分评价但内容稀薄"))

        return signals

    def _behavior_signals(self, review: Review,
                          reviewer_history: list[Review]) -> list[FraudSignal]:
        signals = []
        if not reviewer_history:
            return signals

        # 短时间内大量评论
        if len(reviewer_history) > 10:
            signals.append(FraudSignal("high_review_volume", 0.7,
                           f"该用户评论数 {len(reviewer_history)} 条（异常多）"))

        # 未验证购买
        if not review.verified_purchase:
            signals.append(FraudSignal("unverified_purchase", 0.5, "非验证购买评论"))

        # 全是满分
        if reviewer_history:
            avg_rating = sum(r.rating for r in reviewer_history) / len(reviewer_history)
            if avg_rating >= 4.9:
                signals.append(FraudSignal("always_five_stars", 0.6, f"历史平均评分 {avg_rating:.1f}（疑似刷分）"))

        return signals

    def detect(self, review: Review,
               reviewer_history: list[Review] = None) -> FraudDetectionResult:
        reviewer_history = reviewer_history or []
        signals = (self._text_signals(review) +
                   self._behavior_signals(review, reviewer_history))

        if not signals:
            prob = 0.05
        else:
            prob = 1 - math.prod(1 - s.score for s in signals)
            prob = min(prob, 0.99)

        if prob < 0.3:
            verdict = "CLEAN"
        elif prob < 0.65:
            verdict = "SUSPICIOUS"
        else:
            verdict = "LIKELY_FRAUD"

        return FraudDetectionResult(
            review_id=review.review_id,
            fraud_probability=round(prob, 3),
            fraud_signals=signals,
            verdict=verdict,
        )


def run_review_fraud_demo():
    reviews = [
        Review("R001", "U001", "SKU-F1", 5, "Great product! My baby loves it.", True, "2026-01-10", 15, 32),
        Review("R002", "U002", "SKU-F1", 5, "good", False, "2026-01-11", 0, 4),
        Review("R003", "U003", "SKU-F1", 5, "good good good good good good good", False, "2026-01-11", 0, 37),
    ]
    history_fraud = [Review(f"H{i}", "U002", f"SKU-X{i}", 5, "good", False, "2026-01", 0, 4) for i in range(15)]

    detector = ReviewFraudDetector()
    print("=== 虚假评论检测 ===")
    for i, r in enumerate(reviews):
        hist = history_fraud if r.reviewer_id == "U002" else []
        result = detector.detect(r, hist)
        print(f"评论: {r.review_id} | 用户: {r.reviewer_id}")
        print(f"  欺诈概率: {result.fraud_probability:.1%} | 判定: {result.verdict}")
        for sig in result.fraud_signals:
            print(f"    ⚠️  {sig.signal_name}: {sig.description}")
        print()

    print("✅ 虚假评论检测演示完成")


if __name__ == "__main__":
    run_review_fraud_demo()

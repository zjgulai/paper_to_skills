"""评论质量评分完整流水线

整合: 特征提取 → 质量评分 → 虚假检测 → 过滤决策 → 报告输出

与 AutoTag 集成的便捷接口: `review_quality_pipeline()`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from feature_engine import FeatureExtractor
from quality_scorer import QualityReport, QualityScore, ReviewQualityScorer
from spam_detector import SpamDetectionResult, SpamDetector


@dataclass
class PipelineResult:
    """流水线单条结果"""

    text: str
    quality_score: QualityScore
    spam_detection: SpamDetectionResult
    final_decision: str       # "high_quality" | "low_quality" | "suspicious"
    action: str               # 建议操作

    def to_dict(self) -> dict:
        return {
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "quality_score": self.quality_score.to_dict(),
            "spam_detection": self.spam_detection.to_dict(),
            "final_decision": self.final_decision,
            "action": self.action,
        }


@dataclass
class PipelineReport:
    """流水线批量报告"""

    total_reviews: int
    high_quality: int
    low_quality: int
    suspicious: int
    avg_quality_score: float
    quality_report: QualityReport
    spam_summary: dict

    def to_dict(self) -> dict:
        return {
            "total_reviews": self.total_reviews,
            "high_quality": self.high_quality,
            "low_quality": self.low_quality,
            "suspicious": self.suspicious,
            "avg_quality_score": round(self.avg_quality_score, 1),
            "quality_report": self.quality_report.to_dict(),
            "spam_summary": self.spam_summary,
        }


class ReviewQualityPipeline:
    """评论质量评分完整流水线

    一键处理:
        1. 质量评分 (4 维度)
        2. 虚假检测 (5 规则)
        3. 综合决策 (质量分 + 虚假概率)
        4. 报告生成
    """

    def __init__(
        self,
        quality_threshold: float = 60.0,
        spam_threshold: float = 0.5,
        strict_mode: bool = False,
    ):
        self.scorer = ReviewQualityScorer(threshold=quality_threshold)
        self.spam_detector = SpamDetector(threshold=spam_threshold)
        self.strict_mode = strict_mode  # 严格模式: 可疑评论一律过滤

    def process(
        self,
        text: str,
        rating: Optional[int] = None,
    ) -> PipelineResult:
        """处理单条评论"""
        # 1. 质量评分
        quality = self.scorer.score(text, rating)

        # 2. 虚假检测
        spam = self.spam_detector.detect(text, rating)

        # 3. 综合决策
        decision, action = self._make_decision(quality, spam)

        return PipelineResult(
            text=text,
            quality_score=quality,
            spam_detection=spam,
            final_decision=decision,
            action=action,
        )

    def process_batch(
        self,
        texts: list[str],
        ratings: Optional[list[int]] = None,
    ) -> list[PipelineResult]:
        """批量处理"""
        results = []
        for i, text in enumerate(texts):
            rating = ratings[i] if ratings and i < len(ratings) else None
            results.append(self.process(text, rating))
        return results

    def build_report(self, results: list[PipelineResult]) -> PipelineReport:
        """构建报告"""
        total = len(results)
        high_q = sum(1 for r in results if r.final_decision == "high_quality")
        low_q = sum(1 for r in results if r.final_decision == "low_quality")
        suspicious = sum(1 for r in results if r.final_decision == "suspicious")

        quality_scores = [r.quality_score for r in results]
        quality_report = self.scorer.build_report(quality_scores)

        # 虚假检测汇总
        spam_types: dict[str, int] = {}
        for r in results:
            t = r.spam_detection.detection_type
            spam_types[t] = spam_types.get(t, 0) + 1

        return PipelineReport(
            total_reviews=total,
            high_quality=high_q,
            low_quality=low_q,
            suspicious=suspicious,
            avg_quality_score=sum(r.quality_score.overall_score for r in results) / total if total else 0,
            quality_report=quality_report,
            spam_summary=spam_types,
        )

    def filter_for_analysis(self, results: list[PipelineResult]) -> list[PipelineResult]:
        """过滤出适合下游分析的高质量评论"""
        return [r for r in results if r.final_decision == "high_quality"]

    def _make_decision(
        self,
        quality: QualityScore,
        spam: SpamDetectionResult,
    ) -> tuple[str, str]:
        """综合决策

        决策逻辑：
        - 可疑评论 → 优先过滤（无论质量分高低）
        - 高质量且非可疑 → 通过
        - 低质量且非可疑 → 过滤
        """
        # 严格模式：可疑一律过滤
        if self.strict_mode and spam.is_suspicious:
            return "suspicious", "标记为虚假评论，不入下游分析"

        # 非严格模式：可疑但质量分高的，降级为低质量
        if spam.is_suspicious:
            if quality.overall_score >= 70:
                return "suspicious", "质量分高但疑似虚假，需人工复核"
            return "suspicious", "疑似虚假且质量分低，直接过滤"

        # 正常评论按质量分判断
        if quality.is_high_quality:
            return "high_quality", "高质量评论，进入下游分析"
        return "low_quality", "低质量评论，过滤"


# ── 与 AutoTag 集成的便捷接口 ────────────────────────────────

def review_quality_pipeline(
    texts: list[str],
    ratings: Optional[list[int]] = None,
    quality_threshold: float = 60.0,
    strict_mode: bool = False,
) -> tuple[list[PipelineResult], PipelineReport]:
    """一键质量评估流水线

    Args:
        texts: 评论文本列表
        ratings: 评分列表（1-5星），可选
        quality_threshold: 质量分阈值（默认60）
        strict_mode: 严格模式（默认False）

    Returns:
        (结果列表, 批量报告)
    """
    pipeline = ReviewQualityPipeline(
        quality_threshold=quality_threshold,
        strict_mode=strict_mode,
    )
    results = pipeline.process_batch(texts, ratings)
    report = pipeline.build_report(results)
    return results, report


# ── 测试 ──────────────────────────────────────────────────────

def test_pipeline():
    print("=" * 60)
    print("测试: ReviewQualityPipeline")
    print("=" * 60)

    pipeline = ReviewQualityPipeline(quality_threshold=60.0)

    test_reviews = [
        # 高质量评论
        ("我买了这个纸尿裤三个月了，宝宝现在8个月。之前用过花王和帮宝适，"
         "这个吸水量明显更好，晚上用一片就够了，但是尺码偏大，建议买小一号。"
         "材质很柔软，没有异味，性价比不错。", 5),
        # 低质量-模板
        ("非常好用，强烈推荐！物流很快，很满意！", 5),
        # 矛盾评论
        ("五星好评！这是我用过最差的产品，质量太差了，后悔购买。", 5),
        # 中等质量
        ("纸尿裤不错，但是晚上有时候会漏，建议睡前换一次。", 4),
        # 极短
        ("还行", 3),
        # 虚假-夸张
        ("史上最差！！绝对不要买！！永远后悔！！", 1),
    ]

    texts = [r[0] for r in test_reviews]
    ratings = [r[1] for r in test_reviews]

    print("\n--- 流水线处理 ---")
    results, report = review_quality_pipeline(texts, ratings)

    for i, result in enumerate(results, 1):
        print(f"\n  [{i}] 决策: {result.final_decision}")
        print(f"      质量:{result.quality_score.overall_score:.0f}分 "
              f"虚假:{result.spam_detection.spam_probability:.0%}")
        print(f"      操作: {result.action}")

    print(f"\n--- 报告 ---")
    print(f"  总计: {report.total_reviews}")
    print(f"  高质量: {report.high_quality}")
    print(f"  低质量: {report.low_quality}")
    print(f"  可疑: {report.suspicious}")
    print(f"  平均分: {report.avg_quality_score:.1f}")

    # 验证
    assert report.high_quality >= 1, "应至少识别出1条高质量评论"
    assert report.suspicious >= 1, "应至少识别出1条可疑评论"
    assert report.low_quality >= 1, "应至少识别出1条低质量评论"

    # 过滤测试
    filtered = pipeline.filter_for_analysis(results)
    print(f"\n  过滤后适合分析: {len(filtered)} 条")

    print("\n" + "=" * 60)
    print("流水线测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_pipeline()

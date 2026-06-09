"""评论质量评分引擎

基于特征向量计算综合质量分 (0-100)，支持阈值过滤。
权重设计参考 AutoQual 的维度重要性分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from feature_engine import FeatureExtractor, QualityFeatures


@dataclass
class QualityScore:
    """单条评论质量评分结果"""

    text: str
    overall_score: float          # 综合质量分 0-100
    informativeness: float        # 信息丰富度 0-100
    consistency: float            # 评分一致性 0-100
    authenticity: float           # 语言真实性 0-100
    usefulness: float             # 实用性 0-100
    is_high_quality: bool         # 是否通过质量阈值
    threshold: float              # 使用的阈值
    reason: str                   # 质量判定原因

    def to_dict(self) -> dict:
        return {
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "overall_score": round(self.overall_score, 1),
            "informativeness": round(self.informativeness, 1),
            "consistency": round(self.consistency, 1),
            "authenticity": round(self.authenticity, 1),
            "usefulness": round(self.usefulness, 1),
            "is_high_quality": self.is_high_quality,
            "threshold": self.threshold,
            "reason": self.reason,
        }


@dataclass
class QualityReport:
    """批量质量评估报告"""

    total_reviews: int
    high_quality_count: int
    low_quality_count: int
    avg_score: float
    score_distribution: dict[str, int]  # 分数段分布
    dimension_averages: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "total_reviews": self.total_reviews,
            "high_quality_count": self.high_quality_count,
            "low_quality_count": self.low_quality_count,
            "avg_score": round(self.avg_score, 1),
            "score_distribution": self.score_distribution,
            "dimension_averages": {k: round(v, 2) for k, v in self.dimension_averages.items()},
        }


class ReviewQualityScorer:
    """评论质量评分器

    核心方法：
    1. 提取 4 维度特征
    2. 加权计算综合质量分
    3. 阈值判断 + 原因说明

    权重设计参考 AutoQual 论文中各维度对有用性预测的贡献度。
    """

    # 默认维度权重（可配置）
    DEFAULT_WEIGHTS = {
        "informativeness": 0.30,
        "consistency": 0.25,
        "authenticity": 0.25,
        "usefulness": 0.20,
    }

    # 质量阈值
    DEFAULT_THRESHOLD = 60.0

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.extractor = FeatureExtractor()
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self.threshold = threshold

        # 验证权重和为 1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"权重之和必须等于 1.0，当前为 {total}")

    def score(
        self,
        text: str,
        rating: Optional[int] = None,
    ) -> QualityScore:
        """对单条评论评分

        Args:
            text: 评论文本
            rating: 评分（1-5星），可选

        Returns:
            质量评分结果
        """
        features = self.extractor.extract(text, rating)

        # 计算各维度得分（0-100）
        info_score = features.informativeness * 100
        consistency_score = features.rating_text_consistency * 100
        # 一致性：矛盾时额外扣分
        if features.contradiction_score > 0.5:
            consistency_score = max(0, consistency_score - features.contradiction_score * 50)

        auth_score = features.authenticity * 100
        useful_score = features.usefulness * 100

        # 综合质量分（加权）
        overall = (
            info_score * self.weights["informativeness"] +
            consistency_score * self.weights["consistency"] +
            auth_score * self.weights["authenticity"] +
            useful_score * self.weights["usefulness"]
        )

        # 判定
        is_high_quality = overall >= self.threshold

        # 原因说明
        reason = self._generate_reason(
            overall, info_score, consistency_score, auth_score, useful_score,
            features.contradiction_score,
        )

        return QualityScore(
            text=text,
            overall_score=overall,
            informativeness=info_score,
            consistency=consistency_score,
            authenticity=auth_score,
            usefulness=useful_score,
            is_high_quality=is_high_quality,
            threshold=self.threshold,
            reason=reason,
        )

    def score_batch(
        self,
        texts: list[str],
        ratings: Optional[list[int]] = None,
    ) -> list[QualityScore]:
        """批量评分"""
        results = []
        for i, text in enumerate(texts):
            rating = ratings[i] if ratings and i < len(ratings) else None
            results.append(self.score(text, rating))
        return results

    def filter_high_quality(
        self,
        scores: list[QualityScore],
    ) -> list[QualityScore]:
        """过滤高质量评论"""
        return [s for s in scores if s.is_high_quality]

    def build_report(self, scores: list[QualityScore]) -> QualityReport:
        """构建批量评估报告"""
        total = len(scores)
        high_q = sum(1 for s in scores if s.is_high_quality)
        low_q = total - high_q

        # 分数段分布
        distribution: dict[str, int] = {
            "90-100": 0, "80-89": 0, "70-79": 0,
            "60-69": 0, "50-59": 0, "40-49": 0,
            "30-39": 0, "20-29": 0, "0-19": 0,
        }
        for s in scores:
            score = s.overall_score
            if score >= 90:
                distribution["90-100"] += 1
            elif score >= 80:
                distribution["80-89"] += 1
            elif score >= 70:
                distribution["70-79"] += 1
            elif score >= 60:
                distribution["60-69"] += 1
            elif score >= 50:
                distribution["50-59"] += 1
            elif score >= 40:
                distribution["40-49"] += 1
            elif score >= 30:
                distribution["30-39"] += 1
            elif score >= 20:
                distribution["20-29"] += 1
            else:
                distribution["0-19"] += 1

        # 维度平均分
        dim_avg = {
            "informativeness": sum(s.informativeness for s in scores) / total if total else 0,
            "consistency": sum(s.consistency for s in scores) / total if total else 0,
            "authenticity": sum(s.authenticity for s in scores) / total if total else 0,
            "usefulness": sum(s.usefulness for s in scores) / total if total else 0,
        }

        return QualityReport(
            total_reviews=total,
            high_quality_count=high_q,
            low_quality_count=low_q,
            avg_score=sum(s.overall_score for s in scores) / total if total else 0,
            score_distribution=distribution,
            dimension_averages=dim_avg,
        )

    def _generate_reason(
        self,
        overall: float,
        info: float,
        consistency: float,
        auth: float,
        useful: float,
        contradiction: float,
    ) -> str:
        """生成质量判定原因"""
        reasons = []

        if overall >= 80:
            reasons.append("高质量评论")
        elif overall >= 60:
            reasons.append("中等质量评论")
        else:
            reasons.append("低质量评论")

        # 找出最低维度
        dims = {
            "信息丰富度": info,
            "评分一致性": consistency,
            "语言真实性": auth,
            "实用性": useful,
        }
        weakest_dim = min(dims, key=dims.get)
        weakest_score = dims[weakest_dim]

        if weakest_score < 40:
            reasons.append(f"{weakest_dim}不足({weakest_score:.0f}分)")

        if contradiction > 0.5:
            reasons.append(f"评分与文本矛盾({contradiction:.0%})")

        return "; ".join(reasons)


# ── 测试 ──────────────────────────────────────────────────────

def test_quality_scorer():
    print("=" * 60)
    print("测试: ReviewQualityScorer")
    print("=" * 60)

    scorer = ReviewQualityScorer(threshold=60.0)

    test_cases = [
        # (名称, 文本, 评分, 预期质量)
        (
            "高质量-详细",
            "我买了这个纸尿裤三个月了，宝宝现在8个月。之前用过花王和帮宝适，"
            "这个吸水量明显更好，晚上用一片就够了，但是尺码偏大，建议买小一号。"
            "材质很柔软，没有异味，性价比不错。",
            5, True,
        ),
        (
            "低质量-模板",
            "非常好用，强烈推荐！物流很快，很满意！",
            5, False,
        ),
        (
            "矛盾-五星差评",
            "五星好评！这是我用过最差的产品，质量太差了，后悔购买。",
            5, False,
        ),
        (
            "极短-无信息",
            "还行",
            3, False,
        ),
        (
            "中等-有建议",
            "纸尿裤不错，但是晚上有时候会漏，建议睡前换一次。",
            4, True,
        ),
    ]

    print("\n--- 单条评分 ---")
    for name, text, rating, expected_hq in test_cases:
        result = scorer.score(text, rating)
        status = "✓" if result.is_high_quality == expected_hq else "✗"
        print(f"\n  {status} [{name}] 综合:{result.overall_score:.1f}分 "
              f"{'高质量' if result.is_high_quality else '低质量'}")
        print(f"      信息:{result.informativeness:.0f} 一致:{result.consistency:.0f} "
              f"真实:{result.authenticity:.0f} 实用:{result.usefulness:.0f}")
        print(f"      原因: {result.reason}")
        assert result.is_high_quality == expected_hq, (
            f"{name}: 期望{'高质量' if expected_hq else '低质量'}，"
            f"实际{'高质量' if result.is_high_quality else '低质量'}"
        )

    # 批量测试
    print("\n--- 批量评分 + 报告 ---")
    texts = [t[1] for t in test_cases]
    ratings = [t[2] for t in test_cases]
    scores = scorer.score_batch(texts, ratings)
    report = scorer.build_report(scores)

    print(f"  总计: {report.total_reviews}")
    print(f"  高质量: {report.high_quality_count}")
    print(f"  低质量: {report.low_quality_count}")
    print(f"  平均分: {report.avg_score:.1f}")
    print(f"  分数分布: {report.score_distribution}")

    # 过滤
    filtered = scorer.filter_high_quality(scores)
    print(f"  过滤后: {len(filtered)} 条高质量评论")

    print("\n" + "=" * 60)
    print("质量评分测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_quality_scorer()

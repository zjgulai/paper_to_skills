"""虚假评论检测模块

基于 AutoQual 的可解释特征 + BHeIPCoRT 的评分-文本一致性，
检测虚假评论（刷单/水军/模板评论）。

检测维度：
1. 模板化检测 — 固定句式、重复结构
2. 评分矛盾检测 — 评分与文本情感方向相反
3. 极端情感检测 — 过度夸张的言辞
4. 批量模式检测 — 多条评论之间的相似度
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpamDetectionResult:
    """虚假评论检测结果"""

    text: str
    is_suspicious: bool          # 是否可疑
    spam_probability: float      # 虚假概率 0-1
    detection_type: str          # 检测类型
    confidence: float            # 置信度
    reasons: list[str]           # 检测原因

    def to_dict(self) -> dict:
        return {
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "is_suspicious": self.is_suspicious,
            "spam_probability": round(self.spam_probability, 2),
            "detection_type": self.detection_type,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
        }


class SpamDetector:
    """虚假评论检测器

    多规则组合检测，覆盖常见虚假评论模式。
    """

    # 模板化短语（虚假评论高频使用）
    TEMPLATE_PHRASES = [
        "非常好用，强烈推荐",
        "物流很快，很满意",
        "质量不错，值得购买",
        "很好，下次还会再来",
        "商品很好，物流快",
        "非常满意，五星好评",
        "产品不错，推荐购买",
        "物超所值，好评",
        "物美价廉，满意",
        "服务态度好，好评",
    ]

    # 过度夸张词
    EXAGGERATION_WORDS = [
        "史上最", "绝对", "永远", "一定", "肯定", "百分百",
        "perfect", "amazing", "incredible", "unbelievable",
    ]

    # 虚假评论常见开头
    SUSPICIOUS_STARTS = [
        "我是老顾客",
        "已经买了好几次",
        "回购很多次",
        "这是第N次购买",
        "一直用这个牌子",
    ]

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def detect(self, text: str, rating: Optional[int] = None) -> SpamDetectionResult:
        """检测单条评论是否可疑"""
        reasons = []
        signals = []

        # 1. 模板化检测
        template_score = self._detect_template(text)
        if template_score > 0:
            reasons.append(f"匹配模板短语({template_score:.0%})")
            signals.append(template_score)

        # 2. 评分矛盾检测
        if rating is not None:
            contradiction_score = self._detect_contradiction(text, rating)
            if contradiction_score > 0:
                reasons.append(f"评分与文本矛盾({contradiction_score:.0%})")
                signals.append(contradiction_score)

        # 3. 极端情感检测
        exaggeration_score = self._detect_exaggeration(text)
        if exaggeration_score > 0:
            reasons.append(f"过度夸张({exaggeration_score:.0%})")
            signals.append(exaggeration_score)

        # 4. 可疑开头检测
        suspicious_start_score = self._detect_suspicious_start(text)
        if suspicious_start_score > 0:
            reasons.append(f"虚假评论常见开头({suspicious_start_score:.0%})")
            signals.append(suspicious_start_score)

        # 5. 重复句式检测
        repetition_score = self._detect_repetition(text)
        if repetition_score > 0:
            reasons.append(f"重复句式({repetition_score:.0%})")
            signals.append(repetition_score)

        # 综合评估
        if signals:
            # 取最高信号 + 其他信号的平均值
            max_signal = max(signals)
            avg_signal = sum(signals) / len(signals)
            spam_prob = max_signal * 0.7 + avg_signal * 0.3
        else:
            spam_prob = 0.0

        is_suspicious = spam_prob >= self.threshold

        # 检测类型
        if spam_prob >= 0.8:
            detection_type = "高置信度虚假"
        elif spam_prob >= 0.5:
            detection_type = "疑似虚假"
        else:
            detection_type = "正常"

        return SpamDetectionResult(
            text=text,
            is_suspicious=is_suspicious,
            spam_probability=spam_prob,
            detection_type=detection_type,
            confidence=max(signals) if signals else 0.0,
            reasons=reasons,
        )

    def detect_batch(
        self,
        texts: list[str],
        ratings: Optional[list[int]] = None,
    ) -> list[SpamDetectionResult]:
        """批量检测"""
        results = []
        for i, text in enumerate(texts):
            rating = ratings[i] if ratings and i < len(ratings) else None
            results.append(self.detect(text, rating))
        return results

    # ── 检测规则 ──────────────────────────────────────────────

    def _detect_template(self, text: str) -> float:
        """检测模板化短语"""
        for phrase in self.TEMPLATE_PHRASES:
            if phrase in text:
                return 1.0
        return 0.0

    def _detect_contradiction(self, text: str, rating: int) -> float:
        """检测评分与文本矛盾"""
        # 简单情感词统计
        pos_words = {"好", "不错", "满意", "喜欢", "推荐", "优质", "完美", "棒",
                     "good", "great", "excellent", "love", "perfect"}
        neg_words = {"差", "不好", "失望", "垃圾", "糟糕", "后悔", "差评", "退货",
                     "bad", "terrible", "worst", "hate", "disappointed"}

        pos_count = sum(1 for w in pos_words if w in text)
        neg_count = sum(1 for w in neg_words if w in text)

        # 五星好评但有很多负面词
        if rating >= 4 and neg_count >= 2:
            return min(neg_count * 0.3, 1.0)
        # 一星差评但有很多正面词
        elif rating <= 2 and pos_count >= 2:
            return min(pos_count * 0.3, 1.0)
        return 0.0

    def _detect_exaggeration(self, text: str) -> float:
        """检测过度夸张"""
        count = sum(1 for w in self.EXAGGERATION_WORDS if w in text)
        # 连续感叹号检测
        exclamation_count = text.count("！！") + text.count("!!!")
        total = count + exclamation_count
        return min(total * 0.25, 1.0)

    def _detect_suspicious_start(self, text: str) -> float:
        """检测可疑开头"""
        for start in self.SUSPICIOUS_STARTS:
            if text.startswith(start) or ("，" in text and text.split("，")[0].startswith(start)):
                return 0.6
        return 0.0

    def _detect_repetition(self, text: str) -> float:
        """检测重复句式"""
        sentences = text.replace("！", "。").replace("？", "。").split("。")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        if len(sentences) < 2:
            return 0.0

        # 检测完全重复的句子
        unique = set(sentences)
        if len(unique) < len(sentences):
            return 0.8

        # 检测开头重复
        starts = [s[:4] for s in sentences]
        repeated = len(starts) - len(set(starts))
        if repeated >= 2:
            return min(repeated * 0.3, 0.8)

        return 0.0


# ── 测试 ──────────────────────────────────────────────────────

def test_spam_detector():
    print("=" * 60)
    print("测试: SpamDetector")
    print("=" * 60)

    detector = SpamDetector(threshold=0.5)

    test_cases = [
        # (名称, 文本, 评分, 预期可疑)
        (
            "正常-详细",
            "我买了这个纸尿裤三个月了，宝宝现在8个月。之前用过花王和帮宝适。",
            5, False,
        ),
        (
            "模板评论",
            "非常好用，强烈推荐！物流很快，很满意！",
            5, True,
        ),
        (
            "评分矛盾",
            "五星好评！这是我用过最差的产品，质量太差了，后悔购买。",
            5, True,
        ),
        (
            "夸张评论",
            "史上最差！！绝对不要买！！永远后悔！！",
            1, True,
        ),
        (
            "可疑开头",
            "我是老顾客了，一直用这个牌子，非常好用，强烈推荐。",
            5, True,
        ),
        (
            "重复句式",
            "质量很好。质量很好。物流很快。物流很快。",
            5, True,
        ),
    ]

    print("\n--- 单条检测 ---")
    for name, text, rating, expected in test_cases:
        result = detector.detect(text, rating)
        status = "✓" if result.is_suspicious == expected else "✗"
        print(f"\n  {status} [{name}] 虚假概率:{result.spam_probability:.0%} "
              f"{'可疑' if result.is_suspicious else '正常'}")
        print(f"      类型: {result.detection_type}")
        print(f"      原因: {', '.join(result.reasons)}")
        assert result.is_suspicious == expected, (
            f"{name}: 期望{'可疑' if expected else '正常'}，"
            f"实际{'可疑' if result.is_suspicious else '正常'}"
        )

    print("\n" + "=" * 60)
    print("虚假评论检测测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_spam_detector()

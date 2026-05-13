"""评论质量特征提取引擎

基于 AutoQual (EMNLP 2025) 的可解释特征框架，提取 4 维度质量特征：
1. 信息丰富度 (Informativeness)
2. 评分一致性 (Rating-Text Consistency)
3. 语言真实性 (Authenticity)
4. 实用性 (Usefulness)

生产环境建议:
    - 使用预训练语言模型编码文本，提升特征精度
    - 使用标注数据训练 XGBoost 分类器，替代规则权重
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityFeatures:
    """评论质量特征向量"""

    # 信息丰富度
    text_length: int
    aspect_count: int
    detail_density: float
    vocabulary_diversity: float
    structural_markers: int
    informativeness: float

    # 评分一致性
    rating_text_consistency: float
    sentiment_rating_alignment: float
    contradiction_score: float

    # 语言真实性
    first_person_ratio: float
    template_pattern_score: float
    emotional_extremity: float
    temporal_specificity: float
    authenticity: float

    # 实用性
    has_comparison: bool
    has_advice: bool
    has_usage_scenario: bool
    usefulness: float

    def to_vector(self) -> list[float]:
        """转换为特征向量"""
        return [
            self.informativeness,
            self.rating_text_consistency,
            self.authenticity,
            self.usefulness,
        ]

    def to_dict(self) -> dict:
        return {
            "informativeness": round(self.informativeness, 3),
            "rating_text_consistency": round(self.rating_text_consistency, 3),
            "authenticity": round(self.authenticity, 3),
            "usefulness": round(self.usefulness, 3),
            "details": {
                "text_length": self.text_length,
                "aspect_count": self.aspect_count,
                "detail_density": round(self.detail_density, 3),
                "vocabulary_diversity": round(self.vocabulary_diversity, 3),
                "first_person_ratio": round(self.first_person_ratio, 3),
                "template_pattern_score": round(self.template_pattern_score, 3),
                "emotional_extremity": round(self.emotional_extremity, 3),
                "contradiction_score": round(self.contradiction_score, 3),
                "has_comparison": self.has_comparison,
                "has_advice": self.has_advice,
            },
        }


class FeatureExtractor:
    """评论质量特征提取器

    从文本 + 评分中提取 4 维度可解释特征。
    """

    # 细节词（表示具体描述）
    DETAIL_WORDS = {
        "尺寸", "大小", "颜色", "重量", "材质", "厚度", "宽度", "长度",
        "分钟", "小时", "秒", "天", "周", "月",
        "厘米", "毫米", "克", "千克", "毫升", "升",
        "第一次", "第二次", "上次", "下次", "之前", "之后",
        "左边", "右边", "上面", "下面", "中间",
    }

    # 结构化标记词
    STRUCTURAL_MARKERS = {
        "但是", "不过", "然而", "虽然", "尽管", "可是",
        "因为", "所以", "因此", "于是", "由于", "结果",
        "首先", "其次", "然后", "最后", "第一", "第二",
        "比如", "例如", "像", "比如说",
        "总结", "总的来说", "一句话", "总之",
    }

    # 对比词
    COMPARISON_WORDS = {
        "比", "相比", "对比", "不如", "胜过", "超过", "不如",
        "其他", "别的", "之前用", "以前用", "换过", "试过",
        "better", "worse", "than", "compared", "versus", "vs",
    }

    # 建议词
    ADVICE_WORDS = {
        "建议", "推荐", "值得", "不适合", "适合", "最好", "不要",
        "注意", "小心", "记得", "一定要", "没必要",
        "recommend", "suggest", "advise", "worth", "avoid",
    }

    # 使用场景词
    USAGE_SCENARIO_WORDS = {
        "晚上", "白天", "出门", "在家", "上班", "睡觉", "洗澡",
        "夏天", "冬天", "出门", "旅游", "出差", "运动时",
        "at night", "during the day", "at work", "while sleeping",
    }

    # 时间具体性词
    TEMPORAL_WORDS = {
        "上周", "上个月", "去年", "今天", "昨天", "三天前",
        "买了", "收到货", "用了", "使用", "一段时间后",
        "last week", "a month ago", "yesterday", "today",
    }

    # 情感词（用于一致性检测）
    POS_WORDS = {
        "好", "不错", "满意", "喜欢", "推荐", "优质", "完美", "棒", "赞",
        "good", "great", "excellent", "love", "perfect", "amazing",
    }
    NEG_WORDS = {
        "差", "不好", "失望", "垃圾", "糟糕", "后悔", "差评", "退货",
        "bad", "terrible", "worst", "hate", "disappointed", "awful",
    }

    # 极端词（过度夸张 = 低真实性）
    EXTREME_WORDS = {
        "最差", "最烂", "史上", "绝对", "永远", "一定", "肯定",
        "worst ever", "absolute", "definitely", "always", "never",
    }

    # 模板化短语（批量生成的虚假评论常用）
    TEMPLATE_PHRASES = [
        "非常好用，强烈推荐",
        "物流很快，很满意",
        "质量不错，值得购买",
        "很好，下次还会再来",
        "商品很好，物流快",
        "非常满意，五星好评",
        "产品不错，推荐购买",
    ]

    def __init__(self):
        pass

    def extract(self, text: str, rating: Optional[int] = None) -> QualityFeatures:
        """提取单条评论的质量特征

        Args:
            text: 评论文本
            rating: 评分（1-5星），可选

        Returns:
            质量特征向量
        """
        # 信息丰富度
        info_features = self._extract_informativeness(text)

        # 评分一致性
        consistency_features = self._extract_consistency(text, rating)

        # 语言真实性
        auth_features = self._extract_authenticity(text)

        # 实用性
        useful_features = self._extract_usefulness(text)

        return QualityFeatures(
            text_length=info_features["text_length"],
            aspect_count=info_features["aspect_count"],
            detail_density=info_features["detail_density"],
            vocabulary_diversity=info_features["vocabulary_diversity"],
            structural_markers=info_features["structural_markers"],
            informativeness=info_features["informativeness"],
            rating_text_consistency=consistency_features["rating_text_consistency"],
            sentiment_rating_alignment=consistency_features["sentiment_rating_alignment"],
            contradiction_score=consistency_features["contradiction_score"],
            first_person_ratio=auth_features["first_person_ratio"],
            template_pattern_score=auth_features["template_pattern_score"],
            emotional_extremity=auth_features["emotional_extremity"],
            temporal_specificity=auth_features["temporal_specificity"],
            authenticity=auth_features["authenticity"],
            has_comparison=useful_features["has_comparison"],
            has_advice=useful_features["has_advice"],
            has_usage_scenario=useful_features["has_usage_scenario"],
            usefulness=useful_features["usefulness"],
        )

    # ── 信息丰富度 ──────────────────────────────────────────────

    def _extract_informativeness(self, text: str) -> dict:
        """提取信息丰富度特征"""
        text_len = len(text)

        # 1. 文本长度分（对数缩放，避免超长文本过度得分）
        length_score = min(math.log1p(text_len) / math.log1p(200), 1.0)

        # 2. 方面覆盖数（简化：检测不同产品属性的提及）
        aspect_keywords = {
            "质量", "材质", "手感", "外观", "颜色", "尺寸", "大小",
            "功能", "效果", "性能", "速度", "噪音", "气味", "味道",
            "价格", "性价比", "包装", "物流", "客服", "售后",
            "使用", "体验", "感受", "设计", "细节", "做工",
        }
        aspect_count = sum(1 for kw in aspect_keywords if kw in text)
        aspect_score = min(aspect_count / 5.0, 1.0)

        # 3. 细节密度（具体名词/数量词密度）
        detail_count = sum(1 for w in self.DETAIL_WORDS if w in text)
        detail_density = min(detail_count / 3.0, 1.0)

        # 4. 词汇多样性（Type-Token Ratio 近似）
        words = self._segment_words(text)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0.0
        # 中文短文本 TTR 天然较高，做适当调整
        vocab_diversity = min(ttr * 2.0, 1.0)

        # 5. 结构化标记数
        struct_count = sum(1 for w in self.STRUCTURAL_MARKERS if w in text)
        struct_score = min(struct_count / 2.0, 1.0)

        # 综合信息丰富度（加权平均）
        informativeness = (
            length_score * 0.25 +
            aspect_score * 0.30 +
            detail_density * 0.20 +
            vocab_diversity * 0.15 +
            struct_score * 0.10
        )

        return {
            "text_length": text_len,
            "aspect_count": aspect_count,
            "detail_density": detail_density,
            "vocabulary_diversity": vocab_diversity,
            "structural_markers": struct_count,
            "informativeness": informativeness,
        }

    # ── 评分一致性 ──────────────────────────────────────────────

    def _extract_consistency(self, text: str, rating: Optional[int]) -> dict:
        """提取评分一致性特征"""
        if rating is None:
            # 无评分时，一致性设为中性
            return {
                "rating_text_consistency": 0.5,
                "sentiment_rating_alignment": 0.5,
                "contradiction_score": 0.0,
            }

        # 1. 文本情感极性
        pos_count = sum(1 for w in self.POS_WORDS if w in text)
        neg_count = sum(1 for w in self.NEG_WORDS if w in text)

        if pos_count > neg_count:
            text_sentiment = 1  # 正面
        elif neg_count > pos_count:
            text_sentiment = -1  # 负面
        else:
            text_sentiment = 0  # 中性

        # 2. 评分方向（1-2星=负面, 3星=中性, 4-5星=正面）
        if rating <= 2:
            rating_sentiment = -1
        elif rating == 3:
            rating_sentiment = 0
        else:
            rating_sentiment = 1

        # 3. 一致性得分（越一致越高）
        if text_sentiment == rating_sentiment:
            consistency = 1.0
        elif text_sentiment == 0 or rating_sentiment == 0:
            consistency = 0.6  # 中性偏差，轻微扣分
        else:
            consistency = 0.0  # 完全矛盾

        # 4. 矛盾检测（五星+负面词 / 一星+正面词 = 强矛盾）
        contradiction = 0.0
        if rating >= 4 and neg_count > 0:
            contradiction = min(neg_count * 0.3, 1.0)
        elif rating <= 2 and pos_count > 0:
            contradiction = min(pos_count * 0.3, 1.0)

        # 5. 情感-评分对齐度（评分的极端程度与情感强度是否匹配）
        sentiment_intensity = abs(pos_count - neg_count)
        rating_extremity = abs(rating - 3) / 2.0  # 1星或5星 = 1.0
        alignment = 1.0 - abs(sentiment_intensity / 5.0 - rating_extremity)
        alignment = max(0.0, min(1.0, alignment))

        return {
            "rating_text_consistency": consistency,
            "sentiment_rating_alignment": alignment,
            "contradiction_score": contradiction,
        }

    # ── 语言真实性 ──────────────────────────────────────────────

    def _extract_authenticity(self, text: str) -> dict:
        """提取语言真实性特征"""
        # 1. 第一人称密度（真实评论多用"我"）
        first_person_count = text.count("我") + text.count("我们") + text.count("my") + text.count("I ")
        words = self._segment_words(text)
        first_person_ratio = first_person_count / len(words) if words else 0.0
        first_person_score = min(first_person_ratio * 10.0, 1.0)

        # 2. 模板化模式检测
        template_score = 0.0
        for phrase in self.TEMPLATE_PHRASES:
            if phrase in text:
                template_score = 1.0
                break
        # 也检测重复句式
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 2:
            # 检测句子开头重复
            starts = [s[:3] for s in sentences]
            repeated_starts = len(starts) - len(set(starts))
            if repeated_starts >= 2:
                template_score = max(template_score, 0.5)

        # 3. 情感极端度（过度夸张 = 低真实性）
        extreme_count = sum(1 for w in self.EXTREME_WORDS if w in text)
        emotional_extremity = min(extreme_count * 0.4, 1.0)

        # 4. 时间/场景具体性
        temporal_count = sum(1 for w in self.TEMPORAL_WORDS if w in text)
        temporal_specificity = min(temporal_count / 2.0, 1.0)

        # 综合真实性
        authenticity = (
            first_person_score * 0.25 +
            (1.0 - template_score) * 0.30 +
            (1.0 - emotional_extremity) * 0.25 +
            temporal_specificity * 0.20
        )

        return {
            "first_person_ratio": first_person_ratio,
            "template_pattern_score": template_score,
            "emotional_extremity": emotional_extremity,
            "temporal_specificity": temporal_specificity,
            "authenticity": authenticity,
        }

    # ── 实用性 ─────────────────────────────────────────────────

    def _extract_usefulness(self, text: str) -> dict:
        """提取实用性特征"""
        # 1. 是否包含对比信息
        has_comparison = any(w in text for w in self.COMPARISON_WORDS)

        # 2. 是否包含建议
        has_advice = any(w in text for w in self.ADVICE_WORDS)

        # 3. 是否包含使用场景
        has_scenario = any(w in text for w in self.USAGE_SCENARIO_WORDS)

        # 综合实用性
        usefulness = (
            (0.35 if has_comparison else 0.0) +
            (0.40 if has_advice else 0.0) +
            (0.25 if has_scenario else 0.0)
        )
        # 纯情感宣泄（无对比/建议/场景）且长度短的 = 低实用性
        if not has_comparison and not has_advice and not has_scenario:
            if len(text) < 20:
                usefulness = 0.1
            else:
                usefulness = 0.2

        return {
            "has_comparison": has_comparison,
            "has_advice": has_advice,
            "has_usage_scenario": has_scenario,
            "usefulness": usefulness,
        }

    # ── 工具方法 ──────────────────────────────────────────────

    def _segment_words(self, text: str) -> list[str]:
        """简单中文分词（2-3字滑动窗口）"""
        words = []
        for length in range(2, 4):
            for i in range(len(text) - length + 1):
                word = text[i:i + length]
                if any("\u4e00" <= c <= "\u9fff" for c in word):
                    words.append(word)
        # 也加入英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        words.extend(english_words)
        return words


# ── 测试 ──────────────────────────────────────────────────────

def test_feature_extractor():
    print("=" * 60)
    print("测试: FeatureExtractor")
    print("=" * 60)

    extractor = FeatureExtractor()

    # 高质量评论（信息丰富、一致、真实、实用）
    high_quality = (
        "我买了这个纸尿裤三个月了，宝宝现在8个月。"
        "之前用过花王和帮宝适，这个吸水量明显更好，"
        "晚上用一片就够了，但是尺码偏大，建议买小一号。"
        "材质很柔软，没有异味，性价比不错。"
    )

    # 低质量评论（无信息、不一致、模板化）
    low_quality = "非常好用，强烈推荐！物流很快，很满意！"

    # 矛盾评论（五星好评但负面文字）
    contradictory = (
        "五星好评！这是我用过最差的产品，"
        "质量太差了，后悔购买，完全不值这个价。"
    )

    # 虚假评论（模板化、无第一人称、无细节）
    spam = "质量不错，值得购买。商品很好，物流快。"

    test_cases = [
        ("高质量", high_quality, 5),
        ("低质量", low_quality, 5),
        ("矛盾", contradictory, 5),
        ("虚假", spam, 5),
        ("极短", "还行", 3),
    ]

    print("\n--- 特征提取对比 ---")
    for name, text, rating in test_cases:
        features = extractor.extract(text, rating)
        print(f"\n[{name}] {'=' * 40}")
        print(f"  文本: {text[:30]}...")
        print(f"  信息丰富度: {features.informativeness:.2f}")
        print(f"  评分一致性: {features.rating_text_consistency:.2f}")
        print(f"  语言真实性: {features.authenticity:.2f}")
        print(f"  实用性: {features.usefulness:.2f}")

    # 验证
    hq = extractor.extract(high_quality, 5)
    lq = extractor.extract(low_quality, 5)

    assert hq.informativeness > lq.informativeness, "高质量评论信息丰富度应更高"
    assert hq.authenticity > lq.authenticity, "高质量评论真实性应更高"
    assert hq.usefulness > lq.usefulness, "高质量评论实用性应更高"

    # 矛盾检测
    con = extractor.extract(contradictory, 5)
    assert con.contradiction_score > 0.5, "矛盾评论应有高矛盾分"
    assert con.rating_text_consistency < 0.5, "矛盾评论一致性应低"

    print("\n" + "=" * 60)
    print("特征提取测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_extractor()

"""情感强度量化模块

将 AutoTag 的粗粒度情感(-1/0/+1)量化为细粒度强度分数(-5~+5)，
支持两种输入模式：
- 概率模式：有 pred_probs 时从分布熵/概率差推导
- 规则模式：无 pred_probs 时从情感词密度/否定词/程度副词推导

输出直接对接 Kano 映射和 iReFeed 优先级排序。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from model import PredictionResult


@dataclass
class IntensityResult:
    """单条情感强度量化结果"""

    text: str
    raw_sentiment: int          # 原始情感 -1/0/+1
    intensity: float            # 强度分数 -5.0 ~ +5.0
    intensity_level: str        # "extreme_neg" | "strong_neg" | "moderate_neg" |
                                # "slight_neg" | "neutral" | "slight_pos" |
                                # "moderate_pos" | "strong_pos" | "extreme_pos"
    confidence: float           # 强度置信度 0-1
    reasoning: str = ""         # 强度推导原因

    def to_dict(self) -> dict:
        return {
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "raw_sentiment": self.raw_sentiment,
            "intensity": round(self.intensity, 2),
            "intensity_level": self.intensity_level,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning,
        }


class SentimentIntensityQuantifier:
    """情感强度量化器

    核心洞察：粗粒度情感(-1/0/+1)不足以支撑业务决策。
    知道"负面"不够，需要知道"多负面"——是轻微不满(-1.5)还是极度愤怒(-4.8)。
    强度分数直接影响 Kano 分类和优先级排序。
    """

    # 程度副词（影响情感强度倍数）
    INTENSIFIERS = {
        "非常": 1.5, "特别": 1.4, "十分": 1.4, "极其": 1.6,
        "太": 1.3, "很": 1.2, "相当": 1.2, "比较": 1.1,
        "有点": 0.7, "稍微": 0.6, "略微": 0.6, "一点点": 0.5,
        "根本": 1.5, "完全": 1.4, "绝对": 1.5, "实在": 1.2,
    }
    # 否定词（翻转或削弱情感）
    NEGATORS = {
        "不", "没", "无", "非", "别", "未", "莫", "勿",
        "not", "no", "never", "without",
    }
    # 极端情感词（直接映射到高分）
    EXTREME_POS = {
        "完美", "无可挑剔", "极度满意", "amazing", "perfect",
        "outstanding", "exceptional", "life-changing",
    }
    EXTREME_NEG = {
        "垃圾", "骗子", "坑人", "丧尽天良", "忍无可忍",
        "disaster", "nightmare", "scam", "fraud", "terrible",
    }

    def __init__(self, sentiment_dict: Optional[dict] = None):
        """初始化

        Args:
            sentiment_dict: 自定义情感词典，格式 {"pos": set(), "neg": set()}
        """
        if sentiment_dict:
            self.pos_words = sentiment_dict.get("pos", set())
            self.neg_words = sentiment_dict.get("neg", set())
        else:
            self.pos_words = {
                "好", "不错", "满意", "喜欢", "推荐", "优质", "舒适", "柔软",
                "好用", "完美", "棒", "赞", "给力", "贴心", "值得",
                "good", "great", "excellent", "love", "perfect",
                "comfortable", "soft", "nice", "amazing", "awesome",
            }
            self.neg_words = {
                "差", "不好", "失望", "垃圾", "糟糕", "后悔", "差评", "退货",
                "漏", "硬", "粗糙", "臭", "异味", "坏", "破", "烂", "慢",
                "贵", "坑", "骗", "假", "恶心", "难受", "疼", "痛",
                "bad", "terrible", "worst", "hate", "disappointed",
                "poor", "rough", "hard", "smell", "broken", "leak",
                "expensive", "horrible", "awful",
            }

    def quantify(
        self,
        text: str,
        prediction: PredictionResult,
        pred_probs: Optional[np.ndarray] = None,
    ) -> IntensityResult:
        """量化单条文本的情感强度

        Args:
            text: 原始文本
            prediction: AutoTag 预测结果
            pred_probs: 可选的预测概率分布

        Returns:
            情感强度量化结果
        """
        if pred_probs is not None:
            return self._quantify_from_probs(text, prediction, pred_probs)
        return self._quantify_from_rules(text, prediction)

    def quantify_batch(
        self,
        texts: list[str],
        predictions: list[PredictionResult],
        pred_probs: Optional[np.ndarray] = None,
    ) -> list[IntensityResult]:
        """批量量化"""
        results = []
        for i, (text, pred) in enumerate(zip(texts, predictions)):
            probs = pred_probs[i] if pred_probs is not None and i < len(pred_probs) else None
            results.append(self.quantify(text, pred, probs))
        return results

    # ── 概率模式 ──────────────────────────────────────────────

    def _quantify_from_probs(
        self,
        text: str,
        prediction: PredictionResult,
        pred_probs: np.ndarray,
    ) -> IntensityResult:
        """从预测概率分布推导强度

        方法：
        1. softmax 概率的"确定性"决定强度上限
        2. 与次高概率的差距决定置信度
        3. 最终强度 = 极性 × 确定性 × 缩放因子
        """
        probs = np.array(pred_probs).flatten()
        if len(probs) == 0 or not np.isfinite(probs).all():
            return self._quantify_from_rules(text, prediction)

        # 归一化
        probs = probs / probs.sum() if probs.sum() > 0 else probs

        # 最大概率（确定性）
        max_prob = float(np.max(probs))
        # 次大概率
        sorted_probs = np.sort(probs)[::-1]
        second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        # 概率差距（置信度）
        margin = max_prob - second_prob

        # 确定性映射到强度：max_prob ∈ [0.33, 1.0] → scale ∈ [0.5, 1.0]
        # 三分类均匀分布时 max_prob = 0.33，确定性最低
        certainty = (max_prob - 1.0 / len(probs)) / (1.0 - 1.0 / len(probs))
        certainty = max(certainty, 0.0)

        # 极性 × 最大强度(5) × 确定性 × 置信度调整
        polarity = float(prediction.sentiment)
        if polarity == 0:
            # 中性但有概率偏向 → 轻微偏向
            pos_idx = getattr(prediction, "_pos_idx", -1)
            neg_idx = getattr(prediction, "_neg_idx", -1)
            if pos_idx >= 0 and neg_idx >= 0 and pos_idx < len(probs) and neg_idx < len(probs):
                polarity = 1 if probs[pos_idx] > probs[neg_idx] else -1
                certainty = abs(probs[pos_idx] - probs[neg_idx])

        intensity = polarity * 5.0 * certainty * (0.7 + 0.3 * margin)
        intensity = max(-5.0, min(5.0, intensity))

        # 置信度 = 确定性 × (0.5 + 0.5 × 概率差距)
        confidence = certainty * (0.5 + 0.5 * margin)

        level = self._intensity_to_level(intensity)
        reasoning = (
            f"概率模式: 最大概率={max_prob:.2f}, 次高={second_prob:.2f}, "
            f"确定性={certainty:.2f}, 差距={margin:.2f}"
        )

        return IntensityResult(
            text=text,
            raw_sentiment=prediction.sentiment,
            intensity=intensity,
            intensity_level=level,
            confidence=confidence,
            reasoning=reasoning,
        )

    # ── 规则模式 ──────────────────────────────────────────────

    def _quantify_from_rules(
        self,
        text: str,
        prediction: PredictionResult,
    ) -> IntensityResult:
        """从规则推导强度（无 pred_probs 时的降级方案）

        方法：
        1. 统计情感词数量
        2. 检测程度副词（放大/缩小）
        3. 检测否定词（翻转/削弱）
        4. 检测极端词（直接映射到边界）
        5. 综合计算强度
        """
        text_lower = text.lower()

        # 1. 情感词计数（按长度降序，优先匹配长词避免重复计数）
        pos_sorted = sorted(self.pos_words, key=len, reverse=True)
        neg_sorted = sorted(self.neg_words, key=len, reverse=True)
        pos_hits = [w for w in pos_sorted if w in text_lower]
        neg_hits = [w for w in neg_sorted if w in text_lower]

        # 2. 极端词检测
        for w in self.EXTREME_POS:
            if w in text_lower:
                return IntensityResult(
                    text=text,
                    raw_sentiment=1,
                    intensity=4.5,
                    intensity_level="extreme_pos",
                    confidence=0.9,
                    reasoning=f"检测到极端正面词: '{w}'",
                )
        for w in self.EXTREME_NEG:
            if w in text_lower:
                return IntensityResult(
                    text=text,
                    raw_sentiment=-1,
                    intensity=-4.5,
                    intensity_level="extreme_neg",
                    confidence=0.9,
                    reasoning=f"检测到极端负面词: '{w}'",
                )

        # 3. 计算基础情感得分
        pos_score = len(pos_hits)
        neg_score = len(neg_hits)

        # 4. 程度副词倍数
        multiplier = 1.0
        for word, factor in self.INTENSIFIERS.items():
            if word in text_lower:
                multiplier = max(multiplier, factor)

        # 5. 否定词调整
        negator_count = sum(1 for w in self.NEGATORS if w in text_lower)
        if negator_count > 0:
            # 否定词削弱情感或翻转（简化：削弱30%）
            multiplier *= max(0.3, 1.0 - negator_count * 0.3)

        # 6. 综合强度
        if pos_score > neg_score:
            base = min(pos_score * 1.5, 4.0)
            polarity = 1
        elif neg_score > pos_score:
            base = -min(neg_score * 1.5, 4.0)
            polarity = -1
        else:
            base = 0.0
            polarity = 0

        intensity = base * multiplier
        intensity = max(-5.0, min(5.0, intensity))

        # 7. 置信度 = 情感词数量 / (情感词数量 + 1) × 程度副词调整
        total_hits = pos_score + neg_score
        confidence = min(total_hits / (total_hits + 1), 0.9) * min(multiplier, 1.2) / 1.2

        level = self._intensity_to_level(intensity)
        reasoning = (
            f"规则模式: 正面词×{pos_score}, 负面词×{neg_score}, "
            f"程度倍数={multiplier:.1f}, 否定词×{negator_count}"
        )

        return IntensityResult(
            text=text,
            raw_sentiment=polarity,
            intensity=intensity,
            intensity_level=level,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _intensity_to_level(self, intensity: float) -> str:
        """将连续强度映射到离散等级"""
        if intensity >= 4.0:
            return "extreme_pos"
        elif intensity >= 2.5:
            return "strong_pos"
        elif intensity >= 1.0:
            return "moderate_pos"
        elif intensity > 0.2:
            return "slight_pos"
        elif intensity <= -4.0:
            return "extreme_neg"
        elif intensity <= -2.5:
            return "strong_neg"
        elif intensity <= -1.0:
            return "moderate_neg"
        elif intensity < -0.2:
            return "slight_neg"
        return "neutral"

    def summarize(
        self,
        results: list[IntensityResult],
    ) -> dict:
        """汇总强度分布统计"""
        if not results:
            return {"count": 0}

        intensities = [r.intensity for r in results]
        levels = {}
        for r in results:
            levels[r.intensity_level] = levels.get(r.intensity_level, 0) + 1

        return {
            "count": len(results),
            "mean_intensity": round(sum(intensities) / len(intensities), 2),
            "min_intensity": round(min(intensities), 2),
            "max_intensity": round(max(intensities), 2),
            "extreme_ratio": sum(1 for i in intensities if abs(i) >= 3.5) / len(intensities),
            "level_distribution": levels,
        }


# ── 测试 ──────────────────────────────────────────────────────

def test_sentiment_intensity():
    print("=" * 60)
    print("测试: SentimentIntensityQuantifier")
    print("=" * 60)

    quantifier = SentimentIntensityQuantifier()

    # 测试用例：从轻微不满到极度愤怒
    test_cases = [
        ("还行吧", 0),                          # 中性
        ("有点硬", -1),                         # 轻微负面
        ("比较失望，不太好用", -1),              # 中度负面
        ("非常失望，质量太差", -1),              # 强负面
        ("垃圾产品，骗子商家", -1),              # 极端负面
        ("还不错", 1),                          # 轻微正面
        ("很满意，推荐购买", 1),                 # 中度正面
        ("完美，无可挑剔", 1),                   # 极端正面
        ("不便宜，但还可以", 0),                 # 否定词削弱
        ("不是特别软", 0),                       # 双重否定削弱
    ]

    print("\n--- 规则模式测试 ---")
    for text, expected_sentiment in test_cases:
        pred = PredictionResult(sentiment=expected_sentiment)
        result = quantifier.quantify(text, pred)
        print(
            f"  [{result.intensity:+.1f}] {result.intensity_level:14s} "
            f"(置信度:{result.confidence:.2f}) {text[:20]}..."
        )

    # 概率模式测试
    print("\n--- 概率模式测试 ---")
    # 模拟三分类 softmax: [负, 中, 正]
    prob_cases = [
        ("测试文本", -1, np.array([0.8, 0.15, 0.05])),   # 高置信度负面
        ("测试文本", -1, np.array([0.4, 0.35, 0.25])),   # 低置信度
        ("测试文本", 1, np.array([0.05, 0.1, 0.85])),    # 高置信度正面
    ]
    for text, sent, probs in prob_cases:
        pred = PredictionResult(sentiment=sent)
        result = quantifier.quantify(text, pred, probs)
        print(
            f"  [{result.intensity:+.1f}] {result.intensity_level:14s} "
            f"(置信度:{result.confidence:.2f}) probs={probs}"
        )

    # 汇总测试
    print("\n--- 汇总统计 ---")
    all_preds = [PredictionResult(sentiment=s) for _, s in test_cases]
    results = quantifier.quantify_batch(
        [t for t, _ in test_cases],
        all_preds,
    )
    summary = quantifier.summarize(results)
    print(f"  平均强度: {summary['mean_intensity']}")
    print(f"  强度范围: [{summary['min_intensity']}, {summary['max_intensity']}]")
    print(f"  极端情绪占比: {summary['extreme_ratio']:.1%}")
    print(f"  等级分布: {summary['level_distribution']}")

    # 断言验证
    extreme_neg = [r for r in results if r.intensity_level == "extreme_neg"]
    assert len(extreme_neg) >= 1, "应检测到至少1个极端负面"
    assert extreme_neg[0].intensity < -4.0, "极端负面应<-4.0"

    extreme_pos = [r for r in results if r.intensity_level == "extreme_pos"]
    assert len(extreme_pos) >= 1, "应检测到至少1个极端正面"
    assert extreme_pos[0].intensity > 4.0, "极端正面应>4.0"

    print("\n" + "=" * 60)
    print("情感强度量化测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_sentiment_intensity()

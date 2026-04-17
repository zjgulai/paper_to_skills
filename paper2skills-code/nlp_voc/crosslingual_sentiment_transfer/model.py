"""
Cross-Lingual Sentiment Transfer for Low-Resource Markets

基于论文 "LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis
with LLM Data Augmentation" (Šmíd et al., ACL 2025)

核心思路：利用英语丰富标注数据，通过多语言预训练模型 + LLM数据增强，
实现零样本/低样本跨语言情感分析迁移。

母婴出海场景：覆盖欧美(英/西/法/德)、东南亚(泰/越/印尼)、中东(阿)等多语言市场，
用一套框架分析所有市场的用户评论情感。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. 数据模型
# ---------------------------------------------------------------------------

@dataclass
class SentimentExample:
    """情感分析样本"""
    text: str
    language: str
    aspects: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PseudoLabel:
    """伪标签样本"""
    original_text: str
    generated_text: str
    language: str
    aspects: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# 2. 合成数据集 (模拟英语标注 + 目标语言无标注数据)
# ---------------------------------------------------------------------------

ENGLISH_TRAIN: list[SentimentExample] = [
    SentimentExample(
        "The quality is great but shipping was slow",
        "en",
        aspects=[
            {"term": "quality", "sentiment": "positive"},
            {"term": "shipping", "sentiment": "negative"},
        ],
    ),
    SentimentExample(
        "Excellent design, my baby loves it",
        "en",
        aspects=[
            {"term": "design", "sentiment": "positive"},
            {"term": "baby", "sentiment": "positive"},
        ],
    ),
    SentimentExample(
        "Overpriced and poor customer service",
        "en",
        aspects=[
            {"term": "price", "sentiment": "negative"},
            {"term": "customer service", "sentiment": "negative"},
        ],
    ),
    SentimentExample(
        "Soft material, safe for newborn skin",
        "en",
        aspects=[
            {"term": "material", "sentiment": "positive"},
            {"term": "safety", "sentiment": "positive"},
        ],
    ),
    SentimentExample(
        "Packaging was damaged but product is fine",
        "en",
        aspects=[
            {"term": "packaging", "sentiment": "negative"},
            {"term": "product", "sentiment": "positive"},
        ],
    ),
]

# 目标语言无标注评论 (模拟真实多语言输入)
TARGET_UNLABELED: dict[str, list[str]] = {
    "es": [  # 西班牙语 - 欧美/拉美
        "La calidad es excelente pero el envío tardó mucho",
        "Diseño precioso, mi bebé lo adora",
        "Caro y servicio al cliente deficiente",
        "Material suave, seguro para piel de recién nacido",
        "El empaque llegó dañado pero el producto está bien",
    ],
    "th": [  # 泰语 - 东南亚
        "คุณภาพดีมาก แต่ส่งช้า",
        "ดีไซน์น่ารัก เด็กชอบมาก",
        "แพง และบริการลูกค้าแย่",
        "เนื้อผ้านิ่ม ปลอดภัยสำหรับผิวเด็ก",
        "แพคเกจเสียหาย แต่สินค้าโอเค",
    ],
    "ar": [  # 阿拉伯语 - 中东
        "الجودة ممتازة لكن الشحن كان بطيئاً",
        "تصميم رائع، طفلي يحبه",
        "مبالغ فيه والخدمة سيئة",
        "قماش ناعم، آمن لبشرة الرضع",
        "التغليف تالف لكن المنتج بخير",
    ],
}

# LLM 生成回放缓冲 (模拟根据伪标签生成目标语言句子)
GENERATED_CORPUS: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
    "es": [
        ("La calidad es fantástica aunque el envío fue lento", [
            {"term": "calidad", "sentiment": "positive"},
            {"term": "envío", "sentiment": "negative"},
        ]),
        ("El diseño es adorable y mi bebé disfruta usarlo", [
            {"term": "diseño", "sentiment": "positive"},
            {"term": "bebé", "sentiment": "positive"},
        ]),
        ("Precio excesivo y atención al cliente decepcionante", [
            {"term": "precio", "sentiment": "negative"},
            {"term": "atención al cliente", "sentiment": "negative"},
        ]),
    ],
    "th": [
        ("คุณภาพดีเยี่ยม แต่จัดส่งช้า", [
            {"term": "คุณภาพ", "sentiment": "positive"},
            {"term": "จัดส่ง", "sentiment": "negative"},
        ]),
        ("ดีไซน์น่ารักมาก ลูกชอบใช้", [
            {"term": "ดีไซน์", "sentiment": "positive"},
            {"term": "ลูก", "sentiment": "positive"},
        ]),
        ("ราคาแพงเกินไป บริการแย่", [
            {"term": "ราคา", "sentiment": "negative"},
            {"term": "บริการ", "sentiment": "negative"},
        ]),
    ],
    "ar": [
        ("الجودة رائعة لكن التوصيل كان بطيئاً", [
            {"term": "جودة", "sentiment": "positive"},
            {"term": "توصيل", "sentiment": "negative"},
        ]),
        ("التصميم جميل جداً وطفلي يحبه كثيراً", [
            {"term": "تصميم", "sentiment": "positive"},
            {"term": "طفلي", "sentiment": "positive"},
        ]),
        ("سعر مرتفع وخدمة العملاء سيئة", [
            {"term": "سعر", "sentiment": "negative"},
            {"term": "خدمة العملاء", "sentiment": "negative"},
        ]),
    ],
}


# ---------------------------------------------------------------------------
# 3. 跨语言特征提取器 (基于 XLM-R 思想的多语言词嵌入模拟)
# ---------------------------------------------------------------------------

class MultilingualFeatureExtractor:
    """模拟 XLM-R 风格的多语言特征提取。"""

    # 多语言方面词表 (模拟共享语义空间)
    ASPECT_VOCAB: dict[str, list[str]] = {
        "quality": ["quality", "calidad", "คุณภาพ", "الجودة"],
        "shipping": ["shipping", "envío", "จัดส่ง", "الشحن", "التوصيل"],
        "design": ["design", "diseño", "ดีไซน์", "التصميم"],
        "baby": ["baby", "bebé", "ลูก", "طفلي", "الرضيع"],
        "price": ["price", "precio", "ราคา", "السعر"],
        "service": ["customer service", "servicio al cliente", "บริการ", "خدمة العملاء"],
        "material": ["material", "material", "เนื้อผ้า", "قماش"],
        "safety": ["safety", "seguro", "ปลอดภัย", "آمن"],
        "packaging": ["packaging", "empaque", "แพคเกจ", "التغليف"],
        "product": ["product", "producto", "สินค้า", "المنتج"],
    }

    POSITIVE_WORDS: dict[str, list[str]] = {
        "en": ["great", "excellent", "fantastic", "soft", "safe", "fine", "adorable", "loves", "enjoys"],
        "es": ["excelente", "fantástica", "precioso", "suave", "seguro", "bien", "adorable", "adora", "disfruta"],
        "th": ["ดี", "ดีมาก", "ดีเยี่ยม", "นิ่ม", "ปลอดภัย", "โอเค", "น่ารัก", "ชอบ"],
        "ar": ["ممتازة", "رائعة", "رائع", "جميل", "ناعم", "آمن", "بخير", "يحبه", "يحب"],
    }

    NEGATIVE_WORDS: dict[str, list[str]] = {
        "en": ["slow", "poor", "overpriced", "damaged", "deficient", "disappointing", "bad"],
        "es": ["tardó", "lento", "deficiente", "dañado", "decepcionante", "caro", "malo"],
        "th": ["ช้า", "แย่", "แพง", "เสียหาย", "แย่", "ไม่ดี"],
        "ar": ["بطيئاً", "بطيء", "سيئة", "تالف", "سيء", "مرتفع"],
    }

    def extract_features(self, text: str, language: str) -> NDArray[np.float64]:
        """提取文本的多语言特征向量。"""
        text_lower = text.lower()

        # 方面存在特征
        aspect_feats = []
        for concept, translations in self.ASPECT_VOCAB.items():
            matched = any(t.lower() in text_lower for t in translations)
            aspect_feats.append(1.0 if matched else 0.0)

        # 情感词特征
        pos_words = self.POSITIVE_WORDS.get(language, self.POSITIVE_WORDS["en"])
        neg_words = self.NEGATIVE_WORDS.get(language, self.NEGATIVE_WORDS["en"])

        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        total_sentiment_words = pos_count + neg_count

        sentiment_ratio = (pos_count - neg_count) / max(total_sentiment_words, 1)

        # 文本长度特征
        length_feat = min(len(text.split()) / 20.0, 1.0)

        features = aspect_feats + [sentiment_ratio, length_feat]
        return np.array(features, dtype=np.float64)

    def n_features(self) -> int:
        return len(self.ASPECT_VOCAB) + 2


# ---------------------------------------------------------------------------
# 4. ABSA 分类器 (模拟在英语上训练、跨语言零样本预测)
# ---------------------------------------------------------------------------

class CrossLingualABSAModel:
    """跨语言 ABSA 模型：英语训练 → 目标语言零样本预测。"""

    def __init__(self) -> None:
        self.extractor = MultilingualFeatureExtractor()
        self.weights: NDArray[np.float64] | None = None
        self.bias: float = 0.0

    def _example_to_label(self, example: SentimentExample) -> NDArray[np.float64]:
        """将样本转换为监督标签：每个方面-情感对作为一个目标。"""
        label = np.zeros(self.extractor.n_features(), dtype=np.float64)
        text_lower = example.text.lower()

        # 填充方面存在
        for i, (concept, translations) in enumerate(self.extractor.ASPECT_VOCAB.items()):
            matched = any(t.lower() in text_lower for t in translations)
            if matched:
                # 情感值: +1 positive, -1 negative
                aspect_info = next(
                    (a for a in example.aspects if any(t.lower() in text_lower for t in translations)),
                    None,
                )
                if aspect_info:
                    sentiment_val = 1.0 if aspect_info["sentiment"] == "positive" else -1.0
                    label[i] = sentiment_val

        # 整体情感
        pos = sum(1 for w in self.extractor.POSITIVE_WORDS[example.language] if w in text_lower)
        neg = sum(1 for w in self.extractor.NEGATIVE_WORDS[example.language] if w in text_lower)
        label[-2] = (pos - neg) / max(pos + neg, 1)
        label[-1] = min(len(example.text.split()) / 20.0, 1.0)

        return label

    def fit(self, examples: list[SentimentExample]) -> None:
        """在源语言数据上训练（解析解模拟）。"""
        X = np.array([self.extractor.extract_features(ex.text, ex.language) for ex in examples])
        Y = np.array([self._example_to_label(ex) for ex in examples])

        # 简单线性回归拟合
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        params = np.linalg.lstsq(X_with_bias, Y, rcond=None)[0]

        self.bias = float(params[0, -2])  # 使用整体情感维度的bias
        self.weights = params[1:, -2]     # 使用整体情感维度的weights

    def predict(self, text: str, language: str) -> dict[str, Any]:
        """预测目标语言文本的情感和方面。"""
        features = self.extractor.extract_features(text, language)

        if self.weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # 整体情感预测
        sentiment_score = float(np.dot(self.weights, features) + self.bias)
        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))

        if sentiment_score > 0.2:
            overall = "positive"
        elif sentiment_score < -0.2:
            overall = "negative"
        else:
            overall = "neutral"

        # 方面级预测：基于方面词附近局部情感词判断
        aspect_preds = []
        tokens = text.lower().split()
        pos_words = self.extractor.POSITIVE_WORDS.get(language, self.extractor.POSITIVE_WORDS["en"])
        neg_words = self.extractor.NEGATIVE_WORDS.get(language, self.extractor.NEGATIVE_WORDS["en"])

        for i, (concept, translations) in enumerate(self.extractor.ASPECT_VOCAB.items()):
            if features[i] <= 0.5:
                continue

            matched_term = next(
                (t for t in translations if t.lower() in text.lower()),
                concept,
            )

            # 找到方面词在句子中的位置
            term_pos = -1
            for idx, tok in enumerate(tokens):
                if matched_term.lower() in tok:
                    term_pos = idx
                    break

            # 检查方面词前后3个词内的情感词
            window = tokens[max(0, term_pos - 3):min(len(tokens), term_pos + 4)]
            pos_in_window = sum(1 for w in pos_words if w in " ".join(window))
            neg_in_window = sum(1 for w in neg_words if w in " ".join(window))

            if pos_in_window > neg_in_window:
                polarity = "positive"
            elif neg_in_window > pos_in_window:
                polarity = "negative"
            else:
                # 退回到整体情感
                polarity = overall if overall != "neutral" else "positive"

            aspect_preds.append({
                "term": matched_term,
                "concept": concept,
                "sentiment": polarity,
                "score": round(abs(pos_in_window - neg_in_window) / max(pos_in_window + neg_in_window, 1), 3),
            })

        return {
            "text": text,
            "language": language,
            "overall_sentiment": overall,
            "overall_score": round(sentiment_score, 3),
            "aspects": aspect_preds,
        }


# ---------------------------------------------------------------------------
# 5. LACA 数据增强 (LLM-based Pseudo-Label Generation)
# ---------------------------------------------------------------------------

class LACADataAugmenter:
    """基于 LACA 框架的 LLM 数据增强器。"""

    def __init__(self) -> None:
        self.generated_corpus = GENERATED_CORPUS

    def generate_pseudo_labels(
        self,
        target_texts: list[str],
        language: str,
        model: CrossLingualABSAModel,
    ) -> list[PseudoLabel]:
        """
        对目标语言无标注数据生成伪标签。

        步骤：
        1. 用 ABSA 模型做预测得到噪声标签
        2. 用 LLM 根据噪声标签生成更自然的目标语言句子
        3. 过滤质量不合格的样本
        """
        pseudo_labels: list[PseudoLabel] = []

        for text in target_texts:
            prediction = model.predict(text, language)
            confidence = abs(prediction["overall_score"])

            # 模拟 LLM 根据预测标签生成句子
            generated = self._llm_generate(prediction, language)

            pseudo_labels.append(PseudoLabel(
                original_text=text,
                generated_text=generated["text"],
                language=language,
                aspects=generated["aspects"],
                confidence=round(confidence, 3),
            ))

        return pseudo_labels

    def _llm_generate(self, prediction: dict[str, Any], language: str) -> dict[str, Any]:
        """模拟 LLM 根据预测标签生成目标语言句子。"""
        corpus = self.generated_corpus.get(language, [])
        if not corpus:
            return {"text": prediction["text"], "aspects": prediction["aspects"]}

        # 根据整体情感和方面匹配选择最相似的生成样本
        best_match = random.choice(corpus)
        return {"text": best_match[0], "aspects": best_match[1]}

    def filter_pseudo_labels(
        self,
        pseudo_labels: list[PseudoLabel],
        confidence_threshold: float = 0.3,
    ) -> list[PseudoLabel]:
        """过滤低置信度伪标签。"""
        return [p for p in pseudo_labels if p.confidence >= confidence_threshold]


# ---------------------------------------------------------------------------
# 6. 跨语言训练器 (LACA 核心流程)
# ---------------------------------------------------------------------------

class CrossLingualTrainer:
    """执行 LACA 跨语言训练流程。"""

    def __init__(self) -> None:
        self.model = CrossLingualABSAModel()
        self.augmenter = LACADataAugmenter()

    def train(
        self,
        source_examples: list[SentimentExample],
        target_unlabeled: dict[str, list[str]],
        use_laca: bool = True,
    ) -> dict[str, Any]:
        """
        完整训练流程。

        1. 在源语言数据上训练基线模型
        2. 对目标语言数据做零样本预测
        3. [LACA] 生成伪标签数据并增强
        4. 混合训练得到最终模型
        """
        results = {}

        # Step 1: 源语言训练
        self.model.fit(source_examples)
        results["source_train_size"] = len(source_examples)

        # Step 2 & 3: 目标语言预测 + 数据增强
        all_augmented: list[SentimentExample] = []
        for lang, texts in target_unlabeled.items():
            pseudo_labels = self.augmenter.generate_pseudo_labels(texts, lang, self.model)
            filtered = self.augmenter.filter_pseudo_labels(pseudo_labels)

            for pl in filtered:
                all_augmented.append(SentimentExample(
                    text=pl.generated_text,
                    language=lang,
                    aspects=pl.aspects,
                ))

        results["augmented_size"] = len(all_augmented)

        # Step 4: 混合训练 (如果启用 LACA)
        if use_laca and all_augmented:
            combined = source_examples + all_augmented
            self.model.fit(combined)
            results["final_train_size"] = len(combined)
        else:
            results["final_train_size"] = len(source_examples)

        return results

    def evaluate_zero_shot(self, target_texts: dict[str, list[str]]) -> dict[str, Any]:
        """在目标语言上执行零样本预测并返回结果。"""
        predictions: dict[str, list[dict[str, Any]]] = {}
        for lang, texts in target_texts.items():
            predictions[lang] = [self.model.predict(t, lang) for t in texts]

        # 统计
        stats: dict[str, Any] = {}
        for lang, preds in predictions.items():
            pos = sum(1 for p in preds if p["overall_sentiment"] == "positive")
            neg = sum(1 for p in preds if p["overall_sentiment"] == "negative")
            neu = sum(1 for p in preds if p["overall_sentiment"] == "neutral")
            stats[lang] = {
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "avg_confidence": round(
                    np.mean([abs(p["overall_score"]) for p in preds]), 3
                ),
            }

        return {"predictions": predictions, "statistics": stats}


# ---------------------------------------------------------------------------
# 7. 核心入口
# ---------------------------------------------------------------------------

def run_crosslingual_sentiment_analysis(
    use_laca: bool = True,
) -> dict[str, Any]:
    """
    主入口：运行跨语言情感迁移分析。
    """
    trainer = CrossLingualTrainer()

    train_info = trainer.train(
        source_examples=ENGLISH_TRAIN,
        target_unlabeled=TARGET_UNLABELED,
        use_laca=use_laca,
    )

    eval_results = trainer.evaluate_zero_shot(TARGET_UNLABELED)

    return {
        "training_info": train_info,
        "evaluation": eval_results,
        "method": "LACA (LLM Augmented)" if use_laca else "Zero-Shot XLM-R",
    }


# ---------------------------------------------------------------------------
# 8. 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("跨语言情感迁移 - 母婴出海多语言市场")
    print("=" * 70)

    # 基线：纯零样本
    print("\n📊 基线方法: Zero-Shot 跨语言迁移")
    baseline = run_crosslingual_sentiment_analysis(use_laca=False)
    print(f"   训练样本: {baseline['training_info']['source_train_size']}")
    for lang, stat in baseline["evaluation"]["statistics"].items():
        print(f"   {lang}: 正{stat['positive']} 负{stat['negative']} 中{stat['neutral']} "
              f"| 平均置信度={stat['avg_confidence']}")

    # LACA 增强
    print("\n🚀 LACA 增强后")
    laca = run_crosslingual_sentiment_analysis(use_laca=True)
    print(f"   源语言样本: {laca['training_info']['source_train_size']}")
    print(f"   生成伪标签: {laca['training_info']['augmented_size']}")
    print(f"   总训练样本: {laca['training_info']['final_train_size']}")
    for lang, stat in laca["evaluation"]["statistics"].items():
        print(f"   {lang}: 正{stat['positive']} 负{stat['negative']} 中{stat['neutral']} "
              f"| 平均置信度={stat['avg_confidence']}")

    # 示例预测
    print("\n💬 示例预测 (西班牙语)")
    example = laca["evaluation"]["predictions"]["es"][0]
    print(f"   文本: {example['text']}")
    print(f"   整体情感: {example['overall_sentiment']} ({example['overall_score']})")
    print(f"   方面:")
    for asp in example["aspects"]:
        print(f"     - {asp['term']} ({asp['concept']}): {asp['sentiment']}")

    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)

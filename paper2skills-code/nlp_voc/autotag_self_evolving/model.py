"""多任务层级分类器

实现 Topic(L1-L4) + Sentiment + Verbatim 联合预测。
使用简化版规则基线 + 可选 ML 扩展接口。

生产环境建议:
    - 使用 sentence-transformers 做 embedding
    - 使用 sklearn/xgboost 训练分类器
    - 使用 LLM API 做新标签命名
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from label_system import LabelNode, LabelSystem


@dataclass
class PredictionResult:
    """单条预测结果"""

    l1: Optional[str] = None
    l2: Optional[str] = None
    l3: Optional[str] = None
    l4: Optional[str] = None
    sentiment: int = 0          # -1=负面, 0=中性, +1=正面
    verbatim: str = ""          # 支撑标签的关键短语
    confidence: float = 0.0     # 整体置信度
    is_novel: bool = False      # 是否触发新标签发现

    def to_dict(self) -> dict:
        return {
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3,
            "l4": self.l4,
            "sentiment": self.sentiment,
            "verbatim": self.verbatim,
            "confidence": self.confidence,
            "is_novel": self.is_novel,
        }


class MultiTaskClassifier:
    """多任务层级分类器

    联合预测: Topic(L1-L4) + Sentiment + Verbatim

    当前实现为规则基线版，生产环境可替换为:
        - embedding + 层级分类器 (sklearn/xgboost)
        - 微调语言模型 (BERT/RoBERTa/DeBERTa)
        - LLM 零样本/少样本分类
    """

    # 情感词典（中文）
    SENTIMENT_POS = {
        "好", "不错", "满意", "喜欢", "推荐", "优质", "舒适", "柔软", "好用",
        "完美", "棒", "赞", "good", "great", "excellent", "love", "perfect",
        "comfortable", "soft", "nice",
    }
    SENTIMENT_NEG = {
        "差", "不好", "失望", "垃圾", "糟糕", "后悔", "差评", "退货", "漏",
        "硬", "粗糙", "臭", "异味", "坏", "破", "烂", "慢", "贵",
        "bad", "terrible", "worst", "hate", "disappointed", "poor", "rough",
        "hard", "smell", "broken", "leak", "expensive",
    }

    def __init__(self, label_system: LabelSystem, confidence_threshold: float = 0.6):
        self.label_system = label_system
        self.confidence_threshold = confidence_threshold

    def predict(self, text: str) -> PredictionResult:
        """对单条文本进行联合预测"""
        result = PredictionResult()

        # 1. 情感预测
        result.sentiment = self._predict_sentiment(text)

        # 2. Topic 层级预测 (L1→L2→L3→L4)
        l1_id, l1_conf = self._predict_level(text, level=1)
        if l1_id:
            result.l1 = self.label_system.get(l1_id).name

            l2_id, l2_conf = self._predict_level(text, level=2, parent_id=l1_id)
            if l2_id:
                result.l2 = self.label_system.get(l2_id).name

                l3_id, l3_conf = self._predict_level(text, level=3, parent_id=l2_id)
                if l3_id:
                    result.l3 = self.label_system.get(l3_id).name

                    l4_id, l4_conf = self._predict_level(text, level=4, parent_id=l3_id)
                    if l4_id:
                        result.l4 = self.label_system.get(l4_id).name
                        result.confidence = l4_conf
                    else:
                        # L4 未匹配 → 可能新标签
                        result.confidence = l3_conf
                        if l3_conf < self.confidence_threshold:
                            result.is_novel = True
                else:
                    result.confidence = l2_conf
                    if l2_conf < self.confidence_threshold:
                        result.is_novel = True
            else:
                result.confidence = l1_conf
        else:
            result.is_novel = True
            result.confidence = 0.0

        # 3. Verbatim 提取（简单关键词匹配）
        result.verbatim = self._extract_verbatim(text)

        return result

    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """批量预测"""
        return [self.predict(t) for t in texts]

    def _predict_level(
        self, text: str, level: int, parent_id: Optional[str] = None
    ) -> tuple[Optional[str], float]:
        """预测某一层的标签

        Returns:
            (标签ID, 置信度), 若无匹配返回 (None, 0.0)
        """
        text_lower = text.lower()

        # 获取候选标签
        candidates = [
            node for node in self.label_system.nodes.values()
            if node.level == level
            and node.status == "active"
            and (parent_id is None or node.parent_id == parent_id)
        ]

        best_id: Optional[str] = None
        best_score = 0.0

        for node in candidates:
            score = 0.0
            # 关键词匹配得分
            for kw in node.keywords:
                kw_lower = kw.lower()
                if kw_lower in text_lower:
                    # 精确匹配权重高，部分匹配权重低
                    score += 1.0 if f" {kw_lower} " in f" {text_lower} " else 0.5

            # 标签名称匹配
            if node.name.lower() in text_lower:
                score += 1.5

            if score > best_score:
                best_score = score
                best_id = node.id

        # 归一化为伪置信度
        if best_score > 0:
            confidence = min(best_score / 3.0, 1.0)  # 最高3分对应100%置信度
            return best_id, confidence
        return None, 0.0

    def _predict_sentiment(self, text: str) -> int:
        """预测情感极性"""
        text_lower = text.lower()
        pos_count = sum(1 for w in self.SENTIMENT_POS if w in text_lower)
        neg_count = sum(1 for w in self.SENTIMENT_NEG if w in text_lower)

        if neg_count > pos_count:
            return -1
        elif pos_count > neg_count:
            return 1
        return 0

    def _extract_verbatim(self, text: str) -> str:
        """提取支撑标签的关键短语（简化版）"""
        # 简单策略: 提取包含情感词和标签关键词的 10 字窗口
        text_lower = text.lower()
        best_phrase = ""
        best_score = 0

        for i in range(len(text)):
            window = text_lower[max(0, i-5):min(len(text), i+15)]
            score = 0
            for w in self.SENTIMENT_POS | self.SENTIMENT_NEG:
                if w in window:
                    score += 1
            if score > best_score:
                best_score = score
                best_phrase = text[max(0, i-5):min(len(text), i+15)]

        return best_phrase.strip() if best_phrase else text[:20]


# ── 新标签发现 ────────────────────────────────────────────────

class NovelLabelDetector:
    """新标签发现器

    当分类器置信度低于阈值时，触发新标签候选发现。
    当前实现为规则基线版，生产环境建议:
        - 使用 sentence-transformers 做 embedding
        - 使用 HDBSCAN / KMeans 聚类
        - 使用 LLM 为聚类生成标签名和描述
    """

    def __init__(self, min_cluster_size: int = 5, similarity_threshold: float = 0.85):
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.candidates: list[dict] = []  # 累积的候选标签

    def add_novel_text(self, text: str, predicted_path: Optional[dict] = None) -> None:
        """添加一条无法被现有标签覆盖的文本"""
        self.candidates.append({
            "text": text,
            "path": predicted_path or {},
            "embedding": None,  # 生产环境填充 embedding
        })

    def discover_candidates(self) -> list[dict]:
        """发现新标签候选

        当前为简化规则版:
        1. 按已预测的父标签分组
        2. 在组内做简单关键词共现聚类
        3. 返回聚类摘要作为候选标签

        Returns:
            候选标签列表, 每个包含: name, description, parent_id, sample_texts
        """
        from collections import defaultdict

        # 按父标签分组
        groups: dict[str, list[str]] = defaultdict(list)
        for c in self.candidates:
            # 使用已匹配到的最深层级作为父标签
            parent = c["path"].get("l3") or c["path"].get("l2") or c["path"].get("l1")
            if parent:
                groups[parent].append(c["text"])

        discovered = []
        for parent_name, texts in groups.items():
            if len(texts) < self.min_cluster_size:
                continue

            # 简单关键词提取: 提取高频名词短语
            keywords = self._extract_keywords(texts)
            if keywords:
                discovered.append({
                    "name": f"新: {keywords[0]}",
                    "description": f"从 {len(texts)} 条反馈中自动发现，相关词: {', '.join(keywords[:3])}",
                    "parent_name": parent_name,
                    "sample_texts": texts[:5],
                    "count": len(texts),
                })

        return discovered

    def _extract_keywords(self, texts: list[str]) -> list[str]:
        """从文本列表中提取高频关键词（简化版）"""
        word_counts: dict[str, int] = {}
        stopwords = {"的", "了", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}

        for text in texts:
            # 简单分词: 2-4 字滑动窗口
            for length in range(2, 5):
                for i in range(len(text) - length + 1):
                    word = text[i:i+length]
                    if any(c in word for c in stopwords):
                        continue
                    if re.match(r"^[\u4e00-\u9fa5]+$", word):  # 纯中文
                        word_counts[word] = word_counts.get(word, 0) + 1

        # 返回高频词
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, c in sorted_words[:5] if c >= 2]

    def reset(self) -> None:
        """清空候选缓存"""
        self.candidates = []


# ── 测试 ──────────────────────────────────────────────────────

def test_classifier():
    """测试多任务分类器"""
    from label_system import LabelNode, _create_demo_system

    print("=" * 60)
    print("测试: MultiTaskClassifier")
    print("=" * 60)

    system = _create_demo_system()
    classifier = MultiTaskClassifier(system, confidence_threshold=0.5)

    test_cases = [
        "纸尿裤晚上总是侧漏，宝宝睡不好，很失望",
        "奶粉溶解性不错，宝宝喝了不拉肚子，推荐",
        "物流太慢了，清关等了两周",
        "腰贴太硬，每次换尿布宝宝都哭",
        "这个产品还行吧，没什么特别的",
        "这个纸尿裤有个奇怪的问题，每次用完宝宝大腿内侧会发红，是不是过敏啊",
    ]

    print("\n--- 单条预测 ---")
    for text in test_cases:
        result = classifier.predict(text)
        print(f"\n文本: {text}")
        print(f"  路径: L1={result.l1}, L2={result.l2}, L3={result.l3}, L4={result.l4}")
        print(f"  情感: {'负面' if result.sentiment == -1 else '正面' if result.sentiment == 1 else '中性'}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  新标签候选: {'是' if result.is_novel else '否'}")
        print(f"  关键短语: {result.verbatim}")

    print("\n--- 批量预测 ---")
    results = classifier.predict_batch(test_cases)
    novel_count = sum(1 for r in results if r.is_novel)
    print(f"6 条文本中，{novel_count} 条触发新标签发现")

    print("\n" + "=" * 60)
    print("分类器测试完成 ✓")
    print("=" * 60)


def test_novel_detector():
    """测试新标签发现器"""
    print("\n" + "=" * 60)
    print("测试: NovelLabelDetector")
    print("=" * 60)

    detector = NovelLabelDetector(min_cluster_size=2)

    # 模拟低置信度文本（模拟 "过敏" 这一新痛点）
    novel_texts = [
        "用了这个纸尿裤宝宝大腿发红，是不是过敏",
        "我家宝宝皮肤敏感，用了会起红疹",
        "腰部一圈红红的，像是过敏了",
        "材质可能不适合敏感肌，宝宝起了疹子",
        "之前用别的牌子没事，这个一用就红",
        "红屁屁严重，怀疑是过敏反应",
    ]

    for text in novel_texts:
        detector.add_novel_text(text, predicted_path={"l1": "纸尿裤", "l2": "质量"})

    candidates = detector.discover_candidates()
    print(f"\n发现 {len(candidates)} 个新标签候选:")
    for c in candidates:
        print(f"\n  候选: {c['name']}")
        print(f"  描述: {c['description']}")
        print(f"  父标签: {c['parent_name']}")
        print(f"  样本数: {c['count']}")
        print(f"  示例: {c['sample_texts'][0]}")

    print("\n" + "=" * 60)
    print("新标签发现测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_classifier()
    test_novel_detector()

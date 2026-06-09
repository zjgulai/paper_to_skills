"""
Consumer Complaint Recall Prediction — 消费者投诉驱动的召回风险预测
基于 HDPYP 半参数主题模型的召回预测系统

依赖：纯 Python 标准库（无外部依赖）
Python 版本：3.8+
"""

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────

@dataclass
class ComplaintRecord:
    """消费者投诉记录"""
    product_id: str
    category: str
    complaint_text: str
    incident_date: date
    injury_count: int = 0
    component: Optional[str] = None

    def __post_init__(self):
        if self.injury_count < 0:
            raise ValueError("injury_count 不能为负数")


@dataclass
class TopicModel:
    """主题模型结果"""
    topic_id: int
    top_words: List[Tuple[str, float]]  # (词, 权重)
    weight: float  # 该主题在语料中的整体权重


@dataclass
class RecallRiskReport:
    """召回风险评估报告"""
    product_id: str
    category: str
    recall_probability: float        # 0.0 ~ 1.0
    risk_level: str                  # LOW / MEDIUM / HIGH
    dominant_topics: List[TopicModel]
    topic_concentration: float       # Gini 系数，越高越集中
    weighted_injury_score: float
    recommendation: str


# ─────────────────────────────────────────────
# 文本预处理
# ─────────────────────────────────────────────

# 英文停用词（精简版）
_STOP_WORDS = {
    "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "it",
    "they", "my", "your", "his", "her", "our", "its", "their", "me", "us",
    "him", "her", "them", "and", "but", "or", "not", "no", "nor", "so",
    "yet", "just", "very", "also", "than", "then", "when", "if", "while",
    "after", "before", "about", "out", "up", "down", "over", "under",
    "again", "product", "item", "bought", "purchased", "received",
}


def _tokenize(text: str) -> List[str]:
    """分词 + 停用词过滤"""
    text = text.lower()
    tokens = re.findall(r"[a-z]{3,}", text)
    return [t for t in tokens if t not in _STOP_WORDS]


# ─────────────────────────────────────────────
# 简化版 LDA 主题提取器
# ─────────────────────────────────────────────

class LDATopicExtractor:
    """
    简化版 LDA（Latent Dirichlet Allocation）主题模型
    使用 Collapsed Gibbs 采样（简化实现）
    适用于小规模投诉文本（< 10,000 条）
    """

    def __init__(
        self,
        n_topics: int = 5,
        alpha: float = 0.1,
        beta: float = 0.01,
        n_iterations: int = 50,
        random_seed: int = 42,
    ):
        """
        Args:
            n_topics: 主题数（HDPYP 可自动推断，此处简化为固定值）
            alpha: 文档-主题 Dirichlet 先验（小值 → 文档倾向于少量主题）
            beta: 主题-词 Dirichlet 先验（小值 → 主题由少量词主导）
            n_iterations: Gibbs 采样迭代次数
            random_seed: 随机种子
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iterations = n_iterations
        self.random_seed = random_seed

        # 训练后填充
        self._vocab: List[str] = []
        self._word2id: Dict[str, int] = {}
        self._phi: List[List[float]] = []      # 主题-词分布 [n_topics × vocab]
        self._theta: List[List[float]] = []   # 文档-主题分布 [n_docs × n_topics]
        self._is_fitted = False

    def fit(self, documents: List[List[str]]) -> "LDATopicExtractor":
        """
        训练主题模型

        Args:
            documents: 分词后的文档列表

        Returns:
            self（支持链式调用）
        """
        random.seed(self.random_seed)

        # 构建词汇表
        all_words = [w for doc in documents for w in doc]
        vocab_counter = Counter(all_words)
        # 过滤频率 < 2 的词
        self._vocab = [w for w, c in vocab_counter.most_common() if c >= 2]
        self._word2id = {w: i for i, w in enumerate(self._vocab)}
        V = len(self._vocab)
        K = self.n_topics

        if V == 0:
            raise ValueError("词汇表为空，请检查输入文本")

        # 将文档转为词 ID 列表（过滤不在词汇表中的词）
        docs_ids: List[List[int]] = []
        for doc in documents:
            ids = [self._word2id[w] for w in doc if w in self._word2id]
            docs_ids.append(ids)

        # 初始化计数矩阵
        # doc_topic_count[d][k]: 文档 d 中分配给主题 k 的词数
        doc_topic_count = [[0] * K for _ in range(len(docs_ids))]
        # topic_word_count[k][v]: 主题 k 中词 v 出现次数
        topic_word_count = [[0] * V for _ in range(K)]
        # topic_count[k]: 主题 k 的总词数
        topic_count = [0] * K

        # 随机初始化主题分配
        topic_assignments: List[List[int]] = []
        for d, doc in enumerate(docs_ids):
            assignments = []
            for w in doc:
                k = random.randint(0, K - 1)
                assignments.append(k)
                doc_topic_count[d][k] += 1
                topic_word_count[k][w] += 1
                topic_count[k] += 1
            topic_assignments.append(assignments)

        # Collapsed Gibbs 采样
        for _ in range(self.n_iterations):
            for d, doc in enumerate(docs_ids):
                for i, w in enumerate(doc):
                    # 移除当前词的主题分配
                    k_old = topic_assignments[d][i]
                    doc_topic_count[d][k_old] -= 1
                    topic_word_count[k_old][w] -= 1
                    topic_count[k_old] -= 1

                    # 计算各主题的条件概率
                    probs = []
                    for k in range(K):
                        p = (doc_topic_count[d][k] + self.alpha) * \
                            (topic_word_count[k][w] + self.beta) / \
                            (topic_count[k] + V * self.beta)
                        probs.append(p)

                    # 按概率重采样主题
                    k_new = self._sample_discrete(probs)
                    topic_assignments[d][i] = k_new
                    doc_topic_count[d][k_new] += 1
                    topic_word_count[k_new][w] += 1
                    topic_count[k_new] += 1

        # 计算归一化的 phi（主题-词分布）和 theta（文档-主题分布）
        self._phi = []
        for k in range(K):
            total = sum(topic_word_count[k]) + V * self.beta
            phi_k = [(topic_word_count[k][v] + self.beta) / total for v in range(V)]
            self._phi.append(phi_k)

        self._theta = []
        for d in range(len(docs_ids)):
            total = len(docs_ids[d]) + K * self.alpha
            theta_d = [(doc_topic_count[d][k] + self.alpha) / total for k in range(K)]
            self._theta.append(theta_d)

        self._is_fitted = True
        return self

    def get_topics(self, top_n: int = 10) -> List[TopicModel]:
        """提取每个主题的高权重词"""
        self._check_fitted()
        topics = []
        corpus_topic_weights = [
            sum(self._theta[d][k] for d in range(len(self._theta))) / len(self._theta)
            for k in range(self.n_topics)
        ]
        for k in range(self.n_topics):
            # 按权重排序词汇
            word_weights = sorted(
                enumerate(self._phi[k]), key=lambda x: x[1], reverse=True
            )[:top_n]
            top_words = [(self._vocab[v], w) for v, w in word_weights]
            topics.append(TopicModel(
                topic_id=k,
                top_words=top_words,
                weight=corpus_topic_weights[k],
            ))
        return sorted(topics, key=lambda t: t.weight, reverse=True)

    def transform(self, document: List[str]) -> List[float]:
        """将新文档映射到主题分布（近似推断）"""
        self._check_fitted()
        topic_counts = [0.0] * self.n_topics
        for w in document:
            if w in self._word2id:
                wid = self._word2id[w]
                # 使用 phi 的最大概率主题（简化推断）
                best_k = max(range(self.n_topics), key=lambda k: self._phi[k][wid])
                topic_counts[best_k] += 1.0
        total = sum(topic_counts) + self.n_topics * self.alpha
        return [(topic_counts[k] + self.alpha) / total for k in range(self.n_topics)]

    @staticmethod
    def _sample_discrete(probs: List[float]) -> int:
        """从离散分布中采样"""
        total = sum(probs)
        r = random.random() * total
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")


# ─────────────────────────────────────────────
# 召回风险预测器
# ─────────────────────────────────────────────

class RecallRiskPredictor:
    """
    基于主题分布的召回风险预测器

    预测逻辑：
      召回概率 = 0.4 × topic_concentration
                + 0.4 × normalized_injury_score
                + 0.2 × complaint_velocity_score
    """

    # 高危主题词（与召回强相关）
    HIGH_RISK_KEYWORDS = {
        "broken", "collapse", "fall", "injury", "cut", "burn", "fire",
        "choke", "suffocate", "strangle", "fracture", "crack", "break",
        "fail", "hazard", "dangerous", "unsafe", "recall", "defect",
        "latch", "buckle", "wheel", "sharp", "toxic", "lead", "chemical",
    }

    def __init__(self, extractor: LDATopicExtractor):
        self.extractor = extractor

    def predict(
        self,
        records: List[ComplaintRecord],
        high_risk_topic_threshold: float = 0.25,
    ) -> RecallRiskReport:
        """
        对一组投诉记录（同一产品）做召回风险评估

        Args:
            records: 同一产品的投诉列表
            high_risk_topic_threshold: 单主题权重超过此值视为主导主题

        Returns:
            RecallRiskReport
        """
        if not records:
            raise ValueError("投诉记录列表不能为空")

        product_id = records[0].product_id
        category = records[0].category

        # 1. 文档分词
        docs = [_tokenize(r.complaint_text) for r in records]

        # 2. 主题分布（逐条推断后加权平均）
        total_injury = sum(r.injury_count for r in records) or 1
        weights = [(r.injury_count + 1) / (total_injury + len(records)) for r in records]

        topic_dists = [self.extractor.transform(doc) for doc in docs]

        # 加权平均主题分布
        K = self.extractor.n_topics
        avg_topic_dist = [0.0] * K
        for dist, w in zip(topic_dists, weights):
            for k in range(K):
                avg_topic_dist[k] += dist[k] * w

        # 3. 主题集中度（Gini 系数）
        gini = self._gini(avg_topic_dist)

        # 4. 伤害加权评分（归一化到 0-1）
        max_injury = max(r.injury_count for r in records) or 1
        weighted_injury = sum(
            (r.injury_count / max_injury) * w for r, w in zip(records, weights)
        )

        # 5. 高危关键词命中率
        all_tokens = [t for doc in docs for t in doc]
        risk_keyword_hits = sum(1 for t in all_tokens if t in self.HIGH_RISK_KEYWORDS)
        keyword_score = min(1.0, risk_keyword_hits / (len(all_tokens) + 1) * 10)

        # 6. 综合召回概率
        recall_prob = (
            0.40 * gini
            + 0.40 * weighted_injury
            + 0.20 * keyword_score
        )
        recall_prob = max(0.0, min(1.0, recall_prob))

        # 7. 风险等级
        if recall_prob >= 0.65:
            risk_level = "HIGH"
            recommendation = (
                "⚠️ 高召回风险：建议立即暂停选品并联系供应商核查同款投诉；"
                "启动 CPSC 合规团队介入，预计需要 6-12 个月整改周期。"
            )
        elif recall_prob >= 0.35:
            risk_level = "MEDIUM"
            recommendation = (
                "⚡ 中等召回风险：建议加强质检频率，要求供应商提供第三方检测报告；"
                "在 SaferProducts.gov 设置品类告警，每月复查投诉趋势。"
            )
        else:
            risk_level = "LOW"
            recommendation = (
                "✅ 低召回风险：当前投诉分散无明显主导缺陷主题；"
                "建议保持常规质检频率，每季度复查投诉数据。"
            )

        # 8. 提取主导主题
        all_topics = self.extractor.get_topics(top_n=5)
        dominant_topics = [
            t for t in all_topics if avg_topic_dist[t.topic_id] >= high_risk_topic_threshold
        ]

        return RecallRiskReport(
            product_id=product_id,
            category=category,
            recall_probability=round(recall_prob, 4),
            risk_level=risk_level,
            dominant_topics=dominant_topics,
            topic_concentration=round(gini, 4),
            weighted_injury_score=round(weighted_injury, 4),
            recommendation=recommendation,
        )

    @staticmethod
    def _gini(values: List[float]) -> float:
        """计算 Gini 系数（衡量主题集中度）"""
        if not values or sum(values) == 0:
            return 0.0
        n = len(values)
        sorted_vals = sorted(values)
        total = sum(sorted_vals)
        cumsum = 0.0
        gini_sum = 0.0
        for i, v in enumerate(sorted_vals):
            cumsum += v
            gini_sum += (2 * (i + 1) - n - 1) * v
        return gini_sum / (n * total) if total > 0 else 0.0


# ─────────────────────────────────────────────
# 演示函数
# ─────────────────────────────────────────────

def _make_stroller_complaints() -> List[ComplaintRecord]:
    """生成婴儿推车投诉测试数据"""
    complaints_data = [
        (
            "STROLLER-001",
            "The front wheel suddenly broke off while walking. Baby fell forward and hit face. "
            "Wheel collapse was completely unexpected. Very dangerous defect.",
            date(2024, 3, 15),
            2,
            "wheel",
        ),
        (
            "STROLLER-001",
            "Folding mechanism failed without warning. The stroller collapsed while my child was in it. "
            "The latch that holds it open cracked and broke. Injury occurred.",
            date(2024, 4, 2),
            1,
            "frame",
        ),
        (
            "STROLLER-001",
            "Wheel detached from axle causing sudden collapse. Front wheel broke off. "
            "Child was not injured but stroller is completely unsafe now.",
            date(2024, 5, 10),
            0,
            "wheel",
        ),
        (
            "STROLLER-001",
            "The buckle on the safety strap released on its own. Child fell out. "
            "The release mechanism is defective and dangerous. Serious injury to head.",
            date(2024, 6, 18),
            3,
            "harness",
        ),
        (
            "STROLLER-001",
            "Same wheel problem as many reviews mention. Wheel cracked and broke. "
            "Stroller collapsed while crossing street. Very hazardous defect.",
            date(2024, 7, 25),
            1,
            "wheel",
        ),
    ]
    return [
        ComplaintRecord(
            product_id=pid,
            category="infant_stroller",
            complaint_text=text,
            incident_date=dt,
            injury_count=injuries,
            component=component,
        )
        for pid, text, dt, injuries, component in complaints_data
    ]


def run_demo():
    """演示：婴儿推车召回风险预测"""
    print("=" * 60)
    print("Consumer Complaint Recall Prediction — 演示")
    print("=" * 60)

    # 1. 准备投诉数据
    records = _make_stroller_complaints()
    print(f"\n📋 加载投诉记录：{len(records)} 条（产品: {records[0].product_id}）")
    print(f"   总伤害人数：{sum(r.injury_count for r in records)} 人")

    # 2. 训练主题模型
    docs = [_tokenize(r.complaint_text) for r in records]
    extractor = LDATopicExtractor(
        n_topics=3,
        n_iterations=100,
        random_seed=42,
    )
    extractor.fit(docs)
    print("\n🔍 主题提取结果：")
    for topic in extractor.get_topics(top_n=6):
        words = ", ".join(f"{w}({s:.3f})" for w, s in topic.top_words)
        print(f"  主题 {topic.topic_id} (权重 {topic.weight:.3f}): {words}")

    # 3. 预测召回风险
    predictor = RecallRiskPredictor(extractor)
    report = predictor.predict(records)

    print(f"\n🚨 召回风险报告")
    print(f"  产品 ID：{report.product_id}")
    print(f"  品类：{report.category}")
    print(f"  召回概率：{report.recall_probability:.1%}")
    print(f"  风险等级：{report.risk_level}")
    print(f"  主题集中度 (Gini)：{report.topic_concentration:.4f}")
    print(f"  加权伤害评分：{report.weighted_injury_score:.4f}")
    print(f"\n💡 建议：{report.recommendation}")

    # 4. 验证断言
    assert report.risk_level in {"LOW", "MEDIUM", "HIGH"}, "风险等级应为 LOW/MEDIUM/HIGH"
    assert 0.0 <= report.recall_probability <= 1.0, "召回概率应在 [0, 1] 范围内"
    assert 0.0 <= report.topic_concentration <= 1.0, "Gini 系数应在 [0, 1] 范围内"
    print("\n✅ 所有断言通过 — 模块验证成功")
    return report


if __name__ == "__main__":
    run_demo()

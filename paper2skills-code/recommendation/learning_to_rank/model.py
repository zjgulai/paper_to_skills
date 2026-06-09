"""
NeuralNDCG Learning-to-Rank — 神经排序学习
paper2skills-code: 05-推荐系统 | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass


@dataclass
class SearchResult:
    doc_id: str
    relevance_label: int   # 0=无关 1=相关 2=高度相关
    features: list[float]
    initial_score: float = 0.0
    final_score: float = 0.0


def dcg(relevances: list[int], k: int = 10) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg(predicted_order: list[SearchResult], k: int = 10) -> float:
    ideal = sorted(predicted_order, key=lambda x: -x.relevance_label)
    ideal_dcg = dcg([d.relevance_label for d in ideal], k)
    pred_dcg = dcg([d.relevance_label for d in predicted_order], k)
    return pred_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


class LinearRankingModel:
    """线性 LTR 模型（生产替换为 LambdaMART/NeuralNDCG）"""
    def __init__(self, n_features: int, seed: int = 42):
        random.seed(seed)
        self.weights = [random.gauss(0, 0.1) for _ in range(n_features)]

    def score(self, features: list[float]) -> float:
        return sum(w * f for w, f in zip(self.weights, features[:len(self.weights)]))

    def train(self, results_list: list[list[SearchResult]], lr: float = 0.01, epochs: int = 5):
        for _ in range(epochs):
            for results in results_list:
                for doc in results:
                    pred = self.score(doc.features)
                    grad = (doc.relevance_label / 2.0 - pred) * lr
                    for i in range(len(self.weights)):
                        if i < len(doc.features):
                            self.weights[i] += grad * doc.features[i]


def run_ltr_demo():
    random.seed(42)
    query_results = [
        SearchResult(f"D{i}", random.randint(0, 2), [random.random() for _ in range(10)])
        for i in range(20)
    ]
    model = LinearRankingModel(n_features=10)
    model.train([query_results], epochs=20)

    for doc in query_results:
        doc.final_score = model.score(doc.features)

    baseline = sorted(query_results, key=lambda x: -x.initial_score)
    optimized = sorted(query_results, key=lambda x: -x.final_score)

    ndcg_base = ndcg(baseline, k=10)
    ndcg_opt = ndcg(optimized, k=10)

    print("=== Learning-to-Rank（母婴搜索排序）===")
    print(f"基线 NDCG@10:   {ndcg_base:.4f}")
    print(f"LTR 模型 NDCG@10: {ndcg_opt:.4f}")
    print(f"提升: {(ndcg_opt - ndcg_base) * 100:.1f}%")
    print("Top-5 排序结果:")
    for r in optimized[:5]:
        print(f"  {r.doc_id}: 相关度={r.relevance_label} 分数={r.final_score:.4f}")
    print("✅ 排序学习演示完成")
if __name__ == "__main__":
    run_ltr_demo()

"""
Diversity-Aware Reranking (SMMR)
Skill: Skill-Diversity-Reranking-SMMR.md
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SMMR:
    """Sampling-Based MMR Reranker"""

    def __init__(self, lambda_param=0.7, temperature=1.0):
        self.lambda_param = lambda_param
        self.temperature = temperature

    def rerank(self, item_ids, relevance_scores, item_features, k=10):
        """SMMR 重排序"""
        n = len(item_ids)
        selected, selected_features, remaining = [], [], set(range(n))
        rel_norm = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)

        while len(selected) < k and remaining:
            mmr_scores = []
            for i in remaining:
                rel_term = self.lambda_param * rel_norm[i]
                max_sim = cosine_similarity(item_features[i:i+1], np.array(selected_features))[0].max() if selected_features else 0
                mmr_scores.append((i, rel_term - (1 - self.lambda_param) * max_sim))

            scores = np.array([s for _, s in mmr_scores]) / self.temperature
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()
            idx = np.random.choice(len(mmr_scores), p=probs)
            si, _ = mmr_scores[idx]

            selected.append(si)
            selected_features.append(item_features[si])
            remaining.remove(si)

        return [item_ids[i] for i in selected]


def category_diversity_rerank(item_ids, relevance_scores, categories, max_per_category=2, k=10):
    """品类维度多样性约束"""
    selected, counts = [], {}
    for idx in np.argsort(relevance_scores)[::-1]:
        cat = categories[idx]
        if counts.get(cat, 0) < max_per_category:
            selected.append(item_ids[idx])
            counts[cat] = counts.get(cat, 0) + 1
        if len(selected) >= k:
            break
    return selected


if __name__ == '__main__':
    item_ids = list(range(10))
    relevance = np.array([0.95, 0.93, 0.91, 0.88, 0.85, 0.82, 0.80, 0.78, 0.75, 0.72])
    features = np.random.rand(10, 64)
    features[:3] += 0.5

    smmr = SMMR(lambda_param=0.6, temperature=1.2)
    print("SMMR result:", smmr.rerank(item_ids, relevance, features, k=5))

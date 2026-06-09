"""
Deep Learning Recommendation HI — 异构信息推荐
paper2skills-code: 05-推荐系统 | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass


@dataclass
class HeteroItem:
    item_id: str
    item_type: str  # product / content / bundle
    features: list[float]
    meta: dict


@dataclass
class UserProfile:
    user_id: str
    purchase_history: list[str]
    browse_history: list[str]
    preference_vector: list[float]


@dataclass
class RecResult:
    item_id: str
    score: float
    item_type: str
    reason: str


def dot_product(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def cosine_sim(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot_product(a, b) / (na * nb + 1e-9)


class HeterogeneousRecommender:
    """异构信息融合推荐（用户向量 + 物品向量 + 行为图）"""
    def __init__(self, n_factors: int = 16, seed: int = 42):
        self.n_factors = n_factors
        random.seed(seed)

    def _item_score(self, user: UserProfile, item: HeteroItem) -> float:
        base_score = cosine_sim(user.preference_vector[:self.n_factors],
                                item.features[:self.n_factors])
        purchase_bonus = 0.2 if item.item_id in user.purchase_history else 0.0
        browse_bonus = 0.1 if item.item_id in user.browse_history else 0.0
        type_weight = {"product": 1.0, "bundle": 1.1, "content": 0.8}.get(item.item_type, 1.0)
        return (base_score + purchase_bonus + browse_bonus) * type_weight

    def recommend(self, user: UserProfile, catalog: list[HeteroItem],
                  top_k: int = 5, exclude_purchased: bool = True) -> list[RecResult]:
        candidates = [item for item in catalog
                      if not (exclude_purchased and item.item_id in user.purchase_history)]
        scored = [(item, self._item_score(user, item)) for item in candidates]
        scored.sort(key=lambda x: -x[1])
        results = []
        for item, score in scored[:top_k]:
            reason = (f"基于您的浏览记录推荐" if item.item_id in user.browse_history
                      else f"与您偏好高度匹配（相似度 {score:.2f}）")
            results.append(RecResult(item.item_id, round(score, 4), item.item_type, reason))
        return results


def run_hi_demo():
    random.seed(42)
    n = 16
    user = UserProfile(
        user_id="U001",
        purchase_history=["P001", "P003"],
        browse_history=["P002", "P005"],
        preference_vector=[random.random() for _ in range(n)],
    )
    catalog = [HeteroItem(f"P{i:03d}", "product" if i % 3 != 0 else "bundle",
                          [random.random() for _ in range(n)], {"category": "baby"})
               for i in range(1, 21)]

    rec = HeterogeneousRecommender()
    results = rec.recommend(user, catalog, top_k=5)

    print("=== 异构推荐系统（母婴用户推荐）===")
    for r in results:
        print(f"  [{r.item_type}] {r.item_id}: {r.score:.4f} — {r.reason}")
    print("✅ 异构推荐演示完成")
if __name__ == "__main__":
    run_hi_demo()

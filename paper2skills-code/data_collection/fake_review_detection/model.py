"""
Fake Review Detection
整合 JARVIS + DS-DGA-GCN + CAMERA

论文来源:
  JARVIS:    arXiv:2602.12941
  DS-DGA-GCN: arXiv:2603.08332
  CAMERA:    arXiv:2605.20032
"""

import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ReviewNode:
    review_id: str
    user_id: str
    text: str
    rating: float
    timestamp: float
    platform: str
    is_verified: bool = False
    neighbors: List[str] = field(default_factory=list)
    fraud_score: float = 0.0


class EvidenceGraph:
    """JARVIS 风格：多模态混合检索构建异构证据图"""

    def __init__(self):
        self.nodes: Dict[str, ReviewNode] = {}
        self.user_reviews: Dict[str, List[str]] = {}
        self.edges: List[Tuple[str, str, str]] = []

    def add_review(self, review: ReviewNode):
        self.nodes[review.review_id] = review
        self.user_reviews.setdefault(review.user_id, []).append(review.review_id)

    def build_edges(self):
        for user_id, review_ids in self.user_reviews.items():
            if len(review_ids) > 1:
                for i in range(len(review_ids)):
                    for j in range(i + 1, len(review_ids)):
                        self.edges.append((review_ids[i], review_ids[j], "same_user"))
        reviews_list = list(self.nodes.values())
        for i in range(len(reviews_list)):
            for j in range(i + 1, len(reviews_list)):
                ri, rj = reviews_list[i], reviews_list[j]
                if abs(ri.timestamp - rj.timestamp) < 3600 and ri.user_id != rj.user_id:
                    self.edges.append((ri.review_id, rj.review_id, "temporal_proximity"))

    def get_neighbors(self, review_id: str) -> List[str]:
        return [b for a, b, _ in self.edges if a == review_id] +                [a for a, b, _ in self.edges if b == review_id]


class GraphFraudDetector:
    """DS-DGA-GCN 风格：动态图注意力网络检测刷评群组"""

    BURST_WINDOW = 3600
    MIN_GROUP_SIZE = 3

    def __init__(self, graph: EvidenceGraph):
        self.graph = graph

    def compute_network_features(self, review_id: str) -> Dict[str, float]:
        neighbors = self.graph.get_neighbors(review_id)
        review = self.graph.nodes.get(review_id)
        if not review:
            return {}
        same_user_count = sum(1 for a, b, t in self.graph.edges
                              if t == "same_user" and review_id in (a, b))
        temporal_count = sum(1 for a, b, t in self.graph.edges
                             if t == "temporal_proximity" and review_id in (a, b))
        user_review_count = len(self.graph.user_reviews.get(review.user_id, []))
        extreme_rating = 1.0 if review.rating in (1.0, 5.0) else 0.0
        return {
            "neighbor_count": len(neighbors),
            "same_user_edges": same_user_count,
            "temporal_edges": temporal_count,
            "user_review_count": user_review_count,
            "extreme_rating": extreme_rating,
            "is_verified": float(review.is_verified),
        }

    def fraud_score(self, features: Dict[str, float]) -> float:
        score = 0.0
        if features.get("user_review_count", 0) > 10:
            score += 0.3
        if features.get("temporal_edges", 0) > 2:
            score += 0.25
        if features.get("extreme_rating", 0) > 0 and not features.get("is_verified"):
            score += 0.2
        if features.get("same_user_edges", 0) > 3:
            score += 0.25
        return min(1.0, score)

    def detect_groups(self) -> List[List[str]]:
        visited: Set[str] = set()
        groups = []
        for review_id in self.graph.nodes:
            if review_id in visited:
                continue
            cluster = self._bfs(review_id, visited)
            if len(cluster) >= self.MIN_GROUP_SIZE:
                groups.append(cluster)
        return groups

    def _bfs(self, start: str, visited: Set[str]) -> List[str]:
        queue, cluster = [start], [start]
        visited.add(start)
        while queue:
            node = queue.pop(0)
            for nb in self.graph.get_neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    cluster.append(nb)
        return cluster


class SemanticCamouflageDetector:
    """CAMERA 风格：无监督检测语义伪装（欺诈者刻意模仿正常用户文本）"""

    GENUINE_SIGNALS = ["because", "although", "however", "specifically", "compared",
                       "因为", "虽然", "具体", "相比", "但是"]
    FAKE_SIGNALS = ["amazing", "perfect", "love", "must buy", "highly recommend",
                    "完美", "必买", "强推", "超级好", "太棒了"]

    def camouflage_score(self, text: str) -> float:
        tl = text.lower()
        genuine = sum(1 for s in self.GENUINE_SIGNALS if s in tl)
        fake = sum(1 for s in self.FAKE_SIGNALS if s in tl)
        length_penalty = 0.3 if len(text.split()) < 10 else 0.0
        fake_ratio = fake / max(genuine + fake, 1)
        return min(1.0, fake_ratio * 0.5 + length_penalty)

    def is_camouflaged(self, text: str, threshold: float = 0.6) -> bool:
        return self.camouflage_score(text) >= threshold


class FraudDetectionPipeline:
    def __init__(self):
        self.graph = EvidenceGraph()
        self.semantic_detector = SemanticCamouflageDetector()

    def add_reviews(self, reviews: List[ReviewNode]):
        for r in reviews:
            self.graph.add_review(r)
        self.graph.build_edges()

    def score_reviews(self) -> List[Tuple[str, float, str]]:
        detector = GraphFraudDetector(self.graph)
        results = []
        for review_id, review in self.graph.nodes.items():
            features = detector.compute_network_features(review_id)
            graph_score = detector.fraud_score(features)
            semantic_score = self.semantic_detector.camouflage_score(review.text)
            combined = 0.6 * graph_score + 0.4 * semantic_score
            label = "fraud" if combined > 0.5 else "genuine"
            results.append((review_id, combined, label))
        return results

    def get_fraud_groups(self) -> List[List[str]]:
        return GraphFraudDetector(self.graph).detect_groups()


def test_graph_fraud_detection():
    pipeline = FraudDetectionPipeline()
    reviews = [
        ReviewNode("r1", "bot_u1", "Amazing product! 5 stars!", 5.0, 1000.0, "amazon"),
        ReviewNode("r2", "bot_u2", "Perfect! Must buy!", 5.0, 1050.0, "amazon"),
        ReviewNode("r3", "bot_u3", "Love it! Highly recommend!", 5.0, 1100.0, "amazon"),
        ReviewNode("r4", "real_u1", "Good suction but slightly noisy. Battery lasts about 3 hours, which is acceptable for daily use.", 4.0, 2000.0, "amazon", is_verified=True),
    ]
    pipeline.add_reviews(reviews)
    scores = pipeline.score_reviews()
    bot_scores = [s for rid, s, _ in scores if rid in ("r1", "r2", "r3")]
    real_scores = [s for rid, s, _ in scores if rid == "r4"]
    assert all(b > r for b in bot_scores for r in real_scores), f"Bot scores {bot_scores} should exceed real {real_scores}"
    print(f"[PASS] graph_fraud: bot_avg={sum(bot_scores)/len(bot_scores):.3f}, real={real_scores[0]:.3f}")


def test_semantic_camouflage():
    detector = SemanticCamouflageDetector()
    genuine = "Although the suction is strong, I found the assembly complicated because the parts don\'t align well."
    fake = "Amazing perfect love must buy highly recommend! 5 stars!"
    assert detector.camouflage_score(fake) > detector.camouflage_score(genuine)
    assert detector.is_camouflaged(fake)
    assert not detector.is_camouflaged(genuine)
    print(f"[PASS] semantic: fake={detector.camouflage_score(fake):.3f}, genuine={detector.camouflage_score(genuine):.3f}")


def test_fraud_group_detection():
    pipeline = FraudDetectionPipeline()
    reviews = [ReviewNode(f"r{i}", f"bot_u{i}", "Amazing! Perfect!", 5.0, 1000.0 + i * 100, "amazon")
               for i in range(5)]
    reviews.append(ReviewNode("r_real", "real_user", "Decent pump, lasts 3 hours.", 4.0, 50000.0, "amazon", is_verified=True))
    pipeline.add_reviews(reviews)
    groups = pipeline.get_fraud_groups()
    assert len(groups) >= 1
    assert any(len(g) >= 3 for g in groups)
    print(f"[PASS] fraud_groups: {len(groups)} groups detected, largest={max(len(g) for g in groups)}")


if __name__ == "__main__":
    test_graph_fraud_detection()
    test_semantic_camouflage()
    test_fraud_group_detection()
    print("\n✅ All tests passed")

"""
StaR: Statement-level Ranking for Explainable Recommendation
基于论文: Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation
从评论中提取解释性、原子性、唯一的statements，并进行排序。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple


@dataclass
class Review:
    text: str
    item_id: str = ""
    user_id: str = ""
    rating: int = 5


@dataclass
class Statement:
    text: str
    item_id: str = ""
    sentiment: str = "positive"  # positive / negative / neutral


class StatementExtractor:
    """
    Two-stage extraction pipeline (simulated without external LLM):
    1) candidate extraction: rule-based atomic statement extraction
    2) verification: filter non-explanatory, non-atomic, redundant candidates
    """

    # Heuristic patterns for explanatory statements about product aspects
    POSITIVE_PATTERNS = [
        r"加热均匀", r"温控精准", r"操作简单", r"清洗方便", r"容量大",
        r"静音", r"续航久", r"便携", r"吸力适中", r"佩戴舒适",
    ]
    NEGATIVE_PATTERNS = [
        r"加热不均", r"温控不准", r"操作复杂", r"清洗麻烦", r"容量小",
        r"噪音大", r"续航短", r"笨重", r"吸力过大", r"佩戴不适",
    ]

    def extract_candidates(self, reviews: List[Review]) -> List[Statement]:
        candidates = []
        for review in reviews:
            sentences = re.split(r"[。！；]", review.text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 5:
                    continue
                # Check for aspect-related explanatory content
                matched = False
                sentiment = "neutral"
                if any(p in sent for p in self.POSITIVE_PATTERNS):
                    matched = True
                    sentiment = "positive"
                elif any(p in sent for p in self.NEGATIVE_PATTERNS):
                    matched = True
                    sentiment = "negative"
                if matched:
                    # Atomicity: ensure one opinion about one aspect
                    atomic = self._make_atomic(sent)
                    if atomic:
                        candidates.append(Statement(text=atomic, item_id=review.item_id, sentiment=sentiment))
        return candidates

    def verify(self, candidates: List[Statement]) -> List[Statement]:
        """Filter non-explanatory and non-atomic statements."""
        verified = []
        seen = set()
        for stmt in candidates:
            # Explanatoriness: must describe an item fact affecting user experience
            if not self._is_explanatory(stmt.text):
                continue
            # Atomicity: one opinion about one aspect
            if not self._is_atomic(stmt.text):
                continue
            # Uniqueness will be handled in clustering; here just deduplicate exact matches
            key = stmt.text.lower()
            if key in seen:
                continue
            seen.add(key)
            verified.append(stmt)
        return verified

    def _make_atomic(self, text: str) -> str:
        # Simple heuristic: if sentence is too long, truncate to first clause
        if len(text) > 40:
            parts = text.split("，")
            text = parts[0]
        return text[:50]

    def _is_explanatory(self, text: str) -> bool:
        # Must relate to product attribute or user experience
        experience_keywords = ["加热", "温控", "操作", "清洗", "容量", "静音", "续航", "便携", "吸力", "佩戴", "噪音", "速度"]
        return any(kw in text for kw in experience_keywords)

    def _is_atomic(self, text: str) -> bool:
        # Should not contain multiple contradictory opinions or multiple unrelated aspects
        connectors = ["但是", "然而", "不过", "而且", "同时"]
        return not any(c in text for c in connectors)


class SemanticClusterer:
    """
    Scalable semantic clustering pipeline (simplified):
    1) embed statements
    2) approximate nearest-neighbor retrieval
    3) pairwise filtering by cosine similarity threshold
    4) refinement via connected components + cohesion check
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.threshold = similarity_threshold
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower())

    def fit_transform(self, statements: List[Statement]) -> List[List[float]]:
        docs = [self._tokenize(s.text) for s in statements]
        all_terms = set()
        for d in docs:
            all_terms.update(d)
        self.vocab = {t: i for i, t in enumerate(sorted(all_terms))}
        n = len(docs)
        for t in self.vocab:
            df = sum(1 for d in docs if t in d)
            self.idf[t] = math.log((n + 1) / (df + 1)) + 1
        vectors = []
        for d in docs:
            vec = [0.0] * len(self.vocab)
            tf = Counter(d)
            for t, count in tf.items():
                if t in self.vocab:
                    vec[self.vocab[t]] = count * self.idf.get(t, 1.0)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def _cosine(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def cluster(self, statements: List[Statement]) -> List[List[Statement]]:
        if not statements:
            return []
        vectors = self.fit_transform(statements)
        n = len(statements)
        # Build similarity graph edges
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine(vectors[i], vectors[j])
                if sim >= self.threshold:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        # Connected components as initial clusters
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adjacency[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            # Cohesion check: split if intra-cluster cohesion is too low
            refined = self._refine_component(comp, vectors)
            clusters.extend(refined)
        return [[statements[idx] for idx in c] for c in clusters]

    def _refine_component(self, comp: List[int], vectors: List[List[float]]) -> List[List[int]]:
        if len(comp) <= 2:
            return [comp]
        # Compute pairwise similarities within component
        sims = []
        for i in range(len(comp)):
            for j in range(i + 1, len(comp)):
                sims.append(self._cosine(vectors[comp[i]], vectors[comp[j]]))
        avg_sim = sum(sims) / len(sims) if sims else 1.0
        if avg_sim >= 0.65:
            return [comp]
        # Split around highest-degree pivot (most central element)
        pivot = comp[0]
        pivot_sims = {idx: self._cosine(vectors[pivot], vectors[idx]) for idx in comp}
        high = [idx for idx in comp if pivot_sims[idx] >= 0.7]
        low = [idx for idx in comp if pivot_sims[idx] < 0.7]
        result = []
        if high:
            result.append(high)
        if low:
            result.append(low)
        return result if result else [comp]

    def canonicalize(self, clusters: List[List[Statement]]) -> List[Statement]:
        """Select most central statement as canonical representative per cluster."""
        canonical = []
        for cl in clusters:
            if not cl:
                continue
            # Simple centrality: longest statement (proxy for completeness)
            rep = max(cl, key=lambda s: len(s.text))
            canonical.append(Statement(text=rep.text, item_id=rep.item_id, sentiment=rep.sentiment))
        return canonical


class StatementRanker:
    """
    Popularity-based ranking baselines for global-level and item-level.
    """

    def rank_global(self, statements: List[Statement], top_k: int = 5) -> List[Statement]:
        """Rank by overall frequency across all reviews."""
        freq = Counter(s.text for s in statements)
        # Deduplicate by text while preserving first occurrence sentiment
        seen: Dict[str, Statement] = {}
        for s in statements:
            if s.text not in seen:
                seen[s.text] = s
        unique_stmts = list(seen.values())
        sorted_stmts = sorted(
            unique_stmts,
            key=lambda s: (freq[s.text], {"positive": 2, "neutral": 1, "negative": 0}.get(s.sentiment, 1)),
            reverse=True,
        )
        return sorted_stmts[:top_k]

    def rank_by_item(self, statements: List[Statement], item_id: str, top_k: int = 5) -> List[Statement]:
        """Rank by frequency within a specific item."""
        item_stmts = [s for s in statements if s.item_id == item_id]
        return self.rank_global(item_stmts, top_k=top_k)


class StaRPipeline:
    """End-to-end StaR pipeline."""

    def __init__(self, similarity_threshold: float = 0.75):
        self.extractor = StatementExtractor()
        self.clusterer = SemanticClusterer(similarity_threshold=similarity_threshold)
        self.ranker = StatementRanker()

    def process(self, reviews: List[Review]) -> Dict:
        candidates = self.extractor.extract_candidates(reviews)
        verified = self.extractor.verify(candidates)
        clusters = self.clusterer.cluster(verified)
        canonical = self.clusterer.canonicalize(clusters)
        global_ranked = self.ranker.rank_global(canonical, top_k=5)
        return {
            "raw_reviews": len(reviews),
            "candidates": len(candidates),
            "verified": len(verified),
            "clusters": len(clusters),
            "canonical_statements": len(canonical),
            "top_global_statements": [
                {"text": s.text, "sentiment": s.sentiment} for s in global_ranked
            ],
        }


def build_demo_reviews() -> List[Review]:
    return [
        Review(text="这款暖奶器加热非常均匀，不会出现外热内冷的情况，操作简单一键启动。", item_id="momcozy_warmer_v1", rating=5),
        Review(text="操作简单，清洗也方便，但是温控不太精准，有时候会过热。", item_id="momcozy_warmer_v1", rating=4),
        Review(text="加热速度有点慢，不过加热均匀性很好，温度控制也比较精准。", item_id="momcozy_warmer_v1", rating=4),
        Review(text="清洗很方便，设计合理没有死角，操作界面也很直观。", item_id="momcozy_warmer_v1", rating=5),
        Review(text="温控不精准，加热不均匀，有时候底部热了上面还是凉的，体验一般。", item_id="momcozy_warmer_v1", rating=3),
        Review(text="容量大，可以同时热两奶瓶，操作简单老人也会用。", item_id="momcozy_warmer_v2", rating=5),
        Review(text="暖奶器噪音有点大，夜间使用会吵到宝宝，希望能改进静音设计。", item_id="momcozy_warmer_v2", rating=3),
        Review(text="加热均匀性好，温控精准到每一度，操作简单清洗方便。", item_id="momcozy_warmer_v2", rating=5),
    ]


def demo():
    reviews = build_demo_reviews()
    star = StaRPipeline(similarity_threshold=0.65)
    result = star.process(reviews)
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ------------------ Tests ------------------

def test_extraction_and_verification():
    reviews = build_demo_reviews()
    extractor = StatementExtractor()
    candidates = extractor.extract_candidates(reviews)
    verified = extractor.verify(candidates)
    assert len(candidates) > 0
    assert len(verified) <= len(candidates)
    for v in verified:
        assert len(v.text) >= 5
        assert extractor._is_explanatory(v.text)
        assert extractor._is_atomic(v.text)


def test_clustering_produces_unique_canonical():
    reviews = build_demo_reviews()
    pipeline = StaRPipeline(similarity_threshold=0.65)
    candidates = pipeline.extractor.extract_candidates(reviews)
    verified = pipeline.extractor.verify(candidates)
    clusters = pipeline.clusterer.cluster(verified)
    canonical = pipeline.clusterer.canonicalize(clusters)
    texts = [c.text for c in canonical]
    assert len(texts) == len(set(texts))


def test_ranking_returns_top_k():
    reviews = build_demo_reviews()
    pipeline = StaRPipeline(similarity_threshold=0.65)
    result = pipeline.process(reviews)
    assert len(result["top_global_statements"]) <= 5
    for stmt in result["top_global_statements"]:
        assert "text" in stmt
        assert "sentiment" in stmt


def test_item_level_ranking():
    reviews = build_demo_reviews()
    pipeline = StaRPipeline(similarity_threshold=0.65)
    candidates = pipeline.extractor.extract_candidates(reviews)
    verified = pipeline.extractor.verify(candidates)
    clusters = pipeline.clusterer.cluster(verified)
    canonical = pipeline.clusterer.canonicalize(clusters)
    ranked = pipeline.ranker.rank_by_item(canonical, item_id="momcozy_warmer_v1", top_k=3)
    assert len(ranked) <= 3
    for r in ranked:
        assert r.item_id == "momcozy_warmer_v1" or any(s.item_id == "momcozy_warmer_v1" for s in canonical if s.text == r.text)


def test_end_to_end_structure():
    reviews = build_demo_reviews()
    pipeline = StaRPipeline(similarity_threshold=0.65)
    result = pipeline.process(reviews)
    assert result["raw_reviews"] == len(reviews)
    assert result["verified"] <= result["candidates"]
    assert result["clusters"] <= result["verified"]
    assert result["canonical_statements"] <= result["verified"]


if __name__ == "__main__":
    demo()

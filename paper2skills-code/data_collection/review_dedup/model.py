"""
Review Dedup & Quality Filter
整合 FOLD (在线去重) + ResLPO (质量排序) + Scalable ABSA (跨平台标注)

论文来源:
  FOLD:          arXiv:2606.03001
  ResLPO:        arXiv:2601.07449
  Scalable ABSA: arXiv:2602.21082
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Review:
    review_id: str
    text: str
    platform: str
    timestamp: float = 0.0
    rating: float = 3.0
    verified_purchase: bool = False
    helpful_votes: int = 0
    aspects: Dict[str, str] = field(default_factory=dict)
    quality_score: float = 0.0


PLATFORM_PRIORITY = {"amazon": 3, "independent": 2, "tiktok": 1, "other": 0}


class SimHashSignature:
    def __init__(self, n_bits: int = 64, n_shingles: int = 3):
        self.n_bits = n_bits
        self.n_shingles = n_shingles

    def _shingles(self, text: str) -> List[str]:
        tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
        return [" ".join(tokens[i:i+self.n_shingles])
                for i in range(len(tokens) - self.n_shingles + 1)] or [text[:16]]

    def compute(self, text: str) -> int:
        shingles = self._shingles(text)
        v = [0] * self.n_bits
        for s in shingles:
            h = int(hashlib.md5(s.encode()).hexdigest(), 16)
            for i in range(self.n_bits):
                v[i] += 1 if (h >> i) & 1 else -1
        return sum(1 << i for i in range(self.n_bits) if v[i] > 0)

    def similarity(self, sig_a: int, sig_b: int) -> float:
        return 1.0 - bin(sig_a ^ sig_b).count('1') / self.n_bits


class HNSWIndex:
    def __init__(self, max_neighbors: int = 16, similarity_fn=None):
        self.max_neighbors = max_neighbors
        self.signatures: List[Tuple[str, int]] = []
        self.nodes: Dict[str, int] = {}
        self.graph: Dict[int, List[int]] = {}
        self.sim_fn = similarity_fn or (lambda a, b: 1.0 - bin(a ^ b).count('1') / 64)

    def insert(self, node_id: str, signature: int) -> int:
        idx = len(self.signatures)
        self.signatures.append((node_id, signature))
        self.nodes[node_id] = idx
        neighbors = self._find_neighbors(signature, k=self.max_neighbors)
        self.graph[idx] = neighbors
        for nb in neighbors:
            if nb in self.graph:
                self.graph[nb].append(idx)
                if len(self.graph[nb]) > self.max_neighbors * 2:
                    self.graph[nb] = self.graph[nb][:self.max_neighbors * 2]
        return idx

    def _find_neighbors(self, query_sig: int, k: int) -> List[int]:
        if not self.signatures:
            return []
        scored = sorted(
            ((self.sim_fn(query_sig, sig), i) for i, (_, sig) in enumerate(self.signatures)),
            reverse=True
        )
        return [i for _, i in scored[:k]]

    def query(self, signature: int, k: int = 5) -> List[Tuple[float, str]]:
        if not self.signatures:
            return []
        results = [
            (self.sim_fn(signature, sig), node_id)
            for node_id, sig in self.signatures
        ]
        results.sort(reverse=True)
        return results[:k]


class FOLDDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85, n_bits: int = 64):
        self.threshold = similarity_threshold
        self.hasher = SimHashSignature(n_bits=n_bits)
        self.index = HNSWIndex(similarity_fn=lambda a, b: self.hasher.similarity(a, b))
        self.dedup_log: List[Dict] = []

    def is_duplicate(self, review: Review) -> Tuple[bool, Optional[str]]:
        sig = self.hasher.compute(review.text)
        for sim, existing_id in self.index.query(sig, k=3):
            if sim >= self.threshold:
                return True, existing_id
        return False, None

    def add(self, review: Review) -> bool:
        is_dup, dup_of = self.is_duplicate(review)
        if is_dup:
            self.dedup_log.append({"dropped": review.review_id, "duplicate_of": dup_of})
            return False
        self.index.insert(review.review_id, self.hasher.compute(review.text))
        return True

    def process_batch(self, reviews: List[Review]) -> Tuple[List[Review], int]:
        sorted_reviews = sorted(reviews, key=lambda r: PLATFORM_PRIORITY.get(r.platform, 0), reverse=True)
        kept, dropped = [], 0
        for r in sorted_reviews:
            if self.add(r):
                kept.append(r)
            else:
                dropped += 1
        return kept, dropped

    def stats(self) -> Dict[str, Any]:
        total = len(self.index.signatures) + len(self.dedup_log)
        return {
            "indexed": len(self.index.signatures),
            "dropped": len(self.dedup_log),
            "drop_rate": round(len(self.dedup_log) / max(total, 1), 3),
        }


class ReviewQualityScorer:
    ASPECT_SIGNALS = ["suction","noise","battery","portable","assembly","leak",
                      "吸力","静音","噪音","电池","携带","漏奶","组装"]
    VAGUE_WORDS = {"great","good","nice","ok","okay","fine","好","不错","还行","一般"}
    SPECIFIC_PATTERNS = [r'\d+', r'(hour|minute|day|week|小时|分钟|天)', r'(because|since|因为|所以|but|但是)']

    def score(self, review: Review) -> float:
        return (0.4 * self._density(review.text) +
                0.35 * self._specificity(review.text) +
                0.25 * self._credibility(review))

    def _density(self, text: str) -> float:
        hits = sum(1 for s in self.ASPECT_SIGNALS if s.lower() in text.lower())
        return min(1.0, hits / 3.0)

    def _specificity(self, text: str) -> float:
        words = text.lower().split()
        vague = sum(1 for w in words if w in self.VAGUE_WORDS) / max(len(words), 1)
        specific = sum(1 for p in self.SPECIFIC_PATTERNS if re.search(p, text, re.I)) / len(self.SPECIFIC_PATTERNS)
        return max(0.0, specific - vague * 0.5)

    def _credibility(self, r: Review) -> float:
        s = 0.5 + (0.3 if r.verified_purchase else 0)
        s += 0.1 if r.helpful_votes > 5 else 0
        s += 0.1 if r.helpful_votes > 20 else 0
        return min(1.0, s)

    def rank(self, reviews: List[Review], top_k: int = 30) -> List[Review]:
        for r in reviews:
            r.quality_score = self.score(r)
        return sorted(reviews, key=lambda r: r.quality_score, reverse=True)[:top_k]


class CrossPlatformABSATagger:
    VOCAB = {
        "suction": ["suction","pump","motor","power","吸力","泵","电机"],
        "noise": ["noise","quiet","loud","silent","noisy","噪音","静音","声音"],
        "portability": ["portable","compact","lightweight","heavy","便携","轻便","携带"],
        "battery": ["battery","charge","runtime","电池","充电","续航"],
        "assembly": ["assemble","setup","install","complicated","easy","组装","安装"],
        "leakage": ["leak","spill","seal","drip","漏","密封","漏奶"],
    }
    POS = {"great","good","excellent","love","perfect","best","amazing","好","棒","完美","喜欢","强"}
    NEG = {"bad","terrible","poor","hate","worst","awful","broken","leak","差","糟","烂","坏","漏"}

    def tag(self, review: Review) -> Dict[str, str]:
        tl = review.text.lower()
        words = tl.split()
        result = {}
        for aspect, kws in self.VOCAB.items():
            hit_positions = [i for i, w in enumerate(words) if any(k in w for k in kws)]
            if not hit_positions:
                continue
            context_words = set()
            for pos in hit_positions:
                window = words[max(0, pos-4): pos+5]
                context_words.update(window)
            pos_count = sum(1 for w in self.POS if w in context_words)
            neg_count = sum(1 for w in self.NEG if w in context_words)
            result[aspect] = "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
        return result

    def tag_batch(self, reviews: List[Review]) -> List[Review]:
        for r in reviews:
            r.aspects = self.tag(r)
        return reviews


class ReviewQualityPipeline:
    def __init__(self, similarity_threshold: float = 0.85, top_k: int = 30):
        self.deduplicator = FOLDDeduplicator(similarity_threshold)
        self.scorer = ReviewQualityScorer()
        self.tagger = CrossPlatformABSATagger()

    def run(self, raw_reviews: List[Review]) -> Tuple[List[Review], Dict[str, Any]]:
        deduped, dropped = self.deduplicator.process_batch(raw_reviews)
        ranked = self.scorer.rank(deduped, top_k=min(30, len(deduped)))
        tagged = self.tagger.tag_batch(ranked)
        return tagged, {
            "input": len(raw_reviews),
            "after_dedup": len(deduped),
            "dropped": dropped,
            "final": len(tagged),
        }


def test_simhash_near_duplicate():
    h = SimHashSignature()
    s1 = h.compute("This product is amazing! Best pump ever!")
    s2 = h.compute("This product is amazing, best pump ever!")
    sim = h.similarity(s1, s2)
    assert sim > 0.8, f"Near-duplicates should have high similarity: {sim}"
    s3 = h.compute("Terrible product, broke after one week.")
    sim2 = h.similarity(s1, s3)
    assert sim2 < 0.6, f"Different reviews should have low similarity: {sim2}"
    print(f"[PASS] simhash: near-dup={sim:.3f}, different={sim2:.3f}")


def test_fold_dedup_prefers_amazon():
    dedup = FOLDDeduplicator(similarity_threshold=0.80)
    r_amazon = Review("r1", "Great suction power, very quiet motor", "amazon", verified_purchase=True)
    r_tiktok = Review("r2", "Great suction power very quiet motor", "tiktok")
    kept, dropped = dedup.process_batch([r_tiktok, r_amazon])
    assert dropped == 1
    assert kept[0].platform == "amazon"
    print(f"[PASS] fold_dedup: kept={kept[0].platform}, dropped={dropped}")


def test_quality_scorer_ranks_specific_higher():
    scorer = ReviewQualityScorer()
    vague = Review("r1", "Good product, nice", "amazon")
    specific = Review("r2", "Suction power is excellent, battery lasts 3 hours, very quiet motor", "amazon", verified_purchase=True, helpful_votes=25)
    assert scorer.score(specific) > scorer.score(vague)
    print(f"[PASS] quality_scorer: specific={scorer.score(specific):.3f} > vague={scorer.score(vague):.3f}")


def test_absa_tagger():
    tagger = CrossPlatformABSATagger()
    r = Review("r1", "Great suction and quiet noise but the battery is terrible and leaks", "amazon")
    aspects = tagger.tag(r)
    assert aspects.get("suction") == "positive"
    assert aspects.get("noise") == "positive"
    assert aspects.get("battery") == "negative"
    print(f"[PASS] absa_tagger: {aspects}")


def test_pipeline_end_to_end():
    pipeline = ReviewQualityPipeline(similarity_threshold=0.80)
    reviews = [
        Review("r1", "Amazing suction power, very quiet", "amazon", verified_purchase=True, helpful_votes=15),
        Review("r2", "Amazing suction power very quiet", "tiktok"),
        Review("r3", "Great pump, battery lasts 4 hours, no leaks", "amazon", verified_purchase=True),
        Review("r4", "Good product", "other"),
        Review("r5", "Terrible noise and battery keeps dying after 1 hour", "amazon"),
    ]
    results, stats = pipeline.run(reviews)
    assert stats["dropped"] >= 1
    assert len(results) <= 30
    has_aspects = sum(1 for r in results if r.aspects)
    assert has_aspects >= 2, f"At least 2 reviews should have aspects tagged, got {has_aspects}"
    print(f"[PASS] pipeline: input={stats['input']}, deduped={stats['after_dedup']}, final={stats['final']}, dropped={stats['dropped']}, with_aspects={has_aspects}")


if __name__ == "__main__":
    test_simhash_near_duplicate()
    test_fold_dedup_prefers_amazon()
    test_quality_scorer_ranks_specific_higher()
    test_absa_tagger()
    test_pipeline_end_to_end()
    print("\n✅ All tests passed")

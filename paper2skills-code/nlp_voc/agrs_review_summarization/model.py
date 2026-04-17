"""
AGRS: Aspect-Guided Review Summarization at Scale
基于论文: End-to-End Aspect-Guided Review Summarization at Scale
结合ABSA与引导式摘要，生成基于属性的可解释产品评论摘要。
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Review:
    text: str
    review_id: str = ""
    rating: int = 5
    market: str = ""


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str  # positive / negative / mixed
    review_id: str = ""


@dataclass
class ProductSummary:
    product_name: str
    top_aspects: List[Tuple[str, int]] = field(default_factory=list)
    summary_text: str = ""
    selected_reviews_count: int = 0


class AspectExtractor:
    """Simulated LLM-based aspect extraction from individual reviews."""

    ASPECT_KEYWORDS = {
        "消毒效果": (["消毒", "杀菌", "干净", "卫生"], ["消毒不彻底", "杀菌弱", "不干净"]),
        "烘干功能": (["烘干", "干燥", "无水渍"], ["烘干慢", "不干", "有水渍"]),
        "容量大小": (["容量大", "装得多", "空间大"], ["容量小", "装不下", "空间小"]),
        "操作简便性": (["操作简单", "一键", "方便"], ["操作复杂", "难用", "麻烦"]),
        "噪音控制": (["静音", "噪音小", "安静"], ["噪音大", "吵", "声响"]),
        "外观设计": (["好看", "美观", "精致"], ["丑", "粗糙", "廉价感"]),
        "性价比": (["划算", "值得", "性价比高"], ["贵", "不值", "性价比低"]),
    }

    def extract(self, reviews: List[Review]) -> List[AspectSentiment]:
        results = []
        for review in reviews:
            extracted = []
            for aspect, (pos_kw, neg_kw) in self.ASPECT_KEYWORDS.items():
                pos_hit = any(kw in review.text for kw in pos_kw)
                neg_hit = any(kw in review.text for kw in neg_kw)
                if pos_hit and neg_hit:
                    extracted.append((aspect, "mixed"))
                elif pos_hit:
                    extracted.append((aspect, "positive"))
                elif neg_hit:
                    extracted.append((aspect, "negative"))
            # Cap at 5 aspects per review
            for asp, sent in extracted[:5]:
                results.append(AspectSentiment(aspect=asp, sentiment=sent, review_id=review.review_id))
        return results


class AspectConsolidator:
    """
    Maps fine-grained aspects to broader canonical forms.
    Low-frequency aspects below threshold are merged upward.
    """

    def __init__(self, min_freq_threshold: int = 3):
        self.min_freq = min_freq_threshold
        self.canonical_map: Dict[str, str] = {}

    def consolidate(self, aspects: List[AspectSentiment]) -> List[AspectSentiment]:
        freq = Counter(a.aspect for a in aspects)
        # Determine canonical forms: aspects above threshold keep themselves
        # Low-frequency aspects map to a broader category if available
        canonical = {}
        for asp, count in freq.items():
            if count >= self.min_freq:
                canonical[asp] = asp
            else:
                canonical[asp] = self._broaden(asp)

        consolidated = []
        for a in aspects:
            new_asp = canonical.get(a.aspect, a.aspect)
            consolidated.append(AspectSentiment(
                aspect=new_asp,
                sentiment=a.sentiment,
                review_id=a.review_id,
            ))
        return consolidated

    def _broaden(self, aspect: str) -> str:
        # Simple hierarchy mapping
        hierarchy = {
            "消毒效果": "功能表现",
            "烘干功能": "功能表现",
            "容量大小": "使用体验",
            "操作简便性": "使用体验",
            "噪音控制": "使用体验",
            "外观设计": "外观品质",
            "性价比": "购买决策",
        }
        return hierarchy.get(aspect, "综合体验")


class ReviewSelector:
    """Selects top frequent aspects and samples representative reviews for each."""

    def __init__(self, max_aspects: int = 5, max_reviews_per_product: int = 200):
        self.max_aspects = max_aspects
        self.max_reviews = max_reviews_per_product

    def select(
        self,
        aspects: List[AspectSentiment],
        reviews: List[Review],
        max_reviews_per_aspect: int = 10,
    ) -> Tuple[List[Tuple[str, int]], List[Review], Dict[str, List[str]]]:
        review_map = {r.review_id: r for r in reviews}
        # Count consolidated aspects
        aspect_counts = Counter(a.aspect for a in aspects)
        top_aspects = aspect_counts.most_common(self.max_aspects)

        # For each top aspect, sample reviews mentioning it
        aspect_to_reviews: Dict[str, List[str]] = defaultdict(list)
        for a in aspects:
            if a.aspect in [ta[0] for ta in top_aspects]:
                if a.review_id in review_map:
                    aspect_to_reviews[a.aspect].append(a.review_id)

        selected_review_ids: Set[str] = set()
        sampled_aspect_reviews: Dict[str, List[str]] = {}
        for asp, _ in top_aspects:
            ids = list(set(aspect_to_reviews.get(asp, [])))
            # Weighted random sampling: prefer diverse reviews
            if len(ids) > max_reviews_per_aspect:
                sampled = random.sample(ids, max_reviews_per_aspect)
            else:
                sampled = ids
            sampled_aspect_reviews[asp] = sampled
            selected_review_ids.update(sampled)

        selected_reviews = [review_map[r_id] for r_id in selected_review_ids if r_id in review_map]
        # Cap total reviews
        if len(selected_reviews) > self.max_reviews:
            selected_reviews = selected_reviews[:self.max_reviews]
        return top_aspects, selected_reviews, sampled_aspect_reviews


class GuidedSummarizer:
    """Generates product-level summary from consolidated aspects and selected reviews."""

    def summarize(
        self,
        product_name: str,
        top_aspects: List[Tuple[str, int]],
        aspect_sentiments: List[AspectSentiment],
        sampled_reviews: Dict[str, List[str]],
        review_map: Dict[str, Review],
    ) -> str:
        # Build sentiment summary per top aspect
        lines = []
        for aspect, count in top_aspects:
            sentiments = [a.sentiment for a in aspect_sentiments if a.aspect == aspect]
            pos = sentiments.count("positive")
            neg = sentiments.count("negative")
            mix = sentiments.count("mixed")
            total = len(sentiments)
            if total == 0:
                continue
            pos_ratio = pos / total
            neg_ratio = neg / total
            if pos_ratio >= 0.7:
                tone = "满意度高"
            elif neg_ratio >= 0.5:
                tone = "吐槽较多"
            elif mix >= 0.3:
                tone = "评价分化"
            else:
                tone = "整体尚可"
            lines.append(f"{aspect}（提及{count}次）{tone}")

        # Build a concise narrative
        if not lines:
            return f"{product_name}暂无足够评论数据生成摘要。"

        summary = f"关于{product_name}，用户最关注的是" + "、".join([asp for asp, _ in top_aspects[:3]]) + "。"
        summary += "具体而言，" + "；".join(lines) + "。"
        summary += "这些反馈主要来源于" + str(len(set(review_map[rid].review_id for rs in sampled_reviews.values() for rid in rs if rid in review_map))) + "条代表性评论。"
        return summary[:300]


class AGRSPipeline:
    """End-to-end Aspect-Guided Review Summarization pipeline."""

    def __init__(self, min_freq: int = 3, max_aspects: int = 5, max_reviews: int = 200):
        self.extractor = AspectExtractor()
        self.consolidator = AspectConsolidator(min_freq_threshold=min_freq)
        self.selector = ReviewSelector(max_aspects=max_aspects, max_reviews_per_product=max_reviews)
        self.summarizer = GuidedSummarizer()

    def summarize_product(self, product_name: str, reviews: List[Review]) -> ProductSummary:
        raw_aspects = self.extractor.extract(reviews)
        consolidated = self.consolidator.consolidate(raw_aspects)
        top_aspects, selected_reviews, sampled = self.selector.select(consolidated, reviews)
        review_map = {r.review_id: r for r in reviews}
        summary_text = self.summarizer.summarize(product_name, top_aspects, consolidated, sampled, review_map)
        return ProductSummary(
            product_name=product_name,
            top_aspects=top_aspects,
            summary_text=summary_text,
            selected_reviews_count=len(selected_reviews),
        )


def build_demo_reviews() -> List[Review]:
    return [
        Review(review_id="r1", text="这款Momcozy紫外线消毒器消毒效果很好，烘干功能也不错，但是容量有点小，放不下全套吸奶器配件。", rating=4, market="US"),
        Review(review_id="r2", text="操作简单一键启动，消毒效果彻底，就是烘干时间太长了，而且运行时噪音有点大。", rating=3, market="US"),
        Review(review_id="r3", text="外观设计很好看，放在厨房里很美观，消毒效果和烘干功能都很满意，容量也够用。", rating=5, market="DE"),
        Review(review_id="r4", text="性价比很高，消毒烘干一体很方便，但是操作面板有点复杂，老人不太会用。", rating=4, market="US"),
        Review(review_id="r5", text="容量大，可以同时消毒奶瓶和吸奶器，烘干后没有水渍，静音设计很好。", rating=5, market="DE"),
        Review(review_id="r6", text="消毒效果不错，但是噪音控制一般，晚上使用会影响宝宝睡觉，希望能改进。", rating=3, market="US"),
        Review(review_id="r7", text="操作简单，消毒彻底，外观设计精致，性价比也很高，整体非常满意。", rating=5, market="US"),
        Review(review_id="r8", text="烘干功能不太稳定，有时候奶瓶内壁还有水珠，但消毒效果没问题。", rating=3, market="DE"),
    ]


def demo():
    reviews = build_demo_reviews()
    pipeline = AGRSPipeline(min_freq=2, max_aspects=5, max_reviews=200)
    summary = pipeline.summarize_product("Momcozy 紫外线消毒器", reviews)
    result = {
        "product": summary.product_name,
        "top_aspects": summary.top_aspects,
        "selected_reviews": summary.selected_reviews_count,
        "summary": summary.summary_text,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ------------------ Tests ------------------

def test_aspect_extraction():
    reviews = build_demo_reviews()
    extractor = AspectExtractor()
    aspects = extractor.extract(reviews)
    assert len(aspects) > 0
    for a in aspects:
        assert a.aspect != ""
        assert a.sentiment in ["positive", "negative", "mixed"]


def test_aspect_consolidation():
    reviews = build_demo_reviews()
    extractor = AspectExtractor()
    aspects = extractor.extract(reviews)
    consolidator = AspectConsolidator(min_freq_threshold=2)
    consolidated = consolidator.consolidate(aspects)
    assert len(consolidated) == len(aspects)
    # Some low-freq aspects may be broadened


def test_review_selection_caps():
    reviews = build_demo_reviews()
    extractor = AspectExtractor()
    aspects = extractor.extract(reviews)
    consolidator = AspectConsolidator(min_freq_threshold=2)
    consolidated = consolidator.consolidate(aspects)
    selector = ReviewSelector(max_aspects=5, max_reviews_per_product=200)
    top_aspects, selected_reviews, sampled = selector.select(consolidated, reviews)
    assert len(top_aspects) <= 5
    assert len(selected_reviews) <= 200


def test_summary_output():
    reviews = build_demo_reviews()
    pipeline = AGRSPipeline(min_freq=2, max_aspects=5, max_reviews=200)
    summary = pipeline.summarize_product("Momcozy 紫外线消毒器", reviews)
    assert summary.product_name == "Momcozy 紫外线消毒器"
    assert len(summary.top_aspects) > 0
    assert len(summary.summary_text) > 0
    assert summary.selected_reviews_count > 0


def test_end_to_end_structure():
    reviews = build_demo_reviews()
    pipeline = AGRSPipeline(min_freq=2, max_aspects=5, max_reviews=200)
    summary = pipeline.summarize_product("Momcozy 紫外线消毒器", reviews)
    assert isinstance(summary.summary_text, str)
    assert all(isinstance(a, tuple) and len(a) == 2 for a in summary.top_aspects)


if __name__ == "__main__":
    demo()

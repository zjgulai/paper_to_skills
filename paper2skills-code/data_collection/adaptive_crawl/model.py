"""
Adaptive Crawl Scheduling
整合 SB-CLASSIFIER (Sleeping Bandit) + Neural Prioritisation

论文来源:
  SB-CLASSIFIER: arXiv:2602.11874 (EDBT 2026)
  Neural Prioritisation: arXiv:2506.16146
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CrawlTarget:
    url: str
    priority_score: float = 0.5
    last_crawled: float = 0.0
    change_frequency: float = 1.0
    quality_score: float = 0.5
    crawl_count: int = 0
    relevant: Optional[bool] = None


class SleepingBanditScheduler:
    """SB-CLASSIFIER 风格：Sleeping Bandit 聚焦爬取调度"""

    def __init__(self, exploration_coeff: float = 0.3):
        self.beta = exploration_coeff
        self.targets: Dict[str, CrawlTarget] = {}
        self.t = 0

    def register(self, url: str, initial_priority: float = 0.5):
        self.targets[url] = CrawlTarget(url=url, priority_score=initial_priority)

    def select_next(self, budget: int = 10) -> List[str]:
        self.t += 1
        scored = []
        for url, target in self.targets.items():
            n = max(target.crawl_count, 1)
            ucb = target.priority_score + self.beta * math.sqrt(math.log(self.t + 1) / n)
            scored.append((ucb, url))
        scored.sort(reverse=True)
        return [url for _, url in scored[:budget]]

    def update(self, url: str, is_relevant: bool, quality: float, current_time: float):
        if url not in self.targets:
            return
        t = self.targets[url]
        t.crawl_count += 1
        t.relevant = is_relevant
        t.quality_score = quality
        alpha = 0.3
        new_priority = quality if is_relevant else 0.1
        t.priority_score = (1 - alpha) * t.priority_score + alpha * new_priority
        if t.last_crawled > 0:
            elapsed = current_time - t.last_crawled
            if elapsed > 0:
                t.change_frequency = 0.7 * t.change_frequency + 0.3 * (1.0 / elapsed * 3600)
        t.last_crawled = current_time

    def stats(self) -> Dict[str, Any]:
        crawled = sum(1 for t in self.targets.values() if t.crawl_count > 0)
        relevant = sum(1 for t in self.targets.values() if t.relevant is True)
        return {
            "total": len(self.targets),
            "crawled": crawled,
            "relevant_found": relevant,
            "precision": relevant / max(crawled, 1),
        }


class NeuralQualityPrioritiser:
    """Neural Prioritisation 风格：语义质量评估替代链接图优先级"""

    TARGET_KEYWORDS = {
        "pump": ["pump", "suction", "motor", "breast", "吸奶", "电动", "手动"],
        "baby": ["baby", "infant", "newborn", "母婴", "婴儿", "新生"],
        "supplier": ["supplier", "manufacturer", "factory", "OEM", "供应商", "制造"],
        "competitor": ["competitor", "brand", "product", "竞品", "品牌", "对手"],
    }

    def score(self, url: str, page_snippet: str, target_domain: str) -> float:
        tl = page_snippet.lower()
        keywords = self.TARGET_KEYWORDS.get(target_domain, [])
        keyword_hits = sum(1 for kw in keywords if kw.lower() in tl)
        keyword_score = min(1.0, keyword_hits / max(len(keywords) * 0.3, 1))
        url_score = 0.3 if any(kw in url.lower() for kw in keywords) else 0.0
        length_score = min(1.0, len(page_snippet.split()) / 100.0)
        return 0.5 * keyword_score + 0.3 * url_score + 0.2 * length_score


class AdaptiveCrawlPipeline:
    def __init__(self, target_domain: str = "pump", budget_per_round: int = 5):
        self.scheduler = SleepingBanditScheduler()
        self.prioritiser = NeuralQualityPrioritiser()
        self.target_domain = target_domain
        self.budget = budget_per_round
        self.pages_crawled = 0
        self.pages_relevant = 0

    def seed_urls(self, urls: List[str]):
        for url in urls:
            self.scheduler.register(url)

    def process_crawl_result(self, url: str, page_snippet: str, current_time: float):
        quality = self.prioritiser.score(url, page_snippet, self.target_domain)
        is_relevant = quality > 0.3
        self.scheduler.update(url, is_relevant, quality, current_time)
        self.pages_crawled += 1
        if is_relevant:
            self.pages_relevant += 1
        return is_relevant, quality

    def next_round(self) -> List[str]:
        return self.scheduler.select_next(self.budget)

    def efficiency(self) -> float:
        return self.pages_relevant / max(self.pages_crawled, 1)


def test_bandit_prioritises_relevant():
    scheduler = SleepingBanditScheduler(exploration_coeff=0.1)
    scheduler.register("http://relevant.com", 0.5)
    scheduler.register("http://irrelevant.com", 0.5)
    scheduler.update("http://relevant.com", True, 0.9, 1000.0)
    scheduler.update("http://irrelevant.com", False, 0.1, 1000.0)
    selected = scheduler.select_next(1)
    assert selected[0] == "http://relevant.com"
    print(f"[PASS] bandit_priority: selected={selected[0]}")


def test_neural_quality_score():
    prioritiser = NeuralQualityPrioritiser()
    relevant = "Momcozy M9 breast pump suction motor review infant baby"
    irrelevant = "Best pizza restaurant in New York City"
    s1 = prioritiser.score("http://pump-review.com", relevant, "pump")
    s2 = prioritiser.score("http://pizza.com", irrelevant, "pump")
    assert s1 > s2
    print(f"[PASS] neural_quality: relevant={s1:.3f}, irrelevant={s2:.3f}")


def test_adaptive_pipeline():
    random.seed(42)
    pipeline = AdaptiveCrawlPipeline(target_domain="pump", budget_per_round=3)
    pipeline.seed_urls([
        "http://pump-supplier.com", "http://baby-products.com",
        "http://news-site.com", "http://forum.com", "http://competitor.com",
    ])
    pages = {
        "http://pump-supplier.com": "electric breast pump suction motor baby infant manufacturer",
        "http://baby-products.com": "baby pump suction motor products",
        "http://news-site.com": "stock market news today",
        "http://forum.com": "general discussion forum",
        "http://competitor.com": "competitor pump brand products",
    }
    for round_num in range(3):
        urls = pipeline.next_round()
        for url in urls:
            snippet = pages.get(url, "generic content")
            pipeline.process_crawl_result(url, snippet, float(round_num * 1000))
    stats = pipeline.scheduler.stats()
    assert stats["relevant_found"] >= 1
    assert pipeline.efficiency() > 0
    print(f"[PASS] adaptive_pipeline: crawled={stats['crawled']}, relevant={stats['relevant_found']}, efficiency={pipeline.efficiency():.2f}")


if __name__ == "__main__":
    test_bandit_prioritises_relevant()
    test_neural_quality_score()
    test_adaptive_pipeline()
    print("\n✅ All tests passed")

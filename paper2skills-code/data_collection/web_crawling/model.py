"""
LLM-Focused Web Crawling
整合 W→K→W Pipeline + Webscraper MLLM

论文来源:
  W→K→W:          arXiv:2602.24262
  Webscraper MLLM: arXiv:2603.29161
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Entity:
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_url: str = ""


@dataclass
class KGRelation:
    subject: str
    predicate: str
    obj: str


@dataclass
class KnowledgeGraph:
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[KGRelation] = field(default_factory=list)

    def add_entity(self, entity: Entity):
        self.entities[entity.name] = entity

    def add_relation(self, rel: KGRelation):
        self.relations.append(rel)

    def get_gaps(self, required_attrs: List[str]) -> List[Tuple[str, List[str]]]:
        return [
            (name, [a for a in required_attrs if a not in ent.attributes])
            for name, ent in self.entities.items()
            if any(a not in ent.attributes for a in required_attrs)
        ]

    def get_neighbors(self, entity_name: str) -> List[str]:
        return [
            r.obj if r.subject == entity_name else r.subject
            for r in self.relations
            if entity_name in (r.subject, r.obj)
        ]


class WebToKGExtractor:
    CERT_RE = re.compile(r'\b(CE|FDA|RoHS|REACH|EN71|ASTM|ISO\s*\d+)\b', re.I)
    CAP_RE = re.compile(r'(\d+)\s*(?:万|k)?\s*(?:件|units?|pcs?)\s*(?:per|/)\s*(?:month|year|年|月)', re.I)
    PARTNER_RE = re.compile(
        r'(?:supplier|manufacturer|factory|代工|供应商)[:\s]+([A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff\s]{1,30})',
        re.I
    )

    def extract(self, text: str, url: str, entity_name: str) -> Tuple[List[Entity], List[KGRelation]]:
        entities, relations = [], []
        attrs: Dict[str, Any] = {}
        certs = list({c.upper() for c in self.CERT_RE.findall(text)})
        if certs:
            attrs["certifications"] = certs
        cap = self.CAP_RE.search(text)
        if cap:
            attrs["monthly_capacity"] = int(cap.group(1))
        entities.append(Entity(entity_name, "supplier", attrs, url))
        for m in self.PARTNER_RE.finditer(text):
            partner = m.group(1).strip()
            if 2 < len(partner) < 40 and partner.lower() != entity_name.lower():
                entities.append(Entity(partner, "company", {}, url))
                relations.append(KGRelation(entity_name, "cooperates_with", partner))
        return entities, relations


class KGGapAnalyzer:
    REQUIRED = ["certifications", "monthly_capacity", "min_order", "price_range"]
    QUERY_TEMPLATES = {
        "certifications": "{name} CE FDA certification compliance",
        "monthly_capacity": "{name} production capacity annual output",
        "min_order": "{name} minimum order quantity MOQ",
        "price_range": "{name} OEM price quotation wholesale",
    }

    def analyze_gaps(self, kg: KnowledgeGraph) -> List[Dict[str, Any]]:
        targets = []
        for name, missing in kg.get_gaps(self.REQUIRED):
            if not missing:
                continue
            queries = [self.QUERY_TEMPLATES[a].format(name=name) for a in missing if a in self.QUERY_TEMPLATES]
            targets.append({"entity": name, "missing": missing, "queries": queries, "priority": len(missing)})
        return sorted(targets, key=lambda x: x["priority"], reverse=True)


@dataclass
class ScrapedProduct:
    asin: str
    title: str
    price: float
    rating: float
    review_count: int
    availability: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class MLLMWebScraper:
    PRICE_RE = re.compile(r'\$\s*(\d+(?:\.\d+)?)')
    RATING_RE = re.compile(r'(\d+\.?\d*)\s*(?:out of 5|/5|stars?)', re.I)
    REVIEW_RE = re.compile(r'([\d,]+)\s*(?:ratings?|reviews?)', re.I)
    ASIN_RE = re.compile(r'\b([A-Z0-9]{10})\b')

    def _understand(self, text: str) -> Dict[str, Any]:
        return {
            "page_type": "product_detail" if any(k in text.lower() for k in ["add to cart", "asin", "product information"]) else "listing",
            "has_variants": "variant" in text.lower(),
        }

    def _extract(self, text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        prices = [float(m.group(1)) for m in self.PRICE_RE.finditer(text)]
        if prices:
            result["price"] = min(prices)
        m = self.RATING_RE.search(text)
        if m:
            result["rating"] = float(m.group(1))
        m = self.REVIEW_RE.search(text)
        if m:
            result["review_count"] = int(m.group(1).replace(",", ""))
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            result["title"] = lines[0]
        if re.search(r'in stock|available|ships', text, re.I):
            result["availability"] = "In Stock"
        elif re.search(r'out of stock|unavailable', text, re.I):
            result["availability"] = "Out of Stock"
        else:
            result["availability"] = "Unknown"
        return result

    def _validate(self, data: Dict[str, Any]) -> float:
        issues = 0
        if data.get("price", 0) and not (0.01 <= data["price"] <= 10000):
            issues += 1
        if data.get("rating", 0) and not (0 <= data["rating"] <= 5):
            issues += 1
        return max(0.0, 1.0 - issues * 0.2)

    def scrape(self, page_text: str, asin: str = "") -> Optional[ScrapedProduct]:
        raw = self._extract(page_text)
        confidence = self._validate(raw)
        if not raw.get("title") and not raw.get("price"):
            return None
        if not asin:
            m = self.ASIN_RE.search(page_text)
            asin = m.group(1) if m else "UNKNOWN"
        return ScrapedProduct(
            asin=asin, title=raw.get("title", ""),
            price=raw.get("price", 0.0), rating=raw.get("rating", 0.0),
            review_count=raw.get("review_count", 0),
            availability=raw.get("availability", "Unknown"),
            raw_data=raw, confidence=confidence,
        )


class WKWCrawler:
    def __init__(self, max_iterations: int = 3):
        self.kg = KnowledgeGraph()
        self.extractor = WebToKGExtractor()
        self.gap_analyzer = KGGapAnalyzer()
        self.visited: Set[str] = set()
        self.max_iterations = max_iterations
        self.log: List[Dict] = []

    def seed(self, entities: List[Dict[str, Any]]):
        for e in entities:
            self.kg.add_entity(Entity(e["name"], e.get("type", "unknown"), e.get("attributes", {})))

    def process_page(self, url: str, text: str, entity_name: str):
        if url in self.visited:
            return
        self.visited.add(url)
        ents, rels = self.extractor.extract(text, url, entity_name)
        for ent in ents:
            if ent.name not in self.kg.entities:
                self.kg.add_entity(ent)
            else:
                self.kg.entities[ent.name].attributes.update(ent.attributes)
        for rel in rels:
            self.kg.add_relation(rel)
        self.log.append({"url": url, "entities": len(ents), "relations": len(rels)})

    def get_next_targets(self) -> List[Dict[str, Any]]:
        return self.gap_analyzer.analyze_gaps(self.kg)

    def stats(self) -> Dict[str, Any]:
        return {
            "entities": len(self.kg.entities),
            "relations": len(self.kg.relations),
            "pages_crawled": len(self.visited),
            "gaps_remaining": len(self.gap_analyzer.analyze_gaps(self.kg)),
        }


def test_wkw_kg_construction():
    crawler = WKWCrawler()
    crawler.seed([{"name": "ABC Electronics", "type": "supplier"}])
    page1 = "ABC Electronics is CE FDA certified. Supplier: XYZ Factory for Momcozy. Annual capacity 50万 units/year."
    crawler.process_page("http://abc.com", page1, "ABC Electronics")
    assert "ABC Electronics" in crawler.kg.entities
    assert crawler.kg.entities["ABC Electronics"].attributes.get("certifications")
    assert any("CE" in c for c in crawler.kg.entities["ABC Electronics"].attributes["certifications"])
    print(f"[PASS] wkw_kg: entities={len(crawler.kg.entities)}, relations={len(crawler.kg.relations)}")


def test_wkw_gap_analysis():
    crawler = WKWCrawler()
    crawler.seed([{"name": "Supplier A", "type": "supplier", "attributes": {"certifications": ["CE"]}}])
    gaps = crawler.get_next_targets()
    assert len(gaps) > 0
    missing = gaps[0]["missing"]
    assert "certifications" not in missing
    assert "monthly_capacity" in missing or "min_order" in missing
    print(f"[PASS] wkw_gaps: {len(gaps)} targets, missing={gaps[0]['missing'][:2]}")


def test_mllm_scraper_product():
    scraper = MLLMWebScraper()
    page = "Momcozy M9 Pro Breast Pump\nB0C1234567\n$89.99\n4.6 out of 5 stars\n12,847 ratings\nIn Stock - Ships in 1-2 days"
    result = scraper.scrape(page, "B0C1234567")
    assert result is not None
    assert abs(result.price - 89.99) < 0.01
    assert result.rating == 4.6
    assert result.review_count == 12847
    assert result.availability == "In Stock"
    print(f"[PASS] mllm_scraper: price={result.price}, rating={result.rating}, reviews={result.review_count}")


def test_mllm_scraper_validation():
    scraper = MLLMWebScraper()
    bad_page = "Some Product\n$99999.99\n6.0 out of 5 stars"
    result = scraper.scrape(bad_page)
    if result:
        assert result.confidence < 1.0
    print(f"[PASS] mllm_validation: confidence={result.confidence if result else 'N/A (no product)'}")


def test_full_wkw_iteration():
    crawler = WKWCrawler()
    crawler.seed([
        {"name": "Supplier A", "type": "supplier"},
        {"name": "Supplier B", "type": "supplier"},
    ])
    pages = [
        ("http://a.com", "Supplier A CE certified. capacity 30万 units/year.", "Supplier A"),
        ("http://b.com", "Supplier B FDA RoHS certified. Cooperates with BrandX.", "Supplier B"),
    ]
    for url, text, name in pages:
        crawler.process_page(url, text, name)
    stats = crawler.stats()
    assert stats["entities"] >= 2
    assert stats["pages_crawled"] == 2
    targets = crawler.get_next_targets()
    assert len(targets) > 0
    print(f"[PASS] full_iteration: {stats}, next_targets={len(targets)}")


if __name__ == "__main__":
    test_wkw_kg_construction()
    test_wkw_gap_analysis()
    test_mllm_scraper_product()
    test_mllm_scraper_validation()
    test_full_wkw_iteration()
    print("\n✅ All tests passed")

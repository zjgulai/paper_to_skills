"""
Web Page Change Detection
整合 DiffSpot (VLM视觉差异) + DOM Atomicity (TOCTOU保护)

论文来源:
  DiffSpot:    arXiv:2605.29615
  DOM Atomicity: arXiv:2603.00476
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class PageSnapshot:
    url: str
    timestamp: float
    content_hash: str
    key_fields: Dict[str, Any] = field(default_factory=dict)
    dom_signature: str = ""


@dataclass
class ChangeReport:
    url: str
    changed: bool
    field_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    change_score: float = 0.0
    requires_full_crawl: bool = False


class DOMSignatureExtractor:
    PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
    RATING_RE = re.compile(r"(\d+\.?\d*)\s*(?:out of 5|stars?)", re.I)
    REVIEW_RE = re.compile(r"([\d,]+)\s*(?:ratings?|reviews?)", re.I)
    AVAIL_RE = re.compile(r"(in stock|out of stock|unavailable|available)", re.I)
    BADGE_RE = re.compile(r"(best seller|amazon.s choice|#\d+ in)", re.I)

    def extract(self, page_text: str) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        prices = [float(m.group(1)) for m in self.PRICE_RE.finditer(page_text)]
        if prices:
            fields["price"] = min(prices)
        m = self.RATING_RE.search(page_text)
        if m:
            fields["rating"] = float(m.group(1))
        m = self.REVIEW_RE.search(page_text)
        if m:
            fields["review_count"] = int(m.group(1).replace(",", ""))
        m = self.AVAIL_RE.search(page_text)
        if m:
            fields["availability"] = m.group(1).lower()
        badges = self.BADGE_RE.findall(page_text)
        if badges:
            fields["badges"] = [b.lower() for b in badges]
        return fields

    def dom_signature(self, page_text: str) -> str:
        words = sorted(set(re.sub(r"\s+", " ", page_text.lower()).split()))
        return hashlib.md5(" ".join(words[:50]).encode()).hexdigest()[:16]

    def snapshot(self, url: str, page_text: str) -> PageSnapshot:
        fields = self.extract(page_text)
        content_hash = hashlib.md5(page_text.encode()).hexdigest()
        dom_sig = self.dom_signature(page_text)
        return PageSnapshot(url=url, timestamp=time.time(),
                            content_hash=content_hash, key_fields=fields, dom_signature=dom_sig)


class ChangeDetector:
    """DiffSpot 风格：结构化字段差异检测"""

    CHANGE_WEIGHTS = {"price": 0.4, "availability": 0.35, "rating": 0.1,
                      "review_count": 0.1, "badges": 0.05}

    def __init__(self, price_threshold: float = 0.02, change_trigger: float = 0.15):
        self.price_threshold = price_threshold
        self.change_trigger = change_trigger

    def compare(self, old: PageSnapshot, new: PageSnapshot) -> ChangeReport:
        if old.content_hash == new.content_hash:
            return ChangeReport(url=old.url, changed=False)

        changes: Dict[str, Tuple[Any, Any]] = {}
        score = 0.0

        for field_name, weight in self.CHANGE_WEIGHTS.items():
            old_val = old.key_fields.get(field_name)
            new_val = new.key_fields.get(field_name)
            if old_val is None and new_val is None:
                continue
            if old_val != new_val:
                if field_name == "price" and old_val and new_val:
                    price_change = abs(new_val - old_val) / old_val
                    if price_change >= self.price_threshold:
                        changes[field_name] = (old_val, new_val)
                        score += weight * min(1.0, price_change * 5)
                else:
                    changes[field_name] = (old_val, new_val)
                    score += weight

        changed = score >= self.change_trigger or bool(changes)
        return ChangeReport(
            url=old.url, changed=changed, field_changes=changes,
            change_score=score, requires_full_crawl=score > 0.5,
        )


class TOCTOUGuard:
    """DOM Atomicity 风格：执行前校验防止 TOCTOU 漏洞"""

    def __init__(self, max_age_seconds: float = 30.0):
        self.max_age = max_age_seconds
        self._observed: Dict[str, Tuple[str, float]] = {}

    def observe(self, url: str, dom_signature: str):
        self._observed[url] = (dom_signature, time.time())

    def verify_before_action(self, url: str, current_dom_signature: str) -> Tuple[bool, str]:
        if url not in self._observed:
            return False, "no_prior_observation"
        observed_sig, observed_time = self._observed[url]
        if time.time() - observed_time > self.max_age:
            return False, "observation_expired"
        if observed_sig != current_dom_signature:
            return False, "dom_changed_since_observation"
        return True, "ok"


class ChangeMonitorPipeline:
    def __init__(self, price_threshold: float = 0.02):
        self.extractor = DOMSignatureExtractor()
        self.detector = ChangeDetector(price_threshold)
        self.guard = TOCTOUGuard()
        self.snapshots: Dict[str, PageSnapshot] = {}

    def register(self, url: str, page_text: str):
        snap = self.extractor.snapshot(url, page_text)
        self.snapshots[url] = snap
        self.guard.observe(url, snap.dom_signature)

    def check_change(self, url: str, new_page_text: str) -> ChangeReport:
        new_snap = self.extractor.snapshot(url, new_page_text)
        if url not in self.snapshots:
            self.snapshots[url] = new_snap
            return ChangeReport(url=url, changed=False)
        report = self.detector.compare(self.snapshots[url], new_snap)
        if report.changed:
            self.snapshots[url] = new_snap
            self.guard.observe(url, new_snap.dom_signature)
        return report


def test_no_change_detection():
    pipeline = ChangeMonitorPipeline()
    text = "Momcozy M9\n$89.99\n4.6 out of 5 stars\n12,847 ratings\nIn Stock"
    pipeline.register("http://amazon.com/B001", text)
    report = pipeline.check_change("http://amazon.com/B001", text)
    assert not report.changed
    print(f"[PASS] no_change: changed={report.changed}")


def test_price_change_detection():
    pipeline = ChangeMonitorPipeline(price_threshold=0.01)
    old_text = "Momcozy M9\n$89.99\n4.6 out of 5 stars\n12,847 ratings\nIn Stock"
    new_text = "Momcozy M9\n$79.99\n4.6 out of 5 stars\n12,900 ratings\nIn Stock"
    pipeline.register("http://amazon.com/B001", old_text)
    report = pipeline.check_change("http://amazon.com/B001", new_text)
    assert report.changed
    assert "price" in report.field_changes
    old_price, new_price = report.field_changes["price"]
    assert new_price < old_price
    print(f"[PASS] price_change: ${old_price}→${new_price}, score={report.change_score:.3f}")


def test_availability_change():
    pipeline = ChangeMonitorPipeline()
    in_stock = "Product\n$45.00\n4.5 stars\nIn Stock"
    out_stock = "Product\n$45.00\n4.5 stars\nOut of Stock"
    pipeline.register("http://example.com/p1", in_stock)
    report = pipeline.check_change("http://example.com/p1", out_stock)
    assert report.changed
    assert "availability" in report.field_changes
    print(f"[PASS] availability_change: {report.field_changes['availability']}")


def test_toctou_guard():
    guard = TOCTOUGuard(max_age_seconds=1.0)
    guard.observe("http://page.com", "sig_abc")
    ok, reason = guard.verify_before_action("http://page.com", "sig_abc")
    assert ok and reason == "ok"
    ok2, reason2 = guard.verify_before_action("http://page.com", "sig_xyz")
    assert not ok2 and "changed" in reason2
    print(f"[PASS] toctou_guard: same_sig={ok}, changed_sig={reason2}")


if __name__ == "__main__":
    test_no_change_detection()
    test_price_change_detection()
    test_availability_change()
    test_toctou_guard()
    print("\n✅ All tests passed")

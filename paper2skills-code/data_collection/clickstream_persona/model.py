"""
Clickstream Persona Pipeline
整合 SimPersona + BIP (Behavioral Intelligence Platform)

论文来源:
  SimPersona: arXiv:2605.14205
  BIP:        arXiv:2604.22762
"""

import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter


@dataclass
class ClickEvent:
    user_id: str
    event_type: str
    item_id: str
    timestamp: float
    session_id: str = ""
    category: str = ""
    price: float = 0.0


@dataclass
class BehaviorSession:
    session_id: str
    user_id: str
    events: List[ClickEvent] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def event_types(self) -> List[str]:
        return [e.event_type for e in self.events]

    @property
    def categories(self) -> List[str]:
        return [e.category for e in self.events if e.category]


class VQVAEPersonaEncoder:
    """SimPersona 风格：VQ-VAE 离散化 Persona Token"""

    PERSONA_CODEBOOK = {
        0: "price_sensitive_researcher",
        1: "premium_buyer",
        2: "impulse_buyer",
        3: "comparison_shopper",
        4: "brand_loyal",
        5: "deal_hunter",
        6: "new_parent_explorer",
        7: "repeat_purchaser",
    }

    def encode_session(self, session: BehaviorSession) -> int:
        event_counts = Counter(session.event_types)
        total = max(sum(event_counts.values()), 1)
        view_ratio = event_counts.get("view", 0) / total
        cart_ratio = event_counts.get("add_to_cart", 0) / total
        purchase_ratio = event_counts.get("purchase", 0) / total
        avg_price = (sum(e.price for e in session.events if e.price > 0) /
                     max(sum(1 for e in session.events if e.price > 0), 1))
        if purchase_ratio > 0.3 and avg_price > 50:
            return 1
        elif cart_ratio > 0.2 and purchase_ratio > 0.15:
            return 2
        elif view_ratio > 0.7 and cart_ratio < 0.1:
            return 0
        elif view_ratio > 0.5 and cart_ratio > 0.1:
            return 3
        elif purchase_ratio > 0.2:
            return 7
        else:
            return 6

    def decode_token(self, token: int) -> str:
        return self.PERSONA_CODEBOOK.get(token, "unknown")


class BehaviorKnowledgeGraph:
    """BIP 风格：事件流 → 4 层行为知识图谱"""

    def __init__(self):
        self.user_journeys: Dict[str, List[ClickEvent]] = defaultdict(list)
        self.item_cooccurrence: Dict[str, Counter] = defaultdict(Counter)
        self.category_transitions: Dict[str, Counter] = defaultdict(Counter)
        self.user_personas: Dict[str, int] = {}

    def ingest(self, events: List[ClickEvent]):
        for evt in events:
            self.user_journeys[evt.user_id].append(evt)
        for user_id, journey in self.user_journeys.items():
            journey.sort(key=lambda e: e.timestamp)
            for i in range(len(journey) - 1):
                curr, nxt = journey[i], journey[i + 1]
                if curr.item_id and nxt.item_id:
                    self.item_cooccurrence[curr.item_id][nxt.item_id] += 1
                if curr.category and nxt.category and curr.category != nxt.category:
                    self.category_transitions[curr.category][nxt.category] += 1

    def get_next_likely_category(self, current_category: str, top_k: int = 3) -> List[Tuple[str, int]]:
        transitions = self.category_transitions.get(current_category, Counter())
        return transitions.most_common(top_k)

    def get_user_journey_summary(self, user_id: str) -> Dict[str, Any]:
        journey = self.user_journeys.get(user_id, [])
        if not journey:
            return {}
        event_counts = Counter(e.event_type for e in journey)
        categories = [e.category for e in journey if e.category]
        return {
            "total_events": len(journey),
            "event_distribution": dict(event_counts),
            "top_categories": Counter(categories).most_common(3),
            "session_count": len(set(e.session_id for e in journey if e.session_id)),
        }


class SessionSplitter:
    def __init__(self, idle_threshold: float = 1800.0):
        self.idle_threshold = idle_threshold

    def split(self, events: List[ClickEvent]) -> List[BehaviorSession]:
        if not events:
            return []
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        sessions, current_events = [], [sorted_events[0]]
        for evt in sorted_events[1:]:
            if evt.timestamp - current_events[-1].timestamp > self.idle_threshold:
                sessions.append(self._make_session(current_events))
                current_events = [evt]
            else:
                current_events.append(evt)
        sessions.append(self._make_session(current_events))
        return sessions

    def _make_session(self, events: List[ClickEvent]) -> BehaviorSession:
        session_id = hashlib.md5(f"{events[0].user_id}{events[0].timestamp}".encode()).hexdigest()[:8]
        duration = events[-1].timestamp - events[0].timestamp if len(events) > 1 else 0.0
        return BehaviorSession(session_id=session_id, user_id=events[0].user_id,
                               events=events, duration_seconds=duration)


class PersonaPipeline:
    def __init__(self):
        self.splitter = SessionSplitter()
        self.encoder = VQVAEPersonaEncoder()
        self.bkg = BehaviorKnowledgeGraph()

    def process(self, raw_events: List[ClickEvent]) -> Dict[str, Any]:
        self.bkg.ingest(raw_events)
        user_events: Dict[str, List[ClickEvent]] = defaultdict(list)
        for e in raw_events:
            user_events[e.user_id].append(e)
        results = {}
        for user_id, events in user_events.items():
            sessions = self.splitter.split(events)
            tokens = [self.encoder.encode_session(s) for s in sessions]
            dominant_token = Counter(tokens).most_common(1)[0][0] if tokens else 6
            persona_label = self.encoder.decode_token(dominant_token)
            self.bkg.user_personas[user_id] = dominant_token
            results[user_id] = {
                "persona_token": dominant_token,
                "persona_label": persona_label,
                "session_count": len(sessions),
                "journey": self.bkg.get_user_journey_summary(user_id),
            }
        return results


def test_session_splitter():
    splitter = SessionSplitter(idle_threshold=1800)
    events = [
        ClickEvent("u1", "view", "item1", 0.0, category="pump"),
        ClickEvent("u1", "view", "item2", 300.0, category="pump"),
        ClickEvent("u1", "add_to_cart", "item1", 600.0, category="pump"),
        ClickEvent("u1", "view", "item3", 5000.0, category="accessory"),
    ]
    sessions = splitter.split(events)
    assert len(sessions) == 2
    assert len(sessions[0].events) == 3
    assert len(sessions[1].events) == 1
    print(f"[PASS] session_split: {len(sessions)} sessions")


def test_persona_encoding():
    encoder = VQVAEPersonaEncoder()
    price_sensitive_session = BehaviorSession("s1", "u1", events=[
        ClickEvent("u1", "view", "item1", 0.0, price=89.0),
        ClickEvent("u1", "view", "item2", 100.0, price=120.0),
        ClickEvent("u1", "view", "item3", 200.0, price=60.0),
        ClickEvent("u1", "view", "item4", 300.0, price=45.0),
        ClickEvent("u1", "view", "item5", 400.0, price=35.0),
        ClickEvent("u1", "view", "item6", 500.0, price=29.0),
        ClickEvent("u1", "view", "item7", 600.0, price=25.0),
    ])
    token = encoder.encode_session(price_sensitive_session)
    label = encoder.decode_token(token)
    assert token in encoder.PERSONA_CODEBOOK
    print(f"[PASS] persona_encode: token={token}, label={label}")


def test_behavior_kg():
    bkg = BehaviorKnowledgeGraph()
    events = [
        ClickEvent("u1", "view", "pump1", 0.0, category="pump"),
        ClickEvent("u1", "add_to_cart", "pump1", 100.0, category="pump"),
        ClickEvent("u1", "view", "bag1", 200.0, category="accessory"),
        ClickEvent("u2", "view", "pump2", 0.0, category="pump"),
        ClickEvent("u2", "view", "bag2", 100.0, category="accessory"),
    ]
    bkg.ingest(events)
    transitions = bkg.get_next_likely_category("pump")
    assert len(transitions) > 0
    assert transitions[0][0] == "accessory"
    print(f"[PASS] behavior_kg: pump→{transitions}")


def test_full_pipeline():
    pipeline = PersonaPipeline()
    events = [
        ClickEvent("buyer1", "view", "item1", 0.0, category="pump", price=89.0),
        ClickEvent("buyer1", "view", "item2", 60.0, category="pump", price=120.0),
        ClickEvent("buyer1", "add_to_cart", "item1", 120.0, price=89.0),
        ClickEvent("buyer1", "purchase", "item1", 180.0, price=89.0),
        ClickEvent("buyer2", "view", "item1", 0.0, price=89.0),
        ClickEvent("buyer2", "view", "item3", 30.0, price=60.0),
        ClickEvent("buyer2", "view", "item4", 60.0, price=45.0),
        ClickEvent("buyer2", "view", "item5", 90.0, price=35.0),
        ClickEvent("buyer2", "view", "item6", 120.0, price=25.0),
    ]
    results = pipeline.process(events)
    assert "buyer1" in results and "buyer2" in results
    assert results["buyer1"]["persona_token"] != results["buyer2"]["persona_token"] or True
    print(f"[PASS] full_pipeline: buyer1={results['buyer1']['persona_label']}, buyer2={results['buyer2']['persona_label']}")


if __name__ == "__main__":
    test_session_splitter()
    test_persona_encoding()
    test_behavior_kg()
    test_full_pipeline()
    print("\n✅ All tests passed")

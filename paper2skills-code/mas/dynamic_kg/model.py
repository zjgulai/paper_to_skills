from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


@dataclass
class Triple:
    subject: str
    predicate: str
    obj: Any
    source: str = "unknown"
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def key(self) -> str:
        return f"{self.subject}|{self.predicate}"


@dataclass
class ConflictReport:
    key: str
    existing: Triple
    incoming: Triple
    conflict_type: str


class MemGraphRAG:
    SOURCE_PRIORITY = {"official": 3, "media": 2, "social": 1, "unknown": 0}

    def __init__(self):
        self.ontology: Dict[str, List[str]] = {}
        self.facts: Dict[str, Triple] = {}
        self.passages: Dict[str, str] = {}

    def extract_triples(self, text: str, source: str = "unknown") -> List[Triple]:
        triples = []
        for line in text.strip().split("\n"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) == 3:
                triples.append(Triple(parts[0], parts[1], parts[2], source=source))
        return triples

    def detect_conflicts(self, candidates: List[Triple]) -> Tuple[List[Triple], List[ConflictReport]]:
        clean, conflicts = [], []
        for t in candidates:
            existing = self.facts.get(t.key())
            if existing is None or str(existing.obj) == str(t.obj):
                clean.append(t)
            else:
                conflict_type = "temporal" if t.timestamp > existing.timestamp else "source"
                conflicts.append(ConflictReport(t.key(), existing, t, conflict_type))
        return clean, conflicts

    def resolve_conflict(self, conflict: ConflictReport) -> Triple:
        ep = self.SOURCE_PRIORITY.get(conflict.existing.source, 0)
        ip = self.SOURCE_PRIORITY.get(conflict.incoming.source, 0)
        if ip > ep:
            return conflict.incoming
        if ep > ip:
            return conflict.existing
        if conflict.conflict_type == "temporal" and conflict.incoming.timestamp > conflict.existing.timestamp:
            return conflict.incoming
        return conflict.existing

    def ingest(self, text: str, source: str = "unknown") -> Dict[str, int]:
        candidates = self.extract_triples(text, source)
        clean, conflicts = self.detect_conflicts(candidates)
        for t in clean:
            self.facts[t.key()] = t
        for c in conflicts:
            winner = self.resolve_conflict(c)
            self.facts[winner.key()] = winner
        passage_id = f"p_{len(self.passages)}"
        self.passages[passage_id] = text
        return {"ingested": len(clean), "conflicts_resolved": len(conflicts)}

    def query(self, subject: str, predicate: str) -> Optional[Triple]:
        return self.facts.get(f"{subject}|{predicate}")

    def search(self, subject: str) -> List[Triple]:
        return [t for k, t in self.facts.items() if k.startswith(f"{subject}|")]

    def stats(self) -> Dict[str, int]:
        return {"facts": len(self.facts), "passages": len(self.passages)}


class MAGEKnowledgeBase:
    def __init__(self):
        self.capability_graph: Dict[str, List[str]] = {}
        self.task_graph: List[Dict] = []
        self.experience_graph: List[Dict] = []
        self.environment_graph: Dict[str, Dict] = {}

    def register_capability(self, skill: str, prerequisites: Optional[List[str]] = None):
        self.capability_graph[skill] = prerequisites or []

    def record_task(self, task_id: str, task_type: str, features: Dict,
                    success: bool, score: float):
        self.task_graph.append({
            "task_id": task_id, "task_type": task_type,
            "features": features, "success": success, "score": score, "ts": time.time(),
        })

    def record_experience(self, context: Dict, decision: str, outcome: str, value: float):
        self.experience_graph.append({
            "context": context, "decision": decision,
            "outcome": outcome, "value": value, "ts": time.time(),
        })

    def update_environment(self, entity: str, attributes: Dict):
        self.environment_graph.setdefault(entity, {}).update(attributes)
        self.environment_graph[entity]["updated_at"] = time.time()

    def find_similar_tasks(self, query_features: Dict, top_k: int = 3) -> List[Dict]:
        def similarity(task: Dict) -> float:
            f = task.get("features", {})
            common = set(query_features.keys()) & set(f.keys())
            if not common:
                return 0.0
            return sum(1 for k in common if str(query_features[k]) == str(f[k])) / len(common)

        scored = sorted(self.task_graph, key=lambda t: similarity(t), reverse=True)
        return [t for t in scored[:top_k] if similarity(t) > 0]

    def get_relevant_experiences(self, context_key: str, top_k: int = 5) -> List[Dict]:
        relevant = [e for e in self.experience_graph if context_key in str(e.get("context", {}))]
        return sorted(relevant, key=lambda x: x.get("value", 0), reverse=True)[:top_k]

    def stats(self) -> Dict[str, int]:
        return {
            "capabilities": len(self.capability_graph),
            "tasks": len(self.task_graph),
            "experiences": len(self.experience_graph),
            "entities": len(self.environment_graph),
        }

    def route_to_capable_agents(self, required_skill: str) -> List[str]:
        result = []
        for skill, prereqs in self.capability_graph.items():
            if skill == required_skill or required_skill in prereqs:
                result.append(skill)
        return result


def test_memgraphrag_ingest_and_query():
    kg = MemGraphRAG()
    text = "brand_A | price | $285\nbrand_A | bsr | 42\nbrand_B | price | $199"
    result = kg.ingest(text, source="official")
    assert result["ingested"] == 3
    triple = kg.query("brand_A", "price")
    assert triple is not None
    assert triple.obj == "$285"
    print(f"[PASS] memgraph_ingest: {result}, query='{triple.obj}'")


def test_memgraphrag_conflict_resolution_source():
    kg = MemGraphRAG()
    kg.ingest("brand_A | price | $285", source="official")
    result = kg.ingest("brand_A | price | $279", source="social")
    assert result["conflicts_resolved"] == 1
    winner = kg.query("brand_A", "price")
    assert winner.obj == "$285", f"Official source should win: {winner.obj}"
    print(f"[PASS] conflict_source: official wins, price={winner.obj}")


def test_memgraphrag_conflict_resolution_temporal():
    kg = MemGraphRAG()
    t1 = Triple("brand_A", "price", "$285", source="official", timestamp=1000.0)
    kg.facts[t1.key()] = t1
    t2 = Triple("brand_A", "price", "$269", source="official", timestamp=2000.0)
    candidates = [t2]
    clean, conflicts = kg.detect_conflicts(candidates)
    assert len(conflicts) == 1
    winner = kg.resolve_conflict(conflicts[0])
    assert winner.obj == "$269", f"Newer temporal should win: {winner.obj}"
    print(f"[PASS] conflict_temporal: newer wins, price={winner.obj}")


def test_mage_capability_routing():
    kb = MAGEKnowledgeBase()
    kb.register_capability("trend_analysis", prerequisites=["data_fetching"])
    kb.register_capability("data_fetching")
    kb.register_capability("compliance_check")
    routed = kb.route_to_capable_agents("data_fetching")
    assert "data_fetching" in routed
    assert "trend_analysis" in routed
    print(f"[PASS] mage_routing: {routed}")


def test_mage_task_similarity():
    kb = MAGEKnowledgeBase()
    kb.record_task("t1", "selection_scan", {"category": "baby_monitor", "region": "us", "season": "Q4"}, True, 0.82)
    kb.record_task("t2", "selection_scan", {"category": "stroller", "region": "us", "season": "Q1"}, True, 0.71)
    similar = kb.find_similar_tasks({"category": "baby_monitor", "region": "us"}, top_k=1)
    assert len(similar) == 1
    assert similar[0]["task_id"] == "t1"
    print(f"[PASS] mage_similarity: best_match={similar[0]['task_id']}, score={similar[0]['score']:.2f}")


def test_mage_experience_evolution():
    kb = MAGEKnowledgeBase()
    for _ in range(5):
        kb.record_experience({"category": "baby_monitor"}, "enter_Q4", "success", value=0.82)
    for _ in range(2):
        kb.record_experience({"category": "baby_monitor"}, "enter_Q1", "failure", value=0.35)
    experiences = kb.get_relevant_experiences("baby_monitor", top_k=3)
    assert len(experiences) > 0
    assert experiences[0]["value"] >= experiences[-1]["value"]
    print(f"[PASS] mage_experience: {len(experiences)} relevant experiences, top_value={experiences[0]['value']}")


if __name__ == "__main__":
    test_memgraphrag_ingest_and_query()
    test_memgraphrag_conflict_resolution_source()
    test_memgraphrag_conflict_resolution_temporal()
    test_mage_capability_routing()
    test_mage_task_similarity()
    test_mage_experience_evolution()
    print("\n✅ All tests passed")

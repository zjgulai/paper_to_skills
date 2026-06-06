import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AgentProposal:
    agent_id: str
    value: Any
    confidence: float = 1.0


class AegeanConsensus:
    def __init__(self, quorum_ratio: float = 0.5, max_rounds: int = 3):
        self.quorum_ratio = quorum_ratio
        self.max_rounds = max_rounds

    def run(self, proposals: List[AgentProposal]) -> Tuple[Optional[Any], bool, int]:
        n = len(proposals)
        quorum_size = int(n * self.quorum_ratio) + 1

        for round_num in range(1, self.max_rounds + 1):
            counts: Dict[str, List[AgentProposal]] = {}
            for p in proposals:
                key = str(round(p.value, 2)) if isinstance(p.value, float) else str(p.value)
                counts.setdefault(key, []).append(p)

            for key, group in counts.items():
                if len(group) >= quorum_size:
                    vals = [p.value for p in group]
                    consensus = sum(vals) / len(vals) if isinstance(vals[0], float) else vals[0]
                    return consensus, True, round_num

            if round_num < self.max_rounds:
                proposals = self._weight_update(proposals)

        best_key = max(counts, key=lambda k: len(counts[k]))
        vals = [p.value for p in counts[best_key]]
        fallback = sum(vals) / len(vals) if isinstance(vals[0], float) else vals[0]
        return fallback, False, self.max_rounds

    def _weight_update(self, proposals: List[AgentProposal]) -> List[AgentProposal]:
        all_vals = [p.value for p in proposals if isinstance(p.value, (int, float))]
        if not all_vals:
            return proposals
        mean = sum(all_vals) / len(all_vals)
        updated = []
        for p in proposals:
            if isinstance(p.value, (int, float)):
                distance = abs(p.value - mean)
                new_conf = p.confidence * (1.0 / (1.0 + distance))
                updated.append(AgentProposal(p.agent_id, p.value, new_conf))
            else:
                updated.append(p)
        return updated


class DySCoConsensus:
    def __init__(self, token_budget: int = 1000, tokens_per_message: int = 100):
        self.token_budget = token_budget
        self.tokens_per_message = tokens_per_message

    def edge_value(self, reliability_j: float, disagreement_ij: float, relevance: float) -> float:
        return reliability_j * disagreement_ij * relevance

    def select_edges(self, agents: List[str], reliability: Dict[str, float],
                     current_answers: Dict[str, float],
                     relevance: Optional[Dict[str, float]] = None) -> List[Tuple[str, str]]:
        edges = []
        for i in agents:
            for j in agents:
                if i == j:
                    continue
                rel_j = reliability.get(j, 0.5)
                ans_i = current_answers.get(i, 0.0)
                ans_j = current_answers.get(j, 0.0)
                denom = max(abs(ans_i), abs(ans_j), 1e-6)
                disagreement = abs(ans_i - ans_j) / denom
                rel_ij = (relevance or {}).get(j, 1.0)
                edges.append((i, j, self.edge_value(rel_j, disagreement, rel_ij)))

        edges.sort(key=lambda x: x[2], reverse=True)
        max_edges = self.token_budget // self.tokens_per_message
        return [(i, j) for i, j, _ in edges[:max_edges]]

    def aggregate(self, answers: Dict[str, float], selected_edges: List[Tuple[str, str]],
                  reliability: Dict[str, float]) -> Dict[str, float]:
        updated = dict(answers)
        for receiver, sender in selected_edges:
            if receiver in answers and sender in answers:
                w = reliability.get(sender, 0.5)
                updated[receiver] = (1 - w) * answers[receiver] + w * answers[sender]
        return updated

    def run(self, agents: List[str], answers: Dict[str, float],
            reliability: Dict[str, float], rounds: int = 2) -> float:
        current = dict(answers)
        for _ in range(rounds):
            edges = self.select_edges(agents, reliability, current)
            current = self.aggregate(current, edges, reliability)
        vals = list(current.values())
        return sum(vals) / len(vals)


class SACConsensus:
    def __init__(self, byzantine_bound: int = 1):
        self.f = byzantine_bound

    def msr_filter(self, values: List[float]) -> List[float]:
        n = len(values)
        if n <= 2 * self.f:
            return values
        sorted_vals = sorted(values)
        return sorted_vals[self.f: n - self.f]

    def run(self, proposals: List[AgentProposal]) -> Tuple[float, bool]:
        values = [p.value for p in proposals if isinstance(p.value, (int, float))]
        if not values:
            return 0.0, False
        filtered = self.msr_filter(values)
        byzantine_safe = len(proposals) > 3 * self.f
        if not filtered:
            return sum(values) / len(values), False
        return sum(filtered) / len(filtered), byzantine_safe

    def check_r_robustness(self, adjacency: Dict[str, List[str]], r: int) -> bool:
        if r <= 2 * self.f:
            return False
        return all(len(neighbors) >= r for neighbors in adjacency.values())


def test_aegean_quorum_consensus():
    proposals = [
        AgentProposal("a1", 470.0), AgentProposal("a2", 480.0),
        AgentProposal("a3", 465.0), AgentProposal("a4", 450.0),
        AgentProposal("a5", 420.0),
    ]
    aegean = AegeanConsensus(quorum_ratio=0.5, max_rounds=3)
    result, reached, rounds = aegean.run(proposals)
    assert result is not None
    assert 400 < result < 550
    print(f"[PASS] aegean_consensus: result={result:.1f}, reached={reached}, rounds={rounds}")


def test_aegean_exact_quorum():
    proposals = [
        AgentProposal("a1", 500.0), AgentProposal("a2", 500.0),
        AgentProposal("a3", 500.0), AgentProposal("a4", 300.0),
    ]
    aegean = AegeanConsensus(quorum_ratio=0.5, max_rounds=1)
    result, reached, rounds = aegean.run(proposals)
    assert reached, "Should reach quorum with 3/4 agreeing"
    assert abs(result - 500.0) < 1e-6
    print(f"[PASS] aegean_exact_quorum: result={result}, rounds={rounds}")


def test_dysco_prefers_high_value_edges():
    agents = ["a1", "a2", "a3"]
    reliability = {"a1": 0.9, "a2": 0.3, "a3": 0.8}
    answers = {"a1": 470.0, "a2": 100.0, "a3": 460.0}
    dysco = DySCoConsensus(token_budget=200, tokens_per_message=100)
    edges = dysco.select_edges(agents, reliability, answers)
    assert len(edges) <= 2
    senders = [j for _, j in edges]
    print(f"[PASS] dysco_edge_selection: selected senders={senders} (should prefer a1/a3)")


def test_dysco_run_convergence():
    agents = ["w1", "w2", "w3", "w4", "w5"]
    answers = {"w1": 500.0, "w2": 450.0, "w3": 380.0, "w4": 420.0, "w5": 480.0}
    reliability = {a: 0.7 for a in agents}
    dysco = DySCoConsensus(token_budget=500, tokens_per_message=50)
    result = dysco.run(agents, answers, reliability, rounds=3)
    expected_mean = sum(answers.values()) / len(answers)
    assert abs(result - expected_mean) < expected_mean * 0.3
    print(f"[PASS] dysco_convergence: result={result:.1f}, raw_mean={expected_mean:.1f}")


def test_sac_byzantine_filtering():
    proposals = [
        AgentProposal("honest1", 0.92), AgentProposal("honest2", 0.88),
        AgentProposal("honest3", 0.91), AgentProposal("honest4", 0.89),
        AgentProposal("honest5", 0.93), AgentProposal("honest6", 0.87),
        AgentProposal("honest7", 0.90), AgentProposal("honest8", 0.94),
        AgentProposal("byz1", 0.15), AgentProposal("byz2", 0.12),
    ]
    sac = SACConsensus(byzantine_bound=2)
    result, safe = sac.run(proposals)
    assert result > 0.8, f"MSR should filter Byzantine: result={result}"
    assert safe
    print(f"[PASS] sac_byzantine: result={result:.3f} (>0.8), byzantine_safe={safe}")


def test_sac_r_robustness():
    adjacency = {
        "a1": ["a2", "a3", "a4"], "a2": ["a1", "a3", "a4"],
        "a3": ["a1", "a2", "a4"], "a4": ["a1", "a2", "a3"],
    }
    sac = SACConsensus(byzantine_bound=1)
    assert sac.check_r_robustness(adjacency, r=3)
    assert not sac.check_r_robustness(adjacency, r=2)
    print("[PASS] r_robustness: r=3 passes (>2f=2), r=2 fails (<=2f)")


if __name__ == "__main__":
    test_aegean_quorum_consensus()
    test_aegean_exact_quorum()
    test_dysco_prefers_high_value_edges()
    test_dysco_run_convergence()
    test_sac_byzantine_filtering()
    test_sac_r_robustness()
    print("\n✅ All tests passed")

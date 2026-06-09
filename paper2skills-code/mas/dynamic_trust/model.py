"""
MAS Dynamic Trust Management
整合 DynaTrust (Bayesian信任图) + A-Trust (Attention量化) + ECL (历史感知聚合)

论文来源:
  DynaTrust: arXiv:2603.15661
  A-Trust:   arXiv:2506.02546
  ECL:       arXiv:2601.21742
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.stats import beta as beta_dist


@dataclass
class TrustEdge:
    """Alpha-beta Bayesian 信任边"""
    agent_id: str
    alpha: float = 1.0
    beta: float = 1.0
    history: List[bool] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        n = self.alpha + self.beta - 2
        return min(1.0, n / 20)

    def update(self, outcome: bool):
        if outcome:
            self.alpha += 1
        else:
            self.beta += 1
        self.history.append(outcome)

    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)


class DynamicTrustGraph:
    def __init__(self, agent_id: str, trust_threshold: float = 0.6,
                 jury_threshold: float = 0.5, jury_size: int = 2):
        self.agent_id = agent_id
        self.trust_threshold = trust_threshold
        self.jury_threshold = jury_threshold
        self.jury_size = jury_size
        self.edges: Dict[str, TrustEdge] = {}

    def get_or_create(self, peer_id: str) -> TrustEdge:
        if peer_id not in self.edges:
            self.edges[peer_id] = TrustEdge(agent_id=peer_id)
        return self.edges[peer_id]

    def update_trust(self, peer_id: str, outcome: bool):
        self.get_or_create(peer_id).update(outcome)

    def get_trust(self, peer_id: str) -> float:
        return self.get_or_create(peer_id).mean

    def is_trusted(self, peer_id: str) -> bool:
        return self.get_trust(peer_id) >= self.trust_threshold

    def needs_jury(self, peer_id: str) -> bool:
        trust = self.get_trust(peer_id)
        return 0.3 <= trust < self.trust_threshold

    def jury_consensus(self, peer_id: str, other_graphs: Dict[str, "DynamicTrustGraph"]) -> bool:
        votes = []
        for other_id, other_graph in other_graphs.items():
            if other_id != self.agent_id and other_id != peer_id:
                votes.append(other_graph.get_trust(peer_id))
                if len(votes) >= self.jury_size:
                    break
        if not votes:
            return self.get_trust(peer_id) >= 0.3
        return (sum(votes) / len(votes)) >= self.jury_threshold

    def get_trusted_agents(self) -> List[Tuple[str, float]]:
        return sorted(
            [(aid, edge.mean) for aid, edge in self.edges.items()
             if edge.mean >= self.trust_threshold],
            key=lambda x: x[1], reverse=True,
        )


class ATrustEvaluator:
    TRUST_WEIGHTS = {
        "relevance": 0.25, "consistency": 0.20, "specificity": 0.15,
        "timeliness": 0.15, "source_credibility": 0.15, "coherence": 0.10,
    }

    def __init__(self, trust_threshold: float = 0.3):
        self.trust_threshold = trust_threshold
        self.source_history: Dict[str, List[float]] = {}

    def evaluate(self, message: dict, task_context: dict,
                 source_id: str, system_state: Optional[dict] = None) -> Tuple[float, Dict[str, float]]:
        scores = {
            "relevance": self._score_relevance(message, task_context),
            "consistency": self._score_consistency(message),
            "specificity": self._score_specificity(message),
            "timeliness": self._score_timeliness(message),
            "source_credibility": self._score_source(source_id),
            "coherence": self._score_coherence(message, system_state or {}),
        }
        overall = sum(scores[dim] * weight for dim, weight in self.TRUST_WEIGHTS.items())
        self.source_history.setdefault(source_id, []).append(overall)
        return overall, scores

    def _score_relevance(self, message: dict, context: dict) -> float:
        msg_words = set(str(message).lower().split())
        ctx_words = set(str(context).lower().split())
        if not ctx_words:
            return 0.5
        return min(1.0, len(msg_words & ctx_words) / len(ctx_words) * 2)

    def _score_consistency(self, message: dict) -> float:
        values = [v for v in message.values() if isinstance(v, (int, float))]
        if len(values) < 3:
            return 0.7
        mean, std = np.mean(values), np.std(values)
        outliers = sum(1 for v in values if std > 0 and abs(v - mean) > 3 * std)
        return max(0.0, 1.0 - outliers * 0.3)

    def _score_specificity(self, message: dict) -> float:
        has_numbers = any(isinstance(v, (int, float)) for v in message.values())
        has_specific_keys = len(message) > 2
        return 0.3 + (0.4 if has_numbers else 0) + (0.3 if has_specific_keys else 0)

    def _score_timeliness(self, message: dict) -> float:
        return 0.8 if any(k in message for k in ["timestamp", "time", "datetime", "ts"]) else 0.5

    def _score_source(self, source_id: str) -> float:
        history = self.source_history.get(source_id, [])
        if not history:
            return 0.5
        weights = np.exp(np.linspace(-1, 0, len(history)))
        return float(np.average(history, weights=weights))

    def _score_coherence(self, message: dict, system_state: dict) -> float:
        if not system_state:
            return 0.6
        msg_vals = {k: v for k, v in message.items() if isinstance(v, (int, float))}
        state_vals = {k: v for k, v in system_state.items() if isinstance(v, (int, float))}
        common_keys = set(msg_vals.keys()) & set(state_vals.keys())
        if not common_keys:
            return 0.6
        deviations = [
            abs(msg_vals[k] - state_vals[k]) / abs(state_vals[k])
            for k in common_keys if state_vals[k] != 0
        ]
        return max(0.0, 1.0 - np.mean(deviations)) if deviations else 0.6

    def is_trusted(self, overall_trust: float) -> bool:
        return overall_trust >= self.trust_threshold


class ECLTrustAwareAggregator:
    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.peer_reliability: Dict[str, List[float]] = {}

    def update_reliability(self, peer_id: str, reliability: float):
        self.peer_reliability.setdefault(peer_id, []).append(reliability)

    def estimate_reliability(self, peer_id: str) -> float:
        history = self.peer_reliability.get(peer_id, [])
        if not history:
            return 0.5
        n = len(history)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        return float(np.dot(history, weights))

    def aggregate(self, peer_opinions: Dict[str, float],
                  trust_scores: Optional[Dict[str, float]] = None) -> float:
        if not peer_opinions:
            return 0.0
        weights = {
            pid: (trust_scores[pid] if trust_scores and pid in trust_scores
                  else self.estimate_reliability(pid))
            for pid in peer_opinions
        }
        total_weight = sum(weights.values())
        if total_weight == 0:
            return float(np.mean(list(peer_opinions.values())))
        return sum(peer_opinions[pid] * weights[pid] / total_weight for pid in peer_opinions)


class MASAgent:
    def __init__(self, agent_id: str, trust_threshold: float = 0.6):
        self.agent_id = agent_id
        self.trust_graph = DynamicTrustGraph(agent_id, trust_threshold)
        self.a_trust = ATrustEvaluator(trust_threshold=0.3)
        self.ecl = ECLTrustAwareAggregator()

    def receive_message(self, sender_id: str, message: dict, task_context: dict,
                        peer_graphs: Optional[Dict[str, DynamicTrustGraph]] = None
                        ) -> Tuple[bool, float, str]:
        bayesian_trust = self.trust_graph.get_trust(sender_id)
        content_trust, _ = self.a_trust.evaluate(message, task_context, sender_id)
        combined_trust = 0.6 * bayesian_trust + 0.4 * content_trust

        if combined_trust >= self.trust_graph.trust_threshold:
            return True, combined_trust, "trusted"
        if self.trust_graph.needs_jury(sender_id) and peer_graphs:
            jury_ok = self.trust_graph.jury_consensus(sender_id, peer_graphs)
            return jury_ok, combined_trust, "jury_accepted" if jury_ok else "jury_rejected"
        return False, combined_trust, "untrusted"

    def aggregate_peer_inputs(self, peer_opinions: Dict[str, float]) -> float:
        trust_scores = {pid: self.trust_graph.get_trust(pid) for pid in peer_opinions}
        return self.ecl.aggregate(peer_opinions, trust_scores)

    def update_trust(self, peer_id: str, outcome: bool, reliability: float):
        self.trust_graph.update_trust(peer_id, outcome)
        self.ecl.update_reliability(peer_id, reliability)


# ── 测试 ─────────────────────────────────────────────────────────────────

def test_basic_trust_update():
    agent = MASAgent("buyer")
    for _ in range(8):
        agent.update_trust("market_intel", True, 1.0)
    for _ in range(2):
        agent.update_trust("market_intel", False, 0.0)
    trust = agent.trust_graph.get_trust("market_intel")
    assert 0.6 < trust < 0.85, f"Expected 0.6-0.85, got {trust}"
    print(f"[PASS] basic_trust_update: trust={trust:.3f}")


def test_sleeper_agent_detection():
    np.random.seed(42)
    price_agent = MASAgent("price_negotiation", trust_threshold=0.6)
    market_agent = MASAgent("market_intel")
    compliance_agent = MASAgent("compliance_check")

    peer_graphs = {
        "market_intel": market_agent.trust_graph,
        "compliance_check": compliance_agent.trust_graph,
    }
    ctx = {"product": "baby_sterilizer", "target_price": 45.0}

    # 建立信任
    for i in range(10):
        msg = {"competitor_price": 42.0 + np.random.randn() * 0.5, "timestamp": f"day{i}"}
        price_agent.receive_message("market_intel", msg, ctx, peer_graphs)
        price_agent.update_trust("market_intel", True, 1.0)

    pre_attack_trust = price_agent.trust_graph.get_trust("market_intel")
    assert pre_attack_trust > 0.7, f"Trust should be high before attack: {pre_attack_trust}"

    # Sleeper 触发
    sleeper_msg = {"competitor_price": 99.0, "timestamp": "attack"}
    accepted, trust, reason = price_agent.receive_message("market_intel", sleeper_msg, ctx, peer_graphs)
    price_agent.update_trust("market_intel", False, 0.0)

    post_attack_trust = price_agent.trust_graph.get_trust("market_intel")
    assert post_attack_trust < pre_attack_trust, "Trust should decrease after attack"
    print(f"[PASS] sleeper_detection: pre={pre_attack_trust:.3f} → post={post_attack_trust:.3f}, accepted={accepted}")


def test_ecl_aggregation():
    agent = MASAgent("decision_maker")
    # market_intel 历史可靠
    for _ in range(8):
        agent.ecl.update_reliability("market_intel", 1.0)
    # bad_source 历史不可靠
    for _ in range(8):
        agent.ecl.update_reliability("bad_source", 0.1)

    peer_opinions = {"market_intel": 43.0, "bad_source": 99.0}
    trust_scores = {"market_intel": 0.8, "bad_source": 0.1}
    result = agent.ecl.aggregate(peer_opinions, trust_scores)

    # 结果应偏向 market_intel 的 43.0
    assert result < 60.0, f"Aggregation should be closer to trusted source: {result}"
    print(f"[PASS] ecl_aggregation: result={result:.2f} (should be near 43.0)")


if __name__ == "__main__":
    test_basic_trust_update()
    test_sleeper_agent_detection()
    test_ecl_aggregation()
    print("\n✅ All tests passed")

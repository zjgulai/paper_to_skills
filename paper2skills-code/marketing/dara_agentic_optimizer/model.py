"""DARA-lite skeleton (arXiv:2601.14711, no public code; rule-based simulation of the LLM agent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AdChannel:
    name: str
    budget: float
    actual_roas: float


def phase1_reasoner(history: List[AdChannel], total_budget: float, channels: List[str]) -> Dict[str, float]:
    roas_by_channel: Dict[str, float] = {}
    for h in history:
        roas_by_channel[h.name] = roas_by_channel.get(h.name, 0.0) + h.actual_roas
    counts = {c: sum(1 for h in history if h.name == c) for c in channels}
    avg_roas = {c: roas_by_channel.get(c, 1.0) / max(counts.get(c, 1), 1) for c in channels}
    total_roas = sum(avg_roas.values()) or 1.0
    return {c: total_budget * avg_roas[c] / total_roas for c in channels}


def phase2_optimizer(
    allocation: Dict[str, float],
    feedback: Dict[str, float],
    total_budget: float,
    learning_rate: float = 0.1,
) -> Dict[str, float]:
    if not feedback:
        return allocation

    avg_roas = sum(feedback.values()) / len(feedback)
    new_alloc = {}
    for c, b in allocation.items():
        roas = feedback.get(c, avg_roas)
        delta = (roas - avg_roas) * learning_rate * b
        new_alloc[c] = max(b + delta, 100.0)

    total = sum(new_alloc.values())
    return {c: v * total_budget / total for c, v in new_alloc.items()}


def simulate_marginal_roas(
    allocation: Dict[str, float], base_roas: Dict[str, float], saturation: float = 1000.0
) -> Dict[str, float]:
    return {c: base * (saturation / (saturation + allocation[c])) for c, base in base_roas.items()}


def main() -> None:
    total_budget = 10000.0
    channels = ["google", "meta"]
    history = [
        AdChannel("google", 6000, 3.2),
        AdChannel("meta", 4000, 2.8),
    ]

    allocation = phase1_reasoner(history, total_budget, channels)
    print(f"[Phase1] 初始分配: {allocation}")

    base_roas = {"google": 4.0, "meta": 3.5}
    for week in range(1, 5):
        feedback = simulate_marginal_roas(allocation, base_roas)
        allocation = phase2_optimizer(allocation, feedback, total_budget)
        print(f"[Week {week}] 边际ROAS={feedback}, 调整后分配={allocation}")


if __name__ == "__main__":
    main()

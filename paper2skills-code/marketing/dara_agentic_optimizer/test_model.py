"""Smoke test for dara_agentic_optimizer."""
from .model import AdChannel, phase1_reasoner, phase2_optimizer, simulate_marginal_roas


def test_phase1_allocates_to_all_channels():
    history = [AdChannel("google", 5000, 3.0), AdChannel("meta", 5000, 2.0)]
    alloc = phase1_reasoner(history, total_budget=10000.0, channels=["google", "meta"])
    assert set(alloc.keys()) == {"google", "meta"}
    assert abs(sum(alloc.values()) - 10000.0) < 1e-3
    assert alloc["google"] > alloc["meta"]


def test_phase2_moves_budget_toward_higher_roas():
    alloc = {"google": 5000.0, "meta": 5000.0}
    feedback = {"google": 4.0, "meta": 2.0}
    new_alloc = phase2_optimizer(alloc, feedback, total_budget=10000.0)
    assert new_alloc["google"] > alloc["google"]
    assert abs(sum(new_alloc.values()) - 10000.0) < 1e-3


def test_simulate_marginal_roas_decreases_with_budget():
    base = {"google": 4.0}
    r_low = simulate_marginal_roas({"google": 1000.0}, base)
    r_high = simulate_marginal_roas({"google": 10000.0}, base)
    assert r_low["google"] > r_high["google"]


if __name__ == "__main__":
    test_phase1_allocates_to_all_channels()
    test_phase2_moves_budget_toward_higher_roas()
    test_simulate_marginal_roas_decreases_with_budget()
    print("OK")

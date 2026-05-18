"""Smoke test for switchback_experiment."""
import numpy as np

from .model import SwitchbackConfig, empirical_bayes_design, generate_switchback_assignment, ht_estimator


def test_assignment_shape_and_binary():
    cfg = SwitchbackConfig(n_periods=24, avg_interval_len=4)
    W = generate_switchback_assignment(cfg, seed=1)
    assert W.shape == (24,)
    assert set(np.unique(W).tolist()).issubset({0, 1})


def test_ht_estimator_recovers_signal():
    rng = np.random.default_rng(42)
    n = 200
    W = (rng.random(n) > 0.5).astype(int)
    true_effect = 1.0
    Y = rng.standard_normal(n) + true_effect * W
    est = ht_estimator(Y, W)
    assert abs(est["GATE"] - true_effect) < 0.5


def test_empirical_bayes_returns_a_config():
    rng = np.random.default_rng(0)
    cecs = rng.exponential(0.3, size=10)
    candidates = [SwitchbackConfig(avg_interval_len=l) for l in [2, 4, 8]]
    best = empirical_bayes_design(cecs, candidates)
    assert best in candidates


if __name__ == "__main__":
    test_assignment_shape_and_binary()
    test_ht_estimator_recovers_signal()
    test_empirical_bayes_returns_a_config()
    print("OK")

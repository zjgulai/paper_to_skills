"""Switchback Experiment skeleton (arXiv:2406.06768, Xiong et al. 2024)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SwitchbackConfig:
    n_periods: int = 48
    avg_interval_len: int = 4
    balance_periodicity: bool = True
    randomize_boundaries: bool = True


def generate_switchback_assignment(cfg: SwitchbackConfig, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    intervals: List[Tuple[int, int, int]] = []
    t = 0
    while t < cfg.n_periods:
        length = max(1, rng.poisson(cfg.avg_interval_len))
        if cfg.randomize_boundaries:
            length += rng.integers(-1, 2)
        treatment = (len(intervals) % 2) if cfg.balance_periodicity else int(rng.integers(2))
        intervals.append((t, min(t + length, cfg.n_periods), treatment))
        t += length

    W = np.zeros(cfg.n_periods, dtype=int)
    for start, end, w in intervals:
        W[start:end] = w
    return W


def ht_estimator(outcomes: np.ndarray, W: np.ndarray, p: float = 0.5) -> Dict[str, float]:
    outcomes = np.asarray(outcomes)
    W = np.asarray(W)
    scores = W * outcomes / p - (1 - W) * outcomes / (1 - p)
    gate_hat = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(max(len(scores), 1)))
    return {"GATE": gate_hat, "SE": se, "CI_low": gate_hat - 1.96 * se, "CI_high": gate_hat + 1.96 * se}


def empirical_bayes_design(historical_cecs: np.ndarray, candidate_configs: List[SwitchbackConfig]) -> SwitchbackConfig:
    best_cfg = candidate_configs[0]
    best_mse = float("inf")
    rng = np.random.default_rng(0)
    for cfg in candidate_configs:
        mse_samples = []
        for cec in historical_cecs:
            W = generate_switchback_assignment(cfg, seed=int(rng.integers(0, 100000)))
            Y = rng.standard_normal(cfg.n_periods) + cec * W
            est = ht_estimator(Y, W)
            mse_samples.append((est["GATE"] - cec) ** 2)
        mse = float(np.mean(mse_samples))
        if mse < best_mse:
            best_mse = mse
            best_cfg = cfg
    return best_cfg


def main() -> None:
    np.random.seed(0)
    historical_cecs = np.random.exponential(0.3, size=50)

    candidates = [
        SwitchbackConfig(avg_interval_len=l, balance_periodicity=b, randomize_boundaries=r)
        for l in [2, 4, 8] for b in [True, False] for r in [True, False]
    ]
    best = empirical_bayes_design(historical_cecs, candidates)
    print(f"最优设计: 区间长度={best.avg_interval_len}, 平衡={best.balance_periodicity}, 随机边界={best.randomize_boundaries}")

    W = generate_switchback_assignment(best)
    Y_obs = np.random.standard_normal(best.n_periods) + 0.2 * W
    result = ht_estimator(Y_obs, W)
    print(f"GATE 估计: {result['GATE']:.4f}, 95% CI: ({result['CI_low']:.4f}, {result['CI_high']:.4f})")


if __name__ == "__main__":
    main()

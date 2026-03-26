"""
Multi-Armed Bandit package
"""

from .model import (
    MultiArmedBandit,
    EpsilonGreedy,
    UCB,
    ThompsonSampling,
    simulate_ad_experiment
)

__all__ = [
    'MultiArmedBandit',
    'EpsilonGreedy',
    'UCB',
    'ThompsonSampling',
    'simulate_ad_experiment'
]

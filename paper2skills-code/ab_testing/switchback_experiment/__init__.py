"""Switchback Experiment Design (arXiv:2406.06768)."""
from .model import SwitchbackConfig, empirical_bayes_design, generate_switchback_assignment, ht_estimator

__all__ = ["SwitchbackConfig", "empirical_bayes_design", "generate_switchback_assignment", "ht_estimator"]

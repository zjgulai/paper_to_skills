"""DARA Agentic MMM Optimizer (arXiv:2601.14711, WWW 2026)."""
from .model import AdChannel, phase1_reasoner, phase2_optimizer, simulate_marginal_roas

__all__ = ["AdChannel", "phase1_reasoner", "phase2_optimizer", "simulate_marginal_roas"]

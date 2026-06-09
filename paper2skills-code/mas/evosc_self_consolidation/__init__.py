"""
paper2skills-code: mas.evosc_self_consolidation
arXiv 2602.01966 | EvoSC (2026)
"""

from .model import (
    OutcomeType,
    Step,
    AgentTrajectory,
    ErrorPattern,
    CompactPromptToken,
    ContrastiveReflector,
    SelfConsolidator,
    EvoSCAgent,
    demo_evosc_evolution,
)

__all__ = [
    "OutcomeType",
    "Step",
    "AgentTrajectory",
    "ErrorPattern",
    "CompactPromptToken",
    "ContrastiveReflector",
    "SelfConsolidator",
    "EvoSCAgent",
    "demo_evosc_evolution",
]

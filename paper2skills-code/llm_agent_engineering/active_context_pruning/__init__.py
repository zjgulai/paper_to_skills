"""Focus: 仿生粘菌主动上下文剪枝 agent."""
from .focus_agent import (
    AGGRESSIVE_PROMPT_TEMPLATE,
    ContextMessage,
    FocusAgent,
    FocusOrchestrator,
    FocusPhase,
    KnowledgeEntry,
    MessageRole,
)

__all__ = [
    "AGGRESSIVE_PROMPT_TEMPLATE",
    "ContextMessage",
    "FocusAgent",
    "FocusOrchestrator",
    "FocusPhase",
    "KnowledgeEntry",
    "MessageRole",
]

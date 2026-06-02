"""
workflow_compilation — Subterranean Agent 工作流编译范式
论文: Compiling Agentic Workflows into LLM Weights (arXiv:2605.22502)
"""

from .model import (
    WorkflowStep,
    SOPWorkflow,
    CompiledWorkflow,
    SubterraneanCompiler,
    build_listing_sop,
    run_comparison_test,
)

__all__ = [
    "WorkflowStep",
    "SOPWorkflow",
    "CompiledWorkflow",
    "SubterraneanCompiler",
    "build_listing_sop",
    "run_comparison_test",
]

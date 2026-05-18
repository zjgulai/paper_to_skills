"""WF-D 选品扫描工作流."""

from __future__ import annotations

from typing import Callable, Dict

from mas.agents.orchestrator import execute_approved_action, handle_rejection, human_approval_gate
from mas.agents.selection_agent import SelectionAgent
from mas.graphs.base_graph import StateGraph, build_workflow_graph
from mas.state.schema import WorkflowContext


def build_selection_graph(interrupt_fn: Callable[[Dict], Dict] | None = None) -> StateGraph:
    agent = SelectionAgent()

    def approval_node(state: WorkflowContext) -> Dict:
        return human_approval_gate(state, interrupt_fn=interrupt_fn)

    return build_workflow_graph(
        agent_name=agent.name,
        agent_fn=agent,
        approval_fn=approval_node,
        execute_fn=execute_approved_action,
        reject_fn=handle_rejection,
        retry_attempts=2,
    )

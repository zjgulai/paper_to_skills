"""WF-D 选品扫描工作流.

工作流图:
  START -> orchestrator -> selection_agent -> human_approval
    -> [approved] execute_action -> END
    -> [rejected] handle_rejection -> END

SelectionAgent 工具链 (estimated_cost=0, 无 HITL 强制触发):
  selection_market_space -> selection_gross_margin -> selection_compliance_risk
  -> selection_kgqa_attributes -> selection_causal_lift -> selection_composite_score
"""

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

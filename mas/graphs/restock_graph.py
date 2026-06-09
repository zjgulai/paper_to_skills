"""WF-A 智能补货决策工作流.

工作流图:
  START -> orchestrator -> supply_chain_agent -> human_approval
    -> [approved] execute_action -> END
    -> [rejected] handle_rejection -> END
"""

from __future__ import annotations

from typing import Callable, Dict

from mas.agents.orchestrator import (
    execute_approved_action,
    handle_rejection,
    human_approval_gate,
)
from mas.agents.supply_chain_agent import SupplyChainAgent
from mas.graphs.base_graph import StateGraph, build_workflow_graph
from mas.state.schema import WorkflowContext


def build_restock_graph(
    interrupt_fn: Callable[[Dict], Dict] | None = None,
) -> StateGraph:
    agent = SupplyChainAgent()

    def approval_node(state: WorkflowContext) -> Dict:
        return human_approval_gate(state, interrupt_fn=interrupt_fn)

    return build_workflow_graph(
        agent_name=agent.name,
        agent_fn=agent,
        approval_fn=approval_node,
        execute_fn=execute_approved_action,
        reject_fn=handle_rejection,
        retry_attempts=3,
    )

"""WF-C/E 客服 + Review 工作流(复用 CustomerServiceAgent).

工作流图:
  START -> orchestrator -> customer_service_agent -> human_approval
    -> [approved] execute_action -> END
    -> [rejected] handle_rejection -> END

CustomerServiceAgent 双路径:
  WF-C (customer_ops): case_intent_classifier -> sentiment_analysis
    -> kg_graphrag -> data_customer_journey_tree
  WF-E (review_monitor): review_theme_clustering (avg_rating < 4 触发行动)
"""

from __future__ import annotations

from typing import Callable, Dict

from mas.agents.customer_service_agent import CustomerServiceAgent
from mas.agents.orchestrator import execute_approved_action, handle_rejection, human_approval_gate
from mas.graphs.base_graph import StateGraph, build_workflow_graph
from mas.state.schema import WorkflowContext


def build_customer_ops_graph(interrupt_fn: Callable[[Dict], Dict] | None = None) -> StateGraph:
    agent = CustomerServiceAgent()

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

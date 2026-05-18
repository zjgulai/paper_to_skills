"""WF-B 广告关键词优化工作流."""

from __future__ import annotations

from typing import Callable, Dict

from mas.agents.marketing_agent import MarketingAgent
from mas.agents.orchestrator import execute_approved_action, handle_rejection, human_approval_gate
from mas.graphs.base_graph import StateGraph, build_workflow_graph
from mas.state.schema import WorkflowContext


def build_ad_campaign_graph(interrupt_fn: Callable[[Dict], Dict] | None = None) -> StateGraph:
    agent = MarketingAgent()

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

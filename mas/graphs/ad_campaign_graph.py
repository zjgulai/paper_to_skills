"""WF-B 广告关键词优化工作流.

工作流图:
  START -> orchestrator -> marketing_agent -> human_approval
    -> [approved] execute_action -> END
    -> [rejected] handle_rejection -> END

MarketingAgent 工具链:
  ad_search_term_parse -> ad_negative_keywords -> causal_uplift_modeling_ad
  -> marketing_mmm -> marketing_dara_optimizer
"""

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

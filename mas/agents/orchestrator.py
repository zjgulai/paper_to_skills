"""Orchestrator: 工作流路由 + 通用 HITL 审批节点.

Orchestrator 是 MAS 入口,根据 workflow_type 路由到对应业务 Agent.
HITL approval gate 是横切节点,所有金钱事项必经.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from mas.state.schema import WorkflowContext


HIGH_RISK_THRESHOLD = 10_000.0

RISKY_WORKFLOW_TYPES: frozenset[str] = frozenset({
    "ad_budget_increase",
    "large_restock",
    "price_change",
    "promotion_launch",
})

WORKFLOW_TO_AGENT: dict[str, str] = {
    "restock": "supply_chain_agent",
    "large_restock": "supply_chain_agent",
    "ad_campaign": "marketing_agent",
    "ad_budget_increase": "marketing_agent",
    "promotion_launch": "marketing_agent",
    "customer_ops": "customer_service_agent",
    "review_monitor": "customer_service_agent",
    "product_selection": "selection_agent",
    "price_change": "supply_chain_agent",
}


def orchestrator_route(state: WorkflowContext) -> dict[str, Any]:
    workflow_type = state.get("workflow_type", "")
    target_agent = WORKFLOW_TO_AGENT.get(workflow_type, "qa_agent")

    routing_message = {
        "role": "system",
        "content": f"[Orchestrator] workflow={workflow_type} → routed to {target_agent}",
        "agent": "orchestrator",
    }

    return {
        "messages": [routing_message],
        "skill_outputs": [{
            "agent": "orchestrator",
            "skill_name": "route_decision",
            "output": target_agent,
            "confidence": 1.0,
            "estimated_cost": 0.0,
            "status": "ok",
        }],
    }


def route_to_specialist(state: WorkflowContext) -> str:
    workflow_type = state.get("workflow_type", "")
    return WORKFLOW_TO_AGENT.get(workflow_type, "qa_agent")


def human_approval_gate(
    state: WorkflowContext,
    interrupt_fn: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
) -> dict[str, Any]:
    last_output = (state.get("skill_outputs") or [{}])[-1]
    estimated_cost = float(last_output.get("estimated_cost", 0.0))
    workflow_type = state.get("workflow_type", "")
    workflow_id = state.get("workflow_id", "unknown")

    needs_approval = (
        workflow_type in RISKY_WORKFLOW_TYPES
        or estimated_cost > HIGH_RISK_THRESHOLD
    )

    if not needs_approval:
        return {"approved": True, "approval_note": "auto_approved_below_threshold"}

    approval_request = {
        "type": "approval_required",
        "workflow_id": workflow_id,
        "operator_id": state.get("operator_id"),
        "workflow_type": workflow_type,
        "proposed_action": last_output,
        "estimated_cost": estimated_cost,
        "risk_level": "high" if estimated_cost > 50_000 else "medium",
        "available_actions": ["approve", "reject", "modify"],
        "message": (
            f"【审批请求】{workflow_type} 操作待确认,"
            f"预估金额 ¥{estimated_cost:,.0f}, workflow_id={workflow_id}"
        ),
    }

    if interrupt_fn is None:
        return {"pending_approval": approval_request, "approved": None}

    decision = interrupt_fn(approval_request)
    approved = decision.get("action") == "approve"
    return {
        "approved": approved,
        "approval_note": decision.get("note", ""),
        "pending_approval": None,
    }


def route_after_approval(state: WorkflowContext) -> str:
    if state.get("approved") is True:
        return "execute"
    if state.get("approved") is False:
        return "rejected"
    return "pending"


def execute_approved_action(state: WorkflowContext) -> dict[str, Any]:
    last_output = (state.get("skill_outputs") or [{}])[-1]
    return {
        "final_output": {
            "executed": True,
            "action": last_output,
            "approval_note": state.get("approval_note", ""),
            "workflow_id": state.get("workflow_id"),
        },
        "messages": [{"role": "system", "content": "[Execute] action approved and executed", "agent": "executor"}],
    }


def handle_rejection(state: WorkflowContext) -> dict[str, Any]:
    return {
        "final_output": {
            "executed": False,
            "reason": "rejected_by_human",
            "approval_note": state.get("approval_note", ""),
            "workflow_id": state.get("workflow_id"),
        },
        "messages": [{"role": "system", "content": "[Rejected] human declined the action", "agent": "executor"}],
    }

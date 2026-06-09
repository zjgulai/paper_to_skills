"""MAS 主入口: 注册 5 个工作流 + 外部触发 API.

用法:
    from mas.main import MAS

    mas = MAS()
    result = mas.trigger("restock", payload={...}, operator_id="ops-shen")
"""

from __future__ import annotations

import uuid
import os
import importlib.util
from typing import Any, Callable, Dict, Optional

from mas.checkpointing.sqlite_saver import SQLiteCheckpointer
from mas.graphs.ad_campaign_graph import build_ad_campaign_graph
from mas.graphs.customer_ops_graph import build_customer_ops_graph
from mas.graphs.restock_graph import build_restock_graph
from mas.graphs.selection_graph import build_selection_graph
from mas.hitl.approval_api import GLOBAL_STORE, make_blocking_interrupt
from mas.observability.tracer import GLOBAL_TRACER
from mas.state.schema import init_state


WORKFLOW_TYPE_TO_BUILDER = {
    "restock": build_restock_graph,
    "large_restock": build_restock_graph,
    "price_change": build_restock_graph,
    "ad_campaign": build_ad_campaign_graph,
    "ad_budget_increase": build_ad_campaign_graph,
    "promotion_launch": build_ad_campaign_graph,
    "customer_ops": build_customer_ops_graph,
    "review_monitor": build_customer_ops_graph,
    "product_selection": build_selection_graph,
}


def detect_runtime_modes() -> Dict[str, str]:
    provider = os.environ.get("MAS_LLM_PROVIDER", "anthropic")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if provider == "anthropic":
        provider_available = importlib.util.find_spec("anthropic") is not None
    elif provider == "openai":
        provider_available = importlib.util.find_spec("openai") is not None
    else:
        provider_available = False

    llm_mode = "real_llm" if has_key and provider_available else "stub"
    mcp_mode = os.environ.get("MAS_MCP_MODE", "mcp_stub")
    return {"llm": llm_mode, "mcp": mcp_mode}


class MAS:
    def __init__(
        self,
        checkpointer: Optional[SQLiteCheckpointer] = None,
        interrupt_fn: Optional[Callable[[Dict], Dict]] = None,
        approval_store: Optional[Any] = None,
        default_token_budget: int = 50_000,
    ) -> None:
        self.checkpointer = checkpointer or SQLiteCheckpointer()
        self.interrupt_fn = interrupt_fn
        self.default_token_budget = default_token_budget
        self.tracer = GLOBAL_TRACER
        self.approval_store = approval_store or getattr(interrupt_fn, "approval_store", GLOBAL_STORE)
        self.runtime_modes = detect_runtime_modes()

    def trigger(
        self,
        workflow_type: str,
        payload: Dict[str, Any],
        operator_id: str,
        workflow_id: Optional[str] = None,
        token_budget: Optional[int] = None,
        interrupt_fn: Optional[Callable[[Dict], Dict]] = None,
    ) -> Dict[str, Any]:
        if workflow_type not in WORKFLOW_TYPE_TO_BUILDER:
            return {"error": f"unknown workflow_type: {workflow_type}", "available": list(WORKFLOW_TYPE_TO_BUILDER)}

        wf_id = workflow_id or f"wf-{workflow_type}-{uuid.uuid4().hex[:8]}"
        budget = token_budget or self.default_token_budget
        effective_interrupt = interrupt_fn or self.interrupt_fn

        builder = WORKFLOW_TYPE_TO_BUILDER[workflow_type]
        graph = builder(interrupt_fn=effective_interrupt)

        state = init_state(wf_id, workflow_type, operator_id, payload, token_budget=budget)
        self.tracer.record(wf_id, "workflow_start", agent="orchestrator", tokens=0)

        result = graph.invoke(state, recursion_limit=50)

        self.checkpointer.save(result)
        self.tracer.record(
            wf_id, "workflow_end",
            agent="orchestrator",
            tokens=result.get("token_usage", 0),
            approved=result.get("approved"),
        )
        return result

    def list_pending_approvals(self) -> list:
        return [
            {"workflow_id": r.workflow_id, "workflow_type": r.workflow_type, "cost": r.estimated_cost}
            for r in self.approval_store.list_pending()
        ]

    def resume(self, workflow_id: str, action: str, note: str = "") -> Optional[Dict[str, Any]]:
        req = self.approval_store.resolve(workflow_id, action, note)
        if req is None:
            return None
        snapshot = self.checkpointer.load(workflow_id)
        if snapshot is None:
            return None
        snapshot["approved"] = (action == "approve")
        snapshot["approval_note"] = note
        snapshot["pending_approval"] = None
        snapshot["final_output"] = {
            "executed": action == "approve",
            "reason": "approved_async" if action == "approve" else "rejected_async",
            "approval_note": note,
            "workflow_id": workflow_id,
        }
        self.checkpointer.save(snapshot)
        return snapshot

    def trace_summary(self, workflow_id: str) -> Dict[str, Any]:
        return self.tracer.summary(workflow_id)

    def available_workflows(self) -> list:
        return sorted(WORKFLOW_TYPE_TO_BUILDER.keys())


def _demo() -> None:
    mas = MAS(interrupt_fn=make_blocking_interrupt())
    print(f"可用工作流: {mas.available_workflows()}")
    print(f"运行模式: {mas.runtime_modes}")

    print("\n=== 演示 1: 选品扫描 ===")
    result = mas.trigger(
        workflow_type="product_selection",
        operator_id="demo-user",
        payload={"candidates": [
            {
                "id": "DEMO1", "name": "Demo Formula", "category": "infant_formula",
                "monthly_sales_usd": 100_000, "review_count": 200, "avg_rating": 4.5,
                "bsr_trend_30d": 0.3, "selling_price_usd": 40, "cogs_usd": 15,
                "fba_fee_usd": 4, "freight_usd": 2, "duty_rate": 0.05,
                "description": "organic infant formula HMO DHA stage1-0-6m",
                "seasonality_factor": 1.1, "competitor_count": 10,
            }
        ]},
    )
    print(f"  executed: {result['final_output']['executed']}")
    print(f"  trace: {mas.trace_summary(result['workflow_id'])}")

    print("\n=== 演示 2: 客服 Case 分诊 ===")
    result = mas.trigger(
        workflow_type="customer_ops",
        operator_id="cs-demo",
        payload={"case": {
            "case_id": "DEMO-CS-1",
            "text": "I want to refund this item, just arrived yesterday",
            "days_since_order": 1,
            "user_complaint_history": 0,
            "order_amount": 99.0,
        }},
    )
    print(f"  executed: {result['final_output']['executed']}")
    print(f"  decision: {result['skill_outputs'][-1]['output']['decision']['action']}")


if __name__ == "__main__":
    _demo()

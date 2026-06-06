"""Regression tests for paper2skills governance and MAS state semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mas.agents.base import BaseAgent
from mas.agents.orchestrator import human_approval_gate
from mas.checkpointing.sqlite_saver import SQLiteCheckpointer
from mas.graphs.base_graph import merge_state
from mas.hitl.approval_api import ApprovalStore, make_blocking_interrupt
from mas.main import MAS
from mas.state.schema import init_state


ROOT = Path(__file__).resolve().parents[2]


def _restock_payload() -> dict:
    return {
        "sku_id": "ASYNC-FORMULA",
        "history_daily_sales": [120] * 30,
        "current_stock": 100,
        "in_transit": 0,
        "lead_time_days": 30,
        "moq": 500,
        "unit_cost_rmb": 80.0,
    }


def test_base_agent_returns_token_delta_not_cumulative_total():
    def fixed_llm(_messages, _tools):
        return {
            "output": "ok",
            "skill_name": "tool",
            "confidence": 1.0,
            "estimated_cost": 0.0,
            "token_usage": 100,
        }

    agent = BaseAgent(
        name="TokenAgent",
        role_description="token regression",
        skill_domains=[],
        llm_invoker=fixed_llm,
    )
    state = init_state("wf-token", "restock", "op", {}, token_budget=10_000)
    state["token_usage"] = 50

    delta = agent(state)
    assert delta["token_usage"] == 100
    assert merge_state(state, delta)["token_usage"] == 150


def test_pending_human_approval_preserves_request_without_rejecting():
    state = init_state("wf-pending", "large_restock", "op", {})
    state["skill_outputs"] = [{"estimated_cost": 50_000.0, "status": "ok"}]

    delta = human_approval_gate(
        state,
        interrupt_fn=lambda req: {"action": "pending", "note": "queued"},
    )

    assert delta["approved"] is None
    assert delta["approval_note"] == "queued"
    assert delta["pending_approval"]["workflow_id"] == "wf-pending"
    assert delta["pending_approval"]["available_actions"] == ["approve", "reject", "modify"]


def test_async_pending_workflow_can_resume_after_external_approval(tmp_path):
    store = ApprovalStore()
    checkpointer = SQLiteCheckpointer(str(tmp_path / "checkpoints.db"))
    mas = MAS(checkpointer=checkpointer, interrupt_fn=make_blocking_interrupt(store=store))

    result = mas.trigger(
        workflow_type="large_restock",
        operator_id="ops",
        payload=_restock_payload(),
        workflow_id="wf-async-approval",
    )

    assert result["approved"] is None
    assert result["pending_approval"]["workflow_id"] == "wf-async-approval"
    assert store.list_pending()[0].workflow_id == "wf-async-approval"

    resumed = mas.resume("wf-async-approval", "approve", note="approved externally")

    assert resumed is not None
    assert resumed["approved"] is True
    assert resumed["final_output"]["executed"] is True
    assert checkpointer.load("wf-async-approval")["approved"] is True


def test_domain_registry_includes_compliance_from_claude_source():
    from paper2skills_common.domains import load_domain_registry

    registry = load_domain_registry(ROOT)
    compliance = registry.by_key["compliance"]
    assert compliance.vault_dir == "21-合规决策"
    assert "21-合规决策" in registry.vault_to_key


def test_graph_analyzer_uses_shared_registry_with_compliance_domain():
    script = ROOT / "paper2skills-skills" / "paper-skills-graph" / "scripts" / "skills_graph_analyzer.py"
    spec = importlib.util.spec_from_file_location("skills_graph_analyzer", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    graph = module.SkillsGraph(str(ROOT / "paper2skills-vault"))
    assert "compliance" in graph.domain_mapping.values()


def test_markdown_utf8_health_check_is_clean_after_workflow_skill_repair():
    from paper2skills_common.doctor import check_markdown_utf8

    issues = check_markdown_utf8(ROOT)
    assert issues == []

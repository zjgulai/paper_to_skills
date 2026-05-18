"""阶段 0 烟雾测试: 验证 MAS 骨架完整可跑.

测试覆盖:
  - SkillRegistry 加载 90+ 工具,按领域过滤
  - BaseAgent stub LLM 跑通,token 追踪正确
  - Orchestrator 路由 + HITL gate 双路径(low/high cost)
  - StateGraph 端到端 invoke,merge_state reducer 正确
"""

from __future__ import annotations

from mas.agents.base import BaseAgent
from mas.agents.orchestrator import (
    execute_approved_action,
    handle_rejection,
    human_approval_gate,
    orchestrator_route,
)
from mas.graphs.base_graph import END, START, GraphRecursionError, StateGraph, build_workflow_graph
from mas.skills.registry import SkillRegistry
from mas.state.schema import init_state


def test_skill_registry_bootstrap():
    reg = SkillRegistry()
    assert reg.total_tools() >= 60
    domains = reg.all_domains()
    expected = {
        "supply_chain", "time_series", "causal_inference",
        "marketing", "advertising", "ab_testing",
        "growth_model", "recommendation", "user_analytics",
        "data_agent_llm", "knowledge_graph", "ml_fundamentals",
    }
    assert expected.issubset(set(domains))


def test_registry_domain_filter():
    reg = SkillRegistry()
    sc_tools = reg.get_tools_for_domains(["supply_chain"])
    assert len(sc_tools) == 5
    assert all(t.domain == "supply_chain" for t in sc_tools)


def test_base_agent_stub_invocation():
    agent = BaseAgent(
        name="TestAgent",
        role_description="测试 stub agent",
        skill_domains=["supply_chain"],
    )
    state = init_state("wf-001", "restock", "op-1", {"sku": "TEST"}, token_budget=10_000)
    delta = agent(state)
    assert "skill_outputs" in delta
    assert delta["skill_outputs"][0]["agent"] == "TestAgent"
    assert delta["token_usage"] >= 100


def test_base_agent_budget_skip():
    agent = BaseAgent(name="A", role_description="r", skill_domains=["supply_chain"])
    state = init_state("wf-002", "restock", "op-1", {}, token_budget=500)
    delta = agent(state)
    assert delta.get("error") == "token_budget_low"


def test_orchestrator_low_cost_auto_approve():
    state = init_state("wf-003", "restock", "op-1", {})
    state["skill_outputs"] = [{"estimated_cost": 500.0}]
    delta = human_approval_gate(state, interrupt_fn=None)
    assert delta["approved"] is True
    assert "auto_approved" in delta["approval_note"]


def test_orchestrator_high_cost_needs_approval():
    state = init_state("wf-004", "restock", "op-1", {})
    state["skill_outputs"] = [{"estimated_cost": 50_000.0}]
    delta = human_approval_gate(state, interrupt_fn=None)
    assert delta.get("approved") is None
    assert delta["pending_approval"]["risk_level"] == "medium"
    assert delta["pending_approval"]["estimated_cost"] == 50_000.0


def test_orchestrator_risky_workflow_type():
    state = init_state("wf-005", "ad_budget_increase", "op-1", {})
    state["skill_outputs"] = [{"estimated_cost": 100.0}]
    delta = human_approval_gate(state, interrupt_fn=None)
    assert delta.get("approved") is None
    assert delta["pending_approval"]["workflow_type"] == "ad_budget_increase"


def test_orchestrator_interrupt_callback():
    state = init_state("wf-006", "large_restock", "op-1", {})
    state["skill_outputs"] = [{"estimated_cost": 100_000.0}]

    def approve(req):
        return {"action": "approve", "note": "looks good"}

    delta = human_approval_gate(state, interrupt_fn=approve)
    assert delta["approved"] is True
    assert delta["approval_note"] == "looks good"


def test_end_to_end_low_cost_workflow():
    agent = BaseAgent(name="HelloAgent", role_description="hello", skill_domains=["supply_chain"])

    def approval(state):
        return human_approval_gate(state, interrupt_fn=None)

    graph = build_workflow_graph(
        agent_name="HelloAgent",
        agent_fn=agent,
        approval_fn=approval,
        execute_fn=execute_approved_action,
        reject_fn=handle_rejection,
    )

    state = init_state("wf-e2e-low", "restock", "op-1", {"sku": "HELLO"}, token_budget=20_000)
    state["skill_outputs"] = [{"estimated_cost": 100.0}]
    result = graph.invoke(state)

    assert result["final_output"]["executed"] is True
    assert result["approved"] is True
    assert result["token_usage"] >= 100


def test_end_to_end_high_cost_requires_human():
    class HighCostAgent(BaseAgent):
        @staticmethod
        def _stub_llm(messages, tools):
            return {
                "output": "建议大额补货",
                "skill_name": "supply_two_echelon_drl",
                "confidence": 0.9,
                "estimated_cost": 80_000.0,
                "token_usage": 200,
            }

    agent = HighCostAgent(name="BigSpenderAgent", role_description="贵货", skill_domains=["supply_chain"])

    def approve_callback(state):
        def approve(req):
            return {"action": "approve", "note": f"approved {req['workflow_id']}"}
        return human_approval_gate(state, interrupt_fn=approve)

    graph = build_workflow_graph(
        agent_name="BigSpenderAgent",
        agent_fn=agent,
        approval_fn=approve_callback,
        execute_fn=execute_approved_action,
        reject_fn=handle_rejection,
    )

    state = init_state("wf-e2e-high", "large_restock", "op-1", {"sku": "BIG"}, token_budget=20_000)
    result = graph.invoke(state)

    assert result["final_output"]["executed"] is True
    assert result["approved"] is True
    assert result["approval_note"].startswith("approved")


def test_end_to_end_rejection():
    agent = BaseAgent(name="A", role_description="r", skill_domains=["supply_chain"])

    def reject_callback(state):
        def reject(req):
            return {"action": "reject", "note": "金额超预算"}
        return human_approval_gate(state, interrupt_fn=reject)

    graph = build_workflow_graph(
        agent_name="A",
        agent_fn=agent,
        approval_fn=reject_callback,
        execute_fn=execute_approved_action,
        reject_fn=handle_rejection,
    )

    state = init_state("wf-reject", "large_restock", "op-1", {}, token_budget=20_000)
    state["skill_outputs"] = [{"estimated_cost": 200_000.0}]
    result = graph.invoke(state)

    assert result["final_output"]["executed"] is False
    assert result["final_output"]["reason"] == "rejected_by_human"


def test_recursion_limit_guard():
    graph = StateGraph()

    def loop_node(state):
        return {}

    graph.add_node("loop", loop_node)
    graph.add_edge(START, "loop")
    graph.add_edge("loop", "loop")

    try:
        graph.invoke(init_state("wf-loop", "test", "op", {}), recursion_limit=5)
        assert False, "should have raised GraphRecursionError"
    except GraphRecursionError as e:
        assert "recursion_limit=5" in str(e)


def run_all():
    tests = [
        test_skill_registry_bootstrap,
        test_registry_domain_filter,
        test_base_agent_stub_invocation,
        test_base_agent_budget_skip,
        test_orchestrator_low_cost_auto_approve,
        test_orchestrator_high_cost_needs_approval,
        test_orchestrator_risky_workflow_type,
        test_orchestrator_interrupt_callback,
        test_end_to_end_low_cost_workflow,
        test_end_to_end_high_cost_requires_human,
        test_end_to_end_rejection,
        test_recursion_limit_guard,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, f"AssertionError: {e}"))
            print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ❌ {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n阶段 0 集成测试结果: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())

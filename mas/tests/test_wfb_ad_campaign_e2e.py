"""WF-B 广告关键词优化端到端集成测试."""

from __future__ import annotations

from mas.graphs.ad_campaign_graph import build_ad_campaign_graph
from mas.state.schema import init_state


def _mock_search_terms():
    return [
        {"search_term": "baby formula 0-6m", "clicks": 50, "orders": 8, "spend": 80.0, "revenue": 480.0, "bid": 1.5},
        {"search_term": "infant formula organic", "clicks": 30, "orders": 5, "spend": 50.0, "revenue": 300.0, "bid": 1.2},
        {"search_term": "milk powder hipp", "clicks": 25, "orders": 0, "spend": 40.0, "revenue": 0.0, "bid": 1.0},
        {"search_term": "baby food cheap", "clicks": 100, "orders": 0, "spend": 120.0, "revenue": 0.0, "bid": 0.8},
        {"search_term": "newborn pacifier silicone", "clicks": 20, "orders": 4, "spend": 25.0, "revenue": 200.0, "bid": 1.0},
    ]


def _mock_channel_history():
    return [
        {"channel": "google", "spend": 3000, "revenue": 12000, "roas": 4.0},
        {"channel": "google", "spend": 3500, "revenue": 13000, "roas": 3.71},
        {"channel": "meta", "spend": 2000, "revenue": 6000, "roas": 3.0},
        {"channel": "meta", "spend": 2500, "revenue": 7500, "roas": 3.0},
        {"channel": "tiktok", "spend": 1500, "revenue": 7500, "roas": 5.0},
    ]


def test_wfb_healthy_tacos_auto_approved():
    payload = {
        "search_term_rows": [
            {"search_term": "kw1", "clicks": 50, "orders": 8, "spend": 80, "revenue": 800, "bid": 1.5},
        ],
        "channel_history": _mock_channel_history(),
        "total_budget": 10_000,
        "target_tacos": 0.20,
    }

    graph = build_ad_campaign_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-b-healthy", "ad_campaign", "ops-li", payload, token_budget=20_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    assert analysis["tacos_status"] == "healthy"
    assert result["approved"] is True
    assert result["final_output"]["executed"] is True


def test_wfb_negative_keywords_identified():
    payload = {
        "search_term_rows": _mock_search_terms(),
        "channel_history": _mock_channel_history(),
        "total_budget": 10_000,
        "target_tacos": 0.15,
    }

    captured = {}

    def approve(req):
        captured["req"] = req
        return {"action": "approve", "note": "优化执行"}

    graph = build_ad_campaign_graph(interrupt_fn=approve)
    state = init_state("wf-b-neg", "ad_budget_increase", "ops-li", payload, token_budget=20_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    assert analysis["negatives"]["negative_count"] >= 2
    assert analysis["negatives"]["spend_wasted"] >= 100
    assert "milk powder hipp" in [n["search_term"] for n in analysis["negatives"]["negatives"]]
    assert "baby food cheap" in [n["search_term"] for n in analysis["negatives"]["negatives"]]

    assert "req" in captured
    assert result["final_output"]["executed"] is True


def test_wfb_uplift_promotion_high_roas_terms():
    payload = {
        "search_term_rows": _mock_search_terms(),
        "channel_history": _mock_channel_history(),
        "total_budget": 10_000,
        "target_tacos": 0.15,
    }

    graph = build_ad_campaign_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-b-uplift", "ad_budget_increase", "ops-li", payload, token_budget=20_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    promote_terms = [p["search_term"] for p in analysis["promotions"]["promotions"]]
    assert "baby formula 0-6m" in promote_terms
    assert "infant formula organic" in promote_terms


def test_wfb_dara_cross_channel_allocation_sums_to_budget():
    payload = {
        "search_term_rows": _mock_search_terms(),
        "channel_history": _mock_channel_history(),
        "total_budget": 10_000,
        "target_tacos": 0.15,
    }

    graph = build_ad_campaign_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-b-dara", "ad_campaign", "ops-li", payload, token_budget=20_000)
    result = graph.invoke(state)

    alloc = result["skill_outputs"][-1]["output"]["dara"]["allocation"]
    assert set(alloc.keys()) == {"google", "meta", "tiktok"}
    assert abs(sum(alloc.values()) - 10_000) < 2.0
    assert alloc["tiktok"] > alloc["meta"]


def test_wfb_rejection_path():
    payload = {
        "search_term_rows": _mock_search_terms(),
        "channel_history": _mock_channel_history(),
        "total_budget": 100_000,
        "target_tacos": 0.15,
    }

    graph = build_ad_campaign_graph(interrupt_fn=lambda r: {"action": "reject", "note": "本周冻结预算"})
    state = init_state("wf-b-reject", "ad_budget_increase", "ops-li", payload, token_budget=20_000)
    result = graph.invoke(state)

    assert result["final_output"]["executed"] is False
    assert result["final_output"]["reason"] == "rejected_by_human"


def test_wfb_skill_chain_complete():
    from mas.skills.registry import SkillRegistry
    payload = {
        "search_term_rows": _mock_search_terms(),
        "channel_history": _mock_channel_history(),
        "total_budget": 10_000,
        "target_tacos": 0.15,
    }
    graph = build_ad_campaign_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-b-chain", "ad_campaign", "ops-li", payload, token_budget=20_000)
    result = graph.invoke(state)

    chain = result["skill_outputs"][-1]["output"]["skill_chain"]
    assert "ad_search_term_parse" in chain
    assert "ad_negative_keywords" in chain
    assert "causal_uplift_modeling_ad" in chain
    assert "marketing_mmm" in chain
    assert "marketing_dara_optimizer" in chain

    reg = SkillRegistry()
    ad_tools = {t.name for t in reg.get_tools_for_domains(["advertising", "marketing"])}
    assert "ad_roas_budget_optimization" in ad_tools
    assert "marketing_mmm" in ad_tools


def run_all():
    tests = [
        test_wfb_healthy_tacos_auto_approved,
        test_wfb_negative_keywords_identified,
        test_wfb_uplift_promotion_high_roas_terms,
        test_wfb_dara_cross_channel_allocation_sums_to_budget,
        test_wfb_rejection_path,
        test_wfb_skill_chain_complete,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ❌ {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nWF-B 端到端集成测试: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())

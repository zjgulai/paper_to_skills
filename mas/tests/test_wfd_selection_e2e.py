"""WF-D 选品扫描端到端集成测试."""

from __future__ import annotations

from mas.graphs.selection_graph import build_selection_graph
from mas.state.schema import init_state


def _mock_candidates():
    return [
        {
            "id": "C1",
            "name": "Organic Infant Formula Stage 1",
            "category": "infant_formula",
            "monthly_sales_usd": 120_000,
            "review_count": 230,
            "avg_rating": 4.5,
            "bsr_trend_30d": 0.35,
            "selling_price_usd": 45,
            "cogs_usd": 18,
            "fba_fee_usd": 4,
            "freight_usd": 2,
            "duty_rate": 0.05,
            "description": "Organic infant formula with HMO and DHA stage1-0-6m European brand",
            "seasonality_factor": 1.1,
            "competitor_count": 12,
        },
        {
            "id": "C2",
            "name": "Cheap Plastic Toy with Small Parts",
            "category": "toy",
            "monthly_sales_usd": 80_000,
            "review_count": 800,
            "avg_rating": 3.5,
            "bsr_trend_30d": 0.05,
            "selling_price_usd": 15,
            "cogs_usd": 6,
            "fba_fee_usd": 3,
            "freight_usd": 1,
            "duty_rate": 0.05,
            "description": "Plastic toy with small parts and lead-free paint, battery operated electric",
            "seasonality_factor": 1.0,
            "competitor_count": 30,
        },
        {
            "id": "C3",
            "name": "Anti-Colic Wide Neck Bottle Set",
            "category": "bottle",
            "monthly_sales_usd": 90_000,
            "review_count": 180,
            "avg_rating": 4.7,
            "bsr_trend_30d": 0.45,
            "selling_price_usd": 30,
            "cogs_usd": 10,
            "fba_fee_usd": 3,
            "freight_usd": 1,
            "duty_rate": 0.05,
            "description": "anti-colic wide neck silicone nipple BPA-free baby bottle",
            "seasonality_factor": 1.05,
            "competitor_count": 8,
        },
        {
            "id": "C4",
            "name": "Premium Stroller (low margin)",
            "category": "stroller",
            "monthly_sales_usd": 200_000,
            "review_count": 1500,
            "avg_rating": 4.2,
            "bsr_trend_30d": 0.10,
            "selling_price_usd": 350,
            "cogs_usd": 220,
            "fba_fee_usd": 25,
            "freight_usd": 30,
            "duty_rate": 0.05,
            "description": "lightweight under 6kg one-hand fold compact stroller",
            "seasonality_factor": 1.0,
            "competitor_count": 25,
        },
    ]


def test_wfd_finds_compliant_high_score_candidate():
    payload = {"candidates": _mock_candidates()}
    graph = build_selection_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-d-01", "product_selection", "ops-tang", payload, token_budget=20_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    assert analysis["total_candidates"] == 4
    assert analysis["recommend_count"] >= 1
    assert any(p["candidate_id"] == "C1" for p in analysis["top_picks"])
    assert any(p["candidate_id"] == "C3" for p in analysis["top_picks"])


def test_wfd_excludes_high_compliance_risk():
    payload = {"candidates": _mock_candidates()}
    graph = build_selection_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-d-risk", "product_selection", "ops-tang", payload, token_budget=20_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    assert not any(p["candidate_id"] == "C2" for p in analysis["top_picks"])
    c2 = next(s for s in analysis["scored"] if s["candidate_id"] == "C2")
    assert c2["compliance"]["risk_level"] == "high"


def test_wfd_low_margin_rejected():
    payload = {"candidates": _mock_candidates()}
    graph = build_selection_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-d-margin", "product_selection", "ops-tang", payload, token_budget=20_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    c4 = next(s for s in analysis["scored"] if s["candidate_id"] == "C4")
    assert c4["margin"]["passes_40pct_target"] is False


def test_wfd_auto_approved_zero_cost():
    payload = {"candidates": _mock_candidates()}
    graph = build_selection_graph(interrupt_fn=lambda r: (_ for _ in ()).throw(AssertionError("不应被调用")))
    state = init_state("wf-d-auto", "product_selection", "ops-tang", payload, token_budget=20_000)
    result = graph.invoke(state)

    assert result["approved"] is True
    assert "auto_approved" in result["approval_note"]
    assert result["final_output"]["executed"] is True


def run_all():
    tests = [
        test_wfd_finds_compliant_high_score_candidate,
        test_wfd_excludes_high_compliance_risk,
        test_wfd_low_margin_rejected,
        test_wfd_auto_approved_zero_cost,
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
    print(f"\nWF-D 端到端集成测试: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())

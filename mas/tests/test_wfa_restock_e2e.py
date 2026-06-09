"""WF-A 智能补货端到端集成测试.

3 个真实母婴 SKU 场景:
  1. 低库存爆款奶粉 → 大额补货 (need approval)
  2. 充足库存普通用品 → no_action (auto-approve)
  3. 大促前反事实预测 (含 promotion intervention) → 高额补货 (need approval)
"""

from __future__ import annotations

from mas.graphs.restock_graph import build_restock_graph
from mas.state.schema import init_state


def test_wfa_low_stock_baby_formula_needs_approval():
    history = [120, 110, 130, 125, 140, 135, 145, 150, 160, 155,
               165, 170, 160, 175, 180, 170, 185, 190, 200, 195,
               210, 205, 215, 220, 225, 230, 240, 245, 250, 260]

    payload = {
        "sku_id": "AB-FORMULA-STAGE1-800G",
        "history_daily_sales": history,
        "current_stock": 800,
        "in_transit": 0,
        "lead_time_days": 35,
        "season_multiplier": 1.0,
        "interventions": [],
        "moq": 1000,
        "unit_cost_rmb": 80.0,
        "service_level": 0.95,
    }

    captured = {}

    def approve_callback(req):
        captured["request"] = req
        return {"action": "approve", "note": "正常补货,资金充裕"}

    graph = build_restock_graph(interrupt_fn=approve_callback)
    state = init_state("wf-formula-01", "large_restock", "ops-shen", payload, token_budget=20_000)
    result = graph.invoke(state)

    rec = result["skill_outputs"][-1]["output"]["recommendation"]
    assert rec["recommendation"] == "place_purchase_order"
    assert rec["recommended_qty"] >= rec["moq"]
    assert rec["estimated_cost"] > 10_000

    assert "request" in captured
    assert captured["request"]["workflow_type"] == "large_restock"

    assert result["approved"] is True
    assert result["final_output"]["executed"] is True
    assert "正常补货" in result["approval_note"]


def test_wfa_sufficient_stock_auto_no_action():
    history = [20] * 30
    payload = {
        "sku_id": "AB-PACIFIER-BASIC",
        "history_daily_sales": history,
        "current_stock": 5_000,
        "in_transit": 2_000,
        "lead_time_days": 30,
        "moq": 500,
        "unit_cost_rmb": 5.0,
    }

    def should_not_be_called(req):
        raise AssertionError("不应该触发审批,库存充裕应自动通过")

    graph = build_restock_graph(interrupt_fn=should_not_be_called)
    state = init_state("wf-pacifier-01", "restock", "ops-shen", payload, token_budget=20_000)
    result = graph.invoke(state)

    rec = result["skill_outputs"][-1]["output"]["recommendation"]
    assert rec["recommendation"] == "no_action"
    assert rec["estimated_cost"] == 0
    assert result["approved"] is True
    assert "auto_approved" in result["approval_note"]
    assert result["final_output"]["executed"] is True


def test_wfa_promotion_uplift_drives_large_order():
    history = [50, 48, 52, 55, 47, 53, 51] * 4 + [50, 48]

    payload = {
        "sku_id": "AB-DIAPER-NB-PACK80",
        "history_daily_sales": history,
        "current_stock": 1_500,
        "in_transit": 0,
        "lead_time_days": 28,
        "season_multiplier": 1.0,
        "interventions": [
            {"type": "promotion", "expected_lift": 1.5, "campaign": "Prime Day 2026"},
        ],
        "moq": 2000,
        "unit_cost_rmb": 40.0,
    }

    decision_log = []

    def approve_callback(req):
        decision_log.append({
            "cost": req["estimated_cost"],
            "risk": req["risk_level"],
        })
        return {"action": "approve", "note": f"Prime Day 备货确认 ¥{req['estimated_cost']:,.0f}"}

    graph = build_restock_graph(interrupt_fn=approve_callback)
    state = init_state("wf-diaper-promo", "promotion_launch", "ops-shen", payload, token_budget=20_000)
    result = graph.invoke(state)

    counterfactual = result["skill_outputs"][-1]["output"]["counterfactual"]
    rec = result["skill_outputs"][-1]["output"]["recommendation"]

    assert counterfactual["intervention_lift_pct"] >= 100
    assert rec["recommended_qty"] >= 2000
    assert rec["estimated_cost"] >= 80_000

    assert len(decision_log) == 1
    assert decision_log[0]["risk"] == "high"
    assert result["final_output"]["executed"] is True


def test_wfa_rejection_path():
    history = [200] * 30
    payload = {
        "sku_id": "AB-EXPENSIVE-STROLLER",
        "history_daily_sales": history,
        "current_stock": 50,
        "in_transit": 0,
        "lead_time_days": 60,
        "moq": 100,
        "unit_cost_rmb": 1500.0,
    }

    def reject_callback(req):
        return {"action": "reject", "note": "现金流紧张,推迟到下月"}

    graph = build_restock_graph(interrupt_fn=reject_callback)
    state = init_state("wf-stroller", "large_restock", "ops-shen", payload, token_budget=20_000)
    result = graph.invoke(state)

    assert result["final_output"]["executed"] is False
    assert result["final_output"]["reason"] == "rejected_by_human"
    assert "现金流紧张" in result["approval_note"]


def test_wfa_token_usage_tracked():
    history = [100] * 20
    payload = {
        "sku_id": "AB-TEST",
        "history_daily_sales": history,
        "current_stock": 500,
        "in_transit": 0,
        "lead_time_days": 30,
    }

    graph = build_restock_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-token-test", "restock", "ops-shen", payload, token_budget=20_000)
    result = graph.invoke(state)

    assert result["token_usage"] > 0
    assert result["token_usage"] < state["token_budget"]


def run_all():
    tests = [
        test_wfa_low_stock_baby_formula_needs_approval,
        test_wfa_sufficient_stock_auto_no_action,
        test_wfa_promotion_uplift_drives_large_order,
        test_wfa_rejection_path,
        test_wfa_token_usage_tracked,
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
    print(f"\nWF-A 端到端集成测试: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())

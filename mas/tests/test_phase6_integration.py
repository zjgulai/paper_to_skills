"""阶段 6 集成测试: MAS 入口 + 5 工作流并发触发 + checkpointing + tracer."""

from __future__ import annotations

from mas.checkpointing.sqlite_saver import SQLiteCheckpointer
from mas.hitl.approval_api import GLOBAL_STORE
from mas.main import MAS


def _restock_payload():
    return {
        "sku_id": "SMOKE-FORMULA",
        "history_daily_sales": [120] * 30,
        "current_stock": 1000,
        "in_transit": 0,
        "lead_time_days": 30,
        "moq": 500,
        "unit_cost_rmb": 60.0,
    }


def _ad_payload():
    return {
        "search_term_rows": [
            {"search_term": "kw1", "clicks": 50, "orders": 8, "spend": 80, "revenue": 600, "bid": 1.5},
            {"search_term": "bad_kw", "clicks": 20, "orders": 0, "spend": 30, "revenue": 0, "bid": 1.0},
        ],
        "channel_history": [
            {"channel": "google", "spend": 3000, "revenue": 12000, "roas": 4.0},
            {"channel": "meta", "spend": 2000, "revenue": 6000, "roas": 3.0},
        ],
        "total_budget": 10_000,
        "target_tacos": 0.15,
    }


def _case_payload():
    return {"case": {
        "case_id": "SMOKE-CS",
        "text": "请退款,商品有问题",
        "days_since_order": 3,
        "user_complaint_history": 0,
        "order_amount": 150.0,
    }}


def _review_payload():
    return {"reviews": [{"rating": 5, "text": "great"}, {"rating": 4, "text": "good"}]}


def _selection_payload():
    return {"candidates": [{
        "id": "SMOKE-S1", "name": "Test Formula", "category": "infant_formula",
        "monthly_sales_usd": 100_000, "review_count": 200, "avg_rating": 4.4,
        "bsr_trend_30d": 0.30, "selling_price_usd": 40, "cogs_usd": 15,
        "fba_fee_usd": 4, "freight_usd": 2, "duty_rate": 0.05,
        "description": "organic infant formula HMO DHA stage1-0-6m European brand",
        "seasonality_factor": 1.0, "competitor_count": 10,
    }]}


def test_mas_available_workflows_includes_all_five():
    mas = MAS()
    wfs = mas.available_workflows()
    expected = {"restock", "ad_campaign", "customer_ops", "review_monitor", "product_selection"}
    assert expected.issubset(set(wfs))


def test_mas_triggers_all_5_workflows_in_sequence():
    GLOBAL_STORE._store.clear()
    mas = MAS(interrupt_fn=lambda r: {"action": "approve", "note": "测试通过"})

    triggers = [
        ("restock", _restock_payload()),
        ("ad_campaign", _ad_payload()),
        ("customer_ops", _case_payload()),
        ("review_monitor", _review_payload()),
        ("product_selection", _selection_payload()),
    ]

    workflow_ids = []
    for wf_type, payload in triggers:
        result = mas.trigger(workflow_type=wf_type, operator_id="smoke", payload=payload)
        assert not result.get("error"), f"{wf_type} 失败: {result.get('error')}"
        assert result.get("final_output", {}).get("executed") is not None
        workflow_ids.append(result["workflow_id"])

    for wf_id in workflow_ids:
        snap = mas.checkpointer.load(wf_id)
        assert snap is not None
        assert snap["workflow_id"] == wf_id
        trace = mas.trace_summary(wf_id)
        assert trace["event_count"] >= 2
        assert trace["total_tokens"] > 0


def test_mas_unknown_workflow_returns_error():
    mas = MAS()
    result = mas.trigger("bogus_type", payload={}, operator_id="op")
    assert "error" in result
    assert "unknown" in result["error"]


def test_mas_token_budget_respected():
    mas = MAS(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    result = mas.trigger(
        workflow_type="product_selection",
        operator_id="smoke",
        payload=_selection_payload(),
        token_budget=20_000,
    )
    assert result["token_usage"] < result["token_budget"]


def test_mas_checkpoint_persistence():
    mas = MAS(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    result = mas.trigger("restock", payload=_restock_payload(), operator_id="persist-test")
    wf_id = result["workflow_id"]

    mas2 = MAS()
    snap = mas2.checkpointer.load(wf_id)
    assert snap is not None
    assert snap["operator_id"] == "persist-test"


def test_mas_global_tracer_aggregates():
    mas = MAS(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    initial = mas.tracer.global_summary()["total_workflows"]

    mas.trigger("review_monitor", payload=_review_payload(), operator_id="trace-test")

    after = mas.tracer.global_summary()["total_workflows"]
    assert after >= initial + 1


def run_all():
    tests = [
        test_mas_available_workflows_includes_all_five,
        test_mas_triggers_all_5_workflows_in_sequence,
        test_mas_unknown_workflow_returns_error,
        test_mas_token_budget_respected,
        test_mas_checkpoint_persistence,
        test_mas_global_tracer_aggregates,
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
    print(f"\n阶段 5/6 集成测试: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())

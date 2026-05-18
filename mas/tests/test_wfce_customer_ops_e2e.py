"""WF-C 客服分诊 + WF-E Review 监控 端到端集成测试."""

from __future__ import annotations

from mas.graphs.customer_ops_graph import build_customer_ops_graph
from mas.state.schema import init_state


def test_wfc_chinese_refund_small_auto_approved():
    case = {
        "case_id": "CS-001",
        "text": "你好,我想申请退款,商品昨天才到不需要了",
        "days_since_order": 2,
        "user_complaint_history": 0,
        "order_amount": 199.0,
    }
    graph = build_customer_ops_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-c-zh", "customer_ops", "cs-zhao", {"case": case}, token_budget=10_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    assert analysis["language"] == "zh"
    assert analysis["intent"] == "refund"
    assert analysis["decision"]["action"] == "auto_approve"
    assert result["approved"] is True
    assert result["final_output"]["executed"] is True


def test_wfc_english_complaint_negative_sentiment():
    case = {
        "case_id": "CS-002",
        "text": "This product is awful, the bottle is broken and leaking everywhere",
        "days_since_order": 5,
        "user_complaint_history": 0,
        "order_amount": 89.0,
    }
    graph = build_customer_ops_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-c-en", "customer_ops", "cs-zhao", {"case": case}, token_budget=10_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    assert analysis["language"] == "en"
    assert analysis["intent"] == "complaint"
    assert analysis["sentiment"]["sentiment"] == "negative"
    assert analysis["sentiment"]["score"] < 0
    assert result["final_output"]["executed"] is True


def test_wfc_large_refund_triggers_manual_review():
    case = {
        "case_id": "CS-003",
        "text": "I want to refund this expensive stroller",
        "days_since_order": 10,
        "user_complaint_history": 0,
        "order_amount": 1800.0,
    }
    graph = build_customer_ops_graph(interrupt_fn=lambda r: {"action": "approve", "note": "金额大单确认"})
    state = init_state("wf-c-large", "customer_ops", "cs-zhao", {"case": case}, token_budget=10_000)
    result = graph.invoke(state)

    analysis = result["skill_outputs"][-1]["output"]
    assert analysis["intent"] == "refund"
    assert analysis["decision"]["action"] == "manual_review"
    assert analysis["estimated_cost"] == 1800.0


def test_wfe_review_monitor_low_rating_needs_action():
    reviews = [
        {"rating": 1, "text": "very bad, smell is terrible"},
        {"rating": 2, "text": "size too small for my baby"},
        {"rating": 5, "text": "great product, love it"},
        {"rating": 3, "text": "shipping was slow"},
        {"rating": 2, "text": "defective bottle leaked"},
        {"rating": 4, "text": "okay value"},
        {"rating": 1, "text": "horrible smell"},
        {"rating": 2, "text": "broken on arrival"},
    ]
    graph = build_customer_ops_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-e-low", "review_monitor", "ops-li", {"reviews": reviews}, token_budget=10_000)
    result = graph.invoke(state)

    output = result["skill_outputs"][-1]["output"]
    assert output["avg_rating"] < 4.0
    assert output["needs_action"] is True
    assert output["themes"]["quality"] >= 2
    assert output["themes"]["smell"] >= 1
    assert result["final_output"]["executed"] is True


def test_wfe_review_monitor_high_rating_no_action():
    reviews = [
        {"rating": 5, "text": "great"},
        {"rating": 5, "text": "love it"},
        {"rating": 4, "text": "good"},
        {"rating": 5, "text": "perfect"},
    ]
    graph = build_customer_ops_graph(interrupt_fn=lambda r: {"action": "approve", "note": ""})
    state = init_state("wf-e-high", "review_monitor", "ops-li", {"reviews": reviews}, token_budget=10_000)
    result = graph.invoke(state)

    output = result["skill_outputs"][-1]["output"]
    assert output["avg_rating"] >= 4.0
    assert output["needs_action"] is False
    assert result["final_output"]["executed"] is True


def run_all():
    tests = [
        test_wfc_chinese_refund_small_auto_approved,
        test_wfc_english_complaint_negative_sentiment,
        test_wfc_large_refund_triggers_manual_review,
        test_wfe_review_monitor_low_rating_needs_action,
        test_wfe_review_monitor_high_rating_no_action,
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
    print(f"\nWF-C/E 端到端集成测试: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())

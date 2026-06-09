"""Smoke test for customer_journey_tree."""
from .model import DialogState, build_return_policy_tree, traverse_tree


def test_food_within_7_days_auto_approve():
    tree = build_return_policy_tree()
    state = DialogState(user_intent="return", days_since_order=3, product_category="food", user_complaint_history=0)
    action = traverse_tree(tree, state)
    assert action.action_type == "auto_approve"


def test_long_overdue_high_complaint_transfers_human():
    tree = build_return_policy_tree()
    state = DialogState(user_intent="return", days_since_order=20, product_category="clothing", user_complaint_history=5)
    action = traverse_tree(tree, state)
    assert action.action_type == "transfer_human"


def test_clothing_within_7_days_requires_photo():
    tree = build_return_policy_tree()
    state = DialogState(user_intent="return", days_since_order=5, product_category="clothing", user_complaint_history=1)
    action = traverse_tree(tree, state)
    assert action.action_type == "require_photo"


if __name__ == "__main__":
    test_food_within_7_days_auto_approve()
    test_long_overdue_high_complaint_transfers_human()
    test_clothing_within_7_days_requires_photo()
    print("OK")

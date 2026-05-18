"""Customer Journey Decision Tree skeleton (synthesis of dialog policy induction work)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class DialogState:
    user_intent: str
    days_since_order: int
    product_category: str
    user_complaint_history: int
    severity: str = "low"


@dataclass
class DialogAction:
    action_type: str
    response_template: str


@dataclass
class DecisionNode:
    feature: Optional[str] = None
    threshold: Optional[float] = None
    categories: Optional[Dict[str, "DecisionNode"]] = None
    left: Optional["DecisionNode"] = None
    right: Optional["DecisionNode"] = None
    leaf_action: Optional[DialogAction] = None

    def is_leaf(self) -> bool:
        return self.leaf_action is not None


def build_return_policy_tree() -> DecisionNode:
    return DecisionNode(
        feature="days_since_order",
        threshold=7,
        left=DecisionNode(
            feature="product_category",
            categories={
                "food": DecisionNode(leaf_action=DialogAction("auto_approve", "您好,食品类 7 天内无理由退货已为您批准,退款将在 24 小时内到账")),
                "clothing": DecisionNode(leaf_action=DialogAction("require_photo", "请提供商品图片以便快速处理")),
                "default": DecisionNode(leaf_action=DialogAction("auto_approve", "已为您批准退货申请")),
            },
        ),
        right=DecisionNode(
            feature="user_complaint_history",
            threshold=3,
            left=DecisionNode(leaf_action=DialogAction("manual_review", "您的申请已提交人工审核,1-2 工作日反馈")),
            right=DecisionNode(leaf_action=DialogAction("transfer_human", "为您转接资深客服处理")),
        ),
    )


def traverse_tree(node: DecisionNode, state: DialogState) -> DialogAction:
    if node.is_leaf():
        return node.leaf_action

    if node.categories is not None:
        val = getattr(state, node.feature, "default")
        next_node = node.categories.get(val, node.categories.get("default"))
        return traverse_tree(next_node, state)

    val = getattr(state, node.feature)
    next_node = node.left if val <= (node.threshold or 0) else node.right
    return traverse_tree(next_node, state)


def llm_enhance(
    action: DialogAction,
    state: DialogState,
    llm_fn: Optional[Callable[[str, DialogState], str]] = None,
) -> str:
    if llm_fn is None:
        return action.response_template
    return llm_fn(action.response_template, state)


def main() -> None:
    tree = build_return_policy_tree()
    test_cases = [
        DialogState(user_intent="return", days_since_order=3, product_category="food", user_complaint_history=0),
        DialogState(user_intent="return", days_since_order=5, product_category="clothing", user_complaint_history=1),
        DialogState(user_intent="return", days_since_order=10, product_category="clothing", user_complaint_history=2),
        DialogState(user_intent="return", days_since_order=20, product_category="clothing", user_complaint_history=5),
    ]
    for i, s in enumerate(test_cases, 1):
        action = traverse_tree(tree, s)
        response = llm_enhance(action, s)
        print(f"[{i}] 状态: {s}")
        print(f"    决策: {action.action_type} → '{response}'")
        print()


if __name__ == "__main__":
    main()

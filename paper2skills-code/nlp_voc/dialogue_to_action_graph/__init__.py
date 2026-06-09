"""
Dialogue-to-Action Graph Parser

基于 TOD-Flow 的图条件对话建模思想，将客服对话解析为决策动作图。

Usage:
    from dialogue_to_action_graph import DialogueToActionGraphParser
    parser = DialogueToActionGraphParser()
    graph = parser.parse("User: My pump is broken.\nAgent: I'll help you...")
    print(graph.to_dict())
"""

from .model import (
    DialogueToActionGraphParser,
    DecisionGraph,
    ActionNode,
    ActionEdge,
    NodeType,
    RelationType,
    print_decision_graph,
    analyze_graph_patterns,
)

__all__ = [
    "DialogueToActionGraphParser",
    "DecisionGraph",
    "ActionNode",
    "ActionEdge",
    "NodeType",
    "RelationType",
    "print_decision_graph",
    "analyze_graph_patterns",
]

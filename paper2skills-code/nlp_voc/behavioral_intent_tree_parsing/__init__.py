"""
Behavioral Intent Tree Parser

基于 IntentRec 的层次化多任务学习思想，将用户行为序列解析为层次化意图结构树。

Usage:
    from behavioral_intent_tree_parsing import BehavioralIntentTreeParser, BehaviorEvent
    parser = BehavioralIntentTreeParser()
    events = [BehaviorEvent("2024-01-01", BehaviorType.CLICK)]
    tree = parser.parse("user_001", events)
    print(tree.to_dict())
"""

from .model import (
    BehavioralIntentTreeParser,
    IntentTree,
    IntentNode,
    BehaviorEvent,
    BehaviorType,
    IntentType,
    print_intent_tree,
    analyze_intent_distribution,
)

__all__ = [
    "BehavioralIntentTreeParser",
    "IntentTree",
    "IntentNode",
    "BehaviorEvent",
    "BehaviorType",
    "IntentType",
    "print_intent_tree",
    "analyze_intent_distribution",
]

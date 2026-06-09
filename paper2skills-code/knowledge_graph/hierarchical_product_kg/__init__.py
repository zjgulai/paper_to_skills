"""Hierarchical Product KG Construction package.

Skeleton implementation of arXiv:2410.21237 — paper has no public code.
Production use should swap mock VLM/LLM for InternVL2-8B + Llama3.1-70B.
"""

from .model import (
    BABY_ECOM_SCHEMA,
    ProductKGNode,
    build_product_kg_node,
    evaluate_attribute_accuracy,
    hierarchical_expand,
    prune_graph,
    regex_constrained_json,
)

__all__ = [
    "BABY_ECOM_SCHEMA",
    "ProductKGNode",
    "build_product_kg_node",
    "evaluate_attribute_accuracy",
    "hierarchical_expand",
    "prune_graph",
    "regex_constrained_json",
]

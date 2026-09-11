"""
Cross-lingual Semantic Alignment Parser

基于 Cross-lingual AMR Aligner 的 cross-attention 思想，
将多语言产品描述解析为统一的语义结构。

Usage:
    from crosslingual_semantic_alignment import CrossLingualSemanticAligner
    aligner = CrossLingualSemanticAligner()
    graph = aligner.align({"en": "breast pump...", "zh": "吸奶器..."})
    print(graph.to_dict())
"""

from .model import (
    CrossLingualSemanticAligner,
    UnifiedSemanticGraph,
    SemanticNode,
    SemanticEdge,
    Language,
    print_semantic_graph,
    compare_language_coverage,
)

__all__ = [
    "CrossLingualSemanticAligner",
    "UnifiedSemanticGraph",
    "SemanticNode",
    "SemanticEdge",
    "Language",
    "print_semantic_graph",
    "compare_language_coverage",
]

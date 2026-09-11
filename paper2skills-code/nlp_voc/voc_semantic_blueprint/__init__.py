"""
VOC Semantic Blueprint Extractor

基于 USSA (Zhai et al., ACL 2023) Table-Filling 思想，
将用户评论转换为结构化语义蓝图。

Usage:
    from voc_semantic_blueprint import VOCBlueprintExtractor
    extractor = VOCBlueprintExtractor()
    blueprint = extractor.extract("The suction is strong but noisy.")
    print(blueprint.to_dict())
"""

from .model import (
    VOCBlueprintExtractor,
    VOCBlueprint,
    VOCBlueprintNode,
    RelationTable,
    TableCell,
    print_blueprint,
)

__all__ = [
    "VOCBlueprintExtractor",
    "VOCBlueprint",
    "VOCBlueprintNode",
    "RelationTable",
    "TableCell",
    "print_blueprint",
]

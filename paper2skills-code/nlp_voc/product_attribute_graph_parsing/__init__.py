"""
Product Attribute Graph Parser

基于 Schema-Guided Generation 思想，将电商产品描述解析为层次化属性图谱。

Usage:
    from product_attribute_graph_parsing import ProductAttributeGraphParser
    parser = ProductAttributeGraphParser()
    graph = parser.parse("Momcozy S12", "Ultra-quiet breast pump with 9 suction levels...")
    print(graph.to_dict())
"""

from .model import (
    ProductAttributeGraphParser,
    ProductAttributeGraph,
    AttributeNode,
    SchemaProperty,
    BREAST_PUMP_SCHEMA,
    DIAPER_SCHEMA,
    print_attribute_graph,
    compare_graphs,
)

__all__ = [
    "ProductAttributeGraphParser",
    "ProductAttributeGraph",
    "AttributeNode",
    "SchemaProperty",
    "BREAST_PUMP_SCHEMA",
    "DIAPER_SCHEMA",
    "print_attribute_graph",
    "compare_graphs",
]

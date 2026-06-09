"""Smoke test for hierarchical_product_kg."""
from .model import (
    BABY_ECOM_SCHEMA,
    build_product_kg_node,
    evaluate_attribute_accuracy,
    hierarchical_expand,
    prune_graph,
    regex_constrained_json,
)


def test_hierarchical_expand_chains_to_root():
    chain = hierarchical_expand("Aptamil Stage 1", "Infant Formula")
    assert chain[0] == "Aptamil Stage 1"
    assert "Infant Formula" in chain
    assert "Baby Food" in chain


def test_prune_graph_dedupes_case_and_order():
    nodes = ["Infant Formula", "FORMULA INFANT", "Baby Food"]
    pruned = prune_graph(nodes)
    assert len(pruned) == 2


def test_regex_constrained_json_extracts_valid():
    raw = 'leading text {"category": "Pacifier", "weight_kg": 0.05} trailing'
    parsed = regex_constrained_json(raw)
    assert parsed["category"] == "Pacifier"
    assert parsed["weight_kg"] == 0.05


def test_build_product_kg_node_formula():
    node = build_product_kg_node("/data/products/aptamil_organic.jpg")
    assert node.properties["category"] == "Infant Formula"
    assert "Baby Food" in node.category_hierarchy


def test_evaluate_attribute_accuracy_basic():
    preds = [{"category": "Pacifier", "weight_kg": 0.05}]
    gts = [{"category": "Pacifier", "weight_kg": 0.05}]
    m = evaluate_attribute_accuracy(preds, gts)
    assert m["category_acc"] == 1.0
    assert m["weight_acc"] == 1.0


if __name__ == "__main__":
    test_hierarchical_expand_chains_to_root()
    test_prune_graph_dedupes_case_and_order()
    test_regex_constrained_json_extracts_valid()
    test_build_product_kg_node_formula()
    test_evaluate_attribute_accuracy_basic()
    print("OK")

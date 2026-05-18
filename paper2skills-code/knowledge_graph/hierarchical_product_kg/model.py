"""Hierarchical Product KG Construction — arXiv:2410.21237 minimal skeleton.

Replace `_mock_vlm_extract` and `_mock_llm_infer` with real VLM/LLM calls
(InternVL2-8B + Llama3.1-70B per the paper) in production.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


BABY_ECOM_SCHEMA: Dict[str, object] = {
    "product_name": "str",
    "category": {
        "type": "choices",
        "options": [
            "Infant Formula", "Baby Food", "Baby Bottle", "Pacifier",
            "Diaper", "Baby Clothing", "Stroller", "Baby Carrier",
            "Baby Wipes", "Baby Skincare", "Others",
        ],
    },
    "brand": "str",
    "primary_color": {
        "type": "choices",
        "options": ["White", "Pink", "Blue", "Green", "Yellow", "Purple", "Others"],
    },
    "package_material": {
        "type": "choices",
        "options": ["Plastic", "Metal", "Cardboard", "Glass", "Fabric", "Others"],
    },
    "weight_kg": "float",
    "age_range": {
        "type": "choices",
        "options": ["0-6m", "6-12m", "1-3y", "3-6y", "All ages"],
    },
}


@dataclass
class ProductKGNode:
    properties: Dict[str, object]
    category_hierarchy: List[str]


def _mock_vlm_extract(image_path: str, schema: Dict[str, object]) -> str:
    desc_parts = []
    fname = image_path.lower()
    if "aptamil" in fname or "formula" in fname:
        desc_parts.append("infant formula in metallic cylindrical can")
    if "pacifier" in fname or "dot" in fname:
        desc_parts.append("silicone pacifier in pink/blue plastic packaging")
    if "diaper" in fname:
        desc_parts.append("disposable baby diaper in cardboard box")
    return " | ".join(desc_parts) or "generic baby product packaging"


def _mock_llm_infer(description: str, schema: Dict[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {"product_name": "Auto-Detected Product"}
    desc = description.lower()
    if "formula" in desc:
        output["category"] = "Infant Formula"
        output["package_material"] = "Metal"
        output["weight_kg"] = 0.8
        output["age_range"] = "0-6m"
    elif "pacifier" in desc:
        output["category"] = "Pacifier"
        output["package_material"] = "Plastic"
        output["primary_color"] = "Pink"
        output["age_range"] = "0-6m"
        output["weight_kg"] = 0.05
    elif "diaper" in desc:
        output["category"] = "Diaper"
        output["package_material"] = "Cardboard"
        output["weight_kg"] = 1.5
    else:
        output["category"] = "Others"
        output["weight_kg"] = 0.5
    return output


def regex_constrained_json(raw_text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def hierarchical_expand(
    product_name: str,
    leaf_category: str,
    parent_lookup: Optional[Callable[[str], str]] = None,
    max_levels: int = 4,
) -> List[str]:
    default_taxonomy = {
        "Infant Formula": "Baby Food",
        "Baby Food": "Baby Products",
        "Baby Products": "Mother & Baby",
        "Pacifier": "Baby Feeding",
        "Baby Feeding": "Baby Products",
        "Diaper": "Baby Care",
        "Baby Care": "Baby Products",
    }
    parent_lookup = parent_lookup or (lambda x: default_taxonomy.get(x, ""))

    chain: List[str] = [product_name, leaf_category]
    current = leaf_category
    for _ in range(max_levels):
        parent = parent_lookup(current)
        if not parent or parent == current:
            break
        chain.append(parent)
        current = parent
    return chain


def prune_graph(nodes: List[str]) -> List[str]:
    seen = set()
    pruned: List[str] = []
    for node in nodes:
        key = frozenset(node.lower().split())
        if key not in seen:
            seen.add(key)
            pruned.append(node)
    return pruned


def build_product_kg_node(
    image_path: str,
    schema: Optional[Dict[str, object]] = None,
    vlm_extract: Optional[Callable[[str, Dict[str, object]], str]] = None,
    llm_infer: Optional[Callable[[str, Dict[str, object]], Dict[str, object]]] = None,
) -> ProductKGNode:
    schema = schema or BABY_ECOM_SCHEMA
    vlm_extract = vlm_extract or _mock_vlm_extract
    llm_infer = llm_infer or _mock_llm_infer

    description = vlm_extract(image_path, schema)
    properties = llm_infer(description, schema)

    category = str(properties.get("category", "Others"))
    product_name = str(properties.get("product_name", image_path))
    hierarchy = hierarchical_expand(product_name, category)
    properties["category_hierarchy"] = prune_graph(hierarchy)

    return ProductKGNode(
        properties=properties,
        category_hierarchy=properties["category_hierarchy"],
    )


def evaluate_attribute_accuracy(
    predictions: List[Dict[str, object]],
    ground_truths: List[Dict[str, object]],
    threshold: float = 0.05,
) -> Dict[str, float]:
    metrics = {"category_acc": 0.0, "weight_acc": 0.0}
    n = len(predictions)
    if n == 0:
        return metrics

    cat_hits = sum(
        1 for p, g in zip(predictions, ground_truths)
        if p.get("category") == g.get("category")
    )

    weight_hits = 0
    weight_total = 0
    for p, g in zip(predictions, ground_truths):
        v_p = p.get("weight_kg")
        v_g = g.get("weight_kg")
        if isinstance(v_p, (int, float)) and isinstance(v_g, (int, float)) and v_g != 0:
            weight_total += 1
            err = abs(v_p - v_g) / abs(v_g)
            if err <= threshold:
                weight_hits += 1

    metrics["category_acc"] = cat_hits / n
    metrics["weight_acc"] = weight_hits / weight_total if weight_total else 0.0
    return metrics


def main() -> None:
    print("=" * 60)
    print("Hierarchical Product KG Construction — Demo")
    print("=" * 60)

    sample_skus = [
        "/data/products/aptamil_organic_stage1.jpg",
        "/data/products/pigeon_silicone_pacifier_pink.jpg",
        "/data/products/huggies_diaper_size3.jpg",
    ]

    nodes = [build_product_kg_node(p) for p in sample_skus]
    for sku, node in zip(sample_skus, nodes):
        print(f"\nSKU: {sku}")
        print(json.dumps(node.properties, indent=2, ensure_ascii=False))

    ground_truths = [
        {"category": "Infant Formula", "weight_kg": 0.85},
        {"category": "Pacifier", "weight_kg": 0.05},
        {"category": "Diaper", "weight_kg": 1.4},
    ]
    predictions = [n.properties for n in nodes]
    metrics = evaluate_attribute_accuracy(predictions, ground_truths)
    print("\n" + "=" * 60)
    print(f"Category Acc: {metrics['category_acc']*100:.2f}%")
    print(f"Weight Acc@0.05: {metrics['weight_acc']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

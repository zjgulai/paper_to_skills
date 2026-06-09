"""
Data Provenance & Lineage - Tracing Roots + DEBUGLM
arXiv: 2604.10480 / 2603.17884
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class DatasetNode:
    dataset_id: str
    name: str
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)
    provenance_tag: str = ""

    def __post_init__(self):
        if not self.provenance_tag:
            self.provenance_tag = hashlib.md5(
                f"{self.dataset_id}:{self.name}:{self.version}".encode()
            ).hexdigest()[:12]


@dataclass
class LineageEdge:
    source_id: str
    target_id: str
    relation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLineageGraph:
    def __init__(self):
        self.nodes: Dict[str, DatasetNode] = {}
        self.edges: List[LineageEdge] = []

    def add_dataset(self, dataset: DatasetNode):
        self.nodes[dataset.dataset_id] = dataset

    def add_lineage(self, source_id: str, target_id: str, relation: str = "derived_from"):
        self.edges.append(LineageEdge(source_id, target_id, relation))

    def get_ancestors(self, dataset_id: str, depth: int = 5) -> List[str]:
        ancestors, visited, queue = [], {dataset_id}, [dataset_id]
        current_depth = 0
        while queue and current_depth < depth:
            next_queue = []
            for node in queue:
                for edge in self.edges:
                    if edge.target_id == node and edge.source_id not in visited:
                        visited.add(edge.source_id)
                        ancestors.append(edge.source_id)
                        next_queue.append(edge.source_id)
            queue = next_queue
            current_depth += 1
        return ancestors

    def get_descendants(self, dataset_id: str) -> List[str]:
        descendants, visited, queue = [], {dataset_id}, [dataset_id]
        while queue:
            node = queue.pop(0)
            for edge in self.edges:
                if edge.source_id == node and edge.target_id not in visited:
                    visited.add(edge.target_id)
                    descendants.append(edge.target_id)
                    queue.append(edge.target_id)
        return descendants

    def detect_contamination_paths(self, contaminated_id: str) -> List[str]:
        return self.get_descendants(contaminated_id)

    def stats(self) -> Dict[str, Any]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}


class ProvenanceTagger:
    """DEBUGLM: 嵌入可追踪的数据来源标签，不重训练即可溯源"""

    def __init__(self):
        self.tag_registry: Dict[str, str] = {}

    def tag_dataset(self, dataset: DatasetNode) -> str:
        self.tag_registry[dataset.provenance_tag] = dataset.dataset_id
        return dataset.provenance_tag

    def resolve_tag(self, tag: str) -> Optional[str]:
        return self.tag_registry.get(tag)

    def inject_tag_into_sample(self, sample_text: str, tag: str) -> str:
        return f"[PROVENANCE:{tag}] {sample_text}"

    def extract_tag_from_sample(self, tagged_text: str) -> Optional[str]:
        if tagged_text.startswith("[PROVENANCE:"):
            end = tagged_text.index("]")
            return tagged_text[12:end]
        return None


def test_lineage_graph_construction():
    graph = DataLineageGraph()
    amazon_reviews = DatasetNode("ds_amazon", "Amazon Reviews 2024")
    cleaned = DatasetNode("ds_cleaned", "Cleaned Reviews")
    model_training = DatasetNode("ds_train", "Training Set")
    for ds in [amazon_reviews, cleaned, model_training]:
        graph.add_dataset(ds)
    graph.add_lineage("ds_amazon", "ds_cleaned", "derived_from")
    graph.add_lineage("ds_cleaned", "ds_train", "derived_from")
    ancestors = graph.get_ancestors("ds_train")
    assert "ds_amazon" in ancestors
    assert "ds_cleaned" in ancestors
    print(f"[PASS] lineage_graph: ancestors of ds_train = {ancestors}")


def test_contamination_detection():
    graph = DataLineageGraph()
    for ds_id in ["ds_raw", "ds_intermediate", "ds_model_a", "ds_model_b"]:
        graph.add_dataset(DatasetNode(ds_id, ds_id))
    graph.add_lineage("ds_raw", "ds_intermediate")
    graph.add_lineage("ds_intermediate", "ds_model_a")
    graph.add_lineage("ds_intermediate", "ds_model_b")
    contaminated = graph.detect_contamination_paths("ds_raw")
    assert "ds_intermediate" in contaminated
    assert "ds_model_a" in contaminated
    assert "ds_model_b" in contaminated
    print(f"[PASS] contamination: spreading from ds_raw to {len(contaminated)} downstream datasets")


def test_provenance_tagging():
    tagger = ProvenanceTagger()
    dataset = DatasetNode("ds_001", "Baby Product Reviews")
    tag = tagger.tag_dataset(dataset)
    sample = "Great breast pump, very quiet and portable"
    tagged = tagger.inject_tag_into_sample(sample, tag)
    extracted = tagger.extract_tag_from_sample(tagged)
    assert extracted == tag
    resolved = tagger.resolve_tag(extracted)
    assert resolved == "ds_001"
    print(f"[PASS] provenance_tag: tag={tag[:8]}..., resolved={resolved}")


def test_full_lineage_pipeline():
    graph = DataLineageGraph()
    tagger = ProvenanceTagger()
    datasets = [
        DatasetNode("ds_amazon", "Amazon Baby Reviews"),
        DatasetNode("ds_tiktok", "TikTok Baby Reviews"),
        DatasetNode("ds_merged", "Merged Reviews"),
        DatasetNode("ds_deduped", "Deduped Reviews"),
        DatasetNode("ds_model", "VOC Model Training Data"),
    ]
    for ds in datasets:
        graph.add_dataset(ds)
        tagger.tag_dataset(ds)
    graph.add_lineage("ds_amazon", "ds_merged")
    graph.add_lineage("ds_tiktok", "ds_merged")
    graph.add_lineage("ds_merged", "ds_deduped")
    graph.add_lineage("ds_deduped", "ds_model")
    ancestors = graph.get_ancestors("ds_model")
    assert "ds_amazon" in ancestors and "ds_tiktok" in ancestors
    stats = graph.stats()
    print(f"[PASS] full_lineage: {stats}, model ancestors={len(ancestors)}")


if __name__ == "__main__":
    test_lineage_graph_construction()
    test_contamination_detection()
    test_provenance_tagging()
    test_full_lineage_pipeline()
    print("\n✅ All tests passed")

"""Taxonomy 构建器

管理种子 Taxonomy 的生成和多维 Taxonomy 的组织。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TaxonomyNode:
    """Taxonomy 节点"""

    id: str
    name: str
    description: str = ""
    parent_id: Optional[str] = None
    level: int = 1
    children_ids: list[str] = field(default_factory=list)
    # 统计信息
    text_count: int = 0          # 该节点覆盖的文本数
    coverage: float = 0.0        # 覆盖率
    last_expanded: str = ""      # 上次扩展时间

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parent_id": self.parent_id,
            "level": self.level,
            "children_ids": self.children_ids,
            "text_count": self.text_count,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaxonomyNode:
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            parent_id=d.get("parent_id"),
            level=d.get("level", 1),
            children_ids=d.get("children_ids", []),
            text_count=d.get("text_count", 0),
            coverage=d.get("coverage", 0.0),
        )


class TaxonomyTree:
    """Taxonomy 树

    管理单维度层级标签的树状结构。
    """

    def __init__(self, dimension_name: str = "default"):
        self.dimension_name = dimension_name
        self.nodes: dict[str, TaxonomyNode] = {}
        self.root_ids: list[str] = []  # L1 节点ID

    def add_node(self, node: TaxonomyNode) -> None:
        """添加节点"""
        if node.id in self.nodes:
            raise ValueError(f"节点ID已存在: {node.id}")
        self.nodes[node.id] = node
        if node.level == 1:
            self.root_ids.append(node.id)
        # 更新父节点的 children_ids
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children_ids:
                parent.children_ids.append(node.id)

    def get_node(self, node_id: str) -> TaxonomyNode:
        """获取节点"""
        if node_id not in self.nodes:
            raise ValueError(f"节点不存在: {node_id}")
        return self.nodes[node_id]

    def get_children(self, node_id: str) -> list[TaxonomyNode]:
        """获取子节点"""
        node = self.get_node(node_id)
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_path(self, node_id: str) -> list[TaxonomyNode]:
        """获取从根到该节点的路径"""
        path = []
        current_id: Optional[str] = node_id
        while current_id:
            node = self.nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        return list(reversed(path))

    def get_leaves(self) -> list[TaxonomyNode]:
        """获取所有叶子节点"""
        has_children = set()
        for node in self.nodes.values():
            if node.children_ids:
                has_children.update(node.children_ids)
        return [n for nid, n in self.nodes.items() if nid not in has_children]

    def get_nodes_at_level(self, level: int) -> list[TaxonomyNode]:
        """获取某一层的所有节点"""
        return [n for n in self.nodes.values() if n.level == level]

    def update_stats(self, node_id: str, text_count: int) -> None:
        """更新节点统计信息"""
        node = self.get_node(node_id)
        node.text_count = text_count
        # 递归更新父节点
        if node.parent_id:
            parent = self.get_node(node.parent_id)
            parent_children = self.get_children(node.parent_id)
            parent.text_count = sum(c.text_count for c in parent_children)

    def to_dict(self) -> dict:
        return {
            "dimension_name": self.dimension_name,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaxonomyTree:
        tree = cls(data.get("dimension_name", "default"))
        for nid, node_dict in data.get("nodes", {}).items():
            tree.nodes[nid] = TaxonomyNode.from_dict(node_dict)
        tree.root_ids = [nid for nid, n in tree.nodes.items() if n.level == 1]
        return tree

    def to_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> TaxonomyTree:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        counts = {}
        for node in self.nodes.values():
            counts[node.level] = counts.get(node.level, 0) + 1
        return f"TaxonomyTree({self.dimension_name}, nodes={len(self.nodes)}, levels={counts})"


class MultidimensionalTaxonomy:
    """多维 Taxonomy

    管理多个维度的 Taxonomy，支持交叉分析。
    """

    def __init__(self):
        self.dimensions: dict[str, TaxonomyTree] = {}

    def add_dimension(self, name: str, tree: TaxonomyTree) -> None:
        """添加维度"""
        self.dimensions[name] = tree

    def get_dimension(self, name: str) -> TaxonomyTree:
        """获取维度"""
        if name not in self.dimensions:
            raise ValueError(f"维度不存在: {name}")
        return self.dimensions[name]

    def get_all_paths(self, text_labels: dict[str, str]) -> dict[str, list[str]]:
        """获取一条文本在所有维度上的完整路径

        Args:
            text_labels: {维度名: 叶子节点ID}

        Returns:
            {维度名: [L1名, L2名, ...]}
        """
        result = {}
        for dim_name, node_id in text_labels.items():
            tree = self.dimensions[dim_name]
            path = tree.get_path(node_id)
            result[dim_name] = [n.name for n in path]
        return result

    def summary(self) -> dict:
        """多维 Taxonomy 统计"""
        return {
            "n_dimensions": len(self.dimensions),
            "dimensions": {
                name: {
                    "total_nodes": len(tree.nodes),
                    "max_depth": max((n.level for n in tree.nodes.values()), default=0),
                }
                for name, tree in self.dimensions.items()
            },
        }


# ── 预设种子 Taxonomy（母婴出海）──────────────────────────────

def create_mombaby_seed_taxonomy() -> TaxonomyTree:
    """创建母婴出海场景的种子 Taxonomy"""
    tree = TaxonomyTree(dimension_name="产品品类-问题域")

    # L1: 品类
    tree.add_node(TaxonomyNode("L1-01", "纸尿裤", " diaper 类产品", level=1))
    tree.add_node(TaxonomyNode("L1-02", "奶粉", "婴幼儿奶粉", level=1))
    tree.add_node(TaxonomyNode("L1-03", "防蚊产品", "驱蚊、防蚊相关产品", level=1))

    # L2: 问题域
    tree.add_node(TaxonomyNode("L2-01", "质量", "产品质量相关问题", parent_id="L1-01", level=2))
    tree.add_node(TaxonomyNode("L2-02", "物流", "物流配送相关问题", parent_id="L1-01", level=2))
    tree.add_node(TaxonomyNode("L2-03", "质量", "奶粉质量问题", parent_id="L1-02", level=2))
    tree.add_node(TaxonomyNode("L2-04", "质量", "防蚊产品质量问题", parent_id="L1-03", level=2))
    tree.add_node(TaxonomyNode("L2-05", "效果", "防蚊效果相关问题", parent_id="L1-03", level=2))

    # L3: 细分类别
    tree.add_node(TaxonomyNode("L3-01", "漏尿问题", "漏尿相关", parent_id="L2-01", level=3))
    tree.add_node(TaxonomyNode("L3-02", "材质舒适度", "材质相关", parent_id="L2-01", level=3))
    tree.add_node(TaxonomyNode("L3-03", "尺码偏差", "尺码不合适", parent_id="L2-01", level=3))
    tree.add_node(TaxonomyNode("L3-04", "配送时效", "配送速度", parent_id="L2-02", level=3))
    tree.add_node(TaxonomyNode("L3-05", "溶解性", "奶粉溶解问题", parent_id="L2-03", level=3))
    tree.add_node(TaxonomyNode("L3-06", "驱蚊时长", "驱蚊持续时间", parent_id="L2-05", level=3))
    tree.add_node(TaxonomyNode("L3-07", "粘性持久度", "防蚊贴粘性", parent_id="L2-04", level=3))

    return tree


# ── 测试 ──────────────────────────────────────────────────────

def test_taxonomy():
    print("=" * 60)
    print("测试: TaxonomyBuilder")
    print("=" * 60)

    tree = create_mombaby_seed_taxonomy()
    print(f"\n创建 Taxonomy: {tree}")

    # 测试路径查询
    print("\n--- 路径查询 ---")
    path = tree.get_path("L3-01")
    print(f"L3-01(漏尿问题) 路径: {' → '.join(n.name for n in path)}")

    # 测试子节点
    print("\n--- 子节点查询 ---")
    children = tree.get_children("L1-01")
    print(f"L1-01(纸尿裤) 的 L2: {[c.name for c in children]}")

    # 测试叶子节点
    print("\n--- 叶子节点 ---")
    leaves = tree.get_leaves()
    print(f"叶子标签: {[n.name for n in leaves]}")

    # 测试序列化
    print("\n--- 序列化 ---")
    tree.to_json("/tmp/taxonomy_demo.json")
    restored = TaxonomyTree.from_json("/tmp/taxonomy_demo.json")
    print(f"反序列化后: {restored}")
    assert len(restored.nodes) == len(tree.nodes)
    print("✓ 序列化/反序列化测试通过")

    # 多维 Taxonomy
    print("\n--- 多维 Taxonomy ---")
    multi = MultidimensionalTaxonomy()
    multi.add_dimension("产品品类", tree)

    # 创建第二个维度
    emotion_tree = TaxonomyTree(dimension_name="情感强度")
    emotion_tree.add_node(TaxonomyNode("E1-01", "负面", level=1))
    emotion_tree.add_node(TaxonomyNode("E1-02", "中性", level=1))
    emotion_tree.add_node(TaxonomyNode("E1-03", "正面", level=1))
    emotion_tree.add_node(TaxonomyNode("E2-01", "强烈不满", parent_id="E1-01", level=2))
    emotion_tree.add_node(TaxonomyNode("E2-02", "轻微不满", parent_id="E1-01", level=2))
    emotion_tree.add_node(TaxonomyNode("E2-03", "非常满意", parent_id="E1-03", level=2))
    multi.add_dimension("情感强度", emotion_tree)

    print(multi.summary())

    # 测试交叉路径
    text_labels = {"产品品类": "L3-01", "情感强度": "E2-01"}
    paths = multi.get_all_paths(text_labels)
    print(f"\n交叉路径: {paths}")

    print("\n" + "=" * 60)
    print("Taxonomy 构建器测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_taxonomy()

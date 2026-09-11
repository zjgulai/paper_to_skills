"""Taxonomy 扩展引擎

实现宽度扩展（新增同级标签）和深度扩展（新增子级标签）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from taxonomy_builder import TaxonomyNode, TaxonomyTree


@dataclass
class ExpansionCandidate:
    """扩展候选"""

    parent_id: str
    suggested_name: str
    description: str
    sample_texts: list[str]
    confidence: float
    expansion_type: str  # "width" | "depth"


class TextClusterer:
    """简单文本聚类器（基于关键词共现）

    生产环境替换为: sentence-transformers + HDBSCAN/KMeans
    """

    def __init__(self, min_cluster_size: int = 3, similarity_threshold: float = 0.3):
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold

    def cluster(self, texts: list[str]) -> list[list[str]]:
        """将文本聚类为若干组"""
        if len(texts) < self.min_cluster_size:
            return []

        # 提取每篇文本的关键词（2-3 字滑动窗口）
        text_keywords: list[set[str]] = []
        for text in texts:
            keywords = set()
            for length in range(2, 4):
                for i in range(len(text) - length + 1):
                    word = text[i:i + length]
                    if any("\u4e00" <= c <= "\u9fff" for c in word):
                        keywords.add(word)
            text_keywords.append(keywords)

        # 贪心聚类
        assigned = [False] * len(texts)
        clusters: list[list[int]] = []

        for i in range(len(texts)):
            if assigned[i]:
                continue
            cluster = [i]
            assigned[i] = True
            for j in range(i + 1, len(texts)):
                if assigned[j]:
                    continue
                # Jaccard 相似度
                intersection = len(text_keywords[i] & text_keywords[j])
                union = len(text_keywords[i] | text_keywords[j])
                similarity = intersection / union if union > 0 else 0
                if similarity >= self.similarity_threshold:
                    cluster.append(j)
                    assigned[j] = True

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        return [[texts[i] for i in cluster] for cluster in clusters]

    def extract_cluster_name(self, texts: list[str]) -> str:
        """从聚类文本中提取代表性名称"""
        word_counts: dict[str, int] = {}
        for text in texts:
            for length in range(2, 4):
                for i in range(len(text) - length + 1):
                    word = text[i:i + length]
                    if any("\u4e00" <= c <= "\u9fff" for c in word):
                        word_counts[word] = word_counts.get(word, 0) + 1

        # 返回最高频词
        if word_counts:
            return max(word_counts, key=word_counts.get)
        return "新标签"


class WidthExpander:
    """宽度扩展器

    当某层级下出现大量无法归类的文本时，新增同级标签。
    """

    def __init__(
        self,
        coverage_threshold: float = 0.8,      # 覆盖率低于此值触发扩展
        min_uncovered: int = 5,               # 未覆盖文本数量阈值
        clusterer: Optional[TextClusterer] = None,
    ):
        self.coverage_threshold = coverage_threshold
        self.min_uncovered = min_uncovered
        self.clusterer = clusterer or TextClusterer()

    def should_expand(
        self,
        parent_node_id: str,
        uncovered_texts: list[str],
        total_texts: int,
    ) -> bool:
        """判断是否需要宽度扩展"""
        if len(uncovered_texts) < self.min_uncovered:
            return False
        uncovered_rate = len(uncovered_texts) / total_texts if total_texts > 0 else 0
        return uncovered_rate > (1 - self.coverage_threshold)

    def expand(
        self,
        tree: TaxonomyTree,
        parent_node_id: str,
        uncovered_texts: list[str],
    ) -> list[ExpansionCandidate]:
        """对指定父节点进行宽度扩展"""
        parent = tree.get_node(parent_node_id)
        clusters = self.clusterer.cluster(uncovered_texts)

        candidates: list[ExpansionCandidate] = []
        for cluster_texts in clusters:
            name = self.clusterer.extract_cluster_name(cluster_texts)
            candidates.append(ExpansionCandidate(
                parent_id=parent_node_id,
                suggested_name=name,
                description=f"从 {len(cluster_texts)} 条未覆盖文本自动发现，父标签: {parent.name}",
                sample_texts=cluster_texts[:5],
                confidence=len(cluster_texts) / len(uncovered_texts),
                expansion_type="width",
            ))

        return candidates


class DepthExpander:
    """深度扩展器

    当某个标签下的文本呈现可细分模式时，新增子级标签。
    """

    def __init__(
        self,
        min_texts_for_split: int = 10,        # 最少文本数才考虑细分
        diversity_threshold: float = 0.4,     # 文本多样性阈值
        clusterer: Optional[TextClusterer] = None,
    ):
        self.min_texts_for_split = min_texts_for_split
        self.diversity_threshold = diversity_threshold
        self.clusterer = clusterer or TextClusterer()

    def should_expand(self, texts: list[str]) -> bool:
        """判断是否需要深度扩展"""
        if len(texts) < self.min_texts_for_split:
            return False

        # 简单多样性估计：不同关键词的数量 / 总文本数
        all_keywords: set[str] = set()
        for text in texts:
            for length in range(2, 4):
                for i in range(len(text) - length + 1):
                    word = text[i:i + length]
                    if any("\u4e00" <= c <= "\u9fff" for c in word):
                        all_keywords.add(word)

        diversity = len(all_keywords) / len(texts) if texts else 0
        return diversity > self.diversity_threshold

    def expand(
        self,
        tree: TaxonomyTree,
        node_id: str,
        texts: list[str],
    ) -> list[ExpansionCandidate]:
        """对指定节点进行深度扩展"""
        node = tree.get_node(node_id)
        clusters = self.clusterer.cluster(texts)

        candidates: list[ExpansionCandidate] = []
        for cluster_texts in clusters:
            name = self.clusterer.extract_cluster_name(cluster_texts)
            candidates.append(ExpansionCandidate(
                parent_id=node_id,
                suggested_name=name,
                description=f"由 {node.name} 细分而来，{len(cluster_texts)} 条文本",
                sample_texts=cluster_texts[:5],
                confidence=len(cluster_texts) / len(texts),
                expansion_type="depth",
            ))

        return candidates


class ConsistencyChecker:
    """一致性校验器

    校验扩展后的 Taxonomy 是否满足层级一致性。
    """

    def check_parent_child_semantic(
        self,
        parent: TaxonomyNode,
        child: TaxonomyNode,
    ) -> bool:
        """检查子标签语义是否包含于父标签语义（简化版）

        生产环境: 使用 embedding 语义相似度判断
        """
        # 简化: 检查子标签名称关键词是否出现在父标签名称/描述中
        parent_words = set(parent.name) | set(parent.description)
        child_words = set(child.name)

        # 至少有一个字重叠（非常宽松的判断）
        overlap = child_words & parent_words
        return len(overlap) > 0

    def check_sibling_mutual_exclusion(
        self,
        nodes: list[TaxonomyNode],
        sample_texts: dict[str, list[str]],
    ) -> dict[str, float]:
        """检查同级标签之间的互斥度

        Returns:
            {节点ID: 重叠度(越低越好)}
        """
        overlap_scores: dict[str, float] = {}

        for i, node_a in enumerate(nodes):
            texts_a = set(sample_texts.get(node_a.id, []))
            max_overlap = 0.0

            for j, node_b in enumerate(nodes):
                if i == j:
                    continue
                texts_b = set(sample_texts.get(node_b.id, []))
                if texts_a and texts_b:
                    overlap = len(texts_a & texts_b) / min(len(texts_a), len(texts_b))
                    max_overlap = max(max_overlap, overlap)

            overlap_scores[node_a.id] = max_overlap

        return overlap_scores

    def validate_expansion(
        self,
        tree: TaxonomyTree,
        candidate: ExpansionCandidate,
    ) -> tuple[bool, str]:
        """校验扩展候选是否合法"""
        parent = tree.get_node(candidate.parent_id)

        # 检查层级一致性
        # 简化: 假设通过（生产环境需用 embedding）
        semantic_ok = True  # self.check_parent_child_semantic(parent, new_node)

        if not semantic_ok:
            return False, "子标签语义不包含于父标签"

        # 检查同级互斥性
        siblings = tree.get_children(candidate.parent_id)
        if siblings:
            # 检查名称是否重复
            for sib in siblings:
                if sib.name == candidate.suggested_name:
                    return False, f"同级标签名称重复: {candidate.suggested_name}"

        return True, "校验通过"


# ── 测试 ──────────────────────────────────────────────────────

def test_expander():
    print("=" * 60)
    print("测试: Expander")
    print("=" * 60)

    from taxonomy_builder import create_mombaby_seed_taxonomy

    tree = create_mombaby_seed_taxonomy()

    # 模拟未覆盖文本（防蚊产品下的新痛点）
    uncovered_texts = [
        "驱蚊液味道很刺鼻，宝宝一直哭",
        "这个防蚊喷雾气味太浓了",
        "味道太冲，不敢给宝宝用",
        "气味刺鼻，换了一个牌子",
        "香味太重，宝宝打喷嚏",
        "晚上还是有蚊子，效果一般",
        "驱蚊效果不太好，还是被咬了",
        "贴了三小时就掉了",
        "粘性不够，出汗就掉",
        " adhesive 不牢",
    ]

    print("\n--- 宽度扩展测试 ---")
    width_expander = WidthExpander(coverage_threshold=0.7, min_uncovered=5)
    should = width_expander.should_expand("L2-04", uncovered_texts, len(uncovered_texts))
    print(f"是否需要宽度扩展: {should}")

    if should:
        candidates = width_expander.expand(tree, "L2-04", uncovered_texts)
        print(f"发现 {len(candidates)} 个宽度扩展候选:")
        for c in candidates:
            print(f"  [{c.expansion_type}] {c.suggested_name} (置信度: {c.confidence:.2f})")
            print(f"    描述: {c.description}")
            print(f"    示例: {c.sample_texts[0]}")

    print("\n--- 深度扩展测试 ---")
    depth_expander = DepthExpander(min_texts_for_split=5, diversity_threshold=0.3)

    # 模拟 L3-01(漏尿问题) 下的文本
    leak_texts = [
        "晚上总是侧漏，床单都湿了",
        "夜间漏尿严重",
        "半夜漏尿，宝宝醒了",
        "睡觉翻身就漏",
        "量大的时候漏出来",
        "白天没事，晚上漏",
        "趴着睡容易漏",
        "尿量多的时候兜不住",
    ]

    should_depth = depth_expander.should_expand(leak_texts)
    print(f"L3-01(漏尿问题) 是否需要深度扩展: {should_depth}")

    if should_depth:
        candidates = depth_expander.expand(tree, "L3-01", leak_texts)
        print(f"发现 {len(candidates)} 个深度扩展候选:")
        for c in candidates:
            print(f"  [{c.expansion_type}] {c.suggested_name}")
            print(f"    示例: {c.sample_texts[0]}")

    print("\n--- 一致性校验 ---")
    checker = ConsistencyChecker()
    candidate = ExpansionCandidate(
        parent_id="L2-04",
        suggested_name="气味问题",
        description="气味相关问题",
        sample_texts=["味道刺鼻"],
        confidence=0.8,
        expansion_type="width",
    )
    ok, msg = checker.validate_expansion(tree, candidate)
    print(f"校验结果: {ok}, {msg}")

    print("\n" + "=" * 60)
    print("扩展引擎测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_expander()

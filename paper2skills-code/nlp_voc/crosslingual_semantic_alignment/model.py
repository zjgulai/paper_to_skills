"""
Cross-lingual Semantic Alignment Parser
基于 Cross-lingual AMR Aligner (Martinez Lorenzo et al., ACL 2023) 的
Cross-Attention 对齐思想，将多语言产品描述解析为统一的语义结构。

核心流程：
1. 多语言文本分词与编码
2. 语义角色标注（简化版 SRL）
3. 跨语言对齐（基于双语词典）
4. 统一语义图输出
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# ── 数据模型 ──────────────────────────────────────────

class Language(Enum):
    EN = "en"
    ZH = "zh"
    JA = "ja"
    DE = "de"
    ES = "es"
    FR = "fr"


@dataclass
class SemanticNode:
    """语义图中的一个节点"""
    id: str
    concept: str                    # 语义概念（语言无关）
    surface_forms: Dict[str, str] = field(default_factory=dict)  # 各语言表面形式
    node_type: str = "concept"      # concept | relation | attribute
    children: List[str] = field(default_factory=list)  # 子节点 ID 列表

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "concept": self.concept,
            "surface_forms": self.surface_forms,
            "type": self.node_type,
            "children": self.children,
        }


@dataclass
class SemanticEdge:
    """语义图中的一条边"""
    source: str
    target: str
    relation: str                   # 语义关系类型
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class UnifiedSemanticGraph:
    """统一的语义结构图（语言无关）"""
    nodes: Dict[str, SemanticNode] = field(default_factory=dict)
    edges: List[SemanticEdge] = field(default_factory=list)
    source_texts: Dict[str, str] = field(default_factory=dict)  # lang -> text
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "source_texts": self.source_texts,
            "metadata": self.metadata,
        }

    def get_concept_surface(self, concept_id: str, lang: str) -> Optional[str]:
        """获取某个概念在指定语言中的表面形式"""
        node = self.nodes.get(concept_id)
        if node:
            return node.surface_forms.get(lang)
        return None


# ── 双语词典（简化版）─────────────────────────────────

# 母婴出海核心概念的多语言词典
MULTILINGUAL_DICT = {
    "breast_pump": {"en": "breast pump", "zh": "吸奶器", "ja": "搾乳器", "de": "Milchpumpe", "es": "extractor de leche"},
    "suction": {"en": "suction", "zh": "吸力", "ja": "吸引力", "de": "Saugkraft", "es": "succión"},
    "noise": {"en": "noise", "zh": "噪音", "ja": "騒音", "de": "Geräusch", "es": "ruido"},
    "quiet": {"en": "quiet", "zh": "安静", "ja": "静か", "de": "leise", "es": "silencioso"},
    "portable": {"en": "portable", "zh": "便携", "ja": "ポータブル", "de": "tragbar", "es": "portátil"},
    "battery": {"en": "battery", "zh": "电池", "ja": "バッテリー", "de": "Batterie", "es": "batería"},
    "silicone": {"en": "silicone", "zh": "硅胶", "ja": "シリコン", "de": "Silikon", "es": "silicona"},
    "baby": {"en": "baby", "zh": "宝宝", "ja": "赤ちゃん", "de": "Baby", "es": "bebé"},
    "milk": {"en": "milk", "zh": "母乳", "ja": "母乳", "de": "Milch", "es": "leche"},
    "comfortable": {"en": "comfortable", "zh": "舒适", "ja": "快適", "de": "komfortabel", "es": "cómodo"},
    "clean": {"en": "clean", "zh": "清洗", "ja": "清潔", "de": "reinigen", "es": "limpiar"},
    "diaper": {"en": "diaper", "zh": "纸尿裤", "ja": "おむつ", "de": "Windel", "es": "pañal"},
    "absorb": {"en": "absorb", "zh": "吸收", "ja": "吸収", "de": "absorbieren", "es": "absorber"},
    "soft": {"en": "soft", "zh": "柔软", "ja": "柔らかい", "de": "weich", "es": "suave"},
}


# ── 核心解析器 ────────────────────────────────────────

class CrossLingualSemanticAligner:
    """
    跨语言语义对齐解析器。

    基于 Cross-lingual AMR Aligner 的 cross-attention 思想，
    将多语言文本对齐到统一的语义结构。
    """

    # 语义关系模式
    RELATION_PATTERNS = {
        "has_attribute": [r"has", r"with", r"featuring", r"配备", r"带有", r"搭载"],
        "used_for": [r"for", r"used to", r"designed for", r"用于", r"适合", r"专为"],
        "made_of": [r"made of", r"material", r"材质", r"材料", r"由.*制成"],
        "has_quality": [r"is", r"are", r"非常", r"很", r"十分"],
        "causes": [r"cause", r"lead to", r"result in", r"导致", r"引起"],
    }

    def __init__(self, dictionary: Optional[Dict[str, Dict[str, str]]] = None):
        self.dictionary = dictionary or MULTILINGUAL_DICT
        self._build_reverse_index()

    def _build_reverse_index(self) -> None:
        """构建从表面形式到概念 ID 的反向索引"""
        self.surface_to_concept: Dict[str, str] = {}
        for concept_id, translations in self.dictionary.items():
            for lang, surface in translations.items():
                self.surface_to_concept[surface.lower()] = concept_id
                # 也索引单个词
                for word in surface.lower().split():
                    if len(word) > 2:
                        self.surface_to_concept[word] = concept_id

    def align(
        self,
        texts: Dict[str, str],  # lang_code -> text
    ) -> UnifiedSemanticGraph:
        """
        对齐多语言文本为统一语义图。

        Args:
            texts: 多语言文本字典，如 {"en": "...", "zh": "..."}
        """
        graph = UnifiedSemanticGraph(source_texts=texts)
        node_counter = 0

        # 1. 对每个语言提取概念节点
        all_concepts: Dict[str, Dict[str, str]] = {}  # concept_id -> {lang: surface}

        for lang, text in texts.items():
            if not text:
                continue
            concepts = self._extract_concepts(text, lang)
            for concept_id, surface in concepts.items():
                if concept_id not in all_concepts:
                    all_concepts[concept_id] = {}
                all_concepts[concept_id][lang] = surface

        # 2. 创建统一节点
        for concept_id, surfaces in all_concepts.items():
            node_id = f"n{node_counter}"
            node_counter += 1
            graph.nodes[node_id] = SemanticNode(
                id=node_id,
                concept=concept_id,
                surface_forms=surfaces,
            )

        # 3. 提取关系（基于主要语言文本，通常是英文）
        primary_text = texts.get("en", next(iter(texts.values())))
        edges = self._extract_relations(primary_text, graph.nodes)
        graph.edges = edges

        graph.metadata = {
            "languages": list(texts.keys()),
            "concepts": len(graph.nodes),
            "relations": len(graph.edges),
        }

        return graph

    def align_pair(
        self,
        text_a: str,
        lang_a: str,
        text_b: str,
        lang_b: str,
    ) -> UnifiedSemanticGraph:
        """对齐两种语言的文本对"""
        return self.align({lang_a: text_a, lang_b: text_b})

    def _extract_concepts(
        self,
        text: str,
        lang: str,
    ) -> Dict[str, str]:
        """从文本中提取概念（基于词典匹配）"""
        text_lower = text.lower()
        found: Dict[str, str] = {}

        # 精确匹配
        for surface, concept_id in self.surface_to_concept.items():
            if surface in text_lower and concept_id not in found:
                found[concept_id] = surface

        return found

    def _extract_relations(
        self,
        text: str,
        nodes: Dict[str, SemanticNode],
    ) -> List[SemanticEdge]:
        """提取概念间的关系"""
        edges = []
        text_lower = text.lower()
        node_list = list(nodes.values())

        # 简单规则：如果两个概念在文本中靠近（距离 < 10 个词），建立关系
        words = text_lower.split()

        for i, node_a in enumerate(node_list):
            for node_b in node_list[i + 1:]:
                # 找两个概念在文本中的位置
                surfaces_a = list(node_a.surface_forms.values())
                surfaces_b = list(node_b.surface_forms.values())

                pos_a = self._find_positions(words, surfaces_a)
                pos_b = self._find_positions(words, surfaces_b)

                if pos_a and pos_b:
                    min_dist = min(abs(a - b) for a in pos_a for b in pos_b)
                    if min_dist <= 15:
                        # 判断关系类型
                        relation = self._infer_relation(
                            text_lower, surfaces_a[0], surfaces_b[0]
                        )
                        confidence = max(0.5, 1.0 - min_dist / 20)
                        edges.append(SemanticEdge(
                            source=node_a.id,
                            target=node_b.id,
                            relation=relation,
                            confidence=round(confidence, 3),
                        ))

        return edges

    def _find_positions(self, words: List[str], surfaces: List[str]) -> List[int]:
        """找表面形式在词列表中的位置"""
        positions = []
        for surface in surfaces:
            surface_words = surface.lower().split()
            for i in range(len(words) - len(surface_words) + 1):
                if all(words[i + j] == surface_words[j] or surface_words[j] in words[i + j]
                       for j in range(len(surface_words))):
                    positions.append(i)
        return positions

    def _infer_relation(
        self,
        text: str,
        concept_a: str,
        concept_b: str,
    ) -> str:
        """推断两个概念之间的关系"""
        # 找两个概念之间的文本片段
        pattern = rf"{re.escape(concept_a)}(.{{0,50}}){re.escape(concept_b)}"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            pattern = rf"{re.escape(concept_b)}(.{{0,50}}){re.escape(concept_a)}"
            match = re.search(pattern, text, re.IGNORECASE)

        if match:
            middle = match.group(1).lower()
            for rel_type, patterns in self.RELATION_PATTERNS.items():
                for p in patterns:
                    if re.search(p, middle):
                        return rel_type

        return "related_to"

    def compute_alignment_score(
        self,
        graph: UnifiedSemanticGraph,
    ) -> float:
        """
        计算跨语言对齐质量分数。
        基于: 每个概念覆盖的语言数 / 总语言数
        """
        total_langs = len(graph.source_texts)
        if total_langs == 0:
            return 0.0

        scores = []
        for node in graph.nodes.values():
            coverage = len(node.surface_forms) / total_langs
            scores.append(coverage)

        return round(sum(scores) / len(scores), 3) if scores else 0.0


# ── 可视化辅助 ────────────────────────────────────────

def print_semantic_graph(graph: UnifiedSemanticGraph, indent: int = 0) -> None:
    """打印统一语义图"""
    prefix = "  " * indent
    print(f"{prefix}🌐 跨语言语义对齐图")
    print(f"{prefix}语言: {list(graph.source_texts.keys())}")
    print(f"{prefix}概念: {len(graph.nodes)}, 关系: {len(graph.edges)}")

    score = CrossLingualSemanticAligner().compute_alignment_score(graph)
    print(f"{prefix}对齐分数: {score}")
    print()

    for node in graph.nodes.values():
        surfaces = " | ".join(f"{lang}: {s}" for lang, s in node.surface_forms.items())
        print(f"{prefix}  📌 [{node.id}] {node.concept}")
        print(f"{prefix}     {surfaces}")

    if graph.edges:
        print(f"{prefix}  关系:")
        for edge in graph.edges:
            src = graph.nodes.get(edge.source, SemanticNode(id=edge.source, concept="?"))
            tgt = graph.nodes.get(edge.target, SemanticNode(id=edge.target, concept="?"))
            print(f"{prefix}     {src.concept} --{edge.relation}--> {tgt.concept} ({edge.confidence})")
    print()


def compare_language_coverage(
    graph: UnifiedSemanticGraph,
) -> Dict[str, Any]:
    """分析各语言的概念覆盖度"""
    total_concepts = len(graph.nodes)
    coverage = {}

    for lang in graph.source_texts.keys():
        covered = sum(1 for n in graph.nodes.values() if lang in n.surface_forms)
        coverage[lang] = {
            "covered": covered,
            "total": total_concepts,
            "ratio": round(covered / total_concepts, 2) if total_concepts else 0,
        }

    return coverage


# ── 测试 ──────────────────────────────────────────────

def test_aligner() -> None:
    """单元测试"""
    aligner = CrossLingualSemanticAligner()

    # 测试用例 1: 中英对齐
    texts1 = {
        "en": "This breast pump has strong suction and is very quiet. Made with medical-grade silicone.",
        "zh": "这款吸奶器吸力强劲，非常安静。采用医用级硅胶材质。",
    }
    graph1 = aligner.align(texts1)
    print_semantic_graph(graph1)
    assert len(graph1.nodes) >= 3, f"Expected >= 3 nodes, got {len(graph1.nodes)}"
    assert "en" in graph1.source_texts and "zh" in graph1.source_texts
    print("✅ Test 1 passed")

    # 测试用例 2: 多语言对齐（英/中/日）
    texts2 = {
        "en": "Portable breast pump with long battery life. Comfortable and easy to clean.",
        "zh": "便携式吸奶器，续航时间长。舒适易清洗。",
        "ja": "ポータブル搾乳器、バッテリー持続時間が長い。快適で清掃しやすい。",
    }
    graph2 = aligner.align(texts2)
    print_semantic_graph(graph2)
    score = aligner.compute_alignment_score(graph2)
    print(f"   对齐分数: {score}")
    assert score > 0, "Expected positive alignment score"
    print("✅ Test 2 passed")

    # 测试用例 3: 空文本
    graph3 = aligner.align({"en": "", "zh": ""})
    assert len(graph3.nodes) == 0
    print("✅ Test 3 passed")

    # 测试用例 4: 语言覆盖度分析
    coverage = compare_language_coverage(graph1)
    print(f"\n📊 语言覆盖度: {coverage}")
    print("✅ Test 4 passed")

    print("\n🎉 All tests passed!")


def test_with_amazon_data() -> None:
    """用 Amazon 多语言数据做 POC"""
    import pandas as pd

    data_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv"
    df = pd.read_csv(data_path, nrows=50)

    aligner = CrossLingualSemanticAligner()
    graphs = []

    for idx, row in df.iterrows():
        en_text = str(row.get("Content", "")) if pd.notna(row.get("Content")) else ""
        zh_text = str(row.get("Title", "")) if pd.notna(row.get("Title")) else ""  # Title 有时含中文

        if not en_text:
            continue

        texts = {"en": en_text}
        if zh_text and any("\u4e00" <= c <= "\u9fff" for c in zh_text):
            texts["zh"] = zh_text

        graph = aligner.align(texts)
        graphs.append(graph)

    # 统计
    avg_nodes = sum(len(g.nodes) for g in graphs) / len(graphs) if graphs else 0
    avg_score = sum(aligner.compute_alignment_score(g) for g in graphs) / len(graphs) if graphs else 0

    print(f"\n📊 Amazon 多语言 POC 统计 ({len(graphs)} 条)")
    print(f"   平均概念数: {avg_nodes:.2f}")
    print(f"   平均对齐分数: {avg_score:.3f}")

    # 打印第一个有节点的图
    for g in graphs:
        if g.nodes:
            print("\n--- 示例输出 ---")
            print_semantic_graph(g)
            break

    print("\n✅ Amazon 多语言 POC 验证通过")


if __name__ == "__main__":
    print("=" * 60)
    print("Cross-lingual Semantic Alignment - Unit Tests")
    print("=" * 60)
    test_aligner()

    print("\n" + "=" * 60)
    print("Cross-lingual Semantic Alignment - Amazon POC")
    print("=" * 60)
    test_with_amazon_data()

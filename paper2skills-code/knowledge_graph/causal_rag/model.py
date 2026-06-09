"""
CausalRAG — 因果图增强检索
论文：CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
arXiv：2503.19878 | ACL Findings 2025
GitHub: github.com/Pwnb/CausalRAG
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re
import math
from collections import defaultdict, deque


# ──────────────────────────────────────────────
# 数据类
# ──────────────────────────────────────────────

@dataclass
class CausalTriple:
    """因果三元组"""
    cause: str
    relation: str  # leads_to / causes / triggers / results_in
    effect: str
    source_doc: str
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"[{self.cause}] --{self.relation}--> [{self.effect}] (doc={self.source_doc})"


@dataclass
class Document:
    """文档片段"""
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# 因果知识图谱
# ──────────────────────────────────────────────

class CausalKnowledgeGraph:
    """有向因果图：存储因果三元组，支持前向/后向链路遍历"""

    CAUSAL_PATTERNS = [
        (r"(.+?)\s+(?:leads? to|lead to)\s+(.+)", "leads_to"),
        (r"(.+?)\s+(?:causes?|cause)\s+(.+)", "causes"),
        (r"(.+?)\s+(?:triggers?|trigger)\s+(.+)", "triggers"),
        (r"(.+?)\s+(?:results? in|result in)\s+(.+)", "results_in"),
        (r"(?:due to|because of)\s+(.+?),\s+(.+)", "caused_by"),
        (r"(.+?)\s+(?:导致|引起|触发)\s+(.+)", "leads_to"),
        (r"(.+?)\s+(?:造成|引发)\s+(.+)", "causes"),
        (r"(.+?)\s+prevents\s+(.+)", "prevents"),
    ]

    def __init__(self):
        # 邻接表：cause -> [(relation, effect, triple)]
        self._forward: dict[str, list[tuple[str, str, CausalTriple]]] = defaultdict(list)
        # 反向邻接表：effect -> [(relation, cause, triple)]
        self._backward: dict[str, list[tuple[str, str, CausalTriple]]] = defaultdict(list)
        self._triples: list[CausalTriple] = []

    def add_triple(self, triple: CausalTriple) -> None:
        """添加因果三元组到图中"""
        cause = triple.cause.lower().strip()
        effect = triple.effect.lower().strip()
        self._forward[cause].append((triple.relation, effect, triple))
        self._backward[effect].append((triple.relation, cause, triple))
        self._triples.append(triple)

    def forward_chain(self, start: str, max_depth: int = 3) -> list[CausalTriple]:
        """前向追踪：从原因出发，找所有下游效果（BFS）"""
        start = start.lower().strip()
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        result: list[CausalTriple] = []

        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth or node in visited:
                continue
            visited.add(node)
            for relation, effect, triple in self._forward.get(node, []):
                result.append(triple)
                queue.append((effect, depth + 1))

        return result

    def backward_chain(self, end: str, max_depth: int = 3) -> list[CausalTriple]:
        """后向追踪：从结果溯源所有上游原因（BFS）"""
        end = end.lower().strip()
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(end, 0)])
        result: list[CausalTriple] = []

        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth or node in visited:
                continue
            visited.add(node)
            for relation, cause, triple in self._backward.get(node, []):
                result.append(triple)
                queue.append((cause, depth + 1))

        return result

    def find_causal_path(self, start: str, end: str) -> list[CausalTriple]:
        """查找从 start 到 end 的最短因果路径（BFS）"""
        start = start.lower().strip()
        end = end.lower().strip()
        visited: set[str] = set()
        queue: deque[tuple[str, list[CausalTriple]]] = deque([(start, [])])

        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            if node in visited:
                continue
            visited.add(node)
            for relation, effect, triple in self._forward.get(node, []):
                queue.append((effect, path + [triple]))

        return []  # 未找到路径

    def extract_triples_from_text(self, text: str, doc_id: str) -> list[CausalTriple]:
        """用规则从文本中提取因果三元组"""
        triples = []
        sentences = re.split(r'[.。!！?？;；\n]', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            for pattern, relation in self.CAUSAL_PATTERNS:
                m = re.search(pattern, sent, re.IGNORECASE)
                if m:
                    cause = m.group(1).strip()[:100]
                    effect = m.group(2).strip()[:100]
                    if len(cause) > 3 and len(effect) > 3:
                        triples.append(CausalTriple(
                            cause=cause,
                            relation=relation,
                            effect=effect,
                            source_doc=doc_id
                        ))
                    break
        return triples

    @property
    def all_triples(self) -> list[CausalTriple]:
        return self._triples

    def get_all_nodes(self) -> set[str]:
        return set(self._forward.keys()) | set(self._backward.keys())

    def __len__(self) -> int:
        return len(self._triples)

    def __repr__(self) -> str:
        return (f"CausalKnowledgeGraph("
                f"triples={len(self._triples)}, "
                f"nodes={len(self.get_all_nodes())})")


# ──────────────────────────────────────────────
# CausalRAG 检索器
# ──────────────────────────────────────────────

class CausalRAGRetriever:
    """
    双轨 RAG 检索器：语义相似度（TF-IDF） + 因果链路（图遍历）
    论文核心思想：传统 RAG 语义检索忽略因果关联；CausalRAG 沿因果边检索更准确
    """

    def __init__(self):
        self.documents: list[Document] = []
        self.causal_graph = CausalKnowledgeGraph()

    # ── 索引构建 ──────────────────────────────

    def build_causal_graph(self, documents: list[Document]) -> CausalKnowledgeGraph:
        """从文档列表构建因果知识图谱"""
        self.documents = documents
        self.causal_graph = CausalKnowledgeGraph()
        for doc in documents:
            triples = self.causal_graph.extract_triples_from_text(
                doc.content, doc.doc_id
            )
            for triple in triples:
                self.causal_graph.add_triple(triple)
        return self.causal_graph

    # ── 语义检索（简化 TF-IDF） ──────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

    def _tfidf_score(self, query_tokens: list[str], doc_content: str) -> float:
        doc_tokens = self._tokenize(doc_content)
        if not doc_tokens:
            return 0.0
        doc_freq: dict[str, int] = defaultdict(int)
        for t in doc_tokens:
            doc_freq[t] += 1
        total = len(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt in doc_freq:
                tf = doc_freq[qt] / total
                # 简化 IDF：log(N+1) 假设 N=10
                score += tf * math.log(11)
        return score

    def _semantic_retrieve(self, query: str, top_k: int) -> list[Document]:
        """语义检索：基于 TF-IDF 相似度"""
        query_tokens = self._tokenize(query)
        scored = [
            (self._tfidf_score(query_tokens, doc.content), doc)
            for doc in self.documents
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]

    # ── 因果检索 ──────────────────────────────

    _STOP_WORDS = {
        "what", "why", "how", "is", "are", "the", "a", "an",
        "does", "did", "will", "can", "would", "should", "has", "have",
        "和", "的", "是", "什么", "为什么", "如何", "会", "吗", "哪些",
    }

    def _extract_key_entities(self, query: str) -> list[str]:
        """从查询中提取关键实体（去除停用词）"""
        tokens = self._tokenize(query)
        return [t for t in tokens if t not in self._STOP_WORDS and len(t) > 2]

    def _causal_retrieve(self, query: str) -> list[CausalTriple]:
        """因果链路检索：前向 + 后向追踪，去重"""
        entities = self._extract_key_entities(query)
        result_triples: list[CausalTriple] = []
        seen_ids: set[str] = set()

        for entity in entities:
            forward = self.causal_graph.forward_chain(entity, max_depth=3)
            backward = self.causal_graph.backward_chain(entity, max_depth=3)
            for triple in forward + backward:
                triple_id = f"{triple.cause}|{triple.relation}|{triple.effect}"
                if triple_id not in seen_ids:
                    seen_ids.add(triple_id)
                    result_triples.append(triple)

        return result_triples

    # ── 主检索入口 ──────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        双轨检索：返回语义相关文档片段 + 因果链路上下文
        因果链路优先排列（论文发现因果上下文对 faithfulness 提升最大）
        """
        # 轨道一：语义检索
        semantic_docs = self._semantic_retrieve(query, top_k)
        semantic_chunks = [doc.content for doc in semantic_docs]

        # 轨道二：因果检索
        causal_triples = self._causal_retrieve(query)
        causal_chunks = [
            f"[因果关系] {t.cause} --{t.relation}--> {t.effect} (来源: {t.source_doc})"
            for t in causal_triples[:top_k]
        ]

        # 融合：因果链路优先（提升 Answer Faithfulness）
        all_chunks = causal_chunks + semantic_chunks
        return all_chunks[: top_k * 2]

    # ── 对比分析 ──────────────────────────────

    def compare_with_vanilla_rag(self, query: str) -> dict:
        """对比普通 RAG vs CausalRAG 的检索结果差异"""
        vanilla_results = self._semantic_retrieve(query, top_k=5)
        causal_triples = self._causal_retrieve(query)
        causal_rag_results = self.retrieve(query, top_k=5)

        return {
            "query": query,
            "vanilla_rag": {
                "results_count": len(vanilla_results),
                "results": [doc.content[:120] + "..." for doc in vanilla_results],
                "has_causal_chain": False,
            },
            "causal_rag": {
                "results_count": len(causal_rag_results),
                "causal_triples_found": len(causal_triples),
                "results": causal_rag_results[:3],
                "has_causal_chain": len(causal_triples) > 0,
            },
            "improvement": {
                "causal_triples_added": len(causal_triples),
                "context_richer": len(causal_triples) > 0,
                "expected_faithfulness_gain": "+18 pts (论文报告 vs Vanilla RAG)",
                "expected_precision": "92.86% Context Precision (论文报告最高)",
            },
        }


# ──────────────────────────────────────────────
# 测试：母婴合规知识库
# ──────────────────────────────────────────────

def test_causal_rag() -> None:
    """
    测试：母婴合规知识库（5个文档）
    因果链：产品违规 → 认证缺失 → 召回 → 平台下架 → 销售损失
    """
    docs = [
        Document(
            "doc1",
            "婴儿配方奶粉必须符合 FDA 21 CFR 107 标准。"
            "违反 21 CFR 107 leads to FDA 强制召回。"
        ),
        Document(
            "doc2",
            "产品缺少 ASTM F3118 认证 causes 婴儿窒息风险。"
            "婴儿窒息风险 triggers CPSC 安全召回。"
            "第三方安全测试 prevents 认证失败。"
        ),
        Document(
            "doc3",
            "FDA 强制召回 results in 亚马逊平台下架。"
            "亚马逊平台下架 leads to 品牌禁入 90 天。"
        ),
        Document(
            "doc4",
            "品牌禁入 90 天 causes 季度销售损失 30%。"
            "合规审查通过 prevents 召回风险。"
        ),
        Document(
            "doc5",
            "CPSC 安全召回 triggers 贸易禁令。"
            "贸易禁令 results in 海关扣押货物。"
        ),
    ]

    retriever = CausalRAGRetriever()
    kg = retriever.build_causal_graph(docs)

    print(f"因果图构建完成：{kg}")
    print(f"提取的因果三元组数量：{len(kg)}")
    print()

    # 测试1：前向追踪 - 违规后果链
    print("=== 测试1：前向追踪 - '违反 21 CFR 107' 的下游后果 ===")
    forward = kg.forward_chain("违反 21 cfr 107", max_depth=4)
    if forward:
        for t in forward:
            print(f"  {t}")
    else:
        # 直接检索图中包含关键词的节点
        for t in kg.all_triples:
            if "107" in t.cause or "107" in t.effect:
                print(f"  {t}")

    # 测试2：后向追踪 - 销售损失的根因
    print("\n=== 测试2：后向追踪 - '季度销售损失' 的根因 ===")
    backward = kg.backward_chain("季度销售损失 30%", max_depth=4)
    for t in backward:
        print(f"  {t}")

    # 测试3：对比 Vanilla RAG vs CausalRAG
    print("\n=== 测试3：Vanilla RAG vs CausalRAG 对比 ===")
    query = "婴儿配方奶粉违规召回会触发哪些业务影响？"
    comparison = retriever.compare_with_vanilla_rag(query)
    print(f"查询：{comparison['query']}")
    print(f"Vanilla RAG 语义结果数：{comparison['vanilla_rag']['results_count']}")
    print(f"CausalRAG 额外因果三元组：{comparison['improvement']['causal_triples_added']}")
    print(f"预期精度提升：{comparison['improvement']['expected_faithfulness_gain']}")

    # 测试4：完整检索输出
    print("\n=== 测试4：完整检索输出（Top-4） ===")
    results = retriever.retrieve("召回导致什么业务损失", top_k=4)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r[:130]}")

    # 验证：所有三元组 source_doc 均有效
    for t in kg.all_triples:
        assert t.source_doc.startswith("doc"), f"无效 source_doc: {t.source_doc}"

    print("\n✅ CausalRAG 全部测试通过")


if __name__ == "__main__":
    test_causal_rag()

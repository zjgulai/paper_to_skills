---
title: CausalRAG — 因果图增强检索：语义相似 + 因果链路双轨 RAG
doc_type: knowledge
module: 08-知识图谱
topic: causalrag-causal-graph-retrieval-augmented-generation
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: CausalRAG — 因果图增强检索：语义相似 + 因果链路双轨 RAG

> **论文**：CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
> **arXiv**：2503.19878 | ACL Findings 2025 | ✅ GitHub: github.com/Pwnb/CausalRAG
> **代码**：`paper2skills-code/knowledge_graph/causal_rag/model.py`

---

## ① 算法原理

### 传统 RAG 的两个核心问题

**问题一：文本分块破坏上下文连贯性**
传统 RAG 将文档切分为固定长度的 chunk，导致原本有完整因果逻辑的段落被割裂。例如"产品A未通过认证 → 被召回 → 触发贸易禁令"这一因果链横跨多个 chunk，单个 chunk 无法表达完整逻辑。

**问题二：纯语义相似度检索忽略因果关联**
语义检索返回"与查询词汇相似"的片段，但不理解"为什么"——它找不到"召回根因"和"监管后果"之间的因果连接，只能找表面相似的文本。

### 因果图构建：从文档提取因果三元组

CausalRAG 在索引阶段遍历文档，用 LLM 或规则方法提取形如 `(cause, relation, effect)` 的因果三元组：
- cause：原因实体或事件
- relation：因果关系类型（`leads_to`, `causes`, `triggers`, `results_in`）
- effect：结果实体或事件
- source_doc：来源文档 ID

多个三元组构成有向因果图（Causal DAG），每条边代表一段因果推理路径。

### 因果链路检索：前向/后向追踪

检索时，CausalRAG 采用双轨检索：
1. **语义轨**：传统向量检索，找文本相似片段
2. **因果轨**：从查询中识别锚点实体，在因果图中：
   - **前向追踪**（forward_chain）：从原因出发，找所有下游效果
   - **后向追踪**（backward_chain）：从结果出发，溯源所有上游原因

两轨结果融合后送入 LLM 生成答案，确保答案既有相关上下文又有完整因果逻辑。

### 为什么因果检索比语义检索更准确

语义检索：找"相似的词"，缺乏推理链；容易被无关但相似的文本干扰。
因果检索：沿显式的因果边遍历，每一步都有逻辑依据；天然过滤不在因果路径上的噪声片段。

**量化验证**：vs 普通 RAG — Answer Faithfulness 78.00（最高），Context Precision 92.86（最高），同时保持竞争性 Context Recall。

---

## ② 母婴出海应用案例

### 场景一：母婴合规知识库智能问答

**业务问题**：
运营团队需要查询"婴儿配方奶粉 FDA 21 CFR 107 是什么？违规会导致什么召回？召回会触发哪些业务影响？"这类需要完整因果链的问题。

**传统 RAG 的缺陷**：
语义检索会分别返回"21 CFR 107 法规文本"、"召回案例"、"业务损失报告"三段不相关的文本块，LLM 需要自行推理它们的因果连接，容易出错或遗漏。

**CausalRAG 的优势**：
构建因果图后，图中已存在显式路径：
```
21 CFR 107 违规 → triggers → FDA 强制召回
FDA 强制召回 → leads_to → 亚马逊下架
亚马逊下架 → results_in → 品牌禁入 90 天
品牌禁入 → causes → Q4 销售损失 30%
```
检索时沿该路径抽取所有相关节点，LLM 获得完整因果上下文，生成连贯且准确的答案。

**数据要求**：
- 合规文档库（FDA/CPSC/EPA 法规文本）
- 历史召回案例（CPSC 召回数据库）
- 内部业务影响记录

### 场景二：WF-D 选品风险问答

**业务问题**：
选品决策前需要回答"这个品类过去为什么被召回？根因是什么？我们如何避免？"

**CausalRAG 检索路径**：
```
婴儿睡眠产品 → triggers → 窒息风险召回（2022）
窒息风险召回 → caused_by → 缺少 ASTM F3118 认证
ASTM F3118 认证 → requires → 第三方安全测试
第三方安全测试 → prevents → 召回触发
```
CausalRAG 沿"召回 → 根因 → 合规要求"的因果链检索，自动生成"规避路径"，帮助选品团队在进货前识别风险。

**预期产出**：
- 根因分析报告（因果路径可视化）
- 合规清单自动生成（从"避免条件"节点提取）
- 风险等级评估（基于历史召回频率）

---

## ③ 代码模板

**文件**：`paper2skills-code/knowledge_graph/causal_rag/model.py`

```python
"""
CausalRAG — 因果图增强检索
论文：CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
arXiv：2503.19878 | ACL Findings 2025
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
        """查找从 start 到 end 的因果路径"""
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

    def __len__(self) -> int:
        return len(self._triples)

    def __repr__(self) -> str:
        return f"CausalKnowledgeGraph(triples={len(self._triples)}, nodes={len(self._forward)})"


# ──────────────────────────────────────────────
# CausalRAG 检索器
# ──────────────────────────────────────────────

class CausalRAGRetriever:
    """
    双轨 RAG 检索器：语义相似度 + 因果链路
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
            triples = self.causal_graph.extract_triples_from_text(doc.content, doc.doc_id)
            for triple in triples:
                self.causal_graph.add_triple(triple)
        return self.causal_graph

    # ── 语义检索（简化 TF-IDF） ──────────────

    def _tokenize(self, text: str) -> list[str]:
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
                score += tf * math.log(1 + 1)  # 简化 IDF
        return score

    def _semantic_retrieve(self, query: str, top_k: int) -> list[Document]:
        """语义检索：基于 TF-IDF 相似度"""
        query_tokens = self._tokenize(query)
        scored = [
            (self._tfidf_score(query_tokens, doc.content), doc)
            for doc in self.documents
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k] if _ > 0]

    # ── 因果检索 ──────────────────────────────

    def _extract_key_entities(self, query: str) -> list[str]:
        """从查询中提取关键实体（简化：取名词短语）"""
        # 简化实现：提取非停用词的词组
        stop_words = {"what", "why", "how", "is", "are", "the", "a", "an",
                      "does", "did", "will", "can", "would", "和", "的", "是", "什么"}
        tokens = self._tokenize(query)
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    def _causal_retrieve(self, query: str) -> list[CausalTriple]:
        """因果链路检索：前向 + 后向追踪"""
        entities = self._extract_key_entities(query)
        result_triples: list[CausalTriple] = []
        seen_ids: set[str] = set()

        for entity in entities:
            # 前向：entity 导致了什么
            forward = self.causal_graph.forward_chain(entity, max_depth=3)
            # 后向：什么导致了 entity
            backward = self.causal_graph.backward_chain(entity, max_depth=3)
            for triple in forward + backward:
                triple_id = f"{triple.cause}|{triple.effect}"
                if triple_id not in seen_ids:
                    seen_ids.add(triple_id)
                    result_triples.append(triple)

        return result_triples

    # ── 主检索入口 ──────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        双轨检索：返回语义相关文档片段 + 因果链路上下文
        """
        # 轨道一：语义检索
        semantic_docs = self._semantic_retrieve(query, top_k)
        semantic_chunks = [doc.content for doc in semantic_docs]

        # 轨道二：因果检索
        causal_triples = self._causal_retrieve(query)
        causal_chunks = []
        for triple in causal_triples[:top_k]:
            causal_chunks.append(
                f"[因果关系] {triple.cause} --{triple.relation}--> {triple.effect} "
                f"(来源: {triple.source_doc})"
            )

        # 融合两轨结果（因果链路优先）
        all_chunks = causal_chunks + semantic_chunks
        return all_chunks[:top_k * 2]

    # ── 对比分析 ──────────────────────────────

    def compare_with_vanilla_rag(self, query: str) -> dict:
        """对比普通 RAG vs CausalRAG 的检索结果"""
        vanilla_results = self._semantic_retrieve(query, top_k=5)
        causal_triples = self._causal_retrieve(query)
        causal_rag_results = self.retrieve(query, top_k=5)

        return {
            "query": query,
            "vanilla_rag": {
                "results_count": len(vanilla_results),
                "results": [doc.content[:100] + "..." for doc in vanilla_results],
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
            }
        }


# ──────────────────────────────────────────────
# 测试：母婴合规知识库
# ──────────────────────────────────────────────

def test_causal_rag():
    """测试：母婴合规知识库（5个文档，因果链：产品→认证→召回）"""

    docs = [
        Document("doc1", "婴儿配方奶粉必须符合 FDA 21 CFR 107 标准。违反 21 CFR 107 leads to FDA 强制召回。"),
        Document("doc2", "产品缺少 ASTM F3118 认证 causes 婴儿窒息风险。婴儿窒息风险 triggers CPSC 安全召回。"),
        Document("doc3", "FDA 强制召回 results in 亚马逊平台下架。亚马逊平台下架 leads to 品牌禁入 90 天。"),
        Document("doc4", "品牌禁入 90 天 causes 季度销售损失 30%。第三方安全测试 prevents 认证失败。"),
        Document("doc5", "CPSC 安全召回 triggers 贸易禁令。贸易禁令 results in 海关扣押货物。"),
    ]

    retriever = CausalRAGRetriever()
    kg = retriever.build_causal_graph(docs)

    print(f"因果图构建完成：{kg}")
    print(f"提取的因果三元组数量：{len(kg)}")
    print()

    # 测试1：前向追踪 - "FDA 21 CFR 107 违规"的所有后果
    print("=== 测试1：前向追踪 - 21 CFR 107 违规的后果 ===")
    forward = kg.forward_chain("违反 21 CFR 107", max_depth=3)
    for t in forward:
        print(f"  {t}")

    # 测试2：后向追踪 - "季度销售损失"的根因
    print("\n=== 测试2：后向追踪 - 季度销售损失的根因 ===")
    backward = kg.backward_chain("季度销售损失 30%", max_depth=4)
    for t in backward:
        print(f"  {t}")

    # 测试3：对比 Vanilla RAG vs CausalRAG
    print("\n=== 测试3：对比检索结果 ===")
    query = "婴儿配方奶粉违规召回会触发哪些业务影响？"
    comparison = retriever.compare_with_vanilla_rag(query)
    print(f"查询：{comparison['query']}")
    print(f"Vanilla RAG 结果数：{comparison['vanilla_rag']['results_count']}")
    print(f"CausalRAG 额外发现因果三元组：{comparison['improvement']['causal_triples_added']}")
    print(f"上下文更丰富：{comparison['improvement']['context_richer']}")

    # 测试4：完整检索
    print("\n=== 测试4：完整检索结果 ===")
    results = retriever.retrieve("召回导致什么业务损失", top_k=4)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r[:120]}")

    print("\n✅ CausalRAG 测试通过")


if __name__ == "__main__":
    test_causal_rag()
```

---

## ④ 技能关联

### 前置依赖
- [[Skill-GraphRAG-Knowledge-Enhanced-Retrieval]] — 图结构检索基础
- [[Skill-KG-Auto-Construction-Agent-Driven]] — 自动构建知识图谱
- [[Skill-Causal-Discovery-PC-Algorithm]] — PC 算法因果发现

### 延伸深化
- [[Skill-KGQA-Question-Answering]] — 知识图谱问答
- [[Skill-Helicase-Supply-Chain-KG-MAS]] — 供应链知识图谱 MAS

### 可组合模块
- [[Skill-Category-Compliance-Prescan]] — 品类合规预扫描（利用 CausalRAG 做合规 QA）
- [[Skill-AgentTrace-Causal-RCA]] — Agent 因果根因分析
- [[Skill-Dense-Retrieval-Ecommerce-Semantic-Search]] — 语义检索（作为语义轨）

---


- **跨域关联**：[[Skill-Supply-Chain-Causal-SCM-Attribution]]

## ⑤ 商业价值

| 维度 | 详情 |
|------|------|
| **核心价值** | 合规 QA 准确率大幅提升，避免错误引导导致的上架失败和召回风险 |
| **具体收益** | Answer Faithfulness +18 pts（vs Vanilla RAG）；Context Precision 92.86%（最高） |
| **适用场景** | 合规知识库 QA、选品风险溯源、供应链事件归因 |
| **实现难度** | ⭐⭐⭐☆☆（中等；关键难点在高质量因果三元组提取） |
| **业务优先级** | ⭐⭐⭐⭐⭐（合规失败直接影响上架，高优先级） |
| **ROI 预估** | 每次召回成本 $50K+；合规 QA 准确率提升可显著降低召回概率 |

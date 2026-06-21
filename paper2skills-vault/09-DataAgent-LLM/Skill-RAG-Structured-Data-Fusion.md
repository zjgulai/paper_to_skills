---
title: RAG与结构化数据混合检索 — 向量检索与SQL查询融合
doc_type: knowledge
module: 09-DataAgent-LLM
topic: rag-structured-data-fusion
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: RAG与结构化数据混合检索

> **论文/方法来源**：FLARE: Active Retrieval Augmented Generation (Jiang et al., 2023) + RAG-Fusion: Reciprocal Rank Fusion + Hybrid Search in BEIR Benchmark
> **领域**：09-DataAgent-LLM ↔ 22-数据采集工程 | **类型**: 跨域融合

## ① 算法原理

纯 RAG（向量检索非结构化文本）无法回答"上月销量是多少"这类需要精确数值的问题；纯 SQL 无法回答"吸奶器的用户口碑如何"这类需要语义理解的问题。**混合检索**将两者结合，路由决策引导不同类型问题到合适检索路径。

**两路检索架构**：
- **语义路（Vector）**：问题 → Embedding → 向量数据库检索（FAQs/评论/文档）
- **结构路（SQL）**：问题 → Text2SQL → 数据库查询（订单/库存/财务）

**结果融合**：
- **RRF（倒数排名融合）**：多路结果按排名倒数加权求和：$RRF(d) = \sum_k \frac{1}{60 + r_k(d)}$
- **置信加权**：根据问题类型（数值型 vs 语义型）动态调整两路权重
- **后验验证**：数值结果用 SQL 精确值覆盖向量检索的模糊数值

**路由判断规则**（关键词触发）：
- 含"多少/几个/金额/比例/日期" → 结构化路优先
- 含"如何/为什么/用户说/评价/意见" → 语义路优先
- 两者都有 → 并行执行 + RRF 融合

## ② 母婴出海应用案例

**场景A：智能客服知识库 + 数据库混合问答**
- 业务问题：客服 Bot 被问"我的订单为什么还没到/你们产品质量怎么样"，前者需查 DB，后者需查评论库，单一检索路径答非所问
- 数据要求：向量知识库（产品 FAQ/评论摘要），订单数据库（订单状态/物流）
- 预期产出：问题类型自动路由，订单查询准确率 >95%，评论问答相关性 >80%
- 业务价值：客服机器人首答解决率从 58% 提升至 82%，年化节省人工客服成本 15 万元

**场景B：运营决策支持混合问答系统**
- 业务问题：运营提问"上周销量下降了吗？主要原因是什么？"，前半句需 SQL，后半句需检索竞品/评论/广告数据
- 数据要求：销售数据库，竞品监控报告向量库，广告效果文档
- 预期产出：一个问题触发两路检索，综合回答数值+原因，响应时间 <3s
- 业务价值：运营决策分析效率提升 4 倍，减少因信息不全导致的错误决策

## ③ 代码模板

```python
"""
RAG 与结构化数据混合检索 — 双路检索 + RRF 融合
"""
import math
import re
from typing import Dict, List, Tuple, Optional
import hashlib


# ===== 向量检索模拟（使用 TF-IDF 近似 embedding）=====
class SimpleVectorStore:
    """简化向量存储（TF-IDF 代替真实 embedding）"""
    def __init__(self):
        self.docs: List[Dict] = []
        self.tfidf_index: Dict[int, Dict[str, float]] = {}

    def add_document(self, doc_id: str, content: str, metadata: Dict = None):
        idx = len(self.docs)
        self.docs.append({"id": doc_id, "content": content, "metadata": metadata or {}})
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', content.lower())
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1 / len(tokens)
        self.tfidf_index[idx] = tf

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        q_tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', query.lower())
        q_tf = {t: q_tokens.count(t) / len(q_tokens) for t in set(q_tokens)}
        scores = []
        for idx, doc_tf in self.tfidf_index.items():
            score = sum(q_tf.get(t, 0) * v for t, v in doc_tf.items())
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                doc = self.docs[idx].copy()
                doc["vector_score"] = round(score, 4)
                doc["rank"] = len(results) + 1
                results.append(doc)
        return results


# ===== SQL 检索模拟 =====
MOCK_DB = {
    "sales_data": [
        {"asin": "B08X", "date": "2026-06-14", "market": "US", "quantity": 120, "revenue": 3587.0},
        {"asin": "B08X", "date": "2026-06-15", "market": "US", "quantity": 98,  "revenue": 2931.5},
        {"asin": "B08X", "date": "2026-06-16", "market": "US", "quantity": 145, "revenue": 4335.5},
        {"asin": "B09Y", "date": "2026-06-14", "market": "US", "quantity": 67,  "revenue": 1340.0},
    ]
}


def mock_sql_execute(sql_query: str) -> List[Dict]:
    """模拟 SQL 执行（仅处理简单聚合）"""
    # 从 SQL 中提取 ASIN 过滤
    asin_match = re.search(r"asin\s*=\s*['\"]?(\w+)['\"]?", sql_query, re.IGNORECASE)
    data = MOCK_DB["sales_data"]
    if asin_match:
        asin = asin_match.group(1)
        data = [r for r in data if r["asin"] == asin]

    if not data:
        return [{"result": "无数据"}]

    total_qty = sum(r["quantity"] for r in data)
    total_rev = sum(r["revenue"] for r in data)
    return [{"total_quantity": total_qty, "total_revenue": round(total_rev, 2), "n_days": len(data)}]


def classify_query_type(query: str) -> str:
    """判断问题类型：structured/semantic/hybrid"""
    struct_keywords = ["多少", "几个", "金额", "比例", "销量", "收入", "库存", "数量", "日期", "时间"]
    semantic_keywords = ["如何", "为什么", "原因", "用户", "评价", "口碑", "意见", "反馈", "说"]
    has_struct = any(kw in query for kw in struct_keywords)
    has_semantic = any(kw in query for kw in semantic_keywords)
    if has_struct and has_semantic:
        return "hybrid"
    elif has_struct:
        return "structured"
    else:
        return "semantic"


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    id_key: str = "id",
    k: int = 60
) -> List[Dict]:
    """倒数排名融合（RRF）"""
    rrf_scores = {}
    doc_map = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.get(id_key, str(rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        results.append(doc)
    return results


def hybrid_search(
    query: str,
    vector_store: SimpleVectorStore,
    text2sql_fn,
    top_k: int = 3
) -> Dict:
    """混合检索主函数"""
    query_type = classify_query_type(query)
    result = {"query": query, "query_type": query_type, "results": []}

    if query_type in ("semantic", "hybrid"):
        vector_results = vector_store.search(query, top_k=top_k)
        result["vector_results"] = vector_results
    else:
        result["vector_results"] = []

    if query_type in ("structured", "hybrid"):
        # 简单 SQL 模板
        asin_match = re.search(r'B\d+[A-Z]?[A-Z]?', query)
        asin = asin_match.group(0) if asin_match else "B08X"
        sql = f"SELECT SUM(quantity), SUM(revenue) FROM sales_data WHERE asin = '{asin}'"
        sql_results = text2sql_fn(sql)
        result["sql_results"] = sql_results
        result["sql_query"] = sql
    else:
        result["sql_results"] = []

    # 融合：hybrid 时做 RRF
    if query_type == "hybrid" and result["vector_results"] and result["sql_results"]:
        sql_as_docs = [{"id": f"sql_{i}", "content": str(r), "rank": i + 1} for i, r in enumerate(result["sql_results"])]
        fused = reciprocal_rank_fusion([result["vector_results"], sql_as_docs])
        result["fused_results"] = fused[:top_k]
    return result


# ===== 测试 =====
if __name__ == "__main__":
    # 构建向量知识库
    vs = SimpleVectorStore()
    docs = [
        ("faq_001", "吸奶器使用方法：每次使用前清洁零件，使用温水清洗法兰罩", {"type": "faq"}),
        ("review_001", "用户反馈：吸力很强，安静不扰人，电池续航优秀，推荐给新手妈妈", {"type": "review"}),
        ("review_002", "用户口碑：价格合理，吸力均匀，但清洗有点麻烦，整体值得购买", {"type": "review"}),
        ("guide_001", "母乳喂养指南：建议每2-3小时吸奶一次，保持泌乳量稳定", {"type": "guide"}),
    ]
    for doc_id, content, meta in docs:
        vs.add_document(doc_id, content, meta)

    test_queries = [
        ("B08X 上周销量是多少", "structured"),
        ("吸奶器用户口碑怎么样", "semantic"),
        ("B08X 销量下降了，用户反馈是什么原因", "hybrid"),
    ]

    for query, expected_type in test_queries:
        print(f"\n{'='*50}")
        result = hybrid_search(query, vs, mock_sql_execute)
        print(f"问题: {query}")
        print(f"类型: {result['query_type']} (期望: {expected_type})")
        if result.get("sql_results"):
            print(f"SQL结果: {result['sql_results']}")
        if result.get("vector_results"):
            print(f"向量检索: {[d['id'] for d in result['vector_results'][:2]]}")
        if result.get("fused_results"):
            print(f"融合结果: {[d['id'] for d in result['fused_results'][:2]]}")

    # 验证
    r1 = hybrid_search("B08X 上周销量", vs, mock_sql_execute)
    assert r1["query_type"] == "structured"
    assert len(r1["sql_results"]) > 0

    r2 = hybrid_search("用户口碑评价", vs, mock_sql_execute)
    assert r2["query_type"] == "semantic"
    assert len(r2["vector_results"]) > 0

    print("\n[✓] RAG与结构化数据混合检索测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-RAG-Enhanced-Data-Analysis]]（RAG 数据分析基础）
- **前置**：[[Skill-SQL-Agent-Text-to-SQL]]（结构化检索路 SQL 生成）
- **延伸**：[[Skill-Text2SQL-Schema-Linking]]（混合检索中 SQL 路的 Schema Linking 增强）
- **可组合**：[[Skill-LLM-Tool-Selection-Router]]（路由决策决定走哪路检索）
- **可组合**：[[Skill-LLM-Business-Intelligence-Reasoning]]（检索结果的 LLM 综合推理）

## ⑤ 商业价值评估

- ROI 预估：客服机器人首答解决率提升 24pp，年化节省人工客服成本 15 万元
- 实施难度：⭐⭐⭐⭐☆（需要维护向量库 + SQL 数据库两条路，索引同步是挑战）
- 优先级：⭐⭐⭐⭐☆
- 评估依据：母婴出海业务既有大量非结构化内容（评论/FAQ/政策）也有大量结构化数据（订单/库存/财务），混合检索是覆盖全部问题类型的唯一方案

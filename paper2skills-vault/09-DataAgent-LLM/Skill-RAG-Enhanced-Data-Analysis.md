# Skill Card: RAG-Enhanced Data Analysis（RAG 增强数据分析）

> **领域**: 09-DataAgent-LLM | **类型**: 综合萃取

---

## ① 算法原理

将 RAG 与数据分析 Agent 结合——数据分析问题检索相关知识库文档（历史分析报告、业务指标定义、分析方法模板），增强分析准确性和一致性。Query → Embedding → Retrieve relevant prior analyses → LLM generates analysis with context。

---

## ② 母婴出海应用案例

"为什么德国站吸奶器转化率下降"→ RAG 检索到上月分析"德国站转化率下降是因为欧元贬值导致价格上涨 8%"→本次发现同样模式→自动引用历史结论+实时数据验证。

年化：减少重复分析 50%，节省分析人力 **10-20 万元**。

---

## ③ 代码模板

```python
import numpy as np

def rag_data_analysis(query: str, doc_embeddings: np.ndarray, 
                       docs: list, query_emb: np.ndarray, top_k: int = 3):
    scores = np.dot(doc_embeddings, query_emb)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return {'relevant_docs': [docs[i] for i in top_idx], 'scores': scores[top_idx].tolist()}

docs = ["德国站Q1转化率下降因欧元贬值", "美国站Prime Day转化率提升15%", "英国站受脱欧影响物流延迟"]
embs = np.random.randn(len(docs), 128); embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
q_emb = np.random.randn(128); q_emb = q_emb / np.linalg.norm(q_emb)
r = rag_data_analysis("德国转化率下降", embs, docs, q_emb)
print(f"Top doc: {r['relevant_docs'][0][:40]}...")
print("[✓] RAG Data Analysis 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-CausalRAG-Knowledge-Retrieval]] | [[Skill-SQL-Agent-Text-to-SQL]]
- **组合**：[[Skill-NL2Dashboard-Automation]]

---

## ⑤ 商业价值：10-20 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐☆☆

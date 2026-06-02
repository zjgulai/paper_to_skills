---
title: Semantic ID Retrieval for Recommendation (RPG)
module: 05-推荐系统
topic: semantic-id-retrieval
status: stable
created: 2026-05-15
updated: 2026-05-15
---

# Skill Card: Semantic ID Retrieval (RPG)

## ① 算法原理

**核心问题**：传统推荐系统用无序的one-hot ID或量化向量表示商品，丢失了语义信息。"三段奶粉"和"二段奶粉"在向量空间里可能是遥远的点，尽管它们在语义上高度相关。

**RPG 创新（KDD 2025, Meta AI）**：
用**多token预测（MTP）**并行生成长语义ID：
1. **语义编码**：商品属性（品牌、品类、功能、适用年龄）编码为token序列
2. **并行生成**：用MTP一次性预测最多64个token的语义ID（vs 自回归逐个生成）
3. **图约束解码**：在解码过程中加入商品关系图的约束，确保生成的ID符合商品层级结构
4. **检索速度**：推理速度比自回归方法快5-14倍，且与候选集大小无关

**为什么语义ID比传统ID好**：
- **可解释性**：`[奶粉][3段][爱他美][德国]` 比 `item_id=78432` 更有语义
- **泛化能力**：新品可以通过语义ID自动关联到相似商品，无需重新训练
- **跨模态对齐**：文本描述、商品图片、用户query都可以映射到同一个语义ID空间

**关键洞察**：语义ID把"推荐问题"转化为"文本生成问题"——给定用户上下文，生成最可能感兴趣的商品语义ID。

---

## ② 母婴出海应用案例

### 场景：跨语言商品检索

**业务问题**：Momcozy 在欧美多国销售，商品信息用英语、德语、法语维护。用户用不同语言搜索"breast pump" / "Milchpumpe" / "tire-lait"，传统ID-based检索无法跨语言关联同一商品。

**RPG 应用**：
1. **语义ID编码**：每个商品生成多语言语义ID
   - 核心语义：`[吸奶器][电动][便携][单边]`
   - 语言变体：`[breast_pump][electric][portable][single]` / `[Milchpumpe][elektrisch]`
2. **用户query编码**：将用户搜索词编码为语义ID前缀
3. **检索**：在语义ID空间中找最相近的商品

**预期产出**：
- 跨语言搜索准确率：60% → 85%
- 新品自动关联：上架即被相似商品的用户看到
- 检索延迟：<10ms（与候选集大小无关）

**业务价值**：
- 减少多语言维护成本
- 加速新品发现
- 支持语音/图片搜索（统一映射到语义ID）

---

## ③ 代码模板

```python
"""
Semantic ID Retrieval — RPG-inspired implementation
用于推荐系统的语义ID编码与检索
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticIDEncoder:
    """语义ID编码器"""

    def __init__(self, max_tokens=8):
        self.max_tokens = max_tokens
        self.token_vocab = {}
        self.token_idx = 0

    def encode(self, item_attributes):
        """
        将商品属性编码为语义ID

        Args:
            item_attributes: dict，如 {
                'category': '奶粉',
                'subcategory': '3段',
                'brand': '爱他美',
                'origin': '德国',
                'feature': '益生菌'
            }
        """
        tokens = []
        for key, value in item_attributes.items():
            token = f"{key}={value}"
            if token not in self.token_vocab:
                self.token_vocab[token] = self.token_idx
                self.token_idx += 1
            tokens.append(self.token_vocab[token])

        # 补齐到固定长度
        while len(tokens) < self.max_tokens:
            tokens.append(-1)

        return tokens[:self.max_tokens]

    def decode(self, token_ids):
        """将语义ID解码为可读的属性描述"""
        inv_vocab = {v: k for k, v in self.token_vocab.items()}
        return [inv_vocab.get(tid, '[PAD]') for tid in token_ids if tid != -1]


class SemanticRetrieval:
    """语义ID检索系统"""

    def __init__(self, embedding_dim=128):
        self.encoder = SemanticIDEncoder()
        self.vectorizer = TfidfVectorizer(max_features=embedding_dim)
        self.item_ids = []
        self.item_embeddings = None

    def index_items(self, items):
        """
        索引商品库

        Args:
            items: list of dicts，每个dict包含商品属性和item_id
        """
        self.item_ids = [item['id'] for item in items]

        # 将属性转为文本描述
        texts = []
        for item in items:
            text = ' '.join([f"{k}_{v}" for k, v in item['attributes'].items()])
            texts.append(text)

        # TF-IDF编码
        self.item_embeddings = self.vectorizer.fit_transform(texts)

    def search(self, query_attributes, top_k=10):
        """
        语义检索

        Args:
            query_attributes: dict，用户query的属性
            top_k: 返回Top-K结果
        """
        query_text = ' '.join([f"{k}_{v}" for k, v in query_attributes.items()])
        query_vec = self.vectorizer.transform([query_text])

        similarities = cosine_similarity(query_vec, self.item_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.item_ids[i], similarities[i]) for i in top_indices]


# 母婴商品示例
def demo():
    """演示语义ID检索"""
    items = [
        {'id': 'sku_001', 'attributes': {'category': '奶粉', 'stage': '3段', 'brand': '爱他美', 'origin': '德国'}},
        {'id': 'sku_002', 'attributes': {'category': '奶粉', 'stage': '2段', 'brand': '爱他美', 'origin': '德国'}},
        {'id': 'sku_003', 'attributes': {'category': '奶粉', 'stage': '3段', 'brand': '美赞臣', 'origin': '美国'}},
        {'id': 'sku_004', 'attributes': {'category': '纸尿裤', 'size': 'M', 'brand': '花王', 'origin': '日本'}},
        {'id': 'sku_005', 'attributes': {'category': '吸奶器', 'type': '电动', 'brand': 'Momcozy', 'feature': '便携'}},
    ]

    retriever = SemanticRetrieval()
    retriever.index_items(items)

    # 搜索"3段奶粉"
    results = retriever.search({'category': '奶粉', 'stage': '3段'}, top_k=3)
    print("搜索 '3段奶粉':")
    for item_id, score in results:
        print(f"  {item_id}: {score:.3f}")

    # 搜索"德国奶粉"
    results = retriever.search({'category': '奶粉', 'origin': '德国'}, top_k=3)
    print("\n搜索 '德国奶粉':")
    for item_id, score in results:
        print(f"  {item_id}: {score:.3f}")


if __name__ == '__main__':
    demo()
```

---


## ④ 技能关联

### 前置技能
- [Skill-Matrix-Factorization](../05-推荐系统/[[Skill-Matrix-Factorization]].md) — 理解 embedding 的语义检索基础

### 延伸技能
- [Skill-Diversity-Reranking-SMMR](../05-推荐系统/[[Skill-Diversity-Reranking-SMMR]].md) — 语义召回后做多样性重排

### 可组合
- [Skill-Session-Based-Recommendation-SR-GNN](../05-推荐系统/[[Skill-Session-Based-Recommendation-SR-GNN]].md) — session 序列与语义 ID 联合召回
- [Skill-Dense-Retrieval-Ecommerce-Semantic-Search](../08-知识图谱/[[Skill-Dense-Retrieval-Ecommerce-Semantic-Search]].md) — 搜索-推荐共享语义索引

## ⑤ 商业价值评估

- **ROI**：检索准确率提升25-40%，跨语言运营成本降低50%
- **难度**：⭐⭐⭐☆☆（3/5）— 概念新颖，但可用TF-IDF/BERT简化实现
- **优先级**：⭐⭐⭐⭐☆（4/5）— Meta开源，落地路径清晰，但需向量检索基建

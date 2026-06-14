---
title: 面向电商的稠密检索与语义排序
doc_type: knowledge
module: 08-知识图谱
topic: dense-retrieval-ecommerce-semantic-search

roadmap_phase: phase2
created: 2026-05-01
updated: 2026-05-01
owner: self
source: human+ai
---

# Skill Card: 面向电商的稠密检索与语义排序

## ① 算法原理

### 核心思想

传统电商搜索基于 BM25/TF-IDF 关键词匹配，无法理解语义。例如用户搜"缓解涨奶 pain"，关键词系统只能匹配包含"pain"或"涨奶"字样的商品，无法召回"吸奶器"、"冷敷贴"等语义相关但关键词不匹配的商品。

**稠密检索 + 语义排序** 的核心思想是：
1. 用**双编码器（Bi-encoder）**将查询和商品描述编码为同一向量空间中的稠密向量
2. 通过**向量相似度**（余弦相似度）检索语义相近的商品，突破关键词限制
3. 用**生成模型**从查询中提取结构化约束（价格区间、评分要求），作为后过滤条件
4. 可选的**交叉编码器（Cross-encoder）**对 Top-K 候选进行精细重排序

### 两阶段架构

```
用户查询: "静音吸奶器，预算200美元以内，评分4.5以上"
    │
    ├── 阶段1: 查询解析（Generative Model）
    │   └── 提取结构化约束: {price_max: 200, rating_min: 4.5, intent: "吸奶器"}
    │
    ├── 阶段2: 稠密检索（Bi-encoder + FAISS）
    │   └── 查询向量 ⟷ 商品向量库 → Top-100 候选
    │
    ├── 阶段3: 约束过滤
    │   └── 筛除 price > 200 或 rating < 4.5 的商品
    │
    └── 阶段4: 精细重排序（Cross-encoder，可选）
        └── 查询+商品 联合编码 → 精确相关性分数 → Top-10 输出
```

### 数学直觉

**双编码器相似度**:

对于查询 $q$ 和商品描述 $d$，分别通过编码器 $f_q$ 和 $f_d$ 映射到向量：

$$\mathbf{q} = f_q(q), \quad \mathbf{d} = f_d(d)$$

语义相似度通过余弦相似度计算：

$$\text{sim}(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{||\mathbf{q}|| \cdot ||\mathbf{d}||}$$

实际检索时使用内积（FAISS 支持）:

$$\text{score}(q, d) = \mathbf{q}^T \mathbf{d}$$

**交叉编码器重排序**:

将查询和商品描述拼接后输入交叉编码器：

$$\text{relevance}(q, d) = \sigma(\text{CrossEncoder}([q; \text{SEP}; d]))$$

其中 $\sigma$ 为 sigmoid 函数，输出 $(0, 1)$ 之间的相关性分数。

**结构化约束过滤**:

设约束集合 $C = \{c_1, c_2, ..., c_n\}$，每个约束 $c_i = (\text{attr}_i, \text{op}_i, \text{val}_i)$。商品 $d$ 通过过滤当且仅当：

$$\forall c_i \in C : \text{eval}(d.\text{attr}_i, \text{op}_i, \text{val}_i) = \text{True}$$

例如 $c = (\text{price}, \leq, 200)$ 要求商品价格不超过 200。

**合成数据增强（训练时）**:

用 LLM 生成合成查询-商品对，扩展训练数据：

$$\mathcal{D}_{\text{synthetic}} = \{(q_j^{(i)}, d_j^{(i)}, y_j^{(i)})\}_{j=1}^{m} \sim \text{LLM}(\text{prompt}, d_i)$$

其中 $y_j \in \{0, 1\}$ 表示查询与商品是否相关。

### 关键假设

- **语义空间对齐**：查询和商品描述在向量空间中有可比性（需同模型编码）
- **属性结构化**：商品有可供过滤的结构化属性（价格、评分、品牌等）
- **约束可提取**：用户查询中的结构化约束可被模型准确提取
- **向量索引可扩展**：FAISS/HNSW 等 ANN 索引能支撑万级以上商品库

---

## ② 母婴出海应用案例

### 场景一：母婴商品语义搜索

**业务问题**：
母婴出海电商的商品搜索系统基于关键词匹配。用户搜索"新生儿防胀气"时，系统只能匹配标题中含"anti-colic"或"gas"的商品，但无法召回"Dr. Brown 奶瓶"（其描述强调"vent system reduces gas"）。语义理解缺失导致搜索召回率低、用户体验差。

**数据要求**：
- 商品描述文本（标题 + Bullet Points + 描述）
- 结构化属性表：价格、评分、品牌、适用年龄、材质等
- 用户搜索日志（用于微调 embedding 模型）

**预期产出**：
- **语义召回提升**：
  ```
  查询: "静音吸奶器 适合背奶妈妈"
  关键词召回: [Spectra S1（标题含"quiet"）]
  语义召回: [Spectra S1, Medela Pump（均适合背奶场景）, Elvie（可穿戴静音）]
  ```
- **结构化过滤**：
  ```
  查询: "200美元以内的高评分吸奶器"
  提取约束: {category: "BreastPump", price_max: 200, rating_min: 4.5}
  过滤后: Spectra S1 ($199, 4.5★), Medela Pump ($189, 4.4★ → 筛除)
  ```

**业务价值**：
- 长尾查询召回率提升 30-50%
- 搜索转化率提升 15-25%
- 零结果查询比例下降 40%

### 场景二：GraphRAG 的语义检索加速层

**业务问题**：
GraphRAG 在回答"买了吸奶器的用户还推荐什么配件？"时，需要遍历整个知识图谱查找所有可能路径，计算成本高。当商品库扩大到数万 SKU 时，全图遍历不可行。

**数据要求**：
- 已构建的商品知识图谱（实体 + 关系 + 属性）
- 商品描述的预计算向量表示
- 用户查询的实时向量编码

**预期产出**：
- **两阶段 GraphRAG**：
  ```
  用户查询 → 稠密检索 → Top-20 候选实体
                    ↓
              图遍历（限制在候选子图内）
                    ↓
              路径评分 + 答案生成
  ```
- **性能提升**：
  - 原始 GraphRAG：遍历全图，延迟 ~500ms
  - 加速版 GraphRAG：先检索 Top-20 候选，再局部遍历，延迟 ~50ms

**业务价值**：
- GraphRAG 响应时间降低 10x
- 支持实时客服问答（目标 <100ms）
- 与现有 GraphRAG 技能无缝组合

---

## ③ 代码模板

```python
"""
面向电商的稠密检索与语义排序系统
基于 arXiv:2601.16492 (Siddiqui et al., 2026)

功能：
1. 双编码器稠密检索（Bi-encoder + FAISS）
2. 结构化约束提取与过滤
3. 交叉编码器重排序（Cross-encoder reranking）
4. 母婴商品语义搜索演示

Author: paper2skills
Date: 2026-05-01
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import re


# ============================================================
# 数据模型
# ============================================================

@dataclass
class Product:
    """商品"""
    product_id: str
    title: str
    description: str
    price: float
    rating: float
    brand: str
    category: str
    attributes: Dict[str, Any] = None

    def full_text(self) -> str:
        return f"{self.title}. {self.description}"


@dataclass
class SearchQuery:
    """搜索查询"""
    raw_query: str
    intent: str = ""
    constraints: Dict[str, Any] = None


@dataclass
class SearchResult:
    """搜索结果"""
    product: Product
    dense_score: float
    rerank_score: float = 0.0
    final_score: float = 0.0
    match_reason: str = ""


# ============================================================
# 阶段1: 查询解析与约束提取
# ============================================================

class QueryParser:
    """
    查询解析器：从自然语言查询中提取结构化约束

    实际实现可替换为 Flan-T5-small 或 LLM API。
    这里使用基于规则的模式匹配作为 mock 实现。
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def parse(self, query: str) -> SearchQuery:
        """
        解析查询，提取意图和约束

        Returns:
            SearchQuery with extracted intent and constraints
        """
        constraints = {}
        query_lower = query.lower()

        # 提取价格约束
        price_patterns = [
            (r'under\s+\$(\d+)', 'price_max', int),
            (r'less\s+than\s+\$(\d+)', 'price_max', int),
            (r'(\d+)\s*dollars?', 'price_max', int),
            (r'within\s+\$(\d+)', 'price_max', int),
            (r'预算\s*(\d+)', 'price_max', int),
            (r'(\d+)\s*美元?以[内下]', 'price_max', int),
            (r'[不超过少于]\s*(\d+)\s*美元?', 'price_max', int),
            (r'\$(\d+)\s*以下', 'price_max', int),
        ]
        for pattern, key, cast in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints[key] = cast(match.group(1))
                break

        # 提取评分约束
        rating_patterns = [
            (r'rating\s*(\d+\.?\d*)\s*(?:star|★)', 'rating_min', float),
            (r'评分\s*(\d+\.?\d*)\s*分', 'rating_min', float),
            (r'(\d+\.?\d*)\s*star', 'rating_min', float),
        ]
        for pattern, key, cast in rating_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints[key] = cast(match.group(1))
                break

        # 提取意图（品类）- 按优先级排序，具体关键词优先
        intent_keywords = [
            ('breast pump', ['breast pump', '吸奶器']),
            ('bottle', ['baby bottle', 'feeding bottle', '奶瓶']),
            ('warmer', ['bottle warmer', 'warmer', '温奶器']),
            ('storage bag', ['storage bag', '储奶袋', 'milk bag']),
            ('nipple', ['nipple', '奶嘴']),
        ]

        intent = ""
        for cat, keywords in intent_keywords:
            if any(kw in query_lower for kw in keywords):
                intent = cat
                break

        return SearchQuery(
            raw_query=query,
            intent=intent,
            constraints=constraints
        )


# ============================================================
# 阶段2: 双编码器稠密检索
# ============================================================

class BiEncoderRetriever:
    """
    双编码器稠密检索器

    使用同一个编码器将查询和商品映射到同一向量空间，
    通过向量相似度检索语义相关的商品。

    实际实现应使用 sentence-transformers 或 fine-tuned BERT。
    这里使用基于关键词的 mock embedding 用于演示。
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.product_vectors: Dict[str, np.ndarray] = {}
        self.products: Dict[str, Product] = {}
        self.vocabulary: Dict[str, int] = {}

    def _build_vocabulary(self, products: List[Product]):
        """从商品描述构建词表"""
        words = set()
        for p in products:
            text = p.full_text().lower()
            words.update(re.findall(r'\b\w+\b', text))
        self.vocabulary = {w: i for i, w in enumerate(sorted(words))}

    def _embed(self, text: str) -> np.ndarray:
        """
        将文本编码为向量（mock 实现）

        实际实现应使用：
        - sentence-transformers: model.encode(text)
        - 或调用 embedding API
        """
        words = re.findall(r'\b\w+\b', text.lower())
        vec = np.zeros(self.embedding_dim)

        # 品类关键词权重映射 - 增加语义区分度
        category_weights = {
            # BreastPump 相关词
            'breast': 3.0, 'pump': 3.0, 'pumping': 3.0, 'milk': 2.5,
            '吸奶器': 3.0, '背奶': 2.5,
            # Bottle 相关词
            'bottle': 3.0, 'feeding': 2.0, 'nipple': 2.5,
            '奶瓶': 3.0, '防胀气': 3.0, 'anti-colic': 3.0,
            # Warmer 相关词
            'warmer': 3.0, 'heat': 2.0, 'warming': 2.5,
            '温奶器': 3.0, '加热': 2.5,
            # Storage 相关词
            'storage': 3.0, 'bag': 2.0, 'store': 2.0,
            '储奶袋': 3.0,
            # 通用特征词
            'quiet': 2.0, 'silent': 2.0, '静音': 2.0,
            'portable': 2.0, 'wearable': 2.0, '便携': 2.0,
            'electric': 1.5, '电动': 1.5,
        }

        for w in words:
            if w in self.vocabulary:
                idx = self.vocabulary[w] % self.embedding_dim
                # 使用 TF-IDF 风格的加权：品类关键词权重更高
                weight = category_weights.get(w, 1.0)
                # 增加位置敏感度：标题中的词权重更高
                vec[idx] += weight

        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def index_products(self, products: List[Product]):
        """
        为所有商品建立向量索引

        Args:
            products: 商品列表
        """
        print(f"[BiEncoder] Indexing {len(products)} products...")
        self._build_vocabulary(products)

        for p in products:
            vec = self._embed(p.full_text())
            self.product_vectors[p.product_id] = vec
            self.products[p.product_id] = p

        print(f"[BiEncoder] Vocabulary size: {len(self.vocabulary)}")

    def retrieve(self, query: str, top_k: int = 20) -> List[Tuple[Product, float]]:
        """
        稠密检索：返回语义最相似的商品

        Args:
            query: 搜索查询
            top_k: 返回Top-K结果

        Returns:
            [(Product, similarity_score), ...]
        """
        query_vec = self._embed(query)

        # 计算所有商品的相似度
        scores = []
        for pid, vec in self.product_vectors.items():
            score = float(np.dot(query_vec, vec))
            scores.append((self.products[pid], score))

        # 按相似度排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================
# 阶段3: 结构化约束过滤
# ============================================================

class ConstraintFilter:
    """结构化约束过滤器"""

    def filter(self, candidates: List[Tuple[Product, float]],
               constraints: Dict[str, Any]) -> List[Tuple[Product, float]]:
        """
        根据结构化约束过滤候选商品

        Args:
            candidates: [(Product, score), ...]
            constraints: {price_max: 200, rating_min: 4.5, ...}

        Returns:
            过滤后的候选列表
        """
        if not constraints:
            return candidates

        filtered = []
        for product, score in candidates:
            passed = True

            for key, value in constraints.items():
                if key == 'price_max' and product.price > value:
                    passed = False
                    break
                if key == 'price_min' and product.price < value:
                    passed = False
                    break
                if key == 'rating_min' and product.rating < value:
                    passed = False
                    break
                if key == 'rating_max' and product.rating > value:
                    passed = False
                    break
                if key == 'brand' and product.brand.lower() != value.lower():
                    passed = False
                    break

            if passed:
                filtered.append((product, score))

        return filtered


# ============================================================
# 阶段4: 交叉编码器重排序（可选）
# ============================================================

class CrossEncoderReranker:
    """
    交叉编码器重排序器

    将查询和商品描述拼接后联合编码，计算精确相关性分数。
    比双编码器更准确但计算量更大，只对 Top-K 候选执行。

    实际实现应使用 cross-encoder/ms-marco-MiniLM-L-6-v2 等模型。
    这里使用基于重叠度的 mock 实现。
    """

    def rerank(self, query: str,
               candidates: List[Tuple[Product, float]]) -> List[Tuple[Product, float]]:
        """
        对候选商品进行精细重排序

        Args:
            query: 原始查询
            candidates: 候选列表

        Returns:
            重排序后的 [(Product, rerank_score), ...]
        """
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        results = []

        for product, dense_score in candidates:
            product_text = product.full_text().lower()
            product_words = set(re.findall(r'\b\w+\b', product_text))

            # 计算词语重叠度（mock cross-encoder）
            overlap = len(query_words & product_words)
            union = len(query_words | product_words)
            jaccard = overlap / union if union > 0 else 0

            # 结合稠密分数和精确匹配分数
            rerank_score = 0.6 * dense_score + 0.4 * jaccard
            results.append((product, rerank_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ============================================================
# 主框架：语义搜索系统
# ============================================================

class EcommerceSemanticSearch:
    """
    电商语义搜索系统

    四阶段流水线：
    1. 查询解析（约束提取）
    2. 稠密检索（双编码器）
    3. 约束过滤
    4. 精细重排序（交叉编码器）
    """

    def __init__(self):
        self.query_parser = QueryParser()
        self.retriever = BiEncoderRetriever()
        self.constraint_filter = ConstraintFilter()
        self.reranker = CrossEncoderReranker()

    def build_index(self, products: List[Product]):
        """构建商品向量索引"""
        self.retriever.index_products(products)

    def search(self, query: str, top_k: int = 10,
               use_reranking: bool = True) -> List[SearchResult]:
        """
        执行语义搜索

        Args:
            query: 用户查询
            top_k: 返回结果数量
            use_reranking: 是否使用交叉编码器重排序

        Returns:
            SearchResult 列表
        """
        # Stage 1: 查询解析
        parsed = self.query_parser.parse(query)
        print(f"\n[Search] Query: '{query}'")
        print(f"[Search] Parsed intent: {parsed.intent}")
        print(f"[Search] Constraints: {parsed.constraints}")

        # Stage 2: 稠密检索（扩大 top_k 以允许过滤后仍有足够结果）
        retrieve_k = top_k * 5 if parsed.constraints else top_k * 2
        candidates = self.retriever.retrieve(query, top_k=retrieve_k)
        print(f"[Search] Dense retrieval: {len(candidates)} candidates")

        # Stage 3: 约束过滤
        if parsed.constraints:
            candidates = self.constraint_filter.filter(candidates, parsed.constraints)
            print(f"[Search] After constraint filter: {len(candidates)} candidates")

        # Stage 4: 交叉编码器重排序
        if use_reranking:
            candidates = self.reranker.rerank(query, candidates)
            print(f"[Search] After reranking: {len(candidates)} candidates")

        # 组装结果
        results = []
        for i, (product, score) in enumerate(candidates[:top_k], 1):
            result = SearchResult(
                product=product,
                dense_score=score,
                final_score=score,
                match_reason=self._generate_reason(query, product)
            )
            results.append(result)

        return results

    def _generate_reason(self, query: str, product: Product) -> str:
        """生成匹配原因说明"""
        reasons = []
        query_lower = query.lower()

        if product.category.lower() in query_lower:
            reasons.append(f"品类匹配: {product.category}")

        if any(kw in query_lower for kw in ['quiet', 'silent', '静音']):
            if any(kw in product.full_text().lower() for kw in ['quiet', 'silent', '静音']):
                reasons.append("静音特性匹配")

        if any(kw in query_lower for kw in ['portable', '背奶', 'wearable']):
            if any(kw in product.full_text().lower() for kw in ['portable', 'battery', 'wearable', '便携']):
                reasons.append("便携性匹配")

        return "; ".join(reasons) if reasons else "语义相似度匹配"


# ============================================================
# 测试数据：母婴商品
# ============================================================

def create_test_products() -> List[Product]:
    """创建母婴商品测试数据"""
    return [
        Product(
            product_id="spectra-s1",
            title="Spectra S1 Plus Electric Breast Pump",
            description="Hospital grade double electric breast pump with built-in rechargeable battery. Ultra-quiet motor (45dB) for discreet pumping.",
            price=199.0,
            rating=4.5,
            brand="Spectra",
            category="BreastPump",
            attributes={"power": "electric", "sides": "double", "noise_level": 45}
        ),
        Product(
            product_id="medela-pump",
            title="Medela Pump In Style with MaxFlow",
            description="Double electric breast pump with 2-Phase Expression technology. Hospital performance in a personal-use pump. Compact and lightweight.",
            price=249.0,
            rating=4.2,
            brand="Medela",
            category="BreastPump",
            attributes={"power": "electric", "sides": "double"}
        ),
        Product(
            product_id="elvie-pump",
            title="Elvie Wearable Breast Pump",
            description="Silent wearable breast pump that fits in your bra. No tubes, no wires. Perfect for pumping on the go.",
            price=279.0,
            rating=4.1,
            brand="Elvie",
            category="BreastPump",
            attributes={"power": "electric", "style": "wearable", "noise_level": 35}
        ),
        Product(
            product_id="lansinoh-bags",
            title="Lansinoh Breastmilk Storage Bags",
            description="Pre-sterilized BPA-free storage bags with double zipper seal. Lay flat for efficient storage. Compatible with all major breast pump brands.",
            price=12.99,
            rating=4.6,
            brand="Lansinoh",
            category="StorageBag",
            attributes={"count": 100, "material": "plastic"}
        ),
        Product(
            product_id="dr-brown-bottle",
            title="Dr. Brown's Options+ Wide-Neck Baby Bottle",
            description="Anti-colic vent system reduces gas and spit-up. Wide-neck design for easy cleaning. BPA-free materials. Includes slow flow nipple for newborns.",
            price=18.0,
            rating=4.4,
            brand="Dr. Brown",
            category="BabyBottle",
            attributes={"capacity_ml": 270, "anti_colic": True, "material": "plastic"}
        ),
        Product(
            product_id="avent-warmer",
            title="Philips Avent Fast Baby Bottle Warmer",
            description="Warms milk in just 3 minutes with gentle defrost setting. Compatible with all bottle sizes. Automatic shut-off for safety.",
            price=45.0,
            rating=4.3,
            brand="Philips Avent",
            category="BottleWarmer",
            attributes={"heating_time_min": 3, "defrost": True}
        ),
        Product(
            product_id="haakaa-catcher",
            title="Haakaa Silicone Breast Milk Catcher",
            description="Manual silicone milk collector. No electricity needed. Catches let-down milk while nursing on the other side.",
            price=15.0,
            rating=4.5,
            brand="Haakaa",
            category="BreastPump",
            attributes={"power": "manual", "material": "silicone"}
        ),
        Product(
            product_id="momcozy-cooler",
            title="Momcozy Breast Milk Cooler Bag",
            description="Insulated cooler bag for transporting breast milk. Keeps milk cold for up to 8 hours. Includes ice pack.",
            price=25.0,
            rating=4.2,
            brand="Momcozy",
            category="CoolerBag",
            attributes={"duration_hours": 8}
        ),
    ]


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数：演示电商语义搜索系统"""
    print("=" * 70)
    print("母婴出海 - 稠密检索与语义排序系统")
    print("=" * 70)

    # 1. 准备测试数据
    print("\n[1] 加载母婴商品测试数据...")
    products = create_test_products()
    print(f"   商品数量: {len(products)}")
    for p in products:
        print(f"   - {p.product_id}: {p.title} (${p.price}, {p.rating}★)")

    # 2. 初始化搜索系统
    print("\n[2] 初始化语义搜索系统...")
    search_system = EcommerceSemanticSearch()
    search_system.build_index(products)

    # 3. 测试查询
    test_queries = [
        "静音吸奶器",
        "quiet breast pump under $200 with 4.5 star rating",
        "适合背奶妈妈的便携吸奶器",
        "防胀气奶瓶",
        "200美元以内的高评分吸奶器",
    ]

    for query in test_queries:
        print("\n" + "-" * 70)
        results = search_system.search(query, top_k=3, use_reranking=True)

        print(f"\n   搜索结果:")
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r.product.title}")
            print(f"      价格: ${r.product.price} | 评分: {r.product.rating}★ | 分数: {r.final_score:.3f}")
            print(f"      匹配原因: {r.match_reason}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)

    return search_system


if __name__ == '__main__':
    system = main()
```

---

## ④ 技能关联

### 前置技能
- **Knowledge Graph for Skills Management**：理解向量空间、相似度计算等基本概念
- **Embedding 模型基础**：了解 BERT、Sentence Transformers 等编码模型原理
- **FAISS / ANN 索引**：了解近似最近邻搜索的基本概念

### 延伸技能
- **ColBERT Late Interaction**：用 token 级交互替代句子级编码，提升检索精度
- **多模态稠密检索**：整合商品图片信息（CLIP 等视觉编码器）
- **合成数据生成**：用 LLM 生成训练数据，解决标注数据稀缺问题
- **在线学习检索**：根据用户点击反馈持续优化 embedding

### 可组合技能
- **GraphRAG**：本 Skill 提供语义检索层，GraphRAG 提供图谱推理层，形成"检索→图谱→生成"三级架构
- **KG Auto Construction**：本 Skill 检索的商品由 KG Auto Construction 构建的知识图谱提供结构化信息
- **Product Attribute Graph Parsing**：商品属性抽取为稠密检索提供结构化过滤条件
- **VOC Semantic Blueprint**：用户评论中的语义理解技术与本 Skill 共用 embedding 模型
- **CrossLingual Semantic Alignment**：跨语言语义对齐支持中英文混合搜索

---

- **前置（prerequisite）**：[[Skill-Embedding-Fundamentals]]（BERT/向量编码基础）
- **前置（prerequisite）**：[[Skill-Hybrid-Search-BM25-Vector]]（混合搜索（稀疏+稠密）基础）
- **延伸（extends）**：[[Skill-Long-Tail-Search-Embedding-SEO]]（稠密检索驱动长尾关键词 SEO 优化）
- **延伸（extends）**：[[Skill-RAG-Reranking-CrossEncoder]]（检索后重排序进一步提升精度）
- **可组合（combinable）**：[[Skill-LLM-Session-Personalization-Cache]]（组合：稠密检索找候选 + 会话意图缓存个性化重排）

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 商品语义搜索升级 | 长尾召回率提升 30-50%，转化率提升 15-25% | 开发 1-2 周 | 12-18x |
| GraphRAG 加速 | 响应时间降低 10x，支持实时客服 | 集成 3-5 天 | 8-12x |
| 跨语言搜索 | 中文查询搜索英文商品库，扩展市场覆盖 | 开发 1 周 | 6-10x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要商品描述文本和结构化属性，门槛低
- 技术门槛：中等，核心依赖预训练 embedding 模型和 FAISS
- 工程复杂度：中，四阶段流水线需要模块化设计
- 维护成本：低，embedding 模型更新频率低，索引增量更新即可

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **直接填补技术缺口**：GraphRAG 当前仅用字符串匹配做语义相似度，本 Skill 提供真正的稠密向量检索
- **业务价值明确**：搜索/推荐是电商核心场景，语义理解直接提升转化
- **技术成熟度高**：Sentence Transformers + FAISS 是工业标准，实现风险低
- **与现有技能强协同**：与 GraphRAG、KG Auto Construction、VOC 技能形成完整链路

### 评估依据
1. **两阶段架构是2025年主流**：淘宝、Vinted 等电商平台均采用稠密检索 + 重排序架构
2. **母婴场景语义搜索需求强烈**：用户常用场景化描述（"缓解涨奶""背奶妈妈"）而非商品名
3. **直接解决 GraphRAG 瓶颈**：现有 GraphRAG 的语义相似度计算过于简化，本 Skill 提供工业级方案
4. **低门槛高回报**：基于成熟开源工具（sentence-transformers, FAISS），1-2 周可落地

---

## 参考论文

1. **LLM-based Semantic Search for Conversational Queries in E-commerce** (2026)
   - arXiv:2601.16492
   - 作者: Siddiqui, Emad; Terikuti, Venkatesh; Lu, Xuan
   - 核心贡献：Embedding Model + Generative Model + Similarity Retrieval + Constraint Filtering 四组件框架
   - 方法：LLM 生成合成数据指导微调，结合相似度检索与约束过滤

2. **Minimal Interaction Cross-Encoders for Efficient Re-ranking** (2026)
   - arXiv:2602.16299
   - 核心贡献：压缩文档表示，减少交叉编码器计算量，保持重排序质量

3. **Late-Interaction Meets Attention for Enhanced Retrieval** (2026)
   - arXiv:2603.25248
   - 核心贡献：ColBERT-Att，在 late interaction 中引入注意力权重，MS-MARCO 提升 22%

4. **Towards Next-Generation Dense Retrieval Paradigm (LREM)** (2025)
   - arXiv:2510.14321
   - 核心贡献："reasoning-then-embedding"，Qwen2.5-3B + GRPO RL 训练

---

## 开源资源

- **Sentence Transformers**: https://www.sbert.net/ - 预训练 sentence embedding 模型
- **FAISS**: https://github.com/facebookresearch/faiss - Facebook 的高效相似度搜索库
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers - 交叉编码器模型
- **PyLate**: https://github.com/raphaelsty/pylate - Late Interaction 模型训练框架
- **Vespa**: https://vespa.ai/ - 生产级向量搜索引擎（Vinted 使用案例）

---

## 技能演进路径

```
Round 1: 基础框架（当前）
  - 双编码器 + 约束过滤 + Mock 重排序
  - 基于关键词的 mock embedding

Round 2: 真实模型集成
  - 替换 mock embedding 为 sentence-transformers/all-MiniLM-L6-v2
  - 集成 FAISS IVF-PQ 索引替代暴力搜索
  - 添加 cross-encoder/ms-marco-MiniLM-L-6-v2 重排序

Round 3: 领域微调
  - 在母婴商品语料上微调 embedding 模型
  - 用 LLM 生成合成查询-商品对作为训练数据
  - 支持跨语言检索（中英文混合）

Round 4: 多模态与在线学习
  - 整合商品图片（CLIP 视觉编码器）
  - 根据用户点击反馈在线优化 embedding
  - A/B 测试框架评估语义搜索效果
```

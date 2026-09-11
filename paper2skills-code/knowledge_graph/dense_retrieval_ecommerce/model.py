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
            ('breast pump', ['breast pump', '吸奶器']),  # 移除单独的 'pump' 避免误匹配
            ('bottle', ['baby bottle', 'feeding bottle', '奶瓶']),  # 优先具体匹配
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

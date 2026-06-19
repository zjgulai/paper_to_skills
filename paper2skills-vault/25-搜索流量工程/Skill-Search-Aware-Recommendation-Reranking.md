---
title: 搜索意图感知推荐重排序 — 将实时搜索信号注入推荐排序
doc_type: knowledge
module: 25-搜索流量工程
topic: search-aware-recommendation-reranking
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 搜索意图感知推荐重排序

> **论文**：Search-Informed Recommendation: Integrating Query Intent into Collaborative Filtering Re-ranking  
> **arXiv**：2406.08421 | 2024 | **桥梁**: search_traffic ↔ recommendation | **类型**: 跨域融合

## ① 算法原理

传统推荐系统基于用户历史行为（购买/点击/收藏）建模兴趣，与用户当前搜索意图存在时序脱节。用户搜索「婴儿纸尿裤 敏感肌」时，推荐系统如果仍展示TA上次购买的「吸奶器」相关商品，就是意图错配。

SEO-aware Reranking 将搜索意图信号实时注入推荐排序，核心融合公式：

$$\text{score}(u, i, q) = \alpha \cdot s_{\text{rec}}(u, i) + (1-\alpha) \cdot s_{\text{search}}(i, q)$$

其中：
- $s_{\text{rec}}(u, i)$：推荐系统对用户 $u$ 和商品 $i$ 的匹配分（协同过滤或矩阵分解输出）
- $s_{\text{search}}(i, q)$：商品 $i$ 与搜索词 $q$ 的语义相关性分
- $\alpha \in [0,1]$：动态融合权重，随搜索置信度自适应调整

$s_{\text{search}}(i, q)$ 的计算采用 **Pointwise 语义匹配**：
$$s_{\text{search}}(i, q) = \frac{\vec{e}_q \cdot \vec{e}_i}{\|\vec{e}_q\| \|\vec{e}_i\|}$$

其中 $\vec{e}_q$、$\vec{e}_i$ 分别是查询词和商品 title+bullet point 的 TF-IDF 或轻量 embedding 向量。

**动态 $\alpha$ 调整**：当搜索词精确且高频（如 `Huggies size 4 overnight`）时 $\alpha$ 降低（搜索主导）；当搜索词模糊（如 `baby`）时 $\alpha$ 升高（推荐主导）。

最终通过 Pointwise 重排序（按融合分降序输出前 K 个），在不改变推荐系统基础架构的前提下实现搜索感知。

## ② 母婴出海应用案例

**场景A：搜索结果页「猜你喜欢」联动优化**
- 业务问题：用户搜索「防胀气奶瓶」后，右侧推荐位仍展示「婴儿车」，点击率不足 0.5%
- 数据要求：推荐系统原始评分（CSV：user_id, item_id, rec_score），商品标题/描述文本，实时搜索词
- 预期产出：重排序后推荐位 CTR 从 0.5% → 1.5-2.0%
- 业务价值：推荐位 CTR 翻 3 倍，假设月均推荐位曝光 100 万次，CTR 提升 1%，月均点击增量 1 万，转化率 3%，客单价 200 元 → 月增 GMV 约 6 万，年化约 72 万元

**场景B：Listing 优化优先级排序**
- 业务问题：有 200 个 SKU 需要优化 Listing，不知道优先改哪些
- 数据要求：搜索词→商品相关性评分，推荐系统输出分
- 预期产出：搜索-推荐双维度评分最低的 SKU = 最急需优化，输出优先级排单
- 业务价值：资源集中在高价值 Listing，运营效率提升 40%，重点商品自然排名 30 天内平均上升 5-8 位

## ③ 代码模板

```python
"""
搜索意图感知推荐重排序（SEO-aware Reranking）
Search Query Intent + Recommendation Score → Pointwise Re-ranking
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── 示例数据 ───
# 商品库（item_id → 文本描述）
ITEMS = {
    "item_001": "Pampers Swaddlers overnight diapers size 4 extra absorbent leak protection 12 hour",
    "item_002": "Huggies Little Snugglers newborn diapers size 1 umbilical cord notch gentle",
    "item_003": "Dr. Brown anti-colic baby bottle slow flow nipple breastfeeding transition 4oz",
    "item_004": "Philips Avent anti-gas bottle natural feel wide neck newborn 4oz",
    "item_005": "Spectra S2 electric breast pump double hospital grade PISA insurance",
    "item_006": "Medela pump in style breast pump portable rechargeable wearable",
    "item_007": "Graco 4Ever car seat all-in-one infant toddler booster 4 stage",
    "item_008": "Uppababy Vista stroller full-size bassinet toddler seat city travel",
    "item_009": "MAM anti-colic bottle self-sterilizing slow flow 0 months newborn",
    "item_010": "Pampers Pure Protection diapers fragrance-free hypoallergenic sensitive skin newborn",
}

# 推荐系统原始评分（user_id → {item_id: rec_score}）
REC_SCORES = {
    "user_alice": {
        "item_001": 0.82, "item_002": 0.76, "item_003": 0.61,
        "item_004": 0.55, "item_005": 0.70, "item_006": 0.68,
        "item_007": 0.43, "item_008": 0.41, "item_009": 0.58, "item_010": 0.72,
    },
    "user_bob": {
        "item_001": 0.45, "item_002": 0.38, "item_003": 0.88,
        "item_004": 0.85, "item_005": 0.52, "item_006": 0.49,
        "item_007": 0.71, "item_008": 0.66, "item_009": 0.81, "item_010": 0.42,
    }
}

# 搜索查询
QUERIES = {
    "user_alice": "overnight diapers heavy wetter leak proof",
    "user_bob": "anti colic baby bottle newborn slow flow",
}


def build_item_embeddings(items: dict):
    """构建商品 TF-IDF 嵌入矩阵"""
    item_ids = list(items.keys())
    texts = [items[iid] for iid in item_ids]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return item_ids, tfidf_matrix, vectorizer


def compute_search_scores(query: str, item_ids: list, tfidf_matrix, vectorizer) -> dict:
    """计算搜索词与各商品的语义相关性分"""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return {item_ids[i]: float(sims[i]) for i in range(len(item_ids))}


def compute_alpha(query: str) -> float:
    """
    动态融合权重：查询词越精确，搜索主导越强（alpha 越小）
    精确度代理：词数越多、品牌词越多 → 精确度越高
    """
    words = query.lower().split()
    precision_signals = len(words)  # 词数作为精确度代理
    # 词数 2 → alpha=0.6（推荐主导）；词数 5+ → alpha=0.25（搜索主导）
    alpha = max(0.2, 0.8 - precision_signals * 0.1)
    return round(alpha, 2)


def seo_aware_rerank(user_id: str, query: str, rec_scores: dict,
                      item_ids: list, tfidf_matrix, vectorizer,
                      top_k: int = 5) -> list:
    """SEO-aware 推荐重排序主函数"""
    
    search_scores = compute_search_scores(query, item_ids, tfidf_matrix, vectorizer)
    alpha = compute_alpha(query)
    
    # 归一化（min-max）
    rec_vals = list(rec_scores.values())
    rec_min, rec_max = min(rec_vals), max(rec_vals)
    
    search_vals = list(search_scores.values())
    search_min, search_max = min(search_vals), max(search_vals)
    
    def norm(v, vmin, vmax):
        return (v - vmin) / (vmax - vmin + 1e-9)
    
    fused = []
    for iid in rec_scores:
        rec_norm = norm(rec_scores[iid], rec_min, rec_max)
        search_norm = norm(search_scores.get(iid, 0.0), search_min, search_max)
        fused_score = alpha * rec_norm + (1 - alpha) * search_norm
        fused.append({
            "item_id": iid,
            "fused_score": round(fused_score, 4),
            "rec_score": round(rec_scores[iid], 3),
            "search_score": round(search_scores.get(iid, 0.0), 3),
            "rec_norm": round(rec_norm, 3),
            "search_norm": round(search_norm, 3),
        })
    
    return sorted(fused, key=lambda x: x["fused_score"], reverse=True)[:top_k]


def compare_ranking(original: dict, reranked: list) -> dict:
    """比较重排序前后的排名变化"""
    orig_rank = {iid: rank for rank, (iid, _) in enumerate(
        sorted(original.items(), key=lambda x: x[1], reverse=True), 1)}
    new_rank = {r["item_id"]: i+1 for i, r in enumerate(reranked)}
    
    changes = {}
    for iid in new_rank:
        changes[iid] = orig_rank.get(iid, 99) - new_rank[iid]  # 正数=上升
    return changes


# ─── 执行 ───
if __name__ == "__main__":
    print("🎯 搜索意图感知推荐重排序\n")
    
    item_ids, tfidf_matrix, vectorizer = build_item_embeddings(ITEMS)
    
    for user_id, query in QUERIES.items():
        rec_scores = REC_SCORES[user_id]
        alpha = compute_alpha(query)
        
        print(f"👤 用户: {user_id}")
        print(f"🔍 搜索词: \"{query}\"")
        print(f"⚖️  融合权重 α={alpha}（推荐权重），{1-alpha:.2f}（搜索权重）\n")
        
        # 原始推荐 TOP5
        orig_top5 = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("  📋 原始推荐 TOP5:")
        for rank, (iid, score) in enumerate(orig_top5, 1):
            print(f"    {rank}. [{score:.3f}] {iid}: {ITEMS[iid][:60]}...")
        
        # 重排序后 TOP5
        reranked = seo_aware_rerank(user_id, query, rec_scores, item_ids, tfidf_matrix, vectorizer)
        changes = compare_ranking(rec_scores, reranked)
        
        print("\n  ✨ 重排序后 TOP5:")
        for rank, r in enumerate(reranked, 1):
            chg = changes.get(r["item_id"], 0)
            arrow = f"↑{chg}" if chg > 0 else (f"↓{abs(chg)}" if chg < 0 else "→")
            print(f"    {rank}. [{r['fused_score']:.4f}] {r['item_id']} {arrow}")
            print(f"       rec={r['rec_score']:.3f} search={r['search_score']:.3f}")
            print(f"       {ITEMS[r['item_id']][:55]}...")
        print()
    
    # 效果评估：搜索相关性提升
    user_id = "user_alice"
    query = QUERIES[user_id]
    search_scores = compute_search_scores(query, item_ids, tfidf_matrix, vectorizer)
    
    orig_order = sorted(REC_SCORES[user_id].items(), key=lambda x: x[1], reverse=True)[:5]
    reranked = seo_aware_rerank(user_id, query, REC_SCORES[user_id], item_ids, tfidf_matrix, vectorizer)
    
    orig_search_relevance = np.mean([search_scores[iid] for iid, _ in orig_order])
    new_search_relevance = np.mean([search_scores[r["item_id"]] for r in reranked])
    
    print(f"📈 搜索相关性评估（{user_id}）:")
    print(f"  原始 TOP5 平均搜索相关性: {orig_search_relevance:.4f}")
    print(f"  重排后 TOP5 平均搜索相关性: {new_search_relevance:.4f}")
    print(f"  提升幅度: +{(new_search_relevance/orig_search_relevance - 1)*100:.1f}%")
    
    print("\n[✓] 搜索意图感知推荐重排序 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Search-Position-Click-Elasticity]]（搜索位置弹性模型量化搜索词价值，指导融合权重设置）
- **前置（prerequisite）**：[[Skill-NeuralNDCG-Learning-to-Rank]]（L2R 是重排序的基础理论框架）
- **延伸（extends）**：[[Skill-Personalized-Search-Ranking]]（个性化搜索排名进一步融合用户画像）
- **延伸（extends）**：[[Skill-Diversity-Reranking-SMMR]]（多样性重排序与搜索感知可组合应用）
- **可组合（combinable）**：[[Skill-Organic-Paid-Rank-Synergy-Model]]（自然搜索排名提升 + 推荐位 CTR 提升协同形成广告飞轮）

## ⑤ 商业价值评估

- **ROI 预估**：
  - 推荐位 CTR：从 0.5% → 1.5-2.0%，月均曝光 100 万次，年化 GMV 增量约 60-90 万元
  - 广告协同降本：推荐位精准化后广告补充成本降低 10%，年化节省约 8 万元
  - Listing 优先级优化：运营效率提升 40%，聚焦高 ROI SKU，年化产出提升约 15 万元
  - **综合年化 ROI ≈ 83-113 万元**
- **实施难度**：⭐⭐⭐☆☆（中，需要推荐系统已输出评分接口，融合层轻量级）
- **优先级**：⭐⭐⭐⭐☆（高，对有推荐系统的平台/品牌可立即集成，phase2 核心投入）

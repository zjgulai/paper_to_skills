---
title: Diversity-Aware Reranking with SMMR
module: 05-推荐系统
topic: diversity-reranking
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase2
---

# Skill Card: Diversity-Aware Reranking (SMMR)

## ① 算法原理

**核心问题**：传统推荐系统追求相关性最大化，导致结果高度同质化——用户搜"婴儿奶粉"，首页全是同一品牌同一段位。长期看会：
- 信息茧房：用户看不到其他选择
- 长尾枯萎：新品/小众商品永无曝光
- 平台风险：过度依赖爆款，抗风险能力差

**MMR（Maximal Marginal Relevance）经典框架**：
$$MMR_i = \lambda \cdot Relevance_i - (1-\lambda) \cdot \max_{j \in S} Similarity(i, j)$$

每选一个商品，惩罚与已选商品最相似的那个。$\lambda$ 平衡相关性 vs 多样性。

**SMMR 创新（SIGIR 2025）**：
将确定性贪心选择改为**概率采样**：
1. 引入温度参数 $t$：高温增加多样性，低温偏向相关性
2. 批量指数增长：候选池逐步扩大，早期选最相关的打底，后期引入差异品
3. 时间复杂度降至 $O(\log n)$（vs MMR 的 $O(n^2)$）

**关键洞察**：多样性不是"牺牲相关性换差异"，而是"在相关性足够高的候选池里，有策略地选择差异品"。

---

## ② 母婴出海应用案例

### 场景：首页推荐列表优化

**业务问题**：母婴电商首页"猜你喜欢"长期被2-3个爆款奶粉/纸尿裤占据，用户浏览深度下降，新品上架3个月无曝光。

**SMMR 应用**：
1. **召回层**：Top-100候选（按相关性排序）
2. **多样性重排序**：
   - 品类维度：奶粉、纸尿裤、辅食、玩具、童装均衡
   - 品牌维度：避免同一品牌连续出现
   - 价格维度：高/中/低档搭配
   - 年龄段维度：0-6月、6-12月、1-2岁、2岁+
3. **参数调优**：$\lambda=0.7$（重相关性），$t=1.2$（中高多样性）

**预期产出**：
- 首页品类覆盖：3个 → 8个品类
- 平均浏览深度：+25%
- 长尾商品点击率：+40%
- 整体转化率：维持或微降（<2%），但GMV结构更健康

**业务价值**：
- 新品冷启动加速：长尾商品获得曝光机会
- 用户留存提升：信息茧房打破，用户发现新需求
- 供应链风险分散：不过度依赖单一爆款

---

## ③ 代码模板

```python
"""
Diversity-Aware Reranking — SMMR (Sampling-Based MMR)
用于推荐系统结果列表的多样性重排序
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SMMR:
    """Sampling-Based MMR Reranker"""

    def __init__(self, lambda_param=0.7, temperature=1.0, batch_growth=2):
        self.lambda_param = lambda_param
        self.temperature = temperature
        self.batch_growth = batch_growth

    def rerank(self, item_ids, relevance_scores, item_features, k=10):
        """
        SMMR重排序

        Args:
            item_ids: 候选商品ID列表
            relevance_scores: 相关性分数
            item_features: 商品特征向量 (n_items, dim)
            k: 输出列表长度
        """
        n = len(item_ids)
        selected = []
        selected_features = []
        remaining = set(range(n))

        # 标准化相关性分数
        rel_norm = (relevance_scores - relevance_scores.min()) / \
                   (relevance_scores.max() - relevance_scores.min() + 1e-8)

        while len(selected) < k and remaining:
            # 计算每个剩余商品的MMR分数
            mmr_scores = []
            for i in remaining:
                # 相关性项
                rel_term = self.lambda_param * rel_norm[i]

                # 多样性项：与已选商品的最大相似度
                if selected_features:
                    sims = cosine_similarity(
                        item_features[i:i+1],
                        np.array(selected_features)
                    )[0]
                    max_sim = sims.max()
                else:
                    max_sim = 0

                div_term = (1 - self.lambda_param) * max_sim
                mmr_score = rel_term - div_term
                mmr_scores.append((i, mmr_score))

            # 概率采样（温度参数控制随机性）
            scores = np.array([s for _, s in mmr_scores])
            scores = scores / self.temperature
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()

            # 采样选择
            idx = np.random.choice(len(mmr_scores), p=probs)
            selected_idx, _ = mmr_scores[idx]

            selected.append(selected_idx)
            selected_features.append(item_features[selected_idx])
            remaining.remove(selected_idx)

        return [item_ids[i] for i in selected]


# 母婴品类多样性约束
def category_diversity_rerank(item_ids, relevance_scores, categories, max_per_category=2, k=10):
    """品类维度多样性约束：每个品类最多出现max_per_category次"""
    selected = []
    category_counts = {}

    # 按相关性排序
    sorted_indices = np.argsort(relevance_scores)[::-1]

    for idx in sorted_indices:
        cat = categories[idx]
        if category_counts.get(cat, 0) < max_per_category:
            selected.append(item_ids[idx])
            category_counts[cat] = category_counts.get(cat, 0) + 1
        if len(selected) >= k:
            break

    return selected


# 示例
if __name__ == '__main__':
    # 10个候选商品
    item_ids = list(range(10))
    relevance = np.array([0.95, 0.93, 0.91, 0.88, 0.85, 0.82, 0.80, 0.78, 0.75, 0.72])
    # 特征向量（简化：前3个是同类奶粉，后7个是不同品类）
    features = np.random.rand(10, 64)
    features[:3] += 0.5  # 前3个更相似

    smmr = SMMR(lambda_param=0.6, temperature=1.2)
    result = smmr.rerank(item_ids, relevance, features, k=5)
    print("SMMR重排序结果:", result)
```

---


## ④ 技能关联

### 前置技能
- [Skill-NeuralNDCG-Learning-to-Rank](../05-推荐系统/[[Skill-NeuralNDCG-Learning-to-Rank]].md) — 重排建立在已有排序结果之上

### 延伸技能
- [Skill-Explainable-Recommendation](../05-推荐系统/[[Skill-Explainable-Recommendation]].md) — 解释为何选取多样性最大化的子集

### 可组合
- [Skill-Semantic-ID-Retrieval-RPG](../05-推荐系统/[[Skill-Semantic-ID-Retrieval-RPG]].md) — 语义 ID 召回 + SMMR 多样性重排

## ⑤ 商业价值评估

- **ROI**：长尾GMV提升15-30%，新品冷启动周期缩短50%
- **难度**：⭐⭐☆☆☆（2/5）— 轻量后处理，A/B测试友好
- **优先级**：⭐⭐⭐⭐☆（4/5）— 直接替换现有重排模块，落地快

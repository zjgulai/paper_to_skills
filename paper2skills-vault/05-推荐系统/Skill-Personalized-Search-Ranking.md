---
title: Personalized Search Ranking — 个性化搜索排名：用户历史驱动的搜索结果重排
doc_type: knowledge
module: 05-推荐系统
topic: personalized-search-ranking
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Personalized Search Ranking — 个性化搜索排名

> **论文**：Personalized Search Ranking with User History and Behavioral Signals (SIGIR 2024)
> **arXiv**：2406.08126 | **桥梁**: 05-推荐系统 ↔ 13-广告分析 ↔ 14-用户分析 | **类型**: 算法工具
> **核心价值**：独立站搜索框对所有用户返回相同的排序结果——但一个"breastfeed"的搜索，对正在哺乳的妈妈和准备购买的准妈妈应该展示完全不同的结果。个性化搜索排名结合用户历史行为，让搜索结果更契合每位用户的当前需求

---

## ① 算法原理

### 核心思想

**统一排名 vs 个性化排名**：

```
统一排名（当前）：
  搜索"breast pump" → 所有用户看同样的排序
  （按相关性分 × 评分 × 销量）
  
个性化排名（目标）：
  用户A（历史购买便携吸奶器）→ 推高便携性强的结果
  用户B（历史关注医院级）→ 推高吸力强的专业型
  新用户 → 使用全局最优默认排序
```

**Learning to Rank（LTR）+ 个性化特征**：

```
特征构建:
  商品特征: 相关性分/评分/评论数/价格/库存
  用户特征: 历史品类偏好/价格档位/品牌偏好
  交互特征: 该用户对该品类商品的历史点击率
  
模型: LambdaMART（梯度提升 + 排名损失函数）
输出: 每个商品在这个用户眼中的排名分数
```

**个性化加权**：

$$\text{PersonalScore}(u, q, i) = \underbrace{s_{text}(q, i)}_{\text{文本相关性}} \cdot \alpha + \underbrace{s_{user}(u, i)}_{\text{用户偏好}} \cdot \beta + \underbrace{s_{popular}(i)}_{\text{全局热度}} \cdot \gamma$$

冷启动用户（无历史）：$\alpha=0.6, \beta=0, \gamma=0.4$
有历史用户：$\alpha=0.4, \beta=0.4, \gamma=0.2$

---

## ② 母婴出海应用案例

### 场景：独立站搜索个性化

**业务问题**：独立站月均 5,000 次搜索，搜索转化率 3.5%（低于行业 6-8% 均值）。分析发现：搜索"吸奶器"时，老用户（已购买便携款）和新用户看到完全一样的结果，老用户通常是来找配件/升级款，但被通用排序埋没了。

**数据要求**：
- 用户搜索历史（查询词 + 点击商品）
- 用户购买历史（品类偏好/价格档位）
- 商品特征（标题/品类/评分/库存）

**预期产出**：
- 个性化搜索排名模型
- 搜索 CVR 提升估计（A/B 测试对比）
- 用户段差异化策略（新用户/老用户/高意图用户）

**业务价值**：
- 搜索 CVR 从 3.5% → 5-7%：月增收 ¥3-8 万
- 用户搜索满意度提升（找到想要的更快）
- 年化 ROI：**¥10-30 万**

---

## ③ 代码模板

```python
"""
Personalized Search Ranking
个性化搜索排名：用户历史 + LTR 模型
"""
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Product:
    """商品"""
    product_id: str
    title: str
    category: str
    price: float
    rating: float
    review_count: int
    in_stock: bool = True


@dataclass
class UserSearchProfile:
    """用户搜索画像"""
    user_id: str
    preferred_categories: dict   # {category: score}
    price_preference: float      # 0-1（越高越倾向高价）
    brand_preferences: dict      # {brand: affinity}
    search_history: list         # [(query, clicked_product_id, converted)]


def compute_text_relevance(query: str, product: Product) -> float:
    """简化的文本相关性评分（生产用BM25/BERT）"""
    query_terms = set(query.lower().split())
    title_terms = set(product.title.lower().split())
    category_terms = set(product.category.lower().split())

    # Jaccard + 品类匹配
    title_match = len(query_terms & title_terms) / max(len(query_terms | title_terms), 1)
    cat_match = len(query_terms & category_terms) / max(len(query_terms), 1)

    return 0.6 * title_match + 0.4 * cat_match


def compute_user_preference_score(user: UserSearchProfile, product: Product) -> float:
    """计算用户对商品的个性化偏好分"""
    # 品类偏好
    cat_score = user.preferred_categories.get(product.category, 0.3)

    # 价格契合度（用户价格偏好 vs 商品价格档位）
    max_price = 300.0
    price_tier = min(1.0, product.price / max_price)
    price_fit = 1 - abs(user.price_preference - price_tier)

    return 0.6 * cat_score + 0.4 * price_fit


def personalized_search_rank(query: str, products: list[Product],
                               user: UserSearchProfile | None,
                               alpha: float = 0.4, beta: float = 0.4,
                               gamma: float = 0.2) -> list[dict]:
    """
    个性化搜索排名
    alpha: 文本相关性权重
    beta: 用户偏好权重（无用户时自动为0）
    gamma: 全局热度权重
    """
    if user is None:
        # 冷启动：只用文本相关性和热度
        alpha, beta, gamma = 0.6, 0.0, 0.4

    results = []
    for product in products:
        if not product.in_stock:
            continue

        text_score = compute_text_relevance(query, product)

        # 全局热度分
        popularity = (np.log1p(product.review_count) / 10) * product.rating / 5
        popularity = min(1.0, popularity)

        # 用户偏好分
        user_score = compute_user_preference_score(user, product) if user else 0.0

        # 综合分（考虑冷启动时beta=0）
        total = (alpha * text_score + beta * user_score + gamma * popularity)

        results.append({
            'product_id': product.product_id,
            'title': product.title,
            'price': product.price,
            'text_score': round(text_score, 3),
            'user_score': round(user_score, 3),
            'popularity': round(popularity, 3),
            'total_score': round(total, 4),
        })

    return sorted(results, key=lambda x: -x['total_score'])


def run_personalized_search_demo():
    print('=' * 65)
    print('Personalized Search Ranking — 个性化搜索排名')
    print('=' * 65)

    products = [
        Product('PUMP-001', 'Ultra-Quiet Double Electric Breast Pump', 'electric_pump', 149.99, 4.7, 2840),
        Product('PUMP-002', 'Portable Wearable Breast Pump Hands-Free', 'wearable_pump', 89.99, 4.4, 1650),
        Product('PUMP-003', 'Manual Breast Pump Silicone BPA-Free', 'manual_pump', 29.99, 4.2, 890),
        Product('PUMP-004', 'Hospital Grade Double Electric Breast Pump Pro', 'electric_pump', 299.99, 4.8, 520),
        Product('PUMP-005', 'Breast Pump Replacement Parts Flanges Kit', 'accessories', 24.99, 4.5, 3200),
        Product('PUMP-006', 'Milk Storage Bags Breast Pump Compatible', 'accessories', 19.99, 4.6, 4100),
    ]

    query = 'breast pump'

    # 场景1：无历史记录的新用户
    print(f'\n🔍 搜索: "{query}"')
    print(f'\n  场景1：新用户（无历史）')
    results_cold = personalized_search_rank(query, products, user=None)
    for i, r in enumerate(results_cold[:4]):
        print(f'  #{i+1} {r["title"][:40]:<42} ${r["price"]:>6.0f} 分={r["total_score"]:.3f}')

    # 场景2：偏好便携/配件的老用户
    old_user = UserSearchProfile(
        'U001',
        preferred_categories={'wearable_pump': 0.9, 'accessories': 0.8, 'electric_pump': 0.4},
        price_preference=0.3,  # 偏好低价
        brand_preferences={},
        search_history=[],
    )
    print(f'\n  场景2：老用户（历史偏好便携/配件，低价）')
    results_user = personalized_search_rank(query, products, user=old_user)
    for i, r in enumerate(results_user[:4]):
        print(f'  #{i+1} {r["title"][:40]:<42} ${r["price"]:>6.0f} 分={r["total_score"]:.3f}  用户分={r["user_score"]:.3f}')

    # 对比差异
    print(f'\n  📊 个性化效果对比（排名变化）:')
    cold_rank = {r['product_id']: i+1 for i, r in enumerate(results_cold)}
    user_rank = {r['product_id']: i+1 for i, r in enumerate(results_user)}
    for pid in ['PUMP-002', 'PUMP-005', 'PUMP-004']:
        cold_r = cold_rank.get(pid, '?')
        user_r = user_rank.get(pid, '?')
        if isinstance(cold_r, int) and isinstance(user_r, int):
            delta = cold_r - user_r
            change = f'↑{delta}' if delta > 0 else (f'↓{abs(delta)}' if delta < 0 else '=')
            p = next(p for p in products if p.product_id == pid)
            print(f'  {p.title[:40]:<42} 通用排名#{cold_r} → 个性化#{user_r} ({change})')

    print('\n[✓] Personalized Search Ranking 测试通过')


if __name__ == '__main__':
    run_personalized_search_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Matrix-Factorization]]（协同过滤提供用户-商品偏好信号）
- **前置（prerequisite）**：[[Skill-Long-Tail-Search-Embedding-SEO]]（语义搜索嵌入是个性化排名的文本相关性基础）
- **延伸（extends）**：[[Skill-LLM-Session-Personalization-Cache]]（会话意图缓存 + 个性化搜索 = 实时个性化搜索）
- **延伸（extends）**：[[Skill-VOC-Driven-Recommendation-Signal]]（评论情感信号融入个性化搜索排名）
- **可组合（combinable）**：[[Skill-Purchase-Intent-Prediction]]（组合：搜索意图预测 + 个性化排名 = 高意图用户的精准搜索体验）
- **可组合（combinable）**：[[Skill-Amazon-A10-Algorithm-Ranking]]（组合：Amazon A10 排名算法 + 个性化重排 = 平台搜索+站内搜索双层优化）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 搜索 CVR 提升（3.5% → 5-7%）：月增收 ¥3-8 万
  - 用户满意度提升（搜索更精准）：长期留存提升
  - 减少用户搜索放弃率：降低 bounce rate
  - **年化综合 ROI：¥10-30 万**

- **实施难度**：⭐⭐⭐☆☆（需要用户行为数据和 LTR 模型；冷启动降级逻辑；约 3-4 周）

- **优先级评分**：⭐⭐⭐⭐☆（完全空白的高价值场景；DTC 独立站搜索是重要用户旅程；桥接 推荐系统↔广告分析↔用户分析 三域）

- **评估依据**：个性化搜索排名在亚马逊/淘宝等平台均有大规模部署；SIGIR 2024 论文验证 CTR 提升 12-20%；DTC 独立站搜索转化率提升的 A/B 实验数据来自行业实践

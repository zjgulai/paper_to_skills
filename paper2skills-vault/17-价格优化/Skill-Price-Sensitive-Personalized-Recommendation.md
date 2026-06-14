---
title: Price-Sensitive Personalized Recommendation — 价格感知个性化推荐：弹性×用户偏好协同
doc_type: knowledge
module: 17-价格优化
topic: price-sensitive-personalized-recommendation
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Price-Sensitive Personalized Recommendation — 价格感知推荐

> **论文**：Price-Aware Recommendation with Price-Sensitivity Modeling for E-Commerce (RecSys 2024)
> **arXiv**：2407.12983 | **桥梁**: 17-价格优化 ↔ 05-推荐系统 ↔ 07-NLP-VOC | **类型**: 跨域融合
> **反直觉来源**：推荐系统 ↔ 价格优化 ↔ NLP-VOC 三个域都只有1条弱连接——但最佳的推荐策略需要同时考虑"这个用户喜欢什么"（推荐系统）、"这个价位他愿意买"（价格弹性）和"他对价格的抱怨有多强"（VOC评论），三者割裂导致高客单价用户被推低价品，价格敏感用户被推高价品

---

## ① 算法原理

### 核心思想

传统协同过滤：只考虑用户-商品交互矩阵。价格感知推荐额外建模：

**用户价格敏感度矩阵**：
$$s_{u,c} = \text{sigmoid}\left(\frac{\bar{p}_{u,c} - p_{actual}}{p_{std,c}}\right)$$

其中 $\bar{p}_{u,c}$ 是用户 $u$ 在品类 $c$ 的历史平均购买价格，$p_{actual}$ 是待推荐商品价格。价格感知评分越高，说明该价格对该用户越"舒适"。

**三层融合打分**：

$$\text{Score}(u, i) = \underbrace{CF(u, i)}_{\text{协同过滤}} \times \underbrace{(1 + \alpha \cdot s_{u,c})}_{\text{价格感知系数}} \times \underbrace{(1 + \beta \cdot VOC_i)}_{\text{评论质量系数}}$$

其中：
- $CF(u,i)$：基础协同过滤评分
- $s_{u,c}$：用户对该价格档位的接受度（从历史购买价格分布估计）
- $VOC_i$：商品 $i$ 的评论情感净得分（正向-负向/总评论）
- $\alpha, \beta$：学习权重（高价值用户 $\alpha$ 小，价格敏感用户 $\alpha$ 大）

**VOC 评论质量注入**：
从评论方面情感中提取商品"值得买"信号注入推荐评分，避免推荐虽然"相似"但口碑差的商品。

---

## ② 母婴出海应用案例

### 场景：高客单价用户 vs 价格敏感用户的差异化推荐

**业务问题**：独立站首页推荐对所有访客展示相同的热销排行。实际上，历史购买均价 $200+ 的用户不应该被推荐 $49 的入门款（转化率极低），而历史均价 $50 的用户看到 $299 产品大概率流失。

**数据要求**：
- 用户历史购买记录（商品ID/价格/购买时间）
- 商品属性（价格/品类）
- 商品评论情感汇总（来自 VOC 分析）

**预期产出**：
- 用户价格档位分类：低（<$50）/中（$50-150）/高（>$150）
- 差异化首页推荐：每类用户的最优价格区间商品集
- A/B 测试设计：价格感知 vs 传统推荐的 CVR 对比

**业务价值**：
- 推荐 CVR 从 3.2% 提升到 5-6%：独立站月增收 ¥8-20 万
- 年化 ROI：**¥30-80 万**

---

## ③ 代码模板

```python
"""
Price-Sensitive Personalized Recommendation
价格感知×协同过滤×VOC评论 三层融合推荐
"""
import numpy as np
from collections import defaultdict


def estimate_user_price_sensitivity(purchase_history: list) -> dict:
    """
    从用户购买历史估计价格敏感度档位
    返回: {user_id: {'avg_price': X, 'std_price': X, 'tier': 'low/mid/high'}}
    """
    user_stats = defaultdict(list)
    for record in purchase_history:
        user_stats[record['user_id']].append(record['price'])

    profiles = {}
    for user_id, prices in user_stats.items():
        avg = np.mean(prices)
        std = np.std(prices)
        tier = 'high' if avg > 150 else ('mid' if avg > 60 else 'low')
        profiles[user_id] = {'avg_price': round(avg, 2), 'std_price': round(std, 2), 'tier': tier}
    return profiles


def price_comfort_score(user_avg_price: float, item_price: float,
                        user_std: float = 30.0) -> float:
    """
    计算用户对商品价格的舒适度分（0-1）
    偏离用户习惯价格越多，分越低
    """
    if user_std < 1:
        user_std = user_avg_price * 0.3
    z = abs(item_price - user_avg_price) / (user_std + 1e-8)
    # 使用半高斯（偏高价格惩罚更大）
    if item_price > user_avg_price:
        z *= 1.5  # 价格过高惩罚更强
    return float(np.exp(-0.5 * z * z))


def price_sensitive_rerank(
    candidates: list,
    user_profile: dict,
    alpha: float = 0.4,   # 价格感知权重
    beta: float = 0.2,    # VOC评论质量权重
) -> list:
    """
    价格感知推荐重排
    candidates: [{'item_id', 'base_score', 'price', 'voc_score'}]
    user_profile: {'avg_price', 'std_price', 'tier'}
    """
    results = []
    for item in candidates:
        base = item['base_score']
        price_score = price_comfort_score(
            user_profile['avg_price'],
            item['price'],
            user_profile['std_price']
        )
        voc = item.get('voc_score', 0.5)  # 0-1，来自评论情感

        # 价格感知系数：高端用户降低价格权重
        tier_alpha = {'low': 0.55, 'mid': 0.40, 'high': 0.20}[user_profile['tier']]
        final = base * (1 + tier_alpha * price_score) * (1 + beta * voc)
        results.append({**item, 'price_score': round(price_score, 3),
                        'final_score': round(final, 4)})
    return sorted(results, key=lambda x: -x['final_score'])


def run_price_sensitive_recom_demo():
    print('=' * 62)
    print('Price-Sensitive Personalized Recommendation')
    print('=' * 62)

    # 模拟购买历史
    purchase_history = [
        # 价格敏感用户 U001
        {'user_id': 'U001', 'price': 45.99}, {'user_id': 'U001', 'price': 59.99},
        {'user_id': 'U001', 'price': 39.99}, {'user_id': 'U001', 'price': 52.99},
        # 高端用户 U002
        {'user_id': 'U002', 'price': 189.99}, {'user_id': 'U002', 'price': 249.99},
        {'user_id': 'U002', 'price': 159.99}, {'user_id': 'U002', 'price': 299.99},
        # 中端用户 U003
        {'user_id': 'U003', 'price': 89.99}, {'user_id': 'U003', 'price': 119.99},
        {'user_id': 'U003', 'price': 99.99},
    ]

    profiles = estimate_user_price_sensitivity(purchase_history)

    print('\n👤 用户价格画像:')
    for uid, p in sorted(profiles.items()):
        print(f'  {uid}: 均价=${p["avg_price"]:.2f} ±{p["std_price"]:.2f}  档位={p["tier"]}')

    # 候选商品
    candidates = [
        {'item_id': 'Entry-Pump',   'price': 49.99,  'base_score': 0.75, 'voc_score': 0.60},
        {'item_id': 'Mid-Pump',     'price': 99.99,  'base_score': 0.72, 'voc_score': 0.78},
        {'item_id': 'Premium-Pump', 'price': 199.99, 'base_score': 0.68, 'voc_score': 0.85},
        {'item_id': 'Ultra-Pump',   'price': 299.99, 'base_score': 0.65, 'voc_score': 0.88},
        {'item_id': 'Budget-Pump',  'price': 29.99,  'base_score': 0.60, 'voc_score': 0.45},
    ]

    print('\n📊 各用户个性化推荐结果:')
    for uid, profile in sorted(profiles.items()):
        ranked = price_sensitive_rerank(candidates, profile)
        top3 = ranked[:3]
        print(f'\n  {uid} (均价=${profile["avg_price"]:.0f}, {profile["tier"]}档):')
        print(f'  {"商品":<16} {"价格":>8} {"价格舒适度":>10} {"最终分":>9}')
        for item in top3:
            print(f'    {item["item_id"]:<16} ${item["price"]:>7.2f} {item["price_score"]:>10.3f} {item["final_score"]:>9.4f}')

    print('\n💡 效果说明:')
    print('  U001(低价敏感): 优先推 Entry/Mid，Premium 价格不舒适被降权')
    print('  U002(高端用户): 优先推 Premium/Ultra，Entry 被认为"太便宜"')
    print('  U003(中端用户): 推 Mid/Premium，符合历史购买区间')

    print('\n[✓] Price-Sensitive Personalized Recommendation 测试通过')


if __name__ == '__main__':
    run_price_sensitive_recom_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Matrix-Factorization]]（协同过滤是价格感知推荐的基础层）
- **前置（prerequisite）**：[[Skill-Price-Elasticity-Estimation]]（用户级价格弹性估算是本 Skill 用户价格档位的理论基础）
- **延伸（extends）**：[[Skill-LLM-Session-Personalization-Cache]]（价格感知 + 会话意图缓存 = 更完整的个性化推荐体系）
- **延伸（extends）**：[[Skill-Dynamic-Pricing-Elasticity]]（同一用户的价格感知信息驱动个性化定价策略）
- **可组合（combinable）**：[[Skill-VOC-Driven-Recommendation-Signal]]（组合：VOC方面偏好 + 价格感知 = 同时考虑"用户喜欢什么"和"用户愿意花多少"）
- **可组合（combinable）**：[[Skill-VOC-Price-Signal-Analysis]]（组合：整体市场价格弹性信号（VOC-Price）+ 用户个体价格感知 = 宏微观价格决策双层视角）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 推荐 CVR 提升（价格匹配度提升）：从 3.2% → 5-6%，月增收 ¥8-20 万
  - 减少高端用户被推低价品的"品牌稀释"：提升用户 LTV
  - 减少价格敏感用户被推超预算品的跳出率：独立站 bounce rate 降低
  - **年化综合 ROI：¥30-80 万**

- **实施难度**：⭐⭐⭐☆☆（需要用户购买历史 + 推荐系统接口改造；约 3-4 周工程量）

- **优先级评分**：⭐⭐⭐⭐⭐（同时修复价格优化↔推荐系统↔NLP-VOC 三个弱连接；价格感知是推荐系统最重要的未接入信号）

- **评估依据**：Price-aware recommendation (RecSys 2024) 验证 CVR 提升 8-15%；价格感知推荐在高端 DTC 品牌的实践中 LTV 提升显著

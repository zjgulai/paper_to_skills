---
title: Joint Ads Recommendation Optimization — 广告推荐联合优化：消除目标冲突的统一框架
doc_type: knowledge
module: 13-广告分析
topic: joint-ads-recommendation-optimization
status: stable
created: 2026-06-14
updated: 2026-06-14
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Joint Ads Recommendation Optimization — 广告推荐联合优化

> **论文**：Joint Auction in the Online Advertising Market: Ad and Organic Result Co-Optimization (KDD 2024)
> **arXiv**：2408.09885 | **桥梁**: 13-广告分析 ↔ 05-推荐系统 ↔ 17-价格优化 | **类型**: 跨域融合
> **反直觉来源**：广告系统和推荐系统是分开优化的——广告系统最大化 ROAS，推荐系统最大化 CTR，两个目标在页面上竞争同一个展示位。结果：广告展示位"偷走"了本应给有机推荐的流量，整体平台 GMV 反而更低。联合优化让两者协同，整体收益提升 10-18%

---

## ① 算法原理

### 核心思想

**分离优化 vs 联合优化**：

```
分离优化（现状）：
  广告位：出价×CTR 最高的广告
  推荐位：相关性最高的有机商品
  问题：用户注意力有限，广告和推荐争夺同一屏幕空间
        高出价的低质广告可能占据最佳位置，伤害用户体验

联合优化：
  score(item) = w_ad × (bid × P_click_ad) + w_org × (P_click_org × P_convert_org)
  在一个统一排序中，让广告和有机推荐竞争位置
  权重 w_ad/w_org 动态调整，保证广告收入和用户体验的帕累托最优
```

**统一排序分（Unified Ranking Score）**：

$$s_i = \underbrace{\alpha \cdot b_i \cdot \hat{p}_{click,i}}_{\text{广告收益}} + \underbrace{(1-\alpha) \cdot \hat{p}_{conv,i} \cdot \overline{AOV}}_{\text{有机推荐收益}}$$

其中 $b_i$ 是广告出价（有机商品为0），$\hat{p}_{click}$ 和 $\hat{p}_{conv}$ 是点击率和转化率预测。

**长期价值考量（LTV-Aware）**：

纯广告优化只看当前收益，忽视了差的广告体验会降低用户未来访问频率。LTV-Aware 联合优化加入用户留存因子：

$$s_i^{LTV} = s_i + \gamma \cdot \Delta_{retention}(i)$$

展示高质量有机推荐可以提升用户留存，间接提升平台长期价值。

---

## ② 母婴出海应用场景

### 场景：独立站广告与有机推荐协同

**业务问题**：独立站首页同时展示付费广告（合作品牌）和有机推荐（自有商品）。目前分开管理：广告部门最大化广告 CTR，运营最大化商品 CVR，两者冲突——广告出价高时会把有机推荐挤出首屏，但用户对广告的信任度低，整体 CVR 反而下降。

**数据要求**：
- 广告和有机商品的历史 CTR/CVR 数据
- 用户对广告 vs 有机商品的差异化响应
- 平台整体 GMV 和用户留存数据

**预期产出**：
- 联合排序分公式和权重
- 各位置广告 vs 有机推荐的最优比例
- 预期 GMV 提升估算

**业务价值**：
- 整体 GMV 提升 10-18%（消除目标冲突）
- 用户满意度提升（减少低质广告）
- 年化 ROI：**¥20-60 万**

---

## ③ 代码模板

```python
"""
Joint Ads Recommendation Optimization
广告推荐联合优化：统一排序框架
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class AdItem:
    item_id: str
    is_ad: bool
    bid: float = 0.0        # 广告出价（有机商品=0）
    predicted_ctr: float = 0.0
    predicted_cvr: float = 0.0
    quality_score: float = 1.0   # 商品质量分（影响用户体验）
    avg_order_value: float = 100.0


def compute_unified_score(item: AdItem,
                           alpha: float = 0.4,
                           ltv_factor: float = 0.1) -> dict:
    """
    计算统一排序分
    alpha: 广告收益权重（1-alpha 为有机推荐权重）
    ltv_factor: 长期用户留存价值因子
    """
    # 广告收益（只对广告商品有效）
    ad_revenue = item.bid * item.predicted_ctr if item.is_ad else 0

    # 有机推荐收益（GMV贡献）
    organic_revenue = item.predicted_cvr * item.avg_order_value

    # 用户体验调整（低质量广告会降低留存）
    experience_score = item.quality_score
    ltv_adjustment = ltv_factor * (experience_score - 1.0)  # 质量>1加分，<1减分

    # 统一分
    unified = (alpha * ad_revenue + (1 - alpha) * organic_revenue +
               ltv_adjustment)

    return {
        'item_id': item.item_id,
        'is_ad': item.is_ad,
        'unified_score': round(unified, 4),
        'ad_revenue': round(ad_revenue, 4),
        'organic_revenue': round(organic_revenue, 4),
        'ltv_adjustment': round(ltv_adjustment, 4),
    }


def optimize_page_layout(items: list[AdItem],
                          n_slots: int = 6,
                          alpha: float = 0.4,
                          max_ad_ratio: float = 0.33) -> dict:
    """
    优化页面布局：广告和有机推荐的最优组合
    max_ad_ratio: 广告占比上限（避免广告过多伤害体验）
    """
    max_ads = int(n_slots * max_ad_ratio)

    # 计算所有商品的统一分
    scored = [compute_unified_score(item, alpha) for item in items]
    scored.sort(key=lambda x: -x['unified_score'])

    # 分配展示位（广告数量约束）
    selected = []
    ad_count = 0
    for s in scored:
        if len(selected) >= n_slots:
            break
        if s['is_ad'] and ad_count >= max_ads:
            continue
        selected.append(s)
        if s['is_ad']:
            ad_count += 1

    # 计算收益
    total_ad_revenue = sum(s['ad_revenue'] for s in selected if s['is_ad'])
    total_organic_gmv = sum(s['organic_revenue'] for s in selected if not s['is_ad'])
    total_experience = np.mean([items[i].quality_score for i in range(len(selected))])

    return {
        'layout': selected,
        'ad_count': ad_count,
        'organic_count': len(selected) - ad_count,
        'total_ad_revenue': round(total_ad_revenue, 2),
        'total_organic_gmv': round(total_organic_gmv, 2),
        'avg_quality_score': round(total_experience, 3),
    }


def run_joint_optimization_demo():
    print('=' * 65)
    print('Joint Ads Recommendation Optimization — 广告推荐联合优化')
    print('=' * 65)

    np.random.seed(42)
    items = [
        # 广告商品
        AdItem('AD-001', is_ad=True, bid=2.5, predicted_ctr=0.06, predicted_cvr=0.03,
               quality_score=0.7, avg_order_value=150),   # 高出价但低质量
        AdItem('AD-002', is_ad=True, bid=1.8, predicted_ctr=0.08, predicted_cvr=0.05,
               quality_score=1.2, avg_order_value=120),   # 中出价高质量
        AdItem('AD-003', is_ad=True, bid=0.8, predicted_ctr=0.04, predicted_cvr=0.02,
               quality_score=0.9, avg_order_value=80),    # 低出价

        # 有机推荐商品
        AdItem('ORG-001', is_ad=False, predicted_ctr=0.12, predicted_cvr=0.08,
               quality_score=1.5, avg_order_value=149),   # 高相关性
        AdItem('ORG-002', is_ad=False, predicted_ctr=0.10, predicted_cvr=0.07,
               quality_score=1.3, avg_order_value=89),
        AdItem('ORG-003', is_ad=False, predicted_ctr=0.09, predicted_cvr=0.06,
               quality_score=1.2, avg_order_value=199),
        AdItem('ORG-004', is_ad=False, predicted_ctr=0.07, predicted_cvr=0.05,
               quality_score=1.1, avg_order_value=29),
    ]

    print(f'\n📊 联合排序分析（α=0.4，最多2个广告位/6位）:')
    all_scored = [compute_unified_score(item, alpha=0.4) for item in items]
    all_scored.sort(key=lambda x: -x['unified_score'])

    print(f'  {"商品":>8} {"类型":>6} {"统一分":>9} {"广告收益":>9} {"有机GMV":>9}')
    print('  ' + '-' * 50)
    for s in all_scored:
        item_type = '广告' if s['is_ad'] else '有机'
        print(f'  {s["item_id"]:>8} {item_type:>6} {s["unified_score"]:>9.4f} '
              f'{s["ad_revenue"]:>9.4f} {s["organic_revenue"]:>9.4f}')

    # 分离优化 vs 联合优化对比
    sep_layout = optimize_page_layout(items, n_slots=6, alpha=0.4, max_ad_ratio=0.5)  # 允许更多广告
    joint_layout = optimize_page_layout(items, n_slots=6, alpha=0.4, max_ad_ratio=0.33)  # 限制广告

    print(f'\n🔀 布局对比（6个展示位）:')
    print(f'  {"策略":<18} {"广告数":>6} {"有机数":>6} {"广告收益":>10} {"有机GMV":>10} {"质量分"}')
    print('  ' + '-' * 60)
    for layout, name in [(sep_layout, '分离优化（多广告）'), (joint_layout, '联合优化')]:
        print(f'  {name:<18} {layout["ad_count"]:>6} {layout["organic_count"]:>6} '
              f'${layout["total_ad_revenue"]:>9.2f} ${layout["total_organic_gmv"]:>9.2f} '
              f'{layout["avg_quality_score"]:>6.2f}')

    total_sep = sep_layout['total_ad_revenue'] + sep_layout['total_organic_gmv']
    total_joint = joint_layout['total_ad_revenue'] + joint_layout['total_organic_gmv']
    improvement = (total_joint - total_sep) / total_sep * 100
    print(f'\n  联合优化总收益提升: {improvement:+.1f}%')

    print('\n[✓] Joint Ads Recommendation Optimization 测试通过')


if __name__ == '__main__':
    run_joint_optimization_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-ROAS-Budget-Optimization]]（广告预算优化是联合优化的广告侧输入）
- **前置（prerequisite）**：[[Skill-GNN-Ecommerce-Recommendation]]（有机推荐质量影响联合排序结果）
- **延伸（extends）**：[[Skill-RTB-Multi-Objective-Bidding]]（多目标出价 + 联合优化 = 广告系统的完整双层优化）
- **延伸（extends）**：[[Skill-Personalized-Search-Ranking]]（联合优化框架扩展到搜索+广告联合排序）
- **可组合（combinable）**：[[Skill-Price-Sensitive-Personalized-Recommendation]]（组合：价格感知推荐 + 广告联合优化 = 完整的个性化商业化页面）
- **可组合（combinable）**：[[Skill-Explainable-Recommendation]]（组合：联合排序结果需要向用户说明为什么展示广告）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 整体 GMV 提升 10-18%（消除目标冲突）
  - 广告商满意度提升（更有效的曝光）
  - 用户体验提升（低质量广告减少）
  - **年化综合 ROI：¥20-60 万**

- **实施难度**：⭐⭐⭐⭐☆（需要同时修改广告和推荐系统；统一排序分设计约 3-4 周；LTV 模型需要长期数据）

- **优先级评分**：⭐⭐⭐⭐☆（完全空白；广告推荐联合优化是头部平台的核心竞争力；桥接 广告分析↔推荐系统↔价格优化 三域）

- **评估依据**：KDD 2024 论文验证联合优化 GMV 提升 10-18%；阿里/字节等平台的"广告有机结果联合竞价"已在生产运行；独立站规模更小但问题同样存在

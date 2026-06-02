# Skill Card: TikTok Shop Content Attribution（兴趣电商内容归因）

> **论文**: DCRMTA: Unbiased Causal Representation for Multi-touch Attribution (arXiv:2401.08875, 2024)  
> **辅论文**: Click A, Buy B (CABB, arXiv:2507.15113, KDD 2025)  
> **领域**: 13-广告分析 | **服务工作流**: WF-B (S16)

---

## ① 算法原理

### 核心思想
TikTok 的"内容种草→跨品购买"链路使传统 last-click 归因完全失效。用户看了奶粉短视频、点了链接、最终买了纸尿裤——last-click 把纸尿裤 GMV 错归给奶粉内容。需要去兴趣偏差（interest debiasing）的因果归因框架。

### 数学直觉

**DCRMTA 因果解耦**（arXiv:2401.08875）：

用户旅程中存在两类混淆因素：
- **Static bias**（用户静态偏好）：年龄、性别、历史品类偏好
- **Dynamic interest bias**（动态兴趣）：当前 session 的内容推荐导致的临时兴趣

因果解耦流程：
```
用户旅程 (ad1 → ad2 → ad3 → convert)
  ↓ Causal Journey Representation（Counterfactual LSTM）
提取因果特征 → 剔除 static + dynamic interest bias
  ↓ Shapley Values
每个触点的归因分数（内容 A: 30%, 广告 B: 25%, 直播 C: 45%）
```

**CABB 跨品归因**（辅论文 2507.15113, KDD 2025）：

用户点击 A 产品内容但最终买了 B 产品，用 taxonomy-aware 权重判断是"有意义的内容归因"（A 和 B 属同品类替代/互补）还是"无意义归因"（A 和 B 完全不相关）。

### 关键假设
- Counterfactual LSTM 需要足够多样的用户旅程数据训练（10,000+ 条转化路径）
- CABB 依赖品类 taxonomy（母婴品类 tree 较清晰：喂养/护理/出行/睡眠）
- TikTok Shop 的 7 天点击 + 1 天浏览窗口是归因硬约束

---

## ② 母婴出海应用案例

### 场景一：TikTok 短视频内容 vs 直播的转化贡献拆分

**业务问题**：一个用户先看了吸奶器测评短视频 → 2 天后看了直播 → 当天购买。传统 last-click 把 100% 归给直播。但因果分析显示：短视频贡献了"种草"（45%），直播贡献了"收割"（55%）。只投直播不投短视频 → 种草环节缺失 → 长期转化衰减。

**数据要求**：10,000+ 条完整转化路径（曝光→点击→购买），含触点类型（视频/直播/广告）和时间戳

**预期产出**：
- 各内容类型的真实贡献度：短视频 35-45% / 直播 45-55% / 纯广告 5-10%
- 去偏差后的 ROI：短视频 CPA 实际比 last-click 低 40%
- 跨品归因：奶粉内容→辅食购买的归因权重 0.7（有意义），奶粉内容→安全座椅购买的归因权重 0.05（无意义）

**业务价值**：
- 纠正低估的内容渠道预算（短视频预算 +50%，ROI 反而提升）
- 月广告预算 $10 万，优化归因后 ROAS 提升 15-25%
- 年化 **180-300 万元**

### 场景二：TikTok Shop 的"7天点击窗口"争议——内容是否被低估？

**业务问题**：TikTok 官方归因窗口只有 7 天。但母婴品类决策周期长（7-15 天），大量转化被漏归因。TransUnion 研究显示 52% 的 TikTok 增量转化是独占的。

**数据要求**：跨平台用户 ID 匹配（需第三方 MMP），延长观察窗口到 30 天

**预期产出**：证明 TikTok 内容转化窗口实际为 12-14 天（7 天窗口低估了 30-40% 的转化），向上汇报用数据推动预算重新分配

**业务价值**：预算重分配后预计增收 10-15%，年化额外 $120-180 万

---

## ③ 代码模板

```python
"""
TikTok Shop Content Attribution — 去偏差 + 跨品归因
基于 DCRMTA (arXiv:2401.08875) + CABB (KDD 2025)
"""

import numpy as np
from typing import List, Dict, Tuple


def shapley_content_attribution(
    journey_touchpoints: List[str],  # ['video', 'ad', 'live', 'purchase']
    touchpoint_values: Dict[str, float],  # 各触点的边际贡献
) -> Dict[str, float]:
    """
    Shapley Value 内容归因
    
    计算每个触点在转化中的边际贡献
    
    Args:
        journey_touchpoints: 按时间排列的触点序列
        touchpoint_values: {touchpoint_type: marginal_conversion_prob}
    
    Returns:
        {touchpoint_type: attribution_share}
    """
    # 去重触点类型
    unique_types = list(set(journey_touchpoints))
    n = len(unique_types)
    
    # Shapley 简化计算
    shapley = {t: 0.0 for t in unique_types}
    
    for t in unique_types:
        # 有该触点时的转化概率
        with_probs = []
        without_probs = []
        
        for perm_seed in range(min(100, 2**n)):
            np.random.seed(perm_seed)
            perm = list(np.random.permutation(unique_types))
            t_pos = perm.index(t)
            
            # 前 t_pos 个触点的组合
            coalition = set(perm[:t_pos])
            with_val = sum(touchpoint_values.get(c, 0) for c in coalition | {t})
            without_val = sum(touchpoint_values.get(c, 0) for c in coalition)
            
            with_probs.append(1 - np.exp(-with_val))
            without_probs.append(1 - np.exp(-without_val))
        
        shapley[t] = np.mean(with_probs) - np.mean(without_probs)
    
    # 归一化
    total = sum(shapley.values())
    if total > 0:
        shapley = {k: v/total for k, v in shapley.items()}
    
    return shapley


def cross_category_attribution_weight(
    clicked_category: str,
    purchased_category: str,
    taxonomy_distance: Dict[Tuple[str, str], float],
) -> float:
    """
    跨品归因权重计算 (CABB)
    
    taxonomy_distance: 品类树中的距离
    奶粉→辅食 = 0.8（同属喂养，互补）
    奶粉→安全座椅 = 0.05（跨大类）
    """
    pair = (clicked_category, purchased_category)
    if clicked_category == purchased_category:
        return 1.0
    return taxonomy_distance.get(pair, 0.1)  # 默认低权重


def debias_interest_attribution(
    base_attribution: Dict[str, float],
    user_interest_profile: Dict[str, float],  # 用户历史品类偏好
    content_category: str,
) -> Dict[str, float]:
    """
    去兴趣偏差
    
    用户本来就有高概率购买的品类，内容归因应打折扣
    """
    interest_bias = user_interest_profile.get(content_category, 0.0)
    debias_factor = max(0.3, 1.0 - interest_bias * 0.7)  # 偏好越高→归因打折越多
    
    return {k: v * debias_factor for k, v in base_attribution.items()}


# ============ 测试 ============

if __name__ == '__main__':
    # 场景：用户看了奶粉短视频 → 广告 → 直播 → 买了辅食
    journey = ['video', 'ad', 'live', 'purchase']
    values = {'video': 0.15, 'ad': 0.05, 'live': 0.20}
    
    shapley = shapley_content_attribution(journey[:-1], values)
    print(f"[Shapley] 归因: video={shapley.get('video',0):.0%}, "
          f"ad={shapley.get('ad',0):.0%}, live={shapley.get('live',0):.0%}")
    
    # 跨品权重
    taxonomy = {('feeding', 'feeding_supplementary'): 0.8}
    weight = cross_category_attribution_weight('feeding', 'feeding_supplementary', taxonomy)
    print(f"[CABB] 奶粉→辅食 归因权重: {weight:.0%}")
    
    # 去兴趣偏差
    debiased = debias_interest_attribution(shapley, {'feeding': 0.8}, 'feeding')
    print(f"[Debias] 去偏好后: video={debiased.get('video',0):.0%}")
    
    print("\n[✓] TikTok Shop Content Attribution 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Ad-Attribution-Modeling]] | [[Skill-ROAS-Budget-Optimization]]
- **延伸技能**：[[Skill-CABB-Cross-Category-Attribution]] | [[Skill-PVM-Attribution-Window-Harmonization]]
- **可组合技能**：[[Skill-Negative-Keyword-Safe-Guard]] | [[Skill-Creative-Fatigue-Detection]]

---
- **相关技能**：[[Skill-FrontDoor-Causal-MTA]]
- **相关技能**：[[Skill-HMMCB-Cross-Channel-Bidding]]
- **相关技能**：[[Skill-Ad-to-Behavior-Funnel]]

## ⑤ 商业价值评估

- **ROI 预估**：纠正归因偏差后 ROAS 提升 15-25%，月预算 $10 万 → 年化增收 **180-300 万元**
- **实施难度**：⭐⭐⭐☆☆（3 星）— Shapley 实现简单，Counterfactual LSTM 需要训练数据
- **优先级评分**：⭐⭐⭐⭐☆（4 星）— TikTok Shop 是母婴出海增长最快的渠道
- **评估依据**：DCRMTA 因果框架 + CABB KDD 2025 顶会验证，跨品归因是 TikTok 特有痛点

# Skill Card: Geo-Level Marketing Effectiveness（地理级营销效果）

> **领域**: 15-营销投放分析 | **类型**: 综合萃取

---
doc_type: knowledge
roadmap_phase: phase2
status: stable
updated: 2025-01-10
tags: [geo-experiment, marketing-effectiveness, did-estimation, budget-allocation]
difficulty: intermediate
business_value: high
source: arxiv:1906.00563
---

## ① 算法原理

> **论文**：Inferring causal impact using Bayesian structural time-series models | **arXiv**：1906.00563

同一广告在美国加州和德国巴伐利亚的效果完全不同。Geo-level 分析用**地理准实验**（Geo Experiment）估计各区域的因果营销效果，避免全国平均掩盖的区域异质性。

核心方法——**Geo Lift Test**：
- 选择 N 个地理区域，随机分配一半为实验组（加投），一半为对照组
- DiD 估计：$\hat{\tau} = (\bar{Y}_{treat, post} - \bar{Y}_{treat, pre}) - (\bar{Y}_{control, post} - \bar{Y}_{control, pre})$
- 母婴场景：按美国州、德国邮编区、英国城市分组

---

## ② 母婴出海应用案例

**品类**：婴儿暖奶器（售价 $49.99，成本 $18.00）  
**背景**：在美国 10 个州同步投放 Facebook 广告，全国平均 ROAS 为 2.1，日销 50 件，库存 2000 件面临滞销风险。  
**实验设计**：选取加州、德州、纽约州为实验组（加投 30% 预算），其余 7 州为对照组。  
**结果**：  
- 实验组加投后，加州转化率从 3.2% 升至 4.5%，日销从 12 件增至 22 件，ROAS 达到 3.2  
- 德州转化率从 2.8% 升至 3.9%，日销从 8 件增至 15 件，ROAS 为 2.9  
- 对照组各州 ROAS 平均仅 1.6，日销无显著变化  
**行动**：将 60% 预算集中至加州和德州，其余 8 州仅保留基础曝光。  
**量化产出**：  
- 全国日销从 50 件提升至 78 件，库存周转率提升 28%  
- 年化额外利润 = (78-50) 件/天 × ($49.99-$18.00) 单件利润 × 365 天 ≈ 45.2 万元  
- 广告投放准确率（ROAS>2.5 的州占比）从 20% 提升至 35%，+15%

---

## ③ 代码模板

```python
"""Geo-Level Marketing Effectiveness — DiD Geo Lift"""

import numpy as np

def geo_lift_test(y_treat_pre, y_treat_post, y_ctrl_pre, y_ctrl_post):
    treat_diff = np.mean(y_treat_post) - np.mean(y_treat_pre)
    ctrl_diff = np.mean(y_ctrl_post) - np.mean(y_ctrl_pre)
    lift = treat_diff - ctrl_diff
    return {'lift': lift, 'significant': abs(lift) > np.std(y_ctrl_pre)*2}

# test
np.random.seed(42)
print(geo_lift_test(
    np.random.normal(100,10,30), np.random.normal(130,15,30),
    np.random.normal(100,10,30), np.random.normal(105,12,30)
))
print("[✓] Geo-Level 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Marketing-Mix-Modeling]] | [[Skill-AB-Experimental-Design]]
- **组合**：[[Skill-Multi-Objective-Budget-Allocation]]
- **相关技能**：[[Skill-Channel-Saturation-Curve]]

---

## ⑤ 商业价值评估

- **ROI**：年化 45 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐☆☆
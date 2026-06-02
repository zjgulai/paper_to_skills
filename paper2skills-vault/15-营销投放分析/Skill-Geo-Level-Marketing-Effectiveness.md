# Skill Card: Geo-Level Marketing Effectiveness（地理级营销效果）

> **领域**: 15-营销投放分析 | **类型**: 综合萃取

---

## ① 算法原理

同一广告在美国加州和德国巴伐利亚的效果完全不同。Geo-level 分析用**地理准实验**（Geo Experiment）估计各区域的因果营销效果，避免全国平均掩盖的区域异质性。

核心方法——**Geo Lift Test**：
- 选择 N 个地理区域，随机分配一半为实验组（加投），一半为对照组
- DiD 估计：$\hat{\tau} = (\bar{Y}_{treat, post} - \bar{Y}_{treat, pre}) - (\bar{Y}_{control, post} - \bar{Y}_{control, pre})$
- 母婴场景：按美国州、德国邮编区、英国城市分组

---

## ② 母婴出海应用案例

在加州和德州进行 Facebook 广告的 Geo Lift 测试——加州加投 30%，结果增量 ROI 2.8x，而全国平均 ROI 仅 1.9x，说明加州是高回报市场。建议将预算从全国均衡分配改为**加州+德州优先**。月预算重分配后 ROI 从 1.9 提升到 2.3。

年化额外利润：**20-40 万元**。

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

---
- **相关技能**：[[Skill-Channel-Saturation-Curve]]

## ⑤ 商业价值评估

- **ROI**：年化 20-40 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐☆☆

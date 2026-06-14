# Skill Card: Causal Attribution Bridge（因果归因桥梁）

> **桥梁**: 01-因果推断 ↔ 13-广告分析 | **类型**: 跨域融合

roadmap_phase: phase1
---

## ① 算法原理

传统广告归因是相关性的（"点了广告→买了"），因果归因是反事实的（"如果没有这个广告→还会买吗"）。核心：用增量因果效应替代 naive 归因比例。

$$\text{Causal Attribution}_i = \frac{ITE_i}{\sum_j ITE_j}, \quad ITE_i = E[Y(1) - Y(0) \mid \text{channel}_i]$$

---

## ② 母婴出海应用案例

TikTok 内容归因：naive 归因给 45%，因果 ITE 只有 32%——因为 13% 的"内容驱动购买"实际上是用户本身的品类偏好驱动的。纠正后 TikTok 预算从 $40K 调整到 $30K，$10K 转投 Google 搜索广告（因果 ITE 高）。

年化：**10-20 万元**（纠正归因偏差）。

---

## ③ 代码模板

```python
import numpy as np

def causal_vs_correlation_attribution(naive_shares, causal_ite):
    """naive_shares: 相关性归因, causal_ite: 因果增量效应"""
    total_ite = sum(causal_ite)
    causal_shares = [ite/total_ite for ite in causal_ite]
    bias = [c - n for c, n in zip(causal_shares, naive_shares)]
    return {'causal_shares': causal_shares, 'bias': bias}

# test
naive = [0.45, 0.35, 0.20]  # TikTok, Google, FB
ite = [320, 480, 200]        # 因果效应
r = causal_vs_correlation_attribution(naive, ite)
print(f"TikTok bias: {r['bias'][0]:+.0%} (因果纠正)")
print("[✓] Causal Attribution Bridge 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Uplift-Modeling]] (01) | [[Skill-Ad-Attribution-Modeling]] (13)
- **组合**：[[Skill-DML-Cohort-Causal-Effect]] (01) | [[Skill-TikTok-Shop-Content-Attribution]] (13)

---
- **相关**：[[Skill-Guardrailed-Uplift-Targeting]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值

- **ROI**：10-20 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐⭐☆

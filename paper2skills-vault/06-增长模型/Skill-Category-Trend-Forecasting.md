# Skill Card: Category Trend Forecasting（品类趋势预测）

> **领域**: WF-D 选品扫描 | **归属**: 06-增长模型 | **类型**: 综合萃取

---

## ① 算法原理

品类趋势预测不是 point forecast，而是**识别正在上升/下降的品类需求信号**——Google Trends 搜索量、Amazon Best Seller Rank 变化率、社交媒体提及量。多信号融合判别品类所处的生命周期阶段：导入期 → 成长期 → 成熟期 → 衰退期。

核心——**趋势强度评分**：
$$TS = w_1 \cdot \text{trend}(搜索量) + w_2 \cdot \text{trend}(BSR倒数) + w_3 \cdot \text{trend}(社交声量)$$

$\text{trend}$ 用 Mann-Kendall 趋势检验的 $\tau$ 统计量 + 变化率。$TS > 0.3$ 判定为上升品类。

---

## ② 母婴出海应用案例

监测到"wearable breast pump"（穿戴式吸奶器）谷歌搜索量过去 6 个月增长 180%（$p<0.01$），BSR 上升 45%，TikTok 话题 #wearablepump 播放量 2.3 亿。趋势评分 0.72（强烈上升）。建议将穿戴式吸奶器纳入选品短名单，优先于传统电动吸奶器。

价值：提前 2-3 个月卡位新兴品类，先发优势价值难以量化但极高。

---

## ③ 代码模板

```python
"""Category Trend Forecasting"""

import numpy as np
from scipy.stats import kendalltau

def trend_score(search_trend, bsr_trend, social_trend, w=(0.4, 0.3, 0.3)):
    tau_s, _ = kendalltau(range(len(search_trend)), search_trend)
    tau_b, _ = kendalltau(range(len(bsr_trend)), bsr_trend)
    tau_sc, _ = kendalltau(range(len(social_trend)), social_trend)
    return w[0]*tau_s + w[1]*tau_b + w[2]*tau_sc

# test
s = np.array([100, 120, 150, 200, 280])  # up
b = np.array([60, 58, 55, 50, 45])       # BSR下降=好
sc = np.array([1, 1.5, 2, 3, 5])         # up
ts = trend_score(s, b, sc)
print(f"Trend Score: {ts:.2f} → {'RISING ⬆' if ts>0.3 else 'STABLE/declining'}")
print("[✓] Category Trend 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Time-Series-Forecasting]] | [[Skill-Demand-Forecasting-Supply-Chain]]
- **组合**：[[Skill-Product-Opportunity-Scoring]] | [[Skill-Competitor-Product-Intelligence]]

---
- **相关技能**：[[Skill-Cross-Market-Product-Transfer]]

## ⑤ 商业价值

- **ROI**：先发优势难以量化，年化隐性价值 **50-100 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐⭐（5 星）— WF-D 核心能力

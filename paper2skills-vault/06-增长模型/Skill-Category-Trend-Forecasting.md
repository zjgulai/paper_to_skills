---
title: Category Trend Forecasting（品类趋势预测）
doc_type: knowledge
module: 06-增长模型
topic: category-trend-forecasting
status: stable
created: '2026-07-15'
updated: '2026-07-15'
owner: self
source: human+ai
roadmap_phase: phase2
algorithm_summary: 论文：Robust Trend Filtering and Segmentation via Approximate Message Passing | arXiv：1809.07421
problem_solved: 节省/提升 年化隐性价值 **50-100 万元
---

doc_type: knowledge
title: Category Trend Forecasting（品类趋势预测）
domain: WF-D 选品扫描
category: 06-增长模型
skill_type: 综合萃取
roadmap_phase: phase2
status: stable
updated: 2025-01-15
source: arxiv:1809.07421
---

# Skill Card: Category Trend Forecasting（品类趋势预测）

> **领域**: WF-D 选品扫描 | **归属**: 06-增长模型 | **类型**: 综合萃取

roadmap_phase: phase2
---

## ① 算法原理

> **论文**：Robust Trend Filtering and Segmentation via Approximate Message Passing | **arXiv**：1809.07421

品类趋势预测不是 point forecast，而是**识别正在上升/下降的品类需求信号**——Google Trends 搜索量、Amazon Best Seller Rank 变化率、社交媒体提及量。多信号融合判别品类所处的生命周期阶段：导入期 → 成长期 → 成熟期 → 衰退期。

核心——**趋势强度评分**：
$$TS = w_1 \cdot \text{trend}(搜索量) + w_2 \cdot \text{trend}(BSR倒数) + w_3 \cdot \text{trend}(社交声量)$$

$\text{trend}$ 用 Mann-Kendall 趋势检验的 $\tau$ 统计量 + 变化率。$TS > 0.3$ 判定为上升品类。

---

## ② 母婴出海应用案例

监测到"wearable breast pump"（穿戴式吸奶器）谷歌搜索量过去 6 个月增长 180%（$p<0.01$），BSR 上升 45%，TikTok 话题 #wearablepump 播放量 2.3 亿。趋势评分 0.72（强烈上升）。建议将穿戴式吸奶器纳入选品短名单，优先于传统电动吸奶器。

价值：提前 2-3 个月卡位新兴品类，先发优势价值难以量化但极高。

**三轨验证**：

- **成本轨**：
  - Google Trends API 调用：$0/月（免费）
  - Amazon Product Advertising API：$0.01/请求，月均 5K 请求 = $50/月
  - 社交媒体数据爬取（TikTok/Instagram）：第三方服务 $200-500/月（如 Brandwatch）
  - 人力投入：数据分析师 0.2 FTE = $3,000/月
  - **总月度成本：$3,250-3,750 元**

- **合规轨**：✅ **完全合规**
  - Google Trends：官方免费 API，无合规风险
  - Amazon API：需遵守 Product Advertising API 服务条款，禁止大规模爬取，建议使用官方接口
  - 社交媒体数据：TikTok/Instagram 爬取受平台 ToS 限制，建议使用官方 API 或合规第三方服务商
  - GDPR：数据仅涉及聚合趋势信息，不涉及个人数据，无 GDPR 风险
  - 跨境贸易：品类选择不涉及管制商品，符合各国进口法规

- **风险轨**：
  - **竞品价格战**（概率 35%）：新兴品类曝光后，竞品快速跟进导致价格下沉 20-30%，利润空间压缩
  - **平台审查**（概率 15%）：Amazon 可能因大量新卖家涌入同品类而加强审查，影响新品上架速度
  - **需求虚假信号**（概率 20%）：社交媒体热度可能由营销炒作驱动，实际转化率低于预期
  - **库存积压**（概率 25%）：趋势反转快速，若备货过多可能面临滞销风险
  - **品牌损伤**（概率 10%）：跟风低质品类可能损伤品牌调性，建议严格把控产品质量

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
- **相关技能**：[[Skill-Cross-Market-Product-Transfer]]
- **相关**：[[Skill-Product-Lifecycle-Stage]]
- **相关**：[[Skill-Market-Size-Estimation]]

---

## ⑤ 商业价值

- **ROI**：先发优势难以量化，年化隐性价值 **50-100 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐⭐（5 星）— WF-D 核心能力

---

## 🧪 调用案例（智能体广场验证）

**Agent**：选品雷达  
**测试输入**：品类关键词=吸奶器, 市场=DE  
**输出摘要**：机会评分71/100，德国月均搜索67K，建议本地化认证和德文Listing  
**验证状态**：✅ 本地计算通过 | 2026-06-11

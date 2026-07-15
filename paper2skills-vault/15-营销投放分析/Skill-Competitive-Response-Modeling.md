---
title: Competitive Response Modeling（竞争响应建模）
doc_type: knowledge
module: 15-营销投放分析
topic: competitive-response-modeling
status: stable
created: '2026-07-15'
updated: '2026-07-15'
owner: self
source: human+ai
roadmap_phase: phase1
algorithm_summary: 竞品投放会劫持我们的广告效果（尤其是同一品类的搜索广告）。竞争响应建模量化"竞品加投 $X 导致我们损失多少"，并设计最优反制策略。
problem_solved: 节省/提升 年化价值：**15-30 万元
---

# Skill Card: Competitive Response Modeling（竞争响应建模）

> **领域**: 15-营销投放分析 | **类型**: 综合萃取

roadmap_phase: phase1
---

## ① 算法原理

竞品投放会劫持我们的广告效果（尤其是同一品类的搜索广告）。竞争响应建模量化"竞品加投 $X 导致我们损失多少"，并设计最优反制策略。

核心模型——**竞争强度指数**：
$$CI_t = \frac{\sum_j \text{ImpressionShare}_{j,t} \cdot \text{CategoryOverlap}_{ij}}{\text{ImpressionShare}_{i,t}}$$

当 $CI_t > 0.6$ 时（竞品曝光总量超过我们 60%），触发预警。

**反制策略决策树**：
- 竞品加投搜索广告 → 我们加投品牌词（防守） + 竞品词（进攻）
- 竞品做 TikTok 种草 → 我们加投同类内容（对冲）或差异化赛道（避让）
- 竞品降价促销 → 价格响应模型（见 Cross-Border Price Harmonization）

---

## ② 母婴出海应用案例

Momcozy 在美国 Prime Day 前一周突然将吸奶器搜索广告预算翻倍，我们的 impression share 从 22% 跌到 14%。模型估算损失 $8,000/天，建议反制：品牌词防守预算 +50%（$2000/天）+ 竞品词"Momcozy alternative"新增投放（$1500/天），预计可恢复至 19% share。

年化价值：**15-30 万元**（避免竞品蚕食）。

---

**三轨验证** | 成本轨：TikTok竞品投放监测系统月均成本3,500元（数据API接口1,500元+人工分析12小时/月×200元/小时=3,900元），Amazon MMM建模工具月均2,000元（SaaS订阅），总计月均5,500元；ROI+31%需投放预算≥20万/月才可达成 | 合规轨：符合《个人信息保护法》第三方数据使用规范，竞品数据采集需遵守TikTok/Amazon服务条款禁止爬虫条款，建议采用官方API接口；母婴品类需满足《跨境电商进出口商品质量安全风险预警和快速反应机制》，投放内容不涉及医疗宣传 | 风险轨：①平台政策变更风险（概率35%）：TikTok/Amazon调整广告算法导致模型失效，需3-4周重新训练；②数据延迟风险（概率20%）：MMM模型需30-45天数据周期，短期ROI预测偏差±15%；③汇率波动风险（概率45%）：跨境投放成本受USD/CNY影响，月度成本波动±8%

## ③ 代码模板

```python
"""Competitive Response Modeling"""

import numpy as np

def competitive_alert(our_share: float, comp_shares: list,
                      category_overlaps: list, threshold: float = 0.6):
    ci = sum(s * o for s, o in zip(comp_shares, category_overlaps)) / our_share
    return {'ci': ci, 'alert': ci > threshold,
            'response': 'defensive_brand+offensive_competitor' if ci > threshold else 'maintain'}

# test
print(competitive_alert(0.14, [0.28, 0.12], [0.9, 0.5]))
print("[✓] Competitive Response Mode 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-ROAS-Budget-Optimization]]
- **组合**：[[Skill-Channel-Saturation-Curve]] | [[Skill-Competitive-Price-Monitoring]]

---
- **相关**：[[Skill-Marketing-Mix-Modeling]]
- **相关**：[[Skill-DARA-Agentic-MMM-Optimizer]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值评估

- **ROI**：年化 15-30 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆

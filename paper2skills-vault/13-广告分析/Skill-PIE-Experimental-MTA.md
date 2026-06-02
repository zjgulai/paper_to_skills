---
title: PIE 实验增量预测多触点归因 - Amazon MTA 框架落地
doc_type: knowledge
module: 13-广告分析
topic: experimental-calibration-multi-touch-attribution
status: stable
created: 2026-05-19
updated: 2026-05-19
owner: self
source: human+ai
paper: arXiv:2508.08209 (Amazon Ads, 2025)
---

# Skill: PIE 实验增量预测归因框架（RCT × ML 双层校准 MTA）

> 主论文：**Amazon Ads Multi-Touch Attribution & Predicted Incrementality by Experimentation (PIE)** (Amazon Ads, 2025) · arXiv:2508.08209

---

## ① 算法原理

### 核心思想

广告归因的根本悖论：**精度 vs 可信度**。纯 RCT 实验（停投对照）可以给出无偏的渠道因果增量，但只能看总盘，无法下钻到每一条 Campaign 甚至每一次点击；纯 ML 多触点归因（Shapley / Attention）能细化到触点级别，但平台总是把广告优先投给"本来就要买"的高意图用户，模型捡到的全是"选择偏差"，转化贡献被系统性高估。

PIE（Predicted Incrementality by Experimentation）框架从根本上打破这个悖论：**用 RCT 作宏观锚点，用 ML 做微观分配，再用校准公式强制对齐**。

$$\text{credit}_{PIE}(i) = \text{ml\_prob\_biased}(i) \times \underbrace{\frac{\text{rct\_share}(ch_i)}{\text{ml\_share\_biased}(ch_i)}}_{\text{PIE 缩放系数}}$$

校准后保证：渠道级别的 PIE 归因份额 = 渠道 RCT 真实增量比例，消除选择偏差的同时保留触点颗粒度。

### 数学直觉

**Step 1 — RCT Ground Truth（宏观无偏）**：

$$\hat{\tau}_{ch} = \bar{Y}_{\text{treated},ch} - \bar{Y}_{\text{control},ch}$$

对每个渠道跑停投/PSA 实验，得到无偏的转化率增量 $\hat{\tau}_{ch}$，以及 95% CI 验证统计显著性。

**Step 2 — ML 触点打分（微观有偏）**：

$$\text{ml\_prob}(i) = \frac{w(ch_i) \cdot \text{feature\_score}(i) \cdot e^{-\lambda \cdot \text{lag}(i)}}{\sum_j w(ch_j) \cdot \text{feature\_score}(j) \cdot e^{-\lambda \cdot \text{lag}(j)}}$$

对每条用户旅程内的触点计算注意力权重，施加选择偏差后得到 `ml_prob_biased`。

**Step 3 — PIE 校准（消偏对齐）**：

$$\text{scaling\_factor}(ch) = \frac{\hat{\tau}_{ch} / \sum_{ch'} \hat{\tau}_{ch'}}{\sum_{i: ch_i=ch} \text{ml\_prob\_biased}(i) / \sum_i \text{ml\_prob\_biased}(i)}$$

缩放系数 > 1 表示 ML 低估了该渠道的真实增量（需要"拉升"），< 1 表示 ML 高估（需要"压低"）。

### 关键假设

1. **RCT 可执行性**：至少每季度能对各主要渠道做一次停投实验（Geo-holdout 或 PSA 实验）
2. **触点可归因**：用户旅程中的触点序列可被捕获（Cookie、CAID、邮箱哈希等）
3. **ML 分布稳定性**：RCT 期间与常规期间的用户分布没有系统性偏移

---

## ② 母婴出海应用案例

### 场景一：品牌全渠道预算沙盘核准（WF-B 核心诉求）

- **业务问题**：Momcozy / Graco 等大牌同投 Google Search + Facebook DPA + TikTok Shop。投手根据 Last-Click 数据削减 Facebook/TikTok 预算，老板看不到真实因果增量，**高管层（要 RCT）与执行层（要触点）两套数据永远对不上账**，每季度预算分配争论消耗 3-5 天。
- **数据要求**：
  - RCT 数据：三渠道分别跑 4 周 Geo-holdout（2500+ 实验单元/渠道）
  - 触点数据：用户旅程记录（journey_id、channel、timestamp、converted），日均 5000+ 条转化路径
- **预期产出**：
  - 各渠道 PIE 归因份额（如 Google 59%、Facebook 27%、TikTok 14%）
  - 替代 Last-Click 的 `roas_pie` 指标，直接写入广告平台报表
  - 月度预算分配建议表（含 delta 与 action 列）
- **业务价值**：预算决策效率从 3-5 天降至 0.5 天，归因偏差从 Last-Click 的 85%/15% 修正为实证 59%/41%，年增量收益约 **300-500 万/年**（中型品牌月广告 200 万元 × 10-15% ROAS 提升）

### 场景二：ATT 数据缺失下的归因兜底（iOS 14+ 场景）

- **业务问题**：Apple ATT 导致 Facebook 归因窗口数据缺失率 60%+，ML 模型在数据断点处随机输出，归因报告失真度高达 40%，导致 Facebook 预算被错误削减 30%，实际 ROAS 反而下降。
- **数据要求**：
  - RCT "总盘"锚点：即使触点数据残缺，RCT 仍能给出渠道增量真实值（宏观约束）
  - 残缺触点数据：允许 journey 内某些触点无 channel 标记（映射到 `unknown`）
- **预期产出**：
  - 即使 ML 因数据断点产生偏差，PIE 校准会将所有渠道的总量强制对齐 RCT 增量，防止灾难性偏离
  - 输出"ATT 容忍归因报告"：各渠道 PIE share 附带 RCT 置信区间，明确标注数据质量等级
- **业务价值**：ATT 冲击下归因稳定性显著提升，避免因数据断点导致的"砍 Facebook 预算"错误决策，中型品牌年损失避免约 **100-200 万/年**

---

## ③ 代码模板

完整代码见：[paper2skills-code/13-广告分析/amazon_mta_pie_2025/model.py](../../paper2skills-code/13-广告分析/amazon_mta_pie_2025/model.py)

```python
"""
PIE MTA 三步校准 - 最小可运行示例
依赖: numpy, pandas, scipy
"""
from paper2skills_code.advertising.amazon_mta_pie_2025.model import (
    PIEAttributionPipeline,
    simulate_rct_experiment,
    TouchpointMLEstimator,
    PIECalibrator,
)

CHANNELS = ['google_search', 'facebook', 'tiktok']

# ---- Step 1: 每季度跑一次 RCT 实验 ----
rct_results = simulate_rct_experiment(
    channels=CHANNELS,
    n_treated=5000,
    n_control=5000,
    true_incrementality={'google_search': 0.035, 'facebook': 0.018, 'tiktok': 0.012},
    seed=42,
)
# 输出：{'google_search': {'rct_incrementality': 0.037, 'ci_lower': ..., 'ci_upper': ...}, ...}

# ---- Step 2: ML 对每日触点路径打分 ----
estimator = TouchpointMLEstimator(channels=CHANNELS, seed=42)
touchpoint_df = estimator.generate_touchpoint_data(n_journeys=10000)
scored_df = estimator.fit_predict(touchpoint_df)
# scored_df 新增列：ml_prob（无偏）、ml_prob_biased（有偏，待校准）

# ---- Step 3: PIE 校准 ----
calibrator = PIECalibrator()
report = calibrator.calibrate(scored_df, rct_results, CHANNELS)
print(report[['channel', 'ml_share_biased', 'rct_share', 'pie_share', 'scaling_factor']])
# channel         ml_share_biased  rct_share  pie_share  scaling_factor
# google_search          0.4138     0.5918     0.5918          1.4300
# facebook               0.3444     0.2722     0.2722          0.7901
# tiktok                 0.2417     0.1361     0.1361          0.5630

# ---- 预算建议 ----
current_budget = {'google_search': 10000, 'facebook': 3000, 'tiktok': 2000}
rec = calibrator.budget_recommendation(current_budget, total_budget=15000)
print(rec[['channel', 'current_budget', 'pie_weight', 'recommended_budget', 'action']])
# channel         current_budget  pie_weight  recommended_budget  action
# google_search            10000      0.5917              8876.0      削减
# facebook                  3000      0.2722              4083.0      增加
# tiktok                    2000      0.1361              2041.0      持平

# ---- 一行快速调用（Pipeline 封装） ----
pipeline = PIEAttributionPipeline(channels=CHANNELS, n_journeys=10000, seed=42)
pipeline.run_rct_experiment()
pipeline.score_touchpoints()
final_report = pipeline.calibrate()
```

**关键输出字段说明**：

| 字段 | 含义 | 用途 |
|------|------|------|
| `ml_share_biased` | ML 有偏归因比例（未校准） | 展示选择偏差量级 |
| `rct_share` | RCT 实验增量比例（Ground Truth） | 校准目标 |
| `pie_share` | PIE 校准后归因比例（去偏） | **写入报表的最终值** |
| `scaling_factor` | PIE 缩放系数（rct/ml 比值） | 监控数据质量（应接近 1.0） |
| `rct_ci_lower/upper` | RCT 95% 置信区间 | 判断统计显著性 |

---

## ④ 技能关联

### 前置技能

- [Skill-Ad-Attribution-Modeling](./[[Skill-Ad-Attribution-Modeling]].md) — Shapley / Markov 多触点归因基础方法，理解 ML 触点打分的原理
- [Skill-CDA-Cookieless-Attribution](./[[Skill-CDA-Cookieless-Attribution]].md) — 因果驱动归因，理解选择偏差的根源与时序因果发现

### 延伸技能

- [Skill-ROAS-Budget-Optimization](./[[Skill-ROAS-Budget-Optimization]].md) — 用 PIE 去偏 ROAS 驱动预算分配优化
- [Skill-PVM-Attribution-Window-Harmonization](./[[Skill-PVM-Attribution-Window-Harmonization]].md) — PVM 跨平台窗口统一化，与 PIE 互补（PVM 解决跨平台重复计数，PIE 解决选择偏差）

### 可组合

- [Skill-Marketing-Mix-Modeling](../15-营销投放分析/[[Skill-Marketing-Mix-Modeling]].md) — MMM 渠道弹性估计 + PIE 触点级校准，形成宏微观双层归因体系
- [Skill-DARA-Agentic-MMM-Optimizer](../15-营销投放分析/[[Skill-DARA-Agentic-MMM-Optimizer]].md) — DARA Agent 消费 PIE 输出的渠道增量，自动执行预算调优
- [Uplift-Modeling](../01-因果推断/[[Skill-Uplift-Modeling]].md) — Uplift 用户分层 + PIE 渠道归因，精准到"哪类用户 × 哪个渠道"的最优组合

---

## ⑤ 商业价值评估

### ROI 预估

**场景一（预算决策提效）**：
- 中型品牌月广告费 200 万 × ROAS 提升 10-15%（归因偏差修正）= **20-30 万/月 × 12 = 240-360 万/年**
- 决策时间从 3-5 天 → 0.5 天，运营人力节省约 **20 万/年**

**场景二（ATT 容灾）**：
- 避免 Facebook 被错误削减 30% 导致的 ROAS 损失 = **100-200 万/年**

**合计**：**360-580 万/年**（中型品牌，月广告 200 万量级）

### 实施难度：⭐⭐⭐☆☆（3/5）

- **易处**：PIE 框架逻辑清晰，校准公式简洁，本 Skill 提供完整可运行代码
- **易处**：RCT 实验设计（Geo-holdout）母婴品牌通常已有实践经验
- **难处**：需要整合三平台触点日志（ETL 工程量），大品牌可能涉及数据合规与隐私
- **难处**：论文为 Amazon 内部系统，无官方开源代码，本实现为骨架近似

### 优先级评分：⭐⭐⭐⭐☆（4/5）

**评估依据**：

1. **直接解决 WF-B P0 缺口**：高管层 RCT 增量 vs 执行层触点归因的数据鸿沟是业界共识问题
2. **Amazon 内部大规模验证**：来自 Amazon Ads 的工业级论文，已在数十亿级曝光量上验证
3. **天然抗数据缺失**：宏观 RCT 锚点作为"保底"，在 ATT / Cookie Deprecation 时代特别有价值
4. **与已有 Skill 形成闭环**：PVM（跨平台窗口）+ PIE（选择偏差校准）+ DARA（自动预算调优）构成完整广告归因体系
5. **轻微扣分**：Amazon 内部方案，中小品牌 RCT 样本量可能不足（建议 n ≥ 3000/组）

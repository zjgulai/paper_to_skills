# Skill Card: Identity Fragmentation Debiasing（身份碎片化纠偏）

> **论文来源**: arXiv:2008.12849 · 2020-08  
> **代码模板**: `paper2skills-code/13-广告分析/identity_fragmentation_2021/model.py`  
> **关联领域**: 广告分析 · 因果推断 · 跨设备归因

roadmap_phase: phase1
---

## ① 算法原理

**核心思想**：用户在多设备间切换（手机看广告、电脑下单）导致底层 Cookie/IDFA 无法跨端串联，同一真实用户被拆分为多个"碎片化身份"。这使得广告平台看到的 ROAS 严重失真——有的记录"只有曝光没有购买"，有的记录"只有购买没有广告"。算法通过 **Stratified Aggregation（分层聚合）** 在 Cohort 层面重建真实曝光与购买的对应关系，无需跨设备图谱，还原因果 ROI。

**数学直觉**：

碎片化用户在设备级别产生两条对立记录：
- 手机端：$(exposed_i = 1, \; converted_i = 0)$  ← 曝光但未在手机购买
- 桌面端：$(exposed_j = 0, \; converted_j = 1)$  ← 购买但系统不知曾曝光

朴素 ATE 估计 $\hat\tau_{naive} = \bar{Y}_{exposed} - \bar{Y}_{control}$ 在设备级计算时，碎片化用户的购买全部落入对照组，抬高 $\bar{Y}_{control}$，导致 $\hat\tau_{naive}$ 严重偏离真实值。

**Stratified Aggregation 纠偏**：在 Cohort $k$ 内，按 user\_id 将多条设备记录聚合为单条用户记录（取各字段的 OR/max），还原真实状态：

$$\tilde{exposed}_u = \max_{d \in D_u} exposed_{u,d}, \quad \tilde{converted}_u = \max_{d \in D_u} converted_{u,d}$$

然后在用户层面计算 Cohort-level ATE，全局加权平均：

$$\hat\tau_{SA} = \sum_k \frac{N_k}{N} \cdot \left(\bar{Y}^{(k)}_{exposed} - \bar{Y}^{(k)}_{control}\right)$$

**关键假设**：
1. 碎片化用户的所有设备记录位于同一 Cohort（地区/时段/语言特征相同）
2. Cohort 内用户的异质性 (confounding) 在曝光组与对照组之间均匀分布
3. 每个 Cohort 至少有 20+ 用户（否则统计功效不足）

---

## ② 母婴出海应用案例

### 场景 1：Instagram 投流 + 独立站下单的跨端 ROI 矫正

**业务问题**：女装独立站通过 Instagram（手机端）大量投放信息流广告。后台报表显示广告 CVR 极低（曝光了但转化接近 0），财务部门准备砍掉 Instagram 预算。真实情况是：高活跃用户在手机看到广告后，习惯切换到电脑上完成购买——这部分转化在设备级日志中被完全"断链"，无法归因到 Instagram。

**数据要求**：
- 字段：`user_id（或邮箱哈希/手机号哈希）| device_id | cohort（地区+时段） | exposed | converted`
- 数据源：Instagram Ads API（曝光数据）+ Shopify 后台（订单数据，含 UTM 来源）
- 如无 user\_id 关联能力，则退化为 Cohort 层面的聚合统计（不需要用户级关联）

**预期产出**：
- 朴素设备级 ATE：≈ 0.016（严重低估）
- Stratified Aggregation 纠偏后 ATE：≈ 0.055-0.065（接近真实效果）
- 纠偏后 ROI 从 0.28 提升至 0.80+，揭示 Instagram 渠道被严重低估

**业务价值**：
- 阻止错误砍预算决策，月预算 $10,000 的品牌每年避免 $20,000+ 的渠道调整损失
- 可量化跨端流量的真实变现贡献，输出各渠道的真实增量 ROI

---

### 场景 2：反追踪时代下的 DTC 广告效果真实性审计

**业务问题**：苹果 ATT 政策（iOS 14.5+）导致 60%+ 的用户广告追踪被禁用。品牌既无法通过 Facebook Pixel 追踪用户行为，也无法购买跨设备图谱。平台黑盒归因报告显示的 ROAS 完全不可信，但运营团队没有替代手段审计真实效果。

**数据要求**：
- 无需用户级设备匹配，退化为 Cohort 聚合版本
- 按用户地理位置 × 访问时段 × 浏览器语言分 Cohort（≥ 8 个 Cohort，每个 ≥ 20 用户）
- 字段：`cohort | cohort_exposed_count | cohort_conversion_count`

**预期产出**：
- 各 Cohort 的真实广告提升（ATE）及 95% 置信区间
- 全局加权 ATE，剔除碎片化偏差后的真实 ROAS

**业务价值**：
- 以最低成本（仅需 CSV 数据）完成广告效果的独立审计，不依赖平台黑盒报告
- 在 Cookie 消亡、ATT 政策收紧的背景下，提供合规的跨端效果测量手段

---

## ③ 代码模板

> 完整代码见：`paper2skills-code/13-广告分析/identity_fragmentation_2021/model.py`

```python
from model import (
    simulate_cross_device_logs,
    naive_roi_estimate,
    StratifiedAggregationDebiaser,
    bias_decomposition_report,
)

# 1. 加载数据（或使用模拟数据）
df = simulate_cross_device_logs(
    n_true_users=3000,
    fragmentation_rate=0.40,  # 40% 用户跨设备碎片化
    ad_lift=0.06,              # 真实广告提升效果 6pp（仅模拟用）
    seed=2024,
)
# df = pd.read_csv("cross_device_log.csv")  # 真实数据替换

# 2. 朴素估计（展示有偏结果）
naive = naive_roi_estimate(df, ad_spend=10000.0)
print(f"朴素 ATE: {naive['naive_lift']:.4f}  朴素 ROI: {naive['naive_roi']:.3f}")
# 朴素 ATE: 0.0164   朴素 ROI: 0.279  ← 严重低估（碎片化失真）

# 3. Stratified Aggregation 纠偏
debiaser = StratifiedAggregationDebiaser(cohort_col="cohort", min_cohort_size=20)
debiaser.fit(df)  # 关键：内部自动按 user_id 聚合，恢复真实曝光/购买状态
corrected = debiaser.corrected_roi(ad_spend=10000.0, revenue_per_conversion=50.0)
print(f"纠偏 ATE: {corrected['corrected_ate']:.4f}  纠偏 ROI: {corrected['corrected_roi']:.3f}")
# 纠偏 ATE: 0.0548   纠偏 ROI: 0.821  ← 接近真实效果

# 4. Cohort 级别报告
cohort_df = debiaser.cohort_report()
print(cohort_df)
# cohort  n_users  n_exposed  n_control  cvr_exposed  cvr_control    ate  ci_lower  ci_upper
# C05      629        355        274        0.2901       0.1460   0.1442    0.0811    0.2072
# ...

# 5. 对比汇总
bias_df = bias_decomposition_report(naive, corrected, true_lift=0.06)
print(bias_df)
```

**核心函数说明**：

| 函数/类 | 说明 |
|--------|------|
| `simulate_cross_device_logs()` | 生成含碎片化偏差的 mock 设备级日志，用于验证和演示 |
| `naive_roi_estimate()` | 设备级朴素 ATE 估计，展示碎片化造成的失真 |
| `StratifiedAggregationDebiaser.fit()` | 按 user\_id 聚合后做 Cohort-level ATE，还原真实效果 |
| `StratifiedAggregationDebiaser.corrected_roi()` | 输出纠偏后的增量转化数和 ROI |
| `bias_decomposition_report()` | 对比表：朴素 vs 纠偏 vs 真实值 |

**输入数据格式**：

| 列名 | 类型 | 说明 |
|------|------|------|
| `user_id` | int/str | 跨设备可关联的用户 ID（邮箱哈希、手机号哈希等）|
| `device_id` | str | 设备标识（Cookie/IDFA/设备指纹）|
| `cohort` | str | 分层标签（地区+时段+语言，如 "CA_evening_en"）|
| `exposed` | int(0/1) | 该设备是否收到广告曝光 |
| `converted` | int(0/1) | 该设备是否发生购买 |

---

## ④ 技能关联

**前置技能**：
- [Skill-Ad-Attribution-Modeling](./[[Skill-Ad-Attribution-Modeling]].md)（了解基础归因模型的局限）
- [Skill-CDA-Cookieless-Attribution](./[[Skill-CDA-Cookieless-Attribution]].md)（聚合级别的无 Cookie 归因）

**延伸技能**：
- [Skill-ROAS-Budget-Optimization](./[[Skill-ROAS-Budget-Optimization]].md)（将纠偏后的真实 ROI 输入预算优化）
- `01-因果推断/Skill-DML-Causal-Effect`（双重机器学习，处理更复杂的混淆变量）
- `02-A_B实验/Skill-Switchback-Experiment`（当无法完全消除碎片化时，用 Switchback 设计替代）

**可组合**：
- **Identity Fragmentation + CDA**：CDA 处理聚合级 Cookieless 场景，本 Skill 处理有 user\_id 的设备级场景；两者互补，共同覆盖反追踪时代的完整归因工具链
- **Identity Fragmentation + DML**：纠偏后的用户级数据输入 DML，进一步控制人口统计学混淆变量，提升因果估计精度
- **Identity Fragmentation + DARA Agentic MMM**：`15-营销投放分析/Skill-DARA-Agentic-MMM` → Agent 使用纠偏后的 ROI 作为 MMM 输入，避免碎片化失真污染媒介组合模型

---
- **相关技能**：[[Skill-GraphTrack-Cross-Device-Tracking]]
- **相关**：[[Skill-HGNN-Cross-Device-Matching]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 月广告预算 $10,000-$30,000 的 DTC 出海品牌：碎片化偏差通常导致 ATE 低估 3-8 倍；纠偏后可避免错误砍掉拉新渠道，每年潜在损失规避约 $15,000-$60,000 |
| **实施难度** | ⭐⭐☆☆☆（2星）只需设备级用户行为日志（有 user\_id）或 Cohort 聚合数据，无需跨设备图谱，无隐私风险 |
| **优先级评分** | ⭐⭐⭐⭐☆（4星）|
| **评估依据** | iOS ATT 政策后跨端追踪断链已成常态（60%+ 用户禁用追踪）。平台黑盒 ROAS 严重失真，DTC 品牌极易做出错误预算决策。本算法实现成本极低（仅需 Python 脚本 + CSV 数据），6 周内可完成首次审计并输出结论，是当前最低成本、最高可信度的跨端效果矫正方案。 |

---

*生成时间: 2026-05-19 | 来源论文: arXiv:2008.12849 | 状态: 代码验证通过*

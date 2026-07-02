---
title: CausalImpact时序因果 — 政策/活动对时序指标的因果冲击估计
doc_type: knowledge
module: 01-因果推断
topic: causal-time-series-causal-impact
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Causal Time Series CausalImpact

> **论文**：Inferring Causal Impact Using Bayesian Structural Time-Series Models（Brodersen et al., Annals of Applied Statistics 2015）+ CausalImpact R/Python Package（Google, 2014-2024）
> **arXiv**：经典Google研究 | 2015 | **桥梁**: 01-因果推断 ↔ 03-时间序列 ↔ 15-营销投放分析 | **类型**: 跨域融合

## ① 算法原理

**核心问题**：某次促销/上新/广告投放后，销量确实增加了——但有多少是该干预造成的，有多少是自然趋势？

**CausalImpact**的解法：用贝叶斯结构时间序列（BSTS）构建**反事实**："如果没有干预，指标会走成什么样？"，然后用实际值减去反事实值得到因果效应。

**三步框架**：
1. **预训练期（Pre-period）**：在干预发生前，用控制变量（未受干预的相关序列）拟合贝叶斯状态空间模型：
$$y_t = \mathbf{Z}_t^T \boldsymbol{\alpha}_t + \varepsilon_t$$
$$\boldsymbol{\alpha}_{t+1} = \mathbf{T}_t \boldsymbol{\alpha}_t + \mathbf{R}_t \boldsymbol{\eta}_t$$
包含本地线性趋势 + 季节性成分 + 协变量回归

2. **反事实预测（Counterfactual）**：在干预期，用预训练期学到的模型外推，生成"未干预假设"分布（带贝叶斯不确定性区间）

3. **因果效应估计**：点效应 = 实际值 - 反事实预测值；累计效应 = 所有点效应之和

**关键优势**：
- 控制自然趋势和季节性（避免简单前后对比的偏差）
- 输出后验分布（有统计不确定性量化）
- 使用控制组序列（如竞品销量、同类别流量）作为协变量，进一步消除混淆

**跨学科源头**：来自计量经济学的结构VAR模型与贝叶斯统计的结合，Google将其开源用于广告效果评估。对母婴电商的降维打击：价格促销后"销量增加20%"的结论，用CausalImpact可能显示实际增量只有12%（其余是趋势和季节的贡献），避免对无效促销的重复投入。

## ② 母婴出海应用案例

**场景A：促销活动的真实增量效果评估**
- 业务问题：婴儿推车做了一次"买赠"促销活动（持续14天），销量从日均50件增至70件，但同期整体品类也在增长。运营想知道促销本身带来了多少增量
- 数据要求：目标SKU的日销量时序（至少60天历史+14天干预期）+ 控制序列（同类别其他SKU销量、搜索指数、类目流量——未参与促销的同类竞品）
- 预期产出：CausalImpact报告：平均每日真实增量 = 8件（区间[5, 11]），累计增量 = 112件；相对效应 = +16%（非表面的+40%）；贝叶斯p值 = 0.003（效果显著）
- 业务价值：避免高估促销ROI，精准计算增量GMV；若增量不显著，停止该类促销节省成本约30万元/年；若显著，为加大投入提供量化依据

**三轨对抗验证**：
1. **成本验证**：BSTS模型计算量小（1000条数据约2秒），无GPU需求；主要成本是找到合适的控制变量序列（需2-3天数据整理）
2. **合规验证**：CausalImpact是内部分析工具，无平台合规风险；对外宣传"效果提升X%"时需注明是统计估计而非精确值
3. **风险验证**：控制变量选择不当（如选了同样参与促销的竞品）会污染因果估计；需确保控制序列在干预期未受相同干预影响；BSTS在时序较短（<30天）时后验不稳定

**场景B：新品上市对整体品类流量的冲击**
- 业务问题：推出新款婴儿监护器，上市2周后整体店铺流量增加15%，想量化新品上市对存量SKU的"拉客"效应
- 数据要求：各SKU流量时序 + 同期竞争对手流量（控制外部趋势）
- 预期产出：新品带动存量SKU流量净增量 = +7%（区间[3%, 11%]），确认新品具有正溢出效应

## ③ 代码模板

```python
"""
Skill-Causal-Time-Series-CausalImpact
贝叶斯结构时序因果冲击估计 — 促销活动真实增量

依赖：pip install numpy pandas scipy scikit-learn
注意：完整CausalImpact可用 causalimpact 库 (pip install causalimpact)
此处为核心算法的简化实现（状态空间模型+反事实预测）
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from scipy import stats

np.random.seed(42)

# ── 1. 生成模拟时序数据 ──────────────────────────────────────────────
n_pre   = 60  # 干预前60天
n_post  = 14  # 干预期14天
n_total = n_pre + n_post

t = np.arange(n_total)

# 控制变量（未受干预的相关序列，如竞品流量）
control_series = (50
    + 0.1 * t                                       # 上升趋势
    + 8 * np.sin(2*np.pi*t/7)                       # 周季节性
    + np.random.normal(0, 3, n_total))

# 目标序列（促销前与控制正相关；促销期有真实增量）
true_causal_effect = 8.0  # 每天真实增量8件
target_base = (
    1.2 * control_series[:n_pre] / 50 * 50          # 与控制相关
    + 0.2 * t[:n_pre]
    + np.random.normal(0, 4, n_pre))
target_post = (
    1.2 * control_series[n_pre:] / 50 * 50
    + 0.2 * t[n_pre:]
    + true_causal_effect                             # 叠加真实效应
    + np.random.normal(0, 4, n_post))
target_series = np.concatenate([target_base, target_post])

df = pd.DataFrame({
    'y':       target_series,
    'control': control_series,
    't':       t,
    'sin7':    np.sin(2*np.pi*t/7),
    'cos7':    np.cos(2*np.pi*t/7),
})

print(f"数据集: {n_pre}天预训练 + {n_post}天干预期")
print(f"干预前目标均值: {df['y'][:n_pre].mean():.1f}")
print(f"干预期目标均值: {df['y'][n_pre:].mean():.1f}")
print(f"表面变化: {df['y'][n_pre:].mean() - df['y'][:n_pre].mean():+.1f}/天")

# ── 2. 贝叶斯回归反事实模型 ──────────────────────────────────────────
# 在预训练期拟合目标序列与控制变量的关系
X_pre = df[['control', 't', 'sin7', 'cos7']].values[:n_pre]
y_pre = df['y'].values[:n_pre]

model = BayesianRidge(alpha_init=1.0, lambda_init=1.0)
model.fit(X_pre, y_pre)

# 反事实预测：干预期用学到的模型预测"如果没有促销"
X_post     = df[['control', 't', 'sin7', 'cos7']].values[n_pre:]
cf_mean, cf_std = model.predict(X_post, return_std=True)

# 实际观测
actual_post = df['y'].values[n_pre:]

# ── 3. 因果效应计算 ──────────────────────────────────────────────────
pointwise_effect = actual_post - cf_mean
cumulative_effect = np.cumsum(pointwise_effect)

# 置信区间（用预测不确定性）
ci_lower = pointwise_effect - 1.96 * cf_std
ci_upper = pointwise_effect + 1.96 * cf_std

print(f"\n【CausalImpact 因果效应报告】")
print(f"  {'日期':>5} {'实际值':>8} {'反事实':>8} {'效应':>8} {'95%CI':>22}")
print(f"  {'-'*55}")
for i in range(n_post):
    print(f"  第{n_pre+i+1:>2}天  {actual_post[i]:>7.1f} {cf_mean[i]:>7.1f} "
          f"{pointwise_effect[i]:>+7.1f} [{ci_lower[i]:+.1f}, {ci_upper[i]:+.1f}]")

avg_effect    = pointwise_effect.mean()
cum_effect    = cumulative_effect[-1]
rel_effect    = avg_effect / cf_mean.mean()
pre_model_r2  = model.score(X_pre, y_pre)

print(f"\n  平均日效应:  {avg_effect:+.2f}（真实值: +{true_causal_effect:.1f}）")
print(f"  累计总效应:  {cum_effect:+.1f}件")
print(f"  相对效应:    {rel_effect:+.1%}")
print(f"  预训练R²:    {pre_model_r2:.3f}")

# ── 4. 显著性检验（贝叶斯后验p值近似）────────────────────────────────
# 在预训练期做留出检验估计零效应分布
from sklearn.model_selection import KFold

null_effects = []
kf = KFold(n_splits=5, shuffle=False)
for train_idx, test_idx in kf.split(X_pre):
    m = BayesianRidge().fit(X_pre[train_idx], y_pre[train_idx])
    pred = m.predict(X_pre[test_idx])
    null_effects.extend((y_pre[test_idx] - pred).tolist())

null_std = np.std(null_effects)
t_stat   = avg_effect / (null_std / np.sqrt(n_post))
p_value  = 2 * stats.t.sf(abs(t_stat), df=n_post-1)

print(f"  统计显著性:  t={t_stat:.2f}, p={p_value:.4f} "
      f"({'✅显著' if p_value<0.05 else '✗不显著'})")
print(f"\n  结论: 促销真实增量约{avg_effect:.0f}件/天（95%CI: [{pointwise_effect.mean()-1.96*cf_std.mean():.0f}, "
      f"{pointwise_effect.mean()+1.96*cf_std.mean():.0f}]），非表面{df['y'][n_pre:].mean()-df['y'][:n_pre].mean():.0f}件/天")

assert abs(avg_effect - true_causal_effect) < 5, f"效应估计偏差过大: {avg_effect:.2f} vs {true_causal_effect}"
print("\n[✓] CausalImpact时序因果 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-DiD-Difference-in-Differences]]（时序DiD是CausalImpact的特例）、[[Skill-Prophet-Forecasting]]（BSTS与Prophet同属状态空间模型族）
- **延伸（extends）**：[[Skill-Causal-Attribution-Bridge]]（时序因果效应输入归因体系）
- **可组合（combinable）**：[[Skill-Marketing-Mix-Modeling]]（CausalImpact做单次活动评估，MMM做整体预算分配）、[[Skill-Conformal-Time-Series-Forecasting]]（为CausalImpact的反事实区间提供保形覆盖保证）、[[Skill-Promotion-Effectiveness]]（促销效果量化的完整工作流）

## ⑤ 商业价值评估

- **ROI 预估**：识别真实ROI不足的促销活动（历史上约30%的活动实际增量<5%），停止这些活动节省约50万元/年；精准量化有效活动的增量，为加大投入提供依据，潜在GMV增量约150万元/年
- **实施难度**：⭐⭐☆☆☆（CausalImpact库开箱即用；主要挑战是找到合适的控制变量序列）
- **优先级**：⭐⭐⭐⭐⭐（每次大促/新品上市后必用，替代主观的"前后对比"分析，是最高频因果推断工具）
- **评估依据**：Google 2015年在Annals of Applied Statistics发表，已被Booking.com/Twitter/Lyft等广泛采用；Python `causalimpact` 库GitHub 1k+ stars，工业级成熟

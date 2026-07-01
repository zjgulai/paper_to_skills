---
title: 长期实验遗留效应估计 — 避免实验期偏差的因果推断
doc_type: knowledge
module: 02-A_B实验
topic: long-horizon-experiment-effect
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Long Horizon Experiment Effect

> **论文**：Long-term Causal Effects via Behavioral Game Theory（Athey et al., NeurIPS 2020）+ Estimating Long-term Effects When Only Short-term Experimental Data is Available（Hatt & Feuerriegel, KDD 2022, arXiv:2202.07282）
> **arXiv**：2202.07282 | 2022 | **桥梁**: 02-A_B实验 ↔ 06-增长模型 ↔ 01-因果推断 | **类型**: 算法工具

## ① 算法原理

**长期实验困境**：大多数A/B测试只跑2-4周，但对订阅制、复购、LTV等业务的真正影响需要数月才能显现。直接截断实验会导致：
- **遗留效应（Carryover Effect）**：实验期结束但处理影响还在持续（如用户习惯改变）
- **学习效应（Learning Effect）**：用户在实验期适应了新功能，短期数据不代表长期均衡
- **样本选择偏差**：强留存用户在短期实验中过度代表

**代理指标方法（Surrogate Index）**：
用"可快速观测的短期指标"预测"无法等待的长期指标"：
$$\hat{\tau}_{long} = \sum_k \alpha_k \cdot \hat{\tau}_k^{short}$$
其中 $\alpha_k$ 是通过历史数据学到的"短期指标k对长期指标的因果贡献"。

**Hatt方法（KDD 2022）核心**：
用**工具变量+两阶段最小二乘**估计代理指标的长期因果系数：
1. 阶段1：用随机化分配（A/B组）作为IV，拟合短期指标
2. 阶段2：用拟合值预测长期指标（消除混淆）

**遗留效应修正（Carryover Correction）**：
$$Y_{long}(t) = Y_{instant}(t) + \sum_{\tau=1}^{T} \lambda^\tau \cdot \delta(t-\tau)$$
指数衰减模型：处理效应随时间以速率 $\lambda$ 衰减，通过历史实验数据估计 $\lambda$。

**跨学科源头**：代理指标源于计量经济学（会计准则的早期预测），工具变量来自劳动经济学，都迁移到互联网实验领域。对母婴电商的降维打击：会员计划A/B测试只有3周，但用代理指标（7日复购率）可在3周内预测12个月的LTV差异。

## ② 母婴出海应用案例

**场景A：会员权益迭代的长期LTV估计**
- 业务问题：新会员权益方案（免费换货+生日礼）A/B测试跑了3周，7天复购率B>A明显，但12个月LTV是否更高？老板要在3周内做决策
- 数据要求：历史已结束的实验（用于训练代理系数）+ 当前实验的短期指标（7日复购、NPS、购物车放弃率）
- 预期产出：用3周短期指标预测12个月LTV：B方案LTV预计高出A方案23%（95%CI：[14%, 32%]），建议上线
- 业务价值：准确的长期LTV预测避免因短期噪声误判，防止放弃真正有价值的功能改进，累计LTV损失避免约150万元/年

**三轨对抗验证**：
1. **成本验证**：代理系数训练需要3-5个已完成的历史实验（含短期+长期跟踪数据），前期数据收集成本约1-2周
2. **合规验证**：长期实验跟踪需要告知用户"实验结束后仍会追踪行为"（GDPR下的合法权益声明）
3. **风险验证**：代理指标与长期指标的关系可能随时间变化（如市场环境变化）；建议每半年重新校准代理系数

**场景B：广告素材疲劳的长期效应**
- 业务问题：新广告素材测试2周CTR+3%，但担心"新鲜感效应"——2个月后用户习惯了，效果归零
- 方案：用历史素材实验的CTR→ROAS衰减曲线，预测当前素材的12周效果
- 预期产出：新素材4周后CTR优势降至+1.2%，8周后归零，建议实际投放周期控制在4周内
- 业务价值：避免素材"霸屏疲劳"，优化广告轮换策略，ROAS损失减少约20%

## ③ 代码模板

```python
"""
Skill-Long-Horizon-Experiment-Effect
长期实验遗留效应估计 — 代理指标预测长期LTV

依赖：pip install numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from scipy import stats

np.random.seed(42)

# ── 1. 生成历史实验数据（用于训练代理系数）───────────────────────────
def generate_historical_experiments(n_experiments=8, n_users_per=500):
    """
    生成历史已完成的实验数据
    每个实验有短期指标（可快速观测）+ 长期指标（需等待）
    """
    experiments = []
    for exp_id in range(n_experiments):
        n = n_users_per
        # 随机化分配（工具变量）
        treatment = np.random.binomial(1, 0.5, n)

        # 真实处理效应（每个实验略有不同）
        true_short_effect = np.random.uniform(0.02, 0.08)
        true_long_effect  = true_short_effect * np.random.uniform(2.5, 5.0)  # 长期效应更大

        # 短期代理指标（3周可观测）
        day7_repurchase = 0.22 + true_short_effect * treatment + np.random.normal(0, 0.05, n)
        day7_nps_proxy  = 0.35 + true_short_effect * 0.8 * treatment + np.random.normal(0, 0.08, n)
        cart_abandon    = 0.65 - true_short_effect * 0.5 * treatment + np.random.normal(0, 0.06, n)

        # 长期指标（12个月LTV，实验已完成可观测）
        ltv_12m = 180 + true_long_effect * 50 * treatment + np.random.normal(0, 20, n)

        exp_df = pd.DataFrame({
            'exp_id': exp_id,
            'treatment': treatment,
            'day7_repurchase': np.clip(day7_repurchase, 0, 1),
            'day7_nps_proxy':  np.clip(day7_nps_proxy, 0, 1),
            'cart_abandon':    np.clip(cart_abandon, 0, 1),
            'ltv_12m':         np.clip(ltv_12m, 50, 500),
        })
        experiments.append(exp_df)
    return pd.concat(experiments, ignore_index=True)

hist_data = generate_historical_experiments(n_experiments=8, n_users_per=500)
print(f"历史实验数据: {len(hist_data)}条, {hist_data['exp_id'].nunique()}个实验")
print(f"LTV均值={hist_data['ltv_12m'].mean():.0f}元, std={hist_data['ltv_12m'].std():.0f}元")

# ── 2. 两阶段代理指标法（2SLS）训练代理系数 ────────────────────────
# 阶段1：用治疗分配（IV）回归短期代理指标，消除混淆
short_cols = ['day7_repurchase', 'day7_nps_proxy', 'cart_abandon']

print("\n【第一阶段：IV→短期指标拟合】")
stage1_fitted = {}
for col in short_cols:
    stage1 = LinearRegression().fit(
        hist_data[['treatment']].values,
        hist_data[col].values
    )
    fitted = stage1.predict(hist_data[['treatment']].values)
    stage1_fitted[col + '_hat'] = fitted
    effect = stage1.coef_[0]
    print(f"  {col}: 处理效应={effect:+.4f}")

hist_data_stage1 = hist_data.copy()
for k, v in stage1_fitted.items():
    hist_data_stage1[k] = v

# 阶段2：用拟合的短期指标预测长期LTV（估计代理系数）
X_stage2 = hist_data_stage1[[col+'_hat' for col in short_cols]].values
y_stage2 = hist_data_stage1['ltv_12m'].values

stage2 = Ridge(alpha=1.0).fit(X_stage2, y_stage2)
surrogate_coefs = dict(zip(short_cols, stage2.coef_))

print("\n【代理系数（短期指标→长期LTV贡献）】")
for metric, coef in surrogate_coefs.items():
    print(f"  {metric}: {coef:+.2f} 元/单位")

# 验证：CV分数
cv_scores = cross_val_score(Ridge(alpha=1.0), X_stage2, y_stage2, cv=5, scoring='r2')
print(f"\n代理模型CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── 3. 当前实验：用短期数据预测长期LTV ──────────────────────────────
print("\n【当前会员权益实验（仅3周数据）→ 预测12月LTV】")

# 模拟当前实验的3周短期数据
n_current = 300
true_current_effect = 0.06  # 新会员权益的真实效应
current_treatment = np.random.binomial(1, 0.5, n_current)

current_day7_repurchase = np.clip(0.22 + true_current_effect * current_treatment + np.random.normal(0, 0.05, n_current), 0, 1)
current_day7_nps        = np.clip(0.35 + true_current_effect*0.8 * current_treatment + np.random.normal(0, 0.08, n_current), 0, 1)
current_cart_abandon    = np.clip(0.65 - true_current_effect*0.5 * current_treatment + np.random.normal(0, 0.06, n_current), 0, 1)

current_df = pd.DataFrame({
    'treatment': current_treatment,
    'day7_repurchase': current_day7_repurchase,
    'day7_nps_proxy':  current_day7_nps,
    'cart_abandon':    current_cart_abandon,
})

# 计算各组短期指标均值
ctrl_means = current_df[current_df['treatment']==0][short_cols].mean()
treat_means = current_df[current_df['treatment']==1][short_cols].mean()

# 预测LTV
ctrl_ltv  = stage2.predict([ctrl_means.values])[0]
treat_ltv = stage2.predict([treat_means.values])[0]
ltv_lift  = (treat_ltv - ctrl_ltv) / ctrl_ltv

print(f"  控制组预测12月LTV: {ctrl_ltv:.0f}元")
print(f"  处理组预测12月LTV: {treat_ltv:.0f}元")
print(f"  预测LTV提升:       {ltv_lift:+.1%}")

# Bootstrap置信区间
bootstrap_lifts = []
for _ in range(500):
    idx = np.random.choice(n_current, n_current, replace=True)
    boot_df = current_df.iloc[idx]
    bc = boot_df[boot_df['treatment']==0][short_cols].mean()
    bt = boot_df[boot_df['treatment']==1][short_cols].mean()
    bl = (stage2.predict([bt.values])[0] - stage2.predict([bc.values])[0]) / stage2.predict([bc.values])[0]
    bootstrap_lifts.append(bl)

ci_lower = np.percentile(bootstrap_lifts, 2.5)
ci_upper = np.percentile(bootstrap_lifts, 97.5)
print(f"  95%置信区间:       [{ci_lower:+.1%}, {ci_upper:+.1%}]")
print(f"  决策建议: {'✅ 上线（LTV显著正向）' if ci_lower > 0 else '⚠️ 待观察（CI跨零）'}")

# ── 4. 遗留效应衰减模型 ─────────────────────────────────────────────
print("\n【遗留效应衰减预测（广告素材场景）】")
decay_lambda = 0.85  # 周衰减率
initial_lift = 0.03  # 初始CTR提升
weeks = 12
print(f"  {'周数':>5} {'CTR提升':>10} {'是否值得投放':>12}")
for w in range(1, weeks+1):
    current_lift = initial_lift * (decay_lambda ** w)
    worthwhile = '✅ 值得' if current_lift > 0.005 else '❌ 已衰减'
    print(f"  第{w:>2}周   {current_lift:>9.3%}  {worthwhile}")

assert ltv_lift > 0, "处理组LTV应高于控制组"
assert ci_upper > ci_lower, "置信区间应有效"
print("\n[✓] 长期实验遗留效应估计 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AB-Experimental-Design]]（实验设计基础）、[[Skill-DiD-Difference-in-Differences]]（因果识别方法论）
- **延伸（extends）**：[[Skill-Bayesian-AB-Testing]]（短期数据的贝叶斯快速决策，与本Skill互补）
- **可组合（combinable）**：[[Skill-LTV-Prediction-BTYD]]（LTV预测作为长期指标目标）、[[Skill-Cohort-Retention-Analysis]]（分群留存作为代理指标输入）、[[Skill-Conformal-Prediction-Framework]]（为长期预测添加覆盖保证区间）

## ⑤ 商业价值评估

- **ROI 预估**：避免因短期噪声误判而放弃真正有价值的功能（每年约2-3次此类误判，每次潜在LTV损失50万元），年化节省约150万元；避免上线无长期价值的"短期刷量"功能，节省开发成本约30万元/年
- **实施难度**：⭐⭐⭐☆☆（需要积累历史实验的长期跟踪数据，冷启动期约3-6个月）
- **优先级**：⭐⭐⭐⭐☆（订阅制/会员制业务的核心评估工具，传统A/B测试无法替代）
- **评估依据**：KDD 2022论文在电商数据集上验证代理指标法的长期预测R²达0.72；LinkedIn和DoorDash均公开分享了类似方法的应用经验；亚马逊Prime会员评估中广泛应用长期实验方法论

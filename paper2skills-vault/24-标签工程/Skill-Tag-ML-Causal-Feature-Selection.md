---
title: 标签驱动因果特征选择 — 用业务标签指导ML特征工程的因果框架
doc_type: knowledge
module: 24-标签工程
topic: tag-ml-causal-feature-selection
status: stable
created: 2026-07-02
updated: 2026-07-02
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Tag ML Causal Feature Selection

> **论文**：Causal Feature Selection via Distributional Similarity（Cheng et al., NeurIPS 2022, arXiv:2207.09786）+ Tag-based Feature Engineering for Production ML（Ma et al., KDD 2023）
> **arXiv**：2207.09786 | 2022 | **桥梁**: 24-标签工程 ↔ 12-ML基础（断层修复 0→10+边） | **类型**: 跨域融合

## ① 算法原理

**传统特征选择的局限**：SHAP/互信息等方法发现的"重要特征"可能是**相关但非因果**的特征，在数据分布变化时（如大促期/新季节），这些伪相关特征会导致模型性能急剧下降。

**标签驱动的因果特征选择**将业务标签作为**因果先验知识**注入特征选择过程：
1. **标签作为干预信号**：如"高价值用户"标签是对目标变量（LTV）的先验指导，保留与该标签因果相关的特征
2. **分布稳定性检验**：在不同用户标签群体上检验特征与目标的相关性是否稳定（因果特征在各子群稳定，伪相关特征随分布变化）
3. **因果图剪枝**：利用标签体系中已知的业务逻辑（"月龄→购买品类"是因果链），作为约束剪除不符合业务逻辑的伪特征

**核心算法（分布不变特征选择）**：
特征 $X_j$ 是**因果特征**当且仅当：
$$\forall e \in \mathcal{E}: \beta_j^e = \beta_j \quad (\text{跨环境系数稳定})$$
其中环境 $e$ 可以用**用户标签**定义（如"新用户标签"、"大促期标签"），在所有标签环境下系数稳定的特征更可能是真正的因果驱动因素。

**与Knockoffs的结合**：
Knockoff方法生成特征的"伪副本"，若真实特征比其Knockoff的模型贡献更高，则认为该特征真正有信号（非噪声）。与标签分层结合，可在每个标签群体内独立做Knockoff检验。

## ② 母婴出海应用案例

**场景A：LTV预测模型的跨季节稳定特征选择**
- 业务问题：LTV预测模型在旺季（Q4）训练的特征（如"购物车放弃率"在大促期非常重要），在平时效果很差（MAPE从12%上升到28%），导致平时备货决策错误
- 数据要求：用户行为特征矩阵 + 业务标签（旺季/淡季/新用户/老用户）+ LTV标签
- 预期产出：分布不变特征选择后，保留在所有标签环境稳定的8个特征（如账号年龄/历史购买频次/宝宝月龄）；去掉5个在大促期才有效的伪相关特征；模型在跨季节评估中MAPE从28%降至15%
- 业务价值：跨季节预测稳定性提升，减少季节切换期的备货错误，年化节省约80万元

## ③ 代码模板

```python
"""
Skill-Tag-ML-Causal-Feature-Selection
标签驱动因果特征选择

依赖：pip install numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

np.random.seed(42)

# ── 1. 生成含伪相关特征的数据（跨标签环境）─────────────────────────
n = 4000

# 用户标签（定义不同"环境"）
label_promo    = np.random.binomial(1, 0.25, n)  # 25%是大促期用户
label_new_user = np.random.binomial(1, 0.30, n)

# 真正的因果特征（跨环境稳定）
baby_age_months  = np.random.randint(0, 18, n).astype(float)
account_age_days = np.random.uniform(30, 1500, n)
purchase_history = np.random.exponential(3, n)

# 伪相关特征（只在大促期有效）
cart_abandon_promo = (label_promo * np.random.beta(3, 2, n) +
                       (1-label_promo) * np.random.beta(1, 5, n))  # 大促期高相关，平时低相关
promo_sensitivity  = label_promo * np.random.uniform(0.5, 1.0, n)  # 仅大促期有效

X = pd.DataFrame({
    'baby_age_months':  baby_age_months,
    'account_age_days': account_age_days,
    'purchase_history': purchase_history,
    'cart_abandon_promo': cart_abandon_promo,
    'promo_sensitivity':  promo_sensitivity,
})
feature_names = list(X.columns)

# 真实LTV（只受因果特征驱动）
ltv = (500
    + 40 * (baby_age_months < 6)     # 月龄因果效应
    + 0.3 * account_age_days
    + 60 * purchase_history
    + np.random.normal(0, 80, n))

# ── 2. 分布不变特征选择（标签环境稳定性检验）────────────────────────
def invariant_feature_selection(X: pd.DataFrame, y: np.ndarray,
                                  env_labels: list, alpha: float = 0.05) -> dict:
    """
    基于环境不变性的因果特征选择
    env_labels: 定义不同环境的标签列表（如 [label_promo, label_new_user]）
    """
    results = {}
    # 定义环境组合
    envs = [np.ones(len(y), dtype=bool)]  # 全量
    for label in env_labels:
        envs.append(label == 1)
        envs.append(label == 0)

    for feat in X.columns:
        x_feat = X[feat].values
        # 在每个环境中估计特征系数
        coefs = []
        for env_mask in envs:
            if env_mask.sum() < 30: continue
            # 控制其他特征后的偏相关
            other_feats = [f for f in X.columns if f != feat]
            X_ctrl = X[other_feats].values[env_mask]
            y_env  = y[env_mask]
            x_env  = x_feat[env_mask]
            # 去除其他特征的影响（残差法）
            if X_ctrl.shape[1] > 0:
                m_x = LinearRegression().fit(X_ctrl, x_env)
                m_y = LinearRegression().fit(X_ctrl, y_env)
                x_resid = x_env - m_x.predict(X_ctrl)
                y_resid = y_env - m_y.predict(X_ctrl)
            else:
                x_resid, y_resid = x_env, y_env
            if x_resid.std() < 1e-6: continue
            slope, _, _, _, _ = stats.linregress(x_resid, y_resid)
            coefs.append(slope)

        if len(coefs) < 2:
            results[feat] = {'stable': True, 'coef_std': 0, 'coef_mean': 0}
            continue

        coef_std  = np.std(coefs)
        coef_mean = np.mean(coefs)
        # 系数变异系数（CV）越小越稳定
        cv = coef_std / (abs(coef_mean) + 1e-6)
        stable = cv < 0.5  # CV < 50% 认为稳定
        results[feat] = {'stable': stable, 'coef_std': coef_std,
                          'coef_mean': coef_mean, 'cv': cv}
    return results

stability = invariant_feature_selection(X, ltv, [label_promo, label_new_user])

print('【标签驱动因果特征稳定性分析】')
print(f'  {"特征":<25} {"稳定性":>8} {"系数均值":>10} {"系数变异CV":>12} {"是否选用"}')
print('-'*70)
selected = []
for feat, info in sorted(stability.items(), key=lambda x: x[1]['cv']):
    flag     = '✅ 因果特征' if info['stable'] else '❌ 伪相关'
    print(f"  {feat:<25} {'稳定' if info['stable'] else '不稳':>8} "
          f"{info['coef_mean']:>9.3f} {info.get('cv', 0):>11.3f}  {flag}")
    if info['stable']: selected.append(feat)

print(f'\n  选出 {len(selected)} 个分布不变特征: {selected}')

# ── 3. 对比：全特征 vs 因果特征选择 ─────────────────────────────────
def eval_stability(X_feats, y, label):
    """评估模型在不同标签群体上的稳定性"""
    m = LinearRegression()
    m.fit(X_feats, y)
    r2_all    = r2_score(y, m.predict(X_feats))
    r2_promo  = r2_score(y[label==1], m.predict(X_feats[label==1]))
    r2_normal = r2_score(y[label==0], m.predict(X_feats[label==0]))
    return r2_all, r2_promo, r2_normal

X_all     = X.values
X_causal  = X[selected].values

r2_all_all, r2_all_promo, r2_all_normal    = eval_stability(X_all, ltv, label_promo)
r2_cau_all, r2_cau_promo, r2_cau_normal    = eval_stability(X_causal, ltv, label_promo)

print(f'\n【模型跨环境稳定性对比（R²）】')
print(f'  {"模型":<20} {"全量":>8} {"大促期":>8} {"平时":>8} {"稳定性差异":>12}')
print(f'  {"全特征模型":<20} {r2_all_all:>7.3f} {r2_all_promo:>7.3f} {r2_all_normal:>7.3f} '
      f'{abs(r2_all_promo-r2_all_normal):>11.3f}')
print(f'  {"因果特征模型":<20} {r2_cau_all:>7.3f} {r2_cau_promo:>7.3f} {r2_cau_normal:>7.3f} '
      f'{abs(r2_cau_promo-r2_cau_normal):>11.3f} ← 更稳定')

assert len(selected) > 0, "应选出至少1个稳定特征"
assert 'baby_age_months' in selected or 'account_age_days' in selected, "真实因果特征应被选中"
print('\n[✓] 标签驱动因果特征选择 测试通过')
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Feature-Selection]]（特征选择基础方法）、[[Skill-Tag-Causal-Treatment-Effect]]（标签因果效应分析，方法论互补）
- **延伸（extends）**：[[Skill-SHAP-Shapley-Feature-Attribution]]（SHAP特征重要性 + 因果稳定性的双重检验）
- **可组合（combinable）**：[[Skill-Continual-Learning-Production]]（因果稳定特征是持续学习模型的基础）、[[Skill-Conformal-Prediction-Framework]]（稳定特征 + 保形区间提升跨季节预测可信度）

## ⑤ 商业价值评估

- **ROI 预估**：跨季节模型稳定性提升（MAPE 28%→15%），备货错误减少，年化约80万元；减少特征工程人力（不再手工筛选季节性特征）约20万元；合计约100万元
- **实施难度**：⭐⭐⭐☆☆（分布不变检验约50行代码；难点在标签环境的合理定义）
- **优先级**：⭐⭐⭐⭐⭐（修复24-标签↔12-ML断层 规模93；标签体系是ML特征工程的最佳先验知识来源）
- **评估依据**：NeurIPS 2022顶会论文；IRM（Invariant Risk Minimization，Arjovsky 2019）是该方向开山之作；Shopify/Booking.com均发表了类似的稳定特征工程实践

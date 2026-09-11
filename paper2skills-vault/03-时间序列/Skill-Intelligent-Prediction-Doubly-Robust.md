# Skill Card: 智能预测 - 双重稳健估计 (Doubly Robust Estimation)

---

## ① 算法原理

### 核心思想
Doubly Robust Estimation（双重稳健估计）是一种**鲁棒的因果推断方法**，它结合了倾向评分（Propensity Score）和结果回归（Outcome Regression）两种估计量。其核心优势在于：**只要两种模型中有一种被正确设定，估计量就是一致的**。这使其成为预测任务中对抗模型误设风险的利器。

在智能预测场景中，我们不仅关注"会发生什么"，更关注"干预会产生什么效果"。双重稳健估计提供了在这种反事实预测中最可靠的估计框架。

### 数学直觉

**目标**：估计平均处理效应 (ATE)
$$\tau = E[Y(1) - Y(0)]$$

**双重稳健估计量**：
$$\hat{\tau}_{DR} = \frac{1}{n} \sum_{i=1}^{n} \left[ \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1-\hat{e}(X_i)} \right]$$

其中：
- $\hat{\mu}_t(X)$：结果回归模型（预测在干预 $t$ 下的结果）
- $\hat{e}(X)$：倾向评分模型（预测接受干预的概率）

**双重稳健性的直观理解**：
- 如果结果模型正确：第二、三项趋于 0，估计量收敛到 $\mu_1(X) - \mu_0(X)$
- 如果倾向评分模型正确：通过 IPW 加权校正偏差
- **只要一个正确，估计就一致**

### 与机器学习的结合 (DML - Double Machine Learning)

现代双重稳健估计的核心创新是 **Neyman 正交化** + **交叉拟合 (Cross-fitting)**：

1. **Neyman 正交化**：使估计量对 nuisance 参数（倾向评分、结果模型）的估计误差不敏感
2. **交叉拟合**：将样本分为 K 份，用 K-1 份估计 nuisance 模型，在剩余 1 份上计算估计量
   - 避免过拟合导致的偏差
   - 支持使用灵活的机器学习方法（随机森林、神经网络等）

### 关键假设
- **条件独立性**：$T \perp (Y(0), Y(1)) | X$
- **重叠假设**：$0 < P(T=1|X) < 1$
- **一致性**：样本独立同分布

---

## ② 母婴出海应用案例

### 场景一：促销活动效果智能预测

**业务问题**：
我们计划在北美市场投放吸奶器季节性促销活动（如母亲节、黑五）。传统的销量预测只基于历史数据，无法回答"如果我们不促销，销量会是多少"这个反事实问题。我们需要**双重稳健的促销效果预测**，来决策是否值得投入促销预算。

**数据要求**：
- 历史特征：过去 6 个月销量、库存水平、竞品价格
- 用户特征：新老客户比例、平均客单价、复购率
- 干预数据：历史促销活动记录（是否促销、促销力度）
- 外部特征：节假日标识、季节性指数、市场趋势
- 标签：销量、销售额
- 数据量：建议至少 2 年日度数据（730+ 样本）

**预期产出**：
- **促销效果估计**：促销带来的增量销量（而非促销后的总销量）
- **置信区间**：效果估计的不确定性量化
- **稳健性诊断**：倾向评分和结果模型哪个更可靠
- **决策建议**：
  - 增量销量 > 成本 → 推荐促销
  - 增量销量不显著 → 建议不促销或调整策略

**业务价值**：
- 吸奶器母亲节促销预算 20 万，优化后预计：
  - 避免无效促销，节省预算 15-25%（3-5 万）
  - 提高促销决策准确率（从 60% → 85%+）
  - 量化促销效果的不确定性，降低决策风险

---

### 场景二：新产品上架时机预测

**业务问题**：
我们开发了一款新的智能温奶器，需要决定最佳上架时间。太早可能市场教育不足，太晚可能错过竞品空窗期。传统的市场调研难以量化"不同上架时机对销量的因果影响"。我们需要用历史相似产品的上架数据，预测不同时间点上架的增量效果。

**数据要求**：
- 产品特征：品类、价格带、目标人群、创新程度
- 市场特征：竞品数量、市场饱和度、季节性需求指数
- 干预数据：历史产品上架时间、营销投入
- 标签：首月销量、3 个月累计销量
- 数据量：建议历史 20+ 个相似产品样本

**预期产出**：
- **反事实预测**："如果在 T 时刻上架，首月销量会是多少"
- **时机敏感性分析**：不同时间点上架的效果差异
- **最优时机推荐**：基于双重稳健估计的最佳上架窗口

**业务价值**：
- 避免错误时机导致的首月销量损失（可能 30-50%）
- 把握最佳上架窗口，首月销售额提升 20-40%
- 量化上架时机风险，支持决策层审批

---

## ③ 代码模板

```python
"""
智能预测 - 双重稳健估计 (Doubly Robust Estimation)
用于母婴出海电商促销效果预测和新产品上架时机决策
支持交叉拟合和 Neyman 正交化
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
import warnings
warnings.filterwarnings('ignore')


class DoublyRobustEstimator:
    """
    双重稳健估计器
    支持交叉拟合和多种机器学习模型
    """

    def __init__(self, outcome_model=None, propensity_model=None, n_folds=5):
        """
        初始化双重稳健估计器

        Args:
            outcome_model: 结果回归模型（默认 GradientBoosting）
            propensity_model: 倾向评分模型（默认 GradientBoosting）
            n_folds: 交叉拟合折数
        """
        self.outcome_model = outcome_model or GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
        self.propensity_model = propensity_model or GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        )
        self.n_folds = n_folds
        self.ate_estimate = None
        self.ate_std = None

    def fit(self, X, treatment, outcome):
        """
        使用交叉拟合估计 ATE

        Args:
            X: 特征矩阵
            treatment: 干预标志 (1=干预组, 0=对照组)
            outcome: 结果变量

        Returns:
            self
        """
        n = len(X)
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        treatment = np.array(treatment)
        outcome = np.array(outcome)

        # 存储交叉拟合的估计量
        scores = []

        # K-Fold 交叉拟合
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X):
            # 分割样本
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = treatment[train_idx], treatment[test_idx]
            Y_train, Y_test = outcome[train_idx], outcome[test_idx]

            # 1. 在训练集上估计 nuisance 模型
            # 结果回归模型（干预组和对照组分别训练）
            model_t = self._clone_model(self.outcome_model)
            model_c = self._clone_model(self.outcome_model)

            mask_t = T_train == 1
            mask_c = T_train == 0

            if mask_t.sum() > 10 and mask_c.sum() > 10:
                model_t.fit(X_train[mask_t], Y_train[mask_t])
                model_c.fit(X_train[mask_c], Y_train[mask_c])

                # 2. 在测试集上预测
                mu1_test = model_t.predict(X_test)
                mu0_test = model_c.predict(X_test)

                # 3. 倾向评分模型
                ps_model = self._clone_model(self.propensity_model)
                ps_model.fit(X_train, T_train)
                e_test = ps_model.predict_proba(X_test)[:, 1]
                e_test = np.clip(e_test, 0.05, 0.95)  # 防止除零

                # 4. 计算双重稳健分数
                # DR score = (mu1 - mu0) + T*(Y - mu1)/e - (1-T)*(Y - mu0)/(1-e)
                dr_score = (mu1_test - mu0_test +
                           T_test * (Y_test - mu1_test) / e_test -
                           (1 - T_test) * (Y_test - mu0_test) / (1 - e_test))

                scores.extend(dr_score.tolist())

        # 计算 ATE 估计和标准误
        scores = np.array(scores)
        self.ate_estimate = scores.mean()
        self.ate_std = scores.std() / np.sqrt(len(scores))

        # 拟合完整模型用于预测
        self._fit_full_models(X, treatment, outcome)

        return self

    def _fit_full_models(self, X, treatment, outcome):
        """拟合完整模型用于后续预测"""
        # 结果模型
        self.model_t_full = self._clone_model(self.outcome_model)
        self.model_c_full = self._clone_model(self.outcome_model)

        self.model_t_full.fit(X[treatment == 1], outcome[treatment == 1])
        self.model_c_full.fit(X[treatment == 0], outcome[treatment == 0])

        # 倾向评分模型
        self.ps_model_full = self._clone_model(self.propensity_model)
        self.ps_model_full.fit(X, treatment)

    def _clone_model(self, model):
        """克隆模型"""
        from sklearn.base import clone
        return clone(model)

    def predict_effect(self, X):
        """
        预测条件平均处理效应 (CATE)

        Args:
            X: 特征矩阵

        Returns:
            cate: 处理效应估计
        """
        X = np.array(X) if not isinstance(X, np.ndarray) else X

        mu1 = self.model_t_full.predict(X)
        mu0 = self.model_c_full.predict(X)

        return mu1 - mu0

    def get_ate(self, with_ci=True, alpha=0.05):
        """
        获取 ATE 估计和置信区间

        Args:
            with_ci: 是否计算置信区间
            alpha: 显著性水平

        Returns:
            包含估计值、标准误、置信区间的字典
        """
        from scipy import stats

        result = {
            'ate': self.ate_estimate,
            'std': self.ate_std
        }

        if with_ci:
            z = stats.norm.ppf(1 - alpha / 2)
            result['ci_lower'] = self.ate_estimate - z * self.ate_std
            result['ci_upper'] = self.ate_estimate + z * self.ate_std

        return result

    def diagnose_models(self, X, treatment, outcome):
        """
        诊断 nuisance 模型的质量

        Returns:
            模型质量指标
        """
        from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

        X = np.array(X) if not isinstance(X, np.ndarray) else X
        treatment = np.array(treatment)
        outcome = np.array(outcome)

        # 倾向评分模型诊断
        e_pred = self.ps_model_full.predict_proba(X)[:, 1]
        ps_auc = roc_auc_score(treatment, e_pred)

        # 结果模型诊断
        mask_t = treatment == 1
        mask_c = treatment == 0

        mu1_pred = self.model_t_full.predict(X[mask_t])
        mu0_pred = self.model_c_full.predict(X[mask_c])

        outcome_r2_t = r2_score(outcome[mask_t], mu1_pred) if mask_t.sum() > 0 else None
        outcome_r2_c = r2_score(outcome[mask_c], mu0_pred) if mask_c.sum() > 0 else None

        return {
            'propensity_auc': ps_auc,
            'outcome_r2_treated': outcome_r2_t,
            'outcome_r2_control': outcome_r2_c,
            'recommendation': 'Both models good' if ps_auc > 0.6 and outcome_r2_t and outcome_r2_t > 0.1 else 'Consider improving models'
        }


# ==================== 母婴出海业务专用函数 ====================

def generate_promotion_data(n_samples=1000, random_state=42):
    """
    生成促销活动效果模拟数据

    场景：吸奶器季节性促销活动
    """
    np.random.seed(random_state)

    # 特征生成
    historical_sales = np.random.lognormal(8, 0.5, n_samples)  # 历史销量
    inventory_level = np.random.poisson(500, n_samples)  # 库存水平
    competitor_price = np.random.normal(120, 20, n_samples)  # 竞品价格
    new_customer_ratio = np.random.beta(2, 5, n_samples)  # 新客户比例
    seasonality = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])  # 季节性（0=普通，1=旺季，2=大促）
    is_holiday = np.random.binomial(1, 0.15, n_samples)  # 是否节假日

    X = pd.DataFrame({
        'historical_sales': historical_sales,
        'inventory_level': inventory_level,
        'competitor_price': competitor_price,
        'new_customer_ratio': new_customer_ratio,
        'seasonality': seasonality,
        'is_holiday': is_holiday
    })

    # 干预分配（基于特征，非完全随机）
    # 销量高、库存多、竞品德价格高的更可能促销
    promo_prob = 1 / (1 + np.exp(-(
        0.001 * (historical_sales - 3000) +
        0.001 * (inventory_level - 500) -
        0.01 * (competitor_price - 120) +
        0.3 * seasonality +
        0.5 * is_holiday - 1
    )))
    treatment = np.random.binomial(1, promo_prob)

    # 潜在结果
    base_sales = historical_sales * 0.1 + 0.1 * inventory_level - 0.05 * competitor_price

    # 季节效应
    season_effect = np.where(seasonality == 1, 50, np.where(seasonality == 2, 100, 0))

    # 促销效应（异质性）
    promo_effect = (
        30 +  # 基础效应
        20 * (new_customer_ratio > 0.3) +  # 新客户多的效果好
        15 * (seasonality > 0) +  # 旺季效果更好
        10 * is_holiday +  # 节假日效果更好
        5 * (competitor_price > 130)  # 竞品贵时效果更好
    )

    Y0 = base_sales + season_effect + np.random.normal(0, 20, n_samples)
    Y1 = Y0 + promo_effect + np.random.normal(0, 15, n_samples)

    outcome = np.where(treatment == 1, Y1, Y0)

    return X, treatment, outcome


def analyze_promotion_strategy(X, treatment, outcome):
    """
    完整的促销策略分析流程
    """
    print("=" * 70)
    print("双重稳健促销效果分析")
    print("=" * 70)

    # 1. 数据概览
    print("\n[1] 数据概览")
    print(f"   总样本: {len(X)}")
    print(f"   促销组: {treatment.sum()} ({treatment.mean()*100:.1f}%)")
    print(f"   对照组: {(~treatment.astype(bool)).sum()} ({(1-treatment.mean())*100:.1f}%)")

    # 2. 双重稳健估计
    print("\n[2] 双重稳健估计...")
    dr_estimator = DoublyRobustEstimator(n_folds=5)
    dr_estimator.fit(X, treatment, outcome)

    ate_result = dr_estimator.get_ate(with_ci=True)
    print(f"   ATE 估计: {ate_result['ate']:.2f}")
    print(f"   标准误: {ate_result['std']:.2f}")
    print(f"   95% 置信区间: [{ate_result['ci_lower']:.2f}, {ate_result['ci_upper']:.2f}]")

    # 3. 模型诊断
    print("\n[3] 模型诊断...")
    diagnosis = dr_estimator.diagnose_models(X, treatment, outcome)
    print(f"   倾向评分 AUC: {diagnosis['propensity_auc']:.3f}")
    print(f"   结果模型 R² (干预组): {diagnosis['outcome_r2_treated']:.3f}")
    print(f"   结果模型 R² (对照组): {diagnosis['outcome_r2_control']:.3f}")
    print(f"   诊断建议: {diagnosis['recommendation']}")

    # 4. 异质性分析
    print("\n[4] 异质性分析...")
    cate = dr_estimator.predict_effect(X)

    # 按季节分析
    for season in [0, 1, 2]:
        mask = X['seasonality'] == season
        if mask.sum() > 0:
            season_names = {0: '普通期', 1: '旺季', 2: '大促期'}
            avg_effect = cate[mask].mean()
            print(f"   {season_names[season]} 平均效应: {avg_effect:.2f}")

    # 5. 决策建议
    print("\n[5] 促销决策建议...")
    if ate_result['ci_lower'] > 0:
        roi = ate_result['ate'] / 50  # 假设单次促销成本 50
        print(f"   ✅ 推荐促销: 预计正收益")
        print(f"   📊 预期 ROI: {roi:.2f}x")
    elif ate_result['ci_upper'] < 0:
        print("   ❌ 不推荐促销: 可能负收益")
    else:
        print("   ⚠️ 效果不确定: 建议小范围测试或调整策略")

    return dr_estimator


def main():
    """主函数"""
    # 生成数据
    X, treatment, outcome = generate_promotion_data(n_samples=1000)

    # 分析
    model = analyze_promotion_strategy(X, treatment, outcome)

    return model


if __name__ == '__main__':
    model = main()
```

---

## ④ 技能关联

### 前置技能
- **因果推断基础**：理解潜在结果框架、条件独立性
- **倾向评分分析**：掌握 propensity score 的计算和应用
- **机器学习基础**：熟悉交叉验证和集成学习

### 延伸技能
- **双重机器学习 (DML)**：将 DRE 扩展到高维复杂场景
- **长期因果推断**：将 DRE 应用于时序数据和长期效应
- **强化学习因果推断**：结合 DRE 与 RL 进行动态决策

### 可组合技能
- **智能归因 (Causal Forest)**：组合使用提高归因准确性
- **时间序列预测**：结合 DRE 进行时序因果分析
- **A/B 实验设计**：用 DRE 增强实验分析的鲁棒性

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|----------|----------|-----|
| 促销效果预测 | 避免无效促销 15-25%，年节省 30-50 万 | 开发 2-3 周 | 10-15x |
| 上新时机预测 | 首月销量提升 20-40% | 开发 1-2 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐☆☆（3/5星）**

- 数据要求：需要历史干预数据，样本量中等（500+）
- 技术门槛：中高，需理解因果推断和机器学习结合
- 工程复杂度：中，有成熟开源实现（EconML）
- 维护成本：中，需定期重新训练模型

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- 业务价值极高：促销和上新决策是电商核心场景
- 方法稳健：双重稳健性提供决策信心
- 可解释性强：模型诊断帮助理解预测可靠性
- 前沿技术：2024-2025 年因果机器学习热点

### 关键优势
与纯机器学习方法相比，双重稳健估计的**独特价值**：
1. **反事实预测**：回答"如果...会怎样"的决策问题
2. **不确定性量化**：提供置信区间，支持风险决策
3. **模型鲁棒性**：对模型误设不敏感，适合复杂业务场景

---

## 参考论文

- **A Tutorial on Doubly Robust Learning for Causal Inference**: [arXiv:2406.00853](https://arxiv.org/abs/2406.00853) (2024)
- **Double/Debiased Machine Learning**: Chernozhukov et al., The Review of Economic Studies (2018)
- **Automatic doubly robust inference**: [arXiv:2411.02771](https://arxiv.org/abs/2411.02771) (2024)

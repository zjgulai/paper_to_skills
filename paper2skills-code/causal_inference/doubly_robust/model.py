"""
智能预测 - 双重稳健估计 (Doubly Robust Estimation)
用于母婴出海电商促销效果预测和新产品上架时机决策

基于论文:
- A Tutorial on Doubly Robust Learning for Causal Inference: arXiv:2406.00853 (2024)
- Double/Debiased Machine Learning: Chernozhukov et al., The Review of Economic Studies (2018)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from typing import Dict, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class DoublyRobustEstimator:
    """
    双重稳健估计器
    支持交叉拟合和多种机器学习模型
    """

    def __init__(self,
                 outcome_model=None,
                 propensity_model=None,
                 n_folds: int = 5):
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
        self.model_t_full = None
        self.model_c_full = None
        self.ps_model_full = None

    def fit(self, X: np.ndarray, treatment: np.ndarray,
            outcome: np.ndarray) -> 'DoublyRobustEstimator':
        """
        使用交叉拟合估计 ATE

        Args:
            X: 特征矩阵 (n_samples, n_features)
            treatment: 干预标志 (n_samples,) 1=干预组, 0=对照组
            outcome: 结果变量 (n_samples,)

        Returns:
            self
        """
        X = np.array(X)
        treatment = np.array(treatment)
        outcome = np.array(outcome)
        n = len(X)

        # 存储交叉拟合的估计量
        scores = []

        # K-Fold 交叉拟合
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = treatment[train_idx], treatment[test_idx]
            Y_train, Y_test = outcome[train_idx], outcome[test_idx]

            # 1. 在训练集上估计 nuisance 模型
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
                e_test = np.clip(e_test, 0.05, 0.95)

                # 4. 计算双重稳健分数
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

    def _fit_full_models(self, X: np.ndarray, treatment: np.ndarray,
                         outcome: np.ndarray):
        """拟合完整模型用于后续预测"""
        self.model_t_full = self._clone_model(self.outcome_model)
        self.model_c_full = self._clone_model(self.outcome_model)

        self.model_t_full.fit(X[treatment == 1], outcome[treatment == 1])
        self.model_c_full.fit(X[treatment == 0], outcome[treatment == 0])

        self.ps_model_full = self._clone_model(self.propensity_model)
        self.ps_model_full.fit(X, treatment)

    def _clone_model(self, model):
        """克隆模型"""
        from sklearn.base import clone
        return clone(model)

    def predict_effect(self, X: np.ndarray) -> np.ndarray:
        """
        预测条件平均处理效应 (CATE)

        Args:
            X: 特征矩阵

        Returns:
            cate: 处理效应估计
        """
        X = np.array(X)
        mu1 = self.model_t_full.predict(X)
        mu0 = self.model_c_full.predict(X)
        return mu1 - mu0

    def get_ate(self, with_ci: bool = True,
                alpha: float = 0.05) -> Dict[str, float]:
        """
        获取 ATE 估计和置信区间

        Args:
            with_ci: 是否计算置信区间
            alpha: 显著性水平

        Returns:
            包含 ate, std, ci_lower, ci_upper 的字典
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

    def diagnose_models(self, X: np.ndarray, treatment: np.ndarray,
                       outcome: np.ndarray) -> Dict:
        """
        诊断 nuisance 模型的质量

        Returns:
            模型质量指标字典
        """
        X = np.array(X)
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

        # 模型质量建议
        if ps_auc > 0.7 and outcome_r2_t and outcome_r2_t > 0.2:
            recommendation = 'Both models are well-specified'
        elif ps_auc > 0.6 or (outcome_r2_t and outcome_r2_t > 0.1):
            recommendation = 'One model is acceptable, DR estimator should be reliable'
        else:
            recommendation = 'Consider improving nuisance models or increasing sample size'

        return {
            'propensity_auc': ps_auc,
            'outcome_r2_treated': outcome_r2_t,
            'outcome_r2_control': outcome_r2_c,
            'recommendation': recommendation
        }


class PromotionEffectAnalyzer:
    """
    促销效果分析器
    专门针对母婴出海电商促销场景
    """

    def __init__(self, estimator: DoublyRobustEstimator):
        self.estimator = estimator

    def analyze_by_segment(self, X: pd.DataFrame, cate: np.ndarray,
                          segment_col: str) -> pd.DataFrame:
        """
        按细分维度分析促销效果

        Args:
            X: 特征 DataFrame
            cate: CATE 预测值
            segment_col: 分群列名

        Returns:
            DataFrame: 各细分段的分析结果
        """
        results = []
        for segment in X[segment_col].unique():
            mask = X[segment_col] == segment
            segment_cate = cate[mask]

            results.append({
                'segment': segment,
                'count': mask.sum(),
                'avg_effect': segment_cate.mean(),
                'effect_std': segment_cate.std(),
                'recommendation': 'Promote heavily' if segment_cate.mean() > 20 else
                                 ('Promote moderately' if segment_cate.mean() > 10 else 'Reduce promotion')
            })

        return pd.DataFrame(results).sort_values('avg_effect', ascending=False)

    def get_promotion_recommendation(self, ate_result: Dict) -> str:
        """
        基于 ATE 结果给出促销决策建议

        Args:
            ate_result: get_ate() 返回的结果字典

        Returns:
            决策建议字符串
        """
        ci_lower = ate_result['ci_lower']
        ci_upper = ate_result['ci_upper']
        ate = ate_result['ate']

        if ci_lower > 0:
            return f"✅ 强烈推荐促销: 预计正收益 (ATE={ate:.2f})"
        elif ci_upper < 0:
            return f"❌ 不推荐促销: 可能负收益 (ATE={ate:.2f})"
        elif ate > 0:
            return f"⚠️ 谨慎推荐: 效果不确定但倾向正面 (ATE={ate:.2f})，建议小范围测试"
        else:
            return f"⚠️ 谨慎观望: 效果不确定 (ATE={ate:.2f})，建议优化策略后测试"


def generate_promotion_data(n_samples: int = 1000,
                            random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    生成促销活动效果模拟数据

    场景：吸奶器季节性促销活动
    """
    np.random.seed(random_state)

    historical_sales = np.random.lognormal(8, 0.5, n_samples)
    inventory_level = np.random.poisson(500, n_samples)
    competitor_price = np.random.normal(120, 20, n_samples)
    new_customer_ratio = np.random.beta(2, 5, n_samples)
    seasonality = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    is_holiday = np.random.binomial(1, 0.15, n_samples)

    X = pd.DataFrame({
        'historical_sales': historical_sales,
        'inventory_level': inventory_level,
        'competitor_price': competitor_price,
        'new_customer_ratio': new_customer_ratio,
        'seasonality': seasonality,
        'is_holiday': is_holiday
    })

    # 干预分配
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
    season_effect = np.where(seasonality == 1, 50, np.where(seasonality == 2, 100, 0))
    promo_effect = (
        30 +
        20 * (new_customer_ratio > 0.3) +
        15 * (seasonality > 0) +
        10 * is_holiday +
        5 * (competitor_price > 130)
    )

    Y0 = base_sales + season_effect + np.random.normal(0, 20, n_samples)
    Y1 = Y0 + promo_effect + np.random.normal(0, 15, n_samples)
    outcome = np.where(treatment == 1, Y1, Y0)

    return X, treatment, outcome


def main():
    """主函数：演示双重稳健促销效果分析"""
    print("=" * 70)
    print("双重稳健促销效果分析")
    print("=" * 70)

    # 生成数据
    print("\n[1] 生成促销数据...")
    X, treatment, outcome = generate_promotion_data(n_samples=1000)
    print(f"   总样本: {len(X)}")
    print(f"   促销组: {treatment.sum()} ({treatment.mean()*100:.1f}%)")

    # 双重稳健估计
    print("\n[2] 双重稳健估计...")
    dr_estimator = DoublyRobustEstimator(n_folds=5)
    dr_estimator.fit(X.values, treatment, outcome)

    ate_result = dr_estimator.get_ate(with_ci=True)
    print(f"   ATE 估计: {ate_result['ate']:.2f}")
    print(f"   95% CI: [{ate_result['ci_lower']:.2f}, {ate_result['ci_upper']:.2f}]")

    # 模型诊断
    print("\n[3] 模型诊断...")
    diagnosis = dr_estimator.diagnose_models(X.values, treatment, outcome)
    print(f"   倾向评分 AUC: {diagnosis['propensity_auc']:.3f}")
    print(f"   结果模型 R²: {diagnosis['outcome_r2_treated']:.3f}")
    print(f"   诊断: {diagnosis['recommendation']}")

    # 决策建议
    print("\n[4] 促销决策建议...")
    analyzer = PromotionEffectAnalyzer(dr_estimator)
    recommendation = analyzer.get_promotion_recommendation(ate_result)
    print(f"   {recommendation}")

    # 异质性分析
    print("\n[5] 按季节性分析...")
    cate = dr_estimator.predict_effect(X.values)
    segment_analysis = analyzer.analyze_by_segment(X, cate, 'seasonality')
    print(segment_analysis.to_string(index=False))

    print("\n" + "=" * 70)
    return dr_estimator, analyzer


if __name__ == '__main__':
    model, analyzer = main()

"""
智能归因 - 因果森林 (Causal Forest)
用于母婴出海电商多市场智能广告归因和促销时机优化

基于论文:
- Multi-Study Causal Forest (MCF): arXiv:2502.02110 (2025)
- Generalized Random Forests: Athey et al., Annals of Statistics (2019)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class CausalForestAttribution:
    """
    因果森林智能归因模型
    自动发现异质性处理效应，支持多市场场景
    """

    def __init__(self, n_trees: int = 1000, min_node_size: int = 5,
                 max_depth: Optional[int] = None, use_grf: bool = False):
        """
        初始化因果森林

        Args:
            n_trees: 树的数量（默认 1000）
            min_node_size: 叶子节点最小样本数（默认 5）
            max_depth: 最大深度（默认无限制）
            use_grf: 是否使用 grf 包（需要安装），否则使用 sklearn 回退
        """
        self.n_trees = n_trees
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.use_grf = use_grf
        self.model = None
        self.model_t = None
        self.model_c = None
        self.feature_names = None

    def fit(self, X: np.ndarray, treatment: np.ndarray,
            outcome: np.ndarray, **kwargs) -> 'CausalForestAttribution':
        """
        训练因果森林

        Args:
            X: 特征矩阵 (n_samples, n_features)
            treatment: 干预标志 (n_samples,) 1=干预组, 0=对照组
            outcome: 结果变量 (n_samples,)
            **kwargs: 传递给底层实现的额外参数

        Returns:
            self
        """
        if self.use_grf:
            return self._fit_grf(X, treatment, outcome, **kwargs)
        else:
            return self._fit_fallback(X, treatment, outcome)

    def _fit_grf(self, X, treatment, outcome, **kwargs):
        """使用 grf 包训练（推荐，但需安装）"""
        try:
            from grf import CausalForest
            self.model = CausalForest(
                n_estimators=self.n_trees,
                min_samples_leaf=self.min_node_size,
                max_depth=self.max_depth,
                **kwargs
            )
            self.model.fit(X, treatment, outcome)
            return self
        except ImportError:
            print("Warning: grf not found. Falling back to sklearn.")
            return self._fit_fallback(X, treatment, outcome)

    def _fit_fallback(self, X, treatment, outcome):
        """
        回退方案：使用 GradientBoosting 近似因果森林 (T-Learner)
        """
        X = np.array(X)
        treatment = np.array(treatment)
        outcome = np.array(outcome)

        X_t = X[treatment == 1]
        X_c = X[treatment == 0]
        y_t = outcome[treatment == 1]
        y_c = outcome[treatment == 0]

        self.model_t = GradientBoostingRegressor(
            n_estimators=self.n_trees // 10,
            max_depth=self.max_depth or 4,
            random_state=42
        )
        self.model_c = GradientBoostingRegressor(
            n_estimators=self.n_trees // 10,
            max_depth=self.max_depth or 4,
            random_state=42
        )

        self.model_t.fit(X_t, y_t)
        self.model_c.fit(X_c, y_c)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测 CATE (条件平均处理效应)

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            cate: 处理效应估计 (n_samples,)
        """
        X = np.array(X)

        if self.model is not None:
            return self.model.predict(X)
        else:
            pred_t = self.model_t.predict(X)
            pred_c = self.model_c.predict(X)
            return pred_t - pred_c

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        获取特征重要性

        Returns:
            DataFrame: 特征重要性排序
        """
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif self.model_t is not None:
            # 使用两个模型重要性的平均
            importances = (self.model_t.feature_importances_ +
                          self.model_c.feature_importances_) / 2
        else:
            return None

        return pd.DataFrame({
            'feature': self.feature_names or range(len(importances)),
            'importance': importances
        }).sort_values('importance', ascending=False)


class MultiMarketAttributionAnalyzer:
    """
    多市场归因分析器
    专门针对母婴出海多市场场景
    """

    def __init__(self, model: CausalForestAttribution):
        self.model = model

    def analyze_by_market(self, X: pd.DataFrame, market: np.ndarray,
                          treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
        """
        分析各市场的用户分群效果

        Args:
            X: 特征 DataFrame
            market: 市场标识数组
            treatment: 干预标志
            outcome: 结果变量

        Returns:
            DataFrame: 各市场分析结果
        """
        cate_pred = self.model.predict(X.values)

        results = []
        for m in np.unique(market):
            mask = market == m
            if mask.sum() == 0:
                continue

            market_cate = cate_pred[mask]
            q75, q25 = np.percentile(market_cate, [75, 25])

            results.append({
                'market': m,
                'avg_cate': market_cate.mean(),
                'cate_std': market_cate.std(),
                'high_uplift_pct': (market_cate > q75).mean() * 100,
                'low_uplift_pct': (market_cate < q25).mean() * 100,
                'conversion_treated': outcome[mask & (treatment == 1)].mean(),
                'conversion_control': outcome[mask & (treatment == 0)].mean()
            })

        return pd.DataFrame(results)

    def get_high_uplift_persona(self, X: pd.DataFrame,
                                 cate: np.ndarray,
                                 percentile: float = 80) -> Dict:
        """
        获取高 uplift 用户画像

        Args:
            X: 特征 DataFrame
            cate: CATE 预测值
            percentile: 高 uplift 分位数阈值

        Returns:
            Dict: 用户画像特征
        """
        high_uplift_mask = cate > np.percentile(cate, percentile)
        high_users = X[high_uplift_mask]

        persona = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                persona[col] = {
                    'mean': high_users[col].mean(),
                    'std': high_users[col].std()
                }
            else:
                persona[col] = high_users[col].mode().iloc[0] if len(high_users) > 0 else None

        return persona


def generate_multimarket_data(n_samples: int = 10000,
                               random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成多市场母婴电商模拟数据

    场景：吸奶器在北美/欧洲多市场投放

    Returns:
        X: 特征 DataFrame
        treatment: 干预标志
        outcome: 结果变量
        market: 市场标识
    """
    np.random.seed(random_state)

    # 市场分布
    market = np.random.choice(
        ['US', 'CA', 'UK', 'DE'],
        n_samples,
        p=[0.45, 0.20, 0.20, 0.15]
    )

    # 用户基础特征
    age = np.where(
        market == 'DE',
        np.random.normal(34, 4, n_samples),
        np.random.normal(31, 5, n_samples)
    )
    age = np.clip(age, 25, 45)

    income_level = np.zeros(n_samples, dtype=int)
    for i, m in enumerate(['US', 'CA', 'UK', 'DE']):
        mask = market == m
        probs = {
            'US': [0.25, 0.35, 0.30, 0.10],
            'CA': [0.20, 0.30, 0.35, 0.15],
            'UK': [0.25, 0.35, 0.30, 0.10],
            'DE': [0.15, 0.25, 0.40, 0.20]
        }
        income_level[mask] = np.random.choice([1, 2, 3, 4], mask.sum(), p=probs[m])

    is_first_time = np.random.binomial(1, 0.6, n_samples)
    browsing_pages = np.random.poisson(5, n_samples) + 1
    time_on_site = np.random.exponential(5, n_samples)
    cart_value = np.random.exponential(80, n_samples) * np.random.binomial(1, 0.35, n_samples)

    X = pd.DataFrame({
        'age': age,
        'income_level': income_level,
        'is_first_time': is_first_time,
        'browsing_pages': browsing_pages,
        'time_on_site': time_on_site,
        'cart_value': cart_value,
        'market_US': (market == 'US').astype(int),
        'market_CA': (market == 'CA').astype(int),
        'market_UK': (market == 'UK').astype(int),
        'market_DE': (market == 'DE').astype(int)
    })

    # 干预和结果生成
    treatment_prob = np.where(market == 'US', 0.5,
                     np.where(market == 'DE', 0.4, 0.45))
    treatment = np.random.binomial(1, treatment_prob)

    # 潜在结果
    base_prob = 0.05
    market_effect = np.where(market == 'US', 0.02,
                    np.where(market == 'DE', 0.03, 0.01))
    income_effect = 0.02 * (income_level - 2.5)
    first_time_effect = 0.04 * is_first_time
    cart_effect = 0.08 * (cart_value > 0)

    Y0_prob = base_prob + market_effect + income_effect + first_time_effect + cart_effect
    Y0_prob = np.clip(Y0_prob, 0.01, 0.4)
    Y0 = np.random.binomial(1, Y0_prob)

    ad_effect = np.where(
        market == 'US',
        0.12 * (income_level == 2) + 0.08 * is_first_time,
        np.where(
            market == 'DE',
            0.10 * (income_level >= 3) + 0.06 * (time_on_site > 5),
            0.09 * is_first_time + 0.05 * (cart_value > 50)
        )
    )

    Y1_prob = Y0_prob + ad_effect
    Y1_prob = np.clip(Y1_prob, 0.01, 0.6)
    Y1 = np.random.binomial(1, Y1_prob)

    outcome = np.where(treatment == 1, Y1, Y0)

    return X, treatment, outcome, market


def main():
    """主函数：演示因果森林在多市场归因中的应用"""
    print("=" * 70)
    print("母婴出海 - 因果森林智能归因")
    print("=" * 70)

    # 1. 生成数据
    print("\n[1] 生成多市场数据...")
    X, treatment, outcome, market = generate_multimarket_data(n_samples=10000)
    print(f"   总样本: {len(X)}")
    for m in ['US', 'CA', 'UK', 'DE']:
        count = (market == m).sum()
        print(f"   - {m}: {count} ({count/len(market)*100:.1f}%)")

    # 2. 划分训练/测试集
    X_train, X_test, t_train, t_test, y_train, y_test, m_train, m_test = train_test_split(
        X, treatment, outcome, market, test_size=0.3, random_state=42
    )
    print(f"\n[2] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 3. 训练模型
    print("\n[3] 训练因果森林模型...")
    model = CausalForestAttribution(n_trees=500)
    model.fit(X_train.values, t_train, y_train)

    # 4. 预测
    print("\n[4] 预测处理效应...")
    cate_pred = model.predict(X_test.values)
    print(f"   平均 CATE: {cate_pred.mean():.4f}")
    print(f"   CATE 标准差: {cate_pred.std():.4f}")

    # 5. 多市场分析
    print("\n[5] 分市场分析...")
    analyzer = MultiMarketAttributionAnalyzer(model)
    market_analysis = analyzer.analyze_by_market(X_test, m_test, t_test, y_test)
    print(market_analysis.to_string(index=False))

    # 6. 高 uplift 用户画像
    print("\n[6] 高 uplift 用户画像 (Top 20%)...")
    persona = analyzer.get_high_uplift_persona(X_test, cate_pred, percentile=80)
    print(f"   平均年龄: {persona['age']['mean']:.1f}")
    print(f"   平均收入等级: {persona['income_level']['mean']:.1f}")
    print(f"   新手妈妈占比: {persona['is_first_time']['mean']*100:.1f}%")

    # 7. 投放建议
    print("\n[7] 智能投放建议...")
    print("   - 美国市场：重点投放中等收入新手妈妈")
    print("   - 德国市场：重点投放高收入+深度浏览用户")
    print("   - 高 uplift 用户优先分配预算")

    print("\n" + "=" * 70)
    return model, analyzer


if __name__ == '__main__':
    model, analyzer = main()

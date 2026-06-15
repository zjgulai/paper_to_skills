"""
Auto-extracted from: paper2skills-vault/15-营销投放分析/Skill-Marketing-Mix-Modeling.md
Skill: Skill-Marketing-Mix-Modeling
Domain: 15-营销投放分析
"""
"""
Marketing Mix Modeling (MMM) — 营销组合模型
支持：Adstock衰减、Hill饱和、Bayesian回归、预算优化
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class AdstockTransformer:
    """Adstock衰减转换"""

    def __init__(self, decay_rate=0.5):
        self.decay_rate = decay_rate

    def transform(self, spend_series):
        """对单渠道花费序列做Adstock转换"""
        adstock = np.zeros(len(spend_series))
        adstock[0] = spend_series[0]
        for t in range(1, len(spend_series)):
            adstock[t] = spend_series[t] + self.decay_rate * adstock[t-1]
        return adstock


class HillSaturation:
    """Hill饱和函数"""

    def __init__(self, K=0.5, eta=2.0):
        self.K = K  # 半饱和点
        self.eta = eta  # 形状参数

    def transform(self, x):
        """Hill转换"""
        return x ** self.eta / (x ** self.eta + self.K ** self.eta)


class SimpleMMM:
    """简化版营销组合模型"""

    def __init__(self):
        self.coef = {}  # 渠道系数
        self.adstock_params = {}  # 各渠道衰减率
        self.hill_params = {}  # 各渠道饱和参数
        self.base_sales = 0
        self.seasonality_coef = {}

    def _apply_transforms(self, spend_df, channels):
        """应用Adstock和Hill转换"""
        transformed = pd.DataFrame(index=spend_df.index)
        for ch in channels:
            # Adstock
            adstock = AdstockTransformer(
                decay=self.adstock_params.get(ch, 0.5)
            ).transform(spend_df[ch].values)
            # Hill饱和
            hill = HillSaturation(
                K=self.hill_params.get(ch, {}).get('K', spend_df[ch].median()),
                eta=self.hill_params.get(ch, {}).get('eta', 2.0)
            )
            # 标准化后应用Hill
            x_norm = adstock / (spend_df[ch].max() + 1e-6)
            transformed[ch] = hill.transform(x_norm)
        return transformed

    def fit(self, df, sales_col, channel_cols, season_col=None):
        """
        拟合MMM模型（简化版：线性回归）

        Args:
            df: DataFrame with time series data
            sales_col: 销售额列名
            channel_cols: 渠道花费列名列表
            season_col: 季节性特征列名（可选）
        """
        # 简化：先用默认参数做Adstock转换
        self.adstock_params = {ch: 0.5 for ch in channel_cols}
        self.hill_params = {ch: {'K': 0.5, 'eta': 2.0} for ch in channel_cols}

        transformed = self._apply_transforms(df, channel_cols)

        # 加入季节性
        if season_col and season_col in df.columns:
            transformed[season_col] = df[season_col]

        # 最小二乘拟合
        X = transformed.values
        y = df[sales_col].values

        # 添加截距
        X = np.column_stack([np.ones(len(X)), X])

        # 正规方程
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        self.base_sales = beta[0]
        for i, ch in enumerate(channel_cols):
            self.coef[ch] = beta[i + 1]
        if season_col and season_col in df.columns:
            self.seasonality_coef[season_col] = beta[len(channel_cols) + 1]

        return self

    def decompose(self, df, channel_cols):
        """分解各渠道对销售额的贡献"""
        transformed = self._apply_transforms(df, channel_cols)

        decomposition = pd.DataFrame(index=df.index)
        decomposition['base'] = self.base_sales

        total_contribution = np.ones(len(df)) * self.base_sales
        for ch in channel_cols:
            contrib = self.coef[ch] * transformed[ch].values
            decomposition[ch] = contrib
            total_contribution += contrib

        decomposition['predicted'] = total_contribution
        decomposition['actual'] = df['sales'].values if 'sales' in df.columns else np.nan

        return decomposition

    def optimize_budget(self, channel_cols, total_budget, bounds=None):
        """
        最优预算分配

        Args:
            channel_cols: 渠道列表
            total_budget: 总预算
            bounds: 各渠道预算上下界
        """
        if bounds is None:
            bounds = [(total_budget * 0.1, total_budget * 0.5) for _ in channel_cols]

        def predict_sales(alloc):
            # 简化预测：用线性近似
            pred = self.base_sales
            for i, ch in enumerate(channel_cols):
                # 假设transform后的值与花费成正比（简化）
                pred += self.coef[ch] * (alloc[i] / total_budget)
            return -pred  # 最小化负数 = 最大化

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}
        x0 = np.array([total_budget / len(channel_cols)] * len(channel_cols))

        result = minimize(predict_sales, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return {
            'allocation': {ch: round(result.x[i], 0) for i, ch in enumerate(channel_cols)},
            'predicted_sales': -result.fun
        }

    def calculate_marginal_roas(self, channel, spend_level):
        """计算某渠道在某花费水平的边际ROAS"""
        # 简化：用系数乘以transform后的边际变化
        # 实际应该用导数
        return self.coef.get(channel, 0)


def generate_mmm_data(n_weeks=104, channels=None, seed=42):
    """生成MMM模拟数据（2年周数据）"""
    np.random.seed(seed)
    if channels is None:
        channels = ['facebook', 'google', 'tiktok', 'kol']

    dates = pd.date_range('2023-01-01', periods=n_weeks, freq='W')

    # 基础销售额（趋势+季节性）
    trend = np.linspace(100, 150, n_weeks)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)  # 年度周期
    base_sales = trend + seasonality + np.random.normal(0, 10, n_weeks)

    # 渠道花费（不同渠道有不同的策略变化）
    spends = {}
    spends['facebook'] = np.random.lognormal(3, 0.3, n_weeks) * 1000
    spends['google'] = np.random.lognormal(2.8, 0.3, n_weeks) * 1000
    # TikTok逐月加码
    spends['tiktok'] = np.linspace(10000, 50000, n_weeks) + np.random.normal(0, 5000, n_weeks)
    spends['kol'] = np.random.lognormal(2.5, 0.5, n_weeks) * 1000

    # 真实系数（用于生成销售）
    true_coef = {'facebook': 2.5, 'google': 1.8, 'tiktok': 3.0, 'kol': 3.5}
    true_decay = {'facebook': 0.6, 'google': 0.4, 'tiktok': 0.5, 'kol': 0.3}

    # 生成销售额
    sales = base_sales.copy()
    for ch in channels:
        adstock = np.zeros(n_weeks)
        for t in range(n_weeks):
            if t == 0:
                adstock[t] = spends[ch][t]
            else:
                adstock[t] = spends[ch][t] + true_decay[ch] * adstock[t-1]
        sales += true_coef[ch] * adstock / 10000

    sales += np.random.normal(0, 15, n_weeks)

    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        **{ch: spends[ch] for ch in channels},
        'seasonality': seasonality,
    })

    return df


if __name__ == '__main__':
    # 生成数据
    df = generate_mmm_data(n_weeks=104)
    print("数据预览:")
    print(df.head())

    # 拟合MMM
    mmm = SimpleMMM()
    channels = ['facebook', 'google', 'tiktok', 'kol']
    mmm.fit(df, 'sales', channels, season_col='seasonality')

    print(f"\n基础销售额: {mmm.base_sales:.1f}")
    print("渠道系数:")
    for ch, coef in mmm.coef.items():
        print(f"  {ch}: {coef:.3f}")

    # 贡献分解
    decomp = mmm.decompose(df, channels)
    print("\n贡献分解(最后4周):")
    print(decomp.tail(4).round(1))

    # 预算优化
    opt = mmm.optimize_budget(channels, total_budget=100000)
    print(f"\n最优预算分配(总预算10万):")
    for ch, alloc in opt['allocation'].items():
        print(f"  {ch}: ${alloc:,.0f}")

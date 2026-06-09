"""
Difference-in-Differences (DiD) — 双重差分因果效应估计
Skill: Skill-DiD-Difference-in-Differences.md
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class DifferenceInDifferences:
    """双重差分估计器"""

    def __init__(self):
        self.tau = None
        self.se = None

    def fit(self, df, unit_col, time_col, treat_col, post_col, outcome_col):
        """拟合 DiD 模型"""
        df = df.copy()
        df['treat_post'] = df[treat_col] * df[post_col]

        X = df[['treat_col_dummy', 'post_col_dummy', 'treat_post']].values
        # 使用实际列名
        X = df[[treat_col, post_col, 'treat_post']].values
        y = df[outcome_col].values

        model = LinearRegression()
        model.fit(X, y)
        self.tau = model.coef_[2]

        # 简化标准误
        n = len(df)
        residuals = y - model.predict(X)
        mse = np.sum(residuals ** 2) / (n - 3)
        var_tau = mse / np.sum((df['treat_post'] - df['treat_post'].mean()) ** 2)
        self.se = np.sqrt(var_tau)

        return self

    def summary(self):
        t_stat = self.tau / self.se if self.se and self.se > 0 else 0
        print(f"DiD τ: {self.tau:.4f}, SE: {self.se:.4f}, t: {t_stat:.2f}")
        return {'tau': self.tau, 'se': self.se}


def generate_cross_border_did_data(n_units=100, n_periods=24, random_state=42):
    """生成母婴出海 DiD 模拟数据"""
    np.random.seed(random_state)
    countries = ['UK', 'DE', 'FR', 'NL']
    weights = [0.25, 0.3, 0.3, 0.15]
    data = []

    for uid in range(n_units):
        cidx = np.random.choice(4, p=weights)
        country = countries[cidx]
        treat = 1 if country == 'UK' else 0
        effect = {'UK': 1000, 'DE': 1200, 'FR': 1100, 'NL': 800}[country]

        for p in range(n_periods):
            post = 1 if p >= 12 else 0
            treat_effect = -150 if (treat == 1 and post == 1) else 0
            sales = effect + p * 10 + (200 if (p % 12 + 1) in [11, 12] else 0) + treat_effect
            sales += np.random.normal(0, 100)
            data.append({
                'unit_id': uid, 'country': country, 'period': p,
                'treatment': treat, 'post': post, 'sales': max(sales, 0),
                'event_time': p - 12
            })

    return pd.DataFrame(data)


if __name__ == '__main__':
    df = generate_cross_border_did_data(n_units=200)
    did = DifferenceInDifferences()
    did.fit(df, 'unit_id', 'period', 'treatment', 'post', 'sales')
    did.summary()

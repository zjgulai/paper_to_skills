"""
Instrumental Variables (IV) — 工具变量法
Skill: Skill-IV-Instrumental-Variables.md
"""

import numpy as np
from sklearn.linear_model import LinearRegression


class InstrumentalVariables:
    """2SLS 工具变量估计器"""

    def __init__(self):
        self.iv_coef = None
        self.first_stage_f = None

    def fit(self, X_endog, Z_iv, Y_outcome, X_controls=None):
        """拟合 2SLS"""
        T = np.array(X_endog).ravel()
        Z = np.array(Z_iv).reshape(len(T), -1) if Z_iv.ndim == 1 else np.array(Z_iv)
        Y = np.array(Y_outcome)

        # 第一阶段: Z -> T
        X_a = np.column_stack([Z, X_controls]) if X_controls is not None else Z
        m1 = LinearRegression().fit(X_a, T)
        T_pred = m1.predict(X_a)
        self.first_stage_f = m1.score(X_a, T) * 100  # 简化 F 统计量

        # 第二阶段: T_hat -> Y
        X_b = np.column_stack([T_pred, X_controls]) if X_controls is not None else T_pred.reshape(-1, 1)
        m2 = LinearRegression().fit(X_b, Y)
        self.iv_coef = m2.coef_[0]
        return self

    def summary(self):
        print(f"IV Coef: {self.iv_coef:.4f}, First-stage F: {self.first_stage_f:.2f}")
        return {'iv_coef': self.iv_coef, 'first_stage_f': self.first_stage_f}


def generate_pricing_iv_data(n=5000, seed=42):
    """生成价格弹性 IV 模拟数据"""
    np.random.seed(seed)
    comp_price = np.random.normal(100, 15, n)
    exchange = np.random.normal(1.0, 0.1, n)
    quality = np.random.normal(0, 1, n)

    price = 50 + 0.3 * comp_price + 30 * exchange + 10 * quality + np.random.normal(0, 5, n)
    sales = 1000 - 1.5 * (price - price.mean()) * 5 + 20 * quality + np.random.normal(0, 50, n)

    return {
        'price': price, 'sales': np.maximum(sales, 0),
        'competitor_price': comp_price, 'exchange_rate': exchange
    }


if __name__ == '__main__':
    d = generate_pricing_iv_data()
    iv = InstrumentalVariables()
    Z = np.column_stack([d['competitor_price'], d['exchange_rate']])
    iv.fit(d['price'], Z, d['sales'])
    iv.summary()
    p_mean, q_mean = d['price'].mean(), d['sales'].mean()
    print(f"Price Elasticity (IV): {iv.iv_coef * (p_mean / q_mean):.2f}")

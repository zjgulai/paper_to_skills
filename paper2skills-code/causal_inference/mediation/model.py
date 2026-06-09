"""
Causal Mediation Analysis — 因果中介效应分析
Skill: Skill-Mediation-Causal-Mechanism-Analysis.md
"""

import numpy as np
from sklearn.linear_model import LinearRegression


class MediationAnalysis:
    """中介分析器"""

    def __init__(self):
        self.results = {}

    def fit(self, T, M, Y):
        """拟合中介模型 (T -> M -> Y)"""
        T, M, Y = np.array(T).ravel(), np.array(M).ravel(), np.array(Y)

        # a path: T -> M
        a = LinearRegression().fit(T.reshape(-1, 1), M).coef_[0]
        # b & c': T + M -> Y
        m2 = LinearRegression().fit(np.column_stack([T, M]), Y)
        c_prime, b = m2.coef_[0], m2.coef_[1]
        # total: T -> Y
        c = LinearRegression().fit(T.reshape(-1, 1), Y).coef_[0]

        self.results = {
            'total_effect': c,
            'direct_effect': c_prime,
            'indirect_effect': a * b,
            'a_path': a, 'b_path': b, 'c_prime': c_prime
        }
        return self

    def summary(self):
        r = self.results
        print(f"Total: {r['total_effect']:.4f}, Direct: {r['direct_effect']:.4f}, Indirect: {r['indirect_effect']:.4f}")
        return r


if __name__ == '__main__':
    np.random.seed(42)
    T = np.random.binomial(1, 0.5, 1000)
    M = 0.05 + 0.03 * T + np.random.normal(0, 0.01, 1000)
    Y_prob = 0.02 + 0.5 * M + 0.005 * T + np.random.normal(0, 0.005, 1000)
    Y = (Y_prob > np.percentile(Y_prob, 85)).astype(int)

    ma = MediationAnalysis()
    ma.fit(T, M, Y)
    ma.summary()

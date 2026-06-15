"""
Auto-extracted from: paper2skills-vault/13-广告分析/Skill-ROAS-Budget-Optimization.md
Skill: Skill-ROAS-Budget-Optimization
Domain: 13-广告分析
"""
"""
ROAS Optimization and Budget Allocation — ROAS优化与预算分配
支持：花费-收入曲线拟合、边际ROAS计算、最优分配
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ROASOptimizer:
    """ROAS优化器"""

    def __init__(self):
        self.curve_params = {}

    def fit_spend_revenue_curve(self, channel, spends, revenues):
        """
        拟合花费-收入曲线: Revenue = a * Spend^b

        Args:
            channel: 渠道名
            spends: 历史花费数组
            revenues: 历史收入数组
        """
        spends = np.array(spends)
        revenues = np.array(revenues)

        # 对数线性回归: log(Revenue) = log(a) + b * log(Spend)
        log_spend = np.log(spends + 1)
        log_revenue = np.log(revenues + 1)

        # 简单线性回归
        n = len(spends)
        b = np.sum((log_spend - log_spend.mean()) * (log_revenue - log_revenue.mean())) / \
            np.sum((log_spend - log_spend.mean()) ** 2)
        log_a = log_revenue.mean() - b * log_spend.mean()
        a = np.exp(log_a)

        self.curve_params[channel] = {'a': a, 'b': b}
        return a, b

    def predict_revenue(self, channel, spend):
        """预测给定花费下的收入"""
        if channel not in self.curve_params:
            return 0
        params = self.curve_params[channel]
        return params['a'] * (spend ** params['b'])

    def marginal_roas(self, channel, spend):
        """计算边际ROAS"""
        if channel not in self.curve_params:
            return 0
        params = self.curve_params[channel]
        a, b = params['a'], params['b']
        # d(Revenue)/d(Spend) = a * b * Spend^(b-1)
        return a * b * (spend ** (b - 1))

    def optimize_budget(self, channels, total_budget, min_budget_per_channel=5000):
        """
        最优预算分配

        Args:
            channels: 渠道列表
            total_budget: 总预算
            min_budget_per_channel: 每个渠道最小预算
        """
        def objective(spends):
            # 最大化总收入 = 最小化负收入
            total_revenue = 0
            for i, ch in enumerate(channels):
                total_revenue += self.predict_revenue(ch, spends[i])
            return -total_revenue

        # 约束：总预算 = total_budget
        constraints = {'type': 'eq', 'fun': lambda s: np.sum(s) - total_budget}

        # 边界：每个渠道最少min_budget
        bounds = [(min_budget_per_channel, total_budget * 0.6) for _ in channels]

        # 初始值：平均分配
        x0 = np.array([total_budget / len(channels)] * len(channels))

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        allocation = {ch: round(result.x[i], 0) for i, ch in enumerate(channels)}
        total_revenue = sum(self.predict_revenue(ch, result.x[i]) for i, ch in enumerate(channels))

        return {
            'allocation': allocation,
            'total_revenue': total_revenue,
            'overall_roas': total_revenue / total_budget,
            'marginal_roas': {ch: self.marginal_roas(ch, result.x[i]) for i, ch in enumerate(channels)}
        }

    def compare_scenarios(self, channels, allocations_dict):
        """对比多种预算分配方案"""
        results = []
        for name, alloc in allocations_dict.items():
            total_rev = sum(self.predict_revenue(ch, alloc[ch]) for ch in channels)
            total_spend = sum(alloc.values())
            results.append({
                'scenario': name,
                'total_revenue': total_rev,
                'total_spend': total_spend,
                'roas': total_rev / total_spend,
                'allocation': alloc
            })
        return pd.DataFrame(results)


# 示例
if __name__ == '__main__':
    optimizer = ROASOptimizer()

    # 拟合三个渠道的花费-收入曲线
    channels_data = {
        'Facebook': {'spends': [5000, 10000, 20000, 30000, 50000], 'revenues': [18000, 32000, 55000, 75000, 110000]},
        'Google': {'spends': [3000, 8000, 15000, 25000, 40000], 'revenues': [12000, 28000, 52000, 80000, 120000]},
        'TikTok': {'spends': [2000, 5000, 10000, 20000], 'revenues': [4000, 10000, 22000, 45000]},
    }

    for ch, data in channels_data.items():
        a, b = optimizer.fit_spend_revenue_curve(ch, data['spends'], data['revenues'])
        print(f"{ch}: Revenue = {a:.2f} * Spend^{b:.3f}")

    # 最优分配
    result = optimizer.optimize_budget(['Facebook', 'Google', 'TikTok'], total_budget=100000)
    print(f"\n最优分配:")
    for ch, alloc in result['allocation'].items():
        print(f"  {ch}: ${alloc:,.0f}")
    print(f"  预期总收入: ${result['total_revenue']:,.0f}")
    print(f"  整体ROAS: {result['overall_roas']:.2f}")

    # 对比方案
    print(f"\n方案对比:")
    scenarios = {
        'Current': {'Facebook': 50000, 'Google': 30000, 'TikTok': 20000},
        'Equal': {'Facebook': 33333, 'Google': 33333, 'TikTok': 33334},
        'Optimal': result['allocation'],
    }
    comparison = optimizer.compare_scenarios(['Facebook', 'Google', 'TikTok'], scenarios)
    print(comparison[['scenario', 'total_revenue', 'roas']].to_string(index=False))

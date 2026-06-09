---
title: ROAS Optimization and Ad Budget Allocation
module: 13-广告分析
topic: roas-optimization
status: stable
created: 2026-05-15
updated: 2026-05-15
roadmap_phase: phase1
---

# Skill Card: ROAS Optimization & Budget Allocation

## ① 算法原理

**核心问题**：广告预算有限，如何在不同渠道（Facebook/Google/TikTok）、不同 campaign、不同受众之间分配，使总ROAS（广告支出回报率）最大化？

**ROAS = 广告带来的收入 / 广告花费**

**边际收益递减**：
- 每个渠道都存在"甜蜜点"——预算过少时，算法没足够数据优化；预算过多时， Audience 耗尽，成本飙升
- Facebook的第1万刀可能ROAS=4，第10万刀可能ROAS=1.5
- 最优分配：让每个渠道的**边际ROAS相等**

**预算分配策略**：

| 策略 | 逻辑 | 适用场景 |
|------|------|---------|
| **Equal ROAS** | 各渠道ROAS目标相同 | 成熟稳定期 |
| **Marginal ROAS Equalization** | 各渠道边际ROAS相等时总收益最大 | 有充足数据时 |
| **Portfolio Optimization** | 用均值-方差优化，平衡收益和风险 | 多渠道大规模投放 |
| **Thompson Sampling** | 多臂老虎机，动态探索-利用 | 新渠道测试期 |

**边际ROAS计算**：

对历史数据拟合花费-收入曲线（通常是凹函数）：
$$Revenue = a \cdot Spend^b, \quad 0 < b < 1$$

边际ROAS = d(Revenue)/d(Spend) = $a \cdot b \cdot Spend^{b-1}$

当所有渠道的边际ROAS相等时，总预算分配最优。

**反直觉洞察**：
- 不应该"把所有预算给ROAS最高的渠道"——边际递减会让它迅速变差
- 新渠道的"测试预算"不是浪费——是购买信息的成本，信息价值 > 短期ROAS损失
- 日预算是算法的枷锁——Facebook的算法在3-7天学习期内表现不稳定，日预算太小会导致频繁进入学习期

---

## ② 母婴出海应用案例

### 场景1：三渠道预算重新分配

**业务问题**：Momcozy 月预算50万，当前分配：Facebook 30万（ROAS 2.8）、Google 15万（ROAS 3.5）、TikTok 5万（ROAS 1.8）。团队想把TikTok预算砍了加到Google。

**边际ROAS分析**：

| 渠道 | 当前花费 | 当前ROAS | 边际ROAS | 建议动作 |
|------|---------|---------|---------|---------|
| Facebook | 30万 | 2.8 | 1.5 | 维持 |
| Google | 15万 | 3.5 | 2.0 | 增加预算 |
| TikTok | 5万 | 1.8 | 2.5 | **增加预算** |

**决策反转**：TikTok当前ROAS最低，但边际ROAS最高——说明它还在上升期，加大投入效率最高。Google虽然平均ROAS高，但边际ROAS已经下降。

**新分配**：
- Facebook: 28万（-2万）
- Google: 18万（+3万）
- TikTok: 10万（+5万）

### 场景2：Campaign层级的动态预算调整

**业务问题**：双11期间，有5个Facebook campaign在跑，如何根据实时ROAS动态调整预算？

**Thompson Sampling策略**：
1. 每个campaign是一个"臂"
2. 每小时更新ROAS后验分布
3. 按概率采样选择"可能最好"的campaign加预算
4. 同时保留20%预算给表现一般的campaign（探索）

---

## ③ 代码模板

```python
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
```

---


## ④ 技能关联

### 前置技能
- [Skill-Ad-Attribution-Modeling](../13-广告分析/[[Skill-Ad-Attribution-Modeling]].md) — 归因结果是预算优化的输入

### 延伸技能
- [Skill-Promotion-Effectiveness](../15-营销投放分析/[[Skill-Promotion-Effectiveness]].md) — 预算优化后看促销因果增量

### 可组合
- [Skill-Marketing-Mix-Modeling](../15-营销投放分析/[[Skill-Marketing-Mix-Modeling]].md) — MMM 弹性曲线为优化提供约束


- **可组合（延伸）**：[[Skill-Audience-Knowledge-Graph]] / [[Skill-PVM-Attribution-Window-Harmonization]] / [[Skill-Negative-Keyword-Safe-Guard]] / [[Skill-Negative-Keyword-Safe-Guard]]

## ⑤ 商业价值评估

- **ROI**：预算重新分配后整体ROAS提升20-30%，年增收50万+
- **难度**：⭐⭐⭐☆☆（3/5）— 曲线拟合简单，但边际ROAS概念需要理解
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 直接决定广告预算的ROI

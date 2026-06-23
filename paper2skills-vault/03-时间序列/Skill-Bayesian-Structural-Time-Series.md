---
title: Bayesian Structural Time Series — 贝叶斯结构时间序列分离促销/季节/趋势效应
doc_type: knowledge
module: 03-时间序列
topic: bayesian-structural-time-series
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Bayesian-Structural-Time-Series

## ① 算法原理（≤300字）

**核心问题**：运营人员需要知道「这次销量增长，到底是广告投放有效、季节效应、还是自然趋势？」三种效应混杂在一起，传统方法无法分离。贝叶斯结构时间序列（BSTS）通过显式建模每个组分，给出可解释的分解结果。

**状态空间模型**：
$$y_t = \mu_t + \tau_t + \beta x_t + \epsilon_t$$

- $\mu_t$：局部线性趋势（随机游走 + 斜率漂移）
  $$\mu_{t+1} = \mu_t + \delta_t + \eta_t,\quad \delta_{t+1} = \delta_t + \xi_t$$
- $\tau_t$：季节成分（动态系数，和为零约束）
- $\beta x_t$：回归成分（广告花费、促销标记等外部变量，Spike-and-Slab 先验做变量选择）
- $\epsilon_t$：观测噪声

**贝叶斯推断**：用 MCMC（HMC/NUTS）或卡尔曼滤波+EM 算法估计后验分布。输出各组分的 95% 可信区间，而非点估计。

**因果推断应用**：Google 的 CausalImpact 包基于 BSTS 构建干预分析——在干预前训练模型，对比干预后实际值 vs 反事实预测，量化促销效果的增量。

**优势**：不需要对照组，自动处理缺失值，输出可解释的各成分置信区间。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某母婴卖家在某周投放了 Influencer Campaign（5 万美元），销量随即上涨 40%。但不确定是 Campaign 效果还是恰好赶上了季节旺季。用 CausalImpact/BSTS 做反事实分析。

**数据要求**：干预前 52 周周度销量，广告花费、BSR、节假日标记（协变量），干预后 8 周观测值。

**BSTS 输出**：干预期间实际值 vs 反事实预测（95% CI）。识别 Campaign 带来的增量销量约 +18%，季节效应贡献 +15%，实际 Campaign iROAS 约 2.3（远低于表面的 4.5）。

**量化产出**：修正 iROAS 估计，避免后续 Campaign 预算过度投入，年化节省 **20-40 万元** 无效 Influencer 投放。

## ③ 代码模板

```python
import numpy as np
from scipy.linalg import solve

def bsts_local_level(
    y: np.ndarray,
    sigma_eps: float = 5.0,
    sigma_eta: float = 2.0,
    n_forecast: int = 8
) -> dict:
    """
    简化版局部水平 BSTS（卡尔曼滤波实现）
    y: 观测序列
    sigma_eps: 观测噪声标准差
    sigma_eta: 状态噪声标准差（趋势随机游走）
    n_forecast: 预测步数
    """
    n = len(y)
    # 卡尔曼滤波
    mu = np.zeros(n + n_forecast)
    P = np.zeros(n + n_forecast)
    mu[0] = y[0]
    P[0] = sigma_eta ** 2

    # 滤波阶段
    for t in range(1, n):
        # 预测
        mu_pred = mu[t - 1]
        P_pred = P[t - 1] + sigma_eta ** 2
        # 更新
        K = P_pred / (P_pred + sigma_eps ** 2)  # 卡尔曼增益
        mu[t] = mu_pred + K * (y[t] - mu_pred)
        P[t] = (1 - K) * P_pred

    # 预测阶段（外推）
    for t in range(n, n + n_forecast):
        mu[t] = mu[t - 1]
        P[t] = P[t - 1] + sigma_eta ** 2

    forecasts = mu[n:]
    forecast_ci = 1.96 * np.sqrt(P[n:] + sigma_eps ** 2)

    return {
        'filtered_states': mu[:n],
        'forecasts': forecasts,
        'forecast_lower': forecasts - forecast_ci,
        'forecast_upper': forecasts + forecast_ci,
        'state_variance': P
    }

def causal_impact_estimate(
    y_pre: np.ndarray,
    y_post: np.ndarray,
    sigma_eps: float = 5.0,
    sigma_eta: float = 2.0
) -> dict:
    """简化版因果影响估计"""
    result = bsts_local_level(y_pre, sigma_eps, sigma_eta, n_forecast=len(y_post))
    counterfactual = result['forecasts']
    impact = y_post - counterfactual
    cumulative_impact = np.cumsum(impact)

    return {
        'actual': y_post,
        'counterfactual': counterfactual,
        'impact': impact,
        'cumulative_impact': cumulative_impact,
        'avg_impact': np.mean(impact),
        'relative_impact': np.mean(impact) / np.mean(counterfactual)
    }

# 测试
np.random.seed(42)
y_pre = 100 + np.cumsum(np.random.randn(52) * 3)  # 52 周历史
y_post = y_pre[-8:] + 18 + np.random.randn(8) * 5  # 干预后+18 增量

result = causal_impact_estimate(y_pre, y_post)
assert result['avg_impact'] > 0, "应检测到正向因果效应"
assert 5 < result['avg_impact'] < 40, f"效应幅值合理, 实际: {result['avg_impact']:.1f}"

print(f"平均干预效应: +{result['avg_impact']:.1f} 件/周")
print(f"相对效应: +{result['relative_impact']:.1%}")
print(f"累积增量: {result['cumulative_impact'][-1]:.0f} 件")
print("[✓] Bayesian-Structural-Time-Series 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Prophet-Forecasting]]
- 前置技能：[[Skill-STL-Seasonal-Decomposition]]
- 延伸技能：[[Skill-Conformal-Time-Series-Forecasting]]
- 延伸技能：[[Skill-Identified-Bayesian-MMM]]
- 可组合：[[Skill-Promotion-Demand-Decomposition]]
- 可组合：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值评估

- **ROI量化**: 修正 iROAS 估计，年化节省 20-40 万元无效投放
- **实施难度**: ⭐⭐⭐（需要 52 周历史数据和协变量，推荐 CausalImpact 包）
- **优先级**: ⭐⭐⭐⭐（Influencer/促销效果量化的标准工具）

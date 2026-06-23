---
title: Price Elasticity Time Series Fusion — 价格弹性×时间序列融合预测
doc_type: knowledge
module: 03-时间序列
topic: price-elasticity-time-series-fusion
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Price-Elasticity-Time-Series-Fusion

## ① 算法原理（≤300字）

**核心问题**：纯时序预测假设价格不变，但母婴卖家频繁调价（大促降价 20-40%）。如果预测模型不感知价格变化，就无法预测「明天降价 15%，销量会变多少」，导致备货、广告预算决策失准。

**融合方法**：将需求弹性估计嵌入时序预测框架：
$$\hat{y}_t = \hat{y}_t^{TS} \cdot \left(\frac{p_t}{p_0}\right)^{\hat{\epsilon}}$$

- $\hat{y}_t^{TS}$：时序基础预测（不含价格变化）
- $p_t / p_0$：相对价格比（当前价 vs 基准价）
- $\hat{\epsilon}$：需求价格弹性（负值，通常 -1.5 ~ -3.0）

**弹性估计**：使用对数线性回归：
$$\ln(y_t) = \alpha + \hat{\epsilon} \ln(p_t) + \beta X_t + \epsilon_t$$

其中 $X_t$ 包含时序特征（趋势、季节、节假日虚拟变量），控制混淆因素。

**时变弹性**：弹性本身可能随时间变化（旺季弹性更高），用滚动窗口（26 周）估计动态弹性 $\hat{\epsilon}_t$，结合卡尔曼滤波做平滑。

**关键注意**：大促降价的弹性 ≠ 常规调价的弹性，需要区分「计划内大促」和「价格维护」场景，避免弹性估计被大促数据污染。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某卖家吸奶器 A 款在黑五计划降价 25%，需要预测降价后的销量以确定备货量和 FBA 补仓时机。历史显示该品类弹性约 -2.1。

**数据要求**：52 周价格 + 销量记录，促销标记，BSR 历史。

**融合预测**：时序基础预测 = 200 件/周，价格调整因子 = (75/100)^(-2.1) ≈ 1.65，预测黑五销量 = 200 × 1.65 = 330 件/周。比之前简单拍脑袋 250 件更有根据。

**量化产出**：黑五备货准确率提升 40%，缺货率从 18% 降至 6%，年化减少缺货损失 **25 万元**。

## ③ 代码模板

```python
import numpy as np

def estimate_price_elasticity(
    prices: np.ndarray,
    quantities: np.ndarray,
    controls: np.ndarray = None
) -> dict:
    """
    价格弹性估计（对数线性回归）
    prices: 价格序列
    quantities: 销量序列
    controls: 控制变量矩阵（趋势、季节等）
    """
    log_p = np.log(prices + 1e-8)
    log_q = np.log(quantities + 1e-8)

    if controls is not None:
        X = np.column_stack([np.ones(len(log_p)), log_p, controls])
    else:
        X = np.column_stack([np.ones(len(log_p)), log_p])

    # OLS 估计
    beta = np.linalg.lstsq(X, log_q, rcond=None)[0]
    elasticity = beta[1]  # ln(p) 的系数即弹性

    # 计算 R²
    y_pred = X @ beta
    ss_res = np.sum((log_q - y_pred) ** 2)
    ss_tot = np.sum((log_q - np.mean(log_q)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {'elasticity': elasticity, 'r2': r2, 'beta': beta}

def price_elasticity_ts_forecast(
    base_forecast: float,
    base_price: float,
    new_price: float,
    elasticity: float
) -> dict:
    """
    价格弹性×时序融合预测
    """
    price_ratio = new_price / base_price
    adjustment = price_ratio ** elasticity
    adjusted_forecast = base_forecast * adjustment

    return {
        'base_forecast': base_forecast,
        'price_ratio': price_ratio,
        'elasticity': elasticity,
        'adjustment_factor': adjustment,
        'adjusted_forecast': adjusted_forecast,
        'demand_change_pct': (adjustment - 1) * 100
    }

# 测试
np.random.seed(42)
n = 52
# 生成价格弹性为 -2 的模拟数据
prices = np.random.uniform(90, 120, n)
true_elasticity = -2.0
log_q = -2.0 * np.log(prices) + np.log(1e6) + np.random.randn(n) * 0.1
quantities = np.exp(log_q)

# 估计弹性
est = estimate_price_elasticity(prices, quantities)
assert -3.0 < est['elasticity'] < -1.0, f"弹性应为负数: {est['elasticity']:.2f}"
assert est['r2'] > 0.7, f"R² 应 > 0.7: {est['r2']:.2f}"

# 融合预测：降价 25% 的影响
forecast = price_elasticity_ts_forecast(
    base_forecast=200,
    base_price=100,
    new_price=75,  # 降价 25%
    elasticity=est['elasticity']
)
assert forecast['adjusted_forecast'] > 200, "降价应带来销量增加"
print(f"估计弹性: {est['elasticity']:.2f}（真实值: {true_elasticity}）, R²: {est['r2']:.2f}")
print(f"降价 25% 后预测销量: {forecast['adjusted_forecast']:.0f} 件（基础: 200）")
print(f"需求变化: +{forecast['demand_change_pct']:.1f}%")
print("[✓] Price-Elasticity-Time-Series-Fusion 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Price-Elasticity-Estimation]]
- 前置技能：[[Skill-Dynamic-Pricing-Elasticity]]
- 延伸技能：[[Skill-Causal-RL-Dynamic-Pricing]]
- 延伸技能：[[Skill-Contextual-Dynamic-Pricing-Optimal]]
- 可组合：[[Skill-Demand-Forecasting-Supply-Chain]]
- 可组合：[[Skill-Promotion-Effectiveness]]

## ⑤ 商业价值评估

- **ROI量化**: 大促备货准确率提升 40%，年化减少缺货损失 25 万元
- **实施难度**: ⭐⭐（数据要求：价格+销量历史，统计知识中等）
- **优先级**: ⭐⭐⭐⭐（频繁调价的卖家必备预测修正工具）

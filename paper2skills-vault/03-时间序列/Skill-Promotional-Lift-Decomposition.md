---
title: Promotional Lift Decomposition — 促销提升分解剥离促销带来的虚假需求
doc_type: knowledge
module: 03-时间序列
topic: promotional-lift-demand-decomposition
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Promotional-Lift-Decomposition

## ① 算法原理（≤300字）

**核心问题**：促销期间销量暴涨，但其中一部分是「购买时间前移」（提前购买）或「购买量超用」（囤货），这些并不是真实需求增长——促销结束后往往出现「需求洼地」（post-promotion dip）。如果把促销销量当真实需求用于预测，会严重高估基础需求，导致补货量虚高。

**提升分解模型**：

$$\text{Promo Sales}_t = \text{Base Demand}_t + \text{Real Lift}_t + \text{Forward Buy}_t$$

- **Base Demand**：无促销情况下的自然销量（反事实）
- **Real Lift**：促销真正带来的增量消费（新客/复购加速）
- **Forward Buy**：提前购买量（促销后将出现洼地）

**识别 Forward Buy**：促销后洼地深度 $D = \sum_{k=1}^{K}(B_t - y_{t+k})$，洼地越深说明 Forward Buy 越大。

**回归分解**：
$$\ln(y_t) = \alpha + \beta_1 \text{IsPromo}_t + \beta_2 \text{PostPromo}_{t,1-3wk} + \gamma X_t + \epsilon_t$$
$\beta_1$ = 促销提升系数，$\beta_2$ = 促销后洼地系数（通常为负），$\gamma$ = 季节/趋势控制变量。

真实提升 = $\beta_1 - |\beta_2|$（净增量）

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某卖家 Prime Day 期间销量达平时 6 倍，采购团队以此为基础上调 Q3 全季度备货量 40%。实际上 Prime Day 后 2 周需求骤降到平时 50%（Forward Buy 效应），导致 Q3 末大量滞销库存。

**数据要求**：18 个月周度销量，促销标记（含时间窗口），历史 5 次大促前后数据。

**分解应用**：识别 Prime Day 真实净提升约 150%（而非表面 500%），Forward Buy 比例占峰值的 55%，Q3 备货量从原计划下调 25%，避免滞销。

**量化产出**：避免 Q3 末滞销 1500 件，节省 FBA 处置成本约 **20-30 万元**。

## ③ 代码模板

```python
import numpy as np

def promotional_lift_decomposition(
    y: np.ndarray,
    promo_flags: np.ndarray,
    post_window: int = 3,
    controls: np.ndarray = None
) -> dict:
    """
    促销提升分解
    y: 周度销量序列（对数变换后更稳定）
    promo_flags: 促销标记序列（0/1）
    post_window: 促销后洼地观测窗口（周）
    controls: 控制变量（趋势、季节等）
    """
    n = len(y)
    log_y = np.log(np.maximum(y, 1))

    # 构建促销后哑变量（滞后 1-post_window 周的平均）
    post_promo = np.zeros(n)
    for t in range(n):
        if promo_flags[t] == 1:
            for k in range(1, post_window + 1):
                if t + k < n:
                    post_promo[t + k] = 1

    # 构建设计矩阵
    cols = [np.ones(n), promo_flags, post_promo]
    if controls is not None:
        cols.append(controls)
    X = np.column_stack(cols)

    # OLS 回归
    beta = np.linalg.lstsq(X, log_y, rcond=None)[0]
    promo_lift = np.exp(beta[1]) - 1     # 促销期间的销量提升比例
    post_dip = np.exp(beta[2]) - 1       # 促销后洼地比例（负值）
    net_lift = promo_lift + post_dip      # 净增量

    # 分解各成分
    base = np.exp(beta[0])
    y_pred = np.exp(X @ beta)
    y_base = np.exp(beta[0] + (X[:, 2:] @ beta[2:] if controls is not None else np.zeros(n)))

    return {
        'gross_lift_pct': promo_lift * 100,
        'post_dip_pct': post_dip * 100,
        'net_lift_pct': net_lift * 100,
        'forward_buy_ratio': abs(post_dip) / (promo_lift + 1e-8),
        'base_demand': base,
        'y_pred': y_pred,
        'coefficients': beta
    }

# 测试：模拟 Prime Day 促销场景
np.random.seed(42)
n = 52
base = np.random.poisson(100, n).astype(float)

# 第 20 周 Prime Day：销量暴涨 5x
promo = np.zeros(n)
promo[20] = 1
base[20] *= 5

# 促销后 1-3 周洼地：销量下降 40%
base[21:24] *= 0.6

# 时序趋势控制
trend = np.arange(n) * 0.5

result = promotional_lift_decomposition(base, promo, post_window=3, controls=trend)

assert result['gross_lift_pct'] > 100, f"促销提升应 > 100%，实际: {result['gross_lift_pct']:.1f}%"
assert result['post_dip_pct'] < 0, "促销后应有洼地"
assert result['forward_buy_ratio'] > 0

print(f"促销毛提升: +{result['gross_lift_pct']:.1f}%")
print(f"促销后洼地: {result['post_dip_pct']:.1f}%")
print(f"净真实提升: +{result['net_lift_pct']:.1f}%")
print(f"Forward Buy 比例: {result['forward_buy_ratio']:.1%}（峰值中的囤货/提前购买占比）")
print("[✓] Promotional-Lift-Decomposition 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-Bayesian-Structural-Time-Series]]（反事实基准建模）
> 延伸: [[Skill-Holiday-Spike-Demand-Decomposition]]（节日效应剥离）
> 可组合: [[Skill-Return-Rate-Forecasting-Model]]（促销后退货率预测）

## ⑤ 商业价值评估

- **ROI量化**: 避免大促后滞销，年化节省 FBA 处置成本 20-30 万元
- **实施难度**: ⭐⭐（需要历史促销标记，回归模型简单）
- **优先级**: ⭐⭐⭐⭐⭐（大促驱动的卖家预测精度核心提升）

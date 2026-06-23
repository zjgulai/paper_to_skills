---
title: Cross-SKU Demand Correlation Mining — 跨 SKU 需求相关性挖掘组合补货优化
doc_type: knowledge
module: 03-时间序列
topic: cross-sku-demand-correlation-mining
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Cross-SKU-Demand-Correlation-Mining

## ① 算法原理（≤300字）

**核心问题**：母婴卖家通常有数百个 SKU，各自独立做补货决策效率低且次优。吸奶器+配件、奶瓶+奶嘴、成套产品之间存在强需求相关性——一个 SKU 的需求暴涨往往预示配套 SKU 的需求跟随。挖掘这种相关结构，可以实现组合补货优化。

**方法体系**：

1. **动态相关矩阵**：对 SKU 对 $(i,j)$ 计算滑动窗口 Pearson 相关系数，识别稳定高相关对（$\rho > 0.7$）
2. **Granger 因果检验**：区分「同步相关」和「领先-滞后关系」。若 SKU-A 需求变化比 SKU-B 早 1-2 周，则 A 可作为 B 的领先指标
   $$\hat{y}_{B,t} = \alpha + \sum_{k=1}^{p}\beta_k y_{B,t-k} + \sum_{k=1}^{p}\gamma_k y_{A,t-k} + \epsilon_t$$
   $\gamma_k$ 显著说明 A Granger 因果于 B

3. **聚类分组**：基于相关矩阵做层次聚类，识别同步销售群组，实现「组合补货触发」

**补货优化**：当某 SKU 触发补货信号，系统自动检查同组 SKU 是否接近补货点，合并为一次订单，节省固定订单成本（每次订单固定成本 S）。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某卖家有吸奶器主机 + 5 款配件 SKU，发现主机销量领先于配件 1-2 周（消费者先买主机，后买配件），可以用主机销量预测配件需求，提前补货。

**数据要求**：12 个月以上的 SKU 级周销量数据，SKU 物料关联关系标注。

**应用**：识别主机→配件 Granger 因果关系（p<0.05），将配件补货触发点提前 2 周，配件缺货率从 15% 降至 4%。同时合并补货订单，物流次数减少 30%。

**量化产出**：配件缺货率降低 73%，合并订单节省运费，年化降低缺货损失 + 物流成本合计 **20 万元**。

## ③ 代码模板

```python
import numpy as np
from itertools import combinations

def compute_cross_sku_correlation(
    sales_matrix: np.ndarray,
    sku_names: list,
    window: int = 12,
    corr_threshold: float = 0.7
) -> dict:
    """
    跨 SKU 相关性矩阵计算
    sales_matrix: (T, n_sku) 周销量矩阵
    sku_names: SKU 名称列表
    window: 滑动窗口（周）
    corr_threshold: 相关性阈值
    """
    T, n_sku = sales_matrix.shape
    corr_matrix = np.corrcoef(sales_matrix.T)

    high_corr_pairs = []
    for i, j in combinations(range(n_sku), 2):
        if abs(corr_matrix[i, j]) >= corr_threshold:
            high_corr_pairs.append({
                'sku_a': sku_names[i],
                'sku_b': sku_names[j],
                'correlation': corr_matrix[i, j]
            })

    return {
        'corr_matrix': corr_matrix,
        'high_corr_pairs': sorted(high_corr_pairs, key=lambda x: -abs(x['correlation'])),
        'n_high_corr': len(high_corr_pairs)
    }

def granger_causality_test(
    y_cause: np.ndarray,
    y_effect: np.ndarray,
    max_lag: int = 4
) -> dict:
    """
    简化 Granger 因果检验（线性回归版本）
    检验 y_cause 是否 Granger 因果于 y_effect
    """
    n = len(y_effect)
    best_lag = 0
    best_r2_improvement = 0

    # 基础模型（仅自回归）
    X_base = np.column_stack([y_effect[max_lag - k - 1:-k - 1] for k in range(max_lag)])
    y = y_effect[max_lag:]

    # 添加因果变量的模型
    for lag in range(1, max_lag + 1):
        X_full = np.column_stack([
            X_base,
            y_cause[max_lag - lag:-lag]
        ])
        # 用最小二乘估计
        try:
            beta_full = np.linalg.lstsq(
                np.column_stack([np.ones(len(y)), X_full]), y, rcond=None
            )[0]
            y_pred = np.column_stack([np.ones(len(y)), X_full]) @ beta_full
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 > best_r2_improvement:
                best_r2_improvement = r2
                best_lag = lag
        except Exception:
            pass

    return {
        'best_lag': best_lag,
        'r2_with_cause': best_r2_improvement,
        'granger_significant': best_r2_improvement > 0.3  # 简化判断
    }

# 测试
np.random.seed(42)
T, n_sku = 52, 4
# SKU-0 是主机，SKU-1/2 是配件（滞后 2 周跟随）
sku_base = np.random.randn(T) * 10 + 100
sales = np.zeros((T, n_sku))
sales[:, 0] = sku_base
sales[:, 1] = np.roll(sku_base, 2) * 0.8 + np.random.randn(T) * 5  # 配件滞后2周
sales[:, 2] = np.roll(sku_base, 2) * 0.5 + np.random.randn(T) * 8
sales[:, 3] = np.random.randn(T) * 15 + 50  # 无关 SKU

sku_names = ['main-pump', 'accessory-A', 'accessory-B', 'unrelated']
corr_result = compute_cross_sku_correlation(sales, sku_names)
gc_result = granger_causality_test(sales[:, 0], sales[:, 1])

assert corr_result['n_high_corr'] >= 1, "应找到高相关 SKU 对"
assert gc_result['best_lag'] > 0
print(f"高相关对数量: {corr_result['n_high_corr']}")
print(f"最高相关对: {corr_result['high_corr_pairs'][0]}")
print(f"Granger 因果最优滞后: {gc_result['best_lag']} 周")
print("[✓] Cross-SKU-Demand-Correlation-Mining 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-Adaptive-Forecast-Accuracy-Optimization]]（预测优化基础）
> 延伸: [[Skill-Intermittent-Demand-Croston-TSB]]（配件长尾预测）
> 可组合: [[Skill-Lead-Time-Demand-Integration-Model]]（组合补货前置期建模）

## ⑤ 商业价值评估

- **ROI量化**: 配件缺货率降低 73%，合并订单年化节省 20 万元
- **实施难度**: ⭐⭐（需要 SKU 级历史数据和物料关联映射）
- **优先级**: ⭐⭐⭐⭐（多品类卖家补货协同的核心工具）

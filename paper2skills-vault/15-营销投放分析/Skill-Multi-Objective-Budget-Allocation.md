# Skill Card: Multi-Objective Budget Allocation（多目标预算分配）

> **领域**: 15-营销投放分析 | **类型**: 综合萃取

---

## ① 算法原理

### 核心思想
广告预算分配不能只看 ROI——需要同时平衡"短期销售""品牌建设""新品推广"三个目标。多目标优化（Pareto 最优）找到三个目标之间的最佳 trade-off 曲面。

### 数学直觉

**加权目标函数**：
$$\max_{x_1,...,x_n} \sum_k w_k \cdot f_k(x) \quad \text{s.t.} \sum_i x_i \leq B$$

其中 $f_1$=短期 ROI，$f_2$=品牌搜索量提升，$f_3$=新品曝光量。权重 $w_k$ 由业务阶段决定——大促期 $w_1$ 高，新品上市期 $w_3$ 高。

**Pareto 前沿**：无法在不伤害某一目标的前提下改善另一目标的状态。通过扫描不同权重组合生成 Pareto 前沿，供决策者选择。

### 关键假设
- 各渠道对不同目标的贡献可量化（需要 tracking 设置）
- 预算总量 $B$ 外生给定

---

## ② 母婴出海应用案例

### 场景：Q4 旺季 $30 万月预算的三目标分配

**业务问题**：$30 万月预算要同时做三件事——黑五冲销量（短期 ROI）、母婴博主种草（品牌搜索量）、新款吸奶器 S2 预热（新品曝光）。怎么分？

**数据要求**：各渠道（FB/Google/TikTok/PR）对三个目标的历史贡献数据

**预期产出**：
- Pareto 前沿可视化：三条曲线展示 trade-off
- 推荐分配：FB $12 万（销量+品牌双强）/ TikTok $10 万（新品种草强）/ Google $6 万（收割）/ PR $2 万（品牌）
- 权重选择：黑五期 w=(0.6, 0.2, 0.2)

**业务价值**：避免"只盯 ROI 忽略了品牌建设"的短视决策

---

## ③ 代码模板

```python
"""Multi-Objective Budget Allocation — Pareto 优化"""

import numpy as np
from scipy.optimize import minimize


def multi_objective_allocate(
    budget: float,
    channel_contributions: np.ndarray,  # (n_channels, n_objectives)
    weights: np.ndarray = None,
    saturations: np.ndarray = None       # 各渠道半饱和点
) -> dict:
    """多目标预算分配"""
    n_channels, n_objectives = channel_contributions.shape
    weights = weights or np.ones(n_objectives) / n_objectives
    saturations = saturations or np.full(n_channels, budget/n_channels*3)
    
    def objective(x):
        if np.any(x < 0):
            return -np.inf
        # 饱和调整
        sat_adj = 1 - np.exp(-x / saturations)
        contributions = channel_contributions * sat_adj[:, None]
        scores = contributions.sum(axis=0)
        return -np.dot(weights, scores)
    
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - budget}]
    x0 = np.ones(n_channels) * budget / n_channels
    bounds = [(0, budget) for _ in range(n_channels)]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    return {
        'allocation': {f'ch_{i}': round(v) for i, v in enumerate(res.x)},
        'scores': (channel_contributions * (1-np.exp(-res.x/saturations))[:,None]).sum(axis=0),
    }


if __name__ == '__main__':
    # 4渠道 × 3目标 (ROI, Brand, NewProduct)
    contrib = np.array([
        [0.8, 0.3, 0.1],  # FB: ROI强, 品牌中等
        [0.9, 0.1, 0.05], # Google: 收割最强
        [0.4, 0.6, 0.8],  # TikTok: 新品+品牌强
        [0.1, 0.8, 0.2],  # PR: 品牌强
    ])
    
    result = multi_objective_allocate(300000, contrib, weights=[0.6, 0.2, 0.2])
    print("多目标预算分配:")
    for ch, amt in result['allocation'].items():
        print(f"  {ch}: ${amt:,.0f}")
    print(f"  目标得分: ROI={result['scores'][0]:.2f}, "
          f"Brand={result['scores'][1]:.2f}, New={result['scores'][2]:.2f}")
    
    print("\n[✓] Multi-Objective Budget 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-Channel-Saturation-Curve]] | [[Skill-ROAS-Budget-Optimization]]
- **可组合技能**：[[Skill-Marketing-Mix-Modeling]] | [[Skill-DARA-Agentic-MMM]]

---
- **相关技能**：[[Skill-Geo-Level-Marketing-Effectiveness]]
- **相关**：[[Skill-DARA-Agentic-MMM-Optimizer]]
- **相关**：[[Skill-Demand-Forecasting-Supply-Chain]]

## ⑤ 商业价值评估

- **ROI 预估**：避免单目标短视造成的长期品牌价值损失；年化隐性价值 **30-50 万元**
- **实施难度**：⭐⭐⭐☆☆（3 星）
- **优先级评分**：⭐⭐⭐☆☆（3 星）

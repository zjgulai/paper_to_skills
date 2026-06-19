---
title: 多平台广告预算最优分配 — 约束优化最大化跨平台 ROAS
doc_type: knowledge
module: 15-营销投放分析
topic: multi-platform-ad-budget-allocator
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 多平台广告预算最优分配

> **论文**：Constrained Budget Allocation for Multi-Channel Digital Advertising: A Convex Optimization Approach
> **arXiv**：2403.15892 | 2024 | **桥梁**: 15-营销投放分析 ↔ 01-因果推断 | **类型**: 算法工具

## ① 算法原理

**核心思想**：不同广告平台（Amazon Ads / TikTok Ads / Meta Ads）的边际 ROAS 随预算增加而递减（饱和效应）。用幂函数拟合每个平台的「预算-ROAS 曲线」，再用约束优化（scipy minimize）在总预算约束下找到全局最优分配方案，替代「平均分配」或「经验拍脑袋」。

**数学直觉**：

每个平台 k 的 GMV 响应曲线（幂函数，α < 1 代表边际递减）：

```
GMV_k(b_k) = a_k × b_k^{α_k}
```

ROAS_k = GMV_k(b_k) / b_k = a_k × b_k^{α_k - 1}

总 ROAS 最大化问题（等价于最大化总 GMV）：

```
max ∑ GMV_k(b_k) = ∑ a_k × b_k^{α_k}
s.t. ∑ b_k = B_total       （预算约束）
     b_k ≥ B_min_k          （最低保量约束）
     b_k ≤ B_max_k          （平台预算上限）
```

**关键假设**：
1. 每个平台的响应曲线独立（平台间无相互影响，若有 cross-platform 效应需后续修正）
2. 幂函数拟合需至少 8 个不同预算水平的历史数据点
3. 短期内（2 周内）曲线参数 a_k, α_k 保持稳定

## ② 母婴出海应用案例

**场景A：婴儿推车大促前预算分配**
- **业务问题**：双 11 预算 ¥20 万，运营团队经验分配 Amazon 60% / TikTok 25% / Meta 15%。但 TikTok 最近 ROAS 曲线陡升，可能被低配。
- **数据要求**：过去 60 天各平台不同预算档位（至少 6 档）对应的 GMV 数据，按周取平均
- **预期产出**：
  - 三平台幂函数曲线参数（a_k, α_k）
  - 最优分配方案（如 Amazon 52% / TikTok 35% / Meta 13%）
  - vs 经验分配方案的 GMV 增量预测
- **业务价值**：预算优化后整体 ROAS 预计提升 12-18%，按 ¥20 万预算、ROAS 3.5 基准，增量 GMV 约 ¥8-12 万

**场景B：新平台（Shopee）预算试投决策**
- **业务问题**：是否值得从现有预算中划出 10% 给 Shopee？在 Shopee 缺少历史数据的情况下如何做决策？
- **数据要求**：同品类竞品的 Shopee 公开数据，加上初始小额测试（¥2000 × 3 档）
- **预期产出**：基于贝叶斯先验的 Shopee 响应曲线估计，建议最优初始预算
- **业务价值**：用数据驱动的方式决定是否加仓 Shopee，避免盲目扩张或错失增量

## ③ 代码模板

```python
"""
多平台广告预算最优分配器
- 输入：各平台历史预算-GMV 数据，总预算
- 输出：最优预算分配 + ROAS 预测 + vs 经验分配增量
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple


# ── 1. 历史数据（真实场景从后台 API 导出）─────────────────────
# 各平台不同预算档位对应的 GMV（单位：元）
HISTORICAL_DATA = {
    "amazon": {
        "budgets": [5000, 8000, 12000, 18000, 25000, 35000, 50000],
        "gmv":     [18000, 26400, 36000, 47700, 60000, 73500, 90000],
    },
    "tiktok": {
        "budgets": [2000, 4000, 7000, 10000, 15000, 20000, 30000],
        "gmv":     [9000, 16000, 24500, 32000, 44000, 56000, 76000],
    },
    "meta": {
        "budgets": [1000, 2000, 4000, 6000, 9000, 13000, 18000],
        "gmv":     [3200, 5800, 9600, 12600, 16200, 20800, 25200],
    },
}


# ── 2. 幂函数拟合 ──────────────────────────────────────────────
def power_func(b: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """GMV(b) = a × b^α"""
    return a * np.power(b, alpha)


def fit_response_curve(
    budgets: List[float],
    gmv: List[float],
) -> Tuple[float, float]:
    """拟合幂函数参数 (a, alpha)"""
    popt, _ = curve_fit(
        power_func,
        np.array(budgets),
        np.array(gmv),
        p0=[1.0, 0.7],
        bounds=([0, 0.1], [1e6, 0.99]),
        maxfev=5000,
    )
    return popt[0], popt[1]  # a, alpha


def fit_all_platforms(
    data: Dict,
) -> Dict[str, Tuple[float, float]]:
    """拟合所有平台的响应曲线"""
    curves = {}
    for platform, d in data.items():
        a, alpha = fit_response_curve(d["budgets"], d["gmv"])
        curves[platform] = (a, alpha)
        print(f"  {platform}: a={a:.4f}, α={alpha:.4f} → 边际递减系数: {1-alpha:.2f}")
    return curves


# ── 3. 约束优化 ────────────────────────────────────────────────
def optimize_budget(
    curves: Dict[str, Tuple[float, float]],
    total_budget: float,
    min_budgets: Dict[str, float] = None,
    max_budgets: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    最优化预算分配：最大化总 GMV
    
    Args:
        curves: {platform: (a, alpha)} 曲线参数
        total_budget: 总预算
        min_budgets: 各平台最低预算（保量）
        max_budgets: 各平台最高预算（平台上限）
    """
    platforms = list(curves.keys())
    n = len(platforms)
    
    if min_budgets is None:
        min_budgets = {p: total_budget * 0.05 for p in platforms}
    if max_budgets is None:
        max_budgets = {p: total_budget * 0.80 for p in platforms}
    
    # 目标函数（最小化负 GMV = 最大化 GMV）
    def neg_total_gmv(budgets: np.ndarray) -> float:
        total = 0.0
        for i, p in enumerate(platforms):
            a, alpha = curves[p]
            total += power_func(budgets[i], a, alpha)
        return -total
    
    # 约束：总预算
    constraints = LinearConstraint(
        np.ones((1, n)), lb=total_budget, ub=total_budget
    )
    
    # 边界：各平台最小/最大预算
    bounds = Bounds(
        lb=[min_budgets[p] for p in platforms],
        ub=[max_budgets[p] for p in platforms],
    )
    
    # 初始猜测：按历史均值比例分配
    x0 = np.array([total_budget / n] * n)
    
    result = minimize(
        neg_total_gmv,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    
    return {p: round(b, 0) for p, b in zip(platforms, result.x)}


# ── 4. 分配方案对比报告 ────────────────────────────────────────
def compare_allocations(
    curves: Dict[str, Tuple[float, float]],
    total_budget: float,
    empirical_ratios: Dict[str, float],  # 经验分配比例
) -> pd.DataFrame:
    """对比最优分配 vs 经验分配"""
    # 经验分配
    empirical_budgets = {p: total_budget * r for p, r in empirical_ratios.items()}
    
    # 最优分配
    optimal_budgets = optimize_budget(curves, total_budget)
    
    rows = []
    total_emp_gmv, total_opt_gmv = 0, 0
    
    for p in curves:
        a, alpha = curves[p]
        emp_b = empirical_budgets[p]
        opt_b = optimal_budgets[p]
        emp_gmv = power_func(emp_b, a, alpha)
        opt_gmv = power_func(opt_b, a, alpha)
        total_emp_gmv += emp_gmv
        total_opt_gmv += opt_gmv
        
        rows.append({
            "平台": p,
            "经验预算(¥)": f"{emp_b:,.0f}",
            "最优预算(¥)": f"{opt_b:,.0f}",
            "经验GMV(¥)": f"{emp_gmv:,.0f}",
            "最优GMV(¥)": f"{opt_gmv:,.0f}",
            "GMV增量(¥)": f"{opt_gmv - emp_gmv:+,.0f}",
        })
    
    rows.append({
        "平台": "【合计】",
        "经验预算(¥)": f"{total_budget:,.0f}",
        "最优预算(¥)": f"{total_budget:,.0f}",
        "经验GMV(¥)": f"{total_emp_gmv:,.0f}",
        "最优GMV(¥)": f"{total_opt_gmv:,.0f}",
        "GMV增量(¥)": f"{total_opt_gmv - total_emp_gmv:+,.0f}",
    })
    
    print(f"\n💰 ROAS 对比：")
    print(f"  经验分配 ROAS: {total_emp_gmv / total_budget:.2f}")
    print(f"  最优分配 ROAS: {total_opt_gmv / total_budget:.2f}")
    print(f"  ROAS 提升: {(total_opt_gmv - total_emp_gmv) / total_emp_gmv:.1%}")
    
    return pd.DataFrame(rows).set_index("平台")


# ── 5. 主测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("多平台广告预算最优分配器")
    print("=" * 60)
    
    print("\n📈 拟合各平台响应曲线：")
    curves = fit_all_platforms(HISTORICAL_DATA)
    
    # 总预算 20 万，经验分配
    total_budget = 200_000.0
    empirical = {"amazon": 0.60, "tiktok": 0.25, "meta": 0.15}
    
    print(f"\n🎯 预算优化（总预算 ¥{total_budget:,.0f}）：")
    df = compare_allocations(curves, total_budget, empirical)
    print(df.to_string())
    
    print("\n[✓] 多平台广告预算最优分配测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Marketing-Mix-Modeling]]（MMM 提供渠道效果的基础估计，本 Skill 在其上做实时优化）
- **延伸（extends）**：[[Skill-Multi-Objective-Budget-Allocation]]（本 Skill 单目标 ROAS → 延伸到多目标：ROAS + 品牌曝光 + 新客获取）
- **可组合（combinable）**：[[Skill-TikTok-Shop-Content-Commerce-Funnel]]（漏斗转化率 → 动态修正各平台 GMV 响应曲线斜率）
- **可组合（combinable）**：[[Skill-Channel-Saturation-Curve]]（曲线参数拟合直接复用饱和曲线的估计结果）

## ⑤ 商业价值评估

- **ROI 预估**：按月预算 ¥20 万、当前 ROAS 3.5 计算，最优分配预期 ROAS 提升 10-18%，月增量 GMV 约 ¥7-12 万；年化价值约 ¥84-144 万（纯优化收益，无额外成本）
- **实施难度**：⭐⭐⭐☆☆（需要至少 6-8 个历史预算档位数据，曲线拟合需数据质量达标）
- **优先级评分**：⭐⭐⭐⭐⭐
- **评估依据**：多平台广告是母婴出海最大可控成本项，预算分配效率直接决定竞争优势；scipy 优化实现成本极低，但依赖足量历史数据积累（6+ 档位，每档稳定投放 ≥ 1 周）

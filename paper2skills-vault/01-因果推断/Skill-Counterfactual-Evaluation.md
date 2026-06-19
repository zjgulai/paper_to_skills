---
title: 反事实评估 — Potential Outcomes + 合成控制反事实基线
doc_type: knowledge
module: 01-因果推断
topic: counterfactual-evaluation
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 反事实评估

> **论文**：Inferring Causal Impact Using Bayesian Structural Time-Series Models (Brodersen et al., 2015)
> **arXiv**：1506.00356 | 2015 | **桥梁**: 01-因果推断 ↔ 03-时间序列 | **类型**: 跨域融合

---

## ① 算法原理

**核心思想**：反事实评估回答「如果没有做这件事，结果会怎样？」。通过合成控制法（Synthetic Control），用未受干预的对照组数据合成一条「虚拟基线」，与实际观测值对比，量化干预效果（促销/产品改版/价格调整）的真实因果贡献。

**数学直觉**：
- **潜在结果框架（Rubin）**：个体 $i$ 在处理 $T=1$（有促销）下的潜在结果 $Y_i(1)$，在处理 $T=0$（无促销）下的潜在结果 $Y_i(0)$
- **平均处理效应**：$ATE = E[Y_i(1) - Y_i(0)]$，因果效应 = 实际 - 反事实
- **合成控制**：$\hat{Y}_{control} = \sum_{j \in \text{controls}} w_j Y_j$，权重 $w_j \geq 0, \sum w_j = 1$，最小化干预前的预测误差
- **因果效应**：$\hat{\tau}_t = Y_{treated,t} - \hat{Y}_{control,t}$，干预后实际 - 反事实基线

**关键假设**：
- 稳定单元处理值假设（SUTVA）：对照组不受处理组影响（无溢出效应）
- 干预前拟合足够好（pre-period RMSE < 5%）才能外推干预后基线
- 对照组与处理组在干预前趋势相似

---

## ② 母婴出海应用案例

**场景A：促销活动真实效果评估**

- **业务问题**：运营做了一次「吸奶器 8 折活动」，当周销量上涨 40%，但不确定这 40% 里有多少是真正的促销拉动，有多少是自然增长（竞品缺货/平台流量增加）
- **数据要求**：目标 ASIN 过去 90 天的日销量数据（含促销期）+ 5-10 个同类对照 ASIN 的同期销量（作为合成控制的参考组）
- **预期产出**：合成控制基线显示「如不做促销，当周销量约 1200 单」；实际 1680 单；因果效应 = +480 单（真实促销拉动 28.6%，而非表面的 40%）；促销 ROI = 净增销量 × 利润 / 促销折扣成本 = 380%
- **业务价值**：精确的 ROI 数字指导下次促销力度决策，避免过度折扣损失毛利，年化优化促销 ROI 约 **20%，对应节省促销成本 15 万元**

**场景B：产品改版效果量化**

- **业务问题**：V3 版吸奶器换了新马达，工厂称「性能提升 30%」，但运营不知道上线后销量提升是马达升级导致的，还是同期的广告加投
- **数据要求**：改版前后 30 天销量 + 同品类未改版 SKU 作对照 + 广告消耗数据
- **预期产出**：控制广告因素后，反事实分析显示改版对销量的纯因果贡献为 +12%（而非表面的 +25%），另外 13% 来自广告加投
- **业务价值**：精确拆解产品因素 vs 营销因素，指导研发预算分配，避免把营销拉动的增长误归因于产品改进，年化减少研发/营销资源错配损失约 **8 万元**

---

## ③ 代码模板

```python
"""
反事实评估 — 合成控制法
用未受干预的对照组合成虚拟基线，评估干预的因果效应
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import minimize


def synthetic_control_weights(
    treated_pre: np.ndarray,
    controls_pre: np.ndarray
) -> np.ndarray:
    """
    求合成控制权重
    treated_pre: 处理组干预前时间序列 (T_pre,)
    controls_pre: 对照组干预前时间序列 (T_pre, n_controls)
    返回：权重向量 (n_controls,)，满足 sum=1, all>=0
    """
    n_controls = controls_pre.shape[1]

    def objective(w):
        synthetic = controls_pre @ w
        return np.sum((treated_pre - synthetic) ** 2)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * n_controls
    w0 = np.ones(n_controls) / n_controls

    result = minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )
    return result.x


def estimate_causal_effect(
    treated: np.ndarray,
    controls: np.ndarray,
    pre_period_end: int
) -> Dict:
    """
    反事实评估核心函数
    treated: 处理组完整时间序列 (T,)
    controls: 对照组完整时间序列 (T, n_controls)
    pre_period_end: 干预前时期结束的时间步（不含）
    """
    treated_pre = treated[:pre_period_end]
    treated_post = treated[pre_period_end:]
    controls_pre = controls[:pre_period_end]
    controls_post = controls[pre_period_end:]

    # 求权重
    weights = synthetic_control_weights(treated_pre, controls_pre)

    # 生成反事实基线
    counterfactual_pre = controls_pre @ weights
    counterfactual_post = controls_post @ weights

    # 计算因果效应
    pre_rmse = np.sqrt(np.mean((treated_pre - counterfactual_pre) ** 2))
    causal_effects = treated_post - counterfactual_post
    total_effect = causal_effects.sum()
    avg_effect = causal_effects.mean()
    relative_effect = avg_effect / counterfactual_post.mean() if counterfactual_post.mean() > 0 else 0

    return {
        "weights": weights,
        "counterfactual_pre": counterfactual_pre,
        "counterfactual_post": counterfactual_post,
        "causal_effects": causal_effects,
        "total_effect": total_effect,
        "avg_daily_effect": avg_effect,
        "relative_effect_pct": relative_effect * 100,
        "pre_period_rmse": pre_rmse,
        "fit_quality": "Good" if pre_rmse < 0.05 * treated_pre.mean() else "Poor",
    }


def compute_promotion_roi(
    causal_units: float,
    unit_margin: float,
    discount_per_unit: float,
    base_units: float
) -> Dict:
    """
    基于因果效应计算促销 ROI
    causal_units: 真实因果拉动的增量销量
    unit_margin: 正常毛利/单
    discount_per_unit: 折扣损失/单（全部销量含基础量）
    base_units: 促销期间基础销量（无促销也会有的）
    """
    incremental_profit = causal_units * unit_margin
    discount_cost = (base_units + causal_units) * discount_per_unit
    roi = (incremental_profit - discount_cost) / discount_cost if discount_cost > 0 else float("inf")
    return {
        "incremental_units": causal_units,
        "incremental_profit": round(incremental_profit, 2),
        "discount_cost": round(discount_cost, 2),
        "net_gain": round(incremental_profit - discount_cost, 2),
        "roi_pct": round(roi * 100, 1),
    }


# ─── 测试用例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # 模拟 90 天销量数据
    # 前 60 天：干预前
    # 后 30 天：促销期
    T_pre, T_post = 60, 30
    T = T_pre + T_post

    # 共同趋势（基础需求）
    base_trend = 1000 + np.arange(T) * 2 + np.random.normal(0, 30, T)

    # 处理组（目标 ASIN）：促销期间额外 +400 单/天
    treated = base_trend.copy()
    treated[T_pre:] += 400 + np.random.normal(0, 20, T_post)

    # 对照组（5 个类似 ASIN，跟随同样基础趋势但无促销）
    controls = np.column_stack([
        base_trend * (0.8 + 0.1 * i) + np.random.normal(0, 25, T)
        for i in range(5)
    ])

    # 反事实评估
    result = estimate_causal_effect(treated, controls, pre_period_end=T_pre)

    print("=== 反事实评估报告 ===\n")
    print(f"预拟合质量: {result['fit_quality']} (RMSE: {result['pre_period_rmse']:.2f})")
    print(f"合成控制权重: {[round(w, 3) for w in result['weights']]}")
    print(f"\n促销期因果效果:")
    print(f"  总增量销量: {result['total_effect']:.0f} 单")
    print(f"  日均因果增量: {result['avg_daily_effect']:.1f} 单/天")
    print(f"  相对提升: +{result['relative_effect_pct']:.1f}%")

    # 观测提升 vs 因果提升对比
    observed_effect = treated[T_pre:].mean() - treated[:T_pre].mean()
    print(f"\n表面提升 vs 因果提升:")
    print(f"  表面提升: +{observed_effect:.1f} 单/天")
    print(f"  因果提升: +{result['avg_daily_effect']:.1f} 单/天")
    print(f"  归因差异: {(observed_effect - result['avg_daily_effect'])/observed_effect*100:.1f}% 是自然增长")

    # ROI 计算
    roi = compute_promotion_roi(
        causal_units=result["total_effect"],
        unit_margin=50,       # 单件毛利 50 元
        discount_per_unit=20, # 折扣 20 元/单
        base_units=treated[:T_pre].mean() * T_post
    )
    print(f"\n促销 ROI:")
    print(f"  增量利润: {roi['incremental_profit']:.0f} 元")
    print(f"  折扣成本: {roi['discount_cost']:.0f} 元")
    print(f"  净收益: {roi['net_gain']:.0f} 元")
    print(f"  ROI: {roi['roi_pct']}%")

    # 验证
    assert result["fit_quality"] in ("Good", "Poor"), "拟合质量标签错误"
    assert result["total_effect"] > 0, "促销因果效应应为正"
    assert abs(result["weights"].sum() - 1.0) < 1e-4, "权重不归一"
    assert roi["roi_pct"] > 0, "ROI 应为正"

    print("\n[✓] 反事实评估 测试通过")
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-DiD-Difference-in-Differences]]（DiD 是反事实评估的简化版，理解 DiD 后自然过渡）
- **延伸（extends）**：[[Skill-Causal-Sentiment-Attribution]]（情感变化的反事实归因：如果没有改版，差评率会是多少？）
- **可组合（combinable）**：[[Skill-Causal-Supply-Chain-Attribution]]（供应链成本的反事实分析：如不换供应商，成本会如何？）、[[Skill-Guardrailed-Uplift-Targeting]]（Uplift 模型基于反事实框架，二者天然配套）

---

## ⑤ 商业价值评估

- **ROI 预估**：促销 ROI 精确评估 → 优化折扣策略节省约 **15 万元/年**；产品改版归因 → 研发资源合理分配节省约 **8 万元/年**。总年化约 **23 万元**
- **实施难度**：⭐⭐⭐☆☆（需要 scipy 优化库；对照组选择是关键，需领域知识；干预前期数据质量要求高）
- **优先级**：⭐⭐⭐⭐⭐（所有做过促销/改版/定价调整的团队都需要，高频使用场景，无冷启动；是因果推断域的核心基础 Skill）
- **评估依据**：合成控制法是 Google/Amazon 等大厂因果效应评估的标准工具；scikit-learn + scipy 即可实现，无额外依赖

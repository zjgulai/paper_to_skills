---
title: Promotion-Aware Demand Decomposition
module: 04-供应链
topic: promotional-demand-planning
status: stable
domain: supply_chain
papers:
  - id: "2411.05852"
    title: "SPADE: Split Peak Attention DEcomposition"
    venue: "NeurIPS 2024 Time Series Workshop"
    role: 主论文（神经双通道分离，PPE carry-over 修正，30%提升）
  - id: "Hewage-JoF-2025"
    title: "Enhancing Demand Forecasting: Promotional Effects on Entire Demand Life Cycle"
    venue: "Journal of Forecasting 2025"
    role: Base-Lift 公式 + 完整三段生命周期建模
  - id: "SSRN-4777632"
    title: "Demand Forecasting During Grand Promotion for Online Retailing (JD.com)"
    venue: "SSRN 2024"
    role: 中国电商大促场景验证（618/双11，DWT+Bayesian LASSO）
---

# Skill-Promotion-Demand-Decomposition

## ① 算法原理

**核心思想**：把任意一个 SKU 的历史销量分解为两个互相独立的信号——「基线需求」（正常销售节奏）和「促销 lift」（大促拉动的额外需求）——分别建模、分别备货，并专门处理大促结束后需求虚高的「Post-Promotion Elevation（PPE）」问题，避免系统在大促后 3 个月内持续过量备货。

**三段式促销生命周期（Hewage 2025）**：

```
促销前期（Pre-promo）: 预期心理 → 部分消费者推迟购买，基线小幅下降
促销期间（During）:    需求爆发 = Baseline_t + Lift_t
促销后期（Post-promo）: 囤货效应 → 需求低于正常基线，需单独建模

total_demand_t = Baseline_t + Lift_t + PostDip_t

其中：
  Baseline_t = 去促销数据上拟合的 ARIMA/ML 预测
  Lift_t     = (actual_t - Baseline_t) × 促销标志位
  PostDip_t  = -Baseline_t × stockpiling_ratio  (大促后 2-6 周)
```

**SPADE 双通道架构（arXiv:2411.05852）**：

```
输入：时序销量 + 促销日历标志位（已知的未来大促时间）

Non-PE 分支（基线预测器）：
  RobustConvolution：促销期间用 forward-fill 屏蔽，避免峰值污染基线
  → 输出稳健的基线预测，对 PPE 免疫

PE 分支（促销峰值预测器）：
  PeakAttention：跨历史大促学习峰值幅度模式
  → 只在促销标志位=1 时激活

final_forecast_t = Non-PE_t + PE_t × promo_flag_t

PPE 修正（SPADE 核心贡献）：
  大促结束后 PE 分支自然归零，Non-PE 分支恢复基线
  → 解决传统模型"大促后预测持续偏高 30%"的系统性偏差
```

**DWT 大促频谱分解（SSRN JD.com 2024）**：

```
# 离散小波变换将销量分解为低频基线 + 高频促销波动
coeffs = pywt.wavedec(sales, 'db4', level=3)
  低频系数 (cA3): 基线需求趋势
  高频系数 (cD1, cD2, cD3): 促销冲击 + 季节波动
  
# Bayesian LASSO 稀疏化高频系数，识别真实促销信号
# JD.com 618/双11 实证：预测误差下降 4-11%
```

**关键假设**：
- 大促日期已知（供应商提前 2-3 个月确认）
- 历史大促数据 ≥ 2 次（用于 lift 系数学习）
- 消费者囤货效应（post-dip）在母婴品类约持续 2-4 周

---

## ② 母婴出海应用案例

**场景 A：Momcozy 618 大促备货拆解（5 月初下单，lead time 6 周）**

- **业务问题**：吸奶器 SKU 日常基线销量 80 件/天，历史 618 大促期间销量 350 件/天（lift = 4.4x）。但大促后 3 周销量仅 45 件/天（post-dip）。传统方法按大促均值备货导致大促后积压 6 周。
- **数据要求**：过去 2 年日销量（含促销标志位）、大促日历（618 日期区间）
- **预期产出**：
  ```
  基线备货量（non-promo buffer）: 80件/天 × 15天 = 1,200件
  大促 lift 备货量: (350-80)件/天 × 7天 = 1,890件
  Post-dip 减备量: (80-45)件/天 × 21天 = -735件
  
  最优总备货 = 1,200 + 1,890 - 735 = 2,355件
  vs 传统方法（按大促均值）= 350×22天 = 7,700件（过量 3.3x）
  ```
- **业务价值**：避免过量备货 5,345 件，按 $35/件 × 20% 持有成本 = 节省约 $37,415 仓储资金占用

**场景 B：黑五/Prime Day 四次大促的 lift 系数标定**

- **业务问题**：四次大促（618/双11/黑五/Prime Day）的 lift 系数不同（平台流量结构不同），当前统一用"3倍备货"导致黑五备货不足、双11积压。
- **预期产出**：
  | 大促 | 历史 lift 系数 | Post-dip 周数 | 最优备货系数 |
  |------|--------------|--------------|------------|
  | 618  | 4.4x | 3 周 | 2.8x（扣除 post-dip）|
  | 双11 | 6.2x | 4 周 | 3.9x |
  | 黑五 | 3.1x | 2 周 | 2.3x |
  | Prime Day | 2.8x | 2 周 | 2.1x |

---

## ③ 代码模板

```python
"""
Skill-Promotion-Demand-Decomposition
基于 SPADE (arXiv:2411.05852, NeurIPS 2024) +
    Hewage et al. (Journal of Forecasting 2025) +
    Chi et al. JD.com (SSRN:4777632, 2024)
母婴跨境电商大促需求分解与备货量计算
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromoPlan:
    sku_id: str
    promo_name: str
    baseline_daily: float
    lift_multiplier: float
    promo_days: int
    post_dip_ratio: float
    post_dip_days: int
    pre_dip_ratio: float
    pre_dip_days: int

    @property
    def baseline_stock(self) -> float:
        return self.baseline_daily * (self.pre_dip_days + self.promo_days + self.post_dip_days)

    @property
    def lift_stock(self) -> float:
        return (self.lift_multiplier - 1) * self.baseline_daily * self.promo_days

    @property
    def post_dip_reduction(self) -> float:
        return self.post_dip_ratio * self.baseline_daily * self.post_dip_days

    @property
    def optimal_stock(self) -> float:
        return self.baseline_stock + self.lift_stock - self.post_dip_reduction

    @property
    def naive_stock(self) -> float:
        return self.lift_multiplier * self.baseline_daily * (self.pre_dip_days + self.promo_days + self.post_dip_days)

    @property
    def saving_vs_naive(self) -> float:
        return self.naive_stock - self.optimal_stock


# ── Base-Lift 分解（Hewage 2025 方法）──────────────────────
def decompose_baseline_lift(
    sales: pd.Series,
    promo_flags: pd.Series,
    method: str = "rolling_median",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Base-Lift 分解：total = baseline + lift + post_dip

    Args:
        sales: 日销量时序
        promo_flags: 促销标志（1=促销期，0=正常，-1=post-dip 期）
        method: 基线估算方法（rolling_median / interpolation）

    Returns: (baseline, lift, post_dip)
    """
    baseline = sales.copy().astype(float)

    mask_active = promo_flags != 0
    baseline[mask_active] = np.nan

    if method == "rolling_median":
        baseline = baseline.fillna(
            baseline.rolling(window=14, min_periods=3, center=True).median()
        )
    baseline = baseline.interpolate(method="linear").ffill().bfill()

    lift = pd.Series(0.0, index=sales.index)
    promo_mask = promo_flags == 1
    lift[promo_mask] = (sales[promo_mask] - baseline[promo_mask]).clip(lower=0)

    post_dip = pd.Series(0.0, index=sales.index)
    dip_mask = promo_flags == -1
    post_dip[dip_mask] = (baseline[dip_mask] - sales[dip_mask]).clip(lower=0)

    return baseline, lift, post_dip


# ── 历史 Lift 系数学习 ────────────────────────────────────
def learn_lift_coefficients(
    sales_history: pd.DataFrame,
    promo_calendar: list[dict],
) -> dict:
    """
    从历史大促数据学习各大促的 lift 系数和 post-dip 参数。

    Args:
        sales_history: 含 date/sales/sku 列的 DataFrame
        promo_calendar: [{'name': '618', 'start': '2025-06-18', 'end': '2025-06-20', 'post_days': 21}]

    Returns: {promo_name: {'lift_mean': x, 'lift_std': x, 'post_dip_ratio': x, 'post_dip_days': x}}
    """
    results = {}
    for promo in promo_calendar:
        name = promo["name"]
        start = pd.Timestamp(promo["start"])
        end = pd.Timestamp(promo["end"])
        post_end = end + pd.Timedelta(days=promo.get("post_days", 21))
        pre_start = start - pd.Timedelta(days=30)

        pre_sales = sales_history[
            (sales_history["date"] >= pre_start) & (sales_history["date"] < start)
        ]["sales"].mean()

        promo_sales = sales_history[
            (sales_history["date"] >= start) & (sales_history["date"] <= end)
        ]["sales"].mean()

        post_sales = sales_history[
            (sales_history["date"] > end) & (sales_history["date"] <= post_end)
        ]["sales"].mean()

        lift = promo_sales / pre_sales if pre_sales > 0 else 1.0
        post_dip_ratio = max(0.0, (pre_sales - post_sales) / pre_sales) if pre_sales > 0 else 0.0

        results[name] = {
            "lift_mean": round(lift, 2),
            "post_dip_ratio": round(post_dip_ratio, 2),
            "post_dip_days": promo.get("post_days", 21),
            "baseline_daily": round(pre_sales, 1),
        }
    return results


# ── SPADE 风格 PPE 修正（轻量版）────────────────────────────
def spade_ppe_correction(
    raw_forecast: np.ndarray,
    promo_flags: np.ndarray,
    post_promo_window: int = 21,
    ppe_decay: float = 0.15,
) -> np.ndarray:
    """
    SPADE PPE 修正：大促结束后，逐步将预测拉回基线水平。
    decay: 每天衰减 ppe_decay 比例的 carry-over 偏差。
    """
    corrected = raw_forecast.copy().astype(float)
    n = len(raw_forecast)
    in_ppe = False
    ppe_day = 0

    for t in range(1, n):
        if promo_flags[t - 1] == 1 and promo_flags[t] == 0:
            in_ppe = True
            ppe_day = 0

        if in_ppe:
            ppe_day += 1
            carry_over_bias = raw_forecast[t] - (raw_forecast[t] / (1 + ppe_decay)) ** ppe_day
            corrected[t] = max(0.0, raw_forecast[t] - carry_over_bias * ppe_decay)
            if ppe_day >= post_promo_window:
                in_ppe = False

    return corrected


# ── 备货计划生成 ──────────────────────────────────────────
def generate_promo_stock_plan(
    sku_id: str,
    lift_params: dict,
    lead_time_days: int = 42,
    service_level: float = 0.95,
    demand_cv: float = 0.3,
) -> dict:
    """
    基于分解后的各成分生成备货计划。
    包含安全库存 buffer（考虑 lift 预测不确定性）。
    """
    from scipy import stats

    plans = {}
    for promo_name, params in lift_params.items():
        plan = PromoPlan(
            sku_id=sku_id,
            promo_name=promo_name,
            baseline_daily=params["baseline_daily"],
            lift_multiplier=params["lift_mean"],
            promo_days=7,
            post_dip_ratio=params["post_dip_ratio"],
            post_dip_days=params["post_dip_days"],
            pre_dip_ratio=0.05,
            pre_dip_days=7,
        )

        z = stats.norm.ppf(service_level)
        lift_uncertainty = params["baseline_daily"] * (params["lift_mean"] - 1) * demand_cv
        safety_buffer = z * lift_uncertainty * np.sqrt(lead_time_days / 30)

        plans[promo_name] = {
            "optimal_stock": round(plan.optimal_stock + safety_buffer),
            "naive_stock": round(plan.naive_stock),
            "saving_vs_naive": round(plan.saving_vs_naive - safety_buffer),
            "lift_coefficient": params["lift_mean"],
            "post_dip_ratio": params["post_dip_ratio"],
            "safety_buffer": round(safety_buffer),
        }
    return plans


# ── 示例：Momcozy 四次大促备货计划 ───────────────────────
if __name__ == "__main__":
    momcozy_lift_params = {
        "618":       {"baseline_daily": 80, "lift_mean": 4.4, "post_dip_ratio": 0.40, "post_dip_days": 21},
        "双11":      {"baseline_daily": 80, "lift_mean": 6.2, "post_dip_ratio": 0.50, "post_dip_days": 28},
        "黑五":      {"baseline_daily": 80, "lift_mean": 3.1, "post_dip_ratio": 0.25, "post_dip_days": 14},
        "Prime Day": {"baseline_daily": 80, "lift_mean": 2.8, "post_dip_ratio": 0.20, "post_dip_days": 14},
    }

    plans = generate_promo_stock_plan(
        sku_id="Momcozy-S12-Pro",
        lift_params=momcozy_lift_params,
        lead_time_days=42,
        service_level=0.95,
    )

    print("=" * 65)
    print("Momcozy S12 Pro — 四次大促备货计划（促销需求分解法）")
    print("=" * 65)
    annual_saving = 0
    for promo, p in plans.items():
        saving_usd = p["saving_vs_naive"] * 35 * 0.20
        annual_saving += max(0, saving_usd)
        print(f"\n{promo}:")
        print(f"  最优备货量: {p['optimal_stock']:,} 件  (含安全库存 {p['safety_buffer']:,} 件)")
        print(f"  传统方法量: {p['naive_stock']:,} 件")
        print(f"  节省备货: {p['saving_vs_naive']:,} 件 ≈ ${max(0,saving_usd):,.0f} 资金占用")
        print(f"  Lift系数: {p['lift_coefficient']}x | Post-dip: -{p['post_dip_ratio']*100:.0f}%")

    print(f"\n年化节省（4次大促资金占用）: ${annual_saving:,.0f}")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Demand-Forecasting-Supply-Chain]] — 基线需求预测是本 Skill 的输入
  - [[Skill-Causal-Time-Series-Forecasting-GCF]] — 因果时序预测框架，本 Skill 是其促销场景专项化
- **延伸技能**：
  - [[Skill-Safety-Stock-Replenishment]] — 大促后 post-dip 期间安全库存应相应降低
  - [[Skill-Multi-SKU-Procurement-Budget-Allocation]] — 大促备货量是预算分配的关键输入
- **可组合**：
  - [[Skill-Data-Drift-Detection]] — 大促后基线污染检测（防止大促数据扭曲模型）
  - [[Skill-Lead-Time-Distribution-Risk-GenQOT]] — 大促前 lead time 压缩风险与备货量联合决策

---
- **相关技能**：[[Skill-Supplier-Capacity-Planning]]
- **相关技能**：[[Skill-Inventory-Health-Aging-Attribution]]

## ⑤ 商业价值评估

- **ROI 预估**：
  - Momcozy 四次大促（618/双11/黑五/Prime Day），以 S12 Pro 主力 SKU 估算：
  - 传统"3倍均值备货"年均过量备货约 8,000-12,000 件
  - 促销分解法节省过量备货约 60%，按 $35 × 20% 持有成本 = **年节省约 $33,600-$50,400**
  - 同时降低大促后 3 个月的 FBA 长期仓储费约 $2,000-$4,000
- **实施难度**：⭐⭐☆☆☆（2/5）— pandas + scipy，无需 GPU
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）— 每年 4 次大促，每次备货决策直接影响 $10K+ 资金
- **评估依据**：
  - SPADE 实测：PPE 期间预测改善 30%，NeurIPS 2024 Workshop 验证
  - JD.com 数据：4-11% 预测误差下降（SSRN 2024）
  - Hewage 2025：base-lift 模型准确率与 SOTA ML 相当，实施成本更低

---

## 元信息

```yaml
skill_id: Skill-Promotion-Demand-Decomposition
domain: supply_chain
vault_path: paper2skills-vault/04-供应链/Skill-Promotion-Demand-Decomposition.md
code_path: paper2skills-code/supply_chain/promotion_demand_decomposition/
review_score: 8.5/10
wf_coverage: [WF-A]
created: 2026-05-25
```

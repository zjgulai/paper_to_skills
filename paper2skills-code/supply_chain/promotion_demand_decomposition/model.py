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

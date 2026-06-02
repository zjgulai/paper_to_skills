"""
Skill-New-Product-Inventory-Coldstart
基于 Ban, Gallien & Mersereau M&SOM 2019 (类比SKU残差树) +
    Keskin, Li & Song OR 2023 (Bayesian探索加成) +
    Lee et al. TF&SC 2014 (Bass参数ML估计)
母婴跨境 DTC 新品冷启动库存策略
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar


@dataclass
class AnalogSKU:
    sku_id: str
    monthly_sales: list[float]
    unit_price: float
    category: str
    launch_month: int
    price_tier: str

    @property
    def demand_mean(self) -> float:
        return float(np.mean(self.monthly_sales))

    @property
    def demand_std(self) -> float:
        return float(np.std(self.monthly_sales))

    @property
    def cv(self) -> float:
        return self.demand_std / max(self.demand_mean, 1)


@dataclass
class NewProductSpec:
    sku_id: str
    unit_price: float
    category: str
    price_tier: str
    launch_month: int
    lead_time_months: float = 1.5
    holding_cost_rate: float = 0.20
    stockout_cost_multiplier: float = 2.0


def find_analog_skus(
    new_product: NewProductSpec,
    catalog: list[AnalogSKU],
    n_analogs: int = 3,
) -> list[tuple[AnalogSKU, float]]:
    """
    按相似度匹配类比 SKU（Ban et al. 协变量回归的简化版）。
    相似度 = 类别匹配 × 价格段距离 × 上市季节对齐
    """
    scored = []
    for sku in catalog:
        sim = 0.0
        if sku.category == new_product.category:
            sim += 0.5
        price_diff = abs(sku.unit_price - new_product.unit_price) / max(new_product.unit_price, 1)
        sim += 0.3 * max(0, 1 - price_diff)
        if sku.price_tier == new_product.price_tier:
            sim += 0.2
        scored.append((sku, round(sim, 3)))

    scored.sort(key=lambda x: -x[1])
    return scored[:n_analogs]


def residual_tree_prior(
    analog_skus: list[tuple[AnalogSKU, float]],
) -> tuple[float, float]:
    """
    类比 SKU 加权估计新品需求先验 (μ₀, σ₀)。
    简化版 Ban et al. 残差树：用相似度加权平均。
    """
    total_sim = sum(sim for _, sim in analog_skus)
    if total_sim == 0:
        return 100.0, 50.0

    mu = sum(sku.demand_mean * sim for sku, sim in analog_skus) / total_sim
    sigma = sum(sku.demand_std * sim for sku, sim in analog_skus) / total_sim
    return mu, sigma


def exploration_boost(
    mu_prior: float,
    sigma_prior: float,
    service_level: float = 0.85,
) -> tuple[float, float]:
    """
    Keskin et al. 2023：探索加成 = 短视最优量 + 额外订购量
    探索加成 ∝ 后验离散度指数 σ²/μ（参数不确定性）
    """
    pdi = sigma_prior ** 2 / max(mu_prior, 1)
    z = stats.norm.ppf(service_level)
    q_myopic = mu_prior + z * sigma_prior
    boost = pdi * 0.5
    q_bdp = q_myopic + boost

    return round(q_bdp), round(boost)


def bayesian_update(
    mu_prior: float,
    sigma_prior: float,
    observed_sales: list[float],
    inventory_levels: list[float],
) -> tuple[float, float]:
    """
    上市后贝叶斯更新（处理右删失销量数据）。
    censored_obs: 若库存量 < 真实需求，观测值为库存量（删失）
    """
    for obs, inv in zip(observed_sales, inventory_levels):
        is_censored = obs >= inv * 0.95
        if not is_censored:
            likelihood_precision = 1.0 / max(obs * 0.1, 1) ** 2
            prior_precision = 1.0 / max(sigma_prior, 1) ** 2
            posterior_precision = prior_precision + likelihood_precision
            mu_prior = (prior_precision * mu_prior + likelihood_precision * obs) / posterior_precision
            sigma_prior = np.sqrt(1.0 / posterior_precision)
        else:
            sigma_prior *= 0.95

    return round(mu_prior, 1), round(sigma_prior, 1)


def bass_initial_order(
    market_potential: float,
    p: float = 0.018,
    q: float = 0.32,
    lead_time_months: float = 1.5,
    conservative_factor: float = 0.35,
) -> dict:
    """
    Bass 扩散模型估计无类比SKU时的首批量。
    conservative_factor: 首批试水系数（0.3-0.5）
    """
    peak_demand_month = market_potential * (p + q) ** 2 / (4 * q)
    months_to_peak = np.log(q / p) / (p + q)
    initial_order = peak_demand_month * lead_time_months * conservative_factor

    return {
        "peak_monthly_demand": round(peak_demand_month),
        "months_to_peak": round(months_to_peak, 1),
        "recommended_first_order": round(initial_order),
        "rationale": f"Bass峰值{peak_demand_month:.0f}件/月 × {lead_time_months}月LT × {conservative_factor}保守系数",
    }


def cold_start_plan(
    new_product: NewProductSpec,
    analog_catalog: list[AnalogSKU],
    service_level: float = 0.85,
) -> dict:
    """
    完整冷启动库存计划：类比先验 → 探索加成首批量 → 贝叶斯更新策略
    """
    analogs = find_analog_skus(new_product, analog_catalog)
    if not analogs:
        return {"error": "无类比SKU，使用Bass fallback", "analogs": []}

    mu0, sigma0 = residual_tree_prior(analogs)
    q_first, boost = exploration_boost(mu0, sigma0, service_level)
    q_first_units = round(q_first * new_product.lead_time_months)

    return {
        "prior_demand_mean": round(mu0, 1),
        "prior_demand_std": round(sigma0, 1),
        "cv": round(sigma0 / max(mu0, 1), 2),
        "q_myopic": round(mu0 * new_product.lead_time_months),
        "exploration_boost": round(boost * new_product.lead_time_months),
        "recommended_first_order": q_first_units,
        "analog_skus": [(a.sku_id, round(s, 2)) for a, s in analogs],
        "strategy": "类比SKU + 探索加成" if sigma0 / max(mu0, 1) > 0.25 else "类比SKU（低不确定性，短视策略足够）",
    }


if __name__ == "__main__":
    catalog = [
        AnalogSKU("UV-C-X100", [280,310,290,350,320,340,300,280], 129.0, "sterilizer", 3, "premium"),
        AnalogSKU("UV-C-Basic", [180,210,195,220,200,215,190,205], 89.0, "sterilizer", 6, "mid"),
        AnalogSKU("Steam-Pro", [450,480,460,500,490,510,470,440], 79.0, "sterilizer", 1, "mid"),
        AnalogSKU("Wipe-Warmer", [120,130,115,140,125,135,110,120], 45.0, "accessory", 4, "mid"),
    ]

    new_sku = NewProductSpec("UV-C-X200", 149.0, "sterilizer", "premium", 7, lead_time_months=1.5)

    print("=" * 65)
    print("新品冷启动库存计划：UV-C X200")
    print("=" * 65)

    plan = cold_start_plan(new_sku, catalog)
    print(f"\n先验估计（类比SKU法）:")
    print(f"  月需求均值: {plan['prior_demand_mean']} 件，标准差: {plan['prior_demand_std']} 件 (CV={plan['cv']:.0%})")
    print(f"  类比SKU: {plan['analog_skus']}")
    print(f"\n首批量决策:")
    print(f"  短视策略量: {plan['q_myopic']} 件")
    print(f"  探索加成:  +{plan['exploration_boost']} 件 （{plan['strategy']}）")
    print(f"  推荐首批量: {plan['recommended_first_order']} 件 （覆盖 {new_sku.lead_time_months} 月 LT）")

    print(f"\n上市后贝叶斯更新（模拟前3周）:")
    mu, sigma = plan['prior_demand_mean'], plan['prior_demand_std']
    weekly_sales = [68, 82, 95]
    weekly_inv   = [90, 90, 90]
    for week, (obs, inv) in enumerate(zip(weekly_sales, weekly_inv), 1):
        mu, sigma = bayesian_update(mu, sigma, [obs], [inv])
        print(f"  第{week}周 销量={obs}, 库存={inv} → 更新: μ={mu}, σ={sigma}")

    print(f"\nBass 估计（全新品类 fallback）:")
    bass = bass_initial_order(market_potential=3000, p=0.018, q=0.32, lead_time_months=1.5)
    print(f"  峰值月需求: {bass['peak_monthly_demand']} 件（第 {bass['months_to_peak']} 个月）")
    print(f"  推荐首批量: {bass['recommended_first_order']} 件")
    print(f"  理由: {bass['rationale']}")

"""
Skill-Market-Size-Estimation
基于 G-TAB (arXiv:2007.13861, EPFL) +
    Bass+GT 动态市场潜力 (Hu et al., Kent) +
    Monte Carlo 置信区间 (MDPI Applied Sciences 2023)
母婴跨境电商品类 TAM/SAM/SOM 估算工具
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketSizeResult:
    category: str
    tam_low: float
    tam_mid: float
    tam_high: float
    sam_low: float
    sam_mid: float
    sam_high: float
    som_target: float
    monthly_search_low: int
    monthly_search_mid: int
    monthly_search_high: int
    peak_month_estimate: Optional[int]
    top_sensitive_params: list[str]
    confidence_note: str
    decision: str


# ── G-TAB 校准：GT 指数 → 绝对搜索量 ─────────────────────
def calibrate_gt_volume(
    target_gt_index: float,
    anchor_keyword_monthly_volume: int,
    anchor_gt_peak: float = 100.0,
    rounding_error_pct: float = 0.30,
) -> tuple[int, int, int]:
    """
    G-TAB 方法：将 GT 相对指数校准为绝对月搜索量。
    返回 (low, mid, high) 置信区间。

    arXiv:2007.13861, EPFL Data Science Lab
    calibrated_volume = GT_raw × (R_anchor / m_anchor)
    """
    mid = int(target_gt_index / anchor_gt_peak * anchor_keyword_monthly_volume)
    low = int(mid * (1 - rounding_error_pct))
    high = int(mid * (1 + rounding_error_pct))
    return low, mid, high


# ── Bass 扩散：动态渗透率曲线 ─────────────────────────────
def bass_diffusion_curve(
    market_potential: float,
    p: float = 0.03,
    q: float = 0.38,
    periods: int = 36,
) -> np.ndarray:
    """
    Bass 扩散模型月度需求曲线。
    N(t) = M × [p + q×F(t)] × [1-F(t)]
    消费电子经验参数：p=0.03（创新系数），q=0.38（模仿系数）
    母婴耐用品建议：p=0.02, q=0.30（扩散较慢）
    """
    cumulative = np.zeros(periods)
    demand = np.zeros(periods)
    for t in range(periods):
        ft = cumulative[t - 1] / market_potential if t > 0 else 0.0
        demand[t] = market_potential * (p + q * ft) * (1 - ft)
        cumulative[t] = (cumulative[t - 1] if t > 0 else 0) + demand[t]
    return demand


# ── Monte Carlo TAM 置信区间 ──────────────────────────────
def monte_carlo_tam(
    population: int,
    penetration_rate_range: tuple[float, float],
    asp_mean: float,
    asp_std: float,
    annual_purchase_freq: float = 1.0,
    n_simulations: int = 10000,
    rng_seed: int = 42,
) -> dict:
    """
    Monte Carlo 模拟 TAM 分布。
    输入关键假设的范围，输出均值 ± σ 和龙卷风图敏感度。

    基于 MDPI Applied Sciences 2023 Monte Carlo 框架。
    """
    rng = np.random.default_rng(rng_seed)
    lo, hi = penetration_rate_range
    penetration = rng.uniform(lo, hi, n_simulations)
    asp = rng.normal(asp_mean, asp_std, n_simulations)
    asp = np.clip(asp, asp_mean * 0.5, asp_mean * 2.0)
    growth = rng.triangular(0.05, 0.12, 0.25, n_simulations)

    tam_samples = population * penetration * asp * annual_purchase_freq * (1 + growth)
    tam_samples = tam_samples[tam_samples > 0]

    mean_tam = float(np.mean(tam_samples))
    std_tam  = float(np.std(tam_samples))

    sensitivity = {}
    for param_name, param_values in [
        ("渗透率", penetration),
        ("ASP",    asp),
        ("增长率", growth),
    ]:
        corr = float(np.corrcoef(param_values[:len(tam_samples)], tam_samples)[0, 1])
        sensitivity[param_name] = abs(corr)

    top_params = sorted(sensitivity, key=sensitivity.get, reverse=True)

    return {
        "mean":       mean_tam,
        "std":        std_tam,
        "p10":        float(np.percentile(tam_samples, 10)),
        "p50":        float(np.percentile(tam_samples, 50)),
        "p90":        float(np.percentile(tam_samples, 90)),
        "top_sensitive_params": top_params,
        "sensitivity_scores": sensitivity,
    }


# ── Top-down 估算 ─────────────────────────────────────────
def topdown_estimate(
    global_market_usd: float,
    ecommerce_share: float,
    category_share: float,
    target_segment_share: float,
) -> tuple[float, float, float]:
    """
    TAM/SAM/SOM 三层切分。
    Stanford Biodesign 双轨方法论。
    """
    tam = global_market_usd * ecommerce_share
    sam = tam * category_share
    som = sam * target_segment_share
    return tam, sam, som


# ── Bottom-up 估算 ────────────────────────────────────────
def bottomup_estimate(
    competitor_monthly_sales: list[float],
    asp: float,
    market_coverage_multiplier: float = 1.5,
    own_target_monthly_units: float = 100,
) -> tuple[float, float, float]:
    """
    竞品收入加总 → TAM/SAM/SOM。
    market_coverage_multiplier: 已知竞品销量 × 系数 ≈ 总市场
    （系数 1.5 代表 Top-20 竞品覆盖约 67% 市场）
    """
    sam = sum(competitor_monthly_sales) * asp * 12
    tam = sam * market_coverage_multiplier
    som = own_target_monthly_units * asp * 12
    return tam, sam, som


# ── 主函数 ─────────────────────────────────────────────────
def estimate_market_size(
    category: str,
    target_gt_index: float,
    anchor_keyword: str,
    anchor_monthly_volume: int,
    anchor_gt_peak: float,
    global_market_usd: float,
    ecommerce_share: float,
    category_share: float,
    target_segment_share: float,
    competitor_monthly_sales: list[float],
    asp_mean: float,
    asp_std: float,
    own_target_monthly_units: float = 100,
    population: int = 50_000_000,
    penetration_range: tuple[float, float] = (0.005, 0.020),
    bass_p: float = 0.02,
    bass_q: float = 0.30,
) -> MarketSizeResult:
    search_low, search_mid, search_high = calibrate_gt_volume(
        target_gt_index, anchor_monthly_volume, anchor_gt_peak
    )

    td_tam, td_sam, td_som = topdown_estimate(
        global_market_usd, ecommerce_share, category_share, target_segment_share
    )

    bu_tam, bu_sam, bu_som = bottomup_estimate(
        competitor_monthly_sales, asp_mean,
        own_target_monthly_units=own_target_monthly_units,
    )

    mc = monte_carlo_tam(
        population=population,
        penetration_rate_range=penetration_range,
        asp_mean=asp_mean,
        asp_std=asp_std,
    )

    tam_mid = (td_tam + bu_tam) / 2
    sam_mid = (td_sam + bu_sam) / 2
    cross_check_ok = abs(td_sam - bu_sam) / max(bu_sam, 1) < 0.5

    demand_curve = bass_diffusion_curve(market_potential=sam_mid / 12, p=bass_p, q=bass_q)
    peak_month = int(np.argmax(demand_curve)) + 1

    confidence = "双轨误差 < 50%，估算可信" if cross_check_ok else "双轨误差 > 50%，请重检渗透率假设"

    if sam_mid < 10_000_000:
        decision = "SKIP（SAM < $10M，市场太小）"
    elif sam_mid < 50_000_000:
        decision = "CAUTION（SAM $10M-$50M，细分市场，需强差异化）"
    elif sam_mid < 500_000_000:
        decision = "GO（SAM $50M-$500M，合适规模，有空间）"
    else:
        decision = "CAUTION（SAM > $500M，超大市场但竞争激烈，需找细分切入点）"

    return MarketSizeResult(
        category=category,
        tam_low=mc["p10"], tam_mid=tam_mid, tam_high=mc["p90"],
        sam_low=td_sam * 0.7, sam_mid=sam_mid, sam_high=bu_sam * 1.3,
        som_target=bu_som,
        monthly_search_low=search_low, monthly_search_mid=search_mid, monthly_search_high=search_high,
        peak_month_estimate=peak_month,
        top_sensitive_params=mc["top_sensitive_params"],
        confidence_note=confidence,
        decision=decision,
    )


# ── 示例：baby sterilizer 品类 ───────────────────────────
if __name__ == "__main__":
    result = estimate_market_size(
        category="Baby UV-C Sterilizer（Amazon US）",
        target_gt_index=22.0,
        anchor_keyword="baby bottle",
        anchor_monthly_volume=40_500,
        anchor_gt_peak=100.0,
        global_market_usd=4_640_000_000,
        ecommerce_share=0.12,
        category_share=0.08,
        target_segment_share=0.10,
        competitor_monthly_sales=[850, 620, 480, 350, 290, 210, 180, 160, 140, 120,
                                   100, 90, 85, 80, 75, 70, 65, 60, 55, 50],
        asp_mean=139.0,
        asp_std=25.0,
        own_target_monthly_units=120,
        population=50_000_000,
        penetration_range=(0.003, 0.015),
        bass_p=0.02,
        bass_q=0.30,
    )

    def fmt_m(v):
        return f"${v/1e6:.1f}M"

    print("=" * 65)
    print(f"品类市场规模估算：{result.category}")
    print("=" * 65)
    print(f"\n【TAM】 {fmt_m(result.tam_low)} – {fmt_m(result.tam_mid)} – {fmt_m(result.tam_high)}")
    print(f"【SAM】 {fmt_m(result.sam_low)} – {fmt_m(result.sam_mid)} – {fmt_m(result.sam_high)}")
    print(f"【SOM】 {fmt_m(result.som_target)}（自身3年目标）")
    print(f"\n月搜索量（GT校准）: {result.monthly_search_low:,} – "
          f"{result.monthly_search_mid:,} – {result.monthly_search_high:,}")
    print(f"Bass 需求峰值月份: 第 {result.peak_month_estimate} 个月（从进入市场算起）")
    print(f"\n最敏感参数（龙卷风图）:")
    for i, p in enumerate(result.top_sensitive_params, 1):
        print(f"  #{i}: {p}")
    print(f"\n置信度说明: {result.confidence_note}")
    print(f"决策建议: {result.decision}")

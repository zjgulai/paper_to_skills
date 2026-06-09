"""WF-D 选品情报扫描 Skill 工具.

涵盖:
  - 市场空间过滤 (BSR / 月销 / 评论数 / 评分)
  - 竞争度评估 (头部 ASIN 集中度)
  - 毛利可行性 (售价 - FBA - 头程 - 关税)
  - 母婴合规预判 (品类敏感度)
  - 候选品评分排序 (KGQA 加权)
"""

from __future__ import annotations

from typing import Any, Dict, List


COMPLIANCE_HIGH_RISK_KEYWORDS = {"electric", "battery", "lithium", "lead", "paint", "magnet", "small parts"}
COMPLIANCE_NEEDS_CERT = {
    "infant_formula": ["FDA", "EU IFP"],
    "infant_food": ["FDA", "CFIA"],
    "toy": ["EN 71", "CPSC ASTM F963"],
    "bottle": ["FDA 21 CFR", "EU 10/2011"],
    "carseat": ["FMVSS 213", "ECE R44/R129"],
}


def evaluate_market_space(candidate: Dict[str, Any]) -> Dict[str, Any]:
    monthly_sales = float(candidate.get("monthly_sales_usd", 0))
    review_count = int(candidate.get("review_count", 0))
    avg_rating = float(candidate.get("avg_rating", 0))
    bsr_trend = float(candidate.get("bsr_trend_30d", 0))

    pass_sales = monthly_sales >= 50_000
    pass_review = review_count < 500
    pass_rating = avg_rating >= 3.8
    pass_trend = bsr_trend >= 0.20

    score = sum([pass_sales, pass_review, pass_rating, pass_trend]) / 4.0

    return {
        "skill": "growth_new_product_opportunity",
        "monthly_sales_usd": monthly_sales,
        "passes": {
            "sales_50k": pass_sales,
            "low_competition": pass_review,
            "decent_rating": pass_rating,
            "rising_trend": pass_trend,
        },
        "market_score": round(score, 2),
        "confidence": 0.85,
    }


def estimate_gross_margin(candidate: Dict[str, Any]) -> Dict[str, Any]:
    selling_price = float(candidate.get("selling_price_usd", 0))
    cogs = float(candidate.get("cogs_usd", 0))
    fba_fee = float(candidate.get("fba_fee_usd", 0))
    freight = float(candidate.get("freight_usd", 0))
    duty_rate = float(candidate.get("duty_rate", 0.05))

    duty = selling_price * duty_rate
    total_cost = cogs + fba_fee + freight + duty
    gross_profit = selling_price - total_cost
    margin_pct = (gross_profit / selling_price) if selling_price > 0 else 0.0

    pass_margin = margin_pct >= 0.40

    return {
        "skill": "supply_monodense_price_elasticity",
        "selling_price_usd": selling_price,
        "total_cost_usd": round(total_cost, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "margin_pct": round(margin_pct, 3),
        "passes_40pct_target": pass_margin,
        "confidence": 0.9,
    }


def assess_compliance_risk(candidate: Dict[str, Any]) -> Dict[str, Any]:
    category = candidate.get("category", "").lower()
    description = (candidate.get("description", "") or "").lower()

    high_risk_hits = [kw for kw in COMPLIANCE_HIGH_RISK_KEYWORDS if kw in description]
    required_certs = COMPLIANCE_NEEDS_CERT.get(category, [])

    risk_level = "high" if high_risk_hits else ("medium" if required_certs else "low")

    return {
        "skill": "kg_hierarchical_product",
        "category": category,
        "high_risk_keywords": high_risk_hits,
        "required_certifications": required_certs,
        "risk_level": risk_level,
        "confidence": 0.8,
    }


def kgqa_attribute_lookup(candidate: Dict[str, Any]) -> Dict[str, Any]:
    category = candidate.get("category", "").lower()

    bestseller_attrs = {
        "infant_formula": ["organic", "HMO", "DHA", "stage1-0-6m", "European brand"],
        "bottle": ["anti-colic", "silicone nipple", "wide neck", "BPA-free"],
        "carseat": ["ISOFIX", "5-point harness", "rear-facing", "side impact"],
        "stroller": ["one-hand fold", "compact", "lightweight under 6kg", "all-terrain"],
        "diaper": ["overnight", "ultra-thin", "size NB-XL", "organic cotton"],
    }
    suggested = bestseller_attrs.get(category, [])
    desc = (candidate.get("description", "") or "").lower()
    matched = [a for a in suggested if a.lower() in desc]

    return {
        "skill": "kg_kgqa",
        "category": category,
        "bestseller_attributes": suggested,
        "matched_attributes": matched,
        "match_pct": round(len(matched) / max(len(suggested), 1), 2),
        "confidence": 0.78,
    }


def causal_demand_lift_estimate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    bsr_trend = float(candidate.get("bsr_trend_30d", 0))
    seasonality = float(candidate.get("seasonality_factor", 1.0))
    competitor_count = int(candidate.get("competitor_count", 10))

    base_lift = bsr_trend
    seasonal_lift = (seasonality - 1.0)
    competition_drag = max(0, (competitor_count - 5) * 0.02)
    causal_lift = base_lift + seasonal_lift - competition_drag

    return {
        "skill": "causal_uplift_modeling",
        "bsr_trend": bsr_trend,
        "seasonality_factor": seasonality,
        "competition_drag": round(competition_drag, 3),
        "estimated_causal_lift": round(causal_lift, 3),
        "confidence": 0.75,
    }


def score_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    market = evaluate_market_space(candidate)
    margin = estimate_gross_margin(candidate)
    compliance = assess_compliance_risk(candidate)
    kgqa = kgqa_attribute_lookup(candidate)
    causal = causal_demand_lift_estimate(candidate)

    weights = {"market": 0.30, "margin": 0.25, "compliance": 0.15, "kgqa": 0.15, "causal": 0.15}
    compliance_score = 1.0 if compliance["risk_level"] == "low" else (0.6 if compliance["risk_level"] == "medium" else 0.2)
    margin_score = 1.0 if margin["passes_40pct_target"] else max(0, margin["margin_pct"] / 0.4)
    causal_score = max(0, min(1, 0.5 + causal["estimated_causal_lift"]))

    composite = (
        weights["market"] * market["market_score"]
        + weights["margin"] * margin_score
        + weights["compliance"] * compliance_score
        + weights["kgqa"] * kgqa["match_pct"]
        + weights["causal"] * causal_score
    )

    return {
        "candidate_id": candidate.get("id"),
        "name": candidate.get("name"),
        "market": market,
        "margin": margin,
        "compliance": compliance,
        "kgqa": kgqa,
        "causal": causal,
        "composite_score": round(composite, 3),
        "recommend": composite >= 0.65 and compliance["risk_level"] != "high",
    }


def market_signal_collection(product_category: str, competitor_count: int = 5) -> Dict[str, Any]:
    """实时采集竞品价格和市场信号，基于 Skill-Market-Signal-Realtime-Collection.

    当前使用 mock 数据；生产环境替换为真实 API 采集。
    返回: competitor_prices, trending_products, price_alerts
    """
    # mock 竞品价格数据（生产环境接入实时 API）
    mock_prices = [
        {"asin": f"B0MOCK{i:04d}", "title": f"{product_category} Competitor {i}", "price_usd": 20.0 + i * 3.5}
        for i in range(1, competitor_count + 1)
    ]
    avg_competitor_price = sum(p["price_usd"] for p in mock_prices) / len(mock_prices) if mock_prices else 0.0

    # mock 趋势品（BSR 涨幅 top N）
    mock_trending = [
        {"rank": i, "name": f"{product_category} Trending #{i}", "bsr_delta_7d": -(i * 120)}
        for i in range(1, 4)
    ]

    # mock 价格预警（价格低于均值 15% 触发）
    price_threshold = avg_competitor_price * 0.85
    price_alerts = [
        {"asin": p["asin"], "price_usd": p["price_usd"], "alert": "price_undercut"}
        for p in mock_prices
        if p["price_usd"] < price_threshold
    ]

    return {
        "skill": "market_signal_realtime_collection",
        "product_category": product_category,
        "competitor_count": len(mock_prices),
        "competitor_prices": mock_prices,
        "avg_competitor_price_usd": round(avg_competitor_price, 2),
        "trending_products": mock_trending,
        "price_alerts": price_alerts,
        "confidence": 0.82,
    }


def run_full_wfd_analysis(candidates: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, Any]:
    scored = [score_candidate(c) for c in candidates]
    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    top_picks = [s for s in scored if s["recommend"]][:top_n]

    avg_conf = sum(
        (s["market"]["confidence"] + s["margin"]["confidence"] + s["compliance"]["confidence"]
         + s["kgqa"]["confidence"] + s["causal"]["confidence"]) / 5
        for s in scored
    ) / max(len(scored), 1)

    return {
        "total_candidates": len(candidates),
        "scored": scored,
        "top_picks": top_picks,
        "recommend_count": len(top_picks),
        "estimated_cost": 0.0,
        "skill_chain": [
            "growth_new_product_opportunity",
            "supply_monodense_price_elasticity",
            "kg_hierarchical_product",
            "kg_kgqa",
            "causal_uplift_modeling",
            "market_signal_realtime_collection",
        ],
        "confidence": round(avg_conf, 3),
    }

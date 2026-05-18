"""WF-B 广告关键词优化 Skill 工具的母婴业务语义实现.

涵盖:
  - 搜索词报告解析 / 词级 ROAS 计算
  - 低效词识别 + 否定词建议
  - 高价值词提价 (基于 Uplift 思想)
  - DARA 简化版预算分配 (跨渠道 marginal ROAS 平衡)
  - MMM 渠道弹性预估
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def parse_search_term_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"total_terms": 0, "total_spend": 0.0, "total_revenue": 0.0, "overall_tacos": 0.0}
    total_spend = sum(float(r.get("spend", 0)) for r in rows)
    total_revenue = sum(float(r.get("revenue", 0)) for r in rows)
    total_terms = len(rows)
    tacos = (total_spend / total_revenue) if total_revenue > 0 else float("inf")
    return {
        "skill": "ad_search_term_parse",
        "total_terms": total_terms,
        "total_spend": round(total_spend, 2),
        "total_revenue": round(total_revenue, 2),
        "overall_tacos": round(tacos, 4),
    }


def identify_negative_keywords(
    rows: List[Dict[str, Any]],
    click_threshold: int = 10,
    conversion_threshold: float = 0.0,
) -> Dict[str, Any]:
    negatives = []
    for r in rows:
        clicks = int(r.get("clicks", 0))
        orders = int(r.get("orders", 0))
        spend = float(r.get("spend", 0))
        if clicks >= click_threshold and orders <= conversion_threshold:
            negatives.append({
                "search_term": r.get("search_term"),
                "clicks": clicks,
                "spend_wasted": round(spend, 2),
            })
    total_wasted = sum(n["spend_wasted"] for n in negatives)
    return {
        "skill": "ad_negative_keywords",
        "negatives": negatives,
        "negative_count": len(negatives),
        "spend_wasted": round(total_wasted, 2),
        "confidence": 0.9,
    }


def uplift_keyword_promotion(
    rows: List[Dict[str, Any]],
    roas_threshold: float = 4.0,
    min_orders: int = 3,
) -> Dict[str, Any]:
    promotions = []
    for r in rows:
        spend = float(r.get("spend", 0))
        revenue = float(r.get("revenue", 0))
        orders = int(r.get("orders", 0))
        if spend == 0 or orders < min_orders:
            continue
        roas = revenue / spend
        if roas >= roas_threshold:
            current_bid = float(r.get("bid", 1.0))
            suggested_bid = round(min(current_bid * 1.3, current_bid + 1.0), 2)
            promotions.append({
                "search_term": r.get("search_term"),
                "roas": round(roas, 2),
                "orders": orders,
                "current_bid": current_bid,
                "suggested_bid": suggested_bid,
                "uplift_estimate_pct": round((suggested_bid / current_bid - 1) * 100, 1),
            })
    return {
        "skill": "causal_uplift_modeling_ad",
        "promotions": promotions,
        "promote_count": len(promotions),
        "confidence": 0.82,
    }


def dara_cross_channel_budget(
    history: List[Dict[str, Any]],
    total_budget: float,
    iterations: int = 4,
    learning_rate: float = 0.1,
) -> Dict[str, Any]:
    channels = list({h["channel"] for h in history})
    if not channels:
        return {"skill": "marketing_dara_optimizer", "allocation": {}}

    roas_by_ch: Dict[str, List[float]] = {c: [] for c in channels}
    for h in history:
        roas_by_ch[h["channel"]].append(float(h.get("roas", 1.0)))
    avg_roas = {c: (sum(v) / len(v) if v else 1.0) for c, v in roas_by_ch.items()}

    total_roas = sum(avg_roas.values()) or 1.0
    allocation = {c: total_budget * avg_roas[c] / total_roas for c in channels}

    saturation = 1000.0
    for _ in range(iterations):
        marginal = {c: avg_roas[c] * saturation / (saturation + allocation[c]) for c in channels}
        mean_m = sum(marginal.values()) / len(marginal)
        new_alloc = {}
        for c in channels:
            delta = (marginal[c] - mean_m) * learning_rate * allocation[c]
            new_alloc[c] = max(allocation[c] + delta, 100.0)
        total = sum(new_alloc.values())
        allocation = {c: v * total_budget / total for c, v in new_alloc.items()}

    return {
        "skill": "marketing_dara_optimizer",
        "allocation": {c: round(v, 2) for c, v in allocation.items()},
        "avg_roas_input": {c: round(v, 2) for c, v in avg_roas.items()},
        "iterations": iterations,
        "total_budget": total_budget,
        "confidence": 0.8,
    }


def mmm_channel_elasticity(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_ch: Dict[str, List[Dict[str, float]]] = {}
    for h in history:
        by_ch.setdefault(h["channel"], []).append({
            "spend": float(h.get("spend", 0)),
            "revenue": float(h.get("revenue", 0)),
        })

    elasticity: Dict[str, float] = {}
    for ch, points in by_ch.items():
        if len(points) < 2:
            elasticity[ch] = 1.0
            continue
        log_spend = [math.log(p["spend"] + 1) for p in points]
        log_rev = [math.log(p["revenue"] + 1) for p in points]
        mean_x = sum(log_spend) / len(log_spend)
        mean_y = sum(log_rev) / len(log_rev)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_spend, log_rev))
        den = sum((x - mean_x) ** 2 for x in log_spend) + 1e-9
        elasticity[ch] = round(num / den, 3)

    return {
        "skill": "marketing_mmm",
        "elasticity": elasticity,
        "channels": list(by_ch.keys()),
        "confidence": 0.7,
    }


def run_full_wfb_analysis(
    search_term_rows: List[Dict[str, Any]],
    channel_history: List[Dict[str, Any]],
    total_budget: float,
    target_tacos: float = 0.15,
    cost_per_bid_change: float = 0.0,
) -> Dict[str, Any]:
    summary = parse_search_term_report(search_term_rows)
    negatives = identify_negative_keywords(search_term_rows)
    promotions = uplift_keyword_promotion(search_term_rows)
    mmm = mmm_channel_elasticity(channel_history)
    dara = dara_cross_channel_budget(channel_history, total_budget)

    overall_tacos = summary["overall_tacos"]
    tacos_status = "healthy" if overall_tacos <= target_tacos else "over_target"

    estimated_save = negatives["spend_wasted"]
    estimated_uplift = sum(
        max(0, (p["suggested_bid"] - p["current_bid"]) * 10)
        for p in promotions["promotions"]
    )

    overall_confidence = (
        negatives["confidence"] + promotions["confidence"] + mmm["confidence"] + dara["confidence"]
    ) / 4

    return {
        "summary": summary,
        "negatives": negatives,
        "promotions": promotions,
        "mmm": mmm,
        "dara": dara,
        "tacos_status": tacos_status,
        "target_tacos": target_tacos,
        "estimated_save_rmb": estimated_save,
        "estimated_uplift_rmb": estimated_uplift,
        "estimated_cost": round(estimated_save + estimated_uplift, 2),
        "skill_chain": [
            "ad_search_term_parse",
            "ad_negative_keywords",
            "causal_uplift_modeling_ad",
            "marketing_mmm",
            "marketing_dara_optimizer",
        ],
        "confidence": round(overall_confidence, 3),
    }

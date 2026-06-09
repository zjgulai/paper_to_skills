"""WF-A 供应链 Skill 工具的母婴业务语义实现.

每个 Tool 是 SkillRegistry 中 stub 的"smart"版本,
基于母婴跨境电商的真实业务公式,可被 SupplyChainAgent 直接调用.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SkuDemandProfile:
    sku_id: str
    avg_daily_sales: float
    sales_std: float
    current_stock: int
    in_transit: int
    lead_time_days: float
    is_seasonal: bool = False
    season_multiplier: float = 1.0


def demand_forecast(
    sku_id: str,
    history_daily_sales: List[float],
    forecast_horizon_days: int = 90,
    season_multiplier: float = 1.0,
) -> Dict[str, Any]:
    if not history_daily_sales:
        return {"sku_id": sku_id, "forecast_total": 0, "daily_mean": 0.0, "confidence": 0.0}

    daily_mean = sum(history_daily_sales) / len(history_daily_sales)
    daily_var = sum((x - daily_mean) ** 2 for x in history_daily_sales) / max(len(history_daily_sales) - 1, 1)
    daily_std = math.sqrt(daily_var)

    forecast_total = daily_mean * forecast_horizon_days * season_multiplier
    confidence = max(0.5, 1.0 - daily_std / (daily_mean + 1e-6))

    return {
        "skill": "supply_demand_forecast",
        "sku_id": sku_id,
        "forecast_total": round(forecast_total, 1),
        "daily_mean": round(daily_mean, 2),
        "daily_std": round(daily_std, 2),
        "horizon_days": forecast_horizon_days,
        "season_multiplier": season_multiplier,
        "confidence": round(confidence, 3),
    }


def causal_counterfactual_forecast(
    sku_id: str,
    history_daily_sales: List[float],
    interventions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    base = demand_forecast(sku_id, history_daily_sales, forecast_horizon_days=90)
    daily_mean = base["daily_mean"]

    intervention_lift = 0.0
    for it in interventions:
        if it.get("type") == "promotion":
            intervention_lift += float(it.get("expected_lift", 0.3))
        elif it.get("type") == "stockout":
            intervention_lift -= float(it.get("severity", 0.5))

    counterfactual_daily = daily_mean * (1 + intervention_lift)
    return {
        "skill": "ts_causal_gcf",
        "sku_id": sku_id,
        "base_daily_mean": daily_mean,
        "counterfactual_daily_mean": round(counterfactual_daily, 2),
        "intervention_lift_pct": round(intervention_lift * 100, 1),
        "interventions_count": len(interventions),
        "confidence": 0.78,
    }


def safety_stock_reorder_point(
    sku_id: str,
    daily_mean: float,
    daily_std: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> Dict[str, Any]:
    z = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.99: 2.33}.get(service_level, 1.65)
    safety_stock = z * daily_std * math.sqrt(lead_time_days)
    reorder_point = daily_mean * lead_time_days + safety_stock

    return {
        "skill": "supply_safety_stock_replenishment",
        "sku_id": sku_id,
        "safety_stock": round(safety_stock, 0),
        "reorder_point": round(reorder_point, 0),
        "lead_time_days": lead_time_days,
        "service_level": service_level,
        "z_score": z,
        "confidence": 0.92,
    }


def two_echelon_drl_recommendation(
    sku_id: str,
    forecast_demand_total: float,
    current_stock: int,
    in_transit: int,
    safety_stock: float,
    moq: int = 500,
    unit_cost_rmb: float = 50.0,
) -> Dict[str, Any]:
    desired_stock = forecast_demand_total + safety_stock
    gap = desired_stock - (current_stock + in_transit)

    if gap <= 0:
        recommendation = "no_action"
        recommended_qty = 0
    else:
        recommended_qty = max(moq, int(math.ceil(gap / moq) * moq))
        recommendation = "place_purchase_order"

    estimated_cost = recommended_qty * unit_cost_rmb

    return {
        "skill": "supply_two_echelon_drl",
        "sku_id": sku_id,
        "recommendation": recommendation,
        "recommended_qty": recommended_qty,
        "moq": moq,
        "unit_cost_rmb": unit_cost_rmb,
        "estimated_cost": round(estimated_cost, 0),
        "stock_gap": round(gap, 0),
        "desired_stock": round(desired_stock, 0),
        "confidence": 0.85,
    }


def anomaly_detection(
    sku_id: str,
    recent_daily_sales: List[float],
    baseline_mean: float,
    baseline_std: float,
    z_threshold: float = 2.5,
) -> Dict[str, Any]:
    anomalies = []
    for i, v in enumerate(recent_daily_sales):
        z = (v - baseline_mean) / (baseline_std + 1e-6)
        if abs(z) > z_threshold:
            anomalies.append({"day_index": i, "value": v, "z_score": round(z, 2)})

    return {
        "skill": "ts_anomaly_detection",
        "sku_id": sku_id,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "alert": len(anomalies) > 0,
        "confidence": 0.88,
    }


def run_full_wfa_analysis(
    sku_id: str,
    history_daily_sales: List[float],
    current_stock: int,
    in_transit: int,
    lead_time_days: float,
    season_multiplier: float = 1.0,
    interventions: List[Dict[str, Any]] | None = None,
    moq: int = 500,
    unit_cost_rmb: float = 50.0,
    service_level: float = 0.95,
) -> Dict[str, Any]:
    interventions = interventions or []

    forecast = demand_forecast(sku_id, history_daily_sales, forecast_horizon_days=90, season_multiplier=season_multiplier)

    counterfactual = causal_counterfactual_forecast(sku_id, history_daily_sales, interventions)

    safety = safety_stock_reorder_point(
        sku_id,
        daily_mean=forecast["daily_mean"],
        daily_std=forecast["daily_std"],
        lead_time_days=lead_time_days,
        service_level=service_level,
    )

    recommendation = two_echelon_drl_recommendation(
        sku_id,
        forecast_demand_total=forecast["forecast_total"],
        current_stock=current_stock,
        in_transit=in_transit,
        safety_stock=safety["safety_stock"],
        moq=moq,
        unit_cost_rmb=unit_cost_rmb,
    )

    avg_confidence = (forecast["confidence"] + counterfactual["confidence"]
                      + safety["confidence"] + recommendation["confidence"]) / 4

    return {
        "sku_id": sku_id,
        "forecast": forecast,
        "counterfactual": counterfactual,
        "safety": safety,
        "recommendation": recommendation,
        "estimated_cost": recommendation["estimated_cost"],
        "skill_chain": [
            "supply_demand_forecast",
            "ts_causal_gcf",
            "supply_safety_stock_replenishment",
            "supply_two_echelon_drl",
        ],
        "confidence": round(avg_confidence, 3),
    }

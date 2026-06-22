---
title: Promo-Inventory-Pulse-Auto-Trigger — 大促前14天库存不足自动触发紧急补货与广告降速保护
doc_type: knowledge
module: 04-供应链
topic: promo-inventory-pulse-auto-trigger
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Promo-Inventory-Pulse-Auto-Trigger

> **配对分析层**：[[Skill-Promo-Demand-Forecasting]]
> **决策类型**: 自动触发型 | **触发条件**: 大促开始前14天库存低于预测需求120% | **执行动作**: 自动触发紧急补货采购单 + 广告降速保护库存

## ① 算法原理

核心是「大促需求预测 + 库存缺口检测 + 双轨响应（补货+广告降速）」：

1. **大促需求预测**：基于历史大促倍率（当年 vs 平日销量倍数）预测大促期间总需求。
2. **安全库存计算**：目标库存 = 预测需求 × 安全系数（默认 1.2，即留 20% 安全余量）。
3. **缺口检测**（触发条件 = T-14天）：
   - 当前库存 + 在途库存 < 目标库存 → 触发补货
   - 库存覆盖率 < 70%（严重不足）→ 同时触发广告降速
4. **双轨响应**：
   - **补货通道**：计算补货量（目标 - 当前 - 在途），生成紧急采购单
   - **广告降速**：若库存覆盖率 < 70%，将广告出价下调 40%，减少订单流入速度，延长库存时长
5. **广告恢复**：库存到货后（在途预计到达日内），自动恢复广告出价至原始水平。

## ② 母婴出海应用案例

**场景：Prime Day 前婴儿推车备货**
- 触发时间：Prime Day 开始前 14 天（T=7月2日）
- 状态：当前库存 180 件，在途 50 件（预计 T+5 到货），历史 Prime Day 倍率 4.2x，日均 28 件
- 目标库存：28 × 4.2 × 3天活动 × 1.2安全系数 ≈ 424 件；实际可用 230 件，覆盖率 54%
- 执行：
  - 触发紧急补货 194 件（空运，成本 $2,300）
  - 广告出价下调 40%（$2.20 → $1.32），减少日均订单 35%，保护库存 7 天
  - 补货到达后恢复广告出价
- 业务价值：避免大促期间断货损失 $68,000，广告降速保护期间节省广告费 $1,800

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime, timedelta

def promo_inventory_pulse_auto_trigger(
    skus: List[Dict],
    now: Optional[datetime] = None,
    trigger_days_before: int = 14,
    safety_factor: float = 1.2,
    ad_slowdown_threshold: float = 0.70,
    ad_slowdown_pct: float = 0.40
) -> Dict:
    """
    大促前库存脉冲自动触发器
    
    参数:
        skus: [{
            "sku_id": str, "current_stock": int,
            "in_transit_stock": int, "in_transit_eta_days": int,
            "avg_daily_sales": float,
            "promo_multiplier": float,  # 历史大促倍率
            "promo_duration_days": int, # 大促持续天数
            "promo_start_date": str,    # ISO8601 大促开始日
            "current_bid": float        # 当前广告出价 ($)
        }]
        trigger_days_before: 大促前多少天检测（默认14天）
        safety_factor: 安全库存系数（默认1.2）
        ad_slowdown_threshold: 触发广告降速的覆盖率阈值（默认70%）
        ad_slowdown_pct: 广告降速比例（默认40%）
    
    返回:
        {"triggers": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    triggers = []
    
    for sku in skus:
        sid = sku["sku_id"]
        current_stock = sku["current_stock"]
        in_transit = sku.get("in_transit_stock", 0)
        in_transit_eta = sku.get("in_transit_eta_days", 0)
        daily_sales = max(sku.get("avg_daily_sales", 1), 0.1)
        promo_mult = sku.get("promo_multiplier", 3.0)
        promo_days = sku.get("promo_duration_days", 3)
        promo_start = datetime.fromisoformat(sku["promo_start_date"])
        current_bid = sku.get("current_bid", 1.5)
        
        days_to_promo = (promo_start - now).days
        
        # 检查是否在触发窗口内
        if days_to_promo > trigger_days_before or days_to_promo < 0:
            triggers.append({
                "sku_id": sid,
                "action": "NOT_IN_WINDOW",
                "days_to_promo": days_to_promo,
                "reason": f"距大促{days_to_promo}天，触发窗口为{trigger_days_before}天内"
            })
            continue
        
        # 计算目标库存
        promo_demand = daily_sales * promo_mult * promo_days
        target_stock = promo_demand * safety_factor
        
        # 可用库存（当前 + 在途，若在途能在大促前到达）
        available = current_stock + (in_transit if in_transit_eta <= days_to_promo else 0)
        coverage_rate = available / max(target_stock, 1)
        shortage = max(target_stock - available, 0)
        
        if coverage_rate >= 1.0:
            triggers.append({
                "sku_id": sid,
                "action": "SUFFICIENT",
                "coverage_rate": round(coverage_rate, 2),
                "available_stock": available,
                "target_stock": round(target_stock),
                "reason": "库存充足"
            })
            continue
        
        # 补货触发
        replenish_action = {
            "type": "EMERGENCY_REPLENISHMENT",
            "quantity": int(shortage) + 20,  # 额外 20 件缓冲
            "mode": "airfreight" if days_to_promo <= 10 else "expedited_sea",
            "estimated_arrival_days": 5 if days_to_promo <= 10 else 12,
            "note": f"需在大促前{days_to_promo}天到货"
        }
        
        # 广告降速触发
        ad_action = None
        if coverage_rate < ad_slowdown_threshold:
            new_bid = round(current_bid * (1 - ad_slowdown_pct), 2)
            recovery_days = in_transit_eta if in_transit > 0 else 10
            ad_action = {
                "type": "AD_BID_SLOWDOWN",
                "current_bid": current_bid,
                "adjusted_bid": new_bid,
                "slowdown_pct": ad_slowdown_pct,
                "reason": f"库存覆盖率{coverage_rate:.0%}<{ad_slowdown_threshold:.0%}，降速保护库存",
                "recovery_trigger": f"补货到达后（约{recovery_days}天）自动恢复出价至${current_bid}"
            }
        
        trigger = {
            "sku_id": sid,
            "action": "PROMO_INVENTORY_ALERT",
            "days_to_promo": days_to_promo,
            "current_stock": current_stock,
            "in_transit_stock": in_transit,
            "available_stock": available,
            "target_stock": round(target_stock),
            "shortage": int(shortage),
            "coverage_rate": round(coverage_rate, 2),
            "promo_demand_est": round(promo_demand),
            "replenish_action": replenish_action,
            "ad_action": ad_action,
            "severity": "CRITICAL" if coverage_rate < 0.5 else "HIGH"
        }
        triggers.append(trigger)
    
    alerted = [t for t in triggers if t.get("action") == "PROMO_INVENTORY_ALERT"]
    return {
        "total_skus": len(skus),
        "alerted": len(alerted),
        "critical_count": sum(1 for t in alerted if t.get("severity") == "CRITICAL"),
        "ad_slowdown_count": sum(1 for t in alerted if t.get("ad_action") is not None),
        "triggers": triggers,
        "total_shortage_units": sum(t.get("shortage", 0) for t in alerted)
    }


# 测试
now = datetime(2026, 7, 2, 10, 0, 0)
skus = [
    {
        "sku_id": "STROLLER-001",
        "current_stock": 180, "in_transit_stock": 50, "in_transit_eta_days": 5,
        "avg_daily_sales": 28.0, "promo_multiplier": 4.2, "promo_duration_days": 3,
        "promo_start_date": "2026-07-16T00:00:00",
        "current_bid": 2.20
    },
    {
        "sku_id": "BOTTLE-002",
        "current_stock": 800, "in_transit_stock": 200, "in_transit_eta_days": 3,
        "avg_daily_sales": 50.0, "promo_multiplier": 3.0, "promo_duration_days": 3,
        "promo_start_date": "2026-07-16T00:00:00",
        "current_bid": 1.50
    },
    {
        "sku_id": "WIPES-003",
        "current_stock": 5000, "in_transit_stock": 3000, "in_transit_eta_days": 4,
        "avg_daily_sales": 100.0, "promo_multiplier": 5.0, "promo_duration_days": 3,
        "promo_start_date": "2026-07-16T00:00:00",
        "current_bid": 0.80
    },
]

result = promo_inventory_pulse_auto_trigger(skus, now=now)

assert result["total_skus"] == 3
# STROLLER 应触发（库存不足）
stroller = next(t for t in result["triggers"] if t["sku_id"] == "STROLLER-001")
assert stroller["action"] == "PROMO_INVENTORY_ALERT"
assert stroller["ad_action"] is not None  # 覆盖率低，触发广告降速
assert stroller["replenish_action"]["mode"] == "airfreight"  # 14天内空运

print("[✓] Promo Inventory Pulse Auto Trigger 测试通过")
print(f"  总SKU: {result['total_skus']}，告警: {result['alerted']}，广告降速: {result['ad_slowdown_count']}")
print(f"  总缺口: {result['total_shortage_units']} 件")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Promo-Demand-Forecasting]]（提供大促需求倍率预测）
- **延伸（extends）**：[[Skill-OOS-Emergency-Airfreight-Gate]]（更极端缺货场景的门控升级）
- **可组合（combinable）**：[[Skill-ROAS-Below-Target-Budget-Freeze]]（广告库存联动双重保护）

## ⑤ 商业价值评估
- **ROI量化**：大促断货避免损失 $68,000/次，广告降速节省无效投放 $1,800，空运成本 $2,300，净收益 $67,500
- **实施难度**：⭐⭐⭐☆☆（需大促日历配置 + 库存 API + 广告平台 API 双向接入）
- **优先级**：⭐⭐⭐⭐⭐（大促是全年最高 GMV 窗口，库存断货是不可逆的机会损失）

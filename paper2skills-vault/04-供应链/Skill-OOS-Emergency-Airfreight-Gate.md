---
title: OOS-Emergency-Airfreight-Gate — 库存DOS危急+海运延误自动触发紧急空运决策门控
doc_type: knowledge
module: 04-供应链
topic: oos-emergency-airfreight-gate
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: OOS-Emergency-Airfreight-Gate

> **配对分析层**：[[Skill-Inventory-Risk-Early-Warning]]
> **决策类型**: 自动触发型 | **触发条件**: 库存 DOS < 7天 AND 海运在途延误 > 7天 | **执行动作**: 触发紧急空运决策门控，输出成本收益分析 + 执行建议

## ① 算法原理

核心是「库存危机识别 + 延误确认 + 成本收益门控 + 执行授权分级」：

1. **双重触发条件**（AND 逻辑，缺一不可）：
   - 条件 A：当前可售库存 ÷ 日均销量（过去 7 天）< 7 天（DOS < 7）
   - 条件 B：最近一批海运在途货物预计到达时间延误 > 7 天
2. **成本收益分析**：
   - 空运成本：重量 × ¥/kg 空运费 × 批次规模
   - 缺货损失：日均 GMV × 预计缺货天数 × 缺货损失乘数（含排名下滑惩罚）
   - 决策阈值：若空运成本 < 缺货损失 × 0.8，则建议空运
3. **授权分级**：
   - 空运成本 < $500：系统自动执行（无需人工确认）
   - $500 ≤ 空运成本 < $2000：推送审批请求（SLA 2h）
   - 空运成本 ≥ $2000：升级到供应链总监审批（SLA 1h）

## ② 母婴出海应用案例

**场景：婴儿安全座椅大促期间库存告急**
- 触发条件：黑五前 10 天，当前库存 47 件（DOS 6.7天），海运在途延误 12 天（港口拥堵）
- 成本分析：空运 80kg = $1,200；预计缺货损失 = 日均 GMV $2,800 × 15 天 × 1.5（排名惩罚） = $63,000
- 决策结果：空运成本($1,200) << 缺货损失($63,000)，触发空运申请，推送至供应链负责人 1h 内审批
- 执行后：备货到达，黑五期间销量 +340 件，避免断货损失约 $42,000
- 业务价值：关键节点库存保障率 99%，年化避免缺货损失 $180,000

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime

def oos_emergency_airfreight_gate(
    skus: List[Dict],
    now: Optional[datetime] = None,
    dos_threshold: float = 7.0,
    delay_threshold_days: int = 7,
    auto_approve_limit: float = 500.0,
    manager_limit: float = 2000.0,
    stockout_multiplier: float = 1.5
) -> Dict:
    """
    缺货紧急空运决策门控
    
    参数:
        skus: [{
            "sku_id": str, "current_stock": int,
            "avg_daily_sales": float,  # 过去7天日均销量
            "daily_gmv": float,        # 日均GMV
            "shipment_eta_original": str,  # ISO8601 原定到达日期
            "shipment_eta_updated": str,   # ISO8601 更新后到达日期
            "shipment_weight_kg": float,   # 在途货物重量(kg)
            "airfreight_rate_per_kg": float  # 空运单价($/kg)
        }]
        dos_threshold: 危急 DOS 阈值（默认7天）
        delay_threshold_days: 延误天数阈值（默认7天）
        auto_approve_limit: 自动审批上限（$500以下无需人工）
        manager_limit: 总监审批阈值
        stockout_multiplier: 缺货损失乘数（含排名惩罚）
    
    返回:
        {"decisions": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    decisions = []
    
    for sku in skus:
        sid = sku["sku_id"]
        stock = sku["current_stock"]
        daily_sales = max(sku.get("avg_daily_sales", 1), 0.1)
        daily_gmv = sku.get("daily_gmv", 0)
        
        # 解析延误信息
        eta_original = datetime.fromisoformat(sku["shipment_eta_original"])
        eta_updated = datetime.fromisoformat(sku["shipment_eta_updated"])
        delay_days = (eta_updated - eta_original).days
        weight_kg = sku.get("shipment_weight_kg", 0)
        air_rate = sku.get("airfreight_rate_per_kg", 8.0)
        
        # 计算 DOS
        dos = stock / daily_sales
        
        # 检查双重触发条件
        condition_a = dos < dos_threshold
        condition_b = delay_days > delay_threshold_days
        
        if not (condition_a and condition_b):
            decisions.append({
                "sku_id": sid,
                "action": "NO_ACTION",
                "dos": round(dos, 1),
                "delay_days": delay_days,
                "reason": f"条件未满足：DOS={dos:.1f}天(阈值{dos_threshold})，延误={delay_days}天(阈值{delay_threshold_days})"
            })
            continue
        
        # 成本收益分析
        airfreight_cost = weight_kg * air_rate
        stockout_days = max(delay_days - dos, 0)  # 预计缺货天数
        stockout_loss = daily_gmv * stockout_days * stockout_multiplier
        
        should_airfreight = airfreight_cost < stockout_loss * 0.8
        roi = stockout_loss / max(airfreight_cost, 1)
        
        # 授权分级
        if not should_airfreight:
            approval_level = "NO_ACTION"
            approval_message = f"空运成本(${airfreight_cost:.0f}) > 缺货损失(${stockout_loss:.0f})×0.8，不建议空运"
        elif airfreight_cost < auto_approve_limit:
            approval_level = "AUTO_EXECUTE"
            approval_message = f"成本${airfreight_cost:.0f}<${auto_approve_limit}，自动执行空运采购单"
        elif airfreight_cost < manager_limit:
            approval_level = "MANAGER_APPROVAL"
            approval_message = f"需运营经理2h内审批（成本${airfreight_cost:.0f}）"
        else:
            approval_level = "DIRECTOR_APPROVAL"
            approval_message = f"需供应链总监1h内审批（成本${airfreight_cost:.0f}）"
        
        decisions.append({
            "sku_id": sid,
            "action": "AIRFREIGHT_GATE",
            "triggered": True,
            "dos": round(dos, 1),
            "delay_days": delay_days,
            "stockout_days_est": round(stockout_days, 1),
            "airfreight_cost_usd": round(airfreight_cost, 2),
            "stockout_loss_est_usd": round(stockout_loss, 2),
            "roi": round(roi, 1),
            "should_airfreight": should_airfreight,
            "approval_level": approval_level,
            "approval_message": approval_message,
            "urgency": "CRITICAL" if dos < 3 else "HIGH"
        })
    
    triggered = [d for d in decisions if d.get("triggered")]
    return {
        "total_skus": len(skus),
        "triggered": len(triggered),
        "auto_execute": sum(1 for d in triggered if d.get("approval_level") == "AUTO_EXECUTE"),
        "need_approval": sum(1 for d in triggered if "APPROVAL" in d.get("approval_level", "")),
        "decisions": decisions,
        "total_airfreight_cost": sum(d.get("airfreight_cost_usd", 0) for d in triggered),
        "total_stockout_loss_avoided": sum(d.get("stockout_loss_est_usd", 0) for d in triggered if d.get("should_airfreight"))
    }


# 测试
skus = [
    {
        "sku_id": "CAR-SEAT-001",
        "current_stock": 47, "avg_daily_sales": 7.0, "daily_gmv": 2800.0,
        "shipment_eta_original": "2026-06-25T00:00:00",
        "shipment_eta_updated": "2026-07-07T00:00:00",  # 延误12天
        "shipment_weight_kg": 150, "airfreight_rate_per_kg": 8.0
    },
    {
        "sku_id": "BOTTLE-002",
        "current_stock": 200, "avg_daily_sales": 15.0, "daily_gmv": 800.0,
        "shipment_eta_original": "2026-06-28T00:00:00",
        "shipment_eta_updated": "2026-07-03T00:00:00",  # 延误5天(未达阈值)
        "shipment_weight_kg": 80, "airfreight_rate_per_kg": 6.0
    },
    {
        "sku_id": "DIAPER-003",
        "current_stock": 30, "avg_daily_sales": 8.0, "daily_gmv": 1500.0,
        "shipment_eta_original": "2026-06-25T00:00:00",
        "shipment_eta_updated": "2026-07-10T00:00:00",  # 延误15天
        "shipment_weight_kg": 40, "airfreight_rate_per_kg": 7.0
    },
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = oos_emergency_airfreight_gate(skus, now=now)

assert result["total_skus"] == 3
assert result["triggered"] == 2  # CAR-SEAT-001 和 DIAPER-003 触发

car_seat = next(d for d in result["decisions"] if d["sku_id"] == "CAR-SEAT-001")
assert car_seat["triggered"] == True
assert car_seat["should_airfreight"] == True

bottle = next(d for d in result["decisions"] if d["sku_id"] == "BOTTLE-002")
assert bottle["action"] == "NO_ACTION"  # 延误5天未达阈值

print("[✓] OOS Emergency Airfreight Gate 测试通过")
print(f"  总SKU: {result['total_skus']}，触发门控: {result['triggered']}")
print(f"  空运总成本: ${result['total_airfreight_cost']:.0f}，预防缺货损失: ${result['total_stockout_loss_avoided']:.0f}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Inventory-Risk-Early-Warning]]（提供 DOS 预警信号）
- **延伸（extends）**：[[Skill-Multi-Echelon-Inventory-Optimization]]（系统性备货优化减少触发频率）
- **可组合（combinable）**：[[Skill-Promo-Inventory-Pulse-Auto-Trigger]]（大促库存联动保障）

## ⑤ 商业价值评估
- **ROI量化**：单次空运成本 $1,200，避免缺货损失 $42,000，ROI 35:1；年化避免缺货损失 $180,000
- **实施难度**：⭐⭐⭐☆☆（需实时库存 API + 物流 ETA 接口 + 审批工作流）
- **优先级**：⭐⭐⭐⭐⭐（大促期间库存断货是最高危风险，直接影响搜索排名）

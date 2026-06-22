---
title: Markdown Clearance Auto Trigger — 库龄超标且库存积压时自动触发降价清仓阶梯
doc_type: knowledge
module: 04-供应链
topic: markdown-clearance-auto-trigger
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Markdown Clearance Auto Trigger

> **配对分析层**：[[Skill-Markdown-Optimization]]
> **决策类型**: 自动触发型 | **触发条件**: 库龄>45天 AND 库存>目标库存×1.5 | **执行动作**: 按预设降价阶梯（-10%/-20%/-30%）自动触发清仓

## ① 算法原理

核心是「库龄分级 + 库存超比检测 + 降价阶梯策略 + 收益预测」：

1. **双条件门控**：仅当「库龄>45天」AND「当前库存>目标库存×1.5」同时满足时触发，避免单条件误触发（高库存但新品上架，或库龄长但库存正常）。
   
2. **库龄分级降价阶梯**：
   - 45-60天：-10%（轻度清仓，试探需求弹性）
   - 61-90天：-20%（中度清仓，加快周转）
   - >90天：-30%（深度清仓，止损为主）
   
3. **收益预测**：结合历史价格弹性估算各降价档位的预期销量增幅和毛利变化，选择「毛利最大化」或「速度最大化」两种策略模式。

4. **渐进执行**：不直接跳到最高降幅，每隔7天评估一次库存消化速度，若消化率<目标再提升一个降价档位。

**误触发防护**：新品上市90天内豁免触发（成长期允许高库存）；FBA政策期（Prime Day前）豁免触发。**回滚机制**：若降价后7天销量未提升（同比提升<5%），暂停降价评估库存损坏/季节性因素。

## ② 母婴出海应用案例

**场景：婴儿冬季睡袋库存积压清仓**
- 触发条件：睡袋库龄68天（超45天阈值），当前库存2,400件，目标库存1,200件（2,400/1,200=2.0>1.5），双条件满足
- 执行动作：库龄68天→匹配-20%降价档，原价$45.99→促销价$36.79，预计7天销量提升65%，消化约600件
- 安全护栏：毛利率降至22%（预警线25%），自动通知采购团队评估是否继续；降价后7天若消化率<30%则升级至-30%档
- 业务价值：避免库存报废损失约$15,000，提前回笼资金$22,000，年化库存周转率提升1.2次

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime, date
import numpy as np

def markdown_clearance_auto_trigger(
    skus: List[Dict],
    today: Optional[date] = None,
    age_threshold_days: int = 45,
    inventory_ratio_threshold: float = 1.5,
    markdown_tiers: Optional[List[Dict]] = None,
    new_product_grace_days: int = 90,
    min_margin_rate: float = 0.20,
    strategy: str = "margin_max"
) -> Dict:
    """
    库存清仓自动触发器
    
    参数:
        skus: [{
            "sku_id": str, "current_inventory": int, "target_inventory": int,
            "inventory_age_days": int, "cost_price": float, "current_price": float,
            "launch_date": str (YYYY-MM-DD), "price_elasticity": float (可选)
        }]
        age_threshold_days: 库龄触发阈值（默认45天）
        inventory_ratio_threshold: 库存超比阈值（默认1.5x）
        markdown_tiers: 自定义降价阶梯，默认 [{"age": 45, "discount": 0.10}, ...]
        new_product_grace_days: 新品豁免期（天）
        min_margin_rate: 最低毛利率保护线
        strategy: "margin_max"毛利最大化 或 "speed_max"速度最大化
    
    返回:
        各SKU的清仓决策
    """
    if today is None:
        today = date.today()
    
    if markdown_tiers is None:
        markdown_tiers = [
            {"min_age": 45,  "max_age": 60,  "discount": 0.10},
            {"min_age": 61,  "max_age": 90,  "discount": 0.20},
            {"min_age": 91,  "max_age": 999, "discount": 0.30},
        ]
    
    decisions = []
    
    for sku in skus:
        sku_id = sku["sku_id"]
        curr_inv = sku["current_inventory"]
        target_inv = sku["target_inventory"]
        age_days = sku["inventory_age_days"]
        cost = sku["cost_price"]
        price = sku["current_price"]
        launch_date_str = sku.get("launch_date", "2000-01-01")
        elasticity = sku.get("price_elasticity", -1.5)  # 默认弹性
        
        # 新品豁免期检测
        launch_date = datetime.strptime(launch_date_str, "%Y-%m-%d").date()
        days_since_launch = (today - launch_date).days
        if days_since_launch < new_product_grace_days:
            decisions.append({
                "sku_id": sku_id, "trigger": False,
                "reason": f"新品豁免期（上线{days_since_launch}天<{new_product_grace_days}天）"
            })
            continue
        
        # 双条件门控
        inventory_ratio = curr_inv / max(target_inv, 1)
        age_triggered = age_days > age_threshold_days
        inventory_triggered = inventory_ratio > inventory_ratio_threshold
        
        if not (age_triggered and inventory_triggered):
            decisions.append({
                "sku_id": sku_id, "trigger": False,
                "reason": f"条件未满足: 库龄{age_days}天{'✓' if age_triggered else '✗'}, "
                          f"库存比{inventory_ratio:.1f}x{'✓' if inventory_triggered else '✗'}"
            })
            continue
        
        # 确定降价档位
        applicable_tiers = [t for t in markdown_tiers if t["min_age"] <= age_days <= t["max_age"]]
        if not applicable_tiers:
            decisions.append({"sku_id": sku_id, "trigger": False, "reason": "未找到匹配的降价档位"})
            continue
        
        tier = applicable_tiers[0]
        base_discount = tier["discount"]
        
        # 策略调整
        if strategy == "speed_max":
            # 速度最大化：选最高可用档
            max_tier = max(markdown_tiers, key=lambda t: t["discount"])
            discount = max_tier["discount"] if age_days >= max_tier["min_age"] else base_discount
        else:
            discount = base_discount
        
        new_price = round(price * (1 - discount), 2)
        margin_rate = (new_price - cost) / new_price if new_price > 0 else 0
        
        # 毛利保护线检测
        if margin_rate < min_margin_rate:
            # 调整到最低毛利率对应的价格
            new_price = round(cost / (1 - min_margin_rate), 2)
            discount = (price - new_price) / price
            margin_rate = min_margin_rate
            margin_warning = True
        else:
            margin_warning = False
        
        # 收益预测（基于弹性）
        demand_lift = abs(elasticity) * discount  # 简化估算
        expected_units_sold_7d = int(curr_inv * min(demand_lift, 0.5))  # 最多50%
        expected_revenue_7d = expected_units_sold_7d * new_price
        
        decisions.append({
            "sku_id": sku_id,
            "trigger": True,
            "age_days": age_days,
            "inventory_ratio": round(inventory_ratio, 2),
            "action": "MARKDOWN",
            "discount": discount,
            "current_price": price,
            "new_price": new_price,
            "margin_rate_after": round(margin_rate, 3),
            "margin_warning": margin_warning,
            "forecast_7d": {
                "expected_units": expected_units_sold_7d,
                "expected_revenue": round(expected_revenue_7d, 2)
            },
            "escalation_rule": f"7天后消化率<30%，自动升级至下一降价档",
            "execution_priority": "HIGH" if age_days > 90 else "MEDIUM"
        })
    
    triggered = [d for d in decisions if d.get("trigger")]
    return {
        "total_skus": len(skus),
        "triggered_count": len(triggered),
        "decisions": decisions,
        "summary": f"{len(triggered)}/{len(skus)}个SKU触发清仓降价"
    }


# 测试
from datetime import date, timedelta
today = date(2026, 6, 21)
skus = [
    {  # 触发：库龄68天，库存超2x
        "sku_id": "SLEEP-BAG-XL", "current_inventory": 2400, "target_inventory": 1200,
        "inventory_age_days": 68, "cost_price": 18.0, "current_price": 45.99,
        "launch_date": "2026-01-01", "price_elasticity": -1.8
    },
    {  # 不触发：库龄30天（未超45天）
        "sku_id": "DIAPER-M", "current_inventory": 3000, "target_inventory": 1500,
        "inventory_age_days": 30, "cost_price": 12.0, "current_price": 28.99,
        "launch_date": "2025-06-01", "price_elasticity": -1.2
    },
    {  # 不触发：新品豁免期
        "sku_id": "NEW-BOTTLE", "current_inventory": 500, "target_inventory": 200,
        "inventory_age_days": 50, "cost_price": 8.0, "current_price": 20.99,
        "launch_date": str(today - timedelta(days=60)), "price_elasticity": -1.0
    },
]

result = markdown_clearance_auto_trigger(skus, today=today)

assert result["triggered_count"] == 1
triggered_sku = next(d for d in result["decisions"] if d.get("trigger"))
assert triggered_sku["sku_id"] == "SLEEP-BAG-XL"
assert triggered_sku["discount"] == 0.20  # 库龄68天→-20%
assert triggered_sku["new_price"] < 45.99

# 验证新品豁免
new_sku = next(d for d in result["decisions"] if d["sku_id"] == "NEW-BOTTLE")
assert new_sku["trigger"] == False

print("[✓] Markdown Clearance Auto Trigger决策触发器测试通过")
print(f"  触发SKU: {triggered_sku['sku_id']}, 降价{triggered_sku['discount']:.0%}: ${triggered_sku['current_price']}→${triggered_sku['new_price']}")
print(f"  新品豁免: {new_sku['reason'][:30]}...")
print(f"  摘要: {result['summary']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Markdown-Optimization]]（提供最优降价幅度和弹性估计）
- **延伸（extends）**：[[Skill-Perishable-Inventory-Markdown-Optimization]]（考虑易腐品时效的更复杂降价模型）
- **可组合（combinable）**：[[Skill-Lead-Time-Safety-Stock-Auto-Adjuster]]（清仓腾空后自动触发合理的安全库存调整）

## ⑤ 商业价值评估
- ROI预估：减少库存报废损失15-25%，提升库存周转率0.8-1.5次，年化价值$20,000-$50,000
- 实施难度：⭐⭐☆☆☆（规则明确，需接入WMS库龄数据和定价系统）
- 优先级：⭐⭐⭐⭐⭐

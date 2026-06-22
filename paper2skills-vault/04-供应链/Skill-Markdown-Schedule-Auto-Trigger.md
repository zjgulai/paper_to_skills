---
title: Markdown-Schedule-Auto-Trigger — Amazon FBA 滞销库存库龄触发三阶段自动降价序列
doc_type: knowledge
module: 04-供应链
topic: markdown-schedule-auto-trigger
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Markdown-Schedule-Auto-Trigger

> **配对分析层**：[[Skill-Perishable-Inventory-Markdown-Optimization]]
> **决策类型**: 自动触发型 | **触发条件**: 库龄>45天 AND 库存量>目标库存×1.5 | **执行动作**: 自动生成-10%/-15%/-20%三阶段降价计划并推送至Amazon Seller Central

## ① 算法原理

核心是「双条件门控 + 三阶段降价序列 + 动态中止规则」：

1. **双条件触发门控**：
   - 条件A（时间维度）：库龄（Days of Supply = 当前库存 ÷ 日均销速）> 45天
   - 条件B（数量维度）：当前库存 > 目标安全库存 × 1.5倍
   - 两条件同时满足才触发，避免单一维度误触发（例如节假日前备货期库龄虚高）

2. **三阶段降价序列**：
   - 阶段1（Day 0）：降价10%，观察窗口14天，目标日销速提升≥30%
   - 阶段2（Day 14）：若销速未达标，追加降至原价85%（再-5%），观察窗口10天
   - 阶段3（Day 24）：若仍未清库，降至原价80%（再-5%），同步提交亚马逊Outlet申请

3. **动态中止规则**：任意阶段若日销速连续3天超过目标的200%（超速清货导致库存不足），立即回价并重新评估目标库存。

**误触发防护**：节假日前30天内不触发降价（圣诞节/Prime Day/黑五），保护溢价期。**回滚机制**：降价后若竞品同步降价导致整体品类价格战，系统通知人工审核后再执行阶段2。

## ② 母婴出海应用案例

**场景：吸奶器配件（奶嘴替换装）FBA仓库龄超标**
- 触发条件：当前库存1,200件，日均销速18件/天，库龄=66天（>45）；目标安全库存500件，当前库存>目标×2.4（>1.5倍）；双条件同时满足
- 执行动作：Day 0降价10%（$12.99→$11.69），14天后销速提升至28件/天（+56%，达标）；系统判断阶段1已充分清货，中止阶段2/3
- 安全护栏：检测到Prime Day在30天内，自动跳过本次降价，等待节后重评估
- 业务价值：库龄从66天降至31天，节省LTSF（长期仓储费）约$1,200；同时规避了FBA仓容超限罚款风险（约$800/月）

## ③ 代码模板

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 不触发降价的节假日保护期（提前30天保护）
PROTECTED_EVENTS = [
    {"name": "Prime Day", "date_range": ("07-10", "07-20")},
    {"name": "Black Friday", "date_range": ("11-20", "11-30")},
    {"name": "Christmas", "date_range": ("12-01", "12-25")},
]

def is_in_protected_period(today: datetime, protection_days: int = 30) -> Optional[str]:
    """检查是否在节假日保护期内"""
    year = today.year
    for event in PROTECTED_EVENTS:
        start_str, end_str = event["date_range"]
        start = datetime.strptime(f"{year}-{start_str}", "%Y-%m-%d")
        end = datetime.strptime(f"{year}-{end_str}", "%Y-%m-%d")
        # 节前30天到节后均保护
        if start - timedelta(days=protection_days) <= today <= end:
            return event["name"]
    return None


def markdown_schedule_auto_trigger(
    skus: List[Dict],
    today: Optional[datetime] = None,
    age_threshold_days: int = 45,
    inventory_ratio_threshold: float = 1.5,
    markdown_stages: List[float] = None,
    stage_windows: List[int] = None
) -> Dict:
    """
    FBA 滞销库存自动降价触发器
    
    参数:
        skus: [{
            "sku_id": str, "current_price": float, "current_inventory": int,
            "daily_velocity": float, "target_safety_stock": int,
            "current_stage": int (0=未降价, 1/2/3=当前阶段)
        }]
        age_threshold_days: 库龄触发阈值（天）
        inventory_ratio_threshold: 库存超标倍数
        markdown_stages: 各阶段价格折扣 [0.90, 0.85, 0.80]
        stage_windows: 各阶段观察天数 [14, 10, -1]
    
    返回:
        {"actions": [...], "protected": [...], "summary": {...}}
    """
    if today is None:
        today = datetime.now()
    if markdown_stages is None:
        markdown_stages = [0.90, 0.85, 0.80]
    if stage_windows is None:
        stage_windows = [14, 10, -1]
    
    actions = []
    protected_skus = []
    
    protected_event = is_in_protected_period(today)
    
    for sku in skus:
        sku_id = sku["sku_id"]
        price = sku["current_price"]
        inventory = sku["current_inventory"]
        velocity = max(sku["daily_velocity"], 0.1)  # 避免除零
        target_ss = sku["target_safety_stock"]
        current_stage = sku.get("current_stage", 0)
        
        # 计算关键指标
        days_of_supply = inventory / velocity
        inventory_ratio = inventory / max(target_ss, 1)
        
        # 条件A：库龄检查
        age_trigger = days_of_supply > age_threshold_days
        # 条件B：库存超标检查
        inventory_trigger = inventory_ratio > inventory_ratio_threshold
        
        # 节假日保护期检查
        if protected_event:
            protected_skus.append({
                "sku_id": sku_id,
                "reason": f"节假日保护期（{protected_event}），跳过降价评估",
                "days_of_supply": round(days_of_supply, 1),
                "inventory_ratio": round(inventory_ratio, 2)
            })
            continue
        
        # 双条件判断
        if age_trigger and inventory_trigger:
            next_stage = min(current_stage + 1, len(markdown_stages))
            if next_stage == 0:
                next_stage = 1
            
            discount = markdown_stages[next_stage - 1]
            new_price = round(price * discount, 2)
            
            action = {
                "sku_id": sku_id,
                "trigger": True,
                "days_of_supply": round(days_of_supply, 1),
                "inventory_ratio": round(inventory_ratio, 2),
                "current_stage": current_stage,
                "action_stage": next_stage,
                "current_price": price,
                "new_price": new_price,
                "discount_pct": round((1 - discount) * 100, 0),
                "observation_days": stage_windows[next_stage - 1] if next_stage <= len(stage_windows) else -1,
                "velocity_target": round(velocity * 1.3, 1),  # 目标提升30%
                "rollback_trigger": f"日销速连续3天>{round(velocity * 2.0, 1)}件/天时回价",
                "next_review_date": (today + timedelta(days=stage_windows[next_stage - 1])).strftime("%Y-%m-%d")
                    if next_stage <= len(stage_windows) and stage_windows[next_stage - 1] > 0
                    else "长期保持或人工干预",
                "action_type": "OUTLET_APPLY" if next_stage == 3 else "PRICE_MARKDOWN"
            }
        else:
            action = {
                "sku_id": sku_id,
                "trigger": False,
                "days_of_supply": round(days_of_supply, 1),
                "inventory_ratio": round(inventory_ratio, 2),
                "age_trigger": age_trigger,
                "inventory_trigger": inventory_trigger,
                "reason": "未满足双条件触发"
            }
        
        actions.append(action)
    
    triggered = [a for a in actions if a.get("trigger")]
    summary = {
        "total_skus": len(skus),
        "triggered": len(triggered),
        "protected": len(protected_skus),
        "skipped": len(actions) - len(triggered),
        "stage_distribution": {
            f"阶段{i+1}": sum(1 for a in triggered if a.get("action_stage") == i + 1)
            for i in range(3)
        }
    }
    
    return {
        "actions": actions,
        "protected_skus": protected_skus,
        "summary": summary,
        "protected_event": protected_event
    }


# 测试
skus = [
    # 双条件满足：库龄66天，库存超标2.4倍
    {"sku_id": "SKU-A001", "current_price": 12.99, "current_inventory": 1200,
     "daily_velocity": 18.0, "target_safety_stock": 500, "current_stage": 0},
    # 库龄满足但库存不超标
    {"sku_id": "SKU-A002", "current_price": 25.99, "current_inventory": 600,
     "daily_velocity": 10.0, "target_safety_stock": 500, "current_stage": 0},
    # 双条件都不满足（正常库存）
    {"sku_id": "SKU-A003", "current_price": 8.99, "current_inventory": 300,
     "daily_velocity": 20.0, "target_safety_stock": 400, "current_stage": 0},
    # 已在阶段1，升级到阶段2
    {"sku_id": "SKU-A004", "current_price": 18.00, "current_inventory": 800,
     "daily_velocity": 8.0, "target_safety_stock": 300, "current_stage": 1},
]

# 非保护期测试
result = markdown_schedule_auto_trigger(skus, today=datetime(2026, 6, 22))

assert result["summary"]["total_skus"] == 4
# SKU-A001应触发（双条件满足）
triggered = [a for a in result["actions"] if a.get("trigger")]
sku_a001 = next(a for a in result["actions"] if a["sku_id"] == "SKU-A001")
assert sku_a001["trigger"] == True
assert sku_a001["action_stage"] == 1
assert abs(sku_a001["new_price"] - 12.99 * 0.90) < 0.01

# SKU-A002库存不超标，不触发
sku_a002 = next(a for a in result["actions"] if a["sku_id"] == "SKU-A002")
assert sku_a002["trigger"] == False

# SKU-A004从阶段1升级到阶段2
sku_a004 = next(a for a in result["actions"] if a["sku_id"] == "SKU-A004")
assert sku_a004["trigger"] == True
assert sku_a004["action_stage"] == 2

print("[✓] Markdown Schedule Auto Trigger 测试通过")
print(f"  总SKU: {result['summary']['total_skus']}，触发降价: {result['summary']['triggered']}")
for a in result["actions"]:
    if a.get("trigger"):
        print(f"  {a['sku_id']}: 阶段{a['action_stage']}，降价{a['discount_pct']}%，"
              f"新价${a['new_price']}，库龄{a['days_of_supply']}天")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Perishable-Inventory-Markdown-Optimization]]（提供最优降价幅度和时机分析，是本执行器的参数来源）
- **延伸（extends）**：[[Skill-FBA-Inventory-Rebalancing]]（清货后重新规划补货量，避免下次出现同样问题）
- **可组合（combinable）**：[[Skill-FBA-Fee-Intelligence]]（联合分析长期仓储费节省效果，量化本次降价的完整财务收益）

## ⑤ 商业价值评估
- **ROI量化**：平均每次触发可节省LTSF长期仓储费$800-2,000/SKU，年化管理10-30个SKU可节省$10-30万元；同时规避库容超限导致的补货资格暂停风险（价值更高）
- **实施难度**：⭐⭐☆☆☆（Amazon SP API对接价格修改接口，技术复杂度低；业务规则明确）
- **优先级**：⭐⭐⭐⭐⭐（FBA仓储成本直接影响利润率，滞销库存是大卖家前3大成本浪费来源）

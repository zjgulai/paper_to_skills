---
title: VIP-Tier-Upgrade-Action — LTV超阈值自动触发VIP等级升级与礼遇通知
doc_type: knowledge
module: 14-用户分析
topic: vip-tier-upgrade-action
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: VIP-Tier-Upgrade-Action

> **配对分析层**：[[Skill-User-LTV-Financial-Bridge]]
> **决策类型**: 自动触发型 | **触发条件**: 用户LTV超过等级升级阈值 | **执行动作**: 自动升级VIP等级 + 礼遇通知 + 专属权益激活

## ① 算法原理

核心是「LTV 实时计算 + 阈值比较 + 等级迁移状态机 + 礼遇触发」：

1. **LTV 累计计算**：每次订单完成后实时更新用户累计 LTV（历史消费总额）。
2. **等级阈值定义**（可配置）：
   - Bronze → Silver：LTV ≥ $200
   - Silver → Gold：LTV ≥ $500
   - Gold → Platinum：LTV ≥ $1500
3. **状态机迁移**：检测是否跨越阈值（当前 LTV ≥ 升级阈值 且 历史等级 < 目标等级）。
4. **礼遇序列触发**：等级升级时立即触发「升级通知邮件 + 礼品包邮资格 + 专属折扣码激活」。
5. **降级保护**：等级升级后不因短期 LTV 波动降级（设置 6 个月保级期）。

**关键指标**：升级后 30 天内复购率（VIP 升级锚定效应），目标 >60%。

## ② 母婴出海应用案例

**场景：母婴用品 VIP 体系运营——Silver→Gold 升级**
- 触发条件：用户 U-8821（购买辅食+玩具+安全座椅），今日订单完成后累计 LTV = $512，达到 Gold 阈值 $500
- 执行动作：
  - T+0min：自动升级等级为 Gold，激活「包邮特权 + 8% 永久折扣 + 优先客服通道」
  - T+5min：发送升级通知邮件「恭喜您成为 Gold 会员！专属礼遇已解锁」（含权益卡片）
  - T+1day：推送「Gold 专属商品推荐」（历史购买品类相关的新品）
  - T+7day：实物邮寄「Gold 会员专属礼品袋」（成本 $3，锚定价值感）
- 安全护栏：升级动作幂等（多次判断只执行一次）；退款导致 LTV 跌回阈值以下不立即降级（保级 6 个月）
- 业务价值：Gold 会员 12 个月 LTV 较升级前提升 2.8x，年化增量 GMV $120/人

## ③ 代码模板

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# VIP 等级配置
VIP_TIERS = [
    {"name": "Platinum", "min_ltv": 1500, "perks": ["15%折扣", "免费快递", "专属账户经理", "优先补货"]},
    {"name": "Gold",     "min_ltv": 500,  "perks": ["8%折扣", "包邮", "优先客服通道", "专属礼品袋"]},
    {"name": "Silver",   "min_ltv": 200,  "perks": ["5%折扣", "生日双倍积分", "新品优先试用"]},
    {"name": "Bronze",   "min_ltv": 0,    "perks": ["积分返现1%"]},
]

def get_target_tier(ltv: float) -> Dict:
    for tier in VIP_TIERS:
        if ltv >= tier["min_ltv"]:
            return tier
    return VIP_TIERS[-1]  # Bronze

TIER_RANK = {t["name"]: i for i, t in enumerate(reversed(VIP_TIERS))}

def vip_tier_upgrade_action(
    users: List[Dict],
    now: Optional[datetime] = None,
    protection_months: int = 6
) -> Dict:
    """
    VIP 等级升级执行器
    
    参数:
        users: [{
            "user_id": str, "current_ltv": float,
            "current_tier": str, "last_upgrade_at": str | None,
            "last_order_id": str
        }]
        now: 当前时间（默认当前时间）
        protection_months: 保级保护期（月）
    
    返回:
        {"upgrades": [...], "no_change": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    upgrades = []
    no_change = []
    
    for user in users:
        uid = user["user_id"]
        ltv = user["current_ltv"]
        current_tier_name = user.get("current_tier", "Bronze")
        last_order_id = user.get("last_order_id", "")
        
        target_tier = get_target_tier(ltv)
        current_rank = TIER_RANK.get(current_tier_name, 0)
        target_rank = TIER_RANK.get(target_tier["name"], 0)
        
        if target_rank > current_rank:
            # 触发升级
            upgrade_event = {
                "user_id": uid,
                "action": "TIER_UPGRADE",
                "trigger_order": last_order_id,
                "from_tier": current_tier_name,
                "to_tier": target_tier["name"],
                "ltv_at_upgrade": ltv,
                "unlocked_perks": target_tier["perks"],
                "upgraded_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "protection_until": (now + timedelta(days=30 * protection_months)).strftime("%Y-%m-%d"),
                "notifications": [
                    {
                        "step": 1, "delay_min": 0,
                        "channel": "email",
                        "template": f"vip_upgrade_{target_tier['name'].lower()}",
                        "subject": f"🎉 恭喜升级 {target_tier['name']} 会员！专属礼遇已解锁"
                    },
                    {
                        "step": 2, "delay_min": 1440,  # 1天后
                        "channel": "email",
                        "template": "vip_product_recommendation",
                        "subject": f"{target_tier['name']} 专属新品推荐"
                    }
                ]
            }
            # Platinum 额外触发实物礼品邮寄
            if target_tier["name"] == "Platinum":
                upgrade_event["gift_shipment"] = {
                    "action": "CREATE_GIFT_ORDER",
                    "item": "Platinum 专属礼品盒（价值 $25）",
                    "estimated_ship_days": 3
                }
            elif target_tier["name"] == "Gold":
                upgrade_event["gift_shipment"] = {
                    "action": "CREATE_GIFT_ORDER",
                    "item": "Gold 会员专属礼品袋（价值 $8）",
                    "estimated_ship_days": 5
                }
            
            upgrades.append(upgrade_event)
        
        elif target_rank == current_rank:
            no_change.append({
                "user_id": uid,
                "current_tier": current_tier_name,
                "ltv": ltv,
                "reason": "等级已匹配当前 LTV"
            })
        else:
            # 可能降级，但受保护期保护
            no_change.append({
                "user_id": uid,
                "current_tier": current_tier_name,
                "target_tier": target_tier["name"],
                "ltv": ltv,
                "reason": "LTV 下降但保级期保护，暂不降级"
            })
    
    tier_summary = {}
    for u in upgrades:
        key = f"{u['from_tier']}->{u['to_tier']}"
        tier_summary[key] = tier_summary.get(key, 0) + 1
    
    return {
        "total_users": len(users),
        "upgraded": len(upgrades),
        "no_change": len(no_change),
        "upgrades": upgrades,
        "tier_summary": tier_summary
    }


# 测试
users = [
    {"user_id": "U001", "current_ltv": 512.0, "current_tier": "Silver", "last_order_id": "ORD-8821"},
    {"user_id": "U002", "current_ltv": 1600.0, "current_tier": "Gold",  "last_order_id": "ORD-9022"},
    {"user_id": "U003", "current_ltv": 210.0, "current_tier": "Bronze", "last_order_id": "ORD-7733"},
    {"user_id": "U004", "current_ltv": 350.0, "current_tier": "Silver", "last_order_id": "ORD-6644"},  # 已是Silver，不升级
    {"user_id": "U005", "current_ltv": 80.0,  "current_tier": "Silver", "last_order_id": "ORD-5511"},  # LTV下降，保级
]

now = datetime(2026, 6, 22, 14, 0, 0)
result = vip_tier_upgrade_action(users, now=now)

assert result["total_users"] == 5
assert result["upgraded"] == 3  # U001(Silver→Gold), U002(Gold→Platinum), U003(Bronze→Silver)

upgrade_map = {u["user_id"]: u for u in result["upgrades"]}
assert upgrade_map["U001"]["to_tier"] == "Gold"
assert upgrade_map["U001"]["from_tier"] == "Silver"
assert upgrade_map["U002"]["to_tier"] == "Platinum"
assert "gift_shipment" in upgrade_map["U002"]  # Platinum 有礼品邮寄
assert upgrade_map["U003"]["to_tier"] == "Silver"

print("[✓] VIP Tier Upgrade Action 测试通过")
print(f"  总用户: {result['total_users']}，触发升级: {result['upgraded']}")
print(f"  等级迁移分布: {result['tier_summary']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-User-LTV-Financial-Bridge]]（提供准确的用户 LTV 累计值）
- **延伸（extends）**：[[Skill-RFM-Segment-Campaign-Dispatcher]]（VIP 等级与 RFM 分层协同运营）
- **可组合（combinable）**：[[Skill-High-Value-Customer-Alert-Action]]（VIP 流失预警联动）

## ⑤ 商业价值评估
- **ROI量化**：Gold 会员升级后 12 个月 LTV 提升 2.8x，单用户增量 $120；千人 VIP 池年化 GMV 增量 $120,000
- **实施难度**：⭐⭐☆☆☆（状态机逻辑简单，需对接 CRM + 邮件平台 + 礼品订单系统）
- **优先级**：⭐⭐⭐⭐⭐（VIP 体系是高 LTV 用户锚定的核心机制）

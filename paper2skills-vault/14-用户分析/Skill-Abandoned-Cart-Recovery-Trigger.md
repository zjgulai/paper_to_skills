---
title: Abandoned-Cart-Recovery-Trigger — 加购未购超时自动触发个性化挽回序列
doc_type: knowledge
module: 14-用户分析
topic: abandoned-cart-recovery-trigger
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Abandoned-Cart-Recovery-Trigger

> **配对分析层**：[[Skill-Purchase-Funnel-Drop-Off-Analysis]]
> **决策类型**: 自动触发型 | **触发条件**: 加购后 > 2小时未完成支付 | **执行动作**: 分层发送邮件+优惠券+WhatsApp挽回序列

## ① 算法原理

核心是「加购状态轮询 + 等待超时检测 + 分层触发序列」：

1. **超时检测**：每 30 分钟扫描购物车状态，识别「加购时间 > 2h 且未支付」的用户。
2. **分层判断**：
   - 首次弃购（历史弃购次数 = 0）：发送提醒邮件 + 10% 优惠券
   - 多次弃购（历史弃购次数 ≥ 2）：直接 WhatsApp 主动触达 + 15% 优惠券
   - 高客单价购物车（金额 > $100）：三通道并发（邮件+WhatsApp+短信）
3. **去重防护**：同一用户 24h 内只触发一次，防止骚扰。
4. **回滚机制**：用户完成支付后立即取消后续发送，防止重复推送。

**关键指标**：弃购挽回率（Recovery Rate）= 触发后 48h 内完成支付 / 总触发人数。行业基准约 15-25%，母婴品类可达 28%（高决策成本）。

## ② 母婴出海应用案例

**场景：婴儿安全座椅弃购挽回（客单价 $180-350）**
- 触发条件：用户将某款婴儿座椅加购 2.5h 未支付，购物车金额 $220，首次弃购
- 执行动作：
  - T+2h：发送邮件「您的购物车还在等你——专属优惠 10% off」，含产品评测对比
  - T+24h：WhatsApp 推送「安全认证详情 + 免费安装指导」
  - T+48h：邮件「限时：本周内下单赠送儿童安全检查套装」
- 安全护栏：用户支付后立即终止序列；48h 无响应自动归档
- 业务价值：弃购挽回率从 12% 提升至 29%，月均挽回 GMV 约 $18,000，年化约 $216,000

## ③ 代码模板

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

def abandoned_cart_recovery_trigger(
    cart_events: List[Dict],
    now: Optional[datetime] = None,
    timeout_hours: float = 2.0,
    cooldown_hours: float = 24.0,
    high_value_threshold: float = 100.0
) -> Dict:
    """
    加购弃单挽回触发器
    
    参数:
        cart_events: [{
            "user_id": str, "cart_id": str,
            "added_at": str (ISO8601), "cart_value": float,
            "abandoned_count": int, "paid": bool,
            "last_triggered_at": str | None
        }]
        timeout_hours: 加购后多少小时未支付触发（默认2h）
        cooldown_hours: 同用户触发冷却时间（默认24h）
        high_value_threshold: 高客单价阈值（默认$100）
    
    返回:
        {"triggers": [...], "skipped": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    triggers = []
    skipped = []
    
    for event in cart_events:
        uid = event["user_id"]
        cid = event["cart_id"]
        added_at = datetime.fromisoformat(event["added_at"])
        cart_value = event.get("cart_value", 0)
        abandoned_count = event.get("abandoned_count", 0)
        paid = event.get("paid", False)
        last_triggered_at = event.get("last_triggered_at")
        
        # 已支付：跳过
        if paid:
            skipped.append({"cart_id": cid, "reason": "已完成支付"})
            continue
        
        # 未超时：跳过
        hours_since_add = (now - added_at).total_seconds() / 3600
        if hours_since_add < timeout_hours:
            skipped.append({"cart_id": cid, "reason": f"仅加购{hours_since_add:.1f}h，未到{timeout_hours}h阈值"})
            continue
        
        # 冷却期检查
        if last_triggered_at:
            last_ts = datetime.fromisoformat(last_triggered_at)
            hours_since_trigger = (now - last_ts).total_seconds() / 3600
            if hours_since_trigger < cooldown_hours:
                skipped.append({"cart_id": cid, "reason": f"冷却中（{hours_since_trigger:.1f}h < {cooldown_hours}h）"})
                continue
        
        # 分层触发策略
        channels = []
        discount = 0
        sequence = []
        
        if cart_value >= high_value_threshold:
            # 高客单：三通道并发
            channels = ["email", "whatsapp", "sms"]
            discount = 15
            sequence = [
                {"step": 1, "delay_h": 0,  "channel": "email",    "msg": f"您的高价值购物车（${cart_value:.0f}）还在等你——专属15%折扣"},
                {"step": 2, "delay_h": 4,  "channel": "whatsapp", "msg": "产品安全认证详情 + 专家使用指导"},
                {"step": 3, "delay_h": 24, "channel": "sms",      "msg": f"限时48h：${cart_value:.0f}的专属优惠即将到期"},
            ]
        elif abandoned_count >= 2:
            # 多次弃购：WhatsApp直接触达
            channels = ["whatsapp", "email"]
            discount = 15
            sequence = [
                {"step": 1, "delay_h": 0,  "channel": "whatsapp", "msg": "您好！我们注意到您多次关注此产品，有什么疑问我们来帮您解答？"},
                {"step": 2, "delay_h": 12, "channel": "email",    "msg": f"专属15%折扣 + 免费退换货保障"},
            ]
        else:
            # 首次弃购：邮件序列
            channels = ["email"]
            discount = 10
            sequence = [
                {"step": 1, "delay_h": 0,  "channel": "email", "msg": "您的购物车还在等你——专属10%折扣"},
                {"step": 2, "delay_h": 24, "channel": "email", "msg": "宝宝安全，值得更好的选择（产品对比指南）"},
                {"step": 3, "delay_h": 48, "channel": "email", "msg": "最后机会：本周限时优惠即将结束"},
            ]
        
        # 计算发送时间
        scheduled_sequence = [
            {**s, "send_at": (now + timedelta(hours=s["delay_h"])).strftime("%Y-%m-%dT%H:%M:%S")}
            for s in sequence
        ]
        
        triggers.append({
            "user_id": uid,
            "cart_id": cid,
            "cart_value": cart_value,
            "hours_since_add": round(hours_since_add, 1),
            "abandoned_count": abandoned_count,
            "trigger": True,
            "channels": channels,
            "discount_pct": discount,
            "sequence": scheduled_sequence,
            "stop_condition": "用户支付后立即取消后续发送"
        })
    
    return {
        "total_carts": len(cart_events),
        "triggered": len(triggers),
        "skipped": len(skipped),
        "triggers": triggers,
        "skipped_detail": skipped,
        "stats": {
            "high_value_triggers": sum(1 for t in triggers if t["cart_value"] >= high_value_threshold),
            "multi_abandon_triggers": sum(1 for t in triggers if t["abandoned_count"] >= 2),
            "first_time_triggers": sum(1 for t in triggers if t["abandoned_count"] == 0)
        }
    }


# 测试
cart_events = [
    {"user_id": "U001", "cart_id": "C001", "added_at": "2026-06-22T08:00:00",
     "cart_value": 220.0, "abandoned_count": 0, "paid": False, "last_triggered_at": None},
    {"user_id": "U002", "cart_id": "C002", "added_at": "2026-06-22T09:30:00",
     "cart_value": 45.0, "abandoned_count": 3, "paid": False, "last_triggered_at": None},
    {"user_id": "U003", "cart_id": "C003", "added_at": "2026-06-22T10:00:00",
     "cart_value": 80.0, "abandoned_count": 1, "paid": False, "last_triggered_at": None},
    {"user_id": "U004", "cart_id": "C004", "added_at": "2026-06-22T09:00:00",
     "cart_value": 130.0, "abandoned_count": 0, "paid": True, "last_triggered_at": None},  # 已支付
    {"user_id": "U005", "cart_id": "C005", "added_at": "2026-06-22T09:50:00",
     "cart_value": 60.0, "abandoned_count": 0, "paid": False, "last_triggered_at": None},  # 未超时
]

now = datetime(2026, 6, 22, 12, 0, 0)
result = abandoned_cart_recovery_trigger(cart_events, now=now)

assert result["total_carts"] == 5
assert result["triggered"] == 3  # U001(高客单), U002(多次弃购), U003(首次)
assert result["skipped"] == 2   # U004(已支付), U005(未超时)

# 高客单价校验
high_value = next(t for t in result["triggers"] if t["user_id"] == "U001")
assert high_value["discount_pct"] == 15
assert "whatsapp" in high_value["channels"]
assert "sms" in high_value["channels"]

# 多次弃购校验
multi = next(t for t in result["triggers"] if t["user_id"] == "U002")
assert multi["channels"][0] == "whatsapp"
assert multi["discount_pct"] == 15

print("[✓] Abandoned Cart Recovery Trigger 测试通过")
print(f"  总购物车: {result['total_carts']}，触发挽回: {result['triggered']}，跳过: {result['skipped']}")
print(f"  高客单触发: {result['stats']['high_value_triggers']}，多次弃购触发: {result['stats']['multi_abandon_triggers']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Purchase-Funnel-Drop-Off-Analysis]]（识别漏斗中弃购节点分布）
- **延伸（extends）**：[[Skill-RFM-Segment-Campaign-Dispatcher]]（按用户价值分层触发力度）
- **可组合（combinable）**：[[Skill-Cohort-Churn-Intervention-Dispatcher]]（联合生命周期干预策略）

## ⑤ 商业价值评估
- **ROI量化**：弃购挽回率从 12% → 29%，月均挽回 GMV $18,000，年化 $216,000；优惠券成本约$3,000/月，ROI 6:1
- **实施难度**：⭐⭐☆☆☆（需对接购物车事件流 + WhatsApp Business API）
- **优先级**：⭐⭐⭐⭐⭐（高频高价值，直接影响转化率）

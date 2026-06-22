---
title: Referral-Viral-Loop-Trigger — 高NPS用户自动触发裂变邀请优惠分享码
doc_type: knowledge
module: 06-增长模型
topic: referral-viral-loop-trigger
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Referral-Viral-Loop-Trigger

> **配对分析层**：[[Skill-NPS-Score-Prediction]]
> **决策类型**: 自动触发型 | **触发条件**: 用户NPS代理分>8 AND 购买次数≥2 | **执行动作**: 自动触发裂变邀请（个性化优惠分享码）

## ① 算法原理

核心是「NPS 代理评分 + 复购验证 + 裂变码生成 + 分层激励分发」：

1. **触发条件（AND 逻辑）**：
   - NPS 代理分 > 8（基于购买行为、评论评分、客服互动频次的代理模型预测）
   - 购买次数 ≥ 2（验证真实使用体验，避免一次性买家）
2. **裂变码生成**：每位用户生成唯一 UUID 分享码，绑定推荐人 ID，设置 30 天有效期。
3. **分层激励**：
   - NPS 8-9 分：双向奖励（推荐人 $10 积分，被推荐人 10% 首单折扣）
   - NPS ≥ 9 分：高级双向奖励（推荐人 $15 + 专属 VIP 徽章，被推荐人 15% 折扣）
4. **渠道分发**：优先通过用户历史偏好渠道（邮件/WhatsApp/App Push）分发。
5. **防刷保护**：同一用户每 90 天只触发一次裂变邀请；裂变码只可使用 5 次（防批量分发）。

**关键指标**：裂变系数 K = 每次裂变邀请带来的新用户数，目标 K > 0.5。

## ② 母婴出海应用案例

**场景：婴儿配方奶粉复购用户裂变运营**
- 触发条件：用户 U-3341（购买 3 次奶粉，预测 NPS 代理分 9.1）
- 执行动作：
  - 生成唯一裂变码「MOM-3341-JUNE」，有效期 30 天
  - 发送 WhatsApp 消息「您在妈妈圈的好口碑很重要——分享给身边的妈妈们，您和她都可以享受专属优惠」
  - 被推荐人使用裂变码下单时，推荐人自动获得 $15 积分
- 结果：当月裂变触发 2,200 人，带来新注册 680 人（K=0.31），首月转化 180 人，获客成本 $8.5（vs 广告获客 $42）
- 年化价值：裂变获客成本仅为广告获客的 1/5，年化节省获客成本 $85,000

## ③ 代码模板

```python
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def referral_viral_loop_trigger(
    users: List[Dict],
    now: Optional[datetime] = None,
    nps_threshold: float = 8.0,
    nps_high_threshold: float = 9.0,
    min_purchases: int = 2,
    referral_validity_days: int = 30,
    max_referral_uses: int = 5,
    cooldown_days: int = 90
) -> Dict:
    """
    裂变病毒循环触发器
    
    参数:
        users: [{
            "user_id": str, "predicted_nps": float,
            "purchase_count": int, "preferred_channel": str,
            "last_referral_triggered_at": str | None
        }]
        nps_threshold: 触发裂变的最低NPS代理分
        min_purchases: 最低购买次数
        cooldown_days: 触发冷却天数
    
    返回:
        {"triggers": [...], "skipped": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    triggers = []
    skipped = []
    
    for user in users:
        uid = user["user_id"]
        nps = user.get("predicted_nps", 0)
        purchases = user.get("purchase_count", 0)
        channel = user.get("preferred_channel", "email")
        last_triggered = user.get("last_referral_triggered_at")
        
        # 冷却期检查
        if last_triggered:
            last_ts = datetime.fromisoformat(last_triggered)
            if (now - last_ts).days < cooldown_days:
                skipped.append({"user_id": uid, "reason": f"冷却期内（{cooldown_days}天）"})
                continue
        
        # 条件检查
        if nps < nps_threshold:
            skipped.append({"user_id": uid, "reason": f"NPS代理分{nps:.1f}<{nps_threshold}（非Promoter）"})
            continue
        
        if purchases < min_purchases:
            skipped.append({"user_id": uid, "reason": f"购买次数{purchases}<{min_purchases}（需真实体验）"})
            continue
        
        # 生成唯一裂变码
        code_seed = f"{uid}-{now.strftime('%Y%m')}"
        referral_code = "SHARE-" + hashlib.md5(code_seed.encode()).hexdigest()[:8].upper()
        
        # 分层激励
        if nps >= nps_high_threshold:
            incentive_tier = "PREMIUM"
            referrer_reward = "$15积分 + VIP推荐徽章"
            referee_discount = "15%首单折扣"
            message_tone = "您是我们最特别的家庭成员，您的推荐是给身边妈妈的最好礼物"
        else:
            incentive_tier = "STANDARD"
            referrer_reward = "$10积分"
            referee_discount = "10%首单折扣"
            message_tone = "感谢您的信任，分享给朋友一起享受好产品"
        
        trigger = {
            "user_id": uid,
            "action": "REFERRAL_TRIGGERED",
            "predicted_nps": nps,
            "purchase_count": purchases,
            "referral_code": referral_code,
            "incentive_tier": incentive_tier,
            "referrer_reward": referrer_reward,
            "referee_discount": referee_discount,
            "channel": channel,
            "message_tone": message_tone,
            "valid_until": (now + timedelta(days=referral_validity_days)).strftime("%Y-%m-%d"),
            "max_uses": max_referral_uses,
            "triggered_at": now.strftime("%Y-%m-%dT%H:%M:%S")
        }
        triggers.append(trigger)
    
    premium_count = sum(1 for t in triggers if t.get("incentive_tier") == "PREMIUM")
    
    return {
        "total_users": len(users),
        "triggered": len(triggers),
        "skipped": len(skipped),
        "premium_tier": premium_count,
        "standard_tier": len(triggers) - premium_count,
        "triggers": triggers,
        "skipped_detail": skipped
    }


# 测试
users = [
    {"user_id": "U001", "predicted_nps": 9.1, "purchase_count": 3, "preferred_channel": "whatsapp",
     "last_referral_triggered_at": None},
    {"user_id": "U002", "predicted_nps": 8.3, "purchase_count": 2, "preferred_channel": "email",
     "last_referral_triggered_at": None},
    {"user_id": "U003", "predicted_nps": 7.5, "purchase_count": 4, "preferred_channel": "email",
     "last_referral_triggered_at": None},  # NPS不足
    {"user_id": "U004", "predicted_nps": 9.0, "purchase_count": 1, "preferred_channel": "app",
     "last_referral_triggered_at": None},  # 购买次数不足
    {"user_id": "U005", "predicted_nps": 8.8, "purchase_count": 3, "preferred_channel": "email",
     "last_referral_triggered_at": "2026-05-01T10:00:00"},  # 冷却期内（21天前）
]

now = datetime(2026, 6, 22, 10, 0, 0)
result = referral_viral_loop_trigger(users, now=now)

assert result["total_users"] == 5
assert result["triggered"] == 2  # U001, U002
assert result["skipped"] == 3

trigger_map = {t["user_id"]: t for t in result["triggers"]}
assert trigger_map["U001"]["incentive_tier"] == "PREMIUM"
assert trigger_map["U002"]["incentive_tier"] == "STANDARD"
assert "SHARE-" in trigger_map["U001"]["referral_code"]

print("[✓] Referral Viral Loop Trigger 测试通过")
print(f"  总用户: {result['total_users']}，触发裂变: {result['triggered']}（高级:{result['premium_tier']}，标准:{result['standard_tier']}）")
print(f"  跳过: {result['skipped']} ({[s['reason'] for s in result['skipped_detail']]})")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-NPS-Score-Prediction]]（提供 NPS 代理评分）
- **延伸（extends）**：[[Skill-User-LTV-Financial-Bridge]]（裂变获客 LTV 与广告获客 LTV 对比）
- **可组合（combinable）**：[[Skill-VIP-Tier-Upgrade-Action]]（VIP 升级联动裂变激励）

## ⑤ 商业价值评估
- **ROI量化**：裂变获客成本 $8.5 vs 广告获客 $42，月增量新用户 180 人，年化节省获客成本 $85,000
- **实施难度**：⭐⭐☆☆☆（需裂变码生成系统 + WhatsApp Business API + 积分系统）
- **优先级**：⭐⭐⭐⭐⭐（低成本获客是跨境电商增长的核心杠杆）

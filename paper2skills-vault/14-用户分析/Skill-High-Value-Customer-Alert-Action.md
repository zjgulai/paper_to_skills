---
title: High-Value-Customer-Alert-Action — RFM高价值客户30天沉默自动触发客服主动介入+个性化钩子生成
doc_type: knowledge
module: 14-用户分析
topic: high-value-customer-alert-action
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-High-Value-Customer-Alert-Action

> **配对分析层**：[[Skill-RFM-Customer-Segmentation]]
> **决策类型**: 自动触发型 | **触发条件**: RFM高价值客户（Champions/Loyal）近30天内无购买记录 | **执行动作**: 自动触发客服主动介入（WhatsApp/邮件优先级P0），并生成基于购买历史的个性化钩子文案

## ① 算法原理

核心是「沉默检测 + 高价值过滤 + 个性化钩子生成 + 分级触达策略」：

1. **沉默阈值检测**：
   - Champions（最高价值）：15天无购买即触发早期预警
   - Loyal（高频高M）：30天无购买触发标准预警
   - 触发逻辑：`days_since_last_purchase ≥ alert_threshold AND rfm_segment IN ('Champions', 'Loyal')`

2. **个性化钩子生成**：基于客户购买历史提取3类个性化元素：
   - **宝宝成长钩子**：根据历史订单推断宝宝年龄段，匹配当前阶段需求（「6个月宝宝辅食期」）
   - **品类偏好钩子**：客户最常购买品类作为推荐锚点
   - **价值认可钩子**：历史复购次数作为VIP身份强调（「您已是第8次回购的VIP顾客」）

3. **分级触达渠道**：
   - 历史LTV > $300：WhatsApp + 专属客服人工跟进（24小时响应SLA）
   - $150 ≤ LTV ≤ $300：邮件序列（自动触发，2封）
   - LTV < $150 但仍属高分群：邮件单封（轻量触达）

4. **触达频控**：同一客户本月内仅允许1次主动介入（WhatsApp/人工）+ 1次邮件，防止打扰高价值客户反而流失。

**误触发防护**：客户已在进行中的订单（状态=pending/shipped）自动排除（购物并未真正沉默）。**回滚机制**：触达后若客户主动退订WhatsApp/邮件，立即停止所有主动触达并标记DO_NOT_CONTACT。

## ② 母婴出海应用案例

**场景：母婴品牌高价值客户流失前预警与主动挽留**
- 触发条件：本周RFM模型更新，识别出23名Champions/Loyal客户超过15/30天未购买
  - 其中LTV>$300者：8名（WhatsApp介入级别）
  - $150-300者：11名（邮件级别）
  - <$150者：4名（轻量邮件）
- 执行动作：
  - 8名高LTV客户：WhatsApp模板「[客服名]：Hi [姓名]，您的宝宝应该进入9个月辅食期了，我们新到了一批有机泥糊产品，给您留了VIP优先购链接」
  - 11名中LTV：2封邮件序列，Day0主题「您的VIP专属新品推荐」，Day5主题「宝宝成长必备清单」
- 安全护栏：检查CRM确认8名中无进行中订单；近7天内无任何触达记录才发送
- 业务价值：Champions群主动介入挽留率约28%，8名×28%×$420平均LTV = 增量GMV约$940/周；年化约$49,000

## ③ 代码模板

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 各分群沉默预警阈值（天）
SILENCE_THRESHOLDS = {
    "Champions": 15,
    "Loyal":     30,
}

# LTV分级和触达渠道
LTV_TIERS = [
    {"min_ltv": 300,  "channel": "whatsapp+cs", "label": "高LTV-人工介入"},
    {"min_ltv": 150,  "channel": "email_2step",  "label": "中LTV-邮件序列"},
    {"min_ltv": 0,    "channel": "email_1step",  "label": "低LTV-单封邮件"},
]

# 宝宝月龄成长钩子映射
BABY_AGE_HOOKS = {
    (0, 3):   "新生儿期，产后恢复和初乳准备",
    (3, 6):   "宝宝3-6个月，奶粉/吸奶器核心需求期",
    (6, 9):   "宝宝6-9个月，辅食启蒙阶段",
    (9, 12):  "宝宝9-12个月，手指食物和学步期",
    (12, 18): "宝宝1-1.5岁，断奶过渡和自主进食培养",
    (18, 36): "宝宝1.5-3岁，学前营养和早教玩具期",
    (36, 99): "宝宝3岁以上，进入幼儿成长阶段",
}


def get_baby_age_hook(baby_birth_month: Optional[str], today: datetime) -> str:
    """根据宝宝出生月份生成成长阶段钩子"""
    if not baby_birth_month:
        return "宝宝成长关键期，精选必备好物"
    try:
        birth = datetime.strptime(baby_birth_month, "%Y-%m")
        months = (today.year - birth.year) * 12 + (today.month - birth.month)
        for (lo, hi), hook in BABY_AGE_HOOKS.items():
            if lo <= months < hi:
                return f"宝宝{months}个月大，{hook}"
        return f"宝宝{months}个月大，持续成长中"
    except Exception:
        return "宝宝成长每个阶段都有新需求"


def get_ltv_tier(ltv: float) -> Dict:
    """按LTV分级确定触达渠道"""
    for tier in LTV_TIERS:
        if ltv >= tier["min_ltv"]:
            return tier
    return LTV_TIERS[-1]


def high_value_customer_alert_action(
    customers: List[Dict],
    today: Optional[datetime] = None,
    max_monthly_contacts: int = 1,  # 每月最大主动接触次数
) -> Dict:
    """
    RFM高价值客户沉默预警+主动介入动作生成器
    
    参数:
        customers: [{
            "customer_id": str,
            "rfm_segment": str,         # "Champions" / "Loyal"
            "ltv_estimate": float,
            "days_since_last_purchase": int,
            "has_active_order": bool,   # 是否有进行中订单
            "monthly_contact_count": int,  # 本月已触达次数
            "last_contact_date": Optional[str],  # 上次主动联系日期
            "top_purchase_category": str,
            "purchase_count": int,       # 历史总购买次数
            "baby_birth_month": Optional[str],
            "customer_name": str,
        }]
        today: 当前日期
        max_monthly_contacts: 每月最大触达次数
    
    返回:
        {"alerts": [...], "no_action": [...], "summary": {...}}
    """
    if today is None:
        today = datetime.now()
    
    alerts = []
    no_action = []
    
    for c in customers:
        cid = c["customer_id"]
        segment = c.get("rfm_segment", "")
        ltv = c.get("ltv_estimate", 0)
        days_silent = c.get("days_since_last_purchase", 0)
        has_order = c.get("has_active_order", False)
        monthly_contacts = c.get("monthly_contact_count", 0)
        
        # 跳过不在高价值分群的客户
        if segment not in SILENCE_THRESHOLDS:
            no_action.append({"customer_id": cid, "reason": f"分群{segment}不在预警范围"})
            continue
        
        threshold = SILENCE_THRESHOLDS[segment]
        
        # 跳过条件
        if has_order:
            no_action.append({"customer_id": cid, "reason": "有进行中订单，非真实沉默"})
            continue
        
        if monthly_contacts >= max_monthly_contacts:
            no_action.append({
                "customer_id": cid,
                "reason": f"本月已触达{monthly_contacts}次，达上限{max_monthly_contacts}"
            })
            continue
        
        if days_silent < threshold:
            no_action.append({
                "customer_id": cid,
                "reason": f"沉默{days_silent}天<阈值{threshold}天，暂不触发"
            })
            continue
        
        # 生成个性化钩子
        baby_hook = get_baby_age_hook(c.get("baby_birth_month"), today)
        category_hook = c.get("top_purchase_category", "母婴好物")
        purchase_count = c.get("purchase_count", 1)
        name = c.get("customer_name", "亲爱的顾客")
        
        # LTV分级 → 触达渠道
        tier = get_ltv_tier(ltv)
        channel = tier["channel"]
        
        # 生成触达内容
        if channel == "whatsapp+cs":
            message_template = (
                f"Hi {name}，您已是我们第{purchase_count}次回购的VIP顾客🎉 "
                f"注意到您最近一段时间没来光顾，{baby_hook}，"
                f"我们刚上架了一批精选{category_hook}产品，特地为您保留了VIP优先购名额，"
                f"请问方便聊几分钟吗？"
            )
            action_detail = {
                "channel": "WhatsApp",
                "cs_action": "人工跟进（24小时SLA）",
                "sla_hours": 24,
                "message": message_template
            }
        elif channel == "email_2step":
            action_detail = {
                "channel": "Email",
                "sequence": [
                    {
                        "step": 1, "day_offset": 0,
                        "subject": f"想念您！{category_hook}新品VIP专属",
                        "body_hint": f"{baby_hook}，精选为您推荐"
                    },
                    {
                        "step": 2, "day_offset": 5,
                        "subject": f"宝宝成长清单·专属为您准备",
                        "body_hint": f"您是我们的第{purchase_count}次回购VIP，专属优惠等您来"
                    }
                ]
            }
        else:
            action_detail = {
                "channel": "Email",
                "sequence": [
                    {
                        "step": 1, "day_offset": 0,
                        "subject": f"Hi {name}，{baby_hook}",
                        "body_hint": f"精选{category_hook}，专为您推荐"
                    }
                ]
            }
        
        alert = {
            "customer_id": cid,
            "trigger": True,
            "rfm_segment": segment,
            "days_silent": days_silent,
            "ltv_estimate": ltv,
            "ltv_tier": tier["label"],
            "priority": "P0" if channel == "whatsapp+cs" else "P1",
            "action": action_detail,
            "personalization": {
                "baby_hook": baby_hook,
                "category_hook": category_hook,
                "purchase_count": purchase_count
            }
        }
        alerts.append(alert)
    
    p0_count = sum(1 for a in alerts if a.get("priority") == "P0")
    p1_count = sum(1 for a in alerts if a.get("priority") == "P1")
    total_ltv_at_risk = sum(a["ltv_estimate"] for a in alerts)
    
    summary = {
        "total_input": len(customers),
        "triggered_alerts": len(alerts),
        "no_action": len(no_action),
        "p0_whatsapp_cs": p0_count,
        "p1_email": p1_count,
        "total_ltv_at_risk": round(total_ltv_at_risk, 2),
        "estimated_save_rate": 0.28,  # Champions群历史挽留率
        "estimated_incremental_gmv": round(total_ltv_at_risk * 0.28, 2)
    }
    
    return {
        "alerts": alerts,
        "no_action": no_action,
        "summary": summary
    }


# 测试
customers = [
    # Champions，32天沉默，LTV $420 → WhatsApp+CS (P0)
    {"customer_id": "HV001", "rfm_segment": "Champions", "ltv_estimate": 420,
     "days_since_last_purchase": 32, "has_active_order": False, "monthly_contact_count": 0,
     "top_purchase_category": "吸奶器", "purchase_count": 8,
     "baby_birth_month": "2025-09", "customer_name": "Lisa"},
    # Champions，但有进行中订单 → 跳过
    {"customer_id": "HV002", "rfm_segment": "Champions", "ltv_estimate": 380,
     "days_since_last_purchase": 20, "has_active_order": True, "monthly_contact_count": 0,
     "top_purchase_category": "奶瓶", "purchase_count": 6, "baby_birth_month": None, "customer_name": "Emma"},
    # Loyal，35天沉默，LTV $220 → 邮件2步 (P1)
    {"customer_id": "HV003", "rfm_segment": "Loyal", "ltv_estimate": 220,
     "days_since_last_purchase": 35, "has_active_order": False, "monthly_contact_count": 0,
     "top_purchase_category": "辅食", "purchase_count": 5,
     "baby_birth_month": "2025-06", "customer_name": "Sarah"},
    # Loyal，沉默未达阈值（20天<30天）→ 跳过
    {"customer_id": "HV004", "rfm_segment": "Loyal", "ltv_estimate": 180,
     "days_since_last_purchase": 20, "has_active_order": False, "monthly_contact_count": 0,
     "top_purchase_category": "奶嘴", "purchase_count": 4, "baby_birth_month": None, "customer_name": "Amy"},
    # 本月已触达次数达上限 → 跳过
    {"customer_id": "HV005", "rfm_segment": "Champions", "ltv_estimate": 350,
     "days_since_last_purchase": 25, "has_active_order": False, "monthly_contact_count": 1,
     "top_purchase_category": "玩具", "purchase_count": 7, "baby_birth_month": None, "customer_name": "Mia"},
]

result = high_value_customer_alert_action(customers, today=datetime(2026, 6, 22))

assert result["summary"]["total_input"] == 5
assert result["summary"]["triggered_alerts"] == 2  # HV001, HV003
assert result["summary"]["p0_whatsapp_cs"] == 1     # HV001
assert result["summary"]["p1_email"] == 1            # HV003

hv001 = next(a for a in result["alerts"] if a["customer_id"] == "HV001")
assert hv001["priority"] == "P0"
assert hv001["action"]["channel"] == "WhatsApp"
assert "吸奶器" in hv001["action"]["message"]  # 个性化品类钩子

hv002 = next(a for a in result["no_action"] if a["customer_id"] == "HV002")
assert "进行中订单" in hv002["reason"]

print("[✓] High Value Customer Alert Action 测试通过")
print(f"  总输入: {result['summary']['total_input']}，触发预警: {result['summary']['triggered_alerts']}，无需操作: {result['summary']['no_action']}")
print(f"  P0(WhatsApp+CS): {result['summary']['p0_whatsapp_cs']}人，P1(邮件): {result['summary']['p1_email']}人")
print(f"  高危LTV总量: ${result['summary']['total_ltv_at_risk']}，预计挽回GMV: ${result['summary']['estimated_incremental_gmv']}")
for a in result["alerts"]:
    print(f"  [{a['priority']}] {a['customer_id']} ({a['rfm_segment']}) - 沉默{a['days_silent']}天 - {a['ltv_tier']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-RFM-Customer-Segmentation]]（提供Champions/Loyal分群标签和客户评分，是本预警器的核心分群来源）
- **延伸（extends）**：[[Skill-Customer-Journey-Analytics]]（追踪主动介入触达后的转化漏斗，量化各渠道挽留效果并优化阈值）
- **可组合（combinable）**：[[Skill-RFM-Segment-Campaign-Dispatcher]]（高价值沉默预警与常规RFM营销序列并行运行，前者处理异常状态，后者处理正常运营节奏）

## ⑤ 商业价值评估
- **ROI量化**：Champions群主动介入挽留率约25-30%，年化对50名Champions客户主动介入×28%挽留×$400 LTV = 增量GMV约$56,000；额外规避了高LTV客户流失的品牌口碑风险
- **实施难度**：⭐⭐☆☆☆（WhatsApp Business API对接约1周工程量；邮件平台对接更简单；个性化钩子逻辑规则清晰）
- **优先级**：⭐⭐⭐⭐⭐（高LTV客户是品牌最核心资产，每失去一位Champions意味着失去数百美元长期价值；主动预防远优于被动挽回）

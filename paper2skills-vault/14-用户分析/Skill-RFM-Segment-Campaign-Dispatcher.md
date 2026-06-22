---
title: RFM-Segment-Campaign-Dispatcher — RFM分群结果自动触发差异化营销序列调度器
doc_type: knowledge
module: 14-用户分析
topic: rfm-segment-campaign-dispatcher
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-RFM-Segment-Campaign-Dispatcher

> **配对分析层**：[[Skill-RFM-Customer-Segmentation]]
> **决策类型**: 自动触发型 | **触发条件**: RFM分群标签更新后每周自动调度 | **执行动作**: 高价值→VIP礼遇序列；流失风险→挽回邮件；沉默→唤醒推送

## ① 算法原理

核心是「RFM分群标签读取 + 多路分流调度 + 差异化营销序列生成」：

1. **分群标签映射**：读取上游RFM分群结果，将客户分为4个行动分桶：
   - **Champions**（高R高F高M）：VIP礼遇序列，强化关系
   - **At-Risk**（高M但低R）：挽回邮件序列（3封），优先级P0
   - **Lost-Customers**（极低R极低F）：唤醒推送序列（轻量，低成本）
   - **Others**（普通活跃）：常规周报Newsletter

2. **序列调度逻辑**：按优先级排列（At-Risk > Champions > Lost > Others），高优先级客户先获取有限资源（如客服配额、优惠券库存）。

3. **冷却期控制**：同一客户触达间隔≥7天（Champions为14天），防止轰炸式营销导致退订。

4. **个性化钩子注入**：每条消息携带客户最后购买品类、首次购买时间、宝宝预估月龄（基于购买记录反推），提升相关性。

**误触发防护**：单次调度批次中同一客户仅进入一个桶（去重）；RFM分群数据延迟>7天时告警，暂停调度。**回滚机制**：每批次发送后7天追踪开信率/转化率，若At-Risk序列转化率<1%（历史均值5%），触发序列内容审查。

## ② 母婴出海应用案例

**场景：母婴品牌季度RFM分析完成，驱动差异化邮件营销**
- 触发条件：本季度RFM分群完成，Champions=342人，At-Risk=158人，Lost=890人
- 执行动作：
  - At-Risk 158人（高M近期流失）：P0优先，发送「专属回购礼遇」+15%优惠码，3封序列（Day0/Day5/Day12）
  - Champions 342人：发送「VIP会员专属新品预览」+免费样品邀请
  - Lost 890人：微信/邮件单封唤醒「宝宝成长里程碑」场景关联推送，预算最低
- 安全护栏：At-Risk客户中若近3天内已被客服联系，跳过本批次（对接CRM防重复）
- 业务价值：At-Risk群挽回率历史约12%，每季可挽回约19位客户，人均LTV $220，季度增量GMV约$4,180

## ③ 代码模板

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

# RFM分群到营销序列的映射规则
SEGMENT_CAMPAIGN_MAP = {
    "Champions": {
        "priority": 2,
        "campaign_type": "VIP_LOYALTY",
        "sequence": [
            {"day": 0, "channel": "email", "template": "vip_new_product_preview", "subject": "VIP专属：新品抢先看"},
            {"day": 7, "channel": "sms", "template": "vip_free_sample", "subject": "您的免费样品已备好"},
        ],
        "cooldown_days": 14,
        "max_per_batch": 500
    },
    "At-Risk": {
        "priority": 1,  # 最高优先级
        "campaign_type": "WINBACK_URGENCY",
        "sequence": [
            {"day": 0,  "channel": "email", "template": "winback_offer", "subject": "我们想念您！专属15%回购礼"},
            {"day": 5,  "channel": "email", "template": "winback_reminder", "subject": "宝宝成长，这些产品正合适"},
            {"day": 12, "channel": "email", "template": "winback_last", "subject": "最后机会：您的专属优惠码即将到期"},
        ],
        "cooldown_days": 7,
        "max_per_batch": 300
    },
    "Lost-Customers": {
        "priority": 3,
        "campaign_type": "REACTIVATION_LIGHT",
        "sequence": [
            {"day": 0, "channel": "push", "template": "reactivation_milestone", "subject": "宝宝成长记录：{baby_age_hint}"},
        ],
        "cooldown_days": 7,
        "max_per_batch": 2000
    },
    "Others": {
        "priority": 4,
        "campaign_type": "NEWSLETTER",
        "sequence": [
            {"day": 0, "channel": "email", "template": "weekly_newsletter", "subject": "本周母婴精选"},
        ],
        "cooldown_days": 7,
        "max_per_batch": 5000
    }
}


def rfm_segment_campaign_dispatcher(
    customers: List[Dict],
    today: Optional[datetime] = None,
    coupon_pool_size: int = 200,  # 可用优惠券数量
    cs_capacity_per_day: int = 50  # 客服每日可处理量
) -> Dict:
    """
    RFM分群营销序列调度器
    
    参数:
        customers: [{
            "customer_id": str, "rfm_segment": str,
            "last_campaign_date": Optional[str],  # 上次触达日期
            "last_purchase_category": str,
            "ltv_estimate": float,
            "baby_birth_month": Optional[str],  # 用于计算宝宝月龄钩子
            "cs_contacted_3d": bool  # 近3天是否已被客服联系
        }]
        coupon_pool_size: At-Risk可发放的优惠券总量
        cs_capacity_per_day: 客服每日人工跟进上限
    
    返回:
        {"dispatch_plan": {...}, "resource_allocation": {...}, "summary": {...}}
    """
    if today is None:
        today = datetime.now()
    
    # 按分群归类，去重（一个客户只进一个桶）
    segment_buckets = defaultdict(list)
    skipped = []
    
    for customer in customers:
        cid = customer["customer_id"]
        segment = customer.get("rfm_segment", "Others")
        
        # 冷却期检查
        last_campaign = customer.get("last_campaign_date")
        if last_campaign:
            last_dt = datetime.strptime(last_campaign, "%Y-%m-%d")
            cooldown = SEGMENT_CAMPAIGN_MAP.get(segment, {}).get("cooldown_days", 7)
            if (today - last_dt).days < cooldown:
                skipped.append({"customer_id": cid, "reason": f"冷却期内（{(today - last_dt).days}天<{cooldown}天）"})
                continue
        
        # At-Risk特殊护栏：近3天已被客服联系则跳过
        if segment == "At-Risk" and customer.get("cs_contacted_3d", False):
            skipped.append({"customer_id": cid, "reason": "近3天已被客服联系，跳过本批次"})
            continue
        
        segment_buckets[segment].append(customer)
    
    # 按优先级排序分群
    sorted_segments = sorted(
        SEGMENT_CAMPAIGN_MAP.keys(),
        key=lambda s: SEGMENT_CAMPAIGN_MAP[s]["priority"]
    )
    
    dispatch_plan = {}
    coupon_remaining = coupon_pool_size
    cs_remaining = cs_capacity_per_day * 3  # 3天内可处理量
    
    for segment in sorted_segments:
        config = SEGMENT_CAMPAIGN_MAP[segment]
        customers_in_segment = segment_buckets.get(segment, [])
        
        # 批次上限
        max_batch = config["max_per_batch"]
        # 资源分配：At-Risk消耗优惠券
        if segment == "At-Risk":
            available = min(len(customers_in_segment), max_batch, coupon_remaining)
            coupon_remaining -= available
        else:
            available = min(len(customers_in_segment), max_batch)
        
        selected = customers_in_segment[:available]
        
        # 生成个性化消息计划
        message_plans = []
        for c in selected:
            # 计算宝宝月龄钩子
            baby_hint = "宝宝成长关键期"
            if c.get("baby_birth_month"):
                try:
                    birth = datetime.strptime(c["baby_birth_month"], "%Y-%m")
                    months = (today.year - birth.year) * 12 + (today.month - birth.month)
                    baby_hint = f"宝宝{months}个月大"
                except Exception:
                    pass
            
            for step in config["sequence"]:
                send_date = today + timedelta(days=step["day"])
                message_plans.append({
                    "customer_id": c["customer_id"],
                    "send_date": send_date.strftime("%Y-%m-%d"),
                    "channel": step["channel"],
                    "template": step["template"],
                    "subject": step["subject"].replace("{baby_age_hint}", baby_hint),
                    "personalization": {
                        "last_category": c.get("last_purchase_category", ""),
                        "baby_hint": baby_hint
                    }
                })
        
        dispatch_plan[segment] = {
            "campaign_type": config["campaign_type"],
            "priority": config["priority"],
            "total_eligible": len(customers_in_segment),
            "dispatched": available,
            "waiting_list": len(customers_in_segment) - available,
            "message_plans": message_plans,
            "estimated_ltv_at_risk": round(
                sum(c.get("ltv_estimate", 0) for c in selected), 2
            )
        }
    
    # 资源分配汇总
    resource_allocation = {
        "coupons_used": coupon_pool_size - coupon_remaining,
        "coupons_remaining": coupon_remaining,
        "cs_capacity_used": min(len(dispatch_plan.get("At-Risk", {}).get("dispatched", 0) or
                                 dispatch_plan.get("At-Risk", {}).get("dispatched", 0), cs_remaining),
    }
    
    total_dispatched = sum(v.get("dispatched", 0) for v in dispatch_plan.values())
    summary = {
        "total_customers": len(customers),
        "total_dispatched": total_dispatched,
        "total_skipped": len(skipped),
        "segment_counts": {seg: len(segment_buckets[seg]) for seg in SEGMENT_CAMPAIGN_MAP},
        "total_messages": sum(
            len(v.get("message_plans", [])) for v in dispatch_plan.values()
        )
    }
    
    return {
        "dispatch_plan": dispatch_plan,
        "skipped": skipped,
        "resource_allocation": resource_allocation,
        "summary": summary
    }


# 测试
customers = [
    {"customer_id": "C001", "rfm_segment": "Champions", "last_campaign_date": None,
     "last_purchase_category": "奶瓶", "ltv_estimate": 380, "baby_birth_month": "2025-06", "cs_contacted_3d": False},
    {"customer_id": "C002", "rfm_segment": "At-Risk", "last_campaign_date": None,
     "last_purchase_category": "奶嘴", "ltv_estimate": 220, "baby_birth_month": "2024-12", "cs_contacted_3d": False},
    {"customer_id": "C003", "rfm_segment": "At-Risk", "last_campaign_date": None,
     "last_purchase_category": "吸奶器", "ltv_estimate": 350, "baby_birth_month": None, "cs_contacted_3d": True},  # 近3天已联系
    {"customer_id": "C004", "rfm_segment": "Lost-Customers", "last_campaign_date": "2026-06-20",
     "last_purchase_category": "辅食", "ltv_estimate": 80, "baby_birth_month": "2024-03", "cs_contacted_3d": False},  # 冷却期内
    {"customer_id": "C005", "rfm_segment": "Lost-Customers", "last_campaign_date": "2026-06-01",
     "last_purchase_category": "辅食", "ltv_estimate": 60, "baby_birth_month": "2023-09", "cs_contacted_3d": False},
    {"customer_id": "C006", "rfm_segment": "Others", "last_campaign_date": None,
     "last_purchase_category": "玩具", "ltv_estimate": 150, "baby_birth_month": None, "cs_contacted_3d": False},
]

result = rfm_segment_campaign_dispatcher(customers, today=datetime(2026, 6, 22), coupon_pool_size=100)

assert result["summary"]["total_customers"] == 6
# C003被跳过（客服护栏）
assert any(s["customer_id"] == "C003" for s in result["skipped"])
# C004被跳过（冷却期：2026-06-20距今2天 < 7天）
assert any(s["customer_id"] == "C004" for s in result["skipped"])
# At-Risk应有分发计划（C002满足条件）
assert result["dispatch_plan"]["At-Risk"]["dispatched"] == 1
# At-Risk消耗优惠券
assert result["resource_allocation"]["coupons_used"] == 1

print("[✓] RFM Segment Campaign Dispatcher 测试通过")
print(f"  总客户: {result['summary']['total_customers']}，调度: {result['summary']['total_dispatched']}，跳过: {result['summary']['total_skipped']}")
print(f"  优惠券消耗: {result['resource_allocation']['coupons_used']}/{100}")
for seg, plan in result["dispatch_plan"].items():
    if plan["dispatched"] > 0:
        print(f"  [{seg}] 调度{plan['dispatched']}人，预计守护LTV ${plan['estimated_ltv_at_risk']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-RFM-Customer-Segmentation]]（提供客户RFM分群标签，是本调度器的核心输入）
- **延伸（extends）**：[[Skill-Email-Personalization-Engine]]（提升序列消息个性化程度，基于购买历史动态生成文案）
- **可组合（combinable）**：[[Skill-Customer-Journey-Analytics]]（追踪各分群营销序列的转化漏斗，量化ROI并优化下一周期分配规则）

## ⑤ 商业价值评估
- **ROI量化**：At-Risk群挽回率约10-15%，年化4季度×160人×12%挽回×$220 LTV = 年增量GMV约$16,896；Champions群VIP序列复购率提升约8%，年化LTV增量约$30,000
- **实施难度**：⭐⭐☆☆☆（主要工作是对接邮件/短信平台API，业务规则简洁明确）
- **优先级**：⭐⭐⭐⭐⭐（母婴复购率是核心增长指标，RFM调度是存量运营的标配基础设施）

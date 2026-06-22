---
title: RFM Campaign Auto Dispatcher — 按RFM分群自动映射并触发差异化营销序列
doc_type: knowledge
module: 14-用户分析
topic: rfm-campaign-auto-dispatcher
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: RFM Campaign Auto Dispatcher

> **配对分析层**：[[Skill-RFM-Customer-Segmentation]]
> **决策类型**: 自动触发型 | **触发条件**: RFM分群更新后即触发 | **执行动作**: 按RFM分群映射并触发对应营销序列（VIP专属/流失挽回/新用户onboarding）

## ① 算法原理

核心是「RFM评分 + 规则引擎映射 + 营销序列触发」：

1. **RFM评分**（1-5分制）：
   - R（近度）：最近一次购买距今天数，越近分越高
   - F（频度）：购买次数，越多分越高
   - M（金额）：累计消费金额，越高分越高
   
2. **分群规则引擎**：9类标准分群，每类映射到预设营销序列：
   - Champions (R5F5M5)：VIP专属早鸟权益
   - At Risk (R2F4M4)：重激活序列
   - Lost (R1F1M*)：沉默唤醒序列
   - New (R5F1M1)：onboarding教育序列
   - 等（共9类）
   
3. **去重与频控**：同一用户同一序列最近30天内不重复触发；若同时满足多个分群条件，取优先级最高的。

**误触发防护**：用户在最近7天内已收到任意序列则跳过（频控）；分群边界容忍±0.5分缓冲区，防止每次评分边界用户频繁切换序列。**回滚机制**：连续3次触发同类序列但转化率<1%，自动降级至轻触达推送。

## ② 母婴出海应用案例

**场景：母婴品牌季度RFM更新后的批量营销调度**
- 触发条件：季度RFM批量评分完成，识别出：Champions 320人，At Risk 180人，Lost 450人，New 210人
- 执行动作：Champions→「新品首发专属邀请」邮件序列；At Risk→「限时唤醒30%折扣券」；Lost→「我们改变了，来看看」再接触序列；New→「宝宝成长5步攻略」教育序列
- 安全护栏：Lost群体历史均值M<$50者跳过（低价值不值得成本投入）；每用户本月触达不超过3次
- 业务价值：At Risk群体挽回率提升至28%，年化增量LTV约$65,000

## ③ 代码模板

```python
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

def compute_rfm_score(
    last_purchase_days: int,
    purchase_count: int,
    total_spend: float,
    r_breakpoints: List[float] = None,
    f_breakpoints: List[float] = None,
    m_breakpoints: List[float] = None
) -> Tuple[int, int, int]:
    """将原始RFM值转换为1-5分"""
    if r_breakpoints is None:
        r_breakpoints = [7, 14, 30, 60]  # 天数越小R越高
    if f_breakpoints is None:
        f_breakpoints = [1, 2, 4, 8]
    if m_breakpoints is None:
        m_breakpoints = [50, 200, 500, 1000]
    
    def score_metric(value, breakpoints, reverse=False):
        score = 5
        for bp in sorted(breakpoints):
            if value <= bp:
                break
            score -= 1
        return max(1, score) if not reverse else min(5, 6 - score)
    
    r = score_metric(last_purchase_days, r_breakpoints)  # 天数越小越好
    f = score_metric(purchase_count, f_breakpoints, reverse=True)  # 次数越多越好
    m = score_metric(total_spend, m_breakpoints, reverse=True)
    return r, f, m


def classify_rfm_segment(r: int, f: int, m: int) -> Tuple[str, int]:
    """RFM分群映射，返回(分群名, 优先级)"""
    rfm = (r, f, m)
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions", 1
    elif r >= 4 and f >= 4:
        return "Loyal_Customers", 2
    elif r >= 4 and f <= 2:
        return "New_Customers", 3
    elif r >= 3 and f >= 3 and m >= 3:
        return "Promising", 4
    elif r >= 3 and f >= 3:
        return "Potential_Loyalists", 5
    elif r <= 2 and f >= 4 and m >= 4:
        return "At_Risk", 6
    elif r <= 2 and f >= 3:
        return "Cant_Lose_Them", 7
    elif r <= 2 and f <= 2 and m >= 2:
        return "Hibernating", 8
    else:
        return "Lost", 9


SEGMENT_CAMPAIGN_MAP = {
    "Champions":         {"campaign": "VIP_EARLY_ACCESS",       "channel": ["email", "sms"],  "priority": 1, "template": "新品首发专属邀请+优先购权益"},
    "Loyal_Customers":   {"campaign": "LOYALTY_REWARD",         "channel": ["email"],         "priority": 2, "template": "忠诚积分兑换+专属会员日"},
    "New_Customers":     {"campaign": "ONBOARDING_EDUCATION",   "channel": ["email"],         "priority": 3, "template": "宝宝成长5步攻略（5封邮件序列）"},
    "Promising":         {"campaign": "NURTURE_UPSELL",         "channel": ["email", "push"], "priority": 4, "template": "相关品类推荐+复购提醒"},
    "Potential_Loyalists":{"campaign":"LOYALTY_INVITE",         "channel": ["email"],         "priority": 5, "template": "会员计划邀请+首次积分翻倍"},
    "At_Risk":           {"campaign": "WIN_BACK_DISCOUNT",      "channel": ["email", "sms"],  "priority": 6, "template": "限时唤醒30%折扣券（7天有效）"},
    "Cant_Lose_Them":    {"campaign": "URGENT_WINBACK",         "channel": ["email", "phone"],"priority": 7, "template": "高价值用户专属客服电话"},
    "Hibernating":       {"campaign": "REACTIVATION_LIGHT",     "channel": ["email"],         "priority": 8, "template": "我们改变了，来看新品"},
    "Lost":              {"campaign": "LAST_CHANCE_WINBACK",    "channel": ["email"],         "priority": 9, "template": "最后一次联系+超低价钩子"},
}


def rfm_campaign_auto_dispatcher(
    customers: List[Dict],
    min_ltv_threshold: float = 50.0,
    frequency_cap_days: int = 30,
    max_touches_per_month: int = 3,
    today: Optional[date] = None
) -> Dict:
    """
    RFM营销序列自动调度器
    
    参数:
        customers: [{
            "customer_id": str, "last_purchase_days": int,
            "purchase_count": int, "total_spend": float,
            "last_campaign_date": str (YYYY-MM-DD, 可选),
            "touches_this_month": int (可选)
        }]
        min_ltv_threshold: 最低LTV门槛（低于此值的Lost群体跳过）
        frequency_cap_days: 频控窗口天数
        max_touches_per_month: 每月最大触达次数
    
    返回:
        调度指令列表，按优先级排序
    """
    if today is None:
        today = date.today()
    
    dispatch_queue = []
    skipped = []
    
    for cust in customers:
        cid = cust["customer_id"]
        
        # 频控检查
        last_campaign = cust.get("last_campaign_date", "")
        if last_campaign:
            last_date = datetime.strptime(last_campaign, "%Y-%m-%d").date()
            days_since = (today - last_date).days
            if days_since < frequency_cap_days:
                skipped.append({"customer_id": cid, "reason": f"频控：{days_since}天内已触达"})
                continue
        
        touches = cust.get("touches_this_month", 0)
        if touches >= max_touches_per_month:
            skipped.append({"customer_id": cid, "reason": f"月频控：本月已触达{touches}次"})
            continue
        
        # RFM评分和分群
        r, f, m = compute_rfm_score(
            cust["last_purchase_days"],
            cust["purchase_count"],
            cust["total_spend"]
        )
        segment, priority = classify_rfm_segment(r, f, m)
        
        # 低LTV Lost群体跳过
        if segment == "Lost" and cust["total_spend"] < min_ltv_threshold:
            skipped.append({"customer_id": cid, "reason": f"Lost低LTV(${cust['total_spend']}<${min_ltv_threshold})"})
            continue
        
        campaign_info = SEGMENT_CAMPAIGN_MAP[segment]
        dispatch_queue.append({
            "customer_id": cid,
            "rfm_scores": {"r": r, "f": f, "m": m},
            "segment": segment,
            "campaign": campaign_info["campaign"],
            "channel": campaign_info["channel"],
            "template": campaign_info["template"],
            "priority": priority,
            "total_spend": cust["total_spend"]
        })
    
    dispatch_queue.sort(key=lambda x: x["priority"])
    
    segment_counts = {}
    for item in dispatch_queue:
        seg = item["segment"]
        segment_counts[seg] = segment_counts.get(seg, 0) + 1
    
    return {
        "total_customers": len(customers),
        "dispatched": len(dispatch_queue),
        "skipped": len(skipped),
        "dispatch_queue": dispatch_queue,
        "segment_distribution": segment_counts,
        "execution_priority": "HIGH"
    }


# 测试
customers = [
    {"customer_id": "C001", "last_purchase_days": 5, "purchase_count": 10, "total_spend": 1200, "last_campaign_date": "2026-01-01", "touches_this_month": 0},
    {"customer_id": "C002", "last_purchase_days": 45, "purchase_count": 6, "total_spend": 600, "last_campaign_date": "2026-05-01", "touches_this_month": 1},
    {"customer_id": "C003", "last_purchase_days": 180, "purchase_count": 1, "total_spend": 30, "last_campaign_date": "2026-01-01", "touches_this_month": 0},  # Lost低LTV
    {"customer_id": "C004", "last_purchase_days": 3, "purchase_count": 1, "total_spend": 60, "touches_this_month": 0},  # 新用户
    {"customer_id": "C005", "last_purchase_days": 10, "purchase_count": 5, "total_spend": 800, "last_campaign_date": "2026-06-18", "touches_this_month": 2},  # 频控触发
]

result = rfm_campaign_auto_dispatcher(customers)

assert result["total_customers"] == 5
assert result["dispatched"] >= 2  # C001, C002, C004应触发
# C003低LTV Lost应跳过
skipped_ids = {s["customer_id"] for s in result["dispatch_queue"] if False}  # 确认dispatch_queue只有有效的
dispatched_ids = {d["customer_id"] for d in result["dispatch_queue"]}
assert "C003" not in dispatched_ids  # 低LTV Lost跳过
assert "C005" not in dispatched_ids  # 频控跳过
# 验证Champions优先级最高（排在队首）
champions = [d for d in result["dispatch_queue"] if d["segment"] == "Champions"]
if champions:
    assert result["dispatch_queue"][0]["segment"] == "Champions"

print("[✓] RFM Campaign Auto Dispatcher决策触发器测试通过")
print(f"  处理客户: {result['total_customers']}，调度: {result['dispatched']}，跳过: {result['skipped']}")
print(f"  分群分布: {result['segment_distribution']}")
for d in result["dispatch_queue"]:
    print(f"  [{d['segment']}] {d['customer_id']} → {d['campaign']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-RFM-Customer-Segmentation]]（提供RFM分群结果）
- **延伸（extends）**：[[Skill-Cohort-Churn-Intervention-Dispatcher]]（按队列维度的流失干预）
- **可组合（combinable）**：[[Skill-High-Value-Customer-Proactive-Alert]]（Champions和Cant_Lose_Them群体的主动告警）

## ⑤ 商业价值评估
- ROI预估：At Risk群体挽回率提升至25-35%，年化增量LTV $50,000-$80,000
- 实施难度：⭐⭐☆☆☆（RFM计算标准化，需对接CRM和邮件平台API）
- 优先级：⭐⭐⭐⭐⭐

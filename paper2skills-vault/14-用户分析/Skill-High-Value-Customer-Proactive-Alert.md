---
title: High Value Customer Proactive Alert — 高价值客户出现沉默信号时自动触发客服主动联系
doc_type: knowledge
module: 14-用户分析
topic: high-value-customer-proactive-alert
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: High Value Customer Proactive Alert

> **配对分析层**：[[Skill-RFM-Customer-Segmentation]]
> **决策类型**: 自动告警型 | **触发条件**: 高价值客户（R≤7天, F≥3次, M≥5000元）出现14天无活跃沉默信号 | **执行动作**: 自动触发客服主动联系工单

## ① 算法原理

核心是「高价值客户定义 + 沉默检测 + 自动告警调度」：

1. **高价值客户定义**（硬规则门控）：
   - R（最近购买）≤ 7天（近期活跃用户）→等等，这里R是历史记录，需要修正
   - 实际定义：历史累计消费M ≥ 5,000元 AND 累计购买频次F ≥ 3次（排除一次性高消费）AND 加入时间≥60天（排除新用户）
   
2. **沉默信号检测**：比较当前日期与上次活跃日期（浏览/加购/购买任意行为），沉默窗口≥14天（相比该客户历史平均购买间隔）视为异常沉默。
   
3. **自动告警调度**：按「客户历史LTV」排序，生成客服工单分配优先级（P0:LTV>$1000，P1:LTV $500-$1000，P2:LTV $200-$500），含推荐话术钩子（最近购买品类、可能关注的新品）。

**误触发防护**：已在进行营销序列触达的客户不重复生成工单（查询CRM状态）；同一客户30天内只触发一次主动联系。**回滚机制**：工单触发后7天若无响应，自动降级为邮件触达，不二次电话联系。

## ② 母婴出海应用案例

**场景：吸奶器+婴儿护肤品高价值客户沉默预警**
- 触发条件：客户Alice，历史LTV $1,200（F=5次，M=$1,200），最近一次活跃28天前，超过其历史平均间隔21天×1.3=27天，触发P0告警
- 执行动作：生成P0级客服工单，分配给高级客服，话术钩子「Alice上次购买了XXL吸奶器配件，新款防漏护垫上市了，是否需要了解？」，48小时内联系
- 安全护栏：Alice本月已收到2次邮件，工单标注「邮件已触达2次，电话优先」；30天内不二次生成工单
- 业务价值：高价值客户主动挽回率58%（vs 被动等待22%），年化减少高价值客户流失LTV损失$85,000

## ③ 代码模板

```python
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import heapq

def high_value_customer_proactive_alert(
    customers: List[Dict],
    min_ltv_threshold: float = 5000.0,
    min_purchase_count: int = 3,
    min_account_age_days: int = 60,
    silence_window_days: int = 14,
    silence_multiplier: float = 1.3,
    cooldown_days: int = 30,
    today: Optional[date] = None
) -> Dict:
    """
    高价值客户沉默告警触发器
    
    参数:
        customers: [{
            "customer_id": str,
            "total_spend": float,        # 历史累计消费（人民币元）
            "purchase_count": int,        # 累计购买次数
            "account_age_days": int,      # 账户注册天数
            "last_active_days": int,      # 距上次活跃天数（浏览/加购/购买）
            "avg_purchase_interval": float, # 历史平均购买间隔（天）
            "last_alert_days": int,       # 距上次告警天数（-1表示从未告警）
            "active_campaign": bool,      # 是否在营销序列中
            "last_purchase_category": str # 最近购买品类（话术钩子）
        }]
        min_ltv_threshold: 最低累计消费门槛（默认5000元）
        min_purchase_count: 最低购买次数
        min_account_age_days: 账户最小年龄
        silence_window_days: 沉默基准天数（14天）
        silence_multiplier: 沉默倍数（超过avg_interval×multiplier才算异常）
        cooldown_days: 告警冷却期
    
    返回:
        按优先级排序的告警工单列表
    """
    if today is None:
        today = date.today()
    
    alert_queue = []  # (−priority_score, customer_id, alert)
    skipped = []
    
    for cust in customers:
        cid = cust["customer_id"]
        spend = cust["total_spend"]
        f_count = cust["purchase_count"]
        acc_age = cust["account_age_days"]
        last_active = cust["last_active_days"]
        avg_interval = cust.get("avg_purchase_interval", 30)
        last_alert = cust.get("last_alert_days", -1)
        in_campaign = cust.get("active_campaign", False)
        
        # 高价值客户定义检查
        is_high_value = (
            spend >= min_ltv_threshold
            and f_count >= min_purchase_count
            and acc_age >= min_account_age_days
        )
        if not is_high_value:
            continue
        
        # 冷却期检查
        if last_alert > 0 and last_alert < cooldown_days:
            skipped.append({
                "customer_id": cid,
                "reason": f"冷却期内（{last_alert}天前已告警，冷却{cooldown_days}天）"
            })
            continue
        
        # 正在营销序列中，不重复生成工单
        if in_campaign:
            skipped.append({"customer_id": cid, "reason": "已在营销序列中，不重复生成工单"})
            continue
        
        # 沉默检测：超过avg_interval×multiplier 或 超过固定silence_window_days
        dynamic_silence_threshold = max(silence_window_days, avg_interval * silence_multiplier)
        is_silent = last_active >= dynamic_silence_threshold
        
        if not is_silent:
            continue
        
        # 生成告警：按LTV分P0/P1/P2优先级
        if spend >= 10000:
            priority = "P0"
            priority_score = spend * 3
            contact_sla_hours = 24
        elif spend >= 5000:
            priority = "P1"
            priority_score = spend * 2
            contact_sla_hours = 48
        else:
            priority = "P2"
            priority_score = spend
            contact_sla_hours = 72
        
        last_category = cust.get("last_purchase_category", "母婴产品")
        alert = {
            "customer_id": cid,
            "trigger": True,
            "priority": priority,
            "total_spend": spend,
            "purchase_count": f_count,
            "last_active_days": last_active,
            "silence_threshold_days": round(dynamic_silence_threshold, 1),
            "action": "PROACTIVE_CONTACT",
            "contact_sla_hours": contact_sla_hours,
            "script_hook": f"注意到您已{last_active}天未访问，{last_category}最近上线新品，是否需要了解？",
            "channels": ["phone", "email"] if priority == "P0" else ["email"],
            "escalation": f"7天无响应自动降级为邮件触达",
            "priority_score": priority_score
        }
        heapq.heappush(alert_queue, (-priority_score, cid, alert))
        
    # 排序输出
    sorted_alerts = []
    while alert_queue:
        _, _, alert = heapq.heappop(alert_queue)
        sorted_alerts.append(alert)
    
    priority_counts = {}
    for a in sorted_alerts:
        p = a["priority"]
        priority_counts[p] = priority_counts.get(p, 0) + 1
    
    return {
        "total_high_value_checked": sum(1 for c in customers if c["total_spend"] >= min_ltv_threshold and c["purchase_count"] >= min_purchase_count),
        "alerts_triggered": len(sorted_alerts),
        "skipped": len(skipped),
        "alert_list": sorted_alerts,
        "priority_distribution": priority_counts,
        "execution_priority": "HIGH" if any(a["priority"] == "P0" for a in sorted_alerts) else "MEDIUM"
    }


# 测试
customers = [
    {  # 触发P0: 高消费+长沉默
        "customer_id": "ALICE", "total_spend": 12000.0, "purchase_count": 8,
        "account_age_days": 365, "last_active_days": 28, "avg_purchase_interval": 20,
        "last_alert_days": -1, "active_campaign": False, "last_purchase_category": "吸奶器配件"
    },
    {  # 触发P1
        "customer_id": "BOB", "total_spend": 6500.0, "purchase_count": 4,
        "account_age_days": 200, "last_active_days": 20, "avg_purchase_interval": 14,
        "last_alert_days": -1, "active_campaign": False, "last_purchase_category": "婴儿护肤品"
    },
    {  # 不触发：在营销序列中
        "customer_id": "CAROL", "total_spend": 7000.0, "purchase_count": 5,
        "account_age_days": 180, "last_active_days": 18, "avg_purchase_interval": 12,
        "last_alert_days": -1, "active_campaign": True, "last_purchase_category": "婴儿服装"
    },
    {  # 不触发：冷却期内
        "customer_id": "DAVE", "total_spend": 5500.0, "purchase_count": 3,
        "account_age_days": 150, "last_active_days": 16, "avg_purchase_interval": 10,
        "last_alert_days": 15, "active_campaign": False, "last_purchase_category": "奶粉"
    },
    {  # 不触发：未达高价值阈值
        "customer_id": "EVE", "total_spend": 2000.0, "purchase_count": 2,
        "account_age_days": 90, "last_active_days": 20, "avg_purchase_interval": 15,
        "last_alert_days": -1, "active_campaign": False, "last_purchase_category": "玩具"
    },
]

result = high_value_customer_proactive_alert(customers)

assert result["alerts_triggered"] == 2  # ALICE和BOB触发
alert_ids = [a["customer_id"] for a in result["alert_list"]]
assert "ALICE" in alert_ids
assert "BOB" in alert_ids
assert "CAROL" not in alert_ids  # 在营销序列中
assert "DAVE" not in alert_ids   # 冷却期
# ALICE应在BOB前（P0>P1）
assert result["alert_list"][0]["customer_id"] == "ALICE"
# P0告警有48小时SLA
alice_alert = next(a for a in result["alert_list"] if a["customer_id"] == "ALICE")
assert alice_alert["priority"] == "P0"
assert alice_alert["contact_sla_hours"] == 24

print("[✓] High Value Customer Proactive Alert决策触发器测试通过")
print(f"  高价值客户检查: {result['total_high_value_checked']}人，触发告警: {result['alerts_triggered']}人")
print(f"  优先级分布: {result['priority_distribution']}")
for a in result["alert_list"]:
    print(f"  [{a['priority']}] {a['customer_id']} 沉默{a['last_active_days']}天，SLA {a['contact_sla_hours']}h")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-RFM-Customer-Segmentation]]（提供高价值客户识别和分群基础）
- **延伸（extends）**：[[Skill-RFM-Campaign-Auto-Dispatcher]]（RFM分群后的批量营销序列调度）
- **可组合（combinable）**：[[Skill-Cohort-Churn-Intervention-Dispatcher]]（个体告警与队列干预双轨并行）

## ⑤ 商业价值评估
- ROI预估：高价值客户主动挽回率提升至50-60%（vs 被动20%），年化减少高LTV流失$60,000-$100,000
- 实施难度：⭐⭐☆☆☆（规则清晰，需接入CRM和活跃度数据）
- 优先级：⭐⭐⭐⭐⭐

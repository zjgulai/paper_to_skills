---
title: Cohort Churn Intervention Dispatcher — 低留存队列自动触发差异化挽回干预序列
doc_type: knowledge
module: 14-用户分析
topic: cohort-churn-intervention-dispatcher
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Cohort Churn Intervention Dispatcher

> **配对分析层**：[[Skill-Cohort-Retention-Analysis]]
> **决策类型**: 自动触发型 | **触发条件**: 30日留存率<40%或<20% | **执行动作**: 自动触发挽回邮件序列或人工客服介入

## ① 算法原理

核心是「队列留存分层 + 优先级队列 + 干预序列调度」：

1. **留存分层分类**：
   - 30日留存率 < 20%（危急）：触发「人工客服主动介入」，优先级P0
   - 20% ≤ 留存率 < 40%（风险）：触发「自动挽回邮件序列（3封，间隔3/7/14天）」，优先级P1
   - 40% ≤ 留存率 < 60%（关注）：触发「产品使用提示推送（轻触达）」，优先级P2
   - ≥ 60%（健康）：无干预，进入正常运营

2. **优先级队列**：多个危急队列同时存在时，按「队列规模 × 历史LTV」排序，优先处理价值高的队列。

3. **干预序列调度**：邮件序列自动生成发送时间点，含个性化钩子（队列首次购买品类、购买间隔）。

**误触发防护**：队列规模需≥50人才触发，避免小样本噪声。**回滚机制**：干预后14天留存率对比对照组，若无显著改善（p>0.1）停止该序列后续发送。

## ② 母婴出海应用案例

**场景：婴儿辅食品类用户队列挽回**
- 触发条件：2026年3月入组队列（宝宝6个月辅食期），30日留存率17%（危急），队列规模128人，历史平均LTV $180
- 执行动作：P0优先级，客服48小时内主动联系，话术聚焦「宝宝成长阶段营养需求变化」，推送辅食产品续购券
- 安全护栏：同一用户48小时内不重复触达；客服介入记录存入CRM，避免重复打扰
- 业务价值：P0干预平均留存率提升至35%，年化挽回LTV约$42,000

## ③ 代码模板

```python
import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Optional

def cohort_churn_intervention_dispatcher(
    cohorts: List[Dict],
    today: Optional[datetime] = None,
    critical_threshold: float = 0.20,
    risk_threshold: float = 0.40,
    watch_threshold: float = 0.60,
    min_cohort_size: int = 50
) -> Dict:
    """
    队列流失干预调度器
    
    参数:
        cohorts: [{
            "cohort_id": str, "cohort_month": str,
            "retention_30d": float, "size": int, "avg_ltv": float
        }]
        critical_threshold: 危急留存阈值（默认0.20）
        risk_threshold: 风险留存阈值（默认0.40）
        min_cohort_size: 最小队列规模门控
    
    返回:
        {"interventions": [...], "priority_queue": [...]}
    """
    if today is None:
        today = datetime.now()
    
    # 优先级队列（负值因为heapq是最小堆，需要最大优先级在前）
    # 格式：(-priority_score, cohort_id, intervention)
    pq = []
    all_interventions = []
    
    for cohort in cohorts:
        cid = cohort["cohort_id"]
        retention = cohort["retention_30d"]
        size = cohort["size"]
        ltv = cohort.get("avg_ltv", 100)
        
        # 规模门控
        if size < min_cohort_size:
            all_interventions.append({
                "cohort_id": cid,
                "trigger": False,
                "reason": f"队列规模{size}<{min_cohort_size}，样本不足"
            })
            continue
        
        # 分层判断
        if retention < critical_threshold:
            # P0：人工客服介入
            priority_score = size * ltv * (critical_threshold - retention)
            action = {
                "cohort_id": cid,
                "trigger": True,
                "level": "CRITICAL",
                "priority": "P0",
                "retention_30d": retention,
                "intervention_type": "MANUAL_CUSTOMER_SERVICE",
                "sla_hours": 48,
                "script_hint": f"队列{cid}用户{size}人，留存率{retention:.0%}，重点询问使用反馈",
                "schedule": [
                    {"action": "cs_contact", "within_hours": 48, "channel": "email+phone"}
                ],
                "priority_score": priority_score
            }
            heapq.heappush(pq, (-priority_score, cid, action))
            all_interventions.append(action)
        
        elif retention < risk_threshold:
            # P1：自动邮件挽回序列
            priority_score = size * ltv * (risk_threshold - retention)
            email_schedule = [
                {"step": 1, "day_offset": 0,  "subject": "我们想念你！专属优惠等你来"},
                {"step": 2, "day_offset": 3,  "subject": "宝宝成长阶段，这些产品你需要"},
                {"step": 3, "day_offset": 7,  "subject": "最后提醒：限时专属折扣即将到期"},
                {"step": 4, "day_offset": 14, "subject": "你的专属客服想和你聊聊"},
            ]
            schedule_dates = [
                {**e, "send_date": (today + timedelta(days=e["day_offset"])).strftime("%Y-%m-%d")}
                for e in email_schedule
            ]
            action = {
                "cohort_id": cid,
                "trigger": True,
                "level": "RISK",
                "priority": "P1",
                "retention_30d": retention,
                "intervention_type": "EMAIL_WINBACK_SEQUENCE",
                "email_sequence": schedule_dates,
                "stop_condition": "14天后留存率无显著提升(p>0.1)停止后续发送",
                "priority_score": priority_score
            }
            heapq.heappush(pq, (-priority_score, cid, action))
            all_interventions.append(action)
        
        elif retention < watch_threshold:
            # P2：轻触达推送
            action = {
                "cohort_id": cid,
                "trigger": True,
                "level": "WATCH",
                "priority": "P2",
                "retention_30d": retention,
                "intervention_type": "LIGHT_PUSH_NOTIFICATION",
                "push_content": "产品使用技巧推送",
                "frequency": "每周1次，共2次"
            }
            all_interventions.append(action)
        
        else:
            all_interventions.append({
                "cohort_id": cid,
                "trigger": False,
                "level": "HEALTHY",
                "retention_30d": retention,
                "action": "NO_INTERVENTION"
            })
    
    # 输出优先级排序列表
    sorted_queue = []
    temp_pq = list(pq)
    heapq.heapify(temp_pq)
    while temp_pq:
        score, cid, action = heapq.heappop(temp_pq)
        sorted_queue.append({"priority_score": -score, "cohort_id": cid, "level": action["level"]})
    
    triggered_count = sum(1 for i in all_interventions if i.get("trigger"))
    return {
        "total_cohorts": len(cohorts),
        "triggered": triggered_count,
        "interventions": all_interventions,
        "priority_queue": sorted_queue,
        "execution_summary": f"P0:{sum(1 for i in all_interventions if i.get('priority')=='P0')}个队列需人工介入"
    }


# 测试
cohorts = [
    {"cohort_id": "2026-03", "cohort_month": "2026-03", "retention_30d": 0.17, "size": 128, "avg_ltv": 180},
    {"cohort_id": "2026-02", "cohort_month": "2026-02", "retention_30d": 0.32, "size": 256, "avg_ltv": 150},
    {"cohort_id": "2026-01", "cohort_month": "2026-01", "retention_30d": 0.55, "size": 312, "avg_ltv": 200},
    {"cohort_id": "2025-12", "cohort_month": "2025-12", "retention_30d": 0.72, "size": 200, "avg_ltv": 220},
    {"cohort_id": "2026-04", "cohort_month": "2026-04", "retention_30d": 0.15, "size": 30, "avg_ltv": 90},  # 规模不足
]
result = cohort_churn_intervention_dispatcher(cohorts)

assert result["total_cohorts"] == 5
assert result["triggered"] == 3
# P0队列验证
p0_actions = [i for i in result["interventions"] if i.get("priority") == "P0"]
assert len(p0_actions) == 1
assert p0_actions[0]["cohort_id"] == "2026-03"
# 规模不足的队列不应触发
small_cohort = next(i for i in result["interventions"] if i["cohort_id"] == "2026-04")
assert small_cohort["trigger"] == False

print("[✓] Cohort Churn Intervention Dispatcher决策触发器测试通过")
print(f"  总队列: {result['total_cohorts']}，触发干预: {result['triggered']}")
print(f"  {result['execution_summary']}")
print(f"  优先级队列(Top3): {result['priority_queue'][:3]}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Cohort-Retention-Analysis]]（提供各队列30/60/90日留存率信号）
- **延伸（extends）**：[[Skill-User-LTV-Financial-Bridge]]（按LTV加权优先级得分）
- **可组合（combinable）**：[[Skill-Customer-Journey-Analytics]]（干预效果追踪，A/B对比验证）

## ⑤ 商业价值评估
- ROI预估：P0干预将危急队列留存率从<20%提升至30-40%，年化挽回LTV约$30,000-$60,000
- 实施难度：⭐⭐☆☆☆（规则清晰，需对接CRM和邮件平台）
- 优先级：⭐⭐⭐⭐⭐

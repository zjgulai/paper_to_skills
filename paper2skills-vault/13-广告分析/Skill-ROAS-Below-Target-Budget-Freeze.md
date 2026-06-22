---
title: ROAS-Below-Target-Budget-Freeze — ROAS连续3天低于目标自动冻结广告组预算触发创意审查
doc_type: knowledge
module: 13-广告分析
topic: roas-below-target-budget-freeze
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: ROAS-Below-Target-Budget-Freeze

> **配对分析层**：[[Skill-Ad-ROAS-Attribution-Analysis]]
> **决策类型**: 自动触发型 | **触发条件**: ROAS 连续3天低于目标值 | **执行动作**: 冻结广告组预算，触发创意审查流程

## ① 算法原理

核心是「ROAS 滚动监控 + 连续不达标检测 + 预算冻结 + 创意审查触发」：

1. **ROAS 计算**：ROAS = 广告带来的收入 / 广告花费，按广告组粒度每日计算。
2. **连续检测**：检查最近 N 天 ROAS 是否连续低于目标值（默认 N=3）。
3. **分级响应**：
   - 连续 2 天低于目标：发送预警通知，建议优化
   - 连续 3 天低于目标：冻结该广告组预算，触发创意审查
   - 连续 5 天低于目标：暂停广告组，升级到广告负责人
4. **冻结机制**：冻结期间预算设为 $0（或最低保留出价 $0.10/day），保留广告组数据。
5. **解冻条件**：创意审查完成且新素材上线后，系统自动解冻并恢复原预算。

**误触发防护**：账户整体 ROAS 同期下降 > 20%（行业周期性因素）时，放宽触发阈值 15%，避免因季节性误冻结。

## ② 母婴出海应用案例

**场景：婴儿纸尿裤广告组 ROAS 持续低迷**
- 触发条件：广告组「Diaper-SP-Brand-01」连续 3 天 ROAS = [1.8, 1.6, 1.5]，目标 ROAS = 3.0
- 执行动作：
  - 冻结广告组日预算（$0/day），停止继续亏损投放
  - 自动生成审查任务：检查点击率（CTR 从 0.8% 跌至 0.3%）、搜索词报告（错误触发宽泛词）
  - 推送通知给广告负责人，附 ROAS 趋势图和诊断建议
- 根因发现：主图素材过时（新竞品上线），优化主图后 CTR 恢复至 0.9%
- 解冻后：ROAS 回升至 3.4，节省无效投放 $4,200（冻结期间）
- 业务价值：年化减少无效广告投放 $50,000，整体广告 ACoS 降低 4 个百分点

## ③ 代码模板

```python
from typing import Dict, List, Optional
from datetime import datetime

def roas_below_target_budget_freeze(
    ad_groups: List[Dict],
    now: Optional[datetime] = None,
    warning_days: int = 2,
    freeze_days: int = 3,
    suspend_days: int = 5,
    account_roas_decline_threshold: float = 0.20,
    threshold_relax_factor: float = 0.15
) -> Dict:
    """
    ROAS 低于目标自动冻结执行器
    
    参数:
        ad_groups: [{
            "ad_group_id": str, "ad_group_name": str,
            "daily_roas": List[float],  # 最近N天ROAS，从早到晚排列
            "target_roas": float,
            "daily_budget": float,
            "account_avg_roas_trend": List[float]  # 账户整体ROAS（用于误触发防护）
        }]
        warning_days: 触发预警所需连续低ROAS天数
        freeze_days: 触发冻结所需连续低ROAS天数
        suspend_days: 触发暂停所需连续低ROAS天数
        account_roas_decline_threshold: 账户整体ROAS下降多少比例触发宽松模式
    
    返回:
        {"actions": [...], "stats": {...}}
    """
    if now is None:
        now = datetime.now()
    
    actions = []
    
    for ag in ad_groups:
        agid = ag["ad_group_id"]
        agname = ag.get("ad_group_name", agid)
        daily_roas = ag.get("daily_roas", [])
        target_roas = ag.get("target_roas", 3.0)
        daily_budget = ag.get("daily_budget", 100.0)
        account_roas = ag.get("account_avg_roas_trend", [])
        
        if len(daily_roas) < warning_days:
            actions.append({"ad_group_id": agid, "action": "INSUFFICIENT_DATA",
                            "reason": f"仅{len(daily_roas)}天数据，需至少{warning_days}天"})
            continue
        
        # 误触发防护：检查账户整体ROAS是否也在下降
        adjusted_target = target_roas
        if len(account_roas) >= 3:
            account_recent = sum(account_roas[-3:]) / 3
            account_baseline = sum(account_roas[:3]) / 3 if len(account_roas) >= 6 else account_recent
            account_decline = max(0, (account_baseline - account_recent) / max(account_baseline, 0.1))
            if account_decline > account_roas_decline_threshold:
                adjusted_target = target_roas * (1 - threshold_relax_factor)
        
        # 计算连续不达标天数
        consecutive_fail = 0
        for roas in reversed(daily_roas):
            if roas < adjusted_target:
                consecutive_fail += 1
            else:
                break
        
        current_roas = daily_roas[-1]
        roas_gap_pct = (adjusted_target - current_roas) / adjusted_target
        
        if consecutive_fail >= suspend_days:
            action = {
                "ad_group_id": agid,
                "ad_group_name": agname,
                "action": "SUSPEND_AD_GROUP",
                "consecutive_fail_days": consecutive_fail,
                "current_roas": round(current_roas, 2),
                "target_roas": target_roas,
                "adjusted_target": round(adjusted_target, 2),
                "daily_budget_frozen": daily_budget,
                "severity": "CRITICAL",
                "steps": [
                    "暂停广告组（预算设为$0）",
                    "升级给广告负责人，要求24h内审查",
                    "检查：搜索词报告、创意素材、出价策略、竞品动态",
                    "重构广告组结构后重新启动"
                ],
                "estimated_waste_if_not_acted": round(daily_budget * consecutive_fail, 2)
            }
        elif consecutive_fail >= freeze_days:
            action = {
                "ad_group_id": agid,
                "ad_group_name": agname,
                "action": "FREEZE_BUDGET",
                "consecutive_fail_days": consecutive_fail,
                "current_roas": round(current_roas, 2),
                "target_roas": target_roas,
                "adjusted_target": round(adjusted_target, 2),
                "daily_budget_frozen": daily_budget,
                "severity": "HIGH",
                "freeze_budget_to": 0.10,  # 保留最低出价
                "review_checklist": [
                    "CTR变化趋势（近7天）",
                    "搜索词报告（排除无关词）",
                    "竞品广告动态",
                    "主图/标题素材A/B测试结果",
                    "出价竞争力（与竞品对比）"
                ],
                "unfreeze_condition": "新创意上线 + 48h内ROAS≥目标值×0.9",
                "estimated_waste_saved": round(daily_budget * 3, 2),  # 冻结后节省3天无效投放
                "roas_gap_pct": round(roas_gap_pct * 100, 1)
            }
        elif consecutive_fail >= warning_days:
            action = {
                "ad_group_id": agid,
                "ad_group_name": agname,
                "action": "WARNING_ALERT",
                "consecutive_fail_days": consecutive_fail,
                "current_roas": round(current_roas, 2),
                "target_roas": target_roas,
                "severity": "MEDIUM",
                "suggestion": "建议检查CTR和搜索词，若明天仍低于目标将自动冻结预算",
                "roas_trend": daily_roas[-3:]
            }
        else:
            action = {
                "ad_group_id": agid,
                "ad_group_name": agname,
                "action": "HEALTHY",
                "current_roas": round(current_roas, 2),
                "target_roas": target_roas,
                "consecutive_fail_days": consecutive_fail
            }
        
        actions.append(action)
    
    severity_counts = {}
    for a in actions:
        sev = a.get("severity", "HEALTHY")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    return {
        "total_ad_groups": len(ad_groups),
        "frozen": sum(1 for a in actions if a.get("action") == "FREEZE_BUDGET"),
        "suspended": sum(1 for a in actions if a.get("action") == "SUSPEND_AD_GROUP"),
        "actions": actions,
        "severity_summary": severity_counts
    }


# 测试
ad_groups = [
    {
        "ad_group_id": "AG001", "ad_group_name": "Diaper-SP-Brand-01",
        "daily_roas": [3.2, 2.8, 1.8, 1.6, 1.5],  # 连续3天<3.0
        "target_roas": 3.0, "daily_budget": 200.0,
        "account_avg_roas_trend": [3.5, 3.4, 3.3, 3.2, 3.3, 3.4]  # 账户整体稳定
    },
    {
        "ad_group_id": "AG002", "ad_group_name": "Bottle-SP-Generic",
        "daily_roas": [2.5, 2.8, 2.7, 3.1, 2.9],  # 最近1天<3.0，不触发
        "target_roas": 3.0, "daily_budget": 150.0,
        "account_avg_roas_trend": [3.5, 3.4, 3.3, 3.2, 3.3, 3.4]
    },
    {
        "ad_group_id": "AG003", "ad_group_name": "Toy-Auto-Campaign",
        "daily_roas": [1.2, 1.0, 0.9, 0.8, 0.7, 0.6],  # 连续5天低
        "target_roas": 2.5, "daily_budget": 80.0,
        "account_avg_roas_trend": [3.5, 3.4, 3.3, 3.2, 3.3, 3.4]
    },
]

result = roas_below_target_budget_freeze(ad_groups)

assert result["total_ad_groups"] == 3
action_map = {a["ad_group_id"]: a["action"] for a in result["actions"]}
assert action_map["AG001"] == "FREEZE_BUDGET"
assert action_map["AG002"] == "HEALTHY"  # 只有1天低，不触发
assert action_map["AG003"] == "SUSPEND_AD_GROUP"

freeze_action = next(a for a in result["actions"] if a["ad_group_id"] == "AG001")
assert freeze_action["estimated_waste_saved"] == 600.0  # 200 * 3天

print("[✓] ROAS Below Target Budget Freeze 测试通过")
print(f"  总广告组: {result['total_ad_groups']}，冻结: {result['frozen']}，暂停: {result['suspended']}")
print(f"  严重程度: {result['severity_summary']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Ad-ROAS-Attribution-Analysis]]（提供归因准确的 ROAS 数值）
- **延伸（extends）**：[[Skill-Keyword-Bid-Auto-Adjuster]]（冻结前先尝试出价调整）
- **可组合（combinable）**：[[Skill-Promo-Inventory-Pulse-Auto-Trigger]]（广告降速与库存保护协同）

## ⑤ 商业价值评估
- **ROI量化**：年化减少无效广告投放 $50,000，整体广告 ACoS 降低 4 个百分点，等效 GMV 增量约 $80,000
- **实施难度**：⭐⭐☆☆☆（需广告 API 读写权限 + 通知系统）
- **优先级**：⭐⭐⭐⭐⭐（广告 ACoS 是影响利润率的核心杠杆）

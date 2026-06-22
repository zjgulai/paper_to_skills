---
title: Uplift Intervention Queue Optimizer — 在预算约束下按CATE排序生成最优干预名单
doc_type: knowledge
module: 06-增长模型
topic: uplift-intervention-queue-optimizer
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Uplift Intervention Queue Optimizer

> **配对分析层**：[[Skill-Uplift-Churn-Prediction]]
> **决策类型**: 优化排序型 | **触发条件**: Uplift模型输出CATE分数后 | **执行动作**: 在预算约束下按CATE从高到低排序，生成最优干预名单和优先级队列

## ① 算法原理

核心是「CATE排序 + 预算约束优化 + 干预优先级队列」：

1. **CATE排序**：Uplift模型输出每个用户的CATE（条件平均干预效应），即「给此用户施加干预相比不干预的额外挽回概率」。CATE越高，干预效果越好，越应该优先施加干预。

2. **预算约束优化**（背包问题变体）：
   - 每次干预有固定成本（如邮件$0.5，优惠券$5，客服电话$20）
   - 总预算固定
   - 目标：在预算内最大化总期望挽回价值（CATE × 用户LTV × 干预成本）
   - 简化解：贪心算法按CATE/成本比排序，顺序填充直到预算耗尽

3. **Persuadable Zone过滤**：仅保留「可干预者」（Persuadables）——CATE>0且不属于「无论如何都会留存（Sure Things）」或「无论如何都会流失（Lost Causes）」的用户。

**误触发防护**：CATE置信区间下界<0的用户降级为「轻触达」干预（低成本），不施加高成本干预。**预算安全**：最后预留10%缓冲，不完全用尽预算。

## ② 母婴出海应用案例

**场景：母婴品牌月度留存干预预算分配**
- 触发条件：Uplift模型识别出580名用户CATE>0，月度干预预算$2,000
- 执行动作：Top 100 CATE用户分配「$5优惠券干预」($500)，次200人分配「邮件序列干预」($100)，其余CATE>0者进入「轻触达推送」($180)，剩余$1,220保留
- 安全护栏：每次干预批次不超过500人；CATE<0.05的用户归为「观察组」不干预
- 业务价值：每投入$1干预成本产出$4.2预期留存价值，月化干预ROI 320%

## ③ 代码模板

```python
import numpy as np
from typing import Dict, List, Optional, Tuple

def uplift_intervention_queue_optimizer(
    users: List[Dict],
    total_budget: float,
    intervention_tiers: Optional[List[Dict]] = None,
    cate_min_threshold: float = 0.05,
    budget_reserve_ratio: float = 0.10
) -> Dict:
    """
    Uplift干预名单优化调度器
    
    参数:
        users: [{
            "user_id": str,
            "cate": float,           # 条件平均干预效应（挽回概率提升）
            "cate_ci_lower": float,  # CATE 95%CI下界
            "ltv": float,            # 预测LTV
            "segment": str           # 可选：Sure_Thing/Persuadable/Lost_Cause/Sleeping_Dog
        }]
        total_budget: 总干预预算
        intervention_tiers: 干预层级定义，默认3档
        cate_min_threshold: CATE最低阈值（低于此值不列入高优先级干预）
        budget_reserve_ratio: 预算缓冲比例（不完全用尽）
    
    返回:
        按优先级排序的干预名单
    """
    if intervention_tiers is None:
        intervention_tiers = [
            {"tier": "HIGH",   "cost_per_user": 5.0,  "channel": "coupon+email", "cate_min": 0.15},
            {"tier": "MEDIUM", "cost_per_user": 0.5,  "channel": "email_sequence", "cate_min": 0.05},
            {"tier": "LOW",    "cost_per_user": 0.1,  "channel": "push_notification", "cate_min": 0.01},
        ]
    
    usable_budget = total_budget * (1 - budget_reserve_ratio)
    
    # 1. 过滤：仅处理Persuadables（CATE>0且CI下界不要求，但CATE<0排除）
    eligible = []
    for u in users:
        cate = u["cate"]
        cate_ci_lower = u.get("cate_ci_lower", cate * 0.5)  # 默认保守估计
        segment = u.get("segment", "Unknown")
        
        # 排除Sure Things和Lost Causes
        if segment in ["Sure_Thing", "Lost_Cause", "Sleeping_Dog"]:
            continue
        if cate <= 0:
            continue
        
        eligible.append({
            "user_id": u["user_id"],
            "cate": cate,
            "cate_ci_lower": cate_ci_lower,
            "ltv": u.get("ltv", 100),
            "expected_value": cate * u.get("ltv", 100)  # 期望挽回价值
        })
    
    # 2. 按CATE排序（高到低）
    eligible.sort(key=lambda x: x["cate"], reverse=True)
    
    # 3. 贪心分配：按CATE/成本比分配干预层级
    remaining_budget = usable_budget
    assignments = []
    
    for user in eligible:
        if remaining_budget <= 0:
            break
        
        cate = user["cate"]
        cate_lower = user["cate_ci_lower"]
        
        # 选择干预层级（CI下界<0的用降级处理）
        if cate_lower <= 0:
            eligible_tiers = [t for t in intervention_tiers if t["tier"] == "LOW"]
        else:
            eligible_tiers = [t for t in intervention_tiers if t["cate_min"] <= cate]
            if not eligible_tiers:
                eligible_tiers = [intervention_tiers[-1]]  # 最低档兜底
        
        # 选最高可用档（贪心：尽可能高效利用资源）
        eligible_tiers.sort(key=lambda t: t["cost_per_user"], reverse=True)
        selected_tier = None
        for tier in eligible_tiers:
            if tier["cost_per_user"] <= remaining_budget:
                selected_tier = tier
                break
        
        if selected_tier is None:
            break
        
        remaining_budget -= selected_tier["cost_per_user"]
        assignments.append({
            "user_id": user["user_id"],
            "cate": round(user["cate"], 4),
            "ltv": user["ltv"],
            "expected_value": round(user["expected_value"], 2),
            "intervention_tier": selected_tier["tier"],
            "intervention_cost": selected_tier["cost_per_user"],
            "channel": selected_tier["channel"]
        })
    
    # 4. 汇总统计
    total_spent = usable_budget - remaining_budget
    total_expected_value = sum(a["expected_value"] for a in assignments)
    tier_counts = {}
    for a in assignments:
        tier = a["intervention_tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    return {
        "total_eligible": len(eligible),
        "total_assigned": len(assignments),
        "budget_used": round(total_spent, 2),
        "budget_reserved": round(remaining_budget + total_budget * budget_reserve_ratio, 2),
        "total_budget": total_budget,
        "total_expected_value": round(total_expected_value, 2),
        "roi": round(total_expected_value / total_spent, 2) if total_spent > 0 else 0,
        "intervention_queue": assignments,
        "tier_distribution": tier_counts,
        "execution_priority": "HIGH"
    }


# 测试
np.random.seed(42)
# 生成模拟用户数据
users = []
for i in range(200):
    cate = max(0, np.random.normal(0.1, 0.08))
    users.append({
        "user_id": f"U{i:04d}",
        "cate": round(cate, 4),
        "cate_ci_lower": round(cate - 0.05, 4),
        "ltv": round(np.random.uniform(80, 500), 2),
        "segment": "Persuadable"
    })
# 加入一些Sure_Things和Lost_Causes（应被过滤）
users.append({"user_id": "SURE001", "cate": 0.9, "cate_ci_lower": 0.8, "ltv": 200, "segment": "Sure_Thing"})
users.append({"user_id": "LOST001", "cate": -0.1, "cate_ci_lower": -0.2, "ltv": 100, "segment": "Lost_Cause"})

result = uplift_intervention_queue_optimizer(users, total_budget=2000.0)

assert result["total_assigned"] > 0
assert result["budget_used"] <= 2000.0 * 0.90  # 不超过可用预算
assert result["roi"] > 0
# Sure_Things和Lost_Causes不在结果中
assigned_ids = {a["user_id"] for a in result["intervention_queue"]}
assert "SURE001" not in assigned_ids
assert "LOST001" not in assigned_ids
# 高CATE用户应在队列前面
if len(result["intervention_queue"]) >= 2:
    assert result["intervention_queue"][0]["cate"] >= result["intervention_queue"][-1]["cate"]

print("[✓] Uplift Intervention Queue Optimizer决策触发器测试通过")
print(f"  总目标用户: {result['total_eligible']}，已分配干预: {result['total_assigned']}")
print(f"  预算使用: ${result['budget_used']:.0f} / ${result['total_budget']:.0f}")
print(f"  期望干预ROI: {result['roi']:.1f}x")
print(f"  层级分布: {result['tier_distribution']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Uplift-Churn-Prediction]]（提供用户级CATE分数）
- **延伸（extends）**：[[Skill-Cohort-Churn-Intervention-Dispatcher]]（队列级干预与用户级Uplift优化协同）
- **可组合（combinable）**：[[Skill-RFM-Campaign-Auto-Dispatcher]]（Uplift优先级 × RFM价值分层双轴筛选）

## ⑤ 商业价值评估
- ROI预估：干预ROI通常达3-5x，相比随机干预节省30-50%干预成本，年化节省$15,000-$30,000
- 实施难度：⭐⭐☆☆☆（Uplift模型已有输出时接入简单）
- 优先级：⭐⭐⭐⭐⭐

---
title: Uplift-Intervention-Priority-Queue — Uplift×LTV加权排序生成有限促销资源干预执行队列
doc_type: knowledge
module: 14-用户分析
topic: uplift-intervention-priority-queue
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Uplift-Intervention-Priority-Queue

> **配对分析层**：[[Skill-Uplift-Churn-Prediction]]
> **决策类型**: 自动触发型 | **触发条件**: Uplift模型评分更新后，促销资源有限（优惠券/客服配额不足以覆盖所有候选用户）| **执行动作**: 按Uplift×LTV加权分数排序，生成优先干预执行队列

## ① 算法原理

核心是「双维度加权排序 + 资源约束截断 + 负Uplift过滤」三步执行流程：

1. **优先级评分计算**：
   `Priority Score = Uplift_Score × LTV_Estimate`
   - Uplift_Score：该用户在促销干预下的增量购买概率（来自因果Uplift模型）
   - LTV_Estimate：用户预测生命周期价值（美元）
   - 乘积代表「单位干预预期增量价值」，是资源分配的核心指标

2. **负Uplift过滤**：Uplift_Score < 0（即干预后反而降低购买概率的用户，"抵触型"）自动排除，这类用户被触达会增加退订率。

3. **资源约束截断**：按Priority Score降序排列，取前N名（N由当前可用优惠券/客服配额决定），形成执行队列；剩余用户进入等待池，下次资源补充时优先从等待池顶部取。

4. **分层标签**：队列按分数分三档——TOP_20%（VIP干预，最优质资源）、MID_60%（标准干预）、BOTTOM_20%（轻量触达），资源差异化投入。

**误触发防护**：Uplift分数置信区间宽度>0.3（不确定性过高）时，该用户降级为观察态，不纳入本批执行。**回滚机制**：干预后21天追踪各分档实际转化率，若与Uplift预测相关性<0.3，触发模型重训警告。

## ② 母婴出海应用案例

**场景：618大促前，有限促销券分配给最有潜力的流失预防用户**
- 触发条件：Uplift模型完成评分，1,500名30天内未购买候选用户；促销预算=500张15%优惠券
- 执行动作：按Priority Score排序，取Top 500名生成执行队列
  - TOP_20%（100人）：Uplift×LTV Top，获VIP礼包+15%券+专属客服
  - MID_60%（300人）：标准15%优惠券+自动邮件
  - BOTTOM_20%（100人）：仅轻量推送，不消耗优惠券
- 安全护栏：排除Uplift<0的115名"抵触型"用户（发券反而会退订），实际可干预=1,385人
- 业务价值：相比随机发券，按Uplift×LTV排序可将500券的增量GMV从$4,200（随机）提升至$6,800（+62%）

## ③ 代码模板

```python
import heapq
from typing import Dict, List, Optional, Tuple

# 分层阈值
TIER_THRESHOLDS = {
    "TOP": 0.80,    # 前20%
    "MID": 0.20,    # 20%-80%
    "BOTTOM": 0.0   # 后20%
}


def uplift_intervention_priority_queue(
    users: List[Dict],
    resource_budget: int,
    uplift_ci_threshold: float = 0.3,
    negative_uplift_filter: bool = True
) -> Dict:
    """
    Uplift×LTV加权干预优先级队列生成器
    
    参数:
        users: [{
            "user_id": str,
            "uplift_score": float,      # 干预增量购买概率 [-1, 1]
            "uplift_ci_width": float,   # Uplift置信区间宽度（不确定性）
            "ltv_estimate": float,      # 预测LTV（美元）
            "last_purchase_days": int,  # 距最近购买天数
        }]
        resource_budget: 可用干预资源数量（优惠券数/客服名额）
        uplift_ci_threshold: Uplift不确定性过滤阈值
        negative_uplift_filter: 是否过滤负Uplift用户
    
    返回:
        {"execution_queue": [...], "waitlist": [...], "excluded": [...], "summary": {...}}
    """
    excluded = []
    candidates = []
    
    for u in users:
        uid = u["user_id"]
        uplift = u["uplift_score"]
        ci_width = u.get("uplift_ci_width", 0.0)
        ltv = u["ltv_estimate"]
        
        # 过滤1：负Uplift（干预抵触型用户）
        if negative_uplift_filter and uplift < 0:
            excluded.append({
                "user_id": uid,
                "reason": f"负Uplift={uplift:.3f}，干预可能导致退订，已排除",
                "uplift_score": uplift
            })
            continue
        
        # 过滤2：高不确定性（置信区间过宽）
        if ci_width > uplift_ci_threshold:
            excluded.append({
                "user_id": uid,
                "reason": f"Uplift不确定性过高（CI宽={ci_width:.2f}>{uplift_ci_threshold}），降级为观察态",
                "uplift_score": uplift,
                "ci_width": ci_width
            })
            continue
        
        # 计算优先级分数
        priority_score = uplift * ltv
        candidates.append({
            "user_id": uid,
            "uplift_score": round(uplift, 4),
            "ltv_estimate": ltv,
            "priority_score": round(priority_score, 2),
            "last_purchase_days": u.get("last_purchase_days", 0)
        })
    
    # 按优先级分数降序排列
    candidates.sort(key=lambda x: x["priority_score"], reverse=True)
    
    # 资源约束截断
    execution_queue = candidates[:resource_budget]
    waitlist = candidates[resource_budget:]
    
    # 分层标签（在执行队列内按分数分层）
    n_exec = len(execution_queue)
    top_cutoff = max(1, int(n_exec * 0.20))
    mid_cutoff = max(top_cutoff, int(n_exec * 0.80))
    
    for i, user in enumerate(execution_queue):
        if i < top_cutoff:
            user["tier"] = "TOP_20"
            user["intervention"] = "VIP礼包+优惠券+专属客服"
            user["resource_type"] = "premium"
        elif i < mid_cutoff:
            user["tier"] = "MID_60"
            user["intervention"] = "标准优惠券+自动邮件"
            user["resource_type"] = "standard"
        else:
            user["tier"] = "BOTTOM_20"
            user["intervention"] = "轻量推送（不消耗优惠券）"
            user["resource_type"] = "light"
    
    # 计算预期增量GMV
    expected_incremental_gmv = sum(
        u["uplift_score"] * u["ltv_estimate"] for u in execution_queue
    )
    # 对比：随机分配同等资源的预期增量GMV（使用候选池平均uplift）
    all_uplifts = [c["uplift_score"] for c in candidates]
    avg_uplift = sum(all_uplifts) / len(all_uplifts) if all_uplifts else 0
    avg_ltv = sum(c["ltv_estimate"] for c in candidates) / len(candidates) if candidates else 0
    random_baseline_gmv = avg_uplift * avg_ltv * resource_budget
    
    summary = {
        "total_input": len(users),
        "excluded_negative_uplift": sum(1 for e in excluded if "负Uplift" in e["reason"]),
        "excluded_high_uncertainty": sum(1 for e in excluded if "不确定性" in e["reason"]),
        "candidates": len(candidates),
        "execution_queue_size": len(execution_queue),
        "waitlist_size": len(waitlist),
        "tier_distribution": {
            "TOP_20": sum(1 for u in execution_queue if u.get("tier") == "TOP_20"),
            "MID_60": sum(1 for u in execution_queue if u.get("tier") == "MID_60"),
            "BOTTOM_20": sum(1 for u in execution_queue if u.get("tier") == "BOTTOM_20"),
        },
        "expected_incremental_gmv": round(expected_incremental_gmv, 2),
        "random_baseline_gmv": round(random_baseline_gmv, 2),
        "uplift_vs_random": round(
            (expected_incremental_gmv - random_baseline_gmv) / max(random_baseline_gmv, 1) * 100, 1
        )
    }
    
    return {
        "execution_queue": execution_queue,
        "waitlist": waitlist[:20],  # 仅返回等待池前20名
        "excluded": excluded,
        "summary": summary
    }


# 测试
import random
random.seed(42)

users = []
for i in range(50):
    uplift = random.uniform(-0.15, 0.45)
    ltv = random.uniform(80, 400)
    ci = random.uniform(0.05, 0.45)
    users.append({
        "user_id": f"U{i:04d}",
        "uplift_score": round(uplift, 3),
        "uplift_ci_width": round(ci, 2),
        "ltv_estimate": round(ltv, 2),
        "last_purchase_days": random.randint(10, 90)
    })

result = uplift_intervention_priority_queue(users, resource_budget=20)

# 验证基本结构
assert len(result["execution_queue"]) <= 20
assert all(u["uplift_score"] >= 0 for u in result["execution_queue"])  # 无负Uplift
# 执行队列按优先级降序
scores = [u["priority_score"] for u in result["execution_queue"]]
assert scores == sorted(scores, reverse=True), "执行队列未按分数降序"
# 执行队列优先级分数 >= 等待池分数
if result["execution_queue"] and result["waitlist"]:
    assert result["execution_queue"][-1]["priority_score"] >= result["waitlist"][0]["priority_score"]
# 分层分配正确
tier_counts = result["summary"]["tier_distribution"]
assert tier_counts["TOP_20"] + tier_counts["MID_60"] + tier_counts["BOTTOM_20"] == len(result["execution_queue"])

print("[✓] Uplift Intervention Priority Queue 测试通过")
print(f"  输入: {result['summary']['total_input']}人，候选: {result['summary']['candidates']}人，执行队列: {result['summary']['execution_queue_size']}人")
print(f"  排除负Uplift: {result['summary']['excluded_negative_uplift']}人，排除高不确定性: {result['summary']['excluded_high_uncertainty']}人")
print(f"  预期增量GMV: ${result['summary']['expected_incremental_gmv']:,.2f}（随机基线: ${result['summary']['random_baseline_gmv']:,.2f}，优于随机: +{result['summary']['uplift_vs_random']}%）")
print(f"  分层: TOP={tier_counts['TOP_20']}, MID={tier_counts['MID_60']}, BOTTOM={tier_counts['BOTTOM_20']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Uplift-Churn-Prediction]]（提供Uplift模型评分及置信区间，是本队列生成器的核心数据输入）
- **延伸（extends）**：[[Skill-AB-Test-Sequential-Design]]（对TOP_20档用户做A/B实验验证Uplift预测效果，形成模型校准闭环）
- **可组合（combinable）**：[[Skill-RFM-Segment-Campaign-Dispatcher]]（Uplift队列与RFM分群联合使用，Uplift确定"谁最值得干预"，RFM确定"用什么方式干预"）

## ⑤ 商业价值评估
- **ROI量化**：相比随机发券，Uplift×LTV排序在相同预算下可提升增量GMV约40-70%；500张优惠券场景下，典型增量GMV提升$2,000-4,000/次
- **实施难度**：⭐⭐☆☆☆（主要依赖Uplift模型上游，执行逻辑为简单排序截断）
- **优先级**：⭐⭐⭐⭐⭐（促销资源永远有限，排序效率直接决定营销ROI；是运营自动化的高频核心场景）

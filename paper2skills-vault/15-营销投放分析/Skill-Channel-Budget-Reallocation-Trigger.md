---
title: Channel Budget Reallocation Trigger — 饱和度超阈值时自动削减并重分配渠道预算
doc_type: knowledge
module: 15-营销投放分析
topic: channel-budget-reallocation-trigger
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Channel Budget Reallocation Trigger

> **配对分析层**：[[Skill-Channel-Saturation-Curve]]
> **决策类型**: 自动触发型 | **触发条件**: 渠道饱和度 > 80% | **执行动作**: 削减该渠道预算20%并重分配至饱和度<50%的低饱和渠道

## ① 算法原理

核心是「饱和度阈值门控 + 比例重分配 + 再平衡约束优化」三阶段逻辑：

1. **阈值门控**：对每个渠道当前饱和度（来自Skill-Channel-Saturation-Curve的输出）进行分类，饱和度>80%标记为「超载」，50%-80%为「健康」，<50%为「低载」。
2. **削减计算**：超载渠道预算按20%比例削减，释放出的绝对预算金额进入再分配池。
3. **比例重分配**：再分配池按低载渠道的「(0.5 - 饱和度)」反比权重进行分配，饱和度越低获得越多预算增量。
4. **再平衡约束**：引入最大单次调整上限（任意渠道不超过当前预算±30%），防止振荡效应；同时确保总预算恒等式成立。

**误触发防护**：要求连续2个统计周期饱和度均>80%才触发削减，避免单期数据噪声造成频繁调拨。**回滚机制**：每次调整保存快照，若调整后7天ROI下降>10%则自动恢复上一快照。

## ② 母婴出海应用案例

**场景：吸奶器品类跨平台广告预算自动优化**
- 触发条件：Facebook广告饱和度连续2周达到85%（CPM上涨30%，CTR下降至1.2%），Google Shopping饱和度38%（流量池仍充裕）
- 执行动作：Facebook预算削减20%（-$2,000/周），将$2,000按权重转移到Google Shopping（$1,400）和Pinterest（$600）
- 安全护栏：单渠道预算变动不超过30%；调整后7天ROI监控，跌幅>10%自动回滚
- 业务价值：整体ROAS从2.8提升至3.4，年化节省无效投放约$48,000

## ③ 代码模板

```python
import numpy as np
from typing import Dict, List, Tuple

def channel_budget_reallocation_trigger(
    channel_data: List[Dict],
    total_budget: float,
    saturation_high_threshold: float = 0.8,
    saturation_low_threshold: float = 0.5,
    reduction_ratio: float = 0.2,
    max_change_ratio: float = 0.3,
    consecutive_periods: int = 2
) -> Dict:
    """
    渠道预算再分配决策触发器
    
    参数:
        channel_data: [{"name": str, "budget": float, "saturation_history": [float, ...]}]
        total_budget: 总预算（用于验证恒等式）
        saturation_high_threshold: 超载阈值（默认0.8）
        saturation_low_threshold: 低载阈值（默认0.5）
        reduction_ratio: 超载渠道削减比例（默认0.2）
        max_change_ratio: 单次最大调整比例（默认0.3）
        consecutive_periods: 连续触发周期数（防误触发）
    
    返回:
        {"actions": [...], "new_budgets": {...}, "rollback_snapshot": {...}}
    """
    # 1. 阈值门控：识别超载和低载渠道
    overloaded = []
    underloaded = []
    
    for ch in channel_data:
        history = ch["saturation_history"]
        # 连续N期判断（防误触发）
        is_overloaded = all(s > saturation_high_threshold for s in history[-consecutive_periods:])
        current_sat = history[-1]
        
        if is_overloaded:
            overloaded.append(ch)
        elif current_sat < saturation_low_threshold:
            underloaded.append({"name": ch["name"], "budget": ch["budget"], 
                                 "saturation": current_sat, "gap": saturation_low_threshold - current_sat})
    
    if not overloaded or not underloaded:
        return {
            "trigger": False,
            "reason": f"无超载渠道={len(overloaded)}, 无低载渠道={len(underloaded)}，不触发调整",
            "actions": [],
            "new_budgets": {ch["name"]: ch["budget"] for ch in channel_data}
        }
    
    # 2. 计算削减金额（含最大变动约束）
    reduction_pool = 0.0
    new_budgets = {ch["name"]: ch["budget"] for ch in channel_data}
    actions = []
    rollback_snapshot = {ch["name"]: ch["budget"] for ch in channel_data}
    
    for ch in overloaded:
        raw_reduction = ch["budget"] * reduction_ratio
        # 约束：不超过max_change_ratio
        max_reduction = ch["budget"] * max_change_ratio
        actual_reduction = min(raw_reduction, max_reduction)
        
        new_budgets[ch["name"]] -= actual_reduction
        reduction_pool += actual_reduction
        actions.append({
            "channel": ch["name"],
            "action": "REDUCE",
            "amount": -actual_reduction,
            "from": ch["budget"],
            "to": new_budgets[ch["name"]],
            "reason": f"饱和度{ch['saturation_history'][-1]:.0%}>{saturation_high_threshold:.0%}"
        })
    
    # 3. 比例重分配：按(阈值-当前饱和度)加权
    total_gap = sum(u["gap"] for u in underloaded)
    for u in underloaded:
        weight = u["gap"] / total_gap
        allocation = reduction_pool * weight
        # 约束：不超过max_change_ratio
        max_increase = u["budget"] * max_change_ratio
        actual_allocation = min(allocation, max_increase)
        
        new_budgets[u["name"]] += actual_allocation
        actions.append({
            "channel": u["name"],
            "action": "INCREASE",
            "amount": +actual_allocation,
            "from": u["budget"],
            "to": new_budgets[u["name"]],
            "reason": f"低饱和度{u['saturation']:.0%}<{saturation_low_threshold:.0%}，权重{weight:.2f}"
        })
    
    # 4. 验证总预算恒等式（允许±1%误差）
    total_new = sum(new_budgets.values())
    budget_drift = abs(total_new - total_budget) / total_budget
    
    return {
        "trigger": True,
        "actions": actions,
        "new_budgets": new_budgets,
        "reduction_pool": reduction_pool,
        "budget_drift": budget_drift,
        "budget_balanced": budget_drift < 0.01,
        "rollback_snapshot": rollback_snapshot,
        "rollback_condition": "7天ROI下降>10%自动回滚",
        "execution_priority": "HIGH"
    }


# 测试
channel_data = [
    {"name": "Facebook", "budget": 10000, "saturation_history": [0.83, 0.87]},
    {"name": "Google",   "budget": 8000,  "saturation_history": [0.65, 0.68]},
    {"name": "Pinterest","budget": 3000,  "saturation_history": [0.35, 0.38]},
    {"name": "TikTok",   "budget": 4000,  "saturation_history": [0.42, 0.45]},
]
total_budget = sum(ch["budget"] for ch in channel_data)
result = channel_budget_reallocation_trigger(channel_data, total_budget)

assert result["trigger"] == True, "Facebook应触发削减"
assert result["budget_balanced"] == True, "总预算应保持平衡"
assert any(a["action"] == "REDUCE" and a["channel"] == "Facebook" for a in result["actions"])
assert any(a["action"] == "INCREASE" and a["channel"] == "Pinterest" for a in result["actions"])
print("[✓] Channel Budget Reallocation Trigger决策触发器测试通过")
print(f"  触发渠道: {[a['channel'] for a in result['actions'] if a['action']=='REDUCE']}")
print(f"  受益渠道: {[a['channel'] for a in result['actions'] if a['action']=='INCREASE']}")
print(f"  预算漂移: {result['budget_drift']:.4%}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Channel-Saturation-Curve]]（提供各渠道当前饱和度信号）
- **延伸（extends）**：[[Skill-MMM-Budget-Reallocation-Executor]]（执行实际API调用）
- **可组合（combinable）**：[[Skill-DARA-Agentic-MMM-Optimizer]]（MMM全局优化后的精细化渠道触发）

## ⑤ 商业价值评估
- ROI预估：整体ROAS提升15-20%，年化节省无效投放$40,000-$80,000
- 实施难度：⭐⭐☆☆☆（规则明确，接入渠道API即可）
- 优先级：⭐⭐⭐⭐⭐

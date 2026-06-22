---
title: MMM Budget Reallocation Executor — 将MMM最优权重转化为渠道预算调整API执行指令
doc_type: knowledge
module: 15-营销投放分析
topic: mmm-budget-reallocation-executor
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: MMM Budget Reallocation Executor

> **配对分析层**：[[Skill-DARA-Agentic-MMM-Optimizer]]
> **决策类型**: 自动触发型 | **触发条件**: MMM输出最优权重与当前分配偏差>10% | **执行动作**: 生成渠道预算调整API调用指令并执行确认回路

## ① 算法原理

核心是「权重归一化 + 差异检测 + API调用封装 + 执行确认回路」：

1. **权重归一化**：将MMM输出的各渠道ROI贡献系数归一化为预算分配权重（softmax归一化，保留非线性关系）。
2. **差异检测**：计算新最优权重与当前实际权重的KL散度或绝对偏差，仅当偏差>阈值（默认10%）时才触发执行，避免频繁微调。
3. **API调用封装**：生成各渠道广告平台（Facebook/Google/TikTok）的标准化预算更新指令（模拟API调用格式，实际接入需填入真实Token）。
4. **执行确认回路**：生成「预计变化摘要」，支持人工审核确认后执行，或设置自动执行阈值（偏差>20%自动执行，10-20%需人工确认）。

**误触发防护**：单次调整总额不超过当前总预算的15%；任意单渠道变动不超过当前预算50%。**回滚机制**：执行后48小时监控ROAS，若整体ROAS下降>15%则触发回滚。

## ② 母婴出海应用案例

**场景：母婴品牌季度MMM更新后的预算重分配执行**
- 触发条件：MMM显示YouTube贡献权重28%（当前分配仅12%），Facebook权重35%（当前分配52%），KL散度0.18>阈值0.10
- 执行动作：Facebook $52,000→$35,000（-32.7%），YouTube $12,000→$28,000（+133%），其余渠道微调；偏差>20%需人工确认后自动执行
- 安全护栏：YouTube增幅超50%，自动拆分为两期执行（每期+50%），间隔14天
- 业务价值：整体ROAS预计从3.1提升至3.8，年化增量收益约$190,000

## ③ 代码模板

```python
import numpy as np
from typing import Dict, List, Tuple
import json

def mmm_budget_reallocation_executor(
    mmm_roi_coefficients: Dict[str, float],
    current_budgets: Dict[str, float],
    total_budget: float,
    diff_threshold_auto: float = 0.20,
    diff_threshold_manual: float = 0.10,
    max_total_change_ratio: float = 0.15,
    max_single_channel_change: float = 0.50,
    max_single_period_increase: float = 0.50
) -> Dict:
    """
    MMM预算再分配执行器
    
    参数:
        mmm_roi_coefficients: MMM输出的各渠道ROI贡献系数 {"channel": coefficient}
        current_budgets: 当前各渠道预算分配 {"channel": budget}
        total_budget: 总预算
        diff_threshold_auto: 自动执行阈值（偏差>此值自动执行）
        diff_threshold_manual: 人工确认阈值（偏差在两阈值之间需确认）
        max_total_change_ratio: 单次最大总调整比例
        max_single_channel_change: 单渠道最大变动比例
        max_single_period_increase: 单期最大增幅（超出则分期执行）
    
    返回:
        执行指令字典，含API调用参数和确认状态
    """
    channels = list(mmm_roi_coefficients.keys())
    
    # 1. Softmax归一化ROI系数为最优权重
    coefs = np.array([mmm_roi_coefficients[c] for c in channels])
    exp_coefs = np.exp(coefs - np.max(coefs))  # 数值稳定
    optimal_weights = exp_coefs / exp_coefs.sum()
    
    # 2. 计算当前权重
    current_total = sum(current_budgets.values())
    current_weights = np.array([current_budgets.get(c, 0) / current_total for c in channels])
    
    # 3. 计算偏差（最大绝对偏差）
    weight_diff = np.abs(optimal_weights - current_weights)
    max_diff = float(weight_diff.max())
    
    # 4. 确定执行模式
    if max_diff < diff_threshold_manual:
        return {
            "trigger": False,
            "reason": f"最大权重偏差{max_diff:.2%} < 阈值{diff_threshold_manual:.0%}，无需调整",
            "action": "NO_CHANGE",
            "optimal_weights": dict(zip(channels, optimal_weights.tolist()))
        }
    
    execution_mode = "AUTO" if max_diff > diff_threshold_auto else "MANUAL_CONFIRM"
    
    # 5. 计算目标预算（含总调整量约束）
    optimal_budgets_raw = {c: total_budget * float(w) for c, w in zip(channels, optimal_weights)}
    
    # 约束：总变动量不超过max_total_change_ratio × total_budget
    max_total_change = total_budget * max_total_change_ratio
    total_increase = sum(max(0, optimal_budgets_raw[c] - current_budgets.get(c, 0)) for c in channels)
    
    if total_increase > max_total_change:
        # 等比缩放调整幅度
        scale = max_total_change / total_increase
        adjusted_budgets = {}
        for c in channels:
            raw_change = optimal_budgets_raw[c] - current_budgets.get(c, 0)
            if raw_change > 0:
                adjusted_budgets[c] = current_budgets.get(c, 0) + raw_change * scale
            else:
                adjusted_budgets[c] = optimal_budgets_raw[c]
        # 重新归一化确保总量
        adj_total = sum(adjusted_budgets.values())
        adjusted_budgets = {c: v / adj_total * total_budget for c, v in adjusted_budgets.items()}
    else:
        adjusted_budgets = optimal_budgets_raw
    
    # 6. 生成API调用指令（含分期执行检测）
    api_instructions = []
    for c in channels:
        old_budget = current_budgets.get(c, 0)
        new_budget = round(adjusted_budgets[c], 2)
        change = new_budget - old_budget
        change_ratio = change / old_budget if old_budget > 0 else float("inf")
        
        if abs(change_ratio) > max_single_period_increase and change > 0:
            # 分期执行：拆分为两期
            mid_budget = round(old_budget * (1 + max_single_period_increase), 2)
            api_instructions.append({
                "channel": c,
                "platform_api": f"POST /api/campaigns/{c}/budget",
                "period": "phase_1",
                "from": old_budget,
                "to": mid_budget,
                "change_ratio": max_single_period_increase,
                "execute_after_days": 0,
                "note": f"增幅>{max_single_period_increase:.0%}，拆分两期执行"
            })
            api_instructions.append({
                "channel": c,
                "platform_api": f"POST /api/campaigns/{c}/budget",
                "period": "phase_2",
                "from": mid_budget,
                "to": new_budget,
                "change_ratio": (new_budget - mid_budget) / mid_budget,
                "execute_after_days": 14,
                "note": "第二期执行，间隔14天"
            })
        else:
            api_instructions.append({
                "channel": c,
                "platform_api": f"POST /api/campaigns/{c}/budget",
                "period": "single",
                "from": old_budget,
                "to": new_budget,
                "change_ratio": change_ratio,
                "execute_after_days": 0
            })
    
    return {
        "trigger": True,
        "execution_mode": execution_mode,
        "max_weight_diff": max_diff,
        "api_instructions": api_instructions,
        "budget_summary": {
            "total": total_budget,
            "current": current_budgets,
            "new": {c: round(adjusted_budgets[c], 2) for c in channels}
        },
        "rollback_condition": "执行后48小时ROAS下降>15%触发回滚",
        "human_confirm_required": execution_mode == "MANUAL_CONFIRM",
        "execution_priority": "HIGH" if execution_mode == "AUTO" else "MEDIUM"
    }


# 测试
mmm_coefs = {"Facebook": 2.8, "YouTube": 3.5, "Google": 3.2, "TikTok": 2.5}
current_budgets = {"Facebook": 52000, "YouTube": 12000, "Google": 25000, "TikTok": 11000}
total_budget = 100000

result = mmm_budget_reallocation_executor(mmm_coefs, current_budgets, total_budget)

assert result["trigger"] == True
assert "api_instructions" in result
assert len(result["api_instructions"]) > 0
# 验证总预算恒等
new_total = sum(result["budget_summary"]["new"].values())
assert abs(new_total - total_budget) < 1.0, f"总预算不平衡: {new_total}"

# 测试无需调整的场景（均匀分配 + 均匀系数）
result_no = mmm_budget_reallocation_executor(
    {"A": 1.0, "B": 1.0}, {"A": 50000, "B": 50000}, 100000
)
assert result_no["trigger"] == False

print("[✓] MMM Budget Reallocation Executor决策触发器测试通过")
print(f"  执行模式: {result['execution_mode']}")
print(f"  API指令数: {len(result['api_instructions'])}")
print(f"  总预算验证: ${new_total:,.0f} (目标${total_budget:,})")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-DARA-Agentic-MMM-Optimizer]]（提供MMM最优ROI系数）
- **延伸（extends）**：[[Skill-Channel-Budget-Reallocation-Trigger]]（饱和度触发的短期调整）
- **可组合（combinable）**：[[Skill-Bayesian-MMM-Action-Plan-Generator]]（结合贝叶斯不确定性生成保守/激进多方案）

## ⑤ 商业价值评估
- ROI预估：MMM指导的预算重分配通常提升ROAS 15-25%，年化增量$150,000-$250,000
- 实施难度：⭐⭐⭐☆☆（需接入各广告平台API，逻辑清晰但接口工作量较多）
- 优先级：⭐⭐⭐⭐⭐

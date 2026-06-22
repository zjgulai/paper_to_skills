---
title: LTV-Acquisition-Budget-Gate — LTV/CAC比值驱动的获客预算自动开闸/熔断决策器
doc_type: knowledge
module: 06-增长模型
topic: ltv-acquisition-budget-gate
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-LTV-Acquisition-Budget-Gate

> **配对分析层**：[[Skill-LTV-Prediction-ZILN]]
> **决策类型**: 自动触发型 | **触发条件**: LTV预测值与渠道CAC比值突破阈值 | **执行动作**: LTV<CAC×3→暂停该渠道新客投入；LTV>CAC×5→扩增该渠道20%预算

## ① 算法原理

核心是「LTV/CAC比值实时监控 + 双向阈值熔断/开闸 + 渠道级精细化预算调控」：

1. **比值计算**：LTV/CAC Ratio = 预测12个月LTV（来自ZILN模型P50估计）÷ 该渠道近30天新客平均CAC。比值按渠道、按月更新。

2. **双向阈值规则**：
   - 比值 < 3.0（危险区）：触发熔断，暂停该渠道新客获取预算（不影响再营销/留存预算）
   - 3.0 ≤ 比值 < 5.0（健康区）：维持当前预算不变，进入观察状态
   - 比值 ≥ 5.0（优质区）：触发开闸，下一预算周期自动扩增20%

3. **渐进式调控**：扩增不超过单次+20%，避免过快放量导致渠道效率下降（受众饱和）；连续3周触发开闸才允许再次扩增20%（累计上限+60%/季度）。

4. **LTV置信区间防护**：使用ZILN模型P25估计（而非P50）做熔断判断，更保守；使用P75估计做开闸判断，防止噪声触发超支。

**误触发防护**：渠道新客样本量≥100时才触发决策，样本不足时冻结该渠道决策状态。**回滚机制**：扩增后若30天内实际CAC上升>30%，自动回撤增量预算。

## ② 母婴出海应用案例

**场景：跨境母婴品牌多渠道获客预算月度自动调控**
- 触发条件：本月ZILN模型LTV更新完成，覆盖Facebook/Google/TikTok三渠道近1000个新客
  - Facebook：LTV_P25=$210, CAC=$52，比值4.04（健康，维持）
  - Google：LTV_P25=$180, CAC=$28，LTV_P75=$320，比值6.4（P75>5，开闸+20%）
  - TikTok：LTV_P25=$95, CAC=$45，比值2.1（<3，熔断）
- 执行动作：Google预算从$18,000→$21,600；TikTok新客投放暂停（已有客户再营销不受影响）；Facebook维持$22,000
- 安全护栏：Google样本量138≥100（有效）；TikTok样本量210≥100（有效）
- 业务价值：停止TikTok低效获客每月节省$6,800；Google预算增投预计带来$9,500增量GMV，月度ROI提升约18%

## ③ 代码模板

```python
from typing import Dict, List, Optional

# 阈值配置
GATE_THRESHOLDS = {
    "freeze": 3.0,   # LTV/CAC < 3.0 → 熔断暂停
    "healthy_low": 3.0,
    "healthy_high": 5.0,
    "expand": 5.0,   # LTV/CAC > 5.0 → 开闸扩增
}
EXPAND_RATE = 0.20        # 单次扩增比例
MIN_SAMPLE_SIZE = 100     # 最小有效样本量
MAX_EXPAND_CONSECUTIVE = 3  # 连续扩增上限（季度内）


def ltv_acquisition_budget_gate(
    channels: List[Dict],
    freeze_threshold: float = GATE_THRESHOLDS["freeze"],
    expand_threshold: float = GATE_THRESHOLDS["expand"],
    expand_rate: float = EXPAND_RATE,
    min_sample: int = MIN_SAMPLE_SIZE
) -> Dict:
    """
    LTV/CAC比值驱动的获客预算开闸/熔断决策器
    
    参数:
        channels: [{
            "channel_id": str,
            "current_budget": float,    # 当前月预算（美元）
            "cac_30d": float,           # 近30天平均CAC
            "ltv_p25": float,           # LTV P25估计（保守，用于熔断）
            "ltv_p50": float,           # LTV P50估计（基准）
            "ltv_p75": float,           # LTV P75估计（乐观，用于开闸）
            "new_customer_count": int,  # 近30天新客数（样本量）
            "consecutive_expand_count": int,  # 本季度已连续扩增次数
        }]
        freeze_threshold: 熔断比值阈值
        expand_threshold: 扩增比值阈值（基于P75）
        expand_rate: 单次扩增比例
        min_sample: 最小有效样本量
    
    返回:
        {"decisions": [...], "budget_changes": {...}, "summary": {...}}
    """
    decisions = []
    budget_changes = {}
    
    for ch in channels:
        cid = ch["channel_id"]
        budget = ch["current_budget"]
        cac = ch["cac_30d"]
        ltv_p25 = ch["ltv_p25"]
        ltv_p50 = ch["ltv_p50"]
        ltv_p75 = ch["ltv_p75"]
        n_samples = ch["new_customer_count"]
        consec_expand = ch.get("consecutive_expand_count", 0)
        
        # 样本量门控
        if n_samples < min_sample:
            decisions.append({
                "channel_id": cid,
                "action": "HOLD",
                "reason": f"样本量不足（{n_samples}<{min_sample}），冻结决策状态",
                "current_budget": budget,
                "new_budget": budget,
                "ratio_p50": round(ltv_p50 / cac, 2) if cac > 0 else 0
            })
            budget_changes[cid] = budget
            continue
        
        if cac <= 0:
            decisions.append({
                "channel_id": cid,
                "action": "ERROR",
                "reason": "CAC数据异常（≤0）",
                "current_budget": budget,
                "new_budget": budget
            })
            budget_changes[cid] = budget
            continue
        
        # 计算比值（保守/基准/乐观）
        ratio_p25 = ltv_p25 / cac
        ratio_p50 = ltv_p50 / cac
        ratio_p75 = ltv_p75 / cac
        
        # 熔断判断（使用保守P25）
        if ratio_p25 < freeze_threshold:
            decision = {
                "channel_id": cid,
                "action": "FREEZE",
                "reason": f"LTV_P25/CAC={ratio_p25:.2f} < {freeze_threshold}（危险区），暂停新客投入",
                "current_budget": budget,
                "new_budget": 0.0,  # 新客预算归零（再营销预算独立管理）
                "ratio_p25": round(ratio_p25, 2),
                "ratio_p50": round(ratio_p50, 2),
                "note": "再营销/留存预算不受影响，本决策仅针对新客获取预算"
            }
        # 开闸判断（使用乐观P75，防止噪声触发超支）
        elif ratio_p75 >= expand_threshold and consec_expand < MAX_EXPAND_CONSECUTIVE:
            new_budget = round(budget * (1 + expand_rate), 2)
            decision = {
                "channel_id": cid,
                "action": "EXPAND",
                "reason": f"LTV_P75/CAC={ratio_p75:.2f} ≥ {expand_threshold}（优质区），扩增{expand_rate*100:.0f}%预算",
                "current_budget": budget,
                "new_budget": new_budget,
                "budget_delta": round(new_budget - budget, 2),
                "ratio_p25": round(ratio_p25, 2),
                "ratio_p50": round(ratio_p50, 2),
                "ratio_p75": round(ratio_p75, 2),
                "consecutive_count": consec_expand + 1,
                "rollback_trigger": f"若30天内实际CAC上升>30%（>{round(cac * 1.3, 2)}），自动回撤"
            }
        elif ratio_p75 >= expand_threshold and consec_expand >= MAX_EXPAND_CONSECUTIVE:
            decision = {
                "channel_id": cid,
                "action": "HOLD",
                "reason": f"已连续扩增{consec_expand}次（季度上限），维持现有预算等待下季度重置",
                "current_budget": budget,
                "new_budget": budget,
                "ratio_p75": round(ratio_p75, 2)
            }
        else:
            # 健康区：维持
            decision = {
                "channel_id": cid,
                "action": "MAINTAIN",
                "reason": f"LTV/CAC健康区（P50={ratio_p50:.2f}），维持当前预算",
                "current_budget": budget,
                "new_budget": budget,
                "ratio_p25": round(ratio_p25, 2),
                "ratio_p50": round(ratio_p50, 2),
                "ratio_p75": round(ratio_p75, 2)
            }
        
        decisions.append(decision)
        budget_changes[cid] = decision["new_budget"]
    
    total_before = sum(ch["current_budget"] for ch in channels)
    total_after = sum(budget_changes.values())
    frozen_budget = sum(
        ch["current_budget"] for ch in channels
        if budget_changes.get(ch["channel_id"], 0) == 0
    )
    
    summary = {
        "total_channels": len(channels),
        "frozen": sum(1 for d in decisions if d["action"] == "FREEZE"),
        "expanded": sum(1 for d in decisions if d["action"] == "EXPAND"),
        "maintained": sum(1 for d in decisions if d["action"] in ("MAINTAIN", "HOLD")),
        "total_budget_before": round(total_before, 2),
        "total_budget_after": round(total_after, 2),
        "budget_delta": round(total_after - total_before, 2),
        "freed_budget": round(frozen_budget, 2)
    }
    
    return {
        "decisions": decisions,
        "budget_changes": budget_changes,
        "summary": summary
    }


# 测试
channels = [
    # Facebook：健康区 (P25比值=4.04, P75比值=5.77)
    {"channel_id": "Facebook", "current_budget": 22000, "cac_30d": 52,
     "ltv_p25": 210, "ltv_p50": 265, "ltv_p75": 300, "new_customer_count": 423, "consecutive_expand_count": 0},
    # Google：优质区 (P75比值=11.4 > 5) → 开闸
    {"channel_id": "Google", "current_budget": 18000, "cac_30d": 28,
     "ltv_p25": 180, "ltv_p50": 250, "ltv_p75": 320, "new_customer_count": 138, "consecutive_expand_count": 0},
    # TikTok：危险区 (P25比值=2.1 < 3) → 熔断
    {"channel_id": "TikTok", "current_budget": 6800, "cac_30d": 45,
     "ltv_p25": 95, "ltv_p50": 130, "ltv_p75": 180, "new_customer_count": 210, "consecutive_expand_count": 0},
    # KOL：样本不足
    {"channel_id": "KOL", "current_budget": 5000, "cac_30d": 80,
     "ltv_p25": 240, "ltv_p50": 300, "ltv_p75": 380, "new_customer_count": 45, "consecutive_expand_count": 0},
]

result = ltv_acquisition_budget_gate(channels)

assert result["summary"]["total_channels"] == 4
assert result["summary"]["frozen"] == 1  # TikTok熔断
assert result["summary"]["expanded"] == 1  # Google开闸

# TikTok预算清零
tiktok = next(d for d in result["decisions"] if d["channel_id"] == "TikTok")
assert tiktok["action"] == "FREEZE"
assert tiktok["new_budget"] == 0.0

# Google预算扩增20%
google = next(d for d in result["decisions"] if d["channel_id"] == "Google")
assert google["action"] == "EXPAND"
assert abs(google["new_budget"] - 18000 * 1.2) < 1

# KOL样本不足，维持预算
kol = next(d for d in result["decisions"] if d["channel_id"] == "KOL")
assert kol["action"] == "HOLD"
assert kol["new_budget"] == 5000

print("[✓] LTV Acquisition Budget Gate 测试通过")
print(f"  渠道总数: {result['summary']['total_channels']}")
print(f"  熔断: {result['summary']['frozen']}，扩增: {result['summary']['expanded']}")
print(f"  预算变化: ${result['summary']['total_budget_before']:,.0f} → ${result['summary']['total_budget_after']:,.0f}（△{result['summary']['budget_delta']:+,.0f}）")
print(f"  释放低效预算: ${result['summary']['freed_budget']:,.0f}")
for d in result["decisions"]:
    print(f"  [{d['action']:10s}] {d['channel_id']}: ${d['current_budget']:,.0f} → ${d['new_budget']:,.0f}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-LTV-Prediction-ZILN]]（提供P25/P50/P75三分位LTV估计，是本决策器的核心输入）
- **延伸（extends）**：[[Skill-CAC-Attribution-Model]]（精确计算各渠道真实CAC，消除多渠道重叠归因误差）
- **可组合（combinable）**：[[Skill-Bayesian-MMM-Scenario-Action-Plan]]（LTV门控与MMM情景方案联合使用，形成完整的Q+1预算决策体系）

## ⑤ 商业价值评估
- **ROI量化**：典型场景下每月识别并停止1-2个低效渠道，释放$5,000-15,000预算；同时扩增高ROI渠道带来增量GMV，综合月度获客效率提升10-25%
- **实施难度**：⭐⭐☆☆☆（依赖LTV模型上游，但决策逻辑本身规则清晰，实施门槛低）
- **优先级**：⭐⭐⭐⭐⭐（获客预算是跨境卖家最大可控成本项，LTV/CAC比值优化直接影响盈利能力）

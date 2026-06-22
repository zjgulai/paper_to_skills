---
title: LTV CAC Acquisition Gate — LTV/CAC比率触发渠道获客自动暂停或扩投
doc_type: knowledge
module: 06-增长模型
topic: ltv-cac-acquisition-gate
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: LTV-CAC Acquisition Gate

> **配对分析层**：[[Skill-LTV-Prediction-ZILN]]
> **决策类型**: 自动触发型 | **触发条件**: 渠道LTV/CAC<3 或 >5 | **执行动作**: 自动暂停该渠道新客获取，或提升该渠道预算20%

## ① 算法原理

核心是「LTV/CAC比率计算 + 渠道级别决策 + 预算弹性调整」：

1. **LTV/CAC比率计算**：
   - LTV来自预测模型（如ZILN）输出的12个月预测LTV
   - CAC = 渠道广告支出 / 渠道当月新增客户数
   - 比率 = 预测LTV / CAC
   
2. **三档决策逻辑**：
   - LTV/CAC < 3（低效）：每单亏损或回收周期过长 → 自动暂停该渠道新客获取
   - 3 ≤ LTV/CAC ≤ 5（健康）：维持当前预算，无操作
   - LTV/CAC > 5（高效）：远超投资门槛，存在增长空间 → 自动提升该渠道预算20%
   
3. **滚动更新机制**：每月更新一次渠道LTV/CAC评估，使用近3个月新用户的LTV追踪数据而非单月数据，平滑季节性波动。

**误触发防护**：渠道当月新增客户数≥50才进行评估（样本量门控）；暂停不立即生效，给予7天观察期确认。**回滚机制**：暂停后30天内如果渠道LTV重新评估>3.5，自动恢复投放。

## ② 母婴出海应用案例

**场景：母婴品牌多渠道获客效率评估**
- 触发条件：Pinterest渠道LTV/CAC=2.1（新客LTV预测$85，CAC=$40.4），低于阈值3，触发暂停；YouTube LTV/CAC=6.2（LTV $248，CAC $40），触发扩投
- 执行动作：Pinterest暂停新客获取预算（节省$8,000/月），YouTube预算增加20%（+$4,000/月转入）
- 安全护栏：Pinterest品牌维护最低预算$500/月不受影响；Pinterest暂停7天后重新评估
- 业务价值：资源向高ROI渠道集中，整体获客效率提升，年化减少无效获客支出约$45,000

## ③ 代码模板

```python
from typing import Dict, List, Optional
import numpy as np
from datetime import date

def ltv_cac_acquisition_gate(
    channel_metrics: List[Dict],
    pause_threshold: float = 3.0,
    boost_threshold: float = 5.0,
    boost_ratio: float = 0.20,
    min_new_customers: int = 50,
    min_brand_budget: float = 500.0,
    pause_observation_days: int = 7
) -> Dict:
    """
    LTV/CAC渠道获客门控决策器
    
    参数:
        channel_metrics: [{
            "channel_name": str,
            "predicted_ltv": float,  # 模型预测12月LTV（来自ZILN）
            "cac": float,            # 当月获客成本
            "new_customers_count": int,  # 当月新客数
            "current_budget": float,     # 当前月预算
            "ltv_history": [float]       # 近3月LTV追踪（可选）
        }]
        pause_threshold: 暂停阈值（默认3.0）
        boost_threshold: 扩投阈值（默认5.0）
        boost_ratio: 扩投比例（默认20%）
        min_new_customers: 最小新客数门控
        min_brand_budget: 暂停后品牌最低保留预算
    
    返回:
        各渠道决策和预算调整指令
    """
    decisions = []
    total_budget_change = 0.0
    
    for ch in channel_metrics:
        name = ch["channel_name"]
        ltv = ch["predicted_ltv"]
        cac = ch["cac"]
        new_custs = ch["new_customers_count"]
        current_budget = ch["current_budget"]
        
        # 样本量门控
        if new_custs < min_new_customers:
            decisions.append({
                "channel": name,
                "trigger": False,
                "reason": f"新客量{new_custs}<{min_new_customers}，样本不足，不评估",
                "action": "INSUFFICIENT_DATA"
            })
            continue
        
        # 使用历史平均LTV（若有）
        ltv_history = ch.get("ltv_history", [ltv])
        smoothed_ltv = np.mean(ltv_history[-3:]) if len(ltv_history) >= 3 else ltv
        
        ratio = smoothed_ltv / cac if cac > 0 else 0
        
        if ratio < pause_threshold:
            # 暂停：保留最低品牌预算
            pause_amount = max(0, current_budget - min_brand_budget)
            decisions.append({
                "channel": name,
                "trigger": True,
                "ltv_cac_ratio": round(ratio, 2),
                "action": "PAUSE_ACQUISITION",
                "current_budget": current_budget,
                "paused_budget": round(pause_amount, 2),
                "retained_brand_budget": min_brand_budget,
                "observation_days": pause_observation_days,
                "resume_condition": f"LTV/CAC重新评估>{pause_threshold*1.17:.1f}时恢复",
                "execution_priority": "HIGH"
            })
            total_budget_change -= pause_amount
        
        elif ratio > boost_threshold:
            # 扩投
            boost_amount = round(current_budget * boost_ratio, 2)
            decisions.append({
                "channel": name,
                "trigger": True,
                "ltv_cac_ratio": round(ratio, 2),
                "action": "BOOST_BUDGET",
                "current_budget": current_budget,
                "boost_amount": boost_amount,
                "new_budget": round(current_budget + boost_amount, 2),
                "execution_priority": "MEDIUM"
            })
            total_budget_change += boost_amount
        
        else:
            decisions.append({
                "channel": name,
                "trigger": False,
                "ltv_cac_ratio": round(ratio, 2),
                "action": "MAINTAIN",
                "reason": f"LTV/CAC={ratio:.1f}在健康区间[{pause_threshold},{boost_threshold}]"
            })
    
    # 汇总
    paused = [d for d in decisions if d.get("action") == "PAUSE_ACQUISITION"]
    boosted = [d for d in decisions if d.get("action") == "BOOST_BUDGET"]
    
    return {
        "total_channels": len(channel_metrics),
        "paused_channels": len(paused),
        "boosted_channels": len(boosted),
        "decisions": decisions,
        "budget_impact": {
            "released_from_pause": sum(d["paused_budget"] for d in paused),
            "added_to_boost": sum(d["boost_amount"] for d in boosted),
            "net_change": round(total_budget_change, 2)
        },
        "execution_priority": "HIGH" if paused else "MEDIUM"
    }


# 测试
channel_metrics = [
    {  # Pinterest: LTV/CAC=2.1，触发暂停
        "channel_name": "Pinterest", "predicted_ltv": 85.0, "cac": 40.4,
        "new_customers_count": 120, "current_budget": 8000.0, "ltv_history": [82, 85, 87]
    },
    {  # YouTube: LTV/CAC=6.2，触发扩投
        "channel_name": "YouTube", "predicted_ltv": 248.0, "cac": 40.0,
        "new_customers_count": 85, "current_budget": 20000.0, "ltv_history": [230, 245, 248]
    },
    {  # Facebook: LTV/CAC=4.2，健康维持
        "channel_name": "Facebook", "predicted_ltv": 168.0, "cac": 40.0,
        "new_customers_count": 320, "current_budget": 40000.0
    },
    {  # TikTok: 样本不足
        "channel_name": "TikTok", "predicted_ltv": 110.0, "cac": 35.0,
        "new_customers_count": 30, "current_budget": 5000.0
    },
]

result = ltv_cac_acquisition_gate(channel_metrics)

assert result["paused_channels"] == 1
assert result["boosted_channels"] == 1
pause_dec = next(d for d in result["decisions"] if d.get("action") == "PAUSE_ACQUISITION")
boost_dec = next(d for d in result["decisions"] if d.get("action") == "BOOST_BUDGET")
assert pause_dec["channel"] == "Pinterest"
assert boost_dec["channel"] == "YouTube"
assert pause_dec["retained_brand_budget"] == 500.0
assert boost_dec["boost_amount"] == 20000.0 * 0.20

print("[✓] LTV-CAC Acquisition Gate决策触发器测试通过")
print(f"  暂停渠道: {[d['channel'] for d in result['decisions'] if d.get('action')=='PAUSE_ACQUISITION']}")
print(f"  扩投渠道: {[d['channel'] for d in result['decisions'] if d.get('action')=='BOOST_BUDGET']}")
print(f"  预算影响: 释放${result['budget_impact']['released_from_pause']:,.0f}，增加${result['budget_impact']['added_to_boost']:,.0f}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-LTV-Prediction-ZILN]]（提供渠道新客12月预测LTV）
- **延伸（extends）**：[[Skill-CC-OR-Net-LTV-Prediction]]（结合订购型LTV预测提升精度）
- **可组合（combinable）**：[[Skill-MMM-Budget-Reallocation-Executor]]（LTV/CAC门控与MMM宏观分配协同）

## ⑤ 商业价值评估
- ROI预估：减少低效渠道投入，整体获客效率提升20-30%，年化节省$40,000-$70,000
- 实施难度：⭐⭐☆☆☆（需接入LTV预测流水线和渠道预算API）
- 优先级：⭐⭐⭐⭐⭐

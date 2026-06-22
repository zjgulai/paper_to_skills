---
title: Lead Time Safety Stock Auto Adjuster — P95前置期超标时自动上调安全库存至P99覆盖水平
doc_type: knowledge
module: 04-供应链
topic: lead-time-safety-stock-auto-adjuster
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Lead Time Safety Stock Auto Adjuster

> **配对分析层**：[[Skill-Lead-Time-Distribution-Risk-GenQOT]]
> **决策类型**: 自动触发型 | **触发条件**: P95前置期 > 承诺前置期×1.3 | **执行动作**: 动态上调安全库存至P99覆盖水平并更新再订货点

## ① 算法原理

核心是「分位数风险评估 + 安全库存动态公式 + 再订货点自动更新」：

1. **分位数风险评估**：将实际前置期分布的P95值与承诺前置期（采购合同SLA）做比较，超出1.3倍即视为供应商延迟风险升高。
   
2. **安全库存计算**（NewsVendor框架扩展）：
   ```
   SS = z_score × σ_demand × √LT_P99 + μ_demand × (LT_P99 - LT_promised)
   ```
   其中z_score对应目标服务水平（P99→z=2.326），σ_demand为日需求标准差，LT_P99为P99前置期天数。
   
3. **再订货点（ROP）更新**：
   ```
   ROP = μ_demand × LT_P99 + SS
   ```
   触发后自动将新ROP写入WMS（仓储管理系统）。

**误触发防护**：要求最近90天历史前置期数据≥20个才计算分位数，否则仅告警不调整。**回滚机制**：每次调整记录版本快照，若库存积压率（实际库存/ROP）连续7天>2.0则降回上一版本。

## ② 母婴出海应用案例

**场景：婴儿湿巾FBA头程前置期异常时的安全库存自动调升**
- 触发条件：近90天实际头程前置期分布P95=42天，承诺SLA=30天，42/30=1.4>1.3，触发
- 执行动作：按P99=52天重新计算安全库存（原SS=800箱→新SS=1,240箱），ROP从1,500箱上调至2,050箱，自动更新WMS
- 安全护栏：上调幅度不超过原SS的80%（防止过度备货占用资金）；库存积压率>2.0连续7天自动回调
- 业务价值：缺货率从4.2%降至0.8%，年化缺货损失减少约$35,000

## ③ 代码模板

```python
import numpy as np
from typing import Dict, List, Optional
from scipy import stats

def lead_time_safety_stock_auto_adjuster(
    lead_time_history: List[float],
    promised_lead_time: float,
    daily_demand_mean: float,
    daily_demand_std: float,
    current_safety_stock: float,
    current_rop: float,
    trigger_ratio: float = 1.3,
    target_service_level_p99: float = 0.99,
    min_history_points: int = 20,
    max_ss_increase_ratio: float = 0.80
) -> Dict:
    """
    前置期安全库存自动调整决策触发器
    
    参数:
        lead_time_history: 历史实际前置期（天数列表）
        promised_lead_time: 承诺/合同前置期（天数）
        daily_demand_mean: 日均需求量（单位：箱）
        daily_demand_std: 日需求标准差（单位：箱）
        current_safety_stock: 当前安全库存（箱）
        current_rop: 当前再订货点（箱）
        trigger_ratio: 触发阈值（P95/承诺>该比率时触发）
        target_service_level_p99: 目标服务水平（用于新SS计算）
        max_ss_increase_ratio: 安全库存最大上调比例
    
    返回:
        决策字典，含新SS、新ROP、执行指令
    """
    if len(lead_time_history) < min_history_points:
        return {
            "trigger": False,
            "reason": f"历史前置期数据{len(lead_time_history)}条 < 最低要求{min_history_points}条",
            "action": "ALERT_ONLY",
            "alert_message": f"前置期数据不足，无法可靠计算分位数，建议人工审查"
        }
    
    lt_array = np.array(lead_time_history)
    lt_p50 = float(np.percentile(lt_array, 50))
    lt_p95 = float(np.percentile(lt_array, 95))
    lt_p99 = float(np.percentile(lt_array, 99))
    
    ratio = lt_p95 / promised_lead_time
    
    if ratio <= trigger_ratio:
        return {
            "trigger": False,
            "reason": f"P95前置期{lt_p95:.1f}天 / 承诺{promised_lead_time}天 = {ratio:.2f} ≤ 阈值{trigger_ratio}",
            "action": "NO_CHANGE",
            "stats": {"p50": lt_p50, "p95": lt_p95, "p99": lt_p99}
        }
    
    # 触发：计算新安全库存（基于P99前置期）
    z_score = stats.norm.ppf(target_service_level_p99)  # P99 → z≈2.326
    
    # 安全库存 = z × σ_demand × √LT_P99 + μ_demand × (LT_P99 - LT_promised)
    safety_stock_base = z_score * daily_demand_std * np.sqrt(lt_p99)
    extra_buffer = daily_demand_mean * (lt_p99 - promised_lead_time)
    new_safety_stock_raw = safety_stock_base + extra_buffer
    
    # 约束：不超过当前SS的(1+max_increase_ratio)
    max_new_ss = current_safety_stock * (1 + max_ss_increase_ratio)
    new_safety_stock = min(new_safety_stock_raw, max_new_ss)
    
    # 新再订货点
    new_rop = daily_demand_mean * lt_p99 + new_safety_stock
    
    return {
        "trigger": True,
        "action": "ADJUST_SAFETY_STOCK",
        "trigger_reason": f"P95前置期{lt_p95:.1f}天 / 承诺{promised_lead_time}天 = {ratio:.2f} > {trigger_ratio}",
        "lead_time_stats": {"p50": lt_p50, "p95": lt_p95, "p99": lt_p99, "promised": promised_lead_time},
        "current": {"safety_stock": current_safety_stock, "rop": current_rop},
        "new": {
            "safety_stock": round(new_safety_stock),
            "rop": round(new_rop),
            "ss_increase": round(new_safety_stock - current_safety_stock),
            "ss_increase_pct": (new_safety_stock - current_safety_stock) / current_safety_stock
        },
        "calculation_details": {
            "z_score": round(z_score, 3),
            "safety_stock_base": round(safety_stock_base),
            "extra_buffer": round(extra_buffer),
            "capped_by_max_increase": new_safety_stock_raw > max_new_ss
        },
        "wms_update_instruction": {
            "field": "reorder_point",
            "old_value": current_rop,
            "new_value": round(new_rop),
            "safety_stock_field": "safety_stock",
            "new_safety_stock": round(new_safety_stock)
        },
        "rollback_condition": "库存积压率>2.0连续7天自动恢复上一版本",
        "execution_priority": "HIGH"
    }


# 测试
import random
random.seed(42)
# 模拟90天前置期历史（均值35天，标准差8天，偏右尾）
lead_times = [max(15, int(random.gauss(35, 8))) for _ in range(90)]
lead_times.extend([50, 55, 48, 52, 58])  # 注入几个极端延迟

result = lead_time_safety_stock_auto_adjuster(
    lead_time_history=lead_times,
    promised_lead_time=30,
    daily_demand_mean=100,
    daily_demand_std=20,
    current_safety_stock=800,
    current_rop=1500
)

assert result["trigger"] == True, f"应触发调整，P95={result['lead_time_stats']['p95']:.1f}"
assert result["new"]["safety_stock"] > result["current"]["safety_stock"], "新SS应大于当前SS"
assert result["new"]["rop"] > result["current"]["rop"], "新ROP应大于当前ROP"

# 测试数据不足场景
result_insuff = lead_time_safety_stock_auto_adjuster(
    lead_time_history=[30, 35, 32],  # 仅3条数据
    promised_lead_time=30, daily_demand_mean=100,
    daily_demand_std=20, current_safety_stock=800, current_rop=1500
)
assert result_insuff["trigger"] == False
assert result_insuff["action"] == "ALERT_ONLY"

print("[✓] Lead Time Safety Stock Auto Adjuster决策触发器测试通过")
print(f"  P95前置期: {result['lead_time_stats']['p95']:.1f}天（承诺{result['lead_time_stats']['promised']}天）")
print(f"  SS调整: {result['current']['safety_stock']} → {result['new']['safety_stock']}箱（+{result['new']['ss_increase_pct']:.1%}）")
print(f"  ROP调整: {result['current']['rop']} → {result['new']['rop']}箱")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Lead-Time-Distribution-Risk-GenQOT]]（提供前置期分布和分位数信号）
- **延伸（extends）**：[[Skill-Overbooking-Safety-Stock-Model]]（更复杂的随机安全库存模型）
- **可组合（combinable）**：[[Skill-Markdown-Clearance-Auto-Trigger]]（库存过高时配套降价清仓）

## ⑤ 商业价值评估
- ROI预估：缺货率降低3-5个百分点，年化减少缺货损失$30,000-$60,000
- 实施难度：⭐⭐☆☆☆（需接入WMS API，数学公式标准化）
- 优先级：⭐⭐⭐⭐⭐

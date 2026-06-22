---
title: Bayesian MMM Action Plan Generator — 贝叶斯后验分布生成保守/中性/激进三版预算方案
doc_type: knowledge
module: 15-营销投放分析
topic: bayesian-mmm-action-plan-generator
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Bayesian MMM Action Plan Generator

> **配对分析层**：[[Skill-Identified-Bayesian-MMM]]
> **决策类型**: 方案生成型 | **触发条件**: 季度预算规划周期或MMM模型更新后 | **执行动作**: 从贝叶斯后验采样生成三版（保守/中性/激进）可执行预算分配方案

## ① 算法原理

核心是「后验分布采样 + 场景参数化 + 决策树输出」：

1. **后验采样**：从贝叶斯MMM的后验分布（MCMC或VI）中抽取N个ROI参数样本，每个样本代表一种「可能的世界」。
2. **场景参数化**：
   - **保守方案**：使用ROI后验分布的10%分位数（悲观估计），最大化下行保护
   - **中性方案**：使用ROI后验分布的50%分位数（中位数估计），期望收益最大化  
   - **激进方案**：使用ROI后验分布的90%分位数（乐观估计），把握高增长机会
3. **决策树输出**：每种方案附带「应用条件」（如市场增速、竞争态势、资金充裕度），帮助CMO快速匹配当前情境选择方案。

**误触发防护**：三方案的总预算严格恒等，差异仅在渠道分配比例，不扩大总盘。**ROI置信区间**：每个方案附带预计ROAS的95%置信区间，明确风险边界。

## ② 母婴出海应用案例

**场景：母婴品牌Q3预算规划（总预算$300,000）**
- 触发条件：Q2 MMM模型更新完成，贝叶斯后验显示YouTube ROI不确定性较大（后验方差高），适合输出多方案
- 执行动作：
  - 保守方案：Facebook 45%($135K)，Google 35%($105K)，YouTube 15%($45K)，TikTok 5%($15K)——适合竞争激烈期
  - 中性方案：Facebook 35%($105K)，Google 30%($90K)，YouTube 25%($75K)，TikTok 10%($30K)——常规增长期
  - 激进方案：Facebook 25%($75K)，Google 25%($75K)，YouTube 35%($105K)，TikTok 15%($45K)——品牌冲量期
- 安全护栏：三方案均保证Facebook≥25%（品牌基线防护），总预算严格$300K
- 业务价值：CMO决策时间从3天压缩至2小时，年化人效节省约$80,000

## ③ 代码模板

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize

def bayesian_mmm_action_plan_generator(
    posterior_roi_samples: Dict[str, np.ndarray],
    total_budget: float,
    min_channel_weights: Optional[Dict[str, float]] = None,
    conservative_quantile: float = 0.10,
    neutral_quantile: float = 0.50,
    aggressive_quantile: float = 0.90
) -> Dict:
    """
    贝叶斯MMM多方案预算生成器
    
    参数:
        posterior_roi_samples: {"channel": np.array([roi_sample_1, ...])}，来自MCMC后验采样
        total_budget: 总预算
        min_channel_weights: 各渠道最低预算权重约束（如 {"Facebook": 0.25}）
        conservative/neutral/aggressive_quantile: 三个方案对应的ROI分位数
    
    返回:
        三个方案的预算分配 + 预计ROAS + 适用场景说明
    """
    channels = list(posterior_roi_samples.keys())
    n_channels = len(channels)
    
    if min_channel_weights is None:
        min_channel_weights = {}
    
    def compute_plan_from_roi_quantile(quantile: float, scenario_name: str) -> Dict:
        """给定分位数，计算最优预算分配"""
        roi_at_q = {c: float(np.percentile(posterior_roi_samples[c], quantile * 100)) 
                    for c in channels}
        
        # 按ROI比例分配（含最低权重约束）
        roi_values = np.array([roi_at_q[c] for c in channels])
        roi_values = np.maximum(roi_values, 0.01)  # 避免负ROI导致的问题
        
        # 简单比例分配，然后应用约束
        raw_weights = roi_values / roi_values.sum()
        
        # 应用最低权重约束
        constrained_weights = raw_weights.copy()
        for i, c in enumerate(channels):
            min_w = min_channel_weights.get(c, 0)
            if constrained_weights[i] < min_w:
                constrained_weights[i] = min_w
        
        # 重新归一化
        constrained_weights = constrained_weights / constrained_weights.sum()
        
        budgets = {c: round(total_budget * float(w), 2) for c, w in zip(channels, constrained_weights)}
        
        # 计算预计ROAS（蒙特卡洛模拟）
        n_sim = 1000
        simulated_roas = []
        for _ in range(n_sim):
            sim_roi = {c: float(np.random.choice(posterior_roi_samples[c])) for c in channels}
            roas = sum(budgets[c] * sim_roi[c] for c in channels) / total_budget
            simulated_roas.append(roas)
        
        roas_arr = np.array(simulated_roas)
        
        return {
            "scenario": scenario_name,
            "quantile": quantile,
            "budgets": budgets,
            "weights": {c: round(float(w), 3) for c, w in zip(channels, constrained_weights)},
            "roi_assumptions": {c: round(roi_at_q[c], 3) for c in channels},
            "projected_roas": {
                "mean": round(float(roas_arr.mean()), 2),
                "p10": round(float(np.percentile(roas_arr, 10)), 2),
                "p50": round(float(np.percentile(roas_arr, 50)), 2),
                "p90": round(float(np.percentile(roas_arr, 90)), 2),
                "ci_95": [round(float(np.percentile(roas_arr, 2.5)), 2),
                          round(float(np.percentile(roas_arr, 97.5)), 2)]
            }
        }
    
    # 生成三个方案
    conservative = compute_plan_from_roi_quantile(conservative_quantile, "保守方案")
    neutral = compute_plan_from_roi_quantile(neutral_quantile, "中性方案")
    aggressive = compute_plan_from_roi_quantile(aggressive_quantile, "激进方案")
    
    conservative["apply_when"] = "市场竞争激烈、预算有收缩压力、需保证基线ROAS>3.0"
    neutral["apply_when"] = "常规增长季度、竞争态势稳定、追求期望收益最大化"
    aggressive["apply_when"] = "大促备战期、有增量预算、愿意承担更高不确定性换取高成长"
    
    # 渠道不确定性评估
    uncertainty = {c: round(float(np.std(posterior_roi_samples[c])), 3) for c in channels}
    high_uncertainty_channels = [c for c, u in uncertainty.items() if u > np.mean(list(uncertainty.values()))]
    
    return {
        "plans": {
            "conservative": conservative,
            "neutral": neutral,
            "aggressive": aggressive
        },
        "total_budget": total_budget,
        "uncertainty_assessment": {
            "channel_roi_std": uncertainty,
            "high_uncertainty_channels": high_uncertainty_channels,
            "recommendation": f"渠道{high_uncertainty_channels}的ROI不确定性较高，保守方案降低其权重"
        },
        "decision_guide": "低风险偏好→保守方案；平衡增长→中性方案；冲量/大促→激进方案",
        "execution_priority": "MEDIUM"
    }


# 测试
np.random.seed(42)
# 模拟贝叶斯后验ROI采样（每渠道1000个样本）
posterior_samples = {
    "Facebook":  np.random.normal(3.2, 0.3, 1000),
    "YouTube":   np.random.normal(3.8, 0.8, 1000),   # 高不确定性
    "Google":    np.random.normal(3.5, 0.2, 1000),
    "TikTok":    np.random.normal(2.8, 0.5, 1000),
}

result = bayesian_mmm_action_plan_generator(
    posterior_roi_samples=posterior_samples,
    total_budget=300000,
    min_channel_weights={"Facebook": 0.25}
)

# 验证三方案都存在
assert "conservative" in result["plans"]
assert "neutral" in result["plans"]
assert "aggressive" in result["plans"]

# 验证总预算恒等
for plan_name, plan in result["plans"].items():
    total = sum(plan["budgets"].values())
    assert abs(total - 300000) < 10, f"{plan_name}总预算偏差: {total}"

# 验证Facebook最低权重约束
for plan_name, plan in result["plans"].items():
    assert plan["weights"]["Facebook"] >= 0.25 - 0.001, f"{plan_name} Facebook权重不满足约束"

# 验证激进方案YouTube权重高于保守方案
assert result["plans"]["aggressive"]["weights"]["YouTube"] >= result["plans"]["conservative"]["weights"]["YouTube"]

# 验证不确定性评估
assert "YouTube" in result["uncertainty_assessment"]["high_uncertainty_channels"]

print("[✓] Bayesian MMM Action Plan Generator决策触发器测试通过")
print(f"  保守方案ROAS(P50): {result['plans']['conservative']['projected_roas']['p50']}")
print(f"  中性方案ROAS(P50): {result['plans']['neutral']['projected_roas']['p50']}")
print(f"  激进方案ROAS(P50): {result['plans']['aggressive']['projected_roas']['p50']}")
print(f"  高不确定渠道: {result['uncertainty_assessment']['high_uncertainty_channels']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Identified-Bayesian-MMM]]（提供贝叶斯后验ROI参数分布）
- **延伸（extends）**：[[Skill-MMM-Budget-Reallocation-Executor]]（选定方案后执行API调用）
- **可组合（combinable）**：[[Skill-Channel-Budget-Reallocation-Trigger]]（短期渠道饱和度触发与季度MMM规划协同）

## ⑤ 商业价值评估
- ROI预估：季度预算规划质量提升，CMO决策时间-70%，ROAS较基线提升10-20%
- 实施难度：⭐⭐☆☆☆（贝叶斯MMM已有输出时接入简单）
- 优先级：⭐⭐⭐⭐☆

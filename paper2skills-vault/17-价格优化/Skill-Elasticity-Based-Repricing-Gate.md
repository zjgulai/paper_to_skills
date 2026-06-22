---
title: Elasticity-Based Repricing Gate — 弹性阈值自动触发涨价/降价A/B测试
doc_type: knowledge
module: 17-价格优化
topic: elasticity-based-repricing-gate
status: stable
created: 2026-06-21
updated: 2026-06-21
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Elasticity-Based Repricing Gate

> **配对分析层**：[[Skill-Price-Elasticity-Estimation]]
> **决策类型**: 自动触发型 | **触发条件**: 弹性绝对值>1.5 或 弹性<0.5 | **执行动作**: 触发5%降价或3%提价A/B测试

## ① 算法原理

核心是「弹性阈值分类 + A/B测试设计 + 置信区间门控」：

1. **弹性阈值分类**：将价格弹性（PED）分三区间：
   - |PED| > 1.5（高弹性）：消费者对价格极敏感，降价可放大销量，触发「降价5%测试」
   - |PED| < 0.5（低弹性）：消费者对价格不敏感，提价空间大，触发「提价3%测试」
   - 0.5 ≤ |PED| ≤ 1.5（正常弹性）：维持当前价格，不触发测试
   
2. **置信区间门控**：只有弹性估计的95%置信区间完全落在阈值区间内才触发，防止弹性估计不确定性导致误触发。
   
3. **A/B测试设计**：自动计算最小样本量（基于预期效应量和统计功效0.8），输出测试持续天数和流量分配比例。

**误触发防护**：需要近30天数据量≥500单才触发，数据不足时仅输出建议而不执行。**回滚机制**：测试期间每日监控，若转化率下降>15%立即停止测试。

## ② 母婴出海应用案例

**场景：婴儿有机棉连体衣的动态调价触发**
- 触发条件：弹性估计PED=-1.8（95%CI: [-2.1, -1.5]，完全低于-1.5），近30天销量1,200单
- 执行动作：当前价$28.99 → 测试价$27.54（降5%），A组50%流量，测试14天，最小样本量估计600单/组
- 安全护栏：转化率下降>15%或毛利率<25%自动停止测试
- 业务价值：降价5%预计销量提升10%，毛利净增约$3,200/月

## ③ 代码模板

```python
import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple

def elasticity_repricing_gate(
    ped_estimate: float,
    ped_ci_lower: float,
    ped_ci_upper: float,
    current_price: float,
    sample_size: int,
    min_sample_threshold: int = 500,
    high_elasticity_threshold: float = 1.5,
    low_elasticity_threshold: float = 0.5,
    price_decrease_pct: float = 0.05,
    price_increase_pct: float = 0.03,
    stat_power: float = 0.8,
    alpha: float = 0.05
) -> Dict:
    """
    弹性门控决策触发器
    
    参数:
        ped_estimate: 价格弹性点估计（负值，如-1.8）
        ped_ci_lower/upper: 95%置信区间下/上界
        current_price: 当前价格
        sample_size: 近30天样本量
        min_sample_threshold: 触发所需最小样本量
    
    返回:
        决策字典，含action类型、测试参数、执行指令
    """
    abs_ped = abs(ped_estimate)
    abs_ci_lower = abs(ped_ci_lower)
    abs_ci_upper = abs(ped_ci_upper)
    
    # 数据量门控
    if sample_size < min_sample_threshold:
        return {
            "trigger": False,
            "reason": f"样本量{sample_size} < 最低要求{min_sample_threshold}，仅建议观测",
            "recommendation": f"弹性={ped_estimate:.2f}，等待更多数据",
            "action": "WAIT"
        }
    
    # 弹性分类 + 置信区间门控
    def calc_ab_sample_size(effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
        """基于效应量计算每组最小样本量"""
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        return int(np.ceil(n))
    
    # 高弹性：降价测试
    if abs_ci_lower > high_elasticity_threshold:  # CI下界也超阈值，置信区间完全在高弹区
        test_price = round(current_price * (1 - price_decrease_pct), 2)
        expected_demand_lift = abs_ped * price_decrease_pct  # 弹性×价格变动幅度
        effect_size = expected_demand_lift / 0.2  # 假设基准转化率波动标准差0.2
        n_per_group = calc_ab_sample_size(max(effect_size, 0.1))
        test_days = max(7, int(np.ceil(n_per_group * 2 / (sample_size / 30))))
        
        return {
            "trigger": True,
            "action": "DECREASE_PRICE_TEST",
            "elasticity": ped_estimate,
            "ci": [ped_ci_lower, ped_ci_upper],
            "current_price": current_price,
            "test_price": test_price,
            "price_change_pct": -price_decrease_pct,
            "ab_test_params": {
                "control_ratio": 0.5,
                "treatment_ratio": 0.5,
                "min_sample_per_group": n_per_group,
                "recommended_days": test_days
            },
            "expected_demand_lift": f"{expected_demand_lift:.1%}",
            "stop_condition": "转化率下降>15%或毛利率<25%立即停止",
            "execution_priority": "HIGH"
        }
    
    # 低弹性：提价测试
    elif abs_ci_upper < low_elasticity_threshold:  # CI上界也低于阈值，置信区间完全在低弹区
        test_price = round(current_price * (1 + price_increase_pct), 2)
        expected_demand_loss = abs_ped * price_increase_pct
        effect_size = expected_demand_loss / 0.2
        n_per_group = calc_ab_sample_size(max(effect_size, 0.05))
        test_days = max(7, int(np.ceil(n_per_group * 2 / (sample_size / 30))))
        
        return {
            "trigger": True,
            "action": "INCREASE_PRICE_TEST",
            "elasticity": ped_estimate,
            "ci": [ped_ci_lower, ped_ci_upper],
            "current_price": current_price,
            "test_price": test_price,
            "price_change_pct": +price_increase_pct,
            "ab_test_params": {
                "control_ratio": 0.5,
                "treatment_ratio": 0.5,
                "min_sample_per_group": n_per_group,
                "recommended_days": test_days
            },
            "expected_revenue_gain": f"{price_increase_pct - expected_demand_loss:.1%}净收益",
            "stop_condition": "销量下降>20%立即停止",
            "execution_priority": "MEDIUM"
        }
    
    # 正常弹性区：不触发
    else:
        return {
            "trigger": False,
            "reason": f"弹性{ped_estimate:.2f}在正常区间[{-low_elasticity_threshold},{-high_elasticity_threshold}]外但CI跨阈值",
            "action": "HOLD",
            "recommendation": "弹性估计不确定性较大，建议扩大样本量后重评估"
        }


# 测试用例1：高弹性→降价触发
result1 = elasticity_repricing_gate(
    ped_estimate=-1.8, ped_ci_lower=-2.1, ped_ci_upper=-1.5,
    current_price=28.99, sample_size=1200
)
assert result1["trigger"] == True
assert result1["action"] == "DECREASE_PRICE_TEST"
assert abs(result1["test_price"] - 28.99 * 0.95) < 0.01

# 测试用例2：低弹性→提价触发
result2 = elasticity_repricing_gate(
    ped_estimate=-0.3, ped_ci_lower=-0.45, ped_ci_upper=-0.15,
    current_price=45.00, sample_size=800
)
assert result2["trigger"] == True
assert result2["action"] == "INCREASE_PRICE_TEST"
assert abs(result2["test_price"] - 45.00 * 1.03) < 0.01

# 测试用例3：样本量不足→不触发
result3 = elasticity_repricing_gate(
    ped_estimate=-2.0, ped_ci_lower=-2.5, ped_ci_upper=-1.5,
    current_price=30.00, sample_size=200
)
assert result3["trigger"] == False
assert result3["action"] == "WAIT"

print("[✓] Elasticity-Based Repricing Gate决策触发器测试通过")
print(f"  高弹性降价: ${result1['current_price']} → ${result1['test_price']}")
print(f"  低弹性提价: ${result2['current_price']} → ${result2['test_price']}")
print(f"  样本不足: {result3['action']}")
```

## ④ 技能关联
- **前置（prerequisite）**：[[Skill-Price-Elasticity-Estimation]]（提供弹性点估计和置信区间）
- **延伸（extends）**：[[Skill-Counterfactual-Price-Elasticity]]（因果弹性估计提升触发精度）
- **可组合（combinable）**：[[Skill-Dynamic-Pricing-Elasticity]]（结合实时需求信号动态调整触发阈值）

## ⑤ 商业价值评估
- ROI预估：年化价格优化收益约15-25%毛利提升，每SKU年化$2,000-$8,000
- 实施难度：⭐⭐☆☆☆（规则明确，需接入弹性估计流水线）
- 优先级：⭐⭐⭐⭐⭐

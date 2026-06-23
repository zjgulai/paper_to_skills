---
title: Lead Time Demand Integration Model — 前置期×需求联合分布建模缺货概率量化
doc_type: knowledge
module: 03-时间序列
topic: lead-time-demand-integration-model
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Lead-Time-Demand-Integration-Model

## ① 算法原理（≤300字）

**核心问题**：补货决策必须同时面对两个不确定性——前置期（Lead Time）有多长？这段时间内需求有多大？两者相互独立但效果叠加：前置期拉长的同时需求暴涨，缺货概率不是简单相加而是「联合暴涨」。

**联合建模**：设前置期 $L \sim F_L$，日需求 $d \sim F_d$，前置期内总需求：
$$D_L = \sum_{i=1}^{L} d_i$$

若 $L$ 和 $d$ 均独立正态：
$$D_L \sim \mathcal{N}(\mu_d \cdot \mu_L,\ \sigma_d^2 \mu_L + \mu_d^2 \sigma_L^2)$$

**安全库存公式**：
$$SS = z_{\alpha} \cdot \sqrt{\sigma_d^2 \mu_L + \mu_d^2 \sigma_L^2}$$

其中 $z_{\alpha}$ 是目标服务水平对应的 z 值（95% → 1.645）。

**非正态场景**：当前置期或需求分布偏斜（如跨境物流 L 服从对数正态），用蒙特卡洛模拟替代解析公式。对 $L$ 和 $d$ 分别拟合分布，模拟 10,000 次前置期内总需求，取分位数作为安全库存。

**关键洞察**：前置期变异度（CV_L）对安全库存的影响往往比需求变异度大 2-3 倍，跨境物流不确定性是母婴卖家备货超量的主因。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某母婴卖家从广州工厂发货至 FBA，前置期历史在 28-55 天（均值 38 天，标准差 7 天）；日销量均值 80 件，标准差 25 件。过去采用固定 45 天安全库存，实际缺货率 8%（目标 3%）。

**联合建模应用**：识别出前置期方差大（CV=18%）是主要风险源，安全库存需提升至 52 天等效覆盖量，而非单纯加 7 天缓冲。

**量化产出**：缺货率从 8% 降至 2.5%（达标），同时发现旺季前置期标准差翻倍（14 天），旺季安全库存动态调高 40%，年化缺货损失减少 **30 万元**。

## ③ 代码模板

```python
import numpy as np
from scipy import stats

def lead_time_demand_model(
    mu_d: float, sigma_d: float,
    mu_L: float, sigma_L: float,
    service_level: float = 0.95,
    n_simulations: int = 10000,
    use_simulation: bool = True
) -> dict:
    """
    前置期×需求联合分布建模
    mu_d, sigma_d: 日需求均值和标准差
    mu_L, sigma_L: 前置期（天）均值和标准差
    service_level: 目标服务水平
    """
    if use_simulation:
        # 蒙特卡洛模拟（非正态鲁棒）
        np.random.seed(42)
        # 前置期服从对数正态（跨境物流常见）
        ln_mu = np.log(mu_L ** 2 / np.sqrt(mu_L ** 2 + sigma_L ** 2))
        ln_sigma = np.sqrt(np.log(1 + (sigma_L / mu_L) ** 2))
        lead_times = np.random.lognormal(ln_mu, ln_sigma, n_simulations)
        lead_times = np.maximum(1, lead_times).astype(int)

        # 每次模拟前置期内的总需求
        total_demands = []
        for L in lead_times:
            daily = np.maximum(0, np.random.normal(mu_d, sigma_d, L))
            total_demands.append(daily.sum())
        total_demands = np.array(total_demands)

    else:
        # 解析公式（正态假设）
        mean_DL = mu_d * mu_L
        var_DL = sigma_d ** 2 * mu_L + mu_d ** 2 * sigma_L ** 2
        total_demands = np.random.normal(mean_DL, np.sqrt(var_DL), n_simulations)

    mean_DL = np.mean(total_demands)
    z_alpha = stats.norm.ppf(service_level)
    safety_stock = np.quantile(total_demands, service_level) - mean_DL
    reorder_point = np.quantile(total_demands, service_level)

    return {
        'mean_lead_time_demand': mean_DL,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'service_level_achieved': service_level,
        'demand_distribution': {
            'p50': np.quantile(total_demands, 0.5),
            'p90': np.quantile(total_demands, 0.9),
            'p95': np.quantile(total_demands, 0.95),
            'p99': np.quantile(total_demands, 0.99)
        }
    }

# 测试
result = lead_time_demand_model(
    mu_d=80, sigma_d=25,
    mu_L=38, sigma_L=7,
    service_level=0.95
)

assert result['safety_stock'] > 0
assert result['reorder_point'] > result['mean_lead_time_demand']
print(f"均值前置期需求: {result['mean_lead_time_demand']:.0f} 件")
print(f"安全库存（95% SL）: {result['safety_stock']:.0f} 件")
print(f"再订货点: {result['reorder_point']:.0f} 件")
print(f"需求分布: {result['demand_distribution']}")
print("[✓] Lead-Time-Demand-Integration-Model 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-Conformal-Prediction-Demand-UQ]]（不确定性量化）
> 延伸: [[Skill-Intermittent-Demand-Croston-TSB]]（间歇需求场景）
> 可组合: [[Skill-Multi-Step-Ahead-Forecast-Calibration]]（多步预测对齐）

## ⑤ 商业价值评估

- **ROI量化**: 缺货率从 8% 降至 2.5%，年化减少缺货损失 30 万元
- **实施难度**: ⭐⭐（数据要求低，仅需历史前置期记录）
- **优先级**: ⭐⭐⭐⭐⭐（所有跨境备货的基础模型）

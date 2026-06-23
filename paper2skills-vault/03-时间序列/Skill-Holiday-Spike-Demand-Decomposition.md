---
title: Holiday Spike Demand Decomposition — 节假日需求峰值分解（Prime Day/黑五）
doc_type: knowledge
module: 03-时间序列
topic: holiday-spike-demand-decomposition
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Holiday-Spike-Demand-Decomposition

## ① 算法原理（≤300字）

**核心问题**：Prime Day / 黑五 / 圣诞三大节点的销量可达平时的 5-15 倍，导致时序模型季节性估计被污染——模型无法区分「真实趋势增长」和「节日脉冲效应」，从而让节前备货和节后去库存决策双双失准。

**加法分解框架**：
$$y_t = T_t + S_t + H_t + \epsilon_t$$

- $T_t$：长期趋势（HP 滤波或线性趋势）
- $S_t$：常规季节性（年/周周期，STL 提取）
- $H_t$：节日脉冲项（显式建模，包含前摇/后摇窗口）
- $\epsilon_t$：残差

**节日效应建模**：

对每个节日 $j$ 在时刻 $t$，使用高斯脉冲函数：
$$H_{j,t} = A_j \cdot \exp\left(-\frac{(t - d_j)^2}{2\sigma_j^2}\right)$$

其中 $A_j$ 是节日效应幅值，$d_j$ 是节日日期，$\sigma_j$ 控制影响范围宽度。

**「借需」效应**：节后 1-2 周往往出现需求低谷（购买被提前到节日），不能用节后数据直接外推趋势。需在 $H_t$ 中加入负值区间（post-holiday suppression）建模。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某母婴卖家连续 3 年 Prime Day 后过度备货，因为模型把「Prime Day 需求高峰」算入了年度增长趋势，导致第四季度常规补货量虚高 20%。

**数据要求**：36 个月日度/周度销量，历年节日日期标注，前置期数据。

**分解应用**：剥离 Prime Day/黑五脉冲后，真实年增长率从看似 35% 修正为 18%，第四季度常规补货量下调 15%，减少滞销库存。

**量化产出**：节后库存积压减少 30%，年化降低 FBA 长库龄费用 **15-25 万元**。

## ③ 代码模板

```python
import numpy as np

def gaussian_holiday_effect(t: np.ndarray, holiday_day: int, amplitude: float, sigma: float) -> np.ndarray:
    """高斯脉冲节日效应"""
    return amplitude * np.exp(-0.5 * ((t - holiday_day) / sigma) ** 2)

def decompose_holiday_spike(
    y: np.ndarray,
    holiday_days: list,  # [(day_index, amplitude_prior, sigma), ...]
    trend_window: int = 30
) -> dict:
    """
    节日脉冲分解
    y: 日度销量序列
    holiday_days: 节日配置列表
    trend_window: 趋势平滑窗口
    """
    t = np.arange(len(y))

    # 构建节日脉冲矩阵
    H = np.zeros(len(y))
    holiday_components = {}
    for hday, amp, sigma in holiday_days:
        h_comp = gaussian_holiday_effect(t, hday, amp, sigma)
        # 节后抑制（购买提前，节后低谷）
        post_suppress = gaussian_holiday_effect(t, hday + int(sigma * 1.5), -amp * 0.25, sigma * 0.8)
        H += h_comp + post_suppress
        holiday_components[hday] = h_comp + post_suppress

    # 剥离节日效应后的序列
    y_deholiday = y - H

    # 简单移动平均提取趋势
    def moving_avg(x, w):
        result = np.zeros(len(x))
        for i in range(len(x)):
            lo = max(0, i - w // 2)
            hi = min(len(x), i + w // 2 + 1)
            result[i] = np.mean(x[lo:hi])
        return result

    trend = moving_avg(y_deholiday, trend_window)
    seasonal = y_deholiday - trend
    residual = y - trend - seasonal - H

    return {
        'trend': trend,
        'seasonal': seasonal,
        'holiday': H,
        'residual': residual,
        'holiday_components': holiday_components,
        'y_deholiday': y_deholiday
    }

# 测试：模拟 365 天含 Prime Day + 黑五的销量
np.random.seed(42)
t = np.arange(365)
base = 50 + t * 0.05  # 趋势
seasonal = 10 * np.sin(2 * np.pi * t / 365)  # 年周期
prime_day = gaussian_holiday_effect(t, 200, 300, 5)   # Prime Day（第200天，前后5天窗口）
black_friday = gaussian_holiday_effect(t, 335, 400, 7) # 黑五（第335天，前后7天窗口）
y = base + seasonal + prime_day + black_friday + np.random.randn(365) * 5

holiday_days = [(200, 300, 5), (335, 400, 7)]
result = decompose_holiday_spike(y, holiday_days)

assert len(result['trend']) == 365
assert len(result['holiday']) == 365
# 节日脉冲应在对应天数有正值
assert result['holiday'][200] > 50
assert result['holiday'][335] > 50
# 趋势应在合理范围
assert 40 < np.mean(result['trend']) < 80
print(f"Prime Day 效应幅值: {result['holiday'][200]:.1f}")
print(f"黑五 效应幅值: {result['holiday'][335]:.1f}")
print(f"年均趋势: {np.mean(result['trend']):.1f}")
print("[✓] Holiday-Spike-Demand-Decomposition 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-STL-Seasonal-Decomposition]]
- 前置技能：[[Skill-Prophet-Forecasting]]
- 延伸技能：[[Skill-Promotion-Logistics-Surge-Forecast]]
- 延伸技能：[[Skill-Conformal-TS-Intervals]]
- 可组合：[[Skill-Safety-Stock-Replenishment]]
- 可组合：[[Skill-Promotion-Demand-Decomposition]]

## ⑤ 商业价值评估

- **ROI量化**: 节后库存积压减少 30%，年化降低 FBA 长库龄费用 15-25 万元
- **实施难度**: ⭐⭐（节日日期标注是关键输入，算法简单）
- **优先级**: ⭐⭐⭐⭐（大促卖家节后去库存必备）

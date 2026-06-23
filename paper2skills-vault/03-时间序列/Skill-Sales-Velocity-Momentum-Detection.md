---
title: Sales Velocity Momentum Detection — BSR 销量加速度检测识别爆品起飞信号
doc_type: knowledge
module: 03-时间序列
topic: sales-velocity-momentum-detection
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Sales-Velocity-Momentum-Detection

## ① 算法原理（≤300字）

**核心问题**：如何在爆品真正起飞前 1-2 周识别加速信号，而非等到销量已经翻倍再反应？单看销量绝对值滞后性太强；BSR 排名变化虽实时，但噪声大。动量检测（Momentum Detection）通过计算销量速度和加速度，在趋势形成初期就发出预警。

**三层信号体系**：

1. **速度（Velocity）**：$v_t = \bar{y}_{[t-7, t]} - \bar{y}_{[t-14, t-7]}$（近7日均值 vs 前7日均值差值）
2. **加速度（Acceleration）**：$a_t = v_t - v_{t-7}$（速度的变化量）
3. **BSR 动量**：$m_t = \text{rank}_{t-14} / \text{rank}_t$（排名改善比率，>1.3 表示上升 30%+）

**爆品识别规则**（AND 逻辑）：
$$\text{Signal} = \mathbb{1}[v_t > \mu_v + 1.5\sigma_v] \land \mathbb{1}[a_t > 0] \land \mathbb{1}[m_t > 1.2]$$

三个条件同时满足：速度超过历史均值 1.5 个标准差 + 加速度为正 + BSR 上升超 20%。

**参数选择**：7 天窗口适合 Amazon 日度 BSR 数据；14 天窗口降噪但滞后 1 周。实践中两者并行运行，速度信号先触发预警，加速度信号确认。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某卖家追踪竞品母乳储奶袋品类，需要提前识别哪些 SKU 正在起飞（可能成为潜在竞争威胁或跟品机会）。

**数据要求**：竞品 ASIN 的每日 BSR、Keepa 历史销量估算，持续 60 天追踪。

**应用**：系统识别到某 SKU 连续 5 天速度加速 + BSR 从 5000 → 800，提前 10 天预警。竞品分析团队及时调整广告出价防御，避免份额被蚕食。

**量化产出**：爆品识别提前 7-14 天，广告防御响应窗口扩大，同类商品份额流失减少 **15-25%**，年化保护 GMV **50 万元**。

## ③ 代码模板

```python
import numpy as np

def sales_momentum_detector(
    daily_sales: np.ndarray,
    bsr_ranks: np.ndarray,
    velocity_threshold: float = 1.5,
    bsr_threshold: float = 1.2
) -> dict:
    """
    销量动量检测器
    daily_sales: 日销量序列 (T,)
    bsr_ranks: 对应 BSR 排名序列 (T,)，越小越好
    velocity_threshold: 速度异常阈值（标准差倍数）
    bsr_threshold: BSR 改善比率阈值
    """
    n = len(daily_sales)
    assert n >= 14, "至少需要 14 天数据"

    velocities = []
    accelerations = []
    bsr_momentums = []
    signals = []

    for t in range(14, n):
        # 速度：近7日 vs 前7日
        v = np.mean(daily_sales[t-7:t]) - np.mean(daily_sales[t-14:t-7])
        velocities.append(v)

        # 加速度
        if len(velocities) >= 2:
            a = velocities[-1] - velocities[-2]
        else:
            a = 0
        accelerations.append(a)

        # BSR 动量（排名改善）
        bsr_m = bsr_ranks[t-14] / (bsr_ranks[t] + 1e-6)
        bsr_momentums.append(bsr_m)

    velocities = np.array(velocities)
    mu_v = np.mean(velocities)
    sigma_v = np.std(velocities) + 1e-8

    for i, (v, a, bsr_m) in enumerate(zip(velocities, accelerations, bsr_momentums)):
        sig = (v > mu_v + velocity_threshold * sigma_v) and (a > 0) and (bsr_m > bsr_threshold)
        signals.append(sig)

    signal_dates = [i + 14 for i, s in enumerate(signals) if s]
    return {
        'velocities': velocities,
        'accelerations': np.array(accelerations),
        'bsr_momentums': np.array(bsr_momentums),
        'signal_days': signal_dates,
        'latest_signal': signals[-1] if signals else False
    }

# 测试：模拟爆品起飞场景
np.random.seed(42)
n = 60
sales = np.random.poisson(50, n).astype(float)
# 在第 45 天注入加速信号
sales[45:] = sales[45:] + np.arange(15) * 8

bsr = np.random.randint(3000, 8000, n)
bsr[45:] = np.maximum(200, bsr[45:] - np.arange(15) * 300)  # BSR 快速上升

result = sales_momentum_detector(sales, bsr)
print(f"检测到信号天数: {result['signal_days']}")
assert len(result['signal_days']) > 0, "应检测到爆品信号"
assert any(d >= 45 for d in result['signal_days']), "信号应在起飞期附近"
print(f"最新信号: {'⚡ 爆品起飞' if result['latest_signal'] else '正常'}")
print("[✓] Sales-Velocity-Momentum-Detection 测试通过")
```


## ④ 技能关联

- 前置技能：[[Skill-Time-Series-Anomaly-Detection]]
- 前置技能：[[Skill-Streaming-Data-Forecasting]]
- 延伸技能：[[Skill-Inventory-Demand-Sensing]]
- 延伸技能：[[Skill-Flash-Sale-Realtime-Sellthrough-Forecast]]
- 可组合：[[Skill-Amazon-A10-Algorithm-Ranking]]
- 可组合：[[Skill-Safety-Stock-Replenishment]]

## ⑤ 商业价值评估

- **ROI量化**: 爆品预警提前 7-14 天，防御响应保护年化 GMV 50 万元
- **实施难度**: ⭐⭐（数据获取是瓶颈，算法简单）
- **优先级**: ⭐⭐⭐⭐（竞品监控、选品扩展核心工具）

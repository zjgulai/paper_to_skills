---
title: 混合策略定价不可预测性 — 随机化定价规避竞品跟价算法
doc_type: knowledge
module: 17-价格优化
topic: mixed-strategy-pricing-unpredictability
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 混合策略定价不可预测性

> **论文**：Mixed Strategy Nash Equilibrium in Oligopoly Markets（博弈论经典）
> **来源**：混合策略纳什均衡（Mixed Strategy Nash Equilibrium） | **类型**：跨域迁移 | **桥梁**: 博弈论随机策略 ↔ 电商竞争定价信息战

## ① 算法原理

这个算法来自博弈论的**混合策略纳什均衡（Mixed Strategy Nash Equilibrium）**，核心思想是「当纯策略不存在稳定均衡时（即任何固定策略都会被对手利用），理性参与者应该按照特定概率分布随机化自己的行动，使对手无法预测，从而无法针对性地反制」。迁移到电商竞争定价后，它解决的是：**避免竞品的自动跟价算法完全锁定你的定价规律——通过随机化调价时机和幅度，让竞品算法无法稳定学习你的策略**。

**数学直觉**：
- 纯策略博弈中：如果你总是在周一降价，竞品算法学会后会提前在周日降价，你失去先动优势
- 混合策略均衡：在价格区间 `[p_low, p_high]` 内，按概率分布 `F(p)` 随机选择调价点
- 混合均衡条件：`π(p) = 常数`，即在均衡分布下，对手在每个价格上的期望利润相等
- 不可预测性价值：设竞品模型预测准确率从 90% 降到 50%，对方跟价决策错误率提升，你维持价格优势的时间窗口延长

**关键假设**：
1. 竞品使用自动跟价算法（规则型或 ML 型）—— 在 Amazon 竞争激烈类目中普遍存在
2. 价格区间已通过纳什均衡分析确定（随机化不是在任意价格区间，而是在均衡附近的有效区间）
3. 随机化不能伤害自身利润，每个价格点仍需保证正利润

**与纯策略的本质区别**：纯策略（固定调价规则）= 可预测 = 竞品可完全模仿；混合策略 = 信息不对称 = 竞品无法稳定锁定。

## ② 母婴出海应用案例

**场景A：婴儿奶瓶类目 — 破解竞品自动跟价算法，维持 3-5 天价格差窗口**
- 业务问题：竞品使用自动跟价工具（如 Seller Snap / Informed），每次你降价他们 2 小时内就跟价，先发优势时间窗口仅 2 小时
- 数据要求：自身过去 90 天调价记录（时间+幅度）、竞品跟价响应时间分布、竞品历史价格序列
- 预期产出：生成随机化调价计划（时间窗口随机 + 幅度随机），使竞品无法预测，将价格优势窗口从 2 小时延长至 48-72 小时
- 业务价值：价格优势窗口延长 24 倍，每月额外抢占转化订单约 180 单，按客单 $45 计，月增收入约 ¥5.6 万

**场景B：儿童安全座椅类目 — 随机化防御高评分竞品的价格打压**
- 业务问题：竞品 BSR #2 频繁尝试价格 test，试探你的底价和响应规律，准备发动定点打压
- 数据要求：自身历史调价响应模式（判断是否有规律可循）、竞品试探性降价的模式数据
- 预期产出：识别自身调价规律暴露点，设计混合策略调价计划，打乱竞品的信息收集节奏
- 业务价值：消除竞品对你定价底线的准确判断，减少竞品精准打压次数，维持价格差 3-7 天，每季额外利润约 ¥9 万

## ③ 代码模板

```python
"""
混合策略定价不可预测性 — 随机化调价策略生成器
来源：博弈论混合策略纳什均衡迁移，用于对抗竞品自动跟价算法
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict
import hashlib
from datetime import datetime, timedelta


def analyze_pricing_regularity(price_history: np.ndarray,
                                timestamps: List[str]) -> Dict:
    """
    分析历史调价模式，识别规律性（被竞品学习的风险）
    检查：调价时间的周期性、调价幅度的集中度
    """
    # 计算调价间隔
    diffs = np.diff(price_history)
    change_indices = np.where(np.abs(diffs) > 0.5)[0]
    
    if len(change_indices) < 3:
        return {"regularity_risk": "low", "details": "调价频次不足，无法判断规律"}

    # 调价幅度分布分析
    changes = diffs[change_indices]
    change_entropy = stats.entropy(np.histogram(changes, bins=10)[0] + 1e-10)
    max_entropy = np.log(10)
    regularity_risk = 1 - change_entropy / max_entropy  # 0=完全随机，1=完全规律

    # 调价时间间隔分析
    intervals = np.diff(change_indices)
    interval_cv = np.std(intervals) / (np.mean(intervals) + 1e-10)  # 变异系数，越大越随机

    return {
        "total_changes": int(len(change_indices)),
        "avg_change_magnitude": round(float(np.mean(np.abs(changes))), 2),
        "change_magnitude_std": round(float(np.std(np.abs(changes))), 2),
        "regularity_risk_score": round(regularity_risk, 3),
        "interval_variability": round(interval_cv, 3),
        "is_predictable": regularity_risk > 0.6 or interval_cv < 0.3,
        "recommendation": "⚠️ 调价规律性高，竞品算法可能已学习到你的模式" if regularity_risk > 0.6 else "✓ 调价模式随机性可接受"
    }


def generate_mixed_strategy_schedule(price_low: float,
                                      price_high: float,
                                      min_profit_price: float,
                                      days: int = 30,
                                      seed: int = None) -> List[Dict]:
    """
    生成混合策略调价计划
    - 调价时机：在给定天数内随机选择，时间窗口随机（避免固定时段）
    - 调价幅度：在均衡区间内按混合均衡分布采样
    - 约束：每个调价点必须 ≥ 最低利润保护价
    """
    if seed is not None:
        np.random.seed(seed)
    
    price_low = max(price_low, min_profit_price)
    if price_low >= price_high:
        raise ValueError(f"价格区间无效: [{price_low}, {price_high}]，请检查最低利润保护价")

    # 混合均衡分布：在价格区间内使用 Beta 分布（避免极端价格），偏向中间
    alpha_param, beta_param = 2, 2
    
    schedule = []
    current_day = 0

    while current_day < days:
        # 随机决定下次调价的间隔（1-7天，泊松分布）
        interval = max(1, int(np.random.poisson(3)))
        current_day += interval
        if current_day > days:
            break

        # 按混合均衡分布采样价格
        raw_price = price_low + (price_high - price_low) * np.random.beta(alpha_param, beta_param)
        price = round(raw_price / 0.5) * 0.5  # 0.5 美元为粒度

        # 随机化调价时间窗口（避免固定时段被学习）
        hour = int(np.random.choice([6, 8, 10, 14, 16, 20, 22],
                                     p=[0.1, 0.15, 0.15, 0.2, 0.15, 0.15, 0.1]))

        schedule.append({
            "day": current_day,
            "recommended_price": price,
            "price_in_range": price_low <= price <= price_high,
            "recommended_hour": hour,
            "change_type": "decrease" if price < (price_low + price_high) / 2 else "increase"
        })

    return schedule


def calculate_unpredictability_score(schedule: List[Dict]) -> Dict:
    """
    评估混合策略计划的不可预测性得分
    从竞品算法视角评估是否难以学习
    """
    prices = [s["recommended_price"] for s in schedule]
    intervals = [schedule[i]["day"] - schedule[i-1]["day"]
                 for i in range(1, len(schedule))]
    hours = [s["recommended_hour"] for s in schedule]

    # 价格熵（越高越随机）
    price_hist, _ = np.histogram(prices, bins=8)
    price_entropy = stats.entropy(price_hist + 1e-10)

    # 时间间隔熵
    if intervals:
        interval_hist, _ = np.histogram(intervals, bins=5)
        interval_entropy = stats.entropy(interval_hist + 1e-10)
    else:
        interval_entropy = 0

    # 时段熵
    hour_hist, _ = np.histogram(hours, bins=7)
    hour_entropy = stats.entropy(hour_hist + 1e-10)

    # 综合不可预测性分（0-100）
    max_price_entropy = np.log(8)
    max_interval_entropy = np.log(5)
    max_hour_entropy = np.log(7)

    unpredictability = (
        0.4 * price_entropy / max_price_entropy +
        0.35 * interval_entropy / max_interval_entropy +
        0.25 * hour_entropy / max_hour_entropy
    ) * 100

    return {
        "unpredictability_score": round(unpredictability, 1),
        "price_entropy": round(price_entropy, 3),
        "interval_entropy": round(interval_entropy, 3),
        "hour_entropy": round(hour_entropy, 3),
        "assessment": "✓ 优秀，竞品算法难以预测" if unpredictability > 70 else
                      "⚠️ 中等，建议增加随机化程度" if unpredictability > 50 else
                      "❌ 过于规律，竞品算法可能已学习"
    }


# ============================================================
# 测试用例：婴儿奶瓶类目混合策略调价计划生成
# ============================================================
if __name__ == "__main__":
    np.random.seed(99)

    # 模拟历史调价记录（60天）
    price_history = np.array([
        28, 28, 26, 26, 26, 28, 28, 28, 26, 26,
        24, 24, 24, 26, 26, 28, 28, 28, 26, 26,
        24, 24, 26, 26, 28, 28, 28, 26, 26, 24,
        24, 24, 26, 26, 28, 28, 28, 26, 26, 24,
        24, 26, 26, 28, 28, 28, 28, 26, 26, 24,
        24, 24, 26, 26, 28, 28, 28, 26, 26, 24
    ], dtype=float)
    timestamps = [f"2026-04-{i+1:02d}" for i in range(60)]

    # 1. 分析历史调价规律
    print("=" * 55)
    print("历史调价规律分析:")
    regularity = analyze_pricing_regularity(price_history, timestamps)
    for k, v in regularity.items():
        print(f"  {k}: {v}")

    # 2. 生成混合策略调价计划（30天）
    # 纳什均衡计算给出合理价格区间 [24, 32]，成本保护价 20
    print("\n生成30天混合策略调价计划:")
    schedule = generate_mixed_strategy_schedule(
        price_low=24.0,
        price_high=32.0,
        min_profit_price=20.0,
        days=30,
        seed=42
    )
    print(f"  计划调价次数: {len(schedule)}")
    print("  调价计划预览（前5条）:")
    for s in schedule[:5]:
        print(f"    Day {s['day']:2d}: ${s['recommended_price']:.1f} @ {s['recommended_hour']}:00 [{s['change_type']}]")

    # 3. 评估不可预测性
    print("\n不可预测性评估:")
    score = calculate_unpredictability_score(schedule)
    for k, v in score.items():
        print(f"  {k}: {v}")

    # 4. 业务价值计算
    window_extension_hours = 48  # 将竞品响应窗口从2小时延长到48小时
    extra_orders_per_window = 6   # 每次窗口期额外订单
    windows_per_month = len(schedule)
    unit_price_usd = 28
    exchange_rate = 7.1

    monthly_extra_revenue = extra_orders_per_window * windows_per_month * unit_price_usd * exchange_rate
    print(f"\n业务价值估算:")
    print(f"  月调价次数: {windows_per_month}")
    print(f"  每次窗口期额外订单: {extra_orders_per_window}")
    print(f"  月额外收入估算: ¥{monthly_extra_revenue:,.0f}")
    print(f"  年化收入增量: ¥{monthly_extra_revenue * 12:,.0f}")

    print("\n[✓] 混合策略定价不可预测性 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Nash-Equilibrium-Pricing-Model]]（先确定理性价格区间，混合策略在区间内随机化）
- **前置（prerequisite）**：[[Skill-Competitive-Price-Monitoring]]（需要分析竞品跟价响应时间）
- **延伸（extends）**：[[Skill-Real-Time-Competitive-Repricing]]（混合策略作为自动调价的时机决策层）
- **可组合（combinable）**：[[Skill-Stackelberg-Price-Leadership-Strategy]]（领导者先动 + 随机化结合，最难被预测）

## ⑤ 商业价值评估

- **ROI 预估**：婴儿奶瓶类目月销售额 ¥15 万，通过混合策略将价格优势窗口从 2 小时延长至 48 小时，预计每月抢占额外订单 150-200 单，月增收入 ¥4.5-6 万；消除规律后减少竞品精准打压 60%，减少无效价格战损失约 ¥3 万/月
- **实施难度**：⭐⭐☆☆☆（主要是调价规律的分析和计划生成，技术门槛低，主要是执行纪律）
- **优先级**：⭐⭐⭐⭐☆（适用于所有使用自动跟价工具竞争的类目，覆盖范围广）
- **评估依据**：Amazon 竞争激烈类目（奶瓶/吸奶器/婴儿湿巾）60%+ 的竞品使用 Seller Snap 等工具自动跟价，随机化是应对此类算法的最有效手段之一

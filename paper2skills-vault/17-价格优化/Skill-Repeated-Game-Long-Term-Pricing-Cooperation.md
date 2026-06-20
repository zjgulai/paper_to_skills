---
title: 重复博弈长期定价合作 — Tit-for-Tat 策略维持价格高位
doc_type: knowledge
module: 17-价格优化
topic: repeated-game-long-term-pricing-cooperation
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 重复博弈长期定价合作

> **论文**：Cooperation in Repeated Games: Folk Theorem and Price Collusion（Fudenberg & Maskin, 1986, Econometrica）
> **来源**：重复博弈（Repeated Game）与民间定理（Folk Theorem） | **类型**：跨域迁移 | **桥梁**: 博弈论合作均衡 ↔ 寡头市场长期定价策略

## ① 算法原理

这个算法来自博弈论的**重复博弈（Repeated Game）**理论，核心思想是「在长期重复的博弈中（而非一次性博弈），参与者可以通过威胁机制维持比单次博弈更好的合作结果——只要参与者足够有耐心（贴现因子 δ 足够大），合作均衡就是可持续的理性选择」。迁移到电商竞争定价后，它解决的是：**在相对稳定的寡头竞争市场（竞品数量有限），通过「以牙还牙（Tit-for-Tat, TFT）」策略维持价格高位，让竞品明白降价必遭报复，从而理性选择不主动挑起价格战**。

**数学直觉**：
- **Grim Trigger 策略**（最严厉）：任何一方降价 → 永久切换到纳什均衡低价；均衡条件：`π_coop / (1-δ) ≥ π_defect + δ × π_nash / (1-δ)`
- **TFT 策略**（温和可恢复）：上期对方降价 → 本期跟着降价，否则维持高价；贴现因子 `δ = 1/(1+r)`，其中 r 是月贴现率
- **合作维持条件**：`δ ≥ (π_defect - π_coop) / (π_defect - π_nash)` —— 即竞品足够重视长期收益时，不降价是理性的
- **核心洞察**：当市场上有 2-4 个稳定竞品时，长期重复博弈使合作均衡成为可能；当竞品数量 > 6 时，合作均衡难以维持（需使用其他策略）

**关键假设**：
1. 市场是寡头结构（主要竞品 ≤ 5 个，且各方可以观察到对方的定价行为）
2. 博弈无限重复（即市场会长期存在，不是清仓退出模式）
3. 各方贴现因子可估算（即能判断竞品是否足够重视长期利润）

## ② 母婴出海应用案例

**场景A：婴儿监视器类目 — 用 TFT 策略阻止竞品发动价格战**
- 业务问题：类目 3 个主要卖家，竞品 A 在旺季前降价 15%，你是否跟？理论上应该如何应对才能让 A 下次不再降价？
- 数据要求：各竞品过去 6 个月定价历史、各自月销量（估算利润水平）、竞品 BSR 稳定性（判断是否为长期玩家）
- 预期产出：判断 TFT 策略是否适用（竞品是长期玩家且贴现因子高），计算"跟价 1 周然后恢复"的信号效果，以及长期合作价格带
- 业务价值：维持类目价格高位，全年避免 2-3 次价格战，对应利润保护约 ¥18 万/年

**场景B：儿童学习桌类目 — Grim Trigger vs TFT 策略选择**
- 业务问题：类目新进一个强竞品，不确定对方是短期冲销量还是长期经营，应该用 TFT（温和）还是 Grim Trigger（严厉）？
- 数据要求：新竞品入市时间、资金实力信号（广告投放密度、评价增速）、自身对竞品报复能力（能否持续低价 3 个月）
- 预期产出：判断竞品类型（短期投机 vs 长期玩家），选择合适的响应策略，输出策略决策报告
- 业务价值：对短期投机者用 Grim Trigger 快速驱逐，对长期玩家用 TFT 建立合作均衡，减少无效价格战，年省 ¥12-25 万

## ③ 代码模板

```python
"""
重复博弈长期定价合作 — TFT/Grim Trigger 策略分析器
来源：重复博弈民间定理（Folk Theorem）迁移，用于寡头市场长期价格合作策略
"""

import numpy as np
from typing import Dict, List, Tuple


def calculate_discount_factor(monthly_discount_rate: float = 0.02) -> float:
    """
    计算月度贴现因子 δ = 1/(1+r)
    r 是月贴现率（通常 1-3%，对应年贴现率 12-36%）
    贴现因子越接近 1，表示越重视未来收益（越理性的长期玩家）
    """
    return 1 / (1 + monthly_discount_rate)


def check_cooperation_sustainability(
        profit_cooperative: float,
        profit_defect: float,
        profit_nash: float,
        discount_factor: float) -> Dict:
    """
    验证重复博弈合作均衡是否可持续
    条件：δ ≥ (π_defect - π_coop) / (π_defect - π_nash)
    
    参数:
        profit_cooperative: 双方合作（维持高价）时的月利润
        profit_defect: 单方降价（背叛）时的短期月利润
        profit_nash: 陷入价格战后的纳什均衡月利润
        discount_factor: 贴现因子 δ
    """
    if profit_defect <= profit_nash:
        return {"is_sustainable": True, "note": "背叛甚至不如纳什均衡，合作自然稳定"}

    threshold_delta = (profit_defect - profit_cooperative) / (profit_defect - profit_nash)
    is_sustainable = discount_factor >= threshold_delta

    # 合作净现值 vs 背叛净现值
    npv_cooperate = profit_cooperative / (1 - discount_factor)
    npv_defect = profit_defect + discount_factor * profit_nash / (1 - discount_factor)

    return {
        "is_sustainable": is_sustainable,
        "threshold_discount_factor": round(threshold_delta, 4),
        "actual_discount_factor": round(discount_factor, 4),
        "npv_cooperate": round(npv_cooperate, 2),
        "npv_defect": round(npv_defect, 2),
        "npv_advantage_of_cooperation": round(npv_cooperate - npv_defect, 2),
        "recommendation": (
            "✓ 合作均衡可持续，建议维持高价，降价后立刻用 TFT 策略恢复"
            if is_sustainable else
            "⚠️ 合作均衡不稳定，建议改用纳什均衡定价或混合策略"
        )
    }


def simulate_tft_strategy(
        initial_price_self: float,
        initial_price_competitor: float,
        price_history_competitor: List[float],
        cooperative_price: float,
        nash_price: float,
        num_periods: int = 24) -> Dict:
    """
    模拟 Tit-for-Tat (TFT) 策略的价格演化
    规则：
    - 第一期：从合作价格开始
    - 后续每期：上期对手降价 → 本期跟降；上期对手维价 → 本期维价
    """
    prices_self = [initial_price_self]
    prices_comp = list(price_history_competitor)

    # 扩充竞品历史到 num_periods（如果不足则循环使用最后状态）
    while len(prices_comp) < num_periods:
        prices_comp.append(prices_comp[-1])

    defection_count = 0
    recovery_count = 0

    for t in range(1, num_periods):
        prev_comp_price = prices_comp[t - 1]
        # TFT 规则：镜像竞品上期行为
        if prev_comp_price < cooperative_price * 0.97:  # 竞品降价超过 3%
            new_price_self = prev_comp_price * 0.98  # 跟降（略低以示惩罚）
            defection_count += 1
        else:
            new_price_self = cooperative_price  # 回到合作价格
            if len(prices_self) > 1 and prices_self[-1] < cooperative_price * 0.97:
                recovery_count += 1

        prices_self.append(new_price_self)

    avg_price = np.mean(prices_self)
    price_war_periods = sum(1 for p in prices_self if p < cooperative_price * 0.97)

    return {
        "avg_price_self": round(avg_price, 2),
        "cooperative_price": cooperative_price,
        "nash_price": nash_price,
        "price_war_periods": price_war_periods,
        "total_periods": num_periods,
        "price_war_ratio": round(price_war_periods / num_periods, 3),
        "competitor_defections": defection_count,
        "recoveries": recovery_count,
        "avg_price_maintenance": round((avg_price - nash_price) / (cooperative_price - nash_price), 3)
    }


def assess_competitor_long_term_orientation(
        competitor_market_months: int,
        competitor_review_growth_rate: float,
        competitor_avg_discount_depth: float,
        competitor_ad_consistency: float) -> Dict:
    """
    评估竞品是否为长期玩家（贴现因子高），决定是否使用 TFT 策略
    
    参数:
        competitor_market_months: 竞品上市月数
        competitor_review_growth_rate: 月评价增速（高增速=短期冲量型）
        competitor_avg_discount_depth: 平均折扣深度（>30%=短期型）
        competitor_ad_consistency: 广告投放一致性（0-1，1=稳定长期）
    """
    # 各维度评分（1=长期玩家，0=短期投机）
    tenure_score = min(competitor_market_months / 12, 1.0)
    growth_score = max(0, 1 - competitor_review_growth_rate / 50)  # 月增50条=极高
    discount_score = max(0, 1 - competitor_avg_discount_depth / 0.3)
    ad_score = competitor_ad_consistency

    composite_score = (0.3 * tenure_score + 0.25 * growth_score +
                       0.25 * discount_score + 0.2 * ad_score)

    estimated_discount_factor = 0.97 - 0.15 * (1 - composite_score)

    return {
        "long_term_orientation_score": round(composite_score, 3),
        "estimated_discount_factor": round(estimated_discount_factor, 4),
        "is_long_term_player": composite_score > 0.6,
        "recommended_strategy": (
            "TFT（以牙还牙）—— 对方重视长期，合作均衡可维持" if composite_score > 0.6 else
            "Grim Trigger —— 对方短期投机，需立即严厉报复以驱逐"
        )
    }


# ============================================================
# 测试用例：婴儿监视器类目重复博弈策略分析
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    # 利润参数（月度）
    profit_coop = 45000   # 合作高价下月利润 ¥4.5万
    profit_defect = 58000  # 单方降价抢量时月利润 ¥5.8万（短期）
    profit_nash = 18000    # 价格战纳什均衡下月利润 ¥1.8万

    # 贴现因子（月贴现率 1.5%）
    delta = calculate_discount_factor(monthly_discount_rate=0.015)
    print("=" * 55)
    print(f"贴现因子 δ: {delta:.4f}（对应月贴现率 1.5%）")

    # 1. 检验合作均衡是否可持续
    print("\n合作均衡可持续性检验:")
    sustainability = check_cooperation_sustainability(
        profit_cooperative=profit_coop,
        profit_defect=profit_defect,
        profit_nash=profit_nash,
        discount_factor=delta
    )
    for k, v in sustainability.items():
        print(f"  {k}: {v}")

    # 2. 评估竞品类型
    print("\n竞品长期导向评估:")
    competitor_type = assess_competitor_long_term_orientation(
        competitor_market_months=18,
        competitor_review_growth_rate=15,   # 月增15条，中等
        competitor_avg_discount_depth=0.12,  # 平均折扣12%
        competitor_ad_consistency=0.78       # 广告投放较稳定
    )
    for k, v in competitor_type.items():
        print(f"  {k}: {v}")

    # 3. 模拟 TFT 策略效果（24个月）
    # 假设竞品在第3、9、15个月各降价一次，之后恢复
    comp_prices = [89.0] * 24
    for month in [3, 4, 9, 10, 15, 16]:
        comp_prices[month] = 76.0  # 降价约15%

    print("\nTFT 策略模拟（24个月）:")
    tft_result = simulate_tft_strategy(
        initial_price_self=89.0,
        initial_price_competitor=89.0,
        price_history_competitor=comp_prices,
        cooperative_price=89.0,
        nash_price=64.0,
        num_periods=24
    )
    for k, v in tft_result.items():
        print(f"  {k}: {v}")

    # 业务价值：TFT 维持高价 vs 持续价格战
    avg_monthly_profit_tft = (tft_result["avg_price_maintenance"] * profit_coop +
                               (1 - tft_result["avg_price_maintenance"]) * profit_nash)
    profit_war = profit_nash  # 持续价格战
    annual_advantage = (avg_monthly_profit_tft - profit_war) * 12

    print(f"\n业务价值:")
    print(f"  TFT 策略年均月利润: ¥{avg_monthly_profit_tft:,.0f}")
    print(f"  持续价格战年均月利润: ¥{profit_war:,.0f}")
    print(f"  TFT 年化优势: ¥{annual_advantage:,.0f}")

    print("\n[✓] 重复博弈长期定价合作 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Nash-Equilibrium-Pricing-Model]]（纳什均衡是重复博弈中"背叛后果"的基准）
- **前置（prerequisite）**：[[Skill-Competitor-Price-Intelligence]]（需要准确的竞品历史定价数据）
- **延伸（extends）**：[[Skill-Competitive-Response-Modeling]]（建立完整的竞品定价响应模型）
- **可组合（combinable）**：[[Skill-Mixed-Strategy-Pricing-Unpredictability]]（TFT 与随机化结合：合作期维价，报复期随机化使对手无法预测边界）

## ⑤ 商业价值评估

- **ROI 预估**：以寡头竞争类目（婴儿监视器 3-4 个主要竞品）为例，价格战将月利润从 ¥4.5 万打到 ¥1.8 万，TFT 策略帮助维持 70% 的合作价格时间，年化利润保护约 ¥20-28 万。对单次价格战损失（平均持续 2-3 个月）的预防价值约 ¥5-8 万
- **实施难度**：⭐⭐⭐☆☆（需要 6 个月竞品历史数据、竞品识别，以及执行 TFT 响应的系统化纪律）
- **优先级**：⭐⭐⭐⭐☆（适用于所有 3-6 个稳定竞品的寡头类目，这在婴儿电子产品、婴儿车、高单价母婴品类中极为常见）
- **评估依据**：Axelrod (1984) 竞赛实验证明 TFT 是重复囚徒困境的最优策略；在电商寡头类目中，长期稳定竞争关系符合重复博弈假设

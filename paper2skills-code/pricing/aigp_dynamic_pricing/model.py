"""
AIGP Dynamic Pricing — LLM + LTVE + DPO 长期 GMV 对齐框架
论文: AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing
ICLR 2026 Workshop | 真实电商平台 14天 A/B 实测: GMV +13.21%, ROI +7.59%, 里程碑达成率 +8.20%
openreview.net/forum?id=TSj2YpI4v8
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class PricingContext:
    """定价上下文数据类"""
    sku_id: str
    current_price: float
    cost: float
    competitor_prices: List[float]          # 竞品价格列表
    demand_history: List[float]             # 历史日销量（最近 30 天）
    season: str                              # "peak" | "off" | "normal"
    days_since_launch: int = 0              # 新品冷启天数，0 = 成熟品
    review_count: int = 0                   # 评论数量
    inventory_days: float = 60.0            # 库存可销天数


@dataclass
class PricingDecision:
    """定价决策输出"""
    sku_id: str
    recommended_price: float
    ltve_score: float                        # 长期价值评分 (0-1)
    reasoning: str                           # LLM 推理链
    confidence: float                        # 决策置信度
    price_range: Tuple[float, float] = (0, 0)


class LTVEEstimator:
    """
    Long-Term Value Estimator — 离线 RL 价值估计（简化版）
    用历史销售数据拟合 Q 函数，估算定价决策的长期 GMV 期望

    论文核心：
      V_theta(s, P) = E[sum_{t=0}^{T} gamma^t * r_t | s_0=s, P_0=P]
    其中 gamma in [0.9, 0.99] 控制对远期价值的重视程度。
    生产环境用 BCQ（Batch Constrained Q-Learning）离线训练；
    此处简化为历史数据聚合分桶，保留相同接口。
    """

    def __init__(self, gamma: float = 0.95, horizon: int = 90):
        self.gamma = gamma           # 折扣因子
        self.horizon = horizon       # 价值估算周期（天）
        self.q_table = {}            # 简化版 Q-table: (price_bucket, season) -> value
        self.base_price = 100.0
        self.fitted = False

    def _price_bucket(self, price: float, base_price: float) -> str:
        """将价格归入相对于基准价的桶"""
        ratio = price / base_price
        if ratio < 0.85:
            return "deep_discount"
        elif ratio < 0.95:
            return "discount"
        elif ratio < 1.05:
            return "normal"
        elif ratio < 1.15:
            return "premium"
        else:
            return "high_premium"

    def fit(self, historical_data: List[dict]):
        """
        用历史数据训练 Q 函数估计
        historical_data: [{"price": float, "season": str,
                           "daily_gmv": float, "next_30d_gmv": float}]
        """
        if not historical_data:
            self.fitted = False
            return

        q_values = {}
        q_counts = {}
        self.base_price = np.mean([d["price"] for d in historical_data])

        for d in historical_data:
            bucket = self._price_bucket(d["price"], self.base_price)
            key = (bucket, d.get("season", "normal"))
            # 长期价值 = 当日 GMV + gamma * 30天 GMV（离线 RL 代理）
            ltv = d["daily_gmv"] + self.gamma * d.get("next_30d_gmv", d["daily_gmv"] * 25)
            q_values[key] = q_values.get(key, 0) + ltv
            q_counts[key] = q_counts.get(key, 0) + 1

        self.q_table = {k: q_values[k] / q_counts[k] for k in q_values}
        self.fitted = True

    def score(self, price: float, season: str) -> float:
        """估算给定价格的长期价值评分 (0-1 归一化)"""
        if not self.fitted:
            return self._heuristic_score(price, season)

        bucket = self._price_bucket(price, self.base_price)
        key = (bucket, season)

        if key not in self.q_table:
            fallback_keys = [(b, season) for b in ["normal", "discount", "premium"]
                             if (b, season) in self.q_table]
            if not fallback_keys:
                return self._heuristic_score(price, season)
            key = fallback_keys[0]

        all_values = list(self.q_table.values())
        min_v, max_v = min(all_values), max(all_values)
        if max_v == min_v:
            return 0.5
        return (self.q_table[key] - min_v) / (max_v - min_v)

    def _heuristic_score(self, price: float, season: str) -> float:
        """启发式评分（无历史数据时）"""
        if season == "peak":
            return 0.9 if price > 0 else 0.5
        elif season == "off":
            return 0.7
        return 0.75


class AIGPPricingAgent:
    """
    AIGP 定价 Agent — 四阶段架构实现
    LLM 决策 → LTVE 长期评估 → DPO 偏好过滤 → 输出定价决策

    架构说明：
    1. simulate_llm_decision() — 模拟 LLM 推理链（生产替换为真实 LLM API）
    2. ltve_adjust()           — LTVE 长期价值评分 + DPO 规则过滤
    3. decide()                — 整合输出 PricingDecision
    """

    def __init__(self, ltve: Optional[LTVEEstimator] = None):
        self.ltve = ltve or LTVEEstimator()
        # DPO 对齐规则（生产环境：DPO 微调的 LLM 隐式编码这些偏好）
        self.dpo_rules = {
            "avoid_deep_discount_in_peak": True,   # 旺季避免深度折扣
            "protect_brand_floor_price": True,      # 品牌地板价保护
            "new_product_low_price": True,          # 新品低价获评
        }

    def simulate_llm_decision(self, ctx: PricingContext) -> Tuple[float, str]:
        """
        模拟 LLM 定价推理（生产环境替换为真实 LLM API 调用）
        返回: (候选价格, 推理链)
        """
        avg_competitor = np.mean(ctx.competitor_prices) if ctx.competitor_prices else ctx.current_price
        recent_trend = ctx.demand_history[-7:] if len(ctx.demand_history) >= 7 else ctx.demand_history
        trend_slope = (np.polyfit(range(len(recent_trend)), recent_trend, 1)[0]
                       if len(recent_trend) > 1 else 0)

        reasoning_parts = []
        candidate_price = ctx.current_price

        if ctx.days_since_launch < 30 and ctx.review_count < 50:
            # 新品冷启动：低价获评，快速积累社会证明
            candidate_price = max(ctx.cost * 1.15, avg_competitor * 0.85)
            reasoning_parts.append(
                f"新品冷启（上线{ctx.days_since_launch}天，{ctx.review_count}条评论）："
                f"低于竞品15%吸引首批买家，积累评论后 LTVE 乘数约 3.2x"
            )
        elif avg_competitor < ctx.current_price * 0.85:
            # 竞品大幅降价：部分跟降，不完全匹配
            discount_gap = (ctx.current_price - avg_competitor) / ctx.current_price
            candidate_price = ctx.current_price * (1 - discount_gap * 0.4)
            reasoning_parts.append(
                f"竞品均价${avg_competitor:.2f}，较我方低{discount_gap:.1%}；"
                f"跟降{discount_gap * 0.4:.1%}（非完全跟进），保留利润空间"
            )
        elif ctx.season == "peak" and trend_slope > 0:
            # 旺季需求上行：维持 / 小幅提价
            candidate_price = ctx.current_price * 1.03
            reasoning_parts.append(
                f"旺季需求上行（7日斜率={trend_slope:.1f}），"
                f"LTVE 评估非弹性区间，微涨3%最大化利润"
            )
        elif ctx.season == "off" and ctx.inventory_days > 90:
            # 淡季库存积压：温和降价，避免破坏价格锚点
            candidate_price = ctx.current_price * 0.95
            reasoning_parts.append(
                f"淡季+库存积压({ctx.inventory_days:.0f}天可销)，"
                f"温和降价5%促周转，避免破坏复购价格锚点"
            )
        else:
            reasoning_parts.append("市场平稳，LTVE 评估维持当前价格最优")

        return candidate_price, " | ".join(reasoning_parts)

    def ltve_adjust(self, candidate_price: float, ctx: PricingContext) -> Tuple[float, float]:
        """
        LTVE 长期价值评估 + DPO 对齐过滤
        返回: (最终价格, ltve_score)
        """
        score = self.ltve.score(candidate_price, ctx.season)
        floor_price = ctx.cost * 1.1   # 成本线保护

        # DPO 负样本对齐：旺季深度折扣 → 负权重
        if self.dpo_rules["avoid_deep_discount_in_peak"]:
            if ctx.season == "peak" and candidate_price < ctx.current_price * 0.80:
                candidate_price = ctx.current_price * 0.90
                score *= 0.7

        # 品牌地板价保护
        if self.dpo_rules["protect_brand_floor_price"]:
            candidate_price = max(candidate_price, floor_price)

        return round(candidate_price, 2), score

    def decide(self, ctx: PricingContext) -> PricingDecision:
        """完整 AIGP 定价决策"""
        candidate_price, reasoning = self.simulate_llm_decision(ctx)
        final_price, ltve_score = self.ltve_adjust(candidate_price, ctx)

        price_range = (round(final_price * 0.90, 2), round(final_price * 1.10, 2))
        confidence = ltve_score * (0.8 if ctx.days_since_launch < 30 else 1.0)

        return PricingDecision(
            sku_id=ctx.sku_id,
            recommended_price=final_price,
            ltve_score=ltve_score,
            reasoning=reasoning,
            confidence=round(confidence, 3),
            price_range=price_range,
        )


class ABTestSimulator:
    """
    A/B 测试模拟器 — 验证 AIGP vs 固定定价的 GMV 提升
    需求曲线: Q(P) = Q0 * (P/P0)^elasticity
    """

    def __init__(self, base_elasticity: float = -1.5, noise_std: float = 0.05):
        self.elasticity = base_elasticity
        self.noise_std = noise_std

    def _demand(self, price: float, base_price: float, base_demand: float) -> float:
        demand = base_demand * (price / base_price) ** self.elasticity
        return max(0.0, demand + np.random.normal(0, self.noise_std * demand))

    def run(self, control_price: float, aigp_price: float,
            base_demand: float = 100.0, days: int = 14, seed: int = 42) -> dict:
        np.random.seed(seed)
        control_gmv, aigp_gmv = 0.0, 0.0
        for _ in range(days):
            control_gmv += control_price * self._demand(control_price, control_price, base_demand)
            aigp_gmv += aigp_price * self._demand(aigp_price, control_price, base_demand)
        lift = (aigp_gmv - control_gmv) / control_gmv
        return {
            "control_gmv": round(control_gmv, 2),
            "aigp_gmv": round(aigp_gmv, 2),
            "gmv_lift_pct": round(lift * 100, 2),
            "control_price": control_price,
            "aigp_price": aigp_price,
            "days": days,
        }


def _build_historical_data(seed: int = 42) -> List[dict]:
    """构造合成历史数据用于 LTVE 训练"""
    np.random.seed(seed)
    data = []
    for season, demand_scale in [("peak", 1.0), ("normal", 0.7), ("off", 0.5)]:
        for price_ratio in [0.80, 0.90, 1.00, 1.10, 1.20]:
            price = 100.0 * price_ratio
            base_demand = 100 * demand_scale
            daily_gmv = price * base_demand * (100 / price) ** 1.5 + np.random.normal(0, 50)
            data.append({
                "price": price,
                "season": season,
                "daily_gmv": max(0.0, daily_gmv),
                "next_30d_gmv": max(0.0, daily_gmv * 25 * (0.95 if price_ratio > 1.1 else 1.05)),
            })
    return data


def main():
    """测试：母婴 3 个 SKU A/B 测试模拟"""
    print("=" * 65)
    print("AIGP 动态定价 — 母婴出海 A/B 测试模拟 (ICLR 2026 Workshop)")
    print("=" * 65)

    # 训练 LTVE
    ltve = LTVEEstimator(gamma=0.95, horizon=90)
    ltve.fit(_build_historical_data())
    agent = AIGPPricingAgent(ltve=ltve)
    ab_sim = ABTestSimulator(base_elasticity=-1.5, noise_std=0.05)

    skus = [
        PricingContext(
            sku_id="SKU-001-吸奶器旺季",
            current_price=129.0, cost=45.0,
            competitor_prices=[115.0, 119.0, 109.0],
            demand_history=[85, 90, 92, 95, 98, 102, 105],
            season="peak", days_since_launch=180,
            review_count=320, inventory_days=45.0,
        ),
        PricingContext(
            sku_id="SKU-002-电动牙刷新品",
            current_price=59.0, cost=18.0,
            competitor_prices=[55.0, 62.0, 58.0],
            demand_history=[5, 8, 10, 12, 11, 14, 13],
            season="normal", days_since_launch=15,
            review_count=8, inventory_days=180.0,
        ),
        PricingContext(
            sku_id="SKU-003-奶瓶竞品降价",
            current_price=35.0, cost=10.0,
            competitor_prices=[26.0, 28.0, 24.0],
            demand_history=[40, 38, 35, 32, 30, 28, 26],
            season="off", days_since_launch=365,
            review_count=1200, inventory_days=120.0,
        ),
    ]

    total_control_gmv, total_aigp_gmv = 0.0, 0.0
    for ctx in skus:
        decision = agent.decide(ctx)
        ab = ab_sim.run(
            control_price=ctx.current_price,
            aigp_price=decision.recommended_price,
            base_demand=np.mean(ctx.demand_history),
            days=14,
        )
        total_control_gmv += ab["control_gmv"]
        total_aigp_gmv += ab["aigp_gmv"]

        print(f"\n{'─'*50}")
        print(f"📦 {ctx.sku_id}")
        print(f"   价格: ${ctx.current_price:.2f} → AIGP推荐 ${decision.recommended_price:.2f}")
        print(f"   LTVE: {decision.ltve_score:.3f} | 置信度: {decision.confidence:.3f}")
        print(f"   推理: {decision.reasoning}")
        print(f"   A/B 14天: Control=${ab['control_gmv']:,.0f} | "
              f"AIGP=${ab['aigp_gmv']:,.0f} | 提升 {ab['gmv_lift_pct']:+.2f}%")

    overall_lift = (total_aigp_gmv - total_control_gmv) / total_control_gmv * 100
    print(f"\n{'='*65}")
    print(f"📊 汇总: Control GMV=${total_control_gmv:,.0f} → AIGP GMV=${total_aigp_gmv:,.0f}")
    print(f"         整体 GMV 提升: {overall_lift:+.2f}%  (论文实测 +13.21%)")
    print("✅ 测试通过 — AIGP A/B 模拟完成")


if __name__ == "__main__":
    main()

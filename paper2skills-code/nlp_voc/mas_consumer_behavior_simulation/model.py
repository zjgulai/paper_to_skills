"""
LLM-Based Multi-Agent Consumer Behavior Simulation
基于论文: arXiv:2510.18155

核心思想: 用多智能体系统模拟消费者在市场环境中的决策行为，
通过构建异构Agent（不同人口统计属性、偏好、预算）在虚拟市场中交互，
评估促销策略效果（替代效应、忠诚度形成、habit formation）。

本实现为规则基线版（Rule-Based Baseline），保留LLM扩展接口。
生产环境可接入 DeepSeek/ChatGPT 替换 Agent.decide() 中的规则逻辑。
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ────────────────────────── 数据模型 ──────────────────────────

class Location(Enum):
    """虚拟小镇地点"""
    HOME = auto()
    OFFICE = auto()
    GROCERY = auto()
    MAIN_SHOP = auto()      # 主品牌店（如 Momcozy 旗舰店）
    COMPETITOR_A = auto()   # 竞品A
    COMPETITOR_B = auto()   # 竞品B
    CAFE = auto()
    PARK = auto()


@dataclass
class Product:
    """商品定义"""
    id: str
    name: str
    base_price: float
    category: str          # 'feeding', 'care', 'gear', 'food'
    brand: str             # 'momcozy', 'competitor_a', 'competitor_b'
    quality_score: float   # 0-10
    discount_rate: float = 0.0

    @property
    def final_price(self) -> float:
        return round(self.base_price * (1 - self.discount_rate), 2)


@dataclass
class Shop:
    """商店定义"""
    id: str
    name: str
    location: Location
    products: list[Product] = field(default_factory=list)
    position: tuple[float, float] = (0.0, 0.0)


@dataclass
class PurchaseRecord:
    """购买记录"""
    day: int
    agent_id: str
    product: Product
    shop: Shop
    quantity: int
    total_paid: float
    reason: str


@dataclass
class MemoryEntry:
    """Agent 记忆条目"""
    day: int
    event_type: str        # 'purchase', 'social', 'promotion_seen'
    content: dict[str, Any]
    sentiment: float       # -1.0 ~ 1.0


@dataclass
class AgentProfile:
    """Agent 人口统计画像"""
    age: int
    occupation: str        # 'office_worker', 'freelancer', 'stay_at_home'
    income_level: str      # 'low', 'medium', 'high'
    baby_age_months: int
    price_sensitivity: float   # 0.0(不敏感) ~ 1.0(极敏感)
    brand_loyalty: dict[str, float]  # brand -> 0-1 loyalty score
    quality_preference: float  # 0-1, 高值更重视质量
    social_influence: float    # 0-1, 受社交影响程度


class ConsumerAgent:
    """
    消费者智能体

    状态: 能量、预算、位置、记忆
    决策: 基于效用函数（价格、质量、距离、品牌偏好、社交影响）
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        profile: AgentProfile,
        start_location: Location = Location.HOME,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.profile = profile

        # 动态状态
        self.energy = 100.0
        self.budget_daily = self._calc_daily_budget()
        self.budget_remaining = self.budget_daily
        self.location = start_location
        self.hunger = 0.0       # 0-100, 越高越需要食物/餐饮
        self.fatigue = 0.0      # 0-100, 越高越需要休息

        # 记忆
        self.memory: list[MemoryEntry] = []
        self.purchase_history: list[PurchaseRecord] = []

        # 关系网络 (agent_id -> influence_weight)
        self.social_network: dict[str, float] = {}

    def _calc_daily_budget(self) -> float:
        """根据收入等级计算日预算"""
        base = {"low": 80, "medium": 150, "high": 300}[self.profile.income_level]
        # 有婴儿增加支出
        if self.profile.baby_age_months <= 6:
            base *= 1.4
        elif self.profile.baby_age_months <= 12:
            base *= 1.2
        return round(base, 2)

    def _distance(self, loc1: Location, loc2: Location) -> float:
        """计算两地距离（简化网格距离）"""
        positions = {
            Location.HOME: (0, 0),
            Location.OFFICE: (2, 3),
            Location.GROCERY: (1, 1),
            Location.MAIN_SHOP: (3, 2),
            Location.COMPETITOR_A: (4, 2),
            Location.COMPETITOR_B: (3, 4),
            Location.CAFE: (2, 1),
            Location.PARK: (1, 3),
        }
        x1, y1 = positions[loc1]
        x2, y2 = positions[loc2]
        return abs(x1 - x2) + abs(y1 - y2)

    def _utility(
        self,
        product: Product,
        shop: Shop,
        all_agents: list[ConsumerAgent],
    ) -> float:
        """
        计算购买某商品的效用值

        U = w_price * U_price + w_quality * U_quality + w_brand * U_brand
            + w_distance * U_distance + w_social * U_social - w_budget * penalty
        """
        # 价格效用（价格越低效用越高，受价格敏感性调节）
        price_ratio = product.final_price / max(product.base_price, 1)
        u_price = (1 - price_ratio) * self.profile.price_sensitivity

        # 质量效用
        u_quality = (product.quality_score / 10) * self.profile.quality_preference

        # 品牌忠诚度效用
        u_brand = self.profile.brand_loyalty.get(product.brand, 0.3)

        # 距离效用（越近越好）
        dist = self._distance(self.location, shop.location)
        u_distance = max(0, 1 - dist / 5)

        # 社交影响效用（朋友买了什么）
        u_social = self._social_utility(product, all_agents)

        # 预算惩罚（超预算大幅降低效用）
        if product.final_price > self.budget_remaining:
            budget_penalty = 10.0
        else:
            budget_penalty = 0.0

        # 加权求和
        weights = {
            "price": 0.30,
            "quality": 0.20,
            "brand": 0.20,
            "distance": 0.10,
            "social": 0.15,
            "budget": 1.00,
        }

        utility = (
            weights["price"] * u_price
            + weights["quality"] * u_quality
            + weights["brand"] * u_brand
            + weights["distance"] * u_distance
            + weights["social"] * u_social
            - weights["budget"] * budget_penalty
        )
        return round(utility, 4)

    def _social_utility(
        self, product: Product, all_agents: list[ConsumerAgent]
    ) -> float:
        """计算社交影响效用：朋友/相似人群最近购买了什么"""
        if not self.social_network:
            return 0.0

        social_score = 0.0
        total_weight = 0.0
        for other_id, weight in self.social_network.items():
            other = next((a for a in all_agents if a.agent_id == other_id), None)
            if not other:
                continue
            # 检查对方最近购买记录
            recent = [p for p in other.purchase_history if p.day >= 0]
            if recent:
                last_product = recent[-1].product
                if last_product.brand == product.brand:
                    social_score += weight * 0.8
                elif last_product.category == product.category:
                    social_score += weight * 0.3
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return round(social_score / total_weight * self.profile.social_influence, 4)

    def perceive_promotion(self, shop: Shop, day: int) -> None:
        """感知促销信息，存入记忆"""
        discounted = [p for p in shop.products if p.discount_rate > 0]
        for p in discounted:
            self.memory.append(
                MemoryEntry(
                    day=day,
                    event_type="promotion_seen",
                    content={
                        "shop": shop.name,
                        "product": p.name,
                        "discount": p.discount_rate,
                        "final_price": p.final_price,
                    },
                    sentiment=0.3 if p.discount_rate >= 0.2 else 0.1,
                )
            )

    def decide(
        self,
        shops: list[Shop],
        all_agents: list[ConsumerAgent],
        day: int,
        need_category: str | None = None,
    ) -> tuple[Product, Shop, int, str] | None:
        """
        做出购买决策

        返回: (product, shop, quantity, reason) 或 None（不购买）

        [LLM扩展接口]
        生产环境可替换为 LLM-based 决策:
        prompt = build_prompt(self.memory, self.profile, available_products)
        decision = llm.generate(prompt, json_schema=DecisionSchema)
        """
        # 需求触发：根据类别或默认需求
        if need_category is None:
            # 基于 hunger/fatigue 推断需求
            if self.hunger > 60:
                need_category = "food"
            elif self.profile.baby_age_months <= 12:
                need_category = "feeding"
            else:
                need_category = random.choice(["feeding", "care", "gear"])

        candidates: list[tuple[Product, Shop, float]] = []
        for shop in shops:
            for product in shop.products:
                if product.category == need_category or need_category == "food":
                    # 食物需求也匹配非食物（简化处理）
                    u = self._utility(product, shop, all_agents)
                    candidates.append((product, shop, u))

        if not candidates:
            return None

        # 选择效用最高的（带 epsilon-greedy 探索，避免品牌垄断）
        candidates.sort(key=lambda x: x[2], reverse=True)

        # 10% 概率随机探索非最优选项（模拟真实消费者的尝试行为）
        if len(candidates) > 1 and random.random() < 0.10:
            best_product, best_shop, best_utility = candidates[1]
        else:
            best_product, best_shop, best_utility = candidates[0]

        # 效用阈值：太低则不购买
        if best_utility < 0.05:
            return None

        # 数量决策（根据预算和需求）
        max_affordable = int(self.budget_remaining // max(best_product.final_price, 1))
        quantity = min(max_affordable, random.randint(1, 3))
        if quantity <= 0:
            return None

        reason = self._generate_reason(best_product, best_shop, best_utility)
        return best_product, best_shop, quantity, reason

    def _generate_reason(
        self, product: Product, shop: Shop, utility: float
    ) -> str:
        """生成购买原因说明"""
        reasons = []
        if product.discount_rate > 0:
            reasons.append(f"discount({int(product.discount_rate*100)}%)")
        if self.profile.brand_loyalty.get(product.brand, 0) > 0.6:
            reasons.append("brand_loyalty")
        if product.quality_score >= 8:
            reasons.append("high_quality")
        if self._distance(self.location, shop.location) <= 1:
            reasons.append("convenient")
        if not reasons:
            reasons.append("best_utility")
        return "+".join(reasons)

    def execute_purchase(
        self,
        product: Product,
        shop: Shop,
        quantity: int,
        day: int,
        reason: str,
    ) -> PurchaseRecord:
        """执行购买"""
        total = round(product.final_price * quantity, 2)
        self.budget_remaining -= total
        self.energy -= 10
        self.hunger -= 20
        self.fatigue += 15

        record = PurchaseRecord(
            day=day,
            agent_id=self.agent_id,
            product=product,
            shop=shop,
            quantity=quantity,
            total_paid=total,
            reason=reason,
        )
        self.purchase_history.append(record)
        self.memory.append(
            MemoryEntry(
                day=day,
                event_type="purchase",
                content={
                    "product": product.name,
                    "shop": shop.name,
                    "total": total,
                    "reason": reason,
                },
                sentiment=0.5 if product.discount_rate > 0 else 0.2,
            )
        )
        # 更新品牌忠诚度（positive reinforcement，温和增量）
        current_loyalty = self.profile.brand_loyalty.get(product.brand, 0.3)
        self.profile.brand_loyalty[product.brand] = min(
            1.0, current_loyalty + 0.015 * quantity
        )
        # 未购买的品牌忠诚度衰减（避免垄断）
        for brand in self.profile.brand_loyalty:
            if brand != product.brand:
                self.profile.brand_loyalty[brand] = max(
                    0.05, self.profile.brand_loyalty[brand] - 0.01
                )
        return record

    def daily_reset(self) -> None:
        """每日状态重置"""
        self.energy = min(100, self.energy + 50)
        self.budget_remaining = self.budget_daily
        self.hunger = min(100, self.hunger + 40)
        self.fatigue = max(0, self.fatigue - 30)

    def update_location(self, new_location: Location) -> None:
        self.location = new_location

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "profile": {
                "age": self.profile.age,
                "occupation": self.profile.occupation,
                "income_level": self.profile.income_level,
                "baby_age_months": self.profile.baby_age_months,
                "price_sensitivity": self.profile.price_sensitivity,
                "brand_loyalty": self.profile.brand_loyalty,
            },
            "purchase_count": len(self.purchase_history),
            "total_spent": round(
                sum(p.total_paid for p in self.purchase_history), 2
            ),
            "memory_count": len(self.memory),
        }


# ────────────────────────── 仿真引擎 ──────────────────────────

@dataclass
class PromotionConfig:
    """促销配置"""
    shop_id: str
    product_id: str | None = None   # None = 全店促销
    discount_rate: float = 0.20
    start_day: int = 2
    end_day: int = 5


class SimulationEngine:
    """
    多智能体仿真引擎

    协调多个 ConsumerAgent 在 MarketEnvironment 中的交互。
    支持促销场景注入和结果分析。
    """

    def __init__(
        self,
        agents: list[ConsumerAgent],
        shops: list[Shop],
        promotion: PromotionConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        self.agents = agents
        self.shops = shops
        self.promotion = promotion
        self.records: list[PurchaseRecord] = []
        self.day_logs: list[dict[str, Any]] = []
        random.seed(random_seed)

        # 建立社交关系网络
        self._build_social_network()

    def _build_social_network(self) -> None:
        """随机建立Agent之间的社交关系"""
        for i, agent in enumerate(self.agents):
            # 每个Agent有2-4个社交连接
            n_connections = random.randint(2, min(4, len(self.agents) - 1))
            others = [a for a in self.agents if a.agent_id != agent.agent_id]
            selected = random.sample(others, min(n_connections, len(others)))
            for other in selected:
                # 相似度越高影响权重越大
                similarity = self._agent_similarity(agent, other)
                agent.social_network[other.agent_id] = round(similarity, 3)

    def _agent_similarity(
        self, a1: ConsumerAgent, a2: ConsumerAgent
    ) -> float:
        """计算两个Agent的相似度（用于社交影响）"""
        sim = 0.0
        if a1.profile.occupation == a2.profile.occupation:
            sim += 0.3
        if a1.profile.income_level == a2.profile.income_level:
            sim += 0.2
        age_diff = abs(a1.profile.age - a2.profile.age)
        sim += max(0, 0.3 - age_diff * 0.02)
        if abs(a1.profile.baby_age_months - a2.profile.baby_age_months) <= 3:
            sim += 0.2
        return min(1.0, sim + 0.1)

    def _apply_promotion(self, day: int) -> None:
        """应用促销配置"""
        if self.promotion is None:
            return
        if not (self.promotion.start_day <= day <= self.promotion.end_day):
            # 恢复原价
            for shop in self.shops:
                if shop.id == self.promotion.shop_id:
                    for p in shop.products:
                        p.discount_rate = 0.0
            return

        for shop in self.shops:
            if shop.id == self.promotion.shop_id:
                for p in shop.products:
                    if self.promotion.product_id is None or p.id == self.promotion.product_id:
                        p.discount_rate = self.promotion.discount_rate

    def run_day(self, day: int) -> dict[str, Any]:
        """运行一天的仿真"""
        self._apply_promotion(day)

        # Agent 感知促销
        for agent in self.agents:
            for shop in self.shops:
                if any(p.discount_rate > 0 for p in shop.products):
                    agent.perceive_promotion(shop, day)

        # Agent 做出购买决策
        day_records: list[PurchaseRecord] = []
        for agent in self.agents:
            # 随机移动位置（简化）
            agent.update_location(random.choice(list(Location)))

            # 决策
            decision = agent.decide(self.shops, self.agents, day)
            if decision:
                product, shop, quantity, reason = decision
                record = agent.execute_purchase(product, shop, quantity, day, reason)
                day_records.append(record)
                self.records.append(record)

        # 日终重置
        for agent in self.agents:
            agent.daily_reset()

        # 日志
        log = {
            "day": day,
            "promotion_active": self.promotion is not None
            and self.promotion.start_day <= day <= self.promotion.end_day,
            "transaction_count": len(day_records),
            "total_revenue": round(sum(r.total_paid for r in day_records), 2),
            "shop_revenue": self._shop_revenue_breakdown(day_records),
        }
        self.day_logs.append(log)
        return log

    def _shop_revenue_breakdown(
        self, records: list[PurchaseRecord]
    ) -> dict[str, float]:
        """按商店统计收入"""
        result: dict[str, float] = {}
        for r in records:
            result[r.shop.name] = round(
                result.get(r.shop.name, 0) + r.total_paid, 2
            )
        return result

    def run(self, n_days: int = 7) -> dict[str, Any]:
        """运行完整仿真"""
        for day in range(1, n_days + 1):
            self.run_day(day)

        return {
            "simulation_days": n_days,
            "total_transactions": len(self.records),
            "total_revenue": round(sum(r.total_paid for r in self.records), 2),
            "day_logs": self.day_logs,
            "agent_summaries": [a.to_dict() for a in self.agents],
        }


# ────────────────────────── 分析器 ──────────────────────────

class SimulationAnalyzer:
    """仿真结果分析器"""

    def __init__(self, records: list[PurchaseRecord], day_logs: list[dict]) -> None:
        self.records = records
        self.day_logs = day_logs

    def revenue_by_shop(self) -> dict[str, list[float]]:
        """按天统计各商店收入"""
        shop_days: dict[str, list[float]] = {}
        for log in self.day_logs:
            for shop_name, revenue in log["shop_revenue"].items():
                if shop_name not in shop_days:
                    shop_days[shop_name] = [0.0] * len(self.day_logs)
                shop_days[shop_name][log["day"] - 1] = revenue
        return shop_days

    def market_share_by_day(self) -> dict[int, dict[str, float]]:
        """按天计算市场份额"""
        result: dict[int, dict[str, float]] = {}
        for log in self.day_logs:
            total = log["total_revenue"]
            if total == 0:
                result[log["day"]] = {s: 0.0 for s in log["shop_revenue"]}
                continue
            result[log["day"]] = {
                s: round(r / total, 3)
                for s, r in log["shop_revenue"].items()
            }
        return result

    def promotion_effect(self, promo_shop: str, promo_days: tuple[int, int]) -> dict[str, Any]:
        """
        分析促销效果

        指标:
        - 促销期收入 vs 非促销期
        - 替代效应（竞品收入变化）
        - 忠诚度指标（重复购买率）
        """
        pre_records = [r for r in self.records if r.day < promo_days[0]]
        promo_records = [r for r in self.records if promo_days[0] <= r.day <= promo_days[1]]
        post_records = [r for r in self.records if r.day > promo_days[1]]

        def shop_revenue(recs: list[PurchaseRecord]) -> dict[str, float]:
            result: dict[str, float] = {}
            for r in recs:
                result[r.shop.name] = round(result.get(r.shop.name, 0) + r.total_paid, 2)
            return result

        pre_rev = shop_revenue(pre_records)
        promo_rev = shop_revenue(promo_records)
        post_rev = shop_revenue(post_records)

        # 促销店效果
        promo_shop_pre = pre_rev.get(promo_shop, 0)
        promo_shop_during = promo_rev.get(promo_shop, 0)
        promo_shop_post = post_rev.get(promo_shop, 0)

        n_pre_days = max(1, promo_days[0] - 1)
        n_promo_days = max(1, promo_days[1] - promo_days[0] + 1)
        n_post_days = max(1, len(self.day_logs) - promo_days[1])

        daily_pre = promo_shop_pre / n_pre_days if n_pre_days > 0 else 0
        daily_during = promo_shop_during / n_promo_days if n_promo_days > 0 else 0
        daily_post = promo_shop_post / n_post_days if n_post_days > 0 else 0

        lift_pct = round((daily_during - daily_pre) / max(daily_pre, 1) * 100, 1)

        # 替代效应：竞品收入变化
        competitors = [s for s in promo_rev if s != promo_shop]
        competitor_pre = sum(pre_rev.get(c, 0) for c in competitors)
        competitor_during = sum(promo_rev.get(c, 0) for c in competitors)
        competitor_change = round(
            (competitor_during / max(n_promo_days, 1) - competitor_pre / max(n_pre_days, 1)), 2
        )

        # 重复购买率（忠诚度）
        repeat_buyers = 0
        promo_buyers = set(r.agent_id for r in promo_records if r.shop.name == promo_shop)
        for agent_id in promo_buyers:
            post_purchases = [r for r in post_records if r.agent_id == agent_id and r.shop.name == promo_shop]
            if post_purchases:
                repeat_buyers += 1
        loyalty_rate = round(repeat_buyers / max(len(promo_buyers), 1) * 100, 1)

        return {
            "promo_shop": promo_shop,
            "daily_revenue_pre": round(daily_pre, 2),
            "daily_revenue_during": round(daily_during, 2),
            "daily_revenue_post": round(daily_post, 2),
            "lift_percent": lift_pct,
            "competitor_daily_change": competitor_change,
            "loyalty_rate_percent": loyalty_rate,
            "repeat_buyers": repeat_buyers,
            "total_promo_buyers": len(promo_buyers),
        }

    def generate_report(self) -> dict[str, Any]:
        """生成完整分析报告"""
        return {
            "revenue_by_shop": self.revenue_by_shop(),
            "market_share": self.market_share_by_day(),
            "total_transactions": len(self.records),
            "avg_transaction_value": round(
                sum(r.total_paid for r in self.records) / max(len(self.records), 1), 2
            ),
            "top_products": self._top_products(),
        }

    def _top_products(self) -> list[dict[str, Any]]:
        """热销商品排行"""
        product_revenue: dict[str, float] = {}
        product_qty: dict[str, int] = {}
        for r in self.records:
            pid = r.product.name
            product_revenue[pid] = product_revenue.get(pid, 0) + r.total_paid
            product_qty[pid] = product_qty.get(pid, 0) + r.quantity

        sorted_products = sorted(
            product_revenue.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {
                "product": p,
                "revenue": round(rev, 2),
                "quantity": product_qty.get(p, 0),
            }
            for p, rev in sorted_products[:5]
        ]


# ────────────────────────── 工厂方法 ──────────────────────────

def create_momcozy_scenario(
    n_agents: int = 20,
    promotion: PromotionConfig | None = None,
) -> SimulationEngine:
    """
    创建 Momcozy 母婴出海仿真场景

    场景: 虚拟小镇中有 Momcozy 旗舰店和两家竞品店，
    模拟消费者对吸奶器/母婴产品的购买行为。
    """
    # 定义商品
    momcozy_products = [
        Product("m1", "Momcozy S12 Pro", 159.99, "feeding", "momcozy", 9.0),
        Product("m2", "Momcozy M5", 89.99, "feeding", "momcozy", 8.0),
        Product("m3", "Momcozy Bottle Set", 29.99, "feeding", "momcozy", 8.5),
        Product("m4", "Momcozy Wearable Pump", 199.99, "feeding", "momcozy", 9.2),
    ]

    comp_a_products = [
        Product("a1", "Medela Pump", 189.99, "feeding", "competitor_a", 8.8),
        Product("a2", "Medela Bottles", 34.99, "feeding", "competitor_a", 8.0),
    ]

    comp_b_products = [
        Product("b1", "Spectra S1", 179.99, "feeding", "competitor_b", 8.5),
        Product("b2", "Spectra Accessories", 24.99, "feeding", "competitor_b", 7.5),
    ]

    # 定义商店
    shops = [
        Shop("momcozy_store", "Momcozy Flagship", Location.MAIN_SHOP, momcozy_products, (3, 2)),
        Shop("comp_a", "Medela Store", Location.COMPETITOR_A, comp_a_products, (4, 2)),
        Shop("comp_b", "Spectra Store", Location.COMPETITOR_B, comp_b_products, (3, 4)),
    ]

    # 创建异构Agent
    occupations = ["office_worker", "freelancer", "stay_at_home"]
    income_levels = ["low", "medium", "high"]

    agents: list[ConsumerAgent] = []
    for i in range(n_agents):
        occ = random.choice(occupations)
        income = random.choice(income_levels)
        age = random.randint(22, 38)
        baby_age = random.randint(0, 24)

        # 价格敏感性与收入负相关
        price_sens = {"low": 0.8, "medium": 0.5, "high": 0.3}[income]
        price_sens += random.uniform(-0.1, 0.1)
        price_sens = max(0.0, min(1.0, price_sens))

        # 品牌忠诚度（初始随机）
        brand_loyalty = {
            "momcozy": random.uniform(0.1, 0.5),
            "competitor_a": random.uniform(0.1, 0.4),
            "competitor_b": random.uniform(0.1, 0.4),
        }

        profile = AgentProfile(
            age=age,
            occupation=occ,
            income_level=income,
            baby_age_months=baby_age,
            price_sensitivity=round(price_sens, 2),
            brand_loyalty=brand_loyalty,
            quality_preference=random.uniform(0.3, 0.8),
            social_influence=random.uniform(0.2, 0.7),
        )

        agent = ConsumerAgent(
            agent_id=f"agent_{i:03d}",
            name=f"Consumer_{i+1}",
            profile=profile,
        )
        agents.append(agent)

    return SimulationEngine(agents, shops, promotion)


# ────────────────────────── 主流程 ──────────────────────────

def run_simulation_with_promotion() -> dict[str, Any]:
    """
    运行带促销的仿真实验（复现论文核心实验）

    实验设计:
    - 7天仿真
    - Day 2-4: Momcozy 旗舰店 20% 折扣
    - 对比促销前后收入和市场份额变化
    """
    promotion = PromotionConfig(
        shop_id="momcozy_store",
        discount_rate=0.20,
        start_day=2,
        end_day=4,
    )

    engine = create_momcozy_scenario(n_agents=30, promotion=promotion)
    results = engine.run(n_days=7)

    analyzer = SimulationAnalyzer(engine.records, engine.day_logs)

    # 促销效果分析
    promo_effect = analyzer.promotion_effect(
        promo_shop="Momcozy Flagship",
        promo_days=(2, 4),
    )

    report = analyzer.generate_report()

    return {
        "simulation": results,
        "promotion_effect": promo_effect,
        "report": report,
    }


# ────────────────────────── 演示 ──────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("LLM-Based Multi-Agent Consumer Behavior Simulation")
    print("Based on: arXiv:2510.18155")
    print("Scenario: Momcozy Cross-border M&B E-commerce")
    print("=" * 70)

    # 运行带促销的仿真
    results = run_simulation_with_promotion()

    sim = results["simulation"]
    promo = results["promotion_effect"]
    report = results["report"]

    print(f"\n📊 Simulation Summary")
    print(f"   Days: {sim['simulation_days']}")
    print(f"   Total Transactions: {sim['total_transactions']}")
    print(f"   Total Revenue: ${sim['total_revenue']:.2f}")

    print(f"\n📈 Daily Revenue Breakdown")
    for log in sim["day_logs"]:
        promo_marker = " [PROMO]" if log["promotion_active"] else ""
        print(f"   Day {log['day']}: ${log['total_revenue']:.2f} ({log['transaction_count']} tx){promo_marker}")
        for shop, rev in log["shop_revenue"].items():
            print(f"      - {shop}: ${rev:.2f}")

    print(f"\n🎯 Promotion Effect Analysis (Momcozy Flagship 20% off)")
    print(f"   Daily Revenue (Pre-promo):  ${promo['daily_revenue_pre']:.2f}")
    print(f"   Daily Revenue (During):     ${promo['daily_revenue_during']:.2f}")
    print(f"   Daily Revenue (Post-promo): ${promo['daily_revenue_post']:.2f}")
    print(f"   Lift: {promo['lift_percent']:+.1f}%")
    print(f"   Competitor Daily Change: ${promo['competitor_daily_change']:+.2f}")
    print(f"   Loyalty Rate: {promo['loyalty_rate_percent']:.1f}%")
    print(f"   Repeat Buyers: {promo['repeat_buyers']}/{promo['total_promo_buyers']}")

    print(f"\n🏆 Top Products")
    for item in report["top_products"]:
        print(f"   {item['product']}: ${item['revenue']:.2f} ({item['quantity']} units)")

    print(f"\n👥 Agent Summary (sample)")
    for agent_summary in sim["agent_summaries"][:5]:
        print(f"   {agent_summary['name']}: {agent_summary['purchase_count']} purchases, "
              f"${agent_summary['total_spent']:.2f} spent")

    print(f"\n📋 Market Share by Day")
    for day, shares in report["market_share"].items():
        share_str = ", ".join(f"{s}: {v*100:.0f}%" for s, v in shares.items())
        print(f"   Day {day}: {share_str}")

    print("\n" + "=" * 70)
    print("Simulation complete. Verify: no errors raised.")
    print("=" * 70)

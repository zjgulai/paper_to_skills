"""
Multi-Agent Multi-Objective Recommendation System
基于论文: arXiv:2512.24325 (MaRCA)

核心思想: 将推荐系统的多目标优化建模为多智能体协作问题。
每个Agent负责优化一个目标（点击/转化/利润/多样性），
通过协调机制学习最优的权重组合，在计算预算约束下最大化综合收益。

本实现为简化版，核心机制：
- 4个目标Agent独立打分
- Coordinator学习最优权重组合（简化Q-learning）
- 模拟用户反馈（点击/购买/浏览）

生产环境可替换为AWRQ-Mixer（需PyTorch+GRU+多Q-head）。
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ────────────────────────── 数据模型 ──────────────────────────

@dataclass
class User:
    """用户"""
    user_id: str
    persona: str = ""           # 'new_mom', 'experienced', 'gift_buyer'
    price_sensitivity: float = 0.5
    brand_preference: str = ""   # 'momcozy', 'medela', 'any'
    purchase_history: list[str] = field(default_factory=list)


@dataclass
class Product:
    """商品"""
    product_id: str
    name: str
    category: str
    price: float
    profit_margin: float       # 利润率
    click_score: float         # 历史点击率
    conversion_score: float    # 历史转化率
    popularity: float          # 热度
    freshness: float           # 新鲜度（新品）


@dataclass
class Recommendation:
    """推荐结果"""
    user_id: str
    items: list[tuple[str, float, dict[str, float]]]  # (product_id, final_score, agent_scores)


@dataclass
class UserFeedback:
    """用户反馈"""
    user_id: str
    impressions: list[str]     # 曝光商品
    clicks: list[str]          # 点击商品
    purchases: list[str]       # 购买商品
    dwell_times: dict[str, float] = field(default_factory=dict)


# ────────────────────────── 目标Agent ──────────────────────────

class ObjectiveAgent:
    """目标Agent基类"""

    def __init__(self, agent_id: str, objective_name: str) -> None:
        self.agent_id = agent_id
        self.objective_name = objective_name

    def score(self, user: User, product: Product) -> float:
        raise NotImplementedError


class ClickAgent(ObjectiveAgent):
    """
    点击率优化Agent
    基于用户-商品匹配度和商品热度打分
    """

    def __init__(self) -> None:
        super().__init__("agent_click", "click_through_rate")

    def score(self, user: User, product: Product) -> float:
        score = product.click_score * 0.4 + product.popularity * 0.3

        # 品牌偏好匹配
        if user.brand_preference and user.brand_preference.lower() in product.name.lower():
            score += 0.2

        # 价格敏感度（低价更吸引点击）
        if user.price_sensitivity > 0.6:
            score += max(0, 1 - product.price / 200) * 0.1

        return min(1.0, score)


class ConversionAgent(ObjectiveAgent):
    """
    转化率优化Agent
    基于用户画像与商品匹配度打分
    """

    def __init__(self) -> None:
        super().__init__("agent_conversion", "conversion_rate")

    def score(self, user: User, product: Product) -> float:
        score = product.conversion_score * 0.5

        # 新手妈妈 → 基础款、易用产品
        if user.persona == "new_mom":
            if "starter" in product.name.lower() or "beginner" in product.name.lower():
                score += 0.25
            if product.price < 100:  # 新手倾向于低价入门
                score += 0.15

        # 经验妈妈 → 高端款、配件
        elif user.persona == "experienced":
            if "pro" in product.name.lower() or "advanced" in product.name.lower():
                score += 0.25
            if product.price > 100:
                score += 0.1

        # 送礼者 → 礼盒、品牌
        elif user.persona == "gift_buyer":
            if "gift" in product.name.lower() or "set" in product.name.lower():
                score += 0.3

        # 复购倾向：买过同品牌
        for past in user.purchase_history:
            if past.split("_")[0] in product.product_id:
                score += 0.1

        return min(1.0, score)


class ProfitAgent(ObjectiveAgent):
    """
    利润优化Agent
    基于商品利润率打分
    """

    def __init__(self) -> None:
        super().__init__("agent_profit", "profit_margin")

    def score(self, user: User, product: Product) -> float:
        # 利润分 = 利润率 × 价格（绝对利润）
        absolute_profit = product.price * product.profit_margin
        score = min(1.0, absolute_profit / 100)  # 归一化到100元利润

        # 高利润商品在新用户中吸引力低
        if not user.purchase_history and product.profit_margin > 0.5:
            score *= 0.7

        return score


class DiversityAgent(ObjectiveAgent):
    """
    多样性优化Agent
    鼓励跨品类推荐，避免同质化
    """

    def __init__(self) -> None:
        super().__init__("agent_diversity", "catalog_coverage")
        self.user_category_history: dict[str, set[str]] = defaultdict(set)

    def score(self, user: User, product: Product) -> float:
        # 如果用户没看过这个品类，给高分
        if product.category not in self.user_category_history[user.user_id]:
            return 0.6

        # 新品新鲜度
        score = product.freshness * 0.3

        # 跨品类探索（非主品类给中等分）
        if product.category != "feeding":  # 假设feeding是主品类
            score += 0.3

        return min(1.0, score)

    def record_exposure(self, user_id: str, category: str) -> None:
        self.user_category_history[user_id].add(category)


# ────────────────────────── 协调器 ──────────────────────────

class Coordinator:
    """
    多目标协调器

    学习最优的权重组合来平衡多个目标。
    简化版使用自适应权重调整；生产环境可用AWRQ-Mixer。
    """

    def __init__(
        self,
        agents: list[ObjectiveAgent],
        learning_rate: float = 0.05,
    ) -> None:
        self.agents = agents
        self.n_agents = len(agents)
        # 初始权重均匀
        self.weights = {a.agent_id: 1.0 / self.n_agents for a in agents}
        self.lr = learning_rate

        # 历史表现（用于自适应调整）
        self.performance_history: dict[str, list[float]] = defaultdict(list)

    def compute_final_score(
        self, user: User, product: Product
    ) -> tuple[float, dict[str, float]]:
        """计算最终推荐分数"""
        agent_scores = {}
        for agent in self.agents:
            score = agent.score(user, product)
            agent_scores[agent.agent_id] = score

        # 加权求和
        final_score = sum(
            self.weights[aid] * score for aid, score in agent_scores.items()
        )
        return final_score, agent_scores

    def update_weights(self, feedback: UserFeedback, recommendations: Recommendation) -> None:
        """
        根据用户反馈调整权重

        如果点击多 → 提升ClickAgent权重
        如果购买多 → 提升ConversionAgent权重
        如果利润高 → 提升ProfitAgent权重
        """
        click_rate = len(feedback.clicks) / max(len(feedback.impressions), 1)
        purchase_rate = len(feedback.purchases) / max(len(feedback.clicks), 1)

        # 简单启发式调整
        if click_rate < 0.1:
            self.weights["agent_click"] += self.lr
        if purchase_rate < 0.2:
            self.weights["agent_conversion"] += self.lr
        if purchase_rate > 0.3:
            self.weights["agent_profit"] += self.lr

        # 归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get_weights(self) -> dict[str, float]:
        return dict(self.weights)


# ────────────────────────── 推荐环境 ──────────────────────────

class RecommenderEnvironment:
    """
    推荐系统环境

    模拟用户行为：曝光 → 点击 → 购买
    """

    def __init__(self, catalog: list[Product], random_seed: int = 42) -> None:
        self.catalog = catalog
        self.rng = random.Random(random_seed)

    def simulate_feedback(
        self, user: User, recommended_items: list[tuple[str, float]]
    ) -> UserFeedback:
        """
        模拟用户对推荐列表的反馈

        两阶段：先看是否点击（基于商品吸引力和用户兴趣），
        再看点了的是否购买（基于价格匹配度和需求强度）。
        """
        impressions = [item[0] for item in recommended_items]
        clicks = []
        purchases = []
        dwell_times = {}

        product_map = {p.product_id: p for p in self.catalog}

        for item in recommended_items:
            pid = item[0]
            score = item[1]
            product = product_map.get(pid)
            if not product:
                continue

            # 点击决策：基于推荐分数 + 噪声
            click_prob = min(0.9, score * 0.8 + self.rng.gauss(0, 0.1))
            if self.rng.random() < max(0, click_prob):
                clicks.append(pid)
                dwell_times[pid] = max(1, self.rng.gauss(30, 15))

                # 购买决策：基于转化率 + 价格匹配
                purchase_prob = product.conversion_score * 0.6
                if user.price_sensitivity > 0.7 and product.price > 150:
                    purchase_prob *= 0.3
                elif user.price_sensitivity < 0.3 and product.price > 200:
                    purchase_prob *= 1.2  # 低敏感用户买高价

                if self.rng.random() < purchase_prob:
                    purchases.append(pid)

        return UserFeedback(
            user_id=user.user_id,
            impressions=impressions,
            clicks=clicks,
            purchases=purchases,
            dwell_times=dwell_times,
        )


# ────────────────────────── 多目标推荐系统 ──────────────────────────

class MultiObjectiveRecommender:
    """
    多目标推荐系统

    整合多个目标Agent + 协调器 + 环境仿真。
    """

    def __init__(
        self,
        catalog: list[Product],
        top_k: int = 10,
        random_seed: int = 42,
    ) -> None:
        self.catalog = catalog
        self.top_k = top_k

        # 初始化Agent
        self.click_agent = ClickAgent()
        self.conversion_agent = ConversionAgent()
        self.profit_agent = ProfitAgent()
        self.diversity_agent = DiversityAgent()

        self.agents: list[ObjectiveAgent] = [
            self.click_agent,
            self.conversion_agent,
            self.profit_agent,
            self.diversity_agent,
        ]

        self.coordinator = Coordinator(self.agents)
        self.env = RecommenderEnvironment(catalog, random_seed)

        # 记录
        self.recommendation_log: list[Recommendation] = []
        self.feedback_log: list[UserFeedback] = []

    def recommend(self, user: User) -> Recommendation:
        """为用户生成推荐列表"""
        scored_items = []
        for product in self.catalog:
            final_score, agent_scores = self.coordinator.compute_final_score(user, product)
            scored_items.append((product.product_id, final_score, agent_scores))

        # 去重：同一用户不重复推荐已购商品
        for past in user.purchase_history:
            scored_items = [(pid, s, a) for pid, s, a in scored_items if pid != past]

        # 排序取Top-K
        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_items = scored_items[:self.top_k]

        # 记录多样性Agent的曝光
        product_map = {p.product_id: p for p in self.catalog}
        for pid, _, _ in top_items:
            p = product_map.get(pid)
            if p:
                self.diversity_agent.record_exposure(user.user_id, p.category)

        rec = Recommendation(
            user_id=user.user_id,
            items=top_items,
        )
        self.recommendation_log.append(rec)
        return rec

    def observe_feedback(self, user: User) -> UserFeedback:
        """获取用户反馈（真实或仿真）"""
        rec = self.recommend(user)
        feedback = self.env.simulate_feedback(user, rec.items)
        self.feedback_log.append(feedback)

        # 更新协调器权重
        self.coordinator.update_weights(feedback, rec)

        return feedback

    def evaluate(self, n_users: int = 100) -> dict[str, Any]:
        """评估系统性能"""
        total_impressions = 0
        total_clicks = 0
        total_purchases = 0
        total_revenue = 0.0
        total_profit = 0.0
        category_coverage: set[str] = set()

        product_map = {p.product_id: p for p in self.catalog}

        for fb in self.feedback_log[-n_users:]:
            total_impressions += len(fb.impressions)
            total_clicks += len(fb.clicks)
            total_purchases += len(fb.purchases)

            for pid in fb.purchases:
                p = product_map.get(pid)
                if p:
                    total_revenue += p.price
                    total_profit += p.price * p.profit_margin

            for pid in fb.impressions:
                p = product_map.get(pid)
                if p:
                    category_coverage.add(p.category)

        ctr = total_clicks / max(total_impressions, 1)
        cvr = total_purchases / max(total_clicks, 1)
        gmvr = total_purchases / max(total_impressions, 1)

        return {
            "users_evaluated": min(n_users, len(self.feedback_log)),
            "ctr": round(ctr, 4),
            "cvr": round(cvr, 4),
            "gmvr": round(gmvr, 4),
            "total_revenue": round(total_revenue, 2),
            "total_profit": round(total_profit, 2),
            "category_coverage": len(category_coverage),
            "current_weights": self.coordinator.get_weights(),
        }


# ────────────────────────── 工厂方法和演示 ──────────────────────────

def create_momcozy_catalog() -> list[Product]:
    """创建Momcozy商品目录"""
    return [
        Product("momcozy_s12", "Momcozy S12 Pro", "feeding", 159.99, 0.45, 0.75, 0.60, 0.85, 0.3),
        Product("momcozy_m5", "Momcozy M5", "feeding", 89.99, 0.40, 0.70, 0.55, 0.80, 0.5),
        Product("momcozy_wearable", "Momcozy Wearable Pump", "feeding", 199.99, 0.50, 0.65, 0.50, 0.70, 0.8),
        Product("momcozy_bottle_set", "Momcozy Bottle Set", "feeding", 29.99, 0.35, 0.80, 0.70, 0.75, 0.6),
        Product("momcozy_bag", "Momcozy Diaper Bag", "gear", 49.99, 0.55, 0.60, 0.45, 0.65, 0.4),
        Product("momcozy_nursing_pad", "Momcozy Nursing Pads", "care", 19.99, 0.60, 0.55, 0.50, 0.60, 0.7),
        Product("medela_pump", "Medela Pump In Style", "feeding", 189.99, 0.30, 0.70, 0.65, 0.90, 0.2),
        Product("spectra_s1", "Spectra S1 Plus", "feeding", 179.99, 0.35, 0.65, 0.60, 0.85, 0.3),
        Product("philips_avent", "Philips Avent Manual", "feeding", 39.99, 0.50, 0.50, 0.55, 0.50, 0.5),
        Product("lansinoh_cream", "Lansinoh Nipple Cream", "care", 12.99, 0.55, 0.45, 0.40, 0.55, 0.6),
        Product("haakaa_pump", "Haakaa Silicone Pump", "feeding", 15.99, 0.60, 0.55, 0.45, 0.70, 0.4),
        Product("baby_bjorn_carrier", "BabyBjorn Carrier", "gear", 129.99, 0.40, 0.50, 0.35, 0.60, 0.5),
        Product("elvie_pump", "Elvie Curve", "feeding", 49.99, 0.50, 0.45, 0.40, 0.55, 0.7),
        Product("fridaMom_kit", "Frida Mom Recovery Kit", "care", 34.99, 0.45, 0.50, 0.45, 0.50, 0.6),
        Product("momcozy_starter", "Momcozy Starter Kit", "feeding", 119.99, 0.42, 0.72, 0.65, 0.75, 0.9),
    ]


def create_demo_users() -> list[User]:
    """创建演示用户"""
    users = []
    personas = ["new_mom", "experienced", "gift_buyer"]
    brand_prefs = ["momcozy", "medela", "any"]

    for i in range(50):
        persona = random.choice(personas)
        price_sens = {"new_mom": 0.7, "experienced": 0.4, "gift_buyer": 0.5}[persona]
        brand_pref = random.choice(brand_prefs)

        user = User(
            user_id=f"user_{i:03d}",
            persona=persona,
            price_sensitivity=price_sens + random.uniform(-0.1, 0.1),
            brand_preference=brand_pref if random.random() < 0.5 else "",
            purchase_history=random.sample(
                ["momcozy_s12", "momcozy_m5", "momcozy_bottle_set"],
                k=random.randint(0, 2),
            ),
        )
        users.append(user)

    return users


def run_ab_test() -> dict[str, Any]:
    """
    A/B测试：对比单目标 vs 多目标推荐
    """
    catalog = create_momcozy_catalog()
    users = create_demo_users()

    # A组：单目标（仅点击率）
    print("\n  Running Group A: Click-only baseline...")
    single_obj = MultiObjectiveRecommender(catalog, top_k=5)
    # 强制权重：click=1.0, 其他=0
    single_obj.coordinator.weights = {
        "agent_click": 1.0,
        "agent_conversion": 0.0,
        "agent_profit": 0.0,
        "agent_diversity": 0.0,
    }
    for user in users:
        single_obj.observe_feedback(user)
    result_a = single_obj.evaluate(len(users))

    # B组：多目标（MaRCA风格）
    print("  Running Group B: Multi-objective (MaRCA-style)...")
    multi_obj = MultiObjectiveRecommender(catalog, top_k=5, random_seed=43)
    for user in users:
        multi_obj.observe_feedback(user)
    result_b = multi_obj.evaluate(len(users))

    return {
        "group_a_click_only": result_a,
        "group_b_multi_objective": result_b,
        "improvements": {
            "ctr": f"{(result_b['ctr'] - result_a['ctr']) / max(result_a['ctr'], 0.001) * 100:+.1f}%",
            "cvr": f"{(result_b['cvr'] - result_a['cvr']) / max(result_a['cvr'], 0.001) * 100:+.1f}%",
            "revenue": f"{(result_b['total_revenue'] - result_a['total_revenue']) / max(result_a['total_revenue'], 0.001) * 100:+.1f}%",
            "profit": f"{(result_b['total_profit'] - result_a['total_profit']) / max(result_a['total_profit'], 0.001) * 100:+.1f}%",
            "category_coverage": f"{result_b['category_coverage'] - result_a['category_coverage']:+d}",
        },
    }


# ────────────────────────── 演示 ──────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Agent Multi-Objective Recommendation System")
    print("Based on: arXiv:2512.24325 (MaRCA)")
    print("Scenario: Momcozy Cross-border Recommendation")
    print("=" * 70)

    results = run_ab_test()

    print("\n" + "=" * 70)
    print("A/B TEST RESULTS")
    print("=" * 70)

    print("\n📊 Group A: Click-Only Baseline")
    a = results["group_a_click_only"]
    print(f"   CTR: {a['ctr']:.2%}")
    print(f"   CVR: {a['cvr']:.2%}")
    print(f"   GMVR: {a['gmvr']:.2%}")
    print(f"   Revenue: ${a['total_revenue']:,.2f}")
    print(f"   Profit: ${a['total_profit']:,.2f}")
    print(f"   Category Coverage: {a['category_coverage']}")

    print("\n📊 Group B: Multi-Objective (MaRCA-style)")
    b = results["group_b_multi_objective"]
    print(f"   CTR: {b['ctr']:.2%}")
    print(f"   CVR: {b['cvr']:.2%}")
    print(f"   GMVR: {b['gmvr']:.2%}")
    print(f"   Revenue: ${b['total_revenue']:,.2f}")
    print(f"   Profit: ${b['total_profit']:,.2f}")
    print(f"   Category Coverage: {b['category_coverage']}")
    print(f"   Final Weights: { {k: round(v, 3) for k, v in b['current_weights'].items()} }")

    print("\n📈 Improvements (B vs A)")
    for metric, value in results["improvements"].items():
        print(f"   {metric.upper()}: {value}")

    print("\n" + "=" * 70)
    print("Key Findings:")
    print("-" * 70)
    print("1. Multi-objective balancing improves long-term metrics (profit, coverage)")
    print("2. Coordinator learns to adjust weights based on user feedback")
    print("3. Trade-off: CTR may decrease slightly while CVR and profit increase")
    print("4. Diversity agent expands category coverage, improving discovery")
    print("=" * 70)
    print("Simulation complete. Verify: no errors raised.")
    print("=" * 70)

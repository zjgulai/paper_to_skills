"""
Generative Agent Simulation for Marketing Consumer Behavior
基于生成式智能体的营销消费者行为沙盒仿真模型

论文: LLM-Based Multi-Agent System for Simulating and Analyzing
      Marketing and Consumer Behavior
arXiv: 2510.18155 (ICEBE 2025)

核心思想:
  在虚拟商业沙盒中，赋予每个 Agent 资源约束（钱/时间/精力）和社会记忆
  （历史消费体验、口碑影响），模拟营销事件触发后的消费者涌现行为。

工程设计:
  - 完全 mock，不依赖真实 LLM API，可离线运行
  - 通过基于规则 + 随机扰动模拟 LLM 的"语义判断"
  - 支持口碑传播（Word-of-Mouth）社交网络仿真
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class Resources:
    """Agent 资源约束（每日初始化）"""
    budget: float          # 可支配预算（元）
    time_slots: int        # 可用时间格（1格=30分钟）
    energy: float          # 精力值（0-1）

    def can_afford(self, price: float) -> bool:
        return self.budget >= price

    def has_time(self, slots_needed: int = 1) -> bool:
        return self.time_slots >= slots_needed

    def is_willing(self, threshold: float = 0.3) -> bool:
        """精力高于阈值才主动消费"""
        return self.energy >= threshold


@dataclass
class Memory:
    """Agent 的社会记忆与品牌偏好"""
    brand_experiences: Dict[str, float] = field(default_factory=dict)  # brand -> 满意度(-1~1)
    wom_heard: List[str] = field(default_factory=list)                  # 听到的口碑信息
    visit_history: List[str] = field(default_factory=list)              # 访问过的商家

    def update_experience(self, brand: str, satisfaction: float):
        """更新品牌体验，滑动平均"""
        prev = self.brand_experiences.get(brand, 0.0)
        self.brand_experiences[brand] = 0.7 * prev + 0.3 * satisfaction

    def get_brand_affinity(self, brand: str) -> float:
        """返回品牌偏好（默认中性 0）"""
        return self.brand_experiences.get(brand, 0.0)

    def hear_wom(self, message: str):
        self.wom_heard.append(message)
        # 只保留最近 10 条
        if len(self.wom_heard) > 10:
            self.wom_heard = self.wom_heard[-10:]


@dataclass
class Agent:
    """虚拟消费者智能体"""
    agent_id: str
    persona: str          # 'budget_conscious' | 'time_sensitive' | 'brand_loyal' | 'social_follower'
    resources: Resources
    memory: Memory = field(default_factory=Memory)
    friends: List[str] = field(default_factory=list)  # 社交关系（agent_id 列表）
    action_log: List[Dict] = field(default_factory=list)

    def log_action(self, day: int, action: str, brand: str, detail: str = ""):
        self.action_log.append({
            "day": day,
            "agent_id": self.agent_id,
            "action": action,
            "brand": brand,
            "detail": detail,
        })


@dataclass
class MarketingEvent:
    """营销事件（投放到沙盒的干预）"""
    brand: str
    event_type: str          # 'discount' | 'new_product' | 'membership'
    description: str
    discount_rate: float = 0.0  # 折扣率（0.2 = 八折）
    reach_rate: float = 0.8     # 初始触达率


@dataclass
class Venue:
    """商业场所（咖啡馆、快餐店等）"""
    name: str
    brand: str
    category: str   # 'cafe' | 'fast_food' | 'family_restaurant'
    base_price: float
    quality: float  # 0-1，影响满意度


# ─────────────────────────────────────────────
# 智能体决策引擎（mock LLM 语义判断）
# ─────────────────────────────────────────────

class AgentDecisionEngine:
    """
    模拟 LLM 的语义推理决策，完全基于规则 + 随机扰动
    真实系统会调用 GPT/Claude API 生成自然语言推理后解析行动
    """

    def decide_visit(
        self,
        agent: Agent,
        venue: Venue,
        day: int,
        event: Optional[MarketingEvent] = None,
    ) -> Tuple[bool, str]:
        """
        决策是否访问某商家
        Returns: (will_visit, reason)
        """
        score = 0.0

        # 1. 资源约束检查
        effective_price = venue.base_price
        if event and event.brand == venue.brand and event.event_type == 'discount':
            effective_price *= (1 - event.discount_rate)

        if not agent.resources.can_afford(effective_price):
            return False, "预算不足"
        if not agent.resources.has_time():
            return False, "时间不足"
        if not agent.resources.is_willing():
            return False, "精力不足"

        # 2. 品牌偏好
        affinity = agent.memory.get_brand_affinity(venue.brand)
        score += affinity * 0.3

        # 3. Persona 驱动
        if agent.persona == 'budget_conscious':
            # 有折扣时大幅加分
            if event and event.brand == venue.brand and event.event_type == 'discount':
                score += event.discount_rate * 0.6
            else:
                score -= 0.2
        elif agent.persona == 'time_sensitive':
            # 快餐偏好
            score += 0.2 if venue.category == 'fast_food' else -0.1
        elif agent.persona == 'brand_loyal':
            # 重访历史去过的品牌
            if venue.brand in agent.memory.visit_history:
                score += 0.4
        elif agent.persona == 'social_follower':
            # 口碑影响
            positive_wom = sum(1 for m in agent.memory.wom_heard if venue.brand in m and "好" in m)
            negative_wom = sum(1 for m in agent.memory.wom_heard if venue.brand in m and "差" in m)
            score += (positive_wom - negative_wom) * 0.2

        # 4. 随机扰动（模拟人类不确定性）
        score += random.gauss(0, 0.15)

        # 5. 基准决策阈值
        threshold = 0.1
        will_visit = score >= threshold

        reason = f"score={score:.2f}, persona={agent.persona}"
        return will_visit, reason

    def generate_wom(self, agent: Agent, venue: Venue, satisfaction: float) -> Optional[str]:
        """
        生成口碑传播消息（满意才分享，不满意可能吐槽）
        """
        if satisfaction > 0.6 and random.random() < 0.4:
            return f"{venue.brand}的{venue.category}真的好，性价比很高！"
        elif satisfaction < -0.3 and random.random() < 0.3:
            return f"{venue.brand}今天服务差，踩雷了。"
        return None


# ─────────────────────────────────────────────
# 沙盒仿真引擎
# ─────────────────────────────────────────────

class MarketingSandbox:
    """
    营销沙盒仿真引擎
    模拟 N 个 Agent 在 D 天内对营销事件的响应
    """

    def __init__(self, agents: List[Agent], venues: List[Venue], seed: int = 42):
        self.agents = {a.agent_id: a for a in agents}
        self.venues = venues
        self.engine = AgentDecisionEngine()
        random.seed(seed)

        # 统计累积
        self.daily_visits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.daily_revenue: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.wom_propagation: List[Dict] = []

    def _refresh_resources(self, agent: Agent):
        """每天重置资源（模拟每日可用预算/时间）"""
        agent.resources = Resources(
            budget=random.gauss(
                self._persona_budget(agent.persona), 20
            ),
            time_slots=random.randint(1, 4),
            energy=random.uniform(0.4, 1.0),
        )

    @staticmethod
    def _persona_budget(persona: str) -> float:
        return {
            'budget_conscious': 50,
            'time_sensitive': 80,
            'brand_loyal': 100,
            'social_follower': 70,
        }.get(persona, 70)

    def _propagate_wom(self, sender: Agent, message: str, day: int):
        """口碑在社交网络中一跳传播"""
        for fid in sender.friends:
            receiver = self.agents.get(fid)
            if receiver:
                receiver.memory.hear_wom(message)
                self.wom_propagation.append({
                    "day": day,
                    "from": sender.agent_id,
                    "to": fid,
                    "message": message,
                })

    def run(self, n_days: int, event: Optional[MarketingEvent] = None) -> Dict:
        """
        运行仿真

        Args:
            n_days: 仿真天数
            event: 注入的营销事件（None 表示无干预对照组）

        Returns:
            仿真结果字典
        """
        # 事件触达：随机决定哪些 Agent 在第 1 天看到营销事件
        event_aware: set = set()
        if event:
            for aid, agent in self.agents.items():
                if random.random() < event.reach_rate:
                    agent.memory.hear_wom(f"[广告]{event.brand}：{event.description}")
                    event_aware.add(aid)

        for day in range(1, n_days + 1):
            for agent in self.agents.values():
                self._refresh_resources(agent)

                for venue in self.venues:
                    will_visit, reason = self.engine.decide_visit(
                        agent, venue, day, event
                    )

                    if will_visit:
                        # 计算实际价格
                        price = venue.base_price
                        if event and event.brand == venue.brand and event.event_type == 'discount':
                            price *= (1 - event.discount_rate)

                        # 记录访问
                        agent.resources.budget -= price
                        agent.memory.visit_history.append(venue.brand)

                        # 满意度模拟
                        satisfaction = venue.quality + random.gauss(0, 0.2)
                        satisfaction = max(-1.0, min(1.0, satisfaction))
                        agent.memory.update_experience(venue.brand, satisfaction)

                        agent.log_action(day, "visit", venue.brand,
                                         f"price={price:.1f} sat={satisfaction:.2f}")

                        self.daily_visits[day][venue.brand] += 1
                        self.daily_revenue[day][venue.brand] += price

                        # 口碑传播
                        wom_msg = self.engine.generate_wom(agent, venue, satisfaction)
                        if wom_msg:
                            self._propagate_wom(agent, wom_msg, day)

        return self._aggregate_results(n_days, event)

    def _aggregate_results(self, n_days: int, event: Optional[MarketingEvent]) -> Dict:
        """聚合仿真结果"""
        brand_totals: Dict[str, Dict] = defaultdict(lambda: {"visits": 0, "revenue": 0.0})

        for day_data in self.daily_visits.values():
            for brand, cnt in day_data.items():
                brand_totals[brand]["visits"] += cnt

        for day_data in self.daily_revenue.values():
            for brand, rev in day_data.items():
                brand_totals[brand]["revenue"] += rev

        # 按天聚合（用于趋势分析）
        event_brand = event.brand if event else None
        daily_event_visits = []
        for d in range(1, n_days + 1):
            if event_brand:
                daily_event_visits.append(self.daily_visits[d].get(event_brand, 0))

        # WOM 统计
        wom_by_brand: Dict[str, int] = defaultdict(int)
        for w in self.wom_propagation:
            for venue in self.venues:
                if venue.brand in w["message"]:
                    wom_by_brand[venue.brand] += 1

        return {
            "n_days": n_days,
            "n_agents": len(self.agents),
            "brand_totals": dict(brand_totals),
            "daily_event_visits": daily_event_visits,
            "total_wom_messages": len(self.wom_propagation),
            "wom_by_brand": dict(wom_by_brand),
            "event": {
                "brand": event.brand if event else None,
                "type": event.event_type if event else None,
                "discount": event.discount_rate if event else 0,
            }
        }


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def create_agents(n: int = 50, seed: int = 42) -> List[Agent]:
    """创建 N 个具有不同 Persona 的虚拟消费者 Agent"""
    random.seed(seed)
    personas = ['budget_conscious', 'time_sensitive', 'brand_loyal', 'social_follower']
    agents = []

    for i in range(n):
        persona = personas[i % len(personas)]
        agent = Agent(
            agent_id=f"A{i:04d}",
            persona=persona,
            resources=Resources(
                budget=random.gauss(70, 20),
                time_slots=random.randint(1, 4),
                energy=random.uniform(0.4, 1.0),
            ),
        )
        agents.append(agent)

    # 构建随机社交网络（每人平均 3 个朋友）
    ids = [a.agent_id for a in agents]
    for agent in agents:
        n_friends = random.randint(1, 5)
        candidates = [aid for aid in ids if aid != agent.agent_id]
        agent.friends = random.sample(candidates, min(n_friends, len(candidates)))

    return agents


def create_venues() -> List[Venue]:
    """创建虚拟商业场所"""
    return [
        Venue("星享咖啡", "星享", "cafe",           base_price=35, quality=0.75),
        Venue("麦脆快餐", "麦脆", "fast_food",       base_price=42, quality=0.65),
        Venue("家味餐厅", "家味", "family_restaurant", base_price=68, quality=0.80),
        Venue("速拿咖啡", "速拿", "cafe",             base_price=28, quality=0.60),
    ]


# ─────────────────────────────────────────────
# 对比实验：有营销事件 vs. 无营销事件
# ─────────────────────────────────────────────

def run_ab_simulation(n_agents: int = 100, n_days: int = 7, seed: int = 42) -> Dict:
    """
    对比仿真：
      - Control: 无任何营销干预
      - Treatment: 麦脆快餐 周中八折促销（discount_rate=0.2）
    """
    venues = create_venues()

    # --- Control ---
    control_agents = create_agents(n_agents, seed=seed)
    control_sandbox = MarketingSandbox(control_agents, venues, seed=seed)
    control_result = control_sandbox.run(n_days, event=None)

    # --- Treatment ---
    treatment_agents = create_agents(n_agents, seed=seed + 1)
    treatment_event = MarketingEvent(
        brand="麦脆",
        event_type="discount",
        description="周中特惠！全单八折，仅限周二至周四",
        discount_rate=0.20,
        reach_rate=0.75,
    )
    treatment_sandbox = MarketingSandbox(treatment_agents, venues, seed=seed + 1)
    treatment_result = treatment_sandbox.run(n_days, event=treatment_event)

    return {
        "control": control_result,
        "treatment": treatment_result,
    }


# ─────────────────────────────────────────────
# 结果分析器
# ─────────────────────────────────────────────

class SimulationAnalyzer:
    """分析对比仿真结果"""

    @staticmethod
    def lift(control: int, treatment: int) -> float:
        """计算提升率"""
        if control == 0:
            return float('inf') if treatment > 0 else 0.0
        return (treatment - control) / control

    def print_report(self, results: Dict):
        ctrl = results["control"]
        trt = results["treatment"]

        print("=" * 65)
        print("  营销沙盒仿真报告 (Generative Agent Simulation)")
        print("=" * 65)
        print(f"  仿真规模: {ctrl['n_agents']} agents × {ctrl['n_days']} days")
        print()

        # 品牌维度对比
        print("┌─────────────────┬──────────┬──────────┬──────────┐")
        print("│ 品牌             │ 控制组   │ 实验组   │ 提升率   │")
        print("├─────────────────┼──────────┼──────────┼──────────┤")

        all_brands = set(
            list(ctrl["brand_totals"].keys()) +
            list(trt["brand_totals"].keys())
        )

        for brand in sorted(all_brands):
            c_visits = ctrl["brand_totals"].get(brand, {}).get("visits", 0)
            t_visits = trt["brand_totals"].get(brand, {}).get("visits", 0)
            lift_pct = self.lift(c_visits, t_visits) * 100
            mark = " ←促销" if brand == trt["event"]["brand"] else ""
            print(f"│ {brand:<13}{mark:<3}│ {c_visits:>8} │ {t_visits:>8} │ {lift_pct:>+7.1f}% │")

        print("└─────────────────┴──────────┴──────────┴──────────┘")

        # 促销品牌特写
        promo_brand = trt["event"]["brand"]
        promo_ctrl = ctrl["brand_totals"].get(promo_brand, {}).get("visits", 0)
        promo_trt = trt["brand_totals"].get(promo_brand, {}).get("visits", 0)
        promo_lift = self.lift(promo_ctrl, promo_trt)

        print()
        print(f"  [促销品牌：{promo_brand}]")
        print(f"    折扣力度  : {trt['event']['discount']:.0%} OFF")
        print(f"    访客提升  : {promo_ctrl} → {promo_trt} ({promo_lift:+.1%})")

        c_rev = ctrl["brand_totals"].get(promo_brand, {}).get("revenue", 0)
        t_rev = trt["brand_totals"].get(promo_brand, {}).get("revenue", 0)
        rev_lift = self.lift(c_rev, t_rev)
        print(f"    营收变化  : ¥{c_rev:.0f} → ¥{t_rev:.0f} ({rev_lift:+.1%})")

        # 口碑传播
        print()
        print(f"  [口碑传播]")
        print(f"    控制组 WOM 消息: {ctrl['total_wom_messages']}")
        print(f"    实验组 WOM 消息: {trt['total_wom_messages']}")
        wom_lift = self.lift(ctrl['total_wom_messages'], trt['total_wom_messages'])
        print(f"    WOM 提升       : {wom_lift:+.1%}")

        # 日趋势
        if trt.get("daily_event_visits"):
            print()
            print(f"  [{promo_brand} 每日访客趋势（实验组）]")
            print("    " + " ".join(f"D{i+1}:{v:3d}" for i, v in
                                    enumerate(trt["daily_event_visits"])))

        print()
        print("=" * 65)


# ─────────────────────────────────────────────
# 自测函数
# ─────────────────────────────────────────────

def self_test():
    """完整自测：构建沙盒 → 注入营销事件 → 验证结果合理性"""
    print("运行 Generative Agent Simulation 自测...\n")

    results = run_ab_simulation(n_agents=80, n_days=7, seed=2025)

    analyzer = SimulationAnalyzer()
    analyzer.print_report(results)

    # ── 断言验证 ──
    ctrl = results["control"]
    trt = results["treatment"]
    promo_brand = trt["event"]["brand"]

    ctrl_visits = ctrl["brand_totals"].get(promo_brand, {}).get("visits", 0)
    trt_visits  = trt["brand_totals"].get(promo_brand, {}).get("visits", 0)

    assert ctrl_visits >= 0, "控制组访客数应≥0"
    assert trt_visits  >= 0, "实验组访客数应≥0"

    # 实验组 WOM 应多于控制组（有营销事件）
    assert trt["total_wom_messages"] >= ctrl["total_wom_messages"], \
        f"实验组 WOM({trt['total_wom_messages']}) 应 >= 控制组 WOM({ctrl['total_wom_messages']})"

    # 促销后品牌访客数应有提升（允许轻微随机波动）
    lift = SimulationAnalyzer.lift(ctrl_visits, trt_visits)
    print(f"\n[ASSERT] 促销品牌访客提升率: {lift:+.1%}")
    assert lift > -0.5, f"提升率 {lift:.1%} 异常偏低，请检查参数"

    # 确保所有品牌都有数据
    venues = create_venues()
    all_brands = {v.brand for v in venues}
    for brand in all_brands:
        assert brand in trt["brand_totals"], f"实验组缺少 {brand} 数据"

    print("\n[PASS] 所有断言通过 ✓")
    return results


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    self_test()

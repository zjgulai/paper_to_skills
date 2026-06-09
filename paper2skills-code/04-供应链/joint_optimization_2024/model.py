"""
Fast-Slow Dual-Agent Deep Reinforcement Learning (FSDA-DRL)
===========================================================
论文: Dual-Agent Deep Reinforcement Learning for Dynamic Pricing and Replenishment
      (arXiv: 2410.21109, 2024-10)

核心思想:
  - 定价智能体（快尺度，每天更新）：响应竞品和库存，输出动态折扣。
  - 补货智能体（慢尺度，每周更新）：基于预期价格路径和库存水位，制定进货量。
  - 两个智能体共享环境状态，通过双时间尺度（Two-timescale）协作而非博弈。

业务场景: 跨境电商大促期间（如 Prime Day）的备货 + 动态定价联合决策
  - 减少"供应链备货过多/不足"与"运营随意调价"脱节的问题
  - 目标: 最大化大促周期全局利润（GMV × 毛利率）

自测说明:
  运行 `python model.py` 会依次执行:
  1. 单元测试: 验证各类、奖励函数、智能体动作
  2. 集成测试: 完整仿真 30 天（快尺度 30 步 / 慢尺度 4 步）并打印结果摘要
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1. 环境状态定义
# ---------------------------------------------------------------------------

@dataclass
class EnvState:
    """
    共享环境状态（定价 Agent 和补货 Agent 都能读到）

    Attributes:
        inventory:      当前库存件数
        days_remaining: 大促剩余天数
        competitor_price: 竞品当日售价（元）
        base_price:     商品建议零售价（元）
        cost_price:     采购成本（元）
        pending_order:  在途补货件数（已下单但未到货）
        day:            当前天（1 开始）
    """
    inventory: float
    days_remaining: int
    competitor_price: float
    base_price: float
    cost_price: float
    pending_order: float = 0.0
    day: int = 1

    def to_vector(self) -> List[float]:
        """转为网络输入向量（归一化）"""
        return [
            self.inventory / 10000.0,
            self.days_remaining / 30.0,
            self.competitor_price / self.base_price,
            self.pending_order / 5000.0,
            math.sin(2 * math.pi * self.day / 7),   # 星期特征
            math.cos(2 * math.pi * self.day / 7),
        ]


# ---------------------------------------------------------------------------
# 2. 需求模型
# ---------------------------------------------------------------------------

class DemandModel:
    """
    简化的需求模型，模拟真实市场中价格弹性 + 竞争效应 + 随机波动。

    需求公式（对数-线性弹性模型）:
        D = D_base × (price / base_price)^(-elasticity) × competitor_effect + noise

    Parameters:
        base_demand:  基础日需求量（件/天）
        elasticity:   价格弹性系数（通常 1.0 ~ 2.5）
        noise_std:    需求随机标准差占比
    """

    def __init__(
        self,
        base_demand: float = 500.0,
        elasticity: float = 1.8,
        noise_std: float = 0.15,
        random_seed: int = 42,
    ):
        self.base_demand = base_demand
        self.elasticity = elasticity
        self.noise_std = noise_std
        self._rng = random.Random(random_seed)

    def sample(self, price: float, base_price: float, competitor_price: float) -> float:
        """
        采样当日需求量。

        Args:
            price:            当日实际售价
            base_price:       建议零售价（基准）
            competitor_price: 竞品实时售价

        Returns:
            demand: 预测需求量（件，非负整数）
        """
        # 价格弹性效应
        price_ratio = price / base_price
        price_effect = price_ratio ** (-self.elasticity)

        # 竞争效应：我方价格比竞品低则需求提升
        comp_ratio = price / competitor_price
        competitor_effect = 1.0 + 0.3 * (1.0 - comp_ratio)  # 每便宜 10% 需求+3%

        # 随机扰动（正态噪声）
        noise = 1.0 + self._rng.gauss(0, self.noise_std)
        noise = max(0.1, noise)  # 截断负值

        demand = self.base_demand * price_effect * competitor_effect * noise
        return max(0.0, demand)


# ---------------------------------------------------------------------------
# 3. 快尺度：定价智能体
# ---------------------------------------------------------------------------

class PricingAgent:
    """
    快尺度定价智能体 - 每天执行一次，决定当日折扣率。

    策略（基于规则的 RL 近似，模拟训练后的策略网络行为）:
    - 库存充足 + 剩余天数多 → 保持正常定价
    - 库存紧张（< 安全水位） → 提价（减小折扣）
    - 库存积压 + 临近结束 → 打折清仓

    在真实部署中，此处的 compute_discount() 会由训练好的 DQN/PPO 网络替换。
    """

    # 折扣率可选动作集（离散动作空间，模拟 DQN）
    DISCOUNT_OPTIONS: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

    def __init__(self, safety_stock_ratio: float = 0.20):
        """
        Args:
            safety_stock_ratio: 安全库存水位（占初始库存的比例）
        """
        self.safety_stock_ratio = safety_stock_ratio
        self.initial_inventory: float = 0.0
        self._action_history: List[float] = []

    def reset(self, initial_inventory: float) -> None:
        """每次大促开始时重置"""
        self.initial_inventory = initial_inventory
        self._action_history.clear()

    def compute_discount(self, state: EnvState) -> float:
        """
        根据当前状态选择折扣率（模拟训练后的策略）。

        Returns:
            discount: 折扣率 (0.0~1.0)，1.0 表示原价
        """
        safety_stock = self.initial_inventory * self.safety_stock_ratio

        # 库存吃紧 → 减小折扣（提价保利润）
        if state.inventory < safety_stock:
            discount = 0.95
        # 临近活动结束且库存大量剩余 → 加大折扣
        elif state.days_remaining <= 3 and state.inventory > safety_stock * 2:
            discount = 0.70
        # 库存充足 + 中期 → 标准促销折扣
        elif state.inventory > self.initial_inventory * 0.5:
            discount = 0.85
        # 默认：轻度折扣维持销量
        else:
            discount = 0.80

        # 竞品价格修正：若竞品比我们便宜，稍微跟进
        if state.competitor_price < state.base_price * discount * 0.98:
            discount = max(0.70, discount - 0.05)

        self._action_history.append(discount)
        return discount

    @property
    def avg_discount(self) -> float:
        """历史平均折扣"""
        if not self._action_history:
            return 1.0
        return sum(self._action_history) / len(self._action_history)


# ---------------------------------------------------------------------------
# 4. 慢尺度：补货智能体
# ---------------------------------------------------------------------------

class ReplenishmentAgent:
    """
    慢尺度补货智能体 - 每周执行一次，决定向仓库进货的数量。

    策略（基于规则的 RL 近似）:
    - 根据剩余活动周期 × 预估日均销量 计算目标库存
    - 下单量 = max(0, 目标库存 - 当前库存 - 在途补货)
    - 受仓储容量约束（最大仓容 = max_capacity）

    在真实部署中，此处会由训练好的 DDPG/SAC 网络替换（连续动作空间）。
    """

    def __init__(
        self,
        max_capacity: float = 15000.0,
        lead_time_days: int = 3,
        holding_cost_per_unit: float = 0.5,
    ):
        """
        Args:
            max_capacity:           最大仓储容量（件）
            lead_time_days:         补货到货周期（天）
            holding_cost_per_unit:  每件每天库存持有成本（元）
        """
        self.max_capacity = max_capacity
        self.lead_time_days = lead_time_days
        self.holding_cost_per_unit = holding_cost_per_unit
        self._order_history: List[Tuple[int, float]] = []  # (day, order_qty)

    def compute_order_qty(
        self,
        state: EnvState,
        expected_daily_demand: float,
        pricing_agent_avg_discount: float,
    ) -> float:
        """
        计算本周补货量。

        Args:
            state:                      当前环境状态
            expected_daily_demand:      预期日均需求（基于历史 + 预测）
            pricing_agent_avg_discount: 定价 Agent 的近期平均折扣（影响需求预估）

        Returns:
            order_qty: 建议补货量（件，非负）
        """
        # 折扣影响需求：折扣越低，需求倍增
        demand_multiplier = (1.0 / max(pricing_agent_avg_discount, 0.5)) ** 0.5
        adjusted_demand = expected_daily_demand * demand_multiplier

        # 目标库存 = 剩余天数 × 预期日均需求 × 安全系数
        safety_factor = 1.15  # 15% 安全缓冲
        target_inventory = adjusted_demand * state.days_remaining * safety_factor

        # 目标库存不超过仓储容量
        target_inventory = min(target_inventory, self.max_capacity)

        # 补货量 = 缺口（扣除在途订单）
        available = state.inventory + state.pending_order
        order_qty = max(0.0, target_inventory - available)

        self._order_history.append((state.day, order_qty))
        return order_qty

    @property
    def total_ordered(self) -> float:
        """累计补货总量"""
        return sum(qty for _, qty in self._order_history)


# ---------------------------------------------------------------------------
# 5. 奖励函数
# ---------------------------------------------------------------------------

def compute_reward(
    price: float,
    cost_price: float,
    actual_sales: float,
    holding_inventory: float,
    holding_cost_per_unit: float,
    stockout_units: float,
    stockout_penalty_per_unit: float = 5.0,
) -> float:
    """
    计算单日奖励（利润导向）。

    奖励 = 销售毛利 - 库存持有成本 - 缺货惩罚

    Args:
        price:                    当日售价（元/件）
        cost_price:               成本价（元/件）
        actual_sales:             实际售出件数
        holding_inventory:        当日库存（件）
        holding_cost_per_unit:    每件每天持有成本（元）
        stockout_units:           缺货件数（需求 > 库存时的差额）
        stockout_penalty_per_unit: 每件缺货机会损失惩罚（元）

    Returns:
        reward: 当日奖励（元）
    """
    gross_profit = (price - cost_price) * actual_sales
    holding_cost = holding_inventory * holding_cost_per_unit
    stockout_cost = stockout_units * stockout_penalty_per_unit

    return gross_profit - holding_cost - stockout_cost


# ---------------------------------------------------------------------------
# 6. 仿真环境
# ---------------------------------------------------------------------------

class PromoSimulator:
    """
    大促仿真环境（Episode = 30 天大促）

    时间尺度:
    - 快尺度（定价 Agent）: 每天 step_fast()
    - 慢尺度（补货 Agent）: 每 REPLENISHMENT_INTERVAL 天 step_slow()

    状态更新流程（每天）:
    1. 定价 Agent 决定折扣 → 计算当日售价
    2. 需求模型采样当日需求
    3. 计算实际销售量（min(demand, inventory)）
    4. 计算奖励（毛利 - 持有成本 - 缺货损失）
    5. 更新库存
    6. 若当天是补货日（每 7 天），补货 Agent 决定补货量
    7. 在途库存到货（Lead time 后）
    """

    REPLENISHMENT_INTERVAL: int = 7   # 慢尺度：每 7 天补货一次
    EPISODE_DAYS: int = 30            # 大促周期

    def __init__(
        self,
        initial_inventory: float = 8000.0,
        base_price: float = 299.0,
        cost_price: float = 120.0,
        competitor_price: float = 289.0,
        lead_time_days: int = 3,
        random_seed: int = 42,
    ):
        self.initial_inventory = initial_inventory
        self.base_price = base_price
        self.cost_price = cost_price
        self.lead_time_days = lead_time_days

        self.demand_model = DemandModel(
            base_demand=500.0,
            elasticity=1.8,
            noise_std=0.15,
            random_seed=random_seed,
        )
        self.pricing_agent = PricingAgent(safety_stock_ratio=0.20)
        self.replenishment_agent = ReplenishmentAgent(
            lead_time_days=lead_time_days,
            holding_cost_per_unit=0.5,
        )

        # 竞品价格轨迹（模拟竞对不定期降价）
        self._rng = random.Random(random_seed + 1)
        self._competitor_prices = self._gen_competitor_prices(
            base=competitor_price, days=self.EPISODE_DAYS
        )

        # 仿真状态
        self.state: EnvState = self._init_state()
        self._pending_arrivals: List[Tuple[int, float]] = []  # (arrive_day, qty)

        # 累计统计
        self.total_reward: float = 0.0
        self.total_sales: float = 0.0
        self.total_stockout: float = 0.0
        self.reward_history: List[float] = []
        self.inventory_history: List[float] = []
        self.price_history: List[float] = []

    # ------------------------------------------------------------------ 初始化

    def _init_state(self) -> EnvState:
        return EnvState(
            inventory=self.initial_inventory,
            days_remaining=self.EPISODE_DAYS,
            competitor_price=self._competitor_prices[0],
            base_price=self.base_price,
            cost_price=self.cost_price,
        )

    def _gen_competitor_prices(self, base: float, days: int) -> List[float]:
        """生成竞品价格时间序列（带随机波动）"""
        prices = [base]
        for _ in range(days - 1):
            # 竞品可能不定期降价（10% 概率）
            if self._rng.random() < 0.10:
                change = self._rng.uniform(-0.15, -0.05)
            else:
                change = self._rng.uniform(-0.02, 0.02)
            new_price = max(base * 0.70, prices[-1] * (1 + change))
            prices.append(new_price)
        return prices

    def reset(self) -> EnvState:
        """重置环境到大促第 1 天"""
        self.state = self._init_state()
        self._pending_arrivals.clear()
        self.total_reward = 0.0
        self.total_sales = 0.0
        self.total_stockout = 0.0
        self.reward_history.clear()
        self.inventory_history.clear()
        self.price_history.clear()
        self.pricing_agent.reset(self.initial_inventory)
        return self.state

    # ------------------------------------------------------------------ 单日步骤

    def step(self) -> Tuple[EnvState, float, bool]:
        """
        执行一天的仿真步骤。

        Returns:
            (new_state, reward, done)
            - new_state: 更新后的环境状态
            - reward:    当日奖励
            - done:      大促是否结束
        """
        day = self.state.day
        state = self.state

        # --- 在途补货到货 ---
        arrived = sum(qty for arrive_day, qty in self._pending_arrivals if arrive_day == day)
        if arrived > 0:
            state.inventory = min(
                state.inventory + arrived,
                self.replenishment_agent.max_capacity,
            )
            self._pending_arrivals = [
                (d, q) for d, q in self._pending_arrivals if d != day
            ]
        state.pending_order = sum(q for _, q in self._pending_arrivals)

        # --- 快尺度：定价 Agent 决策 ---
        discount = self.pricing_agent.compute_discount(state)
        price = state.base_price * discount

        # --- 需求采样 ---
        demand = self.demand_model.sample(
            price=price,
            base_price=state.base_price,
            competitor_price=state.competitor_price,
        )

        # --- 销售与缺货 ---
        actual_sales = min(demand, state.inventory)
        stockout = max(0.0, demand - state.inventory)
        state.inventory -= actual_sales

        # --- 奖励计算 ---
        reward = compute_reward(
            price=price,
            cost_price=state.cost_price,
            actual_sales=actual_sales,
            holding_inventory=state.inventory,
            holding_cost_per_unit=self.replenishment_agent.holding_cost_per_unit,
            stockout_units=stockout,
        )

        # --- 慢尺度：补货 Agent 决策（每 7 天一次）---
        if day % self.REPLENISHMENT_INTERVAL == 1 and day > 1:
            # 预期日均需求（用历史均值作为预测）
            recent_sales = self.total_sales / max(1, day - 1)
            order_qty = self.replenishment_agent.compute_order_qty(
                state=state,
                expected_daily_demand=recent_sales,
                pricing_agent_avg_discount=self.pricing_agent.avg_discount,
            )
            if order_qty > 0:
                arrive_day = day + self.lead_time_days
                self._pending_arrivals.append((arrive_day, order_qty))

        # --- 状态更新 ---
        self.total_reward += reward
        self.total_sales += actual_sales
        self.total_stockout += stockout
        self.reward_history.append(reward)
        self.inventory_history.append(state.inventory)
        self.price_history.append(price)

        # 推进时间
        next_day = day + 1
        done = next_day > self.EPISODE_DAYS
        state.day = next_day
        state.days_remaining = self.EPISODE_DAYS - day
        if not done:
            state.competitor_price = self._competitor_prices[day]

        return state, reward, done

    def run_episode(self) -> dict:
        """
        运行完整 30 天大促仿真，返回结果摘要。

        Returns:
            summary dict 包含: total_reward, total_sales, total_stockout,
                               avg_price, service_level, replenishment_orders
        """
        self.reset()
        done = False
        while not done:
            _, _, done = self.step()

        # 计算服务率（填充率）
        total_demand = self.total_sales + self.total_stockout
        service_level = self.total_sales / total_demand if total_demand > 0 else 1.0

        return {
            "total_reward": round(self.total_reward, 2),
            "total_sales_units": round(self.total_sales, 0),
            "total_stockout_units": round(self.total_stockout, 0),
            "avg_discount": round(self.pricing_agent.avg_discount, 3),
            "avg_price": round(sum(self.price_history) / len(self.price_history), 2),
            "service_level": round(service_level, 4),
            "replenishment_count": len(self.replenishment_agent._order_history),
            "total_replenishment_qty": round(self.replenishment_agent.total_ordered, 0),
            "final_inventory": round(self.inventory_history[-1], 0),
        }


# ---------------------------------------------------------------------------
# 7. 自测
# ---------------------------------------------------------------------------

def _run_unit_tests() -> None:
    """单元测试：验证各核心组件的正确性"""
    print("=" * 60)
    print("【单元测试】")

    # 测试 1: EnvState.to_vector() 维度正确
    state = EnvState(
        inventory=5000.0,
        days_remaining=15,
        competitor_price=280.0,
        base_price=299.0,
        cost_price=120.0,
        day=8,
    )
    vec = state.to_vector()
    assert len(vec) == 6, f"期望 6 维状态向量，实际 {len(vec)} 维"
    print(f"  ✓ EnvState.to_vector() 维度正确: {len(vec)}")

    # 测试 2: DemandModel 需求为非负值
    dm = DemandModel(base_demand=500.0, elasticity=1.8, random_seed=0)
    for price_ratio in [0.7, 0.9, 1.0, 1.2]:
        d = dm.sample(299.0 * price_ratio, 299.0, 289.0)
        assert d >= 0, f"需求不应为负: {d}"
    print(f"  ✓ DemandModel 需求均非负")

    # 测试 3: 价格弹性效果（低价 → 高需求）
    dm2 = DemandModel(base_demand=500.0, elasticity=1.8, noise_std=0.0)
    d_low = dm2.sample(0.7 * 299.0, 299.0, 289.0)
    d_high = dm2.sample(1.0 * 299.0, 299.0, 289.0)
    assert d_low > d_high, f"低价需求({d_low:.1f})应大于高价需求({d_high:.1f})"
    print(f"  ✓ 价格弹性: 折扣价需求={d_low:.1f} > 原价需求={d_high:.1f}")

    # 测试 4: compute_reward 正常场景利润为正
    reward = compute_reward(
        price=254.15,     # 折扣价
        cost_price=120.0,
        actual_sales=450.0,
        holding_inventory=4550.0,
        holding_cost_per_unit=0.5,
        stockout_units=0.0,
    )
    assert reward > 0, f"正常销售应盈利，实际 reward={reward}"
    print(f"  ✓ compute_reward 正常场景利润为正: {reward:.2f} 元")

    # 测试 5: PricingAgent 库存紧张时提价（折扣应高）
    agent = PricingAgent(safety_stock_ratio=0.20)
    agent.reset(initial_inventory=8000.0)
    # 库存远低于安全水位（1000 < 8000×0.2=1600）
    low_inv_state = EnvState(
        inventory=1000.0, days_remaining=10,
        competitor_price=289.0, base_price=299.0, cost_price=120.0
    )
    discount_low = agent.compute_discount(low_inv_state)
    # 库存充足时折扣应更低
    high_inv_state = EnvState(
        inventory=6000.0, days_remaining=10,
        competitor_price=289.0, base_price=299.0, cost_price=120.0
    )
    discount_high = agent.compute_discount(high_inv_state)
    assert discount_low >= discount_high, (
        f"库存紧张时折扣({discount_low})应 ≥ 库存充足时折扣({discount_high})"
    )
    print(f"  ✓ PricingAgent: 库存紧张折扣={discount_low}, 库存充足折扣={discount_high}")

    # 测试 6: ReplenishmentAgent 库存充足时不补货
    rep_agent = ReplenishmentAgent()
    full_state = EnvState(
        inventory=14000.0, days_remaining=5,
        competitor_price=289.0, base_price=299.0, cost_price=120.0, pending_order=0.0
    )
    order = rep_agent.compute_order_qty(full_state, 400.0, 0.85)
    assert order == 0.0, f"库存充足+剩余天数少时，补货量应为 0，实际={order}"
    print(f"  ✓ ReplenishmentAgent: 库存充足时补货量=0")

    print("  【所有单元测试通过】\n")


def _run_integration_test() -> None:
    """集成测试：完整运行 30 天大促仿真"""
    print("=" * 60)
    print("【集成测试】30 天大促仿真")

    sim = PromoSimulator(
        initial_inventory=8000.0,
        base_price=299.0,
        cost_price=120.0,
        competitor_price=289.0,
        lead_time_days=3,
        random_seed=42,
    )
    result = sim.run_episode()

    print(f"  总奖励（利润）:       {result['total_reward']:>10,.2f} 元")
    print(f"  总销售量:             {result['total_sales_units']:>10,.0f} 件")
    print(f"  累计缺货量:           {result['total_stockout_units']:>10,.0f} 件")
    print(f"  平均折扣率:           {result['avg_discount']:>10.1%}")
    print(f"  平均售价:             {result['avg_price']:>10.2f} 元")
    print(f"  服务率（填充率）:     {result['service_level']:>10.1%}")
    print(f"  补货次数:             {result['replenishment_count']:>10}")
    print(f"  累计补货量:           {result['total_replenishment_qty']:>10,.0f} 件")
    print(f"  期末库存:             {result['final_inventory']:>10,.0f} 件")

    # 验证约束
    assert result["service_level"] >= 0.0
    assert result["service_level"] <= 1.0
    assert result["total_sales_units"] >= 0
    assert result["avg_price"] > 0

    print("\n  【集成测试通过】")

    # 对比实验：无协同（定价固定 85% 折扣，不补货）
    print("\n【对比实验】无双智能体协同（固定 85% 折扣，不补货）")

    class FixedPriceNoReplenishment:
        """对照组：定价 85% 折扣，不补货"""

        def run(self) -> dict:
            rng = random.Random(42)
            dm = DemandModel(500.0, 1.8, 0.15, random_seed=42)
            inventory = 8000.0
            base_price, cost_price = 299.0, 120.0
            total_reward, total_sales, total_stockout = 0.0, 0.0, 0.0
            for _ in range(30):
                price = base_price * 0.85
                demand = dm.sample(price, base_price, 289.0)
                sales = min(demand, inventory)
                stockout = max(0.0, demand - inventory)
                reward = compute_reward(price, cost_price, sales, inventory, 0.5, stockout)
                inventory -= sales
                total_reward += reward
                total_sales += sales
                total_stockout += stockout
            total_demand = total_sales + total_stockout
            sl = total_sales / total_demand if total_demand > 0 else 1.0
            return {
                "total_reward": round(total_reward, 2),
                "total_sales_units": round(total_sales, 0),
                "service_level": round(sl, 4),
            }

    baseline = FixedPriceNoReplenishment().run()
    print(f"  对照组总利润:         {baseline['total_reward']:>10,.2f} 元")
    print(f"  对照组总销售量:       {baseline['total_sales_units']:>10,.0f} 件")
    print(f"  对照组服务率:         {baseline['service_level']:>10.1%}")

    improvement = result["total_reward"] - baseline["total_reward"]
    print(f"\n  FSDA-DRL 利润提升:    {improvement:>+10,.2f} 元")
    print(f"  FSDA-DRL 服务率提升:  {(result['service_level'] - baseline['service_level']):>+10.1%}")


if __name__ == "__main__":
    print("FSDA-DRL: Fast-Slow Dual-Agent Deep Reinforcement Learning")
    print("用于跨境电商大促期间动态定价 + 补货联合优化\n")

    _run_unit_tests()
    _run_integration_test()

    print("\n" + "=" * 60)
    print("自测全部通过 ✓")

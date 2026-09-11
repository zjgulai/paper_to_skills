"""
Deep RL for Two-Echelon Inventory Optimization
母婴出海场景：区域仓备货优化、新品库存策略

基于论文：Stranieri & Stella (2022) "Comparing Deep RL Algorithms in Two-Echelon Supply Chains"
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, List
from collections import deque
import random


class TwoEchelonInventoryEnv:
    """
    两阶段库存环境

    结构：Factory -> Regional Warehouses -> Customers
    """

    def __init__(self,
                 n_warehouses: int = 3,
                 lead_time_factory: int = 3,
                 lead_time_warehouse: List[int] = None,
                 holding_cost: float = 0.01,
                 stockout_cost: float = 1.0,
                 shipping_cost: float = 0.5,
                 price: float = 10.0,
                 max_inventory: int = 100):
        """
        初始化环境

        Args:
            n_warehouses: 区域仓数量
            lead_time_factory: 工厂到区域仓的交货期（天数）
            lead_time_warehouse: 各区域仓到客户的交货期
            holding_cost: 单位库存持有成本（每天）
            stockout_cost: 单位缺货成本
            shipping_cost: 单位调拨成本
            price: 产品售价
            max_inventory: 最大库存容量
        """
        self.n_warehouses = n_warehouses
        self.lead_time_factory = lead_time_factory
        self.lead_time_warehouse = lead_time_warehouse or [1] * n_warehouses
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.shipping_cost = shipping_cost
        self.price = price
        self.max_inventory = max_inventory

        # 需求参数（模拟季节性）
        self.base_demand = [20, 15, 25]  # 各仓基础需求
        self.seasonality = [0.1, 0.2, 0.15]  # 季节性强度

        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.timestep = 0

        # 各区域仓库存
        self.warehouse_inventory = np.ones(self.n_warehouses) * 50

        # 在途库存（工厂发货，按天数组织）
        self.in_transit_factory = [deque([0] * self.lead_time_factory)
                                   for _ in range(self.n_warehouses)]

        # 历史需求（用于需求预测特征）
        self.demand_history = [deque([20] * 7, maxlen=7) for _ in range(self.n_warehouses)]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state = []

        # 各仓库存水平
        state.extend(self.warehouse_inventory / self.max_inventory)

        # 在途库存（最近即将到达的）
        for w in range(self.n_warehouses):
            if self.in_transit_factory[w]:
                state.append(self.in_transit_factory[w][0] / self.max_inventory)
            else:
                state.append(0)

        # 历史需求均值（7天移动平均）
        for w in range(self.n_warehouses):
            avg_demand = np.mean(self.demand_history[w])
            state.append(avg_demand / 50)  # 归一化

        # 时间特征（月份，用于捕捉季节性）
        month = (self.timestep // 30) % 12
        state.append(np.sin(2 * np.pi * month / 12))  # 月份正弦编码
        state.append(np.cos(2 * np.pi * month / 12))  # 月份余弦编码

        return np.array(state, dtype=np.float32)

    def _generate_demand(self) -> np.ndarray:
        """生成各仓需求（带季节性）"""
        demands = []
        for w in range(self.n_warehouses):
            base = self.base_demand[w]
            # 季节性波动
            season = self.seasonality[w] * base * np.sin(2 * np.pi * self.timestep / 365)
            # 随机噪声
            noise = np.random.normal(0, base * 0.2)
            demand = max(0, int(base + season + noise))
            demands.append(demand)
        return np.array(demands)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作

        Args:
            action: 动作向量 [补货到仓1, 补货到仓2, ...]

        Returns:
            state, reward, done, info
        """
        shipments = np.clip(action, 0, None).astype(int)

        # 1. 接收在途库存（到达的货物）
        for w in range(self.n_warehouses):
            if self.in_transit_factory[w]:
                arrived = self.in_transit_factory[w].popleft()
                self.warehouse_inventory[w] += arrived
                self.in_transit_factory[w].append(0)  # 保持队列长度
            else:
                arrived = 0

        # 2. 新发货进入在途
        for w in range(self.n_warehouses):
            if len(self.in_transit_factory[w]) > 0:
                self.in_transit_factory[w][-1] = shipments[w]

        # 3. 需求到达
        demands = self._generate_demand()

        # 4. 满足需求
        sales = np.minimum(self.warehouse_inventory, demands)
        stockouts = demands - sales
        self.warehouse_inventory -= sales

        # 5. 记录需求历史
        for w in range(self.n_warehouses):
            self.demand_history[w].append(demands[w])

        # 6. 计算奖励
        revenue = np.sum(sales) * self.price
        holding_cost = np.sum(self.warehouse_inventory) * self.holding_cost
        stockout_cost = np.sum(stockouts) * self.stockout_cost
        shipping_cost = np.sum(shipments) * self.shipping_cost

        reward = revenue - holding_cost - stockout_cost - shipping_cost

        # 归一化奖励（便于RL学习）
        reward = reward / 1000.0

        self.timestep += 1
        done = self.timestep >= 365  # 一年的模拟

        info = {
            'sales': sales,
            'stockouts': stockouts,
            'demands': demands,
            'shipments': shipments,
            'inventory': self.warehouse_inventory.copy()
        }

        return self._get_state(), reward, done, info


class ActorNetwork(nn.Module):
    """策略网络（Actor）"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()  # 确保输出为正（库存补货量）
        )

    def forward(self, state):
        return self.net(state)


class CriticNetwork(nn.Module):
    """价值网络（Critic）"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    """PPO (Proximal Policy Optimization) Agent"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 epsilon: float = 0.2, epochs: int = 10):
        """
        初始化PPO智能体

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            epsilon: PPO裁剪参数
            epochs: 每次更新的迭代次数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs

        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 动作噪声（探索）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            mean = self.actor(state_tensor)

            if deterministic:
                action = mean
            else:
                std = torch.exp(self.log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

            return action.squeeze(0).numpy()

    def update(self, states, actions, rewards, next_states, dones):
        """更新策略"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 计算优势函数
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values

        # PPO更新
        for _ in range(self.epochs):
            # 当前策略的动作分布
            mean = self.actor(states)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)

            # 价值损失
            new_values = self.critic(states)
            value_loss = nn.MSELoss()(new_values, td_targets)

            # 策略损失（简化版PPO）
            policy_loss = -(log_probs * advantages).mean()

            # 更新Actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # 更新Critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        return policy_loss.item(), value_loss.item()


def train_inventory_rl():
    """训练库存优化RL模型"""
    print("=" * 60)
    print("两阶段库存优化 - PPO训练")
    print("=" * 60)

    # 创建环境
    env = TwoEchelonInventoryEnv(
        n_warehouses=3,
        lead_time_factory=3,
        holding_cost=0.5,
        stockout_cost=5.0,
        shipping_cost=1.0,
        price=20.0
    )

    state_dim = len(env.reset())
    action_dim = env.n_warehouses

    print(f"\n环境参数:")
    print(f"  区域仓数量: {env.n_warehouses}")
    print(f"  工厂交货期: {env.lead_time_factory}天")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")

    # 创建Agent
    agent = PPOAgent(state_dim, action_dim, lr=3e-4)

    # 训练循环
    n_episodes = 200
    batch_size = 32

    states_buffer = []
    actions_buffer = []
    rewards_buffer = []
    next_states_buffer = []
    dones_buffer = []

    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            # 选择动作
            action = agent.select_action(state, deterministic=False)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储经验
            states_buffer.append(state)
            actions_buffer.append(action)
            rewards_buffer.append(reward)
            next_states_buffer.append(next_state)
            dones_buffer.append(float(done))

            episode_reward += reward
            state = next_state

            # 批量更新
            if len(states_buffer) >= batch_size:
                agent.update(
                    np.array(states_buffer),
                    np.array(actions_buffer),
                    np.array(rewards_buffer),
                    np.array(next_states_buffer),
                    np.array(dones_buffer)
                )
                states_buffer = []
                actions_buffer = []
                rewards_buffer = []
                next_states_buffer = []
                dones_buffer = []

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}")

    # 评估训练好的策略
    print("\n" + "=" * 60)
    print("评估训练后的策略")
    print("=" * 60)

    state = env.reset()
    total_sales = 0
    total_stockouts = 0
    total_holding_cost = 0

    daily_metrics = []

    while True:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)

        daily_metrics.append({
            'day': env.timestep,
            'warehouse_inventory': info['inventory'].tolist(),
            'shipments': info['shipments'].tolist(),
            'demand': info['demands'].tolist(),
            'sales': info['sales'].tolist(),
            'stockouts': info['stockouts'].tolist(),
            'reward': reward
        })

        total_sales += np.sum(info['sales'])
        total_stockouts += np.sum(info['stockouts'])
        total_holding_cost += np.sum(info['inventory']) * env.holding_cost

        state = next_state
        if done:
            break

    df_metrics = pd.DataFrame(daily_metrics)

    print(f"\n年度运营指标:")
    print(f"  总销售量: {int(total_sales)} 件")
    print(f"  总缺货量: {int(total_stockouts)} 件")
    print(f"  满足率: {total_sales / (total_sales + total_stockouts):.1%}")
    print(f"  平均每日库存持有成本: ${total_holding_cost / 365:.2f}")

    # 各仓分析
    print(f"\n各区域仓表现:")
    for w in range(env.n_warehouses):
        wh_sales = sum([d['sales'][w] for d in daily_metrics])
        wh_stockouts = sum([d['stockouts'][w] for d in daily_metrics])
        wh_avg_inv = np.mean([d['warehouse_inventory'][w] for d in daily_metrics])
        print(f"  仓库{w+1}: 销售{int(wh_sales)}件, 缺货{int(wh_stockouts)}件, 平均库存{wh_avg_inv:.1f}件")

    return agent, env, df_metrics


if __name__ == "__main__":
    # 训练模型
    agent, env, metrics = train_inventory_rl()

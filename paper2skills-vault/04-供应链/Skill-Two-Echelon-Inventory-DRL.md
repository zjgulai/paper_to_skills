# Skill Card: Deep RL for Two-Echelon Inventory Optimization

---

## ① 算法原理

### 核心思想
多级库存优化（Multi-Echelon Inventory Optimization, MEIO）解决的是供应链中多个节点（工厂、仓库、门店）的联合库存决策问题。相比传统的单点库存管理，DRL方法将供应链建模为**马尔可夫决策过程（MDP）**，智能体（Agent）学习在每个时间步决定"生产多少、发往哪里"，以最大化长期累积利润。

两阶段库存（Two-Echelon）是MEIO的基础形式，包含：
- **上游节点**：工厂/中央仓（决定生产量、发货量）
- **下游节点**：区域仓/门店（决定订货量、库存策略）

### 数学直觉
**状态空间 S**：
- 各节点库存水平
- 在途库存（已发货未到达）
- 需求预测（历史需求、季节性、趋势）
- 时间特征（月份、促销周期）

**动作空间 A**：
- 工厂生产量
- 各仓库之间的调拨量
- 订货点/订货量决策

**奖励函数 R**：
```
R = 销售收入 - 生产成本 - 库存持有成本 - 缺货惩罚 - 调拨成本
```

**状态转移**：
- 执行动作后，库存水平更新
- 新需求到达，产生销售或缺货
- 在途库存向前推进

**DRL算法选择**：
- **DQN (Deep Q-Network)**：适用于离散动作空间，学习Q(s,a)值函数
- **PPO (Proximal Policy Optimization)**：策略梯度方法，适用于连续动作空间，训练稳定
- **A3C (Asynchronous Advantage Actor-Critic)**：异步并行训练，收敛快

### 关键假设
1. **需求可预测性**：需求虽有随机性，但服从可学习的分布
2. **交货期稳定**：Lead Time虽有波动但可估计
3. **信息共享**：各节点库存信息可实时获取（中心化控制）
4. **无限产能弹性**：或至少在合理范围内产能可扩展
5. **可模拟性**：供应链动态可通过仿真环境近似

---

## ② 母婴出海应用案例

### 场景1：跨境母婴商品区域仓备货优化

**业务问题**
某母婴出海公司在东南亚有1个中央仓（深圳）和3个区域仓（新加坡、雅加达、曼谷）。每个区域的需求模式不同：
- 新加坡：需求稳定，对缺货敏感（用户期望次日达）
- 雅加达：需求波动大，受本地促销影响
- 曼谷：季节性明显，雨季需求下降

传统(s, Q)策略无法处理：
- 跨区域调拨的运输成本差异
- 汇率波动对成本的影响
- 不同品类的季节性差异（如湿巾雨季需求变化小，但户外用品变化大）

**数据要求**
| 字段 | 说明 | 来源 |
|------|------|------|
| date | 日期 | 系统 |
| warehouse | 仓库ID | 主数据 |
| sku | 商品编码 | 主数据 |
| beginning_inventory | 期初库存 | WMS |
| in_transit | 在途库存 | TMS |
| demand | 当日需求（订单量） | OMS |
| fulfilled | 当日满足量 | OMS |
| lost_sales | 缺货损失 | OMS |
| production_cost | 生产成本 | ERP |
| holding_cost_rate | 库存持有费率 | 财务 |
| stockout_penalty | 缺货惩罚成本 | 运营 |

**预期产出**
- 每日各SKU在各仓的补货建议（订货量、调拨量）
- 库存周转率从4次/年提升至6次/年
- 缺货率从8%降至3%以下
- 库存持有成本降低20%

**业务价值**
- **资金效率**：减少滞销库存占压资金，释放现金流
- **用户体验**：降低缺货率，提升订单满足率
- **运营自动化**：减少人工制定补货计划的工作量

### 场景2：新品上市库存策略学习

**业务问题**
母婴行业新品迭代快（如新款奶瓶、新配方奶粉）。传统方法需要3-6个月的历史数据才能建立稳定的需求预测模型，导致：
- 新品上市初期要么大量缺货错失机会，要么过量备货造成滞销
- 不同市场的接受度差异大，难以用统一策略

**数据要求**
- 新品上市后每日销售数据
- 营销投入（广告 spend、曝光量）
- 竞品动态（价格、促销）
- 用户反馈（评论情感、评分）

**预期产出**
- 新品上市前30天的动态库存策略
- 基于实时销售反馈的快速补货触发机制
- 跨区域库存调拨建议（A市场滞销、B市场热销时）

**业务价值**
- **缩短冷启动**：从3个月的学习期缩短至2周
- **降低新品风险**：通过快速反馈调整，减少滞销可能
- **市场响应速度**：快速捕捉爆款信号，及时补货

---

## ③ 代码模板

### 两阶段库存环境 + PPO训练

```python
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
            
            # 比率（新旧策略概率比）
            # 简化版本：直接计算当前策略的优势
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
        wh_sales = df_metrics['sales'].apply(lambda x: x[w]).sum()
        wh_stockouts = df_metrics['stockouts'].apply(lambda x: x[w]).sum()
        wh_avg_inv = df_metrics['warehouse_inventory'].apply(lambda x: x[w]).mean()
        print(f"  仓库{w+1}: 销售{int(wh_sales)}件, 缺货{int(wh_stockouts)}件, 平均库存{wh_avg_inv:.1f}件")
    
    return agent, env, df_metrics


def compare_policies():
    """对比RL策略与基准策略"""
    print("\n" + "=" * 60)
    print("策略对比: RL vs (s, Q)策略")
    print("=" * 60)
    
    # 基准策略：固定(s, Q)
    def base_policy(state, s=20, Q=30):
        """简单的(s, Q)策略"""
        inventory = state[:3] * 100  # 反归一化
        action = np.where(inventory < s, Q, 0)
        return action
    
    env = TwoEchelonInventoryEnv()
    
    results = {}
    
    for policy_name, policy_fn in [('Base (s,Q)', base_policy)]:
        state = env.reset()
        total_reward = 0
        
        while True:
            action = policy_fn(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        
        results[policy_name] = total_reward
        print(f"{policy_name}: 年度总奖励 = ${total_reward * 1000:.2f}")


if __name__ == "__main__":
    # 训练模型
    agent, env, metrics = train_inventory_rl()
    
    # 对比基准
    compare_policies()
```

### 需求预测模块（用于状态特征）

```python
class DemandForecaster:
    """
    需求预测器（用于RL状态中的需求预测特征）
    """
    
    def __init__(self, window_size: int = 14):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def update(self, demand: float):
        """更新历史需求"""
        self.history.append(demand)
    
    def forecast(self, horizon: int = 7) -> float:
        """
        简单移动平均预测
        实际场景中可替换为Prophet/LightGBM等复杂模型
        """
        if len(self.history) < 7:
            return np.mean(self.history) if self.history else 20.0
        
        # 7天移动平均
        recent_mean = np.mean(list(self.history)[-7:])
        
        # 加入趋势
        if len(self.history) >= 14:
            first_week = np.mean(list(self.history)[:7])
            second_week = np.mean(list(self.history)[7:14])
            trend = second_week - first_week
            forecast = recent_mean + trend * (horizon / 7)
            return max(0, forecast)
        
        return recent_mean


def analyze_bullwhip_effect(metrics_df: pd.DataFrame):
    """
    分析牛鞭效应（需求波动向上游放大）
    """
    # 计算需求方差和订货方差
    demand_variance = metrics_df['demand'].apply(np.sum).var()
    shipment_variance = metrics_df['shipments'].apply(np.sum).var()
    
    bullwhip_ratio = shipment_variance / demand_variance if demand_variance > 0 else 1.0
    
    print(f"\n牛鞭效应分析:")
    print(f"  需求方差: {demand_variance:.2f}")
    print(f"  订货方差: {shipment_variance:.2f}")
    print(f"  牛鞭系数: {bullwhip_ratio:.2f} (越接近1越好)")
    
    return bullwhip_ratio


if __name__ == "__main__":
    # 示例：训练后分析
    print("需求预测模块示例")
    forecaster = DemandForecaster()
    
    # 模拟历史需求
    for _ in range(20):
        forecaster.update(np.random.poisson(25))
    
    prediction = forecaster.forecast(horizon=7)
    print(f"7天需求预测: {prediction:.1f} 件/天")
```

---

## ④ 技能关联

### 前置技能
- **库存管理基础**：EOQ、ROP、(s, Q)策略、安全库存计算
- **强化学习基础**：MDP、策略梯度、价值函数
- **供应链知识**：BOM、Lead Time、牛鞭效应

### 延伸技能
- **多智能体RL**：分布式库存决策（各节点独立决策但协作）
- **随机优化**：考虑需求不确定性的鲁棒优化
- **数字孪生**：供应链仿真环境的精细化建模

### 可组合
- **Time Series Forecasting**：为RL状态提供更准确的需求预测特征
- **Causal Forest**：分析不同库存策略对各区域销售的因果效应
- **LTV Prediction**：区分高价值SKU和普通SKU，设置不同服务水平

---

## ⑤ 商业价值评估

### ROI预估
| 指标 | 预估 | 说明 |
|------|------|------|
| 库存周转率 | +40-60% | 智能补货减少冗余库存 |
| 缺货损失 | -50% | 精准预测降低缺货率 |
| 库存持有成本 | -25% | 平均库存水平优化 |
| 人工计划工作量 | -60% | 自动化补货决策 |

### 实施难度
⭐⭐⭐⭐☆ (4/5)
- 需要准确的供应链数据（库存、在途、需求）
- RL训练需要仿真环境，离线训练周期较长
- 需要与WMS/TMS/ERP系统集成
- 上线后需要监控，防止极端决策

### 优先级评分
⭐⭐⭐⭐☆ (4/5)
- 高价值：直接降低库存成本，提升客户满意度
- 战略意义：供应链数字化转型的核心技术
- 可扩展性：从两阶段扩展到多阶段、多SKU复杂场景

### 评估依据
1. **资金效率**：库存是母婴出海企业的主要资金占用，优化库存直接释放现金流
2. **体验提升**：降低缺货率对母婴品类尤为重要（用户不愿等待）
3. **运营自动化**：减少人工制定补货计划的工作量
4. **季节性应对**：DRL能比传统方法更好处理需求波动（如双11、黑五）

---

## 参考资料
- Stranieri, F., & Stella, F. (2022). "Comparing Deep Reinforcement Learning Algorithms in Two-Echelon Supply Chains." arXiv:2204.09603.
- Gijsbrechts, J., et al. (2022). "Can Deep Reinforcement Learning Improve Inventory Management?" Manufacturing & Service Operations Management.
- Clark, A.J. & Scarf, H. (1960). "Optimal policies for a multi-echelon inventory problem." Management Science.

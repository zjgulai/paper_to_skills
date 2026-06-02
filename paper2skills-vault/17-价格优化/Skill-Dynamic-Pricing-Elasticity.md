# Skill Card: Dynamic Pricing with Demand Elasticity（动态定价与需求弹性）

> **领域**: 17-价格优化 | **类型**: 综合萃取 | **服务工作流**: WF-B (广告优化联动定价)

---

## ① 算法原理

### 核心思想
**价格不是固定数字，而是杠杆**——通过估计每个 SKU/市场/时段的需求价格弹性，找到利润最大化的动态定价策略。核心公式：最优价格 $P^* = \frac{\epsilon}{\epsilon+1} \cdot MC$，其中 $\epsilon$ 是需求价格弹性，$MC$ 是边际成本。

### 数学直觉

**需求价格弹性**（Price Elasticity of Demand）：
$$\epsilon = \frac{\%\Delta Q}{\%\Delta P} = \frac{\partial Q/Q}{\partial P/P}$$

- $|\epsilon| > 1$：弹性需求（涨价 → 销量大降 → 总收入降）
- $|\epsilon| < 1$：非弹性需求（涨价 → 销量微降 → 总收入升）
- 母婴品类典型弹性：吸奶器 $|\epsilon| \approx 1.2-1.8$（弹性），配件 $|\epsilon| \approx 0.5-0.8$（非弹性）

**动态定价 DRL 框架**：
- State: 当前价格、库存、竞品价格、时间（季节/促销期）、需求预测
- Action: 调价幅度 $\Delta P \in [-20\%, +20\%]$
- Reward: $R = (P - C) \cdot Q(P) - \lambda \cdot |\Delta P|$（利润 - 调价成本）
- 用 DQN/PPO 学习最优定价策略

### 关键假设
- 需求弹性在短期内稳定（6-12 个月可重新估计）
- 竞品价格可观测（通过爬虫或第三方数据）
- 调价频率有限制（亚马逊每小时限调价次数）

---

## ② 母婴出海应用案例

### 场景：吸奶器多市场价格动态优化

**业务问题**：吸奶器在美国定价 $129，德国 €119，英国 £99。但美国市场竞品 Momcozy 经常在亚马逊闪电促销降价到 $99，导致我们的转化率周期性波动。需要动态调价策略——竞品降价时跟多少？竞品恢复后涨回去吗？

**数据要求**：
- 各市场 12 个月日销量 + 价格历史
- 竞品价格（Momcozy/Medela/Spectra 等 3-5 个竞品）每日监测
- 各市场价格弹性估计（来自历史 A/B 测试或准实验）

**预期产出**：
- 弹性矩阵：美国 $|\epsilon|=1.5$（高度弹性→不宜轻易涨价）、德国 $|\epsilon|=1.1$（中等）、英国 $|\epsilon|=0.9$（偏非弹性→涨价空间大）
- DRL 策略：竞品降价 15% 时，我们跟降 8-12%（而非完全匹配）以平衡利润和份额
- 最优价格区间：美国 $119-135 / 德国 €109-125 / 英国 £95-108

**业务价值**：
- 利润提升：DRL 策略 vs 固定价格，利润率 +8-12%
- 月 GMV $50 万 → 利润提升 $4,000-6,000/月
- 年化 ROI：**50-80 万元**

---

## ③ 代码模板

```python
"""
Dynamic Pricing with Demand Elasticity — DRL + Bayesian 弹性估计
"""

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple


class DemandElasticityEstimator:
    """需求价格弹性估计"""
    
    def __init__(self, prior_elasticity: float = -1.5, prior_std: float = 0.5):
        self.elasticity = prior_elasticity
        self.std = prior_std
        self.n_obs = 0
    
    def update(self, price_change_pct: float, demand_change_pct: float):
        """贝叶斯更新弹性估计"""
        if abs(price_change_pct) < 0.001:
            return
        new_obs = demand_change_pct / price_change_pct
        self.n_obs += 1
        alpha = 1.0 / (1.0 + self.n_obs)
        self.elasticity = (1 - alpha) * self.elasticity + alpha * new_obs
        self.std *= 0.95  # 随观测增加不确定性递减
    
    def optimal_price(self, cost: float, current_price: float,
                      competitor_price: float = None) -> float:
        """计算最优价格"""
        eps = abs(self.elasticity)
        if eps <= 1.0:
            # 非弹性需求 → 涨价
            return current_price * 1.05
        margin = (eps - 1) / eps
        optimal = cost / (1 - margin) if margin > 0 else cost * 2
        
        if competitor_price and competitor_price < optimal:
            # 竞品约束：不高于竞品 15% 以上
            optimal = min(optimal, competitor_price * 1.15)
        
        # 调价幅度限制
        return np.clip(optimal, current_price * 0.8, current_price * 1.2)


class DynamicPricingAgent:
    """简化 DRL 动态定价 Agent"""
    
    def __init__(self, cost: float, elasticity_estimator: DemandElasticityEstimator):
        self.cost = cost
        self.elasticity = elasticity_estimator
        self.price_history = []
        self.demand_history = []
    
    def decide_price(self, current_price: float, inventory: int,
                     demand_forecast: float, competitor_price: float) -> float:
        """
        定价决策
        
        Args:
            current_price: 当前价格
            inventory: 库存量
            demand_forecast: 预测需求
            competitor_price: 竞品价格
        """
        # 库存压力调整
        days_of_stock = inventory / max(demand_forecast, 0.1)
        inventory_pressure = 1.0 if days_of_stock < 14 else 0.0
        
        # 竞品压力
        price_ratio = current_price / competitor_price if competitor_price > 0 else 1.0
        competitor_pressure = max(0, price_ratio - 1.0) if price_ratio > 1.05 else 0.0
        
        # 综合调价幅度
        base_price = self.elasticity.optimal_price(self.cost, current_price, competitor_price)
        adjustment = 1.0 - 0.1 * inventory_pressure - 0.15 * competitor_pressure
        new_price = base_price * np.clip(adjustment, 0.85, 1.05)
        
        return round(new_price, 2)
    
    def observe(self, price: float, demand: float):
        """观测结果并更新弹性"""
        if len(self.price_history) > 0:
            price_change = (price - self.price_history[-1]) / self.price_history[-1]
            demand_change = (demand - self.demand_history[-1]) / max(self.demand_history[-1], 0.01)
            self.elasticity.update(price_change, demand_change)
        self.price_history.append(price)
        self.demand_history.append(demand)


# ============ 测试 ============

if __name__ == '__main__':
    np.random.seed(42)
    
    # 模拟：真实弹性=-1.5, 成本=$80, 初始价格=$129
    estimator = DemandElasticityEstimator(prior_elasticity=-1.2)
    agent = DynamicPricingAgent(cost=80, elasticity_estimator=estimator)
    
    price = 129.0
    demand = 100.0
    competitor = 119.0
    
    print(f"初始: 价格=${price}, 需求={demand:.0f}, 竞品=${competitor}")
    
    for day in range(30):
        # 竞品动态变化
        competitor += np.random.normal(0, 2)
        competitor = np.clip(competitor, 99, 139)
        
        new_price = agent.decide_price(price, inventory=200, 
                                        demand_forecast=demand, 
                                        competitor_price=competitor)
        
        # 模拟需求响应（基于真实弹性 -1.5 + 噪声）
        price_change = (new_price - price) / price
        demand_change = -1.5 * price_change + np.random.normal(0, 0.05)
        demand = max(demand * (1 + demand_change), 20)
        
        agent.observe(new_price, demand)
        price = new_price
    
    print(f"最终: 价格=${price:.2f}, 需求={demand:.0f}, "
          f"估计弹性={estimator.elasticity:.2f} (真实=-1.5)")
    print(f"最优价格: ${estimator.optimal_price(80, price, competitor):.2f}")
    
    assert estimator.elasticity < 0, "Elasticity should be negative"
    print("\n[✓] Dynamic Pricing 测试通过")
```

---

## ④ 技能关联

- **前置技能**：[[Skill-ROAS-Budget-Optimization]] | [[Skill-Demand-Forecasting-Supply-Chain]]
- **延伸技能**：[[Skill-Competitive-Price-Monitoring]] | [[Skill-Cross-Border-Price-Harmonization]]
- **可组合技能**：[[Skill-Uplift-Modeling]]（因果弹性 vs 相关性弹性）| [[Skill-Multi-Channel-Inventory-Pooling]]

---

## ⑤ 商业价值评估

- **ROI 预估**：利润率 +8-12%，月 GMV $50 万 → 年化 **50-80 万元**
- **实施难度**：⭐⭐⭐☆☆（3 星）— 需要持续竞品监测 + A/B 验证
- **优先级评分**：⭐⭐⭐⭐⭐（5 星）— 定价是电商四大杠杆之首（定价 > 流量 > 转化率 > 复购率）
- **评估依据**：价格弹性估计是所有定价决策的基础，新领域 17-价格优化的第一张核心卡片

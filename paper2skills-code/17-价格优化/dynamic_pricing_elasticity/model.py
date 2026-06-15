"""
Auto-extracted from: paper2skills-vault/17-价格优化/Skill-Dynamic-Pricing-Elasticity.md
Skill: Skill-Dynamic-Pricing-Elasticity
Domain: 17-价格优化
"""
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

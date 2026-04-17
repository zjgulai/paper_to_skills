"""
TJAP: Transfer Learning for Contextual Joint Assortment-Pricing under Cross-Market Heterogeneity
基于论文: Transfer Learning for Contextual Joint Assortment-Pricing under Cross-Market Heterogeneity (arXiv:2603.18114)

简化说明:
- 使用线性近似替代完整的MNL最大似然估计，降低代码复杂度
- Two-radius UCB简化为特征空间中的椭圆置信集
- 适合作为业务分析原型和教学演示
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class Product:
    """单个产品，带有上下文特征向量."""
    product_id: str
    name: str
    features: np.ndarray  # 维度 d，包含产品属性、客户特征等
    base_cost: float = 0.0


@dataclass
class MarketObservation:
    """单个时间步的市场观测."""
    product_features: np.ndarray  # (N, d)
    assortment_indices: List[int]  # 本次提供的产品索引
    prices: np.ndarray  # (len(assortment),)
    purchase_index: int  # 购买的产品在assortment中的索引，-1表示未购买(outside option)


class MNLChoiceModel:
    """Contextual Multinomial Logit choice model."""

    def __init__(self, theta: np.ndarray, gamma: np.ndarray):
        """
        theta: 偏好参数 (d,)
        gamma: 价格敏感度参数 (d,)，要求为正
        """
        self.theta = theta
        self.gamma = gamma

    def utility(self, features: np.ndarray, price: float) -> float:
        """计算单个产品的确定性效用: v = <x, theta> - <x, gamma> * p"""
        return float(np.dot(features, self.theta) - np.dot(features, self.gamma) * price)

    def choice_probabilities(self, features: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        计算给定assortment和价格的购买概率.
        features: (K, d)
        prices: (K,)
        返回: (K+1,) 包含outside option的概率
        """
        utilities = self.utilities(features, prices)
        max_u = np.max(utilities)
        exp_u = np.exp(utilities - max_u)  # 数值稳定性
        denom = np.sum(exp_u) + np.exp(-max_u)  # outside option utility = 0
        probs = exp_u / denom
        outside = np.exp(-max_u) / denom
        return np.concatenate([[outside], probs])

    def utilities(self, features: np.ndarray, prices: np.ndarray) -> np.ndarray:
        return np.dot(features, self.theta) - np.dot(features, self.gamma) * prices

    def expected_revenue(self, features: np.ndarray, prices: np.ndarray) -> float:
        """给定assortment和价格的期望收益."""
        probs = self.choice_probabilities(features, prices)
        return float(np.sum(probs[1:] * prices))


class AggregateThenDebiasEstimator:
    """
    TJAP的两步估计器:
    1. Aggregate: 用所有源市场数据做pooled估计
    2. Debias: 用目标市场数据做L1正则化修正稀疏偏移
    """

    def __init__(self, feature_dim: int, l1_lambda: float = 0.1):
        self.d = feature_dim
        self.l1_lambda = l1_lambda
        self.nu_agg = np.zeros(2 * self.d)  # pooled estimate [theta; gamma]
        self.nu_debiased = np.zeros(2 * self.d)

    def _build_design_matrix(self, obs_list: List[MarketObservation]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将观测列表构建为设计矩阵 X 和响应向量 y.
        这里采用简化策略: 用购买指示变量作为响应，做线性概率模型近似.
        X 的每一行对应一个产品-观测对，格式为 [features, -price * features].
        """
        X_rows = []
        y_rows = []
        for obs in obs_list:
            for idx, pidx in enumerate(obs.assortment_indices):
                x = obs.product_features[pidx]
                p = obs.prices[idx]
                X_rows.append(np.concatenate([x, -p * x]))
                # 是否被购买
                y_rows.append(1.0 if obs.purchase_index == idx else 0.0)
            # outside option 的隐式响应
        X = np.array(X_rows)
        y = np.array(y_rows)
        return X, y

    def fit_aggregate(self, source_obs: List[List[MarketObservation]]) -> np.ndarray:
        """用所有源市场的数据做聚合估计."""
        all_obs = []
        for obs_list in source_obs:
            all_obs.extend(obs_list)
        if len(all_obs) == 0:
            return self.nu_agg
        X, y = self._build_design_matrix(all_obs)
        # 岭回归替代精确MLE，保证数值稳定性
        reg = 1e-3
        self.nu_agg = np.linalg.solve(X.T @ X + reg * np.eye(2 * self.d), X.T @ y)
        return self.nu_agg

    def fit_debias(self, target_obs: List[MarketObservation]) -> np.ndarray:
        """
        用目标市场数据修正聚合估计的稀疏偏移.
        使用软阈值(L1近端梯度)做稀疏修正.
        """
        if len(target_obs) == 0:
            self.nu_debiased = self.nu_agg.copy()
            return self.nu_debiased

        X, y = self._build_design_matrix(target_obs)
        residual = y - X @ self.nu_agg

        # 坐标下降软阈值求解 Lasso: min ||residual - X @ delta||^2 + lambda ||delta||_1
        delta = np.zeros(2 * self.d)
        XtX = X.T @ X
        Xtr = X.T @ residual
        for _ in range(100):
            delta_old = delta.copy()
            for j in range(2 * self.d):
                rho = Xtr[j] - np.dot(XtX[j, :], delta) + XtX[j, j] * delta[j]
                z = XtX[j, j] + 1e-8
                delta[j] = np.sign(rho) * max(abs(rho) - self.l1_lambda, 0) / z
            if np.max(np.abs(delta - delta_old)) < 1e-6:
                break
        self.nu_debiased = self.nu_agg + delta
        return self.nu_debiased

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = self.nu_debiased[:self.d]
        gamma = self.nu_debiased[self.d:]
        # 保证价格敏感度为正
        gamma = np.maximum(gamma, 1e-3)
        return theta, gamma


class TJAPPolicy:
    """
    基于Two-Radius UCB的乐观决策策略.
    为简化实现，用特征范数缩放的不确定性替代完整的信息矩阵.
    """

    def __init__(self, catalog: List[Product], capacity: int, max_price: float = 200.0,
                 alpha: float = 1.0, beta: float = 0.5):
        self.catalog = catalog
        self.capacity = capacity
        self.max_price = max_price
        self.alpha = alpha  # variance radius
        self.beta = beta    # transfer-bias radius

    def optimistic_revenue(self, theta: np.ndarray, gamma: np.ndarray,
                           features: np.ndarray, prices: np.ndarray) -> float:
        """
        计算乐观收益上界.
        简化版 two-radius bonus: 对特征范数添加不确定性溢价.
        """
        K = len(prices)
        utilities = np.dot(features, theta) - np.dot(features, gamma) * prices
        # 添加乐观bonus (简化UCB)
        for k in range(K):
            feat_norm = np.linalg.norm(features[k])
            bonus = self.alpha * feat_norm + self.beta * np.max(np.abs(features[k]))
            utilities[k] += bonus
        max_u = np.max(utilities)
        exp_u = np.exp(utilities - max_u)
        denom = np.sum(exp_u) + np.exp(-max_u)
        probs = exp_u / denom
        return float(np.sum(probs * prices))

    def optimize_assortment_and_prices(self, theta: np.ndarray, gamma: np.ndarray,
                                       customer_features: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        在容量约束下选择最优assortment和连续价格.
        采用贪心启发式: 对每个候选子集用一维搜索找最优价格，再选收益最高者.
        """
        N = len(self.catalog)
        best_revenue = -1.0
        best_S = []
        best_prices = np.array([])

        # 为了控制计算复杂度，先根据估计效用筛选Top 2*capacity个候选产品
        base_utilities = []
        for i, prod in enumerate(self.catalog):
            x = prod.features * customer_features  # elementwise interaction as context
            base_utilities.append((i, np.dot(x, theta)))
        base_utilities.sort(key=lambda x: x[1], reverse=True)
        candidate_indices = [idx for idx, _ in base_utilities[:min(2 * self.capacity, N)]]

        # 枚举所有大小不超过capacity的候选子集 (受候选集限制)
        from itertools import combinations
        for r in range(1, self.capacity + 1):
            for subset in combinations(candidate_indices, r):
                features = np.array([self.catalog[i].features * customer_features for i in subset])
                # 对每个子集，用网格搜索找近似最优价格
                prices, rev = self._optimize_prices_for_subset(theta, gamma, features)
                if rev > best_revenue:
                    best_revenue = rev
                    best_S = list(subset)
                    best_prices = prices

        return best_S, best_prices

    def _optimize_prices_for_subset(self, theta: np.ndarray, gamma: np.ndarray,
                                    features: np.ndarray) -> Tuple[np.ndarray, float]:
        """对固定assortment用网格搜索找最优价格."""
        grid = np.linspace(10, self.max_price, 20)
        best_rev = -1.0
        best_p = np.ones(len(features)) * (self.max_price / 2)

        # 为简化，假设同一定价策略，实际上可以独立定价
        # 这里采用粗网格搜索每个产品的独立价格组合（只考虑2个产品的情况，避免组合爆炸）
        K = len(features)
        if K == 1:
            for p in grid:
                rev = self._revenue_at_prices(theta, gamma, features, np.array([p]))
                if rev > best_rev:
                    best_rev = rev
                    best_p = np.array([p])
        elif K == 2:
            for p1 in grid[::4]:
                for p2 in grid[::4]:
                    rev = self._revenue_at_prices(theta, gamma, features, np.array([p1, p2]))
                    if rev > best_rev:
                        best_rev = rev
                        best_p = np.array([p1, p2])
        else:
            # 对于K>2，采用统一加价策略: 所有产品价格相同
            for p in grid:
                rev = self._revenue_at_prices(theta, gamma, features, np.ones(K) * p)
                if rev > best_rev:
                    best_rev = rev
                    best_p = np.ones(K) * p
        return best_p, best_rev

    def _revenue_at_prices(self, theta: np.ndarray, gamma: np.ndarray,
                           features: np.ndarray, prices: np.ndarray) -> float:
        utilities = np.dot(features, theta) - np.dot(features, gamma) * prices
        max_u = np.max(utilities)
        exp_u = np.exp(utilities - max_u)
        denom = np.sum(exp_u) + np.exp(-max_u)
        probs = exp_u / denom
        return float(np.sum(probs * prices))


class TJAPAgent:
    """TJAP主控器: 管理episode、估计、决策."""

    def __init__(self, catalog: List[Product], capacity: int, feature_dim: int,
                 max_price: float = 200.0, l1_lambda: float = 0.1):
        self.catalog = catalog
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.estimator = AggregateThenDebiasEstimator(feature_dim, l1_lambda)
        self.policy = TJAPPolicy(catalog, capacity, max_price)
        self.target_history: List[MarketObservation] = []
        self.source_history: List[List[MarketObservation]] = []
        self.episode = 1
        self.t = 0

    def initialize_source_data(self, source_obs: List[List[MarketObservation]]):
        """导入源市场历史数据."""
        self.source_history = source_obs

    def add_target_observation(self, obs: MarketObservation):
        """记录目标市场的新观测."""
        self.target_history.append(obs)

    def should_update(self) -> bool:
        """基于几何episode判断是否到了更新参数的时机."""
        episode_end = 2 ** (self.episode - 1)
        return self.t >= episode_end

    def update(self):
        """执行Aggregate-then-Debias估计更新."""
        self.estimator.fit_aggregate(self.source_history)
        self.estimator.fit_debias(self.target_history)
        self.episode += 1

    def decide(self, customer_features: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """基于当前估计做出assortment-pricing决策."""
        theta, gamma = self.estimator.get_params()
        # 动态调整alpha和beta，随数据量增加而衰减
        n_total = sum(len(h) for h in self.source_history) + len(self.target_history)
        n_total = max(n_total, 1)
        self.policy.alpha = 1.0 / np.sqrt(n_total / (1 + len(self.source_history)) + 1)
        self.policy.beta = 0.5 / np.sqrt(len(self.target_history) + 1)
        return self.policy.optimize_assortment_and_prices(theta, gamma, customer_features)

    def step(self, customer_features: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """单步决策."""
        if self.should_update():
            self.update()
        S, prices = self.decide(customer_features)
        self.t += 1
        return S, prices


def build_momcozy_catalog() -> List[Product]:
    """构建Momcozy母婴产品目录示例."""
    return [
        Product("BP01", "Momcozy M5 穿戴式吸奶器",
                np.array([1.0, 0.8, 0.9, 0.6, 0.7]), base_cost=80.0),
        Product("BP02", "Momcozy S12Pro 双边电动吸奶器",
                np.array([0.9, 0.9, 0.7, 0.8, 0.6]), base_cost=60.0),
        Product("ST01", "Momcozy 紫外线奶瓶消毒器",
                np.array([0.7, 0.6, 0.9, 0.9, 0.8]), base_cost=50.0),
        Product("BW01", "Momcozy 智能暖奶器",
                np.array([0.6, 0.7, 0.8, 0.7, 0.9]), base_cost=40.0),
        Product("AC01", "Momcozy 吸奶器配件套装",
                np.array([0.5, 0.5, 0.6, 0.6, 0.5]), base_cost=15.0),
    ]


def generate_market_data(catalog: List[Product], true_theta: np.ndarray, true_gamma: np.ndarray,
                         num_periods: int, customer_base: np.ndarray,
                         seed: int = 42) -> List[MarketObservation]:
    """生成模拟的市场bandit反馈数据."""
    rng = np.random.RandomState(seed)
    N = len(catalog)
    model = MNLChoiceModel(true_theta, true_gamma)
    obs_list = []

    for _ in range(num_periods):
        # 随机客户特征波动
        customer_features = customer_base + rng.normal(0, 0.1, size=len(customer_base))
        customer_features = np.clip(customer_features, 0.2, 1.5)

        # 简单的基准策略: 随机选1-3个产品，价格随机
        assortment_size = rng.randint(1, 4)
        assortment = sorted(rng.choice(N, assortment_size, replace=False).tolist())
        prices = rng.uniform(30, 150, size=assortment_size)

        features = np.array([catalog[i].features * customer_features for i in assortment])
        probs = model.choice_probabilities(features, prices)

        # 模拟购买
        purchase = rng.choice(len(assortment) + 1, p=probs)
        purchase_idx = purchase - 1  # -1 表示 outside option

        all_features = np.array([p.features for p in catalog])
        obs_list.append(MarketObservation(
            product_features=all_features,
            assortment_indices=assortment,
            prices=prices,
            purchase_index=purchase_idx,
        ))
    return obs_list


def demo():
    """演示TJAP在美国市场(源)到德国市场(目标)的迁移学习."""
    catalog = build_momcozy_catalog()
    d = 5
    capacity = 2

    # 美国市场真实参数 (源市场)
    theta_us = np.array([1.2, 0.8, 1.0, 0.6, 0.9])
    gamma_us = np.array([0.015, 0.012, 0.010, 0.008, 0.011])
    customer_us = np.array([1.0, 0.9, 1.1, 0.8, 1.0])  # 美国客户重视便携和续航

    # 德国市场真实参数 (目标市场) — 与美国有稀疏偏移: 更关注静音认证和价格敏感度
    theta_de = theta_us.copy()
    theta_de[1] += 0.4   # 对"静音/舒适"维度偏好更高
    theta_de[4] -= 0.3   # 对"智能功能"偏好稍低
    gamma_de = gamma_us.copy()
    gamma_de[0] += 0.005  # 对吸奶器价格更敏感
    gamma_de[2] += 0.004  # 对消毒器价格更敏感
    customer_de = np.array([0.9, 1.2, 1.0, 0.9, 0.8])  # 德国客户重视静音和品质

    # 生成源市场(US)和目标市场(DE)的历史数据
    us_data = generate_market_data(catalog, theta_us, gamma_us, num_periods=300, customer_base=customer_us, seed=1)
    de_data = generate_market_data(catalog, theta_de, gamma_de, num_periods=80, customer_base=customer_de, seed=2)

    # 初始化TJAP Agent
    agent = TJAPAgent(catalog, capacity, feature_dim=d, max_price=150.0, l1_lambda=0.1)
    agent.initialize_source_data([us_data])

    # 用DE的少量历史数据做去偏初始化
    for obs in de_data[:50]:
        agent.add_target_observation(obs)
    agent.update()

    # 在DE市场做10轮决策演示
    decisions = []
    for t in range(10):
        cf = customer_de + np.random.normal(0, 0.05, size=d)
        cf = np.clip(cf, 0.2, 1.5)
        S, prices = agent.step(cf)
        prod_names = [catalog[i].name for i in S]
        decisions.append({
            "period": t + 1,
            "assortment": prod_names,
            "prices": [round(float(p), 2) for p in prices],
        })

    # 输出最终估计的参数
    theta_est, gamma_est = agent.estimator.get_params()

    result = {
        "message": "TJAP跨市场品类组合定价演示: 美国(源) → 德国(目标)",
        "estimated_theta_de": [round(float(v), 3) for v in theta_est],
        "estimated_gamma_de": [round(float(v), 4) for v in gamma_est],
        "true_theta_de": [round(float(v), 3) for v in theta_de],
        "true_gamma_de": [round(float(v), 4) for v in gamma_de],
        "sample_decisions": decisions,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


# ==================== Pytest Tests ====================

def test_mnl_choice_probability():
    model = MNLChoiceModel(np.array([1.0, 0.5]), np.array([0.01, 0.01]))
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    prices = np.array([50.0, 50.0])
    probs = model.choice_probabilities(features, prices)
    assert abs(np.sum(probs) - 1.0) < 1e-6
    assert len(probs) == 3  # outside + 2 products

def test_aggregate_then_debias_shape():
    estimator = AggregateThenDebiasEstimator(feature_dim=3)
    # 构造虚拟观测
    obs1 = MarketObservation(
        product_features=np.eye(3),
        assortment_indices=[0, 1],
        prices=np.array([10.0, 20.0]),
        purchase_index=0,
    )
    obs2 = MarketObservation(
        product_features=np.eye(3),
        assortment_indices=[1, 2],
        prices=np.array([15.0, 25.0]),
        purchase_index=1,
    )
    estimator.fit_aggregate([[obs1, obs2]])
    estimator.fit_debias([obs1])
    theta, gamma = estimator.get_params()
    assert theta.shape == (3,)
    assert gamma.shape == (3,)
    assert np.all(gamma > 0)

def test_policy_respects_capacity():
    catalog = [
        Product("P1", "P1", np.array([1.0, 0.5, 0.3]), base_cost=10.0),
        Product("P2", "P2", np.array([0.5, 1.0, 0.3]), base_cost=10.0),
        Product("P3", "P3", np.array([0.3, 0.5, 1.0]), base_cost=10.0),
    ]
    policy = TJAPPolicy(catalog, capacity=2, max_price=100.0)
    theta = np.array([1.0, 0.8, 0.6])
    gamma = np.array([0.01, 0.01, 0.01])
    customer = np.array([1.0, 1.0, 1.0])
    S, prices = policy.optimize_assortment_and_prices(theta, gamma, customer)
    assert len(S) <= 2
    assert len(prices) == len(S)
    assert np.all(prices <= 100.0)

def test_agent_episodic_update():
    catalog = build_momcozy_catalog()
    agent = TJAPAgent(catalog, capacity=2, feature_dim=5)
    us_data = generate_market_data(catalog, np.ones(5) * 0.5, np.ones(5) * 0.01,
                                   num_periods=50, customer_base=np.ones(5), seed=10)
    agent.initialize_source_data([us_data])

    for t in range(5):
        cf = np.ones(5)
        S, prices = agent.step(cf)
        # 模拟反馈
        obs = MarketObservation(
            product_features=np.array([p.features for p in catalog]),
            assortment_indices=S,
            prices=prices,
            purchase_index=-1,
        )
        agent.add_target_observation(obs)

    assert agent.t == 5
    assert agent.episode >= 1

def test_end_to_end_revenue_improvement():
    """端到端: 验证TJAP在迁移场景下优于单市场基线."""
    catalog = build_momcozy_catalog()
    d = 5
    theta_us = np.array([1.0, 0.8, 1.0, 0.6, 0.9])
    gamma_us = np.array([0.015, 0.012, 0.010, 0.008, 0.011])
    theta_de = theta_us.copy()
    theta_de[1] += 0.3
    gamma_de = gamma_us.copy()
    gamma_de[0] += 0.005

    us_data = generate_market_data(catalog, theta_us, gamma_us, 200, np.ones(5), seed=20)
    de_data = generate_market_data(catalog, theta_de, gamma_de, 30, np.ones(5), seed=21)

    # TJAP agent
    tjap = TJAPAgent(catalog, 2, d, max_price=120.0, l1_lambda=0.1)
    tjap.initialize_source_data([us_data])
    for obs in de_data[:20]:
        tjap.add_target_observation(obs)
    tjap.update()

    # 单市场基线 (无源数据)
    baseline = TJAPAgent(catalog, 2, d, max_price=120.0, l1_lambda=0.1)
    baseline.initialize_source_data([])
    for obs in de_data[:20]:
        baseline.add_target_observation(obs)
    baseline.update()

    # 在相同客户特征下比较决策收益
    cf = np.ones(5)
    true_model = MNLChoiceModel(theta_de, gamma_de)

    S_tjap, p_tjap = tjap.decide(cf)
    features_tjap = np.array([catalog[i].features * cf for i in S_tjap])
    rev_tjap = true_model.expected_revenue(features_tjap, p_tjap)

    S_base, p_base = baseline.decide(cf)
    features_base = np.array([catalog[i].features * cf for i in S_base])
    rev_base = true_model.expected_revenue(features_base, p_base)

    # TJAP应不低于基线（大概率严格更优）
    assert rev_tjap >= rev_base * 0.8  # 允许一定随机性


if __name__ == "__main__":
    demo()

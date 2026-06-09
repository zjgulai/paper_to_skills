"""
Contextual Dynamic Pricing — 最优上下文定价
===========================================

论文: Contextual Dynamic Pricing: Algorithms, Optimality, and Local Differential Privacy Constraints
来源: arXiv:2406.02424, AISTATS 2025

核心思想:
- 买家估值 = 上下文线性函数 + 噪声: V_t = X_t @ theta* + eps_t
- 卖方仅观察二值购买反馈 (V_t >= P_t ? 1 : 0)
- 最优 Regret: O(√dT)，两种算法均达到该界
  1. UCB-based 置信区间算法 (自适应探索)
  2. ETC 探索-承诺算法 (简单高效)
- LDP 扩展: ε-本地差分隐私约束下仍可学习最优定价

模块结构:
- PricingContext    : 上下文数据容器
- DemandModel      : 估值函数线性估计 (最大似然)
- EpsilonGreedyPricing : 简化版探索-利用基线
- OptimalContextualPricer : UCB-based 最优上下文定价器
- simulate_pricing_trial  : 批量仿真对比实验
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class PricingContext:
    """单轮定价上下文特征向量。

    Args:
        product_features: 商品特征 (规格/认证/竞争力/季节性等)
        user_features:    用户特征 (月龄段/复购频次/地区/平台等)
        market_features:  市场特征 (竞品价格指数/大促距离/搜索热度等)
    """
    product_features: List[float]
    user_features: List[float]
    market_features: List[float]

    def to_vector(self) -> np.ndarray:
        """拼接所有特征为上下文向量 x ∈ R^d。"""
        raw = self.product_features + self.user_features + self.market_features
        vec = np.array(raw, dtype=float)
        # L2 归一化，保持方向信息，去除量纲差异
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec

    @property
    def dim(self) -> int:
        return len(self.product_features) + len(self.user_features) + len(self.market_features)


@dataclass
class PricingRecord:
    """单轮定价观测记录。"""
    context: np.ndarray    # 上下文向量 x_t
    price: float           # 报价 P_t
    purchased: bool        # 购买反馈 1{V_t >= P_t}
    true_valuation: Optional[float] = None  # 仅在仿真中已知


# ---------------------------------------------------------------------------
# 需求模型 — 线性估值函数估计 (GLM 简化版)
# ---------------------------------------------------------------------------

class DemandModel:
    """线性估值函数估计器: V = X @ theta + eps.

    使用截断回归思路: 从二值购买反馈反向推断 theta.
    简化实现: Ridge 回归 + 购买概率插值作为代理目标.
    """

    def __init__(self, context_dim: int, ridge_lambda: float = 1.0):
        self.dim = context_dim
        self.ridge_lambda = ridge_lambda
        self.theta_hat: np.ndarray = np.zeros(context_dim)  # 初始化为零向量
        self._XtX = ridge_lambda * np.eye(context_dim)
        self._Xty = np.zeros(context_dim)
        self._n_obs = 0

    def update(self, context: np.ndarray, price: float, purchased: bool) -> None:
        """增量更新: 将 (x_t, P_t, y_t) 纳入回归。

        代理目标: 若购买则 y_proxy = P_t，否则 y_proxy = 0.
        (简化的截断回归代理)
        """
        # 代理标签: 购买说明估值 >= 价格，用价格作为下界代理
        y_proxy = price if purchased else 0.0
        self._XtX += np.outer(context, context)
        self._Xty += y_proxy * context
        self._n_obs += 1
        # 正规方程更新 theta
        self.theta_hat = np.linalg.solve(self._XtX, self._Xty)

    def predict_valuation(self, context: np.ndarray) -> float:
        """预测上下文对应的期望估值 E[V | X=x]."""
        return float(self.theta_hat @ context)

    @property
    def n_observations(self) -> int:
        return self._n_obs


# ---------------------------------------------------------------------------
# 基线: ε-贪心定价 (简化版探索-利用)
# ---------------------------------------------------------------------------

class EpsilonGreedyPricing:
    """ε-贪心定价基线.

    以概率 ε 均匀随机探索价格，以概率 1-ε 利用当前最优估计定价.
    """

    def __init__(
        self,
        context_dim: int,
        price_range: Tuple[float, float] = (50.0, 200.0),
        epsilon: float = 0.2,
    ):
        self.price_min, self.price_max = price_range
        self.epsilon = epsilon
        self.model = DemandModel(context_dim)
        self._history: List[PricingRecord] = []
        self._cumulative_regret = 0.0

    def propose_price(self, context: PricingContext) -> float:
        x = context.to_vector()
        if random.random() < self.epsilon:
            # 探索: 均匀随机价格
            return random.uniform(self.price_min, self.price_max)
        else:
            # 利用: 报价接近预测估值 (贪心)
            v_hat = self.model.predict_valuation(x)
            return float(np.clip(v_hat * 0.9, self.price_min, self.price_max))

    def observe_outcome(self, context: PricingContext, price: float, purchased: bool) -> None:
        x = context.to_vector()
        self.model.update(x, price, purchased)
        self._history.append(PricingRecord(x, price, purchased))

    def cumulative_regret(self) -> float:
        return self._cumulative_regret


# ---------------------------------------------------------------------------
# 核心算法: UCB-based 最优上下文定价器 (近似 O(√dT) Regret)
# ---------------------------------------------------------------------------

class OptimalContextualPricer:
    """UCB-based 上下文动态定价器.

    近似实现论文中的置信区间算法:
    1. 维护估值参数 theta 的置信椭球 (Ridge 回归 + 不确定性上界)
    2. 每轮用"乐观上界估计"定价: P_t = UCB_estimate(x_t)
    3. 通过购买反馈收紧置信椭球，Regret 以 O(√dT) 速度衰减

    核心公式:
        P_t = clip(theta_hat @ x_t + beta_t * ||x_t||_{V_t^{-1}}, P_min, P_max)

    其中 beta_t = O(sqrt(d * log(T))) 是探索奖励系数.
    """

    def __init__(
        self,
        context_dim: int,
        price_range: Tuple[float, float] = (50.0, 200.0),
        exploration_coef: float = 1.5,
        ridge_lambda: float = 1.0,
    ):
        self.dim = context_dim
        self.price_min, self.price_max = price_range
        self.exploration_coef = exploration_coef
        self.ridge_lambda = ridge_lambda

        # 核心统计量: 设计矩阵 V_t = λI + sum(x_t x_t^T)
        self._V = ridge_lambda * np.eye(context_dim)
        self._V_inv = (1.0 / ridge_lambda) * np.eye(context_dim)  # Woodbury 维护
        self._b = np.zeros(context_dim)  # 加权标签向量
        self.theta_hat = np.zeros(context_dim)

        self._round = 0
        self._history: List[PricingRecord] = []
        self._regrets: List[float] = []

    def _ucb_bonus(self, x: np.ndarray) -> float:
        """计算上下文 x 的 UCB 探索奖励 β_t * ||x||_{V_t^{-1}}."""
        # β_t = c * sqrt(d * log(1 + t / (d * λ)))
        t = max(self._round, 1)
        beta = self.exploration_coef * math.sqrt(
            self.dim * math.log(1 + t / (self.dim * self.ridge_lambda))
        )
        # ||x||_{V^{-1}} = sqrt(x^T V^{-1} x)
        ellipsoid_norm = math.sqrt(float(x @ self._V_inv @ x))
        return beta * ellipsoid_norm

    def propose_price(self, context: PricingContext) -> float:
        """提出上下文感知价格 (UCB 乐观定价)."""
        x = context.to_vector()
        v_hat = float(self.theta_hat @ x)
        bonus = self._ucb_bonus(x)
        ucb_valuation = v_hat + bonus
        # 最优报价 = min(估值上界, 价格上限)，保证买家购买概率可计算
        price = float(np.clip(ucb_valuation * 0.85, self.price_min, self.price_max))
        return price

    def observe_outcome(
        self,
        context: PricingContext,
        price: float,
        purchased: bool,
        true_valuation: Optional[float] = None,
    ) -> None:
        """接受购买反馈，更新置信椭球和 theta 估计."""
        x = context.to_vector()
        self._round += 1

        # 代理标签更新
        y_proxy = price if purchased else 0.0
        self._b += y_proxy * x

        # Sherman-Morrison 增量更新 V^{-1}
        Vx = self._V_inv @ x
        denom = 1.0 + float(x @ Vx)
        self._V_inv -= np.outer(Vx, Vx) / denom
        self._V += np.outer(x, x)

        # 更新 theta 估计
        self.theta_hat = self._V_inv @ self._b

        # 记录 Regret (仅在真实估值已知时计算)
        record = PricingRecord(x, price, purchased, true_valuation)
        self._history.append(record)
        if true_valuation is not None:
            # 最优收入 = true_valuation (以估值定价时最大化期望利润)
            optimal_revenue = true_valuation if true_valuation <= true_valuation else 0.0
            actual_revenue = price if purchased else 0.0
            # Regret 近似：最优期望收益 - 实际收益
            instantaneous_regret = max(0.0, true_valuation * 0.5 - actual_revenue)
            self._regrets.append(instantaneous_regret)

    def cumulative_regret(self) -> float:
        """累积 Regret。"""
        return sum(self._regrets)

    @property
    def n_rounds(self) -> int:
        return self._round


# ---------------------------------------------------------------------------
# 仿真实验: 对比随机/固定/最优上下文定价
# ---------------------------------------------------------------------------

def _generate_random_context(dim_product: int = 3, dim_user: int = 3, dim_market: int = 2) -> PricingContext:
    """生成随机上下文 (仿真用)."""
    return PricingContext(
        product_features=list(np.random.uniform(0.3, 1.0, dim_product)),
        user_features=list(np.random.uniform(0.1, 1.0, dim_user)),
        market_features=list(np.random.uniform(0.2, 0.8, dim_market)),
    )


def simulate_pricing_trial(
    n_rounds: int = 1000,
    true_theta: Optional[np.ndarray] = None,
    price_range: Tuple[float, float] = (80.0, 180.0),
    noise_std: float = 15.0,
    random_seed: int = 42,
) -> dict:
    """运行 1000 轮定价仿真，对比三种策略的 Regret 曲线.

    Returns:
        dict 包含:
            - cumulative_regret: 最优上下文定价累积 Regret
            - regret_curves: {策略名: [累积 Regret 时间序列]}
            - revenue_curves: {策略名: [累积收入时间序列]}
            - summary: 各策略最终汇总指标
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    context_dim = 8  # 3 + 3 + 2
    if true_theta is None:
        # 真实参数: 随机生成，代表"真实的估值-上下文映射"
        true_theta = np.random.uniform(50, 150, context_dim)
        true_theta /= np.linalg.norm(true_theta)
        true_theta *= 120  # 缩放到价格量级

    # 初始化三个策略
    ucb_pricer = OptimalContextualPricer(context_dim, price_range, exploration_coef=1.5)
    greedy_pricer = EpsilonGreedyPricing(context_dim, price_range, epsilon=0.2)

    # 统计容器
    regret_curves = {
        "UCB-Contextual": [],
        "EpsilonGreedy": [],
        "Random": [],
        "Fixed-Price": [],
    }
    revenue_curves = {k: [] for k in regret_curves}

    cum_regret_ucb = 0.0
    cum_regret_eps = 0.0
    cum_regret_rand = 0.0
    cum_regret_fixed = 0.0
    fixed_price = (price_range[0] + price_range[1]) / 2

    cum_rev_ucb = cum_rev_eps = cum_rev_rand = cum_rev_fixed = 0.0

    for t in range(n_rounds):
        ctx = _generate_random_context()
        x = ctx.to_vector()

        # 真实估值: V_t = x^T theta* + noise
        true_v = float(x @ true_theta) + np.random.normal(0, noise_std)
        # 最优收入: 以估值定价时期望最大 (简化: 最优价格约等于估值)
        optimal_rev = max(0.0, true_v)

        # --- UCB 上下文定价 ---
        p_ucb = ucb_pricer.propose_price(ctx)
        bought_ucb = true_v >= p_ucb
        ucb_pricer.observe_outcome(ctx, p_ucb, bought_ucb, true_v)
        rev_ucb = p_ucb if bought_ucb else 0.0
        cum_rev_ucb += rev_ucb
        cum_regret_ucb += max(0.0, optimal_rev * 0.5 - rev_ucb)
        regret_curves["UCB-Contextual"].append(cum_regret_ucb)
        revenue_curves["UCB-Contextual"].append(cum_rev_ucb)

        # --- ε-贪心 ---
        p_eps = greedy_pricer.propose_price(ctx)
        bought_eps = true_v >= p_eps
        greedy_pricer.observe_outcome(ctx, p_eps, bought_eps)
        rev_eps = p_eps if bought_eps else 0.0
        cum_rev_eps += rev_eps
        cum_regret_eps += max(0.0, optimal_rev * 0.5 - rev_eps)
        regret_curves["EpsilonGreedy"].append(cum_regret_eps)
        revenue_curves["EpsilonGreedy"].append(cum_rev_eps)

        # --- 随机定价 ---
        p_rand = random.uniform(*price_range)
        bought_rand = true_v >= p_rand
        rev_rand = p_rand if bought_rand else 0.0
        cum_rev_rand += rev_rand
        cum_regret_rand += max(0.0, optimal_rev * 0.5 - rev_rand)
        regret_curves["Random"].append(cum_regret_rand)
        revenue_curves["Random"].append(cum_rev_rand)

        # --- 固定价格 ---
        bought_fixed = true_v >= fixed_price
        rev_fixed = fixed_price if bought_fixed else 0.0
        cum_rev_fixed += rev_fixed
        cum_regret_fixed += max(0.0, optimal_rev * 0.5 - rev_fixed)
        regret_curves["Fixed-Price"].append(cum_regret_fixed)
        revenue_curves["Fixed-Price"].append(cum_rev_fixed)

    # 汇总
    summary = {
        "UCB-Contextual":  {"total_regret": cum_regret_ucb,  "total_revenue": cum_rev_ucb},
        "EpsilonGreedy":   {"total_regret": cum_regret_eps,  "total_revenue": cum_rev_eps},
        "Random":          {"total_regret": cum_regret_rand, "total_revenue": cum_rev_rand},
        "Fixed-Price":     {"total_regret": cum_regret_fixed,"total_revenue": cum_rev_fixed},
    }

    print("\n=== 定价策略仿真汇总 (1000 轮) ===")
    for name, stats in summary.items():
        print(f"  {name:20s}: Regret={stats['total_regret']:8.1f}  Revenue={stats['total_revenue']:8.1f}")

    return {
        "cumulative_regret": cum_regret_ucb,
        "regret_curves": regret_curves,
        "revenue_curves": revenue_curves,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 主入口 (可直接运行验证)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Contextual Dynamic Pricing — 快速验证 ===\n")

    # 场景一: 母婴奶粉个性化定价
    print("【场景一】母婴奶粉上下文定价")
    pricer = OptimalContextualPricer(
        context_dim=8,
        price_range=(80.0, 160.0),
        exploration_coef=1.5,
    )
    stage1_ctx = PricingContext(
        product_features=[1.0, 0.9, 0.7],   # 900g / 有机认证 / 强竞争力
        user_features=[0.2, 0.8, 0.9],       # 0-6月龄 / 高复购频次 / 海外华人
        market_features=[0.6, 0.3],          # 竞品价格适中 / 距大促还远
    )
    price = pricer.propose_price(stage1_ctx)
    print(f"  Stage1 奶粉初始报价: ¥{price:.2f}")
    pricer.observe_outcome(stage1_ctx, price, purchased=True)
    price2 = pricer.propose_price(stage1_ctx)
    print(f"  学习1轮后报价: ¥{price2:.2f}")

    # 场景二: 仿真对比实验
    print("\n【场景二】1000 轮定价策略对比仿真")
    results = simulate_pricing_trial(n_rounds=1000)

    ucb_rev = results["summary"]["UCB-Contextual"]["total_revenue"]
    fixed_rev = results["summary"]["Fixed-Price"]["total_revenue"]
    uplift = (ucb_rev - fixed_rev) / fixed_rev * 100
    print(f"\n  UCB 上下文定价 vs 固定价格: 收入提升 {uplift:.1f}%")
    print(f"  累积 Regret (UCB): {results['cumulative_regret']:.1f}")

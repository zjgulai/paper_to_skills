"""
Thompson Sampling for Multi-Armed Bandit
母婴出海场景：Banner投放优化、广告渠道选择

基于论文：Russo et al. (2018) "A Tutorial on Thompson Sampling"
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class BernoulliThompsonSampling:
    """
    Thompson Sampling for Bernoulli Bandit
    适用于：点击率优化、转化率优化等二元结果场景
    """

    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        初始化Thompson Sampling

        Args:
            n_arms: 动作/臂的数量（如Banner数量、广告渠道数）
            prior_alpha: Beta先验的alpha参数（默认1.0即均匀先验）
            prior_beta: Beta先验的beta参数（默认1.0即均匀先验）
        """
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * prior_alpha
        self.beta = np.ones(n_arms) * prior_beta
        self.n_pulls = np.zeros(n_arms)
        self.successes = np.zeros(n_arms)

    def select_action(self) -> int:
        """
        选择下一个动作
        从每个臂的后验分布采样，选择样本值最大的臂

        Returns:
            选择的动作索引
        """
        # 从Beta分布采样
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float):
        """
        更新后验分布

        Args:
            action: 执行的动作索引
            reward: 奖励（0或1，或0-1之间的值）
        """
        self.n_pulls[action] += 1

        # 对于二元奖励
        if reward > 0:
            self.alpha[action] += 1
            self.successes[action] += 1
        else:
            self.beta[action] += 1

    def get_estimated_rates(self) -> np.ndarray:
        """获取各臂的估计奖励率（后验均值）"""
        return self.alpha / (self.alpha + self.beta)

    def get_uncertainty(self) -> np.ndarray:
        """获取各臂的不确定性（后验标准差）"""
        mean = self.get_estimated_rates()
        var = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        return np.sqrt(var)

    def recommend_budget_allocation(self) -> np.ndarray:
        """
        推荐预算分配比例
        基于各臂是最优臂的概率进行软分配
        """
        # 通过多次采样估计每个臂是最优的概率
        n_samples = 10000
        optimal_counts = np.zeros(self.n_arms)

        for _ in range(n_samples):
            samples = np.random.beta(self.alpha, self.beta)
            optimal_counts[np.argmax(samples)] += 1

        return optimal_counts / n_samples


class ContextualThompsonSampling:
    """
    上下文相关的Thompson Sampling（线性模型版本）
    适用于：用户分群、时段差异化等场景
    """

    def __init__(self, n_arms: int, n_features: int, v: float = 1.0):
        """
        初始化上下文Thompson Sampling

        Args:
            n_arms: 动作数量
            n_features: 上下文特征维度
            v: 探索参数，越大探索越多
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.v = v

        # 为每个臂维护一个线性模型
        self.B = [np.eye(n_features) for _ in range(n_arms)]  # 精度矩阵
        self.mu = [np.zeros(n_features) for _ in range(n_arms)]  # 参数均值
        self.f = [np.zeros(n_features) for _ in range(n_arms)]  # 特征累加

    def select_action(self, context: np.ndarray) -> int:
        """
        基于上下文选择动作

        Args:
            context: 上下文特征向量 (n_features,)

        Returns:
            选择的动作索引
        """
        samples = []
        for arm in range(self.n_arms):
            # 从后验分布采样参数
            mu_hat = np.random.multivariate_normal(
                self.mu[arm],
                self.v**2 * np.linalg.inv(self.B[arm])
            )
            # 计算期望奖励
            expected_reward = np.dot(mu_hat, context)
            samples.append(expected_reward)

        return int(np.argmax(samples))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        更新模型参数

        Args:
            arm: 执行的动作
            context: 上下文特征
            reward: 观察到的奖励
        """
        self.B[arm] += np.outer(context, context)
        self.f[arm] += reward * context
        self.mu[arm] = np.linalg.inv(self.B[arm]).dot(self.f[arm])


def simulate_banner_optimization():
    """
    模拟母婴APP首页Banner优化场景
    """
    # 真实的CTR（实际中未知）
    true_ctrs = np.array([0.03, 0.05, 0.04, 0.02, 0.06])  # 5个Banner的真实点击率
    n_arms = len(true_ctrs)
    n_rounds = 5000

    # 初始化算法
    ts = BernoulliThompsonSampling(n_arms)

    # 记录结果
    rewards = []
    actions = []
    cumulative_reward = 0
    cumulative_rewards = []

    for t in range(n_rounds):
        # 选择Banner
        action = ts.select_action()
        actions.append(action)

        # 模拟用户反馈（点击或不点击）
        reward = np.random.binomial(1, true_ctrs[action])
        rewards.append(reward)

        # 更新模型
        ts.update(action, reward)

        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)

    # 分析结果
    print("=" * 50)
    print("Banner优化结果")
    print("=" * 50)
    print(f"总展示次数: {n_rounds}")
    print(f"总点击次数: {cumulative_reward}")
    print(f"实际CTR: {cumulative_reward / n_rounds:.4f}")
    print(f"理论最优CTR: {max(true_ctrs):.4f}")
    print(f"遗憾比例: {(max(true_ctrs) * n_rounds - cumulative_reward) / (max(true_ctrs) * n_rounds):.2%}")

    print("\n各Banner表现:")
    for i in range(n_arms):
        estimated_ctr = ts.alpha[i] / (ts.alpha[i] + ts.beta[i])
        uncertainty = np.sqrt((ts.alpha[i] * ts.beta[i]) /
                            ((ts.alpha[i] + ts.beta[i])**2 * (ts.alpha[i] + ts.beta[i] + 1)))
        print(f"  Banner {i}: 真实CTR={true_ctrs[i]:.3f}, 估计CTR={estimated_ctr:.3f}±{uncertainty:.3f}, "
              f"展示次数={int(ts.n_pulls[i])}")

    print("\n预算分配建议:")
    allocation = ts.recommend_budget_allocation()
    for i in range(n_arms):
        print(f"  Banner {i}: {allocation[i]:.1%}")

    return ts, rewards, actions, cumulative_rewards


def compare_with_random():
    """
    对比Thompson Sampling与随机选择
    """
    true_ctrs = np.array([0.03, 0.05, 0.04, 0.02, 0.06])
    n_arms = len(true_ctrs)
    n_rounds = 5000
    n_experiments = 100

    ts_rewards = []
    random_rewards = []

    for exp in range(n_experiments):
        # Thompson Sampling
        ts = BernoulliThompsonSampling(n_arms)
        ts_total = 0

        # Random
        random_total = 0

        for t in range(n_rounds):
            # TS
            ts_action = ts.select_action()
            ts_reward = np.random.binomial(1, true_ctrs[ts_action])
            ts.update(ts_action, ts_reward)
            ts_total += ts_reward

            # Random
            random_action = np.random.randint(n_arms)
            random_reward = np.random.binomial(1, true_ctrs[random_action])
            random_total += random_reward

        ts_rewards.append(ts_total)
        random_rewards.append(random_total)

    print("\n" + "=" * 50)
    print("Thompson Sampling vs 随机选择 (100次实验平均)")
    print("=" * 50)
    print(f"Thompson Sampling平均点击: {np.mean(ts_rewards):.1f}")
    print(f"随机选择平均点击: {np.mean(random_rewards):.1f}")
    print(f"提升幅度: {(np.mean(ts_rewards) - np.mean(random_rewards)) / np.mean(random_rewards):.1%}")
    print(f"理论最优点击: {max(true_ctrs) * n_rounds:.1f}")


if __name__ == "__main__":
    # 运行Banner优化模拟
    ts, rewards, actions, cumulative = simulate_banner_optimization()

    # 对比实验
    compare_with_random()

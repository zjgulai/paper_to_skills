"""
Multi-Armed Bandit Implementation
用于母婴出海电商广告素材优化
"""

import numpy as np
import random
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class MultiArmedBandit:
    """多臂老虎机算法基类"""

    def __init__(self, n_arms: int, arm_names: List[str] = None):
        self.n_arms = n_arms
        self.arm_names = arm_names or [f"arm_{i}" for i in range(n_arms)]
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n

    def get_distribution(self) -> Dict:
        total = sum(self.counts)
        return {
            name: {
                'count': self.counts[i],
                'mean_value': self.values[i],
                'weight': self.counts[i] / total if total > 0 else 0
            }
            for i, name in enumerate(self.arm_names)
        }


class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, n_arms: int, epsilon: float = 0.1, arm_names: List[str] = None):
        super().__init__(n_arms, arm_names)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        return int(np.argmax(self.values))


class UCB(MultiArmedBandit):
    def __init__(self, n_arms: int, c: float = 2.0, arm_names: List[str] = None):
        super().__init__(n_arms, arm_names)
        self.c = c
        self.total_counts = 0

    def select_arm(self) -> int:
        self.total_counts += 1
        if self.total_counts <= self.n_arms:
            return self.total_counts - 1
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_counts) / self.counts
        )
        return int(np.argmax(ucb_values))


class ThompsonSampling(MultiArmedBandit):
    """Thompson Sampling - 推荐使用"""

    def __init__(self, n_arms: int, arm_names: List[str] = None):
        super().__init__(n_arms, arm_names)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


def simulate_ad_experiment(bandit, n_rounds=1000, true_ctrs=None):
    """模拟广告实验"""
    if true_ctrs is None:
        true_ctrs = [0.05, 0.08, 0.12, 0.10]

    total_rewards = []

    for t in range(n_rounds):
        arm = bandit.select_arm()
        reward = 1 if random.random() < true_ctrs[arm] else 0
        bandit.update(arm, reward)
        total_rewards.append(sum(total_rewards[-1:]) + reward if total_rewards else reward)

    return {
        'total_rewards': total_rewards,
        'arm_selections': bandit.counts,
        'final_distribution': bandit.get_distribution(),
        'total_reward': sum(total_rewards),
        'optimal_rate': sum(total_rewards) / n_rounds
    }


def main():
    print("=" * 60)
    print("Multi-Armed Bandit 测试")
    print("=" * 60)

    true_ctrs = [0.05, 0.08, 0.12, 0.10]

    print("\n[1] 初始化 Thompson Sampling...")
    bandit = ThompsonSampling(
        n_arms=4,
        arm_names=['ad_creative_A', 'ad_creative_B', 'ad_creative_C', 'ad_creative_D']
    )

    print("\n[2] 运行模拟实验...")
    n_rounds = 1000
    result = simulate_ad_experiment(bandit, n_rounds, true_ctrs)

    print(f"\n[3] 实验结果:")
    print(f"   总轮数: {n_rounds}")
    print(f"   总点击: {result['total_reward']}")
    print(f"   平均 CTR: {result['optimal_rate']*100:.2f}%")

    print("\n[4] 流量分配:")
    dist = result['final_distribution']
    for name, info in dist.items():
        print(f"   {name}: {info['count']} ({info['weight']*100:.1f}%)")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return bandit


if __name__ == '__main__':
    bandit = main()

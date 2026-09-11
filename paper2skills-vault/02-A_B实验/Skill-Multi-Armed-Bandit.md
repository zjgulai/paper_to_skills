# Skill Card: Multi-Armed Bandit (多臂老虎机)

---

## ① 算法原理

### 核心思想
多臂老虎机（Multi-Armed Bandit, MAB）解决的核心问题是：**在探索（exploration）和利用（exploitation）之间取得平衡**。与传统 A/B 测试不同，MAB 在测试过程中动态调整流量分配，将更多流量分配给表现好的广告版本，同时继续探索其他版本，从而在保证学习效果的同时最大化收益。

### 数学直觉

**问题定义**：
- K 个臂（广告版本）$a_1, a_2, ..., a_K$
- 每个臂的奖励分布 $R_i$，期望 $\mu_i$
- 目标是最大化累计奖励 $\sum_{t=1}^{T} r_t$

**UCB (Upper Confidence Bound)** 算法：
$$UCB_i = \bar{X}_i + \sqrt{\frac{2 \ln t}{n_i}}$$

- $\bar{X}_i$ 是臂 i 的平均奖励
- $n_i$ 是臂 i 被选择的次数
- $t$ 是总选择次数
- 第二项是置信上界，反映不确定性

**Thompson Sampling**（推荐使用）：
基于贝叶斯后验采样：

1. 对每个臂维护 Beta 分布参数 $(\alpha_i, \beta_i)$
   - $\alpha_i$ = 成功次数 + 1
   - $\beta_i$ = 失败次数 + 1
2. 每轮从每个臂的 Beta 分布中采样 $\theta_i \sim Beta(\alpha_i, \beta_i)$
3. 选择采样值最大的臂：$a_t = \arg\max_i \theta_i$
4. 观察奖励后更新参数

### 关键假设
- **独立同分布奖励**：每次选择获得的奖励与历史无关
- **平稳环境**：各臂的奖励分布不随时间变化（非平稳环境需用变体）
- **二值奖励**：通常用于点击/转化场景（可扩展到连续奖励）

---

## ② 吸奶器出海应用案例

### 场景一：吸奶器Facebook/Instagram广告素材AB测试优化

**业务问题**：
我们在北美/澳洲投放吸奶器广告时，通常会准备多套广告素材（产品图片、使用场景图、妈妈晒单图、不同文案）。传统方法是固定流量分配 A/B 测试（如 50/50），但：
- 效果差的素材浪费 50% 预算
- 测试周期长（至少 1-2 周）
- 不确定性强，难以决策

使用 MAB 可以动态分配流量，自动淘汰差的素材，放大好的素材。

**数据要求**：
- 广告素材 ID（如 "素材A-哺乳妈妈使用图"、"素材B-产品正面图"）
- 每条广告的曝光、点击、加购、购买数据
- 实时或 near-real-time 数据回流

**预期产出**：
- 每个素材的实时流量权重（自动调整）
- 置信区间和胜出概率
- 自动停投低效素材

**业务价值**：
- 吸奶器客单价 $80-150，广告预算月均 30 万
- 广告预算节省 20-40%（减少低效素材消耗）
- 测试周期缩短 50%+（动态调整代替固定测试）
- 转化率提升 10-20%（流量向高效素材倾斜）
- 预计每月节省 6-12 万广告费

---

### 场景二：吸奶器TikTok/Instagram短视频出价策略优化

**业务问题**：
我们在 TikTok/Instagram Reels 投放吸奶器短视频广告时，需要找到最优出价策略。手动调价效率低，且容易过度优化。使用 MAB 可以自动探索最优出价区间。

**数据要求**：
- 出价区间离散化（如 $0.5, $1.0, $1.5, $2.0, $2.5）
- 每个出价的转化数据
- 国家/地区维度（美国/加拿大/英国/澳洲）

**预期产出**：
- 每个出价的最优概率分布
- 实时调整出价建议
- ROI 预测

**业务价值**：
- 出价优化时间减少 80%+
- 单次转化成本降低 10-15%
- 跨国投放效率提升

---

## ③ 代码模板

```python
"""
Multi-Armed Bandit Implementation
用于母婴出海电商广告素材优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
import warnings
warnings.filterwarnings('ignore')


class MultiArmedBandit:
    """多臂老虎机算法基类"""

    def __init__(self, n_arms: int, arm_names: List[str] = None):
        """
        初始化

        Args:
            n_arms: 臂的数量
            arm_names: 臂的名称列表
        """
        self.n_arms = n_arms
        self.arm_names = arm_names or [f"arm_{i}" for i in range(n_arms)]
        self.counts = np.zeros(n_arms)  # 每个臂的选择次数
        self.values = np.zeros(n_arms)  # 每个臂的平均奖励

    def select_arm(self) -> int:
        """选择臂"""
        raise NotImplementedError

    def update(self, arm: int, reward: float):
        """更新臂的参数"""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # 增量更新
        self.values[arm] = value + (reward - value) / n

    def get_distribution(self) -> Dict:
        """获取当前分布"""
        return {
            name: {
                'count': self.counts[i],
                'mean_value': self.values[i],
                'weight': self.counts[i] / sum(self.counts) if sum(self.counts) > 0 else 0
            }
            for i, name in enumerate(self.arm_names)
        }


class EpsilonGreedy(MultiArmedBandit):
    """Epsilon-Greedy 算法"""

    def __init__(self, n_arms: int, epsilon: float = 0.1, arm_names: List[str] = None):
        super().__init__(n_arms, arm_names)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return int(np.argmax(self.values))


class UCB(MultiArmedBandit):
    """UCB (Upper Confidence Bound) 算法"""

    def __init__(self, n_arms: int, c: float = 2.0, arm_names: List[str] = None):
        super().__init__(n_arms, arm_names)
        self.c = c
        self.total_counts = 0

    def select_arm(self) -> int:
        self.total_counts += 1

        # 确保每个臂至少被选择一次
        if self.total_counts <= self.n_arms:
            return self.total_counts - 1

        # 计算 UCB
        ucb_values = self.values + self.c * np.sqrt(
            np.log(self.total_counts) / self.counts
        )
        return int(np.argmax(ucb_values))


class ThompsonSampling(MultiArmedBandit):
    """Thompson Sampling 算法（推荐使用）"""

    def __init__(self, n_arms: int, arm_names: List[str] = None):
        super().__init__(n_arms, arm_names)
        # Beta 分布参数：(成功次数+1, 失败次数+1)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self) -> int:
        # 从每个臂的 Beta 分布中采样
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """更新 Beta 分布参数"""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]

        # 更新平均奖励
        self.values[arm] = value + (reward - value) / n

        # 更新 Beta 参数（二值奖励）
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


class ContextualBandit(MultiArmedBandit):
    """上下文老虎机（简化版）"""

    def __init__(self, n_arms: int, arm_names: List[str] = None):
        super().__init__(n_arms, arm_names)
        self.models = [LogisticRegression() for _ in range(n_arms)]
        self.is_fitted = [False] * n_arms

    def select_arm(self, context: np.ndarray) -> int:
        """基于上下文选择臂"""
        # 如果模型未训练，随机选择
        if not any(self.is_fitted):
            return random.randint(0, self.n_arms - 1)

        # 预测每个臂的期望奖励
        predicted_values = []
        for i, (model, fitted) in enumerate(zip(self.models, self.is_fitted)):
            if fitted:
                pred = model.predict_proba(context.reshape(1, -1))[0, 1]
                predicted_values.append(pred)
            else:
                predicted_values.append(0.5)  # 默认值

        return int(np.argmax(predicted_values))

    def update(self, arm: int, reward: float, context: np.ndarray):
        """更新模型"""
        # 简化：使用增量更新
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n

        self.is_fitted[arm] = True


# ==================== 模拟实验 ====================

def simulate_ad_experiment(bandit, n_rounds=1000, true_ctrs=None):
    """
    模拟广告实验

    Args:
        bandit: 多臂老虎机算法
        n_rounds: 实验轮数
        true_ctrs: 每个臂的真实 CTR

    Returns:
        results: 实验结果
    """
    if true_ctrs is None:
        true_ctrs = [0.05, 0.08, 0.12, 0.10]  # 假设 4 个广告素材的真实 CTR

    n_arms = len(true_ctrs)
    total_rewards = []
    arm_selections = []

    for t in range(n_rounds):
        # 选择臂
        arm = bandit.select_arm()

        # 模拟点击（伯努利分布）
        reward = 1 if random.random() < true_ctrs[arm] else 0

        # 更新
        bandit.update(arm, reward)

        # 记录
        total_rewards.append(sum(total_rewards) + reward)
        arm_selections.append(arm)

    return {
        'total_rewards': total_rewards,
        'arm_selections': arm_selections,
        'final_distribution': bandit.get_distribution(),
        'total_reward': sum(total_rewards),
        'optimal_rate': sum(total_rewards) / n_rounds
    }


def compare_algorithms():
    """比较不同算法"""
    print("=" * 60)
    print("Multi-Armed Bandit 算法对比")
    print("=" * 60)

    n_rounds = 5000
    true_ctrs = [0.05, 0.08, 0.12, 10.10]

    algorithms = [
        ("Epsilon-Greedy (ε=0.1)", EpsilonGreedy(4, epsilon=0.1)),
        ("UCB (c=2)", UCB(4, c=2)),
        ("Thompson Sampling", ThompsonSampling(4))
    ]

    results = []

    for name, bandit in algorithms:
        result = simulate_ad_experiment(bandit, n_rounds, true_ctrs)
        results.append((name, result))
        print(f"\n[{name}]")
        print(f"   总奖励: {result['total_reward']}")
        print(f"   平均 CTR: {result['optimal_rate']:.4f}")
        print(f"   分布: {result['final_distribution']}")


def main():
    """主函数"""
    print("=" * 60)
    print("Multi-Armed Bandit 测试")
    print("=" * 60)

    # 模拟 4 个广告素材的真实 CTR
    # 素材1: 5%, 素材2: 8%, 素材3: 12%, 素材4: 10%
    true_ctrs = [0.05, 0.08, 0.12, 0.10]

    # 使用 Thompson Sampling
    print("\n[1] 初始化 Thompson Sampling...")
    bandit = ThompsonSampling(
        n_arms=4,
        arm_names=['ad_creative_A', 'ad_creative_B', 'ad_creative_C', 'ad_creative_D']
    )

    # 模拟实验
    print("\n[2] 运行模拟实验...")
    n_rounds = 1000
    total_rewards = []
    arm_selections = []

    for t in range(n_rounds):
        # 选择臂
        arm = bandit.select_arm()

        # 模拟点击
        reward = 1 if random.random() < true_ctrs[arm] else 0

        # 更新
        bandit.update(arm, reward)

        # 记录
        total_rewards.append(sum(total_rewards[-1:]) + reward if total_rewards else reward)
        arm_selections.append(arm)

    # 结果
    print("\n[3] 实验结果...")
    print(f"   总轮数: {n_rounds}")
    print(f"   总点击: {sum(total_rewards)}")
    print(f"   平均 CTR: {sum(total_rewards)/n_rounds*100:.2f}%")

    print("\n[4] 流量分配:")
    dist = bandit.get_distribution()
    for name, info in dist.items():
        print(f"   {name}:")
        print(f"     - 选择次数: {info['count']} ({info['count']/n_rounds*100:.1f}%)")
        print(f"     - 平均 CTR: {info['mean_value']*100:.2f}%")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return bandit


if __name__ == '__main__':
    bandit = main()

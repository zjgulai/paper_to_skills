# Skill Card: Thompson Sampling for Multi-Armed Bandit

---

## ① 算法原理

### 核心思想
Thompson Sampling是一种**基于贝叶斯后验采样的在线决策算法**，用于解决探索-利用权衡（Exploration-Exploitation Tradeoff）问题。算法的核心洞见是：**按照每个动作是最优动作的概率来选择动作**，而非简单地选择当前估计奖励最高的动作。

### 数学直觉
对于Bernoulli Bandit问题（每个动作产生成功/失败结果）：
- **先验分布**：对每个动作的奖励概率θₖ使用Beta分布 Beta(αₖ, βₖ) 建模
- **后验更新**：观察到奖励rₜ后，更新参数
  - 成功时：αₖ ← αₖ + 1
  - 失败时：βₖ ← βₖ + 1
- **采样决策**：每轮从每个动作的后验分布中采样一个估计值 θ̂ₖ ~ Beta(αₖ, βₖ)，选择最大θ̂ₖ对应的动作

**直观理解**：Beta分布的均值代表当前估计的奖励率，方差代表不确定性。Thompson Sampling天然地让不确定性高的动作有更多机会被探索（因其分布更宽，采样出高值的概率更大），同时逐渐聚焦于高奖励动作。

### 关键假设
1. **动作奖励平稳**：各动作的真实奖励率θₖ不随时间变化
2. **反馈即时**：每次动作后立即获得奖励反馈
3. **动作独立**：各动作之间没有相关性（可通过扩展算法放松）
4. **计算可行**：后验分布需要可采样（Beta分布易于采样）

---

## ② 母婴出海应用案例

### 场景1：跨境电商首页Banner智能投放

**业务问题**
母婴出海APP首页有5个Banner位，每个位置可展示不同内容（新品推广、促销活动、育儿知识、用户UGC、跨境物流优势）。运营团队不知道哪种内容组合能带来最高的点击率（CTR）和转化率，传统的A/B测试需要大量流量且无法自适应变化。

**数据要求**
| 字段 | 说明 | 格式 |
|------|------|------|
| timestamp | 展示时间 | datetime |
| banner_id | Banner编号（1-5） | int |
| content_type | 内容类型 | categorical |
| clicked | 是否点击（0/1） | binary |
| converted | 是否转化（0/1） | binary |
| user_segment | 用户分群（新手妈妈/二胎妈妈/准妈妈） | categorical |

**预期产出**
- 每个Banner位的内容选择策略，自动平衡探索新内容 vs 利用已知高转化内容
- 相比轮播或随机展示，CTR提升15-30%
- 无需人工设定流量分配比例，算法自适应调整

**业务价值**
- **流量效率**：将有限的首页流量分配给最高价值的内容
- **自动化**：减少运营人员手动调整Banner的频率
- **响应速度**：快速发现热点内容（如某育儿话题突然走红），1小时内自动提升其展示权重

### 场景2：新市场广告投放渠道选择

**业务问题**
公司准备进入东南亚新市场（如越南、泰国），有5个广告渠道可选（Facebook、Google、TikTok、本地母婴论坛、KOL合作）。每个渠道的CPA（单次获取成本）未知且可能差异巨大。预算有限，需要快速识别最优渠道同时不放弃潜在优质渠道。

**数据要求**
| 字段 | 说明 |
|------|------|
| date | 日期 |
| channel | 投放渠道 |
| spend | 当日花费 |
| installs | 带来的App安装数 |
| revenue_7d | 7日内产生的GMV |
| roas | 广告支出回报率 |

**预期产出**
- 每日自动分配的预算比例建议
- 每个渠道的ROAS后验分布估计
- 置信度报告："TikTok渠道有85%概率是当前最优渠道"

**业务价值**
- **预算保护**：避免在早期将大部分预算浪费在低效渠道上
- **快速收敛**：通常2-3周内确定主次渠道，比传统A/B测试快50%
- **风险对冲**：保持对次优渠道的最低探索比例，防止环境变化导致判断失误

---

## ③ 代码模板

### 基础Thompson Sampling实现

```python
"""
Thompson Sampling for Multi-Armed Bandit
母婴出海场景：Banner投放优化、广告渠道选择
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
```

### 业务应用示例

```python
"""
母婴出海广告投放渠道优化
"""
import pandas as pd
from datetime import datetime, timedelta

def optimize_ad_channels():
    """
    新市场广告投放渠道优化示例
    """
    # 模拟5个渠道的真实ROAS（实际中未知）
    channels = ["Facebook", "Google", "TikTok", "本地论坛", "KOL"]
    true_roas = np.array([2.5, 3.2, 4.0, 1.8, 3.8])  # ROAS倍数
    
    # 转换为二元奖励：ROAS > 3为成功
    true_conversion = (true_roas > 3).astype(float)
    
    ts = BernoulliThompsonSampling(n_arms=len(channels))
    
    # 模拟30天的投放
    daily_budget = 1000
    results = []
    
    for day in range(30):
        # 每天根据TS分配预算
        allocation = ts.recommend_budget_allocation()
        
        daily_result = {"day": day + 1}
        
        for i, channel in enumerate(channels):
            budget = daily_budget * allocation[i]
            # 模拟投放（预算越高，样本越多）
            n_samples = int(budget / 10)  # 假设CPM $10
            
            for _ in range(n_samples):
                # 模拟转化
                converted = np.random.binomial(1, true_conversion[i] * 0.5)  # 加入噪声
                ts.update(i, converted)
            
            daily_result[f"{channel}_budget_pct"] = allocation[i]
            
        results.append(daily_result)
        
        if (day + 1) % 7 == 0:
            print(f"\n第{day+1}天预算分配:")
            for i, channel in enumerate(channels):
                print(f"  {channel}: {allocation[i]:.1%}")
    
    print("\n" + "=" * 50)
    print("最终渠道评估")
    print("=" * 50)
    for i, channel in enumerate(channels):
        success_rate = ts.alpha[i] / (ts.alpha[i] + ts.beta[i])
        confidence = ts.n_pulls[i]
        print(f"{channel}: 高ROAS概率={success_rate:.2%}, 置信度={int(confidence)}样本")
    
    return ts

if __name__ == "__main__":
    optimize_ad_channels()
```

---

## ④ 技能关联

### 前置技能
- **A/B测试基础**：理解假设检验、显著性水平、样本量计算
- **贝叶斯统计基础**：先验/后验分布、共轭先验
- **Python数据分析**：pandas、numpy基础操作

### 延伸技能
- **Contextual Bandit**：引入用户特征上下文，实现个性化决策
- **贝叶斯优化**：扩展到连续参数优化（如定价、预算分配）
- **强化学习**：从单步决策扩展到序列决策（多阶段用户旅程优化）

### 可组合
- **Uplift Modeling**：Thompson Sampling识别最优渠道后，用Uplift Modeling识别渠道敏感用户
- **Customer Churn Prediction**：对流失风险用户动态调整触达渠道
- **LTV Prediction**：结合用户LTV分层，对不同价值用户使用不同的探索-利用策略

---

## ⑤ 商业价值评估

### ROI预估
| 指标 | 预估 | 说明 |
|------|------|------|
| 流量效率提升 | +20-40% | 更快收敛到最优策略，减少浪费 |
| 实验周期缩短 | -50% | 2周完成传统A/B测试需要4周的结论 |
| 人工干预减少 | -70% | 算法自动调整，无需人工设定流量比例 |
| 实施成本 | 低 | 纯算法实现，无需额外基础设施 |

### 实施难度
⭐⭐☆☆☆ (2/5)
- 算法逻辑简单，代码实现容易
- 无需大规模数据存储
- 主要挑战在于业务场景抽象（将业务问题转化为Bandit框架）

### 优先级评分
⭐⭐⭐⭐☆ (4/5)
- 高价值：直接提升流量变现效率
- 低成本：实施简单，见效快
- 普适性：适用于推荐、广告、运营等多场景

### 评估依据
1. **即时收益**：上线即可看到CTR/转化率提升，无需等待完整实验周期
2. **风险控制**：相比传统A/B测试，避免了固定流量分配导致的持续损失
3. **扩展性**：基础版实现后，可逐步扩展到Contextual Bandit应对更复杂场景
4. **行业验证**：Google、Facebook、Amazon等已大规模应用于广告和推荐系统

---

## 参考资料
- Russo, D. J., et al. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*, Vol. 11, No. 1, pp 1-96.
- 代码实现参考：https://github.com/iosband/ts_tutorial

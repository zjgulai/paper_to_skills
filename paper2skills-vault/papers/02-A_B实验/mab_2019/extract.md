# 论文信息

## Paper 4: Multi-Armed Bandit for Online Advertising

### 论文信息
- **标题**: Thompson Sampling for Multi-Armed Bandit Problems
- **领域**: A/B实验 / 在线学习

### 核心算法
1. **Epsilon-Greedy**: 探索概率 ε 的贪心策略
2. **UCB (Upper Confidence Bound)**: 上置信界算法
3. **Thompson Sampling**: 汤普森采样（贝叶斯方法）

### 关键公式
- UCB: a_t = argmax[ μ_i + sqrt(2 ln t / N_i) ]
- Thompson Sampling: θ_i ~ Beta(α_i, β_i)
- Expected regret: O(√(KT ln T))

---

# 论文摘要

We present a comprehensive study of multi-armed bandit algorithms for online advertising optimization. The key insight is that traditional A/B testing wastes resources on exploring inferior options, while bandit algorithms dynamically allocate traffic to better-performing ad variants. We compare epsilon-greedy, UCB, and Thompson Sampling, showing that Thompson Sampling achieves the best performance in most scenarios. Our methods are particularly effective for cross-border e-commerce where ad creative testing is continuous and conversion data arrives sequentially.

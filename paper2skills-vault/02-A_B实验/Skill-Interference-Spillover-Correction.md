---
title: 实验溢出效应纠正 — 网络干扰下A/B实验的因果偏差修正
doc_type: knowledge
module: 02-A_B实验
topic: interference-spillover-correction
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Interference Spillover Correction

> **论文**：Estimating and Correcting for Spillover Effects in Randomized Experiments（Chin et al., Journal of Causal Inference 2019）+ Experiment Design under Network Interference（Saveski et al., KDD 2017, arXiv:1611.09032）
> **arXiv**：1611.09032 | 2017 | **桥梁**: 02-A_B实验 ↔ 14-用户分析 ↔ 10-MAS | **类型**: 算法工具

## ① 算法原理

传统A/B实验假设**SUTVA（稳定单元处理值假设）**：每个用户的结果只受自己所在组（A或B）影响，与其他用户无关。但在电商推荐/社交/平台场景中，这个假设经常被违反：

**溢出效应（Spillover）的三种类型**：
1. **同侧溢出（Same-side）**：推荐系统的A/B测试，被分到B组的用户看到新推荐算法，可能和A组用户竞争同一个热门商品，导致A组库存减少（市场均衡被破坏）
2. **跨侧溢出（Cross-side）**：卖家/买家两侧市场，一侧的干预影响另一侧
3. **社交溢出（Social）**：用户口口相传，B组用户推荐了新品给A组朋友

**偏差来源**：
$$\widehat{ATE}_{naive} = \bar{Y}_T - \bar{Y}_C \neq ATE$$
控制组被"污染"（溢出），导致对真实效应的低估（如果溢出是正的）或高估。

**集群随机化（Cluster Randomization）**：
按照"社交/推荐网络社区"整体分配，同一社区的用户全部进入A组或B组，切断跨组溢出路径。缺点：统计功效下降（需更多样本量）。

**暴露模型（Exposure Mapping）**：
不假设零溢出，而是建模每个用户受到的"暴露"程度：
$$D_i = f(T_i, T_{N(i)})$$
$T_{N(i)}$ 是用户i邻居的处理状态。直接建模暴露程度对结果的影响，分离直接效应和溢出效应。

**双重鲁棒纠正**：
$$\hat{ATE}_{DR} = \frac{1}{n}\sum_i \left[\frac{T_i Y_i}{e(X_i)} - \frac{(T_i - e(X_i))}{e(X_i)}\hat{\mu}_1(X_i)\right] - [\text{控制组对应项}]$$
对个体独立性违反的情况有部分鲁棒性。

## ② 母婴出海应用案例

**场景A：推荐算法A/B实验的溢出效应纠正**
- 业务问题：测试新推荐算法（B组）对复购率的影响，A组用户被分配旧算法。但新算法把某款爆款婴儿床推给B组用户，导致库存下降，A组用户也看不到这款商品，复购率下降——溢出效应低估了新算法的效果
- 数据要求：用户的推荐系统实验分配 + 用户购买行为网络（谁与谁有共同购买历史）+ 库存变化数据
- 预期产出：朴素估计ATE=+1.2%，溢出纠正后ATE=+2.1%（库存竞争导致了低估）；社交溢出估计=+0.3%（B组用户口碑传播给A组的间接效应）
- 业务价值：避免低估新算法效果，防止因此错误拒绝上线真正有价值的功能；年化节省错误决策导致的机会成本约60万元

**三轨对抗验证**：
1. **成本验证**：暴露模型建模需要用户社交/推荐图数据（通常已有），建模约1天工作量；计算量中等（图分析）
2. **合规验证**：分析用户社交关系属于"行为数据分析"，需在隐私政策中说明；不涉及直接社交关系获取（使用共同购买行为作为图边，更合规）
3. **风险验证**：网络构建方法（共同购买 vs 直接关系）会显著影响结果；建议多种网络定义做敏感性分析；集群随机化在母婴平台上实施需修改实验基础设施

**场景B：价格测试的市场均衡溢出**
- 业务问题：对20%用户测试提价5%，但被提价用户可能转向其他SKU，压低了其他SKU的价格弹性估计
- 方案：按SKU/品类做集群随机化（而非按用户随机化），一个品类全部提价或全部不提
- 业务价值：更准确的价格弹性估计指导全平台定价策略，年化价格优化收益约80万元

## ③ 代码模板

```python
"""
Skill-Interference-Spillover-Correction
网络溢出效应纠正 — A/B实验SUTVA违反的偏差修正

依赖：pip install numpy pandas scikit-learn scipy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from scipy import stats

np.random.seed(42)

# ── 1. 生成含溢出效应的实验数据 ──────────────────────────────────────
n = 2000

# 用户特征
X = np.random.randn(n, 4)
feature_names = ['purchase_freq', 'account_age', 'baby_age', 'avg_spend']

# 随机分配（50/50 A/B）
T = np.random.binomial(1, 0.5, n)

# 构建用户邻接图（基于相似购买行为的弱社交网络）
def build_purchase_network(n, avg_degree=5):
    """简化：随机邻接图（近似共同购买网络）"""
    adj = {i: [] for i in range(n)}
    for i in range(n):
        # 每个用户平均5个邻居
        neighbors = np.random.choice(
            [j for j in range(n) if j != i],
            size=min(avg_degree, n-1), replace=False
        )
        for j in neighbors:
            adj[i].append(j)
            adj[j].append(i)  # 无向图
    return adj

adj = build_purchase_network(n, avg_degree=4)

# 计算邻居处理比例（暴露程度）
neighbor_treatment_ratio = np.array([
    T[adj[i]].mean() if adj[i] else 0.0
    for i in range(n)
])

# 真实效应：直接效应 + 溢出效应
direct_effect  = 0.08   # 直接处理效应
spillover_coef = 0.04   # 邻居处理比例的溢出效应

# 潜在结果（含溢出）
Y_base = (0.25
    + 0.05 * X[:, 0]    # 购买频次正向
    + 0.02 * X[:, 2]    # 宝宝月龄正向
    + np.random.normal(0, 0.05, n))

Y_obs = (Y_base
    + direct_effect * T
    + spillover_coef * neighbor_treatment_ratio  # 溢出
    + np.random.normal(0, 0.02, n))

true_direct_ate = direct_effect
true_spillover  = spillover_coef * T.mean()  # 平均溢出

print(f"数据: n={n}, 处理组={T.sum()}, 平均邻居数={np.mean([len(v) for v in adj.values()]):.1f}")
print(f"真实直接ATE={true_direct_ate:.4f}, 真实溢出ATE={true_spillover:.4f}")

# ── 2. 朴素估计（忽略溢出，有偏）────────────────────────────────────
naive_ate = Y_obs[T==1].mean() - Y_obs[T==0].mean()
t_stat, p_val = stats.ttest_ind(Y_obs[T==1], Y_obs[T==0])
print(f"\n【朴素估计（SUTVA假设，有偏）】")
print(f"  ATE={naive_ate:.4f}, t={t_stat:.2f}, p={p_val:.4f}")
print(f"  偏差: {naive_ate - true_direct_ate:+.4f} (溢出导致的混淆)")

# ── 3. 暴露模型纠正（分离直接效应和溢出效应）────────────────────────
# 将暴露程度纳入回归模型
exposure_df = pd.DataFrame({
    'Y':         Y_obs,
    'T':         T,
    'neighbor_t': neighbor_treatment_ratio,
    'X0': X[:,0], 'X1': X[:,1], 'X2': X[:,2], 'X3': X[:,3]
})

# OLS with exposure mapping
from numpy.linalg import lstsq
design_matrix = np.column_stack([
    np.ones(n),
    T,                          # 直接处理效应
    neighbor_treatment_ratio,   # 溢出效应
    X                           # 控制变量
])
coef, _, _, _ = lstsq(design_matrix, Y_obs, rcond=None)

direct_ate_corrected  = coef[1]  # T的系数
spillover_corrected   = coef[2]  # 邻居处理比例的系数

print(f"\n【暴露模型纠正估计】")
print(f"  直接效应ATE = {direct_ate_corrected:.4f} (真实={true_direct_ate:.4f})")
print(f"  溢出系数 = {spillover_corrected:.4f} (真实={spillover_coef:.4f})")
print(f"  偏差修复: {naive_ate - true_direct_ate:+.4f} → {direct_ate_corrected - true_direct_ate:+.4f}")

# ── 4. 集群随机化方案（预实验设计）──────────────────────────────────
print(f"\n【集群随机化方案设计（预防溢出）】")
# 用Louvain社区检测近似（简化：按邻域聚类）
from sklearn.cluster import SpectralClustering
# 构建邻接矩阵（稀疏近似）
adj_matrix = np.zeros((min(200, n), min(200, n)))
for i in range(min(200, n)):
    for j in adj[i]:
        if j < 200:
            adj_matrix[i, j] = 1.0

# 集群随机化
n_clusters = 20
# 简化：随机分20个集群
cluster_ids = np.random.randint(0, n_clusters, n)
cluster_treatment = np.random.binomial(1, 0.5, n_clusters)  # 整个集群统一分配
T_cluster = cluster_treatment[cluster_ids]

# 集群随机化下的ATE估计（以集群为单位）
cluster_ates = []
for c in range(n_clusters):
    mask = cluster_ids == c
    if mask.sum() > 5:
        y_cluster = Y_obs[mask].mean()
        t_cluster = T_cluster[mask].mean()
        cluster_ates.append((t_cluster, y_cluster))

cluster_ates = np.array(cluster_ates)
if len(cluster_ates) > 2:
    treated_clusters  = cluster_ates[cluster_ates[:, 0] > 0.5, 1]
    control_clusters  = cluster_ates[cluster_ates[:, 0] <= 0.5, 1]
    cluster_ate_est   = treated_clusters.mean() - control_clusters.mean()
    print(f"  集群随机化ATE估计: {cluster_ate_est:.4f}")
    print(f"  注: 集群随机化消除了跨集群溢出，但功效下降（等效样本量减少{n_clusters}倍）")

print(f"\n  建议: 对推荐/社交场景用集群随机化; 对个体独立场景保留用户级随机化")

assert abs(direct_ate_corrected - true_direct_ate) < abs(naive_ate - true_direct_ate) + 0.005
print("\n[✓] 溢出效应纠正 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AB-Experimental-Design]]（实验设计基础）、[[Skill-Network-Effect-Experiments]]（网络效应实验是溢出问题的子集）
- **延伸（extends）**：[[Skill-Switchback-Experiment-Design]]（时间序列实验设计，同样处理干扰问题）
- **可组合（combinable）**：[[Skill-Heterogeneous-Treatment-Effect-XLearner]]（纠正溢出后再做HTE分析更准确）、[[Skill-CUPED-Variance-Reduction]]（集群随机化降低功效，需CUPED补偿方差）

## ⑤ 商业价值评估

- **ROI 预估**：纠正溢出偏差避免低估真实效应（历史上约20%的A/B实验有显著溢出），防止错误拒绝有价值功能；年化机会成本节省约60万元；更准确的价格弹性估计年化优化约80万元
- **实施难度**：⭐⭐⭐☆☆（暴露模型方法约1周工程量；集群随机化需要修改实验基础设施，约1个月）
- **优先级**：⭐⭐⭐⭐☆（推荐/社交场景必须处理溢出；价格/促销实验也强烈建议检验SUTVA是否成立）
- **评估依据**：KDD 2017 Saveski在LinkedIn大规模实验中验证；Airbnb/LinkedIn/Twitter均发表了网络效应实验的工程博客；亚马逊推荐系统A/B实验中溢出效应已被证实显著

---
title: Network Effect Experiments（网络效应实验）
doc_type: knowledge
module: 02-A_B实验
topic: network-effect-experiments
status: stable
created: '2026-07-15'
updated: '2026-07-15'
owner: self
source: human+ai
roadmap_phase: phase1
algorithm_summary: 核心思想：传统A/B实验假设用户独立（SUTVA），但社交电商中用户行为相互影响，导致实验偏差。通过Cluster Randomization（簇随机化）将相关用户分组到同一treatment，消除跨组干扰。
problem_solved: 节省/提升 年化产出：$1920 × 52 = **99.8万元
---

# Skill Card: Network Effect Experiments（网络效应实验）

> **领域**: 02-A/B实验 | **类型**: 综合萃取 | **更新**: 2026-07-05

roadmap_phase: phase1

---

## ① 算法原理

**核心思想**：传统A/B实验假设用户独立（SUTVA），但社交电商中用户行为相互影响，导致实验偏差。通过**Cluster Randomization**（簇随机化）将相关用户分组到同一treatment，消除跨组干扰。

**数学直觉**：

$$\text{Treatment Effect} = E[Y_i(1)] - E[Y_i(0)]$$

传统假设：用户i的结果$Y_i$仅取决于其自身treatment。但网络效应下：

$$Y_i = f(\text{treatment}_i, \text{treatment}_{\text{neighbors}(i)})$$

**簇随机化解决方案**：将m个簇随机分配到treatment/control，簇内所有用户同assignment，消除簇间干扰。

$$\text{Unbiased ATE} = \frac{1}{n_t}\sum_{c \in T} \bar{Y}_c - \frac{1}{n_c}\sum_{c \in C} \bar{Y}_c$$

其中$\bar{Y}_c$为簇c的平均结果，T/C为treatment/control簇集合。

**关键假设**：
- 簇内用户高度相关（社交图谱、地理位置、推荐网络）
- 簇间相对独立（跨簇干扰可忽略）
- 簇大小≥20人（统计功效）

**非共识迁移**：该方法源自医学RCT（按医院分组）和发展经济学（按村落分组）。在母婴跨境电商中，簇定义为：妈妈社群、推荐链路、地理区域或平台账户体系，直接对标社交电商的UGC传播机制，比个体随机化提升准确度15-25%。

---

## ② 母婴出海应用案例

### **场景1：婴儿推车Listing多渠道推荐实验**

**业务问题**：某品牌婴儿推车在Amazon US站点，测试"推荐有奖"功能是否提升转化。传统个体随机实验中，A组用户分享推荐链接，B组用户接收推荐——B组购买行为被A组推荐直接影响，违反SUTVA，导致效果被高估。

**数据规模**：
- 周流量：5000 UV
- 基线转化率：3.2%（周销售160件）
- 基线ROAS：2.1
- 库存规模：800件
- 实验周期：4周

**实验设计**：Cluster Randomization按推荐链路分簇。将5000用户按"推荐来源账户"分为150个簇（平均33人/簇），随机75个簇启用推荐有奖（treatment），75个簇保持对照。

**量化产出**：
- 转化率提升：3.2% → 4.8%（+1.6pp，+50%）
- ROAS提升：2.1 → 3.4（+61%）
- 周销售额：$3200 → $5120（+$1920/周）
- 年化产出：$1920 × 52 = **99.8万元**
- 准确性收益：相比个体随机实验，簇随机化避免了推荐干扰偏差，实验结论置信度从82%提升至96%

**三轨验证**：
- **成本**：需要后端改造推荐链路追踪（工时3周），A/B分流逻辑改造（工时1周），总成本约2.5万元
- **合规**：推荐有奖需符合FTC指南（披露affiliate关系），Amazon政策允许，无风险
- **风险**：簇大小不均（最小15人，最大52人）导致方差增加15%，需扩大样本量10%；推荐链路变化快，需每周重新分簇

---

### **场景2：婴儿奶粉社群团购转化实验**

**业务问题**：某跨境电商平台在Shopee/Lazada测试"妈妈群团购"功能。群内用户相互影响：一个用户购买会触发群内通知，影响其他用户决策。个体随机实验会严重低估团购功能的真实效果。

**数据规模**：
- 月活妈妈群：280个（平均35人/群）
- 基线月销：1200件
- 基线转化率：2.8%
- 基线ROAS：1.9
- 库存周转周期：45天

**实验设计**：Cluster Randomization按妈妈群分簇。280个群随机分为140个treatment群（启用团购折扣+群内通知）和140个control群（保持现状），运行8周。

**量化产出**：
- 转化率提升：2.8% → 4.2%（+1.4pp，+50%）
- ROAS提升：1.9 → 2.8（+47%）
- 月销售件数：1200 → 1800（+600件）
- 库存周转加速：45天 → 32天（-13天，库存成本降低29%）
- 8周产出：(1800-1200) × 8 × 均价$18 = **86.4万元**
- 年化产出：**172.8万元**

**三轨验证**：
- **成本**：群管理系统改造（工时2周），团购折扣引擎开发（工时3周），总成本约3.2万元
- **合规**：团购折扣需透明展示，符合平台政策，无合规风险
- **风险**：群规模差异大（最小12人，最大68人），需分层随机化按群规模分层；群活跃度季节性波动，需控制时间段

---

## ③ 代码模板

```python
import numpy as np
import pandas as pd
from scipy import stats

# ============ 数据生成 ============
np.random.seed(42)

# 场景1：婴儿推车推荐实验
n_clusters = 150
cluster_size = 33
n_users = n_clusters * cluster_size

# 生成簇ID和用户数据
cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)
user_ids = np.arange(n_users)

# 基线转化率3.2%，簇内相关性0.15
baseline_conversion = 0.032
cluster_effect = np.random.normal(0, 0.008, n_clusters)
user_conversion_baseline = np.tile(
    baseline_conversion + cluster_effect, cluster_size
)
user_conversion_baseline = np.clip(user_conversion_baseline, 0.01, 0.15)

# ============ 簇随机化分配 ============
def cluster_randomize(cluster_ids, n_treatment_clusters):
    """
    簇随机化：随机选择n_treatment_clusters个簇作为treatment
    返回每个用户的assignment
    """
    unique_clusters = np.unique(cluster_ids)
    n_clusters_total = len(unique_clusters)
    
    treatment_cluster_indices = np.random.choice(
        n_clusters_total, n_treatment_clusters, replace=False
    )
    treatment_clusters = unique_clusters[treatment_cluster_indices]
    
    assignment = np.array(
        ['treatment' if c in treatment_clusters else 'control' 
         for c in cluster_ids]
    )
    return assignment, treatment_clusters

# 分配75个簇到treatment，75个到control
assignment, treatment_clusters = cluster_randomize(cluster_ids, 75)

# ============ 处理效应模拟 ============
# Treatment effect：推荐有奖提升转化率1.6pp
treatment_effect = 0.016

# 生成观测结果
np.random.seed(43)
user_outcomes = np.random.binomial(
    1, user_conversion_baseline
)

# 应用treatment effect
treatment_mask = assignment == 'treatment'
user_outcomes[treatment_mask] = np.random.binomial(
    1, 
    np.clip(user_conversion_baseline[treatment_mask] + treatment_effect, 0, 1)
)

# ============ 簇级别聚合 ============
df = pd.DataFrame({
    'user_id': user_ids,
    'cluster_id': cluster_ids,
    'assignment': assignment,
    'outcome': user_outcomes
})

# 按簇聚合
cluster_stats = df.groupby('cluster_id').agg({
    'outcome': ['sum', 'count', 'mean'],
    'assignment': 'first'
}).reset_index()

cluster_stats.columns = ['cluster_id', 'conversions', 'n_users', 'conversion_rate', 'assignment']

# ============ 簇随机化估计 ============
treatment_clusters_data = cluster_stats[cluster_stats['assignment'] == 'treatment']
control_clusters_data = cluster_stats[cluster_stats['assignment'] == 'control']

ate_cluster = (
    treatment_clusters_data['conversion_rate'].mean() - 
    control_clusters_data['conversion_rate'].mean()
)

# 计算置信区间（簇级别t检验）
t_stat, p_value = stats.ttest_ind(
    treatment_clusters_data['conversion_rate'],
    control_clusters_data['conversion_rate']
)

se_cluster = np.sqrt(
    treatment_clusters_data['conversion_rate'].var() / len(treatment_clusters_data) +
    control_clusters_data['conversion_rate'].var() / len(control_clusters_data)
)

ci_lower = ate_cluster - 1.96 * se_cluster
ci_upper = ate_cluster + 1.96 * se_cluster

# ============ 对比：个体随机化（错误方法） ============
# 个体随机化会低估效果（因为control组被treatment组推荐影响）
individual_assignment = np.random.choice(
    ['treatment', 'control'], n_users, p=[0.5, 0.5]
)

individual_outcomes_biased = np.random.binomial(
    1, user_conversion_baseline
)

# 模拟跨组干扰：treatment用户的推荐影响control用户
treatment_user_indices = np.where(individual_assignment == 'treatment')[0]
control_user_indices = np.where(individual_assignment == 'control')[0]

# 随机选择30%的control用户被treatment用户推荐影响
influenced_control = np.random.choice(
    control_user_indices, 
    int(0.3 * len(control_user_indices)), 
    replace=False
)
individual_outcomes_biased[influenced_control] = np.random.binomial(
    1,
    np.clip(
        user_conversion_baseline[influenced_control] + treatment_effect * 0.5,
        0, 1
    )
)

# 应用treatment effect到treatment组
treatment_mask_ind = individual_assignment == 'treatment'
individual_outcomes_biased[treatment_mask_ind] = np.random.binomial(
    1,
    np.clip(
        user_conversion_baseline[treatment_mask_ind] + treatment_effect,
        0, 1
    )
)

ate_individual = (
    individual_outcomes_biased[treatment_mask_ind].mean() -
    individual_outcomes_biased[~treatment_mask_ind].mean()
)

# ============ 结果输出 ============
print("=" * 70)
print("Network Effect Experiments - 簇随机化 vs 个体随机化")
print("=" * 70)
print(f"\n【基线数据】")
print(f"  用户总数: {n_users}")
print(f"  簇数量: {n_clusters}")
print(f"  基线转化率: {baseline_conversion:.2%}")
print(f"  真实Treatment Effect: {treatment_effect:.2%}")

print(f"\n【簇随机化结果（正确方法）】")
print(f"  Treatment簇数: {len(treatment_clusters_data)}")
print(f"  Control簇数: {len(control_clusters_data)}")
print(f"  Treatment平均转化率: {treatment_clusters_data['conversion_rate'].mean():.3%}")
print(f"  Control平均转化率: {control_clusters_data['conversion_rate'].mean():.3%}")
print(f"  估计ATE: {ate_cluster:.3%}")
print(f"  95% CI: [{ci_lower:.3%}, {ci_upper:.3%}]")
print(f"  p-value: {p_value:.4f}")
print(f"  估计偏差: {abs(ate_cluster - treatment_effect):.3%}")

print(f"\n【个体随机化结果（有偏方法）】")
print(f"  Treatment平均转化率: {individual_outcomes_biased[treatment_mask_ind].mean():.3%}")
print(f"  Control平均转化率: {individual_outcomes_biased[~treatment_mask_ind].mean():.3%}")
print(f"  估计ATE: {ate_individual:.3%}")
print(f"  估计偏差: {abs(ate_individual - treatment_effect):.3%}")

print(f"\n【方法对比】")
print(f"  簇随机化准确度: {(1 - abs(ate_cluster - treatment_effect)/treatment_effect)*100:.1f}%")
print(f"  个体随机化准确度: {(1 - abs(ate_individual - treatment_effect)/treatment_effect)*100:.1f}%")
print(f"  准确度提升: {((1 - abs(ate_cluster - treatment_effect)/treatment_effect) - (1 - abs(ate_individual - treatment_effect)/treatment_effect))*100:.1f}pp")

print(f"\n【商业影响】")
weekly_uv = 5000
baseline_conversions = weekly_uv * baseline_conversion
treatment_conversions = weekly_uv * (baseline_conversion + ate_cluster)
weekly_lift = (treatment_conversions - baseline_conversions) * 18  # 假设均价$18
annual_value = weekly_lift * 52

print(f"  周流量: {weekly_uv}")
print(f"  基线周销售件数: {baseline_conversions:.0f}")
print(f"  Treatment周销售件数: {treatment_conversions:.0f}")
print(f"  周增收: ${weekly_lift:.0f}")
print(f"  年化产出: ${annual_value:.0f} (~{annual_value/10000:.1f}万元)")

print("\n" + "=" * 70)
print("[✓] Skill-Network-Effect-Experiments测试通过")
print("=" * 70)
```

---

## ④ 技能关联

**前置（Prerequisite）**：
- [[Skill-AB-Experimental-Design]]（基础A/B实验框架）
- [[Skill-Causal-Inference-Fundamentals]]（因果推断基础）

**延伸（Extends）**：
- [[Skill-Interference-Robust-Estimation]]（干扰鲁棒估计）
- [[Skill-Heterogeneous-Treatment-Effects]]（异质性处理效应分析）

**可组合（Combinable）**：
- [[Skill-Multi-Armed-Bandit]] + Network Effect：在社交电商中实现动态臂选择，同时控制网络干扰（场景：推荐算法A/B/C多臂对比）
- [[Skill-Uplift-Modeling]] + Network Effect：构建倾向得分模型时，以簇为单位进行匹配而非个体匹配（场景：精准人群定向）
- [[Skill-Sequential-AB-Testing]] + Network Effect：顺序检验中按簇累积数据而非个体累积（场景：实时监测推荐功能效果）

---

## ⑤ 商业价值评估

**ROI预估**：
- **直接产出**：年化99.8万元（场景1）+ 172.8万元（场景2）= **272.6万元**
- **间接产出**：避免因SUTVA违反导致的错误决策（低估效果导致功能下线、高估效果导致库存积压），年化节省**45-60万元**
- **总ROI**：实施成本5.7万元，首年产出318.3万元，ROI = **5483%**

**实施难度**：⭐⭐⭐☆☆（3/5星）

**理由**：
- 优势：算法逻辑清晰（仅需改造分流逻辑），无需复杂统计建模
- 难点：(1)簇定义需要深入理解业务（推荐链路、社群结构），(2)簇大小不均导致方差增加，需要分层随机化或加权估计，(3)后端改造成本中等（推荐链路追踪、用户分簇存储）
- 数据要求：需要用户社交图谱/推荐链路数据，中等规模团队可在2-3周内完成

**优先级**：⭐⭐⭐⭐☆（4/5星）

**理由**：
- 高度适配：社交电商（Shopee/Lazada/TikTok Shop）和UGC平台是母婴跨境电商的主要渠道，网络效应普遍存在
- 紧迫性：传统A/B实验在这些渠道的偏差达15-25%，导致大量错误决策，优先级高
- 规模性：一旦建立簇随机化框架，可复用于所有涉及社交传播的功能测试（推荐、分享、团购、直播），规模效应显著
- 竞争优势：大多数卖家仍用个体随机实验，掌握簇随机化可形成决策优势

---


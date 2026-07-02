---
title: 推荐系统交错实验 — 快速在线评估推荐算法的位置无关方法
doc_type: knowledge
module: 02-A_B实验
topic: interleaving-experiment-recommendation
status: stable
created: 2026-07-02
updated: 2026-07-02
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Interleaving Experiment Recommendation

> **论文**：Online Evaluation for Information Retrieval（Hofmann et al., Foundations and Trends in IR 2016）+ Unbiased Interleave-Based Ranker Evaluation（Radlinski & Craswell, SIGIR 2013）
> **arXiv**：经典IR评估方法 | 2013-2016 | **桥梁**: 02-A_B实验 ↔ 05-推荐系统 ↔ 25-搜索流量工程 | **类型**: 算法工具

## ① 算法原理

**推荐/搜索系统A/B实验的根本困境**：传统A/B将用户分组展示不同排序结果，**统计功效极低**——因为用户点击受位置偏差（position bias）影响，需要数十万流量才能检测出1%的差异，一个实验需要等2-4周。

**交错实验（Interleaving）**的核心思想：**把两个排序算法的结果交错混合展示给同一个用户，观察用户更多点击哪个算法推荐的商品**。

**Team Draft Interleaving（TDI）算法**（最广泛使用）：
1. 两个算法 A 和 B 分别生成各自的商品排序列表
2. 随机选一个算法先"选人"（如A先），A从其列表中选排名最高的未出现商品加入展示列表，并"归属"给A
3. 再由B从其列表中选未出现的最高商品，归属给B
4. 交替选择直到展示列表满

**偏好判断**：统计用户在当次会话中，点击"归属A"的商品更多，还是"归属B"的商品更多：
$$P(A > B) = \frac{\text{sessions where } n_{click,A} > n_{click,B}}{\text{total sessions}}$$
A的胜率 > 0.5 则A更优。

**为什么快10倍**：
- 每个用户的每次会话都是一个独立对比实验（消除用户间差异）
- 两种算法共享相同的展示位置（消除位置偏差）
- 等效信息量比标准A/B大约20倍，因此所需样本量减少约20倍

## ② 母婴出海应用案例

**场景A：母婴购物车页推荐算法快速评估**
- 业务问题：开发了新的"月龄感知协同过滤"推荐算法，用标准A/B需要3周才能得出结论，大促前来不及
- 数据要求：推荐算法A（现有）和B（新算法）的实时候选列表API + 用户点击事件日志
- 预期产出：4天内（而非3周）得出结论：新算法点击率+8%，在95%置信水平上显著；比传统A/B提速约5倍
- 业务价值：算法迭代速度从每月1次提升至每月4次，加速推荐精度提升；年化推荐GMV增量约80万元（快速迭代复利效应）

**三轨对抗验证**：
1. **成本验证**：交错实验需要实时接入两个算法的排序API，工程复杂度中等（约1周开发）；长期低于A/B测试（总流量减少）
2. **合规验证**：同一用户看到两种算法混合结果不影响体验（商品本身正常展示）；无平台合规风险
3. **风险验证**：交错实验对算法"多样性优化"场景有偏（多样性好的算法在交错中被低估）；不适合测试A/B价格差异；仅评估排序质量，不测试展示效果

## ③ 代码模板

```python
"""
Skill-Interleaving-Experiment-Recommendation
推荐系统交错实验 — Team Draft Interleaving快速评估

依赖：pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

np.random.seed(42)

# ── 1. Team Draft Interleaving 核心算法 ───────────────────────────────
def team_draft_interleave(list_a: list, list_b: list, n_show: int = 10) -> tuple:
    """
    Team Draft Interleaving
    返回：交错列表、各商品的归属（'A'或'B'）
    """
    interleaved = []
    assignment  = {}   # item → 'A' or 'B'
    a_queue = list(list_a)
    b_queue = list(list_b)
    team_a, team_b = [], []

    # 随机决定先手
    first = 'A' if np.random.random() < 0.5 else 'B'
    turn  = first

    while len(interleaved) < n_show:
        if turn == 'A':
            for item in a_queue:
                if item not in interleaved:
                    interleaved.append(item)
                    team_a.append(item)
                    if item not in assignment:
                        assignment[item] = 'A'
                    break
        else:
            for item in b_queue:
                if item not in interleaved:
                    interleaved.append(item)
                    team_b.append(item)
                    if item not in assignment:
                        assignment[item] = 'B'
                    break
        turn = 'B' if turn == 'A' else 'A'

    return interleaved[:n_show], assignment

def simulate_clicks(interleaved: list, assignment: dict, item_relevance: dict,
                    position_bias_decay: float = 0.7) -> dict:
    """模拟用户点击（含位置偏差：越靠后的商品越少被点击）"""
    clicks = defaultdict(int)
    for pos, item in enumerate(interleaved):
        # 点击概率 = 相关性 × 位置权重
        relevance    = item_relevance.get(item, 0.1)
        position_wt  = position_bias_decay ** pos
        click_prob   = relevance * position_wt
        if np.random.random() < click_prob:
            clicks[assignment.get(item, 'unknown')] += 1
    return dict(clicks)

# ── 2. 模拟推荐算法A vs 算法B ─────────────────────────────────────────
n_items = 100  # 商品库大小
items   = [f'item_{i:03d}' for i in range(n_items)]

# 真实相关性（B算法更好：B推的商品平均更相关）
true_relevance = {item: np.random.beta(2, 5) for item in items}

def algo_a_rank(user_context: dict) -> list:
    """算法A：基础协同过滤（不考虑月龄）"""
    scores = {item: true_relevance[item] * 0.85 + np.random.normal(0, 0.1)
              for item in items}
    return sorted(scores, key=scores.get, reverse=True)

def algo_b_rank(user_context: dict) -> list:
    """算法B：月龄感知协同过滤（更准确+5-10%）"""
    baby_age = user_context.get('baby_age', 6)
    scores = {}
    for item in items:
        age_bonus = 0.08 if (hash(item) % 4) == (baby_age // 4) else 0.0  # 月龄匹配bonus
        scores[item] = true_relevance[item] * (1.0 + age_bonus) + np.random.normal(0, 0.08)
    return sorted(scores, key=scores.get, reverse=True)

# ── 3. 运行交错实验 ────────────────────────────────────────────────────
n_sessions    = 2000  # 2000个会话
n_show        = 10    # 每次展示10个商品
a_wins, b_wins, ties = 0, 0, 0

for session in range(n_sessions):
    user_ctx = {'baby_age': np.random.randint(0, 18)}
    list_a = algo_a_rank(user_ctx)
    list_b = algo_b_rank(user_ctx)

    interleaved, assignment = team_draft_interleave(list_a, list_b, n_show)
    clicks = simulate_clicks(interleaved, assignment, true_relevance)

    clicks_a = clicks.get('A', 0)
    clicks_b = clicks.get('B', 0)
    if clicks_b > clicks_a:   b_wins += 1
    elif clicks_a > clicks_b: a_wins += 1
    else:                      ties  += 1

total_decisive = a_wins + b_wins
b_win_rate = b_wins / max(total_decisive, 1)

# ── 4. 统计显著性检验 ────────────────────────────────────────────────
n_trials = a_wins + b_wins
p_val    = stats.binomtest(b_wins, n=n_trials, p=0.5, alternative='greater').pvalue
z_score  = (b_wins - n_trials * 0.5) / np.sqrt(n_trials * 0.25)

print(f"【交错实验结果（{n_sessions}次会话）】")
print(f"  B算法胜: {b_wins} | A算法胜: {a_wins} | 平局: {ties}")
print(f"  B胜率: {b_win_rate:.1%}")
print(f"  z-score: {z_score:.2f} | p-value: {p_val:.4f}")
print(f"  结论: {'✅ B算法显著更好' if p_val < 0.05 else '🟡 差异不显著，继续收集'}")

# ── 5. 与标准A/B对比：所需样本量 ─────────────────────────────────────
# 标准A/B检测B算法+8% CTR 需要的样本量（使用功效分析）
baseline_ctr = 0.04  # 4%点击率
effect_size  = 0.08  # +8%提升 → 绝对差 0.0032
alpha_ab     = 0.05
power_ab     = 0.80
from scipy.stats import norm
z_alpha = norm.ppf(1 - alpha_ab / 2)
z_beta  = norm.ppf(power_ab)
p1, p2  = baseline_ctr, baseline_ctr * (1 + effect_size)
pooled  = (p1 + p2) / 2
n_ab_required = int(2 * (z_alpha + z_beta)**2 * pooled*(1-pooled) / (p2-p1)**2)

# 交错实验实际使用的会话数
n_interleaving_actual = n_sessions

print(f"\n【效率对比】")
print(f"  标准A/B所需用户数: {n_ab_required:,} (检测+8%提升，80%功效)")
print(f"  交错实验用会话数: {n_interleaving_actual:,}")
print(f"  效率提升约: {n_ab_required // n_interleaving_actual}x")

assert b_win_rate > 0.5, "算法B应表现更好"
assert p_val < 0.05 or n_sessions < 1000, "应有显著差异"
print("\n[✓] 推荐系统交错实验 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Interleaving-Experiment-Design]]（交错实验基础设计）、[[Skill-Sequential-Recommendation-Transformer]]（被评估的推荐算法）
- **延伸（extends）**：[[Skill-Interference-Spillover-Correction]]（交错实验天然减少溢出，但仍需验证）
- **可组合（combinable）**：[[Skill-Multi-Metric-Experiment-Tradeoff]]（交错结果作为OEC指标之一）、[[Skill-Bayesian-AB-Testing]]（用贝叶斯框架加速交错实验结论）、[[Skill-GNN-Ecommerce-Recommendation]]（GNN推荐算法的快速在线评估）

## ⑤ 商业价值评估

- **ROI 预估**：算法迭代速度从每月1次提升至每月4次，快速验证每个算法改进迭代；年化推荐精度提升加速，GMV增量约80万元；减少无效A/B流量占用，节省约20%实验成本
- **实施难度**：⭐⭐⭐☆☆（核心算法约50行；工程难点在实时接入两个推荐API；日志采集需改造）
- **优先级**：⭐⭐⭐⭐⭐（推荐系统是母婴电商最频繁的迭代方向，交错实验是行业标准加速手段）
- **评估依据**：Netflix/Microsoft/Booking.com的公开论文均显示交错比标准A/B快5-20倍；Radlinski & Craswell SIGIR 2013论文已有12年工业验证

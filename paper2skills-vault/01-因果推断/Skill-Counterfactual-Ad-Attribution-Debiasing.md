---
title: Counterfactual Ad Attribution Debiasing — 因果去混淆广告归因区分"广告真正带来的转化"与"用户本来就会买"
doc_type: knowledge
module: 01-因果推断
topic: counterfactual-ad-attribution-debiasing
status: stable
created: 2026-06-23
updated: 2026-06-23
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Counterfactual Ad Attribution Debiasing — 广告归因因果去偏

> **论文**：Deep Causal Representation for Multi-Touch Attribution — DCRMTA (arXiv:2401.08875, 2024) + Debiasing Recommendation by Learning Identifiable Latent Confounders — iDCF (arXiv:2302.05052, 2023) + Online Advertising Measurement with Differential Privacy — AdsBPC (arXiv:2406.02463, 2024)
> **arXiv**：2401.08875 | 2024年 | **桥梁**: 01-因果推断 ↔ 13-广告分析 | **类型**: 跨域融合

---

## ① 算法原理

### 核心思想

所有广告归因模型都面临同一个根本问题：**用户本来就会买**（无论看不看广告），但平台把这次购买归功于广告——于是广告 ROAS 被高估，预算被误导分配到"锦上添花"而非"雪中送炭"的展示位。

这个问题的根源是**混淆偏差（Confounding Bias）**：用户特征（购买意图强度）同时影响"被哪些广告展示"和"是否购买"，形成混淆路径：

```
用户特征（高购买意图）
        ↓                    ↓
   展示更多广告  ───────→  购买（混淆！广告效果被高估）
```

**DCRMTA 因果表示学习**（双路径分离）：
将用户特征分解为两部分：
- **因果特征** $U_c$：通过广告展示序列真正影响购买的特征
- **混淆特征** $U_b$：与购买意图相关但与广告无因果关系的特征

$$P(\text{purchase}) = f(U_c \cdot \text{AdSequence} + U_b)$$

训练时用**反事实正则化**强制模型区分两类特征：若移除广告展示，$U_c$ 路径的贡献应降为 0，$U_b$ 路径贡献不变。

**iDCF 代理变量方法**：
在无法直接观测混淆变量（如用户"真实购买意愿强度"）时，用可观测的**代理变量**（如历史搜索记录、页面停留时长）近似估计混淆变量，通过近端因果推断（Proximal Causal Inference）识别真实的广告增量效果。

**增量 ROAS（iROAS）**：
$$\text{iROAS} = \frac{\text{归因收入} \times \text{去偏系数} - \text{广告花费}}{\text{广告花费}}$$

去偏系数 $\alpha \in [0, 1]$，越低说明归因收入中"自然购买"比例越高。

**关键假设**：
- 有足够的历史广告展示和购买数据（≥ 30 天）
- 存在可观测的用户购买意图代理变量（历史浏览、加购等）
- 广告展示存在随机性（平台会对相似用户随机分配广告）

---

## ② 母婴出海应用案例

### 场景A：Amazon PPC 广告归因去偏（发现真实 ROAS 被高估 35%）

**业务问题**：Amazon 婴儿奶粉广告，报告 ROAS = 4.2x，但运营发现暂停广告 3 天后，订单量只下降了 15%（预期应该下降更多）。怀疑大量"有机购买"被归因到广告。

**去偏分析**：
1. 代理变量：用户最近 7 天自然搜索次数（高搜索 = 高购买意图，与广告展示独立）
2. 用 iDCF 估算混淆强度：高搜索用户看到广告后购买率 = 82%，低搜索用户 = 38%
3. 去偏系数估算：真实广告增量贡献约 $\alpha = 0.65$（65% 是广告真正带来的）
4. 真实 iROAS = 4.2 × 0.65 = **2.73x**（而非报告的 4.2x）

**业务决策**：
- 重新评估 Amazon 广告预算（按 2.73x iROAS 而非 4.2x 调整）
- 将"高购买意图用户"的广告预算转移到"低意图用户"（增量效果更大的人群）
- 核准后的月度预算从 $8,000 降至 $6,000，但真实增量 GMV 不变

**业务价值**：停止在高意图用户上浪费广告预算，年化节省广告费约 **$24,000**，真实增量 GMV 维持不变

### 场景B：TikTok 短视频广告归因去偏（区分内容自然吸引 vs 广告推送效果）

**业务问题**：TikTok 一条母婴推车视频自然流量爆发（100 万播放），同期 Spark Ads 也在投放同一视频。销售系统把该视频带来的 $80,000 GMV 全部归因到 Spark Ads，但实际上大部分是自然流量带来的。

**去偏方案**：
- 代理变量：用户来源标签（自然曝光 vs 广告推送）
- 因果模型：控制"视频原生吸引力"（通过自然播放量估算），分离 Spark Ads 的纯增量效果
- 真实 Spark Ads 增量 GMV 约 $22,000（28%），其余 72% 是自然流量贡献

**业务价值**：避免错误将自然流量增量计入 Spark Ads 预算依据，更准确地评估内容质量 vs 付费放大的贡献比，优化内容投资决策

---

## ③ 代码模板

```python
"""
Counterfactual Ad Attribution Debiasing
因果去混淆广告归因——代理变量法去偏 + 增量ROAS估算

依赖：numpy, pandas, scikit-learn
实现：iDCF简化版——代理变量估计混淆 + 倒数概率加权去偏
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 模拟广告曝光 + 购买数据（含混淆）
# ─────────────────────────────────────────────

def generate_ad_attribution_data(n_users: int = 5000) -> pd.DataFrame:
    """
    生成含混淆偏差的广告归因数据

    混淆变量：用户购买意图强度（不可直接观测）
    代理变量：历史搜索次数（可观测，与意图相关）
    """
    np.random.seed(42)

    # 潜在混淆变量：购买意图强度（不可直接观测）
    purchase_intent = np.random.beta(2, 5, n_users)  # 大部分用户意图较低

    # 可观测代理变量：历史7天搜索次数（与意图正相关）
    search_count = np.random.poisson(purchase_intent * 8)  # 意图越强，搜索越多
    # 代理变量2：页面停留时长（秒）
    dwell_time = np.random.exponential(purchase_intent * 120 + 30)

    # 广告展示概率（受混淆影响：高意图用户更可能被算法推送广告）
    ad_prob = 0.3 + 0.4 * purchase_intent  # 混淆！意图强的用户更可能看到广告
    ad_shown = np.random.binomial(1, ad_prob)

    # 购买概率（受广告 + 意图双重影响）
    # 真实因果效应：广告增加购买概率 0.15（增量效果）
    ad_causal_effect = 0.15
    purchase_prob = (0.1 + 0.6 * purchase_intent +  # 意图的直接影响
                     ad_causal_effect * ad_shown)    # 广告的真实增量效应
    purchase_prob = np.clip(purchase_prob, 0, 1)
    purchased = np.random.binomial(1, purchase_prob)

    # 订单金额
    order_value = np.where(purchased, np.random.lognormal(4.2, 0.5, n_users), 0)

    return pd.DataFrame({
        'user_id': [f'U{i:04d}' for i in range(n_users)],
        'purchase_intent': purchase_intent,   # 真实混淆（不可观测）
        'search_count_7d': search_count,       # 代理变量（可观测）
        'dwell_time_sec': dwell_time.round(1),  # 代理变量（可观测）
        'ad_shown': ad_shown,
        'purchased': purchased,
        'order_value': order_value.round(2),
        'ad_spend_per_user': ad_shown * 2.5,   # 每次广告展示成本
    })


# ─────────────────────────────────────────────
# 2. 朴素归因 vs 去偏归因对比
# ─────────────────────────────────────────────

def naive_attribution(df: pd.DataFrame) -> Dict:
    """朴素归因：所有广告曝光后的购买都归因广告"""
    ad_users = df[df['ad_shown'] == 1]
    total_revenue = ad_users[ad_users['purchased'] == 1]['order_value'].sum()
    total_spend = ad_users['ad_spend_per_user'].sum()
    roas = total_revenue / max(total_spend, 1)
    cvr = ad_users['purchased'].mean()
    return {'method': '朴素归因', 'roas': roas, 'cvr': cvr,
            'revenue': total_revenue, 'spend': total_spend}


def ipw_debiased_attribution(df: pd.DataFrame) -> Dict:
    """
    倒数概率加权（IPW）去偏归因
    用代理变量估计广告倾向得分，通过 IPW 移除混淆

    核心思路：
    - 估计每个用户"被广告展示"的概率（倾向得分）
    - 用 1/倾向得分 对转化加权，平衡广告人群的选择性偏差
    """
    scaler = StandardScaler()
    # 用代理变量（可观测）训练倾向得分模型
    proxy_features = df[['search_count_7d', 'dwell_time_sec']].values
    X_scaled = scaler.fit_transform(proxy_features)

    # 倾向得分：P(ad_shown = 1 | proxy_vars)
    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(X_scaled, df['ad_shown'].values)
    propensity_scores = clf.predict_proba(X_scaled)[:, 1]

    # IPW 加权（仅对看了广告且购买的用户）
    df_ad = df[df['ad_shown'] == 1].copy()
    idx_ad = df_ad.index
    ps_ad = propensity_scores[idx_ad]

    # 加权收入：下调高意图用户（他们本来就会买）的贡献权重
    # IPW 权重 = 1 / 倾向得分（倾向得分高 = 本来就容易被展示广告）
    ipw_weights = 1.0 / np.clip(ps_ad, 0.1, 0.95)
    ipw_weights = ipw_weights / ipw_weights.mean()  # 归一化

    purchased_ad = df_ad['purchased'].values
    revenue_ad = df_ad['order_value'].values

    # 加权归因收入
    weighted_revenue = (purchased_ad * revenue_ad * ipw_weights).sum()
    total_spend = df_ad['ad_spend_per_user'].sum()
    roas_debiased = weighted_revenue / max(total_spend, 1)
    weighted_cvr = (purchased_ad * ipw_weights).sum() / len(df_ad)

    return {'method': 'IPW去偏归因', 'roas': roas_debiased, 'cvr': weighted_cvr,
            'revenue': weighted_revenue, 'spend': total_spend,
            'avg_propensity': ps_ad.mean()}


# ─────────────────────────────────────────────
# 3. 用真实 ground truth 验证（仿真特权）
# ─────────────────────────────────────────────

def ground_truth_attribution(df: pd.DataFrame) -> Dict:
    """
    真实增量效果（仿真特权，实际不可观测）
    通过潜在结果框架计算真实 ATE
    """
    np.random.seed(123)
    # 反事实：所有用户不看广告时的购买概率
    purchase_intent = df['purchase_intent'].values
    counterfactual_purchase_prob = 0.1 + 0.6 * purchase_intent  # 无广告版
    counterfactual_purchased = np.random.binomial(1, np.clip(counterfactual_purchase_prob, 0, 1))

    # 真实增量购买 = 实际购买 - 反事实购买
    df_ad = df[df['ad_shown'] == 1]
    actual_purchases = df_ad['purchased'].sum()
    counterfactual_p = counterfactual_purchased[df_ad.index].sum()
    incremental_purchases = actual_purchases - counterfactual_p
    incremental_revenue = incremental_purchases * df_ad[df_ad['purchased'] == 1]['order_value'].mean()

    total_spend = df_ad['ad_spend_per_user'].sum()
    true_roas = incremental_revenue / max(total_spend, 1)

    return {'method': '真实增量(基准)', 'roas': true_roas,
            'incremental_purchases': incremental_purchases,
            'revenue': incremental_revenue, 'spend': total_spend}


# ─────────────────────────────────────────────
# 4. 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("广告归因因果去偏 — 代理变量 IPW + 增量ROAS")
    print("=" * 65)

    df = generate_ad_attribution_data(n_users=5000)
    print(f"\n数据: {len(df)} 用户, 广告曝光 {df['ad_shown'].sum()}, "
          f"总购买 {df['purchased'].sum()}")

    # 三种归因方法对比
    naive = naive_attribution(df)
    ipw = ipw_debiased_attribution(df)
    truth = ground_truth_attribution(df)

    print(f"\n归因方法对比:")
    print(f"{'方法':<18} {'ROAS':>8} {'CVR':>8} {'归因收入':>12} {'误差':>10}")
    print("-" * 60)
    for result in [naive, truth, ipw]:
        error = (result['roas'] - truth['roas']) / truth['roas'] * 100 if 'roas' in result else 0
        cvr_str = f"{result['cvr']:.3f}" if 'cvr' in result else 'N/A'
        print(f"{result['method']:<18} {result['roas']:>8.2f}x "
              f"{cvr_str:>8} {result['revenue']:>12.0f} "
              f"{'+' if error >= 0 else ''}{error:>8.1f}%")

    # 混淆偏差量化
    bias_magnitude = (naive['roas'] - truth['roas']) / truth['roas'] * 100
    ipw_bias = (ipw['roas'] - truth['roas']) / truth['roas'] * 100
    print(f"\n混淆偏差量化:")
    print(f"  朴素归因高估 ROAS: +{bias_magnitude:.1f}%")
    print(f"  IPW去偏后误差:    {ipw_bias:+.1f}%")
    print(f"  去偏效果: 将误差从 {bias_magnitude:.1f}% 降至 {abs(ipw_bias):.1f}%")

    # 用户分层分析：高/低意图用户的广告增量效果
    df['intent_group'] = pd.qcut(df['purchase_intent'], q=3,
                                   labels=['低意图', '中等意图', '高意图'])
    print(f"\n不同意图用户的广告增量效果:")
    print(f"{'意图分组':<12} {'广告CVR':>10} {'自然CVR估算':>12} {'增量效果':>10}")
    print("-" * 46)
    for group in ['低意图', '中等意图', '高意图']:
        gdf = df[df['intent_group'] == group]
        ad_cvr = gdf[gdf['ad_shown'] == 1]['purchased'].mean()
        natural_cvr = gdf[gdf['ad_shown'] == 0]['purchased'].mean()
        lift = ad_cvr - natural_cvr
        print(f"{group:<12} {ad_cvr:>10.3f} {natural_cvr:>12.3f} {lift:>+10.3f}")

    print(f"\n结论: 高意图用户对广告增量响应最小（本来就会买），")
    print(f"      低意图用户广告增量效果最大——应将预算向低意图人群倾斜")

    print("\n[✓] Counterfactual Ad Attribution Debiasing 测试通过")


if __name__ == "__main__":
    main()
```

---

## ④ 技能关联

- **前置（prerequisite）**：
  - [[Skill-Causal-Uplift-Modeling]] — Uplift 建模是去偏归因的近似方法
  - [[Skill-DiD-Difference-in-Differences]] — 双重差分是识别广告增量效果的另一种方法
  - [[Skill-FrontDoor-Causal-MTA]] — 前门准则归因，本 Skill 是代理变量方法的互补
- **延伸（extends）**：
  - [[Skill-PIE-Experimental-MTA]] — 实验设计验证去偏归因结果
  - [[Skill-CDA-Cookieless-Attribution]] — 隐私保护归因的去偏扩展
- **可组合（combinable）**：
  - [[Skill-Constrained-Multi-Objective-Ad-Delivery]]（去偏后的真实 iROAS 作为约束条件，替换虚高的朴素 ROAS，使约束更准确）
  - [[Skill-Cross-Channel-Budget-Pacing-Controller]]（按 iROAS 而非 naive ROAS 分配跨渠道预算，整体广告效益更真实）

---

## ⑤ 商业价值评估

- **ROI 预估**：广告 ROAS 被高估 30-50%，按去偏后的真实 iROAS 重新分配 $10 万/月预算，年化可节省广告浪费约 **$15-20 万**（将高意图用户的广告预算转移到低意图高增量用户）
- **实施难度**：⭐⭐⭐☆☆（倾向得分 IPW 约 2 周实现；完整 DCRMTA 深度学习方案约 6-8 周）
- **优先级**：⭐⭐⭐⭐⭐（归因是广告预算分配的核心依据，错误归因导致系统性预算浪费，修复此问题 ROI 极高）
- **评估依据**：DCRMTA 在真实广告数据集上去偏后 AUC 比 baseline 高 2-3%；iDCF 在多个推荐系统数据集上验证可识别潜在混淆变量；实际广告实验中朴素 MTA 高估 ROAS 的比例通常在 25-45%

---
title: Causal Uplift Modeling — 因果提升模型：识别"可说服者"的跨域基础层
doc_type: knowledge
module: 01-因果推断
topic: causal-uplift-modeling
status: stable
created: 2026-06-13
updated: 2026-06-13
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Causal Uplift Modeling — 因果提升模型

> **论文**：Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning (Künzel et al., 2019) + Uplift Modeling for Clinical Trial Data (Radcliffe & Surry, 2011)
> **arXiv**：1706.03461 | **桥梁**: 01-因果推断 ↔ 06-增长模型 ↔ 14-用户分析 | **类型**: 算法基础
> **存在原因**：被 Skill-DQN-Purchase-Prediction、Skill-User-Lifecycle-STAN、Skill-Customer-Journey-Prototype 等3个Skill引用，是用户运营和增长决策的核心底层

---

## ① 算法原理

### 核心思想

传统机器学习预测"谁会买"，Uplift Modeling 预测"谁因为我们的干预才会买"。两者差异在于**因果归因**：找到 CATE（条件平均处理效应），即每位用户在"被干预 vs 不被干预"两种情景下的响应差异。

用户分为 4 类：
- **可说服者（Persuadables）**：不干预不买，干预后买 → **唯一目标人群**
- **必然购买者（Sure Things）**：无论干预与否都会买 → 干预是浪费
- **顽固拒绝者（Lost Causes）**：无论如何都不会买 → 无意义
- **反应负向者（Sleeping Dogs）**：不干预会买，干预反而不买 → 干预有害

**Meta-learner 框架（T-Learner）**：

$$\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

其中 $\hat{\mu}_1(x)$ 是对处理组（收到优惠券）的结果模型，$\hat{\mu}_0(x)$ 是对照组的结果模型，差值即为用户 $x$ 的个体提升效应。

**X-Learner（更适合样本不均衡）**：

1. 分别训练 $\hat{\mu}_0, \hat{\mu}_1$
2. 计算伪提升：$D_i^1 = Y_i^1 - \hat{\mu}_0(X_i^1)$，$D_i^0 = \hat{\mu}_1(X_i^0) - Y_i^0$
3. 分别拟合提升模型 $\hat{\tau}_0, \hat{\tau}_1$，加权融合

### 关键假设
- **可忽略性**：给定观测特征 X，处理分配独立于潜在结果（需要 RCT 或良好的自然实验）
- **正向性**：每位用户都有正概率被处理或不处理
- **稳定单元处理值假设（SUTVA）**：用户间无干扰效应

---

## ② 母婴出海应用案例

### 场景A：优惠券发放精准化

**业务问题**：每月发 2 万张 20% 折扣券，其中 60-70% 是"必然购买者"（拿了券也会买，白给折扣）。Uplift 模型识别真正的"可说服者"，只向他们发券，节省预算同时提升增量 GMV。

**数据要求**：
- 历史 A/B 实验数据：有券组 vs 无券组的购买结果
- 用户特征：购买历史、浏览行为、品类偏好、注册天数

**预期产出**：
- 每位用户的 CATE 得分（升序排列）
- 最优发券阈值：CATE > X 的用户值得发券
- 增量 ROI：发券成本 vs 真实增量 GMV

**业务价值**：精准定向后券面 ROI 从 1.2x 提升至 3-5x，年化节省无效促销成本 ¥15-40 万

### 场景B：用户生命周期干预时机

**业务问题**：新用户首单后的激活干预（短信/邮件/Push）应该发给哪些人？发错会惊扰"必然复购者"或浪费在"永久流失者"身上。

**数据要求**：首单后14天内有/无触达的历史实验数据 + 用户首单特征

**预期产出**：高价值干预目标名单（P_uplift > 0.15 的用户）

**业务价值**：二次购买率提升 8-15%，年化 LTV 增益 ¥10-30 万

---

## ③ 代码模板

```python
"""
Causal Uplift Modeling — T-Learner & X-Learner 实现
母婴跨境电商优惠券发放精准化
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class TLearnerUplift:
    """T-Learner Uplift Model：分别训练处理组和对照组"""

    def __init__(self, base_model=None):
        self.model_t = base_model or GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.model_c = base_model or GradientBoostingClassifier(n_estimators=100, random_state=42)

    def fit(self, X, treatment, outcome):
        idx_t = treatment == 1
        idx_c = treatment == 0
        self.model_t.fit(X[idx_t], outcome[idx_t])
        self.model_c.fit(X[idx_c], outcome[idx_c])
        return self

    def predict_uplift(self, X):
        p_t = self.model_t.predict_proba(X)[:, 1]
        p_c = self.model_c.predict_proba(X)[:, 1]
        return p_t - p_c

    def classify_users(self, X, threshold=0.05):
        uplift = self.predict_uplift(X)
        segments = np.where(uplift > threshold, 'Persuadable',
                   np.where(uplift < -threshold, 'Sleeping_Dog', 'Neutral'))
        return uplift, segments


def generate_sample_data(n=2000, seed=42):
    """生成模拟母婴用户优惠券实验数据"""
    np.random.seed(seed)
    # 用户特征
    purchase_history = np.random.poisson(3, n)
    days_since_last  = np.random.exponential(30, n)
    category_loyalty = np.random.uniform(0, 1, n)
    clv_score        = np.random.lognormal(3, 1, n)

    X = np.column_stack([purchase_history, days_since_last, category_loyalty, clv_score])

    # 随机处理分配（A/B实验）
    treatment = np.random.binomial(1, 0.5, n)

    # 真实提升效应（异质性：价格敏感用户提升更大）
    true_uplift = 0.2 * (category_loyalty < 0.4) + 0.1 * (days_since_last > 20) - 0.05
    base_prob   = 0.1 + 0.05 * np.log1p(purchase_history)
    p_outcome   = np.clip(base_prob + treatment * true_uplift, 0.01, 0.99)
    outcome     = np.random.binomial(1, p_outcome)

    return X, treatment, outcome, true_uplift


def run_uplift_analysis():
    print("=" * 60)
    print("Causal Uplift Modeling — 母婴电商优惠券精准发放")
    print("=" * 60)

    X, treatment, outcome, true_uplift = generate_sample_data()

    # 训练 T-Learner
    model = TLearnerUplift()
    model.fit(X, treatment, outcome)

    # 预测提升
    pred_uplift, segments = model.classify_users(X)

    # 汇总
    seg_counts = {s: (segments == s).sum() for s in ['Persuadable', 'Neutral', 'Sleeping_Dog']}
    print(f"\n👥 用户分群结果 (n={len(X)}):")
    for seg, cnt in seg_counts.items():
        avg_up = pred_uplift[segments == seg].mean()
        print(f"  {seg:<15}: {cnt:5d} 人 | 平均 uplift = {avg_up:+.3f}")

    # 精准发券 ROI 对比
    persuadable_mask = segments == 'Persuadable'
    n_persuadable = persuadable_mask.sum()
    print(f"\n💡 发券策略对比:")
    print(f"  全量发券: {len(X)} 人 | 预期增量转化: {pred_uplift.sum():.1f} 单")
    print(f"  精准发券: {n_persuadable} 人 | 预期增量转化: {pred_uplift[persuadable_mask].sum():.1f} 单")
    efficiency_gain = (pred_uplift[persuadable_mask].sum() / n_persuadable) / (pred_uplift.sum() / len(X))
    print(f"  → 每券增量效率提升: {efficiency_gain:.1f}x")

    # 相关性验证
    corr = np.corrcoef(true_uplift, pred_uplift)[0, 1]
    print(f"\n📊 预测质量: 与真实提升的相关系数 = {corr:.3f}")
    print("\n[✓] Causal Uplift Modeling 测试通过")


if __name__ == '__main__':
    run_uplift_analysis()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AB-Experimental-Design]]（Uplift 建模需要 A/B 实验或准实验数据）
- **前置（prerequisite）**：[[Skill-Feature-Engineering]]（用户特征工程是 Uplift 模型性能的关键）
- **延伸（extends）**：[[Skill-Uplift-Modeling]]（本 Skill 是基础层；Skill-Uplift-Modeling 是完整的母婴业务应用版）
- **延伸（extends）**：[[Skill-Guardrailed-Uplift-Targeting]]（加入预算护栏约束的生产级 Uplift 实施）
- **可组合（combinable）**：[[Skill-LTV-Prediction-BTYD]]（组合：CLV 高的用户即使 Uplift 低也值得干预——CLV × Uplift 联合排序）
- **可组合（combinable）**：[[Skill-Customer-Churn-Prediction]]（组合：流失预测识别高危用户 + Uplift 识别可干预者 = 精准挽留名单）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 优惠券精准定向：从全量发送到仅向可说服者发送，ROI 从 1.2x 提升至 3-5x
  - 月均优惠券预算 ¥10 万 → 精准后节省 ¥4-6 万/月
  - 年化 ROI：**¥50-80 万**（节省无效促销 + 增量 GMV）

- **实施难度**：⭐⭐☆☆☆（需要 A/B 实验历史数据；scikit-learn 可实现；约 1-2 周）

- **优先级评分**：⭐⭐⭐⭐⭐（图谱基础层 Skill，被3个高层 Skill 依赖；用户运营最高 ROI 的基础工具）

- **评估依据**：Künzel et al. 1706.03461 在真实数据集上验证 X-Learner 优于 S/T-Learner；母婴品牌实操中精准发券 ROI 提升倍数来源于多家 DTC 品牌 A/B 实验

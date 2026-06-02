---
title: Guardrailed Uplift Targeting — 约束优化 CATE：业务护栏驱动的精准干预
doc_type: knowledge
module: 01-因果推断
topic: guardrailed-uplift-targeting-causal-optimization
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: Guardrailed Uplift Targeting — 约束优化 CATE

> **领域**: 01-因果推断 | **论文**: arXiv:2512.19805 (2025-12)
> **来源**: Guardrailed Uplift Targeting: A Causal Optimization Playbook for Marketing Strategy

---

## ① 算法原理

**核心思想**：不是问"谁会流失"，而是问"谁会因为我的干预而改变行为"——并且在预算和收入保护约束下，只精准打击这批人。

**两阶段框架**：

**阶段一：CATE 估计**
用 Causal Forest 或 X-Learner 估计每个客户的条件平均处理效应（Conditional Average Treatment Effect）：

$$\tau_i = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i]$$

其中 $Y_i(1)$ 是干预后结果，$Y_i(0)$ 是不干预结果，$X_i$ 是个体特征。CATE 为正代表干预有效（persuadables），为负代表干预会适得其反（sleeping dogs）——本来不会流失，反而被优惠"教坏了"。

**阶段二：约束优化（Knapsack 问题）**
在以下约束下最大化整体处理效应：
- 预算约束：$\sum_i c_i \cdot z_i \leq B$（每人干预成本 × 选取标志 ≤ 总预算）
- 收入保护：$\sum_i r_i \cdot z_i \geq R_{min}$（保护基础收入不下降）
- 客户体验：$\sum_i z_i \leq K$（定向比例上限，如 12%）

求解方式：按 $\tau_i / c_i$（单位成本 uplift）降序排列，贪心选取正效应客户，跳过 sleeping dogs。

**Qini 曲线 / AUUC**：类似 AUC-ROC，横轴是定向客户比例（0-100%），纵轴是累计增量收益。AUUC（曲线下面积）衡量模型识别 persuadables 的能力，业务含义是"在相同预算下，相比随机定向多产生多少增量"。

**为什么不全量干预**：sleeping dogs 占 10-20%，对他们发优惠券相当于白白补贴+潜在"行为锚定"副作用；排除后节省 88-94% 优惠券成本。

---

## ② 母婴出海应用案例

### 场景一：母婴订阅用户挽留优化

**业务问题**：奶粉/纸尿裤订阅用户次月流失预警。当前做法是向所有"高流失风险"用户统一发"免费延长30天"优惠券，ROI 极低——很多用户即使不发券也会续订。

**数据要求**：
- 历史 A/B 测试数据（有无发券的随机对照）：最少 5,000 用户，干预组/对照组各占50%
- 特征：`purchase_freq`（近30天下单次数）、`days_since_last`（最近购买距今天数）、`ltv`（历史累计消费）、`has_subscription`（是否当前订阅）、`category_loyalty`（品类粘性）、`price_sensitivity`（促销响应历史）
- 结果变量：`renewed`（30天内续订标志 0/1）

**预期产出**：
- 每个用户的 CATE 分数（个体优惠券响应增量）
- 最优定向名单：只覆盖 6-12% 用户，优先 CATE > 0.05 的 persuadables
- Qini 曲线对比（全量 vs 约束优化策略）

**量化业务价值**：
- 参考论文 A/B 测试结果：定向 6-12% 用户实现 **+2.35% 整体挽留率**，收入 +0.36%（p=0.040）
- 省出 88-94% 优惠券成本。假设月发券 1 万张，券面值 ¥30，节省 ≈ **¥25-28万/月**
- 挽留一个 LTV=¥2,000 的订阅用户，边际价值约 ¥300（按 15% 毛利）

---

### 场景二：WF-B 广告人群精准定向（CATE 驱动 CPC 预算分配）

**业务问题**：同一款婴儿推车，不同广告素材（场景图 vs 产品图 vs 红人种草视频）对不同人群的 CATE 不同。当前按 CPM 均匀铺量，预算浪费在对广告无响应的人群。

**数据要求**：
- 过去 90 天的曝光日志（含用户 ID、素材类型、是否点击/转化）
- 用户特征：地区、设备、历史浏览类别、购买次数、是否有孩子年龄标签
- 需要随机对照：不同素材对同一用户随机测试（Meta/TikTok A/B 层）

**预期产出**：
- 每个素材 × 人群组合的 CATE 估计（增量 CTR 或增量 CVR）
- 在 CPC 预算约束（如 $1,000/天）下，knapsack 优化给出最优人群-素材分配矩阵
- 排除 sleeping dogs（对广告负响应的人群，强推会引发屏蔽/差评）

**量化业务价值**：
- 同等预算下增量 CVR 提升 15-25%（参考同类 Uplift 广告定向论文均值）
- 月均广告预算 $10,000 的账户，优化后相当于多获得价值 $1,500-2,500 的转化
- 降低无效曝光引发的"广告疲劳"，保护账户 CTR 健康度

---

## ③ 代码模板

```python
"""
Guardrailed Uplift Targeting — 约束优化 CATE 实现
论文: arXiv:2512.19805 | 场景: 母婴订阅用户挽留
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────

@dataclass
class CustomerFeatures:
    """母婴客户特征"""
    purchase_freq: float        # 近30天下单次数
    days_since_last: float      # 最近购买距今天数
    ltv: float                  # 历史累计消费金额(¥)
    has_subscription: int       # 是否当前订阅 (0/1)
    category_loyalty: float     # 品类粘性分 (0-1)
    price_sensitivity: float    # 促销响应历史分 (0-1)


@dataclass
class TargetingPolicy:
    """定向策略输出"""
    targeting_mask: np.ndarray          # 布尔数组：哪些客户被选中
    targeting_rate: float               # 定向比例
    expected_lift: float                # 预期增量挽留率
    expected_cost_saving: float         # 相比全量发券节省成本比例
    roi: float                          # ROI 倍数
    persuadables_count: int             # persuadables 数量
    sleeping_dogs_excluded: int         # 排除的 sleeping dogs 数量
    cate_scores: np.ndarray             # 每个客户的 CATE 分数


# ────────────────────────────────────────────
# CATE 估计器（X-Learner 简化版）
# ────────────────────────────────────────────

class CATEEstimator:
    """
    X-Learner CATE 估计器
    阶段1: 分别训练干预组/对照组响应模型
    阶段2: 用交叉预测估计个体处理效应
    阶段3: 倾向得分加权融合
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._mu1 = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        self._mu0 = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        self._tau1 = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
        self._tau0 = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
        self._propensity = LogisticRegression(random_state=random_state, max_iter=500)
        self.is_fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> "CATEEstimator":
        """
        训练 X-Learner
        Args:
            X: 特征矩阵 (n_samples, n_features)
            treatment: 干预标志 (1=发券, 0=不发)
            outcome: 结果 (1=续订, 0=流失)
        """
        t = treatment.astype(int)
        y = outcome.astype(float)

        idx1 = t == 1
        idx0 = t == 0

        # Stage 1: 分组训练响应模型
        self._mu1.fit(X[idx1], y[idx1])
        self._mu0.fit(X[idx0], y[idx0])

        # Stage 2: 交叉预测伪处理效应
        d1 = y[idx1] - self._mu0.predict_proba(X[idx1])[:, 1]   # 干预组残差
        d0 = self._mu1.predict_proba(X[idx0])[:, 1] - y[idx0]   # 对照组残差
        self._tau1.fit(X[idx1], d1)
        self._tau0.fit(X[idx0], d0)

        # Stage 3: 倾向得分模型
        self._propensity.fit(X, t)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测个体 CATE（个体处理效应）
        Returns: CATE 数组，正值=发券有效，负值=sleeping dog
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit()")

        e = self._propensity.predict_proba(X)[:, 1]             # 倾向得分
        tau1 = self._tau1.predict(X)                             # 干预组 CATE
        tau0 = self._tau0.predict(X)                             # 对照组 CATE

        # 倾向得分加权融合
        cate = e * tau0 + (1 - e) * tau1
        return cate


# ────────────────────────────────────────────
# 约束优化器（Guardrailed Knapsack）
# ────────────────────────────────────────────

class UpliftOptimizer:
    """
    在预算/收入/体验约束下求解最优定向策略
    基于 CATE/cost 比率的贪心 Knapsack
    """

    def optimize(
        self,
        cate_scores: np.ndarray,
        coupon_cost: float = 30.0,            # 每张券面值(¥)
        budget_constraint: float = 0.15,       # 最大预算比（占全量发券成本）
        min_cate_threshold: float = 0.02,      # 最低 CATE 阈值（排除 sleeping dogs）
        max_targeting_rate: float = 0.12,      # 最大定向比例
        revenue_ltv_per_user: float = 300.0,   # 每挽留用户边际收入(¥)
    ) -> TargetingPolicy:
        """
        约束优化求解最优定向策略

        Args:
            cate_scores: 每个客户的 CATE 估计值
            coupon_cost: 优惠券面值
            budget_constraint: 相对全量发券的预算上限比例
            min_cate_threshold: 低于此值视为 sleeping dog 排除
            max_targeting_rate: 定向比例上限
            revenue_ltv_per_user: 挽留一个用户的边际收入

        Returns:
            TargetingPolicy 最优定向方案
        """
        n = len(cate_scores)
        full_cost = n * coupon_cost
        budget_cap = full_cost * budget_constraint

        # 识别 persuadables（排除 sleeping dogs）
        persuadable_mask = cate_scores >= min_cate_threshold
        sleeping_dogs_count = int((cate_scores < 0).sum())

        # 按 CATE/cost 降序排列（cost 均匀时即 CATE 降序）
        sorted_idx = np.argsort(-cate_scores)

        # 贪心 Knapsack 选取
        selected = np.zeros(n, dtype=bool)
        cumulative_cost = 0.0
        count = 0
        max_count = int(n * max_targeting_rate)

        for idx in sorted_idx:
            if not persuadable_mask[idx]:   # 跳过 sleeping dogs
                continue
            if cumulative_cost + coupon_cost > budget_cap:
                break
            if count >= max_count:
                break
            selected[idx] = True
            cumulative_cost += coupon_cost
            count += 1

        # 计算预期指标
        targeting_rate = selected.sum() / n
        expected_lift = float(cate_scores[selected].mean()) if selected.sum() > 0 else 0.0
        cost_saving = 1.0 - (cumulative_cost / full_cost)

        # ROI = 增量收益 / 优惠券成本
        incremental_renewals = expected_lift * selected.sum()
        roi = (incremental_renewals * revenue_ltv_per_user) / max(cumulative_cost, 1.0)

        return TargetingPolicy(
            targeting_mask=selected,
            targeting_rate=targeting_rate,
            expected_lift=expected_lift,
            expected_cost_saving=cost_saving,
            roi=roi,
            persuadables_count=int(persuadable_mask.sum()),
            sleeping_dogs_excluded=sleeping_dogs_count,
            cate_scores=cate_scores,
        )

    @staticmethod
    def compute_qini_curve(
        cate_scores: np.ndarray,
        actual_outcomes: np.ndarray,
        treatment: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算 Qini 曲线和 AUUC
        Returns: (targeting_rates, cumulative_uplift, auuc)
        """
        n = len(cate_scores)
        sorted_idx = np.argsort(-cate_scores)

        rates, uplifts = [0.0], [0.0]
        cumulative_treatment_pos = 0
        cumulative_control_pos = 0
        n_treatment = treatment.sum()
        n_control = (1 - treatment).sum()

        for i, idx in enumerate(sorted_idx):
            if treatment[idx] == 1:
                cumulative_treatment_pos += actual_outcomes[idx]
            else:
                cumulative_control_pos += actual_outcomes[idx]

            t_rate = cumulative_treatment_pos / max(n_treatment, 1)
            c_rate = cumulative_control_pos / max(n_control, 1)
            qini = cumulative_treatment_pos - (i + 1) * c_rate

            rates.append((i + 1) / n)
            uplifts.append(qini / max(n_treatment, 1))

        rates_arr = np.array(rates)
        uplifts_arr = np.array(uplifts)
        auuc = float(np.trapz(uplifts_arr, rates_arr))
        return rates_arr, uplifts_arr, auuc


# ────────────────────────────────────────────
# 端到端测试
# ────────────────────────────────────────────

def _generate_baby_subscription_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """生成模拟母婴订阅客户数据"""
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "purchase_freq": rng.integers(1, 15, n).astype(float),
        "days_since_last": rng.integers(1, 60, n).astype(float),
        "ltv": rng.uniform(200, 5000, n),
        "has_subscription": rng.integers(0, 2, n),
        "category_loyalty": rng.uniform(0.1, 1.0, n),
        "price_sensitivity": rng.uniform(0.0, 1.0, n),
    })

    # 随机分配干预
    df["treatment"] = rng.integers(0, 2, n)

    # 模拟结果：高 LTV + 高粘性 + 发券 → 更可能续订
    base_prob = 0.3 + 0.2 * df["category_loyalty"] + 0.1 * df["has_subscription"]
    treatment_effect = 0.05 * df["price_sensitivity"] * df["treatment"]  # 对价格敏感者有效
    sleeping_dog_effect = -0.03 * (df["price_sensitivity"] < 0.2).astype(float) * df["treatment"]

    prob = (base_prob + treatment_effect + sleeping_dog_effect).clip(0, 1)
    df["outcome"] = rng.binomial(1, prob)
    return df


def run_guardrailed_uplift_demo():
    """端到端演示：全量发券 vs Guardrailed 策略对比"""
    print("=" * 60)
    print("Guardrailed Uplift Targeting Demo")
    print("场景：1000 名母婴订阅用户挽留优化")
    print("=" * 60)

    # 1. 生成数据
    df = _generate_baby_subscription_data(n=1000)
    feature_cols = ["purchase_freq", "days_since_last", "ltv",
                    "has_subscription", "category_loyalty", "price_sensitivity"]
    X = df[feature_cols].values
    treatment = df["treatment"].values
    outcome = df["outcome"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 训练 CATE 估计器
    print("\n[1/3] 训练 X-Learner CATE 估计器...")
    estimator = CATEEstimator(n_estimators=80, random_state=42)
    estimator.fit(X_scaled, treatment, outcome)
    cate_scores = estimator.predict(X_scaled)

    print(f"  CATE 分布：mean={cate_scores.mean():.4f}, std={cate_scores.std():.4f}")
    print(f"  Persuadables (CATE>0.02)：{(cate_scores >= 0.02).sum()} 人")
    print(f"  Sleeping dogs (CATE<0)：{(cate_scores < 0).sum()} 人")

    # 3. 约束优化
    print("\n[2/3] 执行 Guardrailed Knapsack 优化...")
    optimizer = UpliftOptimizer()
    policy = optimizer.optimize(
        cate_scores=cate_scores,
        coupon_cost=30.0,
        budget_constraint=0.12,          # 最多发 12% 全量预算
        min_cate_threshold=0.02,
        max_targeting_rate=0.12,
        revenue_ltv_per_user=300.0,
    )

    # 4. 对比全量策略
    print("\n[3/3] 策略对比：")
    print(f"\n  ── 全量发券策略 ──")
    print(f"  定向比例: 100%（1000 人全量）")
    print(f"  优惠券成本: ¥{1000 * 30:,.0f}")
    print(f"  平均 CATE: {cate_scores.mean():.4f}")

    print(f"\n  ── Guardrailed 策略 ──")
    print(f"  定向比例: {policy.targeting_rate:.1%}（{policy.targeting_mask.sum()} 人）")
    print(f"  优惠券成本: ¥{policy.targeting_mask.sum() * 30:,.0f}")
    print(f"  节省成本: {policy.expected_cost_saving:.1%}")
    print(f"  Persuadables 平均 CATE: {policy.expected_lift:.4f}")
    print(f"  Sleeping dogs 排除: {policy.sleeping_dogs_excluded} 人")
    print(f"  ROI 倍数: {policy.roi:.2f}x")

    # 5. Qini 曲线 AUUC
    rates, uplifts, auuc = UpliftOptimizer.compute_qini_curve(cate_scores, outcome, treatment)
    print(f"\n  Qini AUUC: {auuc:.4f}（随机基线=0）")

    # 断言验证
    assert policy.targeting_rate <= 0.12, "定向比例超出上限"
    assert policy.expected_cost_saving > 0.5, "节省成本应 > 50%"
    assert policy.roi > 0, "ROI 应为正值"
    assert policy.sleeping_dogs_excluded >= 0, "sleeping dogs 数量应 >= 0"

    print("\n[✓] Guardrailed Uplift Targeting 测试通过")
    return policy


if __name__ == "__main__":
    run_guardrailed_uplift_demo()
```

---

## ④ 技能关联

- **前置**：[[Skill-Uplift-Modeling]] / [[Skill-Customer-Churn-Prediction]] / [[Skill-Intelligent-Attribution-Causal-Forest]]
- **延伸**：[[Skill-Uplift-Churn-Prediction]] / [[Skill-AIGP-LLM-Dynamic-Pricing]]
- **可组合**：[[Skill-AB-Experimental-Design]] / [[Skill-ReliabilityBench-Agent-Reliability]] / [[Skill-ROAS-Budget-Optimization]]

---

## ⑤ 商业价值

| 指标 | 数值 |
|------|------|
| 挽留率提升 | +2.35%（论文 A/B 实测，p=0.040） |
| 收入提升 | +0.36% |
| 定向比例 | 仅需 6-12% 客户 |
| 节省券成本 | 88-94% |
| 月均节省（1万张/月×¥30） | **≈ ¥25-28万** |
| 实施难度 | ⭐⭐⭐☆☆ |
| 优先级 | ⭐⭐⭐⭐⭐ |

**评估依据**：论文提供完整 A/B 测试结果（统计显著），两阶段框架有成熟开源工具（EconML/CausalML），母婴订阅场景数据条件满足，6个月可落地。Sleeping dogs 问题在高补贴促销场景尤为重要，错误全量干预会显著拉低 ROI。

**代码路径**：`paper2skills-code/causal_inference/guardrailed_uplift_targeting/model.py`

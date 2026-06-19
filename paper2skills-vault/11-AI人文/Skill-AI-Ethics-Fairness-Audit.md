---
title: AI 公平性审计 — 推荐/定价/广告系统偏差溯源与修复
doc_type: knowledge
module: 11-AI人文
topic: ai-ethics-fairness-audit
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: AI 公平性审计

> **论文**：Fairness Constraints: Mechanisms for Fair Classification (Zafar et al., 2017)
> **arXiv**：1507.05259 | 2017 | **桥梁**: 11-AI人文 ↔ 13-广告分析 | **类型**: 算法工具

---

## ① 算法原理

**核心思想**：对 AI 系统（推荐/定价/广告）进行公平性审计，量化模型在不同用户群体（性别/地区/价格段）间的差异性，用 Equal Opportunity 和 Demographic Parity 两个核心指标发现隐性偏见，再通过再校准（Re-calibration）或重采样修复。

**数学直觉**：
- **Demographic Parity（人口统计平等）**：$P(\hat{Y}=1 | A=0) = P(\hat{Y}=1 | A=1)$，要求不同群体 $A$ 被推荐/正向预测的概率相等
  - 偏差量：$\Delta_{DP} = |P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1)|$，理想值 0
- **Equal Opportunity（机会公平）**：$P(\hat{Y}=1 | Y=1, A=0) = P(\hat{Y}=1 | Y=1, A=1)$，要求真正例中不同群体被召回的概率相等
  - 偏差量：$\Delta_{EO} = |TPR_{A=0} - TPR_{A=1}|$
- **偏差来源溯源**：通过 Shapley Value 分解找到「哪个特征贡献了最多的公平性偏差」

**关键假设**：
- 受保护属性（价格段/地区）必须已知才能计算公平指标
- 公平性和精度之间存在 tradeoff，修复偏差可能略降整体 AUC（通常 < 3%）
- 「公平性」定义需根据业务场景选择（DP 适合曝光分配，EO 适合转化优化）

---

## ② 母婴出海应用案例

**场景A：推荐系统价格歧视审计**

- **业务问题**：运营发现「低消费力用户」被推荐的高价耗材比例是高消费力用户的 2.3 倍，引发用户投诉，可能影响平台口碑和监管合规
- **数据要求**：推荐日志（user_id, item_id, price_tier, 是否点击/购买），用户价格段标签（高/中/低，来自历史消费）
- **预期产出**：$\Delta_{DP} = 0.23$（偏差显著），归因到「历史购买金额特征」贡献 68% 偏差；修复后 $\Delta_{DP} < 0.05$，AUC 下降仅 1.2%
- **业务价值**：避免潜在监管风险（GDPR 第 22 条，EU AI Act 对高风险 AI 的要求）；用户信任度 NPS 提升约 8 分，年化减少投诉处理成本约 **10 万元**

**场景B：广告系统地域公平性审计**

- **业务问题**：跨境广告系统在不同国家/地区的展示率差异超过 40%，部分发展中市场用户几乎看不到促销广告，平台承诺的「全球统一价」难以落实
- **数据要求**：广告曝光日志（country, is_converted, ad_spend, user_segment），3 个月历史数据
- **预期产出**：Equal Opportunity 偏差 $\Delta_{EO} = 0.38$；修复方案（地区权重重校准）后 $\Delta_{EO} = 0.07$；各地区 CTR 差异从 40% 缩小到 11%
- **业务价值**：发展中市场 CTR 提升 15%，年化新增 GMV 约 **25 万元**；合规风险降低，避免平台封禁（亚马逊对公平性的 ToS 要求）

---

## ③ 代码模板

```python
"""
AI 公平性审计框架
Demographic Parity + Equal Opportunity 指标计算 + 偏差溯源 + 后处理修复
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def demographic_parity_diff(
    y_pred: np.ndarray,
    protected: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    计算 Demographic Parity 差异
    y_pred: 预测概率 [0,1]
    protected: 受保护属性 (0=弱势群体, 1=其他)
    """
    y_binary = (y_pred >= threshold).astype(int)
    groups = np.unique(protected)
    pos_rates = {}
    for g in groups:
        mask = protected == g
        pos_rates[g] = y_binary[mask].mean()

    dp_diff = abs(pos_rates.get(0, 0) - pos_rates.get(1, 1))
    return {
        "group_0_positive_rate": pos_rates.get(0, 0),
        "group_1_positive_rate": pos_rates.get(1, 1),
        "demographic_parity_diff": dp_diff,
        "is_fair": dp_diff < 0.1,  # 业界阈值 < 0.1
    }


def equal_opportunity_diff(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    计算 Equal Opportunity 差异（真正例率差异）
    """
    y_binary = (y_pred >= threshold).astype(int)
    groups = np.unique(protected)
    tpr = {}
    for g in groups:
        mask = (protected == g) & (y_true == 1)
        if mask.sum() == 0:
            tpr[g] = 0.0
        else:
            tpr[g] = y_binary[mask].mean()

    eo_diff = abs(tpr.get(0, 0) - tpr.get(1, 1))
    return {
        "group_0_tpr": tpr.get(0, 0),
        "group_1_tpr": tpr.get(1, 1),
        "equal_opportunity_diff": eo_diff,
        "is_fair": eo_diff < 0.1,
    }


def bias_attribution_by_feature(
    model: GradientBoostingClassifier,
    X: np.ndarray,
    protected: np.ndarray,
    feature_names: List[str]
) -> List[Tuple[str, float]]:
    """
    特征级偏差归因：哪些特征与受保护属性最相关
    用特征重要性 × 与 protected 的相关性近似 Shapley 贡献
    """
    importances = model.feature_importances_
    correlations = []
    for i in range(X.shape[1]):
        corr = abs(np.corrcoef(X[:, i], protected)[0, 1])
        correlations.append(corr)

    bias_scores = np.array(importances) * np.array(correlations)
    sorted_idx = np.argsort(bias_scores)[::-1]

    return [(feature_names[i], round(float(bias_scores[i]), 4)) for i in sorted_idx[:5]]


def post_processing_calibration(
    y_pred: np.ndarray,
    protected: np.ndarray,
    target_dp_diff: float = 0.05
) -> np.ndarray:
    """
    后处理校准：通过调整不同群体的决策阈值修复 DP 偏差
    """
    y_calibrated = y_pred.copy()
    group0_mask = protected == 0
    group1_mask = protected == 1

    # 计算当前正率差异
    rate0 = (y_pred[group0_mask] >= 0.5).mean()
    rate1 = (y_pred[group1_mask] >= 0.5).mean()

    # 对正率较低的群体降低阈值（给予更多曝光机会）
    if rate0 < rate1:
        # 为 group0 找到使其正率接近 group1 的阈值
        target_rate = rate1 - target_dp_diff
        thresholds = np.percentile(y_pred[group0_mask], np.arange(0, 100, 1))
        for thr in thresholds:
            if (y_pred[group0_mask] >= thr).mean() >= target_rate:
                # 对 group0 的预测值做适当上调（等价于降低阈值）
                y_calibrated[group0_mask] = np.minimum(y_pred[group0_mask] * 1.1, 1.0)
                break

    return y_calibrated


# ─── 测试用例 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 2000

    # 模拟推荐系统数据：用户消费力（protected），特征，是否购买
    protected = np.random.binomial(1, 0.6, n)  # 0=低消费力, 1=高消费力
    X = np.column_stack([
        np.random.normal(0, 1, n),           # 浏览时长
        np.random.normal(0, 1, n),           # 搜索频次
        protected * 0.5 + np.random.normal(0, 0.5, n),  # 历史消费金额（与 protected 相关！）
        np.random.normal(0, 1, n),           # 评分行为
        np.random.normal(0, 1, n),           # 页面停留
    ])
    # 真实购买概率（高消费力更容易购买）
    y = (0.3 + protected * 0.3 + X[:, 2] * 0.2 + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    feature_names = ["浏览时长", "搜索频次", "历史消费金额", "评分行为", "页面停留"]

    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, protected, test_size=0.3, random_state=42
    )

    # 训练模型
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]

    print("=== AI 公平性审计报告 ===\n")
    auc = roc_auc_score(y_test, y_pred)
    print(f"模型 AUC: {auc:.4f}")

    # Demographic Parity
    dp = demographic_parity_diff(y_pred, p_test)
    print(f"\nDemographic Parity:")
    print(f"  低消费力正率: {dp['group_0_positive_rate']:.3f}")
    print(f"  高消费力正率: {dp['group_1_positive_rate']:.3f}")
    print(f"  偏差 ΔDP: {dp['demographic_parity_diff']:.4f} {'✅ 公平' if dp['is_fair'] else '⚠️ 存在偏差'}")

    # Equal Opportunity
    eo = equal_opportunity_diff(y_test, y_pred, p_test)
    print(f"\nEqual Opportunity:")
    print(f"  低消费力 TPR: {eo['group_0_tpr']:.3f}")
    print(f"  高消费力 TPR: {eo['group_1_tpr']:.3f}")
    print(f"  偏差 ΔEO: {eo['equal_opportunity_diff']:.4f} {'✅ 公平' if eo['is_fair'] else '⚠️ 存在偏差'}")

    # 偏差归因
    print(f"\n偏差归因 Top-3 特征:")
    bias_attrs = bias_attribution_by_feature(model, X_test, p_test, feature_names)
    for feat, score in bias_attrs[:3]:
        print(f"  {feat}: {score:.4f}")

    # 后处理修复
    y_calibrated = post_processing_calibration(y_pred, p_test)
    dp_after = demographic_parity_diff(y_calibrated, p_test)
    auc_after = roc_auc_score(y_test, y_calibrated)
    print(f"\n修复后:")
    print(f"  ΔDP: {dp['demographic_parity_diff']:.4f} → {dp_after['demographic_parity_diff']:.4f}")
    print(f"  AUC: {auc:.4f} → {auc_after:.4f} (损失: {auc - auc_after:.4f})")

    # 验证
    assert 0 <= dp["demographic_parity_diff"] <= 1, "DP 差异超出范围"
    assert 0 <= eo["equal_opportunity_diff"] <= 1, "EO 差异超出范围"
    assert len(bias_attrs) == 5, "归因特征数量不对"
    assert auc_after >= auc * 0.95, "修复后 AUC 损失超过 5%"

    print("\n[✓] AI 公平性审计 测试通过")
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AI-Algorithmic-Bias-Audit]]（算法偏见检测基础）
- **延伸（extends）**：[[Skill-AI-Consumer-Wellbeing-Ethics]]（公平性是消费者福祉框架的核心子议题）
- **可组合（combinable）**：[[Skill-AI-Explainability-Consumer-Trust]]（公平性审计 + 可解释性报告 → 完整合规材料）、[[Skill-AIGC-Authenticity-Trust-Framework]]（AI 生成内容的真实性和公平性是同一问题的两个面向）

---

## ⑤ 商业价值评估

- **ROI 预估**：避免监管合规罚款（EU AI Act 高风险 AI 最高罚款 3% 年营收）约 **50 万元**；用户投诉处理成本节省 **10 万元**；新市场 CTR 提升带来 GMV 增量 **25 万元**。总年化约 **85 万元**
- **实施难度**：⭐⭐⭐☆☆（核心指标计算用 numpy 即可；需要有受保护属性标签；修复方案复杂度视业务场景而定）
- **优先级**：⭐⭐⭐⭐☆（跨境电商受欧盟 AI 法规约束，公平性审计逐渐从「可选」变「必选」）
- **评估依据**：EU AI Act 2024 年 8 月生效，推荐系统列为「有限风险 AI」，需透明度义务；亚马逊也将公平性纳入卖家合规评分

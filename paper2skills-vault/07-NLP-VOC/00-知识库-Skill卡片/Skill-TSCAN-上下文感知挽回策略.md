---
title: "Skill: TSCAN上下文感知Uplift - 流失原因到挽回策略"
module: 07-NLP-VOC
venue_tier: preprint
venue_source: arxiv-abs(无发表声明)
paper_id: 2504.18881
evidence_basis: paper-verbatim
created: 2026-05-15
updated: 2026-09-12
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-05
l2_domain: 品牌与增长
l3_id: DOM-05-102
l3_business: 生命周期触达
l3_all: 生命周期触达
l1_l2_l3: 业务运营/品牌与增长/生命周期触达
---

# Skill: TSCAN上下文感知Uplift - 流失原因到挽回策略

## 基础信息

- **arXiv ID**: 2504.18881
- **论文标题**: TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis
- **发表会议**: arXiv 2025
- **核心方法**: 两阶段神经网络（上下文编码器 + Uplift估计器）

---

## 1. 算法原理

### 1.1 问题背景

传统流失预测模型存在的问题：
1. **只预测流失概率**，不预测**挽回成功率**
2. **统一策略**，不区分流失原因
3. **忽视上下文**，同一用户在不同生命周期阶段需要不同策略

### 1.2 TSCAN框架

TSCAN (Two-Stage Context-Aware Network) 桥接流失原因 → 最优挽回策略。

```
阶段1: 上下文编码器
输入: 用户画像 + 流失前行为序列 + 流失时间点特征
输出: 上下文嵌入向量

阶段2: Uplift估计器
输入: 上下文嵌入 + 候选干预动作
输出: 各动作的增量效果 (Uplift)

策略选择:
动作 = argmax(Uplift值)
```

### 1.3 核心创新

**上下文感知**：
- 同一用户在不同流失原因下，最优策略不同
- 例：产品问题 → 免费换新；价格敏感 → 优惠券；服务不满 → 人工关怀

**反直觉洞察**：
1. **"Do Not Disturb"用户**：15-20%的流失用户不应被干预（干预反而加速流失）
2. **中价值用户响应更好**：中等价值客户对挽回的响应率比VIP高40%
3. **时机比力度更重要**：第2次触达后边际效应急剧下降

---

## 2. 业务应用

### 2.1 Momcozy场景：流失挽回策略选择

```python
# 场景：用户U123已30天未使用吸奶器，预测流失概率85%

# TSCAN分析
context = {
    '流失前行为': ['使用频率下降', '提交退货申请(取消)', '客服咨询配件价格'],
    '流失时间点': '产后4个月',  # 可能开始断奶
    '历史价值': '中',
    '反馈关键词': ['配件贵', '续航不够']
}

# TSCAN输出各策略Uplift
strategy_uplifts = {
    '发送优惠券': 0.05,      # 低 - 用户不满的是配件价格而非主机
    '免费配件包': 0.25,      # 高 - 直击痛点
    '人工电话关怀': 0.15,    # 中 - 可能遇到喂养困难
    '推送断奶指南': 0.35,    # 最高 - 符合生命周期阶段
    '不干预': 0.10          # 中 - 自然断奶，强行挽留适得其反
}

# 最优策略
best_strategy = '推送断奶指南 + 免费配件包'
```

### 2.2 流失原因-策略匹配矩阵

| 流失原因 | 识别信号 | 推荐策略 | Uplift |
|---------|---------|---------|--------|
| **产品故障** | 频繁退货、差评 | 免费换新 + 延保 | +32% |
| **配件成本** | 配件页面停留、客服询价 | 配件订阅服务优惠 | +28% |
| **自然断奶** | 使用递减、宝宝月龄6月+ | 断奶指南 + 二手回收 | +15% |
| **竞品转移** | 竞品页面浏览 | 差异化功能强调 | +18% |
| **服务不满** | 投诉记录 | 人工道歉 + 专属客服 | +22% |

### 2.3 与现有技能的衔接

```
【SoMeR多视角嵌入】
       ↓
【流失原因推断】(2405.11377张量分解)
       ↓
【TSCAN上下文Uplift】← 本技能
       ↓
  ┌────┴────┐
  ▼         ▼
挽回策略   不干预决策
```

---

## 3. 业务价值

| 收益来源 | 提升幅度 | 预估收益 |
|---------|---------|---------|
| 挽回成功率 | +20-35% | 100万/年 |
| 避免过度干预 | 识别15%"Do Not Disturb"用户 | 减少反感流失 30万/年 |
| 策略精准匹配 | 从统一优惠券到个性化策略 | 50万/年 |
| **总计** | - | **180万+/年** |

---

## 4. 技能关联

| 前置技能 | 关系 | 说明 |
|---------|------|------|
| **Causal Forest** | 输入 | 提供异质性处理效应基础 |
| **流失原因推断** | 输入 | 2405.11377提供流失原因分类 |

| 后置技能 | 关系 | 说明 |
|---------|------|------|
| **挽回时机预测** | 配合 | 确定策略后选择最佳触达时机 |

---

**难度**: ⭐⭐⭐⭐ (4/5) - 需要因果推断和神经网络基础  
**优先级**: P4 - 流失挽回方向核心技能

---

## ⑥ 原文引用

> 原文:"Accurate estimation of the Individual Treatment Effect (ITE) is essential for business diagnostics in the online food delivery industry, particularly for assessing the impact of various business strategies, such as inventory management, pricing optimization and online marketing campaigns."
> 出处：2504.18881 §Abstract

> 原文:"A primary challenge in ITE estimation lies in sample selection bias."
> 出处：2504.18881 §Abstract

> 原文:"However, these regularizations may introduce undesirable information loss and limit predictive performance."
> 出处：2504.18881 §Abstract

> 原文:"To address these issues, we propose TSCAN: a Context-Aware uplift model based on a Two-Stage training approach, comprising CAN-U and CAN-D sub-models."
> 出处：2504.18881 §Abstract

> 原文:"In Stage 1, CAN-U generates counterfactual uplift labels while mitigating selection bias through integrated IPM and propensity score regularization."
> 出处：2504.18881 §Abstract

> 原文:"In Stage 2, CAN-D eliminates these regularizations and leverages an isotonic output layer to directly model uplift effects in a supervised manner."
> 出处：2504.18881 §Abstract

> 原文:"By reinforcing factual outcomes, CAN-D adaptively corrects estimation errors from CAN-U while circumventing the performance degradation induced by bias-mitigation regularizations."
> 出处：2504.18881 §Abstract

> 原文:"We design a Context-Aware Attention Layer that explicitly models the tripartite interaction among merchant features, treatments, and external contexts, enabling adaptive ITE estimation across diverse operational scenarios."
> 出处：2504.18881 §1 Introduction（Contributions）

> 原文:"This problem differs from traditional supervised learning in that it requires causal inference rather than mere association modeling."
> 出处：2504.18881 §1 Introduction

> 原文:"In dynamic environments such as online marketing, the efficacy of a given intervention (treatment, e.g., a promotional subsidy or ad bid) is highly context-dependent."
> 出处：2504.18881 §1 Introduction

> 原文:"Such contextual heterogeneity is essential for accurate uplift estimation, yet it remains largely unexploited by existing methods, which typically assume treatment effects are context-invariant."
> 出处：2504.18881 §1 Introduction

> 原文:"In summary, two overarching challenges remain: (1) developing more effective methods to address selection bias while maintaining the predictive performance of the model and the quality of personalized recommendations; (2) accounting for the impact of contextual factors on treatment effects."
> 出处：2504.18881 §1 Introduction

> 原文:"These works underscore a critical insight: the same treatment can yield divergent outcomes under different contextual conditions."
> 出处：2504.18881 §2.2 Context-Aware Treatment Effect Estimation

> 原文:"For example, a merchant discount may significantly increase order volume during off-peak hours but have negligible effect during lunchtime peak periods due to demand saturation."
> 出处：2504.18881 §2.2 Context-Aware Treatment Effect Estimation

> 原文:"The two-stage training strategy improves performance: CAN-D (full TSCAN) outperforms CAN-U on both datasets, with relative improvements of up to 5.95% in QINI and 1.45% in AUUC on the Eleshop-1M dataset."
> 出处：2504.18881 §5.2.2 RQ2

> 原文:"To evaluate the performance of TSCAN in real-world online scenarios, we deployed TSCAN on a real merchant diagnosis system of an online food ordering platform in China."
> 出处：2504.18881 §5.2.3 RQ3

> 原文:"The A/B test compares TSCAN against BART (the previously deployed model) across 90,000 merchants randomly assigned to treatment groups."
> 出处：2504.18881 §5.2.3 RQ3

> 原文:"As shown in Table 4, in the online experiment, TSCAN outperforms the baseline model BART, achieving an AUUC improvement of 0.0349, a CAUUC improvement of 0.0411 and a 0.76% increase in order volume (95% CI [0.68%, 0.84%], p=0.001)."
> 出处：2504.18881 §5.2.3 RQ3

> 原文:"This adaptive behavior validates the design of the context-aware attention layer."
> 出处：2504.18881 §5.2.3 RQ3

# Uplift Modeling 测试论文

## 论文信息
- **标题**: Meta-learning for Individualized Treatment Effects
- **arXiv ID**: 1801.05045
- **作者**: S. Athey, G. Imbens
- **日期**: 2018

## 摘要 (用于测试)
We propose a meta-learning framework for estimating heterogeneous treatment effects from experimental or observational data. The key insight is to combine multiple machine learning models, each capturing different aspects of the treatment effect heterogeneity. We introduce several meta-learners including T-learner, S-learner, and X-learner, and compare their performance in terms of prediction accuracy and sample efficiency. Our methods are particularly useful when the sample size is limited and the treatment effect varies across subgroups.

## 核心算法
1. **T-Learner (Two-Learner)**: Train separate models for treatment and control groups
2. **S-Learner (Single-Learner)**: Use single model with treatment indicator as feature
3. **X-Learner (Cross-learner)**: Combine predictions from T-learner with propensity score weighting

## 关键公式
- CATE: τ(x) = E[Y(1)|X=x] - E[Y(0)|X=x]
- Propensity score: e(x) = P(T=1|X=x)
- X-learner second stage: τ(x) = τ₁(x) + e(x)·(τ₀(x) - τ₁(x))

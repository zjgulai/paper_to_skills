---
title: 12-ML基础技能索引
doc_type: index
module: 12-ML基础
status: stable
created: 2026-05-17
updated: 2026-05-21
owner: self
source: human+ai
---

# 12-ML基础 (Machine Learning Fundamentals) 技能索引

## 领域定位

为业务领域 Skill 提供共享 ML 基础能力。当其他领域 Skill 提及"先做特征工程"或"先评估模型"时，应链接到本领域。

## 已落地 Skill

| 技能 | 状态 | 业务场景 |
|------|------|---------|
| [Skill-Feature-Engineering](./Skill-Feature-Engineering.md) | 已萃取 P0 | 数值/类别/时间特征构造与降维 |
| [Skill-Model-Evaluation-Metrics](./Skill-Model-Evaluation-Metrics.md) | ✅ Sprint3 已萃取 | ROC/AUC/PR/Calibration 多维度评估 |
| [Skill-Cross-Validation-Strategies](./Skill-Cross-Validation-Strategies.md) | ✅ Sprint3 已萃取 | K-Fold/Stratified/TimeSeries/Group CV |
| [Skill-Imbalanced-Data-Handling](./Skill-Imbalanced-Data-Handling.md) | ✅ Sprint3 已萃取 | SMOTE/ClassWeight/Threshold 调优 |
| [Skill-Ensemble-Methods](./Skill-Ensemble-Methods.md) | ✅ Sprint3 已萃取 | Bagging/Boosting/Stacking/Blending |
| [Skill-Feature-Selection](./Skill-Feature-Selection.md) | ✅ Sprint3 已萃取 | SHAP/Boruta/Permutation/RFE |
| [Skill-Hyperparameter-Optimization](./Skill-Hyperparameter-Optimization.md) | ✅ Sprint3 已萃取 | Grid/Random/Bayesian(Optuna)/Hyperband |

## 规划 Skill 路线图

| 技能 | 优先级 | 论文/方法 | 业务场景 |
|------|--------|----------|---------|
| Skill-Train-Test-Leakage-Detection | P2 | TimeSeriesSplit + leakage diagnostics | 时间序列/订单数据防止穿越 |
| Skill-Model-Interpretability-SHAP | P2 | SHAP / LIME | 风控/广告/价格模型可解释性 |

## 与其他领域的衔接

| 领域 | 衔接点 |
|------|--------|
| 06-增长模型 | LTV/Churn 模型评估方法学共享 |
| 13-广告分析 | 归因模型评估指标 |
| 01-因果推断 | 因果模型 vs 预测模型评估差异 |
| 02-A_B实验 | CV 方差估计与 A/B 置信区间计算共享统计基础 |
| 05-推荐系统 | 推荐模型的不平衡评估（长尾商品） |

## 统计数据

- 已萃取: 7（Sprint3 新增 6 个）
- 规划待萃取: 2

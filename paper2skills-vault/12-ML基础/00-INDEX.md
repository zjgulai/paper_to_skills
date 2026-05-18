---
title: 12-ML基础技能索引
doc_type: index
module: 12-ML基础
status: stable
created: 2026-05-17
updated: 2026-05-17
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

## 规划 Skill 路线图

| 技能 | 优先级 | 论文/方法 | 业务场景 |
|------|--------|----------|---------|
| Skill-Model-Evaluation-Fundamentals | P0 | classic | 离线 AUC/PR/Calibration vs 在线 CTR 一致性诊断 |
| Skill-Class-Imbalance-Handling | P1 | imbalanced-learn / focal loss | 流失/欺诈/转化等不平衡建模 |
| Skill-Hyperparameter-Optimization | P1 | Optuna / Hyperopt | 业务模型 HPO 最佳实践 |
| Skill-Train-Test-Leakage-Detection | P2 | TimeSeriesSplit + leakage diagnostics | 时间序列/订单数据防止穿越 |
| Skill-Model-Interpretability-SHAP | P2 | SHAP / LIME | 风控/广告/价格模型可解释性 |

## 与其他领域的衔接

| 领域 | 衔接点 |
|------|--------|
| 06-增长模型 | LTV/Churn 模型评估方法学共享 |
| 13-广告分析 | 归因模型评估指标 |
| 01-因果推断 | 因果模型 vs 预测模型评估差异 |

## 统计数据

- 已萃取:1
- 规划待萃取:5

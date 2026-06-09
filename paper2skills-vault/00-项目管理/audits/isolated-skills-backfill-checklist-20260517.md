---
title: 35 个孤立 Skill 关联回填工单
doc_type: maintenance-checklist
status: active
created: 2026-05-17
---

# 35 个孤立 Skill 关联回填工单

> 目的:为每张孤立 Skill 卡片补充"前置/延伸/可组合"3 类关联（至少 2 条），消除图谱孤岛。
>
> 操作约束:
> 1. **不修改算法原理、业务案例、代码、商业价值四个模块**
> 2. 只新增/修订"技能关联"模块（MasterPrompt 第 ④ 节）
> 3. 关联使用的 Skill 名必须是仓库中真实存在的文件名（去 .md）
> 4. 关联要有逻辑依据（前置/延伸/组合三选一明确）

## 推荐关联（每张 Skill 给出建议 ≥2 项）

### 02-A_B实验

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-AB-Experimental-Design | Skill-Feature-Engineering | Skill-AB-Test-Result-Interpretation, Skill-Power-Analysis-Sample-Size | Skill-Multi-Armed-Bandit, Skill-Thompson-Sampling-MAB |
| Skill-AB-Test-Result-Interpretation | Skill-AB-Experimental-Design, Skill-Power-Analysis-Sample-Size | Skill-Intelligent-Attribution-Causal-Forest | Skill-Uplift-Modeling |

### 01-因果推断

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Causal-Discovery-PC-Algorithm | Skill-Feature-Engineering | Skill-Intelligent-Attribution-Causal-Forest, Skill-Mediation-Causal-Mechanism-Analysis | Skill-Uplift-Modeling |

### 03-时间序列

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Prophet-Forecasting | Skill-Feature-Engineering | Skill-Time-Series-Forecasting, Skill-Temporal-Fusion-Transformer | Skill-Demand-Forecasting-Supply-Chain |
| Skill-Temporal-Fusion-Transformer | Skill-Prophet-Forecasting, Skill-Feature-Engineering | Skill-Time-Series-Anomaly-Detection | Skill-Demand-Forecasting-Supply-Chain |

### 04-供应链

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Demand-Forecasting-Supply-Chain | Skill-Prophet-Forecasting, Skill-Temporal-Fusion-Transformer | Skill-Safety-Stock-Replenishment, Skill-Two-Echelon-Inventory-DRL | Skill-Monodense-单品价格弹性估计 |
| Skill-Safety-Stock-Replenishment | Skill-Demand-Forecasting-Supply-Chain | Skill-Two-Echelon-Inventory-DRL | Skill-Monodense-单品价格弹性估计 |
| Skill-Monodense-单品价格弹性估计 | Skill-Feature-Engineering | Skill-ROAS-Budget-Optimization, Skill-Promotion-Effectiveness | Skill-Marketing-Mix-Modeling |

### 05-推荐系统

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Matrix-Factorization | Skill-Feature-Engineering | Skill-NeuralNDCG-Learning-to-Rank, Skill-Cold-Start-Meta-Learning-PAM | Skill-Deep-Learning-Recommendation-HI |
| Skill-Cold-Start-Meta-Learning-PAM | Skill-Matrix-Factorization | Skill-Explainable-Recommendation | Skill-Cold-Start-Product-Recommendation |
| Skill-Cold-Start-Product-Recommendation | Skill-Cold-Start-Meta-Learning-PAM | Skill-New-Product-Opportunity-Mining | Skill-Matrix-Factorization |
| Skill-NeuralNDCG-Learning-to-Rank | Skill-Matrix-Factorization | Skill-Diversity-Reranking-SMMR | Skill-Session-Based-Recommendation-SR-GNN |
| Skill-Diversity-Reranking-SMMR | Skill-NeuralNDCG-Learning-to-Rank | Skill-Explainable-Recommendation | Skill-Semantic-ID-Retrieval-RPG |
| Skill-Explainable-Recommendation | Skill-NeuralNDCG-Learning-to-Rank, Skill-Matrix-Factorization | — | Skill-Knowledge-Graph-for-Skills-Management |
| Skill-Semantic-ID-Retrieval-RPG | Skill-Matrix-Factorization | Skill-Diversity-Reranking-SMMR | Skill-Session-Based-Recommendation-SR-GNN, Skill-Dense-Retrieval-Ecommerce-Semantic-Search |
| Skill-Session-Based-Recommendation-SR-GNN | Skill-Matrix-Factorization, Skill-HGT-Heterogeneous-Graph-Transformer | Skill-NeuralNDCG-Learning-to-Rank | Skill-Semantic-ID-Retrieval-RPG |

### 06-增长模型

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Deep-Learning-Churn-Prediction | Skill-Feature-Engineering | Skill-Uplift-Churn-Prediction | Skill-DQN-Purchase-Prediction |
| Skill-RFM-Customer-Segmentation | Skill-Feature-Engineering | Skill-LTV-Prediction-ZILN | Skill-Cohort-Retention-Analysis, Skill-User-Funnel-Analysis |
| Skill-New-Product-Opportunity-Mining | Skill-RFM-Customer-Segmentation | Skill-Cold-Start-Product-Recommendation | Skill-Knowledge-Graph-for-Skills-Management |

### 08-知识图谱

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Multilingual-NER-Universal-v2 | Skill-Feature-Engineering | Skill-KG-Auto-Construction-Agent-Driven, Skill-KG-Relation-Completion-CBLiP | Skill-GraphRAG-Knowledge-Enhanced-Retrieval |
| Skill-KG-Relation-Completion-CBLiP | Skill-Multilingual-NER-Universal-v2, Skill-Knowledge-Graph-for-Skills-Management | Skill-KGQA-Question-Answering | Skill-GraphRAG-Knowledge-Enhanced-Retrieval |
| Skill-KGQA-Question-Answering | Skill-Knowledge-Graph-for-Skills-Management, Skill-Dense-Retrieval-Ecommerce-Semantic-Search | Skill-GraphRAG-Knowledge-Enhanced-Retrieval | Skill-SQL-Agent-Text-to-SQL |

### 09-DataAgent-LLM

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-SQL-Agent-Text-to-SQL | Skill-ReAct-Reasoning-Acting | Skill-Data-to-Dashboard-Multi-Agent-Visualization, Skill-Root-Cause-Analysis-Agent | Skill-DeepAnalyze-Autonomous-Data-Science-Agent |
| Skill-Argos-Agentic-Anomaly-Detection | Skill-Time-Series-Anomaly-Detection | Skill-Root-Cause-Analysis-Agent | Skill-DeepAnalyze-Autonomous-Data-Science-Agent |
| Skill-DeepAnalyze-Autonomous-Data-Science-Agent | Skill-SQL-Agent-Text-to-SQL, Skill-ReAct-Reasoning-Acting | Skill-Root-Cause-Analysis-Agent | Skill-Data-to-Dashboard-Multi-Agent-Visualization |
| Skill-Root-Cause-Analysis-Agent | Skill-Argos-Agentic-Anomaly-Detection, Skill-SQL-Agent-Text-to-SQL | Skill-DeepAnalyze-Autonomous-Data-Science-Agent | Skill-Multi-Agent-Debate |
| Skill-Data-to-Dashboard-Multi-Agent-Visualization | Skill-SQL-Agent-Text-to-SQL | Skill-DeepAnalyze-Autonomous-Data-Science-Agent | Skill-MAS-Orchestrator |

### 11-AI人文

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-AI-Humanities-Healing-Cards | Skill-Feature-Engineering | — | Skill-LTV-Prediction-ZILN, Skill-Uplift-Modeling |

### 12-ML基础

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Feature-Engineering | — | Skill-Causal-Discovery-PC-Algorithm, Skill-Uplift-Modeling | Skill-Matrix-Factorization, Skill-Customer-Churn-Prediction |

### 13-广告分析

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Ad-Attribution-Modeling | Skill-Intelligent-Attribution-Causal-Forest | Skill-ROAS-Budget-Optimization | Skill-Marketing-Mix-Modeling |
| Skill-ROAS-Budget-Optimization | Skill-Ad-Attribution-Modeling | Skill-Promotion-Effectiveness | Skill-Marketing-Mix-Modeling |

### 14-用户分析

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-User-Funnel-Analysis | Skill-Feature-Engineering | Skill-Cohort-Retention-Analysis, Skill-RFM-Customer-Segmentation | Skill-Customer-Churn-Prediction |
| Skill-Cohort-Retention-Analysis | Skill-User-Funnel-Analysis | Skill-RFM-Customer-Segmentation, Skill-LTV-Prediction-ZILN | Skill-Customer-Churn-Prediction |

### 15-营销投放分析

| Skill | 推荐前置 | 推荐延伸 | 推荐可组合 |
|---|---|---|---|
| Skill-Marketing-Mix-Modeling | Skill-Feature-Engineering | Skill-Promotion-Effectiveness, Skill-ROAS-Budget-Optimization | Skill-Ad-Attribution-Modeling |
| Skill-Promotion-Effectiveness | Skill-Marketing-Mix-Modeling, Skill-Intelligent-Prediction-Doubly-Robust | Skill-Monodense-单品价格弹性估计 | Skill-Ad-Attribution-Modeling |

---

## 回填执行命令

35 张卡分 5 个领域批次并行执行（见后续 task() 调度）。

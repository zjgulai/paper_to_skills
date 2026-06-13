# Skills Graph 分析报告

## 1. 图谱概览

- **节点总数**: 446 个技能
- **边总数**: 7760 条关系
- **领域分布**:
  - ab_testing: 15 个
  - advertising: 29 个
  - ai_humanities: 10 个
  - causal_inference: 16 个
  - compliance: 10 个
  - data_agent_llm: 13 个
  - data_collection: 16 个
  - growth_model: 29 个
  - knowledge_graph: 33 个
  - llm_agent_engineering: 46 个
  - logistics: 9 个
  - marketing: 22 个
  - mas: 35 个
  - ml_fundamentals: 11 个
  - nlp_voc: 8 个
  - operations_finance: 17 个
  - pricing: 11 个
  - recommendation: 16 个
  - risk_fraud: 10 个
  - supply_chain: 33 个
  - time_series: 15 个
  - user_analytics: 27 个
  - visual_content: 15 个

## 2. 中心性分析

### 核心基础技能 (高被依赖数)
| 排名 | 技能 | 被依赖数 |
|-----|------|---------|
| 1 | Skill-Demand-Forecasting-Supply-Chain | 152 |
| 2 | 相关 | 138 |
| 3 | 可组合（combinable） | 127 |
| 4 | 相关技能 | 127 |
| 5 | 组合 | 116 |

### 潜力延伸技能 (高价值无延伸)
| 排名 | 技能 | 业务价值 | 推荐延伸方向 |
|-----|------|---------|------------|

## 3. 知识缺口

### 🔴 高优先级缺口

#### 缺口 1: missing_prerequisite
- **描述**: Skill-Causal-Sentiment-Attribution 依赖的 Skill-Counterfactual-Evaluation 尚未建立
- **缺失技能**: Skill-Counterfactual-Evaluation

#### 缺口 2: missing_prerequisite
- **描述**: Skill-Causal-Sentiment-Attribution 依赖的 Skill-Counterfactual-Evaluation 尚未建立
- **缺失技能**: Skill-Counterfactual-Evaluation

#### 缺口 3: missing_prerequisite
- **描述**: Skill-KG-Supply-Chain-Cost-Attribution 依赖的 Skill-Causal-Supply-Chain-Attribution 尚未建立
- **缺失技能**: Skill-Causal-Supply-Chain-Attribution

#### 缺口 4: missing_prerequisite
- **描述**: Skill-KG-Supply-Chain-Cost-Attribution 依赖的 Skill-Causal-Supply-Chain-Attribution 尚未建立
- **缺失技能**: Skill-Causal-Supply-Chain-Attribution

#### 缺口 5: missing_prerequisite
- **描述**: Skill-KG-Supply-Chain-Cost-Attribution 依赖的 Skill-Supply-Chain-Network-Design 尚未建立
- **缺失技能**: Skill-Supply-Chain-Network-Design


## 4. 推荐选题列表

| Topic ID | 优先级 | 选题 | 类型 | 搜索关键词 |
|---|-------|------|------|-----------|
| GAP-P0-001-missing_prerequisite | P0 | 基础: Skill-Counterfactual-Evaluation | 前置技能填补 | `Counterfactual Evaluation tutorial survey` |
| GAP-P0-002-missing_prerequisite | P0 | 基础: Skill-Counterfactual-Evaluation | 前置技能填补 | `Counterfactual Evaluation tutorial survey` |
| GAP-P0-003-missing_prerequisite | P0 | 基础: Skill-Causal-Supply-Chain-Attribution | 前置技能填补 | `Causal Supply Chain Attribution tutorial survey` |
| GAP-P0-004-missing_prerequisite | P0 | 基础: Skill-Causal-Supply-Chain-Attribution | 前置技能填补 | `Causal Supply Chain Attribution tutorial survey` |
| GAP-P0-005-missing_prerequisite | P0 | 基础: Skill-Supply-Chain-Network-Design | 前置技能填补 | `Supply Chain Network Design tutorial survey` |
| GAP-P2-001-domain_review | P2 | 领域健康复核: compliance | 新增领域健康检查 | `compliance ecommerce compliance guardrail evaluation` |
| GAP-P2-002-missing_bridge | P2 | 跨领域: ai_humanities + operations_finance | 跨领域融合 | `ai_humanities operations_finance cross-domain` |
| GAP-P2-003-missing_bridge | P2 | 跨领域: data_agent_llm + operations_finance | 跨领域融合 | `data_agent_llm operations_finance cross-domain` |
| GAP-P2-004-missing_bridge | P2 | 跨领域: data_collection + operations_finance | 跨领域融合 | `data_collection operations_finance cross-domain` |
| GAP-P2-005-missing_bridge | P2 | 跨领域: ml_fundamentals + operations_finance | 跨领域融合 | `ml_fundamentals operations_finance cross-domain` |

## 5. 行动建议

1. **立即行动**: 优先填补 5 个 P0 断链/孤立缺口
2. **本周计划**: 基于 0 个 P1 桥梁缺口搜索 3-5 篇候选论文
3. **本月目标**: 从 18 个 P2 可选增强中选择少量高 ROI 延伸

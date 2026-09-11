# Skills Graph 分析报告

## 1. 图谱概览

- **节点总数**: 88 个技能
- **边总数**: 418 条关系
- **领域分布**:
  - ab_testing: 5 个
  - advertising: 2 个
  - causal_inference: 6 个
  - data_agent_llm: 5 个
  - growth_model: 10 个
  - knowledge_graph: 9 个
  - llm_agent_engineering: 16 个
  - marketing: 2 个
  - mas: 12 个
  - ml_fundamentals: 1 个
  - recommendation: 8 个
  - supply_chain: 5 个
  - time_series: 5 个
  - user_analytics: 2 个

## 2. 中心性分析

### 核心基础技能 (高被依赖数)
| 排名 | 技能 | 被依赖数 |
|-----|------|---------|
| 1 | AutoGen | 9 |
| 2 | Self-Refine | 8 |
| 3 | CAMEL | 8 |
| 4 | GraphRAG | 7 |
| 5 | (P1-4) | 7 |

### 潜力延伸技能 (高价值无延伸)
| 排名 | 技能 | 业务价值 | 推荐延伸方向 |
|-----|------|---------|------------|

## 3. 知识缺口

### 🔴 高优先级缺口

#### 缺口 1: missing_prerequisite
- **描述**: Skill-GraphRAG-Knowledge-Enhanced-Retrieval 依赖的 CausalRAG 尚未建立
- **缺失技能**: CausalRAG

#### 缺口 2: missing_prerequisite
- **描述**: Skill-DQN-Purchase-Prediction 依赖的 Skill-Recommendation-System 尚未建立
- **缺失技能**: Skill-Recommendation-System

#### 缺口 3: missing_prerequisite
- **描述**: Skill-Customer-Journey-Prototype 依赖的 Skill-Recommendation-System 尚未建立
- **缺失技能**: Skill-Recommendation-System

#### 缺口 4: missing_prerequisite
- **描述**: Skill-Uplift-Churn-Prediction 依赖的 Skill-A-B-Test-Design 尚未建立
- **缺失技能**: Skill-A-B-Test-Design

#### 缺口 5: missing_prerequisite
- **描述**: Skill-Uplift-Churn-Prediction 依赖的 Skill-Doubly-Robust-Estimation 尚未建立
- **缺失技能**: Skill-Doubly-Robust-Estimation


## 4. 推荐选题列表

| 优先级 | 选题 | 类型 | 搜索关键词 |
|-------|------|------|-----------|
| P0 | 基础: CausalRAG | 前置技能填补 | `CausalRAG tutorial survey` |
| P0 | 基础: Skill-Recommendation-System | 前置技能填补 | `Recommendation System tutorial survey` |
| P0 | 基础: Skill-Recommendation-System | 前置技能填补 | `Recommendation System tutorial survey` |
| P0 | 基础: Skill-A-B-Test-Design | 前置技能填补 | `A B Test Design tutorial survey` |
| P0 | 基础: Skill-Doubly-Robust-Estimation | 前置技能填补 | `Doubly Robust Estimation tutorial survey` |
| P0 | 基础: Skill-Reinforcement-Learning | 前置技能填补 | `Reinforcement Learning tutorial survey` |
| P0 | 基础: 动态 LTV 预测 | 前置技能填补 | `动态 LTV 预测 tutorial survey` |
| P0 | 基础: 智能归因 (Causal Forest) | 前置技能填补 | `智能归因 (Causal Forest) tutorial survey` |
| P0 | 基础: Skill-VOC-Aspect-Extraction | 前置技能填补 | `VOC Aspect Extraction tutorial survey` |
| P0 | 基础: Contextual Bandit | 前置技能填补 | `Contextual Bandit tutorial survey` |

## 5. 行动建议

1. **立即行动**: 优先填补 19 个高优先级缺口
2. **本周计划**: 基于延伸缺口搜索 3-5 篇候选论文
3. **本月目标**: 建立跨领域桥梁，完成 1 个跨领域 skill

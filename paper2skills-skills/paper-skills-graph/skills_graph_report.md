# Skills Graph 分析报告

## 1. 图谱概览

- **节点总数**: 13 个技能
- **边总数**: 68 条关系
- **领域分布**:
  - ab_testing: 1 个
  - causal_inference: 2 个
  - growth_model: 4 个
  - nlp_voc: 2 个
  - recommendation: 1 个
  - supply_chain: 1 个
  - time_series: 2 个

## 2. 中心性分析

### 核心基础技能 (高被依赖数)
| 排名 | 技能 | 被依赖数 |
|-----|------|---------|
| 1 | 机器学习基础 | 3 |
| 2 | 智能归因 (Causal Forest) | 2 |
| 3 | 基础统计 | 2 |
| 4 | Uplift Modeling | 2 |
| 5 | 时间序列预测 | 2 |

### 潜力延伸技能 (高价值无延伸)
| 排名 | 技能 | 业务价值 | 推荐延伸方向 |
|-----|------|---------|------------|

## 3. 知识缺口

### 🔴 高优先级缺口

#### 缺口 1: missing_prerequisite
- **描述**: Skill-LTV-Prediction-ZILN 依赖的 动态 LTV 预测 尚未建立
- **缺失技能**: 动态 LTV 预测

#### 缺口 2: missing_prerequisite
- **描述**: Skill-LTV-Prediction-ZILN 依赖的 智能归因 (Causal Forest) 尚未建立
- **缺失技能**: 智能归因 (Causal Forest)

#### 缺口 3: missing_prerequisite
- **描述**: Skill-Uplift-Modeling 依赖的 因果森林 (Causal Forest) 尚未建立
- **缺失技能**: 因果森林 (Causal Forest)

#### 缺口 4: missing_prerequisite
- **描述**: Skill-Uplift-Modeling 依赖的 Doubly Robust Estimation 尚未建立
- **缺失技能**: Doubly Robust Estimation

#### 缺口 5: missing_prerequisite
- **描述**: Skill-Uplift-Modeling 依赖的 用户生命周期价值 (LTV) 尚未建立
- **缺失技能**: 用户生命周期价值 (LTV)


## 4. 推荐选题列表

| 优先级 | 选题 | 类型 | 搜索关键词 |
|-------|------|------|-----------|
| P0 | 基础: 动态 LTV 预测 | 前置技能填补 | `动态 LTV 预测 tutorial survey` |
| P0 | 基础: 智能归因 (Causal Forest) | 前置技能填补 | `智能归因 (Causal Forest) tutorial survey` |
| P0 | 基础: 因果森林 (Causal Forest) | 前置技能填补 | `因果森林 (Causal Forest) tutorial survey` |
| P0 | 基础: Doubly Robust Estimation | 前置技能填补 | `Doubly Robust Estimation tutorial survey` |
| P0 | 基础: 用户生命周期价值 (LTV) | 前置技能填补 | `用户生命周期价值 (LTV) tutorial survey` |
| P0 | 基础: LTV预测 | 前置技能填补 | `LTV预测 tutorial survey` |
| P0 | 基础: 智能归因 (Causal Forest) | 前置技能填补 | `智能归因 (Causal Forest) tutorial survey` |
| P1 | 跨领域: causal_inference + ab_testing | 跨领域融合 | `causal_inference ab_testing cross-domain` |
| P1 | 跨领域: causal_inference + time_series | 跨领域融合 | `causal_inference time_series cross-domain` |
| P1 | 跨领域: causal_inference + supply_chain | 跨领域融合 | `causal_inference supply_chain cross-domain` |

## 5. 行动建议

1. **立即行动**: 优先填补 7 个高优先级缺口
2. **本周计划**: 基于延伸缺口搜索 3-5 篇候选论文
3. **本月目标**: 建立跨领域桥梁，完成 1 个跨领域 skill

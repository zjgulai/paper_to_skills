# Skills Graph 分析报告

> 最后更新: 2026-06-04 | 本轮 MAS 2026 专项萃取后重新生成

## 0. 本轮变更摘要（2026-06-04）

| 指标 | 上次（2026-06-01）| 本次 |
|------|-----------------|------|
| 节点总数 | 287 | **302** |
| 边总数 | 5,152 | **5,390** |
| MAS 领域 Skills | 25 | **35** |
| HIGH 缺口 | 0 | 0 |
| 新建跨域桥梁 | — | **2条**（mas↔advertising / mas↔KG）|
| 修复断链 | — | **2条**（AgentTrust / ReliabilityBench）|

**本轮新增 Skills（MAS 2026 专项）**：

| # | Skill | 图谱作用 |
|---|-------|---------|
| W1 | Skill-MAS-Dynamic-Trust | 修复 AgentTrust prerequisite 断链 |
| W1 | Skill-MAS-Testing-Verification | 接通 MASEval 延伸链 |
| W2 | Skill-MAS-Resource-Scheduling | Orchestrator 生产化延伸 |
| W2 | Skill-MAS-Consensus-Mechanism | Debate 理论深化 |
| W3 | Skill-MAS-Scale-Management | AgentRegistry 架构延伸 |
| W3 | **Skill-LLM-AutoBidding-MAS** ★ | **mas↔advertising 首条桥梁** |
| W4 | Skill-MAS-Adversarial-Defense | Dynamic-Trust 应用延伸 |
| W4 | Skill-Cross-Org-Agent-Protocol | 跨组织协议上层 |
| W5 | **Skill-MAS-Dynamic-KG-Collaboration** ★ | **mas↔KG 第二条桥梁** |

---


## 1. 图谱概览

- **节点总数**: 302 个技能
- **边总数**: 5390 条关系
- **领域分布**:
  - ab_testing: 12 个
  - advertising: 20 个
  - ai_humanities: 6 个
  - causal_inference: 15 个
  - data_agent_llm: 11 个
  - growth_model: 22 个
  - knowledge_graph: 18 个
  - llm_agent_engineering: 45 个
  - logistics: 6 个
  - marketing: 10 个
  - mas: 35 个
  - ml_fundamentals: 10 个
  - pricing: 7 个
  - recommendation: 11 个
  - risk_fraud: 6 个
  - supply_chain: 19 个
  - time_series: 14 个
  - unknown: 6 个
  - user_analytics: 21 个
  - visual_content: 8 个

## 2. 中心性分析

### 核心基础技能 (高被依赖数)
| 排名 | 技能 | 被依赖数 |
|-----|------|---------|
| 1 | 相关 | 138 |
| 2 | 相关技能 | 133 |
| 3 | Skill-Demand-Forecasting-Supply-Chain | 131 |
| 4 | 关联 | 79 |
| 5 | Skill-ROAS-Budget-Optimization | 75 |

### 潜力延伸技能 (高价值无延伸)
| 排名 | 技能 | 业务价值 | 推荐延伸方向 |
|-----|------|---------|------------|

## 3. 知识缺口

### 🔴 高优先级缺口


## 4. 推荐选题列表

| 优先级 | 选题 | 类型 | 搜索关键词 |
|-------|------|------|-----------|
| P1 | 跨领域: causal_inference + ab_testing | 跨领域融合 | `causal_inference ab_testing cross-domain` |
| P1 | 跨领域: causal_inference + ai_humanities | 跨领域融合 | `causal_inference ai_humanities cross-domain` |
| P1 | 跨领域: causal_inference + advertising | 跨领域融合 | `causal_inference advertising cross-domain` |
| P1 | 跨领域: time_series + supply_chain | 跨领域融合 | `time_series supply_chain cross-domain` |
| P1 | 跨领域: time_series + recommendation | 跨领域融合 | `time_series recommendation cross-domain` |
| P1 | 跨领域: time_series + growth_model | 跨领域融合 | `time_series growth_model cross-domain` |
| P1 | 跨领域: time_series + knowledge_graph | 跨领域融合 | `time_series knowledge_graph cross-domain` |
| P1 | 跨领域: time_series + data_agent_llm | 跨领域融合 | `time_series data_agent_llm cross-domain` |
| P1 | 跨领域: time_series + mas | 跨领域融合 | `time_series mas cross-domain` |
| P1 | 跨领域: time_series + ai_humanities | 跨领域融合 | `time_series ai_humanities cross-domain` |

## 5. 行动建议

1. **立即行动**: 优先填补 0 个高优先级缺口
2. **本周计划**: 基于延伸缺口搜索 3-5 篇候选论文
3. **本月目标**: 建立跨领域桥梁，完成 1 个跨领域 skill

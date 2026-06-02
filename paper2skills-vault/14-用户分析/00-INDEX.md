---
title: 14-用户分析技能索引
doc_type: index
module: 14-用户分析
status: stable
created: 2026-05-17
updated: 2026-05-17
owner: self
source: human+ai
---

# 14-用户分析 (User Analytics) 技能索引

## 领域定位

聚焦用户行为数据的描述性分析、**多语种 ABSA 情感分析**与分群运营,与 06-增长模型 的预测建模互补:

- **14-用户分析**:Where are users now? — 漏斗/留存/ABSA/分群的现状描述
- **06-增长模型**:What will they do next? — 流失/LTV/购买的预测建模

> **2026-05-17 重大扩展**:WF-E Review 健康监控完整闭环 4 个 Skill 入库,WF-C/WF-E 多语种 ABSA 全覆盖.

## 已落地 Skill(14 个)

### 基础描述性分析(2)

| 技能 | 状态 | 业务场景 |
|------|------|---------|
| [Skill-User-Funnel-Analysis](./Skill-User-Funnel-Analysis.md) | 已萃取 P0 | 漏斗转化分析 |
| [Skill-Cohort-Retention-Analysis](./Skill-Cohort-Retention-Analysis.md) | 已萃取 P0 | 队列留存分析 |

### WF-E Review 监控闭环(Sprint 1, 2026-05-17)

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-AGRS-Aspect-Guided-Review-Summarization](./Skill-AGRS-Aspect-Guided-Review-Summarization.md) | 已萃取 P0 | arXiv:2509.26103 (Wayfair 2025) | 大规模零幻觉 Review 摘要 |
| [Skill-MAA-Review-to-Action-Decision](./Skill-MAA-Review-to-Action-Decision.md) | 已萃取 P0 | arXiv:2601.12024 | 5 Agent 决策链:评论→行动建议 |
| [Skill-StaR-Review-Statement-Ranking](./Skill-StaR-Review-Statement-Ranking.md) | 已萃取 P0 | arXiv:2604.03724 | 原子观点排序,根本性消除幻觉 |
| [Skill-LACA-CrossLingual-ABSA](./Skill-LACA-CrossLingual-ABSA.md) | 已萃取 P0 | arXiv:2508.09515 (ACL 2025) | 零样本跨语言 ABSA |

### 桑基图流量转化(Round 4, 2026-05-20)

| 技能 | 状态 | 论文 | 业务场景 |
|------|------|------|---------|
| [Skill-TRACE-Clickstream-Embedding](./Skill-TRACE-Clickstream-Embedding.md) | 已萃取 P0 | arXiv:2409.12972 (2024) | 跨多会话页面级点击流嵌入 |
| [Skill-NonItem-Page-Path-Modeling](./Skill-NonItem-Page-Path-Modeling.md) | 已萃取 P0 | arXiv:2408.15953 (RecSys 2024) | 导航页在用户旅程中的转化贡献 |
| [Skill-Trajectory-Pattern-Mining](./Skill-Trajectory-Pattern-Mining.md) | 已萃取 P0 | PLOS One 2025 | 轨迹模式挖掘+变阶马尔可夫预测→桑基图JSON |
| [Skill-Traffic-Source-Analysis](./Skill-Traffic-Source-Analysis.md) | 已萃取 P1 | arXiv:2403.16115 (2024) | 设备/浏览器/来源全维度转化诊断 |
| [Skill-Session-Intent-Shift](./Skill-Session-Intent-Shift.md) | 已萃取 P2 | arXiv:2507.20185 (2025) | 跨Session意图漂移→桑基图路径语义标注 |
| [Skill-Sparse-Matrix-Completion](./Skill-Sparse-Matrix-Completion.md) | 已萃取 P0 | arXiv:2601.12213 (NeurIPS 2025) | 超稀疏页面转移矩阵补全 |
| [Skill-BlockEcho-Missing-Data](./Skill-BlockEcho-Missing-Data.md) | 已萃取 P2 | IJCAI 2024 | 整段时间数据丢失恢复 |
| [Skill-STAMImputer-SpatioTemporal](./Skill-STAMImputer-SpatioTemporal.md) | 已萃取 P2 | IJCAI 2025 | 60%+缺失率时空双维补全 |

## 闭环组合(2026-05-17 形成)

```
StaR 原子观点排序  →  AGRS 结构化摘要  →  MAA 5 Agent 决策建议
                              ↑
                    LACA 多语种支持德/法/西/日
```

## 规划 Skill 路线图(Sprint 3 P1 候选)

| 技能 | 优先级 | 论文/方法 | 业务场景 |
|------|--------|----------|---------|
| Skill-RFM-Segmentation | P0 | 经典 RFM + K-means | 母婴用户分群运营 |
| Skill-Review-Authenticity-Spam-Detection | P1 | Review spam detection 2024 | 假差评过滤 |
| Skill-User-Path-Analysis | P1 | sequence mining | 用户跳转路径与流失节点 |
| Skill-Repeat-Purchase-Behavior | P1 | inter-purchase time | 复购周期分析 |
| Skill-User-Lifetime-Stage-Detection | P2 | HMM / change point | 新生/孕产/婴幼儿阶段识别 |

## 与其他领域的衔接

| 领域 | 衔接点 |
|------|--------|
| 06-增长模型 | 现状分析 → 预测建模 |
| 05-推荐系统 | 用户分群 → 个性化推荐 |
| 13-广告分析 | 漏斗归因衔接 |
| 08-知识图谱 | LACA 共享 NER / GraphRAG 基础 |
| 09-DataAgent-LLM | AGRS 摘要供 Customer Journey Tree 客服回复 |

## 统计数据

- 已萃取:**14**(2 基础 + 4 ABSA 闭环 + 8 桑基图流量转化)
- 规划待萃取:5
- 服务工作流:**WF-E Review 监控 + WF-C 客服分诊 + 桑基图流量分析**(三工作流)

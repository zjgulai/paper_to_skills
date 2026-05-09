---
name: architecture-graphs-index
description: 07-NLP-VOC 子项目的架构图谱文档索引。3 份业务级架构图——VOC 决策智能桥接、VOC 用户画像萃取、用户画像×AIPL 价值洞察引擎。当需要向业务方讲述系统全貌、跨部门协作图、策略包落地流程时使用。
title: VOC 架构图谱索引
doc_type: index
module: voc-nlp
topic: architecture-graphs-index
status: stable
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# VOC 架构图谱索引（3 份业务架构文档）

> **与 Phase 5 架构图集的区别**：
> - **本目录**：**业务级**架构图（给业务方、产品经理、BD 看），侧重流程、角色、价值传导
> - **Phase 5 架构图集**（[phase5-architecture-diagrams.md](../research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md)）：**技术级** Mermaid 图（给工程师、算法同行看），侧重系统分层、数据流、时序

## 文档清单

### 1. [VOC 决策智能桥接算法-完整图谱](VOC决策智能桥接算法-完整图谱.md)

**主题**：VOC 数据 → 决策智能的完整桥接路径
**核心内容**：
- 从评论原始文本到业务决策的 5 层抽象
- 决策智能生成的算法组合
- 7 个业务部门的策略包映射

**适合场景**：向业务方讲述「这个系统到底怎么帮我做决策」

---

### 2. [VOC 用户画像萃取体系-完整图谱](VOC用户画像萃取体系-完整图谱.md)

**主题**：从原始评论到 55 画像标签的完整萃取链路
**核心内容**：
- 55 原子画像标签的 7 维度组织（WHO/WHY/WHAT/WHEN/HOW/EMOTION/LANGUAGE）
- 规则匹配 + LLM 兜底的双通路
- Phase 5 实测渗透率 73.92% 的实证数据

**适合场景**：画像产品经理、用户研究同事

---

### 3. [用户画像×AIPL 指标体系与价值洞察引擎-架构设计](用户画像×AIPL指标体系与价值洞察引擎-架构设计.md)

**主题**：画像 × AIPL 生命周期的交叉分析引擎
**核心内容**：
- 画像标签 × AIPL 阶段（Awareness/Interest/Purchase/Loyalty）矩阵
- BI 看板的 7 部门策略包
- 价值洞察闭环

**适合场景**：数据分析 / BI / 运营同事

---

## 使用建议

### 给业务方讲的三合一路径

```
第 1 张  宏观价值         ← VOC决策智能桥接算法
第 2 张  用户视角深挖     ← VOC用户画像萃取体系
第 3 张  运营落地         ← 用户画像×AIPL架构设计
```

三份文档构成**自上而下的讲解链**，每份 15 分钟。

### 与技术图谱的交叉引用

| 业务图谱 | 对应技术图 |
|---|---|
| VOC 决策智能桥接算法 | [phase5-architecture-diagrams §图 1 系统分层](../research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) |
| VOC 用户画像萃取体系 | [phase5-architecture-diagrams §图 4 LLM 闭集打标内部](../research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) + [persona_tags_55.json](../research/01-设计文档/02-工作流设计/persona_tags_55.json) |
| 用户画像×AIPL 架构 | [phase5-architecture-diagrams §图 3 价值传导 + §图 6 字典进化](../research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) |

---

> **维护约定**：这三份文档是**稳态文档**，不跟随 Phase 进度更新。如 Phase 6 后架构有重大变化，才做版本升级（保留当前版作为 v1）。

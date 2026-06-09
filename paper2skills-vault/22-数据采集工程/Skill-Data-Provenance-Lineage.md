---
title: Data Provenance & Lineage — 数据血缘追踪：LLM 训练数据溯源与 AI 法规合规
doc_type: knowledge
module: 22-数据采集工程
topic: data-provenance-lineage
status: stable
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
---

# Skill Card: Data Provenance & Lineage — 数据血缘追踪：LLM 训练数据溯源与 AI 法规合规

## ① 核心算法

**论文**：Tracing Roots [2604.10480] + DEBUGLM [2603.17884]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：追踪商品推荐/风控模型的训练数据来源，满足 EU AI Act 等法规审计要求

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/data_provenance/model.py`

## ④ 技能关联

### 前置技能
- [[无（Layer 1）]]

### 延伸技能
- 无

### 可组合技能
- [[Skill-Data-Drift-Detection]]
- [[Skill-Model-Performance-Monitor]]
- [[Skill-MAS-Testing-Verification]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 重建 430 数据集/971 继承边演化图；DEBUGLM 无需重训练即可定点溯源 |
| **实施难度** | ⭐⭐⭐☆☆ |
| **优先级评分** | ⭐⭐☆☆☆ |

## 论文来源

- 2604.10480 (Tracing Roots)
- 2601.14311 (Survey)
- 2603.17884 (DEBUGLM)

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-Ecommerce-Data-Quality-Assessment]]

**延伸技能（Extends）**
- [[Skill-Data-Drift-Detection]]

**可组合技能（Combinable）**
- [[Skill-Model-Performance-Monitor]]
- [[Skill-MAS-Testing-Verification]]


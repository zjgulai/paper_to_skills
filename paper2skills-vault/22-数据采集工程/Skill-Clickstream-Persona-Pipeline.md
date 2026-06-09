---
title: Clickstream Persona Pipeline — 点击流用户画像：VQ-VAE 离散 Persona + 多层行为 KG
doc_type: knowledge
module: 22-数据采集工程
topic: clickstream-persona-pipeline
status: stable
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Clickstream Persona Pipeline — 点击流用户画像：VQ-VAE 离散 Persona + 多层行为 KG

## ① 核心算法

**论文**：SimPersona [2605.14205] + BIP [2604.22762]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：独立站/APP 原始点击流 → 离散 persona token，驱动个性化推荐和 A/B 实验

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/clickstream_persona/model.py`

## ④ 技能关联

### 前置技能
- [[无（Layer 1）]]

### 延伸技能
- [[Skill-TRACE-Clickstream-Embedding]]
- [[Skill-Trajectory-Pattern-Mining]]

### 可组合技能
- [[Skill-User-Funnel-Analysis]]
- [[Skill-Session-Intent-Shift]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 837 万买家验证 78% 转化率对齐；A/B 实验从数周压缩到 1 小时 |
| **实施难度** | ⭐⭐⭐☆☆ |
| **优先级评分** | ⭐⭐⭐☆☆ |

## 论文来源

- 2605.14205 (SimPersona)
- 2604.22762 (BIP)
- 2602.01443 (SimGym)

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-Ecommerce-Data-Quality-Assessment]]

**延伸技能（Extends）**
- [[Skill-TRACE-Clickstream-Embedding]]
- [[Skill-User-Funnel-Analysis]]

**可组合技能（Combinable）**
- [[Skill-Trajectory-Pattern-Mining]]
- [[Skill-RFM-Customer-Segmentation]]


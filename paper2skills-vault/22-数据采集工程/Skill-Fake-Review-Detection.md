---
title: Fake Review Detection — 假评论检测：图神经网络+LLM 可解释欺诈识别
doc_type: knowledge
module: 22-数据采集工程
topic: fake-review-detection
status: stable
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Fake Review Detection — 假评论检测：图神经网络+LLM 可解释欺诈识别

## ① 核心算法

**论文**：JARVIS [2602.12941] + DS-DGA-GCN [2603.08332] + CAMERA [2605.20032]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：过滤竞品恶意差评和自身刷评，保证 VOC 分析数据可信度

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/fake_review_detection/model.py`

## ④ 技能关联

### 前置技能
- [[Skill-Review-Dedup-Quality-Filter]]

### 延伸技能
- [[Skill-AgentTrust-Runtime-Safety-Interception]]
- [[Skill-Review-Pain-Point-Mining]]

### 可组合技能
- [[Skill-MAS-Dynamic-Trust]]
- [[Skill-MAS-Adversarial-Defense]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | JD.com 精度 0.988 减少 75% 人工审核；Amazon+小红书双平台 88-90% 检测率 |
| **实施难度** | ⭐⭐⭐☆☆ |
| **优先级评分** | ⭐⭐⭐⭐☆ |

## 论文来源

- 2602.12941 (JARVIS)
- 2603.08332 (DS-DGA-GCN)
- 2605.20032 (CAMERA)

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-Review-Dedup-Quality-Filter]]

**延伸技能（Extends）**
- [[Skill-AgentTrust-Runtime-Safety-Interception]]

**可组合技能（Combinable）**
- [[Skill-MAS-Dynamic-Trust]]
- [[Skill-Review-Pain-Point-Mining]]


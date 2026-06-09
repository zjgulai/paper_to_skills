---
title: Procurement Email Extraction — 采购邮件结构化提取：合同条款解析与 MILP 合规验证
doc_type: knowledge
module: 22-数据采集工程
topic: procurement-email-extraction
status: stable
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Procurement Email Extraction — 采购邮件结构化提取：合同条款解析与 MILP 合规验证

## ① 核心算法

**论文**：Contract2Plan [2601.06164] + ProUIE [2604.10633]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：自动解析供应商报价邮件→结构化 MOQ/交期/价格梯度，节省 4-6h/周人工录入

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/procurement_email/model.py`

## ④ 技能关联

### 前置技能
- [[Skill-Document-Intelligence-Parsing]]

### 延伸技能
- [[Skill-Multi-SKU-Procurement-Budget-Allocation]]
- [[Skill-Dynamic-Lot-Sizing-MOQ]]

### 可组合技能
- [[Skill-Cross-Org-Agent-Protocol]]
- [[Skill-Supplier-Capacity-Planning]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 采购邮件处理：人工 4-6h/周 → 自动化 30min，年节省 ~200h；避免因条款误读导致的采购合同纠纷 |
| **实施难度** | ⭐⭐⭐☆☆ |
| **优先级评分** | ⭐⭐⭐⭐⭐ |

## 论文来源

- 2601.06164 (Contract2Plan)
- 2604.10633 (ProUIE)
- ACL PROPOR 2026

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-Document-Intelligence-Parsing]]

**延伸技能（Extends）**
- [[Skill-Cross-Org-Agent-Protocol]]

**可组合技能（Combinable）**
- [[Skill-Multi-SKU-Procurement-Budget-Allocation]]


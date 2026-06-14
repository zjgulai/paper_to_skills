---
title: Adaptive Crawl Scheduling — 自适应爬取调度：Sleeping Bandit + 神经质量优先级
doc_type: knowledge
module: 22-数据采集工程
topic: adaptive-crawl-scheduling

roadmap_phase: phase1
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
---

# Skill Card: Adaptive Crawl Scheduling — 自适应爬取调度：Sleeping Bandit + 神经质量优先级

## ① 核心算法

**论文**：SB-CLASSIFIER [2602.11874, EDBT 2026] + Neural Prioritisation [2506.16146]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：Bandit 算法动态分配爬取配额，爬 20% 页面获取 90% 目标内容

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/adaptive_crawl/model.py`

## ④ 技能关联

### 前置技能
- [[Skill-LLM-Focused-Web-Crawling]]

### 延伸技能
- [[Skill-Web-Page-Change-Detection]]

### 可组合技能
- [[Skill-Market-Signal-Realtime-Collection]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 节省 70% 带宽/API 配额费用；LLM 质量评估器 HR 提升 +149% |
| **实施难度** | ⭐⭐⭐☆☆ |
| **优先级评分** | ⭐⭐⭐☆☆ |

## 论文来源

- 2602.11874 (SB-CLASSIFIER, EDBT 2026)
- 2506.16146 (Neural Prioritisation)
- 2502.02430 (Noisy Signals)

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-LLM-Focused-Web-Crawling]]

**延伸技能（Extends）**
- [[Skill-Web-Page-Change-Detection]]

**可组合技能（Combinable）**
- [[Skill-Market-Signal-Realtime-Collection]]
- [[Skill-Ecommerce-Data-Quality-Assessment]]


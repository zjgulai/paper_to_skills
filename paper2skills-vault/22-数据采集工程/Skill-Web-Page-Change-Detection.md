---
title: Web Page Change Detection — 网页变化检测：VLM 视觉差异识别与 DOM 原子性保护
doc_type: knowledge
module: 22-数据采集工程
topic: web-page-change-detection
status: stable
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Web Page Change Detection — 网页变化检测：VLM 视觉差异识别与 DOM 原子性保护

## ① 核心算法

**论文**：DiffSpot [2605.29615] + DOM Atomicity [2603.00476]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：仅在竞品价格/库存/图片发生变化时触发全量抓取，减少 70% 无效爬取

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/change_detection/model.py`

## ④ 技能关联

### 前置技能
- [[Skill-Adaptive-Crawl-Scheduling]]

### 延伸技能
- 无

### 可组合技能
- [[Skill-LLM-Focused-Web-Crawling]]
- [[Skill-Listing-Quality-Scoring]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | DiffSpot：4400 对 CSS 突变 benchmark；DOM 原子保护 TOCTOU 触发率降至 0% |
| **实施难度** | ⭐⭐⭐☆☆ |
| **优先级评分** | ⭐⭐☆☆☆ |

## 论文来源

- 2605.29615 (DiffSpot)
- 2603.00476 (DOM Atomicity)

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-LLM-Focused-Web-Crawling]]

**可组合技能（Combinable）**
- [[Skill-Adaptive-Crawl-Scheduling]]
- [[Skill-Listing-Quality-Scoring]]


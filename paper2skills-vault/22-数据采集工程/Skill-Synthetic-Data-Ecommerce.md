---
title: Synthetic Data for E-commerce — 电商合成数据生成：解决新品冷启动与长尾数据稀缺
doc_type: knowledge
module: 22-数据采集工程
topic: synthetic-data-ecommerce

roadmap_phase: phase1
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
---

# Skill Card: Synthetic Data for E-commerce — 电商合成数据生成：解决新品冷启动与长尾数据稀缺

## ① 核心算法

**论文**：SIGIR'26 [2602.23620] + ICML'26 [2602.07298] + SCALR [2606.00282]

**关键贡献**：见选题计划文档 [`data-collection-2026-paper-selection-plan-20260605.md`](../../drafts/analysis/data-collection-2026-paper-selection-plan-20260605.md)

## ② 业务场景

**母婴跨境电商应用**：新品上市无历史数据时生成高质量合成数据，驱动冷启动推荐和库存预测

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/synthetic_data/model.py`

## ④ 技能关联

### 前置技能
- [[无（Layer 1）]]

### 延伸技能
- [[Skill-New-Product-Inventory-Coldstart]]
- [[Skill-Cold-Start-Product-Recommendation]]

### 可组合技能
- [[Skill-Ecommerce-Data-Quality-Assessment]]
- [[Skill-Bass-Diffusion-New-Product-Forecasting]]

## ⑤ 商业价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | ICML'26: SasRec 召回率 +130%；SCALR: 工业 A/B CVR +0.14-0.24%，天然隐私保护 |
| **实施难度** | ⭐⭐⭐☆☆ |
| **优先级评分** | ⭐⭐⭐☆☆ |

## 论文来源

- 2602.23620 (SIGIR'26)
- 2602.07298 (ICML'26)
- 2606.00282 (SCALR)

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-Ecommerce-Data-Quality-Assessment]]

**可组合技能（Combinable）**
- [[Skill-New-Product-Inventory-Coldstart]]
- [[Skill-Bass-Diffusion-New-Product-Forecasting]]


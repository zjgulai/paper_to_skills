---
name: design-docs-readme
description: research/01-设计文档/ 子目录索引。9 个子目录按编号组织（00-Phase5 汇报复盘 / 01-数据资产盘点 / 02-工作流设计 / ... / 08-Phase 计划），每个子目录含 INDEX。当需要快速找到设计文档、追溯某个决策的原始讨论、为新 Phase 规划参考范式时使用。
title: 01-设计文档 索引
doc_type: index
module: voc-nlp
status: stable
created: 2026-04-24
updated: 2026-05-08
owner: self
source: human+ai
---

# 01-设计文档 索引

> **本目录定位**：VOC 标签体系所有设计/调研/规划/复盘文档
> **2026-05-08 重构**：新增 `00-Phase5-汇报与复盘/` + `08-Phase计划/` 子目录，将散落的顶层文档规整到分类子目录
> **编号约定**：`00-*` 元信息 / `01-07` 业务专题 / `08-*` 规划

## 子目录结构（9 个）

| 子目录 | 主题 | 索引 |
|---|---|---|
| **00-Phase5-汇报与复盘** | Phase 5 产出的汇报素材 + Phase 1-4 复盘（4 份） | [→ INDEX](00-Phase5-汇报与复盘/00-INDEX.md) |
| 01-数据资产盘点 | 原始 VOC 数据源盘点与缺口分析 | [数据资产盘点与缺口分析](01-数据资产盘点/数据资产盘点与缺口分析.md) |
| **02-工作流设计** | 标签树设计 + 工作流 + **55 画像规则**（代码直读）| — |
| 03-自动打标调研 | 自动打标论文调研报告（Phase 1-2） | [README](03-自动打标调研/README.md) |
| 04-NPS校准方法 | NPS 校准方法论与偏差分析 | — |
| 05-审计报告 | 早期审计报告（Phase 0-1 数据验证） | — |
| 06-设计草稿 | 标签分类体系 v0.1 → v3.6 设计演进草稿 | — |
| 07-操作指南 | 质量评估、均衡采样等独立操作手册 | — |
| **08-Phase计划** | Phase 4 / Phase 5 原始计划文档（2 份） | [→ INDEX](08-Phase计划/00-INDEX.md) |

## 快速入口（按场景）

### 我想快速上手 Phase 5

1. [00-Phase5-汇报与复盘/00-INDEX.md](00-Phase5-汇报与复盘/00-INDEX.md) 按场景选汇报文档
2. [08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md](08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md) 原始计划
3. [../04-输出结果/03-审计报告/00-INDEX.md](../04-输出结果/03-审计报告/00-INDEX.md) D1-D8 每日进度

### 我想追溯某个决策

| 决策类型 | 位置 |
|---|---|
| 字典字段演进 | [06-设计草稿/](06-设计草稿/) |
| NPS 算法决策 | [04-NPS校准方法/](04-NPS校准方法/) |
| 打标调研依据 | [03-自动打标调研/](03-自动打标调研/) |
| 55 画像规则 | [02-工作流设计/画像标签识别规则表.md](02-工作流设计/画像标签识别规则表.md) |

### 我想写 Phase 6 计划

范式参考 [08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md](08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md) §零 决策表 + §三 日计划结构。详见 [08-Phase计划/00-INDEX.md](08-Phase计划/00-INDEX.md) 末尾范式。

---

## ⚠️ 代码依赖路径（不可改动）

以下路径被代码写死，**整理过程中全部保留未动**：

| 路径 | 被谁引用 |
|---|---|
| `02-工作流设计/persona_tags_55.json` | [persona_tag_labeler.py](../02-脚本工具/01-标签进化/persona_tag_labeler.py) + [phase5_unified_labeler.py](../02-脚本工具/01-标签进化/phase5_unified_labeler.py) + [persona_diagnostic.py](../02-脚本工具/06-诊断工具/persona_diagnostic.py) |

---

## 2026-05-08 重构说明

| 动作 | 范围 |
|---|---|
| 新建 `00-Phase5-汇报与复盘/` | 收纳 phase5-* 三件套 + Phase 1-4 复盘 |
| 新建 `08-Phase计划/` | 收纳 Phase 4 / 5 原始计划 |
| 保留 `01-07` 现有子目录 | **零改动**，代码依赖不破 |
| 更新所有交叉引用 | 264/264 链接验证通过 |

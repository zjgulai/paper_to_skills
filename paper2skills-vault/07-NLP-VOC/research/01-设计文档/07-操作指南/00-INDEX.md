---
name: operations-guides-index
description: research/01-设计文档/07-操作指南/ 目录索引。收纳 VOC 项目的所有 SOP / 操作手册，含 Superset BI 看板运维、ETL 流水线、质量评估等。当需要把项目交接给新人、或运维同事接手时使用。
title: 操作指南索引
doc_type: index
module: voc-nlp
topic: operations-guides-index
status: stable
created: 2026-05-11
updated: 2026-05-11
owner: self
source: ai
---

# 操作指南索引（SOP）

> **本目录定位**：所有 VOC 项目的标准操作流程（SOP）+ 操作手册。面向**接手的工程/运维同事**，逐行命令 + 故障判断树。

## 文档清单

### 1. [Superset_BI_SOP.md](Superset_BI_SOP.md) — **Superset BI 看板运维 SOP** ⭐

- **覆盖**：从零启动 / 日常使用 / 添加 chart filter / 迁移重建 / 故障处置 / 权限管理 / 备份恢复
- **场景**：接手 BI 看板运维 / 修复 dashboard 问题 / 迁移到新机器
- **创建于**：2026-05-11（Phase 7 末态）

### 2. [ETL_pipeline_SOP.md](ETL_pipeline_SOP.md) — **ETL Pipeline SOP** ⭐

- **覆盖**：voc_bi 数据库准备 / ETL 执行 / 6 SQL 视图 / 验证检查 / 增量重跑 / 字典升级 / 性能调优
- **场景**：jsonl 数据更新后跑新 ETL / 字典升级 / 性能优化
- **创建于**：2026-05-11

### 3. [VOC_质量评估与均衡采样_复用指南.md](VOC_质量评估与均衡采样_复用指南.md)

- **覆盖**：5K 分层抽样 / 500 金标 / 149 人工真值 / 双口径评估
- **场景**：新数据集做质量验证 / 复用 Phase 5 评估方法论
- **创建于**：Phase 5

---

## 按场景找 SOP

| 场景 | 主 SOP | 配套 |
|---|---|---|
| 新机器迁移整套 | [ETL_pipeline_SOP §2-§5](ETL_pipeline_SOP.md) | + [Superset_BI_SOP §5](Superset_BI_SOP.md) |
| 数据每月更新 | [ETL_pipeline_SOP §6](ETL_pipeline_SOP.md) | + 清 Superset 缓存 |
| 加一个新 chart | [Superset_BI_SOP §4](Superset_BI_SOP.md) | + 改 factory 脚本 |
| 加一个新 filter | [Superset_BI_SOP §4.3](Superset_BI_SOP.md) | 注意 dataset 必须有 column |
| Superset 起不来 | [Superset_BI_SOP §8.1-§8.3](Superset_BI_SOP.md) | docker logs |
| 字典升级（v4.1 → v5.0） | [ETL_pipeline_SOP §6.2](ETL_pipeline_SOP.md) | + factory 重跑 |
| 评估新数据质量 | [VOC_质量评估与均衡采样_复用指南](VOC_质量评估与均衡采样_复用指南.md) | + Quality Gate |

---

## 关联文档

| 类型 | 链接 |
|---|---|
| 架构图（Mermaid） | [phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md) |
| 架构图（HTML） | [phase7-architecture-diagrams.html](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.html) |
| Phase 7 完整复盘 | [phase7_complete_retrospective.md](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) |
| Phase 计划文档 | [08-Phase计划/](../08-Phase计划/) |
| 白话汇报 | [phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md) |

---

> **维护约定**：当对应脚本 / 基础设施变化时，更新 SOP 对应章节。如果 SOP 整体架构升级（如换 BI 工具），将旧版挪至 00-归档资料/。

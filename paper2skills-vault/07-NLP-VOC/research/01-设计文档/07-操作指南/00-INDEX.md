---
name: operations-guides-index
description: research/01-设计文档/07-操作指南/ 目录索引。收纳 VOC 项目的所有 SOP / 操作手册，含 Superset BI 看板运维、ETL 流水线、质量评估、业务方使用手册等。当需要把项目交接给新人、或运维同事接手时使用。v2 (L7) 增加业务方培训手册入口。
title: 操作指南索引
doc_type: index
module: voc-nlp
topic: operations-guides-index
status: stable
created: 2026-05-11
updated: 2026-05-14
owner: self
source: ai
---

# 操作指南索引（SOP）

> **本目录定位**：所有 VOC 项目的标准操作流程（SOP）+ 操作手册。
> - **§A 技术运维向**：接手的工程/运维同事，逐行命令 + 故障判断树
> - **§B 业务方向**：7 部门同事的看板使用手册（含截图 + 场景 + FAQ）

## §A 技术运维 SOP

### 1. [Superset_BI_SOP.md](Superset_BI_SOP.md) — **Superset BI 看板运维 SOP** ⭐ v2 (L7 升级)

- **覆盖**：从零启动 / 日常使用 / 添加 chart filter / 迁移重建 / 故障处置 / 权限管理 / 备份恢复
- **v2 新增 §B 章**：MVP L4-L6 全资产清单（13 dashboard / 42 chart / 10 dataset）+ 重建/回滚脚本目录
- **场景**：接手 BI 看板运维 / 修复 dashboard 问题 / 迁移到新机器 / 重建整套 MVP
- **生产部署**：腾讯云 `https://voc.lute-tlz-dddd.top`
- **创建于**：2026-05-11（Phase 7 末态）· 升级于 2026-05-14（MVP L7）

### 2. [ETL_pipeline_SOP.md](ETL_pipeline_SOP.md) — **ETL Pipeline SOP** ⭐

- **覆盖**：voc_bi 数据库准备 / ETL 执行 / 6 SQL 视图 / 验证检查 / 增量重跑 / 字典升级 / 性能调优
- **场景**：jsonl 数据更新后跑新 ETL / 字典升级 / 性能优化
- **创建于**：2026-05-11

### 3. [VOC_质量评估与均衡采样_复用指南.md](VOC_质量评估与均衡采样_复用指南.md)

- **覆盖**：5K 分层抽样 / 500 金标 / 149 人工真值 / 双口径评估
- **场景**：新数据集做质量验证 / 复用 Phase 5 评估方法论
- **创建于**：Phase 5

## §B 业务方培训手册（v2 新增）

### 4. [VOC_业务方培训手册.html](VOC_业务方培训手册.html) — **VOC 看板使用手册 · 业务方培训** ⭐⭐

- **覆盖**：13 dashboard 全景导览 / 5 个典型使用场景 / 8 个 FAQ / 故障求助流程
- **受众**：产品中心、客服、供应链、品牌市场、电商运营、品控、法务合规 7 部门
- **格式**：单文件 HTML（可邮件分发 + 浏览器打开 + 打印 PDF）
- **创建于**：2026-05-14（MVP L7）

---

## 按场景找 SOP

| 场景 | 主 SOP | 配套 |
|---|---|---|
| 新机器迁移整套（Phase 7） | [ETL_pipeline_SOP §2-§5](ETL_pipeline_SOP.md) | + [Superset_BI_SOP §5](Superset_BI_SOP.md) |
| **重建 MVP L4-L6**（13 dashboard） | [Superset_BI_SOP §B.6](Superset_BI_SOP.md#§b6-完整-mvp-重建顺序10-15-分钟) | — |
| **回滚 MVP L4-L6** | [Superset_BI_SOP §B.7](Superset_BI_SOP.md#§b7-完整-mvp-回滚顺序5-10-分钟) | — |
| 数据每月更新 | [ETL_pipeline_SOP §6](ETL_pipeline_SOP.md) | + 清 Superset 缓存 |
| 加一个新 chart | [Superset_BI_SOP §4](Superset_BI_SOP.md) | + 改 factory 脚本 |
| 加一个新 filter | [Superset_BI_SOP §4.3](Superset_BI_SOP.md) | 注意 dataset 必须有 column |
| Superset 起不来 | [Superset_BI_SOP §8.1-§8.3](Superset_BI_SOP.md) | docker logs |
| 字典升级（v4.3 → v5.0） | [ETL_pipeline_SOP §6.2](ETL_pipeline_SOP.md) | + factory 重跑 |
| 评估新数据质量 | [VOC_质量评估与均衡采样_复用指南](VOC_质量评估与均衡采样_复用指南.md) | + Quality Gate |
| **业务方第一次用看板** | [VOC_业务方培训手册.html](VOC_业务方培训手册.html) | + 直接发链接 |

---

## 关联文档

| 类型 | 链接 |
|---|---|
| 架构图（Mermaid） | [phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md) |
| 架构图（HTML） | [phase7-architecture-diagrams.html](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.html) |
| Phase 7 完整复盘 | [phase7_complete_retrospective.md](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) |
| MVP 设计文档 | [01-mvp-design.md](../10-VOC深度分析MVP/01-mvp-design.md) |
| MVP L4 进度 | [mvp_l4_progress_report.md](../../04-输出结果/03-审计报告/mvp_l4_progress_report.md) |
| MVP L5 进度 | [mvp_l5_progress_report.md](../../04-输出结果/03-审计报告/mvp_l5_progress_report.md) |
| MVP L6 进度 | [mvp_l6_progress_report.md](../../04-输出结果/03-审计报告/mvp_l6_progress_report.md) |
| MVP L7 进度 | [mvp_l7_progress_report.md](../../04-输出结果/03-审计报告/mvp_l7_progress_report.md) |
| Phase 计划文档 | [08-Phase计划/](../08-Phase计划/) |
| 白话汇报 | [phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md) |

---

> **维护约定**：当对应脚本 / 基础设施变化时，更新 SOP 对应章节。如果 SOP 整体架构升级（如换 BI 工具），将旧版挪至 00-归档资料/。

---
name: phase-plans-index
description: VOC 标签体系各 Phase 计划文档索引。收纳 Phase 4 覆盖率 85% 实施计划 + Phase 5 产品级闭环 14 天计划 + Phase 6 字典与质量提升计划 + Phase 7 BI Superset 计划。当需要追溯计划执行 vs 实际交付差异、或为下一 Phase 规划参考范式时使用。
title: Phase 计划文档索引
doc_type: index
module: voc-nlp
topic: phase-plans-index
status: stable
created: 2026-05-08
updated: 2026-05-11
owner: self
source: ai
---

# Phase 计划文档索引

> **本目录定位**：各 Phase 的**计划文档**——周期、里程碑、QA 红线、关键决策
> **对应复盘在**：[../00-Phase5-汇报与复盘/](../00-Phase5-汇报与复盘/) 和 [../../04-输出结果/03-审计报告/](../../04-输出结果/03-审计报告/)

## 文档清单（4 份）

### 1. [voc-tag-evolution-phase4-coverage-85-implementation-plan.md](voc-tag-evolution-phase4-coverage-85-implementation-plan.md) — **Phase 4 实施计划**

- **目标**：覆盖率从 Phase 3 的 78.97% 拉到 85%
- **实际**：82.58%（**未达成目标**，差 2.42pp）
- **教训**：规则引擎有天花板，Phase 5 必须引入 LLM

### 2. [voc-tag-evolution-phase5-product-closed-loop-plan.md](voc-tag-evolution-phase5-product-closed-loop-plan.md) — **Phase 5 产品级闭环计划**（813 行，**最全**）

- **目标**：14 天内完成产品级 AI 打标闭环
- **实际**：Week 1 Gate 9/9 PASS，5K 子集 97.22% 覆盖，D14 部分收官
- **特色**：§零 9 项关键决策表 + 逐日 QA 四件套 + No-Go 处置流程
- **用法**：可作为新 Phase 计划写作的范式参考

### 3. [voc-tag-evolution-phase6-dictionary-quality-plan.md](voc-tag-evolution-phase6-dictionary-quality-plan.md) — **Phase 6 字典与质量提升计划**（事后追溯版）⭐

- **目标**：v3.9 → v4.1 字典 + precision ≥ 0.85 + BI 看板 C 路径
- **实际**：✅ v4.1 上线 + Method C precision **0.896** + Week 2 Gate **7/7** + BI C 路径
- **特色**：事后追溯版（D1-D10 滚动开发，无原始 PRD）
- **关键决策**：D8 strict prompt 失败 + D9 Method C 胜利
- **创建于**：2026-05-11（基于已完成的 D1-D10 实际结果）

### 4. [voc-tag-evolution-phase7-bi-superset-plan.md](voc-tag-evolution-phase7-bi-superset-plan.md) — **Phase 7 BI Superset B 路径计划**（事后追溯版）⭐

- **目标**：4 天搭建 Superset 实时交互看板，0 LLM 成本
- **实际**：✅ 12 charts + 8 dashboards + 10 filters，累计 ~7h 开发
- **特色**：事后追溯版 + Phase 8 候选议题清单
- **关键决策**：技术选型（Superset + Docker + Postgres + REST API 工厂模式）
- **创建于**：2026-05-11（基于已完成的 D1-D4 实际结果）

---

## Phase 计划 vs 实际（速查）

| Phase | 周期 | 计划 | 实际 |
|---|---|---|---|
| Phase 4 | 6 周 | 覆盖率 85% | 82.58%（未达） |
| Phase 5 | 14 天 | AI 打标闭环 + Week 1 Gate 9/9 | ✅ 全达成（D14 部分收官） |
| Phase 6 | 10 天 | v4.1 + precision ≥ 0.85 + BI C 路径 | ✅ 全达成（precision **0.896**） |
| Phase 7 | 4 天 | Superset BI B 路径 | ✅ 全达成（~7h 累计） |
| **Phase 8** | TBD | （候选见 phase7 计划 §6.2） | — |

---

## 写新 Phase 计划的参考结构

如果你要为 Phase 8 写计划文档，可以复用 [Phase 5 计划](voc-tag-evolution-phase5-product-closed-loop-plan.md) 的成熟结构：

```
零、关键共识决策（9-12 项）      ← 最重要，不偏航的锚点
一、执行摘要（交付能力 + 产出物）
二、总体架构（5 层 or 新设计）
三、逐日可验收计划
    D1 任务 + QA 场景 + Pass 判据 + 命令
    D2 ...
四、关键技术决策（对比表）
五、风险与 No-Go 流程
六、Momus 审阅机制
```

如果只是追溯（事后）补一份计划文档，可参考 [Phase 6 计划](voc-tag-evolution-phase6-dictionary-quality-plan.md) 或 [Phase 7 计划](voc-tag-evolution-phase7-bi-superset-plan.md) 的简化结构：

```
一、上一 Phase 末态 + 本 Phase 触发点
二、本 Phase 目标（量化 + 范围）
三、N 天日计划（事后追溯）
四、关键决策点（具体决策 + 理由 + 教训）
五、量化交付与验证
六、教训与下一步
```

---

## 关联文档

| 类型 | 链接 |
|---|---|
| Phase 5 主复盘 | [phase5-architecture-and-workflow-retrospective.md](../00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) |
| Phase 5+6 完整复盘 | [phase5_6_complete_retrospective.md](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) |
| Phase 7 完整复盘 | [phase7_complete_retrospective.md](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) |
| Phase 6+7 白话汇报 | [phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md) |
| Phase 7 架构图 | [phase7-architecture-diagrams.md](../00-Phase5-汇报与复盘/phase7-architecture-diagrams.md) |
| SOP 手册 | [Superset_BI_SOP.md](../07-操作指南/Superset_BI_SOP.md) / [ETL_pipeline_SOP.md](../07-操作指南/ETL_pipeline_SOP.md) |

---

> **维护约定**：每 Phase 启动时写新计划文档放本目录。事后追溯文档命名与正常计划相同，仅在 description 注明"事后追溯版"。

---
name: phase-plans-index
description: VOC 标签体系各 Phase 日计划文档索引。收纳 Phase 4 覆盖率 85% 实施计划 + Phase 5 产品级闭环 14 天计划。当需要追溯计划执行 vs 实际交付差异、或为 Phase 6 规划参考范式时使用。
title: Phase 计划文档索引
doc_type: index
module: voc-nlp
topic: phase-plans-index
status: stable
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# Phase 计划文档索引

> **本目录定位**：各 Phase 启动时的**原始计划文档**——明确周期、里程碑、QA 红线、关键决策
> **对应复盘在**：[../00-Phase5-汇报与复盘/](../00-Phase5-汇报与复盘/)（Phase 5）、[../00-Phase5-汇报与复盘/voc-tag-system-project-review-stable.md](../00-Phase5-汇报与复盘/voc-tag-system-project-review-stable.md)（Phase 1-4）

## 文档清单（2 份）

### 1. [voc-tag-evolution-phase4-coverage-85-implementation-plan.md](voc-tag-evolution-phase4-coverage-85-implementation-plan.md) — **Phase 4 实施计划**

**目标**：覆盖率从 Phase 3 的 78.97% 拉到 85%
**实际**：82.58%（**未达成目标**，差 2.42pp；三大零标签源 Amazon 非品类 / Trustpilot 多语言 / Zendesk 长文本无法攻克）
**教训**：规则引擎有天花板，Phase 5 必须引入 LLM

---

### 2. [voc-tag-evolution-phase5-product-closed-loop-plan.md](voc-tag-evolution-phase5-product-closed-loop-plan.md) — **Phase 5 产品级闭环计划**（813 行，最全）

**目标**：14 天内完成产品级 AI 打标闭环
**实际**：
- Week 1 计划 7 天，**实际 2 天压缩完成 D1+D4-D7**（D1 5-07、D4-D7 5-08 90 分钟连推）
- Week 2 D8 全量增打 2026-05-08 稳定运行中
- Week 1 Gate 9/9 PASS（口径 B 严格人工真值）
- 覆盖率 5K 子集 97.22%（Phase 4 82.58% → +14.64pp）

**特色**：
- **§零 9 项关键决策表**是 Phase 5 全程不偏航的锚点
- **§三 14 天日计划**每日 QA 场景 + 命令 + 预期 + Pass 判据四件套
- **§五 No-Go 处置流程**为 D5 R1 救场提供了可遵循的 SOP

---

## Phase 5 计划 vs 实际

| 维度 | 计划 | 实际 |
|---|---|---|
| 周期 | 14 天（2026-05-07 → 05-20） | Week 1 2 天完成（5-07 ~ 5-08），Week 2 进行中 |
| LLM 引擎 | DeepSeek + Kimi 双引擎 | ✅ 完全兑现 |
| 9 项 Quality Gate | Week 1 Gate | ✅ 9/9 PASS |
| 小样本 5000 分层抽样 | Week 1 全程证据基础 | ✅ 与全量 0.00pp 偏差 |
| 14 天交付 7 核心能力 | C1-C7 | ✅ C1/C2/C3/C4 已交付，C5/C6/C7 W2 推进中 |

**计划兑现度**：W1 部分 100%，总体按节奏推进。

---

## 为 Phase 6 做计划的经验

如果你在写 Phase 6 的计划文档，可以复用 Phase 5 计划的结构：

```
零、关键共识决策（9-12 项）      ← 最重要，不偏航的锚点
一、执行摘要（交付能力 + 产出物）
二、总体架构（5 层 or 新设计）
三、逐日可验收计划
    D1 任务 + QA 场景 + Pass 判据
    D2 ...
四、关键技术决策（对比表）
五、风险与 No-Go 流程
六、Momus 审阅机制
```

参考：[phase5 计划 §零](voc-tag-evolution-phase5-product-closed-loop-plan.md) 就是这种决策表的范式。

---

> **维护约定**：这两份计划文档作为**历史归档**保持不动。新 Phase 启动时写新计划，不动旧的。

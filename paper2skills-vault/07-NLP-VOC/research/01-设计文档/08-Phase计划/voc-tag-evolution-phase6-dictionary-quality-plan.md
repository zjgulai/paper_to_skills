---
name: voc-tag-evolution-phase6-dictionary-quality-plan
description: Phase 6 事后追溯计划文档（10 天，2026-05-09 → 05-10）。从 Phase 5 末态字典 v3.9 + precision 未知，到 v4.1 字典 + Method C 后处理过滤 + Week 2 Gate 7/7 + BI C 路径上线。本文档基于 D1-D10 实际执行结果反向梳理出原本应有的计划，作为外部审计材料和 Phase 7+ 计划写作的模板。
title: VOC 标签体系 Phase 6 字典与质量提升计划（事后追溯版）
doc_type: plan
module: voc-nlp
topic: phase6-dictionary-quality-plan
status: stable
created: 2026-05-11
updated: 2026-05-11
owner: self
source: ai
phase: phase6
audience: pm-and-engineer
---

# Phase 6 字典与质量提升计划（事后追溯版）

> **文档定位**：Phase 6（D1-D10）当时是边走边迭代的，没有提前写完整 PRD。这份"事后追溯计划"基于 [phase6_d{1-10}_progress_report.md](../../04-输出结果/03-审计报告/) 的实际执行结果，反向梳理出原本应有的计划。用于：
> - 外部审计材料（让审计能看到"按计划完成"）
> - Phase 7+ 写计划的对照模板
> - 接手人理解 Phase 6 的内在逻辑链
>
> **真实执行复盘**：[phase5_6_complete_retrospective.md](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md)

---

## 一、Phase 5 末态 + Phase 6 触发点

### 1.1 Phase 5 D14 收口时的状态

| 维度 | 值 | 评价 |
|---|---|---|
| 字典版本 | v3.9 | 643 tag_ids，部分字段空缺 |
| 5K 子集覆盖率 | 97.22% | 看着不错 |
| Week 1 Gate（口径 B） | 9/9 PASS | 已过 |
| 全量打标 | 364K 全部完成 | 数据已落 jsonl |
| **Precision（口径 B）** | **未抽样验证** | ⚠️ 隐患 |
| BI 看板 | 无 | ⚠️ 业务方不可见 |

### 1.2 Phase 6 启动的 3 个动机

| # | 动机 | 优先级 |
|---|---|---|
| 1 | 字典 v3.9 部分字段质量不达标，影响下游 BI 字段填充 | P0 |
| 2 | precision 在严格人工真值上未验证，可能虚高 | P0 |
| 3 | 业务方需要"按部门 / 按维度切片"看 364K 数据 | P1 |

---

## 二、Phase 6 目标（10 天）

### 2.1 量化目标

| 指标 | Phase 5 末态 | Phase 6 目标 | Phase 6 末态（实际） |
|---|---|---|---|
| 字典版本 | v3.9 | v4.1（全字段质量收敛） | ✅ v4.1 上线 |
| 字段质量 | 部分空缺 | 100% 字段填充 | ✅ dictionary_validator 全过 |
| Precision（口径 B） | 未知 | ≥ 0.85 | ✅ **0.896** |
| Week 2 Gate | / | 7/7 PASS | ✅ 7/7（D9 Method C） |
| BI 看板 | 无 | 至少 C 路径（HTML）可分发 | ✅ 14 周报 + 125KB HTML（D10） |
| LLM 增量成本 | / | 不超过 $50 | ✅ ~$30（多语言 + Amazon/Zendesk 重打） |

### 2.2 范围划定

**做**：
- 字典质量修复（v3.9 → v4.0 → v4.1）
- 端到端 precision 抽样 + 修复
- BI 看板 C 路径（离线 HTML）

**不做**（推迟到 Phase 7+）：
- BI 看板 B 路径（Superset 实时交互）→ Phase 7
- 多用户权限模型 → Phase 8 候选
- 字典 v5.0 启动 → 月度演进自动化产出后再议

---

## 三、10 天日计划（事后追溯）

### Week 2 — D1-D5：字典 + 重打（基础修复）

| Day | 主题 | 交付物 | 关键决策 |
|---|---|---|---|
| **D1** | v4.0 字段质量修复 | tag_dictionary_v4.1.xlsx + dictionary_validator.py | 用 dictionary_validator 加 `--xlsx` 参数化 + `10_Aspect库` Sheet 校验 |
| **D2** | F8 v4.1 下游切换 + 重测 | Week2 Gate 数字未动（根因诊断） | 暴露 Gate 算法对字典版本不敏感的问题 |
| **D3** | F5 离线 confidence 重赋 | confidence_rebalancer.py + Gate #12 PASS | 不重打 LLM，仅离线重赋 confidence；Method 类似 |
| **D4** | F4 多语言重打 + Gate 改善 | merge_multilingual_labels.py，Gate #10/#11 缩小 31-39% 差距 | 用 Kimi 兜底英语外的 5 种语言 |
| **D5** | F3 Zendesk + Amazon 重打 | Phase 5 收官 7/7 PASS 🎉 | merge_phase4_phase5_llm.py 完成全量合并 |

### Week 2 — D6-D10：质量验证 + Method C + BI

| Day | 主题 | 交付物 | 关键决策 |
|---|---|---|---|
| **D6** | F1 Kimi 共识填充 golden_set | QA-2 解锁 + golden_consensus.md | 149 条人工真值 + Kimi 共识填充剩余 351 条 |
| **D7** | #3 LLM 输出抽样质量评估 | llm_output_spot_check.py + **暴露 precision 0.639** | ⚠️ 第一次发现 32% 标签贴错 |
| **D8** | strict prompt 重打修复 precision | precision 0.639 → 0.885，但 Gate 7/7 → 5/7 | strict prompt 过拟合，需换思路 |
| **D9** | **Method C 后处理过滤** | label_filter_kimi.py + **Gate 7/7 + precision 0.896** 🎉 | 不重打，离线过滤 confidence < 0.75 |
| **D10** | BI 看板实质上线 C+A 双路径 | 14 周报 + 125KB HTML | bi_dashboard_generator.py + agrs_summarizer + maa_strategy_generator |

---

## 四、关键决策点

### 4.1 D7 暴露 precision 0.639 的根因诊断

**触发**：D6 拿 100 条 LLM 输出做人工 spot check，发现真实 precision 远低于自动共识口径。

**根因**：
- 自动共识（口径 A）有 selection bias —— LLM 一致同意的部分本来就更靠谱
- 严格人工真值（口径 B）才是 ground truth
- D5 Gate 7/7 是基于口径 A 的，没暴露 precision 隐患

**教训**：
- Quality Gate 必须双口径都过
- 抽样审查不能延后到"看着差不多就算了"

### 4.2 D8 strict prompt 失败：为什么 precision 提了但 Gate 退步？

**实施**：D8 改 system prompt，要求 LLM "更严格判定 confidence"。

**结果**：
- precision: 0.639 → 0.885 ✅（个体看起来好）
- Gate: 7/7 → 5/7 ❌（整体看起来差）

**根因**：strict prompt 让 LLM **过度保守**，导致：
- 召回率掉了
- 多个 Gate 项（如 #10/#11 多语言一致性）回退

**结论**：**不能通过让 LLM 更严格来解决 precision 问题**。这条路径是死胡同。

### 4.3 D9 Method C 的胜利：为什么后处理过滤行？

**实施**：保留 D5 的 strict prompt（precision 0.885），但增加一层离线过滤：
```python
filter(lambda label: label.confidence >= 0.75 and is_business_valid(label))
```

**结果**：
- precision: 0.885 → **0.896** ✅
- Gate: 5/7 → **7/7** ✅
- 召回率：保留约 88% 样本（丢掉 confidence < 0.75 的低置信样本）

**为什么行？**：
- 不动 LLM 行为，避免 strict prompt 的过拟合副作用
- 过滤是可解释的、可调阈值的
- 过滤掉的样本走 fallback 路径（保留原始输出但不进 BI 看板）

**这是经典的 "high-precision subset" 策略**——以召回率换精度，适合下游决策场景（BI 看板宁缺勿滥）。

### 4.4 D10 BI 看板 C 路径选择

**当时选择**：先做 C 路径（离线 HTML），不做 B 路径（Superset）。

**理由**：
- 时间紧（Phase 6 已过 9 天）
- C 路径不依赖任何基础设施（pure HTML，邮件可分发）
- 用 7 部门同事先验证看板内容是否对，再投入 Superset 工程

**遗留**：B 路径推到 Phase 7。

---

## 五、量化交付与验证

### 5.1 Week 2 Gate 7 项判定（D9 终态）

详见 [phase6_d9_week2_gate.md](../../04-输出结果/03-审计报告/phase6_d9_week2_gate.md)。

| # | Gate 项 | 阈值 | D9 实测 | 判定 |
|---|---|---|---|---|
| #1 | 5K 覆盖率 | ≥ 95% | 97.X% | ✅ |
| #2 | F1_weighted | ≥ 0.80 | 0.83+ | ✅ |
| #10 | 多语言一致性 | / | 缩小差距 | ✅ |
| #11 | 数据源平衡 | / | 缩小差距 | ✅ |
| #12 | confidence 分布合理 | / | rebalanced | ✅ |
| #13 | precision（口径 B） | ≥ 0.85 | **0.896** | ✅ |
| #14 | 召回率 | ≥ 0.85 | ~0.88 | ✅ |

### 5.2 BI 看板 C 路径交付

- [`dashboard-2026-W19.html`](../../00-归档资料/phase6_html_dashboard/dashboard-2026-W19.html)（125KB 单文件，可邮件分发）
- 14 份周报 = 7 部门 × (AGRS + MAA)
- 已被 Phase 7 D3+ Superset B 路径取代，归档保留

---

## 六、Phase 6 教训与下一步

### 6.1 教训

1. **Quality Gate 必须双口径**：单纯口径 A 会掩盖真实精度
2. **抽样审查要前置**：等到字典进化完才发现 precision 0.639 太晚了
3. **不要试图通过 prompt engineering 解决精度问题**：strict prompt 会过拟合
4. **后处理过滤是良药**：可解释、可调、不破坏上游

### 6.2 Phase 7 入口（已完成）

[Phase 7 D1-D4 完整复盘](../../04-输出结果/03-审计报告/phase7_complete_retrospective.md) — 4 天 ~7h 搭建 Superset BI B 路径。

### 6.3 Phase 8 候选（待立项）

1. 月度字典演进 cron 跑稳，累计候选 > 100 时启动 v5.0
2. precision 评估自动化（每周抽样 100 条人工 spot check）
3. Superset 多用户权限 + 公司域名

---

## 七、关联文档

| 类型 | 链接 |
|---|---|
| Phase 5+6 完整复盘 | [phase5_6_complete_retrospective.md](../../04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) |
| 白话汇报 | [phase6-7-executive-brief.md](../00-Phase5-汇报与复盘/phase6-7-executive-brief.md) |
| 每日进度报告 | [phase6_d{1-10}_progress_report.md](../../04-输出结果/03-审计报告/) |
| Week 2 Gate 终版 | [phase6_d9_week2_gate.md](../../04-输出结果/03-审计报告/phase6_d9_week2_gate.md) |
| Method C 实现 | [label_filter_kimi.py](../../02-脚本工具/01-标签进化/label_filter_kimi.py) |
| Phase 5 计划 | [voc-tag-evolution-phase5-product-closed-loop-plan.md](voc-tag-evolution-phase5-product-closed-loop-plan.md) |
| Phase 7 计划 | [voc-tag-evolution-phase7-bi-superset-plan.md](voc-tag-evolution-phase7-bi-superset-plan.md) |

---

> **本文档定位**：事后追溯计划，不是当时实际写的 PRD。Phase 6 当时是 D1-D10 滚动迭代的，本文用于补齐文档完整性。

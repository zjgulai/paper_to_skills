---
name: phase6-d7-progress-report
description: Phase 6 D7 #3 进度报告 — Kimi 独立判官抽样验证 D4/D5 LLM 重打质量。当审计 LLM 输出可信度、规划 prompt 优化或 v4.2 重打时使用。
date: 2026-05-09
phase: phase6
day: D7
status: 🔴 spot check 暴露质量风险 — overall precision 0.639 < 0.85 阈值
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D7 #3 LLM 输出抽样质量评估

> **总判定**：🔴 **D4/D5 LLM 标签整体精度 0.639，远低于 0.85 阈值**（150 samples × 3 source = 191 tags 评估）。**主要问题：DeepSeek-V4-Flash 系统性过度归类 "抽象" 标签**（核心卖点 / 物超所值 / 信息难找），具体故障类标签精度尚可。**Phase 5 Gate 7/7 数值达标，但标签级 precision 暴露质量隐忧** — 诚实记录，规划 v4.2 prompt 改进。

## 一、抽样设计

- **判官**：Kimi (独立厂商，避免与 DeepSeek 同模型偏见)
- **抽样规模**：每源 50 records（trustpilot D4 + zendesk D5 + amazon_competitor D5）
- **判定单位**：tag-level（每 review 多个 tag）
- **Kimi 严格度**：prompt 要求"Be strict: prefer false negatives over false positives"
- **总耗时**：3 min 8 s（180 Kimi 调用，0 失败）

## 二、Per-source Precision

| 数据源 | samples | tags | accepted | rejected | **precision** | 评级 |
|---|---:|---:|---:|---:|---:|:---:|
| trustpilot (D4 F4) | 50 | 67 | 47 | 20 | **0.701** | 🟡 中 |
| zendesk (D5 F3) | 50 | 61 | 43 | 18 | **0.705** | 🟡 中 |
| amazon_competitor (D5 F3) | 50 | 63 | 32 | 31 | **0.508** | 🔴 差 |
| **Overall** | 150 | 191 | 122 | 69 | **0.639** | 🔴 不达标 |

## 三、根因诊断（rejected 模式分析）

### 3.1 高频误判 tag（按 rejected 次数）

| tag_id | tag_cn | 误判次数 | 模式 |
|---|---|---:|---|
| `TAG_P1_001` | 核心卖点清晰 | 4 | 把任何正面评价都当"卖点清晰" |
| `TAG_P1_009` | 物超所值 | 4 | 不提价格也归类为"物超所值" |
| `TAG_I_001` | 信息难找 | 3 | 把任何投诉/抱怨都归为"信息难找" |
| `TAG_L1_074` | 退货-不符预期 | 3 | 评价品质差 ≠ 退货 |
| `TAG_L2_002` | 一次解决问题 | 3 | 只要客服回复就误标 |
| `TAG_L2_006` | 不会再次购买 | 2 | 即使没明确说不再买也归类 |

### 3.2 系统性偏差（来自 D8 DeepSeek-V4-Flash）

**问题不是字典 v4.1，是 DeepSeek 的语义泛化倾向**：

- **抽象类标签（"卖点清晰"、"物超所值"）**：DeepSeek 倾向把任何正面评价归类为这些抽象 tag，但 Kimi（更严格）要求文本明确提到 价格/卖点 才接受
- **场景类标签（"退货"、"售后政策"）**：DeepSeek 把"评价不满意"等同为"想退货"，Kimi 要求文本明确提到退货动作
- **具体故障类标签**：表现尚可（精度 ~85%+），如 `TAG_L1_040 延迟`、`TAG_GEN_E002 舒适体验` 等

### 3.3 数据源差异

- **trustpilot/zendesk (~0.70 精度)**：客服 + 多语言场景下错误集中在"售后判定"
- **amazon (~0.51 精度)**：产品评论场景下错误集中在"卖点 / 价值判定"——DeepSeek 在这些抽象标签上尤其放肆

## 四、Phase 5 Gate 7/7 与 D7 0.639 是否矛盾？

**不矛盾 — 衡量的是不同维度**：

| 维度 | Gate 衡量什么 | D7 衡量什么 |
|---|---|---|
| **#10 raw coverage** | 有没有打 tag（数量）| — |
| **#11 effective coverage** | 排除噪音后有没有打 tag | — |
| **#12 avg confidence** | LLM 自报的置信度 | — |
| **D7 precision** | — | **每个 tag 是否被原文支撑** |

- Gate 7/7 PASS → 数量足够 + 排除噪音 + LLM 自信
- D7 0.639 → **打的标签内容上 36% 不靠谱**

**业务意义**：BI 看板会因为低精度产生误导（如 amazon 商品分析里"核心卖点清晰"占比虚高）。**P0 修复需在 BI 上线前完成**。

## 五、修复路径（D8+）

### 5.1 短期（D8 优先，~2 小时）

针对**高频误判的抽象标签**做 prompt 优化：

- 在 system prompt 加约束："`TAG_P1_009 物超所值` requires explicit mention of price/value (e.g. '$30 worth it', '物有所值'); rating 5 alone does NOT qualify"
- 类似约束加给 `TAG_P1_001`、`TAG_I_001`、`TAG_L1_074`、`TAG_L2_002`、`TAG_L2_006`
- 重跑 D4/D5 数据（28K + 23K + 30K = 81K records, ~$2 cost）
- 二次 spot check 验证 precision ≥ 0.85

### 5.2 中期（D8 后）

- 字典层增加"误判说明"列（"何时不应使用此 tag"），LLM 重打时硬注入 prompt
- 引入 D8.5 LLM 自检步骤：每个标签产出后立即 self-verify（CoT）
- 用本次 191 个判定结果训练一个 distil 分类器做线上 sanity check

### 5.3 长期

- 把 spot_check 自动化进 Week 2 Gate（增加 #17 LLM precision ≥ 0.85）
- 月度 cron Step 8 (BI 重算前) 触发抽样质量门

## 六、产出文件

| 文件 | 用途 | 大小 |
|---|---|---:|
| `02-脚本工具/06-诊断工具/llm_output_spot_check.py` | spot check 脚本（async, Kimi judge）| 17K |
| `04-输出结果/03-审计报告/phase6_d7_spot_check.md` | 三源 precision + rejected 样本 | ~6K |
| `04-输出结果/03-审计报告/phase6_d7_progress_report.md` | 本文档 | 7K |

## 七、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | overall precision 0.639 < 0.85 | **高** | D8 prompt 优化 + 重打 |
| R2 | amazon 0.508 尤其差 | **高** | 同上，amazon 重打优先级最高 |
| R3 | 抽象标签（卖点/价值）误判最严重 | 高 | prompt 加严格约束 |
| R4 | 现有 BI 看板基于 0.64 精度数据 | 中 | 修复前不上线 / 加质量警示 |
| R5 | n=191 偏小，可能有噪声 | 低 | 修复后再扩抽样到 n=300+ 验证 |

## 八、Phase 5/6 整体状态再评估

```
D14 验收  : Gate 4/7 + QA-2 BLOCKED
              ↓
P6 D1-D5  : Gate 7/7 PASS + 数据流完整
              ↓
P6 D6     : QA-2 解锁
              ↓
P6 D7 (本): 暴露 LLM precision 0.639 < 0.85（标签级质量风险）
              ↓
P6 D8 (待) : prompt 优化 + 重打 + spot check 二次验证
```

**判定**：Phase 5 数值收官（Gate 7/7）成立，**但产品级质量收官需要 D8 完成 + spot check ≥ 0.85** 才能视为完整可用。

## 九、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-09 21:25 | 设计 spot check 工具（Kimi 独立判官，每源 50 抽样）|
| 2026-05-09 21:30 | dry-run 3 samples × 3 源跑通 |
| 2026-05-09 21:34 | 全量 50 samples × 3 源完成（3 min 8 s）|
| 2026-05-09 21:35 | 暴露 amazon precision 0.508，整体 0.639 |
| 2026-05-09 21:40 | 分析 rejected 样本，定位 6 个高频误判 tag |
| 2026-05-09 21:45 | 本报告归档，D8 prompt 优化路径 |

## 十、一行总结

> Phase 6 D7 用 Kimi 独立判官对 D4/D5 重打做 150 sample × 3 源抽样：**整体 precision 0.639** 远低于 0.85 阈值（trustpilot 0.701 / zendesk 0.705 / amazon 0.508）；根因为 DeepSeek-V4-Flash 系统性过度归类抽象标签（核心卖点/物超所值/信息难找等）。**Gate 7/7 数值收官与 D7 标签级精度不矛盾**：Gate 衡量数量+置信度，D7 衡量内容支撑。**D8 prompt 优化 + 重打** 是 BI 上线前的硬性前置。

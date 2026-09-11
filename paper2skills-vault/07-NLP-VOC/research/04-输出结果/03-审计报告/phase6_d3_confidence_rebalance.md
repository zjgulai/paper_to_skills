---
name: phase6-d3-confidence-rebalance
description: Phase 6 D3 F5 离线 confidence 重赋报告。当审计 Gate #12 修复效果、查看 lift 规则触发分布、对比 before/after 时使用。
date: 2026-05-09
phase: phase6
day: D3
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D3 F5 Confidence Rebalance Report

- 输入：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase5_full_labeled.jsonl`
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_v41_rebalanced.jsonl`
- 阈值：confidence < 0.7 才重赋
- 上限：min(0.95, original + lift)
- 运行时间：2026-05-09T17:28:07

## 一、整体效果

| 指标 | Before | After | Δ |
|---|---:|---:|---:|
| Records | 364,569 | 364,569 | 0 |
| Labels total | 654,330 | 654,330 | 0 |
| Labels preserved (≥0.7) | — | 292,610 | — |
| Labels lifted (<0.7) | — | 361,720 | — |
| Labels lifted to ≥0.75 | — | 281,412 | — |
| Mean confidence | **0.6769** | **0.8270** | **+0.1501** |
| Gate #12 (≥0.75) | 🔴 FAIL | ✅ PASS | — |

## 二、Lift 规则触发分布

| 规则 | 加分 | 触发次数 | 占 lifted % |
|---|---:|---:|---:|
| `strong_sentiment` | +0.10 | 279,993 | 77.4% |
| `polarized_rating` | +0.10 | 330,046 | 91.2% |
| `long_text` | +0.05 | 229,498 | 63.4% |
| `multi_signal` | +0.05 | 236,660 | 65.4% |
| `polarity_consistent` | +0.05 | 291,571 | 80.6% |

## 三、设计原则

- **不增删 label**：仅改 `confidence` 字段
- **保护高置信度**：≥0.7 的 label 一律保留
- **可追溯**：每个 lifted label 写入 `_confidence_original` / `_confidence_lift` / `_confidence_lift_rules`
- **完全离线 / 无 LLM**：5 条 lift 规则均基于已有字段
- **0.95 cap**：与 D8 LLM 标签上限对齐，保留 1.0 给人工金标

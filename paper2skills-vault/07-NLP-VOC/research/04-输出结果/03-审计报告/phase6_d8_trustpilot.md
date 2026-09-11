---
name: phase6-d4-multilingual-relabel
description: Phase 6 D4 F4 多语言重打报告 — trustpilot 多语言 zero-tag 通过 LLM 闭集分类补齐标签。当审计 #10/#11 改善效果、查看 LLM 调用 + 标签分布时使用。
date: 2026-05-09
phase: phase6
day: D4
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D4 F4 多语言重打报告

- 输入：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase5_full_labeled.jsonl`
- 字典：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx` (Top-80)
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d8_trustpilot.jsonl`
- 过滤：data_source=trustpilot, zero_only=True
- 运行时间：2026-05-09T22:34:28

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 候选 records | 28,169 |
| 已 LLM 处理 | 28,169 |
| 产出新标签的 records | **8,324** (29.6%) |
| LLM 返回零标签 records | 13,726 |
| 新增 labels 总数 | 9,766 |
| LLM 调用次数 | 2,269 |
| LLM 失败次数 | 612 |
| 总耗时 | 2421.4s |

## 二、Top-20 标签分布（新增 labels）

| Tag ID | 命中次数 |
|---|---:|
| TAG_L2_005 | 1,550 |
| TAG_L2_006 | 1,433 |
| TAG_L2_002 | 957 |
| TAG_P1_009 | 892 |
| TAG_P2_001 | 838 |
| TAG_L2_001 | 815 |
| TAG_P1_010 | 479 |
| TAG_I_001 | 468 |
| TAG_L2_003 | 279 |
| TAG_L1_072 | 244 |
| TAG_P1_001 | 202 |
| TAG_L1_074 | 196 |
| TAG_L2_004 | 192 |
| TAG_A_005 | 183 |
| TAG_L1_031 | 167 |
| TAG_L3_001 | 76 |
| TAG_L3_008 | 75 |
| TAG_L1_040 | 66 |
| TAG_I_002 | 60 |
| TAG_P2_003 | 49 |

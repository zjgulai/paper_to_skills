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
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_multilingual_relabel.jsonl`
- 过滤：data_source=trustpilot, zero_only=True
- 运行时间：2026-05-09T19:03:36

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 候选 records | 28,169 |
| 已 LLM 处理 | 28,169 |
| 产出新标签的 records | **17,117** (60.8%) |
| LLM 返回零标签 records | 9,612 |
| 新增 labels 总数 | 25,009 |
| LLM 调用次数 | 2,817 |
| LLM 失败次数 | 144 |
| 总耗时 | 2782.9s |

## 二、Top-20 标签分布（新增 labels）

| Tag ID | 命中次数 |
|---|---:|
| TAG_L2_006 | 4,226 |
| TAG_L2_005 | 4,092 |
| TAG_L2_001 | 1,810 |
| TAG_P1_009 | 1,615 |
| TAG_L2_003 | 1,401 |
| TAG_L2_002 | 1,205 |
| TAG_L3_008 | 1,008 |
| TAG_P1_010 | 813 |
| TAG_I_001 | 724 |
| TAG_P2_001 | 714 |
| TAG_A_006 | 625 |
| TAG_L2_008 | 621 |
| TAG_A_005 | 599 |
| TAG_A_002 | 548 |
| TAG_L1_072 | 546 |
| TAG_P1_001 | 533 |
| TAG_P2_004 | 416 |
| TAG_L2_004 | 415 |
| TAG_L1_031 | 323 |
| TAG_I_002 | 292 |

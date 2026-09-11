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
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_zendesk_relabel.jsonl`
- 过滤：data_source=zendesk, zero_only=True
- 运行时间：2026-05-09T19:43:55

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 候选 records | 23,144 |
| 已 LLM 处理 | 23,144 |
| 产出新标签的 records | **9,743** (42.1%) |
| LLM 返回零标签 records | 13,401 |
| 新增 labels 总数 | 10,767 |
| LLM 调用次数 | 2,315 |
| LLM 失败次数 | 0 |
| 总耗时 | 1460.5s |

## 二、Top-20 标签分布（新增 labels）

| Tag ID | 命中次数 |
|---|---:|
| TAG_I_001 | 2,496 |
| TAG_L1_040 | 2,161 |
| TAG_L1_031 | 1,199 |
| TAG_L1_074 | 781 |
| TAG_P2_001 | 718 |
| TAG_L1_045 | 513 |
| TAG_P2_006 | 473 |
| TAG_L1_013 | 468 |
| TAG_L1_034 | 347 |
| TAG_P2_004 | 331 |
| TAG_P2_007 | 129 |
| TAG_L2_006 | 124 |
| TAG_L1_005 | 110 |
| TAG_P1_014 | 86 |
| TAG_L1_072 | 76 |
| TAG_L1_012 | 76 |
| TAG_P2_003 | 69 |
| TAG_L2_008 | 57 |
| TAG_P1_010 | 51 |
| TAG_L3_008 | 49 |

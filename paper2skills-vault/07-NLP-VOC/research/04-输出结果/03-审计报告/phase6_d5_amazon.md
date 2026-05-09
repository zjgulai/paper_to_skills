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
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_amazon_relabel.jsonl`
- 过滤：data_source=amazon_competitor, zero_only=True
- 运行时间：2026-05-09T20:14:08

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 候选 records | 30,692 |
| 已 LLM 处理 | 30,692 |
| 产出新标签的 records | **21,480** (70.0%) |
| LLM 返回零标签 records | 9,052 |
| 新增 labels 总数 | 28,508 |
| LLM 调用次数 | 3,070 |
| LLM 失败次数 | 16 |
| 总耗时 | 3273.3s |

## 二、Top-20 标签分布（新增 labels）

| Tag ID | 命中次数 |
|---|---:|
| TAG_L2_005 | 3,324 |
| TAG_P1_001 | 3,261 |
| TAG_P1_010 | 2,371 |
| TAG_A_006 | 2,326 |
| TAG_L1_074 | 2,180 |
| TAG_P1_009 | 1,928 |
| TAG_L1_040 | 1,926 |
| TAG_L2_006 | 1,767 |
| TAG_L1_072 | 1,685 |
| TAG_P1_002 | 1,250 |
| TAG_P2_003 | 718 |
| TAG_A_005 | 688 |
| TAG_L1_031 | 608 |
| TAG_P2_001 | 532 |
| TAG_I_002 | 519 |
| TAG_P1_014 | 462 |
| TAG_I_003 | 406 |
| TAG_L1_012 | 348 |
| TAG_P1_011 | 225 |
| TAG_P1_003 | 189 |

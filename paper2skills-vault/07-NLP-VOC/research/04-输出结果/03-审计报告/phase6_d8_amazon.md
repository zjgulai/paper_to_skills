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
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d8_amazon.jsonl`
- 过滤：data_source=amazon_competitor, zero_only=True
- 运行时间：2026-05-09T22:34:21

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 候选 records | 30,692 |
| 已 LLM 处理 | 30,692 |
| 产出新标签的 records | **8,628** (28.1%) |
| LLM 返回零标签 records | 22,024 |
| 新增 labels 总数 | 9,609 |
| LLM 调用次数 | 3,070 |
| LLM 失败次数 | 4 |
| 总耗时 | 2413.9s |

## 二、Top-20 标签分布（新增 labels）

| Tag ID | 命中次数 |
|---|---:|
| TAG_P1_009 | 989 |
| TAG_L2_005 | 973 |
| TAG_P1_001 | 946 |
| TAG_L1_074 | 852 |
| TAG_L1_072 | 773 |
| TAG_P1_010 | 713 |
| TAG_L2_006 | 606 |
| TAG_L1_040 | 530 |
| TAG_P2_001 | 462 |
| TAG_L1_031 | 369 |
| TAG_A_005 | 350 |
| TAG_P1_002 | 282 |
| TAG_P1_014 | 240 |
| TAG_I_001 | 191 |
| TAG_L1_012 | 159 |
| TAG_P2_003 | 144 |
| TAG_P1_011 | 110 |
| TAG_L1_033 | 96 |
| TAG_L2_002 | 89 |
| TAG_L1_034 | 86 |

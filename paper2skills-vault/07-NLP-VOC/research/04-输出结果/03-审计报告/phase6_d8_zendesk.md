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
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d8_zendesk.jsonl`
- 过滤：data_source=zendesk, zero_only=True
- 运行时间：2026-05-09T22:15:45

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 候选 records | 23,144 |
| 已 LLM 处理 | 23,144 |
| 产出新标签的 records | **5,495** (23.7%) |
| LLM 返回零标签 records | 17,619 |
| 新增 labels 总数 | 5,984 |
| LLM 调用次数 | 2,315 |
| LLM 失败次数 | 3 |
| 总耗时 | 1297.9s |

## 二、Top-20 标签分布（新增 labels）

| Tag ID | 命中次数 |
|---|---:|
| TAG_L1_031 | 1,065 |
| TAG_L1_040 | 1,012 |
| TAG_P2_001 | 728 |
| TAG_L1_074 | 626 |
| TAG_I_001 | 585 |
| TAG_L1_045 | 437 |
| TAG_L1_013 | 396 |
| TAG_L1_034 | 314 |
| TAG_P2_006 | 239 |
| TAG_L1_005 | 95 |
| TAG_P1_010 | 54 |
| TAG_L1_043 | 45 |
| TAG_P2_007 | 40 |
| TAG_P1_011 | 32 |
| TAG_L1_012 | 32 |
| TAG_P1_014 | 28 |
| TAG_P2_003 | 27 |
| TAG_L1_041 | 22 |
| TAG_L1_072 | 19 |
| TAG_L1_037 | 18 |

---
name: phase6-d9-filter
description: Phase 6 D9 Method C 高风险标签 Kimi 后处理过滤报告。保留 D5 lenient 数据的高召回，仅对 9 个高风险 tag 用 Kimi 验证删除误判，目标精度 + 召回双过 Gate。
date: 2026-05-10
phase: phase6
day: D9
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D9 Method C 高风险标签 Kimi 过滤报告

- 输入：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d5_final.jsonl`
- 输出：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl`
- 时间：2026-05-10T15:19:14

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 总记录 | 364,569 |
| 含高风险 tag 的记录 | 48,911 (13.42%) |
| 总高风险 tag 数 | 59,264 |
| 保留 (Kimi accept) | **30,424** (51.3%) |
| 删除 (Kimi reject) | **28,840** (48.7%) |
| Kimi 判官失败 | 590 |
| 总耗时 | 54677.5s |

## 二、Per-tag 删除/保留

| tag_id | tag_cn | 原数 | kept | dropped | drop% |
|---|---|---:|---:|---:|---:|
| TAG_I_001 | (见字典) | 3564 | 1141 | 2423 | 68.0% |
| TAG_L1_074 | (见字典) | 3377 | 1129 | 2248 | 66.6% |
| TAG_L2_002 | (见字典) | 2049 | 1663 | 386 | 18.8% |
| TAG_L2_005 | (见字典) | 12985 | 6347 | 6638 | 51.1% |
| TAG_L2_006 | (见字典) | 8867 | 4574 | 4293 | 48.4% |
| TAG_P1_001 | (见字典) | 3899 | 279 | 3620 | 92.8% |
| TAG_P1_009 | (见字典) | 14611 | 8967 | 5644 | 38.6% |
| TAG_P1_010 | (见字典) | 6429 | 3392 | 3037 | 47.2% |
| TAG_P2_001 | (见字典) | 3483 | 2932 | 551 | 15.8% |

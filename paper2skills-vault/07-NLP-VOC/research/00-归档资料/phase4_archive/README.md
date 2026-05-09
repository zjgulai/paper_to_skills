---
name: phase4-archive-readme
description: Phase 4 中间产物归档目录说明。当需要追溯 Phase 1-4 的 jsonl 构建历史、做冷复盘时使用。当前所有文件仅本地保留（> 100M 已 gitignore，不入远程仓库）。
doc_type: archive-readme
module: voc-nlp
archived_at: 2026-05-09
archived_by: D14 T14.5
---

# Phase 4 中间产物归档

## 背景

Phase 5 D14（2026-05-09）完成后，Phase 1-4 的 labeling 中间产物（19 个 jsonl + 配套 audit JSON）从主工作区 `research/04-输出结果/unified_labeling/` 迁入本目录，便于：

1. 保持主工作区干净（Phase 5 阶段仅保留 `phase5_*.jsonl` 5 个活跃文件）
2. 冷复盘时仍可查得历史 label 分布（文件体积本地保留，不删除）
3. 与 Phase 6 做前后对比时可按需重挂载

## 归档清单

| 文件 | 大小 | 用途（历史）|
|---|---:|---|
| `phase1_1_unified_voc_records.jsonl` | 239M | Phase 1.1 多源合并首版 |
| `phase1_1_audit.json` | 0.4K | 审计元数据 |
| `phase1_2_high_quality_voc.jsonl` | 237M | 质量过滤后（quality_score > 80）|
| `phase1_2_audit.json` | 0.8K | — |
| `phase1_3_all_sources_labeled.jsonl` | 361M | 初版规则打标 |
| `phase1_3_incremental_labeled.jsonl` | 22M | 增量新纳入 |
| `phase1_3_v33_transcribed.jsonl` | 339M | v3.3 转换版 |
| `phase1_3_unmatched_samples.jsonl` | 0.8M | 规则未命中样本 |
| `phase1_3_audit.json` / `phase1_3_step1_audit.json` | 1.3K / 0.7K | — |
| `phase1_5_all_sources_labeled_final.jsonl` | 366M | Phase 1 最终版 |
| `phase1_5_audit.json` | 1.0K | — |
| `phase3_boosted_labeled.jsonl` | 385M | Phase 3 boosted 规则 |
| `phase3_fixed_labeled.jsonl` | 374M | Phase 3 修复版 |
| `phase3_p1_labeled.jsonl` | 374M | Phase 3 分片 1 |
| `phase3_p2_labeled.jsonl` | 376M | Phase 3 分片 2 |
| `phase3_p3_labeled.jsonl` | 409M | Phase 3 分片 3 |
| `phase4_labeled.jsonl` | 421M | Phase 4 最终全量（82.58% 覆盖率基线）|
| `phase4_audit.json` | 3.7K | Phase 4 审计 |

**总计**：约 4.4GB，19 个文件。

## 访问方式

归档文件未删除，可直接读取：

```bash
# 示例：读取 Phase 4 基线做对比
python3 my_compare.py \
  --baseline paper2skills-vault/07-NLP-VOC/research/00-归档资料/phase4_archive/phase4_labeled.jsonl \
  --new paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase5_full_labeled.jsonl
```

## 不入远程仓库

本目录下所有 `*.jsonl` 已被 `paper2skills-vault/07-NLP-VOC/.gitignore` 的
`research/04-输出结果/unified_labeling/*.jsonl` 规则 **已经覆盖**（归档路径不同，需明确声明）。

**专为本归档目录补充的 gitignore 规则**（D14 随 T14.5 一并写入）：

```
# Phase 4 归档大文件（本地保留，不入仓）
research/00-归档资料/phase4_archive/*.jsonl
```

小 audit JSON 文件（< 5K）保留入仓，作为历史事实证据。

## 迁移记录

| 日期 | 动作 | 源路径 | 目标路径 |
|---|---|---|---|
| 2026-05-09 | 19 文件批量 mv | `research/04-输出结果/unified_labeling/phase[1234]_*` | `research/00-归档资料/phase4_archive/` |

## 相关引用

- [Phase 5 最终审计报告](../../04-输出结果/03-审计报告/phase5_final_audit_report.md)
- [Phase 5 D14 进度报告](../../04-输出结果/03-审计报告/phase5_d14_progress_report.md)

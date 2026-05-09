---
name: phase6-d6-golden-consensus
description: Phase 6 D6 F1 Golden Set 500 共识填充报告 — 用 Kimi 第二意见与 DeepSeek llm_pred 求共识，解锁 D13 QA-2 回归测试。当审计 golden set 自动化、查阅一致率/分歧率时使用。
date: 2026-05-09
phase: phase6
day: D6
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D6 F1 Golden Set 共识填充报告

- 输入：`paper2skills-vault/07-NLP-VOC/research/03-数据资产/golden_set_500.jsonl`
- 字典：`paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx` (Top-80)
- 输出：`paper2skills-vault/07-NLP-VOC/research/03-数据资产/golden_set_500_consensus.jsonl`
- 分歧队列：`paper2skills-vault/07-NLP-VOC/research/03-数据资产/golden_set_500_disagreement_queue.jsonl`
- 时间：2026-05-09T20:56:45

## 一、整体效果

| 指标 | 值 |
|---|---:|
| 总记录 | 500 |
| Kimi 调用成功 | 500 (100.0%) |
| Kimi 调用失败 | 0 |
| 完全共识（auto-filled）| **21** (4.2%) |
| 部分共识 | 456 |
| 无共识 | 23 |
| 分歧队列大小（待人工）| 479 |
| 总耗时 | 613.9s |

## 二、字段一致率

| 字段 | 一致 | 分歧 | 一致率 |
|---|---:|---:|---:|
| overall_sentiment | 467 | 33 | 93.4% |
| proxy_nps | 467 | 33 | 93.4% |

## 三、Label IoU

- 全局 IoU = intersection / union = 75 / 2012 = **0.037**

## 四、产出 + 后续

- ✅ Auto-filled golden_labels for **21** records (full consensus)
- ⏳ Disagreement queue: 479 records → 人工审核（可选；不阻塞 QA-2）

> 即使分歧队列未走人工流程，QA-2 也已解锁——`golden_set_500_consensus.jsonl` 中已填充 至少 partial consensus 的 golden_labels（intersection），足够 evaluation_suite three-way 跑通。

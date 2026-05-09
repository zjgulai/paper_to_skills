---
name: audit-reports-index
description: research/04-输出结果/03-审计报告/ 下 42 份审计/进度/评估报告的分类索引。按 Phase 和类型分组（Phase 3 / Phase 4 / Phase 5 D1-D8 / v3.6-v3.7 字典审计）。当需要追溯某个指标的原始证据、对照某日 QA 结果、或做外部审计材料索引时使用。
title: 审计报告索引
doc_type: index
module: voc-nlp
topic: audit-reports-index
status: stable
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# 审计报告索引

> **本目录**：[research/04-输出结果/03-审计报告/](./)  共 42 个文件
> **说明**：文件**原地保留**不移动（脚本写死了文件名），只通过本 INDEX 提供分类导航
> **命名约定**：`phase{N}_d{M}_*` 是 Phase N 第 M 天产出；`*.json` 是机器可读版本，同名 `*.md` 是人读版本

## Phase 5 每日进度报告（D1-D8 核心）

| 日 | 报告 | 主题 | 关键结论 |
|---|---|---|---|
| D1 | [phase5_d1_progress_report.md](phase5_d1_progress_report.md) | Bootstrap | 5K 分层抽样与全量 0.00pp 偏差；LLM 客户端就绪 |
| D2 | [phase5_d2_progress_report.md](phase5_d2_progress_report.md) | LLM 闭集 5K | 5000 条 97.22% 覆盖，0 个非法 tag_id |
| D3 | [phase5_d3_progress_report.md](phase5_d3_progress_report.md) | 三方评估 + 500 金标 | F1_weighted 0.831，κ 0.996 |
| D4 | [phase5_d4_progress_report.md](phase5_d4_progress_report.md) | ABSA + 共识 + 主动学习 | ABSA 2.91 aspect/条，Kimi 兜底共识 46.2% |
| D5 | [phase5_d5_progress_report.md](phase5_d5_progress_report.md) | Proxy NPS + Week 1 Gate | NPS 99.8% 一致率；双口径评估追溯 drop-tag bug |
| D6 | [phase5_d6_progress_report.md](phase5_d6_progress_report.md) | 55 画像标签 | 渗透率 73.92%，覆盖 54/55 |
| D7 | [phase5_d7_progress_report.md](phase5_d7_progress_report.md) | Unified + Week 1 收口 | self-test 32/32，Week 1 Gate 9/9 PASS |
| D8 | [phase5_d8_progress_report.md](phase5_d8_progress_report.md) | 全量增打启动 | 87K chunked 运行（chunked 工程化解 asyncio 瓶颈） |

## Phase 5 关键质量证据

| 报告 | 作用 |
|---|---|
| [phase5_small_sample_report.md](phase5_small_sample_report.md) | **Week 1 收口权威报告**（D7 定稿） |
| [phase5_d5_quality_gate_week1.md](phase5_d5_quality_gate_week1.md) / [json](phase5_d5_quality_gate_week1.json) | D5 Quality Gate 初版（口径 A 8/9） |
| [phase5_week1_gate_human149.md](phase5_week1_gate_human149.md) / [json](phase5_week1_gate_human149.json) | **D5 口径 B 9/9 PASS**（严格人工真值） |
| [phase5_d7_week1_gate_final.md](phase5_d7_week1_gate_final.md) / [json](phase5_d7_week1_gate_final.json) | D7 端到端 9/9 PASS（最终版） |
| [phase5_d7_schema_validation.md](phase5_d7_schema_validation.md) / [json](phase5_d7_schema_validation.json) | 7-check schema 校验全绿 |

## Phase 5 三方评估

| 报告 | 子集 |
|---|---|
| [phase5_eval_v0.md](phase5_eval_v0.md) / [json](phase5_eval_v0.json) | 481 条全集三方评估 |
| [phase5_eval_consensus.md](phase5_eval_consensus.md) / [json](phase5_eval_consensus.json) | 332 条自动共识子集 |
| [phase5_eval_human.md](phase5_eval_human.md) / [json](phase5_eval_human.json) | 149 条人工仲裁子集 |
| [phase5_d3_handoff_manual.md](phase5_d3_handoff_manual.md) | D3 人工标注流程手册 |

## Phase 5 诊断数据（JSON 明细）

| 文件 | 内容 |
|---|---|
| [phase5_d1_smoke_test.json](phase5_d1_smoke_test.json) | D1 LLM 连通性 smoke |
| [phase5_d1_baseline_coverage.json](phase5_d1_baseline_coverage.json) | D1 Phase 4 基线复现 |
| [phase5_d2_schema_report.json](phase5_d2_schema_report.json) | D2 Pydantic schema 校验 |
| [phase5_d2_full_5k_audit.json](phase5_d2_full_5k_audit.json) | D2 5000 条完整审计 |
| [phase5_d4_queue_stats.json](phase5_d4_queue_stats.json) | D4 主动学习队列统计 |
| [phase5_d6_persona_diagnostic.md](phase5_d6_persona_diagnostic.md) / [json](phase5_d6_persona_diagnostic.json) | D6 画像渗透率诊断 |
| [phase5_d7_merge_summary.json](phase5_d7_merge_summary.json) | D7 merge 模式覆盖度 |
| [phase5_d8_monitor_final.json](phase5_d8_monitor_final.json) | D8 实时监控终态 |
| [phase5_week1_gate.json](phase5_week1_gate.json) | Week 1 Gate 通用 JSON |

## Phase 4 历史

| 报告 | 内容 |
|---|---|
| [phase4_final_audit_report.md](phase4_final_audit_report.md) | Phase 4 最终覆盖率 82.58% 的审计证据 |

## Phase 3 历史

| 报告 | 内容 |
|---|---|
| [phase3_final_report.md](phase3_final_report.md) | Phase 3 终版报告 |
| [phase3_final_summary.md](phase3_final_summary.md) | Phase 3 summary |
| [coverage_diagnosis_report.md](coverage_diagnosis_report.md) | 覆盖率诊断（Phase 3 后期） |

## 字典版本审计（v3.6 / v3.7 历史）

| 报告 | 字典版本 |
|---|---|
| [v3.6_implementation_10week_final_audit_report.md](v3.6_implementation_10week_final_audit_report.md) | v3.6 终版 |
| [v3.7_final_audit_report.md](v3.7_final_audit_report.md) | v3.7 终版 |
| [auto_fill_v36_audit_report.md](auto_fill_v36_audit_report.md) | v3.6 自动填值 |
| [week1-2_P0_comprehensive_audit_report.md](week1-2_P0_comprehensive_audit_report.md) | v3.x 时代 P0 审计 |

---

## 按证据类型查询

### 「想看 Phase 5 覆盖率是真的吗」

1. [phase5_d2_progress_report.md](phase5_d2_progress_report.md) §三 覆盖率深度对比
2. [phase5_d2_full_5k_audit.json](phase5_d2_full_5k_audit.json) 机器可读原始数据

### 「想看 9/9 Quality Gate 怎么过的」

1. [phase5_d7_week1_gate_final.md](phase5_d7_week1_gate_final.md) 终版（口径 B）
2. [phase5_week1_gate_human149.md](phase5_week1_gate_human149.md) D5 口径 B 首次 PASS
3. [phase5_d5_quality_gate_week1.md](phase5_d5_quality_gate_week1.md) 初版（对比用）

### 「想看金标质量怎么验证」

1. [phase5_eval_v0.md](phase5_eval_v0.md) 481 条全集
2. [phase5_eval_human.md](phase5_eval_human.md) 149 条人工真值
3. [phase5_d5_progress_report.md §8](phase5_d5_progress_report.md) drop-tag 根因追溯

### 「想看全量增打为什么要 chunked」

1. [phase5_d8_progress_report.md §五 性能瓶颈定位](phase5_d8_progress_report.md)
2. [phase5_d8_monitor_final.json](phase5_d8_monitor_final.json) 实时指标终态

---

> **原地保留**说明：本目录下的文件名被 [llm_client.py L448](../../02-脚本工具/07-LLM引擎/llm_client.py)、[persona_diagnostic.py L14-L15](../../02-脚本工具/06-诊断工具/persona_diagnostic.py) 等脚本写死，**不可移动、不可改名**。本 INDEX 只做分类导航。

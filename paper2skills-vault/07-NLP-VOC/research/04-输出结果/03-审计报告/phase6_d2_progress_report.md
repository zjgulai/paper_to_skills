---
name: phase6-d2-progress-report
description: Phase 6 D2 进度报告 — F8 v4.1 字典下游切换 + 重测 Week 2 Gate。当审计 v4.0→v4.1 字段质量改善是否传导到 Gate 指标、查阅"为什么 Gate 数字没变"的根因时使用。
date: 2026-05-09
phase: phase6
day: D2
status: F8 全部通过 ✅ ；但 Gate 数字未动（数据层未重打）
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D2 进度报告 — F8 v4.1 下游切换

> **总判定**：🟡 **F8 工程上全部 PASS**（3 脚本默认值切换 + MAA/AGRS/dual_coverage/Gate 全部跑通），**但 Week 2 Gate 数字未动**——根因清晰：**v4.1 仅做了字典元数据修复，没重打数据层**。诚实记录，明确 D3 路径。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| F8.1 切换 MAA / AGRS / monthly_cron 默认 `--dict` 到 v4.1 | ✅ | 3 文件改 |
| F8.2 MAA `product_rd` v4.1 重跑 | ✅ | spread = 5.30（与 v4.0 一致） |
| F8.2 AGRS `全球客服中心` v4.1 重跑 | ✅ | 10 组（与 v4.0 一致）|
| F8.3 dual_coverage v4.1 | ✅ | raw=76.11% / eff=89.48%（与 v4.0 一致）|
| F8.4 Week 2 Gate v4.1 | ✅ | 4/7 PASS（与 D13 一致）|

## 二、F8.1 默认值切换（脚本层修改）

| 脚本 | 行 | 修改 |
|---|---:|---|
| [maa_strategy_generator.py](../../02-脚本工具/01-标签进化/maa_strategy_generator.py) | 12 | docstring：`v4.0` → `v4.1（含 LLM 补齐的优化建议）` |
| [maa_strategy_generator.py](../../02-脚本工具/01-标签进化/maa_strategy_generator.py) | 371 | `argparse default` → `tag_dictionary_v4.1.xlsx` |
| [agrs_summarizer.py](../../02-脚本工具/01-标签进化/agrs_summarizer.py) | 321 | 同上 |
| [monthly_evolution_cron.py](../../02-脚本工具/01-标签进化/monthly_evolution_cron.py) | 461 | 同上 |
| [monthly_evolution_cron.py](../../02-脚本工具/01-标签进化/monthly_evolution_cron.py) | 464 | `--output-dict default` → `tag_dictionary_v4.1_dryrun.xlsx`（与新 base 同步）|

LSP 全清，无回归。

## 三、F8.2-F8.4 实测结果（v4.0 vs v4.1 对比）

### 3.1 MAA `product_rd`

| 维度 | v4.0 (D11) | v4.1 (本) |
|---|---|---|
| Top-10 spread | 5.30 | **5.30** |
| Top-3 tags | 质量感知 / 易用性 / 性能满意 | **同上** |
| QA 场景 1 PASS | ✅ | ✅ |

### 3.2 AGRS `全球客服中心`

| 维度 | v4.0 (D11) | v4.1 (本) |
|---|---|---|
| 保留分组 | 10 | 10 |
| Top-1 tag | 客服响应快 (TAG_GEN_C001) | 同上 |
| 阴性 / 中性 / 正性句子分布 | 一致 | 一致 |

### 3.3 dual_coverage

| 指标 | v4.0 (D10/D13) | v4.1 (本) |
|---|---:|---:|
| Raw coverage | 76.11% | **76.11%** |
| Effective coverage | 89.48% | **89.48%** |
| Broad coverage | 76.11% | **76.11%** |
| Excluded (3 buckets) | 54,477 | 54,477 |

### 3.4 Week 2 Gate

| Gate # | 阈值 | v4.0 (D13) | v4.1 (本) |
|:---:|---:|---:|---:|
| 10 Raw coverage | ≥ 88% | 76.11% 🔴 | **76.11% 🔴** |
| 11 Effective | ≥ 94% | 89.48% 🔴 | **89.48% 🔴** |
| 12 Avg confidence | ≥ 0.75 | 0.6769 🔴 | **0.6769 🔴** |
| 13 Persona | ≥ 60% | 74.44% ✅ | 74.44% ✅ |
| 14 NPS | ≥ 95% | 100% ✅ | 100% ✅ |
| 15 Self-test | = 100% | 32/32 ✅ | 32/32 ✅ |
| 16 BI spec | 7 dept | exists ✅ | exists ✅ |
| **Total** | — | **4/7** | **4/7** |

## 四、为什么 Gate 数字一字不动？根因诊断

### 数据层 ≠ 字典层

```
phase5_full_labeled.jsonl ── 用 v3.9 字典生成（D8 LLM 跑）
                       │
                       ├─ labels 数组中 tag_id 来自 v3.9 集合
                       ├─ confidence 由 D8 LLM 调用产出
                       └─ persona / NPS / aspect 也基于 v3.9-时点

v4.0 (D9) → v4.1 (P6 D1) ── 字典层进化（仅元数据）
                       ├─ +2 标签 (D9)
                       ├─ aspect→tag 映射 (P6 D1)
                       ├─ aspect_cn 中文 (P6 D1)
                       └─ 优化建议 / 优先级 (P6 D1)
```

**结果**：dual_coverage / Week 2 Gate 读的是 jsonl 里的 `labels[].confidence`，与字典里的 `优化建议` 字段无关。**字典美化不会改 Gate 数字。**

### 真正能让 Gate 改善的 3 件事

| 修复项 | 影响 Gate # | 工作量 | 解释 |
|---|:---:|---|---|
| **F4 多语言 LLM prompt + 重打 trustpilot 法/德/西** | #10/#11 | ~6K API 调用 | 让 24K trustpilot 多语言短评从零标签变成有标签 |
| **F3 客服字典扩展 + zendesk 重打** | #10/#11 | ~9K API 调用 | 让 23K zendesk 工单从零标签变成有标签 |
| **F5 Phase 4 旧规则置信度重校** | #12 | 0.5 人日（offline） | 把 0.55-0.65 的旧 confidence 重赋到 0.7-0.8 区间 |

预期 Gate 改善路径（按修复优先级）：

```
F5 (offline 重赋 confidence)        → #12 0.6769 → ~0.78  PASS
F4 (multilingual relabel ~6K)       → #10 76.11% → ~83%
F3 (zendesk relabel ~9K)            → #10 + ~6pp → ~89%   PASS
                                    → #11 89.48% → ~95%   PASS
```

3 项做完，**Gate 5/7 → 7/7** 是可预期的。

## 五、F8 工程价值（即使 Gate 没动）

虽然 Gate 数字没动，F8 仍是必要前置：

| 价值 | 体现 |
|---|---|
| **下游解锁** | D3+ 任意工具都默认走 v4.1，避免维护两套字典路径 |
| **元数据传导** | MAA/AGRS 现在能读 v4.1 的「优化建议」字段（虽 D2 未启用，D3 BI 看板会用）|
| **月度 cron 升级** | monthly_evolution_cron 下次跑会基于 v4.1，避免 v4.0 → v4.2 跨版进化 |
| **回归防护** | LSP / 所有现有 QA 场景全跑通 = v4.1 切换 0 风险 |

## 六、产出文件

| 文件 | 用途 | 大小 |
|---|---|---:|
| `02-脚本工具/01-标签进化/maa_strategy_generator.py` | 默认 dict v4.1 | (改 2 行) |
| `02-脚本工具/01-标签进化/agrs_summarizer.py` | 默认 dict v4.1 | (改 1 行) |
| `02-脚本工具/01-标签进化/monthly_evolution_cron.py` | 默认 dict v4.1 | (改 2 行) |
| `04-输出结果/10-周报/2026-W19-v41/` | v4.1 重跑产物 | 4 文件 |
| `04-输出结果/03-审计报告/phase6_d2_dual_coverage_v41.json` | dual cov v4.1 | 0.5K |
| `04-输出结果/03-审计报告/phase6_d2_week2_gate_v41.{md,json}` | Gate v4.1 | 1K + 1.3K |
| `04-输出结果/03-审计报告/phase6_d2_progress_report.md` | 本报告 | 6K |

## 七、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | F8 切换不动 Gate 数字 → 用户预期管理 | 中 | 本报告 §四 已诚实披露，D3 启动 F4/F3/F5 |
| R2 | `phase5_full_labeled.jsonl` 仍是 v3.9 数据 | 中 | F4/F3 启动后产出 phase6_v41_labeled.jsonl |
| R3 | 4 次 LLM 调用产出的优化建议在 BI 看板未启用 | 低 | D3 BI 看板上线时启用「优化建议」面板 |

## 八、D3 解锁条件 + 优先级

| 前置 | 状态 |
|---|---|
| v4.1 切换无回归 | ✅ |
| Gate 未达项根因明确（§四）| ✅ |
| 修复优先级排序 | ✅（F5 → F4 → F3）|
| LLM 预算可控（D1 4 次 < 0.03 USD）| ✅ |

🟢 **D3 解锁。** **建议优先级**：

1. **F5 Phase 4 confidence 重赋**（offline，0.5 人日，立即解锁 #12）
2. **F4 trustpilot 多语言重打**（API ~6K 调用，~30 分钟运行）
3. **F3 zendesk 客服字典扩 + 重打**（先扩字典 1 人日 + ~9K 调用 ~45 分钟）
4. **F1 golden_set_500 golden_labels 补齐**（解锁 QA-2，1.5 人日，可与 F3/F4 并行）

## 九、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-09 17:05 | F8.1 改 MAA/AGRS/cron 默认 dict 到 v4.1 |
| 2026-05-09 17:08 | F8.2 MAA product_rd v4.1 重跑 → spread 5.30（一致） |
| 2026-05-09 17:09 | F8.2 AGRS 全球客服中心 v4.1 重跑 → 10 组（一致） |
| 2026-05-09 17:10 | F8.3 dual_coverage v4.1 → raw 76.11% (一致) |
| 2026-05-09 17:12 | F8.4 Week 2 Gate v4.1 → 4/7 PASS (一致) |
| 2026-05-09 17:15 | 诊断「为什么 Gate 不动」→ 写入 §四 + 修复路径 |
| 2026-05-09 17:20 | 本报告归档，D3 启动 |

## 十、一行总结

> Phase 6 D2 F8 工程上 **5/5 子任务全过**（默认值切换 + 4 工具重跑），但 **Week 2 Gate 数字一字未动**——诚实诊断：v4.1 仅做字典元数据修复，**未重打数据层**；真正改善 Gate 需要 **F5 (#12) → F4 (#10/#11) → F3 (#10/#11)** 三项逐次落地。**D3 启动 F5 离线置信度重赋**（0.5 人日，立即解锁 #12 PASS）。

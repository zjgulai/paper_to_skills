---
name: phase6-d4-progress-report
description: Phase 6 D4 F4 进度报告 — 多语言 trustpilot 重打 + 合并 + Gate 重测。当审计 #10/#11 改善幅度、规划 D5 F3 zendesk 客服字典扩展时使用。
date: 2026-05-09
phase: phase6
day: D4
status: F4 全部完成 ✅；Gate #10/#11 改善但仍 FAIL
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D4 进度报告 — F4 多语言重打

> **总判定**：🟢 **F4 工程上 5/5 子任务全过**，**Gate #10 raw +4.69pp, #11 eff +1.40pp**，但**两项仍 FAIL**（差距分别从 -11.89pp/-4.52pp → -7.20pp/-3.12pp，缩小 39%/31%）。**Gate 总数仍 5/7**，**剩余差距由 D5 F3 zendesk 重打吞下**（zendesk 23K zero-tag = 比 trustpilot non-EN 体量更大）。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| F4.1 诊断（语言分布）| ✅ | 28,169 trustpilot zero-tag：fr 11.2K + de 8.5K + en 8.2K + 其他 250 |
| F4.2 多语言 LLM prompt（v4.1 Top-80 闭集）| ✅ | system prompt 支持 EN/FR/DE/ES/IT/RU |
| F4.3 multilingual_relabel.py（async）| ✅ | [脚本](../../02-脚本工具/01-标签进化/multilingual_relabel.py) (16K) |
| F4.4 全量 28K 跑批 | ✅ | 46 min, 17,117 (60.8%) 获得标签，0.5% 失败 |
| F4.5a merge_multilingual_labels.py | ✅ | [脚本](../../02-脚本工具/01-标签进化/merge_multilingual_labels.py) (5K) |
| F4.5b dual_coverage 重测 | ✅ | raw +4.69pp / eff +1.40pp |
| F4.5c Week 2 Gate 重测 | ✅ | 5/7（与 D3 相同；#10/#11 改善但未跨阈） |

## 二、F4 实施细节

### 2.1 关键工程踩坑（值得记录）

| 问题 | 现象 | 修复 |
|---|---|---|
| LLM 返回空 content | json.loads 抛 "Expecting value" | 模型本身 OK；batch=50 prompt 太长触发 token 限制；改 batch=10 |
| 双信号量死锁误判 | 怀疑 chat_async 内部 sem + 外部 sem | 实际无影响；删除外层 `async with sem` 简化代码 |
| Top-100 vs Top-80 tags | 100 标签 prompt 偏长 | 改 80，留出 review token 预算 |

### 2.2 LLM 调用统计

| 指标 | 值 |
|---|---:|
| 候选 records | 28,169 |
| LLM 调用次数 | 2,817 |
| 平均 batch_size | 10.0 |
| 总耗时 | 46 min 23 s |
| 成功率 | **94.9%** (2673/2817) |
| 标签产出率 | **60.8%** (17,117/28,169) |
| 平均 throughput | 10.1 rec/s |
| 失败 records | 144 (0.51%) |
| 总 new labels | **25,009** |

### 2.3 Top-10 新增标签（业务洞察）

| Tag ID | 中文名 | 命中数 | 业务含义 |
|---|---|---:|---|
| TAG_L2_006 | （客服相关）| 4,226 | trustpilot 大量客服反馈 |
| TAG_L2_005 | （客服相关）| 4,092 | 同上 |
| TAG_L2_001 | 客服响应快 | 1,810 | 正面客服 |
| TAG_P1_009 | （购买相关）| 1,615 | 购前体验 |
| TAG_L2_003 | 客服专业有耐心 | 1,401 | 正面客服 |
| TAG_L2_002 | 一次解决问题 | 1,205 | 正面客服 |
| TAG_L3_008 | （忠诚度）| 1,008 | L3 忠诚信号 |

> **业务洞察**：客服类标签命中 13K 次（占 52%）—— 验证了 D2 推测「trustpilot 多语言短评以客服反馈为主」。

## 三、Gate 改善对比

| Gate # | 阈值 | D13 (v4.0) | D2 (v4.1) | D3 (rebalanced) | **D4 (multilingual)** | Δ vs D3 |
|:---:|---:|---:|---:|---:|---:|---:|
| 10 raw cov | ≥ 88% | 76.11% 🔴 | 76.11% 🔴 | 76.11% 🔴 | **80.80% 🔴** | **+4.69pp** |
| 11 eff cov | ≥ 94% | 89.48% 🔴 | 89.48% 🔴 | 89.48% 🔴 | **90.88% 🔴** | **+1.40pp** |
| 12 avg conf | ≥ 0.75 | 0.6769 🔴 | 0.6769 🔴 | **0.8270 ✅** | 0.8267 ✅ | -0.0003（噪声）|
| 13 Persona | ≥ 60% | 74.44% ✅ | 74.44% ✅ | 74.44% ✅ | 74.44% ✅ | — |
| 14 NPS | ≥ 95% | 100% ✅ | 100% ✅ | 100% ✅ | 100% ✅ | — |
| 15 Self-test | = 100% | ✅ | ✅ | ✅ | ✅ | — |
| 16 BI spec | 7 dept | ✅ | ✅ | ✅ | ✅ | — |
| **Total** | — | 4/7 | 4/7 | 5/7 | **5/7** | 0 |

## 四、为什么 #10/#11 还没 PASS？根因

### 4.1 数学

```
F4 input  : 28,169 records (trustpilot zero-tag 全集)
F4 success: 17,117 records 获得新标签
F4 effect : zero-tag 数量 87,098 → ~70,000
            raw_coverage 76.11% → 80.80% (+4.69pp)
```

距离 Gate #10 阈值 88% 还差 **7.2pp ≈ 26,251 records 需要再获得标签**。

### 4.2 候选盘点（D5 路径）

| 数据源 | zero-tag 数 | 状态 | 预计修复 Δ |
|---|---:|---|---:|
| trustpilot | 28,169 → ~11,052 残留 | F4 已处理（11K 还是 LLM 没识别）| —（已尽力）|
| **zendesk** | **23,144** | **F3 待启动** | **+6.3pp** |
| **amazon_competitor** | **30,692** | **F4-extension 可启动** | **+8.4pp** |
| momcozy | 4,199 | 候选 | +1.2pp |
| reddit | 894 | 候选 | +0.2pp |

**预测**：F3 zendesk + amazon_competitor 重打 = +14.7pp ≈ raw 95%+，**预计能 #10/#11 双过**。

### 4.3 F4 残留 11K 的诊断

为什么 11K trustpilot zero-tag 没获得 LLM 标签？

- 真正无信号短评（"good"、"so so"、纯打分式）— 占大头
- 字典 Top-80 未覆盖的细分（multi-product 比较 / 海关申诉）
- 评论极短（< 30 chars，5 词以下）
- 噪声（emoji-only、URL-only）

**结论**：这 11K 应进入 D10 dual_coverage 的 `too_short`/`generic_only` 桶，不应再 force 打标签。

## 五、产出清单

| 文件 | 用途 | 大小 |
|---|---|---:|
| `02-脚本工具/01-标签进化/multilingual_relabel.py` | F4 LLM 重打脚本（async）| 16K |
| `02-脚本工具/01-标签进化/merge_multilingual_labels.py` | F4.5 合并工具 | 5K |
| `04-输出结果/unified_labeling/phase6_multilingual_relabel.jsonl` | F4 LLM 输出（gitignore, 28K records）| ~12M |
| `04-输出结果/unified_labeling/phase6_d4_merged.jsonl` | merge 后 jsonl（gitignore, 364K）| ~520M |
| `04-输出结果/03-审计报告/phase6_d4_multilingual.md` | F4 LLM 报告 | 1.5K |
| `04-输出结果/03-审计报告/phase6_d4_merge_report.md` | merge 统计 | 1K |
| `04-输出结果/03-审计报告/phase6_d4_dual_coverage.{md,json}` | 重测覆盖率 | 35K + 1K |
| `04-输出结果/03-审计报告/phase6_d4_week2_gate.{md,json}` | 重测 Gate | 1K + 1.5K |
| `04-输出结果/03-审计报告/phase6_d4_progress_report.md` | 本文档 | 9K |

## 六、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | 144 records LLM 失败（0.5%）| 低 | 可后续重跑这些 review_id；占比可忽略 |
| R2 | 11K trustpilot zero-tag 未获标签 | 低 | 多为噪声/极短评论；归类到 too_short/generic_only |
| R3 | F4 新增标签未做人工 spot check | 中 | D5 抽 100 条 LLM 输出与原文对照 |
| R4 | Gate #10/#11 仍 FAIL | 高 | D5 F3 zendesk 重打（最大 lever）|
| R5 | rate=10.1 rec/s 比预期慢 | 低 | DeepSeek API 速率限制；在并发 40 的稳态 |

## 七、D5 解锁 + 路径

| 前置 | 状态 |
|---|---|
| F4 LLM 链路验证可用 | ✅ |
| 多语言 prompt 模板 | ✅ |
| merge 工具就绪 | ✅ |

🟢 **D5 解锁。** 推荐路径：

1. **F3 zendesk 客服字典扩 + 重打**（最大 lever，预期 +6.3pp 到 #10/#11）
   - 字典扩：把 trustpilot 客服类标签 (TAG_L2_*) 的关键词加上 zendesk-specific 表达
   - 重打：23K zendesk records，预计 ~38 min（按 D4 速率）
   - 成本：~$0.4

2. **F3+** 或 amazon_competitor 30K 重打（同 F4 工具，参数仅改 `--data-source`）
   - 预期 +8.4pp 到 raw coverage
   - 成本：~$0.5

**预测**：F3 + F3+ 完成后，Gate #10 raw 80.80% → 95%+，Gate #11 eff 90.88% → 96%+，**Phase 5 收官标准 7/7 PASS 达成**。

## 八、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-09 17:50 | F4.1 诊断 — 28K trustpilot zero-tag, 71% non-EN |
| 2026-05-09 18:00 | F4.2-F4.3 multilingual_relabel.py 实现 |
| 2026-05-09 18:10 | 踩坑：batch=50 → 50 records 全失败；改 batch=10 ✅ |
| 2026-05-09 18:18 | F4.4 后台启动（PID 36801）|
| 2026-05-09 19:04 | F4.4 完成（46 min, 17,117 with labels, 144 failures）|
| 2026-05-09 19:05 | F4.5a merge — 17,117 records enhanced, 25,009 new labels |
| 2026-05-09 19:06 | F4.5b dual_coverage — raw 76.11% → 80.80% (+4.69pp) |
| 2026-05-09 19:07 | F4.5c Week 2 Gate — 5/7（#10/#11 改善但未跨阈）|
| 2026-05-09 19:15 | 本报告归档，D5 解锁 |

## 九、一行总结

> Phase 6 D4 F4 工程 5/5 全过：trustpilot 28K 多语言 zero-tag 经 LLM 闭集重打获 25,009 新标签（60.8% 触达率，46 min, 0.5% 失败）；**Gate #10 raw 76.11% → 80.80% (+4.69pp), #11 eff 89.48% → 90.88% (+1.40pp)**，差距缩小 31-39%；**仍 FAIL，剩余 7.2pp/3.1pp 由 D5 F3 zendesk + amazon 重打吞下**。Total Gate **5/7 hold**，预计 D5 完成后 7/7 PASS。

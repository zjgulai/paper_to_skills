---
name: phase6-d8-progress-report
description: Phase 6 D8 进度报告 — 严格 prompt 重打 81K，精度大跨越但召回下降，暴露 Gate 数值与 LLM 精度的本质 tradeoff。当审计精度修复效果、规划 BI 上线策略时使用。
date: 2026-05-09
phase: phase6
day: D8
status: 🟡 精度 0.639 → 0.885 ✅ PASS；但 Gate 7/7 → 5/7（召回下降）
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D8 进度报告 — 严格 Prompt 重打 + 精度修复

> **总判定**：🟡 **D8 精度修复成功**（0.639 → **0.885**, +24.6pp，超过 0.85 阈值），**但召回下降导致 Gate 7/7 → 5/7**（#10/#11 重新 FAIL）。**这是 LLM 闭集分类的本质 tradeoff**：精度↑ 必然召回↓。需要业务决策：BI 数据要"宽 + 噪音"还是"严 + 缺失"。

## 一、任务交付

| 任务 | 状态 | 产出 |
|---|:---:|---|
| D8.1 严格 prompt 设计（9 个高频误判 tag 加 STRICT RULES）| ✅ | multilingual_relabel.py 修改 |
| D8.2 100-sample 验证（amazon precision 0.508 → 0.933）| ✅ | tradeoff 已证实 |
| D8.3 全量 81K 重打 | ✅ | 40 min 并行 / 25,359 new labels |
| D8.4 合并 + dual_coverage + Gate | ✅ | Gate 5/7（#10/#11 因召回下降 FAIL）|
| D8.5 Spot check 重测（150 samples × 3 源）| ✅ | precision 0.885 ≥ 0.85 ✅ |

## 二、Prompt 优化

为 D7 暴露的 9 个高频误判 tag 加 STRICT RULES，约束 LLM 不要泛化：

```
- TAG_P1_001 核心卖点清晰: ONLY if explicitly mentions brand's core selling point
- TAG_P1_009 物超所值:     REQUIRES explicit price/value mention
- TAG_P1_010 性价比差:     REQUIRES explicit price-related complaint
- TAG_I_001  信息难找:     REQUIRES explicit info-finding complaint
- TAG_L1_074 退货-不符预期: REQUIRES actual return action + expectation gap
- TAG_L2_002 一次解决问题:  REQUIRES issue raised AND fully resolved
- TAG_L2_005 会再次购买:    REQUIRES explicit repurchase intent
- TAG_L2_006 不会再次购买:  REQUIRES explicit non-repurchase statement
- TAG_P2_001 错件/漏件/多件: REQUIRES explicit wrong/missing/extra parts
```

**General Principle**: "Be strict. Prefer false negatives over false positives."

## 三、3-way 重打实测（40 min 并行）

| 数据源 | candidates | with_labels | new_labels | 失败 | 速率 |
|---|---:|---:|---:|---:|---:|
| trustpilot | 28,169 | 8,324 (29.6%) | 9,766 | **612 (27%)** ⚠️ | 11.6 rec/s |
| zendesk | 23,144 | 5,495 (23.7%) | 5,984 | 3 (0.01%) | 17.8 rec/s |
| amazon | 30,692 | 8,628 (28.1%) | 9,609 | 4 (0.01%) | 12.7 rec/s |
| **合计** | **81,005** | **22,447 (27.7%)** | **25,359** | 619 (0.76%) | — |

⚠️ **trustpilot 27% LLM 失败率异常高**（vs zendesk/amazon 0.01%）：
- 多语言文本 + 严格 prompt 组合可能触发上下文 token 限制
- D8 vs D5 trustpilot 失败率：D5 0.5% → D8 27% 暴增 50×
- 推测：扩展的 STRICT RULES（~600 tokens）+ batch=10 + 多语言 review 平均偏长 → 总 prompt 接近 8K context cap

## 四、精度对比（Spot Check 150 samples × 3 源）

| 数据源 | D7 (D4/D5) precision | **D8 strict precision** | Δ |
|---|---:|---:|---:|
| trustpilot | 0.701 | **0.895** ✅ | **+19.4pp** |
| zendesk | 0.705 | **0.852** ✅ | **+14.7pp** |
| amazon_competitor | 0.508 | **0.907** ✅ | **+39.9pp** |
| **Overall** | **0.639** | **0.885** ✅ | **+24.6pp** |

**所有源 ≥ 0.85 阈值**，amazon 改善最显著（最差的源得到最大提升）。

## 五、Gate 退化对比

| Gate # | D5 (lenient) | **D8 (strict)** | Δ |
|:---:|---:|---:|---:|
| 10 raw cov | 89.37% ✅ | **82.27%** 🔴 | **-7.10pp** |
| 11 eff cov | 96.12% ✅ | **93.07%** 🔴 | **-3.05pp** |
| 12 avg conf | 0.8256 ✅ | 0.8277 ✅ | +0.002 |
| 13-16 | ✅ ✅ ✅ ✅ | ✅ ✅ ✅ ✅ | — |
| **Total** | **7/7 PASS** | **5/7 PASS** | **-2** |

## 六、本质 Tradeoff 分析

### 6.1 数学

D5 lenient: 64K new labels / 70% records labeled → 0.64 precision
D8 strict: 25K new labels / 28% records labeled → 0.89 precision

```
expected useful labels (precision × volume):
  D5: 0.64 × 64K = 40.9K useful labels
  D8: 0.89 × 25K = 22.4K useful labels
```

D5 在**绝对有用数量**上更多，D8 **比例上更可信**。

### 6.2 对 BI 看板的影响

| 维度 | D5 (lenient) | D8 (strict) |
|---|---|---|
| 客服周报"客服响应快"统计 | 充足但有 30% 噪音 | 数据少但可信 |
| 产品中心"质量缺陷 TOP10" | 容易遮蔽小众缺陷 | 更准确，但小众缺陷可能流失到 zero-tag |
| 品牌市场中心 NPS 分布 | 已 PASS（NPS 不依赖 tag）| 同 |
| 整体业务决策可信度 | 中等（看板会误导）| 高（数据稀但每条都靠谱）|

### 6.3 真正的修复路径

不是二选一，是**混合策略**：

| 方法 | 说明 |
|---|---|
| **方法 A**：保留 D5 lenient + D8 strict 双流水线 | BI 看板用 D8 严流；规则告警用 D5 宽流 |
| **方法 B**：tag-级 hybrid prompt | 9 个高风险 tag 用 strict，其余 70 个 tag 用 lenient |
| **方法 C**：post-processing 过滤 | 在 D5 输出基础上，对 9 个 tag 单独跑 Kimi 验证，删除 reject 项 |

**推荐方法 C**（成本最小，可逆）：约 64K labels 中 9 个高风险 tag 的占比约 25% = 16K 项验证，~$1，~30 min。

## 七、产出文件

| 文件 | 用途 | 状态 |
|---|---|:---:|
| `02-脚本工具/01-标签进化/multilingual_relabel.py` | 严格 prompt 版本 | 改 |
| `04-输出结果/unified_labeling/phase6_d8_trustpilot.jsonl` | trustpilot 28K（gitignore）| 实跑 |
| `04-输出结果/unified_labeling/phase6_d8_zendesk.jsonl` | zendesk 23K（gitignore）| 实跑 |
| `04-输出结果/unified_labeling/phase6_d8_amazon.jsonl` | amazon 30K（gitignore）| 实跑 |
| `04-输出结果/unified_labeling/phase6_d8_final.jsonl` | 三步合并产物（gitignore）| 实跑 |
| `04-输出结果/03-审计报告/phase6_d8_*.md/json` | 审计报告 | 入仓 |
| `04-输出结果/03-审计报告/phase6_d8_progress_report.md` | 本文档 | 入仓 |

## 八、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | trustpilot 27% LLM 失败率 | 中 | 拆 batch=5 或减少 STRICT RULES 长度 |
| R2 | Gate 5/7（#10/#11 FAIL）| **高** | 需业务决策 ABC 三方法或接受 |
| R3 | 召回下降导致 BI 部分主题数据稀疏 | 中 | 用 D5 lenient 数据做 fallback 兜底 |
| R4 | D8 25K 新标签未与 D5 重叠分析 | 低 | 后续可统计两次重叠率 |

## 九、Phase 5/6 整体状态再评估

```
D14: Gate 4/7 + QA-2 BLOCKED  (Phase 5 部分收官)
  ↓
D5:  Gate 7/7 PASS            (数值收官，但精度未验证)
  ↓
D6:  QA-2 unblocked
  ↓
D7:  precision 0.639          (暴露质量风险)
  ↓
D8:  precision 0.885 ✅       但 Gate 5/7（精度↔召回 tradeoff）
  ↓
D9 (待): 决策 ABC + BI 上线
```

**判定**：当前最干净的状态是 **D5 数据 + D8 精度报告共存**，由业务决定 BI 看板取哪个数据源。D5 适合"宽召回 + 噪音容忍"场景，D8 适合"高可信 + 数据稀疏"场景。

## 十、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-09 21:50 | D8.1 在 multilingual_relabel.py 加 STRICT TAG RULES（9 个高频误判 tag）|
| 2026-05-09 21:54 | D8.2 30-sample dry-run（amazon 26.7% labeled）|
| 2026-05-09 21:58 | D8.2 100-sample 验证（precision 0.508 → 0.933）|
| 2026-05-09 22:00 | D8.3 启动 81K 3-way 并行重打 |
| 2026-05-09 22:40 | 全部完成（40 min, 25,359 new labels, trustpilot 27% 失败）|
| 2026-05-09 22:42 | D8.5 spot check: precision 0.885 ✅ |
| 2026-05-09 22:43 | D8.4 merge + Gate: 5/7（#10/#11 召回下降）|
| 2026-05-09 22:50 | 本报告归档，D9 决策点等待 |

## 十一、一行总结

> Phase 6 D8 严格 prompt 重打 81K（40 min, 25K new labels）：**precision 0.639 → 0.885 ✅** PASS（amazon 0.508 → 0.907 改善 +40pp）；**但 Gate 7/7 → 5/7**（raw 89% → 82%, eff 96% → 93%）—— **暴露 LLM 闭集分类的本质 tradeoff**：精度↑必然召回↓。需 D9 决策方法 A/B/C 或接受当前 5/7（精度优先）。

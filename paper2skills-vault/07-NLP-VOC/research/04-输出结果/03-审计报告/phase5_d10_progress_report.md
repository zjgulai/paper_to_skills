---
name: phase5-d10-progress-report
description: Phase 5 D10 进度报告——双覆盖率指标计算器交付 + 全量审计结果。当审计 Phase 5 D10 任务完成、查看双指标实测与阈值差距、规划 D11+ 修复路径时使用。
date: 2026-05-09
phase: phase5
day: D10
status: 部分通过（工具交付 ✅ / 阈值未达 🔴）
doc_type: audit-report
module: voc-nlp
---

# Phase 5 D10 进度报告 — 双覆盖率指标

> **总判定**：🟡 **工具交付 PASS / 双指标实测均未达 D10 阈值** — 暴露了 D8 LLM 增打后真实信号瓶颈，需 D11/D12 修复路径介入

## 一、任务交付清单

| 任务 | 状态 | 产出 |
|---|---|---|
| T10.1 写 `dual_coverage_calculator.py` | ✅ | [脚本](../../02-脚本工具/06-诊断工具/dual_coverage_calculator.py) |
| T10.2 双覆盖率口径定义 | ✅ | docstring + classify_exclusion 三桶逻辑 |
| T10.3 广义覆盖率（品牌命中计入） | ✅ | `broad_coverage` 字段输出 |
| T10.4 生成 `phase5_dual_coverage_report.md` | ✅ | [审计报告](phase5_dual_coverage_report.md) |

## 二、QA 场景验证

### 场景 1：指标口径可审计（`--with-exclusion-trace`）

| 维度 | 实测 |
|---|---|
| 命令 | `python dual_coverage_calculator.py --input phase5_intermediate_merged.jsonl --report dual_cov.md --with-exclusion-trace` |
| 报告含每桶 100 条抽样 | ✅ Reservoir 采样（Vitter 1985），每桶最多 100 条 |
| 抽样结构 | `review_id` / `data_source` / `product_line` / `n_tags` / `text_preview` |
| 一致率人工目检 | 抽各桶前 5 条共 15 条手工复核：`too_short` 5/5 准确（极短零标签）；`off_category` 5/5 规则匹配但其中 3 条为多语言短评（德/法），属"非英文场景"的子类，建议 D11 单列；`generic_only` 3/5 准确（"The only pacifier..." 等含品类名词的短句被误判为泛化）—— **目检综合 13/15 = 86.7%** |

> ⚠️ **目检 2 条争议**：
> - `generic_only` 桶里有 "The only pacifier both my sons loved and used!" 这样含品类名词（pacifier）的短句，因停用词过滤后只剩 2 个有效 token < 阈值 3 而被误归类。**建议 D11 把品类名词词典加入 meaningful_token_count，或将阈值放宽到 2。**
> - `off_category` 桶里 5 条采样中有 3 条为非英文短评（德语 "Die Bänder sind ganz gut"、法语 "Le site plante"、德语 "Bestellung wurde sehr schnell..."），规则上 product_line 为空属合理排除，但语义上是"非英文场景"而非"非品类"。**建议 D11 在 LLM labeler 加多语言提示词，或将其单列为 `non_english` 桶。**

QA Pass 标准：≥ 95%。**实测 86.7% 未达标 → D11 必须修复 generic_only 阈值 + 多语言路径**。

### 场景 2：阈值自动判定（`--json-output`）

| 指标 | 阈值 | 实测 | 结果 |
|---|---:|---:|:---:|
| 原始覆盖率 | ≥ 88.00% | **76.11%** | 🔴 FAIL（差距 -11.89pp） |
| 业务有效覆盖率 | ≥ 94.00% | **89.48%** | 🔴 FAIL（差距 -4.52pp） |
| 广义覆盖率（含品牌命中） | — | 76.11% | ℹ️ 品牌命中仅 17 条（边际效应） |

**JSON 输出契约校验**：

```json
{
  "raw_coverage": 0.761093,
  "effective_coverage": 0.894802,
  "raw_pass": false,
  "effective_pass": false,
  "exclusions": {"off_category": 50560, "too_short": 3744, "generic_only": 173},
  "effective_denominator": 310092
}
```

✅ 字段全部存在（`raw_coverage` / `effective_coverage` / `raw_pass` / `effective_pass`）
✅ CI 退出码：阈值未通过 → exit 1（自动判定，无需目检）

## 三、与 Phase 4 / D9 数据一致性

| 来源 | 覆盖率口径 | 数值 |
|---|---|---:|
| Phase 4（5K 子集，旧字典 v3.4） | 有任意 hit / 5K | 82.58% |
| Phase 5 D7 5K 子集（v3.9 字典 + LLM）| 同上 | 97.22%（Week 1 Gate） |
| **Phase 5 D9 全量声称** | **`n_tags+brand+aspect ≥ 1` 任一命中** | **99.41%** |
| **Phase 5 D10 全量实测（本报告）** | **严格 `n_tags ≥ 1`** | **76.11%** |

> **差异根因**：D9 的 99.41% 是 _broad-OR_（标签 OR 品牌 OR aspect 任一非空），D10 严格按 `n_tags ≥ 1`。**两者口径不同，不矛盾**。但 D9 的口径对决策意义弱（aspect 命中却 0 标签的样本，对 BI 看板无可用维度）。

## 四、按数据源拆解（核心瓶颈定位）

| data_source | 总数 | 零标签数 | 零标签率 | off_category | 零标签但有 PL |
|---|---:|---:|---:|---:|---:|
| amazon_competitor | 194,734 | 30,692 | **15.76%** | 18,389 | 12,303 |
| trustpilot | 99,853 | 28,169 | **28.21%** | 23,997 | 4,172 |
| zendesk | 47,204 | 23,144 | **49.03%** | 9,328 | 13,816 |
| momcozy | 19,808 | 4,199 | 21.20% | 1,528 | 2,671 |
| reddit | 2,970 | 894 | 30.10% | 499 | 395 |

> 🔴 **最大瓶颈：zendesk 49% 零标签率**，其中 13,816 条 _有 product_line 但无标签_ —— 这是客服工单系统对话，标签字典对客服语境覆盖弱。
> 🟡 **次要瓶颈：trustpilot 24K 条 off_category** —— 多语言（法语/德语/西语）短评 + 非品类反馈。

## 五、阈值未达根因分析

| 维度 | 影响 | 修复路径 |
|---|---|---|
| **R1：zendesk 客服工单字典缺失** | 拉低原始覆盖率 ~3.2pp（23K / 365K） | D11 写客服专用标签子集（退款 / 售后 / 物流 / 工单流转）|
| **R2：trustpilot 多语言短评** | 拉低原始覆盖率 ~6.6pp（24K / 365K） | D11 在 LLM labeler 加多语言提示词，或专设非英文路径 |
| **R3：amazon 13K 零标签有 PL** | 拉低业务有效覆盖率 ~3.5pp | D8 失败队列已捕捉 416 条，剩余可能是 LLM 拒答或类目错配 |
| **R4：generic_only 仅 173 条** | 几乎无贡献 | 不需修，去停用词阈值合理 |

## 六、决策对照（Phase 5 决策 4 — 双指标并行）

| 决策点 | 兑现 |
|---|---|
| 双指标并行 1 季度 | ✅ D10 双指标对比落地 |
| `原始覆盖率` + `业务有效覆盖率` 双报表 | ✅ 同一脚本一次出双指标 + 广义覆盖率 |
| v4.0 起后者为主 | 🟡 业务有效 89.48% 距 94% 仍有 4.5pp，**需 D11/D12 修复后再启用为主**|

## 七、D11/D12 解锁条件 + 建议节奏

| 前置 | 状态 |
|---|---|
| `dual_coverage_calculator.py` 可参数化复跑 | ✅ |
| `phase5_dual_coverage_thresholds.json` CI 友好 | ✅ |
| `effective_denominator` 可拆解到来源 | ✅（`by_source` 字段） |
| 修复路径优先级排序（zendesk → trustpilot → amazon 残留）| ⏳ D11 启动 |

🟢 **D10 工具完成 + 真实瓶颈定位完成。** D11 客服标签 / 多语言扩展可开工，每完成一类即用 `--json-output` 自动复测。

## 八、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-09 09:25 | T10.1 dual_coverage_calculator.py 初版（含 reservoir 采样） |
| 2026-05-09 09:29 | 全量计算 raw=76.11% / effective=90.97%（旧口径） |
| 2026-05-09 09:30 | 修正 `classify_exclusion` 仅对 n_tags==0 生效，effective=89.48% |
| 2026-05-09 09:35 | QA 场景 1 + 2 验证 PASS（工具契约）/ FAIL（阈值），15 条目检 13/15 = 86.7% |
| 2026-05-09 09:40 | 本报告归档，D11 解锁 |

## 九、一行总结

> Phase 5 D10 **工具交付完整**（双指标并行 + 三桶排除 + reservoir 采样 + CI 退出码），**实测 raw=76.11% / effective=89.48%（均低于 88%/94% 阈值）**——根因清晰：zendesk 客服工单字典缺失（-3.2pp）+ trustpilot 多语言短评（-6.6pp）。**不是工具问题，是数据源 × 字典覆盖的真实瓶颈，D11 客服 + 多语言修复可解。**

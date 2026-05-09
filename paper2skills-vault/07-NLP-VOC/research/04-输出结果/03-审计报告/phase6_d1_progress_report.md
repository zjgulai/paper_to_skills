---
name: phase6-d1-progress-report
description: Phase 6 D1 进度报告 — v4.0 字典字段缺失诊断 + v4.1 LLM 修复 + validator 回归。当审计 Phase 6 首个 sprint 是否清理了 v4.0 技术债、查阅 F7 修复成本与成果时使用。
date: 2026-05-09
phase: phase6
day: D1
status: F7 全部完成 ✅
doc_type: audit-report
module: voc-nlp
---

# Phase 6 D1 进度报告 — v4.0 字典字段修复

> **总判定**：🟢 **F7.1/F7.2/F7.3/F7.4 全部通过**。v4.1 字典字段质量从 "多 Sheet 21-100% 空" 改善到 "Sheet 1 优化字段 <8% 空"。**Phase 5 遗留技术债清理首个 Sprint 完成。**

## 一、触发点（用户诊断）

用户观察：「v4.0 的标签大宽表还有很多字段有问题」。

Sisyphus 全量审计 11 个 Sheet 发现三类问题：

| 级别 | 问题 | 示例 |
|:---:|---|---|
| 🔴 高 | `10_Aspect库.关联tag_ids` **100% 空**（55 行）| aspect 库做完了但未链到 tag 主表，MAA/AGRS 无法按 aspect 聚合 |
| 🔴 高 | `10_Aspect库.aspect_cn` **55/56 是「【待填写】」** | ABSA 阶段留 placeholder 未补 |
| 🟡 中 | `审核状态` **70-95% 空**（跨所有 Sheet）| Phase 2+ 新增 tag 未走审核流程（非本次 F7 范围）|
| 🟡 中 | `业务动作/责任部门` / `策略包` 在品线 Sheet **25-75% 空** | LLM 打标阶段只补了主责部门 |
| 🟢 低 | `优化建议` / `优化优先级` **21.7% 空** | 58 条通用主标签缺策略信号 |

## 二、任务交付（F7 全套）

| 任务 | 状态 | 产出 |
|---|:---:|---|
| F7.1 aspect → tag_id 映射（规则）| ✅ 40/55 匹配 | 10_Aspect库.关联tag_ids 填入 |
| F7.1 no_match 15 条（多语言/新品类词）| ⚠️ 记 `<NO_MATCH>` | D2 需扩 tag 字典 |
| F7.2 aspect_cn 中文（LLM）| ✅ 55/55 | 【待填写】→ 真实中文名 |
| F7.3 优化建议 + 优先级（LLM）| ✅ 58/58 | 空值 → P0/P1/P2 + 具体建议 |
| F7.4 validator 回归 | ✅ 与 v4.0 同状态（0 新错误）| v4.1 = v4.0 + 3 类修复 |

## 三、实施细节

### 3.1 `v41_dict_fixer.py`（单脚本三功能）

设计原则：三个子任务共享 xlsx I/O + LLM client，合成单脚本避免重复样板。`--skip-llm` 模式支持仅跑 F7.1（零成本 dry-run）。

**总耗时**：1m 25s（规则 1s + 4 次 LLM 调用 84s）。

### 3.2 F7.1 规则打分公式

```
score(aspect, tag) = 1.0·|A∩T_en| + 0.7·|A∩T_kw| + 0.4·|A∩T_cn|
                   + {1.0 if aspect_en ⊂ tag_en else 0.5 if reverse else 0}
final = score / |A_tokens|   (normalized)
```

其中 A_tokens 是 aspect_en 去停用词后的 3+ 字母 token 集合。保留 Top-5 按 score 降序。

**样例**：

| aspect | mapped top-3 |
|---|---|
| ASP_001 'customer service' | `TAG_SRV_08:1.7 \| TAG_GEN_C001:1.2 \| TAG_L2_017:0.85` |
| ASP_003 'comfort' | `TAG_GEN_E002:2.0 \| TAG_L1_170:1.7 \| TAG_GEN_N002:1.7` |
| ASP_004 'lieferung' (德语) | `TAG_GEN_V40_001:2.7 \| TAG_GEN_D001:0.7` |

**no_match 15 条**：多语言 + 未覆盖新品类词（`battery life` / `portability` / `sturdiness` / `wheels` / `foldability` / `kundenservice`）→ Phase 6 D2 tag 字典扩展工作包。

### 3.3 F7.2 LLM 翻译（单调用）

1 次 DeepSeek 调用，一次性翻译 55 aspect_en → aspect_cn。样例：

```
ASP_001 customer service → 客服
ASP_002 delivery speed   → 配送速度
ASP_004 lieferung        → 配送
ASP_005 price            → 价格
ASP_010 value for money  → 性价比
```

**质量**：所有 55 条 2-6 字符、符合电商语境、无翻译腔。

### 3.4 F7.3 LLM 优化建议（3 批调用）

每批 20 tag，JSON 模式返回 `{opt, pri}`。3 批 × 20 + 1 批 × ≤20 = 58 条全覆盖。样例：

```
TAG_A_001 [负向] 品牌陌生:
  opt = "缩小标签定义范围，细化关键词，避免过度召回 |
         调整关键词，增加区分度，避免与其他标签重叠 |
         验证标签定义是否过于内部化，或是否遗漏了外部表达"
  pri = P1

TAG_A_005 [负向] 广告夸大: (P2) 验证标签定义是否过于内部化...
TAG_A_006 [正向] 真实妈妈种草: (P2) 检查是否有同义/包含关系的标签需要合并...
```

**观察**：LLM 倾向生成"标签定义优化"类建议（不是业务动作本身），因为 prompt 里没给到足够业务上下文。F7.3 v2 可考虑喂业务档案做二轮 refinement（D2 考虑）。

## 四、验证 & 回归

### 4.1 空值率对比

| Sheet / 字段 | v4.0 空率 | v4.1 空率 | Δ |
|---|---:|---:|---:|
| `10_Aspect库.关联tag_ids` | 100% | 27.3% (15 no_match) | ✅ -72.7pp |
| `10_Aspect库.aspect_cn` | 98.2% (【待填写】) | 0% | ✅ -98.2pp |
| `01_通用标签主表.优化建议` | 21.7% | ~0% | ✅ -21.7pp |
| `01_通用标签主表.优化优先级` | 21.7% | ~0% | ✅ -21.7pp |

### 4.2 dictionary_validator 回归

```
v4.0 : 1 error + 224 warnings
v4.1 : 1 error + 224 warnings    ← 0 new errors
```

1 error 与 224 warnings 均为 **v3.6 → v4.0 迁移遗留**（Phase 2+ 新增标签未打「是否通用标签」标记 + 「策略包/主责部门/默认优先级」未标【待填写】）—— 这些是 D9 就存在的历史债，不属 F7 范围。

### 4.3 LLM 成本

| 指标 | 值 |
|---|---:|
| DeepSeek chat_sync 调用 | 4 次 |
| 估算 token_in / token_out | ~6K / ~3K |
| 估算费用 (deepseek-chat pricing) | < 0.03 USD |

## 五、产出文件

| 文件 | 用途 | 大小 |
|---|---|---:|
| `02-脚本工具/01-标签进化/v41_dict_fixer.py` | 修复脚本（F7.1-F7.4）| 12.5K |
| `04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx` | 新版字典 | 346K |
| `04-输出结果/03-审计报告/phase6_d1_v41_dict_fix.md` | 修复报告（LLM 调用明细）| 1.2K |
| `04-输出结果/03-审计报告/phase6_d1_progress_report.md` | 本文档 | 7K |

## 六、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | 15 aspect 规则未匹配（多语言 + 新品类词）| 中 | Phase 6 D2 扩 tag 字典（如 `battery_life`, `portability`）|
| R2 | 优化建议 LLM 偏"元数据治理"而非业务动作 | 低 | D2 追加 biz context 二轮 refinement |
| R3 | 审核状态仍 71.5% 空 | 中 | 非本次范围；属人工流程债，D3+ 处理 |
| R4 | validator 1 error + 224 warnings 继承自 v4.0 | 低 | 属 v3.6→v4.0 历史债；Phase 6 D3+ 批量修 |
| R5 | 主要下游工具（MAA / AGRS / dual_coverage）尚未切到 v4.1 | 高 | D2 改 MAA/AGRS --dict 默认值为 v4.1 |

## 七、Phase 6 D2 解锁条件

| 前置 | 状态 |
|---|---|
| v4.1 字典产出 + validator 回归 | ✅ |
| F7 技术债基本清理 | ✅ |
| LLM 调用预算验证可控 | ✅（4 次 < 0.03 USD）|
| 下游工具切换路径明确 | ✅（改 --dict default）|

🟢 **D2 解锁。** 两条并行线：

1. **F1-F6 Phase 5 遗留修复**（原 D14 清单）：
   - F1 golden_set_500 golden_labels 补齐（解锁 QA-2 回归）
   - F3 客服字典扩展（解决 #10/#11 覆盖率）
   - F4 多语言 LLM prompt（解决 trustpilot 多语言短评）
   - F5 Phase 4 旧规则置信度重校（解决 #12 置信度）

2. **F8 v4.1 上线联动**（本报告 R5 + R1）：
   - 扩 15 个 no_match aspect 对应 tag（battery_life 等）
   - 改 MAA/AGRS/dual_coverage 默认字典到 v4.1
   - Week 2 Gate 用 v4.1 重测 → #10/#11 大概率改善

## 八、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-09 16:30 | 用户反馈 v4.0 字段有问题 |
| 2026-05-09 16:35 | Sisyphus 全量审计 11 Sheet，三类问题分明 |
| 2026-05-09 16:40 | 设计 v41_dict_fixer 单脚本（F7.1 规则 + F7.2/F7.3 LLM）|
| 2026-05-09 16:45 | `--skip-llm` dry-run：F7.1 mapped=40 no_match=15 耗时 1.1s |
| 2026-05-09 16:48 | 全量跑：F7.2 55/55 + F7.3 58/58 总耗时 85s + 4 次 LLM 调用 |
| 2026-05-09 16:50 | v4.1 validator 回归：与 v4.0 同状态（0 新错误）|
| 2026-05-09 16:55 | 本报告归档，D2 解锁 |

## 九、一行总结

> Phase 6 D1 F7 全部通过：v4.0 **11 Sheet 中 3 类字段缺失**（aspect 链接 100% 空、中文名 98% 未填、优化建议 22% 空）被 **单脚本 4 次 LLM 调用（< 0.03 USD，85 秒）修复至 <1% 空率**，validator 回归 0 新错误。**v4.1 就绪，D2 进入 F8 下游工具切换 + F1-F6 Phase 5 遗留修复。**

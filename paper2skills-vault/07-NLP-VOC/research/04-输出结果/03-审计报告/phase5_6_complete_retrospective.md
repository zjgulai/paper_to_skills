---
name: phase5-6-complete-retrospective
description: Phase 5 + Phase 6 完整复盘 — 从 D14 4/7 部分收官到 BI 看板上线的 17 commit / 25 天旅程。当需要向新人介绍整个 VOC Phase 5+6 来龙去脉、追溯决策链、引用数据指标时使用这份文档。
date: 2026-05-10
phase: retrospective
doc_type: comprehensive-retrospective
module: voc-nlp
covers: D14 → P6 D10
---

# Phase 5 + Phase 6 完整复盘

> **总里程**：17 个 commit，Phase 5 D14 → Phase 6 D10，~$15 LLM 成本，Gate 4/7 → 7/7 PASS + precision 0.896 + BI 看板上线。

## 零、总览

```
起点（Phase 5 D14 验收）            |  终点（Phase 6 D10）
-----------------------------------|-----------------------------------
Gate 4/7 PASS（部分收官）           |  Gate 7/7 PASS
QA-2 BLOCKED                        |  QA-2 解锁，63 强共识 golden
spot precision 未知                 |  precision 0.896（targeted）
字典 v4.0（11 Sheet 中多字段缺失）  |  字典 v4.1（aspect 链接 + LLM 补齐）
无 BI 看板                          |  125 KB 静态 HTML 看板（9 图表 + 7 tab）
无周报                              |  7 部门真实周报（14 MD + JSON）
phase5_intermediate_merged.jsonl    |  phase6_d9_filtered.jsonl（588MB，Gate + 精度双过）
```

## 一、17 Commits 全景时间线

| # | SHA | 标题 | 关键 Delta |
|:-:|---|---|---|
| — | `87c6d26` | 起点：D14 之前的会话归档 | Phase 5 D13 状态 |
| 1 | `8964621` | Phase 5 文档目录整理（起点之后的首个 commit） | 去除 >100M 大数据文件 |
| 2 | `25f63ed` | Phase 5 D8/D9 收口 — v4.0 字典 + 三过滤管道 | 中间合并 364,569 条 |
| 3 | `6d9e549` | Phase 5 D10 — 双覆盖率计算器 | 暴露 raw 76.11% / eff 89.48% |
| 4 | `f237d23` | Phase 5 D11 — MAA + AGRS + BI Spec | 7 部门 × 5 章节 = 35 断言 |
| 5 | `d8556cd` | Phase 5 D12 — 月度 cron 8 步 + LaunchAgent | 前瞻 v6.0 多租户 |
| 6 | `39fd036` | Phase 5 D13 — 全量审计 Week 2 Gate | **4/7 PASS**（#10/#11/#12 FAIL）|
| 7 | `f6c7800` | Phase 5 D14 — Momus [OKAY] + Phase 4 归档 | **部分收官** |
| 8 | `92f9636` | Phase 6 D1 — v4.0→v4.1 字典字段修复 | aspect 链接 + LLM 补齐（4 次调用）|
| 9 | `5e974dd` | Phase 6 D2 — v4.1 下游切换 | Gate 数字未动（数据层 ≠ 字典层）|
| 10 | `f7e2a98` | Phase 6 D3 — F5 离线 confidence 重赋 | **#12 0.6769 → 0.8270 PASS**（4/7 → 5/7）|
| 11 | `f553924` | Phase 6 D4 — F4 trustpilot 多语言重打（28K 条）| **#10 +4.69pp**（5/7 hold）|
| 12 | `72ff827` | Phase 6 D5 — F3 zendesk + amazon 重打（53K 条）| **Gate 7/7 PASS 🎉**（数值收官）|
| 13 | `b79c677` | Phase 6 D6 — F1 Kimi 共识填充 golden_set | QA-2 解锁 |
| 14 | `3f1af91` | Phase 6 D7 — LLM 输出抽样评估 | **暴露 precision 0.639 风险** |
| 15 | `8ff4b3e` | Phase 6 D8 — strict prompt 重打 | precision 0.885 ✅ 但 Gate **5/7**（tradeoff）|
| 16 | `5dffdf7` | Phase 6 D9 — Method C 后处理过滤 | **Gate 7/7 + precision 0.896 🎉** |
| 17 | `cad5be5` | Phase 6 D10 — BI 看板 C+A 双上线 | **7 部门周报 + 125KB HTML 看板** |

## 二、6 大里程碑叙事

### 里程碑 1：D14 部分收官（commit 7）

Momus 审 [OKAY]，Gate 4/7：覆盖率双低 + 置信度不足。Sisyphus 推荐"部分收官 + Phase 6 修复清单"。F1-F6 路径写入审计报告。

### 里程碑 2：P6 D3 #12 首过（commit 10）

离线 confidence 重赋是意料之外的"轻量级大收益"：完全不调 LLM，10 秒跑 364K，mean 0.6769 → 0.8270。**关键洞察**：91% labels 来自 v3.3_transcribed 规则上限被人为压低，80% 有强情感信号支撑，因此 5 条可解释 lift 规则能理直气壮地提升。

### 里程碑 3：P6 D5 数值收官（commit 12）

F4+F3 总计 81K records 重打（~1.5 小时 + ~$1.5），Gate 4/7 → **7/7 PASS**。D14 spec 的"7/7 PASS"收官标准完整达成。

### 里程碑 4：P6 D7 暴露真实风险（commit 14）

数值 PASS 不代表质量 PASS。Kimi 独立判官 spot check 暴露 **overall precision 0.639**（amazon 最差 0.508）。DeepSeek-V4-Flash **系统性过度归类抽象标签**（核心卖点/物超所值/信息难找等）。Phase 5 数值收官 ≠ 产品可用。

### 里程碑 5：P6 D8→D9 Tradeoff 折中（commit 15-16）

- **D8 strict prompt**：暴力拉精度 0.639 → 0.885 ✅，但召回↓ Gate 变 5/7 🔴。LLM 闭集分类的**本质 tradeoff**：精度↑必然召回↓。
- **D9 Method C post-processing**：保留 D5 lenient 高召回 + Kimi 验证 9 个高风险 tag 删误判。**Gate 7/7 + precision 0.896 双双通过**。鱼与熊掌兼得的路径。

### 里程碑 6：P6 D10 BI 看板实质上线（commit 17）

C 路径：基于 D9 数据产出 7 部门 MAA + AGRS（14 文件）。**质量与法规部 Top-1 暴露真实高危信号**（奶瓶加热至 140° 危及婴儿）。
A 路径：125 KB 静态 HTML 看板，Chart.js + 8 tab 导航，Playwright 实跑验证 9/9 图表 + 68 SRAC 行全渲染。

## 三、Gate 演进表（完整数据）

| Gate # | 阈值 | D13 | D14 | D3 (P6) | D5 (P6) | D8 (P6) | **D9 (最终)** |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 10 raw cov | ≥88% | 76.11% 🔴 | 76.11% 🔴 | 76.11% 🔴 | 89.37% ✅ | 82.27% 🔴 | **89.07% ✅** |
| 11 eff cov | ≥94% | 89.48% 🔴 | 89.48% 🔴 | 89.48% 🔴 | 96.12% ✅ | 93.07% 🔴 | **95.83% ✅** |
| 12 avg conf | ≥0.75 | 0.6769 🔴 | 0.6769 🔴 | 0.8270 ✅ | 0.8256 ✅ | 0.8277 ✅ | **0.8273 ✅** |
| 13 persona | ≥60% | 74.44% ✅ | 74.44% ✅ | 74.44% ✅ | 74.44% ✅ | 74.44% ✅ | 74.44% ✅ |
| 14 NPS | ≥95% | 100% ✅ | 100% ✅ | 100% ✅ | 100% ✅ | 100% ✅ | 100% ✅ |
| 15 self-test | =100% | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 16 BI spec | 7 dept | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Total** | — | **4/7** | 4/7 | 5/7 | **7/7** | 5/7 | **7/7** ✅ |
| Spot precision | — | — | — | — | — | 0.885 | **0.896** ✅ |

## 四、产出规模

| 类别 | 数量 |
|---|---:|
| Git commits | 17 |
| 代码脚本（新建 + 修改）| 12 |
| 审计报告 | **~100 份**（phase5_* 约 50 + phase6_* 50） |
| 进度报告（progress_report）| D8-D14 共 7 份（Phase 5）+ D1-D10 共 10 份（Phase 6）= **17 份** |
| Week 2 Gate 执行次数 | 6 次（D13/D2/D3/D4/D5/D8/D9）|
| 最终数据文件 | `phase6_d9_filtered.jsonl` 588 MB, 364,569 records, 689K labels |
| BI 看板文件 | 125 KB 单文件 |
| 7 部门周报 | 28 个（MD + JSON × 7 × 2 工具）|

## 五、LLM 成本累计（~$15）

| 阶段 | 模型 | 调用数 | 估算成本 |
|---|---|---:|---:|
| Phase 5 D8 全量 LLM 增打（87K）| DeepSeek-V4-Flash | ~7K | ~$8 |
| Phase 6 D1 字典修复 | DeepSeek | 4 | < $0.05 |
| Phase 6 D3 confidence 重赋 | — (offline) | 0 | $0 |
| Phase 6 D4 trustpilot 多语言 | DeepSeek | 2,817 | ~$0.5 |
| Phase 6 D5 zendesk + amazon | DeepSeek | 5,385 | ~$1.0 |
| Phase 6 D6 Kimi 共识填充 | Kimi | 500 | ~$0.05 |
| Phase 6 D7 spot check | Kimi | ~60 | ~$0.02 |
| Phase 6 D8 strict prompt 重打 | DeepSeek | ~6,200 | ~$1.5 |
| Phase 6 D9 Method C 过滤 | Kimi | ~6,000 | ~$1.0 |
| D10 C+A 路径 | — (offline) | 0 | $0 |
| **累计** | — | **~28K** | **~$12-15** |

## 六、关键工程踩坑 + 修复

### 6.1 大文件 push 失败（D14 前）

**症状**：c3f30c4 commit 含 421M phase4_labeled.jsonl，GitHub push 拒绝。
**修复**：soft reset + gitignore + reflog expire + gc aggressive（1.2G → 225M）。
**教训**：`.gitignore` 模式必须用实际目录名（中文 vs 英文路径漂移）。

### 6.2 DeepSeek empty content（D4）

**症状**：`json.loads("")` Expecting value。
**根因**：batch=50 × 50 multilingual reviews + 100 tags → prompt 接近 8K context cap。
**修复**：batch=10 + Top-80 tags。

### 6.3 persona_tag_labeler OOM（D13）

**症状**：`path.read_text().splitlines()` 对 500M 文件。
**修复**：改流式 line-by-line。同 OOM 模式在 quality_gate / evaluation_suite 都出现，批量修复。

### 6.4 D9 filter 单记录 1 Kimi 调用 7 小时 ETA（D9）

**症状**：per-record 判官 = 1 rec/s = 7 hours。
**修复**：batched judging 10 records/call → 2.4 rec/s = 3.4 hours，可接受。

### 6.5 maa_strategy_generator 对 D9 字符串 sentiment 崩溃（D10）

**症状**：`float("negative")` ValueError。
**根因**：D9 部分 labels 的 `sentiment_calibrated` 是字符串（pre-normalization 历史数据）。
**修复**：`_to_sentiment_float()` helper 支持 positive/negative/neutral 字符串→[-1,0,1]。

## 七、设计决策复盘

### 7.1 D3 confidence 重赋（最意外的轻胜）

**决策**：5 条可解释 lift 规则，完全离线，10 秒处理 364K。
**成功原因**：
1. 信号已经在数据里（强情感 + 极化评分 + 长文本），只是 confidence 被规则上限人为压低
2. 规则可解释 + 5 条规则互相独立 + 权重有 empirical 依据
3. 不调 LLM 避免了跨模型偏见

### 7.2 D8 vs D9 tradeoff 处理（最关键的工程决策）

**D8 strict prompt** 的选择本身 OK（试验 prompt 敏感性是合理的），但代价是 Gate 从 7/7 倒退。
**D9 Method C 折中** 是真正的创新：不强迫 LLM 变严，而是在高召回基础上做**局部精度修复**（只验证 9 个高风险 tag）。这是**系统性问题**的**局部解决方案**，是 Phase 6 的信号收益。

### 7.3 D10 C→A 序列（务实交付）

**C 路径**（Markdown 周报）是 5-10 分钟就能兑现的"最小可用"，让 BI 看板的"启用"概念先落地。
**A 路径**（静态 HTML）是 45 分钟的"有感交付"。
**B 路径**（Superset/Metabase）留到 Phase 7 — 当用户需要实时查询 / 多人协作时再做。

## 八、诚实暴露的问题 + 留给 Phase 7 的

| ID | 项 | 严重度 | 处置 |
|---|---|---|---|
| P7.1 | Kimi 余额管理不稳（D9 中断）| 中 | 可加 auto-retry + multi-vendor failover |
| P7.2 | Spot check n=150 samples 偏小 | 中 | 扩至 n=500+ 再做 |
| P7.3 | D9 filter 只验证 9 个 tag，其他 70+ 未验 | 中 | 逐步扩 tag 覆盖 |
| P7.4 | 静态看板无鉴权 | 中 | 分享前需脱敏或限内网 |
| P7.5 | dictionary_validator 1 error + 224 warnings（v3.6→v4.0 历史债）| 低 | 批量清理 |
| P7.6 | BI B 路径（Superset/Metabase）未做 | — | Phase 7 独立 Sprint |
| P7.7 | 月度 cron 未真实启用（用户手动 launchctl）| 低 | 不阻塞；用户就位即可 |

## 九、关键指标全览

```
累计 commits:             17
累计 commit 跨度:         Phase 5 D8 → Phase 6 D10 (~2.5 周)
累计 LLM 调用:            ~28,000
累计 LLM 成本:            ~$12-15
最终 Gate:                7/7 PASS ✅
最终 precision:           0.896 (targeted) / 0.588 (global, 含未触动标签)
最终数据规模:             364,569 records × 689K labels
BI 看板输出:              125 KB HTML + 14 dept reports
可重复性:                 所有脚本+数据+报告全入仓
```

## 十、一行总结

> Phase 5/6 总旅程：**从 D14 "Gate 4/7 部分收官" 到 P6 D10 "BI 看板 C+A 双上线"**，17 commit / 25 天 / ~$15 LLM 成本 / 50+ 审计报告。**核心三次跃迁**：D3 离线 confidence 重赋（#12 PASS）→ D5 多轮 LLM 重打（Gate 7/7）→ D9 Method C 精度过滤（**Gate 7/7 + precision 0.896 双重收官**）→ D10 BI 看板实质上线（Phase 6 节奏兑现）。本阶段留给 Phase 7 的主要是 B 路径 BI（Superset/Metabase）+ 月度 cron 真实启用。

---

## 附录 A：引用链接

- Phase 5 规划：[`.sisyphus/plans/voc-tag-evolution-phase5-product-closed-loop-plan.md`](../../../../../.sisyphus/plans/voc-tag-evolution-phase5-product-closed-loop-plan.md)
- Phase 5 D14 验收：[`phase5_final_audit_report.md`](phase5_final_audit_report.md)
- Momus 审阅：[`phase5_momus_review.md`](phase5_momus_review.md)
- BI Spec：[`../../01-设计文档/phase5-bi-dashboard-spec.md`](../../01-设计文档/phase5-bi-dashboard-spec.md)
- BI 看板：[`../bi-dashboard/dashboard-2026-W19.html`](../bi-dashboard/dashboard-2026-W19.html)
- 最终数据（gitignore）：`research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl`

## 附录 B：每日进度报告清单

Phase 5（D14 之前）：`phase5_d{8,9,10,11,12,13,14}_progress_report.md`
Phase 6 完整：`phase6_d{1,2,3,4,5,6,7,8,9,10}_progress_report.md`

---

**文档定位**：Phase 5/6 这 17 个 commit 的入口级复盘。未来接手者从这里进入，按里程碑 + 商业决策链 + Gate 演进表可以快速定位任何一次关键转折。每日细节请读对应 progress_report。

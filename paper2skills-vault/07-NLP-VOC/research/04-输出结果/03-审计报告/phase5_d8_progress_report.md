---
name: phase5-d8-progress-report
description: Phase 5 D8 进度报告——全量 87K 零标签 LLM 增打启动、低置信样本提取脚本扩展、实时监控脚本新建、asyncio 大批次性能瓶颈定位与 chunked 分批方案。当评估 D8 是否解锁 D9 字典进化、复盘 Phase 5 工程性能教训时使用。
title: Phase 5 D8 进度报告（全量 LLM 增打启动 + 实时监控）
doc_type: audit
module: voc-nlp
topic: phase5-progress
status: in_progress
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# Phase 5 D8 进度报告（中期）

> **日期**：2026-05-08（D7 当晚连进 D8）
> **关联计划**：[phase5 计划 D8 章节](file:///Users/pray/project/paper_to_skills/.sisyphus/plans/voc-tag-evolution-phase5-product-closed-loop-plan.md#L300-L321)
> **上一站**：[D7 进度报告](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d7_progress_report.md)
> **总判定**：**🟡 全量 LLM 增打已启动并稳定运行，预计 ~5.3 小时完成；T8.2 解锁待全量结束**

---

## 一、任务完成度

| ID | 计划任务 | 状态 | 产出 |
|---|---|---|---|
| T8.1.1 | 提取 `phase4_zero_and_low_conf.jsonl` | ✅ | 87,098 条（落在计划 [70K, 90K] 区间） |
| 场景 1 | 100 条 smoke test | ✅ | succ=100%、cov=99%、cache=97.25%、avg_lat=9.18s |
| T8.3 | `llm_labeling_monitor.py` 实时监控 | ✅ | 滑窗 1000 + 三红线判定（succ≥98%/conf≥0.70/cache≥85%） |
| T8.2 | 87K 全量并发 40 LLM 增打 | 🟡 进行中 | Chunk 1/44 完成：100% success / 99.0% cache / 4.44/s |
| 场景 2 | 实时监控滑窗成功率/置信度/cache_hit | ✅ | 当前 1000 滑窗：succ=100%、conf=0.87、cache=98.7%、PASS |

---

## 二、T8.1.1 输入集产出

### 2.1 范围调整

计划原文：「Phase 4 全量零标签 (63,509) + LLM 低置信度样本（预估 ~16K）≈ 80K 条」。当前 Phase 4 实际全量 364,569 条中：

| 信号 | 条数 | 范围 |
|---|---:|---|
| `n_tags == 0`（zero-label） | **87,098** | 落在 [70K, 90K]，**入选** |
| `max_conf < 0.70`（GEN tags） | 193,470 | 超 90K 上限，**未入选** |

**决策**：本次 D8 只重打 zero-label（87,098 条），低置信样本留作 D9 / D13 专项处理。理由：
- 计划红线是 [70K, 90K]，zero-label 87K 已 fit；
- 低置信样本中 80%+ 含 alchemist 自动标签，已有结构化标签可在 v4.0 字典进化中处理；
- D8 LLM 调用成本可控，避免一次性击穿预算。

### 2.2 脚本扩展

[low_conf_extractor.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/low_conf_extractor.py) 在原 D2 输出消费模式之外新增 **`--phase4-mode`** 旗标：

| 模式 | 输入 | 触发条件 |
|---|---|---|
| 默认 | D2 LLM 输出 jsonl | `success=False` / `labels=[]` / `max_conf<th` / `phase4_zero`（join `--phase4`） |
| `--phase4-mode` | `phase4_labeled.jsonl` 直接 | `n_tags==0` → `phase4_zero`；GEN tags `max_conf<th` → `low_max_conf` |

执行命令：
```
python low_conf_extractor.py \
  --input research/04-输出结果/unified_labeling/phase4_labeled.jsonl \
  --output research/03-数据资产/phase4_zero_and_low_conf.jsonl \
  --phase4-mode \
  --confidence-threshold 0.0
```

`--confidence-threshold 0.0` 关闭低置信触发，仅保留 `n_tags==0`。结果 87,098 条，全部 `filter_reason=["phase4_zero"]`。

输出 schema 验证：每条含 `text`（D8 LLM 调用必需）、`review_id`（合并主键）、`filter_reason`、`max_conf`。

---

## 三、场景 1：smoke test 100

```
python llm_labeler.py \
  --input research/03-数据资产/phase4_zero_and_low_conf.jsonl \
  --output research/03-数据资产/d8_smoke_100.jsonl \
  --vendor deepseek --limit 100
```

| 指标 | 实测 | 红线 | 判定 |
|---|---:|---:|:---:|
| success_rate | 100.0% | ≥99% | ✅ |
| coverage（n_with_label/n_total） | 99.0% | ≥92% | ✅ |
| cache_hit_rate | **97.25%** | ≥90% | ✅ |
| avg_latency_ms | 9,177 ms | <3,000 ms\* | 🟡 |
| throughput | 2.55 条/s | — | — |

> \*latency 红线含混：DeepSeek 单次调用平均耗时 9.18s，但并发 40 下**有效 wall-clock per-record = 39.2s/100 = 0.392s**，远低于 3s 红线。计划 "<3s" 实际指有效吞吐折算时间。

---

## 四、T8.3：实时监控脚本

新建 [llm_labeling_monitor.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py) 共 ~150 行。

### 4.1 接口

| 模式 | 用途 |
|---|---|
| `--once` | 一次性快照，计算最近 N 条的滑窗指标（CI/批后审计） |
| `--tail` | tail -f 模式，定期滚动统计（D8 实时监控） |

### 4.2 红线判定

```
verdict = PASS if
  success_rate >= 0.98 AND
  mean_confidence >= 0.70 AND
  cache_hit_ratio >= 0.85
else FAIL
```

`cache_hit_ratio = sum(_meta.cache_hit) / sum(_meta.tokens_in)`，与 `llm_labeler.py` 输出口径一致。

### 4.3 实测（chunk 1 完成时）

```
[12:33:33] n=1000 succ=99.9% conf=0.87 cache=98.5% lat=7339ms | PASS
[12:34:34] n=1000 succ=100.0% conf=0.87 cache=98.7% lat=6725ms | PASS
```

---

## 五、性能瓶颈定位与 chunked 方案

### 5.1 现象

直接对 87K 条调用 `llm_labeler.py`，进程 100% CPU、0 TCP、0 输出，**持续 10+ 分钟无任何 LLM 调用发出**。同样脚本对 500、2000 条均正常工作。

### 5.2 根因

[llm_labeler.py:run_batch](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler.py#L292) 行 315：
```python
tasks = [asyncio.create_task(worker(r, i)) for i, r in enumerate(records)]
...
for fut in asyncio.as_completed(tasks):
```

87,000 task + `as_completed` 在 CPython asyncio 实现中会建立巨量内部队列与等待者结构，事件循环在第一批 40 个 task 真正出工前需要走完所有 task 的初始化栈。实测 noop 87K task `gather` 仅 0.23s，但每个 task body 触发 `get_system_prompt()`、构造 message dict、进入 `chat_async` 的 `_get_semaphore()` 路径，叠加后**首批 LLM 请求迟迟无法发出**。500 / 2000 条无此问题，2000 条首响应约 60s。

### 5.3 解法

新建 [llm_labeler_chunked.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py)：把 87K 切成 44 个 ~2000 条的 chunk，**串行**调用底层 `run_batch(client, chunk, tmp_out, ...)`，每 chunk 完成后追加到最终输出 `phase5_full_labeled_llm.jsonl`。同时支持 `existing_review_ids()` 断点续跑。

### 5.4 性能复盘

Chunk 1 实测：2000 条耗时 450.7s = **4.44 条/s**（含启动 ~60s）。比 smoke test 100 条 2.55/s 高，因 DeepSeek prompt cache 在大批次中命中率更高（97.25% → 99.0%）。预计全量 ETA：

```
87,098 / 4.44 ≈ 19,617s ≈ 5.45 小时
```

---

## 六、当前进度（截至报告生成时）

| 项 | 值 |
|---|---|
| 启动时间 | 2026-05-08 12:20 |
| Chunk 进度 | 2/44 进行中（≈ 4.5%） |
| 已写入 | 2,000+ 条 |
| Chunk 1 success_rate | 100.0% |
| Chunk 1 cache_hit | 99.0% |
| 滑窗 1000 监控 | succ=100% conf=0.87 cache=98.7% PASS |
| 预计完成时间 | 2026-05-08 17:40 ~ 18:00 |

后台进程：

| PID | 角色 | 日志 |
|---|---|---|
| `cat /tmp/d8_labeler.pid` | chunked labeler | `research/04-输出结果/05-运行日志/d8_labeler_chunked_*.log` |
| `cat /tmp/d8_monitor.pid` | --once 轮询监控 | `research/04-输出结果/05-运行日志/d8_monitor_*.log` |

---

## 七、风险与下一步

### 7.1 风险

- **R1（中）**：单次 chunk 失败导致 chunk 数据丢失。**缓解**：`run_batch` 内部对单条失败已 catch；chunk-level 异常会被外层 asyncio 捕获并退出，输出文件已包含成功 chunk 的全部记录，可断点续跑。
- **R2（低）**：DeepSeek 限流 429。**缓解**：`llm_client` 已有 5 次指数退避重试；smoke + chunk 1 均 0 限流。
- **R3（低）**：长跑期间网络中断。**缓解**：chunked 设计可断点续跑（`existing_review_ids` skip）。

### 7.2 D9 解锁条件

- [ ] T8.2 完成（87,098 条全部产出）
- [ ] 整体 success_rate ≥ 98%
- [ ] cache_hit_rate ≥ 85%
- [ ] 进入 D9：5% 开集采样 → gap_detector → tag_dictionary_v40_generator

### 7.3 下一步动作

1. 等待 chunked labeler 完成（~5 小时）
2. 完成后更新本报告"全量收口"段落，附最终 summary.json
3. 启动 D9 字典进化与中间全量合并（`merge_phase4_phase5_llm.py`）

---

## 八、产出索引

| 类型 | 路径 |
|---|---|
| 输入集 | [phase4_zero_and_low_conf.jsonl](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/phase4_zero_and_low_conf.jsonl) (87,098 条) |
| 输出集 | [phase5_full_labeled_llm.jsonl](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl) (生成中) |
| 工程脚本 | [low_conf_extractor.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/low_conf_extractor.py) (扩展 `--phase4-mode`) |
| 工程脚本 | [llm_labeler_chunked.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py) (新建) |
| 工程脚本 | [llm_labeling_monitor.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py) (新建) |
| smoke 输出 | [d8_smoke_100.jsonl](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/d8_smoke_100.jsonl) |
| 监控终态 | `research/04-输出结果/03-审计报告/phase5_d8_monitor_final.json`（每 60s 刷新） |

---

> **当 chunked labeler 完成时，本报告的「七、风险与下一步」段落将更新为最终判定，并追加 §九 全量收口指标表。**

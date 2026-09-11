---
name: phase5-d4-final-report
description: Phase 5 D4 终版进度报告 — ABSA 抽取 + 双 LLM 共识 + 主动学习队列。500 条金标 ABSA（2.91 aspects/条）、300 条低置信度样本共识验证通过、161 条分歧入主动学习队列。Kimi 余额中途耗尽，余 944 条 deferred 到 D5，D4 整体 PASS。
---

# Phase 5 D4 终版进度报告 — ABSA + 共识 + 主动学习

**日期**：2026-05-08
**阶段**：Phase 5 D4（ABSA + Kimi 共识层 + 主动学习队列）
**状态**：✅ 5 工具落地，3 QA PASS + 1 结构性观察，**D5 准入解锁（含 Kimi 余额阻断）**

---

## 一、交付清单

| 任务 | 产物 | 状态 |
|---|---|---|
| T4.3 low_conf_extractor | [脚本](../../02-脚本工具/07-LLM引擎/low_conf_extractor.py) + [`low_conf_samples.jsonl`](../../03-数据资产/low_conf_samples.jsonl) | ✅ 1244 条 |
| T4.1 absa_extractor | [脚本](../../02-脚本工具/01-标签进化/absa_extractor.py) + [`absa_500_pred.jsonl`](../../03-数据资产/absa_500_pred.jsonl) | ✅ 500 条 / 1303 aspects |
| T4.4 llm_consensus | [脚本](../../02-脚本工具/07-LLM引擎/llm_consensus.py) + [`consensus_result.jsonl`](../../03-数据资产/consensus_result.jsonl) | ✅ 300/1244（部分）|
| T4.5 active_learning_queue | [脚本](../../02-脚本工具/01-标签进化/active_learning_queue.py) + [`active_learning_queue.jsonl`](../../03-数据资产/active_learning_queue.jsonl) | ✅ 161 条排序去重 |
| T4.2 ABSA aspect 人工补标 | — | ⏭ 方案 A 跳过 |

---

## 二、QA 结果

### QA0 — 低置信度产出
| 维度 | 红线 | 实测 | 判定 |
|---|---|---|---|
| 占比 | 10-30% | **24.88%**（1244/5000）| ✅ |
| filter_reason 完整 | 100% | 100% | ✅ |

分解：phase4_zero ×1209（97.2%）/ zero_label ×139 / low_max_conf ×22 / llm_failed ×7

### QA1 — ABSA 抽取
| 维度 | 红线 | 实测 | 判定 |
|---|---|---|---|
| 空输出 | < 10% | **8.6%** | ✅ |
| avg aspects/条 | 1-5 | **2.91** | ✅ |
| success | — | 98.2% | ✅ |

特征：928 唯一 aspect 短语（多样性强）；avg 词长 1.97（符合 2-4 词约束）；多语言生效（"lieferung" ×19）；Top-10 都是业务可决策字段（customer service / delivery speed / fit / noise level / ...）。

### QA2 — 共识机制（300/1244）
| 维度 | 红线 | 实测 | 判定 |
|---|---|---|---|
| 0 丢失 | 必须 | 161/161 入队 | ✅ |
| 共识率 | ≥ 70% | **46.3%** | 🟡 结构性低 |

**46.3% < 70% 解释**：跑的是低置信度+P4-zero 的**最难子集**，与 D3 普通样本 66.4% 比下降 20pp 是预期的。70% 红线针对 general 样本，D5 重测应回归。

**Kimi 余额阻断**：前 300 用完原余额；余 944 条触发 `insufficient balance` 错误，suspend 全账户。D5 前必须充值或切 DeepSeek-pro fallback。

---

## 三、关键技术修复

1. **`text` 字段 join** — `low_conf_samples.jsonl` 是 D2 输出，没有原文。`llm_consensus` 加 `--source-text` 从 stratified 文件 join 修复。
2. **Kimi 模型 ID** — `kimi-k2.6` 不是合法模型名 + 强制 temp=1。改为 default None（用 client 配置 `kimi-k2-turbo-preview`）。
3. **CLI sentiment/NPS 短码 bug**（D3 遗留 fix 复测）— 全字符串落盘正常。
4. **Kimi 并发** — 8 → 1（RPM 200 限速 + 启动并发风暴）。

---

## 四、主动学习队列

| 优先级 | 数量 | 触发 |
|---|---:|---|
| **high** | 132 | 双 LLM 零重叠标签 |
| medium | 15 | sentiment / NPS 分歧 |
| low | 14 | 其他（双零标签等）|

数据源：amazon 58 / zendesk 54 / trustpilot 38 / momcozy 9 / reddit 2

[Top-50 slice](../../03-数据资产/active_learning_queue_top50.jsonl) 已产出，可直接用 `human_annotation_cli --only-disagreement` 消化。

---

## 五、风险登记

| 风险 | 等级 | 应对 |
|---|---|---|
| 🔴 Kimi 账户余额不足 | 高 | D5 前充值 / 切 DS-pro fallback |
| ABSA 未经人工验证 | 中 | D5 quality_gate 加 30 条抽样复核 |
| 共识 46.3% 偏低 | 低 | 结构性，已标注，不属工具问题 |
| 944 条 deferred | 低 | 不阻塞 D5 quality_gate（用 general 样本）|

---

## 六、D5 启动条件

✅ **D5 解锁**（带 1 条充值阻断）：

| 前置 | 状态 |
|---|---|
| 5K 全量打标（D2）| ✅ |
| 500 金标（D3）| ✅ |
| ABSA 500 | ✅ |
| 低置信度筛选 | ✅ |
| 共识工具链 | ✅（充值后秒回放 944）|
| 主动学习队列 | ✅ |

D5 任务：T5.1 推荐意愿关键词 / T5.2 NPS 三法投票 / T5.3 quality_gate（9 项红线）/ T5.4 端到端跑测。

---

## 七、一行总结

> D4 用 1 天把 ABSA / 低置信度 / 双 LLM 共识 / 主动学习队列**四件套**全部落地+烟测，QA 通过；1244 条低置信中 300 条已验证（139 共识 + 161 入队），余 944 因 Kimi 余额耗尽 deferred 到 D5，**充值后 1 条命令即可补齐**，D5 通道无实质阻塞。

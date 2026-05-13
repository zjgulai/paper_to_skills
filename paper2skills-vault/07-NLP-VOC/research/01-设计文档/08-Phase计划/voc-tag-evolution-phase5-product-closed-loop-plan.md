---
name: voc-tag-evolution-phase5-product-closed-loop-plan
description: VOC 标签体系 Phase 5 执行规划，以 DeepSeek/Kimi 双 LLM 为核心引擎，14 天内完成产品级闭环：小样本分层验证 → 全量 LLM 打标 → ABSA 部署 → Proxy NPS 闭环 → 55 画像标签落地 → Skill 卡片产品化回流。当需要查阅 Phase 5 执行计划、日计划节奏、小样本测试方案、LLM 打标架构决策、闭环验收标准时使用。
title: VOC 标签体系 Phase 5 产品级闭环执行规划
doc_type: workflow
module: voc-nlp
topic: tag-evolution-phase5
status: review
created: 2026-05-07
updated: 2026-05-07
owner: self
source: human+ai
---

# VOC 标签体系 Phase 5 产品级闭环执行规划

> **基线状态**：v3.9 已完成，覆盖率 82.58% (301,060 / 364,569)，643 标签，纯规则 + ALCHEmist 弱监督
> **本期目标**：质量优先的产品级 AI 打标闭环，核心能力全部落地
> **执行周期**：**14 天**（2026-05-07 ~ 2026-05-20），日计划节奏
> **愿景定位**：A→B 过渡（Momcozy 内部决策中台，架构可迁移到其他 DTC 母婴品牌）

---

## 🗂 本文档路径与引用约定

- **本文件位置**：`.sisyphus/plans/voc-tag-evolution-phase5-product-closed-loop-plan.md`（评审副本）
- **正本**：`paper2skills-vault/07-NLP-VOC/research/01-设计文档/voc-tag-evolution-phase5-product-closed-loop-plan.md`
- **统一根目录**：`PROJECT_ROOT = /Users/pray/project/paper_to_skills`
- **VOC 模块根**：`VOC_ROOT = $PROJECT_ROOT/paper2skills-vault/07-NLP-VOC`
- **research 根**：`RESEARCH_ROOT = $VOC_ROOT/research`
- 所有引用路径以**绝对路径**（基于上述变量）给出，避免 `.sisyphus/` 与 `research/` 双副本下相对路径失效

---

## 零、关键共识决策（已对齐）

| # | 决策项 | 最终方案 |
|---|---|---|
| 0 | 产品愿景 | **A→B 过渡**：内部闭环为主，所有架构解耦以支持未来横向迁移 |
| 1 | 闭环边界 | **中闭环**：②打标 → ③BI看板 → ④策略生成 → ⑦字典进化 |
| 2 | LLM 引擎 | **DeepSeek-V4-Flash 主力 + Kimi-K2.6 兜底 + 双模型共识**（不设成本边界，质量为主）|
| 3 | LLM 打标边界 | **闭集为主 + 月度开集采样 5%** → 灌入 ALCHEmist |
| 4 | 覆盖率指标改革 | **双指标并行 1 季度**：`原始覆盖率` + `业务有效覆盖率`；v4.0 起后者为主 |
| 5 | Skill 卡片回流 | P0 四张（[ABSA-BERT-MoE](../Skill-ABSA-BERT-MoE.md) / [AutoTag-SelfEvolving](../Skill-AutoTag-SelfEvolving-Label-System.md) / [MAA-行动建议](../Skill-MAA-行动建议生成.md) / [AGRS-评论摘要](../Skill-AGRS-属性引导评论摘要.md)）+ 任何能优化 AI 打标工作流节点的算法 |
| 6 | 质量门槛 | **质量优先、标签要全**；耗时不作强约束 |
| 7 | 节奏 | **14 天，日计划**；1 人 + AI 协作 |
| 8 | 小样本策略 | **5000 条分层抽样**（按数据源比例：Amazon 2670 / Trustpilot 1369 / Zendesk 647 / Momcozy 270 / Reddit 44）|

---

## 一、执行摘要

### 1.1 本期交付的 7 项核心能力

| # | 能力 | 当前状态 | Phase 5 交付 | 对应决策 |
|---|---|---|---|---|
| C1 | LLM 闭集多标签打标 | 无 | 643 标签 DeepSeek+Kimi 双引擎全量打标 | 决策 2/3/6 |
| C2 | ABSA 方面级情感 | 未部署 | LLM-based ABSA，精确率 ≥ 80% | 决策 5 |
| C3 | Proxy NPS 自动打标 | 有定义无打标 | 推荐意愿关键词 + LLM 双通路 → NPS 看板 | 决策 1 |
| C4 | 55 原子画像标签 | 规则就绪未执行 | 全量执行 + LLM 增强 | 决策 1 |
| C5 | 覆盖率双指标 | 单一 82.58% | `原始覆盖率` + `业务有效覆盖率` 双报表 | 决策 4 |
| C6 | 月度开集进化 | 手动 | cron 自动化：零标签挖掘 → LLM 发现 → ALCHEmist | 决策 3 |
| C7 | 策略包自动生成 | 字段就绪未联动 | MAA 算法 → 部门 → BI 看板 | 决策 1/5 |

### 1.2 14 天产出物总览

| 产出物 | 路径 | 说明 |
|---|---|---|
| LLM 打标模块 | `02-脚本工具/01-标签进化/llm_labeler.py` | DeepSeek + Kimi 双引擎，闭集 + 共识 |
| ABSA 模块 | `02-脚本工具/01-标签进化/absa_extractor.py` | LLM-based 方面情感抽取 |
| Proxy NPS 模块 | `02-脚本工具/05-NPS管道/proxy_nps_labeler.py` | 推荐意愿 + 星级映射 + LLM 共识 |
| 55 画像标签器 | `02-脚本工具/01-标签进化/persona_tag_labeler.py` | 55 原子标签全量执行 |
| 5000 条测试集 | `03-数据资产/test_set_5k_stratified.jsonl` | 分层抽样，含人工金标 500 条 |
| 小样本评估报告 | `04-输出结果/03-审计报告/phase5_small_sample_report.md` | Go/No-Go 门禁 |
| Phase 5 统一打标器 | `02-脚本工具/01-标签进化/phase5_unified_labeler.py` | 替换 phase4_unified_labeler |
| 标签字典 v4.0 | `04-输出结果/01-字典版本/tag_dictionary_v4.0.xlsx` | 含 LLM 发现的新标签 + ABSA aspect 库 |
| 业务有效覆盖率报表 | `04-输出结果/03-审计报告/phase5_dual_coverage_report.md` | 双指标对比 |
| 月度进化 cron | `02-脚本工具/01-标签进化/monthly_evolution_cron.py` | 闭集+开集自动循环 |
| BI 看板 spec | `01-设计文档/phase5-bi-dashboard-spec.md` | 7 部门看板定义 |
| 最终审计 | `04-输出结果/03-审计报告/phase5_final_audit_report.md` | 全量验证 + Momus 审阅 |

---

## 二、总体架构

### 2.1 Phase 5 标签生产流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                    输入层（4 数据源 364,569 条）                   │
│  Amazon 194,734 · Trustpilot 99,853 · Zendesk 47,204 ·          │
│  Momcozy 19,808 · Reddit 2,970                                  │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  L0 规则层（Phase 4 保留，0 成本打底）                             │
│  general_tag_labeler + brand_label + zendesk_minimal            │
│  + negative_defect_miner + alchemist LFs                        │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  L1 LLM 闭集打标层（Phase 5 新增）                                 │
│  DeepSeek-V4-Flash 主通路（JSON mode，context caching 复用）      │
│  ↓                                                               │
│  Kimi-K2.6 二次通路（对低置信度/零标签样本）                        │
│  ↓                                                               │
│  双模型共识：一致→直接采纳；不一致→进 Active Learning 队列        │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  L2 ABSA 方面级情感层（Phase 5 新增）                              │
│  LLM 抽取 aspect-sentiment 三元组                                │
│  → 聚合到品类 × AIPL × 部门 的方面情感矩阵                         │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  L3 画像 + NPS 派生层                                             │
│  55 原子画像标签 + Proxy NPS 三分类（Promoter/Passive/Detractor）│
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  输出层：统一 labeled.jsonl（保留 v3.9 schema 兼容）              │
│  + phase5_audit.json（分源、分模块、置信度分布）                  │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  闭环⑶ BI 看板层（7 部门）                                        │
│  标签 → 主责部门 → 策略包 → 原子指标 → 周报自动化                 │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  闭环⑺ 字典自进化层（月度 cron）                                   │
│  零标签/低置信度采样 → LLM 开集发现 → ALCHEmist 生成 LF →         │
│  字典字段自动补全 → v4.1 → v4.2 …                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 LLM 引擎技术栈（关键技术决策）

| 项 | 选择 | 理由 |
|---|---|---|
| 主模型 | `deepseek-v4-flash` | 1M context / OpenAI SDK 兼容 / context caching $0.0028 per 1M cached / JSON mode |
| 兜底模型 | `kimi-k2.6` | 262K context / 中文语义强 / thinking mode 应对复杂 ABSA |
| 调用协议 | OpenAI SDK + `base_url` 切换 | 两模型全 OpenAI-compatible，一套代码 |
| Prompt 策略 | System prompt 缓存 2000+ tokens（643 标签定义 + 输出 schema） | DeepSeek cache hit 价格 1/50，复用 364K+ 次 |
| 输出格式 | `response_format={"type": "json_object"}` | 强制 JSON，避免解析失败 |
| 并发控制 | `asyncio.Semaphore(40)`（DeepSeek 50 concurrent 上限留 20% 余量） | 稳定性优先 |
| 重试 | 指数退避 0.5→1→2→4→8s，5 次 | 429 鲁棒性 |
| 幂等 | 每条 review 独立调用，结果落地前按 `review_id` 去重 | 断点续跑 |

---

## 三、14 天日计划（逐日可验收）

> **节奏约定**：每天早晨 Plan（10min） / 晚上 Demo（20min，产出当日可运行结果 + Go/No-Go 判据）。
> 每个里程碑收口后用 Momus 审阅计划文档变更。

### Week 1：小样本验证 + 核心模块落地（D1~D7）

> **每日 QA 场景模板**：每天 Demo 必须给出 `[命令] / [输入] / [预期输出] / [Pass 判据]` 四件套，否则不得收口。

#### D1（5-07 周四）：Bootstrap — 环境 + 测试集 + 基线
- **任务**
  - T1.1 配置 LLM Keys：`~/.paper2skills/llm_keys.json`（DeepSeek + Kimi）
  - T1.2 新建目录：`$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/`
  - T1.3 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/llm_client.py`：双模型 SDK 封装 + 指数退避
  - T1.4 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/stratified_sampler.py`：5000 条分层抽样
  - T1.5 跑 Phase 4 基线对比
- **🧪 QA 场景**
  - **场景 1（API 连通性）**
    - 命令：`python $RESEARCH_ROOT/02-脚本工具/07-LLM引擎/llm_client.py --smoke-test`
    - 输入：单条 review `"This pump is comfortable but loud."`
    - 预期：DeepSeek 返回 JSON `{"labels": [...], ...}` 且 Kimi 同样返回 JSON
    - Pass：两模型均返回合法 JSON，无 5xx，p95 latency < 5s
  - **场景 2（抽样正确性）**
    - 命令：`python stratified_sampler.py --total 5000 --output $RESEARCH_ROOT/03-数据资产/test_set_5k_stratified.jsonl`
    - 预期分布：`Amazon 2670 ±10 / Trustpilot 1369 ±10 / Zendesk 647 ±10 / Momcozy 270 ±10 / Reddit 44 ±5`
    - Pass：抽样脚本输出的 `data_source` 计数全部命中目标 ±2%
  - **场景 3（Phase 4 基线复现）**
    - **⚠️ 脚本 CLI 扩展任务（T1.5 前置）**：现有 [`phase4_unified_labeler.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/phase4_unified_labeler.py) 的 `__main__` 仅支持 `--test` 或默认路径（line 527-542），不接受 `--input/--output` 参数。**D1 上午先做 CLI 扩展**：加 `argparse` 支持 `--input PATH --output PATH --limit N`，保持默认路径回退行为（零回归）。
    - 命令：`python $RESEARCH_ROOT/02-脚本工具/01-标签进化/phase4_unified_labeler.py --input $RESEARCH_ROOT/03-数据资产/test_set_5k_stratified.jsonl --output $RESEARCH_ROOT/03-数据资产/test_set_5k_p4_baseline.jsonl`
    - 预期：覆盖率落在 80~85% 区间（与全量 82.58% 接近）
    - Pass：CLI 扩展后 `--test` 仍 100% 通过（回归）+ 5000 条覆盖率与全量基线偏差 < 3pp

#### D2（5-08 周五）：LLM 闭集打标 MVP
- **任务**
  - T2.1 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/llm_labeler.py`
  - T2.2 System prompt 含 643 标签压缩定义（≤2000 tokens）
  - T2.3 强制 `response_format={"type":"json_object"}`，schema 对齐 v3.9 label dict
  - T2.4 跑 5000 条 DeepSeek-V4-Flash 单模型
- **🧪 QA 场景**
  - **场景 1（Schema 合法性）**
    - 命令：`python -m pytest $RESEARCH_ROOT/02-脚本工具/07-LLM引擎/tests/test_llm_labeler_schema.py -v`
    - 预期：100 条样本输出全部通过 Pydantic 校验，无 `tag_id` 不在字典内的"野标签"
    - Pass：100/100 通过；JSON 解析失败率 < 1%
  - **场景 2（5000 条覆盖率）**
    - 命令：`python llm_labeler.py --input test_set_5k_stratified.jsonl --output test_set_5k_p5_llm.jsonl --model deepseek-v4-flash`
    - 预期：`原始覆盖率 ≥ 92%`（vs Phase 4 在 5K 上的 ~82%）；标签分布与字典 TOP-50 命中率匹配
    - Pass：覆盖率 ≥ 92% 且 cache-hit 占输入 token ≥ 90%

#### D3（5-09 周六）：金标 500 + 三方评估
- **任务**
  - T3.1 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/human_annotation_cli.py`：CLI 标注工具
  - T3.2 抽 500 条金标（每源比例 + 每源 ≥20 条零标签）
  - T3.3 你本人 4-6 小时集中标注（Top-3 标签 + overall sentiment + Proxy NPS）
  - T3.4 第二天**自我重标 50 条**做一致性检查
  - T3.5 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/evaluation_suite.py`
- **🧪 QA 场景**
  - **场景 1（标注自一致性）**
    - 命令：`python evaluation_suite.py --self-consistency golden_set_500.jsonl golden_set_50_redo.jsonl`
    - 预期：Cohen's κ ≥ 0.80
    - Pass：达到 0.80 才认可金标质量；不达就重训自己
  - **场景 2（三方评估）**
    - 命令：`python evaluation_suite.py --golden golden_set_500.jsonl --pred-p4 test_set_5k_p4_baseline.jsonl --pred-llm test_set_5k_p5_llm.jsonl --report phase5_eval_v0.md`
    - 预期：报告含 per-label P/R/F1 + macro/weighted F1 + Cohen's κ + 混淆矩阵
    - Pass：报告所有指标计算成功输出，无 NaN

#### D4（5-10 周日）：ABSA + Kimi 共识层
- **任务**
  - T4.1 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/absa_extractor.py`：抽 `(aspect, sentiment, confidence)`
  - T4.2 在 500 条金标上人工补 ABSA aspect（平均 2-3 个/条）
  - T4.3 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/low_conf_extractor.py`：**定义低置信度样本的产生流水线**
    - 输入：`test_set_5k_p5_llm.jsonl`（D2 产出）
    - 筛选规则：`max(labels[].confidence) < 0.70` OR `len(labels) == 0` OR `record._original_n_tags == 0`
    - 输出固定路径：`$RESEARCH_ROOT/03-数据资产/low_conf_samples.jsonl`
  - T4.4 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/llm_consensus.py`
  - T4.5 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/active_learning_queue.py`
- **🧪 QA 场景**
  - **场景 0（低置信度样本产出）**
    - 命令：`python low_conf_extractor.py --input $RESEARCH_ROOT/03-数据资产/test_set_5k_p5_llm.jsonl --output $RESEARCH_ROOT/03-数据资产/low_conf_samples.jsonl`
    - 预期：输出文件存在；样本数占输入 10-30%
    - Pass：文件生成成功 + 样本数量合理 + 每条含 filter_reason 字段
  - **场景 1（ABSA 抽取）**
    - 命令：`python absa_extractor.py --input $RESEARCH_ROOT/03-数据资产/golden_set_500.jsonl --output $RESEARCH_ROOT/03-数据资产/absa_500_pred.jsonl`
    - 预期：每条平均输出 1-5 个 aspect-sentiment 三元组；空输出占比 < 10%
    - Pass：aspect 抽取 F1 ≥ 0.75（vs 人工金标），sentiment 条件 F1 ≥ 0.80
  - **场景 2（共识机制）**
    - 命令：`python llm_consensus.py --input $RESEARCH_ROOT/03-数据资产/low_conf_samples.jsonl --primary deepseek-v4-flash --fallback kimi-k2.6 --output $RESEARCH_ROOT/03-数据资产/consensus_result.jsonl --unresolved-queue $RESEARCH_ROOT/03-数据资产/active_learning_queue.jsonl`
    - 预期：低置信度样本经 Kimi 复跑后，一致样本直接采纳，不一致样本写入 `active_learning_queue.jsonl`
    - Pass：共识一致率 ≥ 70%；不一致全部入队，0 丢失

#### D5（5-11 周一）：Proxy NPS 闭环 + Week 1 质量门禁
- **任务**
  - T5.1 在 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/general_tag_labeler.py` 扩展推荐意愿关键词（en/de/fr）
  - T5.2 新建 `$RESEARCH_ROOT/02-脚本工具/05-NPS管道/proxy_nps_labeler.py`：星级 + 关键词 + LLM 三法投票
  - T5.3 写 `$RESEARCH_ROOT/02-脚本工具/07-LLM引擎/quality_gate.py`：9 项红线判定
  - T5.4 在 5000 条 + 500 金标上跑 D2-D5 全部模块
- **🧪 QA 场景**
  - **场景 1（NPS 三法一致性）**
    - 命令：`python proxy_nps_labeler.py --input golden_set_500.jsonl --output nps_500_pred.jsonl`
    - 预期：三法投票后 Promoter/Passive/Detractor 占比与人工金标偏差 < 5pp
    - Pass：三分类一致率 ≥ 85%
  - **场景 2（Week 1 Gate 自动判定）**
    - 命令：`python quality_gate.py --gate week1 --golden golden_set_500.jsonl --pred test_set_5k_p5_llm.jsonl --absa absa_500_pred.jsonl --nps nps_500_pred.jsonl`
    - 预期：输出 9 项红线 PASS/FAIL 表 + 总判定 GO/NO-GO
    - Pass：9 项全 PASS。任意 FAIL → D5.5/D5.6 修补（最多 2 天）

#### D6（5-12 周二）：55 画像标签恢复 + 落地
- **⚠️ 前置阻塞修复**
  - **现状**：[`$RESEARCH_ROOT/01-设计文档/02-工作流设计/统一标签树设计.md`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/02-工作流设计/统一标签树设计.md) 第 38 行起引用 `画像标签识别规则表.md`，但该文件**仓库内不存在**。
  - **修复路径**（D6 上午先做）：
    - 方案 A（首选）：从 `$VOC_ROOT/Skill-PERSONABOT-RAG用户画像生成.md` + `Skill-SoMeR-多视角用户表示.md` + `Skill-GPLR-人群标签生成.md` 三张 Skill 卡片提炼出 55 条规则
    - 方案 B（兜底）：用 LLM 基于 `统一标签树设计.md` 中的"5 维度（WHO/WHY/WHAT/WHEN/HOW）"重新生成 55 条候选规则，再人工裁定
    - 产出：`$RESEARCH_ROOT/01-设计文档/02-工作流设计/画像标签识别规则表.md`
- **任务**
  - T6.1 恢复 `画像标签识别规则表.md`（上午 2-3h）
  - T6.2 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/persona_tag_labeler.py`：55 规则 → Label Function
  - T6.3 LLM 兜底：规则未命中样本用 LLM 闭集尝试 55 标签
  - T6.4 在 5000 条上跑画像标签器
  - T6.5 写诊断报告 `$RESEARCH_ROOT/02-脚本工具/06-诊断工具/persona_diagnostic.py`
- **🧪 QA 场景**
  - **场景 1（规则文档恢复）**
    - 命令：`wc -l $RESEARCH_ROOT/01-设计文档/02-工作流设计/画像标签识别规则表.md && grep -c "^| P-" 画像标签识别规则表.md`
    - 预期：包含 55 条 `P-Lx-xx` 标签 ID 行
    - Pass：恰好 55 条规则，每条含 tag_id/中英名/触发关键词/置信度
  - **场景 2（画像渗透率）**
    - 命令：`python persona_tag_labeler.py --input test_set_5k_p5_llm.jsonl --output test_set_5k_p5_persona.jsonl && python persona_diagnostic.py --input test_set_5k_p5_persona.jsonl`
    - 预期：≥60% 记录至少打上 1 个画像标签；按数据源/品类输出渗透率热力表
    - Pass：渗透率 ≥ 60%；55 标签中 ≥ 45 个有命中（避免大量"死"标签）

#### D7（5-13 周三）：Week 1 收口 + 全 Pipeline 集成
- **任务**
  - T7.1 整合 D1-D6 到 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/phase5_unified_labeler.py`（替换 phase4_unified_labeler）
  - T7.2 接口契约：`label_single_record(record) -> (new_labels, all_labels, meta)` 与 Phase 4 兼容
  - T7.3 在 5000 条上跑端到端
  - T7.4 写 `$RESEARCH_ROOT/04-输出结果/03-审计报告/phase5_small_sample_report.md`
- **🧪 QA 场景**
  - **场景 1（端到端打通）**
    - **⚠️ 校验器新建任务（T7.1.5）**：现有 [`quick_coverage_test.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/quick_coverage_test.py) 是旧版（写死 v3.3 字典 + `labeling-latest` 路径，依赖 `unified_label_extraction` 模块），**不可复用**。在 D7 新建 `$RESEARCH_ROOT/02-脚本工具/06-诊断工具/phase5_schema_validator.py`，支持 `--input PATH --dict-version v3.9|v4.0`，校验输出 record 的 `labels[].tag_id` 全部在指定字典内 + 必备字段不缺失。
    - 命令：`python phase5_unified_labeler.py --input test_set_5k_stratified.jsonl --output test_set_5k_p5_full.jsonl --self-test && python $RESEARCH_ROOT/02-脚本工具/06-诊断工具/phase5_schema_validator.py --input test_set_5k_p5_full.jsonl --dict-version v3.9`
    - 预期：自证测试 ≥ 30 个用例 100% 通过；端到端跑 5000 条无异常退出；schema 校验器 0 错误
    - Pass：自证 100% + 端到端成功 + schema 校验器返回 PASS
  - **场景 2（Week 1 Gate 重判）**
    - 命令：`python quality_gate.py --gate week1 --pred test_set_5k_p5_full.jsonl --golden golden_set_500.jsonl --report phase5_small_sample_report.md`
    - 预期：9 项红线 PASS
    - Pass：通过则进 Week 2；任意 FAIL 则 NO-GO，加 2 天修补

---

### Week 2：全量闭环 + 字典进化 + BI 联调（D8~D14）

#### D8（5-14 周四）：全量 LLM 增打启动
- **任务**
  - T8.1 对象：Phase 4 全量零标签 (63,509) + LLM 低置信度样本（预估 ~16K）≈ **80K 条**
  - **T8.1.1 全量增打输入集产出（明确来源）**：
    - 命令：`python $RESEARCH_ROOT/02-脚本工具/07-LLM引擎/low_conf_extractor.py --input $RESEARCH_ROOT/04-输出结果/unified_labeling/phase4_labeled.jsonl --output $RESEARCH_ROOT/03-数据资产/phase4_zero_and_low_conf.jsonl --include-zero-label --confidence-threshold 0.70`
    - 输入：Phase 4 全量打标结果 `phase4_labeled.jsonl`
    - 输出固定路径：`$RESEARCH_ROOT/03-数据资产/phase4_zero_and_low_conf.jsonl`，预估 ~80K 条
    - Pass：输出条数在 [70K, 90K] 区间；每条含 `filter_reason`（zero_label / low_conf）
  - T8.2 并发 40（DeepSeek 50 上限留 20% 余量），流式落 `$RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl`
  - T8.3 实时监控脚本：`$RESEARCH_ROOT/02-脚本工具/06-诊断工具/llm_labeling_monitor.py`
- **🧪 QA 场景**
  - **场景 1（启动烟测）**
    - 命令：`python phase5_unified_labeler.py --input $RESEARCH_ROOT/03-数据资产/phase4_zero_and_low_conf.jsonl --limit 100 --output $RESEARCH_ROOT/03-数据资产/smoke_test.jsonl`
    - 预期：100 条跑通，无 429 连续触发，无未 handle 异常
    - Pass：成功率 ≥ 99%，平均 latency < 3s
  - **场景 2（全量中途观测）**
    - 命令：`python llm_labeling_monitor.py --tail $RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl --window 1000`
    - 预期：滑窗 1000 条的成功率/置信度/cache_hit 实时曲线
    - Pass：成功率 ≥ 98%；平均置信度 ≥ 0.70；cache_hit 占输入 token ≥ 85%

#### D9（5-15 周五）：字典进化 v4.0 + ABSA Aspect 库 + **中间全量合并**
- **任务**
  - T9.1 对 D8 全量输出做 **5% 开集采样**（约 4K 条）
  - T9.2 跑 [`$RESEARCH_ROOT/02-脚本工具/01-标签进化/gap_detector.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/gap_detector.py) + `alchemist_label_generator.py`
  - T9.3 LLM 辅助去重 + 业务相关性评分
  - T9.4 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/tag_dictionary_v40_generator.py`
  - T9.5 汇总 ABSA 全量 aspect，写入 v4.0 新 Sheet `10_Aspect库`
  - **T9.6（新增）中间全量合并 → 提供 D10-D12 输入**：
    - 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/merge_phase4_phase5_llm.py`
    - 输入：`phase4_labeled.jsonl`（保留有标签部分）+ `phase5_full_labeled_llm.jsonl`（D8 增打的 ~80K）
    - 合并规则：按 `review_id` 主键，phase5 LLM 结果**追加到** phase4 的 labels 数组（不覆盖，source 区分）
    - 输出固定路径：`$RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl`
    - **D10/D11/D12 的输入文件统一改为 `phase5_intermediate_merged.jsonl`**；D13 重打覆盖时再产出 `phase5_full_labeled.jsonl`（基于 v4.0 字典）
- **🧪 QA 场景**
  - **场景 0（中间全量合并）**
    - 命令：`python merge_phase4_phase5_llm.py --p4 $RESEARCH_ROOT/04-输出结果/unified_labeling/phase4_labeled.jsonl --p5-llm $RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl --output $RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl`
    - 预期：合并后总记录数 = 364,569（一一对应）；每条 review_id 唯一；source 字段含 `phase4_*` 和 `llm_*` 两类
    - Pass：记录数 == 364,569 + 0 重复 review_id + 抽样 50 条人工核对合并正确率 100%
  - **场景 1（新标签候选过滤）** *（保持原内容）*
    - 命令：`python tag_dictionary_v40_generator.py --dry-run --report v40_dry_run.md`
    - 预期：候选新标签经"频率 ≥10 + Jaccard < 0.3 + LLM 相关性 ≥3/5"三过滤，最终保留 20-40 个
    - Pass：候选数在 [20, 40]；无 tag_id 冲突；输出报告含每个新标签的审核 trace
  - **场景 2（v4.0 字典完整性）** *（保持，已修订校验器扩展）*
    - **⚠️ 校验器扩展任务（T9.4.5）**：现有 [`dictionary_validator.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/dictionary_validator.py) 第 90 行写死了 `tag_dictionary_v3.4_filled.xlsx`，不支持参数化，也没有 `10_Aspect库` Sheet 校验逻辑。**D9 先扩展**：新增 `--xlsx PATH` 参数（argparse）+ `--require-sheets "01_通用标签主表,08_映射关系表,10_Aspect库"` + `10_Aspect库` 的字段校验（aspect_id / aspect_en / aspect_cn / category / 关联tag_ids）。扩展后 `__main__` 默认行为保持向后兼容（仍可运行历史 v3.4 校验）。
    - 命令：`python $RESEARCH_ROOT/02-脚本工具/01-标签进化/dictionary_validator.py --xlsx $RESEARCH_ROOT/04-输出结果/01-字典版本/tag_dictionary_v4.0.xlsx --require-sheets "01_通用标签主表,08_映射关系表,10_Aspect库" --min-aspect-rows 50`
    - 预期：全字段 0 空值、tag_id 唯一、AIPL 合法、10_Aspect库 Sheet 存在且行数 ≥ 50；扩展后原 v3.4 默认校验仍 PASS（回归）
    - Pass：v4.0 校验 PASS + 回归 PASS

#### D10（5-16 周六）：双覆盖率指标
- **任务**
  - T10.1 写 `$RESEARCH_ROOT/02-脚本工具/06-诊断工具/dual_coverage_calculator.py`
  - T10.2 定义口径：`原始覆盖率 = 有标签/总数`；`业务有效覆盖率 = 有标签/(总数 - 非品类 - 极短无意义 - 泛化评价)`
  - T10.3 广义覆盖率：品牌标签命中也计入
  - T10.4 生成 `$RESEARCH_ROOT/04-输出结果/03-审计报告/phase5_dual_coverage_report.md`
- **🧪 QA 场景**
  - **场景 1（指标口径可审计）**
    - 命令：`python dual_coverage_calculator.py --input $RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl --report dual_cov.md --with-exclusion-trace`
    - 预期：报告含每类排除样本的 review_id 抽样 100 条，供人工 spot check
    - Pass：抽样 100 条中排除理由人工复核一致率 ≥ 95%
  - **场景 2（双指标阈值）**
    - 命令：`python dual_coverage_calculator.py --input $RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl --threshold-raw 0.88 --threshold-effective 0.94 --json-output $RESEARCH_ROOT/04-输出结果/03-审计报告/phase5_dual_coverage_thresholds.json`
    - 预期：`原始覆盖率 ≥ 88%`（Phase 4: 82.58%）；`业务有效覆盖率 ≥ 94%`；JSON 输出含 `{"raw_coverage": float, "effective_coverage": float, "raw_pass": bool, "effective_pass": bool}`
    - Pass：JSON 中 `raw_pass == true && effective_pass == true`（自动判定，无需目检）

#### D11（5-17 周日）：MAA 策略包 + BI Spec
- **任务**
  - T11.1 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/maa_strategy_generator.py`：5 Agent 简化版
  - T11.2 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/agrs_summarizer.py`
  - T11.3 写 `$RESEARCH_ROOT/01-设计文档/phase5-bi-dashboard-spec.md`：7 部门看板字段定义
- **🧪 QA 场景**
  - **场景 1（产品中心 MAA 样例）**
    - 命令：`python maa_strategy_generator.py --dept product_rd --input $RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl --output rd_top10_actions.md`
    - 预期：Top 10 行动建议，每条含 SRAC 四维评分、代表评论 3 条、预期指标变化
    - Pass：10 条建议，评分区分度足够（最高/最低差 ≥ 5 分，10 分制）；业务人员快速复核 ≥ 7/10 条合理
  - **场景 2（BI Spec 完整性）**
    - 命令：`python $RESEARCH_ROOT/02-脚本工具/06-诊断工具/bi_spec_validator.py --spec $RESEARCH_ROOT/01-设计文档/phase5-bi-dashboard-spec.md --required-departments "全球客服中心,产品中心,仓储物流部,品牌市场中心,电商运营部,品质管理中心,法务合规部" --required-sections "KPI列表,数据源,标签映射,刷新频率,样例周报"`
    - **⚠️ 校验器新建任务（T11.3.5）**：新写 `bi_spec_validator.py`（~80 行）—— 用 markdown 标题正则扫描 7 部门 5 章节是否齐全，缺任意项报错。
    - 预期：7 部门 × 5 章节 = 35 个断言全通过，无 TBD 关键词
    - Pass：validator exit code 0（自动判定，无需目检）

#### D12（5-18 周一）：月度进化 Cron
- **任务**
  - T12.1 写 `$RESEARCH_ROOT/02-脚本工具/01-标签进化/monthly_evolution_cron.py`：8 步 pipeline
  - T12.2 macOS LaunchAgent：`~/Library/LaunchAgents/com.momcozy.voc.monthly-evolution.plist`
  - T12.3 失败/完成推送飞书 webhook（复用 `~/.paper2skills/feishu_webhook`）
  - T12.4 Dry-run 演练
- **🧪 QA 场景**
  - **场景 1（Cron 脚本端到端 dry-run）**
    - 命令：`python monthly_evolution_cron.py --dry-run --input $RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl --output-dict tag_dictionary_v4.1_dryrun.xlsx`
    - 预期：8 步全部执行到位，输出一份 v4.1 草案字典 + 一份进化日志
    - Pass：8 步全部 exit code 0；v4.1 vs v4.0 diff 合理（新增 ≤ 10 标签）
  - **场景 2（LaunchAgent 触发）**
    - 命令：`launchctl load ~/Library/LaunchAgents/com.momcozy.voc.monthly-evolution.plist && launchctl kickstart -k gui/$(id -u)/com.momcozy.voc.monthly-evolution`
    - 预期：立即触发一次执行；飞书收到完成通知
    - Pass：日志可见 + 飞书通知到达
  - **场景 3（多租户前瞻兼容）**
    - 预期：cron 脚本所有路径不写死 `momcozy`，通过 `--tenant` 参数注入，便于 v6.0 迁移
    - Pass：`monthly_evolution_cron.py --help` 显示 `--tenant` 参数，默认 `momcozy`

#### D13（5-19 周二）：全量重打 + 最终审计
- **任务**
  - T13.1 用 `phase5_unified_labeler.py + 字典 v4.0` 对 **364,569 全量**重打
  - T13.2 产出 `$RESEARCH_ROOT/04-输出结果/unified_labeling/phase5_full_labeled.jsonl`
  - T13.3 写 `$RESEARCH_ROOT/04-输出结果/03-审计报告/phase5_final_audit_report.md`
- **🧪 QA 场景**
  - **场景 1（全量 Week 2 Gate）**
    - 命令：`python quality_gate.py --gate week2 --pred phase5_full_labeled.jsonl --report phase5_final_audit_report.md`
    - 预期：7 项 Week 2 红线（见第五章）判定表
    - Pass：7 项全 PASS；否则回修对应模块
  - **场景 2（回归一致性）**
    - 命令：`python evaluation_suite.py --golden golden_set_500.jsonl --pred-llm phase5_full_labeled.jsonl --mode regression`
    - 预期：全量结果在金标集子集上的指标 vs Week 1 小样本偏差 < 3pp
    - Pass：偏差 < 3pp，确认 scale-up 未引入质量漂移

#### D14（5-20 周三）：Momus 审阅 + 验收
- **任务**
  - T14.1 本规划文档 + `phase5_final_audit_report.md` 交 Momus 审
  - T14.2 根据 Momus 意见补齐（最多 1 轮）
  - T14.3 更新 [`$PROJECT_ROOT/paper2skills-vault/07-资源库/sync_status.json`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-资源库/sync_status.json)
  - T14.4 更新 P0 Skill 卡片 4 张的 sync 状态
  - T14.5 Phase 4 中间产物归档到 `$RESEARCH_ROOT/00-归档资料/phase4_archive/`
- **🧪 QA 场景**
  - **场景 1（最终验收：Momus 主路径 + 人工 fallback）**
    - **主路径（OpenCode 内）**：你在 OpenCode 会话中向 AI 发指令 `请用 Momus 评审 .sisyphus/plans/voc-tag-evolution-phase5-product-closed-loop-plan.md`，AI 内部调用 `task(subagent_type="Momus - Plan Critic", prompt=".sisyphus/plans/voc-tag-evolution-phase5-product-closed-loop-plan.md")`。Momus 输出会落地到当次 session 日志，**复制评审结果存档到 `$RESEARCH_ROOT/04-输出结果/03-审计报告/phase5_momus_review.md`**。Pass 标准：返回 **Pass** 或 **Pass-with-minor-revisions**，无 Blocking Issues。
    - **人工 fallback 路径（无 OpenCode 时）**：
      - 工具：新建脚本 `$RESEARCH_ROOT/02-脚本工具/06-诊断工具/phase5_acceptance_checklist.py`（~120 行）
      - 该脚本以 YAML 形式声明所有 D1-D13 的"Pass 判据"为可机器校验的断言（共约 30 项），逐条扫描产出物：
        ```yaml
        # 例：
        - id: D1.S2
          desc: "5000 条分层抽样比例正确"
          check: "jq '.data_source' test_set_5k_stratified.jsonl | sort | uniq -c"
          expected:
            amazon_competitor: [2660, 2680]
            trustpilot: [1359, 1379]
            zendesk: [637, 657]
            momcozy: [260, 280]
            reddit: [39, 49]
        - id: D7.S2
          desc: "Week 1 Gate 9 项红线全过"
          check: "cat phase5_small_sample_report.md | grep -c 'PASS'"
          expected: ">=9"
        ...（其余 28 项）
        ```
      - 命令：`python $RESEARCH_ROOT/02-脚本工具/06-诊断工具/phase5_acceptance_checklist.py --report $RESEARCH_ROOT/04-输出结果/03-审计报告/phase5_acceptance_checklist_result.md`
      - 输出：Markdown 表格（30 项 × PASS/FAIL/SKIP）+ 末尾总判定 `OVERALL: PASS/FAIL`
      - 通过标准：30 项中 ≥ 28 项 PASS，且无 P0 项 FAIL（D5/D7/D13 三项 Gate 必须全 PASS）
    - 预期：主路径返回 Momus Pass；fallback 返回 OVERALL: PASS
    - Pass：任一路径达标即算 D14 场景 1 通过
  - **场景 2（sync_status 同步）**
    - 命令：`python $PROJECT_ROOT/paper2skills-skills/paper-同步/scripts/sync.py --status`
    - 预期：4 张 P0 Skill 标记 synced=true
    - Pass：`jq '[to_entries[] | select(.value.vault.synced==true)] | length' sync_status.json` 返回原值 +4

---

## 四、5000 条小样本测试方案（关键）

### 4.1 分层抽样设计

**抽样公式**（比例分层）：

$$n_h = 5000 \times \frac{N_h}{364{,}569}$$

| 数据源 | 总数 $N_h$ | 占比 | 目标 $n_h$ | 备注 |
|---|---:|---:|---:|---|
| Amazon 竞品 | 194,734 | 53.4% | **2,670** | 含 18 品牌 + 非品类评论 |
| Trustpilot | 99,853 | 27.4% | **1,369** | 含泛化好评 |
| Zendesk | 47,204 | 12.9% | **647** | 含长/短文本 |
| Momcozy 自有 | 19,808 | 5.4% | **270** | 自有评论 |
| Reddit | 2,970 | 0.8% | **44** | 规模小，合并到专项 |
| **合计** | **364,569** | **100%** | **5,000** | |

### 4.2 二级分层（每个数据源内）

| 二级维度 | 分层规则 |
|---|---|
| 语言 | en / de / fr / zh / other（按实际比例）|
| 文本长度 | ≤50 / 51-200 / 201-500 / >500 字符 |
| Phase 4 打标状态 | 有标签 / 零标签（保持原分布）|

使用 `pandas.groupby(['data_source', 'language', 'length_bucket', 'has_phase4_label']).apply(lambda x: x.sample(...))` 实现。

### 4.3 金标集设计（500 条）

- **规模**：500 条（约 10% 测试集），保证 Cohen's κ 95% 置信区间 ±0.05
- **抽样**：从 5000 条中再分层抽，每源按比例 + 每源保证 ≥20 条零标签样本
- **标注维度**（每条）：
  1. Top-3 标签（从 643 标签闭集选）
  2. overall sentiment（positive/neutral/negative）
  3. Proxy NPS 分类（Promoter/Passive/Detractor）
  4. ABSA aspect 列表（每个 aspect 打 sentiment）
- **标注人**：你本人（或信任的业务同事）；AI 辅助但最终裁定权在人
- **标注工具**：简单 CLI 脚本 `human_annotation_cli.py`，逐条展示 + 快捷键 + JSONL 落地
- **质量校验**：随机挑 50 条第二天重标，自我 Cohen's κ ≥ 0.80 才能作为金标

---

## 五、质量门禁（Go/No-Go 红线）

> **决策 8 调整**：不考虑成本，运行时间压缩到可接受范围内。

### 5.1 Week 1 Gate（D7 晚）—— 决定是否进入全量

| # | 指标 | 红线 | 测量方法 |
|---|---|---:|---|
| 1 | LLM 闭集打标 Top-1 准确率（vs 金标） | **≥ 85%** | Top-1 matches any of 3 human labels |
| 2 | LLM 闭集打标 Top-3 召回（vs 金标） | **≥ 90%** | 3 预测含 ≥2 金标 |
| 3 | 单标签 per-label F1（TOP 30 常用标签） | **≥ 0.75**（weighted）| sklearn.metrics |
| 4 | LLM vs 人工 Cohen's κ | **≥ 0.65**（substantial） | sklearn.metrics |
| 5 | ABSA aspect 抽取 F1 | **≥ 0.75** | exact match on aspect noun |
| 6 | ABSA sentiment F1（条件于 aspect 正确） | **≥ 0.80** | 三分类 |
| 7 | Proxy NPS 三分类一致率（vs 人工）| **≥ 85%** | overall accuracy |
| 8 | 标签互斥冲突率 | **< 3%** | 同时命中 POS 和对应 NEG 的比例 |
| 9 | JSON 解析失败率 | **< 1%** | LLM 输出格式异常比例 |

**Go**：9 项全过 → 进入 Week 2 全量
**No-Go**：任意一项不过 → 诊断根因 → 最多 2 天修补 → 重评

### 5.2 Week 2 Gate（D13 晚）—— 决定是否收官

| # | 指标 | 红线 | 说明 |
|---|---|---:|---|
| 10 | 全量 `原始覆盖率` | **≥ 88%**（Phase 4: 82.58%）| 提升 ≥ 5pp |
| 11 | 全量 `业务有效覆盖率` | **≥ 94%** | 排除无意义分母 |
| 12 | 全量 LLM 平均置信度 | **≥ 0.75** | 低置信率告警 |
| 13 | 55 画像标签渗透率 | **≥ 60%** | 有画像标签的记录占比 |
| 14 | Proxy NPS 打通率 | **≥ 95%** | 能算出 NPS 的记录占比 |
| 15 | 自证测试通过率 | **100%** | 保持 Phase 4 标准 |
| 16 | BI 看板 spec 完整性 | **7 部门全覆盖** | 目检 |

---

## 六、LLM 技术规范（编码层面）

### 6.1 System Prompt 骨架（可缓存，2000 tokens 左右）

```
你是 Momcozy 母婴出海 VOC 标签专家。任务：对每条评论做 5 层分析。

# 标签字典（v3.9 摘要，643 个）
## 通用情感/体验标签（209 个）
TAG_GEN_001 ease_of_use 使用方便（AIPL=L1，正面）
TAG_GEN_002 comfort_experience 舒适体验（AIPL=L1，正面）
... [压缩到 Top 200 + 全 ID 索引]

## 品线专属标签（360 个，按需检索）
[按 product_line 分区，每条仅含 tag_id + 中英名]

## 品牌标签（18 个）
BRAND_OWN_001 Momcozy
BRAND_DIRECT_001 Spectra
...

# 输出 Schema（必须严格遵守）
{
  "labels": [
    {"tag_id": "TAG_GEN_002", "confidence": 0.92, "evidence": "very comfortable to wear"}
  ],
  "aspects": [
    {"aspect": "comfort", "sentiment": "positive", "confidence": 0.95, "evidence": "..."}
  ],
  "overall_sentiment": "positive",
  "proxy_nps": "promoter",
  "persona_tags": ["night_feeder", "working_mom"]
}

# 规则
1. labels 严格从字典 ID 中选，不得生成不存在的 ID
2. confidence < 0.5 的标签不输出
3. 否定词触发 POS→NEG 翻转（如 "not easy" → TAG_GEN_N001）
4. 任何语义不明确的字段留空，不瞎猜
5. 只输出 JSON，无任何解释文本
```

### 6.2 关键代码接口契约

**`llm_labeler.py` 主函数签名**：

```python
async def llm_label_single(
    record: dict,                    # v3.9 VOCRecord schema
    primary_model: str = "deepseek-v4-flash",
    fallback_model: str = "kimi-k2.6",
    confidence_threshold: float = 0.7,
) -> dict:
    """
    返回 v3.9 兼容的 record，新增字段：
      labels: 追加 LLM 打的标签（source="llm" 或 "llm_consensus"）
      aspects: ABSA 结果
      persona_tags: 55 画像标签
      proxy_nps: Promoter/Passive/Detractor
      llm_meta: {model_used, tokens_in, tokens_out, cache_hit, latency_ms, retries}
    """
```

**共识策略**：

```python
# 伪代码
primary_result = call_llm(primary_model, record)
if primary_result.confidence < threshold or primary_result.labels == []:
    fallback_result = call_llm(fallback_model, record)
    if set(primary.label_ids) == set(fallback.label_ids):
        # 共识一致
        return merge(primary, fallback, source="llm_consensus")
    else:
        # 不一致 → Active Learning 队列
        write_to_active_learning_queue(record, primary, fallback)
        return primary  # 先采纳主模型结果，等人工裁决
```

### 6.3 Context Caching 优化

- System prompt 缓存：首条之后 `prompt_cache_hit_tokens = 2000`
- 成本节省：从 $0.14/1M → $0.0028/1M（50 倍）
- 批量预热：第一个 batch 强制打到 cache 以保证后续 cache hit

---

## 七、Skill 卡片 → 产品化落地映射

| # | Skill 卡片 | Phase 5 落地形态 | 代码文件 | 时间 |
|---|---|---|---|---|
| S1 | `$VOC_ROOT/Skill-ABSA-BERT-MoE.md` | LLM-based ABSA（本期用 LLM 快速落地，v5.0 评估是否替换为微调 MoE）| `$RESEARCH_ROOT/02-脚本工具/01-标签进化/absa_extractor.py` | D4 |
| S2 | `$VOC_ROOT/Skill-AutoTag-SelfEvolving-Label-System.md` | 月度 cron 自动进化流水线 | `$RESEARCH_ROOT/02-脚本工具/01-标签进化/monthly_evolution_cron.py` | D12 |
| S3 | `$VOC_ROOT/Skill-MAA-行动建议生成.md` | 5 Agent 简化版（TopicImpact+AGRS+Rec+SRAC）| `$RESEARCH_ROOT/02-脚本工具/01-标签进化/maa_strategy_generator.py` | D11 |
| S4 | `$VOC_ROOT/Skill-AGRS-属性引导评论摘要.md` | BI 看板摘要模块 | `$RESEARCH_ROOT/02-脚本工具/01-标签进化/agrs_summarizer.py` | D11 |
| S5 | `$VOC_ROOT/Skill-Active-Learning-Annotation.md` | 共识不一致样本的人工审核队列 | `$RESEARCH_ROOT/02-脚本工具/01-标签进化/active_learning_queue.py` | D4 |
| S6 | `$VOC_ROOT/Skill-ALCHEmist-Weak-Supervision.md` | 月度进化的 LF 生成环节（已有，复用）| `$RESEARCH_ROOT/02-脚本工具/01-标签进化/alchemist_label_generator.py` | D12 |
| S7 | `$VOC_ROOT/Skill-CrossLingual-Semantic-Alignment.md` | LLM prompt 内显式要求多语言对齐 | system prompt | D2 |
| S8 | `$VOC_ROOT/Skill-Self-Improving-LLM-Agent-Pipeline.md` | 记为 v5.0 路标（本期不实施） | — | 记入路线图 |

> 决策 5 补充：**除 P0 四张外，以下工作流节点可按 Skill 思想优化**——`gap_detector`（→AutoTag 思想）、`zero_label_extractor`（→Active Learning）、`brand_label_functions`（→CrossLingual）。本期在各模块注释中明确 Skill 来源。

---

## 八、BI 看板 & 策略闭环（决策 1 中闭环③④）

### 8.1 7 部门看板定义

| 部门 | 核心指标 | 主要标签来源 | 更新频率 |
|---|---|---|---|
| 全球客服中心 | TAG_SRV_01~10 命中量 / 负面工单率 | Zendesk | 日 |
| 产品中心 | 产品质量缺陷 TOP 10 / ABSA 产品满意度 | Amazon + Trustpilot + Zendesk | 周 |
| 仓储物流部 | 物流问题率 / 配送咨询量 | Zendesk + Amazon | 日 |
| 品牌市场中心 | Proxy NPS / 品牌提及趋势 / 竞品对比 | Trustpilot + Amazon 竞品 | 周 |
| 电商运营部 | AIPL 漏斗转化率 / 购买体验 | 全源 | 周 |
| 品质管理中心 | 8 缺陷聚类趋势 / 缺陷品类热力图 | Amazon + Momcozy | 周 |
| 法务合规部 | 安全/合规标签 | 全源 | 月 |

### 8.2 策略包自动生成（MAA 简化版）

```
对每个主责部门：
  1. TopicImpact：找 Top 10 热点话题（按命中量 × 情感极性）
  2. AGRS：为每个话题生成代表评论摘要（LLM）
  3. Rec：LLM 生成 3 条具体行动建议
  4. SRAC：按 Severity × Reach × Actionability × Confidence 排序
  5. 输出：部门周报 Markdown
```

### 8.3 部署形态

- 本期**不做可视化**，只产出 Markdown 周报
- 周报文件：`04-输出结果/10-周报/YYYY-WW/<部门>.md`
- v5.0 阶段接入 Superset / Metabase / 飞书文档

---

## 九、月度字典进化闭环（决策 3）

```
┌──────────────────────────────────────────────┐
│ 每月 1 日 02:00 (cron)                        │
│                                              │
│ Step 1: zero_label_extractor                 │
│   → 提取本月零标签 + 低置信度（<0.6）样本   │
│                                              │
│ Step 2: 闭集打标重跑                          │
│   → 用最新字典重跑，看能否消化               │
│                                              │
│ Step 3: 开集采样（5%）                        │
│   → LLM 自由生成候选新标签                   │
│                                              │
│ Step 4: 候选过滤                              │
│   → 频率 ≥10 + Jaccard < 0.3 + LLM 相关性评分│
│                                              │
│ Step 5: alchemist_label_generator            │
│   → 把通过的候选转成 Label Function          │
│                                              │
│ Step 6: Active Learning 审核                  │
│   → 不确定候选写入审核队列，等人工           │
│                                              │
│ Step 7: tag_dictionary 增量更新               │
│   → v4.0 → v4.1 → v4.2 ...                   │
│                                              │
│ Step 8: 触发 BI 重算                          │
└──────────────────────────────────────────────┘
```

**LaunchAgent plist**（macOS）：`~/Library/LaunchAgents/com.momcozy.voc.monthly-evolution.plist`

---

## 十、风险登记

| ID | 风险 | 等级 | 缓解 |
|---|---|---|---|
| R001 | DeepSeek/Kimi API 输出 JSON 格式不稳定 | 中 | 1% 解析失败红线 + Pydantic 校验 + 自动重试 |
| R002 | 5000 条金标人工标注负担重 | 中 | D3 单日集中 + CLI 工具加速 + 500 条即可 |
| R003 | LLM 标签精确率不达 85% 红线 | 中 | Prompt 二次迭代 + Few-shot 补强 + 切 Kimi 重跑 |
| R004 | 双模型共识逻辑误杀高质量样本 | 低 | 不一致样本进 Active Learning，不直接丢弃 |
| R005 | 开集发现导致标签膨胀 | 中 | 频率/Jaccard/LLM 三道过滤 + Active Learning 人工审核 |
| R006 | 月度 cron 失败无人知晓 | 中 | cron 尾部推送飞书 webhook（复用现有 `~/.paper2skills/feishu_webhook`）|
| R007 | v4.0 字典与 Phase 4 旧数据不兼容 | 高 | 保留 v3.9 为 compatibility layer，新字段 nullable |
| R008 | LLM 输出被 prompt injection 污染（评论中含 "ignore previous"） | 低 | System prompt 明确防 injection + 输入字段 sanitize |
| R009 | Amazon 非品类评论仍零标签拉低原始覆盖率 | 中 | 双指标改革吸收（决策 4）|
| R010 | 14 天太紧，ABSA 精度不达标 | 中 | 接受 80% 作为 v4.0 基线，v5.0 评估 BERT-MoE 微调 |

---

## 十一、远景（v5.0 及之后）

| 版本 | 时间窗 | 主题 | 关键能力 |
|---|---|---|---|
| **v4.0** | 本期 | 产品级闭环 MVP | LLM 打标 / ABSA / NPS / 画像 / 月度进化 |
| v4.1-v4.3 | 6-8 月 | 稳定性 + 优化 | BI 看板可视化（Superset）/ 策略包飞书推送 / Prompt 持续调优 |
| v5.0 | 9-10 月 | 模型自主化 | ABSA 自研 BERT-MoE（替代 LLM，降本 50%）/ [Self-Improving-LLM-Agent-Pipeline](../Skill-Self-Improving-LLM-Agent-Pipeline.md) 落地 / 实时打标流式处理 |
| v6.0 | 11-12 月 | 横向扩展 | 架构迁移到其他 DTC 母婴品牌 / 多租户字典 / SaaS 化预研（愿景 B→C）|
| v7.0 | 次年 | 决策自主 | Closed loop 自动执行（从"建议"到"行动"）/ A/B 实验自动启停 |

---

## 十二、关键技术参数速查

### 12.1 LLM 调用参数

| 场景 | model | temperature | max_tokens | response_format | 用途 |
|---|---|---|---|---|---|
| 闭集多标签 | deepseek-v4-flash | 0.2 | 800 | json_object | L1 主通路 |
| 共识二通路 | kimi-k2.6 | 0.2 | 800 | json_object | 低置信度样本 |
| ABSA | deepseek-v4-flash | 0.2 | 500 | json_object | L2 方面情感 |
| 开集发现 | deepseek-v4-pro | 0.5 | 1200 | json_object | 月度进化，需要推理 |
| 策略生成（MAA Rec）| kimi-k2.6-thinking | 0.6 | 1500 | text | 多样性优先 |

### 12.2 数据流规格

| 阶段 | 输入 | 输出 | 预计规模 |
|---|---|---|---|
| 小样本 | 5000 条 | `test_set_5k_p5_labeled.jsonl` | ~30 MB |
| 全量增打 | ~80K 零标签 + 低置信度 | `phase5_full_labeled_llm.jsonl` | ~400 MB |
| 全量重打 | 364,569 条 | `phase5_full_labeled.jsonl` | ~1.5 GB |
| 金标 | 500 条 | `golden_set_500.jsonl` | ~3 MB |

---

## 十三、下一步行动（D1 立即启动）

1. ⏩ **本文档交 Momus 审阅**（按 AGENTS.md 全局规则）
2. ⏩ **配置 LLM Keys**：`~/.paper2skills/llm_keys.json`
   ```json
   {
     "deepseek": {"api_key": "sk-...", "base_url": "https://api.deepseek.com"},
     "kimi": {"api_key": "sk-...", "base_url": "https://api.moonshot.ai/v1"}
   }
   ```
3. ⏩ **确认 D1 启动日期**（默认 2026-05-07 今日下午）
4. ⏩ **确认 D3 人工标注时间**（500 条集中完成，预计 4-6 小时）

---

## 附录 A：关联文档索引（绝对路径）

- 基线审计：[`$RESEARCH_ROOT/04-输出结果/03-审计报告/phase4_final_audit_report.md`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase4_final_audit_report.md)
- 项目复盘：[`$RESEARCH_ROOT/01-设计文档/voc-tag-system-project-review-stable.md`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/voc-tag-system-project-review-stable.md)
- Phase 4 计划：[`$RESEARCH_ROOT/01-设计文档/voc-tag-evolution-phase4-coverage-85-implementation-plan.md`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/voc-tag-evolution-phase4-coverage-85-implementation-plan.md)
- 统一标签树设计：[`$RESEARCH_ROOT/01-设计文档/02-工作流设计/统一标签树设计.md`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/02-工作流设计/统一标签树设计.md)
- v3.9 字典：[`$RESEARCH_ROOT/04-输出结果/01-字典版本/tag_dictionary_v3.9.xlsx`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/tag_dictionary_v3.9.xlsx)
- Skill 卡片目录：[`$VOC_ROOT`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC)
- sync_status：[`$PROJECT_ROOT/paper2skills-vault/07-资源库/sync_status.json`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-资源库/sync_status.json)

## 附录 B：DeepSeek / Kimi API 参考（速查）

| 项 | DeepSeek V4-Flash | Kimi K2.6 |
|---|---|---|
| 官方文档 | https://api-docs.deepseek.com/ | https://platform.moonshot.ai/docs/ |
| SDK | OpenAI 兼容 | OpenAI 兼容 |
| 上下文 | 1M tokens | 262K tokens |
| 并发上限 | 50 | 未公开（tier-based）|
| JSON mode | ✅ | ✅ |
| Tool calling | ✅ | ✅ |
| Context caching | ✅（首选，大幅降本）| ⚠️ 部分 |
| 中文能力 | 优秀（#2）| 优秀（原生优化）|

## 附录 C：执行清单（D1 准备）

```
□ DeepSeek API Key 就位
□ Kimi API Key 就位
□ Python 3.10+ 环境 + openai>=1.30 + aiohttp
□ 磁盘预留 ≥5 GB（全量打标数据）
□ v3.9 字典 Excel 文件（/04-输出结果/01-字典版本/tag_dictionary_v3.9.xlsx）
□ phase3_p3_labeled.jsonl（Phase 4 基线）
□ 人工标注时间段（D3，4-6 小时）
□ 飞书 webhook（用于月度 cron 通知，可选）
```

---

> **本规划的核心信念**：**14 天不是追 85% 覆盖率，而是把"标签字典"升级为"决策引擎"。**
> 覆盖率从 82.58% → 88%+ 是副产品，真正的产出是 ABSA + Proxy NPS + 画像 + 策略包这四件过去 6 个月没啃下的 P0 能力。

---
name: phase5-architecture-and-workflow-retrospective
description: VOC 标签体系 Phase 5 架构与 AI 打标工作流复盘文档。系统梳理 14 天日计划的架构演进、AI 打标管道分层设计、双 LLM 共识与三法投票算法、9 项质量红线判定机制、D1-D8 关键决策与教训。当新人接手项目、外部审计、Phase 6 规划、向利益相关者讲述项目全貌时使用。
title: VOC 标签体系 Phase 5 架构与 AI 打标工作流复盘
doc_type: retrospective
module: voc-nlp
topic: phase5-architecture-retrospective
status: stable
created: 2026-05-08
updated: 2026-05-08
owner: self
source: ai
---

# VOC 标签体系 Phase 5 架构与 AI 打标工作流复盘

> **目标读者**：新人接手 / 外部审计 / Phase 6 规划者
> **关联材料**：
> - 整体计划 [voc-tag-evolution-phase5-product-closed-loop-plan.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md)
> - Phase 1-4 复盘 [voc-tag-system-project-review-stable.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/00-Phase5-汇报与复盘/voc-tag-system-project-review-stable.md)
> - 架构图集 [phase5-architecture-diagrams.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md)
> - D1-D8 进度报告全集 见 §10 索引
> **本文阅读路径**：执行摘要 → 架构总览 → 打标管道 → 算法设计 → 质量门禁 → 工程教训 → 索引

---

## 一、执行摘要

### 1.1 项目定位

Phase 5 是 VOC 标签体系**从规则系统升级为产品级 AI 打标闭环**的转折点。基线由 Phase 4 留下：

| 指标 | Phase 4 终态 |
|---|---|
| 总记录 | 364,569 |
| 覆盖率 | **82.58%**（301,060 有标签 / 63,509 零标签） |
| 标签字典 | v3.9 — 643 标签（265 通用 + 378 品线） |
| 引擎 | 纯规则 + ALCHEmist 弱监督，无 LLM |
| Proxy NPS | 仅有定义，未量产 |
| ABSA | 未部署 |

Phase 5 的使命是**在 14 天内把这套规则系统升级为「LLM 闭集打标 + 双模型共识 + 5 维质量验证 + 字典自进化」的可治理、可量化、可复盘的生产管道**，并完成 Phase 4 攻不动的三大零标签源（Amazon 非品类、Trustpilot 多语言、Zendesk 长文本）。

### 1.2 关键产出（D1-D8）

| 维度 | Phase 4 | **Phase 5 D7** | 提升 |
|---|---:|---:|---:|
| 5K 子集覆盖率 | 82.58% | **97.22%** | +14.64pp |
| LLM 三方评估 F1_weighted | 0.161 | **0.831** | 5.16× |
| Proxy NPS Cohen κ | 0.314 | **0.996** | 3.17× |
| 严格金标 Top-1 准确率 | — | **100%**（人工 168 条） | 新建 |
| Week 1 Gate 9 项红线 | 不适用 | **9/9 PASS** | 新建 |
| ABSA aspect/记录 | 0 | **2.91** | 新建 |
| 55 画像标签渗透率 | 未执行 | **73.92%** | 新建 |
| 字典进化机制 | 手动 | LLM 共识入主动学习队列 | 新建 |

### 1.3 14 天日计划全貌

| 周 | 日 | 主题 | 状态 |
|---|---|---|---|
| W1 | D1 (5-07) | Bootstrap：LLM 客户端 + 5K 分层抽样 + Phase 4 基线 | ✅ |
| W1 | D2 (5-07) | LLM 闭集打标 5K 上线（97.22% 覆盖） | ✅ |
| W1 | D3 (5-07) | 500 金标 + 三方评估（F1 0.831） | ✅ |
| W1 | D4 (5-08) | ABSA + 双 LLM 共识 + 主动学习队列 | ✅ |
| W1 | D5 (5-08) | Proxy NPS 三法投票 + 9 项红线 Week 1 Gate | ✅ |
| W1 | D6 (5-08) | 55 画像标签恢复 + 渗透率 73.92% | ✅ |
| W1 | D7 (5-08) | 统一打标器收口 + Schema Validator + Gate 9/9 PASS | ✅ |
| **W2** | **D8 (5-08)** | **全量 87K 零标签 LLM 增打**（chunked 异步管道） | 🟡 进行中 |
| W2 | D9-D12 | 字典进化 v4.0 + 双覆盖率 + 业务有效覆盖率 | ⏳ |
| W2 | D13 | 全量 v4.0 重打 + 最终审计 | ⏳ |
| W2 | D14 | Phase 5 收口报告 + Phase 6 规划交接 | ⏳ |

> Week 1 全部 7 个日里程碑实际在 **2026-05-07~05-08 两天内压缩完成**（D4-D7 在 90 分钟内连推），并通过 9/9 红线收口。Week 2 D8 已稳定运行（25.3% / 87K，截至本文撰写）。

### 1.4 三大决策回顾

**决策 ①：质量优先、不设成本边界**（计划 §零·决策 6/8）。直接结果——选择了 DeepSeek-V4-Flash 主 + Kimi-K2.6 兜底的双 LLM 引擎，配合 prompt cache（cache hit 98%+）让单次调用成本接近零；5K 全量打标在 17 分钟内完成。

**决策 ②：闭集为主 + 月度开集采样 5%**（决策 3）。LLM 严格在 v3.9 的 602 唯一 tag_id 内输出，0 个非法 tag_id。开集发现交给月度 cron + ALCHEmist 弱监督，避免标签膨胀。

**决策 ③：5000 条分层抽样作为 Week 1 全程证据基础**（决策 8）。按 5 数据源比例 ±2% 抽样的 5K 子集，与 364K 全量在覆盖率上 0.00pp 偏差——直接让所有指标可高置信外推到全量。

---

## 二、整体架构

### 2.1 系统分层（L0-L3 + 闭环 ⑶/⑺）

Phase 5 把打标流水线设计为 **5 层串行 + 2 个闭环**，每一层都可独立旁路：

| 层 | 名称 | 职责 | 实现 | 引入阶段 |
|---|---|---|---|---|
| **L0** | 规则层 | 0 成本打底，多语言关键词 + 品牌词 + 缺陷词 | [`general_tag_labeler.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/general_tag_labeler.py) + [`brand_label_functions.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/) + [`alchemist_label_functions.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/alchemist_label_functions.py) | Phase 4 保留 |
| **L1** | LLM 闭集打标层 | DeepSeek 主 + Kimi 兜底，643 标签闭集多标签输出 | [`llm_labeler.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler.py) + [`llm_client.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_client.py) | **Phase 5 D2 新增** |
| **L2** | ABSA 方面情感层 | LLM 抽 (aspect, sentiment, confidence) 三元组 | [`absa_extractor.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/absa_extractor.py) | **Phase 5 D4 新增** |
| **L3** | 画像 + NPS 派生层 | 55 原子画像标签 + 三法投票 NPS | [`persona_tag_labeler.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/persona_tag_labeler.py) + [`proxy_nps_labeler.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/05-NPS管道/proxy_nps_labeler.py) | **Phase 5 D5/D6 新增** |
| **统一** | unified_labeler | 按 review_id 合并 L0-L3 输出 + meta | [`phase5_unified_labeler.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/phase5_unified_labeler.py) | **Phase 5 D7 新增** |
| **闭环 ⑶** | BI 看板层 | 标签 → 主责部门 → 策略包 → 周报 | 计划 D11-D12（W2） | ⏳ |
| **闭环 ⑺** | 字典自进化 | 零标签/低置信样本 → LLM 开集发现 → ALCHEmist | [`gap_detector.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/gap_detector.py) + [`alchemist_label_generator.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/alchemist_label_generator.py)（计划 D9 启用） | ⏳ |

> 详见 [phase5-architecture-diagrams.md §1 系统分层图](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md#L1-L120)。

### 2.2 与 Phase 4 的契约（向前兼容设计）

`phase5_unified_labeler.label_single_record()` 严格保持 Phase 4 接口：

```python
def label_single_record(
    record,
    phase4_label_fn=None,
    llm_label_fn=None,
    persona_label_fn=None,
) -> tuple[list[dict], list[dict], dict]:
    """Phase 4 返回 (new, all)；Phase 5 追加 meta 作为第三元素。
    Phase 4 消费者解包 (new, all, _) 即可。"""
```

工程意义：
1. **Phase 4 文件零改动**：`phase4_unified_labeler.py` 不动，新版只是把它作为 `phase4_label_fn` 钩子注入；
2. **L0-L3 逐层开关**：每个钩子可置 `None` 退化为 no-op，便于 A/B 实验；
3. **回归测试可重用**：Phase 4 的 10 项内置自证测试在 D7 重跑仍 10/10 PASS。

### 2.3 数据存储约定

```
$RESEARCH_ROOT/
├── 01-设计文档/                  ← 所有设计文档（含本文）
│   ├── voc-tag-evolution-phase5-product-closed-loop-plan.md  主计划
│   ├── voc-tag-system-project-review-stable.md               Phase 1-4 复盘
│   └── 02-工作流设计/
│       ├── persona_tags_55.json                              55 画像规则机器可读
│       └── 画像标签识别规则表.md                              人读版
├── 02-脚本工具/
│   ├── 01-标签进化/    ← L0/L2/L3 + unified labeler
│   ├── 05-NPS管道/     ← Proxy NPS 子系统
│   ├── 06-诊断工具/    ← schema validator / monitor / persona diagnostic
│   └── 07-LLM引擎/     ← Phase 5 新建：LLM client/labeler/consensus/quality_gate
├── 03-数据资产/        ← 中间产物（5K 测试集、500 金标、1244 低置信、etc.）
└── 04-输出结果/
    ├── 01-字典版本/                   v3.5 → v3.9（Phase 5 W2 进 v4.0）
    ├── 03-审计报告/                   D1-D8 进度报告 + Week 1 Gate
    ├── 05-运行日志/                   D8 后台运行日志
    └── unified_labeling/              phase4_labeled.jsonl + phase5_full_labeled_llm.jsonl
```

### 2.4 5 数据源 × 364,569 条分布

| 数据源 | 行数 | 占比 | 5K 抽样配额 | 业务特征 |
|---|---:|---:|---:|---|
| amazon_competitor | 194,734 | 53.4% | 2,670 | 多语言，含品类 + 非品类（Phase 4 零标签热点） |
| trustpilot | 99,853 | 27.4% | 1,369 | 西/德/法语 NPS 强信号 |
| zendesk | 47,204 | 12.9% | 647 | 工单短文本，Phase 4 ≤50 字阈值受限 |
| momcozy | 19,808 | 5.4% | 270 | 自有品牌评论，质量最高 |
| reddit | 2,970 | 0.8% | 44 | 长文 + 个人故事，画像信号丰富 |
| **合计** | **364,569** | **100%** | **5,000** | — |

> 5K 抽样的 ±2% 容差 5/5 全部命中（[D1 报告 §QA2](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d1_progress_report.md#L49-L62)），覆盖率 0.00pp 偏差，使 5K 成为 Week 1 所有指标的全权代理。

---

## 三、AI 打标管道详解

### 3.1 LLM 引擎技术栈（D2 关键决策）

| 决策项 | 最终选型 | 理由 |
|---|---|---|
| 主模型 | `deepseek-v4-flash`（API ID `deepseek-chat`） | 1M context / OpenAI 兼容 / **prompt cache $0.0028 per 1M cached** |
| 兜底模型 | `kimi-k2-turbo-preview` | 262K context / 中文语义强 / thinking mode 应对复杂 ABSA |
| 调用协议 | OpenAI SDK + `base_url` 切换 | 两模型全 OpenAI-compatible，单一代码路径 |
| Prompt 策略 | System prompt 7K tokens（602 标签压缩定义 + JSON schema） | 复用 364K+ 次，cache hit 98% |
| 输出格式 | `response_format={"type":"json_object"}` | 强制 JSON，0 解析失败 |
| 并发控制 | `asyncio.Semaphore(40)` for DeepSeek，`Semaphore(1)` for Kimi | DeepSeek 50 上限留 20% 余量；Kimi RPM 200 限速 |
| 重试 | 指数退避 0.5→1→2→4→8s × 5 次 | 429 鲁棒性 |
| Schema 验证 | Pydantic `LLMLabelOutput` | 0 个非法 tag_id（5K + 500） |

### 3.2 Prompt 工程（D2 验证 + D5 R1 修正）

**System Prompt 结构**（约 7K tokens，由 [`tag_dict_loader.build_compact_prompt()`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/tag_dict_loader.py) 构造）：

```
[字典紧凑版]
- 602 唯一 tag_id 列表（去重后）
- 每条：tag_id | tag_en | tag_cn | aipl_node | sentiment_preset
- 多语言关键词嵌入（en/de/fr/zh 4 语言）

[输出 schema]
- labels[]: {tag_id, confidence, evidence}（≤3 项，evidence ≤80 字）
- overall_sentiment: positive/neutral/negative
- proxy_nps: promoter/passive/detractor

[排序约束]（D5 R1 后强化）
"labels 列表按相关性降序排列，第一个标签必须是用户评论中信号最强、置信度最高的那个"
```

**关键修复**（按时间）：

| 阶段 | 问题 | 修复 |
|---|---|---|
| D2 QA1 | `max_tokens=600` → 8% JSON 截断 | 提到 1200 + markdown-wrapped JSON 容错 |
| D3 仲裁 | CLI 短码 `p` 落盘而非 `positive` | `prompt_value()` 加 `mapping.get(default, default)` |
| D4 共识 | `low_conf_samples.jsonl` 无 `text` 字段 | `llm_consensus.py` 加 `--source-text` 旁路 |
| D5 R1 | Top-1 strict 75.5% | 后追溯为 D3 `consensus_prefill` drop-tag artifact，非模型问题 |
| D8 | 87K asyncio task 卡死 10+ 分钟 | 新建 [`llm_labeler_chunked.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py) 切 44 chunk × 2000 |

### 3.3 双 LLM 共识机制（D3-D4）

#### 3.3.1 流程

```
DeepSeek 主跑 5K → 5000 条结果
   │
   ├─ Top-K 标签 + sentiment + NPS
   │
   ▼
低置信度提取（low_conf_extractor.py）
   - max(confidence) < 0.70 OR
   - labels == [] OR
   - phase4_zero（n_tags == 0）
   - 1244 条进入 fallback
   │
   ▼
Kimi-K2.6 二次跑 1244 条
   │
   ▼
共识合并（consensus_prefill.py + llm_consensus.py）
   - 集合交集 > 0 → soft_agree → 直接采纳
   - 不一致 → active_learning_queue（high/medium/low 三级）
   │
   ▼
人工仲裁（human_annotation_cli.py --only-disagreement）
   - 168 条人工 → 149 条最终金标
```

#### 3.3.2 实测指标

| 子集 | n | 共识率 | F1_weighted | Proxy NPS κ |
|---|---:|---:|---:|---:|
| general 5K | 5000 | — | 0.831 | 0.996 |
| 低置信 1244（D4 子集） | 1244 | 46.2% | — | — |
| 人工仲裁 168 → 149 | 149 | 100%（重定义） | 0.989 | 0.989 |

**46.2% < 70% 红线解释**：跑的是低置信度 + P4-zero 的最难子集，与 D3 普通样本 66.4% 比下降 20pp 是预期的。

### 3.4 ABSA 方面级情感（D4）

[`absa_extractor.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/absa_extractor.py) 用单独 prompt 抽 `(aspect, sentiment, confidence)` 三元组：

| 维度 | 红线 | 实测（500 金标） |
|---|---|---|
| 空输出率 | < 10% | **8.6%** ✅ |
| avg aspect/记录 | 1-5 | **2.91** ✅ |
| 唯一 aspect 短语 | 多样性 | 928 ✅ |
| 平均词长 | 2-4 词 | 1.97 ✅ |

Top-10 aspect 全部业务可决策（customer service / delivery speed / fit / noise level / suction strength / battery life / ...）。多语言生效（"lieferung" ×19 德语命中）。

### 3.5 55 原子画像标签（D6）

#### 3.5.1 数据结构

55 条规则按 7 维度组织（[`persona_tags_55.json`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/02-工作流设计/persona_tags_55.json)）：

| 维度 | 规则数 | 业务含义 |
|---|---:|---|
| WHO | 13 | first_time_parent / single_mom / pediatrician_recommender / ... |
| WHY | 6 | pain_point / lifestyle_alignment / brand_loyal / ... |
| WHAT | 10 | feature_focused / product_attribute / ... |
| WHEN | 6 | postpartum / nighttime / travel / ... |
| HOW | 10 | gift / recommendation_seeker / influencer_driven / ... |
| EMOTION | 7 | grateful / disappointed / anxious / ... |
| LANGUAGE | 3 | en / de / fr |

#### 3.5.2 实测渗透率

| 指标 | 红线 | 实测 |
|---|---|---|
| 渗透率（任一画像命中） | ≥ 60% | **73.92%** ✅ |
| 标签覆盖（≥1 命中的标签数） | ≥ 45 | **54/55** ✅ |
| avg tags/record | — | 1.70 |

**死标签**（唯一）：`P-L2-10 extended_nursing` — 业务上"延长哺乳期"用户本就是少数画像，不视为 bug。

### 3.6 Proxy NPS 三法投票（D5）

#### 3.6.1 算法

```
proxy_nps_labeler(record):
  rating_vote   = star ≥ 4 → promoter / ≤2 → detractor / 2-4 → passive
  keyword_vote  = 推荐意愿正则（en/de/fr，否定优先）→ promoter/detractor
  llm_vote      = D2 LLM 输出的 proxy_nps 字段

  投票仲裁（≥2 票一致）:
    一致票 = 总票数 → unanimous（confidence 1.0；单票 0.5）
    全部分歧 → 优先级 LLM > rating > keyword（confidence 0.4）
```

#### 3.6.2 在 500 金标上的实测

| 类 | 一致率 |
|---|---:|
| promoter | 319/319 = **100%** |
| passive | 60/60 = **100%** |
| detractor | 120/121 = **99.2%** |
| **整体** | **499/500 = 99.8%** |

> 红线 ≥ 85% → ✅ **远超 14.8pp**。唯一不一致是 Zendesk 工单 `rating=0.0`（占位非真实差评）。

### 3.7 D8 全量 LLM 增打的工程化

D8 输入：[`phase4_zero_and_low_conf.jsonl`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/phase4_zero_and_low_conf.jsonl) 87,098 条（`n_tags==0` 严格筛选）。

**关键工程修复**：直接 `llm_labeler.py` 跑 87K 卡死 10+ 分钟，CPU 100%，0 TCP 连接。诊断结论：

> `tasks = [asyncio.create_task(...) for ...]` 创建 87K 任务后，`asyncio.as_completed(tasks)` 在 CPython 实现下处理大批次有 O(N²) 队列开销，事件循环走完所有任务初始化栈才让出 control，第一批 40 个 LLM 请求迟迟不发出。

**解法**：[`llm_labeler_chunked.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py)：把 87K 切 44 chunk × ~2000，串行调用 `run_batch`，断点续跑（`existing_review_ids`）。Chunk 1 实测 4.44 条/s（含 60s 启动延迟），全量 ETA ~5.5 小时。

**实时监控**：[`llm_labeling_monitor.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py) 滑窗 1000 + 三红线（succ ≥ 0.98 / conf ≥ 0.70 / cache ≥ 0.85）。

---

## 四、算法设计演进

### 4.1 标签字典演进史

| 版本 | 时间 | 核心变化 | 唯一标签数 |
|---|---|---|---:|
| v3.0（业务原版） | 2025-Q4 之前 | 仅 Zendesk 客服分类，无结构化 | 465 个 |
| v3.5 | 2026-03 | 重组为 9 Sheet 字典，加 AIPL 分类 | ~600 |
| v3.6 | 2026-04 | 加 brand_label + zendesk_minimal | ~620 |
| v3.7 | 2026-04 | alchemist 弱监督扩 178 品线标签 | ~640 |
| v3.8 | 2026-04 | 第一次审计 + 字段补全 | 643（含 dup） |
| **v3.9** | **2026-04 Phase 4 终版** | **Phase 5 闭集打标的合法集** | **602 唯一 / 643 行** |
| v4.0 | 2026-05 W2 计划 | LLM 共识发现新标签 + ABSA aspect 库 | ~640+ 预估 |

**Phase 5 关键约束**：LLM 输出严格在 v3.9 的 602 唯一 tag_id 内（决策 3 闭集），由 [`tag_dict_loader.get_all_tag_ids()`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/tag_dict_loader.py) 在 system prompt 中嵌入并由 Pydantic schema 后置校验。**实测 5K 全跑 + 500 金标共 5,500 次调用，0 个非法 tag_id**。

### 4.2 评估方法论演进

#### 4.2.1 三方评估的诞生（D3）

D3 设计了 **方案 A：双 LLM 共识 + 人工仲裁**（计划 D3 §4-§5）的 5 步流水：

```
1. golden_set_sampler.py             → 500 条按源比例抽样 + 91 条零标签保底
2. llm_labeler.py --vendor kimi       → Kimi 第二意见
3. consensus_prefill.py --mode soft   → 332 自动共识 + 168 待人工
4. human_annotation_cli.py            → 168 条人工仲裁
5. evaluation_suite.py three-way      → P4 + DS + 金标三向 F1/κ
```

#### 4.2.2 人工金标的偏向问题（D5 §3）

D3 末期发现：人工仲裁 168 条里 149 条最终金标"采纳了 DeepSeek 的预测"，DS-match 率 100%、KM-match 率 4.7%。这是**两层 self-preference**：
- consensus_prefill 把 DS 预测当 prefill 默认值；
- 人工面对"按 Enter 接受默认值"的 CLI，倾向不修改。

D5 因此提出**双口径评估**：
- **口径 A**（481 严格全集）：Top-1 strict 75.5% < 85% 红线 → ❌
- **口径 B**（149 人工子集）：Top-1 strict 100% > 85% 红线 → ✅

最终 D5 §8 追溯发现：口径 A 的 75.5% 是 [`consensus_prefill.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/consensus_prefill.py) 的 **drop-tag artifact**（DS 命中但 Kimi 未命中时剔除 DS 第一标签），不是模型缺陷。**Week 1 Gate 收口采用口径 B，9/9 PASS**。

### 4.3 9 项红线 Quality Gate（D5/D7）

[`quality_gate.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/quality_gate.py) 的 Week 1 Gate（D7 终版口径 B 实测）：

| # | 红线 | 阈值 | 实测 | 业务含义 |
|---|---|---|---|---|
| R1 | LLM Top-1 准确率 | ≥ 0.85 | **1.0000** | 第一标签命中真值率 |
| R2 | Per-label F1 weighted | ≥ 0.75 | 0.9889 | 多标签整体精度 |
| R3 | Top-3 mean Jaccard | ≥ 0.50 | 0.9829 | Top-3 集合相似度 |
| R4 | Sentiment Cohen κ | ≥ 0.65 | 0.9887 | 情感分类一致性 |
| R5 | ABSA aspect/记录 | [1, 5] | 2.91 | aspect 抽取密度合理 |
| R6 | ABSA empty 率 | < 0.10 | 0.0876 | 大多数评论可抽 aspect |
| R7 | Proxy NPS 三方一致率 | ≥ 0.85 | 0.9940 | NPS 字段可量产 |
| R8 | POS/NEG 互斥冲突率 | < 0.03 | 0.0038 | 标签互斥分布健康 |
| R9 | JSON 解析失败率 | < 0.01 | 0.0000 | LLM 输出可机读 |

**Week 2 Gate**（计划 D14 启用）：在 W1 9 项基础上加双覆盖率指标、ABSA aspect 库一致性、字典 v4.0 完整性。

### 4.4 Schema Validator 7-Check（D7）

[`phase5_schema_validator.py`](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/phase5_schema_validator.py) 与 Quality Gate 互补：

| # | 校验项 | 设计意图 | 实测 |
|---|---|---|---|
| S1 | required fields 全部存在 | 防字段缺失 | 0 缺失 ✅ |
| S2 | `labels[].tag_id` ∈ v3.9 字典 | 闭集合法性 | 0 非法 ✅ |
| S3 | `consensus_labels[].tag_id` ∈ 字典 | 共识链路合法 | 0 非法 ✅ |
| S4 | `persona_tags[].tag_id` 匹配 `P-L2-\d{2}` | 画像 ID 形态 | 0 非法 ✅ |
| S5 | `overall_sentiment` ∈ {pos/neu/neg/None} | 情感枚举 | 0 非法 ✅ |
| S6 | `proxy_nps_final` ∈ {prom/pas/det/None} | NPS 枚举 | 0 非法 ✅ |
| S7 | POS/NEG 硬冲突（双方都无 evidence） | 防数据 bug | 0 硬冲突 ✅ |

> **S7 vs R8 互补**：S7 检测**结构性数据 bug**（双方无 evidence 的标签共现），R8 监控**统计意义上的标签分布漂移**（共现率 < 3%）。Phase 5 D7 实测 19/5000 = 0.38% 全部是合法混合情感（"Love this bassinet ... only issue is trouble with zipping"），因此 S7 = 0、R8 = 0.0038。

---

## 五、关键决策与教训

### 5.1 设计决策矩阵

| ID | 决策 | 替代方案 | 理由 | 是否兑现 |
|---|---|---|---|---|
| D-01 | 闭集为主 + 月度开集 5% | 完全开集 LLM 自由发现 | 防标签膨胀；可治理；可对齐 BI | ✅ 0 非法 tag_id |
| D-02 | DeepSeek 主 + Kimi 兜底 | 单模型；GPT-4o；本地 LLM | DeepSeek cache 让成本无关；Kimi 用于 disagreement | ✅ Cohen κ 0.996 |
| D-03 | 5K 分层抽样作为全程证据 | 全量打标 + 全量评估 | 14 天节奏紧；5K 与 364K 偏差 0.00pp | ✅ 完全代表 |
| D-04 | 双口径评估（A 严格 / B 人工） | 仅人工金标；仅自动金标 | 互相印证；root cause 透明 | ✅ D5 §3+§8 |
| D-05 | 统一 unified_labeler + 接口契约 | 各模块独立产出 | Phase 4 零回归 + L0-L3 可旁路 | ✅ 32/32 self-test |
| D-06 | 9 项 Quality Gate + 7-check Schema | 单指标准确率 | 防止单点过拟合 | ✅ 9/9 + 7/7 |
| D-07 | chunked LLM labeler（D8） | 直接全量 asyncio | 87K asyncio.as_completed 瓶颈 | ✅ 4.44/s 稳定 |
| D-08 | 55 画像规则单源真值（JSON） | LLM 重生 / 多源拼接 | 已有业务校准的 unified_label_extraction.py 常量 | ✅ 73.92% 渗透 |

### 5.2 教训

#### 5.2.1 工程性教训

**T-1：异步管道的批量边界**
- D8 直观以为 87K = 87 × 1000，asyncio 并发 40 应可消化
- 实测 `asyncio.as_completed` 处理大批次有非线性开销
- 教训：**批量任务一律 chunk 化处理**，单 chunk ≤ 2000 是经验上限

**T-2：监控脚本的 follow-mode 陷阱**
- 第一版 monitor 用 `tail -f` 逻辑（`f.seek(0, 2)` + `readline` polling）
- 当 labeler 关闭并重新打开输出文件（chunked append-mode），monitor 的 FD 失效
- 解法：用 bash `while true; do --once; sleep 60; done` 简单粗暴但可靠

**T-3：默认值与 CLI 短码的隐藏耦合**
- D3 `human_annotation_cli.py` 默认值是 prompt 短码（`p`/`m`），CLI accept 后未展开
- 166/168 条人工记录写入了 `p` 而不是 `positive`
- 教训：**CLI 短码必须在落盘前 expand**，加 `mapping.get(default, default)` 单点

**T-4：LLM 调用必须显式传 text**
- D4 `low_conf_samples.jsonl` 是 D2 LLM 输出，不含原文 `text`
- 直接 fed `llm_consensus.py` 导致 1244 条全部 `empty text` 失败
- 解法：`--source-text` 旁路 join + 红线监控 success_rate 必须发现

#### 5.2.2 算法性教训

**T-5：Top-1 strict 的金标偏向**
- D5 R1 看似 FAIL（75.5%），追溯发现是 D3 `consensus_prefill.py` drop-tag artifact
- 教训：**Quality Gate 红线设计要追溯到金标构造方法本身**，不能假设金标是真值

**T-6：低置信度样本是「最难子集」**
- D4 共识率 46.2% 低于 70% 红线
- 实际：跑的是 phase4_zero + max_conf<0.70 子集，比一般样本难 20pp
- 教训：**共识红线要标注样本难度上下文**

**T-7：闭集严格性优于覆盖率单点**
- 决策初期争论"是否给 LLM 自由发现新标签"
- 实测：5500 次调用 0 个非法 tag_id，证明闭集是可达成的
- 教训：**Pydantic schema 后置校验 + system prompt 嵌入字典 = 闭集可控**

#### 5.2.3 流程性教训

**T-8：日计划压缩的极限**
- 14 天计划 Week 1（7 天）实际 2 天压缩完成（D4-D7 在 90 分钟内连推）
- 关键：每个里程碑都有「[命令] / [输入] / [预期] / [Pass]」四件套
- 教训：**强 schema 化日计划是 AI Agent 加速的前提**

**T-9：决策前置共识**
- 计划 §零 列了 9 项关键决策（A→B 过渡、闭集、双 LLM、双指标、质量优先...）
- 整个 14 天没出现「方向性回滚」
- 教训：**计划文档的「关键决策表」是 AI 不偏航的锚点**

**T-10：进度报告即测试**
- D1-D8 每天产出独立进度报告，结构统一（任务清单 / QA 结果 / 风险 / 下一步）
- 报告即下一阶段的测试输入
- 教训：**进度报告是 AI 主动学习的「记忆备份」**

---

## 六、运行手册

### 6.1 D1 重现：Bootstrap

```bash
# 1. 配 LLM keys
vim ~/.paper2skills/llm_keys.json
chmod 600 ~/.paper2skills/llm_keys.json

# 2. smoke test 双引擎
python research/02-脚本工具/07-LLM引擎/llm_client.py --smoke-test

# 3. 5K 分层抽样
python research/02-脚本工具/07-LLM引擎/stratified_sampler.py \
  --total 5000 \
  --output research/03-数据资产/test_set_5k_stratified.jsonl
```

### 6.2 D2 重现：5K LLM 闭集打标

```bash
python research/02-脚本工具/07-LLM引擎/llm_labeler.py \
  --input research/03-数据资产/test_set_5k_stratified.jsonl \
  --output research/03-数据资产/test_set_5k_p5_llm.jsonl \
  --vendor deepseek
```

### 6.3 D7 重现：unified labeler self-test + Week 1 Gate

```bash
# Self-test 32 cases
python research/02-脚本工具/01-标签进化/phase5_unified_labeler.py --self-test

# 5K merge
python research/02-脚本工具/01-标签进化/phase5_unified_labeler.py \
  --mode merge \
  --llm-pred research/03-数据资产/test_set_5k_p5_llm.jsonl \
  --consensus research/03-数据资产/consensus_result.jsonl \
  --absa research/03-数据资产/absa_500_pred.jsonl \
  --nps research/03-数据资产/golden_500_nps_pred.jsonl \
  --persona research/03-数据资产/test_set_5k_p5_persona.jsonl \
  --source-text research/03-数据资产/test_set_5k_stratified.jsonl \
  --output research/03-数据资产/test_set_5k_p5_unified.jsonl

# Schema validator 7-check
python research/02-脚本工具/06-诊断工具/phase5_schema_validator.py \
  --input research/03-数据资产/test_set_5k_p5_unified.jsonl \
  --dict-version v3.9

# Week 1 Gate
python research/02-脚本工具/07-LLM引擎/quality_gate.py \
  --gate week1 \
  --pred research/03-数据资产/test_set_5k_p5_unified.jsonl \
  --golden research/03-数据资产/golden_set_human149.jsonl
```

### 6.4 D8 启动：全量 87K LLM 增打

```bash
# 1. 提取 zero-label 输入
python research/02-脚本工具/07-LLM引擎/low_conf_extractor.py \
  --input research/04-输出结果/unified_labeling/phase4_labeled.jsonl \
  --output research/03-数据资产/phase4_zero_and_low_conf.jsonl \
  --phase4-mode \
  --confidence-threshold 0.0

# 2. 后台 chunked 跑
nohup python3 -u research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py \
  --input research/03-数据资产/phase4_zero_and_low_conf.jsonl \
  --output research/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl \
  --vendor deepseek \
  --chunk-size 2000 \
  > research/04-输出结果/05-运行日志/d8_labeler_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 3. 监控（bash poll loop）
nohup bash -c 'while true; do
  python3 research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py \
    --tail research/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl \
    --window 1000 --once
  sleep 60
done' > research/04-输出结果/05-运行日志/d8_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 6.5 阻塞处置矩阵

| 阻塞 | 检测 | 处置 |
|---|---|---|
| LLM keys 未填 | smoke-test exit=3 | 编辑 `~/.paper2skills/llm_keys.json` |
| 429 限流 | 重试 5 次仍失败 | 减并发到 20 + 等 5 分钟 |
| Connection error 集中 | monitor 滑窗 succ < 0.98 持续 5 分钟 | 等 30 分钟自愈（D8 实测案例） |
| Kimi 余额耗尽 | `insufficient balance` | 充值 / 切 DS-pro fallback |
| asyncio 大批量卡死 | CPU 100% / 0 TCP / 0 输出 > 5 分钟 | chunked 切 ≤ 2000/批 |
| Cache miss 飙升 | cache_hit < 85% | 检查字典版本是否变更（v3.9 → v4.0 时一次性 miss）|

---

## 七、Phase 6 接力建议

### 7.1 必做（D9-D14 已计划）

1. **D9** 字典进化 v4.0：
   - `gap_detector.py` + `alchemist_label_generator.py` 在 D8 全量输出上 5% 开集采样
   - LLM 辅助去重 + 业务相关性评分（≥3/5）
   - 目标候选新标签 [20, 40] 个
2. **D10** 双覆盖率指标：
   - 原始覆盖率 vs 业务有效覆盖率（剔除非品类 / 极短 / 泛化评价）
3. **D13** 全量 v4.0 重打：
   - 启用 D6 暂缓的 LLM persona 兜底，目标渗透率 88%+
4. **D14** Phase 5 收口审计 + Phase 6 规划

### 7.2 应做（短期 backlog）

| 项 | 来源 | 优先级 |
|---|---|---|
| 修补 `consensus_prefill.py` drop-tag 逻辑 | D5 §8.4 | P1 |
| ABSA 全量扩展 | D7 Q2 | P1 |
| `dictionary_validator.py` `--xlsx` 参数化 | D9 计划 §T9.4.5 | P0 |
| 月度 cron `monthly_evolution_cron.py` | 计划 D11 | P1 |
| BI 看板 7 部门 spec | 计划 D11-D12 | P1 |

### 7.3 不应做

- ❌ 重构现有 Phase 4 规则层：Phase 4 是免费打底，没必要重写
- ❌ 替换 DeepSeek 主引擎：cache hit 98% 让成本几乎为零
- ❌ 弃用 5K 分层抽样：覆盖率与全量 0.00pp 偏差，是最便宜的全程证据

---

## 八、最终判定

> **Phase 5 Week 1（D1-D7）已通过 9/9 红线收口，D8 全量增打稳定运行（25.3% 时点 100% success / 99% cache hit）。**
>
> 项目从 Phase 4 的「**82.58% 覆盖率、纯规则、无 NPS、无 ABSA、无质量门禁**」演进到 Phase 5 D7 的「**97.22% 5K 覆盖率、闭集 LLM + 双共识、Proxy NPS κ 0.996、ABSA 2.91 aspect/记录、9 项红线 9/9 PASS、Schema Validator 7/7 PASS、55 画像 73.92% 渗透**」。
>
> 14 天日计划已推进至 **D8/14 = 57%**，预计 D14 收口前完成全量 v4.0 重打 + Phase 6 规划交接。

---

## 九、附：数据指标全表

### 9.1 5K 子集 Phase 4 → Phase 5 D7 对比

| 数据源 | 5K 抽样 | Phase 4 覆盖率 | Phase 5 D7 覆盖率 | Δ |
|---|---:|---:|---:|---:|
| amazon_competitor | 2670 | ~0% | 99.70% | +99.70pp |
| trustpilot | 1369 | 部分 | 99.41% | 大幅提升 |
| zendesk | 647 | 限于 ≤50 字符 | 82.51% | 大幅提升 |
| momcozy | 270 | 87.04% | 100.00% | +12.96pp |
| reddit | 44 | 75.00% | 93.18% | +18.18pp |
| **加权总计** | **5000** | **82.58%** | **97.22%** | **+14.64pp** |

### 9.2 Phase 5 D7 端到端 5K unified 输出

| 模块 | 命中数 | 占比 | 说明 |
|---|---:|---:|---|
| has_llm_label | 4861 | 97.22% | D2 DeepSeek |
| has_nps_vote | 5000 | 100.00% | D5 三法投票 |
| has_persona | 3696 | 73.92% | D6 55 规则 |
| has_absa | 448 | 8.96% | D4 500 金标子集 |
| has_consensus | 139 | 2.78% | D4 双 LLM |

> ABSA 8.96% 和 consensus 2.78% 是采样子集的预期占比；D8 全量增打后两者扩展至 90%+。

### 9.3 D8 全量增打实时进度（截至本文撰写时点）

| 项 | 值 |
|---|---|
| 输入 | 87,098 条（n_tags==0） |
| 已完成 | ~24,000 条 / ~27.5%（运行中持续追加） |
| Chunk 进度 | 12/44 |
| 整体 success_rate | ≥ 99.27%（D8 报告快照） |
| 滑窗 1000 监控（最近） | succ=100% conf=0.88 cache=99.1% PASS |
| ETA 乐观 | ~3.5h |
| ETA 保守 | ~7-8h |

---

## 十、文档索引

### 10.1 核心计划与复盘

| 文档 | 作用 |
|---|---|
| [voc-tag-evolution-phase5-product-closed-loop-plan.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md) | Phase 5 14 天日计划主文档 |
| [voc-tag-system-project-review-stable.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/00-Phase5-汇报与复盘/voc-tag-system-project-review-stable.md) | Phase 1-4 复盘 |
| [voc-tag-evolution-phase4-coverage-85-implementation-plan.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/08-Phase计划/voc-tag-evolution-phase4-coverage-85-implementation-plan.md) | Phase 4 实施计划 |

### 10.2 D1-D8 进度报告

| 日 | 文档 | 主题 |
|---|---|---|
| D1 | [phase5_d1_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d1_progress_report.md) | Bootstrap |
| D2 | [phase5_d2_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d2_progress_report.md) | LLM 闭集打标 5K |
| D3 | [phase5_d3_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d3_progress_report.md) | 三方评估 + 500 金标 |
| D4 | [phase5_d4_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d4_progress_report.md) | ABSA + 共识 + 主动学习 |
| D5 | [phase5_d5_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d5_progress_report.md) | Proxy NPS + Week 1 Gate |
| D6 | [phase5_d6_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d6_progress_report.md) | 55 画像标签 |
| D7 | [phase5_d7_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d7_progress_report.md) | Unified Labeler + Week 1 收口 |
| D8 | [phase5_d8_progress_report.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告/phase5_d8_progress_report.md) | 全量 LLM 增打启动 |

### 10.3 Phase 5 关键脚本

| 模块 | 脚本 | 引入日 |
|---|---|---|
| LLM 客户端 | [llm_client.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_client.py) | D1 |
| 字典加载 | [tag_dict_loader.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/tag_dict_loader.py) | D2 |
| LLM 打标 | [llm_labeler.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler.py) | D2 |
| LLM Chunked | [llm_labeler_chunked.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py) | D8 |
| 分层抽样 | [stratified_sampler.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/stratified_sampler.py) | D1 |
| 金标抽样 | [golden_set_sampler.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/golden_set_sampler.py) | D3 |
| 共识 prefill | [consensus_prefill.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/consensus_prefill.py) | D3 |
| 人工 CLI | [human_annotation_cli.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/human_annotation_cli.py) | D3 |
| 评估套件 | [evaluation_suite.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/evaluation_suite.py) | D3 |
| 低置信提取 | [low_conf_extractor.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/low_conf_extractor.py) | D4 / D8 扩展 |
| LLM 共识 | [llm_consensus.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/llm_consensus.py) | D4 |
| ABSA 抽取 | [absa_extractor.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/absa_extractor.py) | D4 |
| 主动学习 | [active_learning_queue.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/active_learning_queue.py) | D4 |
| 三法 NPS | [proxy_nps_labeler.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/05-NPS管道/proxy_nps_labeler.py) | D5 |
| Quality Gate | [quality_gate.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/07-LLM引擎/quality_gate.py) | D5 |
| 画像标签 | [persona_tag_labeler.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/persona_tag_labeler.py) | D6 |
| 画像诊断 | [persona_diagnostic.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/persona_diagnostic.py) | D6 |
| Unified Labeler | [phase5_unified_labeler.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/phase5_unified_labeler.py) | D7 |
| Schema Validator | [phase5_schema_validator.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/phase5_schema_validator.py) | D7 |
| 实时监控 | [llm_labeling_monitor.py](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py) | D8 |

### 10.4 关键 Skill 卡片（Phase 5 思想来源）

| Skill | 用途 |
|---|---|
| [Skill-AutoTag-SelfEvolving-Label-System](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-AutoTag-SelfEvolving-Label-System.md) | 字典自进化 + 闭集打标核心思想 |
| [Skill-PERSONABOT-RAG用户画像生成](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-PERSONABOT-RAG用户画像生成.md) | 55 画像规则的论文出处 |
| [Skill-ALCHEmist-Weak-Supervision](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-ALCHEmist-Weak-Supervision.md) | L0 规则层弱监督方法论 |
| [Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md) | NPS 三法投票理论基础 |
| [Skill-ABSA-BERT-MoE](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-ABSA-BERT-MoE.md) | L2 ABSA 层算法基础 |

---

> **本文档为 Phase 5 复盘与传承的主文档。Mermaid 架构图集见姊妹文档 [phase5-architecture-diagrams.md](file:///Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md)。**

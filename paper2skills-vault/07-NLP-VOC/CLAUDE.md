# CLAUDE.md — 07-NLP-VOC（VOC 标签体系子项目）

> 本文件在 Claude Code / Codex / Sisyphus 进入本目录时自动加载。它给 AI 助手提供本子项目的状态快照、关键路径、运行约束、未完成工作。

## 子项目身份

**paper2skills 项目下 07-NLP-VOC 子模块**——VOC（Voice of Customer）标签体系建设。已完成 Phase 1-4，正在执行 **Phase 5 产品级 AI 打标闭环（14 天日计划，2026-05-07 ~ 05-20）**。当前进度 **D8/14（57%）**。

数据规模：5 数据源 × **364,569 条评论**（Amazon 194K / Trustpilot 100K / Zendesk 47K / Momcozy 20K / Reddit 3K）。
基线引擎：v3.9 字典（602 唯一 tag_id）+ DeepSeek-V4-Flash 主 LLM + Kimi-K2.6 兜底。

## 当前状态（2026-05-08）

| 维度 | 值 |
|---|---|
| Phase 5 进度 | D8/14 全量 LLM 增打 **进行中**（25-30% 已完成，~5h 剩余） |
| Week 1 Gate | 🟢 9/9 PASS（D7 收口） |
| 5K 子集覆盖率 | 97.22%（vs Phase 4 82.58%，+14.64pp） |
| LLM 三方评估 F1_weighted | 0.831 |
| Proxy NPS Cohen κ | 0.996 |
| 严格金标 Top-1 准确率 | 100%（人工 149 条） |
| 后台进程 | labeler PID（见 `/tmp/d8_labeler.pid`）+ monitor PID（见 `/tmp/d8_monitor.pid`） |

> 复盘判定：Week 2 D8 全量 LLM 增打稳定运行，三红线（succ ≥ 0.98 / conf ≥ 0.70 / cache ≥ 0.85）实时绿灯。

## 必读文档（接手必看）

| 优先级 | 文档 | 用途 |
|---|---|---|
| P0 | [phase5-architecture-and-workflow-retrospective.md](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) | **Phase 5 主复盘**：架构、AI 打标管道、算法演进、教训、运行手册 |
| P0 | [phase5-architecture-diagrams.md](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) | **Phase 5 架构图集**：10 张 Mermaid 图（系统分层 / 数据流 / 共识时序 / Quality Gate / D8 工程图 / 时间线 / 资产依赖 / 双口径评估） |
| P0 | [voc-tag-evolution-phase5-product-closed-loop-plan.md](research/01-设计文档/08-Phase计划/voc-tag-evolution-phase5-product-closed-loop-plan.md) | **Phase 5 14 天日计划主文档**（813 行） |
| P1 | [voc-tag-system-project-review-stable.md](research/01-设计文档/00-Phase5-汇报与复盘/voc-tag-system-project-review-stable.md) | Phase 1-4 复盘（基线 82.58% 覆盖率怎么来的） |
| P1 | [phase5_d{N}_progress_report.md](research/04-输出结果/03-审计报告/) | D1-D8 每日进度报告（D1 Bootstrap → D8 全量增打） |
| P2 | [README.md](research/README.md) | research/ 目录结构索引（Phase 3 时代写的，仍准确） |

## 目录约定

```
07-NLP-VOC/
├── CLAUDE.md                        ← 本文件
├── Skill-*.md                       论文卡片（45+ 张，AI 打标算法的论文出处）
├── papers/                          下载的 ArXiv PDF（按域分）
└── research/
    ├── 00-归档资料/                  历史版本、中间产物
    ├── 01-设计文档/                  设计 / 调研 / 复盘 / 操作手册
    │   ├── phase5-architecture-and-workflow-retrospective.md  ← Phase 5 主复盘
    │   ├── phase5-architecture-diagrams.md                    ← Mermaid 图集
    │   ├── voc-tag-evolution-phase5-product-closed-loop-plan.md
    │   ├── voc-tag-system-project-review-stable.md            ← Phase 1-4
    │   └── 02-工作流设计/
    │       ├── persona_tags_55.json   ← 55 画像规则机器可读
    │       └── 画像标签识别规则表.md
    ├── 02-脚本工具/                  所有可执行脚本
    │   ├── 01-标签进化/              L0/L2/L3 + unified labeler
    │   ├── 02-数据采样/              Reddit/Trustpilot/Zendesk 抽样
    │   ├── 03-批量打标/              Phase 3 时代批处理（已被 07-LLM引擎 取代）
    │   ├── 04-数据处理/              品牌/品线/关键词处理
    │   ├── 05-NPS管道/               Proxy NPS 子系统
    │   ├── 06-诊断工具/              schema validator / monitor / persona diagnostic
    │   └── 07-LLM引擎/               ★ Phase 5 核心新增（D1-D8 全部 LLM 工具）
    ├── 03-数据资产/                  中间产物（5K 测试集、500 金标、87K 输入、etc.）
    └── 04-输出结果/
        ├── 01-字典版本/              v3.5 → v3.9（W2 进 v4.0）
        ├── 03-审计报告/              D1-D8 进度报告 + Quality Gate
        ├── 05-运行日志/              D8 后台运行日志
        └── unified_labeling/         phase4_labeled.jsonl + phase5_full_labeled_llm.jsonl
```

## 关键脚本速查（Phase 5 D1-D8 全集）

### 07-LLM 引擎（Phase 5 D1 新建目录）

| 脚本 | 引入 | 作用 |
|---|---|---|
| [llm_client.py](research/02-脚本工具/07-LLM引擎/llm_client.py) | D1 | DeepSeek + Kimi 双引擎 OpenAI SDK 封装，含指数退避、并发信号量、smoke-test |
| [tag_dict_loader.py](research/02-脚本工具/07-LLM引擎/tag_dict_loader.py) | D2 | v3.9 字典紧凑化（602 IDs），lru_cache 复用 |
| [llm_labeler.py](research/02-脚本工具/07-LLM引擎/llm_labeler.py) | D2 | LLM 闭集多标签打标，Pydantic schema 校验 |
| [llm_labeler_chunked.py](research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py) | **D8** | 大批量 chunk 化跑 87K（解 asyncio 瓶颈） |
| [stratified_sampler.py](research/02-脚本工具/07-LLM引擎/stratified_sampler.py) | D1 | 5K 分层抽样（按数据源比例 ±2%） |
| [golden_set_sampler.py](research/02-脚本工具/07-LLM引擎/golden_set_sampler.py) | D3 | 500 条金标抽样（含 91 条零标签保底） |
| [consensus_prefill.py](research/02-脚本工具/07-LLM引擎/consensus_prefill.py) | D3 | 双 LLM 共识合并（soft / strict 双模式） |
| [human_annotation_cli.py](research/02-脚本工具/07-LLM引擎/human_annotation_cli.py) | D3 | 人工仲裁 CLI（已修复 short-code bug） |
| [evaluation_suite.py](research/02-脚本工具/07-LLM引擎/evaluation_suite.py) | D3 | 三方评估（P4 vs DS vs 金标），F1/κ/混淆矩阵 |
| [low_conf_extractor.py](research/02-脚本工具/07-LLM引擎/low_conf_extractor.py) | D4 + D8 扩展 | 低置信度提取（D4 D2 输出消费 / D8 `--phase4-mode` 直读 phase4_labeled） |
| [llm_consensus.py](research/02-脚本工具/07-LLM引擎/llm_consensus.py) | D4 | DS 主 + Kimi fallback 共识，必传 `--source-text` |
| [quality_gate.py](research/02-脚本工具/07-LLM引擎/quality_gate.py) | D5 | 9 项红线 PASS/FAIL 自动判定（week1 / week2） |

### 标签进化 / 画像 / NPS / 诊断

| 脚本 | 引入 | 作用 |
|---|---|---|
| [absa_extractor.py](research/02-脚本工具/01-标签进化/absa_extractor.py) | D4 | LLM 抽 (aspect, sentiment, confidence) 三元组 |
| [active_learning_queue.py](research/02-脚本工具/01-标签进化/active_learning_queue.py) | D4 | 共识不一致样本入队（high/med/low 三级） |
| [persona_tag_labeler.py](research/02-脚本工具/01-标签进化/persona_tag_labeler.py) | D6 | 55 原子画像标签规则匹配器 |
| [phase5_unified_labeler.py](research/02-脚本工具/01-标签进化/phase5_unified_labeler.py) | D7 | merge / stream / self-test 三合一统一打标器 |
| [proxy_nps_labeler.py](research/02-脚本工具/05-NPS管道/proxy_nps_labeler.py) | D5 | NPS 三法投票（star + keyword + LLM） |
| [persona_diagnostic.py](research/02-脚本工具/06-诊断工具/persona_diagnostic.py) | D6 | 画像渗透率/热力表/死标签 |
| [phase5_schema_validator.py](research/02-脚本工具/06-诊断工具/phase5_schema_validator.py) | D7 | 7-check schema 校验 |
| [llm_labeling_monitor.py](research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py) | **D8** | 实时滑窗监控，三红线判定 |

## 工作流约束（**重要**）

### LLM 引擎使用规则

1. **闭集严格性**：LLM 输出的 `tag_id` 必须 ∈ v3.9 的 602 唯一 ID。Pydantic 后置校验 + system prompt 嵌入字典 = 0 个非法 tag_id（实测 5500 次调用 0 例外）。
2. **Cache hit 是成本的关键**：DeepSeek prompt cache 让 system prompt（~7K tokens）复用 364K+ 次。**任何修改 system prompt 的操作都会一次性 cache miss，必须在 D9 v4.0 切换时统一做**。
3. **Kimi 并发=1**：RPM 200 限速 + 启动并发风暴。绝不要把 Kimi 并发设到 >1。
4. **大批量 chunk 化**：>2000 条不要直接 `asyncio.as_completed`，会卡死（D8 教训）。用 [llm_labeler_chunked.py](research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py)。

### 评估与质量门禁

1. **9 项红线**：Week 1 Gate 必须 9/9 PASS 才解锁 Week 2。当前 D7 已 9/9 PASS（口径 B 严格人工真值）。
2. **双口径评估**：金标自动共识口径（A）会有 drop-tag artifact；严格人工真值口径（B）才是 ground truth。**Quality Gate 用口径 B**。
3. **5K 子集是全程证据基础**：覆盖率 0.00pp 偏差，所有指标可外推全量 364K。

### Phase 4 兼容契约（不可违反）

`phase5_unified_labeler.label_single_record()` 必须保持：
```python
def label_single_record(record, phase4_label_fn=None, llm_label_fn=None, persona_label_fn=None) \
    -> tuple[list[dict], list[dict], dict]:
    """Phase 4 返回 (new, all)；Phase 5 追加 meta 作第三元素。"""
```

意味着：**Phase 4 文件零改动**。`phase4_unified_labeler.py` 不动，新版只是把它作为 `phase4_label_fn` 钩子注入。这是 D7 self-test 32/32 PASS 的前提。

## 配置依赖

### LLM Keys

```
~/.paper2skills/llm_keys.json    chmod 600
```

包含：DeepSeek api_key + base_url；Kimi api_key + base_url；concurrency 配置（DeepSeek 40 / Kimi 1）。

测试连通性：
```bash
python research/02-脚本工具/07-LLM引擎/llm_client.py --smoke-test
```

### Python 依赖

继承自父项目 `paper2skills-code/requirements.txt` + 额外：
- `openai`（OpenAI SDK，DeepSeek/Kimi 都兼容）
- `pydantic` ≥ 2.0（schema 校验）
- `pyarrow`（Excel 读字典）

## 进行中的工作

> **D8 后台运行中**——**不要中途 kill 这两个进程**（除非确认要重新跑）：
>
> - `cat /tmp/d8_labeler.pid` → llm_labeler_chunked.py
> - `cat /tmp/d8_monitor.pid` → llm_labeling_monitor.py 轮询
>
> 输出落到：[phase5_full_labeled_llm.jsonl](research/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl)
>
> 监控日志：[research/04-输出结果/05-运行日志/d8_monitor_*.log](research/04-输出结果/05-运行日志/)

D8 完成后立即触发的 D9 计划：
1. 5% 开集采样 → `gap_detector.py` + `alchemist_label_generator.py`
2. LLM 辅助去重 + 业务相关性评分（≥3/5），目标 [20, 40] 个候选新标签
3. 写 [merge_phase4_phase5_llm.py](research/02-脚本工具/01-标签进化/) 做中间全量合并
4. 扩展 [dictionary_validator.py](research/02-脚本工具/01-标签进化/dictionary_validator.py) 支持 `--xlsx` 参数化 + `10_Aspect库` Sheet 校验
5. 产出 `tag_dictionary_v4.0.xlsx`

## 高频运行命令

### 重现 D2：5K LLM 闭集打标
```bash
python research/02-脚本工具/07-LLM引擎/llm_labeler.py \
  --input research/03-数据资产/test_set_5k_stratified.jsonl \
  --output research/03-数据资产/test_set_5k_p5_llm.jsonl \
  --vendor deepseek
```

### 重现 D7：unified labeler self-test + Week 1 Gate
```bash
# Self-test 32 用例
python research/02-脚本工具/01-标签进化/phase5_unified_labeler.py --self-test

# Week 1 Gate 9 项红线
python research/02-脚本工具/07-LLM引擎/quality_gate.py \
  --gate week1 \
  --pred research/03-数据资产/test_set_5k_p5_unified.jsonl \
  --golden research/03-数据资产/golden_set_human149.jsonl
```

### D8 监控（任意时刻）
```bash
python3 research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py \
  --tail research/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl \
  --window 1000 --once
```

更多见 [phase5-architecture-and-workflow-retrospective.md §六 运行手册](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md)。

## 阻塞处置矩阵

| 阻塞 | 检测方法 | 处置 |
|---|---|---|
| LLM keys 未填 | smoke-test exit=3 | 编辑 `~/.paper2skills/llm_keys.json` |
| 429 限流 | 重试 5 次仍失败 | 减并发到 20 + 等 5 分钟 |
| Connection error 集中 | monitor 滑窗 succ < 0.98 持续 5 分钟 | 等 30 分钟自愈（D8 实测） |
| Kimi 余额耗尽 | `insufficient balance` | 充值 / 切 DeepSeek-pro fallback |
| asyncio 大批量卡死 | CPU 100% / 0 TCP / 0 输出 > 5 分钟 | chunked 切 ≤ 2000/批 |
| Cache miss 飙升 | cache_hit < 85% | 检查字典版本是否变更（v3.9 → v4.0 时一次性 miss） |

## 不可做（NEVER）

- ❌ 重写 Phase 4 规则层（[phase4_unified_labeler.py](research/02-脚本工具/01-标签进化/phase4_unified_labeler.py) 等）：免费打底，重写无收益
- ❌ 替换 DeepSeek 主引擎：cache hit 98% 让成本几乎为零
- ❌ 修改 system prompt 不做 cache 影响评估：会一次性 miss
- ❌ 删除 `phase5_full_labeled_llm.jsonl` 中已有记录：D8 chunked 支持断点续跑（基于 review_id skip），删除会丢已完成进度
- ❌ 用 inline `style` 给 Mermaid 节点着色：违反 Lute 全局规范，必须 `classDef`
- ❌ 在 [phase5_unified_labeler.py](research/02-脚本工具/01-标签进化/phase5_unified_labeler.py) 改 `label_single_record` 签名：会破坏 Phase 4 兼容契约

## Phase 5 关键决策共识（不可推翻）

| # | 决策 | 不可推翻的理由 |
|---|---|---|
| D-01 | 闭集为主 + 月度开集 5% | 防标签膨胀；可治理；可对齐 BI；实测 0 非法 tag_id |
| D-02 | DeepSeek 主 + Kimi 兜底 | cache hit 让成本无关；Kimi 用于 disagreement |
| D-03 | 5K 分层抽样作为 Week 1 全程证据 | 5K 与 364K 偏差 0.00pp，是最便宜的全程证据 |
| D-04 | 双口径评估（A 严格 / B 人工） | 互相印证；root cause 透明 |
| D-05 | 统一 unified_labeler + 接口契约 | Phase 4 零回归 + L0-L3 可旁路 |
| D-06 | 9 项 Quality Gate + 7-check Schema | 防止单点过拟合 |
| D-07 | chunked LLM labeler（D8） | 87K asyncio.as_completed 瓶颈实测验证 |
| D-08 | 55 画像规则单源真值（JSON） | 已有业务校准的 unified_label_extraction.py 常量 |

详细论证见 [phase5-architecture-and-workflow-retrospective.md §五 关键决策与教训](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md)。

## 文档索引（按角色快速定位）

| 角色 | 入口 |
|---|---|
| **新人接手** | 本文 → [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) §1-§3 → [图集](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) 图 1-4 |
| **复盘传承** | [Phase 1-4 复盘](research/01-设计文档/00-Phase5-汇报与复盘/voc-tag-system-project-review-stable.md) → [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) §五 教训 |
| **外部审计** | [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) §四 + [图集](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) 图 5（9 项红线） |
| **AI 算法同行** | [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) §三 + [Skill-* 论文卡片](.) |
| **运维** | [Phase 5 主复盘](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) §六 运行手册 + 本文阻塞矩阵 |
| **每日进度** | [research/04-输出结果/03-审计报告/phase5_d{N}_progress_report.md](research/04-输出结果/03-审计报告/) |

---

> **📅 文档更新约定**：每次 Phase 推进里程碑（每个 D{N} 完成）后，更新本文「当前状态」「进行中的工作」两节即可，不需要重写其他章节。重写阈值：进入 Phase 6。

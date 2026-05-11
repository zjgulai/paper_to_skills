# CLAUDE.md — 07-NLP-VOC（VOC 标签体系子项目）

> 本文件在 Claude Code / Codex / Sisyphus 进入本目录时自动加载。它给 AI 助手提供本子项目的状态快照、关键路径、运行约束、未完成工作。

## 子项目身份

**paper2skills 项目下 07-NLP-VOC 子模块**——VOC（Voice of Customer）标签体系建设。已完成 Phase 1-7：
- Phase 1-4：基线规则打底，覆盖率 82.58%
- **Phase 5**：产品级 AI 打标闭环，5K 子集覆盖率 97.22%（D14 部分收官）
- **Phase 6**：v4.0/v4.1 字典进化 + Method C 后处理过滤（precision 0.896）+ BI 看板 C 路径上线
- **Phase 7**：Superset BI B 路径完整闭环（D1-D4，2026-05-08 → 05-11，~7h 累计开发）

数据规模：5 数据源 × **364,569 条评论**（Amazon 194K / Trustpilot 100K / Zendesk 47K / Momcozy 20K / Reddit 3K）。
基线引擎：v4.1 字典 + DeepSeek-V4-Flash 主 LLM + Kimi-K2.6 兜底。

## 当前状态（2026-05-11）

| 维度 | 值 |
|---|---|
| **当前阶段** | 🟢 Phase 7 D4 修订完成（Overview filter scope + 饼图 metric 修复） |
| Phase 5 进度 | ✅ D14 部分收官（Momus 审阅通过） |
| Phase 6 进度 | ✅ D10 BI 看板实质上线（C+A 双路径） |
| Phase 7 进度 | ✅ D4 完成，10 native filters 在 8 dashboards 上线 |
| Week 1 Gate | 🟢 9/9 PASS（D7 收口，口径 B 严格人工真值） |
| Week 2 Gate | 🟢 7/7 PASS（D9 Method C 后处理过滤） |
| LLM 三方评估 F1_weighted | 0.831 |
| Phase 6 D9 precision | 0.896（Method C 后处理过滤后） |
| Proxy NPS Cohen κ | 0.996 |
| 严格金标 Top-1 准确率 | 100%（人工 149 条） |
| 全量打标 | 364,569 条 ✅ 全部完成 |
| Superset 状态 | 🟢 Docker 运行中（localhost:8088，admin/voc_admin_2026） |

## 必读文档（接手必看）

| 优先级 | 文档 | 用途 |
|---|---|---|
| P0 | [phase5-architecture-and-workflow-retrospective.md](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) | **Phase 5 主复盘**：架构、AI 打标管道、算法演进、教训、运行手册 |
| P0 | [phase7_complete_retrospective.md](research/04-输出结果/03-审计报告/phase7_complete_retrospective.md) | **Phase 7 完整复盘**：BI B 路径从零到完整闭环 |
| P0 | [phase5_6_complete_retrospective.md](research/04-输出结果/03-审计报告/phase5_6_complete_retrospective.md) | **Phase 5+6 完整复盘**：D14 部分收官到 BI 看板上线的 17 commit 旅程 |
| P0 | [phase5-architecture-diagrams.md](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-diagrams.md) | **架构图集**：10 张 Mermaid 图 |
| P1 | [phase{5,6,7}_d{N}_progress_report.md](research/04-输出结果/03-审计报告/) | 每日进度报告（D1-D14 + Phase 6 D1-D10 + Phase 7 D1-D4） |
| P1 | [voc-tag-system-project-review-stable.md](research/01-设计文档/00-Phase5-汇报与复盘/voc-tag-system-project-review-stable.md) | Phase 1-4 复盘（基线 82.58% 怎么来的） |
| P2 | [README.md](research/README.md) | research/ 目录结构索引 |

## 目录约定

```
07-NLP-VOC/
├── CLAUDE.md                        ← 本文件
├── README.md                        ← 3 分钟入口
├── 00-知识库-Skill卡片/             40+ 张论文卡片
├── papers/                          下载的 ArXiv PDF
└── research/
    ├── 00-归档资料/                  历史归档
    │   ├── phase4_archive/           Phase 1-4 旧产物（13 个大 jsonl 已删，仅保 audit.json）
    │   ├── phase6_html_dashboard/    ⭐ Phase 6 D10 HTML 看板（C 路径，被 Phase 7 替代）
    │   ├── weekly_drafts/            ⭐ W19 / W19-v41 早期周报草稿
    │   └── labeling-outputs/         历史打标输出
    ├── 01-设计文档/
    │   ├── 00-Phase5-汇报与复盘/     ⭐ Phase 5 核心产出
    │   ├── 02-工作流设计/
    │   │   └── persona_tags_55.json  ← labeler 直读
    │   └── 08-Phase计划/             Phase 5 / 6 / 7 计划
    ├── 02-脚本工具/
    │   ├── 01-标签进化/              L0/L2/L3 + unified labeler + Superset factories
    │   │   ├── docker/               ⭐ Phase 7 新增：Superset Docker + factories
    │   │   │   ├── docker-compose.yml
    │   │   │   ├── superset_bootstrap.py
    │   │   │   ├── superset_charts_factory.py
    │   │   │   ├── superset_filters_factory.py
    │   │   │   └── superset_config.py
    │   │   └── sql/
    │   │       ├── voc_bi_schema.sql
    │   │       └── voc_bi_views.sql  ← 6 视图定义
    │   ├── 02-数据采样/
    │   ├── 04-数据处理/
    │   ├── 05-NPS管道/
    │   ├── 06-诊断工具/              schema validator / monitor / dual_coverage / spot_check
    │   └── 07-LLM引擎/               ⭐ Phase 5 核心新增（D1-D8 全部 LLM 工具）
    ├── 03-数据资产/                  中间产物（jsonl，大文件 gitignore 排除）
    └── 04-输出结果/
        ├── 01-字典版本/              v3.5 → v4.1（v4.1 当前生产版本）
        ├── 03-审计报告/              ⭐ 118 个文件（含 INDEX）
        ├── 10-周报/2026-W19-d9/      ⭐ Phase 6 D9 7 部门 × AGRS+MAA = 28 文件
        ├── 11-BI看板/                ⭐ Phase 7 新增：superset_exports/ 8 ZIP
        └── unified_labeling/         ⭐ 当前主源：phase6_d9_filtered.jsonl
```

## 关键脚本速查（Phase 5-7 全集）

### 07-LLM 引擎（Phase 5 D1-D8 全部）

| 脚本 | 引入 | 作用 |
|---|---|---|
| [llm_client.py](research/02-脚本工具/07-LLM引擎/llm_client.py) | D1 | DeepSeek + Kimi 双引擎封装，指数退避、并发信号量、smoke-test |
| [tag_dict_loader.py](research/02-脚本工具/07-LLM引擎/tag_dict_loader.py) | D2 | 字典紧凑化（v3.9 → v4.1），lru_cache 复用 |
| [llm_labeler.py](research/02-脚本工具/07-LLM引擎/llm_labeler.py) | D2 | LLM 闭集多标签打标 |
| [llm_labeler_chunked.py](research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py) | **D8** | 大批量 chunk 化跑 87K（解 asyncio 瓶颈） |
| [stratified_sampler.py](research/02-脚本工具/07-LLM引擎/stratified_sampler.py) | D1 | 5K 分层抽样 |
| [golden_set_sampler.py](research/02-脚本工具/07-LLM引擎/golden_set_sampler.py) | D3 | 500 条金标抽样 |
| [consensus_prefill.py](research/02-脚本工具/07-LLM引擎/consensus_prefill.py) | D3 | 双 LLM 共识合并 |
| [human_annotation_cli.py](research/02-脚本工具/07-LLM引擎/human_annotation_cli.py) | D3 | 人工仲裁 CLI |
| [evaluation_suite.py](research/02-脚本工具/07-LLM引擎/evaluation_suite.py) | D3 | 三方评估（P4 vs DS vs 金标） |
| [llm_consensus.py](research/02-脚本工具/07-LLM引擎/llm_consensus.py) | D4 | DS + Kimi 共识 |
| [quality_gate.py](research/02-脚本工具/07-LLM引擎/quality_gate.py) | D5 | 9 项红线 + Week 2 Gate |

### 标签进化 / 画像 / NPS

| 脚本 | 引入 | 作用 |
|---|---|---|
| [absa_extractor.py](research/02-脚本工具/01-标签进化/absa_extractor.py) | Phase 5 D4 | LLM 抽 ABSA 三元组 |
| [persona_tag_labeler.py](research/02-脚本工具/01-标签进化/persona_tag_labeler.py) | Phase 5 D6 | 55 原子画像规则匹配 |
| [phase5_unified_labeler.py](research/02-脚本工具/01-标签进化/phase5_unified_labeler.py) | Phase 5 D7 | merge / stream / self-test 统一打标器 |
| [merge_phase4_phase5_llm.py](research/02-脚本工具/01-标签进化/merge_phase4_phase5_llm.py) | Phase 5 D7 | 输出 phase5_intermediate_merged.jsonl |
| [confidence_rebalancer.py](research/02-脚本工具/01-标签进化/confidence_rebalancer.py) | Phase 6 D3 | F5 离线 confidence 重赋 |
| [merge_multilingual_labels.py](research/02-脚本工具/01-标签进化/merge_multilingual_labels.py) | Phase 6 D4 | F4 多语言重打合并 |
| [label_filter_kimi.py](research/02-脚本工具/01-标签进化/label_filter_kimi.py) | **Phase 6 D9** | Method C 后处理过滤（precision 0.896） |
| [agrs_summarizer.py](research/02-脚本工具/01-标签进化/agrs_summarizer.py) | Phase 5 D11 | AGRS 评论摘要（周报输入） |
| [maa_strategy_generator.py](research/02-脚本工具/01-标签进化/maa_strategy_generator.py) | Phase 5 D11 | MAA 行动建议（周报输入） |
| [proxy_nps_labeler.py](research/02-脚本工具/05-NPS管道/proxy_nps_labeler.py) | Phase 5 D5 | NPS 三法投票 |

### 诊断工具（Phase 5-7）

| 脚本 | 引入 | 作用 |
|---|---|---|
| [phase5_schema_validator.py](research/02-脚本工具/06-诊断工具/phase5_schema_validator.py) | Phase 5 D7 | 7-check schema |
| [llm_labeling_monitor.py](research/02-脚本工具/06-诊断工具/llm_labeling_monitor.py) | Phase 5 D8 | 实时滑窗监控三红线 |
| [dual_coverage_calculator.py](research/02-脚本工具/06-诊断工具/dual_coverage_calculator.py) | **Phase 5 D10** | 双覆盖率（原始 + 业务有效） |
| [llm_output_spot_check.py](research/02-脚本工具/06-诊断工具/llm_output_spot_check.py) | Phase 6 D7 | LLM 输出抽样质量评估 |
| [bi_spec_validator.py](research/02-脚本工具/06-诊断工具/bi_spec_validator.py) | Phase 6 | BI 看板规格校验 |

### Phase 6 BI 看板（C 路径）

| 脚本 | 作用 |
|---|---|
| [bi_dashboard_generator.py](research/02-脚本工具/01-标签进化/bi_dashboard_generator.py) | 离线渲染单文件 HTML 看板（被 Phase 7 替代，输出移至归档） |

### Phase 7 Superset BI（B 路径，⭐ 当前生产）

| 脚本 | 作用 |
|---|---|
| [docker-compose.yml](research/02-脚本工具/01-标签进化/docker/docker-compose.yml) | Superset 4.1.1 Docker 编排 |
| [superset_bootstrap.py](research/02-脚本工具/01-标签进化/docker/superset_bootstrap.py) | voc_bi 数据库注册 + 6 datasets |
| [superset_charts_factory.py](research/02-脚本工具/01-标签进化/docker/superset_charts_factory.py) | 12 charts + 8 dashboards 自动化（idempotent by name） |
| [superset_filters_factory.py](research/02-脚本工具/01-标签进化/docker/superset_filters_factory.py) | 10 native filters 自动化（按 chart_name 查找 ID） |
| [etl_to_postgres.py](research/02-脚本工具/01-标签进化/etl_to_postgres.py) | phase6_d9_filtered.jsonl → voc_bi（37s 导入 364K） |

## 工作流约束（**重要**）

### LLM 引擎使用规则

1. **闭集严格性**：LLM 输出 tag_id ∈ v4.1 字典闭集，Pydantic 后置校验。
2. **Cache hit 是成本关键**：DeepSeek prompt cache 让 system prompt 复用。**修改 system prompt 须在字典升级时统一做**。
3. **Kimi 并发=1**：RPM 200 限速。绝不要把 Kimi 并发设到 >1。
4. **大批量 chunk 化**：>2000 条不要直接 `asyncio.as_completed`，用 [llm_labeler_chunked.py](research/02-脚本工具/07-LLM引擎/llm_labeler_chunked.py)。

### 评估与质量门禁

1. **Week 1 Gate 9 项红线**：D7 已 9/9 PASS（口径 B）。
2. **Week 2 Gate 7 项红线**：D9 已 7/7 PASS（Method C 后处理过滤后）。
3. **双口径评估**：金标自动共识（A）会有 drop-tag artifact；严格人工真值（B）才是 ground truth。
4. **5K 子集是全程证据基础**：覆盖率 0.00pp 偏差，可外推全量 364K。

### Phase 4 兼容契约（不可违反）

`phase5_unified_labeler.label_single_record()` 必须保持：
```python
def label_single_record(record, phase4_label_fn=None, llm_label_fn=None, persona_label_fn=None) \
    -> tuple[list[dict], list[dict], dict]:
    """Phase 4 返回 (new, all)；Phase 5 追加 meta 作第三元素。"""
```

**Phase 4 文件零改动**。`phase4_unified_labeler.py` 不动，新版仅作为 `phase4_label_fn` 钩子注入。这是 D7 self-test 32/32 PASS 的前提。

### Superset BI 约束（Phase 7 新增）

1. **`voc_bi` Postgres + 6 视图不可重命名**：所有 Superset dataset / chart / dashboard 依赖这些 SQL 视图。
2. **Charts/filters factory idempotent by name**：删除重建 chart 后 ID 会变（如 D4 修复时 chart 4,5 → 13,14），factory 已改为按 `slice_name` 查找，不依赖硬编码 ID。
3. **饼图 metric 用 `COUNT(*)` SQL 形式**：v_review_overview 是聚合视图，不暴露 review_id 列，必须用 `{"expressionType": "SQL", "sqlExpression": "COUNT(*)"}`，不能用 `COUNT(review_id)`。
4. **dashboard ZIP 是真正的 source of truth**：[research/04-输出结果/11-BI看板/superset_exports/](research/04-输出结果/11-BI看板/superset_exports/)，迁移环境只需重导。
5. **已知 bug**：Superset 4.1.1 dashboard-mode 下饼图 pie+SQL-form metric 不渲染（`/explore/` 单图模式正常）。修复方案待 Phase 8。

## 配置依赖

### LLM Keys

```
~/.paper2skills/llm_keys.json    chmod 600
```

包含：DeepSeek api_key + base_url；Kimi api_key + base_url；concurrency 配置（DeepSeek 40 / Kimi 1）。

测试：
```bash
python research/02-脚本工具/07-LLM引擎/llm_client.py --smoke-test
```

### Superset Docker

```bash
cd research/02-脚本工具/01-标签进化/docker/
docker compose up -d
curl -sS http://localhost:8088/health   # 期望 200
```

用户：admin / voc_admin_2026

### Python 依赖

继承 `paper2skills-code/requirements.txt` + 额外：
- `openai`（DeepSeek/Kimi 都兼容）
- `pydantic` ≥ 2.0
- `pyarrow`（Excel 读字典）
- Docker / docker-compose（Superset）

## 当前数据资产（Phase 5-7 末态）

### unified_labeling/（active jsonl，必须保留）

| 文件 | 大小 | 用途 |
|---|---|---|
| `phase6_d9_filtered.jsonl` | 560M | ⭐ Phase 7 ETL 主源（Method C 后处理过滤产物） |
| `phase5_intermediate_merged.jsonl` | 500M | symlink 目标 + 6 个脚本默认输入 |
| `phase5_full_labeled.jsonl` | symlink | → phase5_intermediate_merged.jsonl |
| `phase6_v41_rebalanced.jsonl` | 553M | confidence_rebalancer + merge_multilingual_labels 输入 |
| `phase6_d5_final.jsonl` | 561M | label_filter_kimi 输入 |
| `phase5_full_labeled_llm.jsonl` | 38M | merge_phase4_phase5_llm 输入 |
| `phase6_{amazon,multilingual,zendesk}_relabel.jsonl` | ~4-5M | F3/F4 重打产物 |

### 已删除（2026-05-11 清理，回收 7.1G）

- `phase4_archive/*.jsonl`（13 个 Phase 1-4 旧产物，audit.json 保留作摘要）
- `unified_labeling/phase5_full_persona / phase6_d4_merged / phase6_d5_intermediate / phase6_d8_*`（9 个无引用中间产物）

## 高频运行命令

### 重新跑 ETL → Superset

```bash
# 1. ETL 导入（37s 导入 364K reviews）
python research/02-脚本工具/01-标签进化/etl_to_postgres.py \
  --input research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl \
  --dict research/04-输出结果/01-字典版本/tag_dictionary_v4.1.xlsx

# 2. 重建 Superset 看板
cd research/02-脚本工具/01-标签进化/docker
python3 superset_bootstrap.py
python3 superset_charts_factory.py
python3 superset_filters_factory.py
```

### 重现 D7：Week 1 Gate

```bash
python research/02-脚本工具/01-标签进化/phase5_unified_labeler.py --self-test
python research/02-脚本工具/07-LLM引擎/quality_gate.py --gate week1 \
  --pred research/03-数据资产/test_set_5k_p5_unified.jsonl \
  --golden research/03-数据资产/golden_set_human149.jsonl
```

### 重现 D9：Week 2 Gate（Method C）

```bash
python research/02-脚本工具/01-标签进化/label_filter_kimi.py \
  --input research/04-输出结果/unified_labeling/phase6_d5_final.jsonl \
  --output research/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl
python research/02-脚本工具/07-LLM引擎/quality_gate.py --gate week2 \
  --pred phase6_d9_filtered.jsonl --golden golden_set_human149.jsonl
```

更多见 [phase5-architecture-and-workflow-retrospective.md §六 运行手册](research/01-设计文档/00-Phase5-汇报与复盘/phase5-architecture-and-workflow-retrospective.md) 和 [phase7_complete_retrospective.md](research/04-输出结果/03-审计报告/phase7_complete_retrospective.md)。

## 阻塞处置矩阵

| 阻塞 | 检测方法 | 处置 |
|---|---|---|
| LLM keys 未填 | smoke-test exit=3 | 编辑 `~/.paper2skills/llm_keys.json` |
| 429 限流 | 重试 5 次仍失败 | 减并发到 20 + 等 5 分钟 |
| Connection error 集中 | monitor 滑窗 succ < 0.98 持续 5 分钟 | 等 30 分钟自愈（D8 实测） |
| Kimi 余额耗尽 | `insufficient balance` | 充值 / 切 DeepSeek-pro fallback |
| asyncio 大批量卡死 | CPU 100% / 0 输出 > 5 分钟 | chunked 切 ≤ 2000/批 |
| Cache miss 飙升 | cache_hit < 85% | 检查字典版本是否变更 |
| Superset 启动失败 | `docker ps` 无 voc_superset | `docker compose up -d` + 等 30s |
| Superset chart 不渲染 | dashboard 空白 / canvas 0px | 先看 `/explore/?slice_id=N` 单图模式；饼图 D3 遗留 bug 待修 |

## 不可做（NEVER）

- ❌ 重写 Phase 4 规则层（[phase4_unified_labeler.py](research/02-脚本工具/01-标签进化/phase4_unified_labeler.py)）：免费打底，无收益
- ❌ 替换 DeepSeek 主引擎：cache hit 98% 让成本几乎为零
- ❌ 修改 system prompt 不做 cache 影响评估：会一次性 miss
- ❌ 删除 `phase6_d9_filtered.jsonl`：Phase 7 ETL 主源，下游 Superset 全依赖
- ❌ 删除 `phase5_intermediate_merged.jsonl`：6 个脚本默认输入 + symlink 目标
- ❌ 用 inline `style` 给 Mermaid 节点着色：违反 Lute 全局规范，必须 `classDef`
- ❌ 改 `phase5_unified_labeler.label_single_record` 签名：会破坏 Phase 4 兼容契约
- ❌ 修改 Superset charts factory 时硬编码 chart_id：用 `slice_name` 查找
- ❌ 给饼图 metric 用 `COUNT(review_id)`：v_review_overview 不暴露 review_id，必须 `COUNT(*)` SQL 形式

## Phase 5-7 关键决策共识（不可推翻）

| # | 决策 | 不可推翻的理由 |
|---|---|---|
| D-01 | 闭集为主 + 月度开集 5% | 防标签膨胀；实测 0 非法 tag_id |
| D-02 | DeepSeek 主 + Kimi 兜底 | cache hit 让成本无关 |
| D-03 | 5K 分层抽样作为全程证据 | 与 364K 偏差 0.00pp |
| D-04 | 双口径评估（A 严格 / B 人工） | 互相印证；root cause 透明 |
| D-05 | 统一 unified_labeler + 接口契约 | Phase 4 零回归 + L0-L3 可旁路 |
| D-06 | 9 项 Quality Gate + 7-check Schema | 防止单点过拟合 |
| D-07 | chunked LLM labeler | 87K asyncio.as_completed 瓶颈实测验证 |
| D-08 | 55 画像规则单源真值（JSON） | 已有业务校准 |
| **D-09** | **Method C 后处理过滤**（Phase 6 D9） | precision 0.639 → 0.896，Week 2 Gate 7/7 |
| **D-10** | **Superset B 路径作为主 BI** | 实时交互 > 静态 HTML；Phase 7 完整闭环 |
| **D-11** | **Charts/filters factory by name** | 跨 delete+recreate 鲁棒（D4 实测验证） |

## 文档更新约定

每次 Phase 推进里程碑（每个 D{N} 完成）后，更新本文「当前状态」和「数据资产末态」即可。重写阈值：进入 Phase 8。

---

> **最近更新**：2026-05-11 — Phase 7 D4 完成 + 项目清理（回收 7.1G）+ 重写 CLAUDE.md 涵盖 Phase 6/7。

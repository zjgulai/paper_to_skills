# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**paper2skills** - A system that converts academic papers into actionable business decision skill cards, focused on cross-border e-commerce for mother & baby products (母婴出海跨境电商).

The workflow transforms academic research (primarily from ArXiv) into practical business skills through a 4-step pipeline: paper selection → extraction → review → sync.

> **2026-09 升级中**:检索层正在从「仅 arXiv」扩展为「arXiv + Crossref 顶刊 + 会议 proceedings」多源流水线,
> 并补上可复现评分、三层去重、双门禁(K1 代码可执行 / K2 事实可溯源)。
> 设计见 `paper2skills-research/reports/paper2skills-萃取链路升级方案-v2.md`,
> 实施清单见 `paper2skills-research/reports/实施方案与TODO.md`,
> 论文唯一事实源为 `paper2skills-vault/07-资源库/papers_registry.json`。

## Project Structure

```
├── paper2skills-skills/     # Claude Code skills for the workflow
│   ├── paper-workflow/      # Orchestrates the complete workflow
│   ├── paper-选题/           # Step 1: Paper selection from ArXiv/GitHub
│   ├── paper-萃取/           # Step 2: Extract papers into Skill cards
│   ├── paper-审核/           # Step 3: Quality review
│   └── paper-同步/           # Step 4: Sync to multiple platforms
├── paper2skills-research/   # 调研与流水线工程工作区(脚本 + 中间数据 + 报告)
│   ├── scripts/             # arxiv_harvest / journal_harvest / rank_candidates /
│   │                        # skill_audit / dedup_check / build_registry
│   ├── data/                # 候选池 JSON/CSV、领域 bundle、体检与去重报告
│   └── reports/             # 方案、推荐清单、TODO
├── paper2skills-vault/      # Knowledge base (Obsidian-compatible)
│   ├── 01-因果推断/          # Causal inference skills
│   ├── 02-A_B实验/           # A/B testing skills
│   ├── 03-时间序列/          # Time series skills
│   ├── 04-供应链/            # Supply chain skills
│   ├── 05-推荐系统/          # Recommendation system skills
│   ├── 06-增长模型/          # Growth model skills
│   ├── 08-知识图谱/          # Knowledge graph / GNN skills
│   ├── 09-DataAgent-LLM/     # DataAgent & LLM-powered analytics
│   ├── 10-MAS/               # Multi-Agent System skills
│   ├── 11-AI人文/             # AI × Humanities: healing quotes, philosophical analogies
│   ├── 12-ML基础/             # Machine learning fundamentals
│   ├── 13-广告分析/            # Ad attribution, ROAS optimization
│   ├── 14-用户分析/            # Funnel, cohort, RFM analysis
│   ├── 15-营销投放分析/         # MMM, promotion effectiveness
│   ├── 16-智能体工程/           # LLM Agent Engineering: Skills, Context, MCP/A2A
│   ├── 07-资源库/            # Master Prompt, keywords, sync status
│   └── papers/               # Downloaded papers by domain
└── paper2skills-code/       # Python code templates
    ├── causal_inference/
    ├── ab_testing/
    ├── time_series/
    ├── supply_chain/
    ├── recommendation/
    ├── growth_model/
    ├── nlp_voc/              # 保留:VOC 子项目已迁至 ../ai_nlp_voc/,本目录为代码模板镜像
    ├── knowledge_graph/
    ├── data_agent_llm/
    ├── mas/
    └── llm_agent_engineering/

  说明:11-AI人文 / 12-ML基础 / 13-广告分析 / 14-用户分析 / 15-营销投放分析 五个新业务领域
  目前仅有 vault Skill 卡片,code/ 侧尚未落地子模块,需要时按 Python 包命名规范(英文 snake_case)新建。
```

## NLP-VOC 子项目迁出说明

`07-NLP-VOC` 子项目已于 `2026-05-17 commit 47b1dbf` 独立迁出至 `/Users/pray/project/ai_nlp_voc/`。
本仓库**有意保留**两处残留作为"代码模板 + 论文档案"镜像,而非未清理的脏数据:

- `paper2skills-code/nlp_voc/` (43 子模块) — 历史代码模板,内部 `data_path` 已硬编码改为
  指向 `../ai_nlp_voc/...` 并加 `try/except`,可被新仓库直接 import 复用,不在本项目运行。
- `paper2skills-vault/papers/nlp_voc/` — 原始论文 PDF 档案,留作 paper-skills-graph 选题与
  citation 追踪使用,不再产生新 Skill 卡片。

回退手段:`git reset --hard backup/before-voc-extract-20260517`。

## Key Files

| File | Purpose |
|------|---------|
| `paper2skills-vault/07-资源库/MasterPrompt-v2.md` | **当前生效的 Master Prompt**(含 R1–R5 证据规则、frontmatter v2、硬拦截清单) |
| `paper2skills-vault/07-资源库/MasterPrompt.md` | v1 原文(保留供对照,新卡片一律走 v2) |
| `paper2skills-vault/07-资源库/关键词库.md` | ArXiv search keywords by domain (v1) |
| `paper2skills-vault/07-资源库/venue-whitelist.md` | **venue 白名单 + 降级判定 + 时效基准规则** |
| `paper2skills-vault/07-资源库/papers_registry.json` | **论文唯一事实源**(候选/评分/决策/门禁/交付物回指) |
| `paper2skills-vault/07-资源库/sync_status.json` | Tracks sync status across platforms |
| `paper2skills-vault/07-资源库/gates/gate_g2.json` | **K2 事实可溯源门禁产物**(G1 见 `paper2skills-research/data/verification/`) |
| `paper2skills-skills/paper-萃取/scripts/verify_skill_code.py` | **K1 门禁**:五级代码可执行性验证(语法/编译/导入/运行/断言) |
| `paper2skills-skills/paper-审核/scripts/gate_check.py` | **K2 门禁**:G1 代码 / G2 事实 / G3 业务 三合一 |
| `paper2skills-skills/paper-同步/scripts/sync.py` | Sync script for vault/GitHub/feishu |
| `paper2skills-research/scripts/` | 检索/评分/去重/体检脚本(见下方"检索路线") |

## 门禁体系(K1 / K2,2026-09-12 建立)

**背景**:存量审核靠 LLM 打 7/10 分放行。而 `AutoReproduce`(arXiv:2505.20662) 实测:
LLM 评审认为"很好"的生成代码,**执行率仅 17.94%**;加上执行闭环后才到 94.87%。
`ResearchCodeBench` 附录 G 进一步显示新代码失败中 **58.6% 是语义错误**(能跑但算错)。
→ 故门禁必须产出**可复核的退出码与 stdout**,而不是分数。

### K1 代码可执行 · `paper-萃取/scripts/verify_skill_code.py`

| 级别 | 检查 | 说明 |
|------|------|------|
| L1 | `ast.parse` | 语法 |
| L2 | `py_compile` | 编译 |
| L3 | import 探针 | **缺第三方依赖注入 stub,归因环境;仓库内不存在的本地模块判 ORPHAN_DEP(卡片缺陷)** |
| L4 | 作为脚本执行 | 超时保护 + 独立进程组 SIGKILL |
| L5 | `pytest` | 断言是否真的成立 |

判定:`PASS` / `ENV_BLOCKED`(缺依赖,计入未验证分母) / `ORPHAN_DEP`(卡片引用了不存在的模块) / `FAIL`。

```bash
# 全量(卡片级:自动把卡片内所有 python 块按文档顺序拼成一个模块)
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --all \
  --level 5 --timeout 30 \
  --json-out paper2skills-research/data/verification/k1_l5.json

# 单卡 / 快速语法扫描(秒级)
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <card.md>
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --all --level 2
```

**两条必须遵守的实现约束**(都是实测踩出来的):
1. **必须按卡片拼接验证,不能逐块独立导入** — 卡片普遍是「block1 定义类 → block3 使用」的递进结构,
   逐块导入会产生大量 `NameError` **假阳性**(首次实测 44 个 L3 失败里 19 个是假的)。
   需要逐块定位时用 `--per-block`。
2. **验证必须在断网语义下运行** — 卡片里的 `from_pretrained(...)`/`requests.get(...)` 会挂在网络重试上
   (实测:跑 30 分钟未出结果,而进程 CPU 时间仅 2.2 秒)。脚本已内置 socket 拦截 +
   `HF_HUB_OFFLINE=1` 等环境变量,把网络调用变成即时失败并归因为 `ENV_BLOCKED`。

**首次全量基线(2026-09-12)**:80 张含代码卡片 → **K1 执行率 52.5%**
(PASS 42 / ENV_BLOCKED 6 / ORPHAN_DEP 9 / FAIL 23)。对照 PaperCoder 17.94%、AutoReproduce 94.87%。

### K2 三合一门禁 · `paper-审核/scripts/gate_check.py`

| 门禁 | 检查 | 关键口径 |
|------|------|----------|
| **G1 代码可执行** | 读 K1 产物 | 无 K1 凭证一律判红(禁止凭人工判断放行) |
| **G2 事实可溯源** | 带「度量语义」的数字是否能在证据链中找到出处 | 证据链 = 卡片内 `> 原文:"..."` 引用块 + 同目录 `evidence.md`。**只回答"有没有出处",不判断"对不对"**(对错靠人工抽检) |
| **G3 业务可落地** | 场景是否具体、是否声明数据可得性、ROI 是否有依据、是否关联 ≥2 张卡 | 空泛表述黑名单 + 母婴出海具体信号计数 |

```bash
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --all \
  --k1 paper2skills-research/data/verification/k1_l5.json \
  --outdir paper2skills-vault/07-资源库/gates
```

**首次全量基线(2026-09-12,G2)**:130 张卡 → **57 张通过(43.8%)**,红灯 438 条。
根因:全库 **13,868 个数字 vs 仅 3 行** `> 原文:"..."` 引用块。

> ⚠️ **口径警告**:三个门禁必须**分开报**。`Cited but Not Verified`(arXiv:2605.06635) 实测
> 链接可用 >94%、主题相关 >80%,而**事实一致性只有 39–77%** —— 合成一个「可信度总分」
> 会把最弱的那一维平均掉。

## 检索路线(2026-09 实测结论)

| 路线 | 覆盖 | 调用 | 实测 |
|------|------|------|------|
| arXiv API | 预印本、方法论、模型类 | `https://export.arxiv.org/api/query` + `submittedDate:[start TO end]` | ✅ 49 查询 → 1046 篇(92 天)。**必须 https + `curl -L`**(http 返回 301 空 body) |
| Crossref 期刊 | UTD24/FT50/CCF 期刊**正式在线发表** | `api.crossref.org/journals/{issn}/works?filter=from-pub-date:..,until-pub-date:..` | ✅ 28 刊 → 1751 篇 |
| Crossref 会议录 | KDD/SIGIR/WWW/WSDM 论文集 | `api.crossref.org/works?query.container-title={proceedings 全名}` | ✅ **dl.acm.org 403 的唯一替代**,含 `event.location` |
| PMLR 卷级 BibTeX | UAI/AISTATS/MLSys | `proceedings.mlr.press/v{N}/assets/bib/bibliography.bib` | ✅ **一次调用全量**(UAI v337 = 639KB,330 篇) |
| ACL Anthology | ACL 2026 全 7 卷 | `aclanthology.org/volumes/2026.acl-long.bib` | ✅ 但 >60s,须 `--max-time 180`;**无 JSON 端点** |
| OpenAlex | 跨源补充与交叉校验 | `api.openalex.org/works?filter=primary_location.source.issn:{issn},...` | ✅ 免鉴权;⚠️ **按上线日索引,与 Crossref 封面日语义不同** |
| **OpenReview** | DBLP 镜像 + 已回填 venue 的会议 | ✅ `/notes/search?term=..&limit=1000&type=terms` 与 `/groups?id=..` 实测 **200**;❌ `/notes` 仍 403 | ⚠️ **可用但覆盖易高估**(见下) |
| **DBLP** | — | — | ⛔ **彻底不可用**(见下) |
| Semantic Scholar | — | — | ❌ 429 |

**四个必须记住的坑**:
1. **顶刊"新发表" ≠ 新方法** — 实测 79% 的顶刊文章 DOI 年份段比发表年份早 ≥2 年
   (例:`10.1287/mnsc.2022.02462` 发表于 2026-07-09 但 DOI 为 2022)。萃取时应优先找预印本版本。
2. **Crossref 多关键词抓取会跨组重复计数** — 同一批 120 条记录实际只有 91 篇唯一论文(重复率 24%),
   排序前必须先按 DOI 去重。
3. **Crossref 必须同时打 `from-pub-date` 与 `from-print-pub-date`** — SAGE 系(JM/JMR/POM)的
   `published` 是 online-first 日期,只打前者会漏掉整期。实测 POM:普通过滤 55 条且几乎全无卷期,
   print-date 过滤给出干净的 `{35(7):16, 35(8):16, 35(9):17}`;ISR 的 print-date 过滤结果为 **0**
   (证明其本季纯 Articles in Advance,无正式期号)。
4. **DBLP 不是"限速问题"** — `dblp.org`/`dblp.uni-trier.de`/`dblp.dagstuhl.de` 三镜像全部返回
   **HTTP 200 + Anubis proof-of-work 挑战页**(`<title>Making sure you're not a bot!</title>`)。
   换 UA、降速、换镜像、换 API 路径**全部无效** → **脚本路线应彻底放弃**。
   最危险之处是状态码为 200,脚本若不检查正文会把挑战页当数据解析。
   替代:Crossref(ACM DL DOI) + OpenAlex。

**OpenReview 的精确边界(勿高估)**:`/notes/search` 支持精确短语匹配 venue 且 `limit=1000` 一次全量,
但实测 `"SIGIR 2026"` 返回的 123 条里 117 条 invitation=`Record` 且带 `dblp:` externalId,
**是 DBLP 镜像,不等于正会**(SIGIR 正会 274 篇未覆盖);`"ICML 2026"`/`"ICLR 2026"`/`"ACL 2026"`/
`"AAAI 2026"` 精确短语查询返回 **0**(venue 字段未回填);`"EMNLP 2026"` 返回的 147 条实为
workshop 名(GroundLM/NLP4PI)。**正会枚举仍需个人 token**。

**`NAACL 2026` 不存在**(2026 停办一届):`2026.naacl.org` DNS 不解析、
`aclanthology.org/events/naacl-2026/` 404、`naacl.org` 只挂 2027。
venue 白名单中不得出现该项。完整规则见 `paper2skills-vault/07-资源库/venue-whitelist.md`。

## 存量资产体检(2026-09-12 实测)

> 本节数字为 T0-2 去重**之后**的实测值;括号内为去重前的旧值,保留用于对照。

| 指标 | 实测值 | 说明 |
|------|--------|------|
| 唯一 Skill 卡片 | **130 张** | 去重前文件数 156,`07-NLP-VOC/` 下 26 组同名重复已清理(25 组字节相同直接删、1 组已漂移移入 `_superseded/`) |
| 含 frontmatter | **73/130 (56%)** | 规范不统一 |
| 含 `paper:` 溯源字段 | **6/130 (4.6%)** | 无法反查论文来源。⚠️ 注意:frontmatter 里的 `source:`(值多为 `human+ai`)是**文档来源**,**不是**论文来源,统计时勿混淆 |
| 含 python 代码块 | **80/130 (62%)** / 共 104 个代码块 | — |
| **K1 代码执行率** | **52.5%** | PASS 42 / ENV_BLOCKED 6 / ORPHAN_DEP 9 / FAIL 23。详见下方"门禁体系" |
| **G2 事实溯源通过率** | **43.8%** | 红灯 438 条;根因是全库 13,868 个数字 vs 仅 3 行原文引用块 |
| 领域分布 | 07-NLP-VOC 41 / 16-智能体工程 16 / 10-MAS 12 / 06-增长模型 10 / 08-知识图谱 9 … | 11/12 域各仅 1 张 |

### 已发现的代码硬缺陷(由 K1 首次暴露,均为真缺陷)

8 个代码块连 `ast.parse` 都过不了,例如:
`Skill-AB-Experimental-Design`(中文书名号 `【` 混入代码)、
`Skill-MAS-Orchestrator`(箭头 `→` 混入代码)、
`Skill-Argos-Agentic-Anomaly-Detection`(三引号 f-string 未闭合)、
`Skill-Memory-as-Action` / `Skill-Skill-Lifecycle-Design`(首行非 Python)。
另有 9 张卡片 `import` 了仓库内**根本不存在**的本地模块
(`autotag_self_evolving`、`review_quality_scoring`、`nps_driver_analysis` 等)。

> 这些缺陷此前从未被发现,原因是审核依赖 LLM 阅读而非执行。

## Workflow Commands

### Run Complete Workflow

Use the `paper-workflow` skill to run the complete pipeline:

```bash
# Trigger via natural language to Claude
"Run paper2skills workflow"
"Process this paper through the complete pipeline"
```

### Individual Steps

Each step can be triggered separately via skills:

1. **Paper Selection** (`paper-选题`): "筛选论文" / "Search ArXiv for uplift modeling papers"
2. **Extraction** (`paper-萃取`): "萃取论文" / "Generate skill card from this paper"
3. **Review** (`paper-审核`): "审核 skill" / "Review quality of Skill-Uplift-Modeling"
4. **Sync** (`paper-同步`): Use the sync script directly (see below)

### Sync Script Usage

```bash
# Sync a skill to vault and GitHub
cd paper2skills-skills/paper-同步
python scripts/sync.py --skill Skill-Uplift-Modeling

# Sync to specific targets
python scripts/sync.py --skill Skill-Uplift-Modeling --target vault,github

# View sync status
python scripts/sync.py --skill Skill-Uplift-Modeling --status

# Sync all tracked skills
python scripts/sync.py --status
```

## Skill Card Format

**新卡片一律走 `MasterPrompt-v2.md`**,其结构为 6 段 + frontmatter v2:

0. **frontmatter** — 含 `paper_id` / `paper` / `venue` / `venue_tier` / `evidence_grade` / `related`(缺字段即未完成)
1. **算法原理**(≤300 字)— 核心思想 / 数学直觉 / 关键假设
1b. **反例与适用边界** — 什么时候不要用 / 已知失败模式 / 论文自承局限(无则显式写"论文未讨论")
2. **母婴出海应用案例**(1-2 个)— 业务问题 / 数据要求 / **数据可得性(必填)** / 预期产出 / 业务价值
3. **代码模板** — 可运行 Python + 测试;须能过 K1(不得 import 仓库内不存在的模块、不得有 `plt.show()`/网络请求、须含 assert)
4. **技能关联** — 必须真的引用 `Skill-*.md` 文件名,≥2 个
5. **商业价值评估** — ROI 须给公式或参数来源 + 难度/优先级星级
6. **原文引用(必填,≥3 条)** — `> 原文:"<逐字摘录>"` + 出处(arXiv ID / 章节 / 页码)

> v1 的五段式结构(`MasterPrompt.md`)仅保留供对照。v2 新增的第 0/1b/6 段
> 分别对应 G2 溯源门禁、边界声明要求、证据链要求。

## Code Standards

### Python Code Template Structure

```python
# Each module should have:
# - model.py: Core algorithm implementation
# - __init__.py: Module exports
# - Example data generation functions
# - Business-specific scenario code

# Example: paper2skills-code/causal_inference/uplift_model/model.py
class UpliftModel:
    """Uplift Modeling meta-learner framework"""
    def __init__(self, method='xlearner'): ...
    def fit(self, X, treatment, outcome): ...
    def predict(self, X): ...
```

### Running Code Tests

```bash
# Test a specific model
cd paper2skills-code/causal_inference/uplift_model
python model.py

# Or use pytest (if tests are added)
python -m pytest model.py -v
```

## Domain Mapping

下表 "Code Dir Status" 标识 `paper2skills-code/` 下对应子目录的落地状态:
- ✅ 已落地 — 目录存在,可 import;
- 📦 镜像保留 — 子项目已迁出本仓库,代码模板留作复用引用;
- ⬜ 仅 vault — 当前只有 Skill 卡片,无 code 子目录,需要时按 snake_case 新建。

| English Directory | Chinese Directory | Domain | Code Dir Status |
|-------------------|-------------------|--------|-----------------|
| `causal_inference` | `01-因果推断` | Causal inference, uplift modeling | ✅ |
| `ab_testing` | `02-A_B实验` | A/B testing, multi-armed bandits | ✅ |
| `time_series` | `03-时间序列` | Demand forecasting, time series | ✅ |
| `supply_chain` | `04-供应链` | Inventory optimization | ✅ |
| `recommendation` | `05-推荐系统` | Recommendation systems | ✅ |
| `growth_model` | `06-增长模型` | Churn prediction, LTV | ✅ |
| `nlp_voc` | ~~07-NLP-VOC~~ | 已迁至 `../ai_nlp_voc/`,本仓库保留代码模板 | 📦 |
| `knowledge_graph` | `08-知识图谱` | Heterogeneous graphs, hyperbolic embedding | ✅ |
| `data_agent_llm` | `09-DataAgent-LLM` | DataAgent, LLM-powered data analysis | ✅ |
| `mas` | `10-MAS` | Multi-agent systems, planning, orchestration | ✅ |
| `ai_humanities` | `11-AI人文` | AI × Humanities: cross-modal transfer, LoRA, continual learning, prompt tuning as life metaphors | ⬜ |
| `ml_fundamentals` | `12-ML基础` | Feature engineering, model evaluation fundamentals | ⬜ |
| `advertising` | `13-广告分析` | Ad attribution (Shapley/Markov), ROAS optimization, budget allocation | ⬜ |
| `user_analytics` | `14-用户分析` | Funnel analysis, cohort retention, RFM segmentation | ⬜ |
| `marketing` | `15-营销投放分析` | Marketing Mix Modeling (MMM), promotion effectiveness, causal ML | ⬜ |
| `llm_agent_engineering` | `16-智能体工程` | Agent Skills/Tools, Context Engineering, MCP/A2A protocols, Function Calling (Hermes) | ✅ |

## Quality Standards

Skills must meet these criteria (enforced by `paper-审核`):

- **Algorithm Principle**: Original explanation (not copied), includes math intuition
- **Applications**: Specific scenarios, not generic; must relate to mother & baby cross-border e-commerce
- **Code**: Complete and runnable, includes test cases, clear I/O definitions
- **Skill Relations**: Links to ≥2 existing skills
- **Business Value**: Quantified ROI estimates (no vague terms like "high" or "low")

**Pass threshold**: Total score ≥ 7/10 with code dimension ≥ 7/10

## Working with Skills

### Skill Evolution

Skills can be improved through the `evolve/` directories:

```
paper2skills-skills/paper-选题/evolve/
├── evolution-log.md    # Tracks improvement iterations
└── round-1/            # Specific evolution rounds
```

### Skill File Format

```markdown
---
name: paper-workflow
description: This skill should be used when...
version: 0.1.0
---

# Skill content...
```

## Dependencies

Install Python dependencies:

```bash
cd paper2skills-code
pip install -r requirements.txt
```

Key packages: numpy, pandas, scikit-learn, statsmodels, prophet, causalml, econml

## ArXiv Search Strategy

Use the keyword library at `paper2skills-vault/07-资源库/关键词库.md`:

```bash
# Example ArXiv API query
curl "https://export.arxiv.org/api/query?search_query=all:uplift+modeling&start=0&max_results=10"
```

Search priority: Papers with code implementations > experimental validation > theoretical only. Exclude surveys, meta-analyses, and pure theory papers without experiments.

## Recent Skills Added

| Date | Skill | Domain | Commit |
|------|-------|--------|--------|
| 2026-05-15 | Marketing Mix Modeling (MMM) + Promotion Effectiveness (DML) | 15-营销投放分析 | — |
| 2026-05-15 | Ad Attribution Modeling + ROAS Budget Optimization | 13-广告分析 | — |
| 2026-05-15 | User Funnel Analysis + Cohort Retention Analysis | 14-用户分析 | — |
| 2026-05-15 | Feature Engineering fundamentals | 12-ML基础 | — |
| 2026-05-15 | AI Tech × Healing Quotes Card Library (4 directions: StructLoRA, InfLoRA, Prompt Tuning, Cross-Modal Transfer) | 11-AI人文 | — |
| 2026-05-11 | Phase 7 D4 Superset native filters | 07-NLP-VOC (已迁至 ai_nlp_voc) | `311e3bd` |
| 2026-05-10 | Phase 6 D10 BI dashboard C path | 07-NLP-VOC (已迁至 ai_nlp_voc) | `cad5be5` |
| 2026-05-08 | Phase 7 D1-D3 Superset BI B path | 07-NLP-VOC (已迁至 ai_nlp_voc) | `a765876` / `6f9211d` / `0d92103` |
| 2026-05-06 | Self-Improving LLM Agent Pipeline | 07-NLP-VOC (已迁至 ai_nlp_voc) | `985e82b` |

## Sync Status Tracking

The sync system tracks publication status across platforms:

- **vault**: Obsidian knowledge base
- **github**: Code repository
- **feishu**: Lark/feishu webhook (requires `~/.paper2skills/feishu_webhook` configuration)

Check status in: `paper2skills-vault/07-资源库/sync_status.json`

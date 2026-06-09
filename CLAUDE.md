# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**paper2skills** - A system that converts academic papers into actionable business decision skill cards, focused on cross-border e-commerce for mother & baby products (母婴出海跨境电商).

The workflow transforms academic research (primarily from ArXiv) into practical business skills through a 4-step pipeline: paper selection → extraction → review → sync.

## Project Structure

```
├── paper2skills-skills/     # Claude Code skills for the workflow
│   ├── paper-workflow/      # Orchestrates the complete workflow
│   ├── paper-选题/           # Step 1: Paper selection from ArXiv/GitHub
│   ├── paper-萃取/           # Step 2: Extract papers into Skill cards
│   ├── paper-审核/           # Step 3: Quality review
│   └── paper-同步/           # Step 4: Sync to multiple platforms
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
│   ├── 17-价格优化/            # Dynamic pricing, competitive monitoring, markdown, bundle
│   ├── 18-物流履约/            # Cross-border routing, last-mile delivery, returns
│   ├── 19-风控反欺诈/          # Fake review detection, transaction anomaly, click fraud
│   ├── 20-AI视频生成/          # Virtual anchor demo, product showcase I2V, brand video, UGC
│   ├── 21-合规决策/            # Category compliance prescan, regulatory risk, compliance-as-moat
│   ├── 22-数据采集工程/         # Document intelligence, web crawling, identity resolution, federated collection
│   ├── 23-运营财务/             # FBA fee intelligence, P&L attribution, cash flow forecasting
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
    ├── llm_agent_engineering/
    ├── ml_fundamentals/
    ├── advertising/
    ├── user_analytics/
    ├── marketing/
    ├── pricing/
    ├── logistics/
    ├── risk_fraud/
    ├── visual_content/       # AI视频生成: anchor_demo / product_showcase / brand_video / ugc_talking_head
    └── compliance/           # 合规决策: category_compliance_prescan (Sprint 4 新建)

  已落地 code 子模块: causal_inference / ab_testing / time_series / supply_chain /
    recommendation / growth_model / knowledge_graph / data_agent_llm / mas /
    llm_agent_engineering / ml_fundamentals / advertising / compliance (Sprint 4-5 补全)
  仅 vault Skill 卡片 (按需建立 code 子模块): ai_humanities / user_analytics /
    marketing / pricing / logistics / risk_fraud / visual_content
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
| `paper2skills-vault/07-资源库/MasterPrompt.md` | Master prompt for converting papers to skills |
| `paper2skills-vault/07-资源库/关键词库.md` | ArXiv search keywords by domain |
| `paper2skills-vault/07-资源库/sync_status.json` | Tracks sync status across platforms |
| `paper2skills-skills/paper-同步/scripts/sync.py` | Sync script for vault/GitHub/feishu |

## Workflow Commands

### Local Playbook Preview

```bash
# 1. 重新生成 playbook
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . --vault paper2skills-vault --out playbook

# 2. 启动本地预览服务器（D3 图谱需要 HTTP 协议）
python3 -m http.server 8080 --directory playbook
# 访问: http://localhost:8080
# 注意: D3 ego-graph 和 graph-data.json XHR 在 file:// 协议下不可用
```



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

### Governance and Incremental Workflow

Use these commands before any incremental topic update, paper search, extraction,
Skill creation, or graph optimization:

```bash
# Quality gate: MAS tests, Markdown UTF-8, AST, domain registry, deps, asset alignment
python3 -m paper2skills_common.doctor --json

# Build an auditable paper candidate queue from graph gaps + roadmaps
python3 paper2skills-skills/paper-选题/scripts/build_candidate_queue.py --dry-run

# Run the staged incremental workflow driver. Default behavior is dry-run.
python3 paper2skills-skills/paper-workflow/scripts/run_incremental_workflow.py --one-topic

# Run 20 Darwin-style autoresearch evolution loops.
python3 paper2skills-skills/paper-workflow/scripts/run_darwin_evolution.py --loops 20

# Rebuild derived governance snapshots
python3 paper2skills-skills/paper-同步/scripts/rebuild_sync_status.py --dry-run
python3 paper2skills-skills/paper-同步/scripts/build_asset_inventory.py --dry-run
python3 paper2skills-skills/paper-同步/scripts/build_fitness_snapshot.py --dry-run
```

Generated files such as `paper_candidate_queue.json`, `workflow_runs/*.json`,
`darwin_evolution_runs/*.json`, `darwin_evolution_20loop_report.md`,
`sync_status.json`, `skill_asset_inventory.json`, graph reports, and fitness
snapshots are script-maintained audit artifacts. Do not treat them as a second
manual source of truth for project structure or domain mapping.

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

Each skill card follows a 5-module structure:

1. **Algorithm Principle** (≤300 words) - Core idea, math intuition, key assumptions
2. **Business Applications** (1-2 scenarios) - Specific mother & baby cross-border e-commerce use cases
3. **Code Template** - Runnable Python code with test cases
4. **Skill Relations** - Prerequisites, extensions, combinable skills
5. **Business Value Assessment** - ROI estimate, difficulty rating (1-5 stars), priority score

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
- ⬜ 仅 vault — 当前只有 Skill 卡片,无 code 子目录,需要时按 snake_case 新建.

| English Directory | Chinese Directory | Domain | Skill 数 | Code Dir Status |
|-------------------|-------------------|--------|---------|-----------------|
| `causal_inference` | `01-因果推断` | Causal inference, uplift modeling, DiD, IV | 15 | ✅ |
| `ab_testing` | `02-A_B实验` | A/B testing, multi-armed bandits, sequential testing | 13 | ✅ |
| `time_series` | `03-时间序列` | Demand forecasting, conformal prediction, LLM forecasting | 15 | ✅ |
| `supply_chain` | `04-供应链` | 供应计划全链路（需求预测→促销拆解→预算分配→MOQ批量→产能排产→新品冷启动→健康诊断）+ 库存优化 + 设施选址 | 19 | ✅ |
| `recommendation` | `05-推荐系统` | Recommendation systems, cold start, diversity reranking | 16 | ✅ |
| `growth_model` | `06-增长模型` | Churn, LTV, RFM, market size estimation, PLC stage | 22 | ✅ |
| `nlp_voc` | ~~07-NLP-VOC~~ | 已迁至 `../ai_nlp_voc/`,本仓库保留代码模板 | — | 📦 |
| `knowledge_graph` | `08-知识图谱` | Heterogeneous graphs, GNN, KG construction, KGQA | 28 | ✅ |
| `data_agent_llm` | `09-DataAgent-LLM` | DataAgent, LLM-powered data analysis, NL2Dashboard | 12 | ✅ |
| `mas` | `10-MAS` | Multi-agent systems, planning, orchestration | 35 | ✅ |
| `ai_humanities` | `11-AI人文` | AI × Humanities: cross-modal transfer, LoRA | 7 | ⬜ |
| `ml_fundamentals` | `12-ML基础` | Feature engineering, model evaluation, drift detection, performance monitor | 11 | ✅ |
| `advertising` | `13-广告分析` | Ad attribution, ROAS, Listing quality, cross-device, delayed CVR | 19 | ✅ |
| `user_analytics` | `14-用户分析` | Funnel, cohort, clickstream, trajectory, traffic source | 18 | ⬜ |
| `marketing` | `15-营销投放分析` | MMM, promotion effectiveness, causal ML, channel saturation | 9 | ⬜ |
| `llm_agent_engineering` | `16-智能体工程` | Agent Skills/Tools, Context, MCP/A2A, Safety/Fault/Cost | 19 | ✅ |
| `pricing` | `17-价格优化` | Dynamic pricing, competitive monitoring, markdown, bundle | 5 | ⬜ |
| `logistics` | `18-物流履约` | Cross-border routing, last-mile prediction, returns/reverse logistics | 3 | ⬜ |
| `risk_fraud` | `19-风控反欺诈` | Fake review detection, transaction anomaly, click fraud | 3 | ⬜ |
| `visual_content` | `20-AI视频生成` | Virtual anchor demo, product showcase I2V, brand video, talking-head UGC | 8 | ⬜ |
| `compliance` | `21-合规决策` | **新领域 (2026-05-25)**: Category compliance prescan, regulatory risk, compliance-as-moat | 1 | ✅ |
| `data_collection` | `22-数据采集工程` | **新领域 (2026-06-05)**: Document intelligence, identity resolution, fake review detection, federated collection, web crawling | 13 | ⬜ |
| `operations_finance` | `23-运营财务` | **新领域 (2026-06-09)**: FBA fee intelligence, P&L attribution, cash flow forecasting | 3 | ⬜ |

**说明**：`ml_fundamentals`、`advertising`、`compliance` code 目录已于 Sprint 4-5 补全落地。`user_analytics`、`marketing`、`pricing`、`logistics`、`risk_fraud`、`visual_content` 仍为 vault-only，按需建立 code 子模块。

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

> **Python 版本**: 3.14+（`.python-version` 已锁定）
>
> **macOS 前置**: `brew install libomp`（lightgbm/causalml 运行时依赖）
>
> **MAS 系统**（`mas/`）仅用标准库，无需安装任何包。

```bash
# 使用 venv（PEP 668 系统 Python 禁止直接 pip install）
python3 -m venv .venv
source .venv/bin/activate

# 锁定版本安装（推荐，可复现）
pip install -r paper2skills-code/requirements-lock.txt

# 验证
python -c "import causalml, sklearn, statsmodels; print('OK')"
```

Key packages: numpy 2.4, pandas 3.0, scikit-learn 1.8, statsmodels 0.14, causalml 0.16

## ArXiv Search Strategy

Use the keyword library at `paper2skills-vault/07-资源库/关键词库.md`:

```bash
# Example ArXiv API query
curl "https://export.arxiv.org/api/query?search_query=all:uplift+modeling&start=0&max_results=10"
```

Search priority: Papers with code implementations > experimental validation > theoretical only. Exclude surveys, meta-analyses, and pure theory papers without experiments.

## Recent Skills Added

> 当前文件系统可见 **302 个 Skill** 跨注册领域. 最新图谱 dry-run: 302 节点 / 5390 边 / P0=0 / P1=0 / P2=9. 这些数字应通过 `doctor`、`build_asset_inventory.py`、`skills_graph_analyzer.py` 重建,不再手动维护为权威来源。

### Sprint 5 (2026-05-25) — 供应链供应计划侧 6 个 Skill

供应计划全链路补缺：促销拆解→预算分配→MOQ批量→产能排产→新品冷启动→健康诊断

| 日期 | Skill | 领域 | 核心论文 |
|------|-------|------|---------|
| 2026-05-25 | Promotion-Demand-Decomposition | 04-供应链 | SPADE arXiv:2411.05852 (NeurIPS 2024) + Hewage JoF 2025 + JD.com SSRN:4777632 |
| 2026-05-25 | Multi-SKU-Procurement-Budget-Allocation | 04-供应链 | arXiv:2301.02662 (Knapsack Ordering 2023) + EJOR Vol.315 2024 |
| 2026-05-25 | Dynamic-Lot-Sizing-MOQ | 04-供应链 | EJOR Q-jump 2018 + EJOR JRP+MOQ 2022 (Chugh et al.) |
| 2026-05-25 | Supplier-Capacity-Planning | 04-供应链 | arXiv:2402.14506 (Rolling Horizon 2024) + IJPE Vol.277 2024 + JIMO Vol.20 2024 |
| 2026-05-25 | New-Product-Inventory-Coldstart | 04-供应链 | M&SOM 21(4) 2019 (Zara Residual Tree) + OR 71(5) 2023 (Bayesian Exploration) |
| 2026-05-25 | Inventory-Health-Aging-Attribution | 04-供应链 | JSCDM 2024 + OSCM Forum 2023 + ACM ICGAIB 2025 + arXiv:2404.07523 + arXiv:2308.13118 |

### Sprint 4 (2026-05-25) — 业务完整性 P0 补缺 + 新领域 21-合规决策

WF-D 选品扫描补缺 + 模型生产化横切面 + WF-B Listing 质量门控 + 新建合规决策领域

| 日期 | Skill | 领域 | 核心论文 |
|------|-------|------|---------|
| 2026-05-25 | Listing-Quality-Scoring | 13-广告分析 | KDD'23 Amazon arXiv:2302.01416 + MetaSynth arXiv:2510.01523 + IPL EMNLP'24 |
| 2026-05-25 | Product-Lifecycle-Stage | 06-增长模型 | AAAI 2025 PhaseFormer arXiv:2511.16248 + AVM arXiv:2511.17275 |
| 2026-05-25 | Category-Compliance-Prescan | **21-合规决策** | RECALL-MM arXiv:2503.23213 (ASME IDETC 2025) + Sci.Reports 2025 WOA-BP |
| 2026-05-25 | Market-Size-Estimation | 06-增长模型 | G-TAB arXiv:2007.13861 (EPFL) + Hu Bass+GT (Kent) + MDPI MonteCarlo 2023 |
| 2026-05-25 | Data-Drift-Detection | 12-ML基础 | DriftGuard arXiv:2601.08928 (2026) + Cry Wolf ICLR 2026 Workshop |
| 2026-05-25 | Model-Performance-Monitor | 12-ML基础 | DriftGuard arXiv:2601.08928 + Champion-Challenger 工业实践 |

### Round 5 (2026-05-22) — AI视频生成 新领域 8 个 Skill

**P0 商品上架 (2 个):**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-22 | AnchorCrafter Virtual Anchor Demo | 20-AI视频生成 | arXiv:2411.17383 (中科院+腾讯) |
| 2026-05-22 | Phantom Product Showcase I2V | 20-AI视频生成 | arXiv:2502.11079 (ByteDance ICCV 2025) |

**P1 品牌+UGC (3 个):**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-22 | Aquarius Brand Video Generation | 20-AI视频生成 | arXiv:2505.10584 (工业级) |
| 2026-05-22 | BrandFusion Multi-Agent | 20-AI视频生成 | arXiv:2603.02816 (2026最新) |
| 2026-05-22 | DAWN Talking-Head Review | 20-AI视频生成 | arXiv:2410.13726 |

**P2 技术储备 (3 个):**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-22 | E-Commerce Video Benchmark | 20-AI视频生成 | ICLR 2026 投稿 (淘宝数据) |
| 2026-05-22 | Text-to-Edit Video Ad | 20-AI视频生成 | arXiv:2501.05884 (商汤) |
| 2026-05-22 | Virbo Multilingual Avatar UGC | 20-AI视频生成 | arXiv:2403.11700 (万兴科技) |

### Round 4 (2026-05-20) — 桑基图流量转化全栈 19 个

### Sprint 3 (2026-05-21) — ML基础建设 + 孤立修复 + P1候选 共12新Skill + 18关联回填

**A组: 12-ML基础 6个 Skill (领域 1→7)**

| Date | Skill | Domain | Type |
|------|-------|--------|------|
| 2026-05-21 | Model Evaluation Metrics | 12-ML基础 | 综合萃取 |
| 2026-05-21 | Cross Validation Strategies | 12-ML基础 | 综合萃取 |
| 2026-05-21 | Imbalanced Data Handling | 12-ML基础 | 综合萃取 |
| 2026-05-21 | Ensemble Methods | 12-ML基础 | 综合萃取 |
| 2026-05-21 | Feature Selection (SHAP/Boruta) | 12-ML基础 | 综合萃取 |
| 2026-05-21 | Hyperparameter Optimization | 12-ML基础 | 综合萃取 |

**B组: 18个孤立Skill关联回填 → 967边/0孤立**

**C组: v2 P1候选 6个 Skill**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-21 | Negative Keyword Safe Guard | 13-广告分析 | eBay SIGIR eCom 2025 |
| 2026-05-21 | Creative Fatigue Detection | 13-广告分析 | arXiv:2204.11588 + 2509.09758 |
| 2026-05-21 | Conformal Prediction Demand UQ | 03-时间序列 | arXiv:2307.16895 (NeurIPS 2023) |
| 2026-05-21 | Multi-Channel Inventory Pooling | 04-供应链 | arXiv:2306.11246 + 2310.12183 |
| 2026-05-21 | Amazon ToS Compliance Guardrail | 13-广告分析 | SAFE-AGENT-L (AAAI 2026 Workshop) |
| 2026-05-21 | TikTok Shop Content Attribution | 13-广告分析 | arXiv:2401.08875 + 2507.15113 |

**D组: CausalRAG (最后HIGH缺口)** — 已存在占位卡转正式

### Round 4 (2026-05-20) — 桑基图流量转化全栈 19 个

**第一轮：流量转化基础 (8 个)**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-20 | TRACE Clickstream Embedding | 14-用户分析 | arXiv:2409.12972 (2024) |
| 2026-05-20 | Non-Item Page Path Modeling | 14-用户分析 | arXiv:2408.15953 (RecSys 2024) |
| 2026-05-20 | Trajectory Pattern Mining | 14-用户分析 | PLOS One 2025 |
| 2026-05-20 | HGNN Cross-Device Matching | 13-广告分析 | arXiv:2304.03215 (NVIDIA) |
| 2026-05-20 | GraphTrack Cross-Device Tracking | 13-广告分析 | arXiv:2203.06833 |
| 2026-05-20 | Traffic Source Analysis | 14-用户分析 | arXiv:2403.16115 (2024) |
| 2026-05-20 | CABB Cross-Category Attribution | 13-广告分析 | arXiv:2507.15113 (2025) |
| 2026-05-20 | Session Intent Shift | 14-用户分析 | arXiv:2507.20185 (2025) |

**P0：缺数据可信度 (3 个)**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-20 | Conformal ROI Prediction | 01-因果推断 | arXiv:2407.01065 (2024) |
| 2026-05-20 | Sparse Matrix Completion (Hájek-GD) | 14-用户分析 | arXiv:2601.12213 (NeurIPS 2025) |
| 2026-05-20 | BCCB Causal Bandits | 02-A_B实验 | arXiv:2604.26169 (2025) |

**P1：不确定性量化+实时 (4 个)**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-20 | Utimac Uncertainty-Aware Completion | 14-用户分析 | arXiv:2605.02225 (2025) |
| 2026-05-20 | EPICSCORE Uncertainty Quantification | 01-因果推断 | arXiv:2502.06995 (2025) |
| 2026-05-20 | SSBC Small Sample Conformal | 01-因果推断 | arXiv:2509.15349 (2025) |
| 2026-05-20 | TRACE Delayed CVR | 13-广告分析 | arXiv:2604.23197 (2025) |

**P2：极端场景鲁棒 (4 个)**

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-20 | BlockEcho Block-Wise Missing Data | 14-用户分析 | IJCAI 2024 |
| 2026-05-20 | STAMImputer Spatio-Temporal MoE | 14-用户分析 | IJCAI 2025 |
| 2026-05-20 | TESLA Cascaded NetCVR | 13-广告分析 | arXiv:2601.19965 (Taobao 2025) |
| 2026-05-20 | CSDM Diffusion Cold-Start CTR | 05-推荐系统 | arXiv:2504.06270 (2025) |

### Sprint 2 (2026-05-17 下午) — WF-A/B P0 阻塞缺口 6 个

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-17 | HiFoReAd 分层时序预测调和 | 03-时间序列 | arXiv:2412.14718 (Walmart BigData 2024) |
| 2026-05-17 | Gen-QOT 提前期分布风险 | 04-供应链 | arXiv:2310.17168 (Amazon 2024) |
| 2026-05-17 | Bass + GEANN 新品冷启动需求 | 06-增长模型 | arXiv:2307.03595 (Amazon 2023) |
| 2026-05-17 | Hierarchical Search Intent Classification | 13-广告分析 | arXiv:2403.06021 (Amazon WWW 2024) |
| 2026-05-17 | PVM 跨平台归因窗口统一化 | 13-广告分析 | arXiv:2511.22918 (NeurIPS 2025) |
| 2026-05-17 | Dial-In LLM 客服意图聚类 | 09-DataAgent-LLM | arXiv:2412.09049 (EMNLP 2025) |

### Sprint 1 (2026-05-17 上午) — WF-E/WF-C ABSA 闭环 4 个

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-17 | AGRS 属性引导评论摘要 | 14-用户分析 | arXiv:2509.26103 (Wayfair 2025) |
| 2026-05-17 | MAA 多 Agent 行动建议 | 14-用户分析 | arXiv:2601.12024 |
| 2026-05-17 | StaR 观点语句排序 | 14-用户分析 | arXiv:2604.03724 |
| 2026-05-17 | LACA 跨语言 ABSA | 14-用户分析 | arXiv:2508.09515 (ACL 2025) |

### 6h 迭代 (2026-05-17) — 跨领域桥梁 8 个

| Date | Skill | Domain | Paper |
|------|-------|--------|-------|
| 2026-05-17 | Hierarchical Product KG Construction | 08-知识图谱 | arXiv:2410.21237 |
| 2026-05-17 | Counterfactual Recommendation DCE | 05-推荐系统 | arXiv:2403.00817 (WWW 2024 oral) |
| 2026-05-17 | Switchback Experiment Design | 02-A_B 实验 | arXiv:2406.06768 |
| 2026-05-17 | GCF Causal Time Series Forecasting | 03-时间序列 | AAAI 2025 (Amazon) |
| 2026-05-17 | CoLaKG KG-Augmented Recommendation | 08-知识图谱 | SIGIR 2025 |
| 2026-05-17 | DARA Agentic MMM Optimizer | 15-营销投放 | arXiv:2601.14711 (WWW 2026) |
| 2026-05-17 | DML Cohort Causal Effect | 01-因果推断 | ECML PKDD 2023 (Amazon) |
| 2026-05-17 | Customer Journey Decision Tree | 09-DataAgent-LLM | 综合萃取 |

### Phase 1-3 历史 (2026-03 ~ 2026-05-15)

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

### MAS 多智能体系统 MVP (2026-05-17)

- **5 个业务工作流**: WF-A 智能补货 / WF-B 广告优化 / WF-C 客服分诊 / WF-D 选品扫描 / WF-E Review 监控
- **工作流 Skill 覆盖率** (2026-06-01): WF-A 95% / WF-B 90% / WF-C 85% / WF-D 80% / WF-E 85%
- **14 个核心模块**: agents/ + graphs/ + skills/ + state/ + hitl/ + checkpointing/ + observability/
- **61/61 集成测试全绿**(含 MAS 工作流、MCP 路由、工具链、治理回归、增量 workflow 与 Darwin evolution 回归)
- **入口**: [`mas/main.py`](mas/main.py) + [`mas/README.md`](mas/README.md) 部署指南

## Sync Status Tracking

The sync system tracks publication status across platforms:

- **vault**: Obsidian knowledge base
- **github**: Code repository
- **feishu**: Lark/feishu webhook (requires `~/.paper2skills/feishu_webhook` configuration)

Check status in: `paper2skills-vault/07-资源库/sync_status.json`

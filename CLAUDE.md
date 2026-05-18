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
| `paper2skills-vault/07-资源库/MasterPrompt.md` | Master prompt for converting papers to skills |
| `paper2skills-vault/07-资源库/关键词库.md` | ArXiv search keywords by domain |
| `paper2skills-vault/07-资源库/sync_status.json` | Tracks sync status across platforms |
| `paper2skills-skills/paper-同步/scripts/sync.py` | Sync script for vault/GitHub/feishu |

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

> 总计 **107 个 Skill** 跨 15 个核心领域(+16-智能体工程). 最近 Week 4-5 完成三轮迭代萃取 19 个新 Skill.

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
- **14 个核心模块**: agents/ + graphs/ + skills/ + state/ + hitl/ + checkpointing/ + observability/
- **37/37 集成测试全绿**(MAS 骨架 12 + 各工作流 5+5+5+4 + 集成 6)
- **入口**: [`mas/main.py`](mas/main.py) + [`mas/README.md`](mas/README.md) 部署指南

## Sync Status Tracking

The sync system tracks publication status across platforms:

- **vault**: Obsidian knowledge base
- **github**: Code repository
- **feishu**: Lark/feishu webhook (requires `~/.paper2skills/feishu_webhook` configuration)

Check status in: `paper2skills-vault/07-资源库/sync_status.json`

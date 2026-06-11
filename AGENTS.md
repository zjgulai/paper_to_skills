# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**paper2skills** - A system that converts academic papers into actionable business decision skill cards, focused on cross-border e-commerce for mother & baby products (母婴出海跨境电商).

The workflow transforms academic research (primarily from ArXiv) into practical business skills through a 4-step pipeline: paper selection → extraction → review → sync.

**Live Playbook**: https://skills.lute-tlz-dddd.top

## Current State (2026-06-11)

| Metric | Value |
|--------|-------|
| Skill pages | **398** |
| Domains | **22** |
| Graph edges | **6,618** |
| Agent pages | **12** callable agents |
| New pages | **agent-report.html** |

## Project Structure

```
├── paper2skills-skills/         # Codex skills for the workflow
│   ├── paper-workflow/          # Orchestrates the complete workflow
│   ├── paper-选题/               # Step 1: Paper selection from ArXiv/GitHub
│   ├── paper-萃取/               # Step 2: Extract papers into Skill cards
│   ├── paper-审核/               # Step 3: Quality review
│   ├── paper-同步/               # Step 4: Sync to multiple platforms
│   └── playbook-generator/      # Static HTML playbook builder
│       └── scripts/
│           ├── build_playbook.py          # Main build script
│           └── config/
│               ├── skill_ps_override.yaml     # Problem-solved overrides
│               ├── skill_biz_context_override.yaml
│               └── skill_handbook_map.yaml    # Agent→Playbook mapping
├── paper2skills-vault/          # Knowledge base (Obsidian-compatible)
│   ├── 01-因果推断/              # Causal inference skills
│   ├── 02-A_B实验/               # A/B testing skills
│   ├── 03-时间序列/              # Time series skills
│   ├── 04-供应链/                # Supply chain skills (32 skills)
│   ├── 05-推荐系统/              # Recommendation system skills
│   ├── 06-增长模型/              # Growth model skills (29 skills)
│   ├── 08-知识图谱/              # Knowledge graph / GNN skills (28 skills)
│   ├── 09-DataAgent-LLM/         # DataAgent & LLM-powered analytics
│   ├── 10-MAS/                   # Multi-Agent System skills (35 skills)
│   ├── 11-AI人文/                 # AI × Humanities
│   ├── 12-ML基础/                 # Machine learning fundamentals
│   ├── 13-广告分析/               # Ad attribution, ROAS (24 skills)
│   ├── 14-用户分析/               # Funnel, cohort, RFM (26 skills)
│   ├── 15-营销投放分析/            # MMM, promotion effectiveness (17 skills)
│   ├── 16-智能体工程/              # LLM Agent Engineering (45 skills)
│   ├── 17-价格优化/               # Dynamic pricing (10 skills)
│   ├── 18-物流履约/               # Cross-border logistics (8 skills)
│   ├── 19-风控反欺诈/             # Fraud detection (9 skills)
│   ├── 20-AI视频生成/             # Virtual anchor, brand video (10 skills)
│   ├── 21-合规决策/               # Compliance decisions (10 skills)
│   ├── 22-数据采集工程/            # Data collection & quality (16 skills)
│   ├── 23-运营财务/               # FBA finance, P&L (8 skills)
│   ├── 07-资源库/                 # Master Prompt, keywords, sync status
│   └── papers/                   # Downloaded papers by domain
└── paper2skills-code/           # Python code templates
    ├── causal_inference/
    ├── ab_testing/
    ├── time_series/
    ├── supply_chain/
    ├── recommendation/
    ├── growth_model/
    ├── nlp_voc/
    ├── knowledge_graph/
    └── mas/
```

## Key Files

| File | Purpose |
|------|---------|
| `paper2skills-vault/07-资源库/MasterPrompt.md` | Master prompt for converting papers to skills |
| `paper2skills-vault/07-资源库/关键词库.md` | ArXiv search keywords by domain |
| `paper2skills-vault/07-资源库/sync_status.json` | Tracks sync status across platforms |
| `paper2skills-skills/paper-同步/scripts/sync.py` | Sync script for vault/GitHub/feishu |
| `paper2skills-skills/playbook-generator/scripts/build_playbook.py` | Playbook build script |
| `paper2skills-skills/playbook-generator/scripts/config/skill_handbook_map.yaml` | Maps skills to agent/playbook pages |

## Playbook Pages

The built playbook at `playbook/` contains:

| Page | URL path | Description |
|------|----------|-------------|
| index.html | `/` | Dashboard overview |
| agents.html | `/agents.html` | 12 AI agents with local compute engines |
| agent-report.html | `/agent-report.html` | Persisted agent run reports (localStorage) |
| playbooks/\*.html | `/playbooks/` | 16 business scenario handbooks |
| skills/\*.html | `/skills/` | 398 individual Skill detail pages |
| domains/\*.html | `/domains/` | 22 domain pages |
| ai-roadmap.html | `/ai-roadmap.html` | CEO-facing capability roadmap |
| chat.html | `/chat.html` | AI knowledge base chat |

## Agent Marketplace (agents.html)

12 callable agents with **local compute engines** (no external API):

| Agent ID | Name | Type | Key Inputs |
|----------|------|------|-----------|
| agent-supply-sentinel | 供应链哨兵 | Numeric | stock, velocity, lead_time |
| agent-pricing-advisor | 动态定价顾问 | Numeric | price, cost, comp_range, bsr |
| agent-pnl-analyzer | P&L 透视镜 | Numeric | revenue, cogs, fba, ads, return_rate |
| agent-ad-attribution | 广告归因侦探 | Numeric | platform, spend, target_acos |
| agent-competitor-radar | 竞品雷达站 | Numeric | asins, period, metrics |
| agent-listing-doctor | Listing 医生 | Text | title, bullets, keywords |
| agent-voc-decoder | 用户之声解码器 | Text | reviews, lang |
| agent-cs-triage | 客服分诊台 | Text | tickets, platform, sla |
| agent-account-guardian | 账号风险卫士 | Text | notice, asins, health |
| agent-brand-guardian | 品牌合规卫士 | Text | copy, category, market |
| agent-product-radar | 选品雷达 | Text | keyword, market, budget |
| agent-tiktok-content | TikTok 内容官 | Text | product, audience, style, freq |

Each agent run auto-saves to `localStorage('agentReports')` and appears in `agent-report.html`.

**SEED_VERSION**: `v20260611-r2` — 24 seed reports pre-loaded (2 per agent, different scenarios).

## Workflow Commands

### Run Complete Workflow

Use the `paper-workflow` skill to run the complete pipeline:

```bash
# Trigger via natural language to Codex
"Run paper2skills workflow"
"Process this paper through the complete pipeline"
```

### Individual Steps

Each step can be triggered separately via skills:

1. **Paper Selection** (`paper-选题`): "筛选论文" / "Search ArXiv for uplift modeling papers"
2. **Extraction** (`paper-萃取`): "萃取论文" / "Generate skill card from this paper"
3. **Review** (`paper-审核`): "审核 skill" / "Review quality of Skill-Uplift-Modeling"
4. **Sync** (`paper-同步`): Use the sync script directly (see below)

### Build & Deploy Playbook

```bash
# Build
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . --vault paper2skills-vault --out playbook

# Verify: no 演示模式, compute functions exist, agent-report.html exists
grep -c "演示模式" playbook/agents.html      # expect 0
grep -c "computeSupplySentinel" playbook/agents.html  # expect >0
ls playbook/agent-report.html               # must exist

# Deploy
cd playbook && tar -czf /tmp/pb.tar.gz assets/ domains/ graph/ playbooks/ \
  topics/ workflows/ skills/ agents.html agent-report.html \
  ai-roadmap.html index.html chat.html build-report.json README.md
rsync -avz -e "ssh -i ../ai_video.pem" /tmp/pb.tar.gz ubuntu@101.34.52.232:/tmp/
ssh -i ../ai_video.pem ubuntu@101.34.52.232 \
  "rm -rf /opt/paper2skills/html/* && tar -xzf /tmp/pb.tar.gz -C /opt/paper2skills/html/"
```

### Sync Script Usage

```bash
cd paper2skills-skills/paper-同步
python scripts/sync.py --skill Skill-Uplift-Modeling
python scripts/sync.py --skill Skill-Uplift-Modeling --target vault,github
python scripts/sync.py --status
```

## Skill Card Format

Each skill card follows a 5-module structure:

1. **Algorithm Principle** (≤300 words) - Core idea, math intuition, key assumptions
2. **Business Applications** (1-2 scenarios) - Specific mother & baby cross-border e-commerce use cases
3. **Code Template** - Runnable Python code with test cases
4. **Skill Relations** - Prerequisites, extensions, combinable skills
5. **Business Value Assessment** - ROI estimate, difficulty rating (1-5 stars), priority score

### Agent Calling Case Section

Skills linked to agents contain a `## 🧪 调用案例` section at the bottom:

```markdown
## 🧪 调用案例（智能体广场验证）

**Agent**：{Agent名称}
**测试输入**：{具体参数}
**输出摘要**：{核心输出，2-3行}
**验证状态**：✅ 本地计算通过 | 2026-06-11
```

## Quality Standards

Skills must meet these criteria (enforced by `paper-审核`):

- **Algorithm Principle**: Original explanation (not copied), includes math intuition
- **Applications**: Specific scenarios, not generic; must relate to mother & baby cross-border e-commerce
- **Code**: Complete and runnable, includes test cases, clear I/O definitions
- **Skill Relations**: Links to ≥2 existing skills
- **Business Value**: Quantified ROI estimates (no vague terms like "high" or "low")

**Pass threshold**: Total score ≥ 7/10 with code dimension ≥ 7/10

## Domain Mapping

| English Directory | Chinese Directory | Domain | Skills |
|-------------------|-------------------|--------|--------|
| `causal_inference` | `01-因果推断` | Causal inference, uplift modeling | 15 |
| `ab_testing` | `02-A_B实验` | A/B testing, multi-armed bandits | 13 |
| `time_series` | `03-时间序列` | Demand forecasting | 15 |
| `supply_chain` | `04-供应链` | Inventory optimization | 32 |
| `recommendation` | `05-推荐系统` | Recommendation systems | 16 |
| `growth_model` | `06-增长模型` | Churn prediction, LTV | 29 |
| `knowledge_graph` | `08-知识图谱` | Heterogeneous graphs | 28 |
| `mas` | `10-MAS` | Multi-agent systems | 35 |
| `llm_agent_engineering` | `16-智能体工程` | LLM Agent Engineering | 45 |

## Dependencies

```bash
cd paper2skills-code
pip install -r requirements.txt
```

Key packages: numpy, pandas, scikit-learn, statsmodels, prophet, causalml, econml

## Sync Status Tracking

- **vault**: Obsidian knowledge base
- **github**: Code repository
- **feishu**: Lark/feishu webhook (requires `~/.paper2skills/feishu_webhook`)

Check status: `paper2skills-vault/07-资源库/sync_status.json`


## Project Structure

```
├── paper2skills-skills/     # Codex skills for the workflow
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
│   ├── 07-NLP-VOC/           # NLP/Voice of Customer skills
│   ├── 08-知识图谱/          # Knowledge graph / GNN skills
│   ├── 10-MAS/               # Multi-Agent System skills
│   ├── 07-资源库/            # Master Prompt, keywords, sync status
│   └── papers/               # Downloaded papers by domain
└── paper2skills-code/       # Python code templates
    ├── causal_inference/
    ├── ab_testing/
    ├── time_series/
    ├── supply_chain/
    ├── recommendation/
    ├── growth_model/
    ├── nlp_voc/
    ├── knowledge_graph/
    └── mas/
```

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
# Trigger via natural language to Codex
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

| English Directory | Chinese Directory | Domain |
|-------------------|-------------------|--------|
| `causal_inference` | `01-因果推断` | Causal inference, uplift modeling |
| `ab_testing` | `02-A_B实验` | A/B testing, multi-armed bandits |
| `time_series` | `03-时间序列` | Demand forecasting, time series |
| `supply_chain` | `04-供应链` | Inventory optimization |
| `recommendation` | `05-推荐系统` | Recommendation systems |
| `growth_model` | `06-增长模型` | Churn prediction, LTV |
| `nlp_voc` | `07-NLP-VOC` | Sentiment analysis, VOC |
| `knowledge_graph` | `08-知识图谱` | Heterogeneous graphs, hyperbolic embedding |
| `mas` | `10-MAS` | Multi-agent systems, planning, orchestration |

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

## Sync Status Tracking

The sync system tracks publication status across platforms:

- **vault**: Obsidian knowledge base
- **github**: Code repository
- **feishu**: Lark/feishu webhook (requires `~/.paper2skills/feishu_webhook` configuration)

Check status in: `paper2skills-vault/07-资源库/sync_status.json`

# paper2skills 知识库

> **最近更新:2026-05-17 下午**  · 当前 **107 个 Skill 卡片** 跨 15 个核心领域 + 16-智能体工程

## 项目简介
将前沿学术论文转化为可落地的商业决策技能卡片，专注于母婴出海跨境电商业务场景。

## 目录结构

### 技能领域(括号内为 Skill 数量)
- `01-因果推断/` (7) - 因果效应估计、Uplift / DML / Causal Forest / DiD / IV / Mediation / PC
- `02-A_B实验/` (6) - 实验设计 / MAB / Thompson / Switchback / Power Analysis / Result Interpretation
- `03-时间序列/` (7) - Prophet / TFT / GCF Causal / Doubly Robust / **HiFoReAd 分层调和** / Anomaly / TSF
- `04-供应链/` (6) - Multi-Echelon / Two-Echelon DRL / Safety Stock / **Gen-QOT 提前期风险** / Monodense / Demand
- `05-推荐系统/` (9) - MF / DL-HI / NeuralNDCG / SR-GNN / Cold-Start Meta / **DCE 反事实** / Semantic ID / Diversity / Explainable
- `06-增长模型/` (11) - LTV / Churn / Customer Journey / DQN / Uplift Churn / RFM / Lifecycle / **Bass 冷启动** / Cold-Start Product / DL Churn / New Product Mining
- `07-NLP-VOC/` - 已迁至 ai_nlp_voc 独立仓库,本仓库保留原始论文档案
- `08-知识图谱/` (11) - HGT / HGCN / Dense Retrieval / GraphRAG / KG Auto / **Hierarchical Product KG** / **CoLaKG** / KG Completion / KGQA / Multilingual NER / KG-Skills
- `09-DataAgent-LLM/` (7) - DeepAnalyze / Argos / Dashboard / SQL Agent / RCA / Customer Journey Tree / **Dial-In LLM**
- `10-MAS/` (12) - MAS Orchestrator / Subagent / Skill Registry / AutoGen / MetaGPT / CAMEL / ReAct / ToT / Reflexion / Debate / Memory / Self-Improving
- `11-AI人文/` (1) - AI Tech × Healing Cards
- `12-ML基础/` (1) - Feature Engineering
- `13-广告分析/` (4) - Ad Attribution / ROAS Budget / **Hierarchical Search Intent** / **PVM Attribution Window**
- `14-用户分析/` (6) - Funnel / Cohort / **AGRS** / **MAA** / **StaR** / **LACA**
- `15-营销投放分析/` (3) - Marketing Mix Modeling / **DARA Agentic MMM** / Promotion Effectiveness
- `16-智能体工程/` (16) - Agent Skills / Context / MCP/A2A / 协议落地等 16 项

> **粗体** Skill 为 Week 4-5 三轮迭代(6h + Sprint 1 + Sprint 2)新增萃取的 19 个 P0 Skill.

### 项目管理
- `00-项目管理/项目总结.md` - 整体里程碑与最新状态
- `00-项目管理/进度追踪.md` - Week-by-week 进度
- `00-项目管理/next-papers-roadmap.md` - Sprint 3 P1 候选清单
- `00-项目管理/sprint1-2-iteration-report-20260517.md` - Sprint 1+2 完整复盘
- `00-项目管理/6h-iteration-report-20260517.md` - 6h 迭代复盘
- `00-项目管理/Skill关联图谱.md` - 图谱可视化与缺口热力图

### 资源库
- `07-资源库/MasterPrompt.md` - Master Prompt 完整内容
- `07-资源库/关键词库.md` - 论文搜索关键词
- `07-资源库/审核问题库.md` - 质量审核维度
- `07-资源库/skill-aliases.json` - Skill 命名别名表(消除图谱漂移误判)
- `07-资源库/实施指南/` - MAB / 预测等运营手册

## 使用方法
1. 从 ArXiv 筛选论文(参考 `07-资源库/关键词库.md`)
2. 使用 Master Prompt 生成 Skill 卡片(`07-资源库/MasterPrompt.md`)
3. 质量审核后同步到各平台(`paper-同步` skill)
4. 用 `skills_graph_analyzer.py` 检查图谱状态与新缺口

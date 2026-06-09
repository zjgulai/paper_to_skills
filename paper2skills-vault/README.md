# paper2skills 知识库

> **最近更新：2026-06-06**  · 当前 **320 个 Skill 卡片** 跨 22 个领域 · 图谱 **5,390+ 条边** · missing_prerequisite 断链 = **0** · HIGH 缺口 = **0** · 新建跨域桥梁 **7 条**（mas↔advertising / mas↔KG / data_collection↔recommendation / mas↔advertising×2 / 更多）

## 项目简介
将前沿学术论文转化为可落地的商业决策技能卡片，专注于母婴出海跨境电商业务场景。

## 目录结构

### 技能领域(括号内为 Skill 数量)
- `01-因果推断/` (15) - Uplift / DML / Causal Forest / DiD / IV / Mediation / PC / Conformal ROI / EPICSCORE / SSBC / ClusterSC / Causal-Attribution-Bridge / ...
- `02-A_B实验/` (12) - 实验设计 / MAB / Thompson / Switchback / Power Analysis / Result Interpretation / BCCB / CUPED / Network Effect / Sequential AB / Agentic AB / ...
- `03-时间序列/` (14) - Prophet / TFT / GCF Causal / Doubly Robust / HiFoReAd / Anomaly / TSF / TimeCMA / Conformal UQ / Conformal TS / Multivariate Cointegration / Forecast-Driven-Inventory / ...
- `04-供应链/` (19) - Multi-Echelon / Two-Echelon DRL / Safety Stock / Gen-QOT / Monodense / Demand / FSDA / PASTA / PPO swap / NEO LRP / Multilevel FLP / Inventory-Pooling / Promotion-Demand-Decomposition / Multi-SKU-Procurement / Dynamic-Lot-Sizing-MOQ / Supplier-Capacity-Planning / New-Product-Coldstart / Inventory-Health-Aging / ...
- `05-推荐系统/` (11) - MF / DL-HI / NeuralNDCG / SR-GNN / Cold-Start Meta / DCE 反事实 / Semantic ID / Diversity / Explainable / CSDM Diffusion / CAGED Debiased
- `06-增长模型/` (22) - LTV / Churn / Customer Journey / DQN / Uplift Churn / RFM / Lifecycle / Bass / Cold-Start / DL Churn / New Product Mining / UCB-LDP / CATE-NBA / GenAgent / Category-Trend / Competitor-Intelligence / Product-Opportunity / Supplier-Evaluation / Market-Size-Estimation / Product-Lifecycle-Stage / Conformal-ROI / EPICSCORE（共22）
- `07-NLP-VOC/` - 已迁至 ai_nlp_voc 独立仓库,本仓库保留原始论文档案
- `08-知识图谱/` (18) - HGT / HGCN / Dense Retrieval / GraphRAG / KG Auto / Hierarchical Product KG / CoLaKG / KG Completion / KGQA / Multilingual NER / KG-Skills / AgentRouter / SCKG Risk / CausalRAG / Audience KG / GNN Foundations / ...
- `09-DataAgent-LLM/` (11) - DeepAnalyze / Argos / Dashboard / SQL Agent / RCA / Customer Journey Tree / Dial-In LLM / ProRCA / RAG Enhanced / NL2Dashboard / ...
- **`10-MAS/` (35)** - MAS Orchestrator / Subagent / Skill Registry / AutoGen / MetaGPT / CAMEL / ReAct / ToT / Reflexion / Debate / Memory / Self-Improving / Agent-Production-Engineering / **Dynamic-Trust** / **Testing-Verification** / **Resource-Scheduling** / **Consensus-Mechanism** / **Scale-Management** / **LLM-AutoBidding-MAS** ★ / **Adversarial-Defense** / **Cross-Org-Protocol** / **Dynamic-KG-Collaboration** ★ / ...
- `11-AI人文/` (6) - AI Tech × Healing Cards / ...
- `12-ML基础/` (11) - Feature Engineering / Embedding-Fundamentals / Model Evaluation / Cross Validation / Imbalanced Data / Ensemble / Feature Selection / Hyperparameter Optimization / Data-Drift-Detection / Model-Performance-Monitor / ... - Feature Engineering / Model Evaluation / Cross Validation / Imbalanced Data / Ensemble / Feature Selection / Hyperparameter Optimization / Data-Drift-Detection / Model-Performance-Monitor / ...
- `13-广告分析/` (20) - Ad Attribution / ROAS / Search Intent / PVM / HGNN / GraphTrack / CABB / TRACE Delayed CVR / TESLA / FrontDoor / PIE / Identity Debiasing / Negative Keyword / Creative Fatigue / TikTok Attribution / Compliance Guardrail / Ad-to-Behavior Funnel / CDA Cookieless / Listing-Quality-Scoring / ...
- `14-用户分析/` (21) - Funnel / Cohort / AGRS / MAA / StaR / LACA / TRACE Clickstream / NonItem-Page / Trajectory / Traffic Source / Session Intent / Sparse Matrix / BlockEcho / STAMImputer / Utimac / PersonaBot / GPLR / Review-Pain-Point-Mining / ...
- `15-营销投放分析/` (11) - MMM / **DARA-Agentic-MMM** ★ / Promotion / Bayesian MMM / GenAudience LLM / Channel Saturation / Competitive Response / Multi-Objective Budget / Geo-Level / ... - MMM / DARA Agentic MMM / Promotion / Bayesian MMM / GenAudience LLM / Channel Saturation / Competitive Response / Multi-Objective Budget / Geo-Level / ...
- `16-智能体工程/` (45) - Auto Synthesis / MCP-A2A / Lifecycle / Context Compression / Agentic Memory / Active Pruning / Memory-as-Action / Task-Adaptive / Co-Evolutionary / Hermes Tool-Use / SLM Optimization / Tool Audit / MCP Benchmark / Long-Term Preference / EComStage / Orchestration Trace / Agent Safety / Fault Tolerance / Cost-Aware Scheduling / ...（共45）
- `17-价格优化/` (7) - Dynamic Pricing / Competitive Monitoring / Markdown / Bundle / Cross-Border / ...
- `18-物流履约/` (6) - Cross-Border Routing / Last-Mile Delivery / Returns / ...
- `19-风控反欺诈/` (6) - Review Fraud / Transaction Anomaly / Click Fraud / ...
- `20-AI视频生成/` (9) - AnchorCrafter / Phantom / Aquarius / BrandFusion / DAWN / E-Commerce Benchmark / Text-to-Edit / Virbo / **Brand-Video-Generation** ★
- `21-合规决策/` (6) - Category-Compliance-Prescan / ...
- **`22-数据采集工程/` (15)** ★ - LLM-Focused-Web-Crawling / Adaptive-Crawl / Market-Signal-Realtime-Collection / Realtime-Feature-Collection / Fake-Review-Detection / Document-Intelligence / Privacy-Safe-Identity / Federated-Collection / Synthetic-Data / Data-Quality-Assessment / Clickstream-Persona / Review-Dedup / Data-Provenance / Procurement-Email / Web-Page-Change-Detection

> ★ = 跨域桥梁 Skill / 新建领域（2026-06-04 ~ 2026-06-06）

> **粗体** Skill 为 2026-05-21~22 新增萃取.

### 项目管理
- `00-项目管理/项目总结.md` - 整体里程碑与最新状态
- `00-项目管理/进度追踪.md` - Week-by-week 进度
- `00-项目管理/next-papers-roadmap-v2.md` - Sprint 3-5 路线图
- `00-项目管理/Skill关联图谱.md` - 图谱可视化与缺口热力图

### 资源库
- `07-资源库/MasterPrompt.md` - Master Prompt 完整内容
- `07-资源库/关键词库.md` - 论文搜索关键词
- `07-资源库/审核问题库.md` - 质量审核维度
- `07-资源库/skill-aliases.json` - Skill 命名别名表(消除图谱漂移误判)

## 使用方法
1. 从 ArXiv 筛选论文(参考 `07-资源库/关键词库.md`)
2. 使用 Master Prompt 生成 Skill 卡片(`07-资源库/MasterPrompt.md`)
3. 质量审核后同步到各平台(`paper-同步` skill)
4. 用 `skills_graph_analyzer.py` 检查图谱状态与新缺口

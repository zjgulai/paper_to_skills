# Skills Graph 分析报告

**版本**: v3.1
**更新日期**: 2026-04-29
**同步基准**: Skill关联图谱.md v2.8

---

## 1. 图谱概览

- **节点总数**: 61 个技能（已萃取同步）
- **边总数**: ~200+ 条关系（前置/延伸/可组合）
- **领域数**: 9 个
- **层次层级**: 4 层（基础→进阶→专家→桥接）

### 领域分布

| 领域 | 代码 | 技能数 | 占比 |
|------|------|--------|------|
| 01-因果推断 | CI | 2 | 4% |
| 02-A/B实验 | AB | 3 | 5% |
| 03-时间序列 | TS | 3 | 5% |
| 04-供应链 | SC | 3 | 5% |
| 05-推荐系统 | RS | 3 | 5% |
| 06-增长模型 | GM | 9 | 16% |
| 07-NLP-VOC | NLP | 33 | 54% |
| 08-知识图谱 | KG | 2 | 4% |
| 09-DataAgent-LLM | DA | 3 | 5% |

### 技能清单

**01-因果推断 (2)**
- Uplift Modeling
- Intelligent Attribution (Causal Forest)

**02-A/B实验 (3)**
- A/B Experimental Design
- Multi-Armed Bandit
- Thompson Sampling MAB

**03-时间序列 (3)**
- Time Series Forecasting
- Intelligent Prediction (Doubly Robust)
- Temporal Fusion Transformer

**04-供应链 (4)**
- Multi-Echelon Inventory
- Two-Echelon Inventory DRL
- Monodense 单品价格弹性估计
- MARL多智能体动态定价 (交叉)

**05-推荐系统 (4)**
- Matrix Factorization
- Deep Learning Recommendation (HI)
- NeuralNDCG Learning to Rank
- MAS多目标推荐 (交叉)

**06-增长模型 (9)**
- Customer Churn Prediction
- Deep Learning Churn Prediction
- LTV Prediction (ZILN)
- User Lifecycle (STAN)
- DQN Purchase Prediction
- Uplift Churn Prediction
- Cold Start Product Recommendation
- New Product Opportunity Mining
- Customer Journey Prototype

**07-NLP-VOC (33)**
- ABSA / BERT-MoE ABSA / 大规模ABSA (3个ABSA相关)
- CSK Customer Sentiment Clustering
- NPS Driver Analysis
- AIPL-VOC Lifecycle Tags
- CrossLingual Sentiment Transfer
- ALCHEmist Weak Supervision
- Active Learning Annotation
- AdaNEN Streaming Classifier
- AutoTag Self-Evolving Label System
- Review Quality Scoring
- OpenWorld Class Incremental Learning
- TaxoAdapt Taxonomy Evolution
- REVISION 无点击意图挖掘
- Spiral of Silence 沉默挖掘
- TopicImpact 观点单元提取
- PERSONABOT RAG画像生成
- SoMeR 多视角表示
- GPLR 人群标签生成
- Kano 需求分类与优先级
- iReFeed 需求优先级排序
- TSCAN 上下文Uplift
- OfflineRL 触达时机优化
- MAA 行动建议生成
- StaR 观点语句排序
- AGRS 属性引导评论摘要
- TJAP 跨市场品类组合定价
- VOC-Proxy-NPS-AIPL 统一萃取引擎
- MAS消费者行为仿真 (arXiv:2510.18155)
- MAS多智能体VOC数据分析 (arXiv:2402.01386)
- MARL多智能体动态定价 (arXiv:2507.02698)
- MAS多目标推荐 (arXiv:2512.24325)

**08-知识图谱 (2)**
- Knowledge Graph for Skills Management
- GraphRAG Knowledge Enhanced Retrieval

**09-DataAgent-LLM (3)**
- DeepAnalyze Autonomous Data Science Agent
- Argos Agentic Anomaly Detection
- Data-to-Dashboard Multi-Agent Visualization

---

## 2. 中心性分析

### 核心枢纽技能 (高连接度)

| 排名 | 技能 | 角色 | 连接数 |
|-----|------|------|--------|
| 1 | Uplift Modeling | 因果推断入口 + 增长运营桥梁 | 7 |
| 2 | Kano 需求分类 | VOC→产品决策桥梁 | 6 |
| 3 | Churn Prediction | 用户增长基座 | 6 |
| 4 | SoMeR 多视角表示 | 用户画像基座 | 5 |
| 5 | Demand Forecasting | 预测分析枢纽 | 5 |

### 高价值无延伸技能（潜力点）

| 技能 | 业务价值 | 推荐延伸方向 |
|------|---------|------------|
| NeuralNDCG Learning to Rank | ⭐⭐⭐⭐⭐ | 序列推荐、重排序多样性 |
| Monodense 价格弹性 | ⭐⭐⭐⭐⭐ | ~~动态定价~~ ✅ 已完成(MARL)、促销优化 |
| iReFeed 优先级排序 | ⭐⭐⭐⭐ | 产品路线图自动化 |
| DQN Purchase Prediction | ⭐⭐⭐⭐ | 强化学习推荐 |

---

## 3. 知识缺口

### 已完成的旧缺口（已修复，无需再追踪）

以下缺口在 v2.7 中已标注完成：

- ✅ A/B Experimental Design — 已萃取
- ✅ DeepAnalyze 自主数据科学 Agent — 已萃取
- ✅ Argos Agentic 异常检测 — 已萃取
- ✅ NeuralNDCG Learning to Rank — 已萃取
- ✅ Monodense 单品价格弹性估计 — 已萃取
- ✅ Data-to-Dashboard 多Agent可视化 — 已萃取
- ✅ MAS多目标推荐 — 已萃取 (arXiv:2512.24325)
- ✅ MARL多智能体动态定价 — 已萃取 (arXiv:2507.02698)
- ✅ MAS消费者行为仿真 — 已萃取 (arXiv:2510.18155)
- ✅ MAS多智能体VOC数据分析 — 已萃取 (arXiv:2402.01386)

### 🔴 P0: 修复已标记但未完成的技能

- [ ] **iReFeed 需求优先级排序**
  - 状态: skill card 存在，但代码实现待验证
  - 缺口: Kano → 产品路线图的必经断点
  - 文件: `paper2skills-vault/07-NLP-VOC/Skill-iReFeed-需求优先级排序.md`

### 🟡 P1: 高优先级缺口

| 领域 | 缺口技能 | 类型 | 关联技能 |
|------|---------|------|---------|
| 05-推荐系统 | 序列推荐 Session-based Recommendation | 核心能力 | Matrix Factorization, NeuralNDCG |
| 05-推荐系统 | ~~多目标推荐 Multi-Task Learning~~ | ~~核心能力~~ | ~~Deep Learning Recommendation~~ ✅ 已覆盖(MAS) |
| 05-推荐系统 | 重排序 / 多样性控制 | 核心能力 | NeuralNDCG |
| 02-A/B实验 | 实验设计基础（Power, MDE, Sample Size） | 前置基础 | A/B Experimental Design |
| 04-供应链 | ~~动态定价模型~~ | ~~利润闭环~~ | ~~Monodense 价格弹性~~ ✅ 已萃取(MARL) |
| 01-因果推断 | 因果发现 Causal Discovery | 基座扩展 | Uplift Modeling |
| 08-知识图谱 | 实体抽取 / 关系推理 | 基座能力 | Knowledge Graph Management |

### 🟢 P2: 深度扩展

- ~~序列推荐 Session-based Recommendation~~ ✅ 已完成 (SR-GNN)
- ~~LLM 驱动个性化文案生成~~ ✅ 已完成
- ~~多目标推荐 Multi-Task Learning~~ ✅ 已完成 (MAS多目标推荐)
- ~~客服对话分析技能~~ ✅ 已完成 (MAS VOC数据分析师)
- ~~因果发现 (PC Algorithm)~~ ✅ 已完成

---

## 4. 推荐选题列表

| 优先级 | 选题 | 类型 | 搜索关键词 | 预期填补缺口 |
|-------|------|------|-----------|-------------|
| P0 | iReFeed 代码验证与补齐 | 修复断点 | `iReFeed priority ranking` | Kano→产品路线图 |
| P1 | 序列推荐 | 核心能力 | `session-based recommendation neural` | 05-推荐系统缺口 |
| ~~P1~~ | ~~多目标推荐~~ | ~~核心能力~~ | ~~`multi-task learning recommendation`~~ | ~~05-推荐系统缺口~~ ✅ 已完成(MAS) |
| ~~P1~~ | ~~因果发现~~ | ~~基座扩展~~ | ~~`causal discovery PC algorithm`~~ | ~~01-因果推断缺口~~ ✅ 已完成 |
| ~~P1~~ | ~~动态定价~~ | ~~利润闭环~~ | ~~`dynamic pricing reinforcement learning`~~ | ~~04-供应链缺口~~ ✅ 已完成(MARL) |
| ~~P2~~ | ~~LLM 个性化文案~~ | ~~营销策略~~ | ~~`LLM personalized copy generation`~~ | ~~07-NLP-VOC延伸~~ ✅ 已完成 |
| P2 | 实验设计基础 | 前置补齐 | `A/B test power analysis sample size` | 02-A/B实验缺口 |
| P2 | 重排序/多样性控制 | 核心能力 | `diversity re-ranking recommendation` | 05-推荐系统缺口 |

---

## 5. 跨领域组合推荐

| 组合 | 价值 | 场景 | 所需技能 |
|------|------|------|---------|
| Causal + LTV | ⭐⭐⭐⭐⭐ | 高价值用户精准归因 | Uplift + LTV Prediction |
| VOC + Churn | ⭐⭐⭐⭐ | 情感驱动流失预警 | CSK + Churn Prediction |
| 时序 + 库存 | ⭐⭐⭐⭐ | 预测驱动补货 | Demand Forecasting + Inventory |
| Uplift + MAB | ⭐⭐⭐⭐ | 动态实验优化 | Uplift + Multi-Armed Bandit |
| NeuralNDCG + REVISION | ⭐⭐⭐⭐⭐ | 搜索意图→排序决策 | 意图挖掘 + Learning to Rank |
| DeepAnalyze + Dashboard | ⭐⭐⭐⭐ | 分析结果自动可视化 | Data Science Agent + Visualization |
| Argos + Forecasting | ⭐⭐⭐⭐ | 预测异常自动告警 | 异常检测 + 时序预测 |
| P2 VOC分析 + P3 MARL定价 | ⭐⭐⭐⭐⭐ | 需求洞察→动态定价 | 多Agent分析 + MARL |
| P3 MARL定价 + P4 多目标推荐 | ⭐⭐⭐⭐⭐ | 实时价格→利润排序 | 动态定价 + 推荐系统 |
| P4 多目标推荐 + P1 行为仿真 | ⭐⭐⭐⭐⭐ | 推荐策略→仿真验证 | 推荐 + 消费者仿真 |
| MAS四技能全链路闭环 | ⭐⭐⭐⭐⭐ | 分析→决策→执行→验证 | P2+P3+P4+P1 |

---

## 6. 行动建议

1. **立即行动**: 验证 iReFeed skill card 的代码完整性，补齐断点
2. **本周计划**: MAS四方向已补齐，重点验证串联闭环效果
3. **本月目标**: 05-推荐系统剩余2缺口（序列推荐、重排序多样性），04-供应链已闭环
4. **持续维护**: 每月审查本报告与 Skill关联图谱.md 的一致性
5. **MAS升级**: P1-P4规则基线按标注路径升级为LLM/MARL生产版本

---

**维护者**: Claude Code
**更新频率**: 每月审查一次
**上游同步**: 以 `paper2skills-vault/00-项目管理/Skill关联图谱.md` 为权威来源

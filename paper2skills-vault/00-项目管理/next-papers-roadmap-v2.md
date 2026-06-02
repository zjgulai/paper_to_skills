---
title: Skills Graph 扩展路线图 v3 (Sprint 3-5)
doc_type: roadmap
module: project-management
status: active
created: 2026-05-21
updated: 2026-05-21
owner: self
source: ai
---

# Skills Graph 扩展路线图 v3 (Sprint 3-5)

> **v3 更新(2026-05-21)**: 基于图谱全面审计 (150 节点 / 758 边 / 105 MEDIUM 缺口) 制定三阶段扩展计划。
> **前置**: [next-papers-roadmap.md](./next-papers-roadmap.md) (v2, Sprint 3 原 6 个 P1 候选保留并纳入)
> **v1 历史**: 见 v2 文档中的 Sprint 1+2 完成记录。

---

## 零、当前状态审计 (2026-05-21)

### 图谱快照

| 指标 | 数值 |
|------|------|
| **节点总数** | 150 |
| **边总数** | 758 |
| **HIGH 缺口** | 0 (全部清零) |
| **MEDIUM 缺口** | 105 |
| **孤立 Skill** | 18 (全为 Round 4 桑基图新增) |
| **跨域桥梁缺失** | 87 |
| **领域稀疏度** | 2 个领域仅 1 个 Skill (11/12) |

### 领域分布 (按数量)

| 梯队 | 领域 | 数量 | 状态 |
|------|------|------|------|
| 🟢 饱和 | 14-用户分析 | 17 | 暂不扩展 |
| 🟢 饱和 | 16-智能体工程 | 16 | 仅补生产化缺口 |
| 🟢 饱和 | 06-增长模型 | 14 | 暂不扩展 |
| 🟢 饱和 | 08-知识图谱 | 14 | 仅补 GNN 基础 |
| 🟡 健康 | 13-广告分析 | 13 | 仅补桥梁 |
| 🟡 健康 | 10-MAS | 12 | 仅补安全/容错 |
| 🟡 健康 | 01-因果推断 | 11 | 仅补桥梁 |
| 🟡 健康 | 04-供应链 | 11 | 仅补桥梁 |
| 🟡 健康 | 05-推荐系统 | 11 | 仅补桥梁 |
| 🟠 偏薄 | 02-A_B实验 | 8 | 精补 CUPED/网络效应 |
| 🟠 偏薄 | 03-时间序列 | 8 | 精补 Conformal/协整 |
| 🟠 偏薄 | 09-DataAgent-LLM | 8 | 精补 RAG/NL2Dashboard |
| 🔴 薄弱 | 15-营销投放分析 | 5 | **重点补强** |
| 🔴 极薄 | 11-AI人文 | 1 | **基础建设** |
| 🔴 极薄 | 12-ML基础 | 1 | **基础建设 (P0)** |

### 工作流覆盖率

| 工作流 | 当前覆盖率 | Sprint 3 目标 | 备注 |
|--------|-----------|--------------|------|
| WF-A 智能补货 | 85%+ | 90%+ | 仅余 Conformal UQ + Inventory Pooling |
| WF-B 广告优化 | 75%+ | 85%+ | 有 4 个候选待萃取 |
| WF-C 客服分诊 | 70%+ | 80%+ | 有 Compliance Guardrail |
| WF-D 选品扫描 | **骨架** | **60%+** | **零 Skill 支撑，最大盲区** |
| WF-E Review 监控 | 85%+ | 85%+ | 暂不扩展 |

---

## 一、Sprint 3 — 基础建设 + 孤立修复 (2026-05-21 ~ 05-28)

> **核心目标**: 补齐 ML 基础层、修复 18 个孤立 Skill、完成 Sprint 3 原 6 个 P1 候选

### A 组: 12-ML基础 补强 (6 个新 Skill) — P0

> **理由**: `Skill-Feature-Engineering` 是图谱被依赖数 #1 (16 次依赖)，ML 基础薄导致大量伪缺口。

| # | Skill 名称 | 方向 | ArXiv 搜索关键词 | 代码目录 |
|---|-----------|------|------------------|---------|
| A1 | Skill-Model-Evaluation-Metrics | 模型评估体系 (ROC/AUC/PR/Calibration) | model evaluation classification regression metrics survey | `ml_fundamentals/model_evaluation/` |
| A2 | Skill-Cross-Validation-Strategies | 交叉验证策略 (K-Fold/Stratified/TimeSeries/Group) | cross-validation strategies time series grouped | `ml_fundamentals/cross_validation/` |
| A3 | Skill-Imbalanced-Data-Handling | 不平衡数据处理 (SMOTE/Class Weight/Threshold) | imbalanced learning SMOTE class imbalance | `ml_fundamentals/imbalanced_data/` |
| A4 | Skill-Ensemble-Methods | 集成学习 (Bagging/Boosting/Stacking/Blending) | ensemble methods stacking blending gradient boosting | `ml_fundamentals/ensemble/` |
| A5 | Skill-Feature-Selection | 特征选择 (SHAP/Boruta/Permutation/RFE) | feature selection SHAP Boruta permutation importance | `ml_fundamentals/feature_selection/` |
| A6 | Skill-Hyperparameter-Optimization | 超参调优 (Optuna/Bayesian/Hyperband) | hyperparameter optimization Optuna Bayesian | `ml_fundamentals/hyperparameter_tuning/` |

### B 组: 孤立 Skill 关联回填 (18 个修复，零新增 Skill) — P0

> **操作**: 编辑 18 个孤立 Skill 的"④ 技能关联"部分，添加 combinator/prerequisite 边。

| 批次 | 孤立 Skill | 数量 | 应添加的关联 |
|------|-----------|------|-------------|
| B1-流量转化 | TRACE-Clickstream / NonItem-Page / Trajectory-Pattern / Traffic-Source / Session-Intent-Shift | 5 | ← User-Funnel / Cohort-Retention / RFM |
| B2-广告归因 | HGNN-Cross-Device / GraphTrack / CABB / TESLA-NetCVR / TRACE-Delayed-CVR | 5 | ← Ad-Attribution / ROAS-Budget / Hierarchical-Search-Intent |
| B3-因果推断 | Conformal-ROI / EPICSCORE / SSBC / BCCB-Causal-Bandits | 4 | ← Uplift-Modeling / DML-Cohort / Causal-Forest |
| B4-缺失数据 | BlockEcho / Sparse-Matrix-Completion / STAMImputer / Utimac | 4 | ← Feature-Engineering / AB-Experimental-Design |

### C 组: Sprint 3 原 P1 候选 (6 个新 Skill)

> 来源: [next-papers-roadmap.md](./next-papers-roadmap.md) §0.1

| # | Skill 名称 | 服务工作流 | 论文方向 | 业务紧迫度 |
|---|-----------|-----------|---------|-----------|
| C1 | Skill-Negative-Keyword-Safe-Guard | WF-B (S03/S04) | Bayesian keyword performance estimation small sample | 高 |
| C2 | Skill-Creative-Fatigue-Detection | WF-B (S13) | Creative fatigue survival analysis digital advertising 2024 | 高 |
| C3 | Skill-Conformal-Prediction-Demand-UQ | WF-A (P15) | Conformal Prediction for Time Series 2023 | 中 |
| C4 | Skill-Multi-Channel-Inventory-Pooling | WF-A (P7) | Cross-channel inventory transshipment 2024 | 高 |
| C5 | Skill-Amazon-ToS-Compliance-Guardrail | WF-C + WF-B | LLM compliance guardrail e-commerce 2024 | 极高 |
| C6 | Skill-TikTok-Shop-Content-Attribution | WF-B (S16) | Interest-based commerce attribution social 2025 | 高 |

### D 组: CausalRAG 缺口填补 (1 个新 Skill) — P0

> **理由**: 最后 1 个 HIGH 缺口，阻塞 GraphRAG-Knowledge-Enhanced-Retrieval 下游链路。

| # | Skill 名称 | 方向 | 论文方向 |
|---|-----------|------|---------|
| D1 | Skill-CausalRAG-Knowledge-Retrieval | 因果增强检索 | causal retrieval augmented generation 2024 |

**Sprint 3 交付物**: +13 新 Skill / +18 关联修复 / 图谱预期: ~163 节点 / ~820 边

---

## 二、Sprint 4 — 新领域 + WF-D + 跨域桥梁 (2026-05-28 ~ 06-04)

> **核心目标**: 启动 3 个新业务领域 (价格/物流/风控)、填补 WF-D 选品扫描空白、建立 5 条关键跨域桥梁

### E 组: 17-价格优化 (Pricing) — 新领域 5 个 Skill

> **代码目录**: `paper2skills-code/pricing/` (需新建)
> **Vault 目录**: `paper2skills-vault/17-价格优化/` (需新建)

| # | Skill 名称 | 方向 | 论文方向 |
|---|-----------|------|---------|
| E1 | Skill-Dynamic-Pricing-Elasticity | 动态定价 + 需求弹性 | DRL dynamic pricing e-commerce demand elasticity |
| E2 | Skill-Competitive-Price-Monitoring | 竞品价格监测与响应 | competitive price monitoring matching algorithm |
| E3 | Skill-Markdown-Optimization | 折扣清仓定价优化 | markdown pricing inventory clearance optimization |
| E4 | Skill-Bundle-Pricing-Strategy | 捆绑定价策略 | bundle pricing combinatorial optimization consumer |
| E5 | Skill-Cross-Border-Price-Harmonization | 跨境价格协调 | multi-market pricing exchange rate arbitrage |

### F 组: 15-营销投放分析 补强 (4 个新 Skill)

| # | Skill 名称 | 方向 | 论文方向 |
|---|-----------|------|---------|
| F1 | Skill-Channel-Saturation-Curve | 渠道饱和曲线建模 | advertising saturation curve diminishing returns |
| F2 | Skill-Competitive-Response-Modeling | 竞争响应建模 | competitive advertising response game theory |
| F3 | Skill-Multi-Objective-Budget-Allocation | 多目标预算分配 | multi-objective budget allocation marketing optimization |
| F4 | Skill-Geo-Level-Marketing-Effectiveness | 地理级营销效果 | geo-level marketing effectiveness causal inference |

### G 组: WF-D 选品扫描 (4 个新 Skill) — P0 (最大盲区)

> **理由**: MAS 已有 SelectionAgent + selection_graph 骨架，但零 Skill 支撑。这是目前五大工作流中唯一没有 Skill 覆盖的。

| # | Skill 名称 | 方向 | 论文方向 |
|---|-----------|------|---------|
| G1 | Skill-Category-Trend-Forecasting | 品类趋势预测 | product category trend forecasting e-commerce time series |
| G2 | Skill-Competitor-Product-Intelligence | 竞品选品监测 | competitor product monitoring marketplace intelligence |
| G3 | Skill-Product-Opportunity-Scoring | 新品机会评分卡 | product opportunity identification scoring e-commerce |
| G4 | Skill-Supplier-Evaluation-Model | 供应商评估模型 | supplier evaluation multi-criteria decision |

### H 组: 核心跨域桥梁 (5 个新 Skill)

| # | 桥梁 | Skill 名称 | 论文方向 |
|---|------|-----------|---------|
| H1 | advertising ↔ user_analytics | Skill-Ad-to-Behavior-Funnel | ad click to purchase behavior modeling |
| H2 | time_series ↔ supply_chain | Skill-Forecast-Driven-Inventory | forecast-driven inventory optimization |
| H3 | causal_inference ↔ advertising | Skill-Causal-Attribution-Bridge | causal media attribution incrementality |
| H4 | mas ↔ llm_agent_engineering | Skill-Agent-Production-Engineering | agent production deployment engineering |
| H5 | knowledge_graph ↔ advertising | Skill-Audience-Knowledge-Graph | audience graph advertising targeting |

**Sprint 4 交付物**: +18 新 Skill / 3 个新领域 / WF-D 覆盖率 0%→60% / 图谱预期: ~181 节点 / ~960 边

---

## 三、Sprint 5 — 深化 + 精补 (2026-06-04 ~ 06-11)

> **核心目标**: 完成 18+19 物流/风控双新领域、MAS 生产化缺口、存量领域精补

### I 组: 18-物流履约 (Logistics) — 新领域 3 个 Skill

> **代码目录**: `paper2skills-code/logistics/` (需新建)
> **Vault 目录**: `paper2skills-vault/18-物流履约/` (需新建)

| # | Skill 名称 | 方向 | 论文方向 |
|---|-----------|------|---------|
| I1 | Skill-Cross-Border-Logistics-Routing | 跨境物流路径优化 | cross-border logistics route optimization 2024 |
| I2 | Skill-Last-Mile-Delivery-Prediction | 最后一公里配送时效 | last mile delivery time prediction e-commerce |
| I3 | Skill-Returns-Reverse-Logistics | 退货逆向物流 | reverse logistics returns prediction e-commerce |

### J 组: 19-风控反欺诈 (Risk & Fraud) — 新领域 3 个 Skill

> **代码目录**: `paper2skills-code/risk_fraud/` (需新建)
> **Vault 目录**: `paper2skills-vault/19-风控反欺诈/` (需新建)

| # | Skill 名称 | 方向 | 论文方向 |
|---|-----------|------|---------|
| J1 | Skill-Review-Fraud-Detection | 虚假评论检测 | fake review detection GNN graph neural network 2024 |
| J2 | Skill-Transaction-Anomaly-Detection | 异常交易检测 | anomalous transaction detection e-commerce isolation forest |
| J3 | Skill-Click-Fraud-Detection | 广告刷量检测 | click fraud detection advertising invalid traffic |

### K 组: MAS 生产化缺口 (3 个新 Skill)

| # | Skill 名称 | 方向 | 归属领域 |
|---|-----------|------|---------|
| K1 | Skill-Agent-Safety-Guardrails | Agent 安全对抗 (Prompt Injection/Jailbreak防御) | 16-智能体工程 |
| K2 | Skill-Agent-Fault-Tolerance | Agent 容错回退 (Fallback/CircuitBreaker/Retry) | 16-智能体工程 |
| K3 | Skill-Cost-Aware-Agent-Scheduling | 成本感知调度 (LLM→SLM动态降级) | 16-智能体工程 |

### L 组: 存量领域精补 (8 个新 Skill)

| # | Skill 名称 | 方向 | 归属领域 |
|---|-----------|------|---------|
| L1 | Skill-CUPED-Variance-Reduction | 方差缩减 CUPED/CUPAC | 02-A_B实验 |
| L2 | Skill-Network-Effect-Experiments | 网络效应干扰实验 (SUTVA违反场景) | 02-A_B实验 |
| L3 | Skill-Conformal-TS-Intervals | 时间序列 Conformal 预测区间 | 03-时间序列 |
| L4 | Skill-Multivariate-Cointegration | 多变量协整 VECM | 03-时间序列 |
| L5 | Skill-RAG-Enhanced-Data-Analysis | RAG 增强数据分析 | 09-DataAgent-LLM |
| L6 | Skill-NL2Dashboard-Automation | 自然语言→BI仪表盘自动生成 | 09-DataAgent-LLM |
| L7 | Skill-GNN-Foundations | GNN 基础 (GCN/GAT/GraphSAGE) | 08-知识图谱 |
| L8 | Skill-Sequential-AB-Testing | 序列化 A/B 检验 | 02-A_B实验 |

**Sprint 5 交付物**: +17 新 Skill / 2 个新领域 / 图谱预期: ~198 节点 / ~1100 边

---

## 四、总体执行计划

### 时间线与里程碑

```
Week 1 (Sprint 3) ─────────────────────────────────────────────────
│ A组 ML基础 ×6 │ B组 孤立回填 ×18 │ C组 P1候选 ×6 │ D组 CausalRAG │
│ 目标: 163节点 / 820边 / 领域稀疏度 0 (所有领域 ≥3)            │
└──────────────────────────────────────────────────────────────────

Week 2 (Sprint 4) ─────────────────────────────────────────────────
│ E组 价格优化 ×5 │ F组 营销补强 ×4 │ G组 WF-D ×4 │ H组 桥梁 ×5  │
│ 目标: 181节点 / 960边 / 3新领域 / WF-D 60%覆盖                 │
└──────────────────────────────────────────────────────────────────

Week 3 (Sprint 5) ─────────────────────────────────────────────────
│ I组 物流 ×3 │ J组 风控 ×3 │ K组 MAS ×3 │ L组 精补 ×8          │
│ 目标: 198节点 / 1100边 / 5新领域合计 / 图谱全面连接            │
└──────────────────────────────────────────────────────────────────
```

### 按类型统计

| 类型 | Sprint 3 | Sprint 4 | Sprint 5 | 合计 |
|------|----------|----------|----------|------|
| 新增 Skill | 13 | 18 | 17 | **48** |
| 关联修复 | 18 | 0 | 0 | **18** |
| 新领域 | 0 | 3 (17/18/19) | 2 (区分18/19) | **5** |
| 跨域桥梁 | 0 | 5 | 0 | **5** |

### 总目标

| 指标 | 当前 | 终态 | Δ |
|------|------|------|---|
| 节点数 | 150 | ~198 | +48 |
| 边数 | 758 | ~1100 | +342 |
| 领域数 | 15 | ~20 | +5 |
| 领域稀疏度 | 2 个=1 | 0 个≤3 | 清零 |
| 孤立 Skill | 18 | 0 | -18 |
| WF-D 覆盖率 | 0% | 60%+ | +60% |
| 预计新增 ROI | — | ~3000-5000 万/年 | — |

---

## 五、维护说明

- **双驱动**: 图谱自动审计 (`skills_graph_analyzer.py --gaps`) + 本文人工路线图
- **验收标准**: 每 Sprint 结束后重跑 `--analyze` 验证节点/边/缺口变化
- **文档联动**:
  - 新增领域 → 更新 [CLAUDE.md](../../CLAUDE.md) 的 Domain Mapping
  - 新增 Skill → 追加 [skill-aliases.json](../07-资源库/skill-aliases.json)
  - 完成 Sprint → 更新 [项目总结.md](./项目总结.md)

---

## 六、附录: v2 路线图中已完成项目 (不再重复)

以下为 v2 路线图中已完成或已外迁的项目:

- ✅ Sprint 1: AGRS / MAA / StaR / LACA (WF-E + WF-C)
- ✅ Sprint 2: HiFoReAd / Gen-QOT / Hierarchical-Search-Intent / Dial-In LLM / PVM / Bass
- ✅ 语义蓝图方向 1: Hierarchical Product KG Construction
- ✅ 语义蓝图方向 5: Customer Journey Decision Tree
- 📦 语义蓝图方向 2: VOC 语义蓝图 (已迁 ai_nlp_voc)
- ⏸️ 语义蓝图方向 4: AMR 跨语言对齐 (暂缓)

---

## Sprint 4 — 业务完整性 P0 补缺 + 新领域 (2026-05-25 起)

> **触发**: 2026-05-25 业务完整性深度审计，识别 3 大盲区 6 个 P0 Skill。
> **新建领域**: `21-合规决策` (Category compliance prescan, regulatory risk pre-entry)
> **当前图谱基线**: 205 节点 / 1852 边 / 22 个领域

### 审计结论

| 链路 | 审计覆盖率 | 核心短板 |
|---|---|---|
| WF-A 智能补货 | 78% | 库存可见性、促销解耦 |
| WF-B 广告优化 | 72% | **Listing 质量评分**是最大断层 |
| WF-D 选品扫描 | **15%** | **全图谱最大盲区** |
| 横切面（模型生产化） | **20%** | 107个ML模型全部裸跑无监控 |

---

### 执行两条并行线

```
业务 Skill 线 (WF-D × 3 + WF-B × 1)          治理 Skill 线 (横切面 × 2)
────────────────────────────────────           ─────────────────────────
批次1(并行): B2-PLC + C1-Listing               A0: 生产模型监控现状调研
     ↓ B2完成                                       ↓
批次2(并行): B3-合规预筛 + B1-TAM/SAM          A1: Data-Drift-Detection
     ↓ 全部完成                                     ↓
图谱验证 + 同步                                 A2: Model-Performance-Monitor
```

---

### P0 组: 业务 Skill 线 (4 个新 Skill)

#### B2 · Skill-Product-Lifecycle-Stage
- **领域**: `06-增长模型`
- **代码目录**: `paper2skills-code/growth_model/product_lifecycle_stage/`
- **业务锚点**: WF-D 进入时机判断（UV-C 消毒器品类当前阶段定位）
- **论文关键词**: `product lifecycle stage classification sales pattern Bass diffusion 2024`
- **Skill 卡片必须包含**:
  - 四阶段量化判断规则 (YoY增速 / 竞品数量增长率 / 价格压缩率)
  - PELT 变点检测代码识别阶段切换时间点
  - 进入时机建议矩阵 (阶段 × 竞争密度 → GO/WAIT/NO-GO)
  - 母婴跨境场景示例（baby sterilizer UV-C 品类）
- **图谱关联**:
  - prerequisite ← `Skill-Bass-Diffusion-New-Product-Forecasting`, `Skill-Time-Series-Anomaly-Detection`
  - extends → `Skill-Category-Trend-Forecasting`
  - combinable ↔ `Skill-Market-Size-Estimation` (B1), `Skill-Competitor-Product-Intelligence`
- **审核评分维度** (各2.5分，需≥7/10):
  - 算法原理覆盖度 / 业务场景具体度 / 代码模板可运行性 / 图谱关联完整性

#### C1 · Skill-Listing-Quality-Scoring
- **领域**: `13-广告分析`
- **代码目录**: `paper2skills-code/advertising/listing_quality_scoring/`
- **业务锚点**: WF-B 广告 ROAS 上限由 Listing 质量决定；WF-D 选品维度之一
- **论文关键词**: `product listing quality optimization Amazon e-commerce conversion NLP 2024`
- **Skill 卡片必须包含**:
  - Listing 质量 7 维评分体系（标题关键词密度/Bullet points/描述完整度/图片数量质量/A+内容/Review质量/Q&A覆盖率）
  - NLP 标题质量评分代码（关键词密度 + 语义相关性）
  - CV 图片质量评分代码（分辨率/背景/产品占比）
  - baby sterilizer listing 改版前后评分对比示例
  - Listing 质量 → 预期 CVR 提升映射关系
- **图谱关联**:
  - extends → `Skill-Ad-Attribution-Modeling`, `Skill-ROAS-Budget-Optimization`
  - combinable ↔ `Skill-Hierarchical-Search-Intent-Classification`, `Skill-Creative-Fatigue-Detection`

#### B3 · Skill-Category-Compliance-Prescan
- **领域**: `21-合规决策` ← **新建领域**
- **代码目录**: `paper2skills-code/compliance/category_compliance_prescan/`
- **业务锚点**: WF-D 选品前预判合规门槛（合规既是护城河也是陷阱）
- **论文关键词**: `product safety compliance risk assessment e-commerce FDA recall prediction 2023 2024`
- **Skill 卡片必须包含**:
  - 美国(FDA/CPSC/CPSIA) + 欧盟(CE/REACH/RoHS) 双轨合规速查表
  - 历史召回率 → 品类风险等级映射（低/中/高）
  - NLP 抓取 FDA enforcement database 的代码
  - baby sterilizer UV-C 合规预筛完整示例（含实际召回案例）
  - 合规成本估算模板（认证费 × 时间 × 通过率）
- **图谱关联**:
  - combinable ↔ `Skill-Amazon-ToS-Compliance-Guardrail`（运营期合规互补）
  - combinable ↔ `Skill-Product-Opportunity-Scoring`, `Skill-Supplier-Evaluation-Model`
  - prerequisite ← `Skill-Product-Lifecycle-Stage` (B2，成熟/衰退期合规特征不同)

#### B1 · Skill-Market-Size-Estimation
- **领域**: `06-增长模型`
- **代码目录**: `paper2skills-code/growth_model/market_size_estimation/`
- **业务锚点**: WF-D 进入新品类前的市场容量判断，是 Product-Opportunity-Scoring 的前置输入
- **论文关键词**: `market size estimation TAM SAM bottom-up top-down e-commerce Google Trends 2023 2024`
- **Skill 卡片必须包含**:
  - Top-down vs Bottom-up 两种方法对比
  - Google Trends 指数 → 绝对市场规模转换校准方法
  - baby sterilizer 品类 TAM 估算完整示例（含实际数字）
  - 置信区间输出（非点估计）
- **图谱关联**:
  - prerequisite ← `Skill-Bass-Diffusion-New-Product-Forecasting`
  - extends → `Skill-Product-Opportunity-Scoring`, `Skill-Category-Trend-Forecasting`
  - combinable ↔ `Skill-Competitor-Product-Intelligence`, `Skill-Product-Lifecycle-Stage` (B2)

---

### P0 组: 治理 Skill 线 (2 个新 Skill)

#### A0（前置调研，非 Skill 萃取）
- **内容**: 盘点当前生产 ML 模型清单
  - WF-A: 需求预测模型 (TFT/Prophet)
  - WF-B: MAB 素材测试模型
  - WF-F: Uplift/LTV 模型
- **记录**: 每个模型当前监控现状（无监控/日志告警/无自动重训）
- **输出**: `paper2skills-vault/00-项目管理/ml-models-monitoring-inventory.md`
- **目的**: 确保 A1/A2 Skill 卡片的业务场景章节基于真实生产模型，而非通用 ML 教材

#### A1 · Skill-Data-Drift-Detection
- **领域**: `12-ML基础`
- **代码目录**: `paper2skills-code/ml_fundamentals/data_drift_detection/`
- **业务锚点**: WF-A 需求预测模型上线后 feature drift / WF-F Uplift 模型用户行为分布漂移
- **论文关键词**: `concept drift detection ADWIN PSI data stream machine learning survey 2023 2024`
- **Skill 卡片必须包含**:
  - 统计检验漂移 vs 模型性能漂移的区别
  - PSI (Population Stability Index) 计算代码
  - ADWIN 滑动窗口实现
  - 母婴跨境季节性 vs 真实漂移区分方案
- **图谱关联**:
  - prerequisite ← `Skill-Feature-Engineering`, `Skill-Model-Evaluation-Metrics`
  - extends → `Skill-Model-Performance-Monitor` (A2)
  - combinable ↔ `Skill-Time-Series-Anomaly-Detection`

#### A2 · Skill-Model-Performance-Monitor
- **领域**: `12-ML基础`
- **代码目录**: `paper2skills-code/ml_fundamentals/model_performance_monitor/`
- **业务锚点**: 全链路 — 所有生产 ML 模型的健康看板
- **论文关键词**: `ML model monitoring production AUC degradation shadow mode champion challenger 2024`
- **Skill 卡片必须包含**:
  - 监控指标体系: AUC 衰减 / PSI / CSI / 预测分布漂移
  - Shadow mode vs Canary deployment 决策树
  - 告警 → 回滚 → 重训自动化流程代码框架
  - 大促期间模型性能基线隔离方案
- **图谱关联**:
  - prerequisite ← `Skill-Data-Drift-Detection` (A1), `Skill-Cross-Validation-Strategies`
  - combinable ↔ `Skill-AB-Experimental-Design`, `Skill-Argos-Agentic-Anomaly-Detection`

---

### 执行 Todo（按批次）

#### 阶段 0: 新领域初始化
- [ ] 创建 `paper2skills-vault/21-合规决策/` 目录 ✅ (2026-05-25)
- [ ] 更新 CLAUDE.md 领域映射（目录树 + domain 表） ✅ (2026-05-25)
- [ ] 创建 `paper2skills-code/compliance/` 目录
- [ ] 更新 `sync.py` DOMAINS 列表加入 `compliance → 21-合规决策`
- [ ] 更新 `skill-aliases.json` 预留 compliance 领域条目

#### 阶段 1: 治理线前置调研
- [ ] **A0** 执行生产 ML 模型监控现状调研，输出 `ml-models-monitoring-inventory.md`

#### 阶段 2: 业务线批次1（并行）
- [ ] **B2** 论文选题: Product Lifecycle Stage (关键词: `product lifecycle stage classification Bass diffusion 2024`)
- [ ] **C1** 论文选题: Listing Quality Scoring (关键词: `product listing quality Amazon conversion NLP 2024`)
- [ ] **B2** 萃取 `Skill-Product-Lifecycle-Stage`，审核 ≥7/10
- [ ] **C1** 萃取 `Skill-Listing-Quality-Scoring`，审核 ≥7/10
- [ ] B2+C1 各自完成后立即同步 vault + code，验证图谱连通性

#### 阶段 3: 业务线批次2（B2完成后）
- [ ] **B3** 论文选题: Category Compliance Prescan (关键词: `product safety compliance FDA recall prediction 2024`)
- [ ] **B3** 萃取 `Skill-Category-Compliance-Prescan`，审核 ≥7/10，归入 `21-合规决策`
- [ ] **B1** 论文选题: Market Size Estimation (关键词: `TAM SAM market size estimation e-commerce 2024`)
- [ ] **B1** 萃取 `Skill-Market-Size-Estimation`，审核 ≥7/10
- [ ] B3+B1 各自完成后立即同步，验证图谱连通性

#### 阶段 4: 治理线（A0完成后并行推进）
- [ ] **A1** 论文选题: Data Drift Detection (关键词: `concept drift detection ADWIN PSI 2024`)
- [ ] **A1** 萃取 `Skill-Data-Drift-Detection`，审核 ≥7/10
- [ ] A1 完成后立即同步，验证图谱连通性
- [ ] **A2** 论文选题: Model Performance Monitor (关键词: `ML model monitoring production champion challenger 2024`)
- [ ] **A2** 萃取 `Skill-Model-Performance-Monitor`，审核 ≥7/10
- [ ] A2 完成后立即同步，验证图谱连通性

#### 阶段 5: Sprint 4 收尾验证
- [ ] 运行 `python scripts/skills_graph_analyzer.py --analyze` 输出最终图谱快照
- [ ] 验收标准:
  - 节点 ≥211 (+6)
  - HIGH 缺口 = 0
  - WF-D 覆盖率 ≥40% (从 15% 提升)
  - `21-合规决策` 领域 Skill 数 ≥1
  - 6 个新 Skill 各自关联边 ≥3
- [ ] 更新 `进度追踪.md` Sprint 4 完成状态
- [ ] 更新 CLAUDE.md Recent Skills Added 节

---

### 论文选题 Fallback 规则

若 ArXiv 无评分 ≥7/10 的候选论文（特别是合规预筛方向），允许使用：
1. 工业白皮书（Amazon Seller Central 官方文档、FDA 技术指南）
2. 顶会 Workshop 论文（AAAI/ICML/KDD Workshop）
3. 综合萃取（≥3 篇相关博客/报告 + 1 篇技术论文的混合来源，需在 Skill 卡片注明来源类型）

---

### 审核评分维度（4维，各2.5分，需≥7/10）

| 维度 | 满分 | 达标标准 |
|---|---|---|
| 算法原理覆盖度 | 2.5 | 能用自己语言重述核心算法，有公式或伪代码 |
| 业务场景具体度 | 2.5 | 有母婴跨境/DTC 具体场景示例，有量化指标 |
| 代码模板可运行性 | 2.5 | Python 代码在示例数据上可直接运行，有 requirements |
| 图谱关联完整性 | 2.5 | prerequisite/extends/combinable 各≥1条，指向已存在 Skill |

**通过标准**: 总分 ≥7/10 且无维度低于 1 分

**Sprint 4 预计交付**: +6 新 Skill / +1 新领域(21-合规决策) / 图谱预期: ~211 节点 / ~1900 边 / WF-D 15%→40%

---
## 缺口分析更新 — 2026-06-01

**图谱状态**：246 节点 / 2773 边 / HIGH 缺口 0 / MEDIUM 缺口 122

### P0 — 立即启动

| Skill 候选 | 领域交叉 | 填补类型 | 关键词 |
|---|---|---|---|
| Causal Churn Intervention | 因果推断 × 增长模型 | missing_extension(Churn) | `uplift modeling churn prevention 2025` |
| Contextual Bandit for Pricing | A/B实验 × 价格优化 | missing_extension(MAB) | `contextual bandit pricing e-commerce 2025` |
| Recall Risk Prediction | 21-合规决策 | 薄领域(1 Skill) | `product recall prediction ML 2025 2026` |
| Returns Demand Forecasting | 18-物流履约 | 薄领域(3 Skill) | `reverse logistics returns forecasting 2024` |
| Fake Review Graph Detection | 19-风控反欺诈 | 薄领域(3 Skill) | `review fraud graph neural network 2025` |

### P1 — 本月内（跨域桥梁）

| Skill 候选 | 领域交叉 | 搜索关键词 |
|---|---|---|
| Causal AB Testing CUPED+ | 因果 × A/B | `CUPED variance reduction causal 2024` |
| Supply Chain Causal Attribution | 因果 × 供应链 | `causal supply chain attribution 2024` |
| CausalRAG | 因果 × 知识图谱 | `CausalRAG causal knowledge graph 2025` |
| Agentic Causal Analysis | 因果 × MAS | `LLM agent causal reasoning 2026` |
| Causal MMM Bayesian | 因果 × 营销 | `causal marketing mix modeling 2025` |

### 图谱修复（已执行）
- Skill-Customer-Churn-Prediction → Skill-Uplift-Churn-Prediction (extends) ✅
- Skill-Multi-Armed-Bandit → Thompson-Sampling-MAB + BCCB-Causal-Bandits (extends) ✅

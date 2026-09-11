# paper2skills Skill 关联图谱

**版本**: v2.8
**更新日期**: 2026-04-29
**覆盖技能数**: 52 个已萃取同步技能

---

## 技能总览

| 领域 | 数量 | 已萃取技能 |
|------|------|-----------|
| 01-因果推断 | 3 | Uplift Modeling, Causal Forest, Doubly Robust Estimation |
| 02-A/B实验 | 3 | A/B Experimental Design, Multi-Armed Bandit, Thompson Sampling |
| 03-时间序列 | 3 | Demand Forecasting, Doubly Robust Estimation, Temporal Fusion Transformer |
| 04-供应链 | 4 | Demand Forecasting, Multi-Echelon Inventory, Two-Echelon Inventory DRL, Monodense价格弹性, TJAP跨市场定价 |
| 05-推荐系统 | 3 | Matrix Factorization, Deep Learning Recommendation (HI), NeuralNDCG Learning to Rank |
| 06-增长模型 | 9 | Churn Prediction, New Product Opportunity Mining, Cold Start Recommendation, LTV Prediction, Deep Learning Churn Prediction, User Lifecycle STAN, Customer Journey Prototype, DQN Purchase Prediction, Uplift Churn Prediction |
| 07-NLP-VOC | 32 | 大规模ABSA, BERT-MoE ABSA, CSK情感聚类, NPS驱动因素分析, AIPL-VOC生命周期标签, CrossLingual情感迁移, REVISION无点击意图, Spiral of Silence沉默挖掘, TopicImpact观点单元, PERSONABOT RAG画像, SoMeR多视角表示, GPLR人群标签, Kano需求分类, iReFeed优先级排序, TSCAN上下文Uplift, OfflineRL触达时机, MAA行动建议, StaR观点排序, AGRS评论摘要, TJAP跨市场定价, AdaNEN流式分类, AutoTag自进化标签, ActiveLearning标注, ALCHEmist弱监督, OpenWorld增量学习, ReviewQuality评分, VOC统一萃取引擎, VOC语义蓝图, 产品属性图谱, 客服对话决策图, 行为意图树, 跨语言语义对齐, MAS消费者行为仿真, MAS多智能体VOC数据分析, MARL多智能体动态定价, MAS多目标推荐 |
| 08-知识图谱 | 4 | Knowledge Graph for Skills Management, GraphRAG Knowledge Enhanced Retrieval, KG Auto Construction Agent-Driven, Dense Retrieval Ecommerce Semantic Search |
| 09-DataAgent-LLM | 3 | DeepAnalyze 自主数据科学Agent, Argos Agentic异常检测, Data-to-Dashboard 多Agent可视化 |

---

## 商业决策闭环图谱

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              商业决策全链路 Skill Graph                            │
└─────────────────────────────────────────────────────────────────────────────────┘

【获客与流量】
    REVISION (搜索意图挖掘)
           │
           ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  Uplift      │◄────►│    MAB       │◄────►│  Thompson    │
    │  Modeling    │      │              │      │  Sampling    │
    └──────┬───────┘      └──────┬───────┘      └──────────────┘
           │                     ▲
           │            ┌────────┘
           │            ▼
           │     ┌──────────────┐
           └────►│   A/B Exp    │
                 │   Design     │
                 └──────────────┘
           │
           ▼
【增长与运营】
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  LTV         │◄────►│ Churn        │◄────►│ Uplift       │
    │  Prediction  │      │ Prediction   │      │ Churn        │
    └──────┬───────┘      └──────┬───────┘      └──────────────┘
           │                     │
           ▼                     ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  STAN 生命周期│      │ Customer     │      │ DQN 购买预测  │
    │              │      │ Journey      │      │              │
    └──────────────┘      └──────────────┘      └──────────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 ▼
【推荐与搜索】
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │   Matrix     │◄────►│ Deep Learning│◄────►│ Cold Start   │
    │Factorization │      │ Recommendation│     │Recommendation│
    └──────────────┘      └──────────────┘      └──────────────┘
                                 ▲
                                 │
                         ┌───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  NeuralNDCG  │
                  │ Learning to  │
                  │    Rank      │
                  └──────┬───────┘
                         │
    REVISION (搜索意图) ───┘

                                 │
                                 ▼
【供应链与定价】
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │Demand        │─────►│ Inventory    │◄────►│Two-Echelon  │
    │Forecasting   │      │Optimization  │      │Inventory DRL│
    └──────┬───────┘      └──────┬───────┘      └──────────────┘
           │                     │
           │              ┌──────┘
           │              ▼
           │       ┌──────────────┐
           └──────►│   Monodense  │
                   │ 单品价格弹性  │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │     TJAP     │
                   │ 跨市场组合定价 │
                   └──────────────┘
                                 │
                                 ▼
【VOC 与产品决策】（最完整链路）
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  REVISION    │      │Spiral of     │      │     CSK      │      │   AdaNEN     │
    │  搜索意图     │      │Silence       │      │  情感聚类     │      │ 流式分类/漂移 │
    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
           │                     │                     │                     │
           └─────────────────────┼─────────────────────┘                     │
                                 │                                           │
                                 ▼                                           ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │ TopicImpact  │      │ PERSONABOT   │      │    SoMeR     │      │ AutoTag/     │
    │ 观点单元提取  │      │ RAG画像生成   │      │ 多视角表示    │      │ OpenCML      │
    └──────┬───────┘      └──────────────┘      └──────┬───────┘      └──────┬───────┘
           │                                           │                     │
           │              ┌────────────────────────────┘                     │
           │              ▼                                                     │
           │       ┌──────────────┐                                             │
           │       │    GPLR      │                                             │
           │       │ 人群标签生成  │                                             │
           │       └──────┬───────┘                                             │
           │              │                                                     │
           │              ▼                                                     │
           │       ┌──────────────┐      ┌──────────────┐                       │
           └──────►│     Kano     │─────►│   iReFeed    │                       │
                   │ 需求分类      │      │ 优先级排序    │                       │
                   └──────┬───────┘      └──────┬───────┘                       │
                          │                     │                               │
                          ▼                     ▼                               │
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                │
    │    AIPL      │      │  NPS Driver  │      │   MAA/StaR   │                │
    │ 生命周期标签  │      │  驱动因素分析  │      │  行动/摘要   │                │
    └──────┬───────┘      └──────┬───────┘      └──────────────┘                │
           │                     │                                              │
           │                     ▼                                              │
           │              ┌──────────────┐      ┌──────────────┐                │
           └─────────────►│   AGRS       │◄────►│   TSCAN      │                │
                          │ 评论摘要生成  │      │ 挽回策略     │                │
                          └──────────────┘      └──────────────┘                │
                                                                                │
                          【分布监控层】AdaNEN 实时监控分类质量，触发模型更新告警 │

【数据智能监控】（新增链路）
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │    Argos     │─────►│  DeepAnalyze │─────►│ Data-to-    │
    │Agentic异常检测│      │ 自主数据科学  │      │ Dashboard   │
    └──────┬───────┘      └──────────────┘      └──────────────┘
           │                                      │
           │                                      ▼
           │                               ┌──────────────┐
           │                               │ 自动可视化   │
           │                               │ 仪表板生成   │
           │                               └──────────────┘
           │
           ├──────────────────────────────────────────────────────┐
           │                                                     │
           ▼                                                     ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │ Demand       │      │  VOC数据     │      │ 用户行为     │
    │ Forecasting  │      │ 异常监控     │      │ 异常检测     │
    └──────────────┘      └──────────────┘      └──────────────┘

【用户运营闭环】
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │    TSCAN     │◄────►│  OfflineRL   │      │              │
    │ 上下文Uplift  │      │ 触达时机优化  │      │              │
    └──────────────┘      └──────────────┘      └──────────────┘

【MAS多智能体系统层】(NEW — 4技能闭环)
    ┌──────────────┐
    │  P2 VOC数据  │
    │  分析师(27A) │
    │  主题/模式   │
    └──────┬───────┘
           │ 需求主题/痛点模式
           ▼
    ┌──────────────┐      ┌──────────────┐
    │  P3 MARL     │◄────►│  P4 多目标   │
    │  动态定价    │价格   │  推荐        │
    │  MADDPG/QMIX │      │ AWRQ-Mixer   │
    └──────┬───────┘      └──────┬───────┘
           │                    │ 推荐策略
           │ 定价策略           │
           ▼                    ▼
    ┌─────────────────────────────────────┐
    │      P1 消费者行为仿真              │
    │  虚拟Agent市场 → 策略预演与验证     │
    │  输出: cannibalization/增量/忠诚    │
    └─────────────────────────────────────┘
           │
           └────────────► 回传洞察 → 优化P2/P3/P4

    上游衔接: PERSONABOT画像 → Agent初始画像
             Monodense弹性  → MARL环境参数
             DQN购买预测    → ConversionAgent打分
    下游衔接: P4 → 首页/购物车/邮件推荐
             P3 → TJAP跨市场定价实时调整
             P1 → 大促前策略A/B仿真筛选
```

---

## 关键链路说明

### 链路 1: VOC → 产品决策（已高度完整）

```
评论/搜索/行为数据
    → REVISION/Spiral of Silence/CSK/ABSA (基础VOC层)
    → TopicImpact/PERSONABOT/SoMeR (画像萃取层)
    → GPLR (人群标签)
    → Kano (需求分类)
    → [iReFeed 优先级排序] ← 断点，待补齐
    → 产品路线图
```

**关联强度**: ⭐⭐⭐⭐⭐  
**业务价值**: 从用户声音到产品规划的直接闭环，预期 ROI 20倍+

### 链路 2: 预测 → 库存 → 定价（基础闭环，缺定价基座）

```
Demand Forecasting → Inventory Optimization → Two-Echelon Inventory DRL
                            ↓
                         TJAP (跨市场组合定价)
```

**问题**: TJAP 是高阶应用，缺少基础**价格弹性模型**或**动态定价**作为 Demand Forecasting 到利润最大化的桥梁。

### 链路 3: 用户生命周期运营（较完整）

```
STAN (生命周期阶段)
    → Customer Journey Prototype (旅程序列分析)
    → DQN Purchase Prediction (购买意图预测)
    → Churn Prediction / Uplift Churn Prediction (流失预警)
    → Matrix Factorization / HI (个性化推荐挽留)
    → TSCAN + OfflineRL (挽回策略 + 触达时机)
```

### 链路 4: 实验与优化（已补齐基础）

```
A/B Experimental Design (实验设计基础)
    → Multi-Armed Bandit (动态分配)
    → Thompson Sampling (在线优化)
    → Uplift Modeling (效果评估)
```

### 链路 5: 搜索与推荐（已补齐）

```
REVISION (无点击意图挖掘)
    → NeuralNDCG (Learning to Rank)
    → Deep Learning Recommendation (HI)
    → Matrix Factorization
```

**状态**: ✅ Learning to Rank 已补齐。NeuralNDCG 通过可微分排序松弛直接优化 NDCG，将 REVISION 识别的搜索意图转化为排序决策。

### 链路 6: 数据异常监控 → 自动分析 → 可视化（已完整）

```
多源业务数据（销售/库存/VOC/用户行为）
    → Argos Agentic异常检测（自适应规则生成 + 准确率保证）
    → DeepAnalyze 自主数据科学Agent（异常触发 → 根因分析 → 报告生成）
    → Data-to-Dashboard 多Agent可视化（洞察 → 图表 → 仪表板）
    → 业务决策
```

**关联强度**: ⭐⭐⭐⭐⭐  
**业务价值**: 从被动监控转向主动洞察，Argos 识别异常后自动触发 DeepAnalyze 深度分析，再由 D2D 生成可视化仪表板，全程无需人工介入  
**关键连接**:
- Argos → Demand Forecasting: 销量异常检测 → 预测模型校准信号
- Argos → VOC链路: 评论量/情感异常 → 触发TopicImpact/GPLR分析
- Argos → 增长模型: 用户行为异常 → 触发Churn预警
- DeepAnalyze → Kano: 深度分析报告 → 产品需求输入
- DeepAnalyze → Data-to-Dashboard: 分析报告 → 自动可视化呈现
- Data-to-Dashboard → 管理层: 可视化仪表板 → 决策视图

---

## 核心技能关联矩阵

### 高连接度 Hub Skills

| 排名 | 技能 | 角色 | 直接关联数 |
|------|------|------|-----------|
| 1 | **Uplift Modeling** | 因果推断核心枢纽 | 7 |
| 2 | **Demand Forecasting** | 预测分析基础 | 6 |
| 3 | **Churn Prediction** | 用户增长枢纽 | 6 |
| 4 | **Kano 需求分类** | VOC→产品决策桥梁 | 5 |
| 5 | **MAA 行动建议生成** | VOC 输出端枢纽 | 5 |
| 6 | **SoMeR 多视角表示** | 用户画像基座 | 5 |
| 7 | **DeepAnalyze** | 数据智能分析中枢 | 5 |
| 8 | **P4 MAS多目标推荐** | 策略输出枢纽 | 4 |

### 跨领域组合推荐

| 组合名称 | 技能 A | 技能 B | 组合效果 | 应用场景 |
|---------|--------|--------|---------|---------|
| 精准广告优化 | Uplift Modeling | MAB | 真实增量 + 动态探索 | Facebook/Google 素材测试 |
| 智能补货决策 | Demand Forecasting | Inventory Optimization | 预测驱动库存 | 海外仓备货计划 |
| 流失挽留闭环 | Churn Prediction | Matrix Factorization | 识别风险 + 个性化推荐 | 沉默用户激活 |
| VOC驱动产品决策 | TopicImpact | Kano | 主题提取 + 需求分类 | 季度产品路线图 |
| 人群精准营销 | SoMeR | GPLR | 多视角嵌入 + 可解释标签 | 差异化人群包 |
| 上下文挽回 | TSCAN | OfflineRL | 策略选择 + 时机优化 | 流失用户触达 |
| 实验严谨加速 | A/B Exp Design | CUPED | 样本量规划 + 方差缩减 | 缩短实验周期并保证功效 |
| 在线实验优化 | A/B Exp Design | Thompson Sampling | 离线设计 + 在线探索 | 首页推荐策略测试 |
| 评论洞察流水线 | StaR | AGRS | 去重排序 + 结构化摘要 | 季度复盘报告 |
| NPS归因→需求分类 | NPS Driver | Kano | 归因痛点 + 需求分层 | 产品改进优先级 |
| 贬损者挽回闭环 | NPS Driver | TSCAN | 识别贬损者 + 上下文挽回 | 流失用户挽回 |
| 生命周期精准运营 | AIPL VOC | OfflineRL | 阶段识别 + 时机优化 | 分阶段触达策略 |
| 新品培育闭环 | AIPL VOC | DQN Purchase | 阶段分群 + 购买预测 | 新品冷启动 |
| 多语言舆情一体化 | CrossLingual | NPS Driver | 一套模型覆盖全球 + 归因 | 出海市场监控 |
| 全球用户画像 | CrossLingual | PERSONABOT | 多语言VOC → 统一画像 | 全球化用户洞察 |
| 流式分类自适应 | AdaNEN | InsightNet | 漂移检测 + 高精度静态分类 | 季节性评论分类 |
| 开放世界流式分类 | AdaNEN | OpenCML | 分布漂移 + 新类别发现 | 新品上市监控 |
| 实时情绪监控 | AdaNEN | TSCAN | 分布漂移检测 + 挽回策略匹配 | 大促期情绪监控 |
| 异常→自动分析 | Argos | DeepAnalyze | 异常检测触发 → 深度报告生成 | 销售异常根因分析 |
| 预测校准预警 | Demand Forecasting | Argos | 预测基线 + 自适应异常检测 | 需求预测偏差监控 |
| 评论异常洞察 | AdaNEN | Argos | 分类质量监控 + 数据异常检测 | VOC数据质量告警 |
| 智能产品决策 | DeepAnalyze | Kano | 深度数据分析 → 需求分层 | 季度产品规划 |
| 用户异常挽留 | Argos | Churn Prediction | 行为异常检测 + 流失预警 | 沉默用户识别 |
| 分析+可视化闭环 | DeepAnalyze | Data-to-Dashboard | 深度分析 + 自动图表生成 | 周报自动生成 |
| 异常监控可视化 | Argos | Data-to-Dashboard | 异常检测 + 实时监控面板 | 业务异常告警面板 |
| VOC洞察可视化 | TopicImpact | Data-to-Dashboard | 话题提取 + 可视化呈现 | 产品复盘报告 |
| 预测趋势可视化 | Demand Forecasting | Data-to-Dashboard | 预测结果 + 趋势图表 | 需求预测仪表板 |
| 搜索意图→排序 | REVISION | NeuralNDCG | 意图识别 + 排序优化 | 智能搜索结果排序 |
| MAS分析→定价 | P2 VOC分析师 | P3 MARL定价 | 需求主题→弹性修正 | 大促全链路决策 |
| 定价→推荐联动 | P3 MARL定价 | P4 多目标推荐 | 实时价格→利润排序 | 利润最大化推荐 |
| 推荐→仿真验证 | P4 多目标推荐 | P1 行为仿真 | 推荐策略→预演验证 | 推荐策略迭代 |
| VOC→定价驱动 | P2 VOC分析师 | P3 MARL定价 | 需求洞察→动态调价 | 需求驱动定价 |
| MAS全链路闭环 | P2+P3+P4+P1 | — | 分析→决策→执行→验证 | 大促自动化决策 |
| 仿真→产品决策 | P1 行为仿真 | Kano | 市场响应→需求验证 | 产品改进优先级验证 |

---

## 技能缺口热力图（基于真实状态更新）

| 领域 | 前置缺口 | 延伸缺口 | 桥梁缺口 | 总计 |
|------|---------|---------|---------|------|
| 01-因果推断 | ██░░░░░░ | ████░░░░ | ██░░░░░░ | 2 |
| 02-A/B实验 | ████░░░░ | ████████ | ████░░░░ | **4** |
| 03-时间序列 | ██░░░░░░ | ████░░░░ | ████░░░░ | 3 |
| 04-供应链 | ████░░░░ | ████████ | ██████░░ | 4 |
| 05-推荐系统 | ██████░░ | ███████░ | ██████░░ | **5** |
| 06-增长模型 | ██░░░░░░ | ████░░░░ | ████░░░░ | 3 |
| 07-NLP-VOC | ██░░░░░░ | ██░░░░░░ | ██░░░░░░ | 2 |
| 08-知识图谱 | ██████░░ | ████████ | ███████░ | 5 |
| 09-DataAgent-LLM | ████░░░░ | ████████ | ████░░░░ | 4 |

**高密度缺口领域**: 02-A/B实验、05-推荐系统、08-知识图谱  
**新增领域**: 09-DataAgent-LLM（刚引入，关联待扩展）

**最低密度缺口领域**: 07-NLP-VOC（闭环最完整）

---

## 优先级萃取清单

### P0: 修复已标记但未完成的技能
- [x] **iReFeed 需求优先级排序** (arXiv:2603.28677) ✅ 已验证
  - 状态: skill card 完整 + 代码已落地 + 测试通过
  - 文件: `Skill-iReFeed-需求优先级排序.md`
  - 代码: `paper2skills-code/nlp_voc/irefeed_priority_ranking/model.py`
  - 缺口: Kano → 产品路线图断点已修复

### P1: 补全商业闭环核心缺口（已更新）
- [x] **DeepAnalyze 自主数据科学Agent** (09-DataAgent-LLM)
  - 状态: 已完成 (`Skill-DeepAnalyze-Autonomous-Data-Science-Agent.md`)
  - 覆盖: 五动作编排架构，端到端数据分析报告生成
  - 关联: VOC分析、增长模型、供应链分析
- [x] **Argos Agentic异常检测** (09-DataAgent-LLM)
  - 状态: 已完成 (`Skill-Argos-Agentic-Anomaly-Detection.md`)
  - 覆盖: 三Agent协作规则生成，时序异常检测，准确率保证
  - 关联: 预测校准、用户行为监控、VOC质量告警
- [x] **NeuralNDCG Learning to Rank** (05-推荐系统)
  - 状态: 已完成 (`Skill-NeuralNDCG-Learning-to-Rank.md`)
  - 覆盖: 可微分排序松弛直接优化NDCG，三种LTR范式对比
  - 关联: REVISION搜索意图 → NeuralNDCG排序决策 → DLR推荐
- [x] **动态定价 / 价格弹性模型** (04-供应链)
  - 状态: 已完成 (`Skill-Monodense-单品价格弹性估计.md`)
  - 覆盖: Monodense DLM 单品级价格弹性估计，无需对照实验

### P2: 深度扩展
- [x] **Data-to-Dashboard 多Agent可视化** (09-DataAgent-LLM)
  - 状态: 已完成 (`Skill-Data-to-Dashboard-Multi-Agent-Visualization.md`)
  - 覆盖: 两阶段多Agent架构（Data-to-Insight + Insight-to-Chart），Tree-of-Thoughts 可视化选择
  - 关联: DeepAnalyze 分析结果 → 自动可视化仪表板
- [x] **序列推荐 Session-based Recommendation** (05-推荐系统) ✅ 已完成 (SR-GNN)
- [x] **LLM 驱动个性化文案生成** (07-NLP-VOC 营销策略层) ✅ 已完成
- [x] **多目标推荐 Multi-Objective Recommendation** (05-推荐系统 / 07-NLP-VOC)
  - 状态: 已完成 (`Skill-MAS-Multi-Objective-Recommendation.md`)
  - 覆盖: AWRQ-Mixer多Agent协调，CTR/CVR/利润/多样性四目标平衡
  - 关联: DQN购买预测 → ConversionAgent; PERSONABOT画像 → 匹配打分
- [x] **MARL多智能体动态定价** (04-供应链 / 07-NLP-VOC)
  - 状态: 已完成 (`Skill-MAS-MARL-Dynamic-Pricing.md`)
  - 覆盖: MADDPG/MADQN/QMIX多Agent竞争定价，5策略对比仿真
  - 关联: Monodense弹性 → MARL环境; TJAP → 实时调价执行
- [x] **MAS消费者行为仿真** (07-NLP-VOC)
  - 状态: 已完成 (`Skill-MAS-Consumer-Behavior-Simulation.md`)
  - 覆盖: 虚拟消费者Agent市场仿真，促销效果预演
  - 关联: PERSONABOT画像 → Agent初始化; P3/P4 → 策略验证
- [x] **MAS多智能体VOC数据分析** (07-NLP-VOC)
  - 状态: 已完成 (`Skill-MAS-VOC-Data-Analyst.md`)
  - 覆盖: 27-Agent定性分析管道，主题/编码/模式/洞察/验证/报告
  - 关联: VOC语义蓝图 → Agent输入; Kano/iReFeed → 下游决策
- [x] **KG Auto Construction Agent-Driven** (08-知识图谱)
  - 状态: 已完成 (`Skill-KG-Auto-Construction-Agent-Driven.md`)
  - 覆盖: 三阶段Agent驱动框架（本体创建→精炼→填充），从电商产品描述自动构建KG
  - 关联: VOC语义蓝图/产品属性图谱 → 数据源; GraphRAG → 下游应用
  - 实验: 291商品97%成功率，97.1%属性覆盖率
- [x] **Dense Retrieval Ecommerce Semantic Search** (08-知识图谱)
  - 状态: 已完成 (`Skill-Dense-Retrieval-Ecommerce-Semantic-Search.md`)
  - 覆盖: 四阶段语义搜索（查询解析→稠密检索→约束过滤→交叉编码器重排序）
  - 关联: GraphRAG → 检索加速层; KG Auto Construction → 结构化过滤; VOC语义蓝图 → 语义理解层
  - 场景: 母婴商品语义搜索、GraphRAG加速、跨语言检索

---

## 应用优先级路线图

### Phase 1: 立即行动（本周）
1. 补齐 **iReFeed**，释放 VOC→产品决策完整链路
2. 更新本图谱文档，确保与真实状态一致

### Phase 2: 短期补全（2-4周）
3. ~~萃取 **A/B 实验设计基础**，建立实验严谨性~~ ✅ 已完成
4. ~~萃取 **DeepAnalyze + Argos**，建立数据智能监控链路~~ ✅ 已完成
5. ~~萃取 **动态定价模型**，补齐供应链利润闭环~~ ✅ 已完成
6. ~~萃取 **Learning to Rank**，完善搜索推荐能力~~ ✅ 已完成

### Phase 3: 中期扩展（1-2月）
7. ~~探索 **Data-to-Dashboard**，将 DeepAnalyze 分析结果自动转化为管理层可视化仪表板~~ ✅ 已完成
8. ~~探索 **序列推荐** 和 **文案生成**，提升转化率~~ ✅ 已完成
9. ~~引入 **客服对话分析** 技能，扩展 VOC 数据源~~ ✅ 已完成
10. ~~引入 **MAS多智能体系统** 4技能，建立分析→决策→执行→验证闭环~~ ✅ 已完成

### Phase 4: 深度整合（持续）
11. MAS规则基线升级为LLM/MARL生产版本
12. 建立P2→P3→P4→P1自动化闭环流水线
13. 跨技能A/B测试框架验证联动效果

---

**维护者**: Claude Code  
**更新频率**: 每月审查一次

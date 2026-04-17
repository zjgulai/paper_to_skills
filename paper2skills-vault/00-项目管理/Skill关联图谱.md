# paper2skills Skill 关联图谱

**版本**: v2.3  
**更新日期**: 2026-04-17  
**覆盖技能数**: 43 个已萃取同步技能

---

## 技能总览

| 领域 | 数量 | 已萃取技能 |
|------|------|-----------|
| 01-因果推断 | 3 | Uplift Modeling, Causal Forest, Doubly Robust Estimation |
| 02-A/B实验 | 3 | A/B Experimental Design, Multi-Armed Bandit, Thompson Sampling |
| 03-时间序列 | 3 | Demand Forecasting, Doubly Robust Estimation, Temporal Fusion Transformer |
| 04-供应链 | 4 | Demand Forecasting, Multi-Echelon Inventory, Two-Echelon Inventory DRL, Monodense价格弹性, TJAP跨市场定价 |
| 05-推荐系统 | 2 | Matrix Factorization, Deep Learning Recommendation (HI) |
| 06-增长模型 | 9 | Churn Prediction, New Product Opportunity Mining, Cold Start Recommendation, LTV Prediction, Deep Learning Churn Prediction, User Lifecycle STAN, Customer Journey Prototype, DQN Purchase Prediction, Uplift Churn Prediction |
| 07-NLP-VOC | 22 | 大规模ABSA, BERT-MoE ABSA, CSK情感聚类, NPS驱动因素分析, AIPL-VOC生命周期标签, CrossLingual情感迁移, REVISION无点击意图, Spiral of Silence沉默挖掘, TopicImpact观点单元, PERSONABOT RAG画像, SoMeR多视角表示, GPLR人群标签, Kano需求分类, iReFeed优先级排序, TSCAN上下文Uplift, OfflineRL触达时机, MAA行动建议, StaR观点排序, AGRS评论摘要, TJAP跨市场定价 |
| 08-知识图谱 | 2 | Knowledge Graph for Skills Management, GraphRAG Knowledge Enhanced Retrieval |

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
    REVISION (搜索意图) ─────────┘

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
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  REVISION    │      │Spiral of     │      │     CSK      │
    │  搜索意图     │      │Silence       │      │  情感聚类     │
    └──────┬───────┘      └──────┬───────┘      └──────┬───────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 │
                                 ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │ TopicImpact  │      │ PERSONABOT   │      │    SoMeR     │
    │ 观点单元提取  │      │ RAG画像生成   │      │ 多视角表示    │
    └──────┬───────┘      └──────────────┘      └──────┬───────┘
           │                                           │
           │              ┌────────────────────────────┘
           │              ▼
           │       ┌──────────────┐
           │       │    GPLR      │
           │       │ 人群标签生成  │
           │       └──────┬───────┘
           │              │
           │              ▼
           │       ┌──────────────┐      ┌──────────────┐
           └──────►│     Kano     │─────►│   iReFeed    │
                   │ 需求分类      │      │ 优先级排序    │
                   └──────┬───────┘      └──────┬───────┘
                          │                     │
                          ▼                     ▼
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │    AIPL      │      │  NPS Driver  │      │   MAA/StaR   │
    │ 生命周期标签  │      │  驱动因素分析  │      │  行动/摘要   │
    └──────┬───────┘      └──────┬───────┘      └──────────────┘
           │                     │
           │                     ▼
           │              ┌──────────────┐      ┌──────────────┐
           └─────────────►│   AGRS       │◄────►│   TSCAN      │
                          │ 评论摘要生成  │      │ 挽回策略     │
                          └──────────────┘      └──────────────┘

【用户运营闭环】
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │    TSCAN     │◄────►│  OfflineRL   │      │              │
    │ 上下文Uplift  │      │ 触达时机优化  │      │              │
    └──────────────┘      └──────────────┘      └──────────────┘
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

### 链路 5: 搜索与推荐（有缺口）

```
REVISION (无点击意图挖掘)
    → [Learning to Rank] ← 缺失
    → Deep Learning Recommendation (HI)
    → Matrix Factorization
```

**问题**: REVISION 识别了搜索意图，但缺少将意图转化为排序决策的**搜索排序学习**技能。

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

---

## 技能缺口热力图（基于真实状态更新）

| 领域 | 前置缺口 | 延伸缺口 | 桥梁缺口 | 总计 |
|------|---------|---------|---------|------|
| 01-因果推断 | ██░░░░░░ | ████░░░░ | ██░░░░░░ | 2 |
| 02-A/B实验 | ████░░░░ | ████████ | ████░░░░ | **4** |
| 03-时间序列 | ██░░░░░░ | ████░░░░ | ████░░░░ | 3 |
| 04-供应链 | ████░░░░ | ████████ | ██████░░ | 4 |
| 05-推荐系统 | ████████ | ████████ | ████████ | **8** |
| 06-增长模型 | ██░░░░░░ | ████░░░░ | ████░░░░ | 3 |
| 07-NLP-VOC | ██░░░░░░ | ██░░░░░░ | ██░░░░░░ | 2 |
| 08-知识图谱 | ████████ | ████████ | ████████ | 6 |

**高密度缺口领域**: 02-A/B实验、05-推荐系统、08-知识图谱

**最低密度缺口领域**: 07-NLP-VOC（闭环最完整）

---

## 优先级萃取清单

### P0: 修复已标记但未完成的技能
- [ ] **iReFeed 需求优先级排序** (arXiv:2603.28677)
  - 状态: VOC桥接图谱标记为"已萃取"，但实际无 skill card、无代码
  - 缺口: Kano → 产品路线图的必经断点
  - 预期产出: `Skill-iReFeed-需求优先级排序.md`

### P1: 补全商业闭环核心缺口
- [x] **A/B 实验设计基础** (02-A/B实验)
  - 状态: 已完成 (`Skill-AB-Experimental-Design.md`)
  - 覆盖: 连续/二分类样本量、相对提升 Delta method、Power/MDE、分层随机分配、CUPED
- [ ] **Learning to Rank** (05-推荐系统)
  - 缺口: REVISION 有意图但无排序决策能力
  - 关联: 搜索→推荐的桥梁
- [x] **动态定价 / 价格弹性模型** (04-供应链)
  - 状态: 已完成 (`Skill-Monodense-单品价格弹性估计.md`)
  - 覆盖: Monodense DLM 单品级价格弹性估计，无需对照实验

### P2: 深度扩展
- [ ] **序列推荐 Session-based Recommendation** (05-推荐系统)
- [ ] **LLM 驱动个性化文案生成** (07-NLP-VOC 营销策略层)
- [ ] **多目标推荐 Multi-Task Learning** (05-推荐系统)

---

## 应用优先级路线图

### Phase 1: 立即行动（本周）
1. 补齐 **iReFeed**，释放 VOC→产品决策完整链路
2. 更新本图谱文档，确保与真实状态一致

### Phase 2: 短期补全（2-4周）
3. ~~萃取 **A/B 实验设计基础**，建立实验严谨性~~ ✅ 已完成
4. 萃取 **Learning to Rank**，完善搜索推荐能力
5. ~~萃取 **动态定价模型**，补齐供应链利润闭环~~ ✅ 已完成

### Phase 3: 中期扩展（1-2月）
6. 探索 **序列推荐** 和 **文案生成**，提升转化率
7. 引入 **客服对话分析** 技能，扩展 VOC 数据源

---

**维护者**: Claude Code  
**更新频率**: 每月审查一次

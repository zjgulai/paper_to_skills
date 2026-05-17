# Skills Graph 分析报告

**版本**: v5.0
**更新日期**: 2026-05-15
**同步基准**: 全库 Skill 卡片扫描

---

## 1. 图谱概览

- **节点总数**: 116 个技能（已萃取同步）
- **边总数**: ~400+ 条关系（前置/延伸/可组合）
- **领域数**: 15 个
- **层次层级**: 4 层（基础→进阶→专家→桥接）

### 领域分布

| 领域 | 代码 | 技能数 | 占比 | 变化 |
|------|------|--------|------|------|
| 01-因果推断 | CI | 6 | 5% | +3 (v3.1→v4.0) |
| 02-A/B实验 | AB | 5 | 4% | +2 |
| 03-时间序列 | TS | 5 | 4% | +2 |
| 04-供应链 | SC | 5 | 4% | +2 |
| 05-推荐系统 | RS | 8 | 7% | +3 |
| 06-增长模型 | GM | 10 | 9% | +1 |
| 07-NLP-VOC | NLP | 43 | 37% | — |
| 08-知识图谱 | KG | 9 | 8% | +6 |
| 09-DataAgent-LLM | DA | 5 | 4% | +2 |
| 10-MAS | MAS | 12 | 10% | — |
| 11-AI人文 | AH | 1 | 1% | +1 |
| 12-ML基础 | ML | 1 | 1% | +1 |
| 13-广告分析 | AD | 2 | 2% | **新增** |
| 14-用户分析 | UA | 2 | 2% | **新增** |
| 15-营销投放分析 | MK | 2 | 2% | **新增** |

**分布变化趋势**：
- NLP-VOC 占比从 44% 降至 37%（绝对数量不变，新领域稀释效应）
- 新增 5 个领域（11-12 为补齐，13-15 为业务扩展），领域覆盖从 10 个 → 15 个
- 13-15 为跨境电商核心运营领域，与现有技术领域形成"技术→运营"闭环

---

## 2. 新增技能详情（2026-05-15）

### 11-AI人文（+1）

| 技能 | 核心论文 | 填补缺口 |
|------|---------|---------|
| AI Tech × Healing Quotes | StructLoRA/InfLoRA/Prompt Tuning/Cross-Modal Transfer | 技术概念的人文表达 |

### 12-ML基础（+1）

| 技能 | 核心论文 | 填补缺口 |
|------|---------|---------|
| Feature Engineering | 综合多篇基础论文 | ML pipeline 前置基础 |

### 13-广告分析（+2）【新增领域】

| 技能 | 核心论文 | 业务场景 |
|------|---------|---------|
| Ad Attribution Modeling | Shapley Value + Markov Chain (综合多篇) | 多触点归因，避免Last-Click陷阱 |
| ROAS Budget Optimization | 边际ROAS均衡 + 花费-收入曲线拟合 | 跨渠道预算最优分配 |

**业务闭环**：Attribution（归因）→ ROAS（预算分配）→ MMM（宏观验证）

### 14-用户分析（+2）【新增领域】

| 技能 | 核心论文 | 业务场景 |
|------|---------|---------|
| User Funnel Analysis | 漏斗分析 + PrefixSpan + 马尔可夫链 | 用户行为路径诊断 |
| Cohort Retention Analysis | 幂律留存 + BG/NBD 模型 | 新用户留存追踪与预测 |

**业务闭环**：Funnel（行为诊断）→ Cohort（留存追踪）→ RFM（价值分层）→ LTV（长期价值）

### 15-营销投放分析（+2）【新增领域】

| 技能 | 核心论文 | 业务场景 |
|------|---------|---------|
| Marketing Mix Modeling | Google Meridian / Meta Robyn / DeepCausalMMM | 宏观预算增量效应估计 |
| Promotion Effectiveness | DoorDash Causal ML (KDD 2025) | 促销真实增量评估 |

**业务闭环**：MMM（年度规划）→ Promotion（活动评估）→ Attribution（渠道归因）→ ROAS（执行优化）

---

## 3. 中心性分析

### 核心枢纽技能（高连接度）

| 排名 | 技能 | 角色 | 连接数 | 变化 |
|-----|------|------|--------|------|
| 1 | Uplift Modeling | 因果推断入口 + 增长运营桥梁 | 7 | — |
| 2 | Ad Attribution Modeling | 广告预算决策入口 | 6 | **新增** |
| 3 | Kano 需求分类 | VOC→产品决策桥梁 | 6 | — |
| 4 | Churn Prediction | 用户增长基座 | 6 | — |
| 5 | DiD | 自然实验评估枢纽 | 5 | — |
| 6 | SoMeR 多视角表示 | 用户画像基座 | 5 | — |
| 7 | Demand Forecasting | 预测分析枢纽 | 5 | — |

**新增枢纽**：Ad Attribution Modeling 连接 13-广告分析 / 15-营销投放分析 / 01-因果推断 / 14-用户分析，成为运营侧核心枢纽。

### 孤立技能检测（低连接度，需加强关联）

| 技能 | 领域 | 问题 |
|------|------|------|
| Monodense 价格弹性 | 04-供应链 | 仅与 MARL 定价有连接，与 05-推荐系统/01-因果推断缺少关联 |
| Temporal Fusion Transformer | 03-时间序列 | 与 04-供应链/06-增长模型的连接薄弱 |
| Feature Engineering | 12-ML基础 | 与各领域技能缺少显式关联，需建立"前置基础"连接 |

---

## 4. 知识缺口（更新）

### 🔴 P0: 已修复的缺口（本次行动）

- ✅ **01-因果推断结构性薄弱** — 已修复：从 2 张 → 6 张
- ✅ **05-推荐系统多样性/冷启动/召回缺口** — 已修复：从 4 张 → 8 张
- ✅ **08-知识图谱实体抽取/关系推理** — 已修复：从 2 张 → 9 张
- ✅ **13-15 业务运营领域缺失** — 已修复：新增 3 个领域 6 张 Skill

### 🟡 P1: 仍存在的结构性缺口

| 领域 | 缺口技能 | 类型 | 关联技能 | 紧迫度 |
|------|---------|------|---------|--------|
| 03-时间序列 | N-BEATS / N-HiTS 神经时序 | 深度方法 | Temporal Fusion Transformer | 中 |
| 04-供应链 | 需求预测（基础） | 前置基础 | Inventory, Monodense | 高 |
| 07-NLP-VOC | 代码模板覆盖率 0% | 工程缺口 | 全部 VOC Skill | 高 |
| 10-MAS | 与 15-营销投放的 Agent 应用 | 业务结合 | MMM Agent / 自动预算调整 | 中 |
| 12-ML基础 | 模型评估、超参调优 | 前置基础 | Feature Engineering | 中 |

### 🟢 P2: 深度扩展（可选）

- 因果推断：Synthetic Control（合成控制法）、Regression Discontinuity（断点回归）
- 推荐系统：联邦推荐、可解释推荐
- 广告分析：Incrementality-Based Attribution（增量归因）
- 用户分析：Customer Journey Mapping、Predictive CLV
- 营销投放：Geo-Lift 实验设计、动态创意优化（DCO）

---

## 5. 推荐选题列表（更新）

| 优先级 | 选题 | 类型 | 搜索关键词 | 预期填补缺口 |
|-------|------|------|-----------|-------------|
| P1 | 需求预测基础 | 前置补齐 | `demand forecasting baseline moving average exponential smoothing` | 04-供应链 |
| P1 | 时序异常检测 | 应用场景 | `time series anomaly detection transformer` | 03-时间序列 |
| P1 | ML模型评估体系 | 前置基础 | `model evaluation metrics cross validation hyperparameter tuning` | 12-ML基础 |
| P2 | N-BEATS 神经时序 | 深度方法 | `N-BEATS neural time series forecasting` | 03-时间序列 |
| P2 | 增量归因实验 | 进阶方法 | `geo lift experiment incrementality attribution` | 13-广告分析 |
| P2 | 动态创意优化 | 进阶应用 | `dynamic creative optimization DCO ad personalization` | 15-营销投放 |

---

## 6. 跨领域组合推荐（更新）

### 新增高价值组合（本次补全后）

| 组合 | 价值 | 场景 | 所需技能 |
|------|------|------|---------|
| Attribution + MMM | ⭐⭐⭐⭐⭐ | 归因给出微观分配，MMM验证宏观增量 | Ad Attribution + Marketing Mix Modeling |
| Funnel + Cohort | ⭐⭐⭐⭐⭐ | 漏斗看行为步骤流失，Cohort看时间维度留存 | User Funnel + Cohort Retention |
| ROAS + Promotion | ⭐⭐⭐⭐⭐ | 日常预算优化 + 活动期效果评估 | ROAS Optimization + Promotion Effectiveness |
| DiD + Mediation | ⭐⭐⭐⭐⭐ | 政策/算法更新"效应多大+为什么" | 自然实验 + 机制分解 |
| IV + Causal Forest | ⭐⭐⭐⭐⭐ | 价格弹性"因果估计+异质性" | 工具变量 + 异质性效应 |
| RPG + Multilingual NER | ⭐⭐⭐⭐⭐ | 跨语言商品理解与检索 | 语义ID + 实体识别 |

### 运营-技术桥接组合（新维度）

| 组合 | 价值 | 场景 | 所需技能 |
|------|------|------|---------|
| Cohort + LTV | ⭐⭐⭐⭐⭐ | 留存曲线→LTV预测→预算决策 | Cohort Retention + LTV Prediction |
| Attribution + Uplift | ⭐⭐⭐⭐⭐ | "谁参与了"→"谁被影响了" | Ad Attribution + Uplift Modeling |
| MMM + DML | ⭐⭐⭐⭐ | 宏观渠道效应→微观用户效应 | MMM + Promotion Effectiveness |

---

## 7. 代码模板覆盖检查

| 领域 | 技能数 | 有代码模板 | 覆盖率 |
|------|--------|-----------|--------|
| 01-因果推断 | 6 | Uplift ✓, DiD ✓, IV ✓, Mediation ✓ | 67% |
| 02-A/B实验 | 5 | Power Analysis ✓ | 20% |
| 03-时间序列 | 5 | Forecasting ✓, Anomaly ✓, Prophet ✓ | 60% |
| 04-供应链 | 5 | Demand Forecasting ✓, Safety Stock ✓ | 40% |
| 05-推荐系统 | 8 | MAB ✓ | 13% |
| 06-增长模型 | 10 | RFM ✓ | 10% |
| 07-NLP-VOC | 43 | ❌ 无 | 0% |
| 08-知识图谱 | 9 | graph_rag.py ✓, KGQA ✓ | 22% |
| 09-DataAgent | 5 | SQL Agent ✓, RCA ✓ | 40% |
| 10-MAS | 12 | ❌ 无 | 0% |
| 11-AI人文 | 1 | ❌ 无 | 0% |
| 12-ML基础 | 1 | Feature Engineering ✓ | 100% |
| 13-广告分析 | 2 | Attribution ✓, ROAS ✓ | 100% |
| 14-用户分析 | 2 | Funnel ✓, Cohort ✓ | 100% |
| 15-营销投放 | 2 | MMM ✓, Promotion ✓ | 100% |

**关键发现**：
- 新领域（13-15）代码覆盖率 100%，每个 Skill 都有完整代码模板
- 07-NLP-VOC 和 10-MAS 代码覆盖率仍为 0%，是下一步重点
- 整体代码覆盖率从 ~15% 提升到 ~35%

---

## 8. 行动建议

### 立即行动（本周）

1. **07-NLP-VOC 代码模板补全**：选择 3-5 个核心 Skill 补代码（如 Kano、Aspect Sentiment、TopicImpact）
2. **10-MAS 代码模板**：为 AutoGen / ReAct 补最小可运行示例

### 本月目标

1. **P1 技能补全**：需求预测基础、N-BEATS 神经时序
2. **代码模板覆盖率提升**：从当前 ~35% 提升到 ~50%
3. **13-15 运营领域深化**：每个领域再补 1-2 张 Skill（如 13-增量归因、14-用户旅程映射、15-Geo-Lift）

### 持续维护

1. **每月扫描**：运行脚本自动统计各领域 Skill 数量，生成分布报告
2. **孤立技能关联**：为 Monodense、TFT、Feature Engineering 建立跨领域连接
3. **论文跟踪**：关注 2025-2026 年 KDD/ICML/NeurIPS/ACL/WWW 会议中母婴电商相关论文
4. **运营-技术桥接**：鼓励从 13-15 向技术领域的跨域组合选题

---

**维护者**: Claude Code
**更新频率**: 每月审查一次
**上游同步**: 以全库 `Skill-*.md` 文件扫描为权威来源

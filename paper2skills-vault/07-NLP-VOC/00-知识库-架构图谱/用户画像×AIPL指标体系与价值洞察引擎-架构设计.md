---
title: 用户画像 × AIPL 指标体系与价值洞察引擎 - 架构设计
doc_type: architecture
module: nlp-voc
topic: persona-aipl-insight-engine
status: stable
created: 2026-04-28
updated: 2026-04-28
owner: self
source: ai
---

# 用户画像 × AIPL 指标体系与价值洞察引擎

## 1. 架构概述

### 1.1 设计目标

将现有 30+ NLP-VOC 技能的分散输出，融合为**以用户画像 × AIPL 为骨架的统一指标体系**，并在此基础上实现**监控-分析-决策三层价值洞察**。

### 1.2 核心问题

| 问题 | 现状 | 目标 |
|------|------|------|
| 技能输出碎片化 | 各技能独立输出，缺乏统一视图 | 所有技能输出汇聚到画像×AIPL矩阵 |
| 洞察依赖人工 | 分析师手工整合多技能结果 | 自动异常检测+根因分析+策略推荐 |
| 策略到执行断层 | 洞察到行动没有闭环 | 自动生成行动队列并路由到责任部门 |

### 1.3 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据源层 (已有)                                    │
│  退货留言 / 客服工单 / 商品评论 / Trustpilot / 搜索日志 / 行为数据            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VOC 统一萃取引擎 (已有+改造)                          │
│                                                                             │
│  【改造】原生输出画像×AIPL结构化数据                                          │
│  ├── 376 标签种子匹配 → AIPL 标签                                          │
│  ├── 55 原子画像标签 → 6维画像分组 (WHO/WHY/WHAT/WHEN/HOW/EMOTION)          │
│  ├── 情感校准 → 情感极性/强度                                              │
│  ├── Proxy NPS 计算                                                        │
│  └── 业务元数据聚合 (策略包/主责部门/优先级)                                 │
│                                                                             │
│  输出: VOCLabelExtraction (含 persona_dimensions 字段)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                     画像 × AIPL 指标体系矩阵层 (新增)                        │
│                                                                             │
│  ┌─────────────┬────┬────┬────┬────┬────┬────┬────┐                        │
│  │             │ A  │ I  │ P1 │ P2 │ L1 │ L2 │ L3 │                        │
│  ├─────────────┼────┼────┼────┼────┼────┼────┼────┤                        │
│  │ WHO 人群身份 │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │  ← count              │
│  │ WHY 决策动机 │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │  ← avg_sentiment      │
│  │ WHAT 关注方面│ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │  ← proxy_nps          │
│  │ WHEN 使用场景│ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │  ← mention_rate       │
│  │ HOW 行为模式 │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │  ← top_themes         │
│  │ EMOTION 情感 │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │ ●  │  ← sentiment_dist     │
│  └─────────────┴────┴────┴────┴────┴────┴────┴────┘                        │
│                                                                             │
│  42 个交叉格子 = 6维画像 × 7节点AIPL                                        │
│  每个格子 6 个指标 = 252 个基础指标位                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      三层价值洞察引擎 (新增)                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 【监控层 Monitoring】                                                │   │
│  │ • KPI快照: 整体Proxy NPS / 情感均值 / 漏斗分布                       │   │
│  │ • 异常检测: 6条规则自动扫描42个格子                                  │   │
│  │ • 告警生成: 结构化告警 (level/category/target/metric)                │   │
│  │ • 健康度评分: 0-100 综合健康指数                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 【分析层 Analysis】                                                  │   │
│  │ • 下钻分析: 从格子深入到主题/子维度/原始VOC                          │   │
│  │ • 根因分析: SHAP风格主题归因 + 负面原因聚类                          │   │
│  │ • 对比分析: 画像维度间差异 / AIPL漏斗漏损                            │   │
│  │ • 机会识别: 高满意度低渗透 / 高口碑小众 / 激活空间                   │   │
│  │ • 趋势信号: 漏斗形态 / 阶段集中度                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 【决策层 Decision】                                                  │   │
│  │ • 策略生成: fix(修复) / grow(增长) / convert(转化)                   │   │
│  │ • 行动队列: 按紧急度排序的具体行动                                   │   │
│  │ • 优先级排序: 多因素综合评分 (urgency × type × impact)               │   │
│  │ • 影响预估: NPS提升预测 + ROI估算                                    │   │
│  │ • 自动路由: 策略 → 主责部门 (产品研发/营销/客服/运营)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           策略输出层                                        │
│  • 监控看板 (实时刷新)                                                      │
│  • 异常告警 (推送到钉钉/飞书)                                               │
│  • 分析报告 (周/月自动导出)                                                 │
│  • 行动工单 (自动创建Jira/Tapd)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 画像 × AIPL 指标体系设计

### 2.1 画像维度定义（6维）

基于 VOC Proxy NPS 统一萃取引擎的 **55 原子标签体系**，按6维分组：

| 维度 | 英文 | 子维度 | 原子标签示例 | 数据来源 |
|------|------|--------|-------------|---------|
| **WHO** | 人群身份 | family_role, parenting_stage | working_parent, first_time_parent, newborn_stage | PERSONABOT + 55原子标签 |
| **WHY** | 决策动机 | pain_point, goal | solve_pain_discomfort, time_constrained, price_sensitive | REVISION意图 + TopicImpact观点 |
| **WHAT** | 关注方面 | function_pref | quiet_seeker, portable_seeker, easy_clean_seeker | ABSA/VOC语义蓝图 |
| **WHEN** | 使用场景 | scenario | nighttime_user, workplace_user, travel_user | 行为意图树 + 评论关键词 |
| **HOW** | 行为模式 | decision_style | research_driven, impulse_buyer, brand_loyal | SoMeR嵌入 + GPLR标签 |
| **EMOTION** | 情感状态 | emotional_state | anxiety_driven, satisfied, frustrated, excited | CSK情感聚类 + 情感校准 |

### 2.2 AIPL 节点定义（7节点）

| 节点 | 英文 | 业务含义 | VOC 信号特征 |
|------|------|----------|-------------|
| **A** | Awareness | 品牌认知 | 搜索品牌、首次浏览、被动曝光 |
| **I** | Interest | 产品兴趣 | 加购对比、查看评价、主动咨询 |
| **P1** | Purchase-1st | 首次购买 | 下单评价、开箱反馈、使用初体验 |
| **P2** | Purchase-Repeat | 复购 | 再次购买、配件购买、升级换购 |
| **L1** | Loyalty-Engage | 活跃忠诚 | 持续使用、参与活动、内容互动 |
| **L2** | Loyalty-Advocacy | 推荐忠诚 | 主动推荐、写评价、社交分享 |
| **L3** | Loyalty-Champion | 超级用户 | KOC合作、社群运营、共创产品 |

### 2.3 矩阵单元指标口径

每个交叉格子 (画像维度 × AIPL节点) 包含6个指标：

| 指标 | 类型 | 计算口径 | 业务含义 |
|------|------|---------|---------|
| **count** | int | 该组合下的VOC条数 | 绝对量 |
| **mention_rate** | float | count / 总VOC数 | 相对渗透率 |
| **avg_sentiment** | float | [-1.0, +1.0] | 平均情感极性 |
| **proxy_nps** | float | Promoter% - Detractor% | 净推荐值 |
| **top_themes** | list | Top 3 主题及次数 | 关注焦点 |
| **sentiment_distribution** | dict | positive/neutral/negative | 情感结构 |

### 2.4 聚合视图

除基础矩阵外，支持以下聚合：

| 视图 | 说明 | 用途 |
|------|------|------|
| 行汇总 | 某画像维度在所有AIPL节点的合计 | 该画像的整体健康度 |
| 列汇总 | 某AIPL节点在所有画像维度的合计 | 该阶段的整体表现 |
| 按品线分片 | 不同品线各一个矩阵 | 品线间对比 |
| 按平台分片 | 不同平台各一个矩阵 | 平台间对比 |
| 按时间分片 | 不同时间段各一个矩阵 | 趋势追踪 |

---

## 3. 数据流与技能融合

### 3.1 现有技能输出映射

```
┌─────────────────────────────────────────────────────────────────────┐
│                        现有技能输出                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎                               │
│  ├── 376标签 → aipl_tags (AIPLTagMatch列表)                         │
│  ├── 55原子标签 → persona_atomic (标签名列表)                        │
│  ├── 【新增】6维分组 → persona_dimensions (dict[WHO/WHY/...])       │
│  ├── 情感 → sentiment_polarity / sentiment_intensity                │
│  ├── Proxy NPS → proxy_nps_contribution                             │
│  └── 业务元数据 → strategy_pack / owner_dept / priority              │
│                                                                     │
│  Skill-PERSONABOT-RAG用户画像                                        │
│  └── 结构化画像JSON → 补充到 persona_dimensions.WHO                  │
│                                                                     │
│  Skill-TopicImpact-观点单元画像抽取                                    │
│  └── 观点单元(主题,情感) → 补充到 aipl_tags 和 aspect_sentiments    │
│                                                                     │
│  Skill-VOC-Semantic-Blueprint                                        │
│  └── (方面,情感,持有者,原因)四元组 → 增强根因分析                    │
│                                                                     │
│  Skill-Behavioral-Intent-Tree-Parsing                                │
│  └── 意图层次树 → 验证/修正 aipl_stage 判定                         │
│                                                                     │
│  Skill-SoMeR-多视角用户表示                                          │
│  └── 用户嵌入向量 → 补充到 persona_dimensions.HOW                   │
│                                                                     │
│  Skill-GPLR-人群标签生成                                             │
│  └── 可解释人群标签 → 补充到 persona_dimensions.WHO                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     【融合层】画像×AIPL矩阵                           │
│                                                                     │
│  PersonaAIPLMatrixBuilder.build(extractions)                        │
│  ├── 遍历每条VOCLabelExtraction                                      │
│  ├── 按 persona_dimensions 的6个key分组                             │
│  ├── 按 aipl_stage 归入7个节点之一                                  │
│  ├── 累计 count / sentiments / themes / proxy_nps_counts            │
│  └── 输出: 42个MatrixCell组成的完整矩阵                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 技能关联图谱

```
【数据输入层】
退货留言 / 客服工单 / 商品评论 / Trustpilot / 搜索日志 / 行为数据
    ↓
【基础萃取层】
VOC-Proxy-NPS-AIPL-统一萃取引擎 ← 核心枢纽
    ├── 376标签种子匹配
    ├── 55原子画像标签 → 【新增】6维分组
    ├── 情感校准
    └── Proxy NPS
    ↓
【画像增强层】（可选接入，增强矩阵精度）
├── PERSONABOT → WHO维度增强
├── TopicImpact → WHAT/WHY维度增强
├── VOC语义蓝图 → WHAT/WHY/EMOTION增强
├── 行为意图树 → HOW/WHEN维度验证
├── SoMeR → HOW维度增强
└── GPLR → WHO维度增强
    ↓
【矩阵计算层】
PersonaAIPLMatrixBuilder
    ├── 6维画像 × 7节点AIPL = 42格子
    ├── 每格子6指标 = 252基础指标位
    └── 支持分片聚合（品线/平台/时间）
    ↓
【洞察引擎层】
ValueInsightEngine
    ├── 监控层: 异常检测 + KPI快照 + 告警
    ├── 分析层: 下钻 + 根因 + 对比 + 机会
    └── 决策层: 策略 + 行动队列 + 优先级 + ROI
    ↓
【策略输出层】
看板 / 告警 / 报告 / 工单
```

---

## 4. 三层价值洞察引擎

### 4.1 监控层 (Monitoring Layer)

**职责**: 实时扫描矩阵健康状态，发现异常并告警

**核心能力**:

| 能力 | 实现 | 输出 |
|------|------|------|
| KPI快照 | 全量聚合计算 | 整体NPS/情感/漏斗分布 |
| 异常检测 | 6条规则扫描42格子 | 异常列表(含severity) |
| 告警生成 | 结构化告警模板 | 推送到IM/邮件 |
| 健康度评分 | 100分基准，异常扣分 | 0-100分 |

**异常检测规则**:

| 规则ID | 条件 | 严重程度 | 业务含义 |
|--------|------|---------|---------|
| RULE-01 | avg_sentiment < -0.5 AND count > 5 | high | 系统性负面 |
| RULE-02 | proxy_nps < 0 AND mention_rate > 5% | high | NPS危险区 |
| RULE-03 | 某维度在单节点占比 > 50% AND count > 10 | medium | 阶段过度集中 |
| RULE-04 | avg_sentiment < -0.3 AND count > 3 | medium | 轻度负面 |
| RULE-05 | proxy_nps < 20 AND mention_rate > 3% | low | NPS预警 |
| RULE-06 | count = 0 BUT 同维度其他节点count > 10 | low | 空白格子 |

**告警结构**:

```json
{
  "level": "critical|warning",
  "category": "sentiment_drop|nps_drop|funnel_leak",
  "target": "人群身份 × 首购 Purchase-1st",
  "metric_value": -0.6,
  "threshold": -0.5,
  "description": "情感极性-0.60低于阈值-0.50",
  "recommended_action": "立即排查产品核心体验问题",
  "top_themes": [{"theme": "产品核心性能", "count": 8}],
  "owner": "产品中心/品线"
}
```

### 4.2 分析层 (Analysis Layer)

**职责**: 深度分析数据，回答"为什么"和"是什么"

**核心能力**:

| 能力 | 输入 | 输出 | 实现 |
|------|------|------|------|
| 下钻分析 | 指定格子 (dim, node) | 主题/子维度/情感分布 | 聚合MatrixCell详情 |
| 根因分析 | 指定格子 + 原始VOC | 负面主题归因 + 原因总结 | 主题情感聚类 + VOC语义蓝图 |
| 对比分析 | 两个画像维度 | 差异指标 + 漏斗漏损 | 矩阵切片对比 |
| 机会识别 | 全矩阵 | 高潜力格子列表 | 满意度×渗透率交叉分析 |
| 趋势信号 | 全矩阵 | 漏斗漏损/集中度信号 | 阶段间count比率分析 |

**根因分析示例**:

```
输入: WHAT × P1 (关注方面 × 首购)
输出:
  ├─ sample_size: 15条VOC
  ├─ theme_breakdown:
  │   ├─ 产品核心性能: count=8, avg_sentiment=-0.72, impact=high
  │   ├─ 噪音水平: count=5, avg_sentiment=-0.65, impact=high
  │   └─ 配件质量: count=2, avg_sentiment=-0.40, impact=medium
  ├─ top_negative_themes: ["产品核心性能", "噪音水平"]
  └─ root_cause_summary: "主要由 [产品核心性能, 噪音水平] 方面的负面体验驱动"
```

**对比分析示例**:

```
对比: WHO(working_parent) vs WHAT(quiet_seeker) 在 P1 阶段
输出:
  ├─ WHO: sentiment=+0.00, NPS=+0.0, count=4
  ├─ WHAT: sentiment=-0.50, NPS=-100.0, count=3
  ├─ sentiment_diff: +0.50
  ├─ nps_diff: +100.0
  └─ insight: "职场妈妈整体首购体验中性，但静音关注者首购体验极差"
```

### 4.3 决策层 (Decision Layer)

**职责**: 将洞察转化为可执行策略

**核心能力**:

| 能力 | 输入 | 输出 |
|------|------|------|
| 策略生成 | 异常+机会+漏斗信号 | fix/grow/convert 策略列表 |
| 行动队列 | 策略列表 | 按紧急度排序的具体行动 |
| 优先级排序 | 策略列表 | 综合评分排序 |
| 影响预估 | 策略列表 + 当前矩阵 | NPS提升预测 + ROI估算 |
| 自动路由 | 策略列表 | 主责部门分配 |

**策略类型**:

| 类型 | 触发条件 | 策略示例 | 主责部门 |
|------|---------|---------|---------|
| **fix** | 异常检测 | 针对[噪音水平]启动产品体验优化专项 | 产品中心/品线 |
| **grow** | 机会识别 | 扩大[静音功能]的内容营销覆盖 | 营销增长部 |
| **convert** | 漏斗漏损 | 推出限时优惠+免邮，降低首购门槛 | 用户增长部 |

**优先级评分公式**:

```
score = urgency_score(30/15/5) + type_score(fix=20, convert=25, grow=10) + impact_bonus
```

**行动队列示例**:

```json
[
  {
    "action_id": "STRAT_FIX_WHAT_P1",
    "action": "【紧急】针对 [噪音水平] 启动产品体验优化专项",
    "owner": "产品中心/品线",
    "deadline": "3个工作日内",
    "status": "pending"
  },
  {
    "action_id": "STRAT_CONV_A_I",
    "action": "增加互动式内容（测评视频/对比工具），降低认知到兴趣的门槛",
    "owner": "用户增长部",
    "deadline": "2周内",
    "status": "pending"
  }
]
```

---

## 5. 代码实现

### 5.1 文件结构

```
paper2skills-code/nlp_voc/proxy_nps_aipl_workflow/
├── __init__.py                    # 模块导出（新增矩阵+引擎导出）
├── unified_label_extraction.py    # 【改造】增加 persona_dimensions 字段
├── workflow.py                    # 工作流编排（不变）
├── persona_aipl_matrix.py         # 【新增】画像×AIPL矩阵计算
└── insight_engine.py              # 【新增】三层价值洞察引擎
```

### 5.2 关键API

```python
# 1. 统一萃取（改造后）
from proxy_nps_aipl_workflow import VOCLabelExtractor
extractor = VOCLabelExtractor(tag_dict)
result = extractor.extract(voc_record)
# result.persona_dimensions = {"WHO": ["working_parent"], "WHAT": ["quiet_seeker"]}

# 2. 构建矩阵（新增）
from proxy_nps_aipl_workflow import PersonaAIPLMatrixBuilder
builder = PersonaAIPLMatrixBuilder()
matrix = builder.build(extractions)

# 3. 运行三层洞察（新增）
from proxy_nps_aipl_workflow import ValueInsightEngine
engine = ValueInsightEngine()
report = engine.run(matrix, extractions, drill_down={"persona_dim": "WHAT", "aipl_node": "P1"})

# 4. 仅运行监控层
monitoring = engine.monitoring(matrix)

# 5. 仅运行分析层（带下钻）
analysis = engine.analysis(matrix, extractions, drill_down={"persona_dim": "WHO", "aipl_node": "P1"})

# 6. 仅运行决策层
decisions = engine.decision(matrix, extractions)
```

### 5.3 矩阵单元API

```python
# 获取指定格子
cell = matrix.get_cell("WHAT", "P1")
print(cell.avg_sentiment())      # -0.50
print(cell.proxy_nps())          # -100.0
print(cell.top_themes(3))        # [{"theme": "产品核心性能", "count": 8}]

# 获取维度切片
what_slice = matrix.get_dimension_slice("WHAT")  # WHAT在所有AIPL节点的数据

# 异常检测
anomalies = matrix.detect_anomalies()

# 机会识别
opportunities = matrix.find_opportunities()

# 维度对比
comparison = matrix.compare_dimensions("WHO", "WHAT")
```

---

## 6. 实施路径

### Phase 1: 引擎改造（1-2天）

- [x] 改造 `unified_label_extraction.py`：增加 `persona_dimensions` 字段
- [x] 改造 `DashboardGenerator`：增加 `_calc_persona_aipl_matrix` 方法
- [x] 创建 `persona_aipl_matrix.py`：矩阵数据模型 + 构建器 + 基础分析
- [x] 创建 `insight_engine.py`：三层洞察引擎
- [x] 更新 `__init__.py`：导出新模块
- [ ] 运行测试验证

### Phase 2: 数据验证（2-3天）

- [ ] 用真实VOC数据（355,697条）运行端到端流程
- [ ] 验证矩阵计算结果与业务直觉是否一致
- [ ] 调整异常检测阈值
- [ ] 校准策略推荐规则

### Phase 3: 看板对接（3-5天）

- [ ] 开发监控层看板（实时刷新）
- [ ] 开发分析层交互界面（支持下钻）
- [ ] 开发决策层策略工单（自动创建任务）
- [ ] 对接钉钉/飞书告警

### Phase 4: 技能增强（持续）

- [ ] 接入 VOC语义蓝图 增强根因分析
- [ ] 接入 行为意图树 验证AIPL阶段
- [ ] 接入 SoMeR嵌入 增强HOW维度
- [ ] 接入 跨语言语义对齐 支持多市场矩阵

---

## 7. 预期效果

| 指标 | 现状 | 目标 |
|------|------|------|
| 洞察生成周期 | 人工周级 | 自动天级/实时 |
| 异常发现速度 | 被动等待用户投诉 | 主动预警（T+0） |
| 策略到执行周期 | 周级（人工流转） | 天级（自动路由） |
| 多技能整合成本 | 高（需分析师手工整合） | 低（自动矩阵计算） |
| 覆盖指标数 | 单一维度（NPS/情感） | 252指标位（6×7×6） |

---

## 8. 相关文档

- [Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎](Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md)
- [Skill-AIPL-VOC-Lifecycle-Tags](Skill-AIPL-VOC-Lifecycle-Tags.md)
- [Skill-PERSONABOT-RAG用户画像生成](Skill-PERSONABOT-RAG用户画像生成.md)
- [Skill-SoMeR-多视角用户表示](Skill-SoMeR-多视角用户表示.md)
- [Skill-TopicImpact-观点单元画像抽取](Skill-TopicImpact-观点单元画像抽取.md)
- [Skill-VOC-Semantic-Blueprint](Skill-VOC-Semantic-Blueprint.md)
- [Skill-Behavioral-Intent-Tree-Parsing](Skill-Behavioral-Intent-Tree-Parsing.md)
- [VOC决策智能桥接算法-完整图谱](VOC决策智能桥接算法-完整图谱.md)

---

**文档版本**: v1.0
**创建日期**: 2026-04-28
**适用场景**: Momcozy 母婴出海 VOC 用户画像 × AIPL 指标体系与价值洞察

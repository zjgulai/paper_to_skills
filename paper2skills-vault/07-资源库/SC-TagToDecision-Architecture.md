---
name: sc-tag-to-decision-architecture
description: 供应链「标签工程→决策」全链路 AI 决策系统架构方案。涵盖七层架构设计、标签分类体系、Palantir 方法论映射、企业 AI 知识库依赖关系、分阶段实施路线图与三大架构陷阱。当设计供应链 AI 决策系统、规划标签工程落地、评估 Palantir 方法论迁移时使用。
---

# 供应链「标签工程 → 决策」全链路 AI 决策系统架构方案

> **适用场景**：母婴跨境电商品牌（Amazon FBA + DTC，年 GMV 500万-2000万美元，5-20人运营团队）
> **设计原则**：Palantir 本体论方法论迁移 · 标签作为决策的语义枢纽 · 从分析到自动执行的闭环
> **版本**：v1.0 · 2026-06-18

---

## 一、全局架构：七层设计

```
【外部世界】
ERP/WMS · Amazon FBA API · 物流API · 供应商邮件/PO · 市场数据
                    ↓ CDC / Event Stream / Batch ETL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L1  数据基础层  Foundation
    Golden Record · Data Lineage · OKB Graph (Neo4j+Delta)
    Event Sourcing · Feature Store · Data Reconciliation
                    ↓ 统一实体ID · 去重消歧 · 质量保障
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L2  本体语义层  Ontology
    ObjectTypes: SKU / Supplier / Order / Shipment / Warehouse
    LinkTypes:   SUPPLIES / SHIPS / STORES / CONTAINS / GOVERNS
    LLM-AutoBuild · Schema-Versioning · Supplier-Ontology-Map
                    ↓ 语义对象 · 关系图谱 · 版本化治理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L3  标签工程层  Tag Engineering  ← 系统的语义枢纽
    静态标签: SKU分类 / 供应商等级 / 合规状态 / BOM成本层级
    动态标签: 风险评分 / 库存健康 / 供应商可靠性 / 需求趋势
    预测标签: 断货风险 / 呆滞风险 / 竞品威胁 / 价格弹性
    三层流水线: 规则引擎 → ML分类器 → LLM语义 → 图传播
                    ↓ 结构化标签向量（在线 <20ms 读取）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L4  信号分析层  Signal / Analytics
    KPI监控: 库存健康 / OTD / 供应商评分 / 成本结构
    预测:    需求预测 / GCF隐性需求 / 异常检测
    因果:    Causal-DAG根因归因 / 反事实需求估计
    不确定性: Conformal Prediction 置信区间
                    ↓ 带置信度的洞察信号 + 根因解释
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L5  决策推理层  Decision Intelligence
    What-if:  多情景 Pareto 最优对比
    优化:     MILP多目标约束规划 / DRL动态补货
    推理:     因果决策图 / do-calculus 干预分析
    校准:     置信度门控 → 自动执行 or 升级审核
                    ↓ 结构化决策建议 + 执行优先级
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L6  行动执行层  Action / Execution
    低风险→全自动: MCP写回ERP/WMS · Action-Trigger · 共识补货
    中风险→半自动: SCPA生成报告 · 草案PO · 一键确认
    高风险→人工审: Human-in-Loop审批门控 · 升级告警
                    ↓ 行动执行 + 完整审计日志
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L7  反馈学习层  Feedback / Learning
    数字孪生: 行动结果 → 状态更新 → 下次决策校准
    审计追踪: 每个决策完整记录 → 标签标注 → 模型再训练
    闭环学习: Decision-Outcome → 标签更新 → 信号修正 → 决策优化
                    ↓ 循环回到 L3（标签持续演化）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 二、标签工程：系统的语义枢纽

### 2.1 为什么「标签」是核心而不只是标注

标签在本架构中扮演三个角色，远超"打标签"的字面含义：

| 角色 | 含义 | 示例 |
|------|------|------|
| **决策输入压缩** | 将 50+ 维原始数据压缩为 10 个语义标签，让 Agent 决策有上下文 | `stockout_risk_high` = (DOS<14) AND (supplier_delay>0) AND (no_in_transit) |
| **传播媒介** | 标签通过图关系传播，单点风险→系统性感知 | 供应商风险标签 → 传播到其所有 SKU → 触发级联预警 |
| **行动触发器** | 特定标签组合直接触发 Action，绕过人工判断 | `stockout_risk_high` + `reorder_point_breached` → 自动触发补货 PO |

### 2.2 完整标签分类体系

#### SKU 标签（最核心）

```yaml
SKU标签:
  静态（每月更新）:
    - category: "婴儿消毒/辅食/安全防护/..."
    - margin_tier: "A(>40%) / B(20-40%) / C(<20%)"
    - shelf_life_limited: true/false         # 有保质期约束
    - is_substitutable: true/false           # 有替代产品
    - compliance_status: "compliant/pending/at_risk"
    - velocity_category: "fast/medium/slow"  # 动销速度
  
  动态（每日更新）:
    - inventory_health: "healthy/warning/risk/overstock"
    - dos_current: float                     # 剩余可销天数
    - demand_trend: "rising/stable/falling"  # 近14天趋势
    - promo_active: true/false               # 是否在促销
    - seasonal_flag: "peak/normal/trough"    # 季节性阶段
  
  预测（每日更新，含置信度）:
    - stockout_risk_score: 0-100             # 断货风险评分
    - overstock_risk_score: 0-100            # 呆滞风险评分
    - price_erosion_warning: true/false      # 价格侵蚀预警
    - demand_volatility: "low/medium/high"   # 需求波动性
```

#### 供应商标签

```yaml
供应商标签:
  静态:
    - tier_level: "1(直接)/2(原材料)/3(辅料)"
    - geo_region: "华南/华东/东南亚/..."
    - payment_terms: "Net30/Net60/Prepay"
    - is_single_source: true/false           # 是否唯一供应商（高风险）
  
  动态:
    - reliability_score: 0-100               # 综合可靠性评分
    - lead_time_category: "normal/extended/critical"
    - capacity_utilization: "low/normal/high/overflow"
    - financial_risk_flag: true/false        # 资金链风险
  
  风险:
    - risk_elevated: true/false
    - geopolitical_exposure: "low/medium/high"
    - compliance_gap: true/false             # 认证/合规缺口
```

#### 订单/运输标签

```yaml
订单标签:
  - priority_urgent: true/false              # 是否紧急
  - delay_risk_high: true/false              # 延误风险
  - customs_hold_risk: true/false            # 清关风险
  - logistics_cost_tier: "normal/expensive/urgent_premium"
  - tariff_change_affected: true/false       # 受关税变化影响
```

### 2.3 三层标签生成流水线

```
原始数据输入
    ↓
[规则引擎层] ← 最快，覆盖80%的标签
  threshold规则: DOS < 14 → stockout_risk_high = true
  IF/THEN逻辑: 供应商delay_days > PLT_p90 → lead_time_category = extended
    ↓ 规则无法覆盖的复杂模式
[ML分类层] ← 覆盖15%，处理非线性模式
  XGBoost/LightGBM: 综合历史特征预测风险评分
  异常检测: Isolation Forest 识别异常供应商行为
    ↓ 需要语义理解的边缘案例
[LLM语义层] ← 覆盖5%，处理非结构化信息
  从供应商邮件提取风险信号
  从客服工单提取质量问题标签
    ↓
[图传播层] ← 将节点标签扩散到关联节点
  供应商风险 → 传播到其所有 SKU
  SKU合规问题 → 传播到同款式变体
  原材料价格标签 → 传播到所有包含该原材料的成品（BOM传播）
```

### 2.4 标签→行动触发的关键规则设计

这是 Palantir Action Type 的核心设计，也是"自动化"的实现机制：

```python
# 行动触发规则库（从规则到 Action）
ACTION_TRIGGER_RULES = [
    # 高置信度 → 全自动
    {
        "rule_id": "AT-001",
        "condition": "stockout_risk_score > 80 AND supplier_reliability > 70 AND dos_current < 7",
        "confidence_required": 0.9,
        "action": "auto_create_urgent_po",
        "approval": "none",  # 全自动
        "notify": ["ops_manager"],
    },
    # 中置信度 → 半自动（草案+确认）
    {
        "rule_id": "AT-002",
        "condition": "stockout_risk_score > 60 AND dos_current < 14",
        "confidence_required": 0.7,
        "action": "draft_po_for_approval",
        "approval": "one_click",  # 一键确认
        "notify": ["buyer"],
    },
    # 供应商风险传播 → 系统性预警
    {
        "rule_id": "AT-003",
        "condition": "supplier.risk_elevated = True AND supplier.is_single_source = True",
        "confidence_required": 0.8,
        "action": "trigger_alternative_supplier_search",
        "approval": "none",
        "notify": ["supply_chain_manager"],
    },
    # 呆滞预警 → 清仓建议
    {
        "rule_id": "AT-004",
        "condition": "overstock_risk_score > 75 AND dos_current > 90 AND shelf_life_limited = True",
        "confidence_required": 0.85,
        "action": "generate_clearance_plan",
        "approval": "manager_review",
        "notify": ["ops_manager", "finance"],
    },
]
```

---

## 三、Palantir 方法论映射

| Palantir 概念 | 本架构对应 | 核心 Skill |
|---------------|-----------|-----------|
| **Object Store** | L1 OKB Graph (Neo4j) + Feature Store | [[Skill-Graph-OKB-Design-SC]] |
| **Object Types** | SKU / Supplier / Order / Shipment | [[Skill-Ontology-LLM-AutoBuild-SC]] |
| **Link Types** | 供应链关系图谱（SUPPLIES/SHIPS等） | [[Skill-Supplier-Ontology-Capability-Map]] |
| **Derived Properties** | 动态标签（risk_score等由 ML 计算） | [[Skill-Online-Feature-Store-SC-Realtime]] |
| **Action Types** | 行动触发规则 + MCP 写回 | [[Skill-Supply-Chain-Ontology-Action-Trigger]] |
| **AIP Logic** | 因果决策图 + MILP + LLM 仲裁 | [[Skill-SC-WhatIf-Scenario-Analysis-Engine]] |
| **OSDK Writeback** | MCP 协议 → SAP/WMS API | [[Skill-SC-Agent-MCP-ERP-Integration]] |
| **Audit Trail** | 决策审计追踪本体 | [[Skill-Decision-Audit-Trail-Ontology]] |
| **Simulation Engine** | 数字孪生 What-if 仿真 | [[Skill-SC-Digital-Twin-Sync-Architecture]] |

**从 BI 到 Palantir 本体论的本质跃升**：

```
传统 BI                          Palantir 本体论
─────────────────────────────────────────────────
只读仪表盘                  →   读写决策循环
相关性分析                  →   因果干预（do-calculus）
静态星型模式                →   动态语义图（Objects+Links）
人工导出→手动ERP输入        →   Action 自动写回
月度模型更新                →   反馈闭环持续学习
```

---

## 四、企业 AI 知识库依赖关系

### 4.1 知识库核心资产（生产者）

```
知识库生产者（写入）：
  L1 → 写入: 标准化实体数据（SKU主数据、供应商档案）
  L2 → 写入: 本体 Schema（ObjectType/LinkType 定义）
  L3 → 写入: 标签定义规范 + 历史标签快照
  L6 → 写入: 行动执行记录（审计日志）
  L7 → 写入: 决策结果标注（用于模型再训练）

知识库消费者（读取）：
  L3 → 读取: 规则库（触发阈值）、BOM关系
  L4 → 读取: 历史特征（训练数据）、行业基准
  L5 → 读取: 历史决策案例（相似情景 RAG）
  L6 → 读取: Action 参数模板、审批规则
  Agent → 读取: 供应商档案、合同条款、操作 SOP
```

### 4.2 知识库四类核心资产

| 资产类型 | 内容 | 存储位置 | 更新频率 |
|---------|------|---------|---------|
| **结构知识** | 本体 Schema、关系定义、BOM结构 | Neo4j OKB | 月度/事件触发 |
| **规则知识** | Action 触发规则、审批阈值、SOP | YAML/规则引擎 | 季度评审 |
| **统计知识** | 历史特征、弹性系数、基准指标 | Feature Store | 日度更新 |
| **案例知识** | 历史决策+结果（用于 RAG 检索） | 向量数据库 | 实时追加 |

---

## 五、分阶段实施路线图

### Phase 0：地基（第 1-4 周）—— 不跳过，否则后续全错

**目标**：建立统一实体 ID，不统一就没有一切

```
Week 1-2: SKU 主数据治理
  - 盘点所有渠道的 SKU 编码（ERP/FBA/WMS/供应商各不同）
  - 建立 Golden Record：统一 SKU ID（内部码）→ 映射表
  - 工具: Excel → 后期迁移到 Neo4j
  - 里程碑: 100% SKU 有唯一内部 ID，多源可查

Week 3-4: 供应商本体基础
  - 梳理所有供应商档案（50-200 家），统一 Supplier ID
  - 建立 Supplier → SKU 供货关系（手工 + LLM 辅助）
  - 工具: Airtable/Notion → 后期迁移到图数据库
  - 里程碑: 完整的"谁供什么"关系图
```

**最高 ROI 动作**：统一实体 ID。没有这一步，后续所有分析都是在不同数据集上做，结论互相矛盾。

---

### Phase 1：MVP（第 5-12 周）—— 3 个月见效

**目标**：跑通"标签→预警→行动"最短路径，用最简单场景验证

```
Week 5-6: 库存健康标签（最高价值的第一个标签）
  - 规则引擎版本: DOS = (stock + in_transit) / daily_sales
  - 3 档标签: healthy(30-60天) / warning(14-30天) / risk(<14天) / overstock(>90天)
  - 数据源: FBA库存API + 历史销量
  - 产出: 每日库存健康看板（Excel/Airtable 先跑）

Week 7-8: 补货触发自动化（第一个 Action）
  - 当 DOS < 14 天 AND 无在途货: 自动生成补货提醒邮件（草稿）
  - 第一版用 Python 脚本 + 邮件，不需要 ERP 集成
  - 验证: 补货响应时间从 X 天降到 Y 天（记录基线）

Week 9-10: 供应商风险标签
  - 规则: 近 3 次交货延误率 > 20% → lead_time_category = extended
  - 规则: 只有一个供应商 → is_single_source = True → 高风险
  - 产出: 供应商红黄绿灯看板

Week 11-12: 标签传播（从点到面）
  - 供应商风险标签 → 传播到其 SKU → SKU 自动标记"供应商风险"
  - 产出: "如果供应商 A 出问题，哪些 SKU 受影响" → 秒级回答
  - 里程碑: MVP 验证完成，团队认可价值
```

**Phase 1 预期 ROI**：补货延误减少 30%，断货风险 SKU 提前 14 天识别，运营日均节省 1-2 小时。

---

### Phase 2：扩展（第 13-24 周）—— 半年形成能力

**目标**：引入 ML 预测标签 + 因果分析 + Agent 初步自动化

```
Month 4:
  - 部署 ML 预测标签: 用历史数据训练断货风险模型（XGBoost）
  - 需求预测接入: 用 Prophet/LightGBM 替代人工判断销量趋势
  - 标签置信度: 每个预测标签带 P10/P50/P90 区间

Month 5:
  - 因果分析: Causal-DAG 识别"断货根因"（区分供应延误 vs 需求超预期 vs 安全库存参数错）
  - GCF 隐性需求: 断货/下架期间的真实需求估计，修正历史预测偏差
  - What-if 引擎: 大促备货方案比较（折扣/备货量/物流方式的多情景对比）

Month 6:
  - SCPA 智能体接入: 自然语言查询供应链状态
  - MCP 基础集成: 查询 ERP 数据（只读先行）
  - 多智能体补货共识: 需求/采购/仓储三方自动协商
```

**Phase 2 预期 ROI**：补货决策准确率提升 40%，根因分析从 2 天→30 分钟，大促备货失误率降低 50%。

---

### Phase 3：智能化（第 25-52 周）—— 一年成熟

**目标**：全自动决策闭环，人工只处理异常

```
Q3:
  - Action 自动执行: 低风险补货 PO 全自动创建（不需要人工确认）
  - MCP 双向集成: 写回 ERP/WMS，触发真实操作
  - 数字孪生: 供应链状态实时镜像，What-if 仿真

Q4:
  - 闭环学习: 决策结果→标签修正→模型再训练 → 自动化循环
  - 本体扩展: LLM 从新 PO/邮件自动扩充知识图谱
  - Schema 版本化: 本体升级不中断在线 Agent
```

**Phase 3 预期 ROI**：运营团队供应链日常工作减少 60%，异常响应从小时级→分钟级，年化节省人力成本约 30-80 万元。

---

## 六、三大架构陷阱（最容易犯的决策错误）

### 陷阱 1：跳过实体 ID 统一，直接上 ML

**错误做法**：ERP 里叫「STLPRO-V2」，FBA 叫「B09XXXX」，供应商叫「消毒锅 V2 蓝色版」——三套数据跑了三个模型，结论互相矛盾。

**正确做法**：Phase 0 的 Golden Record 是一切的前提。没有统一 ID，标签无法传播，图谱无法建立，模型训练集是污染的。

**代价估算**：跳过后，平均在 Phase 2 再回头补，需要额外 3-6 个月 + 清洗历史脏数据。

---

### 陷阱 2：把标签设计成"结论"而不是"信号"

**错误做法**：设计一个标签叫 `needs_replenishment = True`，然后围绕这个标签建系统。

**问题**：`needs_replenishment` 是一个决策输出，不是输入信号。当市场条件变化（大促/竞品断货/物流旺季），这个标签的阈值需要不断调整，最终变成一个没人敢动的"魔法变量"。

**正确做法**：标签设计为**可测量的事实状态**（`dos_current=12`、`stockout_risk_score=87`），决策逻辑在 L5 层动态计算，标签层只保存"是什么"不保存"怎么做"。

---

### 陷阱 3：Action 全自动化，没有置信度门控

**错误做法**：模型预测断货 → 直接触发 PO，不管置信度是 55% 还是 95%。

**风险**：一个错误的自动 PO 可能锁定 10-50 万元资金，促销期模型预测失误会导致系统性过量备货。

**正确做法**：三档执行策略
- **置信度 > 90%**：全自动执行，仅通知
- **置信度 70-90%**：生成草案，一键确认（< 5 秒决策）
- **置信度 < 70%**：提交审批，说明不确定性来源

置信度计算用 [[Skill-Decision-Confidence-Calibration-SC]]（Conformal Prediction），必须在 Action 触发前完成校准。

---

## 七、全链路 Skill 索引（726 个中的核心 71 个）

### L1 数据基础层（7 个）

| Skill | 作用 |
|-------|------|
| [[Skill-SKU-Master-Data-Golden-Record]] | SKU 主数据黄金记录、实体消歧 |
| [[Skill-Supply-Chain-Data-Lineage-Tracking]] | 数据血缘追踪 |
| [[Skill-Graph-OKB-Design-SC]] | Neo4j+Delta 双层 OKB 架构 |
| [[Skill-Online-Feature-Store-SC-Realtime]] | 在线特征存储 <20ms |
| [[Skill-Inventory-Event-Sourcing-Architecture]] | 库存事件溯源 |
| [[Skill-Supply-Chain-Data-Mesh-Architecture]] | 数据网格分布式架构 |
| [[Skill-Cross-System-Data-Reconciliation]] | ERP/WMS/FBA 三系统对账 |

### L2 本体语义层（10 个）

| Skill | 作用 |
|-------|------|
| [[Skill-Ontology-LLM-AutoBuild-SC]] | LLM 零样本本体自动构建 ⭐ |
| [[Skill-SC-Ontology-Schema-Versioning]] | Schema 版本化，零停机升级 ⭐ |
| [[Skill-SKU-Entity-Unified-ID-Tagging]] | 统一实体 ID 体系 |
| [[Skill-Supplier-Ontology-Capability-Map]] | 供应商能力本体 |
| [[Skill-SKU-Level-Margin-Attribution-Ontology]] | SKU 毛利归因本体 |
| [[Skill-Supply-Chain-Ontology-Action-Trigger]] | Palantir 风格本体驱动行动 ⭐ |
| [[Skill-Multi-Market-Compliance-Matrix-Ontology]] | 多市场合规矩阵本体 |
| [[Skill-Decision-Audit-Trail-Ontology]] | 决策审计追踪本体 |
| [[Skill-Competitor-SKU-Ontology]] | 竞品 SKU 本体 |
| [[Skill-BOM-Cost-Rollup-Tag-Engine]] | BOM 成本卷积标签 |

### L3 标签工程层（5 个核心）

| Skill | 作用 |
|-------|------|
| [[Skill-Tag-Schema-Engineering-Lifecycle]] | 标签 Schema 设计与生命周期 ⭐ |
| [[Skill-Auto-Tagging-Pipeline-Rule-ML-LLM]] | 三层自动打标流水线 ⭐ |
| [[Skill-Tag-Propagation-Supply-Chain]] | 标签在图上的传播算法 ⭐ |
| [[Skill-Tag-Quality-Coverage-KPI]] | 标签质量 KPI 监控 |
| [[Skill-Predictive-Tag-Engine-Supply-Chain]] | 预测性标签引擎 |

### L4 信号分析层（精选 8 个）

| Skill | 作用 |
|-------|------|
| [[Skill-Supply-Chain-KPI-Health-Dashboard]] | 供应链 KPI 健康看板 |
| [[Skill-Real-Time-Supply-Chain-Drift-Detection]] | 实时漂移检测 |
| [[Skill-Signal-Uncertainty-Quantification-SC]] | 不确定性量化 |
| [[Skill-GCF-Counterfactual-Unobserved-Demand]] | 隐性需求估计 MAPE↓75% ⭐ |
| [[Skill-SC-Causal-DAG-E2E-Attribution]] | 端到端因果根因归因 ⭐ |
| [[Skill-Demand-Forecasting-Supply-Chain]] | 需求预测基础 |
| [[Skill-Forecast-Bias-Adjustment-Detection]] | 预测偏差检测 |
| [[Skill-Supply-Chain-Causal-SCM-Attribution]] | SCM 因果归因 |

### L5 决策推理层（精选 7 个）

| Skill | 作用 |
|-------|------|
| [[Skill-SC-WhatIf-Scenario-Analysis-Engine]] | What-if 多情景 Pareto ⭐ |
| [[Skill-Causal-Decision-Graph-SC-Inference]] | 因果决策图推理 ⭐ |
| [[Skill-Multi-Objective-Constrained-Action-Planning]] | MILP+LLM 多目标规划 ⭐ |
| [[Skill-Decision-Confidence-Calibration-SC]] | Conformal 置信度校准 ⭐ |
| [[Skill-Black-Swan-Scenario-Simulation-Tag]] | 黑天鹅极端情景 |
| [[Skill-Counterfactual-SC-Scenario-Sim]] | 反事实情景模拟 |
| [[Skill-DRL-Inventory-Optimization]] | DRL 动态库存优化 |

### L6 行动执行层（7 个）

| Skill | 作用 |
|-------|------|
| [[Skill-Supply-Chain-Ontology-Action-Trigger]] | Palantir 风格 Action 触发 ⭐ |
| [[Skill-WMS-Exception-Action-Trigger]] | WMS 异常行动触发 |
| [[Skill-Human-in-Loop-Approval-Gate-Tag]] | 人机协作审批门控 ⭐ |
| [[Skill-LLM-SC-MultiAgent-Consensus-Replenishment]] | 三方共识补货 ⭐ |
| [[Skill-SC-Agent-MCP-ERP-Integration]] | MCP 多 ERP 集成 ⭐ |
| [[Skill-SCPA-Autonomous-SC-Planning-Agent]] | 自主规划智能体 ⭐ |
| [[Skill-Automated-Replenishment-Decision-Engine]] | 自动补货决策引擎 |

### L7 反馈学习层（4 个）

| Skill | 作用 |
|-------|------|
| [[Skill-SC-Digital-Twin-Sync-Architecture]] | 数字孪生同步架构 ⭐ |
| [[Skill-Decision-Outcome-Closed-Loop-Learning]] | 决策结果闭环学习 ⭐ |
| [[Skill-CS-Supply-Chain-Feedback-Loop-Tag]] | 客服→供应链反馈闭环 |
| [[Skill-Supply-Chain-Agent-Orchestration-Hub]] | Agent 编排中枢 |

> ⭐ = 优先实施

---

## 八、关键设计决策记录（ADR）

### ADR-001：标签存储选 Feature Store 还是 OKB Graph？

**决策**：两者并存，职责分离
- **Feature Store（DynamoDB/Redis）**：数值型标签（risk_score=87, dos=12），在线 <20ms 读取，供 ML/Agent 实时消费
- **OKB Graph（Neo4j）**：关系型标签（is_single_source、tier_level），支持多跳传播查询

**原则**：能用 Feature Store 解决的不用 Graph，需要"传播"和"关系推理"的用 Graph。

---

### ADR-002：标签打标时机是实时还是批量？

**决策**：分级
- **实时（<1分钟）**：库存健康标签（每次库存变更触发）、延误预警标签（ETA 更新触发）
- **每日批量（00:00-02:00）**：风险评分标签、需求趋势标签、供应商可靠性标签
- **周度**：成本层级标签、季节性标签、合规状态标签

**原则**：标签的更新频率 = 其驱动的 Action 所需要的最低响应速度。

---

### ADR-003：何时引入 LLM，何时坚持规则？

**决策**：明确分工
- **规则引擎优先**：所有有明确阈值的判断（DOS < X，延误 > Y天）——可解释、可审计、不依赖 API
- **ML 次之**：需要多特征综合判断的评分（风险评分、供应商可靠性）
- **LLM 最后**：非结构化信息处理（邮件/评论/文件）、例外情况的边界决策、生成解释性报告

**原则**：越靠近 Action 执行，越要用规则/ML（可靠性优先）；越靠近数据摄入，越可以用 LLM（灵活性优先）。

---

## 九、快速启动清单（第一周可执行）

```bash
# Step 1: 盘点现有 SKU 编码体系
# 列出所有渠道的 SKU ID（ERP/FBA/WMS/供应商），建立映射表

# Step 2: 跑第一个健康标签
python3 - << 'EOF'
import pandas as pd

# 假设已有 FBA 库存 CSV 导出
df = pd.read_csv("fba_inventory.csv")
df["daily_sales_7d"] = df["units_sold_7d"] / 7
df["dos"] = (df["afn_fulfillable_quantity"] + df["reserved_quantity"]) / df["daily_sales_7d"].clip(lower=0.1)
df["inventory_health"] = pd.cut(df["dos"],
    bins=[-1, 14, 30, 90, 9999],
    labels=["RISK", "WARNING", "HEALTHY", "OVERSTOCK"]
)
print(df[["asin", "dos", "inventory_health"]].sort_values("dos"))
df.to_csv("inventory_health_tagged.csv", index=False)
print("第一个健康标签完成 ✓")
EOF

# Step 3: 建立第一个行动触发
# 过滤 RISK SKU → 生成补货提醒列表
```

---

*方案文档 v1.0 · 生成于 2026-06-18 · 基于 Palantir Ontology 方法论 + 726 个供应链 AI Skill Graph*

# FLOW-03 契约撰写 · 批次 03（6 份）

> 公共材料见 `flow-03-common.md`（**先读它**）。本批 5 份 A 模板 / 1 份 B 模板。


---

## CTR-A-018 · 供应商评估

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-014` OEM供应商开发与协同 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-02, FLOW-03, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-018-供应商评估.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：供应商打分可建成多准则决策/AHP-TOPSIS模型并输出可复核的评分与分级。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。
- **§F.5 边界条目**：多准则决策/AHP-TOPSIS 可输出可复核评分与分级 ⇒ 降级条件：若评分权重只能来自现场审核证据，则降 B

**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-02/STG-01 | D | 信号接收 | 接收需求证据、使用问题或产品组合缺口 | 新品机会信号 |
| FLOW-02/STG-02 | R | 范围确定与Case创建 | 确定用户问题、市场、产品范围、验证目标与停止条件 | 新品验证Case Charter |
| FLOW-02/STG-03 | D | 上下文装配 | 装配需求、竞品、产品定义、工程、OEM、质量、合规和验证能力 | FLOW-02 Context Manifest |
| FLOW-02/STG-06 | R | Assurance接收门禁 | 检查可证伪性、实物/专业证据来源、合规、资源与混杂因素 | FLOW-02 Assurance Decision |
| FLOW-02/STG-07 | D | 受控动作 | 发起受控外部验证Intent或记录无需动作；不得把模型判断当实物回执 | 验证尝试或无动作记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 27 张
    - `p2s-agentic-sckg-risk` · 源卡号 `Skill-Agentic-SCKG-Risk` · 08-知识图谱 · Agentic SCKG Risk Analyzer — 供应链知识图谱智能风险分析框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agentic-sckg-risk/SKILL.md`
    - `p2s-climate-esg-supply-chain-tag` · 源卡号 `Skill-Climate-ESG-Supply-Chain-Tag` · 24-标签工程 · 气候与ESG供应链风险标签 — 碳足迹追踪、CSRD合规与可持续供应链评级
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-climate-esg-supply-chain-tag/SKILL.md`
    - `p2s-counterfeit-detection-supply-chain` · 源卡号 `Skill-Counterfeit-Detection-Supply-Chain` · 21-合规决策 · Counterfeit Detection Supply Chain — 供应链假货检测与溯源（区块链+图谱+视觉指纹）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-counterfeit-detection-supply-chain/SKILL.md`
    - `p2s-geopolitical-risk-tag-supply-impact` · 源卡号 `Skill-Geopolitical-Risk-Tag-Supply-Impact` · 24-标签工程 · 地缘政治风险供应链影响标签 — 贸易限制/区域冲突对供应链的实时风险量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-geopolitical-risk-tag-supply-impact/SKILL.md`
    - `p2s-graph-rag-knowledge-retrieval` · 源卡号 `Skill-Graph-RAG-Knowledge-Retrieval` · 08-知识图谱 · Graph RAG Knowledge Retrieval — 知识图谱增强检索：结构化知识驱动精准问答
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-graph-rag-knowledge-retrieval/SKILL.md`
    - `p2s-helicase-supply-chain-kg-mas` · 源卡号 `Skill-Helicase-Supply-Chain-KG-MAS` · 10-MAS · Helicase — 不确定性感知供应链知识图谱：多 Agent 自主构建
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-helicase-supply-chain-kg-mas/SKILL.md`
    - `p2s-kg-rag-structured-knowledge-reasoning` · 源卡号 `Skill-KG-RAG-Structured-Knowledge-Reasoning` · 08-知识图谱 · KG-RAG — 知识图谱结构化推理路径增强
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-rag-structured-knowledge-reasoning/SKILL.md`
    - `p2s-kg-supply-chain-cost-attribution` · 源卡号 `Skill-KG-Supply-Chain-Cost-Attribution` · 08-知识图谱 · KG Supply Chain Cost Attribution — 图神经网络 + 因果推断的供应链成本归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-supply-chain-cost-attribution/SKILL.md`
    - `p2s-multi-factory-capacity-allocation` · 源卡号 `Skill-Multi-Factory-Capacity-Allocation` · 04-供应链 · 多工厂产能分配算法 — OEM/ODM多厂商协同下的弹性产能调度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-factory-capacity-allocation/SKILL.md`
    - `p2s-otif-on-time-in-full-analytics` · 源卡号 `Skill-OTIF-On-Time-In-Full-Analytics` · 04-供应链 · OTIF准时足量交货分析 — 供应商交货履约率量化与预测性缓冲策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-otif-on-time-in-full-analytics/SKILL.md`
    - `p2s-sc-resilience-hypergraph` · 源卡号 `Skill-SC-Resilience-Hypergraph` · 04-供应链 · Supply Chain Resilience Hypergraph — 超图神经网络供应链韧性推断
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sc-resilience-hypergraph/SKILL.md`
    - `p2s-social-engineering-attack-detection` · 源卡号 `Skill-Social-Engineering-Attack-Detection` · 19-风控反欺诈 · Social Engineering Attack Detection — 社会工程攻击检测钓鱼邮件/虚假供应商识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-social-engineering-attack-detection/SKILL.md`
    - `p2s-supplier-capacity-planning` · 源卡号 `Skill-Supplier-Capacity-Planning` · 04-供应链 · Multi-objective multi-site supplier selection and order spli
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-capacity-planning/SKILL.md`
    - `p2s-supplier-delivery-quality-rate-kpi` · 源卡号 `Skill-Supplier-Delivery-Quality-Rate-KPI` · 04-供应链 · 供应商来料质量IQC-KPI体系 — 入库检验合格率、质量成本与批次拒收分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-delivery-quality-rate-kpi/SKILL.md`
    - `p2s-supplier-development-roadmap-tracking` · 源卡号 `Skill-Supplier-Development-Roadmap-Tracking` · 04-供应链 · 供应商发展路线图追踪 — 改进计划执行跟踪与供应商升降级Tag管理
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-development-roadmap-tracking/SKILL.md`
    - `p2s-supplier-evaluation-model` · 源卡号 `Skill-Supplier-Evaluation-Model` · 06-增长模型 · Supplier Evaluation Model（供应商评估模型）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-evaluation-model/SKILL.md`
    - `p2s-supplier-ontology-capability-map` · 源卡号 `Skill-Supplier-Ontology-Capability-Map` · 24-标签工程 · 供应商本体能力图谱 — 产能/质量/认证/风险四维供应商Ontology设计与实例化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-ontology-capability-map/SKILL.md`
    - `p2s-supplier-performance-alert-action` · 源卡号 `Skill-Supplier-Performance-Alert-Action` · 04-供应链 · Supplier-Performance-Alert-Action — 供应商OTIF连续不达标自动触发备选供应商激活
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-performance-alert-action/SKILL.md`
    - `p2s-supplier-performance-scorecard` · 源卡号 `Skill-Supplier-Performance-Scorecard` · 04-供应链 · Supplier Performance Scorecard — 供应商绩效量化追踪系统
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-performance-scorecard/SKILL.md`
    - `p2s-supplier-qualification-onboarding-kpi` · 源卡号 `Skill-Supplier-Qualification-Onboarding-KPI` · 04-供应链 · 供应商准入认证KPI体系 — 新供应商评估、认证审核与准入门槛量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-qualification-onboarding-kpi/SKILL.md`
    - `p2s-supplier-risk-xgboost` · 源卡号 `Skill-Supplier-Risk-XGBoost` · 04-供应链 · Supplier Risk XGBoost — AHP-TOPSIS+XGBoost 供应商风险评分
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-risk-xgboost/SKILL.md`
    - `p2s-supply-chain-carbon-risk-gcn` · 源卡号 `Skill-Supply-Chain-Carbon-Risk-GCN` · 04-供应链 · 供应链碳排放风险 — 元学习GCN低碳转型评估
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-carbon-risk-gcn/SKILL.md`
    - `p2s-supply-chain-due-diligence` · 源卡号 `Skill-Supply-Chain-Due-Diligence` · 21-合规决策 · Supply Chain Due Diligence — 供应链合规尽职调查：劳工+环境+产品三维
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-due-diligence/SKILL.md`
    - `p2s-supply-chain-finance-risk-tag` · 源卡号 `Skill-Supply-Chain-Finance-Risk-Tag` · 24-标签工程 · 供应链金融风险标签 — 融资依赖度/信用评级/现金流压力的综合风险画像
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-finance-risk-tag/SKILL.md`
    - `p2s-supply-chain-network-betweenness-resilience` · 源卡号 `Skill-Supply-Chain-Network-Betweenness-Resilience` · 04-供应链 · 供应链网络中介中心性韧性分析 — 复杂网络科学迁移至供应链瓶颈识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-network-betweenness-resilience/SKILL.md`
    - `p2s-supply-chain-resilience-modeling` · 源卡号 `Skill-Supply-Chain-Resilience-Modeling` · 04-供应链 · Supply Chain Resilience Modeling — 供应链韧性建模：断链风险预测与备选方案规划
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-resilience-modeling/SKILL.md`
    - `p2s-supply-chain-total-cost-tco-model` · 源卡号 `Skill-Supply-Chain-Total-Cost-TCO-Model` · 04-供应链 · 全供应链总成本TCO模型 — 采购+仓储+物流+质量全链路成本分摊与年降目标
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-total-cost-tco-model/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 27。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-024 · 供需协调

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-016` 需求预测与补货计划 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-024-供需协调.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：供需平衡可建为多仓多期分配与S&OP优化模型。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-01/STG-01 | D | 信号接收 | 接收日常经营节奏或销量、转化、库存、广告异常 | 经营信号记录 |
| FLOW-01/STG-02 | R | 范围确定与Case创建 | 确定渠道、市场、账号、商品集合、时间窗、GMV口径与完成条件 | 经营Case Charter |
| FLOW-01/STG-03 | D | 上下文装配 | 按渠道选择主岗位，装配GMV、库存、价格、广告、财务和增量评估能力 | FLOW-01 Context Manifest |
| FLOW-01/STG-06 | R | Assurance接收门禁 | 检查数据质量、库存、预算、现金、账号范围和增量判断 | FLOW-01 Assurance Decision |
| FLOW-01/STG-07 | D | 受控动作 | 提交受策略约束的价格/广告Action Intent，或形成NoActionRecord | 动作尝试或无动作记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 22 张
    - `p2s-ad-spend-inventory-sync` · 源卡号 `Skill-Ad-Spend-Inventory-Sync` · 13-广告分析 · Ad Spend Inventory Sync — 广告投放与库存联动的协同优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ad-spend-inventory-sync/SKILL.md`
    - `p2s-bullwhip-effect-kalman-mitigation` · 源卡号 `Skill-Bullwhip-Effect-Kalman-Mitigation` · 04-供应链 · Bullwhip Effect Kalman Mitigation—用 Kalman Filter 消除牛鞭效应
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bullwhip-effect-kalman-mitigation/SKILL.md`
    - `p2s-bullwhip-effect-mitigation` · 源卡号 `Skill-Bullwhip-Effect-Mitigation` · 04-供应链 · Bullwhip Effect Mitigation — 牛鞭效应量化与 LNN+XGBoost 抑制
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bullwhip-effect-mitigation/SKILL.md`
    - `p2s-causal-rl-decision-making` · 源卡号 `Skill-Causal-RL-Decision-Making` · 01-因果推断 · 因果强化学习 — 从相关驱动到因果驱动的决策优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-rl-decision-making/SKILL.md`
    - `p2s-demand-supply-matching-gap-analysis` · 源卡号 `Skill-Demand-Supply-Matching-Gap-Analysis` · 04-供应链 · 供需缺口分析与优先级分配决策 — 供给不足时的SKU优先级量化与分配算法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-demand-supply-matching-gap-analysis/SKILL.md`
    - `p2s-event-driven-demand-mas` · 源卡号 `Skill-Event-Driven-Demand-MAS` · 10-MAS · Event-Driven Demand MAS — 事件感知补货 MAS：大促/季节自动触发
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-event-driven-demand-mas/SKILL.md`
    - `p2s-flowr-supply-chain-mas` · 源卡号 `Skill-Flowr-Supply-Chain-MAS` · 10-MAS · Flowr — 零售供应链多 Agent 端到端自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-flowr-supply-chain-mas/SKILL.md`
    - `p2s-forecast-bias-adjustment-detection` · 源卡号 `Skill-Forecast-Bias-Adjustment-Detection` · 04-供应链 · 预测偏差加减码检测与校正 — 供应链计划主动修正行为的量化分析与偏差溯源
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-forecast-bias-adjustment-detection/SKILL.md`
    - `p2s-hierarchical-demand-forecasting-reconciliation` · 源卡号 `Skill-Hierarchical-Demand-Forecasting-Reconciliation` · 03-时间序列 · HiFoReAd 多层时序预测调和 - 母婴跨境补货分层一致性
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hierarchical-demand-forecasting-reconciliation/SKILL.md`
    - `p2s-llm-sc-multiagent-consensus-replenishment` · 源卡号 `Skill-LLM-SC-MultiAgent-Consensus-Replenishment` · 24-标签工程 · LLM多智能体共识补货决策 — InvAgent框架：需求/采购/仓储三方博弈自动达成最优
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-sc-multiagent-consensus-replenishment/SKILL.md`
    - `p2s-logistics-plan-three-dimension-accuracy` · 源卡号 `Skill-Logistics-Plan-Three-Dimension-Accuracy` · 04-供应链 · 物流计划进销存三维准确率体系 — 入库/存货/销售三层预测准确率量化与根因归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-plan-three-dimension-accuracy/SKILL.md`
    - `p2s-multi-channel-inventory-pooling` · 源卡号 `Skill-Multi-Channel-Inventory-Pooling` · 04-供应链 · Multi-Channel Inventory Pooling（多渠道库存池化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-channel-inventory-pooling/SKILL.md`
    - `p2s-multi-channel-inventory-sync` · 源卡号 `Skill-Multi-Channel-Inventory-Sync` · 04-供应链 · Multi-Channel Inventory Sync — 多渠道库存协同：Amazon+独立站+TikTok联动库存
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-channel-inventory-sync/SKILL.md`
    - `p2s-omnichannel-inventory-sync` · 源卡号 `Skill-Omnichannel-Inventory-Sync` · 04-供应链 · Omnichannel Inventory Sync — 跨站多平台库存实时同步
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-omnichannel-inventory-sync/SKILL.md`
    - `p2s-omnichannel-order-orchestration-mas` · 源卡号 `Skill-Omnichannel-Order-Orchestration-MAS` · 24-标签工程 · 全渠道订单编排MAS — 多智能体协同的跨平台订单统一调度与冲突消解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-omnichannel-order-orchestration-mas/SKILL.md`
    - `p2s-promo-stocktaking-sop-automation` · 源卡号 `Skill-Promo-Stocktaking-SOP-Automation` · 04-供应链 · 大促盘货S&OP流程自动化 — 备货追踪、预案生成与紧急补货闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promo-stocktaking-sop-automation/SKILL.md`
    - `p2s-promotion-logistics-surge-forecast` · 源卡号 `Skill-Promotion-Logistics-Surge-Forecast` · 15-营销投放分析 · Promotion Logistics Surge Forecast — 大促物流爆仓预测：营销-履约联动容量规划
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promotion-logistics-surge-forecast/SKILL.md`
    - `p2s-sc-causal-dag-e2e-attribution` · 源卡号 `Skill-SC-Causal-DAG-E2E-Attribution` · 24-标签工程 · 供应链端到端因果DAG归因框架 — Amazon PC算法+SCM实现缺货根因30分钟诊断
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sc-causal-dag-e2e-attribution/SKILL.md`
    - `p2s-scpa-autonomous-sc-planning-agent` · 源卡号 `Skill-SCPA-Autonomous-SC-Planning-Agent` · 16-智能体工程 · 自主供应链规划智能体SCPA — 自然语言→结构化规划报告的JD.com完整框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-scpa-autonomous-sc-planning-agent/SKILL.md`
    - `p2s-sop-sales-operations-planning` · 源卡号 `Skill-SOP-Sales-Operations-Planning` · 04-供应链 · S&OP销售与运营计划协同 — 需求供应全链路对齐与月度计划闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sop-sales-operations-planning/SKILL.md`
    - `p2s-supply-chain-causal-scm-attribution` · 源卡号 `Skill-Supply-Chain-Causal-SCM-Attribution` · 04-供应链 · Supply Chain Causal SCM — 数据驱动供应链根因归因：DAG + DoWhy GCM
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-causal-scm-attribution/SKILL.md`
    - `p2s-unified-cross-border-inventory-dispatch` · 源卡号 `Skill-Unified-Cross-Border-Inventory-Dispatch` · 04-供应链 · 一盘货跨境库存统一调度 — 多平台多国统一库存决策与分配引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-unified-cross-border-inventory-dispatch/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 22。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-030 · 到货异常追踪

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-019` 跨境物流与关务 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-03, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-030-到货异常追踪.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：在途时效与到货偏差可用生存分析与异常检测预测。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。
- **§F.5 边界条目**：生存分析与异常检测可预测到达偏差 ⇒ 降级条件：同上

**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |
| FLOW-04/STG-01 | D | 信号接收 | 接收新市场、平台、店铺或商品上线请求 | 市场进入信号 |
| FLOW-04/STG-02 | R | 范围确定与Case创建 | 确定市场、渠道账号、商品集合、时间窗和进入成功条件 | 市场进入Case Charter |
| FLOW-04/STG-03 | D | 上下文装配 | 装配市场机会、准入、IP/合规、本地化、履约、售后、结算和品牌能力 | FLOW-04 Context Manifest |
| FLOW-04/STG-06 | R | Assurance接收门禁 | 检查适用策略、市场规则、账号主体、内容、履约和结算准备 | FLOW-04 Assurance Decision |
| FLOW-04/STG-07 | D | 受控动作 | 提交受控上架Intent或NoActionRecord；具体市场和接口待配置 | 上架尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 14 张
    - `p2s-ar-logistics-visualization` · 源卡号 `Skill-AR-Logistics-Visualization` · 18-物流履约 · AR Logistics Visualization — 增强现实包裹可视化追踪：跨境物流透明化与客服AI视频答复
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ar-logistics-visualization/SKILL.md`
    - `p2s-cold-chain-temperature-monitoring` · 源卡号 `Skill-Cold-Chain-Temperature-Monitoring` · 18-物流履约 · 冷链温控全程监测 — 母婴辅食物流的温度合规智能预警
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cold-chain-temperature-monitoring/SKILL.md`
    - `p2s-cross-border-last-mile-routing` · 源卡号 `Skill-Cross-Border-Last-Mile-Routing` · 18-物流履约 · Cross-Border Last Mile Routing — 跨境最后一公里路由优化：时效×成本双目标决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-last-mile-routing/SKILL.md`
    - `p2s-customs-clearance-risk-scoring` · 源卡号 `Skill-Customs-Clearance-Risk-Scoring` · 18-物流履约 · Customs Clearance Risk Scoring — 跨境清关多维风险预警
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-customs-clearance-risk-scoring/SKILL.md`
    - `p2s-in-transit-inventory-tracking-visibility` · 源卡号 `Skill-In-Transit-Inventory-Tracking-Visibility` · 04-供应链 · 在途库存追踪与全链路可视化 — 海运/空运实物流信息流双轨监控
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-in-transit-inventory-tracking-visibility/SKILL.md`
    - `p2s-inbound-eta-accuracy-kpi` · 源卡号 `Skill-Inbound-ETA-Accuracy-KPI` · 04-供应链 · 在途ETA准确率与到货履约率KPI — 过程数据vs结果数据的全链路在途管理体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inbound-eta-accuracy-kpi/SKILL.md`
    - `p2s-last-mile-delivery-prediction` · 源卡号 `Skill-Last-Mile-Delivery-Prediction` · 18-物流履约 · Last-Mile Delivery Prediction（最后一公里配送时效预测）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-last-mile-delivery-prediction/SKILL.md`
    - `p2s-lead-time-distribution-risk-genqot` · 源卡号 `Skill-Lead-Time-Distribution-Risk-GenQOT` · 04-供应链 · Gen-QOT 提前期分布建模 - 动态安全库存防海运延误
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-lead-time-distribution-risk-genqot/SKILL.md`
    - `p2s-logistics-anomaly-fraud-signal` · 源卡号 `Skill-Logistics-Anomaly-Fraud-Signal` · 18-物流履约 · Logistics Anomaly Fraud Signal — 物流轨迹异常作为欺诈信号特征工程
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-anomaly-fraud-signal/SKILL.md`
    - `p2s-multicarrier-parcel-tracking-fusion` · 源卡号 `Skill-Multicarrier-Parcel-Tracking-Fusion` · 18-物流履约 · 多承运商包裹追踪融合 — 跨境物流可见性引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multicarrier-parcel-tracking-fusion/SKILL.md`
    - `p2s-parcel-damage-prediction` · 源卡号 `Skill-Parcel-Damage-Prediction` · 18-物流履约 · Parcel Damage Prediction — 包裹破损预测（物流轨迹+特征联合模型）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-parcel-damage-prediction/SKILL.md`
    - `p2s-port-congestion-eta-prediction` · 源卡号 `Skill-Port-Congestion-ETA-Prediction` · 18-物流履约 · 港口拥堵ETA预测 — 多因子动态到港时间估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-port-congestion-eta-prediction/SKILL.md`
    - `p2s-real-time-fleet-dynamic-routing` · 源卡号 `Skill-Real-Time-Fleet-Dynamic-Routing` · 18-物流履约 · 实时车队动态重路由 — 突发事件下的物流路径自适应优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-real-time-fleet-dynamic-routing/SKILL.md`
    - `p2s-shipment-risk-tag-realtime-tracker` · 源卡号 `Skill-Shipment-Risk-Tag-Realtime-Tracker` · 24-标签工程 · 在途货物实时风险标签追踪器 — 海运/空运全链路可视化与预警Tag实时更新
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-shipment-risk-tag-realtime-tracker/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 14。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-043 · 促销规划

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-026` 定价促销与商品组合 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-043-促销规划.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：促销排期与力度是带约束的因果增量优化（含蚕食校正与预算分配）。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-01/STG-01 | D | 信号接收 | 接收日常经营节奏或销量、转化、库存、广告异常 | 经营信号记录 |
| FLOW-01/STG-02 | R | 范围确定与Case创建 | 确定渠道、市场、账号、商品集合、时间窗、GMV口径与完成条件 | 经营Case Charter |
| FLOW-01/STG-03 | D | 上下文装配 | 按渠道选择主岗位，装配GMV、库存、价格、广告、财务和增量评估能力 | FLOW-01 Context Manifest |
| FLOW-01/STG-06 | R | Assurance接收门禁 | 检查数据质量、库存、预算、现金、账号范围和增量判断 | FLOW-01 Assurance Decision |
| FLOW-01/STG-07 | D | 受控动作 | 提交受策略约束的价格/广告Action Intent，或形成NoActionRecord | 动作尝试或无动作记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 4 张
    - `p2s-intelligent-prediction-doubly-robust` · 源卡号 `Skill-Intelligent-Prediction-Doubly-Robust` · 03-时间序列 · 智能预测 - 双重稳健估计 (Doubly Robust Estimation)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-intelligent-prediction-doubly-robust/SKILL.md`
    - `p2s-promotion-effectiveness` · 源卡号 `Skill-Promotion-Effectiveness` · 15-营销投放分析 · Promotion Effectiveness Evaluation with Causal ML
      - 全文卡：`/Users/lute/.dsh/skills/p2s-promotion-effectiveness/SKILL.md`
    - `p2s-tree-of-thoughts-planning` · 源卡号 `Skill-Tree-of-Thoughts-Planning` · 10-MAS · Tree of Thoughts — 树搜索式任务规划
      - 全文卡：`/Users/lute/.dsh/skills/p2s-tree-of-thoughts-planning/SKILL.md`
    - `p2s-uplift-modeling` · 源卡号 `Skill-Uplift-Modeling` · 01-因果推断 · Uplift Modeling (元学习框架)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-uplift-modeling/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 31 张
    - `p2s-abandoned-cart-recovery-ml` · 源卡号 `Skill-Abandoned-Cart-Recovery-ML` · 14-用户分析 · Abandoned Cart Recovery ML — 弃购挽回机器学习：预测意图×序列触达×个性化优惠
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-abandoned-cart-recovery-ml/SKILL.md`
    - `p2s-anchoring-effect-pricing-optimization` · 源卡号 `Skill-Anchoring-Effect-Pricing-Optimization` · 17-价格优化 · 锚定效应定价优化 — 划线原价最优锚定比率让相同折扣感知价值提升25%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-anchoring-effect-pricing-optimization/SKILL.md`
    - `p2s-causal-forest-hte-experiment` · 源卡号 `Skill-Causal-Forest-HTE-Experiment` · 02-A_B实验 · 因果森林异质处理效应 — 实验数据挖掘个体级 CATE
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-forest-hte-experiment/SKILL.md`
    - `p2s-causal-forest-hte-experiments` · 源卡号 `Skill-Causal-Forest-HTE-Experiments` · 02-A_B实验 · Causal Forest for HTE in Experiments — 实验数据的异质处理效应估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-forest-hte-experiments/SKILL.md`
    - `p2s-causal-time-series-causalimpact` · 源卡号 `Skill-Causal-Time-Series-CausalImpact` · 01-因果推断 · CausalImpact时序因果 — 政策/活动对时序指标的因果冲击估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-time-series-causalimpact/SKILL.md`
    - `p2s-causal-time-series-forecasting-gcf` · 源卡号 `Skill-Causal-Time-Series-Forecasting-GCF` · 03-时间序列 · 因果时间序列预测 - GCF 反事实需求建模
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-time-series-forecasting-gcf/SKILL.md`
    - `p2s-causal-uplift-modeling` · 源卡号 `Skill-Causal-Uplift-Modeling` · 01-因果推断 · Causal Uplift Modeling — 因果提升模型：识别"可说服者"的跨域基础层
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-uplift-modeling/SKILL.md`
    - `p2s-counterfactual-evaluation` · 源卡号 `Skill-Counterfactual-Evaluation` · 01-因果推断 · 反事实评估 — Potential Outcomes + 合成控制反事实基线
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-counterfactual-evaluation/SKILL.md`
    - `p2s-dml-cohort-causal-effect` · 源卡号 `Skill-DML-Cohort-Causal-Effect` · 01-因果推断 · DML Cohort 因果效应 - 群体异质性 HTE 估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dml-cohort-causal-effect/SKILL.md`
    - `p2s-delayed-conversion-causal-mtl` · 源卡号 `Skill-Delayed-Conversion-Causal-MTL` · 02-A_B实验 · 促销延迟转化因果多任务建模 — 大促期用户行为的反事实分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-delayed-conversion-causal-mtl/SKILL.md`
    - `p2s-flash-sale-price-optimization` · 源卡号 `Skill-Flash-Sale-Price-Optimization` · 17-价格优化 · 闪购定价优化 — 限时折扣期间最优折扣率与时间窗口计算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-flash-sale-price-optimization/SKILL.md`
    - `p2s-flash-sale-realtime-sellthrough-forecast` · 源卡号 `Skill-Flash-Sale-Realtime-Sellthrough-Forecast` · 04-供应链 · 大促实时售罄模拟与流量协同 — 电商大促中实时预测与紧急资源调配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-flash-sale-realtime-sellthrough-forecast/SKILL.md`
    - `p2s-guardrailed-cate-nba` · 源卡号 `Skill-Guardrailed-CATE-NBA` · 06-增长模型 · Guardrailed CATE-NBA
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-guardrailed-cate-nba/SKILL.md`
    - `p2s-heterogeneous-treatment-effect-xlearner` · 源卡号 `Skill-Heterogeneous-Treatment-Effect-XLearner` · 01-因果推断 · X-Learner异质处理效应 — 识别不同用户群的差异化因果效应
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-heterogeneous-treatment-effect-xlearner/SKILL.md`
    - `p2s-inpromo-realtime-decision-kpi` · 源卡号 `Skill-InPromo-Realtime-Decision-KPI` · 04-供应链 · 大促中实时决策KPI与流量协同阈值 — 售罄速率监控/流量协同触发/紧急干预决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inpromo-realtime-decision-kpi/SKILL.md`
    - `p2s-loss-aversion-promotion-design` · 源卡号 `Skill-Loss-Aversion-Promotion-Design` · 15-营销投放分析 · 损失厌恶促销设计 — 「限时最后3件」比「立省30元」点击率高35%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-loss-aversion-promotion-design/SKILL.md`
    - `p2s-markdown-optimization` · 源卡号 `Skill-Markdown-Optimization` · 17-价格优化 · Markdown Optimization（折扣清仓定价优化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-markdown-optimization/SKILL.md`
    - `p2s-marketing-driven-recommendation` · 源卡号 `Skill-Marketing-Driven-Recommendation` · 05-推荐系统 · Marketing-Driven Recommendation — 营销活动驱动的推荐系统：Promotion-Awar
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-marketing-driven-recommendation/SKILL.md`
    - `p2s-perishable-inventory-markdown-optimization` · 源卡号 `Skill-Perishable-Inventory-Markdown-Optimization` · 17-价格优化 · Perishable Inventory Markdown Optimization — 超市易腐品定价运筹学迁移到母婴
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-perishable-inventory-markdown-optimization/SKILL.md`
    - `p2s-personalized-promotion-targeting` · 源卡号 `Skill-Personalized-Promotion-Targeting` · 14-用户分析 · Personalized Promotion Targeting — 个性化促销定向：用户响应异质性建模
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-personalized-promotion-targeting/SKILL.md`
    - `p2s-policy-learning-observational` · 源卡号 `Skill-Policy-Learning-Observational` · 01-因果推断 · Policy Learning from Observational Data — 从观测数据学习最优决策策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-policy-learning-observational/SKILL.md`
    - `p2s-price-elasticity-estimation` · 源卡号 `Skill-Price-Elasticity-Estimation` · 17-价格优化 · Price Elasticity Estimation — 需求价格弹性估算：跨境 SKU 定价底线测算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-elasticity-estimation/SKILL.md`
    - `p2s-price-elasticity-time-series-fusion` · 源卡号 `Skill-Price-Elasticity-Time-Series-Fusion` · 03-时间序列 · Price Elasticity Time Series Fusion — 价格弹性×时间序列融合预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-elasticity-time-series-fusion/SKILL.md`
    - `p2s-promo-inventory-pulse-auto-trigger` · 源卡号 `Skill-Promo-Inventory-Pulse-Auto-Trigger` · 04-供应链 · Promo-Inventory-Pulse-Auto-Trigger — 大促前14天库存不足自动触发紧急补货与广告降速
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promo-inventory-pulse-auto-trigger/SKILL.md`
    - `p2s-promo-roi-attribution-supply-side` · 源卡号 `Skill-Promo-ROI-Attribution-Supply-Side` · 04-供应链 · 促销活动供应侧ROI归因 — 备货成本+促销库存持有成本+尾货损失的全成本核算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promo-roi-attribution-supply-side/SKILL.md`
    - `p2s-promoguardian-promotion-fraud-gnn` · 源卡号 `Skill-PromoGuardian-Promotion-Fraud-GNN` · 19-风控反欺诈 · PromoGuardian — 促销欺诈多关系图神经网络检测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promoguardian-promotion-fraud-gnn/SKILL.md`
    - `p2s-promotion-roi-pre-post-analysis` · 源卡号 `Skill-Promotion-ROI-Pre-Post-Analysis` · 23-运营财务 · Promotion ROI Pre-Post Analysis — 促销ROI前后对比分析（大促PnL归因）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promotion-roi-pre-post-analysis/SKILL.md`
    - `p2s-promotional-lift-decomposition` · 源卡号 `Skill-Promotional-Lift-Decomposition` · 03-时间序列 · Promotional Lift Decomposition — 促销提升分解剥离促销带来的虚假需求
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promotional-lift-decomposition/SKILL.md`
    - `p2s-rl-dynamic-promotion-optimization` · 源卡号 `Skill-RL-Dynamic-Promotion-Optimization` · 15-营销投放分析 · RL Dynamic Promotion Optimization — 强化学习动态促销优化：时机×力度×对象的联合决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-rl-dynamic-promotion-optimization/SKILL.md`
    - `p2s-tag-informed-dynamic-pricing` · 源卡号 `Skill-Tag-Informed-Dynamic-Pricing` · 24-标签工程 · Tag-Informed Dynamic Pricing — SKU标签驱动动态定价策略自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tag-informed-dynamic-pricing/SKILL.md`
    - `p2s-uplift-intervention-priority-queue` · 源卡号 `Skill-Uplift-Intervention-Priority-Queue` · 14-用户分析 · Uplift-Intervention-Priority-Queue — Uplift×LTV加权排序生成有限促销资源干
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-uplift-intervention-priority-queue/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 3 张
    - `Skill-Funnel-Causal-Coupon-Allocation` · 13-广告分析 · Skill-Funnel-Causal-Coupon-Allocation
      - 全文卡：`paper2skills-vault/13-广告分析/Skill-Funnel-Causal-Coupon-Allocation.md`
    - `Skill-MAS-Consumer-Behavior-Simulation` · 07-NLP-VOC · Skill-MAS-Consumer-Behavior-Simulation
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-Consumer-Behavior-Simulation.md`
    - `Skill-MAS-MARL-Dynamic-Pricing` · 07-NLP-VOC · Skill-MAS-MARL-Dynamic-Pricing
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-MARL-Dynamic-Pricing.md`

> 候选总数 38。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-062 · 经营预算

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-041` 经营财务与资金 |
| 域 / 面 | `DOM-07` / `PLN-MGT` |
| flows | FLOW-01, FLOW-02, FLOW-03, FLOW-07, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-062-经营预算.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：预算编制与滚动预测是约束优化与偏差追踪问题。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-08/STG-04 | M | 证据收集与诊断 | 核对经营结果、全成本、自治异常、新品验证和能力表现 | 经营与能力诊断 |
| FLOW-08/STG-05 | M | 方案与标准产物 | 形成资源调整、停止事项、策略或能力Change Proposal | 经营与能力变更建议 |
| FLOW-08/STG-08 | M | 结果核验、关闭与异步学习 | 核对资源/能力结果、观察窗和回退证据 | 经营复盘关闭记录 |
| FLOW-01/STG-01 | D | 信号接收 | 接收日常经营节奏或销量、转化、库存、广告异常 | 经营信号记录 |
| FLOW-01/STG-02 | R | 范围确定与Case创建 | 确定渠道、市场、账号、商品集合、时间窗、GMV口径与完成条件 | 经营Case Charter |
| FLOW-01/STG-03 | D | 上下文装配 | 按渠道选择主岗位，装配GMV、库存、价格、广告、财务和增量评估能力 | FLOW-01 Context Manifest |
| FLOW-01/STG-06 | R | Assurance接收门禁 | 检查数据质量、库存、预算、现金、账号范围和增量判断 | FLOW-01 Assurance Decision |
| FLOW-01/STG-07 | D | 受控动作 | 提交受策略约束的价格/广告Action Intent，或形成NoActionRecord | 动作尝试或无动作记录 |
| FLOW-02/STG-01 | D | 信号接收 | 接收需求证据、使用问题或产品组合缺口 | 新品机会信号 |
| FLOW-02/STG-02 | R | 范围确定与Case创建 | 确定用户问题、市场、产品范围、验证目标与停止条件 | 新品验证Case Charter |
| FLOW-02/STG-03 | D | 上下文装配 | 装配需求、竞品、产品定义、工程、OEM、质量、合规和验证能力 | FLOW-02 Context Manifest |
| FLOW-02/STG-06 | R | Assurance接收门禁 | 检查可证伪性、实物/专业证据来源、合规、资源与混杂因素 | FLOW-02 Assurance Decision |
| FLOW-02/STG-07 | D | 受控动作 | 发起受控外部验证Intent或记录无需动作；不得把模型判断当实物回执 | 验证尝试或无动作记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |
| FLOW-08/STG-01 | D | 信号接收 | 接收经营复盘节奏、重大目标偏差或能力退化信号 | 经营复盘信号 |
| FLOW-08/STG-02 | R | 范围确定与Case创建 | 确定经营范围、观察窗、资源/能力问题和完成条件 | 经营复盘Case Charter |
| FLOW-08/STG-03 | D | 上下文装配 | 装配经营分析、增量、财务、审计、组织、知识和运行能力 | FLOW-08 Context Manifest |
| FLOW-08/STG-06 | R | Assurance接收门禁 | 检查口径、现金、利益冲突、独立审计和观察窗口 | FLOW-08 Assurance Decision |
| FLOW-08/STG-07 | D | 受控动作 | 提交已有策略范围内的资源动作或Change Proposal；Bundle只能走D-023生命周期 | 受控动作或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 4 张
    - `p2s-budget-reforecast-rolling` · 源卡号 `Skill-Budget-Reforecast-Rolling` · 23-运营财务 · Budget Reforecast Rolling — 滚动预算再预测（月度滚动reforecast自动化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-budget-reforecast-rolling/SKILL.md`
    - `p2s-cross-border-cash-flow-forecasting` · 源卡号 `Skill-Cross-Border-Cash-Flow-Forecasting` · 04-供应链 · Cross-Border Cash Flow Forecasting（跨境电商现金流预测与融资窗口规划）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-cash-flow-forecasting/SKILL.md`
    - `p2s-procurement-budget-rolling-reforecast` · 源卡号 `Skill-Procurement-Budget-Rolling-Reforecast` · 04-供应链 · 采购预算滚动重测 — 季度滚动预测与偏差管理的动态预算调整机制
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-procurement-budget-rolling-reforecast/SKILL.md`
    - `p2s-step-back-prompting` · 源卡号 `Skill-Step-Back-Prompting` · 09-DataAgent-LLM · Step-Back Prompting — 抽象推理提升复杂问题准确率
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-step-back-prompting/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 4。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-023 · 纠正预防措施

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-018` 生产协同与质量控制 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-02, FLOW-03, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-023-纠正预防措施.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：CAPA有效性可用前后对比与复发率验证，但措施设计与责任分配属质量体系判断。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-02/STG-01 | D | 信号接收 | 接收需求证据、使用问题或产品组合缺口 | 新品机会信号 |
| FLOW-02/STG-02 | R | 范围确定与Case创建 | 确定用户问题、市场、产品范围、验证目标与停止条件 | 新品验证Case Charter |
| FLOW-02/STG-03 | D | 上下文装配 | 装配需求、竞品、产品定义、工程、OEM、质量、合规和验证能力 | FLOW-02 Context Manifest |
| FLOW-02/STG-06 | R | Assurance接收门禁 | 检查可证伪性、实物/专业证据来源、合规、资源与混杂因素 | FLOW-02 Assurance Decision |
| FLOW-02/STG-07 | D | 受控动作 | 发起受控外部验证Intent或记录无需动作；不得把模型判断当实物回执 | 验证尝试或无动作记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 5 张
    - `p2s-cs-supply-chain-feedback-loop-tag` · 源卡号 `Skill-CS-Supply-Chain-Feedback-Loop-Tag` · 24-标签工程 · 客服反馈供应链改善闭环 — 差评/投诉自动归因到供应链节点并触发改善Action
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cs-supply-chain-feedback-loop-tag/SKILL.md`
    - `p2s-causal-voc-sentiment-attribution` · 源卡号 `Skill-Causal-VOC-Sentiment-Attribution` · 01-因果推断 · Causal VOC Sentiment Attribution — DiD+情感分析识别评论情绪变化真实因果驱动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-voc-sentiment-attribution/SKILL.md`
    - `p2s-customer-complaint-supply-root-cause-kpi` · 源卡号 `Skill-Customer-Complaint-Supply-Root-Cause-KPI` · 04-供应链 · 客诉供应链根因KPI与闭环治理 — 供应链原因导致客诉率/处理时效/预防闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-customer-complaint-supply-root-cause-kpi/SKILL.md`
    - `p2s-return-root-cause-attribution-graph` · 源卡号 `Skill-Return-Root-Cause-Attribution-Graph` · 24-标签工程 · 退货根因归因图谱 — 多层次退货原因知识图谱与供应链改善闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-return-root-cause-attribution-graph/SKILL.md`
    - `p2s-voc-triggered-inventory-signal` · 源卡号 `Skill-VOC-Triggered-Inventory-Signal` · 07-NLP-VOC · VOC Triggered Inventory Signal — VOC触发库存信号（差评激增→供应链预警）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-voc-triggered-inventory-signal/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 5。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。

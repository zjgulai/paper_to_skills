# FLOW-06 契约撰写 · 批次 02（6 份）

> 公共材料见 `flow-06-common.md`（**先读它**）。本批 4 份 A 模板 / 2 份 B 模板。


---

## CTR-A-003 · 依赖协调

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-002` 场景自主编排与异常协调 |
| 域 / 面 | `DOM-01` / `PLN-MGT` |
| flows | FLOW-01, FLOW-02, FLOW-03, FLOW-04, FLOW-05, FLOW-06, FLOW-07, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-003-依赖协调.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：任务依赖与资源冲突是经典调度/约束满足问题（RCPSP/CP），可输出可行序与预留方案。
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
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
| FLOW-06/STG-04 | M | 证据收集与诊断 | 核对业务语义、对象映射、数据来源/质量、现有系统和复用可能 | 能力缺口诊断 |
| FLOW-06/STG-05 | M | 方案与标准产物 | 形成需求契约、数据契约或工具交付方案 | 数据/工具交付包 |
| FLOW-06/STG-08 | M | 结果核验、关闭与异步学习 | 核对业务验收、质量、运行结果并发出异步学习候选 | 能力交付关闭记录 |
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
| FLOW-04/STG-01 | D | 信号接收 | 接收新市场、平台、店铺或商品上线请求 | 市场进入信号 |
| FLOW-04/STG-02 | R | 范围确定与Case创建 | 确定市场、渠道账号、商品集合、时间窗和进入成功条件 | 市场进入Case Charter |
| FLOW-04/STG-03 | D | 上下文装配 | 装配市场机会、准入、IP/合规、本地化、履约、售后、结算和品牌能力 | FLOW-04 Context Manifest |
| FLOW-04/STG-06 | R | Assurance接收门禁 | 检查适用策略、市场规则、账号主体、内容、履约和结算准备 | FLOW-04 Assurance Decision |
| FLOW-04/STG-07 | D | 受控动作 | 提交受控上架Intent或NoActionRecord；具体市场和接口待配置 | 上架尝试或无动作记录 |
| FLOW-05/STG-01 | D | 信号接收 | 接收咨询、订单异常、售后或获许可的生命周期事件 | 客户旅程信号 |
| FLOW-05/STG-02 | R | 范围确定与Case创建 | 确定客户/订单最小范围、问题类型、许可状态和完成条件 | 客户Case Charter |
| FLOW-05/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配服务、体验、教育、CRM、隐私和质量能力 | FLOW-05 Context Manifest |
| FLOW-05/STG-06 | R | Assurance接收门禁 | 检查许可、隐私、健康风险、质量事件、补偿和触达边界 | FLOW-05 Assurance Decision |
| FLOW-05/STG-07 | D | 受控动作 | 提交退款/补救/触达Intent或NoActionRecord | 客户动作尝试或无动作记录 |
| FLOW-06/STG-01 | D | 信号接收 | 接收真实业务能力需求或数据错误 | 数据与工具需求信号 |
| FLOW-06/STG-02 | R | 范围确定与Case创建 | 确定业务任务、对象、现有方法、优先级、约束和验收结果 | 能力交付Case Charter |
| FLOW-06/STG-03 | D | 上下文装配 | 装配口径、数据质量、集成、访问、安全、可靠性和能力治理Skills | FLOW-06 Context Manifest |
| FLOW-06/STG-06 | R | Assurance接收门禁 | 检查Access、数据质量、运行恢复、安全和经营验收设计 | FLOW-06 Assurance Decision |
| FLOW-06/STG-07 | D | 受控动作 | 提交受控发布Change Proposal或NoActionRecord；Bundle变更转入发布生命周期 | 发布尝试或变更提案 |
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
> `paper2skills-research/data/contracts/flow-06-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 8 张
    - `p2s-autogen-multi-agent-conversation` · 源卡号 `Skill-AutoGen-Multi-Agent-Conversation` · 10-MAS · AutoGen — 多智能体对话编排框架
      - 全文卡：`/Users/lute/.dsh/skills/p2s-autogen-multi-agent-conversation/SKILL.md`
    - `p2s-camel-role-playing-agents` · 源卡号 `Skill-CAMEL-Role-Playing-Agents` · 10-MAS · CAMEL — 角色扮演式自主协作多 Agent 框架
      - 全文卡：`/Users/lute/.dsh/skills/p2s-camel-role-playing-agents/SKILL.md`
    - `p2s-mas-orchestrator` · 源卡号 `Skill-MAS-Orchestrator` · 10-MAS · MAS Orchestrator — 多智能体编排与调度
      - 全文卡：`/Users/lute/.dsh/skills/p2s-mas-orchestrator/SKILL.md`
    - `p2s-metagpt-sop-driven-collaboration` · 源卡号 `Skill-MetaGPT-SOP-Driven-Collaboration` · 10-MAS · MetaGPT — SOP 驱动的多智能体协作框架
      - 全文卡：`/Users/lute/.dsh/skills/p2s-metagpt-sop-driven-collaboration/SKILL.md`
    - `p2s-multi-agent-debate` · 源卡号 `Skill-Multi-Agent-Debate` · 10-MAS · Multi-Agent Debate — 多智能体辩论共识
      - 全文卡：`/Users/lute/.dsh/skills/p2s-multi-agent-debate/SKILL.md`
    - `p2s-orchestration-trace-rl` · 源卡号 `Skill-Orchestration-Trace-RL` · 16-智能体工程 · 编排轨迹驱动的强化学习 — MAS RL 三维设计框架
      - 全文卡：`/Users/lute/.dsh/skills/p2s-orchestration-trace-rl/SKILL.md`
    - `p2s-subagent-decomposition` · 源卡号 `Skill-Subagent-Decomposition` · 10-MAS · Subagent Decomposer — 复杂任务子智能体分解
      - 全文卡：`/Users/lute/.dsh/skills/p2s-subagent-decomposition/SKILL.md`
    - `p2s-task-adaptive-topology` · 源卡号 `Skill-Task-Adaptive-Topology` · 16-智能体工程 · 任务自适应拓扑路由 — AdaptOrch 动态多智能体编排
      - 全文卡：`/Users/lute/.dsh/skills/p2s-task-adaptive-topology/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 16 张
    - `p2s-agent-qmix-topology-learning` · 源卡号 `Skill-Agent-QMix-Topology-Learning` · 10-MAS · Agent Q-Mix — MARL 学习最优 MAS 通信拓扑（QMIX 值分解）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-qmix-topology-learning/SKILL.md`
    - `p2s-agent-skill-runtime-orchestrator` · 源卡号 `Skill-Agent-Skill-Runtime-Orchestrator` · 16-智能体工程 · Agent Skill Runtime Orchestrator — 运行时动态选取并执行 Skill 的编排框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-skill-runtime-orchestrator/SKILL.md`
    - `p2s-concat-consensus-decentralized-mas` · 源卡号 `Skill-CONCAT-Consensus-Decentralized-MAS` · 10-MAS · CONCAT共识驱动去中心化MAS协同 — 无需中央Orchestrator的自组织Agent网络
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-concat-consensus-decentralized-mas/SKILL.md`
    - `p2s-combo-inventory-crisis-response` · 源卡号 `Skill-Combo-Inventory-Crisis-Response` · 16-智能体工程 · 库存危机响应 Combo Pattern — 断货/积压异常触发的 5 步自动响应链路
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-combo-inventory-crisis-response/SKILL.md`
    - `p2s-combo-new-product-launch-playbook` · 源卡号 `Skill-Combo-New-Product-Launch-Playbook` · 16-智能体工程 · 新品上市全链路 Combo Pattern — 从蓝海选品到首月排名突破的 7 步编排
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-combo-new-product-launch-playbook/SKILL.md`
    - `p2s-cross-org-agent-protocol` · 源卡号 `Skill-Cross-Org-Agent-Protocol` · 10-MAS · Cross-Org Agent Protocol — 跨组织多智能体协调协议：多委托人、联邦编排、工作区委托
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-org-agent-protocol/SKILL.md`
    - `p2s-dag-task-decomposition-planning` · 源卡号 `Skill-DAG-Task-Decomposition-Planning` · 16-智能体工程 · TDP — DAG 任务解耦规划：82% Token 节省 + 错误隔离
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dag-task-decomposition-planning/SKILL.md`
    - `p2s-dynamic-dag-orchestration` · 源卡号 `Skill-Dynamic-DAG-Orchestration` · 10-MAS · Dynamic DAG Orchestration — 运行时动态调整工作流拓扑
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-dag-orchestration/SKILL.md`
    - `p2s-graph-grounded-mas-protocol` · 源卡号 `Skill-Graph-Grounded-MAS-Protocol` · 10-MAS · G²CP — 图结构 MAS 通信协议：消除级联幻觉
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-graph-grounded-mas-protocol/SKILL.md`
    - `p2s-mas-consensus-mechanism` · 源卡号 `Skill-MAS-Consensus-Mechanism` · 10-MAS · MAS Consensus Mechanism — 多智能体共识协议：分布式一致性与拜占庭容错
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-consensus-mechanism/SKILL.md`
    - `p2s-mas-cross-market-compliance-orchestrator` · 源卡号 `Skill-MAS-Cross-Market-Compliance-Orchestrator` · 10-MAS · MAS跨市场合规编排 — 多市场合规Agent并行处理与冲突解决
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-cross-market-compliance-orchestrator/SKILL.md`
    - `p2s-multi-agent-skill-composition` · 源卡号 `Skill-Multi-Agent-Skill-Composition` · 10-MAS · Multi-Agent Skill Composition — 多 Agent 协作 Skill 链式 DAG 编排
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-agent-skill-composition/SKILL.md`
    - `p2s-paramanager-parallel-orchestration` · 源卡号 `Skill-ParaManager-Parallel-Orchestration` · 10-MAS · ParaManager — 小模型主编排：Agent-as-Tool 并行子任务分解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-paramanager-parallel-orchestration/SKILL.md`
    - `p2s-resmas-resilience-topology-optimization` · 源卡号 `Skill-ResMAS-Resilience-Topology-Optimization` · 10-MAS · ResMAS韧性拓扑优化 — GNN韧性预测+GRPO拓扑生成+拓扑感知Prompt优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-resmas-resilience-topology-optimization/SKILL.md`
    - `p2s-sdof-state-constrained-orchestration` · 源卡号 `Skill-SDOF-State-Constrained-Orchestration` · 10-MAS · SDOF — 状态机约束 MAS 编排：屏蔽非法操作，任务完成率 86.5%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sdof-state-constrained-orchestration/SKILL.md`
    - `p2s-supply-chain-agent-orchestration-hub` · 源卡号 `Skill-Supply-Chain-Agent-Orchestration-Hub` · 24-标签工程 · 供应链Agent编排中枢 — 多Agent协作、任务分发与跨域决策自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-agent-orchestration-hub/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 2 张
    - `Skill-Multi-Agent-Collaboration-Tax` · 10-MAS · Skill-Multi-Agent-Collaboration-Tax
      - 全文卡：`paper2skills-vault/10-MAS/Skill-Multi-Agent-Collaboration-Tax.md`
    - `Skill-Routed-Graph-Handoff` · 10-MAS · Skill-Routed-Graph-Handoff
      - 全文卡：`paper2skills-vault/10-MAS/Skill-Routed-Graph-Handoff.md`

> 候选总数 26。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-038 · 漏斗诊断

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-023` 独立站经营与转化 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-04, FLOW-05, FLOW-06 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-038-漏斗诊断.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：漏斗逐层转化与流失定位是标准漏斗/路径分析与转化率建模。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
| FLOW-06/STG-04 | M | 证据收集与诊断 | 核对业务语义、对象映射、数据来源/质量、现有系统和复用可能 | 能力缺口诊断 |
| FLOW-06/STG-05 | M | 方案与标准产物 | 形成需求契约、数据契约或工具交付方案 | 数据/工具交付包 |
| FLOW-06/STG-08 | M | 结果核验、关闭与异步学习 | 核对业务验收、质量、运行结果并发出异步学习候选 | 能力交付关闭记录 |
| FLOW-01/STG-01 | D | 信号接收 | 接收日常经营节奏或销量、转化、库存、广告异常 | 经营信号记录 |
| FLOW-01/STG-02 | R | 范围确定与Case创建 | 确定渠道、市场、账号、商品集合、时间窗、GMV口径与完成条件 | 经营Case Charter |
| FLOW-01/STG-03 | D | 上下文装配 | 按渠道选择主岗位，装配GMV、库存、价格、广告、财务和增量评估能力 | FLOW-01 Context Manifest |
| FLOW-01/STG-06 | R | Assurance接收门禁 | 检查数据质量、库存、预算、现金、账号范围和增量判断 | FLOW-01 Assurance Decision |
| FLOW-01/STG-07 | D | 受控动作 | 提交受策略约束的价格/广告Action Intent，或形成NoActionRecord | 动作尝试或无动作记录 |
| FLOW-04/STG-01 | D | 信号接收 | 接收新市场、平台、店铺或商品上线请求 | 市场进入信号 |
| FLOW-04/STG-02 | R | 范围确定与Case创建 | 确定市场、渠道账号、商品集合、时间窗和进入成功条件 | 市场进入Case Charter |
| FLOW-04/STG-03 | D | 上下文装配 | 装配市场机会、准入、IP/合规、本地化、履约、售后、结算和品牌能力 | FLOW-04 Context Manifest |
| FLOW-04/STG-06 | R | Assurance接收门禁 | 检查适用策略、市场规则、账号主体、内容、履约和结算准备 | FLOW-04 Assurance Decision |
| FLOW-04/STG-07 | D | 受控动作 | 提交受控上架Intent或NoActionRecord；具体市场和接口待配置 | 上架尝试或无动作记录 |
| FLOW-05/STG-01 | D | 信号接收 | 接收咨询、订单异常、售后或获许可的生命周期事件 | 客户旅程信号 |
| FLOW-05/STG-02 | R | 范围确定与Case创建 | 确定客户/订单最小范围、问题类型、许可状态和完成条件 | 客户Case Charter |
| FLOW-05/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配服务、体验、教育、CRM、隐私和质量能力 | FLOW-05 Context Manifest |
| FLOW-05/STG-06 | R | Assurance接收门禁 | 检查许可、隐私、健康风险、质量事件、补偿和触达边界 | FLOW-05 Assurance Decision |
| FLOW-05/STG-07 | D | 受控动作 | 提交退款/补救/触达Intent或NoActionRecord | 客户动作尝试或无动作记录 |
| FLOW-06/STG-01 | D | 信号接收 | 接收真实业务能力需求或数据错误 | 数据与工具需求信号 |
| FLOW-06/STG-02 | R | 范围确定与Case创建 | 确定业务任务、对象、现有方法、优先级、约束和验收结果 | 能力交付Case Charter |
| FLOW-06/STG-03 | D | 上下文装配 | 装配口径、数据质量、集成、访问、安全、可靠性和能力治理Skills | FLOW-06 Context Manifest |
| FLOW-06/STG-06 | R | Assurance接收门禁 | 检查Access、数据质量、运行恢复、安全和经营验收设计 | FLOW-06 Assurance Decision |
| FLOW-06/STG-07 | D | 受控动作 | 提交受控发布Change Proposal或NoActionRecord；Bundle变更转入发布生命周期 | 发布尝试或变更提案 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-06-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 2 张
    - `p2s-customer-journey-prototype` · 源卡号 `Skill-Customer-Journey-Prototype` · 06-增长模型 · Customer Journey Prototype Detection 客户旅程序列原型检测
      - 全文卡：`/Users/lute/.dsh/skills/p2s-customer-journey-prototype/SKILL.md`
    - `p2s-user-funnel-analysis` · 源卡号 `Skill-User-Funnel-Analysis` · 14-用户分析 · User Funnel and Behavior Path Analysis
      - 全文卡：`/Users/lute/.dsh/skills/p2s-user-funnel-analysis/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 16 张
    - `p2s-ad-to-behavior-funnel` · 源卡号 `Skill-Ad-to-Behavior-Funnel` · 13-广告分析 · Ad-to-Behavior Funnel（广告→用户行为漏斗）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ad-to-behavior-funnel/SKILL.md`
    - `p2s-conformal-roi-prediction` · 源卡号 `Skill-Conformal-ROI-Prediction` · 01-因果推断 · 共形预测ROI区间估计 - 小样本下的可信转化贡献量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-conformal-roi-prediction/SKILL.md`
    - `p2s-cross-border-member-onboarding-optimization` · 源卡号 `Skill-Cross-Border-Member-Onboarding-Optimization` · 06-增长模型 · Cross-Border Member Onboarding Optimization — 跨境会员注册漏斗优化结合早期
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-member-onboarding-optimization/SKILL.md`
    - `p2s-customer-journey-analytics` · 源卡号 `Skill-Customer-Journey-Analytics` · 14-用户分析 · Customer Journey Analytics — 用户旅程分析：全链路转化漏斗诊断与优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-customer-journey-analytics/SKILL.md`
    - `p2s-epicscore-uncertainty` · 源卡号 `Skill-EPICSCORE-Uncertainty` · 01-因果推断 · 认知不确定性共形评分 - 数据稀疏区域自适应区间加宽
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-epicscore-uncertainty/SKILL.md`
    - `p2s-listing-suppression-detection` · 源卡号 `Skill-Listing-Suppression-Detection` · 19-风控反欺诈 · Listing Suppression Detection — Listing 被平台隐藏/降权检测（非账号问题）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-suppression-detection/SKILL.md`
    - `p2s-mas-customer-journey-orchestration` · 源卡号 `Skill-MAS-Customer-Journey-Orchestration` · 10-MAS · 多 Agent 客户旅程编排 — 从首曝光到复购的全链路 Agent 协作
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-customer-journey-orchestration/SKILL.md`
    - `p2s-nonitem-page-path-modeling` · 源卡号 `Skill-NonItem-Page-Path-Modeling` · 14-用户分析 · 非商品页路径建模 - 导航页在用户旅程中的转化贡献
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-nonitem-page-path-modeling/SKILL.md`
    - `p2s-search-funnel-attribution` · 源卡号 `Skill-Search-Funnel-Attribution` · 25-搜索流量工程 · 搜索漏斗分归因 — 搜索→展示→点击→加购→购买各层转化拆解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-funnel-attribution/SKILL.md`
    - `p2s-sequential-user-behavior-modeling` · 源卡号 `Skill-Sequential-User-Behavior-Modeling` · 14-用户分析 · Sequential User Behavior Modeling — 用户行为序列建模：时序上下文驱动的意图理解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sequential-user-behavior-modeling/SKILL.md`
    - `p2s-session-intent-shift` · 源卡号 `Skill-Session-Intent-Shift` · 14-用户分析 · Session意图漂移建模 - 跨会话用户购买意图变化检测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-session-intent-shift/SKILL.md`
    - `p2s-sparse-matrix-completion` · 源卡号 `Skill-Sparse-Matrix-Completion` · 14-用户分析 · 超稀疏矩阵补全 - 每行仅2-5个观测值的页面转移矩阵恢复
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sparse-matrix-completion/SKILL.md`
    - `p2s-tag-driven-user-behavior-analytics` · 源卡号 `Skill-Tag-Driven-User-Behavior-Analytics` · 24-标签工程 · Tag-Driven User Behavior Analytics — 实时行为事件流标签化与动态用户画像
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tag-driven-user-behavior-analytics/SKILL.md`
    - `p2s-tiktok-shop-content-commerce-funnel` · 源卡号 `Skill-TikTok-Shop-Content-Commerce-Funnel` · 15-营销投放分析 · TikTok Shop 内容电商漏斗建模 — 短视频→直播→下单三段式转化率优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-shop-content-commerce-funnel/SKILL.md`
    - `p2s-trajectory-pattern-mining` · 源卡号 `Skill-Trajectory-Pattern-Mining` · 14-用户分析 · 用户行为轨迹模式挖掘与预测 - 变阶马尔可夫模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-trajectory-pattern-mining/SKILL.md`
    - `p2s-utimac-uncertainty-completion` · 源卡号 `Skill-Utimac-Uncertainty-Completion` · 14-用户分析 · 不确定性感知矩阵补全 - 补全值带置信区间的页面转移矩阵恢复
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-utimac-uncertainty-completion/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 18。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-066 · 账号商品映射

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-045` 业务口径与主数据 |
| 域 / 面 | `DOM-08` / `PLN-PLT` |
| flows | FLOW-01, FLOW-06, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-066-账号商品映射.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：跨账号/平台SKU映射是实体解析（entity resolution）与相似匹配问题。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-06/STG-04 | M | 证据收集与诊断 | 核对业务语义、对象映射、数据来源/质量、现有系统和复用可能 | 能力缺口诊断 |
| FLOW-06/STG-05 | M | 方案与标准产物 | 形成需求契约、数据契约或工具交付方案 | 数据/工具交付包 |
| FLOW-06/STG-08 | M | 结果核验、关闭与异步学习 | 核对业务验收、质量、运行结果并发出异步学习候选 | 能力交付关闭记录 |
| FLOW-08/STG-04 | M | 证据收集与诊断 | 核对经营结果、全成本、自治异常、新品验证和能力表现 | 经营与能力诊断 |
| FLOW-08/STG-05 | M | 方案与标准产物 | 形成资源调整、停止事项、策略或能力Change Proposal | 经营与能力变更建议 |
| FLOW-08/STG-08 | M | 结果核验、关闭与异步学习 | 核对资源/能力结果、观察窗和回退证据 | 经营复盘关闭记录 |
| FLOW-01/STG-01 | D | 信号接收 | 接收日常经营节奏或销量、转化、库存、广告异常 | 经营信号记录 |
| FLOW-01/STG-02 | R | 范围确定与Case创建 | 确定渠道、市场、账号、商品集合、时间窗、GMV口径与完成条件 | 经营Case Charter |
| FLOW-01/STG-03 | D | 上下文装配 | 按渠道选择主岗位，装配GMV、库存、价格、广告、财务和增量评估能力 | FLOW-01 Context Manifest |
| FLOW-01/STG-06 | R | Assurance接收门禁 | 检查数据质量、库存、预算、现金、账号范围和增量判断 | FLOW-01 Assurance Decision |
| FLOW-01/STG-07 | D | 受控动作 | 提交受策略约束的价格/广告Action Intent，或形成NoActionRecord | 动作尝试或无动作记录 |
| FLOW-06/STG-01 | D | 信号接收 | 接收真实业务能力需求或数据错误 | 数据与工具需求信号 |
| FLOW-06/STG-02 | R | 范围确定与Case创建 | 确定业务任务、对象、现有方法、优先级、约束和验收结果 | 能力交付Case Charter |
| FLOW-06/STG-03 | D | 上下文装配 | 装配口径、数据质量、集成、访问、安全、可靠性和能力治理Skills | FLOW-06 Context Manifest |
| FLOW-06/STG-06 | R | Assurance接收门禁 | 检查Access、数据质量、运行恢复、安全和经营验收设计 | FLOW-06 Assurance Decision |
| FLOW-06/STG-07 | D | 受控动作 | 提交受控发布Change Proposal或NoActionRecord；Bundle变更转入发布生命周期 | 发布尝试或变更提案 |
| FLOW-08/STG-01 | D | 信号接收 | 接收经营复盘节奏、重大目标偏差或能力退化信号 | 经营复盘信号 |
| FLOW-08/STG-02 | R | 范围确定与Case创建 | 确定经营范围、观察窗、资源/能力问题和完成条件 | 经营复盘Case Charter |
| FLOW-08/STG-03 | D | 上下文装配 | 装配经营分析、增量、财务、审计、组织、知识和运行能力 | FLOW-08 Context Manifest |
| FLOW-08/STG-06 | R | Assurance接收门禁 | 检查口径、现金、利益冲突、独立审计和观察窗口 | FLOW-08 Assurance Decision |
| FLOW-08/STG-07 | D | 受控动作 | 提交已有策略范围内的资源动作或Change Proposal；Bundle只能走D-023生命周期 | 受控动作或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-06-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 7 张
    - `p2s-cross-platform-user-identity-resolution` · 源卡号 `Skill-Cross-Platform-User-Identity-Resolution` · 22-数据采集工程 · Cross-Platform User Identity Resolution — 跨平台用户 ID 统一
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-platform-user-identity-resolution/SKILL.md`
    - `p2s-cross-platform-user-identity` · 源卡号 `Skill-Cross-Platform-User-Identity` · 22-数据采集工程 · 跨平台用户 ID 统一 — Amazon/TikTok/独立站身份解析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-platform-user-identity/SKILL.md`
    - `p2s-entity-resolution-kg-dedup` · 源卡号 `Skill-Entity-Resolution-KG-Dedup` · 08-知识图谱 · KG 实体消歧与去重（Entity Resolution & Deduplication）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-entity-resolution-kg-dedup/SKILL.md`
    - `p2s-privacy-safe-identity-resolution` · 源卡号 `Skill-Privacy-Safe-Identity-Resolution` · 22-数据采集工程 · Privacy-Safe Identity Resolution — 隐私合规跨平台 ID 解析：多方对齐与差分隐私
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-privacy-safe-identity-resolution/SKILL.md`
    - `p2s-product-knowledge-graph-query` · 源卡号 `Skill-Product-Knowledge-Graph-Query` · 08-知识图谱 · Product KG Query — 多 Agent 产品知识图谱查询与 SKU 跨平台映射
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-knowledge-graph-query/SKILL.md`
    - `p2s-sku-entity-unified-id-tagging` · 源卡号 `Skill-SKU-Entity-Unified-ID-Tagging` · 24-标签工程 · SKU跨平台实体统一标识与标签同步 — ASIN/SKU/ERP三码合一的实体对齐与标签一致性保障
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sku-entity-unified-id-tagging/SKILL.md`
    - `p2s-sku-master-data-golden-record` · 源卡号 `Skill-SKU-Master-Data-Golden-Record` · 24-标签工程 · SKU主数据黄金记录治理 — MDM体系、主键统一与多源冲突消解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sku-master-data-golden-record/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 7。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-072 · 容量管理

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-049` Agent平台与可靠运行 |
| 域 / 面 | `DOM-08` / `PLN-PLT` |
| flows | FLOW-06, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-072-容量管理.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：容量与成本核算可用排队论、负载预测与预算优化。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-06/STG-04 | M | 证据收集与诊断 | 核对业务语义、对象映射、数据来源/质量、现有系统和复用可能 | 能力缺口诊断 |
| FLOW-06/STG-05 | M | 方案与标准产物 | 形成需求契约、数据契约或工具交付方案 | 数据/工具交付包 |
| FLOW-06/STG-08 | M | 结果核验、关闭与异步学习 | 核对业务验收、质量、运行结果并发出异步学习候选 | 能力交付关闭记录 |
| FLOW-08/STG-04 | M | 证据收集与诊断 | 核对经营结果、全成本、自治异常、新品验证和能力表现 | 经营与能力诊断 |
| FLOW-08/STG-05 | M | 方案与标准产物 | 形成资源调整、停止事项、策略或能力Change Proposal | 经营与能力变更建议 |
| FLOW-08/STG-08 | M | 结果核验、关闭与异步学习 | 核对资源/能力结果、观察窗和回退证据 | 经营复盘关闭记录 |
| FLOW-06/STG-01 | D | 信号接收 | 接收真实业务能力需求或数据错误 | 数据与工具需求信号 |
| FLOW-06/STG-02 | R | 范围确定与Case创建 | 确定业务任务、对象、现有方法、优先级、约束和验收结果 | 能力交付Case Charter |
| FLOW-06/STG-03 | D | 上下文装配 | 装配口径、数据质量、集成、访问、安全、可靠性和能力治理Skills | FLOW-06 Context Manifest |
| FLOW-06/STG-06 | R | Assurance接收门禁 | 检查Access、数据质量、运行恢复、安全和经营验收设计 | FLOW-06 Assurance Decision |
| FLOW-06/STG-07 | D | 受控动作 | 提交受控发布Change Proposal或NoActionRecord；Bundle变更转入发布生命周期 | 发布尝试或变更提案 |
| FLOW-08/STG-01 | D | 信号接收 | 接收经营复盘节奏、重大目标偏差或能力退化信号 | 经营复盘信号 |
| FLOW-08/STG-02 | R | 范围确定与Case创建 | 确定经营范围、观察窗、资源/能力问题和完成条件 | 经营复盘Case Charter |
| FLOW-08/STG-03 | D | 上下文装配 | 装配经营分析、增量、财务、审计、组织、知识和运行能力 | FLOW-08 Context Manifest |
| FLOW-08/STG-06 | R | Assurance接收门禁 | 检查口径、现金、利益冲突、独立审计和观察窗口 | FLOW-08 Assurance Decision |
| FLOW-08/STG-07 | D | 受控动作 | 提交已有策略范围内的资源动作或Change Proposal；Bundle只能走D-023生命周期 | 受控动作或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-06-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 4 张
    - `p2s-active-context-pruning` · 源卡号 `Skill-Active-Context-Pruning` · 16-智能体工程 · 仿生粘菌主动上下文剪枝 — Focus Agent 自主压缩架构
      - 全文卡：`/Users/lute/.dsh/skills/p2s-active-context-pruning/SKILL.md`
    - `p2s-context-compression` · 源卡号 `Skill-Context-Compression` · 16-智能体工程 · ACON — Agent 长上下文压缩与 NL 准则优化
      - 全文卡：`/Users/lute/.dsh/skills/p2s-context-compression/SKILL.md`
    - `p2s-open-source-tool-use-model` · 源卡号 `Skill-Open-Source-Tool-Use-Model` · 16-智能体工程 · 开源 Tool Use 基座模型选型 — Hermes 4 混合推理家族
      - 全文卡：`/Users/lute/.dsh/skills/p2s-open-source-tool-use-model/SKILL.md`
    - `p2s-slm-tool-calling-optimization` · 源卡号 `Skill-SLM-Tool-Calling-Optimization` · 16-智能体工程 · SLM Tool Calling 成本优化 — 350M 参数击败 LLM
      - 全文卡：`/Users/lute/.dsh/skills/p2s-slm-tool-calling-optimization/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 30 张
    - `p2s-ab-testing-platform-infrastructure` · 源卡号 `Skill-AB-Testing-Platform-Infrastructure` · 02-A_B实验 · AB Testing Platform Infrastructure — A/B 实验平台基础设施：可扩展的在线实验框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ab-testing-platform-infrastructure/SKILL.md`
    - `p2s-ai-carbon-footprint-optimizer` · 源卡号 `Skill-AI-Carbon-Footprint-Optimizer` · 11-AI人文 · 母婴出海AI碳足迹优化器 — 绿色供应链的能耗与排放量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-carbon-footprint-optimizer/SKILL.md`
    - `p2s-adactx-dynamic-context-budget-allocation` · 源卡号 `Skill-AdaCtx-Dynamic-Context-Budget-Allocation` · 10-MAS · AdaCtx动态上下文预算分配 — 子Agent间Token预算自适应调度与边际价值追踪
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-adactx-dynamic-context-budget-allocation/SKILL.md`
    - `p2s-adaptive-crawl-scheduling` · 源卡号 `Skill-Adaptive-Crawl-Scheduling` · 22-数据采集工程 · Adaptive Crawl Scheduling — 自适应爬取调度：Sleeping Bandit + 神经质量优先
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-adaptive-crawl-scheduling/SKILL.md`
    - `p2s-agent-cost-optimization-budget-control` · 源卡号 `Skill-Agent-Cost-Optimization-Budget-Control` · 16-智能体工程 · Agent 成本优化与预算管控 — 让 LLM Agent 可持续运行的 Token 经济学
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-cost-optimization-budget-control/SKILL.md`
    - `p2s-agent-production-engineering` · 源卡号 `Skill-Agent-Production-Engineering` · 10-MAS · Skill Card: Agent Production Engineering（Agent 生产化工程）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-production-engineering/SKILL.md`
    - `p2s-agent-registry-discovery` · 源卡号 `Skill-Agent-Registry-Discovery` · 10-MAS · Agent Registry & Discovery — 动态 Agent 能力注册与路由
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-registry-discovery/SKILL.md`
    - `p2s-bayesian-hyperparameter-optimization` · 源卡号 `Skill-Bayesian-Hyperparameter-Optimization` · 12-ML基础 · 贝叶斯超参搜索 — Optuna/BO 替代随机搜索
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bayesian-hyperparameter-optimization/SKILL.md`
    - `p2s-caster-context-aware-model-routing` · 源卡号 `Skill-CASTER-Context-Aware-Model-Routing` · 10-MAS · CASTER上下文感知模型路由 — 任务步骤级强弱LLM动态调度以破解性能-成本铁三角
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-caster-context-aware-model-routing/SKILL.md`
    - `p2s-context-token-compression` · 源卡号 `Skill-Context-Token-Compression` · 10-MAS · 上下文Token压缩 — Summarizer Agent语义保真压缩与成本效益优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-context-token-compression/SKILL.md`
    - `p2s-cost-aware-agent-scheduling` · 源卡号 `Skill-Cost-Aware-Agent-Scheduling` · 16-智能体工程 · Cost-Aware Agent Scheduling（成本感知智能体调度）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cost-aware-agent-scheduling/SKILL.md`
    - `p2s-graph-knowledge-distillation-recommendation` · 源卡号 `Skill-Graph-Knowledge-Distillation-Recommendation` · 08-知识图谱 · Graph Knowledge Distillation Recommendation — 图知识蒸馏推荐：轻量化GNN
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-graph-knowledge-distillation-recommendation/SKILL.md`
    - `p2s-hnsw-ann-vector-index-engineering` · 源卡号 `Skill-HNSW-ANN-Vector-Index-Engineering` · 08-知识图谱 · HNSW — 向量索引工程与 ANN 检索优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hnsw-ann-vector-index-engineering/SKILL.md`
    - `p2s-kv-cache-optimization-agent` · 源卡号 `Skill-KV-Cache-Optimization-Agent` · 16-智能体工程 · KV-Cache优化 — Agent推理内存效率的系统级提升
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kv-cache-optimization-agent/SKILL.md`
    - `p2s-llmlingua-context-compression` · 源卡号 `Skill-LLMLingua-Context-Compression` · 16-智能体工程 · LLMLingua-2 — 大语言模型上下文压缩
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llmlingua-context-compression/SKILL.md`
    - `p2s-lmm-searcher-multimodal-context` · 源卡号 `Skill-LMM-Searcher-Multimodal-Context` · 16-智能体工程 · LMM-Searcher — 长链多模态 Agent：UID 占位符按需加载图片
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-lmm-searcher-multimodal-context/SKILL.md`
    - `p2s-longrag-long-context-hybrid` · 源卡号 `Skill-LongRAG-Long-Context-Hybrid` · 08-知识图谱 · LongRAG — 长上下文与RAG混合决策策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-longrag-long-context-hybrid/SKILL.md`
    - `p2s-mas-resource-scheduling` · 源卡号 `Skill-MAS-Resource-Scheduling` · 10-MAS · MAS Resource Scheduling — OS 调度原语驱动的多智能体资源管理
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-resource-scheduling/SKILL.md`
    - `p2s-mas-scale-management` · 源卡号 `Skill-MAS-Scale-Management` · 10-MAS · MAS Scale Management — 大规模多智能体集群管理：万级并发、单调扩展、公司制架构
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-scale-management/SKILL.md`
    - `p2s-ml-model-serving-optimization` · 源卡号 `Skill-ML-Model-Serving-Optimization` · 16-智能体工程 · ML Model Serving Optimization — 知识蒸馏 + 量化 + 异步推理的模型生产化部署
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ml-model-serving-optimization/SKILL.md`
    - `p2s-matryoshka-representation-learning` · 源卡号 `Skill-Matryoshka-Representation-Learning` · 08-知识图谱 · Matryoshka表示学习 — 嵌套多尺度嵌入压缩
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-matryoshka-representation-learning/SKILL.md`
    - `p2s-metaie-unified-information-extraction-distillation` · 源卡号 `Skill-MetaIE-Unified-Information-Extraction-Distillation` · 07-NLP-VOC · MetaIE — 统一信息抽取蒸馏框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-metaie-unified-information-extraction-distillation/SKILL.md`
    - `p2s-mixture-of-experts-routing` · 源卡号 `Skill-Mixture-of-Experts-Routing` · 12-ML基础 · Mixture of Experts Routing — MoE 路由机制（多任务条件激活，降低推理成本）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mixture-of-experts-routing/SKILL.md`
    - `p2s-property-graph-query-optimization` · 源卡号 `Skill-Property-Graph-Query-Optimization` · 08-知识图谱 · Property Graph Query Optimization — 属性图查询工程
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-property-graph-query-optimization/SKILL.md`
    - `p2s-rcr-router-role-aware-context-routing` · 源卡号 `Skill-RCR-Router-Role-Aware-Context-Routing` · 10-MAS · RCR-Router角色感知上下文路由 — Token预算约束下的多Agent记忆子集动态分配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-rcr-router-role-aware-context-routing/SKILL.md`
    - `p2s-semantic-cache-rag-acceleration` · 源卡号 `Skill-Semantic-Cache-RAG-Acceleration` · 16-智能体工程 · Skill: Skill-Semantic-Cache-RAG-RAG-Acceleration
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-semantic-cache-rag-acceleration/SKILL.md`
    - `p2s-speculative-decoding-agent` · 源卡号 `Skill-Speculative-Decoding-Agent` · 16-智能体工程 · 投机解码推理加速 — LLM Agent运行成本降低的核心引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-speculative-decoding-agent/SKILL.md`
    - `p2s-speculative-rag` · 源卡号 `Skill-Speculative-RAG` · 08-知识图谱 · Speculative RAG — 推测性检索加速框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-speculative-rag/SKILL.md`
    - `p2s-tokenpilot-lifecycle-context-eviction` · 源卡号 `Skill-TokenPilot-Lifecycle-Context-Eviction` · 10-MAS · TokenPilot生命周期感知上下文驱逐 — 双粒度上下文管理：摄入压实+驱逐调度
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tokenpilot-lifecycle-context-eviction/SKILL.md`
    - `p2s-vectordb-production-engineering` · 源卡号 `Skill-VectorDB-Production-Engineering` · 08-知识图谱 · Skill-VectorDB-Production-Engineering
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-vectordb-production-engineering/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 1 张
    - `Skill-Stateful-Skill-Runtime` · 16-智能体工程 · Skill-Stateful-Skill-Runtime
      - 全文卡：`paper2skills-vault/16-智能体工程/Skill-Stateful-Skill-Runtime.md`

> 候选总数 35。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-055 · 产品准入核对

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-044` 产品合规与隐私 |
| 域 / 面 | `DOM-07` / `PLN-CTL` |
| flows | FLOW-02, FLOW-04, FLOW-05, FLOW-06, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-055-产品准入核对.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：准入矩阵可规则化+法规库匹配，但最终准入须法定依据与外部认证。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
| FLOW-06/STG-04 | M | 证据收集与诊断 | 核对业务语义、对象映射、数据来源/质量、现有系统和复用可能 | 能力缺口诊断 |
| FLOW-06/STG-05 | M | 方案与标准产物 | 形成需求契约、数据契约或工具交付方案 | 数据/工具交付包 |
| FLOW-06/STG-08 | M | 结果核验、关闭与异步学习 | 核对业务验收、质量、运行结果并发出异步学习候选 | 能力交付关闭记录 |
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-02/STG-01 | D | 信号接收 | 接收需求证据、使用问题或产品组合缺口 | 新品机会信号 |
| FLOW-02/STG-02 | R | 范围确定与Case创建 | 确定用户问题、市场、产品范围、验证目标与停止条件 | 新品验证Case Charter |
| FLOW-02/STG-03 | D | 上下文装配 | 装配需求、竞品、产品定义、工程、OEM、质量、合规和验证能力 | FLOW-02 Context Manifest |
| FLOW-02/STG-06 | R | Assurance接收门禁 | 检查可证伪性、实物/专业证据来源、合规、资源与混杂因素 | FLOW-02 Assurance Decision |
| FLOW-02/STG-07 | D | 受控动作 | 发起受控外部验证Intent或记录无需动作；不得把模型判断当实物回执 | 验证尝试或无动作记录 |
| FLOW-04/STG-01 | D | 信号接收 | 接收新市场、平台、店铺或商品上线请求 | 市场进入信号 |
| FLOW-04/STG-02 | R | 范围确定与Case创建 | 确定市场、渠道账号、商品集合、时间窗和进入成功条件 | 市场进入Case Charter |
| FLOW-04/STG-03 | D | 上下文装配 | 装配市场机会、准入、IP/合规、本地化、履约、售后、结算和品牌能力 | FLOW-04 Context Manifest |
| FLOW-04/STG-06 | R | Assurance接收门禁 | 检查适用策略、市场规则、账号主体、内容、履约和结算准备 | FLOW-04 Assurance Decision |
| FLOW-04/STG-07 | D | 受控动作 | 提交受控上架Intent或NoActionRecord；具体市场和接口待配置 | 上架尝试或无动作记录 |
| FLOW-05/STG-01 | D | 信号接收 | 接收咨询、订单异常、售后或获许可的生命周期事件 | 客户旅程信号 |
| FLOW-05/STG-02 | R | 范围确定与Case创建 | 确定客户/订单最小范围、问题类型、许可状态和完成条件 | 客户Case Charter |
| FLOW-05/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配服务、体验、教育、CRM、隐私和质量能力 | FLOW-05 Context Manifest |
| FLOW-05/STG-06 | R | Assurance接收门禁 | 检查许可、隐私、健康风险、质量事件、补偿和触达边界 | FLOW-05 Assurance Decision |
| FLOW-05/STG-07 | D | 受控动作 | 提交退款/补救/触达Intent或NoActionRecord | 客户动作尝试或无动作记录 |
| FLOW-06/STG-01 | D | 信号接收 | 接收真实业务能力需求或数据错误 | 数据与工具需求信号 |
| FLOW-06/STG-02 | R | 范围确定与Case创建 | 确定业务任务、对象、现有方法、优先级、约束和验收结果 | 能力交付Case Charter |
| FLOW-06/STG-03 | D | 上下文装配 | 装配口径、数据质量、集成、访问、安全、可靠性和能力治理Skills | FLOW-06 Context Manifest |
| FLOW-06/STG-06 | R | Assurance接收门禁 | 检查Access、数据质量、运行恢复、安全和经营验收设计 | FLOW-06 Assurance Decision |
| FLOW-06/STG-07 | D | 受控动作 | 提交受控发布Change Proposal或NoActionRecord；Bundle变更转入发布生命周期 | 发布尝试或变更提案 |
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-06-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 46 张
    - `p2s-ai-product-safety-certification` · 源卡号 `Skill-AI-Product-Safety-Certification` · 21-合规决策 · AI Product Safety Certification — AI驱动产品安全认证自动化：从测试报告解析到合规路径
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-product-safety-certification/SKILL.md`
    - `p2s-baby-food-allergen-label-validator` · 源卡号 `Skill-Baby-Food-Allergen-Label-Validator` · 21-合规决策 · 婴儿食品过敏原标签验证器 — FALCPA/FASTER Act九大过敏原自动合规校验
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-baby-food-allergen-label-validator/SKILL.md`
    - `p2s-cpsc-children-product-safety` · 源卡号 `Skill-CPSC-Children-Product-Safety` · 21-合规决策 · CPSC 儿童产品安全合规（美国强制认证）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cpsc-children-product-safety/SKILL.md`
    - `p2s-cpsc-efiling-auto-mapper` · 源卡号 `Skill-CPSC-eFiling-Auto-Mapper` · 21-合规决策 · CPSC eFiling Auto-Mapper — NLP驱动的电子申报字段自动填充
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cpsc-efiling-auto-mapper/SKILL.md`
    - `p2s-california-prop65-label-check` · 源卡号 `Skill-California-Prop65-Label-Check` · 21-合规决策 · 加州65号提案标签合规 — 母婴产品化学物质自动检查
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-california-prop65-label-check/SKILL.md`
    - `p2s-category-compliance-prescan` · 源卡号 `Skill-Category-Compliance-Prescan` · 21-合规决策 · Skill-Category-Compliance-Prescan
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-category-compliance-prescan/SKILL.md`
    - `p2s-climate-esg-supply-chain-tag` · 源卡号 `Skill-Climate-ESG-Supply-Chain-Tag` · 24-标签工程 · 气候与ESG供应链风险标签 — 碳足迹追踪、CSRD合规与可持续供应链评级
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-climate-esg-supply-chain-tag/SKILL.md`
    - `p2s-corrective-rag-crag` · 源卡号 `Skill-Corrective-RAG-CRAG` · 08-知识图谱 · Corrective-RAG — 纠错式检索增强生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-corrective-rag-crag/SKILL.md`
    - `p2s-cross-border-compliance-framework` · 源卡号 `Skill-Cross-Border-Compliance-Framework` · 21-合规决策 · Cross-Border Compliance Framework — 跨境电商多辖区合规自动映射
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-compliance-framework/SKILL.md`
    - `p2s-crossborder-customs-compliance-rate-kpi` · 源卡号 `Skill-CrossBorder-Customs-Compliance-Rate-KPI` · 04-供应链 · 跨境关检务合规率KPI体系 — 清关时效/合规申报率/风险等级分类的全流程量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-crossborder-customs-compliance-rate-kpi/SKILL.md`
    - `p2s-deeprag-step-by-step-retrieval` · 源卡号 `Skill-DeepRAG-Step-by-Step-Retrieval` · 08-知识图谱 · DeepRAG — 逐步推理驱动的原子检索决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-deeprag-step-by-step-retrieval/SKILL.md`
    - `p2s-deepseek-r1-rag-reasoning` · 源卡号 `Skill-DeepSeek-R1-RAG-Reasoning` · 08-知识图谱 · DeepSeek-R1 RAG推理增强 — 逐步推理驱动的自主检索决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-deepseek-r1-rag-reasoning/SKILL.md`
    - `p2s-docre-document-level-relation-extraction` · 源卡号 `Skill-DocRE-Document-Level-Relation-Extraction` · 08-知识图谱 · Skill-DocRE-Document-Level-Relation-Extraction
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-docre-document-level-relation-extraction/SKILL.md`
    - `p2s-epr-auto-calculator` · 源卡号 `Skill-EPR-Auto-Calculator` · 21-合规决策 · EPR扩展生产者责任自动费用测算 — 欧盟/英国/法国合规
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-epr-auto-calculator/SKILL.md`
    - `p2s-epr-extended-producer-responsibility-tag` · 源卡号 `Skill-EPR-Extended-Producer-Responsibility-Tag` · 24-标签工程 · 欧盟EPR扩大生产者责任标签体系 — EPR注册合规、包装标签与回收报告自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-epr-extended-producer-responsibility-tag/SKILL.md`
    - `p2s-frames-rag-factuality-multihop` · 源卡号 `Skill-FRAMES-RAG-Factuality-Multihop` · 08-知识图谱 · Skill-FRAMES-RAG-Factuality-Multihop
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-frames-rag-factuality-multihop/SKILL.md`
    - `p2s-gcc-cpc-document-validator` · 源卡号 `Skill-GCC-CPC-Document-Validator` · 21-合规决策 · GCC/CPC Document Validator — 合规认证文档完整性自动验证
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gcc-cpc-document-validator/SKILL.md`
    - `p2s-gpsr-eu-risk-assessment-auto` · 源卡号 `Skill-GPSR-EU-Risk-Assessment-Auto` · 21-合规决策 · GPSR EU Risk Assessment Auto — 欧盟GPSR风险评估自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gpsr-eu-risk-assessment-auto/SKILL.md`
    - `p2s-green-supply-chain-carbon-footprint` · 源卡号 `Skill-Green-Supply-Chain-Carbon-Footprint` · 04-供应链 · Green Supply Chain Carbon Footprint — 绿色供应链碳足迹：ESG合规与可持续运营优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-green-supply-chain-carbon-footprint/SKILL.md`
    - `p2s-hcce-concept-hierarchy-embedding` · 源卡号 `Skill-HCCE-Concept-Hierarchy-Embedding` · 08-知识图谱 · HCCE — 超球面锥概念层次嵌入
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hcce-concept-hierarchy-embedding/SKILL.md`
    - `p2s-hts-code-risk-classifier` · 源卡号 `Skill-HTS-Code-Risk-Classifier` · 21-合规决策 · HTS Code Risk Classifier — 基于HTS码的CPSC多标签风险分类
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hts-code-risk-classifier/SKILL.md`
    - `p2s-ip-infringement-risk-scan` · 源卡号 `Skill-IP-Infringement-Risk-Scan` · 21-合规决策 · IP Infringement Risk Scan — 上架前知识产权侵权风险自动扫描（商标+专利+著作权）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ip-infringement-risk-scan/SKILL.md`
    - `p2s-in-context-learning-tabular` · 源卡号 `Skill-In-Context-Learning-Tabular` · 12-ML基础 · In-Context Learning表格数据 — 无需训练的少样本电商数据分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-in-context-learning-tabular/SKILL.md`
    - `p2s-infant-formula-fda-compliance-checker` · 源卡号 `Skill-Infant-Formula-FDA-Compliance-Checker` · 21-合规决策 · 婴儿配方奶粉FDA合规检查器 — 自动验证21 CFR Part 107标签合规性
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-infant-formula-fda-compliance-checker/SKILL.md`
    - `p2s-knowledge-conflict-detection-llm` · 源卡号 `Skill-Knowledge-Conflict-Detection-LLM` · 08-知识图谱 · 知识冲突检测 — 参数知识与外部知识的一致性对齐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-knowledge-conflict-detection-llm/SKILL.md`
    - `p2s-mas-compliance-multi-market-orchestrator` · 源卡号 `Skill-MAS-Compliance-Multi-Market-Orchestrator` · 10-MAS · MAS-Compliance-Multi-Market-Orchestrator — 多市场合规检查Agent并行编排与
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-compliance-multi-market-orchestrator/SKILL.md`
    - `p2s-mas-cross-market-compliance-orchestrator` · 源卡号 `Skill-MAS-Cross-Market-Compliance-Orchestrator` · 10-MAS · MAS跨市场合规编排 — 多市场合规Agent并行处理与冲突解决
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-cross-market-compliance-orchestrator/SKILL.md`
    - `p2s-medrag-domain-vertical-rag` · 源卡号 `Skill-MedRAG-Domain-Vertical-RAG` · 08-知识图谱 · 垂直领域RAG — 医疗→电商迁移学习的最佳实践
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-medrag-domain-vertical-rag/SKILL.md`
    - `p2s-multi-market-compliance-matrix-ontology` · 源卡号 `Skill-Multi-Market-Compliance-Matrix-Ontology` · 24-标签工程 · 多市场合规矩阵本体 — US/EU/JP/AU跨境合规要求统一建模与差异分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-market-compliance-matrix-ontology/SKILL.md`
    - `p2s-multimodal-table-understanding` · 源卡号 `Skill-Multimodal-Table-Understanding` · 09-DataAgent-LLM · Multimodal Table Understanding Agent — 表格理解：规格对比/认证矩阵/价格表
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multimodal-table-understanding/SKILL.md`
    - `p2s-new-sku-launch-readiness-gate` · 源卡号 `Skill-New-SKU-Launch-Readiness-Gate` · 04-供应链 · 新品上市准入门控 — 从选品到发布的全量检查清单与Tag驱动的上市就绪评估
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-new-sku-launch-readiness-gate/SKILL.md`
    - `p2s-pre-launch-compliance-gate` · 源卡号 `Skill-Pre-Launch-Compliance-Gate` · 21-合规决策 · Pre-Launch-Compliance-Gate — 新品上架前合规评分低于阈值自动阻断并触发修复工作流
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-pre-launch-compliance-gate/SKILL.md`
    - `p2s-product-safety-complaint-risk-model` · 源卡号 `Skill-Product-Safety-Complaint-Risk-Model` · 19-风控反欺诈 · Product Safety Complaint Risk Model — 产品安全投诉风险模型基于历史预测封号概率
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-safety-complaint-risk-model/SKILL.md`
    - `p2s-product-safety-testing-requirements` · 源卡号 `Skill-Product-Safety-Testing-Requirements` · 21-合规决策 · Product Safety Testing Requirements — 产品安全测试需求：品类×市场映射
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-safety-testing-requirements/SKILL.md`
    - `p2s-rag-cot-interleaved-reasoning` · 源卡号 `Skill-RAG-CoT-Interleaved-Reasoning` · 08-知识图谱 · Skill-RAG-CoT-Interleaved-Reasoning
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-rag-cot-interleaved-reasoning/SKILL.md`
    - `p2s-reach-chemical-compliance` · 源卡号 `Skill-REACH-Chemical-Compliance` · 21-合规决策 · REACH化学品合规 — 母婴玩具/床品化学物质限制自动检查
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-reach-chemical-compliance/SKILL.md`
    - `p2s-recommendation-compliance-filter` · 源卡号 `Skill-Recommendation-Compliance-Filter` · 05-推荐系统 · Recommendation Compliance Filter — 推荐系统实时合规内容过滤
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-recommendation-compliance-filter/SKILL.md`
    - `p2s-regulatory-change-auto-monitor` · 源卡号 `Skill-Regulatory-Change-Auto-Monitor` · 21-合规决策 · Regulatory Change Auto-Monitor — 合规法规变更自动监控：实时追踪政策更新的预警系统
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-regulatory-change-auto-monitor/SKILL.md`
    - `p2s-regulatory-change-impact-propagation` · 源卡号 `Skill-Regulatory-Change-Impact-Propagation` · 24-标签工程 · 法规变更影响传播引擎 — 新规 → 受影响SKU/市场的Tag传播与行动触发
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-regulatory-change-impact-propagation/SKILL.md`
    - `p2s-regulatory-change-monitoring` · 源卡号 `Skill-Regulatory-Change-Monitoring` · 21-合规决策 · Regulatory Change Monitoring — 法规变更自动监控：受影响品类实时映射
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-regulatory-change-monitoring/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 46。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-061 · 集成验证

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-047` 系统集成与业务工具 |
| 域 / 面 | `DOM-08` / `PLN-PLT` |
| flows | FLOW-06 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-061-集成验证.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：契约测试、幂等与回执核对可完全自动执行，但属工程管道，算法非责任实质。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-06/STG-04 | M | 证据收集与诊断 | 核对业务语义、对象映射、数据来源/质量、现有系统和复用可能 | 能力缺口诊断 |
| FLOW-06/STG-05 | M | 方案与标准产物 | 形成需求契约、数据契约或工具交付方案 | 数据/工具交付包 |
| FLOW-06/STG-08 | M | 结果核验、关闭与异步学习 | 核对业务验收、质量、运行结果并发出异步学习候选 | 能力交付关闭记录 |
| FLOW-06/STG-01 | D | 信号接收 | 接收真实业务能力需求或数据错误 | 数据与工具需求信号 |
| FLOW-06/STG-02 | R | 范围确定与Case创建 | 确定业务任务、对象、现有方法、优先级、约束和验收结果 | 能力交付Case Charter |
| FLOW-06/STG-03 | D | 上下文装配 | 装配口径、数据质量、集成、访问、安全、可靠性和能力治理Skills | FLOW-06 Context Manifest |
| FLOW-06/STG-06 | R | Assurance接收门禁 | 检查Access、数据质量、运行恢复、安全和经营验收设计 | FLOW-06 Assurance Decision |
| FLOW-06/STG-07 | D | 受控动作 | 提交受控发布Change Proposal或NoActionRecord；Bundle变更转入发布生命周期 | 发布尝试或变更提案 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-06-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 1 张
    - `p2s-mcp-a2a-protocol-stack` · 源卡号 `Skill-MCP-A2A-Protocol-Stack` · 16-智能体工程 · MCP + A2A 双协议栈 — Orchestrated Multi-Agent 企业架构
      - 全文卡：`/Users/lute/.dsh/skills/p2s-mcp-a2a-protocol-stack/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 5 张
    - `p2s-agent-production-engineering` · 源卡号 `Skill-Agent-Production-Engineering` · 10-MAS · Skill Card: Agent Production Engineering（Agent 生产化工程）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-production-engineering/SKILL.md`
    - `p2s-ldp-identity-aware-protocol` · 源卡号 `Skill-LDP-Identity-Aware-Protocol` · 16-智能体工程 · LDP — 身份感知 Agent 通信协议：模型级路由 + 37% Token 节省
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ldp-identity-aware-protocol/SKILL.md`
    - `p2s-llm-schema-inference` · 源卡号 `Skill-LLM-Schema-Inference` · 09-DataAgent-LLM · LLM 自动推断数据结构 — 无 schema 的电商 API 接入
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-schema-inference/SKILL.md`
    - `p2s-mas-testing-verification` · 源卡号 `Skill-MAS-Testing-Verification` · 10-MAS · MAS Testing & Verification — 多智能体系统测试验证：覆盖制导 Fuzzing + 跨框架可观
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-testing-verification/SKILL.md`
    - `p2s-sc-agent-mcp-erp-integration` · 源卡号 `Skill-SC-Agent-MCP-ERP-Integration` · 16-智能体工程 · 供应链智能体MCP多ERP集成 — Model Context Protocol驱动的多系统双向协调模式
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sc-agent-mcp-erp-integration/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 6。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。

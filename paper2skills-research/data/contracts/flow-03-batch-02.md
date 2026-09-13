# FLOW-03 契约撰写 · 批次 02（6 份）

> 公共材料见 `flow-03-common.md`（**先读它**）。本批 5 份 A 模板 / 1 份 B 模板。


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
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

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

## CTR-A-023 · 补货模拟

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-016` 需求预测与补货计划 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-023-补货模拟.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：补货策略可用(s,S)/报童/多级库存仿真直接求解并输出服务水平-成本前沿。
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

- 有全文卡（优先选，可选到论文参数）：共 6 张
    - `p2s-multi-echelon-inventory` · 源卡号 `Skill-Multi-Echelon-Inventory` · 04-供应链 · Multi-Echelon Inventory Optimization (多阶库存优化)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-multi-echelon-inventory/SKILL.md`
    - `p2s-prophet-forecasting` · 源卡号 `Skill-Prophet-Forecasting` · 03-时间序列 · Prophet Forecasting with Seasonality and Holidays
      - 全文卡：`/Users/lute/.dsh/skills/p2s-prophet-forecasting/SKILL.md`
    - `p2s-safety-stock-replenishment` · 源卡号 `Skill-Safety-Stock-Replenishment` · 04-供应链 · Safety Stock and Replenishment Strategy
      - 全文卡：`/Users/lute/.dsh/skills/p2s-safety-stock-replenishment/SKILL.md`
    - `p2s-temporal-fusion-transformer` · 源卡号 `Skill-Temporal-Fusion-Transformer` · 03-时间序列 · 'Skill: Temporal Fusion Transformer (TFT) 多水平时序预测'
      - 全文卡：`/Users/lute/.dsh/skills/p2s-temporal-fusion-transformer/SKILL.md`
    - `p2s-time-series-forecasting` · 源卡号 `Skill-Time-Series-Forecasting` · 03-时间序列 · Skill Card: 时间序列预测 (Time Series Forecasting)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-time-series-forecasting/SKILL.md`
    - `p2s-two-echelon-inventory-drl` · 源卡号 `Skill-Two-Echelon-Inventory-DRL` · 04-供应链 · Deep RL for Two-Echelon Inventory Optimization
      - 全文卡：`/Users/lute/.dsh/skills/p2s-two-echelon-inventory-drl/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 81 张
    - `p2s-aim-rm-llm-inventory-mas-memory` · 源卡号 `Skill-AIM-RM-LLM-Inventory-MAS-Memory` · 10-MAS · AIM-RM — LLM 多 Agent 库存管理：历史经验相似匹配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-aim-rm-llm-inventory-mas-memory/SKILL.md`
    - `p2s-arima-garch-demand-volatility` · 源卡号 `Skill-ARIMA-GARCH-Demand-Volatility` · 03-时间序列 · ARIMA-GARCH Demand Volatility — 需求波动率预测（不确定性区间建模）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-arima-garch-demand-volatility/SKILL.md`
    - `p2s-adaptive-reorder-point-kalman` · 源卡号 `Skill-Adaptive-Reorder-Point-Kalman` · 04-供应链 · 自适应补货点 Kalman 版 — 让 ROP 随市场需求动态漂移
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-adaptive-reorder-point-kalman/SKILL.md`
    - `p2s-automated-replenishment-decision-engine` · 源卡号 `Skill-Automated-Replenishment-Decision-Engine` · 04-供应链 · Automated Replenishment Decision Engine — 备货决策自动化引擎：从规则到智能补货
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-automated-replenishment-decision-engine/SKILL.md`
    - `p2s-cvar-inventory-risk-portfolio` · 源卡号 `Skill-CVaR-Inventory-Risk-Portfolio` · 04-供应链 · CVaR多SKU库存风险组合 — 金融条件风险价值迁移至库存尾部风险管理
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cvar-inventory-risk-portfolio/SKILL.md`
    - `p2s-causal-ml-feature-engineering` · 源卡号 `Skill-Causal-ML-Feature-Engineering` · 12-ML基础 · Causal ML Feature Engineering — 因果驱动的特征工程：消除混淆提升模型可靠性
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-ml-feature-engineering/SKILL.md`
    - `p2s-causal-rl-decision-making` · 源卡号 `Skill-Causal-RL-Decision-Making` · 01-因果推断 · 因果强化学习 — 从相关驱动到因果驱动的决策优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-rl-decision-making/SKILL.md`
    - `p2s-combo-inventory-crisis-response` · 源卡号 `Skill-Combo-Inventory-Crisis-Response` · 16-智能体工程 · 库存危机响应 Combo Pattern — 断货/积压异常触发的 5 步自动响应链路
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-combo-inventory-crisis-response/SKILL.md`
    - `p2s-conformal-prediction-demand-uq` · 源卡号 `Skill-Conformal-Prediction-Demand-UQ` · 03-时间序列 · Conformal Prediction Demand UQ（需求预测不确定性量化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-conformal-prediction-demand-uq/SKILL.md`
    - `p2s-conformal-prediction-framework` · 源卡号 `Skill-Conformal-Prediction-Framework` · 12-ML基础 · Conformal Prediction — 无分布假设的预测区间保证框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-conformal-prediction-framework/SKILL.md`
    - `p2s-conformal-risk-assessment` · 源卡号 `Skill-Conformal-Risk-Assessment` · 01-因果推断 · Conformal Risk Assessment — 共形预测业务风险量化：覆盖率保证的区间估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-conformal-risk-assessment/SKILL.md`
    - `p2s-conformal-ts-intervals` · 源卡号 `Skill-Conformal-TS-Intervals` · 03-时间序列 · Conformal TS Intervals（时序 Conformal 预测区间）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-conformal-ts-intervals/SKILL.md`
    - `p2s-conformal-time-series-forecasting` · 源卡号 `Skill-Conformal-Time-Series-Forecasting` · 03-时间序列 · Conformal Time Series Forecasting — 共形时序预测：有覆盖保证的需求预测区间
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-conformal-time-series-forecasting/SKILL.md`
    - `p2s-contrastive-time-series-cold-start` · 源卡号 `Skill-Contrastive-Time-Series-Cold-Start` · 03-时间序列 · Contrastive Time Series Cold Start — 对比学习驱动的新品需求冷启动预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-contrastive-time-series-cold-start/SKILL.md`
    - `p2s-counterfactual-sc-scenario-sim` · 源卡号 `Skill-Counterfactual-SC-Scenario-Sim` · 24-标签工程 · 供应链反事实情景仿真 — 决策前的数字沙盘，支撑Palantir高风险Action验证
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-counterfactual-sc-scenario-sim/SKILL.md`
    - `p2s-cross-sku-demand-correlation-mining` · 源卡号 `Skill-Cross-SKU-Demand-Correlation-Mining` · 03-时间序列 · Cross-SKU Demand Correlation Mining — 跨 SKU 需求相关性挖掘组合补货优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-sku-demand-correlation-mining/SKILL.md`
    - `p2s-drl-inventory-optimization` · 源卡号 `Skill-DRL-Inventory-Optimization` · 04-供应链 · DRL Inventory Optimization — 深度强化学习库存优化：端到端自适应补货决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-drl-inventory-optimization/SKILL.md`
    - `p2s-demand-forecasting-booking-curve` · 源卡号 `Skill-Demand-Forecasting-Booking-Curve` · 17-价格优化 · Demand Forecasting via Booking Curve — 酒店预订曲线迁移到电商搜索量超前指标预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-demand-forecasting-booking-curve/SKILL.md`
    - `p2s-demand-quantile-forecast` · 源卡号 `Skill-Demand-Quantile-Forecast` · 03-时间序列 · Demand Quantile Forecast — 需求分位数预测：备货决策的置信区间框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-demand-quantile-forecast/SKILL.md`
    - `p2s-demand-supply-matching-gap-analysis` · 源卡号 `Skill-Demand-Supply-Matching-Gap-Analysis` · 04-供应链 · 供需缺口分析与优先级分配决策 — 供给不足时的SKU优先级量化与分配算法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-demand-supply-matching-gap-analysis/SKILL.md`
    - `p2s-dynamic-abc-stratification-adaptive-policy` · 源卡号 `Skill-Dynamic-ABC-Stratification-Adaptive-Policy` · 04-供应链 · 动态ABC分层与策略自适应 — 帕累托分类自动更新与差异化库存策略绑定
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-abc-stratification-adaptive-policy/SKILL.md`
    - `p2s-dynamic-lot-sizing-moq` · 源卡号 `Skill-Dynamic-Lot-Sizing-MOQ` · 04-供应链 · Efficient Algorithms for the Joint Replenishment Problem wit
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-lot-sizing-moq/SKILL.md`
    - `p2s-event-driven-demand-mas` · 源卡号 `Skill-Event-Driven-Demand-MAS` · 10-MAS · Event-Driven Demand MAS — 事件感知补货 MAS：大促/季节自动触发
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-event-driven-demand-mas/SKILL.md`
    - `p2s-eventcast-llm-event-forecasting` · 源卡号 `Skill-EventCast-LLM-Event-Forecasting` · 03-时间序列 · EventCast — LLM 事件感知需求预测：大促/节假日场景 MAE-57%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-eventcast-llm-event-forecasting/SKILL.md`
    - `p2s-fba-cost-forecast-adjustment` · 源卡号 `Skill-FBA-Cost-Forecast-Adjustment` · 23-运营财务 · FBA Cost Forecast Adjustment — 不对称惩罚驱动的履约成本最小化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fba-cost-forecast-adjustment/SKILL.md`
    - `p2s-fdc-rdc-inventory-allocation` · 源卡号 `Skill-FDC-RDC-Inventory-Allocation` · 04-供应链 · FDC/RDC Inventory Allocation — 前置仓选品与库存分配端到端学习
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fdc-rdc-inventory-allocation/SKILL.md`
    - `p2s-fsda-drl` · 源卡号 `Skill-FSDA-DRL` · 04-供应链 · FSDA-DRL 快慢双智能体动态定价与补货联合优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fsda-drl/SKILL.md`
    - `p2s-fill-rate-oos-cost-quantification` · 源卡号 `Skill-Fill-Rate-OOS-Cost-Quantification` · 04-供应链 · 需求满足率与缺货成本全量化 — Fill Rate三层模型与OOS全链路损失计算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fill-rate-oos-cost-quantification/SKILL.md`
    - `p2s-flexible-supply-chain-small-batch-agile` · 源卡号 `Skill-Flexible-Supply-Chain-Small-Batch-Agile` · 04-供应链 · 柔性供应链小单快返 — SHEIN模式敏捷采购与快速响应算法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-flexible-supply-chain-small-batch-agile/SKILL.md`
    - `p2s-flowr-supply-chain-mas` · 源卡号 `Skill-Flowr-Supply-Chain-MAS` · 10-MAS · Flowr — 零售供应链多 Agent 端到端自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-flowr-supply-chain-mas/SKILL.md`
    - `p2s-forecast-driven-inventory` · 源卡号 `Skill-Forecast-Driven-Inventory` · 03-时间序列 · Forecast-Driven Inventory（预测驱动库存优化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-forecast-driven-inventory/SKILL.md`
    - `p2s-gcf-counterfactual-unobserved-demand` · 源卡号 `Skill-GCF-Counterfactual-Unobserved-Demand` · 24-标签工程 · 图因果预测GCF — 时空GNN+Synthetic Control估计Listing删除的隐性需求
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gcf-counterfactual-unobserved-demand/SKILL.md`
    - `p2s-gp-new-product-demand` · 源卡号 `Skill-GP-New-Product-Demand` · 03-时间序列 · GP New Product Demand — 高斯过程新品冷启动需求预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gp-new-product-demand/SKILL.md`
    - `p2s-holiday-spike-demand-decomposition` · 源卡号 `Skill-Holiday-Spike-Demand-Decomposition` · 03-时间序列 · Holiday Spike Demand Decomposition — 节假日需求峰值分解（Prime Day/黑五）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-holiday-spike-demand-decomposition/SKILL.md`
    - `p2s-ito-doi-inventory-turnover-optimizer` · 源卡号 `Skill-ITO-DOI-Inventory-Turnover-Optimizer` · 04-供应链 · ITO/DOI库存周转率优化闭环 — 库存效率KPI驱动的补货与清仓决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ito-doi-inventory-turnover-optimizer/SKILL.md`
    - `p2s-ito-three-phase-health-tracking` · 源卡号 `Skill-ITO-Three-Phase-Health-Tracking` · 04-供应链 · ITO备货前中后三阶段健康度追踪 — 库存周转全周期过程KPI与干预决策闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ito-three-phase-health-tracking/SKILL.md`
    - `p2s-intermittent-demand-croston-tsb` · 源卡号 `Skill-Intermittent-Demand-Croston-TSB` · 03-时间序列 · Intermittent Demand Croston TSB — 母婴长尾 SKU 间歇需求预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-intermittent-demand-croston-tsb/SKILL.md`
    - `p2s-inventory-demand-sensing` · 源卡号 `Skill-Inventory-Demand-Sensing` · 18-物流履约 · Inventory Demand Sensing — 库存需求感知：实时信号融合驱动智能补货
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-demand-sensing/SKILL.md`
    - `p2s-inventory-turnover-abc-classification` · 源卡号 `Skill-Inventory-Turnover-ABC-Classification` · 04-供应链 · ABC动销率动态分层与差异化策略 — ABCDE五级动销管理与80/20库存结构优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-turnover-abc-classification/SKILL.md`
    - `p2s-llm-sc-multiagent-consensus-replenishment` · 源卡号 `Skill-LLM-SC-MultiAgent-Consensus-Replenishment` · 24-标签工程 · LLM多智能体共识补货决策 — InvAgent框架：需求/采购/仓储三方博弈自动达成最优
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-sc-multiagent-consensus-replenishment/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 3 张
    - `Skill-Decision-Conditioned-Forecasting` · 03-时间序列 · Skill-Decision-Conditioned-Forecasting
      - 全文卡：`paper2skills-vault/03-时间序列/Skill-Decision-Conditioned-Forecasting.md`
    - `Skill-Multi-Warehouse-Allocation-LLM` · 04-供应链 · Skill-Multi-Warehouse-Allocation-LLM
      - 全文卡：`paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md`
    - `Skill-Supply-Network-Simulation` · 04-供应链 · Skill-Supply-Network-Simulation
      - 全文卡：`paper2skills-vault/04-供应链/Skill-Supply-Network-Simulation.md`

> 候选总数 90。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-029 · 物流方案

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-019` 跨境物流与关务 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-03, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-029-物流方案.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：运输方式/路线选择与海运成本估计是网络优化与成本建模问题，可输出方案与成本分布。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


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
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 47 张
    - `p2s-3d-bin-packing-optimization` · 源卡号 `Skill-3D-Bin-Packing-Optimization` · 18-物流履约 · 3D Bin Packing Optimization — 3D 装箱优化：集装箱/货架空间利用率最大化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-3d-bin-packing-optimization/SKILL.md`
    - `p2s-ab-test-logistics-sla` · 源卡号 `Skill-AB-Test-Logistics-SLA` · 02-A_B实验 · 物流 SLA A/B 实验因果效应评估 — 承诺变化对复购率的因果识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ab-test-logistics-sla/SKILL.md`
    - `p2s-alphafold-bin-packing` · 源卡号 `Skill-AlphaFold-Bin-Packing` · 04-供应链 · 蛋白质折叠启发的异形 SKU 极限装箱 (AlphaFold Bin-Packing)
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-alphafold-bin-packing/SKILL.md`
    - `p2s-carrier-selection-ml` · 源卡号 `Skill-Carrier-Selection-ML` · 18-物流履约 · 承运商智能选择 — ML驱动的跨境配送商优化决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-carrier-selection-ml/SKILL.md`
    - `p2s-cold-chain-temperature-monitoring` · 源卡号 `Skill-Cold-Chain-Temperature-Monitoring` · 18-物流履约 · 冷链温控全程监测 — 母婴辅食物流的温度合规智能预警
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cold-chain-temperature-monitoring/SKILL.md`
    - `p2s-cross-border-last-mile-routing` · 源卡号 `Skill-Cross-Border-Last-Mile-Routing` · 18-物流履约 · Cross-Border Last Mile Routing — 跨境最后一公里路由优化：时效×成本双目标决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-last-mile-routing/SKILL.md`
    - `p2s-cross-border-logistics-routing` · 源卡号 `Skill-Cross-Border-Logistics-Routing` · 18-物流履约 · Cross-Border Logistics Routing（跨境物流路径优化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-logistics-routing/SKILL.md`
    - `p2s-crossborder-logistics-mode-selection` · 源卡号 `Skill-CrossBorder-Logistics-Mode-Selection` · 18-物流履约 · 跨境物流模式动态选择 — 需求预测驱动的保税仓与直邮模式联合优化框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-crossborder-logistics-mode-selection/SKILL.md`
    - `p2s-crowdsourced-last-mile-ai-dispatch` · 源卡号 `Skill-Crowdsourced-Last-Mile-AI-Dispatch` · 18-物流履约 · 众包最后一公里AI调度 — 动态定价与配送网络优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-crowdsourced-last-mile-ai-dispatch/SKILL.md`
    - `p2s-dangerous-goods-dg-classification` · 源卡号 `Skill-Dangerous-Goods-DG-Classification` · 21-合规决策 · Dangerous Goods Classification — 危险品自动分类（锂电池/液体/气溶胶跨境合规）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dangerous-goods-dg-classification/SKILL.md`
    - `p2s-delivery-promise-optimization` · 源卡号 `Skill-Delivery-Promise-Optimization` · 18-物流履约 · Delivery Promise Optimization — 时效承诺优化：转化率与准时率的帕累托
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-delivery-promise-optimization/SKILL.md`
    - `p2s-drone-uav-last-mile-delivery` · 源卡号 `Skill-Drone-UAV-Last-Mile-Delivery` · 18-物流履约 · 无人机末端配送调度 — UAV最后一公里的路径规划
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-drone-uav-last-mile-delivery/SKILL.md`
    - `p2s-dynamic-carrier-selection-tag-driven` · 源卡号 `Skill-Dynamic-Carrier-Selection-Tag-Driven` · 24-标签工程 · Tag驱动动态承运商选择引擎 — 基于实时标签的末程承运商智能匹配与成本优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-carrier-selection-tag-driven/SKILL.md`
    - `p2s-first-last-mile-cost-kpi-crossborder` · 源卡号 `Skill-First-Last-Mile-Cost-KPI-CrossBorder` · 04-供应链 · 跨境头程末程成本KPI与路线优化 — 头程运费率/末程成本率/跨境物流综合成本体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-first-last-mile-cost-kpi-crossborder/SKILL.md`
    - `p2s-first-mile-pickup-optimization` · 源卡号 `Skill-First-Mile-Pickup-Optimization` · 18-物流履约 · 首公里取货路径优化 — 国内工厂到货代的调度算法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-first-mile-pickup-optimization/SKILL.md`
    - `p2s-first-mile-pickup-route-optimization` · 源卡号 `Skill-First-Mile-Pickup-Route-Optimization` · 18-物流履约 · First Mile Pickup Route Optimization — 首公里取货路径优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-first-mile-pickup-route-optimization/SKILL.md`
    - `p2s-geopolitical-risk-tag-supply-impact` · 源卡号 `Skill-Geopolitical-Risk-Tag-Supply-Impact` · 24-标签工程 · 地缘政治风险供应链影响标签 — 贸易限制/区域冲突对供应链的实时风险量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-geopolitical-risk-tag-supply-impact/SKILL.md`
    - `p2s-green-logistics-carbon-optimization` · 源卡号 `Skill-Green-Logistics-Carbon-Optimization` · 18-物流履约 · 碳最优物流路径规划 — ESG合规与成本的多目标优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-green-logistics-carbon-optimization/SKILL.md`
    - `p2s-kg-logistics-intelligence` · 源卡号 `Skill-KG-Logistics-Intelligence` · 08-知识图谱 · 知识图谱物流智能 — 供应链实体关系图驱动的物流决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-logistics-intelligence/SKILL.md`
    - `p2s-last-mile-cost-per-zone-analytics` · 源卡号 `Skill-Last-Mile-Cost-Per-Zone-Analytics` · 24-标签工程 · 末程分区成本精算 — 农村/偏远/标准区域差异化成本分解与路线优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-last-mile-cost-per-zone-analytics/SKILL.md`
    - `p2s-last-mile-delivery-prediction` · 源卡号 `Skill-Last-Mile-Delivery-Prediction` · 18-物流履约 · Last-Mile Delivery Prediction（最后一公里配送时效预测）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-last-mile-delivery-prediction/SKILL.md`
    - `p2s-last-mile-network-planning` · 源卡号 `Skill-Last-Mile-Network-Planning` · 18-物流履约 · Last Mile Network Planning — VRP变体+海外仓选址末端网络优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-last-mile-network-planning/SKILL.md`
    - `p2s-local-order-fulfillment-rate-fdc` · 源卡号 `Skill-Local-Order-Fulfillment-Rate-FDC` · 04-供应链 · 本地订单达成率与FDC仓网覆盖KPI — 本地发货率/跨仓调拨成本/仓网优化决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-local-order-fulfillment-rate-fdc/SKILL.md`
    - `p2s-logistics-carbon-scope3-tracker` · 源卡号 `Skill-Logistics-Carbon-Scope3-Tracker` · 18-物流履约 · 物流碳排放Scope3追踪 — 全链路碳足迹核算引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-carbon-scope3-tracker/SKILL.md`
    - `p2s-logistics-cost-lifecycle-kpi` · 源卡号 `Skill-Logistics-Cost-Lifecycle-KPI` · 04-供应链 · 物流成本前中后生命周期管理KPI — 生意前模拟/生意中账单/生意后分析的三段成本闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-cost-lifecycle-kpi/SKILL.md`
    - `p2s-logistics-cost-model` · 源卡号 `Skill-Logistics-Cost-Model` · 23-运营财务 · Logistics Cost Model — 跨境物流全链路成本建模与关税不确定性优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-cost-model/SKILL.md`
    - `p2s-logistics-cost-structure-decomposition` · 源卡号 `Skill-Logistics-Cost-Structure-Decomposition` · 04-供应链 · 全链路物流成本结构分解 — 进存销三段成本拆解与降本杠杆识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-cost-structure-decomposition/SKILL.md`
    - `p2s-logistics-sla-causal-impact` · 源卡号 `Skill-Logistics-SLA-Causal-Impact` · 18-物流履约 · 物流SLA变更的因果影响 — 时效承诺对复购率的DiD分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-sla-causal-impact/SKILL.md`
    - `p2s-multi-temperature-logistics` · 源卡号 `Skill-Multi-Temperature-Logistics` · 18-物流履约 · Multi-Temperature Logistics — 多温区混合配送成本优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-temperature-logistics/SKILL.md`
    - `p2s-multilevel-flp` · 源卡号 `Skill-Multilevel_FLP` · 04-供应链 · Multilevel Facility Location Optimization (多级设施选址优化)
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilevel-flp/SKILL.md`
    - `p2s-neo-lrp` · 源卡号 `Skill-NEO_LRP` · 04-供应链 · NEO-LRP（Neural Embedded Optimization for Location-Routing）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-neo-lrp/SKILL.md`
    - `p2s-oos-emergency-airfreight-gate` · 源卡号 `Skill-OOS-Emergency-Airfreight-Gate` · 04-供应链 · OOS-Emergency-Airfreight-Gate — 库存DOS危急+海运延误自动触发紧急空运决策门控
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-oos-emergency-airfreight-gate/SKILL.md`
    - `p2s-order-routing-intelligence-engine` · 源卡号 `Skill-Order-Routing-Intelligence-Engine` · 24-标签工程 · 智能订单路由引擎 — 多约束订单履约路径优化与实时分配决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-order-routing-intelligence-engine/SKILL.md`
    - `p2s-order-splitting-merging-optimizer` · 源卡号 `Skill-Order-Splitting-Merging-Optimizer` · 04-供应链 · 订单拆合单优化器 — 多仓多渠道场景下拆单合单的成本-时效平衡决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-order-splitting-merging-optimizer/SKILL.md`
    - `p2s-ppo-swap` · 源卡号 `Skill-PPO_swap` · 04-供应链 · PPO-swap（图上设施选址强化学习）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ppo-swap/SKILL.md`
    - `p2s-port-congestion-eta-prediction` · 源卡号 `Skill-Port-Congestion-ETA-Prediction` · 18-物流履约 · 港口拥堵ETA预测 — 多因子动态到港时间估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-port-congestion-eta-prediction/SKILL.md`
    - `p2s-predictive-batch-returns-routing` · 源卡号 `Skill-Predictive-Batch-Returns-Routing` · 18-物流履约 · 退货批量逆向路由预测 — 跨境退货智能分拣与逆向物流优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-predictive-batch-returns-routing/SKILL.md`
    - `p2s-real-time-fleet-dynamic-routing` · 源卡号 `Skill-Real-Time-Fleet-Dynamic-Routing` · 18-物流履约 · 实时车队动态重路由 — 突发事件下的物流路径自适应优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-real-time-fleet-dynamic-routing/SKILL.md`
    - `p2s-spot-freight-consolidation` · 源卡号 `Skill-SPOT-Freight-Consolidation` · 04-供应链 · SPOT Freight Consolidation — 时空模式挖掘的货运拼箱优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-spot-freight-consolidation/SKILL.md`
    - `p2s-search-driven-logistics-promise` · 源卡号 `Skill-Search-Driven-Logistics-Promise` · 25-搜索流量工程 · Search-Driven Logistics Promise — 搜索词意图信号驱动履约时效承诺优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-driven-logistics-promise/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 2 张
    - `Skill-Shipping-Cost-Estimation` · 03-时间序列 · Skill-Shipping-Cost-Estimation
      - 全文卡：`paper2skills-vault/03-时间序列/Skill-Shipping-Cost-Estimation.md`
    - `Skill-Supply-Network-Simulation` · 04-供应链 · Skill-Supply-Network-Simulation
      - 全文卡：`paper2skills-vault/04-供应链/Skill-Supply-Network-Simulation.md`

> 候选总数 49。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-042 · 价格敏感性

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-026` 定价促销与商品组合 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-042-价格敏感性.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：价格弹性可用需求模型与因果实验（价格A/B、DML）直接估参。
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

- 有全文卡（优先选，可选到论文参数）：共 3 张
    - `p2s-iv-instrumental-variables` · 源卡号 `Skill-IV-Instrumental-Variables` · 01-因果推断 · Instrumental Variables (IV) for Causal Inference with Endoge
      - 全文卡：`/Users/lute/.dsh/skills/p2s-iv-instrumental-variables/SKILL.md`
    - `p2s-monodense` · 源卡号 `Skill-Monodense-单品价格弹性估计` · 04-供应链 · Skill: Monodense 单品价格弹性估计
      - 全文卡：`/Users/lute/.dsh/skills/p2s-monodense/SKILL.md`
    - `p2s-promotion-effectiveness` · 源卡号 `Skill-Promotion-Effectiveness` · 15-营销投放分析 · Promotion Effectiveness Evaluation with Causal ML
      - 全文卡：`/Users/lute/.dsh/skills/p2s-promotion-effectiveness/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 50 张
    - `p2s-aigp-llm-dynamic-pricing` · 源卡号 `Skill-AIGP-LLM-Dynamic-Pricing` · 17-价格优化 · AIGP — LLM 动态定价：长期 GMV 对齐框架（+13% GMV A/B实测）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-aigp-llm-dynamic-pricing/SKILL.md`
    - `p2s-anchoring-effect-pricing-optimization` · 源卡号 `Skill-Anchoring-Effect-Pricing-Optimization` · 17-价格优化 · 锚定效应定价优化 — 划线原价最优锚定比率让相同折扣感知价值提升25%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-anchoring-effect-pricing-optimization/SKILL.md`
    - `p2s-bundle-pricing-strategy` · 源卡号 `Skill-Bundle-Pricing-Strategy` · 17-价格优化 · Bundle Pricing Strategy（捆绑定价策略）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bundle-pricing-strategy/SKILL.md`
    - `p2s-causal-rl-dynamic-pricing` · 源卡号 `Skill-Causal-RL-Dynamic-Pricing` · 17-价格优化 · Causal RL Dynamic Pricing — 因果强化学习动态定价：可信赖的自适应价格策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-rl-dynamic-pricing/SKILL.md`
    - `p2s-competitive-price-monitoring` · 源卡号 `Skill-Competitive-Price-Monitoring` · 17-价格优化 · Competitive Price Monitoring（竞品价格监测与响应）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-competitive-price-monitoring/SKILL.md`
    - `p2s-competitor-price-intelligence` · 源卡号 `Skill-Competitor-Price-Intelligence` · 17-价格优化 · Competitor Price Intelligence — 竞品价格实时监控与智能响应
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-competitor-price-intelligence/SKILL.md`
    - `p2s-compliant-dynamic-pricing-guard` · 源卡号 `Skill-Compliant-Dynamic-Pricing-Guard` · 17-价格优化 · Compliant Dynamic Pricing Guard（合规-定价双约束优化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-compliant-dynamic-pricing-guard/SKILL.md`
    - `p2s-contextual-dynamic-pricing-optimal` · 源卡号 `Skill-Contextual-Dynamic-Pricing-Optimal` · 17-价格优化 · Contextual Dynamic Pricing — 最优上下文定价：O(√dT) Regret + LDP 隐私保
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-contextual-dynamic-pricing-optimal/SKILL.md`
    - `p2s-cost-plus-dynamic-tariff-pricing` · 源卡号 `Skill-Cost-Plus-Dynamic-Tariff-Pricing` · 17-价格优化 · 成本加成+关税动态定价 — 关税波动下的自动调价模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cost-plus-dynamic-tariff-pricing/SKILL.md`
    - `p2s-counterfactual-price-elasticity` · 源卡号 `Skill-Counterfactual-Price-Elasticity` · 17-价格优化 · 反事实动态价格弹性测算 (Counterfactual Price Elasticity via DML)
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-counterfactual-price-elasticity/SKILL.md`
    - `p2s-cross-border-price-harmonization` · 源卡号 `Skill-Cross-Border-Price-Harmonization` · 17-价格优化 · Cross-Border Price Harmonization（跨境价格协调）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-price-harmonization/SKILL.md`
    - `p2s-double-debiased-ml-price` · 源卡号 `Skill-Double-Debiased-ML-Price` · 01-因果推断 · 双重去偏机器学习价格弹性 — 高维混淆下的价格因果效应估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-double-debiased-ml-price/SKILL.md`
    - `p2s-dynamic-bundle-pricing` · 源卡号 `Skill-Dynamic-Bundle-Pricing` · 17-价格优化 · Dynamic Bundle Pricing — 动态捆绑定价：配套商品组合最优定价策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-bundle-pricing/SKILL.md`
    - `p2s-dynamic-pricing-elasticity` · 源卡号 `Skill-Dynamic-Pricing-Elasticity` · 17-价格优化 · Dynamic Pricing with Demand Elasticity（动态定价与需求弹性）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-pricing-elasticity/SKILL.md`
    - `p2s-emsr-bid-price-inventory-control` · 源卡号 `Skill-EMSR-Bid-Price-Inventory-Control` · 17-价格优化 · EMSR-b Bid-Price Inventory Control — 酒店边际座位收益模型迁移到FBA库存动态定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-emsr-bid-price-inventory-control/SKILL.md`
    - `p2s-elasticity-based-repricing-gate` · 源卡号 `Skill-Elasticity-Based-Repricing-Gate` · 17-价格优化 · Elasticity-Based Repricing Gate — 弹性阈值自动触发涨价/降价A/B测试
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-elasticity-based-repricing-gate/SKILL.md`
    - `p2s-fsda-drl` · 源卡号 `Skill-FSDA-DRL` · 04-供应链 · FSDA-DRL 快慢双智能体动态定价与补货联合优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fsda-drl/SKILL.md`
    - `p2s-fx-dynamic-pricing-adjustment` · 源卡号 `Skill-FX-Dynamic-Pricing-Adjustment` · 23-运营财务 · 汇率联动动态定价 — 保持目标毛利率的实时定价调整
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fx-dynamic-pricing-adjustment/SKILL.md`
    - `p2s-flash-sale-price-optimization` · 源卡号 `Skill-Flash-Sale-Price-Optimization` · 17-价格优化 · 闪购定价优化 — 限时折扣期间最优折扣率与时间窗口计算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-flash-sale-price-optimization/SKILL.md`
    - `p2s-llm-negotiation-conversion-agent` · 源卡号 `Skill-LLM-Negotiation-Conversion-Agent` · 16-智能体工程 · LLM Negotiation Conversion Agent — LLM 谈判代理驱动的成交率优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-negotiation-conversion-agent/SKILL.md`
    - `p2s-latent-class-demand-segmentation` · 源卡号 `Skill-Latent-Class-Demand-Segmentation` · 14-用户分析 · 潜在类别需求分群 — EM算法自动发现购买决策者类型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-latent-class-demand-segmentation/SKILL.md`
    - `p2s-mappo-gat-dynamic-pricing` · 源卡号 `Skill-MAPPO-GAT-Dynamic-Pricing` · 17-价格优化 · MAPPO+GAT多智能体图注意力动态定价 — 产品关系图驱动的多SKU协同价格优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mappo-gat-dynamic-pricing/SKILL.md`
    - `p2s-mas-dynamic-pricing-coalition` · 源卡号 `Skill-MAS-Dynamic-Pricing-Coalition` · 10-MAS · MAS多SKU定价联盟博弈 — 母婴品牌多SKU组合利润最大化联合定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-dynamic-pricing-coalition/SKILL.md`
    - `p2s-mas-pricing-coalition-stability` · 源卡号 `Skill-MAS-Pricing-Coalition-Stability` · 10-MAS · MAS-Pricing-Coalition-Stability — 多SKU联合定价纳什均衡检测与联合体稳定性维持
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-pricing-coalition-stability/SKILL.md`
    - `p2s-mental-accounting-bundle-psychology` · 源卡号 `Skill-Mental-Accounting-Bundle-Psychology` · 17-价格优化 · 心理账户捆绑定价心理学 — 识别同一心智账户商品组合使 AOV 提升22%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mental-accounting-bundle-psychology/SKILL.md`
    - `p2s-mixed-strategy-pricing-unpredictability` · 源卡号 `Skill-Mixed-Strategy-Pricing-Unpredictability` · 17-价格优化 · 混合策略定价不可预测性 — 随机化定价规避竞品跟价算法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mixed-strategy-pricing-unpredictability/SKILL.md`
    - `p2s-model-compression-edge-deployment` · 源卡号 `Skill-Model-Compression-Edge-Deployment` · 12-ML基础 · 模型压缩与边缘部署 — INT8 量化 + 结构化剪枝 + ONNX 导出
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-model-compression-edge-deployment/SKILL.md`
    - `p2s-multi-channel-price-consistency` · 源卡号 `Skill-Multi-Channel-Price-Consistency` · 17-价格优化 · Multi-Channel Price Consistency — 多渠道价格一致性管理（防跨平台价格冲突）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-channel-price-consistency/SKILL.md`
    - `p2s-nash-equilibrium-pricing-model` · 源卡号 `Skill-Nash-Equilibrium-Pricing-Model` · 17-价格优化 · 纳什均衡定价模型 — 多卖家竞争价格博弈均衡求解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-nash-equilibrium-pricing-model/SKILL.md`
    - `p2s-personalized-ml-pricing` · 源卡号 `Skill-Personalized-ML-Pricing` · 17-价格优化 · Personalized ML Pricing — 个性化 ML 定价：用户级支付意愿驱动的差异化定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-personalized-ml-pricing/SKILL.md`
    - `p2s-price-elasticity-estimation` · 源卡号 `Skill-Price-Elasticity-Estimation` · 17-价格优化 · Price Elasticity Estimation — 需求价格弹性估算：跨境 SKU 定价底线测算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-elasticity-estimation/SKILL.md`
    - `p2s-price-elasticity-time-series-fusion` · 源卡号 `Skill-Price-Elasticity-Time-Series-Fusion` · 03-时间序列 · Price Elasticity Time Series Fusion — 价格弹性×时间序列融合预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-elasticity-time-series-fusion/SKILL.md`
    - `p2s-price-fence-segmentation-ecommerce` · 源卡号 `Skill-Price-Fence-Segmentation-Ecommerce` · 17-价格优化 · Price Fence Segmentation — 航空分舱定价策略迁移到母婴电商三级价格歧视
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-fence-segmentation-ecommerce/SKILL.md`
    - `p2s-price-scraping-defense` · 源卡号 `Skill-Price-Scraping-Defense` · 17-价格优化 · Price Scraping Defense — 价格爬取防御（防竞品监控+反监测策略）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-scraping-defense/SKILL.md`
    - `p2s-price-sensitive-personalized-recommendation` · 源卡号 `Skill-Price-Sensitive-Personalized-Recommendation` · 17-价格优化 · Price-Sensitive Personalized Recommendation — 价格感知个性化推荐：弹性×用
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-sensitive-personalized-recommendation/SKILL.md`
    - `p2s-price-sensitive-recommendation` · 源卡号 `Skill-Price-Sensitive-Recommendation` · 05-推荐系统 · Price-Sensitive Recommendation — 价格感知推荐：弹性感知的个性化定价与排序融合
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-sensitive-recommendation/SKILL.md`
    - `p2s-pricing-causal-identification` · 源卡号 `Skill-Pricing-Causal-Identification` · 17-价格优化 · Pricing Causal Identification — 工具变量 IV 识别真实价格弹性（去除内生性偏差）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-pricing-causal-identification/SKILL.md`
    - `p2s-psychological-pricing-ab-test` · 源卡号 `Skill-Psychological-Pricing-AB-Test` · 17-价格优化 · 心理定价A/B测试 — $9.99 vs $10 效果量化与最优价格尾数选择
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-psychological-pricing-ab-test/SKILL.md`
    - `p2s-rl-dynamic-promotion-optimization` · 源卡号 `Skill-RL-Dynamic-Promotion-Optimization` · 15-营销投放分析 · RL Dynamic Promotion Optimization — 强化学习动态促销优化：时机×力度×对象的联合决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-rl-dynamic-promotion-optimization/SKILL.md`
    - `p2s-real-time-competitive-repricing` · 源卡号 `Skill-Real-Time-Competitive-Repricing` · 17-价格优化 · Real-Time Competitive Repricing — 竞品价格监测与深度强化学习自动重定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-real-time-competitive-repricing/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 2 张
    - `Skill-MAS-MARL-Dynamic-Pricing` · 07-NLP-VOC · Skill-MAS-MARL-Dynamic-Pricing
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-MARL-Dynamic-Pricing.md`
    - `Skill-TJAP-跨市场品类组合定价` · 07-NLP-VOC · Skill-TJAP-跨市场品类组合定价
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-TJAP-跨市场品类组合定价.md`

> 候选总数 55。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-061 · 资金预测

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-041` 经营财务与资金 |
| 域 / 面 | `DOM-07` / `PLN-MGT` |
| flows | FLOW-01, FLOW-02, FLOW-03, FLOW-07, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-061-资金预测.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：现金流预测是标准时序预测加场景模拟问题（含汇率与账期）。
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
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 19 张
    - `p2s-accounts-receivable-intelligence` · 源卡号 `Skill-Accounts-Receivable-Intelligence` · 23-运营财务 · Accounts Receivable Intelligence — 账期智能管理：跨境应收账款预测与催收优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-accounts-receivable-intelligence/SKILL.md`
    - `p2s-amazon-lending-decision` · 源卡号 `Skill-Amazon-Lending-Decision` · 23-运营财务 · Amazon Lending Decision — 电商平台卖家信用评估与融资决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-amazon-lending-decision/SKILL.md`
    - `p2s-amazon-payment-cycle-forecast` · 源卡号 `Skill-Amazon-Payment-Cycle-Forecast` · 23-运营财务 · Amazon Payment Cycle Forecast — Amazon 回款周期预测与现金流规划
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-amazon-payment-cycle-forecast/SKILL.md`
    - `p2s-budget-reforecast-rolling` · 源卡号 `Skill-Budget-Reforecast-Rolling` · 23-运营财务 · Budget Reforecast Rolling — 滚动预算再预测（月度滚动reforecast自动化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-budget-reforecast-rolling/SKILL.md`
    - `p2s-cash-conversion-cycle-optimization` · 源卡号 `Skill-Cash-Conversion-Cycle-Optimization` · 23-运营财务 · Skill-Cash-Conversion-Cycle-Optimization — 现金转换周期优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cash-conversion-cycle-optimization/SKILL.md`
    - `p2s-cross-border-cash-flow-forecasting` · 源卡号 `Skill-Cross-Border-Cash-Flow-Forecasting` · 04-供应链 · Cross-Border Cash Flow Forecasting（跨境电商现金流预测与融资窗口规划）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-cash-flow-forecasting/SKILL.md`
    - `p2s-dynamic-payment-terms-tag-engine` · 源卡号 `Skill-Dynamic-Payment-Terms-Tag-Engine` · 04-供应链 · 动态账期标签引擎 — 基于现金流预测的供应商账期智能优化与动态调整
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-payment-terms-tag-engine/SKILL.md`
    - `p2s-fx-exposure-measurement` · 源卡号 `Skill-FX-Exposure-Measurement` · 23-运营财务 · 外汇敞口测量 — 跨境电商货币风险定量分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fx-exposure-measurement/SKILL.md`
    - `p2s-fx-hedging-strategy` · 源卡号 `Skill-FX-Hedging-Strategy` · 23-运营财务 · FX Hedging Strategy — 跨境汇率风险对冲：动态套期保值降低外汇损失
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fx-hedging-strategy/SKILL.md`
    - `p2s-fx-natural-hedging-strategy` · 源卡号 `Skill-FX-Natural-Hedging-Strategy` · 23-运营财务 · 自然对冲策略 — 跨境电商外汇敞口零成本对冲
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fx-natural-hedging-strategy/SKILL.md`
    - `p2s-inventory-financing-optimization` · 源卡号 `Skill-Inventory-Financing-Optimization` · 23-运营财务 · Inventory Financing Optimization — 库存融资与供应链金融决策优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-financing-optimization/SKILL.md`
    - `p2s-moq-payment-terms-optimization` · 源卡号 `Skill-MOQ-Payment-Terms-Optimization` · 04-供应链 · MOQ与账期联动优化决策 — 最小起订量与付款条件的现金流效益模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-moq-payment-terms-optimization/SKILL.md`
    - `p2s-multicurrency-fx-hedging` · 源卡号 `Skill-Multicurrency-FX-Hedging` · 23-运营财务 · Multicurrency FX Hedging — 跨境卖家多货币外汇风险对冲
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multicurrency-fx-hedging/SKILL.md`
    - `p2s-operating-cash-flow-forecast` · 源卡号 `Skill-Operating-Cash-Flow-Forecast` · 23-运营财务 · Operating Cash Flow Forecast — 需求预测驱动的运营现金流预测与库存融资优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-operating-cash-flow-forecast/SKILL.md`
    - `p2s-supply-chain-finance-risk-modeling` · 源卡号 `Skill-Supply-Chain-Finance-Risk-Modeling` · 23-运营财务 · Supply Chain Finance Risk Modeling — 供应链金融风险建模：跨境贸易融资信用评估
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-finance-risk-modeling/SKILL.md`
    - `p2s-supply-chain-working-capital-optimization` · 源卡号 `Skill-Supply-Chain-Working-Capital-Optimization` · 04-供应链 · 供应链营运资金优化与CCC现金转换周期 — DIO/DSO/DPO三角分析与现金效率提升
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-working-capital-optimization/SKILL.md`
    - `p2s-user-ltv-financial-bridge` · 源卡号 `Skill-User-LTV-Financial-Bridge` · 14-用户分析 · User LTV Financial Bridge — 用户生命周期价值预测驱动财务规划
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-user-ltv-financial-bridge/SKILL.md`
    - `p2s-working-capital-cycle-optimizer` · 源卡号 `Skill-Working-Capital-Cycle-Optimizer` · 23-运营财务 · Skill-Working-Capital-Cycle-Optimizer — 营运资金周期优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-working-capital-cycle-optimizer/SKILL.md`
    - `p2s-working-capital-stress-test` · 源卡号 `Skill-Working-Capital-Stress-Test` · 23-运营财务 · Working Capital Stress Test — 营运资金压力测试（旺季备货资金缺口模拟）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-working-capital-stress-test/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 19。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-022 · 生产异常协调

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-018` 生产协同与质量控制 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-02, FLOW-03, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-022-生产异常协调.md` |
| `status` 初值 | **待卡**（无候选卡 —— 这就是扩充工单） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：异常可在线检测与分类，但处置须结合产线实况与供方配合。
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
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 0 张
    （无）
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 0。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。

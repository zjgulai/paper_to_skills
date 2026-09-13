# FLOW-07 契约撰写 · 批次 02（9 份）

> 公共材料见 `flow-07-common.md`（**先读它**）。本批 4 份 A 模板 / 5 份 B 模板。


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
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

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
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

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

## CTR-A-032 · 退货分流

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-020` 仓储履约与退货处置 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-03, FLOW-05, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-032-退货分流.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：退货处置路径可建模为成本最优决策（退回/翻新/弃置）并输出分流建议。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |
| FLOW-05/STG-01 | D | 信号接收 | 接收咨询、订单异常、售后或获许可的生命周期事件 | 客户旅程信号 |
| FLOW-05/STG-02 | R | 范围确定与Case创建 | 确定客户/订单最小范围、问题类型、许可状态和完成条件 | 客户Case Charter |
| FLOW-05/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配服务、体验、教育、CRM、隐私和质量能力 | FLOW-05 Context Manifest |
| FLOW-05/STG-06 | R | Assurance接收门禁 | 检查许可、隐私、健康风险、质量事件、补偿和触达边界 | FLOW-05 Assurance Decision |
| FLOW-05/STG-07 | D | 受控动作 | 提交退款/补救/触达Intent或NoActionRecord | 客户动作尝试或无动作记录 |
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 14 张
    - `p2s-cross-border-return-rate-by-country-kpi` · 源卡号 `Skill-Cross-Border-Return-Rate-By-Country-KPI` · 04-供应链 · 跨境分国退货率KPI与差异分析 — 美/德/英退货率差异根因与退货成本管控
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-return-rate-by-country-kpi/SKILL.md`
    - `p2s-parcel-damage-prediction` · 源卡号 `Skill-Parcel-Damage-Prediction` · 18-物流履约 · Parcel Damage Prediction — 包裹破损预测（物流轨迹+特征联合模型）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-parcel-damage-prediction/SKILL.md`
    - `p2s-predictive-batch-returns-routing` · 源卡号 `Skill-Predictive-Batch-Returns-Routing` · 18-物流履约 · 退货批量逆向路由预测 — 跨境退货智能分拣与逆向物流优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-predictive-batch-returns-routing/SKILL.md`
    - `p2s-predictive-returns-management` · 源卡号 `Skill-Predictive-Returns-Management` · 18-物流履约 · Predictive Returns Management — 退货量预测与主动处理：降低逆向物流成本
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-predictive-returns-management/SKILL.md`
    - `p2s-return-fraud-detection-tag-engine` · 源卡号 `Skill-Return-Fraud-Detection-Tag-Engine` · 24-标签工程 · 退货欺诈检测标签引擎 — 识别假退货/滥用退货政策的行为模式与Tag自动标记
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-return-fraud-detection-tag-engine/SKILL.md`
    - `p2s-return-fraud-detection` · 源卡号 `Skill-Return-Fraud-Detection` · 19-风控反欺诈 · Return Fraud Detection — 退货欺诈识别：GNN 检测虚假退货与促销滥用
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-return-fraud-detection/SKILL.md`
    - `p2s-return-rate-forecasting-model` · 源卡号 `Skill-Return-Rate-Forecasting-Model` · 03-时间序列 · Return Rate Forecasting Model — 退货率时序预测节后退货浪潮提前量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-return-rate-forecasting-model/SKILL.md`
    - `p2s-return-root-cause-attribution-graph` · 源卡号 `Skill-Return-Root-Cause-Attribution-Graph` · 24-标签工程 · 退货根因归因图谱 — 多层次退货原因知识图谱与供应链改善闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-return-root-cause-attribution-graph/SKILL.md`
    - `p2s-return-value-recovery-auction` · 源卡号 `Skill-Return-Value-Recovery-Auction` · 04-供应链 · 退货价值回收竞价 — 二手/翻新/零部件/捐赠的最优处置价格发现机制
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-return-value-recovery-auction/SKILL.md`
    - `p2s-returnformer-returns-prediction` · 源卡号 `Skill-Returnformer-Returns-Prediction` · 04-供应链 · Returnformer Returns Prediction — 图 Transformer 电商退货预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-returnformer-returns-prediction/SKILL.md`
    - `p2s-returns-quality-grading-engine` · 源卡号 `Skill-Returns-Quality-Grading-Engine` · 18-物流履约 · 退货品质分级引擎 — 计算机视觉驱动的再销售路径决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-returns-quality-grading-engine/SKILL.md`
    - `p2s-returns-reverse-logistics` · 源卡号 `Skill-Returns-Reverse-Logistics` · 18-物流履约 · Returns Reverse Logistics（退货逆向物流）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-returns-reverse-logistics/SKILL.md`
    - `p2s-reverse-logistics-disposition-optimization` · 源卡号 `Skill-Reverse-Logistics-Disposition-Optimization` · 04-供应链 · 逆向物流全链路处置决策 — 退货质检分流、二次销售与价值最大化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-reverse-logistics-disposition-optimization/SKILL.md`
    - `p2s-voc-returns-cost-driver` · 源卡号 `Skill-VOC-Returns-Cost-Driver` · 07-NLP-VOC · VOC Returns Cost Driver — 退货原因 NLP 分析：从评论挖掘退货成本驱动因子
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-voc-returns-cost-driver/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 14。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
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
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

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

## CTR-B-007 · 利益冲突检查

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-005` 内控审计与独立复核 |
| 域 / 面 | `DOM-01` / `PLN-CTL` |
| flows | FLOW-07, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-007-利益冲突检查.md` |
| `status` 初值 | **待卡**（无候选卡 —— 这就是扩充工单） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：关联方可用图挖掘发现，但冲突披露与回避决定属治理判断。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-08/STG-04 | M | 证据收集与诊断 | 核对经营结果、全成本、自治异常、新品验证和能力表现 | 经营与能力诊断 |
| FLOW-08/STG-05 | M | 方案与标准产物 | 形成资源调整、停止事项、策略或能力Change Proposal | 经营与能力变更建议 |
| FLOW-08/STG-08 | M | 结果核验、关闭与异步学习 | 核对资源/能力结果、观察窗和回退证据 | 经营复盘关闭记录 |
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
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 0 张
    （无）
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 0。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-021 · 产能调查

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-014` OEM供应商开发与协同 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-02, FLOW-03, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-021-产能调查.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：产能与节拍可建模（排队论/瓶颈分析），但输入依赖供方声明，需审核验证。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。
- **§F.5 边界条目**：排队论与瓶颈分析可量化产能与节拍 ⇒ 降级条件：输入若全依赖供方声明，则降 B

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
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 5 张
    - `p2s-agentic-sckg-risk` · 源卡号 `Skill-Agentic-SCKG-Risk` · 08-知识图谱 · Agentic SCKG Risk Analyzer — 供应链知识图谱智能风险分析框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agentic-sckg-risk/SKILL.md`
    - `p2s-capacity-constraint-production-schedule-kpi` · 源卡号 `Skill-Capacity-Constraint-Production-Schedule-KPI` · 04-供应链 · 产能约束生产排程KPI — 产能利用率/排程达成率/换线时间/瓶颈识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-capacity-constraint-production-schedule-kpi/SKILL.md`
    - `p2s-multi-factory-capacity-allocation` · 源卡号 `Skill-Multi-Factory-Capacity-Allocation` · 04-供应链 · 多工厂产能分配算法 — OEM/ODM多厂商协同下的弹性产能调度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-factory-capacity-allocation/SKILL.md`
    - `p2s-supplier-capacity-booking-engine` · 源卡号 `Skill-Supplier-Capacity-Booking-Engine` · 04-供应链 · 供应商产能预订引擎 — 旺季弹性产能锁定与长期产能保障协议管理
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-capacity-booking-engine/SKILL.md`
    - `p2s-supplier-capacity-planning` · 源卡号 `Skill-Supplier-Capacity-Planning` · 04-供应链 · Multi-objective multi-site supplier selection and order spli
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-capacity-planning/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 5。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-034 · 传播规划

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-029` 品牌战略与传播 |
| 域 / 面 | `DOM-05` / `PLN-OPS` |
| flows | FLOW-02, FLOW-04, FLOW-05, FLOW-07, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-034-传播规划.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：媒介组合可用MMM/频次模型优化，但品牌主张与调性须人定。
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
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-08/STG-04 | M | 证据收集与诊断 | 核对经营结果、全成本、自治异常、新品验证和能力表现 | 经营与能力诊断 |
| FLOW-08/STG-05 | M | 方案与标准产物 | 形成资源调整、停止事项、策略或能力Change Proposal | 经营与能力变更建议 |
| FLOW-08/STG-08 | M | 结果核验、关闭与异步学习 | 核对资源/能力结果、观察窗和回退证据 | 经营复盘关闭记录 |
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
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 7 张
    - `p2s-interference-networks-causal` · 源卡号 `Skill-Interference-Networks-Causal` · 01-因果推断 · 网络干扰因果推断 — 社交溢出效应与 Peer Effects 估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-interference-networks-causal/SKILL.md`
    - `p2s-share-of-voice-tracking` · 源卡号 `Skill-Share-of-Voice-Tracking` · 13-广告分析 · Share of Voice Tracking — AI 时代跨平台品牌可见度份额测量
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-share-of-voice-tracking/SKILL.md`
    - `p2s-social-network-viral-growth-simulation` · 源卡号 `Skill-Social-Network-Viral-Growth-Simulation` · 06-增长模型 · 社交网络病毒式增长模拟与放大 — 跨境品牌UGC传播建模与爆发点预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-social-network-viral-growth-simulation/SKILL.md`
    - `p2s-tiktok-algorithm-content-boost` · 源卡号 `Skill-TikTok-Algorithm-Content-Boost` · 20-AI视频生成 · TikTok Algorithm Content Boost — FYP 算法建模与内容传播速度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-algorithm-content-boost/SKILL.md`
    - `p2s-ugc-viral-content-potential-scorer` · 源卡号 `Skill-UGC-Viral-Content-Potential-Scorer` · 07-NLP-VOC · UGC病毒传播潜力评分 — 识别可引发病毒扩散的高潜力内容
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ugc-viral-content-potential-scorer/SKILL.md`
    - `p2s-viral-marketing-model` · 源卡号 `Skill-Viral-Marketing-Model` · 06-增长模型 · Viral Marketing Model — 病毒式传播建模：K 因子裂变增长的量化设计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-viral-marketing-model/SKILL.md`
    - `p2s-zero-click-search-optimization` · 源卡号 `Skill-Zero-Click-Search-Optimization` · 13-广告分析 · Zero-Click Search Optimization — 零点击搜索时代的流量保全与特色摘要抢占
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-zero-click-search-optimization/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 7。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-045 · 服务补救

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-037` 售后客诉与服务补救 |
| 域 / 面 | `DOM-06` / `PLN-OPS` |
| flows | FLOW-05, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-045-服务补救.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：补救效果可用增量测量，但方案受许可、健康风险与政策边界约束。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-05/STG-01 | D | 信号接收 | 接收咨询、订单异常、售后或获许可的生命周期事件 | 客户旅程信号 |
| FLOW-05/STG-02 | R | 范围确定与Case创建 | 确定客户/订单最小范围、问题类型、许可状态和完成条件 | 客户Case Charter |
| FLOW-05/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配服务、体验、教育、CRM、隐私和质量能力 | FLOW-05 Context Manifest |
| FLOW-05/STG-06 | R | Assurance接收门禁 | 检查许可、隐私、健康风险、质量事件、补偿和触达边界 | FLOW-05 Assurance Decision |
| FLOW-05/STG-07 | D | 受控动作 | 提交退款/补救/触达Intent或NoActionRecord | 客户动作尝试或无动作记录 |
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 3 张
    - `p2s-emotional-ai-customer-care` · 源卡号 `Skill-Emotional-AI-Customer-Care` · 11-AI人文 · Emotional AI Customer Care — 情感感知客服：高压场景的同理心 AI
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-emotional-ai-customer-care/SKILL.md`
    - `p2s-proactive-customer-alert-supply-chain` · 源卡号 `Skill-Proactive-Customer-Alert-Supply-Chain` · 24-标签工程 · 主动客户预警供应链 — 基于在途延误Tag的客户主动通知与体验保护
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-proactive-customer-alert-supply-chain/SKILL.md`
    - `p2s-scrable-review-response-generation` · 源卡号 `Skill-SCRABLE-Review-Response-Generation` · 14-用户分析 · SCRABLE Review Response Generation — RAG+LLM 自优化差评回复生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-scrable-review-response-generation/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 3。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
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
> `paper2skills-research/data/contracts/flow-07-workpack.json` 的 `card_candidates`。

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

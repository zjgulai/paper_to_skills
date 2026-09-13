# FLOW-01 契约撰写 · 批次 02（9 份）

> 公共材料见 `flow-01-common.md`（**先读它**）。本批 7 份 A 模板 / 2 份 B 模板。


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

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：任务依赖与资源冲突是经典调度/约束满足问题（RCPSP/CP），可输出可行序与预留方案。


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
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

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

## CTR-A-024 · 供需协调

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-016` 需求预测与补货计划 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-024-供需协调.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：供需平衡可建为多仓多期分配与S&OP优化模型。


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
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

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

## CTR-A-035 · 商品诊断

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-022` Amazon商品与搜索运营 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-035-商品诊断.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：Listing质量与转化诊断可用特征归因与预测模型输出优先修复清单。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
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

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 4 张
    - `p2s-dense-retrieval-ecommerce-semantic-search` · 源卡号 `Skill-Dense-Retrieval-Ecommerce-Semantic-Search` · 08-知识图谱 · 面向电商的稠密检索与语义排序
      - 全文卡：`/Users/lute/.dsh/skills/p2s-dense-retrieval-ecommerce-semantic-search/SKILL.md`
    - `p2s-diversity-reranking-smmr` · 源卡号 `Skill-Diversity-Reranking-SMMR` · 05-推荐系统 · Diversity-Aware Reranking with SMMR
      - 全文卡：`/Users/lute/.dsh/skills/p2s-diversity-reranking-smmr/SKILL.md`
    - `p2s-explainable-recommendation` · 源卡号 `Skill-Explainable-Recommendation` · 05-推荐系统 · Explainable Recommendation for Business Trust
      - 全文卡：`/Users/lute/.dsh/skills/p2s-explainable-recommendation/SKILL.md`
    - `p2s-kg-relation-completion-cblip` · 源卡号 `Skill-KG-Relation-Completion-CBLiP` · 08-知识图谱 · Knowledge Graph Relation Completion with CBLiP
      - 全文卡：`/Users/lute/.dsh/skills/p2s-kg-relation-completion-cblip/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 33 张
    - `p2s-a9-algorithm-sales-velocity-optimization` · 源卡号 `Skill-A9-Algorithm-Sales-Velocity-Optimization` · 25-搜索流量工程 · Skill-A9-Algorithm-Sales-Velocity-Optimization — A9 算法销量速度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-a9-algorithm-sales-velocity-optimization/SKILL.md`
    - `p2s-amazon-a10-algorithm-ranking` · 源卡号 `Skill-Amazon-A10-Algorithm-Ranking` · 13-广告分析 · Amazon A10 Algorithm Ranking — 亚马逊搜索排名因子建模与 Listing 可见度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-amazon-a10-algorithm-ranking/SKILL.md`
    - `p2s-amazon-search-ranking-factor-model` · 源卡号 `Skill-Amazon-Search-Ranking-Factor-Model` · 25-搜索流量工程 · Amazon 搜索排名因子权重建模 — 用 LightGBM+SHAP 解构 A9/A10 算法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-amazon-search-ranking-factor-model/SKILL.md`
    - `p2s-anomaly-detection-foundation-model` · 源卡号 `Skill-Anomaly-Detection-Foundation-Model` · 19-风控反欺诈 · Anomaly Detection Foundation Model — 异常检测基础模型：零样本时序异常感知
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-anomaly-detection-foundation-model/SKILL.md`
    - `p2s-autopkg-multimodal-product-attribute-kg` · 源卡号 `Skill-AutoPKG-Multimodal-Product-Attribute-KG` · 08-知识图谱 · AutoPKG — 多模态产品属性图谱自动构建：文本+图片→GMV提升
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-autopkg-multimodal-product-attribute-kg/SKILL.md`
    - `p2s-brand-hijacking-realtime-monitor` · 源卡号 `Skill-Brand-Hijacking-Realtime-Monitor` · 19-风控反欺诈 · Buy Box劫持实时监控 — 品牌跟卖检测与自动预警
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-brand-hijacking-realtime-monitor/SKILL.md`
    - `p2s-brand-keyword-hijack-alert` · 源卡号 `Skill-Brand-Keyword-Hijack-Alert` · 25-搜索流量工程 · Brand-Keyword-Hijack-Alert — 品牌词搜索下竞品展示份额超30%触发品牌防守广告扩展
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-brand-keyword-hijack-alert/SKILL.md`
    - `p2s-brand-listing-hijacking-detection` · 源卡号 `Skill-Brand-Listing-Hijacking-Detection` · 19-风控反欺诈 · Brand Listing Hijacking Detection — 电商品牌 Listing 劫持网络检测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-brand-listing-hijacking-detection/SKILL.md`
    - `p2s-caged-debiased-rec` · 源卡号 `Skill-CAGED-Debiased-Rec` · 05-推荐系统 · 因果图聚合权重去偏推荐 - CAGED
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-caged-debiased-rec/SKILL.md`
    - `p2s-category-tree-placement-optimizer` · 源卡号 `Skill-Category-Tree-Placement-Optimizer` · 25-搜索流量工程 · Skill-Category-Tree-Placement-Optimizer — 品类树节点竞争密度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-category-tree-placement-optimizer/SKILL.md`
    - `p2s-click-through-rate-title-optimizer` · 源卡号 `Skill-Click-Through-Rate-Title-Optimizer` · 25-搜索流量工程 · Skill-Click-Through-Rate-Title-Optimizer — 标题 CTR 机器学习优化器
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-click-through-rate-title-optimizer/SKILL.md`
    - `p2s-competitive-price-monitoring` · 源卡号 `Skill-Competitive-Price-Monitoring` · 17-价格优化 · Competitive Price Monitoring（竞品价格监测与响应）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-competitive-price-monitoring/SKILL.md`
    - `p2s-competitor-price-intelligence` · 源卡号 `Skill-Competitor-Price-Intelligence` · 17-价格优化 · Competitor Price Intelligence — 竞品价格实时监控与智能响应
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-competitor-price-intelligence/SKILL.md`
    - `p2s-demand-anomaly-causal-attribution` · 源卡号 `Skill-Demand-Anomaly-Causal-Attribution` · 03-时间序列 · Demand Anomaly Causal Attribution — 需求异常因果归因（销量突变根因诊断）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-demand-anomaly-causal-attribution/SKILL.md`
    - `p2s-hierarchical-product-kg-construction` · 源卡号 `Skill-Hierarchical-Product-KG-Construction` · 08-知识图谱 · 层级商品知识图谱自动构建（图片→KG）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hierarchical-product-kg-construction/SKILL.md`
    - `p2s-hijacker-seller-network-analysis` · 源卡号 `Skill-Hijacker-Seller-Network-Analysis` · 19-风控反欺诈 · Hijacker Seller Network Analysis — 跟卖卖家网络图谱识别有组织的跟卖集团
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hijacker-seller-network-analysis/SKILL.md`
    - `p2s-index-health-monitoring` · 源卡号 `Skill-Index-Health-Monitoring` · 25-搜索流量工程 · 索引健康度监控 — 全链路搜索索引覆盖率与收录状态追踪
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-index-health-monitoring/SKILL.md`
    - `p2s-kg-application-patterns` · 源卡号 `Skill-KG-Application-Patterns` · 08-知识图谱 · KG Application Patterns — 知识图谱下游应用：从构建到推荐/搜索/冷启动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-application-patterns/SKILL.md`
    - `p2s-keyword-cannibalization-detection` · 源卡号 `Skill-Keyword-Cannibalization-Detection` · 25-搜索流量工程 · 关键词自相竞争检测 — 用二部图 Jaccard 算法发现同店 SKU 流量内耗
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-keyword-cannibalization-detection/SKILL.md`
    - `p2s-knowledge-graph-rec` · 源卡号 `Skill-Knowledge-Graph-Rec` · 05-推荐系统 · Knowledge Graph Enhanced Recommendation — 知识图谱增强推荐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-knowledge-graph-rec/SKILL.md`
    - `p2s-listing-health-diagnostic` · 源卡号 `Skill-Listing-Health-Diagnostic` · 13-广告分析 · Listing Health Diagnostic — Listing 快诊：多维度评分驱动的上架质量诊断
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-health-diagnostic/SKILL.md`
    - `p2s-listing-suppression-detection` · 源卡号 `Skill-Listing-Suppression-Detection` · 19-风控反欺诈 · Listing Suppression Detection — Listing 被平台隐藏/降权检测（非账号问题）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-suppression-detection/SKILL.md`
    - `p2s-multimodal-product-rec` · 源卡号 `Skill-MultiModal-Product-Rec` · 05-推荐系统 · MultiModal Product Recommendation — 多模态产品推荐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multimodal-product-rec/SKILL.md`
    - `p2s-position-bias-debiasing-rec` · 源卡号 `Skill-Position-Bias-Debiasing-Rec` · 05-推荐系统 · Position Bias Debiasing Recommendation — 位置偏差去偏公平推荐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-position-bias-debiasing-rec/SKILL.md`
    - `p2s-price-signal-collection` · 源卡号 `Skill-Price-Signal-Collection` · 17-价格优化 · Price Signal Collection — 竞品价格信号实时采集与结构化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-price-signal-collection/SKILL.md`
    - `p2s-real-time-competitive-repricing` · 源卡号 `Skill-Real-Time-Competitive-Repricing` · 17-价格优化 · Real-Time Competitive Repricing — 竞品价格监测与深度强化学习自动重定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-real-time-competitive-repricing/SKILL.md`
    - `p2s-revenue-per-available-sku-revpas` · 源卡号 `Skill-Revenue-Per-Available-SKU-REVPAS` · 17-价格优化 · Revenue Per Available SKU（REVPAS）— 以可售库存单位衡量定价效率
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-revenue-per-available-sku-revpas/SKILL.md`
    - `p2s-sku-level-margin-attribution-ontology` · 源卡号 `Skill-SKU-Level-Margin-Attribution-Ontology` · 24-标签工程 · SKU级利润归因本体 — 从GMV到净利润的全链路成本拆解与Tag驱动的利润诊断
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sku-level-margin-attribution-ontology/SKILL.md`
    - `p2s-search-position-click-elasticity` · 源卡号 `Skill-Search-Position-Click-Elasticity` · 25-搜索流量工程 · 搜索位置点击弹性分析 — 用 Examination Model 分离位置偏差估计真实 CTR
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-position-click-elasticity/SKILL.md`
    - `p2s-search-rank-recovery-auto-action` · 源卡号 `Skill-Search-Rank-Recovery-Auto-Action` · 25-搜索流量工程 · Search-Rank-Recovery-Auto-Action — 核心关键词排名跌出Page1自动触发三步恢复行动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-rank-recovery-auto-action/SKILL.md`
    - `p2s-search-share-of-voice` · 源卡号 `Skill-Search-Share-of-Voice` · 25-搜索流量工程 · 搜索声量份额 — 关键词维度市场份额追踪与竞争格局监测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-share-of-voice/SKILL.md`
    - `p2s-search-signal-realtime-pipeline` · 源卡号 `Skill-Search-Signal-Realtime-Pipeline` · 25-搜索流量工程 · A9 搜索信号实时采集 Pipeline — 排名变化与竞品动态的流式监控
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-signal-realtime-pipeline/SKILL.md`
    - `p2s-vlm-ecommerce-adaptation` · 源卡号 `Skill-VLM-Ecommerce-Adaptation` · 16-智能体工程 · VLM E-commerce Adaptation — 大规模视觉语言模型电商适配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-vlm-ecommerce-adaptation/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 1 张
    - `Skill-Product-Attribute-Graph-Parsing` · 07-NLP-VOC · Skill-Product-Attribute-Graph-Parsing
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Product-Attribute-Graph-Parsing.md`

> 候选总数 38。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-041 · 线索评估

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-025` 零售渠道与B2B拓展 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-041-线索评估.md` |
| `status` 初值 | **待卡**（无候选卡 —— 这就是扩充工单） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：B2B线索评分是标准Lead Scoring/倾向建模问题，可直接排序。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
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

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

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

## CTR-A-048 · 内容实验

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-030` 内容与创意策划 |
| 域 / 面 | `DOM-05` / `PLN-OPS` |
| flows | FLOW-01, FLOW-02, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-048-内容实验.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：内容A/B与多元素测试（DOE/bandit）是标准实验设计问题。
- **§F.5 边界条目**：DOE/bandit 是标准实验设计 ⇒ 降级条件：若内容产出本身无法被算法替代，则降 B

**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
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
| FLOW-04/STG-01 | D | 信号接收 | 接收新市场、平台、店铺或商品上线请求 | 市场进入信号 |
| FLOW-04/STG-02 | R | 范围确定与Case创建 | 确定市场、渠道账号、商品集合、时间窗和进入成功条件 | 市场进入Case Charter |
| FLOW-04/STG-03 | D | 上下文装配 | 装配市场机会、准入、IP/合规、本地化、履约、售后、结算和品牌能力 | FLOW-04 Context Manifest |
| FLOW-04/STG-06 | R | Assurance接收门禁 | 检查适用策略、市场规则、账号主体、内容、履约和结算准备 | FLOW-04 Assurance Decision |
| FLOW-04/STG-07 | D | 受控动作 | 提交受控上架Intent或NoActionRecord；具体市场和接口待配置 | 上架尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 1 张
    - `p2s-ab-experimental-design` · 源卡号 `Skill-AB-Experimental-Design` · 02-A_B实验 · Skill: A/B 实验设计基础
      - 全文卡：`/Users/lute/.dsh/skills/p2s-ab-experimental-design/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 10 张
    - `p2s-ai-video-ab-test-creative` · 源卡号 `Skill-AI-Video-AB-Test-Creative` · 20-AI视频生成 · AI Video AB Test Creative — AI视频素材A/B测试（广告创意自动化迭代优化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-video-ab-test-creative/SKILL.md`
    - `p2s-causal-attribution-bridge` · 源卡号 `Skill-Causal-Attribution-Bridge` · 01-因果推断 · Causal Attribution Bridge（因果归因桥梁）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-attribution-bridge/SKILL.md`
    - `p2s-loss-aversion-promotion-design` · 源卡号 `Skill-Loss-Aversion-Promotion-Design` · 15-营销投放分析 · 损失厌恶促销设计 — 「限时最后3件」比「立省30元」点击率高35%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-loss-aversion-promotion-design/SKILL.md`
    - `p2s-multi-cell-factorial-experiment` · 源卡号 `Skill-Multi-Cell-Factorial-Experiment` · 02-A_B实验 · Multi-cell Factorial Experiment — 多因素析因实验设计（电商 2^k 实验）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-cell-factorial-experiment/SKILL.md`
    - `p2s-nlp-copy-ab-test-optimizer` · 源卡号 `Skill-NLP-Copy-AB-Test-Optimizer` · 02-A_B实验 · NLP Copy AB Test Optimizer — CTR 加权偏好优化 + 遗传算法的文案 A/B 测试闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-nlp-copy-ab-test-optimizer/SKILL.md`
    - `p2s-shopify-landing-page-cro` · 源卡号 `Skill-Shopify-Landing-Page-CRO` · 14-用户分析 · Shopify Landing Page CRO — 独立站落地页转化率优化：ML驱动的A/B测试与个性化元素配置
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-shopify-landing-page-cro/SKILL.md`
    - `p2s-short-video-commerce-attribution` · 源卡号 `Skill-Short-Video-Commerce-Attribution` · 13-广告分析 · 短视频内容归因建模 — 视频特征 XGBoost 驱动转化分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-short-video-commerce-attribution/SKILL.md`
    - `p2s-short-video-thumbnail-ctr-predictor` · 源卡号 `Skill-Short-Video-Thumbnail-CTR-Predictor` · 20-AI视频生成 · Short Video Thumbnail CTR Predictor — 封面图CTR预测（AB测试前AI预评估）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-short-video-thumbnail-ctr-predictor/SKILL.md`
    - `p2s-tiktok-shop-content-attribution` · 源卡号 `Skill-TikTok-Shop-Content-Attribution` · 13-广告分析 · TikTok Shop Content Attribution — 短视频带货兴趣图谱归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-shop-content-attribution/SKILL.md`
    - `p2s-video-emotion-resonance-score` · 源卡号 `Skill-Video-Emotion-Resonance-Score` · 20-AI视频生成 · Video Emotion Resonance Score — 视频情感共鸣评分（母婴内容感动度量化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-video-emotion-resonance-score/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 2 张
    - `Skill-Persona-Based-AB-Simulation` · 02-A_B实验 · Skill-Persona-Based-AB-Simulation
      - 全文卡：`paper2skills-vault/02-A_B实验/Skill-Persona-Based-AB-Simulation.md`
    - `Skill-Self-Improving-LLM-Agent-Pipeline` · 07-NLP-VOC · Skill-Self-Improving-LLM-Agent-Pipeline
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-Self-Improving-LLM-Agent-Pipeline.md`

> 候选总数 13。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-058 · 渠道对账

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-040` GMV结算与会计对账 |
| 域 / 面 | `DOM-07` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03, FLOW-05, FLOW-06, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-058-渠道对账.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：对账是记录匹配问题（模糊匹配、异常检测、差异归因），可高度自动化。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
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
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 1 张
    - `p2s-agentic-etl-data-pipeline` · 源卡号 `Skill-Agentic-ETL-Data-Pipeline` · 09-DataAgent-LLM · Agentic ETL数据管道 — LLM驱动的自动化数据清洗与转换
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agentic-etl-data-pipeline/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 1。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-065 · 指标契约

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-045` 业务口径与主数据 |
| 域 / 面 | `DOM-08` / `PLN-PLT` |
| flows | FLOW-01, FLOW-06, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-065-指标契约.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：指标定义一致性可用语义比对与血缘校验自动化（含口径冲突检测）。


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
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 2 张
    - `p2s-feature-engineering` · 源卡号 `Skill-Feature-Engineering` · 12-ML基础 · Feature Engineering for E-Commerce Machine Learning
      - 全文卡：`/Users/lute/.dsh/skills/p2s-feature-engineering/SKILL.md`
    - `p2s-voc-proxy-nps-aipl` · 源卡号 `Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎` · 07-NLP-VOC · 'Skill: VOC Proxy NPS × AIPL 统一标签萃取引擎'
      - 全文卡：`/Users/lute/.dsh/skills/p2s-voc-proxy-nps-aipl/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 6 张
    - `p2s-advertising-api-unified-schema` · 源卡号 `Skill-Advertising-API-Unified-Schema` · 22-数据采集工程 · Advertising API Unified Schema — 多平台广告统一数据模型（Amazon/Meta/Tik
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-advertising-api-unified-schema/SKILL.md`
    - `p2s-data-mesh-ecommerce` · 源卡号 `Skill-Data-Mesh-Ecommerce` · 22-数据采集工程 · Data Mesh for Ecommerce — 数据网格架构
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-data-mesh-ecommerce/SKILL.md`
    - `p2s-pvm-attribution-window-harmonization` · 源卡号 `Skill-PVM-Attribution-Window-Harmonization` · 13-广告分析 · PVM 跨平台广告归因窗口统一化 - 母婴跨境多渠道 ROAS 去偏
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-pvm-attribution-window-harmonization/SKILL.md`
    - `p2s-supply-chain-ml-features` · 源卡号 `Skill-Supply-Chain-ML-Features` · 12-ML基础 · Supply Chain ML Feature Engineering — 供应链 ML 特征工程：时序+图+统计三维
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-ml-features/SKILL.md`
    - `p2s-tag-schema-engineering-lifecycle` · 源卡号 `Skill-Tag-Schema-Engineering-Lifecycle` · 24-标签工程 · 标签Schema工程与生命周期管理 — 企业级Tag类型设计、Schema约束与版本治理
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tag-schema-engineering-lifecycle/SKILL.md`
    - `p2s-text2sql-schema-linking` · 源卡号 `Skill-Text2SQL-Schema-Linking` · 09-DataAgent-LLM · Schema-Linking感知Text2SQL — 列对齐与表联接推断
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-text2sql-schema-linking/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 8。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-026 · 站点运营

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-023` 独立站经营与转化 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-04, FLOW-05, FLOW-06 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-026-站点运营.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：站点指标可监测分析，但运营决策依赖内容与品牌判断。


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
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 15 张
    - `p2s-a9-algorithm-sales-velocity-optimization` · 源卡号 `Skill-A9-Algorithm-Sales-Velocity-Optimization` · 25-搜索流量工程 · Skill-A9-Algorithm-Sales-Velocity-Optimization — A9 算法销量速度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-a9-algorithm-sales-velocity-optimization/SKILL.md`
    - `p2s-category-tree-placement-optimizer` · 源卡号 `Skill-Category-Tree-Placement-Optimizer` · 25-搜索流量工程 · Skill-Category-Tree-Placement-Optimizer — 品类树节点竞争密度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-category-tree-placement-optimizer/SKILL.md`
    - `p2s-cross-platform-user-transfer` · 源卡号 `Skill-Cross-Platform-User-Transfer` · 14-用户分析 · Cross-Platform User Behavior Transfer — 跨平台用户行为迁移：亚马逊行为驱动独立站
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-platform-user-transfer/SKILL.md`
    - `p2s-graph-foundation-model-recommendation` · 源卡号 `Skill-Graph-Foundation-Model-Recommendation` · 08-知识图谱 · Graph Foundation Model Recommendation — 图基础模型推荐：跨图迁移的零样本推荐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-graph-foundation-model-recommendation/SKILL.md`
    - `p2s-index-health-monitoring` · 源卡号 `Skill-Index-Health-Monitoring` · 25-搜索流量工程 · 索引健康度监控 — 全链路搜索索引覆盖率与收录状态追踪
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-index-health-monitoring/SKILL.md`
    - `p2s-live-commerce-stream-algorithm` · 源卡号 `Skill-Live-Commerce-Stream-Algorithm` · 20-AI视频生成 · Live Commerce Stream Algorithm — 直播电商算法建模与互动信号优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-live-commerce-stream-algorithm/SKILL.md`
    - `p2s-live-stream-conversion-predictor` · 源卡号 `Skill-Live-Stream-Conversion-Predictor` · 20-AI视频生成 · Live Stream Conversion Predictor — 直播转化率实时预测（GMV预警+切品时机）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-live-stream-conversion-predictor/SKILL.md`
    - `p2s-mas-ecommerce-ops-automation` · 源卡号 `Skill-MAS-Ecommerce-Ops-Automation` · 10-MAS · 多 Agent 电商运营自动化 — 补货/广告/客服全天候自动化编排
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-ecommerce-ops-automation/SKILL.md`
    - `p2s-multi-channel-price-consistency` · 源卡号 `Skill-Multi-Channel-Price-Consistency` · 17-价格优化 · Multi-Channel Price Consistency — 多渠道价格一致性管理（防跨平台价格冲突）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-channel-price-consistency/SKILL.md`
    - `p2s-nonitem-page-path-modeling` · 源卡号 `Skill-NonItem-Page-Path-Modeling` · 14-用户分析 · 非商品页路径建模 - 导航页在用户旅程中的转化贡献
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-nonitem-page-path-modeling/SKILL.md`
    - `p2s-search-driven-logistics-promise` · 源卡号 `Skill-Search-Driven-Logistics-Promise` · 25-搜索流量工程 · Search-Driven Logistics Promise — 搜索词意图信号驱动履约时效承诺优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-driven-logistics-promise/SKILL.md`
    - `p2s-search-rank-recovery-auto-action` · 源卡号 `Skill-Search-Rank-Recovery-Auto-Action` · 25-搜索流量工程 · Search-Rank-Recovery-Auto-Action — 核心关键词排名跌出Page1自动触发三步恢复行动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-rank-recovery-auto-action/SKILL.md`
    - `p2s-tag-enhanced-personalized-recommendation` · 源卡号 `Skill-Tag-Enhanced-Personalized-Recommendation` · 24-标签工程 · Tag-Enhanced Personalized Recommendation — 标签感知的个性化推荐系统
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tag-enhanced-personalized-recommendation/SKILL.md`
    - `p2s-trajectory-pattern-mining` · 源卡号 `Skill-Trajectory-Pattern-Mining` · 14-用户分析 · 用户行为轨迹模式挖掘与预测 - 变阶马尔可夫模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-trajectory-pattern-mining/SKILL.md`
    - `p2s-voc-driven-recommendation-signal` · 源卡号 `Skill-VOC-Driven-Recommendation-Signal` · 07-NLP-VOC · VOC-Driven Recommendation Signal — 评论语义驱动的推荐增强：NLP-VOC×推荐系统桥
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-voc-driven-recommendation-signal/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 15。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-032 · 本地化

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-028` 本地化与市场适配 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-02, FLOW-04, FLOW-05 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-032-本地化.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：翻译可用MT+质量估计自动化，但市场语境适配需母语审校与当地证据。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
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

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 1 张
    - `p2s-multilingual-ner-universal-v2` · 源卡号 `Skill-Multilingual-NER-Universal-v2` · 08-知识图谱 · Multilingual Named Entity Recognition (Universal NER v2)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-multilingual-ner-universal-v2/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 19 张
    - `p2s-ai-cultural-sensitivity-localization` · 源卡号 `Skill-AI-Cultural-Sensitivity-Localization` · 11-AI人文 · AI Cultural Sensitivity Localization — AI文化敏感性本地化评估（跨市场内容风险）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-cultural-sensitivity-localization/SKILL.md`
    - `p2s-aquarius-brand-video-generation` · 源卡号 `Skill-Aquarius-Brand-Video-Generation` · 20-AI视频生成 · Aquarius — Brand Video Generation（品牌营销视频生成）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-aquarius-brand-video-generation/SKILL.md`
    - `p2s-cross-border-price-harmonization` · 源卡号 `Skill-Cross-Border-Price-Harmonization` · 17-价格优化 · Cross-Border Price Harmonization（跨境价格协调）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-price-harmonization/SKILL.md`
    - `p2s-cross-cultural-content-adaptation` · 源卡号 `Skill-Cross-Cultural-Content-Adaptation` · 11-AI人文 · 跨文化内容自动适配 — 文化距离量化与内容风格迁移
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-cultural-content-adaptation/SKILL.md`
    - `p2s-cross-cultural-marketing-adaptation` · 源卡号 `Skill-Cross-Cultural-Marketing-Adaptation` · 11-AI人文 · Cross-Cultural Marketing Adaptation — 多语言 CAM 嵌入驱动的跨文化营销适配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-cultural-marketing-adaptation/SKILL.md`
    - `p2s-cross-market-content-localization` · 源卡号 `Skill-Cross-Market-Content-Localization` · 20-AI视频生成 · 文化感知内容本地化 — 跨市场文化风险识别与自动替换
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-market-content-localization/SKILL.md`
    - `p2s-cultural-adaptation-agent` · 源卡号 `Skill-Cultural-Adaptation-Agent` · 16-智能体工程 · Cultural Adaptation Agent — 跨文化适应：母婴跨境的本地化 AI 策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cultural-adaptation-agent/SKILL.md`
    - `p2s-few-shot-review-classification` · 源卡号 `Skill-Few-Shot-Review-Classification` · 07-NLP-VOC · 少样本评论分类 — Prototypical Networks 从英语迁移到新市场语言
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-few-shot-review-classification/SKILL.md`
    - `p2s-international-search-localization` · 源卡号 `Skill-International-Search-Localization` · 25-搜索流量工程 · Skill-International-Search-Localization — 跨市场搜索关键词本地化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-international-search-localization/SKILL.md`
    - `p2s-multi-market-search-localization` · 源卡号 `Skill-Multi-Market-Search-Localization` · 25-搜索流量工程 · 多市场搜索词本地化 — 多站点关键词迁移与本地语言适配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-market-search-localization/SKILL.md`
    - `p2s-multi-market-voc-cross-analysis` · 源卡号 `Skill-Multi-Market-VOC-Cross-Analysis` · 07-NLP-VOC · Skill-Multi-Market-VOC-Cross-Analysis — 多市场VOC交叉分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-market-voc-cross-analysis/SKILL.md`
    - `p2s-multilingual-customer-service-translation` · 源卡号 `Skill-Multilingual-Customer-Service-Translation` · 14-用户分析 · Multilingual Customer Service Translation — 多语言客服自动翻译与情绪感知保全
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-customer-service-translation/SKILL.md`
    - `p2s-multilingual-listing-generation` · 源卡号 `Skill-Multilingual-Listing-Generation` · 20-AI视频生成 · LLM多语言Listing批量生成 — 1小时完成5语言新品入驻
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-listing-generation/SKILL.md`
    - `p2s-multilingual-listing-localization` · 源卡号 `Skill-Multilingual-Listing-Localization` · 13-广告分析 · Multilingual Listing Localization — LLM 驱动的多语言 Listing 本地化与文
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-listing-localization/SKILL.md`
    - `p2s-multilingual-live-virtual-anchor-clone` · 源卡号 `Skill-Multilingual-Live-Virtual-Anchor-Clone` · 20-AI视频生成 · 多语言直播虚拟主播实时克隆 — 跨语言直播带货数字人生成与实时驱动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-live-virtual-anchor-clone/SKILL.md`
    - `p2s-multilingual-nlp-pipeline` · 源卡号 `Skill-Multilingual-NLP-Pipeline` · 07-NLP-VOC · 多语言 NLP 管道 — mBERT Zero-Shot 跨语言情感/实体提取
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-nlp-pipeline/SKILL.md`
    - `p2s-multilingual-sentiment-alignment` · 源卡号 `Skill-Multilingual-Sentiment-Alignment` · 07-NLP-VOC · Multilingual Sentiment Alignment — 多语言情感对齐（跨语言评论情感一致性）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-sentiment-alignment/SKILL.md`
    - `p2s-multilingual-subtitle-auto-generator` · 源卡号 `Skill-Multilingual-Subtitle-Auto-Generator` · 20-AI视频生成 · Skill-Multilingual-Subtitle-Auto-Generator — 多语言字幕自动生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-subtitle-auto-generator/SKILL.md`
    - `p2s-virbo-multilingual-avatar-ugc` · 源卡号 `Skill-Virbo-Multilingual-Avatar-UGC` · 20-AI视频生成 · Virbo — Multilingual Avatar UGC（多语言虚拟人UGC批量生产）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-virbo-multilingual-avatar-ugc/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 20。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。

# FLOW-02 契约撰写 · 批次 13（3 份）

> 公共材料见 `flow-02-common.md`（**先读它**）。本批 1 份 A 模板 / 2 份 B 模板。


---

## CTR-A-028 · 质量分析

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-018` 生产协同与质量控制 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-02, FLOW-03, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-028-质量分析.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：SPC控制图、过程能力指数与根因相关分析是成熟质量统计方法。


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
> `paper2skills-research/data/contracts/flow-02-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 12 张
    - `p2s-cold-chain-temperature-monitoring` · 源卡号 `Skill-Cold-Chain-Temperature-Monitoring` · 18-物流履约 · 冷链温控全程监测 — 母婴辅食物流的温度合规智能预警
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cold-chain-temperature-monitoring/SKILL.md`
    - `p2s-consumer-complaint-recall-prediction` · 源卡号 `Skill-Consumer-Complaint-Recall-Prediction` · 21-合规决策 · Consumer Complaint Recall Prediction — 消费者投诉驱动的召回风险预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-consumer-complaint-recall-prediction/SKILL.md`
    - `p2s-customer-complaint-supply-root-cause-kpi` · 源卡号 `Skill-Customer-Complaint-Supply-Root-Cause-KPI` · 04-供应链 · 客诉供应链根因KPI与闭环治理 — 供应链原因导致客诉率/处理时效/预防闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-customer-complaint-supply-root-cause-kpi/SKILL.md`
    - `p2s-fake-review-detection` · 源卡号 `Skill-Fake-Review-Detection` · 22-数据采集工程 · Fake Review Detection — 假评论检测：图神经网络+LLM 可解释欺诈识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fake-review-detection/SKILL.md`
    - `p2s-graph-rag-knowledge-retrieval` · 源卡号 `Skill-Graph-RAG-Knowledge-Retrieval` · 08-知识图谱 · Graph RAG Knowledge Retrieval — 知识图谱增强检索：结构化知识驱动精准问答
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-graph-rag-knowledge-retrieval/SKILL.md`
    - `p2s-return-root-cause-attribution-graph` · 源卡号 `Skill-Return-Root-Cause-Attribution-Graph` · 24-标签工程 · 退货根因归因图谱 — 多层次退货原因知识图谱与供应链改善闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-return-root-cause-attribution-graph/SKILL.md`
    - `p2s-review-velocity-anomaly-detector` · 源卡号 `Skill-Review-Velocity-Anomaly-Detector` · 22-数据采集工程 · 差评速率异常检测 — 区分竞品恶意攻击与真实产品问题
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-review-velocity-anomaly-detector/SKILL.md`
    - `p2s-safety-concern-signal-extraction` · 源卡号 `Skill-Safety-Concern-Signal-Extraction` · 07-NLP-VOC · Skill-Safety-Concern-Signal-Extraction — 安全隐患信号提取
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-safety-concern-signal-extraction/SKILL.md`
    - `p2s-supplier-delivery-quality-rate-kpi` · 源卡号 `Skill-Supplier-Delivery-Quality-Rate-KPI` · 04-供应链 · 供应商来料质量IQC-KPI体系 — 入库检验合格率、质量成本与批次拒收分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-delivery-quality-rate-kpi/SKILL.md`
    - `p2s-supply-chain-counterfeit-detection` · 源卡号 `Skill-Supply-Chain-Counterfeit-Detection` · 19-风控反欺诈 · Supply Chain Counterfeit Detection — 供应链仿冒品检测原材料/包装视觉验真
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-counterfeit-detection/SKILL.md`
    - `p2s-voc-triggered-inventory-signal` · 源卡号 `Skill-VOC-Triggered-Inventory-Signal` · 07-NLP-VOC · VOC Triggered Inventory Signal — VOC触发库存信号（差评激增→供应链预警）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-voc-triggered-inventory-signal/SKILL.md`
    - `p2s-warehouse-inbound-quality-accuracy-kpi` · 源卡号 `Skill-Warehouse-Inbound-Quality-Accuracy-KPI` · 04-供应链 · 仓储入库质量与收货准确率KPI — 收货差异检测、破损率管控与验收时效
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-warehouse-inbound-quality-accuracy-kpi/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 12。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-018 · 软件需求分析

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-012` 软件算法与应用生态 |
| 域 / 面 | `DOM-02` / `PLN-OPS` |
| flows | FLOW-02, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-018-软件需求分析.md` |
| `status` 初值 | **待卡**（无候选卡 —— 这就是扩充工单） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：需求歧义与一致性可NLP/形式化辅助，但需求确认依赖业务与架构判断。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
| FLOW-02/STG-01 | D | 信号接收 | 接收需求证据、使用问题或产品组合缺口 | 新品机会信号 |
| FLOW-02/STG-02 | R | 范围确定与Case创建 | 确定用户问题、市场、产品范围、验证目标与停止条件 | 新品验证Case Charter |
| FLOW-02/STG-03 | D | 上下文装配 | 装配需求、竞品、产品定义、工程、OEM、质量、合规和验证能力 | FLOW-02 Context Manifest |
| FLOW-02/STG-06 | R | Assurance接收门禁 | 检查可证伪性、实物/专业证据来源、合规、资源与混杂因素 | FLOW-02 Assurance Decision |
| FLOW-02/STG-07 | D | 受控动作 | 发起受控外部验证Intent或记录无需动作；不得把模型判断当实物回执 | 验证尝试或无动作记录 |
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-02-workpack.json` 的 `card_candidates`。

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

## CTR-B-055 · 产品准入核对

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-044` 产品合规与隐私 |
| 域 / 面 | `DOM-07` / `PLN-CTL` |
| flows | FLOW-02, FLOW-04, FLOW-05, FLOW-06, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-055-产品准入核对.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：准入矩阵可规则化+法规库匹配，但最终准入须法定依据与外部认证。


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
> `paper2skills-research/data/contracts/flow-02-workpack.json` 的 `card_candidates`。

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

# FLOW-03 契约撰写 · 批次 04（6 份）

> 公共材料见 `flow-03-common.md`（**先读它**）。本批 5 份 A 模板 / 1 份 B 模板。


---

## CTR-A-019 · 采购比价

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-015` 采购与合同履约 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-03 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-019-采购比价.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：比价与最优采购批量是标准运筹问题（EOQ、供应商组合优化、价格指数比对）。
> 引用纪律：这句判定可以进正文，但**不得标成「材料 §F.3 原文」**。S1 实测 139 份作业包全都
> 把它标成了「材料 §F.3 原文」，撰写人于是写出「材料 §F.3 原文写明『流程合理性须业务确认』」
> 这类**把我们的分析记到材料账上**的句子 —— 核验器实测该句在材料里 0 命中。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-03/STG-04 | M | 证据收集与诊断 | 核对可售/预留/在途、已有订单、需求区间、交付周期、产能、条款和现金 | 供需证据与约束诊断 |
| FLOW-03/STG-05 | M | 方案与标准产物 | 比较补货、调拨、控投放和清货方案 | 供应计划与动作方案 |
| FLOW-03/STG-08 | M | 结果核验、关闭与异步学习 | 核对订单、到货、质检、可售、结算和资金影响 | 供需关闭记录 |
| FLOW-03/STG-01 | D | 信号接收 | 接收库存、预测、交期、采购或履约变化 | 供需信号 |
| FLOW-03/STG-02 | R | 范围确定与Case创建 | 确定账号、SKU/商品、仓库、供应商、时间窗和供需目标 | 供需Case Charter |
| FLOW-03/STG-03 | D | 上下文装配 | 装配预测、库存、采购、OEM、物流、财务与质量能力 | FLOW-03 Context Manifest |
| FLOW-03/STG-06 | R | Assurance接收门禁 | 检查重复订单、MOQ、产能、现金、质量、合同和对象状态 | FLOW-03 Assurance Decision |
| FLOW-03/STG-07 | D | 受控动作 | 提交采购/调拨Intent或NoActionRecord；具体策略和ERP schema待配置 | 执行尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 1 张
    - `p2s-did-difference-in-differences` · 源卡号 `Skill-DiD-Difference-in-Differences` · 01-因果推断 · Difference-in-Differences (DiD) for Causal Effect Estimation
      - 全文卡：`/Users/lute/.dsh/skills/p2s-did-difference-in-differences/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 13 张
    - `p2s-agenticpay-procurement-negotiation` · 源卡号 `Skill-AgenticPay-Procurement-Negotiation` · 10-MAS · AgenticPay — LLM 多 Agent 采购谈判：自主完成价格与 MOQ 协商
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agenticpay-procurement-negotiation/SKILL.md`
    - `p2s-bom-cost-rollup-tag-engine` · 源卡号 `Skill-BOM-Cost-Rollup-Tag-Engine` · 04-供应链 · BOM成本卷积标签引擎 — 从原材料到制成品的全层级成本精算与Tag驱动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bom-cost-rollup-tag-engine/SKILL.md`
    - `p2s-carrier-selection-ml` · 源卡号 `Skill-Carrier-Selection-ML` · 18-物流履约 · 承运商智能选择 — ML驱动的跨境配送商优化决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-carrier-selection-ml/SKILL.md`
    - `p2s-document-intelligence-parsing` · 源卡号 `Skill-Document-Intelligence-Parsing` · 22-数据采集工程 · Document Intelligence Parsing — LLM 驱动的文档智能解析：图文统一 OCR、跨页表格恢
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-document-intelligence-parsing/SKILL.md`
    - `p2s-dynamic-lot-sizing-moq` · 源卡号 `Skill-Dynamic-Lot-Sizing-MOQ` · 04-供应链 · Efficient Algorithms for the Joint Replenishment Problem wit
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-lot-sizing-moq/SKILL.md`
    - `p2s-dynamic-payment-terms-tag-engine` · 源卡号 `Skill-Dynamic-Payment-Terms-Tag-Engine` · 04-供应链 · 动态账期标签引擎 — 基于现金流预测的供应商账期智能优化与动态调整
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-payment-terms-tag-engine/SKILL.md`
    - `p2s-moq-payment-terms-optimization` · 源卡号 `Skill-MOQ-Payment-Terms-Optimization` · 04-供应链 · MOQ与账期联动优化决策 — 最小起订量与付款条件的现金流效益模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-moq-payment-terms-optimization/SKILL.md`
    - `p2s-multi-sku-procurement-budget-allocation` · 源卡号 `Skill-Multi-SKU-Procurement-Budget-Allocation` · 04-供应链 · Constructing decision rules for multiproduct newsvendors
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-sku-procurement-budget-allocation/SKILL.md`
    - `p2s-procurement-budget-rolling-reforecast` · 源卡号 `Skill-Procurement-Budget-Rolling-Reforecast` · 04-供应链 · 采购预算滚动重测 — 季度滚动预测与偏差管理的动态预算调整机制
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-procurement-budget-rolling-reforecast/SKILL.md`
    - `p2s-procurement-cost-kpi-price-achievement` · 源卡号 `Skill-Procurement-Cost-KPI-Price-Achievement` · 04-供应链 · 采购价格达成率与降本KPI体系 — 全链路降本量化追踪与价格偏差归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-procurement-cost-kpi-price-achievement/SKILL.md`
    - `p2s-procurement-email-extraction` · 源卡号 `Skill-Procurement-Email-Extraction` · 22-数据采集工程 · Procurement Email Extraction — 采购邮件结构化提取：多供应商报价聚合与合规验证
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-procurement-email-extraction/SKILL.md`
    - `p2s-supplier-negotiation-llm-agent` · 源卡号 `Skill-Supplier-Negotiation-LLM-Agent` · 16-智能体工程 · LLM驱动供应商谈判智能体 — 结构化采购谈判自动化与议价策略优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-negotiation-llm-agent/SKILL.md`
    - `p2s-supply-chain-total-cost-tco-model` · 源卡号 `Skill-Supply-Chain-Total-Cost-TCO-Model` · 04-供应链 · 全供应链总成本TCO模型 — 采购+仓储+物流+质量全链路成本分摊与年降目标
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-total-cost-tco-model/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 14。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-025 · 库存分层

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-017` 库存与商品生命周期 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-025-库存分层.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：ABC/XYZ分层与安全库存分位计算是标准库存分析，可直接产出分层与目标库存。
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
| FLOW-07/STG-04 | M | 证据收集与诊断 | 核对事件时间线、受影响批次/商品/账号、已发生动作、库存和客户范围 | 事件范围与根因诊断 |
| FLOW-07/STG-05 | M | 方案与标准产物 | 形成隔离、处置、客户影响和恢复方案 | 重大事件处置包 |
| FLOW-07/STG-08 | M | 结果核验、关闭与异步学习 | 核对紧急保护与正常处置的逐对象权威回执、影响受控、根因和恢复证据 | 重大事件关闭与全动作对账记录 |
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
| FLOW-07/STG-01 | D | 信号接收 | 接收并验证质量、严重客诉或账号重大风险信号；符合D-026候选条件时交由模型外Emergency Guard评估 | 重大事件信号与Guard评估入口 |
| FLOW-07/STG-02 | R | 范围确定与Case创建 | 按已发布规则形成唯一FLOW-07 Case Charter；PROTECT或FREEZE时原子写入Case、STG-01/02接收记录与Guard证据 | 重大事件Case Charter与本地原子提交记录 |
| FLOW-07/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配产品、供应、服务、渠道、法务、合规和安全能力 | FLOW-07 Context Manifest |
| FLOW-07/STG-06 | R | Assurance接收门禁 | 检查证据、范围、适用策略、外部承诺和恢复条件 | FLOW-07 Assurance Decision |
| FLOW-07/STG-07 | D | 受控动作 | 处理正常处置Intent或NoActionRecord；恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁和永久变更只能在本阶段执行 | 正常处置、恢复尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 2 张
    - `p2s-multi-echelon-inventory` · 源卡号 `Skill-Multi-Echelon-Inventory` · 04-供应链 · Multi-Echelon Inventory Optimization (多阶库存优化)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-multi-echelon-inventory/SKILL.md`
    - `p2s-two-echelon-inventory-drl` · 源卡号 `Skill-Two-Echelon-Inventory-DRL` · 04-供应链 · Deep RL for Two-Echelon Inventory Optimization
      - 全文卡：`/Users/lute/.dsh/skills/p2s-two-echelon-inventory-drl/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 29 张
    - `p2s-aim-rm-llm-inventory-mas-memory` · 源卡号 `Skill-AIM-RM-LLM-Inventory-MAS-Memory` · 10-MAS · AIM-RM — LLM 多 Agent 库存管理：历史经验相似匹配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-aim-rm-llm-inventory-mas-memory/SKILL.md`
    - `p2s-bonded-warehouse-inventory-intelligence` · 源卡号 `Skill-Bonded-Warehouse-Inventory-Intelligence` · 18-物流履约 · 保税仓智能库存 — 监管合规×资金效率双优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bonded-warehouse-inventory-intelligence/SKILL.md`
    - `p2s-cvar-inventory-risk-portfolio` · 源卡号 `Skill-CVaR-Inventory-Risk-Portfolio` · 04-供应链 · CVaR多SKU库存风险组合 — 金融条件风险价值迁移至库存尾部风险管理
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cvar-inventory-risk-portfolio/SKILL.md`
    - `p2s-causal-rl-decision-making` · 源卡号 `Skill-Causal-RL-Decision-Making` · 01-因果推断 · 因果强化学习 — 从相关驱动到因果驱动的决策优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-rl-decision-making/SKILL.md`
    - `p2s-conformal-risk-assessment` · 源卡号 `Skill-Conformal-Risk-Assessment` · 01-因果推断 · Conformal Risk Assessment — 共形预测业务风险量化：覆盖率保证的区间估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-conformal-risk-assessment/SKILL.md`
    - `p2s-dynamic-abc-stratification-adaptive-policy` · 源卡号 `Skill-Dynamic-ABC-Stratification-Adaptive-Policy` · 04-供应链 · 动态ABC分层与策略自适应 — 帕累托分类自动更新与差异化库存策略绑定
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-abc-stratification-adaptive-policy/SKILL.md`
    - `p2s-emsr-bid-price-inventory-control` · 源卡号 `Skill-EMSR-Bid-Price-Inventory-Control` · 17-价格优化 · EMSR-b Bid-Price Inventory Control — 酒店边际座位收益模型迁移到FBA库存动态定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-emsr-bid-price-inventory-control/SKILL.md`
    - `p2s-expiry-date-aging-baby-products-kpi` · 源卡号 `Skill-Expiry-Date-Aging-Baby-Products-KPI` · 04-供应链 · 母婴产品效期管理与临期品KPI — 保质期预警/临期库存占比/过期销毁成本管控
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-expiry-date-aging-baby-products-kpi/SKILL.md`
    - `p2s-fba-stranded-unfulfillable-inventory-kpi` · 源卡号 `Skill-FBA-Stranded-Unfulfillable-Inventory-KPI` · 04-供应链 · FBA滞销不可售库存KPI与处置策略 — 滞销率/仓储过长费预警/移除决策优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fba-stranded-unfulfillable-inventory-kpi/SKILL.md`
    - `p2s-gmroi-inventory-investment-efficiency` · 源卡号 `Skill-GMROI-Inventory-Investment-Efficiency` · 04-供应链 · GMROI库存资金投资回报优化 — 毛利润/平均库存比率最大化决策模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gmroi-inventory-investment-efficiency/SKILL.md`
    - `p2s-healthy-inventory-three-layer-kpi` · 源卡号 `Skill-Healthy-Inventory-Three-Layer-KPI` · 04-供应链 · 健康库存三层数字化KPI体系 — 可视层/分析层/应用层的量化指标与联通机制
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-healthy-inventory-three-layer-kpi/SKILL.md`
    - `p2s-ito-three-phase-health-tracking` · 源卡号 `Skill-ITO-Three-Phase-Health-Tracking` · 04-供应链 · ITO备货前中后三阶段健康度追踪 — 库存周转全周期过程KPI与干预决策闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ito-three-phase-health-tracking/SKILL.md`
    - `p2s-inventory-aging-cost-management` · 源卡号 `Skill-Inventory-Aging-Cost-Management` · 04-供应链 · 库龄分段管理与资金成本化 — 库存账龄结构诊断、持有成本精算与阶梯清仓触发
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-aging-cost-management/SKILL.md`
    - `p2s-inventory-carrying-cost-model` · 源卡号 `Skill-Inventory-Carrying-Cost-Model` · 23-运营财务 · Skill-Inventory-Carrying-Cost-Model — 库存持有成本模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-carrying-cost-model/SKILL.md`
    - `p2s-inventory-health-aging-attribution` · 源卡号 `Skill-Inventory-Health-Aging-Attribution` · 04-供应链 · Business Metric-Aware Forecasting for Inventory Management
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-health-aging-attribution/SKILL.md`
    - `p2s-inventory-positioning-multi-dc` · 源卡号 `Skill-Inventory-Positioning-Multi-DC` · 18-物流履约 · Inventory Positioning Multi-DC — 多仓库存定位优化（FBA+海外仓+自营仓协同）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-positioning-multi-dc/SKILL.md`
    - `p2s-inventory-turnover-abc-classification` · 源卡号 `Skill-Inventory-Turnover-ABC-Classification` · 04-供应链 · ABC动销率动态分层与差异化策略 — ABCDE五级动销管理与80/20库存结构优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-turnover-abc-classification/SKILL.md`
    - `p2s-llm-multi-dc-inventory` · 源卡号 `Skill-LLM-Multi-DC-Inventory` · 04-供应链 · LLM Multi-DC Inventory — LLM 驱动多仓库存再平衡与可解释优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-multi-dc-inventory/SKILL.md`
    - `p2s-long-tail-sku-clearance-optimization` · 源卡号 `Skill-Long-Tail-SKU-Clearance-Optimization` · 04-供应链 · 长尾SKU管理与滞销清仓优化 — 缺货率与长尾品双向治理算法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-long-tail-sku-clearance-optimization/SKILL.md`
    - `p2s-markdown-clearance-auto-trigger` · 源卡号 `Skill-Markdown-Clearance-Auto-Trigger` · 04-供应链 · Markdown Clearance Auto Trigger — 库龄超标且库存积压时自动触发降价清仓阶梯
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-markdown-clearance-auto-trigger/SKILL.md`
    - `p2s-markdown-schedule-auto-trigger` · 源卡号 `Skill-Markdown-Schedule-Auto-Trigger` · 04-供应链 · Markdown-Schedule-Auto-Trigger — Amazon FBA 滞销库存库龄触发三阶段自动降价序
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-markdown-schedule-auto-trigger/SKILL.md`
    - `p2s-model-calibration` · 源卡号 `Skill-Model-Calibration` · 12-ML基础 · Model Calibration — 让预测概率真正可信的校准技术
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-model-calibration/SKILL.md`
    - `p2s-multi-sku-copula-risk` · 源卡号 `Skill-Multi-SKU-Copula-Risk` · 04-供应链 · Multi-SKU Copula Risk — 多SKU库存风险联合建模：Copula 协动分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-sku-copula-risk/SKILL.md`
    - `p2s-purchase-sales-inventory-3d-tracking` · 源卡号 `Skill-Purchase-Sales-Inventory-3D-Tracking` · 04-供应链 · 进销存三维动态追踪 — 进货/销售/库存联动监控与比率预警体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-purchase-sales-inventory-3d-tracking/SKILL.md`
    - `p2s-sku-warehouse-footprint-monitoring` · 源卡号 `Skill-SKU-Warehouse-Footprint-Monitoring` · 04-供应链 · SKU级仓容占用实时监控 — 体积换算精细化测算与存山如山预警机制
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sku-warehouse-footprint-monitoring/SKILL.md`
    - `p2s-state-space-inventory-signal-smoothing` · 源卡号 `Skill-State-Space-Inventory-Signal-Smoothing` · 04-供应链 · 状态空间库存信号平滑 — FBA数据三层分解（趋势+季节+噪声）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-state-space-inventory-signal-smoothing/SKILL.md`
    - `p2s-supplier-lead-time-buffer` · 源卡号 `Skill-Supplier-Lead-Time-Buffer` · 18-物流履约 · Supplier Lead Time Buffer — 供应商交货期缓冲：非正态分布下的安全库存
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-lead-time-buffer/SKILL.md`
    - `p2s-supply-chain-kpi-health-dashboard` · 源卡号 `Skill-Supply-Chain-KPI-Health-Dashboard` · 04-供应链 · 全链路供应链KPI健康度仪表盘 — 三层KPI体系整合、健康评分与智能预警
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-kpi-health-dashboard/SKILL.md`
    - `p2s-warehouse-capacity-efficiency-planning` · 源卡号 `Skill-Warehouse-Capacity-Efficiency-Planning` · 04-供应链 · 仓容管理与仓储效率规划 — 仓容测算、仓储效率模拟与精细化5步运营
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-warehouse-capacity-efficiency-planning/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 31。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-031 · 履约异常

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-020` 仓储履约与退货处置 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-03, FLOW-05, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-031-履约异常.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：履约异常检测与订单风险评分可用分类模型加确定性规则叠加。
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
> `paper2skills-research/data/contracts/flow-03-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 8 张
    - `p2s-logistics-anomaly-fraud-signal` · 源卡号 `Skill-Logistics-Anomaly-Fraud-Signal` · 18-物流履约 · Logistics Anomaly Fraud Signal — 物流轨迹异常作为欺诈信号特征工程
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-anomaly-fraud-signal/SKILL.md`
    - `p2s-logistics-fraud-detection` · 源卡号 `Skill-Logistics-Fraud-Detection` · 18-物流履约 · Logistics Fraud Detection — 物流链路欺诈检测：虚假收货、刷单物流与地址篡改的识别与拦截
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-fraud-detection/SKILL.md`
    - `p2s-multicarrier-parcel-tracking-fusion` · 源卡号 `Skill-Multicarrier-Parcel-Tracking-Fusion` · 18-物流履约 · 多承运商包裹追踪融合 — 跨境物流可见性引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multicarrier-parcel-tracking-fusion/SKILL.md`
    - `p2s-order-accuracy-exception-rate-kpi` · 源卡号 `Skill-Order-Accuracy-Exception-Rate-KPI` · 04-供应链 · 订单准确率与异常处理KPI — 录单差错率/错发漏发率/订单异常闭环体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-order-accuracy-exception-rate-kpi/SKILL.md`
    - `p2s-order-cycle-time-otd-analytics` · 源卡号 `Skill-Order-Cycle-Time-OTD-Analytics` · 04-供应链 · 订单交付周期OTD全链路分解 — On-Time Delivery率/交付阶段拆解/延迟根因归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-order-cycle-time-otd-analytics/SKILL.md`
    - `p2s-order-fulfillment-rate-dispatch-timeliness` · 源卡号 `Skill-Order-Fulfillment-Rate-Dispatch-Timeliness` · 04-供应链 · 订单履约率与发货及时率 — 全链路订单从下单到签收的履约质量量化体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-order-fulfillment-rate-dispatch-timeliness/SKILL.md`
    - `p2s-proactive-customer-alert-supply-chain` · 源卡号 `Skill-Proactive-Customer-Alert-Supply-Chain` · 24-标签工程 · 主动客户预警供应链 — 基于在途延误Tag的客户主动通知与体验保护
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-proactive-customer-alert-supply-chain/SKILL.md`
    - `p2s-wms-exception-action-trigger` · 源卡号 `Skill-WMS-Exception-Action-Trigger` · 24-标签工程 · WMS异常Tag触发引擎 — 仓储操作异常实时检测、标签化与自动处置触发
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-wms-exception-action-trigger/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 8。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-044 · 组合设计

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-026` 定价促销与商品组合 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-044-组合设计.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：商品组合与捆绑是购物篮关联分析与组合定价优化问题。
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

- 有全文卡（优先选，可选到论文参数）：共 1 张
    - `p2s-iv-instrumental-variables` · 源卡号 `Skill-IV-Instrumental-Variables` · 01-因果推断 · Instrumental Variables (IV) for Causal Inference with Endoge
      - 全文卡：`/Users/lute/.dsh/skills/p2s-iv-instrumental-variables/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 15 张
    - `p2s-bundle-pricing-strategy` · 源卡号 `Skill-Bundle-Pricing-Strategy` · 17-价格优化 · Bundle Pricing Strategy（捆绑定价策略）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bundle-pricing-strategy/SKILL.md`
    - `p2s-bundle-recommendation-complementary` · 源卡号 `Skill-Bundle-Recommendation-Complementary` · 05-推荐系统 · Bundle Recommendation Complementary — 互补商品捆绑推荐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bundle-recommendation-complementary/SKILL.md`
    - `p2s-cross-sell-llm-gnn` · 源卡号 `Skill-Cross-Sell-LLM-GNN` · 06-增长模型 · 交叉销售LLM+GNN — 三阶段粗到精检索框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-sell-llm-gnn/SKILL.md`
    - `p2s-double-debiased-ml-price` · 源卡号 `Skill-Double-Debiased-ML-Price` · 01-因果推断 · 双重去偏机器学习价格弹性 — 高维混淆下的价格因果效应估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-double-debiased-ml-price/SKILL.md`
    - `p2s-dynamic-bundle-pricing` · 源卡号 `Skill-Dynamic-Bundle-Pricing` · 17-价格优化 · Dynamic Bundle Pricing — 动态捆绑定价：配套商品组合最优定价策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dynamic-bundle-pricing/SKILL.md`
    - `p2s-gnn-ecommerce-recommendation` · 源卡号 `Skill-GNN-Ecommerce-Recommendation` · 05-推荐系统 · GNN Ecommerce Recommendation — 图神经网络电商推荐：用户-商品图谱深度学习
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gnn-ecommerce-recommendation/SKILL.md`
    - `p2s-gnn-foundations` · 源卡号 `Skill-GNN-Foundations` · 08-知识图谱 · GNN Foundations（图神经网络基础）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gnn-foundations/SKILL.md`
    - `p2s-graph-attention-network-recommendation` · 源卡号 `Skill-Graph-Attention-Network-Recommendation` · 08-知识图谱 · Graph Attention Network Recommendation — 图注意力网络推荐：动态权重的高精度图推
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-graph-attention-network-recommendation/SKILL.md`
    - `p2s-kg-augmented-recommendation-colakg` · 源卡号 `Skill-KG-Augmented-Recommendation-CoLaKG` · 08-知识图谱 · 知识图谱增强推荐 - CoLaKG (LLM × KG)
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-augmented-recommendation-colakg/SKILL.md`
    - `p2s-knowledge-graph-rec` · 源卡号 `Skill-Knowledge-Graph-Rec` · 05-推荐系统 · Knowledge Graph Enhanced Recommendation — 知识图谱增强推荐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-knowledge-graph-rec/SKILL.md`
    - `p2s-mappo-gat-dynamic-pricing` · 源卡号 `Skill-MAPPO-GAT-Dynamic-Pricing` · 17-价格优化 · MAPPO+GAT多智能体图注意力动态定价 — 产品关系图驱动的多SKU协同价格优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mappo-gat-dynamic-pricing/SKILL.md`
    - `p2s-mas-dynamic-pricing-coalition` · 源卡号 `Skill-MAS-Dynamic-Pricing-Coalition` · 10-MAS · MAS多SKU定价联盟博弈 — 母婴品牌多SKU组合利润最大化联合定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-dynamic-pricing-coalition/SKILL.md`
    - `p2s-mas-pricing-coalition-stability` · 源卡号 `Skill-MAS-Pricing-Coalition-Stability` · 10-MAS · MAS-Pricing-Coalition-Stability — 多SKU联合定价纳什均衡检测与联合体稳定性维持
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-pricing-coalition-stability/SKILL.md`
    - `p2s-mental-accounting-bundle-psychology` · 源卡号 `Skill-Mental-Accounting-Bundle-Psychology` · 17-价格优化 · 心理账户捆绑定价心理学 — 识别同一心智账户商品组合使 AOV 提升22%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mental-accounting-bundle-psychology/SKILL.md`
    - `p2s-topological-data-analysis-cross-sell` · 源卡号 `Skill-Topological-Data-Analysis-Cross-Sell` · 05-推荐系统 · 拓扑数据分析 (TDA) 挖掘时空隐性关联销售路径
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-topological-data-analysis-cross-sell/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 2 张
    - `Skill-MAS-Multi-Objective-Recommendation` · 07-NLP-VOC · Skill-MAS-Multi-Objective-Recommendation
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-MAS-Multi-Objective-Recommendation.md`
    - `Skill-TJAP-跨市场品类组合定价` · 07-NLP-VOC · Skill-TJAP-跨市场品类组合定价
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-TJAP-跨市场品类组合定价.md`

> 候选总数 18。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-063 · 经济性分析

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-041` 经营财务与资金 |
| 域 / 面 | `DOM-07` / `PLN-MGT` |
| flows | FLOW-01, FLOW-02, FLOW-03, FLOW-07, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-063-经济性分析.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：单位经济、贡献毛利与盈亏平衡是结构化财务建模。
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

- 有全文卡（优先选，可选到论文参数）：共 2 张
    - `p2s-did-difference-in-differences` · 源卡号 `Skill-DiD-Difference-in-Differences` · 01-因果推断 · Difference-in-Differences (DiD) for Causal Effect Estimation
      - 全文卡：`/Users/lute/.dsh/skills/p2s-did-difference-in-differences/SKILL.md`
    - `p2s-promotion-effectiveness` · 源卡号 `Skill-Promotion-Effectiveness` · 15-营销投放分析 · Promotion Effectiveness Evaluation with Causal ML
      - 全文卡：`/Users/lute/.dsh/skills/p2s-promotion-effectiveness/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 55 张
    - `p2s-aigc-revenue-attribution` · 源卡号 `Skill-AIGC-Revenue-Attribution` · 11-AI人文 · AIGC Revenue Attribution — AI内容生成 ROI 财务归因：从内容投入到 GMV 的量化路径
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-aigc-revenue-attribution/SKILL.md`
    - `p2s-advertising-tacos-pnl-integration` · 源卡号 `Skill-Advertising-TACOS-PnL-Integration` · 23-运营财务 · Skill-Advertising-TACOS-PnL-Integration — 广告TACoS与P&L集成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-advertising-tacos-pnl-integration/SKILL.md`
    - `p2s-agent-finance-autopilot` · 源卡号 `Skill-Agent-Finance-Autopilot` · 23-运营财务 · Agent Finance Autopilot — LLM 多 Agent 财务自动化与 P&L 实时分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-finance-autopilot/SKILL.md`
    - `p2s-agent-roi-measurement-framework` · 源卡号 `Skill-Agent-ROI-Measurement-Framework` · 16-智能体工程 · Agent ROI 测量框架 — 量化 AI Agent 实际商业价值的三维评估体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-roi-measurement-framework/SKILL.md`
    - `p2s-agent-workforce-replacement-calculator` · 源卡号 `Skill-Agent-Workforce-Replacement-Calculator` · 16-智能体工程 · AI Agent 人力替代计算器 — 量化哪些运营岗位可被 Agent 替代及 ROI
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-workforce-replacement-calculator/SKILL.md`
    - `p2s-agentic-pnl-analyst` · 源卡号 `Skill-Agentic-PnL-Analyst` · 09-DataAgent-LLM · Agent 驱动的 P&L 归因分析 — SKU 级成本拆解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agentic-pnl-analyst/SKILL.md`
    - `p2s-amazon-lending-decision` · 源卡号 `Skill-Amazon-Lending-Decision` · 23-运营财务 · Amazon Lending Decision — 电商平台卖家信用评估与融资决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-amazon-lending-decision/SKILL.md`
    - `p2s-bom-cost-rollup-tag-engine` · 源卡号 `Skill-BOM-Cost-Rollup-Tag-Engine` · 04-供应链 · BOM成本卷积标签引擎 — 从原材料到制成品的全层级成本精算与Tag驱动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bom-cost-rollup-tag-engine/SKILL.md`
    - `p2s-cash-conversion-cycle-optimization` · 源卡号 `Skill-Cash-Conversion-Cycle-Optimization` · 23-运营财务 · Skill-Cash-Conversion-Cycle-Optimization — 现金转换周期优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cash-conversion-cycle-optimization/SKILL.md`
    - `p2s-causal-supply-chain-attribution` · 源卡号 `Skill-Causal-Supply-Chain-Attribution` · 04-供应链 · 供应链成本因果归因 — DAG 因果图拆解成本驱动因子
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-supply-chain-attribution/SKILL.md`
    - `p2s-churn-revenue-impact` · 源卡号 `Skill-Churn-Revenue-Impact` · 23-运营财务 · Churn Revenue Impact — 用户流失的财务损失量化与 P&L 影响分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-churn-revenue-impact/SKILL.md`
    - `p2s-commodity-futures-cost-baseline` · 源卡号 `Skill-Commodity-Futures-Cost-Baseline` · 04-供应链 · 大宗商品期货驱动的竞品成本底线穿透 (Commodity Futures Arbitrage)
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-commodity-futures-cost-baseline/SKILL.md`
    - `p2s-contribution-margin-by-channel` · 源卡号 `Skill-Contribution-Margin-By-Channel` · 23-运营财务 · Contribution Margin By Channel — 按渠道贡献毛利分析（多平台ROAS→毛利拆解）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-contribution-margin-by-channel/SKILL.md`
    - `p2s-cost-plus-dynamic-tariff-pricing` · 源卡号 `Skill-Cost-Plus-Dynamic-Tariff-Pricing` · 17-价格优化 · 成本加成+关税动态定价 — 关税波动下的自动调价模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cost-plus-dynamic-tariff-pricing/SKILL.md`
    - `p2s-fba-cost-forecast-adjustment` · 源卡号 `Skill-FBA-Cost-Forecast-Adjustment` · 23-运营财务 · FBA Cost Forecast Adjustment — 不对称惩罚驱动的履约成本最小化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fba-cost-forecast-adjustment/SKILL.md`
    - `p2s-fx-dynamic-pricing-adjustment` · 源卡号 `Skill-FX-Dynamic-Pricing-Adjustment` · 23-运营财务 · 汇率联动动态定价 — 保持目标毛利率的实时定价调整
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fx-dynamic-pricing-adjustment/SKILL.md`
    - `p2s-fx-exposure-measurement` · 源卡号 `Skill-FX-Exposure-Measurement` · 23-运营财务 · 外汇敞口测量 — 跨境电商货币风险定量分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fx-exposure-measurement/SKILL.md`
    - `p2s-fx-hedging-strategy` · 源卡号 `Skill-FX-Hedging-Strategy` · 23-运营财务 · FX Hedging Strategy — 跨境汇率风险对冲：动态套期保值降低外汇损失
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fx-hedging-strategy/SKILL.md`
    - `p2s-first-last-mile-cost-kpi-crossborder` · 源卡号 `Skill-First-Last-Mile-Cost-KPI-CrossBorder` · 04-供应链 · 跨境头程末程成本KPI与路线优化 — 头程运费率/末程成本率/跨境物流综合成本体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-first-last-mile-cost-kpi-crossborder/SKILL.md`
    - `p2s-forecast-to-pl-bridge` · 源卡号 `Skill-Forecast-to-PL-Bridge` · 23-运营财务 · Forecast-to-PL-Bridge — 需求预测误差的财务损失量化与成本优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-forecast-to-pl-bridge/SKILL.md`
    - `p2s-fraud-pl-impact` · 源卡号 `Skill-Fraud-PL-Impact` · 23-运营财务 · Fraud PL Impact — 电商欺诈的财务损失量化与检测成本权衡
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fraud-pl-impact/SKILL.md`
    - `p2s-gmroi-inventory-investment-efficiency` · 源卡号 `Skill-GMROI-Inventory-Investment-Efficiency` · 04-供应链 · GMROI库存资金投资回报优化 — 毛利润/平均库存比率最大化决策模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gmroi-inventory-investment-efficiency/SKILL.md`
    - `p2s-inventory-carrying-cost-model` · 源卡号 `Skill-Inventory-Carrying-Cost-Model` · 23-运营财务 · Skill-Inventory-Carrying-Cost-Model — 库存持有成本模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-carrying-cost-model/SKILL.md`
    - `p2s-inventory-financing-optimization` · 源卡号 `Skill-Inventory-Financing-Optimization` · 23-运营财务 · Inventory Financing Optimization — 库存融资与供应链金融决策优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-financing-optimization/SKILL.md`
    - `p2s-kg-supply-chain-cost-attribution` · 源卡号 `Skill-KG-Supply-Chain-Cost-Attribution` · 08-知识图谱 · KG Supply Chain Cost Attribution — 图神经网络 + 因果推断的供应链成本归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-supply-chain-cost-attribution/SKILL.md`
    - `p2s-llm-financial-report-analyst` · 源卡号 `Skill-LLM-Financial-Report-Analyst` · 09-DataAgent-LLM · LLM Financial Report Analyst — 迭代精化 + 代码验证的智能财务报告解读
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-financial-report-analyst/SKILL.md`
    - `p2s-logistics-cost-model` · 源卡号 `Skill-Logistics-Cost-Model` · 23-运营财务 · Logistics Cost Model — 跨境物流全链路成本建模与关税不确定性优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-cost-model/SKILL.md`
    - `p2s-logistics-cost-structure-decomposition` · 源卡号 `Skill-Logistics-Cost-Structure-Decomposition` · 04-供应链 · 全链路物流成本结构分解 — 进存销三段成本拆解与降本杠杆识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-cost-structure-decomposition/SKILL.md`
    - `p2s-mmm-budget-pl-alignment` · 源卡号 `Skill-MMM-Budget-PL-Alignment` · 15-营销投放分析 · MMM Budget PL Alignment — 营销预算分配与利润约束下的 ROI 优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mmm-budget-pl-alignment/SKILL.md`
    - `p2s-membership-tier-design-optimization` · 源卡号 `Skill-Membership-Tier-Design-Optimization` · 06-增长模型 · Membership Tier Design Optimization — 多层会员体系结构的最优设计与 CLV 最大化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-membership-tier-design-optimization/SKILL.md`
    - `p2s-montecarlo-tariff-risk` · 源卡号 `Skill-MonteCarlo-Tariff-Risk` · 21-合规决策 · 蒙特卡洛地缘政治尾部风险量化 (Monte Carlo Tariff Risk)
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-montecarlo-tariff-risk/SKILL.md`
    - `p2s-multi-step-reasoning-bi` · 源卡号 `Skill-Multi-Step-Reasoning-BI` · 09-DataAgent-LLM · 多步推理BI分析 — LLM链式推理自动生成财务归因报告
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-step-reasoning-bi/SKILL.md`
    - `p2s-multicurrency-fx-hedging` · 源卡号 `Skill-Multicurrency-FX-Hedging` · 23-运营财务 · Multicurrency FX Hedging — 跨境卖家多货币外汇风险对冲
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multicurrency-fx-hedging/SKILL.md`
    - `p2s-operating-cash-flow-forecast` · 源卡号 `Skill-Operating-Cash-Flow-Forecast` · 23-运营财务 · Operating Cash Flow Forecast — 需求预测驱动的运营现金流预测与库存融资优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-operating-cash-flow-forecast/SKILL.md`
    - `p2s-pl-attribution-analysis` · 源卡号 `Skill-PL-Attribution-Analysis` · 23-运营财务 · P&L Attribution Analysis（SKU 级损益归因分析）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-pl-attribution-analysis/SKILL.md`
    - `p2s-points-expiry-redemption-liability-model` · 源卡号 `Skill-Points-Expiry-Redemption-Liability-Model` · 06-增长模型 · Points Expiry Redemption Liability Model — 积分过期负债精算与兑换率动态定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-points-expiry-redemption-liability-model/SKILL.md`
    - `p2s-procurement-cost-kpi-price-achievement` · 源卡号 `Skill-Procurement-Cost-KPI-Price-Achievement` · 04-供应链 · 采购价格达成率与降本KPI体系 — 全链路降本量化追踪与价格偏差归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-procurement-cost-kpi-price-achievement/SKILL.md`
    - `p2s-profitability-waterfall-by-asin` · 源卡号 `Skill-Profitability-Waterfall-By-ASIN` · 23-运营财务 · Skill-Profitability-Waterfall-By-ASIN — 单品盈利瀑布分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-profitability-waterfall-by-asin/SKILL.md`
    - `p2s-promo-roi-attribution-supply-side` · 源卡号 `Skill-Promo-ROI-Attribution-Supply-Side` · 04-供应链 · 促销活动供应侧ROI归因 — 备货成本+促销库存持有成本+尾货损失的全成本核算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promo-roi-attribution-supply-side/SKILL.md`
    - `p2s-promotion-roi-pre-post-analysis` · 源卡号 `Skill-Promotion-ROI-Pre-Post-Analysis` · 23-运营财务 · Promotion ROI Pre-Post Analysis — 促销ROI前后对比分析（大促PnL归因）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-promotion-roi-pre-post-analysis/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 1 张
    - `Skill-Shipping-Cost-Estimation` · 03-时间序列 · Skill-Shipping-Cost-Estimation
      - 全文卡：`paper2skills-vault/03-时间序列/Skill-Shipping-Cost-Estimation.md`

> 候选总数 58。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-024 · 关务资料检查

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-019` 跨境物流与关务 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-03, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-024-关务资料检查.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（⚠️ 来源＝本项目《经营侧组织模型精确抽取》`reports/_survey_org_model.md` §F.3 的判定，**二手来源，不是材料原文**，别改判）**：单证字段一致性可规则+OCR校验，但归类与合规判定须法定依据与外部确认。
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
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 15 张
    - `p2s-atlas-hts-tariff-classification` · 源卡号 `Skill-ATLAS-HTS-Tariff-Classification` · 04-供应链 · ATLAS HTS Tariff Classification — LLM 驱动跨境 HS 关税编码自动分类
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-atlas-hts-tariff-classification/SKILL.md`
    - `p2s-bonded-warehouse-inventory-intelligence` · 源卡号 `Skill-Bonded-Warehouse-Inventory-Intelligence` · 18-物流履约 · 保税仓智能库存 — 监管合规×资金效率双优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bonded-warehouse-inventory-intelligence/SKILL.md`
    - `p2s-bonded-zone-compliance-auto` · 源卡号 `Skill-Bonded-Zone-Compliance-Auto` · 18-物流履约 · Bonded Zone Compliance Auto — 保税区合规自动化（保税仓入出区监管）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bonded-zone-compliance-auto/SKILL.md`
    - `p2s-cpsc-efiling-auto-mapper` · 源卡号 `Skill-CPSC-eFiling-Auto-Mapper` · 21-合规决策 · CPSC eFiling Auto-Mapper — NLP驱动的电子申报字段自动填充
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cpsc-efiling-auto-mapper/SKILL.md`
    - `p2s-cross-border-tax-tariff-modeling` · 源卡号 `Skill-Cross-Border-Tax-Tariff-Modeling` · 23-运营财务 · HTS Tariff Intelligence — LLM 驱动的跨境关税分类与节税优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-tax-tariff-modeling/SKILL.md`
    - `p2s-crossborder-customs-compliance-rate-kpi` · 源卡号 `Skill-CrossBorder-Customs-Compliance-Rate-KPI` · 04-供应链 · 跨境关检务合规率KPI体系 — 清关时效/合规申报率/风险等级分类的全流程量化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-crossborder-customs-compliance-rate-kpi/SKILL.md`
    - `p2s-customs-clearance-risk-scoring` · 源卡号 `Skill-Customs-Clearance-Risk-Scoring` · 18-物流履约 · Customs Clearance Risk Scoring — 跨境清关多维风险预警
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-customs-clearance-risk-scoring/SKILL.md`
    - `p2s-dangerous-goods-dg-classification` · 源卡号 `Skill-Dangerous-Goods-DG-Classification` · 21-合规决策 · Dangerous Goods Classification — 危险品自动分类（锂电池/液体/气溶胶跨境合规）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dangerous-goods-dg-classification/SKILL.md`
    - `p2s-gcc-cpc-document-validator` · 源卡号 `Skill-GCC-CPC-Document-Validator` · 21-合规决策 · GCC/CPC Document Validator — 合规认证文档完整性自动验证
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gcc-cpc-document-validator/SKILL.md`
    - `p2s-green-supply-chain-carbon-footprint` · 源卡号 `Skill-Green-Supply-Chain-Carbon-Footprint` · 04-供应链 · Green Supply Chain Carbon Footprint — 绿色供应链碳足迹：ESG合规与可持续运营优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-green-supply-chain-carbon-footprint/SKILL.md`
    - `p2s-hs-code-auto-classification` · 源卡号 `Skill-HS-Code-Auto-Classification` · 18-物流履约 · HS编码自动分类 — 跨境关税智能核算引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hs-code-auto-classification/SKILL.md`
    - `p2s-hts-agentic-tariff-classification` · 源卡号 `Skill-HTS-Agentic-Tariff-Classification` · 21-合规决策 · HTS多Agent关税编码分类 — 共识验证+层级推理+不确定性感知的跨境清关自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hts-agentic-tariff-classification/SKILL.md`
    - `p2s-hts-tariff-classification` · 源卡号 `Skill-HTS-Tariff-Classification` · 21-合规决策 · HTS 关税编码分类与优化（跨境电商关税合规）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hts-tariff-classification/SKILL.md`
    - `p2s-kg-logistics-intelligence` · 源卡号 `Skill-KG-Logistics-Intelligence` · 08-知识图谱 · 知识图谱物流智能 — 供应链实体关系图驱动的物流决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-logistics-intelligence/SKILL.md`
    - `p2s-new-sku-launch-readiness-gate` · 源卡号 `Skill-New-SKU-Launch-Readiness-Gate` · 04-供应链 · 新品上市准入门控 — 从选品到发布的全量检查清单与Tag驱动的上市就绪评估
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-new-sku-launch-readiness-gate/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 15。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。

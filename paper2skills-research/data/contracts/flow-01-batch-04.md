# FLOW-01 契约撰写 · 批次 04（9 份）

> 公共材料见 `flow-01-common.md`（**先读它**）。本批 7 份 A 模板 / 2 份 B 模板。


---

## CTR-A-005 · 经营预测

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-003` 经营分析与决策支持 |
| 域 / 面 | `DOM-01` / `PLN-MGT` |
| flows | FLOW-01, FLOW-06, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-005-经营预测.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：销量/收入/现金的时序与层级预测是标准预测问题，可直接输出预测值与区间。


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

- 有全文卡（优先选，可选到论文参数）：共 1 张
    - `p2s-deepanalyze-autonomous-data-science-agent` · 源卡号 `Skill-DeepAnalyze-Autonomous-Data-Science-Agent` · 09-DataAgent-LLM · DeepAnalyze — 自主数据科学Agent
      - 全文卡：`/Users/lute/.dsh/skills/p2s-deepanalyze-autonomous-data-science-agent/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 8 张
    - `p2s-aigp-llm-dynamic-pricing` · 源卡号 `Skill-AIGP-LLM-Dynamic-Pricing` · 17-价格优化 · AIGP — LLM 动态定价：长期 GMV 对齐框架（+13% GMV A/B实测）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-aigp-llm-dynamic-pricing/SKILL.md`
    - `p2s-brand-penetration-modeling` · 源卡号 `Skill-Brand-Penetration-Modeling` · 15-营销投放分析 · Brand Penetration Modeling — 品牌渗透率建模：Bass 扩散 × 跨市场品牌增长预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-brand-penetration-modeling/SKILL.md`
    - `p2s-budget-reforecast-rolling` · 源卡号 `Skill-Budget-Reforecast-Rolling` · 23-运营财务 · Budget Reforecast Rolling — 滚动预算再预测（月度滚动reforecast自动化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-budget-reforecast-rolling/SKILL.md`
    - `p2s-holiday-spike-demand-decomposition` · 源卡号 `Skill-Holiday-Spike-Demand-Decomposition` · 03-时间序列 · Holiday Spike Demand Decomposition — 节假日需求峰值分解（Prime Day/黑五）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-holiday-spike-demand-decomposition/SKILL.md`
    - `p2s-llm-business-intelligence-reasoning` · 源卡号 `Skill-LLM-Business-Intelligence-Reasoning` · 09-DataAgent-LLM · LLM Business Intelligence Reasoning — LLM 商业智能推理：从数据到决策的 CoT
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-business-intelligence-reasoning/SKILL.md`
    - `p2s-llm-ecommerce-report-automation` · 源卡号 `Skill-LLM-Ecommerce-Report-Automation` · 09-DataAgent-LLM · LLM 驱动电商经营报表自动生成 — 日报/周报零代码落地
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-ecommerce-report-automation/SKILL.md`
    - `p2s-market-size-estimation` · 源卡号 `Skill-Market-Size-Estimation` · 06-增长模型 · Skill-Market-Size-Estimation
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-market-size-estimation/SKILL.md`
    - `p2s-user-ltv-financial-bridge` · 源卡号 `Skill-User-LTV-Financial-Bridge` · 14-用户分析 · User LTV Financial Bridge — 用户生命周期价值预测驱动财务规划
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-user-ltv-financial-bridge/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 9。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-026 · 生命周期分析

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-017` 库存与商品生命周期 |
| 域 / 面 | `DOM-03` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-026-生命周期分析.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：产品生命周期阶段可用生长曲线/变点/生存分析识别并驱动清货时点。


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
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 6 张
    - `p2s-cross-border-returns-cost-model` · 源卡号 `Skill-Cross-Border-Returns-Cost-Model` · 18-物流履约 · Cross-Border Returns Cost Model — 跨境退货成本建模
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-returns-cost-model/SKILL.md`
    - `p2s-expiry-date-aging-baby-products-kpi` · 源卡号 `Skill-Expiry-Date-Aging-Baby-Products-KPI` · 04-供应链 · 母婴产品效期管理与临期品KPI — 保质期预警/临期库存占比/过期销毁成本管控
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-expiry-date-aging-baby-products-kpi/SKILL.md`
    - `p2s-fba-fee-intelligence` · 源卡号 `Skill-FBA-Fee-Intelligence` · 23-运营财务 · FBA Fee Intelligence（FBA 费用结构分析与长库龄预警）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fba-fee-intelligence/SKILL.md`
    - `p2s-inventory-write-down-impairment` · 源卡号 `Skill-Inventory-Write-Down-Impairment` · 23-运营财务 · Inventory Write-Down Impairment — 库存减值准备计算（滞销品会计处理+税务）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-write-down-impairment/SKILL.md`
    - `p2s-logistics-cost-pl-attribution` · 源卡号 `Skill-Logistics-Cost-PL-Attribution` · 18-物流履约 · Logistics Cost PL Attribution — 物流成本 P&L 归因：每单头程+FBA+退货的利润拆解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-logistics-cost-pl-attribution/SKILL.md`
    - `p2s-product-lifecycle-stage` · 源卡号 `Skill-Product-Lifecycle-Stage` · 06-增长模型 · Skill-Product-Lifecycle-Stage
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-lifecycle-stage/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 6。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-037 · Listing优化

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-022` Amazon商品与搜索运营 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-037-Listing优化.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：标题/要点优化可用排序学习与A/B实验驱动（关键词覆盖与CTR模型）。


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
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 43 张
    - `p2s-a-plus-content-template-engine` · 源卡号 `Skill-A-Plus-Content-Template-Engine` · 20-AI视频生成 · A+Content模板引擎 — VOC驱动的A+内容自动排版生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-a-plus-content-template-engine/SKILL.md`
    - `p2s-a-plus-content-video-embedding` · 源卡号 `Skill-A-Plus-Content-Video-Embedding` · 20-AI视频生成 · Skill-A-Plus-Content-Video-Embedding — A+ 内容视频嵌入转化率优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-a-plus-content-video-embedding/SKILL.md`
    - `p2s-amazon-a10-algorithm-ranking` · 源卡号 `Skill-Amazon-A10-Algorithm-Ranking` · 13-广告分析 · Amazon A10 Algorithm Ranking — 亚马逊搜索排名因子建模与 Listing 可见度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-amazon-a10-algorithm-ranking/SKILL.md`
    - `p2s-amazon-external-traffic-boost` · 源卡号 `Skill-Amazon-External-Traffic-Boost` · 13-广告分析 · Amazon External Traffic Boost — 站外流量对 A10 排名速度的提升效应建模
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-amazon-external-traffic-boost/SKILL.md`
    - `p2s-autopkg-multimodal-product-attribute-kg` · 源卡号 `Skill-AutoPKG-Multimodal-Product-Attribute-KG` · 08-知识图谱 · AutoPKG — 多模态产品属性图谱自动构建：文本+图片→GMV提升
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-autopkg-multimodal-product-attribute-kg/SKILL.md`
    - `p2s-autoqual-review-quality-assessment` · 源卡号 `Skill-AutoQual-Review-Quality-Assessment` · 14-用户分析 · AutoQual Review Quality Assessment — LLM Agent 自动化评论质量评估
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-autoqual-review-quality-assessment/SKILL.md`
    - `p2s-baby-age-stage-demand-segmentation` · 源卡号 `Skill-Baby-Age-Stage-Demand-Segmentation` · 07-NLP-VOC · Skill-Baby-Age-Stage-Demand-Segmentation — 婴儿月龄需求分层
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-baby-age-stage-demand-segmentation/SKILL.md`
    - `p2s-click-through-rate-title-optimizer` · 源卡号 `Skill-Click-Through-Rate-Title-Optimizer` · 25-搜索流量工程 · Skill-Click-Through-Rate-Title-Optimizer — 标题 CTR 机器学习优化器
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-click-through-rate-title-optimizer/SKILL.md`
    - `p2s-competitor-keyword-gap-analysis` · 源卡号 `Skill-Competitor-Keyword-Gap-Analysis` · 25-搜索流量工程 · Skill-Competitor-Keyword-Gap-Analysis — 竞品关键词缺口分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-competitor-keyword-gap-analysis/SKILL.md`
    - `p2s-continuous-nlp-seo-morphing` · 源卡号 `Skill-Continuous-NLP-SEO-Morphing` · 22-数据采集工程 · 连续语义漂移 SEO 文本对抗变异
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-continuous-nlp-seo-morphing/SKILL.md`
    - `p2s-cross-cultural-marketing-adaptation` · 源卡号 `Skill-Cross-Cultural-Marketing-Adaptation` · 11-AI人文 · Cross-Cultural Marketing Adaptation — 多语言 CAM 嵌入驱动的跨文化营销适配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-cultural-marketing-adaptation/SKILL.md`
    - `p2s-cross-platform-listing-sync-optimizer` · 源卡号 `Skill-Cross-Platform-Listing-Sync-Optimizer` · 13-广告分析 · 多平台 Listing 差异化同步优化器 — Amazon / TikTok Shop / Shopee 参数矩阵建模
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-platform-listing-sync-optimizer/SKILL.md`
    - `p2s-gan-red-team-listing` · 源卡号 `Skill-GAN-Red-Team-Listing` · 16-智能体工程 · GAN 红队驱动的 Listing 上线前免疫接种 (Adversarial Listing Defense)
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gan-red-team-listing/SKILL.md`
    - `p2s-hierarchical-product-kg-construction` · 源卡号 `Skill-Hierarchical-Product-KG-Construction` · 08-知识图谱 · 层级商品知识图谱自动构建（图片→KG）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hierarchical-product-kg-construction/SKILL.md`
    - `p2s-llm-search-query-expansion` · 源卡号 `Skill-LLM-Search-Query-Expansion` · 25-搜索流量工程 · LLM搜索词语义扩展 — 用大模型覆盖长尾搜索意图提升Listing覆盖面
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-llm-search-query-expansion/SKILL.md`
    - `p2s-listing-ab-testing-automation` · 源卡号 `Skill-Listing-AB-Testing-Automation` · 13-广告分析 · Listing AB Testing Automation — LLM Agent 驱动的 Listing A/B 测试
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-ab-testing-automation/SKILL.md`
    - `p2s-listing-ai-copywriting` · 源卡号 `Skill-Listing-AI-Copywriting` · 13-广告分析 · Amazon Listing 文案 AI 生成（标题+Bullet+描述全套）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-ai-copywriting/SKILL.md`
    - `p2s-listing-compliance-auto-repair` · 源卡号 `Skill-Listing-Compliance-Auto-Repair` · 21-合规决策 · Listing Compliance Auto Repair — AI 驱动违规 Listing 自动修复
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-compliance-auto-repair/SKILL.md`
    - `p2s-listing-conversion-rate-optimizer` · 源卡号 `Skill-Listing-Conversion-Rate-Optimizer` · 25-搜索流量工程 · Skill-Listing-Conversion-Rate-Optimizer — Listing 转化率 A/B 测试
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-conversion-rate-optimizer/SKILL.md`
    - `p2s-listing-health-diagnostic` · 源卡号 `Skill-Listing-Health-Diagnostic` · 13-广告分析 · Listing Health Diagnostic — Listing 快诊：多维度评分驱动的上架质量诊断
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-health-diagnostic/SKILL.md`
    - `p2s-listing-quality-scoring` · 源卡号 `Skill-Listing-Quality-Scoring` · 13-广告分析 · Skill-Listing-Quality-Scoring
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-quality-scoring/SKILL.md`
    - `p2s-listing-semantic-relevance-scoring` · 源卡号 `Skill-Listing-Semantic-Relevance-Scoring` · 25-搜索流量工程 · Listing 语义相关性评分 — 用 Sentence-BERT 诊断 SEO 缺口
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-listing-semantic-relevance-scoring/SKILL.md`
    - `p2s-long-tail-keyword-mining` · 源卡号 `Skill-Long-Tail-Keyword-Mining` · 25-搜索流量工程 · 长尾关键词挖掘 — NLP词频+竞品反查+搜索建议词三路融合
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-long-tail-keyword-mining/SKILL.md`
    - `p2s-long-tail-search-embedding-seo` · 源卡号 `Skill-Long-Tail-Search-Embedding-SEO` · 13-广告分析 · Long Tail Search Embedding SEO — 双塔嵌入模型驱动的电商长尾搜索词排名优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-long-tail-search-embedding-seo/SKILL.md`
    - `p2s-mas-search-optimization` · 源卡号 `Skill-MAS-Search-Optimization` · 10-MAS · MAS驱动搜索流量优化 — 多智能体协同的A9算法自动化运营
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-search-optimization/SKILL.md`
    - `p2s-multilingual-listing-generation` · 源卡号 `Skill-Multilingual-Listing-Generation` · 20-AI视频生成 · LLM多语言Listing批量生成 — 1小时完成5语言新品入驻
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-listing-generation/SKILL.md`
    - `p2s-multilingual-listing-localization` · 源卡号 `Skill-Multilingual-Listing-Localization` · 13-广告分析 · Multilingual Listing Localization — LLM 驱动的多语言 Listing 本地化与文
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multilingual-listing-localization/SKILL.md`
    - `p2s-multimodal-product-understanding` · 源卡号 `Skill-Multimodal-Product-Understanding` · 08-知识图谱 · Multimodal Product Understanding — 多模态商品理解：图文统一表示驱动搜索与推荐
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multimodal-product-understanding/SKILL.md`
    - `p2s-nlp-copy-ab-test-optimizer` · 源卡号 `Skill-NLP-Copy-AB-Test-Optimizer` · 02-A_B实验 · NLP Copy AB Test Optimizer — CTR 加权偏好优化 + 遗传算法的文案 A/B 测试闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-nlp-copy-ab-test-optimizer/SKILL.md`
    - `p2s-product-3d-visualization-ar-preview` · 源卡号 `Skill-Product-3D-Visualization-AR-Preview` · 20-AI视频生成 · Product-3D-Visualization-AR-Preview — 产品 3D 可视化与 AR 预览降低退货率
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-3d-visualization-ar-preview/SKILL.md`
    - `p2s-product-attribute-completion` · 源卡号 `Skill-Product-Attribute-Completion` · 22-数据采集工程 · Product Attribute Completion — 商品属性自动补全：AI 填补 Listing 属性空白
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-attribute-completion/SKILL.md`
    - `p2s-product-image-quality-assessment` · 源卡号 `Skill-Product-Image-Quality-Assessment` · 20-AI视频生成 · CLIP商品主图质量评分 — 上新门控自动化评估引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-image-quality-assessment/SKILL.md`
    - `p2s-review-helpfulness-prediction` · 源卡号 `Skill-Review-Helpfulness-Prediction` · 07-NLP-VOC · Review Helpfulness Prediction — 评论有用性预测：识别高说服力评论提升转化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-review-helpfulness-prediction/SKILL.md`
    - `p2s-review-helpfulness-ranking-model` · 源卡号 `Skill-Review-Helpfulness-Ranking-Model` · 07-NLP-VOC · Skill-Review-Helpfulness-Ranking-Model — 评论有用性排序模型
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-review-helpfulness-ranking-model/SKILL.md`
    - `p2s-review-keyword-mining-seo` · 源卡号 `Skill-Review-Keyword-Mining-SEO` · 25-搜索流量工程 · Skill-Review-Keyword-Mining-SEO — 评论关键词挖掘 SEO
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-review-keyword-mining-seo/SKILL.md`
    - `p2s-reward-model-rlhf-ecommerce` · 源卡号 `Skill-Reward-Model-RLHF-Ecommerce` · 12-ML基础 · RLHF奖励模型电商 — 人类偏好对齐的电商AI质量提升
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-reward-model-rlhf-ecommerce/SKILL.md`
    - `p2s-seo-organic-ranking-optimization` · 源卡号 `Skill-SEO-Organic-Ranking-Optimization` · 13-广告分析 · SEO Organic Ranking Optimization — 电商 SEO 自然排名元数据优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-seo-organic-ranking-optimization/SKILL.md`
    - `p2s-search-compliance-guard` · 源卡号 `Skill-Search-Compliance-Guard` · 25-搜索流量工程 · 搜索词合规预扫描 — Amazon TOS + FDA 双轨违禁词实时过滤
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-compliance-guard/SKILL.md`
    - `p2s-search-tag-keyword-auto-mapping` · 源卡号 `Skill-Search-Tag-Keyword-Auto-Mapping` · 25-搜索流量工程 · 标签→关键词自动映射矩阵 — SKU 标签体系驱动搜索词包生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-tag-keyword-auto-mapping/SKILL.md`
    - `p2s-seasonal-keyword-rotation-strategy` · 源卡号 `Skill-Seasonal-Keyword-Rotation-Strategy` · 25-搜索流量工程 · Skill-Seasonal-Keyword-Rotation-Strategy — 季节性关键词轮换策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-seasonal-keyword-rotation-strategy/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 2 张
    - `Skill-Agentic-Catalog-Enrichment` · 00-电商Agent · Skill-Agentic-Catalog-Enrichment
      - 全文卡：`paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md`
    - `Skill-LLM-Personalized-Marketing-Copy-Generation` · 07-NLP-VOC · Skill-LLM-Personalized-Marketing-Copy-Generation
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-LLM-Personalized-Marketing-Copy-Generation.md`

> 候选总数 45。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
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

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：促销排期与力度是带约束的因果增量优化（含蚕食校正与预算分配）。


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

## CTR-A-050 · 预算分配

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-032` 效果广告投放 |
| 域 / 面 | `DOM-05` / `PLN-OPS` |
| flows | FLOW-01 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-050-预算分配.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：跨渠道预算分配是约束优化/bandit与边际ROI均衡问题。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-01/STG-04 | M | 证据收集与诊断 | 核对同口径订单/GMV、可售/在途库存、价格、促销、广告、供应与账号健康 | 经营证据与偏差诊断 |
| FLOW-01/STG-05 | M | 方案与标准产物 | 比较价格、广告、库存与停止条件，形成联合经营行动包 | 联合经营行动包 |
| FLOW-01/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐对象回执、GMV、取消退款、库存、现金与增量结果 | 经营关闭记录 |
| FLOW-01/STG-01 | D | 信号接收 | 接收日常经营节奏或销量、转化、库存、广告异常 | 经营信号记录 |
| FLOW-01/STG-02 | R | 范围确定与Case创建 | 确定渠道、市场、账号、商品集合、时间窗、GMV口径与完成条件 | 经营Case Charter |
| FLOW-01/STG-03 | D | 上下文装配 | 按渠道选择主岗位，装配GMV、库存、价格、广告、财务和增量评估能力 | FLOW-01 Context Manifest |
| FLOW-01/STG-06 | R | Assurance接收门禁 | 检查数据质量、库存、预算、现金、账号范围和增量判断 | FLOW-01 Assurance Decision |
| FLOW-01/STG-07 | D | 受控动作 | 提交受策略约束的价格/广告Action Intent，或形成NoActionRecord | 动作尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-01-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 3 张
    - `p2s-intelligent-attribution-causal-forest` · 源卡号 `Skill-Intelligent-Attribution-Causal-Forest` · 01-因果推断 · 智能归因 - 因果森林 (Causal Forest)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-intelligent-attribution-causal-forest/SKILL.md`
    - `p2s-marketing-mix-modeling` · 源卡号 `Skill-Marketing-Mix-Modeling` · 15-营销投放分析 · Marketing Mix Modeling (MMM) for Macro Budget Allocation
      - 全文卡：`/Users/lute/.dsh/skills/p2s-marketing-mix-modeling/SKILL.md`
    - `p2s-roas-budget-optimization` · 源卡号 `Skill-ROAS-Budget-Optimization` · 13-广告分析 · ROAS Optimization and Ad Budget Allocation
      - 全文卡：`/Users/lute/.dsh/skills/p2s-roas-budget-optimization/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 48 张
    - `p2s-ad-spend-inventory-sync` · 源卡号 `Skill-Ad-Spend-Inventory-Sync` · 13-广告分析 · Ad Spend Inventory Sync — 广告投放与库存联动的协同优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ad-spend-inventory-sync/SKILL.md`
    - `p2s-auction-theory-advertising-bidding` · 源卡号 `Skill-Auction-Theory-Advertising-Bidding` · 13-广告分析 · 拍卖理论广告竞价优化 — GSP 拍卖机制下的最优出价策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-auction-theory-advertising-bidding/SKILL.md`
    - `p2s-autobidding-budget-allocation-optimization` · 源卡号 `Skill-Autobidding-Budget-Allocation-Optimization` · 13-广告分析 · 自动化竞价与预算动态分配 — 跨平台广告Autobidding与ROI约束下的预算优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-autobidding-budget-allocation-optimization/SKILL.md`
    - `p2s-bccb-causal-bandits` · 源卡号 `Skill-BCCB-Causal-Bandits` · 02-A_B实验 · 预算约束因果Bandit - 新渠道从Day1开始的转化率估计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bccb-causal-bandits/SKILL.md`
    - `p2s-bayesian-mmm-action-plan-generator` · 源卡号 `Skill-Bayesian-MMM-Action-Plan-Generator` · 15-营销投放分析 · Bayesian MMM Action Plan Generator — 贝叶斯后验分布生成保守/中性/激进三版预算方案
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bayesian-mmm-action-plan-generator/SKILL.md`
    - `p2s-bayesian-mmm-scenario-action-plan` · 源卡号 `Skill-Bayesian-MMM-Scenario-Action-Plan` · 15-营销投放分析 · Bayesian-MMM-Scenario-Action-Plan — 贝叶斯MMM后验驱动Q+1季度预算三情景决策方案
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-bayesian-mmm-scenario-action-plan/SKILL.md`
    - `p2s-brand-defense-search-strategy` · 源卡号 `Skill-Brand-Defense-Search-Strategy` · 25-搜索流量工程 · 品牌词防守策略 — 品牌词投放防竞品截流与自然排名协同
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-brand-defense-search-strategy/SKILL.md`
    - `p2s-channel-budget-reallocation-trigger` · 源卡号 `Skill-Channel-Budget-Reallocation-Trigger` · 15-营销投放分析 · Channel Budget Reallocation Trigger — 饱和度超阈值时自动削减并重分配渠道预算
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-channel-budget-reallocation-trigger/SKILL.md`
    - `p2s-channel-saturation-curve` · 源卡号 `Skill-Channel-Saturation-Curve` · 15-营销投放分析 · Channel Saturation Curve（渠道饱和曲线建模）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-channel-saturation-curve/SKILL.md`
    - `p2s-combo-ad-roi-maximizer` · 源卡号 `Skill-Combo-Ad-ROI-Maximizer` · 15-营销投放分析 · 广告ROI最大化 Combo Pattern — 从归因分析到预算自动重分配的 6 步优化链路
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-combo-ad-roi-maximizer/SKILL.md`
    - `p2s-constrained-multi-objective-ad-delivery` · 源卡号 `Skill-Constrained-Multi-Objective-Ad-Delivery` · 13-广告分析 · Constrained Multi-Objective Ad Delivery — 约束多目标广告投放同时满足 ROAS
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-constrained-multi-objective-ad-delivery/SKILL.md`
    - `p2s-content-roi-budget-shift-trigger` · 源卡号 `Skill-Content-ROI-Budget-Shift-Trigger` · 13-广告分析 · Content ROI Budget Shift Trigger — 内容ROI连续低于目标时自动削减并转向高ROI内容
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-content-roi-budget-shift-trigger/SKILL.md`
    - `p2s-counterfactual-ad-attribution-debiasing` · 源卡号 `Skill-Counterfactual-Ad-Attribution-Debiasing` · 01-因果推断 · Counterfactual Ad Attribution Debiasing — 因果去混淆广告归因区分"广告真正带来
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-counterfactual-ad-attribution-debiasing/SKILL.md`
    - `p2s-cross-channel-budget-pacing-controller` · 源卡号 `Skill-Cross-Channel-Budget-Pacing-Controller` · 15-营销投放分析 · Cross-Channel Budget Pacing Controller — 跨渠道预算节奏控制防止早耗尽与末期停投
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-channel-budget-pacing-controller/SKILL.md`
    - `p2s-dara-agentic-mmm-optimizer` · 源卡号 `Skill-DARA-Agentic-MMM-Optimizer` · 15-营销投放分析 · DARA - LLM+RL 双阶段广告预算分配 Agent
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dara-agentic-mmm-optimizer/SKILL.md`
    - `p2s-dara-agentic-mmm` · 源卡号 `Skill-DARA-Agentic-MMM` · 15-营销投放分析 · DARA Agentic MMM — LLM Agent 驱动的营销组合建模：双阶段自动调参与智能归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dara-agentic-mmm/SKILL.md`
    - `p2s-dtc-customer-acquisition-attribution` · 源卡号 `Skill-DTC-Customer-Acquisition-Attribution` · 06-增长模型 · DTC Customer Acquisition Attribution — 独立站全渠道获客归因：从首触到首单的因果追
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dtc-customer-acquisition-attribution/SKILL.md`
    - `p2s-full-funnel-growth-dashboard` · 源卡号 `Skill-Full-Funnel-Growth-Dashboard` · 14-用户分析 · Full Funnel Growth Dashboard — 多归因视角聚合的全漏斗增长量化看板
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-full-funnel-growth-dashboard/SKILL.md`
    - `p2s-generative-bidding-moe` · 源卡号 `Skill-Generative-Bidding-MoE` · 13-广告分析 · 生成式广告竞价 — MoE路由+因果Transformer
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-generative-bidding-moe/SKILL.md`
    - `p2s-geo-holdout-experiment` · 源卡号 `Skill-Geo-Holdout-Experiment` · 02-A_B实验 · Geo Holdout实验 — 地理区域控制实验消除网络效应偏差
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-geo-holdout-experiment/SKILL.md`
    - `p2s-geo-level-marketing-effectiveness` · 源卡号 `Skill-Geo-Level-Marketing-Effectiveness` · 15-营销投放分析 · Geo-Level Marketing Effectiveness（地理级营销效果）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-geo-level-marketing-effectiveness/SKILL.md`
    - `p2s-hmmcb-cross-channel-bidding` · 源卡号 `Skill-HMMCB-Cross-Channel-Bidding` · 13-广告分析 · HMMCB — 跨渠道广告竞价 MARL：CPC 约束下最大化总点击（美团真实 A/B）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hmmcb-cross-channel-bidding/SKILL.md`
    - `p2s-inpromo-realtime-decision-kpi` · 源卡号 `Skill-InPromo-Realtime-Decision-KPI` · 04-供应链 · 大促中实时决策KPI与流量协同阈值 — 售罄速率监控/流量协同触发/紧急干预决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inpromo-realtime-decision-kpi/SKILL.md`
    - `p2s-keyword-bid-auto-adjuster` · 源卡号 `Skill-Keyword-Bid-Auto-Adjuster` · 13-广告分析 · Keyword-Bid-Auto-Adjuster — 关键词转化率偏差自动调整出价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-keyword-bid-auto-adjuster/SKILL.md`
    - `p2s-ltv-acquisition-budget-gate` · 源卡号 `Skill-LTV-Acquisition-Budget-Gate` · 06-增长模型 · LTV-Acquisition-Budget-Gate — LTV/CAC比值驱动的获客预算自动开闸/熔断决策器
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ltv-acquisition-budget-gate/SKILL.md`
    - `p2s-ltv-cac-acquisition-gate` · 源卡号 `Skill-LTV-CAC-Acquisition-Gate` · 06-增长模型 · LTV CAC Acquisition Gate — LTV/CAC比率触发渠道获客自动暂停或扩投
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ltv-cac-acquisition-gate/SKILL.md`
    - `p2s-mas-ad-budget-multi-platform-negotiation` · 源卡号 `Skill-MAS-Ad-Budget-Multi-Platform-Negotiation` · 10-MAS · MAS多平台广告预算Stackelberg博弈 — 多平台广告Agent序贯博弈预算分配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-ad-budget-multi-platform-negotiation/SKILL.md`
    - `p2s-mmm-budget-pl-alignment` · 源卡号 `Skill-MMM-Budget-PL-Alignment` · 15-营销投放分析 · MMM Budget PL Alignment — 营销预算分配与利润约束下的 ROI 优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mmm-budget-pl-alignment/SKILL.md`
    - `p2s-mmm-budget-reallocation-executor` · 源卡号 `Skill-MMM-Budget-Reallocation-Executor` · 15-营销投放分析 · MMM Budget Reallocation Executor — 将MMM最优权重转化为渠道预算调整API执行指令
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mmm-budget-reallocation-executor/SKILL.md`
    - `p2s-mtl-multi-objective-ad-optimization` · 源卡号 `Skill-MTL-Multi-Objective-Ad-Optimization` · 13-广告分析 · MTL 广告多目标联合优化 — ROAS/排名/曝光 Pareto 最优分配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mtl-multi-objective-ad-optimization/SKILL.md`
    - `p2s-multi-objective-budget-allocation` · 源卡号 `Skill-Multi-Objective-Budget-Allocation` · 15-营销投放分析 · Multi-Objective Budget Allocation（多目标预算分配）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-objective-budget-allocation/SKILL.md`
    - `p2s-multi-platform-ad-budget-allocator` · 源卡号 `Skill-Multi-Platform-Ad-Budget-Allocator` · 15-营销投放分析 · 多平台广告预算最优分配 — 约束优化最大化跨平台 ROAS
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-platform-ad-budget-allocator/SKILL.md`
    - `p2s-multi-seller-account-portfolio` · 源卡号 `Skill-Multi-Seller-Account-Portfolio` · 23-运营财务 · Multi-Seller Account Portfolio Management — 多店铺账号组合管理：跨账号资源协
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-seller-account-portfolio/SKILL.md`
    - `p2s-nonlinear-multi-touch-attribution` · 源卡号 `Skill-Nonlinear-Multi-Touch-Attribution` · 14-用户分析 · 多触点非线性归因建模 — 跨渠道用户旅程因果归因与预算决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-nonlinear-multi-touch-attribution/SKILL.md`
    - `p2s-ppc-bid-manipulation-defense` · 源卡号 `Skill-PPC-Bid-Manipulation-Defense` · 19-风控反欺诈 · PPC Bid Manipulation Defense — 识别竞品恶意点击耗费广告费
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ppc-bid-manipulation-defense/SKILL.md`
    - `p2s-ppc-keyword-bid-automation` · 源卡号 `Skill-PPC-Keyword-Bid-Automation` · 13-广告分析 · PPC Keyword Bid Automation — PPC 关键词出价自动化：ML 驱动的竞价优化引擎
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ppc-keyword-bid-automation/SKILL.md`
    - `p2s-ppc-rule-automation-engine` · 源卡号 `Skill-PPC-Rule-Automation-Engine` · 13-广告分析 · PPC Rule Automation Engine — PPC 规则自动化引擎：条件触发的广告出价智能运营
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ppc-rule-automation-engine/SKILL.md`
    - `p2s-qubo-ad-budget-allocation` · 源卡号 `Skill-QUBO-Ad-Budget-Allocation` · 13-广告分析 · 量子退火 (QUBO) 驱动的跨平台广告预算全局最优分配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-qubo-ad-budget-allocation/SKILL.md`
    - `p2s-roas-below-target-budget-freeze` · 源卡号 `Skill-ROAS-Below-Target-Budget-Freeze` · 13-广告分析 · ROAS-Below-Target-Budget-Freeze — ROAS连续3天低于目标自动冻结广告组预算触发创意审
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-roas-below-target-budget-freeze/SKILL.md`
    - `p2s-rtb-realtime-bidding-optimization` · 源卡号 `Skill-RTB-Realtime-Bidding-Optimization` · 13-广告分析 · RTB Realtime Bidding Optimization — 生成式自动出价与多约束实时竞价优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-rtb-realtime-bidding-optimization/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 2 张
    - `Skill-Cannibalization-Corrected-Attribution` · 13-广告分析 · Skill-Cannibalization-Corrected-Attribution
      - 全文卡：`paper2skills-vault/13-广告分析/Skill-Cannibalization-Corrected-Attribution.md`
    - `Skill-Causal-Budget-Allocation` · 13-广告分析 · Skill-Causal-Budget-Allocation
      - 全文卡：`paper2skills-vault/13-广告分析/Skill-Causal-Budget-Allocation.md`

> 候选总数 53。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-060 · 差异追踪

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-040` GMV结算与会计对账 |
| 域 / 面 | `DOM-07` / `PLN-OPS` |
| flows | FLOW-01, FLOW-03, FLOW-05, FLOW-06, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-060-差异追踪.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：差异追踪可用异常检测与根因相关分析持续定位。


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
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 9 张
    - `p2s-accounts-receivable-intelligence` · 源卡号 `Skill-Accounts-Receivable-Intelligence` · 23-运营财务 · Accounts Receivable Intelligence — 账期智能管理：跨境应收账款预测与催收优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-accounts-receivable-intelligence/SKILL.md`
    - `p2s-agent-finance-autopilot` · 源卡号 `Skill-Agent-Finance-Autopilot` · 23-运营财务 · Agent Finance Autopilot — LLM 多 Agent 财务自动化与 P&L 实时分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-agent-finance-autopilot/SKILL.md`
    - `p2s-fba-fee-intelligence` · 源卡号 `Skill-FBA-Fee-Intelligence` · 23-运营财务 · FBA Fee Intelligence（FBA 费用结构分析与长库龄预警）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fba-fee-intelligence/SKILL.md`
    - `p2s-fba-fee-waterfall-attribution` · 源卡号 `Skill-FBA-Fee-Waterfall-Attribution` · 23-运营财务 · Skill-FBA-Fee-Waterfall-Attribution — FBA费用瀑布归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fba-fee-waterfall-attribution/SKILL.md`
    - `p2s-inventory-theft-warehouse-anomaly` · 源卡号 `Skill-Inventory-Theft-Warehouse-Anomaly` · 19-风控反欺诈 · Inventory Theft Warehouse Anomaly — 海外仓库存异常检测盗窃/错发/损耗识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-inventory-theft-warehouse-anomaly/SKILL.md`
    - `p2s-mas-revenue-operations` · 源卡号 `Skill-MAS-Revenue-Operations` · 10-MAS · MAS运营财务协同 — 多智能体驱动的P&L实时归因与决策
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-revenue-operations/SKILL.md`
    - `p2s-pl-attribution-analysis` · 源卡号 `Skill-PL-Attribution-Analysis` · 23-运营财务 · P&L Attribution Analysis（SKU 级损益归因分析）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-pl-attribution-analysis/SKILL.md`
    - `p2s-platform-commission-fee-tracker` · 源卡号 `Skill-Platform-Commission-Fee-Tracker` · 23-运营财务 · Platform Commission Fee Tracker — 平台佣金费率变化监控（Amazon/TikTok/S
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-platform-commission-fee-tracker/SKILL.md`
    - `p2s-profitability-waterfall-by-asin` · 源卡号 `Skill-Profitability-Waterfall-By-ASIN` · 23-运营财务 · Skill-Profitability-Waterfall-By-ASIN — 单品盈利瀑布分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-profitability-waterfall-by-asin/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 9。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-A-067 · 主数据治理

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-045` 业务口径与主数据 |
| 域 / 面 | `DOM-08` / `PLN-PLT` |
| flows | FLOW-01, FLOW-06, FLOW-08 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-067-主数据治理.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：主数据去重、合并与质量度量为标准MDM算法。


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

- 有全文卡（优先选，可选到论文参数）：共 5 张
    - `p2s-hgcn-hyperbolic-graph-convolutional-networks` · 源卡号 `Skill-HGCN-Hyperbolic-Graph-Convolutional-Networks` · 08-知识图谱 · HGCN — 双曲图卷积网络
      - 全文卡：`/Users/lute/.dsh/skills/p2s-hgcn-hyperbolic-graph-convolutional-networks/SKILL.md`
    - `p2s-hgt-heterogeneous-graph-transformer` · 源卡号 `Skill-HGT-Heterogeneous-Graph-Transformer` · 08-知识图谱 · HGT — 异构图 Transformer 表示学习
      - 全文卡：`/Users/lute/.dsh/skills/p2s-hgt-heterogeneous-graph-transformer/SKILL.md`
    - `p2s-kg-auto-construction-agent-driven` · 源卡号 `Skill-KG-Auto-Construction-Agent-Driven` · 08-知识图谱 · AI Agent 驱动的电商知识图谱自动构建
      - 全文卡：`/Users/lute/.dsh/skills/p2s-kg-auto-construction-agent-driven/SKILL.md`
    - `p2s-kg-relation-completion-cblip` · 源卡号 `Skill-KG-Relation-Completion-CBLiP` · 08-知识图谱 · Knowledge Graph Relation Completion with CBLiP
      - 全文卡：`/Users/lute/.dsh/skills/p2s-kg-relation-completion-cblip/SKILL.md`
    - `p2s-multilingual-ner-universal-v2` · 源卡号 `Skill-Multilingual-NER-Universal-v2` · 08-知识图谱 · Multilingual Named Entity Recognition (Universal NER v2)
      - 全文卡：`/Users/lute/.dsh/skills/p2s-multilingual-ner-universal-v2/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 35 张
    - `p2s-autopkg-multimodal-product-attribute-kg` · 源卡号 `Skill-AutoPKG-Multimodal-Product-Attribute-KG` · 08-知识图谱 · AutoPKG — 多模态产品属性图谱自动构建：文本+图片→GMV提升
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-autopkg-multimodal-product-attribute-kg/SKILL.md`
    - `p2s-cross-platform-user-identity-resolution` · 源卡号 `Skill-Cross-Platform-User-Identity-Resolution` · 22-数据采集工程 · Cross-Platform User Identity Resolution — 跨平台用户 ID 统一
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-platform-user-identity-resolution/SKILL.md`
    - `p2s-cross-platform-user-identity` · 源卡号 `Skill-Cross-Platform-User-Identity` · 22-数据采集工程 · 跨平台用户 ID 统一 — Amazon/TikTok/独立站身份解析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-platform-user-identity/SKILL.md`
    - `p2s-cross-system-data-reconciliation` · 源卡号 `Skill-Cross-System-Data-Reconciliation` · 24-标签工程 · 跨系统数据对账 — ERP/WMS/OMS三系统库存一致性自动比对与差异处置
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-system-data-reconciliation/SKILL.md`
    - `p2s-dial-kg-schema-free-incremental` · 源卡号 `Skill-DIAL-KG-Schema-Free-Incremental` · 08-知识图谱 · DIAL-KG无Schema增量知识图谱构建 — 动态Schema归纳+治理裁决+增量演化闭环
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-dial-kg-schema-free-incremental/SKILL.md`
    - `p2s-data-mesh-ecommerce` · 源卡号 `Skill-Data-Mesh-Ecommerce` · 22-数据采集工程 · Data Mesh for Ecommerce — 数据网格架构
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-data-mesh-ecommerce/SKILL.md`
    - `p2s-ecommerce-data-quality-assessment` · 源卡号 `Skill-Ecommerce-Data-Quality-Assessment` · 22-数据采集工程 · E-commerce Data Quality Assessment — 产品目录数据质量双轨评分与门控
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ecommerce-data-quality-assessment/SKILL.md`
    - `p2s-entity-resolution-kg-dedup` · 源卡号 `Skill-Entity-Resolution-KG-Dedup` · 08-知识图谱 · KG 实体消歧与去重（Entity Resolution & Deduplication）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-entity-resolution-kg-dedup/SKILL.md`
    - `p2s-fastkge-incremental-lora-kg-embedding` · 源卡号 `Skill-FastKGE-Incremental-LoRA-KG-Embedding` · 08-知识图谱 · FastKGE — 增量 LoRA 知识图谱嵌入
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-fastkge-incremental-lora-kg-embedding/SKILL.md`
    - `p2s-graph-okb-design-sc` · 源卡号 `Skill-Graph-OKB-Design-SC` · 24-标签工程 · 供应链操作知识库OKB图谱设计 — Neo4j+Delta双层架构与CDC实时同步策略
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-graph-okb-design-sc/SKILL.md`
    - `p2s-hcce-concept-hierarchy-embedding` · 源卡号 `Skill-HCCE-Concept-Hierarchy-Embedding` · 08-知识图谱 · HCCE — 超球面锥概念层次嵌入
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hcce-concept-hierarchy-embedding/SKILL.md`
    - `p2s-hierarchical-product-kg-construction` · 源卡号 `Skill-Hierarchical-Product-KG-Construction` · 08-知识图谱 · 层级商品知识图谱自动构建（图片→KG）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-hierarchical-product-kg-construction/SKILL.md`
    - `p2s-kg-application-patterns` · 源卡号 `Skill-KG-Application-Patterns` · 08-知识图谱 · KG Application Patterns — 知识图谱下游应用：从构建到推荐/搜索/冷启动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-application-patterns/SKILL.md`
    - `p2s-kg-data-fusion-pipeline` · 源卡号 `Skill-KG-Data-Fusion-Pipeline` · 08-知识图谱 · KG Data Fusion Pipeline — 多源采集数据驱动的知识图谱自动构建：竞品属性图谱融合
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-data-fusion-pipeline/SKILL.md`
    - `p2s-kg-incremental-update-pipeline` · 源卡号 `Skill-KG-Incremental-Update-Pipeline` · 22-数据采集工程 · 知识图谱增量更新流水线 — 实体变更检测与一致性维护
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-incremental-update-pipeline/SKILL.md`
    - `p2s-kg-incremental-update` · 源卡号 `Skill-KG-Incremental-Update` · 08-知识图谱 · 知识图谱增量更新（KG Incremental Update）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-kg-incremental-update/SKILL.md`
    - `p2s-knowledge-graph-auto-update` · 源卡号 `Skill-Knowledge-Graph-Auto-Update` · 22-数据采集工程 · Knowledge Graph Auto-Update — 知识图谱增量更新与一致性维护
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-knowledge-graph-auto-update/SKILL.md`
    - `p2s-low-cost-dependency-kg-construction` · 源卡号 `Skill-Low-Cost-Dependency-KG-Construction` · 08-知识图谱 · 低成本依存解析KG构建 — 1/10成本达到LLM 94%质量的无监督知识图谱构建
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-low-cost-dependency-kg-construction/SKILL.md`
    - `p2s-mas-dynamic-kg-collaboration` · 源卡号 `Skill-MAS-Dynamic-KG-Collaboration` · 10-MAS · MAS Dynamic KG Collaboration — 多智能体动态知识图谱协同：实时构建、冲突解决、协同进化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-dynamic-kg-collaboration/SKILL.md`
    - `p2s-multi-source-user-identity-unification` · 源卡号 `Skill-Multi-Source-User-Identity-Unification` · 14-用户分析 · Multi-Source User Identity Unification — 跨平台用户身份统一打通 Amazon/
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-source-user-identity-unification/SKILL.md`
    - `p2s-multi-task-user-representation` · 源卡号 `Skill-Multi-Task-User-Representation` · 14-用户分析 · Multi-Task User Representation — 多任务用户表示学习：统一用户画像驱动全业务
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multi-task-user-representation/SKILL.md`
    - `p2s-ontology-llm-autobuild-sc` · 源卡号 `Skill-Ontology-LLM-AutoBuild-SC` · 24-标签工程 · LLM驱动供应链本体自动构建 — 从ERP文档到语义图谱的零样本迭代萃取
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ontology-llm-autobuild-sc/SKILL.md`
    - `p2s-ontology-schema-design` · 源卡号 `Skill-Ontology-Schema-Design` · 08-知识图谱 · 领域 Ontology 与图谱 Schema 设计
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ontology-schema-design/SKILL.md`
    - `p2s-openre-llm-knowledge-extraction` · 源卡号 `Skill-OpenRE-LLM-Knowledge-Extraction` · 08-知识图谱 · Skill-OpenRE-LLM-Knowledge-Extraction
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-openre-llm-knowledge-extraction/SKILL.md`
    - `p2s-privacy-safe-identity-resolution` · 源卡号 `Skill-Privacy-Safe-Identity-Resolution` · 22-数据采集工程 · Privacy-Safe Identity Resolution — 隐私合规跨平台 ID 解析：多方对齐与差分隐私
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-privacy-safe-identity-resolution/SKILL.md`
    - `p2s-product-attribute-completion` · 源卡号 `Skill-Product-Attribute-Completion` · 22-数据采集工程 · Product Attribute Completion — 商品属性自动补全：AI 填补 Listing 属性空白
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-attribute-completion/SKILL.md`
    - `p2s-product-knowledge-graph-query` · 源卡号 `Skill-Product-Knowledge-Graph-Query` · 08-知识图谱 · Product KG Query — 多 Agent 产品知识图谱查询与 SKU 跨平台映射
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-product-knowledge-graph-query/SKILL.md`
    - `p2s-sc-ontology-schema-versioning` · 源卡号 `Skill-SC-Ontology-Schema-Versioning` · 24-标签工程 · 供应链本体Schema版本化与迁移 — 向后兼容的本体演化策略防止Agent系统崩溃
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sc-ontology-schema-versioning/SKILL.md`
    - `p2s-sku-entity-unified-id-tagging` · 源卡号 `Skill-SKU-Entity-Unified-ID-Tagging` · 24-标签工程 · SKU跨平台实体统一标识与标签同步 — ASIN/SKU/ERP三码合一的实体对齐与标签一致性保障
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sku-entity-unified-id-tagging/SKILL.md`
    - `p2s-sku-master-data-golden-record` · 源卡号 `Skill-SKU-Master-Data-Golden-Record` · 24-标签工程 · SKU主数据黄金记录治理 — MDM体系、主键统一与多源冲突消解
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-sku-master-data-golden-record/SKILL.md`
    - `p2s-supplier-ontology-capability-map` · 源卡号 `Skill-Supplier-Ontology-Capability-Map` · 24-标签工程 · 供应商本体能力图谱 — 产能/质量/认证/风险四维供应商Ontology设计与实例化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supplier-ontology-capability-map/SKILL.md`
    - `p2s-supply-chain-data-mesh-architecture` · 源卡号 `Skill-Supply-Chain-Data-Mesh-Architecture` · 24-标签工程 · 供应链数据网格架构 — 领域自治的分布式数据治理与跨域数据共享协议
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-supply-chain-data-mesh-architecture/SKILL.md`
    - `p2s-tag-schema-engineering-lifecycle` · 源卡号 `Skill-Tag-Schema-Engineering-Lifecycle` · 24-标签工程 · 标签Schema工程与生命周期管理 — 企业级Tag类型设计、Schema约束与版本治理
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tag-schema-engineering-lifecycle/SKILL.md`
    - `p2s-tagrag-hierarchical-label-kg` · 源卡号 `Skill-TagRAG-Hierarchical-Label-KG` · 08-知识图谱 · TagRAG层级标签知识图谱 — 对象标签链驱动的超高效KG构建与检索
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tagrag-hierarchical-label-kg/SKILL.md`
    - `p2s-itext2kg-schema-free-kg-induction` · 源卡号 `Skill-iText2KG-Schema-Free-KG-Induction` · 08-知识图谱 · iText2KG — 零样本增量知识图谱构建
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-itext2kg-schema-free-kg-induction/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 3 张
    - `Skill-Agentic-Catalog-Enrichment` · 00-电商Agent · Skill-Agentic-Catalog-Enrichment
      - 全文卡：`paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md`
    - `Skill-CrossLingual-Semantic-Alignment` · 07-NLP-VOC · Skill-CrossLingual-Semantic-Alignment
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-CrossLingual-Semantic-Alignment.md`
    - `Skill-Live-Catalog-Conversational-Rec` · 00-电商Agent · Skill-Live-Catalog-Conversational-Rec
      - 全文卡：`paper2skills-vault/00-电商Agent/Skill-Live-Catalog-Conversational-Rec.md`

> 候选总数 43。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-028 · 市场进入

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-024` 其他平台与新市场经营 |
| 域 / 面 | `DOM-04` / `PLN-OPS` |
| flows | FLOW-01, FLOW-02, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-028-市场进入.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：进入决策可用市场规模/竞争强度模型打分，但准入与履约准备须外部证据。


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

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 16 张
    - `p2s-ai-product-safety-certification` · 源卡号 `Skill-AI-Product-Safety-Certification` · 21-合规决策 · AI Product Safety Certification — AI驱动产品安全认证自动化：从测试报告解析到合规路径
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-product-safety-certification/SKILL.md`
    - `p2s-augmented-synthetic-control-ml` · 源卡号 `Skill-Augmented-Synthetic-Control-ML` · 01-因果推断 · Augmented Synthetic Control — 机器学习增强的合成控制法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-augmented-synthetic-control-ml/SKILL.md`
    - `p2s-brand-penetration-modeling` · 源卡号 `Skill-Brand-Penetration-Modeling` · 15-营销投放分析 · Brand Penetration Modeling — 品牌渗透率建模：Bass 扩散 × 跨市场品牌增长预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-brand-penetration-modeling/SKILL.md`
    - `p2s-causal-representation-disentangle` · 源卡号 `Skill-Causal-Representation-Disentangle` · 01-因果推断 · Causal Representation Learning Disentangle — 解耦因果表示学习（VAE+因果
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-representation-disentangle/SKILL.md`
    - `p2s-causal-representation-learning` · 源卡号 `Skill-Causal-Representation-Learning` · 01-因果推断 · 因果表示学习 — 从观测数据中学习干预不变的因果潜变量
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-representation-learning/SKILL.md`
    - `p2s-causal-representation-transfer-learning` · 源卡号 `Skill-Causal-Representation-Transfer-Learning` · 12-ML基础 · 因果表示学习跨域迁移 — 从源域提取不变因果特征用于目标域零样本适配
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-representation-transfer-learning/SKILL.md`
    - `p2s-cross-border-compliance-framework` · 源卡号 `Skill-Cross-Border-Compliance-Framework` · 21-合规决策 · Cross-Border Compliance Framework — 跨境电商多辖区合规自动映射
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-compliance-framework/SKILL.md`
    - `p2s-cross-market-product-transfer` · 源卡号 `Skill-Cross-Market-Product-Transfer` · 06-增长模型 · Cross-Market Product Transfer（跨市场产品适配性预测）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-market-product-transfer/SKILL.md`
    - `p2s-cross-market-transfer-demand` · 源卡号 `Skill-Cross-Market-Transfer-Demand` · 03-时间序列 · 跨市场需求迁移学习 — 用成熟市场数据加速新市场冷启动
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-market-transfer-demand/SKILL.md`
    - `p2s-gpsr-eu-risk-assessment-auto` · 源卡号 `Skill-GPSR-EU-Risk-Assessment-Auto` · 21-合规决策 · GPSR EU Risk Assessment Auto — 欧盟GPSR风险评估自动化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-gpsr-eu-risk-assessment-auto/SKILL.md`
    - `p2s-mas-compliance-multi-market-orchestrator` · 源卡号 `Skill-MAS-Compliance-Multi-Market-Orchestrator` · 10-MAS · MAS-Compliance-Multi-Market-Orchestrator — 多市场合规检查Agent并行编排与
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-compliance-multi-market-orchestrator/SKILL.md`
    - `p2s-multimarket-expansion-readiness-scorer` · 源卡号 `Skill-Multimarket-Expansion-Readiness-Scorer` · 06-增长模型 · Multimarket Expansion Readiness Scorer（多市场拓展就绪度评分）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-multimarket-expansion-readiness-scorer/SKILL.md`
    - `p2s-new-market-entry-readiness-gate` · 源卡号 `Skill-New-Market-Entry-Readiness-Gate` · 06-增长模型 · New-Market-Entry-Readiness-Gate — 新市场进入评分超阈值自动生成进入Checklist并
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-new-market-entry-readiness-gate/SKILL.md`
    - `p2s-ot-cross-market-demand-transfer` · 源卡号 `Skill-OT-Cross-Market-Demand-Transfer` · 04-供应链 · OT Cross-Market Demand Transfer — 最优传输跨市场需求分布迁移
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ot-cross-market-demand-transfer/SKILL.md`
    - `p2s-ssbc-small-sample-conformal` · 源卡号 `Skill-SSBC-Small-Sample-Conformal` · 01-因果推断 · 小样本Beta修正共形预测 - 50个样本也能保证覆盖
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ssbc-small-sample-conformal/SKILL.md`
    - `p2s-shopee-lazada-sea-market-intelligence` · 源卡号 `Skill-Shopee-Lazada-SEA-Market-Intelligence` · 14-用户分析 · 东南亚市场情报与选品分群 — Shopee/Lazada 热搜词 + K-Means 价格带定位
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-shopee-lazada-sea-market-intelligence/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 1 张
    - `Skill-TJAP-跨市场品类组合定价` · 07-NLP-VOC · Skill-TJAP-跨市场品类组合定价
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-TJAP-跨市场品类组合定价.md`

> 候选总数 17。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-035 · 内容策划

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-030` 内容与创意策划 |
| 域 / 面 | `DOM-05` / `PLN-OPS` |
| flows | FLOW-01, FLOW-02, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-035-内容策划.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：内容可生成与排序优化，但选题与主张需品牌与合规判断。


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

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 26 张
    - `p2s-a-plus-content-template-engine` · 源卡号 `Skill-A-Plus-Content-Template-Engine` · 20-AI视频生成 · A+Content模板引擎 — VOC驱动的A+内容自动排版生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-a-plus-content-template-engine/SKILL.md`
    - `p2s-ai-brand-storytelling` · 源卡号 `Skill-AI-Brand-Storytelling` · 11-AI人文 · AI Brand Storytelling — AI 辅助品牌故事创作：情感连接与文化适应
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-brand-storytelling/SKILL.md`
    - `p2s-ai-content-marketing-growth` · 源卡号 `Skill-AI-Content-Marketing-Growth` · 11-AI人文 · AI内容营销增长 — AIGC驱动的内容→用户增长因果链路
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-content-marketing-growth/SKILL.md`
    - `p2s-ai-product-video-script-generator` · 源卡号 `Skill-AI-Product-Video-Script-Generator` · 20-AI视频生成 · Skill-AI-Product-Video-Script-Generator — AI 母婴产品视频脚本自动生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-product-video-script-generator/SKILL.md`
    - `p2s-ai-video-script-generation` · 源卡号 `Skill-AI-Video-Script-Generation` · 20-AI视频生成 · AI Video Script Generation — 分层 CoT 电商短视频脚本自动生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-video-script-generation/SKILL.md`
    - `p2s-causal-seo-search-attribution` · 源卡号 `Skill-Causal-SEO-Search-Attribution` · 25-搜索流量工程 · 搜索流量因果归因 — SEO排名提升的真实增量效果识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-causal-seo-search-attribution/SKILL.md`
    - `p2s-content-roi-budget-shift-trigger` · 源卡号 `Skill-Content-ROI-Budget-Shift-Trigger` · 13-广告分析 · Content ROI Budget Shift Trigger — 内容ROI连续低于目标时自动削减并转向高ROI内容
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-content-roi-budget-shift-trigger/SKILL.md`
    - `p2s-epidemiological-viral-traffic-sir` · 源卡号 `Skill-Epidemiological-Viral-Traffic-SIR` · 06-增长模型 · SIR传染病模型TikTok流量拐点预测 — 提前2周锁定爆款峰值
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-epidemiological-viral-traffic-sir/SKILL.md`
    - `p2s-geo-generative-engine-optimization` · 源卡号 `Skill-GEO-Generative-Engine-Optimization` · 15-营销投放分析 · GEO — 生成式引擎优化：让 AI 搜索引擎主动引用你的品牌内容
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-geo-generative-engine-optimization/SKILL.md`
    - `p2s-live-script-optimization-nlp` · 源卡号 `Skill-Live-Script-Optimization-NLP` · 07-NLP-VOC · 直播话术优化NLP — 情感曲线+格兰杰因果挖掘高转化话术模式
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-live-script-optimization-nlp/SKILL.md`
    - `p2s-loss-aversion-promotion-design` · 源卡号 `Skill-Loss-Aversion-Promotion-Design` · 15-营销投放分析 · 损失厌恶促销设计 — 「限时最后3件」比「立省30元」点击率高35%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-loss-aversion-promotion-design/SKILL.md`
    - `p2s-mas-video-content-optimization` · 源卡号 `Skill-MAS-Video-Content-Optimization` · 10-MAS · MAS视频内容优化 — 多智能体协同的短视频全链路运营
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mas-video-content-optimization/SKILL.md`
    - `p2s-relate-rl-ad-text-generation` · 源卡号 `Skill-RELATE-RL-Ad-Text-Generation` · 13-广告分析 · RELATE强化学习广告文案生成 — RL端到端优化CTCVR的LLM广告创意框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-relate-rl-ad-text-generation/SKILL.md`
    - `p2s-reward-model-rlhf-ecommerce` · 源卡号 `Skill-Reward-Model-RLHF-Ecommerce` · 12-ML基础 · RLHF奖励模型电商 — 人类偏好对齐的电商AI质量提升
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-reward-model-rlhf-ecommerce/SKILL.md`
    - `p2s-search-organic-growth-attribution` · 源卡号 `Skill-Search-Organic-Growth-Attribution` · 25-搜索流量工程 · 搜索有机增长归因 — SEO流量驱动用户生命周期价值的因果建模
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-search-organic-growth-attribution/SKILL.md`
    - `p2s-short-video-commerce-attribution` · 源卡号 `Skill-Short-Video-Commerce-Attribution` · 13-广告分析 · 短视频内容归因建模 — 视频特征 XGBoost 驱动转化分析
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-short-video-commerce-attribution/SKILL.md`
    - `p2s-tag-video-commerce-tagging` · 源卡号 `Skill-Tag-Video-Commerce-Tagging` · 24-标签工程 · 视频商品标签化 — AI驱动的短视频商品识别与自动标签体系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tag-video-commerce-tagging/SKILL.md`
    - `p2s-tiktok-algorithm-content-boost` · 源卡号 `Skill-TikTok-Algorithm-Content-Boost` · 20-AI视频生成 · TikTok Algorithm Content Boost — FYP 算法建模与内容传播速度优化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-algorithm-content-boost/SKILL.md`
    - `p2s-tiktok-algorithm-traffic-amplification` · 源卡号 `Skill-TikTok-Algorithm-Traffic-Amplification` · 15-营销投放分析 · TikTok推流算法流量放大 — 逆向工程完播率/互动率临界点识别
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-algorithm-traffic-amplification/SKILL.md`
    - `p2s-tiktok-content-lifecycle-analytics` · 源卡号 `Skill-TikTok-Content-Lifecycle-Analytics` · 20-AI视频生成 · TikTok内容生命周期分析 — 指数衰减建模与二次投流最佳时机
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-content-lifecycle-analytics/SKILL.md`
    - `p2s-tiktok-hook-optimizer` · 源卡号 `Skill-TikTok-Hook-Optimizer` · 20-AI视频生成 · Skill-TikTok-Hook-Optimizer — TikTok 开头钩子前3秒留存预测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-hook-optimizer/SKILL.md`
    - `p2s-tiktok-shop-content-attribution` · 源卡号 `Skill-TikTok-Shop-Content-Attribution` · 13-广告分析 · TikTok Shop Content Attribution — 短视频带货兴趣图谱归因
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-tiktok-shop-content-attribution/SKILL.md`
    - `p2s-ugc-viral-content-potential-scorer` · 源卡号 `Skill-UGC-Viral-Content-Potential-Scorer` · 07-NLP-VOC · UGC病毒传播潜力评分 — 识别可引发病毒扩散的高潜力内容
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ugc-viral-content-potential-scorer/SKILL.md`
    - `p2s-video-emotion-resonance-score` · 源卡号 `Skill-Video-Emotion-Resonance-Score` · 20-AI视频生成 · Video Emotion Resonance Score — 视频情感共鸣评分（母婴内容感动度量化）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-video-emotion-resonance-score/SKILL.md`
    - `p2s-videorag-video-knowledge-retrieval` · 源卡号 `Skill-VideoRAG-Video-Knowledge-Retrieval` · 08-知识图谱 · VideoRAG — 视频知识库检索增强生成
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-videorag-video-knowledge-retrieval/SKILL.md`
    - `p2s-zero-click-search-optimization` · 源卡号 `Skill-Zero-Click-Search-Optimization` · 13-广告分析 · Zero-Click Search Optimization — 零点击搜索时代的流量保全与特色摘要抢占
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-zero-click-search-optimization/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 1 张
    - `Skill-LLM-Personalized-Marketing-Copy-Generation` · 07-NLP-VOC · Skill-LLM-Personalized-Marketing-Copy-Generation
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-LLM-Personalized-Marketing-Copy-Generation.md`

> 候选总数 27。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。

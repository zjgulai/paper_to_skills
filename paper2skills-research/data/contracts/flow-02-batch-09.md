# FLOW-02 契约撰写 · 批次 09（3 份）

> 公共材料见 `flow-02-common.md`（**先读它**）。本批 1 份 A 模板 / 2 份 B 模板。


---

## CTR-A-015 · 算法评估设计

| 项 | 值 |
|---|---|
| 模板 | **A**（由算法可服务性 `A` 唯一导出） |
| 岗位 | `AGT-012` 软件算法与应用生态 |
| 域 / 面 | `DOM-02` / `PLN-OPS` |
| flows | FLOW-02, FLOW-07 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/A/CTR-A-015-算法评估设计.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 A 类（材料 §F.3 原文，别改判）**：评估协议、离线指标与离线-在线一致性检验是标准机器学习评估工程。


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

- 有全文卡（优先选，可选到论文参数）：共 1 张
    - `p2s-agent-stage-evaluation` · 源卡号 `Skill-Agent-Stage-Evaluation` · 16-智能体工程 · EComStage — 电商 Agent 三阶段(Perception/Planning/Action)双向 Bench
      - 全文卡：`/Users/lute/.dsh/skills/p2s-agent-stage-evaluation/SKILL.md`
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 20 张
    - `p2s-ai-algorithmic-bias-audit` · 源卡号 `Skill-AI-Algorithmic-Bias-Audit` · 11-AI人文 · AI Algorithmic Bias Audit — AI 算法偏见审计：跨境电商推荐公平性检测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-algorithmic-bias-audit/SKILL.md`
    - `p2s-ai-ethics-fairness-audit` · 源卡号 `Skill-AI-Ethics-Fairness-Audit` · 11-AI人文 · AI 公平性审计 — 推荐/定价/广告系统偏差溯源与修复
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ai-ethics-fairness-audit/SKILL.md`
    - `p2s-algorithmic-fairness-in-pricing` · 源卡号 `Skill-Algorithmic-Fairness-in-Pricing` · 11-AI人文 · 算法定价公平性审计 — 群体公平指标与价格歧视检测
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-algorithmic-fairness-in-pricing/SKILL.md`
    - `p2s-crag-comprehensive-rag-benchmark` · 源卡号 `Skill-CRAG-Comprehensive-RAG-Benchmark` · 08-知识图谱 · CRAG — 综合RAG评测基准
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-crag-comprehensive-rag-benchmark/SKILL.md`
    - `p2s-cross-cultural-ai-bias-correction` · 源卡号 `Skill-Cross-Cultural-AI-Bias-Correction` · 11-AI人文 · Cross Cultural AI Bias Correction
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-cultural-ai-bias-correction/SKILL.md`
    - `p2s-cross-validation-strategies` · 源卡号 `Skill-Cross-Validation-Strategies` · 12-ML基础 · Cross-Validation Strategies（交叉验证策略）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-validation-strategies/SKILL.md`
    - `p2s-flashrag-efficient-rag-toolkit` · 源卡号 `Skill-FlashRAG-Efficient-RAG-Toolkit` · 08-知识图谱 · FlashRAG — 高效模块化RAG研究工具包
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-flashrag-efficient-rag-toolkit/SKILL.md`
    - `p2s-helmet-longcontext-rag-evaluation` · 源卡号 `Skill-HELMET-LongContext-RAG-Evaluation` · 08-知识图谱 · Skill-HELMET-LongContext-RAG-Evaluation
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-helmet-longcontext-rag-evaluation/SKILL.md`
    - `p2s-interference-spillover-correction` · 源卡号 `Skill-Interference-Spillover-Correction` · 02-A_B实验 · 实验溢出效应纠正 — 网络干扰下A/B实验的因果偏差修正
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-interference-spillover-correction/SKILL.md`
    - `p2s-interleaving-experiment-design` · 源卡号 `Skill-Interleaving-Experiment-Design` · 02-A_B实验 · Interleaving 实验设计 — 用混排对照替代传统 A/B，提升排序策略评估效率 10 倍
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-interleaving-experiment-design/SKILL.md`
    - `p2s-interleaving-experiment-recommendation` · 源卡号 `Skill-Interleaving-Experiment-Recommendation` · 02-A_B实验 · 推荐系统交错实验 — 快速在线评估推荐算法的位置无关方法
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-interleaving-experiment-recommendation/SKILL.md`
    - `p2s-interleaving-ranking-ab-test` · 源卡号 `Skill-Interleaving-Ranking-AB-Test` · 02-A_B实验 · Interleaving for Ranking AB Test — 排序系统的交叉实验（比传统 AB 更灵敏）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-interleaving-ranking-ab-test/SKILL.md`
    - `p2s-long-term-holdout-group-design` · 源卡号 `Skill-Long-Term-Holdout-Group-Design` · 02-A_B实验 · Long-term Holdout Group Design — 长期保留组设计（防止学习效应污染）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-long-term-holdout-group-design/SKILL.md`
    - `p2s-mteb-embedding-benchmark-selection` · 源卡号 `Skill-MTEB-Embedding-Benchmark-Selection` · 08-知识图谱 · Skill-MTEB-Embedding-Benchmark-Selection
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-mteb-embedding-benchmark-selection/SKILL.md`
    - `p2s-model-evaluation-metrics` · 源卡号 `Skill-Model-Evaluation-Metrics` · 12-ML基础 · Skill Card: Model Evaluation Metrics（模型评估体系）
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-model-evaluation-metrics/SKILL.md`
    - `p2s-rag-vs-finetuning-decision-framework` · 源卡号 `Skill-RAG-vs-Finetuning-Decision-Framework` · 08-知识图谱 · Skill-RAG-vs-Finetuning-Decision-Framework
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-rag-vs-finetuning-decision-framework/SKILL.md`
    - `p2s-raglab-research-rag-framework` · 源卡号 `Skill-RAGLAB-Research-RAG-Framework` · 08-知识图谱 · RAGLAB — 研究导向的模块化RAG实验框架
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-raglab-research-rag-framework/SKILL.md`
    - `p2s-ranking-interleaving-ab` · 源卡号 `Skill-Ranking-Interleaving-AB` · 02-A_B实验 · 排序交叉实验 — 比传统 AB 更灵敏的排序系统评估
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-ranking-interleaving-ab/SKILL.md`
    - `p2s-reliabilitybench-agent-reliability` · 源卡号 `Skill-ReliabilityBench-Agent-Reliability` · 16-智能体工程 · ReliabilityBench — Agent 生产可靠性三维评估：pass@1 高估 20-40%
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-reliabilitybench-agent-reliability/SKILL.md`
    - `p2s-switchback-experiment-design` · 源卡号 `Skill-Switchback-Experiment-Design` · 02-A_B实验 · Switchback 实验设计 - 数据驱动的双边市场实验
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-switchback-experiment-design/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 21。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-014 · 产品范围管理

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-009` 产品定义与商业立项 |
| 域 / 面 | `DOM-02` / `PLN-OPS` |
| flows | FLOW-02, FLOW-04 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-014-产品范围管理.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：范围变更影响可估算，但裁剪与承诺变更属立项治理决定。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-04/STG-04 | M | 证据收集与诊断 | 核对机会、商品主数据、证据、宣称、税务、履约、售后和账号准备 | 市场进入诊断 |
| FLOW-04/STG-05 | M | 方案与标准产物 | 形成准入证据矩阵、本地化内容和市场进入包 | 市场进入包 |
| FLOW-04/STG-08 | M | 结果核验、关闭与异步学习 | 核对逐商品上线回执、异常和初步经营结果 | 市场进入关闭记录 |
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
> `paper2skills-research/data/contracts/flow-02-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 0 张
    （无）
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 1 张
    - `Skill-iReFeed-需求优先级排序` · 07-NLP-VOC · Skill-iReFeed-需求优先级排序
      - 全文卡：`paper2skills-vault/07-NLP-VOC/00-知识库-Skill卡片/Skill-iReFeed-需求优先级排序.md`

> 候选总数 1。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。


---

## CTR-B-050 · 会员活动

| 项 | 值 |
|---|---|
| 模板 | **B**（由算法可服务性 `B` 唯一导出） |
| 岗位 | `AGT-039` 用户教育与会员社区 |
| 域 / 面 | `DOM-06` / `PLN-OPS` |
| flows | FLOW-02, FLOW-05 |
| 骨架文件 | `paper2skills-vault/07-资源库/contracts/B/CTR-B-050-会员活动.md` |
| `status` 初值 | **可写**（有候选卡） |

**本责任为什么是 B 类（材料 §F.3 原文，别改判）**：活动效果可用增量测量，但权益设计与活动创意属运营判断。


**本契约自己的格（只能引用这些格号）**：

| 格 | 类型 | 阶段 | 业务步骤 | 标准产物 |
|---|---|---|---|---|
| FLOW-02/STG-04 | M | 证据收集与诊断 | 区分事实与假设，核对VOC、替代方案、技术/OEM/质量/准入证据 | 机会与可行性诊断 |
| FLOW-02/STG-05 | M | 方案与标准产物 | 形成商业假设、产品定义和验证方案 | 新品验证证据包 |
| FLOW-02/STG-08 | M | 结果核验、关闭与异步学习 | 核对验证结果并形成继续、转向或停止结论 | 新品验证关闭记录 |
| FLOW-05/STG-04 | M | 证据收集与诊断 | 核对客户与订单事实、历史问题、服务规则、触达许可和产品适用性 | 客户问题诊断 |
| FLOW-05/STG-05 | M | 方案与标准产物 | 形成答复、补救、教育或复购实验方案 | 服务与留存方案 |
| FLOW-05/STG-08 | M | 结果核验、关闭与异步学习 | 核对问题解决、补救回执、根因反馈和复购结果 | 客户旅程关闭记录 |
| FLOW-02/STG-01 | D | 信号接收 | 接收需求证据、使用问题或产品组合缺口 | 新品机会信号 |
| FLOW-02/STG-02 | R | 范围确定与Case创建 | 确定用户问题、市场、产品范围、验证目标与停止条件 | 新品验证Case Charter |
| FLOW-02/STG-03 | D | 上下文装配 | 装配需求、竞品、产品定义、工程、OEM、质量、合规和验证能力 | FLOW-02 Context Manifest |
| FLOW-02/STG-06 | R | Assurance接收门禁 | 检查可证伪性、实物/专业证据来源、合规、资源与混杂因素 | FLOW-02 Assurance Decision |
| FLOW-02/STG-07 | D | 受控动作 | 发起受控外部验证Intent或记录无需动作；不得把模型判断当实物回执 | 验证尝试或无动作记录 |
| FLOW-05/STG-01 | D | 信号接收 | 接收咨询、订单异常、售后或获许可的生命周期事件 | 客户旅程信号 |
| FLOW-05/STG-02 | R | 范围确定与Case创建 | 确定客户/订单最小范围、问题类型、许可状态和完成条件 | 客户Case Charter |
| FLOW-05/STG-03 | D | 上下文装配 | 按事件类型选择主岗位，装配服务、体验、教育、CRM、隐私和质量能力 | FLOW-05 Context Manifest |
| FLOW-05/STG-06 | R | Assurance接收门禁 | 检查许可、隐私、健康风险、质量事件、补偿和触达边界 | FLOW-05 Assurance Decision |
| FLOW-05/STG-07 | D | 受控动作 | 提交退款/补救/触达Intent或NoActionRecord | 客户动作尝试或无动作记录 |

**候选方法卡**（`cards` 从下面选；A 模板 §1 的「不得移植的具体数值」**必须来自你实际读过的卡**）：

> ⚠️ **下面是完整清单，不是节选**（首版只渲染前 6 条，实测撰写人因此漏卡 —— 一位撰写人看到
> 「共 13 张」却只有 6 条，只能从可见的 6 张里挑）。机读版在
> `paper2skills-research/data/contracts/flow-02-workpack.json` 的 `card_candidates`。

- 有全文卡（优先选，可选到论文参数）：共 0 张
    （无）
- 只有 legacy 预览版（可引 slug，但 §1 必须写明「论文实验参数未随卡进入本包」，清单按卡面可见数字列）：共 11 张
    - `p2s-cross-border-member-onboarding-optimization` · 源卡号 `Skill-Cross-Border-Member-Onboarding-Optimization` · 06-增长模型 · Cross-Border Member Onboarding Optimization — 跨境会员注册漏斗优化结合早期
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-cross-border-member-onboarding-optimization/SKILL.md`
    - `p2s-generative-agent-simulation` · 源卡号 `Skill-Generative-Agent-Simulation` · 06-增长模型 · 生成式智能体营销沙盒仿真 - 零数据消费者行为推演
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-generative-agent-simulation/SKILL.md`
    - `p2s-high-value-customer-alert-action` · 源卡号 `Skill-High-Value-Customer-Alert-Action` · 14-用户分析 · High-Value-Customer-Alert-Action — RFM高价值客户30天沉默自动触发客服主动介入+个
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-high-value-customer-alert-action/SKILL.md`
    - `p2s-high-value-customer-proactive-alert` · 源卡号 `Skill-High-Value-Customer-Proactive-Alert` · 14-用户分析 · High Value Customer Proactive Alert — 高价值客户出现沉默信号时自动触发客服主动联系
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-high-value-customer-proactive-alert/SKILL.md`
    - `p2s-loyalty-program-roi-modeling` · 源卡号 `Skill-Loyalty-Program-ROI-Modeling` · 06-增长模型 · Loyalty Program ROI Modeling — 双重差分评估会员积分体系 LTV 增量价值
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-loyalty-program-roi-modeling/SKILL.md`
    - `p2s-member-lifecycle-intervention-sequencing` · 源卡号 `Skill-Member-Lifecycle-Intervention-Sequencing` · 06-增长模型 · Member Lifecycle Intervention Sequencing — RL 序列干预优化会员生命周期各阶
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-member-lifecycle-intervention-sequencing/SKILL.md`
    - `p2s-membership-tier-design-optimization` · 源卡号 `Skill-Membership-Tier-Design-Optimization` · 06-增长模型 · Membership Tier Design Optimization — 多层会员体系结构的最优设计与 CLV 最大化
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-membership-tier-design-optimization/SKILL.md`
    - `p2s-points-expiry-redemption-liability-model` · 源卡号 `Skill-Points-Expiry-Redemption-Liability-Model` · 06-增长模型 · Points Expiry Redemption Liability Model — 积分过期负债精算与兑换率动态定价
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-points-expiry-redemption-liability-model/SKILL.md`
    - `p2s-rdd-regression-discontinuity-design` · 源卡号 `Skill-RDD-Regression-Discontinuity-Design` · 01-因果推断 · 断点回归设计 — 利用政策阈值识别因果效应
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-rdd-regression-discontinuity-design/SKILL.md`
    - `p2s-topological-data-analysis-cross-sell` · 源卡号 `Skill-Topological-Data-Analysis-Cross-Sell` · 05-推荐系统 · 拓扑数据分析 (TDA) 挖掘时空隐性关联销售路径
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-topological-data-analysis-cross-sell/SKILL.md`
    - `p2s-vip-tier-upgrade-action` · 源卡号 `Skill-VIP-Tier-Upgrade-Action` · 14-用户分析 · VIP-Tier-Upgrade-Action — LTV超阈值自动触发VIP等级升级与礼遇通知
      - 注意：**legacy 预览版**（八段式：④/⑤ 段是占位串、代码以 `references/implementation.py` 随卡且为节选、⑧ 论文出处常写「未自动抽取」或「待人工判定」）→ `/Users/lute/.dsh/skills/p2s-vip-tier-upgrade-action/SKILL.md`
- 仅精选线（尚未换底进产品线，引 `card_id`，并注明「S5 换底后替换为已装 slug」）：共 0 张
    （无）

> 候选总数 11。**`cards` 只写 1–3 张代表卡**（同族，不是全部）——
> 契约按**责任**建、不按卡建，§1 必须写一句「本契约对同族方法卡通用；换卡时只替换方法实现，
> 其余要求不变」。

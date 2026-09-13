---
contract_id: CTR-A-003
template: A
responsibility: 依赖协调
role_id: AGT-002
role_title: 场景自主编排与异常协调
domain_id: DOM-01
plane_id: PLN-MGT
serviceability: A
flows: [FLOW-01, FLOW-02, FLOW-03, FLOW-04, FLOW-05, FLOW-06, FLOW-07, FLOW-08]
method_cells: [FLOW-01/STG-04, FLOW-01/STG-05, FLOW-01/STG-08, FLOW-02/STG-04, FLOW-02/STG-05, FLOW-02/STG-08, FLOW-03/STG-04, FLOW-03/STG-05, FLOW-03/STG-08, FLOW-04/STG-04, FLOW-04/STG-05, FLOW-04/STG-08, FLOW-05/STG-04, FLOW-05/STG-05, FLOW-05/STG-08, FLOW-06/STG-04, FLOW-06/STG-05, FLOW-06/STG-08, FLOW-07/STG-04, FLOW-07/STG-05, FLOW-07/STG-08, FLOW-08/STG-04, FLOW-08/STG-05, FLOW-08/STG-08]
rule_cells: [FLOW-01/STG-01, FLOW-01/STG-02, FLOW-01/STG-03, FLOW-01/STG-06, FLOW-01/STG-07, FLOW-02/STG-01, FLOW-02/STG-02, FLOW-02/STG-03, FLOW-02/STG-06, FLOW-02/STG-07, FLOW-03/STG-01, FLOW-03/STG-02, FLOW-03/STG-03, FLOW-03/STG-06, FLOW-03/STG-07, FLOW-04/STG-01, FLOW-04/STG-02, FLOW-04/STG-03, FLOW-04/STG-06, FLOW-04/STG-07, FLOW-05/STG-01, FLOW-05/STG-02, FLOW-05/STG-03, FLOW-05/STG-06, FLOW-05/STG-07, FLOW-06/STG-01, FLOW-06/STG-02, FLOW-06/STG-03, FLOW-06/STG-06, FLOW-06/STG-07, FLOW-07/STG-01, FLOW-07/STG-02, FLOW-07/STG-03, FLOW-07/STG-06, FLOW-07/STG-07, FLOW-08/STG-01, FLOW-08/STG-02, FLOW-08/STG-03, FLOW-08/STG-06, FLOW-08/STG-07]
cards: [p2s-concat-consensus-decentralized-mas, p2s-dag-task-decomposition-planning, p2s-dynamic-dag-orchestration, p2s-mas-orchestrator]
status: 方法待启用
blocked_by: Case/Event Ledger 的「预留 vs 实际执行」序列尚未产生（材料自承 50 岗位业务能力名称为待落实需求、未证明已接入企业系统，故需先建该事件流），五维之③回溯深度在预留维度上为空 ⇒ 缓冲项 b(r) 无历史序列可用，预留审批暂按硬约束 D(r,W) ≤ A(r,W) 判定
---

# 依赖协调 · 供给契约

## 1 方法来源（卡供方法）

- 方法族：**多任务／多岗位之间的依赖排序与资源冲突消解**（并行申报 → 等价合并 → 代表请求 → 依赖边修剪 → 无环可行序）。族内各卡的方法实现可互换，本契约取的是族骨架，不是某一张卡的实现。
- 代表卡：`p2s-concat-consensus-decentralized-mas`（源卡号 `Skill-CONCAT-Consensus-Decentralized-MAS`，源域 10-MAS，图谱 L3 = DOM-01-006 依赖协调；卡面论文 arXiv:2605.29612，另引 arXiv:2602.00966）
- 同族已装卡（同一 L3「依赖协调」，可整卡替换代表卡）：`p2s-dag-task-decomposition-planning`（DAG 解耦与局部重算）、`p2s-dynamic-dag-orchestration`（运行时改拓扑）、`p2s-mas-orchestrator`（调度与失败恢复）。
- 可搬的部分（族级，一段话）：把「各任务先并行独立申报自己的资源占用与上下游依赖，再按等价性合并、按证据强度选代表、按贡献修剪依赖边，最后输出一份无环可行序」这条骨架搬到本责任上——① 每张 Case 在范围确定阶段自行申报资源预留与依赖，不排队等中央逐个问询；② 指向同一资源对象的等价请求合并为一条代表请求，真分歧各自保留并显式暴露，不做多数投票抹平；③ 代表请求由证据强度与权威来源选出，不由发起方自封；④ 依赖边只保留对结论有贡献的边；⑤ 单个 Case 的等待或冻结不阻塞其余 Case 的阶段推进（但不得绕过 STG-06 接收门禁）。
- **不得移植的部分**（按代表卡逐条列出卡内论文实验条件的具体数值，任一条都不得进入本业务口径；换入同族卡时，同样要逐条剔除该卡自身的实验数值）：
  1. 共识聚类的语义相似度阈值 `similarity_threshold = 0.35`（卡内实现默认值）；
  2. 通信边修剪的效益阈值 `min_benefit_threshold = 0.20`（卡内实现默认值）；
  3. 论文实测性能：2.02x 效率提升、50.1% 延迟降低；
  4. 场景 A 的规模设定：US/UK/DE/AU 四个市场、四个合规 Agent；
  5. 场景 B 的规模设定：5 个专业 Agent（价格／质量／交期／合规／物流）评估 10 个候选供应商、串行 120 秒降至并行 45 秒（−62.5%）、O(log N) Beacon 路由；
  6. 卡面 ROI 数字群：ROI 62.5%、日处理 50 次、每次 2.5 min、每月 22 天、月节省 2750 min ≈ 45 小时、$1125/月、系统成本 $5 万、ROI ≈ 270%；
  7. 卡面适用规模门槛：「5 个以上 Agent」「>100 次/天」；
  8. 卡面数据依赖声明「无需历史数据」——本责任的预留缓冲恰恰必须由本业务台账的历史序列算出，该声明不得作为本契约的数据前提；
  9. 卡内 `confidence` 的「0–1 自评置信度」口径与「按最高置信度选举组长」——本业务的优先级不由自评置信度决定（见第 3 段 T3）；
  10. 卡内以「答案词集 Jaccard 相似度」判等答案的做法——本业务判等用资源对象标识集合与时间窗的集合运算，不用文本相似度；
  11. 卡面包体积数：实现 253 行、节选 60 行、代码块 3 个、卡面记的规划路径 `paper2skills-code/mas/concat_consensus_decentralized_mas`（该路径不在包内）——不得据此推断本业务的实现规模或工期。
- 清单完备性声明：卡面未出现消融设置与迭代轮数，本清单**不是节选**——凡卡内出现的论文实验数值均已逐条列入。
- 本契约对同族方法卡通用；换卡时只替换方法实现，数据要求、标定规则与冻结条件不变。

## 2 数据要求（五维）

| 维 | 取值 |
|---|---|
| ① 获取路径 | 自有埋点（需先由 Case Control 落盘 Case/Event Ledger 事件流——Case Charter 的 `resource_reservation_refs` 与 `resource_conflict_inputs`、Stage Acceptance Record、Execution Attempt 与权威回执）；资源可用量取平台后台（渠道侧可售／在途库存、账号动作配额、广告预算余额与花费用量）与采购、财务台账 |
| ② 粒度 | 预留与占用＝Case × 资源对象 × 时间窗；依赖边＝Case × Case × 依赖类型；资源对象按 AGT-045 主数据标识（SKU／账号／仓库／预算科目／供应商） |
| ③ 回溯深度 | 存量经营数据 2 个完整年度（决策 Q11）；「预留 vs 实际执行」序列在本业务尚未产生，见 blocked_by |
| ④ 新鲜度 | 每次进入 STG-02 前重读一次；某资源对象的占用以最后一条 Stage Acceptance Record 为准，其时间戳早于本次范围确定时刻即判过期，按八阶段协议的 required_input_missing_stale_or_unavailable 置 WAITING |
| ⑤ 口径归属 | AGT-045 指标契约与主数据（资源对象标识与可用量定义）；每条记录随带 AGT-046 的质量状态、来源与新鲜度 |

说明：① 所列两段来源中，平台后台与采购／财务台账是本企业现存系统；Case/Event Ledger 事件流是必须先建的一段，缺口已如实记入 `blocked_by`，不在正文用措辞掩盖。

## 3 标定规则

- 取值来源：自有埋点（Case/Event Ledger 的 Case Charter 与 Stage Acceptance Record 事件流）＋平台后台与采购／财务台账的权威可用量；不引用卡内任何论文参数。
- **T1 依赖图（边从哪来，不含相似度分数）**：顶点＝未关闭 Case（OPEN／WAITING／REWORK_REQUIRED）。边 b→a（b 等 a）成立当且仅当下列任一为真——(i) 对象重叠：a 与 b 的 Case Charter 显式业务范围内、按 AGT-045 主数据标识的资源对象集合交集非空；(ii) 阶段产物依赖：按 CASE-PROTOCOL-8 的阶段产物表，b 当前阶段待接收的产物类型由 a 的当前或更晚阶段产出（即 b 在等 a 的 Stage Acceptance Record）；(iii) 预留覆盖：b 待申领的（资源对象，时间窗）与 a 已占用的（资源对象，时间窗）重叠。三类之外的通信或相关关系不建边。
- **T2 冲突判定（口径与算式）**：对每个资源对象 r 与时间窗 W，D(r,W) ＝ Σ 所有未关闭 Case 的 `resource_reservation_refs` 中落在 (r,W) 的申领量；A(r,W) 按对象类别取权威值——预算科目＝平台后台／财务台账的科目可用余额（预算 − 已承诺 − 已执行）；可售库存＝库存台账可售量（不含已预留与在途）；账号动作配额＝平台后台当期可执行上限；供应商产能与采购在途＝采购台账未占用产能与在途量。判定：D(r,W) ≤ A(r,W) ⇒ 直接写入预留、不排队；D(r,W) > A(r,W) ⇒ 命中 resource_conflict_detected_before_reservation_or_external_attempt，该 Case 置 WAITING 并生成 REC-CASE-EXCEPTION，不得静默抢占，也不得按先到先得放行。
- **T3 可行序与预留**：
  ① 可行序＝对 T1 边集做拓扑排序，输出无环序；出现环（互相等待）即不输出序，置 WAITING／FROZEN 并把环上报，环内任何 Case 不得进入受控动作阶段；
  ② 冲突集内定序（字典序，逐级只读台账字段）：命中 D-026 紧急保护通道的 Case 优先；其次已产生不可逆外部承诺或已执行动作的 Case 优先（避免同一对象重复写入）；再次出度大者优先（按 T1 现算的依赖数，即阻塞下游更多者）；三级仍并列即判据不唯一，不自动裁决，置 WAITING 并上报该 FLOW 主岗位与人工；
  ③ 预留量与缓冲：预留量＝该 Case 在 STG-05 产物中申领的对象与量上限；缓冲 b(r) ＝近 2 个完整年度已关闭 Case 在资源对象 r 上的 max〔(申领量 − 实际执行量) ÷ 申领量〕（序列由 STG-08 关闭记录回写）；仅当 D(r,W) × (1 + b(r)) ≤ A(r,W) 时批准预留。序列为空时缓冲项恒为 0（预留审批即退化为 T2 的硬约束），本责任不另设经验值；台账上线满 2 个完整年度后按上式直接算出；
  ④ 已写入预留后再出现资源所有权或外部状态不确定 ⇒ 置 FROZEN，按八阶段协议的 reconcile_reservation_and_external_state_before_any_new_action 先对账再动。
- 产出（本规则定值的字段）：
  - STG-02 的 8 份 Charter（经营Case Charter／新品验证Case Charter／供需Case Charter／市场进入Case Charter／客户Case Charter／能力交付Case Charter／重大事件Case Charter／经营复盘Case Charter）：`resource_reservation_refs` 与 `resource_conflict_inputs` 两个接收门禁字段由 T1、T2 定值；
  - STG-04 的 8 份诊断产物（FLOW-01/STG-04 经营证据与偏差诊断 … FLOW-08/STG-04 经营与能力诊断）：依赖边集与冲突集由 T1、T2 定值；
  - STG-05 的 8 份方案产物（FLOW-01/STG-05 联合经营行动包 … FLOW-08/STG-05 经营与能力变更建议）：可行序、预留量与停止条件里的资源位由 T3 定值；
  - STG-08 的 8 份关闭产物（FLOW-01/STG-08 经营关闭记录 … FLOW-08/STG-08 经营复盘关闭记录）：预留与实际的偏差回写，构成 T3 的 b(r) 序列。

## 4 重标定触发条件

- **口径变**：AGT-045 对资源对象标识或可用量定义发布新版本；AGT-046 的质量状态／来源／新鲜度口径改版。
- **数据源变**：Case/Event Ledger 首次上线或换版（缓冲项必须由「不设」改为按 b(r) 实算）；库存台账、采购台账、财务台账或平台后台任一被替换或改字段。
- **结构变**：CASE-PROTOCOL-8 的阶段产物表或接收门禁版本变化（新增／取消产物类型、增删阶段）；三条 FLOW 主岗位 selector 由 production_ready=false 变为可唯一确定（FLOW-01／FLOW-05／FLOW-07），适用主岗位范围随之重算。
- **主体变**：开放事实 O1（Amazon 65% 与独立站 15% 之外的其余 20% 渠道构成未明）一旦定名，或新增市场／渠道／账号主体，资源对象集合与账号范围重算。
- **周期到期**：每 2 个完整年度（决策 Q11 的历史长度）重算一次 b(r) 与资源冲突频次。

## 5 适用 FLOW

FLOW-01, FLOW-02, FLOW-03, FLOW-04, FLOW-05, FLOW-06, FLOW-07, FLOW-08

## 6 冻结与不许自动放行的情形

- 资源冲突未解（D(r,W) > A(r,W)）且定序三级并列 ⇒ 不得进入受控动作，置 WAITING 并上报裁决。
- 依赖边集出现环（互相等待）⇒ 环内任何 Case 不得放行。
- 必需输入缺失、质量状态非可用或新鲜度超时（AGT-046 口径）⇒ 置 WAITING，依赖与新鲜度复核通过前不得推进。
- 已写入预留后资源所有权或外部状态不确定 ⇒ 置 FROZEN，须先完成预留与外部状态对账。
- 同一资源对象上存在已执行但回执未对平的动作 ⇒ 不得对该对象发起新动作；已成功动作不得随整单重跑。
- 跨 FLOW 争用同一库存或预算科目 ⇒ 本责任只输出冲突集与可行序候选，不得单方裁决；裁决权在该 FLOW 的主岗位与人工。
- STG-06 接收门禁未通过，或仍有未解决的语义发现 ⇒ 不得放行，发起方不得覆盖拒绝结论。
- Case 处于 FROZEN，或其固定的能力包进入 QUARANTINED ⇒ 不得发起新动作。
- 命中 D-026 的模型外紧急保护通道时，冻结与恢复、重新开放、扩大范围、外部承诺、退款、召回、销毁与永久变更只能在正常 STG-07 执行；本责任的可行序不得把紧急通道当成一条捷径。
- FLOW-01/STG-02 的接收门禁要求把 `resource_conflict_inputs` 记入 Case Charter；该字段缺失时不得把 Case 从范围确定推到下一阶段。
- 本段落在 R/D 格（STG-01／STG-02／STG-03／STG-06／STG-07）上的只是上列判据与门禁条件，不引入判别式或预测式。

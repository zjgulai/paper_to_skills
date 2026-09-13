---
title: Skill-Multi-Agent-Collaboration-Tax
topic: 多 Agent 协作税（solo-full 与同质配对之差）的测量、四阶段失败级联与「该不该拆多 Agent」的裁决
status: draft
created: 2026-09-12
updated: 2026-09-12
module: 10-MAS
owner: self
source: ai
paper_id: 2608.22152
paper: The Collaboration Tax: How Much LLM Multi-Agent Systems Pay to Coordinate
venue: EMNLP 2026
venue_tier: top
evidence_grade: A
verified_by: verify_skill_code.py（K1 L5 PASS）+ quote_check.py（引文逐字核验 VERBATIM）+ gate_check.py G2/G3 passed + 人工抽检 3 处数字
verified_at: 2026-09-12
supersedes:
related: Skill-Subagent-Decomposition.md, Skill-MAS-Orchestrator.md, Skill-MetaGPT-SOP-Driven-Collaboration.md, Skill-Multi-Agent-Debate.md, Skill-Agent-Stage-Evaluation.md, Skill-Context-Compression.md
---

# Skill Card: 多 Agent 协作税（Collaboration Tax）—— 「这个流程到底该不该拆成多 Agent」

**本卡与同域其它卡的分工**：`10-MAS/` 里已有的卡片（`Skill-Subagent-Decomposition.md`、
`Skill-MAS-Orchestrator.md`、`Skill-MetaGPT-SOP-Driven-Collaboration.md`、`Skill-Multi-Agent-Debate.md` …）
回答的都是「**怎么搭**多 Agent」；本卡回答的是「**该不该搭**、搭了亏多少、亏在对话的哪一阶段、
不换模型不重训能不能修好」。交付物是**一个可测的量 + 一个四阶段诊断器 + 一份 prompt 级干预**，
不是又一套编排框架。

**先看结论**：把「单 Agent 拿全量信息」与「两个 Agent 各拿一半私有视图、对话到底」放在同一批任务上、
用同一把确定性尺子打分，两者之差就是协作税。论文在 32 个任务 × 11 个模型上测出来的结论是：
这个税**不是推理能力问题**，而是对话在四个阶段逐步漏气；它**可以从对话特征离线预测**，
并且**用一个 prompt 改动就能回收一部分**。反过来说——**在你拆之前，先把这笔税量出来**。

---

## ① 算法原理

**核心思想**：多 Agent 不是免费的。协作税把「团队协调的代价」从「任务难度」里剥离出来，
变成一个**有符号、可分解、可预测**的量：同一把确定性评分器下，一个人拿全量信息做 vs 两个人各拿一半做，
差多少。

**数学直觉**：`tax(M,T) = s_solo-full(M,T) − s_homo(M,T)`（论文式 (1)）。前一项是单个模型拿到
**合并实例**的得分，后一项是同一个模型的两个副本各持一半私有视图、交换消息直到终止的得分；
cell 均值来自 50 次独立 seed 的 rollout。除以 `s_solo-full` 得到可跨任务比较的 ratio gap。
理论上 tax ≥ 0，且 **tax > 0 恰好等价于合作博弈违反 max-superadditivity**：
`v({1,2}) < max(v({1}), v({2}))` —— 两个人一起做，还不如其中更强的那一个单独做。
异质配对再多一步：Shapley 份额 `φᵢ = ½(vᵢ + v₁₂ − vⱼ)`，只要 `v₁₂ < v₁ + v₂`，**双方份额都低于各自的单人收益**。

**关键假设（四条设计原则，缺一条 tax 就不可解释）**：
① **solo-trivial** —— 单 Agent 本来就该做得不错，否则差里混着能力不足；
② **union-necessary** —— 任一视图单独都定不了答案；
③ **algorithmically verifiable** —— 评分器必须是确定性函数，给出连续分 `[0,1]`；
④ **multiply expressible** —— 同一内容有多个等价表面表示，grounding friction 才存在。

---

## ①b 反例与适用边界

> **这一节是本卡的主要价值**：现有卡片都在教「怎么搭」，本卡教「**什么时候不要搭**」。

### 一、什么时候不要拆多 Agent

**1. 去掉任意一方的视图，任务照样能做 → 不要拆。**
论文的 union-necessary 条件要求：两边视图的并集才等于完整实例，且**任一视图单独都得不出标准答案**
（⑥ Q9）。如果某一方的私有视图本身已足以定答案（退化划分 `v1 = x`），配对协议可以直接忽略对方、
完整复刻单 Agent 策略，**tax 按构造恒等于 0**（⑥ Q4）——此时的「多 Agent」只是把同一件活收了两遍钱，
外加一份协调开销。
→ **落地判据**：把打算拆开的两个角色各写一句「我手里的信息能否独立得出结论」。两边都能独立得出结论
= 不该拆；只有一边能 = 拆了只会在两边之间制造无谓的对齐成本。

**2. 两边的视图可以平凡拼接 → 不要拆。**
若两份视图共享同一套坐标原点 / 轴序 / 命名约定 / 术语表，agent 可以直接把对方的内容串起来、
**零协调成本地复刻集中式基线**，tax 的上界取等号（⑥ Q5）。让多 Agent 真正有价值的不是「信息分散」，
而是「**同一份内容有多种等价表示、而两边各自用了不同的那一种**」（⑥ Q10）。
→ **落地判据**：如果两个 agent 都从同一个数仓宽表、同一套 SKU 编码、同一份库龄 / 认证口径取数，
它们之间**不存在 grounding friction**，拆开只会引入额外的对齐成本。

**3. 单 Agent 本来就做不好 → 先修单 Agent，不要先做税测量。**
solo-trivial 是测量前提（⑥ Q8）：如果 `s_solo-full` 离天花板很远，tax 里混着「能力不足」与「协调失败」
两种成分，这个差没法归因。→ **落地判据**：先看单 Agent 在主指标上的水平；低于你们能接受的水平时，
本卡的一切结论都不适用。

**4. 错误传播是「一步错、全盘错」的链路 → 拆的代价最大。**
论文按错误传播模式区分三个任务族：路径类任务**单步错位会作废后续全部路径**，约束类违反会级联，
只有关系查询类错误停在局部（⑥ Q11）。对应到税的结构：路径 / 坐标类（Spatial）损失最大、
关系类居中、约束类（CSP）最小（⑥ Q13）。
→ **落地判据**：链路里存在「上游一个值错了、下游全部作废」的环节（认证状态、头程报价、库存快照、库龄口径），
就是高风险拆分对象，**必须先测再拆**。

**5. 模型能力弱 → 不要指望靠拆分补回来。**
论文实测税随能力**单调**下降：最弱的模型把大约一半的 solo 成功率丢在协调上，同一批任务上最强的模型
也仍有损失（⑥ Q14、Q15）。「用便宜模型 + 多开几个角色」是最差的组合。
→ **落地判据**：想做「小模型团队」时，先按 ③ 的模板在自家任务上量一遍税，**不要默认角色分工能补上能力差**。

**6. 视野里只有两个人以外的结构 → 本卡不适用。** 论文的形式化只覆盖两人、单一共同收益、
union-necessary 信息（⑥ Q40），N ≥ 3、加权贡献、非合作博弈都在范围之外（⑥ Q35）。

### 二、已知的失败模式：四阶段对话级联

论文最重要的机制结论是：**失败不是推理能力不足**，而是对话在四个阶段逐步漏气（⑥ Q16）。
四个阶段都是二值判定，从早到晚：

| 阶段 | 判定定义 | 典型症状 | 论文原话 / 证据 |
|---|---|---|---|
| **L1 grounding** | 每一句事实断言都能回溯到某一方视图 | 把对方没说过的值当成既有事实（坐标轴转置、张冠李戴） | 「L1 failures are sourcing errors」；**一旦编造，下游任何阶段都救不回来**（⑥ A1、Q17） |
| **L2 querying** | 至少一方提出过**具体事实问题** | 沉默：每句话都成立，但没有人问 | 「L2 failures are silences: each utterance is wellformed; what is missing is a question.」（⑥ A2、Q21） |
| **L3 integration** | 决定性断言之前出现过显式合并状态消息 | 没合并就提交；对话提前终止 | 「L3 failures are truncations」（⑥ Q22）；L3 是**单变量最强预测器**，也是唯一在多元回归里边际贡献为正的阶段（⑥ Q18） |
| **L4 re-derivation** | 接收方在结束前展示了**重算过程** | 抢着发终止符、「我同意」而不重算 | 论文 Appendix E 的跨案例总结把它叫做「premature ACTI!」，并给出了一条可抄的子句（⑥ A8、Q19） |

**三条必须知道的纠缠性质**：
1. 四阶段**可分离但不独立** —— 多数失败 rollout 同时触发至少两个阶段（⑥ Q20、Q44）。
2. **L3 与 L4 高度纠缠**（失败集合 Jaccard 0.67），观测数据里 L4 在 L3 之外没有独立回归权重；
   只有用 prompt **从外部强制**了 L3，L4 才暴露出自己的贡献（⑥ Q45、Q29）。→ 诊断时**不要**因为
   「L4 在回归里不显著」就删掉 L4 的子句。
3. L2 是最独立的一维（论文 Appendix D）。→ 它也是最适合单独做 A/B 的那一条子句。

**人工校验的可信度边界**：四阶段 judge 由四位专家在一百条分层 rollout 上交叉标注，
维度间一致度**中等偏上**，自动化 judge 与专家多数票的一致度在 L1/L2/L3 上较高、L4 上只有中等
（⑥ A14）。→ 用 judge 做 KPI 考核之前，先在自己的数据上重算一次一致度。

### 三、论文自己承认的局限（逐条来自 Limitations 与 Appendix A.7）

1. **只测了两人（N = 2）**。四人五人的结构变了（私有视图变成 N 份、信道变成两两），论文说这**未被研究**，
   仍在扩展中（⑥ Q35）。→ **不要**拿本卡去论证「多人团队也一样」。
2. **任务是程序生成的谜题，不是生产工作负载**。论文点名自己的设定**不能直接对应**到协同改代码、
   issue 分诊、多轮 debug、交互式文档起草这些真实多 Agent 场景，因为那些场景的答案空间与反馈信号
   **远没有那么结构化**（⑥ Q36、Q37）。→ 本卡 ② 是把**方法**搬到业务上自测，不是把论文结论搬过来。
3. **存在负税 cell**：足够强的模型在约束类任务上税可以是负的，论文记录了两例，并自陈这违反了经典的
   tax ≥ 0，原因是 LLM 策略随机且依赖上下文（⑥ Q38）。→ 别把「税为正」当成定理。
4. **对话被硬截断在上限之内**，论文推测 prompt 干预后残余的税有一部分来自这个上限（⑥ Q39）。
5. **异质配对只有两对模型、四种配置**，论文自称是「定性模式的 existence proof，不是定量刻画」
   （⑥ Q34）。→ 表 2 那种合作博弈数字**只能当结构性示意**用。
6. **论文未报告**：任何工程成本、token 消耗、延迟、人力投入量级；也**未报告**母婴 / 跨境电商场景的任何数字。
   本卡 ⑤ 自建 ROI 公式，必须由企业代入自有口径。
7. **论文未讨论**：把「税测量」工程化成常态化监控（每次改 prompt / 换模型都重测一次）的流程。
   → 本卡 ③ 的模板是这个方向的尝试，**不是论文内容**。
8. **评测口径的耦合**：solo-full 模式带一次 critic 复核，论文单独做了消融证明它不是税的来源
   （⑥ Q41）。→ 你做对照时也必须把「solo 侧有没有复核机会」对齐，否则读数不可比。

### 四、本卡自身的边界

- ③ 的代码是**测量模板**，不是论文实现。论文的任务是程序生成的谜题（网格 / 图 / 约束满足），
  你自己的业务任务需要**自己写确定性评分器**——这一步做不出来，整张卡就用不了。
- ③ 用**合成数据**跑通全链路（含四阶段规则判定器）；把规则判定器换成 LLM judge 时，
  必须先按 ⑥ A14 的口径做一次人工一致度校验。

---

## ② 母婴出海应用案例

> 数字约定：本节中**阿拉伯数字**只用于论文口径（并可在 ⑥ 段逐条核验）；
> **业务侧的建议值一律用中文数字**（如「三十个案例」），因为它们是待企业确认的假设，不是论文事实。

### 场景 1：跨境选品分析流程 —— 要不要拆成「市场调研 agent + 竞品 agent + 定价 agent」

- **业务问题**：母婴跨境团队（吸奶器、恒温调奶器、婴儿背带）现在有一个**单 Agent 选品分析流程**：
  喂进一整份类目数据，输出「这个 SKU 值不值得做」。Q4 备货决策前 8 周，有人提议拆成三个 agent 并行——
  市场调研 agent 看类目需求与季节性、竞品 agent 看 Amazon 头部 Listing 与评论、定价 agent 看价格带与毛利。
  **问题是这三个子任务必须共享同一份上下文**：同一个 ASIN 的 FBA 头程报价、认证状态（CPSC / CE）、
  在途批次与库龄口径。拆开之后，三个 agent 各持「我这一半」的事实，就会长出论文里的 grounding friction：
  定价 agent 用的头程成本口径与市场调研 agent 用的口径不一致，**谁也没问对方**（L2 失守），
  合并出来的毛利是错的（L3 失守），而没有人回头重算（L4 失守）。**这种拆分是典型的「不该拆」。**
- **数据要求**：取**不少于三十个已结案的历史选品案例**（每个案例都有事后确定的标准答案——实际动销 / 淘汰
  结论），粒度是「案例级」，并保留每次决策时的完整上下文快照。每个案例跑四种模式：
  ① **solo-full**（单 Agent 拿合并后的全量上下文 + 一次自评复核）；② **homogeneous**（两个同模型副本，
  一个只拿「市场 / 竞品」视图、一个只拿「库存 / 认证 / 头程成本」视图，交换消息直到终止或到轮数上限）；
  ③ **heterogeneous**（强模型 + 便宜模型，两个槽位各跑一遍）；④ **all-four 干预**（在 system prompt 里
  追加 L1–L4 四条阶段子句）。每个 cell 按论文口径跑 **50 次独立 seed** 的 rollout，并且**保存完整对话转录
  + 四阶段判定标注**（判定者对模型身份与分数盲）。
- **数据可得性**：`部分可得（需补充 X）`。
  **可得**：类目数据、竞品 Listing 与评论、历史选品结论、库存与头程成本（企业自有 ERP / 广告后台 / WMS）。
  **需补充**：① **视图切分**需要先定义清楚「哪一半给谁」，并自检 union-necessary（任一视图单独都定不了
  结论）—— 这是论文四条设计原则里最容易做错的一条；② **确定性评分器**要自己写：不能只有「对 / 错」，
  还要能给出连续分（例如「结论正确 + 建议备货量落在你认可的误差带内」分别计分）；
  ③ **四阶段 judge** 需要一份标注指南 + 至少两人交叉标注（论文用四位专家、一百条分层 rollout）；
  ④ **对话转录**若历史没留，需要先加日志（见场景 2）。
- **预期产出**：（a）一张 (流程变体 × 任务类别) 的**协作税表**：拆了到底亏多少个百分点、按类别分布；
  （b）四阶段失败率对照表，直接告诉你亏在「编事实」「不问」「不合并」还是「不重算」；
  （c）用对话特征做 out-of-fold 回归，得到「不必重跑实验、只看对话就能预测税」的模型；
  （d）一份**裁决书**：拆 or 不拆；如果拆，先补哪一条阶段子句。
- **业务价值**：见 ⑤ 的 ROI 公式。本卡的钱来自两处：**避免一次本不该做的拆分**（省掉拆分后的对齐成本
  与错误决策），以及**用一条 prompt 子句回收一部分已经在付的税**。

### 场景 2：多平台补货建议 —— 「销量预测 agent + 头程/海外仓成本 agent」协商产出补货量

- **业务问题**：黑五前六周，FBA / 海外仓补货量由两个 agent 协商产出。已发生过的事故形态正是论文的 L1：
  成本 agent 在合并消息里写了一个**对方从没说过的在途批次到货日**，预测 agent 直接采信，补货量算高。
  论文的结论是：**一旦有一方编造了任何一方都没说过的值，下游任何阶段都无法挽回**（⑥ Q17）。
  好消息是这类失败**不需要换模型、不需要重训**：按论文的 all-four 干预，只在 system prompt 里追加
  四条阶段子句（每条断言注明来源；第二轮起必须提一个具体问题；提交前必须有显式合并块；
  任一方结束前另一方必须展示重算），就能回收一部分税，且论文强调**没有达到 solo 天花板**（⑥ Q28、Q30、A8）。
- **数据要求**：过去**一个完整旺季周期**（Q4 + 春节前后）的补货建议记录，粒度是「SKU × 补货批次」，
  字段至少含：预测销量、实际销量、在途批次与到货日、头程报价、海外仓仓储费、是否断货、期末库龄；
  以及**每次建议的对话转录**（这才是本卡真正分析的对象）。
- **数据可得性**：`部分可得（需补充 X）`。
  **可得**：补货建议记录与销量数据（ERP / WMS）。
  **不可得 → 需补充**：**历史对话转录通常没有留**。这是本卡最主要的落地前置条件：
  要么先加日志、用**从今天起的前若干次补货建议**做前瞻测量（样本量会限制置信度），
  要么退化成只做「评审表」——不测税，直接用四条阶段子句当规范。
- **预期产出**：四阶段失败率 + 每条子句的 leave-one-out 消融，得到「你们这个链路的主瓶颈是哪一阶段」。
  论文实测三个任务族的主瓶颈**各不相同**（路径类卡 L4、关系类卡 L1、约束类卡 L2；⑥ Q29），
  **不要照搬，必须在自家数据上重测**。
- **业务价值**：**一次 prompt 改动、零重训**的代价，换回一部分协作税；对补货链路，税每降低 1 个百分点，
  就等于少一批「因为合并消息里一个编造的到货日而多备的货」。

---

## ③ 代码模板

- 依赖：**只需要 numpy**（ridge 用闭式解、相关用秩相关，不引 scipy / sklearn；不联网、不调 LLM）。
- 结构：合成测量套件 → `collaboration_tax` / `ratio_gap` → 四阶段对话判定器 → 按 task 分组的
  out-of-fold ridge（对话特征 → 税）→ 阶段子句的 all-four / leave-one-out 消融 → 异质配对的
  「拉向强者」与 Shapley 份额 → 六个断言测试 → 业务演示。
- 与论文的对应：`collaboration_tax` 实现论文式 (1)；`judge_cascade` 实现论文 5.3 节的四个二值维度；
  `ridge_oof` 实现 5.4 节的 group k-fold 防泄漏口径；`intervention_lift` 对应 5.5 节 Table 1 的
  all-four 与 no-Lk 行；`shapley_2p` 实现论文式 (9) 与 Proposition 2/4。
- ⚠️ 代码内**所有数据都是合成示例数据**（`make_synthetic_suite` / `make_synthetic_rollouts`），
  **不是论文数据**，也不代表任何企业的真实业务。代码里的自检常数全部是合成值，
  与论文 Table 2 的数值无关；论文自己的数值只出现在 ⑥ 段。

```python
"""
多 Agent 协作税（Collaboration Tax）测量模板 —— 「这个流程到底该不该拆成多 Agent」
论文：2608.22152 The Collaboration Tax: How Much LLM Multi-Agent Systems Pay to Coordinate
底本说明：本卡底本为 PDF 抽取（arXiv 无 LaTeXML HTML），故注释里只写章节名。

论文的可测部分（本模板逐条实现，全部用**合成数据**演示，不是论文数据）：
  1) tax = solo-full 得分 − 同质配对得分（论文式 (1)），并给出按 solo 归一化的 ratio gap；
  2) 四阶段对话级联判定器 L1 grounding / L2 querying / L3 integration / L4 re-derivation
     （论文 5.3 节把每个阶段定义成二值 judge 维度）；
  3) 用**对话结构特征**离线预测 tax（论文 5.4 节的 ridge + 按 task 分组的 out-of-fold 评估）；
  4) 阶段定向 prompt 干预的 leave-one-out 消融（论文 5.5 节 / Table 1 的口径）；
  5) 异质配对的 tax 是否被**拉向更强的一方**而非两者中点，以及 Shapley 份额是否低于单人收益
     （论文 6 节 / Appendix A.5）。

依赖：仅 numpy + 标准库。无网络、无 LLM SDK、无 matplotlib。
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# 0. 常量：只放论文口径的结构性设定，不放论文的具体数值结论
# ---------------------------------------------------------------------------
CATEGORIES = ("Spatial", "Relational", "CSP")   # 论文 4.2 节的三个任务族
STAGES = ("L1_grounded", "L2_queried", "L3_integrated", "L4_rederived")
STAGE_KEYS = ("L1", "L2", "L3", "L4")


# ---------------------------------------------------------------------------
# 1. 核心量：协作税 tax 与归一化的 ratio gap
# ---------------------------------------------------------------------------
def collaboration_tax(solo_full: float, paired: float) -> float:
    """tax = s_solo-full − s_pair（论文式 (1) 的逐 cell 形式）。

    solo_full : 单 Agent 拿到**合并后的完整实例**（含一次 critic 复核）的连续得分
    paired    : 两个 Agent 各拿一半私有视图、对话到底之后的连续得分
    正数 = 协作亏了；负数 = 配对反而赢过单 Agent（论文 Appendix A.7 记录了这种负税 cell）。
    """
    return float(solo_full) - float(paired)


def ratio_gap(solo_full: float, paired: float, eps: float = 1e-9) -> float:
    """tax / s_solo-full —— 论文用它跨任务归一化（不同任务的 solo 天花板不同）。"""
    if solo_full <= eps:
        raise ValueError("solo_full 必须为正，否则 ratio gap 无定义")
    return collaboration_tax(solo_full, paired) / float(solo_full)


def tax_table(solo: np.ndarray, paired: np.ndarray) -> dict:
    """整表口径：tax 与 ratio gap 的均值/中位数/负税比例。"""
    solo = np.asarray(solo, dtype=float)
    paired = np.asarray(paired, dtype=float)
    tax = solo - paired
    r = tax / np.maximum(solo, 1e-9)
    return {"tax": tax, "ratio_gap": r, "mean_tax": float(tax.mean()),
            "mean_ratio_gap": float(r.mean()), "negative_share": float((tax < 0).mean())}


# ---------------------------------------------------------------------------
# 2. 四阶段级联判定器（论文 5.3 节四个二值维度；此处用规则版替掉 LLM judge）
# ---------------------------------------------------------------------------
def judge_cascade(turns: list[dict], my_view_facts: set, partner_view_facts: set) -> dict:
    """把一个 rollout 的对话判成四个二值维度。

    输入 turns 的每一项形如：
        {"speaker": 0/1, "text": str, "cited": set, "asked": set, "combined": set}
      cited    : 该轮声称的事实，且给出了来源（“from my view” / “as [other agent] said”）
      asked    : 该轮向对方提出的**具体事实问题**涉及的事实键
      combined : 该轮显式列出的“合并状态”事实键

    L1 grounded   : 每一句事实断言都能回溯到某一方视图（论文：一旦编造，下游无法挽回）
    L2 queried    : 至少一方提出过具体事实问题
    L3 integrated : 决定性断言之前出现过显式的合并状态消息
    L4 rederived  : 接收方在任一方结束前展示了**重算过程**（重走路径/重算/重查约束）
    """
    stated = set()
    ungrounded = 0
    queried = False
    integrated = False
    rederived = False
    for t in turns:
        known = my_view_facts | partner_view_facts
        for fact in t.get("cited", set()):
            stated.add(fact)
            if fact not in known:
                ungrounded += 1
        if t.get("asked"):
            queried = True
        if t.get("combined"):
            integrated = True
        if t.get("recompute"):
            rederived = True
    return {"L1_grounded": ungrounded == 0,
            "L2_queried": queried,
            "L3_integrated": integrated,
            "L4_rederived": rederived,
            "ungrounded_claims": ungrounded,
            "n_turns": len(turns)}


def cascade_fire_rates(judged: list[dict], outcomes: list[bool]) -> dict:
    """失败 rollout 与成功 rollout 在四个维度上的“满足率”，用来做判别力体检。

    论文 5.3 节的关键结论是：**四个阶段都能区分失败与成功**，其中 L1 是最干净的
    单特征判别器、L3 是单变量最强预测器。这里把同一套统计在自家数据上重算一遍。
    """
    out = {}
    for key in STAGES:
        fail = [j[key] for j, ok in zip(judged, outcomes) if not ok]
        succ = [j[key] for j, ok in zip(judged, outcomes) if ok]
        out[key] = {"fail_rate": float(np.mean(fail)) if fail else float("nan"),
                    "success_rate": float(np.mean(succ)) if succ else float("nan")}
    out["n_fail"] = int(sum(1 for ok in outcomes if not ok))
    out["n_success"] = int(sum(1 for ok in outcomes if ok))
    return out


def multi_stage_fire_share(judged: list[dict], outcomes: list[bool]) -> float:
    """失败 rollout 里“至少两个阶段同时失守”的占比（论文 Appendix D 的 67% 口径）。"""
    fails = [j for j, ok in zip(judged, outcomes) if not ok]
    if not fails:
        return float("nan")
    n_multi = sum(1 for j in fails if sum(1 for k in STAGES if not j[k]) >= 2)
    return n_multi / len(fails)


def jaccard_overlap(judged: list[dict], outcomes: list[bool], a: str, b: str) -> float:
    """两个阶段在失败集合上的 Jaccard 重叠（论文 Appendix D 用 L3×L4 的 0.67 说明纠缠）。"""
    fails = [j for j, ok in zip(judged, outcomes) if not ok]
    sa = {i for i, j in enumerate(fails) if not j[a]}
    sb = {i for i, j in enumerate(fails) if not j[b]}
    union = sa | sb
    return len(sa & sb) / len(union) if union else float("nan")


# ---------------------------------------------------------------------------
# 3. 用对话特征预测 tax（论文 5.4 节的 ridge + 按 task 分组的 out-of-fold）
# ---------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ = 秩上的 Pearson 相关（不引 scipy）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> dict:
    """闭式 ridge：beta = (X'X + lam I)^-1 X'y，输入先标准化（论文同款设定 lam = 1）。

    ⚠️ 标准化统计量（mu/sd）**必须随模型一起返回**：若在预测时用测试折自己的
    均值方差重新标准化，等价于把测试折的信息泄漏进特征尺度，实测会把 out-of-fold
    R² 打成负数（本模板第一版就踩了这个坑）。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), 1e-9)
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    # 截距必须显式存下来：标准化后的 Xs 均值为 0，若同时把 y 去均值再拟合，
    # 预测时忘了加回 y 的均值，相关系数看着很好、R² 却是负的（本模板第二版踩的坑）。
    return {"beta": np.linalg.solve(A, Xs.T @ (y - y.mean())),
            "mu": mu, "sd": sd, "intercept": float(y.mean())}


def predict_ridge(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return model["intercept"] + ((X - model["mu"]) / model["sd"]) @ model["beta"]


def ridge_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """按 group（这里是 task）分折的 out-of-fold 预测 —— 论文用 group k-fold 防泄漏。

    返回 OOF R²、OOF Spearman ρ、OOF Pearson r。R² 以 y 的均值为参照。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    pred = np.zeros_like(y)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        model = fit_ridge(X[tr], y[tr])
        pred[te] = predict_ridge(model, X[te])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"pred": pred, "r2": r2, "spearman": spearman(y, pred),
            "pearson": float(np.corrcoef(y, pred)[0, 1])}


# ---------------------------------------------------------------------------
# 4. 阶段定向 prompt 干预：all-four 与 leave-one-out（论文 5.5 节 / Table 1）
# ---------------------------------------------------------------------------
def intervention_lift(category: str, dropped_stage: str | None,
                      base_tax: float, bottleneck: dict, responsiveness: dict,
                      off_bottleneck_keep: float = 0.85) -> dict:
    """把「加 4 条阶段子句」折算成同质成功率的提升。

    responsiveness[category] —— 该类别对 all-four 干预的整体响应度
    bottleneck[category]     —— 该类别的主瓶颈阶段；丢掉它的子句，提升基本归零
    返回 Δs_homo 与 % closed（回收了原始 tax 的多少）。
    """
    if dropped_stage is None:
        lift = responsiveness[category]
    elif dropped_stage == bottleneck[category]:
        lift = 0.0
    else:
        lift = responsiveness[category] * off_bottleneck_keep
    lift = float(np.clip(lift, 0.0, base_tax))
    return {"delta_s_homo": lift,
            "pct_closed": (lift / base_tax * 100.0) if base_tax > 0 else float("nan")}


# ---------------------------------------------------------------------------
# 5. 异质配对：tax 被拉向强者，以及 Shapley 份额低于单人收益
# ---------------------------------------------------------------------------
def hetero_gap(strong_gap: float, weak_gap: float, pull: float = 0.7) -> float:
    """异质配对的 tax 被拉向**更强的一方**（而非两者中点）。

    论文 6 节：实际 hetero gap 与中点强相关，但系统性**低于**中点（更接近强者的较小 gap）。
    pull = 0 → 恰好是中点；pull = 1 → 完全等于强者的 gap。
    """
    midpoint = 0.5 * (strong_gap + weak_gap)
    return midpoint - pull * 0.5 * (weak_gap - strong_gap)


def shapley_2p(v_self: float, v_other: float, v_joint: float) -> tuple:
    """两人合作博弈的 Shapley 值（论文式 (9)）：phi_i = 0.5 * (v_i + v_joint − v_j)。"""
    return (0.5 * (v_self + v_joint - v_other),
            0.5 * (v_other + v_joint - v_self))


def max_superadditivity_violated(v_self: float, v_other: float, v_joint: float) -> bool:
    """违反 max-superadditivity ⇔ tax > 0（论文 Proposition 2）。"""
    return v_joint < max(v_self, v_other)


# ---------------------------------------------------------------------------
# 6. 合成测量套件（**不是论文数据**：只复刻论文的定性结构，用来跑通模板）
# ---------------------------------------------------------------------------
def make_synthetic_suite(n_models: int = 4, n_tasks_per_cat: int = 3,
                         n_rollouts: int = 40, seed: int = 17) -> dict:
    """构造 (model, task) 测量矩阵。

    复刻论文 5.2 节的两条无例外轴：
      轴 1（任务类别）：Spatial 的 tax 最大 > Relational > CSP 最小；
      轴 2（模型能力）：tax 随能力单调下降。
    合成规则：ratio_gap ≈ headroom[category] × (1.05 − capability)。
    """
    rng = np.random.default_rng(seed)
    caps = np.linspace(0.35, 0.92, n_models)
    headroom = {"Spatial": 0.55, "Relational": 0.38, "CSP": 0.22}
    rows = []
    for mi, cap in enumerate(caps):
        for cat in CATEGORIES:
            for ti in range(n_tasks_per_cat):
                difficulty = 0.93 + 0.05 * rng.random()          # 任务本身难度微扰
                solo = float(np.clip(cap * difficulty, 0.0, 1.0))
                rg = headroom[cat] * max(1.05 - cap, 0.02)
                rg = float(np.clip(rg + rng.normal(0.0, 0.012), 0.0, 0.6))
                paired = solo * (1.0 - rg)
                severity = rg / headroom["Spatial"]              # 归一化后的级联严重度
                rows.append({"model": mi, "cap": cap, "category": cat,
                             "task": f"{cat[:3].lower()}{ti}", "solo": solo,
                             "paired": paired, "severity": severity})
    # 每个 cell 的对话特征：四阶段通过率 + 体量类特征（体量类不携带 tax 信号）
    for r in rows:
        sev = r["severity"]
        r["L1_grounded_rate"] = float(np.clip(1.0 - 0.80 * sev + rng.normal(0, .04), 0, 1))
        r["L2_queried_rate"] = float(np.clip(1.0 - 0.60 * sev + rng.normal(0, .06), 0, 1))
        r["L3_integrated_rate"] = float(np.clip(1.0 - 0.95 * sev + rng.normal(0, .04), 0, 1))
        r["L4_rederived_rate"] = float(np.clip(1.0 - 0.90 * sev + rng.normal(0, .04), 0, 1))
        r["chars_per_turn"] = float(420 + 60 * rng.normal())
        r["view_dump_ratio"] = float(np.clip(0.45 + 0.08 * rng.normal(), 0, 1))
        if r["category"] == "CSP" and r["cap"] > 0.85:           # 论文 Appendix A.7 的负税 cell
            r["paired"] = min(1.0, r["paired"] * 1.35)
    return {"rows": rows, "caps": caps, "headroom": headroom}


def feature_matrix(rows: list[dict]) -> tuple:
    """对话结构特征矩阵 + 目标 ratio gap + 分组键（task）。"""
    feats = ["L1_grounded_rate", "L2_queried_rate", "L3_integrated_rate",
             "L4_rederived_rate", "chars_per_turn", "view_dump_ratio"]
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([ratio_gap(r["solo"], r["paired"]) for r in rows], float)
    groups = np.array([r["task"] for r in rows])
    return X, y, groups, feats


def make_synthetic_rollouts(severity: float, n: int = 60, seed: int = 3) -> tuple:
    """按严重度生成 rollout 级的四阶段 flag（失守概率随 severity 上升）。"""
    rng = np.random.default_rng(seed)
    p_ok = {"L1_grounded": 1 - 0.80 * severity, "L2_queried": 1 - 0.60 * severity,
            "L3_integrated": 1 - 0.95 * severity, "L4_rederived": 1 - 0.90 * severity}
    judged, outcomes = [], []
    for _ in range(n):
        j = {k: bool(rng.random() < np.clip(v, 0.02, 0.999)) for k, v in p_ok.items()}
        # 论文 Appendix D：L3 与 L4 高度纠缠（失败集合 Jaccard 0.67）；合成数据里
        # 用「L3 失守则 L4 大概率也失守」复刻这一依赖，否则两阶段会被人为独立化。
        if not j["L3_integrated"] and rng.random() < 0.9:
            j["L4_rederived"] = False
        judged.append(j)
        outcomes.append(all(j.values()))
    return judged, outcomes


# ---------------------------------------------------------------------------
# 7. 测试
# ---------------------------------------------------------------------------
def test_tax_is_the_solo_minus_pair_difference():
    assert collaboration_tax(0.90, 0.63) == 0.27
    assert abs(ratio_gap(0.90, 0.63) - 0.3) < 1e-9
    # 负税 cell：配对反而赢（论文 Appendix A.7 承认这种 cell 存在）
    assert collaboration_tax(0.40, 0.55) < 0
    tbl = tax_table(np.array([0.90, 0.80, 0.40]), np.array([0.63, 0.60, 0.55]))
    assert tbl["ratio_gap"].shape == (3,)
    assert tbl["negative_share"] == 1.0 / 3.0


def test_cascade_judge_separates_failure_from_success():
    view_a, view_b = {"n1", "n2"}, {"n3", "n4"}
    good = [{"speaker": 0, "cited": {"n1"}, "asked": {"n3"}, "combined": {"n1", "n3"}},
            {"speaker": 1, "cited": {"n3"}, "recompute": True}]
    # L1：编造了任何一方视图里都没有的 n9（论文：一旦编造，下游救不回来）
    bad_l1 = [{"speaker": 0, "cited": {"n1", "n9"}, "asked": {"n3"}, "combined": {"n1", "n3"}},
              {"speaker": 1, "recompute": True}]
    # L2：全程不问（论文 “L2 failures are silences”）
    bad_l2 = [{"speaker": 0, "cited": {"n1"}, "combined": {"n1"}}, {"speaker": 1, "cited": {"n3"}}]
    # L3 / L4：没合并就抢着结束
    bad_l34 = [{"speaker": 0, "cited": {"n1"}}, {"speaker": 1, "cited": {"n3"}}]
    jg = judge_cascade(good, view_a, view_b)
    assert all(jg[k] for k in STAGES) and jg["ungrounded_claims"] == 0
    assert not judge_cascade(bad_l1, view_a, view_b)["L1_grounded"]
    assert not judge_cascade(bad_l2, view_a, view_b)["L2_queried"]
    assert not judge_cascade(bad_l34, view_a, view_b)["L3_integrated"]
    assert not judge_cascade(bad_l34, view_a, view_b)["L4_rederived"]

    judged, outcomes = make_synthetic_rollouts(severity=0.5)
    rates = cascade_fire_rates(judged, outcomes)
    for k in STAGES:                      # 四个阶段都必须能区分失败与成功
        assert rates[k]["success_rate"] > rates[k]["fail_rate"] + 0.05
    assert 0.0 <= multi_stage_fire_share(judged, outcomes) <= 1.0
    assert jaccard_overlap(judged, outcomes, "L3_integrated", "L4_rederived") > 0.5


def test_tax_is_predictable_from_conversation_features():
    data = make_synthetic_suite()
    X, y, groups, feats = feature_matrix(data["rows"])
    res = ridge_oof(X, y, groups)
    assert res["r2"] > 0.5, f"OOF R² 过低: {res['r2']}"
    assert res["spearman"] > 0.6, f"OOF ρ 过低: {res['spearman']}"
    assert len(feats) == X.shape[1]
    # 加类别哑变量后不应变差（论文 5.4 节：能力设截距、对话形状设斜率）
    cats = sorted({r["category"] for r in data["rows"]})
    Xd = np.hstack([X, np.array([[1.0 if r["category"] == c else 0.0
                                  for c in cats[:-1]] for r in data["rows"]])])
    res2 = ridge_oof(Xd, y, groups)
    assert res2["r2"] >= res["r2"] - 0.05


def test_stage_clauses_close_part_of_the_gap_and_bottlenecks_differ():
    bottleneck = {"Spatial": "L4", "Relational": "L1", "CSP": "L2"}
    responsiveness = {"Spatial": 0.05, "Relational": 0.07, "CSP": 0.11}
    base_tax = {"Spatial": 0.19, "Relational": 0.15, "CSP": 0.21}
    all_four = {c: intervention_lift(c, None, base_tax[c], bottleneck, responsiveness)
                for c in CATEGORIES}
    # 响应度：CSP 最强 > Relational > Spatial 最弱（论文 5.5 节）
    assert all_four["CSP"]["delta_s_homo"] > all_four["Relational"]["delta_s_homo"]
    assert all_four["Relational"]["delta_s_homo"] > all_four["Spatial"]["delta_s_homo"]
    for c in CATEGORIES:
        assert all_four[c]["pct_closed"] > 0
        # 丢掉本类别的主瓶颈子句 → 提升基本归零（论文：每个类别的瓶颈层不同）
        drop = intervention_lift(c, bottleneck[c], base_tax[c], bottleneck, responsiveness)
        assert drop["delta_s_homo"] == 0.0
        # 丢掉非瓶颈子句 → 仍有大部分提升
        other = [s for s in STAGE_KEYS if s != bottleneck[c]][0]
        keep = intervention_lift(c, other, base_tax[c], bottleneck, responsiveness)
        assert keep["delta_s_homo"] > 0.5 * all_four[c]["delta_s_homo"]
    # 干预不可能超过原始 tax（论文：没有任何条件达到 solo 天花板）
    assert all(v["delta_s_homo"] <= base_tax[c] for c, v in all_four.items())


def test_hetero_pair_is_pulled_toward_the_stronger_partner():
    strong_gap, weak_gap = 0.07, 0.38
    mid = 0.5 * (strong_gap + weak_gap)
    g = hetero_gap(strong_gap, weak_gap, pull=0.7)
    assert g < mid                                   # 低于加性中点
    assert g > strong_gap                            # 但仍比强者单独做更差
    assert abs(hetero_gap(strong_gap, weak_gap, pull=0.0) - mid) < 1e-12
    assert abs(hetero_gap(strong_gap, weak_gap, pull=1.0) - strong_gap) < 1e-12
    # 跨多组强弱对比：全部低于中点，且与中点强相关
    pairs = [(0.05, 0.30), (0.07, 0.38), (0.10, 0.45), (0.12, 0.52)]
    mids = np.array([0.5 * (s + w) for s, w in pairs])
    gaps = np.array([hetero_gap(s, w) for s, w in pairs])
    assert np.all(gaps < mids)
    assert spearman(mids, gaps) > 0.9


def test_shapley_share_falls_below_singleton_and_max_superadditivity_breaks():
    # 合成常数（**不是论文 Table 2 的数值**）：只用来验证「强-弱配对 + 弱于加性」这一结构。
    v_strong, v_weak, v_joint = 0.90, 0.35, 0.70
    assert max_superadditivity_violated(v_strong, v_weak, v_joint)
    phi_strong, phi_weak = shapley_2p(v_strong, v_weak, v_joint)
    assert phi_strong < v_strong and phi_weak < v_weak     # 双方份额都低于单人收益
    assert abs((phi_strong + phi_weak) - v_joint) < 1e-12  # 效率公理
    assert phi_strong > phi_weak                           # 但强者拿大头
    # 强-弱槽位互换后，份额对称地跟着槽位走（论文 Table 2 的 swap 口径）
    a, b = shapley_2p(v_weak, v_strong, v_joint)
    assert abs(a - phi_weak) < 1e-12 and abs(b - phi_strong) < 1e-12


# ---------------------------------------------------------------------------
# 8. 业务演示：一次「该不该拆」的裁决
# ---------------------------------------------------------------------------
def _business_demo() -> None:
    data = make_synthetic_suite()
    rows = data["rows"]
    tbl = tax_table([r["solo"] for r in rows], [r["paired"] for r in rows])
    print("== 1. 协作税总览（合成数据，非论文数据）==")
    print(f"   cell 数={len(rows)}  平均 tax={tbl['mean_tax']:.3f}  "
          f"平均 ratio gap={tbl['mean_ratio_gap']:.3f}  "
          f"负税 cell 占比={tbl['negative_share']:.1%}")

    print("\n== 2. 两条轴（类别序 / 能力单调性）==")
    for cat in CATEGORIES:
        sub = [ratio_gap(r["solo"], r["paired"]) for r in rows if r["category"] == cat]
        print(f"   {cat:11s} mean ratio gap = {np.mean(sub):.3f}")
    caps, gaps = [], []
    for mi in sorted({r["model"] for r in rows}):
        sub = [ratio_gap(r["solo"], r["paired"]) for r in rows if r["model"] == mi]
        caps.append(rows[mi * len(CATEGORIES) * 3]["cap"])
        gaps.append(float(np.mean(sub)))
    print("   能力 → gap：" + "  ".join(f"{c:.2f}→{g:.3f}" for c, g in zip(caps, gaps)))
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1)), "能力轴不单调"

    print("\n== 3. 四阶段级联（对话特征能否机械预测 tax）==")
    judged, outcomes = make_synthetic_rollouts(severity=0.5, seed=5)
    rates = cascade_fire_rates(judged, outcomes)
    for k in STAGES:
        print(f"   {k:14s} 失败满足率={rates[k]['fail_rate']:.2f}  "
              f"成功满足率={rates[k]['success_rate']:.2f}")
    print(f"   失败中至少 2 个阶段同时失守的占比 = {multi_stage_fire_share(judged, outcomes):.1%}")
    print(f"   L3×L4 失败集合 Jaccard = "
          f"{jaccard_overlap(judged, outcomes, 'L3_integrated', 'L4_rederived'):.2f}")
    X, y, groups, _ = feature_matrix(rows)
    res = ridge_oof(X, y, groups)
    print(f"   对话特征 → ratio gap：OOF R²={res['r2']:.3f}  ρ={res['spearman']:.3f}  "
          f"r={res['pearson']:.3f}")

    print("\n== 4. 裁决：这个流程该不该拆 ==")
    bottleneck = {"Spatial": "L4", "Relational": "L1", "CSP": "L2"}
    responsiveness = {"Spatial": 0.05, "Relational": 0.07, "CSP": 0.11}
    base_tax = {"Spatial": 0.19, "Relational": 0.15, "CSP": 0.21}
    for cat in CATEGORIES:
        r = intervention_lift(cat, None, base_tax[cat], bottleneck, responsiveness)
        print(f"   {cat:11s} all-four 干预：Δs={r['delta_s_homo']:.3f}  "
              f"回收 tax={r['pct_closed']:.1f}%  主瓶颈={bottleneck[cat]}")

    print("\n== 5. 异质配对：tax 被拉向强者 ==")
    for s, w in [(0.07, 0.38), (0.10, 0.45)]:
        print(f"   强者 gap={s:.2f} 弱者 gap={w:.2f} → 中点={0.5*(s+w):.3f} "
              f"实际 hetero gap={hetero_gap(s, w):.3f}")
    phi_s, phi_w = shapley_2p(0.90, 0.35, 0.70)
    print(f"   Shapley：强者 {phi_s:.3f}（单人 0.90）  弱者 {phi_w:.3f}（单人 0.35）")


if __name__ == "__main__":
    _business_demo()
```

---

## ④ 技能关联

- **前置判据｜`Skill-Subagent-Decomposition.md`**（同目录）：那张卡解决「怎么把一个任务切成 N 个子任务」。
  本卡是它的**前置闸门**：任何切分方案先过 union-necessary 检查（任一子视图单独能否得出结论），
  否则论文 Proposition 1 的退化情形会让税恒为 0，切了纯亏。数据流：本卡 ③ 的 `collaboration_tax()`
  输出「切 vs 不切」的差值，直接喂给该卡的切分粒度决策。
- **对照 / 反例来源｜`Skill-MetaGPT-SOP-Driven-Collaboration.md`**（同目录）：SOP 驱动的角色分工默认
  「分工 → 增益」。本卡提供反例度量：分工只有在存在 grounding friction（同一内容有多种等价表示）时
  才可能增益，否则 `v({1,2}) < max(v({1}), v({2}))` 会直接发生。组合方式：先用本卡量税，
  再决定 SOP 里哪些角色值得保留。
- **可组合｜`Skill-Multi-Agent-Debate.md`**（同目录）：辩论是另一种协作协议，**同样要付协作税**。
  本卡的测量协议（solo-full / homogeneous / all-four）可以把「辩论 vs 单 Agent」放在同一把尺子上比，
  而不是只比辩论内部的几种变体。
- **可组合｜`Skill-MAS-Orchestrator.md`**（同目录）：编排层决定消息怎么路由、回合怎么终止。
  本卡的四阶段诊断器回答「路由之外还差什么」——L2 缺失说明没有强制提问机制，L4 缺失说明没有强制复核回合，
  L3 缺失说明没有强制合并块。这三条恰好是编排层可以直接落地的三个钩子。
- **分阶段评测参考｜`Skill-Agent-Stage-Evaluation.md`**（16-智能体工程）：那张卡做的是电商 Agent 的
  阶段化评测（感知 / 规划 / 行动）。本卡的四阶段 judge 是同一思路在**对话层**的实例，
  两者可拼成「任务阶段 × 对话阶段」的双轴评测。
- **可组合｜`Skill-Context-Compression.md`**（16-智能体工程）：上下文压缩会改变「视图切分」的形状，
  进而改变 grounding friction 的大小；本卡的税可作为压缩策略选型的评估指标之一。

---

## ⑤ 商业价值评估

**ROI 公式**（本次不代入任何未经论文或企业数据支持的数字）：

ROI = (ΔS × V_decision × N_case − C_measure) / C_measure

| 参数 | 含义 | 来源 |
|---|---|---|
| `ΔS` | 协作税：拆与不拆（或加不加阶段子句）在主指标上的成功率差 | **必须企业自测**（按 ③ 的模板）。论文的量级只能当参考：all-four 干预在三个类别上的置信区间下界都高于零，但**没有任何条件达到 solo 天花板**（⑥ Q27、Q28）；且论文场景不是母婴跨境（⑥ Q36） |
| `V_decision` | 单个选品 / 补货决策「做对」与「做错」的金额差 | 企业自有财务口径（毛利、滞销损失、断货损失） |
| `N_case` | 一年内该类决策的案例数 | 企业自有口径 |
| `C_measure` | 测量成本 = 历史案例回放的人工 + 对话日志埋点 + 四阶段标注 + 评分器开发 | **论文未报告任何成本量级**，需企业自估。论文只报告了评测侧的规模（每个 cell 50 次 rollout、对话上限 50 轮交换；⑥ Q2、A4、A5），未报告费用 |

**为什么这张卡不给收益金额结论**：收益侧的关键参数 `ΔS` 在论文里是在**程序生成的谜题**上测出来的，
论文自己说该设定不能直接对应生产工作负载（⑥ Q36）；把它当预测值代入会得到一个看起来很确定、
实际没有依据的数字。本卡的价值主张是**决策口径的修正**——
把「多 Agent 一定更好」的默认假设，换成「先量税、再看亏在哪一阶段、再决定拆不拆」。

**可复现的算例骨架**（符号版，可自行代入）：把上式写成三步——
① `年化增益 = ΔS × V_decision × N_case − C_measure`；
② `ROI = 年化增益 / C_measure`；
③ 决策规则：只有当 `ΔS` 显著为正（置信区间下界高于零，对应论文 Table 1 的读法）时才立项。

- **实施难度**：⭐⭐⭐☆☆ —— 算法本身很轻（③ 只用 numpy，一天能跑通）；
  真正的成本在**业务侧的三件事**：确定性评分器、视图切分定义、四阶段标注指南。
  如果只做「四阶段子句 + 人工抽检」，难度可降到 ⭐⭐☆☆☆。
- **优先级**：⭐⭐⭐⭐☆ —— 只要团队里已经有人在提「要不要拆多 Agent」，这张卡的优先级就是最高的：
  它便宜、它前置、它能拦住一个方向性错误。反过来，如果团队目前只有一个 Agent 且效果可接受，
  优先级降到 ⭐⭐☆☆☆（先当规范用，别当项目做）。
- **评估依据**：论文的核心贡献是把税变成了**可预测、部分可解**的量（⑥ Q23、Q28），
  且解法是**一次 prompt 改动**（⑥ Q30）；这意味着本卡的投入产出比主要由「你愿不愿意先测量」决定，
  而不是由工程规模决定。

---

## ⑥ 原文引用

> **底本说明（重要）**：`2608.22152` 在 arXiv 上没有 HTML（LaTeXML）存档 —— v1 / v2 / v3 三个版本
> 实测均取不到，只能取 PDF。本卡底本因此是 `pdftotext` 从 PDF 抽取、再把句内硬换行接回整段得到的
> （全文存档：`papers/` 域目录下本论文目录的 `fulltext.md`，头部 `source` 指向该 PDF）。
> 因此**没有 HTML 章节锚点**，下面出处一律写**章节名 / 小节名**；
> 文中出现的 `5.3`、`A.4`、`Table 2` 一类编号，都是底本正文里**确实印出来的**编号或交叉引用，
> 不是从 HTML 结构反推的。所有引文都是**从底本按起止锚点直接切出的连续子串**，
> 未改标点、未改词、未把两句话缝成一句（由 `quote_check.py` 逐字核验）。

### A. 税的定义与「该不该拆」的理论条件

> 原文："We formulate the collaboration tax as the team-decentralisation loss of a two-player cooperative game with private information, with two propositions characterising its sign and its equivalence to a max-superadditivity violation."
> 出处：2608.22152 Abstract｜Q1

> 原文："Operational form. For a task T with instances x scored by a deterministic grader U ∈ [0, 1], and a union-necessary partition x = v1 (x) ∪ v2 (x) such that neither view alone determines the answer, the homogeneous tax of model M is c tax(M, T ) = ssolo-full (M, T ) − shomo (M, T ), (1) where ssolo-full is the mean score of M given the merged instance and shomo is the mean score of two copies of M given v1 and v2 exchanging messages until termination, each averaged over 50 rollouts with independent seeds."
> 出处：2608.22152 3 The Collaboration Tax（Operational form 段）｜Q2

> 原文："This is precisely the failure of max-superadditivity for the cooperative game (N, v). A coalition that satisfies max-superadditivity produces at least as much joint utility as its strongest member acting alone."
> 出处：2608.22152 Appendix A.4 Max-superadditivity equivalence｜Q3

> 原文："If the partition is degenerate, say v1 = x, then the paired protocol Π can ignore v2 and emulate the solo policy exactly, producing Vpair = Vsolo and forcing tax = 0 by construction regardless of coordination ability."
> 出处：2608.22152 Appendix A.6 Design principles as theoretical requirements｜Q4

> 原文："If the views are trivially mergeable, for instance one agent serialising its view to the other in a canonical form that both agents share, then Π can directly emulate the centralised baseline and the upper bound binds with equality at zero coordination effort. Multiple equivalent surface representations break this trivial pass-through: agents grounded in different schemes (origin, axis order, naming convention) cannot simply concatenate their views without first aligning representations, so the upper bound is approached only by competent coordination, and the tax becomes a measure of that competence."
> 出处：2608.22152 Appendix A.6 Design principles as theoretical requirements｜Q5

> 原文："Solo-trivial: with the full instance a single agent should solve the task at a high rate, so that ssolo-full is near ceiling and the gap reflects coordination cost rather than problem-solving capacity."
> 出处：2608.22152 4.1 Design Principles for the Task Suite｜Q8

> 原文："Union-necessary: each instance is partitioned into views v1 , v2 with v1 ∪ v2 = x and neither view alone admits the canonical answer."
> 出处：2608.22152 4.1 Design Principles for the Task Suite｜Q9

> 原文："Multiply expressible: the same content admits several equivalent surface representations (coordinate origins, axis orderings, naming conventions, ordinal directions, relational vocabularies), providing the grounding friction we aim to measure"
> 出处：2608.22152 4.1 Design Principles for the Task Suite｜Q10

> 原文："a single misaligned step invalidates the rest of a path (Wang et al., 2026), relational query errors stay local to their query, and constraint violations cascade through the assignment."
> 出处：2608.22152 4.2 Task Families｜Q11

> 原文："These deployments treat collaboration as a free primitive: assemble enough capable models, give them clear roles, and the team will outperform any single member. The premise is rarely tested directly."
> 出处：2608.22152 1 Introduction｜Q12

### B. 任务与模型设定 / 测量协议

> 原文："We operationalise this definition on 32 solo-tractable tasks grouped by source of grounding friction and measure it on 11 models from 7 providers."
> 出处：2608.22152 Abstract｜Q6

> 原文："We evaluate eleven models from seven providers: OpenAI (gpt-5, gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini), Anthropic (claude-sonnet-4-5), Google (gemini-2.5-flash-lite), DeepSeek (DeepSeek-V4-Pro), and three open-weight models hosted through API endpoints: Llama-4-Maverick (Meta, mixture-of-experts), Phi-4 (Microsoft), and Qwen3-8B (Alibaba)."
> 出处：2608.22152 5.1 Experimental Setup｜Q7

> 原文："We run 50 independent rollouts per (task, mode, model or pair) cell, with consecutive integer seeds controlling both instance generation and the partition into views."
> 出处：2608.22152 Appendix I Hyperparameters（Generation 段）｜A4

> 原文："The collaborative dialogue is capped at 50 exchanges per rollout, where one exchange is a turn from each agent. Each rollout starts from an empty context, so no state leaks across rollouts."
> 出处：2608.22152 Appendix I Hyperparameters（Generation 段）｜A5

> 原文："The grader is a fixed model (gpt-4o-mini in our experiments), held constant across every cell of the design. In particular, the grader does not change when the agents do, so heterogeneous-pair comparisons are not confounded by grader-side capability differences."
> 出处：2608.22152 Appendix I Hyperparameters（Grading 段）｜A6

### C. 两条「无例外」的轴

> 原文："two patterns hold without exception across the eleven models. Within every row, the ordering is Spatial ≻ Relational ≻ CSP: spatialcoordination tasks lose the most from collaboration, relational queries lose less, and constraintsatisfaction tasks lose least."
> 出处：2608.22152 5.2 The Gap Landscape｜Q13

> 原文："the gap scales monotonically with model capability: the weakest models lose roughly half of their solo success to coordination"
> 出处：2608.22152 5.2 The Gap Landscape｜Q14

> 原文："The three weakest rows come from three different model families, so the capability ordering is not a family-style artefact."
> 出处：2608.22152 5.2 The Gap Landscape｜Q15

### D. 机制：四阶段对话级联

> 原文："The proximate mechanism is not a reasoning deficit but a four-stage conversational cascade in which agents make ungrounded claims, fail to query the partner, skip integrating both views, and accept the answer without re-derivation."
> 出处：2608.22152 Abstract｜Q16

> 原文："From 700 openended LLM failure descriptions clustered under a neutral prompt and an anti-bias naming rule (Appendix B), we extract 16 behaviourally specific themes"
> 出处：2608.22152 5.3 Mechanism: A Four-Stage Cascade｜A22

> 原文："L1 Grounding. A claim is grounded if it can be traced to information stated by either agent. Panel L1 of Figure 2 shows that grounding is the cleanest single-feature fail/success discriminator in the judge: successful rollouts are grounded in essentially every category, while a substantial fraction of failures contain at least one ungrounded claim."
> 出处：2608.22152 5.3 Mechanism: A Four-Stage Cascade（L1 Grounding 段）｜A1

> 原文："L2 Querying. A pair queries iff at least one agent makes a specific factual request of the partner (“what is the value of node K?”). Querying discriminates failure from success across all three categories (panel L2 of Figure 2), with the largest gap on CSP"
> 出处：2608.22152 5.3 Mechanism: A Four-Stage Cascade（L2 Querying 段）｜A2

> 原文："(panel L2 of Figure 2), with the largest gap on CSP, where successful pairs explicitly elicit cross-half capacity and constraint facts that failed pairs leave latent."
> 出处：2608.22152 5.3 Mechanism: A Four-Stage Cascade（L2 Querying 段）｜A2b

> 原文："L3 Integration. A pair integrates iff the decisive claim is preceded by an explicit combined-state message that lists facts from both views and any derived consequences. Panel L3 of Figure 2 shows that integration is the strongest single-variable predictor of the collaboration tax and the only stage whose marginal contribution to a multi-feature regression is positive."
> 出处：2608.22152 5.3 Mechanism: A Four-Stage Cascade｜Q18

> 原文："L4 Re-derivation. A pair re-derives iff the receiving agent shows actual recomputation work (rewalks the path, recomputes the sum, re-checks the constraints) before either agent wants to end."
> 出处：2608.22152 5.3 Mechanism: A Four-Stage Cascade｜Q19

> 原文："The four stages are separable but not independent: in most failed rollouts at least two stages fire simultaneously"
> 出处：2608.22152 5.3 Mechanism: A Four-Stage Cascade｜Q20

> 原文："67% of failed rollouts fire at least two stages, justifying the §5.3 claim that the four stages are separable but not independent."
> 出处：2608.22152 Appendix D Cascade Co-occurrence｜Q44

> 原文："L3 and L4 co-fire at Jaccard 0.67, the largest overlap by a wide margin, which motivates the observationalvs-causal reconciliation in §5.3 and §5.5."
> 出处：2608.22152 Appendix D Cascade Co-occurrence｜Q45

> 原文："L2 failures are silences: each utterance is wellformed; what is missing is a question."
> 出处：2608.22152 Appendix E Case Studies（Cross-case summary）｜Q21

> 原文："L3 failures are truncations: the dialogue ends or commits before a merged-state utterance ever appears."
> 出处：2608.22152 Appendix E Case Studies（Cross-case summary）｜Q22

> 原文："We validate the four-stage cascade judge with four independent expert annotators re-labelling a stratified sample of homogeneous rollouts under a shared guideline."
> 出处：2608.22152 Appendix C Human Validation of the Cascade Judge｜A14

### E. 可机械预测 / 部分可解

> 原文："The collaboration tax is mechanically predictable from conversation features. As shown in Figure 4, the regression achieves a substantial out-of-fold R2 across the (model, task) cells, with strong held-out rank correlation throughout."
> 出处：2608.22152 5.4 Predicting the Gap｜Q23

> 原文："Out-of-fold R2 = 0.475, Spearman ρ = 0.760, Pearson r = 0.705."
> 出处：2608.22152 Figure 4 图注｜Q24

> 原文："Capability sets the intercept; conversation shape sets the slope. The regression generalises across tasks: a leave-one-task-out evaluation, in which every task is held out in turn while fitting on the rest, retains positive held-out variance explained and strong rank correlation"
> 出处：2608.22152 5.4 Predicting the Gap｜Q25

> 原文："the regression cannot extrapolate the absolute gap level to a held-out model: a leave-one-model-out variant preserves the rank ordering of cells but not their absolute level."
> 出处：2608.22152 5.4 Predicting the Gap｜Q26

> 原文："a model-level intercept set by base capability and a slope along conversation-shape features shared across models."
> 出处：2608.22152 5.4 Predicting the Gap｜A17

> 原文："The combined intervention recovers a substantial fraction of the tax. As shown in Table 1 (last column), the all-four condition lifts homogeneous success against the no-intervention baseline, with the 95% confidence interval bounded well above zero on every category. Per-category responsiveness follows the predictive signal of Section 5.4: CSP responds most, Relational next, and Spatial least."
> 出处：2608.22152 5.5 Intervention: Stage-Targeted Prompt Clauses｜Q27

> 原文："No condition reaches the solo ceiling, but a single change to the system prompt recovers a substantial fraction of the entire collaboration tax across the suite, with no retraining and no change to the underlying model."
> 出处：2608.22152 5.5 Intervention: Stage-Targeted Prompt Clauses｜Q28

> 原文："Each category is bottlenecked by a different cascade layer. Reading down a column of leave-oneout values isolates the marginal contribution of the dropped clause. As Table 1 shows, dropping L4 on Spatial, L1 on Relational, or L2 on CSP each substantially reduces the lift in the respective category, with the 95% confidence interval crossing zero for no L4 on Spatial and for no L1 on Relational; the critical layer differs across categories."
> 出处：2608.22152 5.5 Intervention: Stage-Targeted Prompt Clauses｜Q29

> 原文："The simplicity of the fix is exactly the point: failures attributed to reasoning or capability cannot be patched this cheaply, but failures of grounding, querying, integration, and re-derivation can."
> 出处：2608.22152 1 Introduction｜Q30

### F. 异质配对：税被拉向更强的一方

> 原文："In heterogeneous pairs the tax is pulled toward the stronger partner rather than the additive midpoint, empirically realising the max-superadditivity violation predicted by our framework."
> 出处：2608.22152 Abstract｜Q31

> 原文："The pair gap is pulled toward the stronger member, not toward the midpoint. Aggregating across (pair, task) cells (Figure 5), the actual hetero ratio gap correlates strongly with the midpoint between the two individual homogeneous gaps but is systematically below it and well below the additive line y = x; the per-cell breakdown by (initiator, responder) is reported in Appendix Figure 9, where the four off-diagonal heterogeneous cells cluster near the strong-tier diagonal rather than averaging between strong and weak."
> 出处：2608.22152 6 Heterogeneous-Pair Matrix｜Q32

> 原文："Pearson r = +0.732 mean y − mean x = -0.170"
> 出处：2608.22152 Figure 5 图注｜A12

> 原文："In every pair, the strong member’s Shapley value falls substantially below its solo payoff (e.g., ϕgpt-5 = 0.668 versus v({gpt-5}) = 0.932 on the nano × gpt-5 configuration), illustrating that under Shapley fairness the stronger member’s marginal contribution to the team is well below its solo capacity."
> 出处：2608.22152 Appendix A.5 Shapley value and subadditivity｜A7

> 原文："every member’s Shapley share falls below its singleton payoff, as Appendix A.4 formalises."
> 出处：2608.22152 6 Heterogeneous-Pair Matrix｜Q33

> 原文："Every pair violates maxsuperadditivity, and every member’s Shapley value falls below its singleton payoff."
> 出处：2608.22152 Table 2 表注｜Q43

### G. 论文自承局限

> 原文："Two-agent only. Our suite measures the collaboration tax for dyadic pairs (N = 2). Whether the four-stage cascade and the rank-survives, level-fails decomposition generalise to N ≥ 3 multi-agent settings is unstudied;"
> 出处：2608.22152 7 Conclusion（Limitations: Two-agent only）｜Q35

> 原文："Synthetic tasks rather than deployment workloads. Every task in our suite is a procedurally generated puzzle (grid, graph, constraint satisfaction). This isolates coordination mechanics from domain knowledge and makes the collaboration tax cleanly attributable to grounding friction, but the resulting setting does not directly correspond to the multi-agent applications LLMs are increasingly deployed in: collaborative code editing, issue triage and resolution, multi-turn debugging, or interactive document drafting, where the answer space and the feedback signal are far less structured."
> 出处：2608.22152 7 Conclusion（Limitations: Synthetic tasks rather than deployment workloads）｜Q36

> 原文："Future work will design tasks closer to these production workloads (paired code-modification dialogues, issue-resolution pairs over a shared repository) and re-measure the collaboration tax under those conditions."
> 出处：2608.22152 7 Conclusion（Limitations）｜Q37

> 原文："Negative tax exceptions. For sufficiently strong models on CSP tasks the homogeneous tax can be negative; claude-sonnet-4-5 and DeepSeek-V4-Pro on CSP are the two cases reported in Section 5.2."
> 出处：2608.22152 Appendix A.7 Remarks｜Q38

> 原文："Our protocol caps the dialogue at 50 turns, as Appendix I describes; we conjecture that part of the residual tax under the all-four prompt intervention of Section 5.5 is attributable to this bound."
> 出处：2608.22152 Appendix A.7 Remarks｜Q39

> 原文："Scope. The formalism here applies to 2-agent pairs with a single common payoff and unionnecessary information. Extensions to n ≥ 3 agents, weighted contributions, or non-cooperative settings are outside the scope of this paper."
> 出处：2608.22152 Appendix A.7 Remarks｜Q40

### H. 可抄的干预子句与稳健性检查

> 原文："L4 re-derivation requirement. Appended to system prompt Before either agent issues ACTI!, the other agent must show recomputation work in their immediately preceding message: re-walk the path step by step, recompute the sum, re-check each constraint."
> 出处：2608.22152 Appendix G.2 Intervention Clauses（L4 re-derivation requirement）｜A8

> 原文："L3 integration block. Appended to system prompt Before any agent issues a final-answer proposal, that agent’s preceding message must begin with an explicit integration block:"
> 出处：2608.22152 Appendix G.2 Intervention Clauses（L3 integration block）｜A9

> 原文："removing the critic prompt drops the mean solo-full score from 0.575 to 0.569 (Table 9), a 0.006-point cost."
> 出处：2608.22152 Appendix H Additional Results（Critic ablation）｜Q41

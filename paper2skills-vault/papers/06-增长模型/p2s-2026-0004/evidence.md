---
title: 证据链 — Seasonal false alarms in customer churn and decline early-warning systems
doc_type: evidence
module: papers/06-增长模型/p2s-2026-0004
status: verified
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2608.18174
registry_id: p2s-2026-0004
paper: Seasonal false alarms in customer churn and decline early-warning systems: adjacent-window labels confound seasonality with decline, and a year-over-year correction
venue: arXiv preprint
venue_tier: preprint
evidence_grade: A
fulltext: papers/06-增长模型/p2s-2026-0004/fulltext.md（arXiv HTML v1 存档，83,336 字符）
card: paper2skills-vault/06-增长模型/Skill-Seasonal-Aligned-Churn-Label.md
---

# 证据链：2608.18174 → Skill-Seasonal-Aligned-Churn-Label

## 0. 核验结论（可复跑）

```text
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <card>
  → ✅ PASS  Skill-Seasonal-Aligned-Churn-Label.md#stitched(2块)
     L1_SYNTAX/L2_COMPILE/L3_IMPORT/L4_SMOKE/L5_TEST 全部 True（6 个断言全绿）

python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <card>
  → ✅ VERBATIM  40/40 逐字；n_fabricated=0；n_fuzzy=0；n_spliced=0

python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <card> --only G2
  → "passed": true；unsourced_metric=0；traceability_pct=100.0

python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
  → ✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接（证明上面的 VERBATIM 有意义）
```

## 1. 立项材料两项待核事实的核查结论

| 待核事实 | 结论 | 原文依据 |
|---|---|---|
| 生产系统约 **1/3** 行动清单是误报 | **确认成立**，但要注意口径分层 | 摘要：「每三个被服务的行动位里有一个给了在季节性对齐标签下会消失的标记」；引言：「约每三个被服务的标记里有一个在季节性对齐重标下消失」。⚠️ 1/3 是**部署系统服务清单**层面的数字；统一协议下的**材料队列**对齐不一致率是 50%，公共面板 37%–69% —— 分母不同，不可混用 |
| 论文自带复现包 | **确认存在**（arXiv ancillary files），但**不含生产面板** | 「脚本、预指定分析计划、带时间戳的结果工件……随本预印本作为 arXiv ancillary files 提供」；公共面板清洗步骤在 replication package 中；生产面板为专有数据，仅以比率/比值/计数出现 |

> 注：本卡未下载 arXiv ancillary files，因此**只核对了全文中的声明**，没有核对包内文件是否可下载、能否一键跑通。

## 2. 卡片正文数字 → 原文出处对照

下表覆盖卡片正文里出现的**全部实质数字**（去重后逐条列出）。每一行的引文块都能在 ⑥ 段（与本文第 3 节同源）逐字查到；该表由脚本生成并做过自检：表中每个数字都必须真实出现在它所引的引文块内，否则脚本报错。

| 数字 | 卡片位置 | 引文块 | 论文章节 |
|---|---|---|---|
| `0.767 → 0.864` | ⑤ 评估依据（同一模型只换标签） | Q20 | §5.6 With the model held fixed, relabeling moves holdout skill |
| `0.736 → 0.857、0.253 → 0.367、0.430 → 0.743` | ⑤（与上一条同源，卡片正文未逐一展开） | Q20 | §5.6 With the model held fixed, relabeling moves holdout skill |
| `0.015` | ⑤ 评估依据（换更强学习器 TabPFN 的 PR-AUC 变化） | Q43 | §5.6 With the model held fixed, relabeling moves holdout skill |
| `（无数值）货币金额口径` | ⑤ 「论文未给金额」的依据 | Q44 | §6 The correction in practice |
| `119 → 79` | ② 场景一 业务价值；⑤ ROI 公式 | Q23、Q24 | §6 The correction in practice |
| `40` | ⑤ ROI 公式（每周期节省的干预位） | Q24 | §6 The correction in practice |
| `14% / 6%` | ⑤ 边界情形（r=5 / r=10 的净收益优势） | Q42 | §6 The correction in practice |
| `0.70 / 0.86` | ①b 第 3 条（持续强增长下召回变严） | Q30 | §7 Discussion and limitations |
| `0.905 / 0.820` | ⑥（真衰退召回：对齐 vs 相邻） | Q12 | §5.2 The synthetic sweep isolates the mechanism |
| `0.78 / 0.44` | ①b 已知失败模式第 3 条（日历特征无用） | Q22 | §5.6 With the model held fixed, relabeling moves holdout skill |
| `0.96` | ⑥（模型对自己的标签可以很好看） | Q39 | §5.6 With the model held fixed, relabeling moves holdout skill |
| `27.3% / 0.01% / 0.03%` | ⑥（合成振幅扫描的误报率） | Q11 | §5.2 The synthetic sweep isolates the mechanism |
| `2.4% / 9.3%` | ⑥（生产训练集 4 月 / 10 月锚点事件率） | Q5 | §1 Introduction |
| `50%` | ② 口径核对；⑥（材料队列对齐不一致率） | Q8 | §5.1 The anchor-month artifact replicates on public panels |
| `2.4 / 0.284 / 0.054` | ⑥（材料队列 max/min 比与 CV） | Q8 | §5.1 The anchor-month artifact replicates on public panels |
| `37–69%` | ② 口径核对（公共面板不一致率） | Q1、Q13 | §Abstract；§5.3 Pooling does not repair the labels |
| `28–50%` | ② 口径核对（生产面板不一致率） | Q1 | §Abstract |
| `9.2% / 9.5%` | ② 场景二（相邻臂 vs 品类先验臂的误报率） | Q18、Q19 | §5.5 Alternative remedies, measured（Table 2，Adjacent 行）；§5.5 Alternative remedies, measured（Table 2，Deseason. (pooled idx) 行） |
| `两三倍（原文 two to more than three times）` | ② 场景二（先验臂留下的离散度） | Q17 | §5.5 Alternative remedies, measured |
| `1/3（原文 one in three）` | ② 口径核对（服务清单层面的误报比例） | Q2、Q3 | §1 Introduction |
| `12+k / 2k` | ① 关键假设③；①b；② 场景一数据要求 | Q25 | §6 The correction in practice |
| `75%` | ② 场景一 数据要求（围栏口径） | Q35 | §4.2 Protocol |
| `214 / 383 / 50%` | ⑥（材料买家的季节摆动幅度） | Q10 | §4.1 Panels |
| `2.1 / 30%` | ⑥（增长方向的对齐不一致率） | Q26 | §5.7 Robustness |
| `27–41% / 6–19%` | ⑥（多年持续衰退者的命中率对比） | Q15 | §5.4 What the disagreement events actually are |
| `83% / 70%` | ⑥（去掉季节形状后可解释的不一致占比） | Q37 | §5.4 What the disagreement events actually are |
| `50K / 100K / 200K` | ② 场景一 数据要求（材料性金额门槛三档） | Q9 | §5.1 The anchor-month artifact replicates on public panels |
| `400` | ⑥（合成面板每个振幅的实体数） | Q36 | §4.1 Panels |
| `5.5% / 7.0%` | ⑥（两个目标流行度不同，口径警告） | Q21 | §5.6 With the model held fixed, relabeling moves holdout skill |

### 结构性数字（非事实断言，无需出处）

| 数字 | 说明 |
|---|---|
| `2026 / 09` | frontmatter created/updated 与变更记录日期（YYYY-MM-DD）（结构性） |
| `2608.18174` | frontmatter paper_id（arXiv 标识符）与 ⑥ 的出处行（标识符） |
| `1 / 4 / 7 / ④ ⑤ ⑥` | 章节号、列表序号、以及「Table 1 / Table 3」的表号引用（结构性） |
| `11` | 场景一「黑五（11 月）旺季一过」—— 日历月指代，非论文数字（业务日历） |
| `8` | 场景二「出海首年只有 8–14 个月数据」—— 本卡对企业自身历史长度的假设，非论文数字（业务假设） |

## 3. 逐字引文（与卡片 ⑥ 段同源，全部经 quote_check.py 判 VERBATIM）

### 3.1 机制与命题

**[Q4]**

> 原文："The construction has a defect: its two windows cover different calendar months. For a seasonal entity the two windows sit at different points of the seasonal cycle. The label therefore fires on seasonal descent and heals on seasonal ascent, whether or not the entity’s trajectory has changed."
> 出处：2608.18174 §1 Introduction

**[Q6]**

> 原文："For a purely seasonal entity anchored just after its peak, the trailing baseline sums the high season and the outcome window sums the low season. The adjacent ratio then reads 0.5, and the label declares “decay” with no decline anywhere in the series. The year-over-year baseline reads parity."
> 出处：2608.18174 §3.2 Why the adjacent label is calendar-dependent

**[Q7]**

> 原文："Part (ii) is the mechanism in one line. The twelve anchor-month ratios have geometric mean exactly one. Any profile whose window sums are not all equal therefore places some anchors below parity."
> 出处：2608.18174 §3.2 Why the adjacent label is calendar-dependent

**[Q5]**

> 原文："Across its 10,426 buyer–month training observations, the decay-event rate ranged from 2.4% at April anchors to 9.3% at October anchors. About half of the decay events dissolved under a seasonally aligned definition."
> 出处：2608.18174 §1 Introduction

### 3.2 生产系统与公共面板的实测

**[Q9]**

> 原文："Raising it from zero through $50K, $100K, and $200K of trailing annual value moves the ratio $1.3\to 2.1\to 2.4\to 3.4$."
> 出处：2608.18174 §5.1 The anchor-month artifact replicates on public panels

**[Q8]**

> 原文："On the production material cohort (the deployed materiality floor applied to the full panel) the adjacent decay rate troughs at 3.9% in April and peaks at 9.1% in November. That is a max/min ratio of 2.4, with a coefficient of variation of 0.284. The aligned label sits nearly flat between 7.1% and 8.6% (CV 0.054). Half of the adjacent events (50%, interval $[45,54]$) have no aligned counterpart."
> 出处：2608.18174 §5.1 The anchor-month artifact replicates on public panels

**[Q10]**

> 原文："Seasonality is substantial: 214 of 383 material buyers swing more than 50% from peak to trough within a year."
> 出处：2608.18174 §4.1 Panels

**[Q11]**

> 原文："The adjacent label’s false-event rate rises from below 0.01% at $a=0$ through 3.0% at $a=0.4$ to 27.3% at $a=0.8$. The aligned label stays at or below 0.03% at every amplitude."
> 出处：2608.18174 §5.2 The synthetic sweep isolates the mechanism

**[Q12]**

> 原文："On the injected true declines, recall is 0.905 aligned against 0.820 adjacent (Figure 2b)."
> 出处：2608.18174 §5.2 The synthetic sweep isolates the mechanism

**[Q13]**

> 原文："Pooled across every anchor month, 37–69% of the adjacent label’s decay events on the public panels have no aligned counterpart."
> 出处：2608.18174 §5.3 Pooling does not repair the labels

**[Q14]**

> 原文："Pooling fixes the composition of the training set without fixing the labels inside it."
> 出处：2608.18174 §5.3 Pooling does not repair the labels

**[Q35]**

> 原文："An entity–anchor observation is valid when at least 75% of the trailing twelve available months are positive (the public analogue of the production materiality floor)."
> 出处：2608.18174 §4.2 Protocol

### 3.3 修正的代价、替代方案与稳健性

**[Q16]**

> 原文："Pooled indices fail where seasonal phases are heterogeneous — a panel-level index cannot serve entities that peak in different months."
> 出处：2608.18174 §5.5 Alternative remedies, measured

**[Q17]**

> 原文："On M5 and production, where profiles drift and history is shorter, they leave two to more than three times the aligned label’s dispersion. They also demand a year more history."
> 出处：2608.18174 §5.5 Alternative remedies, measured

**[Q18]**

> 原文："| Adjacent | 12 | 9.2% | 0.82 | 0.235 | 0.195 | 0.267 |"
> 出处：2608.18174 §5.5 Alternative remedies, measured（Table 2，Adjacent 行）

**[Q19]**

> 原文："| Deseason. (pooled idx) | 18 | 9.5% | 0.83 | 0.216 | 0.186 | 0.121 |"
> 出处：2608.18174 §5.5 Alternative remedies, measured（Table 2，Deseason. (pooled idx) 行）

**[Q34]**

> 原文："Among the arms that flatten the curve, only the aligned label keeps the six-month horizon, at tied-lowest history cost."
> 出处：2608.18174 §5.5 Alternative remedies, measured

**[Q15]**

> 原文："The aligned label fires on 27–41% of following-year anchors against 6–19% for the adjacent label, whose trailing baseline ratchets down with the decline."
> 出处：2608.18174 §5.4 What the disagreement events actually are

**[Q37]**

> 原文："Judged without precedence, however, 83% of the production disagreements on which the deseasonalized label is defined are removed by seasonal adjustment alone (70% on M5)."
> 出处：2608.18174 §5.4 What the disagreement events actually are

**[Q26]**

> 原文："On the production material cohort the adjacent growth label’s anchor-month ratio is 2.1, and 30% of its events lack an aligned counterpart."
> 出处：2608.18174 §5.7 Robustness

### 3.4 模型层与决策层

**[Q20]**

> 原文："ROC-AUC rises from 0.767 to 0.864 for the decay direction and from 0.736 to 0.857 for growth. Precision–recall AUC rises from 0.253 to 0.367 and from 0.430 to 0.743."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

**[Q21]**

> 原文："The caution: the two rows of each pair score different response variables with different prevalences (decay: 5.5% adjacent against 7.0% aligned at training). The comparison therefore does not show one model beating another on a fixed task."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

**[Q22]**

> 原文："Detecting declines in progress, the aligned-trained model reaches ROC 0.78. The adjacent-trained model scores 0.44, below chance, having learned to rank seasonal descent above genuine decline. The calendar features repair nothing (0.44)."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

**[Q43]**

> 原文："Swapping learners under the same labels moved holdout skill insignificantly or negatively. The strongest challenger, the TabPFN tabular foundation model (Hollmann et al., 2025), changed precision–recall AUC by $-0.015$ in both directions."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

**[Q39]**

> 原文："Against its own labels the adjacent-trained model scores 0.96 — a model can look excellent against a defective target."
> 出处：2608.18174 §5.6 With the model held fixed, relabeling moves holdout skill

**[Q33]**

> 原文："The served action list is the unit of cost: every flagged account consumes account-manager attention."
> 出处：2608.18174 §6 The correction in practice

**[Q23]**

> 原文："Relabeling shrank the served “shrinking” classification by a third (119 to 79 accounts)."
> 出处：2608.18174 §6 The correction in practice

**[Q24]**

> 原文："The adjacent list spends 119 intervention units per cycle to reach what the aligned list reaches with 79. That is a saving of 40 units per cycle at any $G$. In net-benefit terms ($Gr-79$ against $Gr-119$) the advantage is those same 40 units at every $r$."
> 出处：2608.18174 §6 The correction in practice

**[Q42]**

> 原文："At the boundary case $G=79$ this amounts to roughly double the net benefit at $r=2$, 14% more at $r=5$, and 6% more at $r=10$."
> 出处：2608.18174 §6 The correction in practice

**[Q25]**

> 原文："The aligned baseline needs the outcome window’s calendar months one year back: roughly $12+k$ months of history before an entity’s first label, against $2k$ for the adjacent construction."
> 出处：2608.18174 §6 The correction in practice

**[Q44]**

> 原文："A currency translation requires per-account margins and intervention costs that are confidential."
> 出处：2608.18174 §6 The correction in practice

### 3.5 复现材料与边界

**[Q31]**

> 原文："Scripts, the pre-specified analysis plan, and timestamped result artifacts sufficient to reproduce every synthetic and public-panel number accompany this preprint as arXiv ancillary files."
> 出处：2608.18174 Data and code availability

**[Q36]**

> 原文："Each has multiplicative month-of-year seasonality with a random peak month and amplitude $a\in\{0,0.2,0.4,0.6,0.8\}$ (400 entities per amplitude), lognormal multiplicative noise, and no trend and no true decline."
> 出处：2608.18174 §4.1 Panels

**[Q32]**

> 原文："First, the controlled model-skill comparison (Table 3) is single-organization. The public panels support the artifact, the mechanism, and the disagreement accounting. The synthetic ground-truth experiment supports the model-level claim. The transfer of the production ROC gains is demonstrated in one system only. Second, our aligned construction assumes annual periodicity with a twelve-month lag. Entities with non-annual cycles, or panels dominated by mobile holidays — the calendar effects that survive year-on-year comparison in official-statistics practice (Eurostat, 2024) — need alignment to the same phase of the relevant cycle. We have not tested that. Third, the aligned baseline replays history. One anomalous prior-year window (a promotion spike, an outage) corrupts up to $k$ of the following year’s labels, where an adjacent baseline would self-heal within $k$ months. Entities are also unlabeled for their first $12+k$ months. That excludes the early-relationship segment where non-contractual defection concentrates, a cost young customer bases bear most. Fourth, the construction we correct is the threshold-ratio label. Inactivity-based churn definitions carry a related seasonal confound (seasonal quiet reads as churn) with a different structure our proposition does not cover. The latent-attrition lineage (Bachmann et al., 2021; Wünderlich et al., 2022) avoids discrete labels altogether. Our results say nothing against those choices. They say that where threshold-ratio window labels are used on seasonal entities, the calendar belongs in the label’s definition and not only in the model’s features. The features-only version of that advice is now measured and found wanting (Section 5.6)."
> 出处：2608.18174 §7 Discussion and limitations（局限 1）

**[Q28]**

> 原文："Second, our aligned construction assumes annual periodicity with a twelve-month lag. Entities with non-annual cycles, or panels dominated by mobile holidays — the calendar effects that survive year-on-year comparison in official-statistics practice (Eurostat, 2024) — need alignment to the same phase of the relevant cycle. We have not tested that."
> 出处：2608.18174 §7 Discussion and limitations（局限 2）

**[Q29]**

> 原文："Third, the aligned baseline replays history. One anomalous prior-year window (a promotion spike, an outage) corrupts up to $k$ of the following year’s labels, where an adjacent baseline would self-heal within $k$ months. Entities are also unlabeled for their first $12+k$ months. That excludes the early-relationship segment where non-contractual defection concentrates, a cost young customer bases bear most."
> 出处：2608.18174 §7 Discussion and limitations（局限 3）

**[Q41]**

> 原文："Fourth, the construction we correct is the threshold-ratio label. Inactivity-based churn definitions carry a related seasonal confound (seasonal quiet reads as churn) with a different structure our proposition does not cover."
> 出处：2608.18174 §7 Discussion and limitations（局限 4）

**[Q30]**

> 原文："Where strong growth is sustained, the cut is also effectively stricter (synthetic recall 0.70 against the adjacent label’s 0.86)."
> 出处：2608.18174 §7 Discussion and limitations

**[Q27]**

> 原文："The case alignment truly cannot see is decline followed by stabilization: a current-state question rather than a new-decline question."
> 出处：2608.18174 §6 The correction in practice

**[Q1]**

> 原文："Of the adjacent-window decay events, 37–69% on the public panels and 28–50% in production have no counterpart under a seasonally aligned definition."
> 出处：2608.18174 §Abstract

## 4. 未能核验 / 只能定性表述的项（诚实清单）

1. **货币金额**：论文明确「换算成货币需要每账号毛利与干预成本，而这些是保密的」，全文**没有报告任何金额**。因此本卡 ⑤ 只给容量口径公式与论文的净收益边界情形，**不写货币数字**；货币参数属于企业内取数。
2. **「品类季节性先验」这一具体退路，论文没有逐字实验过**：论文最接近的臂是 §5.5 的 pooled 指数去季节化臂（其合成误报率与相邻窗口臂几乎持平、真实面板离散度仍高于对齐臂）。本卡 ② 场景二把它当作业务映射来用，**属于外推**，这一点已在卡片中标注；本卡 ③ 的自造面板演示复现了同一现象（先验把误报高峰换个月份而未压平曲线），但那是**本卡代码的合成输出，不是论文数字**。
3. **宝宝月龄生命周期**：论文完全未涉及母婴月龄生命周期，也未测试「按生命周期分层后再做同比」。本卡 ①② 中的月龄讨论来自 MasterPrompt v2 的增长模型领域规则（业务领域知识）与本卡作者的判断，**不是论文结论**；论文只把「非年度周期需对齐到相关周期的同一相位」列为**未测试**的局限。
4. **本卡代码的全部运行输出**（合成面板上的误报率、召回、曲线离散度 CV、k=12 的日历中性）都是**自造数据**的结果，用于演示机制；卡片正文没有把它们当作论文证据引用，只在 ② 场景二做了定性描述（「误报高峰被搬到别的月份、曲线没被压平」）。
5. **模型层收益的迁移性**：论文自承受控对比只在**单一组织**上做过；公共面板支持的是机制与不一致率，合成真值支持的是模型层结论。迁移到我们自己的店铺面板需要自行复现这个对比。
6. **arXiv ancillary 包未实际下载验证**（见第 1 节注）。
7. **论文未报告**的量级（本卡因此不写）：单位干预成本、单账号毛利、清单干预后的实际留存提升、以及任何以货币计价的 ROI。

## 5. 全文存档

- 底本：`paper2skills-vault/papers/06-增长模型/p2s-2026-0004/fulltext.md`（83336 字节，402 行）
- 来源：https://arxiv.org/html/2608.18174v1（arXiv:2608.18174，2026-08-17 发布）
- 引用块核验口径：最长**连续**匹配段占比 ≥ 0.995 判 VERBATIM（拼接引文会被单独标记并拦截）


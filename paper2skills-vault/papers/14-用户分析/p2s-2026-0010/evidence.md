---
title: 证据链 — p2s-2026-0010（Skill-Incrementality-Measurement）
doc_type: evidence
module: 14-用户分析
paper_id: 2607.09608
registry_id: p2s-2026-0010
related_card: paper2skills-vault/14-用户分析/Skill-Incrementality-Measurement.md
fulltext: paper2skills-vault/papers/14-用户分析/p2s-2026-0010/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2607.09608 *Media Measurement and the Assisted Own Goal: Attribution, Marketing-Mix Models, and Individual-Level Incrementality*

- **全文存档**：`paper2skills-vault/papers/14-用户分析/p2s-2026-0010/fulltext.md`
  （339 行 / 36,091 字符；`arxiv_id: 2607.09608`、`paper_id: p2s-2026-0010`，
  source `https://arxiv.org/html/2607.09608v1`；`evidence_grade: A` = 全文可得）
- **作者 / 日期**：Tobias B. Konitzer, PhD；标题页日期 `06/21/2026`；带 JEL 分类码 `M37, L81, D83, C63`。
- **venue**：全文**没有任何 venue 标注**（无期刊/会议页眉、无致谢投稿信息），
  按 registry 口径记 `venue: arXiv preprint` / `venue_tier: preprint`，**不得**当作已录用论文引用。
- **引文全部为逐字切片**：下列每条 `> 原文："…"` 都是从 `fulltext.md` 按**行号 + 起止锚点**直接切出的
  连续子串（脚本化切片，非手抄），没有拼接、没有改写、没有跨段缝合。

---

## 0. 核验凭证（可复核的退出码 / stdout）

```
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C"
已索引仓库内本地模块 86 个（可自动解析的卡片将真正执行）
另有 54 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）
✅ PASS         Skill-Incrementality-Measurement.md#stitched(2块)

==================================================================
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | 🟠 MIGRATED 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）
==================================================================
[exit: 0]

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$C"
✅ VERBATIM   Skill-Incrementality-Measurement.md  31/31 逐字

共 1 张卡：1 通过，0 含伪造引文，0 无全文可核验
[exit: 0]

$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C"
G1 门禁：0/1 通过 (0.0%)  红灯 1  黄灯 0
  ❌ [G1-NO-EVIDENCE] 无 K1 验证凭证 —— 请先运行 verify_skill_code.py，不得凭人工判断放行
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 2
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
[exit: 1]   ← 退出码 1 只由 G1 的「无凭证」造成，见下

# 单卡模式下 gate_check 不会自动找到 K1 报告（设计如此：无凭证一律判红）。
# 补上 K1 凭证后 G1 转绿：
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" \
      --json-out /tmp/k1_p2s-2026-0010.json
$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" \
      --k1 /tmp/k1_p2s-2026-0010.json
G1 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 2
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
[exit: 0]
```

**G2 的两条黄灯**（不阻塞，逐条说明）：

| # | 黄灯 | 内容 | 判定 |
|---|---|---|---|
| 1-2 | `G2-UNSOURCED-GENERAL` | `12` ×2 | 全部来自 v2 模板强制的 frontmatter 日期 `2026-09-12`（`created`/`updated`）——结构性字段，不是事实断言 |

写作过程中还出现过第三类黄灯（`16.16` / `1.02`，即卡片 ③ 输出围栏里本卡自己跑出的模拟值被正文复述了一次），
已改写为不重复数值的表述后消除。**G2 的 `metric_numbers = 0`**：正文里没有任何「带度量语义且无出处」的数字。

G2 关键计数（脚本产出）：

```text
metric_numbers=0  sourced=0  unsourced_metric=0  unsourced_general=2
evidence_sources=31  quotes_total=31  quotes_verbatim=31  quotes_fuzzy=0
quotes_fabricated=0  quotes_spliced=0  sourced_structural_only=0  cn_numeral_claims=0
```

**引文核验器自证**（证明它真的能区分真引文 / 伪造 / 拼接）：

```
$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
1 真引文     : VERBATIM   连续度=1.0
2 纯伪造     : FABRICATED 连续度=0.124   ← 必须 FABRICATED
3 拼接引文   : FABRICATED 连续度=0.681 召回=1.0 拼接标记=True   ← 必须被拦下
4 排版差异   : VERBATIM   连续度=1.0   ← 必须 VERBATIM
✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接
```

---

## 1. registry 逐项核对表（`papers_registry.json` → `p2s-2026-0010`）

核对方式：逐条回 `fulltext.md` 找依据，行号见最后一列。**「论文未报告」= 不能替论文补。**

| registry 字段 | registry 记的值 | 逐字核对结论 | 依据 |
|---|---|---|---|
| `paper_id` | `p2s-2026-0010` | ✅ 与存档目录名、fulltext 头部注释一致 | fulltext L2-3 |
| `identifiers.arxiv` | `2607.09608` | ✅ 与 fulltext 头部 `arxiv_id` 一致 | fulltext L2 |
| `title` | *Media Measurement and the Assisted Own Goal: Attribution, Marketing-Mix Models, and Individual-Level Incrementality* | ✅ 逐字一致（原文标题跨三行排版） | fulltext L9-11 |
| `url` | `https://arxiv.org/abs/2607.09608` | ✅ 与存档 source `.../html/2607.09608v1` 同篇 | fulltext L4 |
| `published` | `2026-07-10` | ⚠️ **无法从全文核验**：标题页写的是 `06/21/2026`（作者自署日期）；arXiv 上线日不在全文里。两者不冲突但**不是同一个量**，引用时勿混用 | fulltext L15 |
| `journal` / `venue` | `""` / `""` | ✅ 全文确无 venue 标注 | 全文无页眉/期刊信息 |
| `venue_tier` | `preprint` | ✅ 无录用证据 → 记 preprint 正确 | — |
| `domain` | `14-用户分析` | ✅ 本项目分配（论文本身无领域标注） | — |
| `priority` / `score` | `P0` / `24.7` | — 流程字段，无法从论文核验 | — |
| `decision` | `extract` | ✅ 见 6 节「R4 硬拦截清单的边界」——本卡按此决定出卡，但该决定值得 parent 复核 | — |
| `decision_reason` | 「被助攻的乌龙球：站外种草引发增量但被下游 marketplace 记账，ROAS 系统性失真；跨境最贵的测量缺口」 | ⚠️ **三句半采信、一句须改口径**：前两句逐字有据（Q3/Q4/Q8）；**「跨境」在全文零命中**，是本项目的业务外推，不是论文结论 | Q3, Q4, Q8；grep 见 6.2 |
| `data_availability` | **现为 `synthetic`**（原记 `conditional`，见 2 节） | ✅ 现行值正确；原值 `conditional` 有误导性 | §6 全文；Q5 |
| `note` | 「论文**无真实数据**：纯理论 + 一个模拟研究（Section 6，200,000 simulated intents）…」 | ✅ 主体正确（逐字有据）；⚠️ 后半句「自建独立站可行」是本项目业务判断，论文未讨论独立站/DTC 场景 | Q5, Q24, Q25 |
| `outputs.skill_card` | `…/14-用户分析/Skill-<方法名>.md`（占位） | ✅ 本次落地为 `Skill-Incrementality-Measurement.md` | — |
| `outputs.code_dir` | `paper2skills-code/user_analytics/<algo>` | ⚠️ **未落地**：代码模板内嵌在卡片 ③ 段，由 K1 验证（与 CLAUDE.md 对 3A/3B 的说明一致） | — |
| `outputs.evidence` | `…/p2s-2026-0010/evidence.md` | ✅ 本文件 | — |
| `gates` | `code/evidence/business: pending` | ➡️ 本次实测：**K1 PASS / G2 通过 / G3 通过**（见 0 节），等待 parent 回填 | — |
| `status` | `shortlisted` | ➡️ 出卡完成后应由 parent 更新为 `extracted` | — |

---

## 2. `data_availability` 判定（本次任务的重点）

### 2.1 事实

- **任务下达时 registry 记的是 `conditional`**，note 为「需受众级随机化：自建独立站可行，纯平台站内卖家须降级为诊断框架」。
- **写作过程中 registry 已被改动**：现值 `data_availability: "synthetic"`，revision_log 记录
  `"p2s-2026-0010: data_availability 'conditional' -> 'synthetic'（论文无真实数据）"`
  （revision_log 条目标为 `by: PHASE3-3C`，2026-09-12）。
- **本文的独立判定与现行值一致**：`synthetic` 正确。理由与行号如下。

### 2.2 论文无真实数据 —— 逐条依据（行号）

1. **作者自述这是「模拟设计」**：贡献列表最后一条逐字写明 *"We outline a simulation design under which
   the attribution bias, the induced revenue loss, and the accuracy of the individual-level model can be
   quantified (Section 6)."* —— 注意动词是 **outline**（提出/勾画），不是 report/measure（⑥ Q5，L79）。
2. **§6 的数据生成过程全部是模拟**：*"spend creates intent ($\beta=0.5$), each intent converts on the
   marketplace with probability $\tau$ … with recovery rate $\varphi$."*（⑥ Q24，L273）。
3. **§6 的样本量是模拟样本**：*"For 200,000 simulated intents per cell …"*（⑥ Q25，L277）。
4. **§6 的三个 panel 全是模拟输出**：wedge（L277）、backfire（L281）、recovery（L285，
   「200,000 users per arm」同样是模拟）。Figure 1 的图注逐字写明 *"(a) Attributed-to-true ROAS ratios
   from 200,000 simulated intents per cell (circles) against the theoretical wedge…"*（L287）。
5. **全文检索**：`dataset` 0 命中、`case study` 0 命中、`field experiment` 的 3 处全部是**参考文献标题**
   （Blake et al. 2015 / Gordon et al. 2019 / Johnson 2023），不是本文的实验；`empirical` 仅 1 处，
   出现在 §2.5 描述 φ 时（*"empirically $\phi$ is close to $0$"*，⑥ Q6）——那是**定性断言，没有给数据**。
6. **论文未提供任何可获取的数据集**：没有数据链接、没有仓库、没有复现包；§5.1 引用的生产实践
   （ambient randomization「as deployed by GrowthLoop」）指向的是**厂商白皮书**（GrowthLoop 2026a/2026b
   的 storyblok PDF 链接），不是可下载的学术数据。

### 2.3 为什么原来的 `conditional` 有误导性

`conditional` 的常规读法是「**有数据，但获取有附加条件**」（例如私有面板、需申请、需签协议）。
本论文的真实状态是 **「没有数据，只有一套可以被任何人重新跑一遍的模拟设定」**。两者对下游的影响完全不同：

- 读成 `conditional` → 下游会以为「去谈个数据授权就能拿到实测结果」；
- 实际是 `synthetic` → 下游应当预期「一切数字都是设定值/模拟值，**没有任何可外推的实测参数**」。
  本卡 ⑤ 的收益推算因此必须写成「公式演练」而不是预测。

**结论与建议**：维持现行的 `synthetic`。若 registry 后续只允许 `available/partial/unavailable` 三值，
应记为 `unavailable`（论文**确实没有**任何真实数据），**绝不能**退回 `conditional`。

### 2.4 note 里那句业务判断：部分采信

- ✅ 采信：「需受众级随机化」——有逐字依据（§5.1 的 ambient 随机化设计，⑥ Q15；§5.2 的 ITT estimand，⑥ Q17）。
- ✅ 采信：「纯平台站内卖家须降级为诊断框架」——与 §5.4 的**枚举约束**（拉新广告违反约束，⑥ Q22）
  和 §5.2 的**渠道完整结果**要求（⑥ Q16）一致。
- ⚠️ 需要收窄：「**自建独立站可行**」是**本项目的业务推断**，论文里没有独立站、DTC、跨境电商任何字样。
  更准确的说法是：自建站能拿到**可枚举的第一方 ID**（解决随机化），但**渠道完整结果**仍取决于下游
  marketplace 的订单能否回到同一 user/order key —— 真正的瓶颈在这里，而不是随机化本身。
  （这也是本卡 ② 场景 1 的「数据可得性」写成 `部分可得（需补充 X）` 的原因。）

---

## 3. 卡片正文数字 → 出处对照表

**A 类·论文事实数字**（必须能被 ⑥ 段逐字引文覆盖；G2 实测 `metric_numbers = 0`，
即正文里没有任何「带度量语义且无出处」的数字）：

| 卡片中的数字 | 含义 | 论文位置 | 引文键 |
|---|---|---|---|
| `1 − τ(1 − φ)` | 可观测因子 κ 的闭式表达 | §2.5 式 (4)，L129 | Q7 |
| `1/(1−β)` | Proposition 2 里投入比的指数 | §2.7，L159 | Q10 |
| `0.5`（β） | §6 模拟设定的规模弹性 | §6，L273 | Q24 |
| `0.75` / `0`（τ / φ） | §6 wedge 的模拟设定（「τ=0.75 且无信号回收」） | §6，L277 | Q25 |
| `200,000` | §6 每个 cell 的模拟 intents 数 | §6，L277 | Q25 |
| `0.7` / `0.1`（τ / φ） | §6 recovery 的模拟设定 | §6，L285 | Q27 |
| `15.4 ± 1.0` | §6 模拟中渠道完整 ITT 的估计值（**模拟输出，非实测**） | §6，L285 | Q28 |
| `2,226` / `R² 0.88` | **论文转引 Gordon et al. (2023) 的 campaign 级 PIE 结果**，不是本文实验 | §5.3，L249 | Q20 |

**B 类·本卡可复算数字**（放在代码围栏 / 输出围栏 / 算式块内，G2 按 v2.1 约定豁免）：

| 位置 | 数字 | 性质 |
|---|---|---|
| ③ 代码输出围栏 | `0.2505`、`16.16 ± 1.02`、`6.25 ± 0.62`、`0.387`、`0.370`、`1.957`、`0.773`、`0.0104`、`0.0493`、`0.0101` | **本卡合成数据的运行输出**（numpy 固定种子），非论文数据、非企业数据 |
| ⑤ 算式块 | `κ=0.25`、`6.25%`、`25%`、`75%` | **本卡按论文式 (1)(4) 与 Proposition 2 的推算**，代入的 τ/φ/β 是论文 §6 的**模拟设定值** |

**C 类·结构性字段**：frontmatter 的 `2026-09-12`（created/updated）、`module: 14-用户分析`、
`paper_id: 2607.09608`、章节号 `§2.5`/`§5.3`/`式 (4)`/`Figure 1`、交叉引用键 `Q1…Q31`。

---

## 4. 逐字引文全表（31 条，与卡片 ⑥ 段一一对应）

### A. own goal 的定义与机制

> 原文："We term the un-credited, demand-generating impression an assist, in the football sense: it set up the goal. But because the assist is invisible to the measurement system in attribution-based systems, it is scored against the originating platform—an own goal."
> 出处：2607.09608 §1 Introduction｜L33｜Q3

> 原文："Hence the assisted own goal hypothesis: the act of successfully generating demand can, through the attribution layer, either not increase, or in some cases even reduce the generating platform’s measured performance and therefore its advertising revenue."
> 出处：2607.09608 §1 Introduction｜L33｜Q4

> 原文："When the purchase occurs on a trusted marketplace that does not pass conversion signals from the originating platform back to the advertiser – and that is true for most inter-marketplace transactions – the conversion is simply not observed by the advertiser’s attribution system."
> 出处：2607.09608 §1 Introduction｜L31｜Q2

> 原文："Because ITT contrasts are computed on channel-complete outcomes, the estimator is unbiased and the own goal disappears."
> 出处：2607.09608 §Abstract｜L19｜Q1

### B. 归因为什么看不见（τ 与 φ 的机制）

> 原文："Trusted marketplaces characteristically suppress third-party signal or share any data of its referrer (in this case $G$), so empirically $\phi$ is close to $0$."
> 出处：2607.09608 §2.5 The attribution measurement layer｜L123｜Q6

> 原文："$\kappa(\tau,\phi)\;\equiv\;(1-\tau)+\phi\,\tau\;=\;1-\tau(1-\phi)\;\in(0,1].$"
> 出处：2607.09608 §2.5 式 (4)｜L129｜Q7

> 原文："It is worth pointing out that the failure is not the crediting heuristic (i.e. last- vs. multi-touch), but the observability constraint: while multi-touch or data-driven attribution re-divides credit among observed touchpoints, the own goal removes the conversion from the observable universe. No re-weighting of visible touchpoints can restore credit for this invisible conversion."
> 出处：2607.09608 §4.1 Attribution-based measurement｜L203｜Q11

### C. 理论结果（Proposition 1 / 2 / 3）

> 原文："Measured ROAS understates true incremental ROAS by exactly the observability factor $1-\tau(1-\phi)$ with equality if and only if either there is no distrust ($\tau=0$) or off-platform purchases are perfectly back-propagated ($\phi=1$). The wedge is strictly increasing in distrust $\tau$ and strictly decreasing in the recovery rate $\phi$."
> 出处：2607.09608 §2.6 Proposition 1｜L143｜Q8

> 原文："Under ROAS-thresholding allocation, the demand-generating platform receives strictly less than first-best spend whenever $\tau>0$ and $\phi<1$:"
> 出处：2607.09608 §2.7 Proposition 2｜L157｜Q9

> 原文："$\frac{s_{G}^{\text{attr}}}{s_{G}^{\text{true}}}=\kappa(\tau,\phi)^{\frac{1}{1-\beta}}=\big(1-\tau(1-\phi)\big)^{\frac{1}{1-\beta}}\;<\;1.$"
> 出处：2607.09608 §2.7 Proposition 2 比值式｜L159｜Q10

> 原文："Under randomization of $Z_{i}$ and channel-complete outcome measurement, $\Delta^{\text{ITT}}$ identifies the average incremental purchase effect per assigned user and is invariant to the diversion share $\tau$, the recovery rate $\phi$, and the marketplace claim share $\eta$."
> 出处：2607.09608 §5.2 Proposition 3｜L243｜Q18

> 原文："The proposition states the formal sense in which the own goal is a measurement artifact: the same assist that is invisible to attribution is fully recoverable by a channel-complete ITT under a randomized experiment."
> 出处：2607.09608 §5.2｜L245｜Q19

### D. 另外两个测量口径为什么不解决它

> 原文："First, and most directly tied to our mechanism, the harvester’s spend is endogenous to the generator’s demand: $R$ prices and sells sponsored placements against the arriving intent, so any typical regression model regressing total purchase intent on $s_{R}$ and $S_{G}$ does not know how much of the generated demand to attribute to the harvesting channel—the own goal reappears as simultaneity or multi-colinearity bias rather than signal loss."
> 出处：2607.09608 §4.2 Marketing-mix models｜L209｜Q12

> 原文："Third, MMM resolves channels by week or quarter, not by day, so it cannot drive the thresholding decisions even when its aggregate reading is correct."
> 出处：2607.09608 §4.2 Marketing-mix models｜L209｜Q13

> 原文："Such estimates recover $\mathrm{ROAS}^{\text{true}}_{G}$ and are, by construction, invariant to the diversion share $\tau$: a holdout simply buys less of the product in total, wherever those sales would have occurred."
> 出处：2607.09608 §4.3 Incrementality-based experimentation｜L213｜Q14

### E. 测量方案的两个部件（ambient 随机化 + 个体级扩展）

> 原文："Assignment is a deterministic hash of the user identifier salted by an audience-specific key: user $i$ is assigned to control in audience $a$ if and only if $h(i,a)\bmod 100<100\,c_{a}$, where $c_{a}$ is the audience’s control percentage."
> 出处：2607.09608 §5.1 Ambient audience-level randomization｜L231｜Q15

> 原文："the outcome measured in the brand’s first-party transaction data—channel-complete by construction, in the sense that it aggregates purchases wherever they are booked: on the generator’s storefront, on the marketplace, or offline."
> 出处：2607.09608 §5.2 Intent-to-treat as the estimand｜L235｜Q16

> 原文："Using assignment rather than exposure avoids conditioning on the ad platform’s endogenous delivery decisions (who saw the ad is algorithmically selected; who was assigned is controled by GrowthLoop’s randomization procedure), and matches the advertiser’s decision variable: budget buys assignment, not exposure."
> 出处：2607.09608 §5.2｜L239｜Q17

> 原文："using 2,226 Meta RCTs, it trains a model mapping campaign features—including post-determined aggregates such as exposure rates and last-click conversions, which would be invalid controls in a causal regression but are valid predictors once identification is handled by the experiments—to experiment-identified incrementality, achieving out-of-sample $R^{2}=0.88$ against $R^{2}=0.19$ for seven-day last-click attribution."
> 出处：2607.09608 §5.3（**论文转引 Gordon et al. (2023) 的 campaign 级结果，不是本论文的实验**）｜L249｜Q20

> 原文："The feature coefficients can now be projected onto any audience with no holdout given that the audience in question has the same features at both individual- and campaign-level as the initial set of experiments."
> 出处：2607.09608 §5.3 Projection onto campaigns without holdouts｜L259｜Q21

> 原文："Assigning converters randomly after the fact is independent of treatment by construction and dilutes the intent-to-treat effect toward zero; assigning by observed exposure conditions on the platform’s endogenous delivery reproduces the attribution bias. Assignment must precede exposure."
> 出处：2607.09608 §5.4 Acquisition advertising and the enumerability constraint｜L263｜Q22

> 原文："Note that platform-side lift studies delegate individual-level randomization to the party that can enumerate at auction time—though their platform-observed outcomes are not channel-complete, so the own goal survives inside the lift test itself."
> 出处：2607.09608 §5.4｜L265｜Q23

### F. 模拟研究（§6）—— 论文没有真实数据，本段全部是模拟设定与模拟输出

> 原文："We outline a simulation design under which the attribution bias, the induced revenue loss, and the accuracy of the individual-level model can be quantified (Section 6)."
> 出处：2607.09608 §1 Contributions｜L79｜Q5

> 原文："Consumer journeys are drawn from the data-generating process of Section 2: spend creates intent ($\beta=0.5$), each intent converts on the marketplace with probability $\tau$ and on the generator’s storefront with probability $1-\tau$, and the attribution layer matches on-platform purchases perfectly but recovers diverted purchases only with recovery rate $\varphi$."
> 出处：2607.09608 §6 Simulation Study｜L273｜Q24

> 原文："For 200,000 simulated intents per cell we compute the ratio of attributed to true conversions across $\tau\in\{0,.25,.5,.75,.95\}$ and $\varphi\in\{0,0.3\}$. The simulated ratios (circles in panel a) mimic $\kappa=1-\tau(1-\varphi)$ to three decimal places: at $\tau=0.75$ with no signal recovery, the generator is credited with exactly one quarter of the conversions it caused."
> 出处：2607.09608 §6 The wedge｜L277｜Q25

> 原文："First-best spend grows with $\alpha$ throughout, but attributed revenue peaks at $\alpha\approx 1.7$ and then declines: by $\alpha=2.6$ the generator has lost nearly half of its peak revenue while being more than twice as effective as at baseline."
> 出处：2607.09608 §6 The backfire｜L281｜Q26

> 原文："Finally we run the ambient experiment of Section 5 at $\tau=0.7$, $\varphi=0.1$: 200,000 users per arm, a 2% baseline conversion rate on channel-complete outcomes, and a true incremental effect of 15 conversions per 1,000 assigned users."
> 出处：2607.09608 §6 Recovery｜L285｜Q27

> 原文："The ITT contrast estimates $15.4\pm 1.0$—the truth, within sampling error—while last-touch attribution reports $5.5$, which is $\kappa\times$truth to the decimal (panel c)."
> 出处：2607.09608 §6 Recovery｜L285｜Q28

> 原文："Between the two sits the MMM benchmark: aggregating the same journeys to 600 market-level cells and regressing total sales on assigned reach and the retailer’s sponsored spend—which endogenously tracks arriving demand—yields $9.7\pm 2.4$, with the shortfall re-credited to $R$: the coefficient on $R$’s spend is positive even though $R$’s advertising causes nothing. In other words, MMM sees the diverted demand in aggregate but falsely credits part of it to the harvester, exactly as Section 4 anticipates."
> 出处：2607.09608 §6 Recovery｜L285｜Q29

### G. 论文自承局限（§7 Limitations）

> 原文："The model is deliberately parsimonious: a single product, a single generator and retailer, a static one-shot allocation, and a reduced-form trust parameter."
> 出处：2607.09608 §7 Limitations｜L303｜Q30

> 原文："The measurement model of Section 5 adds its own assumptions: channel-complete first-party outcomes, negligible cross-experiment interaction (Section 5.1), and transportability of conditional effects to un-experimented audiences; non-changing addressable audiences."
> 出处：2607.09608 §7 Limitations｜L303｜Q31

---

## 5. 未能核验 / 「论文未报告」清单

**这一节是本次交付的诚实边界：以下内容本卡一律没有写数字，因为它们无法核验或论文根本没给。**

### 5.1 论文未报告（不能替它编）

1. **任何真实投放的收益/损失幅度** —— 没有实测 ROAS、没有实测 τ、φ。
2. **任何成本量级** —— 实施成本、人力、周期、工程投入，全文零提及（⑤ 段因此写「企业自估」）。
3. **任何行业的 α、β 估计** —— §6 的 β=0.5 是**模拟设定**，不是估计结果。
4. **个体级模型的表现** —— §5.3 只给设计；论文转引的 PIE 指标（2,226 RCT、R² 0.88 vs 0.19）
   是 Gordon et al. (2023) 的 **campaign 粒**结果。**本文没有跑个体级实验**。
5. **图 1 各 panel 的完整数值表** —— 只有散点/线（正文给了几个点：τ=0.75 的 1/4、α≈1.7 的峰值、
   α=2.6 掉一半、15.4/5.5/9.7），没有表格化的完整数字。
6. **MMM 基准的实现细节** —— §6 只说「aggregating the same journeys to 600 market-level cells」，
   未给回归设定、adstock/saturation 形式、先验。
7. **母婴 / 跨境电商 / 独立站场景** —— 全文 `cross-border`、`e-commerce`、`mother`、`baby` 全部 **0 命中**。
   卡片 ② 的场景是业务背景（本项目给定），**没有任何数字来自论文**。
8. **平台站内与独立站的双边身份对齐方案** —— 论文只说结果必须 channel-complete，未给工程实现。

### 5.2 无法核验

1. **arXiv 上线日 `2026-07-10`**（registry `published`）：全文只有作者自署的 `06/21/2026`。
2. **实际投稿/录用状态**：全文无 venue 标注 → 记 `preprint`；**不得**在任何下游材料里当作已录用论文。
3. **「跨境最贵的测量缺口」这一判断**：见 1 节与 6.2，属项目侧外推。
4. **论文与作者生产实现的等价性**：无公开代码；K1 只证明**卡片自带代码**可执行（L5 断言全绿），
   **不证明**它与作者（或 GrowthLoop）的实际系统一致。
5. **registry 记的 `score: 24.7` / `priority: P0`**：流程评分，无法从论文核验。

---

## 6. 本次核对新发现的问题（registry 与论文之外，论文自身的不一致）

### 6.1 §2.4 的示例算术自相矛盾（论文内部错误）

> 原文："Imagine the following scenario: Acme, a consumer brand, spends a fixed amount on $G$ to generate 1,000 net new potential buyers on TikTik, a social medial network known for sensationalist content and little oversight. Assume the distrust factor of TikTik is 0.3. which roughly translates to the following: out of 10 buyers persuaded by ads consumed on TikTik, 3 buyers do not hold enough trust to transact via Acme’s storefront on TikTik, and instead transact elsewhere. In our case, 3,000 buyers avoid transaction on TikTik, and instead transact on a trusted marketplace retailer $R$, which for our purposes we shall call Amafon."
> 出处：2607.09608 §2.4 A simple example of demand generation under trust-driven channel choice｜L119

同一段里先说生成 **1,000** 个新买家、不信任系数 **0.3**（即十分之三转投），随后却写
**3,000** buyers avoid transaction —— 按 τ=0.3 应当是 300。**相差一个数量级，属预印本笔误。**
本卡因此**完全没有引用 §2.4 的任何数字**（只用它理解机制）；⑤ 段的缺口推算改用 §6 的设定值。

### 6.2 「跨境」在论文零命中

`grep -i` 结果：`cross-border` 0、`crossborder` 0、`e-commerce` 0、`ecommerce` 0、
`mother` 0、`baby` 0、`infant` 0、`TikTok` 0（论文用的是虚构名 `TikTik` 与 `Amafon`）。
→ registry 的 `decision_reason` 后半句「**跨境**最贵的测量缺口」是**本项目的外推判断**，
不是论文结论；卡片 ② 的母婴场景也据此显式标注为「本项目给定的业务背景」。

### 6.3 R4 硬拦截清单的边界（需要 parent 决策，本卡不擅自改 registry）

`MasterPrompt-v2.md` 的 R4 有一条：**「❌ 纯理论无实验：无任何数据集、无基线对比」**。
本文的实际情况是：**有理论（3 个 Proposition）、有一个自造的模拟研究、但没有任何真实数据集，
也没有对外部数据集的基线对比**（§6 的对比对象 truth / ITT / MMM / attribution 全部是它自己模拟出来的）。
按最严格的读法，本文会落在 R4 的边界上；按「有形式化模型 + 可复现模拟实验」的读法，它不命中。
本次按 registry 的 `decision: extract` 出卡，并在卡片开头写了**证据强度声明**；
**建议 parent 在回填 gates 时一并确认这条口径**。

### 6.4 §5 的相当一部分「生产实践」依据来自厂商白皮书

§1 贡献列表与 §5.1 都提到 ambient randomization 是「as deployed by GrowthLoop」，
引用条目指向 `GrowthLoop (2026a)/(2026b)` 的两份 **storyblok PDF 白皮书**：

> 原文："GrowthLoop (2026a) audience-level-randomization-vs-global-control. https://a.storyblok.com/f/340788/x/3afcf1437d/audience-level-randomization-vs-global-control-exec-summary-tk-7-2026.pdf."
> 出处：2607.09608 §References｜L321

> 原文："GrowthLoop (2026b) Always-on Measurement fragments. https://a.storyblok.com/f/340788/x/44b0959729/always_on_measurement-tk-7-2026.pdf"
> 出处：2607.09608 §References｜L323

这不是同行评议证据。卡片 ① 因此只把它写成**设计选择**（§5.1 的三条性质），
不把「GrowthLoop 已在大规模生产验证」写成事实。

### 6.5 预印本未校订：符号与拼写

- **符号不一致**：摘要与 §2.5 写 `\phi`（13 处），§6 全程写 `\varphi`（5 处），指同一个回收率。
  本卡正文统一写 φ，但**引文保持原文各自的写法**。
- **拼写/排版**：`eposure`(L227)、`controled`(L239)、`corret`(L131)、`multiclolinearity`(L49)
  与 `multi-colinearity`(L299) 两种拼法并存、`TikTik`/`Amafon`(L119) 为虚构品牌、L19 摘要里的
  `experiment —and` 有一个多余空格。引文一律**照原文抄**（不改错字），卡片正文不使用这些词。

---

## 7. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
C=paper2skills-vault/14-用户分析/Skill-Incrementality-Measurement.md

# 1) 三条门禁（任务给定口径）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C"
python3 paper2skills-skills/paper-审核/scripts/quote_check.py   --card "$C"
python3 paper2skills-skills/paper-审核/scripts/gate_check.py    --card "$C"

# 2) 补上 K1 凭证后重跑 K2（G1 由「无凭证」转绿）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" \
  --json-out /tmp/k1_p2s-2026-0010.json
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" \
  --k1 /tmp/k1_p2s-2026-0010.json

# 3) 引文核验器自证
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
```

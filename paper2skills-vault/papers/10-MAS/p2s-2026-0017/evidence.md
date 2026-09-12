---
title: 证据链 — p2s-2026-0017（Skill-Multi-Agent-Collaboration-Tax）
doc_type: evidence
module: 10-MAS
paper_id: 2608.22152
registry_id: p2s-2026-0017
related_card: paper2skills-vault/10-MAS/Skill-Multi-Agent-Collaboration-Tax.md
fulltext: paper2skills-vault/papers/10-MAS/p2s-2026-0017/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2608.22152 *The Collaboration Tax: How Much LLM Multi-Agent Systems Pay to Coordinate*

- **全文存档**：`paper2skills-vault/papers/10-MAS/p2s-2026-0017/fulltext.md`
  （1,381 行 / 84,796 字节，`evidence_grade: A`）
- **引文全部为逐字切片**：下面 55 条 `> 原文："…"` 都由脚本按「起止锚点」从 `fulltext.md`
  **直接切出的连续子串**（`/tmp` 下的一次性生成脚本，锚点命中唯一性在切片时校验），
  没有拼接、没有改写、没有跨段缝合。切片后逐条过 `quote_check.check_one()`，55/55 判 `VERBATIM`。

---

## 0. 核验凭证（可复核的退出码 / stdout）

以下为**现场实跑**的原始 stdout（`2026-09-12`，仓库根目录）：

```
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card $C
已索引仓库内本地模块 86 个（可自动解析的卡片将真正执行）
另有 54 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）
✅ PASS         Skill-Multi-Agent-Collaboration-Tax.md#stitched(1块)

==================================================================
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | 🟠 MIGRATED 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）
==================================================================

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card $C
✅ VERBATIM   Skill-Multi-Agent-Collaboration-Tax.md  55/55 逐字

共 1 张卡：1 通过（有引用块且逐字可核），0 含伪造引文，0 无全文可核验，0 无引用块（**不等于通过**）

$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card $C     # 未传 --k1，G1 按设计报无凭证

======================================================================
G1 门禁：0/1 通过 (0.0%)  红灯 1  黄灯 0
======================================================================
  ❌ paper2skills-vault/10-MAS/Skill-Multi-Agent-Collaboration-Tax.md
       [G1-NO-EVIDENCE] 无 K1 验证凭证 —— 请先运行 verify_skill_code.py，不得凭人工判断放行

======================================================================
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 4
======================================================================

======================================================================
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================

$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card $C --k1 /tmp/k1.json

======================================================================
G1 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================

======================================================================
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 4
======================================================================

======================================================================
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================
```

说明：第三条命令（`gate_check.py --card <卡>`，与本文档同步的口径）在**未传 `--k1`** 时
G1 必然报红（`G1-NO-EVIDENCE`：无 K1 验证凭证），这是门禁的设计——**禁止凭人工判断放行**，
不是卡片缺陷。补跑 `--k1`（K1 现场产物 JSON，本次落在 `/tmp` 下，未写入仓库）后 G1 转绿。

---

## 1. 底本 provenance 声明（⚠️ 本篇是 PDF 抽取，不是 HTML）

| 项 | 实测 |
|---|---|
| 一手来源 | `https://arxiv.org/pdf/2608.22152`（`fulltext.md` 头部注释记录的 `source`） |
| arXiv HTML（LaTeXML） | **不可得**。v1 / v2 / v3 三个版本实测均取不到 HTML 存档（`arxiv.org/abs/2608.22152v*` 无 HTML 版），因此**没有 LaTeXML 章节锚点** |
| 抽取方式 | `pdftotext` 从 PDF 抽文本 → 再把**句内硬换行接回整段** |
| 后果 1 | **引文出处只能写章节名 / 小节名**（如 `Abstract`、`5.3 Mechanism: A Four-Stage Cascade`、`Appendix A.6`）。本文档与卡片里出现的 `5.3`、`A.4`、`Table 2`、`Figure 5` 一类编号，都是**底本正文里确实印出来的**编号或交叉引用，不是从 HTML 结构反推的 |
| 后果 2 | **版面噪声真实存在**：表格被拉成一行（Table 1、Table 2、Table 3、Table 6~9 的数字与行标签被拆到同一行的不同位置）、图注与坐标轴标签混排（Figure 3/5/6/7/9）、公式变成符号串（式 (1)~(10)）、页眉与番号混入正文。**故凡表格/公式区域的数字，本卡一律不逐字引用** |
| 后果 3 | 连字符断词已由脚本接回，但**仍有残留**：底本里可见 `spatialcoordination`、`constraintsatisfaction`、`wellformed`、`unionnecessary`、`maxsuperadditivity`、`leaveoneout`、`openended`、`observationavs`（`observationalvs`）、`numberlink` 等。本卡的引文**原样保留**这些粘连形式（逐字优先于美观），读者看到时请自行还原 |
| 后果 4 | 底本 `7 Conclusion — Limitations` 处夹了一个 PDF 控制字符（`\x01`），本卡引用该段时**在控制字符之前收尾**，避免把不可打印字符带进引文 |

**未采用的替代路线**：没有改用「摘要页」「Semantic Scholar 摘要」「作者主页」等二手来源补章节号——
`MasterPrompt-v2` 明确要求引文必须是**未截断的论文原文**；二手摘要的措辞与正文常不一致，
引用它会把「出处为真」降级成「看起来像」。

---

## 2. registry 逐项核对表（`papers_registry.json` → `p2s-2026-0017`）

| registry 字段 / 断言 | 原文核对结论 | 依据 |
|---|---|---|
| `arxiv: 2608.22152` | ✅ 一致 | 底本页眉 `arXiv:2608.22152v1 [cs.CL] 23 Aug 2026` |
| `published: 2026-08-23` | ✅ 一致 | 同上（`v1` 日期） |
| `title` | ✅ 一致（逐字） | 底本首行 |
| `decision: extract` | ✅ 无异议 | — |
| `decision_reason`：「**32 任务**」 | ✅ **成立** | 摘要 `32 solo-tractable tasks grouped by source of grounding friction`（Q6）；Appendix F `The suite spans 32 tasks across three structural families.`；Table 8 逐项列出 Spatial 11 + Relational 11 + CSP 10 = 32 |
| `decision_reason`：「**11 模型**」 | ⚠️ **成立但需收窄**：原文写的是 `11 models from **7 providers**`，registry 漏掉了「来自 7 家提供方」这个限定 | 摘要逐字：`measure it on 11 models from 7 providers`（Q6）；5.1 节 `We evaluate eleven models from seven providers:` 并列出名单（Q7）。**另注**：论文自己在贡献列表里写成 `11 models from 7 **families**`——同一篇论文里 `providers` 与 `families` 两种措辞并存（计数都是 11 与 7）。本卡按摘要口径记「11 models / 7 providers」 |
| `decision_reason`：「量化'多 Agent 协作税'」 | ✅ 成立 | Q1 / Q2 |
| `decision_reason`：「四阶段失败级联」 | ✅ 成立，逐字命中 | Q16（摘要）；5.3 节四段定义（A1、A2、Q18、Q19） |
| `decision_reason` 的编者按语：「→ **什么时候不该拆多 Agent**」 | ⚠️ **不是论文原句**，是 registry 编者的提炼。**但该判断在论文里有逐字依据**：退化划分使 tax 恒等于 0（Q4）、视图可平凡拼接时上界取等（Q5）、solo-trivial 是测量前提（Q8） | Q4 / Q5 / Q8 / Q9 / Q10 |
| `note`：「EMNLP 2026 主会」 | ⚠️ **底本无法核验**。全文 `EMNLP` / `Empirical Methods` 只命中 1 处，且出现在**参考文献** Sun et al. (2025) 里，与本论文自身无关；底本页面**没有任何 venue 标注**（无页眉会议名、无 camera-ready 脚注）。按 registry 口径记 `venue: EMNLP 2026`，但**本卡不据此声称「已录用」** | `grep -c "EMNLP\|Empirical Methods" fulltext.md` → 1（命中位置在 References） |
| `venue_tier: top` | ⚠️ **不是 MasterPrompt-v2 的合法枚举值**。v2 规定 `venue_tier ∈ {CCF-A, CCF-B, UTD24, FT50, preprint, non-paper}`；`top` 是 registry 自造标签。按 registry 原样记录以免下游对不上号，**同时提示**：按 CCF 目录 EMNLP 属 **CCF-B** | MasterPrompt-v2 frontmatter 规范 |
| `note`：「现有 **16 张卡**全在讲'怎么搭'」 | ⚠️ **计数口径不明，需 registry 维护者确认**。本卡撰写时实测：`10-MAS/` 域（含 `00-知识库-Skill卡片/` 子目录）共 12 张 `Skill-*.md`（另有本次并行新增的 `Skill-Routed-Graph-Handoff.md` 与本卡，合计 14）；`16-智能体工程/` 域恰好 16 张。**「16 张」与后者的数量吻合，疑为跨域计数**。本卡的定位不受影响（10-MAS 域内确实没有「该不该拆」主题的卡） | `find paper2skills-vault/10-MAS -name "Skill-*.md" \| wc -l` → 12（新卡创建前）；`find paper2skills-vault/16-智能体工程 -name "Skill-*.md" \| wc -l` → 16（并行新卡创建前） |
| `data_availability: "available"` | ⚠️ **部分采信**：论文在 Ethics Statement 里**承诺**发布任务生成器 / 切分器 / 评分器 / prompt / 匿名化对话转录，但底本里对应的 `Code` / `Website` 标记只剩符号串（PDF 抽取把超链接丢了），**看不到任何可读 URL**。故记 `available（论文宣称将发布，底本未给出可读链接）`，不作为「已可下载」使用 | Ethics Statement 段 |
| `score: 46.4` / `priority: P0` / `outputs.*` | 不核验（registry 自记项） | — |

### 与 registry 的结论差异汇总

1. **「11 模型」应写成「11 models from 7 providers」**（可再注明论文自己在贡献列表里写成 `7 families`）。
2. **venue 无法从底本核验**，只能沿用 registry；`venue_tier: top` 不属 v2 枚举（EMNLP 按 CCF 为 CCF-B）。
3. **`data_availability: available` 应理解为「宣称将发布」**，底本无可用链接。
4. **`note` 的「16 张卡」计数口径需确认**（10-MAS 域实测 12 张）。
5. 「**什么时候不该拆多 Agent**」是编者按语而非论文原句，但**有逐字理论依据**（Q4 / Q5 / Q8）。

---

## 3. 卡片正文数字 → 出处对照表

卡片正文里出现的**每一个阿拉伯数字**都在下表内；表外的数字只有三类：
① v2 模板强制的 frontmatter 字段（`created` / `updated` 日期、`module: 10-MAS` 里的域编号，
由 G2 记为 4 条黄灯，见 §0）；② 结构指代（`Table 2`、`Figure 5`、`Appendix A.6`、arXiv ID `2608.22152`
—— 由门禁按「结构前缀」规则豁免）；③ ③ 段代码围栏内的**合成示例数据**（围栏在 G2 中被整块剥离）。
**业务侧的建议值一律用中文数字**（如「三十个案例」「一百条」），并在卡片 ② 段显式声明该约定。

| 卡片中的数字 | 含义 | 出处（章节名） | 引文键 |
|---|---|---|---|
| `32`（任务）、`11`（模型）、`7`（providers） | 实验规模（registry 头条数字） | Abstract | Q6 |
| `eleven` / `seven` + 完整模型名单 | 参与评测的模型清单 | 5.1 Experimental Setup | Q7 |
| `50`（次 rollout / cell） | cell 均值的采样口径 | 3 The Collaboration Tax（Operational form） | Q2 |
| `50`（次独立 rollout）、独立 seed | 生成与评分协议 | Appendix I Hyperparameters | A4 |
| `50`（轮对话交换上限） | 配对对话的硬截断 | Appendix I Hyperparameters | A5、Q39 |
| `700`（条失败描述）→ `16`（个行为主题） | 四阶段 judge 的归纳来源 | 5.3 Mechanism: A Four-Stage Cascade | A22 |
| `67%` | 失败 rollout 中「至少两个阶段同时失守」的占比 | Appendix D Cascade Co-occurrence | Q44 |
| `0.67`（Jaccard） | L3 × L4 失败集合的重叠度（纠缠证据） | Appendix D Cascade Co-occurrence | Q45 |
| `0.475` / `0.760` / `0.705` | 对话特征 → ratio gap 的 out-of-fold R² / Spearman ρ / Pearson r | Figure 4 图注 | Q24 |
| `95%`（置信区间） | all-four 干预与 leave-one-out 的区间口径 | 5.5 Intervention | Q27、Q29 |
| `0.575` → `0.569`（差 `0.006`） | gpt-4o-mini 上 critic 复核对 solo 分数的贡献（排除「税来自 critic」） | Appendix H Additional Results | Q41 |
| `0.932` / `0.668` | 强成员单人收益 vs 其 Shapley 份额（nano × gpt-5 配置） | Appendix A.5 Shapley value and subadditivity | A7 |
| `+0.732` / `-0.170` | hetero gap 与「两者中点」的 Pearson r 与均值差（落在中点之下） | Figure 5 图注 | A12 |
| `+12` / `+6` / `+2` … 等 Figure 3 单元格数值 | **本卡不引用**（版面噪声：数值与模型标签被拆散，无法逐字成句） | — | — |
| Table 1 / Table 2 表内数值 | **本卡只在正文定性引用其结论**，不逐字引用表内数字（同上） | 5.5 / 6 | Q27、Q29、Q33、Q43 |

---

## 4. 逐字引文全表（55 条，与卡片 ⑥ 段一一对应）

> 每条都由脚本从底本按锚点切成连续子串；出处写章节名 / 小节名（见 §1）。
> 键名仅供卡片与本文档互相回指。

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

> 原文："L2 failures are silences: each utterance is wellformed; what is missing is a question."
> 出处：2608.22152 Appendix E Case Studies（Cross-case summary）｜Q21

> 原文："L3 failures are truncations: the dialogue ends or commits before a merged-state utterance ever appears."
> 出处：2608.22152 Appendix E Case Studies（Cross-case summary）｜Q22

> 原文："67% of failed rollouts fire at least two stages, justifying the §5.3 claim that the four stages are separable but not independent."
> 出处：2608.22152 Appendix D Cascade Co-occurrence｜Q44

> 原文："L3 and L4 co-fire at Jaccard 0.67, the largest overlap by a wide margin, which motivates the observationalvs-causal reconciliation in §5.3 and §5.5."
> 出处：2608.22152 Appendix D Cascade Co-occurrence｜Q45

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

---

## 5. 未能核验 / 只能定性陈述的清单

本卡在写作中**主动放弃**了以下内容，原因是无法逐字核验、或论文本身没给：

1. **母婴 / 跨境电商场景的任何数字** —— 论文完全没有这个话题。卡片 ② 段的品类、平台、
   字段与流程全部是业务背景（由业务方给定），**没有任何数字来自论文**；ROI 公式里的
   `ΔS`、`V_decision`、`N_case`、`C_measure` 是符号而非数值。
2. **工程成本 / token 消耗 / 延迟 / 人力投入** —— 论文未报告任何量级。卡片 ⑤ 明确标注「需企业自估」。
   论文只报告了评测侧的规模（50 rollouts/cell、50 轮交换上限），不是成本。
3. **Table 1 的具体数值**（all-four 与四条 leave-one-out 的 Δs、% closed 及其置信区间）——
   底本里该表被拉成一行，数值与行标签错位，逐字引用会误导读者；卡片只引用正文的定性结论
   （Q27 / Q28 / Q29）。
4. **Table 2 的具体数值行**（四组配对各自的 `v({1})`、`v({2})`、`v({1,2})`、`φ1`、`φ2`）——
   同上（版面噪声）。卡片只引用其表述（Q33、Q43）与 Appendix A.5 正文里可成句的一组对照
   （`0.932` vs `0.668`，A7）。
5. **Table 3~Table 9 的表内数值**（失败主题聚类计数、标注者一致性 κ 矩阵、阶段共现分布、
   critic 消融的绝对分数等）—— 只在能连续成句处引用（A14、Q44、Q45、Q41），其余不复述。
6. **Figure 3 / 6 / 7 / 9 的单元格数值**（各 (model, category) 的 ratio gap、各任务 ratio gap、
   M1 ridge 的标准化系数、异质配对热力图）—— 底本里这些数值与坐标轴标签、图例混排，
   无法逐字成句；卡片只引用正文对这些图的定性描述（Q13 / Q14 / Q15 / Q25 / Q26）。
7. **三个任务族的 per-family 数字级差异** —— 卡片只给论文的类别序（Spatial ≻ Relational ≻ CSP，Q13）
   与「随能力单调下降」（Q14），**不给具体百分点**。
8. **任务级细节**（32 个任务各自的生成器 / 切分器 / 评分规则）—— Appendix F 给了 Table 8 的
   一行式摘要，正文只对 maze / relaquery / schedule 三个代表任务做了深挖；卡片未逐项复述。
9. **超参数**（temperature、top-p、输出上限、各模型的默认参数差异）—— 卡片只引用有明确数值的
   三条（50 rollouts、50 轮上限、grader 固定为 gpt-4o-mini；A4 / A5 / A6），其余未给。
10. **API 侧工程细节**（120 秒超时、重试次数、Azure 内容过滤触发条件）—— Appendix J 有，但与
    本卡的决策用途无关，未引用。
11. **venue** —— 底本无任何 venue 标注（见 §2），只能沿用 registry。
12. **与作者实现的一致性** —— 论文承诺发布代码与匿名化转录（Ethics Statement），但底本无可读链接；
    K1 只证明**卡片自带代码**可执行（L5 断言全绿），**不证明**它与论文实现一致，也不证明论文的
    32 个任务能被复现。
13. **「tax 测量」的工程化流程** —— 论文没有任何关于「把税测量做成常态化监控」的讨论；
    卡片 ①b 显式标注「论文未讨论」。

---

## 6. 论文自承局限要点（回原文逐条摘出，供卡片 ①b 引用）

| # | 局限 | 原文键 | 对本卡的影响 |
|---|---|---|---|
| 1 | **只测两人（N = 2）**，N ≥ 3 时私有视图结构与对话动力学都发生质变，论文称**未被研究** | Q35、Q40 | 不得外推到多人团队；形式化范围明确限于 2-agent + 单一共同收益 + union-necessary |
| 2 | **任务是程序生成的谜题**，论文自陈该设定**不能直接对应**协同改代码 / issue 分诊 / 多轮 debug / 文档起草，那些场景「答案空间与反馈信号远没有那么结构化」 | Q36、Q37 | ② 是把方法搬到业务上自测，不是把论文结论搬过来 |
| 3 | **存在负税 cell**（强模型在 CSP 类上 tax 为负），论文自陈违反经典 tax ≥ 0，原因是 LLM 策略随机且依赖上下文 | Q38 | 「税为正」不是定理；自己的数据里出现负税要能解释 |
| 4 | **对话硬截断**；论文推测 all-four 干预后残余的税有一部分来自这个上限 | Q39 | 自家实验必须报告轮数上限，并与论文口径对齐 |
| 5 | **异质配对只有两对模型、四种配置**，论文自称是「定性模式的 existence proof，而非定量刻画」，且家庭级效应不可检测 | Q34 | Table 2 的合作博弈数字只能当结构性示意 |
| 6 | **回归不能外推绝对水平**：留一模型时秩保持、绝对水平丢失；能力设截距、对话形状设斜率 | Q26、Q25、A17 | 「用别家的系数直接套自己的绝对值」不成立 |
| 7 | **judge 的人工一致度有上限**（四维中等偏上；自动化 judge 与专家多数票在 L4 上只有中等） | A14 | 用 judge 做考核前先自测一致度 |
| 8 | **solo 侧带 critic 复核** | Q41 | 对照实验必须对齐「solo 有没有复核机会」，否则读数不可比 |

---

## 7. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
C=paper2skills-vault/10-MAS/Skill-Multi-Agent-Collaboration-Tax.md
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C"
python3 paper2skills-skills/paper-审核/scripts/quote_check.py   --card "$C"
python3 paper2skills-skills/paper-审核/scripts/gate_check.py    --card "$C"

# G1 需要 K1 凭证（否则按设计报 G1-NO-EVIDENCE）：
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" --json-out /tmp/k1.json
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" --k1 /tmp/k1.json
```

- 引文自检（证明核验器本身能区分真引文 / 伪造 / 拼接）：
  `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest`

---

## 8. 附：本卡 ③ 段代码的定位（避免误读）

- ③ 段代码是**测量模板**，不是论文实现的重现。它用**合成数据**跑通
  「算税 → 四阶段判定 → 用对话特征预测税 → 阶段子句消融 → 异质配对拉向强者 / Shapley 份额」全链路，
  六个 `test_*` 断言全部落在**合成结构**上，**不构成对论文任何数值的复现**。
- 代码内的自检常数（`0.90 / 0.35 / 0.70` 等）是**合成值**，与论文 Table 2 的数值无关：
  后者只在卡片 ⑥ 段以逐字引文形式出现（A7 / Q33 / Q43）。两类数字**不得互相印证**。

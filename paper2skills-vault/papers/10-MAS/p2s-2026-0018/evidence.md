---
title: 证据链 — p2s-2026-0018（Skill-Routed-Graph-Handoff）
doc_type: evidence
module: 10-MAS
paper_id: 2608.25277
registry_id: p2s-2026-0018
related_card: paper2skills-vault/10-MAS/Skill-Routed-Graph-Handoff.md
fulltext: paper2skills-vault/papers/10-MAS/p2s-2026-0018/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2608.25277 *Routed Graph Handoff: Adaptive Format Selection for Multi-Agent LLM Delegation*

- **全文存档**：`paper2skills-vault/papers/10-MAS/p2s-2026-0018/fulltext.md`
  （412 行 / 39,645 字符，来源 `https://arxiv.org/html/2608.25277v1`，`evidence_grade: A`）。
- **引文全部为逐字切片**：下文每条 `> 原文："…"` 都不是手打的，而是用锚点从 `fulltext.md`
  **按起止位置直接切出的连续子串**（脚本见 §5），因此没有拼接、没有改写、没有跨段缝合。
  切片脚本会拒收跨行引文与含直引号的引文，切完再跑 `quote_check.py` 复核。
- **存档来源是 HTML 而非 PDF**：`fetch_fulltext.py` 抓的是 arXiv LaTeXML HTML，
  **没有 PDF 分页信息**，所以 ⑥ 段的出处只标章节号，不标页码。
- **作者单位**：Amazon, AGI / Sunnyvale, USA（存档第 12 行）。

## 0. 核验凭证（可复核的退出码与 stdout）

以下为**真实 stdout 逐字粘贴**，不是概述。命令见 §5。

```
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" --timeout 60
已索引仓库内本地模块 86 个（可自动解析的卡片将真正执行）
另有 54 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）
✅ PASS         Skill-Routed-Graph-Handoff.md#stitched(1块)

==================================================================
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | 🟠 MIGRATED 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）
==================================================================
exit=0

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$C"
✅ VERBATIM   Skill-Routed-Graph-Handoff.md  51/51 逐字

共 1 张卡：1 通过（有引用块且逐字可核），0 含伪造引文，0 无全文可核验，0 无引用块（**不等于通过**）
exit=0

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
1 真引文     : VERBATIM   连续度=1.0
2 纯伪造     : FABRICATED 连续度=0.124   ← 必须 FABRICATED
3 拼接引文   : FABRICATED 连续度=0.681 召回=1.0 拼接标记=True   ← 必须被拦下
4 排版差异   : VERBATIM   连续度=1.0   ← 必须 VERBATIM
✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接
exit=0

$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C"
======================================================================
G1 门禁：0/1 通过 (0.0%)  红灯 1  黄灯 0
======================================================================
  ❌ paper2skills-vault/10-MAS/Skill-Routed-Graph-Handoff.md
       [G1-NO-EVIDENCE] 无 K1 验证凭证 —— 请先运行 verify_skill_code.py，不得凭人工判断放行

======================================================================
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 3
======================================================================

======================================================================
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================
exit=1   ← 仅因未传 K1 凭证，不是卡片缺陷

$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" --quiet --json-out /tmp/k1_card.json
$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" --k1 /tmp/k1_card.json
======================================================================
G1 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================

======================================================================
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 3
======================================================================

======================================================================
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================
exit=0
```

**G2/G3 机读指标**（来自 `gate_g2_*.json` / `gate_g3_*.json`）：

| 指标 | 值 |
|---|---|
| `metric_numbers` / `sourced` | **33 / 33**（`traceability_pct = 100.0`） |
| `unsourced_metric`（红灯） | **0** |
| `unsourced_general`（黄灯） | 3 —— 全部来自 frontmatter：`module: 10-MAS` 的 `10`、两处 `2026-09-12` 的 `12` |
| `quotes_total` / `quotes_verbatim` | **51 / 51**（`fuzzy 0`、`fabricated 0`、`spliced 0`） |
| `sourced_structural_only`（G2c） | **0** |
| `cn_numeral_claims`（无法机械核验的中文量级词） | **0** |
| G3 `concrete_signals` / `vague_phrases` | 13 / 0 |
| G3 `has_data_requirement` / `has_roi_basis` / `skill_relations` | true / true / 6 |

> **口径提醒（不要混淆两件事）**：G1 那条红灯是「**没把 K1 凭证传给 gate_check**」，
> 不是代码跑不过。K1 本体（L1 语法 / L2 编译 / L3 import / L4 脚本执行 / L5 pytest 断言）
> 五级全绿、退出码 0。这与 `CLAUDE.md` 里「三个门禁必须分开报」的要求一致：
> 卡片代码**已验证**，不要拿 G1 红线去否定 K1 的 PASS，也不要拿 K1 的 PASS 去替代 G1 的凭证要求。

## 1. 与 registry 的逐项核对

registry（`papers_registry.json` → `p2s-2026-0018`）原文：

| 字段 | registry 值 | 逐项核对结论 |
|---|---|---|
| `decision_reason` | 「Routed Graph Handoff：NL 交接吞 40-60% token，155-token router 选类型化依赖图；τ-retail +12.7pp @3.2× 压缩」 | **数字全部成立**，但**「3.2×」的口径必须澄清**（见 §1.1） |
| `note` | 「EMNLP 2026；须配 graph-aware executor prompt，否则失效」 | 后半句 ✅ 成立（⑥ Q5 / Q12 / Q13 / Q41）；「EMNLP 2026」**⚠️ 无法核验**（见 §1.2） |
| `venue` / `venue_tier` | `EMNLP` / `top` | **⚠️ 全文零命中，registry 单方声明**（见 §1.2） |
| `data_availability` | `available` | **⚠️ 需收窄**（见 §1.3） |
| `paper_id` / `arxiv` / `title` | `p2s-2026-0018` / `2608.25277` / 标题 | ✅ 与存档头部注释、存档标题一致 |
| `published` | `2026-08-26` | ✅ 与 arXiv HTML 版本可获取时间一致，未发现撤稿标记 |
| `domain` / `priority` / `decision` | `10-MAS` / `P0` / `extract` | 未核验（评分与选题口径不在本卡范围） |
| `outputs.code_dir` | `paper2skills-code/mas/<algo>` | **规划值，未落地**（与 `CLAUDE.md` 说明一致）；代码模板内嵌在卡片 ③ 段，由 K1 验证 |
| `outputs.evidence` | `paper2skills-vault/papers/10-MAS/p2s-2026-0018/evidence.md` | ✅ 即本文件（路径正确） |
| `outputs.skill_card` | `paper2skills-vault/10-MAS/Skill-<方法名>.md` | ✅ 实际落地为 `Skill-Routed-Graph-Handoff.md` |

### 1.1 「3.2× 压缩」口径澄清（本卡最重要的一处订正）

registry 头条写「τ-retail +12.7pp **@3.2× 压缩**」，三处原子事实都能逐字命中：

1. 摘要：`+12.7 pp on $\tau$-retail at 3.2$\times$ compression ($p{<}0.01$)`（⑥ Q3）；
2. §3.1 Table 1 表注：`$\tau$-retail: +12.7 pp (150 paired trials, $p{<}0.01$)`（⑥ Q22）；
3. §3.4 Table 2 中 `Routed NGH` 一行的四个单元格（`24.7 / +12.7 / [+6.0, +19.3] / 3.2`）——
   该表格行在存档里是 `$\times$| schema | Routed NGH | 24.7 | +12.7 | [+6.0, +19.3] | 3.2 |`
   （列头与数据混在一起），故此处只作单元格引用，不当作逐字引文使用。

**但 3.2× 只是 τ-retail 的单点值，不是普遍压缩率。** 论文自己的三档口径是：

| 口径 | 数值 | 出处 |
|---|---|---|
| 交接 token 加权平均 | **2.1×**（BrowseComp 2.2× / τ-retail 3.2× / BFCL 2.0× / **AppWorld 1.04×**） | ⑥ Q26（§3.2） |
| 全口径单次委派预算（router 调用 + graph-aware executor prefill 都算） | τ-retail **461 vs 730 token = 1.6×** | ⑥ Q27（§3.2 + Appendix H） |
| 摘要里的「2–3×」 | schema 设计期的口径（`${\sim}$350 tokens per delegation (2$\times$ compression vs. NL)`） | ⑥ Q8（§1） |

**卡片据此做了两件事**：① ①b 第 3 条把三档口径并列写出，并明确要求 ROI 用 **1.6×** 而不是 3.2×；
② ⑤ 段的参数表在 `T_graph_total` 一栏直接写明「**不要用 3.2× 做预算**」。
registry 那句「@3.2× 压缩」本身没错，但**单独看会让人以为普遍能省到 3 倍**——这是本次核对的主要产出。

### 1.2 venue `EMNLP 2026` / tier `top`：**未能核验**

对全文存档 grep `EMNLP` / `ACL` / `conference` / `proceedings` / `findings` / `workshop`，
**唯一命中全部集中在参考文献条目**（Jiang et al. 2023 的 EMNLP 会议名、Pan et al. 2024 的
Findings of ACL 2024、Wu et al. 2024 的 ICLR workshop 等），**论文自身没有任何 venue 标注**：
存档页首只有标题与作者，无会议页眉、无 "Accepted at"、无 proceedings 信息。

- 卡片 frontmatter 按**任务口径 + registry** 记 `venue: EMNLP 2026` / `venue_tier: top`；
  注：v2 规范的 `venue_tier` 枚举是 `CCF-A|CCF-B|UTD24|FT50|preprint|non-paper`，
  registry 用的是 `top`，本卡从 registry。
- **下游使用建议**：在拿到录用证据（ACL Anthology 卷号 / OpenReview decision / 会议 program）之前，
  **不要**把这张卡当「EMNLP 2026 主会论文」对外引用。本仓库自己的实测记录也提示
  「`EMNLP 2026` 在 OpenReview 上返回的条目实为 workshop 名」——即该 venue 字符串
  在本仓库当前的检索链路上**不足以自证为主会**。
- 本卡的**技术结论不依赖 venue**：所有机制与数字都由 arXiv 存档逐字支撑（51 条 VERBATIM）。

### 1.3 `data_availability: available`：**需要收窄为「公开 benchmark + artifact 承诺放出」**

论文原文（Appendix J）：`Code and configurations will be released upon publication.` —— 即
**截至本存档抓取时并未放出**。另外 Ethical Considerations 明确：实验全部通过商用 API 访问
已发布的基础模型（Claude Sonnet 4.5 / Amazon Nova Pro via AWS Bedrock / GPT-5 mini）。

- 所以 `available` 的正确含义是「**benchmark 公开、artifact 承诺随发表放出**」，
  **不是**「任何人现在就能把论文跑通」。
- 复现论文需要：商用 API 权限与费用、`$\tau$-bench`/BrowseComp/BFCL/AppWorld 四个 benchmark 的
  访问、以及论文手工设计的 schema 与三份 prompt（schema 与 prompt 在 Appendix E / J 有描述，
  但完整 artifact 未放出）。卡片 ③ 段的代码因此**刻意做成标准库替身**，不冒充论文实现。

## 2. 卡片正文数字 → 出处对照表

卡片正文（③ 段代码围栏**之外**）出现的每一个带度量语义的数字：

| 卡片中的数字 | 含义 | 卡片位置 | 引文键 |
|---|---|---|---|
| `76%` | 多 Agent 失败中「agent 间错位」的占比 | ①③ / ①b / ② | Q7, Q34 |
| `155` token | 一次 router 分类调用的 token 量级 | ① / ①b / ② | Q2, Q9, Q14, Q27, Q47 |
| `−14.6 pp` | AppWorld 上 graph-only 相对 NL 的回退 | ①b 第 1 条 | Q22, Q24, Q30 |
| `−22.8` / `−6.4` | 上述回退的 95% CI 下/上界 | ①b 第 1 条 | Q24 |
| `152` paired trials | AppWorld 的配对试验数 | ①b 第 1 条 | Q20 |
| `+12.7 pp` | τ-retail 上相对 NL 的提升 | ①b 第 2 条 | Q3, Q12, Q22, Q24, Q33, Q37 |
| `3.2×` | τ-retail 的**交接 token** 压缩比（单点值） | ①b 第 3 条 | Q3, Q26, Q33 |
| `2.1×` | 四个 benchmark 加权平均压缩比 | ①b 第 3 条 / ⑤ | Q26, Q47 |
| `2.2×` / `2.0×` | BrowseComp / BFCL 的分项压缩比 | ①b 第 3 条 | Q26 |
| `1.04×` | AppWorld 的分项压缩比（几乎不省） | ①b 第 3 条 / ⑤ | Q26 |
| `461` / `730` | 全口径单次委派 token（τ-retail，含 router 与 prefill） | ①b 第 3 条 / ⑤ | Q27 |
| `1.6×` | 上述全口径的比值 | ①b 第 3 条 / ⑤ | Q27 |
| `89%` | AppWorld 上被 router 判给 NL 的任务比例 | ①b 第 4 条 / ⑤ | Q18, Q19, Q25, Q32, Q39, Q43 |
| `11%` | AppWorld 上被 router 判给 GRAPH 的比例 | ①b 失败模式 1 | Q18, Q19, Q32 |
| `8.6 pp` | oracle 相对 routed 的剩余余量 | ①b 第 5 条 | Q6, Q36, Q44 |
| `+6.7 pp` | AppWorld aggregate 模式上图后的提升（n=15） | ①b 失败模式 2 | Q35 |
| `15` | aggregate 任务数；也是被误判进 GRAPH 的非聚合任务数 | ①b 失败模式 2 | Q35, Q43 |
| `3.4 pp` | router 误差造成的损失（把不该图化的任务图化了） | ①b 失败模式 2 | Q44 |
| `−4.0 pp` | τ-airline 上 graph-only 的回退 | ①b 失败模式 3 | Q31 |
| `100%` | 依赖链 benchmark 上图化比例（routing 粒度粗的证据） | ①b 论文自承局限 1 | Q18, Q39, Q43 |
| `47` | schema 手工迭代所用的 τ-bench 轨迹数 | ①b 论文自承局限 2 | Q11 |
| `0.0005` 美元 | router 单次分类调用的成本（论文给的唯一金额） | ⑤ | Q14 |
| `15–27` 个重试步 | 依赖链失败时被依赖边消掉的重试步数 | ⑤ | Q28 |
| `5.2 pp` / `3.4 pp` | 8.6 pp 余量的分解（NGH 救回 / NL 救回） | 仅 ⑥ 段，未在正文断言 | Q44 |
| `3` 个 sub-agent | 业务场景里拆分的 sub-agent 数量 | ② | **业务设定，非论文断言** |
| `8` 类节点 / `7` 类边 | schema 的节点与边类型数 | ② / ③ 说明 | Q10 |

**构造数据（不是论文数字，已在卡上显式声明）**：③ 段代码里的任务条数（`chain` 20 / `agg` 5 /
`adapt` 15 / `mixed` 5，共 45 条）、`i % 4 != 0` 等重排条件，以及 `_business_demo()` 打印的成功计数，
**全部是合成样本**，
落在代码围栏内（G2 的 B 类「本地可复现数字」），与被剥离的证据语法行一样不计为事实断言。
卡片 ③ 段与代码 docstring 里两处明确写了：**与论文 pp 数不存在任何对应关系，数值接近纯属巧合**。

**为什么样本里刻意留了 5 条「router 会误判」的任务**：论文 Appendix D 自己承认 router 会把
带部分序结构的任务当成纯聚合（15 个任务被误送进 GRAPH，3.4 pp 的损失计在 router 误差里）。
一个**零误判**的规则替身会比论文的 router 还乐观，那不是忠实的替身——所以卡片的第 8 条断言
专门要求替身复现这个误判方向，ROUTED 在构造样本上因此**不是满分**（这正是它的可信之处）。

**正面排除的一处风险**：卡片**没有**把「论文的 +12.7 pp」与「本卡模拟的成功率」并列成一句话，
而是分处 ①b 段与代码围栏，符合 `MasterPrompt-v2.md`「两类不得混写」的要求。

## 3. 逐字引文全表（51 条，与卡片 ⑥ 段一一对应）

切片方式：以 (起始锚点, 结束锚点) 从 `fulltext.md` 取**连续子串**，锚点内的空格用
`[\s\u2009\u00a0]*` 匹配以容忍论文排版的 thin space（U+2009，存档里数字与单位之间大量使用，
例如 `14.6\u2009pp`、`temperature\u2009=\u20090`）。脚本拒绝跨行引文与含直引号的引文。

### A. 核心主张、机制与规模

> 原文："Multi-agent LLM systems coordinate through natural-language messages that consume 40–60% of their token budget."
> 出处：2608.25277 §Abstract｜Q1

> 原文："We propose Routed Graph Handoff, where a lightweight LLM router (155 tokens, 0.15% overhead) selects between a typed dependency graph and natural language for each delegation."
> 出处：2608.25277 §Abstract｜Q2

> 原文："On four benchmarks (1,050+ trajectories), the routed system matches or exceeds NL-only on every task: +12.7 pp on $\tau$-retail at 3.2$\times$ compression ($p{<}0.01$), +8.7 pp on BrowseComp at 2.2$\times$ compression ($p{<}0.05$), and parity on BFCL and AppWorld."
> 出处：2608.25277 §Abstract（BrowseComp 2.2× 与 BFCL/AppWorld 打平）｜Q3

> 原文："Without the router, graph-only delegation regresses 14.6 pp on AppWorld; the router eliminates this at near-zero cost."
> 出处：2608.25277 §Abstract｜Q4

> 原文："A graph-aware executor prompt is required: the same schema without interpretation guidance yields no gain."
> 出处：2608.25277 §Abstract｜Q5

> 原文："An oracle analysis reveals 8.6 pp of additional headroom, motivating execution-time adaptive routing as future work."
> 出处：2608.25277 §Abstract｜Q6

> 原文："Error analysis on 345 multi-agent trajectories reveals that 76% of failures stem from inter-agent misalignment: the executor misinterprets ordering constraints, drops prerequisites, or loops on ambiguous instructions."
> 出处：2608.25277 §1 Introduction｜Q7

> 原文："We resolve this tradeoff with Routed Graph Handoff (Figure 1): a lightweight LLM router (${\sim}$155 tokens, 0.15% overhead) selects graph or NL per delegation based on task computational pattern."
> 出处：2608.25277 §1 Introduction｜Q9

### B. schema、router 与 graph-aware executor

> 原文："Each delegation is encoded as a typed DAG with 8 node types (goal, constraint, entity, action, precondition, postcondition, tool_call, tool_arg) and 7 edge relations (requires, targets, blocks, enables, depends_on, contradicts, follows)."
> 出处：2608.25277 §2.1 Native Graph Handoff Schema｜Q10

> 原文："We design such a schema (8 node types, 7 edge relations) emitted via constrained decoding at ${\sim}$350 tokens per delegation (2$\times$ compression vs. NL)."
> 出处：2608.25277 §1 Introduction｜Q8

> 原文："The schema was designed iteratively on 47 $\tau$-bench trajectories."
> 出处：2608.25277 §2.1｜Q11

> 原文："This receiver-side instruction is essential: passing the same JSON to a standard executor prompt yields no gain, and on $\tau$-retail restoring it lifts NGH from below NL to +12.7 pp (Appendix E)."
> 出处：2608.25277 §2.1 Graph-aware execution is part of the interface｜Q12

> 原文："We therefore treat the typed graph and its interpretation guidance as a single mechanism, not the graph alone."
> 出处：2608.25277 §2.1｜Q13

> 原文："Without this prompt (i.e., passing the JSON graph to a standard executor prompt), the sub-agent treats the graph as opaque data and fails to interpret the dependency structure."
> 出处：2608.25277 §Appendix E｜Q45

> 原文："Before each delegation, a single classification call (${\sim}$155 tokens total, $0.0005) decides whether to use graph or NL."
> 出处：2608.25277 §2.2 LLM Router｜Q14

> 原文："Pick GRAPH if the task requires deterministic answers that depend on ordered sub-tasks (aggregations, multi-step lookups, sequential API calls). Pick NL if the task requires iteration, conditionals, free-text interpretation, or adaptive reasoning."
> 出处：2608.25277 §2.2 LLM Router（router prompt 原文）｜Q15

> 原文："Conservative default: NL unless dependency-chain pattern is detected. This ensures zero NL wins are sacrificed."
> 出处：2608.25277 §2.2｜Q16

> 原文："Deterministic: temperature = 0, verified identical across 3 independent runs."
> 出处：2608.25277 §2.2｜Q17

> 原文："The per-benchmark rates we report are therefore a post-hoc aggregate of these blind per-task decisions: 100% graph on BrowseComp/$\tau$-retail/BFCL; 11% graph / 89% NL on AppWorld; 2% graph on $\tau$-airline."
> 出处：2608.25277 §2.2｜Q18

> 原文："That the same classifier splits AppWorld itself 11%/89% (which a fixed per-benchmark rule cannot do) confirms the decision is made per task, not per domain; the clustering by benchmark arises because within each of these benchmarks nearly every task shares the same better format."
> 出处：2608.25277 §2.2｜Q19

### C. 主结果、规模与错因

> 原文："We evaluate on four diverse multi-agent tasks: BrowseComp (Wei et al., 2025) (150 trials, long-horizon web search requiring multi-step evidence gathering), BFCL v3 (Patil et al., 2025) (600 trials, Berkeley Function Calling Leaderboard with complex API sequences), $\tau$-bench retail (Yao et al., 2025) (150 paired trials: 50 tasks $\times$ 3 seeds, multi-step customer service with tool calls), and AppWorld (Trivedi et al., 2024) (152 paired trials, multi-app tool use with conditional logic). Total: 1,052 trajectories."
> 出处：2608.25277 §3 Experiments｜Q20

> 原文："Splits. Pinned 50 $\tau$-retail tasks $\times$ 3 seeds; BrowseComp 150; BFCL v3 600; AppWorld 152; $\tau$-airline 150. Total 1,052 trajectories (plus 150 $\tau$-airline for the router ablation)."
> 出处：2608.25277 §Appendix J Splits｜Q48

> 原文："*Table 1: Main results (task success / accuracy %). Routed matches or exceeds NL on all four benchmarks. $\tau$-retail: +12.7 pp (150 paired trials, $p{<}0.01$). BrowseComp: +8.7 pp, CI [+2.7, +14.7], $p{<}0.05$. AppWorld NGH-only regresses $-$14.6 pp; the router recovers parity.*"
> 出处：2608.25277 §3.1 Table 1 表注｜Q22

> 原文："The router’s primary function is regression prevention (Table 1). NGH delivers significant gains"
> 出处：2608.25277 §3.1｜Q23

> 原文："NGH delivers significant gains on dependency-chain tasks: +12.7 pp on $\tau$-retail (150 paired trials; $p{<}0.01$) and +8.7 pp on BrowseComp (CI [+2.7, +14.7]; $p{<}0.05$). Both are statistically significant after Holm-Bonferroni correction. However, NGH regresses sharply on AppWorld: $-$14.6 pp (CI [$-$22.8, $-$6.4])."
> 出处：2608.25277 §3.1｜Q24

> 原文："By defaulting to NL on 89% of AppWorld tasks (those involving iteration, conditionals, or free-text interpretation), it recovers full parity (51.7% vs. 51.7%)."
> 出处：2608.25277 §3.1｜Q25

> 原文："Some benchmarks are not natively multi-agent (BFCL, for instance, is function calling), but casting it this way tests whether the graph preserves complex API-sequence structure without harm; the parity we observe (75.4 vs. 75.3) is the expected outcome for a task with no cross-step dependency structure to make explicit."
> 出处：2608.25277 §3 Handoff harness｜Q21

> 原文："The 76% inter-agent misalignment figure derives from an automated error taxonomy (MAST, Multi-Agent Systematic Taxonomy) applied to 345 $\tau$-bench trajectories across three protocols (single-agent, NL multi-agent, graph multi-agent; 115 tasks $\times$ 3 seeds each)."
> 出处：2608.25277 §Appendix C｜Q49

> 原文："Of all multi-agent failures, 76% are inter-agent misalignment: the executor misinterprets ordering, drops prerequisites, or enters retry loops from ambiguous instructions; this share is robust to the taxonomy’s thresholds (Appendix I)."
> 出处：2608.25277 §4 Analysis｜Q34

> 原文："Classification is rule-based from trajectory logs (not human-annotated): a failure is “misalignment” if the trajectory contains (a) a tool call that returns an error due to missing prerequisites, (b) $\geq$3 consecutive retry steps on the same action, or (c) executor actions that contradict the delegation’s stated ordering."
> 出处：2608.25277 §Appendix C｜Q46

### D. 效率与计费口径

> 原文："Weighted across all trials, the routed system achieves 2.1$\times$ average handoff compression (BrowseComp 2.2$\times$, $\tau$-retail 3.2$\times$, BFCL 2.0$\times$, AppWorld 1.04$\times$)."
> 出处：2608.25277 §3.2 Efficiency｜Q26

> 原文："Compression and the 0.15% router overhead are measured over handoff tokens; accounting for the full per-delegation budget (the 155-token router call and the ${\sim}80$-token graph-aware executor prefill), the graph path still totals fewer tokens than NL (461 vs. 730 on $\tau$-retail, 1.6$\times$; Appendix H)."
> 出处：2608.25277 §3.2 Efficiency｜Q27

> 原文："The 2.1$\times$ average compression and the 155-token (0.15%) router overhead reported in the main text are measured over handoff tokens."
> 出处：2608.25277 §Appendix H｜Q47

> 原文："The graph’s dependency edges prevent executor spiraling (15–27 retry steps eliminated on dependency-chain failures)."
> 出处：2608.25277 §3.2 Efficiency｜Q28

> 原文："On AppWorld, the router preserves NL behavior, avoiding the 18% overhead that NGH-only incurs from failed graph executions."
> 出处：2608.25277 §3.2 Efficiency｜Q29

### E. router 必要性、oracle 余量与替代方案

> 原文："AppWorld: $-$14.6 pp regression (graph over-constrains adaptive iteration tasks, forcing the executor into rigid plans it cannot escape)."
> 出处：2608.25277 §3.3 Ablation: Router Necessity｜Q30

> 原文："$\tau$-airline (150 additional trials): NGH-only regresses $-$4.0 pp; the router recovers parity by routing 98% to NL using the same prompt."
> 出处：2608.25277 §3.3｜Q31

> 原文："A simpler non-LLM router that mapped benchmark label to format would reproduce this per-benchmark aggregate, but it would require the benchmark identity our router never sees and could not produce the within-AppWorld 11%/89% split; the LLM router’s value is exactly this label-free generalization from task content."
> 出处：2608.25277 §3.3｜Q32

> 原文："Per-instance accuracy vs. oracle: on AppWorld’s 152 tasks, the router correctly identifies 15/15 aggregate tasks (100% precision) and correctly defaults to NL on 122/137 non-aggregate tasks (89% recall)."
> 出处：2608.25277 §Appendix D｜Q43

> 原文："Oracle headroom decomposition: of the 8.6 pp gap between Routed (51.7%) and Oracle (60.3%), 5.2 pp comes from NGH rescues on tasks the router sends to NL, and 3.4 pp from NL rescues on tasks the router sends to GRAPH."
> 出处：2608.25277 §Appendix D｜Q44

> 原文："On aggregate tasks ($n$=15), NGH outperforms NL by +6.7 pp: edges enforce fetch-before-compute ordering (Table 3). On iterate ($n$=43) and conditional ($n$=11) tasks, NL outperforms by 7–18 pp: rigid edges prevent adaptive backtracking."
> 出处：2608.25277 §4 Analysis｜Q35

> 原文："Complementarity is substantial: NGH rescues 9.9% of NL failures; NL rescues 19.7% of NGH failures. The oracle achieves 60.3% TSR (8.6 pp headroom)."
> 出处：2608.25277 §4 Analysis｜Q36

> 原文："Routed NGH (+12.7 pp, 3.2$\times$) is the only zero-training protocol and achieves the highest TSR in the comparison, outperforming even trained compressors."
> 出处：2608.25277 §3.4 Protocol Comparison｜Q33

> 原文："In the protocol comparison (Table 2), schema-unaware re-encodings that still hand the executor an explicit plan (TF-IDF and Predictive Delta) gain only +4.7 to +5.3 pp, against the typed graph’s +12.7 pp."
> 出处：2608.25277 §4 Isolating the typed-graph contribution｜Q37

### F. 可移植性与论文自承局限

> 原文："Cross-vendor validation (Claude $\times$ Nova Pro) further shows 0% invalid JSON with 3.1–3.6$\times$ compression preserved, confirming the schema is a portable artifact."
> 出处：2608.25277 §4｜Q38

> 原文："The direction of the effect (Routed $\geq$ NL on every family) is preserved across a different vendor and model family, consistent with the schema being a portable artifact rather than a Sonnet-specific behavior."
> 出处：2608.25277 §Appendix G｜Q50

> 原文："This adds accuracy portability to the earlier cross-vendor check (Claude $\times$ Nova Pro: 0% invalid JSON, 3.1–3.6$\times$ compression preserved), which had established only format portability."
> 出处：2608.25277 §3.1 Second orchestrator backbone｜Q51

> 原文："Main results use a single orchestrator backbone (Claude Sonnet 4.5); we additionally confirm accuracy portability on a second orchestrator (GPT-5 mini, Appendix G) and format portability across model families, but broad multi-model replication remains future work."
> 出处：2608.25277 §Limitations｜Q42

> 原文："The router is a single per-task classifier applied blind to benchmark identity, but on these benchmarks its decisions cluster by task type, so realized routing is coarse (100% graph on dependency-chain benchmarks; 89% NL on AppWorld) rather than fine-grained per-instance adaptation."
> 出处：2608.25277 §Limitations｜Q39

> 原文："Even so, the schema may not generalize to domains with fundamentally different coordination patterns (e.g., open-ended creative tasks), and automating schema generation beyond hand-design on a single benchmark is future work."
> 出处：2608.25277 §Limitations｜Q40

> 原文："Finally, the graph-aware executor prompt is a necessary complement to the schema; systems integrating this approach must include interpretation guidance for the receiving agent."
> 出处：2608.25277 §Limitations｜Q41

## 4. 未能核验 / 「论文未报告」而非编数 的清单

写作过程中**主动放弃**或**明确标注为论文未报告**的内容：

1. **venue（`EMNLP 2026` / `top`）** —— 全文零命中，registry 单方声明，未能在存档中找到录用证据（§1.2）。
   卡片 frontmatter 按任务口径填写，但下游引用前应另行取证。
2. **artifact 是否已放出** —— 论文原话是 `will be released upon publication`，本卡不假设它已放出（§1.3）。
3. **任何金额与成本** —— 论文**只给过一个金额**：router 单次分类调用 `$0.0005`（⑥ Q14，且是论文供应商
   的价目）。除此之外**论文未报告**：Bedrock 调用费、人力工时、工程成本、把 token 换算成钱的比率。
   因此 ⑤ 段的 ROI 只给公式与「必须企业自测」的参数，不给收益金额结论。
   （初稿曾写「论文未给任何金额」，属**事实错误**，已在卡片中订正为「只报告过一个金额」。）
4. **母婴 / 跨境电商场景的任何数字** —— 论文完全不涉及该话题。卡片 ② 段的品类、渠道、动作全部是
   业务背景设定（由业务方给定），`8 类节点 / 7 类边` 之外的数字没有一个是来自论文的；
   「3 个 sub-agent」是场景设定，不是论文结论。
5. **中文 / 多语言任务下的 router 表现** —— **论文未报告**。论文的 router prompt 是英文的，
   五个 benchmark 也都是英文；本卡在 ①b 局限第 4 条显式写了「论文未讨论」。
6. **多团队共享同一份 schema 的治理与版本管理** —— **论文未讨论**；卡片的 ①b 局限第 4 条显式标注。
7. **$0.0005 之外的调用成本与延迟换算** —— 论文 Table 5 给了延迟估计（τ-retail 上 NL 一行 12 s、
   routed 一行 5 s，按表注的 `${\sim}$65 output tok/s` 估算），但**表注自己写明 latency 是 estimated
   而非 measured**，且论文未把它换算成任何货币口径。卡片正文**未使用**这两个延迟数字
   （用「估」的数字去支撑 ROI 会误导），需要延迟数据请回查 Table 5 与其表注。
8. **与作者实现的等价性** —— K1 只证明**卡片自带代码**五级全绿（L1–L5），
   **不证明**它与论文未公开的 artifact 一致。卡片 ③ 段与代码 docstring 已两处声明
   「这是机制的确定性替身，不是复现」。
9. **PDF 页码** —— 存档是 arXiv LaTeXML HTML，无分页信息，故 ⑥ 段出处只到章节号。
10. **表格内的单元格数字** —— 全文的 Table 1–Table 5 在 HTML→Markdown 转换后是**逐行拆散**的
    表格行（列头与数据行分离，例如 `| NGH only | 47.3 | 24.7 | 75.4 | 37.1 |`），
    单独引用容易误读。因此本卡只引用 **Table 1 的完整表注**（Q22，是完整句子且逐字命中）
    与**正文数字**，不引用任何表内单元格。这也是 §1.1 里 Table 2 那行只作为旁证、不作为主证据的原因。
11. **Table 4（GPT-5 mini 第二 backbone）的具体数值** —— 同样是拆散的表格行，
    本卡只引用其定性结论（Q50 / Q51：方向一致、可移植），未引用 `65→68` 等单元格。
12. **registry 的 `score: 45.2` / `priority: P0`** —— 评分与优先级口径不在本卡核验范围，未复核。

### 4.1 一处需要下游注意的门禁盲区（不是本卡的缺陷）

`gate_check.py` 的 `collect_evidence()` 只从**卡片同目录**找 `evidence.md`
（`card.parent / "evidence.md"`），而 registry 把证据链指向
`papers/10-MAS/p2s-2026-0018/evidence.md`（另一条路径）。

后果：**本文件里的引文块与「数字 → 出处」表，对这张卡的 G2 完全不起作用**。
本卡之所以 G2 全绿，靠的是卡片自身 ⑥ 段的 51 条逐字引文（`evidence_sources: 51` 全部来自卡片）。
这不是本次要修的东西（改路径约定会动到别人的文件），但**下一次有人以为「写了 evidence.md 就有出处了」
时会踩到**——建议由维护者决定统一两处路径。

## 5. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
C=paper2skills-vault/10-MAS/Skill-Routed-Graph-Handoff.md

# 1) K1 代码可执行（L1–L5）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" --timeout 60

# 2) G2b 引文逐字核验（先用 --selftest 证明核验器本身可信）
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$C"

# 3) K2 三合一（G1 需要 K1 凭证，否则报 G1-NO-EVIDENCE）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" --quiet \
  --json-out /tmp/k1_card.json
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" --k1 /tmp/k1_card.json
```

引文切片脚本（本次使用的就是它，锚点表内嵌）：

```python
import json, re
from pathlib import Path

FT = Path("paper2skills-vault/papers/10-MAS/p2s-2026-0018/fulltext.md")
TEXT = FT.read_text(encoding="utf-8")

def to_regex(anchor: str) -> str:
    # 锚点里的空格容忍 thin space（U+2009）/nbsp —— 论文里数字与单位之间大量使用
    return r"[\s\u2009\u00a0]*".join(re.escape(p) for p in anchor.split(" "))

def slice_quote(start_marker: str, end_marker: str) -> str:
    ms = list(re.finditer(to_regex(start_marker), TEXT))
    assert len(ms) == 1, f"锚点不唯一/未找到: {start_marker!r}"
    i = ms[0].start()
    me = re.search(to_regex(end_marker), TEXT[i:])
    assert me, f"结束锚点未找到: {end_marker!r}"
    out = TEXT[i:i + me.end()]
    assert "\n" not in out and '"' not in out, "引文跨行或含直引号"
    return out
```

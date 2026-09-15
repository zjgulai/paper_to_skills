# PHASE6 P2 · 约束词显著性与过滤器未接线

**日期**：2026-09-14 · **执行**：主控 · **前置**：P1（域名唯一事实源，台账 #99–#104）
**验收面**：61 → **64 条**（新增 L19a / L19b / L19c）

---

## 0. 一句话

上一批把「补 `text-to-sql` 会放进 4 篇顺带提及的论文，区分点是**标题 vs 摘要的显著性**」
登记为一条独立工单。本批把它做成了可测的形式，**三条结论全部与登记的不一样**：
代价是 **9 篇不是 4 篇**、**该假设的两个机械形式都被实测推翻**、
而**过滤器根本没有下游** —— 于是这条工单原本要动的那个数，**不在任何产物里**。

---

## 1. 工单原文与它被测的地方

工单登记在两处（内容一致）：

- `paper2skills-vault/07-资源库/关键词库-v2.md` §3 末尾的「代价已登记、不掩盖」
- `candidate_filter.py` 的 `DOMAIN_CONSTRAINT` 上方编译期注释

原文：补 `text-to-sql` 会同时放进 4 篇「摘要里顺带提到 text-to-SQL 而主题是别的」的论文
（`Non-vacuous Generalization Bounds` / `Memory Reward Inflation` / `AuthentiCity` /
`Knowing When Not to Reuse`）。真命中与它们的区分点是**标题 vs 摘要的显著性**，
而现有约束词规则表达不了这个区分（改它会同时动到已判的 905 篇）⇒ 登记为独立工单。

---

## 2. 逐条读原文：代价是 9 篇不是 4 篇

**范围界定（这决定了后面所有读数能不能用）**：上一批**新补进约束表**的 7 个主题短语 ——
`text-to-sql` / `data analysis agent` / `autonomous data science`（09 域）、
`aspect-based sentiment` / `opinion mining`（07 域）、`marketing mix` / `media mix`（15 域）——
在 3 个域里**只在摘要命中**（题名不含该短语）的 (论文, 域) 对，共 **28 条**。

⭐ 这 28 条是**全体，不是抽样**。这一点很重要：抽样只能给估计，全体普查给的是**判决**。

逐条读 arXiv 摘要原文（上下文由脚本从池子逐字截取、不是我复述）后：

| 判定 | 条数 | 含义 |
|---|---|---|
| **POLLUTION** | **9** | 主题在别处，本域短语只作为**评测设置的一员**或**评测基准名**出现 |
| **BORDERLINE** | 4 | 与 09 域相邻但本域短语不是研究对象；判真命中或污染**都能说通** |
| **GENUINE** | 15 | 本域短语是论文自己的研究对象 |

产物：`paper2skills-research/data/constraint-salience-labels.json`（逐条 id / 域 / 词 /
摘要内位置 / **逐字上下文** / 一句理由 / 置信度）。由 `--check-labels` 守时效。

**登记里那 4 篇只是最容易看见的 4 篇。** 另外 5 篇（`UpgradeBench` / `EvolveNet` / `TTHE` /
`Constrained Decoding for Diffusion LMs` / `Reproducing and Stress-Testing …`）
都是同一形态（列举评测域 / 评测设置），只是当时没人逐条读完。

---

## 3. 假设的两个机械形式，都错 —— 而且错在相反的方向

### 3.1 字面版：「约束词必须命中标题」

**会砍掉 597 个「仅摘要」保留对里的 326 个（54.6%）。**

而登记说真命中「题名即主题」—— 实测真命中里有大量论文题名**不含**该短语：
`Metadata Reconstruction from Values Alone: Recovering Column Semantics`、
`Never the Number: Structural Abstention …`、`Causal Episodic Memory for Feedback-Driven Agent Repair`、
`Finding the Right Tables and Columns`… 这些**全是真命中**（摘要首句或前段即研究对象）。

⇒ 该形式把「题名恰好写了这个短语」当成了「这是它的主题」，**是一个代理指标**，
而代理指标与被代理的事情在真语料上不等价（本仓库 #82 同型）。

### 3.2 深度版：「只在摘要后半出现 ⇒ 顺带」

摘要中点（50%）是一个**不需要拟合**的自然分界：前半陈述问题与贡献，后半陈述实验设置、
评测与结果。用它判：

| 面 | 读数 |
|---|---|
| 28 条标注集 | 抓到 **6/9** 真污染，**漏 3 篇**；同时误标 **1 篇**真命中（`Hierarchical Clustering … Bayesian marketing mix model`，摘要 70% 处） |
| 全池 326 个「仅摘要」保留对 | 标出 **68 条**，其中**只有 6 条是那 9 篇真污染** ⇒ **62 条是别的** |

那 62 条长什么样（示例，逐条见 `--check` 的读数）：`online`@0.94、`user`@0.95、
`product`@0.93、`tabular`@0.78、`inventory`@0.89、`retrieval`@0.95、`table`@0.81 …
**这些词的「尾部出现」不表示跑题** —— 它们是**上下文词**：论文是不是这一域的，
由它**已经命中了本域收割查询**保证；`inventory` 在后半出现，论文照样是供应链论文。

⇒ **该形式既不全也不准，两个方向都错。**

### 3.3 为什么深度版会漏那 3 篇 —— 机械原因，不是手感问题

命中的「位置」取的是**约束词表里所有命中词的最浅位置**。而 09 域的约束表是：

```
[analytics, business, enterprise, database, table, text-to-sql, data analysis agent, autonomous data science]
```

那 3 篇的主题短语 `text-to-sql` 埋在摘要 **75% / 78% / 82%**，眼看要被标出来 ——
**却被同一个表里的泛词 `table` 救了**：

| 论文 | 主题短语位置 | 救它的词 | 该词位置 |
|---|---|---|---|
| `Constrained Decoding for Diffusion Language Models` | `text-to-sql` @78% | `table` | **31%** |
| `TTHE: Test-Time Harness Evolution` | `text-to-sql` @75% | `table` | **6%** |
| `EvolveNet: Collaborative Harness Evolution` | `text-to-sql` @82% | `table` | **5%** |

⭐ **一个泛词就能救活主题短语埋在八成处的论文。**

**这是本批对「约束词表混了两个语义类」这件事给出的机械证明**，也是下一个人真正的起点：
**先把表拆开（定义性主题短语 vs 业务/上下文落点词），再谈显著性**；
在表拆开之前，任何位置规则都会被泛词短路。

> 我试过一条「机械地拆表」的判据：**该词是不是本域收割查询里被引号括起的短语**
> （`abs:"text-to-SQL"`）—— 来源现成，不用新造。**它太宽**：`abs:"table"`、`abs:"agent"`、
> `abs:"llm"`、`abs:"e-commerce"` 也都是引号里的词。**该判据已否决，记录在此免得下一个人重走。**

---

## 4. 交付：把「看不见」变成「看得见、可复核」，判定一字不改

假设被推翻之后，「改成丢弃规则」这条路就没有依据了 —— 按本仓库危险性排序
**3 > 2 > 1 > 0**，拿一个在全池上既漏 3 又误标 62 的判据去丢数据，
就是把真命中陆续变成**静默损失**（「没测到」的一种伪装）。

于是本批交付的是**证据形态**：

`candidate_filter.constraint_evidence(item, domain)` → 五态

| 形态 | 含义 |
|---|---|
| `SELF_EVIDENT` | 命中题名 ⇒ 自证（**必须最先判**，见 §4.1） |
| `TOPICAL_WINDOW` | 只在摘要，最浅命中落在前半（问题与贡献区） |
| `INCIDENTAL_MENTION` | 只在摘要，命中**全部**落在后半（评测与结果区） |
| `META_ONLY` | 只在 `comment` / `journal_ref` 命中（元数据不是论文自述） |
| `NO_CONSTRAINT` | 该域不设约束词 ⇒ 本条对它们**无从判定**（与「没命中」是两件事） |

全池读数（772 个保留对）：`SELF_EVIDENT` **386** · `TOPICAL_WINDOW` **258** ·
`NO_CONSTRAINT` **96** · `INCIDENTAL_MENTION` **68** · `META_ONLY` **1**。

它**逐 (论文, 域) 落进 `--apply` 产物**、计数进 `--check` 基线、名单进读数。
**没有这一栏时，9 篇污染与 19 篇真命中在文件里长得一模一样** —— 那才是本批要修的缺陷。

### 4.1 ⭐ 一条被自己撞出来的判据：题名自证必须先于位置判定

本批的测量脚本首版按约束词表顺序「撞上第一个就 break」。于是：

- `Replacing Training with Memory: Listwise Selection for **Text-to-SQL**`
- `Spider 2.0-AIFunc: Extending Real-World **Text-to-SQL**`
- `How Far Do On-Prem Open LLMs Get on **Text-to-SQL**?`

三篇**题目就是** text-to-SQL 的论文，只因先撞上表里的 `table`（且 `table` 落在摘要后段）
就被标成「顺带提及」。

⇒ 判据里**顺序本身会产生假阳性**。现在 `constraint_evidence()` 把题名自证**前置**，
三条真 id 作为回归用例锁进 `selftest`，变异 **M9**（取消前置）必须打红。

### 4.2 守卫判据：不许把它接进丢弃

- `selftest`：一篇 `INCIDENTAL` 的论文**必须仍被保留**（`judge()` 不读证据形态）；
- **反向控制**：同形态但另犯别的规则的论文**必须照样被丢** —— 证明上面那条不是恒真；
- **变异 M10**：把 `INCIDENTAL` 接进丢弃路径 ⇒ 自检必须打红。

本仓库 #10 的教训是「新增豁免必须同时写清什么情况下不许豁免」；这里反过来：
**新增「只标不判」，必须写清什么情况下不许不判。**

---

## 5. ⭐ 最重的一条：过滤器根本没有下游（台账 #105）

做这条工单时撞出来的，比工单本身重。

**证据（三条独立）**：

1. `关键词库-v2.md` §0 写着「过滤阶段：`arxiv_harvest.py` 的 `apply_negative_filter()` /
   `require_constraint()`」。**这两个函数在全仓库里只出现一次 —— 就是那一行。**
   `arxiv_harvest.py` 的全部函数是 `build_query` / `fetch` / `parse` / `main`，没有任何过滤。
   这句话读起来像在引用现成代码，实际在描述一件从未存在过的事。
2. `candidate_filter.py --apply` 的产物 `arxiv_candidates_filtered.json`：
   **从未生成过**（全历史 `git log -S` 只有创建它的那次提交提到这个名字），仓内也没有。
3. 第 3 步 `rank_candidates.py` 读的是**未过滤**的 `arxiv_candidates.json`；
   下游 `build_registry.py` 读 `recommendations.json`。**过滤环节不在链上。**

**后果（现算）**：那 9 篇污染**确实进了 `recommendations.csv`**（逐 id grep 命中）；
而 P1 上一批修好的「141 篇静默放行」，其影响面也只到**一份报告与一道门禁**。
本文件的负向词与约束词**至今不影响进入打分与短名单的论文集合**。

未接线，但**已可见且不可再被忘记**：新增门禁 **L19c** 逐字读文档里的**机读规格块**，
`wired=false` ⇒ 不许有脚本读过滤产物；改成 `true` 而无人读它 ⇒ 判红。
**任一方向改变都必须同时改文档。**

> 接线本身是独立工单：它会级联到 `recommendations` / `shortlist` / `registry`
> （274 篇会从打分集合里消失），按 #103 的纪律，**一次只动一件事**。

---

## 6. 验收面（61 → 64）

| 门禁 | 内容 |
|---|---|
| **L19a** | `candidate_filter --check-labels` —— 校准集仍描述着当前池子（逐条 id / 域 / 词 / 形态 / 保留态五项） |
| **L19b** | `candidate_filter --mutate` —— 变异 **10/10 打红 · 10/10 红在正确的判据上 · 10/10 已生效于探针** |
| **L19c** | `candidate_filter --check-spec` —— 规格块点名的接线点真实存在 + 接线状态与代码一致（**双向锁**） |

读数是**现算**，不是抄的：

```
candidate_filter --selftest   exit 0   （新增 6 条形态判据 + 2 条守卫 + 3 条真 id 回归）
candidate_filter --mutate     exit 0   10/10 · 10/10 · 10/10
candidate_filter --check      exit 0   判定 1046 · 保留 772（**与基线逐项相等**）
candidate_filter --check-labels exit 0 校准集 28 条仍有效
candidate_filter --check-spec exit 0   规格块 3 条步骤
run_phase6_gates --selftest   exit 0   32/32（含 L19c 的 8 条端到端）
```

**反向控制（本批的核心不变式）**：`judge()` 一字未改 ⇒ 判定 **1046 / 772** 逐项不变。
分级是**读数**不是判据 —— 形态漂了但判定没漂时，`--check` 打 🟡 而**不判红**
（否则一改词表就红成一片，人会去把它关掉），且**不许再说「逐项相等」**（首版两句并存，
自相矛盾，已修）。

---

## 7. 本批自己撞出的仪器缺陷

| # | 缺陷 | 怎么发现的 | 修法 |
|---|---|---|---|
| **#105** | **过滤器无下游**（设计稿点名的两个函数全仓只出现一次） | 追「约束词规则改了会影响谁」时 | 登记 + L19c 双向锁 |
| **#106** | **判据把事故记录读成现行声明**：`--check-spec` 首版按**散文**里的 `` `fn()` `` 抓点名 ⇒ 我在**订正段里引用**的那两个历史错名被判成 2 条假红（与 **#94** 同型） | `--check-spec` 首跑 | 判据改为**只读机读规格块**；端到端用例 ③ 用「散文写着错名、规格块干净」的夹具把边界钉死 |
| **#107** | **J3 被静默跳过**：`wired=` 不是行首 token，首版 `line.startswith("wired=")` 永远取不到 ⇒ `wired` 恒为 None ⇒ **整条接线判据一次都没执行，而屏幕上一切正常**（#99 同族） | 端到端用例 ⑤⑥ 两条都「该红却绿」 | 改为块内正则取 `wired=`，**取不到 ⇒ exit 2**；新增端到端用例 **④b** 专喂没有 `wired=` 的块 |
| **#108** | **邻近启发式当判据**：读者检测首版是「产物名 ±400 字符内有 `read_text`」⇒ 判红了 `run_phase6_gates.py` 里**那段构造假读者的夹具字符串**（与 #79/#80 同族） | `--check-spec` 在真仓库上判红 | 改 **AST 判**：只认真的把该字符串传进 `open`/`read_text`/`load(s)` 的位置 |
| **#109** | **夹具红/绿得不是地方**：L19c 端到端首版夹具只建了 3 个被点名文件中的 1 个 ⇒ 「干净夹具必须 exit 0」因 **J1 找不到文件**而红（#101 的镜像） | `run_phase6_gates --selftest` 18/20 | 夹具默认建齐规格块点名的全部文件 —— **除被测判据之外的一切都必须成立** |
| — | **校准集生成器的两处**：域取 `resolved[0]` 而非「三个目标域里的那个」；判据把「**该主题短语**只在摘要」错写成「所有约束词都不在标题」 | `--check-labels` **首跑即判红 11 条** | 两处都修 —— 这个门禁自己抓住了自己的生成器 |

---

## 8. 仍未做（在册不藏）

1. **接线**：把 `candidate_filter` 接进第 2 环（独立工单；会级联到 recommendations / shortlist / registry）。
2. **拆表**：把约束词表拆成「定义性主题短语」与「业务/上下文落点词」两类 ——
   §3.3 已给出机械证明与一条**被否决的**候选判据。**在拆表之前，任何显著性规则都会被泛词短路。**
3. **`--check` 的形态漂移只打 🟡 不判红**：这是刻意的（防噪声），但它意味着
   **形态可以被静默改掉**。当前靠 `L19a` 的校准集兜一层，未全覆盖。
4. **指令预算余量**：本批收口时 `render_bytes` **64,712** / 65,536 ⇒ 余量约 **824 B**（薄）。⚠️ 报告初稿里写的 64,846 / 690 B 是**中间量**（后续又改过 `CLAUDE.md`）—— 按本仓库纪律，**数字以现跑为准**。
   下一次往 `CLAUDE.md` 加内容前**必须先跑 `check_instruction_budget.py --json`**，
   必要时按先例把长行整段搬到 `reports/`。

---

## 9. 一句话给下一个人

**这条工单原本要改的那个数，不在任何产物里。**
在拆开约束词表、并把过滤器接进流水线之前，讨论「显著性」讨论的是**一份报告**。

---
title: 证据链 — p2s-2026-0013（Skill-Persona-Based-AB-Simulation）
doc_type: evidence
module: 02-A_B实验
paper_id: 2609.01038
registry_id: p2s-2026-0013
related_card: paper2skills-vault/02-A_B实验/Skill-Persona-Based-AB-Simulation.md
fulltext: paper2skills-vault/papers/02-A_B实验/p2s-2026-0013/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2609.01038 *Data-Driven Persona-Conditioned Agents for A/B Test Simulation*

- **全文存档**：`paper2skills-vault/papers/02-A_B实验/p2s-2026-0013/fulltext.md`
  （699 行 / 约 71 KB / 70,815 字符，`evidence_grade: A`）
- **存档里没有本论文自身的 venue 标注** —— 页眉只有标题与四位作者（Amazon，Luxembourg / Barcelona），
  全文 `EMNLP 2026`、`Industry Track` 均 **0 命中**（唯一出现的 `EMNLP '25` 是参考文献 Kolluri et al. 2025）。
  卡片因此**按 registry 口径**记 `venue: EMNLP 2026 Industry Track` / `venue_tier: top`，
  但**该 venue 未能在全文底本中复核**。
- **引文全部为逐字切片**：下列每条引文都是按起止锚点从 `fulltext.md` 直接切出的**连续子串**，
  没有拼接、没有改写、没有跨段缝合；41/41 经 `quote_check.py` 判 `VERBATIM`。

## 0. 核验凭证（可复核的退出码 / stdout）

```
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py \
      --card paper2skills-vault/02-A_B实验/Skill-Persona-Based-AB-Simulation.md \
      --json-out /tmp/p2s0013/k1.json
已索引仓库内本地模块 86 个（可自动解析的卡片将真正执行）
另有 54 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）
✅ PASS         Skill-Persona-Based-AB-Simulation.md#stitched(1块)
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | 🟠 MIGRATED 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <card>
✅ VERBATIM   Skill-Persona-Based-AB-Simulation.md  41/41 逐字
共 1 张卡：1 通过（有引用块且逐字可核），0 含伪造引文，0 无全文可核验，0 无引用块（**不等于通过**）

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
1 真引文     : VERBATIM   连续度=1.0
2 纯伪造     : FABRICATED 连续度=0.124   ← 必须 FABRICATED
3 拼接引文   : FABRICATED 连续度=0.681 召回=1.0 拼接标记=True   ← 必须被拦下
4 排版差异   : VERBATIM   连续度=1.0   ← 必须 VERBATIM
✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接

$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <card> --k1 /tmp/p2s0013/k1.json
G1 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 4
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
（退出码 0）
```

⚠️ **`gate_check.py` 不带 `--k1` 时 G1 必然判红**（`G1-NO-EVIDENCE`：无 K1 验证凭证一律判红，
禁止凭人工判断放行）—— 这是门禁的设计，不是卡片缺陷。本卡按规范先用 `verify_skill_code.py
--json-out` 产出凭证，再把该 JSON 传给 `--k1`。K1 凭证写在 `/tmp`（本次交付只落两个文件），
命令见 §5。

G2 / G3 的明细指标（`gate_check.py` 内部 `GateResult.metrics`，非 stdout，单独记录以免与上面的
stdout 混淆）：

```
G2: metric_numbers=18  sourced=18  unsourced_metric=0  unsourced_general=4  traceability_pct=100.0
    quotes_total=41  quotes_verbatim=41  quotes_fuzzy=0  quotes_fabricated=0  quotes_spliced=0
    quotes_unverifiable=0  quote_verdict=VERBATIM  sourced_structural_only=0  cn_numeral_claims=0
G3: concrete_signals=10  vague_phrases=0  has_data_requirement=True  has_roi_basis=True  skill_relations=6
```

**G2 的 4 条黄灯全部不是论文事实**，逐条列出（便于人工复核，而不是"剩几条黄灯"）：

| # | 数字 | 来源 | 判定 |
|---|------|------|------|
| 1–2 | `12` | frontmatter 的 `created: 2026-09-12` / `updated: 2026-09-12`（v2 模板强制字段） | 非断言，无需出处 |
| 3–4 | `41` | 卡片 frontmatter 的 `verified_by` 里写的「quote_check.py（41/41 VERBATIM）」自指 | 非论文事实，本文件的复核数 |

## 1. registry 逐项核对

registry（`papers_registry.json` → `p2s-2026-0013`）相关字段原文：

```
decision_reason: 人格条件 A/B 仿真：40 个真实 A/B 上方向准确率 0.75-0.90，用于实验**事前粗筛**（母婴流量小、周期长）
note:            EMNLP 2026 Industry Track 已核实；只能当粗筛器，不能替代实验
venue:           EMNLP          venue_tier: top
data_availability: available
outputs.skill_card: paper2skills-vault/02-A_B实验/Skill-<方法名>.md
```

### 1.1 ⚠️ 核心口径澄清：`0.75–0.90` 被压平了，两端来源不同

摘要原文只有一句话给这个区间：「On a benchmark of 40 A/B tests spanning two metric types,
our best configuration achieves 0.75–0.90 directional accuracy **depending on the test metric**」（Q12）。
逐项拆开后，**这个区间的两端不是同一个配置、同一个数据源、同一个指标**：

| 数字 | 真正来源（谁 × 哪类度量） | 出处键 |
|---|---|---|
| **0.75** | pairwise rating 的最优配置（平台自有数据 / 深度池）在 **CTR** 上的 Acc | Q9（§5.1 Takeaway） |
| **0.80** | 同一配置在 **subscription** 上的 Acc | Q9 |
| **0.90** | **公开电商（open e-commerce）人格**在 **subscription** 测试上的 Acc | Q18（§5.2） |
| **0.80** | 被 0.90 超过的那个：**平台自有数据（proprietary）** 在 subscription 上的 Acc | Q18 |
| **0.70–0.90** | 「域内行为数据（in-domain）」的整体区间（两类度量合计） | Q22（§5.2 Takeaway） |
| **0.57–0.69** | 「域外来源（out-of-domain）」的整体区间 | Q22 |
| **0.65 / 0.60** | Rotten Tomatoes（影视/娱乐偏好）人格的 CTR / subscription Acc | Q19 |
| **0.70 / 0.64** | 平台自有数据人格的 CTR Acc / SignOv（**§5.2 正文口径**，见 §1.3 的不一致说明） | Q21 |

**结论**：registry 写的「40 个真实 A/B 上方向准确率 0.75-0.90」在数值上成立，但**丢掉了两个限定**：
① 它是「best configuration」在**两类度量**上的跨度，不是单一指标的准确率；
② 区间上端 0.90 由**公开电商人格 × 订阅测试**取得，**不是平台自有数据**（平台自有数据是 CTR 0.75 /
subscription 0.80），也不是 CTR。正确的业务读法是：
**要拿高分，人格数据必须与业务域对齐；拿不到平台第一方数据时，用同类目公开电商行为数据可以媲美
平台自有数据（0.80 → 0.90，且仅在订阅类测试上实测）**。这一点已写进卡片 ② 场景 1 的"数据可得性"。

### 1.2 registry 其它字段的逐项核对

| 字段 | registry 原写 | 逐项核对结论 |
|---|---|---|
| 「40 个真实 A/B」 | 40 | ✅ 摘要（Q12）、§1（Q13）、§4.1（Q14）三处一致；**但是两类度量各 20 个**，不是宽泛的多任务基准（Q16） |
| 「人格条件 A/B 仿真」 | — | ✅ 与原文措辞一致：persona-conditioned agents + structured question task（Q5） |
| 「用于实验**事前粗筛**」 | — | ✅ 原文 pre-screening / filters clearly inferior treatment candidates（Q4） |
| note「只能当粗筛器，不能替代实验」 | — | ✅ 原文 §6：cannot fully replace human A/B tests—but it does not need to（Q3）。**该边界已写进卡片 ①b 第 1 条** |
| 「（母婴流量小、周期长）」 | — | ⚠️ **这是业务侧类比，论文完全没有母婴 / 跨境电商内容**：全文 `母婴` / `cross-border` / `mother` / `baby` / `infant` / `maternal` **全部 0 命中**。卡片 ② 段的品类与渠道是业务背景，**没有任何一个数字来自论文** |
| `venue: EMNLP` / `venue_tier: top` | EMNLP / top | ⚠️ 两个问题：① 全文底本里**没有本论文自身的 venue 标注**（`EMNLP 2026`、`Industry Track` 0 命中），无法在底本复核，只能按 registry 记；② `venue-whitelist.md` 把 EMNLP 列为 **CCF-B**，与 registry 的 `top` 不一致 —— 卡片按 registry 记 `top`，建议 registry 明确 tier 口径（本卡未改 registry） |
| `data_availability: available` | available | ❌ **建议改为 `partial`**：论文 Limitations 明确写「the benchmark ground-truth labels are not publicly released, limiting external reproducibility」（Q38）。公开可得的是**方法论**与三类公开人格数据源（社科问卷 / 影视评分 / 开放电商交易），**不是那 40 个测试的标签**。与 2608.25871（私有面板 → `partial`）是同一类问题 |
| `outputs.skill_card` | `Skill-<方法名>.md` 占位 | ⚠️ 占位未回填。实际交付文件名 `Skill-Persona-Based-AB-Simulation.md`（本卡只读 registry，未做任何修改） |
| `outputs.code_dir` | `paper2skills-code/ab_testing/<algo>` | ⚠️ 未落地；按仓库现状（CLAUDE.md 已注明 3A/3B/3C 的 `code_dir` 均为规划值），代码模板内嵌在卡片 ③ 段，由 K1 门禁验证 |

### 1.3 论文**内部**数字不一致（影响引用口径，必须记录）

本文正文与附录全表在同一配置上给出**不同数值**，因此卡片**一律引用正文叙述句**，不引用表行：

| 位置 | 数值 | 冲突点 |
|---|---|---|
| §5.1 Takeaway / Table 1 / Table 7 | pairwise rating 的 CTR Acc = **0.75** | 与下一行并存 |
| §5.2 正文（Q21） / §5.5 正文 / Table 11 | 同一配置的 CTR Acc = **0.70**（SignOv 0.64） | 0.70 vs 0.75 并存，论文未解释 |
| §5.3 正文（Q24） | 深度池 vs 代表池 CTR = **0.75 vs. 0.60** | 差值 0.15 |
| 附录 Table 9 | 深度池 vs 代表池 CTR = **0.70 vs. 0.55** | 同一对比、差值同为 0.15，但两个数都低 0.05 |
| Table 4 | All(935)：CTR **0.75** / sub **0.80** | 主表 |
| 附录 Table 10 | All(935)：CTR **0.80** / sub **0.90** | 同一行，附录数值整体更高 |
| §5.3 正文 | 「distributional metrics differ only marginally (**SignBC 0.69 vs. 0.67**)」 | Table 3 该两列是 Acc 与 SignOv（0.68 vs. 0.69），附录 Table 9 的 SignBC 是 0.63 vs. 0.61 —— 三个来源两两不符 |

**卡片采取的口径**：F 段引文全部来自**正文句子**；卡片正文出现的每个度量数字都能在 ⑥ 段的正文引文里
找到（见 §2）。表行数字**一律不引用**，原因是 HTML→Markdown 转换后表行被压成
`0.75.10`、`0.40.11`、`0.90.07` 这种「均值与 ±SE 粘连」的脏串（例：`| Pairwise rating | 0.75.10 | 0.68.09 |
0.80.09 | 0.72.07 |`），逐字引用会误导读者。

## 2. 卡片正文数字 → 出处对照表

| 卡片中的数字 | 含义 | 引文键 |
|---|---|---|
| `40`（+`20`/类） | benchmark 测试数 / 每类度量的测试数 | Q12 Q13 Q14 Q35 |
| `0.75–0.90` | **摘要口径**的最优配置方向准确率区间（两端来源不同，见 §1.1） | Q12 |
| `0.90` vs. `0.80` | 公开电商人格 vs 平台自有数据（subscription 测试） | Q18 |
| `0.70–0.90` / `0.57–0.69` | 域内 vs 域外数据源的整体区间 | Q22 |
| `0.65` / `0.60` | Rotten Tomatoes 人格的 CTR / subscription Acc | Q19 |
| `0.70` / `0.64` | 平台自有数据人格的 CTR Acc / SignOv（§5.2 正文口径） | Q21 |
| `0.75` vs. `0.60` | 深度池 vs 代表池 CTR Acc | Q24 |
| `0.80` / `0.80` | 两池 subscription Acc（无显著差异） | Q24 Q27 |
| `0.30` vs. `0.80` | demographics-only 的订阅 Acc 退化 | Q30 |
| `0.40` vs. `0.45`、`0.60` vs. `0.65` | 单一通用人格 vs 完全无人格 | Q31 |
| `935` | 平台侧两个人格池各含人格数 | Q17 |
| `500` / `1pp` / `2×` | 子采样规模 / 与全池的差距 / 成本下降 | Q28 |
| `100` | 极端预算下仍可用的人格数 | Q29 |
| `20`（笔交易） | 行为稀疏的可操作门槛 | Q23 |
| `1–10` | 评分量纲 | Q41 |
| `0.5` | Acc 定义里的 Φ 阈值 | Q11 |
| `4.5` | Claude Sonnet 4.5（单一 LLM 局限） | Q39 |
| `2009` | 引文中的 Kohavi et al. 2009（A/B 成本论证） | Q1 |
| `12` ×2、`41` ×2 | frontmatter 日期 / 「41/41 VERBATIM」自指 —— **非论文事实**（G2 黄灯，见 §0） | — |

**卡片正文中不存在"论文事实数字"与"本卡模拟输出数字"并列**：③ 段代码打印出的准确率（合成世界，
规则替身）全部在代码围栏内，且紧跟一段显式声明「与论文数值同量级纯属巧合、二者不可互相印证」。
卡片正文里没有任何来自本卡模拟输出的百分比。

**业务侧参数**：卡片 ② 段的品类、渠道、周期（「跑满一个 A/B 通常要按周计」）均为**业务背景经验值**，
刻意未写成数字，并在文中标注「业务侧经验值，论文未报告任何周期数字」。

## 3. 逐字引文全表（41 条，与卡片 ⑥ 段一一对应）

本段每条引文均从 `fulltext.md` 按起止锚点**逐字切出**（连续子串，无拼接、无改写）；
全部经 `quote_check.py` 判定 `VERBATIM`。

### A. 为什么需要它：A/B 的流量与周期成本，以及「只做粗筛」的定位

> 原文："Online controlled experiments remain the gold standard for validating product changes, yet each test requires sufficient user traffic, engineering effort, and typically weeks of data collection to reach statistical significance (Kohavi et al., 2009). These costs limit how many ideas teams can evaluate."
> 出处：2609.01038 §1 Introduction｜Q1

> 原文："A particularly compelling application is the simulation of online controlled experiments (A/B tests): if persona-conditioned agents can reliably predict whether users prefer a treatment variant over a control, teams could pre-screen design candidates offline—reducing the time, traffic, and experimentation cost (Rieder et al., 2026; Castelo et al., 2026)."
> 出处：2609.01038 §1 Introduction｜Q2

> 原文："With current accuracy levels, the proposed framework cannot fully replace human A/B tests—but it does not need to."
> 出处：2609.01038 §6 Discussion · Potential applications｜Q3

> 原文："A potential application could be a pre-screening tool that filters clearly inferior treatment candidates before they consume traffic and prioritizes the experiments by ranking proposed changes by predicted impact."
> 出处：2609.01038 §6 Discussion · Potential applications｜Q4

### B. 方法：结构化提问 + 配对评分 + 方向聚合

> 原文："We frame A/B test simulation as a structured question task: each persona-conditioned agent is presented with variant screenshots and asked to evaluate them with respect to a target metric."
> 出处：2609.01038 §B.1 Question Design: Designs Description｜Q5

> 原文："We study four formats varying along two axes—isolation vs. comparison (whether the agent sees one variant or both) and binary vs. rating (whether the response is yes/no or a 1–10 score)."
> 出处：2609.01038 §3.3 Question Design｜Q41

> 原文："In independent formats, each variant is shown separately; in pairwise formats, both variants are presented together with order randomized per persona to control positional bias."
> 出处：2609.01038 §3.3 Question Design｜Q6

> 原文："Independent formats perform poorly, suggesting agents struggle to calibrate scores without comparative context."
> 出处：2609.01038 §5.1 Question Design Comparison｜Q7

> 原文："The binary pairwise format achieves strong results on subscription tests but fails on CTR, indicating that optimal design depends on metric type."
> 出处：2609.01038 §5.1 Question Design Comparison｜Q8

> 原文："Pairwise rating is the most effective question format, achieving 0.75 accuracy on CTR and 0.80 on subscription tests. All subsequent experiments use this format."
> 出处：2609.01038 §5.1 Question Design Comparison · Takeaway｜Q9

> 原文："For each A/B test, we collect per-persona $(s_{ref},s_{treat})$ tuples and compute the predicted effect as $\hat{\delta}_{s}=\frac{1}{N}\sum_{i=1}^{N}\frac{s_{treat}^{(i)}-s_{ref}^{(i)}}{s_{ref}^{(i)}}$."
> 出处：2609.01038 §I.5 Score Extraction｜Q10

> 原文："Accuracy (Acc): $\mathbb{I}[(p{-}0.5)(q{-}0.5)>0]$; Sign overlap (SignOv): $1-|p-q|$; Sign Bhattacharyya (SignBC): $(\sqrt{pq}+\sqrt{(1{-}p)(1{-}q)})^{2}$."
> 出处：2609.01038 §3.6 Evaluation Metrics｜Q11

### C. 基准：40 个测试、两类度量

> 原文："On a benchmark of 40 A/B tests spanning two metric types, our best configuration achieves 0.75–0.90 directional accuracy depending on the test metric, demonstrating that data-driven personas are a viable path toward fast, low-cost experiment pre-screening."
> 出处：2609.01038 §Abstract｜Q12

> 原文："We evaluate our framework on a benchmark of 40 A/B tests spanning two metric types—click-through rate (CTR) and subscriptions—and organize experiments around four research questions:"
> 出处：2609.01038 §1 Introduction｜Q13

> 原文："The original candidate set contained over 50 CTR tests and 40 subscription tests; applying these thresholds excluded tests with ambiguous ground truth, yielding the final benchmark of 40 tests (20 per metric)."
> 出处：2609.01038 §4.1 Benchmark Construction｜Q14

> 原文："The benchmark reflects a curated experimental sample and should not be interpreted as representative of any specific platform’s full user base or operational A/B testing infrastructure."
> 出处：2609.01038 §4.1 Benchmark Construction｜Q15

> 原文："The benchmark spans two metric types: click-through rate (engagement) and subscriptions (sign-up intent), evaluated with the same pipeline but different question framing."
> 出处：2609.01038 §4.1 Benchmark Construction｜Q16

### D. 人格数据源与域对齐（registry 的 0.75–0.90 分项来源）

> 原文："We compare two persona pools (both containing 935 personas)."
> 出处：2609.01038 §4.2 Personas Pool Construction｜Q17

> 原文："Among the external persona sources, open e-commerce data performs best and surpasses platform data on subscription tests (0.90 vs. 0.80 accuracy), likely due to domain alignment—e-commerce browsing and purchasing signals are directly relevant to evaluating widget engagement and subscription intent."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q18

> 原文："Rotten Tomatoes personas, grounded in entertainment preferences, show reasonable performance on CTR tests (0.65) but degrade on subscription tests (0.60), suggesting that out-of-domain behavioral data provides insufficient signal for metric-specific predictions."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q19

> 原文："Survey-based personas perform moderately without excelling on either metric."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q20

> 原文："Personas constructed from platform behavioral data achieve competitive results on CTR tests (0.70 accuracy, 0.64 SignOv) and competitive performance on subscription tests."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas｜Q21

> 原文："Domain alignment is crucial for personas effectiveness. In-domain behavioral data achieves 0.70–0.90 accuracy, while out-of-domain sources drop to 0.57–0.69. Public e-commerce data can rival platform-specific personas."
> 出处：2609.01038 §5.2 Synthetic vs Data-Driven Personas · Takeaway｜Q22

> 原文："Domain alignment matters more than data volume or source exclusivity. Public e-commerce data rivals platform-specific personas (Table 2), lowering adoption barriers. However, below ${\sim}20$ recorded transactions, simulation degrades as the LLM defaults to generic reasoning."
> 出处：2609.01038 §6 Discussion · Data requirements｜Q23

### E. 行为深度 vs 群体多样性、子采样、消融

> 原文："The deep pool significantly outperforms the representative pool on CTR accuracy (0.75 vs. 0.60), while on subscription tests there is no statistically significant difference on any metric (0.80 accuracy for both)."
> 出处：2609.01038 §5.3 Persona Pool Comparison｜Q24

> 原文："We hypothesize that for sparse personas, the LLM lacks sufficient behavioral grounding and defaults to generic reasoning rather than user-specific preferences, explaining the CTR accuracy gap."
> 出处：2609.01038 §5.3 Persona Pool Comparison｜Q25

> 原文："These gaps likely arise from LLM inference biases during persona generation rather than sampling limitations, and represent a primary lever for future improvement."
> 出处：2609.01038 §5.3 Persona Pool Comparison｜Q26

> 原文："Behavioral depth yields a statistically significant advantage on CTR accuracy, but demographic diversity fully compensates on subscription tests (no significant difference on any metric)."
> 出处：2609.01038 §5.3 Persona Pool Comparison · Takeaway｜Q27

> 原文："All subsampling strategies preserve near-full-pool accuracy at 500 personas (within 1pp on CTR, matching or exceeding on subscriptions), potentially enabling up to 2$\times$ cost reduction."
> 出处：2609.01038 §5.4 Population Sampling Efficiency · Takeaway｜Q28

> 原文："Even at $n{=}100$, all algorithms remain competitive (0.77–0.80 CTR, 0.82–0.86 subscriptions), making routine simulation viable under constrained budgets."
> 出处：2609.01038 §6 Discussion · Cost–quality trade-off｜Q29

> 原文："Demographics-only degrade substantially on subscriptions (0.30 vs. 0.80), indicating that behavioral profiles are critical for higher-salience decisions."
> 出处：2609.01038 §5.5 Ablation: Component Contributions｜Q30

> 原文："A single generic persona does not outperform the unconditioned LLM (0.40 vs. 0.45 on CTR, 0.60 vs. 0.65 on subscriptions), confirming that population diversity—not mere persona framing—drives the improvement."
> 出处：2609.01038 §5.5 Ablation: Component Contributions｜Q31

### F. 边界与论文自承局限

> 原文："With batch inference, a full simulation could complete in hours at a fraction of a multi-week experiment cost, potentially enabling teams to explore a broader design space without proportionally increasing experimentation overhead."
> 出处：2609.01038 §6 Discussion · Potential applications｜Q32

> 原文："Our experiments show that simulations are most reliable when the underlying effect size is large, as is typical of high-salience decisions. Conversely, the framework is least trustworthy for near-zero effects where small perturbations flip the predicted direction. This suggests that outputs could be more useful as a ranking signal rather than a binary decision criterion."
> 出处：2609.01038 §6 Discussion · When to trust simulation｜Q33

> 原文："Our framework evaluates agents on isolated screenshots rather than full page contexts, removing situational factors (browsing intent, session history, surrounding content) that influence real user decisions."
> 出处：2609.01038 §Limitations｜Q34

> 原文："The benchmark is limited to 40 tests from a single e-commerce domain and two metric types; generalization to other domains, metric types, or multi-step user journeys remains untested."
> 出处：2609.01038 §Limitations｜Q35

> 原文："Persona demographics are inferred by the LLM from behavioral signals rather than self-reported, introducing systematic biases that may distort population-level predictions."
> 出处：2609.01038 §Limitations｜Q36

> 原文："LLM positivity bias and anchoring effects likely produce systematically optimistic treatment evaluations, a tendency that our sign-based metrics partially mask when both control and treatment are equally inflated."
> 出处：2609.01038 §Limitations｜Q37

> 原文："Finally, the benchmark ground-truth labels are not publicly released, limiting external reproducibility; however, the methodology is fully reproducible with any preference dataset, as we demonstrate with public data sources that produce competitive results."
> 出处：2609.01038 §Limitations｜Q38

> 原文："All experiments use a single LLM (Claude Sonnet 4.5); while we validate consistency across Claude Haiku and Opus (Appendix D), generalization to non-Anthropic models remains untested."
> 出处：2609.01038 §Limitations｜Q39

> 原文："These results suggest that data-driven persona simulation could serve as a viable tool for pre-screening A/B test candidates, potentially reducing wasted experimentation traffic while maintaining directional accuracy that may be sufficient for prioritization decisions."
> 出处：2609.01038 §7 Conclusions｜Q40

## 4. 「论文未报告」/ 未能核验清单

本卡写作中**主动放弃**了以下内容，原因是论文没给、或给了但无法逐字可靠引用：

1. **母婴 / 跨境电商 / 非英语市场的一切数字** —— 全文 `母婴`、`cross-border`、`mother`、`baby`、
   `infant`、`maternal` **全部 0 命中**。卡片 ② 段的场景是业务侧构造，**没有任何数字来自论文**；
   ⑤ 段 ROI 公式里的参数是符号而非数值。
2. **任何成本金额** —— **论文未报告**。全文 `dollar` / `USD` 0 命中；唯一的成本陈述是定性的：
   「a full simulation could complete in hours at a fraction of a multi-week experiment cost」（Q32）。
   token 消耗、人力投入、许可证费用均未报告。
3. **业务收益（GMV / ROAS / 转化率 / 留存）** —— **论文未报告**。论文只报方向类指标
   （Acc / SignOv / SignBC），并在附录 C.1 明确论证**不采用**幅度型指标，因此连「效应量预测误差」
   这类数字也没有。
4. **省下多少流量 / 多少时间** —— **论文未报告**具体数字，只有 Q32 的定性表述与 Q28 的
   仿真自身成本下降（up to 2×）。
5. **那 40 个测试的 ground-truth 标签** —— 论文明确「not publicly released」（Q38），
   所以**本卡的 benchmark 无法在权威口径下复现**；卡片的校准建议是「企业自建历史 A/B 对账集」。
6. **survey（社科问卷）数据源的分项数字（正文无）** —— §5.2 正文对 survey 只给定性结论
   「perform moderately without excelling on either metric」（Q20）；分项数字（CTR / subscription）
   仅出现在 Table 2 / Table 8 的**行内**，且存档中该行被压成
   `| Survey data | 0.60.11 | 0.58.10 | 0.75.10 | 0.68.07 |` 这类粘连串。
   → 按「不把脏串当引文」的约定，卡片**不引用 survey 的分项数字**，只在 ⑥ 段保留 Q20 的定性句。
   （这也是本次交付中唯一一处"用户要求覆盖但无法逐字引用"的数字：**survey 的 0.60 / 0.75 无正文句子可引**。）
7. **三类公开人格池的规模（每类 1000 人格）** —— 正文有（§4.2），但卡片未使用该数字，故未引用。
8. **表 1–5 / 7–11 的所有行内数字** —— HTML→Markdown 后「均值±SE」粘连（`0.75.10`），
   逐字引用会误导；卡片只引正文叙述句（理由见 §1.3）。
9. **完整 prompt 原文** —— 附录 I 开头自陈模板是「slightly simplified for clarity」的版本，
   实验最终版未公开；卡片 ③ 段的提问模板是**按 §I.4 的结构重写**的，不是逐字复制。
10. **代码 / 数据是否开源** —— 论文未给出仓库链接，全文 `github` 0 命中；**未能核验**（既不能
    说"有"，也不能说"无"）。
11. **平台身份** —— §4.1 只写 "marketing A/B tests from e-commerce platform"，未指明平台；
    论文自己也声明「不应被解读为该平台完整用户群或运营 A/B 体系的代表」（Q15）。
12. **与作者实现的等价性** —— K1 只证明**卡片自带代码**可执行（L4/L5 全绿），**不证明**它与论文
    未公开的实现一致；③ 段的 agent 是**确定性规则替身**，不是 Claude Sonnet 4.5。
13. **venue** —— 存档无本论文 venue 标注（见文件头说明），按 registry 口径记录，**未能在底本复核**。

## 5. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
CARD=paper2skills-vault/02-A_B实验/Skill-Persona-Based-AB-Simulation.md

# K1：代码可执行 + 产出凭证（凭证写到 /tmp，不污染仓库）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$CARD" \
    --json-out /tmp/p2s0013/k1.json

# G2b：引文逐字核验（先自证核验器可信）
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$CARD"

# 三合一门禁（G1 必须带 --k1，否则按设计判红）
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$CARD" --k1 /tmp/p2s0013/k1.json
```

---
title: 证据链 — p2s-2026-0011（Skill-Agentic-Catalog-Enrichment）
doc_type: evidence
module: 00-电商Agent
paper_id: 2608.20844
registry_id: p2s-2026-0011
related_card: paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md
fulltext: paper2skills-vault/papers/00-电商Agent/p2s-2026-0011/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2608.20844 *TRACE: Agentic Catalog Enrichment with Multi-source Evidence Grounding*

- **全文存档**：`paper2skills-vault/papers/00-电商Agent/p2s-2026-0011/fulltext.md`
  （321 行 / 35,572 字符，`evidence_grade: A`）
- **venue**：存档正文**无任何录用标注**（无页眉、无会议名、无 camera-ready 标记），
  按 registry 口径保持 `venue: arXiv preprint` / `venue_tier: preprint`。
- **引文全部为逐字切片**：下列每条 `> 原文："…"` 都是从 `fulltext.md` 按「行号 + 起止锚点」
  **程序化切出的连续子串**（见 §6 复现方式），没有拼接、没有改写、没有跨段缝合。
  切出后另做了一次「必须是 fulltext 的逐字子串」断言，再交给 `quote_check.py` 复核。

## 0. 核验凭证（真实 stdout）

```
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py \
      --card paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md
已索引仓库内本地模块 86 个（可自动解析的卡片将真正执行）
另有 54 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）
✅ PASS         Skill-Agentic-Catalog-Enrichment.md#stitched(2块)

==================================================================
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | 🟠 MIGRATED 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）
==================================================================
```

K1 五个阶段逐项：`L1_SYNTAX True` / `L2_COMPILE True` / `L3_IMPORT True` / `L4_SMOKE True` / `L5_TEST True`
（L5 pytest：`9 passed`）。

```
$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py \
      --card paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md
✅ VERBATIM   Skill-Agentic-Catalog-Enrichment.md  45/45 逐字

共 1 张卡：1 通过，0 含伪造引文，0 无全文可核验
```

```
$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py \
      --card paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md \
      --k1 <K1 报告 JSON>

======================================================================
G1 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================

======================================================================
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 2
======================================================================

======================================================================
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================
exit=0
```

- **G2 的 2 条黄灯**全部来自 v2 模板强制的 frontmatter 字段：`created: 2026-09-12` 与
  `updated: 2026-09-12` 里的 `12` 被当作「一般数字无出处」。不属于事实断言。
- G2 指标：`metric_numbers 32 / sourced 32 / unsourced_metric 0 / traceability_pct 100.0 /
  quotes_verbatim 45 / quotes_fuzzy 0 / quotes_fabricated 0 / quotes_spliced 0 /
  sourced_structural_only 0 / cn_numeral_claims 0 / fulltext_archived true`。
- **不传 `--k1` 时 G1 判红（`G1-NO-EVIDENCE`）** —— 这是门禁的设计口径：
  「无 K1 凭证一律判红，不得凭人工判断放行」。本卡的 K1 凭证由上面第一条命令现场产生。

## 1. registry 逐项核对表

registry 记录（`papers_registry.json` → `p2s-2026-0011`）**note 明确要求**：
「无 venue；数字来自完整摘要，萃取前须回原文核对」。**下表就是这次核对的结论。**

| registry 字段 | registry 原值 | 回原文核对结论 |
|---|---|---|
| `identifiers.arxiv` | `2608.20844` | ✅ 与存档头部 `arxiv_id` 一致 |
| `title` | TRACE: Agentic Catalog Enrichment with Multi-source Evidence Grounding | ✅ 与存档标题逐字一致 |
| `published` | `2026-08-21` | ⚠️ **无法核验**：全文存档里没有任何日期。按 registry 记录保留，本卡不引用该日期 |
| `journal` / `venue` | `""` / `""` | ✅ 保持为空；正文无录用证据 → 记 `arXiv preprint` / `preprint` |
| `venue_tier` | `preprint` | ✅ 无录用证据，不得抬级 |
| `domain` | `00-电商Agent` | ✅ 本轮新建域，本卡为该域第一张 |
| `decision_reason` | 「TRACE：Scout/Judge 双 agent + 多源证据接地做目录属性补全，生产四个垂类曝光加权覆盖率 +90.4%，结账转化 +0.48%」 | ⚠️ **部分修正，见 §2**：Scout/Judge 双 agent ✅、多源证据接地 ✅、目录属性补全 ✅、四个垂类 ✅、结账转化 `+0.48%` ✅；**但 `+90.4%` 只出现在摘要**，正文写的是 `over 90%` 且没有分垂类精确表 |
| `data_availability` | `available` | ❌ **建议修正为 `partial`**：论文的数据来自「某电商平台生产目录的采样」，**未公开数据集**；离线评测集（Grocery/Alcohol 的 500 SKU、Electronics/Home Improvement 的 955 SKU）都没有发布链接或获取途径，也无法从论文复现证据源 |
| `note` | 「无 venue；数字来自完整摘要，萃取前须回原文核对」 | ✅ 本次已完成该核对，结果见 §2 |
| `priority` / `decision` / `score` / `source_route` / `gates` / `status` | `P0` / `extract` / `25.2` / `arxiv` / 三项 `pending` / `shortlisted` | — 流程字段，不在事实核对范围 |
| `outputs.skill_card` | `paper2skills-vault/00-电商Agent/Skill-<方法名>.md`（占位） | ✅ 实际交付 `paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md` |
| `outputs.evidence` | `paper2skills-vault/papers/00-电商Agent/p2s-2026-0011/evidence.md` | ✅ 即本文件 |
| `outputs.code_dir` | `paper2skills-code/ecommerce_agent/<algo>` | ⚠️ **未落地**（规划值）：代码模板内嵌在本卡 ③ 段，由 K1 门禁验证；`paper2skills-code/ecommerce_agent/` 目录不存在 |

**registry 需要下游注意的两处**：`decision_reason` 里的 `+90.4%`（口径见 §2）与
`data_availability: available`（建议改 `partial`）。按项目惯例，
**这类 registry 与原文不符比卡片错误更危险** —— registry 是「论文唯一事实来源」，下游全部依赖它。

## 2. ⚠️ 摘要与正文的口径不一致（本轮核对的最重要发现）

| 指标 | 摘要怎么写 | 正文怎么写 | 判定 |
|---|---|---|---|
| 曝光加权补全覆盖率的跨垂类提升 | `90.4%`（Q4） | `over 90%`，且**没有给分垂类的精确表**（Q35） | ⚠️ **不一致**：`90.4%` **只在摘要出现**。卡片一律写成「摘要报 90.4%，正文表述为 over 90%」，**不把它包装成正文表格里的精确值** |
| 人工评测精度与覆盖 | `98.2%` accuracy at `74.7%` coverage（Q3） | §4.3 给出同一对数字（Q26）；§4.1 交代该口径只覆盖 Grocery 与 Alcohol（Q23） | ✅ 一致，但**适用范围限于一类目对** |
| 结账转化提升 | `+0.48%`（Q5） | Table 2 与 §5.1 给出同一数字，并附 95% 置信区间与 p 值（Q37 / Q40） | ✅ 一致，正文信息更全（摘要未给区间与显著性） |
| 生产补全规模 | 摘要未提 | `31 million SKUs`（Q34） | 只在正文出现，可按正文引用 |

## 3. 卡片正文数字 → 出处对照表

| 卡片中的数字 | 含义 | § 位置 | 引文键 |
|---|---|---|---|
| `90.4%` | 曝光加权补全覆盖率的跨垂类提升（**摘要口径**） | §Abstract | Q4 |
| `over 90%` | 同一指标的**正文口径**，无精确值 | §5 Deployment | Q35 |
| `98.2%` / `74.7%` | 人工验证的抽取精度 / 属性覆盖率（Grocery + Alcohol） | §Abstract / §4.3 | Q3 / Q26 |
| `+0.48%` | 结账转化的相对变化（线上随机实验） | §Abstract / Table 2 / §5.1 | Q5 / Q37 / Q40 |
| `+0.04%, +0.92%` / `0.034` | 上述提升的 95% 置信区间 / p 值 | Table 2 | Q37 |
| `+1.18%` / `+0.24%, +2.12%` / `0.014` | 重度用户子群的相对变化 / 区间 / p 值 | Table 2 | Q38 |
| `−1.08%` / `−2.04%, −0.13%` / `0.026` | 缺件·错件率的相对变化 / 区间 / p 值 | Table 2 | Q39 |
| `95%`（置信区间） | 线上实验的区间口径 | Table 2 表注 | Q45 |
| `90%` / `10%` | 线上实验的处理组 / 留出组流量配比 | §5.1 | Q36 |
| `31 million`（SKUs） | 生产部署补全的 SKU 规模 | §5 Deployment | Q34 |
| `11` → `409` | 目标属性数：Grocery 最少 → Home Improvement 最多 | §4.1 | Q22 |
| `500` / `2,497` | Grocery + Alcohol 评测集规模（人工参考值 + 独立审计员复核） | §4.1 | Q23 |
| `955` / `4,990` | Electronics + Home Improvement 评测集规模 | §4.1 | Q24 |
| `87.8%` / `97.4%` | 该数据集的属性覆盖率 / 判官支持率 | §4.3 | Q29 |
| `98.4%` / `87.8%` / `12.2%` | 判为 PASS 的值人工确认正确的比例 / 分歧中的误拒 / 误收 | §4.3 | Q28 |
| `0.5`（个百分点） | 换 backbone 后抽取覆盖率的提升 | §4.4 | Q31 |
| `85.5%` → `81.9%` | 发布覆盖率的下降（与上一条同一次替换） | §4.4 | Q31 |
| `> 7`（倍） | 同一次替换带来的推理成本上升 | §4.4 | Q31 |
| `73.2%` / `63.5%` | 另两个 backbone 的发布覆盖率 | §4.4 | Q32 |
| `judge-supported rate` 定义 | PASS 或 UNVERIFIED 占全部已抽取值的比例 | §4.2 / Table 1 表注 | Q25 / Q30 |
| 一次侦察调用 + 一次裁决调用 | 每个合格 SKU 的调用量（成本估算的唯一依据） | 附录 A | Q44 |

对照表之外的数字只有三类：v2 模板强制的 frontmatter 字段（`module: 00-电商Agent`、
`created`/`updated` 日期）、交叉引用键（`Q1`…`Q45`，前缀是字母，不构成数字断言）、
以及 ③ 段代码块内的**合成示例数据**（代码围栏在 G2 中被剥离，不属于事实断言）。

## 4. 逐字引文全表（45 条，与卡片 ⑥ 段一一对应）

### A. 摘要（含 §2 标注的摘要/正文不一致项）

> 原文："On an offline human evaluation dataset, TRACE’s proposed attribute values were 98.2% accurate at 74.7% attribute coverage."
> 出处：2608.20844 §Abstract｜Q3

> 原文："Deployed in production on an industry-scale catalog, TRACE increased impression-weighted enrichment coverage across four business verticals by 90.4%."
> 出处：2608.20844 §Abstract｜Q4

> 原文："An online experiment subsequently showed that surfacing the enriched attributes on the product detail page increased checkout conversion by 0.48%."
> 出处：2608.20844 §Abstract｜Q5

### B. 问题与定位

> 原文："Product catalogs underpin search, discovery, and recommendation in e-commerce, yet they are often attribute-sparse: the attributes shoppers and downstream systems rely on are either buried in unstructured content such as titles and images or missing from the catalog altogether."
> 出处：2608.20844 §1 Introduction｜Q1

> 原文："Manually enriching e-commerce catalogs is impractical given their scale and rapid growth."
> 出处：2608.20844 §Abstract｜Q2

> 原文："Some attribute values cannot be reliably inferred from owned data sources and must instead be sourced externally and verified against the exact product."
> 出处：2608.20844 §1 Introduction｜Q6

> 原文："Accuracy estimated from a point-in-time catalog audit may become less representative as the catalog’s product mix evolves."
> 出处：2608.20844 §1 Introduction｜Q7

> 原文："Missing or inaccurate attribute values can mislead shoppers and degrade fulfillment quality; for safety-sensitive attributes such as allergens or dietary restrictions, they can have especially serious consequences."
> 出处：2608.20844 §1 Introduction｜Q8

> 原文："We incorporate identity-matched search grounding to recover attribute values that cannot be reliably inferred from owned data sources."
> 出处：2608.20844 §1 Introduction｜Q9

> 原文："We place a JudgeAgent in the serving path as a verify-before-write gate, applying a consistent evidence standard to each proposed value and making publication quality less sensitive to shifts in the catalog’s product mix."
> 出处：2608.20844 §1 Introduction｜Q10

### C. 方法：ScoutAgent / JudgeAgent / 写库门

> 原文："TRACE implements this process as a two-stage verify-before-write architecture. The ScoutAgent gathers evidence from multiple sources, verifies that externally retrieved evidence refers to the target product, and proposes grounded attribute values. The JudgeAgent then re-examines each candidate under a stricter verification policy and determines whether it is eligible for publication, should be blocked, or requires human review. Figure 1 shows the end-to-end workflow, and Algorithm 1 formalizes the procedure."
> 出处：2608.20844 §3.1 Overview｜Q12

> 原文："where $v_{a}$ is the proposed value, $E_{a}$ is its supporting evidence, $\tau_{a}$ records the evidence-source types, $q_{a}$ is a model-reported confidence score, and $z_{a}$ records the extraction status as one of extracted, not_found, not_applicable, ambiguous, or conflict."
> 出处：2608.20844 §3.1 Overview｜Q11

> 原文："The ScoutAgent gathers and reconciles evidence for each target attribute in two stages. It first considers readily available, product-linked information, including textual fields and product images from seller-provided catalog data and syndicated product data. When this information is insufficient to determine a reliable value for the target attribute, the ScoutAgent uses web search to gather additional evidence."
> 出处：2608.20844 §3.2 The ScoutAgent｜Q13

> 原文："This separation allows the ScoutAgent to prioritize textual evidence while still using images for attributes expressed visually, such as material, certification marks, and on-package claims."
> 出处：2608.20844 §3.2 The ScoutAgent｜Q14

> 原文："Web search may return pages that appear relevant to the query but refer to a different product or product variant."
> 出处：2608.20844 §3.2 Identity-grounded web retrieval｜Q15

> 原文："Evidence from $p$ is used only when the ScoutAgent determines that the page describes the target product; otherwise, the page is discarded."
> 出处：2608.20844 §3.2 Identity-grounded web retrieval｜Q16

> 原文："It maps benign variations to a common representation, such as “NiMH” and “nickel-metal hydride,” or “60 Hz” and “60Hz.”"
> 出处：2608.20844 §3.2 Evidence reconciliation and abstention｜Q17

> 原文："When the available evidence is insufficient, ambiguous, or conflicting, the ScoutAgent abstains rather than inferring a value from background knowledge."
> 出处：2608.20844 §3.2 Evidence reconciliation and abstention｜Q18

> 原文："It applies a stricter evidence policy than ScoutAgent, focusing on whether available evidence supports the proposed value for the target product."
> 出处：2608.20844 §3.3 The JudgeAgent｜Q19

> 原文："The distinction between UNVERIFIED and UNCERTAIN separates a lack of confirming evidence from active disagreement among the available evidence."
> 出处：2608.20844 §3.3 Verdict taxonomy｜Q20

> 原文："Candidates below the model-reported confidence threshold $\theta$ are blocked. Among the remaining candidates, those receiving PASS or UNVERIFIED are written, those receiving FAIL are blocked, and those receiving UNCERTAIN are routed to human review together with their evidence trail."
> 出处：2608.20844 §3.3 From verdict to catalog action（式 (3)）｜Q21

> 原文："Unpopulated attributes are excluded because only proposed values enter the production verification and write gate."
> 出处：2608.20844 §4.3 Evaluation Results｜Q27

> 原文："TRACE makes one ScoutAgent call and one JudgeAgent call per eligible SKU; each call returns a map of per-attribute outputs."
> 出处：2608.20844 §Appendix A Condensed Agent Prompt Templates｜Q44

### D. 离线评测

> 原文："We evaluate TRACE on products from four business verticals: Grocery, Alcohol, Electronics, and Home Improvement. The number of distinct target attributes ranges from 11 in Grocery to 409 in Home Improvement."
> 出处：2608.20844 §4.1 Data Collection｜Q22

> 原文："This dataset contains 500 SKUs and 2,497 target SKU–attribute pairs. Human annotators established the reference attribute values, and a separate group of auditors reviewed the values proposed by the ScoutAgent."
> 出处：2608.20844 §4.1 Data Collection｜Q23

> 原文："This dataset contains 955 SKUs and 4,990 target SKU–attribute pairs. Because exhaustive human labeling was not available for these verticals, we use the JudgeAgent to adjudicate the complete dataset. The JudgeAgent results provide a scalable operational quality signal."
> 出处：2608.20844 §4.1 Data Collection｜Q24

> 原文："In particular, we refer to the fraction receiving PASS or UNVERIFIED as the judge-supported rate. This metric measures compliance with the JudgeAgent’s evidence policy and is not interpreted as human-validated accuracy."
> 出处：2608.20844 §4.2 Evaluation Metrics｜Q25

> 原文："Judge-supported rate is the fraction of all extracted values receiving a PASS or UNVERIFIED verdict; UNCERTAIN and invalid responses remain in the denominator. Publication coverage is the fraction of requested attributes receiving one of these two verdicts. Costs are normalized to Gemini 2.5 Flash."
> 出处：2608.20844 §4.4 Table 1 表注｜Q30

> 原文："On the fully human-labeled Grocery and Alcohol dataset, the ScoutAgent achieved 98.2% extraction accuracy at 74.7% attribute coverage."
> 出处：2608.20844 §4.3 Evaluation Results｜Q26

> 原文："Among values assigned PASS, 98.4% were confirmed correct by human reviewers. Of the disagreements between the JudgeAgent and human reviewers, 87.8% were false rejections — values assigned FAIL but judged correct by humans — whereas 12.2% were false acceptances."
> 出处：2608.20844 §4.3 Evaluation Results｜Q28

> 原文："On the Electronics and Home Improvement dataset, the ScoutAgent achieved 87.8% attribute coverage. Of the extracted values, 97.4% received a PASS or UNVERIFIED verdict from the JudgeAgent."
> 出处：2608.20844 §4.3 Evaluation Results｜Q29

> 原文："Although Gemini 3.5 Flash increases extraction coverage by $0.5$ percentage points, its lower judge-supported rate reduces publication coverage from 85.5% to 81.9%, while increasing inference cost by more than $7\times$."
> 出处：2608.20844 §4.4 LLM Backbone Comparison｜Q31

> 原文："GPT-5.4 and Claude Sonnet 5 achieve still lower publication coverage, at 73.2% and 63.5%, respectively."
> 出处：2608.20844 §4.4 LLM Backbone Comparison｜Q32

> 原文："Error analysis shows that the lower judge-supported rates of the alternative backbones arise primarily from unsupported or partial extractions rather than explicit contradictions or hallucinations. These errors are concentrated in evidence-intensive attributes, including unit count and free-text descriptions."
> 出处：2608.20844 §4.4 LLM Backbone Comparison｜Q33

### E. 生产部署与线上实验

> 原文："We deployed TRACE in production and enriched 31 million SKUs across four business verticals."
> 出处：2608.20844 §5 Deployment｜Q34

> 原文："This increased impression-weighted enrichment coverage, defined as the share of customer impressions associated with product records carrying enriched attributes, by over 90% across these verticals."
> 出处：2608.20844 §5 Deployment｜Q35

> 原文："We evaluate this hypothesis through a five-week randomized A/B test with 90% of traffic assigned to treatment and 10% to holdout."
> 出处：2608.20844 §5.1 User Impact｜Q36

> 原文："Effects are reported as relative changes versus the control group, with 95% confidence intervals and $p$-values."
> 出处：2608.20844 §5.1 Table 2 表注｜Q45

> 原文："| Checkout conversion | +0.48% | [+0.04%, +0.92%] | 0.034 |"
> 出处：2608.20844 §5.1 Table 2（Checkout conversion）｜Q37

> 原文："| Checkout conversion (power users) | +1.18% | [+0.24%, +2.12%] | 0.014 |"
> 出处：2608.20844 §5.1 Table 2（Checkout conversion, power users）｜Q38

> 原文："| Missing/incorrect item rate | −1.08% | [−2.04%, −0.13%] | 0.026 |"
> 出处：2608.20844 §5.1 Table 2（Missing/incorrect item rate）｜Q39

> 原文："As shown in Table 2, the enriched PDP increased checkout conversion by $0.48\%$, with a larger $1.18\%$ increase among power users. It also reduced the missing/incorrect-item rate by $1.08\%$."
> 出处：2608.20844 §5.1 User Impact｜Q40

### F. 论文自承局限

> 原文："The Electronics and Home Improvement results use judge-supported rate as a scalable operational metric rather than as a substitute for human-validated precision. The JudgeAgent was calibrated on the Grocery and Alcohol human audit, while its transfer to other categories has received more limited human evaluation."
> 出处：2608.20844 §6 Limitations｜Q41

> 原文："Moreover, because the ScoutAgent and JudgeAgent use models from the same family in the primary configuration, they may exhibit correlated failure modes."
> 出处：2608.20844 §6 Limitations｜Q42

> 原文："The online experiment evaluates the end-to-end effect of displaying enriched product pages. It therefore demonstrates the value of the deployed system as a whole, but does not isolate the contribution of the JudgeAgent or write-gating policy."
> 出处：2608.20844 §6 Limitations｜Q43

## 5. 未能核验 / 论文未报告 / 主动放弃的断言（显式清单）

### 5.1 论文**未报告**，卡片因此不写数字

1. **成本量级**：论文未给任何金额、单价、算力或延迟数字。卡片 ⑤ 的 `C_sys` 只写
   「每个合格 SKU 一次侦察调用 + 一次裁决调用」（Q44 是唯一可用的成本线索），不给任何金额。
2. **单 SKU 处理耗时 / 吞吐**：**论文未报告**。卡片不提任何吞吐或延迟数字。
3. **人工补全一个 SKU-属性对需要多久**：**论文未报告**（论文只说人工补全「在规模与增速下不可行」，
   是定性判断，Q2）。卡片因此**不编造任何人工工时数字**，② 段也不给「节省 X% 人力」。
4. **UNCERTAIN / REVIEW 的占比**：**论文未报告**（论文给了四值裁决的分类逻辑与判官支持率的定义，
   但没给四个裁决各自的分垂类占比）。卡片 ⑤ 明确写「本卡不对复核工作量做任何估计」。
5. **阈值 θ 的取值**：论文只把它写成符号，未给具体数值。卡片 ③ 的 `theta=0.6` 是**本卡代码的默认参数**，
   不是论文值，代码注释里已标明。
6. **母婴 / 跨境电商场景的任何数字**：论文完全没有这个话题。卡片 ② 的市场、品类、渠道全部是业务背景
   （由业务方给定），**没有任何数字来自论文**；ROI 公式里的 `m`、`c_fix` 是符号而非数值。
7. **发表时间**：全文存档无日期，registry 的 `2026-08-21` **未经核验**。

### 5.2 论文有数字但**口径不支持**按原样使用

1. **`90.4%`**：只在摘要（Q4），正文对同一指标写 `over 90%` 且无分垂类表（Q35）。已在 §2 标注，
   卡片 ⑤ 用「口径提示」块显式说明。
2. **`98.2%` / `74.7%`**：人工口径，但只覆盖 Grocery + Alcohol 这一对垂类（Q23 / Q26）；
   另外两个垂类用的是判官自评（Q24），不是人工精度（Q25 / Q41）。
3. **`+0.48%` / `+1.18%` / `−1.08%`**：是**整条链路 + PDP 展示**的端到端效果，
   不能拆给 JudgeAgent 或写库门（Q43）；场景也不是母婴跨境。
4. **Table 1 的 backbone 对比**：作者自己写明「所有 Scout 变体都用同一个固定判官评测，
   因此这是运营行为的受控比较，而非人工验证精度的估计」（Q30 / §4.4 末段）。
   卡片只引用它来说明「更贵更强 ≠ 发布覆盖率更高」，不把它当作质量结论。
5. **`87.8%` / `97.4%`**：这两个数是 judge-supported rate 口径，**不是**人工验证精度（Q25）。

### 5.3 主动放弃引用

1. **Figure 1 / Figure 2 / Figure 3 / Figure 4 的图内内容**：存档只保留了图注，图本体不在 Markdown 里，
   无法逐字核验，故不引用图中任何信息（含两张 prompt 模板图的实际内容）。
2. **Algorithm 1 的逐行伪代码**：存档以列表形式保留，但夹着 LaTeX 标记，逐字引用会误导读者；
   卡片改用文字复述其结构（模板 → 自有来源 → 未解决才检索 → 归一/弃权 → 判官 → 写库策略），
   并引用论文自己的文字描述（Q12）作为出处。
3. **参考文献条目**：与本卡断言无关，不引用。
4. **`Declarations on Generative AI`**：与卡片结论无关。
5. **与作者实现的等价性**：论文**未公开代码**（全文无代码仓库链接）。K1 只证明**卡片自带代码**
   可执行（L1–L5 全绿），**不证明**它与论文的生产实现一致；卡片 ③ 段已显式声明
   「Scout / Judge 是确定性规则替身，不是论文里的 LLM Agent，输出不可用来印证论文数字」。

## 6. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
C=paper2skills-vault/00-电商Agent/Skill-Agentic-Catalog-Enrichment.md

# 1) K1 代码可执行（五级）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" \
  --json-out paper2skills-research/data/verification/k1_p2s-2026-0011.json

# 2) G2b 引文逐字核验（并自证核验器可信）
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$C"

# 3) K2 三合一门禁（G1 需 K1 凭证；不传 --k1 时 G1 按设计判红）
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" \
  --k1 paper2skills-research/data/verification/k1_p2s-2026-0011.json
```

引文切片的复现方式：本卡 ⑥ 段与本文件 §4 的每一条引文，都由脚本按
「`fulltext.md` 行号 + 起始锚点 + 结束锚点」切出（例：`Q37 = 第 243 行`,
`'| Checkout conversion |'` → `'| 0.034 |'`），并在写入前断言
「切出的字符串必须是 `fulltext.md` 的逐字子串、且在卡片中只出现一次」。
因此手工修改引文会立即被 `quote_check.py` 判为 FUZZY / FABRICATED。

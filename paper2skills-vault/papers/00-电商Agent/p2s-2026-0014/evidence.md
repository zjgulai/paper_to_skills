---
title: 证据链 — p2s-2026-0014（Skill-Live-Catalog-Conversational-Rec）
doc_type: evidence
module: 00-电商Agent
paper_id: 2608.27006
registry_id: p2s-2026-0014
related_card: paper2skills-vault/00-电商Agent/Skill-Live-Catalog-Conversational-Rec.md
fulltext: paper2skills-vault/papers/00-电商Agent/p2s-2026-0014/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2608.27006 *Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval*

- **全文存档**：`paper2skills-vault/papers/00-电商Agent/p2s-2026-0014/fulltext.md`
  （122 行 / 17,289 字节，`evidence_grade: A`；头注释载明来源 `https://arxiv.org/html/2608.27006v1`）
- **章节全集**（由存档标题行机械枚举，非人工估计）：
  `1. Introduction`｜`2. System Overview`（`2.1. Self-Refreshing Retriever`：Fetch and parse /
  IDs, hashes, and embeddings / Change classes；`2.2. Conversational Pipeline`）｜`3. Demonstration`｜
  `4. Concluding Remarks`｜`Appendix A Engine Architecture`｜`References`。
  **不存在 Experiments / Evaluation / Results 章节** —— 这是本卡所有边界声明的根据。
- **venue**：存档首页载明 `Conference: 20th ACM Conference on Recommender Systems; September 27-October 02, 2026;
  Minneapolis, MN, USA`、`DOI: 10.1145/3773078.3841297`、`ISBN: 979-8-4007-2284-4/2026/09`。
  存档内**没有** `demo track` 之类的分会场字样；「Demo」这一标注来自论文自身的 §3 `Demonstration`、
  三页篇幅与 registry 的 note，按 `MasterPrompt-v2.md` R3「demo 必须显式标注」记为
  `venue: RecSys 2026 (Demo)` / `venue_tier: top`。
- **引文全部为逐字切片**：卡片 ⑥ 段的每条 `> 原文："…"` 均由脚本按起止锚点从上述存档中
  **直接切出**（切出时对每个片段做 exact-substring 断言），没有拼接、没有改写、没有跨段缝合。
- ⚠️ **存档没有页码标记**（全文 0 处页码），因此出处一律给到**章节号**，不给「PDF 第 N 页」。

---

## 0. 核验凭证（可复核的退出码 / stdout）

```
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <card>
已索引仓库内本地模块 86 个（可自动解析的卡片将真正执行）
另有 54 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）
✅ PASS         Skill-Live-Catalog-Conversational-Rec.md#stitched(1块)

==================================================================
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | 🟠 MIGRATED 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）
==================================================================
```

逐级明细（`--json-out` 产物）：

```
L1_SYNTAX  passed=True  ast.parse 通过
L2_COMPILE passed=True  py_compile 通过
L3_IMPORT  passed=True  模块可 import
L4_SMOKE   passed=True  脚本执行成功
L5_TEST    passed=True  pytest 全绿
```

```
$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <card>
✅ VERBATIM   Skill-Live-Catalog-Conversational-Rec.md  37/37 逐字

共 1 张卡：1 通过，0 含伪造引文，0 无全文可核验
```

```
$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
1 真引文     : VERBATIM   连续度=1.0
2 纯伪造     : FABRICATED 连续度=0.124   ← 必须 FABRICATED
3 拼接引文   : FABRICATED 连续度=0.681 召回=1.0 拼接标记=True   ← 必须被拦下
4 排版差异   : VERBATIM   连续度=1.0   ← 必须 VERBATIM
✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接
```

```
$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <card> --k1 <K1 JSON>
======================================================================
G1 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================
======================================================================
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 2
======================================================================
======================================================================
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================
```

- **G2 的两条黄灯**均为 v2 模板强制的 frontmatter 日期字段（`created: 2026-09-12` / `updated: 2026-09-12`
  里的 `12`），与参考卡 `Skill-Causal-Budget-Allocation.md` 的黄灯同源，不构成事实断言。
- **G2 的 6 个高价值断言全部有出处**（`traceability_pct = 100.0`），引文 37/37 逐字，0 伪造 0 近似
  0 拼接 0「仅结构性出处」。
- ⚠️ **G1 必须带 K1 凭证才判绿**：`gate_check.py --card <card>`（不带 `--k1`）会判
  `G1-NO-EVIDENCE 无 K1 验证凭证`，这是门禁的有意设计（「不得凭人工判断放行」），
  不是本卡缺陷。上面的 G1 绿灯是在传入本次 K1 JSON 后得到的。

### 0.1 ⚠️ 本轮发现的门禁自身缺陷：K1 的 L3 探针在 CPython v3.14 下与 `@dataclass` 不兼容

本卡 ③ 段的代码**最初用了 `@dataclass`**，K1 报出：

```
L3_IMPORT  False  [block1] import 时崩溃
  File ".../unit.py", line 91, in <module>
    @dataclass
  File "<python3.14>/dataclasses.py", line 814, in _is_type
    ns = sys.modules.get(cls.__module__).__dict__
AttributeError: 'NoneType' object has no attribute '__dict__'
```

**根因在探针，不在被测模块。** `verify_skill_code.py` 的 L3 探针写法是：

```python
spec = u.spec_from_file_location('unit', r'{src.name}')
m = u.module_from_spec(spec)
spec.loader.exec_module(m)          # ← 没有 sys.modules['unit'] = m
```

Python 3.14 的 `dataclasses._is_type` 会执行 `sys.modules.get(cls.__module__).__dict__`；
被测模块**没有注册进 `sys.modules`** 时该表达式返回 `None` → `AttributeError`。
最小复现（无需任何卡片）：

```bash
python3 - <<'EOF'
import importlib.util as u, textwrap
src = "/tmp/unit_dc.py"
open(src, "w").write(textwrap.dedent('''
    from __future__ import annotations
    from dataclasses import dataclass
    @dataclass
    class A:
        x: int = 0
'''))
spec = u.spec_from_file_location('unit', src)
m = u.module_from_spec(spec)
spec.loader.exec_module(m)      # AttributeError
EOF
```

**影响面**：不止本卡 —— 全仓库**任何**含 `@dataclass` 的卡片在当前环境（CPython v3.14）下
L3_IMPORT 都会失败。由于 K1 的总体判定是「L4 或 L5 通过即 PASS」，这些卡的总判定仍是 PASS，
但报告行会挂上一句 `← L3_IMPORT: [block1] import 时崩溃`，容易被误读成卡片缺陷。
**建议的修法**（留给门禁维护者，本卡不动门禁脚本）：探针里补一行 `sys.modules['unit'] = m`，
或改用 `runpy` / `importlib.import_module` 的方式加载。

**本卡的处理**：把 `@dataclass` 换成 `typing.NamedTuple` + 普通类（机制、断言、输出一字未改），
使 L1–L5 五级全绿，并在卡片 ③ 段把这件事显式写出来，避免下一位读者重踩。
这不是「为了绿灯改代码」——判据（可执行性、断言）一条都没有放宽，只是绕开了一个已知的探针 bug。

---

## 1. 与 registry 的逐项核对（`papers_registry.json` → `p2s-2026-0014`）

| registry 字段 | registry 原值 | 回原文核对 | 结论 |
|---|---|---|---|
| `title` | Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval | 存档一级标题逐字一致 | ✅ 一致 |
| `identifiers.arxiv` / `url` | 2608.27006 / arxiv.org/abs/2608.27006 | 存档头注释 `arxiv_id: 2608.27006` | ✅ 一致 |
| `venue` / `venue_tier` | `RecSys` / `top` | 存档首页：20th ACM Conference on Recommender Systems（RecSys '26），Minneapolis，DOI 10.1145/3773078.3841297 | ✅ 一致（按 R3 补注 Demo 分会场属性） |
| `domain` | `00-电商Agent` | 本域新建，本卡为域内第一张卡 | ✅ 一致 |
| `note` | 「RecSys 2026，**3 页短文（Demo）**，标注时勿写成完整方法论文」 | 章节全集见上，无 Experiments/Evaluation；§3 标题即 `Demonstration` | ✅ **该提醒完全成立，已落实**（卡片开头的诚实边界声明 + ①b） |
| `decision_reason` | 「…**LLM 只做意图分类**，检索/重排走专用函数」 | 原文：`using an LLM only for intent classification **and preference elicitation** while retrieval, reranking, **and diversity selection** run as dedicated functions`（⑥ Q17） | ❌ **两处漏项，已在卡片中写准**（见下方 1.1） |
| `data_availability` | `available` | §3 原文：`Documentation, engine excerpts, a demo video, and a runnable synthetic sync are public; the full engine, catalogue, and Infobip integration remain private.`（⑥ Q28） | ⚠️ **建议收窄为 `partial`**（见下方 1.2） |
| `outputs.code_dir` | `paper2skills-code/ecommerce_agent/<algo>` | 该目录不存在；本卡按本批惯例把代码模板内嵌在卡片 ③ 段 | ⚠️ **规划值，未落地**（与 CLAUDE.md 对 3A/3B 的记载一致） |
| `outputs.skill_card` / `outputs.evidence` | 模板路径 | 实际路径见本文件 frontmatter | ✅ 已按模板落地 |
| `decision` / `priority` / `score` | extract / P0 / 64.8 | 属流程元数据，不涉原文 | — 不核 |
| `gates.*` | 三项 `pending` | 本轮三门禁结果见 §0 | ⚠️ **需由上游更新为实测结论**（本卡无权改 registry） |

### 1.1 ❌ 出入一：registry 把 LLM 的职责说少了一半

- **registry 写**：「LLM 只做意图分类，检索/重排走专用函数」
- **原文写**（⑥ Q17，逐字）：
  `A controller-based dialogue layer consumes this index, using an LLM only for intent classification and preference elicitation while retrieval, reranking, and diversity selection run as dedicated functions.`
- **两处漏项**：
  1. LLM 做的是**两件事**：`intent classification` **与** `preference elicitation`（偏好询问）；
     registry 只留了前者。这不是措辞小事 —— 它把「对话层里唯一需要生成模型的部分」说成了只做分类，
     会让人误以为偏好澄清是规则/模板实现的。原文还有独立证据：§2.2 的 elicitor 子代理
     「ask one to three clarifying questions when preferences are vague」（⑥ Q18），
     以及 Figure 2 图注的 `calling a generative model only for intent and elicitation`（⑥ Q34）。
  2. 「走专用函数」的清单也少一项：原文是 `retrieval, reranking, **and diversity selection**`，
     registry 只写了「检索/重排」。多样性选择同样是专用函数（§2.2：`a greedy selector adds brand and
     category variety`，⑥ Q19）。
- **本卡的处理**：① 段、①b、④ 段与 ⑥ 段一律写成「意图分类 **+ 偏好 elicitation**」；
  ④ 段把 diversity selection 显式挂在 `Skill-Diversity-Reranking-SMMR.md` 上。
  **建议上游同步修正 registry 的 `decision_reason`。**

### 1.2 ⚠️ 出入二：`data_availability: available` 偏高

论文自己划了公开边界（⑥ Q28）：公开的是**文档、引擎节选、一段 demo 视频，以及一个可运行的合成同步**；
**完整引擎、目录数据与 Infobip 集成保持私有**。因此按 `2608.25871` 那次的同一口径，
`data_availability` 更准确的取值是 **`partial`**（有可运行的合成同步与文档，但没有真实目录与完整引擎）。
本卡 ①b 已写明「本卡只复现机制，不声称与作者实现等价」。**建议上游修正 registry。**

---

## 2. 卡片正文数字 → 出处对照表

卡片**散文**里出现的数字极少（这是论文的性质，不是省略）。逐条如下：

| 卡片中的数字 | 含义 | 位置 | 引文键 |
|---|---|---|---|
| 3 页 | 论文篇幅（Demo 短文） | ①b / 开头边界声明 | 存档结构（章节全集见文件头）+ registry `note` |
| 500（条记录） | Table 1 的案例目录规模（匿名） | ② 场景 1、⑤ 参数表 | ⑥ Q21 |
| 1.8%（–12.3%） | 增量同步耗时占全量重建的下界 | ② 场景 1、⑤ 参数表 | ⑥ Q21/Q22 |
| 12.3% | 同上，上界（改描述/类目场景） | ⑤ 参数表 | ⑥ Q22 |
| 2.914 s | 全量重建基准耗时 | ⑥ Q21 的引文内 | ⑥ Q21 |
| 0.053 / 1.8%｜0.321 / 11.0%｜0.072 / 2.5%｜0.357 / 12.3%｜0.062 / 2.1% | Table 1 五行（无变化 / 新增 / 改价缺货 / 改描述类目 / 删除）的秒数与占全量比例 | ⑥ Q21/Q22 的引文内 | ⑥ Q22 |
| 三次或五次运行 | Table 1 取中位数的重复次数 | ⑥ Q21 的引文内 | ⑥ Q21 |
| RecSys 2026 / 2608.27006 / p2s-2026-0014 | venue 年份、arXiv ID、registry id（**标识符，不是事实断言**） | frontmatter / 全文 | 存档头注释与首页 |
| 2026-09-12 | 卡片创建/更新日期（模板强制字段） | frontmatter | — （G2 的两条黄灯即来自这里） |
| CPython v3.14 | 门禁探针 bug 的环境版本 | ③ 段说明 | 本文件 §0.1 的可复现命令 |

对照表之外的数字只有两类：③ 段代码块内的**构造数据**与本地可复现输出
（代码围栏在 G2 中整体剥离，按 `MasterPrompt-v2.md` 属 B 类「本地可复现数字」），
以及 ⑥ 段引用块自身的编号（`Q01`…`Q37`，字母在前，不构成数字断言）。

### 2.1 ⚠️ 与萃取简报的口径差异（**必须显式记录**）

萃取简报预先告知：「这是一篇 Demo 短文，**没有实验、没有评测、没有任何效果数字**」，
并要求卡片里「绝对不能出现任何性能数字（准确率、转化率、**延迟**、成本一律没有）」。

**逐字核对后，这条前提只有一半成立：**

- ✅ **成立的部分**：「没有 Experiments / Evaluation 章节」为真（§0 的章节全集可机械复核）；
  **没有任何推荐质量 / 效果数字**（准确率、转化率、GMV、留存、召回）也为真 —— 论文自己说
  `The evaluation covers synchronization not ranking quality`（⑥ Q30）。
- ❌ **不成立的部分**：论文**确实报告了量化数字**，集中在 §3 的 **Table 1**
  「增量同步 vs 全量重建」的耗时对照（匿名 500 条记录目录、三次或五次运行取中位数、
  全量重建 2.914 s、五个变化场景各占 1.8% / 11.0% / 2.5% / 12.3% / 2.1%）。
  这些是 **同步成本（含耗时）** 数字，逐字存在于全文存档中。

**本卡的处理（并说明为什么不是删掉它们）**：

1. **保留**，并在 ⑥ 段 G 组逐字给出出处。理由：它们是论文**唯一的量化证据**，
   也是 Concluding Remarks 里 `a measured case study`（⑥ Q29）所指的那份案例研究。
   若把它们整体删掉、再在 ①b 写「论文未报告任何数字」，那句声明本身就是**假的** ——
   为了迎合简报前提而写一句不成立的边界声明，比引用真实数字更危险。
2. **严格限定口径**：卡片在开头边界声明、② 场景 1、⑤ 参数表三处反复写明
   「这是**同步成本**占比，**不是** GMV / 转化率 / 推荐质量」，并明确标注
   `只能当方向性参考`（规模、enrichment 调用数、向量库都与企业不同）。
3. **不越界**：卡片全文**没有**任何效果数字、没有把 Table 1 的数字外推成业务收益，
   ⑤ 段明确写「本卡不给收益金额结论」。
4. **结论**：简报的**意图**（不许把 Demo 短文写成方法论文、不许编效果数字）已 100% 执行；
   简报的**事实前提**（论文没有任何量化数字）经逐字核对不成立，此处如实记录，
   以免下一位读者据此把 Table 1 误当成不存在。

---

## 3. 逐字引文全表（37 条，与卡片 ⑥ 段一一对应）

> 每条均为从 `fulltext.md` 按锚点**直接切出**的连续子串；`Q22` 是 Table 1 表体，
> 底本里由空行分隔的多行在规范化（`\s+` → 单空格）后与卡片中的单行写法逐字等价 ——
> 这不是拼接：**行内每个单元格与行序都未改动**，只是把行间的空行折叠成空格，
> 而折叠前后经 `quote_check.py` 的 `normalize()` 得到同一串。

### A. 问题：静态预索引的评测设定 vs 持续变化的真实目录

> 原文："Conversational recommender systems based on large language models (LLMs) are usually evaluated on static, pre-indexed item collections, yet e-commerce catalogues change continuously as products are added or removed, repriced, and restocked."
> 出处：2608.27006 §Abstract

> 原文："Most large language model (LLM)-based conversational recommender systems (CRSs) are evaluated over fixed benchmark collections (He et al., 2023; Jannach et al., 2021), but in production the catalogue is a live object, continually updated."
> 出处：2608.27006 §1 Introduction

> 原文："Re-indexing the whole catalogue on every change is wasteful, yet letting the index drift degrades recommendations and surfaces out-of-stock or discontinued items."
> 出处：2608.27006 §1 Introduction

> 原文："Our emphasis is orthogonal to model quality."
> 出处：2608.27006 §1 Introduction

> 原文："We treat catalogue freshness—keeping the index consistent with a live assortment—as the engineering problem that makes such systems production-viable, complementing work on adapting LLM recommenders to refreshed indices (He et al., 2025)."
> 出处：2608.27006 §1 Introduction

### B. self-refreshing retriever 的定义与「只处理增量」

> 原文："Its central component is a self-refreshing retriever that ingests a merchant product feed, enriches the records, and synchronizes them into a vector index."
> 出处：2608.27006 §Abstract

> 原文："On each run, per-item hashes identify which products are new, changed, deleted, or unchanged, so only the delta is processed rather than rebuilding the whole catalogue."
> 出处：2608.27006 §Abstract

> 原文："We demonstrate a conversational shopping assistant built around one contribution: a self-refreshing retriever that re-embeds only new or semantically changed products, keeping synchronization proportional to the changed subset."
> 出处：2608.27006 §1 Introduction

> 原文："Each manual or scheduled run compares the latest catalogue snapshot with the index and applies only the difference; it does not monitor the feed continuously."
> 出处：2608.27006 §2.1 Self-Refreshing Retriever

> 原文："Our proof of concept uses ChromaDB through a swappable VectorStore interface."
> 出处：2608.27006 §2 System Overview

### C. 三种标识的分工：stable ID / full hash / semantic hash

> 原文："A stable product ID links snapshots and drives exact updates and deletions, but cannot answer natural-language queries."
> 出处：2608.27006 §2.1 IDs, hashes, and embeddings

> 原文："A full hash detects any feed-field change; a semantic hash over name, description, brand, and category identifies changes requiring re-embedding."
> 出处：2608.27006 §2.1 IDs, hashes, and embeddings

> 原文："The resulting vectors make products retrievable; the generative LLM is only an enrichment fallback."
> 出处：2608.27006 §2.1 IDs, hashes, and embeddings

### D. 五个变化类别与分流规则

> 原文："Comparing IDs and hashes yields five disjoint classes."
> 出处：2608.27006 §2.1 Change classes

> 原文："New and semantically changed records are enriched, embedded, and upserted; enrichment resolves category paths and extracts attributes by rule, with generative fallback."
> 出处：2608.27006 §2.1 Change classes

> 原文："Metadata-only changes, such as price or stock, retain the vector while updating the record and filters."
> 出处：2608.27006 §2.1 Change classes

> 原文："Deleted records are removed, and unchanged records are skipped."
> 出处：2608.27006 §2.1 Change classes

> 原文："New and semantically changed items are enriched, embedded, and upserted. Metadata-only changes update the stored record while keeping its vector. Deleted items are removed, and unchanged items are skipped."
> 出处：2608.27006 §Figure 2 图注（Engine architecture）

### E. LLM 只做意图分类与偏好 elicitation（registry 出入的关键证据）

> 原文："A controller-based dialogue layer consumes this index, using an LLM only for intent classification and preference elicitation while retrieval, reranking, and diversity selection run as dedicated functions."
> 出处：2608.27006 §Abstract

> 原文："calling a generative model only for intent and elicitation"
> 出处：2608.27006 §Figure 2 图注（Engine architecture）

> 原文："The conversation pipeline follows an orchestrator-as-controller pattern (Yao et al., 2023; Schick et al., 2023; Huang et al., 2025): a generative model classifies messages into eight intents, composes replies, and uses an elicitor sub-agent to ask one to three clarifying questions when preferences are vague (Shimazu, 2001; Sun and Zhang, 2018)."
> 出处：2608.27006 §2.2 Conversational Pipeline

> 原文："Recommendation uses content-based semantic retrieval: query and product text share one embedding space, metadata filters restrict candidates, an optional non-generative model reranks them (Yang and Chen, 2024; Kemper et al., 2024), and a greedy selector adds brand and category variety."
> 出处：2608.27006 §2.2 Conversational Pipeline

> 原文："This generation-free path keeps cost predictable, although embedding and reranking still call external models (Kolb et al., 2025); the pipeline detects the user’s language, retrieves in English, and replies in that language."
> 出处：2608.27006 §2.2 Conversational Pipeline

> 原文："The engine has three subsystems (Appendix A): a catalogue pipeline that ingests and indexes products, a conversation pipeline that handles multi-turn dialogue, and a storage layer, written by the former and read by the latter, providing vector search, user profiles, and session state. This shared storage decouples catalogue synchronization from dialogue, allowing each pipeline to run independently. All generative, embedding, and reranking calls use a single proxy, making model choices configuration rather than code."
> 出处：2608.27006 §2 System Overview

### F. 演示形态：WhatsApp 购物助手

> 原文："Our demonstration is a WhatsApp shopping assistant in which catalogue changes reach the recommendations after the next successful sync."
> 出处：2608.27006 §Abstract

> 原文："Users access the live demonstration on any smartphone through WhatsApp, with no application to install (Figure 1)."
> 出处：2608.27006 §3 Demonstration

> 原文："Sessions may be anonymous or personalised from prior purchases; the assistant elicits preferences, searches the live catalogue, and returns diverse in-stock products with links, and each successful sync exposes catalogue changes."
> 出处：2608.27006 §3 Demonstration

### G. Table 1：论文唯一的量化证据（同步成本，不是推荐质量）

> 原文："Incremental synchronization of an anonymized 500-record catalogue (medians over three or five runs; full rebuild: 2.914 s)."
> 出处：2608.27006 §3 Demonstration（Table 1 表注）

> 原文："| Change | E | Emb | M | D | Time (s) | Full (%) | | None | ✗ | ✗ | ✗ | ✗ | 0.053 | 1.8 | | Add product | ✓ | ✓ | ✗ | ✗ | 0.321 | 11.0 | | Price/stock | ✗ | ✗ | ✓ | ✗ | 0.072 | 2.5 | | Description/category | ✓ | ✓ | ✗ | ✗ | 0.357 | 12.3 | | Delete product | ✗ | ✗ | ✗ | ✓ | 0.062 | 2.1 |"
> 出处：2608.27006 §3 Demonstration（Table 1 表体）

> 原文："Table 1 confirms the intended change classification: price or stock changes update only metadata, description or category changes re-enrich and re-embed the record, and all ID, hash, feed-field, and embedding-call checks passed."
> 出处：2608.27006 §3 Demonstration

### H. 论文自承的边界、局限与可复现性

> 原文："Writes are non-transactional, so production requires truncated-feed detection, staging or rollback, and abnormal-delta monitoring; we make no tens-of-millions-scale claim."
> 出处：2608.27006 §3 Demonstration

> 原文："Documentation, engine excerpts, a demo video, and a runnable synthetic sync are public; the full engine, catalogue, and Infobip integration remain private."
> 出处：2608.27006 §3 Demonstration

> 原文："Synchronization rejects parser failures, invalid or duplicate IDs, and incompatible index versions, and rejects empty or incomplete snapshots to prevent inferred mass deletion; it prepares enrichment and embeddings before writing, then applies upserts and metadata updates before deletions."
> 出处：2608.27006 §3 Demonstration

> 原文："The prototype accepts Google Merchant Center Atom feeds directly; other catalogue sources or messaging channels require adapters to the product schema or engine API, while filtering, batching, consistency, and operations remain backend concerns behind the VectorStore interface."
> 出处：2608.27006 §3 Demonstration

> 原文："We presented a conversational shopping assistant whose self-refreshing retriever keeps a vector index consistent with a live catalogue, processing only new, changed, or removed products on each sync; a measured case study confirms the classification and operation counts behave as intended, and the assistant is deployed as a WhatsApp demo."
> 出处：2608.27006 §4 Concluding Remarks

> 原文："The evaluation covers synchronization not ranking quality; an offline relevance study and a live user study of recommendation quality and cost remain future work, alongside freshness-aware retrieval and hybrid ranking."
> 出处：2608.27006 §4 Concluding Remarks

> 原文："A live chatbot, documentation, and a recorded walkthrough are available at [https://github.com/infobip/infobip-agentic-crs]."
> 出处：2608.27006 §Abstract

### 3.1 Table 1 报告值的逐值回查

Table 1 的每个报告值在切片时都单独做过 exact-substring 断言（任一值在底本里找不到，切片脚本即报错退出），
再由 `quote_check.py` 对整块引文判 VERBATIM：

| 场景 | E | Emb | M | D | Time (s) | Full (%) |
|---|---|---|---|---|---|---|
| None | ✗ | ✗ | ✗ | ✗ | 0.053 | 1.8 |
| Add product | ✓ | ✓ | ✗ | ✗ | 0.321 | 11.0 |
| Price/stock | ✗ | ✗ | ✓ | ✗ | 0.072 | 2.5 |
| Description/category | ✓ | ✓ | ✗ | ✗ | 0.357 | 12.3 |
| Delete product | ✗ | ✗ | ✗ | ✓ | 0.062 | 2.1 |

- 表注另载：目录为 **anonymized 500-record catalogue**，数值为 **medians over three or five runs**，
  全量重建基准 **2.914 s**。
- **口径提醒**：这张表证明的是「分类与操作数符合预期 + 增量确实比全量省」，
  **不证明**推荐质量、转化或客诉有任何变化。

---

## 4. 论文未报告 / 未能核验的显式清单

本卡在写作中**主动放弃**了以下内容，原因是论文没给或无法逐字核验：

1. **任何推荐质量指标** —— 论文完全没有（⑥ Q30）。卡片不写准确率、召回、转化率、GMV、留存。
2. **任何业务收益金额** —— ⑤ 段只有符号公式，明确写「不给收益金额结论」。
3. **任何成本 / 人力 / 并发量级** —— 论文未报告；⑤ 段把 `c_compute` / `C_build` 全标为「企业自估」。
4. **千万级规模的任何结论** —— 论文主动放弃（`we make no tens-of-millions-scale claim`，⑥ Q27）。
5. **Table 1 的「三次或五次运行」到底哪个场景用了几次** —— 表注只说
   「medians over three or five runs」，**未逐行标注**；卡片因此只写区间含义，不给单场景的重复次数。
6. **开源仓库的任何活跃度指标**（star 数、提交频率、最近更新时间）—— 本卡**未访问**该仓库，
   按「不编造数字」的原则只引用论文给出的 URL（⑥ Q32），不描述仓库状态。
7. **页码** —— 存档无页码标记（全文 0 处），出处只到章节号。
8. **`venue` 的分会场归属** —— 存档内无 `demo track` 字样；「Demo」由 §3 标题、三页篇幅与
   registry note 推断，已在卡片 frontmatter 显式标注为 `RecSys 2026 (Demo)`。
9. **与作者实现的等价性** —— 完整引擎与目录私有（⑥ Q28）；K1 只证明**卡片自带代码**可执行
   （L1–L5 全绿），**不证明**它与论文未公开的引擎等价。
10. **多语言检索的召回质量** —— 论文只描述了「检测语言 → 用英文检索 → 用该语言回复」（⑥ Q20），
     **未做任何评测**；卡片 ①b 把这列为已知失败模式并注明「论文未做这项评测」。
11. **feed 时效与真实库存的一致性** —— 论文未讨论；卡片 ② / ⑤ 明确要求企业自测后才可写进预算。
12. **母婴 / 跨境电商场景的任何数字** —— 论文完全没有这个话题；② 段的品类与渠道是业务背景，
    **没有一个数字来自论文**。
13. **同域卡片 `Skill-Agentic-Catalog-Enrichment.md` 的具体内容** —— 该卡由**另一路并行 agent**
    在本轮同时产出（同目录 `00-电商Agent/`，论文 `2608.20844` TRACE）。本卡 ④ 段只引用了它的
    **文件名、标题与主题**（用于说明同一管线的上下游分工），**未逐段阅读其正文**，
    因此不对其结论负责；引用口径已在卡片 ④ 段显式标注。
    核对方式：`ls paper2skills-vault/00-电商Agent/` + 读取该文件 frontmatter 与章节标题行。

## 5. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
CARD=paper2skills-vault/00-电商Agent/Skill-Live-Catalog-Conversational-Rec.md

# 1. K1：代码可执行（L1–L5）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$CARD" \
  --json-out /tmp/p2s/k1.json

# 2. G2b：引文逐字核验（先自检，再核卡片）
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$CARD"

# 3. G1/G2/G3：带 K1 凭证跑，G1 才会判绿
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$CARD" \
  --k1 /tmp/p2s/k1.json --outdir /tmp/p2s/gates
```

---
title: venue 白名单与层级判定规则
doc_type: reference
module: 07-资源库
status: active
created: 2026-09-12
updated: 2026-09-12
owner: self
source: human+ai
---

# venue 白名单与层级判定规则

> 配套：`关键词库-v2.md`（检索）、`papers_registry.json`（venue_tier 字段）、
> `paper2skills-research/scripts/journal_harvest.py`（Crossref 路线）。
>
> **本文件的目的**：让「这篇论文值不值得萃取」的第一道筛子可复现，
> 而不是靠每次现查影响因子。

---

## 0. 为什么需要这份白名单

本轮实测暴露了两个必须用规则解决的问题：

| 问题 | 实测证据 | 规则对策 |
|------|----------|----------|
| **抬级**：workshop / findings / demo 被当成主会 | arXiv `comment` 字段里 152 篇带 venue 信号，其中混有 Workshop / Findings / Demonstrations | 见 §3 降级判定表 |
| **顶刊「新发表」≠ 新方法** | **79%** 的顶刊文章 DOI 年份段比发表年份早 ≥2 年（`10.1287/mnsc.2022.02462` 发表于 2026-07-09，DOI 却是 2022） | 见 §4 时效基准规则 |

---

## 1. 层级定义

| tier | 含义 | 在本项目中的处置 |
|------|------|------------------|
| `UTD24` | 德克萨斯大学达拉斯分校 24 本顶级商科期刊 | 最高优先；但**必须查预印本时效** |
| `FT50` | 金融时报 50 本商科期刊 | 高优先；同上 |
| `CCF-A` | 中国计算机学会 A 类（含会议与期刊） | 高优先；方法类首选 |
| `CCF-B` | CCF B 类 | 中优先；仅当方法确实新且业务契合 |
| `field-top` | 领域公认顶会（非 CCF 列表内但业界地位高） | 中优先 |
| `preprint` | arXiv 预印本（尚未见刊） | **本项目主力来源**；方法论类优先 |
| `non-paper` | 非论文来源（官方文档、行业报告、内部事故复盘） | 允许，用于补学术空白（见 §5） |

---

## 2. 白名单（本轮实际抓取的 28 刊 + 常用会议）

### 2.1 期刊（Crossref 路线，`journal_harvest.py` 已内置 ISSN）

**UTD24 / FT50 主力**

| 刊名 | ISSN | tier | 本项目相关方向 |
|------|------|------|----------------|
| Management Science | 0025-1909 | UTD24 | 决策模型、收益管理、因果推断 |
| Marketing Science | 0732-2399 | UTD24 | 广告增量、促销效果、MMM |
| Journal of Marketing | 0022-2429 | UTD24 | 增长策略、渠道、客户旅程 |
| Journal of Marketing Research | 0022-2437 | UTD24 | 实验设计、测量模型 |
| Journal of Consumer Research | 0093-5301 | UTD24 | 消费者行为（母婴品类心理） |
| Information Systems Research | 1047-7047 | UTD24 | 数据产品、推荐、平台治理 |
| MIS Quarterly | 0276-7783 | UTD24 | AI 治理、Agent 组织影响 |
| Operations Research | 0030-364X | UTD24 | 库存、履约、网络优化 |
| Manufacturing & Service Operations Mgmt | 1523-4614 | UTD24 | 供应链、多级库存 |
| Production and Operations Management | 1059-1478 | UTD24 | 库存与履约 |
| Journal of Operations Management | 0272-6963 | UTD24 | 供应链韧性 |
| Harvard Business Review | 0017-8012 | FT50 | 组织与战略（非方法） |

**CCF-A 期刊（AI / 数据挖掘 / IR）**

| 刊名 | ISSN | tier | 方向 |
|------|------|------|------|
| IEEE TKDE | 1041-4347 | CCF-A | 数据挖掘通用 |
| ACM TOIS | 1046-8188 | CCF-A | 信息检索、推荐 |
| IEEE TPAMI | 0162-8828 | CCF-A | 机器学习基础 |
| JMLR | 1532-4435 | CCF-A | 机器学习理论 |
| Artificial Intelligence | 0004-3702 | CCF-A | AI 基础 |
| ACM TIST | 2157-6904 | CCF-B | 智能系统 |

> 完整 ISSN 清单见 `paper2skills-research/scripts/journal_harvest.py` 的 `JOURNALS` 常量。

### 2.2 会议（arXiv `comment`/`journal_ref` 捕获 + 人工核对）

| 会议 | tier | 本项目主要用途 | 本季代表论文 |
|------|------|----------------|--------------|
| KDD | CCF-A | 需求预测、因果、推荐 | `2608.25871` CEDAR |
| SIGIR | CCF-A | 检索、推荐、复购 | `2607.12714` |
| WWW / TheWebConf | CCF-A | 电商与图 | — |
| CIKM | CCF-B | 电商、归因、转化 | `2608.11675` FunnelCausalNet |
| WSDM | CCF-B | 检索、推荐 | — |
| RecSys | CCF-B | 推荐系统 | — |
| IJCAI | CCF-B | AI 通用 | — |
| AAMAS | CCF-B | 多智能体（本项目 10-MAS 域对口） | — |
| UAI | CCF-B | 不确定性、因果 | — |
| EMNLP | CCF-B | NLP、VOC | — |
| NeurIPS / ICML / ICLR / ACL / AAAI | CCF-A | ML 基础、表格模型 | `2609.04540` Mitra-v2 |
| TKDE / TOIS | CCF-A | 数据挖掘 / 信息检索 | — |
| TNNLS | CCF-B | 神经网络 | — |
| **TIST** | ⚠️ **CCF-C**（**2026 版目录修正：原记为 B 是错的**） | 智能系统 | — |
| **TBD**（IEEE Trans. Big Data） | ⚠️ **CCF-C**（交叉/综合/新兴 #5） | 大数据 | — |
| **AISTATS** | ⚠️ **CCF-C** | 统计 ML | — |
| **COLM** | ⚠️ **不在 CCF 目录** | 语言模型 | — |
| **MLSys** | ⚠️ **不在 CCF 目录** | ML 系统 | — |
| ACL/EMNLP **Findings** | 降级 | **不等同主会** | — |
| ADKDD (KDD workshop) | 降级 | 工业实践，方法新意有限 | `2606.26690` |
| **NAACL 2026** | ❌ **不存在** | 2026 年停办一届（详见 §3.1） | — |

---

## 3. 降级判定表（抬级防护，**必查**）
扫描 arXiv `comment` / `journal_ref` 字段，命中左侧任一关键词即降级：

| comment 中出现 | 判定 | 处置 |
|----------------|------|------|
| `Workshop` / `workshop` | 降一级（CCF-A → CCF-B） | 除非论文有真实线上实验数字 |
| `Findings` (ACL/EMNLP Findings) | 降一级 | 同上 |
| `Demonstrations` / `Demo` | 降为 `field-top` 以下 | 方法新意通常不足，仅在有开源实现时收 |
| `Short paper` / `Extended abstract` / `Poster` | 降为 `preprint` | 通常信息量不足，不出卡 |
| `Under review` / `Submitted to` / `Preprint` | 保持 `preprint` | **不得**写成已接收 |
| `Accepted at` + 明确主会名 | 保持该 tier | 正常 |
| `to appear` | 保持，但标注 | 引用时用 `to appear` |

**反向规则（本轮实测得到的抬级案例，作为反例记忆）**：
- `2606.26690` 标注 ADKDD 2026（KDD workshop）→ 我们把它降级看待，
  但因**有 TikTok 多市场线上部署数字（蚕食率 −15pp）**，仍进入 P0。
  → **规则：venue 降级不自动否决，真实线上实验数字可翻案。**
- `2608.11675` CIKM 2026（CCF-B）→ 因有 GMV 分解 + 多档券分配 + 误差降 18–48%，进 P0。
- `2608.25871` CEDAR → **已核实出现在 KDD 2026 proceedings**，可安全标 `CCF-A`。

---

## 4. 时效基准规则（顶刊专属，**最容易踩的坑**）

**事实**：79% 的顶刊文章 DOI 年份段比发表年份早 ≥2 年。
Crossref 的 `published-online` 日期只反映**上线时间**，不是**方法产生时间**。

### 判定流程

```
拿到一篇 Crossref 顶刊文章
  ↓
1. 读 DOI 年份段（如 10.1287/mnsc.2022.02462 → 2022）
  ↓
2. 若 DOI 年份 ≤ 发表年份 − 2：
     标记 stale_method: true
     去 arXiv / SSRN / 作者主页 找预印本，取首版时间作为「方法年龄」
  ↓
3. 若预印本首版距今 > 3 年：
     除非是经典必收（明确标注为经典），否则不纳入本轮 P0
  ↓
4. 萃取时优先使用预印本版本（全文可得，且与作者原意一致）
```

### 实测样本

| DOI | 发表日 | DOI 年份 | 差 | 处置 |
|-----|--------|----------|-----|------|
| `10.1287/mnsc.2022.02462` | 2026-07-09 | 2022 | 4 年 | `stale_method: true`，找预印本 |
| 其余跨组重复 24% 的记录 | — | — | — | 先按 DOI 去重再排序 |

> **Crossref 跨组重复计数**：同一批 120 条记录实际只有 **91 篇唯一论文（重复率 24%）**。
> `make_journal_bundles.py` 已加 DOI 去重；任何新的期刊检索脚本都必须先去重再排序。

---

## 5. `non-paper` 来源的准入规则

学术空白必须靠非论文来源补（本轮实证：00-电商Agent 域 22 篇里，
关税/CE-FDA-CPSC/IP/汇率/逆向物流 **0 篇**）。但非论文来源**极易变成搬运**，故设门槛：

| 要求 | 说明 |
|------|------|
| 来源可公开引用 | 官方文档、法规原文、平台政策页、行业白皮书；**不接受**无出处的「据说」 |
| 必须标注更新频率 | 政策类内容会变，卡片必须写明「最后核对日期」与「建议复查周期」 |
| 同样过 G1/G2/G3 | 代码仍要跑通；事实仍要溯源（出处改为官方文档 URL + 章节） |
| `venue_tier: non-paper` | frontmatter 必须显式标注，禁止伪装成论文来源 |
| 不得声称因果结论 | 只能陈述规则与流程，不能给「提升了 X%」这类数字 |

---

## 6. 本季（2026-06-12 → 2026-09-12）实测覆盖情况

| 路线 | 工具 | 覆盖 | 实测 |
|------|------|------|------|
| arXiv API | `arxiv_harvest.py` | 预印本、方法论、模型类 | ✅ 49 查询 → **1046 篇**（92 天）；⚠️ 必须 `https://` + `curl -L`（`http://` 返回 301 空 body） |
| Crossref 期刊 | `journal_harvest.py` | UTD24/FT50/CCF 期刊 | ✅ 28 刊 → **1751 篇**（业务命中 783） |
| Crossref 会议录 | 待补 | KDD/SIGIR/WWW/WSDM 会议录（**dl.acm.org 403 的唯一替代**） | ✅ 免鉴权，含 `event.location`；DOI 前缀见 §7 |
| PMLR 卷级 BibTeX | 待补 | UAI / AISTATS / MLSys 部分 | ✅ **一次调用全量**：`proceedings.mlr.press/v{N}/assets/bib/bibliography.bib`（UAI v337 = 639 KB） |
| ACL Anthology | 待补 | ACL 2026 全 7 卷 | ✅ 但**慢**（`2026.acl-long.bib` >60s，须 `--max-time 180`）；无 JSON 端点 |
| 会议 proceedings | arXiv `comment`/`journal_ref` | 各会 | ⚠️ 部分覆盖（152 篇带 venue 信号） |
| **OpenReview** | — | — | ⚠️ **见 §7.1：403 有解，但覆盖被严重误估** |
| **DBLP** | — | — | ⛔ **彻底不可用，见 §7.2** |
| Semantic Scholar | — | — | ❌ **429 Too Many Requests** |
| OpenAlex | 待补 | 跨源补充与卷期交叉校验 | ✅ 免鉴权；⚠️ **按上线日索引，与 Crossref 封面日语义不同** |

**结论**：本季检索层实际是「**Crossref 打底 + arXiv 补预印本 + proceedings 一次性 BibTeX 兜底**」。
所有可自动化的正式出版物路线**全部免鉴权**。

---

## 7. 两个必须纠正的路线结论（原判断有误，已实测推翻）

### 7.1 OpenReview：不是「❌ 403 不可用」，而是「部分可用且覆盖易高估」

**事实**：`/notes` 端点确实 403（`ChallengeRequiredError`），但**同域名另两个端点可用**：

```
✅ GET https://api2.openreview.net/notes/search?term=%22SIGIR%202026%22&limit=1000&type=terms
       → 200，一次返回全部 123 条（limit=1000，比 offset 分页省事；count 上限 10000）
       → 唯一合法 type 是 terms（all/source 返回 400）
✅ GET https://api2.openreview.net/groups?id=ICML.cc%2F2026%2FConference
       → 200，返回 venue 元数据（start_date / location / website），可做日期交叉校验
❌ GET https://api2.openreview.net/notes?content.venue=...      → 403
❌ https://openreview.net/pdf?id=...                            → 403
```

**⚠️ 覆盖真相（最容易踩的坑）**：

| 查询 | 返回条数 | 实际构成 | 性质 |
|------|---------|---------|------|
| `"SIGIR 2026"` | 123 | invitation=`Record` 117 / `Direct_Upload` 5，`externalIds` 带 `dblp:` 116 | **DBLP 镜像，≠ 正会**（正会 274 篇**未覆盖**） |
| `"KDD (1) 2026"` | 82 | `Record` 82 | **DBLP 镜像** |
| `"UAI 2026"` | 414 | invitation=`Submission` 412 | 真 OpenReview 投稿（可用） |
| `"ICML 2026"` / `"ICLR 2026"` / `"ACL 2026"` / `"AAAI 2026"` | **0** | venue 字段未回填 | **正会枚举仍需个人 token** |
| `"EMNLP 2026"` | 147 | venue 实为 `GroundLM` / `NLP4PI` | 是 workshop 名，**不是主会** |

**结论**：可用，但**只能当补充路线，不得据此声称「会议全覆盖」**。

### 7.2 DBLP：不是「⚠️ 需带 UA 限速」，而是 **完全不可用**

| 测试 | 结果 |
|------|------|
| `dblp.org/search/publ/api?q=KDD+2026&format=json` | 连接**超时**（HTTP 000） |
| 同上 + Chrome UA | 仍然超时 |
| `dblp.uni-trier.de/…` | **HTTP 200 但返回 Anubis 挑战页** `<title>Making sure you're not a bot!</title>` |
| `dblp.dagstuhl.de/…`（第三镜像） | 同样挑战页 |
| `dblp.org/db/conf/kdd/kdd2026.html` | **200 + 挑战页**（不是 429） |

**`dblp.org` 已启用 Anubis proof-of-work 反爬，返回 HTTP 200 + 挑战页。**
这正是它最危险之处：**脚本只看状态码会误判为成功，然后把挑战页当数据解析**。
换 UA、降速、换镜像、换 API 路径**全部无效** → **脚本路线应彻底放弃**。

**替代**：Crossref（ACM DL DOI）+ OpenAlex。

### 7.3 Crossref 必须同时打两个日期过滤器（否则漏整期）

**SAGE 系（Journal of Marketing / JMR / POM）的 `published` 是 online-first 日期**，
只用 `from-pub-date` 会漏掉整期。实测对比：

| 期刊 | 只用 `from-pub-date` | 同时用 `from-print-pub-date` |
|------|---------------------|------------------------------|
| POM | 55 条，且几乎全无卷期 | **49 条干净的 `{35(7):16, 35(8):16, 35(9):17}`** |
| ISR | — | **55 条，print-date 过滤 → 0**（决定性证据：ISR 本季**无任何正式期号**，纯 Articles in Advance） |

**规则**：必须并行打两次，并按来源分别标注，
避免把 OpenAlex 的**上线日**误当**封面期**（二者语义不同）。

### 7.4 `NAACL 2026` 不存在（唯一假 venue）

2026 年停办一届。四条独立证据：
① `naacl.org` 首页只挂 NAACL **2027**；② `2026.naacl.org` **DNS 不解析**；
③ `aclanthology.org/events/naacl-2026/` → **404**；④ ACL Rolling Review 的 Venue 表只有 NAACL 2027。

⚠️ **陷阱**：conferencedeadlines.com 把 NAACL 2027 误标为「NAACL 2026 / San Francisco」。
**任何检索脚本的 venue 白名单里不得出现 NAACL 2026。**

---

## 8. 本季新发现的 DOI 前缀与取数入口（可直接复制）

| 会议/卷 | DOI 前缀 / 入口 | 上线日 |
|---------|----------------|--------|
| SIGIR 2026 | `10.1145/3805712` | 2026-07-19（窗内） |
| KDD 2026 **V.2** | `10.1145/3770855`（V.1 = `…3770854`，窗外） | 2026-08-08（窗内） |
| WWW 2026 | `10.1145/3774904`（Companion `…3774905`） | 2026-04-12（**窗外**） |
| WSDM 2026 | `10.1145/3773966` | 2026-02-21（窗外） |
| UAI 2026 | `proceedings.mlr.press/v337/assets/bib/bibliography.bib`（330 篇） | 2026-08-06（窗内） |
| AISTATS 2026 | `proceedings.mlr.press/v300/…` | 2026-08-30（窗内） |
| MLSys 2026 | `proceedings.mlsys.org/paper_files/paper/2026`（135 篇） | 未核实 |
| AAMAS 2026 | `ifaamas.org/Proceedings/aamas2026/forms/contents.htm`（649 PDF） | 未核实 |
| ACL 2026 | `aclanthology.org/volumes/2026.acl-long.bib` | 2026-06-23（窗内） |
| IJCAI 2026 | `2026.ijcai.org/accepted-papers/`（988 条名单；**proceedings 404**） | — |
| COLM 2026 | `colm.eventhosts.cc/Conferences/2026/AcceptedPapers`（⚠️ 按 `<a>` 抓得 0，须纯文本解析） | — |
| CIKM 2026 | ❌ 无 proceedings。⚠️ **流传的 `10.1145/3583780` 经 Crossref 查证实为 CIKM '23** | — |
| ICML 2026 | 名单有（`icml.cc/virtual/2026/papers.html`）；**PMLR 卷未上线**。⚠️ **「v340」已证伪**（v340 = MLHC 2026） | — |
| INFORMS 目录级 | `pubsonline.informs.org/action/showFeed?type=etoc&feed=rss&jc={mnsc\|mksc\|msom\|opre\|isre}`（5/5 可用） | — |
| Journal of Retailing 目录级 | `rss.sciencedirect.com/publication/science/00224359` ✅ | — |

**已确认阻断（脚本不要尝试）**：`dl.acm.org`（403 Cloudflare）、`journals.sagepub.com`（DNS 被劫持到 Facebook IP，TCP 超时）、
`api.semanticscholar.org`（429）、`tandfonline.com` / `sciencedirect.com` HTML / `misq.umn.edu` / `pubsonline.informs.org` HTML（403）、
`api.openreview.net/notes`（403）、DBLP 全镜像。

---

## 9. ISSN 陷阱（写检索代码时必查）

| 陷阱 | 说明 |
|------|------|
| Journal of Retailing e-ISSN `1873-3271` | Crossref「Resource not found」→ **只能用印刷版 `0022-4359`** |
| TBD `2372-2096` | **不是电子 ISSN**（是 CD-ROM，且被 ISSN Portal 标为 Incorrect）。`journals/2372-2096` → 404，但 `works?filter=issn:2372-2096` → 200。**正确值：`2332-7790`** |
| POM DOI 前缀变更 | 已迁至 SAGE，前缀由 `10.1111/poms.*` 变为 **`10.1177/10591478*`**；老流水线按 Wiley 前缀匹配会**全漏** |
| Crossref 按期次**封面日**索引 vs OpenAlex 按**上线日**索引 | 混用会得出错误卷期区间（TBD 12(5) 封面期实为 2026-10，窗外） |

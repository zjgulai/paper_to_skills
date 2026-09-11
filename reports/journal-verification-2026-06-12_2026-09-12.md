# IEEE/ACM 期刊核实报告（窗口 2026-06-12 → 2026-09-12）

核实日期：2026-09-11（UTC）。所有卷/期号均来自 API 实测，非记忆。

## 0. API 调用实测状态

| 调用 | 结果 |
|---|---|
| `api.crossref.org/journals/1041-4347` 等 9 个 ISSN | **200 OK**，均返回正确刊名 |
| `api.crossref.org/journals/2372-2096` | **404**（该 ISSN 无期刊级记录） |
| `api.crossref.org/journals/<ISSN>/works?filter=from-pub-date,until-pub-date` | **200 OK**（TOIS 首次返回空体，重试 200） |
| `api.openalex.org/works?filter=primary_location.source.issn:...` | **200 OK** |
| `api.openalex.org/sources/issn:...` | **200 OK** |
| `portal.issn.org/resource/ISSN/<ISSN>` | **200 OK** |
| `dblp.org/db/journals/<key>/index.html` | **200 但为 Anubis 挑战页**，非 TOC |
| `dl.acm.org/toc|journal/...` | **403 Cloudflare "Just a moment..."** |
| `ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=69 / 6687317` | **200 OK** |
| `ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385` | **418**（限流/风控，间歇性） |
| `ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=...` | **200 OK** |
| `developer.ieee.org/docs/read/Metadata_API_details` | **200 OK** |
| Semantic Scholar | 按要求回避 |

## 1. 六项要点总表

| 项目 | TKDE | TOIS | TIST | TNNLS | TBD |
|---|---|---|---|---|---|
| 全称 | IEEE Transactions on Knowledge and Data Engineering | ACM Transactions on Information Systems | ACM Transactions on Intelligent Systems and Technology | IEEE Transactions on Neural Networks and Learning Systems | IEEE Transactions on Big Data |
| 出版方 | IEEE (IEEE Computer Society) | ACM | ACM | IEEE | IEEE (IEEE Computer Society) |
| CCF | **A**（数据库/数据挖掘/内容检索） | **A**（同左） | **不在现行 CCF 目录中**（见 §3 更正） | **B**（人工智能 B 类 #12） | **C**（交叉/综合/新兴 C 类 #5） |
| 印刷 ISSN | 1041-4347 | 1046-8188 | 2157-6904 | 2162-237X | 无独立印刷版 |
| 电子 ISSN | 1558-2191（另 2326-3865 = 光盘版） | 1558-2868 | 2157-6912 | 2162-2388 | 2332-7790（Online，ISSN-L） |
| 窗口内卷期 | **38(7) Jul / 38(8) Aug / 38(9) Sep** | **44(5) Jun / 44(6) Jul / 44(7) Aug–Sep** + Just Accepted | **17(4) Jun / 17(5) Jun–Aug / 17(6) Aug** + Just Accepted | **37(7) Jul / 37(8) Aug / 37(9) Sep** | **12(4) Aug 2026** |
| 窗口内 Crossref 条目 | 148 | 63 | 87 | 148 | 29 |
| 目录公开 | 是 | 是 | 是 | 是 | 是 |
| 全文公开 | 否（订阅墙） | 否（订阅墙） | 否（订阅墙） | 否（订阅墙） | 否（订阅墙） |
| arXiv 预印本 | 通常有 | 通常有 | 通常有 | 通常有 | 混合，不确定 |

## 2. 逐刊明细

### IEEE TKDE
- **ISSN 实测**：`https://api.crossref.org/journals/1041-4347` → **200**，标题 `IEEE Transactions on Knowledge and Data Engineering`，ISSN 数组 `['1041-4347','1558-2191','2326-3865']`。ISSN Portal：1041-4347 = **Print**，Other media = Online(1558-2191) / Optical disk(2326-3865)。
- **窗口内卷期（Crossref 实测 200，total=148，全部取样）**：`38(7) Jul=49`、`38(8) Aug=49`、`38(9) Sep=50`。**无 38(10)**。
- **注意**：38(6) = 2026-06 为窗口前一卷期；Crossref 对月粒度日期按当月 1 日入库，故 `from-pub-date:2026-06-12` 将其排除。
- **OpenAlex 差异**：按**上线日期**过滤时出现 `38(9) 2026-07-07`、`38(10) 2026-07-22…08-14` —— 即已定 10 月期的稿件在 7–8 月就已在线（IEEE Early Access 语义）。Crossref 按**期次封面日期**记 10 月，故落在窗口外。
- **官方 URL**：`https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=69`（HTTP 200 实测）
- **滚动模型**：IEEE **Early Access**（录用后先在线，后归期）
- **检索路由（均实测）**：
  - Crossref：`https://api.crossref.org/journals/1041-4347/works?filter=from-pub-date:2026-06-12,until-pub-date:2026-09-12&rows=5&select=DOI,title,published,volume,issue` → 200，148 条
  - OpenAlex：`https://api.openalex.org/works?filter=primary_location.source.issn:1041-4347,from_publication_date:2026-06-12,to_publication_date:2026-09-12&per-page=5` → 200，count=84
  - DOI 前缀 `10.1109`；实测 `https://doi.org/10.1109/TKDE.2026.3682749` → 302 → `https://ieeexplore.ieee.org/document/11479824/`
  - DBLP `journals/tkde` → Anubis 挑战页
- **arXiv**：**通常有**。CS 领域会议前预印本已成常态（2013 年 48.6% → 2022 年 90.1%，arXiv:2401.11116）。单篇需逐一核实。

### ACM TOIS
- **ISSN 实测**：`/journals/1046-8188` → **200**，`ACM Transactions on Information Systems`，ISSN `['1046-8188','1558-2868']`。Portal：1046-8188 = **Print**，Online = 1558-2868。
- **窗口内卷期（Crossref 200，total=63）**：`44(5) 2026-06=8`、`44(6) 2026-07=24`、`44(7) 2026-08=8` + `44(7) 2026-09=5`、**无卷期号（Just Accepted）= 7月1 + 8月13 + 9月4 = 18**。
- **OpenAlex**：200，count=39（含 `44/6 2026-06-13`、`44/7 2026-07-25`）。
- **官方 URL**：`https://dl.acm.org/journal/tois`
- **滚动模型**：ACM **Just Accepted**（AAM，录用即挂网，后归期）→ `https://dl.acm.org/toc/tois/justaccepted`（该路径由检索确认存在）
- **TOC 页模式**：`https://dl.acm.org/toc/tois/<年>/<卷>/<期>`，例 `https://dl.acm.org/toc/tois/2026/44/7`
- **检索路由**：DOI 前缀 `10.1145`，实测 `https://doi.org/10.1145/3822505` → 302 → `https://dl.acm.org/doi/10.1145/3822505`
- **⚠️ 可达性实测**：`dl.acm.org` 全部路径对非浏览器客户端返回 **403 Cloudflare "Just a moment..."**（TOIS/TIST 期刊页与 TOC 页均 403）。**目录页对人类浏览器公开，对脚本不公开**。DBLP `journals/tois` 被 Anubis 拦截。
- **arXiv**：**通常有**。

### ACM TIST
- **⚠️ CCF 更正**：任务前提「TIST CCF B」**不成立**。直接抓取 CCF 官网全部 10 个学科页面（`https://www.ccf.org.cn/Academic_Evaluation/<domain>/`）并全文检索 `Intelligent Systems and Technology` → **10/10 页面 0 命中**。仅有的 "TIST" 字面命中是 `AISTATS` / `Statistical` 的子串误报。TIST **不在现行 CCF 推荐国际学术刊物目录中**。（"曾为 2019 版 CCF B" 一说 **未经核实**。）
- **ISSN 实测**：`/journals/2157-6904` → **200**，ISSN `['2157-6904','2157-6912']`。Portal：2157-6904 = **Print**，Online = 2157-6912。
- **窗口内卷期（Crossref 200，total=87）**：`17(4) 2026-06=4`、`17(5) 06=5 / 07=15 / 08=10`、`17(6) 2026-08=12`、**无卷期号（Just Accepted）= 06:2 / 07:15 / 08:18 / 09:6 = 41**（占比 47%）。
- **OpenAlex**：200，count=62。
- **官方 URL**：`https://dl.acm.org/journal/tist`；Just Accepted：`https://dl.acm.org/toc/tist/justaccepted`（模式同 TOIS；本环境 403 未能直取，**该具体 URL 未经核实**）
- **检索路由**：DOI 前缀 `10.1145`，TOC 模式 `https://dl.acm.org/toc/tist/2026/17/6`（403 于本环境）
- **arXiv**：**通常有**。

### IEEE TNNLS
- **CCF 实测**：`https://www.ccf.org.cn/Academic_Evaluation/AI` → B 类 #12 `TNNLS IEEE Transactions on Neural Networks and learning systems IEEE`，**CCF B（人工智能）**。
- **ISSN 实测**：`/journals/2162-237X` → **200**，ISSN `['2162-237X','2162-2388']`。Portal：2162-237X = **Print**，Online = 2162-2388。
- **窗口内卷期（Crossref 200，total=148）**：`37(7) Jul=40`、`37(8) Aug=39`、`37(9) Sep=69`。全部带期号，无 Just Accepted 空档。
- **OpenAlex**：200，但 **count 仅 14** —— OpenAlex 对 IEEE 期刊覆盖明显偏低，**窗口内计数应以 Crossref 为准**。
- **官方 URL**：`https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385`（本次实测 **418 限流**，punumber 值本身 **未经二次确认**）
- **检索路由**：DOI 前缀 `10.1109`；Crossref/OpenAlex 同上替换 ISSN；DBLP `journals/tnnls` 被拦截
- **arXiv**：**通常有**。

### IEEE TBD
- **⚠️ CCF 更正**：任务前提「TBD not ranked」**不成立**。CCF 官网「交叉/综合/新兴」C 类 #5 = `TBD | IEEE Transactions on Big Data | IEEE | https://dblp.uni-trier.de/db/journals/tbd/`，即 **CCF C 类**（该页全文检索 `Transactions on Big Data` 命中 1 次，其余 9 个学科页 0 命中）。
- **⚠️ ISSN 更正**：任务前提「电子 ISSN 2372-2096」**不准确**。ISSN Portal 实测：
  - `https://portal.issn.org/resource/ISSN/2332-7790` → **200**，标题 `IEEE transactions on big data (Online)`，**Medium: Online，ISSN-L: 2332-7790**
  - `https://portal.issn.org/resource/ISSN/2372-2096` → **200**，标题 `IEEE transactions on big data (CD-ROM)`，**Medium: Optical disk**，且 2332-7790 记录中明确标注 **"Incorrect ISSN (1): 2372-2096"**
  - 即：**2332-7790 = 电子版 ISSN + ISSN-L；2372-2096 = 光盘版 ISSN（已被标为不正确）**。该刊无独立印刷 ISSN。
  - Crossref：`/journals/2372-2096` → **404**（无期刊级记录）；但 `works?filter=issn:2372-2096` → **200，29 条**（该 ISSN 仅存在于条目级元数据）
- **窗口内卷期（Crossref 200，total=29）**：**全部为 `12(4) 2026-08`**。全年对照：`12(1) Feb=23`、`12(2) Apr=28`、`12(3) Jun=27`、`12(4) Aug=29`、`12(5) Oct=36`、无卷期号（Early Access）=36。**12(5) 封面期为 2026-10，落在窗口外。**
- **OpenAlex 差异**：按上线日期过滤时出现 `12(5) 2026-06-15/18` 等 14 条 —— 已在 6 月上线、归入 10 月期。**Crossref 期次日期与 OpenAlex 上线日期语义不同，混用会得出错误的卷期区间。**
- **官方 URL**：`https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6687317`（HTTP 200 实测）
- **检索路由**：DOI 前缀 `10.1109`，实测 `https://doi.org/10.1109/TBDATA.2026.3702367` → 302 → `https://ieeexplore.ieee.org/document/11557190/`
- **arXiv**：**混合**。该刊含大量系统/应用类工作，预印本比例低于 TKDE/TNNLS，**具体比例未经核实**。

### IEEE Big Data 会议（辨析）
- **"IEEE Big Data" 不是期刊**；期刊是 **IEEE Transactions on Big Data (TBD)**，已如上。
- **IEEE BigData 2026 会议**：**2026-12-14 → 12-17**，Sheraton Phoenix Downtown, Phoenix, AZ, USA。截稿：全文 **2026-08-21**，录用通知 **2026-10-24**，定稿 **2026-11-14**。来源 `https://bigdataieee.org/BigData2026`、`https://bigdataieee.org/BigData2026/calls/papers`
- **对本窗口的影响**：会议**尚无任何论文在本窗口内正式出版**（通知 10-24，会议 12 月），窗口内检索该会议只能得到 0 条正式论文。
- **CCF 等级**：`IEEE BigData` 会议 = **CCF C 类会议**（交叉/综合/新兴 C 类 #3）。

## 3. IEEE Xplore API（单独核实）
- **官方文档 URL（实测 200）**：`https://developer.ieee.org/docs/read/Metadata_API_details`（返回真实内容："Search Parameters / Parameter / Description…"）
- **是否需要 Key：需要。** `https://developer.ieee.org/API_Terms_of_Use2`："To gain access to one or more IEEE Xplore APIs, you must request and obtain a user account by registering… IEEE will review your application and, if approved, will assign one or more unique API keys（'API Key'）"
- **全文接口另需机构订阅 token**：Open Access API / Full-Text Access API 需 API key；面向订阅内容的 Full-Text 接口还需 **机构认证 token**。
- **结论（诚实陈述）**：**Xplore 没有免鉴权公开 API**，与 Crossref/OpenAlex 不同。无 Key 时的可行替代 = Crossref + OpenAlex（均免鉴权、实测可用），DOI 经 `https://doi.org/` 解析到 Xplore 文章页。

## 4. 机器可读检索路由（汇总，逐条实测）
```
# Crossref（免鉴权，全部 200 OK）
https://api.crossref.org/journals/1041-4347/works?filter=from-pub-date:2026-06-12,until-pub-date:2026-09-12&rows=5&select=DOI,title,published,volume,issue
https://api.crossref.org/journals/1046-8188/works?...   # TOIS  总计 63
https://api.crossref.org/journals/2157-6904/works?...   # TIST  总计 87
https://api.crossref.org/journals/2162-237X/works?...   # TNNLS 总计 148
https://api.crossref.org/journals/2332-7790/works?...   # TBD   总计 29
# OpenAlex（免鉴权，全部 200 OK）
https://api.openalex.org/works?filter=primary_location.source.issn:<ISSN>,from_publication_date:2026-06-12,to_publication_date:2026-09-12&per-page=5
# 计数：TKDE 84 | TOIS 39 | TIST 62 | TNNLS 14 | TBD 14
# IEEE：DOI 前缀 10.1109 → https://ieeexplore.ieee.org/document/<id>/
# ACM ：DOI 前缀 10.1145 → https://dl.acm.org/doi/<DOI>
# ACM TOC：https://dl.acm.org/toc/<key>/<年>/<卷>/<期>（key = tois|tist）
# ACM Just Accepted：https://dl.acm.org/toc/<key>/justaccepted
# DBLP：journals/tkde|tois|tist|tnnls|tbd → 均返回 Anubis 挑战页（HTTP 200，内容为 "Making sure you're not a bot!"），非可用路由
```

## 5. 不确定 / 未经核实项
- TIST「曾为 2019 版 CCF B」— **未经核实**
- TNNLS 的 Xplore punumber=5962385 — **未经核实**（实测 418 限流）
- `https://dl.acm.org/toc/tist/justaccepted` 精确路径 — **未经核实**（本环境 403）
- IEEE/ACM 各刊**窗口内** arXiv 预印本**实际比例** — 无逐篇核实，仅给出领域性判断
- FT50 名单中是否含 TOIS — **未经核实**。已确认的是 **UTD24 信息系统类 = ISR / INFORMS Journal on Computing / MIS Quarterly**，TOIS **不在**其中。
- ABS/AJG 2024：TOIS **4\***、TKDE **3**、TNNLS **3**（来源为二手汇总页，**建议以 AJG 官方为准**）。TIST、TBD 的 AJG 等级 **未经核实**。

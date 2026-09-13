# PHASE6 · S10 · venue 三词表统一 + 回填 + 分域定线

> 任务卡：板 `B16`（`81b1f533-db9e-4a1f-94b7-04ac2b4431a3`）
> 生成器：`paper2skills-research/scripts/build_venue_tiers.py`（**venue 词表的唯一实现处**）
> 证据采集：`paper2skills-research/scripts/fetch_venue_sources.py`
> 机读产物：`paper2skills-research/data/venue_tier_mapping.json` · `venue_tier_backfill.json` ·
> `venue_sources/arxiv_abs.json`（89 条） · `venue_sources/crossref.json`（12 条）
> 文档落点：`paper2skills-vault/07-资源库/venue-whitelist.md` §10（**脚本生成，人不许手改**）
> 产品侧投递：`dsh-paper2skills/data/venue-tiers.json`（与规范表**逐字节相等**，J9 断言）

---

## 0. 一页结论

| 项 | 改前（`08dd624`，S10 动工前的 HEAD） | 改后（实测，现算） |
|----|--------------------------------------|--------------------|
| 卡片侧 `venue_tier` 覆盖率 | **25/146 = 17.1%** | **146/146 = 100.0%** |
| `venue_tier` 取值域 | 6 种，其中 **3 种不在白名单**（`top`4 / `workshop`2 / `demo`1） | **规范 7 档中的 5 档**（无非规范残留） |
| 卡片侧 `venue_source`（出处锚点） | 0/146 | **146/146** |
| 分域定线 | 无（只有散文） | 16 个技术域**逐域**有类别 + 双侧规则 + 独立判词 |
| 门禁 | — | `--check` **exit 0**；selftest **18/18 用例 · 变异 17/17 抓住** |

**`old_value != new_value` 的实测条数 = 128**（= 121 新赋值 + 7 重定级）。
这个数是本任务最能说明问题的读数：它由账的 `change_status` 三态直接给出，且**任何人可用
`git show 08dd624:<path>` 复算**。

---

## 1. 复核：板上的实测事实**逐条成立**

复算命令与结果（`python3 build_venue_tiers.py --check` + 独立脚本）：

| 板上原话 | 复核结果 |
|----------|----------|
| 白名单定义 7 档 | ✅ `venue-whitelist.md` §1 表实有 7 行（`UTD24/FT50/CCF-A/CCF-B/field-top/preprint/non-paper`） |
| registry 用 `second` 1 | ✅ 复算成立（`p2s-2026-0019`） |
| 卡片用 `top`4 / `workshop`2 / `demo`1 | ✅ 逐字复算成立 |
| **卡片侧覆盖率 25/146 = 17%，121 张无值** | ✅ **与 `git show 08dd624` 逐档相等**（9 preprint / 5 CCF-A / 4 CCF-B / 4 top / 2 workshop / 1 demo） |
| 三个互不兼容的词表并存 | ✅ **实为五处**，见 §2 |
| 铁律 1 | ✅ 全程只用 arXiv abs 页 + Crossref；**账里 0 条依据来自底本正文**（主控已独立复核） |

⚠️ 一处**口径订正**：板上写「registry 用 preprint 18 / CCF-B 11 / UTD24 8 / CCF-A 5 / second 1 /
non-paper 1」共 45 条 —— **这组数复算不出来**。现算的 registry（45 条记录）是
`preprint 18 / CCF-B 11 / UTD24 8 / CCF-A 5 / second 1 / non-paper 1 / 空串 1`（**多一条空串**）。
差值 1 条，来源是 `p2s-2026-0037` 的 `venue_tier` 是空串（板上那组数把它漏了）。
**登记不重判**。

---

## 2. ① 词表统一（可机械执行）

### 2.1 先说一个板上没提的事：**不是三套词表，是五处**

任务书写「三套互不兼容的词表」。实测**五处**都各自定义了 `venue_tier` 的取值域：

| # | 位置 | 它定义的取值域 | 是否在本次统一范围 |
|---|------|----------------|--------------------|
| ① | `paper2skills-vault/07-资源库/venue-whitelist.md` §1 | 7 档 | ✅ **是规范源** |
| ② | `papers_registry.json` 的 `venue_tier` | 7 档 + `second` + 空串 | ✅ 登记（**只读不写**） |
| ③ | 卡片 frontmatter 的 `venue_tier` | 7 档 + `top` / `workshop` / `demo` | ✅ 写者（本轮统一） |
| ④ | `dsh-paper2skills/lib/axis.js`（**产品侧**） | 7 档 + `unlabeled` + 遗留清单（`top`/`workshop`/`demo`，`to: null`） | ✅ 消费（投递规范表，J9 断言逐字节相等） |
| ⑤ | `paper2skills-vault/07-资源库/MasterPrompt-v2.md` frontmatter 模板 | `CCF-A\|CCF-B\|UTD24\|FT50\|preprint\|non-paper\|workshop\|demo`（**8 值，把轨道标记混进档位**） | ⚠️ **登记，本轮不改** —— 见 §2.4 |

④ 是板上明确要求先查的「产品侧有没有第四套词表」。**有**，而且它不是第二份事实源：
`axis.js` 的注释自己就写着「映射到哪一档是 S10 的决策，本模块**不替它决定**（`to: null`）」。
本轮的处置是**投递**：`--apply --sync-product` 把规范表写成
`dsh-paper2skills/data/venue-tiers.json`，`--check` 的 **J9** 断言它与规范表**逐字节相等**。
⇒ 产品侧从「自己有一份词表」变成「**消费**规范表」，两处不会再分叉。

### 2.2 落成唯一机读实现

* **唯一实现处**：`paper2skills-research/scripts/build_venue_tiers.py` 的模块级常量
  （`CANONICAL_TIERS` / `TIER_ALIAS` / `TRACK_RULES` / `REGISTRY_ALIAS` / `VENUE_NAME_TIER` /
  `UNLISTED_VENUE` / `JOURNAL_NAME_TIER`）。
* **唯一产物**：`paper2skills-research/data/venue_tier_mapping.json`（由 `render_map_bytes()` 生成）。
* 文档侧（`venue-whitelist.md` §10）与产品侧副本**都不是手写的**：
  §10 由 `render_inventory_block()` 生成、`--check` **J4 逐字比对**；
  产品副本由同一函数写、`--check` **J9 逐字节比对**。
* 白名单 §1 的**散文** 7 档表也被 `--check` **J5** 逐项断言相等（散文表与规范表不许漂）。

⇒ 「只有一处实现」不是靠约定，是靠 **J4/J5/J9 三条判据**钉住的。

### 2.3 映射规则（每一个都有理由，理由可复核）

| 来源 | 原取值 | → 规范档 | 判据类型 | 理由（可复核） |
|------|--------|----------|----------|----------------|
| 卡片 | `workshop` | `preprint` | §3「降一级」后折叠 | 主会 CCF-A→CCF-B；主会 CCF-B→**CCF-C，而 CCF-C 不在白名单 7 档** ⇒ 折叠为 `preprint`。不折叠就等于把「降一级」写成空话 |
| 卡片 | `demo` | `preprint` | 照抄已写死的裁决 | whitelist §3「降为 field-top 以下」+ **MasterPrompt-v2 R4 明文写死 `demo → preprint`**（人工裁决先例 `p2s-2026-0014` RecSys'26 Demo）。本表照抄，不另立解释 |
| 卡片 | `findings` | `preprint` | 同 `workshop` | §3 把 Findings 与 Workshop 并列「降一级」 |
| 卡片 | `short-paper` | `preprint` | §3 明文 | 「Short paper / Extended abstract / Poster → 降为 preprint」 |
| 卡片 | **`top`** | **（不定值，逐卡裁决）** | **不给默认值** | 它的语义在卡片之间**不一致**：4 张卡里既有指**轨道**（EMNLP Industry Track）的，也有指**层级**（主会）的。给一个默认映射，就是把不明语义静默当成某一档（与 F2 风险 N4 同型） |
| registry | `second` | **（不定值）** | 笔误登记 | `p2s-2026-0019` 同时记 `venue=SIGMOD` / `venue_tier=second`。**ACM SIGMOD 是 CCF-A，若 venue 真为 SIGMOD 则 tier 不可能是 `second`** —— 内部不自洽已由两处独立记录（该卡 `evidence.md` §2、卡正文「registry 存疑项」）。**本脚本不改 registry**（F2：不造第二份事实源） |
| registry | `""`（空串） | `unlabeled` | 缺失 | 等同未标注，不计入覆盖率分子 |

**`top` 的逐卡裁决与依据**（全部写进 `venue_tier_mapping.json` 的 `card_resolution`）：

| 卡 | 原值 | 新值 | 依据（**仪器读数，逐字**） |
|----|------|------|----------------------------|
| `Skill-Multi-Agent-Collaboration-Tax` | `top` | `CCF-B` | abs 页 Comments 逐字 `EMNLP 2026 Main Conference` ⇒ 声明的是**层级** |
| `Skill-Routed-Graph-Handoff` | `top` | `CCF-B` | Comments 逐字 `Accepted in EMNLP 2026` |
| `Skill-Stateful-Skill-Runtime` | `top` | `CCF-B` | Comments 逐字 `accepted at EMNLP`（⚠️ **未写年份**，弱信号，已单列低置信） |
| `Skill-Persona-Based-AB-Simulation` | `top` | `CCF-B` | Comments 逐字 `Work accepted at EMNLP 2026 Industry Track` ⇒ **轨道**语义，见下 |

### 2.4 ⚠️ 一处**保留分歧**：轨道判定必须分两类（实测缺陷 + 人工裁决）

首版把 `workshop` 与 `industry track` 混成同一个降级正则，**实测把
`Skill-Auto-Skill-Synthesis`（SIGIR 2026 Industry Track）从 CCF-A 误降到 CCF-B**，
而 registry `p2s-2026-0007` 记的是 `venue=SIGIR / tier=CCF-A` —— 误降制造了第二处账实矛盾。

修法（已固化为用例 **T25** + 反向控制）：

| 类 | 例 | 论文集是谁的 | 处置 |
|----|----|--------------|------|
| **主会自带的 track** | Industry Track / Findings / Main Conference | **主会论文集**（`10.1145/3805712.3808466` 在 Crossref 解析为 `SIGIR '26 proceedings, page 4763-4768`） | **不降级** |
| **独立征稿的 workshop** | ADKDD / AI4M / BayLearn / GOBLIN | **workshop 自己的卷**（AI4M 的 DOI 落在 Springer CCIS 15241） | §3「降一级」 |

⚠️ 但 `Skill-Persona-Based-AB-Simulation`（EMNLP 2026 Industry Track）本轮**仍判 CCF-B**，
这是**唯一的例外，由人工裁决，不是脚本默认**：白名单 §7.1 实测记 EMNLP 2026 的 workshop 名是
`GroundLM`/`NLP4PI`（**不含** Industry Track），但该卡底本自述「Industry Track」而未说明是否独立征稿
⇒ 取**保守**的 CCF-B。**两条同形案例给出不同档位，理由已逐条写在卡上**，交后续复核。

**MasterPrompt-v2 的 frontmatter 模板（⑤）本轮不改**：它把 `workshop`/`demo` 列在 `venue_tier`
的枚举里（其 R3 又说「标了轨道就不要同时声称主会层级」，自相矛盾），但它同时**载着 R4
「demo → preprint」的裁决先例** —— 改它要动裁决本身。处置：`venue-whitelist.md` §10.5 写明
「**`venue_tier` 只放 7 档，轨道信息移到 `venue_track`**，以本节为准」，并在本报告登记。

---

## 3. ② 回填 121 张卡

### 3.1 覆盖率（**下界与上界都进退出码**）

```
覆盖率 = 146/146 = 100.0%   门槛 0.90 ≤ r ≤ 1.00
改前 =  25/146 =  17.1%
```

**验收目标 ≥90% 已达成**。但**下界不是唯一判据**：本仓库实测过
`check_contracts.py` 打出「核对率 102.2% 也照样通过」（台账 #26/#27），
⇒ 上界 `r > 1.00` 与「分子 + 未标注 ≠ 总数」**同样进退出码**（J3）。

### 3.2 账：三态与 `old_value`（**这是本次返工最多的一处**）

`venue_tier_backfill.json` 的 `_meta.snapshot_ref` = **`08dd624`**（S10 动工前的 HEAD，SHA 已钉死）。

| `change_status` | 条数 | 含义 | `old_value` |
|-----------------|------|------|-------------|
| `as-is` | **18** | 动工前已有该字段，**逐字未变** | 与 `new_value` 相同 |
| `retiered` | **7** | 动工前有值，本次**改了** | `top`×4 / `workshop`×2 / `demo`×1 |
| `backfilled` | **121** | 动工前**没有**该字段，本次新赋值 | `""` |
| `no-snapshot` | **0** | 取不到动工前版本 | — |

**`old_value != new_value` 实测 = 128 = 121 + 7**（与 `git show 08dd624` 逐档复算相等）。
18 + 7 = 25 = 动工前有该字段的卡数 ⇒ 恒等式闭合。

**为什么会有这次返工**（三处同形缺陷，都得记下来）：

| # | 缺陷 | 现象 | 根因 |
|---|------|------|------|
| ① | `old_value` 取自**写盘之后**的内存 | 146 条 `old == new`，「改动数 = 0」，而实际 121 条是新赋值 | 账答不出它存在的那个问题。**由主控独立复核抓出** |
| ② | `reason` 的人读列复用判定链那句**会过期的话** | 121 行 `backfilled` 的账里 **111 行**写着「已是规范档，逐字保留」—— 为一桩**从未发生过的保留**作证 | 机读列已诚实，人读列没跟上（与台账 #18「判据是真的，报告把它翻译成了另一句话」同族）。**由主控独立复核抓出** |
| ③ | `evidence` 在**重跑时现算** | 126 行变成 `frontmatter venue_tier='non-paper'` —— 那是**结论**不是**依据** | 与 ① 完全同形，只是换了一列。**本轮自查抓出** |
| ③b | 修完 ① 后**又塌一次** | 主控把落盘提交成 `e7f74b9` 后重跑，`HEAD` 已是改动后 ⇒ **三态整片塌成 `as-is`、`changed` 归零** | 同一个缺陷长在**另一个指针**上 ⇒ 快照必须钉在**不移动的 SHA** 上 |

⇒ 三条判据固化：**J14**（`old_value` 必须与快照逐字相等、`changed` 计数 = `old≠new` 实测条数、
`evidence` 不许是盘上取值的复述）+ **T23/T23b/T27/T27b** 四组用例（含反向控制：
把「写盘后的值」当快照 ⇒ 三态必须塌成 `as-is`，**正是首版的缺陷**）。

`evidence` 现算的分布（**依据**，非复述）：

| `evidence` 首词 | 条数 | 读数形态举例 |
|-----------------|------|--------------|
| `arxiv-abs` | 89 | `Comments='Accepted at ACM SIGIR 2026 Industry Track. 18 pages…' DOI='https://doi.org/10.1145/3805712.3808466'` |
| `card-metadata` | 56 | `evidence_basis=author-practice（卡自承无论文来源 ⇒ 无 venue 可判，按白名单 §5 落 non-paper）` |
| `crossref` | 1 | `doi=10.1145/3639054 container-title='ACM Transactions on Knowledge Discovery from Data' page='1-24'` |
| **读盘回显** | **0** | — |

### 3.3 判定来源分布

| `venue_source` | 条数 | 说明 |
|----------------|------|------|
| `arxiv-abs` | 58 | abs 页 `Comments:` / `journal_ref`（**铁律 1 的首选仪器**） |
| `evidence_basis` | 48 | 卡自承 `author-practice` ⇒ `non-paper` 来源类别 |
| `EXPLICIT_RETIERS` | 20 | 逐卡人工裁决（`top` / 身份冲突 / 无 paper_id 的保守判） |
| `frontmatter-as-is` | 14 | 原本就是规范档，本次未改动 |
| `unlisted-venue` | 5 | 命中已登记的「非白名单会/刊」表 ⇒ 保守落 `preprint` |
| `TRACK_RULES` | 1 | 轨道标记折叠 |

### 3.4 无法回填的卡

**`UNRESOLVABLE` 台账空（0 张）** —— 146 张全部拿到了规范档取值。但**「拿到值」≠「证据一样硬」**，
所以另立两份台账，条数与理由都逐条在册：

* **低置信 17 条**（`LOW_CONFIDENCE`）：值拿到了，但仪器弱。逐条理由见
  `venue_tier_mapping.json` 的 `low_confidence`。典型：
  * `Skill-AGRS-属性引导评论摘要` / `Skill-MAA-行动建议生成` / `Skill-StaR-观点语句排序` —— 卡内**自承**「全文尚未入库，且未记录其 arXiv/DOI 编号」⇒ 无仪器可核，按保守口径记 `preprint`；
  * `Skill-Two-Echelon-Inventory-DRL` / `Skill-Cannibalization-Corrected-Attribution` —— 非白名单 workshop；
  * `Skill-Monodense-单品价格弹性估计` / `Skill-MAS-Consumer-Behavior-Simulation` / `Skill-Behavioral-Intent-Tree-Parsing` —— 非白名单会议（AAIML / ICEBE / BayLearn）；
  * `Skill-SQL-Agent-Access-Control` —— 见 §3.6；
  * `Skill-Open-Source-Tool-Use-Model`（Hermes 4 Technical Report）等。

* **身份错位 8 条**（`IDENTITY_MISMATCH`）：`paper_id` 指向的 arXiv 论文与卡声明的论文**不一致**。
  其中 **7 条是「卡 `paper` 字段写成 arXiv 号的自指」**（无法对齐，按号判）；
  **1 条是真错位**：`Skill-Reflexion-Self-Improvement` 的 `paper_id=2303.11366` 是
  Shinn 等的 *Reflexion*（ICLR 2023，CCF-A），而卡 `paper` 字段写的是 Cassano 等的**同名另一篇**
  —— **卡正文自己写明了这一点**。处置：tier 按**实际取到的那个号**的元数据判，但既不能算在
  另一篇头上，也不能按 CCF-A 计 ⇒ **记为 `preprint`** 并登记，等人工确认讲的是哪一篇。

* **卡–论文宽度不匹配**（`SCOPE_MISMATCH`，本轮实为 0 条）：卡的**综合卡**形态
  （卡名是技术名而非论文标题）会让身份无法对齐。实测 7 张这样的卡，但**全部经卡正文
  「论文来源: <标题> / arXiv ID: <号>」逐字声明核对后确认对齐** ⇒ 未登记为错位。
  ⚠️ 这条判据本身是实测撞出来的：首版只认 `paper` 字段与 slug，把 13 张**身份明确**的卡
  判成「无法对齐」—— **正是漏洞 #11「判据只认一种字段名」的同型**。现按四源并列判定
  （`paper` 字段 → slug → 卡正文引的标题 → 卡 `title` 字段），任一命中即算对齐。

### 3.5 数据源、礼貌与可复跑

* **arXiv**：`https://arxiv.org/abs/<id>` 的 `Comments:` / `journal_ref` / `DOI`。
  ⚠️ **`export.arxiv.org/api/query` 在本机实测返回 `Rate exceeded.`（HTTP 200 + 13 字节文本）**
  —— 是 **HTTP 200 的失败**，脚本若不检查正文会把这 13 字节当数据（与 DBLP 的 Anubis 挑战页同型）。
  故走 abs 页 HTML，并对 `len(body) > 5000` 做硬校验。
* **Crossref**：`api.crossref.org/works/<doi>` 的 `container-title` / `event` / `page` /
  `published-online` / `published-print`。
* 每次请求间隔 ≥1.5s；失败重试 3 次（指数退避），**不无限循环**；单条超时 30s；
  **增量写盘**（每抓一条落一次），中断后重跑只补缺的。
* 实抓：arXiv **89/89 成功（0 失败）**；Crossref **12/12 成功**。

### 3.6 ⚠️ 一处**反直觉的实测**：铁律 1 反过来救回了 SIGMOD

`Skill-SQL-Agent-Access-Control`（`p2s-2026-0019` / arXiv `2607.22115`）的卡正文与
`evidence.md` §2 都据「**全文逐字检索 `SIGMOD`，全部命中都落在参考文献区**」判定
「论文本身没有 venue 证据」⇒ 卡片记 `venue_tier: preprint`。

本轮用**铁律 1 规定的仪器**复核，arXiv abs 页 `Comments` 逐字是：

```
Accepted at ACM SIGMOD 2027
```

⇒ **有正式录用声明**，且 SIGMOD = CCF-A。卡侧那句「论文本身没有 venue 证据」**用错了仪器**
（与台账 A3/A1 两次误判同型）。

**但本轮不改卡上的值**，理由三条（都登记在册）：① 它不是「回填 121 张」的成员（动工前已有值）；
② registry 同一篇记 `venue_tier: second`，改卡会让 registry 成为**唯一**的错误处，
制造第二处账实矛盾 —— 而本脚本**不许改 registry**；③ 纪律 6「登记不重判」。
⇒ **登记为低置信 + 在报告里点名**，工单交人工先修 registry 的 `second`，再改卡。

---

## 4. ③ 分域定线（**可判据**，不是形容词）

### 4.1 判据的三件套（缺一不可，J8 全部进退出码）

**① 判词（不引用下表结论，可独立复核）**

> 一个技术域属「商科实证类」当且仅当该域的论文来源主流是 UTD24/FT50 商科期刊，
> **或**该域的方法必须靠观测数据做识别（实验 / 准实验 / 结构模型），
> 因而未评审的预印本不能作为业务结论的依据。否则属「方法论/工程类」。

**② 类别 → 是否接受 `preprint`（双边写法：接受/不接受各写一条，判据只写单边等于没有判据）**

| 类别 | 是否接受 `preprint` | 取值域（正面） | 禁止（反面） | 为什么 |
|------|---------------------|----------------|--------------|--------|
| 方法论/工程类 | **✅ 接受** | `{UTD24, FT50, CCF-A, CCF-B, field-top, preprint}` | `non-paper`（无来源内容不得伪装成方法论文） | arXiv 生态原生：主战场本就是预印本+顶会，正式见刊常滞后 1–2 年；拒绝 preprint 会让本域整体无来源可用 |
| 商科实证类 | **❌ 不接受** | `{UTD24, FT50, field-top}` | `preprint`（含 arXiv 正式预印本） | 预印本生态弱，顶刊是唯一来源；实证结论的效力来自同行评审与真实数据 |

**③ 域 → 类别表（域线表 17 条：卡里实际出现的 16 个技术域**逐域一条**，
+ 1 个卡里还没有但已登记的技术域。未登记的技术域 ⇒ J8 判红，不给默认类别）**

> 实测：域线表 17 条 / 卡里出现 16 个域，**差集 = `11-AI人文` 在表里但卡里还没有**
> （该域在 PRODUCT 侧有卡，精选线暂无）；反方向差集为空（**卡里出现的域全部有归属**）。
> 这一条刻意留着：它就是「未登记即判红」这条判据的**正向对照** —— 表是允许先于卡存在的。

| 技术域 | 类别 | 接受 preprint | 判定依据 |
|--------|------|---------------|----------|
| `00-电商Agent` | 方法论/工程类 | ✅ | 会话推荐与目录补全，均为工程系统；RecSys→arXiv 链路完整 |
| `01-因果推断` | 方法论/工程类 | ✅ | 方法工具箱（PC/DiD/IV/中介/uplift），arXiv 与顶会双轨 |
| `02-A_B实验` | 方法论/工程类 | ✅ | 实验设计与样本量计算，CIKM/arXiv 主场 |
| `03-时间序列` | 方法论/工程类 | ✅ | 预测模型（Prophet/TFT/异常检测），ML 会议主场 |
| `04-供应链` | 方法论/工程类 | ✅ | 库存与履约优化，OR/MS 与 arXiv 双轨 |
| `05-推荐系统` | 方法论/工程类 | ✅ | 召回/排序/冷启动，RecSys/SIGIR/KDD 主场 |
| `06-增长模型` | 方法论/工程类 | ✅ | 流失预测与 LTV 建模属 ML 方法；UTD24 的 CLV 文献是其背书而非来源 |
| `07-NLP-VOC` | 方法论/工程类 | ✅ | 情感分析/观点抽取/多语 NER，ACL/EMNLP/arXiv 主场 |
| `08-知识图谱` | 方法论/工程类 | ✅ | 异质图/双曲嵌入/知识补全，ML 会议主场 |
| `09-DataAgent-LLM` | 方法论/工程类 | ✅ | Text-to-SQL 与数据分析 Agent，系统与基准类，arXiv 主场 |
| `10-MAS` | 方法论/工程类 | ✅ | 多智能体协作与编排，AAMAS/arXiv 主场 |
| `11-AI人文` | 方法论/工程类 | ✅ | LoRA/持续学习/提示微调，ML 方法侧 |
| `12-ML基础` | 方法论/工程类 | ✅ | 特征工程与模型评估，ML 方法侧 |
| `13-广告分析` | **商科实证类** | ❌ | 归因与 ROAS 的权威结论出自 Marketing Science/JM（UTD24）；且 registry 已把 UTD24 论文（`p2s-2026-0038` Management Science）投放到本域 |
| `14-用户分析` | **商科实证类** | ❌ | 漏斗与留存分析的权威结论出自 JMR/Marketing Science（UTD24）—— 白名单 §2.1 的 UTD24 清单含这两刊 |
| `15-营销投放分析` | **商科实证类** | ❌ | MMM 与促销效果的权威来源是 Marketing Science/JM/JMR（UTD24），预印本生态弱 |

J8 另有两条**正面义务**判据：商科实证类**必须点名权威来源刊**（「重要」「相关」这类形容词直接打红）；
每个类别必须**四键齐全**（`preprint_accepted` / `why` / `requires` / `forbids`）。

### 4.2 ⚠️ 分域线下的既有卡：**3 张，登记不重判**

入卡早于本线的卡，其 venue 取值**不动**（纪律 6），逐张列为 `line_violations` 一等输出：

| 卡 | 域 | 现值 | 处置 |
|----|----|------|------|
| `Skill-Cannibalization-Corrected-Attribution` | 13-广告分析 | `preprint`（原值也是 preprint） | 登记；本卡是 ADKDD workshop 短文（白名单 §3 正落此档） |
| `Skill-Causal-Budget-Allocation` | 13-广告分析 | `preprint` | 登记 |
| `Skill-Incrementality-Measurement` | 14-用户分析 | `preprint` | 登记（方法论文，arXiv 来源） |

⇒ 这 3 张说明一个真问题：**商科实证域当前的卡全部来自 arXiv**。这直接引出下一条。

### 4.3 ⚠️⚠️ 实质发现：**精选线里 UTD24/FT50 = 0，而 registry 那 8 条一条卡都没出**

板上的验收问「回填后 `venue_tier` 覆盖率」，但**另一件更值得测的事**是：
`by_new_value` 里 **UTD24 = 0、FT50 = 0**，而 registry 那边记着 UTD24 8 条。
这是 (a) 精选线里确实没有、还是 (b) 回填把它们折叠进了别档？

**用 evidence 列实测 ⇒ 是 (a)，且原因比 (a) 更具体**：

```
registry 8 条 UTD24 记录的 outputs.skill_card：
  p2s-2026-0038 Management Science           → paper2skills-vault/13-广告分析/Skill-<方法名>.md    卡不存在
  p2s-2026-0039 Marketing Science            → paper2skills-vault/15-营销投放分析/Skill-<方法名>.md  卡不存在
  p2s-2026-0040 Information Systems Research → .../15-营销投放分析/Skill-<方法名>.md              卡不存在
  p2s-2026-0041..0043 Management Science     → .../04-供应链、02-A_B实验/Skill-<方法名>.md        卡不存在
  p2s-2026-0044 Information Systems Research → .../02-A_B实验/Skill-<方法名>.md                   卡不存在
  p2s-2026-0045 Management Science           → .../05-推荐系统/Skill-<方法名>.md                  卡不存在
  8/8 全部是 `Skill-<方法名>.md` 占位；∩ 卡片 paper_id = ∅；卡侧 venue 字段无一处出现商科顶刊
```

**结论**：精选线 146 张里 UTD24/FT50 **真的是 0**，而且**不是回填折叠造成的**
（回填的来源分布里根本没有 UTD24 这一档：`arxiv-abs / evidence_basis / EXPLICIT_RETIERS /
frontmatter-as-is / unlisted-venue / TRACK_RULES`）。**是那 8 篇顶刊论文还没出卡。**

⇒ 这是**关于精选线本身的实质发现**，与分域定线直接相关，写进报告：

| 含义 | 说明 |
|------|------|
| 商科实证类在精选线里**缺席** | 3 个商科实证域（13/14/15）共 10 张卡，**全部来自 arXiv**，0 张来自 UTD24/FT50 |
| 但登记层**已经在投放** | registry 往 **5 个域**投了 8 条 UTD24：13×(1) / 15×(2) / 04×(2) / 02×(2) / 05×(1) |
| 对分域线的推论 | 分域线是**面向后续入卡**的：13/14/15 现在的不合规卡是历史存量，**下一步入卡必须带 UTD24/FT50 或 field-top 来源**；而 02/04/05 的 UTD24 投放要与「本域已定方法论/工程类」这条线对齐 —— 要么把它们归入商科实证类，要么承认本域是**双轨** |
| 对 Q4「入卡取弱门槛」的含义 | 见 §4.4 |

**这条口径是脚本的一等输出**（`registry_top_journal_domains()`），每次 `--check` 都打印，
不是手写进报告的 —— 否则下一次入库时它就会腐烂。

### 4.4 顶刊不是新方法来源（板上的已知坑，写进定线）

**实测事实**：79% 的顶刊文章 DOI 年份段比发表年份早 ≥2 年
（`10.1287/mnsc.2022.02462` 发表于 2026-07-09，DOI 段却是 2022）。
Crossref 的 `published-online` 只反映**上线时间**，不是**方法产生时间**。

| 域类别 | 顶刊在这里的角色 | **对 Q4「入卡取弱门槛」的含义** |
|--------|------------------|-----------------------------------|
| 商科实证类 | **既有方法的权威背书**，不是新方法来源 | 弱门槛拦的应是**识别可信度**（数据与设计），**不是方法新颖度** —— 顶刊卡进来时方法可能已 2–4 年 |
| 方法论/工程类 | 次要来源（arXiv/顶会才是主场） | 弱门槛照常按 L3 ∈ A/B 判；顶刊卡若方法陈旧，走 MasterPrompt-v2 的 `stale_method` 规则 |

**可执行推论**：`venue_tier ∈ {UTD24, FT50}` 的卡，**不得**因「顶刊新发表」被当作新方法入选；
必须同时给出预印本首版时间（方法年龄）或显式标 `stale_method`。
（本轮精选线里这类卡为 0，故该判据**尚无对象** —— 它是**面向后续入卡**的，已在 §10.4 成文。）

---

## 5. ④ 门禁脚本

`paper2skills-research/scripts/build_venue_tiers.py`

| 退出码 | 含义 |
|--------|------|
| **0** | 全过 |
| **1** | 判红 |
| **2** | **输入没拿到（≠ 通过）** —— arXiv/Crossref 元数据缺档、vault 里 0 张 Skill 卡 |
| **3** | **内部错误（≠ 判红）** —— 本轮实测抓到过一次（`UnboundLocalError`），当场按 3 退出而不是静默 |

`--check` / `--apply`（写回卡片 frontmatter + 文档块 + 映射表；配 `--sync-product`）/
`--selftest` / `--json-out` / `--no-mutations`。

### 5.1 判据 J1–J15

| # | 判据 | 变异样本 |
|---|------|----------|
| J1 | 卡片侧取值 ⊆ 规范 7 档 ∪ {unlabeled} | T1 / T1b |
| J2 | 覆盖率**下界** ≥ 90% | T2 / T2b |
| J3 | 覆盖率**上界** ≤ 100%；分子分母**各自数出来**且等式自洽 | T3 / T3b / T3c |
| J4 | `venue-whitelist.md` §10 机读块与生成器**逐字相等** | T4 / T4b / T4c |
| J5 | §1 散文 7 档表 == 规范 7 档 | T5 / T5b / T5c |
| J6 | 同一卡 frontmatter 字段不重复 | T6 / T6b |
| J7 | 三套词表每个非规范取值都有映射规则；registry 表外取值逐条有理由 | T7 / T7b |
| J8 | 域线：域有归属、类别双侧、依据不是形容词、商科实证类点名来源刊 | T8 / T8b / T8c / T8d / T8e |
| J9 | 产品侧副本与规范表**逐字节相等** | T9 / T9b / T9c |
| J10 | 有 `paper_id` 的卡必须**判定完成**（无 `needs-resolution` / 无元数据缺档 / 无未登记身份冲突） | T10 / T10b / T10c |
| J11 | 台账完备（双向）：登记为不可回填的必须真的不可回填；不可回填的必须在册 | T11 / T11b / T11c / T11d |
| J12 | **只增不改**：原值非空不得改写（除显式重定级），且理由必须点名仪器 | T12 / T12b / T12c |
| J13 | 身份三态：对齐 / 无法对齐（须登记）/ 冲突（须登记原因），**逐条必须有处置** | T13 / T13b |
| J14 | 账的溯源强度：`old_value` 取自快照、三态与快照吻合、`changed` 计数 = `old≠new`、**`evidence` 必须是依据不是复述** | T23 / T23b / T23c / T27 / T27b |
| J15 | 映射表内**不得有重复键**（Python dict 字面量重复键是静默的，取最后一个） | T24 |

### 5.2 自证：**18 组用例 · 变异 17/17 抓住**

**变异测试是真的在做变异**：把某条判据在**脚本副本**里改坏（`return []` / `return {}` /
`return ("as-is","")` / `return True`），跑**副本的自己**的 `--selftest`，
改坏后自检**必须变红**。首版这里是个空壳（11 条里 8 条漏网），
修法是加返回类型注解 + 子进程跑真 CLI。

**变异把 4 处真缺陷逼了出来**（这是本文件最该留的部分）：

| # | 缺陷 | 判据 |
|---|------|------|
| ① | 写字段会写出**同名字段两份**（`venue` 行在前、`venue_tier` 行在后时） | **T21**（4 种子例）—— 实测污染 4 张卡，J6 抓到 |
| ② | `demo` 只在**读时**折叠，盘上永远留着非规范值 | **T22** —— 实测第 1 轮漏掉 1 张，靠人肉 grep 才发现 |
| ③ | 值合规但**缺出处锚点**时被判「不用写」 | **T26** —— 实测 18 张卡的 `venue_source` 缺失而门禁全绿 |
| ④ | 判据**次序**：`EXPLICIT_RETIERS` 排在「值对就不动」之后 ⇒ 显式重定级永远不落盘 | **T26 当场报红** |

📌 新纪律（本仓库第 4 次同型）：**新门禁的第一个自检用例应当是「把门禁自己改坏，看它报不报」**。

---

## 6. 四道门禁：改前 / 改后（**逐项比对**）

| 门禁 | 改前 | 改后 | 差 |
|------|------|------|----|
| `verify_skill_code.py --all --level 5`（K1） | 101 单元：PASS **62** / ENV_BLOCKED 7 / ORPHAN 0 / MIGRATED 9 / FAIL **23**；执行率 **61.4%** | 101 单元：PASS **62** / ENV_BLOCKED **8** / ORPHAN 0 / MIGRATED 9 / FAIL **22**；执行率 **61.4%** | **FAIL 23→22、ENV_BLOCKED 7→8**（同一个单元在两次运行间从 FAIL 变 ENV_BLOCKED，属**环境归因波动**，非卡片变化） |
| `quote_check.py --all`（G2b） | VERBATIM **90** / NO_FULLTEXT 2；伪造 **0** 近似 **0** 拼接 **0** | VERBATIM **90** / NO_FULLTEXT 2；伪造 **0** 近似 **0** 拼接 **0** | **逐项不变** |
| `gate_check.py --all --k1`（K2） | G1 **62/146 = 42.5%**（红 68 黄 7）· G2 **16/146 = 16.3%**（红 1523 黄 692，可核验分母 98）· G3 **67/146 = 45.9%**（红 93 黄 171） | G1 **62/146 = 42.5%**（红 **67** 黄 **8**）· G2 **16/146 = 16.3%**（红 1523 黄 692）· G3 **67/146 = 45.9%**（红 **92** 黄 **171**） | **通过数全不变**；G1 红 68→67、G3 红 93→92（各 −1，黄灯相应 +1） |
| `repo_health.py` | C1 0 组重复 · **C2 77**（缺 `venue` 65 / `venue_tier` 未列）· C3 48 · C4 0 · C5 72 · C6 同左 · C7 0 · C8 58 · **C10 CRITICAL** | C1 同 · **C2 77**（缺 `paper_id` 8 / `paper` 29 / `venue` 65 / **`venue_tier` 0** / `evidence_grade` 77）· C3 **54** · C4 0 · C5 72 · C6 同左 · C7 0 · C8 58 · **C10 CRITICAL** | C2 总数不变但**多出 `venue_tier 缺 0` 这一行**（此前该字段根本不在分解里）—— 即 S10 的成果现在**被门禁自己记着** |

**纪律 3 的兑现**：`gate_check.py` 会先剥 frontmatter 再扫数字（漏洞 #12 的修复），
**新增字段没有让任何卡的红灯数上升** —— G1 与 G3 各降 1 条，逐项核对无一张卡是**因新增字段**变红。
那 2 条降幅来自 K1 的 ENV_BLOCKED 波动（同一单元 `FAIL → ENV_BLOCKED`，归因环境），
**不是卡片变化**，如实登记。

### 6.1 并发写入者（S13 的 `l3_*`）**没有被抹掉**

主控预警：146 张卡的 frontmatter 有**两个写入者**（S10 写 `venue_tier`，S13 写 `l3_*`）。
本脚本按四条纪律做（**这是可复跑的，不是一次性的**）：

1. **逐卡 read-modify-write**：每张卡在**写入前一刻**重新从磁盘读，不用批读时的内存副本；
2. 只做「插入 `venue_tier`/`venue_source`」或「替换 `venue_tier` 行」，**其余行逐字节带过**；
3. 每 20 张**重扫目录**，报告是否出现了别人的字段；
4. 写盘后**立刻复读该卡**，断言 `venue_tier` 在、别人的字段也在，任一丢失即**抛错**
   （`RuntimeError`，不许静默重写）。

**实测（现算，非计划值）**：

```
venue_tier  覆盖率 : 146/146
venue_source 覆盖率 : 146/146
l1_/l2_/l3_ 覆盖率 : 146/146   {l1_id, l1_plane, l2_id, l2_domain, l3_id, l3_business, l3_all, l1_l2_l3} 各 146
两字段共存          : 146/146
重复字段            : 0
```

---

## 7. 未做项 / 登记不重判 / 已知残留

| # | 项 | 为什么不做 |
|---|----|-----------|
| 1 | **不改 `papers_registry.json`** | F2 硬约束「不造第二份事实源」。`second` 这条笔误只登记（§2.3 / §3.6），并在报告里点名 |
| 2 | **不改 `MasterPrompt-v2.md`** 的 8 值枚举 | 它载着 R4「`demo → preprint`」的人工裁决先例，改它要动裁决本身。已在 `venue-whitelist.md` §10.5 写明冲突与本轮取哪一边 |
| 3 | **不改 `capability-graph.json` 与 `build_capability_graph.py`** | 纪律 5（正被其他任务依赖） |
| 4 | **不重判 3 张违反域线的既有卡** | 纪律 6 |
| 5 | **不给 `Skill-SQL-Agent-Access-Control` 改判 CCF-A** | §3.6 三条理由；改为登记低置信 + 点名人工作业（先修 registry 的 `second`） |
| 6 | **7 张 `paper` 字段自指的卡未补全标题** | 只登记（`IDENTITY_MISMATCH`）；补标题属卡片内容修改，超出本任务「只改 `venue_tier`（+ 新增字段）」的范围 |
| 7 | **`venue_source` 是新增字段** | 为让回填**可审计**所必需（缺它则「按什么判的」无从复核）。这是任务书允许的「新增字段，若必要」；它**不参与**任何门禁判定，只被 J10/J14 读 |
| 8 | **顶刊时效判据（§4.4）尚无对象** | 精选线 UTD24/FT50 = 0（§4.3）。判据已写进 §10.4，面向后续入卡 |
| 9 | **C10（`verified_by` 时间锚点）仍 CRITICAL** | 与本任务无关（7 张声明与实测矛盾），本轮未动 |
| 10 | **D3 正文重复（S1 的独立欠账）** | 与本任务无关 |

### 7.1 与文档不符的实测（登记不重判）

| 文档说 | 实测 |
|--------|------|
| 板上「registry 用 preprint 18 / CCF-B 11 / UTD24 8 / CCF-A 5 / second 1 / non-paper 1」= 45 | 复算为 **46 个非空计数** —— 多一条**空串**（`p2s-2026-0037`） |
| 板上「三套互不兼容的词表」 | 实为 **5 处**（+ 产品侧 `axis.js` + `MasterPrompt-v2` 的模板枚举） |
| 板上「多张卡的 ⑧ 段自承『卡页记的 arXiv 号指向无关论文』」 | 本仓 146 张里实测 **1 条真错位**（`Skill-Reflexion-Self-Improvement`，且卡自己写明了）；另 7 条是「`paper` 字段写成 arXiv 号自指」⇒ 无法对齐，非「指向无关论文」。**板的原话与实测不完全同形，如实登记** |
| 板上「`2305.12345` 被 19 张卡共用；`2106.09876` 被 4 张共用」 | 本仓 146 张里**最大共用数是 3**（`2310.08560`）；`2305.12345` / `2106.09876` **在本仓不存在**。⇒ 那组数出自另一批语料，**本仓不适用**，如实登记 |
| CLAUDE.md 记「K1 执行率 62.0%」 | 现测 **61.4%**（101 单元 PASS 62）。差异属既有口径，本轮未动 |

---

## 8. 复跑命令

```bash
# 0) 证据采集（仅在需要补新论文时跑；增量、可中断续跑）
python3 paper2skills-research/scripts/fetch_venue_sources.py --ids-from-cards

# 1) 自证：18 组用例 + 变异矩阵（变异会起子进程，约 1–2 分钟）
python3 paper2skills-research/scripts/build_venue_tiers.py --selftest

# 2) 判据（只读；退出码 0/1/2/3）
python3 paper2skills-research/scripts/build_venue_tiers.py --check --json-out /tmp/vt.json

# 3) 落盘（写卡片 frontmatter + 文档块 + 映射表 + 产品侧副本）
python3 paper2skills-research/scripts/build_venue_tiers.py --apply --sync-product

# 4) 四道门禁（改卡片后必跑，分开报）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --all --level 5 --timeout 30 \
  --json-out paper2skills-research/data/verification/k1_l5.json
python3 paper2skills-skills/paper-审核/scripts/quote_check.py  --all --json-out /tmp/quote.json --quiet
python3 paper2skills-skills/paper-审核/scripts/gate_check.py   --all \
  --k1 paper2skills-research/data/verification/k1_l5.json --outdir paper2skills-vault/07-资源库/gates
python3 paper2skills-skills/paper-维护/scripts/repo_health.py  --json-out /tmp/repo_health.json

# 5) 独立复算账（**不读账，直接比 git**）
git show 08dd624:paper2skills-vault/03-时间序列/Skill-Prophet-Forecasting.md | grep venue_tier
python3 -c "import json;d=json.load(open('paper2skills-research/data/venue_tier_backfill.json'));\
print(d['change_summary']['by_change_status'], d['change_summary']['changed'])"
```

---

## 9. 交接

| 给谁 | 什么 |
|------|------|
| **S13**（`l3_*`） | 两个写入者已实测共存 **146/146**，互不抹掉。S13 若再落盘，字段插入点现在会落在 `l3_*` 之后 —— 逻辑无冲突，但**请保留逐卡 read-modify-write** |
| **S12 / S5** | `venue_source` 是新增字段；产品侧 `data/venue-tiers.json` 是**投递副本**，请勿手改（J9 会打红） |
| **人工（venue 工单）** | ① registry `p2s-2026-0019` 的 `venue_tier: second` → 应为 `CCF-A`（abs 页 Comments 逐字 `Accepted at ACM SIGMOD 2027`），修完再改该卡；② `Skill-Persona-Based-AB-Simulation` 的 Industry Track 例外裁决；③ 7 张 `paper` 字段自指的卡补全论文标题 |
| **后续入卡（T-D2）** | 13/14/15 三域**不接受 preprint**，且这三域现在 0 张 UTD24/FT50 卡而 registry 已投放 3 条 —— 下一步入卡必须带 UTD24/FT50 或 `field-top` 来源 |

# PHASE6 S5 / B11 · p2s 换底 + 两语料收编 + `references/` 挂全文

> 执行者：子代理 B11 · 2026-09-13
> 工作目录：`paper_to_skills`（python/报告/门禁）与 `magpie-horch`（产品侧插件）。**两仓均未 commit**，留给主控统一提交。
> 三个脚本：`paper2skills-research/scripts/check_card_identity.py`（同物判定器）、`paper2skills-research/scripts/rebase_p2s_cards.py`（换底器）、`magpie-horch/packages/capabilities/dsh-paper2skills/scripts/append-selected-line.mjs`（精选线接入分类底本）

---

## 0. 一句话结论

断口接上了：**已安装的卡里含逐字引文的，从 0 张变成 145 张**（内联 **1,346 条**逐字引文，占完整卡的 **2,109 条中的 63.8%**；未能内联的部分**逐条声明、可复算**，全文在同目录 `references/full-card.md`）。
交付 **93 张换底 + 52 张新增 + 1,245 张标 `preview`**（计划 53 张中 **1 张被可见阻断**，见 §6.2）。
`SKILL.md ≤ 12 KB` 硬门禁在**已装卡**上实测量过：**1,390/1,390 全过**。
**代价必须一起说清**：换底让两条上游门禁失效 —— 一条我修好了（L3a），**一条我修不了（L3c，52 条 J4）**，因为它的豁免按设计是**人的决定、脚本不得代做**（§7）。

---

## 1. 复算任务卡的事实：**三处不一致，以本报告实测为准**

| 事实 | 任务卡写的 | **我实测** | 差异说明 |
|---|---|---|---|
| p2s 卡数 | 1338 | **1338** ✅ | 一致 |
| 每张都有 `p2s_card_id` | 1338/1338 | **1338/1338** ✅ | 一致 |
| 命中 vault 精选线 | 93 | **93** ✅ | 一致 |
| vault 独有 | 53 | **53** ✅ | 一致（换底后 1 张被阻断，见 §6.2） |
| SKILL.md 体积 | min 3697 / p50 9399 / p90 10983 / max 12211，超 12KB **0** | min 3697 / p50 **9400** / p90 **10986** / max **12211**，超 12KB **0** | p50/p90 差 1–3 字节（口径：我按 Python `len(s.encode())`；任务卡疑为 `stat -f%z`）。结论不变 |
| **vault 卡全文** | **约 43 KB** | **p50 15,155 / p90 42,841 / max 67,355 / 均值 20,436 字节** | ❌ **任务卡的「43 KB」是 p90，不是典型值**。真实典型值是 **15 KB**，最小值仅 **3,370 字节**（146 张里 **41 张 ≤ 12 KB**，本来装得进 SKILL.md）。这个差异直接影响「全文必须走 `references/`」的强度：它是**大多数卡成立、41 张不必要**的约束 |
| 已安装卡含逐字引文 | 0 条 | **0 条** ✅（换底前）；换底后 **145 张 / 1,346 条** | 一致 |

**复算命令**（`stat` 与 Python 两种口径都跑过，结论一致）：

```bash
cd /Users/lute/project/paper_to_skills
python3 - <<'PY'
from pathlib import Path
v=Path("paper2skills-vault")
s=sorted(p.stat().st_size for p in v.rglob("Skill-*.md") if ".git" not in p.parts and "node_modules" not in p.parts)
print(len(s), s[0], s[len(s)//2], s[int(len(s)*.9)], s[-1], round(sum(s)/len(s)))
PY
```

---

## 2. 【任务要求的第一件事】重画两边字段对照表

任务卡说「分类字段只在旧 p2s 卡」——**这句话已经不成立**。实测（换底后现场）：

| 字段 | p2s（1390） | vault（146） | 判读 |
|---|---|---|---|
| `l1_id` / `l1_plane` / `l2_id` / `l2_domain` / `l3_id` / `l3_business` / `l3_all` / `l1_l2_l3` | **1390** | **146** | **两边都有，且都是 100%** —— **没有任何分类字段是单边的** |
| `title` | 1390 | 146 | 两边都有（值常不同：p2s 用产品标题，vault 用卡页标题） |
| `name` / `description` | 1390/1390、1390/6 | 6 / 6 | 名义共有，实际 p2s 独有 |
| **只有 p2s** | `p2s_card_id` 1390 · `p2s_src_domain` 1390 · `description` 1390 · `enabled` 1390 · `disable-model-invocation` 1390 · `user-invocable` 1390 · `quality_tier` 1390 · `user_summary` 1338 · `whenToUse` 1338 · `user_try` 1326 · `workflow` 1263 · 透传 `p2s_venue` 33 / `p2s_venue_tier` 145 / `p2s_evidence_grade` 21 / `p2s_paper_id` 90（并发会话落的） · **本次新增 `rebase_*` 13 个键 × 145** | — | 全是**产品侧的 UI / 路由 / 换底台账**字段 |
| **只有 vault** | — | `venue_tier` 146 · `venue_source` 146 · `module` 146 · `updated` 143 · `created` 141 · `evidence_basis` 127 · `source` 120 · `status` 96 · `topic` 96 · `paper_id` 90 · `owner` 72 · `paper` 69 · `doc_type` 48 · `venue` 33 · `related` 27 · `verified_by`/`verified_at` 25 · `evidence_grade` 21 · `supersedes` 18 · `version` 4 · `anthology_id` 1 | 全是**溯源 / 证据 / 策展元数据** |

### 这个答案如何改变了合并规则

1. **分类字段不需要「继承 + 合并」** —— 两边本来就是同一套 8 个字段、同源同值（F5 的零漂移门禁逐字断言过 93 张）。换底时我把旧 p2s 卡的这 8 个字段**原样带过**，没有做任何合并或选择。
2. **真正需要搬的是「只有 vault 有」的那 20 个字段** —— 而其中 4 个（`venue` / `venue_tier` / `evidence_grade` / `paper_id`）**已经有一条既有的透传通道**（§3.3），所以我按**同一套命名**写入，而不是另立一套。
3. **`quality_tier` 两边都缺**（换底前 p2s=0、vault=0，实测）⇒ 它是本次**新造**的字段，属于「只有 p2s 有」那一列。

---

## 3. 交付与设计

### 3.1 换底后的卡形态（`staging/<L2域>/<slug>/`）

```
SKILL.md                          ≤ 12 KB（staging 预算 12,240 = 12,288 − 48 装机余量）
  frontmatter                     分类 8 字段（继承旧卡）+ 产品 UX 字段（继承）
                                  + quality_tier + p2s_* 溯源（与既有透传同名同值）
                                  + rebase_* 13 个台账键（新事实，带前缀防撞）
  底本声明                        完整卡路径 / sha256 / 字节 / 行数 / 引文条数
  ## 换底正文（完整卡逐字摘录）      完整卡正文的开头若干行（跳过代码围栏与引文区间）
  ## 原文引用（逐字 · …）          按完整卡顺序的**整条**引文，区间逐字节等于完整卡的一段
  <旧 S2a 合成四段>                能装下就原样带过（输入/输出契约·执行步骤·边界与不做·技能关联）
  ## ⑦ 代码节选                    指针形态 → references/implementation.py（不内联代码）
  ## ⑧ 论文来源                    **溯源从 vault 卡取**；旧预览页的出处声明原样留档
references/full-card.md           完整卡**逐字节**副本（sha256 写进 frontmatter）
references/legacy-preview-card.md  被换掉的八段预览页**逐字节**留档（93 张；可回看、可回退）
references/implementation.py       **未动**（既有先例，本次不碰）
```

**为什么 `references/full-card.md` 是逐字节副本而不是摘要**：源指纹纪律（约束⑤）。`--check` 每次重算源卡 sha256，与 frontmatter 里记录的对比，**漂移逐条报出**。

### 3.2 「全文装不进 12 KB」逼出的投影规则（这是本任务最难的一处）

12 KB 是硬门禁，而 vault 卡 p50 15 KB、max 67 KB ⇒ **投影不可避免**。规则做成机械可复算：

1. **引文一条不截断**——按完整卡顺序**整条**取（按行取会在引文中间断开，那是 G2 漏洞 #15「引文被截成残句」的形态，而残句**仍逐字存在于底本**，门禁看不出来）。
2. **换底正文**取完整卡正文开头若干行（跳过代码围栏、跳过已内联的引文区间），预算为剩余空间。复现既有 `references/implementation.py` 的形态（「卡面节选正是该文件的**开头**，逐行连续前缀」）。
3. 旧 S2a 四段能装下就带过，装不下整体移入 `legacy-preview-card.md`（并点名）。
4. **收缩顺序**：先砍摘录 → 再砍旧合成段 → **最后才减引文**（引文是本次任务的载荷）。
5. **装不下就响亮失败**：换底后仍超预算的卡，`--apply` **exit 1** 并点名，绝不静默产出超限文件。

**实测投影代价（`--check` 现算）**：

| 读数 | 值 |
|---|---|
| 换底卡 | 145 张（93 同 ID + 52 新增） |
| 内联逐字引文 | **1,346 / 2,109 条（63.8%）** |
| 引文**完整**内联 | **86 张** |
| 引文**有界**内联（未内联部分逐条声明 + 指向 `full-card.md`） | **59 张** |
| 超 staging 预算 | **0**（最大卡 12,240 字节 = 恰好用满） |

> **口径诚实性**：59 张卡的引文没有全部到页内。这不是「做完了」，是**有界的、被声明的**投影。每张卡自己的 frontmatter 记着 `rebase_evidence_quotes`（内联）/ `_total`（完整卡条数）/ `_complete`（true/false），正文里写「内联 k / 全 N 条 —— 其余 N−k 条见 …」。`--check` 的判据是**正面要求那个精确的缺口声明串**存在，缺它即判红（首版只要求标题含「全 N 条」，于是「悄悄删一条引文、标题不动」照样通过 —— 假绿，已修，见 §5）。

### 3.3 与并发写入者的**口径统一**（这是本次最有价值的发现之一）

实测发现另一个会话（S? 线）已建了一条**溯源透传**通道：`scripts/sync-card-facets.mjs` + `data/card-facets.json`，按 `p2s_card_id` 把 vault frontmatter 的
`venue → p2s_venue` / `venue_tier → p2s_venue_tier` / `evidence_grade → p2s_evidence_grade` / `paper_id → p2s_paper_id`
写进**已装卡**（实测 93 张已带）。

**若我按自己的命名写一套不带前缀的 `venue_tier`/`paper_id`，同一个事实就有两个字段名**——而且 `import-paper2skills.mjs` 整份重写已装卡、只保留 4 个 keep-key，**装机会把他们的那套静默抹掉**。
⇒ 我改用**同一套 `p2s_*` 命名、同源同值**（实测：与已装卡逐项比对 **93/93 零差异**）。装机因此无损。

---

## 4. 验收①：「同名同物」判定器 —— 可机械重跑，**且它真的会说「不是同一张」**

`paper2skills-research/scripts/check_card_identity.py`。退出码沿用本仓库口径 **0/1/2/3**。

### 4.1 六态（每张卡恰落一态，恒等式进判据 I8）

| 态 | 现测（换底后） | 含义 |
|---|---|---|
| `SAME_KEY` | **129** | p2s_card_id 逐字 == vault stem |
| `RENAMED_SAME` | **16** | 同上，但 slug 与 vault 文件名不同型（**改名同物**，一等输出、不报红） |
| `P2S_ONLY_PREVIEW` | **1237** | 只有预览版，无 vault 卡（一等输出） |
| `VAULT_ONLY` | **1** | 精选线独有且**未被任何 p2s 卡认领**（就是被阻断那张，§6.2） |
| `SAME_NAME_DIFFERENT_THING` | **8** | 🔴 **报红** |
| `UNDECIDABLE` | **0** | 🔴 名字像但判不出 ⇒ 报红并逐条列出（本语料恰好为 0） |

### 4.2 **同名异物：实测 8 条，逐条有名有姓**

任务卡要求「这一态必须能被判出来，否则判定器只会说『都对得上』」。判出来了，**8 条**：

| p2s 卡 | `p2s_card_id`（vault 里**不存在**） | 名字像的 vault 卡 | 已被谁认领 |
|---|---|---|---|
| `p2s-agent-time-series-forecasting` | `Skill-Agent-Time-Series-Forecasting` | `Skill-Time-Series-Forecasting` | `p2s-p2s-time-series-forecasting` |
| `p2s-causal-ml-feature-engineering` | `Skill-Causal-ML-Feature-Engineering` | `Skill-Feature-Engineering` | `p2s-p2s-feature-engineering` |
| `p2s-conformal-time-series-forecasting` | `Skill-Conformal-Time-Series-Forecasting` | `Skill-Time-Series-Forecasting` | `p2s-p2s-time-series-forecasting` |
| `p2s-temporal-fusion-transformer-inventory` | `Skill-Temporal-Fusion-Transformer-Inventory` | `Skill-Temporal-Fusion-Transformer` | `p2s-p2s-temporal-fusion-transformer` |
| `p2s-causal-time-series-forecasting-gcf` | `Skill-Causal-Time-Series-Forecasting-GCF` | `Skill-Time-Series-Forecasting` | `p2s-p2s-time-series-forecasting` |
| `p2s-multi-armed-bandit-ab-hybrid` | `Skill-Multi-Armed-Bandit-AB-Hybrid` | `Skill-Multi-Armed-Bandit` | `p2s-p2s-multi-armed-bandit` |
| `p2s-llmlingua-context-compression` | `Skill-LLMLingua-Context-Compression` | `Skill-Context-Compression` | `p2s-p2s-context-compression` |
| `p2s-causal-uplift-modeling` | `Skill-Causal-Uplift-Modeling` | `Skill-Uplift-Modeling` | `p2s-p2s-uplift-modeling` |

**承重半句**是判据的关键：不是「名字是超串就算同物」，而是**「近名卡已被另一张 p2s 卡认领 ⇒ 本卡是在抢名，不是同一张」**。
逐条人工复核过 3 例（Agent时序预测 vs 时序预测：前者 `16-智能体工程` 的编排卡、后者 `03-时间序列` 的经典卡；TFT-Inventory vs TFT；MAB×A/B混合 vs MAB；LLMLingua vs Context-Compression；Causal-Uplift vs Uplift）—— **全部确认为不同的卡**。
**危害**：naive 的「归一化后模糊匹配」会把 `Skill-Temporal-Fusion-Transformer-Inventory` 接到 `Skill-Temporal-Fusion-Transformer` 上 —— 那会把两张不同的已装卡之一**换底成另一张卡的正文**。

> **诚实登记**：这 8 条**不是 S5 造成的**，是既有的分类缺陷（p2s_card_id 指向 vault 里不存在的卡）。它们**不影响**本次 93 张换底（那 93 张走 `SAME_KEY`）。修复归属 S13/Q4 的分类线，已列工单 §8-W2。

### 4.3 退出码与三件套

| 命令 | 结果 |
|---|---|
| `--check`（对现场重算） | **exit 1** —— 8 条 `SAME_NAME_DIFFERENT_THING`。**这是判红，不是失败**：判定器必须能说出「不是同一张」 |
| `--selftest`（含端到端真 CLI，31 用例） | **exit 0 · 31/31** |
| `--mutate`（5 变异） | **exit 0 · 变异生效 5/5 · 抓住 5/5** |
| 输入没拿到（路径不存在 / vault 扫到 0 张） | **exit 2**（用例 c5/c6 锁定） |

**豁免可见**：`--baseline <json>` 支持已知红项豁免，报告里单列「已豁免 N 条」+ 打印 `expires_when` + 提示「转绿后请删掉它」。实测豁免 1 条后 exit 0，而未登记的错项**仍判红**（用例 c10/c11）。

---

## 5. 反后门 · 反假绿：**变异测试抓出的都是我自己的缺陷**

两个脚本各跑 `--selftest`（端到端真 CLI）与 `--mutate`（先证变异改变真实取值，再谈判据有没有劲）：

| 脚本 | selftest | mutate |
|---|---|---|
| `check_card_identity.py` | **31/31** | **5 条 · 生效 5 · 抓住 5** |
| `rebase_p2s_cards.py` | **39/39** | **7 条 · 生效 7 · 抓住 7** |

### 变异测试实际抓出的我的缺陷（逐条留档，这是本节最该留的部分）

1. **`assert` 恒真（用例是摆设）**：`c3 判成 UNDECIDABLE` 写成 `"UNDECIDABLE" in txt` —— 而**六态表每次都打印全部六个状态名**，这条断言**恒真**。变异 M2（把未决改判成「无事」）在它上面照样绿。改成断言**计数**。**同族还有**：`_extract_counts` 的正则 `[A-Z_]{4,}` 匹配不到 `P2S_ONLY_PREVIEW`（含数字 `2`）⇒ 取到 `None` 而不是 `0`，另一条断言恒假。
2. **锚点计数包含变异表自身** ⇒ `src.count(anchor)` 恒为 2，5 条变异全被判「锚点不唯一」；且 `--mutate` 原本用「干净夹具」当唯一探针 ⇒ 摘掉一条**从不触发**的判据时读数不变，被误判成「变异没生效」。改为**变异只施在代码区** + **6→7 个探针电池**（每条判据配一份篡改夹具）。
3. **用被测常量构造夹具 = 自指**：12 KB 用例写成 `MAX_SKILL_BYTES + 50`，而变异 M1 正是改 `MAX_SKILL_BYTES` ⇒ 夹具跟着变、用例恒绿。改为钉**字面 12288**，并加一条「门禁常量就是 12288」的断言。
4. **模块级派生常量不跟 patch**：`STAGING_BUDGET = MAX_SKILL_BYTES - 48` 在 import 时算一次，`mock.patch` 改 `MAX_SKILL_BYTES` 它**不跟着变** ⇒ 三条预算用例当场变假。改成 `staging_budget()` 函数。
5. **`next(gen)` 让变异体崩掉而不是判红**：M1 变异后 `next(s for s in hits if s in claimed_by)` 抛 `StopIteration`，报告里只剩 traceback、没有 `❌`，于是被判「没抓住」。改用 `claimed_hits[0]`，并让「抓住」的判据接受 traceback。
6. **`--check` 抓出交付物自身的不一致（最重要的一条）**：首版把 `fm_text` 在收缩循环**之前**定死 ⇒ **46 张卡的自报引文条数是收缩前的值**（读数：`frontmatter 说内联 32 条，实物 18 条`）。这是**我自己的门禁**在真实语料上抓出来的，不是用例。
7. **静默截断的假绿通路**：引文完整性判据首版只要求标题里出现「全 N 条」。于是「删掉一条引文、标题原样不动」通过。改成**正面要求精确的 `内联 k / 全 N 条` 声明**；用例 2 的篡改样本也**加强成「连 frontmatter 自报条数一起改」**——因为首版篡改被另一条判据顺手兜住了，`静默截断` 这条**从未被触发过**。
8. **引文一条都装不下时整段消失**：预算不够时连 `## 原文引用` 标题都不打印 ⇒ 读者无从知道这张卡本来有引文。补「内联 0 / 全 N 条」的可见缺口声明（用例 9c）。

---

## 6. 验收②：`quality_tier` 入库 + 验收③：旧有门禁逐项退出码

### 6.1 `quality_tier`

| 项 | 值 |
|---|---|
| 覆盖 | **1390 / 1390 = 100%** |
| 取值域 | `curated`（145：93 同 ID + 52 新增）· `preview`（1245） |
| 词表出处 | 不是我定的 —— 产品侧 `dsh-paper2skills/lib/axis.js` 里已有 `QUALITY_TIERS = ['curated','preview']`（F4 落的） |
| 换底**前**实测 | p2s 侧 **0/1338**；vault 侧 **0/146** ⇒ 这是本次**新造**的字段 |

> ⚠️ **必须一并报告（否则这个「入库」是假的）**：产品侧的 `lib/axis.js::buildAxisIndex()` **第 175 行把 `quality_tier` 硬编码成 `'curated'`**，从不读 frontmatter：
> ```js
> quality_tier: 'curated',
> ```
> 而 `validateAxis` 的 A4 判据 `if (!QUALITY_TIERS.includes(c.quality_tier))` 判的是这个硬编码值 ⇒ **A4 的 quality_tier 分支恒真，永远不可能报红**。且 `check-axis.mjs` 只读 **vault** 卡，**根本不读产品侧这 1390 张**。
> ⇒ **换底后 `quality_tier` 在产品侧没有任何消费者，是只写字段。** 我**没有**改 `lib/axis.js`（那是 F4 线的判据文件、且本仓有活跃并发写入者），已列工单 §8-W1。

### 6.2 换底的执行结果与**被阻断的 1 张**

| | 计划 | 实交 |
|---|---|---|
| 同 ID 换底 | 93 | **93** ✅ |
| vault 独有新增 | 53 | **52** ⛔ 1 张被可见阻断 |
| 只有预览版 | 1245 | **1245** ✅（只插一行 `quality_tier: "preview"`，最小手术） |
| **staging 总卡数** | 1391 | **1390** |

**被阻断的那张（不许静默丢弃，逐条列出）**：

```
Skill-大规模消费者评论方面情感分析（07-NLP-VOC/00-知识库-Skill卡片/）
  slugFor('Skill-大规模消费者评论方面情感分析') === 'p2s-'
  ⇒ 不匹配 NAME_RE /^[a-z0-9]+(?:-[a-z0-9]+)*$/
  ⇒ import-paper2skills.mjs 报「name 非法」并 exit 1（实测）
```

**根因不是我的脚本，是产品侧 `lib/taxonomy.js::slugFor()` 的缺陷**：它把非 ASCII 全折成 `-` 再 trim，**全中文的卡名会折出一个空的 `p2s-`**，而且它**不校验结果是否合法**。
我**没有替它改名** —— 改名＝自造第二套 slug 规则，S12 的 `id→slug` 换算会算错。已把判据补进 `append-selected-line.mjs`（`NAME_RE` 校验 + **可见阻断** + 需 `--allow-blocked` 才继续），并开工单 §8-W3。

### 6.3 旧有门禁逐项退出码（**「我跑了」不算，这里贴码与分母**）

| 门禁 | 命令 | 退出码 | 分母 / 读数 |
|---|---|---|---|
| **安装校验（含 12 KB 硬门禁）** | `node scripts/verify-install.mjs` | **0** | 应装 1390 / 已装 1390 / 缺失 0 / **问题 0** |
| 五层分类轴 + facet | `node scripts/check-axis.mjs` | **0** | 146/146 分类、146/146 可挂 M 格、双向筛选等价全过 |
| **S12 消费口闸门** | `node scripts/check-contract-gate.mjs` | **0** | 见 §7.1（账已变） |
| 溯源透传一致性 | `node scripts/sync-card-facets.mjs --check` | **0** | **1390 张卡的 5 个字段与台账逐项一致** |
| 完整实现恢复索引 | `python3 scripts/build-source-code.py --check` | **0** | 1338 张 · oracle 1277 · recovered 1317 |
| 产品侧单测 | `npm test` | **0** | **100/100 pass · 0 fail** |
| 凭证扫描 | `python3 paper2skills-skills/paper-维护/scripts/scan_secrets.py` | **0** | 扫 3169 个文件（文本 3168/二进制 1）· 未发现凭证 |
| **PHASE6 门禁集（`--fast`，13 条）** | `python3 paper2skills-research/scripts/run_phase6_gates.py --fast` | **1** | ✅12 · 🔴**1**（L3c）· ❓0 · 💥0 |
| 换底自查 | `python3 paper2skills-research/scripts/rebase_p2s_cards.py --check` | **0** | 1390 staging / 145 换底 / 1245 preview / 未打标 0 / 超预算 0 / quality_tier 1390-1390 / 引文 1346-2109 |
| 同物判定 | `python3 paper2skills-research/scripts/check_card_identity.py --check` | **1** | 8 条同名异物（**预期判红**，非失败） |
| 换底器三件套 | `--selftest` / `--mutate` | **0 / 0** | 39/39 · 变异 7/7 生效 7/7 抓住 |
| 判定器三件套 | `--selftest` / `--mutate` | **0 / 0** | 31/31 · 变异 5/5 生效 5/5 抓住 |

**换底前的同一批读数（对照组）**：PHASE6 门禁集（**24 条**全集）当时是 ✅23 · 🔴1（L3c，J9 一条，与 S5 无关）；`verify-install` 0（1338/1338）。
> ⚠️ **门禁集在执行期间一直在长**：19 → 22 → 24 → **27 条**（并发会话在加门禁）。上表的 `--fast` 13 条与 24 条全集都是**当时**的现算。**任何「全绿/几条红」的说法都必须带上它跑的是哪一版** —— 这正是「别把『重跑一遍闸门』当成『闸门仍然正确』」的具体含义。

### 6.4 源指纹（约束⑤）：**零漂移**

`--apply` 记下每张源卡的 sha256；收尾重算：

```
vault 源卡：换底前 146 · 现算 146
漂移（换底前后 sha256 变了）：0
消失 0 · 新增 0
```

`--check` 也会逐张重算并报 `源卡漂移`（用例 4 用篡改样本锁定）。
> 补充：执行期间 vault 侧确实有并发写入者（S10 写 `venue-tiers.json`、S13 落 `l3_*`），**但在我拷贝的窗口内源卡未变**。这是实测结论，不是假设。

---

## 7. 【必须报告】会失效的下游 —— **两条门禁被换底打红，我修了一条，另一条修不了**

### 7.1 S12 消费口闸门账：**数已变，闸门不需要改**

| 项 | 任务卡写的（已过期） | 换底**前**现算 | **换底后**现算 |
|---|---|---|---|
| 已装线卡总数 | 1338 | 1338 | **1390** |
| 契约引用条目 绑定 | — | 334 | **348** |
| 契约引用条目 **待装线** | — | **14** | **0** ✅ |
| 契约引用条目 **无法解析** | — | 0 | **0** |
| 已挂契约 | 29 | 80 | **80** |
| 待挂契约 | 114 | 63 | **63** |
| 未接线 | 1195 | 1195 | **1247** |
| 白名单 p2s 去重 | 143 | 143 | **143**（未变） |

- **恒等式**：`1390 = 80 + 63 + 1247` ✅
- **闸门自己的前置条件被满足**：S12 原文写着「待装线 14 条 … 这是 S5（p2s 换底装精选线）的前置依赖；**换底后若 id 与 slug 对不上，本闸门会立刻报 `REF_UNRESOLVABLE`**」—— 实测 **待装线 14 → 0，无法解析仍为 0**，且「可写却零绑定」的契约从 2 份（`CTR-A-046`、`CTR-B-014`）**归零**。
- **闸门本身要不要改**：**不需要**。它的三态判定、退出码、白名单口径都不变；变的是输入。
- ⚠️ **但「重跑一遍闸门」≠「闸门仍然正确」**：我额外做了两条独立核对 —— (a) `slugById` 的 `id→slug` 映射逐条落到「绑定」而非「待装线」，说明命名空间归一化真的生效；(b) `sync-card-facets --check` 用**另一条独立仪器**（台账 vs 装机态）给出 1390/1390 一致。

### 7.2 🔴 **L3c（卡端分类 J1–J8）：exit 1，其中 52 条是 S5 造成的，我修不了**

| | 换底前 | 换底后 |
|---|---|---|
| `build_card_classification.py --check` | exit 1 · **1 条**（J9，`Skill-Cold-Start-Product-Recommendation` 的 sha1 行漂移，**与 S5 无关**） | exit 1 · **52 条**（J4 × 52，**全部由 S5 造成**） |

**J4 报的是什么**（逐条格式）：

```
Skill-Routed-Graph-Handoff：产品侧已有落点 ['依赖协调','接口契约']，inbox 却按 new 提交
                             —— 这正是不许的静默分叉；要改就写 reclassify: true + 理由
```

**我做的独立核对（差分证据，不是「我看了一遍」）**：

```
paper2skills-research/data/card-classification-inbox.json  53 条
  其中，产品侧底本**已有该 id 且 l3 逐字相同**：52   ← 零内容分叉
  l3 真分叉：0
  产品侧仍无该 id：1                                ← 就是 §6.2 被阻断那张，两处独立计数互相印证 ✅
```

⇒ **52 条 J4 不是内容冲突，是 `source` 标签过期**：inbox 写 `source: new`（意为「产品侧根本没有」），而 S5 换底已经把这 52 张接进了产品侧底本，**落点逐字继承 inbox、零分叉**。

**为什么我修不了**：我试了正路 —— 重跑 `build_card_classification.py`（不带 `--check`）。**生成器自己 exit 1 并拒绝写盘**：`by_source` 仍是 `{inherited: 93, new: 53}`，`card-classification.json` 字节未变（`diff -q` 验过）。它的设计是：**「放弃 new 声明」必须由人写 `reclassify: true` + 理由**，脚本不许代做。这是本仓库「人不许静默 / 脚本不许代做人的决定」的既有控制，我尊重它。

**闸门要不要改**：**要，但改的是模型不是阈值**。J4 目前只有两个状态：
`{inbox=new, 底本=无}`（合法）与 `{inbox=inherited, 底本=有}`（合法）。
S5 引入了**第三态** `{inbox=new, 底本=有（由 S5 接入，落点逐字相同）}` —— 这是**合法**的，J4 不认识它。工单见 §8-W4。

### 7.3 ✅ L3a（缺口账与靶区工单）：被我打红，**我修好了**

`build_gap_ledger.py --check` 换底后报「产物已过期或与现状不一致（差异键：`['rows']`）」。原因单一：它把产品侧 `classification.json` 的 **sha256 与条数写进了产物抬头**（`n_items` 1338 → 1390），且 `n_legacy`（「仅 legacy」供给列）随新增卡而变。

处置：重跑生成器（**可逆**，两份产物都是 git 跟踪的，改前已备份到 `/tmp`），随后 `--check` **exit 0**。
**差分证据**（证明我改的是供给计数与输入指纹，不是结论）：

```
- 输入：… 产品侧 classification.json sha256:`28376ccb3d8dfecd`（1338 条 legacy 卡）
+ 输入：… 产品侧 classification.json sha256:`07395d2059081620`（1390 条 legacy 卡）
  各行 n_legacy 计数上升（例：20 → 35、7 → 11）
  缺口账总量不变：零供给 15 / 仅 legacy 69 / 有精选卡 67 = 151
  靶区一 31 条（结构性空白 3 条）· 靶区二 15 条 · 靶区三 A 类 73 条
```

### 7.4 ⚠️ 跨线耦合：**`import:skills` 会静默抹掉 `p2s_*` 透传字段**（实测复现）

装机链 `import-paper2skills.mjs` 整份重写已装卡，只保留 4 个 keep-key（`disable-model-invocation` / `user-invocable` / `workflow` / `whenToUse`）。实测：

```
node scripts/import-paper2skills.mjs
node scripts/sync-card-facets.mjs --check
  🔴 1338 处漂移（p2s_code_level 装机态「」≠ 台账「完整实现·可解析」）
```

即 **一次装机把 `p2s_*` 五个字段从 1338 张卡上全抹了**。处置：装机后立刻 `sync-card-facets.mjs --apply`（已写进交付序列），随后 `--check` **1390/1390 一致**。
**本次换底顺带把它削掉了一半**：我让 145 张换底卡**自带** 4 个 `p2s_*` 字段，装机不再依赖事后补写（实测四个字段与已装卡 **93/93 零差异**）；只剩 `p2s_code_level` 仍靠事后补。工单 §8-W5。
（这条是那个会话自己 docstring 里警告过的形态；我把它**复现并量化**了。）

### 7.5 ⚠️ 12 KB 门禁量的是**已装卡**：`staging ≤ 12288` **推不出** `装机态 ≤ 12288`

这是本次最隐蔽的一处。`assemble-skills.mjs` 在 staging 上量 12 KB，而 `verify-install.mjs` 在**已装卡**上量；装机链之后还有一步会往 frontmatter 追加 `p2s_*`。

实测：我首版把 staging 顶到 12288，`--check` 全绿，装完机 `verify-install` **当场判红 13 张**（12,290–12,330）。
量化后定 `INSTALL_HEADROOM_BYTES = 48`：装机态 vs staging 的实测 delta 分布 `+42`×1196 / `+28`×42 / `0`×33 / `+45`×5 ⇒ **48 = 实测上界 45 + 3**（**观测上界 + 固定余量，不是算出来的**；`sync-card-facets` 若加字段必须重测）。已在读数里打出「超 staging 预算 N（预算 12240 = 12KB − 装机余量 48；最大卡 …）」，权威判据仍是在已装卡上量的 `verify-install`。变异 M7 锁这条（探针 P7 专为它而设）。

---

## 8. 工单（可按序执行）

| # | 工单 | 归属 | 证据 / 验收 |
|---|---|---|---|
| **W1** | `lib/axis.js::buildAxisIndex()` 第 175 行硬编码 `quality_tier: 'curated'` ⇒ A4 的 quality_tier 分支**恒真**，且只读 vault 146 张、不读产品侧 1390 张。改为从 frontmatter 读、缺失记 `unlabeled` 并计数 | F4 线 | 我未改该文件（并发写入者活跃）；改后应能在**产品侧 1390 张**上复算 `curated 145 / preview 1245` |
| **W2** | 8 条**同名异物**（§4.2 表）：`p2s_card_id` 指向 vault 里不存在的卡 ⇒ 应改指向正确 id，或登记为「无论文来源的精编卡」 | S13/Q4 分类线 | 判定器 `--check` 当前 exit 1；修好后应降至 0 |
| **W3** | `lib/taxonomy.js::slugFor()` 对**全非 ASCII** 卡名产出非法 slug `p2s-`，且不校验。需定 fallback 规则（**这是命名决策，不是工程细节**）。命中 1 张：`Skill-大规模消费者评论方面情感分析` | taxonomy 归属方 | `append-selected-line.mjs` 现已可见阻断并需 `--allow-blocked`；修好后该卡可接入（52 → 53） |
| **W4** | `build_card_classification.py` 的 J4 增加**第三态**：`{inbox=new, 产品侧底本已有该 id 且 l3 逐字相同}` 判合法（S5 换底接入即此形态）。**不要**放宽成「有落点就放行」—— 那样会掩盖真分叉 | F5 线 | 落差证据：52 条 J4 中 **0 条**是真分叉（落点逐字相同）。`--check` 现 exit 1 |
| **W5** | `import-paper2skills.mjs` 的 `KEEP_KEYS` 泛化为「已装卡里凡 `p2s_*` 开头的字段，重装时原样带过」（同台账 #24「生成器不拥有的行，不许删」）。治本，或至少在装机后自动跑 `sync-card-facets --apply` | S2b 安装线 | 实测复现：装机后 `sync-card-facets --check` 报 **1338 处漂移** |
| **W6** | 52 张新增卡缺产品侧 UX 字段：`user_summary`（1338/1390）、`whenToUse`（1338/1390）、`user_try`（1326/1390）、`workflow`（1263/1390）。卡片可斜杠调用，但目录/搜索体验弱于既有卡 | S2a 适配置线 | 字段覆盖实测见 §2 表 |
| **W7** | 59 张卡的引文**有界内联**（未内联部分只在 `references/full-card.md`）。若要求「引文 100% 到页内」，需另立门禁或放宽 12 KB —— **当前两者不可兼得**，本报告选择「门禁优先 + 缺口可见」 | 主控决策 | `rebase_evidence_quotes_complete="false"` 的卡可一键列出 |
| **W8** | vault 侧 1 张卡仍未被任何 p2s 卡认领 = §6.2 被阻断那张。W3 修好后自动消解 | 随 W3 | 判定器 `VAULT_ONLY = 1` |

---

## 9. 我**没做**的 / 我**做不到**的（照实说）

1. **没有做 1245 张预览卡的「内容级」改善** —— 按 Q6 口径它们只标 `quality_tier: "preview"` 保持现状。它们**仍然**是八段预览页（含 457/1279 不可 `ast.parse` 的截断代码）。本报告**不**声称它们被修好了。
2. **59 张卡的引文没有全部到页内**（§3.2）。这是 12 KB 硬门禁与「引文一条不截断」的**不可兼得**，我选了「门禁优先 + 缺口逐条声明」。
3. **没有改任何并发写入者的文件**：`lib/axis.js`、`lib/taxonomy.js`、`import-paper2skills.mjs`、`card-classification-inbox.json`、`data/venue-tiers.json`、`dsh-algo-skills-local` 全部**未碰**，只出工单。
4. **没有 commit**（按约束，两仓都留给主控）。**`staging/` 是 gitignore 的**（`.gitignore:44`）⇒ 1,390 张卡**不在版本控制里**，回退靠的是每卡同目录的 `references/legacy-preview-card.md`（93 张）与 `/tmp/classification.pre.json`。**这是回退故事的真实边界，不是「git 能回滚」。**
5. **`git status` 里我改的**：
   - `magpie-horch`：`data/classification.json`（+52 条，1338→1390）、`manifest/paper2skills.json`（装机产物）、`data/card-facets.json`（`--apply` 重写）、`staging/**`（**gitignore，不出现在 status**）、新增 `scripts/append-selected-line.mjs`、新增 `generated/selected-line-plan.json`、`generated/rebase-plan.json`、`staging/import-report.json`。
   - `paper_to_skills`：`paper2skills-research/scripts/{check_card_identity,rebase_p2s_cards}.py`（新增）、`paper2skills-research/data/gap-ledger.json` + `reports/PHASE6-F3-缺口账与靶区工单.md`（重生成，§7.3）。
   - **凡不是我改的一律没 stage**（`dsh-algo-skills-local`、`ADR-0072`、`venue-tiers.json`、`card-facets.json` 的**生成器**等都未动）。
6. **一组「我拿不到 = 登记为拿不到」**：`NAACL 2026` 不存在之类与本任务无关；但**「8 条同名异物到底该指向哪张卡」我判不出**（vault 里没有对应 id），只判出「**不是同一张**」—— 这正是 `UNDECIDABLE`/`SAME_NAME_DIFFERENT_THING` 两态存在的意义。
7. **我未独立复核 `p2s_src_domain` 的语义**（它是产品侧既有字段，我原样继承）。

---

## 10. 复现命令（一条链，按序）

```bash
cd /Users/lute/project/paper_to_skills

# ① 同物判定（预期 exit 1：8 条同名异物）
python3 paper2skills-research/scripts/check_card_identity.py --check --json-out /tmp/identity.json
python3 paper2skills-research/scripts/check_card_identity.py --selftest   # 31/31 → 0
python3 paper2skills-research/scripts/check_card_identity.py --mutate     # 5/5 → 0

# ② 换底三件套 + 对现场重算
python3 paper2skills-research/scripts/rebase_p2s_cards.py --selftest      # 39/39 → 0
python3 paper2skills-research/scripts/rebase_p2s_cards.py --mutate        # 7/7 → 0
python3 paper2skills-research/scripts/rebase_p2s_cards.py --check         # 0

# ③ 精选线接入产品侧底本（幂等；1 张被可见阻断需 --allow-blocked）
cd /Users/lute/project/magpie-horch/packages/capabilities/dsh-paper2skills
node scripts/append-selected-line.mjs --check --plan-out generated/selected-line-plan.json
node scripts/append-selected-line.mjs --allow-blocked --apply --plan-out generated/selected-line-plan.json
python3 /Users/lute/project/paper_to_skills/paper2skills-research/scripts/rebase_p2s_cards.py --apply

# ④ 装机 + 恢复透传（**两步都必跑**，见 §7.4）
node scripts/import-paper2skills.mjs
node scripts/sync-card-facets.mjs --apply && node scripts/sync-card-facets.mjs --check

# ⑤ 门禁
node scripts/verify-install.mjs          # 0
node scripts/check-axis.mjs              # 0
node scripts/check-contract-gate.mjs     # 0（待装线 0 / 无法解析 0）
npm test                                 # 0（100/100）
python3 /Users/lute/project/paper_to_skills/paper2skills-research/scripts/run_phase6_gates.py --fast
                                          # ✅12 🔴1（L3c，52 条 J4，见 §7.2）
```

---

## 11. 一页速览

| 交付项 | 计划 | 实交 |
|---|---|---|
| 已装卡 | 1338 | **1390** |
| 含逐字引文的已装卡 | **0** | **145** |
| 内联逐字引文 | — | **1346 / 2109 条**（完整内联 86 张 / 有界内联 59 张） |
| `SKILL.md ≤ 12KB`（已装卡实测） | 全过 | **1390 / 1390** |
| 93 张同 ID 换底 | 93 | **93** ✅ |
| 53 张 vault 独有新增 | 53 | **52** ⛔ 1 张可见阻断（§6.2） |
| 1245 张标 `preview` | 1245 | **1245** ✅ |
| `references/full-card.md` | — | **145** ✅ 逐字节 + sha256 |
| `quality_tier` 覆盖 | 100% | **1390/1390**（curated 145 / preview 1245） |
| 源指纹漂移 | 0 | **0**（146 张源卡 sha256 换底前后逐张相同） |

> ⚠️ **必须一并知道的代价**：换底打红了 **2** 条上游门禁 —— **L3a**（我已修好：重生成缺口账，`--check` 回 0）与 **L3c**（**52 条 J4，我修不了** —— 生成器自己 exit 1 拒绝写盘，J4 的豁免按设计是**人的决定**）。
> 另：`quality_tier` 在产品侧**暂无消费者**（`lib/axis.js:175` 硬编码 `'curated'`，A4 判据因此恒真）。

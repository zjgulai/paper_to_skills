# PHASE6 F4 + F5 · 五层分类轴与 53 张卡的分类

> 生成：2026-09-13　|　对应任务板 B3（F4）/ B4（F5）
> 结论一句话：**轴的判据在 `lib/axis.js` 里只有一份（复用 taxonomy.js），53 张卡全部落地，
> 而「母婴锚点」这件事本轮没能找到可信的仪器 —— 两把尺子都留档，分歧如实登记。**

---

## 0. 交付物

| 类型 | 路径 | 说明 |
|---|---|---|
| 卡端事实源（入库） | `paper2skills-vault/07-资源库/card-classification.json` | 146 张卡的 L3 落点 + 21 条技术域错位登记 + 锚点扫描 + 分歧表（122 KB） |
| 输入（入库） | `paper2skills-research/data/card-classification-inbox.json` | 53 张卡的逐卡结论（5 个分类子代理）+ 错位登记 + 锚点意见 |
| 生成器 + 门禁 | `paper2skills-research/scripts/build_card_classification.py` | J1–J8；`--check` 退出码 0/1/**2**；`--excerpts` 出题面；`--selftest` 12 用例 |
| 轴定义（产品侧） | `dsh-paper2skills/lib/axis.js` | 五层链 + 三个 facet；判据复用 `lib/taxonomy.js`，**不重造分类学** |
| 轴门禁（产品侧） | `dsh-paper2skills/scripts/check-axis.mjs` | 在真实 vault 上跑 A1–A6；退出码 0/1/2 |
| 轴单测（产品侧） | `dsh-paper2skills/test/axis.spec.mjs` | 17 用例 / 9 份变异样本 |
| 上游改动 | `build_capability_graph.py` 卡↔L3 换源；`build_gap_ledger.py` 数值随动 + J7 needle 收窄 | 见 §4 |

---

## 1. F4：五层分类轴（技术域降为 facet）

五层（**任务书口径**）：①50 岗位 → ②151 责任 → ③64 场景格 → ④8 方案域 + 139 契约 → ⑤N 算法卡。
⚠️ 方案文档 §7 写成 L0–L5（6 行，契约单列）；两者节点相同、只是切分不同，`AXIS` 里注明了。

三个 facet（**只用于筛选，不参与主分类、不参与任何门禁判定**）：

| facet | 取值 | 实测 |
|---|---|---|
| `tech_domain` | vault 顶层数字目录 | **16 个**（`00-项目管理` / `07-资源库` / `11-AI人文` 无卡） |
| `venue_tier` | 白名单 7 档 + `unlabeled` | **121/146 = 82.9% 未标注**（S10 负责回填）；已登记遗留值 `top×4 / workshop×2 / demo×1` |
| `quality_tier` | `curated` / `preview` | 本文件 146 张全为 `curated` |

### 1.1 技术域必须取**顶层**目录，不能取卡的父目录名

实测：`00-知识库-Skill卡片` 是 `07-NLP-VOC/` 与 `10-MAS/` 的**内层**目录，42 张卡住在那儿。
若按 `p.parent.name` 取，这 42 张会被折叠成一个**假域**，且 `10-MAS` 与 `07-NLP-VOC` 再也分不开。
⚠️ 形态正则 `^\d{2}-` 对假域是**放行**的（`00-知识库-Skill卡片` 也匹配）⇒
所以判据必须是「**在 vault 顶层实有的域目录里**」这第二层，单靠形态抓不住。
`test/axis.spec.mjs` 变异 6 用构造样本锁定这一点；`--selftest` 另用真实 vault 的
42 张卡的 `path` 做一次整批替换证明会报红。

### 1.2 venue 遗留值：**登记不报红**，但表外取值报红

白名单 7 档里没有 `top` / `workshop` / `demo`，而它们是**状态标记**而非档位
（whitelist §3 的降级表把 Workshop 判「降一级」、Demo 判「降为 field-top 以下」——
描述的是**要不要降**，不是降完是哪一档）。
⇒ 映射到哪一档是 **S10** 的决策，`lib/axis.js` **不替它决定**（`to: null`）。
本模块只做一件事：**登记**。出现在表里 ⇒ 计数、不报红；出现在表外 ⇒ 报红。
另配一条**腐烂检查**：登记了却不再出现 ⇒ 提示从 `VENUE_LEGACY_PENDING` 删除
（不是报红 —— 那会把「S10 修好了」判成失败）。

### 1.3 双向筛选：两条**各自独立**建的索引

`byBusiness()` 只读 `l3` / `cells`；`byFacet()` 只读 `path` / `frontmatter`。
两者字段不重叠，故 `axisAgree()` 是**真断言**而不是同义反复；穷举
「16 个技术域 × 8 个责任域 = 128 个组合」全部等价（`check-axis.mjs`）。

### 1.4 A5：frontmatter 解析**不猜**

只认单行 `key: value`；缩进续行 / `- ` 列表项记进 `unparsed` 并由门禁报红
（「拿不到就炸，不要给默认值」）。
⚠️ **首版把 `#` 注释也算 unparsed，在真实 vault 上误报 12 条**（`# created 未知：本卡早于…`
这类编者按）。注释是合法 YAML ⇒ 现在跳过注释，但**缩进/列表项仍然报红**。
单测里把两个方向都锁住了。

---

## 2. F5：53 张卡全部落地

### 2.1 怎么判的

53 张卡按 id 分成 5 批（11/11/11/11/9），交 5 个子代理逐卡读全文后按
`L3 白名单 151 条`（逐字，禁改写/简称/新造）出结论。
**合并前不信自述，回到题面文件点 id**：断言每批与 `batch-0N.md` **同序同集**、
L3 逐字 ∈ 151 条、无 `l3` 越界 —— 实测 0 条不符。

### 2.2 结果

| | 值 |
|---|---|
| 已分类 | **146/146**（新增 53 张全部落地，0 张空挂） |
| 置信度 | `high@f5-review 23` / `medium@f5-review 30` / `high@product-pipeline 50` / `medium@... 42` / `low@... 1` |
| L3 覆盖 | **67/151**（84 条责任名在精选线里零供给 —— 这是 S4 的靶区） |
| 可挂 M 格 | **146/146**（24 个 M 格全部被覆盖） |

⚠️ **置信度分两个来源**：93 张继承的是产品侧流水线的判定，53 张是本次子代理的判定。
同名列 `high` 来自两个不同的判定器 ⇒ 产物里带 `confidence_source` 字段，`by_confidence`
按 `值@来源` 分组。**只取 l3 然后把 confidence 填成自己的默认值，等于把别人的结论伪装成自己的。**

### 2.3 零漂移门禁（J4）

精选线的 93 张与产品侧 1338 张的落点共用同一批卡 ⇒ `source: inherited` 的条目**逐条断言取值相等**。
要改必须显式写 `reclassify: true` + `inherited_l3` + 理由；**静默分叉即报红**。
本轮实测：53 张新卡**与产品侧零重叠**（已断言），故全部是 `new`，未触碰这条门禁。
`--selftest` 用 4 个篡改样本锁定该门禁（静默分叉 / 空声明 / 无理由 / inbox 重复 id）。

---

## 3. 技术域错位：21 条在册，**只登记不移动**

F5 边界点名 5 张（Monodense 定价卡、Cold-Start 推荐卡、Opportunity-Mining 选品卡、
两张「元卡」），5 个子代理又独立报出 16 张。**合计 21 条**，每条带 `actual_business` 与来源。

逐条复核了 F5 点名的 5 张（不是抄文档）：

| 卡 | 现挂技术域 | 卡内实证 |
|---|---|---|
| `Skill-Monodense-单品价格弹性估计` | 04-供应链 | `description` 自述「为**动态定价**和促销决策提供量化依据」 |
| `Skill-Cold-Start-Product-Recommendation` | 06-增长模型 | `paper: Large Language Model Simulator for **Cold-Start Recommendation**` |
| `Skill-New-Product-Opportunity-Mining` | 06-增长模型 | `paper: Startup Success Forecasting Framework`（→ 选品/组合决策） |
| `Skill-Knowledge-Graph-for-Skills-Management` | 08-知识图谱 | ② 场景是**团队内部**新人的技能学习路径推荐 |
| `Skill-Skill-Lifecycle-Design` | 16-智能体工程 | ② 场景是重构**本项目自己的** Skill 库（元卡） |

**为什么不移动**：移动目录会断掉卡内 `related:` 与论文档案引用；而技术域只是 facet，
**主分类以 L3 为准**。F5 任务卡原文也是「错位只登记与重分类」。

---

## 4. 「母婴锚点」：两把尺子都不够用，都留档

### 4.1 子代理的判定不是测量

53 张里 4 张判 `baby_anchor: false`。但**各批口径不一致**：批次 02 自己就写明
「若你们的判据要求『必须含跨境渠道要素』，这张可复议为 false」——
即同一批里，`GPLR`（母婴侧很硬、0 次跨境词）判 true，而 `REVISION`（有跨境词）判 false。

### 4.2 机器扫描：首版量的是**章节标题**

把「有没有锚到母婴出海」降成两个可复算的计数（母婴品类词 / 出海要素词，词表在生成器里）。
**首版直接扫全文，146 张里 139 张判成 `both`** —— 因为每张卡都有
`## ② 母婴出海应用案例` 这个**样板标题**，`出海`/`母婴` 两个词在每张卡上都命中一次。
**量到的是章节名，不是卡的业务内容**（与 CLAUDE.md 记的「判据用错了信息源」同族第四次）。
修法：`body_lines()` 先剥标题行，并加一条**仪器自证**（只含样板标题、正文无锚点词的卡必须判 `neither`）。
修后分布：`both 105 / category-only 35 / export-only 4 / neither 2`。

### 4.3 分歧 10 条，如实登记

| 方向 | 条数 | 例 |
|---|---|---|
| 子代理判 true，扫描判 `category-only`（有母婴品类、正文零出海词） | 8 | `Skill-GPLR-人群标签生成`（16/0）、`Skill-PERSONABOT-RAG用户画像生成`（19/0） |
| 子代理判 true，扫描判 `export-only`（正文零母婴品类词） | 2 | `Skill-Cannibalization-Corrected-Attribution`（0/11） |
| 子代理判 false，扫描判 `both` | 0（剥标题后） | — |

**处置**：`anchor_opinions`（意见，带口径）与 `anchor_scan`（测量，带词表）**同时入库**，
分歧单列 `anchor_disagreements`。**不合成一个「锚点分」** —— 那会把两把尺子的差异平均掉。

### 4.4 调查报告 §5.3 的 18 条：**5 条可复现，13 条仍是散文**

§5.3 判的是「场景泛化 / 未落到母婴单品」（散文）；机器只能答「正文**有没有**母婴品类词」。
⇒ 拆两档：零品类命中的 5 条（`AB-Test-Result-Interpretation`、`OpenWorld-Class-Incremental-Learning`、
`Root-Cause-Analysis-Agent`、`Tree-of-Thoughts-Planning`、`Cannibalization-Corrected-Attribution`）
**可被关键字复现**；其余 13 条只有散文支撑，**不得说成已验证**。

---

## 5. 上游影响：F5 让缺口账整体变了一次（**上游变了，不是判据变了**）

精选线有 L3 落点的卡 93 → 146，于是 F3 的缺口账每一格都变：

| | F3 时点 | F5 时点 | 差 |
|---|---:|---:|---:|
| 有精选卡 | 53 | **67** | +14 |
| 仅 legacy | 81 | **69** | −12 |
| 零供给 | 17 | **15** | −2 |
| 靶区一 | 36 | **31** | −5 |
| 零供给结构空白 | 10 | **9** | −1 |

**A/B/C 判定、边界条目、L3 语义一个都没动**，变的只是「这一格有没有卡」。
处置：① 方案 §4 两套数并列（括号保留 F3 值），脚本里 `EXPECT_TABLE_F3` 留历史基线、
现行断言只认 F5 值；② 工单报告自动带差额表；③ 新增方案 §4.2 记录本次变化与两条纪律。

### J7 的**第二次**假阳性（同族，当场登记）

F5 在图谱脚本里加了一句改动说明「F3 修在 `build_gap_ledger.py`」，
**裸模块名 needle `gap_ledger` 当场把它判成「分数进了门禁」**。
处置同 F3：继续收窄 needle 到**产物标识**（`gap-ledger` / `缺口账.json` / `PHASE6-F3`）
与**能表达数据流的语句**（`import gap_ledger` / `from build_gap_ledger`）；
裸模块名被移除 —— 它同时是「脚本文件名」和「产物名」，**无法区分两者**。
并补**反向探针**：只提脚本名的散文**不得**判红（2 种写法实测 0 误报）。
> 这条纪律（**豁免条款必须比拦截条款测得更严**）是 F3 留下的，F5 第一次真的用上。

### 门禁缺陷 #13 同族**第三次**

`build_capability_graph.py` 有两处 `relative_to(REPO)`，F3 修的时候没扫到它。
本次统一走 `_rel()`，并在 `--selftest` 里补 **3 个探针**（仓库内 / `/tmp/` 仓库外 / 前缀相似的仓库外路径）。
⇒ 该族从「逐条修」升级为「一族的通用写法 + 探针守卫」。

---

## 6. 本轮被自己实测推翻/修正的四条

1. **「53 张与产品侧重叠」是错的** —— 实测**零重叠**，故 `reclassify` 机制本轮一次都没用上（但仍被自检锁定）。
2. **F5 点名的「04-供应链/Skill-Monodense」查不到该 id** —— 真名是 `Skill-Monodense-单品价格弹性估计`；
   而且它在产品侧**已分类为 `价格敏感性`** ⇒ 这 5 条错位是**技术域错位**，不是 L3 错位。
3. **锚点扫描首版量的是章节标题**（见 §4.2）。
4. **frontmatter 的 `#` 注释被误判为「没解析出来」**（12 条假阳性，见 §1.4）。

---

## 7. 复算命令

```bash
# F5：卡端分类（0/1/2）
python3 paper2skills-research/scripts/build_card_classification.py --check
python3 paper2skills-research/scripts/build_card_classification.py --selftest   # 12 用例
# 图谱（含卡↔L3 换源）
python3 paper2skills-research/scripts/build_capability_graph.py --check
python3 paper2skills-research/scripts/build_capability_graph.py --check-taxonomy
python3 paper2skills-research/scripts/build_capability_graph.py --selftest
# 缺口账（随动）
python3 paper2skills-research/scripts/build_gap_ledger.py --check
python3 paper2skills-research/scripts/build_gap_ledger.py --selftest
# F4：轴（产品侧）
cd /Users/lute/project/Magpie-Horch/packages/capabilities/dsh-paper2skills
node scripts/check-axis.mjs                       # 真实 vault
node scripts/check-axis.mjs --chain Skill-Agentic-Catalog-Enrichment
node --test test/axis.spec.mjs                    # 17 用例 / 9 变异
```

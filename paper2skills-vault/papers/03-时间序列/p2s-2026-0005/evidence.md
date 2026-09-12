# evidence.md — p2s-2026-0005

> 本文件是 `Skill-Shipping-Cost-Estimation.md` 的**证据档案**。
> 卡片 ⑥ 段是给读者的引用；本文件是给审核者的**核验记录**（含工具、命令、结果、未采信项）。

## 1. 论文元数据

| 项 | 值 |
|---|---|
| paper_id（registry） | `p2s-2026-0005` |
| arXiv | `2607.16230` |
| 标题（原文） | RouteCost: A Production-Inspired Multi-Stage Framework for Pre-Order Shipping Cost Estimation in E-Commerce |
| 作者单位 | Northeastern University（Boston；College of Engineering / Khoury / College of Professional Studies） |
| venue | **arXiv preprint**（无会议/期刊标记） |
| venue_tier | `preprint` |
| evidence_grade | `A`（全文可得，LaTeXML HTML 转换） |
| 全文存档 | `paper2skills-vault/papers/03-时间序列/p2s-2026-0005/fulltext.md`（28,793 字符，来自 arXiv LaTeXML HTML v1） |
| 抓取工具 | `paper2skills-research/scripts/fetch_fulltext.py` |
| 数据集性质 | **合成数据集**（§6 自承 "operationally grounded synthetic dataset"），费率卡为 "simplified FedEx-like" 合成结构（§4） |
| 代码公开性 | **无公开代码**（全文无仓库链接；registry 备注「摘要未说明代码」） |
| 卡片的代码来源 | 本卡 ③ 段是按论文 §3.3 的功能分解**自行实现**的业务化简化版，**不是官方实现，也不是复现** |

**venue 判定的直接证据**：全文头部只有一行
`Conference: Conference Title; 2026; TBDCCS: Computing methodologies ...` ——
`Conference Title` 与 `TBDCCS` 都是**未填的模板占位符**，不构成 venue 证据。
按 `venue-whitelist.md` 的 R3 规则记 `arXiv preprint` / `preprint`。

## 2. 核验记录

| 门禁 | 命令 | 结果 |
|---|---|---|
| K1 代码可执行 | `python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <卡片>` | ✅ **PASS**（3 块拼接为 1 单元；L1 语法 / L2 编译 / L3 import / L4 执行 / L5 断言 全绿；8 个 `test_*`） |
| 引文逐字核验 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <卡片>` | ✅ **VERBATIM 36/36**，0 近似，0 伪造，0 拼接 |
| 引文核验器自检 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest` | ✅ 能区分真引文 / 伪造 / 拼接（见 §5） |
| K2 · G1 代码可执行 | `python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <卡片> --k1 <K1 JSON>` | ✅ **passed: true**，红灯 0，黄灯 0 |
| K2 · G2 事实可溯源 | 同上 | ✅ **passed: true**，红灯 0，黄灯 2 |
| K2 · G3 业务可落地 | 同上 | ✅ **passed: true**，红灯 0，黄灯 0 |

> ⚠️ **G1 必须带 `--k1`**：`gate_check.py` 在没有 K1 产物时会直接判
> `G1-NO-EVIDENCE` 红灯（这是设计如此 —— 禁止凭人工判断放行）。
> 本卡的 K1 凭证由第一条命令现场产出后传给 `--k1`。

**G2 的两个黄灯**（均为非阻塞、已人工确认）：

| 黄灯 | 内容 | 处理 |
|---|---|---|
| `G2-UNSOURCED-GENERAL` ×2 | 一般数字 `12` 无出处 | 来自 frontmatter 的 `created: 2026-09-12` / `updated: 2026-09-12` —— **文档日期，不是事实断言**，模板要求必填，不构成断言 |
| （已消除）`G2-CN-NUMERAL-CLAIM` `十倍` | 初稿 ①b 写过「批发价差几十倍」 | **门禁抓出一处真实问题**：论文只说 "very different wholesale costs"，未给量级 → 已改写为「差得再多（论文未报告批发价差异的量级）」 |

G2 关键指标（`gate_g2` JSON）：`metric_numbers=3, sourced=3, unsourced_metric=0,
traceability_pct=100.0, quotes_total=36, quotes_verbatim=36, quotes_fabricated=0,
quotes_spliced=0, sourced_structural_only=0, fulltext_archived=true`。

## 3. 引文清单（逐字，与卡片 ⑥ 段一一对应）

> ⚠️ **本表是索引，不是权威副本。** `quote_check.py` 只把 `> 原文："..."` 形式的块当作引用块，
> 因此下表的表格单元格形式**不参与**自动核验。权威副本在卡片 ⑥ 段（那 36 条已逐字核验 VERBATIM）。
> 改引文请改卡片 ⑥ 段，再回来同步本表；只改本表不会被任何门禁发现。

| # | 章节 | 引文（逐字） |
|---|---|---|
| 1 | §Abstract | We propose RouteCost, a production-inspired multi-stage framework that decomposes the problem into time-aware demand forecasting, fee-card-informed baseline pricing, Stage 2 residual correction, and proxy-based box-consolidation inference. |
| 2 | §Abstract | In practice, shipping cost is shaped not only by distance but also by destination demand mix, billable weight, dimensional pricing, surcharge triggers, and latent operational effects such as shipment consolidation. |
| 3 | §Abstract | Across over 250,000 orders, 260 products, and 18 months of order history, the framework improves predictive quality and aggregate calibration while preserving route-level interpretability. |
| 4 | §1 | In small-parcel environments, this task is shaped not only by geographic distance, but also by carrier pricing rules, dimensional billing, destination-specific demand patterns, and operational effects that may not be directly observable at prediction time (Bogyrbayeva et al., 2024; Mangiaracina et al., 2019). |
| 5 | §1 | For example, larger products often also have higher wholesale cost, so wholesale cost and parcel cost can appear positively correlated across categories such as stools, chairs, sofas, and beds. |
| 6 | §1 | Two rugs with identical shipment dimensions but very different wholesale costs—for instance, a low-cost rug and a hand-knotted Persian rug—should incur essentially the same shipping cost. |
| 7 | §1 | We make three contributions: (1) we formulate pre-order shipping cost estimation as a route-weighted expectation problem; (2) we introduce a multi-stage architecture that separates demand forecasting, baseline pricing, residual correction, and proxy-based box-consolidation inference; and (3) we show through temporal backtesting and route-level decomposition that structured decomposition improves both predictive quality and interpretability. |
| 8 | §2 | Consolidation has long been recognized as an important logistics strategy because combining multiple shipments can reduce transportation cost and exploit scale economies (Hall, 1987; Wei et al., 2021). |
| 9 | §3.1 | RouteCost contains four modules: (1) time-aware demand forecasting, which estimates route weights by forecasting destination-zone shares over time; (2) fee-card-informed Stage 1 pricing, which produces a structured baseline estimate using billable weight, dimensions, rate-card lookups, and surcharge features; (3) Stage 2 residual correction, which captures nonlinear error not explained by the baseline; and (4) proxy-based box-consolidation inference, which estimates latent savings associated with likely consolidation opportunities. |
| 10 | §3.2 | Under our simplified fulfillment setting, we set Boston as the single in-stock warehouse and all the order shipment originates from Boston and deliver through a single parcel carrier (FedEx). |
| 11 | §3.2 | In practice, these destination zones are obtained from the FedEx zone locator: after specifying Boston ZIP 02108 as the shipping origin, each destination ZIP code is assigned a zone index ranging from 1 to 8. |
| 12 | §3.2 | More generally, if the fulfillment system includes multiple warehouses and multiple carriers, the route space expands beyond zones alone to a cross-product of destination regions, warehouse choices, and carrier choices. |
| 13 | §3.3 | Rather than assuming a fixed destination distribution, we construct monthly zone shares and smooth them using short rolling windows. |
| 14 | §3.3 | Because destination patterns vary across product groups, we also estimate category-specific monthly zone shares and apply a hierarchical fallback strategy during inference. |
| 15 | §3.3 | Carrier rate-card information remains an important signal, but the final baseline estimate is learned through a regularized Ridge regression model that integrates billable weight, physical weight, dimensional weight, package dimensions, destination zone, synthetic rate-card lookup values, and surcharge-related flags. |
| 16 | §3.3 | Stage 2 then learns a nonlinear residual correction over the Stage 1 output using gradient boosting. |
| 17 | §3.3 | Finally, box consolidation is inferred from weak operational proxies such as same-day same-ZIP density, same-day same-zone density, family-level average quantity, size compatibility, package split risk, and a composite consolidation opportunity score. |
| 18 | §3.3 | Because the final estimate is computed as the sum of route-level weighted contributions, the model can be audited at both the prediction level and the route-composition level. |
| 19 | §3.3 | This separation reduces the need for a single model to jointly absorb pricing rules, demand patterns, and hidden operational effects in one end-to-end mapping. |
| 20 | §4 | The dataset contains 250,000 order records, 260 products, and 18 months of transaction history. |
| 21 | §4 | The synthetic pricing layer is built from a simplified FedEx-like rate-card structure indexed by destination zone and billable weight. |
| 22 | §4 | We use a time-based split in which earlier months are used for training and the final three months are reserved for holdout evaluation. |
| 23 | §4 | Importantly, this chronological split is used for backtesting rather than final deployment; once evaluation is complete, the production-facing version of the model can be refit on the full available history so that the most recent demand and surcharge patterns are retained. |
| 24 | §4 表 1 | `\| Orders \| 250,000 line-item orders \|` … `\| Evaluation \| Time-based holdout; MAE, MAPE, aggregate error \|`（原文为多行表格，卡片中按单行摘录，行序与原文一致） |
| 25 | §5.1 | Adding Stage 2 improves the estimate by correcting nonlinear residual error, while the full framework further benefits from the inferred consolidation effect. |
| 26 | §5.1 | Overall, the full model achieves the best holdout performance, indicating that each additional stage contributes useful information beyond the structured baseline. |
| 27 | §5.1 表 2 | `\| Stage 1 \| 1.735 \| 0.070 \|` + `\| Stage 1 + Stage 2 \| 1.893 \| 0.078 \|` + `\| Full Model \| 1.649 \| 0.064 \|`（原文为多行表格，卡片中按单行摘录，行序与原文一致） |
| 28 | §5.2 | This metric is especially useful because a model with acceptable per-order accuracy can still generate biased monthly totals, which would be problematic for pricing, budgeting, and gross-margin planning. |
| 29 | §5.2 | As shown in Figure 6, the full model remains close to zero aggregate bias in most months, with fluctuations contained within a narrow range of roughly -0.14% to +0.06%. |
| 30 | §5.3 | In practical terms, route cost and route contribution are not identical: some far-away zones contribute strongly because route costs are high even when demand share is modest, while some nearby zones dominate mainly because demand is concentrated there. |
| 31 | §5.3 | This route-weighted interpretation is especially valuable for downstream pricing discussions because it ties expected cost back to both operational demand patterns and pricing mechanics. |
| 32 | §6 | The evaluation uses an operationally grounded synthetic dataset and a simplified fulfillment network with a single origin, single carrier, and single service level, which reduces route complexity. |
| 33 | §6 | In addition, the consolidation target is inferred through pseudo-label construction rather than learned from true shipment-level consolidation outcomes. |
| 34 | §6 | In a multi-warehouse setting, maintaining a separate zone map for each origin would not scale well. |
| 35 | §6 | A further advantage of this decomposition is robustness under distribution shift. |
| 36 | §6 | In such settings, a modular design supports targeted refresh of the affected components: demand models can be updated when destination mix changes, while pricing logic or rate-card inputs can be revised when carrier pricing rules change. |

## 4. 数字台账（正文每个数字 → 出处）

G2 的剥离规则：代码围栏与 `> 原文：` / `> 出处：` 两类证据语法行**不参与**扫描。
因此下表只列**正文散文**里的数字（含 frontmatter）。逐项实测（把 `gate_check`
的数字抽取器直接跑在卡片上）：

| 数字（正文出现处） | 类别 | 出处 |
|---|---|---|
| `250,000`（② 数据要求 / ③ 警示块） | 论文事实 | ⑥ #3 §Abstract、#20 §4、#24 表 1 |
| `260`（② / ③ 警示块） | 论文事实 | ⑥ #3 §Abstract、#20 §4、#24 表 1 |
| `18`（② 数据要求「≥18 个月」/ ③ 警示块） | 论文事实 | ⑥ #3 §Abstract、#20 §4、#24 表 1 |
| `8`（② 业务价值「8 个分区的加权」） | 论文事实（结构） | ⑥ #11 §3.2「zone index ranging from 1 to 8」、#24 表 1「8 destination zones」 |
| `1.735` / `1.893` / `1.649`（③ 警示块） | 论文事实 | ⑥ #27 §5.1 表 2 |
| `1.735` / `1.893`（③ 警示块「方向差异」句） | 论文事实 | 同上 |
| `6`（② 场景一标题「旺季前 6 周」） | **业务动作参数**（非论文事实） | 无论文出处；属 v2 模板要求的"具体动作时点"，与 ⑥ #29 的 Figure 6 季度图中数字 `6` 同形但**不是同一含义**，不构成引用 |
| `12`（frontmatter `created` / `updated`） | 文档日期 | 非事实断言（G2 黄灯，已人工确认） |
| `2607.16230`（frontmatter `paper_id` / ⑥ 出处行） | 标识符 | arXiv ID，结构指代，不构成断言 |
| `2606.…` 等 | — | 卡片未出现其他论文 ID |

**结论**：正文中**没有任何"带度量语义"的数字缺少出处**（`unsourced_metric = 0`，`traceability_pct = 100`）；
唯一两个无出处的一般数字是 frontmatter 的文档日期。
③ 段的演示输出数字（订单量、MAE、单件运费等）**全部位于 `text` 输出围栏内**，按 MasterPrompt v2.1 的
「B 类 · 本地可复现数字」约定不计为事实断言 —— 它们可自行运行验证。

## 5. 引用核验器为何可信（不是"加了个检查"而已）

`quote_check.py --selftest` 的四个用例（实测输出）：

| 用例 | 结果 |
|---|---|
| 真引文（论文原句） | ✅ VERBATIM，连续度 1.0 |
| 纯伪造（术语对但句子不存在） | ✅ FABRICATED，连续度 0.124 |
| **拼接**（摘要句 + 引言句缝成一句） | ✅ FABRICATED，连续度 0.681，**n-gram 召回却是 1.0** |
| 真引文含排版差异 | ✅ VERBATIM，连续度 1.0 |

第 3 行是关键：该核验器最初用「全文档 n-gram 覆盖率」判定，而**拼接引文的覆盖率是 1.0**
（每个碎片都能在文档某处找到）→ 拼接可以通过。改为「最长**连续**匹配段占引文的比例」后才拦得住。
这正是 `Cited but Not Verified`(arXiv:2605.06635) 所指的失效模式：
**「有引用」与「引用为真」是两件事**。

本卡的 36 条引文的 `longest_run_ratio` **全部为 1.0**（整句连续命中），`n_spliced = 0`。

**引文的生产方式**：全部从 `fulltext.md` **程序化抽取整句**（按锚点定位 → 向前/向后找句末终止符），
再回跑 `quote_check.check_one()` 逐条验证，**没有一条是凭记忆手打的**。
表 1 / 表 2 的引文直接取原文连续子串（原文行间有空行，`normalize()` 会把空白折叠成单空格，
故单行摘录仍逐字命中），并在卡片中显式注明"原文为多行表格，按单行摘录，行序与原文一致"。

## 6. 未能核验 / 有意保留为定性的项

以下各项**论文未报告**，卡片中一律未编造，或改写成定性表述：

| # | 未报告项 | 卡片如何处理 |
|---|---|---|
| 1 | **货币单位与 MAE 量纲** | 卡片写「论文未讨论」，并明确本模板的费率卡数值单位（元/票、元·kg⁻¹）是**示例值** |
| 2 | **与纯查表 / 黑箱模型的对照数字** | §6 只有定性声称"费率卡基线比纯查表或无约束黑箱更可靠"，**没有给任何对照数字** → 卡片不引用任何"优于 X%"式断言 |
| 3 | **Ridge 与梯度提升的超参数** | 未报告 α、树数、深度、学习率 → 卡片 ③ 段自设默认值并注明是自设 |
| 4 | **分品类 / 分分区的误差分解** | 只有整体 MAE/MAPE 三个变体 → 卡片不做任何分层误差断言 |
| 5 | **Stage 1 + Stage 2 的 MAE 为何高于 Stage 1** | 表 2 显示 1.893 > 1.735，而 §5.1 正文写 Stage 2 "improves" —— **论文未解释该不一致**，卡片据实并列并在 ①b 标为失败模式，不替论文圆场 |
| 6 | **显著性检验 / 置信区间** | 未报告 → 卡片不做显著性断言 |
| 7 | **费率卡的具体数值**（分区费率、附加费金额、体积重除数） | 未报告 → 卡片 ③ 段的费率卡全部标注为示例值 |
| 8 | **数据集本身** | 合成数据集，未给生成脚本或下载链接 → 卡片 ③ 段自带数据生成函数，可独立复现 |
| 9 | **装箱合并的真实结果** | 论文自承是伪标签构造（⑥ #33）→ 卡片把"合并节省"明确定位为**估计的区域性下调**，而非逐单可对账的折扣 |
| 10 | **多仓 / 多承运商下的路由表示** | §6 只给了方向（共享区域划分、仓 × 区域），**未实现也未验证** → 卡片在 ①b 明确写"不要直接套用" |
| 11 | **头程（工厂 → 海外仓）** | 论文全文未涉及头程/关务/入仓预约 → 卡片在 ①b 显式划出边界，说明必须重做路由定义，**不得**用本卡费率卡逻辑估头程 |
| 12 | **贵司侧的 ROI 参数** | 论文无法提供 → 卡片 ⑤ 段只给公式与参数来源，**不填任何金额**（填入即违反 R1） |

## 7. 与 registry 记录的核对

registry（`papers_registry.json`）对本篇的记录：`venue: ""`、`venue_tier: "preprint"`、
`data_availability: "available"`、`note: "无 venue；摘要未说明代码"`、
`decision_reason: "RouteCost：运费成本预估四段式（需求预测→费率卡→残差→装箱合并推断），直连毛利率与定价"`。

| registry 断言 | 核验结论 |
|---|---|
| 四段式：需求预测 → 费率卡 → 残差 → 装箱合并推断 | ✅ 属实，§3.1（⑥ #9 逐字）；§Abstract 与 §7 结论同样表述 |
| **直连毛利率与定价** | ✅ 属实：§5.2 明确说月度总额有偏会 "problematic for pricing, budgeting, and gross-margin planning"（⑥ #28）；§5.3 说路由加权解释 "especially valuable for downstream pricing discussions"（⑥ #31） |
| 无 venue | ✅ 属实：全文头部仅有未填的模板占位符（见 §1） |
| 摘要未说明代码 | ✅ 属实，且全文亦无仓库/数据链接 → 本卡按**无公开代码**处理 |
| `venue_tier: preprint` | ✅ 一致 |
| `data_availability: available` | ⚠️ **需澄清**：该字段指的是**论文全文可得**（本卡 `evidence_grade: A` 成立）；但论文的**数据集本身并未公开**——它是合成数据集，且未给生成脚本或数据链接。若把 `available` 读成"数据可得"，会误导落地方。本卡 ② 段的「数据可得性」按**贵司自有数据**独立判断，未沿用该字段 |

**结论：registry 的技术性判断与原文一致，无需修正；仅 `data_availability` 的语义需按上表澄清。**

## 8. 时效性（R4 检查）

- registry 记 `published: 2026-06-24`（arXiv v1）；全文正文只标 `2026`。
- 首次公开距今 < 3 个月，符合 `venue-whitelist.md` 的时效基准，不存在
  「顶刊新发表但方法陈旧」问题（该问题在 OpenAlex 统计口径上影响甚广）。
- R4 硬拦截清单逐条核对：非撤稿；有数据集与三变体基线对比（非纯理论）；
  非 survey / meta-analysis / position paper；非纯 benchmark 报告 / demo / workshop 短文；
  未声称是主会（本卡按 preprint 标注）；全文**无任何针对 LLM 的提示注入指令**。
  → **未命中任何一条硬拦截**。

## 9. 本卡暴露的一处真实缺陷（门禁抓出的）

初稿 ①b 写了「两张尺寸完全相同的毯子，**批发价差几十倍**，运费几乎一样」——
论文原文只说 "very different wholesale costs"，**从未给过任何量级**。
这句被 `gate_check.py` 的 `G2-CN-NUMERAL-CLAIM` 以黄灯报出
（中文数字量级断言无法自动核验，脚本要求人工确认），人工复核后判定为**自己编的数量级**，已改写为
「批发价差得再多（论文只说 "very different wholesale costs"，**未报告批发价差异的量级**，本卡不补）」。

> 这条黄灯的价值在于：**它是一个"看起来无害"的修饰词**——不写数字、不带单位，
> 若不是门禁把中文数字量级单独拎出来，这类断言会完全绕过数字检查。
> 与 CLAUDE.md 记录的「中文数字不匹配」漏洞（写成"两三倍"即绕过全部数字检查）是同一类失败模式的变体。

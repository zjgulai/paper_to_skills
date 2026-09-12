# evidence.md — p2s-2026-0009

> 本文件是 `Skill-Supply-Network-Simulation.md` 的**证据档案**。
> 卡片 ⑥ 段是给读者的引用；本文件是给审核者的**核验记录**（工具、命令、结果、以及未采信项）。

## 1. 论文元数据

| 项 | 值 |
|---|---|
| paper_id（registry） | `p2s-2026-0009` |
| arXiv | `2607.09745`（v1） |
| 标题（原文） | SupplyNetPy: An Open-Source Python Library for High-Fidelity Modeling and Simulation of Arbitrary Supply Chain and Inventory Networks |
| 作者 / 单位 | Tushar Lone、Neha Karanjkar —— Indian Institute of Technology Goa（School of Mathematics and Computer Science） |
| venue | **Winter Simulation Conference 2026**（作者自述为 accepted paper 的作者 preprint） |
| venue_tier | 本卡记 `CCF-B`；registry 记 `second`，按 venue 白名单口径归一化，差异见 §7 |
| registry `published` | 2026-07-03 |
| registry `data_availability` | `available`（`note`：Winter Simulation Conference 2026；**有开源实现**） |
| 全文存档 | `paper2skills-vault/papers/04-供应链/p2s-2026-0009/fulltext.md`（390 行 / 43,959 字节，来自 `https://arxiv.org/html/2607.09745v1` 的 LaTeXML HTML） |
| 抓取工具 | `paper2skills-research/scripts/fetch_fulltext.py`（**未**重新下载，直接使用既有存档） |

## 2. 核验记录

| 门禁 | 命令 | 结果 |
|---|---|---|
| 引文逐字核验 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <卡片>` | ✅ **VERBATIM 38/38**，0 近似，0 伪造，0 拼接 |
| 引文核验器自检 | `python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest` | ✅ 四个用例（真引文 / 伪造 / 拼接 / 排版差异）全部按预期判定，见 §5 |
| K1 代码可执行 | `python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <卡片> --level 5` | ✅ **PASS**（3 个代码块拼成 1 个单元，L1 语法 / L2 编译 / L3 导入 / L4 执行 / L5 pytest 全绿，7 个 `test_*` 通过） |
| K2 · G1 代码可执行 | `python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <卡片> --k1 <K1 JSON>` | ✅ `"passed": true`，红灯 0，黄灯 0 |
| K2 · G2 事实可溯源 | 同上 | ✅ `"passed": true`，红灯 0；黄灯 2（均为 frontmatter 日期里的 `12`，属日期串非事实断言） |
| K2 · G3 业务可落地 | 同上 | ✅ `"passed": true`，红灯 0，黄灯 0（母婴出海具体信号 13 个、空泛表述 0 处、数据可得性已声明、ROI 有公式） |

G2 的关键指标（来自 `gate_check.py` 的 metrics）：`metric_numbers: 10`、`sourced: 10`、
`unsourced_metric: 0`、`traceability_pct: 100.0`、`quotes_verbatim: 38`、`quotes_fabricated: 0`、
`sourced_structural_only: 0`、`cn_numeral_claims: 0`。

## 3. 引文清单（逐字，与卡片 ⑥ 段一一对应）

每条均已在全文底本中**连续命中**：`quote_check.py` 报告的最长连续匹配段占比 = 1.0（判 VERBATIM）。
引文由脚本从 `fulltext.md` 按锚点**切片**生成（见 §8"卡片是怎么写出来的"），不是手抄。

按主题索引：

- **venue（作者自述）** → #37
- **能力声明与网络模型** → #1, #28, #27, #34
- **易腐库存** → #3, #6, #30, #31
- **节点中断** → #4, #8, #33
- **随机需求与提前期、策略与 API** → #29, #5, #7
- **安装方式、仓库与发布历史** → #2, #10, #38, #9, #35, #36
- **验证：解析基准** → #11, #12, #13, #14, #15, #16
- **验证：商业工具逐组件比对** → #21, #22, #23
- **验证：已发表案例研究复现** → #17, #18
- **性能** → #19, #20, #26
- **论文自承的局限** → #24, #25, #32

完整清单：

| # | 章节 | 引文（逐字） |
|---|---|---|
| 1 | §Abstract | It supports multiple replenishment policies, perishable inventory, node disruptions, and stochastic demand and lead times. |
| 2 | §1.1 | It is installable from the Python Package Index via pip install supplynetpy with detailed documentation, user guides and full examples at https://supplychainsimulation.github.io/SupplyNetPy, and source code on GitHub [8]. |
| 3 | §1.1 | Perishable inventory is supported with per-unit expiry tracking using a first-in, first-out (FIFO) discipline and waste cost accounting, making the library attractive for use cases such as food SCs, milk distribution networks, cold chains for vaccines, and pharmaceutical SCs. |
| 4 | §3.1（Node 类条目） | Supports stochastic disruption modeling via a configurable failure probability (failure_p) and Python callables for disruption duration and recovery time. |
| 5 | §3.1（Link 类条目） | Attributes include source node, sink node, transportation cost, and lead time (specified as a Python callable for deterministic or stochastic lead times). |
| 6 | §3.1（Inventory 类条目） | For perishable inventory, each unit is tagged with its manufacture date and shelf life, and the expired items are removed using a FIFO discipline. |
| 7 | §3.2 | When a node has multiple upstream suppliers, a supplier selection strategy determines which supplier fulfills each replenishment order. |
| 8 | §3.3 | Nodes are configured with failure probability and callable disruption and recovery durations. |
| 9 | §3.5 | The library was subsequently expanded, improved, and thoroughly validated [10], and was released publicly on GitHub under an MIT license in 2025 as SupplyNetPy. |
| 10 | §4.2 | The full model configurations, tested parameter ranges, and numerical results of this validation study are available here: https://github.com/SupplyChainSimulation/SupplyNetPy/tree/main/validation. |
| 11 | §4.1（Newsvendor problem 条目） | We swept $Q$ from 10 to 200, ran 1,000 simulation replications at each value, and verified that the simulated profit curve peaks at $Q\approx 110$, matching the analytical optimum $Q^{*}\approx 110$. |
| 12 | §4.1（Newsvendor problem 条目） | At this optimum the mean simulated profit is $278.2$ with a 95% CI of $[273.5,282.8]$. |
| 13 | §4.1（Economic Order Quantity 条目） | Over a long horizon (4,000 days), the simulated cost-minimizing lot size was approximately 1,010 units, within 3% of the analytical value. |
| 14 | §4.1（Economic Order Quantity 条目） | Because the EOQ total-cost curve is very flat near its optimum (the cost penalty at 1,010 versus 980 units is under 0.1%), this agreement validates the (R, Q) replenishment policy implementation and cost tracking. |
| 15 | §4.1（Safety stock estimation 条目） | Over 100 simulation replications, the estimated safety stock level was 1,346.8 units (95% CI $[1{,}335.9,1{,}357.6]$), the average inventory level was 6,311.7 units (95% CI $[6{,}303.1,6{,}320.4]$), and the average order flow time was 13.81 days (95% CI $[13.79,13.83]$). |
| 16 | §4.1（Safety stock estimation 条目） | The simulated safety stock and average inventory lie above their analytical values of 1,000 and 6,000 units, whereas the flow time lies below its analytical value of 2.4 weeks (16.8 days). These expected deviations arise because resampling negative demand realizations (an artifact of the normal approximation) makes the effective demand higher than the nominal normal demand assumed analytically, pushing the two inventory measures above and the flow time below their respective analytical values. |
| 17 | §5.1（Implementation in SupplyNetPy 段） | An exhaustive grid search over $(s,S)$ was performed, running 1,000 replications per parameter combination (chosen based on a convergence analysis of the standard error on mean daily cost). |
| 18 | §5.1（Results 段） | Further, the optimal region identified by SupplyNetPy is consistent with the original study. |
| 19 | §5.2 | Execution time grows linearly with simulation length, with tight median confidence intervals throughout. |
| 20 | §5.2 | Scaling with $N$ is close to linear, and the per-event cost grows only gradually as $N$ increases, so SupplyNetPy introduces no super-linear overhead from component management or statistics collection. |
| 21 | §4.2（Results Summary） | For most deterministic model configurations (non-stochastic demand and constant lead times), SupplyNetPy and AnyLogistix produced identical results across all tracked metrics for both single-echelon and two-echelon configurations over long simulations. |
| 22 | §4.2 | We found a few pathological configurations where results differed even though both models were deterministic and the implementation was correct. |
| 23 | §4.2（A） | Each tool resolves such ties deterministically but differently: SupplyNetPy uses an event ID assigned at event creation time, whereas AnyLogistix uses the order in which processes register events. |
| 24 | §1.1 | The current implementation does not yet support continuous material flows (e.g., pipelines as links), multi-product shared inventory, or advanced logistics features such as fleet management and vehicle routing. |
| 25 | §6 | While the library currently supports both discrete and real-valued inventory, the inventory transport is modeled as discrete events. |
| 26 | §5.2 | Replications are independent and were executed in parallel on a dual-socket Intel Xeon Gold 6148 server (two 20-core CPUs at 2.40 GHz, 80 logical cores) running 64-bit Linux. |
| 27 | §2 | There is a dearth of well-maintained, well-documented open-source libraries specifically targeted for the discrete-event simulation (DES) of arbitrary SC networks. |
| 28 | §3.1 | The graph need not be a tree; an inventory node can be connected to multiple upstream manufacturers or suppliers, with a configurable supplier selection policy to choose among them. |
| 29 | §3.3 | Demand arrival times, order quantities, and link lead times are all specified as Python callables, enabling any distribution (Poisson, normal, empirical, or user-defined). |
| 30 | §5.1（Problem description） | The pharmacy stocks a perishable drug with a finite shelf life. Expired units are discarded and unmet demand is lost. |
| 31 | §5.1（Results） | The independent replication of this case study’s results confirms that SupplyNetPy correctly models perishable inventory with FIFO expiry, stochastic demand, probabilistic supply disruptions, and the associated cost structure. |
| 32 | §6 | Key features include: (i) support for modeling perishable inventory, (ii) node disruption modeling for resilience assessment, (iii) several types of built-in replenishment policies and supplier selection strategies, all extensible via class inheritance, and (iv) a dual API enabling both rapid functional-style prototyping and deep customization. |
| 33 | §4.2 | There is no single universally correct scheme for simultaneous-event handling in DES, and both tools behave correctly according to their own documented semantics. |
| 34 | §1.1 | The library has been thoroughly validated through comparison with analytical benchmarks, against a commercial simulation tool (AnyLogistix) for unit-tests with deterministic configurations, and against published results from a case study. |
| 35 | §3.5 | SupplyNetPy has been under continuous development since 2022. |
| 36 | Acknowledgements | the development of SupplyNet Web, a web-based GUI for SupplyNetPy, available at https://supply-net-web.vercel.app/. |
| 37 | 作者信息栏（作者自述） | This is the author’s preprint of a paper accepted at Winter Simulation Conference, 2026. |
| 38 | 参考文献 [8] | K. Lone (2024) SupplyNetPy github repository. Note: https://github.com/SupplyChainSimulation/SupplyNetPy |

> ⚠️ **本表是索引，不是权威副本。** `quote_check.py` 只把 `> 原文："..."` 形式的块当作引用块，
> 因此上表的表格单元格形式**不参与**自动核验。权威副本在卡片 ⑥ 段（那 38 条已逐字核验 VERBATIM）。
> 改引文请改卡片 ⑥ 段，再回来同步本表；只改本表不会被任何门禁发现。

## 4. registry 断言逐项核对（本卡被要求核对的重点）

registry 的 `decision_reason` 写着：「SupplyNetPy：开源 Python 库，多级网络离散事件仿真，
支持易腐库存/节点中断/随机提前期；落地成本最低」。逐项核对：

| registry 断言 | 核验结论 | 逐字证据（卡片 ⑥ 编号） |
|---|---|---|
| **多级网络离散事件仿真** | ✅ 属实。网络被建模为有向图（节点 + 链路），且**可以不是树**：一个持货节点可挂多个上游并用供应商选择策略决定由谁履约；库基于 SimPy 做事件驱动仿真；文档另有面包房多级网络算例 | #1（Abstract 的四能力总述）、#28（§3.1 图结构）、#27（§2 开源生态缺口） |
| **易腐库存** | ✅ 属实。per-unit 效期追踪 + FIFO + 报废成本记账；复现的案例研究正是"医院药房 + 有限保质期药品 + 过期丢弃" | #3（§1.1）、#6（§3.1 Inventory 条目）、#30、#31（§5.1） |
| **节点中断** | ✅ 属实，且是**节点与链路双通道**：节点有失效概率 `failure_p` 与可调用的中断 / 恢复时长；链路另有链路级失效概率；缺额经未履约订单与缺货向下游传导 | #4（§3.1 Node 条目）、#8（§3.3）、#33（§4.2 同时事件语义） |
| **随机提前期** | ✅ 属实。链路提前期写成 Python callable（确定性或随机皆可）；需求到达时刻与单笔订货量同为 callable，支持任意分布 | #5（§3.1 Link 条目）、#29（§3.3） |
| **有开源实现** | ✅ 属实，但**论文只给"可安装 + 文档站 + 仓库"三类定位，没给版本号**：见 §6 的逐字清单 | #2（§1.1 安装与文档）、#38（参考文献 [8] 仓库地址）、#10（§4.2 validation 目录）、#9（§3.5 MIT / 2025） |
| **落地成本最低** | ⚠️ **部分采信**。这是对"建模仿真这件事"的判断（配置化建模、pip 可装），**不等于**数据侧成本低：批次效期台账与中断样本仍需补齐。卡片 ⑤ 已把难度评分与数据风险分开写 | 无逐字证据（属 registry 判断，非论文陈述） |

## 5. 引用核验器为何可信（`--selftest` 实测）

`quote_check.py` 用四个用例自证能区分真引文 / 伪造 / 拼接：

| 用例 | 结果 |
|---|---|
| 真引文（论文原句） | ✅ VERBATIM，连续度 1.0 |
| 纯伪造（术语对但句子不存在） | ✅ FABRICATED，连续度 0.124 |
| **拼接**（摘要句 + 引言句缝成一句） | ✅ FABRICATED，连续度 0.681，**n-gram 召回却是 1.0** |
| 真引文含排版差异 | ✅ VERBATIM，连续度 1.0 |

第三行是关键：核验器最初用「全文档 n-gram 覆盖率」判定，而拼接引文的覆盖率是 **1.0**
（每个碎片都能在文档某处找到）→ 拼接可以通过。改用「最长**连续**匹配段占引文的比例」后才拦得住。
本卡的 38 条引文全部是**连续切片**，因此 `n_spliced: 0`。

## 6. 包名 / 仓库地址的现实情况（本卡最需要交代清楚的一项）

论文**给出了**开源实现的名字与位置，逐字如下（全部可在 ⑥ 核对）：

| 项 | 论文原文（逐字） | ⑥ 编号 |
|---|---|---|
| 包名与安装方式 | `pip install supplynetpy`（原文：installable from the Python Package Index via …） | #2 |
| 文档站 | `https://supplychainsimulation.github.io/SupplyNetPy` | #2 |
| 源码托管 | "source code on GitHub [8]" | #2 |
| 仓库地址（参考文献 [8]） | `https://github.com/SupplyChainSimulation/SupplyNetPy` | #38 |
| 验证材料目录 | `https://github.com/SupplyChainSimulation/SupplyNetPy/tree/main/validation` | #10 |
| 许可证与发布年 | "released publicly on GitHub under an MIT license in 2025 as SupplyNetPy" | #9 |
| 网页 GUI（仅见于致谢） | `https://supply-net-web.vercel.app/` | #36 |

**三处必须说清的限制**：

1. **论文内部的导入名与包名不一致**：§3.4 的示例代码写的是 `import SupplyNetPy.Components as scm`（混合大小写），
   而 §1.1 的安装命令是全小写的 `pip install supplynetpy`。论文没有解释两者关系。
2. **本卡未联网核实** PyPI 页面与 GitHub 仓库是否存在、是否活跃、当前版本号是多少 ——
   卡片与代码都刻意**没有**联网请求，也没有 `import` 该包（未安装，会被 K1 判 `ORPHAN_DEP`）。
3. **因此卡片 ③ 的代码是自包含重写**（标准库 + numpy），只借用论文的概念结构，
   不是该库的 API，也不能替代它；生产环境应直接用官方实现。

## 7. 未能核验 / 有意保留为定性的项

与卡片末尾"本卡未能核验"清单一致，此处给出更完整的口径：

- **Table 1（工具对比表）逐格内容未采信**：该表在 HTML→Markdown 转换后列结构错位
  （表头与数据行的单元格数不一致），无法可靠对齐列名与单元格。故卡片**不引用** Table 1 的任何单元格，
  只引用正文关于"开源替代品不支持任意网络 / 部分不支持仿真 / 维护停滞"的定性陈述。
- **性能只有标度律、没有绝对值**：论文只报告"执行时间随仿真时长线性增长、随网络规模接近线性"，
  未给单次仿真耗时、每秒事件数或内存占用。卡片因此不引用任何性能绝对值；
  ⑥ 保留其硬件配置（双路 Xeon Gold 6148 / 80 逻辑核 / 64 位 Linux）仅供读者判断适用边界。
- **中断与易腐没有独立单点验证数字**：这两项特性是通过**案例研究复现**（成本曲面对比图）间接验证的，
  没有像 EOQ / 安全库存那样的单点读数。卡片按此做定性表述，未编造数字。
- **图 2 只有图、无数字**：药房案例研究的成本曲面以图呈现，卡片只引用"最优区域与原研究一致"这句原文。
- **页码**：全文存档无页码锚点，⑥ 的出处一律只写到章节或条目，**不写"PDF 第 N 页"**。
- **registry 的 `venue_tier: second` → `CCF-B`**：这是本卡按 venue 白名单口径做的归一化，
  不是论文的说法；论文本身只说 accepted at Winter Simulation Conference 2026。
- **贵司侧的 ROI 参数**（丢单件数、报废件数、单件毛利、单位持有成本、中断窗口）：
  论文无法提供，卡片 ⑤ 只给公式与参数来源，不填具体金额 —— 填具体数字即违反 R1。

## 8. 卡片是怎么写出来的（可复核的构造过程）

1. 引文**不是手抄**：`fulltext.md` 里对每条引文用「起始锚点 + 结束锚点」定位后**切片**，
   构造上即为连续子串；切片结果再注入卡片 ⑥ 段（脚本做法见本文件同目录的操作记录）。
   这样从机制上排除了拼接与凭记忆复述 —— `quote_check.py` 的 `longest_run_ratio` 全为 1.0 即为证据。
2. 代码**先在独立文件里跑通**（7 个 `test_*` 全绿、演示输出由脚本落盘），再按块注入卡片，
   因此卡片里的"实际运行输出"围栏与代码严格同源，不是事后补写的样例。
3. 卡片里的论文事实数字（报童 Q*≈110、EOQ 偏差 3% 与成本差 0.1%、安全库存 1,346.8 对解析 1,000 等）
   全部落在 ⑥ 的逐字引用块内；本模板的模拟输出（满足率、丢单、报废等）全部落在代码 / 输出围栏内并标注
   "合成参数、可复现" —— **两类数字没有出现在同一句话里**。

## 9. 本卡的数字账（论文事实数字 vs 本地可复现数字）

| 数字 | 类型 | 出处 / 位置 |
|---|---|---|
| 报童问题：扫描区间 Q = 10–200、分析最优 Q*≈110、每个取值 1,000 次重复 | A · 论文事实 | §4.1 Newsvendor problem 条目（⑥ #11） |
| 报童问题：最优处平均利润 278.2，95% CI [273.5, 282.8] | A · 论文事实 | §4.1 Newsvendor problem 条目（⑥ #12） |
| EOQ：4,000 天长跑下仿真最优批量约 1,010，与分析解相差 3% | A · 论文事实 | §4.1 Economic Order Quantity 条目（⑥ #13） |
| EOQ：1,010 与 980 的成本惩罚低于 0.1% | A · 论文事实 | §4.1 Economic Order Quantity 条目（⑥ #14） |
| 安全库存：100 次重复、估计值 1,346.8（CI [1,335.9, 1,357.6]）、平均在库 6,311.7（CI [6,303.1, 6,320.4]）、平均订单流时间 13.81 天（CI [13.79, 13.83]） | A · 论文事实 | §4.1 Safety stock estimation 条目（⑥ #15） |
| 安全库存：解析值 1,000 与 6,000，流时间解析值 2.4 周（16.8 天） | A · 论文事实 | §4.1 Safety stock estimation 条目（⑥ #16） |
| 药房案例研究：每个 (s,S) 参数组合 1,000 次重复 | A · 论文事实 | §5.1 Implementation in SupplyNetPy 段（⑥ #17） |
| 持续开发自 2022 起；2025 年以 MIT 许可公开发布 | A · 论文事实 | §3.5（⑥ #35、#9） |
| 性能实验硬件：双路 Xeon Gold 6148（20 核 ×2 / 2.40 GHz / 80 逻辑核）、64 位 Linux | A · 论文事实（仅存于 ⑥，正文未使用） | §5.2（⑥ #26） |
| 卡片 ③ 输出围栏里的满足率 / 丢单件数 / 报废件数 / 报废成本 / 平均在库 | B · 本地可复现（合成参数） | 卡片 ③ 的两处 `text` 围栏；`python3 <本卡代码>.py` 可复现，**与论文数字无关** |

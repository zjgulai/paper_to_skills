# PHASE6 S4 · 缺口驱动检索式（31 条靶区工单 → 三段式检索式）

> 生成器：`paper2skills-research/scripts/build_search_queries.py`　|　复算：`python3 paper2skills-research/scripts/build_search_queries.py --check`
> 机读产物：`paper2skills-research/data/search-queries.json`

**这一步在解决什么**：F3 的靶区工单是一张**缺口清单**，不是一张**采购单**。
没有 S4，缺口账只是一张纸 —— 它说「线索评估 零供给」，但没人知道拿什么词去捞。
S4 把每条工单翻译成一份**三段式检索式**（正向 / 负向 / 约束），
并且这份检索式**能被 `candidate_filter.py` 直接消费**（同一个配置格式，不另造一套）。

**硬顺序**（方案 §8.1）：「S4 必须排在 S1 之后 —— 入卡挂载点就是契约。」
所以每条检索式都带 `responsibility` / `l3` / `rank` / `contract_id`，
**键名与 `check_contracts.py` 认的字段对齐**，产出候选卡之后能挂回那份契约。

---

## 1. 判据清单与实测

| # | 判据 | **反面（这条判据在抓什么错）** | 实测 |
|---|---|---|---|
| J1 | 输入齐备性 | 缺输入却报「通过」——「没东西可查」被当成「查过了没问题」 | ✅ 4/4 输入到位；缺任一 ⇒ exit 2（合成环境下实测 exit 2） |
| J2 | 覆盖率是一等输出：生成数 / 工单数，生成 0 条判红 | 一条都没生成却报「全部通过」 | ✅ 28/31（90.3%）；空输入 ⇒ exit 2 已实测 |
| J3 | 结构空白显式排除 + **两口径**逐个对基线 | 给「走 SOP、不占检索预算」的条目生成检索式；或只对一种口径成立就报过 | ✅ 严 3 / 宽 5 / 账内 9（三个数各自与基线相等） |
| J4 | 可复算：同输入两次运行逐字节相同 | 字典序/时间戳进产物 ⇒ 检索式不可复算；或**改了内容仍报「逐字节相同」**（假绿，以「反向控制：交换词序必须变」兜住） | ✅ 逐字节相同（md5 两次一致）+ 反向控制生效 |
| J5 | **改了工单，检索式必须变**（当作失败条件断言） | 假仪表：改责任名/域/岗位，输出岿然不动（等于所有工单共用一张模板）；反面用「改无关字段必须**不**变」兜住「什么都变」 | ✅ 责任名 1 条 / 域 1 条 / 岗位 1 条发生变化；改无关字段 0 条变化 |
| J6 | 不同工单的检索式彼此不同（相似度分布 + 空壳） | 模板套壳（31 条只换责任名）。四个方向都判：全向量上界、交叉项上界、空壳计数、**空交叉项计数（两个方向）**、同域池条目对上限 | ✅ 交叉项 Jaccard max 0.2（均 0.0025）／全向量 max 0.9118；空壳 0 条；空交叉项 2 条；同域池条目对 4 对 |
| J7 | **负向词按域生效**，不是全局清单 | 无脑全局负向 ⇒ 误杀方法论论文（`trial`/`cohort`/`ad`）；或「按域」只是文档说法、代码里其实全域生效 | ✅ 从 filter **现算**分歧矩阵：56 个词在不同域状态相反，56 个被**真实配对**用上，载体 28 条 |
| J8 | 产物能被 `candidate_filter.py` **直接消费**（同一个配置格式） | 另造一套格式 ⇒ 生成完还要人肉翻译，检索式形同废纸；或域名不在 `DOMAIN_DIR` 里 ⇒ filter **静默跳过**（保留全部 = 没过滤） | ✅ 域名全在 `DOMAIN_DIR`；负向词/约束词与 filter 逐项相等（3 域走 `harvest_domain_of` 桥接，已登记） |
| J9 | 契约挂载点：`responsibility` / `contract_id` 与 `check_contracts.py` 键名对齐 | 自己发明键名 ⇒ 卡产出后无处可挂；或一份没挂上却报过；或同一责任两份正式契约时静默挑一份 | ✅ 28/28 条挂上契约（契约池 139 份） |
| J10 | 排序权重**不进任何判定**（风险 N1） | 把 `zero_gap`/`score` 之类排序分当成放行条件 ⇒ 分数决定谁过闸；反面：全文扫会打中文档注释（假红），故按 AST 取判据函数体 | ✅ AST 扫 11 个判据函数、0 命中；反向探针注入即被抓；反向控制（注入非判据函数**不**报）生效 |

退出码：**0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 内部错误（≠ 判红）**。

---

## 2. 覆盖率是一等输出

- 靶区工单（`worklist`）：**31 条**
- 生成检索式：**28 条**　⇒ **覆盖率 90.3%**
- 结构空白（严口径，账内 flag）：**3 条**被显式排除 ⇒ **可检索位 28**
- 结构空白（宽口径，按岗位 `AGT-009/AGT-010/AGT-011/AGT-012/AGT-013` 读）：**5 条** ⇒ **可检索位 26**
- 账内（151 行）结构空白条目：**9 条**，岗位 ['AGT-009', 'AGT-010', 'AGT-011', 'AGT-012', 'AGT-013']

> ⚠️ **两口径差额（登记，不擅自放宽）**：F3 §3 脚注把这件事写得很清楚 ——
> §4 的结构性空白名单是 **L3 级**的零供给条目；若改按**岗位**读，
> 靶区一里落在 `AGT-009..013` 上的还有 可用性验证、测试方案（它们有 legacy 卡）。
> **差额由所有者裁决，脚本不擅自放宽** —— 本脚本两个数都算、都报，
> 并用 `--check` 把两个数**各自**钉在基线上（任一个变了就判红）。

### 被排除的条目（走 SOP，不占检索预算）

| 位次 | 责任名 | 岗位 | 域 | 口径 |
|---|---|---|---|---|
| 9 | BOM与材料分析 | AGT-011 | — | 严（账内 flag） |
| 10 | 商业案例 | AGT-009 | — | 严（账内 flag） |
| 15 | 研发计划 | AGT-013 | — | 严（账内 flag） |
| 17 | 可用性验证 | AGT-010 | — | **仅宽口径**（有 legacy 卡，账内 flag 为假） |
| 18 | 测试方案 | AGT-013 | — | **仅宽口径**（有 legacy 卡，账内 flag 为假） |

---

## 3. 相似度分布与空壳计数（判据「不同工单的检索式必须彼此不同」）

两两比对数：**378**（其中交叉项可比 **325** 对）

| 口径 | 最小 | 均值 | 最大 |
|---|---:|---:|---:|
| 交叉项（把域宽词收紧到「这条责任」的英文词） | 0.0 | 0.0025 | 0.2 |
| 全向量（域词 + 交叉项） | 0.0 | 0.103 | 0.9118 |

- **最相似的一对（交叉项口径）**：`22:履约异常` ↔ `31:履约跟踪`　Jaccard = **0.2**（全向量 0.3846）
- **最相似的一对（全向量口径）**：`1:线索评估` ↔ `21:达人筛选`　Jaccard = **0.9118**
- **空壳（只换责任名）**：**0 条**（口径：交叉项非空、且被另一条完全包含）
- **空交叉项（取不到材）**：**2 条** —— 1:线索评估、18:测试方案
  - ⚠️ 空交叉项的条目**从交叉项两两比对里剔除**：两条空集的相似度**没有定义**，不是 1.0。第一版把它们算成 Jaccard=1.0，报出一对**不存在**的「只换责任名」假红。
  - 它们仍参与**全向量**比对（域词一样 ⇒ 全向量 Jaccard 高，这个是真实读数）。
  - 两类缺陷**分开报**：空壳 = 套壳；空交叉项 = 没料。**不能被平均成一个数**。
- **同检索域池的条目对**：**4 对** —— 它们本来就该捞同一批论文，全向量 Jaccard 天然高（这是**定义**，不是缺陷）；区分它们的判据是交叉项。逐对：1:线索评估 ↔ 21:达人筛选、3:行动组合 ↔ 14:平台运营、13:供需协调 ↔ 26:订单协调、7:生命周期分析 ↔ 25:退货分流
- 互不相同的全向量：**28 / 28**；互不相同的**非空**交叉项向量：**26 / 26**

> ⚠️ **判据只写单边等于没有判据**（本仓库吃过这个亏：覆盖率 >100% 也照样通过）。
> 所以这里同时给**上界**（最像的一对）与**下界**（空壳条数 + distinct 向量数）；三个方向都进退出码。

---

## 4. 同词异义反例（判据「负向词按域生效」的实测）

本仪器**不维护**一张同词异义清单（清单会腐烂）—— 它每次从 `candidate_filter.py` **现算**：把该域的全部按域负向词在各域下的状态排成矩阵，再挑出**状态相反**的词。

- 参与计算的按域负向词：**67 个**
- 其中**在不同域下状态相反**的：**56 个**
（下表**全部**列出 —— 只列前几条会让「56 个」这个数字无法复核）
- 这些分歧词里，**真的被本次检索式用上**的：**56 个词**（判据：存在一对条目，该词在 A 上判负向、在 B 上不判——两侧都必须有定义）
- 负向段里**真的带上了**至少一个分歧词的检索式：**28 / 28 条**

| 词 | 判负向的条目 | 不判负向的条目 |
|---|---|---|
| `academic promotion` | 行动组合 | 差异追踪 |
| `adhd` | 线索评估 | 渠道对账 |
| `adversarial attack` | 线索评估 | 渠道对账 |
| `adversarial example` | 线索评估 | 渠道对账 |
| `alzheimer` | 线索评估 | 渠道对账 |
| `attrition rate` | 可用性验证 | 测试方案 |
| `biomarker` | 复购实验 | 供应商评估 |
| `biomedical` | 收入与费用核对 | 生命周期分析 |
| `chemistry` | 收入与费用核对 | 生命周期分析 |
| `climate` | 经营预算 | 差异追踪 |
| `clinical note` | 品牌反馈 | 可用性验证 |
| `code generation benchmark` | 渠道对账 | 行动组合 |
| `cohort study` | 账号诊断 | 供需协调 |
| `compiler` | 生命周期分析 | 账号商品映射 |
| `crystal` | 收入与费用核对 | 生命周期分析 |
| `diagnosis` | 账号诊断 | 供需协调 |
| `drug` | 复购实验 | 供应商评估 |
| `drug discovery` | 收入与费用核对 | 生命周期分析 |
| `embodied agent` | 账号商品映射 | 账号诊断 |
| `employee turnover` | 可用性验证 | 测试方案 |
| `epidemiolog` | 账号诊断 | 供需协调 |
| `faculty` | 行动组合 | 差异追踪 |
| `fake news` | 品牌反馈 | 可用性验证 |
| `friend recommendation` | 线索评估 | 渠道对账 |
| `gene` | 复购实验 | 供应商评估 |
| `gui agent` | 渠道对账 | 行动组合 |
| `hate speech` | 品牌反馈 | 可用性验证 |
| `health promotion` | 行动组合 | 差异追踪 |
| `image classification` | 渠道对账 | 行动组合 |
| `in vitro` | 差异追踪 | 收入与费用核对 |
| `load forecasting` | 经营预算 | 差异追踪 |
| `machine translation` | 品牌反馈 | 可用性验证 |
| `malware` | 生命周期分析 | 账号商品映射 |
| `molecular` | 收入与费用核对 | 生命周期分析 |
| `movie` | 线索评估 | 渠道对账 |
| `multi-agent reinforcement learning` | 供需协调 | 平台运营 |
| `music recommendation` | 线索评估 | 渠道对账 |
| `news recommendation` | 线索评估 | 渠道对账 |
| `npm` | 生命周期分析 | 账号商品映射 |
| `object detection` | 渠道对账 | 行动组合 |
| `pet scan` | 账号诊断 | 供需协调 |
| `poi recommendation` | 线索评估 | 渠道对账 |
| `protein` | 收入与费用核对 | 生命周期分析 |
| `pypi` | 生命周期分析 | 账号商品映射 |
| `repository` | 生命周期分析 | 账号商品映射 |
| `robot` | 差异追踪 | 收入与费用核对 |
| `robot manipulation` | 账号商品映射 | 账号诊断 |
| `seismic` | 经营预算 | 差异追踪 |
| `speech recognition` | 渠道对账 | 行动组合 |
| `student dropout` | 可用性验证 | 测试方案 |
| `traffic flow` | 经营预算 | 差异追踪 |
| `vulnerability` | 生命周期分析 | 账号商品映射 |
| `weather` | 经营预算 | 差异追踪 |
| `web agent` | 渠道对账 | 行动组合 |
| `wet lab` | 差异追踪 | 收入与费用核对 |
| `wind power` | 经营预算 | 差异追踪 |

### 三组实测反例逐条对账（**登记不重判**）

| 词 | 词库 §1 的记录 | filter 实测 | 判定 |
|---|---|---|---|
| `trial` | 词库 §1：「`trial` 只在 14 域生效」；01/02 域是 RCT 方法论词，**不得**列负向 | 14-用户分析 负向表=**无对应实现** | 🔴 **词库有记录、代码无实现**（登记，不擅自改 filter） |
| `cohort` | 词库 §1：「医学里是队列研究，与 cohort retention 同词不同义」（故 filter 用收窄形式 `cohort study`） | 14-用户分析 负向表=['cohort study'] | ✅ 以收窄形式实现 |
| `ad` | 词库 §2 13 域：「`adversarial attack`（ad 歧义）、ADHD、Alzheimer」（故 filter 用收窄形式 `adversarial attack`） | 13-广告分析 负向表=['adversarial attack', 'adhd', 'adversarial example'] | ✅ 以收窄形式实现 |

---

## 5. 责任名 → 检索域（**显式决策表**，逐条可复核）

为什么不能自动推：`关键词库-v2.md` 的域词条是**英文**，L3 责任名是**中文**。
实测用「中文词 ∩ 英文词」做词面匹配，**31 条全部得 0 分** ——
那个仪器看不见这种对应关系（同本仓库铁律：**判某个东西不存在之前，先问仪器能不能看见它**）。

| 位次 | 责任名 | 检索域（主 / 次） | 选择理由 |
|---:|---|---|---|
| 1 | 线索评估 | `05-推荐系统` / `13-广告分析` | 线索打分与排序 = 推荐/排序问题；B2B 线索优先级含广告/渠道口径。 |
| 2 | 渠道对账 | `09-DataAgent-LLM` / `12-ML基础` | 对账是「多源记录对齐 + 差异定位」，先走数据代理（Text-to-SQL/对账 Agent），再用统计口径判差异显著性。 |
| 3 | 行动组合 | `13-广告分析` / `15-营销投放分析` | 经营动作组合的取舍 = 预算分配/投放效率问题（广告分析主口径）。 |
| 4 | 经营预算 | `03-时间序列` / `15-营销投放分析` | 预算 = 滚动预测（时序主口径）+ 投入产出（营销投放次口径）。 |
| 5 | 差异追踪 | `12-ML基础` / `02-A_B实验` | 差异定位 = 统计检验与异常检测（ML 基础主口径）；「差异是否显著」用实验设计口径。 |
| 6 | 收入与费用核对 | `09-DataAgent-LLM` / `08-知识图谱` | 核对以数据/账目系统直出为主（数据代理主口径）+ 实体对齐（知识图谱次口径）。 |
| 7 | 生命周期分析 | `04-供应链` / `05-推荐系统` | 商品生命周期 = 库存/效期/长库龄（供应链主口径）+ 生命周期阶段与序列建模（推荐系统次口径）。 |
| 8 | 资金预测 | `03-时间序列` / `04-供应链` | 现金流/账期预测 = 时序主口径；应收应付与采购周期挂在供应链次口径。 |
| 11 | 账号商品映射 | `08-知识图谱` / `16-智能体工程` | 账号↔商品↔Listing 的实体映射 = 实体链接与知识图谱主口径 + 工具化映射 Agent（智能体工程）。 |
| 12 | 账号诊断 | `12-ML基础` / `14-用户分析` | 账号健康诊断 = 指标异常检测（ML 基础主口径）+ 卖家行为分析（用户分析次口径）。 |
| 13 | 供需协调 | `04-供应链` / `10-MAS` | 供需匹配 = 供应链主口径；跨主体协同编排（供应计划与到货对齐）用多 Agent 次口径。 |
| 14 | 平台运营 | `13-广告分析` / `15-营销投放分析` | 多平台运营动作同样落在投放与促销口径（广告 + 营销投放）。 |
| 16 | 品牌反馈 | `07-NLP-VOC` / `14-用户分析` | 品牌反馈 = 评论/舆情挖掘（NLP-VOC 主口径），落到用户侧行为时用用户分析。 |
| 17 | 可用性验证 | `02-A_B实验` / `06-增长模型` | 可用性验证 = 任务级实验与统计检验（A/B 主口径）+ 留存/采纳度量（增长模型次口径）。 |
| 18 | 测试方案 | `02-A_B实验` / `10-MAS` | 组合测试/抽样方案 = 实验设计主口径（pairwise/正交表）；用例生成与执行编排用多 Agent。 |
| 19 | 资源情景比较 | `15-营销投放分析` / `04-供应链` | 情景比较 = 投入产出的情景/灵敏度分析（营销投放主口径）+ 资源约束与产能（供应链次口径）。 |
| 20 | 知识产权检索 | `08-知识图谱` / `07-NLP-VOC` | 知产检索 = 实体/关系抽取 + 图检索（知识图谱主口径）+ 文本检索与相似度（NLP-VOC 次口径）。 |
| 21 | 达人筛选 | `05-推荐系统` / `13-广告分析` | 达人筛选 = 匹配与排序（推荐系统主口径）+ 效果归因（广告分析次口径）。 |
| 22 | 履约异常 | `04-供应链` / `06-增长模型` | 履约异常 = 供应链主口径 + 履约失败与流失预警（增长模型次口径）。 |
| 23 | 抽样审计 | `12-ML基础` / `07-NLP-VOC` | 抽样审计 = 统计抽样与检验（ML 基础主口径）+ 单据与凭据文本核对（NLP-VOC 次口径）。 |
| 24 | 质量分析 | `04-供应链` / `02-A_B实验` | 质量分析 = 质量/根因（供应链主口径）+ 抽样与统计判定（实验设计次口径）。 |
| 25 | 退货分流 | `04-供应链` / `05-推荐系统` | 退货分流 = 逆向物流与退货成本（供应链主口径）+ 分单路由排序（推荐系统）。 |
| 26 | 订单协调 | `04-供应链` / `10-MAS` | 订单协调 = 采购/交付协同（供应链主口径）+ 多主体编排（MAS 次口径）。 |
| 27 | 趋势监测 | `06-增长模型` / `03-时间序列` | 趋势监测 = 品类趋势预测（增长模型主口径）+ 时序次口径。 |
| 28 | 复购实验 | `01-因果推断` / `02-A_B实验` | 复购实验 = 增量/因果识别（因果推断主口径）+ 实验设计（A/B 次口径）。 |
| 29 | 供应商评估 | `04-供应链` / `08-知识图谱` | 供应商评估 = 多准则决策与供应风险（供应链主口径）+ 供应链知识图谱（图谱次口径）。 |
| 30 | 到货异常追踪 | `04-供应链` / `16-智能体工程` | 到货偏差 = 供应链主口径 + 实体/轨迹状态追踪的工具化 Agent（智能体工程次口径）。 |
| 31 | 履约跟踪 | `04-供应链` / `09-DataAgent-LLM` | 在途跟踪 = 供应链主口径 + 在途状态查询与报表（数据代理/Text-to-SQL 次口径）。 |

**已装线 `src_domain` → filter 域的桥接**（9 个域在 filter 词汇表之外，本表只登记实际用到的 3 条）：

| 源域 | 折到 | 语义缺口 |
|---|---|---|
| `17-价格优化` | `15-营销投放分析` | 登记，不擅自放宽 |
| `18-物流履约` | `04-供应链` | 登记，不擅自放宽 |
| `23-运营财务` | `03-时间序列` | 登记，不擅自放宽 |

---

## 6. 检索式全文（逐条）

每条包含：**正向词**（三段式的第一段）×  **负向词**（第二段，按域生效）× **约束词**（第三段）+ 契约挂载点。

### 位次 1 · 线索评估 · 零售渠道与B2B拓展（渠道经营）

- 服务性：`A`　|　检索域：`05-推荐系统`、`13-广告分析`
- **入卡挂载点**：`responsibility: 线索评估`　`contract_id: CTR-A-041`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`
- **正向词**（31）：`recommender system`、`large language model`、`sequential recommendation`、`generative recommendation`、`cold-start`、`recommendation`、`advertising`、`auction`、`bidding`、`attribution`、`marketing`、`budget allocation`、`ROI`、`recommendation system`、`collaborative filtering`、`semantic ID`、`cold-start recommendation`、`conversational recommendation`、`reranking`、`session-based recommendation`、`catalog enrichment`、`incrementality`、`geo experiment`、`marketing mix model`、`MMM`、`ROAS`、`sponsored search`、`coupon allocation`、`cannibalization`、`uPlift for advertising`、`ad ranking`
  - ⚠️ **无交叉项**（该责任无 legacy 卡可取材）—— 见 §7 未做项
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`、`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`
  - 仅 `05-推荐系统` 生效：`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`
  - 仅 `13-广告分析` 生效：`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`
- **约束词**（11）：`e-commerce`、`product`、`item`、`user`、`session`、`catalog`、`advertis`、`marketing`、`campaign`、`budget`、`roas`

### 位次 2 · 渠道对账 · GMV结算与会计对账（财务与合规）

- 服务性：`A`　|　检索域：`09-DataAgent-LLM`、`12-ML基础`
- **入卡挂载点**：`responsibility: 渠道对账`　`contract_id: CTR-A-058`　（匹配 role_id+l3）
- 服务格（M 格，15 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-05/STG-04`、`FLOW-05/STG-05`、`FLOW-05/STG-08`、`FLOW-06/STG-04`、`FLOW-06/STG-05`、`FLOW-06/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（23）：`data analysis agent`、`text-to-SQL`、`autonomous data science`、`insight generation`、`table`、`agent`、`reasoning`、`feature engineering`、`tabular`、`tabular foundation model`、`tabular deep learning`、`data agent`、`root cause analysis agent`、`dashboard generation`、`table question answering`、`data wrangling LLM`、`feature transformation`、`model evaluation`、`calibration`、`class imbalance`、`distribution shift`、`leakage`、`agentic etl`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`agentic etl`
- **负向词**（26，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`code generation benchmark`、`gui agent`、`web agent`、`speech recognition`、`image classification`、`object detection`
  - 仅 `09-DataAgent-LLM` 生效：`code generation benchmark`、`gui agent`、`web agent`
  - 仅 `12-ML基础` 生效：`speech recognition`、`image classification`、`object detection`
- **约束词**（8）：`analytics`、`business`、`enterprise`、`database`、`table`、`tabular`、`prediction`、`dataset`

### 位次 3 · 行动组合 · Amazon业务经营（渠道经营）

- 服务性：`A`　|　检索域：`13-广告分析`、`15-营销投放分析`
- **入卡挂载点**：`responsibility: 行动组合`　`contract_id: CTR-A-034`　（匹配 role_id+l3）
- 服务格（M 格，21 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`、`FLOW-06/STG-04`、`FLOW-06/STG-05`、`FLOW-06/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（32）：`advertising`、`auction`、`bidding`、`attribution`、`marketing`、`budget allocation`、`ROI`、`marketing mix model`、`media mix`、`promotion`、`causal`、`retail`、`incrementality`、`geo experiment`、`MMM`、`ROAS`、`sponsored search`、`coupon allocation`、`cannibalization`、`uPlift for advertising`、`ad ranking`、`promotion effectiveness`、`discount elasticity`、`price optimization`、`causal machine learning`、`double machine learning`、`DML`、`sales lift`、`trade promotion`、`price promotion`、`markdown optimization`、`competitive response modeling`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`competitive response modeling`
- **负向词**（27，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`、`health promotion`、`academic promotion`、`faculty`
  - 仅 `13-广告分析` 生效：`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`
  - 仅 `15-营销投放分析` 生效：`health promotion`、`academic promotion`、`faculty`
- **约束词**（10）：`advertis`、`marketing`、`campaign`、`budget`、`roas`、`retail`、`e-commerce`、`sales`、`price`、`promotion`

### 位次 4 · 经营预算 · 经营财务与资金（财务与合规）

- 服务性：`A`　|　检索域：`03-时间序列`、`15-营销投放分析`
- **入卡挂载点**：`responsibility: 经营预算`　`contract_id: CTR-A-062`　（匹配 role_id+l3）
- 服务格（M 格，15 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（30）：`demand forecasting`、`time series`、`foundation model`、`intermittent demand`、`hierarchical forecasting`、`marketing mix model`、`media mix`、`promotion`、`causal`、`retail`、`time series forecasting`、`sales prediction`、`prediction interval`、`conformal prediction`、`temporal fusion transformer`、`event-driven forecasting`、`inventory forecasting`、`promotion effectiveness`、`discount elasticity`、`price optimization`、`causal machine learning`、`double machine learning`、`DML`、`sales lift`、`trade promotion`、`price promotion`、`markdown optimization`、`step back prompting`、`budget reforecast rolling`、`cross border cash flow forecasting`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`step back prompting`、`budget reforecast rolling`、`cross border cash flow forecasting`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`weather`、`climate`、`traffic flow`、`wind power`、`load forecasting`、`seismic`、`health promotion`、`academic promotion`、`faculty`
  - 仅 `03-时间序列` 生效：`weather`、`climate`、`traffic flow`、`wind power`、`load forecasting`、`eeg`、`ecg`、`seismic`
  - 仅 `15-营销投放分析` 生效：`health promotion`、`academic promotion`、`faculty`
- **约束词**（8）：`demand`、`sales`、`retail`、`inventory`、`e-commerce`、`supply chain`、`price`、`promotion`

### 位次 5 · 差异追踪 · GMV结算与会计对账（财务与合规）

- 服务性：`A`　|　检索域：`12-ML基础`、`02-A_B实验`
- **入卡挂载点**：`responsibility: 差异追踪`　`contract_id: CTR-A-060`　（匹配 role_id+l3）
- 服务格（M 格，15 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-05/STG-04`、`FLOW-05/STG-05`、`FLOW-05/STG-08`、`FLOW-06/STG-04`、`FLOW-06/STG-05`、`FLOW-06/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（26）：`feature engineering`、`tabular`、`tabular foundation model`、`tabular deep learning`、`A/B testing`、`online controlled experiment`、`multi-armed bandit`、`recommendation`、`sequential testing`、`experiment assignment`、`feature transformation`、`model evaluation`、`calibration`、`class imbalance`、`distribution shift`、`leakage`、`experiment design`、`statistical power`、`always-valid inference`、`variance reduction`、`CUPED`、`switchback experiment`、`interference experiment`、`contextual bandit`、`Thompson sampling`、`inventory theft warehouse anomaly`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`inventory theft warehouse anomaly`
- **负向词**（26，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`speech recognition`、`image classification`、`object detection`、`wet lab`、`robot`、`in vitro`
  - 仅 `12-ML基础` 生效：`speech recognition`、`image classification`、`object detection`
  - 仅 `02-A_B实验` 生效：`wet lab`、`robot`、`in vitro`
- **约束词**（11）：`tabular`、`business`、`prediction`、`dataset`、`online`、`platform`、`marketplace`、`advertis`、`pricing`、`user`、`e-commerce`

### 位次 6 · 收入与费用核对 · GMV结算与会计对账（财务与合规）

- 服务性：`A`　|　检索域：`09-DataAgent-LLM`、`08-知识图谱`
- **入卡挂载点**：`responsibility: 收入与费用核对`　`contract_id: CTR-A-059`　（匹配 role_id+l3）
- 服务格（M 格，15 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-05/STG-04`、`FLOW-05/STG-05`、`FLOW-05/STG-08`、`FLOW-06/STG-04`、`FLOW-06/STG-05`、`FLOW-06/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（26）：`data analysis agent`、`text-to-SQL`、`autonomous data science`、`insight generation`、`table`、`agent`、`reasoning`、`knowledge graph`、`retrieval augmented`、`graph neural network`、`e-commerce`、`entity alignment`、`knowledge graph completion`、`data agent`、`root cause analysis agent`、`dashboard generation`、`table question answering`、`data wrangling LLM`、`heterogeneous graph`、`hyperbolic embedding`、`entity linking`、`relation extraction`、`graph retrieval`、`GraphRAG`、`knowledge graph construction`、`llm financial report analyst`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`llm financial report analyst`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`code generation benchmark`、`gui agent`、`web agent`、`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`
  - 仅 `09-DataAgent-LLM` 生效：`code generation benchmark`、`gui agent`、`web agent`
  - 仅 `08-知识图谱` 生效：`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`
- **约束词**（10）：`analytics`、`business`、`enterprise`、`database`、`table`、`e-commerce`、`product`、`customer`、`agent`、`retrieval`

### 位次 7 · 生命周期分析 · 库存与商品生命周期（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`05-推荐系统`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 生命周期分析`　`contract_id: CTR-A-026`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`
- **正向词**（35）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`recommender system`、`sequential recommendation`、`generative recommendation`、`cold-start`、`recommendation`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`recommendation system`、`collaborative filtering`、`semantic ID`、`cold-start recommendation`、`conversational recommendation`、`reranking`、`session-based recommendation`、`catalog enrichment`、`product lifecycle stage`、`logistics cost pl attribution`、`cross border returns cost model`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`product lifecycle stage`、`logistics cost pl attribution`、`cross border returns cost model`
- **负向词**（31，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `05-推荐系统` 生效：`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`
- **约束词**（14）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`e-commerce`、`product`、`item`、`user`、`session`、`catalog`

### 位次 8 · 资金预测 · 经营财务与资金（财务与合规）

- 服务性：`A`　|　检索域：`03-时间序列`、`04-供应链`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 资金预测`　`contract_id: CTR-A-061`　（匹配 role_id+l3）
- 服务格（M 格，15 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（35）：`demand forecasting`、`time series`、`foundation model`、`intermittent demand`、`hierarchical forecasting`、`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`time series forecasting`、`sales prediction`、`prediction interval`、`conformal prediction`、`temporal fusion transformer`、`event-driven forecasting`、`inventory forecasting`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`fx hedging strategy`、`amazon lending decision`、`multicurrency fx hedging`、`budget reforecast rolling`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`fx hedging strategy`、`amazon lending decision`、`multicurrency fx hedging`、`budget reforecast rolling`
- **负向词**（32，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`weather`、`climate`、`traffic flow`、`wind power`、`load forecasting`、`seismic`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `03-时间序列` 生效：`weather`、`climate`、`traffic flow`、`wind power`、`load forecasting`、`eeg`、`ecg`、`seismic`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
- **约束词**（12）：`demand`、`sales`、`retail`、`inventory`、`e-commerce`、`supply chain`、`goods`、`physical`、`warehouse`、`fulfillment`、`logistics`、`order`

### 位次 11 · 账号商品映射 · 业务口径与主数据（数据与AI运行）

- 服务性：`A`　|　检索域：`08-知识图谱`、`16-智能体工程`
- **入卡挂载点**：`responsibility: 账号商品映射`　`contract_id: CTR-A-066`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-06/STG-04`、`FLOW-06/STG-05`、`FLOW-06/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（39）：`knowledge graph`、`retrieval augmented`、`graph neural network`、`e-commerce`、`entity alignment`、`knowledge graph completion`、`agent skills`、`skill library`、`LLM`、`context engineering`、`context compression`、`agent`、`model context protocol`、`agent2agent`、`tool use`、`benchmark`、`agent memory`、`long-term`、`heterogeneous graph`、`hyperbolic embedding`、`entity linking`、`relation extraction`、`graph retrieval`、`GraphRAG`、`knowledge graph construction`、`agent skill`、`tool calling`、`function calling`、`MCP`、`agent observability`、`agent evaluation`、`LLM-as-a-judge`、`rubric`、`prompt injection`、`agent safety`、`asin sku erp`、`amazon tiktok`、`product kg query`、`privacy safe identity resolution`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`asin sku erp`、`amazon tiktok`、`product kg query`、`privacy safe identity resolution`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`、`embodied agent`、`robot manipulation`、`gui agent`
  - 仅 `08-知识图谱` 生效：`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`
  - 仅 `16-智能体工程` 生效：`embodied agent`、`robot manipulation`、`gui agent`
- **约束词**（8）：`e-commerce`、`product`、`customer`、`agent`、`retrieval`、`llm`、`tool`、`context`

### 位次 12 · 账号诊断 · 店铺账号健康与规则（渠道经营）

- 服务性：`A`　|　检索域：`12-ML基础`、`14-用户分析`　|　重灾区域：`14-用户分析`
- **入卡挂载点**：`responsibility: 账号诊断`　`contract_id: CTR-A-045`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`
- **正向词**（29）：`feature engineering`、`tabular`、`tabular foundation model`、`tabular deep learning`、`funnel`、`conversion`、`cohort`、`survival analysis`、`user`、`feature transformation`、`model evaluation`、`calibration`、`class imbalance`、`distribution shift`、`leakage`、`funnel analysis`、`cohort retention`、`RFM segmentation`、`user segmentation`、`behavioral analytics`、`session analysis`、`conversion funnel`、`customer journey`、`path analysis`、`survival analysis for users`、`amazon poa`、`voc fraud review detection`、`seller rating attack pattern`、`account health proactive monitor`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`amazon poa`、`voc fraud review detection`、`seller rating attack pattern`、`account health proactive monitor`
- **负向词**（27，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`speech recognition`、`image classification`、`object detection`、`pet scan`、`cohort study`、`epidemiolog`、`diagnosis`
  - 仅 `12-ML基础` 生效：`speech recognition`、`image classification`、`object detection`
  - 仅 `14-用户分析` 生效：`clinical`、`patient`、`mri`、`eeg`、`fmri`、`lesion`、`radiotherapy`、`pet scan`、`tumor`、`cohort study`、`epidemiolog`、`diagnosis`
- **约束词**（10）：`tabular`、`business`、`prediction`、`dataset`、`e-commerce`、`retail`、`subscription`、`churn`、`user`、`customer`

### 位次 13 · 供需协调 · 需求预测与补货计划（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`10-MAS`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 供需协调`　`contract_id: CTR-A-024`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`
- **正向词**（41）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`multi-agent`、`LLM`、`coordination`、`agent`、`simulation`、`consumer`、`multi-agent reinforcement learning`、`pricing`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`multi-agent system`、`agent collaboration`、`agent orchestration`、`role assignment`、`communication protocol`、`A2A`、`agent handoff`、`debate`、`voting`、`collaboration tax`、`ad spend inventory sync`、`event driven demand mas`、`supply chain causal scm`、`bullwhip effect mitigation`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`ad spend inventory sync`、`event driven demand mas`、`supply chain causal scm`、`bullwhip effect mitigation`
- **负向词**（27，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`multi-agent reinforcement learning`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `10-MAS` 生效：`multi-agent reinforcement learning`
- **约束词**（12）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`llm`、`agent`、`task`、`tool`

### 位次 14 · 平台运营 · 其他平台与新市场经营（渠道经营）

- 服务性：`A`　|　检索域：`13-广告分析`、`15-营销投放分析`
- **入卡挂载点**：`responsibility: 平台运营`　`contract_id: CTR-A-040`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-01/STG-04`、`FLOW-01/STG-05`、`FLOW-01/STG-08`、`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`
- **正向词**（34）：`advertising`、`auction`、`bidding`、`attribution`、`marketing`、`budget allocation`、`ROI`、`marketing mix model`、`media mix`、`promotion`、`causal`、`retail`、`incrementality`、`geo experiment`、`MMM`、`ROAS`、`sponsored search`、`coupon allocation`、`cannibalization`、`uPlift for advertising`、`ad ranking`、`promotion effectiveness`、`discount elasticity`、`price optimization`、`causal machine learning`、`double machine learning`、`DML`、`sales lift`、`trade promotion`、`price promotion`、`markdown optimization`、`tiktok cvr`、`amazon tiktok shop shopee`、`cross platform transfer recommendation`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`tiktok cvr`、`amazon tiktok shop shopee`、`cross platform transfer recommendation`
- **负向词**（27，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`、`health promotion`、`academic promotion`、`faculty`
  - 仅 `13-广告分析` 生效：`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`
  - 仅 `15-营销投放分析` 生效：`health promotion`、`academic promotion`、`faculty`
- **约束词**（10）：`advertis`、`marketing`、`campaign`、`budget`、`roas`、`retail`、`e-commerce`、`sales`、`price`、`promotion`

### 位次 16 · 品牌反馈 · 品牌战略与传播（品牌与增长）

- 服务性：`A`　|　检索域：`07-NLP-VOC`、`14-用户分析`　|　重灾区域：`14-用户分析`
- **入卡挂载点**：`responsibility: 品牌反馈`　`contract_id: CTR-A-047`　（匹配 role_id+l3）
- 服务格（M 格，15 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`、`FLOW-05/STG-04`、`FLOW-05/STG-05`、`FLOW-05/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（41）：`aspect-based sentiment`、`opinion mining`、`customer review`、`LLM`、`large language model`、`review summarization`、`user feedback`、`mining`、`funnel`、`conversion`、`cohort`、`survival analysis`、`user`、`product review summarization`、`product`、`review`、`multimodal review`、`review image understanding`、`aspect extraction`、`voice of customer`、`customer feedback categorization`、`complaint analysis`、`LLM-as-annotator`、`label taxonomy evolution`、`self-evolving label system`、`intent classification`、`NPS prediction`、`funnel analysis`、`cohort retention`、`RFM segmentation`、`user segmentation`、`behavioral analytics`、`session analysis`、`conversion funnel`、`customer journey`、`path analysis`、`survival analysis for users`、`video sentiment analysis voc`、`reddit community signal mining`、`brand safety video content filter`、`cross platform brand search volume`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`video sentiment analysis voc`、`reddit community signal mining`、`brand safety video content filter`、`cross platform brand search volume`
- **负向词**（28，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`clinical note`、`hate speech`、`fake news`、`machine translation`、`pet scan`、`cohort study`、`epidemiolog`、`diagnosis`
  - 仅 `07-NLP-VOC` 生效：`clinical note`、`hate speech`、`fake news`、`machine translation`
  - 仅 `14-用户分析` 生效：`clinical`、`patient`、`mri`、`eeg`、`fmri`、`lesion`、`radiotherapy`、`pet scan`、`tumor`、`cohort study`、`epidemiolog`、`diagnosis`
- **约束词**（9）：`e-commerce`、`product`、`customer`、`review`、`feedback`、`retail`、`subscription`、`churn`、`user`

### 位次 17 · 可用性验证 · 工业设计与用户体验（产品与创新）

- 服务性：`A`　|　检索域：`02-A_B实验`、`06-增长模型`
- **入卡挂载点**：`responsibility: 可用性验证`　`contract_id: CTR-A-013`　（匹配 role_id+l3）
- 服务格（M 格，3 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`
- **正向词**（32）：`A/B testing`、`online controlled experiment`、`multi-armed bandit`、`recommendation`、`sequential testing`、`experiment assignment`、`churn prediction`、`customer lifetime value`、`user growth`、`retention`、`uplift`、`conversion rate prediction`、`e-commerce`、`experiment design`、`statistical power`、`always-valid inference`、`variance reduction`、`CUPED`、`switchback experiment`、`interference experiment`、`contextual bandit`、`Thompson sampling`、`LTV`、`repurchase`、`user lifecycle`、`RFM`、`survival analysis`、`uplift churn`、`seasonality`、`early warning system`、`renewal prediction`、`inclusive design accessibility ai`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`inclusive design accessibility ai`
- **负向词**（26，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`wet lab`、`robot`、`in vitro`、`employee turnover`、`student dropout`、`attrition rate`
  - 仅 `02-A_B实验` 生效：`wet lab`、`robot`、`in vitro`
  - 仅 `06-增长模型` 生效：`employee turnover`、`student dropout`、`attrition rate`
- **约束词**（10）：`online`、`platform`、`marketplace`、`advertis`、`pricing`、`user`、`e-commerce`、`subscription`、`retail`、`customer`

### 位次 18 · 测试方案 · 产品验证与研发项目（产品与创新）

- 服务性：`A`　|　检索域：`02-A_B实验`、`10-MAS`
- **入卡挂载点**：`responsibility: 测试方案`　`contract_id: CTR-A-017`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`
- **正向词**（33）：`A/B testing`、`online controlled experiment`、`multi-armed bandit`、`recommendation`、`sequential testing`、`experiment assignment`、`multi-agent`、`LLM`、`coordination`、`agent`、`simulation`、`consumer`、`multi-agent reinforcement learning`、`pricing`、`experiment design`、`statistical power`、`always-valid inference`、`variance reduction`、`CUPED`、`switchback experiment`、`interference experiment`、`contextual bandit`、`Thompson sampling`、`multi-agent system`、`agent collaboration`、`agent orchestration`、`role assignment`、`communication protocol`、`A2A`、`agent handoff`、`debate`、`voting`、`collaboration tax`
  - ⚠️ **无交叉项**（该责任无 legacy 卡可取材）—— 见 §7 未做项
- **负向词**（24，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`wet lab`、`robot`、`in vitro`、`multi-agent reinforcement learning`
  - 仅 `02-A_B实验` 生效：`wet lab`、`robot`、`in vitro`
  - 仅 `10-MAS` 生效：`multi-agent reinforcement learning`
- **约束词**（11）：`online`、`platform`、`marketplace`、`advertis`、`pricing`、`user`、`e-commerce`、`llm`、`agent`、`task`、`tool`

### 位次 19 · 资源情景比较 · 经营目标与资源统筹（经营与组织）

- 服务性：`A`　|　检索域：`15-营销投放分析`、`04-供应链`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 资源情景比较`　`contract_id: CTR-A-001`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（35）：`marketing mix model`、`media mix`、`promotion`、`causal`、`retail`、`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`promotion effectiveness`、`discount elasticity`、`price optimization`、`causal machine learning`、`double machine learning`、`DML`、`sales lift`、`trade promotion`、`price promotion`、`markdown optimization`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`multi objective budget allocation`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`multi objective budget allocation`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`health promotion`、`academic promotion`、`faculty`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `15-营销投放分析` 生效：`health promotion`、`academic promotion`、`faculty`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
- **约束词**（12）：`retail`、`e-commerce`、`sales`、`price`、`promotion`、`goods`、`physical`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`

### 位次 20 · 知识产权检索 · 法务与知识产权（财务与合规）

- 服务性：`A`　|　检索域：`08-知识图谱`、`07-NLP-VOC`
- **入卡挂载点**：`responsibility: 知识产权检索`　`contract_id: CTR-A-064`　（匹配 role_id+l3）
- 服务格（M 格，12 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（36）：`knowledge graph`、`retrieval augmented`、`graph neural network`、`e-commerce`、`entity alignment`、`knowledge graph completion`、`aspect-based sentiment`、`opinion mining`、`customer review`、`LLM`、`large language model`、`review summarization`、`user feedback`、`mining`、`heterogeneous graph`、`hyperbolic embedding`、`entity linking`、`relation extraction`、`graph retrieval`、`GraphRAG`、`knowledge graph construction`、`product review summarization`、`product`、`review`、`multimodal review`、`review image understanding`、`aspect extraction`、`voice of customer`、`customer feedback categorization`、`complaint analysis`、`LLM-as-annotator`、`label taxonomy evolution`、`self-evolving label system`、`intent classification`、`NPS prediction`、`gan listing adversarial listing defense`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`gan listing adversarial listing defense`
- **负向词**（30，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`、`clinical note`、`hate speech`、`fake news`、`machine translation`
  - 仅 `08-知识图谱` 生效：`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`
  - 仅 `07-NLP-VOC` 生效：`clinical note`、`hate speech`、`fake news`、`machine translation`
- **约束词**（7）：`e-commerce`、`product`、`customer`、`agent`、`retrieval`、`review`、`feedback`

### 位次 21 · 达人筛选 · 达人与联盟合作（品牌与增长）

- 服务性：`A`　|　检索域：`05-推荐系统`、`13-广告分析`
- **入卡挂载点**：`responsibility: 达人筛选`　`contract_id: CTR-A-052`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`
- **正向词**（34）：`recommender system`、`large language model`、`sequential recommendation`、`generative recommendation`、`cold-start`、`recommendation`、`advertising`、`auction`、`bidding`、`attribution`、`marketing`、`budget allocation`、`ROI`、`recommendation system`、`collaborative filtering`、`semantic ID`、`cold-start recommendation`、`conversational recommendation`、`reranking`、`session-based recommendation`、`catalog enrichment`、`incrementality`、`geo experiment`、`marketing mix model`、`MMM`、`ROAS`、`sponsored search`、`coupon allocation`、`cannibalization`、`uPlift for advertising`、`ad ranking`、`tiktok roi`、`kol creator matching`、`social proof viral recommendation`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`tiktok roi`、`kol creator matching`、`social proof viral recommendation`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`、`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`
  - 仅 `05-推荐系统` 生效：`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`
  - 仅 `13-广告分析` 生效：`adversarial attack`、`adhd`、`alzheimer`、`adversarial example`
- **约束词**（11）：`e-commerce`、`product`、`item`、`user`、`session`、`catalog`、`advertis`、`marketing`、`campaign`、`budget`、`roas`

### 位次 22 · 履约异常 · 仓储履约与退货处置（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`06-增长模型`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 履约异常`　`contract_id: CTR-A-031`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-05/STG-04`、`FLOW-05/STG-05`、`FLOW-05/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`
- **正向词**（38）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`churn prediction`、`customer lifetime value`、`user growth`、`retention`、`uplift`、`conversion rate prediction`、`e-commerce`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`LTV`、`repurchase`、`user lifecycle`、`RFM`、`survival analysis`、`uplift churn`、`seasonality`、`early warning system`、`renewal prediction`、`time delivery`、`logistics fraud detection`、`logistics anomaly fraud signal`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`time delivery`、`logistics fraud detection`、`logistics anomaly fraud signal`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`employee turnover`、`student dropout`、`attrition rate`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `06-增长模型` 生效：`employee turnover`、`student dropout`、`attrition rate`
- **约束词**（12）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`subscription`、`e-commerce`、`customer`、`platform`

### 位次 23 · 抽样审计 · 内控审计与独立复核（经营与组织）

- 服务性：`A`　|　检索域：`12-ML基础`、`07-NLP-VOC`
- **入卡挂载点**：`responsibility: 抽样审计`　`contract_id: CTR-A-007`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`、`FLOW-08/STG-04`、`FLOW-08/STG-05`、`FLOW-08/STG-08`
- **正向词**（35）：`feature engineering`、`tabular`、`tabular foundation model`、`tabular deep learning`、`aspect-based sentiment`、`opinion mining`、`customer review`、`LLM`、`large language model`、`review summarization`、`user feedback`、`mining`、`feature transformation`、`model evaluation`、`calibration`、`class imbalance`、`distribution shift`、`leakage`、`product review summarization`、`product`、`review`、`multimodal review`、`review image understanding`、`aspect extraction`、`voice of customer`、`customer feedback categorization`、`complaint analysis`、`LLM-as-annotator`、`label taxonomy evolution`、`self-evolving label system`、`intent classification`、`NPS prediction`、`context learning`、`class imbalance handling`、`compliance ml risk scoring`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`context learning`、`class imbalance handling`、`compliance ml risk scoring`
- **负向词**（27，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`speech recognition`、`image classification`、`object detection`、`clinical note`、`hate speech`、`fake news`、`machine translation`
  - 仅 `12-ML基础` 生效：`speech recognition`、`image classification`、`object detection`
  - 仅 `07-NLP-VOC` 生效：`clinical note`、`hate speech`、`fake news`、`machine translation`
- **约束词**（9）：`tabular`、`business`、`prediction`、`dataset`、`e-commerce`、`product`、`customer`、`review`、`feedback`

### 位次 24 · 质量分析 · 生产协同与质量控制（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`02-A_B实验`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 质量分析`　`contract_id: CTR-A-028`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`
- **正向词**（37）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`A/B testing`、`online controlled experiment`、`multi-armed bandit`、`recommendation`、`sequential testing`、`experiment assignment`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`experiment design`、`statistical power`、`always-valid inference`、`variance reduction`、`CUPED`、`switchback experiment`、`interference experiment`、`contextual bandit`、`Thompson sampling`、`graph rag knowledge retrieval`、`voc triggered inventory signal`、`safety concern signal extraction`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`graph rag knowledge retrieval`、`voc triggered inventory signal`、`safety concern signal extraction`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`wet lab`、`robot`、`in vitro`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `02-A_B实验` 生效：`wet lab`、`robot`、`in vitro`
- **约束词**（15）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`online`、`platform`、`marketplace`、`advertis`、`pricing`、`user`、`e-commerce`

### 位次 25 · 退货分流 · 仓储履约与退货处置（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`05-推荐系统`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 退货分流`　`contract_id: CTR-A-032`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-05/STG-04`、`FLOW-05/STG-05`、`FLOW-05/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`
- **正向词**（36）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`recommender system`、`sequential recommendation`、`generative recommendation`、`cold-start`、`recommendation`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`recommendation system`、`collaborative filtering`、`semantic ID`、`cold-start recommendation`、`conversational recommendation`、`reranking`、`session-based recommendation`、`catalog enrichment`、`voc returns cost driver`、`parcel damage prediction`、`returns reverse logistics`、`predictive returns management`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`voc returns cost driver`、`parcel damage prediction`、`returns reverse logistics`、`predictive returns management`
- **负向词**（31，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `05-推荐系统` 生效：`movie`、`music recommendation`、`news recommendation`、`poi recommendation`、`friend recommendation`
- **约束词**（14）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`e-commerce`、`product`、`item`、`user`、`session`、`catalog`

### 位次 26 · 订单协调 · 采购与合同履约（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`10-MAS`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 订单协调`　`contract_id: CTR-A-020`　（匹配 role_id+l3）
- 服务格（M 格，3 个）：`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`
- **正向词**（39）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`multi-agent`、`LLM`、`coordination`、`agent`、`simulation`、`consumer`、`multi-agent reinforcement learning`、`pricing`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`multi-agent system`、`agent collaboration`、`agent orchestration`、`role assignment`、`communication protocol`、`A2A`、`agent handoff`、`debate`、`voting`、`collaboration tax`、`model context protocol`、`supply chain resilience modeling`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`model context protocol`、`supply chain resilience modeling`
- **负向词**（27，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`multi-agent reinforcement learning`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `10-MAS` 生效：`multi-agent reinforcement learning`
- **约束词**（12）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`llm`、`agent`、`task`、`tool`

### 位次 27 · 趋势监测 · 市场竞争与机会研究（产品与创新）

- 服务性：`A`　|　检索域：`06-增长模型`、`03-时间序列`
- **入卡挂载点**：`responsibility: 趋势监测`　`contract_id: CTR-A-010`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`
- **正向词**（32）：`churn prediction`、`customer lifetime value`、`user growth`、`retention`、`uplift`、`conversion rate prediction`、`e-commerce`、`demand forecasting`、`time series`、`foundation model`、`intermittent demand`、`hierarchical forecasting`、`LTV`、`repurchase`、`user lifecycle`、`RFM`、`survival analysis`、`uplift churn`、`seasonality`、`early warning system`、`renewal prediction`、`time series forecasting`、`sales prediction`、`prediction interval`、`conformal prediction`、`temporal fusion transformer`、`event-driven forecasting`、`inventory forecasting`、`sir tiktok`、`amazon tiktok`、`google trends`、`demand signal nowcasting`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`sir tiktok`、`amazon tiktok`、`google trends`、`demand signal nowcasting`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`employee turnover`、`student dropout`、`attrition rate`、`weather`、`climate`、`traffic flow`、`wind power`、`load forecasting`、`seismic`
  - 仅 `06-增长模型` 生效：`employee turnover`、`student dropout`、`attrition rate`
  - 仅 `03-时间序列` 生效：`weather`、`climate`、`traffic flow`、`wind power`、`load forecasting`、`eeg`、`ecg`、`seismic`
- **约束词**（9）：`subscription`、`e-commerce`、`retail`、`customer`、`platform`、`demand`、`sales`、`inventory`、`supply chain`

### 位次 28 · 复购实验 · CRM留存与复购（品牌与增长）

- 服务性：`A`　|　检索域：`01-因果推断`、`02-A_B实验`
- **入卡挂载点**：`responsibility: 复购实验`　`contract_id: CTR-A-055`　（匹配 role_id+l3）
- 服务格（M 格，3 个）：`FLOW-05/STG-04`、`FLOW-05/STG-05`、`FLOW-05/STG-08`
- **正向词**（34）：`uplift modeling`、`heterogeneous treatment effect`、`causal inference`、`e-commerce`、`causal`、`marketing`、`incrementality`、`A/B testing`、`online controlled experiment`、`multi-armed bandit`、`recommendation`、`sequential testing`、`experiment assignment`、`treatment effect estimation`、`propensity score`、`doubly robust`、`instrumental variables`、`difference-in-differences`、`synthetic control`、`causal discovery`、`mediation analysis`、`experiment design`、`statistical power`、`always-valid inference`、`variance reduction`、`CUPED`、`switchback experiment`、`interference experiment`、`contextual bandit`、`Thompson sampling`、`notears dagma`、`email sequence rl optimizer`、`purchase sequence prediction`、`email sequence multiarm optimizer`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`notears dagma`、`email sequence rl optimizer`、`purchase sequence prediction`、`email sequence multiarm optimizer`
- **负向词**（27，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`drug`、`gene`、`epidemiolog`、`biomarker`、`wet lab`、`robot`、`in vitro`
  - 仅 `01-因果推断` 生效：`clinical`、`patient`、`drug`、`gene`、`epidemiolog`、`biomarker`
  - 仅 `02-A_B实验` 生效：`wet lab`、`robot`、`in vitro`
- **约束词**（12）：`e-commerce`、`retail`、`advertis`、`marketing`、`pricing`、`subscription`、`customer`、`business`、`online`、`platform`、`marketplace`、`user`

### 位次 29 · 供应商评估 · OEM供应商开发与协同（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`08-知识图谱`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 供应商评估`　`contract_id: CTR-A-018`　（匹配 role_id+l3）
- 服务格（M 格，9 个）：`FLOW-02/STG-04`、`FLOW-02/STG-05`、`FLOW-02/STG-08`、`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-07/STG-04`、`FLOW-07/STG-05`、`FLOW-07/STG-08`
- **正向词**（36）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`knowledge graph`、`retrieval augmented`、`graph neural network`、`e-commerce`、`entity alignment`、`knowledge graph completion`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`heterogeneous graph`、`hyperbolic embedding`、`entity linking`、`relation extraction`、`graph retrieval`、`GraphRAG`、`knowledge graph construction`、`supplier risk xgboost`、`supplier evaluation model`、`agentic sckg risk analyzer`、`graph rag knowledge retrieval`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`supplier risk xgboost`、`supplier evaluation model`、`agentic sckg risk analyzer`、`graph rag knowledge retrieval`
- **负向词**（32，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `08-知识图谱` 生效：`molecular`、`protein`、`drug discovery`、`biomedical`、`chemistry`、`crystal`
- **约束词**（13）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`e-commerce`、`product`、`customer`、`agent`、`retrieval`

### 位次 30 · 到货异常追踪 · 跨境物流与关务（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`16-智能体工程`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 到货异常追踪`　`contract_id: CTR-A-030`　（匹配 role_id+l3）
- 服务格（M 格，6 个）：`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`、`FLOW-04/STG-04`、`FLOW-04/STG-05`、`FLOW-04/STG-08`
- **正向词**（45）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`agent skills`、`skill library`、`LLM`、`context engineering`、`context compression`、`agent`、`model context protocol`、`agent2agent`、`tool use`、`benchmark`、`agent memory`、`long-term`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`agent skill`、`tool calling`、`function calling`、`MCP`、`agent observability`、`agent evaluation`、`LLM-as-a-judge`、`rubric`、`prompt injection`、`agent safety`、`parcel damage prediction`、`ar logistics visualization`、`last mile delivery prediction`、`cross border last mile routing`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`parcel damage prediction`、`ar logistics visualization`、`last mile delivery prediction`、`cross border last mile routing`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`embodied agent`、`robot manipulation`、`gui agent`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `16-智能体工程` 生效：`embodied agent`、`robot manipulation`、`gui agent`
- **约束词**（12）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`llm`、`agent`、`tool`、`context`

### 位次 31 · 履约跟踪 · 采购与合同履约（供应与履约）

- 服务性：`A`　|　检索域：`04-供应链`、`09-DataAgent-LLM`　|　重灾区域：`04-供应链`
- **入卡挂载点**：`responsibility: 履约跟踪`　`contract_id: CTR-A-021`　（匹配 role_id+l3）
- 服务格（M 格，3 个）：`FLOW-03/STG-04`、`FLOW-03/STG-05`、`FLOW-03/STG-08`
- **正向词**（34）：`inventory management`、`reinforcement learning`、`deep learning`、`supply chain`、`large language model`、`replenishment`、`newsvendor`、`data analysis agent`、`text-to-SQL`、`autonomous data science`、`insight generation`、`table`、`agent`、`reasoning`、`inventory optimization`、`multi-echelon inventory`、`supply network design`、`warehouse allocation`、`order fulfillment`、`lead time uncertainty`、`safety stock`、`perishable inventory`、`supply chain resilience`、`disruption`、`logistics cost`、`last-mile delivery`、`data agent`、`root cause analysis agent`、`dashboard generation`、`table question answering`、`data wrangling LLM`、`fill rate oos`、`time delivery`、`supplier performance alert action`
  - 其中**交叉项**（该责任独有的英文词，来自实测挂过本 L3 的卡）：`fill rate oos`、`time delivery`、`supplier performance alert action`
- **负向词**（29，全局 + 本域）：`software supply chain`、`sbom`、`software bill of materials`、`package detection`、`dependency vulnerability`、`model lineage`、`model card`、`kv cache`、`serving throughput`、`gpu kernel`、`clinical`、`patient`、`lesion`、`mri`、`eeg`、`ecg`、`fmri`、`radiotherapy`、`histopathology`、`tumor`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`、`code generation benchmark`、`gui agent`、`web agent`
  - 仅 `04-供应链` 生效：`software supply chain`、`sbom`、`npm`、`pypi`、`malware`、`vulnerability`、`repository`、`compiler`
  - 仅 `09-DataAgent-LLM` 生效：`code generation benchmark`、`gui agent`、`web agent`
- **约束词**（13）：`goods`、`physical`、`retail`、`warehouse`、`fulfillment`、`inventory`、`logistics`、`order`、`analytics`、`business`、`enterprise`、`database`、`table`

---

## 6b. 端到端变异（**每条判据都要配篡改样本**）

本仓库的纪律原话：**每新增一条断言，必须同时加一份篡改样本** ——
曾有 4 处断言因为真实数据本来就满足它，把断言整行删掉自检照样全绿（= 没断言）。
所以这里做**两层**证明，`--selftest`（单元级）与 `--mutate`（端到端级）：

### 层一 · 单元级 `--selftest`：**20 用例 · 26 变异样本**

把判据函数喂给合成输入，证明**那个函数**会报警。这是必要条件，不是充分条件 ——
它证明不了「判据在真实数据上会失败」。

### 层二 · 端到端 `--mutate`：在**真实数据**上逐条弄坏判据

**值变异**（改变一个真实取值让它违反判据）：**13/13 判红**，覆盖判据 **10/10**

| 判据 | 值变异 | 打红 | 触发的判据原文 |
|---|---|---|---|
| J1 | 把「输入齐备」改成假 ⇒ 输入没拿到必须判红（不是「通过」） | ✅ | 输入没拿到：合成环境：靶区工单不存在 |
| J2 | 把工单截短成 10 条 ⇒ 生成数 8 ≠ 基线 28（覆盖率读数随之改变） | ✅ | J2 生成数 8 ≠ 基线 28（覆盖率 80.0%）—— 覆盖率与生成数**两侧都要钉** |
| J3 | 把工单里 2 条标成结构空白 ⇒ 严口径 5 ≠ 基线 3 | ✅ | J3 strict_worklist 实测 5 ≠ 基线 3 |
|  | 把全部结构空白标记清掉 ⇒ 生成 31 条 ≠ 可检索位 28 | ✅ | J3 strict_worklist 实测 0 ≠ 基线 3 |
| J4 | 让每次构建都现场取一个时间戳 ⇒ 两次运行不再逐字节相同 | ✅ | J4 两次运行的检索式不是逐字节相同 —— 检索式不可复算 |
| J5 | 把位次 1/2 两条工单都改写成同一责任名+同一域 ⇒ 对应检索式应随之变化 | ✅ | J5 篡改 domain 后没有任何检索式变化 —— 检索式是假仪表 |
| J6 | 把所有交叉项统一成同一个词 ⇒ 交叉项 Jaccard = 1.0（模板套壳） | ✅ | J6 最相似的一对 11:账号商品映射 ↔ 12:账号诊断 交叉项 Jaccard=1.0 ≥ 0.8 —— 只换了责任名 |
|  | 把所有正向词统一成同一个列表 ⇒ 全向量 Jaccard = 1.0 | ✅ | J6 最相似的一对 11:账号商品映射 ↔ 12:账号诊断 全向量 Jaccard=1.0 ≥ 0.95 —— 模板套壳 |
| J7 | 把 4 组同词异义反例换成全局负向词（仪器用错） | ✅ | J7 反例失效：'clinical' 不再是分歧词（各域状态一致）——「同一词在不同域含义相反」在 filter 里已不成立，必须复核 |
| J8 | 把一个检索域换成 candidate_filter 不认识的域名 ⇒ filter 会静默跳过 | ✅ | J8 域 ['99-不存在的域'] 不在 candidate_filter.DOMAIN_DIR 里 —— filter 会把它们**静默跳过**（保留全部，等于没过滤） |
| J9 | 把**全部**契约挂载点清空 ⇒ 一条都没挂上 | ✅ | J9 1:线索评估 在契约池里有对应契约，却没挂上 —— 索引键写错了 |
|  | 只把**一条**契约挂载点清空 ⇒ 掉了一条也算掉（不许全有全无） | ✅ | J9 1:线索评估 在契约池里有对应契约，却没挂上 —— 索引键写错了 |
| J10 | 把工单的 weights 塞进产物 ⇒ 排序分数进了产物 | ✅ | J10 产物里出现排序权重名 —— 分数只用于排序 |

**删除型变异**（把判据整段删掉/清空）：**2/5 判红**。

> ⚠️ **这两个数必须分开报，不能平均成一个「变异 N/N」。**
> 删除型里未判红的那 3 条（J2 覆盖率、J7 同词异义反例、J8 域名合法性）
> **不构成「判据有效」的证据** —— 真实数据本来就满足它们，删掉当然还是绿的。
> 它们的证据在**值变异**那一栏（三条都打红了）。**把两类混在一起报，
> 就等于用「删了还是绿」冒充「判据没问题」。**

### 这一节自己抓到的东西（**登记，都是实测**）

| # | 撞出的缺陷 | 形态 | 修法 |
|---|---|---|---|
| 1 | J6 把两条**空交叉项**算成 Jaccard=1.0 | 假红（报出一对不存在的「只换责任名」） | 空集之间相似度**无定义**；空交叉项条目从交叉项比对里剔除，单独计数 |
| 2 | J9 只判「全挂上 / 一条都没挂」 | **单边判据**（28 条里掉 1 条照样过） | 补「挂载数 = 应有数」这一侧 + 索引键核查 |
| 3 | J10 的 needle 表**内联在判据函数体里** | 自证假红（扫描打中了自己） | needle 移到模块常量，函数体只留不含字面量的引用 |
| 4 | J10 用**裸子串**匹配 needle | 假红（输入字段名 `in_first_slice` 含 `first_slice`） | 改词边界正则 `(?<![A-Za-z0-9_])…(?![A-Za-z0-9_])` |
| 5 | J7 拿 `clinical`/`eeg` 当「同词异义」反例 | **仪器用错**（它们在 `GLOBAL_NEGATIVE` 里，全域生效，根本不可能是分歧词） | 反例改用按域负向词，并加「全局词不得冒充分歧反例」的断言 |
| 6 | 变异本身**没施上力**（截短到 28 条时尾部正好都是结构空白 ⇒ 生成数没变） | 假阴性（看起来像「判据无效」，其实是变异没生效） | 先证明变异改变了真实取值，再谈判据 |
| 7 | `judge_j4` 拿**同一个对象**跟自己比 | 假绿（`build` 级变异永远打不红） | 「重算」必须真的重算（`lambda: _rebuild(...)`） |

---

## 7. 未做 / 继承接项

- **只生成检索式，不执行检索** —— 本任务（S4）的交付物止于「缺口 → 检索式」。真正发查询、收割、打分属下一步。
- 🔴 **`arxiv_harvest.py` 的域标签与 `candidate_filter.py` 有 4 处实测不一致**（`07-VOC舆情` vs `07-NLP-VOC`、`09-DataAgent` vs `09-DataAgent-LLM`、`15-营销投放` vs `15-营销投放分析`、`00-电商Agent` 在 filter 里**根本没有**）：实测候选池 1046 篇里 **141 篇**（89 + 37 + 9 + 6）的 `query_groups` **全都**不在 `DOMAIN_DIR` 里，会被 `filter_pool` 的 `if not domains: keep` 分支**整篇放行**（负向词与约束词都不生效）。⚠️ 本数字**首次报成了 113** —— 漏算了 `00-电商Agent` 那 37 篇；「漏算」的原因是当时只看了「filter 里有同名域」的三对，没查「filter 里压根没有的域」。**这正是本仓库那条铁律的又一例**：判「某个东西不存在」之前，先问仪器能不能看见它。本任务用 `harvest_domain_of` 在**本侧**折好这 3 对，`00-电商Agent` 不参与本次检索域，**不动上游脚本**（纪律 6），已登录。
- **结构空白那 3 条不给检索式**（走 SOP）。已装线里与它们同名 L3 的卡为 0，因此连交叉项都取不到材 —— 这不是本脚本的缺陷，是「结构性空白」的定义。
- **线索评估（位次 1）有 legacy 卡为 0**，但账里 `structural_blank=false` ⇒ 它**进入检索预算**、却**没有交叉项**。已在 §6 逐条标注 ⚠️，请所有者裁决（补齐它的 legacy 归类，或接受它只有域词）。
- **契约挂载点覆盖率**见 J9 —— `responsibility` 与 `contract_id` 的键名**照抄** `check_contracts.py`，没有自造键名。
- **排序权重不进任何判定**：J10 用 AST 扫本文件的 `judge_*` 函数体，并配反向探针（往判据函数体注入 `zero_gap`，扫描必须报出来）。
- 🔴 **登记（不重判）：`关键词库-v2.md` §1 与 `candidate_filter.py` 实测不符** —— 词库说「负向词 `trial` 只在 14 域生效」，实测 `trial` **在 filter 的任何域的负向表里都不存在**（`14-用户分析` 用的是收窄形式 `cohort study` / `pet scan`，但没有 `trial`；`02-A_B实验` 不列 `trial` 是对的，但 14 域也没列）。`ad` 同理（用 `adversarial attack` / `adhd` / `alzheimer` 收窄实现）。按纪律**只登记不重判、不擅自改 filter**；同时配了一条能打红的判据：**若哪天 filter 真把 `trial` 列进 01/02 域（即开始误杀 RCT 方法论论文）必须报红**（`FORBIDDEN_NEGATIVES`，selftest 里有对应变异）。
- 🔴 **登记（不重判）：上游图谱在本任务执行期间被改动过** —— 工单账 `_meta.graph_sha256` 记的是 `be36dd1d197def0e`，而 `capability-graph.json` 现在的 sha256 前 16 位是 `ecda4970bbb7b56b`（文件 mtime 也晚于工单生成时间）。⇒ **工单可能已滞后于图谱**。本任务（S4）只消费工单、不重建它（纪律 6：不改缺口账生成器），故只登记：下一次跑 F3 的 `--check` 会由**它自己的**判据报出来。产物的 `_input_fingerprints` 已把两侧指纹都记下，消费方可自查。实测确认：`build_gap_ledger.py --check` 现在确实判红（红在「工单与图谱/分类现状不一致」），而那是**上游改动**造成的，不是本脚本改的。

---

## 8. 基线与回归

```json
{
 "strict_worklist": 3,
 "wide_by_role_worklist": 5,
 "ledger_rows_blank": 9,
 "worklist_total": 31,
 "generated": 28,
 "min_junction_min": 0.05,
 "min_full_min": 0.05,
 "max_full_max": 0.95,
 "max_junction_max": 0.8,
 "shell_max": 0,
 "empty_junction_max": 2,
 "empty_junction_min": 2,
 "same_pool_max": 6,
 "n_divergent_terms_min": 3,
 "n_divergent_used_min": 50,
 "n_divergent_carriers_min": 28,
 "n_contracts": 139,
 "measured": {
  "generated": 28,
  "junction_max": 0.2,
  "full_max": 0.9118,
  "shell_count": 0,
  "n_divergent_terms": 56,
  "n_divergent_used": 56,
  "n_divergent_carriers": 28,
  "n_contracts": 139,
  "mounted": 28
 }
}
```


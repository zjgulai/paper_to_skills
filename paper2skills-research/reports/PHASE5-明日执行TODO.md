---
title: paper2skills PHASE5 —— 明日执行 TODO（2026-09-13）
doc_type: plan
module: 00-项目管理
status: active
created: 2026-09-12
owner: self
source: human+ai
supersedes: 实施方案与TODO.md 中未完成部分的执行视图
---

# PHASE5 明日执行 TODO（2026-09-13 周日）

> 本文件是 **2026-09-12 PHASE4 收工后的执行视图**，只回答一个问题：
> **明天这一天，按什么顺序做什么，每件事怎么算做完了。**
>
> 上游文档：`实施方案与TODO.md`（总计划）、`PHASE4-发现的内容缺陷.md`（缺陷台账）、
> `CLAUDE.md`（唯一信息源）。本文件只**引用**它们，不复制它们的结论。

---

## 0. 一页纸：明天怎么打

| 时段 | 内容 | 为什么排这个顺序 |
|------|------|------------------|
| **上午 P0** | 口径与门禁 5 件（X1–X5） | **不做 X1，P0-2 的修复成果无法度量** —— G2 通过率被「作者外推」结构性封顶在 ~16%，不动口径，明天再怎么修卡数字都不动 |
| **下午 P0** | 甲类卡片缺陷 Top5（Y1–Y5） | 这 5 张**不是精度问题，是卡片在讲另一篇论文**，读者据此做决策会错 |
| **收尾 P1** | 档案卫生 + registry 补齐（Z1–Z4） | 便宜、可并行，且直接影响 G2 证据链的可信度 |
| **顺延** | PHASE6 内容扩展（原 PHASE4 的 T4-1～T4-4）、15 篇萃取 backlog、PHASE5 常态化 | 不是明天的事，见 §3 |

**如果明天只有半天，只做 X1 + Y1 + Y2。** 理由见各自条目的「不做会怎样」。

---

## 1. 盘点：截至 2026-09-12 14:10 的真实状态

### 1.1 已完成（不再列入任何 TODO）

| 阶段 | 状态 | 证据 |
|------|------|------|
| PHASE 0（T0-1～T0-4） | ✅ 全部完成 | T0-2 后 `find … \| uniq -d` 为空；路径约定已进 `paper-维护` 检查范围 |
| PHASE 1（T1-2～T1-5） | ✅ 完成 | 4 个脚本 + `venue-whitelist.md` + `关键词库-v2.md` + `scoring_config.json` |
| PHASE 2（T2-1/3/5/6/7） | ✅ 完成 | 6 个门禁脚本均可 `--selftest` 自证 |
| PHASE 3（3A–3E，T3-1～T3-19） | ✅ **全部完成**（含原标「进行中」的 T3-18/T3-19，已在 3E 交付） | 18 张卡三门前全绿；4 张既有卡增强 |
| PHASE 4-A 存量卡证据链补齐 | ✅ 完成 | 129 张卡 +5,721 行；引文 729→2,087 条；0 伪造 |

### 1.2 未完成（本文件要解决的）

| # | 类别 | 规模 |
|---|------|------|
| A | 口径与门禁欠账 | 5 项（其中 3 项是**已实测的假绿灯/假红灯**，2 项是**工具自身崩溃或误报**） |
| B | 甲类卡片实质不符 | 13 张 |
| C | 乙类数字/口径偏差 | 7 张 |
| D | 丙类档案/工具卫生 | 8 项 |
| E | 原 PHASE4 内容扩展（T4-1～T4-4） | 4 项，**一项未动** |
| F | registry 欠账 | **12 篇** `extract` 无任何产出；8 条标题待补；70 张卡的 `paper_id` 不在 registry |
| G | 遗留工具（T1-1 迁移 / T1-6 / T2-4） | 3 项 |
| H | PHASE5 常态化（T5-1～T5-5） | 5 项 |
| I | 待业务方决策 | 4 项（其中 1 项是安全事项） |

### 1.3 ⚠️ 命名消歧：仓库里现在有两个「PHASE4」

| 名称 | 指什么 | 状态 |
|------|--------|------|
| **PHASE4（原计划）** | 空白领域补卡：T4-1 12-ML基础三张卡 / T4-2 17-跨境合规 / T4-3 11-AI人文改流程 / T4-4 06-增长模型供给策略 | ❌ **一项未执行** |
| **PHASE4-A（实际执行）** | 存量 146 张卡的证据链补齐（F3 补引文 + F4 补来源声明） | ✅ 完成 |

> **本文件起，原计划的 PHASE4 改称 PHASE6**，避免「PHASE4 已经做完了」这句话在
> 团队里指代错误的东西。`实施方案与TODO.md` 已同步加注。

---

## 2. 明日任务卡

### ⬛ P0 · 上午：口径与门禁（约 3.5 小时）

---

#### X1 · 「作者外推」标注机制（**明天最重要的一件事**）

**为什么必须做**：PHASE4 三个 F3 分组独立统计后得出同一结论 ——
**残余 G2 红灯的 75%–85% 是「作者外推」**：v1 五段式 ⑤「商业价值评估」与 ②「应用案例」里的
工期 / ROI / 倍数 / 示例数据，**论文里本就不存在**。

现在 `gate_check` 把它们与「论文断言」混在同一个分母里判红，后果是：
- 继续补引文**不会**让通过率上升（已经补到 0 伪造，还是 16.3%）；
- 团队看到的红灯数**不反映真实欠账**，无法据此排期；
- 更糟的是，它会诱导后来者去「把 ROI 数字找个论文出处」—— 那正是**制造假引文**的动机。

**动作**（四步，缺一步就是放水）：
1. 定义标注语法：在 ⑤ 段或某个表格**之前**加一行机器可读标记，例如
   `> **本段/本表为作者估算，非论文结论** <!-- extrapolation:scope=section -->`
2. `gate_check` 把落在标记作用域内的数字判为**第三态** `EXTRAPOLATION`（**不是 PASS**），
   单列 `n_extrapolated`；
3. **作用域必须封死**：标记只对 ⑤/② 段生效；标到 ① 算法原理或 ⑥ 原文引用段一律**判红**；
4. **反后门自测（必做）**：往一张当前全绿的卡中间插入一个**未标注**的 ROI 数字 → 必须重新变红。
   这条作为 `gate_check --selftest` 的新用例锁定。

**验收**：
```bash
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --selftest   # 含新用例，全绿
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --all \
  --k1 paper2skills-research/data/verification/k1_l5.json \
  --outdir paper2skills-vault/07-资源库/gates
# 输出必须同时给出三个数：论文断言通过率 / 外推标注数 / 未标注红灯数
```

**风险与对策**：这正是漏洞 #10 的翻版 ——「新增一个不阻塞的结局本身就是一次放水」。
所以第 4 步的反后门自测**不是可选项**；没有它，这个机制等于给全库发了免检章。

**预计**：2 小时（含 4 个 selftest 用例）

---

#### X2 · 修 G2c 的三处「结构性语境」误判（C6 / C7 / C8）

**为什么**：这三条都是 PHASE4 实测出来的，**已在 `PHASE4-发现的内容缺陷.md` 丙类登记**，
但一条未修。它们的共性是**用「语境」而不是「内容」判出处**，于是语境本身成了洗白工具：

| # | 缺陷 | 实例 |
|---|------|------|
| C6 | 参考文献编号 `[N]` 被当成结构性上下文 | `[8]` 洗白了 `8%`、`[5,6]` 洗白了 `6倍`、`Table 3` 洗白了 `3倍` |
| C7 | 正文里的日期串仍被当断言 | `**发表日期**: 2025-05-16` 拆出 `2025`/`16` 判红 |
| C8 | 卡片自己的 LaTeX 公式被误切 | `$t=0$` → 判「高价值断言 `0$` 无出处」 |

**动作**：
- C6：把 `[N]` / `[N,M]` 从「结构性前缀」白名单里剔除（它证明的是**编号**，不是**被编号的那个数**）
- C7：对 `20\d\d-\d\d-\d\d` 整体豁免，与 `strip_frontmatter` 同源处理
- C8：数字抽取跳过 `$...$` 公式段

**验收**：修完重跑 `--all`，逐条核对 C6/C7/C8 的三类实例**方向正确**
（C6 红灯应**上升**、C7/C8 黄灯/红灯应**下降**），并把三个方向的数字记进 CLAUDE.md。
⚠️ **必须逐类核对方向**，不能只看总数 —— 一类上升一类下降会互相抵消。

**预计**：45 分钟

---

#### X3 · `repo_health.py` 的两个 bug（**本次盘点新发现**）

**为什么**：这两个是**今天盘点时实测撞出来的**，未在任何文档里登记过。

**(a) `--json-out` 崩溃 —— PHASE5 常态化直接依赖它**

```bash
$ python3 paper2skills-skills/paper-维护/scripts/repo_health.py \
    --json-out paper2skills-research/data/health/repo_health.json
...
FileNotFoundError: [Errno 2] No such file or directory:
  'paper2skills-research/data/health/repo_health.json'
exit=1
```
体检报告**全部正常打印完**，最后一步写 JSON 时因 `data/health/` 目录不存在而崩，**退出码 1**。
后果：T5-4「周日体检 + 趋势追踪」拿不到可比较的历史文件，而且这个失败**看起来像体检失败**，
实际体检是通过的。修法：`args.json_out.parent.mkdir(parents=True, exist_ok=True)`。

**(b) C2 不认 `author-practice` 口径 —— 126 张误报**

C2 现报「有 frontmatter 但缺 v2 必填字段 **126 张**」。逐张核对后，**48 张是
`evidence_basis: author-practice` 的卡**，它们**按设计就不该有** `paper_id/paper/venue/venue_tier` ——
这正是 PHASE4 漏洞 #8/#10 建立的三类口径。C2 没跟上口径。

真实欠账（本次重算）：

| 项 | 数量 |
|----|------|
| 完全无 frontmatter | **2 张**（`Skill-Two-Echelon-Inventory-DRL.md`、`Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md`）|
| 有来源卡中缺 `paper_id` | 8 张 |
| 有来源卡中缺 `paper` | 29 张 |
| 有来源卡中缺 `venue` | 66 张 |
| 有来源卡中缺 `venue_tier` | 74 张 |
| 有来源卡中缺 `evidence_grade` | 78 张 |

**动作**：C2 拆成两个判定 ——「有来源卡缺 v2 字段」（真缺陷）与「author-practice 卡缺来源字段」（豁免）。

**验收**：C2 在 author-practice 卡上不再报警；同时**用构造样本自证仍会抓真缺陷**
（这就是 X2 那条「豁免条款必须比拦截条款测得更严」的同一条纪律）。

**预计**：40 分钟

---

#### X4 · `quote_check` 补 arXiv 版本号校验（C5）

**为什么**：现在的核验**不区分版本**。甲类 A6 就是这个漏洞的实例 ——
`Skill-Reflexion-Self-Improvement.md` 引的是 **NeurIPS 正式版**结论（HumanEval 91% pass@1），
而底本是 **arXiv v1**（只评测 AlfWorld 与 HotPotQA，`HumanEval`/`pass@1` 各 0 命中）。
**残句逐字存在于底本**，所以门禁绿；但绿的是**另一句话**。

**动作**：`fetch_fulltext.py` 落盘时把版本号写进底本头部（如 `2303.11366v1`）并纳入 `_index_keys`；
`quote_check` 报告里显示「卡片声称版本 vs 底本版本」，不一致时**降级为黄灯并显式提示**。

**验收**：`--selftest` 新增一个「v1 底本 + 正式版结论」用例；全库重跑，找出所有版本错配的卡。

**预计**：45 分钟

---

#### X5 · `provenance_audit` 的判据回归（**防止今晚的修改引入新误判**）

**为什么**：PHASE4 中它已经误判过一次 —— 只认 arXiv/DOI，把 7 张「用标题声明来源」的卡
判成「无论文来源」，F4 照章给 4 张加了**假声明**。
**判某个东西「不存在」之前先 `ls` 一次**，这条纪律已写进 CLAUDE.md，但工具本身没有回归网。

**动作**：给 `provenance_audit.py` 的「用标题声明来源」这条判据补 `--selftest` 用例
（构造一张只有 `paper:` 标题、无 ID 的卡，必须判为「有来源」而不是「无论文来源」）。

**验收**：`python3 paper2skills-research/scripts/provenance_audit.py --selftest` 全绿；
重跑后 `VERIFIED/RETROFIT_READY/NEEDS_FULLTEXT/NO_PAPER_SOURCE` 四层计数与今日一致
（`83/8/7/48`，`provenance_audit.json` 已存档可比对）。

**预计**：20 分钟

---

### ⬛ P0 · 下午：甲类卡片缺陷 Top5（约 3.5 小时）

> 甲类共 13 张，全部登记在 `PHASE4-发现的内容缺陷.md`。明天只做**风险最高的 5 张** ——
> 判据是「**这张卡会让读者以为论文说了它没说的话**」。其余 8 张顺延（§3）。

#### Y1 · `Skill-Uplift-Churn-Prediction.md` —— 把论文的负结果写成了正结果 ⚠️ **最高风险**

论文 §6 原文：*"the performance of the outcome RF model is consistently the highest,
showing that the uplift approach is not always preferable"*。
卡片却称「X-Learner 在 Qini 曲线上显著优于 T-Learner 和 S-Learner」，
而底本 `X-Learner`/`S-Learner`/`Qini` **零命中**（论文基准是 outcome RF / T-learner RF / Uplift RF + AUUC）。
另作者名写错（底本为 Verhelst / Mercier / Shrestha / Bontempi）。

**动作**：按底本重写 ①②⑤，把「uplift 更优」改为论文的真实结论（**普通预测模型反而最好**），
并把这条负结果写进 1b「反例与适用边界」—— 它本身就是最有价值的适用边界。

**验收**：卡内所有方法名/指标名在底本可检索到；三门前全绿。

---

#### Y2 · `Skill-Cold-Start-Product-Recommendation.md` —— 方法名不符 + 数字编造

卡片称方法为 `ColdLLM`（WSDM 2025、LLaMA-7B）；底本方法名是 **`LLM-InS`**，
标题为 *Large Language Model **Interaction** Simulator for Cold-Start **Item** Recommendation*，
全文 `ColdLLM`/`WSDM`/`LLaMA-7B` **零命中**。另 `10.79%` 在全文**不存在任何形式**。

**动作**：改方法名与出处；`10.79%` 删除或替换为底本真实数字（**不得保留无出处的数字**）。

**验收**：同上。

---

#### Y3 · `Skill-DQN-Purchase-Prediction.md` —— 把 supervised 任务描述成 RL 闭环

底本明确写 *"While not directly applicable in our supervised setting…"*；
卡片却称「Epsilon-Greedy 探索学会干预时机」。另 ③ 声称含 Attention，
而底本架构是 2 层 LSTM + BN + dropout + dense，**无 attention 组件**。

**动作**：删掉 RL 探索叙事，改为论文真实的监督式框架；③ 段与底本架构对齐。

---

#### Y4 · `Skill-User-Lifecycle-STAN.md` —— 把 MMOE 的机制写成 STAN 的机制

底本里 `gating network` **只出现在 MMOE baseline 的描述中**；STAN 自身是
latent stage representation + `Beta(α_k, β_k)`。
另 AIPL 阶段体系**非论文所有**（底本是 New/Wander/Stick/Loyal），
且 `3.05%` 被误标为「留存率」（论文是人均停留时长）。

**动作**：机制描述与底本对齐；阶段体系改为论文原名并注明「AIPL 为本项目映射」；
指标口径订正。

---

#### Y5 · `Skill-Auto-Skill-Synthesis.md` —— 没有底本证据的录用声明

卡片写「SIGIR 2026 Industry Track 阿里云已验证」，但底本全文 `SIGIR` **0 次**、首页无 venue 行。
与 `2606.29366`「首页写 EJOR 但无录用证据」同类。

**动作**：降为 `preprint`；或先取证（找会议名单/DOI）再恢复。
**默认按保守口径降级** —— 无证据的录用声明与假引文同级。

**验收（Y1–Y5 共同）**：
```bash
C=paper2skills-vault/<域>/Skill-<名>.md
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C"
python3 paper2skills-skills/paper-审核/scripts/quote_check.py   --card "$C"
python3 paper2skills-skills/paper-审核/scripts/gate_check.py    --card "$C"
```
三门前全绿，且**每张卡修完在 `PHASE4-发现的内容缺陷.md` 的对应行标注「已修 + commit」**
（台账必须闭环，否则下一轮会重复排查）。

**预计**：5 张 × 40 分钟

---

### ⬛ P1 · 收尾：档案卫生 + registry（约 2 小时，可并行）

#### Z1 · 隔离 `extract.md` 污染（丙类 C1 / C2）

- `papers/01-因果推断/uplift_model_2019/extract.md` 把 arXiv:1801.05045 说成
  Athey & Imbens 的 treatment effects 论文，**而该 ID 实为 hep-th 物理论文**，摘要亦为编造。
- `papers/02-A_B实验/mab_2019/extract.md`、`papers/03-时间序列/forecasting_2019/extract.md`
  的「论文摘要」是 LLM 风格**合成文本**（其中一条写着
  "cross-border e-commerce where ad creative testing is continuous" —— 非任何真实摘要）。

**动作**：① 给这两个目录的 `extract.md` 加显式作废头（或移入 `_superseded/`）；
② **在 `gate_check` 的证据收集里明确排除历史 `extract.md`**。
第 ② 条是重点 —— 否则等于**把合成引文接进证据链**。

**验收**：构造一张引用 `extract.md` 内容的卡，门禁必须**不认**它为出处。

#### Z2 · 补齐 2 张无 frontmatter 的卡（X3(b) 的真缺陷部分）

`Skill-Two-Echelon-Inventory-DRL.md`（供应链）、`Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md`（知识图谱）。
后者已在 3E 补了 18 条引文，**却仍无 frontmatter / evidence_basis** —— PHASE4 声称的
「frontmatter 146/146」实为 **144/146**，CLAUDE.md 需订正。

#### Z3 · registry 标题回填 8 条

`p2s-2026-0038`～`0045` 的 `title` 仍是 `(待补：见 note)`，
**标题实际写在 `decision_reason` 里**（如 0038 = *Algorithmic Pricing, Price Wars, and Tacit Collusion:
Evidence from E-Commerce*, Management Science 2026-07-09）。这是纯机械回填，10 分钟。

#### Z4 · registry 覆盖率登记（**登记，不在明天修**）

「卡片 `paper_id` 不在 registry」= **70 张**。这不是错误，而是 **registry 只覆盖 45 条本轮检索结果**，
而存量 146 张卡的历史论文来自更早的批次。明天只做一件事：
**在 registry 里加一个 `coverage_note` 说明这 70 张的归属规则**，
并让 `registry_consistency.py` 把覆盖率作为一等输出（它已有这个能力，需在报告里显式打印）。
**不要明天去补 70 条记录** —— 那是几天的活，且要先设计 schema 兼容。

---

## 3. 顺延清单（明确不做，避免明天被临时拉进来）

| # | 内容 | 顺延理由 |
|---|------|----------|
| S1 | 甲类剩余 8 张（A4/A5/A6/A7/A8/A11/A12/A13） | 风险低于 Top5；A6 的修法是**补抓全文**而非改卡，归入 S2 |
| S2 | 乙类 7 张（B1–B7） | 数字/口径偏差，性质较轻 |
| S3 | **PHASE6 内容扩展**（原 PHASE4 的 T4-1～T4-4） | 12-ML基础三张卡 / 17-跨境合规新域 / 11-AI人文改流程 / 06-增长模型供给策略 —— 需要独立一天 |
| S4 | **12 篇** `extract` backlog 出卡 | 每张卡 1–2 小时；其中 P0 三篇：`0007` 推荐系统 / `0012` 推荐系统 / `0015` Mitra-v2（12-ML基础） |
| S5 | T1-6 `registry.py` 读写工具 / T1-1 脚本迁到 `paper-选题/scripts/` / T2-4 `extract_code_blocks.py` | 工具债，不影响当前门禁（T2-4 已判定为「未采纳的备选方案」而非欠账） |
| S6 | C3 孤儿档案 `papers/02-A_B实验/1803.06258/paper.pdf` 裁决 | 需人工判断保留价值 |
| S7 | 硬编码路径 16 处（C3 检查） | 其中 `cd /Users/lute/project/...` 出现在**新生成的 evidence.md** 里，可批量脚本化 |
| S8 | PHASE5 常态化 T5-1～T5-5 | 常态化的前提是 X1/X3 先落地（否则周报数字无意义） |

---

## 4. 待业务方决策（**明天必须问出去**，否则相关卡会一直空转）

| # | 问题 | 卡住什么 |
|---|------|----------|
| Q1 | **`DDDD.pem` 私钥是否需要轮换？** 它曾长期明文躺在当时**无版本控制**的仓库根目录，且未被任何脚本引用 | **安全事项，优先级最高** —— 若曾用于生产（云主机 SSH / 支付回调验签），建议直接轮换 |
| Q2 | 投放主体是**自有独立站**还是**第三方平台店**？ | `Skill-Incrementality-Measurement` 的落地形态（前者可做受众级随机化，后者只能降级为诊断框架） |
| Q3 | 出海历史是否 ≥2 个完整年度（月度活跃面板）？ | `Skill-Seasonal-Aligned-Churn-Label` 是否需要品类季节性先验替代 |
| Q4 | 是否有 RCT / holdback 流量？ | `Skill-Funnel-Causal-Coupon-Allocation`、`Skill-Causal-Budget-Allocation` 的理论前提 |

> Q2–Q4 已挂在 `实施方案与TODO.md` 附 B，**从 PHASE3 挂到今天仍未回答**。
> 明天应把它们转成一句可直接回答的话问出去，而不是继续留在表格里。

---

## 5. 本次盘点新发现（4 条，均已写进 CLAUDE.md）

| # | 发现 | 性质 |
|---|------|------|
| N1 | **`repo_health.py --json-out` 因父目录不存在而崩溃**（退出码 1，报告本身却是通过的） | 工具 bug，**PHASE5 常态化直接依赖它** → X3(a) |
| N2 | **`repo_health` C2 不认 `author-practice` 口径**，把 48 张设计上就无来源字段的卡算成「缺 v2 字段」，报出 126 张的假欠账 | 口径未跟上漏洞 #8/#10 → X3(b) |
| N3 | **CLAUDE.md 的「frontmatter 146/146」不成立**，实为 144/146；且其中一张（GraphRAG）**刚在 3E 被补过引文却仍无 frontmatter** | 文档与实物不一致 → Z2 |
| N4 | **`registry` 的「未出卡」统计会漏掉增强交付** —— 只数 `outputs.skill_card` 会把 `0023`/`0026`/`0030` 三篇**已作为 3E 增强交付**的论文误记成 backlog（15 篇 vs 真值 **12 篇**） | 判据只认一种字段 → 与漏洞 #11 同源。**统计脚本必须同时看 `skill_card` 与 `enhanced_cards`** |

> N1/N2/N4 有共同点，与 PHASE4 的三次教训同源：
> **「提升来自修门禁而非改资产」在本仓库已发生 4 次。**
> 所以每次盘点都要**主动去撞工具**，而不是只看它有没有报错 ——
> 本次盘点撞出的 N1（崩溃）、N2（口径失真）、N4（统计漏项）**都没有被任何门禁报警**。

---

## 6. 明日收工验收（照抄执行）

```bash
cd /Users/lute/project/paper_to_skills

# 1) 六个脚本自证可信
for s in paper2skills-skills/paper-萃取/scripts/verify_skill_code.py \
         paper2skills-skills/paper-审核/scripts/gate_check.py \
         paper2skills-skills/paper-审核/scripts/quote_check.py \
         paper2skills-skills/paper-维护/scripts/repo_health.py \
         paper2skills-research/scripts/provenance_audit.py \
         paper2skills-research/scripts/registry_consistency.py; do
  echo "== $s"; python3 "$s" --selftest || echo "❌ $s selftest 失败"
done

# 2) 全量重跑三门禁 + 引文核验 + 体检
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --all --level 5 --timeout 30 \
  --json-out paper2skills-research/data/verification/k1_l5.json
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --all \
  --k1 paper2skills-research/data/verification/k1_l5.json \
  --outdir paper2skills-vault/07-资源库/gates
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --all --json \
  --json-out paper2skills-research/data/quote_check_all.json
python3 paper2skills-skills/paper-维护/scripts/repo_health.py \
  --json-out paper2skills-research/data/health/repo_health.json   # X3(a) 修完这句才不崩

# 3) 汇报口径（三个数必须分开报，禁止合并成「可信度总分」）
#    论文断言通过率 / 外推标注数 / 未标注红灯数
```

**收工时的硬要求**：
1. `git status` 干净，卡片改动与基础设施改动**分开提交**；
2. `PHASE4-发现的内容缺陷.md` 里已修的条目**逐行标注 commit**；
3. CLAUDE.md 的存量资产体检表**更新到明天的新数字**，并写明「与昨日不可直接对比」的原因。

---

## 附 · 明日数字基线（2026-09-12 14:10 实测，供明天对比）

| 指标 | 值 |
|------|-----|
| 卡片总数 | 146 |
| K1 执行率 | 62.0%（100 单元：PASS 62 / ENV_BLOCKED 7 / MIGRATED_DEP 9 / FAIL 22 / ORPHAN 0） |
| G1 | 42.5%（62/146，红灯 68 / 黄灯 7） |
| G2 | 16.3%（16/98 可核验分母，红灯 1,476 / 黄灯 645，48 张无法核验） |
| G3 | 45.9%（67/146，红灯 93 / 黄灯 174） |
| 引文逐字核验 | 90 张 VERBATIM / 2 张 NO_FULLTEXT；**2,087 条，0 伪造 0 近似 0 拼接** |
| provenance 四层 | VERIFIED 83 / RETROFIT_READY 8 / NEEDS_FULLTEXT 7 / NO_PAPER_SOURCE 48 |
| registry 45 条 | **已交付卡 16 + 已交付为增强 3 + 真·待萃取 12 + watch 14** |
| repo_health | C1 ✅ / C2 ⚠️(口径失真) / C3 🟡16 / C4 ✅ / C5 ✅(覆盖率待显式) / C6 ✅ / C7 ✅ / C8 🟡58 |

> ⚠️ **统计「未出卡」时必须同时看 `outputs.skill_card` 与 `outputs.enhanced_cards`。**
> 本次盘点第一遍只数了 `skill_card`，把 `0023`/`0026`/`0030` 三篇**已作为 3E 增强交付**的论文
> 误记成「待萃取」，得出 15 篇（真值 12 篇）。**这与漏洞 #11 同源：判据只认一种字段，
> 就会把另一种形态的产出漏掉。**

> ⚠️ **不要拿「红灯总数」当进度指标。** PHASE4 已证明：修门禁会让红灯**上升**（#14 使 1,308→1,476）。
> 明天 X1 落地后，G2 的红灯还会再变一次。**要看的是「论文断言通过率」与「未标注红灯数」两个分开的数。**

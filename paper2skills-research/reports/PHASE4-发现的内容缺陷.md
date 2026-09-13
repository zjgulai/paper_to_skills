# PHASE4 发现的存量卡内容缺陷（不改卡，只登记）

> 本文件是 PHASE4「补引文」过程**反向发现**的卡片内容问题清单。
> 按本轮既定口径（红线 4「标注不删」），**这些卡片一律未被修改** ——
> 补引文只新增 ⑥ 段，不动既有正文与数字。故此处登记，供下一轮专项修复。
>
> **为什么这类发现比数字对不上更严重**：数字偏差是精度问题，
> 而「方法名写错 / 把负结果写成正结果 / 编造录用声明」是**卡片在讲另一篇论文**，
> 会让读者据此做出错误决策。下表按严重度排序。

## 甲类 · 卡片与论文实质不符（必须修，不是润色）

| # | 卡片 | 问题 | 证据 |
|---|------|------|------|
| A1 | `06-增长模型/Skill-Uplift-Churn-Prediction.md` | **把论文的负结果写成了正结果**。卡片称「X-Learner 在 Qini 曲线上显著优于 T-Learner 和 S-Learner」；底本 `X-Learner`/`S-Learner`/`Qini` **零命中**，论文基准是 outcome RF / T-learner RF / Uplift RF + AUUC，且结论**相反**：§6「the performance of the outcome RF model is consistently the highest, showing that the uplift approach is not always preferable」。另作者名写错（底本作者为 Verhelst / Mercier / Shrestha / Bontempi） | 子代理 F3 group 1 逐词 grep |
| A2 | `06-增长模型/Skill-Cold-Start-Product-Recommendation.md` | **方法名与论文不一致**：卡片称 `ColdLLM`（WSDM 2025、LLaMA-7B）；底本方法名是 **LLM-InS**，标题为 *Large Language Model **Interaction** Simulator for Cold-Start **Item** Recommendation*，全文 `ColdLLM`/`WSDM`/`LLaMA-7B` 零命中。另 `10.79%` 在全文**不存在任何形式**，疑似编造 | 同上 |
| A3 | `16-智能体工程/Skill-Auto-Skill-Synthesis.md` | **没有底本证据的录用声明**：卡片写「SIGIR 2026 Industry Track 阿里云已验证」，但底本全文 `SIGIR` **0 次**、首页无任何 venue 行。与 `2606.29366`「首页写 EJOR 但无录用证据」同类 → 应降为 `preprint` 或先取证 | 子代理 F3 group 2 |
| A4 | `16-智能体工程/Skill-Agent-Stage-Evaluation.md` | 数字与底本不符：卡片写 GPT-4o「82.30 vs Claude 84.21」，底本 Table 3 里 GPT-4o Average 是 **83.17**；§4.3 只有定性表述 | 同上 |
| A5 | `16-智能体工程/Skill-Co-Evolutionary-Skill-Verification.md` | 跨行混用 + 归因错：卡片写「65% 跌到 8%（对应 Mistral Large 3 baseline）」；底本 Table A3 里 `65.0` 是 **GPT-5.2 挂 Opus skill**、`8.4` 是 **Qwen3 Coder 无 skill**、而 Mistral Large 3 的 no-skill 是 **4.9**。另 `+18pp ~ +40pp` 是两个不同对照被并成一个区间 | 同上 |
| A6 | `10-MAS/00-知识库-Skill卡片/Skill-Reflexion-Self-Improvement.md` | **底本是 arXiv v1，卡片引的是 NeurIPS 正式版结论**：⑤「HumanEval 91% pass@1」在底本中 `HumanEval`/`pass@1` **各 0 命中**（v1 只评测 AlfWorld 与 HotPotQA）；卡片标题用的方法标签「Verbal Reinforcement Learning」也 0 命中。**修法**：补抓 `2303.11366v3+` 或 NeurIPS 版全文，而非改卡片 | 子代理 F3 group 6 |
| A7 | `10-MAS/00-知识库-Skill卡片/Skill-MetaGPT-SOP-Driven-Collaboration.md` | 3 个数字疑似二手信息：①「提升 5.4%」、⑤「HumanEval 85.9% / MBPP 87.7%」在底本**全部 0 命中**；论文实际报告的是 516 秒 + $1.12、成功率 51.43%。疑来自官方仓库 README | 同上 |
| A8 | `10-MAS/00-知识库-Skill卡片/Skill-Agent-Memory-Learning.md` | 阈值不是论文结论：①「上下文窗口 ~70% 预警、100% 强制」在底本 0 命中（论文只写 "warnings regarding token limitations"，未给阈值）；③「Main Context 8K-128K」中 128K 不在论文 Table 1（2k/4k/8k/16k/32k/100k）；5 个 `*_memory_*` 函数名底本全无 | 同上 |
| A9 | `06-增长模型/Skill-DQN-Purchase-Prediction.md` | **把 supervised 任务描述成 RL 闭环**：卡片称「Epsilon-Greedy 探索学会干预时机」；底本明确 "While not directly applicable in our supervised setting…"；③ 声称含 Attention，但架构是 2 层 LSTM + BN + dropout + dense，**无 attention 组件** | 子代理 F3 group 1 |
| A10 | `06-增长模型/Skill-User-Lifecycle-STAN.md` | **把 MMOE 的机制写成 STAN 的机制**：卡片用「门控网络」描述阶段表示，而底本里 `gating network` 只出现在 **MMOE baseline** 的描述中；STAN 自身是 latent stage representation + `Beta(α_k, β_k)`。另 AIPL 阶段体系非论文所有（底本是 New/Wander/Stick/Loyal），`3.05%` 被误标为「留存率」，论文是人均停留时长 | 同上 |
| A11 | `05-推荐系统/Skill-Session-Based-Recommendation-SR-GNN.md` | 注意力公式与论文式 (6)–(7) 不同（卡片把 `s_g` 放进 α 的 sigmoid 内；底本 α 用 local embedding、最终表示是拼接后线性变换）；论用 gated GNN（GRU 门控），卡片实现的是加权求和传播。另 `8.3%` 出处不明（对 NARM 实为 8.07%，对最强基线 STAMP 仅 4.3%） | 同上 |
| A12 | `10-MAS/00-知识库-Skill卡片/Skill-CAMEL-Role-Playing-Agents.md` | 方法名口径：**论文没有把框架命名为 "CAMEL"** —— 摘要自述是 "a novel communicative agent framework named role-playing"，全文 `CAMEL` 仅 3 处且都是数据集/终止符名。另参考论文标题漏 `Scale` | 子代理 F3 group 6 |
| A13 | `10-MAS/00-知识库-Skill卡片/Skill-Multi-Agent-Debate.md` | 选择性引用：⑤「GPT-3.5 + MAD 超越 GPT-4」**只在 Commonsense MT 上成立**；论文 §4.3 有反例（Counter-Intuitive AR 上 MAD 不如 GPT-4）。且 ①「Judge 能公正裁决」假设被论文 §5 **证伪**（judge 偏向同 backbone）。另作者机构写错（CUHK 应为清华 / SJTU） | 同上 |

## 乙类 · 数字/口径偏差（应修，但性质较轻）

| # | 卡片 | 问题 |
|---|------|------|
| B1 | `07-NLP-VOC/.../Skill-NPS-Driver-Analysis.md` | **指标属本论文、方法学属另一篇**：底本 `SHAP`/`ridge`/`permutation`/`LASSO` **全 0**；论文真实归因法是**属性级线性回归系数**；`NPS` 只出现 1 次且是类比（论文因变量是 Yelp 星级）。卡片尾部声明的补充论文 `XCom-SHAP 2603.01212` 本仓库**只有 PDF 无 fulltext** |
| B2 | `16-智能体工程/Skill-MCP-A2A-Protocol-Stack.md` | §VIII 全部案例数字是**转引**（底本标 `[15]/[16]/[17]`），非本论文自测；卡片据此推 ROI 时应标明二手来源 |
| B3 | `12-ML基础/Skill-Customer-Journey-Prototype`（`06-增长模型/`） | 序列距离公式与论文不同：卡片 `d = min_ops / max(len)`；底本 §2.2.1 是按 stage 加权求和、**不除以 max(len)**；原型检测算法（k-medoids++ + 轮廓系数）也与卡片写的 argmax-min 不同 |
| B4 | `07-NLP-VOC/.../Skill-OfflineRL-触达时机优化.md` | 方法名错：卡片写「离线RL (CQL)」，论文实际用 **Double DQN**（全文 `CQL` 只出现在参考文献 [16]）。另 3 组数字与论文口径差很远（卡 +15-20% vs 论文 CTR +4.53%；卡 -30% vs 论文 unfollow -4.37%；`18%` 全文零命中） |
| B5 | `10-MAS/.../Skill-MAS-Multi-Objective-Recommendation.md` | 框架写反：三件套方法名全命中（是同一篇），但卡片「每个 Agent 负责一个**目标**」是作者自创，论文自述是把推荐**阶段**建模为协作 Agent |
| B6 | `10-MAS/.../Skill-MAS-VOC-Data-Analyst.md` | `27 agents` 对得上，但卡片 ③ 列的 `DataIngestionAgent`/`ThematicAnalysisAgent` 等**类名零命中**；论文真实 agent 名是 `Agent Summary`/`Agent Coders` 等 |
| B7 | `（数据点）Skill-MAS-MAS-MARL-Dynamic-Pricing` 等 | 见各卡 ⑥ 段注记 |

## 丙类 · 工具/仓库卫生（影响门禁可信度，已修或待修）

| # | 问题 | 状态 |
|---|------|------|
| C1 | `papers/01-因果推断/uplift_model_2019/extract.md` **把 arXiv:1801.05045 说成 Athey & Imbens 的 treatment effects 论文**，而该 ID 实为 hep-th 物理论文；同目录 PDF 就是那篇物理文。摘要亦为编造 | ✅ **已修**（`c9e8151`）：加文件内作废头，逐条写明该份错在哪。另 `cead938` 让 `gate_check` **硬拦截** |
| C2 | `papers/02-A_B实验/mab_2019/extract.md`、`papers/03-时间序列/forecasting_2019/extract.md` 的「论文摘要」是 LLM 风格**合成文本**（其中一条还写着 "cross-border e-commerce where ad creative testing is continuous"，非任何真实摘要）；旁边才是真本 PDF（1707.02038 / 1912.09363） | ✅ **已修**（`c9e8151` 作废头 + `cead938` 硬拦截）。**C2 末尾的建议已按原话实现**：G2 证据收集现在明确排除历史 `extract.md`。⚠️ 实测补充：全库实有 **15 份** `extract.md`（台账只点名 3 份），另 12 份**未审计** → 判据对**全部 15 份**生效，使「其余是否干净」不再影响正确性 |
| C3 | `papers/02-A_B实验/1803.06258/paper.pdf`（Liu & Chamberlain）**全库无任何卡片引用** | 孤儿档案，待裁决保留/清理 |
| C4 | 底本含 **Unicode 细空格**（`+4.5\u2009pp`），肉眼与普通空格无异，手写锚点必然失配 | 已确认 `quote_check.normalize()` 的 NFKC 会等同二者；抽取工具须先归一化 |
| C5 | `quote_check` **不校验 arXiv 版本号** | **待修**：`fetch_fulltext.py` 落盘时应把版本号写进底本头部（如 `2303.11366v1`）并纳入 `_index_keys`，否则「引正式版结论、底本是 v1」会静默通过（A6 即此情形） |
| C6 | G2c 未把**参考文献编号 `[N]`** 视为结构性上下文 | **待修**：实测 `[8]` 洗白了 `8%`、`[5,6]` 洗白了 `6倍`、`Table 3` 洗白了 `3倍` |
| C7 | `strip_frontmatter` 只剥 YAML 块，**正文日期串**（`**发表日期**: 2025-05-16`）仍被当断言 | **待修**：对 `20\d\d-\d\d-\d\d` 做同样豁免 |
| C8 | 卡片自身 LaTeX 公式被误切：`$t=0$` → 判「高价值断言 `0$` 无出处」；`\mathcal{R}{=}1` → `1$` | **待修**：数字抽取应跳过 `$...$` 公式段 |

## 已在本轮修复（记录，避免重复排查）

| 修复 | 内容 |
|------|------|
| 漏洞 #13 自引文洗白 | `QUOTE_RE` 引号不成对 + `re.S` 跨行吞正文 → 卡片自己的声明被当成"引文"，自己的数字变"有出处"。实测 `Skill-AB-Experimental-Design` 假 traceability=76.9%（`sourced=20`），修后为真实的 `sourced=0`、红灯 27。两层都修：① 引号成对同型 + 去掉 `re.S`；② `UNVERIFIABLE` 不再计入出处。`--selftest` 用例 6/7 锁定 |
| 漏洞 #12 frontmatter 当断言 | `created`/`updated` 日期与 `paper_id` 被当数字扫描 → 65 张卡 126 条假黄灯。修后黄灯 767→575、红灯 2,649→1,308 |
| 判定误收 | 7 张「用标题声明来源、无 ID」的卡曾被误判为「无论文来源」，并让 F4 给 4 张加上了「无对应论文来源」的**假声明**（与卡片自己的 ③ 段 docstring 自相矛盾）。已修 `provenance_audit` 判据并逐张订正声明 |

## 数据点 · 门禁口径对「作者外推」的判断

三个 F3 分组独立得出同一结论：**补引文后剩余红灯的主体是「作者外推」**，
即 v1 五段式的 ⑤「商业价值评估」与 ②「应用案例」里的工期/ROI/倍数/示例数据。

- group 2（16-智能体工程 9 张）：剩余 170 条红灯中 **128 条（75%）** 属作者外推
- group 6（10-MAS 9 张）：约 **85%** 属作者外推
- group 1（10 张）：同样以 ROI/业务假设为主

这些数字**论文里本就不存在**，补引文不可能也不应该消除它们。
→ **结论：G2 的可核验通过率有结构性上限**，除非新增一个「业务外推」的显式标注机制
（例如要求 ROI 表标注「本表为作者估算」并把它从 G2 断言集中排除）。
这是下一轮该做的**口径设计**，而不是继续补引文。

---

## 甲类逐条处置（2026-09-13，PHASE5 下午段）

> **索引消歧**：PHASE5 TODO 里的 Y1–Y5 与本表的 A 编号**不是同一套编号**。
> 对应关系：**Y1=A1、Y2=A2、Y3=A9、Y4=A10、Y5=A3**（按「读者会据此做错决策」的风险排的 Top5）。

| # | 卡片 | 状态 | commit |
|---|------|------|--------|
| **A1** | `Skill-Uplift-Churn-Prediction` | ✅ **已修**（真缺陷） | `22e625b` |
| **A2** | `Skill-Cold-Start-Product-Recommendation` | ✅ **已修**，但**原判「编造」不成立** —— 见下方订正 | `e00e80d` |
| **A3** | `Skill-Auto-Skill-Synthesis` | ✅ **已修**，但**原处置「应降为 preprint」是错的** —— 见下方订正 | `1c9949d` |
| **A9** | `Skill-DQN-Purchase-Prediction` | ✅ **已修**（真缺陷），另核出 2 处新不符 | `d879300` |
| **A10** | `Skill-User-Lifecycle-STAN` | ✅ **已修**，但**归因写法必须改** —— 见下方订正 | `c248957` |
| A4/A5/A6/A7/A8/A11/A12/A13 | 其余 8 张 | ⏸ **顺延**（PHASE5 TODO §3 S1）。⚠️ **A6 的修法是「补抓正式版全文」而非改卡**，与 A2 同型 | — |

### ⚠️⚠️ 三条「缺陷」经复核**不成立** —— 台账的取证方法本身错了

这三条我都派了子代理，其中**两条被子代理当场拒绝执行并升级**（A2、A3）。
逐条复核后确认**它们不是卡片缺陷**，而是**台账的取证方法有范畴错误**：

| # | 台账原判 | 复核结论 | 证据 |
|---|---------|---------|------|
| **A2** | 卡片称 `ColdLLM`/WSDM 2025/`10.79%` 而底本是 `LLM-InS`，**「疑似编造」** | ❌ **不成立**。`2402.09176` 有 **v1/v2 两版**，仓库底本抓的是 **v1**，而**卡片写的是 v2（WSDM 2025 正式版）**。台账称「编造」的每一项都在 v2 **逐字命中** | `ColdLLM` 在 v2 命中 62 次；arXiv abs `Comments: accepted by WSDM 2025`；DOI `10.1145/3701551.3703546` |
| **A3** | 卡片写「SIGIR 2026 Industry Track」而底本 `SIGIR` 0 次，**「无录用证据 → 应降为 preprint」** | ❌ **不成立**。**论文正文从不自述 venue**，故底本正文对 venue 声明而言是**错误的仪器**。录用为真 | arXiv abs v2 `Comments: Accepted at ACM SIGIR 2026 Industry Track.`；ACM DOI `10.1145/3805712.3808466`（SIGIR '26 proceedings，**pages 4763-4768**）；本仓库 `venue-whitelist.md` §8 自己就登记着该 DOI 前缀 |
| **A1** 附带 | 子代理报 `ECML PKDD 2023 Workshop` 底本零命中 →「疑似版本错配，待定」 | ❌ **不成立**。该 venue 为真，证据同样在底本之外 | arXiv abs `Comments: …post-proceedings of the ECML PKDD 2023 Workshop on Uplift Modeling and Causal Machine Learning for Operational Decision Making` |

**根因（一条，三个变体）**：台账把「**与本地 v1 底本不一致**」直接等同于「**卡片与论文实质不符**」。
但卡片追的往往是**正式发表版**，而 venue 这类事实**本来就不写在论文正文里**。

> ### 📌 由此定下的两条取证规则（凌驾于本表原判法）
> 1. **venue / 发表类声明 → 一律用 arXiv 元数据（`Comments` / `journal_ref`）与出版方 DOI 判定，
>    不得用底本正文判定。** 正文里 0 命中是**常态，不是反证**。
> 2. **内容类声明 → 必须先问「卡片写的是哪一版」，再判对错。**
>    底本与卡片版本不一致时，先怀疑**底本抓错版本**，而不是先怀疑卡片。
>
> 这两条与本仓库既有教训**同源**：漏洞 #11（判据只认一种字段名）、K1 的 `ORPHAN_DEP`
> （判据只认一种路径）、C2 口径未跟上口径。**这是第四种变体：判据用错了信息源。**

### A10 的归因写法必须改（否则下一个人一次 grep 就能推翻整条台账）
台账原文写「底本里 `gating network` **只出现在 MMOE baseline 的描述中**」——**不成立**。
实测 2 处命中：① Fig. 4 **本文框架图**的标签；② **§3.1 STAN 自己的 backbone**。
MMOE baseline 用的词是 `gates`。
**结论（卡片把 §3.1 backbone 的门控写成了阶段表示机制）不变**，但归因必须按实测写。

### A2 留下的一个结构性教训（值得单列）
卡片正文依 **v2**、而 ⑥ 段 11 条引文里 **10 条是 v1 特有措辞** ——
`quote_check` 之所以一直报 VERBATIM，**只是因为底本恰好是 v1**。
即：**「逐字核验通过」证明不了「引文与正文同源」**。
已按 A 案换 v2 底本（v1 留档为 `fulltext.v1.md`），引文全部改写为 v2 逐字，版本错配黄灯清零。

### 本轮另在甲类卡片上新核出、并经登记的问题（非 A 类原判）
| 卡片 | 新发现问题 | 处置 |
|------|-----------|------|
| `Skill-Uplift-Churn-Prediction` | ③ 段引用的 `paper2skills-code/growth_model/uplift_churn_prediction/model.py` 内含 `SLearner`/`XLearner`/`Qini`，**与论文基准（outcome RF / uplift RF / T-learner RF + AUUC）不符** | 未改（超出「保守修改」范围），已在卡内加显式注记隔断误导。**建议另开一单**同步该代码包 |
| `Skill-DQN-Purchase-Prediction` | ①「行业基准 AUC-ROC 0.65-0.75」**编造**；② 论文**自己的表 V** 是 `0.6257`，**低于它同时评测的全部传统基线**（LR 0.8152 / RF 0.8599 / XGBoost 0.8366） | ✅ 已修（删编造值、优先级 5/5→3/5）。另：`model.py` 仍是 bidirectional LSTM + 自注意力，**与 §III-C 不符** → 同上，建议单独派单 |
| `Skill-Auto-Skill-Synthesis` | 参考的 `arxiv:2602.12430` **是另一篇论文**（底本 [5] 实为 Anthropic 工程博客，未给 arXiv ID） | ✅ 已修（`1c9949d`） |

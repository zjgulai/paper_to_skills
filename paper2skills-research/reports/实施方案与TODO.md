---
title: paper2skills 萃取链路升级 —— 实施方案与 TODO
doc_type: plan
module: 00-项目管理
status: active
created: 2026-09-12
updated: 2026-09-12
owner: self
source: human+ai
execution_view: paper2skills-research/reports/PHASE5-明日执行TODO.md
---

# paper2skills 萃取链路升级 —— 实施方案与TODO

> 配套文档：`paper2skills-research/reports/paper2skills-萃取链路升级方案-v2.md`（设计）、
> `paper2skills-research/reports/近三个月论文更新与推荐清单.md`（本轮论文结论）、
> `paper2skills-vault/07-资源库/papers_registry.json`（唯一事实源，45 条决策记录）
>
> 📌 **本文件是「总计划」（阶段划分与验收标准），不是「执行视图」。**
> 每日/每轮该做什么，看在 `paper2skills-research/reports/PHASE5-明日执行TODO.md`
> —— 那一份每次收工后重写，本文件只在**阶段边界变化**时更新。
>
> ⚠️ **2026-09-12 命名消歧（重要）**：
> - **PHASE4-A** = 实际执行的「存量卡证据链补齐」（129 张卡补引文/来源声明）→ ✅ 完成，见 §PHASE4-A
> - **PHASE4（本节 §PHASE4）** = 原计划的「空白领域补卡」→ ❌ **一项未执行**，
>   自 2026-09-12 起**改称 PHASE6**，以免「PHASE4 做完了」被误解

## 0. 总览

| 阶段 | 目标 | 交付物 | 预计工作量 |
|------|------|--------|-----------|
| **PHASE 0** | 止血：修失效路径 + 清重复卡片 | 仓库可用 | 1 小时 |
| **PHASE 1** | 补齐检索基建：多源检索 + 三段式关键词 + venue 白名单 | 4 个脚本 + 2 份资源库文档 | 半天 |
| **PHASE 2** | 萃取流水线化：registry + 证据规则 + 双门禁脚本 | 3 个脚本 + MasterPrompt v2 + 卡片模板 v2 | 1 天 |
| **PHASE 3** | 首批萃取：P0 队列 19 篇 → 出 8-10 张卡 | 卡片 + 代码 + 验证报告 | 分批，每张卡 1-2 小时 |
| **PHASE 4A** | 存量 146 张卡证据链补齐（F3 补引文 / F4 补来源声明） | 129 张卡 + 门禁加固 | ✅ 已完成 |
| **PHASE 6** | ~~PHASE 4~~ 空白领域补卡：12-ML基础 / 17-跨境合规 | 3+3 张卡 | 1 天（**未开始**） |
| **PHASE 5** | 常态化：周更 + 周体检 + 季度复盘 | 例行流程 | 每周 1 小时 |

**验收总标准**：连续两周的周更短名单 ≤20 条且有分数与理由；每张新卡 G1/G2/G3 全绿；仓库周检重复卡片 0、frontmatter 缺失 0、路径失效 0。

---

## PHASE 0 · 止血（1 小时）

- [x] **T0-1 修正失效的绝对路径** ✅ 本轮已完成
  - `paper-同步/scripts/sync.py` 的 `BASE_DIR` 改为按脚本位置反推 + 支持 `PAPER2SKILLS_ROOT` 环境变量；DOMAINS 映射从 6 个补到 18 个
  - 6 个 SKILL.md / 文档里的 `/Users/pray/project/paper_to_skills` → `<REPO_ROOT>`
  - `paper-skills-graph/scripts/skills_graph_analyzer.py` 默认路径改为相对定位
  - 验收：`python3 paper2skills-skills/paper-同步/scripts/sync.py --status` 能正常列出同步状态 ✅

- [x] **T0-2 清理 26 组重复 Skill 卡片**
  - 事实：`07-NLP-VOC/` 下 26 组同名卡片各存两份（25 组字节相同，`Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md` 两份已漂移 6 字节）
  - 动作：① 人工比对漂移的那一份，决定保留版本；② 以 `00-知识库-Skill卡片/` 为唯一存放地（与 10-MAS 结构一致），删除顶层副本；③ 删除前 `git`/备份
  - 验收：`find paper2skills-vault -name 'Skill-*.md' | sed 's#.*/##' | sort | uniq -d` 输出为空

- [x] **T0-3 建立"路径约定"并写进 `paper-维护` 的检查范围**
  - 事实：本轮清理了 `paper2skills-skills/` 下 22 处失效的 `/Users/pray/project/paper_to_skills` 硬编码
    （6 个 SKILL.md/文档 + `sync.py` + `skills_graph_analyzer.py`），仓库内仍存在的 `/Users/pray` 全部是
    **有意保留**的两类：① `paper2skills-code/nlp_voc/**` 与迁移说明（指向已迁出的 `../ai_nlp_voc/`）；
    ② `logs/` 与 `evolve/round-1/` 历史归档（不可改写）与 `01-MasterPrompt设计/` 草稿。
  - 约定：**可执行代码中的路径必须相对化**（`Path(__file__).resolve().parents[N]` 或 `PAPER2SKILLS_ROOT` 环境变量）；
    文档用 `<REPO_ROOT>` 占位；历史归档不改写，只加时效注记。
  - 验收：`paper-维护` 的路径检查只对 `paper2skills-skills/**` 与 `paper2skills-code/**`（排除 `nlp_voc/`）报警

- [x] **T0-4 新建 `paper2skills-research/` 的定位说明**
  - 说明该目录是"调研与流水线工程"工作区（脚本 + 中间数据 + 报告），不属于交付的 vault/code
  - 在 `CLAUDE.md` 的项目结构中补一行（✅ 本轮已完成）

---

## PHASE 1 · 检索基建（半天）

- [ ] **T1-1 沉淀 arXiv 收割脚本**（脚本已完成，**迁移到正式位置未做**）
  - 现状：`paper2skills-research/scripts/arxiv_harvest.py`（49 查询组，本轮跑出 1046 篇）
  - 动作：复制到 `paper2skills-skills/paper-选题/scripts/arxiv_harvest.py`
  - ⚠️ 2026-09-12 盘点核实：`paper2skills-skills/paper-选题/` **下没有 `scripts/` 目录**，迁移未执行
  - 影响：低（脚本可用，只是不在 `paper-选题` skill 的预期位置）

- [x] **T1-2 沉淀 Crossref 期刊收割脚本**（✅ 已完成）
  - 现状：`paper2skills-research/scripts/journal_harvest.py`（28 刊，本轮跑出 1751 篇）
  - 验收：`--days 7` 能在 1 分钟内跑完增量

- [x] **T1-3 新建 `paper2skills-vault/07-资源库/venue-whitelist.md`**
  - 内容：顶会/顶刊白名单（UTD24 / FT50 / CCF-A/B / 领域顶会）+ workshop/findings/demo 降级规则
  - 必须收录本轮实测发现的 6 处抬级案例作为反例
  - 验收：给出"arXiv comment 含 Workshop/Findings/Demonstrations/Under review/manuscript → 降级"的判定表

- [x] **T1-4 新建 `paper2skills-vault/07-资源库/关键词库-v2.md`（三段式）** ✅ 并落地为可执行过滤器 `candidate_filter.py`（实测丢弃 25.8%，逐篇抽检无误杀）
  - 正向词：现有 16 域的检索词（从 `关键词库.md` 继承）
  - **负向词**（本轮实测必需）：`software supply chain`、`SBOM`、`package detection`、`model lineage`、`LLM inference`、`MRI`、`EEG`、`radiotherapy`、`clinical`、`lesion`、`PET`（04 域污染率 27%、14 域 64%）
  - **约束词**：14 域加 `+e-commerce/retail/subscription/churn/retention`；04 域加 `+inventory/fulfillment/warehouse`
  - 拆词：07-VOC 域把 `product review summarization` / `opinion mining` / `multimodal review` / `aspect extraction` 分成 4 条独立查询（本轮 92 天仅命中 9 篇 = 过窄）

- [x] **T1-5 评分器落地为脚本**（✅ 已迁移 + 参数化）
  - 现状：`rank_candidates.py`（6 维评分，本轮跑出 19 篇 ≥60 分）
  - 动作：把阈值、权重、缺口关键词表抽到 `paper2skills-vault/07-资源库/scoring_config.json`

- [ ] **T1-6 建立 `papers_registry.json` 读写工具**
  - 现状：`build_registry.py`（初版已生成 45 条）
  - 新增：`registry.py` 提供 `add/update/query/decide` 四个子命令，所有状态变更走它，避免手改 JSON

---

## PHASE 2 · 萃取流水线化（1 天）

- [x] **T2-1 升级 `MasterPrompt.md` 到 v2** ✅ 已完成
  - 交付：`paper2skills-vault/07-资源库/MasterPrompt-v2.md`（v1 原文保留供对照）
  - 含 R1–R5 证据规则、frontmatter v2（`paper_id/venue/venue_tier/evidence_grade/verified_by/related`）、
    「数据要求 + 企业内是否可得」必填行、**①b 反例与适用边界**、**⑥ 原文引用（≥3 条）**、
    R4 硬拦截清单（撤稿 / 纯理论 / survey / workshop 抬级 / PDF 提示注入 / 顶刊 stale_method）、
    以及出卡前自检清单

- [x] **T2-3 新建 `paper-萃取/scripts/verify_skill_code.py`（原计划 T2-4 提前）** ✅ 已完成
  - 五级验证 L1 语法 / L2 编译 / L3 导入 / L4 运行 / L5 断言
  - 判定 `PASS` / `ENV_BLOCKED` / **`ORPHAN_DEP`**（新增第四类）/ `FAIL`
  - **首次全量基线：80 张含代码卡片 → K1 执行率 52.5%**
    （PASS 42 / ENV_BLOCKED 6 / ORPHAN_DEP 9 / FAIL 23）
  - 验收（原计划「故意塞一个语法错误的卡片，确认被拦」）✅ 通过——实测拦下 8 个真语法错误

- [x] **T2-5 新建 `paper-审核/scripts/gate_check.py`（G1/G2/G3 三合一）** ✅ 已完成
  - G1 读 K1 产物（无凭证一律判红）；G2 度量数字溯源；G3 业务具体性与数据可得性
  - **首次全量基线（G2）：130 张卡 → 57 张通过（43.8%），红灯 438 条**
  - 验收（原计划「预期大量红色，这正是要暴露的问题」）✅ 完全符合预期

- [x] **T1-3 新建 `paper2skills-vault/07-资源库/venue-whitelist.md`** ✅ 已完成
  - 含 tier 定义、28 刊 ISSN 表、会议白名单、§3 降级判定表（含 6 处抬级反例）
  - **§7 纠正了两个原判断错误**：OpenReview「❌403」→「部分可用但覆盖易高估」；
    DBLP「⚠️UA限速」→「HTTP 200 + Anubis 挑战页，脚本路线放弃」
  - §7.3 Crossref 双日期过滤器规则；§7.4 `NAACL 2026` 不存在；§9 ISSN 陷阱

- [x] **T0-2 清理 26 组重复 Skill 卡片** ✅ 已完成
  - 25 组字节相同 → 删顶层副本；1 张仅顶层存在 → 迁入唯一存放地；
    1 张已漂移（产品研发部→产品中心）→ 保留新版，旧版移入 `_superseded/`
  - 验收：`find paper2skills-vault -name 'Skill-*.md' | uniq -d` 输出为空 ✅
  - **前置动作**：仓库此前**不是 git 仓库**，已先 `git init` 建立安全网（见下方"附 C"）

- [x] **T2-2 升级 `paper-萃取/SKILL.md`** ✅ 2026-09-12 复核：Step 5 已改为调用 `verify_skill_code.py`，
  并新增 **Step 5b 生成 evidence.md**（G2 溯源凭证，必做）与 **Step 5c 运行 K2 三合一门禁**；
  Step 6 明确「前置条件：Step 5 未通过不得执行」
- [ ] **T2-4 新建 `paper-萃取/scripts/extract_code_blocks.py`**（**未做，已降级为可选**）
  - 现在 K1 已能直接验证卡片内代码块，此脚本降级为「落盘到 `paper2skills-code/`」的可选步骤
  - ⚠️ 2026-09-12 盘点核实：`paper2skills-skills/paper-萃取/scripts/` 下只有 `verify_skill_code.py`
  - 影响：低 —— `CLAUDE.md` 已明确 registry 的 `outputs.code_dir` 是**规划值**，
    代码模板内嵌在卡片 ③ 段由 K1 验证，故本项不是欠账，而是一个未采纳的备选方案
- [x] **T2-6 `paper-同步/scripts/sync.py` 加门禁前置** ✅ 已完成（并删除了早期重复列出的同名条目）
  - 同步前现场跑 K2 三合一门禁（G1/G2/G3），任一红灯即**拒绝同步**（退出码 2）
  - 三档强度 `--gate enforce|warn|off`，默认 enforce
  - 验收 ✅：对存量卡 `Skill-ROAS-Budget-Optimization` 执行同步 → 被拒绝并逐条列出
    G2 红灯 4 条 + G3 红灯 1 条；4 张新卡 → G1/G2/G3 全绿放行
  - 设计要点：绕过必须**显式且留痕** —— `--force-gates "<理由>"` 才能放行，
    理由与时间写入 `sync_status.json` 的 `_gate_override`；不做「打印一条黄字就放过」那种门禁
  - 修了一个自己引入的 bug：`--gate warn` 最初与 enforce 行为完全相同（都 return 2），
    会让用户以为在「只看警告」，最后去用 `--gate off` 把门禁整个关掉 —— 比不加更糟

- [x] **T2-7 新建 `paper-维护` skill（仓库体检）** ✅ 已完成
  - `paper2skills-skills/paper-维护/`：`SKILL.md` + `scripts/repo_health.py`
  - 8 项检查 C1–C8：重复卡片 / frontmatter / 硬编码路径 / **代码围栏结构** / registry 一致性 /
    门禁汇总与时效 / 仓库卫生 / v2 段落完整性
  - 频率：每周；`--json-out` 供趋势追踪；`--selftest` 用构造样本证明检查真的会报警
  - 验收 ✅：`--selftest` 4 项全绿；对全库跑出真实欠账
    （C8：130/135 张缺 ⑥ 段、13 张仅有 ①②③；C3：7 处绝对路径；C2：57 张无 frontmatter）
  - **首轮就抓出 3 张真正未闭合的代码围栏并修好**（Multi-Armed-Bandit / Matrix-Factorization /
    Customer-Churn-Prediction），并修正了一处自身过度表述：围栏未闭合 **不会**让 K1 失败
    （K1 把 EOF 当隐式闭合），真实后果是渲染错乱 —— K1 全绿不代表围栏没问题
  - 开发过程中修了 **5 个自身 bug，全部是假红灯**：C3 白名单漏 vault 侧（误报 290+ 处）、
    C4 把缩进 1–3 空格的示例围栏当硬缺陷、C4 的 `critical` 把提示当缺陷、
    C6 读错 K1 时间戳位置、C7 只数文件个数而不区分是否被 git 跟踪

---

## PHASE 3 · 首批萃取（P0 队列，分批）

> 每张卡交付：Skill 卡片 + 代码 + evidence.md + 三份 gate JSON。数据可得性已在 registry 标注。

**批次 3A · 广告与增长（业务杠杆最高，4 张）** ✅ 全部完成
- [x] T3-1 `2606.26690` 归因蚕食校正 → `13-广告分析/Skill-Cannibalization-Corrected-Attribution.md` ✅ K1 PASS / 引文 18-18 VERBATIM / G2·G3 绿
- [x] T3-2 `2608.11675` FunnelCausalNet 多档券 uplift → `13-广告分析/Skill-Funnel-Causal-Coupon-Allocation.md` ✅ K1 PASS / 引文 28-28 VERBATIM / G2·G3 绿
- [x] T3-3 `2608.10182` 因果约束下的预算分配 → `13-广告分析/Skill-Causal-Budget-Allocation.md` ✅ K1 PASS / 引文 38-38 VERBATIM / G2·G3 绿（registry 的 +7.20% 已三处核实）
- [x] T3-4 `2608.18174` 季节性流失误报修正 → `06-增长模型/Skill-Seasonal-Aligned-Churn-Label.md` ✅ K1 PASS / 引文 40-40 VERBATIM / G2·G3 绿（复现包=arXiv ancillary files，未下载）

**批次 3B · 预测与库存（4 张）** ✅ 全部完成
- [x] T3-5 `2608.25871` CEDAR → `03-时间序列/Skill-Decision-Conditioned-Forecasting.md` ✅ K1 PASS / 引文 47-47 / G2·G3 绿（**修正了 registry 两处抬级：「备货计划」与 data_availability**）
- [x] T3-6 `2607.16230` RouteCost → `03-时间序列/Skill-Shipping-Cost-Estimation.md` ✅ K1 PASS / 引文 36-36 / G2·G3 绿
- [x] T3-7 `2607.09745` SupplyNetPy → `04-供应链/Skill-Supply-Network-Simulation.md` ✅ K1 PASS / 引文 38-38 / G2·G3 绿（开源包名与仓库 URL 已逐字核实）
- [x] T3-8 `2606.29366` ORLA → `04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md` ✅ K1 PASS / 引文 41-41 / G2·G3 绿（**registry 的「优先保 FBA 不断货」与原文不符，已修正并记录**）

**批次 3C · 用户与电商（3 张）** ✅ 全部完成
- [x] T3-9 `2607.09608` 增量测量（站外种草→站内成交）→ `14-用户分析/Skill-Incrementality-Measurement.md` ✅ K1 PASS / 引文 31-31 / G2·G3 绿（**论文无真实数据**：纯理论 + 模拟研究，`data_availability` 已由 conditional 改 synthetic）
- [x] T3-10 `2608.27006` 活跃目录增量购物助手 → `00-电商Agent/Skill-Live-Catalog-Conversational-Rec.md` ✅ K1 PASS / 引文 37-37 / G2·G3 绿（**Demo 短文，R4 冲突待裁决**；`venue_tier` 由 top 改 preprint）
- [x] T3-11 `2608.20844` TRACE 目录属性补全 → `00-电商Agent/Skill-Agentic-Catalog-Enrichment.md` ✅ K1 PASS / 引文 45-45 / G2·G3 绿（摘要报 +90.4% 而正文仅 "over 90%"，已在卡内钉死口径）

**批次 3D · Agent 工程与实验（5 张）** ✅ 全部完成
- [x] T3-12 `2608.26263` SKILL.state → `16-智能体工程/Skill-Stateful-Skill-Runtime.md` ✅ K1 PASS / 引文 50-50 / G2·G3 绿（摘要**一个数字都没有**，压缩比 16.2× 取自正文；另记「单步 prompt 反而更高」这条反直觉边界）
- [x] T3-13 `2608.22152` Collaboration Tax → `10-MAS/Skill-Multi-Agent-Collaboration-Tax.md` ✅ K1 PASS / 引文 55-55 / G2·G3 绿（**底本来自 PDF**：arXiv 无 LaTeXML HTML，v1/v2/v3 全 404）
- [x] T3-14 `2608.25277` Routed Graph Handoff → `10-MAS/Skill-Routed-Graph-Handoff.md` ✅ K1 PASS / 引文 51-51 / G2·G3 绿（3.2× 是 τ-retail 单点，加权平均 2.1×；graph-only 在 AppWorld 回退 −14.6pp）
- [x] T3-15 `2607.22115` RBAC Text-to-SQL 门禁 → `09-DataAgent-LLM/Skill-SQL-Agent-Access-Control.md` ✅ K1 PASS / 引文 39-39 / G2·G3 绿（**SIGMOD 录用无原文证据**，卡片按保守口径记 preprint）
- [x] T3-16 `2609.01038` 人格条件 A/B 仿真 → `02-A_B实验/Skill-Persona-Based-AB-Simulation.md` ✅ K1 PASS / 引文 41-41 / G2·G3 绿（0.75–0.90 分项来源不同：0.90 来自公开电商人格 × subscription）

**批次 3E · 增强而非新建（避免重复出卡）**
- [x] T3-17 `2608.28978` 负结果 → `Skill-GraphRAG-Knowledge-Enhanced-Retrieval`（引用块 0→18/18）与 `Skill-Agentic-Memory-Management`（0→17/17）各加"反例与适用边界"小节 ✅（**registry「劝退早期投图记忆」比原文宽，已按论文自设范围限定修正**：「this extraction-based pipeline rather than graph-structured memory in general」）
- [x] **T3-18** `2608.09162` 数值特征变换 → 增强 `Skill-Feature-Engineering`（引用块 0→**30/30**；摘要称「一致优于所有基线」，但论文自己的 Table 1 里 `mlp`/`mlp_plr` 两行 PLE 优于 stretch —— 反例已写进 1b）✅
- [x] **T3-19** `2608.10240` 顺序模态丢弃 → 增强 `Skill-Cold-Start-Meta-Learning-PAM`（引用块 0→**32/32**）✅

---

## PHASE 4A · 存量卡证据链补齐 ✅ 已完成（2026-09-12）

> 这不是原计划的 PHASE4 —— 见文件头「命名消歧」。原 PHASE4 已改称 PHASE6。
> 完整报告见 `PHASE4-发现的内容缺陷.md`，口径结论已写进 `CLAUDE.md`。

**做法**：用 `provenance_audit.py` 把 146 张卡按「能不能修」分四层，再分层处置：

| 层 | 张数 | 处置 |
|----|------|------|
| 已核验 | 83 | 无需动 |
| 可补引文（有全文底本） | 8 | F3 补逐字引文 |
| 可补引文（需先抓全文） | 7 | 先 `fetch_fulltext` 再补 |
| 无论文来源 | 48 | F4 补 `evidence_basis: author-practice` + 证据基础声明 |

**交付**：129 张卡、+5,721 行 / −103 行（102 行是 frontmatter 元数据规范化），
**没有改写任何既有正文与数字**。引文 729 → **2,087 条**，0 伪造 0 近似 0 拼接。

**顺带封堵 5 个门禁假绿灯（#11–#15）**，其中 #13（自引文洗白）与 #14（中文卡数字被
系统性漏检）**由子代理在干活时实测发现并复现**，不是设计时想到的。

**两条必须传给下一轮的结论**：
1. **残余 G2 红灯的 75%–85% 是「作者外推」**（⑤ 商业价值评估与 ② 应用案例里的
   工期/ROI/倍数），论文里本就不存在 → **G2 通过率有结构性上限**，
   下一步该做**口径设计**而不是继续补引文。
2. **修门禁会让红灯上升**（#14 使 1,308 → 1,476）→ **不能拿红灯总数当进度指标**。

---

## PHASE 6（原 PHASE 4）· 空白领域补卡（1 天）❌ 未开始

> 2026-09-12 盘点确认：以下四项**一项未执行**。原编号 PHASE4 已改称 PHASE6。
> 执行顺序见 `PHASE5-明日执行TODO.md` §3 顺延清单 S3。

- [ ] **T4-1 12-ML基础：表格基础模型三张卡**
  - `2609.04540` Mitra-v2 + `2608.01400` TabDPT-Turbo → `Skill-Tabular-Foundation-Model-Baseline.md`（**含 HF 权重与代码，当周可跑**）
  - `2608.18849` GEAR + `2608.10837` TACTICL → `Skill-TFM-Distillation-Deployment.md`
  - `2606.31474` TabPATE → `Skill-Tabular-ICL-Privacy-Boundary.md`（GDPR/CCPA 合规）
  - 验收：baseline 卡能在本地跑通并用本项目 `paper2skills-data/` 的数据出结果

- [ ] **T4-2 新建 `17-跨境合规` 领域（**非论文来源**）**
  - 背景：本轮 00-电商Agent 域 22 篇里，关税/CE-FDA-CPSC/IP/汇率/逆向物流 **0 篇** —— 学术真空，但这是母婴出海真正的决策变量
  - 卡片 1：`Skill-Platform-Policy-Compliance-Monitor.md` — 平台政策变更的结构化抽取与影响面分析
  - 卡片 2：`Skill-Certification-Requirement-Matrix.md` — 认证矩阵（CE/FDA/CPSC/REACH）按品类×目标市场
  - 卡片 3：`Skill-Cross-Modal-Spec-Consistency.md` — 借鉴 `2607.25959` Kontrast 做"详情页文字/规格表/后台属性表三处对不上"的自动查错（合规与退货风险）
  - 验收：卡片同样过 G1/G2/G3；数据来源与更新频率写清

- [ ] **T4-3 11-AI人文 改流程**
  - 停用 arXiv 萃取（本轮 21/21 命中都不是人文）；改为"提示词工程 + 人工策展"流水线
  - 把 4 篇有跨域价值的（Latent-LoRA / One-Adapter / Attention is Case-Sensitive / Bengali PEFT）改挂 12-ML基础 或 07

- [ ] **T4-4 06-增长模型 供给策略调整**
  - 事实：92 天仅 5 篇命中（其他域 59-84）
  - 动作：从 01/02/13/14/15 交叉复用（`2608.18174` 是唯一直接可用的），不指望 arXiv 增量

---

## PHASE 5 · 常态化（每周）

> ⚠️ **2026-09-12 盘点：PHASE5 的启动前置条件尚未满足。**
> - **T5-4 现在跑不通** —— `repo_health.py --json-out` 因 `data/health/` 目录不存在而崩溃（退出码 1），
>   趋势追踪拿不到历史文件。修法见 `PHASE5-明日执行TODO.md` §X3(a)。
> - **T5-3 的「G1/G2/G3 全绿」标准目前不可达** —— G2 通过率被「作者外推」结构性封顶在 ~16%，
>   需要先落**口径设计**（§X1），否则每周都会看到一张永远修不绿的表。
> **结论：先把 X1/X3 做完，再启动本阶段。**

- [ ] **T5-1 周一自动化（30 min，脚本）**：arXiv + Crossref 增量 → 评分 → 去重 → 更新 registry → 生成 `shortlist.md`
- [ ] **T5-2 周二人工（30 min）**：从短名单勾 P0/P1，写入 registry `decision` + 理由
- [ ] **T5-3 周三至周五（AI 主导）**：萃取 1-3 张卡 → G1/G2/G3 → 抽检 1 张的数字出处
- [ ] **T5-4 周日（10 min）**：`paper-维护` 体检 ⛔ **被 X3(a) 阻塞**
- [ ] **T5-5 每季度**：刷新缺口表 / venue 白名单 / 关键词库；统计卡片使用率（哪些卡被真实项目引用过），淘汰零使用卡片

> 📌 **PHASE5 的日常执行视图不在本文件**，在 `PHASE5-明日执行TODO.md`
> （该文件按「哪一天做什么」组织，每次收工后重写；本文件只在阶段边界变化时更新）。

---

## 附 A0 · registry 欠账（2026-09-12 盘点新增，**不属于任何阶段，是横切问题**）

| 项 | 数量 | 性质 |
|----|------|------|
| `decision: extract` 且**无任何产出** | **12 篇** | 论文在册、决策已下、卡没做。P0 三篇：`0007`（推荐系统）/ `0012`（推荐系统）/ `0015` Mitra-v2（12-ML基础，**唯一与 PHASE6 T4-1 重合的一篇**） |
| ~~15 篇待萃取~~ —— **勘误** | 3 篇 | `0023` / `0026` / `0030` **不是欠账**：它们已作为 PHASE3-3E 的**增强**交付（`outputs.enhanced_cards` 非空且路径存在）。⚠️ **统计「未出卡」时必须同时看 `skill_card` 与 `enhanced_cards`** —— 只数前者会虚报 3 篇 |
| `title` 仍是 `(待补：见 note)` | **8 条**（`p2s-2026-0038`～`0045`） | **纯机械回填**（标题实际写在 `decision_reason` 里），约 10 分钟 |
| 卡片 `paper_id` **不在 registry** | **70 张** | ⚠️ **不是错误** —— registry 只覆盖本轮检索的 45 条，存量 146 张卡的历史论文来自更早批次。**不要现在去补 70 条记录**（几天工作量，且需先设计 schema 兼容），先加 `coverage_note` 说明归属规则 |

> registry 45 条的完整口径：**已交付卡 16 + 已交付为增强 3 + 真·待萃取 12 + watch 14 = 45**。
> 处理顺序：8 条标题回填 → 12 篇 backlog 排期 → 70 张覆盖率**登记不修**。
> 详见 `PHASE5-明日执行TODO.md` §Z3 / §Z4 / §S4。

---

## 附 A · 本轮调研产出的可复用资产

| 资产 | 路径 | 用途 |
|------|------|------|
| arXiv 收割器 | `paper2skills-research/scripts/arxiv_harvest.py` | 周更路线 A |
| 期刊收割器 | `paper2skills-research/scripts/journal_harvest.py` | 周更路线 B |
| 评分器 | `paper2skills-research/scripts/rank_candidates.py` | S2 归一评分 |
| 卡片体检器 | `paper2skills-research/scripts/skill_audit.py` | S5 / `paper-维护` |
| 去重器 | `paper2skills-research/scripts/dedup_check.py` | S3 三层去重 |
| registry 生成器 | `paper2skills-research/scripts/build_registry.py` | 事实源初版 |
| bundle 生成器 | `make_bundles.py` / `make_journal_bundles.py` | LLM 精读批处理 |
| 候选数据 | `data/arxiv_candidates.json`（1046）/ `data/journal_candidates.json`（1751） | 复现与追溯 |
| 领域 bundle | `data/bundles/*.json`（21 个） | 后续增量对比基线 |

## 附 B · 关键风险与前置澄清

> ⚠️ **2026-09-12 盘点：1–3 号问题从 PHASE3 挂到今天仍未回答。**
> 已转写为可直接回答的问句，见 `PHASE5-明日执行TODO.md` §4 ——
> **留在表格里不会有人回答，必须问出去**。

| # | 风险/待澄清 | 影响的 TODO | 需谁决策 | 2026-09-12 状态 |
|---|-------------|-------------|----------|------------------|
| 1 | 投放主体是**自有独立站**还是**第三方平台店**？ | T3-9 的落地形态（前者可做受众级随机化，后者降级为诊断框架） | 业务方 | 🔴 **未回答**（已交付卡 `Skill-Incrementality-Measurement`，落地形态悬空） |
| 2 | 出海历史是否 ≥2 个完整年度（月度活跃面板）？ | T3-4 是否需要品类季节性先验替代 | 业务方 | 🔴 **未回答**（已交付卡 `Skill-Seasonal-Aligned-Churn-Label`） |
| 3 | 是否有 RCT / holdback 流量？ | T3-2、T3-3 的理论前提 | 业务方 | 🔴 **未回答**（已交付卡 `Skill-Funnel-Causal-Coupon-Allocation`、`Skill-Causal-Budget-Allocation`） |
| 4 | OpenReview / DBLP 取数受限 | 会议季路线只能走 Crossref + PMLR + ACL Anthology + 名单页；**已精确界定**（见 venue-whitelist §7） | 无需决策，接受限制 | ✅ 已接受 |
| 5 | 数字幻觉（本轮已实证：4 篇摘要含缺陷、150 篇源码被截断） | G2 门禁 + 抽检是**必需**而非可选 | 执行纪律 | ✅ 已落为 K1/K2 门禁 + `--selftest` |
| 6 | **`DDDD.pem` 私钥仍在本地磁盘**，且曾长期存放于当时无版本控制的仓库根目录 | 见附 C —— 若曾用于生产（云主机 SSH / 支付回调验签），应轮换 | **业务方（安全事项）** | 🔴 **未确认**（优先级最高） |

## 附 C · 安全事件与仓库纳管（2026-09-12）

**事件**：本仓库此前**不是 git 仓库**（无 `.git`），意味着任何删除都不可回退。
在建立安全网的过程中，发现仓库根目录存在 **`DDDD.pem`——一个未加密的 RSA 私钥**，
且**未被任何脚本引用**。

**处置**：
1. 先写 `.gitignore`（排除 PDF / Excel / 500MB+ jsonl / node_modules 等大体积产物），
   首次 commit 纳管文本资产；
2. 发现 `DDDD.pem` 被纳管后，**删除 `.git` 重建历史**（该 commit 从未推送，故无泄露面），
   并在 `.gitignore` 中加入 `*.pem` / `*.key` / `id_rsa*` / `credentials.json` 等硬性规则；
3. 复查 `git ls-files` 中已无密钥类文件。

**最终状态**：2753 个文本文件、`.git` 体积 38MB，含完整回退能力。

> ⚠️ **需要业务方确认**：`DDDD.pem` 仍在本地磁盘（`-rw-------`，仅属主可读）。
> 若它曾被用于生产环境（云主机 SSH、支付回调验签等），**建议轮换该密钥对**——
> 它长期以明文形式存在于一个当时无版本控制的目录中。

## 附 D · 本轮引入的两条实现铁律（写代码前必读）

| # | 铁律 | 违反后的实测后果 |
|---|------|------------------|
| 1 | **验证代码块必须按卡片拼接，不能逐块独立导入** | 首轮 K1 报 44 个 L3 导入失败，其中 **19 个是假阳性**——卡片是「block1 定义类 → block3 使用」的递进结构，逐块导入必然 `NameError`。修正后执行率从 46.2% 升到 52.5% |
| 2 | **验证必须在断网语义下运行** | 首轮全量门禁**跑 30 分钟未出结果**，而进程 CPU 时间仅 **2.2 秒**——时间全花在网络重试上。加入 socket 拦截 + `HF_HUB_OFFLINE=1` 后，同一全量跑完只需 **2 分 55 秒** |

> 教训的共性：**这两条都会让「验证工具本身」成为最大的假信号源**。
> 门禁工具的可信度需要先于被门禁对象建立。


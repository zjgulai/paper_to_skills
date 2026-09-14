# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**paper2skills** - A system that converts academic papers into actionable business decision skill cards, focused on cross-border e-commerce for mother & baby products (母婴出海跨境电商).

The workflow transforms academic research (primarily from ArXiv) into practical business skills through a 4-step pipeline: paper selection → extraction → review → sync.

> **2026-09 升级中**:检索层正在从「仅 arXiv」扩展为「arXiv + Crossref 顶刊 + 会议 proceedings」多源流水线,
> 并补上可复现评分、三层去重、双门禁(K1 代码可执行 / K2 事实可溯源)。
> 设计见 `paper2skills-research/reports/paper2skills-萃取链路升级方案-v2.md`,
> 实施清单见 `paper2skills-research/reports/实施方案与TODO.md`,
> **当前执行视图见 `paper2skills-research/reports/PHASE5-明日执行TODO.md`**(每次收工后重写这一份),
> 论文唯一事实源为 `paper2skills-vault/07-资源库/papers_registry.json`。

## Project Structure

```
├── paper2skills-skills/     # Claude Code skills for the workflow
│   ├── paper-workflow/      # Orchestrates the complete workflow
│   ├── paper-选题/           # Step 1: Paper selection from ArXiv/GitHub
│   ├── paper-萃取/           # Step 2: Extract papers into Skill cards
│   ├── paper-审核/           # Step 3: Quality review
│   └── paper-同步/           # Step 4: Sync to multiple platforms
├── paper2skills-research/   # 调研与流水线工程工作区(脚本 + 中间数据 + 报告)
│   ├── scripts/             # arxiv_harvest / journal_harvest / rank_candidates /
│   │                        # skill_audit / dedup_check / build_registry
│   ├── data/                # 候选池 JSON/CSV、领域 bundle、体检与去重报告
│   └── reports/             # 方案、推荐清单、TODO
├── paper2skills-vault/      # Knowledge base (Obsidian-compatible)
│   ├── 00-电商Agent/          # Live-catalog conversational rec, agentic catalog enrichment
│   ├── 01-因果推断/          # Causal inference skills
│   ├── 02-A_B实验/           # A/B testing skills
│   ├── 03-时间序列/          # Time series skills
│   ├── 04-供应链/            # Supply chain skills
│   ├── 05-推荐系统/          # Recommendation system skills
│   ├── 06-增长模型/          # Growth model skills
│   ├── 08-知识图谱/          # Knowledge graph / GNN skills
│   ├── 09-DataAgent-LLM/     # DataAgent & LLM-powered analytics
│   ├── 10-MAS/               # Multi-Agent System skills
│   ├── 11-AI人文/             # AI × Humanities: healing quotes, philosophical analogies
│   ├── 12-ML基础/             # Machine learning fundamentals
│   ├── 13-广告分析/            # Ad attribution, ROAS optimization
│   ├── 14-用户分析/            # Funnel, cohort, RFM analysis
│   ├── 15-营销投放分析/         # MMM, promotion effectiveness
│   ├── 16-智能体工程/           # LLM Agent Engineering: Skills, Context, MCP/A2A
│   ├── 07-资源库/            # Master Prompt, keywords, sync status
│   └── papers/               # Downloaded papers by domain
└── paper2skills-code/       # Python code templates
    ├── causal_inference/
    ├── ab_testing/
    ├── time_series/
    ├── supply_chain/
    ├── recommendation/
    ├── growth_model/
    ├── nlp_voc/              # 保留:VOC 子项目已迁至 ../ai_nlp_voc/,本目录为代码模板镜像
    ├── knowledge_graph/
    ├── data_agent_llm/
    ├── mas/
    └── llm_agent_engineering/

  说明:00-电商Agent / 11-AI人文 / 12-ML基础 / 13-广告分析 / 14-用户分析 / 15-营销投放分析 六个新业务领域
  目前仅有 vault Skill 卡片,code/ 侧尚未落地子模块,需要时按 Python 包命名规范(英文 snake_case)新建。
  ⚠️ registry 的 `outputs.code_dir` 字段(如 `paper2skills-code/ecommerce_agent/<algo>`)是**规划值**,
  3A/3B/3C 共 11 张卡均**未**落地该目录 —— 代码模板实际内嵌在卡片的「③ 代码模板」段,由 K1 门禁验证。
```

## NLP-VOC 子项目迁出说明

`07-NLP-VOC` 子项目已于 `2026-05-17 commit 47b1dbf` 独立迁出至 `/Users/pray/project/ai_nlp_voc/`。
本仓库**有意保留**两处残留作为"代码模板 + 论文档案"镜像,而非未清理的脏数据:

- `paper2skills-code/nlp_voc/` (43 子模块) — 历史代码模板,内部 `data_path` 已硬编码改为
  指向 `../ai_nlp_voc/...` 并加 `try/except`,可被新仓库直接 import 复用,不在本项目运行。
- `paper2skills-vault/papers/nlp_voc/` — 原始论文 PDF 档案,留作 paper-skills-graph 选题与
  citation 追踪使用,不再产生新 Skill 卡片。

回退手段:`git reset --hard backup/before-voc-extract-20260517`。

## Key Files

| File | Purpose |
|------|---------|
| `paper2skills-vault/07-资源库/MasterPrompt-v2.md` | **当前生效的 Master Prompt**(含 R1–R5 证据规则、frontmatter v2、硬拦截清单) |
| `paper2skills-vault/07-资源库/MasterPrompt.md` | v1 原文(保留供对照,新卡片一律走 v2) |
| `paper2skills-vault/07-资源库/关键词库.md` | ArXiv search keywords by domain (v1) |
| `paper2skills-vault/07-资源库/venue-whitelist.md` | **venue 白名单 + 降级判定 + 时效基准规则** |
| `paper2skills-vault/07-资源库/papers_registry.json` | **论文唯一事实源**(候选/评分/决策/门禁/交付物回指) |
| `paper2skills-vault/07-资源库/sync_status.json` | Tracks sync status across platforms |
| `paper2skills-vault/07-资源库/gates/gate_g2.json` | **K2 事实可溯源门禁产物**(G1 见 `paper2skills-research/data/verification/`) |
| `paper2skills-skills/paper-萃取/scripts/verify_skill_code.py` | **K1 门禁**:五级代码可执行性验证(语法/编译/导入/运行/断言) |
| `paper2skills-skills/paper-审核/scripts/gate_check.py` | **K2 门禁**:G1 代码 / G2 事实(含 G2a·G2b·G2c) / G3 业务 |
| `paper2skills-skills/paper-审核/scripts/quote_check.py` | **G2b 引文逐字核验**:把每条 `> 原文:"..."` 回查论文全文存档,判 VERBATIM/FUZZY/FABRICATED。`--selftest` **5 + 10 组用例**(2026-09-13 X4 起)自证能区分真引文/伪造/拼接。⚠️ 另含**版本层**:底本 v1 而卡片引正式版结论时判 `MISMATCH`(**黄灯,不改 `verdict`**)—— 实测 A6 即此形态 |
| `paper2skills-research/scripts/fetch_fulltext.py` | **全文抓取**:arXiv LaTeXML HTML → Markdown(保留章节号),存到 `papers/<域>/<paper_id>/fulltext.md`;萃取前必跑,否则引文无从核验。PHASE4 起支持 `--pdf`(本地 PDF→fulltext,含硬换行接回)与 `--from-worklist`(按审计结果批量补) |
| `paper2skills-research/scripts/provenance_audit.py` | **卡片论文溯源可达性体检**:把「G2 全红」拆成**已核验 / 可修 / 无论文来源**三层,回答「哪些卡该修、哪些卡修不了」。`--selftest` 锁定判定链端到端互斥(17 用例)。⚠️ 首版只扫 frontmatter 导致 47 张卡被误判,现扫正文。⚠️ **文档与代码不符(已知,未修)**:本节一度写作「剥尾部参考区」,代码实际是 `body = text[:12000]` 的**截断** —— 截断与剥离在长卡上不等价,修它会动计数,故先如实登记 |
| `paper2skills-research/scripts/registry_consistency.py` | **registry 门禁声明 vs 实物核对**。把**覆盖率**作为一等输出(核对到 N/M),低于 90% 拒绝给「一致」结论。⚠️ 它诞生的原因就是前身脚本静默跳过 3 条 `enhanced_cards` 记录却报「无不一致」 |
| `paper2skills-vault/07-资源库/关键词库-v2.md` | **三段式检索词**(正向 / 负向 / 约束词)。⚠️ 负向词**按域生效** —— 同一个词在不同域含义相反(`trial`/`cohort`/`ad`),无脑全局负向会误杀方法论论文 |
| `paper2skills-research/scripts/candidate_filter.py` | 三段式过滤的**可执行部件**(配置若只有文字一定会腐烂)。实测丢弃 25.8%,逐篇抽检无误杀;`--selftest` 含同词异义反例 |
| `paper2skills-vault/07-资源库/scoring_config.json` | 评分权重/阈值/关键词表外置。缺文件时脚本退回内置默认值(不静默用空值) |
| `paper2skills-vault/07-资源库/capability-graph.json` | **五层能力图谱（PHASE6 F2 起，v3）**：50 岗位 / 4 面 / 8 域 / 151 L3(+A/B/C) / 20 SCN / **64 格(FLOW×STG，逐格 M/R/D)** / **方案层（由 `07-资源库/solutions/*.md` 派生，S1 起）** / 146 张卡（**146/146 有 L3 归属**，F5 起）。由 `paper2skills-research/scripts/build_capability_graph.py` 从《AI组织变革》材料生成；`--check` 比对（已剔 `_meta.generated`）、`--check-taxonomy` 与产品侧 `dsh-paper2skills/data/taxonomy.json` 逐项断言相等（**证明没有另造第二份 L1–L3 事实源**）、`--cell FLOW-01/STG-04` 回答「有无方案/有无卡/本岗是否接线」。⚠️ **`cell_kind` 由 `model_participation` 机器导出且映射表无默认值**（风险 N4：D/R 格出现算法模型即判错）。⚠️ **方案层是派生不是手写**：把 `solutions/` 换空 ⇒ 64 格 `solution_refs` 全清空（变异 A）；方案域缺 `flow_id` ⇒ 拒绝出图（变异 B）。⚠️ 图上另有 3 条 join 陷阱（`plane_id` 只在 organization-graph；岗位↔场景 127 vs 641 两口径；AGT-012↔SCN-020 孤立绑定） |
| `paper2skills-vault/07-资源库/contracts/` | **契约层 L4（PHASE6 F6 起；S1 第一批 54 份已交付）**：139 份供给契约（A 73 标定 / B 66 完整性；**139 = 151 − 12**，C 类不建契约）。**拦两种错**：A 拦「参数移植」（把论文实验取值当业务取值 —— `dsh-paper2skills` run1 holdout Δ=−0.85 的机制），B 拦「完整性虚报」。frontmatter **全部由 `build_contracts.py` 生成，人不许手改**；`--flow FLOW-01 --batch-only` 按 FLOW 切片且**批次规模由「能力贡献岗位的 A/B 责任」算出**（实测 54，任务卡原写的 39 复算不出来 —— 39 = 25 + 14 两个不同口径的数相加）。`cards` 口径：已装线（产品侧 `classification.json` 1338 张，写 **slug** 供 S12 认）× 精选线（146 张，按 `id` 建联 93 张）。⚠️ **`可写/待卡` 不是放行依据**，是缺口账。**闸门已于 S12 落地（2026-09-13）**：`cards` 里实测混着**三个命名空间**（116 条 slug + 5 条精选线 id；已装线 id 可经 `id→slug` 归一化）⇒ 判定必须逐条落到「绑定 / 待装线 / 无法解析」三态，任何一条都不许静默丢弃。v1 底本已移到 `reports/PHASE6-F6F7-契约v1底本存档/`（留原地会与 v2 同 id 重复入账） |
| `check_material_residue.py` | **材料归属残留器（S1 收口期，#67 交付；#78 接线；W-67c/W-67d 收口）**：颗粒度比引文器宽（**blockquote 条目块 + 括号跨度**），专抓续行/枚举形态的漏网。**三族分开判，三族都真判**：**家族二**（`材料 §X` 而材料没有该编号 —— 材料只有 `1–12`/`R01–R05`，`§E.`/`§F.` 是**我方综述**的章节号）⇒ **L4k**；**家族一**（块内声称材料而词查无实据）⇒ **L4m**；**家族三 · 归属形态**（一个值必须住在 §0 它自己那一栏里）⇒ **L4n**（W-67d 新增）。补丁两份：`patch_material_attribution.py`（家族二，40 锚点）与 `patch_material_quarter_tier.py`（**W-67c/W-67d 共 83 处**：**37 搬家**（项从 (a) 搬到「业务侧默认」）+ **25 就地短标⇒搬家**（删的是**指针**）+ **1 只备案**（§0 一个字都没提）+ **10 附录挪位**（从 (c) 子句尾巴挪回默认栏）+ **2 B-056 归位**（删一个**本批插错的**幻影项 + 改述 57 字重复句）+ **7 正文插入** + **3 改述**）。读数：家族二 **38→0**、家族一 **66→41→4**（余 4 处逐处复核为假阳性 ⇒ `data/material-residue-baseline.json` 可见豁免）、家族三 **76→0**。`--selftest` 残留器 **21/21** · 补丁 **38/38**（含三族开关**双向**隔离 · ⑯ 两脚本 `normalize()` 逐字节相等 · baseline 双向 · **家族三四条判据 + 非空过守卫**）。⚠️ **#79/#80**：注释声明「同尺」实际不同尺、去归属语只查 `材料` 附近不查**词自己**附近 ⇒ 共 17 处假阳性。⚠️ **#82（纪律改写）**：「不删一字」被写成代理指标「只做插入」⇒ 逼出嵌套括号 / 粘字 / **标了归属却仍留着错话**；正确形式是「**所删必现于所增，改述必须逐处点名**」。⚠️ **#84**：豁免提示只 printf 则**绿门禁 stdout 根本不显示** ⇒ 过期豁免必须判红。⚠️ **#85–#90**：把「13 处」当全部（grep 只认一种动词，**实测 25**）· 判据拿 `（b）` 当子串找 ⇒ 4 份同行写的子句标记被**假红** · `（b）`/`（c）` 同行时附录挂错栏（**10 份**）· 豁免判据拿 `lcp_ge` 的**第一个**窗口比 ⇒ 恒报过期 · 新类漏了三态守卫 ⇒ 幂等报成过期 · 两个类动同一文件 ⇒ 前一类的「已应用」判据被后一类合法拆掉（改由 `RULES_DEDUP` **派生**第三态）。详见 **`reports/PHASE6-F8-S1W-材料归属改标与残留门禁.md`** |
| `check_contracts.py` + `build_contract_workpack.py` | **契约层门禁与作业包（F6/S1）**：判据 **J1–J13**，退出码 **0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 门禁内部错误（≠ 判红）**；`--selftest` **123/123**（27 变异样本 · 变异 21/21）。**正式位置只有 `contracts/A` 与 `contracts/B`**，别处的 `CTR-*.md` 报「样本/存档（不计入 139）」；计数超界与「同一责任两份正式契约」都进退出码；`cov > 1.0` 判红。⚠️ **「契约进提示词能提升产出」已被两轮证伪**（F6 中位 3 vs 对照 19.5；F7 削到只剩取值仍只有单卡臂的 47%）⇒ 契约层是**人的标定物 + 消费口闸门条件（S12）**，不是载荷。⚠️ **D27/D28**：D27 是**潜伏的假绿**（邻句豁免只认**词**不认**话题**，63/66 份 B 契约的样板句即可救活真缺口句），D28 是修它时**自己造出来的假红**（按列判 → 按行判）；完整教训见 **`reports/PHASE6-工具详解与缺陷史.md`** |
| `check_material_citations.py` | **材料引文核验器（S1 收口期）**：契约里凡声称「材料『X』」的 X 必须真的在材料里。**三种判定必须分开**（`attrib` 核验 / `non_attrib` 材料只是普通名词 / `unclassified` 不猜 ⇒ **exit 3**）、**两种罪必须分开**（`missing` 凭空 vs `altered` 声称原文却改字，后者**回捞材料原句**打印）。纪律：**活体探针每次跑**（不通 ⇒ exit 3）· **材料读不到 ⇒ exit 2 不是 0** · `non_attrib` 的份数与名单是一等输出。`--selftest` **15/15**；已是验收面里的独立门禁 **L4e**（端到端 + 反向控制 3/3）。⚠️ **#76：去归属标记自己含 `材料` 二字 ⇒ 被它读成「材料原文」**（`（不是材料原文）：「X」` 的 `run` 成了 `原文**）`，命中 `RUN_ATTRIB`）⇒ 1 条假红 + 3 种形态 `unclassified` ⇒ exit 3；已加 `mask_deattrib()`（否定/缺失形态**等长**遮罩），**反向控制**：`（材料原文）：「X」` 且 X 不在材料里**必须照旧判红**。⚠️ **数字随语料变过**：初测 **108 处 / 67 份** → 同质化整改后 **30 处 / 24 份** —— **是契约变了不是判据变了**，两套数并列。⚠️ 污染源是**四类**（其中「二手来源被标成材料原文」那类来自我方综述的 `§F.3`，**材料里没有这个章节号**）；**模板本身是干净的，种子只在冻结的 `contracts/v2/` 样本里** —— **冻结不等于无害**。完整清单与它自己先后被修掉的 4 个缺陷见 **`reports/PHASE6-工具详解与缺陷史.md`** 与 **`reports/PHASE6-F8-S1-材料引文核验.md`** |
| `paper2skills-research/scripts/run_phase6_gates.py` | **PHASE6 验收面**：一条命令跑完全部门禁（现 **52 条**），也是本阶段第一次有「整体状态」这个读数 —— 此前 11 个门禁脚本分散在三种命令形态里，每个数字只活在产它的那份报告里。⚠️ **2026-09-13 两次覆盖审计**（台账 #68–#75）：两次都靠**机械对差**「有 `--check`/`--selftest` 的脚本」与门槛清单，两次都差出**已交付却未接线**的门禁 —— 第一次 4 条（S3/S5/S9/S10 → L9–L12，其中 `route_backlog --check` **接上时就是红的**）、第二次又是 **`scan_secrets.py`（CLAUDE.md 自己写着「推送前必跑」的那一道）也从未接线而当时正红**。**与台账 #23 同型**；**第三次（#71）是 `scan_secrets.py`，第四次（#78）是 `check_material_residue.py`** —— 后者的根因不同：**它不是忘了接**，而是 #67 的「家族一未定标 ⇒ 不得进门禁」这条**正确**判断被加在了**整台仪器**上 ⇒ 另一族**精确的**判据（章节号残留 38 处）跟着一起隐形两天。⇒ **豁免的颗粒度就是判据的颗粒度**；现按 `--family` 分族判定，未判的那一族**明写「只扫不判」**。⇒ **「交付」不是「接线」，这是每次交付都要问的一句**。三条硬纪律：① **退出码不许合并成一个分数**，逐门禁给码、汇总只给各类几条，危险性排序 **3 > 2 > 1 > 0**（「没测到」比「测了是红的」更危险：红会有人去修，没测到没人知道）；② **一个门禁都没跑到 = exit 3，不是 0**；③ **豁免必须可见、带到期条件**，且**转绿时提示「请取消豁免」**（永久豁免＝台账 #5 那种腐烂）。`--only` / `--fast` / `--list` / `--json-out`；`--selftest` 11/11（含用假门禁验证退出码**传播**）；`--mutate` 把被守脚本改坏，验证端到端用例有劲。⚠️ **它自己被 `--mutate` 抓出两个缺陷**：变异体因硬编码路径**从没被跑过**、变异锚点文本在文件里出现两次导致**变异没施上力** —— 这是「**先证明变异改变了真实取值，再谈判据有没有劲**」这条纪律的来源，完整记录见 **`reports/PHASE6-工具详解与缺陷史.md`** |
| `dsh-paper2skills/lib/contract-gate.js` + `scripts/check-contract-gate.mjs` | **S12 消费口闸门（2026-09-13）**：判据只有一处实现，**独立核对器与白名单生成器共用它**（风险 N2）。**闸门落在哪**：1338 张 p2s 卡全部 `disable-model-invocation: true`，只经 50 个岗位 preset 的 `skill-subset` 白名单露出 ⇒ 那里就是模型目录的唯一入口，S12 是**给一道已存在的门补判据**，不是新增门。**账（现算）**：已装线 1338 = **已挂契约 29 + 待挂契约 114 + 未接线 1195**；白名单里 p2s 去重 **143**（按行 147），平台原生技能 115 不归本闸门管。三态：`count`（默认，过渡期只计数，归位态写进 manifest 的 `x_lute.skills.contract_gate`，页面显示「待挂契约」）/ `enforce`（硬拦：147 → **32** 条，跨 47 个岗位）/ `off`（显式退出）。⚠️ **对照测量六条判据 A–F 全绿**，含**因果翻面**（摘掉引用某卡的契约 ⇒ 该卡确实退出模型目录），仪器 `scripts/role-presets/measure-contract-gate.mjs`。⚠️ **读 preset 必须先试 `x_lute.skills.subset` 再试顶层 `skills.subset`，两个都拿不到要 exit 2** —— 第一版只读顶层，50 个岗位全读空却照样打出「已挂契约 0 / 未接线 1338」的**假绿账**（台账 #23） |
| `scripts/role-presets/unmanaged-rows.mjs` | **「生成器不拥有的行，不许删」（台账 #24）**：`generate.mjs` 整文件重写 `agent.cordis.yml`，实测把 `agt-033` 里按 ADR-0061 手插的 `product-kol-hunter` 行**静默删掉**（页面表现为「产品卡点了打不开」）。现改为：非生成器产出的行块**逐块带过、插回原位**（前驱行 id 决定位置，Cordis 的行是顺序敏感的挂载次序）、位置退化时**点名报出**。8 用例 / 变异 3/3 |
| `paper2skills-research/scripts/build_gap_ledger.py` | **缺口账与靶区工单（PHASE6 F3 起）**：把 151 责任名 × A/B/C × 供给状态**算出来**（不抄文档）并与方案 §4 逐格断言相等（J1–J5）；产物 `data/gap-ledger.json` + `reports/PHASE6-F3-缺口账与靶区工单.md`（31 条工单按位次排序）。排序权重**只用于排序**（J7 扫 8 个门禁脚本 + 3 种注入样本 + **2 种反向散文探针**自证，风险 N1）。⚠️ 首跑抓到材料矛盾：`产能调查` §F.5 记 A、§F.3/§F.4 判 B ⇒ 登记不重判。⚠️ **F5 后本节数字整体变过一次**（有精选卡 53→67、靶区一 36→31）：上游补了 53 张卡的分类，不是判据变了 —— 两套数并列在方案 §4/§4.2 |
| `paper2skills-vault/07-资源库/card-classification.json` | **精选线 146 张卡的分类落点（PHASE6 F5 起，五层轴的卡端事实源）**：146/146 有 L3，含 21 条技术域错位登记。生成器 `build_card_classification.py`（J1–J8，`--check` 0/1/2）。⚠️ **零漂移门禁**：93 张逐字继承产品侧 `classification.json`，要改必须显式 `reclassify` + 理由，静默分叉报红。详见 `reports/PHASE6-F4F5-五层分类轴与53张卡分类.md` |
| `paper2skills-research/scripts/build_card_classification.py` | **卡端分类生成器 + 门禁（PHASE6 F5 起）**：J1–J8，`--check` 0/1/**2（输入没拿到 ≠ 通过）**，selftest 12 用例。⚠️ 内含一个实测撞出的**仪器缺陷**：首版把样板标题 `## ② 母婴出海应用案例` 当成卡的出海要素（146 张里 139 张误判）⇒ 现先剥标题行，并配「样板标题不得产生命中」的仪器自证 |
| `dsh-paper2skills/lib/axis.js` + `scripts/check-axis.mjs` | **五层分类轴 + facet 层（PHASE6 F4 起，产品侧）**：复用 `lib/taxonomy.js` 判据（不重造分类学），加五层链与三个 facet（`tech_domain`/`venue_tier`/`quality_tier`）。`test/axis.spec.mjs` 17 用例 / 9 变异。⚠️ 技术域取 **vault 顶层目录**而非父目录名（否则 `00-知识库-Skill卡片` 会把 07-NLP-VOC 与 10-MAS 混成假域）；技术域是 facet，**错位只登记不移动** |
| `paper2skills-research/scripts/check_contract_dedup.py` | **跨契约同质化机检（S1 收口期）**：判据 D1（A §2 / B §5 五维表**缺值处置列**跨契约**同维**重复）/ D2（B §2 `→ 替代路径：` 行重复）/ D3（正文长行重复，**默认关闭**）/ D4（覆盖率，含「某判据开着却 0 条可比条目」⇒ **exit 2**）/ D5（**每次运行**的活体探针：正探针抓到、反探针不误报、专名遮罩必须活着 ⇒ 死了 **exit 3**）。算法是 n 字滑窗纯字符串比较，**无模糊/语义相似度**；另有一份**不复用检测器任何函数**的暴力实现独立复核 D1。**整改前后（同一冻结快照）**：D1 **2113 对 / 涉事 291 格 / 125 份 → 0**、D2 **538 对 / 55 份 → 0**（整改 **135 份 / 483 格**）。⚠️ 整改中由撰写人撞出的**假绿通路**：把 `替换` 写成 `**替换**` 或换全角标点即可切断 18 字窗口（补 NFKC + 去 `*` 后 **D1 当场 1554 → 2074**）。⚠️ **n=18 不是数据选出来的阈值**（曲线 12→6354 … 18→2113 … 40→75，**平滑无天然拐点**），唯一硬依据是它与规范/现场实测同尺 —— **如实登记，不许写成「由数据确定」**；D3 那条全语料级欠账（剔除模板原句后仍 6934 条）已单开工单。详见 **`reports/PHASE6-工具详解与缺陷史.md`** |
| `paper2skills-skills/paper-维护/scripts/repo_health.py` | **仓库体检**(C1–C8 + **C10**:重复卡片/frontmatter/路径/围栏结构/registry/门禁时效/卫生/段落完整性/**验证声明时间锚点**)。`--selftest` 用构造样本证明检查真的会报警。⚠️ 编号跳过 C9:该号预留给「悬空 `Skill-*.md` 引用」(见 §PHASE5 遗留),**尚未实现** |
| `paper2skills-research/scripts/check_instruction_budget.py` | **工作区 instruction 预算门禁（2026-09-13 立，验收面 L1a/L1b/L1c）**：守的是**其它门禁都看不见的一类丢失** —— 磁盘上 `CLAUDE.md` 是完整的，而 agent 读到的被 harness **静默截断**、`AGENTS.md` 与 `~/.dsh/AGENTS.md` **整份没进上下文**（实测 `truncated CLAUDE.md from 66175 to 65244 bytes`；同族已踩第二次，上一次砍掉的恰好是 `Skill Card Format` 那节）。**复刻 harness 两条渲染路径**（首次加载 / 变更对账，包装实测 **430 vs 292 B**），截断点**逐字节复算**出 harness 报的 `65244` —— 不是抄的常数。三态 **FULL(0) / OMITTED(0，一等输出) / TRUNCATED(1)**；预算取不到 ⇒ **exit 2**。`--selftest` 15/15（真实事件回归 + 边界正反证 + 端到端反向控制）· **变异 5/5**（含「锚点被变异表自己抄一遍」这个 S11 踩过的坑）。⚠️ 现状 **三份指令文件全部装得下** —— **余量 ＝ 65,536 − 渲染后字节**，**每次量都要现跑**（`--json` 给 `total`/`render_bytes`）：2026-09-13 W-67d 后实测 `total` 63,637 / `render_bytes` 64,012 ⇒ 余量约 **1.5 KB**。⚠️ 这个数**会自己变**（连写读数这个动作都改它）⇒ 只给**算法**，数字以现跑为准）。它已经**两次**逼着搬家：把 `Key Files` 四条最长的行搬到 `reports/PHASE6-工具详解与缺陷史.md`（余量 3.3 KB），以及把 **G2 的 18 条漏洞表整段**搬到 `reports/G2漏洞表归档-18条.md`（+2.7 KB 的新增把它挤爆、**`AGENTS.md` 与 `~/.dsh/AGENTS.md` 当时被整份丢弃**）—— **每次按「逐字节复制 + 留指针」办** |
| `paper2skills-research/scripts/check_card_identity.py` + `data/card-identity-baseline.json` | **「同名同物」判定器（S5 换底的前提；验收面 L8a/L8b/L8c）**：p2s 卡（1390）与精选卡（146）是不是同一张，判成**六态**（SAME_KEY / RENAMED_SAME / P2S_ONLY_PREVIEW / VAULT_ONLY / **SAME_NAME_DIFFERENT_THING** / **UNDECIDABLE**）且恒等式进判据（防静默丢卡）。⚠️ 它交付时就是 **exit 1（8 条 I3）而没人接进验收面** ⇒ 验收面报「全绿」、判据在报红。⚠️ 那 8 条的处置是**可见豁免、不是修**：主控抽样四条复核，**四条全部**在 legacy 线（`playbook/domains/*.html`）里有同名条目 ⇒ 「vault 里不存在」是**判据只看得见一个语料库**；改 `p2s_card_id` 去指向近名卡＝把两张不同的卡说成同一张，正是 I3 要防的事。baseline 带 `expires_when` 且转绿时提示删除。selftest 31/31 · **变异 5/5（每条打印「已生效于探针」）** |
| `paper2skills-skills/paper-维护/scripts/scan_secrets.py` | **推送前凭证扫描门禁（现 L15a/L15b）**。13 条规则 / 10 类凭证;**先把 `\"` 与 `&quot;` 归一化成 `"` 再匹配**,故一条规则即覆盖全部转义形态。扫到 **0 个文件时判失败(退出码 2)** —— 「没东西可查」不等于「查过了没问题」。⚠️ **2026-09-13：它从来没进过验收面，而当时它正红着**（3 条，由 `check_key_exposure.py` 的固件字面量造成）—— 与台账 **#23** 同型、**两天内第二次**，见 **#71**。⚠️ 新增**墓碑归一化**：filter-repo 的清理标记 `***REMOVED-…***` 不是凭证，是「这里曾有过凭证」的**物证**（不处理会产生「清理得越彻底、门禁越红」的荒谬读数）；`tombstones()` 单独把它当读数报出来，`--selftest` **用例 21** 的主体是反向控制（**墓碑旁边的真 key 必须照样命中**，见 #73） |
| `check_key_exposure.py` + `data/key-exposure-whitelist.json` | **凭证暴露面门禁（L14a/L14b；补上一行的盲区）**：上一行**只扫已入库文件**，查不到「**未跟踪但也未忽略**」的密钥（一次 `git add -A` 即入库）—— S8 实测的真暴露面正是这一态。五态含**公开证书须白名单写明理由**。⚠️ `--check` 进验收面；**`--sweep-roots` 广扫刻意不进**（随环境漂移，是读数不是判决）。实测**零硬编码复现**了 S8 的两条结论。⚠️ 它的自测固件原先把 PEM 头写成**完整字面量** ⇒ 把 `scan_secrets.py` 弄红（**教训写在隔壁文件里＝没写**，#72）|
| `check_history_secrets.py` + `data/exposed-credential-registry.json` + `data/history-secrets-whitelist.json` | **凭证扫描第三缝：git 对象库（L16a/L16b/L16c）**。前两行都只看**此刻**（已入库内容 / 工作区路径），于是「**曾经 commit 进去、后来又删掉**」的私钥对两者都是绿的，而 `git cat-file` 一行就能取回 —— 构造样本实测两门禁**同时报绿**。三条判据**分开报**：**A 路径史**（任何 ref 历史上 `--diff-filter=A` 加过密钥形态文件名）/ **B 对象内容**（对象库里任何 blob 的正文）/ **C 已知暴露值还原**（相邻字面量拼起来能否还原出登记在册的值 —— 拆开的凭证**按定义**骗得过模式扫描器，C 是唯一能看见它的判据；登记表**只存 len+sha256**）。REACHABLE（会随 push 出去）与 **DANGLING**（只在本机对象库，处置是 `git gc`）**分开**。⭐ **规则：仍然有效（live）的凭证不许豁免** —— 白名单写了理由也不生效、门禁照红（豁免的语义是「无可行动作」，活凭证有可行动作）。⚠️ **L16a 现在预期判红**：实测抓到一个**仍然有效**的飞书 webhook（只读 GET `code=19002`，对照不存在的 UUID 为 `19001`）在公开仓库历史里**可一行还原** —— 修本地文件追不回，只能在飞书后台轮换，红在这里对应 S8 唯一的**活**未闭环项。自检 **29 断言 / 变异 5/5**（含「C 抓得住而 B 抓不到」与「live 不许豁免」两条反向控制）。完整记录见 **`reports/PHASE6-F8-S8-凭证扫描第三缝.md`**（含 #74/#75） |
| `paper2skills-skills/paper-同步/scripts/sync.py` | Sync script;**同步前现场跑 K2 门禁,红灯即拒绝**(退出码 2);绕过须 `--force-gates "<理由>"` 并留痕 |
| `paper2skills-research/scripts/` | 检索/评分/去重/体检脚本(见下方"检索路线") |

## 版本控制与远端同步(2026-09-13 建立)

**本地仓库**:`/Users/lute/project/paper_to_skills`,分支 `main`,已设上游 `origin/main`。
**远端**:https://github.com/zjgulai/paper_to_skills (**PUBLIC**)

### ⚠️ 关键事实:本地与远端是**两套不同的库**,不是新旧关系

2026-09-13 首次接入远端时做了完整盘点,结论如下 —— **不要再假设「本地是最新的、远端是旧的」**:

| 维度 | 本地(精选线) | 远端 main(接入前) |
|------|-------------|-------------------|
| git 历史 | 28 commit,全部 2026-09-12 起 | 最后推送 2026-07-15 |
| **共同祖先** | **无**(unrelated histories) | 无 |
| Skill 卡片 | **146 张 / 16 个域** | **1,229 张 / 26 个域** |
| 同名卡片交集 | 92 张 | 92 张 |
| 各自独有 | 54 张 | **1,137 张** |
| frontmatter 风格 | MasterPrompt v2 + K1/K2 门禁口径 | `doc_type: knowledge` + `roadmap_phase` |
| tracked 体积 | 105 MB | 2.35 GB |

远端另有 **9 个本地根本不存在的业务域**:`17-价格优化` / `18-物流履约` / `19-风控反欺诈` /
`20-AI视频生成` / `21-合规决策` / `22-数据采集工程` / `23-运营财务` / `24-标签工程` / `25-搜索流量工程`。

> 本地 `playbook/` 目录里存着那套语料的**渲染快照**(`build-report.json`:`skill_pages: 1338, domains: 25`),
> 即内容并未完全丢失,但**可编辑的 `.md` 源头不在本地**。

### 处置(2026-09-13,经所有者决策选 A 案)

按「以本地当前形态为主」执行,但**远端原状先钉住、一条命令可恢复**:

| ref | 指向 | 含义 |
|-----|------|------|
| `refs/heads/main` | `d9186b2` | 本地精选线,已覆盖 |
| `refs/heads/legacy/main-20260715` | `dc4912a` | **接入前的远端 main 原状,完整保留 1,229 张卡** |
| `refs/tags/archive-pre-local-20260715` | `dc4912a` | 同一提交的 tag 锚点 |
| `refs/heads/gh-pages` | `0730b89` | 线上站点,**未动** |
| `refs/heads/feat/voc-deep-analysis-mvp` | `31927ed` | **未动** |

恢复旧语料:`git fetch origin legacy/main-20260715` 即可取回全部 1,137 张卡。

推送用的是 `--force-with-lease=main:<sha>`(先 `gh api` 确认远端 SHA 再断言),
**不用裸 `--force`** —— 裸 force 会在远端被他人更新时静默覆盖。

### 凭证事件(2026-09-13,首次推送前发现)

推送前全库扫描命中两类硬编码凭证。**两者都不是 K1 / K2 / repo_health 任何一道门禁能发现的** ——
那三道查重复卡/frontmatter/路径/围栏/registry/时效/卫生/段落完整性,**没有一项查凭证**。
仓库带着它们一路 commit 了 28 次,每次门禁都是绿的。

| 凭证 | 位置 | 判定 | 处置 |
|------|------|------|------|
| `sk-aae1…37bd`(DeepSeek key) | `playbook/` 下 6 个文件,3 种转义形态各一份 | 实测 `GET api.deepseek.com/models` → **HTTP 401,已失效**;远端**没有**它,推上去是**新**泄露 | 改为读 `DEEPSEEK_API_KEY` 环境变量 |
| 飞书机器人 webhook `a32b3ab7…47e9` | `playbook/agents.html` | **已在公开仓库 main 上**裸奔(自 2026-07-15 前即如此)——**既成事实** | 改为读 `window.__PLAYBOOK_CONFIG__.feishuHook`;**必须在飞书后台轮换**(改本地文件追不回已暴露的那份) |

后续处理:`.gitignore` 补规则 → `git filter-repo --replace-text` 重写全部 28 个 commit
(旧提交里现在是 `***REMOVED-DEEPSEEK-KEY***`) → 新建 `scan_secrets.py` 把这次检查固化成门禁。
`.git/` 改写前已备份到 `/tmp/pts-git-backup-20260913-115715`。

> **`DDDD.pem` 三点更正**（完整清单/鉴定/暴露面见 **`reports/PHASE6-F8-S8-凭证清点与暴露面.md`**）：
> ① 首行 `BEGIN RSA PRIVATE KEY` ⇒ 是 **RSA 私钥**，不是旧记载以为的「SSH 公钥」；
> ② 副本是 **4 份不是 3 份**（漏记的正是**本仓库根目录**那份）；同批另有**一把不同的** `ai_video.pem` **10 份**；
> ③ **两把都从未入库**（`scan_secrets.py` 因此**一把都查不到** —— 它只扫已入库文件，是适用范围不是失灵）。
> ⚠️ 轮换顺序**不可颠倒**：旧密钥追加 → 验证能登录 → **才**移除旧公钥。

### 推送前必跑

```bash
S=paper2skills-skills/paper-维护/scripts
python3 $S/scan_secrets.py --json-out paper2skills-research/data/health/secrets.json
python3 $S/check_history_secrets.py --check   # 对象库（含历史里删过的）
python3 paper2skills-research/scripts/check_key_exposure.py --check  # 工作区未忽略的
```

退出码:0 = 干净;1 = 命中凭证;2 = **一个文件都没扫到**(这不是干净,是没测)。
**三道缺一不可** —— 它们看的是三个不同的切片（已入库内容 / 对象库含历史 / 工作区路径）。
该仓库 `secret_scanning_push_protection` 已开启,GitHub 会直接拒收含凭证的 push。

> ⚠️ **本机推 GitHub 必须走代理**（2026-09-13 实测定位）：`127.0.0.1:7890` 有代理在听、macOS 系统代理也指着它，但 **`curl`/`git` 都不读系统代理**且 `http_proxy` 为空 ⇒ 症状是「DNS 正常、baidu 200、GitHub 全系 000 超时」，看着像网络故障，**其实是没人用那个代理**。修法 `git -c http.proxy=http://127.0.0.1:7890 push origin main`（`gh` 用 `HTTPS_PROXY`）。⚠️ 别写死进 `~/.gitconfig` —— CodeUp 走直连就通（302），全局代理会把它也绕出去。

## 门禁体系(K1 / K2,2026-09-12 建立)

**背景**:存量审核靠 LLM 打 7/10 分放行。而 `AutoReproduce`(arXiv:2505.20662) 实测:
LLM 评审认为"很好"的生成代码,**执行率仅 17.94%**;加上执行闭环后才到 94.87%。
`ResearchCodeBench` 附录 G 进一步显示新代码失败中 **58.6% 是语义错误**(能跑但算错)。
→ 故门禁必须产出**可复核的退出码与 stdout**,而不是分数。

### K1 代码可执行 · `paper-萃取/scripts/verify_skill_code.py`

| 级别 | 检查 | 说明 |
|------|------|------|
| L1 | `ast.parse` | 语法 |
| L2 | `py_compile` | 编译 |
| L3 | import 探针 | **缺第三方依赖注入 stub,归因环境;仓库内确实不存在的本地模块判 ORPHAN_DEP(卡片缺陷);只存在于已迁出镜像 `nlp_voc/` 的判 MIGRATED_DEP(非缺陷)** |
| L4 | 作为脚本执行 | 超时保护 + 独立进程组 SIGKILL |
| L5 | `pytest` | 断言是否真的成立 |

判定:`PASS` / `ENV_BLOCKED`(缺依赖,计入未验证分母) / `ORPHAN_DEP`(模块在仓库内确实不存在 → 卡片缺陷) /
`MIGRATED_DEP`(模块只在 `paper2skills-code/nlp_voc/` 镜像里 → **非卡片缺陷**,计入未验证分母但不计入失败) / `FAIL`。

> 判 `ORPHAN_DEP` 之前**必须 `ls` 一次确认模块真的不存在**。2026-09-12 实测:
> 9 张卡曾被判 ORPHAN 并写进「必须修」清单,`ls` 后全部推翻 —— 它们引用的模块都在 nlp_voc 镜像里。
> `verify_skill_code.py --selftest` 用四个用例锁定 `PASS/ORPHAN/MIGRATED` 三类互斥可区分。

```bash
# 全量(卡片级:自动把卡片内所有 python 块按文档顺序拼成一个模块)
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --all \
  --level 5 --timeout 30 \
  --json-out paper2skills-research/data/verification/k1_l5.json

# 单卡 / 快速语法扫描(秒级)
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <card.md>
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --all --level 2
```

**两条必须遵守的实现约束**(都是实测踩出来的):
1. **必须按卡片拼接验证,不能逐块独立导入** — 卡片普遍是「block1 定义类 → block3 使用」的递进结构,
   逐块导入会产生大量 `NameError` **假阳性**(首次实测 44 个 L3 失败里 19 个是假的)。
   需要逐块定位时用 `--per-block`。
2. **验证必须在断网语义下运行** — 卡片里的 `from_pretrained(...)`/`requests.get(...)` 会挂在网络重试上
   (实测:跑 30 分钟未出结果,而进程 CPU 时间仅 2.2 秒)。脚本已内置 socket 拦截 +
   `HF_HUB_OFFLINE=1` 等环境变量,把网络调用变成即时失败并归因为 `ENV_BLOCKED`。

**首次全量基线(2026-09-12)**:84 个代码单元 → **K1 执行率 54.8%**
(PASS 46 / ENV_BLOCKED 7 / ORPHAN_DEP 9 / FAIL 22),**语法级失败已归零**。
对照 PaperCoder 17.94%、AutoReproduce 94.87%。

> 迭代轨迹:46.2% → 52.5%(修 ORPHAN 假阳性)→ 54.8%(修围栏结构)。
> 前两次提升都来自**修正门禁自身的缺陷**,而非改写卡片——这说明
> **门禁工具的可信度必须先于被门禁对象建立**。

### K2 三合一门禁 · `paper-审核/scripts/gate_check.py`

| 门禁 | 检查 | 关键口径 |
|------|------|----------|
| **G1 代码可执行** | 读 K1 产物 | 无 K1 凭证一律判红(禁止凭人工判断放行) |
| **G2 事实可溯源** | 见下方 G2a/G2b/G2c 三层 | 2026-09-12 由「有没有出处」升级为「出处对不对」 |
| **G3 业务可落地** | 场景是否具体、是否声明数据可得性、ROI 是否有依据、是否关联 ≥2 张卡 | 空泛表述黑名单 + 母婴出海具体信号计数。⚠️ **2026-09-13：先剥 frontmatter 再扫**（台账 #70）—— 元数据不得当业务场景信号：实测 5 张卡各虚增 1 个信号、1 条红线被抹（`l3_all: 售后处理 / 客诉分诊` 的「客诉」替卡自己作证）。**`related:` 计入关联声明，`venue_source` 不计入**（后者是溯源不是关联），两个来源分开报 |

**G2 的三层(2026-09-12 加固,每层都由实测缺陷驱动)**

| 层 | 回答的问题 | 实现 | 判罚 |
|----|-----------|------|------|
| **G2a 有出处** | 带度量语义的数字能否在证据链里找到 | 原逻辑 | 红灯 |
| **G2b 出处为真** | 引用块是否**逐字**存在于论文全文存档 | `quote_check.py` | 伪造 → 红灯 |
| **G2c 出处实质** | 数字是否只在引文的**结构性语境**(表号/图号/样本量)中出现 | 最长连续匹配 + 结构前缀识别 | 黄灯待人工确认 |

```bash
# 单卡(写卡时用,三个脚本都要过)
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <card.md>
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <card.md>
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <card.md>

# 全量
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --all \
  --k1 paper2skills-research/data/verification/k1_l5.json \
  --outdir paper2skills-vault/07-资源库/gates
```

**G2 基线(2026-09-12,131 张卡)**:旧报告写的「**57 张通过(43.8%)**」**不成立**,已作废。两点原因:

1. 旧 `passed` 判定写死 `len(red)==0`,于是**没有任何数字的卡片自动通过** —— 那是「没东西可查」,不是「有证据」。
2. `NOISE_PATTERNS` 里 `^\d{1,2}$` 与 `^v?\d+\.\d+$` 两条,把**所有 1–2 位数字与所有小数**一律豁免 ——
   含 `提升 15%`、`92.2%`。旧基线把这些断言整个漏掉了。

加固后的真实基线:**0/130 通过**(新出的卡片 `Skill-Cannibalization-Corrected-Attribution.md` 是首个通过者,1/131),
红灯 2,742 条。根因是**全库 `> 原文:"..."` 引用块实测为 0 条**
(旧报告称「仅 3 行」,那是 crude grep 的假阳性 —— 那 3 处是「数据忠实于原文」这类散文,不是引用块)。

> 这个修正本身是重要教训:**门禁数字变差,不一定是资产变差,可能是门禁终于开始测真东西了。**
> 三个假绿灯(上面 1、2 与 evidence.md 无过滤收录)全部是「门禁自己放水」,比假红灯危险得多。

**G2 的 18 条已封堵漏洞**（全部由实测发现，不是理论担忧）

> 📁 **逐条表格（#1–#18：漏洞 / 后果 / 修法与实测读数）已搬到
> `paper2skills-research/reports/G2漏洞表归档-18条.md`**（2026-09-13，因 instruction 预算超限搬出，
> **逐字节复制，内容未改**）。逐条对应的现场记录见 `PHASE6-工具详解与缺陷史.md`。

留在本文件的**结构性结论**（比单条修复重要）：
- **每一条都是「门禁自己放水」**，而假绿灯比假红灯危险得多（假红会有人去看，假绿没人知道）。
- 三类反复出现的形态：**判据只认一种写法**（#2 引号正则、#11 只认一种字段名、#14 漏汉字后的数字）、
  **豁免条款本身成了后门**（#3 `evidence.md` 全收、#4/#5 数字豁免、#13 引文跨行）、
  **判据用错了信息源**（#16 引用编号、#17 日期与 LaTeX、#18 标识符）。
- ⚠️ **#10 的教训单列**：修 #8/#9 时新增了一个「不阻塞」的结局，**这本身就是一次放水**。
  任何「新增豁免」都必须同时写清「什么情况下不许豁免」—— **门禁的豁免条款必须比它的拦截条款测得更严**。

### 出卡 / 体检的日常命令(按顺序)

```bash
# 1. 抓全文(缺这一步则引文无从核验)
python3 paper2skills-research/scripts/fetch_fulltext.py --arxiv <id> --domain <域> --paper-id <p2s-id>

# 2. 出卡后三门禁自验(全部必须绿)
C=paper2skills-vault/<域>/Skill-<名>.md
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C"
python3 paper2skills-skills/paper-审核/scripts/quote_check.py   --card "$C"
python3 paper2skills-skills/paper-审核/scripts/gate_check.py    --card "$C"

# 3. 同步(门禁红灯会被拒绝)
python3 paper2skills-skills/paper-同步/scripts/sync.py --skill <名>

# 4. 每周体检
python3 paper2skills-skills/paper-维护/scripts/repo_health.py \
  --json-out paper2skills-research/data/health/repo_health.json
```

**三个脚本都要跑,不能只跑 gate_check**:`gate_check` 的 G1 依赖 K1 产物,
而 `quote_check` 才是判定引文真伪的那一层。只跑 `gate_check` 会得到一个
「有出处但出处可能是编的」的绿灯 —— 这正是本轮封堵的漏洞之一。

## ⚠️ 事实核验的两条铁律(2026-09-13 立,由三次误判换来)

这两条凌驾于「与底本逐字比对」之上 —— 因为**底本不是万能的仪器**。

### 铁律 1:venue / 发表类声明 → 用 arXiv 元数据 + 出版方 DOI 判定,**不得用底本正文判定**

**论文正文几乎从不写自己的 venue。** 所以「底本里 `SIGIR` 0 命中」**不是**「没有录用证据」——
它只是**用错了仪器**。正确仪器:
- arXiv abs 页的 `Comments:` 字段(`Accepted at …` / `post-proceedings of …` / `Workshop` / `Under review`)
- `journal_ref` 字段
- **出版方 DOI**(Crossref:`container-title` / `event` / `pages` / `published-online`)

**实测代价**:台账 A3 曾据「底本 `SIGIR` 0 次」判定「无录用证据 → 应降为 preprint」,
A1 曾据「底本 `ECML`/`PKDD` 零命中」判定「疑似版本错配」——**两条都是假的**。
复核:arXiv v2 `Comments` 明写 `Accepted at ACM SIGIR 2026 Industry Track`,
DOI `10.1145/3805712.3808466` 在 Crossref 解析为 SIGIR '26 proceedings(page 4763-4768);
`2312.07206` 的 `Comments` 写明 `post-proceedings of the ECML PKDD 2023 Workshop`。
**照原判执行会删掉两个真实且可核验的录用事实。**

### 铁律 2:内容类声明 → 先问「卡片写的是**哪一版**」,再判对错

**底本与卡片版本不一致时,先怀疑底本抓错了版本,而不是先怀疑卡片。**

**实测代价**:台账 A2 曾判定卡片「方法名与论文不一致、`10.79%` 疑似编造」。
真相是 `2402.09176` 有 **v1/v2 两版**,仓库底本抓的是 **v1**,而**卡片写的是 v2(WSDM 2025 正式版)**;
台账称「编造」的每一项都在 v2 **逐字命中**(`ColdLLM` 62 次)。**照原判执行会把一张正确的卡改成错的。**

> ### 这两条与本仓库既有教训同源,只是第四个变体
> 漏洞 #11(判据只认一种字段名)、K1 的 `ORPHAN_DEP`(判据只认一种路径)、
> C2(口径未跟上口径)—— **本轮是「判据用错了信息源」。**
> 共同结构:**判据的适用范围被默认成了全体,而它其实只覆盖一个子集。**
>
> 📌 推论(比单条修复更重要):**每次「某个东西不存在」的结论,都要先问「我用的仪器能看见它吗」。**
> 这与「判某个东西不存在之前先 `ls` 一次」是同一条纪律的推广。

## 检索路线(2026-09 实测结论)

| 路线 | 覆盖 | 调用 | 实测 |
|------|------|------|------|
| arXiv API | 预印本、方法论、模型类 | `https://export.arxiv.org/api/query` + `submittedDate:[start TO end]` | ✅ 49 查询 → 1046 篇(92 天)。**必须 https + `curl -L`**(http 返回 301 空 body) |
| Crossref 期刊 | UTD24/FT50/CCF 期刊**正式在线发表** | `api.crossref.org/journals/{issn}/works?filter=from-pub-date:..,until-pub-date:..` | ✅ 28 刊 → 1751 篇 |
| Crossref 会议录 | KDD/SIGIR/WWW/WSDM 论文集 | `api.crossref.org/works?query.container-title={proceedings 全名}` | ✅ **dl.acm.org 403 的唯一替代**,含 `event.location` |
| PMLR 卷级 BibTeX | UAI/AISTATS/MLSys | `proceedings.mlr.press/v{N}/assets/bib/bibliography.bib` | ✅ **一次调用全量**(UAI v337 = 639KB,330 篇) |
| ACL Anthology | ACL 2026 全 7 卷 | `aclanthology.org/volumes/2026.acl-long.bib` | ✅ 但 >60s,须 `--max-time 180`;**无 JSON 端点** |
| OpenAlex | 跨源补充与交叉校验 | `api.openalex.org/works?filter=primary_location.source.issn:{issn},...` | ✅ 免鉴权;⚠️ **按上线日索引,与 Crossref 封面日语义不同** |
| **OpenReview** | DBLP 镜像 + 已回填 venue 的会议 | ✅ `/notes/search?term=..&limit=1000&type=terms` 与 `/groups?id=..` 实测 **200**;❌ `/notes` 仍 403 | ⚠️ **可用但覆盖易高估**(见下) |
| **DBLP** | — | — | ⛔ **彻底不可用**(见下) |
| Semantic Scholar | — | — | ❌ 429 |

**四个必须记住的坑**:
1. **顶刊"新发表" ≠ 新方法** — 实测 79% 的顶刊文章 DOI 年份段比发表年份早 ≥2 年
   (例:`10.1287/mnsc.2022.02462` 发表于 2026-07-09 但 DOI 为 2022)。萃取时应优先找预印本版本。
2. **Crossref 多关键词抓取会跨组重复计数** — 同一批 120 条记录实际只有 91 篇唯一论文(重复率 24%),
   排序前必须先按 DOI 去重。
3. **Crossref 必须同时打 `from-pub-date` 与 `from-print-pub-date`** — SAGE 系(JM/JMR/POM)的
   `published` 是 online-first 日期,只打前者会漏掉整期。实测 POM:普通过滤 55 条且几乎全无卷期,
   print-date 过滤给出干净的 `{35(7):16, 35(8):16, 35(9):17}`;ISR 的 print-date 过滤结果为 **0**
   (证明其本季纯 Articles in Advance,无正式期号)。
4. **DBLP 不是"限速问题"** — `dblp.org`/`dblp.uni-trier.de`/`dblp.dagstuhl.de` 三镜像全部返回
   **HTTP 200 + Anubis proof-of-work 挑战页**(`<title>Making sure you're not a bot!</title>`)。
   换 UA、降速、换镜像、换 API 路径**全部无效** → **脚本路线应彻底放弃**。
   最危险之处是状态码为 200,脚本若不检查正文会把挑战页当数据解析。
   替代:Crossref(ACM DL DOI) + OpenAlex。

**OpenReview 的精确边界(勿高估)**:`/notes/search` 支持精确短语匹配 venue 且 `limit=1000` 一次全量,
但实测 `"SIGIR 2026"` 返回的 123 条里 117 条 invitation=`Record` 且带 `dblp:` externalId,
**是 DBLP 镜像,不等于正会**(SIGIR 正会 274 篇未覆盖);`"ICML 2026"`/`"ICLR 2026"`/`"ACL 2026"`/
`"AAAI 2026"` 精确短语查询返回 **0**(venue 字段未回填);`"EMNLP 2026"` 返回的 147 条实为
workshop 名(GroundLM/NLP4PI)。**正会枚举仍需个人 token**。

**`NAACL 2026` 不存在**(2026 停办一届):`2026.naacl.org` DNS 不解析、
`aclanthology.org/events/naacl-2026/` 404、`naacl.org` 只挂 2027。
venue 白名单中不得出现该项。完整规则见 `paper2skills-vault/07-资源库/venue-whitelist.md`。

## 存量资产体检（基线）↗

> 📁 **逐项基线表（卡片/引文/K1/G1/G2/G3 通过率、v2 字段欠账、provenance 五层、registry、
> 领域分布）与 PHASE4-A / PHASE3 交付明细、registry 反向修正的 4 处断言，已移至
> `paper2skills-research/reports/基线体检归档-20260912.md`**（2026-09-13，因工作区 instruction
> 预算超限被搬出，**逐字节复制，内容未改**）。

**结论留在本文件**：146 张卡 / 90 张有引文（2,087 条，**0 伪造**）/ K1 执行率 **62.0%** /
G1 **42.5%** / G2 **15.3%**（可核验分母 98，另 48 张设计上无论文来源）/ G3 **45.9%**。
⚠️ **G2 的可核验通过率有结构性上限** —— 残余红灯主体是「作者外推」（⑤ 商业价值与 ② 应用案例里的
工期/ROI/倍数，论文里本就不存在），**继续补引文不会让它们变绿**。
⚠️ **数变差通常是门禁变准**：本仓库已发生 4 次「提升来自修门禁而非改资产」
（46.2→52.5→54.8% / ORPHAN 9→0 / 红灯 1,308→1,476 / C2 口径失真）—— 所以每次盘点都要**主动去撞工具**，
而不是只看它有没有报错。

## Workflow Commands

### Run Complete Workflow

Use the `paper-workflow` skill to run the complete pipeline:

```bash
# Trigger via natural language to Claude
"Run paper2skills workflow"
"Process this paper through the complete pipeline"
```

### Individual Steps

Each step can be triggered separately via skills:

1. **Paper Selection** (`paper-选题`): "筛选论文" / "Search ArXiv for uplift modeling papers"
2. **Extraction** (`paper-萃取`): "萃取论文" / "Generate skill card from this paper"
3. **Review** (`paper-审核`): "审核 skill" / "Review quality of Skill-Uplift-Modeling"
4. **Sync** (`paper-同步`): Use the sync script directly (see below)

### Sync Script Usage

```bash
# Sync a skill to vault and GitHub
cd paper2skills-skills/paper-同步
python scripts/sync.py --skill Skill-Uplift-Modeling

# Sync to specific targets
python scripts/sync.py --skill Skill-Uplift-Modeling --target vault,github

# View sync status
python scripts/sync.py --skill Skill-Uplift-Modeling --status

# Sync all tracked skills
python scripts/sync.py --status
```

## Skill Card Format

**新卡片一律走 `MasterPrompt-v2.md`**,其结构为 6 段 + frontmatter v2:

0. **frontmatter** — 含 `paper_id` / `paper` / `venue` / `venue_tier` / `evidence_grade` / `related`(缺字段即未完成)
1. **算法原理**(≤300 字)— 核心思想 / 数学直觉 / 关键假设
1b. **反例与适用边界** — 什么时候不要用 / 已知失败模式 / 论文自承局限(无则显式写"论文未讨论")
2. **母婴出海应用案例**(1-2 个)— 业务问题 / 数据要求 / **数据可得性(必填)** / 预期产出 / 业务价值
3. **代码模板** — 可运行 Python + 测试;须能过 K1(不得 import 仓库内不存在的模块、不得有 `plt.show()`/网络请求、须含 assert)
4. **技能关联** — 必须真的引用 `Skill-*.md` 文件名,≥2 个
5. **商业价值评估** — ROI 须给公式或参数来源 + 难度/优先级星级
6. **原文引用(必填,≥3 条)** — `> 原文:"<逐字摘录>"` + 出处(arXiv ID / 章节 / 页码)

> v1 的五段式结构(`MasterPrompt.md`)仅保留供对照。v2 新增的第 0/1b/6 段
> 分别对应 G2 溯源门禁、边界声明要求、证据链要求。

## Code Standards

### Python Code Template Structure

```python
# Each module should have:
# - model.py: Core algorithm implementation
# - __init__.py: Module exports
# - Example data generation functions
# - Business-specific scenario code

# Example: paper2skills-code/causal_inference/uplift_model/model.py
class UpliftModel:
    """Uplift Modeling meta-learner framework"""
    def __init__(self, method='xlearner'): ...
    def fit(self, X, treatment, outcome): ...
    def predict(self, X): ...
```

### Running Code Tests

```bash
# Test a specific model
cd paper2skills-code/causal_inference/uplift_model
python model.py

# Or use pytest (if tests are added)
python -m pytest model.py -v
```

## Domain Mapping

下表 "Code Dir Status" 标识 `paper2skills-code/` 下对应子目录的落地状态:
- ✅ 已落地 — 目录存在,可 import;
- 📦 镜像保留 — 子项目已迁出本仓库,代码模板留作复用引用;
- ⬜ 仅 vault — 当前只有 Skill 卡片,无 code 子目录,需要时按 snake_case 新建。

| English Directory | Chinese Directory | Domain | Code Dir Status |
|-------------------|-------------------|--------|-----------------|
| `ecommerce_agent` | `00-电商Agent` | Live-catalog conversational rec, agentic catalog enrichment | ⬜ |
| `causal_inference` | `01-因果推断` | Causal inference, uplift modeling | ✅ |
| `ab_testing` | `02-A_B实验` | A/B testing, multi-armed bandits | ✅ |
| `time_series` | `03-时间序列` | Demand forecasting, time series | ✅ |
| `supply_chain` | `04-供应链` | Inventory optimization | ✅ |
| `recommendation` | `05-推荐系统` | Recommendation systems | ✅ |
| `growth_model` | `06-增长模型` | Churn prediction, LTV | ✅ |
| `nlp_voc` | ~~07-NLP-VOC~~ | 已迁至 `../ai_nlp_voc/`,本仓库保留代码模板 | 📦 |
| `knowledge_graph` | `08-知识图谱` | Heterogeneous graphs, hyperbolic embedding | ✅ |
| `data_agent_llm` | `09-DataAgent-LLM` | DataAgent, LLM-powered data analysis | ✅ |
| `mas` | `10-MAS` | Multi-agent systems, planning, orchestration | ✅ |
| `ai_humanities` | `11-AI人文` | AI × Humanities: cross-modal transfer, LoRA, continual learning, prompt tuning as life metaphors | ⬜ |
| `ml_fundamentals` | `12-ML基础` | Feature engineering, model evaluation fundamentals | ⬜ |
| `advertising` | `13-广告分析` | Ad attribution (Shapley/Markov), ROAS optimization, budget allocation | ⬜ |
| `user_analytics` | `14-用户分析` | Funnel analysis, cohort retention, RFM segmentation | ⬜ |
| `marketing` | `15-营销投放分析` | Marketing Mix Modeling (MMM), promotion effectiveness, causal ML | ⬜ |
| `llm_agent_engineering` | `16-智能体工程` | Agent Skills/Tools, Context Engineering, MCP/A2A protocols, Function Calling (Hermes) | ✅ |

## Quality Standards

Skills must meet these criteria (enforced by `paper-审核`):

- **Algorithm Principle**: Original explanation (not copied), includes math intuition
- **Applications**: Specific scenarios, not generic; must relate to mother & baby cross-border e-commerce
- **Code**: Complete and runnable, includes test cases, clear I/O definitions
- **Skill Relations**: Links to ≥2 existing skills
- **Business Value**: Quantified ROI estimates (no vague terms like "high" or "low")

**Pass threshold**: Total score ≥ 7/10 with code dimension ≥ 7/10

## Working with Skills

### Skill Evolution

Skills can be improved through the `evolve/` directories:

```
paper2skills-skills/paper-选题/evolve/
├── evolution-log.md    # Tracks improvement iterations
└── round-1/            # Specific evolution rounds
```

### Skill File Format

```markdown
---
name: paper-workflow
description: This skill should be used when...
version: 0.1.0
---

# Skill content...
```

## Dependencies

Install Python dependencies:

```bash
cd paper2skills-code
pip install -r requirements.txt
```

Key packages: numpy, pandas, scikit-learn, statsmodels, prophet, causalml, econml

## ArXiv Search Strategy

Use the keyword library at `paper2skills-vault/07-资源库/关键词库.md`:

```bash
# Example ArXiv API query
curl "https://export.arxiv.org/api/query?search_query=all:uplift+modeling&start=0&max_results=10"
```

Search priority: Papers with code implementations > experimental validation > theoretical only. Exclude surveys, meta-analyses, and pure theory papers without experiments.

## Recent Skills Added

| Date | Skill | Domain | Commit |
|------|-------|--------|--------|
| 2026-09-12 | **PHASE4-A（存量卡证据链补齐，不新增卡片）**: 129 张卡 +5,721 行；引文 729→2,087 条（+90 张卡有引文）；封堵 5 个门禁假绿灯（#11–#15）；F3/F4 分层工单全覆盖 | 全库 | `a8055a7` / `ec716da` / `8b05d6f` |
| 2026-09-12 | **PHASE3 批次 3E（增强既有卡，不新增）**: GraphRAG 18/18 · Agentic-Memory 17/17 · Feature-Engineering 30/30 · Cold-Start-PAM 32/32 | 08-知识图谱 · 16-智能体工程 · 12-ML基础 · 05-推荐系统 | `8c8652d` / `07141ca` |
| 2026-09-12 | **PHASE3 批次 3A–3D（18 张新卡，三门前全绿）**: 归因蚕食校正 / 多档券 uplift / 因果预算分配 / 季节性流失标签 / 决策条件预测 / 海运成本 / 供应链仿真 / 多仓分配 / 增量测量 / 活跃目录推荐 / 目录属性补全 / 状态化 Skill 运行时 / 多智能体协作税 / 路由图交接 / Text-to-SQL 权限门禁 / 人格条件 A/B 仿真 | 13-广告分析 · 06-增长模型 · 03-时间序列 · 04-供应链 · 14-用户分析 · 00-电商Agent · 16-智能体工程 · 10-MAS · 09-DataAgent-LLM · 02-A_B实验 | 见 `10e97da` 起 |
| 2026-05-15 | Marketing Mix Modeling (MMM) + Promotion Effectiveness (DML) | 15-营销投放分析 | — |
| 2026-05-15 | Ad Attribution Modeling + ROAS Budget Optimization | 13-广告分析 | — |
| 2026-05-15 | User Funnel Analysis + Cohort Retention Analysis | 14-用户分析 | — |
| 2026-05-15 | Feature Engineering fundamentals | 12-ML基础 | — |
| 2026-05-15 | AI Tech × Healing Quotes Card Library (4 directions: StructLoRA, InfLoRA, Prompt Tuning, Cross-Modal Transfer) | 11-AI人文 | — |
| 2026-05-11 | Phase 7 D4 Superset native filters | 07-NLP-VOC (已迁至 ai_nlp_voc) | `311e3bd` |
| 2026-05-10 | Phase 6 D10 BI dashboard C path | 07-NLP-VOC (已迁至 ai_nlp_voc) | `cad5be5` |
| 2026-05-08 | Phase 7 D1-D3 Superset BI B path | 07-NLP-VOC (已迁至 ai_nlp_voc) | `a765876` / `6f9211d` / `0d92103` |
| 2026-05-06 | Self-Improving LLM Agent Pipeline | 07-NLP-VOC (已迁至 ai_nlp_voc) | `985e82b` |

## Sync Status Tracking

The sync system tracks publication status across platforms:

- **vault**: Obsidian knowledge base
- **github**: Code repository
- **feishu**: Lark/feishu webhook (requires `~/.paper2skills/feishu_webhook` configuration)

Check status in: `paper2skills-vault/07-资源库/sync_status.json`

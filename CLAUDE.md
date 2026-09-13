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
| `paper2skills-skills/paper-审核/scripts/quote_check.py` | **G2b 引文逐字核验**:把每条 `> 原文:"..."` 回查论文全文存档,判 VERBATIM/FUZZY/FABRICATED。`--selftest` 用四个用例自证能区分真引文/伪造/拼接 |
| `paper2skills-research/scripts/fetch_fulltext.py` | **全文抓取**:arXiv LaTeXML HTML → Markdown(保留章节号),存到 `papers/<域>/<paper_id>/fulltext.md`;萃取前必跑,否则引文无从核验。PHASE4 起支持 `--pdf`(本地 PDF→fulltext,含硬换行接回)与 `--from-worklist`(按审计结果批量补) |
| `paper2skills-research/scripts/provenance_audit.py` | **卡片论文溯源可达性体检**:把「G2 全红」拆成**已核验 / 可修 / 无论文来源**三层,回答「哪些卡该修、哪些卡修不了」。`--selftest` 锁定三层判定互斥。⚠️ 首版只扫 frontmatter 导致 47 张卡被误判,现扫正文并剥尾部参考区 |
| `paper2skills-research/scripts/registry_consistency.py` | **registry 门禁声明 vs 实物核对**。把**覆盖率**作为一等输出(核对到 N/M),低于 90% 拒绝给「一致」结论。⚠️ 它诞生的原因就是前身脚本静默跳过 3 条 `enhanced_cards` 记录却报「无不一致」 |
| `paper2skills-vault/07-资源库/关键词库-v2.md` | **三段式检索词**(正向 / 负向 / 约束词)。⚠️ 负向词**按域生效** —— 同一个词在不同域含义相反(`trial`/`cohort`/`ad`),无脑全局负向会误杀方法论论文 |
| `paper2skills-research/scripts/candidate_filter.py` | 三段式过滤的**可执行部件**(配置若只有文字一定会腐烂)。实测丢弃 25.8%,逐篇抽检无误杀;`--selftest` 含同词异义反例 |
| `paper2skills-vault/07-资源库/scoring_config.json` | 评分权重/阈值/关键词表外置。缺文件时脚本退回内置默认值(不静默用空值) |
| `paper2skills-skills/paper-维护/scripts/repo_health.py` | **仓库体检**(C1–C8:重复卡片/frontmatter/路径/围栏结构/registry/门禁时效/卫生/段落完整性)。`--selftest` 用构造样本证明检查真的会报警 |
| `paper2skills-skills/paper-维护/scripts/scan_secrets.py` | **推送前凭证扫描门禁**。13 条规则 / 10 类凭证;**先把 `\"` 与 `&quot;` 归一化成 `"` 再匹配**,故一条规则即覆盖全部转义形态。扫到 **0 个文件时判失败(退出码 2)** —— 「没东西可查」不等于「查过了没问题」 |
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

> **`DDDD.pem` 的更正**:它首行是 `BEGIN RSA PRIVATE KEY`,即 **RSA 私钥**,
> 不是此前记载/以为的「SSH 公钥」。本仓库从未 commit 过它(28 个 commit 全扫,0 命中),
> 但它另有 3 份副本散落在 `~/Downloads/`、`~/project/思维模型/`、`~/project/Agent/lute-momcozy-audit/`。

### 推送前必跑

```bash
python3 paper2skills-skills/paper-维护/scripts/scan_secrets.py \
  --json-out paper2skills-research/data/health/secrets.json
```

退出码:0 = 干净;1 = 命中凭证;2 = **一个文件都没扫到**(这不是干净,是没测)。
该仓库 `secret_scanning_push_protection` 已开启,GitHub 会直接拒收含凭证的 push。

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
| **G3 业务可落地** | 场景是否具体、是否声明数据可得性、ROI 是否有依据、是否关联 ≥2 张卡 | 空泛表述黑名单 + 母婴出海具体信号计数 |

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

**已封堵的 11 个 G2 漏洞(全部由实测发现,不是理论担忧)**

| # | 漏洞 | 后果 |
|---|------|------|
| 1 | 只判「有无出处」不判「出处真假」 | 伪造一段带数字的引文即可让任意数字过闸 |
| 2 | 两个脚本各有一套引用正则(`> 原文:"..."` 匹配不到) | 带合格引文的卡片反被判「无任何出处」——**假红灯** |
| 3 | `evidence.md` 数字无条件全收 | 在 evidence.md 裸写 `ROI 提升 42.7%` 即可洗白 |
| 4 | `^\d{1,2}$` 豁免 | **所有 1–2 位数字**免检,含 `提升 15%` |
| 5 | `^v?\d+\.\d+$` 豁免 | **所有小数**免检,含 `92.2%` |
| 6 | 行内代码被剥离 | 把论文事实写成 `` `2–3 倍` `` 即绕过 G2 |
| 7 | 中文数字不匹配 | 写成「两三倍」即绕过全部数字检查(跨语言数字匹配不可机械化,现改为黄灯提示) |
| 8 | **分母把两类人混在一起**(2026-09-12) | 56 张**从设计上无论文来源**的经验卡被记成「有缺陷」→ 通过率 11% 同时混进了两类完全不同的东西。现拆 `outcome: PASS / FAIL / UNVERIFIABLE`,`evidence_basis` 声明决定分类,**通过率只在可核验分母上算** |
| 9 | **「无法核验」被平均成「通过」** | 与 #8 同源:若不拆,把 56 张没法验的卡并进分母,全库通过率会因为**多了没法验的卡**而看起来变好。现 `summarise()` 单列 `cards_unverifiable` |
| 10 | **新口径可能变成新后门**(预防性) | 新增一个「不阻塞」的结局本身就是一次放水风险 —— 加一行 `evidence_basis: author-practice` 就能全库免检。故 `card_has_paper_source()` 检测**声明与实物矛盾**:卡里明明有 arXiv/DOI 却自称无论文来源 → `G2-BASIS-CONTRADICTION` 红灯 |
| 11 | **尾部署「参考论文」区被当成来源声明** | 实测 `Skill-Intelligent-Attribution-Causal-Forest` 的「参考论文」出现在第 13853 字符(卡片总长 14069),一张纯经验卡会因为列了几篇延伸阅读而被迫按「有来源卡」审查。现 `strip_reference_section()` 先剥掉尾部参考区再判。⚠️ 词汇表必须含 `参考资料` —— 全库 43 张用 `参考论文`,另 3 张用 `参考资料`(实测 `Skill-Two-Echelon-Inventory-DRL` 的该段是**混合表**,第 1 条就是本卡来源论文) |
| 12 | **frontmatter 元数据被当成事实断言**(2026-09-12 由 F4 子代理发现) | `created: 2026-05-15` / `updated: 2026-09-12` 被数字正则拆出 `15`、`12` 判成「一般数字无出处」;**实测影响 65 张卡、126 条黄灯**,全是元数据。frontmatter 里的 `paper_id` 同样被当数字 → 红灯也随之虚高。现 `strip_frontmatter()` 在数字扫描前剥掉整个 YAML 块(它只用于**分类**,不承载断言) |
| 13 | **自引文洗白**(2026-09-12 由 F4 子代理实测复现) | `QUOTE_RE` 把开引号集 `["“「『]` 与闭引号集 `["”」』]` 写成两个**独立**字符类 → **任何闭引号能闭合任何开引号**;配合 `re.S`,`> 「…」` 这类行会**跨行吞掉卡片正文**。叠加「`UNVERIFIABLE` 引文也算出处」这条善意豁免后形成完整后门:**写一段引文 + 一个指不到底本的 `paper_id`,卡片自己的数字就成了「有出处」**。实测 `Skill-AB-Experimental-Design` 被抽出一条 **2777 字符的「引文」**(90% 是卡片自身正文),`sourced=20/26、traceability=76.9%` —— 红灯 28→6 是假象。两层都修:①引号**成对同型**+去掉 `re.S`;②`UNVERIFIABLE` **不再计入出处**(只出黄灯)。`--selftest` 用例 6/7 锁定 |
| 14 | **中文卡的数字被系统性漏检**(2026-09-12 由子代理实测发现) | `NUM_TOKEN_RE` 的前置断言写成 `(?<![\w.])`,而 Python 的 `\w` **把汉字也算单词字符** → **紧跟在汉字之后的数字完全不被扫描**。实测:`提升 35%`→扫到;`准确率92.2%`/`返工50%`/`落地率40%`/`转化率提升40%`→**全空**;`ROI约7-10倍`→只扫到后一个。而中文卡绝大多数断言恰恰是「名词+数字+单位」无空格形式 —— **最该检查的一类恰好是唯一被静默跳过的一类**。修法:`(?<![0-9A-Za-z_])`(拉丁字母/数字/下划线才算 token 内部字符)。修后红灯 1,308→1,476、黄灯 575→645,**这才是真实值** |
| 15 | **引文内嵌双引号被截成残句**(2026-09-12 由子代理发现 1 例,全库扫描另查出 3 例) | `"[^"\n]+"` 非贪婪,遇到内层引号就闭合。实测 `> 原文:"These results indicate that the LLM-based "Fuzzy" Random Forest model is …"` 被抽成 **42 字符的残句** `These results indicate that the LLM-based ` —— 而**残句确实逐字存在于底本**,于是报 VERBATIM:**门禁核验的不是卡片声称的那句话,而是一段更短的碎片**,且报告里看不出来(`n_quotes` 与逐字命中都正常)。全库 2,085 条引文中 **4 条**中招。修法:ASCII 分支改**贪婪** `"(.+)"`,匹配到行内最后一个引号;`--selftest` 用例 5 锁定「抽出的必须与写入的一致」 |

> **#10 的教训值得单列**:修复 #8/#9 时新增了一个「不阻塞」的结局,**这本身就是一次放水**。
> 任何「新增豁免」都必须同时写清「什么情况下不许豁免」,否则修复动作会制造下一个漏洞。
> `gate_check.py --selftest` 的用例 3(声明与实物矛盾)与用例 4(伪造引文不得因声明免检)
> 就是为此而设 —— **门禁的豁免条款必须比它的拦截条款测得更严**。

> **引文核验器必须先自证可信**:`quote_check.py --selftest` 用四个用例锁定行为,
> 其中「拼接引文」一条是开发中实测发现的漏洞 —— 把摘要句与引言句缝成一句(论文里不存在这句话)时,
> 全文档 n-gram 覆盖率仍是 **1.0**(每个碎片都能在文档某处找到),连续度只有 0.681。
> 故判定改用「最长**连续**匹配段占引文的比例」。

> ⚠️ **口径警告**:三个门禁必须**分开报**。`Cited but Not Verified`(arXiv:2605.06635) 实测
> 链接可用 >94%、主题相关 >80%,而**事实一致性只有 39–77%** —— 合成一个「可信度总分」
> 会把最弱的那一维平均掉。

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

## 存量资产体检(2026-09-12 实测)

> 本节数字为 T0-2 去重 + PHASE3 批次 3A **之后**的实测值。
> ⚠️ **门禁口径在 2026-09-12 当天由松变紧**(修了 7 个假绿灯),所以 K1/G1/G2/G3 四个数字
> **不能与当天早些时候的报告直接对比** —— 门禁数字变差,可能是门禁终于开始测真东西了。

> ⚠️ **本节已更新到 PHASE3 批次 3A–3E + PHASE4-A 之后**（2026-09-12 14:10 实测）。旧值在下方「基线变化」里对照。

| 指标 | 实测值 | 说明 |
|------|--------|------|
| 唯一 Skill 卡片 | **146 张** | 去重前 156;PHASE3 新增 8 张(3A–3D)+ 增强 4 张既有卡(3E,不新增)。`07-NLP-VOC/` 下 26 组同名重复已在 T0-2 清理 |
| 含 `> 原文:"..."` 引用块 | **90/146 张 / 2,087 条** | PHASE4-A 前为 20 张 / 729 条 —— **+70 张、+1,358 条**,全部逐字核验通过 |
| 含 frontmatter | **144/146** | ⚠️ **不是 146/146** —— `Skill-Two-Echelon-Inventory-DRL.md` 与 `Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md` **至今无 frontmatter**(后者已在 3E 补了 18 条引文,却仍缺)。此处曾误记为 146/146,2026-09-12 盘点时由 `repo_health` C2 与人工复核推翻 |
| v2 必填字段实际欠账 | `paper_id` 8 / `paper` 29 / `venue` 66 / `venue_tier` 74 / `evidence_grade` 78(共 96 张有来源卡) | ⚠️ `repo_health` C2 报的「126 张」**含 48 张 `author-practice` 卡的误报** —— 它们按设计就不该有来源字段(见漏洞 #8/#10 的三类口径),C2 尚未跟上该口径 |
| 含 `paper:` / `paper_id:` 溯源字段 | 98 张 | 另有 48 张声明 `author-practice`(设计上无论文来源),见 `provenance_audit.py` |
| 含 ` ```python ` 代码块 | **97/146 张 (66.4%) / 141 个块** | 按行首 ```python 围栏直接计数;K1 的「100 单元」是把卡片内代码块**按文档顺序拼成一个模块**后的单元数,两者口径不同,不要互相换算 |
| **K1 代码执行率** | **62.0%** | 100 单元:PASS 62 / ENV_BLOCKED 7 / **ORPHAN_DEP 0** / MIGRATED_DEP 9 / FAIL 22;语法级失败 0 |
| **G1 门禁通过率** | **42.5%** | 62/146,红灯 68 / 黄灯 7 |
| **G2 事实溯源通过率** | **16.3%** | **16/98**(可核验分母),红灯 1,476 / 黄灯 645;另有 **48 张无法核验**(无论文来源) |
| **G3 业务可落地通过率** | **45.9%** | 67/146,红灯 93 / 黄灯 174 |
| **引文逐字核验** | **2,087 条,90/90 张 VERBATIM** | **0 伪造 0 近似 0 拼接**;另 2 张判 `NO_FULLTEXT`(有来源、无全文底本) |
| provenance 四层(split) | VERIFIED 83 / RETROFIT_READY 8 / NEEDS_FULLTEXT 7 / **NO_PAPER_SOURCE 48** | `provenance_audit.py` 产出,四类互斥 |
| registry | 45 条(extract 31 / watch 14)。**已交付卡 16 + 已交付为增强 3,真·待萃取 12** | ⚠️ 统计时必须同时看 `outputs.skill_card` **与** `outputs.enhanced_cards` —— 只数前者会把 3 篇已作为 3E 增强交付的论文(`0023`/`0026`/`0030`)误记成「未出卡」。另有 **8 条 `title` 仍是 `(待补：见 note)`**(标题实际写在 `decision_reason` 里) |
| 仓库体检欠账 | C3 🟡16 / C8 🟡58 | C1/C4/C5/C6/C7 ✅;⚠️ **C2 口径失真**(见上表) |
| 领域分布 | 07-NLP-VOC 41 / 16-智能体工程 17 / 10-MAS 14 / 06-增长模型 11 / 08-知识图谱 9 … | 11/12 域各仅 1 张 |

> ⚠️ **G2 红灯比 PHASE4 之前「更多」是正常的**:PHASE4 修了 3 个假绿灯(#12/#13/#14),
> 其中 #14(中文卡数字被系统性漏检)让红灯从 1,308 升到 **1,476** —— 那是**门禁终于开始
> 测真东西了**,不是资产变差。同一时期卡片侧新增了 1,358 条逐字引文。
> **两个方向的数字同时变化时,不要拿「红灯总数」当进度指标**;要看
> 「可核验分母上的通过率」与「引文逐字核验条数」。
> 📌 本仓库已发生 **4 次「提升来自修门禁而非改资产」**(46.2→52.5→54.8%、ORPHAN 9→0、
> 红灯 1,308→1,476、以及本轮的 C2 口径失真)—— 所以每次盘点都要**主动去撞工具**,
> 而不是只看它有没有报错。

### PHASE4(2026-09-12)· 存量卡证据链补齐

> ⚠️ **命名消歧:本仓库现在有两个「PHASE4」,不要混用。**
> - **PHASE4-A(实际执行)** = 本节的「存量卡证据链补齐」 → ✅ 已完成
> - **PHASE4(原计划)** = 空白领域补卡(T4-1 12-ML基础三张卡 / T4-2 17-跨境合规新域 /
>   T4-3 11-AI人文改流程 / T4-4 06-增长模型供给策略) → ❌ **一项未执行**,
>   自 2026-09-12 起**改称 PHASE6**
>
> **下一轮执行视图见 `paper2skills-research/reports/PHASE5-明日执行TODO.md`。**

把 146 张卡按「能不能修」分成四层，然后分别处理(`provenance_audit.py` 产出工单):

| 层 | 张数 | 处置 | 结果 |
|----|------|------|------|
| 已核验 | 83 | 无需动 | 保持 |
| 可补引文(有全文底本) | 8 | F3 | 补逐字引文 |
| 可补引文(需先抓全文) | 7 | F3 | 先 `fetch_fulltext` 再补 |
| 无论文来源 | 48 | F4 | 补 `evidence_basis: author-practice` + 证据基础声明 |

**交付**:129 张卡、**+5,721 行 / −103 行**(其中 102 行是 frontmatter 元数据规范化,
1 行是补文件末尾换行)—— **没有改写任何既有正文与数字**。

> **一个必须记住的口径**:补引文后剩余的红灯,主体是**「作者外推」**。
> 三个 F3 分组独立统计后一致:**75%–85% 的残余红灯是 v1 五段式 ⑤「商业价值评估」
> 与 ②「应用案例」里的工期/ROI/倍数/示例数据** —— 论文里本就不存在这些事实。
> 故 **G2 的可核验通过率有结构性上限**;继续补引文不会让它们变绿。
> 下一步该做的是**口径设计**(要求 ROI 表显式标注「本表为作者估算」并从断言集排除),
> 而不是继续找引文。

**反向发现**:补引文过程暴露了 **13 类卡片内容缺陷 + 8 类仓库卫生问题**,全部**未改卡**
(按「标注不删」口径),登记在 `paper2skills-research/reports/PHASE4-发现的内容缺陷.md`。
最严重的一条:`Skill-Uplift-Churn-Prediction.md` **把论文的负结果写成了正结果**
(论文 §6 明说普通预测模型反而最好、uplift 不一定更优;卡片称 X-Learner 最优)。
另一条:`Skill-CSK-Customer-Sentiment-Clustering.md` 的 arXiv ID **指向另一篇论文**
(该卡已在 frontmatter 注释里记录真实出处,并**刻意不填 `paper_id`**)。

**基线变化(2026-09-12 PHASE3 3A–3E 前后)**

| 指标 | 3A/3B 后 | 3C/3D/3E 后 | 变化原因 |
|---|---|---|---|
| 卡片数 | 138 | **146** | +8(3C 3 张 / 3D 5 张);3E 是增强不新增 |
| K1 执行率 | 58.7%(92 单元) | **62.0%**(100 单元) | 新增 8 个代码单元全 PASS |
| ORPHAN_DEP | 9 | **0** | **判定修正**,非卡片改动(见下「已迁出镜像」一节) |
| G2 通过 | 8/138 | **16/146** | 新增 8 张全过 |
| 引文数 | 286 | **729** | 3C/3D 新增 349 条 + 3E 增强新增 94 条,全 VERBATIM |

> ⚠️ 存量 130 张的 G2 仍全红,根因未变:**引用块为 0**。这不是「新卡拉高了均值」,
> 而是「新卡是唯一有证据链的资产」。

### PHASE3 批次 3A 交付(4 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `13-广告分析/Skill-Cannibalization-Corrected-Attribution.md` | 2606.26690 | ADKDD 2026 (**workshop**) | PASS | 18/18 | ✅ | ✅ |
| `13-广告分析/Skill-Funnel-Causal-Coupon-Allocation.md` | 2608.11675 | CIKM 2026 | PASS | 28/28 | ✅ | ✅ |
| `13-广告分析/Skill-Causal-Budget-Allocation.md` | 2608.10182 | arXiv preprint | PASS | 38/38 | ✅ | ✅ |
| `06-增长模型/Skill-Seasonal-Aligned-Churn-Label.md` | 2608.18174 | arXiv preprint | PASS | 40/40 | ✅ | ✅ |

### PHASE3 批次 3B 交付(4 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `03-时间序列/Skill-Decision-Conditioned-Forecasting.md` | 2608.25871 | **KDD 2026** (CCF-A) | PASS | 47/47 | ✅ | ✅ |
| `03-时间序列/Skill-Shipping-Cost-Estimation.md` | 2607.16230 | arXiv preprint | PASS | 36/36 | ✅ | ✅ |
| `04-供应链/Skill-Supply-Network-Simulation.md` | 2607.09745 | WSC 2026 (CCF-B) | PASS | 38/38 | ✅ | ✅ |
| `04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md` | 2606.29366 | arXiv preprint | PASS | 41/41 | ✅ | ✅ |

### PHASE3 批次 3C 交付(3 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `14-用户分析/Skill-Incrementality-Measurement.md` | 2607.09608 | arXiv preprint | PASS | 31/31 | ✅ | ✅ |
| `00-电商Agent/Skill-Live-Catalog-Conversational-Rec.md` | 2608.27006 | RecSys'26 **Demo** | PASS | 37/37 | ✅ | ✅ |
| `00-电商Agent/Skill-Agentic-Catalog-Enrichment.md` | 2608.20844 | arXiv preprint | PASS | 45/45 | ✅ | ✅ |

### PHASE3 批次 3D 交付(5 张,三门前全绿)

| 卡片 | 论文 | venue | K1 | 引文 | G2 | G3 |
|------|------|-------|----|------|----|----|
| `16-智能体工程/Skill-Stateful-Skill-Runtime.md` | 2608.26263 | arXiv preprint | PASS | 50/50 | ✅ | ✅ |
| `10-MAS/Skill-Multi-Agent-Collaboration-Tax.md` | 2608.22152 | arXiv preprint | PASS | 55/55 | ✅ | ✅ |
| `10-MAS/Skill-Routed-Graph-Handoff.md` | 2608.25277 | arXiv preprint | PASS | 51/51 | ✅ | ✅ |
| `09-DataAgent-LLM/Skill-SQL-Agent-Access-Control.md` | 2607.22115 | arXiv preprint | PASS | 39/39 | ✅ | ✅ |
| `02-A_B实验/Skill-Persona-Based-AB-Simulation.md` | 2609.01038 | arXiv preprint | PASS | 41/41 | ✅ | ✅ |

### PHASE3 批次 3E 交付(增强既有卡,不新增)

| 论文 | 被增强的卡 | 引用块 | 说明 |
|------|-----------|--------|------|
| 2608.28978 | `08-知识图谱/Skill-GraphRAG-Knowledge-Enhanced-Retrieval.md` | 0 → **18/18** | 负结果:「反例与适用边界」 |
| 2608.28978 | `16-智能体工程/Skill-Agentic-Memory-Management.md` | 0 → **17/17** | 同上 |
| 2608.09162 | `12-ML基础/Skill-Feature-Engineering.md` | 0 → **30/30** | 数值特征变换形式化为优化问题（摘要称「一致优于所有基线」，但论文自己的 Table 1 里 `mlp`/`mlp_plr` 两行 PLE 优于 stretch —— 反例已写进 1b）|
| 2608.10240 | `05-推荐系统/Skill-Cold-Start-Meta-Learning-PAM.md` | 0 → **32/32** | 顺序模态丢弃 |

**这 16 张(3A–3D)也是全库唯一的 G2 通过者** —— 存量 130 张的 G2 全红,根因是**引用块为 0**。

### registry 被出卡过程反向修正的断言(4 处)

出卡要求逐项核对 registry 的 `decision_reason` / `note`，因此发现了 4 处 registry 与原文不符。
**这类错误比卡片错误更危险 —— registry 是「论文唯一事实源」，下游全部依赖它。**

| 论文 | registry 原写 | 逐项核对结论 |
|---|---|---|
| 2608.25871 | 「给定预算排期与**备货计划**的销量」 | ❌ 抬级：原文动作向量**只有折扣与广告花费两个变量**，不含补货/备货。已删「备货计划」 |
| 2608.25871 | `data_availability: available` | ❌ 修正为 `partial`：数据为阿里 1688 私有面板，论文仅称「正在推进发布部分匿名版本」 |
| 2606.29366 | 「支持自然语言约束（'优先保 FBA 不断货'）」 | ❌ 与原文措辞不符：全文 `FBA`/`cross-border`/`stockout`/`lead time` **零命中**；论文自己的例子是「Enforce minimum TID」 |
| 2607.09745 | 「落地成本最低」 | ⚠️ 只部分采信：建模侧成立，数据侧（批次效期台账、中断样本）仍需补齐 |

另修正一处 venue 判断：2606.29366 全文首页页眉写 `Journal: European Journal of Operational Research`，
但**无录用证据** → 仍记 `preprint`，并禁止当作 EJOR 论文引用。

### 已发现的缺陷(由 K1 首次暴露,已修复)

首轮报出「8 个语法错误」,**逐个人工核对后发现全部是 markdown 围栏结构问题,不是代码缺陷**——
这个区分很重要:用 LLM 阅读永远发现不了,只有执行才暴露。

| 缺陷类 | 数量 | 实例 | 修复 |
|--------|------|------|------|
| 伪代码被标成 ```python | 9 处 | `Skill-AB-Experimental-Design`(ASCII 框图)、`Skill-Memory-as-Action`(函数签名) | 改标 `pseudocode` / `yaml` |
| 代码块内含嵌套 ``` | 1 张 | `Skill-Argos-Agentic-Anomaly-Detection`(f-string 里嵌 markdown) | 外层改 **4 反引号**(Markdown 标准做法);此前该块被截成两半、后半段整个丢失 |
| 代码围栏**未闭合** | 2 张 | `Skill-Session-Based-Recommendation-SR-GNN`、`Skill-Knowledge-Graph-for-Skills-Management` | 在代码真实结束处补闭栏;此前 ④技能关联 等正文被当作 Python |
| 真代码笔误 | 1 处 | `self._ Aspects = None`(标识符中间多空格) | 改 `self._aspects` |

**~~仍待修(9 张)~~ —— 已于 2026-09-12 撤销,这 9 张不是卡片缺陷。**

原判定:9 张卡 `import` 了「仓库内**根本不存在**」的本地模块,判 `ORPHAN_DEP`,列入必须修清单。
**逐个人工 `ls` 核对后全部推翻** —— 这 9 个模块**都真实存在于仓库**:
`autotag_self_evolving`、`review_quality_scoring`、`nps_driver_analysis`、
`behavioral_intent_tree_parsing`、`crosslingual_semantic_alignment`、
`crosslingual_sentiment_transfer`、`dialogue_to_action_graph`、
`product_attribute_graph_parsing`、`voc_semantic_blueprint`
全部位于 `paper2skills-code/nlp_voc/<mod>/`(`07-NLP-VOC` 迁出后保留的代码模板镜像)。

**根因是门禁自己的一条 `continue`**:`repo_local_module_index()` 里
`if "nlp_voc" in p.parts: continue` —— 索引阶段就把镜像排除掉,于是分类器
只能把它归成 ORPHAN。判定文案写「模块不存在」,**而这个描述是假的**。

**为什么这个误判有代价**:`ORPHAN_DEP` 与「镜像不可在此运行」的**补救动作相反** ——
前者要改卡片,后者要改环境(或接受它不可验证)。混为一谈会把 9 张没有缺陷的卡
送进「必须修卡片」队列,而真正该做的是承认这批卡在本机无法验证。

**修法**:新增第四类判定 `MIGRATED_DEP`(🟠)。模块存在于 `paper2skills-code/nlp_voc/`
→ 判 MIGRATED,计入未验证分母但**不计入失败、也不算卡片缺陷**;
只有模块在仓库内**确实找不到**才判 `ORPHAN_DEP`。
`--selftest` 用四个用例锁定三类判定互斥可区分(样本名从索引现取,不写死 —— 写死会腐烂,
第一版就因写死 `causal_inference` 报了一次假失败)。

> **实测效果**:全量 L3 扫描 `ORPHAN_DEP 9 → 0`、`MIGRATED_DEP 0 → 9`,退出码从 1 变 0。
> 这是本项目第三次「提升来自修门禁而非改卡片」—— 与 46.2%→52.5%→54.8% 那次同源。
> 教训:**判某个东西「不存在」之前,先 `ls` 一次。**

### 门禁自身的 bug(全部由实测暴露,`--selftest` 逐条锁定)

| # | bug | 后果 | 锁定方式 |
|---|-----|------|----------|
| 1 | `looks_like_python` 启发式过宽(把含 `if`/`for` 的**文档**判为代码) | 3 个假阳性,报告出现不存在的「语法错误」 | — |
| 2 | 围栏正则不支持**变长反引号**(只认 3 个) | 嵌套块被截断;4 反引号块被整个漏掉(表现为"该卡无代码") | — |
| 3 | `extract_all_python` 回退分支误用 `m.group(1)` | 同上 | — |
| 4 | 索引阶段 `if "nlp_voc" in p.parts: continue` | **9 张卡被误判 `ORPHAN_DEP`** 并列入「必须修卡片」,而模块其实都在仓库里 | K1 `--selftest` 四用例 |
| 5 | **L3 探针不注册 `sys.modules`** | CPython 3.14 下凡「`from __future__ import annotations` + `@dataclass`」的模块崩在 `dataclasses._is_type`,被记成卡片的 `import 时崩溃` —— **影响全仓库所有 dataclass 卡** | K1 `--selftest` 端到端用例 |
| 6 | **`gate_check` 找不到 evidence.md** | 只在卡片同目录找,而约定是 `papers/<域>/<p2s-id>/evidence.md` → **PHASE3 全部新卡的 evidence.md 对 G2 毫无作用**,漏洞 #3 的加固成了空转 | 见下 |
| 7 | **`quote_check` 把「没有引用块」算作「通过」** | `NO_QUOTES` 的卡被并进「N 通过」汇总 —— 「没东西可查」≠「出处为真」,又一个假绿灯 | 汇总行现单列 |
| 8 | **`repo_health --json-out` 因父目录不存在而崩溃**(2026-09-12 盘点新发现) | 体检报告**全部正常打印完**、结论「✅ 无 CRITICAL」,最后一步写 JSON 时 `FileNotFoundError` → **退出码 1**。后果:① PHASE5 T5-4「周日体检 + 趋势追踪」拿不到可比较的历史文件;② 这个失败**看起来像体检失败**,而体检其实是过的 —— 与 #6 同类的「静默退回/假失败」 | **待修**(`json_out.parent.mkdir(parents=True, exist_ok=True)`) |
| 9 | **`repo_health` C2 不认 `author-practice` 口径**(同上) | C2 报「有 frontmatter 但缺 v2 必填字段 **126 张**」,其中 **48 张是 `evidence_basis: author-practice` 卡** —— 它们**按设计就不该有**来源字段(漏洞 #8/#10 的三类口径)。真实欠账:无 frontmatter **2 张**;96 张有来源卡中缺 `venue` 66 / `venue_tier` 74 / `evidence_grade` 78。**口径未跟上口径**,与漏洞 #11(参考区词汇表)同源 | **待修**;豁免后仍需用构造样本自证会抓真缺陷 |

**#5 的最小复现**(`from __future__ import annotations` 是触发条件 —— 它让所有注解变成字符串,
`dataclasses._is_type` 才会走 forward-ref 分支去查 `sys.modules[cls.__module__]`):
```python
spec = u.spec_from_file_location('unit', 'unit.py')
m = u.module_from_spec(spec)
spec.loader.exec_module(m)        # ❌ AttributeError: 'NoneType' object has no attribute '__dict__'
sys.modules['unit'] = m           # ✅ 补这一行即通过
```
三个子代理各自撞上并**各自改卡片绕开**(去掉 future import)。这说明:
**门禁缺陷会以「大家都学会绕路」的形式被消化掉,而不是被报告** —— 所以 `--selftest` 比文档更可靠。

**#6 的定位过程本身是个教训**:第一版修复按 registry 的 `paper_id` 匹配,全部落空且**静默退回旧行为**。
根因是本仓库一个命名坑:**卡片 frontmatter 的 `paper_id` 是 arXiv ID(如 `2608.25277`),
registry 的 `paper_id` 是项目内 ID(如 `p2s-2026-0018`)** —— 同名字段、不同语义。
现按 `registry.paper_id` / `identifiers.arxiv` / `outputs.skill_card` 三条线索依次认领。
> **修门禁时必须验证「修复真的生效」,而不是「门禁还绿着」。** 第一版修复后门禁照样绿,
> 因为它压根没找到文件 —— 绿得和没修一样。

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

---
title: paper2skills 萃取链路升级方案 v2.0
doc_type: design
module: 00-项目管理
status: draft
created: 2026-09-12
owner: self
source: human+ai
---

# paper2skills 萃取链路升级方案 v2.0

> 依据：2026-09-12 深度调研（arXiv 近 92 天 1046 篇 + Crossref 顶刊 1751 篇 + 130 张存量卡片体检 + 四个领域子评审 + 顶刊子评审）
> 目标：把「每天 2 小时人工挑论文」改造成「一条可复现、可验证、可去重的流水线」

---

## 一、为什么必须升级：现状的 7 个硬事实

| # | 事实 | 证据 | 后果 |
|---|------|------|------|
| 1 | **检索面只有 arXiv** | `paper-选题/SKILL.md` 只写 arXiv API + GitHub | UTD24/FT50 顶刊（Management Science / M&SOM / ISR / Marketing Science）在窗口内新增 1751 篇，其中 783 篇标题命中业务关键词，**完全不在视野内** |
| 2 | **路径硬编码已失效** | `paper-萃取/SKILL.md`、`paper-同步/SKILL.md`、`scripts/sync.py` 全指向 `/Users/pray/project/paper_to_skills`；实际仓库在 `/Users/lute/...` | 同步脚本直接跑不起来；新会话按卡片指引会写错目录 |
| 3 | **卡片存在 26 组重复** | `07-NLP-VOC/` 下同名卡片各存两份（25 组字节完全相同，1 组已漂移到不同内容） | 130 张唯一卡片 vs 156 个文件；改一张卡不会同步另一张，审核结果不可信 |
| 4 | **卡片规范不统一** | 156 个文件里只有 73 个有 frontmatter（46%）；只有 **6 张卡（3%）有 `paper:` 溯源字段** | 无法回答"这张卡来自哪篇论文、哪一版"；论文更新无法反向定位要改的卡 |
| 5 | **代码可执行性无证据** | 仅 95/156 张卡有 Python 代码块，69/156 有 `def/class`；`paper-萃取` 的"强制验证"靠人工自觉，无脚本、无产物留存 | "验证通过"无法复核，回归时无法重跑 |
| 6 | **选题无去重、无 venue 校验** | 本轮实测：子代理发现 6 处 venue 抬级（workshop 标成主会）、1 处标题与摘要矛盾、1 处"错误投稿"自述、1 处跨域重复 | 会把 workshop 短文当顶会成果萃取；同一方法重复出卡 |
| 7 | **选纸标准在 persona 里，不在代码里** | `paper-选题` 的评分表（算法创新 20% / 实验 30% / 落地 30% / 适配 20%）只写在 SKILL.md，靠 LLM 心算 | 每次评分不可复现，无法比较"这周 vs 上周"的候选质量 |

**一句话结论**：现有链路是"人 + Prompt"，缺的是"数据 + 脚本 + 门禁"。升级不推翻现有 5 模块卡片规范，而是给它补上**可复现的检索、可计算的评分、可自动化的去重与验证**。

---

## 二、目标形态：Paper→Skill 五段式流水线

```
┌─ S1 多源检索 ─┐  ┌─ S2 归一评分 ─┐  ┌─ S3 三层去重 ─┐  ┌─ S4 证据化萃取 ─┐  ┌─ S5 三重门禁 ─┐
│ arXiv API     │→ │ venue 层级    │→ │ L1 硬 ID      │→ │ 全文证据表      │→ │ 代码可执行     │
│ Crossref 顶刊 │  │ 业务相关性    │  │ L2 方法名     │  │ 5 模块卡片      │  │ 事实可溯源     │
│ 会议 proceedings│ │ 工程可得性    │  │ L3 语义簇     │  │ 代码 + 测试     │  │ 业务可落地     │
│ GitHub 高星   │  │ 时效 + 缺口   │  │ + 卡内去重    │  │ 溯源 frontmatter│  │ + 双人抽检     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └───────┬────────┘  └──────┬───────┘
       │                 │                 │                  │                  │
  papers_registry   shortlist.md      dedup_report.json   Skill-*.md        gate_report.json
  （唯一事实源）      （人读清单）        （拦截记录）        （交付物）          （门禁凭证）
```

配套三条**状态机**（写入 `papers_registry.json`，是唯一事实源）：

```
new → scored → shortlisted → decision:extract → extracting → extracted → reviewed → synced
                              ↘ decision:watch（观察）  ↘ decision:skip（排除，记原因）
```

---

## 三、S1 多源检索层

### 3.1 四条检索路线（按性价比排序）

| 路线 | 覆盖 | 调用方式 | 本轮实测结果 | 可靠性 |
|------|------|----------|--------------|--------|
| **A. arXiv API** | 预印本、方法论、模型类（占业务增量 60%） | `export.arxiv.org/api/query`，支持 `submittedDate:[start TO end]` | 49 条查询 → **1046 篇**去重候选（92 天） | ✅ 稳定，无限流问题 |
| **B. Crossref 期刊** | UTD24/FT50/CCF-A 期刊的**正式在线发表**（arXiv 完全没有的实证/理论类） | `api.crossref.org/journals/{issn}/works?filter=from-pub-date:..,until-pub-date:..` | 28 刊 → **1751 篇**，其中业务命中 783 篇 | ✅ 稳定；无需 key |
| **C. 会议 proceedings** | KDD/SIGIR/RecSys/CIKM 等已开会年份的正式论文集 | 先由 arXiv `comment`/`journal_ref` 字段捕获（本轮 152 篇带 venue 信号，其中 40 EMNLP / 25 RecSys / 23 KDD / 23 CIKM）；正式 proceedings 走 ACM DL / DBLP（本轮 DBLP 遇 bot 挑战，需浏览器或带 UA 限速重试） | comment 字段命中 152 篇 | ⚠️ 部分受限，作为补充 |
| **D. OpenReview** | ICLR/NeurIPS 投稿与接收列表 | `api2.openreview.net/notes?content.venue=...` | ❌ 本轮返回 **403 ChallengeRequiredError**（人机验证）；需浏览器会话或个人 token | ⚠️ 需人工/浏览器辅助 |

**落地约定**：
- A + B 做**全自动周更**（每周一跑一次，产出增量短名单）。
- C 做**会议季专用**（KDD 8 月、EMNLP 11 月、ICML 7 月、SIGIR 7 月、RecSys 9 月前后各跑一次）。
- D 在会议放榜周（如 NeurIPS 9/24 放榜）用浏览器人工导出一次，人工导入 registry。

### 3.2 关键词库要拆成"正向 + 负向 + 约束词"三段

本轮实测的检索污染率（这是必须修的第一优先级）：

| 领域 | 污染率 | 污染样本 | 修正 |
|------|--------|----------|------|
| 04-供应链 | 27%（6/22） | 软件供应链、SBOM、PyPI 恶意包、LLM 血缘、LLM 推理降级 | 负向词：`software supply chain`、`SBOM`、`bill of materials`、`package detection`、`model lineage`、`LLM inference`、`provenance` |
| 14-用户分析 | **64%（14/22）** | 医疗影像、EEG、放疗、PET、RICH 评测、LLM 推理预算 | 约束词：`+e-commerce/retail/subscription/churn/retention/customer journey`；负向词：`MRI/EEG/radiotherapy/clinical/patient/lesion/PET/diagnosis` |
| 07-VOC舆情 | 命中过窄（92 天仅 9 篇） | — | 拆词重跑：`product review summarization` / `opinion mining` / `multimodal review` / `aspect extraction` / `review-driven product improvement` 分开搜 |
| 11-AI人文 | **方向性错误（21/21 不是人文）** | `continual learning` 36 次、`LoRA` 24 次 | **该域停用 arXiv 萃取**（见 §6.4） |

### 3.3 交付脚本（本轮已落地，可直接复用）

| 脚本 | 作用 | 状态 |
|------|------|------|
| `paper2skills-research/scripts/arxiv_harvest.py` | arXiv 92 天窗口批量收割（49 查询组、去重、CSV+JSON） | ✅ 已跑通，产出 1046 篇 |
| `paper2skills-research/scripts/journal_harvest.py` | Crossref 28 刊窗口收割（ISSN+日期过滤、业务命中标注） | ✅ 已跑通，产出 1751 篇 |
| `paper2skills-research/scripts/make_bundles.py` | 按业务域切分候选 bundle（供 LLM 精读） | ✅ 已跑通，17 个 bundle |
| `paper2skills-research/scripts/make_journal_bundles.py` | 按决策节点切分期刊 bundle | ✅ 已跑通，4 个 bundle |

---

## 四、S2 归一评分层

### 4.1 统一评分模型（把 persona 里的标准搬进代码）

`score = A(venue 0-30) + B(code 0-15) + C(business 0-20) + D(method 0-20) + E(freshness 0-8) + F(gap 0-7)`

| 维度 | 取值规则 | 为什么这么定 |
|------|----------|--------------|
| **A venue 层级** | 顶会/顶刊主会 **30**；workshop/findings/demo **18**；二线会议/期刊 **10**；arXiv only **0**；命中 `Workshop/Findings/Demonstrations/Under review/manuscript` 一律降级 | 本轮实测 heuristic 会把 workshop 抬成主会（6 例），必须**规则降级 + 原文复核**双保险 |
| **B 工程可得性** | 摘要/comment 有 GitHub/HF 链接 **+11**；有"open-source/release/repository"表述 **+4** | 有代码的论文落地时间是自研的 1/3 |
| **C 业务相关性** | 强业务词（e-commerce/ads/bidding/pricing/inventory/replenishment/churn/uplift…）每个 **+3.5**（上限 14）；弱业务词每个 **+1.2**（上限 6）；命中母婴品类锚点（baby/infant/breast pump/diaper…）**+4** | 用真实标题验证过：本轮把 CHAP/SMD/FunnelCausalNet 等排到前列，也能压掉纯理论 |
| **D 方法可萃取性** | "we propose/framework/algorithm/experiments/benchmark/ablation"每个 **+3**；命中 survey/review/position paper/roadmap 每个 **−8**，标题以 A Survey 开头直接封顶 4 分 | 直接实现"排除综述、排除纯理论"的既有标准 |
| **E 时效** | `8 − 天数/12` | 让新论文有优势但不过度 |
| **F 知识缺口** | 命中项目空白方向（见 §4.2）每个 **+2.5**（上限 7） | 让选题服务于"补齐能力地图"，而非随机捡漏 |

**分层阈值（本轮实测分布）**：≥70 分 4 篇 / ≥60 分 19 篇 / ≥50 分 59 篇 / ≥40 分 133 篇 / ≥30 分 290 篇。
→ 建议**每周只处理 ≥50 分的 10-20 篇**，其余进观察池。

### 4.2 知识缺口表（F 维度来源，需按季度刷新）

| 缺口 | 判定关键词 | 现状 |
|------|-----------|------|
| 广告归因/增量 | attribution / incrementality / geo experiment / MMM | 有 2 张卡，但缺"归因 vs 增量校正" |
| 实验平台/序贯检验 | sequential test / switchback / CUPED / variance reduction | 有 3 张卡，缺工程化实验平台 |
| 选品/组合优化 | assortment / product selection / category management | **空白** |
| 供应链-LLM/OR | supply chain + LLM / MIP formulation / replenishment policy | 空白（本轮 3 篇高分候选） |
| 推荐-生成式 | generative recommendation / semantic id / LLM ranker | 空白 |
| Agent 技能/上下文 | agent skill / skill library / context compression | 已有 16 张，需"状态化 vs 压缩派"选型 |
| 自动化评测 | LLM-as-judge / rubric / evaluation harness | 部分覆盖 |
| 跨境合规/多语言 | multilingual / compliance / customs / CPSC | **空白（重大缺口）** |
| 表格基础模型 | tabular foundation model / in-context learning | **空白（本轮最大红利）** |

### 4.3 交付物

- `data/shortlist.md`：按领域分组的 Top 10 人读清单（本轮已生成）
- `paper2skills-vault/07-资源库/venue-whitelist.md`：venue 层级白名单（**待新建**）
- `paper2skills-vault/07-资源库/关键词库-v2.md`：三段式关键词（正向/负向/约束，**待新建**）

---

## 五、S3 三层去重层

| 层 | 规则 | 阻断动作 | 本轮实测 |
|----|------|----------|----------|
| **L1 硬 ID** | 候选 arXiv ID / DOI 是否已出现在任何卡片的 `paper:` 字段或正文引用 | 直接标 `dup`，不进入萃取 | 0 命中（说明新候选与存量无同篇） |
| **L2 方法名近重** | 候选标题 token ∩ 卡片名 token ≥ 3，或方法名（去掉 Skill- 前缀后）编辑距离 ≤ 2 | 标 `near-dup`，需人工选择"增强已有卡"或"新建" | 本轮 0 组（存量卡片名多为方法专属词） |
| **L3 语义簇** | 候选摘要 embedding 与卡片正文 embedding 相似度 > 阈值（建议 0.82），或落在同一主题簇（如 6 篇 GraphRAG 变体、15 篇 LoRA 变体） | 进入"合并候选池"，由 LLM 判定是"增强/替换/新建" | 子代理已就 GraphRAG 与 LoRA 两个簇给出结论 |

**同时必须做的仓库级去重**（本轮发现的 26 组重复卡片）：
- 处置：保留 `00-知识库-Skill卡片/` 作为唯一存放地（与 10-MAS 现有结构一致），删除顶层副本；`Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md` 的两份已漂移，需人工比对后合并。
- 门禁：新增 `check_duplicate_cards.py`，CI/周检时若出现同名卡片直接失败。

**交付脚本**：`paper2skills-research/scripts/dedup_check.py`（✅ 已落地，L1/L2 已跑通；L3 待接 embedding）

---

## 六、S4 证据化萃取层

### 6.1 卡片 frontmatter 升级（不破坏既有格式，全部为新增字段）

```yaml
---
title: Skill-XXX — 一句话中文名
doc_type: knowledge
module: 13-广告分析
topic: cannibalization-corrected-attribution
status: stable              # draft | verified | stable | deprecated
created: 2026-09-12
updated: 2026-09-12
owner: self
source: human+ai

# —— 以下为 v2 新增溯源字段 ——
paper_id: p2s-2026-0001                 # registry 主键
paper: arXiv:2606.26690                 # 人可读 ID（arXiv / DOI）
paper_title: "Attributed, But Not Incremental: Cannibalization-Corrected Attribution"
venue: ADKDD 2026                       # 规范化的 venue（workshop 必须标注）
venue_tier: workshop                    # top / second / workshop / preprint
paper_date: 2026-06-25
code_available: false                   # 是否有官方开源
code_self_built: true                   # 本项目是否自研实现
evidence_grade: A                       # A=摘要+全文核对数字  B=仅摘要  C=仅标题推测
verified_by: scripts/verify_skill_code.py@2026-09-12
supersedes: []                          # 被本卡替换的旧卡
related: [Skill-Ad-Attribution-Modeling]
---
```

### 6.2 证据规则（治"数字幻觉"）

本轮实测的教训：**150 篇 bundle 的摘要被截断（900/部分 500 字符），把截断摘要里的半句话当结论会直接污染卡片**。抽查已发现：
- `2607.27546` 摘要 "memory reductions of up to 7.6" **缺单位**（应为 ×7.6）
- `2608.09162` 摘要以 "Our framework" 半句截断
- `2606.20961` 摘要残留 `\textbf{}` LaTeX 标签
- `2607.29459` **标题（TFGformer）与摘要（CrossRAG）内容不一致**

因此明确定为规则：

| 规则 | 要求 |
|------|------|
| R1 | 卡片里出现的每个**数字**（准确率、收益 %、成本降幅）必须来自**未截断原文**，并在卡片中以 `> 原文："..." (Section X)` 形式给出出处 |
| R2 | 摘要不可得或截断时，`evidence_grade` 降为 B 或 C，卡片**不得**给具体数字，只写"论文报告了正向收益，具体数值需回原文" |
| R3 | venue 必须按白名单规范化；workshop/findings/demo 一律显式标注，禁止写成主会 |
| R4 | 标题与摘要矛盾、自述"错误投稿"、非论文（书籍草稿）的条目，**硬拦截**，不进入萃取 |
| R5 | 数据可得性必须显式声明：卡片新增一行 `数据要求：企业内是否可得（是/否/需替换语料）`；不可得则降级为 P2 或只萃取"决策结构" |

### 6.3 代码验证闭环（把"强制"变成脚本）

现状问题：`paper-萃取/SKILL.md` 说"未通过验证不得保存"，但没有脚本、没有产物，全靠自觉。

补上三个脚本（**待实现**）：

| 脚本 | 职责 | 门禁条件 |
|------|------|----------|
| `extract_code_blocks.py` | 从 Skill 卡片抽出全部 ```python 块，写入 `paper2skills-code/<domain>/<algo>/model.py` + `test_model.py` | 至少 1 个可 import 的模块 |
| `verify_skill_code.py` | `python -m py_compile` → 有测试则 `pytest -q` → 无测试则执行 `python model.py`；记录 stdout/耗时/退出码 | **退出码必须为 0**；失败则卡片状态置 `draft`，禁止同步 |
| `sync_ledger.py` | 把验证结果写回卡片 frontmatter 的 `verified_by` 与 `evidence_grade`，并更新 `sync_status.json` | 卡片与 ledger 不一致即失败 |

产物：`paper2skills-vault/papers/<domain>/<paper-id>/verification_report.md`（现状模板保留，但由脚本生成）。

### 6.4 分域策略调整（本轮调研得出的重要结论）

| 领域 | 结论 | 动作 |
|------|------|------|
| **12-ML基础** | 不是空白，是"漏了"。92 天内 20/22 篇摘要含 "tabular foundation" → 表格基础模型（TFM）对本项目是直接红利（选品打分表、广告报表、库存/退货表都是中小表） | **优先补 3 张卡**：TFM 开箱基线 / TFM 蒸馏上线 / TFM 特征与合规边界 |
| **11-AI人文** | 21/21 命中都不是人文（是 LoRA/持续学习）。该域的交付瓶颈是**文案质量**，不是论文供给 | **停用 arXiv 萃取**，改"提示词工程 + 人工策展"流水线；有业务价值的 4 篇改挂 12-ML基础/07 |
| **06-增长模型** | 92 天仅 5 篇命中（其他域 59-84）→ 供给荒漠 | 不指望 arXiv 增量；把力气放在 01/02/13/14/15 的交叉复用 |
| **14-用户分析** | 64% 污染；存量"漏斗 + cohort"能力未被超越 | 关键词加约束，本轮不新增卡片 |
| **00-电商Agent** | 买家侧购物 Agent 很多，但真正的**跨境**工作只有 1 篇；关税/合规/认证/IP/汇率/逆向物流是**学术真空** | 只借鉴评测环境与协议；跨境合规方向自建 |
| **07-VOC舆情** | 命中过窄（9 篇） | 拆词重跑关键词 |

---

## 七、S5 三重门禁层

萃取完成不等于可同步，必须过三道门禁（每道都由脚本产出 JSON 凭证）：

| 门禁 | 检查项 | 通过标准 | 凭证 |
|------|--------|----------|------|
| **G1 代码可执行** | `py_compile` + 运行 + 测试 | 退出码 0，无占位符，有 I/O 定义 | `gate_code.json` |
| **G2 事实可溯源** | 每个数字有原文出处；venue 合规；`evidence_grade` ≥ B | 无未标注数字、无 venue 抬级 | `gate_evidence.json` |
| **G3 业务可落地** | 应用案例含：业务问题 + 数据要求 + 预期产出 + 量化业务价值；数据可得性已声明 | 四要素齐全且数据可得性非"否" | `gate_business.json` |

三道全绿 → 卡片 `status: stable`，进入 `paper-同步`。任一红色 → 卡片 `status: draft` + 写入 `paper2skills-vault/07-资源库/审核问题库.md`。

**抽样复核**：每批次（建议 5 张卡）由人抽检 1 张，核对 G2 的原文出处是否真实（本轮调研已证明 LLM 会把截断摘要的残句当结论）。

---

## 八、与现有 5 个 Skill 的关系（改动清单）

| 现有 Skill | 改动 | 具体动作 |
|-----------|------|----------|
| `paper-选题` | **重写为 S1+S2+S3** | ① 加 Crossref 路线；② 加三段式关键词库引用；③ 加 venue 白名单校验与降级规则；④ 评分改为脚本可复现；⑤ 输出 `papers_registry.json` 记录 |
| `paper-萃取` | **增强为 S4** | ① 加 frontmatter v2 溯源字段模板；② 加 R1-R5 证据规则；③ 修正失效路径（`/Users/pray` → 仓库相对路径）；④ 验证步骤改为调用脚本 |
| `paper-审核` | **增强为 S5** | ① 三道门禁脚本化；② 评分表与 S2 评分口径对齐（避免"选题 7 分 ≠ 审核 7 分"）；③ 加"数字溯源"专项检查 |
| `paper-同步` | **修路径 + 加门禁前置** | ① `sync.py` 的 `BASE_DIR` 改为脚本自身定位（`Path(__file__).parents[3]`）；② 同步前校验 `gate_*.json` 全绿 |
| `paper-skills-graph` | **接缺口表** | ① 把 §4.2 缺口表纳入图谱输出；② 增加"重复簇"检测（26 组重复应当由它发现） |
| **新增** `paper-维护` | 新建 | 仓库体检：重复卡片、frontmatter 缺失、路径失效、ledger 不一致；每周跑一次 |

---

## 九、目录结构与唯一事实源

```
paper2skills-vault/papers/<领域>/<paper_id>/
├── paper.pdf                # 原文（可选，但 evidence_grade A 必须有）
├── notes.md                 # 阅读笔记（算法清单/公式/实验设置）
├── evidence.md              # 【新】数字→原文出处的证据表（R1 的载体）
├── extract.md               # 萃取过程记录
├── verification_report.md   # 【改】由脚本生成
└── gate_{code,evidence,business}.json   # 【新】门禁凭证

paper2skills-vault/07-资源库/
├── papers_registry.json     # 【新】唯一事实源（状态机 + 去重键 + 交付物回指）
├── venue-whitelist.md       # 【新】venue 层级白名单
├── 关键词库-v2.md            # 【改】正向/负向/约束三段式
├── MasterPrompt.md          # 【改】加 R1-R5 证据规则与 frontmatter v2 模板
├── sync_status.json         # 【改】加 gate 状态
└── 审核问题库.md             # 保留
```

`papers_registry.json` 单条记录示例：

```json
{
  "paper_id": "p2s-2026-0001",
  "identifiers": { "arxiv": "2606.26690", "doi": null },
  "title": "Attributed, But Not Incremental: Cannibalization-Corrected Attribution",
  "url": "https://arxiv.org/abs/2606.26690",
  "published": "2026-06-25",
  "venue": "ADKDD 2026",
  "venue_tier": "workshop",
  "domain": "13-广告分析",
  "score": 74.5,
  "score_breakdown": { "venue": 18, "code": 0, "business": 18.5, "method": 18, "fresh": 7.9, "gap": 5 },
  "decision": "extract",
  "decision_reason": "增量校正直击跨境站内站外归因断裂；需自研代码",
  "data_availability": "available",
  "dedup": { "l1": null, "l2": [], "l3_cluster": null },
  "outputs": { "skill_card": "paper2skills-vault/13-广告分析/Skill-Cannibalization-Corrected-Attribution.md",
               "code_dir": "paper2skills-code/advertising/cannibalization_corrected_attribution",
               "evidence": "paper2skills-vault/papers/13-广告分析/p2s-2026-0001/evidence.md" },
  "gates": { "code": "pending", "evidence": "pending", "business": "pending" },
  "status": "shortlisted"
}
```

---

## 十、节奏与验收

| 周期 | 动作 | 产出 | 验收 |
|------|------|------|------|
| 每周一（30 min 自动） | S1 跑 arXiv + Crossref 增量 → S2 评分 → S3 去重 | `shortlist.md`（新增候选 ≥50 分） | 短名单 ≤20 条，每条有分数与推荐理由 |
| 每周二（人工 30 min） | 人从短名单勾选 P0/P1 → 写入 registry `decision` | registry 更新 | 至少 1 条 `extract`，其余明确 `watch`/`skip` 并写原因 |
| 周三至周五（AI 主导） | S4 萃取 + S5 门禁 | 1-3 张卡片 + 代码 + 凭证 | G1/G2/G3 全绿；抽检 1 张数字可溯源 |
| 每周日（10 min） | `paper-维护` 体检 | 体检报告 | 重复卡片 0、frontmatter 缺失 0、路径失效 0 |
| 每季度 | 刷新缺口表、venue 白名单、关键词库；评估卡片使用率 | v2 文档更新 | 缺口表至少新增/关闭 2 项 |

---

## 十一、风险与对策

| 风险 | 本轮证据 | 对策 |
|------|----------|------|
| **数字幻觉**（把截断摘要当结论） | 150 篇 bundle 摘要被截断；4 篇有明显数据缺陷 | R1/R2 证据规则 + `evidence.md` + 抽检 |
| **venue 抬级** | 6 处 workshop 被标主会；1 处投稿被标接收 | 白名单 + 规则降级 + 原文复核 |
| **重复出卡** | 存量已有 26 组重复；本轮 GraphRAG 6 篇同族、LoRA 15 篇同族 | L1/L2/L3 三层去重 + 簇级合并决策 |
| **数据不可得** | 多篇需 RCT / 受众级随机化 / 完整多阶网络 / 社交图谱 | R5 显式声明 + 数据可得性访谈前置 |
| **OpenReview/DBLP 取数受限** | 403 challenge / 429 bot 挑战 | 会议季改浏览器人工导出；不依赖它们做主线 |
| **卡片腐化** | 1 组重复卡片已漂移（6 字节差异） | 唯一存放地 + 重复检测门禁 |

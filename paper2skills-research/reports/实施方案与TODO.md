---
title: paper2skills 萃取链路升级 —— 实施方案与 TODO
doc_type: plan
module: 00-项目管理
status: draft
created: 2026-09-12
owner: self
source: human+ai
---

# paper2skills 萃取链路升级 —— 实施方案与 TODO

> 配套文档：`paper2skills-research/reports/paper2skills-萃取链路升级方案-v2.md`（设计）、
> `paper2skills-research/reports/近三个月论文更新与推荐清单.md`（本轮论文结论）、
> `paper2skills-vault/07-资源库/papers_registry.json`（唯一事实源，45 条决策记录）

## 0. 总览

| 阶段 | 目标 | 交付物 | 预计工作量 |
|------|------|--------|-----------|
| **PHASE 0** | 止血：修失效路径 + 清重复卡片 | 仓库可用 | 1 小时 |
| **PHASE 1** | 补齐检索基建：多源检索 + 三段式关键词 + venue 白名单 | 4 个脚本 + 2 份资源库文档 | 半天 |
| **PHASE 2** | 萃取流水线化：registry + 证据规则 + 双门禁脚本 | 3 个脚本 + MasterPrompt v2 + 卡片模板 v2 | 1 天 |
| **PHASE 3** | 首批萃取：P0 队列 19 篇 → 出 8-10 张卡 | 卡片 + 代码 + 验证报告 | 分批，每张卡 1-2 小时 |
| **PHASE 4** | 空白领域补卡：12-ML基础 / 17-跨境合规 | 3+3 张卡 | 1 天 |
| **PHASE 5** | 常态化：周更 + 周体检 + 季度复盘 | 例行流程 | 每周 1 小时 |

**验收总标准**：连续两周的周更短名单 ≤20 条且有分数与理由；每张新卡 G1/G2/G3 全绿；仓库周检重复卡片 0、frontmatter 缺失 0、路径失效 0。

---

## PHASE 0 · 止血（1 小时）

- [ ] **T0-1 修正失效的绝对路径** ✅ 本轮已完成
  - `paper-同步/scripts/sync.py` 的 `BASE_DIR` 改为按脚本位置反推 + 支持 `PAPER2SKILLS_ROOT` 环境变量；DOMAINS 映射从 6 个补到 18 个
  - 6 个 SKILL.md / 文档里的 `/Users/pray/project/paper_to_skills` → `<REPO_ROOT>`
  - `paper-skills-graph/scripts/skills_graph_analyzer.py` 默认路径改为相对定位
  - 验收：`python3 paper2skills-skills/paper-同步/scripts/sync.py --status` 能正常列出同步状态 ✅

- [ ] **T0-2 清理 26 组重复 Skill 卡片**
  - 事实：`07-NLP-VOC/` 下 26 组同名卡片各存两份（25 组字节相同，`Skill-VOC-Proxy-NPS-AIPL-统一萃取引擎.md` 两份已漂移 6 字节）
  - 动作：① 人工比对漂移的那一份，决定保留版本；② 以 `00-知识库-Skill卡片/` 为唯一存放地（与 10-MAS 结构一致），删除顶层副本；③ 删除前 `git`/备份
  - 验收：`find paper2skills-vault -name 'Skill-*.md' | sed 's#.*/##' | sort | uniq -d` 输出为空

- [ ] **T0-3 建立"路径约定"并写进 `paper-维护` 的检查范围**
  - 事实：本轮清理了 `paper2skills-skills/` 下 22 处失效的 `/Users/pray/project/paper_to_skills` 硬编码
    （6 个 SKILL.md/文档 + `sync.py` + `skills_graph_analyzer.py`），仓库内仍存在的 `/Users/pray` 全部是
    **有意保留**的两类：① `paper2skills-code/nlp_voc/**` 与迁移说明（指向已迁出的 `../ai_nlp_voc/`）；
    ② `logs/` 与 `evolve/round-1/` 历史归档（不可改写）与 `01-MasterPrompt设计/` 草稿。
  - 约定：**可执行代码中的路径必须相对化**（`Path(__file__).resolve().parents[N]` 或 `PAPER2SKILLS_ROOT` 环境变量）；
    文档用 `<REPO_ROOT>` 占位；历史归档不改写，只加时效注记。
  - 验收：`paper-维护` 的路径检查只对 `paper2skills-skills/**` 与 `paper2skills-code/**`（排除 `nlp_voc/`）报警

- [ ] **T0-4 新建 `paper2skills-research/` 的定位说明**
  - 说明该目录是"调研与流水线工程"工作区（脚本 + 中间数据 + 报告），不属于交付的 vault/code
  - 在 `CLAUDE.md` 的项目结构中补一行（✅ 本轮已完成）

---

## PHASE 1 · 检索基建（半天）

- [ ] **T1-1 沉淀 arXiv 收割脚本**（✅ 已完成，需迁移到正式位置）
  - 现状：`paper2skills-research/scripts/arxiv_harvest.py`（49 查询组，本轮跑出 1046 篇）
  - 动作：复制到 `paper2skills-skills/paper-选题/scripts/arxiv_harvest.py`

- [ ] **T1-2 沉淀 Crossref 期刊收割脚本**（✅ 已完成）
  - 现状：`paper2skills-research/scripts/journal_harvest.py`（28 刊，本轮跑出 1751 篇）
  - 验收：`--days 7` 能在 1 分钟内跑完增量

- [ ] **T1-3 新建 `paper2skills-vault/07-资源库/venue-whitelist.md`**
  - 内容：顶会/顶刊白名单（UTD24 / FT50 / CCF-A/B / 领域顶会）+ workshop/findings/demo 降级规则
  - 必须收录本轮实测发现的 6 处抬级案例作为反例
  - 验收：给出"arXiv comment 含 Workshop/Findings/Demonstrations/Under review/manuscript → 降级"的判定表

- [ ] **T1-4 新建 `paper2skills-vault/07-资源库/关键词库-v2.md`（三段式）**
  - 正向词：现有 16 域的检索词（从 `关键词库.md` 继承）
  - **负向词**（本轮实测必需）：`software supply chain`、`SBOM`、`package detection`、`model lineage`、`LLM inference`、`MRI`、`EEG`、`radiotherapy`、`clinical`、`lesion`、`PET`（04 域污染率 27%、14 域 64%）
  - **约束词**：14 域加 `+e-commerce/retail/subscription/churn/retention`；04 域加 `+inventory/fulfillment/warehouse`
  - 拆词：07-VOC 域把 `product review summarization` / `opinion mining` / `multimodal review` / `aspect extraction` 分成 4 条独立查询（本轮 92 天仅命中 9 篇 = 过窄）

- [ ] **T1-5 评分器落地为脚本**（✅ 已完成，需迁移 + 参数化）
  - 现状：`rank_candidates.py`（6 维评分，本轮跑出 19 篇 ≥60 分）
  - 动作：把阈值、权重、缺口关键词表抽到 `paper2skills-vault/07-资源库/scoring_config.json`

- [ ] **T1-6 建立 `papers_registry.json` 读写工具**
  - 现状：`build_registry.py`（初版已生成 45 条）
  - 新增：`registry.py` 提供 `add/update/query/decide` 四个子命令，所有状态变更走它，避免手改 JSON

---

## PHASE 2 · 萃取流水线化（1 天）

- [ ] **T2-1 升级 `MasterPrompt.md` 到 v2**
  - 新增 R1-R5 证据规则（数字必须来自未截断原文并给出处；摘要不可得则不给数字；venue 规范化；硬拦截条目；数据可得性声明）
  - 新增 frontmatter v2 模板（`paper_id/paper/venue/venue_tier/evidence_grade/verified_by/supersedes/related`）
  - 新增"数据要求：企业内是否可得"必填行
  - 验收：用新模板重跑 1 张已有卡，确认字段齐全

- [ ] **T2-2 升级 `paper-萃取/SKILL.md`**
  - 修正路径（✅ 本轮已改为 `<REPO_ROOT>`）
  - 把 Step 5"代码验证"从人工描述改为**调用脚本**：`python3 scripts/verify_skill_code.py <card>`
  - 加 Step 4.5：生成 `evidence.md`（每个数字的原文出处）

- [ ] **T2-3 新建 `paper2skills-skills/paper-萃取/scripts/extract_code_blocks.py`**
  - 从卡片抽 ```python 块 → `paper2skills-code/<domain>/<algo>/model.py` + `test_model.py`
  - 验收：对 3 张已有卡试跑，产出可 import 的文件

- [ ] **T2-4 新建 `paper2skills-skills/paper-萃取/scripts/verify_skill_code.py`**
  - 流程：`py_compile` → 有 pytest 则 `pytest -q` → 否则 `python model.py`；记录退出码/耗时/stdout 到 `verification_report.md`
  - 门禁：退出码非 0 → 卡片 `status: draft`，写 `审核问题库.md`，**禁止同步**
  - 验收：故意塞一个语法错误的卡片，确认被拦

- [ ] **T2-5 新建 `paper2skills-skills/paper-审核/scripts/gate_check.py`**
  - G1 代码可执行 / G2 事实可溯源 / G3 业务可落地，各产出 `gate_*.json`
  - G2 的实现：正则抽取卡片中所有数字 → 要求在 `evidence.md` 或卡片的 `> 原文："..."` 引用块中有对应出处
  - 验收：对已有卡跑一遍，报告"无出处数字"计数（预期：大量红色，这正是要暴露的问题）

- [ ] **T2-6 `paper-同步/scripts/sync.py` 加门禁前置**
  - 同步前校验三份 `gate_*.json` 全绿，否则拒绝
  - 验收：对一张未过门禁的卡执行同步 → 被拒绝并给出原因

- [ ] **T2-7 新建 `paper-维护` skill（仓库体检）**
  - 检查：重复卡片、frontmatter 缺失、路径失效（`/Users/pray` 类）、registry 与卡片不一致、ledger 过期
  - 频率：每周；输出体检报告
  - 验收：首次运行能报出 T0-2 的重复与 T0-1 的路径问题（回归测试）

---

## PHASE 3 · 首批萃取（P0 队列，分批）

> 每张卡交付：Skill 卡片 + 代码 + evidence.md + 三份 gate JSON。数据可得性已在 registry 标注。

**批次 3A · 广告与增长（业务杠杆最高，4 张）**
- [ ] T3-1 `2606.26690` 归因蚕食校正 → `13-广告分析/Skill-Cannibalization-Corrected-Attribution.md`
- [ ] T3-2 `2608.11675` FunnelCausalNet 多档券 uplift → `13-广告分析/Skill-Funnel-Causal-Coupon-Allocation.md`
- [ ] T3-3 `2608.10182` 因果约束下的预算分配 → `13-广告分析/Skill-Causal-Budget-Allocation.md`
- [ ] T3-4 `2608.18174` 季节性流失误报修正 → `06-增长模型/Skill-Seasonal-Aligned-Churn-Label.md`（难度⭐1，可先做）

**批次 3B · 预测与库存（4 张）**
- [ ] T3-5 `2608.25871` CEDAR 决策条件化需求预测 → `03-时间序列/Skill-Decision-Conditioned-Forecasting.md`
- [ ] T3-6 `2607.16230` RouteCost 运费成本预估 → `03-时间序列/Skill-Shipping-Cost-Estimation.md`
- [ ] T3-7 `2607.09745` SupplyNetPy 多级供应链仿真 → `04-供应链/Skill-Supply-Network-Simulation.md`（**有开源库，最省力**）
- [ ] T3-8 `2606.29366` ORLA 多仓库存分配（与 `2607.25956` 合并）→ `04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md`

**批次 3C · 用户与电商（3 张）**
- [ ] T3-9 `2607.09608` 增量测量（站外种草→站内成交）→ `14-用户分析/Skill-Incrementality-Measurement.md`
- [ ] T3-10 `2608.27006` 活跃目录增量购物助手 → `00-电商Agent/Skill-Live-Catalog-Conversational-Rec.md`
- [ ] T3-11 `2608.20844` TRACE 目录属性补全 → `00-电商Agent/Skill-Agentic-Catalog-Enrichment.md`

**批次 3D · Agent 工程与实验（5 张）**
- [ ] T3-12 `2608.26263` SKILL.state → `16-智能体工程/Skill-Stateful-Skill-Runtime.md`
- [ ] T3-13 `2608.22152` Collaboration Tax → `10-MAS/Skill-Multi-Agent-Collaboration-Tax.md`
- [ ] T3-14 `2608.25277` Routed Graph Handoff → `10-MAS/Skill-Routed-Graph-Handoff.md`
- [ ] T3-15 `2607.22115` RBAC Text-to-SQL 门禁 → `09-DataAgent-LLM/Skill-SQL-Agent-Access-Control.md`
- [ ] T3-16 `2609.01038` 人格条件 A/B 仿真 → `02-A_B实验/Skill-Persona-Based-AB-Simulation.md`

**批次 3E · 增强而非新建（避免重复出卡）**
- [ ] T3-17 `2608.28978` 负结果 → 在 `Skill-GraphRAG-Knowledge-Enhanced-Retrieval` 与 `Skill-Agentic-Memory-Management` 各加"反例与适用边界"小节
- [ ] T3-18 `2608.09162` 数值特征变换 → 增强 `Skill-Feature-Engineering`
- [ ] T3-19 `2608.10240` 顺序模态丢弃 → 增强推荐域卡片（代码仅四行，收益明确）

---

## PHASE 4 · 空白领域补卡（1 天）

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

- [ ] **T5-1 周一自动化（30 min，脚本）**：arXiv + Crossref 增量 → 评分 → 去重 → 更新 registry → 生成 `shortlist.md`
- [ ] **T5-2 周二人工（30 min）**：从短名单勾 P0/P1，写入 registry `decision` + 理由
- [ ] **T5-3 周三至周五（AI 主导）**：萃取 1-3 张卡 → G1/G2/G3 → 抽检 1 张的数字出处
- [ ] **T5-4 周日（10 min）**：`paper-维护` 体检
- [ ] **T5-5 每季度**：刷新缺口表 / venue 白名单 / 关键词库；统计卡片使用率（哪些卡被真实项目引用过），淘汰零使用卡片

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

| # | 风险/待澄清 | 影响的 TODO | 需谁决策 |
|---|-------------|-------------|----------|
| 1 | 投放主体是**自有独立站**还是**第三方平台店**？ | T3-9 的落地形态（前者可做受众级随机化，后者降级为诊断框架） | 业务方 |
| 2 | 出海历史是否 ≥2 个完整年度（月度活跃面板）？ | T3-4 是否需要品类季节性先验替代 | 业务方 |
| 3 | 是否有 RCT / holdback 流量？ | T3-2、T3-3 的理论前提 | 业务方 |
| 4 | OpenReview / DBLP 取数受限（403 / bot 挑战） | 会议季路线 C/D 只能人工导出 | 无需决策，接受限制 |
| 5 | 数字幻觉（本轮已实证：4 篇摘要含缺陷、150 篇源码被截断） | G2 门禁 + 抽检是**必需**而非可选 | 执行纪律 |

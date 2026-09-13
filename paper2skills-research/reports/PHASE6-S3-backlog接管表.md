# PHASE6 S3 · extract backlog 靶区接管表（**由脚本生成，勿手改**）

> 生成时间：2026-09-13T20:30:46+0800　|　生成器：`paper2skills-research/scripts/route_backlog.py`
> 复算：`python3 paper2skills-research/scripts/route_backlog.py --check`
> 输入：`papers_registry.json` sha256:`34311e1e343e479b` · `gap-ledger.json` sha256:`06fdb22a11784f3a` · `backlog-l3-map.json` sha256:`368fedc17a08d13b` · `capability-graph.json` sha256:`be36dd1d197def0e`

**backlog 是现算的，不是抄的。** 口径：registry 里 `decision=extract` ∧ 无已交付产物。实测 **12 篇**（naive 只数 `skill_card` 得 15 篇，扣掉 3 篇增强交付才对齐 —— 即 PHASE5 登记的 N4）。

## 1. 逐篇定位与它服务的 L3 责任

| # | paper_id | 技术域 | 标题 | 服务的 L3（岗位）| 逐字证据 | 依据字段 | 证据类别 |
|---:|---|---|---|---|---|---|---|
| 1 | `p2s-2026-0015` | 12-ML基础 | Mitra-v2 Technical Report | 趋势监测（AGT-007）、算法评估设计（AGT-012） | 「选品打分表」、「免调参基线」 | decision_reason | business_action |
| 2 | `p2s-2026-0007` | 05-推荐系统 | Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy | 复购实验（AGT-034）、生命周期触达（AGT-034） | 「耗材复购召回」、「个性化补货节奏」 | decision_reason | business_action |
| 3 | `p2s-2026-0024` | 14-用户分析 | Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation | 复购实验（AGT-034）、漏斗诊断（AGT-023） | 「复购/订阅」、「多任务漏斗依赖强度」 | decision_reason | business_action |
| 4 | `p2s-2026-0033` | 09-DataAgent-LLM | Beyond the Harness: End-to-End Optimization of Context Artifacts for Enterprise Text-to-SQL | 指标契约（AGT-045）、业务工具实现（AGT-047） | 「跨境口径」、「Text-to-SQL」 | decision_reason、title | business_action |
| 5 | `p2s-2026-0020` | 02-A_B实验 | BAFF: Bid-Aware Filter Family for Mitigating Training Data Interference in RTB A/B Tests | 广告实验（AGT-032）、实验设计（AGT-035） | 「RTB 的 A/B」、「三阶段测量协议」 | decision_reason | business_action |
| 6 | `p2s-2026-0022` | 05-推荐系统 | RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations | 转化优化（AGT-023） | 「点击流」 | decision_reason | capability_inference |
| 7 | `p2s-2026-0021` | 13-广告分析 | HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising | 预算分配（AGT-032）、投放诊断（AGT-032） | 「竞价」、「在线学习」 | decision_reason | business_action |
| 8 | `p2s-2026-0012` | 05-推荐系统 | PinEqualizer: Full Funnel Content Exploration and Debiasing System at Pinterest | 内容实验（AGT-030）、算法评估设计（AGT-012） | 「全漏斗内容探索」、「评估骨架」 | decision_reason | business_action |
| 9 | `p2s-2026-0034` | 10-MAS | MAPLE-Guard: Memory-Aware Link Enforcement Against Memory-Link Poisoning in Multi-Agent Systems | 访问控制（AGT-050）、安全事件处理（AGT-050） | 「记忆安全基线」、「毒记忆」 | decision_reason | business_action |
| 10 | `p2s-2026-0028` | 16-智能体工程 | PGMem: Tightly Coupled Persona-Memory Graph for Lifelong Personalized Agents | 知识溯源（AGT-048）、分群（AGT-034） | 「证据效度」、「画像」 | decision_reason | business_action |
| 11 | `p2s-2026-0032` | 10-MAS | AiFlow: Token-Native Reactive Orchestration with Bounded Backpressure for Streaming LLM Applications | 容量管理（AGT-049）、失败恢复（AGT-049） | 「有界背压」、「取消传播」 | decision_reason | business_action |
| 12 | `p2s-2026-0025` | 12-ML基础 | GEAR: Generative Expansion and Real Anchoring for Two-Stage Distillation of Tabular Foundation Models | 业务工具实现（AGT-047） | 「线上跑不动」 | decision_reason | business_action |

## 2. 新旧位次对照（旧 = 人工 priority + 论文分数；新 = 靶区缺口）

| 新位次 | 旧位次 | 变化 | paper_id | 分层 | 靶区位次 | 主 L3 → 岗位 | 供给 |
|---:|---:|---:|---|---|---:|---|---|
| 1 | 3 | +2 | `p2s-2026-0015` | **靶区内** | 27 | 趋势监测 → AGT-007 市场竞争与机会研究 | legacy |
| 2 | 2 | 0 | `p2s-2026-0007` | **靶区内** | 28 | 复购实验 → AGT-034 CRM留存与复购 | legacy |
| 3 | 7 | +4 | `p2s-2026-0024` | **靶区内** | 28 | 复购实验 → AGT-034 CRM留存与复购 | legacy |
| 4 | 10 | +6 | `p2s-2026-0033` | 靶区外 | 靶区外 #22 | 指标契约 → AGT-045 业务口径与主数据 | curated |
| 5 | 8 | +3 | `p2s-2026-0020` | 靶区外 | 靶区外 #34 | 实验设计 → AGT-035 增长实验与增量评估 | curated |
| 6 | 4 | -2 | `p2s-2026-0022` | 靶区外 | 靶区外 #38 | 转化优化 → AGT-023 独立站经营与转化 | curated |
| 7 | 6 | -1 | `p2s-2026-0021` | 靶区外 | 靶区外 #62 | 预算分配 → AGT-032 效果广告投放 | curated |
| 8 | 1 | -7 | `p2s-2026-0012` | 靶区外 | 靶区外 #65 | 内容实验 → AGT-030 内容与创意策划 | curated |
| 9 | 12 | +3 | `p2s-2026-0034` | 靶区外 | 靶区外 #82 | 访问控制 → AGT-050 信息安全与权限 | curated |
| 10 | 5 | -5 | `p2s-2026-0028` | 靶区外 | 靶区外 #101 | 知识溯源 → AGT-048 知识技能与Playbook治理 | curated |
| 11 | 9 | -2 | `p2s-2026-0032` | 靶区外 | 靶区外 #109 | 容量管理 → AGT-049 Agent平台与可靠运行 | curated |
| 12 | 11 | -1 | `p2s-2026-0025` | 靶区外 | 靶区外 #118 | 业务工具实现 → AGT-047 系统集成与业务工具 | curated |

分层口径：**靶区内** = 至少一条服务的 L3 落在 F3 靶区一（A ∩ 无精选卡）；**靶区外** = 服务的 L3 在精选线都已有供给（或不是 A 类）⇒ **不占检索预算位**，只作为「出卡后挂到哪个既有责任」的落点。

## 3. 覆盖率（一等输出）

- backlog 分母（实测）：**12**
- 落到 ≥1 个 L3 责任：**12/12**（覆盖率 100%）
- **未落到任何 L3 责任**：0 篇（无）
- 落在**靶区内**（占检索预算位）：**3/12**：`p2s-2026-0015`→#27 趋势监测(AGT-007)、`p2s-2026-0007`→#28 复购实验(AGT-034)、`p2s-2026-0024`→#28 复购实验(AGT-034)
- 落在**靶区外**（已有供给，具名登记、不静默丢弃）：9 篇：`p2s-2026-0033`→指标契约(curated)、`p2s-2026-0020`→实验设计(curated)、`p2s-2026-0022`→转化优化(curated)、`p2s-2026-0021`→预算分配(curated)、`p2s-2026-0012`→内容实验(curated)、`p2s-2026-0034`→访问控制(curated)、`p2s-2026-0028`→知识溯源(curated)、`p2s-2026-0032`→容量管理(curated)、`p2s-2026-0025`→业务工具实现(curated)
- 靠**旧顺序（分数）**决定位次的篇数：1（只在靶区位次相同的组内生效）
- 低置信映射（`capability_inference`，原文未命名业务责任）：**1** 篇：`p2s-2026-0022`（转化优化）
- 新旧位次位移：**11/12** 篇位次发生变化，最大位移 7 位

## 4. 被判据挡住的东西（反后门）

- `venue_tier` 在排序键里出现 0 次；`score` 只以预计算整数 `old_rank` 进入**末位** tiebreak。
- 「按论文 score 排」的顺序：`p2s-2026-0022`、`p2s-2026-0028`、`p2s-2026-0021`、`p2s-2026-0024`、`p2s-2026-0012`、`p2s-2026-0007`、`p2s-2026-0015`、`p2s-2026-0020`、`p2s-2026-0032`、`p2s-2026-0033`、`p2s-2026-0025`、`p2s-2026-0034`
- 接管后的顺序：`p2s-2026-0015`、`p2s-2026-0007`、`p2s-2026-0024`、`p2s-2026-0033`、`p2s-2026-0020`、`p2s-2026-0022`、`p2s-2026-0021`、`p2s-2026-0012`、`p2s-2026-0034`、`p2s-2026-0028`、`p2s-2026-0032`、`p2s-2026-0025`
- 两者**不同** ⇒ 缺口确实在主导排序；`--selftest` 另有一份注入样本自证。

## 5. 判据自检

| # | 判据 | 状态 |
|---|---|---|
| J1 | 映射完整性（白名单 + 逐字证据 + 无孤儿） | ✅ 12/12 篇有映射，证据逐字命中 |
| J2 | 覆盖率一等输出 / 未落具名 / 空输入退 2 | ✅ 落 L3 12/12，未落 0，靶区内 3 |
| J3 | 改靶区权重位次必须变（承接 F3 J6） | ✅ 4 组权重 → 4 种顺序 |
| J4 | 与 F3 逐条一致（已入库 + 现场生成器） | ✅ 靶区 31 条逐条相等（已入库 + 现场生成器两处） |
| J5 | 反后门：排序键 AST 扫描 + 分数注入样本 | ✅ venue_tier 0 命中；接管序 ≠ 按 score 序 |
| J6 | 反向控制：干净输入 exit 0 | ✅ 本行即证据 |
| J8 | 接管确实改变了旧顺序（双边） | ✅ 11/12 篇位次动了，最大位移 7 |
| J3b | 位次靠旧顺序 tiebreak 决定的篇数（可见账） | ✅ 1 篇；低置信映射 1 篇 |
| J9 | 输入新鲜度：缺口账声明的图谱 sha vs 现状（风险 N6） | ✅ 图谱已变（7ede8be116ca7672→be36dd1d197def0e），现算核对：stale_immaterial（图谱换了底本（多为 cells[].solution_refs / solutions 层），但本脚本排序所依赖的 151 行（L3/服务性/供给/FLOW/边界）与 31 条工单**现算逐条相同** ⇒ 不影响位次） |
| J7 | registry 自洽（N4：15 = 12 + 3） | ✅ naive 15 − 增强交付 3 = 12 |

## 6. 未做 / 留给下游

- **不执行扩充**：本表只改「下一条该萃取哪篇」的顺序，不抓论文、不出卡（S3 的边界）。
- **没有逐篇实读全文**：映射依据是 registry 的 `decision_reason`／`title`（逐字取证），不是论文全文；`capability_inference` 那类已显式标出。
- **不改契约/图谱/缺口账**：`build_gap_ledger.py` 只被 import 当 oracle，未改动一行。


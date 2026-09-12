---
title: 证据链 — p2s-2026-0016（Skill-Stateful-Skill-Runtime）
doc_type: evidence
module: 16-智能体工程
paper_id: 2608.26263
registry_id: p2s-2026-0016
related_card: paper2skills-vault/16-智能体工程/Skill-Stateful-Skill-Runtime.md
fulltext: paper2skills-vault/papers/16-智能体工程/p2s-2026-0016/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2608.26263 *SKILL.state: Scalable Long-Horizon Agent Skills*

- **全文存档**：`paper2skills-vault/papers/16-智能体工程/p2s-2026-0016/fulltext.md`
  （965 行 / 49,889 字符，`evidence_grade: A`）
- **venue 核对**：全文**没有任何录用线索**（无 `Conference:` 页眉、无 ACL/EMNLP 模板痕迹）。
  全文里 `EMNLP` 的命中**全部在参考文献**中，指的是 LLMLingua（Jiang et al., 2023）与
  Li et al. (2023) 的发表处，**不是本文的 venue**。
  → 卡片 frontmatter 仍按 registry 记 `venue: EMNLP` / `venue_tier: top`，
  **但本卡不得据此声称"本文发表于 EMNLP"**；引用时按 arXiv preprint 引用。
- **引文全部为逐字切片**：下列每条 `> 原文："…"` 都是用 `fulltext.md` 的
  「行号 + 起止锚点」**程序化切出的连续子串**，没有拼接、没有改写、没有跨段缝合。
  之所以用程序切片而不是手抄：拼接型伪造在 n-gram 覆盖率上仍是 1.0，
  只有"最长连续匹配段"才拦得住（`quote_check.py --selftest` 用例 3）。

## 0. 核验凭证（真实 stdout / 退出码）

```
$ C=paper2skills-vault/16-智能体工程/Skill-Stateful-Skill-Runtime.md

$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C"
已索引仓库内本地模块 86 个（可自动解析的卡片将真正执行）
另有 54 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）
✅ PASS         Skill-Stateful-Skill-Runtime.md#stitched(2块)

==================================================================
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | 🟠 MIGRATED 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）
==================================================================
```

K1 分级明细（取自 `--json-out` 产物）：`L1_SYNTAX 通过` / `L2_COMPILE 通过` / `L3_IMPORT 模块可 import` /
`L4_SMOKE 脚本执行成功` / `L5_TEST pytest 全绿（7 passed）`，代码 441 行。

```
$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$C"
✅ VERBATIM   Skill-Stateful-Skill-Runtime.md  50/50 逐字

共 1 张卡：1 通过（有引用块且逐字可核），0 含伪造引文，0 无全文可核验，0 无引用块（**不等于通过**）
```

```
$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$C" --k1 <K1 产物>

======================================================================
G1 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================

======================================================================
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 3
======================================================================

======================================================================
G3 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 0
======================================================================
[exit code: 0]
```

G2 的关键计数（门禁 JSON `metrics`）：
`metric_numbers 6 / sourced 6 / unsourced_metric 0 / traceability_pct 100.0 /
quotes_total 50 / quotes_verbatim 50 / quotes_fuzzy 0 / quotes_fabricated 0 /
quotes_spliced 0 / sourced_structural_only 0 / cn_numeral_claims 0 / fulltext_archived true`。

G2 的 3 条黄灯（**非阻塞**，如实列出）：`90天`（场景 1 的业务参数，已就地标注"业务参数，非论文数字"）、
`10`×2（`10-MAS` 域目录名，是路径指代不是断言）。

```
$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
1 真引文     : VERBATIM   连续度=1.0
2 纯伪造     : FABRICATED 连续度=0.124   ← 必须 FABRICATED
3 拼接引文   : FABRICATED 连续度=0.681 召回=1.0 拼接标记=True   ← 必须被拦下
4 排版差异   : VERBATIM   连续度=1.0   ← 必须 VERBATIM
✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接
```

## 1. registry 逐项核对表（`papers_registry.json` → `p2s-2026-0016`）

| registry 字段 | registry 写的 | 回原文核对结论 | 处置 |
|---|---|---|---|
| `title` | SKILL.state: Scalable Long-Horizon Agent Skills | ✅ 与全文标题一致（fulltext L9） | 采用 |
| `arxiv` | 2608.26263 | ✅ 与存档头注释 `arxiv_id: 2608.26263` 一致 | 采用 |
| `venue` / `venue_tier` | `EMNLP` / `top` | ⚠️ **全文无任何录用线索**（详见开头 venue 核对） | 沿用 registry，但本卡不得声称"发表于 EMNLP"；引用按 arXiv preprint |
| `decision_reason` | 「不可变 Skill 规格 + 可变结构化执行状态，prompt 不随步数增长 → Skill 运行时该做状态机而非 chat loop」 | ✅ 与原文完全相符：§3 定义 A_t=(P,Σ_t,O_t)，§3.3 给出 \|P_t\| 与 t 无关，§1 贡献 1 明写 O(1)/O(T) | 采用 |
| `note` | 「摘要数字截断，须回原文核实压缩比」 | ✅ **这条提醒成立**，见第 2 节：摘要**一个数字都没有**，压缩比/准确率全部只能从正文结果表取 | 已在卡片正文按正文数字写，并逐条给出 ⑥ 段出处 |
| `data_availability` | `available` | ⚠️ **部分成立**：InterCode CTF 与 Sierra τ-Bench 是第三方公开基准（论文明确列出出处）；但 SkillExecBench 是**自建**基准，论文只给了环境设计与任务生成伪代码（附录 B），**全文没有**任何代码/数据/模型发布声明：实测 grep `github` / `open-source` **零命中**；`release` 的唯一命中是 Software Repository 环境的动作 `CreateRelease`，`available` 的命中是附录 A 的措辞 `the available action space` 与算法 2 的变量名 —— 均与发布声明无关 | 卡片不声称"数据公开可得"；场景段的"数据可得性"按企业自有数据填写 |
| `priority` / `score` | `P0` / 48.2 | —（registry 侧排序值，不是论文事实） | 不核对 |
| `outputs.skill_card` | `paper2skills-vault/16-智能体工程/Skill-<方法名>.md`（模板占位） | — | 实际落地为 `Skill-Stateful-Skill-Runtime.md` |
| `outputs.code_dir` | `paper2skills-code/llm_agent_engineering/<algo>` | —（规划值；按 CLAUDE.md，代码模板内嵌卡片 ③ 段） | 未落地 code 目录，由 K1 验证卡片内代码 |

**registry 与原文的三处不符/需收窄（本卡发现）**

1. **venue 无原文证据**：`EMNLP` / `top` 只能沿用 registry，不能作为论文事实引用（见上表）。
2. **`data_availability: available` 需收窄为"部分"**：可得的只有两个第三方公开基准；
   自建基准 SkillExecBench 与环境代码均无发布声明。
3. **registry 没有记的一个事实**：论文表 4 里 SKILL.state 在 τ-Bench Retail 上的**单步平均 prompt
   反而高于** ReAct 基线（⑥ Q37/Q38）。若只看 registry 的"prompt 不随步数增长"，会误读成
   "prompt 一定更小"。卡片 ①b 已把这条写成显式边界。

## 2. 「摘要数字截断」的核实结论（registry `note` 的正面回应）

**核实结论：registry 这条 note 成立 —— 摘要里没有任何具体数字。**

摘要（`fulltext.md` 第 15 行）关于效果只有一句定性描述：

> 原文："Across diverse datasets, models, and execution environments, SKILL.state improves task accuracy while substantially reducing cumulative token consumption."
> 出处：2608.26263 §Abstract 摘要｜Q02

通读摘要全文：没有百分比、没有倍数、没有 token 数，也没有准确率。因此：

- **能在摘要里引用的结论只有"准确率提升 + 累计 token 显著下降"这一条定性表述**；
- 卡片里出现的每一个压缩比/准确率数字，**全部取自正文**（§5.2 正文、§5.5 正文、§5.6 正文，
  以及表 1/表 4/表 5 的表格行），并在 ⑥ 段逐条给出逐字引文；
- 摘要**没有**说"提高了多少"，所以卡片也**不写**"论文报告提升 X%"这类由摘要外推的句子。

**从正文取到的压缩比与准确率（本卡实际使用）**

| 口径 | 数字 | 出处 | 引文键 |
|---|---|---|---|
| Warehouse（T=100）累计 token 缩减 vs Stateful 基线 | **16.2×**（1,062,387 → 65,408 tokens） | §5.2 正文 | Q25 |
| Warehouse（T=200）准确率 / 累计 token | 0.94 / 122k（Memory 基线 6.1M） | §5.2 正文 | Q26 |
| Warehouse 平坦 prompt 区间（T=10…200） | ~1,736–1,905 tokens | §5.2 正文 + 表 1 | Q23 |
| InterCode CTF pass@1 与差值 | 54.2%（+7.8 / +12.4 个百分点） | §5.5 正文 | Q30 |
| InterCode CTF 累计 token 降幅 | 60.4%（vs ReAct）/ 65.9%（vs Stateful） | §5.5 正文 | Q30 |
| τ-Bench Retail pass rate | 58.3% | §5.5 正文 | Q31 |
| τ-Bench Airline pass rate 与 token 降幅 | 32.4% / 40.5%（vs ReAct）/ 45.4%（vs Stateful） | §5.5 正文 | Q32 |
| **同预算（~1,800 tokens）对照**：滑动窗口 / LLMLingua / SKILL.state | 0.18 / 0.22 / 0.94 | §5.6 正文 + 表 5 | Q33 / Q36 |
| 同预算对照：Summary-capped / 无预算 ReAct | 0.52 / 0.84 | 表 5 | Q35 / Q34 |
| 噪声鲁棒性（T=50）：Prompt 基线 / SKILL.state | 0.68→0.53 / ≥0.97 | §5.3 正文 + 表 2 | Q27 |
| 状态恢复：历史式基线 / SKILL.state 恢复步数 | 5–8 步 / 0 步 | §5.4 正文 + 表 3 | Q28 |
| 开放权重模型错误分布 | 68% / 20% / 12%（Gemma-4-31B, T=100, score 0.42） | §5.7 正文 | Q39–Q42 |

**「同预算」是这张卡最有说服力的一点**：论文不是拿结构化状态去比一个"不设预算的长 prompt"，
而是把 `Truncated (Sliding Window)`、`Summary-capped`、`ReAct + LLMLingua` 三个压缩基线
**钉在 SKILL.state 自己的 token 预算上**（⑥ Q22），结果三者分别掉到 0.18 / 0.52 / 0.22，
SKILL.state 保持 0.94（⑥ Q33/Q34/Q35/Q36）。也就是说：**差距不来自"prompt 更短"，
而来自"状态是结构化的"** —— 统计式压缩会删掉语义上必需的槽位标识符（⑥ Q33）。

## 3. 卡片正文数字 → 出处对照表

| 卡片里的数字 | 含义 | 位置 | 引文键 |
|---|---|---|---|
| `16.2×` | Warehouse T=100 累计 token 相对 Stateful 基线的缩减 | ② 场景 1、⑤ | Q25 |
| `1,062,387` / `65,408` | 上述两者的原始 token 数 | ② 场景 1、⑤ | Q25 |
| `0.94` / `122k` / `6.1M` | T=200 的准确率 / SKILL.state 累计 token / Memory 基线累计 token | ①b、⑤ | Q26 |
| `0.68` / `0.53` / `0.97` | 噪声下的 Prompt 基线准确率下降 / SKILL.state 的下界 | ①b | Q27 |
| `60.4%` / `65.9%` | InterCode CTF 累计 token 降幅 | ⑤ | Q30 |
| `54.2%` / `7.8` / `12.4` | InterCode CTF pass@1 与其相对差值 | ⑤ | Q30 |
| `58.3%` / `32.4%` / `40.5%` / `45.4%` | τ-Bench Retail / Airline 的通过率与 token 降幅 | ⑤ | Q31, Q32 |
| `0.18` / `0.22` / `0.52` / `0.84` / `0.94` | 同预算对照下四个 runtime + 无预算 ReAct 的得分 | ①b、⑤ | Q33–Q36 |
| `1,800`（预算）/ `1,905` | 表 5 的对照预算 / SKILL.state 平均 prompt | ③、⑤ | Q22, Q36 |
| `1,736`–`1,905` | 表 1 里 SKILL.state 的平坦 prompt 区间 | ①b | Q23 |
| `3,325` / `2,819` | τ-Bench Retail 单步平均 prompt（SKILL.state 高于 ReAct） | ①b（定性描述，数字留在 ⑥） | Q37, Q38 |
| `43.2%` / `48.2%` / `21.8%` | 表 4 中 ReAct 基线在三份基准上的通过率 | ⑥ 表 4 行 | Q37 |
| `68%` / `20%` / `12%` / `0.42` | 开放权重模型的错误分布与总分 | ①b | Q39–Q42 |
| `100` / `5-field` | InterCode CTF 题数 / 复用同一套 schema 的字段数 | ①、④ | Q08 |
| `90 天` | 场景 1 要求的工单历史长度（**业务参数，非论文数字**，卡内已就地标注） | ② | —— |
| `10-MAS` | 关联卡片所在域目录名（路径指代） | ④ | —— |

对照表之外，卡片正文**不存在**其他带度量语义的数字；③ 段代码围栏内的数字全部是
**构造数据与本地可复现输出**（G2 会剥离代码围栏），与论文数字是两个口径，卡内已显式声明
不可互相印证。

## 4. 逐字引文全表（50 条，与卡片 ⑥ 段一一对应）

### A. 问题陈述：对话式 runtime 为什么会随长程退化

> 原文："Existing agent runtimes maintain execution by continually appending observations, actions, and intermediate reasoning traces to an ever-growing conversation history, causing latency degradation and context-poisoning failures over long horizons."
> 出处：2608.26263 §Abstract 摘要｜Q01

> 原文："Across diverse datasets, models, and execution environments, SKILL.state improves task accuracy while substantially reducing cumulative token consumption."
> 出处：2608.26263 §Abstract 摘要｜Q02

> 原文："Modern agent runtimes almost universally adopt a conversational execution model. At every execution step, the language model receives the original skill specification together with an ever-growing transcript of previous reasoning, actions, observations, and tool outputs (Yao et al., 2022; Mialon et al., 2023)."
> 出处：2608.26263 §1 Introduction｜Q03

> 原文："Prompt size grows with execution length, increasing token consumption and inference cost (Liu et al., 2024; Xiao et al., 2024a). Historical observations and obsolete reasoning remain embedded in the context long after they cease to be relevant, requiring the model to continually distinguish current facts from historical artifacts. Consequently, execution correctness increasingly depends on reconstructing state from accumulated textual history."
> 出处：2608.26263 §1 Introduction｜Q04


### B. 架构定义：每步只喂 (P, Σ_t, O_t)

> 原文："At execution step $t$, the language model receives only three inputs:"
> 出处：2608.26263 §1 Introduction｜Q05

> 原文："where $P$ is the immutable procedural specification, $\Sigma_{t}$ is the structured execution state at step $t$, and $O_{t}$ is the latest observation received from the environment. The language model never receives previous observations, previous actions, or previous reasoning traces."
> 出处：2608.26263 §3 SKILL.state｜Q06

> 原文："At each step, the runtime constructs a prompt from $(P,\Sigma_{t},O_{t})$, invokes the language model, deterministically validates the proposed state transition, updates the execution state, executes the selected action, and repeats the process using the updated state."
> 出处：2608.26263 §3 SKILL.state（Figure 1 执行循环）｜Q07

> 原文："Schemas are authored once per domain rather than per task; for example, across all 100 diverse challenge instances in the InterCode CTF benchmark, the agent reuses a single static 5-field schema (discovered_flags, tested_hypotheses, active_files, working_dir, cmd_summary)."
> 出处：2608.26263 §3.1 Execution State and Schema Authoring｜Q08

> 原文："After producing a validated state update, the intermediate reasoning trace is discarded while only the updated execution state is retained. Consequently, execution depends strictly on the current world state instead of replaying historical trajectories."
> 出处：2608.26263 §1 Introduction｜Q09


### C. 中间推理的丢弃机制与状态更新的校验

> 原文："Crucially, within-step multi-step reasoning is fully intact during generation to support complex deductive planning. However, once the state transition has been validated and applied, the reasoning trace $R_{t}$ is discarded permanently and never appears in subsequent prompts."
> 出处：2608.26263 §3.2 Reasoning and State Transitions｜Q10

> 原文："where $\oplus$ denotes the runtime’s dictionary merge operator with null-deletion semantics. This model projects transient reasoning into persistent structured state, allowing only information required for future execution to survive across interactions."
> 出处：2608.26263 §3.2 Reasoning and State Transitions（式 (4) 后）｜Q11

> 原文："Because schema ownership and validation reside in the deterministic runtime rather than the model, malformed outputs cannot corrupt persistent state $\Sigma_{t}$; an invalid patch triggers a rollback-retry cycle."
> 出处：2608.26263 §7 Limitations｜Q12


### D. O(1) prompt / O(T) 累计 token 的复杂度主张

> 原文："We propose SKILL.state, a runtime architecture that executes procedural skills through explicit structured execution state where intermediate reasoning is discarded after each step, proving a strictly bounded $\mathcal{O}(1)$ prompt footprint and $\mathcal{O}(T)$ cumulative token complexity."
> 出处：2608.26263 §1 Introduction（贡献 1）｜Q13

> 原文："Let $T$ denote the execution horizon. For conversational runtimes, prompt length grows with the accumulated interaction history, $|C_{t}|=\mathcal{O}(t)$, leading to cumulative token complexity:"
> 出处：2608.26263 §3.3 Complexity Analysis（式 (5) 前）｜Q14

> 原文："In contrast, SKILL.state maintains only the procedural specification, structured execution state, and latest observation:"
> 出处：2608.26263 §3.3 Complexity Analysis（式 (6) 前）｜Q15

> 原文："which is asymptotically bounded and independent of the number of previously executed turns $t$. Consequently, cumulative prompt complexity grows strictly linearly with the execution horizon:"
> 出处：2608.26263 §3.3 Complexity Analysis（式 (7) 前）｜Q16

> 原文："By discarding intermediate reasoning traces after each validated transition, SKILL.state maintains a bounded $\mathcal{O}(1)$ prompt footprint and scales linearly $\mathcal{O}(T)$ in cumulative tokens."
> 出处：2608.26263 §6 Conclusion｜Q17


### E. 对照组与「同预算」的压缩基线

> 原文："Memory (Summarization-style): Maintains a rolling 3-step conversational window alongside a periodically updated natural language summary of past interactions (Packer et al., 2023)."
> 出处：2608.26263 §5.1 Experimental Setup（Primary Runtime Paradigms）｜Q18

> 原文："Truncated (Sliding Window): Retains only the most recent interaction turns that fit within a fixed token budget."
> 出处：2608.26263 §5.1 Experimental Setup（Budget-Matched and Compression Controls）｜Q19

> 原文："Summary-capped: Strictly enforces a hard token ceiling on the natural language summary."
> 出处：2608.26263 §5.1 Experimental Setup（同组）｜Q20

> 原文："ReAct + LLMLingua (Jiang et al., 2023): Uses budget-aware small-model perplexity compression to prune tokens from the full history down to the target budget."
> 出处：2608.26263 §5.1 Experimental Setup（同组）｜Q21

> 原文："To determine whether SKILL.state’s performance gains stem merely from shorter prompts or from structured state representation, we evaluate budget-matched baselines on Warehouse ($T=100$, Gemini-3-Flash) pinned to the token budget of SKILL.state ($\sim$1,800 tokens)."
> 出处：2608.26263 §5.6 Experiment 5: Budget-Matched Controls and Statistical Compression｜Q22


### F. 结果数字（压缩比、准确率、公开基准）

> 原文："Results: As shown in Table 1, SKILL.state matches or exceeds baseline accuracy across all horizons while maintaining a flat prompt size ($\sim$1,736–1,905 tokens)."
> 出处：2608.26263 §5.2 Experiment 1: Long-Horizon Execution Scaling｜Q23

> 原文："In contrast, history-appending baselines suffer quadratic token accumulation $\mathcal{O}(T^{2})$."
> 出处：2608.26263 §5.2 Experiment 1（同段）｜Q24

> 原文："At $T=100$, the Stateful baseline consumes 1,062,387 tokens, whereas SKILL.state consumes only 65,408 tokens (a $16.2\times$ token reduction)."
> 出处：2608.26263 §5.2 Experiment 1（同段）｜Q25

> 原文："At $T=200$, SKILL.state maintains 0.94 accuracy consuming 122k tokens, while the Memory baseline inflates to 6.1M tokens."
> 出处：2608.26263 §5.2 Experiment 1（同段）｜Q26

> 原文："As shown in Table 2, the standard Prompt runtime degrades sharply from 0.68 at low noise down to 0.53 at high noise. In contrast, SKILL.state maintains robust task completion ($\geq 0.97$) across all noise levels because distractors are filtered out during state patch generation and never enter subsequent prompts."
> 出处：2608.26263 §5.3 Experiment 2: Context Corruption (Noise Robustness)｜Q27

> 原文："As shown in Table 3, history-based baselines hallucinate for 5 to 8 consecutive turns because obsolete facts in their prompt history overpower contradictory new observations. In sharp contrast, SKILL.state requires zero recovery steps: because its decisions depend on the current structured state, the state is updated immediately upon receiving the corrective alert."
> 出处：2608.26263 §5.4 Experiment 3: State Recovery｜Q28

> 原文："Results: As shown in Table 4, SKILL.state achieves the highest task completion rates across all three benchmarks while substantially cutting cumulative token consumption."
> 出处：2608.26263 §5.5 Experiment 4: Public Interactive Benchmarks｜Q29

> 原文："In InterCode CTF, maintaining explicit hypotheses and discovered flags in $\Sigma_{t}$ prevents the model from repeating failed commands, increasing pass@1 to 54.2% (+7.8 points over the strongest baseline and +12.4 points over Stateful) while cutting total tokens by 60.4% vs. ReAct and 65.9% vs. Stateful."
> 出处：2608.26263 §5.5 Experiment 4（同段）｜Q30

> 原文："In $\tau$-Bench Retail, SKILL.state leads with 58.3% pass rate at the lowest total token cost."
> 出处：2608.26263 §5.5 Experiment 4（同段）｜Q31

> 原文："In $\tau$-Bench Airline, where complex database responses cause baseline prompts to peak above 11,000 tokens/step, SKILL.state maintains a flat footprint of $\sim$2,800 tokens/step and achieves a 32.4% pass rate, saving 40.5% tokens vs. ReAct and 45.4% vs. Stateful."
> 出处：2608.26263 §5.5 Experiment 4（同段）｜Q32

> 原文："Sliding-window truncation drops to 0.18 because critical early inventory allocations are evicted. LLMLingua drops to 0.22 because statistical entropy filtering removes seemingly redundant slot identifiers that are semantically vital. In contrast, SKILL.state achieves 0.94 score, demonstrating that structured state maintenance preserves exact relational dependencies that statistical compressors destroy."
> 出处：2608.26263 §5.6 Experiment 5｜Q33

> 原文："| Full ReAct (Unbounded) | 0.84 | 36,362 | 1,245,413 |"
> 出处：2608.26263 §5.6 表 5（Budget $\sim$1,800 tokens 对照）｜Q34

> 原文："| Summary-capped | 0.52 | 1,840 | 63,400 |"
> 出处：2608.26263 §5.6 表 5｜Q35

> 原文："| SKILL.state (Structured) | 0.94 | 1,905 | 65,408 |"
> 出处：2608.26263 §5.6 表 5｜Q36

> 原文："| Prompt (ReAct) | 43.2% | 1,909 | 977k | 48.2% | 2,819 | 4.48M | 21.8% | 5,100 | 4.85M |"
> 出处：2608.26263 §5.5 表 4（InterCode CTF / τ-Bench Retail / τ-Bench Airline）｜Q37

> 原文："| SKILL.state | 54.2% | 813 | 387k | 58.3% | 3,325 | 3.47M | 32.4% | 2,800 | 2.88M |"
> 出处：2608.26263 §5.5 表 4｜Q38


### G. 失败模式、统计口径与模型设置

> 原文："On open-weight models (Gemma-4-31B at $T=100$, score 0.42), we analyze failure logs and categorize errors into three distinct modes:"
> 出处：2608.26263 §5.7 Error Taxonomy for Open-Weight Models｜Q39

> 原文："Premature State Overwrite / Deletion (68%): The model accidentally omits existing keys during state update rather than merging in-place."
> 出处：2608.26263 §5.7（失败模式 1）｜Q40

> 原文："Schema Comprehension / Type Coercion (20%): Inconsistencies between expected nested lists and dictionaries."
> 出处：2608.26263 §5.7（失败模式 2）｜Q41

> 原文："JSON Syntax / Formatting Slips (12%): Malformed JSON delimiters or trailing commas."
> 出处：2608.26263 §5.7（失败模式 3）｜Q42

> 原文："This error distribution shows that small-model degradation stems from structured output adherence rather than reasoning capacity, motivating constrained decoding in future runtime iterations."
> 出处：2608.26263 §5.7（同段）｜Q43

> 原文："Statistical Significance: All synthetic experiments are evaluated across 5 distinct procedural generator seeds. Results are reported as mean $\pm$ sample standard deviation. Differences between SKILL.state and baselines at extended horizons ($T\geq 50$) are statistically significant (paired $t$-test, $p<0.01$)."
> 出处：2608.26263 §5.1 Experimental Setup｜Q44

> 原文："Underlying Models: Evaluations are conducted across proprietary and open-weight models: Gemini-3-Flash, Gemma-4-31B-it, and Qwen-3-8B-it. Decoding is controlled at temperature $0.0$ and top-$p$ $1.0$ across all runs to ensure deterministic reproducibility."
> 出处：2608.26263 §5.1 Experimental Setup｜Q45


### H. 相关工作定位与论文自承局限

> 原文："Frameworks like LangGraph use auxiliary structured state to orchestrate workflows across agent nodes. However, these systems still rely on conversational transcripts as the primary reasoning substrate. SKILL.state replaces this substrate by discarding intermediate reasoning traces immediately after producing validated state transitions."
> 出处：2608.26263 §2.2 Memory Architectures for Long-Horizon Agents｜Q46

> 原文："Rather than attempting to process or compress extended conversational histories, SKILL.state prevents history accumulation entirely by maintaining the canonical execution state required for the next computation."
> 出处：2608.26263 §2.4 Context Management and Long-Context Reasoning｜Q47

> 原文："SKILL.state assumes that the execution state can be made a sufficient statistic for future execution: that everything in the past bearing on future actions can be projected into the structured state as soon as it becomes known."
> 出处：2608.26263 §7 Limitations｜Q48

> 原文："However, this assumption fails in three distinct settings: (1) when no fixed schema is known in advance and the relevant state structure must be discovered dynamically during execution; (2) when a correct state update depends on an earlier observation whose relevance was not recognized when first observed, and was therefore never committed to state; and (3) when the task objective is defined over the historical trajectory itself (e.g., auditing, debugging provenance, or explaining past actions), where interaction history is the target output rather than operational overhead."
> 出处：2608.26263 §7 Limitations（同段）｜Q49

> 原文："Our current implementation focuses on single-agent procedural execution. While the explicit state abstraction extends naturally to multi-agent systems—where a shared execution state acts as the central coordination substrate instead of exchanging quadratic conversational transcripts—multi-agent environments introduce concurrent writes, requiring deterministic conflict-resolution semantics in the merge operator $\oplus$ that our single-agent setting does not exercise."
> 出处：2608.26263 §7 Limitations｜Q50


## 5. 未能核验 / 只能用「论文未报告」的地方（逐条列出，不编数）

1. **任何货币成本、时延数字、工程人力**：论文**未报告**。摘要里的 `latency degradation`
   是定性描述（⑥ Q01），全文没有延迟毫秒数、$ 金额或人天。卡片 ⑤ 因此只给 ROI 公式与
   参数来源，**不给金额结论**。
2. **母婴 / 跨境电商场景的任何内容**：论文**未报告**。它的公开基准是 Linux CTF 与
   零售/航空客服（τ-Bench），与本卡场景只有机制同构，**没有**数据迁移证据。卡片 ② 的
   品类、渠道、周期全部是业务侧设定，并已标注业务参数。
3. **公开代码 / 数据发布声明**：**未找到**。全文无 `github` / `release` / `open-source` /
   发布链接；SkillExecBench 只有环境设计与任务生成伪代码（附录 B）。
   （实测口径：`github` / `open-source` 零命中；`release` 只命中环境动作 `CreateRelease`；
   `available` 只命中附录 A 措辞与算法 2 变量名。）
   → 「本卡 ③ 段可运行」**只证明卡片自带代码可执行，不证明**它与作者实现一致。
4. **venue**：全文**无录用证据**（见开头），沿用 registry。
5. **表 1 / 表 4 / 表 5 中未被逐字引用的格子**：例如表 1 的 T=10/25/50 各行、
   表 4 的 Memory/Stateful 行、附录 D 表 6–8（Gemma / Qwen）的具体数值 —— 存档中完整，
   但本卡未逐条引用，**因此卡片正文不使用这些数字**（避免"数字在表里"与"数字有逐字出处"
   混为一谈）。
6. **图 1 的图形内容**：HTML→Markdown 后只剩图注
   `*Figure 1: Overview of the SKILL.state architecture.*`，图形本身无法核验，故只引用
   §3 正文对执行循环的文字描述（⑥ Q07）。
7. **Algorithm 1 / Algorithm 2 的逐行伪代码**：存档中是编号列表，且含 em space（U+2003）
   等排版字符，逐字引用会引入排版噪声，故只引用 §3.2 正文的等价描述（⑥ Q10）。**这不是
   无法核验，而是主动选择更干净的引用底本。**
8. **多 agent 的并发写冲突消解方案**：论文明确说这是单 agent 设置没有检验的部分（⑥ Q50），
   **未给出** `⊕` 的冲突消解语义。卡片 ①b 据此写成"暂不适用于多 agent 共享状态"。
9. **语法约束解码的具体效果**：论文只说"integrating grammar-constrained decoding can eliminate
   syntactic formatting errors"（⑥ Q12），**未报告**任何实验数字。卡片因此不写"能消除多少
   格式错误"。
10. **本卡 ③ 段演示输出的倍数（本地 13.0 倍）与论文报告的 16.2×**：两者口径不同
    （前者是构造数据下的字符数比、对照的是 ReAct 式 transcript；后者是论文 Warehouse T=100 的
    token 比、对照的是 LangGraph 式 Stateful 基线），卡片已显式声明**不可互相印证**。

## 6. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
C=paper2skills-vault/16-智能体工程/Skill-Stateful-Skill-Runtime.md
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$C" --json-out /tmp/k1.json
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
python3 paper2skills-skills/paper-审核/scripts/quote_check.py   --card "$C"
python3 paper2skills-skills/paper-审核/scripts/gate_check.py    --card "$C" --k1 /tmp/k1.json
```

---
title: 证据链 · p2s-2026-0008（2606.29366）
doc_type: evidence
module: papers/04-供应链
status: draft
created: 2026-09-12
updated: 2026-09-12
owner: self
source: ai
paper_id: 2606.29366
paper: Solver-Verified Formulation Generation and Selection for Multi-Warehouse Inventory Allocation Using Large Language Models
venue: arXiv preprint
venue_tier: preprint
evidence_grade: A
related: ../../../04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md
---

# 证据链 · 2606.29366（registry id `p2s-2026-0008`）

## 0. 交付物与核验状态

| 项 | 值 |
|---|---|
| 卡片 | `paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md` |
| 全文底本 | `paper2skills-vault/papers/04-供应链/p2s-2026-0008/fulltext.md`（532 行 / 58716 字符，arXiv LaTeXML HTML → Markdown） |
| 引文总数 | 41 条（卡片 ⑥ 段） |
| 引文逐字核验 | **41/41 VERBATIM**，`n_fuzzy: 0`，`n_fabricated: 0`，`n_spliced: 0` |
| 核验器自证 | `quote_check.py --selftest` → **✅ 自检通过**（真引文 VERBATIM / 纯伪造 FABRICATED / 拼接 FABRICATED / 排版差异 VERBATIM） |
| K1 代码可执行 | **PASS**（`Skill-Multi-Warehouse-Allocation-LLM.md#stitched(4块)`，L1–L5 全绿，pytest 6 passed） |
| K2 门禁 | **G1 passed / G2 passed / G3 passed，红灯 0**（G2 黄灯 2 条均为 frontmatter `created`/`updated` 日期里的 `12`；G3 黄灯 0） |

**G1 需要 K1 凭证**（`gate_check.py` 的设计如此：无 K1 产物一律判红，禁止凭人工判断放行）。
不带 `--k1` 直接跑 `gate_check.py --card <卡片>` 时 G1 必然红灯，这不是卡片缺陷 —— 参考卡片
`13-广告分析/Skill-Cannibalization-Corrected-Attribution.md`（全库首个三门通过者）同样如此。
本章末尾给出两条命令。

---

## 1. 卡片正文数字 → 出处（逐条）

> 口径：卡片正文（散文）里出现的度量数字必须落进 ⑥ 段的某条逐字引文；本表把对应关系列全。
> B 类「本地可复现数字」（本模板运行输出、按贵司参数代入的算式）一律在代码/输出围栏内，不在此表。

| 卡片正文断言 | 位置 | 出处（引文编号见第 2 节） |
|---|---|---|
| 语法准确率 **100%**（SFT+KTO） | ①b | #23（§6.2 表 4 行，L374）、#26（§6.2，L380） |
| 执行成功率 **99.9%** | ①b | #24（§6.2 表 4 行，L376）、#26（§6.2，L380） |
| 执行失败率 **1.0% → 0.1%** | ①b | #25（§6.2 表 4 行，L378）、#26（§6.2，L380） |
| 固定 KI 公式整体 **3.4 个百分点**、**29** 个生产批次、**26** 个有改善 | ⑤、①b | #4（Abstract，L16）、#31（§7.1，L398） |
| 公式选择后整体 **4.5 个百分点**（含其代价：26 个获益批次里 **7** 个不如固定公式） | ⑤ | #4（Abstract，L16）、#31（§7.1，L398）、#35（§7.1，L402） |
| 未训练任务 **5** 个类别、方案合法 **3** 个 | ①b | #36（§7.2，L440）、#37（§7.2，L456）、#38/#39（§7.2 表 6，L452/L454） |
| SFT 覆盖 **18** 个公式 / **17** 个变体、注入 **6** 类错误、混合比 **85%/10%/5%** | ①b | #19、#20、#21（§6.1，L362） |
| 留出测试集 **10,000+** 实例（卡片仅引述，未做量级推断） | ①b 的引用说明 | #22（§6.2，L368） |
| 「≥8 周」预测历史 | ② 数据要求、④ | **不是论文事实**：贵司数据要求，按业务口径自定 |

**中文数字量级断言**：0 条（门禁 `cn_numeral_claims: 0`）——
卡片里所有来自论文的量级都用阿拉伯数字表达，且都能回到引文。

**「论文未报告，故本卡不写数字」的地方**（改用了定性表述）：
跨境头程时效方差与清关不确定性、断货惩罚与仓储费的金额标定、需求预测误差对分仓结论的传导、
多期/网络级结果、推理成本与求解耗时、专家数量 $E$ / Top-$K$ 的 $K$ / 温度 $\kappa$ 的取值
（论文只给符号），以及生产批次的 SKU 覆盖数。

---

## 2. 引文清单（41 条，全部 VERBATIM）

> 卡片 ⑥ 段的每一条 `> 原文："..."` 与紧随的 `> 出处：...`。`L*` 为
> `fulltext.md` 中的行号（**不是 PDF 页码** —— 本地没有 PDF 原件，页码无法核验，
> 故一律改用可机器复核的存档行号）。

| # | 出处 | 引文（截断显示） |
|---|---|---|
| 1 | `2606.29366 Abstract（fulltext.md L16）` | Balance-oriented multi-warehouse inventory allocation is a recurring decision … |
| 2 | `2606.29366 Abstract（fulltext.md L16）` | In practice, allocation requirements are often scenario-dependent and expresse… |
| 3 | `2606.29366 Abstract（fulltext.md L16）` | The LLM component generates candidate formulations and executable solver code … |
| 4 | `2606.29366 Abstract（fulltext.md L16）` | Experimental results on 29 production evaluation batches from JD.com show that… |
| 5 | `2606.29366 §1 Introduction（fulltext.md L26）` | We measure inventory coverage through Target Inventory Days (TID), defined as … |
| 6 | `2606.29366 §3.1 Problem Setup（fulltext.md L62）` | For a given stock keeping unit (SKU), the input consists of: (i) the on-hand i… |
| 7 | `2606.29366 §3.1 Problem Setup（fulltext.md L64）` | The decision is an integer allocation plan $x_{k}\in\mathbb{Z}_{\geq 0}$ for e… |
| 8 | `2606.29366 §3.1 Problem Setup（fulltext.md L78）` | Based on these notions, the multi-warehouse allocation accuracy (or balance ra… |
| 9 | `2606.29366 §4 A Family of OR Formulations（fulltext.md L100）` | In this section, three complementary mixed-integer formulations are developed,… |
| 10 | `2606.29366 §4.4 Modular Heterogeneous Constraints（fulltext.md L204）` | real-world decision instances often involve heterogeneous operational rules, s… |
| 11 | `2606.29366 §4.5 Penalty-Based Relaxation（fulltext.md L294）` | A natural consequence of heterogeneous side constraints is that strict formula… |
| 12 | `2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L328）` | A key design choice is that the solver serves both as an execution engine and … |
| 13 | `2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L328）` | The hard-constraint model is solved first. If the solver reports infeasibility… |
| 14 | `2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L332）` | Our SFT dataset contains solver-verifiable PMC triples paired with structured … |
| 15 | `2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L332）` | Furthermore, we include targeted negative instances (e.g., non-linear modeling… |
| 16 | `2606.29366 §5.1 Problem–Model–Code Representation（fulltext.md L336）` | If the generated code is executable, and its executed allocation matches the g… |
| 17 | `2606.29366 §5.2 Learning-Based Formulation-Selection（fulltext.md L340）` | Their outputs are then combined through score-aware weighting, and feasibility… |
| 18 | `2606.29366 §6 Computational Evaluation（fulltext.md L358）` | For LLM training, we fine-tune the Qwen3-8B base model by minimizing the stand… |
| 19 | `2606.29366 §6.1 Data Pools for Post-Training（fulltext.md L362）` | We build PMC-style SFT data for all 18 formulations reported in Table 2, cover… |
| 20 | `2606.29366 §6.1 Data Pools for Post-Training（fulltext.md L362）` | Negative multi-warehouse inventory allocation SFT samples are further construc… |
| 21 | `2606.29366 §6.1 Data Pools for Post-Training（fulltext.md L362）` | In the final mixture, the proportions of multi-warehouse inventory allocation … |
| 22 | `2606.29366 §6.2 Code Generation Reliability（fulltext.md L368）` | The evaluation is conducted on a held-out test set containing more than 10,000… |
| 23 | `2606.29366 §6.2 Code Generation Reliability；表 4 Code Syntax Accuracy 行（fulltext.md L374）` | \| Code Syntax Accuracy \| 99.9% \| 100% \| |
| 24 | `2606.29366 §6.2 Code Generation Reliability；表 4 Code Execution Success Rate 行（fulltext.md L376）` | \| Code Execution Success Rate \| 99% \| 99.9% \| |
| 25 | `2606.29366 §6.2 Code Generation Reliability；表 4 Code Execution Failure Rate 行（fulltext.md L378）` | \| Code Execution Failure Rate \| 1% \| 0.1% \| |
| 26 | `2606.29366 §6.2 Code Generation Reliability（fulltext.md L380）` | On the held-out test set, the SFT+KTO variant attains 100% code syntax accurac… |
| 27 | `2606.29366 §6.2 Code Generation Reliability（fulltext.md L380）` | It is worth noting that these results are obtained under a constrained PMC pro… |
| 28 | `2606.29366 §6.3 Relaxation（fulltext.md L386）` | In this task, each predefined warehouse group is required to receive at least … |
| 29 | `2606.29366 §6.3 Relaxation（fulltext.md L386）` | A typical conflicting case arises when the groups are disjoint and their requi… |
| 30 | `2606.29366 §6.3 Relaxation；group minimum share 算例参数（fulltext.md L386）` | In our task, we set $R=252$, $(D_{1},\ldots,D_{8})=(18.1,9.7,20.3,44.7,17.4,11… |
| 31 | `2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L398）` | The fixed KI formulation improves 26 of the 29 production batches and achieves… |
| 32 | `2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）` | For example, in B19 and B20, the KI formulation improves accuracy by 0.8 and 2… |
| 33 | `2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）` | For all batches in which KI improves accuracy by more than 8 pp, the formulati… |
| 34 | `2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）` | Moreover, the number of batches with gains above 5 pp increases from 11 under … |
| 35 | `2606.29366 §7.1 Production-Batch Allocation Accuracy Evaluation（fulltext.md L402）` | KI outperforms the formulation-selection method in 7 of the 26 improved batche… |
| 36 | `2606.29366 §7.2 Generalization Evaluation（fulltext.md L440）` | For 5 task categories not included in post-training, we run inference with LLM… |
| 37 | `2606.29366 §7.2 Generalization Evaluation（fulltext.md L456）` | Our LLM generates syntactically correct and executable solver code for all 5 s… |
| 38 | `2606.29366 §7.2 Generalization Evaluation；表 6 Fixed allocation ratio 行（fulltext.md L452）` | \| Fixed allocation ratio \| Enforce a fixed allocation ratio between two wareho… |
| 39 | `2606.29366 §7.2 Generalization Evaluation；表 6 Minimum TID requirement 行（论文自己举的自然语言约束例子）（fulltext.md L454）` | \| Minimum TID requirement \| Enforce minimum TID \| ✓ \| ✗ \| |
| 40 | `2606.29366 §8 Conclusion（fulltext.md L462）` | Third, the framework can be extended from single-period allocation to multi-pe… |
| 41 | `2606.29366 首页页眉（投稿目标标注，非录用信息；fulltext.md L11）` | Journal: European Journal of Operational Research |

---

## 3. registry 备注核对：`'优先保 FBA 不断货'`

`papers_registry.json` 的 `decision_reason` 写：

> ORLA：求解器验证的 MIP 建模生成与选择，做多仓库存分配；支持自然语言约束（'优先保 FBA 不断货'）

**核对结论：该举例不是论文原话，且论文全文与「FBA」毫无关系。**

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| 全文是否出现 `FBA` | `grep -i "FBA" fulltext.md` | **0 命中** |
| 全文是否出现 `cross-border`（跨境） | `grep -i "cross-border" fulltext.md` | **0 命中** |
| 全文是否出现 `stockout` / `stock-out`（断货） | `grep -i "stockout\|stock-out" fulltext.md` | **0 命中** |

论文自己举的「自然语言 / 半结构化约束」措辞是：

- Abstract：`allocation requirements are often scenario-dependent and expressed in semi-structured or natural-language form rather than as ready-to-solve operations research (OR) formulations`（卡片 ⑥ #2，L16）
- §4.4：`ratio constraints, lower and upper allocation bounds, case-pack restrictions, group-level service requirements, and cardinality controls`（卡片 ⑥ #10，L204）
- §7.2 表 6 的任务名：`Enforce minimum TID`（`Minimum TID requirement`，AV ✗）、
  `Enforce a fixed allocation ratio between two warehouses`（`Fixed allocation ratio`，AV ✗）、
  `Enforce equal replenishment across warehouses`、`Enforce minimum order quantity per warehouse`、
  `Enforce mutual exclusivity among specified warehouses`（卡片 ⑥ #38/#39，L452/L454）

**因此**：registry 那句话是**业务侧对「组级服务要求 / 最小 TID 要求」的转译**，属于合理的选题理由，
但**不能当作论文引文**。卡片处理方式：在开头的「registry 备注核对结论」块里显式说明，
并在 ② 段用运营原话（「宁可 A 仓多备也不能断货，B 仓仓储费太贵尽量别压货」，**明确标注为本卡的场景转译**）
做业务映射 —— 论文措辞与业务话术在卡片里从不混写。

---

## 4. 与 2607.25956 的关系

registry `note` 写「与 2607.25956 高度重叠，建议合并为一张卡」。

- `grep "2607" fulltext.md` → **0 命中**：本文的正文与参考文献列表**都没有**引用 2607.25956。
- 按任务约束，本次**未抓取、未核验** 2607.25956 的内容。
- 因此本卡**不对两篇的关系作任何断言**：既不说「重叠」，也不说「不重叠」；
  只在卡片开头注明「registry 有此备注、本文未引用该文、合并时机由后续去重流程决定」。
- 合并若要执行，需要另一篇的萃取结果到手后再做 —— 本卡已写成自包含一页，
  不依赖 2607.25956 的任何结论。

---

## 5. 未能核验 / 论文未报告 —— 诚实边界清单

### 5.1 核验不了的

| 项 | 原因 | 处理 |
|---|---|---|
| **PDF 页码** | 本地只有 LaTeX→Markdown 存档，无 PDF 原件，页码无法机器复核 | ⑥ 段的出处一律写 `fulltext.md L<行号>`，不写 PDF 页码 |
| **venue 是否为 EJOR** | 全文首页页眉写着 `Journal: European Journal of Operational Research`，但 registry 的 `venue` 为空、`venue_tier: preprint`，本次也未在 EJOR 的录用信息中确认 | 按 R3 记为 `arXiv preprint`，并在卡片开头显式提示「不得当作 EJOR 论文引用」 |
| **`p2s-2026-0008` 与 arXiv 版的一致性** | 建档时下载的是 arXiv HTML 版（`https://arxiv.org/html/2606.29366v1`），未与后续版本比对 | 引文只对本存档负责；若论文出新版需重跑 `quote_check.py` |
| **另一篇重叠论文 2607.25956** | 任务约束：不得抓取 | 见第 4 节 |

### 5.2 论文**未报告**、故卡片不给数字的

以下均以 grep 该全文存档确认 0 命中（或只有符号无取值）：

| 主题 | 证据 | 卡片处理 |
|---|---|---|
| 断货惩罚 / 仓储费 / 持有成本的金额或权重 | `holding cost`、`shortage cost`、`penalty cost`、`warehouse rent` 各 0 命中 | 只写成业务侧待标定参数（⑤ 的 ROI 公式用贵司数据代入） |
| 跨境头程时效方差、清关不确定性 | `lead time` / `lead-time` / `cross-border` 各 0 命中；`uncertainty` 0 命中 | ①b 显式写「论文未讨论」 |
| 需求预测误差对分仓结论的传导 | 同上（论文把 $D_k$ 当已知输入） | ② 数据要求里提示为贵司侧风险 |
| 推理成本 / 求解耗时 / 硬件 | `latency`、`inference cost`、`GPU`、`A100`、`runtime`、`seconds`、`solve time`、`time limit` 各 0 命中 | 卡片不给任何成本量级 |
| 训练超参（epoch / 学习率 / batch size）与数据绝对规模 | `epoch`、`learning rate`、`batch size` 各 0 命中；只给了 Qwen3-8B、AdamW、NEFTune 与混合比例 | 只引论文给了的那几项 |
| 选择层的 $E$（专家数）、$K$、温度 $\kappa$ 的取值 | 论文只给符号 `$E$`、`Top-$K$`、`\kappa>0` | 卡片不给数字 |
| 生产批次的 SKU 覆盖数 | 论文只说 29 个批次 | 卡片只写 29 个批次 |
| 论文的独立 Limitations 章节 | `grep -i "limitations" fulltext.md` → 0 命中 | ①b 写明「论文没有独立的 Limitations 章节」，局限取自 §6.2 与 §8 |

### 5.3 卡片里**不是**论文事实的东西（避免误读）

- ③ 段的 4 个 python 代码块是**业务化精简实现**：不调用 SCIP、不调用 Qwen3-8B、不用 LightGBM，
  分别是「自写的可分离凸精确解」「规则式中文字段解析器（代替 LLM）」「岭回归（代替 LightGBM）」。
  KI 族（含 0-1 变量与大 M）**本卡未实现**，已在卡片中写明需交给真正的 MILP 求解器。
- 输出围栏里的所有数字（分仓量、平衡率、TID、松弛量）都来自**合成实例**，卡片已用两处 ⚠️ 声明
  「与论文读数不可互相印证」，且论文数字一律只出现在 ⑥ 段引文块里。
- ② 段的运营原话、仓名（FBA-美西ONT8 / FBA-美东BWI2 / 海外仓-长滩 / 海外仓-新泽西）、
  「≥8 周预测历史」等均为**场景设定**，不是论文内容。
- ①b 里「不要用 LLM 生成模型」的第 4 条（不可逆决策保留人工闸门）与「SD 族最优 ≠ 平衡率最高」
  一条，已在卡片中显式标注为**本卡判断 / 本卡实测观察**，不冒充论文结论。

### 5.4 K1 报告的位置

K1 的 JSON 产物写在临时目录（`/tmp/p2s_wip/k1.json`），**没有**写进仓库 ——
本次任务的约束是「除两个交付物外不得修改任何文件」。需要入库时按下面命令自行生成。

---

## 6. 复现命令

```text
# 1) K1 代码可执行（应输出 PASS）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py \
  --card paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md

# 2) 引文逐字核验（应输出 VERBATIM，n_fuzzy=0 / n_fabricated=0）
python3 paper2skills-skills/paper-审核/scripts/quote_check.py \
  --card paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md

# 3) K2 三合一门禁（G1 需要 K1 凭证，先落一份 JSON 再喂给 --k1）
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py \
  --card paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md \
  --json-out /tmp/k1_mwa.json
python3 paper2skills-skills/paper-审核/scripts/gate_check.py \
  --card paper2skills-vault/04-供应链/Skill-Multi-Warehouse-Allocation-LLM.md \
  --k1 /tmp/k1_mwa.json

# 4) 核验器自证
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
```

**本次实测结论**（2026-09-12）：

```text
K1   : ✅ PASS  Skill-Multi-Warehouse-Allocation-LLM.md#stitched(4块)  （L1–L5 全绿，pytest 6 passed）
引文 : ✅ VERBATIM  41/41 逐字（0 近似 / 0 伪造）
G1   : 1/1 通过 (100.0%)  红灯 0  黄灯 0
G2   : 1/1 通过 (100.0%)  红灯 0  黄灯 2   ← 2 条黄灯均为 frontmatter 日期里的 `12`
G3   : 1/1 通过 (100.0%)  红灯 0  黄灯 0   （具体业务信号 13 个；空泛表述 0 处；关联 Skill 6 张）
```

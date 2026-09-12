---
title: 证据链 — p2s-2026-0003（Skill-Causal-Budget-Allocation）
doc_type: evidence
module: 13-广告分析
paper_id: 2608.10182
registry_id: p2s-2026-0003
related_card: paper2skills-vault/13-广告分析/Skill-Causal-Budget-Allocation.md
fulltext: paper2skills-vault/papers/13-广告分析/p2s-2026-0003/fulltext.md
status: verified
created: 2026-09-12
updated: 2026-09-12
source: ai
---

# 证据链：2608.10182 *From Prediction to Incrementality: Causal Optimization for Large-Scale Targeting and Recommendation*

- **全文存档**：`paper2skills-vault/papers/13-广告分析/p2s-2026-0003/fulltext.md`
  （470 行 / 约 60 KB，`evidence_grade: A`）
- **正文无 venue 标注**（页眉为 `Conference: XXX; XXXX; XXXX` 占位），按 registry 口径记
  `venue: arXiv preprint` / `venue_tier: preprint`。
- **引文全部为逐字切片**：下列每条 `> 原文："…"` 都是从 `fulltext.md` 按起止锚点**直接切出的连续子串**，
  没有拼接、没有改写、没有跨段缝合。

## 0. 核验凭证（可复核的退出码 / stdout）

```
$ python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card <card>
✅ PASS         Skill-Causal-Budget-Allocation.md#stitched(2块)
单元总数 1 | ✅ PASS 1 | 🟡 ENV_BLOCKED 0 | 🔴 ORPHAN 0 | ❌ FAIL 0 | ⚪ 未执行 0
K1 执行率 = 100.0%

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card <card>
✅ VERBATIM   Skill-Causal-Budget-Allocation.md  38/38 逐字
共 1 张卡：1 通过，0 含伪造引文，0 无全文可核验

$ python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
1 真引文     : VERBATIM   连续度=1.0
2 纯伪造     : FABRICATED 连续度=0.124   ← 必须 FABRICATED
3 拼接引文   : FABRICATED 连续度=0.681 召回=1.0 拼接标记=True   ← 必须被拦下
4 排版差异   : VERBATIM   连续度=1.0   ← 必须 VERBATIM
✅ 自检通过：门禁能区分真引文 / 伪造 / 拼接

$ python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card <card> --only G2
G2 门禁：1/1 通过 (100.0%)  红灯 0  黄灯 5
（黄灯 5 条全部来自 v2 模板强制的 frontmatter 字段：`module: 13-广告分析` 与两处 `2026-09-12` 日期）
```

补充（不在本卡验收口径内，但一并记录）：G1 门禁 1/1 通过（红灯 0、黄灯 0，依据上表 K1 JSON），
G3 门禁 1/1 通过（红灯 0、黄灯 0）。

## 1. registry 头条数字核实

registry（`papers_registry.json` → `p2s-2026-0003`）`decision_reason` 记为：
「从预测转向增量：因果网络+贝叶斯 bandit+对偶 LP 做全局约束下的受限分配，线上主指标 +7.20%」。

**核实结论：数字确认，但口径需要收窄**（三处都指向同一个数）：

1. 摘要写的是 `$+7.20\%$ lift in the primary long-term-value metric`（Q1）；
2. §1 贡献列表写的是 `$+7.20\%$ lift in the primary KPI ($p=0.041$)`；
3. §5.5 写的是 `$+7.20\%$ lift ($p=0.041$, 95% CI: $[0.31\%,14.09\%]$)`，实验是
   **LinkedIn Feed marketing traffic 上的八周线上 A/B**（Q2）。

需要收窄的三点：
- 指标是 **long-term-value 指标**，不是 GMV / ROAS / 转化率；
- 场景是 LinkedIn 站内营销触达，**不是**母婴 / 跨境电商；
- 结果是**整条端到端策略**（因果打分 + 探索 + LP + withhold 规则）的系统级效果，论文明确说明
  不能拆给单个组件（Q3）。

## 2. 卡片正文数字 → 出处对照表

| 卡片中的数字 | 含义 | § 位置 | 引文键 |
|---|---|---|---|
| `+7.20%` | 线上端到端策略在**主长期价值指标**上的增量（registry 头条数字） | §Abstract / §5.5 | Q1 / Q2 |
| `p=0.041` | 上述提升的显著性水平 | §5.5 | Q2 |
| `95% CI [0.31%, 14.09%]` | 上述提升的置信区间 | §5.5 | Q2 |
| `50/50` | 线上实验的随机分臂比例 | §5.4 | Q24 |
| `50`（次模型更新） | 多轮模拟中 Bandit 版开始反超 greedy 版的位置（"roughly"） | §4.3.2 | Q19 |
| `0.857` → `0.826` | 去掉 dense 特征后 outcome AUROC 的变化 | §A.3 | Q22 |
| `99%`（以上） | 对偶热启动在稳定输入分布下达到当前最优解的比例 | §2.3.1 | Q8 |
| `7` 天 / `30` 天 | 处理窗长度 / 最短后续观察窗长度 | §5.1 | Q11, Q12 |
| `200` 轮 / `30` 次 | 多轮模拟的轮数 / 计算置信区间的重复次数 | §4.2 / 图 2 注 | Q17, Q18 |
| `34` → `5` | OBD 原始商品数 → 映射后的动作数（含"不推荐"） | §4.1 | Q20 |
| `400`K | 离线实验与 OBD 的成员规模量级 | §4.1 / §4.2 | Q20, Q16 |
| `$0.1` | 每个推荐/定向动作的成本设定 | §4.1 | Q21 |
| `8` × `5` | 消融配置数 × 每个配置重复次数 | §4.4 | Q15 |
| `3.5` | 内点法复杂度的指数（对偶分解为线性/迭代） | §2.3.1 | Q6 |
| `10^-3` / `<0.1%` | 平滑参数 γ 的选取规则 / ridge 扰动占目标的比例 | §2.3.1 | Q7 |

对照表之外的数字只有三类：v2 模板强制的 frontmatter 字段（`module`、`created`/`updated` 日期）、
交叉引用键（`Q1`…`Q39`，字母在前不构成数字断言）、以及 ③ 段代码块内的**合成示例数据**
（代码块在 G2 中会被剥离，不属于事实断言）。

## 3. 逐字引文全表（38 条，与卡片 ⑥ 段一一对应）

### A. 线上结果与可外推性

> 原文："The end-to-end treatment policy delivered a statistically significant $+7.20\%$ lift in the primary long-term-value metric, demonstrating the feasibility of production-scale causal optimization under business constraints."
> 出处：2608.10182 §摘要 Abstract｜Q1

> 原文："Measured against long-term-value metrics over an eight-week period, the treatment arm achieved a statistically significant $+7.20\%$ lift ($p=0.041$, 95% CI: $[0.31\%,14.09\%]$)."
> 出处：2608.10182 §5.5 Results｜Q2

> 原文："The online result therefore measures the system-level impact of the complete production policy, while the offline studies examine individual mechanisms under controlled settings."
> 出处：2608.10182 §5.5 Results｜Q3

> 原文："Members were randomly assigned 50/50 to the two experiment arms."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q24

> 原文："We also distill production lessons on causal training-data construction and cost and delivery control, which were critical to successful deployment."
> 出处：2608.10182 §摘要 Abstract｜Q31


### B. 方法：因果估计 / 探索 / 对偶 LP

> 原文："Incremental optimization requires estimating user-level treatment effects that quantify the expected lift from a targeting or recommendation action."
> 出处：2608.10182 §2.1 Incremental Modeling｜Q36

> 原文："Two standard assumptions identify $\tau(X)$ from observational data: unconfoundedness, $\{Y(0),Y(1)\}\perp\!\!\!\perp T\mid X$, and overlap, $0<e(X)<1$ for all $X$, where $e(X)=P(T=1\mid X)$ is the propensity score."
> 出处：2608.10182 §2.1 Incremental Modeling｜Q9

> 原文："Thus exploration guarantees positivity conditional on the feasible action."
> 出处：2608.10182 §2.2 Neural Bandit Exploration｜Q10

> 原文："We approximate the posterior over network parameters by a Gaussian centered at the MAP solution $\hat{\theta}_{\mathrm{MAP}}$ via the linearized Laplace approximation (LLA)"
> 出处：2608.10182 §2.2.1 Neural Thompson sampling via Laplace approximation｜Q37

> 原文："In the absence of shared allocation constraints, the resulting policy is equivalent to Thompson sampling: for each user, it selects the feasible action that maximizes the sampled incremental reward."
> 出处：2608.10182 §2.3 Large-scale Allocation with Constraints｜Q33

> 原文："Suppressing the round index, we relax $x_{u,i,t}$ to represent an action probability and solve the resulting large-scale problem using a smoothed dual-decomposition method (Basu et al., 2020)."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q5

> 原文："The solver then maximizes $g_{\gamma}$ over the $K$-dimensional dual with Nesterov-accelerated ascent, giving per-iteration cost linear in $|\mathcal{U}|\cdot|\mathcal{I}|$ versus $O((|\mathcal{U}||\mathcal{I}|)^{3.5})$ for interior-point methods."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q6

> 原文："The regularization $\gamma$ is picked as the largest value satisfying $\frac{\gamma\,\hat{x}^{T}\hat{x}}{2\,|c^{T}\hat{x}|}<10^{-3}$, so the ridge perturbation contributes $<0.1\%$ of the objective and the perturbed optimum is practically indistinguishable from the true LP optimum; when $A$ is ill-conditioned across constraint scales, we apply Jacobi row preconditioning."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q7

> 原文："At production scale, each round contains tens of millions of users and hundreds of items, so its batch has $|\mathcal{U}|\times|\mathcal{I}|$ variables and is intractable for general-purpose solvers."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q4

> 原文："In steady state, we warm-start the dual from the previous period’s $\lambda^{*}$, which under stable input distributions (KS-tested) achieves over 99% of the current optimum and also serves as an SLA fallback when the solver does not converge in time."
> 出处：2608.10182 §2.3.1 Scalability via Dual Decomposition｜Q8

> 原文："The framework also supports sequential context and multi-outcome, attribute-conditioned scoring through a Transformer encoder and outcome embeddings."
> 出处：2608.10182 §摘要 Abstract｜Q32


### C. 离线验证的规模与结果

> 原文："We utilize the random-policy subset, synthetically mapping the original 34 products that were recommended to $>400$K users to 5 distinct actions that are relevant to the production incrementality use case: recommendation to one of four business lines or no-recommendation."
> 出处：2608.10182 §4.1 Dataset｜Q20

> 原文："The no-recommendation action is a key difference between incremental and non-incremental targeting."
> 出处：2608.10182 §4.1 Dataset｜Q34

> 原文："Finally, we assign a cost of $0.1 to each recommendation/targeting action to capture operational and bidding costs."
> 出处：2608.10182 §4.1 Dataset｜Q21

> 原文："We solve the offline simulation LP with Google OR-Tools, which is tractable at this scale (${\sim}400$K members, 5 actions) and convenient for reproduction; at full production scale we use the dual-decomposition method described in Section 2.3 instead."
> 出处：2608.10182 §4.2 Setup｜Q16

> 原文："We then simulate deployment over $T=200$ rounds using the prediction set as the environment: at each round, each method selects actions for the current batch, observes the realized rewards from its own recommendations, updates its training data accordingly, and incrementally updates the model before the next round."
> 出处：2608.10182 §4.2 Setup｜Q17

> 原文："Figure 2. Average cumulative reward and 95% confidence intervals after 200 rounds of feedback. The confidence intervals are calculated from 30 simulation runs."
> 出处：2608.10182 §Figure 2 图注（§4.3.2）｜Q18

> 原文："This reflects the short-term cost of exploration in exchange for long-term gains: after roughly 50 model updates, the Bandit Incremental Model begins to outperform both greedy variants."
> 出处：2608.10182 §4.3.2 Multi-turn evaluation｜Q19

> 原文："We tested 8 configurations on a fixed train/validation snapshot, each repeated 5 times with common hyperparameters."
> 出处：2608.10182 §4.4 Ablation Study｜Q15

> 原文："As shown in Table 2, incremental scores lead to higher rewards compared to propensity scores."
> 出处：2608.10182 §4.3.1 Single-turn evaluation｜Q38

> 原文："We also provide the corresponding send volumes in Table 3, where it can be seen that incremental targeting leads to a higher percentage of no-recommendations as the engine is able to identify members likely to convert organically."
> 出处：2608.10182 §4.3.1 Single-turn evaluation｜Q39

> 原文："Removing dense features is neutral or beneficial for uplift AUUC despite reducing outcome AUROC from 0.857 to 0.826 and minimally affecting treatment AUROC."
> 出处：2608.10182 §A.3 Uplift Results｜Q22


### D. 生产落地：数据构造、投递控制、实验脚手架

> 原文："For each member, production samples $D=R-(W_{C}+W_{T}+W_{D})+U$, where $U\sim\operatorname{Uniform}\{0,\ldots,W_{D}-1\}$, $W_{D}=90$ days, $W_{T}=7$ days, and $W_{C}=30$ days. Thus $D\in[R-127,R-38]$."
> 出处：2608.10182 §5.1 Training-Data Construction for Causal Estimation｜Q11

> 原文："We set $T=1$ when at least one qualifying email send, on-platform impression, or video view occurs in $[D,D+7)$. We set $Y=1$ when the corresponding business-line or product-family conversion occurs in $[D+7,R]$, and $Y=0$ otherwise."
> 出处：2608.10182 §5.1 Training-Data Construction for Causal Estimation｜Q12

> 原文："Randomizing $D$ avoids a last-touch label, captures long-term action effects, and preserves variable-length histories."
> 出处：2608.10182 §5.1 Training-Data Construction for Causal Estimation｜Q35

> 原文："Let $B$ be the committed budget, $S_{t}$ the cumulative realized spend, and $q(t/H)$ the desired cumulative pacing curve over a horizon $H$, with $q(0)=0$ and $q(1)=1$."
> 出处：2608.10182 §5.2 Cost and Delivery Control｜Q14

> 原文："An early A/B test run without the above controls showed the treatment arm under-delivering: the causal policy withholds and reallocates sends, lowering treatment impressions relative to control. A raw total-bookings comparison then penalizes the treatment for delivering less rather than for choosing worse, confounding policy quality with delivery volume. Constraining treatment to the BAU impression and cost envelope equalizes delivery across arms and restores a clean, like-for-like read."
> 出处：2608.10182 §5.2 Cost and Delivery Control｜Q13

> 原文："A meaningful policy difference is that the treatment arm can withhold a send whenever the predicted incremental value is negative or no feasible positive-incremental option exists."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q30

> 原文："First, when out-of-scope campaigns are suppressed via an exclusion segment defined by dynamic criteria (company, locale, activity), members drift across arms in a way correlated with platform activity, contaminating the intent-to-treat contrast."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q25

> 原文："Second, when arms are assembled from shared segment definitions, a control audience can silently inherit a treatment send decision through a reused sub-segment."
> 出处：2608.10182 §5.4 Agentic Experimentation｜Q26


### E. 论文自承局限

> 原文："Although the complete research architecture contains several losses and optional modules, the serving path is modular rather than a jointly tuned monolith: outcome embeddings are used only for multi-product scoring, LLA is applied after supervised training to the last layer, and the LP consumes exported scores independently of model training. This separation lets each module be disabled or validated without retraining the rest of the decision pipeline, while the remaining tuning burden and feature sensitivity are limitations of the current shared-representation model."
> 出处：2608.10182 §3.4 Complete Loss Function｜Q27

> 原文："Because this diagnostic uses one held-out product and sampled embeddings rather than prospective real launches, it does not establish zero-shot production effectiveness."
> 出处：2608.10182 §A.4 Outcome-Embedding Diagnostic｜Q28

> 原文："This remains a hypothesis rather than an automated feature-selection mechanism; architectures such as FlexTENet (Curth and van der Schaar, 2021) could explicitly separate the two subspaces."
> 出处：2608.10182 §A.3 Uplift Results｜Q23

## 4. 未能核验 / 只能定性陈述的清单

本卡在写作中**主动放弃**了以下内容，原因是无法逐字核验或论文本身没给：

1. **母婴 / 跨境电商场景的任何数字** —— 论文完全没有这个话题。卡片 ② 段的市场、品类、渠道
   全部是业务背景（由业务方给定），**没有任何数字来自论文**；ROI 公式里的 `B`、`m`、`C_sys`
   是符号而非数值。
2. **工程成本 / 人力投入 / 求解器耗时** —— 论文未报告任何量级，卡片 ⑤ 明确标注"需企业自估"。
3. **`+7.20%` 的组件级归因** —— 论文明确拒绝拆分（Q3）。卡片因此写明"不要把它当成 LP 层单独值 7.20%"。
4. **Table 2 / Table 3 的具体数值**（平均收益、平均成本、平均净收益、各业务线发送量）——
   HTML→Markdown 转换后这些表格的**列序错位**（数值与行标签被拆到同一行的不同位置），
   逐字引用会误导读者，故卡片只引用其**定性结论**（Q38 / Q39），不引用表内数字。
5. **Table 1 的各业务线均价与体量上下界**（`$5/$10/$10/$200`、`5%–10% / 5%–30% / 30%–50%`）——
   行数据在存档中完整，但本卡的业务场景不使用 OBD 的业务线口径，故未引用。
6. **超参具体值**（学习率、epoch 数、batch size、hidden size）—— 附录 A.1 只说这些在各配置间
   "保持不变"，未给数值；卡片未给。
7. **AUUC / AUROC 的具体数值曲线** —— 论文只在图 3 给误差棒，正文未列数字；卡片只引用
   §A.3 正文给出的 `0.857 → 0.826`（Q22）。
8. **平台站内广告与独立站的双边身份对齐方案** —— 论文未讨论；卡片 ①b 显式标注"论文未讨论"。
9. **与作者实现的等价性** —— registry 备注"无公开代码，需自研"；K1 只证明**卡片自带代码**
   可执行（L5 断言全绿），**不证明**它与论文未公开的生产实现一致。
10. **venue** —— 存档正文无 venue 标注（页眉是 `Conference: XXX; XXXX; XXXX` 占位符），
    故按 registry 口径记 `arXiv preprint` / `preprint`。

## 5. 复现方式

```bash
cd /Users/lute/project/paper_to_skills
CARD=paper2skills-vault/13-广告分析/Skill-Causal-Budget-Allocation.md
python3 paper2skills-skills/paper-萃取/scripts/verify_skill_code.py --card "$CARD"
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card "$CARD"
python3 paper2skills-skills/paper-审核/scripts/gate_check.py --card "$CARD" --only G2
```

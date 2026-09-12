# evidence.md — p2s-2026-0002 证据链

> 论文：**FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation**
> arXiv：`2608.11675`　venue：CIKM 2026（`venue_tier: CCF-B`）
> 全文存档（本文件所有引文的核验底本）：`paper2skills-vault/papers/13-广告分析/p2s-2026-0002/fulltext.md`
> 对应 Skill 卡片：`paper2skills-vault/13-广告分析/Skill-Funnel-Causal-Coupon-Allocation.md`
> registry 事实（`07-资源库/papers_registry.json`）：`data_availability: partial`；
> note：「CIKM 2026；假设 RCT，平台卖家需退化为观察数据」

---

## 1. 引文是怎么产生的（可复现步骤）

本文件与卡片 ⑥ 段里的每一条 `> 原文："..."` 都不是手抄的，而是**按 (行号, 起始标记, 结束标记)
从 `fulltext.md` 里机械切出的连续子串**，因此不存在「两段缝成一句」的拼接风险。

核验命令（两条都必须报绿）：

```bash
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card paper2skills-vault/13-广告分析/Skill-Funnel-Causal-Coupon-Allocation.md
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --card paper2skills-vault/papers/13-广告分析/p2s-2026-0002/evidence.md
python3 paper2skills-skills/paper-审核/scripts/quote_check.py --selftest   # 先证明核验器本身能抓伪造/拼接
```

> ⚠️ 口径提醒：`gate_check.py` 的 G2 只读取**卡片同目录**的 `evidence.md`
> （即 `13-广告分析/evidence.md`）。本文件位于论文档案目录，**不参与**卡片的 G2 计算——
> 卡片本身已自带 ⑥ 段全部引文，G2 是自足的。本文件是给人看的完整证据链与审计留痕。

---

## 2. 数字 → 出处总表

表内数值均取自第 3 节的逐字引文；只有「论文报告了但本卡未引用」一栏是登记项，不是断言。

| # | 数字 | 论文语境 | 出处 |
|---|---|---|---|
| 1 | `μ_gmv = μ_conv · μ_val` | 漏斗组合（Eq. 5） | §Abstract / §4.1 |
| 2 | `Var(Y^g) = p·σ_v² + p(1-p)·μ_v²` | 方差分解（Eq. 7） | §4.2 |
| 3 | `1 / (1 + (1-p)·μ_v²/σ_v²)` | 理想化领先阶 MSE 比（Eq. 8） | §4.2 |
| 4 | 0.615 / 0.613 | AUUC_GMV：EFIN 最高，本方法第二（一个种子标准差内） | §5.2 Table 3 |
| 5 | 0.048 / 0.058 | PEHE_CVR：DualHeadNet / 本方法 | §5.2 Table 3 |
| 6 | 60 以上 → 0 | 漏斗违例率：直接 GMV 回归 vs 硬漏斗组合 | §5.3 Table 4 |
| 7 | 18–48 | 零膨胀压力测试的 PEHE_GMV 降幅（%） | §5.3 Table 5 |
| 8 | 4.6–45.4 | 压力测试扫描到的转化率区间（%） | §5.3 Table 5 |
| 9 | 24.3 / 48.3 | 峰值收益所在的转化率水平（%）与对应降幅（%） | §5.3 Table 5 |
| 10 | 3.92 / 3.07 | 紧预算（0.05）下锚定拉格朗日 vs 随机分配的 ΔROI | §5.5 Table 7 |
| 11 | 3–15 pp | 联合 conformal 覆盖超出名义 1−α 的幅度 | §5.6 Table 8 |
| 12 | 0.05 / 0.10 / 0.20 | conformal 名义水平 α 的三个取值 | §5.6 Table 8 / §4.3 |
| 13 | 30 → 324 | 训练墙钟时间（十万 → 一百万用户，秒） | §5.7 Table 10 |
| 14 | 0.13 | 百万用户、八档下拉格朗日对偶更新的耗时（秒） | §5.7 |
| 15 | 4.98 百万 / 2.79 百万 | 工业 RCT 曝光记录数 / 去重用户数 | §5.8 |
| 16 | 50 千 / 4.93 百万 | 每种子的训练片 / 留出片规模 | §5.8 |
| 17 | 0.2–0.3 | 论文当作敏感性参数扫描的平台佣金率 γ | §5.8 |
| 18 | 3.3–5.0 | 盈亏平衡 ΔROI = 1/γ 的对应区间 | §5.8 |
| 19 | 7/7 | 工业锚点上本方法种子平均 ΔROI 最高 | §7 / §5.8 Table 11 |
| 20 | 0.18–0.21 | 中到大锚点区间领先第二名的 ROI 单位数 | §5.8 |
| 21 | 25–60 | 上述领先幅度所在的锚点区间（ΔGMV%） | §5.8 Table 11 |
| 22 | 0.747 / 0.739 | Hillstrom 上营收导向 ranker 的 AUUC_GMV（反例） | §5.2 |
| 23 | 0.25 / 0.6 | 冲突筛查峰值 F1 与对应注入相关系数 | §5.4 Table 6 |
| 24 | 8 / 八档 / 0–14 | 合成数据基线转化率（%）、档位数、折扣区间（%） | §5.1 |

### 论文报告了、但本卡未引用的数字（登记项，供后续补充）

以下数字在论文中存在，本卡出于「不堆砌」的考虑没有写进正文，**也未做任何断言**：

- §5.2 Table 3 其余方法的 AUUC_GMV / PEHE_GMV / PEHE_CVR / ATE_err 全量行；
- §5.3 Table 4 的 PEHE_GMV 三档样本量（A 模式 25.94 / 25.48 / 25.80）与 B/C/D 各模式；
- §5.4 Table 6 的 precision / recall / F1 四个相关水平；
- §5.6 Table 8 的边际覆盖与联合覆盖逐格数值、Table 9 的区间宽度（`w_τg` 量级为 10 的四次方货币单位）；
- §5.7 Table 10 的 conformal 校准与 LP 求解耗时逐行；
- §5.8 Table 11 六个模型 × 七个锚点的逐格 ΔROI 与标准差；
- §5.8 的 LP 前沿最右端延伸（六个模型各自的最大 realized ΔGMV%）。

---

## 3. 逐字引文（与卡片 ⑥ 段同源）

### 3.1 漏斗结构、识别与理论

> 原文："We propose FunnelCausalNet, an uplift estimator that couples a binary conversion head with a nonnegative conditional-value head under the funnel composition $\mu_{\mathrm{gmv}}=\mu_{\mathrm{conv}}\,\mu_{\mathrm{val}}$."
> 出处：2608.11675 §Abstract

> 原文："We identify $\tau^{c}_{t}$ and $\tau^{g}_{t}$ under randomized $T\mid X$ (RCT) as in standard analyses; we do not claim identification from purely observational logs."
> 出处：2608.11675 §3（Causal targets）

> 原文："$\mathrm{Var}(Y^{g}\mid X{=}x,T{=}t)=p\,\sigma_{v}^{2}+p(1-p)\,\mu_{v}^{2}.$"
> 出处：2608.11675 §4.2（Eq. 7）

> 原文："The first term is the within-converter variance; the second is the Bernoulli switching variance contributed by the zero mass."
> 出处：2608.11675 §4.2（Eq. 7 解读）

> 原文："Within these idealized assumptions, the ratio in (8) is below one whenever $(1{-}p)\mu_{v}^{2}/\sigma_{v}^{2}{>}0$, and shrinks as the zero mass $(1{-}p)$ grows or $\mu_{v}$ dominates $\sigma_{v}$."
> 出处：2608.11675 §4.2（Eq. 8 的 operational regime）

> 原文："Eq. (8) is an idealized pointwise variance comparison, not a universal optimality theorem or a guarantee for CATE ranking."
> 出处：2608.11675 §4.2（Eq. 8 的适用限制）

> 原文："We apply additive shifts estimated from RCT arm-wise averages on a held-in slice before forming rewards fed to the allocator, improving $\Delta\mathrm{ROI}$-style objectives without retraining."
> 出处：2608.11675 §4.4（Anchoring）

### 3.2 合成数据上的估计质量与漏斗消融

> 原文："EFIN attains the highest AUUC_GMV ($0.615$); FunnelCausalNet ranks second ($0.613$, within one seed standard deviation)"
> 出处：2608.11675 §5.2（Table 3）

> 原文："PEHE_CVR is led by DualHeadNet ($0.048$); FunnelCausalNet ($0.058$) remains competitive, confirming that funnel coupling does not destroy conversion-head identifiability."
> 出处：2608.11675 §5.2（Table 3）

> 原文："Hard coupling achieves the lowest PEHE_GMV at $10\mathrm{K}$, $20\mathrm{K}$, and $100\mathrm{K}$ samples, while the funnel-violation rate of A remains at $\gtrsim 60\%$ versus $0\%$ for C."
> 出处：2608.11675 §5.3（Table 4）

> 原文："Funnel composition reduces PEHE_GMV by $18$–$48\%$ across the tested $\hat{p}\in[4.6\%,45.4\%]$ range, with peak benefit at moderate-high zero inflation"
> 出处：2608.11675 §5.3（Table 5）

> 原文："generator parameters (baseline conversion ${\approx}8\%$, eight tiers $0\%$–$14\%$) fall inside operationally common e-commerce coupon ranges"
> 出处：2608.11675 §5.1（Semi-synthetic calibration disclosure）

### 3.3 预算分配与不确定性层

> 原文："The anchored-Lagrangian pipeline attains higher $\Delta\mathrm{ROI}$ than random allocation under tight budgets—for example, $3.92$ versus $3.07$ at $B/B_{\mathrm{free}}{=}0.05$—with lower realized cost and competitive incremental GMV."
> 出处：2608.11675 §5.5（Table 7）

> 原文："Joint empirical coverage consistently exceeds nominal $1{-}\alpha$ by 3–15 pp across $\alpha\in\{0.05,0.10,0.20\}$"
> 出处：2608.11675 §5.6（Table 8）

> 原文："recommend wider nominal $\alpha\in[0.10,0.20]$ when widths must remain actionable, and pair intervals with anchored point estimates when feeding optimizers, because marginally valid lower-conformal bounds for $\tau^{g}$ at narrow $\alpha$ can be so pessimistic under zero inflation that budgeted LCB policies collapse to all-control assignments (Sec. 5.5)."
> 出处：2608.11675 §4.3（Deployment stance）

> 原文："Peak F1 reaches $\approx 0.25$ at $\rho_{\mathrm{conf}}{=}0.6$"
> 出处：2608.11675 §5.4（Table 6）

> 原文："Training scales sublinearly between $N{=}10^{4}$ and $10^{6}$ in our sweeps ($\approx 30\,\mathrm{s}\to 324\,\mathrm{s}$, $\sim 10\times$ wall-clock for $100\times$ users). Conformal calibration stays below one second even at $N{=}10^{6}$. Lagrangian dual updates stay near $0.13\,\mathrm{s}$ at one million users for $K{=}8$, whereas dense LP relaxations exceed tens of seconds already at $N{=}10^{5}$ and fail at larger $N$ due to memory."
> 出处：2608.11675 §5.7（Table 10）

### 3.4 工业多臂 RCT

> 原文："totaling $\approx 4.98\times 10^{6}$ exposure records from $\approx 2.79\times 10^{6}$ distinct users overall. For each of three permutation seeds we shuffle the full table, take the first $N_{\mathrm{train}}{=}50\mathrm{K}$ records for training, and retain the remaining $\approx 4.93$M exposure records per seed for evaluation."
> 出处：2608.11675 §5.8

> 原文："We treat the platform commission rate $\gamma$ as a sensitivity parameter over $[0.2,0.3]$, a band typical of online travel/coupon programs; the break-even point is $\Delta\mathrm{ROI}\!=\!1/\gamma\!\in\![3.3,5.0]$. Operating below the band ($\Delta\mathrm{ROI}\!<\!3$) means incremental commission no longer offsets subsidy cost"
> 出处：2608.11675 §5.8（Practical operating regime）

> 原文："at mid-to-large anchors ($25\%$–$60\%$) FunnelCausalNet’s mean exceeds the second-best by $0.18$–$0.21$ ROI units."
> 出处：2608.11675 §5.8（Table 11）

> 原文："Per-anchor paired-bootstrap CIs over three permutation seeds include $0$, so individual rows are not formally significant."
> 出处：2608.11675 §5.8（Table 11 caption）

> 原文："on industrial multi-arm RCT logs, FunnelCausalNet has the highest seed-averaged mean LP-frontier $\Delta\mathrm{ROI}$ at all $7/7$ reported anchors, although their correlation and the three permutation splits preclude an independent-anchor significance claim."
> 出处：2608.11675 §7（Conclusion）

### 3.5 适用边界与可复现性

> 原文："Empirically, revenue-focused rankers RERUM ($0.747$) and DualHeadNet ($0.739$) lead AUUC_GMV on Hillstrom, while all multi-tier funnel-aware deep models (DESCN, ECUP, FunnelCausalNet) underperform."
> 出处：2608.11675 §5.2（Public-RCT scope boundary）

> 原文："All causal interpretations assume RCT-like randomized assignment, not observational identification."
> 出处：2608.11675 §6（Identification scope and limitations）

> 原文："E7 uses record-level permutation splits, so repeated users can appear in both training and hold-out slices"
> 出处：2608.11675 §6（record-level permutation splits）

> 原文："Finally, the observed coupon arms are discrete offers: the model does not exploit smoothness or monotonicity across a continuous coupon dose"
> 出处：2608.11675 §6（discrete offer arms）

> 原文："The current version does not include a public code artifact."
> 出处：2608.11675 §5.9（Reproducibility）

> 原文："quantitative effect sizes, per-bucket exposure ratios, and ablation traces remain unavailable under the platform agreement"
> 出处：2608.11675 §5.8（Online consistency check）

### 3.6 补充逐字摘录：关键表格行（带具体数值）

> 原文："| FunnelCausalNet | 0.613 | 31.18 | 0.058 | 21.67 |"
> 出处：2608.11675 §5.2 Table 3（FunnelCausalNet 行）

> 原文："$-1.5$ $24.3\%$ $39.22{\pm}8.96$ $\mathbf{20.29}{\pm}5.70$ $\mathbf{+48.3\%}$"
> 出处：2608.11675 §5.3 Table 5（峰值收益行）

> 原文："$0.05$ | | funnel_ip_anchored | 9 072 | 2 315 | 3.92 |"
> 出处：2608.11675 §5.5 Table 7（紧预算下锚定分配行）

> 原文："$0.05$ | | baseline_random | 8 783 | 2 864 | 3.07 |"
> 出处：2608.11675 §5.5 Table 7（紧预算下随机分配行）

> 原文："$10\mathrm{K}$ | | 30.70 | 0.020 | 0.0019 | 0.429 |"
> 出处：2608.11675 §5.7 Table 10（十万用户行）

> 原文："$10^{6}$ | | 324.01 | 0.338 | 0.127 | — |"
> 出处：2608.11675 §5.7 Table 10（百万用户行）

> 原文："$10$ $4.13\!\pm\!0.62$ $4.77\!\pm\!0.39$ $4.11\!\pm\!0.08$ $3.13\!\pm\!\mathrm{n/a}$ $4.84\!\pm\!0.45$ $\mathbf{4.94\!\pm\!1.08}$"
> 出处：2608.11675 §5.8 Table 11（最小锚点行）

---

## 4. 数据可得性判定（对应 registry 的 `partial` 与「平台卖家需退化为观察数据」）

| 结论 | 依据 |
|---|---|
| 论文的全部因果解释都**要求 RCT 式随机分配** | §3 Causal targets；§6 Identification scope |
| 论文的工业证据来自**私有酒店券 RCT**，数据不可再分发、结论不可独立复现 | §5.8；§5.9 Reproducibility |
| **不含公开代码 artifact** | §5.9 Reproducibility |
| 公开基准上**没有一致增益**（营收导向 ranker 更强） | §5.2 Public-RCT scope boundary |
| 在线 A/B 一致性检查**不提供量化细节** | §5.8 Online consistency check |
| 平台店铺（Amazon 等）**无用户级随机化** → 只能退化为分层弹性估计 + 地理/时间 holdback，卡片 ② 场景一已据此标注「不可得」 | §3 + §6（识别口径）+ 业务侧事实 |

---

## 5. 未能核验 / 论文未报告清单（如实登记，不补数）

1. **母婴出海、Amazon 站内、独立站链路的收益量级**：论文完全没有涉及，本卡不做任何外推。
2. **论文的在线实验 effect size**：论文明确说在平台协议下不可用（见 Q28），因此任何「线上提升 X%」的说法都无依据。
3. **表格里的逐格数值**（见第 2 节末栏）：存在于论文中，但本卡未逐字摘录，故正文不引用。
4. **命题的渐近假设是否在神经网络实现上成立**：论文自陈未验证（Q6），本卡照抄这一保留意见，不声称已成立。
5. **连续券剂量下的表现**：论文未做（Q23）。
6. **公开代码**：论文未提供（Q24），本卡 ③ 段代码是**按论文公式自行实现的最小可运行版本**，
   只复现结构（漏斗组合 / 方差分解 / MSE 比 / 拉格朗日分配 / 加性锚定），**不声称复现论文任何实验数值**。

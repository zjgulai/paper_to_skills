<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2608.11675
     paper_id : p2s-2026-0002
     source   : https://arxiv.org/html/2608.11675v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation

CCS: Information systems Recommender systemsCCS: Information systems Online advertisingCCS: Computing methodologies Causal reasoning and diagnosticsCCS: Computing methodologies Supervised learning by regression
Yu Zhang Affiliation: AMap Alibaba Group, Beijing, China email: yuanyu.zy@alibaba-inc.com , Zhihan Wang Affiliation: AMap Alibaba Group, Beijing, China email: yingchen.wzh@alibaba-inc.com , Guanlin Chen Affiliation: AMap Alibaba Group, Beijing, China email: cgl517639@alibaba-inc.com , Min Jiang Affiliation: AMap Alibaba Group, Beijing, China email: jiangmin.jiang@alibaba-inc.com and Shuai Li Affiliation: AMap Alibaba Group, Beijing, China email: lion.lis@alibaba-inc.com

###### Abstract.

Coupon campaigns aim to lift both conversion and revenue, but gross merchandise value (GMV) inherits a deterministic funnel structure from conversion and conditional order value and is typically zero-inflated and heavy-tailed. We propose FunnelCausalNet, an uplift estimator that couples a binary conversion head with a nonnegative conditional-value head under the funnel composition $\mu_{\mathrm{gmv}}=\mu_{\mathrm{conv}}\,\mu_{\mathrm{val}}$. Under explicit RCT, support, rate-gap, and cross-head covariance-control assumptions, we derive an idealized leading-order MSE-ratio comparison that identifies a variance regime in which funnel composition can reduce pointwise estimation variance; it is a regime heuristic rather than a guarantee for the shared-representation neural implementation. The estimator is paired with marginal split-conformal summaries on each outcome’s CATE (Bonferroni union for two-number joint coverage, treated as audit/monitoring bands) and a Lagrangian budgeted allocator that consumes RCT-anchored estimates for subsidy-aware ROI accounting. On semi-synthetic multi-tier Criteo-MT7 with oracle individualized treatment effects, FunnelCausalNet’s mean AUUC_GMV is within one seed standard deviation of the leading recent feature-interaction baseline among eleven baselines, and a controlled funnel-coupling ablation reduces GMV effect error against direct GMV regression by $18$–$48\%$ across the tested zero-inflation regimes. On de-identified industrial Hotel-Coupon RCT logs with $\approx\!4.9\!\times\!10^{6}$ hold-out exposure records per seed, RCT-consistent expected-outcome (EOM) evaluation sweeps full LP frontiers, and FunnelCausalNet attains the best seed-averaged mean $\Delta\mathrm{ROI}$ at $7/7$ correlated EOM anchors in $10\%$–$60\%$; we treat this as descriptive frontier consistency rather than independent-anchor significance. On sparse binary-spend public benchmarks, revenue-focused rankers can dominate uplift-curve proxies; we foreground this regime boundary explicitly.

###### Keywords:

uplift modeling; causal inference; heterogeneous treatment effects; coupon allocation; multi-arm randomized experiments; conformal prediction; Lagrangian relaxation; e-commerce

## 1. Introduction

Digital coupon programs aim to lift both the probability that a user converts and the revenue generated conditional on conversion. In practice, marketing teams often estimate conversion uplift and revenue-related quantities through loosely coupled pipelines—separate models for conversion probability and order value—and combine predictions downstream for targeting or budget allocation. This decoupled workflow ignores three structural features that routinely appear in coupon randomized controlled trials (RCTs).

(i) Funnel identity. Gross merchandise value (GMV) satisfies $Y^{g}{=}0$ whenever $Y^{c}{=}0$, so GMV decomposes algebraically into conversion mass and conditional spend. Treating GMV as an unconstrained continuous response under extreme zero inflation produces variance-dominated estimates of heterogeneous treatment effects (HTE) on revenue. (ii) Ranking divergence. Ordering users by estimated conversion uplift can disagree substantially with ordering by estimated GMV uplift. Under tight budgets, this inconsistency directly translates into lost incremental GMV relative to revenue-aligned objectives. (iii) Multi-tier action space with tier-specific conversion- and revenue-elasticities. Retailers choose among multiple discount tiers; in our industrial RCT logs different coupon strengths exhibit quantitatively different elasticities on conversion probability and on conditional spend—a heavier coupon may convert more users while also reshaping the spend distribution among converters in ways that a single binary lift cannot pin down. Decision-making therefore requires assigning limited subsidy budgets across users and arms, and a binary-encouragement formulation cannot answer “who should receive which tier” under explicit cost-of-promotion constraints.

These observations motivate funnel-aware multi-outcome uplift modeling: jointly estimate heterogeneous causal effects on conversion and GMV while respecting the funnel structure, quantify uncertainty to support conservative deployment, and feed estimates into scalable budgeted multi-tier allocation. We treat the contribution as an end-to-end stack—point estimates, audit-oriented uncertainty, and budgeted assignment with subsidy accounting—rather than a single estimator.

### Contributions

(1) Funnel-structured uplift estimation under zero inflation, with a regime-guided variance analysis. We present FunnelCausalNet, an estimator that couples a binary conversion head with a nonnegative spending head consistent with the deterministic zero mass on GMV. Under explicit assumptions on RCT identification, support, convergence rates, and covariance control (Sec. 4.2, Proposition 2), we derive an idealized leading-order MSE-ratio comparison for the high-zero-inflation regime. The comparison provides directional guidance; it does not guarantee dominance for the shared-representation neural implementation or across datasets. (2) Budgeted multi-tier allocation with RCT anchoring. We combine funnel estimates with Lagrangian relaxation for large-scale multi-tier assignment under subsidy budgets, and absorb additive shifts estimated from RCT arm averages on a held-in slice to mitigate systematic GMV-level bias before forming allocator rewards. (3) Auditable joint uncertainty layer. We supply marginal split-conformal intervals on each outcome’s CATE summary together with a Bonferroni union (a finite-sample valid two-number joint coverage statement) and a Top-$K$ boundary screen flagging users with unstable cross-objective rankings; these are positioned as audit/monitoring bands for compliance review rather than allocator inputs (Sec. 4.3).

Empirical scope. We evaluate against eleven baselines—meta-learners (S-, T-, X-Learner (Künzel et al., 2019)), causal forests (Wager and Athey, 2018; Athey et al., 2019), a dual-head network, CFRNet (Shalit et al., 2017), DragonNet (Shi et al., 2019), EFIN (Liu et al., 2023), DESCN (Zhong et al., 2022), ECUP (Huang et al., 2024), and RERUM (He et al., 2024)—on (i) semi-synthetic Criteo-MT7 with oracle individualized treatment effects (ITEs), (ii) the public Hillstrom (Hillstrom, 2008) RCT (3-arm encouragement collapsed to a single send-vs-control contrast as in standard uplift evaluation), (iii) controlled funnel ablations, (iv) joint conformal coverage and computational scaling to $10^{6}$ users, and (v) a large industrial Hotel-Coupon multi-arm RCT with $\approx\!4.9\times 10^{6}$ hold-out exposure records per seed under RCT-consistent expected-outcome (EOM) evaluation (Yan et al., 2023) that sweeps full LP frontiers, with $\Delta\mathrm{GMV\%}$ anchors chosen so realized $\Delta\mathrm{ROI}$ values straddle the break-even band predicted by a commission-rate sensitivity scan (Sec. 5.8).

Honest benchmarking. Funnel coupling targets multi-tier RCT regimes in which heterogeneous coupon strengths induce distinct conversion- and revenue-elasticities. Public benchmarks built around a single send-vs-control encouragement (e.g., Hillstrom) instantiate a different decision problem—there is no tier-strength axis along which conversion- and spend-elasticities can differ—so revenue-focused rankers tuned for binary AUUC proxies can score higher. We surface this scope boundary in Section 6 rather than suppressing it. The anchored allocator and the audit conformal layer nonetheless provide a uniform pipeline aligned with subsidy accounting in all regimes.

## 2. Related Work

#### Uplift modeling and heterogeneous treatment effects.

Classical uplift estimators include meta-learners (S/T/X) (Künzel et al., 2019), tree-based CATE estimators such as causal forests (Wager and Athey, 2018; Athey et al., 2019), R-learner-style residualization (Nie and Wager, 2021), representation-based networks (Shalit et al., 2017), and propensity-aware dual-head architectures (Shi et al., 2019); theoretical generalization guarantees for uplift have also been established under appropriate assumptions (Betlei et al., 2021). Surveys synthesize the area (Zhang et al., 2021; Devriendt et al., 2018; Gutierrez and Gérardy, 2017). More recent CIKM work fuses incomplete observational logs with RCT data to identify HTEs when randomized data alone is small (Yao et al., 2024), complementary to our RCT-anchored funnel composition. Recommender-systems work additionally reformulates top-$N$ recommendation as treatment-effect estimation under exposure-ratio policies (Chen et al., 2024). Most deployed uplift studies emphasize a single outcome (often conversion) or regress GMV directly without algebraic coupling, which becomes inefficient under dominant zero mass.

#### Entire-space modeling and deep multi-outcome uplift.

Post-click conversion-rate estimation widely uses entire-space multi-task objectives that share statistical strength across stages (Ma et al., 2018; Wang et al., 2022). Deep uplift architectures such as DESCN (Zhong et al., 2022) represent treatment heterogeneity over multiple stages, EFIN (Liu et al., 2023) models treatment-aware feature interactions for fine-grained ITE, multi-treatment multi-task uplift addresses tiered responses across arms (Wei et al., 2024; Zhao et al., 2017), and chain-style models combine awareness–conversion stages with treatment-aware modules (Huang et al., 2024). Closely related, customer-lifetime-value (CLTV) estimators tackle the heavy-tailed continuous-revenue head with mixture-of-distribution selection (Weng et al., 2024), complementary to but distinct from explicit funnel composition. These objectives improve predictive accuracy under multi-task supervision but do not enforce the deterministic funnel link between binary conversion and nonnegative GMV when estimating uplift.

#### Revenue uplift and budgeted coupon allocation.

Revenue-focused uplift methods emphasize ranking quality under heavy-tailed continuous outcomes (He et al., 2024). Industrial deployments couple uplift estimates with constrained allocation: real-time coupon allocation cast as a multi-choice knapsack with intent detection (Li et al., 2020), Lagrangian-style dual updates for real-time coupon allocation (Tu et al., 2024; Kong et al., 2026), unified marketing-budget allocation plugging heterogeneous value into constrained assignment (Zhao et al., 2019), online multi-choice knapsack personalization driven by uplift (Albert and Goldenberg, 2022), and end-to-end differentiable allocation for budgeted incentives (Sun et al., 2024). We pair funnel-coupled estimates with RCT anchoring and Lagrangian relaxation, keeping the funnel identity throughout the stack rather than only at the prediction layer.

#### Uncertainty quantification under treatment effects.

Conformal prediction yields finite-sample marginal coverage under exchangeability (Vovk et al., 2005), and conformalized quantile regression tightens intervals for continuous outcomes (Romano et al., 2019). Counterfactual conformal inference extends to ITE-style targets under appropriate sampling designs (Lei and Candès, 2021), and conformal calibration must respect RCT splits to retain causal validity (Alaa and van der Schaar, 2019). We compose marginal CQR-style intervals on dual outcomes under a Bonferroni union, producing finite-sample valid two-number coverage statements for joint events rather than a uniformly simultaneous bivariate band.

#### Hold-out evaluation under RCT logs.

Yan et al. (Yan et al., 2023) formalize an expected-outcome metric (EOM) that uses RCT logs to evaluate budgeted policies via Hájek IPW on policy-matched subsets while sweeping a dual multiplier; we adopt this protocol on the industrial OTA data to obtain a full $(\Delta\mathrm{GMV\%},\Delta\mathrm{ROI})$ frontier rather than a single operating point.

#### Positioning.

Deep uplift architectures (Zhong et al., 2022) and revenue-centric estimators (He et al., 2024) flexibly model treatment heterogeneity but do not encode the deterministic funnel identity; dual-head networks typically estimate parallel heads without enforcing algebraic consistency between conversion and GMV expectations. These families address different layers of the deployment problem: revenue rankers optimize ordering without enforcing the conversion–GMV support relation, multi-task uplift models do not by themselves provide subsidy-aware allocation, and budgeted incentive methods generally consume effect estimates as plug-ins without RCT arm-level recalibration. Our contribution is the integration of funnel-consistent estimation, audit-oriented dual-outcome summaries, and tier-aware allocation; we do not claim that each constituent mechanism is individually new. Table 1 summarizes the resulting scope differences.

*Table 1. Positioning relative to representative method families. “Funnel” = enforce $\mu_{\mathrm{gmv}}{=}\mu_{\mathrm{conv}}\mu_{\mathrm{val}}$; “Joint UQ” = joint conformal summaries on dual outcomes; “Multi-tier alloc.” = native budgeted allocation across $K{>}1$ promotion arms.*

| Family | Funnel | Joint UQ | Multi-tier alloc. | Coupon-GMV limitation |

| Meta-learners / forests (Künzel et al., 2019; Wager and Athey, 2018) | – | – | via post-hoc | Single outcome; ad-hoc composition for GMV. |

| Entire-space CVR (Ma et al., 2018; Wang et al., 2022) | partial | – | – | Prediction-targeted, not RCT CATE. |

| Deep uplift (Zhong et al., 2022; Wei et al., 2024; Huang et al., 2024) | – | – | native | Parallel heads without algebraic consistency. |

| Revenue uplift (He et al., 2024) | – | – | via post-hoc | Ranks GMV; ignores deterministic zeros. |

| Budgeted incentives (Albert and Goldenberg, 2022; Sun et al., 2024; Tu et al., 2024; Zhao et al., 2019) | – | – | native | Uplift signals plugged in but not coupled. |

| This work | hard | Bonferroni | LP+Lagrange | Funnel + joint conformal + anchored multi-tier allocation. |

## 3. Problem Formulation

#### Observables and funnel support.

Consider a coupon RCT with observables $(X,T,Y^{c},Y^{g})$, where $X\in\mathcal{X}$ are user features, $T\in\{0,1,\ldots,K\}$ is a discrete treatment indicator ($T{=}0$ is control), $Y^{c}\in\{0,1\}$ is conversion, and $Y^{g}\in\mathbb{R}_{\geq 0}$ is GMV. Here $T$ indexes the coupon offers randomized in the logs; the method does not interpolate a continuous dose–response curve between those arms. Throughout we impose the funnel support restriction

$Y^{g}=0\quad\text{whenever}\quad Y^{c}=0,$ | (1) | | | |

so GMV is undefined as a positive outcome until conversion occurs and conditional spend is only meaningful on the converting subpopulation.

#### Causal targets.

Let $Y^{c}(t),Y^{g}(t)$ denote potential outcomes under assignment $t$. For each non-control arm $t\in\{1,\ldots,K\}$, define arm-specific CATEs

$\tau^{c}_{t}(x)=\mathbb{E}[Y^{c}(t){-}Y^{c}(0)\mid X{=}x],~~\tau^{g}_{t}(x)=\mathbb{E}[Y^{g}(t){-}Y^{g}(0)\mid X{=}x].$ | (2) | | | |

We identify $\tau^{c}_{t}$ and $\tau^{g}_{t}$ under randomized $T\mid X$ (RCT) as in standard analyses; we do not claim identification from purely observational logs.

#### Tower identity.

For any fixed arm $t$, the law of iterated expectations gives

$\mathbb{E}[Y^{g}(t)]=\mathbb{E}\!\big[\,Y^{c}(t)\cdot\mathbb{E}[Y^{g}(t)\mid Y^{c}(t){=}1,X]\,\big],$ | (3) | | | |

separating level calibration of GMV (anchoring metrics in Sec. 4.4) from heterogeneous ordering (PEHE/AUUC in Sec. 5).

#### Decision problem: budgeted multi-tier allocation.

A deterministic policy $\pi:\mathcal{X}\to\{0,\ldots,K\}$ assigns each user to control or one tier. Let $c(x,k)$ denote the predicted incremental subsidy cost of assigning $x$ to tier $k$, derived from the tier-specific coupon terms and the campaign-specific accounting base. Feasible policies satisfy a total budget $B{>}0$:

$\textstyle\sum_{i}c(x_{i},\pi(x_{i}))\leq B,~~\pi(x_{i})\in\{0,\ldots,K\}.$ | (4) | | | |

Objectives include maximizing incremental GMV $\textstyle\sum_{i}\mathbb{E}[\tau^{g}_{\pi(x_{i})}(x_{i})]$ or its ROI-style surrogate $\Delta\mathrm{ROI}{:=}\sum_{i}\hat{\tau}^{g}_{\pi(x_{i})}(x_{i})/\sum_{i}\hat{c}_{i,\pi(x_{i})}$.

#### Dual-objective tension.

When rankings induced by $\tau^{c}$ and $\tau^{g}$ disagree, no single scalar objective is universally aligned with business preferences. The conflict diagnostic in Sec. 4.3 does not solve a general multi-objective program; it flags individuals whose objective-wise rankings and intervals jointly indicate instability near budgeted Top-$K$ cuts.

## 4. Methodology

### 4.1. Funnel-structured uplift estimation

We estimate multi-arm conversion probabilities $\mu_{\mathrm{conv}}^{(t)}(x)$ and nonnegative conditional order-value expectations $\mu_{\mathrm{val}}^{(t)}(x)$ with shared representations. GMV expectations obey the funnel composition

$\mu_{\mathrm{gmv}}^{(t)}(x)=\mu_{\mathrm{conv}}^{(t)}(x)\,\mu_{\mathrm{val}}^{(t)}(x)$ | (5) | | | |

after numerical stabilization (clipping, nonnegative projections). Training combines Bernoulli conversion losses with squared error on $\log(1+\mathrm{GMV})$ among converters; inference maps normalized logits back to currency units using a LogNormal-style mean correction. The total objective allows optional consistency and monotonicity terms:

$\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{conv}}+\alpha\,\mathcal{L}_{\mathrm{val}}+\beta\,\mathcal{L}_{\mathrm{consist}}+\gamma\,\mathcal{L}_{\mathrm{mono}}.$ | (6) | | | |

“Soft funnel” variants replace the hard product (5) with large penalties. They can be preferable when stage labels are asynchronously logged, missing, or otherwise make the support relation approximate. In our RCT logs the support identity is verified by construction, and Sec. 5.3 shows that retaining violations through a soft penalty does not match hard composition under extreme zero inflation.

### 4.2. Variance decomposition and leading-order MSE ratio

Fix $(X,T){=}(x,t)$. For the following propositions, all moments are conditional on this event; let $p{:=}\mathbb{E}[Y^{c}]$, $\mu_{v}{:=}\mathbb{E}[Y^{g}\mid Y^{c}{=}1]$, and $\sigma_{v}^{2}{:=}\mathrm{Var}(Y^{g}\mid Y^{c}{=}1)$.

###### Proposition 0 (Variance decomposition).

$\mathrm{Var}(Y^{g}\mid X{=}x,T{=}t)=p\,\sigma_{v}^{2}+p(1-p)\,\mu_{v}^{2}.$ | (7) | | | |

The first term is the within-converter variance; the second is the Bernoulli switching variance contributed by the zero mass.

###### Proposition 0 (Idealized leading-order MSE ratio under a rate gap).

Let $\hat{\mu}_{g}^{\mathrm{direct}}$ be a direct nonparametric squared-error estimator of $\mathbb{E}[Y^{g}\mid X{=}x,T{=}t]$ and $\hat{\mu}_{g}^{\mathrm{funnel}}{:=}\hat{\mu}_{\mathrm{conv}}\hat{\mu}_{\mathrm{val}}$ the funnel composition estimator. Assume:

-

(RCT identification.) $T\!\perp\!(Y^{c}(\cdot),Y^{g}(\cdot))\mid X$ and the propensity $\Pr(T{=}t\mid X)$ is bounded away from $0$ at $x$.

-

(Funnel support.) $Y^{g}{=}0$ whenever $Y^{c}{=}0$, with $\sigma_{v}^{2}\!\in\!(0,\infty)$ and $\mu_{v}\!\in\!(0,\infty)$.

-

(Conv-head parametric rate.) $\hat{\mu}_{\mathrm{conv}}$ is fit by a (correctly-specified) parametric Bernoulli model on the full $n$-sample, so $\hat{p}{-}p{=}O_{p}(n^{-1/2})$ at $(x,t)$.

-

(Value-head and direct nonparametric variance.) $\hat{\mu}_{\mathrm{val}}$ is fit nonparametrically on the converter subsample and $\hat{\mu}_{g}^{\mathrm{direct}}$ is fit nonparametrically on the full sample, both with negligible bias under standard undersmoothing and asymptotic pointwise variances scaling as $1/r_{n}$ for an effective-sample-size sequence $r_{n}\!\to\!\infty$ with $r_{n}=o(n)$: $\mathrm{Var}(\hat{\mu}_{\mathrm{val}})\!\sim\!\sigma_{v}^{2}/(p\,r_{n})$ and $\mathrm{Var}(\hat{\mu}_{g}^{\mathrm{direct}})\!\sim\!\mathrm{Var}(Y^{g}\mid X{=}x,T{=}t)/r_{n}$.

-

(Cross-head covariance control.) Either independent sample splitting makes $\mathrm{Cov}(\hat{p},\hat{\mu}_{v}){=}0$ in the idealized analysis (Chernozhukov et al., 2018), or the covariance is $o(r_{n}^{-1})$. This condition is not guaranteed by a shared-representation neural implementation.

Then, applying the delta method to $(p,\mu_{v})\!\mapsto\!p\mu_{v}$, the leading-order pointwise MSEs satisfy

$\lim_{n\to\infty}\frac{\mathrm{MSE}(\hat{\mu}_{g}^{\mathrm{funnel}})}{\mathrm{MSE}(\hat{\mu}_{g}^{\mathrm{direct}})}=\frac{p\,\sigma_{v}^{2}}{p\,\sigma_{v}^{2}+p(1-p)\,\mu_{v}^{2}}=\frac{1}{1+(1-p)\,\mu_{v}^{2}/\sigma_{v}^{2}}.$ | (8) | | | |

A proof sketch is given in Appendix A.

Sufficient-regime reading. Eq. (8) is an idealized pointwise variance comparison, not a universal optimality theorem or a guarantee for CATE ranking. It relies on the parametric rate gap (A3) and covariance control (A5), which make the Bernoulli switching and cross-head terms vanish faster than the within-converter contribution $p\,\sigma_{v}^{2}/r_{n}$. If both heads are estimated nonparametrically at the same rate, the ratio collapses to one. Shared neural representations can also induce correlated finite-sample errors, and systematic biases in the two heads can be multiplied by the product composition. We therefore use (8) only as a regime indicator; the controlled E2 stress test (Sec. 5.3) probes whether its predicted direction appears empirically across $\hat{p}\!\in\![5,45]\%$ without validating the asymptotic assumptions.

Operational regime. Within these idealized assumptions, the ratio in (8) is below one whenever $(1{-}p)\mu_{v}^{2}/\sigma_{v}^{2}{>}0$, and shrinks as the zero mass $(1{-}p)$ grows or $\mu_{v}$ dominates $\sigma_{v}$. This is the same hurdle/two-part structural intuition long studied in econometrics (Cragg, 1971; Mullahy, 1986; Lambert, 1992); our contribution is to connect the leading-order ratio under the rate-gap regime to the coupon-uplift setting, not to claim a new funnel identity or universal dominance.

Remark (Bernoulli–LogNormal likelihood alignment). When focal weights are inactive, the LogNormal dispersion in the converting subsample is fixed during value-head fitting, and a log-domain Gaussian surrogate matches the implementation’s $\log(1{+}\mathrm{GMV})$ regression among converters, the loss $\mathcal{L}_{\mathrm{conv}}{+}\alpha\,\mathcal{L}_{\mathrm{val}}$ coincides with the negative log-likelihood of a hierarchical Bernoulli–LogNormal model for $(Y^{c},Y^{g})$ up to additive constants depending only on hyperparameters. The default hard-mode training path (BCE-with-logits $+$ converter MSE on normalized $\log(1{+}\mathrm{GMV})$) instantiates this idealized limit; the ziln variant swaps in focal-BCE and explicit LogNormal NLL with typically frozen dispersion (Sec. 5.3). Auxiliary monotonicity and consistency losses are outside this alignment.

### 4.3. Joint conformal intervals and conflict screening

Scope. We compose marginal split-conformal intervals on each outcome’s CATE summary and apply a Bonferroni union across the two margins. Under standard split-conformal assumptions on a disjoint calibration fold, both intervals jointly cover their respective targets with probability at least $1{-}\alpha$ at nominal level $\alpha/2$ per margin. This is a finite-sample valid two-number coverage statement, not a simultaneous band over the bivariate CATE surface.

Deployment stance. In practice, Bonferroni splits, finite-sample CQR offsets, and heavy-tailed GMV residuals drive empirical joint coverage above the nominal $1{-}\alpha$ (Sec. 5.6). We therefore treat intervals primarily as auditable monitoring bands for compliance and risk review, recommend wider nominal $\alpha\in[0.10,0.20]$ when widths must remain actionable, and pair intervals with anchored point estimates when feeding optimizers, because marginally valid lower-conformal bounds for $\tau^{g}$ at narrow $\alpha$ can be so pessimistic under zero inflation that budgeted LCB policies collapse to all-control assignments (Sec. 5.5).

Conflict diagnostic. A Top-$K$ boundary screen flags users with (i) disagreement between $\tau^{c}$ and $\tau^{g}$ rankings, (ii) wide dual intervals or predictions near decision cutoffs, and (iii) instability near budgeted thresholds. The rule is a risk-disclosure layer for manual review or conservative assignment, not a precision-calibrated detector of latent business conflicts.

### 4.4. Budgeted multi-tier allocation with anchoring

Given predicted incremental rewards $\hat{\tau}^{g}_{t}(x)$ and costs $c(x,t)$, we maximize the budgeted assignment using Lagrangian relaxation: dual updates over a scalar multiplier $\lambda$ approximately enforce (4) while inner problems decouple across users for scalability. Moderate-scale LP relaxations serve as references where memory permits and provide the headline industrial allocator in Sec. 5.8.

Anchoring. Predicted GMV levels can exhibit systematic bias under extreme zero inflation (for example, underestimating control-arm GMV mass). We apply additive shifts estimated from RCT arm-wise averages on a held-in slice before forming rewards fed to the allocator, improving $\Delta\mathrm{ROI}$-style objectives without retraining. End-to-end anchor losses are left for future work.

Train–calibrate–test. We use disjoint splits: training folds for model fitting, a separate calibration fold for conformal quantiles, and held-out test folds for uplift metrics and allocation summaries. User-level clustering is recommended when the same customer could otherwise leak across folds.

## 5. Experiments

### 5.1. Datasets and protocol

We compare twelve methods: FunnelCausalNet plus eleven baselines spanning meta-learners (S/T/X) (Künzel et al., 2019), causal forests (Wager and Athey, 2018; Athey et al., 2019), a dual-head network, CFRNet representation balancing (Shalit et al., 2017), DragonNet propensity-aware twin-head (Shi et al., 2019), EFIN explicit feature–treatment interaction (Liu et al., 2023), DESCN-style (Zhong et al., 2022) and ECUP-style (Huang et al., 2024) deep uplift, and RERUM-style (He et al., 2024) revenue ranking uplift. Multi-tier extensions of binary-treatment originals (CFRNet, DragonNet) follow common practice (Zhao et al., 2017; Wei et al., 2024): per-arm outcome heads with the binary balancing/propensity penalty replaced by a multi-arm aggregation (mean pairwise linear-MMD against control for CFRNet; multi-class softmax cross-entropy for DragonNet); EFIN keeps its intent-attention block plus per-arm explicit feature-treatment cross-interaction. All deep baselines are reimplemented under a unified PyTorch pipeline so that training schedules, hyperparameters, and evaluation interfaces are identical across methods. Models train for $25$ epochs with Adam; uplift metrics aggregate three seeds. Fixed experiment configurations and seeds are used throughout; internal reruns reproduce the reported aggregates up to floating-point nondeterminism.

*Table 2. Datasets used in the main matrix.*

$N$ | Dataset | (default) | Arms | Evaluation |

$10\mathrm{K}$ $8{+}1$ | Criteo-MT7 (semi-synth.) | tr./eval | | Oracle ITE; PEHE, AUUC |

$\sim 64\mathrm{K}$ | Hillstrom (Hillstrom, 2008) | | Binary | AUUC proxies; no oracle PEHE |

$5\mathrm{M}$$50\mathrm{K}/4.9\mathrm{M}$ | OTA Hotel-Coupon (de-id.) | records; tr./eval | Multi-tier | RCT EOM |

#### Semi-synthetic calibration disclosure.

Criteo-MT7’s generator parameters (baseline conversion ${\approx}8\%$, eight tiers $0\%$–$14\%$) fall inside operationally common e-commerce coupon ranges and are not tuned to match any specific industrial dataset. To rule out calibration that selectively favors funnel composition, the E2b stress test (Sec. 5.3) sweeps the conversion baseline across $\hat{p}\in[4.6\%,45.4\%]$; the funnel benefit is monotone throughout this range.

#### Public benchmark coverage.

We surveyed the public corpora cited across the closest baselines (Sec. 5.1) and across multi-treatment ITE methodology papers (drawing on the e-commerce uplift surveys of (Devriendt et al., 2018; Gutierrez and Gérardy, 2017)), including MEMENTO (Mondal et al., 2022), whose own experiments rely on Amazon-private and fully synthetic data because no public multi-treatment RCT was available to its authors. HTE classics from medical or educational RCTs (IHDP (Hill, 2011), ACIC, Mindsets, TWINS; surveyed in (Devriendt et al., 2018)) pair binary treatment with continuous outcomes; multi-arm semi-synthetic surfaces (News, TCGA) pair tiered treatments with simulated outcomes over text or genomic covariates rather than coupon RCTs; non-commercial multi-arm RCTs from political (Gerber et al.’s GOTV, 5 arms) or clinical (Colon, 3 arms) trials are likewise scope-mismatched with the coupon-strength setting. E-commerce uplift releases—Hillstrom (Hillstrom, 2008), Lenta (Lenta Group, 2020), MegaFon (MegaFon, 2021), Criteo Uplift v2.x (Diemert et al., 2018), and the DESCN-companion Lazada release (Zhong et al., 2022)—either implement single-encouragement send-vs-control RCT designs or, in Hillstrom’s three-arm form, contrast different message types (men vs. women catalog) rather than coupon-strength tiers, so the tier-specific conversion- and revenue-elasticities motivation (iii) of Sec. 1 is not realized. Multi-arm public collections such as Tianchi-O2O (Alibaba Tianchi, 2018) provide tiered discount rates but only coupon-redemption labels with no continuous GMV supervision and are observational rather than randomized; the Open Bandit Dataset (Saito et al., 2021) is a recommendation-policy log rather than a coupon-strength RCT. The closest publicly discussed multi-tier coupon RCT is the MT-LIFT release shipped with ECUP (Huang et al., 2024) (5-arm Meituan food-delivery, ${\approx}5.5\mathrm{M}$ samples), but it provides only binary chain labels (click and conversion) and no continuous-spend / GMV outcome, so the within-converter funnel-value head this paper targets cannot be supervised on it. To our knowledge, no public benchmark simultaneously realizes multi-tier coupon assignment, continuous GMV ground truth, and strict RCT randomization. We therefore evaluate on (a) a semi-synthetic multi-tier surface (Criteo-MT7, oracle ITE), (b) one widely cited public RCT (Hillstrom, included as a scope-boundary disclosure), and (c) a large industrial multi-tier RCT (OTA Hotel-Coupon) that instantiates the target regime at production scale. The implementation uses a modular ingestion interface so that additional multi-tier corpora can be evaluated without changing the model or evaluation logic.

Metrics. AUUC_GMV and AUUC_CVR summarize uplift-curve area (higher is better). PEHE_GMV and PEHE_CVR are defined where identifiable ground truth is available (Criteo-MT7). ATE error measures GMV average-treatment-effect error when defined. Hillstrom and OTA do not admit the same PEHE as MT7; we report ranking and calibration metrics appropriate to each source. The industrial protocol additionally reports the expected-outcome metric (EOM) of (Yan et al., 2023) via Hájek IPW on policy-matched RCT subsets while sweeping a dual multiplier $\alpha$ to trace the $(\Delta\mathrm{GMV\%},\Delta\mathrm{ROI})$ frontier.

### 5.2. Uplift estimation quality (E1)

Table 3 reports three-seed means on Criteo-MT7 at $N{=}10\mathrm{K}$. EFIN attains the highest AUUC_GMV ($0.615$); FunnelCausalNet ranks second ($0.613$, within one seed standard deviation), indicating that EFIN’s explicit feature-treatment cross-interaction aligns particularly well with this synthetic generator’s tier-aware nonlinearity. Crucially, the semi-synthetic advantage does not carry over to the production OTA RCT (Sec. 5.8), where FunnelCausalNet leads at every $\Delta\mathrm{GMV\%}$ anchor. PEHE_CVR is led by DualHeadNet ($0.048$); FunnelCausalNet ($0.058$) remains competitive, confirming that funnel coupling does not destroy conversion-head identifiability. ATE_GMV_err favors shallower models (Causal Forest, S-Learner) that compress the conditional-mean range—level calibration and heterogeneous ranking are distinct objectives.

*Table 3. Criteo-MT7 estimation quality (E1; $N{=}10\mathrm{K}$, three-seed means). Arrows indicate desired direction; boldface marks the best mean per column.*

$\uparrow$ $\downarrow$ $\downarrow$ $\downarrow$| Method | AUUC_GMV | PEHE_GMV | PEHE_CVR | ATE_err |

| EFIN | 0.615 | 26.70 | 0.054 | 12.41 |

| FunnelCausalNet | 0.613 | 31.18 | 0.058 | 21.67 |

| DESCN-style | 0.605 | 30.91 | 0.054 | 20.88 |

| DragonNet | 0.593 | 27.96 | 0.056 | 14.47 |

| CFRNet | 0.568 | 29.29 | 0.064 | 14.06 |

| ECUP | 0.567 | 32.39 | 0.055 | 22.51 |

| RERUM | 0.562 | 39.17 | 0.072 | 28.75 |

| DualHeadNet | 0.514 | 32.44 | 0.048 | 7.43 |

| S-Learner | 0.508 | 43.71 | 0.067 | 5.98 |

| X-Learner | 0.508 | 148.7 | 0.136 | 8.14 |

| T-Learner | 0.507 | 197.5 | 0.182 | 6.02 |

| Causal Forest | 0.489 | 56.73 | 0.080 | 4.62 |

Public-RCT scope boundary. Hillstrom is a single-encouragement RCT in which two message-type arms are pooled against the no-send control; positives are sparse and there is no coupon-strength axis along which conversion- and spend-elasticities can differ. The multi-tier decision problem this paper targets (Sec. 1, contribution (iii)) is therefore not realized, and the funnel composition reduces to estimating a near-degenerate spending head downstream of a single conversion lift. With extremely sparse converters, the value head has too few effective samples to outperform a direct rank-style estimator on revenue. Empirically, revenue-focused rankers RERUM ($0.747$) and DualHeadNet ($0.739$) lead AUUC_GMV on Hillstrom, while all multi-tier funnel-aware deep models (DESCN, ECUP, FunnelCausalNet) underperform. This result may reflect both scope mismatch and a finite-sample converter bottleneck, and it is direct evidence that FunnelCausalNet is not broadly dominant on binary public benchmarks. The industrial OTA multi-arm RCT in Sec. 5.8 is the target regime rather than proof of transfer beyond it.

### 5.3. Funnel ablation (E2)

We compare four modes on Criteo-MT7: direct GMV regression (A), soft funnel penalties (B), hard funnel composition (C), and a ZILN-style likelihood path (D), using five seeds for each sample size and mode. Hard coupling achieves the lowest PEHE_GMV at $10\mathrm{K}$, $20\mathrm{K}$, and $100\mathrm{K}$ samples, while the funnel-violation rate of A remains at $\gtrsim 60\%$ versus $0\%$ for C. At $N{=}100\mathrm{K}$, D approaches C ($17.7$ vs. $16.0$; relative excess $\approx 11\%$), consistent with the Bernoulli$\times$LogNormal narrative; at $N{=}10\mathrm{K}$, D underperforms hard composition because of limited converter sample for fitting the alternate likelihood. This ablation isolates the funnel formulation while holding the training harness fixed; E4 separately compares allocation variants. We do not claim a full factorial decomposition of estimator architecture, anchoring, conformal diagnostics, and allocation.

*Table 4. E2 Criteo-MT7 funnel ablation: mean PEHE_GMV (lower better) and funnel-violation rate (%, 5 seeds). Modes: A=direct GMV regression, B=soft penalty, C=hard funnel, D=ZILN-style.*

$\downarrow$ $\downarrow$| | PEHE_GMV | Violation (%) |

$N$ | | A | B | C | D | A | B | C | D |

$10\mathrm{K}$ | | 25.94 | 26.12 | 20.26 | 38.02 | 62.4 | 1.21 | 0.00 | 0.00 |

$20\mathrm{K}$ | | 25.48 | 25.91 | 15.29 | 23.23 | 64.0 | 1.34 | 0.00 | 0.00 |

$100\mathrm{K}$ | | 25.80 | 25.43 | 15.97 | 17.67 | 71.2 | 1.25 | 0.00 | 0.00 |

Prop. 2 zero-inflation stress test. We probe the variance regime suggested by (8) on a controlled semi-synthetic surface by sweeping the conversion baseline logit $\mu_{p}$ of the Criteo-MT7 generator at $N{=}20\mathrm{K}$, $5$ seeds each, contrasting hard funnel composition (C_hard) against direct GMV regression (A_direct). Table 5 reports observed conversion rate $\hat{p}$ and PEHE_GMV ratio across four levels. Funnel composition reduces PEHE_GMV by $18$–$48\%$ across the tested $\hat{p}\in[4.6\%,45.4\%]$ range, with peak benefit at moderate-high zero inflation; this direction is consistent with Eq. (8), but the ablation does not verify its asymptotic assumptions or establish general dominance. The finite-sample dip at the most extreme $\hat{p}\!=\!4.6\%$ end is consistent with sparse-converter noise on the value head.

*Table 5. E2 extension: Prop. 2 zero-inflation stress test on Criteo-MT7 ($N{=}20\mathrm{K}$, $5$ seeds). PEHE_GMV mean$\pm$std for direct GMV regression (A) versus hard funnel composition (C); benefit$=1{-}\mathrm{PEHE}_{C}/\mathrm{PEHE}_{A}$. All four tested levels favor C, with peak benefit at moderate-high zero inflation, consistent with the direction of (8).*

$\mu_{p}$ $\hat{p}$ | Level | | PEHE_GMV (A) | PEHE_GMV (C) | Benefit |

$-3.5$ $4.6\%$ $12.22{\pm}3.35$ $\mathbf{10.01}{\pm}3.57$ $+18.1\%$| | | | | |

$-2.4$ $11.9\%$ $25.43{\pm}6.25$ $\mathbf{15.77}{\pm}5.79$ $+38.0\%$| | | | | |

$-1.5$ $24.3\%$ $39.22{\pm}8.96$ $\mathbf{20.29}{\pm}5.70$ $\mathbf{+48.3\%}$| | | | | |

$-0.5$ $45.4\%$ $55.11{\pm}15.91$ $\mathbf{30.54}{\pm}5.65$ $+44.6\%$| | | | | |

### 5.4. Conflict diagnostic as audit layer (E3)

This subsection sanity-checks the Top-$K$ conflict screen of Sec. 4.3 as an audit signal, not as a production classifier; it is not part of the funnel-estimation or budgeted-allocation pipelines. We use synthetic stress tests with controllable injection (no latent conflict labels exist on real coupon logs), varying an injection correlation $\rho_{\mathrm{conf}}$ between conversion and GMV uplift signals. Table 6 reports three-seed means of precision, recall, and F1. Peak F1 reaches $\approx 0.25$ at $\rho_{\mathrm{conf}}{=}0.6$; the rule’s value is to surface users near budget cuts whose objective-wise orderings disagree, not to act as a calibrated detector.

*Table 6. E3 semi-synthetic conflict detection: precision / recall / F1 versus injected correlation $\rho_{\mathrm{conf}}$ (three-seed means).*

$\rho_{\mathrm{conf}}$ | | Precision | Recall | F1 |

$0.0$ | | 0.146 | 0.116 | 0.120 |

$0.3$ | | 0.201 | 0.175 | 0.179 |

$0.6$ | | 0.286 | 0.218 | 0.246 |

$0.9$ | | 0.203 | 0.116 | 0.146 |

### 5.5. Budgeted allocation on MT7 (E4)

On semi-synthetic Criteo-MT7 at $N{=}20\mathrm{K}$ with eight tiers and realistic cost presets, we pipeline FunnelCausalNet predictions through joint conformal summaries (where applicable) and budgeted allocation. With tier discount rates $d_{k}$, the solver uses costs $d_{k}\hat{\mu}_{g}^{(k)}(x)$, while evaluation applies the same rule to the oracle GMV surface. Table 7 reports the mean oracle incremental-GMV surrogate $\tau_{g}$, realized subsidy cost, and their ratio $\Delta\mathrm{ROI}$ across three seeds.

The anchored-Lagrangian pipeline attains higher $\Delta\mathrm{ROI}$ than random allocation under tight budgets—for example, $3.92$ versus $3.07$ at $B/B_{\mathrm{free}}{=}0.05$—with lower realized cost and competitive incremental GMV. This comparison changes anchoring and allocation jointly and therefore does not isolate the anchoring contribution. LP relaxation often achieves higher raw incremental GMV but spends more budget; at $B/B_{\mathrm{free}}{=}0.50$ the anchored-Lagrangian pipeline tracks alternatives on $\Delta\mathrm{ROI}$ while trading off peak GMV. LCB-driven assignments (funnel_ip_lcb) degenerate to all-control allocations in these logs (zero realized lift and cost), consistent with pessimistic lower-conformal surfaces under heavy zero inflation; they are omitted from Table 7, motivating the deployment stance in Sec. 4.3 (wider $\alpha$ or anchored point estimates rather than narrow-$\alpha$ LCB).

*Table 7. E4 semi-synthetic MT7 budgeted allocation versus baselines (three-seed means). $\Delta\mathrm{ROI}=$(oracle incremental GMV)/(realized subsidy cost).*

$B/B_{\mathrm{free}}$ $\tau_{g}$ $\Delta\mathrm{ROI}$| | Strategy | surr. | Cost | |

$0.05$ | | baseline_random | 8 783 | 2 864 | 3.07 |

$0.05$ | | baseline_topk | 8 813 | 2 409 | 3.66 |

$0.05$ | | funnel_ip_lp | 9 986 | 2 910 | 3.43 |

$0.05$ | | funnel_ip_anchored | 9 072 | 2 315 | 3.92 |

$0.10$ | | baseline_random | 16 460 | 5 580 | 2.95 |

$0.10$ | | funnel_ip_anchored | 16 379 | 4 413 | 3.71 |

$0.50$ | | baseline_topk | 71 906 | 23 593 | 3.05 |

$0.50$ | | funnel_ip_lp | 72 870 | 24 069 | 3.03 |

$0.50$ | | funnel_ip_anchored | 63 766 | 21 254 | 3.00 |

### 5.6. Joint conformal coverage (E5)

We run split conformal with Bonferroni separation across conversion and GMV outcomes using three seeds on MT7 ($N{=}20\mathrm{K}$) and OTA ($N{=}50\mathrm{K}$). Table 8 (left block) reports marginal and joint outcome-layer coverage; the right block of the same table quantifies systematic over-coverage of the joint event relative to nominal $1{-}\alpha$. Table 9 lists oracle-$\tau$ coverage on MT7 (identifiable) together with mean $\tau_{g}$ interval width in currency units.

*Table 8. E5 outcome-layer empirical coverage (three-seed means; rounded).*

$\alpha$ $c{=}1$ | Dataset | | cov_conv | cov_val() | cov_joint |

| MT7 | 0.05 | 0.989 | 0.986 | 0.987 |

| MT7 | 0.10 | 0.975 | 0.986 | 0.974 |

| MT7 | 0.20 | 0.947 | 0.949 | 0.942 |

| OTA | 0.05 | 0.989 | 0.986 | 0.988 |

| OTA | 0.10 | 0.975 | 0.980 | 0.974 |

| OTA | 0.20 | 0.950 | 0.959 | 0.948 |

$\alpha$ $\Delta$ $\Delta$| | nominal | MT7 | OTA |

$0.05$ $+0.037$ $+0.038$| | 0.95 | | |

$0.10$ $+0.074$ $+0.074$| | 0.90 | | |

$0.20$ $+0.142$ $+0.148$| | 0.80 | | |

$\Delta=$$-(1{-}\alpha)$| cov_joint in pp. |

*Table 9. E5 MT7 oracle-$\tau$ coverage and mean $\tau_{g}$ interval width (currency units, three-seed means). OTA $\tau$-oracle summaries omitted.*

$\alpha$ $\tau_{c}$ $\tau_{g}$ $w_{\tau_{g}}$ $w_{\tau_{c}}$| | cov | cov | Mean | Mean |

$0.05$ | | 1.000 | 1.000 | 11 741 | 2.000 |

$0.10$ | | 1.000 | 1.000 | 11 637 | 2.000 |

$0.20$ | | 1.000 | 1.000 | 8 748 | 2.000 |

Joint empirical coverage consistently exceeds nominal $1{-}\alpha$ by 3–15 pp across $\alpha\in\{0.05,0.10,0.20\}$, reflecting conservative finite-sample CQR offsets combined with the Bonferroni union (Fig. 1). On MT7 the oracle $\tau$-intervals are fully covered in these runs while $w_{\tau_{g}}$ remains on the order of $10^{4}$ currency units, so LCB-driven actions at narrow $\alpha$ are often vacuous without wider $\alpha$ or anchored point estimates (Sec. 5.5). On OTA, conversion-interval width on the probability scale drops sharply between $\alpha{=}0.05$ and $0.10$ (Fig. 2), reflecting probability-axis saturation near width one at narrow $\alpha$.

Bar chart of empirical outcome-layer conformal coverage versus nominal alpha for MT7 and OTA datasets. For each dataset and alpha in 0.05/0.10/0.20, three bars show marginal conversion coverage, conditional GMV-given-conversion coverage, and joint event coverage; joint coverage exceeds nominal 1-alpha by approximately 3 to 15 percentage points across alpha settings.

*Figure 1. E5 outcome-layer coverage vs. $\alpha$ (marginal conversion, conditional GMV given conversion, and joint event). Joint empirical coverage exceeds nominal $1{-}\alpha$ across $\alpha$.Bar chart of empirical outcome-layer conformal coverage versus nominal alpha for MT7 and OTA datasets. For each dataset and alpha in 0.05/0.10/0.20, three bars show marginal conversion coverage, conditional GMV-given-conversion coverage, and joint event coverage; joint coverage exceeds nominal 1-alpha by approximately 3 to 15 percentage points across alpha settings.*

Line plot of mean conformal interval width versus nominal alpha for conversion and GMV heads on MT7 and OTA datasets. The conversion-head curve on OTA saturates near width one as alpha narrows toward 0.05 due to probability-axis bounding; the GMV head reported on log(1+GMV) scale narrows monotonically as alpha widens.

*Figure 2. E5 mean interval width vs. $\alpha$: conversion head saturates near width 1 at narrow $\alpha$ on OTA; GMV head on the $\log(1{+}\mathrm{GMV})$ scale.Line plot of mean conformal interval width versus nominal alpha for conversion and GMV heads on MT7 and OTA datasets. The conversion-head curve on OTA saturates near width one as alpha narrows toward 0.05 due to probability-axis bounding; the GMV head reported on log(1+GMV) scale narrows monotonically as alpha widens.*

### 5.7. Computational scalability (E6)

We measure training, conformal calibration, inference, and allocation wall-clock versus $N$ and $K$. Figure 3 plots $\log$-$\log$ scaling curves; Table 10 excerpts $K{=}8$ timings.

*Table 10. E6 wall-clock seconds ($K{=}8$, three-seed means; LP omitted where runs failed/OOM).*

$N$ | | Train | Conf. cal. | IP Lagrange | IP LP |

$10\mathrm{K}$ | | 30.70 | 0.020 | 0.0019 | 0.429 |

$50\mathrm{K}$ | | 57.40 | 0.047 | 0.0073 | 11.26 |

$500\mathrm{K}$ | | 188.41 | 0.178 | 0.0599 | — |

$10^{6}$ | | 324.01 | 0.338 | 0.127 | — |

Training scales sublinearly between $N{=}10^{4}$ and $10^{6}$ in our sweeps ($\approx 30\,\mathrm{s}\to 324\,\mathrm{s}$, $\sim 10\times$ wall-clock for $100\times$ users). Conformal calibration stays below one second even at $N{=}10^{6}$. Lagrangian dual updates stay near $0.13\,\mathrm{s}$ at one million users for $K{=}8$, whereas dense LP relaxations exceed tens of seconds already at $N{=}10^{5}$ and fail at larger $N$ due to memory. Production deployments therefore emphasize Lagrangian schemes with rounding while LP is reserved for moderate-scale benchmarking (including the E7 industrial headline).

Four-panel log-log scaling plot of training wall-clock vs N, integer-program solvers (LP relaxation and Lagrangian dual) vs N, Lagrangian dual updates vs number of arms K, and inference plus conformal phases vs N. Training scales sublinearly between 10 thousand and 1 million users; LP fails at 500 thousand or more users due to memory while Lagrangian dual stays under 0.13 seconds at 1 million users for K equals 8.

*Figure 3. E6 computational scaling (log-log): training, IP solvers, Lagrange vs. $K$, inference+conformal. Lagrangian allocation remains tractable at million-user scale.Four-panel log-log scaling plot of training wall-clock vs N, integer-program solvers (LP relaxation and Lagrangian dual) vs N, Lagrangian dual updates vs number of arms K, and inference plus conformal phases vs N. Training scales sublinearly between 10 thousand and 1 million users; LP fails at 500 thousand or more users due to memory while Lagrangian dual stays under 0.13 seconds at 1 million users for K equals 8.*

### 5.8. Industrial OTA: full hold-out EOM (E7)

We complement subsampled AUUC evidence with a large hold-out evaluation on de-identified Hotel-Coupon multi-arm RCT logs totaling $\approx 4.98\times 10^{6}$ exposure records from $\approx 2.79\times 10^{6}$ distinct users overall. For each of three permutation seeds we shuffle the full table, take the first $N_{\mathrm{train}}{=}50\mathrm{K}$ records for training, and retain the remaining $\approx 4.93$M exposure records per seed for evaluation. Imputation and $z$-score standardization use statistics fit only on the training slice and applied disjointly to the hold-out so evaluation-set margins do not leak into normalization.

#### Practical operating regime.

The data come from a de-identified hotel-coupon RCT. We treat the platform commission rate $\gamma$ as a sensitivity parameter over $[0.2,0.3]$, a band typical of online travel/coupon programs; the break-even point is $\Delta\mathrm{ROI}\!=\!1/\gamma\!\in\![3.3,5.0]$. Operating below the band ($\Delta\mathrm{ROI}\!<\!3$) means incremental commission no longer offsets subsidy cost; well above ($\Delta\mathrm{ROI}\!>\!5$) subsidies are so tight that absolute incremental GMV is rarely operationally meaningful. The $\Delta\mathrm{GMV\%}$ anchors in Table 11 straddle this band, with smaller anchors probing the tight-budget regime where ranking quality dominates and larger anchors approaching the break-even boundary. Conclusions are insensitive to the specific $\gamma$ within $[0.2,0.3]$.

Coupon planning is expressed under alternative incremental-GMV targets $\Delta\mathrm{GMV\%}$ rather than a single universal budget. EOM evaluation follows (Yan et al., 2023): for each dual multiplier $\alpha$, we recommend arms via the LP-relaxation KKT solution on predicted GMV lifts and costs, retain users whose randomized assignment matches the recommendation, and estimate incremental GMV with Hájek IPW on that subset relative to hold-out control mean $V_{\mathrm{ctl}}$. Let $V_{\alpha}$ be the Hájek-IPW mean GMV and $C_{\alpha}$ the corresponding Hájek-IPW mean subsidy cost on the policy-matched subset. As $\alpha$ varies, each model traces a full $(\Delta\mathrm{GMV\%},\Delta\mathrm{ROI})$ frontier, where $\Delta\mathrm{GMV\%}=100(V_{\alpha}-V_{\mathrm{ctl}})/V_{\mathrm{ctl}}$ and $\Delta\mathrm{ROI}=(V_{\alpha}-V_{\mathrm{ctl}})/C_{\alpha}$. The latter is the same subsidy-cost surrogate as Sec. 5.5, not reconciled store-level profit. Models are six multi-arm deep uplift networks—ECUP (Huang et al., 2024), RERUM (He et al., 2024), CFRNet (Shalit et al., 2017), DragonNet (Shi et al., 2019), EFIN (Liu et al., 2023), and FunnelCausalNet—trained for $30$ epochs per seed; the headline policy is LP allocation.

*Table 11. OTA full hold-out EOM (E7): $\Delta\mathrm{ROI}$ mean$\pm$std across three permutation seeds at representative incremental-GMV anchors read off the LP frontier ($N_{\mathrm{train}}{=}50\mathrm{K}$, $\approx 4.93\mathrm{M}$ hold-out exposure records per seed). The seeds are permutation splits of the same hold-out, not independent RCTs; per-anchor paired-bootstrap $95\%$ CIs of (FunnelCausalNet $-$ second-best) over $n{=}3$ include $0$ ($[-0.42,+0.35]$ at $10\%$ and $[-0.07,+0.38]$ at $60\%$). FunnelCausalNet has the highest seed-averaged mean at all $7/7$ anchors. A naive one-sided sign calculation gives $(1/2)^{7}\!\approx\!0.008$, but the anchors are correlated points on one LP frontier, so this number is descriptive rather than an independent-anchor significance test. DragonNet@$10\%$ has one seed on the frontier and no std. Boldface marks the best mean.*

$\Delta\mathrm{GMV\%}$ | | ECUP | RERUM | CFRNet | DragonNet | EFIN | FunnelCausalNet |

$10$ $4.13\!\pm\!0.62$ $4.77\!\pm\!0.39$ $4.11\!\pm\!0.08$ $3.13\!\pm\!\mathrm{n/a}$ $4.84\!\pm\!0.45$ $\mathbf{4.94\!\pm\!1.08}$| | | | | | | |

$20$ $4.00\!\pm\!0.32$ $4.40\!\pm\!0.48$ $3.92\!\pm\!0.05$ $4.30\!\pm\!0.49$ $4.23\!\pm\!0.17$ $\mathbf{4.57\!\pm\!0.37}$| | | | | | | |

$25$ $3.87\!\pm\!0.27$ $4.23\!\pm\!0.39$ $3.83\!\pm\!0.10$ $4.12\!\pm\!0.35$ $4.08\!\pm\!0.15$ $\mathbf{4.44\!\pm\!0.22}$| | | | | | | |

$30$ $3.82\!\pm\!0.24$ $4.08\!\pm\!0.30$ $3.75\!\pm\!0.14$ $3.97\!\pm\!0.26$ $3.96\!\pm\!0.14$ $\mathbf{4.30\!\pm\!0.15}$| | | | | | | |

$40$ $3.73\!\pm\!0.15$ $3.81\!\pm\!0.20$ $3.63\!\pm\!0.23$ $3.74\!\pm\!0.18$ $3.77\!\pm\!0.13$ $\mathbf{4.01\!\pm\!0.17}$| | | | | | | |

$50$ $3.60\!\pm\!0.10$ $3.62\!\pm\!0.17$ $3.52\!\pm\!0.17$ $3.58\!\pm\!0.10$ $3.62\!\pm\!0.13$ $\mathbf{3.80\!\pm\!0.15}$| | | | | | | |

$60$ $3.49\!\pm\!0.08$ $3.50\!\pm\!0.11$ $3.41\!\pm\!0.10$ $3.49\!\pm\!0.00$ $3.50\!\pm\!0.09$ $\mathbf{3.60\!\pm\!0.14}$| | | | | | | |

For raw-magnitude calibration, the coarse marginal per-user GMV contrast between the strongest arm and control (${\approx}92.7\%$, three-seed average) is not the same estimand as the EOM horizontal axis $\Delta\mathrm{GMV\%}$ (an LP-policy IPW estimate at fixed dual $\alpha$). The swept LP frontiers reach different right-end extents (maximum realized $\Delta\mathrm{GMV\%}$: $\approx 72.2$ for ECUP, $74.7$ for RERUM, $86.9$ for CFRNet, $83.4$ for DragonNet, $61.8$ for EFIN, and $90.4$ for FunnelCausalNet), so curves are best read as full traces rather than single-number summaries; the larger extent under FunnelCausalNet means it can express more aggressive operating regimes that the other rankers cannot reach.

Reading Table 11 honestly. FunnelCausalNet attains the highest mean $\Delta\mathrm{ROI}$ at every anchor. The closest competitor varies by regime: at the small-anchor end ($\Delta\mathrm{GMV\%}{=}10\%$–$20\%$) it is EFIN or RERUM (both within one standard deviation), while at mid-to-large anchors ($25\%$–$60\%$) FunnelCausalNet’s mean exceeds the second-best by $0.18$–$0.21$ ROI units. CFRNet sits in the lowest band at every anchor, consistent with linear-MMD balancing being designed for binary rather than tier-specific elasticities. Per-anchor paired-bootstrap CIs over three permutation seeds include $0$, so individual rows are not formally significant. The $7/7$ wins summarize the direction of one correlated frontier and do not establish cross-anchor significance. We report LP as the headline allocator because the EOM protocol (Yan et al., 2023) is defined with LP-relaxation KKT recommendations, matching both the E4 budgeted experiments and the online consistency check (Sec. 5.8).

#### Online consistency check.

An internal online evaluation against the platform’s incumbent uplift baseline under the same LP allocator is directionally consistent with the offline EOM ordering in Table 11. We do not use it as headline evidence: quantitative effect sizes, per-bucket exposure ratios, and ablation traces remain unavailable under the platform agreement, so the paper’s verifiable claims rely on the reported RCT/EOM aggregates and public or semi-synthetic experiments.

### 5.9. Reproducibility

All public and semi-synthetic experiments use fixed seeds, unified configurations, and recorded run manifests; internal reruns reproduce the reported aggregates up to floating-point nondeterminism. The current version does not include a public code artifact. Public datasets remain available from their cited sources. Industrial OTA Hotel-Coupon micro-data cannot be redistributed under the platform agreement; we report aggregate metrics only, so the industrial experiment cannot be independently reproduced externally.

## 6. Discussion

#### Regime-guided method choice.

Funnel coupling is competitive on the semi-synthetic multi-tier MT7 surface—FunnelCausalNet’s AUUC_GMV ($0.613$) is within one seed standard deviation of the leading EFIN ($0.615$)—and has the highest mean $\Delta\mathrm{ROI}$ at all $7/7$ reported industrial EOM anchors (Table 11). The anchors share one LP frontier and the seeds reuse one RCT through permutation splits, so this is descriptive consistency, not independent-test evidence. On single-encouragement public RCTs (e.g., Hillstrom), every tested multi-tier funnel-aware deep model underperforms revenue-centric alternatives; these results define a generalization boundary rather than evidence to assume transfer. Practical guidance is therefore regime-dependent: use hard composition when the funnel support identity is exact, consider soft penalties when logging makes that identity approximate, and evaluate revenue ranking and allocation separately across the available treatment design.

#### Ranking vs. calibration.

Training emphasizes heterogeneous ordering (PEHE/AUUC); absolute ATE-style GMV calibration can remain imperfect under heavy tails. RCT-arm anchoring mitigates systematic level bias feeding budgeted objectives without retraining; forcing marginal ATE agreement would require additional regularization or doubly robust corrections and is left to future work.

#### Conformal conservatism.

Bonferroni splits and finite-sample CQR offsets yield conservative joint coverage (empirical $1{-}\alpha$ exceeds nominal by $3$–$15$ pp in our sweeps), so narrow-$\alpha$ LCB allocation collapses toward all-control under heavy tails (Sec. 5.5). Wider nominal $\alpha$ or anchored point estimates remain the pragmatic pairing for actionable budgets.

#### Industrial RCTs: metric/policy interplay.

OTA-style analyses mix subsidy costs, GMV lifts, and commission assumptions. Sec. 5.8 adds a complementary massive hold-out EOM check: sweeping LP policies traces full $(\Delta\mathrm{GMV\%},\Delta\mathrm{ROI})$ curves, and at representative incremental-GMV anchors in $10\%$–$60\%$ FunnelCausalNet leads all multi-arm deep uplift baselines on mean $\Delta\mathrm{ROI}$ even though subsampled AUUC gaps are tight; EFIN’s MT7 advantage does not transfer (its LP-frontier reach is the smallest, $\approx\!61.8\%$ vs. FunnelCausalNet’s $\approx\!90.4\%$). Reported $\Delta\mathrm{ROI}$ is a cost-construct surrogate, not reconciled store-level profit.

#### Identification scope and limitations.

All causal interpretations assume RCT-like randomized assignment, not observational identification. The strongest allocation evidence comes from a private industrial RCT that cannot be independently reproduced, while the public binary benchmark does not show consistent gains; the results therefore support the target multi-tier regime rather than broad dominance. Three permutation splits and correlated EOM anchors limit inferential power. Moreover, E7 uses record-level permutation splits, so repeated users can appear in both training and hold-out slices; this further limits IID and interval interpretations and motivates user-grouped splitting and clustered uncertainty in follow-up validation. The component studies isolate funnel structure and allocator variants but not a full factorial pipeline decomposition. Proposition 4.2 is an idealized pointwise comparison whose rate and covariance assumptions need not hold for shared neural heads. Joint conformal coverage is marginal and conservative. Finally, the observed coupon arms are discrete offers: the model does not exploit smoothness or monotonicity across a continuous coupon dose, which is an important extension when treatment intensity is not operationally tiered.

## 7. Conclusion

Coupon uplift in digital commerce must respect the funnel restriction linking conversion and GMV, cope with extreme zero inflation on revenue, and support multi-tier subsidy decisions under budgets. We presented FunnelCausalNet, which encodes funnel composition in estimation, provides an idealized rate-gap variance comparison (Prop. 2), pairs a Lagrangian budgeted allocator with RCT-arm anchoring, and exposes a Bonferroni-union joint conformal layer plus Top-$K$ conflict screen as audit-only risk-disclosure bands. Enforcing $\mu_{\mathrm{gmv}}{=}\mu_{\mathrm{conv}}\mu_{\mathrm{val}}$ reduces GMV effect-estimation error by $18$–$48\%$ across the tested $\hat{p}\!\in\![5,45]\%$ range on Criteo-MT7 (Table 5); on industrial multi-arm RCT logs, FunnelCausalNet has the highest seed-averaged mean LP-frontier $\Delta\mathrm{ROI}$ at all $7/7$ reported anchors, although their correlation and the three permutation splits preclude an independent-anchor significance claim. FunnelCausalNet does not lead every semi-synthetic or public benchmark, so the evidence supports a practically important multi-tier regime rather than universal superiority. Future work includes continuous-dose extensions, doubly robust calibration under sparse converters, and broader public or online validation.

## Appendix A Proof sketch for Proposition 2 (delta method under a rate gap)

Fix $(X,T){=}(x,t)$ and let $\mu_{g}{=}\mathbb{E}[Y^{g}\mid X{=}x,T{=}t]{=}p\mu_{v}$. Under (A1)–(A2), $\mu_{g}$ is identified from the RCT logs and the population funnel composition follows by iterated expectations. Prop. 1 separates the within-converter and Bernoulli switching terms. Under (A4), the direct estimator has pointwise asymptotic variance

$\mathrm{Var}(\hat{\mu}_{g}^{\mathrm{direct}})\sim\{p\sigma_{v}^{2}+p(1-p)\mu_{v}^{2}\}/r_{n},$ | | | |

with negligible bias under standard undersmoothing. The delta method gives

$\displaystyle\mathrm{Var}(\hat{\mu}_{g}^{\mathrm{funnel}})={}$ $\displaystyle\mu_{v}^{2}\mathrm{Var}(\hat{p})+p^{2}\mathrm{Var}(\hat{\mu}_{v})$ | | | | |

$\displaystyle+2p\mu_{v}\mathrm{Cov}(\hat{p},\hat{\mu}_{v}),$ | | | | |

where $\mathrm{Var}(\hat{p}){=}O(n^{-1})$ under (A3) and $\mathrm{Var}(\hat{\mu}_{v}){=}\sigma_{v}^{2}/(pr_{n})$ under (A4). Assumption (A5) sets the covariance to zero for estimates fit on independent folds; otherwise it requires $o(1/r_{n})$. Cauchy–Schwarz gives the upper order $O((nr_{n})^{-1/2})$, which is $o(1/r_{n})$ when $r_{n}{=}o(n)$. The conv-head variance contribution is also $o(1/r_{n})$, so the funnel variance reduces to $p\sigma_{v}^{2}/r_{n}+o(1/r_{n})$ and the leading-order ratio gives Eq. (8).

This calculation assumes negligible bias. Correlated systematic errors from shared neural representations can change the finite-sample product error and are not covered by the proposition. The argument follows the standard hurdle/two-part decomposition pattern of (Cragg, 1971; Mullahy, 1986; Lambert, 1992); here it is a regime heuristic for coupon uplift rather than a neural-model guarantee.

## GenAI Usage Disclosure

In accordance with the CIKM 2026 generative-AI policy, we disclose that generative AI assistants (large language models) were used during manuscript preparation for (i) language polishing, consistency checks, and editorial restructuring of author-provided technical text, including clarification of the scope and assumptions of theoretical and empirical claims; and (ii) refactoring of plotting and CSV-aggregation utilities. The authors determined the technical contributions, derivations, experimental designs, analyses, and result interpretations; reviewed and revised all generated text and code before inclusion; and take full responsibility for the manuscript. No GenAI-generated output was used as data or experimental evidence.

## References

- Alaa and van der Schaar (2019) Ahmed M. Alaa and Mihaela van der Schaar. 2019. Validating causal inference models via conformal prediction. Proceedings of Machine Learning Research 89 (2019), 124–133.

- Albert and Goldenberg (2022) Javier Albert and Dmitri Goldenberg. 2022. E-commerce promotions personalization via online multiple-choice knapsack with uplift modeling. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management. 2864–2872. doi:10.1145/3511808.3557229

- Alibaba Tianchi (2018) Alibaba Tianchi. 2018. Tianchi O2O Coupon Usage Forecast. Public competition dataset, https://tianchi.aliyun.com/competition/entrance/231593; observational logs with coupon-redemption labels and no continuous GMV outcome.

- Athey et al. (2019) Susan Athey, Julie Tibshirani, and Stefan Wager. 2019. Generalized random forests. The Annals of Statistics 47, 2 (2019), 1148–1178. doi:10.1214/18-AOS1709

- Betlei et al. (2021) Artem Betlei, Eustache Diemert, and Massih-Reza Amini. 2021. Uplift Modeling with Generalization Guarantees. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 55–65. doi:10.1145/3447548.3467395

- Chen et al. (2024) Jiaju Chen, Wenjie Wang, Chongming Gao, Peng Wu, Jianxiong Wei, and Qingsong Hua. 2024. Treatment Effect Estimation for User Interest Exploration on Recommender Systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1861–1871. doi:10.1145/3626772.3657736

- Chernozhukov et al. (2018) Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. 2018. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal 21, 1 (2018), C1–C68. doi:10.1111/ectj.12097

- Cragg (1971) John G. Cragg. 1971. Some statistical models for limited dependent variables with application to the demand for durable goods. Econometrica 39, 5 (1971), 829–844. doi:10.2307/1909582

- Devriendt et al. (2018) Floris Devriendt, Darie Moldovan, and Wouter Verbeke. 2018. A literature survey and experimental evaluation of the state-of-the-art in uplift modeling: A stepping stone toward the development of prescriptive analytics. Big Data 6, 1 (2018), 13–41. doi:10.1089/big.2017.0104

- Diemert et al. (2018) Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini. 2018. A large scale benchmark for uplift modeling. In Proceedings of the AdKDD and TargetAd Workshop, KDD ’18. ACM, London, United Kingdom, 1–6.

- Gutierrez and Gérardy (2017) Pierre Gutierrez and Jean-Yves Gérardy. 2017. Causal inference and uplift modelling: A review of the literature. In Proceedings of the International Conference on Predictive Applications and APIs (PAPIs) (Proceedings of Machine Learning Research, Vol. 67). PMLR, 1–13.

- He et al. (2024) Bowei He, Yunpeng Weng, Xing Tang, Ziqiang Cui, Zexu Sun, Liang Chen, Xiuqiang He, and Chen Ma. 2024. Rankability-enhanced Revenue Uplift Modeling Framework for Online Marketing. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 5093–5104. doi:10.1145/3637528.3671516

- Hill (2011) Jennifer L. Hill. 2011. Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics 20, 1 (2011), 217–240. doi:10.1198/jcgs.2010.08162

- Hillstrom (2008) Kevin Hillstrom. 2008. The MineThatData E-Mail Analytics And Data Mining Challenge. Online dataset and blog post. https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html; 64 000-customer randomized email-marketing dataset widely used as an uplift-modeling benchmark.

- Huang et al. (2024) Yinqiu Huang, Shuli Wang, Min Gao, Xue Wei, Changhao Li, Chuan Luo, Yinhua Zhu, Xiong Xiao, and Yi Luo. 2024. Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing. In Companion Proceedings of the ACM Web Conference 2024. 226–234. doi:10.1145/3589335.3648320

- Kong et al. (2026) Li Kong, Bingzhe Wang, Zhou Chen, Suhan Hu, Yuchao Ma, Qi Qi, Suoyuan Song, and Bicheng Jin. 2026. SACO: Sequence-Aware Constrained Optimization Framework for Coupon Distribution in E-commerce. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. 15027–15035. doi:10.1609/aaai.v40i17.38525

- Künzel et al. (2019) Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. 2019. Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences 116, 10 (2019), 4156–4165. doi:10.1073/pnas.1804597116

- Lambert (1992) Diane Lambert. 1992. Zero-inflated Poisson regression, with an application to defects in manufacturing. Technometrics 34, 1 (1992), 1–14. doi:10.2307/1269547

- Lei and Candès (2021) Lihua Lei and Emmanuel J. Candès. 2021. Conformal inference of counterfactuals and individual treatment effects. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 83, 5 (2021), 911–938. doi:10.1111/rssb.12445

- Lenta Group (2020) Lenta Group. 2020. Lenta uplift modeling dataset. Public retail-loyalty dataset, https://github.com/maks-sh/scikit-uplift; binary SMS encouragement, no continuous-spend supervision.

- Li et al. (2020) Liangwei Li, Liucheng Sun, Chenwei Weng, Chengfu Huo, and Weijun Ren. 2020. Spending Money Wisely: Online Electronic Coupon Allocation based on Real-Time User Intent Detection. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management. 2597–2604. doi:10.1145/3340531.3412745

- Liu et al. (2023) Dugang Liu, Xing Tang, Han Gao, Fuyuan Lyu, and Xiuqiang He. 2023. Explicit Feature Interaction-aware Uplift Network for Online Marketing. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 4507–4515. doi:10.1145/3580305.3599820

- Ma et al. (2018) Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. 2018. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate. In Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval. 1137–1140. doi:10.1145/3209978.3210104

- MegaFon (2021) MegaFon. 2021. MegaFon Uplift Competition dataset. Public synthetic uplift challenge dataset; binary treatment, no tier-strength axis.

- Mondal et al. (2022) Abhirup Mondal, Anirban Majumder, and Vineet Chaoji. 2022. MEMENTO: Neural Model for Estimating Individual Treatment Effects for Multiple Treatments. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 3381–3390. doi:10.1145/3511808.3557125

- Mullahy (1986) John Mullahy. 1986. Specification and testing of some modified count data models. Journal of Econometrics 33, 3 (1986), 341–365. doi:10.1016/0304-4076(86)90002-3

- Nie and Wager (2021) Xinkun Nie and Stefan Wager. 2021. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika 108, 2 (2021), 299–319. doi:10.1093/biomet/asaa076

- Romano et al. (2019) Yaniv Romano, Evan Patterson, and Emmanuel Candès. 2019. Conformalized quantile regression. In Advances in Neural Information Processing Systems, Vol. 32. 3543–3553.

- Saito et al. (2021) Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita. 2021. Open Bandit Dataset and Pipeline: Towards realistic and reproducible off-policy evaluation. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track.

- Shalit et al. (2017) Uri Shalit, Fredrik D. Johansson, and David Sontag. 2017. Estimating individual treatment effect: Generalization bounds and algorithms. In Proceedings of the 34th International Conference on Machine Learning. PMLR, 3076–3085.

- Shi et al. (2019) Claudia Shi, David M. Blei, and Victor Veitch. 2019. Adapting Neural Networks for the Estimation of Treatment Effects. In Advances in Neural Information Processing Systems, Vol. 32.

- Sun et al. (2024) Zexu Sun, Hao Yang, Dugang Liu, Yunpeng Weng, Xing Tang, and Xiuqiang He. 2024. End-to-End Cost-Effective Incentive Recommendation under Budget Constraint with Uplift Modeling. In Proceedings of the 18th ACM Conference on Recommender Systems (RecSys ’24). 560–569. doi:10.1145/3640457.3688147

- Tu et al. (2024) Jinglong Tu, Qi Qi, Zhilin Li, and Shuanglong Fan. 2024. Data-driven real-time coupon allocation in the online platform. arXiv:2406.05987 [cs.LG]

- Vovk et al. (2005) Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. 2005. Algorithmic Learning in a Random World. Springer, New York.

- Wager and Athey (2018) Stefan Wager and Susan Athey. 2018. Estimation and inference of heterogeneous treatment effects using random forests. J. Amer. Statist. Assoc. 113, 523 (2018), 1228–1242. doi:10.1080/01621459.2017.1319839

- Wang et al. (2022) Hao Wang, Tai-Wei Chang, Tianqiao Liu, Jianmin Huang, Zhichao Chen, Chao Yu, Ruopeng Li, and Wei Chu. 2022. ESCM2: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 363–372. doi:10.1145/3477495.3531972

- Wei et al. (2024) Yuxiang Wei, Zhaoxin Qiu, Yingjie Li, Yuke Sun, and Xiaoling Li. 2024. Multi-Treatment Multi-Task Uplift Modeling for Enhancing User Growth. arXiv:2408.12803 [cs.LG]

- Weng et al. (2024) Yunpeng Weng, Xing Tang, Zhenhao Xu, Fuyuan Lyu, Dugang Liu, Zexu Sun, and Xiuqiang He. 2024. OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. doi:10.1145/3627673.3679712

- Yan et al. (2023) Ziang Yan, Shusen Wang, Guorui Zhou, Jingjian Lin, and Peng Jiang. 2023. An End-to-End Framework for Marketing Effectiveness Optimization under Budget Constraint. arXiv:2302.04477 [cs.LG]

- Yao et al. (2024) Dong Yao, Caizhi Tang, Qing Cui, and Longfei Li. 2024. Combining Incomplete Observational and Randomized Data for Heterogeneous Treatment Effects. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. doi:10.1145/3627673.3679593

- Zhang et al. (2021) Weijia Zhang, Jiuyong Li, and Lin Liu. 2021. A unified survey of treatment effect heterogeneity modelling and uplift modelling. Comput. Surveys 54, 8 (2021), 1–36. doi:10.1145/3466818

- Zhao et al. (2019) Kui Zhao, Junhao Wang, Bo Long, Jian Xu, and Kun Gai. 2019. A unified framework for marketing budget allocation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2820–2828. doi:10.1145/3292500.3330787

- Zhao et al. (2017) Yan Zhao, Xiao Fang, and David Simchi-Levi. 2017. Uplift modeling with multiple treatments and general response types. In Proceedings of the 2017 SIAM International Conference on Data Mining (SDM). 588–596. doi:10.1137/1.9781611974973.66

- Zhong et al. (2022) Kailiang Zhong, Fengtong Xiao, Yan Ren, Yaorong Liang, Wenqing Yao, Xiaofeng Yang, and Ling Cen. 2022. DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 4612–4620. doi:10.1145/3534678.3539198

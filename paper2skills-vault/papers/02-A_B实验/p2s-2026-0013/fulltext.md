<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2609.01038
     paper_id : p2s-2026-0013
     source   : https://arxiv.org/html/2609.01038v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Data-Driven Persona-Conditioned Agents for A/B Test Simulation

Ziyad Benomar Affiliation: Amazon Affiliation: Luxembourg, Luxembourg Email: zbenomar@amazon.lu    Weronika Łajewska Affiliation: Amazon Affiliation: Luxembourg, Luxembourg Email: lajewska@amazon.lu    Leonardo Perelli Affiliation: Amazon Affiliation: Luxembourg, Luxembourg Email: lperelli@amazon.lu    Saab Mansour Affiliation: Amazon Affiliation: Barcelona, Spain Email: saabm@amazon.es

###### Abstract

A/B testing is the gold standard for evaluating product changes, but each experiment requires real user traffic, engineering effort, and weeks of measurement. We propose a simulation framework that predicts A/B test outcomes using LLM-powered agents conditioned on data-driven personas grounded in real user behavioral signals. Unlike prior work that relies on synthetic or rule-based personas, our agents are constructed from anonymized behavioral data—activity patterns, engagement signals, and inferred demographics—enabling more faithful population modeling. We frame A/B test simulation as a structured question task and systematically study (i) question design formats, (ii) the impact of persona data source and domain alignment, (iii) the trade-off between per-persona behavioral depth and population diversity, and (iv) efficient population subsampling. On a benchmark of 40 A/B tests spanning two metric types, our best configuration achieves 0.75–0.90 directional accuracy depending on the test metric, demonstrating that data-driven personas are a viable path toward fast, low-cost experiment pre-screening.

## 1 Introduction

LLMs can emulate human decision-making across tasks from survey response prediction (Argyle et al., 2023; Aher et al., 2023) to preference elicitation and behavioral role-play (Wang et al., 2025; Mansour et al., 2025). When conditioned on detailed user profiles, LLM-powered agents approximate the judgments of specific population segments, enabling scalable, low-cost simulation of collective user behavior (Park et al., 2024; Bui et al., 2025). A particularly compelling application is the simulation of online controlled experiments (A/B tests): if persona-conditioned agents can reliably predict whether users prefer a treatment variant over a control, teams could pre-screen design candidates offline—reducing the time, traffic, and experimentation cost (Rieder et al., 2026; Castelo et al., 2026).

Online controlled experiments remain the gold standard for validating product changes, yet each test requires sufficient user traffic, engineering effort, and typically weeks of data collection to reach statistical significance (Kohavi et al., 2009). These costs limit how many ideas teams can evaluate. Recent work explores LLM-based simulation as a complement to live experimentation. SimAB (Rieder et al., 2026) shows that persona-conditioned agents can predict directional outcomes of A/B tests, but relies on synthetic personas generated from variant screenshots without grounding in real user behavior. SimGym (Castelo et al., 2026) replays real traffic traces through browser agents to simulate e-commerce experiments, but requires access to detailed session logs and does not study persona construction. Both systems highlight the promise of simulation, yet leave open how to construct user representations that capture the behavioral diversity of real population.

We present a simulation framework that predicts A/B test outcomes using LLM agents conditioned on data-driven personas grounded in real behavioral data. Unlike synthetic personas derived from screenshots or rule-based descriptions (Rieder et al., 2026), ours are constructed from anonymized behavioral signals spanning months of user activity (Mansour et al., 2025). This grounding lets us address questions prior work cannot: how does behavioral depth affect simulation fidelity? What is the trade-off between per-persona richness and population representativeness? How can a persona pool be subsampled without degrading quality? We frame A/B test simulation as a structured question task and study the interplay between question design, persona construction, and population sampling—providing the first empirical analysis of these factors in an A/B experimentation setting.

We evaluate our framework on a benchmark of 40 A/B tests spanning two metric types—click-through rate (CTR) and subscriptions—and organize experiments around four research questions:

-

How does the question format used to elicit agent preferences affect simulation accuracy?

-

Do data-driven personas outperform synthetic personas for A/B test simulation?

-

What is the trade-off between per-persona behavioral depth and population-level diversity?

-

How aggressively can the persona pool be subsampled while preserving simulation quality?

Overall, we make four contributions. First, we establish A/B test simulation via structured questions as a task and provide the first systematic comparison of question formats for eliciting agent preferences. Second, we show that domain alignment of persona source data drives simulation quality, with public behavioral data rivaling platform-specific personas. Third, we empirically characterize the trade-off between per-persona behavioral depth and population diversity. Fourth, we demonstrate that subsampling enables a 2$\times$ cost reduction with no measurable quality loss.

## 2 Related Work

#### LLM-based user simulation.

LLMs can reproduce human response distributions across demographic groups (Argyle et al., 2023; Aher et al., 2023) and approximate population-level behavioral patterns (Park et al., 2024; Bui et al., 2025). However, they exhibit systematic biases—positional preferences, sycophancy, majority over-representation (Bisbee et al., 2024; Dominguez-Olmedo et al., 2024)—that persona conditioning only partially mitigates (Hu and Collier, 2024; Mansour et al., 2025), especially when personas lack grounding in observed data.

#### A/B test simulation.

SimAB Rieder et al. (2026) frames A/B testing as a preference task with persona-conditioned agents, but uses entirely synthetic personas generated from variant screenshots, achieving 67% agreement on 47 tests. SimGym Castelo et al. (2026) grounds simulation in real traffic traces but requires full session logs and does not decouple persona construction from the simulation task. We differ by (i) constructing personas from real behavioral data independently of test stimuli, (ii) studying how persona pool composition affects simulation fidelity, and (iii) introducing subsampling for cost reduction.

#### Persona generation from data.

Mansour et al. (2025) induce shopping personas from anonymized behavioral data—clickstream, purchases, and pricing signals—and show that persona-conditioned agents better approximate population-level behavior than unprompted models. Ge et al. (2025) scale persona creation to one billion synthetic profiles but without grounding in real user traces. The role-playing literature Chen et al. (2024); Tseng et al. (2024) finds that richer persona descriptions improve behavioral consistency, yet the source domain and granularity of persona data matter substantially: personas induced from out-of-domain sources rarely outperform basic demographic conditioning unless well-matched to the target population. This motivates our study of how behavioral depth, domain alignment, and population representativeness interact in the A/B testing context.

#### Evaluation of simulation alignment.

Evaluating the quality of A/B test simulations can focus on predicting which version performs better (sign-based metrics) or measuring the magnitude of differences (distributional alignment metrics). We adopt a multi-faceted evaluation framework centered on sign-based metrics, including directional accuracy, sign overlap, and sign Bhattacharyya distance, which measure alignment at multiple granularities from individual experiments to distributional patterns. While no standardized evaluation framework currently exists for A/B test simulation specifically, these sign-based approaches address the core question of whether simulations can reliably inform experimental decisions. This evaluation strategy prioritizes capturing the directional patterns that drive real-world A/B testing decisions over achieving perfect numerical precision.

## 3 A/B Testing Simulation

In this section, we describe our simulation framework for predicting A/B test outcomes using persona-conditioned LLM agents. We first present the overall pipeline, then detail the three core components: question design for eliciting agent preferences, data-driven persona construction, and population sampling for cost-efficient simulation.

### 3.1 Problem Formulation

Given an A/B test comparing control $C$ against treatment $T$ on a metric such as click-through rate or subscriptions, the ground truth is the sampling distribution of the relative treatment effect $\hat{\delta}_{\text{rel}}=\frac{\hat{\mu}_{t}-\hat{\mu}_{c}}{\hat{\mu}_{c}}$, represented as $\mathcal{N}(\hat{\delta}_{\text{rel}},\hat{\sigma}_{\text{rel}}^{2})$. Our objective is to predict the sign of this effect—whether the treatment improves, harms, or has negligible impact on the metric—without running the experiment on real users. We simulate this by deploying persona-conditioned LLM agents that score each variant, producing a predicted relative effect distribution $\mathcal{N}(\hat{\delta}_{s},\hat{\sigma}_{s}^{2})$ whose sign alignment with the ground truth measures simulation quality. See Appendix A for full derivation.

### 3.2 Framework Overview

Our framework consists of three stages:

#### Persona construction.

Each persona is constructed from anonymized, aggregated behavioral signals. An LLM processes these signals into a structured persona (inferred demographics, behavioral summaries, engagement patterns, synthesized narratives, etc.) that serves as conditioning context during simulation. Persona pool construction trades off behavioral depth against population diversity; subsampling strategies reduce redundancy for cost-efficient inference (Sections 3.4, 3.5).

#### Simulation.

Given an A/B test with control $C$ and treatments ${T_{1},\ldots,T_{k}}$, each represented as a visual stimulus, and a target metric, we present the variants to each persona-conditioned agent via a structured question. The agent is asked to evaluate them with respect to a target metric and return a numeric score for each variant along with free-text reasoning grounded in its persona profile. The question format—how variants are presented and how responses are elicited—is a design choice that affects simulation quality (Section 3.3).

#### Aggregation & evaluation.

Per-persona scores are aggregated into a predicted distribution of the treatment effect $\mathcal{N}(\hat{\delta}_{s},\hat{\sigma}_{s})$, where $\hat{\delta_{s}}=\bar{s}(T)-\bar{s}(C)$, $\bar{s}(\cdot)$ denotes the mean score across all personas, and $\hat{\sigma}_{s}$ the standard error (SE) of the distribution of score differences over the personas. We evaluate sign alignment using three metrics (Section 3.6).

### 3.3 Question Design

The question design determines how variants are shown and how preferences are elicited. We study four formats varying along two axes—isolation vs. comparison (whether the agent sees one variant or both) and binary vs. rating (whether the response is yes/no or a 1–10 score). In independent formats, each variant is shown separately; in pairwise formats, both variants are presented together with order randomized per persona to control positional bias. Rating formats provide finer-grained signal than binary responses. Full question design descriptions are in Appendix B.1.

### 3.4 Data-Driven Persona Construction

We construct personas by processing anonymized behavioral records through an LLM, which synthesizes raw signals into structured profiles capturing inferred demographics, behavioral statistics, activity patterns, and shopping characteristics (see Appendix I.3 for full list of attributes). A key design choice is the composition of the persona pool. Two competing objectives arise: behavioral depth—maximizing the amount of data available per persona to produce richer, more reliable profiles (deep pool)—and population diversity—ensuring the pool covers the demographic and behavioral breadth of the target user base (representative pool). These objectives are in tension as highly active users yield richer personas but may not be representative of the broader population.

### 3.5 Population Sampling

Running all personas through the simulation is expensive: inference cost scales linearly with pool size, and large pools often contain redundant profiles that contribute minimal additional signal. We study subsampling strategies that select a smaller subset of personas while preserving the distributional properties of the full pool. We compare three algorithms: uniform random sampling, Kernel Herding (Chen et al., 2012), which greedily selects personas minimizing maximum mean discrepancy to the full pool, and Greedy Farthest sampling (Eldar et al., 1997), which iteratively selects the persona farthest from all the previously selected ones.

### 3.6 Evaluation Metrics

Let $p=\Phi(\hat{\delta}/\hat{\sigma})$ and $q=\Phi(\hat{\delta}_{s}/\hat{\sigma}_{s})$ denote the probability of positive effect under ground truth and simulation respectively. We report: Accuracy (Acc): $\mathbb{I}[(p{-}0.5)(q{-}0.5)>0]$; Sign overlap (SignOv): $1-|p-q|$; Sign Bhattacharyya (SignBC): $(\sqrt{pq}+\sqrt{(1{-}p)(1{-}q)})^{2}$. Acc captures binary correctness; SignOv and SignBC additionally measure confidence calibration. See Appendix C for detailed interpretation of the metrics.

## 4 Experimental Setup

We evaluate our simulation framework on a sample of marketing A/B tests from e-commerce platform.

### 4.1 Benchmark Construction

Our benchmark comprises 40 experimental test pairs evaluating visual design variants, curated from a sample of historical e-commerce interaction data. The benchmark reflects a curated experimental sample and should not be interpreted as representative of any specific platform’s full user base or operational A/B testing infrastructure. Each test compares a control widget against one or more treatment variants, with estimated treatment directions provided as confidence intervals on the metric difference (Treatment $-$ Control). These confidence intervals define a Gaussian distribution $\mathcal{N}(\mu,\sigma)$ over the true effect size, from which we derive directional labels. We classify each test in terms of the certainty of its ground-truth outcome as: positive ($\Pr(\delta>0)>\tau$), negative ($\Pr(\delta<0)>\tau$), or negligible ($\Pr(|\delta|<\epsilon)>\tau$), excluding indecisive cases. Thresholds $\tau$ and $\epsilon$ are calibrated per test metric to reflect observed effect size distributions ($\tau=0.8$, $\epsilon=0.01$ for CTR; $\tau=0.7$, $\epsilon=0.1$ for subscriptions). The original candidate set contained over 50 CTR tests and 40 subscription tests; applying these thresholds excluded tests with ambiguous ground truth, yielding the final benchmark of 40 tests (20 per metric). The benchmark spans two metric types: click-through rate (engagement) and subscriptions (sign-up intent), evaluated with the same pipeline but different question framing.

### 4.2 Personas Pool Construction

Personas are constructed from anonymized, aggregated behavioral signals from an e-commerce platform via an LLM that generates structured profiles. We compare two persona pools (both containing 935 personas). Deep pool contains personas sourced from de-identified activity patterns with high engagement levels, yielding dense behavioral histories. The richness of the source data produces detailed, high-confidence profiles but skews the pool toward power users who may not be representative of the broader population. Representative pool is a set of users sampled via stratified sampling across population segments to ensure diversity. Individual personas carry sparser behavioral data, but the pool collectively covers a wider range of demographic and behavioral profiles. This pool tests whether breadth of coverage can compensate for reduced per-persona depth at the aggregate simulation level. Additionally, we construct persona pools (each containing 1000 personas) from three public sources—social science survey data (Kolluri et al., 2025), movie ratings (Leone, 2024), and open e-commerce transactions (Berke et al., 2024)—to serve as baselines for the data-driven approach and to test whether personas based on public data can rival platform-specific personas.

### 4.3 Implementation Details

All experiments use Claude Sonnet 4.5 (Anthropic, 2025c) via cloud-based batch inference. Persona generation uses two sequential LLM flows (demographics + behavioral narratives) with structured output schemas. Simulation uses temperature 0 for reproducibility, JSON-constrained responses via assistant prefilling, and per-persona randomization of test variant order (see Appendix I).

## 5 Results

We present results organized around the research questions from Section 1, evaluating question design (RQ1), persona data sources (RQ2), the depth–diversity trade-off in persona pool construction (RQ3), and population sampling efficiency (RQ4). In all the following, to assess statistical significance between simulation variants, we apply one-sided paired $t$-tests with $\alpha{=}0.05$

### 5.1 Question Design Comparison

| Question Design | CTR | Subscriptions |

| Acc | SignOv | Acc | SignOv |

$\pm$ $\pm$ $\pm$ $\pm$| Indep. binary | 0.40.11 | 0.51.09 | 0.40.11 | 0.56.07 |

$\pm$ $\pm$ $\pm$ $\pm$| Pairwise binary | 0.45.11 | 0.50.09 | 0.80.08 | 0.73.07 |

$\pm$ $\pm$ $\pm$ $\pm$| Indep. rating | 0.40.11 | 0.52.10 | 0.45.11 | 0.47.08 |

$\pm$ $\pm$ $\pm$ $\pm$| Pairwise rating | 0.75.10 | 0.68.09 | 0.80.09 | 0.72.07 |

*Table 1: Question design impact on simulation accuracy comparing independent (Indep.), pairwise, binary and rating combinations (mean $\pm$ standard error (SE), $n{=}20$ tests per use case). *

This section addresses RQ1: the impact of question format on simulation accuracy (Table 1). Pairwise rating consistently outperforms alternatives across all metrics (one-sided paired $t$-tests, $\alpha{=}0.05$). Independent formats perform poorly, suggesting agents struggle to calibrate scores without comparative context. Pairwise comparison provides a natural reference point that reduces anchoring, while numeric scales capture preference intensity beyond binary choices. The binary pairwise format achieves strong results on subscription tests but fails on CTR, indicating that optimal design depends on metric type. On the combined benchmark ($n=40$), pairwise rating significantly outperforms independent binary on Acc and SignOv, significantly outperforms independent rating on all three metrics, and significantly outperforms pairwise binary on Acc and SignBC. The gap with pairwise binary on SignOv does not reach significance.

#### Takeaway:

Pairwise rating is the most effective question format, achieving 0.75 accuracy on CTR and 0.80 on subscription tests. All subsequent experiments use this format.

### 5.2 Synthetic vs Data-Driven Personas

| Persona Source | CTR | Subscriptions |

| Acc | SignOv | Acc | SignOv |

$\pm$ $\pm$ $\pm$ $\pm$| Survey data | 0.60.11 | 0.58.10 | 0.75.10 | 0.68.07 |

$\pm$ $\pm$ $\pm$ $\pm$| Rotten Tomatoes | 0.65.11 | 0.63.09 | 0.60.11 | 0.57.08 |

$\pm$ $\pm$ $\pm$ $\pm$| Open e-comm. | 0.70.11 | 0.67.09 | 0.90.07 | 0.77.06 |

$\pm$ $\pm$ $\pm$ $\pm$| Proprietary | 0.75.10 | 0.68.09 | 0.80.09 | 0.72.07 |

*Table 2: Simulation accuracy across persona sources (mean $\pm$ SE, $n{=}20$ tests per use case). Full results in Appendix F.*

This section addresses RQ2: whether data-driven personas outperform synthetic personas (Table 2). Personas constructed from platform behavioral data achieve competitive results on CTR tests (0.70 accuracy, 0.64 SignOv) and competitive performance on subscription tests. Among the external persona sources, open e-commerce data performs best and surpasses platform data on subscription tests (0.90 vs. 0.80 accuracy), likely due to domain alignment—e-commerce browsing and purchasing signals are directly relevant to evaluating widget engagement and subscription intent. Rotten Tomatoes personas, grounded in entertainment preferences, show reasonable performance on CTR tests (0.65) but degrade on subscription tests (0.60), suggesting that out-of-domain behavioral data provides insufficient signal for metric-specific predictions. Survey-based personas perform moderately without excelling on either metric.

#### Takeaway:

Domain alignment is crucial for personas effectiveness. In-domain behavioral data achieves 0.70–0.90 accuracy, while out-of-domain sources drop to 0.57–0.69. Public e-commerce data can rival platform-specific personas.

### 5.3 Persona Pool Comparison

| Persona Pool | CTR | Subscriptions |

| Acc | SignOv | Acc | SignOv |

$\pm$ $\pm$ $\pm$ $\pm$| Deep | 0.75.10 | 0.68.09 | 0.80.09 | 0.72.07 |

$\pm$ $\pm$ $\pm$ $\pm$| Representative | 0.60.11 | 0.69.09 | 0.80.09 | 0.73.07 |

*Table 3: Simulation accuracy across persona pools with large purchases number (Deep) vs representative (mean $\pm$ SE, $n{=}20$ tests per use case). Full results in Appendix F.*

*Figure 1: Persona pool comparison: (a) behavioral depth (total # transactions), (b) data sparsity, (c) demographic diversity on key under-represented segments, and (d) age distribution with concentration bias.*

This section addresses RQ3: the trade-off between per-persona behavioral depth and population-level diversity (Table 3). The deep pool significantly outperforms the representative pool on CTR accuracy (0.75 vs. 0.60), while on subscription tests there is no statistically significant difference on any metric (0.80 accuracy for both). Despite the accuracy gap on CTR tests, the distributional metrics differ only marginally (SignBC 0.69 vs. 0.67), suggesting that demographic coverage largely compensates for reduced behavioral depth at the population level. The deep pool is characterized by dense interaction histories vs. sparse signals for the representative pool (Figure 1; panel a), with a substantial portion of representative personas having minimal interaction data (panel b). We hypothesize that for sparse personas, the LLM lacks sufficient behavioral grounding and defaults to generic reasoning rather than user-specific preferences, explaining the CTR accuracy gap. The representative pool compensates through demographic diversity (panel c): it introduces substantially more renters, non-college-educated users, and younger demographics that are nearly absent from the deep pool, which skews toward homeowning, educated users. Both pools exhibit complementary demographic gaps (panel d): the representative pool over-concentrates in the 30–40 age range with minimal older representation, while the deep pool is more spread but still under-represents younger users. These gaps likely arise from LLM inference biases during persona generation rather than sampling limitations, and represent a primary lever for future improvement.

#### Takeaway:

Behavioral depth yields a statistically significant advantage on CTR accuracy, but demographic diversity fully compensates on subscription tests (no significant difference on any metric). The remaining gap points to persona quality—sparse-data handling and demographic calibration.

### 5.4 Population Sampling Efficiency

| | CTR | Subscriptions |

| Algorithm | Acc | SignOv | Acc | SignOv |

| Random | 0.74 | 0.68 | 0.81 | 0.70 |

| Kernel Herding | 0.73 | 0.67 | 0.81 | 0.70 |

| Greedy Farthest | 0.75 | 0.68 | 0.80 | 0.69 |

| All (935) | 0.75 | 0.68 | 0.80 | 0.72 |

*Table 4: Subsampling performance at 500 personas (mean $\pm$ SE across 1000 trials, all SEs are below 0.005 hence omitted for brevity). Full results in Appendix F.*

This section addresses RQ4: the trade-off between persona pool size reduction via subsampling and preserving simulation quality. We compare three subsampling strategies—random, Greedy Farthest, and Kernel Herding—for selecting representative personas from a pool of 935 personas. These algorithms operate on vector representations, we apply them to persona embeddings computed using Qwen2.5-32B-Instruct (Team, 2024). Detailed algorithm descriptions are provided in Appendix H.

Table 4 presents the performance metrics when sampling 500 personas from the pool of 935 using each algorithm. Results are averaged over 1000 independent trials. At 500 personas, Kernel Herding performs best for subscription tests while Greedy Farthest is best for CTR. However, even random sampling achieves strong performance, suggesting reduced persona sets provide robust signals.

#### Takeaway:

All subsampling strategies preserve near-full-pool accuracy at 500 personas (within 1pp on CTR, matching or exceeding on subscriptions), potentially enabling up to 2$\times$ cost reduction. Pairwise comparison format provides natural robustness, though geometric methods offer consistent modest gains, particularly for sign-based metrics.

### 5.5 Ablation: Component Contributions

To isolate the contribution of each framework component, we evaluate four baselines using the pairwise rating format (Table 5). Two baselines use the full deep persona pool (935 personas), modifying the persona representation or prompting strategy: demographics only strips the persona to demographic attributes (age, gender, education, income, etc.); no reasoning uses the full persona but instructs the model to output only scores without reasoning (no chain-of-thought requirement). Two baselines test minimal-conditioning scenarios: generic shopper conditions the model on a single fixed persona applied uniformly across all tests; no persona removes persona conditioning entirely.

| Configuration | CTR | Subscriptions |

| Acc | SignOv | Acc | SignOv |

$\pm$ $\pm$ $\pm$ $\pm$| Pairwise rating | 0.75.10 | 0.68.09 | 0.80.09 | 0.72.07 |

$\pm$ $\pm$ $\pm$ $\pm$| Demographics | 0.65.11 | 0.57.08 | 0.30.11 | 0.39.07 |

$\pm$ $\pm$ $\pm$ $\pm$| No reasoning | 0.60.11 | 0.60.09 | 0.80.09 | 0.73.06 |

$\pm$ $\pm$ $\pm$ $\pm$| Generic shopper | 0.40.11 | 0.44.10 | 0.60.11 | 0.64.08 |

$\pm$ $\pm$ $\pm$ $\pm$| No persona | 0.45.11 | 0.49.10 | 0.65.11 | 0.63.08 |

*Table 5: Ablation studies comparing the best system (pairwise rating) vs demographics only and no reasoning using 935 personas. We include "non-simulation" baselines that do not use personas (single LLM call with generic shopper and no persona). *

Full behavioral personas achieve +25–30pp accuracy lift on CTR and +15–20pp on subscriptions over unconditioned baselines. Demographics-only degrade substantially on subscriptions (0.30 vs. 0.80), indicating that behavioral profiles are critical for higher-salience decisions. Removing reasoning drops CTR accuracy from 0.70 to 0.60 while subscription accuracy is preserved (0.80), suggesting that CTR prediction benefits from explicit persona-grounded deliberation. A single generic persona does not outperform the unconditioned LLM (0.40 vs. 0.45 on CTR, 0.60 vs. 0.65 on subscriptions), confirming that population diversity—not mere persona framing—drives the improvement.

#### Takeaway:

Persona conditioning provides substantial lift over simpler alternatives. Behavioral depth beyond demographics is most critical for subscription tests, while chain-of-thought reasoning primarily benefits CTR prediction. Population diversity is essential—a single generic persona offers no advantage over no persona at all.

### 5.6 Multi-Model Validation

To assess whether our findings generalize beyond a single LLM, we replicate the question design comparison (Table 1) with two additional models: Claude Haiku (Anthropic, 2025a) and Claude Opus (Anthropic, 2025b). See appendix D for detailed results and discussion.

#### Takeaway:

Pairwise formats consistently outperform independent formats across all three models. The choice between pairwise rating and pairwise binary is model-dependent, but pairwise rating offers greater interpretability at comparable accuracy.

## 6 Discussion

#### Potential applications.

With current accuracy levels, the proposed framework cannot fully replace human A/B tests—but it does not need to. A potential application could be a pre-screening tool that filters clearly inferior treatment candidates before they consume traffic and prioritizes the experiments by ranking proposed changes by predicted impact. With batch inference, a full simulation could complete in hours at a fraction of a multi-week experiment cost, potentially enabling teams to explore a broader design space without proportionally increasing experimentation overhead.

#### When to trust simulation.

Our experiments show that simulations are most reliable when the underlying effect size is large, as is typical of high-salience decisions. Conversely, the framework is least trustworthy for near-zero effects where small perturbations flip the predicted direction. This suggests that outputs could be more useful as a ranking signal rather than a binary decision criterion.

#### Cost–quality trade-off.

Subsampling to 500 personas performs within 1pp of the full pool across all metrics for subscription tests, and matches the full pool on CTR. Even at $n{=}100$, all algorithms remain competitive (0.77–0.80 CTR, 0.82–0.86 subscriptions), making routine simulation viable under constrained budgets.

#### Data requirements.

Domain alignment matters more than data volume or source exclusivity. Public e-commerce data rivals platform-specific personas (Table 2), lowering adoption barriers. However, below ${\sim}20$ recorded transactions, simulation degrades as the LLM defaults to generic reasoning.

#### Enabling reproducible research.

While our benchmark relies on proprietary A/B test outcomes, the methodology is fully reproducible with public resources. Any preference dataset with ground-truth rankings—movie ratings, product reviews, website redesign preferences—can be reframed as a simulated A/B test by treating item pairs as control/treatment variants and user preferences as ground-truth effect directions. Extended discussion and implementation guidelines are provided in Appendix E.

## 7 Conclusions

We presented a framework for simulating A/B tests using LLM agents conditioned on data-driven personas constructed from real behavioral data. Through systematic evaluation on 40 A/B tests across two metric types, we established four findings: (i) pairwise rating is the most effective question format for eliciting agent preferences, achieving 0.75–0.80 directional accuracy; (ii) domain alignment of persona source data is the primary driver of simulation quality, with publicly available e-commerce data rivaling platform-specific personas; (iii) demographic diversity can largely compensate for reduced per-persona behavioral depth at the aggregate level; and (iv) Kernel Herding enables efficient subsampling to 500 personas with negligible quality loss. These results suggest that data-driven persona simulation could serve as a viable tool for pre-screening A/B test candidates, potentially reducing wasted experimentation traffic while maintaining directional accuracy that may be sufficient for prioritization decisions.

## Limitations

Our framework evaluates agents on isolated screenshots rather than full page contexts, removing situational factors (browsing intent, session history, surrounding content) that influence real user decisions. The benchmark is limited to 40 tests from a single e-commerce domain and two metric types; generalization to other domains, metric types, or multi-step user journeys remains untested. Persona demographics are inferred by the LLM from behavioral signals rather than self-reported, introducing systematic biases that may distort population-level predictions. LLM positivity bias and anchoring effects likely produce systematically optimistic treatment evaluations, a tendency that our sign-based metrics partially mask when both control and treatment are equally inflated. Persona conditioning may also amplify affective alignment (increasing emotional validation and hedging in model responses) even when epistemic independence is preserved (Kelley and Riedl, 2026). Our pairwise format partially mitigates this, as uniform positivity bias cancels when computing relative treatment effects. Additionally, inferred demographics depend on how identity is cued through behavioral signals; different cue types can yield inconsistent demographic conditioning (Tonneau et al., 2026), and our pipeline relies on a single cue modality (aggregated purchase behavior). All experiments use a single LLM (Claude Sonnet 4.5); while we validate consistency across Claude Haiku and Opus (Appendix D), generalization to non-Anthropic models remains untested. Few-shot prompting was not explored, as no ground-truth persona $\rightarrow$ score pairs exist to serve as demonstrations without risking score anchoring. Finally, the benchmark ground-truth labels are not publicly released, limiting external reproducibility; however, the methodology is fully reproducible with any preference dataset, as we demonstrate with public data sources that produce competitive results.

## Ethical Considerations

All behavioral data used for persona construction is anonymized and de-identified prior to processing; no personally identifiable information is included in the pipeline. Inferred demographic attributes (age, gender, income) are generated by the LLM as approximate statistical priors — they may not reflect users’ actual characteristics and are not validated against self-reported data. These attributes serve solely as conditioning signals for population-level simulation and are not attributed to individuals or used for any downstream profiling or targeting purpose. The persona generation pipeline does not retain explicit links to source records.

## References

- Aher et al. (2023) G. Aher, R. I. Arriaga, and A. T. Kalai Using large language models to simulate multiple humans and replicate human subject studies. In Proceedings of the 40th International Conference on Machine Learning, ICML ’23. External Links: Link Cited by: §1, §2.

- Anthropic (2025a) Anthropic Introducing Claude Haiku 4.5. Note: https://www.anthropic.com/news/claude-haiku-4-5 Cited by: Appendix D, §5.6.

- Anthropic (2025b) Anthropic Introducing Claude Opus 4.5. Note: https://www.anthropic.com/news/claude-opus-4-5 Cited by: Appendix D, §5.6.

- Anthropic (2025c) Anthropic Introducing Claude Sonnet 4.5. Note: https://www.anthropic.com/news/claude-sonnet-4-5 Cited by: §I.1, §4.3.

- Argyle et al. (2023) L. P. Argyle, E. C. Busby, N. Fulda, J. R. Gubler, C. Rytting, and D. Wingate Out of one, many: using language models to simulate human samples. Political Analysis 31 (3), pp. 337–351. External Links: Link Cited by: §1, §2.

- Berke et al. (2024) A. Berke, D. Calacci, R. Mahari, T. Yabe, K. Larson, and S. Pentland Open e-commerce 1.0: Five years of crowdsourced U.S. Amazon purchase histories with user demographics. Sci Data 11 (491). External Links: Link Cited by: Appendix E, §4.2.

- Bisbee et al. (2024) J. Bisbee, J. D. Clinton, C. Dorff, B. Kenkel, and J. M. Larson Synthetic replacements for human survey data? The perils of large language models. Political Analysis 32 (4), pp. 401–416. Cited by: §2.

- Bui et al. (2025) N. Bui, H. T. Nguyen, S. Kumar, J. Theodore, W. Qiu, V. A. Nguyen, and R. Ying Mixture-of-personas language models for population simulation. In Findings of the Association for Computational Linguistics: ACL 2025, ACL ’25, pp. 24761–24778. External Links: Link Cited by: §1, §2.

- Castelo et al. (2026) A. Castelo, Z. Z. Foumani, A. Fan, K. Y. Koay, V. Malik, Y. Zhu, H. Li, M. Feghhi, R. Uliana, S. Xie, Z. Zhang, A. O. Martins, M. Zhao, F. Pelland, J. Faerman, N. LeBlanc, A. Glazer, A. McNamara, L. Wang, and Z. Wu SimGym: traffic-grounded browser agents for offline A/B testing in e-commerce. External Links: 2602.01443, Link Cited by: §1, §1, §2.

- Chen et al. (2024) J. Chen, X. Wang, R. Xu, S. Yuan, Y. Zhang, W. Shi, J. Xie, S. Li, R. Yang, T. Zhu, A. Chen, N. Li, L. Chen, C. Hu, S. Wu, S. Ren, Z. Fu, and Y. Xiao From persona to personalization: a survey on role-playing language agents. External Links: 2404.18231, Link Cited by: §2.

- Chen et al. (2012) Y. Chen, M. Welling, and A. Smola Super-samples from kernel herding. External Links: 1203.3472, Link Cited by: 3rd item, §3.5.

- Dominguez-Olmedo et al. (2024) R. Dominguez-Olmedo, M. Hardt, and C. Mendler-Dünner Questioning the survey responses of large language models. In Advances in Neural Information Processing Systems, NeurIPS ’24, pp. 45850–45878. External Links: Link Cited by: §2.

- Eldar et al. (1997) Y. Eldar, M. Lindenbaum, M. Porat, and Y. Y. Zeevi The farthest point strategy for progressive image sampling. IEEE transactions on image processing 6 (9), pp. 1305–1315. External Links: Link Cited by: §3.5.

- Ge et al. (2025) T. Ge, X. Chan, X. Wang, D. Yu, H. Mi, and D. Yu Scaling synthetic data creation with 1,000,000,000 personas. External Links: 2406.20094, Link Cited by: §2.

- Goldberg et al. (2001) K. Goldberg, T. Roeder, D. Gupta, and C. Perkins Eigentaste: a constant time collaborative filtering algorithm. Information Retrieval 4 (2), pp. 133–151. External Links: Link Cited by: Appendix E.

- Harper and Konstan (2015) F. M. Harper and J. A. Konstan The MovieLens datasets: history and context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5 (4), pp. 1–19. External Links: Link Cited by: Appendix E.

- Hu and Collier (2024) T. Hu and N. Collier Quantifying the persona effect in LLM simulations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL ’24, pp. 10289–10307. External Links: Link Cited by: §2.

- Kelley and Riedl (2026) S. W. Kelley and C. Riedl Personalization increases affective alignment but has role-dependent effects on epistemic independence in llms. External Links: 2603.00024, Link Cited by: Limitations.

- Kohavi et al. (2009) R. Kohavi, R. Longbotham, D. Sommerfield, and R. M. Henne Controlled experiments on the web: survey and practical guide. Data Min. Knowl. Discov. 18 (1), pp. 140–181. External Links: Link Cited by: §1.

- Kolluri et al. (2025) A. Kolluri, S. Wu, J. S. Park, and M. S. Bernstein Finetuning LLMs for human behavior prediction in social science experiments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP ’25, pp. 30096–30111. External Links: Link Cited by: §4.2.

- Leone (2024) S. Leone Rotten tomatoes movies and critic reviews dataset. Note: Kaggle External Links: Link Cited by: §4.2.

- Mansour et al. (2025) S. Mansour, L. Perelli, L. Mainetti, G. Davidson, and S. D’Amato PAARS: persona aligned agentic retail shoppers. In Proceedings of the 1st Workshop for Research on Agent Language Models (REALM 2025), REALM ’25, pp. 143–159. External Links: Link Cited by: §1, §1, §2, §2.

- Park et al. (2024) J. S. Park, C. Q. Zou, A. Shaw, B. M. Hill, C. Cai, M. R. Morris, R. Willer, P. Liang, and M. S. Bernstein Generative agent simulations of 1,000 people. External Links: 2411.10109v1 Cited by: §1, §2.

- Rieder et al. (2026) T. Rieder, M. Schneider, M. Truss, V. Tsaplin, A. Rublea, S. Dere, F. C. Sanz, T. Reiss, and M. D. Dogan SimAB: simulating A/B tests with persona-conditioned ai agents for rapid design evaluation. External Links: 2603.01024, Link Cited by: Appendix G, §1, §1, §1, §2.

- Team (2024) Q. Team Qwen2.5: a party of foundation models. External Links: Link Cited by: Appendix H, §5.4.

- Tonneau et al. (2026) M. Tonneau, N. K. R. Seghal, N. Malhotra, S. Kazemi, V. Orozco-Olvera, A. M. M. Boudet, L. Subramanian, S. P. Fraiberger, S. C. Guntuku, and V. Hofmann Different demographic cues yield inconsistent conclusions about llm personalization and bias. External Links: 2601.18486, Link Cited by: Limitations.

- Tseng et al. (2024) Y. Tseng, Y. Huang, T. Hsiao, W. Chen, C. Huang, Y. Meng, and Y. Chen Two tales of persona in LLMs: a survey of role-playing and personalization. In Findings of the Association for Computational Linguistics: EMNLP 2024, EMNLP ’24, pp. 16612–16631. External Links: Link Cited by: §2.

- Wang et al. (2025) L. Wang, J. Zhang, H. Yang, Z. Chen, J. Tang, Z. Zhang, X. Chen, Y. Lin, H. Sun, R. Song, X. Zhao, J. Xu, Z. Dou, J. Wang, and J. Wen User behavior simulation with large language model-based agents. ACM Trans. Inf. Syst. 43 (2), pp. 1–37. External Links: Link Cited by: §1.

- Ziegler et al. (2005) C. Ziegler, S. M. McNee, J. A. Konstan, and G. Lausen Improving recommendation lists through topic diversification. In Proceedings of the 14th international conference on World Wide Web, pp. 22–32. External Links: Link Cited by: Appendix E.

## Appendix A Problem Formulation: Full Derivation

#### A/B testing.

A/B testing compares a control group ($C$) and treatment group ($T$) on metrics such as click-through rate (CTR). The experimental framework tests the null hypothesis $H_{0}:\mu_{t}=\mu_{c}$ against the alternative $H_{1}:\mu_{t}\neq\mu_{c}$, where $\mu_{t}$ and $\mu_{c}$ represent the true treatment and control CTRs. The design requires pre-specifying Type I error $\alpha$ (typically 0.05) and Type II error $\beta$, which together with the minimum detectable effect determine required sample sizes.

Upon completion, the experiment yields an observed effect size $\hat{\delta}=\hat{\mu}_{t}-\hat{\mu}_{c}$. By the Central Limit Theorem, this estimator follows:

$\hat{\delta}\sim\mathcal{N}\left(\delta_{true},\frac{\sigma_{t}^{2}}{n_{t}}+\frac{\sigma_{c}^{2}}{n_{c}}\right)$ | | | |

For simulation purposes, we establish ground truth as the sampling distribution of the estimator, centered on the observed effect $\mathcal{N}\left(\hat{\delta}_{obs},\frac{\hat{\sigma}_{t}^{2}}{n_{t}}+\frac{\hat{\sigma}_{c}^{2}}{n_{c}}\right)$. This distribution represents what would be observed across repeated identical experiments and serves as the benchmark for evaluating agent predictions. It captures both the best available estimate of the true effect ($\hat{\delta}_{obs}$) and the appropriate level of uncertainty due to sampling variability.

#### Simulated A/B testing.

We consider A/B testing tasks where control $C$ and treatment $T$ variants are visual stimuli such as advertisement designs or website layouts, evaluated on metrics like click-through rate. The objective is to predict the distribution of the treatment effect size that would emerge from human population testing. To simulate this without requiring actual human participants, we employ LLM-based agents operating under distinct personas. Each persona provides contextual background for an agent to role-play when presented with $C$ versus $T$, generating responses that approximate different human user segments. The simulation outputs a predicted normal distribution of the mean effect size, characterized by estimated mean and standard error. Simulation quality is measured by how closely this predicted distribution matches the true effect size distribution obtained from actual human A/B testing.

## Appendix B Question Design: Extended Analysis

### B.1 Designs Description

We frame A/B test simulation as a structured question task: each persona-conditioned agent is presented with variant screenshots and asked to evaluate them with respect to a target metric. The question format determines how variants are shown and how preferences are elicited, introducing a design choice that affects both the cognitive framing for the LLM and the resulting simulation quality. We study four question designs:

-

Independent binary: Each variant is shown separately, and the agent decides whether they would engage or not, giving only a binary signal. Responses to different variants are fully independent and unanchored, closely mimicking real-world user behavior.

-

Independent rating: Similar to the binary setting, each variant is shown in isolation, and here the agent is asked to rate it on a 1–10 Likert scale without seeing alternatives. This provides greater response granularity and signal strength.

-

Pairwise binary: Each question presents a pair of variants, asking the agent whether they would engage with each one. This enables direct comparison between the two options. Variant order is randomized per persona to mitigate positional bias.

-

Pairwise rating: As in pairwise binary, every pair of variants is presented simultaneously, but agents rate each option on a 1–10 Likert scale rather than making binary engagement decisions.

The formats differ in whether the agent sees variants in isolation or in context of alternatives. This choice interacts with the LLM’s tendency toward anchoring and positional bias, making question design a non-trivial factor in simulation accuracy.

## Appendix C Evaluation Metrics: Intuition via Toy Examples

The human A/B test outcome represents ground truth as a normal distribution $\mathcal{N}(\hat{\delta}_{\text{rel}},\hat{\sigma}_{\text{rel}}^{2})$, while the simulation produces a predicted distribution $\mathcal{N}(\hat{\delta}_{s},\hat{\sigma}_{s}^{2})$. This predicted distribution is obtained by computing the relative effect predicted by each persona, then calculating the mean and standard error across these persona-level relative scores. Since we focus on sign alignment (i.e., predicting which version drives higher engagement), we evaluate whether both distributions agree on the direction of the effect. Let $p=\Phi\left(\frac{\hat{\delta}_{\text{rel}}}{\hat{\sigma}_{\text{rel}}}\right)$ denote the probability that the ground truth relative effect is positive, and $q=\Phi\left(\frac{\hat{\delta}_{s}}{\hat{\sigma}_{s}}\right)$ the probability that the simulated effect is positive, where $\Phi$ is the standard normal CDF. We compare $p$ and $q$ using three complementary metrics:

-

Simple accuracy: $\mathbb{I}[(p-0.5)(q-0.5)>0]$, which equals 1 if and only if $\hat{\delta}$ and $\hat{\delta}_{s}$ have the same sign.

-

Sign overlap: $\min(p,q)+\min(1-p,1-q)=1-|p-q|$. This smooth metric represents the probability mass where both distributions agree on sign. It equals 1 when $p=q$ (perfect agreement) and 0 when one probability is 0 and the other is 1 (complete disagreement). It also corresponds to 1 minus the total variation distance between the corresponding Bernoulli distributions.

-

Sign Bhattacharyya coefficient: $\text{BC}^{2}=(\sqrt{pq}+\sqrt{(1-p)(1-q)})^{2}$. This metric also achieves its maximum of 1 when $p=q$ but penalizes disagreement differently than sign overlap. While sign overlap penalizes linearly in $|p-q|$, the Bhattacharyya coefficient provides a more gradual penalty that is less sensitive to small deviations when both $p$ and $q$ are near the boundaries (0 or 1). Note that this metric also equals the squared cosine similarity between $(\sqrt{p},\sqrt{1-p})$ and $(\sqrt{q},\sqrt{1-q})$.

*Figure 2: Sign metrics for ground truth and predicted effect size distribution.*

Figure 2 compares the metrics Acc, SignBC, and SignOv for different input Gaussians. As shown in the figure, Acc can be highly noisy, especially when the means of both distributions are near zero. In such cases, small perturbations can flip the sign of either distribution, causing accuracy to change abruptly from 0 to 1 or vice versa. Moreover, accuracy does not account for the uncertainty in either distribution. In contrast, SignOv and SignBC exhibit smoother behavior that varies continuously with respect to the input distributions.

### C.1 Magnitude Alignment Metrics

In practice, the primary decision derived from an A/B test is whether to ship a treatment—a decision that hinges almost entirely on the direction (sign) of the effect, not its precise magnitude. A simulation that correctly predicts that variant B increases engagement by some positive amount provides actionable guidance, even if it overestimates or underestimates the exact lift. Conversely, a simulation that precisely estimates the magnitude but assigns it the wrong sign leads to an incorrect ship/no-ship decision.

Classical distributional distance metrics such as the Kullback–Leibler (KL) divergence and the Wasserstein distance measure how closely the predicted effect-size distribution matches the ground truth in its entirety. While appropriate when the full distribution is of interest, they conflate two fundamentally different types of error: directional error (predicting the wrong sign) and magnitude error (predicting the correct sign but with inaccurate magnitude). As illustrated in Figure 3, a prediction that correctly identifies a positive effect but overestimates its magnitude receives good scores under the sign metrics but poor ones under KL and Wasserstein distances (the lower the better), despite being operationally correct.

*Figure 3: Sign alignment vs magnitude alignment.*

That said, magnitude metrics remain valuable in other settings where precise effect-size estimation is critical—such as cost-benefit analysis—where knowing how much matters as much as knowing the direction.

## Appendix D Multi-model Validation Extended

To assess whether our findings generalize beyond a single LLM, we replicate the question design comparison (Table 1) with two additional models: Claude Haiku (Anthropic, 2025a) and Claude Opus (Anthropic, 2025b) (Table 6). All other settings remain identical (deep persona pool, temperature 0, pairwise randomization).

| Model | Question Design | Acc | SignBC | SignOv |

$\pm$ $\pm$ $\pm$| Haiku | Independent binary | 0.550.08 | 0.620.06 | 0.590.06 |

$\pm$ $\pm$ $\pm$| Pairwise binary | 0.750.07 | 0.650.06 | 0.640.06 |

$\pm$ $\pm$ $\pm$| Independent rating | 0.650.08 | 0.660.06 | 0.640.06 |

$\pm$ $\pm$ $\pm$| Pairwise rating | 0.700.07 | 0.650.06 | 0.640.06 |

$\pm$ $\pm$ $\pm$| Sonnet | Independent binary | 0.400.08 | 0.590.06 | 0.540.06 |

$\pm$ $\pm$ $\pm$| Pairwise binary | 0.620.07 | 0.620.06 | 0.610.06 |

$\pm$ $\pm$ $\pm$| Independent rating | 0.430.08 | 0.520.06 | 0.490.06 |

$\pm$ $\pm$ $\pm$| Pairwise rating | 0.780.07 | 0.710.06 | 0.700.06 |

$\pm$ $\pm$ $\pm$| Opus | Independent binary | 0.550.08 | 0.560.06 | 0.520.06 |

$\pm$ $\pm$ $\pm$| Pairwise binary | 0.750.07 | 0.730.05 | 0.720.05 |

$\pm$ $\pm$ $\pm$| Independent rating | 0.500.08 | 0.500.07 | 0.470.07 |

$\pm$ $\pm$ $\pm$| Pairwise rating | 0.600.08 | 0.640.06 | 0.630.06 |

*Table 6: Robustness across models with different capabilities to various question designs (combined benchmark, $n=40$). Values are mean $\pm$ SE. Statistical significance via one-sided paired $t$-tests ($\alpha=0.05$). *

Across all three models, pairwise settings consistently outperform independent settings. On the combined benchmark, the difference between pairwise rating and pairwise binary is not statistically significant for any model, though pairwise rating achieves the highest combined accuracy for Sonnet 4.5. The pairwise binary format performs particularly well on Haiku and Opus, driven by strong subscription test performance. This consistency confirms that the advantage of pairwise presentation—forcing comparative evaluation rather than uncalibrated absolute scoring—is a robust finding independent of the specific model used. The pairwise rating format additionally provides more interpretable, granular responses, justifying its use as the default configuration.

## Appendix E Open Datasets and Personas

To demonstrate reproducibility of our results, we evaluate our framework on three public datasets: Book Crossing (Ziegler et al., 2005), Jester Jokes (Goldberg et al., 2001), and MovieLens (Harper and Konstan, 2015). For each dataset, we sample 100 items and construct all $\binom{100}{2}=4{,}950$ pairwise comparisons, using the difference in mean user ratings as the ground-truth effect size. We evaluate two persona configurations: the deep persona pool from our main experiments and a public e-commerce persona set based on open shopping profiles (Berke et al., 2024).

The results establish a concrete path for reproducibility: researchers can construct both the persona pool and the evaluation benchmark from permissive public datasets, enabling direct extension and comparison without access to proprietary experimentation infrastructure.

#### Dataset Preparation.

We sample 100 items from each dataset:

-

Book Crossing: 100 books, rated on a 1-10 integer scale

-

Jester Jokes: 100 jokes, rated on a -10 to 10 scale (0.5 increments)

-

MovieLens: 100 movies with movie IDs, rated on a 0.5-5.0 scale (0.5 increments)

Each item has ground truth statistics computed from real user ratings: mean rating ($\mu_{GT}$) and standard error of the mean ($\sigma_{GT}$).

#### Pairwise Comparison Framework

For each pair of items $(A,B)$ within a dataset, we construct an A/B test scenario:

-

Ground truth effect distribution: The difference in mean ratings, $\Delta_{GT}=\mu_{A}-\mu_{B}$, with uncertainty $\sigma_{\Delta,GT}=\sqrt{\sigma_{A}^{2}+\sigma_{B}^{2}}$

-

Predicted effect: We query each persona to rate each item, then considering the items $(A,B)$, we compute the difference in ratings per persona, then aggregate to obtain the empirical mean $\Delta_{pred}$ and standard error $\sigma_{\Delta,pred}$.

This yields $\binom{100}{2}=4,950$ pairwise comparisons per dataset.

#### Persona Configurations.

We evaluate two persona configurations:

-

Deep personas: The same deep pool of personas studied in the main body of the paper

-

E-commerce personas: public persona set based on online shopping profiles.

#### Evaluation Metrics.

For each pairwise comparison, we compute the probability that the treatment is better than reference, from both ground truth and simulation data: $p_{GT}=P(\Delta_{GT}>0)$ and $p_{pred}=P(\Delta_{pred}>0)$, assuming Gaussian distributions.

#### Results.

*Figure 4: Accuracy of A/B test prediction vs absolute effect size, comparing our deep pool of personas against public open e-commerce personas. The trends look similar, hence supporting reproducibility and future work without access to proprietary data.*

Figure 4 shows the Accuracy metric as a function of effect size (absolute difference in mean ratings) across the three public datasets, evaluated with both persona configurations. Each point represents the average accuracy within effect size bins.

The results demonstrate that persona-based A/B test simulation can be evaluated on public benchmarks. Both persona configurations exhibit consistent behavior: accuracy increases monotonically with effect size across all three datasets, which means larger rating differences lead to more reliable predictions. This provides a sanity check for the simulation framework and aligns with the intuition that distinguishing between significantly different items is easier than comparing similar ones.

Importantly, this evaluation framework enables researchers to benchmark A/B test simulation approaches, reproduce experiments from our paper, and develop further research in this direction, using publicly available datasets and personas.

## Appendix F Full Results Tables

Tables 7–11 report the complete results for the main experiments, including the SignBC metric omitted from the main text due to space constraints, and combined results over the full benchmark ($n{=}40$). All values are mean $\pm$ SE; combined mean = average of CTR and Subscriptions means; combined SE = $\frac{1}{2}\sqrt{\text{SE}_{\text{CTR}}^{2}+\text{SE}_{\text{Sub}}^{2}}$.

$n{=}20$ $n{=}20$ $n{=}40$| Question Design | CTR () | Subscriptions () | Combined () |

| Acc | SignBC | SignOv | Acc | SignBC | SignOv | Acc | SignBC | SignOv |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Independent binary | 0.40.11 | 0.53.09 | 0.51.09 | 0.40.11 | 0.65.08 | 0.56.07 | 0.40.08 | 0.59.06 | 0.54.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Pairwise binary | 0.45.11 | 0.51.09 | 0.50.09 | 0.80.08 | 0.73.07 | 0.73.07 | 0.62.07 | 0.62.06 | 0.61.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Independent rating | 0.40.11 | 0.54.10 | 0.52.10 | 0.45.11 | 0.50.08 | 0.47.08 | 0.43.08 | 0.52.06 | 0.49.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Pairwise rating | 0.75.10 | 0.69.09 | 0.68.09 | 0.80.09 | 0.74.07 | 0.72.07 | 0.78.07 | 0.71.06 | 0.70.06 |

*Table 7: Survey design impact on simulation accuracy (mean $\pm$ SE). Full version of Table 1.*

$n{=}20$ $n{=}20$ $n{=}40$| Persona Source | CTR () | Subscriptions () | Combined () |

| Acc | SignBC | SignOv | Acc | SignBC | SignOv | Acc | SignBC | SignOv |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Survey data | 0.60.11 | 0.59.10 | 0.58.10 | 0.75.10 | 0.71.07 | 0.68.07 | 0.68.07 | 0.65.06 | 0.63.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Rotten Tomatoes | 0.65.11 | 0.64.09 | 0.63.09 | 0.60.11 | 0.59.08 | 0.57.08 | 0.62.08 | 0.61.06 | 0.60.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Open e-commerce | 0.70.11 | 0.68.10 | 0.67.09 | 0.90.07 | 0.77.06 | 0.77.06 | 0.80.07 | 0.73.06 | 0.72.05 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Platform data (ours) | 0.75.10 | 0.69.09 | 0.68.09 | 0.80.09 | 0.74.07 | 0.72.07 | 0.78.07 | 0.71.06 | 0.70.06 |

*Table 8: Simulation accuracy across persona sources (mean $\pm$ SE). Full version of Table 2.*

$n{=}20$ $n{=}20$ $n{=}40$| Persona Pool | CTR () | Subscriptions () | Combined () |

| Acc | SignOv | SignBC | Acc | SignOv | SignBC | Acc | SignOv | SignBC |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Deep | 0.70.10 | 0.62.09 | 0.63.09 | 0.70.09 | 0.64.07 | 0.67.07 | 0.71.07 | 0.63.06 | 0.64.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Representative | 0.55.11 | 0.63.09 | 0.61.10 | 0.70.09 | 0.65.07 | 0.67.07 | 0.63.07 | 0.64.06 | 0.64.06 |

*Table 9: Simulation accuracy across persona pools (mean $\pm$ SE). Full version of Table 3.*

$n{=}20$ $n{=}20$ $n{=}40$| | CTR () | Subscriptions () | Combined () |

| Algorithm | Acc | SignOv | SignBC | Acc | SignOv | SignBC | Acc | SignOv | SignBC |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Random | 0.79.003 | 0.74.003 | 0.75.003 | 0.91.002 | 0.78.003 | 0.79.003 | 0.85.002 | 0.76.002 | 0.77.002 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Kernel Herding | 0.78.001 | 0.74.001 | 0.75.001 | 0.91.001 | 0.78.001 | 0.80.001 | 0.84.001 | 0.76.001 | 0.78.001 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Greedy Farthest | 0.80.002 | 0.74.002 | 0.75.002 | 0.90.001 | 0.77.002 | 0.80.002 | 0.85.001 | 0.76.001 | 0.78.001 |

| All (935) | 0.80 | 0.74 | 0.75 | 0.90 | 0.80 | 0.81 | 0.85 | 0.77 | 0.78 |

*Table 10: Subsampling performance at 500 personas (mean $\pm$ SE across 1000 trials). Full version of Table 4.*

$n{=}20$ $n{=}20$ $n{=}40$| Configuration | CTR () | Subscriptions () | Combined () |

| Acc | SignBC | SignOv | Acc | SignBC | SignOv | Acc | SignBC | SignOv |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Pairwise rating (full) | 0.70.11 | 0.64.09 | 0.64.09 | 0.80.09 | 0.72.07 | 0.71.07 | 0.75.07 | 0.68.06 | 0.68.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Demographics only (full) | 0.65.11 | 0.63.09 | 0.57.08 | 0.30.11 | 0.51.07 | 0.39.07 | 0.47.08 | 0.57.06 | 0.48.05 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| No reasoning (full) | 0.60.11 | 0.61.09 | 0.60.09 | 0.80.09 | 0.76.06 | 0.73.06 | 0.70.07 | 0.69.05 | 0.67.05 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Generic shopper (single) | 0.40.11 | 0.44.10 | 0.44.10 | 0.60.11 | 0.65.08 | 0.64.08 | 0.50.08 | 0.55.06 | 0.54.06 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| No persona (single) | 0.45.11 | 0.49.10 | 0.49.10 | 0.65.11 | 0.65.08 | 0.63.08 | 0.55.08 | 0.57.06 | 0.56.06 |

*Table 11: Component contributions to simulation accuracy (mean $\pm$ SE). Full version of Table 5. Full pool baselines use 935 personas (full); single-call baselines use one fixed prompt (single).*

## Appendix G Comparison with SimAB Benchmark

We investigated the feasibility of benchmarking our framework against SimAB Rieder et al. (2026). SimAB reports 67% overall accuracy (83% for high-confidence cases) on a 47-test corpus, of which the Wikimedia Foundation fundraising tests (2010–2011) constitute the only publicly reproducible subset. These tests involve banner designs and landing page variants for Wikipedia donation campaigns, with ground truth derived from documented conversion metrics (donations/impression, amount/impression, click-through rate).

#### Reproducibility barriers.

SimAB sources its experimental data from the Wikimedia 2010 Banner Testing and 2011 Test Updates pages. However, the authors do not specify which subset of tests from these sources were used, nor do they publish a code repository or processed datasets. As a result, there is no reliable way to identify the exact experimental conditions under which their reported numbers were obtained, precluding a direct numerical comparison. Beyond reproducibility, several fundamental methodological differences complicate a fair comparison, summarised in Table 12.

| Dimension | Our Framework | SimAB |

| Persona source | Data-driven (behavioural) | Synthetic (from screenshots) |

| Question format | Pairwise rating (1–10) | Binary preference (C/Ch/None) |

| Bias mitigation | Random order per persona | Deterministic counterbalancing |

| Population size | Fixed (100 personas) | Variable (30–200, early stopping) |

$\to$ $\to$| Aggregation | Mean effect + SE Normal dist. | Majority vote significance test |

| Primary metric | Directional accuracy + SignBC | Accuracy + F1 |

| Stimuli | Visual screenshots + textual question framing | Visual screenshots |

| Ground truth source | Experimental ground-truth labels | Documented campaign outcomes |

*Table 12: Key methodological differences between our framework and SimAB.*

## Appendix H Population Subsampling: Extended Results

We compute dense vector embeddings for each persona using Qwen2.5-32B-Instruct (Team, 2024), which captures semantic characteristics of persona profiles. The subsampling algorithms then operate on these embeddings:

-

Random: Uniform random sampling without replacement

-

Greedy Farthest: Iteratively selects personas maximizing minimum distance to already selected personas, ensuring geometric diversity by spreading selected points across the embedding space. We randomize the first point to enable variation across trials; subsequent selections are deterministic given the first point.

-

Kernel Herding: Sequentially selects personas to minimize Maximum Mean Discrepancy (MMD) between the subsample and the full distribution (Chen et al., 2012). While Kernel Herding is inherently deterministic, we implement a variant that randomizes the first point to generate diverse trials; all subsequent selections are deterministic.

*Figure 5: Marketing alignment metrics vs sample size for the different sampling algorithms.*

*Figure 6: Subscription alignment metrics vs sample size for the different sampling algorithms.*

Figures 5 and 6 show the evolution of the sign metrics with the size of the sample, comparing the performance of the three algorithms. Each point in the figures corresponds to the average over 1000 independent trials, with error bars representing standard error (SE). For CTR tests, Figure 5 shows that even a small sample of size 100 can recover the performance of the full pool of personas, with Greedy Farthest slightly outperforming Random sampling and Kernel-Herding. For subscription tests, Figure 6 suggests that Kernel-herding has the best performance among the three algorithms, but the performance gap is again small.

For both use-cases, even random sampling gives strong results. A main advantage of Greedy Farthest or kernel-herding is their small variance compared to random sampling.

Note that in some cases, the sampled personas outperform the baseline of using the entire pool. This is because the full persona set may produce overly confident predictions that, while directionally correct, exhibit excessive certainty compared to the true distribution’s inherent uncertainty. This overconfidence can harm uncertainty-aware metrics such as SignOv and SignBC. Subsampling introduces natural variance that can yield closer distributions to the ground truth with appropriate uncertainty levels. Furthermore, when the full set’s prediction is directionally incorrect, the added uncertainty from fewer personas can actually improve alignment with ground truth.

## Appendix I Implementation Details

The prompts and templates in the sections below are slightly simplified for clarity; the final versions used in the experiments include additional formatting instructions and edge-case handling.

### I.1 Model Configuration

All experiments use Claude Sonnet 4.5 (Anthropic, 2025c) accessed via cloud-based batch inference with the following parameters:

-

Temperature: 0 (for reproducibility)

-

Max tokens: 3,000

-

Response format: JSON (enforced via assistant prefill <JSON> {)

### I.2 Persona Generation

Personas are generated through two sequential LLM flows:

-

Demographics flow: Infers age, gender, income, education, household composition, and other attributes from behavioral patterns using structured output schemas with constrained category options.

-

Shopping characteristics flow: Synthesizes narrative descriptions of brand affinity, price sensitivity, quality expectations, and shopping style from the same behavioral signals.

Both flows use Claude Sonnet 4.5 with structured JSON output schemas to ensure consistent persona formats across the pool.

### I.3 Persona Template

We construct each persona by processing anonymized, aggregated behavioral patterns through an LLM, which synthesizes these signals into a structured profile. The profile captures multiple complementary dimensions:

-

Inferred demographics: attributes such as age, gender, income, and household composition, derived from behavioral patterns rather than self-reported.

-

Behavioral statistics: quantitative summaries of activity intensity, spending patterns, and category preferences.

-

Engagement patterns: conversion funnels and temporal regularities that characterize how the user interacts with the platform.

-

LLM-synthesized narratives: free-text interest profiles and behavioral insight summaries that provide the model with a coherent characterization to role-play during simulation.

During simulation, the full persona profile is rendered into a structured prompt that instructs the LLM to embody the corresponding user—maintaining first-person perspective, using language appropriate to their background, and making decisions consistent with their established behavior patterns. Fields are populated from the persona generation pipeline.

## ROLE INSTRUCTION You are now embodying the persona of a specific online retail customer. Respond as if you were this person, with their demographic background, interests, and behavioral patterns. Maintain first-person perspective throughout. ## CUSTOMER PROFILE ### DEMOGRAPHIC BACKGROUND - Age: {age} - Gender: {gender} - Education: {education} - Income Level: {income} - Home Status: {home_ownership} - Family: {parental_status} - Location: {urbanicity} ### INTERESTS AND PREFERENCES {interests} ### SHOPPING BEHAVIOR PROFILE - Brand Perspective: {brand_perception} - Price Sensitivity: {price_perception} - Quality Expectations: {quality_perception} - Use of Reviews: {reviews_perception} - Value Assessment: {value_perception} ### PURCHASE HISTORY (aggregated) - Activity Level: {activity_tier} - Purchase Frequency: {frequency_tier} - Spending Tier: {spending_tier} - Price Range: {price_range} ### ENGAGEMENT METRICS - Search to View Rate: {search_to_view} - View to Purchase Rate: {view_to_purchase} ## BEHAVIORAL INSIGHTS {behavioral_reasoning} ## EMBODIMENT GUIDELINES 1. Use language appropriate for your background. 2. Incorporate your characteristic approach to evaluating products. 3. Express opinions consistent with your profile.

### I.4 Question Prompt Template (Pairwise Rating)

System: You are an AI assistant role-playing a customer with specific characteristics. You are answering a structured question about visual design variants. User: 1. Background about your character: <background>{persona}</background> 2. Your task: Rate each option on a 1-10 scale. Provide your scores in the "answer" field and a reason in the "reason" field. 3. Response format (valid JSON): {"reason": "explanation grounded in persona", "answer": "widget_1:7, widget_2:4"} <question>{question_text}</question> <options> 1: {variant_label_1} [image] 2: {variant_label_2} [image] </options>

The agent receives a system prompt instructing it to role-play as the persona, followed by the structured question with variant screenshots. Images are encoded as base64 images and passed inline within the prompt as image content blocks, allowing the model to process visual stimuli directly alongside the textual question and persona context. For independent rating, each variant is presented in a separate prompt without alternatives. Tested variant order is randomized per persona using the persona identifier as a deterministic seed, ensuring reproducibility while controlling for positional bias. Variant labels use neutral identifiers (widget_{n/m}) to avoid priming effects. Each simulation run processes all personas in parallel via batch jobs, with automatic batching based on estimated prompt size.

### I.5 Score Extraction

Scores are extracted from the JSON response’s answer field by parsing the widget_name:score pairs. For each A/B test, we collect per-persona $(s_{ref},s_{treat})$ tuples and compute the predicted effect as $\hat{\delta}_{s}=\frac{1}{N}\sum_{i=1}^{N}\frac{s_{treat}^{(i)}-s_{ref}^{(i)}}{s_{ref}^{(i)}}$.

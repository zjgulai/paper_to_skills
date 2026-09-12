<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2608.22152
     paper_id : p2s-2026-0017
     source   : https://arxiv.org/pdf/2608.22152
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

The Collaboration Tax:
How Much LLM Multi-Agent Systems Pay to Coordinate Weixiang Sun1 Zehong Wang1,† Hong Huang2,3 Colby Nelson1 Yijun Ma1 Yanfang Ye1,† 1 University of Notre Dame 2 Meta Superintelligence Labs 3 Simon Fraser University † Corresponding Authors <wsun4,zwang43,yye7>@nd.edu § Code  Website Abstract

arXiv:2608.22152v1 [cs.CL] 23 Aug 2026

Multi-agent systems built from large language models are deployed widely, yet how much performance is lost when two LLMs must coordinate rather than act alone remains unclear. We formulate the collaboration tax as the team-decentralisation loss of a two-player cooperative game with private information, with two propositions characterising its sign and its equivalence to a max-superadditivity violation. We operationalise this definition on 32 solo-tractable tasks grouped by source of grounding friction and measure it on 11 models from 7 providers. The tax is structured along two no-exception axes: a category ordering across every model and a monotonic decrease with capability. The proximate mechanism is not a reasoning deficit but a four-stage conversational cascade in which agents make ungrounded claims, fail to query the partner, skip integrating both views, and accept the answer without re-derivation. The tax is mechanically predictable from conversation features and partly tractable: a prompt intervention targeting all four stages closes a substantial fraction of the gap, with the dominant bottleneck differing across categories. In heterogeneous pairs the tax is pulled toward the stronger partner rather than the additive midpoint, empirically realising the max-superadditivity violation predicted by our framework. Together these results recast collaboration in LLM systems as a measurable, predictable, and partly tractable cost.

1

Introduction

Multi-agent systems built from large language models have become a fast-growing line of work (Guo et al., 2024; Ye et al., 2025) and now drive real applications across software engineering, planning, social simulation, and inference-time debate (Qian et al., 2024; Hong et al., 2024; Wu et al., 2024; Li et al., 2023; Yuan et al., 2024; Park et al., 2023;
Huang et al., 2024; Sun et al., 2026; Du et al.,

Real-world intuition Software Issue repo, source files

Private Code

Private tests, logs, requirements…

Code Agent

Test Agent Define test

Write/edit code

Expected: better outcome

What actually happens View 1

View 2

Agent A

Agent B

(Private)

(Private)

Grounding Querying

Outcome Solo Collab

Integration Re-derivation

Collab Tax

Figure 1: The collaboration tax. Two LLM agents with private views on a shared problem (top) are expected to outperform either alone; in practice, paired execution underperforms the solo upper bound by a collaboration tax that decomposes into four conversational stages:
grounding, querying, integration, and re-derivation (bottom).

2024; Wang et al., 2026). These deployments treat collaboration as a free primitive: assemble enough capable models, give them clear roles, and the team will outperform any single member. The premise is rarely tested directly. As the unit of deployment shifts from a single LLM to a collection of communicating ones, a basic empirical question shifts with it: when two or more LLMs must coordinate to solve a problem they could each handle alone, do they actually succeed together, and if not, why?
A small set of recent benchmarks has begun to probe this question directly (Davidson et al., 2025;
Eisenstein et al., 2026; Sun et al., 2025; Liu et al.,

2024; Yadav et al., 2026; Cemri et al., 2026), but each measurement is local to one task domain or one pair configuration. The literature still lacks a structured, model- and task-agnostic account of coordination cost itself, with a known sign and a decomposition across task types and pair configurations (Section 2 discusses each line in detail).
Three open questions structure our investigation.
First, is there a systematic coordination cost when LLMs are paired, and what governs its size? The question asks whether the collaboration gap is a structural property of paired LLM execution or an artefact of one task type, and if structural, whether its size is driven primarily by model capability, by task structure, or by their interaction.
Second, when failure occurs, where in the conversation does it originate? The breakdown could lie in reasoning (the model cannot compute the answer), in information access (the model lacks the relevant facts), in inter-agent communication (the facts exist but go unshared), or in verification (the receiver accepts a wrong answer); each candidate location implies a different intervention.
Third, can the cost be reduced without retraining? This question separates structural conversational failures, which prompt-level changes can plausibly target, from intrinsic capability failures, which they cannot. We address all three.
To address these questions we formulate the collaboration tax as a task-agnostic quantity anchored in the team-decentralisation framework of Marschak (1955) and the cooperative game theory of Shapley et al. (1953), with two Propositions that characterise its sign and tie its positivity to a structural failure of the underlying cooperative game (Section 3). We operationalise this definition on a 32-task suite grouped into three families chosen to span three distinct sources of grounding friction in the sense of Clark and Brennan (1991);
Brennan and Clark (1996); Pickering and Garrod (2004): Spatial (reference-frame alignment), Relational (lexical and category alignment), and CSP (indexing and ordinal alignment). Every task is solo-tractable by design, so the gap reflects coordination rather than problem-solving capacity. We evaluate 11 models from 7 providers in solo, homogeneous, and heterogeneous configurations.
We find that the collaboration tax is not a single number but a structured, mechanistic phenomenon.
The cost has two no-exception orderings, one over task categories and one over model capability, and is mechanically predictable from conversation-

shape features alone, with capability setting the intercept of the gap and conversation shape its slope.
The proximate failure mechanism is a four-stage cascade, not a reasoning failure: from grounding, where agents fabricate facts not stated by either view, to querying, where they fail to ask the partner for the facts they do not have, to integration, where they skip combining the two views before committing, and finally to re-derivation, where the receiver accepts the answer without recomputing it. The mechanism extends to heterogeneous pairs:
the cost is asymmetrically borne by the stronger partner, while the cascade dimensions continue to discriminate failure from success.
Building on this cascade insight, we design a prompt-level combined intervention that appends one stage-targeted clause for each of the four stages to the system prompt. The combined intervention closes a substantial fraction of the tax; per-stage leave-one-out ablations show that the dominant bottleneck differs across categories. The simplicity of the fix is exactly the point: failures attributed to reasoning or capability cannot be patched this cheaply, but failures of grounding, querying, integration, and re-derivation can.
Our contributions are as follows:
• A formal definition of the collaboration tax as the success-rate gap between fullinformation solo and split-view paired execution, with a cooperative-game interpretation that isolates coordination cost from problemsolving capacity.
• A structurally categorised task suite of 32 solo-tractable tasks across Spatial, Relational, and CSP families, evaluated on 11 models from 7 families in solo, homogeneous, and slot-swapped heterogeneous configurations.
• Mechanistic insights into the collaboration tax: a 2D landscape over capability and category, a four-stage conversational cascade (grounding, querying, integration, rederivation) rather than a reasoning failure, and a regression decomposition in which capability sets the intercept and conversation shape sets the slope.
• Stage-targeted prompt interventions that partially close the gap, with a leave-one-out ablation showing that the dominant bottleneck layer differs across categories.

2

Related Work

LLM-based agent collaboration. The closest prior work is Davidson et al. (2025), who introduce the collaboration gap on a split-view maze and show that a strong-primer relay recovers most of it. Related benchmarks probe coordination from other angles: turn-count effects (Eisenstein et al., 2026), Overcooked-style cooperation (Sun et al., 2025), single-agent agentic tasks (Liu et al., 2024), planner-with-rule-based-partner pairs (Zhang et al., 2024), zero-cost cooperation games (Yadav et al., 2026), and failure-mode taxonomies (Cemri et al., 2026). A parallel engineering line orchestrates LLMs as cooperating agents (Wu et al., 2024; Hong et al., 2024; Li et al., 2023; Qian et al., 2024) and uses debate to improve reasoning (Du et al., 2024);
theory-of-mind probes (Kosinski, 2024; Strachan et al., 2024) and cooperative reinforcement learning (Zhou et al., 2025) examine related capacities without a structural account of where coordination fails. The classical theory of grounding (Clark and Brennan, 1991; Clark, 1996; Levinson, 2003) motivates the multiple equivalent representations our tasks exploit as natural friction. Each prior thread uses either a single task type or a homogeneous pair; our 32-task suite across three structural families and a heterogeneous-pair matrix supplies the structured diagnostic these threads lack.
Cooperative games with asymmetric information. Our split-view setup inherits a longer tradition of two-player asymmetric-information cooperation. Bard et al. (2020) formalise Hanabi as a benchmark for partial-observability cooperation, and recent neural agents (V Sudhakar et al., 2025)
pursue zero-shot coordination across unseen partners. Overcooked-AI (Carroll et al., 2019) adds embodied real-time cooperation between learned policies and human surrogates, and textual referent games such as OneCommon (Udagawa and Aizawa, 2019) test grounding under privately-held continuous context. In dialogue, Li et al. (2025) annotate misunderstandings in MapTask transcripts to study how grounding breaks down in human dyads.
Across all of these, the underlying problem is nontrivial even with full information, so coordination cost is entangled with problem-solving difficulty, and most use binary success metrics that wash out the continuous signal needed to distinguish nearmisses from catastrophic failures. Our tasks are instead deliberately solo-trivial and graded on a continuous [0, 1] scale, so the gap between solo

and collaborative performance can be attributed to coordination rather than to a model’s underlying ability to solve the task, and partial progress is preserved as a graded outcome.

3

The Collaboration Tax

To compare coordination cost across tasks, models, and pair configurations, and to know what the resulting quantity means when it is zero, positive, or transferred between dyads, we need a task-agnostic definition with a known sign and a structural interpretation rather than a number reported in isolation on a single task. Davidson et al. (2025) make the cost explicit on a split-view maze by comparing one LLM given the full instance to two copies coordinating over complementary halves; we generalise this construction to any task admitting a unionnecessary partition and a continuous deterministic grader, and anchor the resulting quantity as the team-decentralisation loss of Marschak (1955) for a two-player cooperative game with private information (Shapley et al., 1953). Two Propositions, stated and proved in Appendix A, pin down its structure: information dominance guarantees that the collaboration tax is non-negative whenever the paired protocol’s policy is dominated by the optimal centralised policy, explaining why it is positive on the vast majority of cells we measure (Section 5.2 reports the small set of empirical negative cases); and max-superadditivity equivalence states that a positive collaboration tax is exactly the failure of the underlying cooperative game to satisfy v({1, 2}) ≥ max(v({1}), v({2})), supplying the cooperative-game vocabulary for the asymmetric pair effects of Section 6.
Operational form. For a task T with instances x scored by a deterministic grader U ∈ [0, 1], and a union-necessary partition x = v1 (x) ∪ v2 (x) such that neither view alone determines the answer, the homogeneous tax of model M is c tax(M, T ) = ssolo-full (M, T ) − shomo (M, T ), (1)
where ssolo-full is the mean score of M given the merged instance and shomo is the mean score of two copies of M given v1 and v2 exchanging messages until termination, each averaged over 50 rollouts with independent seeds. We additionally report the c solo-full to normalise across tasks ratio version tax/s with different solo ceilings. The heterogeneous case substitutes the stronger member’s solo score for the first term.

4

Tasks and Protocols

4.1

Design Principles for the Task Suite

For Equation 1 to cleanly measure coordination cost, every task in the suite satisfies four properties, three of which we inherit from Davidson et al.
(2025). Solo-trivial: with the full instance a single agent should solve the task at a high rate, so that ssolo-full is near ceiling and the gap reflects coordination cost rather than problem-solving capacity.
Union-necessary: each instance is partitioned into views v1 , v2 with v1 ∪ v2 = x and neither view alone admits the canonical answer. Algorithmically verifiable: ground truth is computed by a deterministic procedure (BFS for shortest paths, topological sort for orderings, a SAT-style solver for constraint puzzles) yielding a continuous score in [0, 1], with path-task graders enumerating all sixteen origin / orientation / axis-order schemes following Davidson et al. (2025). Multiply expressible: the same content admits several equivalent surface representations (coordinate origins, axis orderings, naming conventions, ordinal directions, relational vocabularies), providing the grounding friction we aim to measure (Clark and Brennan, 1991; Levinson, 2003). Each principle is required by Definition 1 or Proposition 1: without it the tax either degenerates to zero by construction, becomes ill-defined as a population quantity, or mixes coordination cost with solo problem-solving capacity.
Appendix A.6 formalises each requirement.
4.2

Task Families

The suite is organised into three families, chosen for three reasons. First, they span the principal answer structures that LLMs are asked to produce in multi-agent deployments: sequences, relational queries, and constraint-satisfying assignments. Second, they cover three distinct sources of grounding friction in the sense of Section 4.1: reference-frame alignment (Spatial), lexical and category alignment in the sense of Brennan and Clark (1996); Pickering and Garrod (2004) (Relational), and indexing and ordinal alignment (CSP). Third, they expose different error-propagation patterns: a single misaligned step invalidates the rest of a path (Wang et al., 2026), relational query errors stay local to their query, and constraint violations cascade through the assignment. Together they let us ask whether collaboration ability is a single capacity or a profile that varies with task structure, and whether per-family bottlenecks differ. The full per-task an-

swer types, grader rules, and coverage within each family are reported in Appendix F.
4.3

Solo and Collaborative Modes

We evaluate each task instance under four canonical modes. The first two are single-agent controls;
the latter two are two-agent collaborations. In every mode the final answer is scored against the deterministic ground truth (pipeline described under Metrics below).
Solo control. In solo-full a single agent receives the merged instance v1 ∪ v2 and produces a candidate answer in one completion. The candidate is then returned to the same model for a critic pass that may revise it before grading. The critic step pushes the tax toward pure coordination cost rather than a turn-count asymmetry between solo and collaboration: the collaboration mode receives multiple rounds of partner messages, each effectively acting as another pass over the answer, so giving solo a single revision opportunity matches that structural affordance. Empirically the critic contributes a small fraction of the solo score (Appendix H), well below the gap to collaboration on the same model. Solo-full is the coordination-free upper bound and serves as the denominator of the gap.
Homogeneous collaboration. Two instances of the same model, instantiated with independent contexts, receive v1 and v2 respectively. They exchange messages under a shared system prompt that explains the split-view protocol; each message is prefixed with [other agent]: when delivered to its counterpart. The dialogue continues until either agent emits the termination marker ACTI!
together with a candidate answer, or a turn cap of 50 exchanges is reached. The candidate is then passed to the grader.
Heterogeneous collaboration. Identical to homogeneous, except that the two agent roles are filled by different models (or by different checkpoints of the same family). This mode is the basis of the pair-composition analysis in Section 6.
Metrics. For each (task, mode, model or pair) we run 50 rollouts with independent seeds and report the mean continuous score s ∈ [0, 1]. The LLM never assigns a score itself; it only translates freeform text into the structured answer that the deterministic program then scores (per-task grader rules in Appendix F). The collaboration tax of Equation 1

homo failure 1.00

rate

1.00 0.75

homo success

1.00

0.98

0.95

0.85 0.59

0.50

0.61 0.49

0.48

0.56 0.44

0.57

0.51

0.50

0.90

0.82

0.50

0.39 0.29

0.32

0.28

Spatial

Relational

0.25 0.00

Spatial

Relational

CSP

L1 grounded every claim traceable to source

Spatial

Relational

CSP

Spatial

L2 queried asked partner for specific facts

0.85

0.81

Relational

CSP

L3 integrated answer combines both views

0.30

CSP

L4 re-derived re-verified before committing

Figure 2: Four-stage cascade by outcome and category. Each panel reports the rate at which homogeneous failures (lighter bars) and successes (saturated bars) satisfy one of the four cascade dimensions, separated by task category. L1 grounded: every claim is traceable to a stated source. L2 queried: at least one agent asked the partner a specific factual question. L3 integrated: the decisive claim was preceded by an explicit combined-state message.
L4 re-derived: the receiving agent showed recomputation work before either agent emitted ACTI!. Every stage discriminates failures from successes.

5 5.1

Experiment Experimental Setup

We evaluate eleven models from seven providers:
OpenAI (gpt-5, gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini), Anthropic (claude-sonnet-4-5), Google (gemini-2.5-flash-lite), DeepSeek (DeepSeek-V4-Pro), and three open-weight models hosted through API endpoints:
Llama-4-Maverick (Meta, mixture-of-experts), Phi-4 (Microsoft), and Qwen3-8B (Alibaba). The set spans a wide capability range while limiting redundancy within any single family. All models are queried through OpenAI-compatible chat APIs with no fine-tuning, no parameter access, and no sampling control beyond temperature. Detailed hyperparameters (temperatures, rollout count, turn cap) are reported in Appendix I.
5.2

The Gap Landscape

The collaboration tax is structured by both model capability and task type. As shown in Figure 3, two patterns hold without exception across the eleven models. Within every row, the ordering is Spatial ≻ Relational ≻ CSP: spatialcoordination tasks lose the most from collaboration, relational queries lose less, and constraintsatisfaction tasks lose least. Across rows, the gap scales monotonically with model capability: the weakest models lose roughly half of their solo success to coordination, while the top-tier models are

+12

+6

+2

+23

+3

-6

+19

+8

-5

+45

+16

+8

+53

+19

+13

+15

+44

+25

+24

+0

+57

+27

+16

+50

+35

+23

-FL

+59

+31

+21

B

+58

+34

+34

o

+63

+50

+38

Spatial

Relational

CSP

t-5 gp 5 -4.
et

nn

So

-V

DS

ro 4-P

av -M a-4 m ni Lla mi .14 t ini gp o-m 4 t no gp na -5t gp i-4 Ph 2.5

ini

m

Ge

3-8

n we

Q

.1

n -na

t-4 gp

+65 +45 +30

ratio_gap (%)

and its ratio variant are over the solo-full and homogeneous modes; the heterogeneous mode enters the analysis in Section 6.

-10

Figure 3: Homogeneous ratio gap by model and category. Each cell shows the mean ratio gap (in percentage points) for one (model, category) pair, with rows sorted by each model’s overall mean. The gap is structured by both model capability and task category.

also affected. The three weakest rows come from three different model families, so the capability ordering is not a family-style artefact.
A per-task decomposition (Figure 6) is reported in Appendix H.
5.3

Mechanism: A Four-Stage Cascade

We next ask why the gap arises. From 700 openended LLM failure descriptions clustered under a neutral prompt and an anti-bias naming rule (Appendix B), we extract 16 behaviourally specific themes and group them along the conversationalgrounding pipeline of Clark and Brennan (1991);
Clark (1996): claims (L1 Grounding), queries (L2 Querying), uptake (L3 Integration), and acknowl-

Spatial

Relational

L3 Integration. A pair integrates iff the decisive claim is preceded by an explicit combined-state message that lists facts from both views and any derived consequences. Panel L3 of Figure 2 shows that integration is the strongest single-variable predictor of the collaboration tax and the only stage whose marginal contribution to a multi-feature regression is positive. The effect peaks on Spatial, where path tasks carry persistent state and a single missed integration corrupts all subsequent moves.

CSP

1.0

predicted ratio_gap (OOF)

0.8

OOF R² = +0.475 OOF Spearman ρ = +0.760 OOF Pearson r = +0.705 mean y − mean x = +0.007

0.6

0.4

0.2

0.0 y=x OLS: y = 0.60 x +0.12

−0.2 −0.2

0.0

0.2

0.4

0.6

0.8

1.0

actual ratio_gap gpt-5 Sonnet-4.5 DS-V4-Pro

Llama-4-Mav gpt-4.1-mini gpt-4o-mini

gpt-5-nano Phi-4 Gemini-2.5-FL

Qwen3-8B gpt-4.1-nano

Figure 4: Out-of-fold predicted vs actual ratio gap.
The M1 ridge regression fit on (model, task) cells with folds grouped by task. Out-of-fold R2 = 0.475, Spearman ρ = 0.760, Pearson r = 0.705.

edgment (L4 Re-derivation). Each stage is operationalised as a binary judge dimension applied to every homogeneous rollout. The four stages, from earliest to latest:
L1 Grounding. A claim is grounded if it can be traced to information stated by either agent. Panel L1 of Figure 2 shows that grounding is the cleanest single-feature fail/success discriminator in the judge: successful rollouts are grounded in essentially every category, while a substantial fraction of failures contain at least one ungrounded claim.
It also acts as a strict necessary condition: once an agent fabricates a value not stated by either view, no downstream stage can recover.
L2 Querying. A pair queries iff at least one agent makes a specific factual request of the partner (“what is the value of node K?”). Querying discriminates failure from success across all three categories (panel L2 of Figure 2), with the largest gap on CSP, where successful pairs explicitly elicit cross-half capacity and constraint facts that failed pairs leave latent. Spatial and Relational pairs show smaller but non-zero querying gaps, because much information already arrives passively when agents dump their views. L2 is also the most independent of the four dimensions, the layer most likely to fire alone in a failed rollout.

L4 Re-derivation. A pair re-derives iff the receiving agent shows actual recomputation work (rewalks the path, recomputes the sum, re-checks the constraints) before either agent wants to end. Rederivation tracks the ratio gap almost as strongly as integration in absolute correlation, but its marginal contribution beyond integration is essentially zero:
failures of L3 and L4 co-occur in the majority of failures, since the receiver cannot easily re-derive when the proposer never integrated. Whether this reflects redundancy or observational entanglement is tested by the intervention in Section 5.5.
Cascade structure. The four stages are separable but not independent: in most failed rollouts at least two stages fire simultaneously, and L3 and L4 are entangled in baseline data, with full co-occurrence statistics in Appendix D and one illustrative singlestage failure per dimension in Appendix E. To validate the judge labels we re-annotated 100 stratified homogeneous rollouts with four independent expert annotators under a shared guideline: inter-rater agreement is moderate to substantial across all four stages (Fleiss’ κ in [0.40, 0.69]), and agreement between the automated judge and the expert majority is substantial on L1, L2, and L3 (Cohen’s κ in [0.67, 0.73]) and moderate on L4 (κ = 0.51)
(Appendix C). We test the implied prediction, that stage-targeted prompt interventions produce a separable lift per stage, in Section 5.5.
5.4

Predicting the Gap

If the gap is driven by mechanical conversational behaviours, it should be predictable from those behaviours. We fit a ridge regression (Ridge, λ = 1, inputs standardised) of the per-cell ratio gap on a panel of conversation-structural features (sample size, length, agreement and disagreement marker density, view-disclosure ratios, coordinate-token density, and turn-pacing statistics) together with model and category dummies. To prevent leakage, folds are grouped by task: every held-out fold

Spatial

Relational

CSP

Overall

Condition

∆shomo

% closed

∆shomo

% closed

∆shomo

% closed

∆shomo

% closed

All four no L1 no L2 no L3 no L4

+0.051±.026 +0.066±.027 +0.034±.048 +0.059±.018 −0.004±.027

+27.1±12.3 +37.3±16.2 +12.1±26.0 +30.3±9.9 +2.3±12.5

+0.072±.029 −0.005±.038 +0.030±.030 +0.044±.042 +0.015±.035

+41.4±38.0 −2.4±25.4 +18.4±21.2 +27.5±36.8 +2.8±37.5

+0.111±.074 +0.093±.065 +0.055±.028 +0.119±.078 +0.064±.063

+52.5±32.3 +45.6±35.3 +28.2±20.5 +55.2±22.1 +29.4±25.6

+0.079±.031 +0.051±.033 +0.040±.022 +0.075±.035 +0.027±.030

+38.3±14.1 +27.7±15.1 +18.4±13.4 +36.3±13.5 +9.9±13.4

Table 1: Stage-targeted prompt-intervention ablation. The all four condition adds the L1 grounding, L2 query, L3 integration, and L4 re-derivation clauses to the system prompt; the four no Lk rows are leave-one-out variants that drop the clause for layer k. ∆shomo is the absolute lift in the homogeneous success rate against the no-intervention baseline; % closed is the fraction of the original tax that the intervention recovers. Subscripts give the half-width of the 95% confidence interval. Bold marks the column-wise point-estimate maximum.

contains tasks not seen during fitting.
The collaboration tax is mechanically predictable from conversation features. As shown in Figure 4, the regression achieves a substantial out-of-fold R2 across the (model, task) cells, with strong held-out rank correlation throughout.
Adding the cascade judge labels to the regression does not meaningfully improve the fit beyond integration (L3), indicating that the structural features already encode most of the predictive signal carried by the judge labels and that L4 collinearity with L3 leaves little additional variance to explain.
Capability sets the intercept; conversation shape sets the slope. The regression generalises across tasks: a leave-one-task-out evaluation, in which every task is held out in turn while fitting on the rest, retains positive held-out variance explained and strong rank correlation, with the majority of held-out tasks individually positive. By contrast, the regression cannot extrapolate the absolute gap level to a held-out model: a leave-one-model-out variant preserves the rank ordering of cells but not their absolute level. The coefficients in Appendix Figure 7 show that the largest positive contributors are model dummies for the weakest models, while the largest negative contributors are the CSP dummy and conversation-shape features. We read this as a two-component decomposition of the gap:
a model-level intercept set by base capability and a slope along conversation-shape features shared across models.
5.5

Intervention: Stage-Targeted Prompt Clauses

If the four cascade stages of Section 5.3 are genuine mechanisms rather than correlates of failure, then prompt clauses targeted at each stage should produce a measurable lift in the homogeneous suc-

cess rate, and the relative magnitude across stages should track the per-category fail rates of Section 5.3. We test five conditions on top of the homogeneous baseline of Section 5.1: an all-four condition that appends a stage-specific clause for every layer to the system prompt (a grounding clause for L1, a query mandate for L2, an integration block for L3, and a re-derivation requirement for L4), and four leave-one-out variants no Lk that drop the k-th clause while keeping the other three. The four clauses are listed verbatim in Appendix G.
The combined intervention recovers a substantial fraction of the tax. As shown in Table 1 (last column), the all-four condition lifts homogeneous success against the no-intervention baseline, with the 95% confidence interval bounded well above zero on every category. Per-category responsiveness follows the predictive signal of Section 5.4:
CSP responds most, Relational next, and Spatial least. No condition reaches the solo ceiling, but a single change to the system prompt recovers a substantial fraction of the entire collaboration tax across the suite, with no retraining and no change to the underlying model.
Each category is bottlenecked by a different cascade layer. Reading down a column of leave-oneout values isolates the marginal contribution of the dropped clause. As Table 1 shows, dropping L4 on Spatial, L1 on Relational, or L2 on CSP each substantially reduces the lift in the respective category, with the 95% confidence interval crossing zero for no L4 on Spatial and for no L1 on Relational; the critical layer differs across categories.
The Spatial result is the strongest evidence that L4 is causally separable from L3: in the observational data of Section 5.3 the two co-occur and L4 carries no separable regression weight beyond L3, but once the L3 clause externally enforces integration,

Spatial

Relational

CSP

Pearson r = +0.732 mean y − mean x = -0.170

hetero pair gap: g_AB

0.6

0.4

Δ = -0.17

0.2

0.0

−0.2 y=x OLS: y = 1.35 x -0.27

−0.4 −0.4

−0.2

0.0

0.2

0.4

0.6

mean of two homo gaps

Figure 5: Additive-null scatter for heterogeneous pairs. Each point is one (pair, task) cell: x is the average of the two participating models’ homogeneous gaps; y is the actual hetero gap. The OLS fit (y = 1.35x − 0.27)
and the y = x reference line are shown. Hetero gaps fall below the additive midpoint.

L4 carries the Spatial bottleneck on its own. In two cases, no L1 on Spatial and no L3 on CSP, the targeted leave-one-out point estimate exceeds the all-four point estimate, though the confidence intervals overlap heavily, indicating at most nominal headroom for category-selective prompts over the universal one.

6

Pair

v({1})

v({2})

v({1,2})

ϕ1

ϕ2

Sonnet × nano nano × Sonnet nano × gpt-5 gpt-5 × nano

0.869 0.352 0.352 0.932

0.352 0.869 0.932 0.352

0.731 0.732 0.756 0.706

0.624 0.108 0.088 0.643

0.107 0.624 0.668 0.063

Table 2: Cooperative-game values and Shapley allocations. Characteristic-function values v(·) and Shapley values ϕ for the four heterogeneous pair configurations, averaged across tasks. Every pair violates maxsuperadditivity, and every member’s Shapley value falls below its singleton payoff.

The pair gap is pulled toward the stronger member, not toward the midpoint. Aggregating across (pair, task) cells (Figure 5), the actual hetero ratio gap correlates strongly with the midpoint between the two individual homogeneous gaps but is systematically below it and well below the additive line y = x; the per-cell breakdown by (initiator, responder) is reported in Appendix Figure 9, where the four off-diagonal heterogeneous cells cluster near the strong-tier diagonal rather than averaging between strong and weak. Equivalently, all four configurations violate max-superadditivity (Shapley et al., 1953) on the majority of tasks, and Table 2 translates this into cooperative-game terms:
every member’s Shapley share falls below its singleton payoff, as Appendix A.4 formalises. Samefamily and cross-family pairs are not visibly separated; with one pair per side, family-level effects are not detectable. Two further robustness checks are reported in Appendix H.

Heterogeneous-Pair Matrix 7

We now ask whether the same gap structure holds when the two agents are different models. Running the full 11 × 11 pair matrix is too costly for the question we care about, so we instead pick two strong-weak contrasts that hold the capability gap roughly fixed while varying family identity. Both contrasts use gpt-4.1-nano as the weak slot: one pairs it with claude-sonnet-4-5 (crossfamily, Anthropic and OpenAI), the other with gpt-5 (same-family, both OpenAI). For each pair we run agent-1 / agent-2 in both orderings, giving four heterogeneous configurations on the same task suite. With only two pairs and four configurations, our heterogeneous results are reported as an existence proof of a qualitative pattern rather than a quantitative characterisation; we use them to probe one structural prediction of the cooperative-game framing of Section 3.

Conclusion

We have formulated the collaboration tax as the team-decentralisation loss of Marschak (1955) for two-player cooperative games with private information, and measured it across 32 tasks and 11 models. The tax is structured along two no-exception orderings (capability and task category), driven by a four-stage conversational cascade rather than a reasoning deficit, mechanically predictable from conversation features, partly tractable through a single stage-targeted prompt intervention, and pulled toward the stronger member in heterogeneous pairs.
These results recast LLM coordination cost as a structured, predictable, and partly tractable phenomenon, and they suggest that practical multiagent deployments can recover a substantial fraction of solo capacity through prompt-level changes alone, without retraining or pair-specific tuning.

Limitations Two-agent only. Our suite measures the collaboration tax for dyadic pairs (N = 2). Whether the four-stage cascade and the rank-survives, level-fails decomposition generalise to N ≥ 3 multi-agent settings is unstudied; both the  partition structure (now N private views and N2 pairwise channels) and the conversational dynamics (now a multi-party dialogue with floor management) change qualitatively beyond the dyadic case. We are extending the framework to multi-agent collaboration as ongoing work.
Synthetic tasks rather than deployment workloads. Every task in our suite is a procedurally generated puzzle (grid, graph, constraint satisfaction). This isolates coordination mechanics from domain knowledge and makes the collaboration tax cleanly attributable to grounding friction, but the resulting setting does not directly correspond to the multi-agent applications LLMs are increasingly deployed in: collaborative code editing, issue triage and resolution, multi-turn debugging, or interactive document drafting, where the answer space and the feedback signal are far less structured. Future work will design tasks closer to these production workloads (paired code-modification dialogues, issue-resolution pairs over a shared repository) and re-measure the collaboration tax under those conditions.

Ethics Statement This paper involves no human subjects, no scraping of user data, and no demographic targeting.
All experiments use procedurally generated puzzle instances and commercial LLM APIs with no fine-tuning. Released artifacts (task generators, splitters, graders, prompts, and anonymised conversation transcripts) enable reproduction and followup analysis. The mechanistic insights into where multi-agent LLM coordination breaks down are intended to support more robust collaborative systems; we are not aware of dual-use risks specific to this work beyond those already present in publicly available LLM agent benchmarks.

References Nolan Bard, Jakob N Foerster, Sarath Chandar, Neil Burch, Marc Lanctot, H Francis Song, Emilio Parisotto, Vincent Dumoulin, Subhodeep Moitra, Edward Hughes, and 1 others. 2020. The hanabi chal-

lenge: A new frontier for ai research. Artificial Intelligence, 280:103216.
Susan E Brennan and Herbert H Clark. 1996. Conceptual pacts and lexical choice in conversation. Journal of experimental psychology: Learning, memory, and cognition, 22(6):1482.
Micah Carroll, Rohin Shah, Mark K Ho, Tom Griffiths, Sanjit Seshia, Pieter Abbeel, and Anca Dragan. 2019.
On the utility of learning about humans for human-ai coordination. Advances in neural information processing systems, 32.
Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, and 1 others. 2026. Why do multiagent llm systems fail? Advances in Neural Information Processing Systems, 38.
Herbert H Clark. 1996. Using language. Cambridge university press.
Herbert H Clark and Susan E Brennan. 1991. Grounding in communication.
Tim R Davidson, Adam Fourney, Saleema Amershi, Robert West, Eric Horvitz, and Ece Kamar.
2025. The collaboration gap. arXiv preprint arXiv:2511.02687.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. 2024. Improving factuality and reasoning in language models through multiagent debate. In Forty-first international conference on machine learning.
Jacob Eisenstein, Fantine Huot, Adam Fisch, Jonathan Berant, and Mirella Lapata. 2026. Mt-pingeval: Evaluating multi-turn collaboration with private information games. arXiv preprint arXiv:2602.24188.
Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges.
arXiv preprint arXiv:2402.01680.
Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework.
In The Twelfth International Conference on Learning Representations.
Yue Huang, Zhengqing Yuan, Yujun Zhou, Kehan Guo, Xiangqi Wang, Haomin Zhuang, Weixiang Sun, Lichao Sun, Jindong Wang, Yanfang Ye, and 1 others.
2024. Social science meets llms: How reliable are large language models in social simulations? arXiv preprint arXiv:2410.23426.

Michal Kosinski. 2024. Evaluating large language models in theory of mind tasks. Proceedings of the National Academy of Sciences, 121(45):e2405460121.
Stephen C Levinson. 2003. Space in language and cognition: Explorations in cognitive diversity, volume 5.
Cambridge University Press.
Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel:
Communicative agents for" mind" exploration of large language model society. Advances in neural information processing systems, 36:51991–52008.

Weixiang Sun, Shang Ma, Yiyang Li, Tianyi Ma, Zehong Wang, Colby Nelson, Xusheng Xiao, and Yanfang Ye. 2026. Prescam: A benchmark for predicting scam progression from early conversations. arXiv preprint arXiv:2605.12243.
Takuma Udagawa and Akiko Aizawa. 2019. A natural language corpus of common grounding under continuous and partially-observable context. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 7120–7127.

Nan Li, Albert Gatt, and Massimo Poesio. 2025.
Grounded misunderstandings in asymmetric dialogue: A perspectivist annotation scheme for maptask. arXiv preprint arXiv:2511.03718.

Arjun V Sudhakar, Hadi Nekoei, Mathieu Reymond, Miao Liu, Janarthanan Rajendran, and Sarath Chandar. 2025. A generalist hanabi agent. In International Conference on Learning Representations, volume 2025, pages 38363–38383.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, and 1 others. 2024. Agentbench:
Evaluating llms as agents. In International Conference on Learning Representations, volume 2024, pages 52989–53046.

Zehong Wang, Fang Wu, Hongru Wang, Xiangru Tang, Bolian Li, Zhenfei Yin, Yijun Ma, Yiyang Li, Weixiang Sun, Xiusi Chen, and 1 others. 2026. Why reasoning fails to plan: A planning-centric analysis of long-horizon decision making in llm agents. arXiv preprint arXiv:2601.22311.

Jakob Marschak. 1955. Elements for a theory of teams.
Management science, 1(2):127–137.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, and 1 others. 2024. Autogen: Enabling next-gen llm applications via multiagent conversations. In First conference on language modeling.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22.
Martin J Pickering and Simon Garrod. 2004. Toward a mechanistic psychology of dialogue. Behavioral and brain sciences, 27(2):169–190.
Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, and 1 others. 2024. Chatdev: Communicative agents for software development. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pages 15174–15186.
Lloyd S Shapley and 1 others. 1953. A value for nperson games.
James WA Strachan, Dalila Albergo, Giulia Borghini, Oriana Pansardi, Eugenio Scaliti, Saurabh Gupta, Krati Saxena, Alessandro Rufo, Stefano Panzeri, Guido Manzi, and 1 others. 2024. Testing theory of mind in large language models and humans. Nature human behaviour, 8(7):1285–1295.
Haochen Sun, Shuwen Zhang, Lujie Niu, Lei Ren, Hao Xu, Hao Fu, Fangkun Zhao, Caixia Yuan, and Xiaojie Wang. 2025. Collab-overcooked: Benchmarking and evaluating large language models as collaborative agents. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 4922–4951.

Advait Yadav, Sid Black, and Oliver Sourbut. 2026.
More capable, less cooperative?
when llms fail at zero-cost collaboration. arXiv preprint arXiv:2604.07821.
Yanfang Ye, Zheyuan Zhang, Tianyi Ma, Zehong Wang, Yiyang Li, Shifu Hou, Weixiang Sun, Kaiwen Shi, Yijun Ma, Wei Song, and 1 others. 2025. Llms4all:
A review on large language models for research and applications in academic disciplines. arXiv e-prints, pages arXiv–2509.
Zhengqing Yuan, Yixin Liu, Yihan Cao, Weixiang Sun, Haolong Jia, Ruoxi Chen, Zhaoxu Li, Bin Lin, Li Yuan, Lifang He, and 1 others. 2024. Mora: Enabling generalist video generation via a multi-agent framework. arXiv preprint arXiv:2403.13248.
Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. 2024. Building cooperative embodied agents modularly with large language models.
In International Conference on Learning Representations, volume 2024, pages 19373–19401.
Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar, and Xian Li. 2025. Sweet-rl: Training multi-turn llm agents on collaborative reasoning tasks. arXiv preprint arXiv:2503.15478.

A

Theoretical Setup

This appendix gives the formal Definition of the collaboration tax summarised in Section 3, the cooperative-game formalisation underlying it, and proofs of the two Propositions invoked there, plus two further propositions on Shapley value allocation that justify the asymmetric pair effects discussed in Section 6.

By the union-necessary property of the partition introduced in Section 4.1, v({i}) is realised by a single agent given the full instance x = v1 (x) ∪ v2 (x), and v({1, 2}) is realised by the paired protocol on the same instance distribution. In this notation, Definition 1 reads  tax (M1 , M2 ), T =  (8)
max v({1}), v({2}) − v({1, 2}).

A.1

A.3

Formal definition

Definition 1 (Collaboration Tax). Let T = (X , µ, U ) be a task specified by an instance distribution µ over an instance space X and a deterministic grader U : X × A → [0, 1] that maps an instance and an answer to a continuous score. Let x 7→ (v1 (x), v2 (x)) be a union-necessary partition satisfying v1 (x) ∪ v2 (x) = x for every x ∈ X . For a pair of language-model agents (M1 , M2 ) executing a paired protocol Π that produces a candidate answer Π(M1 , M2 ; v1 (x), v2 (x)) ∈ A from the private views, and writing M (x) for the answer produced by a single instance of model M given the full instance x, the collaboration tax of the pair on task T under protocol Π is   Vsolo := max Ex∼µ U (x, Mi (x)) , (2)
i∈{1,2}   Vpair := Ex∼µ U x, Π(M1 , M2 ; v1 (x), v2 (x)) , (3)
 tax (M1 , M2 ), T = Vsolo − Vpair .
(4)
For homogeneous pairs M1 = M2 = M the maximum reduces to Ex [U (x, M (x))], and we write tax(M, T ).
c of Equation 1 in The operational estimator tax Section 3 replaces Vsolo and Vpair by their cellmeans over 50 rollouts per cell with independent seeds.
A.2

Cooperative team formulation

Following Marschak (1955), we model each paired execution as a 2-agent cooperative team problem with private information. Let (N, v) denote a cooperative game with player set N = {1, 2} and characteristic function v : 2N → [0, 1] given by

Information dominance

Proposition 1 (Information dominance). If a paired protocol Π produces an answer distribution that is measurable with respect to (v1 (x), v2 (x)), and Vsolo in Definition 1 is realised by a policy that is optimal among all measurable functions of x, then tax((M1 , M2 ), T ) ≥ 0.
Proof. Let π Π : X → ∆(A) denote the answer distribution induced by the paired protocol, marginalised over the conversation, as a function of the joint private views (v1 (x), v2 (x)).
By union-necessity, (v1 (x), v2 (x)) is a measurable function of x, so π Π is also a measurable function of x, and the expected payoff Ex [U (x, a) | a ∼ π Π (x)] is well-defined.
Let π ⋆ ∈ arg maxπ Ex,a∼π(x) [U (x, a)] over all measurable policies π : X → ∆(A).
Since π Π lies in the feasible set, we have E[U (x, π ⋆ (x))] ≥ E[U (x, π Π (x))]. Under the assumption in the proposition, Vsolo in Definition 1 attains the supremum on the right-hand side, so tax((M1 , M2 ), T ) ≥ 0.
c in Equation 1 reThe operational estimator tax places the optimal centralised policy with the policy realised by a single LLM agent given the full instance. In general this single-agent policy is not the supremum over all measurable policies (an LLM with x in context may still produce a suboptimal c can take negative answer), and accordingly tax values when the paired protocol happens to outperform the single-agent baseline. Section 5.2 reports the small set of such cells.
A.4

Max-superadditivity equivalence

Proposition 2 (Max-superadditivity). For any pair (M1 , M2 ) and task T , tax((M1 , M2 ), T ) > 0 if and only if the cooperative game with v(∅) = 0, (5)
value function v({i}) = E[U (x, Mi (x))] and   v({i}) = Ex∼µ U (x, Mi (x)) , (6) v({1, 2}) = E[U (x, Π(M1 , M2 ; v1 (x), v2 (x)))]
  v({1, 2}) = Ex∼µ U (x, Π(M1 , M2 ; v1 (x), v2 (x))) . violates max-superadditivity:
v({1, 2}) < (7) max(v({1}), v({2})).

Proof. Substituting Equation 8, tax((M1 , M2 ), T ) > 0 is equivalent to max(v({1}), v({2})) − v({1, 2}) > 0, that is v({1, 2}) < max(v({1}), v({2})). This is precisely the failure of max-superadditivity for the cooperative game (N, v).
A coalition that satisfies max-superadditivity produces at least as much joint utility as its strongest member acting alone. The empirical content of Proposition 2 is therefore the claim that LLM pairs across our four heterogeneous configurations fail this property on the majority of tasks, as Table 2 of Section 6 reports.
A.5

Shapley value and subadditivity

The Shapley value (Shapley et al., 1953) provides an axiomatic allocation of the joint payoff v({1, 2})
between the two members under efficiency, symmetry, and the null-player axiom.
Proposition 3 (Shapley value of a 2-agent cooperative game). For the cooperative game (N, v) with N = {1, 2}, the Shapley value of player i is   ϕi = 12 v({i})+v({1, 2})−v({j}) , j ̸= i.
(9)
Proof. The Shapley value of player i is the average marginal contribution over orderings σ of N :
 1 X ϕi = v(Siσ ∪ {i}) − v(Siσ ) , |N |! σ where Siσ is the set of players preceding i in ordering σ. For |N | = 2 there are two orderings:
(i, j) contributes v({i})−v(∅) = v({i}), and (j, i)
contributes v({1, 2}) − v({j}). Averaging yields Equation 9.
Proposition 4 (Subadditivity implies both members lose Shapley share). If v({1, 2}) < v({1}) + v({2}), then ϕi < v({i}) for both i ∈ {1, 2}.
Proof. From Equation 9,  ϕi − v({i}) = 12 v({1, 2}) − v({1}) − v({2}) , (10)
which is negative under the hypothesis. The righthand side does not depend on i, so both members lose by the same absolute amount.
Proposition 4 is empirically applicable: all four of our heterogeneous pair configurations satisfy v({1, 2}) < v({1}) + v({2}) on the majority of tasks, so both members’ Shapley values are strictly

below their singleton payoffs. Table 2 reports the cooperative-game values and Shapley values averaged across the 32 tasks. In every pair, the strong member’s Shapley value falls substantially below its solo payoff (e.g., ϕgpt-5 = 0.668 versus v({gpt-5}) = 0.932 on the nano × gpt-5 configuration), illustrating that under Shapley fairness the stronger member’s marginal contribution to the team is well below its solo capacity.
A.6

Design principles as theoretical requirements

We sketch why each of the four design principles of Section 4.1 is required by Definition 1 or Proposition 1, in the sense that violating it forces the tax to be uninformative, ill-defined, or zero by construction.
Solo-trivial. When Vsolo is far from ceiling, both Vsolo and Vpair are bounded above by problemsolving capacity rather than by coordination quality.
The difference Vsolo − Vpair then aggregates two unc near related sources of failure, and an empirical tax zero is indistinguishable from a regime where both the solo agent and the pair fail for problem-solving reasons. Tuning the task so that Vsolo is near ceiling restores the interpretation of the tax as a clean coordination signal.
Union-necessary. Definition 1 assumes a partition v1 ∪ v2 = x with neither view alone realising the canonical answer. If the partition is degenerate, say v1 = x, then the paired protocol Π can ignore v2 and emulate the solo policy exactly, producing Vpair = Vsolo and forcing tax = 0 by construction regardless of coordination ability. Union-necessity is the minimum partition non-degeneracy that gives the tax a chance to be positive.
Algorithmically verifiable. Definition 1 requires U : X × A → [0, 1] to be a deterministic function, so that Vsolo and Vpair are population quantities and the cell-level estimator over 50 rollouts has variance only from agent stochasticity. A stochastic grader adds an independent noise source to both terms that is not absorbed by the within-cell averaging, so the estimator becomes inconsistent for the population tax at any finite sample size.
Multiply expressible. Proposition 1 states tax ≥ 0 under information dominance; the upper bound Vpair = Vsolo is attainable in principle whenever Π can recover the full instance from the views. If the views are trivially mergeable, for instance one

agent serialising its view to the other in a canonical form that both agents share, then Π can directly emulate the centralised baseline and the upper bound binds with equality at zero coordination effort. Multiple equivalent surface representations break this trivial pass-through: agents grounded in different schemes (origin, axis order, naming convention)
cannot simply concatenate their views without first aligning representations, so the upper bound is approached only by competent coordination, and the tax becomes a measure of that competence.
A.7

Remarks

Negative tax exceptions. For sufficiently strong models on CSP tasks the homogeneous tax can be negative; claude-sonnet-4-5 and DeepSeek-V4-Pro on CSP are the two cases reported in Section 5.2. This violates the classical tax ≥ 0 inequality and reflects the fact that LLM policies are stochastic and context-dependent: an agent’s effective decision distribution can differ between solo and paired modes, breaking the policy-consistency assumption Marschak (1955)
invokes when proving non-negativity.
Bounded protocol. The classical equality condition (tax = 0 iff communication is sufficient to reconstruct v1 ∪ v2 ) requires unbounded message exchange. Our protocol caps the dialogue at 50 turns, as Appendix I describes; we conjecture that part of the residual tax under the all-four prompt intervention of Section 5.5 is attributable to this bound.
Scope. The formalism here applies to 2-agent pairs with a single common payoff and unionnecessary information. Extensions to n ≥ 3 agents, weighted contributions, or non-cooperative settings are outside the scope of this paper.

B

Failure-Description Clustering

This appendix details the pipeline that produced the sixteen failure themes referenced in Section 5.3 and the cluster-to-cascade-stage mapping that motivated the four-stage judge.
Pipeline. For each failed homogeneous rollout we ask an LLM (gpt-4o-mini, temperature 0.2)
for an open-ended description of what went wrong in the conversation, under a strictly neutral prompt that does not pre-list candidate failure types and does not name any stage of the protocol. We collect 700 such descriptions covering all eleven models

and all three task categories, embed them with TFIDF on lower-cased uni- and bi-grams, and cluster the embeddings with cosine k-means. The number of clusters, k = 16, is chosen as the smallest setting under which every cluster is dominated by a single task family (Spatial, Relational, or CSP);
beyond k = 16 further splits produce sub-clusters within a single task type rather than new failure mechanisms.
Anti-bias naming rule. We then ask the same LLM to produce a short, behaviourally specific name for each cluster from its centroid descriptions, under a strict constraint: a black-list of generic words (verification, verify, pool, share, communicate, collaborate, cross-check, calculation, format) is forbidden anywhere in the name.
The rule prevents the namer from collapsing distinct mechanisms into the same vacuous descriptor (every spatial coordination failure being labelled “verification” would erase exactly the structure we are trying to surface) and forces names that refer to concrete observable behaviours, such as moved_without_confirming_layout or failed_to_clarify_conflict_constraints.
Cluster-to-cascade mapping. We inspected the 16 cluster names and their centroid transcripts and grouped each cluster by which stage of the dialogue the failure occurs at, following the classical grounding pipeline of Clark and Brennan (1991); Clark (1996): claims that should be sourced from either view (L1 Grounding), queries that elicit missing information from the partner (L2 Querying), uptake that integrates the two views into a combined statement (L3 Integration), and acknowledgment that re-derives the joint claim before committing (L4 Re-derivation). The resulting per-cluster mapping is reported in Table 3. The four stages are encoded as four independent judge prompts that a separate LLM applies to every rollout without exposure to the cluster labels (Appendix G), and validated against three blind expert annotators in Appendix C.

C

Human Validation of the Cascade Judge

We validate the four-stage cascade judge with four independent expert annotators re-labelling a stratified sample of homogeneous rollouts under a shared guideline. We report per-dimension Fleiss’ κ across the four experts, pairwise Cohen’s κ for

Stage

Cluster name

n

Top task

Category

L1 Grounding

stated_incomplete_attribute_info stated_incorrect_counts_without_confirmation stated_incorrect_availability_times failed_weight_confirmation assumed_correct_mappings_without_checking assumed_incorrect_word_fit assumed_pairs_without_validation

78 64 34 32 29 24 19

attrcombo graphattr seating graph cipherdecode crossword numberlink

Relational Relational CSP Relational CSP CSP Spatial

L2 Querying

failed_to_confirm_wall_layout failed_to_clarify_conflict_constraints

48 33

geomaze meetingmatrix

Spatial CSP

L3 Integration

moved_without_confirming_layout proposed_order_without_full_analysis stated_status_without_confirmation ignored_blocked_cells_in_path failed_capacity_check_before_assignment

109 37 33 30 25

cube order floodfill timedpath binpack

Spatial CSP Spatial Spatial CSP

L4 Re-derivation

confirmed_incorrect_paths_without_checking failed_to_verify_triangle_existence

56 49

treesum trianglecount

Relational Relational

Table 3: Cluster-to-cascade-stage mapping. For each of the 16 failure themes: cluster name, number of descriptions, top task, dominant category. Centroid examples and full per-cluster distributions are released with the code.

every expert pair, and Cohen’s κ between the automated cascade judge of Section 5.3 and the majority vote of the four experts.

pairwise means across the six pairs closely track the Fleiss’ κ values in Table 4.
Pair

Sample. 100 rollouts stratified jointly on outcome and category (50 failures, 50 successes;
roughly 33 per task category; all 11 models and 31 of 32 tasks represented). Per-dimension stratification on the automated judge ensures both label values appear in the sample for every dimension.
All annotators worked blind to model identity, rollout score, and the automated judge’s label.
Inter-annotator agreement. Agreement is moderate to substantial across all four stages, with L2 and L3 substantial and L1 and L4 moderate; L1 is the lowest because identifying every fabricated atomic value in long deductive transcripts is intrinsically threshold-sensitive.
Dimension

Fleiss’ κ

L1 grounded L2 queried L3 integrated L4 re-derived

0.398 0.650 0.694 0.567

mean

0.577

Table 4: Inter-annotator agreement across four expert annotators. Landis-Koch: ≥ 0.40 moderate, ≥ 0.60 substantial. Bold marks the highest dimension.

Pairwise Cohen’s κ for the six expert pairs is reported in Table 5. The closest pair (E1 × E2)
reaches almost-perfect agreement on L3 (κ = 0.832) and substantial agreement on L2 (0.772);

L1

L2

L3

L4

E1 × E2 E1 × E3 E1 × E4 E2 × E3 E2 × E4 E3 × E4

0.494 0.342 0.398 0.400 0.458 0.306

0.772 0.583 0.659 0.683 0.764 0.467

0.832 0.646 0.739 0.608 0.699 0.661

0.544 0.696 0.666 0.584 0.368 0.654

mean

0.400

0.655

0.698

0.585

Table 5: Pairwise Cohen’s κ across the six expert pairs. Bold marks the column-wise maximum. Pair means closely match the Fleiss’ κ values in Table 4.

Automated-judge alignment. Cohen’s κ between the automated cascade judge and the majority vote of the four experts is substantial on L1 (κ = 0.707), L2 (0.668), and L3 (0.725), and moderate on L4 (0.508). Judge-vs-majority agreement on L1 (0.707) exceeds inter-annotator agreement on the same dimension (0.398), consistent with majority voting denoising individual-annotator threshold variation; this supports the use of the automated cascade labels in Section 5.3 and Figure 2.
Guideline. The shared guideline contains: (i)
verbatim L1 to L4 definitions from Section 5.3;
(ii) a list of common boundary cases, for example that an incorrect final answer is not by itself an L1 violation; (iii) an independence reminder that each dimension must be judged separately, without one stage’s decision contaminating another; and (iv) several worked examples drawn from real tran-

scripts that cover the frequent edge cases on each dimension. The guideline document is released with the code.

D

Cascade Co-occurrence

This appendix reports the joint firing pattern of the four cascade stages on the failed homogeneous rollouts. A stage is said to fire on a rollout when the corresponding judge dimension is 0 (the mechanism is absent).
Most failures fire multiple stages simultaneously.
Table 6 shows the distribution of the number of stages firing per failed rollout. 67% of failed rollouts fire at least two stages, justifying the §5.3 claim that the four stages are separable but not independent.
# stages firing

% of failed rollouts

0 1 2 3 4

12.8% 20.2% 22.5% 22.7% 21.8%

Table 6: Number of cascade stages firing per failed rollout. Distribution over failed homogeneous rollouts.

L3 and L4 are the most entangled pair; L2 is the most independent. Table 7 reports pairwise Jaccard overlap (intersection / union of failure sets)
for every pair of cascade dimensions. L3 and L4 co-fire at Jaccard 0.67, the largest overlap by a wide margin, which motivates the observationalvs-causal reconciliation in §5.3 and §5.5. L2 has the lowest average pairwise Jaccard with the other three stages and the highest sole-fire rate (13.9% of failed rollouts fire L2 and no other stage), confirming L2 as the most independent of the four dimensions. L3 never fires alone in our data (0% sole-fire), consistent with L3 sitting late in the cascade so that an integration failure is virtually always accompanied by an earlier grounding, query, or re-derivation failure as well.
L1 L1 L2 L3 L4

0.37 0.54 0.56

E Case Studies: One Failure per Cascade Stage To complement the quantitative cascade analysis of Section 5.3 and Appendix D, we walk through one illustrative homogeneous-mode failure per cascade stage.
E.1

L1 Grounding: streetmap, claude-sonnet-4-5

Setup. A 7 × 7 street map. Agent 1 sees one-way arrows and . cells; Agent 2 sees walls (#), arrows, and . cells. Both have @ at (0, 0) and * at (6, 6).
Agent 1’s view contains no # cells, so any claim about walls must be sourced from Agent 2.
What happened. In turn 2 Agent 2 produces a “combined map” before Agent 1 has shared anything mappable to coordinates. The walls Agent 2 lists at (1, 3), (4, 2), (5, 2), (6, 3), (6, 4), (0, 6) do not exist at those positions in Agent 2’s own grid; they correspond to Agent 2’s true walls with row and column axes swapped (true walls at (3, 1), (2, 4), (2, 5), (3, 6), (4, 6), (6, 0)). Agent 2 fabricates a global state from its private grid under the wrong axis convention. Agent 1 corrects coordinates in turn 3 but never challenges the wall list, so the fabricated walls propagate into the agreed route.
[turn 2, Agent 2]
Analysis:
- Start: (0,0), Goal: (6,6)
- I see walls at (1,1), (1,3), (3,3), (4,2), (5,2), (6,3), (6,4), (0,6)
- One-way streets at (2,1), (0,2), (2,3), (4,4), (0,5), (1,6)

The grounding chain (“I see walls at X”) is broken at the moment of utterance.
Why this is L1. A query was made (“what does the right half look like?”), an integration block was produced, and the receiving agent verified the route step by step. The chain is sound; the upstream input was fictional.

L2

L3

L4

E.2

0.37

0.54 0.39

0.56 0.51 0.67

Setup. A 4-position Einstein-style puzzle. Each agent holds six private clues. Each agent must elicit clues from the partner; emitting one’s own clues serially does not establish that the partner has shared theirs.

0.39 0.51

0.67

Table 7: Pairwise Jaccard overlap of the four cascade dimensions. Computed over failed homogeneous rollouts. L3 and L4 are the most entangled pair (bold).

L2 Querying: logic, gpt-4o-mini

What happened. Across seven turns the agents take turns announcing a single new clue per mes-

sage, make a deduction from the union of clues stated so far, and never ask the partner for a specific fact. No utterance of the form “please state whether the Spanish person drinks water” appears in the transcript.
[turn 1, Agent 1]
The red house must be in position 1, 2, or 3 ...
I will share my next clue: the tea drinker is NOT the blue house.
[turn 2, Agent 2]
Since the Spanish person is in position 1, the red house must be in position 2 or 3 ...
I'll add: the green house is to the LEFT of yellow.
[turn 6, Agent 2]
Yes, that is correct! ... ACTI!

The final answer matches 1 of 12 attribute cells;
score 0.083.
Why this is L2. Every clue that was stated came from one of the agents’ private clue lists, so claims that were made are grounded. The problem is the absence of a query mechanism: the secondmessage query mandate from the L2 intervention clause would have forced an explicit ask about the missing positional constraints that the agents instead inferred from thin air.
E.3

Why this is L3. L4 requires the receiver to rederive before accepting, and one cannot re-derive nothing. The integration step where the two agents would have written down their merged map state never occurs. This case also illustrates the observation in Appendix D that L3 has 0% sole-fire rate:
when integration is absent, the proposal is empty enough that L4 has no content either, so L3 and L4 co-fire.
E.4

Setup. A 7×7 split-blueprint maze. Agent 1 sees the left columns and the start @; Agent 2 sees the right columns and the goal *. Start at (3, 3), goal at (2, 6).
What happened. The exchange runs cleanly through L1 (no fabricated cells), L2 (Agent 1 asks “where is the goal? what does the right half look like?”), and L3 (Agent 2 prints an explicit combined map). Agent 2 then proposes a route and emits ACTI! in the same message, without waiting for Agent 1 to re-walk the path.
[turn 1, Agent 2]
Combined map (0-indexed rows 0-6, columns 0-6):
Row 0: . . # # . . .
Row 1: . # . . . . .
Row 2: . # # # . . * Row 3: . . @ . # . . <- start (3,3)
Row 4: # # . . # . .
Row 5: . . . . . . .
Row 6: . . # . . # .

L3 Integration: maze, DeepSeek-V4-Pro

Setup. A 7 × 7 split-view maze. Both views contain enough partial information that an integration step is required before the first move.
What happened. The transcript is four messages long. Agent 1 surveys their own grid, proposes a single first move, asks an open question, and Agent 2 immediately terminates.
[turn 0, Agent 1]
Looking at my map: ... my current position is at (0,0) marked with @.
I'll propose my first move: move right to (0,1).
What do you see on your map?
[turn 1, Agent 2]
Move right to (0,1).
[turn 2, Agent 1]
I agree. Let's both move right to (0,1).
... What do you see after moving?
[turn 3, Agent 2]
ACTI!

Proposed route: (3,3), (3,4), (3,5), (3,6), (2,6)
Check walls: (3,4) is '.', (3,5) is '.', (3,6) is '.', (2,6) is '*'. ACTI!

The integration block is inconsistent with itself:
the combined map lists (3, 4) as #, but the route check immediately below says (3, 4) is .. A rederivation pass by Agent 1 would have caught the contradiction in the very block Agent 2 wrote.
Why this is L4. Integration was produced (the explicit combined-map block). The breakdown is at the final accept step: the receiver was given no chance to recompute, and the proposer selfterminated. The L4 clause requiring the other agent to show recomputation work before either emits ACTI! is exactly the structural fix.
E.5

There is no combined-state block, no enumeration of cells either agent has resolved, and no plan beyond move 1. The decisive claim is two coordinates long, (0, 0) and (0, 1); score 0.083.

L4 Re-derivation: blueprintmaze, DeepSeek-V4-Pro

Cross-case summary

• L1 failures are sourcing errors: the surface form of the claim is well-formed English but the content is invented (here by axis transposition of the speaker’s own grid).

• L2 failures are silences: each utterance is wellformed; what is missing is a question.
• L3 failures are truncations: the dialogue ends or commits before a merged-state utterance ever appears.

grader scores per-query exact match against the ground-truth ancestry computed on the merged DAG, averaged over the q queries.

The four cases come from three task categories (Spatial, CSP, Spatial, Spatial), consistent with the per-category bottleneck pattern of Section 5.5: Spatial path tasks expose L1, L3, and L4 most cleanly, while CSP logic puzzles expose the absence of L2 queries.

CSP representative: schedule. The answer is an assignment of m events to time slots respecting joint availability and time-zone constraints. The full instance is a set of participants with per-zone availability windows together with m events, each requiring a subset of participants. The partition splits availability information: each agent sees the windows for one half of the participants, so feasibility for any cross-half event requires the agents to combine their views. The grader counts the number of events placed in a slot where all required attendees are available in their local zone, divided by m.

F

G

Prompts

G.1

Protocol Prompts

• L4 failures are premature ACTI!: the proposer self-confirms instead of the receiver rederiving.

Full Task Specifications

The suite spans 32 tasks across three structural families. Table 8 summarises every task with its answer type and one-line grader rule. We then deep-dive on a single representative task per family, chosen for pedagogical clarity rather than for being typical:
maze (Spatial, the canonical task of Davidson et al., 2025), relaquery (Relational), and schedule (CSP).
The remaining 29 tasks follow the same structure with task-specific instance generators, splitters, and graders; the full per-task definitions are released as part of the code accompanying this paper.
Spatial representative: maze. The answer is a sequence a = (a0 , a1 , . . . , ak ) of (row, col) coordinates from start a0 to goal ak on an n × n grid.
The full instance is a grid of cells labelled wall (#), path (.), start (@), goal (*). The partition independently masks roughly half of each agent’s cells (cells become ?) such that the union recovers the full grid. The grader extracts the proposed sequence from the dialogue, brute-forces over all 16 origin / orientation / axis-order combinations following Davidson et al. (2025), and reports the prefix of moves that are valid in the original grid, normalised by the optimal-path length.
Relational representative: relaquery. The answer is a list of q relational queries answered as triples (s, rel, o) over a directed family DAG with parent_of edges. The full instance is the union of two edge-disjoint subgraphs, one per agent (Pattern B split): every parent edge belongs to exactly one agent’s view, so queries that traverse cross-partition paths require explicit information exchange. The

Collaboration system prompt. Each agent in a paired rollout receives the following system prompt, which establishes the relay protocol with the partner.
System Message The user will act as an intermediary between you and another agent.
The user will directly forward your messages to the other agent and vice versa.
The user will not see or modify the messages, but will relay them as is.
Messages coming from the other agent will be prefixed with “[other agent]:”.
Messages coming from the user will be prefixed with “[user]:”.
Do not add any additional prefixes or suffixes to your own messages.

Critic prompt. In the solo modes, after producing a candidate answer, the same model receives the following critic prompt before grading.
User Message Carefully review the final solution you have provided above.
Ensure that it is complete, valid, and consistent with every rule and constraint of the task.
If needed, make adjustments to ensure your answer meets all requirements.
After you have finished reviewing, please submit your final solution.

Grader prompt. The grader is invoked once per rollout with the full dialogue. It extracts the final proposed answer from the natural-language exchange and returns a structured YAML object containing both the parsed answer and, where the surface representation can vary (coordinate origin, axis order, naming convention), the inferred answer schema.

Family

Task

Answer

Grader

maze

coordinate sequence

blueprintmaze conflictpath cube floodfill geomaze numberlink rulemaze streetmap terrain timedpath

coordinate sequence two coordinate sequences 3D coordinate sequence set of reachable cells coordinate sequence multi-path coordinate sequences coordinate sequence coordinate sequence coordinate sequence timed coordinate sequence

valid-prefix length / optimal length, brute-forced over 16 schemes valid-prefix length / optimal length joint valid moves / optimal joint plan valid-prefix length / optimal length IoU against the true reachable set valid-prefix length / optimal length fraction of endpoint pairs correctly connected valid-prefix length / optimal length under colour-rule valid-prefix length / optimal length under one-way rules optimal cost / achieved cost time-feasible valid-prefix length / optimal length

Relational

aliasedtaxonomy ancestorpath dagorder eulerpath graph graphattr graphpath noderanking relaquery treesum trianglecount

node-pair alignment list of ancestor paths topological order edge sequence coordinate sequence on graph per-query answers edge sequence ordered top-3 nodes per-query answers numerical sum set of triangles

per-pair correctness per-query exact match 1− normalised Kendall tau fraction of edges covered without repeat optimal-path correctness per-query exact match fraction of correct edges ordered set IoU per-query exact match exact value set IoU

CSP

attrcombo binpack cipherdecode crossword eventlog logic meetingmatrix order schedule seating

per-query answers item-to-bin assignment decoded string filled grid per-query answers full assignment schedule assignment total order event-to-slot assignment circular order

per-query exact match fraction of items respecting capacity character accuracy cell accuracy per-query exact match cell accuracy fraction of constraints satisfied 1− normalised Kendall tau fraction of events placed in a feasible slot 1− normalised circular distance

Spatial

Table 8: The 32-task suite by structural family. Each row gives the answer type produced by the agents and the one-line grader rule, normalised to [0, 1]. Deep-dive specifications for the three representative tasks (maze, relaquery, schedule) appear below; the full task-specific definitions for all 32 are released with the code.

User Message You will be given a dialogue produced by one or two agents attempting to solve the task above.
Your job is to extract the final proposed answer from the dialogue, infer the answer schema where applicable, and return the result as a YAML object.
Do not include any commentary or analysis outside the YAML.
# Answer schema <task-specific YAML schema>

final answer.

L2 query mandate.
Appended to system prompt On your second message, before making any claim about positions or connectivity that your partner might know better, ask one specific question of the form “Please state X.” If your partner cannot answer, mark X as unknown. Do not act on assumed values for X.

# Dialogue <dialogue transcript>

G.2

Intervention Clauses

Each clause below is appended to the collaboration system prompt during the corresponding intervention condition of Section 5.5. The all-four condition appends all four; each no Lk condition appends three.
L1 grounding clause.
Appended to system prompt For every quantitative claim you make (a value, count, weight, position, edge, or mapping), cite the source: either “from my view, . . . ” or “as [other agent] said in turn N , . . . ”. Claims without a cited source must be marked as (assumed) and may not appear in the

L3 integration block.
Appended to system prompt Before any agent issues a final-answer proposal, that agent’s preceding message must begin with an explicit integration block: “Combined state from both views: [facts from my view + facts from partner’s view + derived consequences].” Proposals not preceded by an integration block in the same turn are invalid; the other agent should reject them and request integration.

L4 re-derivation requirement.
Appended to system prompt Before either agent issues ACTI!, the other agent must show recomputation work in their immediately preceding message: re-walk the path step by step, recompute the sum, re-check each constraint.
“I agree” or “confirmed” without the recomputation is invalid; the proposer must reject it and ask for explicit re-derivation.

H

Additional Results

OOF +0.106 R² = +0.480 OOF ρ = +0.766

gpt-4.1-nano

Per-task decomposition within each category.
The per-task decomposition (Figure 6) reproduces the category effect at finer resolution. The high end of the distribution is dominated by Spatial path tasks; the low end is dominated by CSP tasks whose answer is a short assignment over named entities.
Two tasks, numberlink and crossword, fall short of the solo-tractability target and their ratio gaps are reported but treated as exploratory.

+0.095

char_per_turn

+0.076

Gemini-2.5-FL Qwen3-8B

+0.070

gpt-4o-mini

+0.066

gpt-4.1-mini

+0.062

Phi-4

+0.056 +0.050

Llama-4-Mav -0.070

view2_dump_ratio -0.108

Relational

CSP

model category structural

-0.122

last3_chars

-0.141

−0.20

−0.15

−0.10

−0.05

0.00

0.05

0.10

0.15

0.20

standardised coefficient on ratio_gap

58

maze numberlink† blueprintmaze streetmap terrain rulemaze geomaze cube treesum conflictpath timedpath relaquery noderanking ancestorpath trianglecount cipherdecode floodfill aliasedtaxonomy logic meetingmatrix eulerpath graphattr graph dagorder attrcombo eventlog binpack seating graphpath order crossword† schedule

54 53 52 51 43 43 42 35 35 33 31 30 29 28 26 25 22 20 19 18 18 16 16 15 14 14 13 12 12 11 11

0

10

20

Spatial Relational CSP † solo_full < 0.55 30

40

50

60

ratio_gap (%, mean across 11 models)

Figure 6: Per-task ratio gap, by category. Mean ratio gap on the homogeneous mode, by task, averaged across the eleven models. Colours indicate task category. The † marker flags the two tasks (numberlink, crossword) whose solo-full success rate falls short of the solo-tractability target; their ratio gaps are reported but treated as exploratory.

Per-cell heterogeneous heatmap. Figure 9 reports the 3 × 3 heterogeneous-pair ratio-gap heatmap underlying the aggregated scatter of Figure 5. The four off-diagonal cells cluster close to the strong-tier diagonal rather than averaging between strong and weak baselines.
Cascade signatures transfer to heterogeneous rollouts. We re-apply the four-stage cascade judge of Section 5.3 to the heterogeneous rollouts (Figure 8). All four upstream stages continue to discriminate failure from success in the same direction as the homogeneous case: successful hetero pairs ground, query, integrate, and re-derive at high

Figure 7: Top standardised coefficients of the M1 ridge. M1 fits the per-cell ratio gap on conversationstructural features plus model and category dummies.
Model dummies are coloured separately. The model achieves out-of-fold R2 = 0.480 and Spearman ρ = 0.766 under group k-fold by task.

rates across categories, while failed hetero pairs hit each of these rates at substantially lower levels. The querying rate jumps most strongly: hetero pairs query more often than homogeneous pairs on Spatial and Relational, suggesting that pair asymmetry forces an explicit query that homogeneous pairs skip. The qualitative failure mode is the same as in the homogeneous case: failed coordination corresponds to a near-absence of the upstream cascade stages, irrespective of whether the two agents come from the same model or different ones.
Critic ablation. We ablate the solo critic pass on gpt-4o-mini, a lower-tier model on which the critic has the largest plausible room to lift solo performance. Across all 32 tasks at 50 rollouts per cell, removing the critic prompt drops the mean solo-full score from 0.575 to 0.569 (Table 9), a 0.006-point cost. The critic therefore contributes a much smaller fraction of the solo score than the gap to homogeneous collaboration spans on this model, so the tax is not an artefact of the critic step.
Solo-full setting

Mean score

With critic pass Without critic pass

0.5753 0.5690

∆ (critic contribution)

+0.0063

Table 9: Critic ablation on gpt-4o-mini. Mean solofull score across the 32 tasks at 50 rollouts per cell, with and without the critic prompt of Appendix G.1. The critic accounts for 0.6 percentage points of solo score.

homo failure

1.00

1.001.00

0.981.00

hetero failure

homo success

1.00 0.96

hetero success

0.99 0.95

0.98

rate

0.85

0.75

0.590.61

0.60

0.66 0.61

0.49

0.50

0.72 0.64 0.56 0.48

0.62 0.44

0.64

0.63

0.82

0.57

0.51

0.50

0.94

0.94 0.90

Spatial

Relational

CSP

L1 grounded every claim traceable to source

Spatial

0.39 0.32

Relational

CSP

Spatial

L2 queried asked partner for specific facts

0.850.87

0.53 0.50

0.390.37 0.34 0.29

0.25 0.00

0.810.80

0.70

Relational

CSP

L3 integrated answer combines both views

Spatial

0.39 0.28

Relational

0.300.31

CSP

L4 re-derived re-verified before committing

gpt-5

Figure 8: Four-stage cascade in heterogeneous rollouts. Layered onto the homogeneous bars of Figure 2. Each panel reports the rate at which the corresponding cascade dimension fires, separated by category and outcome. The cascade signature continues to discriminate failure from success.

+6

n/a

+15

+65

+30 +15 n/a

+10

gpt-5

+7

+10

+11

+51

+0

ratio_gap (%)

agent 1 (initiator)
gpt-4.1-nano Sonnet-4.5

+45

-10

Sonnet-4.5 gpt-4.1-nano agent 2 (responder)

Figure 9: Heterogeneous pair ratio gap heatmap.
Rows index the agent-1 (initiator) model, columns index the agent-2 (responder) model. Diagonal cells (boxed) reproduce the homogeneous baselines; offdiagonal cells are averaged across tasks. Hatched cells (Sonnet-4.5 × gpt-5) were not run. The pair gap is pulled toward the stronger member.

I

Hyperparameters

Generation. Agents are sampled at temperature 0.7 with the model’s default top-p and no outputtoken cap. The collaborative dialogue is capped at 50 exchanges per rollout, where one exchange is a turn from each agent. Each rollout starts from an empty context, so no state leaks across rollouts. We run 50 independent rollouts per (task, mode, model or pair) cell, with consecutive integer seeds controlling both instance generation and the partition into views.
Grading. The grader is a fixed model (gpt-4o-mini in our experiments), held constant across every cell of the design. In particular, the grader does not change when the agents

do, so heterogeneous-pair comparisons are not confounded by grader-side capability differences.
The grader is queried at temperature 0.5, lower than the agent temperature so that grading variance does not dominate downstream comparisons, and we report its continuous score in [0, 1].

J

API Robustness

All API calls flow through a unified chat() wrapper with a 120-second per-call timeout. Transient network failures trigger up to 5 retries with exponential backoff. We separately observed that pathtask transcripts occasionally trigger Azure content filters when consecutive mask cells and the start or goal markers co-occur in the visible portion; for this class of failure we apply up to 6 retries with longer backoff, and the path-task renderer emits only the visible cells of each view to reduce false triggers.


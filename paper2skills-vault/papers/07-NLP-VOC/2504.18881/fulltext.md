<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2504.18881
     paper_id : 2504.18881
     source   : paper2skills-vault/papers/07-NLP-VOC/2504.18881/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

arXiv:2504.18881v3 [cs.LG] 2 Feb 2026

HANGTAO ZHANG, Rajax Network Technology (Taobao Shangou of Alibaba), China ZHE LI∗ , Rajax Network Technology (Taobao Shangou of Alibaba), China KAIRUI ZHANG, Rajax Network Technology (Taobao Shangou of Alibaba), China Accurate estimation of the Individual Treatment Effect (ITE) is essential for business diagnostics in the online food delivery industry, particularly for assessing the impact of various business strategies, such as inventory management, pricing optimization and online marketing campaigns. A primary challenge in ITE estimation lies in sample selection bias. Conventional approaches utilize treatment regularization techniques such as Integral Probability Metrics (IPM), re-weighting, and propensity score modeling to mitigate this bias. However, these regularizations may introduce undesirable information loss and limit predictive performance. Moreover, treatment effects exhibit substantial heterogeneity across external contextual factors, such as market demand, competitor activity and time dynamics, yet existing methods fail to adequately model the interaction between treatments and context, limiting their causal expressiveness. To address these issues, we propose TSCAN:
a Context-Aware uplift model based on a Two-Stage training approach, comprising CAN-U and CAN-D sub-models. In Stage 1, CAN-U generates counterfactual uplift labels while mitigating selection bias through integrated IPM and propensity score regularization. In Stage 2, CAN-D eliminates these regularizations and leverages an isotonic output layer to directly model uplift effects in a supervised manner. By reinforcing factual outcomes, CAN-D adaptively corrects estimation errors from CAN-U while circumventing the performance degradation induced by bias-mitigation regularizations. Additionally, both stages incorporate a ContextAware Attention mechanism that dynamically fuses the embeddings of merchants, treatments and contextual covariates, thereby capturing context-dependent heterogeneity in treatment effects. We conduct extensive experiments on two real-world datasets to validate the effectiveness of TSCAN. Ultimately, the deployment of our model for real-world merchant diagnosis on one of China’s largest online food ordering platforms validates its practical utility and impact.
Additional Key Words and Phrases: Uplift modeling, Individual Treatment Effect Estimation, Two-Stage training, Context-Aware interaction ACM Reference Format:
Hangtao Zhang, Zhe Li, and Kairui Zhang. 2025. TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis. 1, 1 (February 2025), 20 pages. https://doi.org/XXXXXXX.
XXXXXX.X ∗ Corresponding author.

Authors’ Contact Information: Hangtao Zhang, zht267501@alibaba-inc.com, Rajax Network Technology (Taobao Shangou of Alibaba), Hangzhou, China; Zhe Li, lz171761@alibaba-inc.com, Rajax Network Technology (Taobao Shangou of Alibaba), Shanghai, China; Kairui Zhang, kairui.zhang@alibaba-inc.com, Rajax Network Technology (Taobao Shangou of Alibaba), Shanghai, China.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM XXXX-XXXX/2025/2-ART https://doi.org/XXXXXXX.XXXXXX.X , Vol. 1, No. 1, Article . Publication date: February 2025.

2

1

Hangtao Zhang, Zhe Li, Kairui Zhang

Introduction

In recent years, the e-commerce sector, particularly the food ordering industry, has seen rapid growth, with China emerging as the largest market, boasting a size of $40.2 billion in 2024 [1].
This growth has prompted an increasing number of independent merchants to shift towards online sales [2]. However, many of these merchants lack sufficient experience in online operations, which hinders their effective utilization of complex management workflows and diverse marketing tools [3]. To assist these merchants in accurately identifying business issues and delivering personalized solutions, it is crucial to assess the impact of each diagnostic on their operations. This problem differs from traditional supervised learning in that it requires causal inference rather than mere association modeling. In real-world scenarios, we typically only observe a merchant’s response to a specific marketing strategy (i.e., whether they participated in a particular marketing activity or not), but rarely observe their performance under both participating and non-participating strategies simultaneously. To address this problem, researchers have developed methods for modeling individual uplift, known as Individual Treatment Effect (ITE) Estimation [4, 5]. These techniques are primarily used to evaluate the magnitude of the response to an intervention (treatment) across different individuals. The main ITE estimation methods [6, 7] include: Meta-Learning Methods (e.g., S-Learner [8], T-Learner [8], X-Learner [8]), Tree-based Methods (e.g., BART [9], CausalForest [10]), and Deep Learning Methods (e.g., TransTEE [11], CEVAE [12], CFR-ISW [13]). Recent research trends indicate that deep learning-based methods are gaining popularity due to their powerful non-linear representation and feature interaction capabilities [7, 11, 13–15]. These deep learning models can be further categorized as follows [7]:
• Balanced Representation Learning (e.g., BNN [16], TransTEE).
• Covariate Confounding Learning (e.g., CEVAE, Dragonnet [17]).
• Generative Adversarial Network (GAN)-based models (e.g., CEGAN [18], GANITE [19], SCIGAN [20]).
These methods employ a variety of techniques to mitigate sample selection bias and the adverse effects of confounding factors. For instance, balanced representation learning typically leverages the Integral Probability Metric (IPM) [21] as a regularization term to minimize the distributional discrepancy between representations under control (𝑡 = 0) and treatment (𝑡 = 1) conditions, i.e., between 𝑝 (Φ(𝑥)|𝑡 = 0) and 𝑝 (Φ(𝑥)|𝑡 = 1) [22, 23]. Additionally, CFR-ISW [13] enhances this approach by incorporating a re-weighting strategy, in which the sample weight is related to the propensity score. Methods based on covariate confounding learning primarily encode both observed and unobserved confounders using approaches such as autoencoders or propensity score prediction networks, thereby eliminating the influence of these confounding factors [24–26].
Despite these advances, several limitations persist in current ITE estimation frameworks. Many branch-structured neural architectures and meta-learning algorithms remain restricted to discrete treatments and cannot naturally accommodate continuous treatment spaces. Furthermore, strategies designed to correct for selection bias often introduce auxiliary estimation errors [27]. For example, overly stringent balancing constraints may inadvertently discard outcome-predictive features, degrading model performance [27, 28]. Similarly, reweighting based on propensity scores can distort the empirical data distribution, disproportionately amplifying the influence of rare or long-tail samples [29]. Incorporating a propensity score prediction task will also affect the outcome prediction [13], as evidenced by empirical observations that ablating such regularization components often improves accuracy on observed outcomes under identical training conditions.
Beyond bias correction, a critical gap lies in the inadequate integration of contextual information into treatment effect modeling. In dynamic environments such as online marketing, the efficacy of a given intervention (treatment, e.g., a promotional subsidy or ad bid) is highly context-dependent. For , Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

3

instance, consumer responsiveness to merchant subsidies tends to diminish in high-demand markets, whereas sensitivity increases in saturated or competitive markets with abundant alternatives.
Similarly, the impact of a fixed ad bid varies significantly with the bidding behavior of competitors—a key contextual variable. Such contextual heterogeneity is essential for accurate uplift estimation, yet it remains largely unexploited by existing methods, which typically assume treatment effects are context-invariant.
In summary, two overarching challenges remain: (1) developing more effective methods to address selection bias while maintaining the predictive performance of the model and the quality of personalized recommendations; (2) accounting for the impact of contextual factors on treatment effects.
The main contributions of this work are:
(1) We propose TSCAN, a novel two-stage context-aware uplift modeling framework that decouples bias mitigation (Stage 1) from direct uplift prediction (Stage 2), thereby avoiding the performance degradation caused by conventional regularizations.
(2) We design a Context-Aware Attention Layer that explicitly models the tripartite interaction among merchant features, treatments, and external contexts, enabling adaptive ITE estimation across diverse operational scenarios.
(3) We systematically validate TSCAN through three lenses: (RQ1) benchmark performance:
showing consistent superiority over seven state-of-the-art baselines on two large-scale realworld datasets; (RQ2) architectural ablation: demonstrating the individual value of two-stage training, context-aware attention, and isotonic output; and (RQ3) real-world impact: achieving a 0.76% increase in merchant orders in a live A/B test on one of China’s largest food delivery platforms, which demonstrates its practical value.
2

RELATED WORK

In this section, we provide a concise overview of the primary existing works on uplift models, context-aware treatment effect estimation and feature interaction methods.
2.1

Uplift Modeling: From Meta-Learners to Deep Causal Representations

Uplift modeling aims to estimate heterogeneous treatment effects from observational or experimental data, with applications ranging from personalized medicine [30] to digital marketing [31].
Early approaches adopted meta-learning frameworks (e.g., S-, T-, and X-Learners [8]), which repurpose standard regressors but often fail to capture complex treatment-covariate interactions.
Tree-based methods such as Bayesian Additive Regression Trees (BART) [9] and Causal Forest [10]
improve interpretability and perform well under low-dimensional settings. However, they struggle to effectively model the high-dimensional and sparse features typical of online platforms.
Recent advances leverage deep neural networks to learn balanced representations that mitigate confounding bias [7, 11, 14, 32, 33]. Notable examples include TarNet [22] and CFR [13], which use IPM such as Maximum Mean Discrepancy (MMD) to align treatment and control group distributions in latent space. Dragonnet [17] and CEVAE [12] integrate propensity score estimation to adjust for selection bias, while GAN-based models (e.g., GANITE [19]) simulate counterfactual outcomes.
However, as highlighted in [27, 28], strict enforcement of distributional balance can discard outcomepredictive information, leading to degraded practical performance—a key limitation our two-stage design seeks to overcome.
, Vol. 1, No. 1, Article . Publication date: February 2025.

4

2.2

Hangtao Zhang, Zhe Li, Kairui Zhang

Context-Aware Treatment Effect Estimation

Recent studies increasingly recognize that treatment effects are not static but modulated by external contexts. Huang et al. [34] propose ECUP, a context-enhanced uplift model that leverages user behavior chains and campaign timing to capture phase-dependent treatment efficacy in marketing.
Separately, UMLC [35] addresses robustness in real-time interventions by grouping large-scale contextual features (e.g., merchant, time and location) via response-guided clustering, enabling stable ITE estimation across volatile environments. Though not causal, Afzal et al. [36] demonstrate in recommendation systems that fusing multi-dimensional contexts—temporal, geographic, and social—through deep interaction layers significantly improves context-dependent response modeling, a principle highly relevant to uplift.
These works underscore a critical insight: the same treatment can yield divergent outcomes under different contextual conditions. For example, a merchant discount may significantly increase order volume during off-peak hours but have negligible effect during lunchtime peak periods due to demand saturation. Despite this, most existing uplift models—including TransTEE [11], EFIN [14], and CFR-ISW [13]—either treat context as ordinary covariates or ignore it entirely, failing to model the tripartite interaction among merchant, treatment and context.
Our work bridges this gap by explicitly designing a Context-Aware Attention Layer that dynamically reweights merchant and treatment representations based on contextual embeddings. Through this, we enable end-to-end interaction within a unified architecture.
2.3

Feature Interaction in Causal Models

Accurately capturing how treatments interact with individual characteristics is essential for modeling heterogeneous effects. A growing line of work focuses explicitly on feature interaction in causal models. For instance, TransTEE [11] encodes continuous treatments into semantic embeddings and fuses them with covariates through attention mechanisms to estimate dose-response functions.
EFIN [14] proposes an Explicit Feature Interaction-aware Uplift Network that employs a treatmentgated module to dynamically adjust the importance of customer and product features based on the treatment type, thereby modeling how different offers motivate distinct user segments. MTMT [37]
introduces a treatment–user feature interaction module to model correlations between treatments and user features. However, these approaches primarily model direct pairwise interactions between treatment and covariates, assuming that the interaction pattern is fixed across all samples. In this work, we combine a Treatment-Aware Attention Network with a Context-Aware Attention Layer that explicitly models the dynamic treatment-feature interaction.
In summary, while prior work has made significant strides in representation balancing and treatment-aware interaction, none simultaneously addresses (1) the trade-off between bias correction and predictive fidelity, (2) the dynamic modulation of treatment effects by external context.
The proposed TSCAN framework is designed specifically to resolve these dual challenges.
3

PRELIMINARIES

In the scenario of estimating treatment effects for business diagnosis in the online food ordering 𝑁 industry, our goal is to estimate the ITE using the observed data 𝐷 = {(𝑋𝑖 , 𝑡𝑖 , 𝑦𝑖 )}𝑖=1 , where 𝑋𝑖 , 𝑡𝑖 , and 𝑦𝑖 represent the merchant features, treatment feature, and outcome value, respectively. The treatment variable 𝑡𝑖 can be binary, indicating whether a merchant has initiated a marketing activity (i.e., 𝑡𝑖 ∈ {0, 1}), or continuous, such as the number of customer reviews (i.e., 𝑡𝑖 ∈ R). The outcome variable 𝑦𝑖 is continuous, representing the merchant’s order count or revenue (i.e., 𝑦𝑖 ∈ R). The sample size is denoted by 𝑁 . The potential outcome of the 𝑖-th instance under treatment value 𝑘 is denoted as 𝑦𝑖 (𝑡𝑖 = 𝑘), and the conditional probability of assigning treatment 𝑘 given features , Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

5

𝑋𝑖 is expressed as 𝜋 (𝑋𝑖 , 𝑘) = 𝑃 (𝑡𝑖 = 𝑘 | 𝑋𝑖 ), commonly referred to as the propensity score. In observational data, we can typically observe the outcome values under a specific treatment rather than under all possible treatments. This limitation distinguishes uplift modeling from traditional supervised learning. Uplift modeling seeks to accurately estimate the expected outcome for each instance across different treatments (under ignorability assumption):
𝜏 (𝑋𝑖 , 𝑘 1, 𝑘 0 ) = 𝐸 [𝑦𝑖 (𝑘 1 ) − 𝑦𝑖 (𝑘 0 ) | 𝑋𝑖 ]

(1)

= 𝐸 [𝑦𝑖 | 𝑡𝑖 = 𝑘 1, 𝑋𝑖 ] − 𝐸 [𝑦𝑖 | 𝑡𝑖 = 𝑘 0, 𝑋𝑖 ]
4

(2)

THE PROPOSED METHOD

4.1

Design Rationale

To address the dual challenges identified in Section 1: (1) The performance degradation caused by conventional treatment regularizations (e.g., IPM [21], propensity score prediction [38]), (2)
The underutilization of contextual features in modeling treatment effect heterogeneity, this paper introduces TSCAN (Two-Stage Context-Aware uplift Network), a novel uplift modeling framework comprising two sub-models: CAN-U and CAN-D.
The core insight of TSCAN is to decouple bias mitigation from direct uplift prediction. In Stage 1, CAN-U is trained with IPM and propensity score regularization to generate high-quality pseudouplift labels while reducing selection bias. In Stage 2, CAN-D leverages these labels to perform supervised uplift learning—but crucially without the regularizations that compromise predictive accuracy. Instead, CAN-D employs an isotonic output layer [39] to directly model uplift in an interpretable manner. This two-stage design allows TSCAN to enjoy the benefits of bias correction while avoiding its pitfalls.
In the following, we first present the TSCAN model architecture (including CAN-U and CAND, Section 4.2), detailing how merchant, treatment, and contextual features are fused through attention mechanisms. We then describe the two-stage training procedure (Section 4.3), which enables TSCAN to balance causal robustness and predictive fidelity.
Model Architecture of TSCAN

Context feature

Treatment feature

𝑡

Representation Constraint Module (Only CAN-U)

Multilayer perceptron

Self-Attention "

ℎ"#$

e0 e1 e2 e3 e4 e5 e6 e7 e8

𝜋(𝑡|𝜙)
Treatment isotonic encoding

"

𝐼𝑃𝑀(𝑝!! , 𝑝!"! )
Treatment-aware Attention

Isotonic Output Layer

ℎ!#$ ℎ!

Isotonic Output Layer

Treatment-aware Attention Layer *N

loss

Multilayer perceptron Uplift weight

Isotonic encoding

0 1

0

0

0

0

0

0

0

5

1

1

1

1

1

1

0

0

0 0

7 1

1

1

1

1

1

1

1

0

☉ Uplift weight w0 w1 w2 w3 w4 w5 w6 w7 w8

☉ 𝑦(

=

Shop feature

Context Aware attention Layer

4.2

𝑦"

1 1 0 0 0

0

w0

5

w0+w1+w2+w3+w4+w5

7

w0+w1+w2+w3+w4+w5+w6+w7

Feature encoder

Fig. 1. The network architecture of CAN-U and CAN-D.

4.2.1 Overall Architecture. Both CAN-U and CAN-D adopt the same backbone architecture, shown in Figure 1. The only structural difference lies in the presence of the Representation Constraint Module (active only in CAN-U) and the use of the Isotonic Output Layer (in both models, but trained differently). The data flow is as follows:
, Vol. 1, No. 1, Article . Publication date: February 2025.

6

Hangtao Zhang, Zhe Li, Kairui Zhang

• Feature Encoder: Converts raw features into dense embeddings.
• Context-Aware Attention Layer: Fuses merchant and treatment embeddings with contextual embeddings to produce context-adaptive representations. This module directly addresses the limitation of existing models in their inadequate incorporation of relevant contextual factors [34, 35].
• Representation Constraint Module (CAN-U only): For CAN-U, this module applies IPM loss and propensity score prediction to mitigate selection bias during training.
• Treatment-Aware Attention Network: Further refines the merchant representation by conditioning on the treatment. This final representation encapsulates the context-dependent relationship between the merchant and the treatment.
• Isotonic Output Layer: Directly models uplift effects in a regularization-free and supervised manner.
We now detail each component.
4.2.2 Feature Encoder. This module converts raw merchant, contextual, and treatment features into dense and comparable embeddings. Each merchant instance is represented by a tuple (𝑋, 𝑡, 𝐶, 𝑦), where 𝑋 denotes merchant-specific attributes (e.g., rating and operating hours), 𝑡 is the treatment (binary or continuous), 𝐶 is the external context (e.g., time-of-day, district type and supply–demand ratio), and 𝑦 is the observed outcome (e.g., order count).
Sparse categorical features undergo an embedding table lookup, while continuous features are transformed via a linear layer. The treatment 𝑡 is encoded in the same manner.
( 𝑒𝑘 =

𝐸𝑘cat (𝑥 𝑘 ), w𝑘 𝑥 𝑘 + b𝑘 ,

if 𝑘 ∈ Ωcat if 𝑘 ∈ Ω num

(3)

This yields three embedding vectors: merchant embeddings 𝑒𝑠 , contextual embeddings 𝑒𝑐 , and treatment embedding 𝑒𝑡 . These serve as inputs to the subsequent context-aware interaction layers.
4.2.3 Context-Aware Attention Layer. This layer dynamically modulates merchant and treatment representations according to the external context, enabling the model to adapt its inference to varying operational environments. For instance, a discount may substantially increase order volume during off-peak hours but exhibit minimal effect during lunchtime peaks owing to saturated demand.
To capture such context-dependent heterogeneity, the Context-Aware Attention Layer adaptively reweights merchant and treatment representations based on the current context.
Context Aware Attention Layer

Context Aware Gate Attention Gate Attention Weight

Self Attention

Sigmoid

MLP Layer Context Aware Gate Attention

Context Feature

Shop Feature

Concat Flat & MLP

Context Feature

Shop Feature

Fig. 2. The architecture of Context-Aware Attention Layer and the Context-Aware Gate Attention.
, Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

7

Specifically, for merchant features, the context-aware weight 𝑎𝑠 is computed by concatenating 𝑒𝑠 with a summarized representation of 𝑒𝑐 (via an MLP) and passing it through a sigmoid-activated linear layer. Figure 2 illustrates this process.
 𝑎𝑠 = 1 + 𝜎 W𝑝 [e𝑠 ; MLP(Flat(e𝑐 ))] + b𝑝 ,

(4)

where 𝑎𝑠 acts as a context-gated attention weight, ensuring that context-relevant merchant features are amplified. W𝑝 and b𝑝 are the parameters of the MLP layer, “;” denotes concatenation operator, and 𝜎 refers to the activation function. The adaptive merchant representation is then:
h𝑠 = 𝑎𝑠 ⊙ e𝑠

(5)

Finally, a self-attention operation is performed on [h𝑠 , e𝑐 ] to facilitate richer interactions, producing the final context-aware merchant representation hcal . A similar gating is applied to e𝑡 :
 𝑎𝑡 = 1 + 𝜎 W𝑡 [e𝑡 ; MLP(Flat(e𝑐 ))] + b𝑡 ,

(6)

h𝑡 = 𝑎𝑡 ⊙ e𝑡 .

(7)

This design ensures that the same merchant–treatment pair yields different latent representations under different contexts, enabling fine-grained ITE estimation.
4.2.4 Treatment-Aware Attention Network. This module refines the context-aware merchant representation by explicitly conditioning it on the treatment, thereby modeling the treatment-merchant interaction in a way that is sensitive to the current context.
Treatment Aware Attention layer Concat

Treatment Aware Gate Attention Gate Attention Weight

Self Attention Sigmoid

MLP Layer Treatment Aware Gate Attn

Treatment Feature

Input Feature

Concat

Treatment Feature

Input Feature

Fig. 3. The architecture of Treatment-Aware Attention Layer and the Treatment-Aware Gate Attention.

Aiming to model how treatments interact with merchant characteristics, we employ a treatmentgated attention mechanism, but crucially extend it to be context-conditioned.
Specifically, the merchant representation hcal is weighted by a gate 𝛼 tal which is a function of both hcal and h𝑡 :
 𝛼 tal = 1 + 𝜎 W𝑞 [hcal ; h𝑡 ] + b𝑞 , (8)
where W𝑞 and b𝑞 represent the gate attention parameters. Subsequently, a self-attention operation is performed on [𝛼 tal ⊙ hcal, h𝑡 ] to facilitate richer interactions, yielding the final representation:
htal = Self-Attn( [𝛼 tal ⊙ hcal, h𝑡 ]),

(9)

, Vol. 1, No. 1, Article . Publication date: February 2025.

8

Hangtao Zhang, Zhe Li, Kairui Zhang

where htal captures the contextually adaptive interaction between the merchant and the treatment.
This design allows the same merchant-treatment pair to produce different interaction strengths in different environments. Figure 3 illustrates this process.
4.2.5 Representation Constraint Module (CAN-U Only). This module (exclusively used in CAN-U)
injects dual bias-mitigation regularizations: IPM [21] and adversarial propensity score estimation [38]. Their joint application has been shown to yield more robust distributional alignment than either method in isolation [13, 40].
• IPM [21]: IPM (e.g., MMD) is utilized to minimize the distance between the distributions 𝑡 =0 and ℎ𝑡 =1 , aiming to remove confounding by aligning latent representations [21, 22].
of ℎ𝑐𝑎𝑙 𝑐𝑎𝑙 To extend this methodology to continuous treatments, we sort and split the samples into two parts according to treatment values within each batch. Subsequently, we compute the distribution distance between the upper 50% and lower 50% of these samples. The IPM loss can be formulated as:
 LIPM = sup E𝑥∼𝑝𝑡 ∈𝑇0 [𝑓 (𝑥)] − E𝑥∼𝑝𝑡 ∈𝑇1 [𝑓 (𝑥)]
(10)
| | 𝑓 | | H𝑘 ≤1

= ||𝜇 (𝑝𝑡 ∈𝑇0 ) − 𝜇 (𝑝𝑡 ∈𝑇1 )|| H𝑘

(11)

• Propensity Score Prediction [38]: We estimate the propensity score via an auxiliary network 𝜋 (·). The propensity score regularization accounts for selection bias by ensuring that 𝑓𝜃 (𝑥, 𝑡) remains invariant under perturbations, thereby enhancing the robustness of the uplift model [11]. The propensity score prediction loss can be formulated as:
L𝜋 =

𝑛 ∑︁

2 𝑡𝑖 − 𝜋 (𝑡ˆ𝑖 | 𝜙 (𝑥𝑖 ))

(12)

𝑖=1

While the integration of IPM and propensity-based regularization improves causal identifiability, it introduces a trade-off: enforcing strict distributional balance may inadvertently suppress outcomepredictive features that are correlated with treatment assignment [27, 28, 41]. Such regularizationinduced information loss can impair the fidelity of outcome prediction. Therefore, rather than using this constrained model as the final estimator, we restrict the Representation Constraint Module to Stage 1 (CAN-U) solely for generating reliable pseudo-uplift labels. The final prediction is delegated to CAN-D in Stage 2, which operates without these regularizations and thus preserves full predictive capacity.
4.2.6 Isotonic Output Layer. The Isotonic Output Layer [39] is introduced to enable direct modeling of uplift effects in a supervised manner, allowing the CAN-D to learn from pseudo-uplift labels generated by CAN-U, while being free from the above regularizations. By explicitly parameterizing the incremental effect of each treatment level, this layer facilitates joint supervision on both factual outcomes and uplift increments, thereby enabling CAN-D to inherit causal knowledge from Stage 1 while adaptively correcting its estimation errors through factual outcome reinforcement (a detailed discussion will be provided in Section 4.3).
Concretely, for a treatment value 𝑡, 𝑡 ∈ [0, 1], we discretize it into 𝑀 + 1 (M=1 for binary treatment) ordered levels and apply isotonic encoding:
IE(𝑡𝑖 ) = [1, . . . , 1, 0, . . . , 0], | {z } | {z } 𝑘+1

where 𝑘 = ⌊𝑡𝑖 · 𝑀⌋

(13)

𝑀 −𝑘

The model then predicts an uplift weight vector w𝑖 = [𝑣𝑖,0, 𝑣𝑖,1, . . . , 𝑣𝑖,𝑀 ] ⊤ , where each 𝑣𝑖,𝑘 ≥ 0 represents the marginal uplift contributed by the 𝑘-th treatment level. The predicted factual outcome , Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

9

under treatment 𝑡 𝑓 is:
𝑑 𝑦ˆ𝑖,𝑓 =

𝑘𝑓 ∑︁

with 𝑘 𝑓 = ⌊𝑡 𝑓 · 𝑀⌋

𝑣𝑖,𝑘 ,

(14)

𝑘=0

The uplift from 𝑡 𝑓 to a counterfactual 𝑡𝑐 𝑓 > 𝑡 𝑓 is directly computed as the sum over the incremental segment:
𝑘𝑐 𝑓 ∑︁ 𝑢ˆ𝑖𝑑 = 𝑣𝑖,𝑘 (15)
𝑘=𝑘 𝑓 +1

This formulation enables CAN-D to be trained with a dual-loss objective (loss of 𝑦ˆ𝑑𝑓 and 𝑢ˆ𝑑 ), as demonstrated in Figure 4. By explicitly modeling the incremental effect of treatment levels, the Isotonic Output Layer allows CAN-D to effectively learn from both factual and counterfactual signals, thereby enhancing its uplift estimation accuracy.
𝑦"!"

𝑡!
1 1 0 0 0

𝐿 𝑦! , 𝑦"!"

⦿

Multilayer perceptron Uplift weight 1 1 1 0 0 𝑡"!
Counter factual

0 0 1 0 0

⦿

𝑢""

𝐿 𝑢, 𝑢""

∆𝑡 = 𝑡"! − 𝑡!

Fig. 4. An illustration of prediction process for the factual outcome, counterfactual outcome, and the corresponding uplift.

This incremental learning strategy decouples factual and counterfactual learning, reducing error propagation—since 𝑦 𝑓 is observed, its prediction is well-constrained, while 𝑢˜ only needs to model the difference, which is typically lower-variance than the full counterfactual outcome [42, 43].
4.3

Two-Stage Training Process

The two-stage training process is proposed to solve the regularization-induced information loss demonstrated in Section 4.2.5, where bias correction and final prediction are decoupled into two specialized stages, rather than compromising between these competing objectives within a single model (Figure 5). Stage 1 (CAN-U) focuses exclusively on generating unbiased pseudo-uplift labels through rigorous bias correction, while Stage 2 (CAN-D) leverages these labels to optimize predictive performance without regularization constraints.
𝑁 Stage 1: In this stage, the CAN-U model is trained on the observed dataset 𝐷 = {(𝑥𝑖 , 𝑡𝑖 , 𝑦𝑖 )}𝑖=1 , incorporating the Representation Constraint Module to mitigate selection bias via dual regularization mechanisms [13]. The overall training objective is formulated as a minimax optimization problem that jointly accounts for factual outcome prediction, distributional balance, and adversarial propensity score regularization:
L𝜃 = L (𝑦ˆ𝑖 , 𝑦𝑖 ) + 𝛼 LIPM + 𝛽R (ℎ)

(16)

min max (L𝜃 − 𝜆L𝜋 ) ,

(17)

𝜃

𝜋

, Vol. 1, No. 1, Article . Publication date: February 2025.

10

Hangtao Zhang, Zhe Li, Kairui Zhang

Stage 1

Stage 2

Loss(𝑦"" )

Loss(𝑦""# )+Loss(𝑢 " #)

𝑦"!"

CAN-U Training

Factual sample

Counterfactual Sampling

Sample selection

CAN-D

+

Prediction

Supplementary features

𝑢) = |𝑦"!" − 𝑦"" | {𝑋, 𝑡" , 𝑦" }

Counterfactual Sampling

4.3 4.4 4.5 4.6 4.7

Counterfactual sample {𝑋, 𝑡 !" }

{𝑋, 𝑦" , 𝑡" }

Generated dataset {𝑋, 𝑡" , 𝑦" , 𝑡 !" ,𝑢) }

shop_rating = 4.5 Training

Factual sample

Prediction

Fig. 5. Diagram illustrating TSCAN’s two-stage training process and counterfactual sampling diagram, where the black solid line represents the training flow and the blue dotted line represents the prediction flow.

where L (𝑦ˆ𝑖 , 𝑦𝑖 ) denotes the factual outcome prediction loss, LIPM is the Integral Probability Metric loss enforcing distributional balance between treatment groups, L𝜋 is the propensity score estimation loss, and R (ℎ) represents ℓ2 regularization. 𝜆, 𝛼 and 𝛽 are hyperparameters that control the relative importance of the adversarial propensity task, the IPM loss, and the ℓ2 regularization, respectively.
The adversarial training procedure follows a bilevel minimax strategy [44], which jointly optimizes factual outcome prediction, enforces distributional balance between treatment groups via IPM, and adversarially suppresses treatment-predictive information in the latent representation through propensity score estimation. Consequently, the learned representation exhibits improved alignment across treatment groups in the latent space, enhancing the robustness of individual treatment effect estimation [11, 44].
Stage 2: Given a factual observation (𝑋, 𝑡 𝑓 , 𝑦 𝑓 ), we construct a corresponding counterfactual instance (𝑋, 𝑡𝑐 𝑓 ) by selecting an alternative treatment assignment 𝑡𝑐 𝑓 ≠ 𝑡 𝑓 . The counterfactual treatment 𝑡𝑐 𝑓 is sampled either uniformly at random from the support of the treatment variable or according to a predefined probabilistic strategy (e.g., based on empirical treatment distribution or domain heuristics). Then, the pseudo-uplift label is derived as 𝑢˜ = 𝑦ˆ (𝑋, 𝑡𝑐 𝑓 ) − 𝑦ˆ (𝑋, 𝑡 𝑓 ) for each observation, where 𝑦ˆ (𝑋, 𝑡) denotes the outcome predicted by the CAN-U model trained in Stage 1. This difference-based formulation aligns with theoretical results in causal inference [45, 46], which demonstrate that estimating treatment effects is statistically more efficient than estimating potential outcomes directly, particularly in high-dimensional settings. The resulting complete and unbiased dataset 𝐷˜ = (𝑋, 𝑡 𝑓 , 𝑦 𝑓 , 𝑡𝑐 𝑓 , ˜ 𝑢) enables CAN-D to be trained via a dual-loss objective:
L𝑑 = L (𝑦 𝑓 , 𝑦ˆ𝑑𝑓 ) + 𝛾 L ( ˜ 𝑢, 𝑢ˆ𝑑 )

(18)

where the first term reinforces factual outcome prediction accuracy, while the second term transfers causal knowledge from CAN-U. The weight parameter 𝛾 balances these objectives.
This two-stage architecture provides three critical advantages over conventional approaches:
(1) Bias-Variance Trade-off Optimization: By separating bias correction from final prediction, we avoid the regularization-induced variance inflation documented in [27, 28]. CAN-D focuses exclusively on minimizing prediction error without constraints.
, Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

11

(2) Error Correction Mechanism: The factual outcome reinforcement term L (𝑦 𝑓 , 𝑦ˆ𝑑𝑓 ) enables CAN-D to correct systematic errors in predictions from CAN-U. Empirical validation is provided in Section 5.
(3) Contextual Adaptation: Without regularization constraints, CAN-D’s context-aware attention layers can fully exploit contextual information that might have been suppressed by IPM regularization in Stage 1. This addresses the contextual underutilization problem highlighted in recent causal literature [34].
5

EMPIRICAL EVALUATIONS

In this section, we design and conduct a series of comprehensive experiments to address the following three research questions: RQ1: How does the proposed TSCAN model perform compared to the baselines? RQ2: What is the contribution of each component in the model? RQ3: How does the TSCAN model perform in real-world online scenarios?
5.1

Experimental Setup

5.1.1 Datasets. We conduct experiments on two real-world datasets sourced from Taobao Shangou (previously called Ele.me), one of China’s largest online food ordering platforms:
• Eleshop-1M: This dataset contains online data of 1 million catering merchants. The treatment variable is continuous, defined as the merchant’s average shop rating. The outcome is the total order count over a fixed period. Merchant features include operational attributes, such as shop ratings, operating hours, number of dishes and number of reviews. Contextual features capture the external market environment, such as business district type, regional user attributes, and the local supply-demand status at the time of observation.
• Shop Activities: This dataset contains data from 700k merchants. The treatment variable is binary, indicating whether a merchant participated in a specific “new customer coupons” marketing activity. The outcome is the total order count. Merchant features include menu categories, store exposure, and prices of main dishes. Contextual features include business district type, user type, time-period, average discount in the business district, and supplydemand status. We provide a detailed schema and summary statistics for both the Eleshop-1M and Shop Activities datasets in Table 1.
Table 1. Schema and summary statistics for the Eleshop-1M and Shop Activities datasets.

Attribute Task Type Total Samples Train / Test Split Merchant Features Context Features Total Input Features Treatment Variable Outcome Variable

Eleshop-1M

Shop Activities

Continuous Treatment 1,000,000 800k / 200k

Binary Treatment 700,000 500k / 200k

42 19 61

48 23 71

average customer rating participation in a new customer activity order count (continuous)
order count (continuous)

5.1.2 Evaluation Metrics. We evaluate the performance of TSCAN and other baseline models using two widely-recognized metrics (QINI and AUUC) and two contextualized derived metrics (CAUUC , Vol. 1, No. 1, Article . Publication date: February 2025.

12

Hangtao Zhang, Zhe Li, Kairui Zhang

and CQINI). In addition, we include a key visualization tool (Gain Curve) to enable a more intuitive comparison of model performance.
• Normalized QINI and AUUC: QINI evaluates the effectiveness of uplift models in distinguishing between subsets of a population that respond differently to a treatment [47]. AUUC provides a standardized performance measure that reflects the model’s ability to accurately identify and segment the population based on their uplift potential [48].
• Context-wise AUUC (CAUUC) and Context-wise QINI (CQINI): Standard AUUC and QINI provide global performance summaries but may obscure significant variations in model behavior across different operational contexts (e.g., high vs low supply–demand environments). When evaluating AUUC, uplift values from different contexts are mixed together, making it impossible to evaluate the ranking effect of the same context accurately. Therefore, to rigorously evaluate a model’s ability to capture context-dependent heterogeneity in treatment effects, this paper proposes two context-stratified variants: CAUUC and CQINI.
These metrics compute a weighted average of AUUC/QINI scores across merchant subgroups defined by distinct contextual conditions, thereby offering a more granular assessment of contextual sensitivity and robustness.
Specifically, CAUUC is the weighted average of AUUC for merchant groups across different contexts:
Í𝐺 𝑔=1 𝑁𝑔 · AUUC𝑔 CAUUC = (19)
Í𝐺 𝑔=1 𝑁𝑔 where 𝐺 is the number of merchant groups stratified by different contexts, 𝑁𝑔 is the sample count of group 𝑔, and AUUC𝑔 is the AUUC value of group 𝑔.
CQINI is the weighted average of QINI for merchant groups across different contexts:
Í𝐺 𝑔=1 𝑁𝑔 · QINI𝑔 CQINI = (20)
Í𝐺 𝑔=1 𝑁𝑔 where QINI𝑔 is the QINI value of group 𝑔. In these experiments, contextual groups are defined by business district type and time-period. For continuous treatments, we discretize them into multiple intervals and compute the average AUUC and average QINI for each interval.
• Gain Curve: The Gain Curve is a visualization tool used to evaluate the effectiveness of uplift models. It plots the cumulative uplift against the proportion of the population targeted, ranked by predicted uplift scores. A steeper Gain Curve indicates superior model performance, as it shows that the model can achieve higher uplift by targeting a smaller subset of the population.
5.1.3 Baselines and Parameter Settings. To comprehensively evaluate the proposed TSCAN framework, we select representative methods spanning three major categories of uplift modeling approaches: meta-learners, Tree-based methods and Deep learning methods.
• S-Learner [8]: A meta-learner that estimates treatment effects by including treatment as a feature in a single model.
• T-Learner [8]: A meta-learner that trains separate models for treatment and control groups and computes their difference.
• X-Learner [8]: An extension of T-Learner that incorporates propensity scores and crossfitting to reduce bias in imbalanced settings.
• BART [9]: It employs a sum-of-trees approach with regularization to estimate heterogeneous treatment effects.
, Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

13

• Causal Forest [10]: It extends random forests with specific splitting criteria designed for causal inference.
• TarNet [22]: It learns balanced, treatment-invariant representations of covariates by using two separate prediction heads for treatment and control conditions, while sharing a common representation layer;
• DragonNet [17]: This model jointly learns a shared representation of covariates and estimates potential outcomes under treatment and control conditions using a modified architecture inspired by the TarNet framework;
• TransTEE [11]: This approach introduces transformer architecture for uplift modeling with explicit treatment representation;
• EFIN [14]: It designs a treatment-gated feature interaction network to capture heterogeneous treatment effects;
• CFR-ISW [13]: It combines representation balancing with importance sampling weighting;
• DESCN [49]: This model captures the integrated information of treatment and response through a cross network in a multi-task learning manner, and it employs an intermediate pseudo treatment effect prediction network to relieve sample imbalance.
In Stage 1, CAN-U is trained with early stopping using Adam optimizer (learning rate=0.015, 𝛽 1 =0.9, 𝛽 2 =0.999). The regularization weights are set as 𝜆=0.5 (adversarial task), 𝛼 =0.01 (IPM loss), and 𝛽 =1e-5 (ℓ2 regularization). Subsequently, for each sample in the dataset, we generate its corresponding counterfactual counterpart. Then the trained CAN-U model is applied to these original–counterfactual outcome pairs to estimate the uplift label for each instance. In Stage 2, CAN-D is trained with early stopping using the same optimizer settings on the dataset constructed from Stage 1. The uplift prediction loss weight 𝛾 is set to 0.6 through grid search on the validation set.
Since DragonNet, TarNet, and DESCN do not natively support continuous treatments, we extend their architectures to handle continuous treatments on the Eleshop-1M dataset by replacing the dual-head output layer with 𝐾 = 5 parallel heads, following the approach of [11, 14]. Since XLearner is difficult to extend, we do not report its results on this dataset. All models are implemented using Python 3.8 and PyTorch. We employ the Maximum Mean Discrepancy (MMD) as the IPM loss. All experiments are repeated five times, and the results are averaged.
5.2

Overall Performance Assessment

5.2.1 RQ1: How does the proposed TSCAN model perform compared to the baseline models? The performance of TSCAN compared with the baseline models on the datasets Eleshop-1M and Shop Activities is shown in Table 2. From the results, we can draw the following conclusions:
(1) TSCAN achieves the best performance on both the Eleshop-1M and Shop Activities datasets.
On the Eleshop-1M dataset with continuous treatments, TSCAN surpasses the best baseline (TransTEE) by 0.0049 in AUUC and 0.0153 in CAUUC. On the Shop Activities dataset with binary treatments, TSCAN outperforms DESCN by +0.0033 in AUUC and +0.0080 in QINI. This consistent superiority across different treatment types validates TSCAN’s flexibility and robustness. The performance gains on context-aware metrics (CAUUC/CQINI) confirm that the Context-Aware Attention Layer effectively captures context-dependent treatment effects. This advantage is amplified on the Eleshop-1M dataset with continuous treatments, highlighting the benefit of the isotonic output layer for modeling incremental treatment effects.
(2) Deep learning models generally outperform traditional meta-learners and tree-based approaches, validating the crucial role of complex feature interactions in merchant diagnosis scenarios.
, Vol. 1, No. 1, Article . Publication date: February 2025.

14

Hangtao Zhang, Zhe Li, Kairui Zhang

Notably, within the deep learning category, models explicitly designed for treatment effect estimation (TransTEE, CFR-ISW and EFIN) achieve higher performance than S-Learner and T-Learner, confirming that specialized causal architectures yield more accurate uplift estimates. However, TSCAN’s two-stage training strategy provides a significant advantage over all these approaches, as it avoids the information loss caused by regularization-induced bias while maintaining causal robustness.
(3) Significant performance differences exist between models when handling different treatment types. For continuous treatments (Eleshop-1M dataset), TransTEE demonstrates stronger performance among baselines (AUUC=0.7489), leveraging its transformer architecture to model dose-response relationships effectively. For binary treatments (Shop Activities), DESCN achieves competitive results (QINI=0.0974). DragonNet performs consistently well across both treatment types (AUUC=0.7202 for continuous and 0.6301 for binary), demonstrating the effectiveness of its adversarial balancing approach. By contrast, the performance of tree-based models varies significantly depending on the treatment type, suggesting they are particularly well-suited to binary treatments.
TSCAN, however, maintains strong performance across both treatment types, demonstrating the versatility of the proposed architecture.
To intuitively compare the cumulative uplift effects across different models, the Gain Curves of various approaches on the benchmarks are demonstrated in Figure 6. In Gain Curve, a steeper trajectory reaching higher uplift faster indicates better model performance. The ideal curve approaches the upper-left corner, while the diagonal (black dotted curve) represents random ordering.
As shown in Figure 6, TSCAN (red curve) exhibits the steepest ascent among the compared models, confirming its enhanced capability to identify high-uplift merchants through effective causal effect prioritization.
Table 2. Model performance comparison on the two datasets

Dataset Metrics

CQINI

Eleshop-1M QINI CAUUC

S-Learner T-Learner X-Learner BART Causal Forest TarNet DragonNet TransTEE EFIN CFR-ISW DESCN TSCAN (ours)

0.1708 0.1836 0.2026 0.2163 – – 0.1236 0.1615 0.1887 0.2014 0.2323 0.2248 0.2470 0.2393 0.2652 0.2608 0.2344 0.2267 0.2536 0.2489 0.2581 0.2524 0.2839 0.2687

AUUC

CQINI

Shop Activities QINI CAUUC

0.6442 0.6649 0.0874 0.0828 0.6751 0.6871 0.0881 0.0842 – – 0.0890 0.0886 0.6242 0.6420 0.0914 0.0861 0.6630 0.6785 0.0889 0.0873 0.7142 0.7079 0.0936 0.0969 0.7266 0.7202 0.0930 0.0953 0.7533 0.7489 0.0919 0.0935 0.7162 0.7107 0.0828 0.0837 0.7434 0.7388 0.0906 0.0925 0.7489 0.7442 0.0938 0.0974 0.7686 0.7538 0.0994 0.1054

AUUC

0.5819 0.5800 0.6106 0.5955 0.6164 0.6068 0.6196 0.6096 0.6142 0.6030 0.6335 0.6283 0.6329 0.6301 0.6270 0.6256 0.5947 0.5949 0.6217 0.6167 0.6325 0.6295 0.6379 0.6328

5.2.2 RQ2: What is the contribution of each component in the model? We conduct ablation studies to quantify the contribution of four core components in TSCAN:
• Two-stage training strategy: To isolate the effect of decoupled bias correction, we compare the full TSCAN model (CAN-D submodel trained in Stage 2) against CAN-U (the Stage 1 output).
, Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis S-Learner T-Learner BART CausalForest DragonNet TarNet TrasTEE EFIN CFR DESCN TSCAN Random

1.0

1.0

0.8

0.8

Gain

S-Learner T-Learner BART CausalForest DragonNet TarNet TrasTEE EFIN CFR DESCN TSCAN Random

0.4

0.2

0.0 0.0

0.2

0.4

0.6

Cumulative Target Population

0.8

(a) Gain curve on the Eleshop-1M.

1.0

0.6

Gain

0.6

15

0.4

0.2

0.0 0.0

0.2

0.4

0.6

Cumulative Target Population

0.8

1.0

(b) Gain curve on the Shop Activities.

Fig. 6. Gain curves for uplift prediction on benchmark datasets.

• Context-Aware Attention layer: To evaluate whether explicit context-merchant interaction modeling outperforms naive feature fusion, we create TSCAN-RA by replacing the ContextAware Attention layer in both submodels with a standard fully-connected layer that treats context features as ordinary merchant features.
• Contextual features: To measure the necessity of contextual information for treatment effect estimation, we construct TSCAN-RC by removing all contextual features from the input.
• Isotonic output layer: To assess the impact of isotonic output layer on uplift estimation, we develop TSCAN-RISO by replacing the isotonic output layer in CAN-D with a standard fully-connected layer, thereby reducing the dual-loss objective to outcome prediction loss.
The experimental results are presented in Table 3. Based on the experimental results, the following conclusions can be established:
(1) The two-stage training strategy improves performance: CAN-D (full TSCAN) outperforms CAN-U on both datasets, with relative improvements of up to 5.95% in QINI and 1.45% in AUUC on the Eleshop-1M dataset. This validates the hypothesis that decoupling bias correction from prediction avoids the information loss inherent in single-stage regularization approaches. The performance gap between CAN-U and CAN-D is larger on CAUUC and CQINI metrics (CAUUC difference of 0.0212 vs. AUUC difference of 0.0108), indicating that removing regularization constraints particularly benefits context-sensitive estimation.
(2) Contextual information is crucial: removing contextual features (TSCAN-RC) causes performance degradation (AUUC drops by 0.0856 on Eleshop-1M), confirming the hypothesis that treatment effects are highly context-dependent. Second, the Context-Aware Attention Layer provides significant gains over treating context as ordinary features (TSCAN-RA vs. TSCAN), particularly on context-stratified metrics (CAUUC improves by 0.0306 on Eleshop-1M), demonstrating its effectiveness in modeling merchant-treatment-context interactions.
(3) The isotonic output layer is critical for accurate uplift estimation. TSCAN-RISO shows performance degradation compared to full TSCAN (AUUC drops by 0.0162), particularly on continuous treatment tasks. By modeling the uplift value directly, CAN-D is able to focus on the supervised learning of uplift labels while maintaining the accuracy of factual outcome prediction.
To investigate the influence of the uplift prediction loss weight 𝛾 on the CAN-D model, we conduct a series of comparative experiments using different values of 𝛾, with AUUC and CAUUC serving as the primary evaluation metrics. The results are presented in Figure 7. As 𝛾 varies from 0.5 , Vol. 1, No. 1, Article . Publication date: February 2025.

16

Hangtao Zhang, Zhe Li, Kairui Zhang

Table 3. Model performance comparison of TSCAN-RC, TSCAN-RA, TSCAN-RISO, CAN-U and TSCAN (the CAN-D sub-model)

Dataset Metrics

CQINI

Eleshop-1M QINI CAUUC

TSCAN-RC TSCAN-RA TSCAN-RISO CAN-U TSCAN (CAN-D)

0.1737 0.1847 0.2526 0.2429 0.2602 0.2490 0.2639 0.2536 0.2839 0.2687

AUUC

CQINI

Shop Activities QINI CAUUC

0.6536 0.6682 0.0865 0.0812 0.7380 0.7334 0.0939 0.0972 0.7452 0.7376 0.0933 0.0969 0.7474 0.7430 0.0941 0.0977 0.7686 0.7538 0.0994 0.1054

AUUC

0.5812 0.5735 0.6234 0.6302 0.6227 0.6296 0.6283 0.6309 0.6379 0.6328

to 0.8, both AUUC and CAUUC initially increase and subsequently decline, indicating the sensitivity of model performance to this hyperparameter. The optimal value is found to be approximately 𝛾=0.6, yielding an AUUC of 0.6328 and a CAUUC of 0.6379. For reference, the black dashed line denotes the performance of the CAN-U baseline (AUUC: 0.6309; CAUUC: 0.6283). At 𝛾=0.6, CAN-D achieves a relative improvement of 0.0019 (0.3%) in AUUC and 0.0096 (1.5%) in CAUUC over CAN-U.
These results demonstrate that the proposed two-stage training strategy effectively enhances the model’s counterfactual prediction accuracy and its capacity to leverage contextual information.

0.638

0.6328

0.6364

0.633

0.6314

0.6315

0.6307 0.6276

0.628

0.623

CAUUC

AUUC

0.633

0.6379

0.638

0.6297 0.628

0.623

0.618 0.5

0.6

CAN-U baseline

γ

0.7

0.8

CAN-D with γ

0.618 0.5

0.6

γ CAN-U baseline

0.7

0.8

CAN-D with γ

Fig. 7. Performance of CAN-D under different values of the uplift prediction loss weight 𝛾

5.2.3 RQ3: How does the TSCAN model perform in real-world online scenarios? To evaluate the performance of TSCAN in real-world online scenarios, we deployed TSCAN on a real merchant diagnosis system of an online food ordering platform in China. In this application, we estimate the increase in order volume (the outcome) after merchants adopt specific suggestions, such as configuring discount vouchers for new customers and setting up shop posters. The A/B test compares TSCAN against BART (the previously deployed model) across 90,000 merchants randomly assigned to treatment groups. Each merchant received personalized diagnostic suggestions ranked by predicted uplift. As shown in Table 4, in the online experiment, TSCAN outperforms the baseline model BART, achieving an AUUC improvement of 0.0349, a CAUUC improvement of 0.0411 and a 0.76% increase in order volume (95% CI [0.68%, 0.84%], p=0.001). Manual analysis of high-performing market segments reveals that TSCAN effectively identifies context-specific opportunities: during peak demand periods in business districts, it correctly down-weights discount suggestions and upweights exposure-boosting suggestions; conversely, during low demand periods in residential areas, , Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

17

it recommends targeted discounts. This adaptive behavior validates the design of the context-aware attention layer.
Table 4. Online performance comparison of TSCAN and baseline model BART

Metrics Base (BART)
TSCAN

6

CAUUC

AUUC

Order increase

0.6331 0.6742

0.6370 0.6719

0.00% +0.76%

Conclusion

In this paper, we introduce TSCAN, a Context-Aware uplift model based on a two-stage training approach. TSCAN effectively mitigates the negative impacts of traditional regularization methods such as IPM loss and propensity score prediction by employing a two-stage training strategy with CAN-U and CAN-D models. Additionally, by integrating a Context-Aware attention layer, TSCAN leverages contextual features to enhance the accuracy of treatment effect estimation across diverse environments. Through extensive experiments on two large-scale real-world datasets and a live deployment on one of China’s largest food ordering platforms, we demonstrate that TSCAN achieves consistent improvements across multiple evaluation metrics. Despite its empirical success, our approach has several limitations. First, the two-stage training process increases computational complexity and training time compared to single-stage models. Second, while our experiments span both continuous and binary treatments, the datasets originate from the online food ordering industry, which may limit the generalizability of the proposed findings to substantially different domain contexts or treatment types. Finally, the isotonic output layer assumes monotonic treatment effects within discretized intervals, which may not hold for treatments with non-monotonic doseresponse relationships.
There are several potential future works for exploration. We will investigate distillation techniques to compress the two-stage architecture into an end-to-end model without significant performance degradation. Second, we plan to develop unsupervised clustering techniques to automatically identify latent contextual segments from time-series merchant behavior data. Finally, we will validate TSCAN across diverse e-commerce domains to assess its transferability and robustness. These improvements will further bridge the gap between causal machine learning theory and practical business applications.
References [1] David Curry. 2025.
Food Delivery App Revenue and Usage Statistics (2025).
Website.
https://www.businessofapps.com/data/food-delivery-app-market/..
[2] Shanqi Zhang, Hui Luan, Feng Zhen, Yu Kong, and Guangliang Xi. 2023. Does online food delivery improve the equity of food accessibility? A case study of Nanjing, China. Journal of Transport Geography 107 (2023), 103516.
doi:10.1016/j.jtrangeo.2022.103516 [3] Xu Ji, Xuerong Li, and Shouyang Wang. 2024. Balance between profit and fairness: Regulation of online food delivery OFD platforms. International Journal of Production Economics 269 (2024), 109144. doi:10.1016/j.ijpe.2024.109144 [4] Liuyi Yao, Zhixuan Chu, Sheng Li, Yaliang Li, Jing Gao, and Aidong Zhang. 2021. A Survey on Causal Inference. ACM Trans. Knowl. Discov. Data 15, 5, Article 74 (May 2021), 46 pages. doi:10.1145/3444944 [5] Weijia Zhang, Jiuyong Li, and Lin Liu. 2021. A unified survey of treatment effect heterogeneity modeling and uplift modeling. arXiv:2007.12769 [stat.ME] https://arxiv.org/abs/2007.12769 [6] Liuyi Yao, Zhixuan Chu, Sheng Li, Yaliang Li, Jing Gao, and Aidong Zhang. 2021. A Survey on Causal Inference. ACM Trans. Knowl. Discov. Data 15, 5, Article 74 (May 2021), 46 pages. doi:10.1145/3444944 , Vol. 1, No. 1, Article . Publication date: February 2025.

18

Hangtao Zhang, Zhe Li, Kairui Zhang

[7] Zongyu Li, Zheng Hua Zhu, Xiaoning Guo, Shuai Zheng, Zhenyu Guo, Siwei Qiang, and Yao Zhao. 2022. A survey of deep causal models and their industrial applications. Artif. Intell. Rev. 57 (2022), 298. https://api.semanticscholar.org/ CorpusID:253523500 [8] Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. 2019. Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences 116, 10 (Feb. 2019), 4156–4165.
doi:10.1073/pnas.1804597116 [9] Hugh A. Chipman, Edward I. George, and Robert E. McCulloch. 2010. BART: Bayesian additive regression trees. The Annals of Applied Statistics 4, 1 (March 2010). doi:10.1214/09-aoas285 [10] Vikas Ramachandra, Susan Athey, and Stefan Wager. 2015. Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. J. Amer. Statist. Assoc. 113 (10 2015). doi:10.1080/01621459.2017.1319839 [11] Yi-Fan Zhang, Hanlin Zhang, Zachary Chase Lipton, Li Erran Li, and Eric P. Xing. 2022. Exploring Transformer Backbones for Heterogeneous Treatment Effect Estimation. Trans. Mach. Learn. Res. 2023 (2022). https:
//api.semanticscholar.org/CorpusID:249151761 [12] Christos Louizos, Uri Shalit, Joris M. Mooij, David A. Sontag, Richard S. Zemel, and Max Welling. 2017. Causal Effect Inference with Deep Latent-Variable Models. In Neural Information Processing Systems. https://api.semanticscholar.
org/CorpusID:260564 [13] Negar Hassanpour and Russell Greiner. 2019. CounterFactual Regression with Importance Sampling Weights. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19. International Joint Conferences on Artificial Intelligence Organization, 5880–5887. doi:10.24963/ijcai.2019/815 [14] Dugang Liu, Xing Tang, Han Gao, Fuyuan Lyu, and Xiuqiang He. 2023. Explicit Feature Interaction-aware Uplift Network for Online Marketing. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Long Beach, CA, USA) (KDD ’23). Association for Computing Machinery, New York, NY, USA, 4507–4515.
doi:10.1145/3580305.3599820 [15] Ruoqi Liu, Changchang Yin, and Ping Zhang. 2020. Estimating Individual Treatment Effects with Time-Varying Confounders. In 2020 IEEE International Conference on Data Mining ICDM. 382–391. doi:10.1109/ICDM50108.2020.00047 [16] Fredrik D. Johansson, Uri Shalit, and David Sontag. 2016. Learning representations for counterfactual inference. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48 (New York, NY, USA) (ICML’16, Vol. 48). JMLR.org, New York, New York, USA, 3020–3029.
[17] Claudia Shi, David M. Blei, and Victor Veitch. 2019. Adapting neural networks for the estimation of treatment effects.
Curran Associates Inc., Red Hook, NY, USA.
[18] Changhee Lee, Nicholas Mastronarde, and Mihaela van der Schaar. 2018. Estimation of Individual Treatment Effect in Latent Confounder Models via Adversarial Learning. ArXiv abs/1811.08943 (2018). https://api.semanticscholar.org/ CorpusID:53716914 [19] Jinsung Yoon, James Jordon, and Mihaela van der Schaar. 2018. GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net. https://openreview.net/ forum?id=ByKWUeWA[20] Ioana Bica, James Jordon, and Mihaela van der Schaar. 2020. Estimating the effects of continuous-valued interventions using generative adversarial networks. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 1379, 12 pages.
[21] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. 2012. A kernel two-sample test. J. Mach. Learn. Res. 13, null (March 2012), 723–773.
[22] Uri Shalit, Fredrik D. Johansson, and David Sontag. 2017. Estimating individual treatment effect: generalization bounds and algorithms. In Proceedings of the 34th International Conference on Machine Learning - Volume 70 (Sydney, NSW, Australia) (ICML’17). JMLR.org, 3076–3085.
[23] Insung Kong, Yuha Park, Joonhyuk Jung, Kwonsang Lee, and Yongdai Kim. 2023. Covariate balancing using the integral probability metric for causal inference. 17430–17461 pages. arXiv:2305.13715 [stat.ML] https://arxiv.org/abs/2305.13715 [24] Tobias Hatt and Stefan Feuerriegel. 2021. Estimating Average Treatment Effects via Orthogonal Regularization.
In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (Virtual Event, Queensland, Australia) (CIKM ’21). Association for Computing Machinery, New York, NY, USA, 680–689. doi:10.1145/ 3459637.3482339 [25] Anpeng Wu, Kun Kuang, Junkun Yuan, Bo Li, Pan Zhou, Jianrong Tao, Qiang Zhu, Yueting Zhuang, and Fei Wu.
2020. Learning Decomposed Representation for Counterfactual Inference. ArXiv abs/2006.07040 (2020). https:
//api.semanticscholar.org/CorpusID:219636246 [26] Jing Ma, Ruocheng Guo, Aidong Zhang, and Jundong Li. 2021. Multi-Cause Effect Estimation with Disentangled Confounder Representation. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence,

, Vol. 1, No. 1, Article . Publication date: February 2025.

TSCAN: Context-Aware Uplift Modeling via Two-Stage Training for Online Merchant Business Diagnosis

19

IJCAI-21, Zhi-Hua Zhou (Ed.). International Joint Conferences on Artificial Intelligence Organization, 2790–2796.
doi:10.24963/ijcai.2021/384 Main Track.
[27] Serge Assaad, Shuxi Zeng, Chenyang Tao, Shounak Datta, Nikhil Mehta, Ricardo Henao, Fan Li, and Lawrence Carin. 2020. Counterfactual Representation Learning with Balancing Weights. ArXiv abs/2010.12618 (2020). https:
//api.semanticscholar.org/CorpusID:225067078 [28] Ahmed Alaa and Mihaela van der Schaar. 2018. Limits of Estimating Heterogeneous Treatment Effects: Guidelines for Practical Algorithm Design. In Proceedings of the 35th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 80), Jennifer Dy and Andreas Krause (Eds.). PMLR, 129–138. https://proceedings.mlr.
press/v80/alaa18a.html [29] Fredrik Johansson, Nathan Kallus, Uri Shalit, and David Sontag. 2018. Learning Weighted Representations for Generalization Across Designs. arXiv: Machine Learning (02 2018). doi:10.48550/arXiv.1802.08598 [30] Hal R. Varian. 2016. Causal inference in economics and marketing. Proceedings of the National Academy of Sciences 113, 27 (2016), 7310–7315. doi:10.1073/pnas.1510479113 arXiv:https://www.pnas.org/doi/pdf/10.1073/pnas.1510479113 [31] Andrew Forney and Scott Mueller. 2022. Causal inference in AI education: A primer. Journal of Causal Inference 10, 1 (2022), 141–173. doi:doi:10.1515/jci-2021-0048 [32] Qiang Huang, Jing Ma, Jundong Li, Ruocheng Guo, Huiyan Sun, and Yi Chang. 2023. Modeling Interference for Individual Treatment Effect Estimation from Networked Observational Data. ACM Trans. Knowl. Discov. Data 18, 3, Article 48 (Dec. 2023), 21 pages. doi:10.1145/3628449 [33] Kai Lagemann, Christian Lagemann, Bernd Taschler, and Sach Mukherjee. 2023. Deep learning of causal structures in high dimensions under data limitations. Nature Machine Intelligence 5, 11 (2023), 1306–1316. doi:doi:10.1038/s42256023-00744-z [34] Yinqiu Huang, Shuli Wang, Min Gao, Xue Wei, Changhao Li, Chuan Luo, Yinhua Zhu, Xiong Xiao, and Yi Luo. 2024.
Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing. Companion Proceedings of the ACM Web Conference 2024 (2024). https://api.semanticscholar.org/CorpusID:267499863 [35] Zexu Sun, Qiyu Han, Minqin Zhu, Hao Gong, Dugang Liu, and Chen Ma. 2025. Robust Uplift Modeling with Large-Scale Contexts for Real-time Marketing. Association for Computing Machinery, New York, NY, USA. https:
//doi.org/10.1145/3690624.3709293 [36] Ifra Afzal, Burcu Yilmazel, and Cihan Kaleli. 2024. An Approach for Multi-Context-Aware Multi-Criteria Recommender Systems Based on Deep Learning. IEEE Access 12 (2024), 99936–99948. doi:10.1109/ACCESS.2024.3428630 [37] Yuxiang Wei, Zhaoxin Qiu, Yingjie Li, Yuke Sun, and Xiaoling Li. 2024. Multi-Treatment Multi-Task Uplift Modeling for Enhancing User Growth. arXiv:2408.12803 [cs.LG] https://arxiv.org/abs/2408.12803 [38] Benjamin Y. Andrew, M. Alan Brookhart, Rupert Pearse, Karthik Raghunathan, and Vijay Krishnamoorthy. 2023.
Propensity score methods in observational research: brief review and guide for authors. British Journal of Anaesthesia 131, 5 (2023), 805–809. doi:10.1016/j.bja.2023.06.054 [39] Jiachi Zhao, Hongwen Zhang, Yue Wang, Yiteng Zhai, and Yao Yang. 2024. Deep Isotonic Embedding Network: A flexible Monotonic Neural Network. Neural Networks 171 (2024), 457–465. doi:10.1016/j.neunet.2023.12.026 [40] Nathan Kallus. 2020. DeepMatch: balancing deep covariate representations for causal inference using adversarial training. In Proceedings of the 37th International Conference on Machine Learning (ICML’20). JMLR.org, Article 470, 11 pages.
[41] Siyi Wang, Yiyan Huang, Cheuk Hang Leung, Chaoqun Wang, and Qi Wu. 2026. A two-stage disentangled and balanced representation learning method for counterfactual regression. Information Sciences 730 (2026), 122886.
doi:10.1016/j.ins.2025.122886 [42] Yikun Zhang and Yen-Chi Chen. 2025. Doubly Robust Inference on Causal Derivative Effects for Continuous Treatments.
arXiv:2501.06969 [stat.ME] https://arxiv.org/abs/2501.06969 [43] Dominik Rothenhäusler and Bin Yu. 2020. Incremental causal effects. arXiv:1907.13258 [stat.ME] https://arxiv.org/ abs/1907.13258 [44] Amirreza Kazemi and Martin Ester. 2023. Adversarially Balanced Representation for Continuous Treatment Effect Estimation. ArXiv abs/2312.10570 (2023). https://api.semanticscholar.org/CorpusID:266348271 [45] Guido Imbens and Jeffrey Wooldridge. 2008. Recent Developments in the Econometrics of Program Evaluation. Journal of Economic Literature 47 (09 2008), 5–86. doi:10.3386/w14251 [46] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. 2018. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal 21, 1 (01 2018), C1–C68. doi:10.1111/ectj.12097 arXiv:https://academic.oup.com/ectj/articlepdf/21/1/C1/27684918/ectj00c1.pdf [47] Dmitri Goldenberg, Hugo Manuel Proença, Amit Livne, Felipe Moraes, Javier Albert, and Bracha Shapira. 2025.
Converted Data is All You Need for Causal Optimization of e-Commerce Promotions. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management (Seoul, Republic of Korea) (CIKM ’25). Association

, Vol. 1, No. 1, Article . Publication date: February 2025.

20

Hangtao Zhang, Zhe Li, Kairui Zhang

for Computing Machinery, New York, NY, USA, 5666–5673. doi:10.1145/3746252.3761573 [48] Dugang Liu, Xing Tang, Yang Qiao, Miao Liu, Zexu Sun, Xiuqiang He, and Zhong Ming. 2024. Benchmarking for Deep Uplift Modeling in Online Marketing. ArXiv abs/2406.00335 (2024). https://api.semanticscholar.org/CorpusID:270214925 [49] Kailiang Zhong, Fengtong Xiao, Yan Ren, Yaorong Liang, Wenqing Yao, Xiaofeng Yang, and Ling Cen. 2022. DESCN:
Deep Entire Space Cross Networks for Individual Treatment Effect Estimation. Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining 4612–4620 (Aug. 2022), 4612–4620. doi:10.1145/3534678.3539198

, Vol. 1, No. 1, Article . Publication date: February 2025.


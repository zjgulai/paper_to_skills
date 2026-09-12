<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2608.10182
     paper_id : p2s-2026-0003
     source   : https://arxiv.org/html/2608.10182v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# From Prediction to Incrementality: Causal Optimization for Large-Scale Targeting and Recommendation

DOI: XXXXXXX.XXXXXXXConference: XXX; XXXX; XXXXISBN: XXXXXCCS: Mathematics of computing Probability and statisticsCCS: Computing methodologies Machine learningCCS: Information systems Information retrieval
Changshuai Wei Note: These authors contributed equally Note: Corresponding Author email: chawei@linkedin.com Affiliation: LinkedIn, Seattle, USA , John Bencina email: jbencina@linkedin.com Affiliation: LinkedIn, Sunnyvale, USA , Phuc Nguyen email: honnguyen@linkedin.com Affiliation: LinkedIn, Sunnyvale, USA , Andre Assuncao Silva T Ribeiro email: aribeiro@linkedin.com Affiliation: LinkedIn, Sunnyvale, USA and Benjamin Zelditch email: bzelditch@linkedin.com Affiliation: LinkedIn, Sunnyvale, USA

2026

###### Abstract.

Large-scale targeting and recommendation systems are typically built around predictive scores fed into heuristic or local allocation. When the business goal is incremental impact, as in marketing campaigns, incentives, and notifications, this paradigm systematically misallocates resources toward users who would have acted anyway. We present a decision-centric framework that instead optimizes causal effects under global constraints, aligning three components under a single objective: a causal neural network with a Transformer backbone for individual treatment-effect estimation, a Bayesian neural-bandit layer for uncertainty-aware exploration, and a dual-based large-scale linear-programming layer for constrained allocation. The framework also supports sequential context and multi-outcome, attribute-conditioned scoring through a Transformer encoder and outcome embeddings. We evaluate it with offline simulations on a public bandit dataset, targeted architectural ablations, and an online A/B test on LinkedIn Feed marketing traffic. We also distill production lessons on causal training-data construction and cost and delivery control, which were critical to successful deployment. The end-to-end treatment policy delivered a statistically significant $+7.20\%$ lift in the primary long-term-value metric, demonstrating the feasibility of production-scale causal optimization under business constraints.

###### Keywords:

Causal ML, Transformer, Bandit, Linear Program

## 1. Introduction

Large-scale targeting and recommendation systems are a core component of modern online platforms, supporting applications such as advertising, marketing outreach, notifications, and content recommendation. In practice, these systems are typically built by training predictive models to estimate user response probabilities (such as click-through or conversion likelihood) and then ranking or selecting items based on these predictions. This paradigm has proven effective for optimizing engagement metrics at scale and has become the dominant approach in both industrial deployments and academic research.

However, predictive modeling alone is fundamentally misaligned with the objectives of many real-world targeting problems. In marketing and incentive-driven settings, the goal is not to predict outcomes under historical policies, but to estimate the incremental impact of an intervention relative to a counterfactual baseline. Observed user responses are confounded by prior targeting decisions, exposure mechanisms, and user self-selection, implying that high predicted response does not necessarily correspond to high causal value. Prior work has demonstrated that learning and evaluation based on biased recommendation logs can systematically misestimate the true effect of interventions when deployed for decision-making (Schnabel et al., 2016; Wang et al., 2018).

This challenge has motivated increasing interest in causal machine learning for recommender and targeting systems. A line of work reframes recommendations as treatments and user interactions as potential outcomes, enabling principled correction for exposure bias and confounding (Schnabel et al., 2016; Bonner and Vasile, 2018). More broadly, causal perspectives on recommendation have been surveyed extensively, highlighting the limitations of purely predictive approaches and emphasizing the role of causal inference in reliable decision-making (Gao et al., 2024).

In marketing and targeting applications, uplift modeling and treatment effect estimation explicitly target incrementality by estimating individual-level causal effects rather than response probabilities. Foundational work has studied the theoretical properties of individual treatment effect estimation and proposed representation-learning-based approaches to mitigate confounding bias (Shalit et al., 2017). More recent neural architectures, such as DragonNet, jointly model potential outcomes and treatment assignment, demonstrating improved stability and accuracy in treatment effect estimation (Shi et al., 2019).

Complementary to causal modeling, a rich literature studies counterfactual evaluation and learning from logged bandit feedback. Off-policy evaluation methods enable unbiased estimation of policy value without exhaustive online experimentation and have been extended to large action spaces and slate recommendation settings (Swaminathan and Joachims, 2017; McInerney et al., 2020). While these approaches provide principled tools for evaluation and exploration, they typically focus on estimating or comparing policies rather than optimizing decisions for incremental outcomes.

At the same time, real-world targeting and recommendation systems must operate under complex global constraints, including budgets, capacity limits, frequency caps, and coverage or fairness requirements. To address these challenges, many industrial platforms rely on large-scale constrained optimization, often formulated as linear or mixed-integer programs, to translate model outputs into coordinated decisions. Theoretical foundations for robust and constrained optimization are well established (Bertsimas et al., 2011), and industrial systems have demonstrated the effectiveness of combining learned utility models with constraint-aware optimization to improve system-level objectives (Agarwal et al., 2015; Makhijani et al., 2019; Wei et al., 2024).

Despite their success, existing optimization-based systems almost universally rely on predictive scores or locally estimated uplift values as inputs. When incrementality matters, optimizing constrained decisions using non-causal or locally optimal signals can systematically misallocate resources toward users who would have acted regardless of intervention, thereby reducing true return on investment. Conversely, causal effect estimates alone are insufficient without a principled optimization layer to enforce constraints and optimize system-level objectives.

We present a decision-centric formulation in which incrementality and constraints are addressed jointly: the system optimizes causal effects under global constraints, replacing the more common pattern of optimizing predictive or proxy signals under heuristic allocation. This reframes targeting as choosing users with high incremental impact and as globally coordinated allocation, rather than as ranking high-response users. The three components fit together naturally under this objective: causal modeling defines what is optimized, uncertainty-aware exploration shapes the data on which the causal estimates are learned, and constrained optimization translates those estimates into feasible decisions under shared resources. We instantiate the framework with a Transformer-augmented DragonNet causal head, a Bayesian neural-bandit layer, and a large-scale LP layer. While our primary motivation arises from marketing, the formulation is not domain-specific and extends to other settings that allocate limited intervention capacity under constraints.

Our contributions are: (i) a decision-centric formulation that aligns causal estimation, exploration, and constrained allocation under a single objective; (ii) two DragonNet architectural extensions, a Transformer encoder for marketing-touchpoint sequences and a shared outcome-embedding head, for multi-outcome and attribute-conditioned incremental scoring; and (iii) a productionized pipeline combining the causal head with neural-bandit exploration and a dual-based large-scale LP solver, evaluated on a public bandit dataset and as an end-to-end policy in a large online A/B test on LinkedIn Feed marketing traffic that delivered a $+7.20\%$ lift in the primary KPI ($p=0.041$).

### 1.1. Related Work

Recent work has begun to incorporate treatment effect estimation and uplift modeling directly into recommender systems, moving beyond purely predictive objectives. Chen et al. (Chen et al., 2024) propose using individual treatment effect estimation to guide user interest exploration, demonstrating that causal signals can improve exploration efficiency compared to prediction-based uncertainty. Meng et al. (Meng et al., 2025) introduce a coarse-to-fine dynamic uplift modeling framework for real-time video recommendation, showing that explicit uplift estimation can improve ranking quality at scale. Sun et al. (Sun et al., 2024) study incentive recommendation under budget constraints and propose an end-to-end uplift-based framework to improve cost-effectiveness.

While these approaches demonstrate the practical value of causal and uplift modeling in recommender systems, they primarily focus on local decision-making, such as improving ranking quality or applying uplift-based heuristics under relatively simple constraints. Exploration is typically handled implicitly or through heuristic mechanisms, and optimization is performed at the level of individual users or items rather than as a coordinated system-level problem.

In contrast, our work integrates causal effect estimation, principled exploration, and constrained optimization into a unified decision framework. We combine DragonNet-style causal modeling with Transformer-based sequential representations and a neural bandit layer, enabling uncertainty-aware exploration guided by estimated incremental effects. On the optimization side, we formulate targeting decisions as a large-scale constrained optimization problem and solve it using a dual-based approach that scales to extreme problem sizes. This allows us to coordinate decisions across users, items, and campaigns while enforcing complex global constraints, going beyond ranking-based or locally optimal uplift methods.

## 2. Incremental Optimization Framework

### 2.1. Incremental Modeling

*Figure 1. Incremental model architecture overview. The model extends DragonNet with a transformer for temporal modeling and outcome embeddings for multi-product prediction. The core model utilizes member-level embeddings, dense features, and the marketing interaction sequence. The auxiliary autoencoder model utilizes product-level embeddings corresponding to each modeled outcome.*

Incremental optimization requires estimating user-level treatment effects that quantify the expected lift from a targeting or recommendation action. Under the potential outcomes framework, let $Y(1)$ and $Y(0)$ denote the potential outcomes under treatment ($T=1$) and control ($T=0$), and let $X\in\mathcal{X}$ denote observed covariates. The Conditional Average Treatment Effect (CATE), or Individual Treatment Effect (ITE) at the covariate level, is

$\tau(X)\;=\;\mathbb{E}[Y(1)-Y(0)\mid X]\;=\;\mu_{1}(X)-\mu_{0}(X),$ | (1) | | | |

where $\mu_{t}(X)=\mathbb{E}[Y(t)\mid X]$. Two standard assumptions identify $\tau(X)$ from observational data: unconfoundedness, $\{Y(0),Y(1)\}\perp\!\!\!\perp T\mid X$, and overlap, $0<e(X)<1$ for all $X$, where $e(X)=P(T=1\mid X)$ is the propensity score. Under these, $\mu_{t}(X)=\mathbb{E}[Y\mid X,T=t]$ and $\tau(X)$ becomes point-identified from the observed data distribution.

Various estimators for $\tau(X)$ have been proposed, including S-Learner, T-Learner, X-Learner, and Double Machine Learning (DML) (Künzel et al., 2019; Chernozhukov et al., 2018). These approaches typically require multi-stage modeling pipelines, which introduce operational complexity at our scale of hundreds of millions of members. We instead adopt a modified version of DragonNet (Shi et al., 2019), shown in Figure 1, which jointly estimates $\mu_{1}(X)$, $\mu_{0}(X)$, and $e(X)$ in a single forward pass.

DragonNet extends TARNET (Shalit et al., 2017) by adding a propensity head to a shared-representation architecture: a shared encoder $\Phi$ feeds two outcome heads estimating $\mu_{1}$ and $\mu_{0}$ (with observed-treatment gradient routing), and a third head estimating $e(X)$. The motivation is propensity sufficiency: under unconfoundedness, the propensity score is a balancing score, so any representation $\Phi$ that preserves $e(X)$ retains all information needed for CATE estimation. Outcome-only training tends to discard propensity signal in pursuit of marginal-likelihood fit, biasing $\hat{\tau}=\hat{\mu}_{1}-\hat{\mu}_{0}$ toward the dominant treatment arm. Jointly optimizing the propensity head therefore acts as an architectural regularizer on $\Phi$, and in practice yields markedly more stable CATE estimates (Shi et al., 2019).

We extend DragonNet with three key enhancements for recommender systems. First, we feed sequential user-event features through a transformer layer (Section 3) before they enter $\Phi$, letting the encoder absorb temporal context such as recency and engagement cadence. Second, we extend the outcome heads to a multi-outcome configuration using learned product embeddings, so the model can score newly introduced products without adding a product-specific output head. Third, and more importantly, we couple this causal predictor with the neural bandit in Section 2.2, forming a DragonBandit policy that explores over incremental action values rather than predicted responses.

### 2.2. Neural Bandit Exploration

Reliance on logged data without any mechanism for exploration reinforces feedback loops, leading to “rich-get-richer” dynamics and suboptimal long-term performance (Li et al., 2010; Su et al., 2024; Swaminathan and Joachims, 2015; Nguyen et al., 2026). For causal modeling, exploration is additionally important to improve overlap and reduce dependence on the incumbent logging policy (Schnabel et al., 2016; Kasy and Sautmann, 2021), supporting causal identification where the required assumptions hold (Neal, 2020). At each round, DragonBandit draws posterior incremental scores rather than passing deterministic point estimates to the allocator in Section 2.3. The multi-turn simulation examines this mechanism under feedback-loop bias.

Let $h_{t}$ denote the history and remaining constraint capacity at round $t$, $\mathcal{C}_{t}(h_{t})$ the corresponding feasible set, and $x_{t}^{*}(\tilde{\tau}_{t})$ its sampled-score solution. For nonzero exploration, the sampled logits have a nondegenerate Gaussian distribution; because the sigmoid heads are monotone, every open score ordering has positive probability. Therefore,

$\pi_{\alpha}(i\mid u,h_{t})=\Pr\!\left[x^{*}_{u,i,t}(\tilde{\tau}_{t})=1\mid h_{t}\right]>0$ | (2) | | | |

for every $(u,i)$ contained in some $x_{t}\in\mathcal{C}_{t}(h_{t})$. Thus exploration guarantees positivity conditional on the feasible action.

#### 2.2.1. Neural Thompson sampling via Laplace approximation

We approximate the posterior over network parameters by a Gaussian centered at the MAP solution $\hat{\theta}_{\mathrm{MAP}}$ via the linearized Laplace approximation (LLA) (Su et al., 2024; Daxberger et al., 2021; Foong et al., 2019; Raha et al., ). The LLA computes the Gauss–Newton curvature of the log-likelihood at $\hat{\theta}_{\mathrm{MAP}}$ and propagates it through the network’s local linearization, yielding a closed-form Gaussian on the logit $f_{\theta}(x^{*})$ with mean $f_{\hat{\theta}}(x^{*})$ and variance $\sigma_{0}^{2}+g(x^{*})^{\top}\Omega^{-1}g(x^{*})$, where $g(x^{*})$ is the Jacobian and $\Omega$ is the curvature matrix. Thompson sampling proceeds by drawing a logit sample and applying $\sigma(\cdot)$, so the same supervised checkpoint can be deployed with or without exploration and no retraining is required. Production scale is achieved through lightweight curvature approximations (low-rank, diagonal, or last-layer variants (Nilsen et al., 2022; Zhang et al., 2020; Riquelme et al., 2018)); we use the last-layer variant for our pretrained DragonNet.

We also evaluated Bayes by Backprop (Appendix B.1) offline, but use LLA in production because it adds uncertainty sampling to the existing checkpoint with substantially lower training and serving overhead.

### 2.3. Large-scale Allocation with Constraints

The neural bandit improves decisions across rounds, but each round must still coordinate assignments under a common set of business constraints. Let $x_{u,i,t}$ indicate whether item $i\in\mathcal{I}$ is allocated to user $u\in\mathcal{U}$ at round $t=1,\ldots,T$. Using the sampled incremental objective $\tilde{\tau}_{u,i,t}^{\textrm{obj}}$ from DragonBandit, the horizon objective is

$\displaystyle\mathop{\text{max}}_{x_{u,i,t}}$ $\displaystyle\sum_{t=1}^{T}\sum_{u,i}\tilde{\tau}_{u,i,t}^{\textrm{obj}}x_{u,i,t},$ | (3) | | | | |

$\displaystyle\sum_{u,i}\tau_{u,i,t}^{\textrm{guardrail}_{k}}x_{u,i,t}\leq C_{\text{guardrail}_{k}},\quad\forall\,t,k,$ | | s.t. | | |

$\displaystyle\sum_{i}x_{u,i,t}\leq C_{\text{fcap}},\quad\forall\,u,t,$ | | | | |

$\displaystyle x_{u,i,t}\in\{0,1\},\quad\forall\,u,i,t.$ | | | | |

Here $\tau_{u,i,t}^{y}=f_{t}^{y}(u,i,1)-f_{t}^{y}(u,i,0)$ is the round-specific sampled treatment effect on metric $y$. We use a sequential policy that solves one constrained optimization problem per round using the current posterior sample from DragonBandit. After each round, DragonBandit is updated using the observed feedback before producing the posterior sample for the next round. In the absence of shared allocation constraints, the resulting policy is equivalent to Thompson sampling: for each user, it selects the feasible action that maximizes the sampled incremental reward.

#### 2.3.1. Scalability via Dual Decomposition

At production scale, each round contains tens of millions of users and hundreds of items, so its batch has $|\mathcal{U}|\times|\mathcal{I}|$ variables and is intractable for general-purpose solvers. Suppressing the round index, we relax $x_{u,i,t}$ to represent an action probability and solve the resulting large-scale problem using a smoothed dual-decomposition method (Basu et al., 2020). The method adds a small ridge perturbation $\frac{\gamma}{2}\|x\|^{2}$ to the primal and dualizes the $K$ global constraints with multipliers $\lambda\geq 0$; by Danskin’s theorem, the resulting dual $g_{\gamma}(\lambda)$ is differentiable with Lipschitz gradient $\nabla g_{\gamma}(\lambda)=A\,x_{\gamma}^{*}(\lambda)-b$, where the primal minimizer decomposes per user:

$x_{\gamma,u}^{*}(\lambda)=\Pi_{\mathcal{C}_{u}}\!\left[-\tfrac{1}{\gamma}\bigl(A_{u}^{T}\lambda+c_{u}\bigr)\right],$ | (4) | | | |

with $\Pi_{\mathcal{C}_{u}}$ an $O(|\mathcal{I}|\log|\mathcal{I}|)$ projection onto the per-user frequency-cap polytope. The solver then maximizes $g_{\gamma}$ over the $K$-dimensional dual with Nesterov-accelerated ascent, giving per-iteration cost linear in $|\mathcal{U}|\cdot|\mathcal{I}|$ versus $O((|\mathcal{U}||\mathcal{I}|)^{3.5})$ for interior-point methods.

The regularization $\gamma$ is picked as the largest value satisfying $\frac{\gamma\,\hat{x}^{T}\hat{x}}{2\,|c^{T}\hat{x}|}<10^{-3}$, so the ridge perturbation contributes $<0.1\%$ of the objective and the perturbed optimum is practically indistinguishable from the true LP optimum; when $A$ is ill-conditioned across constraint scales, we apply Jacobi row preconditioning. In steady state, we warm-start the dual from the previous period’s $\lambda^{*}$, which under stable input distributions (KS-tested) achieves over 99% of the current optimum and also serves as an SLA fallback when the solver does not converge in time.

## 3. Incrementality Transformer Architecture

### 3.1. Temporal Features

We construct a touchpoint sequence $S=(s_{1},\ldots,s_{L})$ over a lookback window, where each $s_{i}$ encodes a (channel, action) interaction such as (AD, IMPRESSION) or (EMAIL, OPEN); we also inject prior conversion events into the sequence. Each token is mapped through a learned embedding matrix $E_{T}\in\mathbb{R}^{V\times d}$, with $V$ interaction types plus a leading [CLS] token (BERT-style (Devlin et al., 2019)) that summarizes the sequence even when empty. We use the temporal encoding described below.

##### Temporal position.

Member interactions occur on irregular schedules, so each token combines its interaction embedding with a day-since embedding $E_{DAYS}$, day-of-week embedding $E_{DOW}$, and the small-embedding-friendly sinusoidal positional encoding $PE_{i}$ of Foumani et al. (2023):

$E_{i}=E_{T}(s_{i})+E_{DAYS}(t_{i})+E_{DOW}(\text{dow}_{i})+PE_{i}.$ | (5) | | | |

This represents both sequence order and elapsed time without expanding the interaction vocabulary. Appendix B.2 gives the sinusoidal parameterization.

##### Multi-head attention.

We apply standard multi-head self-attention (Vaswani et al., 2017) with $Q=K=V=(E_{1},\ldots,E_{L})$, and pool by taking only the [CLS] output as the shared representation $E_{S}\in\mathbb{R}^{d}$, which is concatenated with member embeddings and dense features before $\Phi$.

### 3.2. Multi-Outcome Extension

Marketing campaigns generate multiple outcomes (products) per member, so we extend the single-outcome DragonNet to a multi-label setting $Y_{k}$, $k\in\{1,\ldots,K\}$. Because per-product conversion rates are highly imbalanced, we use inverse-frequency weights $c_{k}$ and combine the standard BCE loss (with sigmoid $\sigma$) into a per-head outcome loss

$\mathcal{L}_{Y|T=t}=\tfrac{1}{K}\!\sum_{k}c_{k}\cdot\text{BCE}(y_{k},\hat{y}_{k}),\quad\mathcal{L}_{Y}=\mathcal{L}_{Y|T=1}+\mathcal{L}_{Y|T=0},$ | (6) | | | |

counting each head’s loss only when the corresponding $T$ is observed (as in DragonNet/TARNET). Treatment prediction uses standard BCE on the propensity-head output $\hat{e}(X)$, $\mathcal{L}_{T}=\mathbb{E}[\text{BCE}(T,\hat{e}(X))]$.

Following Shi et al. (2019), we add targeted regularization based on the efficient influence function (EIF) for the ATE under unconfoundedness (Chernozhukov et al., 2018):

$\phi_{\text{EIF}}=\hat{\tau}(X)+\tfrac{T}{\hat{e}(X)}(Y-\hat{\mu}_{1})-\tfrac{1-T}{1-\hat{e}(X)}(Y-\hat{\mu}_{0})-\text{ATE}.$ | (7) | | | |

The EIF is Neyman-orthogonal. We use its one-step correction as a training regularizer, not as a substitute for identification assumptions or as a stand-alone guarantee of unbiased CATE estimation:

$\mathcal{L}_{\text{tarreg}}=\mathbb{E}\!\left[\bigl\|Y-(\hat{Y}+\epsilon\,\psi)\bigr\|^{2}\right],$ | (8) | | | |

with $\psi=T/\hat{e}(X)-(1-T)/(1-\hat{e}(X))$ (the TMLE clever covariate (van der Laan and Rubin, 2006)), learnable scalar $\epsilon$, and $\hat{Y}$ the prediction from the head matching observed $T$.

### 3.3. Outcome Embeddings

New-product launches create a cold-start problem: marketers want to promote an offering before the model has seen examples for it. Our use of “CLIP-inspired” refers narrowly to a normalized shared embedding space with temperature-scaled similarity; it does not use CLIP’s large-scale contrastive pretraining. Instead of aligning images and text, we align a member’s outcome state with an attribute-derived product representation, allowing the same member encoder to produce scores for products whose attributes can be embedded, including products unseen during training.

Concretely, each outcome tower (a two-dense-block MLP with linear projection) drops the final sigmoid and returns hidden logits $h_{t}(X),h_{nt}(X)\in\mathbb{R}^{d_{o}}$ representing the member’s expected state under $T=1$ and $T=0$. A single shared outcome embedding matrix $E_{O}\in\mathbb{R}^{K\times d_{o}}$ encodes the $K$ products and is consumed by both towers, reflecting the fact that the product’s intrinsic semantics are invariant to treatment.

#### 3.3.1. Outcome Autoencoder

$E_{O}$ is the bottleneck of a lightweight autoencoder $E_{O}=\text{Encoder}(E^{\text{input}}_{O})$ over outcome attribute representations. $E^{\text{input}}_{O}$ is constructed by mapping each product attribute through a learned embedding and concatenating; attributes can be either structured (one-hot product taxonomy, business-line, format) or unstructured (LLM-derived embeddings of product descriptions). The autoencoder is trained jointly with the main task using MSE reconstruction

$\mathcal{L}_{\text{recon}}=\tfrac{1}{K}\!\sum_{k}\bigl\|E^{\text{input}}_{O,k}-\text{Decoder}(E_{O,k})\bigr\|^{2}.$ | (9) | | | |

This serves as semantic regularization: compared with a directly trained, task-specific $E_{O}$, the reconstruction objective encourages $E_{O}$ to retain the geometry of the input attribute space. The same setup enables large-scale embedding-space simulation, since pre-computed $h_{t}(X)$ and $h_{nt}(X)$ can be paired with arbitrary $E_{O}$ samples without re-running the full model.

#### 3.3.2. Outcome Matrix Layer

The final outcome logits are L2-normalized dot products with learnable log-inverse-temperatures $\nu_{t},\nu_{nt}$ (initialized to $\log 14.0$ following CLIP):

$\hat{y}_{t}=\tfrac{h_{t}(X)}{\|h_{t}(X)\|}\cdot\tfrac{E_{O}^{T}}{\|E_{O}\|}\cdot\exp(\nu_{t}),\quad\hat{y}_{nt}=\tfrac{h_{nt}(X)}{\|h_{nt}(X)\|}\cdot\tfrac{E_{O}^{T}}{\|E_{O}\|}\cdot\exp(\nu_{nt}).$ | (10) | | | |

Each row of $E_{O}$ is L2-normalized so that similarity depends purely on the angle between member-state and product representations, keeping logits on a consistent scale across products and treatment arms.

### 3.4. Complete Loss Function

The complete training objective sums the outcome, treatment, targeted-regularization, and reconstruction losses,

$\mathcal{L}=\mathcal{L}_{Y|T=1}+\mathcal{L}_{Y|T=0}+\mathcal{L}_{T}+\mathcal{L}_{\text{tarreg}}+\lambda_{\text{recon}}\mathcal{L}_{\text{recon}},$ | (11) | | | |

where $\lambda_{\text{recon}}$ weights the reconstruction loss; $\mathcal{L}_{\text{recon}}$ is included only when outcome attribute encodings are provided.

Although the complete research architecture contains several losses and optional modules, the serving path is modular rather than a jointly tuned monolith: outcome embeddings are used only for multi-product scoring, LLA is applied after supervised training to the last layer, and the LP consumes exported scores independently of model training. This separation lets each module be disabled or validated without retraining the rest of the decision pipeline, while the remaining tuning burden and feature sensitivity are limitations of the current shared-representation model.

## 4. Offline Simulations

In order to showcase the strengths of our proposed methodology, we perform an offline simulation study with a publicly available dataset. We focus on comparing the following approaches to targeting recommendations.

-

Incremental Modeling + Constrained Optimization (ours): integrates uplift modeling with constrained optimization for system-level incremental targeting; uses dual decomposition (Basu et al., 2020) at production scale and OR-Tools for simulation studies.

-

Bandit Incremental Modeling + Constrained Optimization: extends our method with BNN-based exploration to address feedback-loop bias (Nguyen et al., 2026); same solver setup as above.

-

Propensity Modeling + Constrained Optimization: optimizes predicted scores under constraints (Agarwal et al., 2015; Makhijani et al., 2019) without modeling organic outcomes.

-

Incremental Modeling + Ranking: ranks by estimated uplift (Chen et al., 2020; Meng et al., 2025; Sun et al., 2024) but ignores global constraints.

-

Propensity Modeling + Ranking: ranks by predicted response probability (Li et al., 2010; Covington et al., 2016; Cheng et al., 2016; Ying et al., 2018).

### 4.1. Dataset

We use the Open Bandit Dataset (OBD) (Saito et al., 2020), a real-world logged bandit dataset. We utilize the random-policy subset, synthetically mapping the original 34 products that were recommended to $>400$K users to 5 distinct actions that are relevant to the production incrementality use case: recommendation to one of four business lines or no-recommendation. The no-recommendation action is a key difference between incremental and non-incremental targeting. We consider the logged reward within the OBD dataset as indicative of a conversion event to one of the four business lines. We augment that reward by coupling it with the average product price of the four business lines and apply per-business-line LP volume bounds, as shown in Table 1.

*Table 1. Average product price and LP constraint bounds (as % of audience) for each business line.*

| Business Line | Avg. Price | Min. Vol. | Max. Vol. |

| A | $5 | 5% | 10% |

| B | $10 | 5% | 30% |

| C | $10 | 5% | 30% |

| D | $200 | 30% | 50% |

We also constructed a treatment variable that indicates whether a member was exposed to a marketing campaign showcasing one of the four business lines. This setup mirrors the production ecosystem and allows us to demonstrate the importance of incremental modeling. Finally, we assign a cost of $0.1 to each recommendation/targeting action to capture operational and bidding costs.

### 4.2. Setup

We performed an 80/20 split of the full dataset into training and prediction sets. We trained the predictive models (incremental and propensity) on the training set, generating incremental/propensity scores for the prediction set. We applied constrained optimization or ranking to arrive at the final recommendations for the prediction set (one of the five actions described above), with the objective function constructed by coupling the incremental/propensity scores for conversion to each business line with the corresponding average product price.

We solve the offline simulation LP with Google OR-Tools, which is tractable at this scale (${\sim}400$K members, 5 actions) and convenient for reproduction; at full production scale we use the dual-decomposition method described in Section 2.3 instead. We apply the volume bounds from Table 1. The ranking baseline assigns each member to the action with the highest predictive score, disregarding global business constraints.

We consider two simulation regimes: a single-turn (static) evaluation and a multi-turn (online) evaluation. For the single-turn setting, we construct the training log from a random logging policy in which all actions are approximately equally represented, and we evaluate each method once on the full prediction set using fixed model scores. For the multi-turn setting, we intentionally bias the training log by severely under-sampling a subset of actions, so that their estimated uplifts become high-variance and can even exhibit sign errors relative to the OBD ground-truth uplifts. We then simulate deployment over $T=200$ rounds using the prediction set as the environment: at each round, each method selects actions for the current batch, observes the realized rewards from its own recommendations, updates its training data accordingly, and incrementally updates the model before the next round.

### 4.3. Results

#### 4.3.1. Single-turn evaluation

We evaluated the different methods by calculating the corresponding average rewards with Doubly Robust Policy Evaluation. As shown in Table 2, incremental scores lead to higher rewards compared to propensity scores. We also provide the corresponding send volumes in Table 3, where it can be seen that incremental targeting leads to a higher percentage of no-recommendations as the engine is able to identify members likely to convert organically.

*Table 2. Average Rewards, Costs, and Net Returns*

| ML Model | Optim. | Avg Reward | Avg Cost | Avg Net Return |

$\$0.55\pm 0.14$ $\$0.091\pm 0.001$ $\$0.46\pm 0.13$| Incremental | Constr. Opt. | | | |

$\$0.49\pm 0.18$ $\$0.1\pm 0$ $\$0.39\pm 0.18$| Propensity | Constr. Opt. | | | |

$\$0.46\pm 0.15$ $\$0.090\pm 0.001$ $\$0.37\pm 0.15$| Incremental | Ranking | | | |

$\$0.41\pm 0.20$ $\$0.1\pm 0$ $\$0.31\pm 0.20$| Propensity | Ranking | | | |

*Table 3. Send Volumes*

| Method | Bus. Line A | Bus. Line B | Bus. Line C | Bus. Line D | No Rec. |

$5\%$ $30\%$ $26\%$ $30\%$ $9\%$| Incremental + Constr. Opt. | | | | | |

$5\%$ $18\%$ $27\%$ $50\%$ $0\%$| Propensity + Constr. Opt. | | | | | |

$0\%$ $46\%$ $44\%$ $0\%$ $10\%$| Incremental + Ranking | | | | | |

$0\%$ $0\%$ $0\%$ $100\%$ $0\%$| Propensity + Ranking | | | | | |

The send volumes also highlight the difference between Constrained Optimization and Ranking. As expected, Ranking provides solutions that deviate significantly from the desired business constraints.

We show the average costs in Table 2, where it can be seen that incremental modeling leads to reduced cost due to the reduced send volumes described above. The corresponding net returns are shown in Table 2, where Incremental Modeling coupled with Constrained Optimization provides the highest overall performance.

#### 4.3.2. Multi-turn evaluation

The previous single-turn evaluation shows that only approaches with Constrained Optimization can adequately satisfy the volume constraint bounds induced by business constraints, so we restrict attention to these approaches in the multi-turn evaluation. This multi-turn evaluation demonstrates the advantage of having an explicit mechanism to balance exploration and exploitation at the system level. The initial training-data bias we inject is designed to mirror common challenges in industry applications, such as cold-start, low representation for certain actions, and non-stationarity where the true reward function evolves over time (e.g., an initially low-performing action later becomes high-performing). Figure 2(a) shows that the Bandit Incremental Model is able to overcome this bias and ultimately outperform the other approaches after learning from the outcomes of its own actions, even though it may underperform its greedy counterpart in the first few steps of online learning. This reflects the short-term cost of exploration in exchange for long-term gains: after roughly 50 model updates, the Bandit Incremental Model begins to outperform both greedy variants. Finally, we show that the benefit of exploration is more pronounced when the initial training-data bias is larger, as illustrated in Figure 2(b).

*(a) Small bias setting.*

*(b) Large bias setting.*

*Figure 2. Average cumulative reward and 95% confidence intervals after 200 rounds of feedback. The confidence intervals are calculated from 30 simulation runs.*

### 4.4. Ablation Study

We tested 8 configurations on a fixed train/validation snapshot, each repeated 5 times with common hyperparameters. Figure 3(a) summarizes outcome, treatment, and uplift performance. Adding outcome embeddings preserves outcome AUROC and produces the highest uplift AUUC point estimate. Removing dense features reduces outcome AUROC, while its uplift effect depends on the configuration: AUUC increases for the base model but decreases slightly when bandit exploration is enabled. This is consistent with the dense features being prognostic rather than treatment-effect modifiers. Appendix A provides the full methodology, results, and outcome-embedding diagnostic.

*(a) Outcome prediction AUROC.*

*(b) Treatment prediction AUROC.*

*(c) Uplift AUUC.*

*Figure 3. Ablation results. Error bars denote 95% confidence intervals from 5 training runs.*

## 5. Production Deployment and Experimentation

We now turn from modeling to production. Deploying causal targeting at scale taught us that a handful of pragmatic enhancements beyond the core model and optimizer are what make the system work in practice. The most important are the construction of training data and the control of spend and delivery. These enhancements are largely invisible to offline simulations and ablation studies, which hold the data-generating process and delivery pipeline fixed, yet in our experience they are important for real-world success. We describe these enhancements first, followed by the serving architecture, agentic experimentation setup, and online results.

### 5.1. Training-Data Construction for Causal Estimation

Production logs allow us to build sequences of marketing touches and conversions, but causal estimation requires careful construction of the context sequence $X$, treatment action $T$, and conversion label $Y$. The context must contain only information available before treatment, while treatment and outcome are assigned from subsequent non-overlapping windows.

Let $R$ denote the snapshot run date. For each member, production samples $D=R-(W_{C}+W_{T}+W_{D})+U$, where $U\sim\operatorname{Uniform}\{0,\ldots,W_{D}-1\}$, $W_{D}=90$ days, $W_{T}=7$ days, and $W_{C}=30$ days. Thus $D\in[R-127,R-38]$. The model input contains marketing interactions and prior conversions in $[D-60,D)$; post-$D$ events are excluded. We set $T=1$ when at least one qualifying email send, on-platform impression, or video view occurs in $[D,D+7)$. We set $Y=1$ when the corresponding business-line or product-family conversion occurs in $[D+7,R]$, and $Y=0$ otherwise. This reserves at least 30 days of follow-up while earlier dates have longer outcome windows. Randomizing $D$ avoids a last-touch label, captures long-term action effects, and preserves variable-length histories.

### 5.2. Cost and Delivery Control

Controlling how much the causal policy spends and delivers matters for two reasons: a standing business requirement to pace committed budgets over the fiscal quarter, and an experimentation requirement to compare arms at matched delivery. Both stem from the same underlying gap between an assignment and its realized delivery.

At a given round, with $t$ suppressed, the LP in (3) chooses assignments $x_{u,i}$, but an assignment is not a guaranteed delivery in an auction-mediated, activity-gated channel. A member may not return during the campaign window, or the send may lose the downstream auction. We therefore estimate the probability of delivery,

$p_{u,i}=P(\text{delivered within window}\mid u,i),$ | (12) | | | |

using a classifier with isotonic calibration per member-lifecycle segment. The estimate enters the LP through the expected-impression constraint $\sum_{u,i}p_{u,i}x_{u,i}\in[C^{l}_{\text{imp}},C^{u}_{\text{imp}}]$ and the expected-cost constraint $\sum_{u,i}\hat{c}_{u,i}x_{u,i}\in[C^{l}_{\text{cost}},C^{u}_{\text{cost}}]$. Their bounds are calibrated to the observed BAU delivery envelope. This prevents the optimizer from concentrating assignments on members who appear incremental but are difficult or expensive to reach.

Across multiple rounds, such as in quarterly budget pacing or cumulative spend matching in an experiment, realized delivery and cost may differ from their LP predictions. We therefore introduce a feedback controller that adjusts the round-level cost target. Let $B$ be the committed budget, $S_{t}$ the cumulative realized spend, and $q(t/H)$ the desired cumulative pacing curve over a horizon $H$, with $q(0)=0$ and $q(1)=1$. The outer loop computes

$S_{t}^{\star}=Bq(t/H),\qquad e_{t}=S_{t}^{\star}-S_{t}.$ | (13) | | | |

Let $D_{t}$ denote the realized spend during interval $t$, and let $A_{t}=\sum_{u,i}\hat{c}_{u,i,t}x_{u,i,t}$ denote the LP-predicted spend of the selected assignments. The inner loop estimates the spend-realization ratio as $r_{t}=D_{t}/A_{t}$ and smooths it using $\hat{r}_{t}=\alpha r_{t}+(1-\alpha)\hat{r}_{t-1}$, where $\alpha\in(0,1]$ weights the newest observation. It then updates the LP cost target:

$\displaystyle C_{\text{cost},t+1}$ $\displaystyle=\Pi_{[C_{\min},C_{\max}]}\left(C_{\text{cost},t}+\kappa\frac{e_{t}}{\max(\hat{r}_{t},\epsilon)}\right),$ | (14) | | | | |

where $\kappa>0$ is the controller gain. The target $C_{\text{cost},t}$ can be converted into a two-sided LP constraint using a predefined tolerance around the target.

When realized spend falls behind the pacing curve, the controller raises the assignment cap in proportion to the gap and estimated deliverability; when spend runs ahead, it lowers the cap. The projection preserves operational bounds.

An early A/B test run without the above controls showed the treatment arm under-delivering: the causal policy withholds and reallocates sends, lowering treatment impressions relative to control. A raw total-bookings comparison then penalizes the treatment for delivering less rather than for choosing worse, confounding policy quality with delivery volume. Constraining treatment to the BAU impression and cost envelope equalizes delivery across arms and restores a clean, like-for-like read.

### 5.3. Serving Architecture

The system is served as a batch pipeline. DragonBandit scores $\mu_{1}$, $\mu_{0}$, and $e$ for the eligible population and samples incremental objectives via last-layer linearized Laplace Thompson sampling (Section 2.2); a dual-decomposition solver (Basu et al., 2020) computes the constrained allocation (3) under global guardrails, warm-started from the previous period’s dual $\lambda^{\star}$; and the feedback controller (Section 5.2) adjusts budget caps from realized deliverability before assignments are materialized to the campaign-management system. Because scoring and allocation are decoupled from serving, the same supervised checkpoint is deployed with or without exploration, and non-convergence solver falls back to the warm-started dual, which recovers over 99% of the current optimum under stable input distributions within the SLA.

### 5.4. Agentic Experimentation

We evaluated the end-to-end framework with an eight-week online A/B test on LinkedIn Feed marketing traffic, against the deployed business-as-usual (BAU) targeting stack. BAU is a standard two-tier recommender: the retrieval tier pre-selects each campaign’s audience using a propensity-based scoring model together with marketer-defined criteria, and the ranking tier is a second propensity-based model that estimates engagement probability and is followed by ranking heuristics for final assignment. The treatment arm keeps the marketer-defined criteria in retrieval and replaces both propensity layers with a single decision layer that scores members by predicted causal uplift across all campaigns and assigns them via the LP under global constraints. A meaningful policy difference is that the treatment arm can withhold a send whenever the predicted incremental value is negative or no feasible positive-incremental option exists. Members were randomly assigned 50/50 to the two experiment arms.

Standing up a representative BAU control and a matched treatment arm at scale requires configuring hundreds of live campaigns across products, segments, and budgets. Manual configuration is infeasible and, we found, subject to two subtle failure modes that bias measurement yet are invisible offline. First, when out-of-scope campaigns are suppressed via an exclusion segment defined by dynamic criteria (company, locale, activity), members drift across arms in a way correlated with platform activity, contaminating the intent-to-treat contrast. Second, when arms are assembled from shared segment definitions, a control audience can silently inherit a treatment send decision through a reused sub-segment. We therefore built an agentic campaign-setup tool that duplicates campaigns into hundreds of budget-split variants, freezes each audience to a static snapshot for the experiment’s duration, audits segment lineage so that no control segment depends on a treatment-derived decision, and materializes campaigns as drafts promoted by a single activation toggle with symmetric rollback. This provides identical scaffolding across arms, so the only systematic difference is the decision layer. It also improves operational safety. Budget parity is enforced by setting caps proportional to audience sizes and recalibrating as arm sizes drift.

### 5.5. Results

Measured against long-term-value metrics over an eight-week period, the treatment arm achieved a statistically significant $+7.20\%$ lift ($p=0.041$, 95% CI: $[0.31\%,14.09\%]$). The experiment evaluates the end-to-end policy as deployed. Its components satisfy coupled system requirements rather than representing independent product features: causal scoring defines the incremental objective, the exploration policy provides stochastic treatment variation that improves positivity (overlap), the LP enforces global business constraints, and the withhold rule prevents assignments with negative incremental value. Removing any of these components changes the operating requirements or the data-generating policy. The online result therefore measures the system-level impact of the complete production policy, while the offline studies examine individual mechanisms under controlled settings.

## Appendix A Detailed Ablation Study

### A.1. Protocol

We ran targeted ablation simulations on a static train/validation snapshot, holding hyperparameters (learning rate, epochs, batch size, hidden sizes) constant across configurations and varying only the indicated change. We tested 8 configurations, each repeated 5 times. The base model is the architecture from Section 3 on top of the DragonNet causal head, without bandit sampling or outcome embeddings (OE); we then added bandit and OE separately, and additionally removed three input categories: dense features, entity embeddings, and sequential interactions. These are not exhaustive component ablations: in particular, they do not separately isolate the Transformer operator from its sequence input, targeted regularization, or every interaction among loss terms.

### A.2. Outcome and Treatment Metrics

Outcome ROC-AUC is computed against the common validation dataset using the observed treatment for masking, $\hat{y}=T\hat{y}_{t}+(1-T)\hat{y}_{nt}$, and macro-averaged across labels. Treatment AUROC evaluates the propensity head against the observed treatment assignment.

### A.3. Uplift Results

Following CausalML (Chen et al., 2020), we rank observations by predicted uplift and compute

$\text{CumLift}(k)=\tfrac{\sum_{i=1}^{k}Y_{i}T_{i}}{\sum_{i=1}^{k}T_{i}+\epsilon}-\tfrac{\sum_{i=1}^{k}Y_{i}(1-T_{i})}{\sum_{i=1}^{k}(1-T_{i})+\epsilon},$ | (15) | | | |

with $\text{CumGain}(k)=k\,\text{CumLift}(k)$ and normalized area under the gain curve as AUUC. Implementations of AUUC and Qini differ (Gutierrez and Gérardy, 2017; Radcliffe, 2007; Devriendt et al., 2020); we use the CausalML implementation.

Removing dense features is neutral or beneficial for uplift AUUC despite reducing outcome AUROC from 0.857 to 0.826 and minimally affecting treatment AUROC. A plausible explanation is that these features are prognostic and predict outcomes regardless of treatment, rather than serving as treatment-effect modifiers. They may therefore add variance when the shared DragonNet representation must serve both propensity and outcome losses. This remains a hypothesis rather than an automated feature-selection mechanism; architectures such as FlexTENet (Curth and van der Schaar, 2021) could explicitly separate the two subspaces.

### A.4. Outcome-Embedding Diagnostic

We train on 9 of 10 outcomes, draw 5,000 hypothetical products from the learned embedding space, and score each against the held-out 10th product across the validation members. Figure 4 shows that samples near semantically similar trained products have higher AUROC against the held-out target, indicating that the geometry retains outcome-relevant structure. Because this diagnostic uses one held-out product and sampled embeddings rather than prospective real launches, it does not establish zero-shot production effectiveness.

*Figure 4. Outcome-embedding PCA. Each sampled hypothetical product is colored by AUROC z-score against the held-out product.*

## Appendix B Additional Architectural Details

### B.1. Bayes by Backprop

Bayes by Backprop (Blundell et al., 2015; Kingma et al., 2015) learns a Gaussian variational posterior $q_{\phi}(\theta)$ by maximizing the evidence lower bound. Predictive probabilities are approximated by Monte Carlo sampling,

$P(y^{*}=1\mid x^{*})\approx\frac{1}{M}\sum_{m=1}^{M}\sigma(f_{\theta^{(m)}}(x^{*})),\qquad\theta^{(m)}\sim q_{\phi}.$ | (16) | | | |

Compared with LLA, it captures non-local posterior uncertainty but adds stochastic training and repeated inference-time sampling. For a minibatch variant, an additional $\beta\mathcal{L}_{KL}$ term regularizes the posterior toward the prior; $\beta=1$ gives the standard ELBO objective. We did not explore alternative KL-weight schedules in the reported experiments.

### B.2. Temporal Position Encoding

For sequence position $pos$ and embedding coordinate $i$, we use the small-embedding-friendly sinusoidal variant

$PE_{(pos,2i)}=\sin(pos\,\omega_{i}^{\mathrm{new}}),\qquad PE_{(pos,2i+1)}=\cos(pos\,\omega_{i}^{\mathrm{new}}),$ | (17) | | | |

where $\omega_{i}^{\mathrm{new}}=\omega_{i}d/L$ and $\omega_{i}=10000^{-2i/d}$. The scaling adapts the standard sinusoid to the short interaction vocabulary and sequence length used in our setting.

## Ethical Considerations

We identify no ethical implications specific to the proposed optimization method beyond the standard privacy, fairness, and experimentation considerations of large-scale recommendation systems; deployments should follow applicable data-governance and review processes.

## References

- Agarwal et al. (2015) D. Agarwal, S. Chatterjee, Y. Yang, and L. Zhang Constrained optimization for homepage relevance. In Proceedings of the 24th International Conference on World Wide Web, pp. 375–384. Cited by: §1, 3rd item.

- Basu et al. (2020) K. Basu, A. Ghoting, R. Mazumder, and Y. Pan ECLIPSE: an extreme-scale linear program solver for web-applications. External Links: 2007.15936, Link Cited by: §2.3.1, 1st item, §5.3.

- Bertsimas et al. (2011) D. Bertsimas, D. B. Brown, and C. Caramanis The theory of robust optimization. SIAM Review. Cited by: §1.

- Blundell et al. (2015) C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra Weight uncertainty in neural networks. External Links: 1505.05424, Link Cited by: §B.1.

- Bonner and Vasile (2018) S. Bonner and F. Vasile Causal embeddings for recommendation. In Proceedings of the 12th ACM Conference on Recommender Systems, Cited by: §1.

- Chen et al. (2020) H. Chen, T. Harinen, J. Lee, M. Yung, and Z. Zhao CausalML: python package for causal machine learning. External Links: 2002.11631 Cited by: §A.3, 4th item.

- Chen et al. (2024) J. Chen, W. Wenjie, C. Gao, P. Wu, J. Wei, and Q. Hua Treatment effect estimation for user interest exploration on recommender systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1861–1871. Cited by: §1.1.

- Cheng et al. (2016) H. Cheng, L. Koc, J. Harmsen, T. Shaked, T. Chandra, H. Aradhye, G. Anderson, G. Corrado, W. Chai, M. Ispir, R. Anil, Z. Haque, L. Hong, V. Jain, X. Liu, and H. Shah Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems (DLRS 2016), pp. 7–10. Cited by: 5th item.

- Chernozhukov et al. (2018) V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal 21 (1), pp. C1–C68. External Links: ISSN 1368-4221, Document, Link Cited by: §2.1, §3.2.

- Covington et al. (2016) P. Covington, J. Adams, and E. Sargin Deep neural networks for YouTube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems (RecSys ’16), pp. 191–198. Cited by: 5th item.

- Curth and van der Schaar (2021) A. Curth and M. van der Schaar On inductive biases for heterogeneous treatment effect estimation. External Links: 2106.03765, Link Cited by: §A.3.

- Daxberger et al. (2021) E. Daxberger, A. Kristiadi, A. Immer, R. Eschenhagen, M. Bauer, and P. Hennig Laplace redux-effortless bayesian deep learning. Advances in neural information processing systems 34, pp. 20089–20103. Cited by: §2.2.1.

- Devlin et al. (2019) J. Devlin, M. Chang, K. Lee, and K. Toutanova BERT: pre-training of deep bidirectional transformers for language understanding. External Links: 1810.04805, Link Cited by: §3.1.

- Devriendt et al. (2020) F. Devriendt, T. Guns, and W. Verbeke Learning to rank for uplift modeling. CoRR abs/2002.05897. External Links: Link, 2002.05897 Cited by: §A.3.

- Foong et al. (2019) A. Y. Foong, Y. Li, J. M. Hernández-Lobato, and R. E. Turner ’In-between’uncertainty in bayesian neural networks. arXiv preprint arXiv:1906.11537. Cited by: §2.2.1.

- Foumani et al. (2023) N. M. Foumani, C. W. Tan, G. I. Webb, and M. Salehi Improving position encoding of transformers for multivariate time series classification. Data Mining and Knowledge Discovery 38 (1), pp. 22–48. External Links: ISSN 1573-756X, Link, Document Cited by: §3.1.

- Gao et al. (2024) C. Gao, Y. Zheng, W. Wang, F. Feng, X. He, and Y. Li Causal inference in recommender systems: a survey and future directions. ACM Transactions on Information Systems 42 (4), pp. 1–32. Cited by: §1.

- Gutierrez and Gérardy (2017) P. Gutierrez and J. Gérardy Causal inference and uplift modelling: a review of the literature. In Proceedings of The 3rd International Conference on Predictive Applications and APIs, C. Hardgrove, L. Dorard, K. Thompson, and F. Douetteau (Eds.), Proceedings of Machine Learning Research, Vol. 67, pp. 1–13. External Links: Link Cited by: §A.3.

- Kasy and Sautmann (2021) M. Kasy and A. Sautmann Adaptive treatment assignment in experiments for policy choice. Econometrica 89 (1), pp. 113–132. Cited by: §2.2.

- Kingma et al. (2015) D. P. Kingma, T. Salimans, and M. Welling Variational dropout and the local reparameterization trick. Advances in neural information processing systems 28. Cited by: §B.1.

- Künzel et al. (2019) S. R. Künzel, J. S. Sekhon, P. J. Bickel, and B. Yu Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences 116 (10), pp. 4156–4165. External Links: ISSN 1091-6490, Link, Document Cited by: §2.1.

- Li et al. (2010) L. Li, W. Chu, J. Langford, and R. E. Schapire A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web, pp. 661–670. Cited by: §2.2, 5th item.

- Makhijani et al. (2019) R. Makhijani, S. Chakrabarti, D. Struble, and Y. Liu LORE: a large-scale offer recommendation engine with eligibility and capacity constraints. In Proceedings of the 13th ACM Conference on Recommender Systems, pp. 160–168. Cited by: §1, 3rd item.

- McInerney et al. (2020) J. McInerney, B. Brost, P. Chandar, R. Mehrotra, and B. Carterette Counterfactual evaluation of slate recommendations with sequential reward interactions. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1779–1788. Cited by: §1.

- Meng et al. (2025) C. Meng, C. Zhai, X. Wang, S. Liu, X. Feng, L. Hu, X. Li, H. Li, and K. Gai Enhancing online video recommendation via a coarse-to-fine dynamic uplift modeling framework. In Proceedings of the 19th ACM Conference on Recommender Systems, pp. 82–92. Cited by: §1.1, 4th item.

- Neal (2020) B. Neal Introduction to causal inference. Course lecture notes (draft) 132. Cited by: §2.2.

- Nguyen et al. (2026) P. Nguyen, B. Zelditch, J. Chen, R. Patra, and C. Wei BanditLP: large-scale stochastic optimization for personalized recommendations. arXiv preprint arXiv:2601.15552. Cited by: §2.2, 2nd item.

- Nilsen et al. (2022) G. K. Nilsen, A. Z. Munthe-Kaas, H. J. Skaug, and M. Brun Epistemic uncertainty quantification in deep learning classification by the delta method. Neural networks 145, pp. 164–176. Cited by: §2.2.1.

- Radcliffe (2007) N. Radcliffe Using control groups to target on predicted lift: building and assessing uplift model. Direct Marketing Analytics Journal, pp. 14–21 (English). Cited by: §A.3.

- [30] S. Raha, K. Khare, and R. K. Patra Computationally efficient laplace approximations for neural networks. In NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty, Cited by: §2.2.1.

- Riquelme et al. (2018) C. Riquelme, G. Tucker, and J. Snoek Deep bayesian bandits showdown. In International conference on learning representations, Vol. 9. Cited by: §2.2.1.

- Saito et al. (2020) Y. Saito, S. Aihara, M. Matsutani, and Y. Narita Open bandit dataset and pipeline: towards realistic and reproducible off-policy evaluation. arXiv preprint arXiv:2008.07146. Cited by: §4.1.

- Schnabel et al. (2016) T. Schnabel, A. Swaminathan, A. Singh, N. Chandak, and T. Joachims Recommendations as treatments: debiasing learning and evaluation. In Proceedings of the 33rd International Conference on Machine Learning, Cited by: §1, §1, §2.2.

- Shalit et al. (2017) U. Shalit, F. D. Johansson, and D. Sontag Estimating individual treatment effect: generalization bounds and algorithms. In Proceedings of the 34th International Conference on Machine Learning, Cited by: §1, §2.1.

- Shi et al. (2019) C. Shi, D. Blei, and V. Veitch Adapting neural networks for the estimation of treatment effects. In Advances in Neural Information Processing Systems, Cited by: §1, §2.1, §2.1, §3.2.

- Su et al. (2024) Y. Su, X. Wang, E. Y. Le, L. Liu, Y. Li, H. Lu, B. Lipshitz, S. Badam, L. Heldt, S. Bi, et al. Long-term value of exploration: measurements, findings and algorithms. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pp. 636–644. Cited by: §2.2.1, §2.2.

- Sun et al. (2024) Z. Sun, H. Yang, D. Liu, Y. Weng, X. Tang, and X. He End-to-end cost-effective incentive recommendation under budget constraint with uplift modeling. In Proceedings of the 18th ACM Conference on Recommender Systems, pp. 560–569. Cited by: §1.1, 4th item.

- Swaminathan and Joachims (2015) A. Swaminathan and T. Joachims Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research 16 (1), pp. 1731–1755. Cited by: §2.2.

- Swaminathan and Joachims (2017) A. Swaminathan and T. Joachims Off-policy evaluation for slate recommendation. In Advances in Neural Information Processing Systems, Cited by: §1.

- van der Laan and Rubin (2006) M. J. van der Laan and D. Rubin Targeted maximum likelihood learning. The International Journal of Biostatistics 2 (1). External Links: Document Cited by: §3.2.

- Vaswani et al. (2017) A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin Attention is all you need. Advances in Neural Information Processing Systems. Cited by: §3.1.

- Wang et al. (2018) Y. Wang, D. Liang, L. Charlin, and D. M. Blei The deconfounded recommender: a causal inference approach to recommendation. Cited by: §1.

- Wei et al. (2024) C. Wei, B. Zelditch, J. Chen, A. A. S. T. Ribeiro, J. K. Tay, B. O. Elizondo, S. K. Selvaraj, A. Gupta, and L. B. D. Almeida Neural optimization with adaptive heuristics for intelligent marketing system. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 5938–5949. Cited by: §1.

- Ying et al. (2018) R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, and J. Leskovec Graph convolutional neural networks for web-scale recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18), pp. 974–983. Cited by: 5th item.

- Zhang et al. (2020) W. Zhang, D. Zhou, L. Li, and Q. Gu Neural thompson sampling. arXiv preprint arXiv:2010.00827. Cited by: §2.2.1.

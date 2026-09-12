<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2608.25871
     paper_id : p2s-2026-0006
     source   : https://arxiv.org/html/2608.25871v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# CEDAR: Controlled and Event-Driven Demand Forecasting via Residual Decomposition

Conference: KDD’26,August 09–13, 2026, Jeju Island, Republic of Korea; August 09–13, 2026; Jeju Island, Republic of KoreaConference: Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2; August 09–13, 2026; Jeju Island, Republic of KoreaProceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’26), August 09–13, 2026, Jeju Island, Republic of KoreaDOI: 10.1145/3770855.3818338ISBN: 979-8-4007-2259-2/2026/08CCS: Information systems Data miningCCS: Applied computing Forecasting
Junjie Meng Affiliation: School of Artificial Intelligence and Data Science, University of Science and Technology of China, Hefei, China email: mengjre@gmail.com , Ranxu Zhang Affiliation: School of Artificial Intelligence and Data Science, University of Science and Technology of China, Hefei, China email: zkd_zrx@mail.ustc.edu.cn , Zi-an Zhang Affiliation: Alibaba Group, Hangzhou, China email: zhangzian.zza@alibaba-inc.com , Shujun Liu Affiliation: Alibaba Group, Hangzhou, China email: liushujun_uestc@163.com , Xiaoning Qi Affiliation: Alibaba Group, Hangzhou, China email: xiaoning.qxn@alibaba-inc.com , Xiaozhou Xu Affiliation: Alibaba Group, Hangzhou, China email: heixia.xxz@alibaba-inc.com , Yanyong Zhang Affiliation: School of Artificial Intelligence and Data Science, University of Science and Technology of China, Hefei, China email: yanyongz@ustc.edu.cn , Hui Xiong Affiliation: Thrust of Artificial Intelligence, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China Affiliation: Department of Computer Science and Engineering, The Hong Kong University of Science and Technology, Hong Kong SAR, China email: xionghui@ust.hk and Chao Wang Note: Chao Wang is the corresponding author. Affiliation: School of Artificial Intelligence and Data Science, University of Science and Technology of China, Hefei, China email: wangchaoai@ustc.edu.cn

© cc

###### Abstract.

Forecasting in large-scale e-commerce marketplaces is increasingly required to support planning: merchants need to evaluate sales outcomes under future action sequences such as budget schedules, rather than passively predicting what happens next. However, most existing time series forecasting (TSF) approaches remain inherently passive. Even when incorporating operational decisions as auxiliary covariates, they typically optimize for correlation-based extrapolation under historical policies. This design suffers from autoregressive inertia and conflates endogenous market evolution with decision-induced transitions, leading to policy-insensitive rollouts and unreliable counterfactual analysis. To bridge this gap, we propose CEDAR (Controlled and Event-Driven Demand forecasting via Action-aware Residual decomposition), a two-stage framework for robust decision-conditioned simulation. In Stage I, an Action-Interleaved Transformer learns controllable action-conditioned state transitions for rollout under planned interventions. In Stage II, a Residual Correction Module leverages external event signals and LLM-assisted text representations to align noisy event descriptions with product context and correct event-driven deviations. Our study is enabled by a large-scale real-world dataset from Alibaba 1688, comprising approximately 32 million product trajectories with paired state–action sequences and aligned event signals. Extensive offline experiments and online controlled experiments in production demonstrate that CEDAR consistently improves simulation accuracy over strong TSF baselines and delivers practical gains for real-world budget planning.

###### Keywords:

Time Series Forecasting, Decision-Conditioned Simulation, Sales Forecasting

## 1. Introduction

*(a) Training loss & evaluation MSE visualization.*

*(b) Comparison of multivariate time-series models. T represents the length of prediction.*

*Figure 1. Performance comparison of four TSF baselines. All models are trained on the E-Com 15-Week dataset for 10 steps prediction. *

Time series forecasting (TSF) (Wen et al., 2019; Cleveland et al., 1990; Goswami et al., ; Wang et al., 2018) is widely used in e-commerce to support inventory planning, pricing, and marketing optimization (Das et al., 2024; Wen et al., 2020). However, in large-scale marketplaces such as Alibaba 1688, forecasting is not merely about predicting the future; it is about evaluating planned interventions. Merchants continuously act on the system—adjusting prices, launching promotions, and allocating advertising budgets—with the explicit goal of reshaping future demand trajectories. Consequently, the central question is not simply “what will happen next?”, but rather “what would happen if I follow a particular budget schedule (and other actions) over the next weeks?”. This shifts the goal from passive forecasting to decision-conditioned simulation (Aksu et al., ; Wilder et al., 2019): given historical states/actions and a future action sequence, the model should roll out the future demand trajectory under that plan.

Despite rapid progress in TSF architectures (Nelson, 1998; Hu et al., 2024), most popular forecasters are still trained and evaluated in a passive setting, ignoring the exogenous change. When directly applied to decision-conditioned rollout, such state-only TSF baselines (e.g., Informer (Zhou et al., 2021), PatchTST (Nie et al., 2023), PETFormer (Lin et al., 2024), Timer-XL (Liu et al., )) exhibit strong autoregressive inertia: they extrapolate along historical trends but cannot respond to counterfactual action schedules. As shown in Figure 1, these models perform poorly when action data is missing.

A natural remedy is to incorporate merchant operations into multivariate TSF models by treating actions (e.g., discounts, ad spend) as exogenous covariates (Salinas et al., 2020; Ansari et al., ; Ye et al., 2024; Lim et al., 2021), and to further refine predictions via exogenous corrections such as RevPred (Potapczynski et al., 2024). While this often improves short-horizon accuracy under historically observed policies, the covariate-fusion design is still limited for interventional simulation: it typically treats controllable interventions indistinguishably from passive context by simply concatenating them with states. As a result, the model tends to capture correlations in the historical joint distribution (Carta et al., 2018; Qiu et al., 2017), entangling endogenous market evolution with decision-induced transitions and even mixing action effects with non-stationary exogenous shocks (Zhang et al., 2026; Liu et al., 2026).

Building a faithful decision-conditioned simulator therefore faces two pivotal challenges. The first is controllable dynamics modeling. To enable reliable rollouts, the model must learn how actions explicitly drive state transitions ($\mathbf{s}_{t-1}\xrightarrow{\mathbf{a}_{t}}\mathbf{s}_{t}$), rather than just fitting the joint distribution of states and actions. The second is exogenous disentanglement. Real-world demand is frequently perturbed by non-stationary external factors (e.g., viral trends, holidays) that are not fully explainable by internal state-action history. If these exogenous shocks are not separated from action effects, the simulator will misattribute demand changes, leading to erroneous credit assignment and disastrous budget decisions. Therefore, we argue that robust what-if analysis in e-commerce requires an explicit, action-conditioned transition mechanism that is rollout-stable under novel action sequences, together with a separate component to absorb non-stationary exogenous shocks.

To this end, we propose CEDAR (Controlled and Event-Driven Demand forecasting via Action-aware Residual decomposition), a novel framework designed for robust decision-conditioned simulation. Our study is enabled by a large-scale real-world dataset from Alibaba 1688, containing over 32 million product trajectories with paired state–action sequences and aligned exogenous event signals, which also supports online evaluation in a production environment. Departing from the monolithic covariate-fusion approach, CEDAR explicitly disentangles the sales generation process into two stages. In Stage I, an Action-Interleaved Transformer (AIT) models decision-conditioned state transitions by interleaving state and action tokens in the causal order $\mathbf{s}_{t-1}\rightarrow\mathbf{a}_{t}\rightarrow\mathbf{s}_{t}$, encouraging the backbone to capture how interventions drive subsequent state changes. In Stage II, a Residual Correction Module predicts the residual between the simulated trajectory and observations using external event signals, leveraging LLM-enhanced text understanding to align noisy, unstructured event descriptions with product contexts and correct event-driven deviations. This separation preserves controllability for planning while improving robustness to bursty shocks, enabling reliable multi-step simulation under alternative merchant action plans.

In summary, our contributions are:

-

We formulate merchant-facing sales forecasting as a decision-conditioned simulation problem, highlighting the limitations of passive TSF under sequential interventions.

-

We propose CEDAR, a two-stage action-aware framework that (i) learns an explicit action-conditioned transition model via an Action-Interleaved Transformer, and (ii) corrects exogenous, non-stationary deviations with an event-driven residual module.

-

We validate our approach on a massive industrial dataset of 32 million trajectories and conduct online A/B testing in a real production environment. Results demonstrate that CEDAR significantly outperforms state-of-the-art baselines in simulation accuracy and delivers substantial efficiency gains in real-world budget planning. We are also working to release a version of this dataset to the community.

## 2. Related Work

### 2.1. General Time Series Forecasting and Covariate-Aware Modeling

Time Series Forecasting (TSF) (Zhou et al., 2025; Wen et al., 2021) has evolved from statistical models and recurrent neural networks to Transformer-based architectures (Wolff et al., 2024; Hu et al., 2025) and large-scale foundation models. Informer introduces probabilistic sparse attention for efficient long-horizon modeling, while Autoformer (Wu et al., 2021) and FEDformer (Zhou et al., 2022) incorporate decomposition and frequency-domain learning to better capture complex temporal patterns. More recent methods such as PatchTST (Nie et al., 2023) and iTransformer (Liu et al., 2023) further refine representation learning through patch-wise tokenization and dimension inversion, achieving strong performance on multivariate forecasting benchmarks (Qin et al., 2025).

Beyond pure autoregressive modeling, a parallel line of research incorporates rich covariates to improve multi-horizon forecasting (Wang et al., 2021b; Wang et al., 2023). Temporal Fusion Transformer (TFT) (Punati et al., 2025; Lim et al., 2021) represents a highly influential framework in this direction, integrating variable selection networks, gated residual connections, and attention mechanisms to adaptively fuse historical observations, known future inputs, and static features. Despite its empirical success, TFT (Oliveira and Ramos, 2024) fundamentally follows a covariate-conditioned forecasting paradigm, modeling $p(\mathbf{s}_{t+1}\mid\mathbf{s}_{\leq t},\mathbf{a}_{\leq t})$ via feature-level fusion rather than explicitly learning action-conditioned state transitions. As a result, TFT primarily captures statistical correlations under historically observed policies, which limits its robustness under policy shifts and counterfactual simulation, where future action sequences deviate from training distributions.

More broadly, most multivariate TSF models (Woo et al., 2022) treat controllable variables as auxiliary covariates, implicitly assuming invariant system dynamics (Meng et al., 2026). This assumption is misaligned with decision-intensive environments such as e-commerce, where merchant actions actively reshape future trajectories. In contrast, our work formulates sales forecasting as a decision-conditioned simulation problem and explicitly models the transition operator $\mathbf{s}_{t+1}=f(\mathbf{s}_{t},\mathbf{a}_{t+1})$. By interleaving state and action tokens in a unified Transformer, we encode the causal temporal ordering $\mathbf{s}_{t-1}\rightarrow\mathbf{a}_{t}\rightarrow\mathbf{s}_{t}$, enabling robust learning of action-conditioned dynamics and stable long-horizon rollout under novel intervention strategies.

### 2.2. Offline Reinforcement Learning via Sequence Modeling

Recent advances in offline reinforcement learning (Levine et al., 2020; Nakamoto et al., 2023; Huang et al., 2024; Kumar et al., 2020; Wang et al., 2021a) have reformulated policy learning and planning as a conditional sequence modeling problem. Decision Transformer (DT) (Chen et al., 2021) pioneered this paradigm by conditioning autoregressive transformers on desired return-to-go, demonstrating that generic sequence models can achieve competitive control performance without explicit dynamic programming. Trajectory Transformer (Janner et al., 2021) further discretized continuous trajectories into token sequences and enabled long-horizon planning via beam search. Subsequent extensions improved data efficiency and deployment flexibility, including Online Decision Transformer (Zheng et al., 2022), Q-learning Decision Transformer (QDT) (Yamagata et al., 2023), and Value-Guided Decision Transformer (VDT) (Zheng et al., 2025), which introduced critic guidance, advantage weighting, and value regularization to stabilize learning and mitigate compounding errors.

Despite their success in control-centric benchmarks, these approaches are not directly applicable to merchant-facing sales forecasting and simulation. In typical e-commerce scenarios, the primary quantity of interest is the future sales trajectory itself, which simultaneously plays the role of system state and optimization objective. More fundamentally, existing DT-style models are designed to infer an implicit policy that maximizes long-term return under a fixed environment, rather than to explicitly learn a controllable state transition mechanism. In merchant budget planning, however, the core requirement is not to discover a single optimal policy, but to evaluate and compare multiple candidate strategies through reliable what-if simulation. This necessitates disentangling endogenous market dynamics from action-induced transitions (Zhang et al., 2026; Liu et al., 2026), and explicitly modeling how alternative interventions reshape future trajectories. Recent explorations in causal generative modeling, such as DoFlow (Wu et al., 2025), have highlighted the necessity of incorporating causal flows for robust interventional and counterfactual time-series prediction (Rafetseder et al., 2013), ensuring that simulated trajectories remain physically and logically consistent under distribution shifts.

In contrast to policy-centric sequence modeling, our work focuses on learning an action-conditioned state transition operator for stable multi-step rollout. By separating controllable effects from latent external fluctuations, CEDAR enables robust trajectory simulation under diverse intervention plans, providing a principled foundation for decision-aware forecasting and strategic exploration in highly non-stationary e-commerce environments.

*Figure 2. Framework of CEDAR. The AIT module is trained in the first stage, and frozen in the second stage.*

## 3. Method

We formalize merchant-facing sales forecasting as a decision-conditioned simulation problem. At each discrete time step $t$, the system is characterized by a product state vector $\mathbf{s}_{t}\in\mathbb{R}^{d_{s}}$ that captures endogenous signals such as impressions, clicks, favorites, and purchases, together with a merchant action vector $\mathbf{a}_{t}\in\mathbb{R}^{d_{a}}$ encoding controllable interventions including discount strategies and marketing expenditures. Both states and actions are aggregated over a fixed temporal granularity (e.g., weekly) in the E-Com 15-week dataset.

Given historical observations $\{\mathbf{s}_{1:T},\mathbf{a}_{1:T}\}$ and a planned future action sequence $\mathbf{a}_{T+1:T+H}$, our objective is to simulate the counterfactual evolution of product trajectories $\mathbf{s}_{T+1:T+H}$ under alternative merchant policies. This formulation enables forward rollout of system dynamics conditioned on hypothetical budget allocation strategies, thereby supporting downstream tasks such as operational planning, strategy evaluation, and optimal budget scheduling. Unlike conventional TSF settings that focus on passive extrapolation, our goal is to learn a controllable state transition model that enables stable long-horizon simulation and reliable what-if analysis.

### 3.1. Decision-Conditioned Transition Modeling

Traditional multivariate TSF formulations typically aim to learn a conditional distribution $p(\mathbf{s}_{t+1}\mid\mathbf{s}_{\leq t},\mathbf{a}_{\leq t})$, where merchant actions are treated as auxiliary covariates. While effective for short-term prediction under historically observed policies, this paradigm fundamentally conflates endogenous system evolution and decision-induced dynamics, resulting in models that primarily capture correlations rather than learning explicit action-conditioned transition mechanisms.

In contrast, decision-conditioned simulation requires modeling a controlled dynamical system governed by a transition operator

$\mathcal{T}:(\mathbf{s}_{t},\mathbf{a}_{t+1})\mapsto\mathbf{s}_{t+1},$ | (1) | | | |

which describes how merchant interventions actively shape future trajectories. This formulation aligns with the core requirement of budget planning and strategy exploration, where future actions are intentionally optimized and may deviate significantly from historical distributions. Learning such an explicit transition operator enables faithful rollout under hypothetical action sequences and mitigates the extrapolation failures commonly observed in covariate-based TSF models. While some approaches incorporate external context, they typically fail to disentangle endogenous system dynamics from exogenous market shocks, leading to confounded state representations.

Motivated by this perspective, we design the Action-Interleaved Transformer (AIT), which treats both states and actions as first-class tokens and explicitly encodes their causal temporal ordering, thereby introducing a structural inductive bias toward action-conditioned transition modeling.

### 3.2. Stage I: Action-Interleaved Transformer

The first stage of CEDAR focuses on learning action-conditioned state transition dynamics. Instead of concatenating actions as exogenous features, we model both states and actions as interleaved tokens in a unified temporal sequence. Concretely, for each time step $t$, we construct a token ordering

$\mathbf{s}_{t-1}\rightarrow\mathbf{a}_{t}\rightarrow\mathbf{s}_{t},$ | (2) | | | |

which reflects the natural causal structure of merchant operations: historical system states inform merchant decisions, and these decisions subsequently drive state transitions.

Each state token and action token is first projected into a shared embedding space through separate linear encoders, preserving their semantic roles while enabling joint attention. A causal Transformer backbone is then applied over the interleaved sequence, ensuring that predictions at time $t+1$ depend only on past and present information. Formally, the model learns a parametric transition function

$\hat{\mathbf{s}}_{t+1}=f_{\theta}(\mathbf{s}_{\leq t},\mathbf{a}_{\leq t+1}),$ | (3) | | | |

where $f_{\theta}$ is implemented by the Action-Interleaved Transformer.

The interleaving design imposes a structured attention pattern that explicitly aligns merchant actions with subsequent state transitions, enabling the model to learn directed influence pathways such as $\mathbf{a}_{t}\rightarrow\mathbf{s}_{t}$. Compared to covariate-based architectures, this design encourages the Transformer to focus on controllable dynamics, improving stability and generalization when simulating under novel policies.

Furthermore, product state vectors exhibit strong internal structure, including funnel-like dependencies from exposure to conversion. The self-attention mechanism within AIT naturally captures such hierarchical interactions while modeling their modulation by merchant actions, enabling fine-grained transition learning across heterogeneous behavioral signals.

AIT is trained using a one-step prediction loss over observed trajectories:

$\mathcal{L}_{\text{AIT}}=\sum_{t}\|\hat{\mathbf{s}}_{t+1}-\mathbf{s}_{t+1}\|_{2}^{2}.$ | (4) | | | |

### 3.3. Stage II: Residual Correction with External Signals

While AIT captures endogenous dynamics and the direct effects of merchant actions, real-world sales trajectories are also influenced by latent, time-varying exogenous factors, including breaking news, social media trends, seasonal effects, and macroeconomic shocks. These influences introduce non-stationary perturbations that are difficult to infer solely from historical state-action trajectories, particularly in highly volatile markets.

To account for such effects without compromising the controllability and stability of the learned transition operator, we introduce a Residual Correction Module that explicitly models deviations induced by external factors. Specifically, we decompose the system evolution as

$\mathbf{s}_{t+1}=f_{\theta}(\mathbf{s}_{\leq t},\mathbf{a}_{\leq t+1})+\epsilon_{t},$ | (5) | | | |

where $f_{\theta}$ captures controllable dynamics and $\epsilon_{t}$ represents latent external perturbations. Stage I models the former, while Stage II estimates the latter.

##### External Signal Construction.

We leverage a large language model (LLM) to extract structured representations of exogenous market signals (Wang et al., 2026; Meng et al., 2026). For each time window $t$, we collect news from the previous week and prompt the LLM to extract product-level keywords indicative of potential demand surges. Simultaneously, major holidays and seasonal events occurring in the current window are appended as additional signals. Rather than treating these cues as isolated tags, we instruct the LLM to synthesize them into a coherent natural language description, which is subsequently encoded by a pre-trained text encoder to produce a dense hotspot embedding $\mathbf{h}_{t}$ that captures the semantic context of external influences. Details of hotspot embedding generation can be found in Appendix A.

##### Residual Prediction.

As shown in Figure 2, we construct item status embedding through concatenating the embeddings of item titles and the tags, the predicted next state $\hat{\mathbf{s}}_{t+1}$ and recent historical states $\mathbf{s}_{t-k:t}$. The hotspot embedding $\mathbf{h}_{t}$ is then combined with the item status embedding via a cross-attention module, enabling dynamic alignment between external events and product-level temporal patterns. The resulting representation is passed through a multi-layer perceptron to estimate the residual correction:

$\Delta\mathbf{s}_{t+1}=g_{\phi}(\mathbf{h}_{t},\hat{\mathbf{s}}_{t+1},\mathbf{s}_{t-k:t}).$ | (6) | | | |

The loss of Stage-II training is calculated through:

$\mathcal{L}_{\text{RC}}=\sum_{t}\|{\mathbf{s}}_{t+1}-\hat{\mathbf{s}}_{t+1}-\Delta\mathbf{s}_{t+1}\|_{2}^{2}.$ | (7) | | | |

The final simulated state is then obtained as

$\tilde{\mathbf{s}}_{t+1}=\hat{\mathbf{s}}_{t+1}+\Delta\mathbf{s}_{t+1}.$ | (8) | | | |

### 3.4. Discussion

#### 3.4.1. Why exclude temporal signals

During our empirical investigation, we also explored explicitly modeling periodic temporal signals and injecting them as additional perturbation embeddings into the predictive sequence. However, experimental results indicate that such temporal embeddings are difficult to align with the semantic representations of bursty external events extracted from text, leading to ineffective fusion. Moreover, explicitly modeling standalone temporal embeddings substantially increases the learning complexity of the network, resulting in degraded predictive performance.

We hypothesize that, in e-commerce settings, the primary performance gains attributed to temporal signals largely stem from holiday-driven demand fluctuations. Since major holidays and seasonal events are already explicitly incorporated into the Residual Correction Module through LLM-based external signal construction, introducing timestamp embeddings as an additional perturbation provides limited marginal benefit. Consequently, we do not include explicit timestamp embeddings in CEDAR.

#### 3.4.2. The design of Residual Correction

While prior work (Potapczynski et al., 2024) similarly utilized a residual module to refine predictions from the backbone model, it relied on simple MLPs to process external covariates independently. This approach treats all covariates uniformly, thereby overlooking the semantic alignment between external shocks and specific items. For instance, the demand for seasonal items, such as Christmas hats, is unlikely to surge during the Chinese New Year, despite the presence of a significant festival event. In contrast, CEDAR employs a cross-attention mechanism. Specifically, the encoded external shock embeddings serve as queries to explicitly explore and model the relevance between global events and item-specific dynamics. This mechanism enables CEDAR to dynamically align external impacts with internal product states, thereby significantly enhancing the model’s effectiveness in capturing complex dependencies.

#### 3.4.3. Training and Inference Pipeline

##### Training Phase

We adopt a decoupled two-stage training paradigm. This design choice is primarily motivated by the need to isolate controllable system dynamics from stochastic external perturbations. By separating the learning process, we ensure that the action-conditioned transition operator remains stable under significant policy shifts, while the residual correction module can flexibly adapt to non-stationary market conditions. This decoupling prevents the model from over-relying on exogenous correlations, which empirically leads to substantial improvements in long-horizon rollout accuracy and enhances the robustness of "what-if" strategy simulations.

##### Inference Phase

During the iterative simulation process, the state for each successive time step is generated through a sequential execution of the model components, where the Action-Interleaved Transformer first generates an initial state estimate based on the historical trajectory and planned interventions, which is then refined by the Residual Correction Module to account for adjustments necessitated by latent external signals. The final predicted state for the time step is obtained as the sum of these two outputs; to perform multi-step forecasting, this result is appended to the historical sequence and fed back into the model in an auto-regressive manner, enabling the stable simulation of future product trajectories over an extended horizon.

*Table 1. Performance comparison of CEDAR and other baselines. The next 5 setting predicts the next 5 weeks given the previous 10 weeks, while the next 10 setting predicts the next 10 weeks given the previous 5 weeks. All offline results are reported as mean $\pm$ standard deviation over five independent runs.*

| Metric | Setting | CEDAR | Informer | TFT | PatchTST | PETFormer | Timer-XL |

$\downarrow$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| MSE | next 10 | 0.4140.015 | 32.221 | 0.7220.024 | 0.8490.026 | 0.7430.021 | 2.980.14 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| next 5 | 0.1820.006 | 3.441.8 | 0.5720.015 | 0.4240.011 | 0.4340.013 | 1.340.05 |

$\downarrow$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| MAE | next 10 | 0.1320.002 | 3.232.1 | 0.1940.007 | 0.2010.006 | 0.1920.006 | 0.6270.035 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| next 5 | 0.06030.001 | 0.7300.35 | 0.1750.005 | 0.1290.004 | 0.1390.004 | 0.4720.018 |

$\downarrow$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| NMSE | next 10 | 0.1890.004 | 14.79.2 | 0.3370.012 | 0.3870.013 | 0.3390.011 | 1.360.08 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| next 5 | 0.08300.001 | 1.570.80 | 0.2670.009 | 0.1910.006 | 0.1980.007 | 0.6120.022 |

## 4. Experiments

In this section, we conduct extensive experiments to systematically evaluate the effectiveness of the proposed CEDAR framework, including both the Action-Interleaved Transformer (AIT) and the Residual Correction Module. Our evaluation is designed to answer the following key research questions:

-

RQ1: How does CEDAR perform compared to state-of-the-art time series forecasting baselines in decision-conditioned sales simulation?

-

RQ2: To what extent does explicitly modeling merchant actions as a dedicated modality improve forecasting accuracy and rollout stability?

-

RQ3: Can the proposed Residual Correction Module effectively capture latent external impacts and further refine simulated trajectories?

-

RQ4: What is the real-world impact of deploying CEDAR in online budget planning scenarios on merchant engagement and platform performance?

### 4.1. Evaluation Setup

#### 4.1.1. Dataset

To support large-scale training and evaluation, we construct a comprehensive e-commerce dataset from Alibaba 1688, covering approximately 32 million product trajectories spanning the years 2024 and 2025. Each trajectory consists of a sequence of product states and merchant actions aggregated over rolling 7-day windows.

Specifically, the state vector includes nine core indicators: 7-day aggregated impressions, page views, favorites, add-to-cart events, number of buyers, gross merchandise volume (GMV), advertisement impressions, advertisement clicks, and the number of customer inquiries. The action vector contains two controllable variables: the 7-day average marketing discount (defined as the ratio between the discounted price and the original price) and the total advertising expenditure during the same period.

We segment each trajectory into overlapping windows of 15 consecutive weeks. A new window is sampled every 15 weeks, while remaining weeks within each year are backward-sampled to construct additional windows. This process yields approximately 32 million training samples. To fully evaluate the model’s ability on unseen external shocks, we reserve the final window of 2025 as the test set and use all preceding windows for training, because random sampling may cause shortcut learning on already observed external shocks.

Each product in the dataset is annotated with a three-level category taxonomy, consisting of a primary category, secondary category, and fine-grained subcategory. At the finest granularity, the taxonomy contains 8,942 distinct subcategories. Due to the substantial heterogeneity in scale across product categories, we perform category-wise normalization. Specifically, for each subcategory, we compute the mean and standard deviation of all nine state variables and two action variables, and apply z-score normalization within each subcategory to mitigate scale disparities and stabilize model training.

To account for abrupt external shocks and bursty demand fluctuations, we further incorporate exogenous event signals derived from both on-platform trending search topics and off-platform public hotspots, along with a curated list of 26 major holidays. These signals are used to support the modeling of latent external factors, with representative examples provided in the Appendix.

As the largest domestic B2B wholesale platform in China, Alibaba 1688 provides a uniquely rich environment for studying decision-conditioned forecasting and budget planning. The resulting E-Comm 15-Week dataset represents one of the largest real-world benchmarks for action-aware sales simulation, and we are actively working toward releasing a partially anonymized version to facilitate future research.

#### 4.1.2. Baselines

We compare CEDAR against a diverse set of representative time series forecasting baselines, covering both classical architectures and recent foundation models. The selected methods span convolutional, attention-based, patch-wise, and large-scale pretrained paradigms, providing a comprehensive evaluation across different modeling principles. Specifically, we consider:

-

Informer (Zhou et al., 2021) is a canonical Transformer-based forecasting model that introduces probabilistic sparse attention to efficiently model long-range temporal dependencies.

-

TFT (Lim et al., 2021) integrates exogenous covariates by variable selection networks, gated residual connections and attention mechanisms.

-

PatchTST (Nie et al., 2023) reformulates time series forecasting by segmenting the input sequence into local patches and applying channel-independent Transformer encoders.

-

PETFormer (Lin et al., 2024) integrates periodicity-aware encoding and efficient temporal attention mechanisms to explicitly capture seasonal structures and long-term dependencies.

-

Timer-XL (Liu et al., ) is a large-scale pretrained time series foundation model trained on heterogeneous real-world corpora.

For a fair comparison, all baselines are trained under identical data splits, input horizons, and prediction windows. All offline results are reported as the mean and standard deviation over five independent runs with different random seeds. This protocol allows us to evaluate whether the observed improvements are robust to optimization randomness rather than arising from a single favorable initialization. For models that support exogenous covariates, merchant actions are concatenated with the state variables as additional input channels. This setup reflects the prevailing practice in multivariate TSF, where actions are treated as auxiliary contextual features. In contrast, CEDAR explicitly models merchant actions as a first-class modality and interleaves them with state transitions, enabling direct learning of decision-conditioned dynamics and stable multi-step rollouts under alternative action plans. A detailed wall-clock cost analysis is provided in Appendix C.

*Table 2. Ablation study results using MSE and MAE metrics. One-stage training refers to jointly training the Action-Interleaved Transformer and the Residual Correction Module.*

$\downarrow$ $\downarrow$| Ablation Variants | MSE | MAE |

| next 10 | next 5 | next 10 | next 5 |

| Full Model | 0.414 | 0.182 | 0.132 | 0.0603 |

| w/o Recent states | 0.443 | 0.209 | 0.137 | 0.0634 |

| w/o Product metadata | 0.466 | 0.227 | 0.143 | 0.0654 |

| w/o AIT Prediction | 0.499 | 0.264 | 0.181 | 0.0697 |

| w/o Holiday keywords | 0.451 | 0.213 | 0.169 | 0.0651 |

| w/o News keywords | 0.417 | 0.188 | 0.141 | 0.0601 |

| w/o All(AIT only) | 0.489 | 0.252 | 0.177 | 0.0691 |

| Temporal shuffle | 0.527 | 0.274 | 0.194 | 0.0712 |

| One stage | 0.471 | 0.231 | 0.158 | 0.0674 |

#### 4.1.3. Evaluation Metrics & Implementation Details

We adopt three widely used time series forecasting metrics, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and Normalized Mean Squared Error (NMSE). The NMSE is computed as $NMSE=\frac{MSE}{Mean(y^{2})}$. To better align with real-world merchant requirements, we consider two evaluation settings: (i) forecasting the next 10 weeks given the past 5 weeks of observations, and (ii) forecasting the next 5 weeks given the past 10 weeks of observations.

All experiments are conducted on a cluster equipped with 4 NVIDIA H20 GPUs for both training and inference. The hidden dimension of all models is uniformly set to 256. For Transformer-based models, including CEDAR, we configure the number of attention heads and layers as $n_{\text{head}}=4$ and $n_{\text{layer}}=5$, respectively.

Given that the majority of textual data in our dataset is in Chinese, we adopt the BGE-zh-v1.5 model as our text encoder, with an embedding dimension of 1024. For hotspot keyword extraction and sentence organization, we employ the Qwen-Plus model.

*Figure 3. Visualization of predicted views trajectories of different baselines.*

### 4.2. Validation of CEDAR (RQ1)

Table 1 reports the quantitative comparison between CEDAR and a diverse set of representative TSF baselines, including Informer, PatchTST, PETFormer, and Timer-XL, under different forecasting horizons. We observe that CEDAR consistently achieves the best performance across all metrics and horizons, demonstrating substantial improvements over existing methods.

Specifically, at horizon next 5, CEDAR attains an MSE of $0.182$, significantly outperforming the strongest baseline PatchTST $0.424$ and PETFormer $0.434$, corresponding to relative improvements of $57.1\%$ and $58.1\%$, respectively. Similar trends are observed under NMSE, where CEDAR reduces the error to $0.083$, yielding more than $56\%$ improvement over the best baseline. For the next 10 horizon, CEDAR also achieves the lowest MSE $0.414$ and NMSE $0.189$, consistently outperforming all competing approaches.

Notably, classical TSF models such as Informer suffer from severe performance degradation, with MSE exceeding $30$ at next 10. This phenomenon highlights the limitation of decision-unaware forecasting models, which fail to capture the strong and non-stationary effects induced by merchant actions (e.g., pricing and traffic investments). In contrast, CEDAR explicitly models the interaction between system states and merchant decisions, enabling robust and accurate simulation under dynamic, intervention-driven environments. Besides, the ablation results in Table 2 also prove the effectiveness of two-stage training.

To further evaluate CEDAR’s performance on long-horizon prediction, we conduct additional long-horizon forecasting experiments, with detailed results reported in Appendix D.

Overall, these results verify the superiority and robustness of CEDAR in decision-conditioned sales forecasting, particularly in scenarios characterized by strong action-driven distribution shifts.

### 4.3. Action Modality Ablation (RQ2)

To evaluate the impact of explicitly modeling merchant actions as a dedicated modality, we conduct a targeted analysis focusing on traffic forecasting, where advertising expenditure exhibits the strongest correlation with future dynamics. We compare CEDAR against multiple baselines that incorporate actions as auxiliary covariates, including PatchTST, Timer-XL, as well as an additional multilayer perceptron (MLP) baseline that directly concatenates historical states, product metadata, and action variables as input features.

Figure 3 illustrates the predicted traffic trajectories for a representative product, where the merchant launches two advertising campaigns starting at weeks 5 and 12, respectively. Notably, baselines that treat actions as exogenous covariates exhibit limited sensitivity to intervention signals, particularly during the early phase of training. In the first 10 weeks, these models tend to extrapolate local trends from historical traffic peaks, with predicted local maxima typically lagging behind previously observed high-traffic points. This behavior reflects a strong reliance on autoregressive correlations, causing action effects to be largely diluted by dominant state dynamics.

In contrast, CEDAR, by explicitly interleaving actions with state transitions, demonstrates substantially improved responsiveness to marketing interventions. In particular, CEDAR accurately captures the decline in future traffic following the reduction of advertising expenditure, correctly anticipating the downward trend rather than merely propagating historical patterns. This ability enables stable and realistic multi-step rollouts under dynamically changing action plans, which is critical for budget planning and strategy exploration.

These findings indicate that modeling merchant actions as a first-class modality introduces a crucial structural inductive bias, allowing the model to disentangle endogenous temporal dynamics from decision-induced transitions. As a result, CEDAR achieves superior forecasting accuracy and significantly enhanced rollout stability in counterfactual scenarios involving policy shifts.

### 4.4. External Shock Controls (RQ3)

To assess whether the proposed Residual Correction Module effectively captures latent external impacts and refines simulated trajectories, we conduct a dedicated ablation study, with quantitative results reported in Table 2. These variants remove or perturb different input sources used by the Residual Correction Module. In addition, we include a temporal-shuffle variant, where holiday and hotspot signals are randomly reassigned to different dates. This variant performs worse than the aligned-event setting and even degrades relative to AIT-only in next-5 MSE, indicating that CEDAR benefits from temporally meaningful event-demand alignment rather than merely using event embeddings as generic auxiliary features.

Overall, we observe that incorporating the Residual Correction Module yields a modest improvement in MSE but leads to a substantial reduction in MAE. This discrepancy is expected, as external shocks often manifest as abrupt, localized deviations rather than long-term trend shifts. Consequently, the module primarily enhances short-horizon accuracy and robustness to bursty fluctuations, which is better reflected by MAE than by MSE.

Beyond aggregate error metrics, the Residual Correction Module plays a critical diagnostic role in preserving trajectory diversity and capturing idiosyncratic dynamics across similar products. Without this module, the model tends to generate highly similar trend forecasts for items belonging to the same fine-grained category, resulting in homogenized predictions that fail to reflect product-specific external influences. By explicitly modeling residual signals induced by latent exogenous factors, the proposed module reintroduces trajectory heterogeneity and substantially improves realism in multi-step rollouts.

We further illustrate the critical effect of external shock modeling through a real-world case study shown in Figure 4. The example corresponds to a sudden viral trend on social media surrounding “Tanghulu with Naipizi” (a fusion snack combining milk skin and candied hawthorn), which triggered a sharp surge in demand for a particular tanghulu supplier. In this setting, the model is tasked with performing next-1 forecasting, using observations from the preceding five weeks to predict the state vector of the subsequent week.

As observed, baseline models lacking explicit external event modeling mechanisms continue to extrapolate along previously observed trajectories, failing to anticipate the upcoming sales spike. In contrast, the Residual Correction Module successfully extracts highly relevant keywords (e.g., “Tanghulu”) from trending search and external hotspot signals, enabling the model to correctly predict the abrupt increase in traffic and sales. This qualitative result demonstrates that the proposed module effectively captures latent external shocks and injects critical event-driven signals into the forecasting process.

Taken together, both quantitative and qualitative evidence confirms that the Residual Correction Module is essential for robust short-term forecasting under volatile market conditions, significantly enhancing the model’s responsiveness to bursty external events and improving the fidelity of simulated trajectories.

*Figure 4. Visualization of predicted views trajectories with next-1 setting of different baselines. *

### 4.5. Public Dataset Generalization

To evaluate generalization beyond Alibaba 1688, we conduct additional experiments on the public Kaggle Store Sales dataset. Although this dataset contains retail demand series, promotion-related covariates, and calendar events, it differs from our setting because it is organized at the store-family level and lacks merchant actions with explicit budget-planning semantics. Thus, this experiment mainly tests whether the event-aware residual decomposition transfers to a public retail forecasting scenario, rather than fully reproducing our counterfactual budget-planning task.

For this public benchmark, we construct event inputs from holiday metadata and use a training protocol consistent with our Alibaba 1688 experiments. The results are reported in Table 3. CEDAR consistently outperforms the representative time-series baselines across all metrics under the next-5 setting. Specifically, CEDAR reduces MSE from $0.6321$ to $0.5819$ compared with the strongest baseline PETFormer, and also achieves lower MAE and NMSE than both PatchTST and PETFormer. These results suggest that the proposed decomposition between base forecasting and event-driven residual correction is not specific to the Alibaba 1688 dataset, and can also provide benefits in public retail forecasting scenarios where external events influence future demand.

*Table 3. Performance comparison on the public Kaggle Store Sales dataset under the next-5 setting.*

| Metric | CEDAR | PatchTST | PETFormer |

$\downarrow$ | MSE | 0.5819 | 0.6814 | 0.6321 |

$\downarrow$ | MAE | 0.3680 | 0.4112 | 0.4623 |

$\downarrow$ | NMSE | 0.3778 | 0.4425 | 0.4104 |

### 4.6. Online A/B Test Results (RQ4)

To validate the practical effectiveness of CEDAR in real-world budget planning scenarios, we deploy the proposed framework in Alibaba 1688’s online advertising and marketing optimization system and conduct a large-scale A/B test from January 1 to January 30, 2026. The initial deployment successfully engaged 239 cooperative merchants across 245 orders for the prediction service, facilitating a total transaction value of 8.77 million RMB with an initial repurchase rate of 61%.

In the online setting, CEDAR is used to generate decision-conditioned sales simulations under alternative budget allocation strategies, which are then integrated into the platform’s recommendation and planning pipeline. The control group follows the existing production model based on a diffusion-based time series forecasting model with only total budgets, while the treatment group adopts CEDAR-driven budget planning and traffic allocation strategies.

The experimental results demonstrate substantial and consistent improvements. Specifically, merchants in the treatment group achieve a 13%(46,471 vs 41,228) increase in lifetime value (LTV) and a 15% improvement in store-level return on investment (ROI) on average. These gains are observed across both short-term promotional campaigns and long-term operational planning, indicating robust performance under diverse business conditions.

We attribute the observed improvements to two key factors. First, by explicitly modeling action-conditioned state transitions, CEDAR enables more accurate simulation of future demand trajectories under alternative marketing strategies, allowing merchants to proactively adjust budgets before traffic and conversion dynamics materialize. Second, the incorporation of external shock modeling further enhances responsiveness to bursty demand and emerging trends, enabling timely reallocation of marketing resources during critical periods. Together, these capabilities significantly reduce ineffective ad spend and improve the alignment between budget allocation and true market demand.

Overall, the online A/B test in Alibaba 1688 results provide strong empirical evidence that decision-conditioned simulation is not only theoretically well-motivated but also delivers tangible business value at scale. This validates the practical relevance of CEDAR for real-world merchant-facing decision support systems and highlights its potential for broader deployment in large-scale e-commerce platforms.

## 5. Conclusion

In this work, we propose CEDAR, a decision-conditioned sales simulation framework that explicitly decouples endogenous market dynamics from latent external shocks, while treating both merchant actions and product states as first-class modalities. CEDAR consists of two core components: an Action-Interleaved Transformer that models action-conditioned state transitions through interleaved token sequences, and a Residual Correction Module that leverages external event signals to predict and correct the discrepancy between base forecasts and real-world observations. To effectively optimize these two components, we adopt a two-stage training strategy that stabilizes long-horizon rollout and enhances robustness under volatile market conditions. Extensive experiments on the large-scale E-Comm 15-Week benchmark demonstrate consistent improvements over state-of-the-art baselines, while large-scale online A/B tests on the Alibaba 1688 platform further validate the practical value of CEDAR, yielding significant gains in merchant lifetime value and store-level return on investment.

In summary, this work highlights the importance of explicit action-conditioned modeling and external shock disentanglement for reliable what-if analysis and budget planning in e-commerce systems, paving the way for more robust and decision-aware forecasting frameworks in complex, intervention-driven environments.

###### Acknowledgements.

This work was supported in part by the National Natural Science Foundation of China (Grant No. 62506348), the Natural Science Foundation of Anhui Province (Grant No. 2508085QF211), New Generation Artificial Intelligence-National Science and Technology Major Project (Grant No. 2025ZD0122601), the CCF-1688 Yuanbao Cooperation Fund (Grant No. CCF-Alibaba2025005), the National Key R&D Program of China (Grant No. 2023YFF0725001), the National Natural Science Foundation of China (Grant No. 92370204), the Guangdong Basic and Applied Basic Research Foundation (Grant No. 2023B1515120057), the Key-Area Special Project of Guangdong Provincial Ordinary Universities (2024ZDZX1007).

## References

- [1] T. Aksu, G. Woo, J. Liu, X. Liu, C. Liu, S. Savarese, C. Xiong, and D. Sahoo GIFT-eval: a benchmark for general time series forecasting model evaluation. In NeurIPS Workshop on Time Series in the Age of Large Models, Cited by: §1.

- [2] A. F. Ansari, L. Stella, A. C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor, et al. Chronos: learning the language of time series. Transactions on Machine Learning Research. Cited by: §1.

- Carta et al. (2018) S. Carta, A. Medda, A. Pili, D. Reforgiato Recupero, and R. Saia Forecasting e-commerce products prices by combining an autoregressive integrated moving average (arima) model and google trends data. Future Internet 11 (1), pp. 5. Cited by: §1.

- Chen et al. (2021) L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch Decision transformer: reinforcement learning via sequence modeling. External Links: 2106.01345, Link Cited by: §2.2.

- Cleveland et al. (1990) R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning STL: a seasonal-trend decomposition. Journal of official statistics 6 (1), pp. 3–73. Cited by: §1.

- Das et al. (2024) A. Das, W. Kong, R. Sen, and Y. Zhou A decoder-only foundation model for time-series forecasting. In Forty-first International Conference on Machine Learning, Cited by: §1.

- [7] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski MOMENT: a family of open time-series foundation models. Cited by: §1.

- Hu et al. (2024) S. Hu, L. Shen, Y. Zhang, Y. Chen, and D. Tao On transforming reinforcement learning with transformers: the development trajectory. IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12), pp. 8580–8599. Cited by: §1.

- Hu et al. (2025) Z. Hu, Y. Hu, and H. Li Multi-task temporal fusion transformer for joint sales and inventory forecasting in amazon e-commerce supply chain. arXiv preprint arXiv:2512.00370. Cited by: §2.1.

- Huang et al. (2024) L. Huang, B. Dong, and W. Zhang Efficient offline reinforcement learning with relaxed conservatism. IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (8), pp. 5260–5272. Cited by: §2.2.

- Janner et al. (2021) M. Janner, Q. Li, and S. Levine Offline reinforcement learning as one big sequence modeling problem. Advances in neural information processing systems 34, pp. 1273–1286. Cited by: §2.2.

- Kumar et al. (2020) A. Kumar, A. Zhou, G. Tucker, and S. Levine Conservative q-learning for offline reinforcement learning. Advances in neural information processing systems 33, pp. 1179–1191. Cited by: §2.2.

- Levine et al. (2020) S. Levine, A. Kumar, G. Tucker, and J. Fu Offline reinforcement learning: tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643. Cited by: §2.2.

- Lim et al. (2021) B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister Temporal fusion transformers for interpretable multi-horizon time series forecasting. International journal of forecasting 37 (4), pp. 1748–1764. Cited by: §1, §2.1, 2nd item.

- Lin et al. (2024) S. Lin, W. Lin, W. Wu, S. Wang, and Y. Wang Petformer: long-term time series forecasting via placeholder-enhanced transformer. IEEE Transactions on Emerging Topics in Computational Intelligence. Cited by: §1, 4th item.

- Liu et al. (2026) L. Liu, Y. Song, D. Shen, B. Yin, H. Li, Y. Zhang, and C. Wang Rethinking popularity bias in collaborative filtering via analytical vector decomposition. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pp. 879–890. Cited by: §1, §2.2.

- Liu et al. (2023) Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long Itransformer: inverted transformers are effective for time series forecasting. arXiv preprint arXiv:2310.06625. Cited by: §2.1.

- [18] Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long Timer-xl: long-context transformers for unified time series forecasting. In The Thirteenth International Conference on Learning Representations, Cited by: §1, 5th item.

- Meng et al. (2026) J. Meng, R. zhang, W. Wu, R. Zhang, C. Qin, Q. Zhang, Q. Liu, H. Xiong, and C. Wang Turning semantics into topology: llm-driven attribute augmentation for collaborative filtering. External Links: 2602.21099, Link Cited by: §2.1, §3.3.

- Nakamoto et al. (2023) M. Nakamoto, S. Zhai, A. Singh, M. Sobol Mark, Y. Ma, C. Finn, A. Kumar, and S. Levine Cal-ql: calibrated offline rl pre-training for efficient online fine-tuning. Advances in Neural Information Processing Systems 36, pp. 62244–62269. Cited by: §2.2.

- Nelson (1998) B. K. Nelson Time series analysis using autoregressive integrated moving average (arima) models. Academic emergency medicine 5 (7), pp. 739–744. Cited by: §1.

- Nie et al. (2023) Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam A time series is worth 64 words: long-term forecasting with transformers. External Links: 2211.14730, Link Cited by: §1, §2.1, 3rd item.

- Oliveira and Ramos (2024) J. M. Oliveira and P. Ramos Evaluating the effectiveness of time series transformers for demand forecasting in retail. Mathematics 12 (17), pp. 2728. Cited by: §2.1.

- Potapczynski et al. (2024) A. Potapczynski, K. G. Olivares, M. Wolff, A. G. Wilson, D. Efimov, and V. Quenneville-Belair Effectively leveraging exogenous information across neural forecasters. Cited by: §1, §3.4.2.

- Punati et al. (2025) S. B. Punati, S. Kanta, U. B. Cheerala, M. G. Lanjewar, and P. Damacharla Temporal fusion transformer for multi-horizon probabilistic forecasting of weekly retail sales. arXiv preprint arXiv:2511.00552. Cited by: §2.1.

- Qin et al. (2025) C. Qin, X. Chen, C. Wang, P. Wu, X. Chen, Y. Cheng, J. Zhao, M. Xiao, X. Dong, Q. Long, et al. Scihorizon: benchmarking ai-for-science readiness from scientific data to large language models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 5754–5765. Cited by: §2.1.

- Qiu et al. (2017) M. Qiu, F. Li, S. Wang, X. Gao, Y. Chen, W. Zhao, H. Chen, J. Huang, and W. Chu AliMe chat: a sequence to sequence and rerank based chatbot engine. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), R. Barzilay and M. Kan (Eds.), Vancouver, Canada, pp. 498–503. External Links: Link, Document Cited by: §1.

- Rafetseder et al. (2013) E. Rafetseder, M. Schwitalla, and J. Perner Counterfactual reasoning: from childhood to adulthood. Journal of experimental child psychology 114 (3), pp. 389–404. Cited by: §2.2.

- Salinas et al. (2020) D. Salinas, V. Flunkert, J. Gasthaus, and T. Januschowski DeepAR: probabilistic forecasting with autoregressive recurrent networks. International journal of forecasting 36 (3), pp. 1181–1191. Cited by: §1.

- Wang et al. (2026) C. Wang, Y. Song, J. Ye, C. Qin, D. Shen, L. Liu, X. Wang, and Y. Zhang Face: a general framework for mapping collaborative filtering embeddings into llm tokens. Advances in Neural Information Processing Systems 38, pp. 146012–146039. Cited by: §3.3.

- Wang et al. (2021a) C. Wang, H. Zhu, Q. Hao, K. Xiao, and H. Xiong Variable interval time sequence modeling for career trajectory prediction: deep collaborative perspective. In Proceedings of the Web Conference 2021, pp. 612–623. Cited by: §2.2.

- Wang et al. (2021b) C. Wang, H. Zhu, P. Wang, C. Zhu, X. Zhang, E. Chen, and H. Xiong Personalized and explainable employee training course recommendations: a bayesian variational approach. ACM Transactions on Information Systems (TOIS) 40 (4), pp. 1–32. Cited by: §2.1.

- Wang et al. (2023) C. Wang, H. Zhu, C. Zhu, C. Qin, E. Chen, and H. Xiong Setrank: a setwise bayesian approach for collaborative ranking in recommender system. ACM Transactions on Information Systems 42 (2), pp. 1–32. Cited by: §2.1.

- Wang et al. (2018) J. Wang, Z. Wang, J. Li, and J. Wu Multilevel wavelet decomposition network for interpretable time series analysis. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2437–2446. Cited by: §1.

- Wen et al. (2019) Q. Wen, J. Gao, X. Song, L. Sun, H. Xu, and S. Zhu RobustSTL: a robust seasonal-trend decomposition algorithm for long time series. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, pp. 5409–5416. Cited by: §1.

- Wen et al. (2021) Q. Wen, K. He, L. Sun, Y. Zhang, M. Ke, and H. Xu RobustPeriod: time-frequency mining for robust multiple periodicity detection. In Proceedings of the 2021 International Conference on Management of Data (SIGMOD ’21), pp. 205–215. Cited by: §2.1.

- Wen et al. (2020) Q. Wen, Z. Zhang, Y. Li, and L. Sun Fast RobustSTL: efficient and robust seasonal-trend decomposition for time series with complex patterns. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’20), pp. 2203–2213. Cited by: §1.

- Wilder et al. (2019) B. Wilder, B. Dilkina, and M. Tambe Melding the data-decisions pipeline: decision-focused learning for combinatorial optimization. In Proceedings of the AAAI conference on artificial intelligence, Vol. 33, pp. 1658–1665. Cited by: §1.

- Wolff et al. (2024) M. Wolff, K. G. Olivares, B. N. Oreshkin, S. Ruan, S. Yang, A. Katoch, S. Ramasubramanian, Y. Zhang, M. W. Mahoney, D. Efimov, et al. SPADE split peak attention decomposition. In NeurIPS Workshop on Time Series in the Age of Large Models, Cited by: §2.1.

- Woo et al. (2022) G. Woo, C. Liu, D. Sahoo, A. Kumar, and S. Hoi Etsformer: exponential smoothing transformers for time-series forecasting. arXiv preprint arXiv:2202.01381. Cited by: §2.1.

- Wu et al. (2025) D. Wu, F. Qiu, and Y. Xie DoFlow: causal generative flows for interventional and counterfactual time-series prediction. arXiv e-prints, pp. arXiv–2511. Cited by: §2.2.

- Wu et al. (2021) H. Wu, J. Xu, J. Wang, and M. Long Autoformer: decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. W. Vaughan (Eds.), Vol. 34, pp. 22419–22430. External Links: Link Cited by: §2.1.

- Yamagata et al. (2023) T. Yamagata, A. Khalil, and R. Santos-Rodriguez Q-learning decision transformer: leveraging dynamic programming for conditional sequence modelling in offline rl. In International Conference on Machine Learning, pp. 38989–39007. Cited by: §2.2.

- Ye et al. (2024) J. Ye, W. Zhang, K. Yi, Y. Yu, Z. Li, J. Li, and F. Tsung A survey of time series foundation models: generalizing time series representation with large language model. CoRR. Cited by: §1.

- Zhang et al. (2026) R. Zhang, J. Meng, Y. Sun, Z. Xu, B. Yin, H. Li, Y. Zhang, and C. Wang MCLMR: a model-agnostic causal learning framework for multi-behavior recommendation. In Proceedings of the ACM Web Conference 2026, WWW ’26, New York, NY, USA, pp. 6481–6492. External Links: ISBN 9798400723070, Link, Document Cited by: §1, §2.2.

- Zheng et al. (2025) H. Zheng, L. Shen, Y. Luo, D. Ye, S. Xu, B. Du, J. Shen, and D. Tao Value-guided decision transformer: a unified reinforcement learning framework for online and offline settings. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, External Links: Link Cited by: §2.2.

- Zheng et al. (2022) Q. Zheng, A. Zhang, and A. Grover Online decision transformer. In Proceedings of the 39th International Conference on Machine Learning, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.), Proceedings of Machine Learning Research, Vol. 162, pp. 27042–27059. External Links: Link Cited by: §2.2.

- Zhou et al. (2021) H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang Informer: beyond efficient transformer for long sequence time-series forecasting. Proceedings of the AAAI Conference on Artificial Intelligence 35 (12), pp. 11106–11115. External Links: Link, Document Cited by: §1, 1st item.

- Zhou et al. (2025) S. Zhou, H. Schöner, H. Lyu, E. Fouché, and S. Wang BALM-tsf: balanced multimodal alignment for llm-based time series forecasting. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pp. 4498–4508. Cited by: §2.1.

- Zhou et al. (2022) T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin FEDformer: frequency enhanced decomposed transformer for long-term series forecasting. In Proceedings of the 39th International Conference on Machine Learning, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.), Proceedings of Machine Learning Research, Vol. 162, pp. 27268–27286. External Links: Link Cited by: §2.1.

## Appendix A Details of LLM-based Hotspot Information Extraction

In this section, we detail the implementation of the Hotspot Information Extraction module. Raw social media trends are inherently noisy and often dominated by entertainment gossip, which can mislead forecasting models if used directly. To ensure high-quality semantic representations, we design a two-stage LLM prompting pipeline. The first stage filters noise and extracts commerce-relevant tags, while the second stage synthesizes these tags with calendar events into a single, coherent natural language sentence.

### A.1. Stage 1: Noise Filtering and Tag Extraction

For each forecasting window $t$, we first collect raw trending lists $\mathcal{T}_{t}$ from major social platforms (eg.Red notebook and Douyin ) over the past week ($[t-7,t-1]$). In this stage, the LLM acts as an information filter. We prompt it to discard irrelevant news and extract only keywords (tags) indicative of potential product demand.

Let the output of this first stage be a set of filtered commerce tags, denoted as $\mathcal{K}_{t}$.

### A.2. Stage 2: Semantic Synthesis with Calendar Events

Simply embedding a list of isolated tags $\mathcal{K}_{t}$ fails to capture the underlying market narrative and temporal context. Therefore, in the second stage, we fuse $\mathcal{K}_{t}$ with a structured list of upcoming calendar events $\mathcal{E}_{t}$ (e.g., public holidays, shopping festivals) occurring within the target window $t$. The LLM is instructed to synthesize these elements into semantically complete sentences $S_{t}$.

### A.3. Context Embedding Generation

The output of Stage 2 ars fluent sentences $S_{t}$ that encapsulates the macro-environmental factors and market dynamics. Finally, we employ a pre-trained text encoder to transform $S_{t}$ into a dense vector representation:

$\mathbf{h}_{t}=\text{Encoder}(S_{t})\in\mathbb{R}^{d_{h}}$ | | | |

where $d_{h}$ is the embedding dimension. This dense hotspot embedding $\mathbf{h}_{t}$ serves as a critical global exogenous context signal for our framework, empowering the model to adjust its predictions based on a semantic understanding of concurrent external drivers.

## Appendix B Causal Identifiability and Structural Invariance Analysis of CEDAR

From the perspective of Structural Causal Models (SCM), the e-commerce sales system is formalized as a controlled stochastic process. We posit that the underlying data generation process follows an Additive Noise Model defined as:

$S_{t+1}:=f_{\theta}(S_{\leq t},A_{t+1})+\mathcal{E}(Z_{t+1},U_{t})$ | (9) | | | |

where $S$ denotes endogenous states, $A$ represents merchant interventions, $Z$ represents observed exogenous events, and $U$ denotes unobserved noise.

The CEDAR framework leverages this structural equation to achieve an effective approximation of the interventional distribution $P(S_{t+1}\mid do(A_{t+1}),S_{\leq t})$ through architectural causal disentanglement. Specifically: (1) Learning Action-Conditioned Transition Operators: The AIT module in the first stage leverages an interleaved sequence structure to explicitly model the endogenous mechanism $f_{\theta}$. This approximation relies on the assumption of Sequential Ignorability, which posits that potential outcomes $S(a)$ are conditionally independent of the current action given the history $\mathcal{H}_{t}=\{S_{\leq t},Z_{\leq t+1}\}$:

$S_{t+1}(a)\perp\!\!\!\perp A_{t+1}\mid\mathcal{H}_{t},\quad\forall a\in\mathcal{A}$ | (10) | | | |

(2) Orthogonal Decomposition of Exogenous Shocks: The residual correction module in the second stage explicitly captures the exogenous term $\mathcal{E}$. By incorporating LLM-enhanced event semantics $Z$, the model effectively blocks back-door paths induced by latent confounders such as market trends.

This separated modeling of Endogenous Mechanism + Exogenous Perturbation fundamentally ensures the Structural Invariance of the model during Counterfactual Simulation. It enables robust estimation of the marginal causal effects derived from hypothetical budget strategies, thereby overcoming the confounding bias inherent in traditional correlation-based models.

## Appendix C Time Cost Analysis

The wall-clock cost analysis is reported in Table 4. Compared with standard forecasting baselines, CEDAR introduces additional computation mainly from the two-stage training pipeline and the LLM-based event embedding construction. Specifically, Stage I takes about 154 minutes to train, while Stage II adds another 80 minutes for learning the residual correction module. The total training time of CEDAR is therefore approximately 234 minutes, which is higher than PatchTST but substantially lower than PETFormer in our implementation. The event embedding generation requires around 4 hours; however, this step is performed only once as offline preprocessing and the resulting embeddings can be reused across downstream training and inference. Therefore, its cost is amortized over all products, time windows, and subsequent model updates. Considering the consistent improvements in both short-horizon and long-horizon forecasting accuracy, the additional computational overhead is acceptable for large-scale industrial deployment, especially in budget planning scenarios where simulation quality is more critical than one-time preprocessing cost.

*Table 4. Wall-clock cost analysis of CEDAR and representative baselines. The LLM-based event embedding generation is a one-time offline preprocessing cost.*

| Component | Time / Epoch | Epochs | Total Time | Notes |

$\approx$ | CEDAR Stage I | 11 min | 14 | 154 min (2h 34m) | Base-stage training |

$\approx$ | CEDAR Stage II | 16 min | 5 | 80 min (1h 20m) | Residual correction training |

$\approx$ | Event embedding generation | – | – | 4h | One-time offline cost |

$\approx$ | PatchTST | 7 min | 17 | 119 min (1h 59m) | Baseline reference |

$\approx$ | PETFormer | 21 min | 31 | 651 min (10h 51m) | Baseline reference |

## Appendix D Long-horizon Performance Comparison

To further examine rollout stability, we extend the evaluation horizon from next-5 and next-10 to next-15, next-20, and next-25, using the same 10-week historical input. As shown in Table 5, all models exhibit larger errors as the horizon increases, which is expected due to autoregressive error accumulation. Nevertheless, CEDAR degrades more gracefully than the strongest baselines. At next-25, CEDAR obtains an MSE of 1.612, substantially lower than PatchTST 2.847 and PETFormer 2.561. This result suggests that explicitly modeling action-conditioned transitions and event-driven residuals improves not only short-horizon accuracy but also the fidelity of long-horizon counterfactual rollouts.

*Table 5. Long-horizon rollout performance on Alibaba 1688 measured by MSE. All settings use 10 weeks of historical observations as input.*

| Horizon | CEDAR | PatchTST | PETFormer |

| next 5 | 0.182 | 0.424 | 0.434 |

| next 10 | 0.414 | 0.849 | 0.743 |

| next 15 | 0.734 | 1.526 | 1.382 |

| next 20 | 1.103 | 2.173 | 1.945 |

| next 25 | 1.612 | 2.847 | 2.561 |

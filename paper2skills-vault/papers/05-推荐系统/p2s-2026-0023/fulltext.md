<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2608.10240
     paper_id : p2s-2026-0023
     source   : https://arxiv.org/html/2608.10240v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Sequential Modality Dropout for Robust Multi-Modal Sequential RecommendationNote: Accepted at the 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 7–11, 2026, Rome, Italy. This is the authors’ preprint version.

Conference: The 35th ACM International Conference on Information and Knowledge Management; November 7–11, 2026; Rome, ItalyCCS: Information systems Recommender systems
Guanqun Yang email: guanqun.yang@outlook.com Affiliation: Stevens Institute of Technology, Hoboken, NJ, USA and Wenlong Zhang email: wzhang71@stevens.edu Affiliation: Stevens Institute of Technology, Hoboken, NJ, USA

2026

###### Abstract.

Multi-modal sequential recommenders assume every item carries every modality, but real product catalogs often miss images or text, and a model trained on complete data loses much of its recommendation accuracy when a modality is unavailable at serving time. We propose Sequential Modality Dropout (SMD): during training, each modality stream (image and text) is independently erased with probability $p$ for an entire user interaction history, so the model learns to predict the next item without relying on any single modality. We measure robustness by retention, the fraction of a model’s full-modality accuracy (HR@10) that survives when a modality is removed at test time. Across four backbones (MM-SASRec, IISAN, MISSRec, and fMRLRec) on four Amazon domains, SMD raises text retention by 1.0 to 3.2$\times$ at essentially no cost to full-modality accuracy; under an extreme 95% per-item missing rate, it retains 61% of HR@10 versus 22% without (a 2.8$\times$ improvement). An optional cross-modal reconstruction loss further lifts retention from 90% to 98% on a simple additive backbone under severe text missingness. SMD is a four-line, architecture-agnostic change that makes multi-modal sequential recommenders robust to the missing modalities they actually encounter in deployment.

Code: https://github.com/guanqun-yang/SMD

###### Keywords:

sequential recommendation, multi-modality, dropout, robustness

## 1. Introduction

Many recent sequential recommenders incorporate frozen content features, such as product images encoded by CLIP or ViT and text encoded by BERT or Llama, alongside or instead of the learned ID embeddings of SASRec (Kang and McAuley, 2018), to capture preferences that IDs alone miss (Wang et al., 2023; Fu et al., 2024; Wang et al., 2024; Fu et al., 2025a; Hong et al., 2025; Hu et al., 2024; Fu et al., 2025b). These methods are trained and evaluated on benchmarks in which every item carries every modality, but real product catalogs routinely violate this assumption (Fu et al., 2026): 34.3% of Toys & Games and 48.3% of Beauty & Personal Care items have no text description, and 26% to 41% of items have no image on the four MISSRec Amazon domains. Removing text from a vanilla multi-modal SASRec at test time drops HR@10 to 40 to 73% of its full-modality value (Table 1(b)), and under a 95% per-item missing rate retention collapses to 22% (Figure 2).

*Figure 1. Sequential Modality Dropout (SMD). A four-line per-sample Bernoulli modality mask injected at the fusion point of any multi-modal sequential backbone (MM-SASRec (Kang and McAuley, 2018), MISSRec (Wang et al., 2023), IISAN (Fu et al., 2024), fMRLRec (Wang et al., 2024)). At test time, the same trained backbone retains 61% of HR@10 at $p_{\mathrm{miss}}\!=\!0.95$ versus 22% for the unmodified model.*

Missing modalities, however, have been studied almost entirely outside the sequential setting. Within recommendation, the problem is treated as a collaborative-filtering (CF) one: FeatProp (Malitesta et al., 2024) propagates features along an item-to-item co-interaction graph, MMGACL (Zhao et al., 2025) completes features by diffusion with bimodal attention, SiBraR (Ganhör et al., 2024) trains a single-branch encoder on random modality subsets, and LRMM (Wang et al., 2018) applies modality dropout to review-based rating prediction, none of which model the temporal item ordering that sequential recommenders exploit. Outside recommendation, Bernoulli modality masking is a mature training-time tool, used for audiovisual gesture recognition (Neverova et al., 2015), talking-face synthesis (Abdelaziz et al., 2020), medical-image segmentation (Liu et al., 2022), and masked cross-modal projection (Nezakati et al., 2024), but always on a single sample, where no user sequence exists. Neither line covers multi-modal sequential recommendation, a gap that a recent survey of 354 missing-modality papers (Wu et al., 2026) names explicitly as an under-explored temporal setting. We close it by bringing modality masking into sequential recommendation with a design built for the temporal structure of user histories.

We propose Sequential Modality Dropout (SMD; Figure 1), a per-sample Bernoulli modality mask injected at the fusion point of any multi-modal sequential recommender. For each training sample, each modality stream (image and text) is independently zeroed with probability $p$, and the same mask applies to every item in the user’s chronological sequence. This design preserves the temporal signal the sequential Transformer relies on, and matches real catalog missingness, which tends to cluster within a session or a product category (e.g., image-CDN outages) rather than varying independently across items. SMD is also a black-box plug-in: the same module integrates into MM-SASRec, IISAN (Fu et al., 2024), MISSRec (Wang et al., 2023), and fMRLRec (Wang et al., 2024) without per-architecture tuning and leaves the host optimizer and hyperparameters untouched.

Our contributions answer three research questions, examined in turn in Section 3:

-

A universal plug-in across backbones (RQ1). On Amazon Scientific across the four backbones, SMD lifts HR@10 text retention from 18–94% to 56–99% (Table 1(a)); on MM-SASRec across four Amazon domains, it lifts text retention from 40–73% to 79–97% (Table 1(b)). Across all 11 of the 16 possible (backbone, dataset) combinations, the robustness gain is 1.0 to 3.2$\times$ at essentially no cost to full-modality accuracy.

-

Robustness that scales to extreme missingness (RQ2). On a 0 to 95% per-item missing-rate sweep, MM-SASRec with SMD retains 61% of HR@10 at $p_{\mathrm{miss}}\!=\!0.95$ versus 22% for the unmodified model (Figure 2), and per-user paired tests on 121k users confirm the gains are statistically significant (unlikely to arise by chance).

-

An opt-in cross-modal reconstruction loss (RQ3). For simple additive backbones under severe text missingness, an auxiliary loss that trains the two modality projections to predict each other lifts text retention from 90% to 98% on Beauty & Personal Care (48% text-missing; Table 1(c)).

## 2. Sequential Modality Dropout

SMD has three parts, described in turn: the multi-modal backbone it plugs into (Section 2.1), the per-sample modality-masking mechanism at its core (Section 2.2), and an optional cross-modal reconstruction loss for severe missingness (Section 2.3).

### 2.1. Multi-Modal Sequential Backbone

We work with the SASRec backbone (Kang and McAuley, 2018) extended with the additive multi-modal fusion of Equation 1 (we refer to this construction as MM-SASRec, following the multi-modal extension used in MISSRec (Wang et al., 2023) and IISAN (Fu et al., 2024)). Each item $i$ has a learned ID embedding $\mathbf{e}_{i}^{\mathrm{ID}}\in\mathbb{R}^{d}$ and two frozen content vectors: an image embedding $\mathbf{v}_{i}$ from a CLIP visual encoder and a text embedding $\mathbf{t}_{i}$ from a language encoder. Two linear projections $f_{v}$ and $f_{t}$ map the content vectors into the model’s $d$-dimensional hidden space, and the per-item representation is the additive fusion

$\mathbf{e}_{i}=\mathbf{e}_{i}^{\mathrm{ID}}+a_{i}^{v}\,f_{v}(\mathbf{v}_{i})+a_{i}^{t}\,f_{t}(\mathbf{t}_{i}),$ | (1) | | | |

where $a_{i}^{v},a_{i}^{t}\in\{0,1\}$ are catalog availability indicators (a missing modality contributes zero). The sequence $\mathbf{e}_{i_{1}},\ldots,\mathbf{e}_{i_{n-1}}$ is fed to a causal Transformer block stack and a linear output head scores all items in the catalog. The training objective is the standard binary cross-entropy loss with one positive and one sampled negative per position.

### 2.2. Modality Masking

#### Mechanism.

SMD inserts a per-sample Bernoulli mask on the modality streams immediately before fusion (Figure 1). During training, we draw an independent Bernoulli mask for each sample $b$ in the minibatch and each modality $m\in\{v,t\}$,

$m_{b}^{(m)}\sim\mathrm{Bernoulli}(1-p),\quad\tilde{f}_{m}(\mathbf{x}_{i}^{(m)})=m_{b}^{(m)}\cdot f_{m}(\mathbf{x}_{i}^{(m)}),$ | (2) | | | |

and substitute $\tilde{f}_{m}$ for $f_{m}$ in Equation 1. The mask is per-sample, not per-item: all items in user $b$’s sequence share the same modality mask. This matches how modalities go missing in real catalogs, where an entire product category or data source tends to lack a modality at once, rather than an artificial pattern where missing items are scattered evenly through the sequence (for example, a category whose text descriptions are absent in bulk, or an image feed unavailable for a whole batch of items). At test time the mask is not applied, except for the deterministic masks used in our robustness evaluation (Section 3.1).

#### Relation to Standard Dropout.

Unlike standard dropout, which at test time rescales activations by $1/(1-p)$ to preserve their expected magnitude as a regularizer (Srivastava et al., [n. d.]), SMD zeroes a whole modality with no rescaling, following the established modality-dropout convention (Neverova et al., 2015; Abdelaziz et al., 2020). Its goal is invariance to a genuinely missing modality rather than variance reduction, so the full-modality input seen at test is simply the $p\!=\!0$ case the model already encountered during training, and no compensating scale is required.

#### Why Per-Sample.

A natural alternative is a per-item mask that decides separately for each item whether to drop a modality, so some items in a sequence keep both modalities while others lose one. We use the per-sample mask instead because real missingness is whole-modality and structured by category or source rather than independent across items: a modality tends to go absent in a block, the same pattern that produces the cold-start problem (Wang et al., 2018; Ganhör et al., 2024), and in our data the text-missing rate varies sharply across categories (34.3% to 48.3%). A per-item mask would instead train the model for an independent, missing-at-random pattern that is rarely seen in deployment (Wang et al., 2018).

#### Implementation.

The entire mechanism is the four-line modification at the fusion point of the host model shown in Figure 1. We apply the same template at the corresponding line in each of MM-SASRec, IISAN, MISSRec, and fMRLRec; the only per-architecture decision is which tensor in the fusion path carries the unpooled modality embeddings.

### 2.3. Cross-Modal Reconstruction

For backbones whose fusion layer offers no native fallback path (e.g., the simple additive fusion in Equation 1), we additionally consider a small cross-modal reconstruction loss in the spirit of MMP’s masked modality projection (Nezakati et al., 2024)

$\mathcal{L}_{\mathrm{rec}}=\tfrac{1}{2}\!\left(\|g_{t\to v}(f_{t}(\mathbf{t}_{i}))-f_{v}(\mathbf{v}_{i})\|^{2}+\|g_{v\to t}(f_{v}(\mathbf{v}_{i}))-f_{t}(\mathbf{t}_{i})\|^{2}\right),$ | (3) | | | |

where $g_{t\to v}$ and $g_{v\to t}$ are two-layer projections trained jointly with the recommender. The total objective is $\mathcal{L}=\mathcal{L}_{\mathrm{BCE}}+\lambda\,\mathcal{L}_{\mathrm{rec}}$, where we set $\lambda=0.01$ from preliminary runs; a larger $\lambda=0.1$ reduced full-modality accuracy. The intuition is that, when modality $m$ is masked at test time, the model can implicitly recover an estimate of it through the surviving modality. We show in Section 3.4 that this auxiliary loss is worth its accuracy cost only on simple additive backbones in extreme-missingness regimes.

## 3. Experiments

We evaluate SMD on the four-domain MISSRec benchmark, organizing the study around three questions:

Does SMD work across architectures?

How does SMD scale with the missing rate?

Can a cross-modal reconstruction loss boost SMD further?

### 3.1. Setup

#### Datasets.

We used four Amazon domains following the MISSRec benchmark (Wang et al., 2023): Scientific, Instruments, Arts, and Office, spanning 4,385 to 25,986 items, with 26% to 41% image-missing rates measured on our re-downloaded catalog. All splits were chronological, with a maximum sequence length of 10, and full-catalog ranking (each held-out item is scored against every catalog item, not a sampled subset). The reconstruction-loss study also used Beauty & Personal Care from the Amazon Reviews 2023 release, 10-core filtered, with 48.3% of items missing text in our measurement.

#### Backbones.

We evaluated SMD on four backbones spanning the design space: MM-SASRec (additive fusion of Equation 1 on a 1-block hidden-64 SASRec (Kang and McAuley, 2018)), IISAN (Fu et al., 2024) (frozen ViT and BERT encoders adapted by small trainable side-networks, i.e., parameter-efficient fine-tuning), MISSRec (Wang et al., 2023) (a Transformer with dynamic modality fusion, fine-tuned from a 100-epoch cross-domain checkpoint), and fMRLRec (Wang et al., 2024) (an efficient linear-recurrent, state-space sequence model whose Matryoshka embeddings can be truncated to smaller sizes). For each backbone, we trained two checkpoints (one without SMD and one with SMD at $p\!=\!0.3$) using identical hyperparameters, optimizers, and seeds, sharing frozen MISSRec CLIP ViT-B/32 features (512-dim) across backbones; SMD is the only difference between the two.

#### Evaluation Protocols.

A single test-time missingness pattern can over- or under-state robustness, so we use two complementary protocols:

-

Protocol 1 (full-modality removal) zeroes one or both modalities for every item at test time, yielding the conditions full, no_text, no_image, and no_modal; it matches a categorical or systemic outage in which an entire modality is unavailable (Ganhör et al., 2024).

-

Protocol 2 (per-item drop) zeroes each modality of each item independently with probability $p_{\mathrm{miss}}\in[0,0.95]$, averaged over 5 random seeds, matching item-level missingness in real catalogs (Malitesta et al., 2024; Fu et al., 2026).

Together they cover the worst case (every item missing the same modality) and the realistic average case (items missing modalities at random). For each checkpoint we report HR@10 ($\times 100$) and the retention rate $R$; comparing a model trained with versus without SMD gives the robustness gain $G$:

$R(\theta)=\frac{\mathrm{HR@10}(\theta;\,\text{missing})}{\mathrm{HR@10}(\theta;\,\text{full})},\quad G=\frac{R(\theta_{\mathrm{SMD}})}{R(\theta_{\mathrm{No\,SMD}})},$ | (4) | | | |

where $R(\theta)$ is defined for a single trained checkpoint $\theta$ and $G$ compares the two checkpoints trained with matched hyperparameters and seeds.

### 3.2. RQ1: SMD Works Across Architectures

*Table 1. Results (HR@10 $\times 100$). The retention rate is $R=\text{HR@10}(\text{missing})/\text{HR@10}(\text{full})$ for the removed modality, and the robustness gain is $G=R_{\text{SMD}}/R_{\text{No SMD}}$; dropout is $p\!=\!0.3$ unless noted. (a) SMD as a plug-in across four backbones; (b) MM-SASRec across four domains; (c) the optional cross-modal reconstruction loss.*

*(a) Plug-in across four backbones (Scientific, text removed); best text retention per backbone in bold.*

$\boldsymbol{R}_{\text{text}}$ $\boldsymbol{G}$| Backbone | Train. | Full | No txt | | |

| IISAN (Fu et al., 2024) | – | 6.43 | 1.13 | 18% | – |

$3.2\times$| SMD | 6.15 | 3.44 | 56% | |

| MISSRec (Wang et al., 2023) | – | 13.53 | 5.80 | 43% | – |

$1.6\times$| SMD | 13.27 | 9.31 | 70% | |

| fMRLRec (Wang et al., 2024) | – | 4.83 | 4.54 | 94% | – |

$1.1\times$| SMD | 4.79 | 4.75 | 99% | |

| MM-SASRec (Kang and McAuley, 2018) | – | 6.18 | 2.45 | 40% | – |

$2.1\times$| SMD | 6.31 | 5.22 | 83% | |

*(b) MM-SASRec across four domains (text or image removed); the better No-SMD/SMD value per cell in bold.*

$\boldsymbol{R}_{\text{txt}}$ $\boldsymbol{R}_{\text{img}}$| Dataset | Train. | Full | No txt | No img | | |

| Scientific | – | 6.18 | 2.45 | 5.39 | 40% | 87% |

| SMD | 6.31 | 5.22 | 5.93 | 83% | 94% |

| Instruments | – | 7.14 | 5.18 | 6.34 | 73% | 89% |

| SMD | 7.13 | 6.95 | 6.85 | 97% | 96% |

| Arts | – | 4.75 | 2.32 | 3.68 | 49% | 77% |

| SMD | 5.06 | 4.01 | 4.67 | 79% | 92% |

| Office | – | 5.30 | 2.63 | 3.57 | 50% | 67% |

| SMD | 5.62 | 4.62 | 4.89 | 82% | 87% |

*(c) Reconstruction loss (MISSRec on Scientific, MM-SASRec on Beauty); best No-text HR@10 and retention in bold.*

$\boldsymbol{R}_{\text{text}}$| Backbone | Training | Full | No txt | |

| MISSRec | – | 13.53 | 5.80 | 43% |

| SMD | 13.27 | 9.31 | 70% |

$+$ | SMD recon | 12.96 | 9.49 | 73% |

| MM-SASRec | – | 0.72 | 0.31 | 43% |

$p\!=\!0.3$ | SMD () | 0.63 | 0.57 | 90% |

$p\!=\!0.5$ | SMD () | 0.56 | 0.48 | 86% |

$+$ | SMD recon | 0.52 | 0.51 | 98% |

Table 1(a) reports HR@10 under text removal for the four backbones on Amazon Scientific, where text retention without SMD varies most across backbones, from 18% (IISAN) to 94% (fMRLRec). SMD improves text retention in every backbone, with gain $G\in[1.1\times,\,3.2\times]$: the largest lifts are on IISAN (18% to 56%, 3.2$\times$) and MM-SASRec (40% to 83%, 2.1$\times$), while fMRLRec, already 94% retained because its concat-plus-projection fusion learns near-redundant modality embeddings, rises only to 99%.

Table 1(b) reports MM-SASRec when text and when images are removed across four domains: SMD lifts text retention from 40–73% to 79–97% and image retention from 67–89% to 87–96%, while matching or exceeding peak HR@10 on every dataset. Text removal is consistently the harder condition because the text encoder carries product-name information that the image cannot recover.

Beyond these two slices, we ran all 16 (backbone, dataset) combinations and summarize them here, omitting the full table only to respect the page limit: 11 of the 16 improve, with robustness gain 1.0 to 3.2$\times$ and a mean peak-HR@10 change of +0.8% (range -4.4% to +6.5%). The two largest accuracy losses, IISAN/Scientific (-4.4%) and IISAN/Instruments (-4.1%), occur exactly where SMD delivers its largest retention gains (3.2$\times$ and 1.4$\times$).

RQ1 Takeaway. SMD helps on every recommender we tested, regardless of how it fuses image and text; the biggest robustness gains coincide with the biggest full-modality accuracy drops, yet even those drops stay small (at most 4.4%).

### 3.3. RQ2: SMD Scales to Extreme Missingness

*Figure 2. Per-item missing-rate sweep on Scientific (MM-SASRec, mean of 5 seeds). At $p_{\mathrm{miss}}\!=\!0.95$, SMD retains 61% of HR@10 versus 22% without.*

Figure 2 sweeps $p_{\mathrm{miss}}$ from 0% to 95% in 5% increments on Scientific. The two curves diverge monotonically from $p_{\mathrm{miss}}\!=\!0$: the SMD curve is approximately flat (slope $\approx-0.027$ HR@10 per unit $p_{\mathrm{miss}}$), while the No-SMD curve falls steeply (slope $\approx-0.051$); at $p_{\mathrm{miss}}\!=\!0.95$, SMD retains 61% versus 22% for the same model without SMD, a 2.8$\times$ improvement. The same pattern holds on Arts and Office: at 70% per-item drop SMD retains 71–78% versus 46–53% without SMD; we plot only the Scientific sweep (Figure 2) and omit the per-domain tables to respect the page limit.

To rule out user-level noise, we pair each user’s HR@10 and NDCG@10 between the No-SMD and SMD checkpoints across three datasets (121k users). Because HR@10 is binary per user, we test it with McNemar’s test, which weighs how many users SMD flips from miss to hit against the reverse; NDCG@10 is a continuous score (higher when the held-out item is ranked nearer the top), so we use the Wilcoxon signed-rank test, which ranks the per-user differences and assumes no particular distribution. Of the resulting 24 tests (12 (condition, dataset) pairs $\times$ 2 tests), 22 are significant at $\alpha=0.05$, meaning a gap this large is very unlikely if the two models were equivalent, with $p$-values from $4.3\times 10^{-3}$ to $1.0\times 10^{-233}$; the two exceptions are the full-modality McNemar tests on Scientific and Instruments, where SMD flips similar numbers of users each way. We ran all 24 tests and report these aggregates in place of the full table, which the page limit does not permit us to include.

RQ2 Takeaway. Robustness scales with missingness: the gap between SMD and the unmodified model widens monotonically and remains statistically significant across datasets and conditions.

### 3.4. RQ3: A Cross-Modal Reconstruction Loss for Severe Missingness

The auxiliary loss $\mathcal{L}_{\mathrm{rec}}$ (Equation 3) trains the two modality projections to predict each other, so the model can implicitly recover a missing modality at test time. Its value depends sharply on the backbone (Table 1(c)). On a strong dynamic-fusion backbone (MISSRec on Scientific) it adds only 3 points of text retention (70% to 73%) while costing 2.3% of full-modality HR@10, a net loss across most metrics MISSRec’s own results (Wang et al., 2023) report. On a simple additive backbone under severe missingness (MM-SASRec on Beauty & Personal Care, 48% text-missing), it instead lifts retention from 90% to 98%, because the model has no dynamic fusion to fall back on.

RQ3 Takeaway. The reconstruction loss helps only when fusion is simple and missingness is severe; SMD alone is the default, with the loss an opt-in enhancement.

## 4. Conclusion

SMD is a four-line, per-sample training-time mask at the fusion point of any multi-modal sequential recommender, addressing the accuracy loss that missing modalities cause at serving time. Across four backbones and four Amazon domains, it lifts HR@10 text retention by 1.0 to 3.2$\times$ over 11 of 16 combinations at +0.8% mean peak HR@10, and under an extreme 95% per-item missing rate it retains 61% of HR@10 versus 22% without. Per-user tests on 121k users confirm these gains are statistically significant. An optional cross-modal reconstruction loss lifts retention from 90% to 98% on simple additive backbones under severe missingness. Because SMD is a self-contained change at the fusion step, it can be combined with complementary techniques such as feature propagation (Malitesta et al., 2024) and cross-modal reconstruction (Nezakati et al., 2024). Two directions remain open: extending beyond image and text to audio, video, or structured attributes, and handling modality corruption rather than absence.

## GenAI Usage Disclosure

The authors used Claude Code and the Gemini CLI only for technical execution, such as generating code for diagrams and plots, and for polishing the manuscript’s readability. The foundational research, including algorithm design, methodology, and the initial draft, was entirely the authors’ own intellectual work. No AI tools were used to formulate research ideas, create primary data, or execute evaluations.

## References

- Abdelaziz et al. (2020) Ahmed Hussen Abdelaziz, Barry-John Theobald, Paul Dixon, Reinhard Knothe, Nicholas Apostoloff, and Sachin Kajareker. 2020. Modality Dropout for Improved Performance-driven Talking Faces. arXiv:2005.13616 [eess] doi:10.48550/arXiv.2005.13616

- Fu et al. (2026) Junchen Fu, Wenhao Deng, Kaiwen Zheng, Ioannis Arapakis, Yu Ye, Yongxin Ni, Joemon M. Jose, and Xuri Ge. 2026. Benchmarking Multimodal Large Language Models for Missing Modality Completion in Product Catalogues. arXiv:2601.19750 [cs] doi:10.48550/arXiv.2601.19750

- Fu et al. (2024) Junchen Fu, Xuri Ge, Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, Jie Wang, and Joemon M. Jose. 2024. IISAN: Efficiently Adapting Multimodal Representation for Sequential Recommendation with Decoupled PEFT. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 687–697. arXiv:2404.02059 [cs] doi:10.1145/3626772.3657725

- Fu et al. (2025b) Junchen Fu, Yongxin Ni, Joemon M. Jose, Ioannis Arapakis, Kaiwen Zheng, Youhua Li, and Xuri Ge. 2025b. CROSSAN: Towards Efficient and Effective Adaptation of Multiple Multimodal Foundation Models for Sequential Recommendation. arXiv:2504.10307 [cs] doi:10.48550/arXiv.2504.10307

- Fu et al. (2025a) Yongrui Fu, Jian Liu, Tao Li, Zonggang Wu, Shouke Qin, and Hanmeng Liu. 2025a. Multimodal Fusion And Sparse Attention-based Alignment Model for Long Sequential Recommendation. arXiv:2508.09664 [cs] doi:10.48550/arXiv.2508.09664

- Ganhör et al. (2024) Christian Ganhör, Marta Moscati, Anna Hausberger, Shah Nawaz, and Markus Schedl. 2024. A Multimodal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios. In 18th ACM Conference on Recommender Systems. 1290–1295. arXiv:2409.17864 [cs] doi:10.1145/3640457.3688009

- Hong et al. (2025) Ming-Yi Hong, Yen-Jung Hsu, Miao-Chen Chiang, and Che Lin. 2025. MTSTRec: Multimodal Time-Aligned Shared Token Recommender. In Proceedings of the 42nd International Conference on Machine Learning. PMLR, 23640–23661.

- Hu et al. (2024) Jiaxi Hu, Jingtong Gao, Xiangyu Zhao, Yuehong Hu, Yuxuan Liang, Yiqi Wang, Ming He, Zitao Liu, and Hongzhi Yin. 2024. BiVRec: Bidirectional View-based Multimodal Sequential Recommendation. arXiv:2402.17334 [cs] doi:10.48550/arXiv.2402.17334

- Kang and McAuley (2018) Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. arXiv:1808.09781 [cs] doi:10.48550/arXiv.1808.09781

- Liu et al. (2022) Han Liu, Yubo Fan, Hao Li, Jiacheng Wang, Dewei Hu, Can Cui, Ho Hin Lee, Huahong Zhang, and Ipek Oguz. 2022. ModDrop++: A Dynamic Filter Network with Intra-subject Co-training for Multiple Sclerosis Lesion Segmentation with Missing Modalities. arXiv:2203.04959 [eess] doi:10.48550/arXiv.2203.04959

- Malitesta et al. (2024) Daniele Malitesta, Emanuele Rossi, Claudio Pomo, Fragkiskos D. Malliaros, and Tommaso Di Noia. 2024. Dealing with Missing Modalities in Multimodal Recommendation: A Feature Propagation-based Approach. arXiv:2403.19841 [cs] doi:10.48550/arXiv.2403.19841

- Neverova et al. (2015) Natalia Neverova, Christian Wolf, Graham W. Taylor, and Florian Nebout. 2015. ModDrop: Adaptive Multi-Modal Gesture Recognition. arXiv:1501.00102 [cs] doi:10.48550/arXiv.1501.00102

- Nezakati et al. (2024) Niki Nezakati, Md Kaykobad Reza, Ameya Patil, Mashhour Solh, and M. Salman Asif. 2024. MMP: Towards Robust Multi-Modal Learning with Masked Modality Projection. arXiv:2410.03010 [cs] doi:10.48550/arXiv.2410.03010

- Srivastava et al. ([n. d.]) Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. [n. d.]. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. ([n. d.]).

- Wang et al. (2018) Cheng Wang, Mathias Niepert, and Hui Li. 2018. LRMM: Learning to Recommend with Missing Modalities. arXiv:1808.06791 [cs] doi:10.48550/arXiv.1808.06791

- Wang et al. (2023) Jinpeng Wang, Ziyun Zeng, Yunxiao Wang, Yuting Wang, Xingyu Lu, Tianxiang Li, Jun Yuan, Rui Zhang, Hai-Tao Zheng, and Shu-Tao Xia. 2023. MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation. In Proceedings of the 31st ACM International Conference on Multimedia. 6548–6557. arXiv:2308.11175 [cs] doi:10.1145/3581783.3611967

- Wang et al. (2024) Yueqi Wang, Zhenrui Yue, Huimin Zeng, Dong Wang, and Julian McAuley. 2024. Train Once, Deploy Anywhere: Matryoshka Representation Learning for Multimodal Recommendation. arXiv:2409.16627 [cs] doi:10.48550/arXiv.2409.16627

- Wu et al. (2026) Renjie Wu, Hu Wang, Hsiang-Ting Chen, and Gustavo Carneiro. 2026. Deep Multimodal Learning with Missing Modality: A Survey. arXiv:2409.07825 [cs] doi:10.48550/arXiv.2409.07825

- Zhao et al. (2025) Wenqian Zhao, Kai Yang, Peijin Ding, Ce Na, and Wen Li. 2025. Graph Attention Contrastive Learning with Missing Modality for Multimodal Recommendation. Knowledge-Based Systems 311 (Feb. 2025), 113035. doi:10.1016/j.knosys.2025.113035

## Appendix A Dataset Statistics

Table 2 reports the per-domain catalog size, total number of interactions, and image-missing rate of our re-downloaded MISSRec catalog used in the main experiments; these numbers differ from the coverage figures in MISSRec’s original Table 1 (Scientific 26.75%, Pantry 93.65%, Instruments 63.12%, Arts 44.90%, Office 63.99%) because we re-downloaded images directly from the Amazon Reviews 2023 release rather than reusing the MISSRec-provided archives.

*Table 2. MISSRec benchmark datasets. “Image-miss.” is the fraction of catalog items whose image is unavailable in our re-download from the Amazon Reviews 2023 release; the Office rate was not recorded at download time and is marked “n/a”.*

| Dataset | # Items | # Inter. | Image-miss. |

| Scientific | 4,385 | 51 k | 32.8% |

| Instruments | 9,964 | 134 k | 25.7% |

| Arts | 21,019 | 259 k | 40.9% |

| Office | 25,986 | 310 k | n/a |

## Appendix B Full Plug-In Robustness Results

Table 3 reports the full SMD plug-in results across the four backbones and three Amazon domains referenced in Section 3.2. The condensed Scientific-only view in Table 1(a) of the main paper is a slice of this table.

*Table 3. Full SMD plug-in results across four backbones and three Amazon domains. HR@10$\times 100$. Best $R_{\text{text}}$ per pair in bold. Arts runs are omitted for IISAN, MISSRec, and fMRLRec for the same compute-budget reason; the MM-SASRec/Arts cell is the $11$th cell in our main claim and appears in Table 1(b). Office runs are omitted for MISSRec and fMRLRec because their training cost on the larger Office catalog ($25{,}986$ items, $310$k interactions) is prohibitive at our compute budget; the trend holds on Office for IISAN ($34\%\!\to\!77\%$) and MM-SASRec ($50\%\!\to\!82\%$).*

$\boldsymbol{R}_{\text{text}}$ $\boldsymbol{G}$| Backbone | Dataset | Train. | Full | No txt | | |

| IISAN (Fu et al., 2024) | Scientific | – | 6.43 | 1.13 | 18% | – |

$3.2\times$| SMD | 6.15 | 3.44 | 56% | |

| Instruments | – | 8.74 | 5.17 | 59% | – |

$1.4\times$| SMD | 8.38 | 6.87 | 82% | |

| Office | – | 6.46 | 2.19 | 34% | – |

$2.3\times$| SMD | 6.59 | 5.05 | 77% | |

| MISSRec (Wang et al., 2023) | Scientific | – | 13.53 | 5.80 | 43% | – |

$1.6\times$| SMD | 13.27 | 9.31 | 70% | |

| Instruments | – | 12.92 | 7.10 | 55% | – |

$1.4\times$| SMD | 12.85 | 9.71 | 76% | |

| fMRLRec (Wang et al., 2024) | Scientific | – | 4.83 | 4.54 | 94% | – |

$1.1\times$| SMD | 4.79 | 4.75 | 99% | |

| Instruments | – | 5.07 | 4.82 | 95% | – |

$1.0\times$| SMD | 5.27 | 5.05 | 96% | |

| MM-SASRec (Kang and McAuley, 2018) | Scientific | – | 6.18 | 2.45 | 40% | – |

$2.1\times$| SMD | 6.31 | 5.22 | 83% | |

| Instruments | – | 7.14 | 5.18 | 73% | – |

$1.3\times$| SMD | 7.13 | 6.95 | 97% | |

| Office | – | 5.30 | 2.63 | 50% | – |

$1.7\times$| SMD | 5.62 | 4.62 | 82% | |

## Appendix C Per-Item Missingness Curve

Table 4 reports the full per-item missing-rate sweep on Scientific (MM-SASRec) used to draw Figure 2 of the main paper, with values at every $5\%$ step. Table 5 and Table 6 report coarser four-point sweeps on Arts and Office ($p_{\mathrm{miss}}\in\{0.10,\,0.30,\,0.50,\,0.70\}$), which show the same qualitative trend: at $p_{\mathrm{miss}}\!=\!0.70$, SMD retains $71$–$78\%$ of HR@10 versus $46$–$53\%$ without SMD.

*Table 4. Full per-item missing-rate sweep on Scientific (MM-SASRec, mean of $5$ seeds). HR@10$\times 100$.*

$\boldsymbol{p_{\text{miss}}}$ $\boldsymbol{R}$ $\boldsymbol{R}$| | No SMD HR | SMD HR | (No SMD) | (SMD) |

| 0% | 6.22 | 6.55 | 100% | 100% |

| 5% | 5.95 | 6.46 | 96% | 99% |

| 10% | 5.68 | 6.31 | 91% | 96% |

| 15% | 5.43 | 6.22 | 87% | 95% |

| 20% | 5.22 | 6.15 | 84% | 94% |

| 25% | 5.01 | 6.01 | 81% | 92% |

| 30% | 4.80 | 5.91 | 77% | 90% |

| 35% | 4.57 | 5.76 | 74% | 88% |

| 40% | 4.32 | 5.66 | 69% | 86% |

| 45% | 4.10 | 5.56 | 66% | 85% |

| 50% | 3.85 | 5.43 | 62% | 83% |

| 55% | 3.58 | 5.29 | 58% | 81% |

| 60% | 3.30 | 5.13 | 53% | 78% |

| 65% | 3.04 | 4.97 | 49% | 76% |

| 70% | 2.77 | 4.81 | 44% | 73% |

| 75% | 2.47 | 4.65 | 40% | 71% |

| 80% | 2.20 | 4.48 | 35% | 68% |

| 85% | 1.92 | 4.39 | 31% | 67% |

| 90% | 1.61 | 4.20 | 26% | 64% |

| 95% | 1.36 | 4.00 | 22% | 61% |

*Table 5. Per-item missing-rate sweep on Arts (MM-SASRec, mean of $5$ seeds). HR@10$\times 100$.*

$\boldsymbol{p_{\text{miss}}}$ $\boldsymbol{R}$ $\boldsymbol{R}$| | No SMD HR | SMD HR | (No SMD) | (SMD) |

| 0% | 4.82 | 5.17 | 100% | 100% |

| 10% | 4.56 | 5.02 | 95% | 97% |

| 30% | 3.99 | 4.78 | 83% | 92% |

| 50% | 3.32 | 4.44 | 69% | 86% |

| 70% | 2.54 | 4.05 | 53% | 78% |

*Table 6. Per-item missing-rate sweep on Office (MM-SASRec, mean of $5$ seeds). HR@10$\times 100$.*

$\boldsymbol{p_{\text{miss}}}$ $\boldsymbol{R}$ $\boldsymbol{R}$| | No SMD HR | SMD HR | (No SMD) | (SMD) |

| 0% | 5.30 | 5.62 | 100% | 100% |

| 10% | 4.90 | 5.43 | 92% | 97% |

| 30% | 4.13 | 5.03 | 78% | 89% |

| 50% | 3.29 | 4.55 | 62% | 81% |

| 70% | 2.46 | 3.98 | 46% | 71% |

## Appendix D Per-User Paired Significance

Table 7 asks a simple question: for each of the $121$k users in our test set, does SMD produce statistically better HR@10 and NDCG@10 than the unmodified model? On $22$ of the $24$ (condition, dataset) combinations, the answer is yes at $p<0.05$ (McNemar’s exact test for HR@10; Wilcoxon signed-rank for NDCG@10). The large user count makes this a strong test even though we retrain each model only once. The two combinations that miss $p<0.05$ are both HR@10 comparisons under full modality, on Scientific and Instruments. HR@10 is a binary metric (correct item in the top-$10$ or not), and on these two cells roughly equal numbers of users flip miss-to-hit and hit-to-miss under SMD, so the HR@10 differences cancel out at the population level. The corresponding NDCG@10 tests on the same two cells still pass at $p\leq 7\times 10^{-3}$: NDCG rewards ranking the correct item higher within the top-$10$, and SMD does so on average even when it does not push new items across the top-$10$ boundary.

*Table 7. Per-user paired tests on MM-SASRec. Wilcoxon $p$-value on NDCG@10, McNemar’s exact $p$ on HR@10. HR/NDCG $\times 100$. Bold $p$ are significant at $\alpha\!=\!0.05$.*

$\boldsymbol{p}$ $\boldsymbol{p}$| Dataset | Cond. | HR@10 | NDCG@10 | McN. | Wilc. |

| | | – | SMD | – | SMD | | |

| Sci. (8 442) | full | 5.79 | 6.09 | 3.21 | 3.57 | 0.24 | 4.3e-3 |

| no img | 4.92 | 5.54 | 2.55 | 3.31 | 9.0e-3 | 1.6e-8 |

| no txt | 2.61 | 4.63 | 1.50 | 2.68 | 9.6e-18 | 2.2e-18 |

| no mod. | 1.87 | 3.38 | 0.85 | 1.84 | 4.4e-12 | 2.3e-14 |

| Inst. (24 962) | full | 7.07 | 7.23 | 4.27 | 4.55 | 0.24 | 6.8e-3 |

| no img | 6.79 | 7.12 | 3.64 | 4.14 | 1.8e-2 | 8.3e-7 |

| no txt | 5.52 | 6.83 | 3.37 | 4.28 | 6.1e-23 | 1.5e-26 |

| no mod. | 2.81 | 6.03 | 1.43 | 3.48 | 1.7e-105 | 1.3e-96 |

| Off. (87 346) | full | 5.16 | 5.43 | 3.21 | 3.51 | 7.2e-5 | 1.2e-12 |

| no img | 3.71 | 4.85 | 2.03 | 3.02 | 5.2e-59 | 6.7e-104 |

| no txt | 2.37 | 4.49 | 1.28 | 2.82 | 1.5e-211 | 1.0e-233 |

| no mod. | 1.28 | 3.15 | 0.62 | 1.79 | 1.1e-198 | 2.8e-183 |

## Appendix E Cross-Modal Reconstruction Results

Table 1(c) gives the full cross-modal reconstruction results referenced in Section 3.4. The MISSRec configuration uses $\lambda\!=\!0.01$; we found $\lambda\!=\!0.1$ to hurt accuracy.

$\boldsymbol{R}_{\text{text}}$| Backbone | Training | Full | No txt | |

| MISSRec | – | 13.53 | 5.80 | 43% |

| SMD | 13.27 | 9.31 | 70% |

$+$ | SMD recon | 12.96 | 9.49 | 73% |

| MM-SASRec | – | 0.72 | 0.31 | 43% |

$p\!=\!0.3$ | SMD () | 0.63 | 0.57 | 90% |

$p\!=\!0.5$ | SMD () | 0.56 | 0.48 | 86% |

$+$ | SMD recon | 0.52 | 0.51 | 98% |

## Appendix F Implementation

The SMD module is the same $4$-line block in every host model; only the tensor names differ:

| Architecture | Injection point | Host repository |

| MM-SASRec | after image_proj/text_proj, before additive fusion | github.com/kang205/SASRec |

| IISAN | dataset __getitem__, on cached ViT/BERT tensors | github.com/GAIR-Lab/IISAN |

| MISSRec | after text_adaptor/img_adaptor, before fusion | github.com/gimpong/MM23-MISSRec |

$+$ | fMRLRec | after token_lang/token_img, before concatprojection | github.com/yueqirex/fMRLRec |

Source code for SMD and the baseline plug-in modifications, training scripts, and the JSON outputs backing every number in this paper are released at https://github.com/guanqun-yang/SMD.

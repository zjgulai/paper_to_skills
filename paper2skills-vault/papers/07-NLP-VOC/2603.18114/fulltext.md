<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2603.18114
     paper_id : 2603.18114
     source   : paper2skills-vault/papers/nlp_voc/2603.18114/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

Transfer Learning for Contextual Joint Assortment-Pricing under Cross-Market Heterogeneity arXiv:2603.18114v1 [stat.ME] 18 Mar 2026

Elynn Chen♯ ♯⋄ †

Xi Chen ⋄

Yi Zhang†

New York University, Stern School of Business

Tsinghua University, School of Economics and Management

March 20, 2026

Abstract

We study transfer learning for contextual joint assortment-pricing under a multinomial logit choice model with bandit feedback. A seller operates across multiple related markets and observes only posted prices and realized purchases.
While data from source markets can accelerate learning in a target market, cross-market differences in customer preferences may introduce systematic bias if pooled indiscriminately.
We model heterogeneity through a structured utility shift, where markets share a common contextual utility structure but differ along a sparse set of latent preference coordinates. Building on this, we develop Transfer Joint Assortment-Pricing (TJAP), a bias-aware framework that combines aggregatethen-debias estimation with a UCB-style policy. TJAP constructs two-radius confidence bounds that separately capture statistical uncertainty and transferinduced bias, uniformly over continuous prices.
 q √  T e d + s T , We establish matching minimax regret bounds of order O 0 1+H revealing a transparent variance-bias tradeoff: transfer accelerates learning along shared preference directions, while heterogeneous components impose an irreducible adaptation cost. Numerical experiments corroborate the theory, showing that TJAP outperforms both target-only learning and naive pooling while remaining robust to cross-market differences.
Keywords: transfer learning; joint assortment-pricing; revenue management; contextual multinomial logit bandits; minimax regret.

1

1

Introduction

Digital platforms and retailers increasingly operate across multiple related markets. Firms routinely launch the same product lines in different cities, expand sequentially into new regions, or run repeated market-level experiments to refine pricing and assortment decisions.
Such multi-market deployments and repeated experimentation are central to modern datadriven revenue management (e.g., Ban & Keskin (2021); Bastani et al. (2022)). In each new deployment, managers face a fundamental tension: decisions must be made quickly under substantial uncertainty, yet the firm often possesses extensive historical data from prior markets that may be similar but not identical. Leveraging prior data or past experiments can significantly improve learning efficiency (e.g., Bu et al. (2020);Bastani (2021)), but cross-market heterogeneity in customer preferences can limit its effectiveness. In particular, naive pooling across markets may introduce systematic bias and lead to distorted decisions when markets differ in economically meaningful ways.
This challenge is especially acute in joint assortment-pricing problems, where firms must simultaneously determine which products to offer and what prices to charge. In many retail and platform settings, these decisions are intrinsically coupled: assortments impose discrete combinatorial constraints, prices are continuous decision variables, and demand depends nonlinearly on both through substitution effects. Moreover, firms typically observe only bandit feedback – posted prices and realized purchases – rather than full demand curves.
As a result, errors in learning customer preferences propagate through both pricing and assortment decisions, amplifying their impact on revenue outcomes.
While incorporating data from related markets can substantially reduce estimation uncertainty, doing so without explicitly accounting for heterogeneity risks introducing persistent bias. This leads to a fundamental question:
“When and how can data from related markets safely accelerate contextual joint assortment-pricing under discrete-choice demand with bandit feedback?” A large literature in revenue management studies learning and optimization under discretechoice demand models, particularly the multinomial logit (MNL) model, developing algorithms and performance guarantees for dynamic pricing, assortment selection, and their contextual extensions (Keskin & Zeevi 2014, Javanmard et al. 2020, Agrawal et al. 2017, Chen et al. 2020, Oh & Iyengar 2021). These works, however, typically treat each market independently and do not leverage cross-market information in a principled way. In parallel, the statistics and machine learning literature has established that transfer learning and multitask learning can substantially improve sample efficiency when tasks share structure (Bastani 2021, Li et al. 2022, Tian & Feng 2023, Bastani et al. 2022, Xu & Bastani 2024, 2025). Yet these approaches largely focus on linear or full-feedback settings and do not accommodate the discrete-choice structure and joint pricing–assortment decisions central to revenue management.
2

As a result, despite the prevalence of multi-market deployments in practice, there is currently no framework that enables safe and theoretically grounded transfer for contextual joint assortment-pricing under discrete-choice demand with bandit feedback.

Structured Transfer in Joint Assortment-Pricing We study a multi-market setting under a contextual multinomial logit model, consisting of one target market and multiple source markets. Rather than assuming markets are identical or arbitrarily different, we model cross-market heterogeneity through a structured preference shift: markets share a common contextual utility structure, while deviations relative to the target market are confined to a sparse set of preference coordinates.
This formulation reflects a realistic operational perspective. In practice, markets often differ along a limited number of salient dimensions, such as price sensitivity for specific product categories or affinity toward certain attributes, while remaining broadly aligned elsewhere. The sparsity structure captures such localized heterogeneity while preserving sufficient structure for statistical efficiency.
Importantly, these heterogeneous coordinates are not known a priori, but are learned from data in a data-driven manner through the estimation procedure. The sparsity structure therefore serves as a statistical regularity that enables the model to identify and adapt to these differences while maintaining efficiency in high dimensions. It yields a natural separation between shared preference directions, where source data can safely reduce uncertainty, and shifted coordinates, where target-specific adaptation is unavoidable.

A Bias-Aware Transfer and Decision Framework Building on this structure, we develop Transfer Joint Assortment-Pricing (TJAP), a unified learning-and-decision framework for contextual joint assortment-pricing across heterogeneous markets. The design of TJAP follows a simple principle:
“Information shared across markets should be pooled to reduce variance; deviations specific to the target market must be isolated to prevent bias.” To implement this principle, TJAP integrates three components.
First, we propose an aggregate-then-debias estimation procedure. Source-market data are pooled to estimate shared preference components, thereby reducing estimation variance.
Because source markets may differ from the target along a sparse set of coordinates, the pooled estimator is then refined using target-market data via an ℓ1 -regularized debiasing step that corrects for sparse deviations.
Second, we develop a bias-aware optimistic decision rule. We construct frequentist UCB-style confidence bounds that incorporate a two-radius structure: one radius captures statistical uncertainty shrinking with pooled information, while the other accounts for 3

residual transfer bias due to heterogeneity. These bounds are designed to hold uniformly over continuous prices, enabling their integration into the joint assortment-pricing problem.
Third, we incorporate episodic information-geometry control to stabilize learning under adaptive decisions. Because pricing and assortment choices influence the informativeness of future observations, TJAP freezes the information geometry within each episode and invokes targeted exploration only when necessary to ensure identifiability in the target market.
Together, these components yield a unified framework that balances variance reduction through transfer with bias control through target adaptation.

Contributions This paper makes four primary contributions.
First, we introduce a structured transfer-learning formulation for contextual joint assortment-pricing under multinomial logit demand. We model cross-market heterogeneity through sparse preference shifts, in which markets share a common contextual utility structure while differing along a small number of latent preference coordinates. This formulation captures realistic multi-market variation while enabling principled information sharing across markets.
Second, we develop a bias-aware learning-and-decision framework, Transfer Joint AssortmentPricing (TJAP), that explicitly separates variance reduction from bias control. The framework integrates an aggregate–then–debias estimation procedure with a two-radius, priceuniform optimistic policy and episodic information-geometry control. This design enables effective reuse of source-market data while preventing negative transfer arising from structural heterogeneity, and accommodates the joint optimization of discrete assortments and continuous prices under bandit feedback.
Third, we establish finite-time regret guarantees that reveal a transparent variance-bias decomposition in transfer learning. The regret scales as r e d O

√ T + s0 T 1+H

!
,

where H is the number of source markets, d is the feature dimension, and s0 measures the sparsity of cross-market preference shifts. The first term captures variance reduction from transfer, while the second term reflects an irreducible adaptation cost along heterogeneous coordinates.
Fourth, we establish a matching minimax lower bound, showing that this variance–bias tradeoff is fundamental. In particular, no policy can improve the dependence on H or eliminate the s0 -driven term over the structured preference-shift class. Together, these results precisely characterize the value of transfer: when heterogeneity is sparse, transfer 4

yields substantial gains; when heterogeneity is diffuse, these gains saturate; and bias-aware correction is essential for safe data reuse. This yields the first characterization of transfer limits in joint assortment-pricing.
Beyond formal guarantees, our results provide operational insights for multi-market revenue management. The variance-bias decomposition clarifies when historical markets should be pooled and when caution is required. When cross-market differences are localized, leveraging auxiliary markets can significantly accelerate learning and improve both pricing and assortment decisions. In contrast, indiscriminate aggregation without bias correction can lead to systematically distorted decisions under structural heterogeneity. Our numerical experiments corroborate these insights, showing that TJAP consistently outperforms both target-only learning and naive pooling while remaining robust to cross-market differences.

1.1

Related Work and Our Distinction

Our paper lies at the intersection of online revenue management under discrete-choice demand, bandit learning for structured decision problems, and transfer learning across related environments. We connect these literatures in a setting that is both practically important and technically challenging: contextual joint assortment-pricing under multinomial logit demand with bandit feedback and cross-market heterogeneity.
Online pricing and assortment under discrete-choice demand. Choice-based revenue management studies pricing and assortment decisions under random-utility models, most prominently the multinomial logit (MNL) model for its tractability and substitution structure (Talluri & Van Ryzin 2006, Kök et al. 2008). In static settings, joint assortmentpricing under MNL admits tractable structure and has been extended to richer behavioral models (Wang 2012, Aouad et al. 2018, 2021, Gao et al. 2021, Najafi et al. 2025). Related work also considers assortment and pricing without explicit demand prediction or under alternative feedback models (Chen et al. 2023, Chen, Cire, Gao & Wang 2025).
In online settings, most studies treat assortment learning and pricing learning separately. For MNL assortment bandits, both Bayesian and UCB-style approaches establish finite-time regret guarantees in non-contextual and contextual formulations (Agrawal et al.
2017, 2018, Cheung & Simchi-Levi 2017, Chen et al. 2020, Oh & Iyengar 2019, 2021, Ou et al. 2018, Perivier & Goyal 2022). Dynamic pricing under demand uncertainty, including high-dimensional and personalized settings, has also been extensively studied (Keskin & Zeevi 2014, Ban & Keskin 2021, den Boer & Keskin 2022, Chen, Liu & Wu 2025). However, online learning for joint assortment-pricing remains largely confined to single-market formulations (Miao & Chao 2021, Chen et al. 2021, Erginbas et al. 2025, Kim & Oh 2025, Chen & Shi 2019, Jia et al. 2024, Feng & Zhu 2023), and does not address principled cross-market information reuse.

5

Bandits and structured exploration. Contextual bandits provide the methodological foundation for learning with covariates, with sharp regret guarantees for linear and generalized linear reward models using self-normalized confidence bounds (Dani et al. 2008, Rusmevichientong & Tsitsiklis 2010, Abbasi-Yadkori et al. 2011, Filippi et al. 2010, Li et al. 2017, Chen, Chen, Jing & Liu 2025). Combinatorial bandits extend these tools to constrained action spaces (Chen et al. 2013, Kveton et al. 2015). Within discrete-choice bandits, MNL-based methods exploit logit structure to handle combinatorial feasibility while achieving tight regret bounds (Agrawal et al. 2018, Chen et al. 2020, Oh & Iyengar 2021).
Our approach builds on the frequentist UCB paradigm, which has proven effective in balancing exploration and exploitation in classical, linear, and generalized linear bandits (Auer 2002, Lattimore & Szepesvári 2020, Chu et al. 2011, Li et al. 2017, Oh & Iyengar 2021, Erginbas et al. 2025). We extend this framework to joint assortment-pricing by developing confidence bounds that are uniform over prices and remain valid under discrete-choice substitution effects. Moreover, our optimism explicitly accounts for transfer-induced bias, which is not present in existing assortment-only or pricing-only bandit models.
Transfer, multitask, and offline-to-online learning across markets. A parallel literature studies transfer and multitask learning under structured heterogeneity. In supervised settings, debiasing and robust procedures exploit sparse task differences or distributional shifts (Bastani 2021, Li et al. 2022, Tian & Feng 2023, Liu et al. 2023, Huang et al.
2025, Zeng et al. 2024, Chai, Liu, Chen & Yan 2026). In sequential decision problems, multitask and meta-learning frameworks analyze information sharing across tasks (Bastani et al. 2022, Xu & Bastani 2025, Huang et al. 2025, Repasky et al. 2024, Kim et al. 2025, Chen, Li & Jordan 2025, Chai, Chen & Yang 2025, Chai, Chen & Fan 2025, Zhou et al. 2025, Chai, Zhang, Chen & Yan 2026). Related work in pricing studies offline-to-online reuse and meta dynamic pricing across experiments (Bu et al. 2020, 2025, Bastani et al. 2022, Han et al. 2025, Zhang, Zhu & Xie 2025, Zhang, Chen & Yan 2025). These approaches primarily focus on pricing-only or linear-demand models and do not accommodate joint assortment interactions under discrete-choice demand, nor do they explicitly address bias arising from cross-market heterogeneity.
Our Distinction. To our knowledge, this paper is the first to study safe transfer across heterogeneous markets in contextual joint assortment-pricing under multinomial logit demand with bandit feedback. Our contribution is not simply to combine ingredients from these literatures, but to develop a framework tailored to the joint pricing-assortment setting. Specifically, we contribute: (i) a structured multi-market model with sparse utility shifts that makes cross-market similarity statistically exploitable without assuming identical markets; (ii) a bias-aware aggregate-then-debias estimation strategy that separates shared 6

learning from target-specific adaptation; (iii) a frequentist, UCB-style optimistic policy with confidence bounds that are uniform over prices and explicitly account for transferinduced bias; and (iv) matching minimax regret bounds that identify the fundamental variance–bias tradeoff governing transfer in this setting.
Together, these results provide a unified characterization of when transfer improves learning, when its benefits saturate, and why bias-aware correction is essential for reliable data reuse in joint assortment-pricing.
Organization. Section 2 formulates the multi-market contextual joint assortment-pricing problem. Section 3 presents the proposed algorithm. Section 4 establishes theoretical guarantees. Section 5 reports numerical experiments. Section 6 presents the key steps of the proofs. Section 7 concludes and discusses future research directions.

2

Multi-Market Joint Assortment-Pricing Problem

We consider a seller operating across multiple related markets, such as geographic locations, store formats, or platform environments. The seller’s primary objective is to optimize revenue in a designated target market, indexed by superscript (0), which may correspond to a newly launched or strategically important market with limited historical data. In addition, the seller has access to data from H source markets, indexed by (h) for h ∈ [H], which represent previously operated or concurrently running markets with richer data histories. While decisions are made only in the target market, data from source markets may be leveraged to accelerate learning, provided cross-market differences are properly accounted for.
Time is discrete and indexed by t = 1, . . . , T . In each period, customers arrive in every (h)
market together with observable contextual information. For market h, let xit ∈ Rd denote the covariate vector associated with product i at time t, capturing observable product attributes, customer characteristics, or their interactions. These covariates are observed prior to decision making.
The seller makes decisions only in the target market. At each time t, after observing (0)
(0)
(0)
{xit }i∈[N ] , the seller selects an assortment St ⊆ [N ] satisfying |St | ≤ K, and posts (0)
(0)
prices pit for each i ∈ St . Products not included in the assortment are unavailable for purchase. In contrast, source markets generate data according to their own assortment and pricing policies, which are not controlled by the seller in our formulation. This asymmetric structure reflects settings in which firms deploy new pricing and assortment policies in a focal market while relying on historical data from related markets.
After observing the offered assortment and prices, the customer in each market either purchases one of the available products or chooses the outside option. The seller observes

7

realized purchases and revenues but does not observe latent utilities or counterfactual demand. Learning therefore proceeds under bandit feedback.
The objective is to design a policy that maximizes cumulative expected revenue in the target market over horizon T . Performance is evaluated relative to a clairvoyant benchmark that knows the true target-market demand parameters and, in each period, selects the revenue-maximizing assortment and prices given the realized context.
This multi-market formulation makes data reuse intrinsic to the learning problem. The central challenge is to leverage information from related markets to reduce estimation variance while remaining robust to systematic differences in customer preferences across markets.

2.1

Choice Model and Revenue Structure

We model customer demand in each market using a contextual multinomial logit (MNL)
specification. For market h ∈ {0} ∪ [H], product i, and period t, the latent utility takes the form (h)
(h)
(h)
(h)
(h)
(1)
fit = ⟨xit , θ (h) ⟩ − ⟨xit , γ (h) ⟩pit + εit , (h)

where xit ∈ Rd is the observed feature vector, θ (h) captures baseline preference for product (h)
attributes, γ (h) captures price sensitivity, and εit are i.i.d. standard Gumbel shocks.
(h)
(h)
Under this model, when assortment St and prices pt are offered in market h, the (h)
probability that a customer in market h purchases product i ∈ St is (h)

(h)
(h)
(h)
qt (i | St , pt ) =

exp(vit )
, P (h)
1 + ℓ∈S (h) exp(vℓt )

(h)

(h)

(h)

(h)

vit := ⟨xit , θ (h) ⟩ − ⟨xit , γ (h) ⟩pit ,

(2)

t

where the outside option has normalized utility zero.
Given assortment S and price vector p, the expected revenue in market h at time t is (h)

Rt (S, p) :=

X

(h)

pi qt (i | S, p).

(3)

i∈S (0)

In our setting, the seller optimizes Rt (S, p) in the target market, while source markets follow the same structural demand model and generate data for learning.
A useful structural property of the MNL model is that optimal prices are uniformly bounded. Under positive price sensitivity, the revenue-maximizing price for any product is finite and depends only on primitive problem constants. Accordingly, without loss of generality, we restrict attention to prices in a compact interval [0, P̄ ] for some finite P̄ (see Lemma 23 in the Appendix).
Together, the shared MNL structure and bounded price domain define a common revenue landscape across markets, while allowing market-specific parameters ν (h) := (θ (h) , γ (h) )
8

to capture heterogeneity in baseline preferences and price sensitivity.

2.2

Learning Objective and Regret

The preference parameters ν (h) are unknown in all markets. While source-market observations provide auxiliary information, the seller’s objective is entirely target-market–centric:
to maximize cumulative expected revenue in the target market h = 0.
(0)
(0)
Let (St , pt ) denote the assortment-price pair selected in the target market at time (0)
(0)
(0)
t. The expected revenue in period t is Rt (St , pt ), as defined in (3). Learning proceeds sequentially under bandit feedback, as only realized purchases and revenues are observed.
To evaluate performance, we compare any policy to a clairvoyant benchmark that knows the true target-market parameter ν (0) . Given realized contexts at time t, the benchmark selects (0)
(St∗ , p∗t ) ∈ argmax Rt (S, p), S∈SK , p∈[0,P̄ ]N

where SK := {S ⊆ [N ] : |S| ≤ K}. The regret of a policy π over horizon T is therefore Regret(T ; π) :=

T  X



(0)
(0)
(0)
(0)
Rt (St∗ , p∗t ) − Rt (St , pt )

.

t=1

This benchmark isolates the cost of demand learning in the target market. Source-market data influence regret only indirectly through their effect on parameter estimation and decision quality in the target market.

2.3

Structured Cross-Market Heterogeneity (Utility Shift Model)

We model cross-market heterogeneity through a structured preference shift. Markets share a common contextual utility structure, while differences relative to the target market are confined to a limited set of economically meaningful dimensions. This reflects environments in which markets may vary in sensitivity to particular attributes or price responsiveness, yet exhibit similar substitution patterns across products.
Assumption 1 (Structured Preference Shift). There exists an index set S∗ ⊆ [2d] with |S∗ | ≤ s0 such that, for every source market h ∈ [H], (ν (0) − ν (h) )j = 0

for all j ∈ / S∗ .

That is, discrepancies between the target market and each source market are supported on a common subset of at most s0 coordinates.
The sparsity level s0 quantifies the degree of cross-market similarity. When s0 is small, markets differ only along a limited number of preference dimensions, enabling substantial 9

information sharing. When s0 is large, heterogeneity becomes more diffuse and transfer becomes less effective.
The requirement of a common support across source markets captures settings in which structural differences arise from a stable set of market-specific factors, rather than arbitrary idiosyncratic shifts. As we show later, this structure yields a transparent variance-bias tradeoff: source data reduce estimation variance along shared directions, while adaptation along shifted coordinates incurs a cost that depends on s0 .

3

Algorithmic Framework: Transfer Joint AssortmentPricing

We develop Transfer Joint Assortment-Pricing (TJAP), a unified learning-and-decision framework for contextual joint assortment-pricing across multiple markets. The goal is to leverage data from related source markets to accelerate learning in a target market, while remaining robust to cross-market heterogeneity. At a high level, TJAP is designed around a central principle:
“Shared information should be pooled to reduce variance, while market-specific deviations must be isolated to control transfer bias.” This principle reflects the variance–bias tradeoff inherent in transfer learning under heterogeneous markets.
High-level overview. TJAP alternates between learning shared structure across markets and adapting to target-specific differences. In each episode, the algorithm first aggregates source-market data to estimate common preference components, and then refines this estimate using target-market observations to correct for sparse deviations. Given this bias-aware estimate, it selects assortments and prices by optimizing an optimistic revenue objective that accounts for both statistical uncertainty and potential transfer bias. To ensure stable learning under adaptive decisions, the algorithm updates estimates episodically and invokes targeted exploration only when the target-market data are insufficient to identify heterogeneous components.
Concretely, TJAP consists of three components:
(i) Aggregate-then-debias estimation. Source-market data are first pooled to estimate shared preference components, and the resulting estimator is then adjusted using targetmarket observations to correct for sparse deviations.
(ii) Optimistic decision making with price-uniform confidence bounds. To balance exploration and exploitation, TJAP adopts a frequentist UCB-style optimism principle tailored to joint assortment-pricing. Unlike standard bandit settings with finite action sets, prices 10

are continuous and enter revenue nonlinearly. We therefore construct utility-level confidence bounds that are uniform over prices and incorporate them into an optimistic revenue objective.
(iii) Episodic information-geometry control. Because adaptive assortment and pricing decisions affect the informativeness of future data, TJAP stabilizes learning through an episodic design that freezes the confidence geometry within each episode. Source-market data tighten the geometry, while a target-market Fisher information criterion ensures identifiability, with randomized exploration invoked only when necessary.
Together, these components implement a unified variance–bias strategy: source markets reduce estimation variance along shared directions, while target-market adjustments control transfer-induced bias. Section 3.1- 3.3 develop each component in detail; complete pseudocode is provided in Algorithm 1 and illustrated in Figure 1.
episode 𝑚 length 𝜏𝑚 = 2𝑚−1

episode 𝑚 − 1 length 𝜏𝑚−1 = 2𝑚−2

𝑞𝑚-1

Fisher Information Matrix 𝐻 0

ℎ

𝑊𝑚−1 = 𝑉𝜏𝑚−1 + ෍ 𝑉𝜏𝑚−1 ℎ=1

(𝑆𝜏𝑚−1 +1 , 𝑝𝜏𝑚−1 +1 )

(𝑆𝑡−1 , 𝑝𝑡−1 )

ℎ

ℎ 𝐼𝑡−1 (𝜈Ƹ 𝑚-1 )

𝐼𝜏𝑚−1 +1 (𝜈Ƹ 𝑚-1 ) ……

(𝑆𝜏𝑚 , 𝑝𝜏𝑚 )

(𝑆𝑡 , 𝑝𝑡 )

ℎ

……

𝐼𝜏𝑚 (𝜈Ƹ 𝑚-1 )

per-period matrix 𝑡−1

ℎ

෍ Aggregate-then-Debias aggregate on source data from episode 𝑚 − 1 debias with target data from episode 𝑚 − 1

𝐼𝑢

= 𝑉𝑡

ℎ

ℎ

𝑉𝜏𝑚

𝑢=𝜏𝑚−1 +1

rolling matrix 𝐻 0

ℎ

𝑊𝑚 = 𝑉𝜏𝑚 + ෍ 𝑉𝜏𝑚

(𝑎𝑔)

𝜈Ƹ 𝑚-1 ← 𝜈Ƹ𝑚-1 + 𝛿መ𝑚-1

episodic matrix

ℎ=1

Optimistic Decision 𝜅 2 𝐾𝑞

𝑐ǁ

𝑚-1 𝑚𝑖𝑛 If 𝜆𝑚𝑖𝑛 𝑉𝑡0 ≤ 2 randomly choose 𝑆𝑡 , 𝑝𝑡 Else 𝑆𝑡 , 𝑝𝑡 = argmax 𝑅෨ 𝑡

Figure 1: Schematic illustration of Algorithm 1 over two episodes. At the start of episode b m−1 is computed from episode m−1 data via the aggregate-then-debias m, the estimate ν procedure (Section 3.1). During episode m, Fisher information is accumulated via per(h)
(h)
period increments It (b ν m−1 ), updating the rolling matrices Vt and yielding the pooled matrix Wm (Section 3.3). The geometry from the previous episode, Wm−1 , is used to construct the UCB bonus throughout episode m (Section 3.2). The policy follows optimistic et (·) (Section 3.2). A forced-exploration is considered only in the selection by maximizing R (0)
final qm−1 periods, when the target curvature (via Vt ) is insufficient.
For clarity of exposition, we first present the framework under homogeneous covariates, so that cross-market heterogeneity arises solely through the structured utility shift 11

Algorithm 1: TJAP-CWF (h)

(h)

(h)

(h)

Input: Streaming data {{xit }i∈[N ] , St , pt , y t }t≥1 for h ∈ {0} ∪ [H]
(0)
(h)
Initialize: V0 ← 02d×2d ; V0 ← 02d×2d for all h ∈ [H]
1 for t ∈ [2d] do (0)
(0)
2 Randomly choose St ∈ SK and pt ∼ Uniform([0, P̄ ]N );
P (0)
(0)
(0)
(0)
3 Vt ← Vt−1 + K12 i∈S (0) e xit (e xit )⊤ ;
t

for each episode m = 2, 3, . . . do 5 Compute τm ← 2m−1 , Tm ← {τm−1 + 1, . . . , τm };

4

// (i) aggregate on source data from episode m − 1 6

1 b (ag)
ν m−1 ← argminν∈R2d H|Tm−1 |

P

h∈[H]

P

(h)
t∈Tm−1 ℓt (ν);

// (ii) debias with target data from episode m − 1 7 8 9 10 11 12 13 14 15

bm−1 ← argmin 2d δ δ∈R



1 (0)

P

|Tm−1 |

(0)

t∈Tm−1

(0)
(ag)
ℓt (b ν m−1 + δ) + λm−1 ∥δ∥1



;

b b m−1 ← ν b (ag)
Set ν m−1 + δ m−1 ;
P (h)
(0)
Set Wm−1 ← Vτm−1 + H h=1 Vτm−1 ;
(0)
(h)
Vτm−1 ← 02d×2d ; Vτm−1 ← 02d×2d , ∀h ∈ [H];
for each period t ∈ Tm do emin κ2 Kqm−1 C (0)
if τm − t ≤ qm−1 and λmin (Vt ) ≤ then 2 (0)
(0)
Randomly choose St ∈ SK and pt ∼ Uniform([0, P̄ ]N );
else (0)
(0)
Offer (St , pt ) = argmax

et (S, p; αm−1 , βm−1 , Wm−1 );
R

S∈SK , p∈P N 16 17

for h ∈ {0} ∪ [H] do P (h)
(h)
(h)
(h)
(h)
xit )⊤ − ν m−1 ) e xit (e Vt+1 ← Vt + i∈S (h) qit (b t P P (h)
(h)
(h)
(h)
xjt )⊤ ;
ν m−1 ) e xit (e ν m−1 )qjt (b (h)
(h) qit (b i∈S j∈S t

t

(Assumption 5). The extension to covariate shifts is discussed in Section 3.4.
Episodic updates. TJAP operates in episodes of geometrically increasing length. Let τ1 := 1 and define episode endpoints τm := 2m−1 for m = 2, 3, . . .. Episode m consists of time indices Tm := {τm−1 + 1, . . . , min{τm , T }}.
The horizon [T ] is therefore partitioned into M = ⌈log2 (T + 1)⌉ episodes, so parameter updates occur only O(log T ) times (Auer et al. 2008, Zhang, Chen & Yan 2025).
At the beginning of episode m, parameter estimates and confidence sets are updated using data collected in episode m − 1. Within each episode, these quantities are held fixed.
This freezing of the information geometry stabilizes learning under adaptive pricing and assortment decisions while ensuring that estimation error contracts geometrically over time.
12

(h)

For each market h ∈ {0} ∪ [H], let Tm−1 := Tm−1 denote the set of periods in episode (h)
m−1. Source-market data {Tm−1 }H h=1 are pooled to estimate shared preference components, (0)
while target-market data Tm−1 are used for debiasing and identifiability control.

3.1

Aggregate-then-Debias Estimation

The structured preference shift in Section 2.3 suggests separating estimation into shared and market-specific components. In each episode, TJAP first aggregates source-market data to estimate the common preference structure, thereby reducing estimation variance along directions that are stable across markets. Because source markets may differ from the target along a sparse set of coordinates, the aggregate estimator may be biased relative to the target parameter. To correct this bias, we refine the aggregate center using only target-market data. This second step estimates the sparse discrepancy between the pooled source center and the target preference parameter, yielding an estimator that balances cross-market variance reduction with target-specific adaptation.
For each market h, let  (h) (h) (h) (h)
(h)
(h)
Dm−1 := (xit , St , pt , y t ) : t ∈ Tm−1 (h)

denote the data collected in episode m − 1, where xit ∈ Rd denotes the observed covariate vector associated with product i in market h, capturing product attributes, customer (h)
characteristics, or their interactions; St ⊆ [N ] is the assortment offered in market h (h)
(h)
at time t; pt = (pit )i∈S (h) is the vector of prices posted for the offered products; and t

(h)
(h)
y t ∈ {0, 1}|St |+1 is the realized purchase indicator, including the outside option, with (h)
exactly one entry equal to one. TJAP uses source-market data {Dm−1 }H h=1 to estimate (0)
shared preference components and target-market data Dm−1 to correct for sparse devia-

tions.
Writing the systematic utility in augmented form, (0)

(0)

vit (p; ν) = ⟨e xit (p), ν⟩, (0)

(0)

(0)

where e xit (p) = (xit , −pxit ) ∈ R2d , the negative log-likelihood for a single observation in market h is   (h) (h)
X exp(⟨e xit (pit ), ν⟩)
(h)
(h)
.
ℓt (ν) = − yit log P (h) (h)
1 + j∈S (h) exp(⟨e xjt (pjt ), ν⟩)
(h)
i∈St

∪{0}

t

13

Step I. Aggregation (variance reduction). At the beginning of episode m, TJAP first computes an aggregate estimator using only source-market data:
H X X (h)
1 ag b m−1 ∈ argmin PH ℓt (ν).
ν ν∈R2d h=1 |Tm−1 | h=1 t∈Tm−1

(4)

Because source markets share preference structure outside the sparse shift set, pooling b ag reduces variance along shared coordinates. In particular, the variance of ν m−1 contracts at a rate proportional to the total amount of source data, which scales with H.
To align the debiasing step with the pooled estimator in (4), we define the ground-truth aggregate center as the population minimizer of the same pooled objective. Let Tm−1 denote the set of decision epochs within episode m − 1, and define the episode-m−1 population pooled objective   H X X 1 1 (h)
E (5)
Lag ℓt (ν) , m−1 (ν) := H h=1 |Tm−1 | t∈T m−1

where the expectation is taken under the data-generating process of each source market in episode m − 1. We then define ag ν ag m−1 ∈ argmin Lm−1 (ν).

(6)

ν∈R2d

This population center is the limit point that the pooled estimator in (4) targets.
Step II. Debiasing (Target Adaptation). To correct the (potential) mismatch between the source pooled center and the target parameter, TJAP refines the aggregate center using target-market data only. Let δ ∗m−1 := ν (0) − ν ag m−1

(7)

denote the discrepancy between the population pooled source center and the true target parameter. Under Assumption 1, δ ∗m−1 is supported on at most s0 coordinates.
TJAP estimates δ ∗m−1 by solving an ℓ1 -regularized likelihood problem on the target data:
    1 X (0) ag  bm−1 ∈ argmin b m−1 + δ + λm−1 ∥δ∥1 , ℓt ν δ  δ∈R2d  |Tm−1 | t∈Tm−1

where λm−1 > 0 is a regularization parameter. The final estimator used in episode m is b b m−1 := ν b ag ν m−1 + δ m−1 .

14

3.2

Optimistic Decision Rule with Price-Uniform Confidence Bounds

Having constructed a bias-aware estimator of the target preference parameter, we now translate statistical uncertainty into pricing and assortment decisions. TJAP adopts a frequentist UCB-style optimism principle to balance exploration and exploitation.
Unlike standard bandit settings with finite action sets, joint assortment-pricing involves combinatorial assortment choices and continuous price decisions that enter revenue nonlinearly. We therefore construct confidence bounds at the utility level that are uniform over prices and embed them into an optimistic revenue objective.
To quantify statistical uncertainty in a way that is compatible with adaptive decisions, TJAP maintains an episodic information matrix Wm−1 defined in (10). The matrix Wm−1 aggregates information from all markets over episode m − 1. Throughout episode m, the b m−1 and the episodic information matrix Wm−1 are fixed.
parameter estimate ν Two-radius optimistic utility. At episode m, TJAP constructs an intermediate optimistic utility (0)
b m−1 ⟩ + uit (p), v̄it (p) := ⟨e xit (p), ν where the bonus uit (p) has a two-radius form (0)

(0)

variance term

transfer-bias term

−1 + βm−1 ∥e xit (p)∥∞ , uit (p) = αm−1 ∥e xit (p)∥Wm−1 {z } {z } | |

(8)

−1 where ∥ · ∥Wm−1 and ∥ · ∥∞ denote the Mahalanobis norm and infinity norm, respectively.
The first term reflects statistical uncertainty and shrinks as pooled information accumulates across markets. The second term accounts for residual transfer bias arising from sparse cross-market shifts. They arise directly from the estimation error decomposition in Section b m−1 denote the current estimator and ν (0) the true parameter. Under self3.1. Let ν normalized concentration, we have

⊤ e (0)
b m−1 − ν (0)
x ν it (p)



(0)

≤ αm−1 e xit (p) W −1 , m−1

so the Mahalanobis norm naturally scales uncertainty by the inverse information matrix, upweighting poorly explored directions. In addition, under the sparse utility-shift model, b m−1 − ν (0) satisfies ∥rm−1 ∥1 ≤ βm−1 . By Hölder’s inequality, the residual rm−1 := ν (0)
(0)
⊤ e (0)
x xit (p)∥∞ ∥rm−1 ∥1 ≤ βm−1 ∥e xit (p)∥∞ , it (p) rm−1 ≤ ∥e

which explains the ℓ∞ -based transfer-bias radius.

15

Price-uniform envelope. The intermediate bound v̄it (p) may not preserve the monotonicity of utility in price implied by positive price sensitivity (Assumption 2). To restore this structure while maintaining optimism, we leverage the structural property vit′ (p) ≤ −L0 < 0 and tighten the bound via a monotone-Lipschitz envelope:
 veit (p) := min v̄it (p′ ) − L0 (p − p′ ) , ′

p ∈ [0, P̄ ],

p ≤p

(9)

where L0 is the uniform lower bound on price sensitivity. By construction (Lemma 24), veit (p) is decreasing, L0 -Lipschitz, and satisfies (0)

vit (p) ≤ veit (p) ≤ v̄it (p) for all p ∈ [0, P̄ ].
Optimistic revenue maximization. For assortment S and price vector p, we define the optimistic revenue P et (S, p; αm−1 , βm−1 , Wm−1 ) := R

eit (pi )
i∈S pi exp v

1+

P

 .

ejt (pj )
j∈S exp v

At each period t in episode m, TJAP selects (0)

(0)

(St , pt ) ∈

argmax S∈SK

et (S, p; αm−1 , βm−1 , Wm−1 ).
R

, p∈[0,P̄ ]N

Because veit (·) is decreasing and Lipschitz, the optimization over prices admits the same fixed-point characterization as in classical MNL joint pricing-assortment problems (Wang 2012). In particular, for any fixed assortment S, the optimal prices can be computed efficiently by solving a one-dimensional fixed-point equation, and the overall maximization remains tractable.
Lemma 16 further shows that this preserves optimism at the optimized revenue level:
(0)

(0)

et (St , pt ; αm−1 , βm−1 , Wm−1 ).
Rt (St∗ , p∗t ) ≤ R This construction yields a valid optimistic upper bound on the true target-market revenue uniformly over prices and assortments. In Section 4, we show that the resulting instantaneous regret decomposes into a variance term governed by αm−1 and a transfer-bias term governed by βm−1 .

3.3

Information Geometry and Episodic Control

The optimistic decision rule in Section 3.2 relies on confidence bounds that must remain valid under adaptively chosen assortments and prices. Because the informativeness of observations depends endogenously on past actions, naive online updates of confidence 16

geometry can lead to ill-conditioned matrices and invalidate self-normalized concentration arguments.
TJAP addresses this issue through episodic information-geometry control. Within each episode, both the parameter estimate and the information matrix used in the confidence radius are frozen. This separation decouples confidence construction from within-episode adaptivity, allowing standard self-normalized concentration arguments to apply.
The Fisher information matrix measures the informativeness of the data and controls the self-normalized confidence radius that enters our optimism bonus. In implementation, we maintain three Fisher information matrices.
Episodic information matrix. Fix an episode m. For each market h ∈ {0} ∪ [H], a time s, and a parameter vector ν ∈ R2d . Conditioning on the realized context and actions (h)
(h)
(h)
(h)
(xs , Ss , ps ), the per-period Fisher information matrix Is (ν) of the MNL model at parameter ν is defined as the conditional expectation of the score outer product. The explicit formula is:
Is(h) (ν) =

X

(h)

(h)

(h)⊤

qis (ν) e xis e xis

X X

−

(h)

(h)

i∈Ss

i∈Ss (h)

(h)

(h)

(h)

(h)

(h)⊤

e is e xjs , qis (ν) qjs (ν) x

(h)

j∈Ss (h)

(h)

e is := e xis (pis ), and qis (ν) is the purchase probability given where we use the shorthand x in (2).
Rolling Fisher information within an episode. At a time t ∈ Tm in the current episode, we define the rolling Fisher information matrix by (h)
Vt

:=

t X

Is(h) (b ν m−1 ) ,

s=τm−1 +1

b m−1 .
where the per-period Fisher information matrix is evaluated at the fixed estimate ν (h)
The matrix Vt accumulates curvature information generated within the current episode and is reset to zero at the beginning of the next episode. Among these matrices, the (0)
target-market Fisher information Vt plays a special role: it tracks the extent to which the algorithm has explored directions that are informative for identifying the target-market parameters under the current policy.
Episodic information matrix and geometry freezing. At the end of episode m − 1, TJAP aggregates curvature information from all markets to form the episodic information matrix H X (0)
Vτ(h)
.
(10)
Wm−1 := Vτm−1 + m−1 h=1

17

which defines the confidence geometry used throughout episode m (Section 3.2). The matrix Wm−1 remains fixed within the episode and enters only through the variance term of the optimistic objective. This geometry freezing decouples confidence construction from withinepisode adaptive decisions, ensuring that estimation error can be controlled uniformly over Tm through self-normalized martingale concentration. At the same time, source-market data contribute directly to Wm−1 , tightening the information geometry and thereby shrinking the variance radius.
Identifiability gate and forced exploration. While pooled curvature tightens confidence regions, consistent estimation of the sparse target deviation requires sufficient curvature in the target-market likelihood. TJAP therefore monitors the minimum eigenvalue (0)
of the rolling target Fisher matrix Vt .
If, near  the end of an episode, the target curvature falls below a prescribed threshold, (0)
i.e. λmin Vt ≥ c0 fails for a prescribed constant c0 > 0, the algorithm enters a short forced-exploration phase in the target market. During this phase, assortments and prices are selected from a fixed exploratory distribution designed to inject information along all coordinates. The duration of this phase is chosen so that, with high probability, the target Fisher matrix satisfies the required eigenvalue condition (Lemma 14).
Crucially, this exploration is episodic and gate-controlled: if the optimistic policy already generates sufficient target-market curvature, no additional exploration is invoked. As shown in Section 4, the total number of forced rounds across all episodes is logarithmic in T , and their contribution to regret is asymptotically negligible.
Forced-exploration length. Let qm−1 denote the length of the forced-exploration window available at the end of episode m. We choose qm−1 to be the smallest integer such that, under the exploration policy (uniformly random assortments and prices), the target Fisher information satisfies the eigenvalue condition in Lemma 14 with failure budget ηm−1 . Equivalently, qm−1 is selected so that with probability at least 1 − ηm−1 , the forced (0)
exploration increases λmin (Vt ) above the prescribed threshold. In our analysis, this choice P implies m qm = e O(d), so forced exploration contributes only a lower-order term to regret.
Role in the analysis. The episodic information-geometry control serves two purposes in the analysis. First, it guarantees that the variance radius in the optimistic utility remains well defined and contracts at a rate governed by the aggregated information in Wm−1 .
Second, by enforcing target-market curvature through the identifiability gate, it ensures that the debiasing step satisfies a uniform restricted eigenvalue condition, which is essential for controlling the transfer-bias radius.
Together, episodic freezing and gate-controlled exploration allow TJAP to combine aggressive cross-market variance reduction with stable and identifiable target-market learning 18

under adaptive joint assortment-pricing decisions.

3.4

Extension to Heterogeneous Covariates

Thus far, we have assumed homogeneous covariate distributions across markets, so that cross-market heterogeneity arises solely through preference shifts. When covariate distributions differ across markets, naive pooling may distort the information geometry and overstate the effective sample size contributed by source markets. To address this, TJAP is modified through the construction of the aggregate center in (4), and the Mahalanobis term inside the variance bonus in (8).
Reweighted episodic information geometry. Under heterogeneous covariates, the key change is to reweight the pooled Fisher information. Concretely, we replace the unweighted pooled information matrix in (10) by Wm−1 := Vτ(0)
+ m−1

H X

ωh,m−1 Vτ(h)
, m−1

(11)

h=1

where ωh,m−1 ≥ 0 controls the contribution of source market h to the pooled geometry. In the homogeneous setting, ωh,m−1 ≡ 1, and (10) is recovered.
The variance bonus in (8) continues to use the Mahalanobis norm induced by this reweighted matrix. By downweighting misaligned sources, the confidence geometry more accurately reflects the curvature of the target-market likelihood.
Reweighting the aggregate center. To maintain consistency between the pooled center and the information geometry, the aggregate estimator in (4) can be modified to b (ag)
ν m−1 ∈ argmin PH ν∈R2d

1

H X

h=1 ωh,m−1 h=1

ωh,m−1 ·

1

X

|Tm−1 | t∈T

(h)

ℓt (ν).

(12)

m−1

Thus both curvature and pooled estimation are aligned with the effective target geometry.
Choice of weights. We propose two lightweight choices that fit seamlessly into TJAP.
(0)

(i) Sample-level importance reweighting. If a density ratio wh (x) = dPx(h) (x) is available or dPx can be reliably estimated, the rolling Fisher matrix may be replaced by an importanceweighted version X (h)
(h)
Veτ(h)
:= wh (xt ) It (b ν m−1 ), m−1 t∈Tm−1

19

(h)
so that, in expectation, source curvature E[Veτm−1 ] aligns with the target geometry.
P (0)
e (h)
In this case, we use Wm−1 = Vτm−1 + H h=1 Vτm−1 and no additional market-level downweighting is required.

(ii) Market-level information weights. When density-ratio estimation is impractical, a (h)
more robust alternative is to retain Vτm−1 and set ωh,m−1 based on a covariatemismatch score, ωh,m−1 =

1 (0)
(h)
1 + χ2 (Pbx ∥Pbx )

or

n co ωh,m−1 = min 1, , ρbh

where ρbh is an estimated density-ratio bound and c > 0 caps the influence of any single source. These choices preserve positive definiteness of Wm−1 while preventing heavily shifted sources from dominating the pooled geometry.
The remainder of the algorithm, including the debiasing step and the optimistic decision rule, remains unchanged. The extension therefore preserves the variance-bias structure of TJAP, with covariate shift handled through controlled reweighting of source curvature.

4

Theoretical Results and Insights

We now establish finite-time regret guarantees for TJAP under the structural conditions introduced in Section 2. Our analysis characterizes how transfer from H source markets accelerates learning in the target market, while explicitly quantifying the residual cost of cross-market preference shifts.
We begin by formalizing standard regularity conditions that ensure well-behaved pricing and estimation under the MNL model.
Assumption 2 (Positive price sensitivity). There exists L0 > 0 such that min i∈[N ], t∈[T ]

⟨xit , γ⟩ ≥ L0 .

This condition ensures that utility is strictly decreasing in price and guarantees the existence of finite revenue-maximizing prices.
Assumption 3 (Bounded parameter space). We assume that ∥xit ∥∞ ≤ 1, ∀i ∈ [N ], t ∈ [T ], and ∥(θ, γ)∥ ≤ 1.
This normalization ensures scale-free regret bounds and bounded curvature of the loglikelihood.

20

Assumption 4 (Non-degeneracy). There exist constants κ > 0 and r > 0 such that for all feasible assortments S ∈ SK , prices p ∈ [0, P̄ ]N , and parameters ν satisfying ∥ν − ν (0) ∥2 ≤ r,  min qt i S, p, ν ≥ κ.
i∈S∪{0}

Assumption 4 ensures that choice probabilities are uniformly bounded away from zero in a neighborhood of the true parameter, yielding well-conditioned Fisher information and valid confidence sets. It is standard in MNL bandits literature (Oh & Iyengar 2019, Chen et al. 2020).
To isolate transfer effects from distributional differences in observed features, we impose:
Assumption 5 (Homogeneous Covariates with Bounded Eigenvalues). For each h ∈ {0} ∪ (h)
[H], covariates xit are drawn i.i.d. across items, rounds and markets from a fixed, but a priori unknown, distribution Px , supported on a bounded set X ⊂ Rd , with mean zero and (h)
(h)
covariance matrix Σ = E[xit (xit )⊤ ]. The covariance matrix Σ satisfies 0 < Cmin ≤ λmin (Σ) ≤ λmax (Σ) ≤ Cmax < ∞, where λmin (Σ) and λmax denote the minimum and maximum eigenvalues of Σ respectively.
Assumption 5 plays two important roles. (i) The boundedness of the covariate support ensures that regret bounds are scale-free, a standard technical condition in online learning literature (Javanmard & Nazerzadeh 2019, Erginbas et al. 2025). This requirement is not restrictive, as it is satisfied by many practically relevant distributions, such as truncated Gaussian or uniform distributions. (ii) The assumption that covariates share a common distribution across all markets (h ∈ {0} ∪ [H]) serves to isolate cross-market heterogeneity to model (utility) shift rather than distributional (covariate) shift. This modeling choice is particularly reasonable when the underlying populations are comparable in their observed characteristics, while differences across markets manifest primarily in latent preference parameters. By decoupling feature distributional similarity from preference heterogeneity, the assumption provides a clean setting to analyze the impact of transfer. This condition can be relaxed to allow covariate shift, as discussed in Section 3.4.
Under these conditions, we first characterize the statistical convergence of the target parameter estimator, and then establish regret guarantees and fundamental limits of transfer learning in this setting.

4.1

Statistical Error of the Aggregate-Then-Debias Estimator

We first establish the statistical accuracy of the aggregate-then-debias estimator, and then translate this result into regret guarantees that quantify the benefit and limitation of transfer.
21

b m−1 denote the estimator constructed at the beginning of episode m, based on Let ν source-market aggregation and target-only ℓ1 -debiasing. Let ν (0) denote the true target parameter. The following theorem provides a high-probability bound that decomposes estimation error into a variance component, reduced through cross-market pooling, and a bias component, driven by sparse preference shifts.
Theorem 6 (Statistical Error Bound). Under Assumptions 1–5, there exist constants Cv , Cb > 0, depending only on L0 , Cmin , Cmax , κ, r, such that with probability at least 1 − η, for every episode m, s ∥b ν m−1 − ν (0) ∥2 ≤ Cv

d log((1 + H)T /η)
+ Cb s0 (1 + H) τm−1

s

log(dT /η)
.
τm−1

q  −1 The bound separates two sources of estimation error: firstly, the variance term, O d τm−1 (1 + H)−1 decreases as  additional source markets contribute curvature. and the transfer-bias term:
 −1/2 O s0 τm−1 , reflects the cost of estimating sparse target-specific deviations and does not benefit from additional sources. Thus, pooling H source markets effectively increases the sample size for shared preference coordinates by a factor of 1 + H, while estimation along shifted coordinates depends solely on target data.

4.2

Regret Upper Bound

We now quantify the learning performance of TJAP under the structural conditions introduced above. The tuning parameters in TJAP directly reflect the two-layer estimation structure introduced in Section 3. Specifically:
• αm controls the variance radius in the optimistic utility and arises from self-normalized concentration in the pooled information geometry. It scales with the effective curvature accumulated across the target and source markets.
• λm is the regularization parameter in the ℓ1 -debiasing step and determines the accuracy with which sparse target-specific deviations are estimated from target-market data.
• βm is the resulting transfer-bias radius, proportional to s0 λm , and captures the residual error induced by cross-market preference shifts.
The forced-exploration length qm ensures that the target Fisher information satisfies a restricted eigenvalue condition, which is necessary for stable sparse recovery.
With these choices, the estimation error admits a variance-bias decomposition that propagates to regret. The following result provides a finite-time regret bound that makes 22

explicit how transfer from H source markets accelerates learning through pooled curvature while accounting for residual error induced by cross-market preference shifts.
Theorem 7 (Regret Upper Bound). Suppose Assumptions 1–5 hold. Run Algorithm 1 with episodic parameters s αm = cα

s

 tr(Wm )  2 2d log 1 + + log , 2d λ0 ηm

λm = cλ

log(2d/ηm )
, |Tm |

βm =

cβ s 0 λ m , ϕ2m

and choose the forced-exploration schedule qm according to Lemma 14 with per-episode failure budget ηm , where qm is the (episode-dependent) exploration-window length defined P in Section 3.3, {ηm }m≥1 satisfies m ηm ≤ T −2 , and ϕ2m is the restricted strong convexity constant in (21).
Then there exist constants C1 , C2 , C3 > 0, depending only on L0 , P̄ , Cmin , Cmax , κ, r, such that the cumulative expected regret up to horizon T satisfies "
Regret(T ; π) ≤ K P̄ C1 d

s

T log (1 + H)T 3 1+H



# q   + C2 s0 T log 4dT 2 + C3 d log2 4dT 2 .
(13)

The regret bound exhibits a transparent variance-bias decomposition. The leading  term scales as O d T 1/2 (1 + H)−1/2 , reflecting variance reduction from pooling H source markets. Under homogeneous covariates, source data contribute curvature in the same geometry as the target, effectively increasing the information available for shared preference directions by a factor of 1 + H. Consequently, the statistical uncertainty contracts at rate √ 1/ 1 + H.
 √  The second term, O s0 T , captures residual transfer bias due to sparse cross-market preference shifts. This term depends only on the sparsity level s0 and does not improve with additional source markets. Intuitively, deviations confined to shifted coordinates must be learned from target data alone.
The remaining logarithmic term arises from episodic identifiability control and forced √ exploration; it is lower order relative to the main T contributions. In particular, when s0 is small relative to d, transfer yields substantial gains, while when heterogeneity is diffuse (s0 large), the improvement from additional source markets diminishes.

4.3

Regret Lower Bound

The following theorem characterizes the fundamental limits of transfer learning under the structured preference-shift model.
Theorem 8 (Minimax lower bound). Under the utility model (1) with Assumptions 1–5, for any d ≥ 1, K ∈ [d], and sparsity level s0 ∈ {0, 1, . . . , min{K, d}}, there exists a constant 23

c0 > 0, depending only on L0 , P̄ , Cmin , Cmax , such that for all horizons T , "r inf π

sup

Regret(T ; π) ≥ c0

instances

K (d − s0 ) T + s0 1+H

√

# KT ,

(14)

where the infimum is taken over all policies π, and the supremum is over all problem instances satisfying Assumptions 1–5.
The lower bound mirrors the two components in Theorem 7. The first term scales as p (d − s0 )T (1 + H)−1 , showing that only the shared coordinates benefit from additional √ source markets. The second term scales as s0 T , reflecting the unavoidable cost of learning target-specific deviations. Crucially, this term does not improve with H, since source observations carry no information about shifted coordinates.
Comparing Theorems 7 and 8, we see that TJAP achieves the optimal dependence on H, s0 , and T up to logarithmic factors.

4.4

Structural Implications of Transfer in Joint Assortment-Pricing

Theorems 7 and 8 isolate two distinct forces that govern the value of transfer: statistical uncertainty, which can be reduced by pooling data from multiple markets, and structural mismatch, which cannot. The upper bound (13) makes this separation explicit, and the lower bound (14) shows that the same qualitative trade-off is unavoidable over the problem class.
Transfer accelerates learning along shared directions. The leading regret term p scales as T /(1 + H), demonstrating that additional source markets reduce statistical uncertainty in proportion to pooled curvature. When covariates are homogeneous, source observations contribute information in the same geometric directions as target data. As a result, estimation along shared preference coordinates benefits from an effective sample size proportional to 1 + H. The diminishing marginal improvement in H reflects classical concentration behavior.
√ Heterogeneity imposes an intrinsic ceiling. The second term, proportional to s0 T is independent of H. This term corresponds to coordinates where the target preference differs from all sources. Because source data contain no information about these shifted components, their estimation must rely entirely on target observations. The lower bound confirms that this cost is unavoidable: no algorithm can eliminate the s0 -driven contribution without additional structural assumptions. Thus, transfer is fundamentally asymmetric. It improves learning only along directions where markets are behaviorally aligned and provides no benefit where structural mismatch persists.

24

Sparse heterogeneity determines the value of transfer. The sparsity level s0 emerges as the key measure of cross-market similarity. When heterogeneity is localized (small s0 ), transfer yields substantial gains. When heterogeneity is diffuse (large s0 ), the bias term dominates and the marginal benefit of additional source markets diminishes. In the extreme case s0 ≈ d, transfer offers little improvement over target-only learning.
Exploration overhead is secondary. The logarithmic term in the upper bound arises √ from episodic identifiability control. Its contribution is lower order relative to the T terms and does not alter the fundamental variance–bias structure.

5

Numerical Experiments

We evaluate TJAP on synthetic instances designed to reflect the structured preference-shift model in Section 2. Our experiments aim to examine three operational questions:
• How much do additional markets accelerate joint learning? Regret should decrease √ with the number of source markets H, at a rate consistent with the 1/ 1 + H scaling in Theorem 2.
• When does transfer meaningfully improve joint assortment-pricing? When crossmarket differences are localized (small s0 ), transfer should substantially improve decision quality; when heterogeneity is widespread, the gains should taper off.
• Is naive data pooling operationally safe? Aggregating data without correcting for cross-market shifts should degrade performance when markets differ structurally.

5.1

Synthetic Data Generator

All markets follow the contextual MNL model with a common covariate distribution, so cross-market heterogeneity arises exclusively through preference shifts (Assumption 1). We vary:
• Feature dimension d ∈ {10, 20, 50}, • Sparsity level s0 ∈ {0.2d, 0.3d}, • Catalog size N ∈ {30, 100}, • Capacity K = 5, • Number of source markets H ∈ {0, 1, 3, 5}, • Horizon T = 2000.
25

Contexts are generated i.i.d. across periods and shared across markets to isolate preference heterogeneity from covariate shift. Target parameters are drawn randomly, and each source market is generated by introducing an s0 -sparse utility shift relative to the target. Details of the generator are summarized as follows.
• Catalog and contexts. At each period t = 1, . . . , T , we draw a context matrix X t ∈ RN ×d with rows xit ∈ Rd sampled i.i.d. from a common distribution. Concretely, we generate xit by drawing zit ∼ N (0, Id ) and setting xit = |zit |, followed by entrywise clipping to [0, 1] to ensure ∥xit ∥∞ ≤ 1 ( Assumption 3). We use the same X t across all markets at time t to isolate cross-market differences to preference parameters rather than realized contexts.
• Target parameters. We sample θ (0) ∼ N (0, Id ) and draw γ (0) with strictly positive entries. Since xit ≥ 0 componentwise by construction, we can rescale γ (0) by a positive scalar to satisfy Assumption 2.
• Source parameters. For each source market h ∈ [H], we generate a sparse shift (h)
δ (h) ∈ R2d with support size s0 drawn uniformly at random: δ j = ±∆ on the support and 0 otherwise. We set (h)

θ (h) = θ (0) + δ 1:d ,

(h)

γ (h) = max{10−3 , γ (0) + δ d+1:2d }

(componentwise maximum) to maintain positive price sensitivity. We choose ∆ small enough so that the lower bound in Assumption 2 continues to hold across markets.
This construction matches the task-similarity model with an s0 -sparse discrepancy ( Assumption 1).
This construction allows us to directly test the variance-bias tradeoff characterized in Section 4.

5.2

Baselines

We compare TJAP with four baselines: CAP (Erginbas et al. 2025), M3P (Javanmard et al. 2020), ONS-MPP (Perivier & Goyal 2022), and a naive pooled estimator Pool(H).
CAP is a contextual joint assortment-pricing algorithm and serves as a natural benchmark in the single-market setting; we run it on the target market. M3P and ONS-MPP are pricing-centric methods; to respect the capacity constraint, at each period we rank items by the current estimated systematic utilities and offer the top K items, while using the method’s posted prices for this subset. The pooled estimator Pool(H) aggregates all observations from the target and H sources and fits a single parameter ν via the same likelihood update as TJAP, ignoring cross-market heterogeneity and performing no debiasing. Each configuration is repeated 10 times with independent seeds, and we report mean cumulative regret.
26

(a)
d = 10, s0 = 2, K = 5, N = 30

(b)
d = 20, s0 = 6, K = 5, N = 100

(c)
d = 50, s0 = 15, K = 5, N = 100

(d)
d = 10, s0 = 2, K = 5, N = 30

(e)
d = 20, s0 = 6, K = 5, N = 100

(f)
d = 50, s0 = 15, K = 5, N = 100

Figure 2: Cumulative regret on synthetic instances under varying feature dimension d, sparsity level s0 , and catalog size N . Top row: TJAP with H ∈ {0, 1, 3, 5} compared against CAP, M3P, and ONS–MPP. Bottom row: TJAP with H ∈ {0, 1, 3, 5} compared against the pooled estimator Pool(H) for H ∈ {1, 3, 5}. Each curve is averaged over 10 independent runs; all methods share the same price range [0, P ] and observe identical contexts.

5.3

Main findings

Figure 2 reports cumulative regret for 3 configurations (with the remaining 9 configurations deferred to Figure 3). The experimental results closely mirror the structural insights developed in Section 4 and provide clear evidence on when and how transfer improves joint assortment-pricing.
(i) Additional markets accelerate joint learning.
Across all configurations in Figure 2, cumulative regret under TJAP decreases systematically as the number of source markets H increases. The improvement is monotone in H, and the gap between H = 0 and H = 5 is substantial when preference shifts are sparse.
Operationally, this indicates that incorporating related markets materially accelerates 27

convergence of both pricing and assortment decisions in the target market. The observed improvement is consistent with the theoretical scaling in Theorem 7, where pooled curvature effectively increases the information available for shared preference directions.
(ii) Transfer-aware learning dominates single-market learning.
With transfer enabled, TJAP uniformly outperforms CAP, confirming that auxiliary markets can materially accelerate learning in the target market. In addition, even at H = 0, TJAP improves upon CAP in our implementation, consistent with the benefits of episodic updates and information-matrix control. We also observe lower runtime for TJAP due to parameter updates being performed only at episode boundaries.
(iii) Naive pooling is operationally unsafe under heterogeneity.
The pooled estimator POOL(H), which aggregates data across markets without debiasing, reduces variance but ignores cross-market shifts. As shown in the bottom panels of Figure 2, POOL(H) is uniformly dominated by TJAP with the same H, and the performance gap widens as s0 increases.
This confirms that simply aggregating historical markets can lead to systematically distorted pricing and assortment decisions when markets differ. The aggregate-then-debias design of TJAP successfully captures the informational benefit of pooling while correcting for structural differences.
(iv) Joint modeling of assortment and pricing is essential.
Even without transfer (H = 0), TJAP outperforms M3P and ONS-MPP. This highlights the value of joint decision making: learning the preference parameters through the MNL choice structure and optimizing assortments and prices jointly yields performance gains that pricing-only or pricing-centric methods coupled with a heuristic assortment rule do not capture.
(v) Exploration overhead is limited.
Forced exploration is invoked only when target-market curvature is insufficient. Empirically, the number of forced rounds is small relative to the horizon, and their contribution to regret is negligible compared with the dominant learning terms. This supports the theoretical claim that identifiability control introduces only lower-order regret.
Overall, the experiments validate the central message of the paper: cross-market information can significantly accelerate joint assortment-pricing decisions when heterogeneity is structured, but indiscriminate pooling can degrade performance when structural differences are not accounted for.

28

6

Proof of Main Results

In this section, we outline the main steps in the proofs of Theorems 7 and 8. We emphasize the proof map and intuition; all technical proofs and complete lemma statements are deferred to the appendix.

6.1

Important Lemmas

We restate three core lemmas that drive the upper-bound proof sketch. For readability, we provide streamlined statements here; the appendix contains the full statements and proofs.
(α)

Lemma 9 (Self-normalized concentration). On the event Em−1 , the aggregated estimator satisfies the self-normalized bound (ag)
b (ag)
ν m−1 − ν m−1 W̄m−1 ≤ αm−1 ,

where αm−1 scales as  r   tr(W̄m−1 )
1 + log (α) , d log 1 + αm−1 = Θ d ηm−1

with constants depending only on (L0 , P , Cmin , Cmax ).
Lemma 9 provides the variance radius in the geometry induced by W̄m−1 . In the regret −1 , so larger pooled analysis, αm−1 always appears multiplied by a design norm ∥ · ∥W̄m−1 information directly shrinks the effective statistical error.
q  (β)
Lemma 13 (Target-only debiasing). Choose the ℓ1 -penalty as λm−1 = Θ log(d/ηm−1 )/|Tm−1 | , (β)

with constants depending only on (L0 , P ). On the event Em−1 , the debiasing estimator obeys bm−1 − δ ∗ ∥1 ≤ βm−1 := ∥δ m−1

cβ s0 λm−1 , ϕ2m−1

for a constant cβ > 0 depending only on (L0 , P , Cmin , Cmax ).
Lemma 13 controls the transfer bias due to an s0 -sparse target–source shift. This radius is driven by target data and the target restricted eigenvalue ϕm−1 , and therefore does not improve with more source markets.
Lemma 16 (Revenue optimism). Fix a finite set S and P = [0, P ]. Let vi , vei : P → R be decreasing functions, and define the price-optimized revenue functional VS (·) as in the appendix. If vi (p) ≤ vei (p) for all i ∈ S and all p ∈ P , then VS (v) ≤ VS (e v ).
29

Lemma 16 ensures that optimism is preserved after optimizing over prices (and assortments): upper confidence envelopes at the utility level imply an upper bound on the optimized revenue, despite the arg max depending on the utilities.

6.2

Proof Sketch of Upper Bound (Theorem 7)

Episodic geometry and confidence radii. At the start of episode m, we freeze the episodic information matrix. Lemma 9 yields a self-normalized concentration event on which the aggregated estimator lies in an ellipsoid of radius αm−1 under the metric induced by W̄m−1 . Under homogeneous covariates (Assumption 5), the pooled Fisher informa−1 ≍ tion grows proportionally to (1 + H)τm−1 , implying the geometric shrinkage ∥ · ∥W̄m−1 p 1/ (1 + H)τm−1 (up to constants).
Target-only debiasing and transfer bias. We first compute the aggregated center b (ag)
ν m−1 using all markets in episode m − 1, and then correct it using only target data via p  an ℓ1 -regularized debiasing step. With λm−1 = Θ log(d/ηm−1 )/|Tm−1 | , Lemma 13 yields bm−1 − δ ∗ ∥1 ≤ βm−1 , where βm−1 = Θ(s0 λm−1 ) up to the target an ℓ1 error bound ∥δ m−1 restricted eigenvalue factor ϕ−2 m−1 . Consequently, the bias contribution scales linearly in s0 and does not benefit from additional sources.
(0)

(0)

Optimism and instantaneous regret. The algorithm selects (St , pt ) by maximizing an optimistic revenue objective constructed from upper confidence envelopes veit . Lemma 16 ensures that such utility-level optimism is preserved after optimizing over prices, so the optimized revenue under the true utilities is upper bounded by the algorithm’s optimistic value.
Combining this with Lipschitz continuity of MNL revenue in utilities and the variance–bias decomposition induced by the radii (αm−1 , βm−1 ), we obtain a per-round regret bound of the form s "
# 2 d(1 + P )
Regrett ≲ KP αm−1 + βm−1 (1 + P ) , t ∈ Tm , λmin (W̄m−1 )
where the first term is the variance contribution and the second term is the bias contribution.
Summation over episodes and top-up rounds. Summing the variance part over t ∈ Tm and using the episodic geometry above gives X

r (var)
Regrett

≲ KP d

t∈Tm

30

|Tm | 1+H

q  log (1 + H)T 3 ,

and summing the bias part yields X

(bias)

Regrett

≲ KP (1 + P ) s0

p |Tm | log(4dT 2 ).

t∈Tm

p √ √ P p Using m |Tm | ≲ T delivers the leading-order terms KP d T /(1 + H) and KP s0 T in Theorem 7. Finally, the forced-exploration gate contributes only a lower-order “top-up” e log T ), and each term because the total number of forced rounds across episodes is O(d such round incurs at most KP regret. Combining these bounds and integrating over the failure probabilities yields the expected regret guarantee.

6.3

Proof Sketch of Lower Bound (Theorem 8)

Hard instance and task similarity. We consider N = d items with deterministic covariates xjt = ej , and set γ (h) = L0 1d for all h ∈ {0} ∪ [H], so Assumption 2 holds.
Partition coordinates into a shared block Jvar and a shifted block Jsh :
[d] = Jvar ∪ Jsh ,

|Jvar | = d − s0 , |Jsh | = s0 .

Draw independent Rademacher signs u = (uj )j∈Jvar and w = (wj )j∈Jsh , and set (h)

βj

=

 ∆

j ∈ Jvar , h ∈ {0} ∪ [H],

0,

j ∈ Jsh , h ∈ [H],

var uj ,

(0)

βj =

 ∆

var uj ,

j ∈ Jvar ,

∆shf wj , j ∈ Jsh .

The target–source shift is supported on Jsh with ℓ0 -size at most s0 , so Assumption 1 holds.
We analyze a genie-aided model in which each target action is also played in all sources and all H + 1 outcomes are revealed; this can only make learning easier, so any lower bound remains valid.
Revenue structure. Lemma 19 shows that the single-item price-optimized revenue r∗ (β)
is differentiable and strictly increasing on a compact cube, with r∗ (+∆) − r∗ (−∆) ≍ ∆ for small ∆ > 0. Appendix Lemma 20 shows that, when each item is priced at pi = p∗ (βi ), the multi-item revenue is bounded below by a diluted sum of single-item revenues and that ∗ the clairvoyant top-K assortment SK (β) consists of the K largest coordinates of β. Hence flipping the sign of βj changes the clairvoyant revenue by order ∆ per round whenever ∗ j ∈ SK (β), and these gaps add across coordinates.
KL control and exposure. Let Pu,w be the joint law of the (target+source) trajectory under (u, w) and a fixed policy. For j ∈ Jvar (shared block), Lemma 21 yields KL Pu,w

Pu(j) ,w



≲ (1 + H) ∆2var E[Nj (T )].
31

For j ∈ Jsh (shift block), source observations are identical under both signs, so KL Pu,w

Pu,w(j)



≲ ∆2shf E[Nj (T )],

with no (1 + H) factor. Randomizing item labels and using exchangeability, Lemma 22 gives E[Nj (T )] = KT /d for all j ∈ [d]. Pinsker’s inequality then converts these KL bounds into total-variation bounds for the corresponding binary tests.
From testing to regret. For j ∈ Jvar , Le Cam’s lemma lower bounds the Bayes error in terms of the total variation bound above. Whenever j belongs to the clairvoyant topK set, misclassifying uj reverses its contribution and causes a per-round loss of order ∆var . Aggregating over the d − s0 shared coordinates via an Assouad-type argument and q K(d−s0 )T  d 2 . For j ∈ Jsh , calibrating ∆var ≍ (1+H) KT yields a cumulative contribution Ω 1+H source observations carry no information, so the same testing argument has no (1 + H)
d benefit. Choosing ∆2shf ≍ KT (and taking ∆shf large enough so that shifted coordinates with wj = +1 enter the clairvoyant top-K set when s0 ≤ K) yields a cumulative contribution √ Ω(s0 KT ). Adding the two contributions and applying Yao’s minimax principle gives Theorem 8.

7

Conclusion and Future Directions

This paper develops a transfer learning framework for contextual joint assortment-pricing under an MNL choice model with bandit feedback. We leverage data from multiple source markets to accelerate learning in a designated target market while explicitly accounting for cross-market preference shifts. Our approach builds on an aggregate-then-debias estimation pipeline: it first pools information across source markets to estimate the shared contextual preference structure, and then applies a target-driven debiasing step to adapt to sparse market-specific deviations. Combined with an optimistic learning rule and informationmatrix control to ensure identifiability under adaptively chosen assortments and prices, this design enables effective transfer while guarding against negative transfer when markets differ.
On the theoretical side, we establish matching minimax upper and lower regret bounds that quantify both the benefit and the limitation of transfer. The results show that additional source markets improve performance precisely along shared preference directions, yielding substantial gains when heterogeneity between source and target markets is sufficiently sparse, while target-specific directions incur an unavoidable cost. On the empirical side, our numerical experiments corroborate these insights: incorporating source-market data consistently reduces regret relative to baselines that learn solely from the target market. Together, these findings demonstrate that cross-market information can materially 32

reduce exploration in joint assortment-pricing, and they provide a principled variance-bias perspective on when transfer is beneficial.
Several directions remain open. First, we assume that a set of informative source markets is available, whereas in practice the learner may need to discover which markets are reliably informative and to downweight or discard harmful sources online. Second, our analysis focuses on sparse preference shifts; extending the framework to richer and more flexible notions of relatedness, such as ℓq -type approximate sparsity, low-rank or latentfactor structure across markets, and distributional notions of similarity that couple contexts with preferences, would broaden applicability and sharpen guidance for heterogeneous multi-market deployments. Progress on these questions would further clarify how transfer learning can be deployed safely and effectively in operational decision-making systems.

References Abbasi-Yadkori, Y., Pál, D. & Szepesvári, C. (2011), ‘Improved algorithms for linear stochastic bandits’, Advances in neural information processing systems 24.
Agrawal, S., Avadhanula, V., Goyal, V. & Zeevi, A. (2017), Thompson sampling for the mnl-bandit, in ‘Conference on learning theory’, PMLR, pp. 76–78.
Agrawal, S., Avadhanula, V., Goyal, V. & Zeevi, A. (2018), ‘Mnl-bandit: A dynamic learning approach to assortment selection’.
URL: https://arxiv.org/abs/1706.03880 Aouad, A., Farias, V. & Levi, R. (2021), ‘Assortment optimization under consider-thenchoose choice models’, Management Science 67(6), 3368–3386.
Aouad, A., Farias, V., Levi, R. & Segev, D. (2018), ‘The approximability of assortment optimization under ranking preferences’, Operations Research 66(6), 1661–1669.
Auer, P. (2002), ‘Using confidence bounds for exploitation-exploration trade-offs’, Journal of machine learning research 3(Nov), 397–422.
Auer, P., Jaksch, T. & Ortner, R. (2008), ‘Near-optimal regret bounds for reinforcement learning’, Advances in neural information processing systems 21.
Ban, G.-Y. & Keskin, N. B. (2021), ‘Personalized dynamic pricing with machine learning: High-dimensional features and heterogeneous elasticity’, Management Science 67(9), 5549–5568.
Bastani, H. (2021), ‘Predicting with proxies: Transfer learning in high dimension’, Management Science 67(5), 2964–2984.

33

Bastani, H., Simchi-Levi, D. & Zhu, R. (2022), ‘Meta dynamic pricing: Transfer learning across experiments’, Management Science 68(3), 1865–1881.
Bu, J., Simchi-Levi, D. & Xu, Y. (2020), Online pricing with offline data: Phase transition and inverse square law, in ‘international conference on machine learning’, PMLR, pp. 1202–1210.
Bu, J., Simchi-Levi, D., Xu, Y. & Zhai, C. W. S. (2025), ‘Feature-based dynamic pricing with online learning and offline data’, Available at SSRN 5261068 .
Chai, J., Chen, E. & Fan, J. (2025), ‘Deep transfer Q-learning for offline non-stationary reinforcement learning’, arXiv preprint arXiv:2501.04870 .
Chai, J., Chen, E. & Yang, L. (2025), Transfer Q-learning with composite mdp structures, in ‘Forty-second International Conference on Machine Learning (ICML 2025)’.
Chai, J., Liu, X., Chen, E. & Yan, Y. (2026), ‘Low-rank plus sparse matrix transfer learning under growing representations and ambient dimensions’, arXiv preprint arXiv:2601.21873 .
Chai, J., Zhang, E., Chen, E. & Yan, Y. (2026), ‘Optimistic transfer under task shift via bellman alignment’, arXiv preprint arXiv:2601.21924 .
Chen, B., Chao, X. & Shi, C. (2021), ‘Nonparametric learning algorithms for joint pricing and inventory control with lost sales and censored demand’, Mathematics of Operations Research 46(2), 726–756.
Chen, E., Chen, X., Jing, W. & Liu, X. (2025), ‘Stochastic linear bandits with latent heterogeneity’, arXiv preprint arXiv:2502.00423 .
Chen, E., Li, S. & Jordan, M. I. (2025), ‘Transfer Q-learning for finite-horizon markov decision processes’, Electronic Journal of Statistics 19(2), 5289–5312.
Chen, N., Cire, A. A., Gao, P. & Wang, S. (2025), ‘Assortment optimization without prediction: An end-to-end framework with transaction data’, Available at SSRN 5280529 .
Chen, N., Cire, A. A., Hu, M. & Lagzi, S. (2023), ‘Model-free assortment pricing with transaction data’, Management Science 69(10), 5830–5847.
Chen, N., Liu, Y. & Wu, R. (2025), ‘Dynamic pricing of limited inventories with word-ofmouth effect’, Available at SSRN 5297977 .
Chen, W., Wang, Y. & Yuan, Y. (2013), Combinatorial multi-armed bandit: General framework and applications, in ‘ICML’, pp. 151–159.
34

Chen, X., Wang, Y. & Zhou, Y. (2020), ‘Dynamic assortment optimization with changing contextual information’, Journal of machine learning research 21(216), 1–44.
Chen, Y. & Shi, C. (2019), ‘Joint pricing and inventory management with strategic customers’, Operations Research 67(6), 1610–1627.
Cheung, W. C. & Simchi-Levi, D. (2017), ‘Thompson sampling for online personalized assortment optimization problems with multinomial logit choice models’, Available at SSRN 3075658 .
Chu, W., Li, L., Reyzin, L. & Schapire, R. (2011), Contextual bandits with linear payoff functions, in ‘Proceedings of the fourteenth international conference on artificial intelligence and statistics’, JMLR Workshop and Conference Proceedings, pp. 208–214.
Dani, V., Hayes, T. P. & Kakade, S. M. (2008), Stochastic linear optimization under bandit feedback, in ‘21st Annual Conference on Learning Theory’, pp. 355–366.
den Boer, A. V. & Keskin, N. B. (2022), ‘Dynamic pricing with demand learning and reference effects’, Management Science 68(10), 7112–7130.
Erginbas, Y. E., Courtade, T. A. & Ramchandran, K. (2025), ‘Online assortment and price optimization under contextual choice models’, arXiv preprint arXiv:2503.11819 .
Feng, Q. & Zhu, R. (2023), ‘Principro: Data-driven algorithms for joint pricing and inventory control under price protection’, Available at SSRN 4511384 .
Filippi, S., Cappe, O., Garivier, A. & Szepesvári, C. (2010), ‘Parametric bandits: The generalized linear case’, Advances in neural information processing systems 23.
Gao, P., Ma, Y., Chen, N., Gallego, G., Li, A., Rusmevichientong, P. & Topaloglu, H.
(2021), ‘Assortment optimization and pricing under the multinomial logit model with impatient customers: Sequential recommendation and selection’, Operations research 69(5), 1509–1532.
Han, J., Hu, M., Xu, Y. & Zhang, X. (2025), ‘Meta dynamic pricing with nonparametric empirical bayes’, Available at SSRN .
Huang, X., Xu, K., Lee, D., Hassani, H., Bastani, H. & Dobriban, E. (2025), ‘Optimal multitask linear regression and contextual bandits under sparse heterogeneity’, Journal of the American Statistical Association pp. 1–14.
Javanmard, A. & Nazerzadeh, H. (2019), ‘Dynamic pricing in high-dimensions’, The Journal of Machine Learning Research 20(1), 315–363.

35

Javanmard, A., Nazerzadeh, H. & Shao, S. (2020), Multi-product dynamic pricing in highdimensions with heterogeneous price sensitivity, in ‘2020 IEEE International Symposium on Information Theory (ISIT)’, IEEE, pp. 2652–2657.
Jia, H., Shi, C. & Shen, S. (2024), ‘Online learning and pricing for service systems with reusable resources’, Operations Research 72(3), 1203–1241.
Keskin, N. B. & Zeevi, A. (2014), ‘Dynamic pricing with an unknown demand model:
Asymptotically optimal semi-myopic policies’, Operations research 62(5), 1142–1167.
Kim, J.-h. & Oh, M.-h. (2025), ‘Dynamic assortment selection and pricing with censored preference feedback’, arXiv preprint arXiv:2504.02324 . Conference version: ICLR 2025.
Kim, K. R., Wang, Y., Li, X. & Chen, G. (2025), Collaborative prediction: to join or to disjoin datasets, in ‘Proceedings of the Forty-First Conference on Uncertainty in Artificial Intelligence’, UAI ’25, JMLR.org.
Kök, A. G., Fisher, M. L. & Vaidyanathan, R. (2008), ‘Assortment planning: Review of literature and industry practice’, Retail supply chain management: Quantitative models and empirical studies pp. 99–153.
Kveton, B., Szepesvári, C., Wen, Z. & Ashkan, A. (2015), Cascading bandits: Learning to rank in the cascade model, in ‘ICML’, pp. 767–776.
Lattimore, T. & Szepesvári, C. (2020), Bandit algorithms, Cambridge University Press.
Li, L., Lu, Y. & Zhou, D. (2017), Provably optimal algorithms for generalized linear contextual bandits, in ‘Proceedings of the 34th International Conference on Machine Learning - Volume 70’, ICML’17, JMLR.org, p. 2071–2080.
Li, S., Cai, T. T. & Li, H. (2022), ‘Transfer learning for high-dimensional linear regression:
Prediction, estimation and minimax optimality’, Journal of the Royal Statistical Society Series B: Statistical Methodology 84(1), 149–173.
Liu, S., Xu, A., Li, Z., Wu, J. & Fan, J. (2023), ‘Unified transfer learning models for high-dimensional data’, arXiv preprint .
Miao, S. & Chao, X. (2021), ‘Dynamic joint assortment and pricing optimization with demand learning’, Manufacturing & Service Operations Management 23(2), 525–545.
Najafi, S., Sun, Z. & Jasin, S. (2025), ‘Pricing and assortment optimization under an mnl model with default specific consideration’.
Oh, M.-h. & Iyengar, G. (2019), Thompson sampling for multinomial logit contextual bandits, Curran Associates Inc., Red Hook, NY, USA.
36

Oh, M.-h. & Iyengar, G. (2021), Multinomial logit contextual bandits: Provable optimality and practicality, in ‘Proceedings of the AAAI conference on artificial intelligence’, number 10, pp. 9205–9213.
Ou, M., Li, N., Zhu, S. & Jin, R. (2018), Multinomial logit bandit with linear utility functions, in ‘Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence (IJCAI-18)’, International Joint Conferences on Artificial Intelligence Organization, pp. 2602–2608.
URL: https://doi.org/10.24963/ijcai.2018/361 Perivier, N. & Goyal, V. (2022), ‘Dynamic pricing and assortment under a contextual mnl demand’, Advances in Neural Information Processing Systems 35, 3461–3474.
Repasky, M., Wang, H. & Xie, Y. (2024), ‘Multi-agent reinforcement learning for joint police patrol and dispatch’, arXiv preprint arXiv:2409.02246 .
Rusmevichientong, P. & Tsitsiklis, J. N. (2010), ‘Linearly parameterized bandits’, Mathematics of Operations Research 35(2), 395–411.
Talluri, K. T. & Van Ryzin, G. J. (2006), The theory and practice of revenue management, Vol. 68, Springer Science & Business Media.
Tian, Y. & Feng, Y. (2023), ‘Transfer learning under high-dimensional generalized linear models’, Journal of the American Statistical Association 118(544), 2684–2697.
Tropp, J. A. et al. (2015), ‘An introduction to matrix concentration inequalities’, Foundations and Trends® in Machine Learning 8(1-2), 1–230.
Wang, R. (2012), ‘Capacitated assortment and price optimization under the multinomial logit model’, Operations Research Letters .
Xu, K. & Bastani, H. (2024), ‘Multitask learning and bandits via robust statistics’, arXiv preprint . Latest version accessed 2025.
Xu, K. & Bastani, H. (2025), ‘Multitask learning and bandits via robust statistics’, Management Science .
Zeng, Y., Liu, J., Lam, H. & Namkoong, H. (2024), ‘Llm embeddings improve test-time adaptation to tabular y|x-shifts’, arXiv preprint arXiv:2410.07395 .
Zhang, Y., Chen, E. & Yan, Y. (2025), ‘Transfer faster, price smarter: Minimax dynamic pricing under cross-market preference shift’, The 39th Conference on Neural Information Processing Systems (NeurIPS 2025) .
Zhang, Y., Zhu, R. & Xie, Q. (2025), ‘Contextual online pricing with (biased) offline data’, arXiv preprint arXiv:2507.02762 .
37

Zhou, R., Chen, C. & Chen, E. (2025), ‘Prior-aligned meta-rl: Thompson sampling with learned priors and guarantees in finite-horizon mdps’, arXiv preprint arXiv:2510.05446 .

38

SUPPLEMENTARY MATERIAL of “Transfer Learning for Contextual Joint Assortment-Pricing under Cross-Market Heterogeneity” Supplemental Material of “Transfer Learning for Contextual Joint Assortment-Pricing: Multi-Source Utility Shift under Multinomial Logit Model”

Appendix A

Notations

We use the standard Landau notation, where f (t) = O(g(t)) (equivalently, g(t) = Ω(f (t)))
if f (t) ≤ Cg(t) holds for “large” t (not necessarily asymptotic) and for some positive constant C independent of t. Similarly, for t ≤ T , we write f (t) = e O(g(t)) if f (t) ≤ Cg(t)(log T )c holds for “large” t, where C > 0 and c ∈ R are constants independent of t and T .
Let lowercase letter x, boldface letter x, boldface capital letter X, and blackboardbold letter X represent scalar, vector, matrix, and tensor, respectively. The calligraphy letter X represents operator. We use the notation [N ] to refer to the positive integer set {1, . . . , N } for N ∈ Z+ . Let C, c, C0 , c0 , . . . denote generic constants, where the uppercase and lowercase letters represent large and small constants, respectively. The actual values of these generic constants may vary from time to time. For any matrix X, we use xi· , xj , and xij to refer to its i-th row, j-th column, and ij-th entry, respectively. All vectors are column vectors and row vectors are written as x⊤ for any vector x.
P For any vector x = (x1 , . . . , xp )⊤ , let ∥x∥ := ∥x∥2 = ( pi=1 x2i )1/2 be the ℓ2 -norm, Pp and let ∥x∥1 = i=1 |xi | be the ℓ1 -norm. When X is a square matrix, we denote by Tr (X), λmax (X), and λmin (X) the trace, maximum and minimum singular value of X, respectively. For two matrices of the same dimension, define the inner product ⟨X 1 , X 2 ⟩ = Tr(X ⊤ 1 X 2 ).

Appendix B

Proof of Theorem 6

∗ Using ν (0) = ν ag m−1 + δ m−1 , we have the exact decomposition

  ag ∗ b b m−1 − ν (0) = ν b ag ν m−1 − ν m−1 + δ m−1 − δ m−1 .

(15)

ag b ag The proof proceeds by bounding: (i) the variance term ν m−1 −ν m−1 in the pooled geometry bm−1 −δ ∗ in ℓ1 (hence and converting it to an ℓ2 bound; and (ii) the bias-correction term δ m−1 in ℓ2 ). We then combine them via (15).

39

B.1

Tuning parameters and global good event

Fix a global failure probability η ∈ (0, 1). We define four families of high-probability events:
• E (α) : variance event controlling the self-normalized radius αm−1 ;
• E (β) : debiasing event controlling the transfer-bias radius βm−1 ;
• E (gate) : forced-exploration gate event ensuring adequate target curvature;
• E (W ) : pooled Fisher-growth event ensuring λmin (Wm−1 ) grows linearly.
Episode-wise budgets (for α, β, gate). To avoid ambiguity, we allocate separate perepisode budgets using the summable schedule:
η̄m := Then

3η 6(η/2)
= 2 2, 2 2 π m π m

(α)
(β)
(gate)
ηm = ηm = ηm :=

η̄m η = 2 2.
3 π m

(α)
(β)
(gate)
) ≤ η/2.
m≥1 (ηm + ηm + ηm

P

Single budget (for pooled growth). Set η (W ) := η/2 and construct E (W ) with P(E (W ) ) ≥ 1 − η (W ) .
Global good event. Let M := ⌈log2 T ⌉ be the number of episodes. Define E := E

(W )

∩

M  \

 (α)
(β)
(gate)
Em ∩ Em ∩ Em ,

(16)

m=1 (α)

(β)

where Em is the event asserted by Lemma 9, Em is the event asserted by Lemma 13, and (gate)
Em is the event asserted by Lemma 14. By a union bound and the budget construction, P(E) ≥ 1 − η.
In the rest of the proof we work conditionally on E.

B.2

Variance control via self-normalized concentration

Recall the episodic pooled information matrix Wm−1 :=

Vτ(0)
+ m−1

H X

Vτ(h)
.
m−1

(17)

h=1

For analysis introduce the ridge-regularized matrix W̄m−1 := Wm−1 + λ0 I2d , with fixed λ0 > 0.

40

Lemma 9 (Variance radius αm−1 ). Consider episode m. There exists a constant cα > 0, (α)
depending only on (L0 , P , Cmin , Cmax ), such that on Em−1 , s ag ∥b ν ag m−1 − ν m−1 ∥W̄m−1 ≤ αm−1 := cα

 tr(W̄m−1 )  2 d log 1 + + log (α) .
dλ0 ηm−1

(18)

A detailed proof is given in Appendix B.6.1.

B.3

Converting pooled-geometry control to an ℓ2 bound

Lemma 9 controls the aggregation error in the W̄m−1 -geometry. To obtain an ℓ2 bound, we need a lower bound on λmin (W̄m−1 ). This is supplied by the pooled Fisher-growth event E (W ) .
Lemma 10 (Pooled Fisher growth). There exist constants cW > 0 and m0 ∈ N, depending only on (L0 , P , Cmin , Cmax , κ), such that for any η (W ) ∈ (0, 1) we can construct an event E (W ) with P(E (W ) ) ≥ 1 − η (W ) on which λmin Wm−1



≥ cW (1 + H) Cmin τm−1

for all episodes m ≥ m0 .

Consequently, for all such m,   λmin W̄m−1 = λmin Wm−1 + λ0 I2d ≥ λ0 + cW (1 + H) Cmin τm−1 ≍ (1 + H) Cmin τm−1 .
The proof is given in Appendix C.4.6.
Immediate ℓ2 consequence. On E, for all m ≥ m0 , combine Lemma 9 with the eigenvalue lower bound:
s ag ag ∥b ν − ν ∥ α 1 m−1 m−1 W̄m−1 m−1 ag p ≤p ≲ αm−1 .
∥b ν ag m−1 − ν m−1 ∥2 ≤ (1 + H)τm−1 λmin (W̄m−1 )
λmin (W̄m−1 )
(19)
For the finitely many initial episodes m < m0 , τm−1 is O(1), so the same display holds after enlarging constants.

B.4

Debiasing and transfer-bias radius βm−1

We next control the error of the ℓ1 -penalized target correction. Let δ ∗m−1 be defined by (7), and let S ∗ := supp(δ ∗m−1 ), |S ∗ | ≤ s0 (Assumption 1).

41

TJAP estimates δ ∗m−1 from target data by solving ( bm−1 ∈ arg min δ

X

|Tm−1 | t∈T

δ∈R2d

Let Lm−1 (δ) := |Tm−1 |−1 ∇2 Lm−1 (δ).

)

1

(0)
ℓt

b ag ν m−1 + δ



+ λm−1 ∥δ∥1 .

(20)

m−1

(0)
ν ag m−1 + δ) and denote its Hessian by H(δ) := t∈Tm−1 ℓt (b

P

Definition 11 (Target-only restricted eigenvalue). In episode m, the target restricted eigenvalue (RE) constant on the 2s0 -sparse cone is ϕ2m−1 :=

u⊤ H(δ ∗m−1 )u .
∥u∥0 ≤2s0 ∥u∥22

(21)

min

Lemma 12 (Uniform target restricted eigenvalue). Suppose the forced-exploration gate in Lemma 14 is used with a schedule {Λm }m≥1 such that, on the global good event E, (0)

λmin Vτm −1



≥ Λm−1

for all episodes m.

Then there exists a constant ϕ∗ > 0, depending only on (L0 , P , Cmin , Cmax , κ, r), such that on E, ϕ2m−1 ≥ ϕ∗ for all episodes m.
We defer the proof of Lemma 12 to Appendix B.6.2.
(β)

Lemma 13 (Transfer-bias radius βm−1 ). For the failure budget ηm−1 ∈ (0, 1), choose s λm−1 = cλ

(β)  log 2d/ηm−1 , |Tm−1 |

(22)

for a constant cλ > 0 depending only on (L0 , P ). There exists cβ > 0, depending only on (β)
(L0 , P , Cmin , Cmax ), such that on the event Em−1 , bm−1 − δ ∗ ∥1 ≤ ∥δ m−1

cβ s0 λm−1 .
ϕ2m−1

Consequently, define the debiasing radius βm−1 :=

cβ s0 λm−1 .
ϕ2m−1

A detailed proof is given in Appendix B.6.3.

42

(23)

B.5

Total error bound and completion of Theorem 6

Work on the global good event E. Fix any episode m. Using the decomposition (15) and the triangle inequality, ag ∗ b ∥b ν m−1 − ν (0) ∥2 ≤ ∥b ν ag m−1 − ν m−1 ∥2 + ∥δ m−1 − δ m−1 ∥2 .

(24)

Variance term. By (19) and the definition of αm−1 , and using that tr(W̄m−1 ) ≲ d + (1 + H)τm−1 on E, we obtain, for a constant Cv > 0, s ag ∥b ν ag m−1 − ν m−1 ∥2 ≤ Cv

 d log (1 + H)T /η .
(1 + H)τm−1

(25)

Bias term. By Lemma 13 and ∥x∥2 ≤ ∥x∥1 , bm−1 − δ ∗ ∥2 ≤ ∥δ bm−1 − δ ∗ ∥1 ≤ cβ s0 λm−1 .
∥δ m−1 m−1 ϕ2m−1 Using Lemma 12 to substitute ϕ2m−1 ≥ ϕ∗ and |Tm−1 | ≍ τm−1 under the doubling schedule yields, for a constant Cb > 0, s bm−1 − δ ∗ ∥2 ≤ Cb s0 ∥δ m−1

log(dT /η)
.
τm−1

(26)

Combine. Substituting (25) and (26) into (24) proves that on E, for every episode m, s ∥b ν m−1 − ν (0) ∥2 ≤ Cv



d log (1 + H)T /η + C b s0 (1 + H)τm−1

s

log(dT /η)
.
τm−1

Since P(E) ≥ 1 − η, this completes the proof of Theorem 6.
□

B.6

Technical lemmas for Theorem 6

B.6.1

Proof of Lemma 9

We write the negative log-likelihood of all observations used to form the aggregation center b ag ν m−1 (and the corresponding pooled geometry Wm−1 ) as ℓm−1 (ν) := −

X t∈Im−1

43

log Pr(Yt | Xt ), ν

where Im−1 is the union of all rounds contributing to Wm−1 under the episode-freezing rule.
Denote the gradient by gm−1 (ν) := ∇ℓm−1 (ν) =

X

st (ν),

t∈Im−1

where st (ν) is the single-round score. For an assortment (St , pt ) and parameter ν, st (ν) =

X

 e xit 1{Yt = i} − qit (ν) ,

i∈St

with qit (ν) the MNL choice probability and e xit = (xit , −pit xit ). The per-round Fisher increment is f⊤ (Diag(qt (ν)) − qt (ν)qt (ν)⊤ )X ft .
It (ν) = X t The proof proceeds in two steps: (i) a self-normalized concentration for the score at ν ag m−1 in the geometry W̄m−1 ; (ii) conversion to a parameter error bound using local strong convexity.
Self-normalized score bound. Fix any u ∈ R2d . Define the centered single-round score at the population pooled center ν ag m−1 by     ag set := st ν ag Ft−1 .
m−1 − E st ν m−1 Define the scalar martingale Mt (u) :=

X

⟨u, ses ⟩,

Vt (u) :=

s∈Im−1 :s≤t

X

 Var ⟨u, ses ⟩ | Fs−1 .

s∈Im−1 :s≤t

Then Mt (u) is a martingale with bounded increments.
2 xit ∥22 ≤ d(1 + P ) and |St | ≤ K, hence Using ∥xit ∥∞ ≤ 1 and pit ∈ [0, P ], we have ∥e ⟨u, set ⟩ ≤ 2

X

q 2 |⟨u, e xit ⟩| ≤ 2 K d(1 + P ) ∥u∥2 .

i∈St

Therefore, Freedman’s inequality for martingales with bounded increments implies that for any δ ∈ (0, 1), ( P M|Im−1 | (u) ≥

r

1 2 2V|Im−1 | (u) log + δ 3

44

)
q 1 2 ≤ δ.
K d(1 + P ) ∥u∥2 log δ

(27)

Moreover, since Var(Z | Ft−1 ) ≤ E[Z 2 | Ft−1 ], we have X

V|Im−1 | (u) ≤

  X     u, It ν ag E ⟨u, set ⟩2 Ft−1 ≤ u⊤  m−1 t∈Im−1

t∈Im−1

where the last inequality uses the standard quadratic-form upper bound induced by the model-based Fisher increment It (·). By Assumption 4 and boundedness of covariates, the right-hand side is dominated by u⊤ W̄m−1 u up to constants; in particular, V|Im−1 | (u) ≤ u⊤ W̄m−1 u.
Let E := {u ∈ R2d : u⊤ W̄m−1 u ≤ 1}, and let N be a 1/2-net of E under the norm ∥ · ∥W̄m−1 . Standard volumetric estimates give tr(W̄m−1 ) d .
|N | ≤ 1 + dλ0 

(α)

Applying (27) to each u ∈ N with δ := ηm−1 /(2|N |), and using V|Im−1 | (u) ≤ 1 and −1/2 (α)
∥u∥2 ≤ λ0 for u ∈ E, we obtain that with probability at least 1 − ηm−1 , |M|I | (u)| ≤ C′ max p m−1 ⊤ u∈N u W̄m−1 u

s

 tr(W̄m−1 )  2 d log 1 + + log (α)
dλ0 ηm−1

for some constant C ′ > 0. A standard net-to-uniform argument (lifting from N to E) then yields P |⟨u, t∈Im−1 set ⟩| |M|I | (u)| sup p ≤ 2 max p m−1 u∈N u̸=0 u⊤ W̄m−1 u u⊤ W̄m−1 u with the same probability.
Identification of the centered score sum. Under Option I, the ground-truth aggregate center ν ag m−1 is defined as a population minimizer of the pooled objective corresponding to ℓm−1 (·).
To make this definition compatible with adaptive designs, we define the episode-m − 1 conditional pooled risk i X h   fr ℓ̄m−1 (ν) := E ℓm−1 (ν) Fm−1 = E − log Pr(Yt | Xt ) Ft−1 , t∈Im−1

ν

fr where Fm−1 is the σ-field generated by the episode-freezing rule, i.e., the predictable design ft )t∈Im−1 . Let (St , pt , X ν ag m−1 ∈ arg min ℓ̄m−1 (ν).
ν∈R2d

Then the first-order optimality condition yields X

    E st ν ag Ft−1 = ∇ℓ̄m−1 ν ag m−1 m−1 = 0,

t∈Im−1

45

and hence X X   ag gm−1 ν ag = s ν = set .
t m−1 m−1 t∈Im−1

t∈Im−1

Therefore, gm−1 (ν ag m−1 ) W̄ −1 = sup m−1 u̸=0

P |⟨u, t∈Im−1 set ⟩| |M|I | (u)| p ≤ 2 max p m−1 , u∈N u⊤ W̄m−1 u u⊤ W̄m−1 u

and absorbing constants into C1 := 2C ′ gives s

 tr(W̄m−1 )  2 d log 1 + + log (α) , dλ0 ηm−1

gm−1 (ν ag m−1 ) W̄ −1 ≤ C1 m−1 as claimed.

From score to parameter. By optimality of the MLE, gm−1 (b ν ag m−1 ) = 0, and by conag ag vexity, ℓm−1 (b ν m−1 ) − ℓm−1 (ν m−1 ) ≥ 0. A second-order Taylor expansion around ν ag m−1 , together with local strong convexity and the self-normalized bound, yields (18).
□ B.6.2

Proof of Lemma 12 (ag)

Recall that the normalized target loss around ν m−1 is Lm−1 (δ) =

1

X

|Tm−1 |

ℓt (δ),

(0)

t∈Tm−1

with Hessian H(δ) = ∇2 Lm−1 (δ) =

1 |Tm−1 |

X

  ft .
f⊤ Diag(qt (δ)) − qt (δ)qt (δ)⊤ X X t

(0)

t∈Tm−1

By Assumption 5, all choice probabilities are uniformly bounded away from zero on a ball of radius r around the true target parameter ν (0) , for all feasible assortments and prices. Since the aggregate-and-debias estimator stays in this ball on the global good event E (cf. Lemmas 9 and 13), there exist constants 0 < ccurv ≤ Ccurv < ∞, depending only on (L0 , P , κ, r), such that for every episode m and every δ on the line segment joining 0 and δ ∗m−1 , (0)
(0)
V V ccurv τm −1 ⪯ H(δ) ⪯ Ccurv τm −1 .
(28)
|Tm−1 | |Tm−1 | The lower bound in (28) is standard in high-dimensional GLM analysis: it follows from the bounded covariates in Assumption 5, the non-degeneracy of the MNL link in Assumption 4,

46

and the fact that the Fisher information is the conditional expectation of the Hessian (Tian & Feng 2023).
e (0)
On the other hand, Assumption 1 states that the augmented covariates x it have co⊤ e := E[e e ≥ Cmin > 0. Combining this with (28) and the variance matrix Σ xe x ] with λmin (Σ)
gate condition in Lemma 14, we obtain that on E u⊤ H(δ ∗m−1 )u ≥ ccurv

Λm−1 ∥u∥22 ≥ ccurv Cmin ∥u∥22 , |Tm−1 |

∀u ∈ R2d

for all m large enough. Restricting u to the 2s0 -sparse cone in Definition 11, this implies ϕ2m−1 =

u⊤ H(δ ∗m−1 )u ≥ ϕ∗ := ccurv Cmin > 0 ∥u∥0 ≤2s0 ∥u∥22 min

for all episodes m on the global good event E. The constant ϕ∗ depends only on (L0 , P , Cmin , Cmax , κ, r)
and is independent of H and m, which proves Lemma 12.
□ B.6.3

Proof of Lemma 13

The proof follows the standard Lasso template: (i) control of ∥g(δ ∗m−1 )∥∞ ; (ii) a cone condition from the basic inequality using the sparsity of δ ∗m−1 ; (iii) restricted curvature via the RE constant ϕ2m−1 .
Gradient ℓ∞ -bound at δ ∗m−1 . Define g(δ) := ∇Lm−1 (δ). For any coordinate j ∈ [2d], gj (δ) =

1

X

|Tm−1 | t∈T

Zt,j (δ),

m−1

b ag where Zt,j (δ) is the j-th coordinate of the target score at parameter ν m−1 + δ. Using bounded covariates and prices, each Zt,j (δ) is bounded by a constant B depending only on (P ).
At δ = δ ∗m−1 , the parameter equals  ag ∗ (0)
b ag b ag ν + ν m−1 + δ m−1 = ν m−1 − ν m−1 .
(α)

On Em−1 , the aggregation error is controlled in a neighborhood where the score is Lipschitz.
ag b ag Thus, writing em−1 := ν m−1 − ν m−1 , a mean-value expansion yields g(δ ∗m−1 ) = g0 + H(e δ) em−1 , where g0 is the empirical score at the true target parameter ν (0) (hence centered), and e 2 is uniformly bounded by a constant depending only on (L0 , P ) under the local ∥H(δ)∥

47

parameter-space assumptions. Therefore, for a constant cg > 0, ∥g(δ ∗m−1 )∥∞ ≤ ∥g0 ∥∞ + cg ∥em−1 ∥2 .
Applying Hoeffding’s inequality and a union bound over j = 1, . . . , 2d yields ∥g0 ∥∞ ≤ (β)
λm−1 /4 with probability at least 1 − ηm−1 /2 for λm−1 as in (22) with cλ sufficiently large.
p (α)
Moreover, on Em−1 we have ∥em−1 ∥2 ≲ d/((1 + H)τm−1 ), so enlarging cλ (still depending only on primitive constants) ensures cg ∥em−1 ∥2 ≤ λm−1 /4 for all m (absorbing finitely many (α)
(β)
initial episodes into constants). Hence, on Em−1 ∩ Em−1 , ∥g(δ ∗m−1 )∥∞ ≤

λm−1 .
2

(29)

bm−1 bm−1 − δ ∗ . Optimality of δ Basic inequality and cone condition. Let ∆ := δ m−1 gives Lm−1 (δ ∗m−1 + ∆) + λm−1 ∥δ ∗m−1 + ∆∥1 ≤ Lm−1 (δ ∗m−1 ) + λm−1 ∥δ ∗m−1 ∥1 .
By convexity, Lm−1 (δ ∗m−1 + ∆) − Lm−1 (δ ∗m−1 ) ≥ ⟨g(δ ∗m−1 ), ∆⟩.
Thus  ⟨g(δ ∗m−1 ), ∆⟩ ≤ λm−1 ∥δ ∗m−1 ∥1 − ∥δ ∗m−1 + ∆∥1 .
Since δ ∗m−1 is supported on S ∗ , we have ∥δ ∗m−1 ∥1 − ∥δ ∗m−1 + ∆∥1 ≤ ∥∆S ∗ ∥1 − ∥∆(S ∗ )c ∥1 . On (29), λm−1 |⟨g(δ ∗m−1 ), ∆⟩| ≤ ∥g(δ ∗m−1 )∥∞ ∥∆∥1 ≤ ∥∆∥1 , 2 hence  λm−1 ∥∆∥1 ≤ λm−1 ∥∆S ∗ ∥1 − ∥∆(S ∗ )c ∥1 , 2 which implies the cone condition ∥∆(S ∗ )c ∥1 ≤ 3∥∆S ∗ ∥1 ,

√ ∥∆∥1 ≤ 4∥∆S ∗ ∥1 ≤ 4 s0 ∥∆∥2 .

Curvature and RE. A Taylor expansion yields 1 e Lm−1 (δ ∗m−1 + ∆) − Lm−1 (δ ∗m−1 ) = ⟨g(δ ∗m−1 ), ∆⟩ + ∆⊤ H(δ)∆, 2 ∗ e on the segment between δ ∗ for some δ m−1 and δ m−1 + ∆. Local strong convexity implies e ⪰ csc H(δ ∗ ). Combining with the basic inequality and (29) gives H(δ)
m−1

csc ⊤ 3λm−1 ∆ H(δ ∗m−1 )∆ ≤ ∥∆S ∗ ∥1 .
2 2

48

By the RE definition (21) and the cone condition, ∆⊤ H(δ ∗m−1 )∆ ≥ ϕ2m−1 ∥∆∥22 ≥

ϕ2m−1 ∥∆S ∗ ∥21 .
s0

Hence ∥∆S ∗ ∥1 ≤ 3s0 λm−1 /(csc ϕ2m−1 ), and therefore bm−1 − δ ∗ ∥1 = ∥∆∥1 ≤ 4∥∆S ∗ ∥1 ≤ ∥δ m−1

12s0 λm−1 .
csc ϕ2m−1

Setting cβ := 12/csc yields (23).

Appendix C

□

Proof of Theorem 7

In the main text, we present only the expected version of the regret upper bound. Here we first prove a high-probability bound, and then deduce the expected bound.
Throughout, we work on the global good event E defined in (16). On E, all concentration and curvature statements needed below hold simultaneously.

C.1

Forced-exploration gate qm−1 (0)

The following lemma quantifies how large qm−1 needs to be in order to ensure that λmin (Vt )
exceeds a threshold Λm−1 with high probability.
Lemma 14 (Forced-exploration gate). Fix an episode m and a terminal window of q (0)
consecutive forced rounds in the target market, during which St is drawn uniformly from (0) i.i.d.
SK and prices pti ∼ Unif[0, P ], independently of covariates. Within episode m, the b m−1 . Let plug-in parameter for Fisher updates is frozen at ν (0)

Yt := It

 b m−1 ∈ R2d×2d ν

be the per-round Fisher increment. The matrices {Yt } are independent, positive semidefinite, and satisfy E[Yt ] ⪰ µI2d ,

emin , µ := κ2 K C

0 ⪯ Yt ⪯ RI2d ,

2

R := Kd(1 + P ).

If qm−1 is chosen so that for some ε ∈ (0, 1), (1 − ε)qm−1 µ ≥ Λm−1

qm−1 ≥

and

(gate)

2R 2d log (gate) , 2 εµ ηm−1

(0)

then, with probability at least 1 − ηm−1 , λmin (Vt ) ≥ Λm−1 at the end of the forced window.
The proof, based on Tropp’s matrix Chernoff inequality, is given in Appendix C.4.1.
49

Corollary 15 (Explicit gate rule). With ε = 1/2, it suffices to choose &

(

qm = max

C.2

2

2Λm 8Kd(1 + P )
2d , log (gate)
2 2 emin emin κ KC κ KC ηm

)' .

Instantaneous regret under optimism

Lemma 16 (Revenue optimism). Let S be a finite item set and P = [0, P ]. For each i ∈ S, let vi , vei : P → R be decreasing and L0 -Lipschitz functions, and define vi (pi )
i∈S pi e P , 1 + j∈S evj (pj )

P

RS (p) :=

VS (v) := sup RS (p).
p∈P N

If vi (p) ≤ vei (p) for all i ∈ S and all p ∈ P , then VS (v) ≤ VS (e v ).
The proof is given in Appendix C.4.3.
At time t inside episode m, the algorithm chooses (St , pt ) ∈ arg

max

S∈SK , p∈P N

et (S, p), R

et (S, p) := Rt (S, p; αm−1 , βm−1 , Wm−1 ), R

where ve is defined by (9). Let (St∗ , p∗t ) be the clairvoyant maximizer of Rt (S, p). Since vit ≤ veit pointwise, Lemma 16 yields et (St , pt ), Rt (St∗ , p∗t ) = VSt∗ (v) ≤ VSt∗ (e v ) ≤ VSt (e v) = R so the instantaneous regret satisfies et (St , pt ) − Rt (St , pt ).
Regrett := Rt (St∗ , p∗t ) − Rt (St , pt ) ≤ R

(30)

For fixed (S, p), direct differentiation gives ∂Rt (S, p)
= qkt (pk − Rt (S, p)).
∂vkt Since 0 ≤ Rt (S, p) ≤ P and 0 ≤ qkt ≤ 1, we have |∂R/∂vkt | ≤ P , and hence et (S, p) − Rt (S, p) ≤ P 0≤R

X i∈S

 sup veit (pi ) − vit (pi ) .

(31)

pi ∈P

Lemma 17 (Pointwise utility gap). There exists a constant C > 0, depending only on (α)
(β)
(L0 , P , Cmin , Cmax ), such that on Em−1 ∩ Em−1 , for all t ∈ Tm , i ∈ [N ], and p ∈ [0, P ], 



−1 0 ≤ veit (p) − vit (p) ≤ C αm−1 ∥e xit (p)∥W̄m−1 + βm−1 ∥e xit (p)∥∞ .

50

(32)

The (corrected) proof is given in Appendix C.4.4.
2 xit (p)∥22 ≤ d(1+P ), Combining (30), (31), and (32), and using ∥e xit (p)∥∞ ≤ 1+P and ∥e we obtain:
(α)

(β)

Lemma 18 (Instantaneous regret bound). On Em−1 ∩ Em−1 , any round t ∈ Tm satisfies 

s

Regrett ≤ KP C αm−1



2

d(1 + P )
+ βm−1 (1 + P ) .
λmin (W̄m−1 )

(33)

A detailed proof is given in Appendix C.4.5.

C.3

Cumulative regret over episodes

We now sum the instantaneous bound (33) across episodes. Write c for a generic positive constant depending only on the primitive problem parameters.
Growth of the pooled geometry. On E (W ) , Lemma 10 yields, for all sufficiently large m,  λmin W̄m−1 ≳ (1 + H) Cmin τm−1 .
(34)
Variance contribution. On episode m, the variance contribution is s VarRegrett := KP C αm−1

2

d(1 + P )
.
λmin (W̄m−1 )

Using (34) and |Tm | ≍ τm−1 under doubling, s X

VarRegrett ≤ KP C αm−1

t∈Tm

2

d(1 + P )
|Tm | ≲ KP C αm−1 (1 + H)Cmin τm−1

s

2

d(1 + P ) p |Tm |.
(1 + H)Cmin (35)

p From (18) and the budget schedule, αm−1 ≲ d log((1 + H)T ) + log(1/η), hence summing √ P p (35) over m and using m |Tm | ≲ T gives M X X m=1 t∈Tm

r VarRegrett ≤ c KP d

T 1+H

q  log (1 + H)T + log(1/η).

Bias contribution. From (33), the bias contribution is BiasRegrett := KP C βm−1 (1 + P ).

51

(36)

Using (22), (23), and ϕ2m−1 ≥ ϕ∗ , s0 βm−1 ≤ c ϕ∗

s

log(dT /η)
s0 ≍ c |Tm−1 | ϕ∗

Thus X

BiasRegrett ≤ c KP (1 + P )

t∈Tm

Summing over episodes and using M X X

s

log(dT /η)
.
|Tm |

s0 p |Tm | log(dT /η).
ϕ∗

(37)

√ P p |T | ≲ T yields m m

BiasRegrett ≤ c KP (1 + P )

m=1 t∈Tm

s0 p T log(dT /η).
ϕ∗

(38)

P

≤ c d log T ·

Forced-exploration top-up. By Lemma 14 and Corollary 15, polylog(d, T, 1/η). Each forced round incurs at most KP regret, so

m qm

TopUp ≤ c KP d log T · polylog(d, T, 1/η).

(39)

High-probability and expected regret. On E, combining (36), (38), and (39) gives Regret(T ) :=

T X

Rt (St∗ , p∗t ) − Rt (St , pt )



t=1

r ≤ C1 KP d

T 1+H

q

 s0 p log (1 + H)T + log(1/η) + C2 KP (1 + P )
T log(dT /η)
ϕ∗

+ C3 KP d log T · polylog(d, T, 1/η).

(40)

Outside E we have the crude bound Regret(T ) ≤ KP T and P(E c ) ≤ η, hence E[Regret(T )1E c ] ≤ KP T η.
Taking, e.g., η = T −2 makes this term negligible and yields the expected regret bound in Theorem 7 (absorbing polylog factors).
□

C.4

Technical lemmas for Theorem 7

C.4.1

Proof of Lemma 14 (0)

(0)

During the forced rounds, (St , pt ) are i.i.d. and independent of the covariates, while the b m−1 is fixed within the episode. Thus Yt = It(0) (b plug-in parameter ν ν m−1 ) are independent PSD matrices.

52

For any u ∈ R2d , ft u)⊤ Mt (X ft u), u⊤ Yt u = (X

Mt := Diag(qt ) − qt qt⊤ ,

ft stacks the augmented features and qt is the choice probability vector at ν b m−1 .
where X 2 (0)
ft has squared norm at most d(1 + P ), and |St | ≤ K, Since ∥Mt ∥2 ≤ 1 and each row of X 2 ft ∥2 ≤ Kd(1 + P ), so we have ∥X 2 ft ∥2 ∥u∥2 ≤ Kd(1 + P 2 )∥u∥2 .
u⊤ Yt u ≤ ∥X 2 2 2 2

Thus Yt ⪯ RI2d with R := Kd(1 + P ).
Assumption 4 implies that the outside option and each item are chosen with probability b m−1 inside the good event. A direct at least κ > 0 in a neighborhood of ν (0) , hence at ν (0)
|St | calculation shows that for any z ∈ R , ⊤

z Mt z =

X

qti zi2 −

i

Since each qti ≥ κ and

P

X

qti zi

2

=

i

X

qti (zi − µ)2 ,

µ :=

i

X

qti zi .

i

i qti ≤ 1 − κ (outside option at least κ), we have

z ⊤ Mt z ≥ κ

X

qti (zi − µ)2 ≥ κ2 ∥z∥22 .

i

Consequently, for any u,   ft u∥2 ≥ κ2 K C emin ∥u∥2 , u⊤ E[Yt ]u = E[z ⊤ Mt z] ≥ κ2 E[∥z∥22 ] = κ2 E ∥X 2 2 emin > 0 is the minimum eigenvalue of the covariance matrix of the augmented where C emin .
features. Thus E[Yt ] ⪰ µI2d with µ := κ2 K C Pq Let Sq := s=1 Yts be the sum of q independent PSD matrices. By Tropp’s matrix Chernoff lower-tail bound, n o  ε2 qµ  .
P λmin (Sq ) ≤ (1 − ε)qµ ≤ 2d exp − 2R Hence if (1 − ε)qµ ≥ Λm−1

and q ≥

2R 2d log (gate) , 2 εµ ηm−1 (gate)

(0)

then λmin (Sq ) ≥ Λm−1 with probability at least 1 − ηm−1 . Since Vt at the end of the forced window equals the starting value plus Sq and the starting value is PSD, we conclude (0)
λmin (Vt ) ≥ Λm−1 as desired.
□

53

C.4.2

Proof of Lemma 24

Fix i, t and suppress indices. Let v̄ : [0, P ] → R be an upper bound of v, and define ve(p) := min {v̄(p′ ) − L0 (p − p′ )}.
′ p ≤p

Monotonicity. If p1 < p2 , then the set {p′ : p′ ≤ p1 } is contained in {p′ : p′ ≤ p2 }, and for each p′ ≤ p1 , v̄(p′ ) − L0 (p2 − p′ ) ≤ v̄(p′ ) − L0 (p1 − p′ ).
Taking the minimum over p′ ≤ p1 on both sides shows ve(p2 ) ≤ ve(p1 ), so ve is decreasing.
Lipschitz property. Let p1 < p2 . For any p′ ≤ p1 , v̄(p′ ) − L0 (p1 − p′ ) = v̄(p′ ) − L0 (p2 − p′ ) + L0 (p2 − p1 ) ≥ ve(p2 ) + L0 (p2 − p1 ).
Taking the minimum over p′ ≤ p1 yields ve(p1 ) ≥ ve(p2 ) + L0 (p2 − p1 ). Similarly, exchanging the roles of p1 and p2 yields ve(p2 ) ≥ ve(p1 ) + L0 (p1 − p2 ). Thus |e v (p2 ) − ve(p1 )| ≤ L0 |p2 − p1 |, so ve is L0 -Lipschitz.
Upper and lower bounds. For any p, ve(p) = min {v̄(p′ ) − L0 (p − p′ )} ≤ v̄(p) − L0 (p − p) = v̄(p).
′ p ≤p

For any p′ ≤ p, v̄(p′ ) − L0 (p − p′ ) ≥ v(p′ ) − L0 (p − p′ ) ≥ v(p), because v is decreasing with slope at most −L0 (by Assumption 2). Taking the minimum over p′ ≤ p yields ve(p) ≥ v(p). This proves v(p) ≤ ve(p) ≤ v̄(p) for all p.
□ C.4.3

Proof of Lemma 16

We follow the fixed-point argument based on (Wang 2012). For a given family of utilities u = {ui }i∈S and scalar µ ∈ R, define ϕi (µ; u) := sup(p − µ)eui (p) , p∈P

ΦS (µ; u) :=

X

ϕi (µ; u).

i∈S

Since P = [0, P ] is compact and p 7→ (p − µ)eui (p) is continuous, the supremum is attained.
Each ϕi (µ; u) is nonincreasing in µ (pointwise supremum of affine functions with slope −eui (p) ≤ 0), hence ΦS (µ; u) is nonincreasing.
Moreover, if ui (p) ≤ u ei (p) for all p, then eui (p) ≤ euei (p) for all p, hence ϕi (µ; u) ≤ ϕi (µ; u e)
and thus ΦS (µ; u) ≤ ΦS (µ; u e) for all µ.
(41)
54

Fixed-point characterization. Fix u. For each µ ∈ [0, P ], define F (µ) := ΦS (µ; u) − µ.
At µ = 0, we have ϕi (0; u) ≥ 0 and so F (0) ≥ 0. At µ = P , each (p − P )eui (p) ≤ 0, so ϕi (P ; u) ≤ 0 and F (P ) ≤ −P < 0. Since ΦS (·; u) is nonincreasing and −µ is strictly decreasing, F is strictly decreasing and continuous, hence there exists a unique µ∗ (u) ∈ [0, P ] such that F (µ∗ (u)) = 0.
We claim that VS (u) = µ∗ (u).
(42)
Let RS (p; u) be the revenue function. For any p ∈ P N , write u♯i := ui (pi ), and set µ(p) := RS (p; u). Then P u♯i i∈S pi e µ(p) = ♯ , P 1 + j∈S euj so µ(p) =

X

♯

(pi − µ(p))eui ,

i∈S

and therefore µ(p) =

X

♯

(pi − µ(p))eui ≤

X

′

sup (p′i − µ(p))eui (pi ) = ΦS (µ(p); u).
′

i∈S pi ∈P

i∈S

Hence F (µ(p)) = ΦS (µ(p); u) − µ(p) ≥ 0, and since F is strictly decreasing with unique zero at µ∗ (u), it follows that µ(p) ≤ µ∗ (u) for all p, so sup RS (p; u) ≤ µ∗ (u).
p∈P N

Conversely, for each µ ∈ [0, P ] and each i, pick pi (µ) ∈ arg maxp∈P (p−µ)eui (p) (existence follows from compactness). Let p∗ := p(µ∗ (u)). Then X ∗ (p∗i − µ∗ (u))eui (pi ) = ΦS (µ∗ (u); u) = µ∗ (u), i∈S

so

∗ ui (p∗i )
µ∗ (u) + µ∗ (u)
i∈S pi e RS (p ; u) = = P P uj (p∗ ) = µ∗ (u).
uj (p∗j )
1 + j∈S e 1+ je j ∗

P

Therefore VS (u) ≥ µ∗ (u), and together with the previous bound we obtain VS (u) = µ∗ (u), proving (42).

55

Monotonicity in the utilities. Now suppose vi ≤ vei pointwise for all i ∈ S. By (41), ΦS (µ) ≤ ΦS (µ; ve) for all µ.
Let µ∗ (v) and µ∗ (e v ) be the unique zeros of µ = ΦS (µ) and µ = ΦS (µ; ve). Then 0 = ΦS (µ∗ (v)) − µ∗ (v) ≤ ΦS (µ∗ (v); ve) − µ∗ (v), so Fve(µ∗ (v)) ≥ 0 for Fve(µ) := ΦS (µ; ve) − µ. Since Fve is strictly decreasing with zero at µ∗ (e v ), it follows that µ∗ (v) ≤ µ∗ (e v ). Using (42) for both v and ve, we obtain VS (v) ≤ VS (e v ), proving the lemma.
□ C.4.4

Proof of Lemma 17

(α)

On Em−1 , Lemma 9 gives ag ∥b ν ag m−1 − ν m−1 ∥W̄m−1 ≤ αm−1 ,

hence for any item i, time t ∈ Tm , and price p, ag −1 .
|e xit (p)⊤ (b ν ag xit (p)∥W̄m−1 m−1 − ν m−1 )| ≤ αm−1 ∥e (β)

On Em−1 , Lemma 13 gives bm−1 − δ ∗ ∥1 ≤ βm−1 , ∥δ m−1 so by Hölder, bm−1 − δ ∗ )| ≤ ∥e |e xit (p)⊤ (δ xit (p)∥∞ βm−1 .
m−1 Using the identity ag ∗ b b m−1 − ν (0) = (b ν ν ag m−1 − ν m−1 ) + (δ m−1 − δ m−1 ),

we obtain −1 + βm−1 ∥e xit (p)∥∞ .
|e xit (p)⊤ (b ν m−1 − ν (0) )| ≤ αm−1 ∥e xit (p)∥W̄m−1

The rest of the argument (envelope step from v̄ to ve) is unchanged and yields (32).
C.4.5

□

Proof of Lemma 18

Combining (30), (31) and (32) yields Regrett ≤ P

X i∈St

  X  −1 sup veit (pi )−vit (pi ) ≤ P C sup αm−1 ∥e xit (pi )∥W̄m−1 +βm−1 ∥e xit (pi )∥∞ .

pi ∈P

i∈St

56

pi ∈P

p 2 −1 ≤ ∥e xit (p)∥2 / λmin (W̄m−1 ), Using ∥e xit (p)∥∞ ≤ 1+P and ∥e xit (p)∥22 ≤ d(1+P ), as well as ∥e xit (p)∥W̄m−1 gives s 2

d(1 + P )
, λmin (W̄m−1 )

−1 ≤ sup ∥e xit (pi )∥W̄m−1

pi ∈P

Hence



s

Regrett ≤ KP C αm−1

sup ∥e xit (pi )∥∞ ≤ 1 + P .
pi ∈P

2



d(1 + P )
+ βm−1 (1 + P ) , λmin (W̄m−1 )

which is exactly (33).
□ C.4.6

Proof of Lemma 10

Assumptions 2–4 imply that, as long as ν stays in a fixed ball around ν (0) , all choice probabilities are uniformly bounded away from 0 and 1, and the weight matrix  (h)
(h)
(h)
Mt (ν) := Diag qt (ν) − qt (ν)qt (ν)⊤ has spectrum contained in [cprob , Cprob ] for some constants 0 < cprob ≤ Cprob < ∞ depending only on (L0 , P , κ, r).
(h)
By Assumption 5, the augmented covariates e xit are i.i.d. across t and h, with common 2 (h)
e satisfying λmin ( e covariance matrix Σ Σ) ≥ Cmin > 0 and ∥e xit ∥22 ≤ d(1 + P ) almost surely.
Combining these facts, we obtain that for each h and each t, h i   (h)
(h)
b b e ⪰ µW Cmin I2d , 0 ⪯ It ν m−1 ⪯ RW I2d , E It ν m−1 Ft−1 ⪰ µW Σ for some finite constants RW , µW > 0, where the conditional expectation is taken with respect to the fresh covariates at time t (which are independent of Ft−1 ). In particular,  (h)  E Vτm −1 =

X

 (h)
 (h)
E It ν bm−1 ⪰ µW Cmin |Tm−1 | I2d .

(h)

t∈Tm−1

 (h)
The matrices It ν bm−1 are independent across t (conditional on the actions) and satisfy the uniform spectral bounds above. Therefore, Tropp’s matrix Chernoff inequality (Tropp et al. 2015) yields that for any ε ∈ (0, 1),   n o (h)  (h)
(h)
P λmin Vτm −1 ≤ (1 − ε) µW Cmin |Tm−1 | ≤ 2d exp − cCh ε2 |Tm−1 | , for some constant cCh > 0 depending only on RW and µW . Choosing ε = 1/2, we obtain (h)

λmin Vτm −1



≥ c0 Cmin τm−1

57

for all markets h ∈ {0} ∪ [H] and all episodes m ≥ m0 , on an event of probability at least 1 − ηW /2 after a union bound across h and m, where c0 > 0 and m0 depend only on the primitive problem constants.
P (h)
Finally, since Wm−1 = H h=0 Vτm −1 , we have λmin Wm−1



≥

H X

(h)

λmin Vτm −1



≥ (1 + H) c0 Cmin τm−1 ,

h=0

which gives the claimed bound with cW := c0 . This completes the proof of Lemma 10. □

Appendix D

Proof of Theorem 8

Throughout this section we work with a simplified non-contextual representation in the intercept domain. This is without loss of generality under Assumptions 2–1, since we can realize the constructed intercepts via basis-vector covariates (see the discussion at the end of this section).

D.1

Single–item revenue and a K–item dilution bound

We begin with basic properties of the single–item MNL revenue and its extension to multi–item assortments.
Let ez , γ := L0 , r(p; β) := p σ(β − γp), σ(z) := 1 + ez be the single–item expected revenue at price p ∈ P = [0, P ] for intercept β ∈ R. Define the price–optimized single–item revenue r∗ (β) := max r(p; β),

p∗ (β) ∈ arg max r(p; β).

p∈P

p∈P

For a finite set S of items and price vector p ∈ P N , the corresponding MNL revenue is R(S, p; β) :=

X i∈S

pi

eβi −γpi P .
1 + j∈S eβj −γpj

The clairvoyant optimum is ∗ RK (β) :=

max

S∈SK , p∈P N

R(S, p; β).

Lemma 19 (Single–item optimizer and smoothness). Fix γ > 0 and price domain P =

58

[0, P ]. Let q(p; β) :=

eβ−γp , 1 + eβ−γp

r∗ (β) := max r(p; β).

r(p; β) := p q(p; β),

p∈P

Then:
(i) For every β ∈ R, r(·; β) has a unique maximizer p∗ (β) ∈ (0, P ]. If p∗ (β) < P (interior maximizer), then  γ p∗ (β) 1 − q ∗ (β) = 1,

 q ∗ (β) := q p∗ (β); β .

(43)

(ii) Let B ⊂ R be compact and assume q ∗ (β) ∈ [q, q] ⊂ (0, 1) for all β ∈ B (hence p∗ (β)
is interior). Then r∗ is differentiable on B and q ∗ (β)
d ∗ r (β) = , dβ γ

q q |β −β ′ | ≤ |r∗ (β)−r∗ (β ′ )| ≤ |β −β ′ | γ γ

(β, β ′ ∈ B). (44)

In particular r∗ is strictly increasing on B.
The proof appears in Section D.6.1. The next lemma lower-bounds the multi–item revenue in terms of the sum of single–item optima, with a dilution factor capturing the cannibalization effect.
Lemma 20 (K–item dilution bound). Fix a compact parameter cube B = [−β0 , β0 ] such that q ∗ (β) ∈ [q, q] ⊂ (0, 1) for all β ∈ B. Define α :=

q .
1−q

Then for any S ⊆ [N ] with |S| ≤ K, if we set pi = p∗ (βi ) for i ∈ S we have R(S, p; β) ≥

X 1 r∗ (βi ).
1 + Kα i∈S

(45)

Moreover, since r∗ is strictly increasing on B, the optimal top–K assortment ∗ (β) ∈ arg SK

X

max S⊆[N ],|S|≤K

r∗ (βi )

i∈S

is given by the indices of the K largest coordinates of β (ties broken arbitrarily), and ∗ RK (β) ≥

1 1 + Kα

X

r∗ (βi ).

(46)

∗ (β)
i∈SK

The proof is given in Section D.6.2. The monotonicity of r∗ from Lemma 19 (ii) ensures P that maximizing i∈S r∗ (βi ) over |S| ≤ K is equivalent to picking the K largest intercepts.
59

D.2

Hard instance family

We now construct a finite family of problem instances indexed by a sign vector ω, consisting of a shared block of d − s0 coordinates on which all markets agree, and a shifted block of s0 coordinates on which only the target differs from the sources.
Catalog, covariates, and intercepts. We take the catalog size N = d and identify item j with coordinate j ∈ [d]. Set deterministic features xj := ej ∈ Rd ,

e j (p) := [ej , −p ej ] ∈ R2d , x

so that Assumption 3 holds with ∥xj ∥∞ ≤ 1 and the augmented covariance satisfies Σ(e) := (e)
E[e xe x⊤ ] = I2d , hence Cmin > 0. These are consistent with Assumption 5.
We set the price-sensitivity vector so that ⟨xj , γ (h) ⟩ ≡ L0 ,

for all j, h, (h)

and encode the market-specific differences only in the intercept component βj The utility of item j in market h at price p is (h)

:= ⟨xj , θ (h) ⟩.

(h)

(h)

vjt (p) = βj − L0 p + εjt , (h)

with Gumbel noise εjt , so the MNL choice probabilities take the standard form.
Shared vs. shifted coordinates. Partition the index set as [d] = Jvar ∪ Jsh ,

|Jvar | = d − s0 ,

|Jsh | = s0 .

We introduce two independent sign vectors:
u = (uj )j∈Jvar ∈ {−1, +1}d−s0 ,

w = (wj )j∈Jsh ∈ {−1, +1}s0 .

We will place a uniform product prior on (u, w) (each coordinate independent Rademacher).
Choose two small positive numbers ∆var , ∆shf ∈ (0, β0 ] such that [−2∆shf , 2∆shf ] ⊂ B for the compact cube in Lemma 19, and that all induced logits remain in the non-degenerate regime of Assumption 4. We define the intercepts as follows:
• For shared coordinates j ∈ Jvar (variance block), (h)

βj (u, w) = ∆var uj

60

for all h ∈ {0} ∪ [H].

(47)

• For shifted coordinates j ∈ Jsh (transfer block), (h)

(0)

βj (u, w) = 0 (h ∈ [H]),

βj (u, w) = ∆shf wj .

(48)

In words: along Jvar all markets share the same sign–shifted intercept, while along Jsh only the target market is shifted and the sources remain at zero.
The target–source discrepancy vector in market h is  0, j ∈ Jvar , (h)
(h)
(0)
(h)
δ := ν − ν =⇒ δ j = ∆ w , j ∈ J .
shf j sh Thus ∥δ (h) ∥0 ≤ s0 for all h, so Assumption 1 holds with the same s0 as in the theorem.
Since all intercepts lie in [−β0 , β0 ] for sufficiently small ∆var , ∆shf , Assumptions 2–4 remain valid with the same constants (L0 , P , Cmin , Cmax ).
(0)

Genie–aided sources. At each round t, the policy selects an assortment St ∈ SK (0)
(0)
(0)
and a price vector pt ∈ P St for the target market and observes a purchase Yt ∈ (0)
St ∪ {0}. To make the lower bound stronger, we give the learner additional feedback: for (h)
each round and each source market h ∈ [H], we also reveal an independent purchase Yt (0)
(0)
from the same action (St , pt ) under the corresponding intercept vector β (h) . This genieaided feedback can only make learning easier; therefore any lower bound proved under this enriched observation model applies a fortiori to the original setting.
Let Pu,w denote the joint law of the entire trajectory (0)

(0)

(0)

(1)

(H)

ZT := {(St , pt , Yt , Yt , . . . , Yt

)}Tt=1

under a fixed policy π and instance (u, w).

D.3

Per–coordinate KL bounds

We now control the Kullback–Leibler (KL) divergence between neighboring instances that differ only in one coordinate of u or w. For j ∈ [d] define the neighbor sign vectors  (−u , u ), j ∈ J , j −j var u(j) := u, j ∈ Jsh ,

w(j) :=

 w,

j ∈ Jvar ,

(−wj , w−j ), j ∈ Jsh ,

where u−j and w−j denote the coordinates other than j.
PT (0)
Let Nj (T ) := t=1 1{j ∈ St } be the number of target rounds in which item j is included in the offered assortment.

61

Lemma 21 (Per–coordinate KL control). There exist finite constants ckl , c′kl > 0 depending only on (L0 , P , Cmin , Cmax ) such that for any policy π and any (u, w), KL Pu,w KL Pu,w

Pu(j) ,w



≤ (1 + H) ckl ∆2var Eu,w [Nj (T )] ,

Pu,w(j)



≤ c′kl ∆2shf Eu,w [Nj (T )] ,

j ∈ Jvar ,

j ∈ Jsh .

(49)
(50)

The proof in Section D.6.3 uses the chain rule for KL under adaptivity and a quadratic Lipschitz bound of the categorical KL in the logits, together with the fact that flipping a single intercept changes only one logit, with magnitude at most a constant multiple of ∆var or ∆shf .
To turn these into explicit bounds we need a symmetry property of the exposure counts Nj (T ). We follow a standard device in minimax lower bounds and apply a random permutation to item labels.
Lemma 22 (Exchangeability of exposures). Let Π be a uniform random permutation of the item indices [d], independent of (u, w) and of the policy’s internal randomness. Consider the permuted instances in which all intercepts are relabeled by Π, but the policy is not (0)
informed of Π. Denote by Nj (T ) the number of times item j appears in St under this randomization. Then for any fixed policy π and any (u, w), E[Nj (T )] =

KT d

for all j ∈ [d],

(51)

where the expectation is with respect to (u, w), Π, the policy’s randomness, and the trajectory.
The proof (Section D.6.4) uses the exchangeability of the permuted labels and the fact P that j Nj (T ) = KT .
Combining Lemmas 21 and 22 yields the simplified average KL bounds KL Pu,w KL Pu,w

D.4

Pu(j) ,w



≤ (1 + H) ckl ∆2var

Pu,w(j)



KT c′kl ∆2shf ,

≤

d

KT , d

j ∈ Jvar , (52)
j ∈ Jsh .

Testing reduction and revenue separation

We now relate the difficulty of testing the sign of a coordinate to a revenue gap, using Lemmas 19 and 20.
Mixture distributions and total variation. Fix a coordinate j ∈ Jvar . Let P+,j be the mixture law of ZT under the uniform prior on (u, w) conditioned on uj = +1 (and averaging over all other signs and the permutation Π), and let P−,j be defined analogously 62

for uj = −1. By convexity of KL, KL P+,j

P−,j



h

≤ E KL Pu,w

Pu(j) ,w

i KT uj = +1 ≤ (1 + H) ckl ∆2var .
d



By Pinsker’s inequality, TV P+,j , P−,j



≤

q

r 1 KL P+,j ∥ P−,j 2



≤

(1 + H) ckl ∆2var KT .
2 d

(53)

e+,j (resp. P e−,j ) be the mixture law conditioned on wj = +1 Similarly, for j ∈ Jsh , let P (resp. wj = −1) and averaged over (u, w−j , Π). Convexity and (52) give 



e+,j KL P

≤ c′kl ∆2shf

e−,j P

hence

r e+,j , P e−,j TV P



≤

KT , d

c′kl ∆2shf KT .
2 d

(54)

Per–coordinate revenue separation. By Lemma 19(ii), for |β| ≤ 2∆shf we have r∗ (+∆var ) − r∗ (−∆var ) ≥

2q ∆var , γ

r∗ (+∆shf ) − r∗ (−∆shf ) ≥

2q ∆shf .
γ

Denote these single–item gaps by ∆(var)
:= r∗ (+∆var ) − r∗ (−∆var ), r

∆(shf)
:= r∗ (+∆shf ) − r∗ (−∆shf ).
r

(h)

In our intercept construction, all βj lie in [−β0 , β0 ] for sufficiently small ∆var , ∆shf , so Lemma 20 applies with the same constants (q, q, α) for all coordinates.
Consider first j ∈ Jvar . Since along these coordinates sources and target share the same intercepts (eq. (47)), the clairvoyant target–market value for any (u, w) obeys  ∗ RK β (0) (u, w) ≥

1 1 + Kα

X

 (0)
r∗ βi (u, w) .

(55)

∗ (β (0) (u,w))
i∈SK

∗ By Lemma 19, the sum on the right is maximized by taking SK (β (0) (u, w)) equal to the indices of the K largest intercepts. In particular, among the d − s0 shared coordinates ∗ in Jvar , the Kvar := max{K − s0 , 0} largest ones (by intercept) belong to SK . Under the uniform prior on u and the random permutation Π, every j ∈ Jvar is equally likely to be one of these Kvar “shared slots.” Therefore, for any j ∈ Jvar ,

 ∗ P j ∈ SK (β (0) (u, w)) =

63

Kvar .
d − s0

∗ On those rounds t when j ∈ SK (β (0) (u, w)), the sign of uj determines whether item j appears with intercept +∆var or −∆var in the clairvoyant top–K set, creating a per–round revenue difference of at least (var)

2q ∆r ≥ ∆var .
1 + Kα γ(1 + Kα)
Similarly, for j ∈ Jsh , along those coordinates we have source intercept 0 and target intercept ∆shf wj (eq. (48)). Since the magnitude ∆shf is strictly larger than any shared magnitude ∆var (we will enforce ∆shf > ∆var below), every shifted coordinate with wj = +1 has a strictly larger intercept than any shared positive coordinate. When s0 ≤ K, it follows ∗ that for each (u, w) the clairvoyant SK (β (0) (u, w)) contains all indices j ∈ Jsh with wj = +1.
Thus for each such j,  ∗ P j ∈ SK (β (0) (u, w)) wj = +1 = 1, and on those instances, flipping wj from +1 to −1 induces a per–round revenue decrease of at least (shf)
2q ∆r ≥ ∆shf .
1 + Kα γ(1 + Kα)
(var)

(shf)

Summarizing, there exist constants cgap , cgap > 0 depending only on (L0 , P , q, q) such that ∗ per–round gap for j ∈ Jvar when j ∈ SK ≥ c(var)
(56)
gap ∆var , and ∗ per–round gap for j ∈ Jsh when j ∈ SK ≥ c(shf)
gap ∆shf .

D.5

(57)

Lower bound via a coordinate-wise testing argument

We now connect the testing difficulty and the revenue gaps to a minimax regret lower bound.
We work with the uniform product prior on (u, w) and the random permutation Π described above. Let RT (π; u, w) denote the regret of policy π over horizon T against the clairvoyant benchmark under instance (u, w), and let RT (π) be the resulting Bayes risk:
  RT (π) := Eu,w,Π RT (π; u, w) , where the expectation is over (u, w), Π, and the trajectory induced by π. By Yao’s minimax principle, inf sup Eu,w [RT (π; u, w)] ≥ inf RT (π).
π (u,w)

π

Thus it suffices to lower bound RT (π) uniformly over policies π.
The argument splits naturally into two parts: the contribution from the shared coordi-

64

nates Jvar and the contribution from the shifted coordinates Jsh .
D.5.1

Variance term: shared coordinates Jvar

Fix j ∈ Jvar . Consider the binary testing problem:
H+,j : uj = +1

vs.

H−,j : uj = −1,

with the prior P(H+,j ) = P(H−,j ) = 1/2 and all other (u−j , w, Π) drawn from their respective priors. Under any policy π, let ψj (ZT ) ∈ {+1, −1} be any estimator of uj based on the full trajectory (possibly randomized). The optimal Bayes error probability for this test satisfies the Le Cam bound  inf max P(ψj = −1 | H+,j ), P(ψj = +1 | H−,j )
ψj

≥

 1 1 − TV(P+,j , P−,j ) , 2

where P±,j are the mixture laws defined before (53).
∗ (β (0) (u, w)), the expected singleOn the event that uj is misclassified by ψj and j ∈ SK (var)
round revenue of π is at least cgap ∆var below the clairvoyant optimum (by (56)). Integrating over time and the prior, the total expected contribution of coordinate j to the Bayes regret is at least h i (var,j)
(var)
∗ (0)
RT (π) ≥ cgap ∆var E 1{j ∈ SK (β (u, w))} · # rounds · 1{ψj misclassifies uj } .
∗ Since SK (β (0) (u, w)) is fixed given (u, w) and Π, and does not depend on the policy, the number of rounds is exactly T , and by the symmetry argument above,

 Kvar ∗ P j ∈ SK (β (0) (u, w)) = , d − s0 while the misclassification probability is at least 12 (1 − TV(P+,j , P−,j )). Hence (var,j)
RT (π)

≥

c(var)
gap ∆var T

 Kvar 1  · 1 − TV(P+,j , P−,j ) .
d − s0 2

(58)

Using the TV bound (53) and choosing ∆var so that the upper bound on TV is bounded away from 1, we obtain an explicit lower bound. Specifically, take s ∆var :=

d , 4(1 + H) ckl KT

(59)

so that r TV(P+,j , P−,j ) ≤

(1 + H)ckl ∆2var KT · = 2 d 65

r

1 1 1 3 · = √ < .
2 4 4 2 2

Hence 1 − TV(P+,j , P−,j ) ≥ cTV for some universal cTV ∈ (0, 1).
(var)
Substituting into (58) and using ∆r ≥ (2q/γ)∆var , we find (var,j)

RT

(π) ≥ cvar,1 ∆var T

Kvar Kvar = cvar,1 T d − s0 d − s0

s

d , (1 + H) KT

for some constant cvar,1 > 0 depending only on (L0 , P , q, q). Summing over all j ∈ Jvar and using |Jvar | = d − s0 , (var)
RT (π) :=

X

(var,j)
RT (π)

j∈Jvar

s d Kvar T ≥ (d − s0 ) · cvar,1 d − s0 (1 + H) KT s r dT K (d − s0 ) T = cvar,1 Kvar ≥ cvar,2 , (1 + H) K 1+H

(60)

p where in the last step we used Kvar = max{K − s0 , 0} and absorbed the factor (d − s0 )/d into the constant cvar,2 (this factor is at most 1 and does not depend on T, K, H). When s0 = 0, Kvar = K and (60) matches the desired scaling exactly.
D.5.2

Transfer term: shifted coordinates Jsh

We now handle the s0 shifted coordinates Jsh . For each j ∈ Jsh , consider the binary hypotheses H+,j : wj = +1 vs.
H−,j : wj = −1, with equal prior probabilities and averaging over (u, w−j , Π). As before, let ϕj (ZT ) ∈ {+1, −1} be any estimator of wj . The optimal Bayes error is at least  1 e+,j , P e−,j )
1 − TV(P 2 e±,j are defined before (54).
by Le Cam’s lemma, where P ∗ On instances with wj = +1, the clairvoyant SK always includes j (as long as s0 ≤ (shf)
K), and flipping wj to −1 decreases the per–round optimal revenue by at least cgap ∆shf (equation (57)). Thus the contribution from coordinate j to the Bayes regret satisfies (sh,j)
RT (π)

 1 1 e e ≥ · 1 − TV(P+,j , P−,j ) c(shf)
gap ∆shf T, 2 2

(61)

where the first factor 1/2 is the prior probability of wj = +1, and the second 1/2 comes from the Bayes testing error lower bound.
From (54), choose s d ∆shf := , (62)
′ 4ckl KT 66

so that

r e+,j , P e−,j ) ≤ TV(P

1 3 c′kl ∆2shf KT · = √ < , 2 d 4 2 2 (shf)

e+,j , P e−,j ) ≥ c′ for some constant c′ ∈ (0, 1). Using ∆r and hence 1 − TV(P TV TV (2q/γ)∆shf we obtain r (sh,j)
RT (π)

≥ csh,1 ∆shf T = csh,1 T

≥

d , KT

for some constant csh,1 > 0 depending only on (L0 , P , q, q). Summing over the s0 shifted coordinates and using that each coordinate’s regret is realized on disjoint indices, (sh)

RT (π) :=

X

(sh,j)

RT

(π) ≥ csh,2 s0

√ K T,

(63)

j∈Jsh

√ where we have absorbed the factor d into csh,2 since d is fixed and the lower bound is uniform in T, K, H.
Importantly, the KL bound (50) for shifted coordinates does not have a factor (1+H), so √ there is no 1/ 1 + H improvement in (63): source observations are identically distributed under both wj = +1 and wj = −1, and hence provide no information about the target-only shifts.
D.5.3

Combining the two contributions

Combining (60) and (63), we obtain that for any policy π, r RT (π) ≥ c1

√ K (d − s0 ) T + c2 s0 K T , 1+H

for some constants c1 , c2 > 0 depending only on (L0 , P , Cmin , Cmax ). By Yao’s principle,   inf sup Eu,w RT (π; u, w) ≥ inf RT (π), π (u,w)

π

which proves the stated lower bound in Theorem 8 (up to a relabeling of constants).
Remark 1 (On the condition s0 ≤ K). When s0 > K, at most min{s0 , K} shifted coordinates can enter the clairvoyant top–K set per round. Repeating the above argument √ with min{s0 , K} in place of s0 yields a lower bound of order min{s0 , K} KT for the shift contribution, without affecting the qualitative conclusions.
Remark 2 (Realization via contextual features). Our construction was presented in terms of intercept vectors β (h) . It can be realized within the original contextual model (1) by taking (h)
xjt = ej for each item j and all t, setting θ (h) so that ⟨xj , θ (h) ⟩ = βj and γ (h) so that 67

⟨xj , γ (h) ⟩ = L0 . The resulting instances satisfy Assumptions 2–1, and the above analysis applies directly.

D.6

Technical Lemmas for the Lower Bound

D.6.1

Proof of Lemma 19

The argument is identical to the standard analysis of unimodality and envelope theorem for single–item MNL pricing; we reproduce it here for completeness.
For fixed β, p 7→ r(p; β) is continuous on the nonempty compact set P , so a maximizer ∗ p (β) ∈ P exists (Weierstrass theorem). Write q(p) := q(p; β) for brevity. We have   ′ ′ ′ q (p) = −γq(p)(1 − q(p)), r (p; β) = q(p) + p q (p) = q(p) 1 − γp (1 − q(p)) .
Define ϕ(p) := γp (1 − q(p)). Then    ϕ′ (p) = γ (1 − q(p)) + p(−q ′ (p)) = γ(1 − q(p)) 1 + γp q(p) > 0, so ϕ is strictly increasing on [0, ∞). At p = 0, q(0) ∈ (0, 1), so r′ (0; β) = q(0) > 0, whereas limp→∞ ϕ(p) = ∞, hence limp→∞ r′ (p; β) = −∞. Therefore r′ (·; β) crosses zero exactly once on (0, ∞), say at p◦ (β) > 0, with r′ > 0 on (0, p◦ ) and r′ < 0 on (p◦ , ∞); thus r(·; β)
is strictly unimodal and has a unique maximizer on P , namely p∗ (β) = min{p◦ (β), P } ∈ (0, P ]. If p∗ (β) < P , then r′ (p∗ (β); β) = 0, which is exactly (43).
For part (ii), on B the maximizer is unique and interior, and r is C 1 in (p, β). Danskin’s envelope theorem therefore yields   d ∗ ∂ r (β) = r p∗ (β); β = p∗ (β) q ∗ (β) 1 − q ∗ (β) .
dβ ∂β Using the first-order condition (43), γp∗ (β)(1 − q ∗ (β)) = 1, we get d ∗ q ∗ (β)
r (β) = .
dβ γ Since q ∗ (β) ∈ [q, q] on B, hq qi d ∗ r (β) ∈ , dβ γ γ

(β ∈ B).

By the mean–value theorem, for any β, β ′ ∈ B there exists βe between them such that |r∗ (β) − r∗ (β ′ )| =

hq qi d ∗ e r (β) · |β − β ′ | ∈ , · |β − β ′ |.
dβ γ γ

This proves (44) and strict monotonicity on B.

68

□

D.6.2

Proof of Lemma 20

Let S ⊆ [N ] with |S| ≤ K, and set pi = p∗ (βi ) for i ∈ S (Lemma 19). Write θi := eβi −γpi and qi∗ := θi /(1 + θi ) = q ∗ (βi ), and ri∗ := r∗ (βi ) = pi qi∗ . The MNL revenue with outside option is P i∈S pi θi P .
R(S, p; β) = 1 + j∈S θj Since ri∗ = pi θi /(1 + θi ), we have pi θi = ri∗ (1 + θi ), so ∗ i∈S ri (1 + θi )

P ∗ i∈S ri P ≥ , 1 + j∈S θj

P R(S, p; β) =

1+

P

j∈S θj

where the inequality uses 1 + θi ≥ 1. Using qi∗ ≤ q, θi = so

qi∗ q ≤ = α, ∗ 1 − qi 1−q

P

j∈S θj ≤ Kα and therefore ∗ i∈S ri

P R(S, p; β) ≥

=

1 + Kα

X 1 r∗ (βi ).
1 + Kα i∈S

∗ This proves (45). For (46), note that r∗ is strictly increasing on B, so any set SK (β) that P maximizes i∈S r∗ (βi ) over |S| ≤ K is the set of K largest βi . Maximizing R(S, p; β) over ∗ (S, p) and evaluating the right-hand side on SK (β) with pi = p∗ (βi ) yields (46).
□

D.6.3

Proof of Lemma 21

Fix a policy π and an instance (u, w). Let Ft−1 be the sigma–field generated by the history (0)
(0)
up to time t − 1. Conditionally on Ft−1 , the action (St , pt ) is Ft−1 –measurable and the (0)
(1)
(H)
observation vector (Yt , Yt , . . . , Yt ) is drawn as an (1 + H)–fold product of categorical variables with parameter vector determined by the current logits. For two neighboring instances (say, (u, w) and (u(j) , w)), the chain rule for KL divergence gives KL Pu,w



Pu(j) ,w =

T X

h



Eu,w KL Cat

⊗(1+H)

(pt )

Cat

⊗(1+H)

 i (e pt ) Ft−1 ,

(64)

t=1 (0)

e t are the categorical probability vectors over St ∪ {0} under the two where pt and p instances, given Ft−1 . Since the 1 + H draws are conditionally i.i.d.,   KL Cat⊗(1+H) (pt ) Cat⊗(1+H) (e pt ) = (1 + H) KL(Cat(pt ) ∥ Cat(e pt ))

69

for j ∈ Jvar . For j ∈ Jsh , the source logits coincide under both (u, w) and (u, w(j) ), hence only the target factor contributes and the multiplier (1 + H) drops.
et be the logit vectors under the two instances, To bound the one-step KL, let zt and z P e t = softmax(e and pt = softmax(zt ), p zt ). The softmax log-partition A(z) = log(1 + i ezi )
is C 2 with Hessian ∇2 A(ξ) = Diag(p) − pp⊤ having operator norm at most 1/4. The KL divergence equals the Bregman divergence of A, et ⟩, KL(Cat(pt ) ∥ Cat(e pt )) = A(zt ) − A(e zt ) − ⟨∇A(e zt ), zt − z so by the mean–value form of smoothness, KL(Cat(pt ) ∥ Cat(e pt )) ≤

1 1 et )⊤ ∇2 A(ξt )(zt − z et ) ≤ ∥zt − z et ∥22 , (zt − z 2 8

et .
for some ξt on the line segment between zt and z Under our single–coordinate flips, the two instances differ in exactly one intercept: if (0)
et , and if j ∈ St(0) , then the logit for item j changes by ±2∆var (shared)
j∈ / St , then zt = z or ±2∆shf (shifted), with all other logits unchanged. Therefore (0)

et ∥22 ≤ 4 ∆2var 1{j ∈ St } ∥zt − z for j ∈ Jvar and analogously with ∆shf for j ∈ Jsh . Thus (0)

KL(Cat(pt ) ∥ Cat(e pt )) ≤ c0 ∆2var 1{j ∈ St },

1 c0 := , 2

and similarly with ∆shf for j ∈ Jsh . Substituting into (64), summing over t, and taking expectations over (u, w) gives (49)–(50) with ckl = c0 and c′kl = c0 , or slightly larger constants if we absorb model-dependent refinements (e.g., lower bounds on choice probabilities) into them.
□ D.6.4

Proof of Lemma 22

Let Π be a uniform random permutation of [d], independent of (u, w) and the policy’s randomness. In the permuted instance, item labels are relabeled by Π, but the policy acts in the same way as a function of the history (it does not know Π). For each t and j, define P (0)
It,j := 1{j ∈ St } and Nj (T ) := Tt=1 It,j .
By symmetry of the random permutation, for each fixed t the joint law of (It,1 , . . . , It,d )
is exchangeable. Hence for all j, E[It,j ] = E[It,1 ]
(0)

On the other hand, |St | =

Pd

for all t.

j=1 It,j = K almost surely, so taking expectations and using

70

exchangeability, d X

E[It,j ] = E

j=1

d hX

i

It,j = K

=⇒

E[It,j ] =

j=1

K d

for all j.

Summing over t = 1, . . . , T gives E[Nj (T )] =

T X

E[It,j ] =

t=1

KT , d

as claimed.

□

Appendix E E.1

Miscellaneous Proofs

Proof of Lemma 23

Lemma 23 (Bounded optimal price). Consider the optimization problem (??). Then the revenue maximizer (p∗it )i∈S admits a common finite upper bound P̄ , i.e., (p∗it )i∈S ∈ [0, P̄ ].
  P Let qit (p) := exp{vit (pi )} 1 + j∈S exp{vjt (pj )} be the MNL choice probability and write X pj qjt (p).
G(p) := j∈S

The expected revenue is Rt (S, p) = G(p). Differentiating with respect to pi and using the standard MNL calculus ∂qit = qit (1 − qit )vit′ (pi ), ∂pi

∂qjt = −qjt qit vit′ (pi ) (j ̸= i), ∂pi

we obtain ∂Rt ∂qit X ∂qjt = qit + pi + pj ∂pi ∂pi ∂pi j̸=i !
= qit + qit vit′ (pi ) pi (1 − qit ) −

X

pj qjt (p)

j̸=i

= qit + qit vit′ (pi ) pi − G(p)



  = qit 1 + vit′ (pi ) pi − G(p) .
At any maximizer p∗ of Rt (S, ·) with p∗i > 0 (the multi-item revenue is smooth and vanishes as any pi → ∞, so maximizers are finite and satisfy the first-order condition), we must have  ∂Rt ∗ (p ) = 0 =⇒ 1 + vit′ (p∗it ) p∗it − G(p∗ ) = 0 for all i ∈ S, ∂pi 71

since qit (p∗ ) > 0. Let Bt := G(p∗ ) =

X

∗ p∗jt qjt ,

∗ qjt := qjt (p∗ ).

j∈S

Then the first-order condition can be rewritten as p∗it +

1 vit′ (p∗it )

= Bt ,

∀i ∈ S.

(65)

We now eliminate p∗it in favour of Bt . Summing p∗it qit∗ over i ∈ S and using (65), we obtain  X X 1 ∗ ∗ Bt − ′ ∗ Bt = pit qit = qit∗ v (p )
it it i∈S i∈S X X q∗ it .
qit∗ − = Bt ′ v (p∗ )
i∈S i∈S it it ∗ Writing q0t for the outside-option probability, we have ∗ Bt q0t =−

∗ ∗ i∈S qit = 1 − q0t , so

P

qit∗ .
′ ∗ v (p )
it it i∈S

X

∗

∗ Using qit∗ = evit (pit ) q0t , we get

∗ ∗ Bt q0t = −q0t

X evit (p∗it )
i∈S

=⇒

vit′ (p∗it )

Bt = −

X evit (p∗it )
i∈S

vit′ (p∗it )

.

Thus, for each i, the quantity evit (p)
git (B) := − ′ vit (p)

with p ≥ 0 such that p +

1 vit′ (p)

=B

is well-defined at B = Bt (since p = p∗it is feasible), and we can write the fixed-point equation X Bt = git (Bt ).
(66)
i∈S

Next we upper bound git (B) in terms of B. By Assumption 2, vit′ (p) ≤ −L0 < 0 and vit is decreasing. Moreover, bounded parameters and covariates imply there exists a finite constant M > 0 such that vit (0) ≤ M for all (i, t), (67)
and hence, for all p ≥ 0, vit (p) ≤ vit (0) − L0 p ≤ M − L0 p.
72

(68)

Now fix B ≥ 0 and any feasible p ≥ 0 in the definition of git (B), i.e., p + 1/vit′ (p) = B.
Since vit′ (p) < 0 we have 1/vit′ (p) ≤ 0, and thus p=B−

1 vit′ (p)

≥ B.

(69)

Using monotonicity and (68), vit (p) ≤ vit (B) ≤ M − L0 B.
Furthermore, vit′ (p) ≤ −L0 < 0 implies 1 1 − ′ ≤ , vit (p)
L0

=⇒

evit (p)
1 vit (p)
− ′ ≤ e .
vit (p)
L0

Combining, evit (p)
1 vit (p)
1 M −L0 B git (B) = − ′ ≤ e ≤ e .
vit (p)
L0 L0 Summing over i ∈ S and using |S| ≤ K in (66), we obtain Bt =

X

git (Bt ) ≤

i∈S

K M −L0 Bt e .
L0

Equivalently, L0 Bt eL0 Bt ≤ KeM .
Let W (·) be the Lambert–W function. Then  L0 Bt ≤ W KeM ,

=⇒

Bt ≤

 1 W KeM .
L0

Using the standard bound W (z) ≤ log z for z ≥ e (or the finer bound W (z) ≤ log z − log log z + 0.6), we get M + log K + c0 Bt ≤ =: P0 , L0 for some absolute constant c0 > 0.
Finally, from (65) and vit′ (p) ≤ −L0 we have p∗it = Bt −

1 vit′ (p∗it )

≤ Bt +

1 1 ≤ P0 + =: P̄ .
L0 L0

Since p∗it ≥ 0 by feasibility, this shows that every optimal price lies in [0, P̄ ], which completes the proof.
□

73

E.2

Proof of Lemma 24

Lemma 24 (Monotone–Lipschitz envelope). Let v̄it (p) be any pointwise upper bound of vit (p) for all p ∈ [0, P̄ ]. Define n o ′ ′ veit (p) := min v̄ (p )
− L (p − p )
.
it 0 ′ p ≤p

Then veit (p) is decreasing, satisfies veit (p) ≤ veit (p′ ) − L0 (p − p′ ) for all p ≥ p′ , and vit (p) ≤ veit (p) for all p.
In the following proof we suppress the (i, t) subscripts for clarity and write v(p), v̄(p), and ve(p).
By Assumption 2, the utility takes the form v(p) = ⟨x, θ⟩ − ⟨x, γ⟩p,

⟨x, γ⟩ ≥ L0 > 0,

so v is differentiable with derivative v ′ (p) = −⟨x, γ⟩ ≤ −L0 < 0.
Thus v is decreasing, and for any 0 ≤ p′ ≤ p ≤ P , Z p ′ v(p) − v(p ) = v ′ (s) ds ≤ −L0 (p − p′ ) =⇒

v(p) ≤ v(p′ ) − L0 (p − p′ ).

(70)

p′

Recall the definition  ′ ve(p) := min v̄(p ) − L0 (p − p′ ) , ′ p ≤p

p ∈ [0, P ],

(71)

where v̄ is any (continuous) pointwise upper bound of v on [0, P ].
(i) ve is well-defined and ve ≤ v̄. Since v̄ is continuous, the function g(p, p′ ) := v̄(p′ ) − L0 (p − p′ )
is continuous on the compact set {(p, p′ ) : 0 ≤ p′ ≤ p ≤ P }, so the minimum in (71) exists for every p. Moreover, the feasible set for p′ contains p itself, hence ve(p) ≤ v̄(p) − L0 (p − p) = v̄(p),

∀p ∈ [0, P ].

(ii) ve is decreasing. Let 0 ≤ p1 < p2 ≤ P . Since the feasible set for p2 strictly contains that for p1 ,  ′  ′ ′ ve(p2 ) = min v̄(p )
− L (p − p )
≤ min v̄(p ) − L0 (p2 − p′ ) .
0 2 ′ ′ p ≤p2

p ≤p1

74

For each fixed p′ ≤ p1 , the map p 7→ v̄(p′ ) − L0 (p − p′ ) is decreasing in p, so v̄(p′ ) − L0 (p2 − p′ ) ≤ v̄(p′ ) − L0 (p1 − p′ ).
Taking the minimum over p′ ≤ p1 yields  ′ v̄(p ) − L0 (p1 − p′ ) = ve(p1 ), ve(p2 ) ≤ min ′ p ≤p1

so ve is decreasing.
(iii) ve is L0 -Lipschitz. Let 0 ≤ p1 < p2 ≤ P , and let p∗2 ∈ [0, p2 ] be a minimizer in the definition of ve(p2 ):
ve(p2 ) = v̄(p∗2 ) − L0 (p2 − p∗2 ).
Then, by feasibility of p∗2 for p1 , ve(p1 ) ≤ v̄(p∗2 ) − L0 (p1 − p∗2 )
 = v̄(p∗2 ) − L0 (p2 − p∗2 ) +L0 (p2 − p1 ), | {z } = v e(p2 )

so ve(p1 ) − ve(p2 ) ≤ L0 (p2 − p1 ). Combined with the monotonicity from (ii), this implies |e v (p2 ) − ve(p1 )| ≤ L0 |p2 − p1 |, i.e., ve is L0 -Lipschitz on [0, P ].
(iv) Majorization: v ≤ ve. From (70), for any 0 ≤ p′ ≤ p ≤ P , v(p) ≤ v(p′ ) − L0 (p − p′ ).
Because v̄ is a pointwise upper bound of v, we have v(p′ ) ≤ v̄(p′ ), so v(p) ≤ v̄(p′ ) − L0 (p − p′ )

for all p′ ≤ p.

Taking the minimum over p′ ≤ p gives  ′ v(p) ≤ min v̄(p ) − L0 (p − p′ ) = ve(p), ′ p ≤p

for every p ∈ [0, P ].
Combining (i)–(iv), we conclude that ve is decreasing, L0 -Lipschitz, and satisfies v(p) ≤ ve(p) ≤ v̄(p) for all p ∈ [0, P ], as claimed.
□

75

Appendix F

Additional Experimental Results

Figures 3 present the cumulative regret under other parameter configurations. The findings here echo the same trend: regret decreases with more source markets, and TJAP consistently outperforms the pricing-only baselines. Taken together, the experiments demonstrate that our algorithm is robust and effective under different sparsity levels.

76

(a)
d = 10, s0 = 3, K = 5, N = 100

(b)
d = 20, s0 = 4, K = 5, N = 30

(c)
d = 50, s0 = 10, K = 5, N = 30

(d)
d = 10, s0 = 3, K = 5, N = 100

(e)
d = 20, s0 = 4, K = 5, N = 30

(f)
d = 50, s0 = 10, K = 5, N = 30

(g)
d = 10, s0 = 2, K = 5, N = 100

(h)
d = 20, s0 = 4, K = 5, N = 100

(i)
d = 50, s0 = 10, K = 5, N = 100

(j)
d = 10, s0 = 2, K = 5, N = 100

(k)
d = 20, s0 = 4, K = 5, N = 100

(l)
d = 50, s0 = 10, K = 5, N = 100

77

(a)
d = 10, s0 = 3, K = 5, N = 30

(b)
d = 20, s0 = 6, K = 5, N = 30

(c)
d = 50, s0 = 15, K = 5, N = 30

(d)
d = 10, s0 = 3, K = 5, N = 30

(e)
d = 20, s0 = 6, K = 5, N = 30

(f)
d = 50, s0 = 15, K = 5, N = 30

Figure 3: Cumulative regret on synthetic instances under varying feature dimension d, sparsity level s0 , and catalog size N . Top row: TJAP with H ∈ {0, 1, 3, 5} compared against CAP, M3P, and ONS–MPP. Bottom row: TJAP with H ∈ {0, 1, 3, 5} compared against the pooled estimator Pool(H) for H ∈ {1, 3, 5}. Each curve is averaged over 10 independent runs; all methods share the same price range [0, P ] and observe identical contexts.

78


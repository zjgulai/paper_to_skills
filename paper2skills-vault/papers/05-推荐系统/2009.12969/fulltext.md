<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2009.12969
     paper_id : 2009.12969
     source   : https://arxiv.org/html/2009.12969v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Simultaneous Relevance and Diversity:
A New Recommendation Inference Approach

Yifang Liu    Zhentao Xu    Qiyuan An    Yang Yi    Yanzhi Wang    Trevor Hastie

###### Abstract

Relevance and diversity are both important to the success of recommender systems, as they help users to discover from a large pool of items a compact set of candidates that are not only interesting but exploratory as well. The challenge is that relevance and diversity usually act as two competing objectives in conventional recommender systems, which necessities the classic trade-off between exploitation and exploration. Traditionally, higher diversity often means sacrifice on relevance and vice versa. We propose a new approach, heterogeneous inference, which extends the general collaborative filtering (CF) by introducing a new way of CF inference, negative-to-positive. Heterogeneous inference achieves divergent relevance, where relevance and diversity support each other as two collaborating objectives in one recommendation model, and where recommendation diversity is an inherent outcome of the relevance inference process. Benefiting from its succinctness and flexibility, our approach is applicable to a wide range of recommendation scenarios/use-cases at various sophistication levels. Our analysis and experiments on public datasets and real-world production data show that our approach outperforms existing methods on relevance and diversity simultaneously.

1 Smule, 2 Virginia Tech, 3 Northeastern University, 4 Stanford University ifnliu@gmail.com, frankxu@umich.edu, anqiyuan,yangyi8@vt.edu, yanz.wang@northeastern.edu, hastie@stanford.edu

## Introduction

Relevance and diversity are both important to recommender systems, as they help users to discover from a large pool of items a compact set of candidates that are not only interesting/relevant but inspiring/exploratory as well.

As a popular recommendation approach, the conventional collaborative filtering (CF) focuses primarily on inferring the relevance of items to individual users. The relevance is usually translated from inter-item similarity, which can be inferred by positive feedback to different items from the same user(s). Because both the source item and target item in the inter-item similarity inference are positive feedback (a.k.a. positive engagement), we consider conventional CF being based on positive-to-positive (p2p) inference.

CF by p2p is adopted in a wide variety of successful recommendation algorithms. A main strength of CF is the high relevance and precision of its recommendations, while it also leads to a challenge for CF — shrinking recommendation diversity: over time, the recommendation becomes more and more focused on what the user has shown strong interest in, while the scope of predicted user interest becomes narrower and narrower. As a result, less variety of content is recommended to the user. The converging recommendation scope clearly impairs the user’s exploration horizon. In the long term, the converging recommendation feedback loop excludes a large space of potentially interesting items/topics from recommendations to the user. For this reason, this paper considers conventional CF as convergent relevance (CR) oriented. In other words, on the trade-off between exploitation vs. exploration, conventional CF tends to weigh mostly on exploitation at the cost of poor exploration. The concern of shrinking diversity is also known as the Rabbit Hole problem, the Filter Bubble issue, or the Echo Chamber effect in different domains (Jiang et al. 2019; Ge et al. 2020; Antikacioglu and Ravi 2017; Nguyen et al. 2014; Knijnenburg, Sivakumar, and Wilkinson 2016).

To tackle this challenge at its core, we propose a new recommendation approach, Heterogeneous Inference (HI), which fundamentally extends the CF approach by introducing into CF a new channel of relevance inference, negative-to-positive (n2p) inference, in addition to the existing p2p inference. Like p2p, n2p makes the observation that when users are not interested in one item (i.e., the negative) they tend to be interested in some other items (i.e., the positive). In a loose statistical sense, ”similarity” in relevance inference context can be viewed as a proxy for ”correlation” between entities. CF only cares about positive correlation (p2p). HI leverages both positive correlation (p2p) and negative correlation (n2p) in one cohesive model.

HI’s potential to address the shrinking diversity problem can be mostly attributed to the divergent nature of n2p inference. Intuitively, there are many possible positives given one negative. Therefore, n2p is able to infuse diversity into its own relevance inference. By n2p inference, both relevance and diversity are inherent outcomes of the same relevance inference process. The intrinsically-diverse relevance is at the core of our Divergent Relevance (DR) concept.

The main contribution of this paper includes:

(1) We present a new concept, divergent relevance (DR), for achieving relevance and diversity collaboratively in recommendations as two inherent outcomes of one relevance inference process. This is realized by our new recommendation approach, heterogeneous inference (HI), which extends CF with a negative-to-positive (n2p) inference component.

(2) HI provides a flexible, general framework for divergent relevance inference, which is suitable for different levels of model/algorithm sophistication and is applicable to a variety of scenarios in recommendation. HI can work either as a standalone recommendation algorithm (e.g., a real-time personalized recommender, a candidate generator) or as a module integrated in a sophisticated recommendation model (e.g., an embedding module inside a neural network).

(3) HI’s effectiveness and efficiency are demonstrated through evaluations on both a well-known public recommendation dataset and real-world industrial production data.

## Preliminaries

*Figure 1: Distribution of relevance over items. (a) p2p convergent relevance: precise and narrow relevance prediction. (b) n2p divergent relevance: relevance prediction spread out across all topics. (c) p2p relevance distribution: largely skewed. (d) n2p relevance distribution: relatively flat and even.*

In general, CF and HI are applicable to both implicit feedback (e.g., user selecting/skipping an item) and explicit feedback (e.g., user rating an item). User feedback can assume different value types, e.g., real values, discrete numbers, and categories. For the sake of clear demonstration and comparison, this paper explains HI algorithms with binary user feedback observations. Any real-valued feedback is converted into either a positive feedback or a negative feedback according to a preset threshold, before being used as the label of a data point during evaluation.

### Notation and problem statement

Given an set of observed user-item engagement examples,

$\mathbf{X}$ An m-by-n matrix representing the positive feedback of $m$ users to $n$ items. Each element $x_{i,j}\in\{0,1\}$ indicates whether user $i$ has a positive feedback to an impression of item $j$: $1$ means positive feedback, $0$ means negative or unobserved feedback.

$\mathbf{O}$ An m-by-n matrix representing the observation of $m$ users’ feedback to $n$ items. Each element $o_{i,j}\in\{0,1\}$ indicates whether user $i$ has an impression on item $j$: $1$ means observed, $0$ means not observed. Note that $\mathbf{O}$ is the information usually ignored by conventional recommendation algorithms.

$\tilde{x}_{i,j}$ An unobserved data point (i.e., $o_{i,j}=0$), whose binary value indicates whether user $i$ has a positive feedback to item $j$, when item $j$ is presented to user $i$.

###### Problem 1

Engagement Prediction Given a set of observed engagements by a group of users with a group of content items, predict $p(\tilde{x})$, the likelihood of a given user giving positive feedback to a specific item when the corresponding impression presents. In other words, the problem is to estimate the following posterior:

$\displaystyle p(\tilde{x}|\mathbf{X},\mathbf{O})$ | | | | (1) |

where $\tilde{x}$ can be any hypothetical future impression of item j on user i, $\tilde{x}_{i,j}$.

In conventional CF recommenders, because the information in $\mathbf{O}$ is usually overlooked, the engagement prediction target degenerates to a simpler posterior:

$\displaystyle p(\tilde{x}|\mathbf{X})$ | | | | (2) |

### Related work

If the relevance of item $j$ to user $i$ is implied by the likelihood of the user giving positive feedback to (being interested in) the item, i.e., $p(x_{i,j}=1)$, recommendation diversity can be interpreted as the distribution of the relevance over different recommendation items. The wider the relevance is spread out across different items, the higher the diversity is.

Conventional CF. CF comes in various flavors of algorithms or models. For example, user/item collaborative filtering (Zhao et al. 2015; Davidson et al. 2010; Mnih and Salakhutdinov 2008; Salakhutdinov and Mnih 2008; Srebro, Rennie, and Jaakkola 2012; He et al. 2014; Gomez-Uribe and Hunt 2016), deep learning (Covington, Adams, and Sargin 2016; Krichene et al. 2018; Cheng et al. 2016; Salakhutdinov, Mnih, and Hinton 2007; den Oord, Dieleman, and Schrauwen 2013; Wang, Wang, and Yeung 2015; Zhai et al. 2017), deep embedding models (Krichene et al. 2018; Okura et al. 2017), Factorization Machines (Rendle 2010), Matrix Factorization (Koren, Bell, and Volinsky 2009; Zhao et al. 2015; Lee and Seung 1999; Mnih and Salakhutdinov 2008; Salakhutdinov and Mnih 2008; Srebro, Rennie, and Jaakkola 2012), ALS (Hastie et al. 2015), SVD++ (Koren 2008), PITF (Rendle and Schmidt-Thieme 2010), and FPMC (Rendle, Freudenthaler, and Schmidt-Thieme 2010).

Regardless of the specific algorithm, recommendation relevance in CF relies on certain forms of p2p inference, which is derived from the user-item engagement matrix, $\mathbf{X}$. Recommendation diversity is the objective of a separate model/algorithm component (other than CF), which uses various different strategies explained later in this section. In existing recommendation approaches, diverse recommendation means deviation from top-relevance recommendation, i.e., diversity is in the opposite direction of conventional CF. Therefore, in traditional recommender systems the CF component and the diversity component compete against each other in two separate processes. Conventional CF has to trade a certain degree of relevance for a certain level of diversity in the recommendations, i.e., make a trade-off between the two competing objectives.

A batch recommendation process usually works in two stages: candidate generation and ranking. Next, we explain how CF by p2p works in matrix factorization fashion for candidate generation.

Step 1. rank-$k$ matrix factorization of $\mathbf{X}^{T}\mathbf{X}$:

$\displaystyle\mathbf{X}^{T}\mathbf{X}\approx\mbox{\ }\mathbf{P}\mbox{\ }\mathbf{Q},\mbox{\ where\ }\mathbf{P}\in\Re^{n\times k},\mathbf{Q}\in\Re^{k\times n}$ | | | | (3) |

Step 2. smooth approximation of $\mathbf{X}^{T}\mathbf{X}$:

$\displaystyle\mathbf{C}=\mbox{\ }\mathbf{P}\mbox{\ }\mathbf{Q},\mbox{\ where\ }\mathbf{C}\in\Re^{n\times n}$ | | | | (4) |

Step 3. inferred items’ relevance scores per user:

$\displaystyle{}^{*}\mathbf{X}=\mbox{\ }\mathbf{X}\mbox{\ }\mathbf{C},\mbox{\ where\ }^{*}\mathbf{X}\in\Re^{m\times n}$ | | | | (5) |

Step 1 obtains the input matrix of p2p item-to-item similarity, $\mathbf{X}^{T}\mathbf{X}$, by multiplying the transpose of the user-item positive engagement matrix with itself. It indicates the connection between items that received positive feedback from the same users. Step 2 calculates $\mathbf{C}=\mathbf{P}\mathbf{Q}$, a smoothed approximation of the p2p inter-item similarity matrix, based on the rank-$k$ matrix factorization from Step 1. $\mathbf{C}$ replaces zeros in $\mathbf{X}^{T}\mathbf{X}$ with similarity values estimated by the factorization. Step 3 estimates the relevance scores of all items for every individual user, by multiplying the original user-item matrix with the smoothed p2p inter-item similarity matrix. The recommendation candidates for each user can be selected as the items with the highest relevance scores in the corresponding row of matrix ${}^{*}\mathbf{X}$. This candidate relevance estimation procedure is typical CF by p2p.

Figure 1 (a) explains the convergent relevance (CR) nature of the CF by p2p, which causes the issue of shrinking recommendation diversity. Suppose we have 3 top-level latent topics: $A$, $B$, and $C$. Under each of these topics, we have sub-topics: 1, 2, and 3. Items similar to those in $A_{1}$ and $A_{2}$ that received positive feedback, will most likely fall under the same top-level topic, $A$. Most of them may even belong to one of the sub-topic there, e.g., $A_{1}$, which may be the focus of the next group of recommendations. Over time, the items recommended through p2p inference may converge to a highly-relevant but narrowing topic. Figure 1 (c) illustrates the skewed distribution of p2p relevance across all items.

Recommendation diversity improvement. A large amount of research has been devoted into enhancing recommendation diversity. The most direct method of enhancing diversity is using calculated metrics (e.g. Entropy (Noia et al. 2017), Item Similarity Scores (Bradley and Smyth 2001; Castagnos, Brun, and Boyer 2013; L’Huillier, Castagnos, and Boyer 2014)) as proxies of diversity (e.g. Item Popularity (Hurley and Zhang 2011; Adomavicius and Kwon 2012)) to augment the scoring models of existing recommender systems. Some applications leverage content-specific data, such as tags and topics (Zhang, Zheng, and Zeng 2016; Vargas et al. 2014), to ensure a good mix of items across different categories, i.e., to improve recommendation’s genre coverage. Calibrated Recommendations (Steck 2018) propose a way to encourage the recommendations in different categories/areas, based on the user’s interest distribution. More sophisticated approaches include graph-based algorithms that formulate diversity as a max-flow problem (Adomavicius and Kwon 2011) or Markov Chain (Paudel, F.Christoffel, and C.Newell 2016). Matrix factorization methods characterize item similarities by projecting their properties as latent features (Paudel, Haas, and Bernstein 2017).

Most of existing methods treat diversity as a competing objective against relevance. They improve recommendation diversity in a separate process (usually as a re-ranking criterion) from the one used for recommendation relevance improvement. As a result, any increase in recommendation diversity would hurt the quality of recommendation relevance, under the assumption that one has to choose between diversity and relevance in the trade-off.

Negative feedback in deep learning. Recently, some research is done to leverage both positive and negative feedback inside a deep learning model for recommendations, e.g., deep reinforcement learning (Zhao et al. 2018; Zou et al. 2019), deep neural network for interaction embedding (Xie et al. 2020; Zhang et al. 2019; Gauci et al. 2018). These methods achieved notable improvement on recommendation relevance, while their impact on recommendation diversity is unknown.

These methods do not provide a general, self-contained framework for relevance inference, i.e., the use of negative feedback is mixed into the overall recommendation model on a case-by-case basis. In fact, their implicit inference with negative feedback acts in an accessory role to support the main recommendation model for its use case. Meanwhile, their model complexity due to the tailored DNN design makes them impractical for time-constrained use cases, e.g., realtime personalized recommendation.

## Our approach

This section starts with establishing the crucial foundation of HI on our general-purpose feedback encoding scheme. Then, we present how HI simultaneously achieves relevance and diversity as two collaborating objectives in the inherent outcomes of one relevance inference model. Finally, exemplary algorithm implementations of HI for several recommendation scenarios/use-cases are explained.

### Feedback-Differentiating Encoding

Conventional feedback encoding (CONFE), $\mathbf{X}$, does not differentiate unknown feedback from negative feedback — usually, they are represented by the same values or are both missing in the dataset.

One straightforward way to encode the difference between negative and unknown is to represent negative with a number (e.g., $0$ or $-1$) and unknown as a non-number (e.g., Null or NaN) in the implementation of the CF algorithm. For example, Apache Spark realizes this type of encoding in their ALS algorithm implementation. However, this negative-unknown differentiating capability is not always supported in CF algorithm implementations.

Therefore, we propose a general-purpose feedback-differentiating encoding (FEEDE) scheme, which precisely encodes different types of feedback, regardless whether the CF algorithm implementation supports it or not. FEEDE demonstrates the feedback-differentiating capability by complementing the conventional positive feedback encoding, $\mathbf{X}$, with an additional negative feedback encoding, $\mathbf{Y}$,

$\displaystyle\mathbf{Y}=\mathbf{O}-\mathbf{X}$ | | | | (6) |

By this definition, in binary feedback context, element $y_{i,j}$ in $\mathbf{Y}$ represents whether an impression gets negative feedback:

$\displaystyle y_{i,j}=\begin{cases}1&\mbox{\ \ \ $x_{i,j}=0$ and $o_{i,j}=1$}\\
0&\mbox{\ \ \ ($x_{i,j}=1$ and $o_{i,j}=1$) or $o_{i,j}=0$}\\
\end{cases}$ | | | |

Note that for implicit feedback, FEEDE relies on an assumption that the impression of an item on a user is observable, i.e., $\mathbf{O}$ is known. FEEDE’s advantage over CONFE is explained in the following analysis.

If the feedback is a range of integers (like typical ratings on a scale of $1$ to $h$), the definition of $\mathbf{Y}$ still works with an extra step of rating normalization. For example, with $\{1,...,5\}$ ratings on individual items: each element $x_{i,j}$ in $\mathbf{X}$ is obtained as $x_{i,j}=Rating_{i,j}/6$. Then, the same definition, $\mathbf{Y}=\mathbf{O}-\mathbf{X}$, implies:

$\displaystyle y_{i,j}=\begin{cases}1-x_{i,j}&\mbox{\ \ \ $o_{i,j}=1$}\\
0&\mbox{\ \ \ $o_{i,j}=0$}\\
\end{cases}$ | | | |

###### Analysis 1

For a engagement prediction problem, FEEDE feedback encodes more information in the predictive model than CONFE. The information gain of FEEDE over CONFE is given by the conditional mutual information, $I(\tilde{x};\mathbf{O}|\mathbf{X})$, or equivalently $I(\tilde{x};\mathbf{Y}|\mathbf{X})$.

###### Proof 1

Engagement prediction by CONFE in conventional recommender systems is specified by the conditional probability in Equation 2. Engagement prediction by FEEDE in heterogeneous inference recommender systems is specified by the conditional probability in Equation 1.

By the definition of conditional mutual information, we have

$\displaystyle I(\tilde{x};\mathbf{O}|\mathbf{X})=H(\tilde{x}|\mathbf{X})-H(\tilde{x}|\mathbf{O},\mathbf{X})=$ | | | | |

$\displaystyle\sum_{X\in\mathcal{X}}p(X)\sum_{\tilde{x}\in\{0,1\}}\sum_{O\in\mathcal{O}}p(\tilde{x};O|X)\log\frac{p(\tilde{x};O|X)}{p(\tilde{x}|X)p(O|X)}$ | | | | |

where $H(\tilde{x}|\mathbf{X})$ is the conditional entropy representing the amount of uncertainty in $\tilde{x}$ given the observation of $\mathbf{X}$; $H(\tilde{x}|\mathbf{O},\mathbf{X})$ is the conditional entropy representing the amount of uncertainty in $\tilde{x}$ given the observation of $\mathbf{X}$ and $\mathbf{O}$; $\mathcal{X}$ and $\mathcal{O}$ are the alphabet (the set of all possible values) of $\mathbf{X}$ and $\mathbf{O}$, respectively.

Since $H(\tilde{x}|\mathbf{X})$ and $H(\tilde{x}|\mathbf{O},\mathbf{X})$ represent the entropy in the predictive conditional probability encoded by CONFE and FEEDE respectively, $I(\tilde{x};\mathbf{O}|\mathbf{X})=H(\tilde{x}|\mathbf{X})-H(\tilde{x}|\mathbf{O},\mathbf{X})$ gives the information difference between them. $I(\tilde{x};\mathbf{O}|\mathbf{X})>0$ when random variables $\tilde{x}$ and $O$ are not independent from each other given $X$, which is usually true in our feedback-based engagement prediction problem.

By $\mathbf{Y}=\mathbf{O}-\mathbf{X}$, given either $(\mathbf{O},\mathbf{X})$ or $(\mathbf{Y},\mathbf{X})$, the other can be precisely calculated. Thus, they contain equivalent information — $H(\tilde{x}|\mathbf{O},\mathbf{X})$ and $H(\tilde{x}|\mathbf{Y},\mathbf{X})$ have the same amount of uncertainty. Therefore, $I(\tilde{x};\mathbf{O}|\mathbf{X})=H(\tilde{x}|\mathbf{X})-H(\tilde{x}|\mathbf{O},\mathbf{X})=H(\tilde{x}|\mathbf{X})-H(\tilde{x}|\mathbf{Y},\mathbf{X})=I(\tilde{x};\mathbf{Y}|\mathbf{X})$.

### HI for divergent relevance: two goals, one model

HI combines p2p and n2p inference in one cohesive recommendation model. It is able to gain relevance and diversity collaboratively as inherent outcomes of one relevance inference process, i.e., divergent relevance (DR).

HI inference framework for divergent relevance. HI is built upon latent representations of entities (i.e., items and users), for example, via matrix factorization or embedding sub-network in a deep neural network. This way, the divergent relevance between two entities can be captured as the ”similarity” between their latent representations. The latent representations are optimized/trained, according to the data of (positive and/or negative) correlation between two entities in the corresponding application/context.

The divergent nature of n2p, i.e., relevance spreading over a wide variety of items in different topics, is illustrated in Figure 1 (b). Suppose that we have the same top-level latent topics and sub-topics as in the example from the previous section. Two items in $A_{1}$ and $A_{2}$, which received negative feedback from the user in the past, may suggest potential positive feedback to items in different top-level topics: $A$, $B$, and $C$ (mainly spread over the latter two). It is not hard to imagine that over time the recommendation based on n2p inference is not likely to converge to a narrowing topic. This is also reflected in the relatively flat distribution of n2p relevance over all items, illustrated by Figure 1 (d).

From the algorithm perspective, HI is an extension of CF approach. Because of its succinctness and generality, HI can work as a standalone recommender algorithm, or it can be integrated as an embedding module into a sophisticated recommendation model. Next, we demonstrate several exemplary algorithms, which implement HI for three different recommendation scenarios: candidate generation, ranking, and realtime personalized recommendation.

*Figure 2: Model architecture. Input vectors pass through embedding modules (${}^{+}f_{U}$, ${}^{+}f_{V}$ for p2p inference; ${}^{-}f_{U}$, ${}^{-}f_{V}$ for n2p inference; $\mathbf{W}$ for p2p-n2p interaction). Then, the latent representation goes through $g$ to output. First, embedding modules are trained and fixed. Then, $g$ is trained.*

### Candidate generation by HI

HI candidate generation is a combination of p2p and n2p similarity estimation. The two channels of inference (p2p and n2p) can be mixed by a preset or a dynamic candidate ratio, e.g., p2p vs. n2p = 70% vs. 30%. Next, we outline candidate generation by n2p in matrix factorization style.

Step 1. rank-$k$ matrix factorization of $\mathbf{Y}^{T}\mathbf{X}$:

$\displaystyle\mathbf{Y}^{T}\mathbf{X}\approx\mbox{\ }\mathbf{R}\mbox{\ }\mathbf{S},\mbox{\ where\ }\mathbf{R}\in\Re^{n\times k},\mathbf{S}\in\Re^{k\times n}$ | | | | (7) |

Step 2. smooth approximation of $\mathbf{Y}^{T}\mathbf{X}$:

$\displaystyle\mathbf{D}=\mbox{\ }\mathbf{R}\mbox{\ }\mathbf{S},\mbox{\ where\ }\mathbf{D}\in\Re^{n\times n}$ | | | | (8) |

Step 3. inferred divergent candidate items per user:

$\displaystyle{}^{*}\mathbf{Z}=\mbox{\ }\mathbf{Y}\mbox{\ }\mathbf{D},\mbox{\ where\ }^{*}\mathbf{Z}\in\Re^{m\times n}$ | | | | (9) |

The three steps for n2p candidate generation are similar to those of the p2p candidate generation, while the key difference lies in that the original item-item similarity matrix is obtained by multiplying the transpose of the user-item negative engagement matrix with the user-item positive engagement matrix. As a result, the smoothed n2p item similarity matrix, $\mathbf{D}$, implies the correlation between positively-engaged items and negatively-engaged items by the same users. Thus, we call this relevance inference negative-to-positive (n2p) inference.

### Ranking by HI

Similar to candidate generation, HI ranking also integrates convergent relevance by p2p and divergent relevance by n2p. This can be realized by integrating p2p and n2p as embedding sub-models into an overall relevance prediction model. Because of DNN’s strength in joint optimization of embedding and predicting sub-models, we choose it as the overall relevance prediction model to demonstrate HI ranking.

The DNN model architecture is illustrated in Figure 2. The upper left sub-network corresponds to the p2p inference channel. Specifically, the input of positive feedback user vector, $\mathbf{X}_{i,:}$, and item vector, $\mathbf{X}_{:,j}$, are passed through ${}^{+}f_{U}$ and ${}^{+}f_{V}$, and are mapped to their $k$-factor latent row vectors (in p2p context) ${}^{+}U_{i}$ and ${}^{+}V_{j}$, respectively. Here, ${}^{+}U_{i}$ and ${}^{+}V_{j}$ represent the latent factors of user $i$ and item $j$, respectively, when only positive feedback is used to infer the likelihood of positive feedback involving user $i$ and item $j$, respectively.

The lower left sub-network corresponds to the n2p inference channel. Row vectors ${}^{-}U_{i}$ and ${}^{-}V_{j}$ are the latent representations (in n2p context) of $\mathbf{Y}_{i,:}$ and $\mathbf{X}_{:,j}$, respectively. Here, ${}^{-}U_{i}$ and ${}^{-}V_{j}$ represent the latent factors of user $i$ and item $j$, respectively, when only negative feedback is used to infer the likelihood of positive feedback involving user $i$ and item $j$, respectively. ${}^{+}r_{i,j}$ and ${}^{-}r_{i,j}$ are the relevance scores derived from p2p channel and n2p channel, respectively.

Matrix $\mathbf{W}\in\Re^{k\times k}$ represents the interaction between p2p and n2p latent factors. The interaction indicates how the likelihood of positive engagement by user $i$ on item $j$ is impacted by the positive feedback factors jointly with the negative feedback factors. Specifically, an element $w_{p,q}$ in interaction matrix $\mathbf{W}$ for item $j$ indicates the likelihood of a user likes item $j$, given that the user does not like latent topic $p$ but likes latent topic $q$.

The advantage of the multi-module architecture is three-fold: (1) Separate p2p and n2p modules help balance relevance and diversity. It can also speed up the training by using pre-trained sub-models. (2) Available approximations of the embedding modules help cold start and responsiveness at prediction time. (3) Individual modules can be selected or deselected to suit different purposes/objectives.

Training has two consecutive phases: (1) Backward propagation is performed to optimize the sub-networks of ${}^{+}f_{U}$, ${}^{-}f_{U}$, ${}^{+}f_{V}$, and ${}^{-}f_{V}$. They are trained using $x_{i,j}$ (representing user $i$’s feedback to item $j$ when $o_{i,j}=1$) as the identical common target for ${}^{+}r_{i,j}$, ${}^{-}r_{i,j}$ and ${}^{*}r_{i,j}$, which are feedback estimated by positive feedback, negative feedback, and positive-joint-negative feedback, respectively. Once the sub-networks between the input and $g$ are trained in phase 1, they are fixed. (2) Backward propagation optimizes the sub-network of $g$ (a number of dense layers), where the ground truth for $\hat{p}(\tilde{x}_{i,j})$ is the value of $x_{i,j}$ (only if $o_{i,j}=1$).

Training loss of the entire embedding sub-network is

$\displaystyle\mathcal{L}=$ $\displaystyle(^{+}\mathcal{L})+\alpha(^{-}\mathcal{L})+\gamma(^{*}\mathcal{L})+\mbox{\ }$ | | | | | (10) |

$\displaystyle{}^{+}\lambda(\sum_{i}\|^{+}U_{i}\|_{2}^{2}+\sum_{j}\|^{+}V_{j}\|_{2}^{2})+$ | | | | |

$\displaystyle{}^{-}\lambda(\sum_{i}\|^{-}U_{i}\|_{2}^{2}+\sum_{j}\|^{-}V_{j}\|_{2}^{2})$ | | | | |

where

$\displaystyle{}^{+}\mathcal{L}=$ $\displaystyle\sum_{i,j}(x_{i,j}-(^{+}\mathbf{U}_{i})\mbox{\ }(^{+}\mathbf{V}_{j})^{T})^{2},\mbox{\ \ p2p loss}$ | | | | |

$\displaystyle{}^{-}\mathcal{L}=$ $\displaystyle\sum_{i,j}(x_{i,j}-(^{-}\mathbf{U}_{i})\mbox{\ }(^{-}\mathbf{V}_{j})^{T})^{2},\mbox{\ \ n2p loss}$ | | | | |

$\displaystyle{}^{*}\mathcal{L}=$ $\displaystyle\sum_{i,j}(x_{i,j}-(^{-}\mathbf{U}_{i})\mathbf{W}(^{+}\mathbf{U}_{i})^{T})^{2},\mbox{\ \ interaction loss}$ | | | | |

Prediction operates in two scenarios: (1) Warm start: the user and the item both have enough engagement data for an established profile. Normal forward pass is made through the whole network. (2) Cold start: the user does not have enough engagement data for an established profile. In order to obtain a reasonable prediction in near realtime, the latent vectors, (${}^{+}U_{i}$, ${}^{-}U_{i}$, and $\mathbf{W}$), are approximated with pre-trained models. The forward pass starts from them and goes through $g$ to the output. The following approximation is based on the matrix factorization in candidate generation.

${}^{+}U_{i}\approx\mbox{\ }$ $\displaystyle(\mathbf{XP})_{i,:}$ | | | | | (11) |

${}^{-}U_{i}\approx\mbox{\ }$ $\displaystyle(\mathbf{YR})_{i,:}$ | | | | | (12) |

$\displaystyle\mathbf{W}\approx\mbox{\ }$ $\displaystyle(\mathbf{S}_{:,j})(\mathbf{Q}_{:,j})^{T}$ | | | | | (13) |

where $\mathbf{P}$, $\mathbf{Q}$, $\mathbf{R}$, and $\mathbf{S}$ are results from the matrix factorization in Equations 3–9.

### Realtime recommendation by HI

A realtime recommender responds to users’ on-the-fly activities with new recommendations in realtime. A typical example of this application scenario is Next-Up video recommendations. Based on which items the user has interacted with in the immediately previous session, a realtime recommender comes up with new recommendations within milliseconds in reaction to the triggering events (e.g., clicks, long/short views, or likes). Being an extension of CF, HI algorithms’ succinctness and flexibility make them an ideal candidate for realtime personalized recommendation engine. An example algorithm is outlined as follows.

Step 0. input matrix concatenating p2p and n2p parts:

$\displaystyle\mathbf{H}=\begin{bmatrix}\mathbf{X}^{T}\mathbf{X}\\
\mathbf{Y}^{T}\mathbf{X}\end{bmatrix},\mbox{\ where\ }\mathbf{H}\in\Re^{2n\times n}$ | | | | (14) |

Step 1. rank-$k$ matrix factorization of $\mathbf{H}$:

$\displaystyle\mathbf{H}\approx\mbox{\ }\mathbf{A}\mbox{\ }\mathbf{B},\mbox{\ where\ }\mathbf{A}\in\Re^{2n\times k},\mathbf{B}\in\Re^{k\times n}$ | | | | (15) |

Step 2. smooth approximation of $\mathbf{H}$:

$\displaystyle\mathbf{H}^{\prime}=\mbox{\ }\mathbf{A}\mbox{\ }\mathbf{B},\mbox{\ where\ }\mathbf{H}^{\prime}\in\Re^{2n\times n}$ | | | | (16) |

Step 3. inferred p2p-&-n2p recommendations per user:

$\displaystyle{}^{*}\mathbf{G}_{i,:}=\mbox{\ }[\mathbf{X},\beta\mathbf{Y}]_{i,:}\mbox{\ }\mathbf{H}^{\prime},\mbox{\ where\ }^{*}\mathbf{G}\in\Re^{m\times n}$ | | | | (17) |

Step 0–2 correspond to offline model training. $\mathbf{H}$ contains both p2p and n2p inference components, therefore the MF on $\mathbf{H}$ is able to model p2p and n2p as well as the interaction between them. Step 3 performs online (realtime) inference for new recommendations for current user $i$.

## Experiments

| AUC-ROC, all users | AUC-ROC, D1–D7 users | mAP |

| CF-NN | CF-MF | CF-DA | HI-NN | CF-NN | CF-MF | CF-DA | HI-NN | CF-NN | CF-MF | HI-NN |

$0.616$ $0.589$ $0.577$ $0.557$ $0.550$ $0.525$ $0.501$ | | | | 0.765 | 0.561 | | | 0.734 | | | 0.662 |

*Table 1: Test-BT. (Left tab): AUC, all users. (Middle tab): AUC, D1–D7 users. (Right tab): mAP. *

This section evaluates and compares our approach, HI, with other state-of-the-art recommendation algorithms with and without diversity enhancement techniques on a well-known public datasets (MovieLens dataset) and a down-sampled real-world application production data.

### Evaluation metrics

Recommendation relevance (traditional recommendation quality) is measured by AUC ROC (area under the ROC curve), mAP (mean average precision) of the model against testing data, precision/recall, and empirical engagement metric in actual production environment (in Test-BT).

By a widely-adopted definition of recommendation diversity, the diversity between two items $i$ and $j$ is calculated as $Diversity_{(i,j)}=1-Similarity_{(i,j)}$ (Bradley and Smyth 2001; Castagnos, Brun, and Boyer 2013; L’Huillier, Castagnos, and Boyer 2014). $Similarity_{(i,j)}$ is obtained as the inferred inter-item similarity ${}^{*}x_{i,j}$ in Equation 5 from conventional CF. Then, the diversity of each user, i.e., the diversity of the top-$n$ items (by inferred ratings) recommended to the user, is computed as the average diversity over all pairs on the recommendation list for the user, i.e., $Diversity_{\mbox{user}}=\sum_{i,j\in\mbox{recommendations2user}}Diversity_{(i,j)}/(2n(n-1))$.

### Algorithms and tests

HI algorithms in the tests are as follows.

HI-RT: realtime personalized recommendation by HI’s MF implementation, as outlined in Equations 14–17.

HI-NN: candidate generation by HI’s candidate generation, as outlined in Equations 7–9; ranking by deep neural network, as outlined in Figure 2 and Equation 10. HI candidate generation assumes a mixture ratio of p2p:n2p = 67:33.

Conventional CF algorithms in the tests are listed below.

CF-RT: realtime personalized recommendation by p2p MF, as outlined in Equations 3–5.

CF-MF: candidate generation and ranking by p2p MF, as outlined in Equations 3–5.

CF-NN: candidate generation by p2p MF; ranking by deep neural network, as outlined in Figure 2 and Equation 10 with only the p2p-related components: ${}^{+}f_{U}$, ${}^{+}f_{V}$, ${}^{+}\mathcal{L}$.

CF-DA: candidate generation and ranking by p2p MF; followed by diversity-aware re-ranking, where the re-ranking score is calculated as a weighted sum of relevance score and diversity score: $Relevance+\phi Diversity$. This represents a large group of ranking score augmentation algorithm for diversity improvement (Bradley and Smyth 2001; Castagnos, Brun, and Boyer 2013; L’Huillier, Castagnos, and Boyer 2014).

CF-DM: candidate generation and ranking by p2p MF; followed by diversity-maximizing re-ranking, where the re-ranking selects $n$ final recommendations that maximize the overall diversity (from the top-$n^{\prime}$ items by p2p MF), where $n^{\prime}=5n$. This represents a group of diversity maximizing algorithms (Adomavicius and Kwon 2011; Paudel, F.Christoffel, and C.Newell 2016). In our experiments, $n=10$ for all algorithms, $n^{\prime}=50$ for CF-DM.

Our choice of MF implementation is Apache Spark ALS, for its proven effectiveness and multi-thread capability. The target rank of MF in all algorithms is set to $k=10$.

In the two tests, we compare algorithms in their pure CF form, without using any content-based features in the models. This is for keeping the comparison on a clean common ground and for keeping the focus on the essence of approaches. This leaves a large space for improvement beyond the models in the test, by enriching the model features and/or including algorithms beyond CF.

Test-RT: Use-case: realtime personalized recommendation; Data: MovieLens dataset. The whole dataset is divided into a training dataset and a test dataset (on a 80:20 training:test ratio), based on the timestamp of every event/example. Overall, MovieLens contains over $6000$ user and around $4000$ items.

Test-BT: Use-case: batch personalized recommendation, including candidate generation and ranking; Data: a down-sample of production data from the recommender on the Explore-like page of a real-world user-generated multimedia content sharing platform. The dataset is obtained by down-sampling production data during a couple of weeks’ time, which contains around one million unique users and one million unique video items. As in Test-RT, examples in the training set are strictly older than those in the test set in time. In this use case, the feedback from a user to an item indicates whether the user long-watches the video (watch more than a certain percentage of the video).

### Results and observations

Experiment results are organized under the two tests.

Test-RT. There are three algorithms in this test: CF-RT, CF-DM, and HI-RT. Overall, HI-RT outperforms both baseline algorithms: CF-RT and CF-DM simultanuously on all performance metrics: AUC ROC, precision (at a given recall value), recall (at a given precision value), diversity median, and diversity 25 percentile.

| | CF-RT | CF-DM | HI-RT |

$0.556$ $0.556$ | AUC-ROC | | | 0.669 |

$0.8$ $0.847$ $0.847$ | Precision (recall=) | | | 0.875 |

$0.85$ $0.78$ $0.78$ | Recall (precision=) | | | 0.91 |

$0.294$ $0.554$ | Diversity median | | | 0.751 |

$0.269$ $0.523$ | Diversity 25%ile | | | 0.723 |

*Table 2: Test-RT. Relevance and diversity results*

| | CF-NN | CF-DA | HI-NN |

$0.226$ $0.2352$ | Diversity median | | | 0.266 |

$0.1006$ $0.1078$ | Diversity 25%ile | | | 0.1243 |

*Table 3: Test-BT. Diversity results*

Table 2 summarizes the performance metrics from all three algorithms. HI has significant advantage over the baselines on diversity as well. CF-DM reduces the diversity advantage of HI-RT from $155\%$ down to $36\%$, at the cost of lower precision and recall in the final recommendations (which is not factored into our performance measurement, allowing an extra benefit for CF-DM baseline algorithm). HI has $20\%$ higher AUC, $16.7\%$ higher recall, and $3.3\%$ higher precision than the baselines. The large advantage on recall shows HI has higher coverage due to divergent relevance.

Detailed diversity distribution over users can be found in the left chart of Figure 3. The user-level recommendation diversity in CF-RT and CF-DM concentrates around $0.3$–$0.4$, with spikes around $1.0$, which correspond to users having little positive feedback in the past. Conventional CF without n2p inference cannot come up with meaningful recommendations for those users, i.e., recommendation for them are random selections, thus the high diversity.

Recommendation processing latency: HI-RT and CF-RT take $4ms$ on average; CF-DM takes around $7ms$ (due to its extra re-ranking processing), in a multi-thread program.

Test-BT. There are four algorithms in this test: CF-MF (representing the conventional CF by MF algorithms), CF-NN (conventional CF enhanced by deep neural network), CF-DA (conventional CF with diversity-aware re-ranking), and HI-NN (HI candidate generation and ranking).

Overall, HI-NN outperforms all baseline algorithms (CF-NN, CF-MF, and CF-DA) in both relevance and diversity, across model performance evaluation and online A/B tests. The AUC results are shown in the left and middle tabs of Table 1 for all users and (day 1–7) new users, respectively. Note that HI’s performance advantage further expands when the audience changes from all users to new users. This implies that HI is also helpful for user cold start. Similar improvement on mAP can be seen in the right tab of Table 1.

Table 3 shows diversity measurement of the algorithms. With the addition of diversity metric in the re-ranking step, CF-DA does improve recommendation diversity over other CF algorithms, but still has lower diversity than HI-NN, at a much lower AUC than HI-NN. The right chart in Figure 3 shows the user-level recommendation diversity distribution over users: CF-DA vs. HI-NN.

In a two-way A/B test on the real-world production, HI-NN achieved a $32.05\%$ lift over CF-NN on a production engagement performance metric, long-watch.

All algorithms run in a cluster of 15 nodes, each of which is equipped with 20 CPU cores and 250 GB memory. The model training runtimes and inference runtimes are similar across the algorithms in this test.

*Figure 3: User-level diversity distribution over all users.
(Left figure): Test-RT. (Right figure): Test-BT.*

## Conclusion

We presented heterogeneous inference (HI) to achieve divergent relevance, where diversity and relevance are two inherent outcomes of one negative-to-positive driven inference process. Meanwhile, HI’s generality and succinctness allows it to be applied to various recommendation scenarios/use-cases, including realtime personalized recommender.

## References

- Adomavicius and Kwon (2011) Adomavicius, G.; and Kwon, Y. 2011. Maximizing Aggregate Recommendation Diversity: A Graph-Theoretic Approach. In DiveRS.

- Adomavicius and Kwon (2012) Adomavicius, G.; and Kwon, Y. 2012. Improving Aggregate Recommendation Diversity Using Ranking-Based Techniques. In IEEE Transactions on Knowledge and Data Engineering, 896–911.

- Antikacioglu and Ravi (2017) Antikacioglu, A.; and Ravi, R. 2017. Post processing recommender systems for diversity. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 707–716.

- Bradley and Smyth (2001) Bradley, K.; and Smyth, B. 2001. Improving recommendation diversity. In Proceedings of the Twelfth Irish Conference on Artificial Intelligence and Cognitive Science, Maynooth, Ireland, 85–94. Citeseer.

- Castagnos, Brun, and Boyer (2013) Castagnos, S.; Brun, A.; and Boyer, A. 2013. When Diversity Is Needed… But Not Expected!

- Cheng et al. (2016) Cheng, H.-T.; Koc, L.; Harmsen, J.; Shaked, T.; Chandra, T.; Aradhye, H.; Anderson, G.; Corrado, G.; Chai, W.; Ispir, M.; et al. 2016. Wide & deep learning for recommender systems. In ACM Recsys, 7–10.

- Covington, Adams, and Sargin (2016) Covington, P.; Adams, J.; and Sargin, E. 2016. Deep Neural Networks for YouTube Recommendations. In ACM conference on recommender systems, 191–198.

- Davidson et al. (2010) Davidson, J.; Liebald, B.; Liu, J.; Nandy, P.; Vleet, T. V.; Gargi, U.; Gupta, S.; He, Y.; Lambert, M.; Livingston, B.; and Sampath, D. 2010. The YouTube Video Recommendation System. In ACM Recsys.

- den Oord, Dieleman, and Schrauwen (2013) den Oord, A. V.; Dieleman, S.; and Schrauwen, B. 2013. Deep content-based music recommendation. In NIPS, 2643–2651.

- Gauci et al. (2018) Gauci, J.; Conti, E.; Liang, Y.; Virochsiri, K.; He, Y.; Kaden, Z.; Narayanan, V.; Ye, X.; Chen, Z.; and Fujimoto, S. 2018. Horizon: Facebook’s open source applied reinforcement learning platform. arXiv preprint arXiv:1811.00260 .

- Ge et al. (2020) Ge, Y.; Zhao, S.; Zhou, H.; Pei, C.; Sun, F.; Ou, W.; and Zhang, Y. 2020. Understanding Echo Chambers in E-commerce Recommender Systems. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2261–2270.

- Gomez-Uribe and Hunt (2016) Gomez-Uribe, C. A.; and Hunt, N. 2016. The netflix recommender system: Algorithms, business value, and innovation. In TMIS.

- Hastie et al. (2015) Hastie, T.; Mazumder, R.; Lee, J.; and Zadeh, R. 2015. Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares. In JLMR.

- He et al. (2014) He, X.; Pan, J.; Jin, O.; Xu, T.; Liu, B.; Xu, T.; Shi, Y.; Atallah, A.; Herbrich, R.; Bowers, S.; and et al. 2014. Practical lessons from predicting clicks on ads at facebook. In International Workshop on Data Mining for Online Advertising, 1–9.

- Hurley and Zhang (2011) Hurley, N.; and Zhang, M. 2011. Novelty and Diversity in Top-N Recommendation – Analysis and Evaluation. In ACM Transactions on Internet Technology (TOIT), 14.

- Jiang et al. (2019) Jiang, R.; Chiappa, S.; Lattimore, T.; György, A.; and Kohli, P. 2019. Degenerate feedback loops in recommender systems. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, 383–390.

- Knijnenburg, Sivakumar, and Wilkinson (2016) Knijnenburg, B. P.; Sivakumar, S.; and Wilkinson, D. 2016. Recommender systems for self-actualization. In Proceedings of the 10th ACM Conference on Recommender Systems, 11–14.

- Koren (2008) Koren, Y. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In ACM SIGKDD international conference on knowledge discovery and data mining, 426–434.

- Koren, Bell, and Volinsky (2009) Koren, Y.; Bell, R.; and Volinsky, C. 2009. Matrix Factorization Techniques for Recommender Systems. In Computer, 30–37.

- Krichene et al. (2018) Krichene, W.; Mayoraz, N.; Rendle, S.; Zhang, L.; Yi, X.; Hong, L.; Chi, E.; and Anderson, J. 2018. Effcient training on very large corpora via gramian estimation. In arXiv.

- Lee and Seung (1999) Lee, D. D.; and Seung, H. S. 1999. Learning the parts of objects by non-negative matrix factorization. In Nature.

- L’Huillier, Castagnos, and Boyer (2014) L’Huillier, A.; Castagnos, S.; and Boyer, A. 2014. Understanding usages by modeling diversity over time.

- Mnih and Salakhutdinov (2008) Mnih, A.; and Salakhutdinov, R. R. 2008. Probabilistic matrix factorization. In NIPS, 1257–1264.

- Nguyen et al. (2014) Nguyen, T. T.; Hui, P.-M.; Harper, F. M.; Terveen, L.; and Konstan, J. A. 2014. Exploring the filter bubble: the effect of using recommender systems on content diversity. In Proceedings of the 23rd international conference on World wide web, 677–686.

- Noia et al. (2017) Noia, T. D.; Rosati, J.; Tomeo, P.; and Sciascio, E. D. 2017. Adaptive multi-attribute diversity for recommender systems. volume 382-383, 234 – 253.

- Okura et al. (2017) Okura, S.; Tagami, Y.; Ono, S.; and Tajima, A. 2017. Embedding-based News Recommendation for Millions of Users. In SIGKDD.

- Paudel, F.Christoffel, and C.Newell (2016) Paudel, B.; F.Christoffel; and C.Newell, A. 2016. Updatable, Accurate, Diverse, and Scalable Recommendations for Interactive Applications. In ACM Transactions on Interactive Intelligent Systems.

- Paudel, Haas, and Bernstein (2017) Paudel, B.; Haas, T.; and Bernstein, A. 2017. Fewer Flops at the Top: Accuracy, Diversity, and Regularization in Two-Class Collaborative Filtering. In RecSys, 215–223.

- Rendle (2010) Rendle, S. 2010. Factorization Machines. In ICDM: IEEE International Conference on Data Mining, 995–1000.

- Rendle, Freudenthaler, and Schmidt-Thieme (2010) Rendle, S.; Freudenthaler, C.; and Schmidt-Thieme, L. 2010. Factorizing personalized markov chains for next-basket recommendation. In WWW: the 19th international conference on world wide web, 811–820.

- Rendle and Schmidt-Thieme (2010) Rendle, S.; and Schmidt-Thieme, L. 2010. Pairwise interaction tensor factorization for personalized tag recommendation. In WSDM: the third ACM international conference on web search and data mining, 81–90.

- Salakhutdinov and Mnih (2008) Salakhutdinov, R.; and Mnih, A. 2008. Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. In ACM ICML, 880–887.

- Salakhutdinov, Mnih, and Hinton (2007) Salakhutdinov, R.; Mnih, A.; and Hinton, G. 2007. Restricted Boltzmann machines for collaborative filtering. In ICML, 791–798.

- Srebro, Rennie, and Jaakkola (2012) Srebro, N.; Rennie, J.; and Jaakkola, T. S. 2012. Maximum-margin matrix factorization. In NIPS, 1329–1336.

- Steck (2018) Steck, H. 2018. Calibrated recommendations. In Proceedings of the 12th ACM conference on recommender systems, 154–162.

- Vargas et al. (2014) Vargas, S.; Baltrunas, L.; Karatzoglou, A.; and Castells, P. 2014. Coverage, redundancy and size-awareness in genre diversity for recommender systems. In Proceedings of the 8th ACM Conference on Recommender systems, 209–216.

- Wang, Wang, and Yeung (2015) Wang, H.; Wang, N.; and Yeung, D.-Y. 2015. Collaborative deep learning for recommender systems. In SIGKDD, 1235–1244.

- Xie et al. (2020) Xie, R.; Ling, C.; Wang, Y.; Wang, R.; Xia, F.; and Lin, L. 2020. Deep Feedback Network for Recommendation. In Proceedings of IJCAI-PRICAI.

- Zhai et al. (2017) Zhai, A.; Kislyuk, D.; Jing, Y.; Feng, M.; Tzeng, E.; Donahue, J.; Du, Y. L.; and Darrell, T. 2017. Visual discovery at pinterest. In International Conference on World Wide Web Companion, 515–524.

- Zhang et al. (2019) Zhang, S.; Yao, L.; Sun, A.; and Tay, Y. 2019. Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR) 52(1): 1–38.

- Zhang, Zheng, and Zeng (2016) Zhang, Z.; Zheng, X.; and Zeng, D. 2016. A framework for diversifying recommendation lists by user interest expansion. In Knowledge-Based Systems.

- Zhao et al. (2018) Zhao, X.; Zhang, L.; Ding, Z.; Xia, L.; Tang, J.; and Yin, D. 2018. Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning. In ACM SIGKDD, 1040––1048.

- Zhao et al. (2015) Zhao, Z.; Cheng, Z.; Hong, L.; and Chi, E. H. 2015. Improving user topic interest profiles by behavior factorization. In International Conference on World Wide Web, 1406–1416.

- Zou et al. (2019) Zou, L.; Xia, L.; Ding, Z.; Song, J.; Liu, W.; and Yin, D. 2019. Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2810–2818.

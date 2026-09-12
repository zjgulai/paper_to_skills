<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2602.16299
     paper_id : 2602.16299
     source   : https://arxiv.org/html/2602.16299v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# MICE : Minimal Interaction Cross-Encoders for efficient Re-ranking

DOI: XXXXXXX.XXXXXXXConference: Make sure to enter the correct conference title from your rights confirmation email; June 03–05, 2018; Woodstock, NY
Mathias Vast Note: Both authors contributed equally to this research. email: mathias.vast@isir.upmc.fr Affiliation: Sinequa by ChapsVision, Paris, France Affiliation: Sorbonne Université, CNRS, ISIR, Paris, France , Victor Morand email: victor.morand@isir.upmc.fr Affiliation: Sorbonne Université, CNRS, ISIR, Paris, France , Basile Van Cooten email: bvancooten@chapsvision.com Affiliation: Sinequa by ChapsVision, Paris, France , Laure Soulier email: laure.soulier@isir.upmc.fr Affiliation: Sorbonne Université, CNRS, ISIR, Paris, France , Josiane Mothe email: josiane.mothe@irit.fr Affiliation: University of Toulouse, IRIT, Toulouse, France and Benjamin Piwowarski email: benjamin.piwowarski@cnrs.fr Affiliation: Sorbonne Université, CNRS, ISIR, Paris, France

2018

###### Abstract.

Cross-encoders deliver state-of-the-art ranking effectiveness in information retrieval, but have a high inference cost. This prevents them from being used as first-stage rankers, but also incurs a cost when re-ranking documents. Prior work has addressed this bottleneck from two largely separate directions: accelerating cross-encoder inference by sparsifying the attention process or improving first-stage retrieval effectiveness using more complex models, e.g. late-interaction ones. In this work, we propose to bridge these two approaches, based on an in-depth understanding of the internal mechanisms of cross-encoders. Starting from cross-encoders, we show that it is possible to derive a new late-interaction-like architecture by carefully removing detrimental or unnecessary interactions. We name this architecture MICE (Minimal Interaction Cross-Encoders). We extensively evaluate MICE across both in-domain (ID) and out-of-domain (OOD) datasets. MICE decreases fourfold the inference latency compared to standard cross-encoders, matching late-interaction models like ColBERT while retaining most of cross-encoder ID effectiveness and demonstrating superior generalization abilities in OOD.

https://github.com/xpmir/mice

## 1. Introduction

Since the advent of transformer-based models (Vaswani et al., 2017), a wide range of neural Information Retrieval (IR) architectures have been proposed, ranging from representation-based models (bi-encoders) to interaction-based models (cross-encoders), using dense or sparse, single or multi-vector representations. Although cross-encoders offer state-of-the-art ranking performance, they are computationally prohibitive when applied exhaustively to large corpora (Yates et al., 2021a; Nogueira and Cho, 2019). This bottleneck has maintained the prevalent retrieve-and-rerank paradigm, where a fast initial retriever (Karpukhin et al., 2020; Amati and Van Rijsbergen, 2002) filters the corpus down to a candidate pool that is then re-ranked by a cross-encoder like (Nogueira and Cho, 2019). This two-step approach has substantively improved the state of the art compared to pre-BERT models (Yates et al., 2021b).

However, the retrieve-and-rerank paradigm introduces fundamental limitations. First, re-ranking is bounded by first-stage retriever, i.e., if the relevant documents or passages have not been retrieved, the re-ranker has no opportunity to recover them. Second, the re-ranking step is affected by the cross-encoder efficiency.

The first limitation is addressed by architectures that improve effectiveness while remaining efficient enough for full-corpus search. A representative class of models are late interaction models, such as ColBERT (Khattab and Zaharia, 2020), that keep a token level representation of documents and queries (like cross-encoders), but delay (only contextualized representations) and simplify (max similarity) their interactions. However, late interaction models still do not reach the effectiveness levels of cross-encoders.

*Figure 1. MICE Architecture: stripping cross-encoders to keep the strict minimum interactions that maintain effectiveness.*

The second limitation is addressed by reducing the cross-encoder inference cost. This can be done either by optimizing or sparsifying the self-attention mechanism, the transformer architecture (Schlatt et al., 2024), or by reducing the number of candidates seen by the re-ranker through pruning/cascade strategies (Meng et al., 2024; Campagnano et al., 2025)). However, these approaches are still more expensive to run than late interaction models.

Our work connects these two lines of research by deriving an efficient, late-interaction-style ranker directly from a conventional cross-encoder. Specifically, we pursue two main goals. First, in Section 3, we develop a masking approach where we strip away some of the interaction mechanisms used in re-ranking to lower the inference cost of cross-encoders so that it falls below that of a late interaction model such as ColBERT. Specifically, we focus on interactions considered potentially superfluous by previous interpretability studies on internal cross-encoder mechanisms (Lu et al., 2025; Zhan et al., 2020) to minimize the impact on effectiveness. Second, we use these findings to adapt the cross-encoder architecture so that it retains only the necessary interactions, increasingly resembling a model suitable for use as a first-stage ranker. We call this new architecture MICE (Minimal-Interaction Cross-Encoder, described in Section 4 and depicted in Figure 1), and demonstrate across two distinct backbones, BERT (Devlin et al., 2019) and the more recent ModernBERT (Warner et al., 2024), that MICE does not degrade the ranking performance compared to the initial cross-encoder, while being $4\times$ more efficient. MICE strikes an ideal balance, preserving cross-encoder ranking quality while matching ColBERT’s efficiency.

In this work, we address the following research questions:

###### RQ 1

How much interactions can be removed to improve the efficiency of a cross-encoder, while maintaining its effectiveness?

###### RQ 2

Can we further improve the effectiveness by training cross-encoders without these superfluous interactions?

###### RQ 3

Can we design a late-interaction-like architecture, while maintaining cross-encoder effectiveness, paving the way for first-stage rankers?

## 2. Related Works

Backbone models for neural-based efficient IR

Cross-encoder models (Nogueira and Cho, 2019) are highly effective, but their poor efficiency has led IR practitioners to explore more efficient alternatives. One strategy is to avoid matching explicitly tokens within the model’s self-attention module and to encode separately queries and documents (Karpukhin et al., 2020). Bi-encoders index the document embeddings and compute relevance scores as dot-products; making them very efficient first-stage retrievers. But this ability comes with reduced effectiveness, especially in out-of-domain scenarios (Rosa et al., 2022; Thakur et al., 2021), as the model cannot explicitly model query-document interactions.

To improve the effectiveness of dense retrievers, new architectures emerged. Late-interaction models, such as ColBERT (Khattab and Zaharia, 2020), encode queries and passages into multiple vectors (one per token) and aggregate the score of each query token by computing the maximum similarity (MaxSim operator) with a document token. This results in greater effectiveness, compared to plain bi-encoders, but also induces an efficiency burden on the storage requirement of the system that the follow-up works were supposed to alleviate (Santhanam et al., 2022b; Santhanam et al., 2022a). Meanwhile, Learned Sparse Retrieval models, such as SPLADE (Formal et al., 2021), also improve the effectiveness of bi-encoders, especially in out-of-domain generalization, while retaining their efficiency during retrieval thanks to their sparse nature and inverted indexes.

Finally, Poly-encoders (Humeau et al., 2020) are hybrid architectures, more efficient than cross-encoders and more effective than bi-encoders. They summarize the documents into $M$ vectors, reducing the amount of computations done in self-attention. However, poly-encoders are highly sensitive to the choice of $M$ and only match the effectiveness of a cross-encoder for high $M$ values (360 in the original paper, longer than the average MS MARCO document). These works improve the efficiency of cross-encoders but propose less effective architectures. Another direction is to preserve their effectiveness while making them more efficient. As for all transformer-based models (Vaswani et al., 2017), this means focusing on their main computational bottleneck: the self-attention mechanism (Lin et al., 2017).

Targeting the self-attention mechanism for efficiency

Self-attention underpins transformer representations but slows inference. Many works, not only in IR (Tay et al., 2022), have attempted to reduce its complexity (Wang et al., 2020; Wu et al., 2021). Despite showing promising results in language modeling tasks, these approaches generally transfer poorly to specific tasks, especially in IR, where they are heavily involved in the detection of semantic and lexical matches between query and document tokens (Lu et al., 2025). Alternatively, Sparse Attention approaches, such as Longformer (Beltagy et al., 2020) or BigBird (Zaheer et al., 2020), improve self-attention efficiency by restricting token interactions to local windows. These methods posit that not all token interactions are required within self-attention to achieve the same level of effectiveness. They have been applied in IR (Sekulić et al., 2020), including to cross-encoders (Schlatt et al., 2024), substantially improving efficiency without hurting effectiveness.

Improving efficiency by reducing the size of the models

Beyond self-attention, many works explored improving deep neural network efficiency. They include knowledge distillation (Hinton et al., 2015), where a more effective but less efficient teacher model is used to train a smaller and more efficient student model, pruning (Frankle and Carbin, 2019; Campos et al., 2023), which involves removing unnecessary parts of the model to reduce computational burden, or compressing document representations to reduce overhead (Déjean and Clinchant, 2025; Santhanam et al., 2022a). These approaches have been applied to IR to improve LLMs efficiency (Lei et al., 2025; Schlatt et al., 2025). Finally, mid-fusion transformers (Cao et al., 2020; Tan and Bansal, 2019) also target models’ efficiency. This architecture encodes separately two modalities using the lower layers of a transformer model before using the upper layers on contextualized representations. In IR, MacAvaney et al. (2020) introduced PreTTR, a mid-fusion cross-encoder, and addressed the issue of storing document vectors by efficiently compressing them with an Auto-Encoder. While PreTTR manages to improve the efficiency of the base cross-encoder, its offline document encoding strategy assumes that the query length is known ahead of time. Our work alleviates this issue by 1) testing a ModernBERT backbone (Warner et al., 2024) that does not rely on positional embeddings, but instead uses the more effective rotary positional embedding (Su et al., 2024), and 2) separating the query and document encoder in mid-fusion on the BERT backbone (Devlin et al., 2019).

## 3. Towards minimal interaction cross-encoders

Before proposing a new cross-encoder architecture, it is necessary to study the importance of each type of interaction in cross-encoders. In this section, we study the impact of masking interactions between the different parts of a cross encoder’s input: [CLS], query $Q$, document $D$ and [SEP] tokens, to identify which are necessary or not. As our approach heavily relies on identifying which interactions can be removed safely and which are vital and must be kept, we first provide some details on cross-encoder and on the notation we use throughout the paper, before describing the current state of the understanding of how cross-encoders work internally.

### 3.1. Background

Cross-encoders classify a query-document $(Q,D)$ couple. Their input is typically composed of the following sequence of tokens: [CLS] $q_{1}\ldots q_{n}$ [SEP1] $d_{1}\ldots d_{m}$ [SEP2] ($n$ query tokens and $m$ document tokens). The transformer architecture updates the representation of the input tokens after each layer $\ell\in[1\ldots L]$ by performing a sequence of two main processing steps, namely residual updates through self-attention and feed-forward networks. Finally, the classification head of a cross-encoder is applied to the token representation [CLS] of the last layer $L$, which outputs a relevance score for the document w.r.t. the query.

Cross-encoders use self-attention to capture interaction signals between two tokens $a$ and $b$, by transferring part of the information contained in the representation of the token $a$ and encoding it inside the representation of the token $b$, and vice-versa. When these two tokens belong to the same part (query or document in IR), the self-attention moves context, or semantic information, between them and contextualizes them. This mechanism partly explains the success of transformer models in NLP tasks. In IR, these interactions have a very different meaning when token $a$ belongs to the query and token $b$ to the document (or reciprocally). As self-attention can move information from the input part of the token $a$ and compare it to the information already encoded in the token $b$ (e.g., the identity of the token, the semantic of its context, etc.), this mechanism is key in detecting matching signals between documents and queries.

For that reason, studies attempting to reverse-engineer the inner working of cross-encoders particularly focus on interpreting the interactions between the different input parts through the self-attention. For instance, Lu et al. show that it allows cross-encoders to detect, not only exact matching signals – what a lexical retriever like BM25 (Amati and Van Rijsbergen, 2002) would do – but also semantic matching signals. Zhan et al. show that the relevance prediction process inside cross-encoders is decomposed into multiple consecutive stages. In the first layers, the model contextualizes query and document tokens. At this stage, query-document interactions only play a minor role, but once their semantics have been properly encoded, the model starts using query-document interaction signals. Lu et al. (2025) further provide empirical evidence that matching signals are captured by the self-attention, in so-called matching heads, and then aggregated inside the query tokens by contextual query representation heads. Ultimately, relevance scoring heads scan query tokens to retrieve relevance information and to encode it in the [CLS] representation for the final prediction. Their findings suggest that information does not flow freely between input parts inside cross-encoders, but instead roughly flows from the document tokens towards the query tokens (after having been contextualized), and then from the query tokens towards the [CLS]. This hypothesis is further strengthened by Zhan et al. (2020), who suggested, without testing the hypothesis, that transfers of information from query tokens to document tokens could be removed with only a minor impact on performance. Zhan et al. (2020) also describe the crucial role of the [CLS] and [SEP] tokens as attention sinks, across most of the model’s layers, which are used when a query token should not match a document token (or vice-versa) – the attention is then concentrated on [CLS] or [SEP] tokens which do not impact the representation of the query (or document) token. Such attention sinks are well known in transformer architectures, and thought of as no-op operations (Clark et al., 2019).

This overview of the internal mechanisms of cross-encoders underlines the key role of self-attention. Yet, this is also the main reason of the cross-encoder inference cost. This motivates studying first which interactions can be removed safely from cross-encoders.

We denote input parts as $X,Y\in\{\texttt{[CLS]},Q,$ [SEP1], $D$, [SEP2]} and $Y\leftarrow X$ (resp. $Y\not\leftarrow X$) a transfer of information (resp. blocking the transfer) from a token in $X$ to a token in $Y$ through the self-attention mechanism (or equivalently that $Y$ attends to $X$).

### 3.2. Our approach

To assess the importance of a given interaction in predicting relevance, we mask the associated block in the cross-encoder self-attention weight matrices, i.e., we set the logits of this block to $-\infty$, before Softmax, so the sum of attention of scores equal to 1. We measure the importance of an interaction by the impact of masking that interaction on the model’s effectiveness. This is illustrated in Figure 2 where Masking Step 2 is used to prevent the transfer of information from the query $Q$ to the document $D$, i.e. $D\not\leftarrow Q$. Note that as depicted in Figure 2, the different masks are cumulative, for example Masking Step 2 also applies Masking Step 0 and Masking Step 1. Following the flow of information between the input parts considered in previous work, (Zhan et al., 2020; Lu et al., 2025), we consider four increasingly important interaction masking steps, all summarized in Figure 2.

###### Masking Step 0

Across all layers, we begin by blocking all transfers towards [SEP] ($\texttt{[SEP]}\not\leftarrow\{\texttt{[CLS]},Q,D\}$) and from the [CLS] towards other parts ($\{Q,\texttt{[SEP]},D\}\not\leftarrow\texttt{[CLS]}$). We also explicitly force the [SEP1] and [SEP2] tokens to be dedicated attention sink respectively for the query $Q$ and the document $D$, i.e., we allow $Q\leftarrow$[SEP1] and $D\leftarrow$[SEP2], as attention sinks are crucial for processing non desirable interactions Zhan et al. (2020) while blocking $Q\not\leftarrow$[SEP2] and $D\not\leftarrow$[SEP1]. For the same reasons, we further prevent the [CLS] from sending information to any other input part, except itself. We expect Masking Step 0 to have little to no impact on a cross-encoder effectiveness, as it basically reduces the amount of noise received by the query and document tokens.
Masking Step 0: $\texttt{[SEP]}\not\leftarrow\{\texttt{[CLS]},Q,D\}$, $\{Q,\texttt{[SEP]},D\}\not\leftarrow\texttt{[CLS]}$, $Q\not\leftarrow$[SEP2] and $D\not\leftarrow$[SEP1]

###### Masking Step 1

Across all layers, information from the document (and its attention sink to prevent information leakage) to the [CLS] ($\texttt{[CLS]}\not\leftarrow\{D$,[SEP2]$\}$) is blocked. This mask is motivated by the empirical evidence showing that query tokens are the recipients of the interactions with the document, thus storing relevance signals before passing them to the [CLS] (Lu et al., 2025). We also expect this step to have only a marginal impact on effectiveness.
Masking Step 1: Masking Step 0 and $\texttt{[CLS]}\not\leftarrow\{D$,[SEP2]$\}$

###### Masking Step 2

Across all layers, we mask the query to document flow ($D\not\leftarrow Q$). Studies showed that both query-document interactions are not as important to the overall process (Zhan et al., 2020; Schlatt et al., 2024). Removing $D\not\leftarrow Q$ across all layers is expected to produce only a limited drop in effectiveness, while being a first step towards totally separating query and document tokens contextualization, paving the way for more efficient cross-encoders where document token representations can be computed offline (MacAvaney et al., 2020).
Masking Step 2: Masking Step 1 and $D\not\leftarrow Q$

*Figure 2. Masking approach. Interactions between input parts ([CLS], $Q$, $D$, [SEP]) are blocked using cumulative masking. Colors indicate the step where masking begins, ending with Masking Step 3 in a complete $Q\not\leftrightarrow D$ separation (block-diagonal structure). Green blocks denote permanently preserved interactions and attention sinks.*

###### Masking Step 3

Across the first $\ell$ layers of the cross-encoder, we block the flow of information from the document to the query ($Q\not\leftarrow D$). This is rooted on the assumption that in the first few layers, the model focuses more on contextualizing the query and document tokens, independently, and interactions across these two parts are secondary (Zhan et al., 2020). Applying Masking Step 3 until layer $\ell$, in addition to Masking Step 2 (across all layers), allows to finally separate the query and document contextualization, an important milestone towards achieving more efficient cross-encoders. We define the optimal value $\ell^{*}$ as the highest that preserves the base model’s effectiveness.
Masking Step 3: Masking Step 2 and $Q\not\leftarrow D$ (up to layer $\ell^{*}$)

In this way, the set of possible interactions decreases after each step, until we obtain the minimal set required to maintain the original cross-encoder effectiveness. Note that we conduct this analysis on both on-the-shelf cross-encoders and on cross-encoders fine-tuned on the IR task with the masks. Figure 2 illustrates the application of these four consecutive steps on the weight matrix of any given self-attention module. From a higher standpoint, it further summarizes our goal: limit the set of interactions inside the cross-encoder to document and query tokens contextualization ($D\leftarrow D$ and $Q\leftarrow Q$), document to query interactions ($Q\leftarrow D$) in a limited number of layers, query to [CLS] transfers ($\texttt{[CLS]}\leftarrow\{Q,\texttt{[CLS]}{}\}$, as well as the attention from the query and document to their attention sinks ($Q\leftarrow$[SEP1] and $D\leftarrow$[SEP2]).

### 3.3. Experimental setup

##### Backbones

We consider two distinct backbones: BERT (Devlin et al., 2019), which has been extensively studied (Rogers et al., 2020; Ferrando et al., 2024) and has been successfully applied on the IR task, and ModernBERT (Warner et al., 2024), a recent update to the original BERT architecture with stronger capabilities (notably in long context processing). While our approach can be applied to analyze any cross-encoder, we focus here on cross-encoders based on small language models.Using here small models is further motivated by the need to run an extensive series of training runs and evaluations for our masking study that would otherwise be prohibitively expensive on larger backbones. In addition, despite the emergence of LLMs in IR (Ma et al., 2024), these smaller architectures remain highly relevant and competitive for ranking tasks due to their efficiency (Déjean et al., 2024). We study 3 cross-encoders based on the two backbones cited above: MiniLM-v2 (Wang et al., 2021), a compact yet effective model based on BERT (Devlin et al., 2019), and optimized via deep self-attention distillation, and two sizes of Ettin (Weller et al., 2025) encoders, differing in their number of parameters (17M and 32M), based on the ModernBERT architecture (Warner et al., 2024). We further detail their configuration in Table 1.

*Table 1. Configurations of the models used in this work.*

| MiniLM | Base | microsoft/MiniLM-L12-H384-uncased |

| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L12-v2 |

| Architecture | Layers=12, Params=33M, Hidden=384, Heads=12 |

| Ettin-17 | Base | jhu-clsp/ettin-encoder-17m |

| Cross-Encoder | tomaarsen/ms-marco-ettin-17m-reranker |

| Architecture | Layers=7, Params=17M, Hidden=256, Heads=4 |

| Ettin-32 | Base | jhu-clsp/ettin-encoder-32m |

| Cross-Encoder | tomaarsen/ms-marco-ettin-32m-reranker |

| Architecture | Layers=10, Params=32M, Hidden=384, Heads=6 |

##### Baselines

For this initial analysis, we first reproduce the off-the-shelf cross-encoders (based on MiniLM-v2 and Ettin models) that we use as baselines to control the effect of our interaction masking. Although we also directly compare with existing HuggingFace checkpoints ( in Table 1), we found empirically that their effectiveness were hard to match, mostly because their validation set includes BEIR datasets, while we consider BEIR to be OOD and keep it for evaluations only.

As baselines, we also consider Sparse CE (Schlatt et al., 2024), a cross-encoder whose attention is sparsified so that information transfers from the document to the query are blocked ($Q\not\leftarrow D$). This contradicts the statement discussed on the prevalence of that direction (compared to $D\leftarrow Q$) within the interactions between queries and documents for cross-encoders. However, it manages to preserve the original cross-encoder effectiveness when fine-tuned with this mask. We expect this baseline to confirm that designing our approach in light of the current understanding of these models can help the masked cross-encoder to better retain its original effectiveness. Note that we discarded the sliding-window attention over document tokens as this is orthogonal to our work (and detrimental for small windows).

Finally, we also compare with PreTTR (MacAvaney et al., 2020), which implements mid-fusion, by blocking query-document interactions in the early layers. Our approach, however, differs in later layers, where PreTTR allows all interactions. We view this model as an intermediary step between the original cross-encoder and MICE. We do not reproduce the entire PreTTR model (which includes an encoder-decoder to compress intermediate document token representations for their storage), but only the independent contextualization of queries and documents across the first $\ell$ layers (presented results are thus an upper bound for PreTTR).

##### Training.

To fine-tune our models and baselines for the IR task, we rely on distillation with the MarginMSE loss (Hofstätter et al., 2020). This loss yields superior retrieval performance compared to the standard Binary Cross Entropy (BCE) (Xu et al., 2025), by preserving the magnitude of the relevance difference (margin) between positive and negative pairs, rather than treating them as binary labels. As a teacher, we use the set of re-rankers scores on the MS MARCO passage ranking dataset (Bajaj et al., 2016) from Hofstätter et al., already used to train efficient re-ranking architectures such as ColBERT (Khattab and Zaharia, 2020).

To ensure consistency and strict comparability throughout our study, we use the same training hyperparameters in all experiments. All models are trained for 125,000 steps using a batch size of 32, a learning rate of $7\texttimes 10^{-6}$, and 5,000 warmup steps. We validate every 10,000 steps based on RR@10 performance on the MS MARCO development set and keep the best checkpoint.

##### Evaluation.

We evaluate our models both in-domain (ID), using MS MARCO (Bajaj et al., 2016) dev set (MSM) and the high-quality annotations of the TREC Deep Learning tracks from 2019 (DL19) and 2020 (DL20) (Craswell et al., 2020; Craswell et al., 2021); and out-of-domain (OOD), where we rely on the publicly available subset of datasets of the BEIR benchmark (Thakur et al., 2021). This includes the following list of 13 datasets: ArguAna (Ar), Climate-FEVER (CF), DBPedia (DB), FEVER (FE), FiQA (Fi), HotpotQA (HPQ), NFCorpus (NFC), Natural Questions (NQ), Quora Question Pairs (Q), SCIDOCS (SD), SciFact (SF), Touché-2020 (T-v2), and TREC-COVID (T-C).

### 3.4. Results and Analysis

#### 3.4.1. Impact of masking superfluous interactions (RQ 1)

To assess how masking superfluous interactions between input parts within the self-attention modules affects cross-encoder effectiveness, we report in Table 2 the results obtained by applying our masks to off-the-shelf cross-encoder models based on MiniLM-v2 and the two Ettin models.

*Table 2. Masking off-the-shelf cross-encoders with our approach (Section 3). Reranking is performed over 1000 documents retrieved by BM25.*

| | | In-domain | Average |

| | Re-Ranker | MSM | DL19 | DL20 | ID | BEIR |

| MiniLM | Cross-Encoder | 45.7 | 75.5 | 73.6 | 64.9 | 49.5 |

| + Masking Step 0 | 44.4 | 74.3 | 72.0 | 63.5 | 48.2 |

| + Masking Step 1 | 44.4 | 73.4 | 72.0 | 63.2 | 48.0 |

| + Masking Step 2 | 27.8 | 61.5 | 59.8 | 49.7 | 36.6 |

| Ettin-17M | Cross-Encoder | 39.0 | 68.4 | 67.0 | 58.1 | 44.6 |

| + Masking Step 0 | 3.1 | 11.1 | 9.7 | 8.0 | 3.2 |

| + Masking Step 1 | 5.2 | 14.9 | 15.2 | 11.8 | 4.6 |

| + Masking Step 2 | 4.5 | 13.9 | 13.7 | 10.7 | 4.0 |

| Ettin-32M | Cross-Encoder | 43.7 | 70.8 | 71.3 | 61.9 | 48.1 |

| + Masking Step 0 | 12.0 | 29.0 | 29.3 | 23.4 | 7.5 |

| + Masking Step 1 | 27.0 | 56.4 | 52.9 | 45.4 | 26.8 |

| + Masking Step 2 | 21.3 | 47.8 | 47.8 | 39.0 | 24.0 |

We observe that the MiniLM-based cross-encoder and the two Ettin-based cross-encoders respond very differently to the different masking strategies. Although the base effectiveness of the MiniLM model remains stable for masks 0 and 1 (around 1 nDCG@10 point in average on both ID and OOD), these masks impact severely cross-encoders based on Ettin (more than -10 nDCG@10 for Masking Step 0). A possible explanation is that our masking steps are derived from Lu et al. (2025), who study the MiniLM-v2 cross-encoder model only; they may not fit the internal mechanisms of Ettin cross-encoders. For instance on Ettin, while Masking Step 0 and Masking Step 1 are intended to target interactions that should have only a marginal effect on model performance (by limiting information flow toward attention sinks, as observed with MiniLM), it is plausible that, because they are based on ModernBERT and pre-trained with mechanisms such as sliding-window attention (unlike BERT-base models), their attention sinks are different from those of BERT-based models. As a result, and given the effect of our masking strategy that redistributes the attention probability that was concentrated on the sink, our masks may be less appropriate for Ettin than for MiniLM. We also observe that Masking Step 1 increases performance over Masking Step 0 for Ettin cross-encoders, but that their performance remains much lower than with the unmasked model. Finally, while Masking Step 0 and Masking Step 1 only slightly impact MiniLM cross-encoder, our results indicate that further masking (Masking Step 2) leads to a more substantial decrease of its effectiveness (-10 on nDCG@10 for both ID and OOD).

Together, these results indicate that it is possible to remove some interactions inside the self-attention modules of a cross-encoder, without impacting its effectiveness (see Masking Step 0 and Masking Step 1 for MiniLM). However, doing so requires a good understanding of the model internal mechanisms as acknowledged by the results with Ettin. These insights, in addition to the substantial drop induced by Masking Step 2, suggest that it is possible to maintain a cross-encoder performance while removing unnecessary interactions (RQ 1) only up to a certain point. Removing these interactions on an already fine-tuned model seems to harm its effectiveness, showing that the fine-tuned models still have learned to use some of the information transfers we mask to predict relevance. Consequently, empirical evidence from Sparse CE shows that fine-tuning can preserve the base model’s effectiveness even when a key information transfer is removed. This naturally motivates fine-tuning the re-ranker with our masks applied (see RQ 2), what we study in the next section.

*Table 3. Re-ranking evaluation results over 1k docs/query from BM25 in nDCG@10 (over 5 seeds) for the masking experiment (Section 3), comparing models with and w/o masking and fine-tuning (versus - off-the-shelf models). $\uparrow$ and $\downarrow$ marks a statistically significant difference between a fine-tuned model, with a mask, and its fine-tuned counterpart, w/o mask. Bold values marks the best averaged value per backbone, “_” indicates second best.*

| | | In-domain | BEIR (OOD) | Average |

| | Re-Ranker | MSM | DL19 | DL20 | Ar | CF | DB | FE | Fi | HPQ | NFC | NQ | Q | SD | SF | T-v2 | T-C | ID | OOD |

| BM25 | 23.0 | 51.2 | 47.7 | 30.0 | 16.5 | 31.8 | 65.1 | 23.6 | 63.3 | 32.2 | 30.6 | 78.9 | 14.0 | 67.9 | 45.4 | 59.5 | 42.5 | 43.0 |

| MiniLM | Sparse CE (Schlatt et al., 2024) | 44.1 | 74.3 | 70.7 | 6.1 | 20.4 | 44.5 | 80.0 | 34.9 | 72.0 | 29.2 | 53.8 | 74.6 | 14.5 | 61.5 | 31.0 | 60.8 | 63.0 | 44.9 |

$\ell 4$ | PreTTR (MacAvaney et al., 2020) () | 44.5 | 73.8 | 71.9 | 16.8 | 27.5 | 45.0 | 83.5 | 37.4 | 72.2 | 33.1 | 56.1 | 81.0 | 15.3 | 68.8 | 35.8 | 67.4 | 63.4 | 49.2 |

| Cross-Encoder | 45.7 | 75.5 | 73.6 | 20.6 | 25.8 | 46.6 | 82.7 | 36.6 | 74.1 | 32.5 | 56.3 | 82.0 | 15.3 | 68.5 | 35.2 | 66.8 | 64.9 | 49.5 |

| Baseline | 44.7 | 73.8 | 72.9 | 14.8 | 15.7 | 46.0 | 73.4 | 34.0 | 72.4 | 27.1 | 56.6 | 80.4 | 12.5 | 55.3 | 29.9 | 63.4 | 63.8 | 44.7 |

$\uparrow$ $\uparrow$ $\downarrow$ | Masking Step 0 | 44.6 | 73.8 | 73.3 | 14.6 | 19.0 | 45.6 | 80.6 | 35.7 | 72.3 | 28.2 | 56.5 | 78.6 | 13.6 | 58.3 | 34.5 | 64.5 | 63.9 | 46.3 |

$\uparrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\downarrow$ $\uparrow$ $\uparrow$ | Masking Step 1 | 44.8 | 73.8 | 73.3 | 25.4 | 26.2 | 45.9 | 82.7 | 37.4 | 73.8 | 33.7 | 56.4 | 79.5 | 15.4 | 68.8 | 38.0 | 68.9 | 64.0 | 50.2 |

$\uparrow$ $\uparrow$ $\uparrow$ $\downarrow$ $\uparrow$ $\uparrow$ | Masking Step 2 | 44.1 | 73.1 | 71.8 | 14.8 | 26.8 | 45.4 | 81.4 | 37.4 | 72.5 | 34.5 | 55.2 | 77.5 | 15.6 | 69.4 | 37.5 | 68.8 | 63.0 | 49.0 |

$\ell$ $\downarrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\downarrow$ $\downarrow$ $\uparrow$ $\uparrow$ | Masking Step 3-4 | 43.9 | 73.0 | 70.9 | 10.4 | 26.0 | 44.2 | 80.4 | 36.3 | 72.3 | 33.4 | 54.3 | 71.1 | 14.9 | 68.2 | 35.5 | 66.5 | 62.6 | 47.2 |

| Ettin-32M | Cross-Encoder | 43.7 | 70.8 | 71.3 | 10.0 | 23.5 | 42.5 | 81.8 | 36.1 | 70.9 | 33.8 | 52.7 | 79.8 | 14.9 | 70.2 | 38.1 | 71.6 | 61.9 | 48.1 |

| Baseline | 43.1 | 70.3 | 69.7 | 11.9 | 24.0 | 41.3 | 82.8 | 36.6 | 70.0 | 33.8 | 52.4 | 79.6 | 14.6 | 68.8 | 41.1 | 71.8 | 61.0 | 48.3 |

$\uparrow$ $\downarrow$ | Masking Step 0 | 42.8 | 70.0 | 69.7 | 14.2 | 23.9 | 40.4 | 82.6 | 36.7 | 69.8 | 34.0 | 51.9 | 71.4 | 14.4 | 70.0 | 40.2 | 71.3 | 60.8 | 47.8 |

$\uparrow$ $\downarrow$ $\downarrow$ | Masking Step 1 | 42.6 | 70.8 | 69.2 | 18.2 | 25.8 | 40.6 | 81.8 | 36.9 | 70.6 | 34.1 | 51.9 | 77.3 | 14.5 | 70.3 | 39.7 | 71.5 | 60.9 | 48.7 |

$\uparrow$ $\downarrow$ $\downarrow$ $\downarrow$ | Masking Step 2 | 42.3 | 69.6 | 68.5 | 17.0 | 24.0 | 40.4 | 79.0 | 36.4 | 69.2 | 34.1 | 50.8 | 60.0 | 14.2 | 71.4 | 39.7 | 70.8 | 60.1 | 46.7 |

$\ell$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ | Masking Step 3-6 | 41.9 | 69.4 | 68.6 | 9.2 | 22.9 | 37.8 | 78.3 | 34.4 | 65.0 | 32.8 | 49.8 | 37.8 | 12.4 | 66.8 | 40.8 | 68.7 | 59.9 | 42.8 |

| Ettin-17M | Cross-Encoder | 39.0 | 68.4 | 67.0 | 18.3 | 21.9 | 36.3 | 74.5 | 29.9 | 64.8 | 30.7 | 46.2 | 78.6 | 12.5 | 67.8 | 31.6 | 66.7 | 58.1 | 44.6 |

| Baseline | 38.5 | 65.1 | 63.5 | 12.8 | 21.0 | 33.4 | 77.2 | 31.0 | 60.0 | 31.4 | 45.7 | 76.7 | 12.2 | 66.0 | 36.9 | 66.6 | 55.7 | 43.9 |

$\downarrow$ $\uparrow$ $\downarrow$ | Masking Step 0 | 38.6 | 65.2 | 63.1 | 9.0 | 21.5 | 33.1 | 76.5 | 30.5 | 61.6 | 31.5 | 45.9 | 71.9 | 12.0 | 65.6 | 37.7 | 65.2 | 55.7 | 43.2 |

$\downarrow$ $\uparrow$ $\uparrow$ | Masking Step 1 | 38.2 | 65.5 | 62.9 | 13.9 | 22.7 | 33.7 | 75.7 | 30.9 | 64.1 | 31.4 | 45.4 | 79.5 | 12.5 | 65.8 | 39.1 | 64.4 | 55.5 | 44.6 |

$\uparrow$ $\downarrow$ $\uparrow$ $\downarrow$ | Masking Step 2 | 38.1 | 65.3 | 62.2 | 15.6 | 22.2 | 34.1 | 75.6 | 30.0 | 63.7 | 31.4 | 44.7 | 72.1 | 12.1 | 64.7 | 39.7 | 64.3 | 55.2 | 43.9 |

$\ell 3$ $\uparrow$ $\downarrow$ $\uparrow$ $\downarrow$ | Masking Step 3- | 37.8 | 66.1 | 62.4 | 14.0 | 22.9 | 33.1 | 73.8 | 29.3 | 62.1 | 30.6 | 44.3 | 71.4 | 11.9 | 62.9 | 41.2 | 64.9 | 55.4 | 43.3 |

#### 3.4.2. Impact of masking when training cross-encoders (RQ 2)

Given the limits of directly applying our masks on off-the-shelf cross-encoders, we now fine-tune pretrained models, with the masks, on the re-ranking task. We use the training setup described in subsection 3.3 and learn a new cross-encoder for each mask, as well as the baseline described in subsection 3.3. Table 3 reports the average nDCG@10 (ID and OOD) on 5 random seeds.

First, our reproduced MiniLM cross-encoder do not exacly the off-the-shelf checkpoint, particularly on OOD (44.7 vs. 49.5 avg. nDCG@10), a gap expected given the original model’s use of BEIR data for validation, and that we report means over several seeds. Conversely, our Ettin-based reproductions fully match the performance of their HuggingFace counterparts.

Secondly, we note that fine-tuning these checkpoints with Masking Step 0, either matches the performance of the unmasked baseline (across the two Ettin backbones) or exceeds them (for MiniLM, especially in OOD with 46.3 vs 44.7 nDCG@10 points). Compared to the severe impact of Masking Step 0 on the off-the-shelf cross-encoders, this underlines the importance of fine-tuning these models while masking and confirms that the role of the [SEP] tokens is not directly tied to the relevance prediction process.

Across all backbones, Table 3 shows that further masking up to Masking Step 2 either exceeds or matches the performances of the fine-tuned baselines, both ID and OOD. We observe a gain of 5.5 nDCG@10 points in average for Masking Step 1 on MiniLM, across OOD datasets over the fine-tuned baseline. With this masking strategy, MiniLM reaches the best performances, both for ID and OOD, even surpassing the original cross-encoder HuggingFace checkpoint on BEIR. We can draw to conclusions from that. First, this confirms that [CLS] does not need to attend to the document to receive the appropriate relevance signals (Masking Step 1). On the contrary, OOD results show that cross-encoders capture spurious correlations when allowing this transfer of information. Second, $D\not\leftarrow Q$ (Masking Step 2) across all layers does not harm the effectiveness of cross-encoders, while this ablation can potentially lead to substantial gain in efficiency. This contradicts the masking strategy of Sparse CE (Schlatt et al., 2024) ($Q\not\leftarrow D$) as we observe no improvement on the effectiveness compared to the unmasked MiniLM cross-encoder (63.8 vs 63.0 on average in ID, and 44.7 vs 44.9 on average in OOD). This suggests that our decision is key in not only maintaining, but also increasing, the cross-encoder effectiveness, while improving its efficiency.

*(a) MiniLM-L12-v2*

*(b) Ettin-17M*

*(c) Ettin-32M*

*Figure 3. In-domain nDCG@10 when masking all interactions between $Q$ and $D$ up to a given layer in the transformer (Masking Step 3). *

On the impact of Masking Step 3 and the optimal layer $\ell^{*}$ before which query and document tokens can be processed independently, i.e., without interaction ($Q\not\leftarrow D$ and $D\not\leftarrow Q$), Figure 3 shows the nDCG@10 averaged on ID collections as a function of the layer $\ell$. For MiniLM, we observe that it is possible to contextualize the query and document tokens independently, without harming the ID effectiveness, up to the layer $\ell=4$ (out of 12 total layers). There are two potential explanations for the ID effectiveness drop after $\ell\geq 5$: (1) the remaining interaction layers are not enough to properly detect all the relevance signals; (2) after layer $\ell=4$, further contextualizing query and document tokens independently introduces signals in their representation that are detrimental to the IR task. We partially address this question in section 4, when designing MICE’s architecture. For Ettin models, the optimal $\ell^{*}$ are 3 (out of 7) for Ettin-17M and 6 (out of 10) for Ettin-32M. We refer to the layer $\ell^{*}$ as the first interaction layer of each model.

With the optimal layers selected on ID performance, we report in Table 3 the detailed impacts for Masking Step 3. We observe a consistent drop in OOD performance compared to using only Masking Step 2, but its scale deviates a lot depending on the model. In particular, MiniLM loses around 1.8 nDCG@10 points in average between Masking Step 3 and Masking Step 2, but its OOD effectiveness remains 3 points above its unmasked baseline. Meanwhile, Ettin-17M loses only 0.6 point in average, but drops slightly below its unmasked baseline (matching Masking Step 2). Finally, the most important drop (almost -4 for nDCG@10) is observed for Ettin-32M. At the same time, we observe very little variation in ID effectiveness between the different masking steps, and compared to the baseline (see the evolution, per backbone, of the average in the ID column of Table 3). This suggests that our masks do not prevent the models from learning correctly the IR task, but, depending on the base model, may hinder their OOD robustness. For additional context, we report the results of applying a mid-fusion-like separation, up to the first interaction layer $\ell^{*}=4$ (as we found it suitable for MiniLM cross-encoder in Figure 3), to reproduce PreTTR (MacAvaney et al., 2020) on top of MiniLM. We observe similar levels of performance between this PreTTR reproduction and Masking Step 2 (still below Masking Step 1) and a difference of respectively 1 and 2 nDCG@10 points in average over ID and OOD compared to Masking Step 3. This lower gap can be explained by the richer interactions allowed in PreTTR across its interaction layers, yet, as we also expect substantial gains in efficiency thanks to our other masks, we believe that our trade-off between efficiency and effectiveness will nonetheless exceed the one achieved by PreTTR.

This second set of analysis demonstrates that masking while fine-tuning the cross-encoders not only preserved the baseline effectiveness but leads to substantial gains (RQ 2), especially in OOD generalization. This observation stays valid across the three cross-encoders we study, especially for Ettin cross-scorers who both suffered dramatic performance drops after Masking Step 0 in the absence of fine-tuning (see Table 2).

### 3.5. Intermediate Conclusions

Our experiments with Masking Step 1 revealed that masking direct $\texttt{[CLS]}\not\leftarrow D$ interactions consistently improves both ID and OOD effectiveness. Furthermore, Masking Step 2 confirmed that preventing information transfer from the query to the document $D\not\leftarrow Q$ does not harm the model, as long a the information can flow from the document to the query ($Q\leftarrow D$). Finally, with Masking Step 3, we corroborate previous studies on mid-fusion architectures (MacAvaney et al., 2020; Cao et al., 2020), showing that queries and documents can be contextualized independently in the lower layers, without compromising the performance of the full cross-encoder.

This section demonstrates that masking targeted interactions in a cross-encoder can actively enhance its re-ranking performance. This gain is particularly pronounced in OOD scenarios, suggesting that masking these interactions acts as a form of regularization, preventing the model from overfitting the training corpus. When cross-encoders are fine-tuned on the IR task while being masked (see answer to RQ 2), this observation is valid with both MiniLM and Ettin backbones and opens the way to more efficient and as effective cross-encoders.

## 4. MICE - Minimal Interation Cross Encoder

Building on insights from Section 3, we propose a novel, streamlined ranker architecture we call MICE (Minimal Interaction Cross-Encoder). We design MICE as a late-interaction-like architecture, thus more efficient than traditional cross-encoders, and study its effectiveness to answer RQ 3. By discarding superfluous interactions, we show that MICE becomes significantly lighter than standard cross-encoders while maintaining competitive re-ranking performance.

### 4.1. Architecture

Compared to a standard cross-encoder, MICE differs in three principal architectural choices: (1) Mid-fusion first encodes the query $Q$ and document $D$ independently; (2) Light Cross-Attention only transfers information from a frozen document representation to the query; (3) Layer Dropping reduces the number of interaction layers. We detail each of these architectural choices in the following paragraphs. An overview of MICE is depicted in Figure 1.

*Table 4. Re-ranking evaluation results over 1k docs/query from BM25 in nDCG@10 (over 5 seeds) for the experiments on MICE. Mask-X models are fine-tuned with Masking Step 3 (see Section 3). “MICE-$\ell$X+[Y/all]” models indicates using only Y interaction layers (or all otherwise) starting from layer X. $\uparrow$ and $\downarrow$ marks a statistically significant difference between Mask or MICE models and their cross-encoder counterparts. Bold values marks the best averaged value per backbone, “_” indicates second best. Gray values indicate BM25 either matches or exceeds the baseline (corresp. average is reported in BM25-hard column).*

| | | In-domain (ID) | BEIR (OOD) | Average |

| Re-ranker | MSM | DL19 | DL20 | Ar | CF | DB | FE | Fi | HPQ | NFC | NQ | Q | SD | SF | T-v2 | T-C | ID | OOD | BM25-hard |

| | BM25 | 23.0 | 51.2 | 47.7 | 30.0 | 16.5 | 31.8 | 65.1 | 23.6 | 63.3 | 32.2 | 30.6 | 78.9 | 14.0 | 67.9 | 45.4 | 59.5 | 42.5 | 43.0 | - |

| ColBERTv2 (Santhanam et al., 2022b) | 45.0 | 74.6 | 73.4 | 33.8 | 17.0 | 44.3 | 73.0 | 34.2 | 66.2 | 33.4 | 53.9 | 86.3 | 14.1 | 63.8 | 34.2 | 66.7 | 64.3 | 47.8 | - |

| MiniLM | ColBERTv2 | 37.3 | 67.8 | 66.3 | 31.4 | 17.3 | 32.1 | 71.5 | 26.0 | 55.4 | 29.9 | 44.9 | 83.2 | 11.3 | 60.8 | 27.3 | 62.6 | 57.2 | 42.6 | 29.7 |

| Baseline | 44.7 | 73.8 | 72.9 | 14.8 | 15.7 | 46.0 | 73.4 | 34.0 | 72.4 | 27.1 | 56.6 | 80.4 | 12.5 | 55.3 | 29.9 | 63.4 | 63.8 | 44.7 | 25.9 |

$\ell$ $\downarrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\downarrow$ $\downarrow$ $\uparrow$ $\uparrow$ | Masking Step 3-4 | 43.9 | 73.0 | 70.9 | 10.4 | 26.0 | 44.2 | 80.4 | 36.3 | 72.3 | 33.4 | 54.3 | 71.1 | 14.9 | 68.2 | 35.5 | 66.5 | 62.6 | 47.2 | 31.4 |

$\ell$ $\downarrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\downarrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\uparrow$ | MICE-4+all | 43.1 | 73.4 | 70.1 | 39.0 | 25.1 | 44.4 | 79.1 | 35.6 | 72.0 | 34.5 | 52.2 | 81.4 | 14.8 | 70.5 | 39.4 | 69.2 | 62.2 | 50.5 | 37.2 |

$\ell$ $\downarrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\downarrow$ $\uparrow$ $\downarrow$ $\uparrow$ $\uparrow$ $\uparrow$ $\uparrow$ | MICE-4+3 | 42.1 | 72.6 | 69.3 | 41.0 | 24.4 | 43.7 | 79.8 | 34.7 | 71.3 | 34.3 | 51.4 | 81.4 | 14.9 | 69.2 | 39.5 | 69.4 | 61.3 | 50.4 | 37.2 |

| Ettin-32M | Baseline | 42.8 | 70.5 | 69.3 | 12.0 | 23.4 | 41.1 | 82.3 | 36.4 | 69.5 | 33.7 | 52.1 | 79.0 | 14.5 | 68.8 | 41.2 | 72.0 | 60.9 | 48.2 | 36.7 |

$\ell$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ | Masking Step 3-6 | 41.9 | 69.4 | 68.6 | 9.2 | 22.9 | 37.8 | 78.3 | 34.4 | 65.0 | 32.8 | 49.8 | 37.8 | 12.4 | 66.8 | 40.8 | 68.7 | 59.9 | 42.8 | 25.1 |

$\ell$ $\uparrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\uparrow$ $\downarrow$ | MICE-6+all | 42.4 | 70.3 | 69.5 | 14.8 | 21.6 | 39.7 | 71.6 | 26.2 | 68.0 | 32.9 | 49.1 | 82.4 | 11.9 | 68.4 | 41.9 | 72.9 | 60.7 | 46.3 | 37.8 |

$\ell$ $\downarrow$ $\uparrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\uparrow$ $\downarrow$ | MICE-6+3 | 41.9 | 69.3 | 68.4 | 18.8 | 21.6 | 38.9 | 71.5 | 27.3 | 67.3 | 32.8 | 48.5 | 82.0 | 11.9 | 68.9 | 41.9 | 71.5 | 59.8 | 46.4 | 38.7 |

| Ettin-17M | Baseline | 38.6 | 64.9 | 63.6 | 12.9 | 21.0 | 33.6 | 77.5 | 31.0 | 60.5 | 31.5 | 45.7 | 76.7 | 12.2 | 66.1 | 36.4 | 66.7 | 55.7 | 44.0 | 42.3 |

$\ell$ $\uparrow$ $\downarrow$ $\uparrow$ $\downarrow$ | Masking Step 3-3 | 37.8 | 66.1 | 62.4 | 14.0 | 22.9 | 33.1 | 73.8 | 29.3 | 62.1 | 30.6 | 44.3 | 71.4 | 11.9 | 62.9 | 41.2 | 64.9 | 55.4 | 43.3 | 42.0 |

$\ell$ $\uparrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\uparrow$ $\downarrow$ $\uparrow$ $\downarrow$ | MICE-3+all | 37.7 | 66.1 | 62.2 | 20.1 | 19.1 | 33.3 | 68.0 | 24.4 | 62.0 | 29.9 | 42.8 | 81.3 | 9.6 | 67.0 | 40.2 | 66.6 | 55.3 | 43.4 | 44.3 |

$\ell$ $\uparrow$ $\downarrow$ $\downarrow$ $\downarrow$ $\uparrow$ $\downarrow$ $\uparrow$ $\downarrow$ | MICE-3+3 | 38.0 | 65.7 | 62.4 | 20.2 | 19.1 | 33.9 | 67.5 | 24.5 | 63.1 | 29.9 | 43.1 | 80.8 | 9.2 | 67.3 | 44.2 | 67.3 | 55.4 | 43.9 | 44.9 |

A crucial design choice in MICE is the independent document and query contextualization. We rely on the original $\ell^{*}$ first layer (Section 3.5) found using the Masking Step 3. We use a cleaner separation than what is used in PreTTR (MacAvaney et al., 2020), where these layers were based on placeholders when encoding the document.

Based on the findings on the effect of mask 2, which only allows $Q\leftarrow D$, we go further by only computing cross-attention from frozen document representations to the query. This significantly reduces computational cost by eliminating expensive self-attention over the numerous document tokens. In doing so, we hypothesize that hidden state of the document does not require further contextualization during the interaction phase. We seed the cross-attention with the original self-attention weights. This approach shares similarities with Late-Interaction architectures like ColBERT (Khattab and Zaharia, 2020) but offers higher expressivity, as cross-attentions in MICE can capture more granular, non-linear dependencies between query and document tokens. We also implement Masking Step 1 by masking in the cross-attention the transfers from document to the [CLS].

We hypothesize that the final layers of a transformer pretrained on masked language modeling are highly specialized for token prediction and may be non-essential for the re-ranking task. Consequently, we prune these top layers. The results, presented in Figure 4, confirm this hypothesis: for MiniLM, MICE-$\ell$4+3 model, beginning query and document interactions at layer 4 and keeping only 3 interaction layers, can discard up to 5 layers, utilizing only 7 out of the original 12 backbone layers, without any degradation in effectiveness. For comparisons, we also train MICE-$\ell^{*}$+all variants for each backbone, with all the interaction layers preserved.

*Figure 4. Impact of dropping backbone’s late layers in MICE. 3 interaction layers consistently recovers full performance.*

### 4.2. Baselines

We compare MICE variants with the corresponding unmasked cross-encoders (same baselines as in Section 3). We also compare with ColBERTv2 (Santhanam et al., 2022b), a very strong late-interaction model known for its effectiveness, but which uses the much less expressive MaxSim operation. This makes it a good candidate to measure the effectiveness of our interaction layers’s design. Because ColBERTv2 is based on a much bigger encoder - based on a BERT backbone with $\sim$110M parameters- than the ones used in this work, we also include our own reproduction, based on the MiniLM-v2 backbone.

### 4.3. Results

For a thorough comparison, the experimental setup is the same as in Section 3, including training parameters, evaluation protocol and checkpoints detailed in Section 3.3. We report in Table 4 the results for both backbones, with and without layer dropping. To illustrate the transition between masking the original cross-encoder and MICE, we include results from the Mask 3 as well as our reproduced cross-encoder baselines reported from Table 3.

First, while MICE exhibits a marginal regression in ID effectiveness compared to the full baseline (see table 4), it retains the robust OOD generalization gains observed in Section 3.5. When compared directly against their Masking Step 3 counterparts (which still compute self-attention over document tokens in interaction layers), the MICE-all variants of each backbone show no statistically significant performance drop. This validates our architectural hypothesis from Section 4.1 that document representations do not require further contextualization in interaction layers.

Our hypothesis that late layers of an MLM-trained backbone can be dropped for re-ranking – initially based on ID evaluations in Figure 4 – is strongly corroborated by the averaged OOD results in table 4. The MiniLM-based MICE-$\ell$4+3 (utilizing only 7 of the 12 backbone layers) exceeds its Masking Step 3-$\ell$4 cross-encoder counterpart on OOD tasks with an nDCG@10 of 50.5 (+6 points), outperforming even the previous score of 50.2 reached by the Masking Step 1 baseline in Table 3. Ettin-based models show a different trend: MICE underperforms or matches the average OOD performance of the unmasked baselines (on Ettin-32M, 46.4 for MICE-$\ell$6+3 vs. 48.2).

Although the very efficient standard ColBERTv2 (based on BERT-base) is stronger both in ID and OOD, MICE demonstrates superior effectiveness when matching the backbone size. It surpasses our reproduced MiniLM-ColBERTv2 by 5 and 8 nDCG@10 points on ID and OOD datasets (see table 4), respectively.

In summary, we show that MICE successfully maintains the ID performance of a standard cross-encoder while delivering superior OOD generalization, despite a noticeable representational disadvantage due to the architectural constraints we applied on MICE. If we focus on the subset of "BM25-hard" datasets (last column of Table 4), i.e., where BM25 matches or exceeds the corresponding unmasked cross-encoder (see gray cells in the row Baseline of each backbone in Table 4 for the reference), we note that all the MICE variants manage to outperform their baseline by at least 1.1 and up to 11.3 nDCG@10 points. This also holds for the variants based on Ettin, whose effectiveness is otherwise on par or below their baseline, suggesting that the design of our interaction layers helps MICE to better capture signals of exact match, otherwise missed by a traditional cross-encoder. We observe the same trend for the MiniLM-ColBERT (+3.8 points compared to the MiniLM baseline on the BM25-hard datasets), but the gains are much more limited compared to the corresponding MICE variant (+11.3). This highlights the superior expressiveness of the cross-attention compared to the MaxSim even for detecting explicit signals such as exact-term matchings.

The answer to RQ 3 is therefore positive – we successfully leveraged the insights from our masking analysis to transpose the effectiveness of a cross-encoder into a novel and more efficient architecture that eliminates superfluous interactions.

### 4.4. Efficiency Analysis

To complement the evaluation of the re-ranking effectiveness of MICE, we conduct an efficiency analysis to quantify its potential benefits over established re-ranking architectures. We compare its latency and memory footprint with the standard cross-encoder architecture (entire forward pass over a query-document pair) and ColBERT (separately encoding the query and document before computing a MaxSim over their compressed representations). As MICE can theoretically be used in a setup where document tokens are pre-computed offline, we also present efficiency measures in this setup, which are not applicable to a cross-encoder. In practice, we use the same MiniLM-L12-v2 backbone for each architecture (cross-encoder, ColBERT and MICE) to ensure a fair comparison. We measure the inference time averaged over 100 forward passes on a maximum load setup (512 doc. token length, batch size 128 using an 12Gb Nvidia TITAN-V GPU) and report results in Table 5.

*Table 5. Efficiency comparison of retrieval models. All based on a MiniLM-L12-v2 backbone for comparison.*

| | Model | Precomp. | #param | Latency (ms) | Docs/s | Peak Mem |

$\ell$ $113.28\pm 12.05$ | MiniLM | MICE 4+3 | | 26.3M | | 1130 | 598.44 MB |

$130.36\pm 7.86$ | ColBERT | | 33.4M | | 982 | 331.77 MB |

$470.22\pm 4.87$ | Cross-Encoder | | 33.4M | | 267 | 1193.52 MB |

$\ell$ $241.05\pm 6.25$ | MICE 4+3 | | 33.4M | | 531 | 1071.61 MB |

$498.48\pm 8.65$ | ColBERT | | 33.4M | | 257 | 1195.27 MB |

MICE achieves a $2\times$ speedup over standard cross-encoders, rising to $4\times$ with pre-computed document representations—effectively matching ColBERT’s latency ($1.15\times$) while delivering superior performance (See section 4.3). This is achieved by using fewer layers (i.e., the last layers of the backbone are dropped) and reducing interaction to cross-attention. Compared to ColBERT, we, however, have a larger memory footprint ($1.8\times$).

### 4.5. Scaling laws of MICE

Our main experiments used lightweight backbones (less than 32M parameters) to facilitate extensive experimentation. Here, we report results on scaling laws for MICE. We train a series of models using backbones from the Ettin suite (Weller et al., 2025), a consistent family of ModernBERT-based models ranging in size, allowing for a controlled study of scaling laws and performance trends. The results of this analysis, presented in Figure 5, show that although Ettin-based MICE slightly underperform compared to a standard cross-encoder, its effectiveness scales similarly with the backbone’s size.

*Figure 5. Scaling law of MICE against standard cross-encoder using backbones from the Ettin (Weller et al., 2025) suite.*

## 5. Limitations

Although we demonstrate that MICE enables pre-computing document vectors, matching ColBERT efficiency in this setup, we do not implement the full indexing and first-stage retrieval pipeline in this study. However, in the future, we could leverage existing and well-established methods for indexing and storing multi-vector corpora (MacAvaney et al., 2020; Santhanam et al., 2022a), to effectively turn MICE into a first stage lite cross-encoder.

We further simplified the architecture by removing the self-attention mechanism over query tokens in the interaction layers. This ablation resulted in a collapse in ranking performance, indicating that while the update of document tokens can be bypassed, maintaining intra-query contextualization is critical for effective re-ranking. We also further compressed MICE by reducing the size of its interaction layers by scaling down the hidden states dimension and the hidden size as well as the number of attention heads. This approach proved less effective than layer pruning, as we cannot initialize the compressed interaction layers using the pre-trained weights of the backbone with original dimensions, but instead initialized them randomly, which disrupted the preconditioning we leveraged using pre-trained weights. We identify this as a key direction for future work, where knowledge distillation (e.g., MiniLM-v2 style attention distillation) could effectively guide the training of these low-dimensional interaction layers.

## 6. Conclusion

In this work, motivated by insights from interpretability (Zhan et al., 2020; Lu et al., 2025) and preliminary experiments, we proposed MICE (Minimal-Interaction Cross-Encoder), a new architecture that starts bridging the gap between lightweight late-interaction models and heavy cross-encoders. By strategically combining mid-fusion, cross-attention, and layer pruning, MICE defines a new effectiveness-efficiency compromise.

Our experiments across the BERT and ModernBERT backbones confirm this. MICE decreases fourfold the inference latency compared to standard cross-encoders, matching late-interaction models like ColBERT while retaining most of cross-encoder ID effectiveness and demonstrating superior generalization abilities in OOD.

## Acknowledgements

The authors acknowledge the ANR – FRANCE (French National Research Agency) for its financial support of the GUIDANCE project n°ANR-23-IAS1-0003 as well as the Chaire Multi-Modal/LLM ANR Cluster IA ANR-23-IACL-0007. This work was granted access to the HPC resources of IDRIS under the allocations 2025-A0191016944, 2024-AD011015440R1 and 2025-AD011014444R2 made by GENCI. The authors also gratefully acknowledge the support of the Centre National de la Recherche Scientifique (CNRS) through a research delegation awarded to J. Mothe.

## References

- Amati and Van Rijsbergen (2002) Gianni Amati and Cornelis Joost Van Rijsbergen. 2002. Probabilistic models of information retrieval based on measuring the divergence from randomness. ACM Trans. Inf. Syst. 20, 4 (Oct. 2002), 357–389. doi:10.1145/582415.582416

- Bajaj et al. (2016) Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. 2016. Ms marco: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268 (2016).

- Beltagy et al. (2020) Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The Long-Document Transformer. ArXiv abs/2004.05150 (2020). https://api.semanticscholar.org/CorpusID:215737171

- Campagnano et al. (2025) Cesare Campagnano, Antonio Mallia, Jack Pertschuk, and Fabrizio Silvestri. 2025. E2Rank: Efficient and Effective Layer-Wise Reranking. In European Conference on Information Retrieval. Springer, 417–426.

- Campos et al. (2023) Daniel Campos, Alexandre Marques, Tuan Nguyen, Mark Kurtz, and ChengXiang Zhai. 2023. Sparse*BERT: Sparse Models Generalize To New tasks and Domains. arXiv:2205.12452 [cs.CL] https://arxiv.org/abs/2205.12452

- Cao et al. (2020) Qingqing Cao, Harsh Trivedi, Aruna Balasubramanian, and Niranjan Balasubramanian. 2020. DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (Eds.). Association for Computational Linguistics, Online, 4487–4497. doi:10.18653/v1/2020.acl-main.411

- Clark et al. (2019) Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. 2019. What Does BERT Look at? An Analysis of BERT‘s Attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, Tal Linzen, Grzegorz Chrupała, Yonatan Belinkov, and Dieuwke Hupkes (Eds.). Association for Computational Linguistics, Florence, Italy, 276–286. doi:10.18653/v1/W19-4828

- Craswell et al. (2021) Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2021. Overview of the TREC 2020 deep learning track. In Text REtrieval Conference (TREC). TREC. https://www.microsoft.com/en-us/research/publication/overview-of-the-trec-2020-deep-learning-track/

- Craswell et al. (2020) Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. 2020. Overview of the TREC 2019 deep learning track. In Text REtrieval Conference (TREC). TREC. https://www.microsoft.com/en-us/research/publication/overview-of-the-trec-2019-deep-learning-track/

- Déjean and Clinchant (2025) Hervé Déjean and Stéphane Clinchant. 2025. Reranking with Compressed Document Representation. arXiv:2505.15394 [cs] doi:10.48550/arXiv.2505.15394

- Déjean et al. (2024) Hervé Déjean, Stéphane Clinchant, and Thibault Formal. 2024. A Thorough Comparison of Cross-Encoders and LLMs for Reranking SPLADE. ArXiv abs/2403.10407 (2024). https://api.semanticscholar.org/CorpusID:268510535

- Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Jill Burstein, Christy Doran, and Thamar Solorio (Eds.). Association for Computational Linguistics, Minneapolis, Minnesota, 4171–4186. doi:10.18653/v1/N19-1423

- Ferrando et al. (2024) Javier Ferrando, Gabriele Sarti, Arianna Bisazza, and Marta R. Costa-jussà. 2024. A Primer on the Inner Workings of Transformer-based Language Models. arXiv:2405.00208 [cs.CL] https://arxiv.org/abs/2405.00208

- Formal et al. (2021) Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 2288–2292. doi:10.1145/3404835.3463098

- Frankle and Carbin (2019) Jonathan Frankle and Michael Carbin. 2019. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net. https://openreview.net/forum?id=rJl-b3RcF7

- Hinton et al. (2015) Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the Knowledge in a Neural Network. arXiv:1503.02531 [stat.ML] https://arxiv.org/abs/1503.02531

- Hofstätter et al. (2020) Sebastian Hofstätter, Sophia Althammer, M. Schröder, Mete Sertkan, and A. Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. ArXiv (Oct. 2020).

- Humeau et al. (2020) Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. In International Conference on Learning Representations. https://openreview.net/forum?id=SkxgnnNFvH

- Karpukhin et al. (2020) Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 6769–6781. doi:10.18653/v1/2020.emnlp-main.550

- Khattab and Zaharia (2020) Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, China) (SIGIR ’20). Association for Computing Machinery, New York, NY, USA, 39–48. doi:10.1145/3397271.3401075

- Lei et al. (2025) Yibin Lei, Shwai He, Ang Li, and Andrew Yates. 2025. Making Large Language Models Efficient Dense Retrievers. arXiv:2512.20612 [cs.IR] https://arxiv.org/abs/2512.20612

- Lin et al. (2017) Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. 2017. A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING. In International Conference on Learning Representations. https://openreview.net/forum?id=BJC_jUqxe

- Lu et al. (2025) Meng Lu, Catherine Chen, and Carsten Eickhoff. 2025. Pathway to Relevance: How Cross-Encoders Implement a Semantic Variant of BM25. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China, 25525–25547. doi:10.18653/v1/2025.emnlp-main.1297

- Ma et al. (2024) Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Fine-Tuning LLaMA for Multi-Stage Text Retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (Washington DC, USA) (SIGIR ’24). Association for Computing Machinery, New York, NY, USA, 2421–2425. doi:10.1145/3626772.3657951

- MacAvaney et al. (2020) Sean MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Nazli Goharian, and Ophir Frieder. 2020. Efficient Document Re-Ranking for Transformers by Precomputing Term Representations. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’20). Association for Computing Machinery, New York, NY, USA, 49–58. doi:10.1145/3397271.3401093

- Meng et al. (2024) Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, and Maarten de Rijke. 2024. Ranked List Truncation for Large Language Model-based Re-Ranking. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (Washington DC, USA) (SIGIR ’24). Association for Computing Machinery, New York, NY, USA, 141–151. doi:10.1145/3626772.3657864

- Nogueira and Cho (2019) Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019).

- Rogers et al. (2020) Anna Rogers, Olga Kovaleva, and Anna Rumshisky. 2020. A Primer in BERTology: What We Know About How BERT Works. Transactions of the Association for Computational Linguistics 8 (2020), 842–866. doi:10.1162/tacl_a_00349

- Rosa et al. (2022) Guilherme Rosa, Luiz Bonifacio, Vitor Jeronymo, Hugo Abonizio, Marzieh Fadaee, Roberto Lotufo, and Rodrigo Nogueira. 2022. In Defense of Cross-Encoders for Zero-Shot Retrieval. arXiv:2212.06121 [cs.IR] https://arxiv.org/abs/2212.06121

- Santhanam et al. (2022a) Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022a. PLAID: an efficient engine for late interaction retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 1747–1756.

- Santhanam et al. (2022b) Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022b. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz (Eds.). Association for Computational Linguistics, Seattle, United States, 3715–3734. doi:10.18653/v1/2022.naacl-main.272

- Schlatt et al. (2024) Ferdinand Schlatt, Maik Fröbe, and Matthias Hagen. 2024. Investigating the Effects of Sparse Attention on Cross-Encoders. Vol. 14608. 173–190. arXiv:2312.17649 [cs] doi:10.1007/978-3-031-56027-9_11

- Schlatt et al. (2025) Ferdinand Schlatt, Maik Fröbe, Harrisen Scells, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Benno Stein, Martin Potthast, and Matthias Hagen. 2025. Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-ranking. Springer Nature Switzerland, 323–334. doi:10.1007/978-3-031-88714-7_31

- Sekulić et al. (2020) Ivan Sekulić, Amir Soleimani, Mohammad Aliannejadi, and Fabio Crestani. 2020. Longformer for MS MARCO Document Re-ranking Task. arXiv:2009.09392 [cs.IR] https://arxiv.org/abs/2009.09392

- Su et al. (2024) Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. RoFormer: Enhanced transformer with Rotary Position Embedding. Neurocomputing 568 (2024), 127063. doi:10.1016/j.neucom.2023.127063

- Tan and Bansal (2019) Hao Tan and Mohit Bansal. 2019. LXMERT: Learning Cross-Modality Encoder Representations from Transformers. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 5100–5111. doi:10.18653/v1/D19-1514

- Tay et al. (2022) Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. 2022. Efficient Transformers: A Survey. ACM Comput. Surv. 55, 6, Article 109 (Dec. 2022), 28 pages. doi:10.1145/3530811

- Thakur et al. (2021) Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview.net/forum?id=wCu6T5xFjeJ

- Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoeit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Poloshukin. 2017. Attention Is All You Need. In 31st Conference on Neural Information Processing Systems (NIPS 2017). doi:10.48550/arXiv.1706.03762

- Wang et al. (2020) Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. 2020. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768 (2020).

- Wang et al. (2021) Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong, and Furu Wei. 2021. MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021. Association for Computational Linguistics, Online, 2140–2151. doi:10.18653/v1/2021.findings-acl.188

- Warner et al. (2024) Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo Poli. 2024. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. arXiv:2412.13663 [cs.CL] https://arxiv.org/abs/2412.13663

- Weller et al. (2025) Orion Weller, Kathryn Ricci, Marc Marone, Antoine Chaffin, Dawn Lawrie, and Benjamin Van Durme. 2025. Seq vs Seq: An Open Suite of Paired Encoders and Decoders. (2025). arXiv:2507.11412 [cs.CL] https://arxiv.org/abs/2507.11412

- Wu et al. (2021) Chuhan Wu, Fangzhao Wu, Tao Qi, and Yongfeng Huang. 2021. Fastformer: Additive Attention Can Be All You Need. ArXiv abs/2108.09084 (2021). https://api.semanticscholar.org/CorpusID:237266377

- Xu et al. (2025) Zhichao Xu, Zhiqi Huang, Shengyao Zhuang, and Vivek Srikumar. 2025. Distillation versus Contrastive Learning: How to Train Your Rerankers. arXiv:2507.08336 [cs.CL] https://arxiv.org/abs/2507.08336

- Yates et al. (2021a) Andrew Yates, Rodrigo Nogueira, and Jimmy Lin. 2021a. Pretrained transformers for text ranking: BERT and beyond. In Proceedings of the 14th ACM International Conference on web search and data mining. 1154–1156.

- Yates et al. (2021b) Andrew Yates, Rodrigo Nogueira, and Jimmy Lin. 2021b. Pretrained Transformers for Text Ranking: BERT and Beyond. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, Canada) (SIGIR ’21). Association for Computing Machinery, New York, NY, USA, 2666–2668. doi:10.1145/3404835.3462812

- Zaheer et al. (2020) Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. 2020. Big bird: transformers for longer sequences. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 1450, 15 pages.

- Zhan et al. (2020) Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. An Analysis of BERT in Document Ranking. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, China) (SIGIR ’20). Association for Computing Machinery, New York, NY, USA, 1941–1944. doi:10.1145/3397271.3401325

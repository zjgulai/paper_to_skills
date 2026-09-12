<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2510.14321
     paper_id : 2510.14321
     source   : https://arxiv.org/html/2510.14321v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Large Reasoning Embedding Models:
Towards Next-Generation Dense Retrieval Paradigm

DOI: XXXXXXX.XXXXXXXConference: Make sure to enter the correct conference title from your rights confirmation email; June 03–05, 2018; Woodstock, NYISBN: 978-1-4503-XXXX-X/2018/06CCS: Information systems Language models
Jianting Tang Affiliation: University of Science and Technology of China, State Key Laboratory of Cognitive Intelligence, Hefei, Anhui, China email: jiantingtang@mail.ustc.edu.cn , Dongshuai Li Affiliation: Taobao & Tmall Group of Alibaba, Zhejiang, Hangzhou, China email: lidongshuai.lds@taobao.com , Tao Wen Affiliation: Taobao & Tmall Group of Alibaba, Zhejiang, Hangzhou, China email: wentao.wen@alibaba-inc.com , Fuyu Lv Affiliation: Taobao & Tmall Group of Alibaba, Zhejiang, Hangzhou, China email: fuyu.lfy@taobao.com , Dan Ou Affiliation: Taobao & Tmall Group of Alibaba, Zhejiang, Hangzhou, China email: oudan.od@taobao.com and Linli Xu Affiliation: University of Science and Technology of China, State Key Laboratory of Cognitive Intelligence, Hefei, Anhui, China email: linlixu@ustc.edu.cn

Received 5 June 2009

###### Abstract.

In modern e-commerce search systems, dense retrieval has become an indispensable component. By computing similarities between query and item (product) embeddings, it efficiently selects candidate products from large-scale repositories. With the breakthroughs in large language models (LLMs), mainstream embedding models have gradually shifted from BERT to LLMs for more accurate text modeling. However, these models still adopt direct-embedding methods, and the semantic accuracy of embeddings remains inadequate. Therefore, contrastive learning is heavily employed to achieve tight semantic alignment between positive pairs. Consequently, such models tend to capture statistical co-occurrence patterns in the training data, biasing them toward shallow lexical and semantic matches. For difficult queries exhibiting notable lexical disparity from target items, the performance degrades significantly. In this work, we propose the Large Reasoning Embedding Model (LREM), which novelly integrates reasoning processes into representation learning. For difficult queries, LREM first conducts reasoning to achieve a deep understanding of the original query, and then produces a reasoning-augmented query embedding for retrieval. This reasoning process effectively bridges the semantic gap between original queries and target items, significantly improving retrieval accuracy. Specifically, we adopt a two-stage training process: the first stage optimizes the LLM on carefully curated Query-CoT-Item triplets with SFT and InfoNCE losses to establish preliminary reasoning and embedding capabilities, and the second stage further refines the reasoning trajectories via reinforcement learning (RL). Extensive offline and online experiments validate the effectiveness of LREM, leading to its deployment on China’s largest e-commerce platform since August 2025.

###### Keywords:

Dense Retrieval, Large Language Models, Reasoning, Embedding

## 1. Introduction

With the advancement of deep learning, dense retrieval (Guo et al., 2016; Guo et al., 2020; Mitra and Craswell, 2017) has become an indispensable component of modern e-commerce search systems. By leveraging text embedding models pre-trained on large-scale corpora, dense retrieval is effective at capturing semantic relationships between query texts and item texts. During retrieval, it compares the query embedding against all pre-stored item embeddings to return the most relevant candidates.

The semantic accuracy of embeddings is critical to dense retrieval performance. In the past, BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), and T5 (Raffel et al., 2020) were widely adopted as text embedding models. Recently, LLMs such as LLaMA4 (Meta, 2025), Gemma3 (Team et al., 2025), and Qwen3 (Yang et al., 2025) have achieved significant breakthroughs in text understanding and generation. Motivated by these advances, recent work has explored adapting LLMs into embedding models to leverage their extensive world knowledge and powerful language modeling capabilities. For example, RepLLaMA (Ma et al., 2024) directly uses the last-token hidden state of LLaMA2 (Touvron et al., 2023) as the text embedding. NV-Embed (Lee et al., 2024) modifies the LLM architecture by introducing bidirectional attention and a novel latent attention layer to produce more accurate embeddings.

*Figure 1. Comparison between traditional direct-embedding and the proposed reasoning-then-embedding dense retriever (LREM). LREM leverages reasoning to enable deep query understanding and accurate embeddings, overcoming the superficiality of direct-embedding methods. *

Despite leveraging more powerful LLMs, current dense retrievers still adopt direct-embedding methods (Ma et al., 2024; Li et al., 2024a; Lee et al., 2024) that generate embeddings in a single forward pass, in which the semantic accuracy of embeddings remains significantly constrained. To compensate, these models rely heavily on contrastive learning (Khosla et al., 2020) to forcibly align human-annotated positive query-item pairs in the embedding space, while pushing negative pairs apart. Consequently, this paradigm encourages the model to exploit superficial co-occurrence patterns in the training data and to engage in shallow lexical or semantic matching (Su et al., 2024; Shao et al., 2025; Das et al., 2025). When handling difficult queries that exhibit notable lexical disparity from the target items, performance degrades significantly. As illustrated in Figure 1, for the query “Drinks That Are More Invigorating Than Tea”, the direct-embedding dense retriever produces an inaccurate embedding, returning a large proportion of tea-based drinks rather than intended targets like Coffee or Red Bull.

As inherently generative models, LLMs excel at achieving precise semantic understanding of texts through explicit chain-of-thought (CoT) reasoning (Wei et al., 2022; Yao et al., 2023; Besta et al., 2024; Shao et al., 2024). However, harnessing this distinctive capability for dense retrieval remains largely underexplored in the current literature. Therefore, to unlock the full potential of LLMs, we propose the Large Reasoning Embedding Model (LREM), an entirely new paradigm that integrates reasoning into representation learning. For difficult queries, where directly generating the embedding in a single forward pass results in poor semantic accuracy, LREM performs an explicit reasoning process through CoT generation. Consequently, the CoT serves as a semantic bridge, effectively linking original difficult queries with target items and ultimately enhancing retrieval performance. This reasoning-then-embedding paradigm advances the development of a more intelligent generation of dense retrievers.

To equip LREM with both powerful reasoning and embedding capabilities, we develop a sophisticated data construction pipeline, as illustrated in Figure 2. First, we collect queries from online logs—specifically those exhibiting poor performance under traditional direct-embedding dense retrievers. Then, we employ the advanced Qwen3-30B-A3B-Instruct (Yang et al., 2025) LLM to generate a CoT for each query. To maximize information density and emphasize critical contents, each CoT is finally structured as a compact list of keywords. Then, we feed the query with CoT into a traditional dense retriever to obtain candidate items and filter the results with a relevance model. Consequently, this pipeline yields approximately 75.06 million Query-CoT-Item triples. We adopt a two-stage training process. In the first stage, LREM is jointly optimized via supervised fine-tuning (SFT) and InfoNCE losses, thereby acquiring preliminary reasoning and embedding capabilities. In the second stage, to further refine the reasoning trajectories and enhance CoT quality, we apply a GRPO-based (Shao et al., 2024) RL algorithm while retaining the InfoNCE loss to maintain embedding alignment.

In summary, the contributions of this work are as follows:

-

We propose LREM, a next-generation dense retriever based on a novel reasoning-then-embedding paradigm, effectively overcoming the shallow semantic matching limitations of direct-embedding approaches.

-

We introduce an effective data construction pipeline and a two-stage training process that fully exploit the LLM’s reasoning and embedding capabilities.

-

Extensive offline and online experiments demonstrate the effectiveness of our proposed LREM, establishing a solid foundation for future research in dense retrieval.

## 2. Related Work

### 2.1. Dense Retrieval

In dense retrieval (Zhan et al., 2021; Chen et al., 2024; Kong et al., 2022), queries and items are encoded into a shared semantic space, and relevant items are returned based on embedding similarity using approximate nearest neighbor (ANN) (Arya et al., 1998; Liu et al., 2004; Indyk and Motwani, 1998) search algorithms. Traditional dense retrieval methods primarily rely on fine-tuning pre-trained encoders such as BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), or T5 (Raffel et al., 2020) via contrastive learning to align text embeddings. Recently, increasing studies have adopted LLMs as backbones (Lee et al., 2025; Wang et al., 2024b; Meng et al., 2024; Li et al., 2023) for dense retrieval, leveraging their superior semantic understanding capability and extensive world knowledge. Among them, RepLLaMA (Ma et al., 2024) is the first to demonstrate the effectiveness of directly fine-tuning an open-source LLM for dense retrieval, establishing a strong baseline. Llama2Vec (Li et al., 2024a) applies unsupervised post-pretraining with novel embedding-oriented objectives (EBAE and EBAR) to further improve embedding quality. NV-Embed (Lee et al., 2024) reconfigures the decoder-only architecture with bidirectional attention and an additional latent attention layer, enhancing embedding discriminability. ICL-Embedder (Li et al., 2024b) further enhances generalization by explicitly training the model to leverage in-context examples during embedding generation, resulting in robust few-shot retrieval capabilities. Currently, LLM-based embedding models have achieved SOTA performance across various text retrieval benchmarks, including the MTEB (Muennighoff et al., 2022). However, these models still follow the direct-embedding method and perform poorly on difficult queries that requiring reasoning.

### 2.2. LLM Reasoning

Early methods—like Chain-of-Thought (CoT) (Wei et al., 2022), Tree-of-Thought (ToT) (Yao et al., 2023), and Graph-of-Thought (GoT) (Besta et al., 2024)—relied on human-crafted prompting strategies to guide LLMs in producing reasoning steps or exploring branching trajectories (Zhou et al., 2022; Imani et al., 2023; Zhang et al., 2024a; Wang et al., 2024a). However, within these frameworks, LLMs are unable to learn from prior explorations and to develop inherent reasoning capabilities. This limitation has driven the development of RL–based methods (Li et al., 2025b; Huang et al., 2025; Cui et al., 2025; Zhang et al., 2024b) for eliciting LLM reasoning. Proximal Policy Optimization (PPO) (Schulman et al., 2017; Schulman et al., 2015; Heess et al., 2017; Ouyang et al., 2022) serves as the foundational critic-based baseline, leveraging a learned value model to provide token-level advantage estimates but incurring substantial computational overhead. In verifiable reasoning settings, where reliable sequence-level rewards are available, research has increasingly favored critic-free variants that dispense with the value model entirely. Group Relative Policy Optimization (GRPO) (Shao et al., 2024) exemplifies this shift by replacing critic-based advantages with group-normalized returns across multiple rollouts of the same prompt, reducing variance and stabilizing updates. Building on GRPO, Dynamic sAmpling Policy Optimization (DAPO) (Yu et al., 2025) incorporates dynamic sampling and “clip-higher” objectives to mitigate entropy collapse and concentrate computation on medium-difficulty prompts, thereby enhancing exploration efficiency. Currently, RL (Liu et al., [n. d.]; Zheng et al., 2025; Zhao et al., 2025; Xiaomi et al., 2025; Yi et al., 2025) has offered a promising approach to training LLMs for sophisticated reasoning.

### 2.3. Reasoning-Intensive Retrieval

Reasoning-intensive retrieval (Su et al., 2024; Long et al., 2025; Niu et al., 2024; Lyu et al., 2025; Li et al., 2024c; Zhong et al., 2025) targets documents that cannot be found through simple lexical matching and require reasoning to bridge the semantic gap. Some studies adopt a data-centric approach, directly constructing datasets of reasoning query–item pairs to train embedding models. RaDeR (Das et al., 2025) collects LLM reasoning trajectories in mathematical problem-solving, employing self-reflective relevance evaluation to generate diverse queries and challenging hard negatives, while ReasonIR (Shao et al., 2025) synthesizes varied-length reasoning queries and negatives from seed documents through ReasonIR-Synthesizer, leading to SOTA performance on the BRIGHT (Su et al., 2024) benchmark. Others focus on training specialized query rewriting models to refine queries prior to retrieval. TongSearch-QR (Qin et al., 2025) trains small-scale query reasoning models via GRPO with a semi-rule-based reward, while DeepRetrieval (Jiang et al., 2025) optimizes query rewriting for retrieval via PPO using retrieval metrics as rewards. Some work integrates reasoning and retrieval in an iterative manner (Zhang et al., 2024c; Gao et al., 2025; Muennighoff et al., 2024). R3-RAG (Li et al., 2025a) trains LLMs with both process and outcome rewards to adaptively perform multi-step reasoning and document retrieval, outperforming fixed human-designed workflows. Compared with these complex query preprocessing and multi-stage pipelines, we propose LREM, an entirely new dense retriever that seamlessly integrates reasoning and embedding into a unified process.

## 3. Methodology

In this section, we first formalize the operational process of traditional direct-embedding dense retrievers and our proposed LREM (§3.1). We then describe the construction pipeline for the training data (§3.2), followed by the presentation of LREM’s two-stage training process (§3.3 and §3.4).

### 3.1. Preliminary

Let $q_{i}$ denote a query and $\mathcal{D}=\{d_{1},\dots,d_{n}\}$ denote the collection of candidate items. A dense retriever $f_{\theta}$ aims to project both the query and all candidate items into a shared semantic embedding space, where the set of relevant items for $q_{i}$, represented as $D^{+}_{q_{i}}=\{d^{+}_{q_{i},1},\dots,d^{+}_{q_{i},m}\}$ with $m\ll n$, can be retrieved according to a similarity measure $s(\boldsymbol{q}_{i},\boldsymbol{d}_{j})\in\mathbb{R}$.

For traditional dense retrievers, the query embedding is directly computed as $\boldsymbol{q}_{i}=f_{\theta}(q_{i})$. In contrast, LREM performs reasoning to deeply understand the query before deriving the embedding,

$\displaystyle c_{i}=f^{\mathrm{gen}}_{\theta}(q_{i})=(t_{1},t_{2},\dots,t_{l_{i}}),\quad t_{k}\in\mathcal{V},$ | (1) | | | |

$\displaystyle\boldsymbol{q}_{i}=f^{\mathrm{emb}}_{\theta}([q_{i};c_{i}]),$ | (2) | | | |

where $l_{i}$ is the number of tokens in the generated CoT $c_{i}$, and $\mathcal{V}$ denotes the vocabulary of the LLM. The $c_{i}$ is concatenated with the original $q_{i}$ to form the reasoning-augmented query $[q_{i};c_{i}]$ and then encoded into the final query embedding $\boldsymbol{q}_{i}$.

With item embeddings pre-computed directly as $\boldsymbol{d}_{j}=f_{\theta}(d_{j})$, retrieval proceeds by ranking candidates in $\mathcal{D}$ according to the similarity measure and selecting the top-K results,

$\displaystyle\mathcal{D}_{q_{i}}=\operatorname{TopK}_{d_{j}\in\mathcal{D}}s(\boldsymbol{q}_{i},\boldsymbol{d}_{j}).$ | (3) | | | |

*Figure 2. (1) Data Construction: An LLM generates keyword-based CoTs for each query. By comparing the retrieval results of a traditional dense retriever with and without the CoT, we discard queries where the CoT provides no gains, and filter truly relevant items via an advanced relevance model for the remaining queries. (2) Cold Start: LREM is trained on Query-CoT-Item triplets, where the SFT loss optimizes LREM’s reasoning process and the InfoNCE loss aligns the reasoning-augmented query embedding with the item embedding. (3) Reinforcement Learning: LREM is trained on Query-Item pairs, where the GRPO loss encourages exploration of superior reasoning trajectories under the guidance of a reward system, and the InfoNCE loss concurrently aligns the embeddings. Note that the same LREM is applied to both the query-side and the item-side embedding.*

### 3.2. Data Construction

Given an input query, LREM first engages in reasoning to ensure a deep understanding of its semantics before generating the corresponding query embedding, thereby mitigating inaccuracies arising from the direct-embedding approach. Therefore, LREM needs to possess both reasoning and embedding capabilities. We adopt a two-stage training paradigm: Stage 1 uses carefully constructed Query-CoT-Item triplets for a cold-start training to establish preliminary reasoning and embedding capabilities; Stage 2 applies reinforcement learning to further optimize the reasoning trajectory. This section elaborates on the construction pipeline for the Query-CoT-Item triplet dataset.

#### 3.2.1. CoT Generation

We employ a large-parameter 30B Mixture-of-Experts (MoE) LLM (Yang et al., 2025) to generate high-quality CoT for queries collected from online logs, thereby enabling this powerful model to teach LREM how to reason. Since LREM generates CoT sequentially during online retrieval, general CoTs—formatted as fluent, natural language sentences—can be verbose and introduce significant latency, potentially leading to request timeouts. To mitigate this, we structure each CoT training sample as a compact list of keywords. Figure 2 illustrates the dedicated prompt framework for the CoT construction process, with more details provided in the Appendix. First, we prompt the LLM to perform unconstrained reasoning on the given query, without imposing any restrictions on output format or style, in order to preserve the model’s maximum reasoning capacity. We then feed both the original query and the unconstrained reasoning output back into the LLM, prompting it to perform information extraction to identify relevant keywords. Finally, we apply rule-based post-processing to the extracted results to remove duplicate keywords, keywords overlapping with the query, and any prohibited keywords. To this point, we derive a compact CoT for each query that effectively captures its semantics.

#### 3.2.2. Item Filtering

For difficult queries, such as “Drinks That Are More Invigorating Than Tea”, collecting correctly matched items to form the training data is non-trivial. If a traditional dense retriever directly encodes the original query into an embedding and retrieves the top-$k$ items based on embedding similarity, most of the returned items are irrelevant. We denote this retrieved results as set ①. Therefore, we combine the original query with the previously constructed CoT to jointly obtain the query embedding through the same retriever, and denote the retrieved top-$k$ items as set ②. By leveraging additional CoT information, set ② contains a substantially higher proportion of relevant items. Subsequently, we further process the two sets. We obtain the difference set ②-①, in which all items are retrieved due to the incorporation of CoT information, indicating that the CoT is closely related to the textual content of these items. First, we examine the size of the difference set. If it is 0, indicating that incorporating CoT for the query has no effect on retrieval results and yields no performance gain, this query is directly discarded. Otherwise, we employ our internal advanced relevance model TaoSR1 (Dong et al., 2025)—a 42B MoE LLM trained via multi-stage RL and equipped with thinking capabilities—to assess the relevance between the original query and each item in set ②-①, retaining only items that are judged as relevant. Therefore, in the final constructed Query-CoT-Item triplets, the CoT plays a pivotal role in accurately retrieving the target items. Such data are randomly split into two subsets: one for the cold-start stage using full Query-CoT-Item triplets, and one for the RL stage using only Query-Item pairs.

### 3.3. Cold Start

To enable LREM to reason in the desired keyword-based format and to generate effective embeddings for dense retrieval, we conduct cold-start training on the constructed Query-CoT-Item triplets. Meanwhile, each CoT sample is truncated to a maximum length of $l$, training LREM to efficiently reason over key information within the length limit and avoid request timeouts during online retrieval. Specifically, we introduce three special tokens—<think>, </think>, and <emb>—into LREM’s vocabulary, and employ SFT loss to train the model to reason in the format “<think> Specific CoT </think><emb>”. As the CoTs in the training data are produced by an advanced 30B MoE LLM, this process effectively distills the teacher model’s superior reasoning capabilities into LREM. Formally, the training loss for this process is defined as a standard next-token prediction objective:

$\mathcal{L}_{\text{SFT}}=-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{l_{i}}\log P(t_{j}\mid q_{i},t_{<j}),$ | (4) | | | |

where $q_{i}$ denotes the input query, $t_{j}$ is the $j$-th token in the CoT, $l_{i}$ is the total number of tokens in the CoT, and $N$ is the batch size.

Concurrently with the generative objective, we incorporate the in-batch contrastive learning via the InfoNCE loss to minimize the embedding distance between positive query-item pairs. Specifically, the last hidden state of the <emb> token on the query side is adopted as the overall query embedding. Due to the LLM’s causal attention mechanism, this embedding holistically integrates information from both the original query and the corresponding CoT, thereby possessing more accurate semantics. On the item side, we manually append an <emb> token to the product text title, and likewise use its last hidden state as the item embedding. Both the query and item embeddings are derived from the same LREM.

Considering a mini-batch of $N$ training triples $\{(q_{i},c_{i},d_{i})\}_{i=1}^{N}$, the process for obtaining the query embedding $\boldsymbol{q}_{i}$, the process for obtaining the item embedding $\boldsymbol{d}_{i}$, and the corresponding InfoNCE loss can be formally expressed as:

$\displaystyle\boldsymbol{q}_{i}=\text{H}_{\text{<emb>}}^{\text{[-1]}}[\text{LREM{}}(q_{i}\text{<think>}c_{i}\text{</think>}\text{<emb>})],$ | (5) | | | |

$\displaystyle\boldsymbol{d}_{i}=\text{H}_{\text{<emb>}}^{\text{[-1]}}[\text{LREM{}}(d_{i}\text{<emb>})],$ | (6) | | | |

$\displaystyle\mathcal{L}_{\text{InfoNCE}}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(s(\boldsymbol{q}_{i},\boldsymbol{d}_{i})/\tau)}{\sum_{j=1}^{N}\exp(s(\boldsymbol{q}_{i},\boldsymbol{d}_{j})/\tau)},$ | (7) | | | |

where $\text{LREM{}}(\cdot)$ denotes the LREM processing the input text and outputting a sequence of hidden states, and $\text{H}_{\text{<emb>}}^{\text{[-1]}}$ denotes selecting the last-layer hidden state of the <emb> token. $s(\boldsymbol{q}_{i},\boldsymbol{d}_{j})=\frac{\boldsymbol{q}_{i}^{\top}\boldsymbol{d}_{j}}{\|\boldsymbol{q}_{i}\|\|\boldsymbol{d}_{j}\|}$ computes the cosine similarity between $\boldsymbol{q}_{i}$ and $\boldsymbol{d}_{j}$, and $\tau$ denotes the temperature coefficient.

Finally, the total loss in the cold-start stage is defined as:

$\mathcal{L}=\lambda_{1}\mathcal{L}_{\text{SFT}}+\lambda_{2}\mathcal{L}_{\text{InfoNCE}},$ | (8) | | | |

where $\lambda_{1}$ and $\lambda_{2}$ are the loss coefficients.

### 3.4. Reinforcement Learning

After the cold-start stage, LREM acquires preliminary reasoning and embedding capabilities. However, its reasoning ability is significantly constrained by the quality of the constructed CoT data. As the model is primarily engaged in imitation learning, this substantially hinders the full activation of its intrinsic reasoning capacity. Therefore, at this stage, we employ GRPO to encourage LREM to conduct extensive reasoning exploration, fostering the generation of superior reasoning trajectories. For each $q_{i}$, LREM samples a group of $G$ CoTs, denoted as $\{c_{i}^{g}\}_{g=1}^{G}$, which are then assessed by the reward system from three dimensions: format, length and retrieval accuracy. LREM is encouraged to increase the generation likelihood of CoTs with higher reward scores.

#### 3.4.1. Format Reward

During online inference, the query embedding is only accessible when LREM generates CoTs in the correct “<think> Specific CoT </think><emb>” format. We therefore incorporate a format reward to encourage adherence.

$r_{\text{format}}=\begin{cases}1&\text{if the CoT meets the format specification}\\
0&\text{otherwise}\end{cases}.$ | (9) | | | |

#### 3.4.2. Length Reward

To avoid request timeouts caused by auto-regressive generation of excessively long CoTs, we encourage the model to limit its CoT length to at most $l$.

$r_{\text{length}}=\begin{cases}1&\text{If the CoT length is }\leq l\\
0&\text{otherwise}\end{cases}.$ | (10) | | | |

#### 3.4.3. Retrieval Accuracy Reward

The ultimate objective of this task is to accurately retrieve the target item. Therefore, we propose a retrieval accuracy reward to guide LREM’s exploration. Specifically, for $q_{i}$ and associated $c_{i}^{g}$, we first obtain the corresponding query embedding $\boldsymbol{q}_{i}^{g}$. We then compute its cosine similarity with all item embeddings $\{\boldsymbol{d}_{j}\}_{j=1}^{N}$ in the current batch, and rank these items by the similarity. The retrieval accuracy reward is based on the rank position of $d_{i}$ (i.e., the ground-truth item paired with $q_{i}$). Higher ranks result in larger rewards, whereas lower ranks result in smaller ones. Formally, the reward is defined as,

$\displaystyle\text{rank}(d_{i})=1+\sum_{\begin{subarray}{c}j=1,j\neq i\end{subarray}}^{N}\mathbb{I}\left(s(\boldsymbol{q}_{i}^{g},\boldsymbol{d}_{j})>s(\boldsymbol{q}_{i}^{g},\boldsymbol{d}_{i})\right),$ | (11) | | | |

$\displaystyle r_{\text{accuracy}}=1-\frac{\log\text{rank}(d_{i})}{\log N},$ | (12) | | | |

where $\mathbb{I}$ is the indicator function, returning 1 if the condition holds and 0 otherwise.

Finally, the overall reward is defined as,

$r=\beta_{1}r_{\text{format}}+\beta_{2}r_{\text{length}}+\beta_{3}r_{\text{accuracy}}$ | (13) | | | |

where $\beta_{1}$, $\beta_{2}$ and $\beta_{3}$ are the reward coefficients.

*Table 1. Comparison of offline evaluation results across four challenging query categories. Overall results on all queries are in gray columns. The best results across all models are in bold, and the best baseline results are underlined. *

| Methods | HitRate@6000 | Precision@100 |

| | Q&A | Alternative | Negative | Knowledge | Overall | Q&A | Alternative | Negative | Knowledge | Overall |

| BERT | 11.73 | 30.38 | 34.40 | 23.30 | 24.96 | 69.60 | 26.86 | 57.40 | 50.49 | 51.09 |

| Query-Rewrite | 14.70 | 42.02 | 24.81 | 31.42 | 28.24 | 84.52 | 36.39 | 49.90 | 62.65 | 58.37 |

| Qwen2.5 (Uni-Attn. Last) | 14.61 | 42.20 | 39.54 | 36.24 | 32.52 | 86.20 | 36.17 | 64.05 | 68.44 | 65.38 |

| Qwen2.5 (Uni-Attn. Mean) | 14.47 | 42.05 | 39.33 | 36.18 | 32.38 | 86.14 | 36.02 | 63.90 | 68.23 | 65.24 |

| Qwen2.5 (Uni-Attn. Latent) | 14.76 | 42.43 | 39.65 | 36.44 | 32.69 | 86.06 | 35.91 | 63.84 | 68.09 | 65.14 |

| Qwen2.5 (Uni-Attn. Ly4) | 14.70 | 42.32 | 39.58 | 36.32 | 32.60 | 86.29 | 36.53 | 64.08 | 68.57 | 65.52 |

| Qwen2.5 (Bi-Attn. Last) | 14.95 | 42.54 | 39.94 | 36.61 | 32.89 | 86.35 | 36.86 | 64.16 | 68.72 | 65.66 |

| LREM (Cold Start) | 14.92 | 41.79 | 39.30 | 36.20 | 32.45 | 85.73 | 35.47 | 63.11 | 68.34 | 64.83 |

| LREM (Cold Start+RL) | 17.82 | 45.01 | 41.55 | 37.29 | 34.78 | 89.97 | 40.18 | 66.34 | 69.94 | 68.22 |

#### 3.4.4. Training Objective

We adopt a GRPO-based RL algorithm to optimize LREM’s reasoning trajectory. The loss is defined as:

$\displaystyle\mathcal{L}_{\mathrm{GRPO}}$ $\displaystyle=-\mathbb{E}\bigg[\sum_{i=1}^{G}\min\Bigg(\frac{\pi_{\theta}(c_{i}|q)}{\pi_{\theta_{\mathrm{old}}}(c_{i}|q)}\,A_{i},$ | | | | |

$\displaystyle\qquad\qquad\mathrm{clip}\!\left(\frac{\pi_{\theta}(c_{i}|q)}{\pi_{\theta_{\mathrm{old}}}(c_{i}|q)},\ 1-\epsilon,\ 1+\epsilon\right)A_{i}\Bigg)\bigg],$ | (14) | | | | |

where $A_{i}=\frac{r_{i}-\text{mean}(\{r_{1},r_{2},\cdots,r_{G}\})}{\text{std}(\{r_{1},r_{2},\cdots,r_{G}\})}$ denotes the advantage.

Concurrently, we employ the InfoNCE loss to dynamically align all reasoning-augmented query embeddings $\{\boldsymbol{q}_{i}^{g}\}_{g=1}^{G}$ corresponding to $q_{i}$, with the target item embedding $\boldsymbol{d}_{i}$. The loss is defined as:

$\mathcal{L}_{\text{InfoNCE}}=-\frac{1}{NG}\sum_{i=1}^{N}\sum_{g=1}^{G}\log\frac{\exp\left(s(\boldsymbol{q}_{i}^{g},\boldsymbol{d}_{i})/\tau\right)}{\sum_{j=1}^{N}\exp\left(s(\boldsymbol{q}_{i}^{g},\boldsymbol{d}_{j})/\tau\right)},$ | (15) | | | |

where $G$ denotes the sampling group size.

Finally, the total loss in the RL stage is defined as:

$\mathcal{L}=\gamma_{1}\mathcal{L}_{\text{GRPO}}+\gamma_{2}\mathcal{L}_{\text{InfoNCE}},$ | (16) | | | |

where $\gamma_{1}$ and $\gamma_{2}$ are the loss coefficients.

## 4. Experiments

### 4.1. Experimental Setup

#### 4.1.1. Dataset

By using Qwen3-30B-A3B-Instruct (Yang et al., 2025) to construct CoT for each query and an advanced relevance model TaoSR1 (Dong et al., 2025) to identify relevant items, we ultimately obtain 75.06 million Query-CoT-Item triples. We randomly set aside 4 million Query-Item pairs for the RL stage, and use the remaining data for the cold-start stage. To rigorously assess LREM’s performance, the test set predominantly includes four extremely challenging query categories: question-answering (Q&A), affordable alternative, negative, and knowledge-intensive queries. In total, the test set contains 7209 queries, with a candidate product pool of 76.63 million items. More details are provided in the Appendix.

#### 4.1.2. Metrics

In offline experiments, the performance is evaluated based on Hit Rate (Herlocker et al., 2004) and Precision (Cleverdon et al., 1966) metrics, and in online experiments, the GSB (Good/Same/Bad) (Hawking et al., 2001) metric is adopted.

-

HitRate@6000: It measures the ratio of ground-truth items that appear within the top 6000 retrieved items, relative to the total number of ground-truth items.

-

Precision@100: It evaluates the proportion of the top 100 retrieved items that are judged as relevant by TaoSR1.

-

GSB: It evaluates the superiority of a test bucket against the base bucket in A/B experiments by having human assessors compare the test model’s retrieval results with the base model’s for the same queries side-by-side. GSB $+x\%$ indicates that the test bucket outperforms the base bucket on $x\%$ of queries.

#### 4.1.3. Implementation Details

We adopt Qwen2.5-3B-Instruct (An Yang et al., 2024) and conduct training on 128 GPUs. During the cold start stage, CoT training samples are truncated to a maximum length of $l=16$. The loss coefficients are set to $\lambda_{1}=0.1$ and $\lambda_{2}=1$, with a per-GPU batch size of 128, a learning rate of 1e-5, and a cosine scheduler with a warmup ratio of 0.03. The model is trained for 1 epoch with all parameters optimized. In the reinforcement learning stage, $G=8$ CoTs are sampled for each query. In the length reward, the threshold is set to $l=16$, and reward coefficients are configured as $\beta_{1}=0.5$, $\beta_{2}=0.2$ and $\beta_{3}=1$. The loss coefficients are set to $\gamma_{1}=1$ and $\gamma_{2}=0.1$, with a per-GPU batch size of 256, a learning rate of 1e-6, and a cosine scheduler with a warmup ratio of 0.03. The model is trained for 1 epoch with all parameters updated.

#### 4.1.4. Baselines

We adopt Qwen2.5-3B-Instruct as the base model and reproduce a range of mainstream methods and configurations to serve as baselines. All baselines are trained for one epoch on the full 75.06 million Query-Item pairs.

-

BERT (Xiao et al., 2022): A 12-layer 110M BERT trained via the RetroMAE (Xiao et al., 2022) method and contrastive learning.

-

Query-Rewrite (Feng et al., 2025): Query rewriting via CSA-QR (Feng et al., 2025), followed by retrieval via inverted index.

-

Qwen2.5 (Uni-Attn. Last) (Ma et al., 2024): Unidirect-attention; embedding from last token’s final-layer hidden state.

-

Qwen2.5 (Uni-Attn. Mean) (Lee et al., 2025): Unidirect-attention; embedding from mean of all final-layer token hidden states.

-

Qwen2.5 (Uni-Attn. Latent) (Lee et al., 2024): Unidirect-attention; embedding from NV-Embed’s latent attention layer.

-

Qwen2.5 (Uni-Attn. Ly4) (Tenney et al., 2019): Unidirect-attention; embedding from mean of last token’s final 4 layer hidden states.

-

Qwen2.5 (Bi-Attn. Last) (Li et al., 2023): Bidirect-attention; embedding from last token’s final-layer hidden state.

*Figure 3. Representative examples from the four query categories. Compared with the direct-embedding method Qwen2.5 (Uni-Attn. Last), which performs only superficial lexical matching, LREM adopts a novel reasoning-then-embedding paradigm, enabling accurate query semantic understanding and target item retrieval. *

### 4.2. Offline Experiments

#### 4.2.1. Main Results

Offline evaluations are conducted based on the HitRate@6000 and Precision@100 metrics. As shown in Table 1, LREM trained with both cold-start and reinforcement learning stages achieves the best performance, outperforming the strongest baseline, Qwen2.5 (Bi-Attn. Last), by 5.75% on HitRate@6000 and by 3.90% on Precision@100 over all queries. Notably, the Q&A and Alternative categories achieve the most substantial improvements, with gains of $19.20\%$ and $5.81\%$ on HitRate@6000, and $4.19\%$ and $9.01\%$ on Precision@100, respectively. This improvement clearly highlights the superiority of LREM’s reasoning-then-embedding paradigm. Qwen2.5 (Uni-Attn. Last) outperforms BERT, primarily due to its larger parameter scale and extensive next-token-prediction pre-training on broad corpora. However, its unidirectional attention mechanism prevents earlier tokens from incorporating information from subsequent ones, which to some extent constrains the accuracy of semantic embeddings. Therefore, Qwen2.5 (Bi-Attn. Last) achieves slightly better performance. Since LREM requires both reasoning and embedding capabilities, it retains a unidirectional attention architecture. Despite this constraint, it achieves the best performance, demonstrating that explicit reasoning effectively overcomes the fundamental flaw of superficial semantic understanding in direct-embedding methods. Compared with the Query-Rewrite method, query rewriting inevitably incurs information loss from the original query, and the accumulation of errors across multiple stages further limits the final retrieval performance. In contrast, as a unified model, LREM fosters deeper semantic understanding by incorporating reasoning before generating the final embedding, effectively bridging the semantic gap between original queries and target items.

#### 4.2.2. Case Studies

As illustrated in Figure 3, we present several representative examples to illustrate the effectiveness of LREM. For queries “What’s Needed to Ride an E-Bike”, “Xbox Controller Alternative”, and “Non-waisted Dress”, the direct-embedding method Qwen2.5 (Uni-Attn. Last) fails to capture the underlying intent of the original query and merely performs superficial lexical matching. Consequently, it incorrectly retrieves items whose textual titles contain keywords like “E-Bike”, “Xbox Controller”, and “Waisted Dress”. In contrast, LREM conducts explicit reasoning on the query before generating embeddings, thoroughly uncovering the underlying product-search intent behind the queries. Specifically, it infers the first query likely requires items such as “Helmet, Riding Gloves, Reflective Vest”; the second suggests suitable substitutes like “THUNDEROBOT, BETOP”; and the third refers to “Loose-fit Dress, Straight Maxi Dress”. Through this reasoning process, LREM ultimately generates accurate embeddings and successfully retrieves the correct items. For the query “October Fruits”, Qwen2.5 (Uni-Attn. Last) naively retrieves various fruits, many of which are not typically harvested in October, indicating its limited understanding of temporal semantics. In contrast, LREM accurately captures the seasonal context expressed in the original query.

*Table 2. Retrieval performance under different manual modifications of LREM’s CoT content.*

| Methods | HitRate@6000 | Precision@100 |

| LREM | 34.78 | 68.22 |

| LREM (Empty-CoT) | 31.59 | 64.25 |

| LREM (Random-CoT) | 30.16 | 62.32 |

| LREM (Query-CoT) | 32.54 | 65.63 |

### 4.3. Ablation Experiments

#### 4.3.1. Effect of Reinforcement Learning

In the cold start stage, LREM develops preliminary reasoning and embedding capacities. In this section, we assess the effect of RL. As shown in Table 1, RL contributes critically to the final performance gains, substantially improves retrieval effectiveness across various types of queries, with an overall gain of 7.18% in HitRate@6000 and 5.23% in Precision@100. Representative examples are presented in Figure 4. For the query “Meats That Pair Well with Brandy”, the cold-start model exhibits incorrect reasoning, which misguides embedding generation and retrieves entirely irrelevant items. After RL, LREM performs more accurate reasoning, enabling correct item retrieval. For “Japanese Pilot Pen Alternative”, the cold-start model performs ineffective reasoning—“Pilot Alternative, Inexpensive”—which fails to delve into the original query and reason out additional valuable information, leading to persistent retrieval errors. In contrast, the RL-enhanced LREM effectively follows the reasoning trajectory involving “M&G, Deli, Zebra”, returning intended items. For “Birthday Gift for 18-Year-Old Girl”, the cold-start model conducts suboptimal reasoning, suggesting plausible but generic items such as “Smart Fitness Band, AirPods”. RL enables LREM to conduct more targeted reasoning—“Jewelry Set, Necklace, Bracelet”—better aligning with the intended gifting context. In the cold-start stage, LREM primarily relies on imitation learning and is constrained by the quality of the constructed CoT data. RL effectively unlocks LREM’s inherent capacity for deeper query reasoning and facilitates the exploration of superior reasoning trajectories.

*Figure 4. Comparison of generated CoT and retrieval results between LREM (Cold Start) and LREM (Cold Start + RL).*

#### 4.3.2. Effect of CoT Content

The CoT generates by LREM offers a transparent view into the model’s internal process of query understanding. To quantitatively assess the impact of CoT content on both embedding accuracy and retrieval performance, We deliberately prevent LREM from generating the CoT on its own. Instead, we manually construct three CoT variants: (1)Empty-CoT, without any reasoning content (i.e., Query<think></think><emb>); (2) Random-CoT, containing a sequence of random tokens (i.e., Query<think>Random Tokens</think><emb>); and (3) Query-CoT, which simply repeats the original query within the reasoning field (i.e., Query<think>Query</think><emb>). As shown in Table 2, LREM (Empty-CoT) exhibits a substantial performance drop compared to LREM—HR@6000 decreases by 9.17% and Precision@100 decreases by 5.82%—because it essentially degenerates into traditional direct-embedding methods, preventing the model from developing a deep understanding of the query before generating its embedding. LREM (Random-CoT) exhibits the worst performance, as it introduces substantial noise unrelated to the original query. Although LREM (Query-CoT) does not contribute to achieving a deeper level of semantic understanding, the simple repetition of the original query, to some extent, alleviates the limitations of unidirectional attention—where earlier tokens cannot attend to later ones—thereby yielding marginally better performance compared to LREM (Empty-CoT), with +0.95 points in HR@6000 and +1.38 points in Precision@100.

Retrieval performance across varying CoT lengths.

*Figure 5. Retrieval performance across varying CoT lengths generated by LREM.Retrieval performance across varying CoT lengths.*

#### 4.3.3. Effect of CoT Length

To investigate the influence of CoT length on retrieval performance, we train LREM variants with different reasoning lengths by adjusting $l$. As shown in Figure 5, extending the CoT length in LREM from 16 to 32 tokens yields a noticeable improvement in retrieval performance, as overly restrictive length constraints can hinder the model’s reasoning process, leading to an incomplete understanding of the query. However, further increasing the CoT length to 48 or 64 tokens leads to a decline in performance. This is attributed to the fact that, LREM conducts reasoning in a keyword-centric manner, excessively long and divergent keyword sequences may diminish semantic precision and distract the model’s focus, thereby introducing difficulties for the final embedding. In our final models, $l$ is set to 16 to achieve a balance between retrieval performance and computational efficiency.

### 4.4. Online Experiments

*Table 3. Online A/B testing results.*

| | Q&A | Alternative | Negative | Knowledge |

| GSB | +7.39% | +7.27% | +15.7% | +4.94% |

We conduct an online A/B test via side-by-side human evaluations on 2000 live queries (four categories), comparing the experimental bucket (LREM) against the base bucket (the current best online model). Results show consistent improvements: +7.39% on Q&A, +7.27% on Alternative, +15.7% on Negative, and +4.94% on Knowledge queries. Due to the additional reasoning process incorporated by LREM during online retrieval, it incurs extra latency, with the average retrieval time rising from 15ms to 50ms relative to the base bucket. Within the maximum allowable latency, LREM trades longer processing time for more accurate retrieval. In e-commerce search scenarios, where queries often involve subtle preferences or implicit needs, LREM’s reasoning-then-embedding approach effectively links user queries with intended products, leading to more precise and meaningful retrieval.

## 5. Conclusion

This paper proposes LREM, a novel reasoning-then-embedding dense retriever. In contrast to traditional direct-embedding models, LREM first performs reasoning over the original query to ensure a deep semantic understanding, and then produces the corresponding embeddings. By introducing the explicit reasoning process, LREM goes beyond superficial lexical matching and achieves superior retrieval performance on highly difficult queries—such as Q&A, alternative, negative, and knowledge types. Extensive offline and online experiments demonstrate LREM’s impressive performance, positioning LREM as an important step toward the development of next-generation, more intelligent embedding models.

## References

- An Yang et al. (2024) Beichen Zhang Binyuan Hui Bo Zheng Bowen Yu Chengyuan Li Dayiheng Liu Fei Huang Haoran Wei Huan Lin An Yang, Baosong Yang et al. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115 (2024).

- Arya et al. (1998) Sunil Arya, David M Mount, Nathan S Netanyahu, Ruth Silverman, and Angela Y Wu. 1998. An optimal algorithm for approximate nearest neighbor searching fixed dimensions. Journal of the ACM (JACM) 45, 6 (1998), 891–923.

- Besta et al. (2024) Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. 2024. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, Vol. 38. 17682–17690.

- Chen et al. (2024) Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. 2024. Dense x retrieval: What retrieval granularity should we use?. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 15159–15177.

- Cleverdon et al. (1966) Cyril Cleverdon, Jack Mills, and Michael Keen. 1966. FACTORS DETERMINING THE PERFORMANCE OF INDEXING SYSTEMS VOLUME 1. DESIGN. Vol. 1. Parts.

- Cui et al. (2025) Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. 2025. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456 (2025).

- Das et al. (2025) Debrup Das, Sam O’ Nuallain, and Razieh Rahimi. 2025. RaDeR: Reasoning-aware Dense Retrieval Models. arXiv preprint arXiv:2505.18405 (2025).

- Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 4171–4186.

- Dong et al. (2025) Chenhe Dong, Shaowei Yao, Pengkun Jiao, Jianhui Yang, Yiming Jin, Zerui Huang, Xiaojiang Zhou, Dan Ou, and Haihong Tang. 2025. TaoSR1: The Thinking Model for E-commerce Relevance Search. arXiv preprint arXiv:2508.12365 (2025).

- Feng et al. (2025) Yunling Feng, Gui Ling, Yue Jiang, Jianfeng Huang, Dan Ou, Qingwen Liu, Fuyu Lv, and Yajing Xu. 2025. Complicated Semantic Alignment for Long-Tail Query Rewriting in Taobao Search Based on Large Language Model. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 4435–4446.

- Gao et al. (2025) Yunfan Gao, Yun Xiong, Yijie Zhong, Yuxi Bi, Ming Xue, and Haofen Wang. 2025. Synergizing rag and reasoning: A systematic review. arXiv preprint arXiv:2504.15909 (2025).

- Guo et al. (2016) Jiafeng Guo, Yixing Fan, Qingyao Ai, and W Bruce Croft. 2016. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM international on conference on information and knowledge management. 55–64.

- Guo et al. (2020) Jiafeng Guo, Yixing Fan, Liang Pang, Liu Yang, Qingyao Ai, Hamed Zamani, Chen Wu, W Bruce Croft, and Xueqi Cheng. 2020. A deep look into neural ranking models for information retrieval. Information Processing & Management 57, 6 (2020), 102067.

- Hawking et al. (2001) David Hawking, Nick Craswell, Peter Bailey, and Kathleen Griffihs. 2001. Measuring search engine quality. Information retrieval 4, 1 (2001), 33–59.

- Heess et al. (2017) Nicolas Heess, Dhruva TB, Srinivasan Sriram, J Lemmon, J Merel, G Wayne, Y Tassa, T Erez, Z Wang, S Eslami, et al. 2017. Emergence of locomotion behaviours in rich environments. arXiv 2017. arXiv preprint arXiv:1707.02286 (2017).

- Herlocker et al. (2004) Jonathan L Herlocker, Joseph A Konstan, Loren G Terveen, and John T Riedl. 2004. Evaluating collaborative filtering recommender systems. ACM Transactions on Information Systems (TOIS) 22, 1 (2004), 5–53.

- Huang et al. (2025) Chenghua Huang, Lu Wang, Fangkai Yang, Pu Zhao, Zhixu Li, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. 2025. Lean and mean: Decoupled value policy optimization with global value guidance. arXiv preprint arXiv:2502.16944 (2025).

- Imani et al. (2023) Shima Imani, Liang Du, and Harsh Shrivastava. 2023. Mathprompter: Mathematical reasoning using large language models. arXiv preprint arXiv:2303.05398 (2023).

- Indyk and Motwani (1998) Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing. 604–613.

- Jiang et al. (2025) Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, and Jiawei Han. 2025. Deepretrieval: Hacking real search engines and retrievers with large language models via reinforcement learning. arXiv preprint arXiv:2503.00223 (2025).

- Khosla et al. (2020) Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. Advances in neural information processing systems 33 (2020), 18661–18673.

- Kong et al. (2022) Weize Kong, Swaraj Khadanga, Cheng Li, Shaleen Kumar Gupta, Mingyang Zhang, Wensong Xu, and Michael Bendersky. 2022. Multi-aspect dense retrieval. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3178–3186.

- Lee et al. (2024) Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428 (2024).

- Lee et al. (2025) Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, et al. 2025. Gemini embedding: Generalizable embeddings from gemini. arXiv preprint arXiv:2503.07891 (2025).

- Li et al. (2024a) Chaofan Li, Zheng Liu, Shitao Xiao, Yingxia Shao, and Defu Lian. 2024a. Llama2vec: Unsupervised adaptation of large language models for dense retrieval. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 3490–3500.

- Li et al. (2024b) Chaofan Li, MingHao Qin, Shitao Xiao, Jianlyu Chen, Kun Luo, Yingxia Shao, Defu Lian, and Zheng Liu. 2024b. Making text embedders few-shot learners. arXiv preprint arXiv:2409.15700 (2024).

- Li et al. (2024c) Xingxuan Li, Weiwen Xu, Ruochen Zhao, Fangkai Jiao, Shafiq Joty, and Lidong Bing. 2024c. Can we further elicit reasoning in llms? critic-guided planning with retrieval-augmentation for solving challenging tasks. arXiv preprint arXiv:2410.01428 (2024).

- Li et al. (2025b) Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025b. Limr: Less is more for rl scaling. arXiv preprint arXiv:2502.11886 (2025).

- Li et al. (2025a) Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, and Xipeng Qiu. 2025a. R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning. arXiv preprint arXiv:2505.23794 (2025).

- Li et al. (2023) Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281 (2023).

- Liu et al. (2004) Ting Liu, Andrew Moore, Ke Yang, and Alexander Gray. 2004. An investigation of practical approximate nearest neighbor algorithms. Advances in neural information processing systems 17 (2004).

- Liu et al. (2019) Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 (2019).

- Liu et al. ([n. d.]) Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. [n. d.]. Understanding r1-zero-like training: A critical perspective, 2025. URL https://arxiv. org/abs/2503.20783 ([n. d.]).

- Long et al. (2025) Meixiu Long, Duolin Sun, Dan Yang, Junjie Wang, Yue Shen, Jian Wang, Peng Wei, Jinjie Gu, and Jiahai Wang. 2025. DIVER: A Multi-Stage Approach for Reasoning-intensive Information Retrieval. arXiv preprint arXiv:2508.07995 (2025).

- Lyu et al. (2025) Xinxi Lyu, Michael Duan, Rulin Shao, Pang Wei Koh, and Sewon Min. 2025. Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks. arXiv preprint arXiv:2507.01297 (2025).

- Ma et al. (2024) Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Fine-tuning llama for multi-stage text retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2421–2425.

- Meng et al. (2024) Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. 2024. Sfrembedding-mistral: enhance text retrieval with transfer learning. Salesforce AI Research Blog 3 (2024), 6.

- Meta (2025) AI Meta. 2025. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai. meta. com/blog/llama-4-multimodal-intelligence/, checked on 4, 7 (2025), 2025.

- Mitra and Craswell (2017) Bhaskar Mitra and Nick Craswell. 2017. Neural models for information retrieval. arXiv preprint arXiv:1705.01509 (2017).

- Muennighoff et al. (2024) Niklas Muennighoff, SU Hongjin, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. 2024. Generative representational instruction tuning. In The Thirteenth International Conference on Learning Representations.

- Muennighoff et al. (2022) Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2022. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316 (2022).

- Niu et al. (2024) Tong Niu, Shafiq Joty, Ye Liu, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. 2024. Judgerank: Leveraging large language models for reasoning-intensive reranking. arXiv preprint arXiv:2411.00142 (2024).

- Ouyang et al. (2022) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems 35 (2022), 27730–27744.

- Qin et al. (2025) Xubo Qin, Jun Bai, Jiaqi Li, Zixia Jia, and Zilong Zheng. 2025. TongSearch-QR: Reinforced Query Reasoning for Retrieval. arXiv preprint arXiv:2506.11603 (2025).

- Raffel et al. (2020) Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.

- Schulman et al. (2015) John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. 2015. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438 (2015).

- Schulman et al. (2017) John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).

- Shao et al. (2025) Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muennighoff, Xi Victoria Lin, Daniela Rus, Bryan Kian Hsiang Low, Sewon Min, Wen-tau Yih, Pang Wei Koh, et al. 2025. ReasonIR: Training Retrievers for Reasoning Tasks. arXiv preprint arXiv:2504.20595 (2025).

- Shao et al. (2024) Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).

- Su et al. (2024) Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan Shi, Zachary S Siegel, Michael Tang, et al. 2024. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. arXiv preprint arXiv:2407.12883 (2024).

- Team et al. (2025) Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786 (2025).

- Tenney et al. (2019) Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019. BERT rediscovers the classical NLP pipeline. arXiv preprint arXiv:1905.05950 (2019).

- Touvron et al. (2023) Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).

- Wang et al. (2024a) Evan Wang, Federico Cassano, Catherine Wu, Yunfeng Bai, Will Song, Vaskar Nath, Ziwen Han, Sean Hendryx, Summer Yue, and Hugh Zhang. 2024a. Planning in natural language improves llm search for code generation. arXiv preprint arXiv:2409.03733 (2024).

- Wang et al. (2024b) Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024b. Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672 (2024).

- Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35 (2022), 24824–24837.

- Xiao et al. (2022) Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. 2022. RetroMAE: Pre-training retrieval-oriented language models via masked auto-encoder. arXiv preprint arXiv:2205.12035 (2022).

- Xiaomi et al. (2025) LLM Xiaomi, Bingquan Xia, Bowen Shen, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, et al. 2025. MiMo: Unlocking the Reasoning Potential of Language Model–From Pretraining to Posttraining. arXiv preprint arXiv:2505.07608 (2025).

- Yang et al. (2025) An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

- Yao et al. (2023) Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems 36 (2023), 11809–11822.

- Yi et al. (2025) Jingyang Yi, Jiazheng Wang, and Sida Li. 2025. Shorterbetter: Guiding reasoning models to find optimal inference length for efficient reasoning. arXiv preprint arXiv:2504.21370 (2025).

- Yu et al. (2025) Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. 2025. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv. org/abs/2503.14476 (2025).

- Zhan et al. (2021) Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 1503–1512.

- Zhang et al. (2024c) Jintian Zhang, Cheng Peng, Mengshu Sun, Xiang Chen, Lei Liang, Zhiqiang Zhang, Jun Zhou, Huajun Chen, and Ningyu Zhang. 2024c. Onegen: Efficient one-pass unified generation and retrieval for llms. arXiv preprint arXiv:2409.05152 (2024).

- Zhang et al. (2024b) Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, and Min Lin. 2024b. Chain of preference optimization: Improving chain-of-thought reasoning in llms. Advances in Neural Information Processing Systems 37 (2024), 333–356.

- Zhang et al. (2024a) Yongheng Zhang, Qiguang Chen, Min Li, Wanxiang Che, and Libo Qin. 2024a. AutoCAP: Towards automatic cross-lingual alignment planning for zero-shot chain-of-thought. arXiv preprint arXiv:2406.13940 (2024).

- Zhao et al. (2025) Yuzhong Zhao, Yue Liu, Junpeng Liu, Jingye Chen, Xun Wu, Yaru Hao, Tengchao Lv, Shaohan Huang, Lei Cui, Qixiang Ye, et al. 2025. Geometric-mean policy optimization. arXiv preprint arXiv:2507.20673 (2025).

- Zheng et al. (2025) Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. 2025. Group sequence policy optimization. arXiv preprint arXiv:2507.18071 (2025).

- Zhong et al. (2025) Yunfei Zhong, Jun Yang, Yixing Fan, Jiafeng Guo, Lixin Su, Maarten de Rijke, Ruqing Zhang, Dawei Yin, and Xueqi Cheng. 2025. Reasoning-enhanced Query Understanding through Decomposition and Interpretation. arXiv preprint arXiv:2509.06544 (2025).

- Zhou et al. (2022) Fan Zhou, Haoyu Dong, Qian Liu, Zhoujun Cheng, Shi Han, and Dongmei Zhang. 2022. Reflection of thought: Inversely eliciting numerical reasoning in language models via solving linear systems. arXiv preprint arXiv:2210.05075 (2022).

*Table 4. Detailed prompts for instructing Qwen3-30B-A3B-Instruct in unconstrained reasoning.*

| System Prompt |

| You are a query assistant for a Chinese e-commerce search platform, highly adept at understanding various user queries. |

| User Prompt |

| The following are examples illustrating the interpretation of several common query types: |

| |

| 1. Substitute-seeking queries: For instance, when a user inputs “ysl100 dupe”, the intent should be understood as seeking a more affordable alternative lipstick with the same shade as YSL 100. |

| 2. Lookalike-seeking queries: For example, the query “lululemon lookalike yoga pants” indicates the user’s desire to find yoga pants from other brands that resemble Lululemon in style and fit. |

| 3. Question-answering queries: When a user enters “what size for a 1.5-year-old boy”, the intent is to purchase clothing or footwear for a 1.5-year-old boy, requiring a recommendation based on appropriate age-specific sizing. |

| 4. General queries: For example, a query like “dresses” represents a standard product search, which can be processed using general retrieval and understanding methods. |

| |

| Based on the above examples, please analyze the following query: “La Mer dupe” |

| Response |

| The query “La Mer dupe” refers to products that deliver similar repair, hydration, and anti-aging benefits as La Mer, but at a lower price. La Mer is known for high-performance formulations, with flagship products like The Moisturizing Cream and The Concentrate positioned in the luxury segment. While its proprietary Miracle Broth™ remains unique, alternatives using peptides, ceramides, hyaluronic acid, centella asiatica, and niacinamide can offer comparable functional benefits. Below are recommended alternatives, categorized by product function. |

| 1. Alternatives to The Moisturizing Cream (Focus: Repair + Anti-Aging) |

| Winona Barrier Repair Cream |

| →Ideal for sensitive skin; strengthens the skin barrier with mild, effective ingredients. High value for daily repair. |

| Proya Red Carpet Peptide Cream |

| →Combines peptides and niacinamide for anti-aging. Rich yet non-greasy, priced at 1/5 of La Mer. |

| The Ordinary 10% Niacinamide + 1% Zinc (paired with moisturizer) |

| →Budget-friendly option for oil control, brightening, and repair when layered. |

| 2. Alternatives to The Concentrate (Focus: Repair + Brightening) |

| SkinCeuticals CE Ferulic Serum |

| →Premium alternative with strong antioxidant and anti-photodamage effects; more affordable than La Mer. |

| Runbaiyan Hyaluronic Acid Repair Serum |

| →Simple, effective formula with hyaluronic acid and ceramides; suitable for daily barrier support. |

| HBN Hyaluronic Acid Repair Serum |

| →Contains hyaluronic acid and centella asiatica; targets overnight recovery and soothing, ideal for younger skin. |

| Key Considerations |

| La Mer’s Miracle Broth™ has no direct equivalent. |

| Functional similarity can be achieved through targeted ingredient combinations. Selection should be based on skin type (dry/oily/sensitive) and individual needs, not trends. |

*Table 5. Detailed prompts for instructing Qwen3-30B-A3B-Instruct in information extraction.*

| System Prompt |

| You are a data processing assistant specialized in the Chinese e-commerce domain, proficient in extracting keyword phrases from the reasoning outputs of LLMs that are closely relevant to users’ search queries. |

| User Prompt |

| Please return the extraction results according to the following requirements: |

| 1. Output keyword phrases only—eliminate duplicates, avoid redundant prefixes, and exclude any terms unrelated to products or brands. |

| 2. If the user query explicitly targets a specific brand, do not include other brand names. |

| 3. For question-answering or dupe-seeking queries, extract the product/brand name or specific product attributes. |

| 4. If multiple keywords are extracted, separate them with “,”. |

| |

| Now, given the following input, please extract the relevant keywords accordingly: |

| Query: “La Mer dupe” |

| Reasoning: “LLMs reasoning output from the previous unconstrained reasoning phase” |

| Response |

| Winona, Proya, The Ordinary, SkinCeuticals, Runbaiyan, Winona, HBN |

*Table 6. Complete query-side and item-side inputs for the direct-embedding dense retrieval model in constructing items set ②-①.*

| Complete Query-Side Input for Constructing Items Set ① |

| <|im_start|>system\n You are a helpful, respectful and honest assistant.<|im_end|> <|im_start|>user\n La Mer dupe <|im_end|> <|im_start|>assistant\n <|im_end|> |

| Complete Query-Side Input for Constructing Items Set ② |

| <|im_start|>system\n You are a helpful, respectful and honest assistant.<|im_end|> <|im_start|>user\n La Mer dupe (Winona, Proya, The Ordinary, SkinCeuticals, Runbaiyan, HBN) <|im_end|> <|im_start|>assistant\n <|im_end|> |

| Complete Item-Side Input |

| <|im_start|>system\n You are a helpful, respectful and honest assistant.<|im_end|> <|im_start|>user\n Winona Soothing & Hydrating Special Care Cream 5g + Repair Essence, Lightweight Sunscreen for Sensitive Skin <|im_end|><|im_start|>assistant\n <|im_end|> |

*Table 7. The relevance assessment process in TaoSR1.*

| Input |

| Human: <cot> |

| Query: La Mer dupe |

| Item: La Mer essence lotion set with firming, moisturizing, soothing, and anti-wrinkle skincare |

| Output |

| 1. Query Intent Analysis |

| Category Information: No explicit category indicated. |

| Attributes: Brand (“La Mer”), Special Attribute (“dupe”/affordable alternative). |

| 2. Item Information Analysis |

| Regarding the brand intent (“La Mer”), the item is explicitly described as “La Mer.” |

| Regarding the “dupe” (affordable alternative) intent, the item description does not convey any indication of being a substitute or lower-cost alternative. Instead, it directly uses the original brand name, which contradicts the “dupe” intent. |

| 3. Category Matching |

| The query expresses no explicit category intent. Therefore, category comparison is not applicable. (No comparison performed) |

| 4. Attribute Matching |

| Brand: The query specifies “La Mer” and the item is labeled as “La Mer” → Fully Matched |

| Special Attribute (Dupe): The query seeks an affordable alternative, but the item does not indicate cost-effectiveness or substitution. Moreover, the use of the official brand name in the title suggests it is an original product, which directly conflicts with the “dupe” requirement. → Complete Mismatch |

| 5. Final Relevance Assessment |

| Rationale: Although the brand attribute is fully matched, the item fails to satisfy the “dupe” intent and explicitly promotes the original brand, which fundamentally contradicts the user’s request for an affordable alternative. According to the relevance tier mapping rules, when category matching is “not applicable” and attribute matching contains a “complete mismatch” the overall relevance level is classified as – Partial Mismatch. |

| |

| Relevance Level: – Partial Mismatch |

*Table 8. Examples of constructed Query-CoT-Item triplets.*

| Query |

| La Mer dupe |

| CoT |

| Winona, Proya, The Ordinary, SkinCeuticals, Runbaiyan, HBN |

| Items |

| 1. Winona Soothing & Hydrating Special Care Cream 5g + Repair Essence, Lightweight Sunscreen for Sensitive Skin |

| 2. Winona Barrier Repair Cream 5g – Hydrating, Redness Relief |

| 3. Proya Ruby Peptide Cream 3.0 – Anti-Wrinkle, Firming, Hydrating & Plumping Skincare |

| 4. HBN Recovery Essence 2.0 – Pre-Serum with Yeast, Hyaluronic Acid & Ceramides for Soothing, Hydration & Skin Barrier Repair |

*Table 9. Complete query-side and item-side inputs for LREM in the cold-start stage.*

| Complete Query-Side Input for LREM |

| <|im_start|>system |

| You are a query assistant for a Chinese e-commerce search platform, and you are adept at understanding various user queries.<|im_end|> |

| <|im_start|>user |

| Please think and reason about the given query to fully understand the product search intent behind it. Consider and reason from multiple angles about various potential related phrases to the query. Do not include the query itself in the potential phrases you infer. Please output in the format: <think> Various Potential Related Phrases Associated with the Query <\think><emb>. |

| Query: La Mer dupe<|im_end|> |

| <|im_start|>assistant |

| <think> Winona, Proya, The Ordinary, SkinCeuticals, Runbaiyan, HBN <\think><emb><|im_end|> |

| Complete Item-Side Input for LREM |

| <|im_start|>system |

| You are a query assistant for a Chinese e-commerce search platform, and you are adept at understanding various products.<|im_end|> |

| <|im_start|>user |

| Item: Winona Soothing & Hydrating Special Care Cream 5g + Repair Essence, Lightweight Sunscreen for Sensitive Skin<|im_end|> |

| <|im_start|>assistant |

| <emb><|im_end|> |

*Table 10. Examples of CoTs sampled from LREM during RL.*

| Query |

| Eliminate Climbing Ivy |

| Sampled CoTs |

| 1. <think>Specialized Herbicide, Plant Inhibitor<\think><emb> |

| 2. <think>Plant Growth Inhibitor, Weed Remover<\think><emb> |

| 3. <think>Plant Pruning Shears, Eco-friendly Remover |

| <\think><emb> |

| 4. <think>Eco-friendly Vine Pruner, Herbicide<\think><emb> |

| 5. <think>Weed Remover, Plant Restoration Activated Carbon<\think><emb> |

| 6. <think>Herbicide, Biopesticide<\think><emb> |

| 7. <think>Hexazinone, Hexazinone Powder, Plant Inhibitor |

| <\think><emb> |

| 8. <think>Weed Remover, Plant Repellent<\think><emb> |

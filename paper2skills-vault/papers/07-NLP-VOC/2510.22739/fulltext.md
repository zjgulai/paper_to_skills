<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2510.22739
     paper_id : 2510.22739
     source   : paper2skills-vault/papers/07-NLP-VOC/2510.22739/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

REVISION:Reflective Intent Mining and Online Reasoning Auxiliary for E-commerce Visual Search System Optimization Yiwen Tang* , Qiuyu Zhao* , Zenghui Sun, Jinsong Lan, Xiaoyong Zhu, Bo Zheng

arXiv:2510.22739v2 [cs.IR] 4 Mar 2026

Alibaba Group, China {tangyiwen.tyw, xiaoyong.z, bozheng}@alibaba-inc.com, {zhaoqiuyu.zqy, zenghui.szh, jinsonglan.ljs}@taobao.com Abstract—In Taobao e-commerce visual search, user behavior analysis reveals a large proportion of no-click requests, suggesting diverse and implicit user intents. These intents are expressed in various forms and are difficult to mine and discover, thereby leading to the limited adaptability and lag in platform strategies.
This greatly restricts users’ ability to express diverse intents and hinders the scalability of the visual search system. This mismatch between user implicit intent expression and system response defines the User–SearchSys Intent Discrepancy. To alleviate the issue, we propose a novel framework REVISION.
This framework integrates offline reasoning mining with online decision-making and execution, enabling adaptive strategies to solve implicit user demands. In the offline stage, we construct a periodic pipeline to mine discrepancies from historical no-click requests. Leveraging large models, we analyze implicit intent factors and infer optimal suggestions by jointly reasoning over query and product metadata. These inferred suggestions serve as actionable insights for refining platform strategies. In the online stage, REVISION-R1-3B, trained on the curated offline data, performs holistic analysis over query images and associated historical products to generate optimization plans and adaptively schedule strategies across the search pipeline. Our framework offers a streamlined paradigm for integrating large models with traditional search systems, enabling end-to-end intelligent optimization across information aggregation and user interaction.
Experimental results demonstrate that our approach improves the efficiency of implicit intent mining from large-scale search logs and significantly reduces the no-click rate.
Index Terms—Large Model, Intent Mining, Visual Search System

I. I NTRODUCTION Compared with text-based product search, e-commerce visual search is affected by a broader range of factors, including shooting angles, image quality, and user behavior patterns.
Taobao’s Pai Li Tao visual search system [14] alleviates these challenges through a modular pipeline integrating various specialized components for diverse user scenarios. Despite the remarkable progress achieved under the paradigm, substantial no-click queries persist in large-scale systems, primarily due to the misinterpretation of user intent and the misalignment between retrieved products and implicit demands.
Empirical evidence indicates that user queries reflect diverse implicit intents beyond purchase-oriented goals. However, conventional search systems exhibit limited intent coverage, constraining user-platform interaction to a single ”vi* Equal contribution.

sual matching” paradigm focused on image-to-image retrieval.
This results in suboptimal user experience and restricts the platform’s functional diversification. We identify the issue as User–SearchSys Intent Discrepancy, attributed to inefficient mining of large-scale search logs and the predefined and constrained strategies in traditional visual search systems. We further outline two key challenges underlying the problem.
Challenge 1. Mining and Discovery from Massive Data.
Existing methods for mining implicit intents from largescale search logs rely on periodic manual annotation of sampled queries to identify potential mismatches. Annotators are guided by predefined tool lists from the visual search pipeline to select suitable tools for intent discrepancies. However, these approaches face critical scalability bottlenecks: manual annotation is inefficient, and the resulting limited data and handcrafted solutions cover only a narrow range of real user intents. Attempts to replace manual annotation with algorithmic mining still depend on manually defined criteria, failing to fundamentally resolve the scalability issue.
Challenge 2. Online Strategy Optimization. Constrained by inefficient manual mining, existing visual search systems primarily leverage user-side information to enhance query features and employ rule-based strategies with discrete classifiers for fixed explicit intent optimization. These methods neglect implicit intent representation in user queries through historical search results, lacking global collaborative optimization. Moreover, they fail to support dynamic strategy adaptation required for handling multiple coexisting implicit intents within queries.
We thus ask: Can we develop a framework that efficiently identifies discrepancies in historical requests and effectively addresses online user queries through Vision language model(VLM) [53]-based understanding and reasoning? In response, we propose REVISION, first leveraging VLMs for two synergistic stages: efficient offline intent mining from historical requests and real-time online strategy optimization through flexible dispatch.
1) Offline Stage. We design a periodically executed pipeline that aggregates similar no-click queries and associates them with actionable platform strategies. Based on Qwen2.5VL72B [15], we extract visual features from queries and their corresponding retrieved products, integrate product attributes (price, title), and generate preliminary optimization sugges-

tions for both user-side and platform-side strategies. Furthermore, leveraging the advanced reasoning capability of Qwen330B-A3B [17], we refine these suggestions with detailed visual information, richer product attributes, and manually defined rules, yielding actionable optimization signals. The offline mining results provide insights for improving existing tools and inspire the design of optimization strategies.
2) Online Stage. We convert offline optimization signals into trainable multiple-choice decision-making tasks. Inspired by Plan-Then-Execute [52], REVISION-R1, built upon Qwen2.5VL-3B [15], is trained using offline mining data and suggestions to reason over real-time user query images and corresponding historical product results, dynamically predicting strategy optimization plans. The predicted actions select and execute appropriate downstream tools sequentially.
With this framework, we evolve the visual search system from simple image-to-image matching into an agentic architecture capable of collaborative multi-tool retrieval with implicit intent understanding. We further design efficient online deployment strategies to seamlessly integrate our framework with the existing visual search systems.
The REVISION framework enables effective intent mining and discovery in the offline stage and performs global agentic optimization among the entire search pipeline in the online stage. In the online A/B test, compared with previous pipeline, the ratio of no-click queries decreases by 13.91% for trigger subset, while the Click-Through Rate (CTR), order volume, and Gross Merchandise Value (GMV) increase by 10.73%, 13.60%, and 10.73%, respectively. Human evaluation further validates the effectiveness of our approach. These results demonstrate the practical potential and effectiveness of the proposed framework in e-commerce visual search systems.
II. R ELATED W ORKS Intent mining and understanding in Search and Recommender Systems. Li J et al. employs Graph and Bert [27] for query classification. Wang J et al. [42] integrates multiple dynamic behavior sequences and a dual-encoder to address the user intent understanding in cross-lingual search.
J. Niu et al. [1] mines user preferences, user intentions, and potential relationships between items, and models them via contrastive learning to optimize user intent understanding in recommendation. L. Sang et al. [3] also performs intent optimization in Contrastive Learning (CL)-based recommender systems, leveraging a Heterogeneous Graph to incorporate intent information into the recommendation process. Sun K et al. [4] primarily address the problem of intent recognition in multi-modal scenarios by exploiting intra-video and cross-video context interactions to enhance intent modeling, while leveraging a cross-video bank for retrieval to mitigate error accumulation. New intent discovery leverages multitask pre-training and contrastive learning with clustering [48].
S2TM [47] integrate social and item semantics to handle sparsity and long-tailed distributions. AttnMix [46] captures short-term intent via order-invariant session representations,

while IOCLRec [50] disentangles mixed intents through multigranularity contrastive learning and dynamic segmentation.
MIND [45] infers intents using multimodal signals and LLMs by identifying shared attributes among co-purchased items.
VLM and LLM Applications in Search Systems. Several works [10], [11] employ LLMs to refine user queries via rewriting, improving search effectiveness. LAPS [12] enhances conversational search personalization by modeling user preferences from past interactions. Pasa [8] utilizes dual LLM agents for query expansion and relevance evaluation in academic search. BASES [13] simulates user profiles and interactions using LLM-based agents tailored to specific personas, enabling realistic behavior modeling. Baidu AI Search [7], described by Yuchen Li et al., integrates multiple LLM agents in a collaborative framework to respond to user queries effectively.
OSrCIR [49] uses reflective chain-of-thought to infer modification intents before retrieval, mitigating semantic drift. Chainof-Intent [51] combines Hidden Markov Models(HMMs) with LLMs to model intent transitions and synthesize dialogues.
Most related work differs fundamentally in task formulation, input/output specifications, and evaluation environment, focusing on behavior modeling or item ranking rather than system-level planning, targeting query rewrites instead of executable tool sequences, or using static datasets lacking multimodal result-aware feedback. Given these incompatible task paradigms, direct comparison remains impractical. REVISION provides a unified reasoning framework that achieves greater data efficiency without large-scale pre-training or explicit user feedback, and effectively learns to infer what users need.
III. M ETHOD A. Motivation and Overview Figure 1 illustrates the REVISION paradigm, comprising asynchronous offline and online stages. In the offline stage, we leverage Qwen2.5VL and Qwen3 to jointly analyze historical query images and the corresponding retrieval products, extracting a series of actionable optimization signals. Based on these signals, we perform hierarchical clustering via phrase mapping and vector similarity matching [22]. This yields multiple coarse-grained clusters, each further divided into fine-grained sub-clusters. These fine-grained groups indicate specific downstream tools, forming a curated tool list. During online model training, we utilize the offline-generated data (a query paired with retrieval products, thinking process and ordered tool lists) as training samples. A VLM is trained to reason and predict appropriate tool sequences. In the online stage, the trained model assigns executable tools to different search system stages based on real-time queries and historical retrieval products, optimizing outputs to mitigate ambiguity in request intent. Required information for each tool is obtained from the query and products.
B. Offline Stage The offline stage automatically mines users’ latent intents from large-scale non-click search logs through multimodal

VLM Prompt

LLM Prompt

This is a E-commerce search request, you need perform a thorough visual analysis of <query>,<product1>..<productN>, Extract textual information, Compare the query image with candidate products, Develop targeted retrieval or recommendation strategies, leverage user preference and intent signals, you need 1.Provide a structured reasoning analysis…..
2.Derive targeted interaction strategies…..

You are an expert in the field of e-commerce.
Follow these steps:1 <user intent analysis rules>, <results analysis rules><diff compare standards>,

Reasoning possible no-click factors: Visual Feature Discrepancy:xxx, Functional Requirement Gap:xxx, Quality Expectation Mismatch:xxx, Usage Scenario Incompatibility:xx Please follow the output format as :

Suggestions Search Logs

Vector Matching

LLM

Visual Descriptions multi-dimensional info

Visual Search System

Query Parser

Ranker

Recall

Final Results

“Provide the tool indices that best match the request intent, ordered by execution sequence.”

xL

Phrase Mapping

1.Coarse Level:
Price-Related 2.Fine-Grained Level:
Price Segmentation

Fine-Grained Summarization

Platform Tool List (1).Textual augment search

（1） Thinking （2） Process （5）

FFN

Text Tokenizer

(a) Offline Stage

Self-Attn.

…

Retrieval Products

4.Price-based segmentation → Display results by price ranges such as 50–150 and 150–300.

Assign & execute

Visual Encoder

History Requests

Thinking

VLM

Retrieval Products User Query

Hierarchical Clustering

1. Search condition refinement → Combine the features "black" and "dress" with the OCRextracted brand name "Blendy".

Rule-based Heuristics

Query

<Output Structure>

Optimization Signals

(2).Product display metadata (N).Search result summarization

（6）

(b) Online Stage - REVISION-R1 Fig. 1. (a) illustrates the offline stage, where VLMs and LLMs are employed for reasoning, and hierarchical clustering is applied to organize the inferred results. The fine-grained clustering outcomes guide the expansion of the tool list. (b) shows the online stage, where the trained REVISION-R1-3B assigns executable tools to the visual search pipeline based on the query image, enabling targeted interventions across different stages of the process.

understanding of historical queries and retrieved products, followed by structured reasoning over extracted visual signals.
a) Analysis Steps of Large Models: The sampled data comprises a query image paired with multiple historical products, including corresponding titles, prices, and similarity scores to the query. We first input data into Qwen2.5VL-72B to extract visual information from the query and products.
Without any manual constraints, the model then generates preliminary optimization suggestions, typically related to direct visual or semantic factors such as category, appearance, and price differences. To enhance reasoning beyond visual understanding, we further employ Qwen3-30B-A3B as the textbased reasoning model. Beyond the outputs from Qwen2.5VL, we provide Qwen3 with abundant product metadata (e.g., origin, package size) and domain expert rules encoding visual search system biases. The former includes attributes such as origin and package size, and the latter introduces the inductive bias of the visual search system, encoded by domain experts in the form of rules. For instance, if the retrieval results show large price discrepancies concentrated in low ranges, it would be suggested to supplement with similar high-end products. Integrating information from all four sources, Qwen3 generates a sequence of optimization signals in the format of "Action -> Info", serving as valuable training data for downstream tools. Action denotes the specific operation such as search condition refinement or price range segmentation; Info represents the information required to execute the action, e.g., visual features like “black” and OCR-extracted brand text “Blendy” for search conditions.
b) Hierarchical Clustering Algorithm: Let A = {ai }N i=1 be short textual “actions” extracted from optimization signals. We build a two-level hierarchy over a predefined ontology of executable main categories C, each with subcategories

Sc . Each ai is first preprocessed by removing punctuation and index markers, then tokenized with a word segmenter.
For any label ℓ (either c ∈ C or s ∈ Sc ), we maintain a keyword synonym list Kℓ used for robust lexical matching.
For action a and label ℓ, we compute a synonym-overlap score ssyn (a, ℓ) as the fraction of tokens in a that match Kℓ under substring equivalence, and a semantic score ssem (a, ℓ) via cosine similarity between sentence-transformer embeddings of a and the concatenation of Kℓ . We then form a combination sα (a, ℓ) = α ssyn (a, ℓ) + (1 − α) ssem (a, ℓ), with α = 0.7 for main-category assignment and α = 0.6 for subcategory assignment.
Level 1. Each ai is assigned to arg max s0.7 (ai , c)
c∈C

if the maximal score exceeds a confidence threshold τ1 = 0.40; otherwise it is marked unassigned. For unassigned items, we compute pairwise similarities and run DBSCAN [30] (Density-based clustering algorithm) on the precomputed distance matrix dij = max{0, 1 − cos(ai , aj )} with ε = 0.5 and min samples = 2, yielding auxiliary semantic clusters.
Level 2. For main category c, the assigned actions are further partitioned over Sc using s0.6 (a, s) and a relaxed threshold τ2 = 0.35; items below the threshold are routed to an “other” bucket under c. Finally, the algorithm returns a hierarchical mapping (main category → subcategories → actions).
c) Component-based Tools: Based on the executable signals from clustering, we extend and upgrade the existing tools and modularize them as standardized components for online scheduling integration(e.g. Display metadata adjustment component, Search result summarization component, External

Textual search component and Target products labeling component). These components enable flexible composition via graphical configuration. The system dynamically constructs Directed Acyclic Graphs [23] for chain execution based on the planning list through automated configuration generation.
C. Online Stage Trained on offline-mined data and suggestions, the online model captures real-time user intent and performs agentic tools invocation to optimize search results.
a) Training Data Collection: We replace them with clustered sub-categories, transforming the original format from Action → Info to Sub-category → Info.
The Sub-category pairs are extracted and mapped to ordered numeric labels, such as (1), based on their positions in the tool list. As a result, the training data follows the structure:
Query + Retrieval Results → (1) (3) (5),

multiple sub-reward components. We then calculate the groupnormalized advantage:
Ai =

ri − mean{r1 , . . . , rN } , std{r1 , . . . , rN } + δ

where δ is a small constant added for numerical stability. Our reward function consists of two components:
• Format reward (rformat ): This encourages the model to produce structured reasoning sequences using specific format tokens such as <think>...</think>.
A score of 1.0 is assigned if the <think> tag is used correctly; otherwise, the score is 0.
• Answer accuracy reward (rans ): This measures the correctness of the final answer. A reward of 1.0 is assigned when either the predicted tool types or their execution order matches the ground truth, 2.0 when both match, and 0 otherwise.
IV. E XPERIMENTS A. Offline and Online Setups

which significantly improves controllability and reduces response time. We observe that incorporating reasoning traces into the training data improves prediction accuracy. To achieve this, we leverage Qwen3 to rephrase and compress the offline thinking process. The output format is structured as:
Thinking Process + (1) (3) (5). At the tool level, the required execution information Info is extracted from the query and the retrieval products.
b) REVISION-R1 Training: Stage 1: Supervised Fine-Tuning. We perform supervised fine-tuning using Qwen2.5VL-3B. The visual input V consists of a query image Iq and up to K = 12 retrieval product images {I1 , I2 , . . . , IK }. The textual input T includes: (1) Product metadata: structured strings, e.g., Product i: price = 19, title = stylish black off-shoulder dress, quantity = 1. (2) Tool component details:
represented by Index, Title, and Description. The description specifies the component functionality, along with its expected input and output formats. The model is trained to reason and predict the tool components used for optimization, along with their execution order. The output sequence is denoted as Y = (y1 , y2 , . . . , yT ). We adopt the standard next-token prediction and optimize the cross-entropy loss:
LCE = −

T X

log P (yt | y<t , V, T ).

t=1

Stage 2: Reinforcement Learning. To enhance the reasoning capability of the REVISION-R1, we build upon the existing GRPO [24] algorithm and sample the hard training data used during the SFT stage [25] to guide the model to generate reasoning sequences autonomously. We formulate the model as a policy function πθ . During training, for each query–retrieval list pair (q, R), we sample N candidate outputs {o1 , . . . , oN } from the current policy πθ . For each candidate oi , we compute a task-specific reward ri = R(q, oi ), which is composed of

a) Offline Mining pipeline details: Our offline mining stage is conducted on millions of unique image queries sampled weekly from historical no-click requests. For Qwen2.5VL, we limit the input product images to 12 per query.
Qwen2.5VL-72B is deployed on 4 PPU [43] (Alibabadeveloped accelerator) GPUs. For Qwen3-30B-A3B (deployed on 2 PPU GPUs), to reduce interference from irrelevant information, we rank the product metadata by importance and select the top 10 elements as input. The VLM-LLM mining pipeline is powered by 60 PPU [43] GPUs. This pipeline can process and analyze millions of query data within a single day.
In hierarchical clustering, the main categories C and corresponding subcategories Sc are automatically extracted by Qwen3 from the analysis step, not manually predefined. The clustering is scheduled weekly as part of the regular data processing pipeline. We ensure that the main categories, subcategories, and keyword list Kℓ are sufficiently large to provide broad coverage. For large-scale and continuously updated data, incremental clustering efficiently incorporates newly generated data without reprocessing the entire dataset.
b) Offline to Online Data Management: In the offline stage, We target no-click queries—image uploads without clicks within 30 seconds. After filtering bot traffic and lowquality images via CNN classifiers, we collect 8–12 million such queries daily from Taobao. The offline pipeline is orchestrated weekly via Airflow, with all intermediate artifacts stored in versioned partitioned Hive tables for traceability.
An Image Query Caching Module is deployed to reduce redundant processing of semantically similar image queries by reusing historical reasoning results. It combines a global image feature extractor with a vector database to detect nearduplicate queries (similarity >0.85). Highly similar queries directly reuse cached global decisions while executing only downstream actions. Over time, this cache covers about 30% of queries, achieving over 93% accuracy and significantly reducing computation costs.

Case A Query Image

Traditional Results

With REVISION

Case B

Query Image

Traditional Results

With REVISION

Reflecting & Planning

Reflecting & Planning

“Conditional filter”:
“Material”

“Textual search”:
“布拉⽒酵⺟菌散”

“preference ranking”:
“999 Gold”

“Textual vs visual priority”:
“Textual Search” “Summarization”:
“Default:999 gold,” “Product title adjustment”:
“highlight text query”

“External API”:
“Real-time Gold Price”

“Metadata adjustment”:
“highlight unit price”

Case C

Case D Reflecting & Planning Reflecting & Planning “Fine-Grained classifier”:
“query-doc Ball valve”

“Product Instruction”:
“Installation, buying”

“Standards”:
“Size, Material”

“Easily confused categories”:
“Lotion Toner Refill bottle”

“User disambiguation”:
“Refill bottle”

“Product filtering”:
“User demand filtering”

Fig. 2. Online Comparisons of interface between traditional results and search system with REVISION. Some commercial and sensitive information of Query images and results is masked.Case A: REVISION identifies the user’s implicit need for medication information and prioritizes explanatory results over purely visual matches. Case B: REVISION restructures jewelry results using material- and price-aware signals to reduce cognitive load caused by visually similar but attribute-divergent products. Case C: REVISION highlights key specifications and functional details for standardized components, minimizing users’ verification effort. Case D: REVISION resolves ambiguity in visually unclear queries through preference-guided interaction.

c) Online deployment details: Our serving system is adapted from SGLang [21], deployed on a cluster of 45 PPU GPUs. We adopt the Prefill–Decode decoupled architecture [16] and design a real-time scheduling mechanism that monitors task load at the millisecond level, dynamically routing requests to low-load workers for maximum throughput.
After generating the plan list, the system concurrently fetches required external data alongside the main search pipeline to minimize end-to-end latency. The additional latency is primarily due to the REVISION-R1 model in the triggered subset, contributing 95–100ms to TP99 [44] (Latency at 99th percentile) and 45ms to average response time.
d) Model Training Details: In the Supervised FineTuning (SFT) stage, we collect 4.3M training samples by sampling hundreds of thousands from each main category, with thinking processes limited to 256 tokens. For the Reinforcement Learning (RL) stage, we identify 680K difficult samples as training data where SFT predictions mismatch ground truth on the same 4.3M dataset.
B. Evaluation and Ablation Study a) Offline Pipeline Evaluation: We randomly sampled 10,000 online queries that triggered optimization strategies and recruited 10 assessors with search ranking expertise. For each query, assessors reviewed the query image with professional annotations of ground-truth intent descriptions, then examined the top-1 and top-4 retrieved products from both the baseline system (traditional PaiLiTao pipeline) and the test system (our REVISION offline mining pipeline), including product images, titles, and prices. Effectiveness is measured by Top-1/Top-4 Relevance: assessors simulated user behavior to judge whether

at least one highly relevant product aligned with the groundtruth intent appeared in the final ranked results. As shown in Table I, REVISION’s offline mining pipeline significantly outperformed the baseline, improving search quality by 37.99% in top-1 results and 34.21% in top-4 results. The inter-assessor agreement reached 91%, indicating high consistency.
b) REVISION-R1 Evaluation: We designate the aforementioned annotated data as the test set, with ground truth from the offline stage comprising a reasoning process and tool indices in the form of labels such as (1), (3). To measure the effectiveness of online intent capture, model performance is evaluated along two key dimensions: First, thinking content accuracy (reasoning quality), assessed via LLM-as-a-judge (Qwen3) and natural language generation metrics (CIDEr [31], BLEU-4 [33], METEOR [34], and ROUGE [32], standard text similarity metrics) to assess the validity of the reasoning content and its alignment with human-corrected ground truth.
Second, answer accuracy (tool-calling correctness), which is further divided into two aspects: (1) exact/partial tool selection match, and (2) tool sequence order match. For baseline comparison, we select GPT-4o [18], Seed-1.5VL [20], and Gemini 2.5 Pro [28], using the same set of prompts. As shown in Table IV, our REVISION-R1 significantly outperforms other models and demonstrates that both the SFT and RL training stages are indispensable. In the thinking content evaluation, REVISION-R1 outperforms OmniSearch [29] by 13.6% on the Qwen3 metric. In the answer accuracy evaluation, REVISION-R1 achieves 16.4% and 18.7% higher tool matching and order matching rates, respectively, compared with OmniSearch, which is a GPT-4V–based adaptive retrieval planning agent.

TABLE I O FFLINE S TAGE E VALUATION

TABLE II O NLINE PERFORMANCE OF A/B TESTS

TABLE III A BLATION STUDY ON M INING F REQUENCY

Metric

Base Result

Test Result

Diff

Online Metric

CTR

CVR

Order Count

GMV

No-click Ratio

Top1 Relevance Top4 Relevance

28.57% 36.38%

66.56% 70.59%

+37.99% +34.21%

Trigger Subset General Set

+10.73% +0.44%

+8.82% +0.23%

+13.60% +0.58%

+10.73% +1.04%

-13.91% -0.92%

TABLE IV B ENCHMARK RESULTS ON INTENT REFLECTION , WITH THINKING CONTENT ACCURACY (Q WEN 3, CIDE R , BLEU-4, METEOR, ROUGE)
AND ANSWER ACCURACY (T OOL M ATCH AND O RDER M ATCH )
Model

Qwen3 CIDEr BLEU-4 METEOR ROUGE

Tool Order Match Match

GPT-4o [18]
Seed-1.5VL [20]
Gemini 2.5 Pro [28]

45.1 36.0 40.9

56.8 49.1 61.5

26.5 13.8 21.9

16.7 18.2 11.4

32.8 24.7 26.9

45.3 42.6 50.1

24.7 27.8 30.3

OmniSearch [29]
REVISION-R1 -SFT -RL

54.6 67.0 45.3 53.9

70.2 91.7 70.1 83.2

36.0 53.1 37.9 46.0

19.6 30.3 26.0 28.5

51.9 61.9 42.5 59.6

58.8 75.2 55.9 61.0

39.4 58.1 37.6 43.4

TABLE V H YPERPARAMETER S ENSITIVITY A NALYSIS IN O FFLINE M INING Configuration — Confidence Thresholds (τ )
Weighting Factors (α)

Ours High Precision High Recall Syntax-Dominant Semantic-Dominant

Hyperparameters α1 /α2 τ1 /τ2 0.7 / 0.6 0.40 / 0.35 0.7 / 0.6 0.55 / 0.50 0.7 / 0.6 0.30 / 0.25 0.9 / 0.8 0.40 / 0.35 0.4 / 0.3 0.40 / 0.35

Relevance (%)
Top-1 Top-4 66.56 70.59 63.12 67.85 58.45 62.90 60.23 64.15 61.88 65.40

TABLE VI A NALYSIS OF O NLINE R EINFORCEMENT L EARNING T RAINING .
Configuration Ours Group Size Small Group Large Group Reward Design w/o Format Reward Binary Answer Reward

Hyperparameters Group Size (G)
Reward Setup 8 rf ormat + rans

Performance (%)
Thinking Accuracy Tool Match 67.0 75.2

4 12

rf ormat + rans rf ormat + rans

62.4 67.3

70.8 75.0

8 8

rans only rf ormat + rbinary

43.8 61.1

48.2 68.4

TABLE VII A BLATION S TUDY ON C LUSTERING D ISTANCE M ETRICS TopN

Ours (Cosine)

Variant A (L2 )

Variant B (L1 )

Top-1 Top-4

66.56 70.59

62.84 67.20

59.15 64.05

c) Online A/B Test: To evaluate the real-world performancek, REVISION was deployed in Taobao Visual Search for an online A/B test. We allocated 10% of user traffic to each strategy to rigorously assess its effectiveness and stability. Optimized results displayed only for queries with low relevance scores, with limited daily cue frequency per user to ensure smooth experience. As shown in Table II, the test was conducted over seven days. The triggered subset achieved notable gains over the baseline, while the overall experiment group showed consistent positive improvements.
Approximately 17% of total traffic fell within the triggered subset, and all results were statistically significant (p<0.05).
d) Ablation Study of Offline Mining Frequency: As shown in Table III, higher offline mining frequency (e.g., T+4) improves cache hit rate and query freshness by enriching intent cache with recent trends. However, it captures volatile, noise-driven signals rather than stable user behavior patterns, introducing unstable new intents that interfere with product recommendation and negatively impact user clicks and purchases, while incurring additional GPU costs. In mature search systems, user behavior evolves slowly, and valuable new intents require sufficient signal accumulation, clustering, manual validation, and A/B testing before deployment. Moderately

T+4 T+8 (Ours)
T+16

Hit Rate

CTR

CVR

GMV

Non-click Ratio

+GPUs

31.1% 30.4% 29.8%

+0.41% +0.44% +0.40%

+0.19% +0.23% +0.22%

+0.97% +1.04% +1.01%

-0.86% -0.92% -0.90%

35 0 0

reducing mining frequency (e.g., T+8) captures stable intent trends that better align with genuine user intents, generating more stable ranking results that effectively encourage user engagement and conversion, while using only online elastic resources without extra GPU cost, thereby achieving a better benefit–cost trade-off. Further reduction (e.g., T+16) yields no significant gains over T+8.
e) Ablation Study of Data Engineering Aspects: We conduct ablation studies on critical data engineering components: (1) offline mining hyperparameters, (2) online training configurations, and (3) DBSCAN similarity metrics.
Offline Mining Hyperparameters. Table V shows our configuration (α = 0.7/0.6, τ = 0.40/0.35) achieves optimal balance. High τ reduces recall by filtering signals; low τ introduces noise. Balanced α combines lexical precision and semantic generalization, syntax-dominant settings miss LLM expression diversity, while semantic-dominant settings suffer drift. We use 12 input images balancing API cost and quality.
Online Training Hyperparameters. Table VI shows N = 8 provides optimal performance-efficiency trade-off for GRPO group size. Smaller N increases variance (-4.6% thinking accuracy at N = 4); larger N yields marginal gains (+0.3% at N = 12) with high memory cost. Format reward rf ormat is essential, removal disrupts reasoning chains (-27.0% tool match). Graded answer rewards (0/1/2) outperform binary rewards by encouraging full correctness. Learning rate 1 × 10−6 balances performance and stability.
DBSCAN Similarity Metrics. Table VII shows Cosine distance significantly outperforms Euclidean (L2 ) and Manhattan (L1 ). Cosine captures angular similarity in Sentence-BERT embeddings, while magnitude-sensitive metrics fail to cluster semantically similar but magnitude-variant signals, reducing relevance (e.g., -3.72% top-1 for Euclidean).
V. C ONCLUSION The paper introduces the implicit User–SearchSys Intent Discrepancy problem and proposes REVISION, a VLM-based agentic search framework integrating offline intent mining with online reasoning. Offline, REVISION identifies and clusters intent discrepancies from large-scale search logs for strategy design. Online, it autonomously orchestrates tools for agentic optimization. A/B testing validates its effectiveness in alleviating intent discrepancies and enhancing user experience. Beyond e-commerce, REVISION offers transferable insights for integrating large language models (LLMs) into search systems by leveraging data-driven workflows over handcrafted rules. It demonstrates that no-click interactions yield valuable signals when interpreted by reasoning models, with implications extending to recommendation and conversational systems. Future work will unify offline traces with online fine-grained signals as memory and perception, propelling REVISION into a selfevolving agentic search system.

R EFERENCES [1] J. Niu, W. Zhou, F. Luo, Y. Zhang, J. Zeng and J. Wen, ”Intent-Guided Bilateral Long and Short-Term Information Mining With Contrastive Learning for Sequential Recommendation,” in IEEE Transactions on Services Computing, vol. 18, no. 1, pp. 212-225, Jan.-Feb. 2025.
[2] Wang J, Zhao Q, Xi Y. Cross-lingual Search Intent Understanding Framework Based on Multi-modal User Behavior. Annals of Applied Sciences. 2024 Nov 8;5(1).
[3] L. Sang, Y. Wang, Y. Zhang, Y. Zhang and X. Wu, ”Intent-Guided Heterogeneous Graph Contrastive Learning for Recommendation,” in IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 4, pp. 1915-1929, April 2025, doi: 10.1109/TKDE.2025.3536096.
[4] Sun K, Xie Z, Ye M, Zhang H. Contextual augmented global contrast for multimodal intent recognition. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024 (pp.
26963-26973).
[5] Yang A, Li A, Yang B, Zhang B, Hui B, Zheng B, Yu B, Gao C, Huang C, Lv C, Zheng C. Qwen3 technical report. arXiv preprint arXiv:2505.09388. 2025 May 14.
[6] Fang J, Gao S, Ren P, Chen X, Verberne S, Ren Z. A multi-agent conversational recommender system. arXiv preprint arXiv:2402.01135.
2024 Feb 2.
[7] Li Y, Cai H, Kong R, Chen X, Chen J, Yang J, Zhang H, Li J, Wu J, Chen Y, Qu C. Towards AI Search Paradigm. arXiv preprint arXiv:2506.17188. 2025 Jun 20.
[8] He Y, Huang G, Feng P, Lin Y, Zhang Y, Li H. Pasa: An llm agent for comprehensive academic paper search. arXiv preprint arXiv:2501.10120.
2025 Jan 17.
[9] He Y, Liu X, Zhang A, Ma Y, Chua TS. Llm2rec: Large language models are powerful embedding models for sequential recommendation.
InProceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2 2025 Aug 3 (pp. 896-907).
[10] Wang L, Yang N, Wei F. Query2doc: Query expansion with large language models. arXiv preprint arXiv:2303.07678. 2023 Mar 14.
[11] Liu J, Mozafari B. Query rewriting via large language models. arXiv preprint arXiv:2403.09060. 2024 Mar 14.
[12] Joko H, Chatterjee S, Ramsay A, De Vries AP, Dalton J, Hasibi F.
Doing personal laps: Llm-augmented dialogue construction for personalized multi-session conversational search. InProceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval 2024 Jul 10 (pp. 796-806).
[13] Ren R, Qiu P, Qu Y, Liu J, Zhao WX, Wu H, Wen JR, Wang H. Bases:
Large-scale web search user simulation with large language model based agents. arXiv preprint arXiv:2402.17505. 2024 Feb 27.
[14] Zhang Y, Pan P, Zheng Y, Zhao K, Zhang Y, Ren X, Jin R. Visual search at alibaba. InProceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining 2018 Jul 19 (pp.
993-1001).
[15] Bai S, Chen K, Liu X, Wang J, Ge W, Song S, Dang K, Wang P, Wang S, Tang J, Zhong H. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. 2025 Feb 19.
[16] Zhong Y, Liu S, Chen J, Hu J, Zhu Y, Liu X, Jin X, Zhang H.
DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. In18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24) 2024 (pp. 193-210).
[17] Yang A, Li A, Yang B, Zhang B, Hui B, Zheng B, Yu B, Gao C, Huang C, Lv C, Zheng C. Qwen3 technical report. arXiv preprint arXiv:2505.09388. 2025 May 14.
[18] Hurst A, Lerer A, Goucher AP, Perelman A, Ramesh A, Clark A, Ostrow AJ, Welihinda A, Hayes A, Radford A. Gpt-4o system card.
arXiv preprint arXiv:2410.21276. 2024 Oct 25.
[19] Bai S, Chen K, Liu X, Wang J, Ge W, Song S, Dang K, Wang P, Wang S, Tang J, Zhong H. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923. 2025 Feb 19.
[20] Guo D, Wu F, Zhu F, Leng F, Shi G, Chen H, Fan H, Wang J, Jiang J, Wang J, Chen J. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062. 2025 May 11.
[21] Zheng L, Yin L, Xie Z, Sun CL, Huang J, Yu CH, Cao S, Kozyrakis C, Stoica I, Gonzalez JE, Barrett C. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems. 2024 Dec 16;37:62557-83.
[22] Reimers N, Gurevych I. Sentence-bert: Sentence embeddings using siamese bert-networks[J]. arXiv preprint arXiv:1908.10084, 2019.

[23] Lipsky A M, Greenland S. Causal directed acyclic graphs[J]. Jama, 2022, 327(11): 1083-1084.
[24] Shao Z, Wang P, Zhu Q, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models[J]. arXiv preprint arXiv:2402.03300, 2024.
[25] Yu Q, Zhang Z, Zhu R, et al. Dapo: An open-source llm reinforcement learning system at scale[J]. arXiv preprint arXiv:2503.14476, 2025.
[26] Li J, Zeng W, Cheng S, Ma Y, Tang J, Wang S, Yin D. Graph enhanced bert for query understanding. InProceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval 2023 Jul 19 (pp. 3315-3319).
[27] Devlin J, Chang M-W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
arXiv preprint arXiv:1810.04805 [cs.CL]. 2019. Available from:
https://arxiv.org/abs/1810.04805 [28] Comanici G, Bieber E, Schaekermann M, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities[J]. arXiv preprint arXiv:2507.06261, 2025.
[29] Li Y, Li Y, Wang X, et al. Benchmarking multimodal retrieval augmented generation with dynamic vqa dataset and self-adaptive planning agent[J].
arXiv preprint arXiv:2411.02937, 2024.
[30] Wang D, Lu X, Rinaldo A. Dbscan: Optimal rates for density-based cluster estimation. Journal of machine learning research. 2019;20(170):1-50.
[31] Vedantam R, Lawrence Zitnick C, Parikh D. Cider: Consensus-based image description evaluation. InProceedings of the IEEE conference on computer vision and pattern recognition 2015 (pp. 4566-4575).
[32] Lin CY. Rouge: A package for automatic evaluation of summaries.
InText summarization branches out 2004 Jul (pp. 74-81).
[33] Papineni K, Roukos S, Ward T, Zhu WJ. Bleu: a method for automatic evaluation of machine translation. InProceedings of the 40th annual meeting of the Association for Computational Linguistics 2002 Jul (pp.
311-318).
[34] Banerjee S, Lavie A. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. InProceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization 2005 Jun (pp. 65-72).
[35] Xu B, Wang W, Shi H, Ding W, Jing H, Fang T, Bai J, Liu X, Yu C, Li Z, Luo C. Mind: Multimodal shopping intention distillation from large vision-language models for e-commerce purchase understanding. arXiv preprint arXiv:2406.10701. 2024 Jun 15.
[36] Zhang P, Guo J, Li C, Xie Y, Kim JB, Zhang Y, Xie X, Wang H, Kim S. Efficiently leveraging multi-level user intent for session-based recommendation via atten-mixer network. InProceedings of the sixteenth ACM international conference on web search and data mining 2023 Feb 27 (pp. 168-176).
[37] Wang W, Cheng X, Liu Z, Lin Y, Shen Y, Hu B, Zhang Z, Zeng X, Zhou J, Gu J, Luo M. Intent mining: A social and semantic enhanced topic model for operation-friendly digital marketing. In2022 IEEE 38th International Conference on Data Engineering (ICDE) 2022 May 9 (pp.
3254-3267). IEEE.
[38] Zhang Y, Zhang H, Zhan LM, Wu XM, Lam A. New intent discovery with pre-training and contrastive learning. arXiv preprint arXiv:2205.12914. 2022 May 25.
[39] Tang Y, Zhang J, Qin X, et al. Reason-before-retrieve: One-stage reflective chain-of-thoughts for training-free zero-shot composed image retrieval[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 14400-14410.
[40] Wang W, Ma J, Zhang Y, et al. Intent Oriented Contrastive Learning for Sequential Recommendation[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(12): 12748-12756.
[41] Liu J, Tan Y K, Fu B, et al. From Intents to Conversations: Generating Intent-Driven Dialogues with Contrastive Learning for Multi-Turn Classification[C]//Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 2025: 1861-1871.
[42] Wang J, Zhao Q, Xi Y. Cross-lingual Search Intent Understanding Framework Based on Multi-modal User Behavior. Annals of Applied Sciences. 2024 Nov 8;5(1).
[43] EE Times. ”Alibaba Unveils Own AI Chip, Mounting Direct Challenge to Nvidia,” EE Times, Sep. 2025. [Online]. Available:
https://www.eetimes.com/alibaba-unveils-own-ai-chip-mounting-directchallenge-to-nvidia/ [44] Dean J, Barroso L A. The Tail at Scale. Communications of the ACM.
2013 Feb;56(2):74-80.

[45] Xu B, Wang W, Shi H, Ding W, Jing H, Fang T, Bai J, Liu X, Yu C, Li Z, Luo C. Mind: Multimodal shopping intention distillation from large vision-language models for e-commerce purchase understanding. arXiv preprint arXiv:2406.10701. 2024 Jun 15.
[46] Zhang P, Guo J, Li C, Xie Y, Kim JB, Zhang Y, Xie X, Wang H, Kim S. Efficiently leveraging multi-level user intent for session-based recommendation via atten-mixer network. InProceedings of the sixteenth ACM international conference on web search and data mining 2023 Feb 27 (pp. 168-176).
[47] Wang W, Cheng X, Liu Z, Lin Y, Shen Y, Hu B, Zhang Z, Zeng X, Zhou J, Gu J, Luo M. Intent mining: A social and semantic enhanced topic model for operation-friendly digital marketing. In2022 IEEE 38th International Conference on Data Engineering (ICDE) 2022 May 9 (pp.
3254-3267). IEEE.
[48] Zhang Y, Zhang H, Zhan LM, Wu XM, Lam A. New intent discovery with pre-training and contrastive learning. arXiv preprint arXiv:2205.12914. 2022 May 25.
[49] Tang Y, Zhang J, Qin X, et al. Reason-before-retrieve: One-stage reflective chain-of-thoughts for training-free zero-shot composed image retrieval[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 14400-14410.
[50] Wang W, Ma J, Zhang Y, et al. Intent Oriented Contrastive Learning for Sequential Recommendation[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(12): 12748-12756.
[51] Liu J, Tan Y K, Fu B, et al. From Intents to Conversations: Generating Intent-Driven Dialogues with Contrastive Learning for Multi-Turn Classification[C]//Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 2025: 1861-1871.
[52] He G, Demartini G, Gadiraju U. Plan-then-execute: An empirical study of user trust and team performance when using llm agents as a daily assistant. InProceedings of the 2025 CHI Conference on Human Factors in Computing Systems 2025 Apr 26 (pp. 1-22).
[53] Minaee S, Mikolov T, Nikzad N, et al. Large language models: A survey[J]. arXiv preprint arXiv:2402.06196, 2024.


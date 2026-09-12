<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2601.16492
     paper_id : 2601.16492
     source   : https://arxiv.org/html/2601.16492v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# LLM-based Semantic Search for Conversational Queries in E-commerceThanks: This is a preprint under review.

CCS: Information systems Information retrieval query processingCCS: Information systems Retrieval models and rankingCCS: Computing methodologies Natural language processing
Emad Siddiqui Affiliation: University of Arizona, Tucson, AZ, USA email: emadsiddiqui@arizona.edu , Venkatesh Terikuti Affiliation: University of Arizona, Tucson, AZ, USA email: venkatesht@arizona.edu and Xuan Lu Affiliation: University of Arizona, Tucson, AZ, USA email: luxuan@arizona.edu

###### Abstract.

Conversational user queries are increasingly challenging traditional e-commerce platforms, whose search systems are typically optimized for keyword-based queries. We present an LLM-based semantic search framework that effectively captures user intent from conversational queries by combining domain-specific embeddings with structured filters. To address the challenge of limited labeled data, we generate synthetic data using LLMs to guide the fine-tuning of two models: an embedding model that positions semantically similar products close together in the representation space, and a generative model for converting natural language queries into structured constraints. By combining similarity-based retrieval with constraint-based filtering, our framework achieves strong precision and recall across various settings compared to baseline approaches on a real-world dataset.

###### Keywords:

semantic search, LLM, sentence transformer, synthetic data, e-commerce

## 1. Introduction

Web users have been trained to search using keywords for decades, particularly since the rise of Google in the early 2000s. However, this is beginning to change with the emergence of large language models (LLMs) and their integration into many aspects of daily life (Reid, 2025)(Bloomreach, 2025). As users grow accustomed to interacting with LLMs through natural language, their search behavior may shift accordingly—they are increasingly phrasing their queries in natural, conversational language, similar to how they communicate with LLMs. This shift is already evident in recent empirical findings: a 2025 survey shows that 54% of consumers have shifted toward conversational search habits, favoring natural language over keywords (Bloomreach, 2025). E-commerce platforms, which rely heavily on search functionality, are increasingly challenged by this emerging trend. For example, instead of searching for “iPhone 14 Pro case” and manually applying filters such as review scores, a user may prefer to enter a query that includes all their requirements, such as “Find me a iPhone 14 Pro case in green color priced between $15 and $20 with strong feedback.” However, as e-commerce search systems are typically optimized for keyword-based queries, they often struggle to capture user intent. Research shows that 69% of consumers use the search bar immediately upon visiting an e-commerce site, yet 80% abandon the platform due to unsatisfactory search results (Kashyap, 2023).

To effectively capture user intent from a search query expressed in conversational natural language, especially with requirements stated explicitly or implicitly, a search system must be able to accurately extract and structure those intents. Additionally, a search system needs to learn and index products in a way that is both effective and efficient for retrieval. Transformers (Vaswani et al., 2017) have been widely adopted for such tasks, with existing work primarily based on BERT-like models that focus on word-level representations. In recent years, Sentence Transformers (Reimers and Gurevych, 2019b) have gained increasing attention for their ability to capture semantic meaning at the sentence level, making them more suitable for this new setting (Aamir et al., 2024).

Despite recent advances, two main challenges remain in achieving effective semantic search for e-commerce. First, Sentence Transformers need to be fine-tuned to adapt to specific domains, but labeled data is often scarce, making it unclear which products should be positioned closer together in the embedding space. Second, embeddings often struggle to represent numerical values and categorical information (Wallace et al., 2019), which may encode key requirements, as illustrated in the example query above.

To address these challenges, we present a LLM-based semantic search system that effectively captures user intent from conversational search queries and retrieves relevant products from large-scale catalogs. Specifically, we propose generating synthetic queries for products by leveraging LLMs’ world knowledge and reasoning capabilities, and using the inherent semantic links between these synthetic queries and product information to guide the fine-tuning of Sentence Transformers. The same fine-tuned Sentence Transformer is then used to embed user queries for similarity-based retrieval. To capture numerical values and categorical information, we fine-tune a generative model to convert user input into structured filters that are applied before retrieval, ensuring that only items meeting the extracted constraints are considered. To achieve this, we continue using an LLM to enrich the synthetic queries with numerical and categorical constraints and generate corresponding structured filters, which are then used to guide the fine-tuning of the selected generative model.

We use a large-scale Amazon Review dataset (Hou et al., 2024), covering 1.3 million products, to fine-tune our models. To evaluate the semantic search performance of the proposed system, we construct a test set based on query–product relevance annotations from the Amazon ESCI dataset (Reddy et al., 2022). Experimental results show that our system achieves precision@$k$ scores of 0.32, 0.20, and 0.13 for $k=1,5,10$, respectively, and recall@$k$ scores of 0.16, 0.44, and 0.57 for $k=1,5,10$, respectively, significantly outperforming baselines.

Our contributions can be summarized as follows:

-

We propose a framework to investigate how to capture user intent from conversational queries with requirements, expressed either explicitly or implicitly, to improve e-commerce search performance. Our approach, which combines domain-specific embeddings with structured filters, achieves strong precision and recall compared to baseline methods.

-

We propose using LLMs to generate synthetic data to guide domain-specific fine-tuning of models at various stages. The generated queries (both raw and enriched), along with their corresponding structured filters, convey inherent semantic links to products, thereby contributing to effective semantic search.

-

We create a set of user queries that reflect diverse levels of complexity in user requirements, along with carefully generated labels within the Amazon ESCI dataset. We release this dataset as a benchmark to support future research on conversational search queries.

## 2. Related work

We briefly discuss related work in the following areas within the scope of e-commerce.

$\bullet$ Conversational Search. Conversational search (Radlinski and Craswell, 2017; Yang et al., 2018) explores how to support natural language interactions between users and information retrieval systems in a dialogue setting, with the goal of collecting user needs in a natural and efficient manner. For example, prior research has investigated a multi-turn dialogue approach (Liu et al., 2020) that incorporates attention mechanisms to improve customer service interactions. A related task, known as conversational recommendation (Liu et al., 2023; Jia et al., 2022), aims to recommend items to users based on information extracted from dialogues. Our focus differs in that we target search queries rather than full dialogues in e-commerce, referred to as conversational queries, which are single-turn inputs where user intent is fully expressed.

$\bullet$ Semantic Search. Semantic search aims to address the limitations of keyword-based retrieval by learning dense representations of queries and items. Transformers (Vaswani et al., 2017) (such as ERT-like models (Devlin et al., 2018)), including Sentence Transformers (Reimers and Gurevych, 2019b) (such as Sentence-BERT (Reimers and Gurevych, 2019a)), have been widely adopted in various retrieval tasks, including e-commerce. However, existing work typically uses pre-trained transformers or fine-tunes them on domain-specific data. For example, Nigam et al.(Nigam et al., 2019) demonstrated that sentence-level embeddings trained on click data can significantly improve relevance across multi-million–item catalogs. Our work goes further by introducing synthetic queries to guide transformer fine-tuning, leveraging the rich knowledge and reasoning capabilities of LLMs.

$\bullet$ Synthetic Data. Cho et al. (Cho et al., 2022) introduced the idea of using pre-trained language models to generate synthetic queries for every document, enabling dense retrievers to be trained without manual annotations. Extensions of this approach to e-commerce include the work of Chaudhary et al.(Chaudhary et al., 2023), who evaluated synthetic queries on three public retail datasets, and Jagatap et al.(Jagatap et al., 2024), who addressed cold-start ranking in new categories. However, these studies focus exclusively on retrieval or ranking. Our work uses synthetic data at various stages of the semantic search system, including not only the similarity-based retrieval but also the structured filter extraction.

$\bullet$ Structured Filter Generation. Extracting structured constraints from search queries (e.g., price ranges, categories, or review scores) is crucial for satisfying user requirements and is complementary to embedding-based methods. Traditional approaches primarily rely on rule-based techniques. Loughnane et al.(Loughnane et al., 2024) proposed a transformer-based NER pipeline for identifying spans like “price” or “color.” While their pipeline focuses on span detection and relies on rule-based logic for filter application, we take a different approach by training a generative model to directly produce structured filters from natural language queries. Toolkits such as LangChain’s self-querying retriever (LangChain Team, n.d.) demonstrate how an LLM can generate structured filters that are applied after an similarity-based search. While promising, these demonstrations have been limited to small-scale document collections and do not address the scalability, and catalog complexity challenges inherent in retail search systems operating at the scale of millions of items.

## 3. Dataset

We used the Amazon Reviews 2023 dataset (Hou et al., 2024), focusing on the Cell Phones & Accessories category. The dataset contains approximately 1.3 million products spanning the years 1996 to 2023. Each product record includes a unique identifier, textual information including title of the product, description, features, and technical specifications, numerical information including price, the number of reviews, and the average rating of the product, and a label of subcategory.

$\bullet$ Dataset Preprocessing. We apply a two-step process to preprocess the dataset. First, the textual information was cleaned by (1) removing HTML tags, URLs, and non-ASCII characters; (2) converting all text to lowercase; and (3) trimming whitespace. Second, since a manual check revealed instances of incorrect product categorization, we corrected the subcategory labels using the Gemini Flash reasoning model (Google AI for Developers, 2025), which is scalable to large datasets like ours through API usage. Specifically, given a product’s textual information, the model was instructed to assign the product to either the Cell Phones or Cell Phone Accessories subcategory. Discrepancies between the newly assigned labels and the original labels are reviewed to ensure correctness, contributing to improved reliability for downstream tasks such as filtering.

*Figure 1. The LLM-based Semantic Search Framework for Conversational Queries. For each input user query, the framework outputs a ranked list of relevant products by combining similarity search with constraint-based filtering. Embedding component: this component fine-tunes a Sentence Transformer using synthetic queries generated by an LLM from the product catalog. The fine-tuned model is then used to compute embeddings for both the products and the user query. Structure component: this component fine-tunes a generative model to extract numerical and categorical information from user queries and convert it into structured filters. Note that the target product catalog used to generate product embeddings may differ from the catalog used to train the embedding component.*

## 4. Methods

As shown in Figure 1, our framework computes semantic embeddings for each product in the product catalog. For each user query, the system extracts user intent by combining an embedding with structured filters. It then excludes irrelevant products using the structured filters, and finally produces a ranked list of products based on similarity scores using the query embedding. We now describe our methods for generating meaningful embeddings and accurate structured filters, corresponding to the embedding and structure components in Figure 1, respectively. All training was conducted on a single NVIDIA V100 GPU (32GB), which provids sufficient memory for efficient fine-tuning of both models.

### 4.1. Product Embedding Generation

A key component of our framework is generating embeddings for all products in the target catalog to enable accurate and efficient retrieval for user queries. To achieve this, we fine-tune a sentence transformer using synthetic queries to fully capture semantic information in the embeddings and use FAISS to index them for accelerated retrieval.

#### 4.1.1. Sentence transformer selection.

Sentence Transformers is a widely used framework for generating dense vector representations that captures the overall meaning of an entire sentence or paragraph unlike standard transformer models, which produce token-level embeddings. This sentence-level representation enables efficient similarity computation for downstream tasks like semantic search and retrieval (Reimers and Gurevych, 2019a), suitable for applications requiring semantic similarity. We select multi-qa-MiniLM-L6-cos-v1 (Sentence-Transformers, 2024), a model that well balances accuracy and latency with a lightweight architecture that can be efficiently scaled to millions of records without compromising retrieval quality. Additionally, this model offers an extended input capacity (i.e., 512 tokens), which is beneficial for long texts such as ours (i.e., product title, description, features, and technical specifications).

#### 4.1.2. Synthetic query generation.

While using a pre-trained sentence transformer is the default option for generating embeddings, fine-tuning the model with domain-specific information typically improves performance. In addition, we aim for the embedding algorithm to learn how to represent product information in a way that aligns with the user intent expressed in potential queries for that product—so that the product is identified as a correct match when such a query occurs.

To achieve this, we generate query-product pairs, where the queries are synthesized from products’ textual information, which merges product title, description, features, and technical specifications, using a Gemini Flash model. The Gemini models (used here and in other parts of this work) are selected for their API accessibility, large context window (which allows processing multiple products in a single prompt, thereby improving efficiency), and free-tier availability. Other LLMs including open-weight models may also be capable of performing this task. We employ zero-shot prompting to generate high-quality synthetic queries. As shown in Figure 2, the model is instructed to return multiple synthetic queries for each product. We apply this prompt to half of the products in our dataset, with each product receiving approximately 10 diverse synthetic queries, resulting in a total of 6,503,773 unique query-product pairs. This covers 669,294 unique products and 3,452,309 unique queries.

We randomly sample 100 products (50 cell phones and 50 accessories) together with their corresponding synthetic queries to demonstrate that the generated queries are relevant, diverse, and natural-sounding, and closely resemble real user behaviors. We provide an analysis of the generated queries with representative examples in the Appendix.

*Figure 2. Prompt to generate synthetic queries for products. The <PRODUCTS> placeholder is replaced with a list of products, each described by its unique identifier (i.e., parent_asin), product title, features, description, and technical specifications.*

#### 4.1.3. Model fine-tuning.

We then use the 6.5 million query-product pairs to fine-tune the sentence transformer. Each training sample consists of a query and the textual information of the corresponding product. We use MultipleNegativesRankingLoss (Sentence-Transformers Team, n.d.) as the loss function defined as follows:

$\mathcal{J}(\mathbf{x},\mathbf{y},\theta)=-\frac{1}{K}\sum_{i=1}^{K}\left[S(x_{i},y_{i})-\log\sum_{j=1}^{K}e^{S(x_{i},y_{j})}\right]$ | (1) | | | |

where $\mathcal{J}(\mathbf{x},\mathbf{y},\theta)$ denotes the approximated mean negative log-likelihood of the correct responses in a batch of items $\mathbf{x}=(x_{1},x_{2},\dots,x_{K})$ (i.e., synthetic queries) and corresponding correct responses $\mathbf{y}=(y_{1},y_{2},\dots,y_{K})$ (i.e., products), with a batch size $K$ set to 168 in our case. Recall that the synthetic queries are generated using an LLM based on product information, the product paired with a query is regarded as the query’s correct response. The similarity function $S$, parameterized by the neural network weights $\theta$, is used to measure how well a query matches a product. The model is trained to minimize $\mathcal{J}$, thereby assigning higher similarity scores to generated query-product pairs (i.e., $(x_{i},y_{i})$) to mismatched pairs (i.e., $(x_{i},y_{j})$ where $i\neq j$), effectively learning to rank relevant responses higher (Henderson et al., 2017). For each batch, we ensure that every selected product or synthetic query appears only once to avoid false negatives. As a result, a product is more likely to be retrieved and ranked highly when a query similar to the one paired with that product is issued.

#### 4.1.4. Indexing embeddings

The fine-tuned sentence transformer is then used to compute embeddings for all products in the catalog for application. To enable fast and scalable similarity search across large product catalog, we utilize Facebook AI Similarity Search (FAISS) (Douze et al., 2024) to index the embeddings. FAISS is a highly efficient library for searching and clustering dense vector representations. Furthermore, to ensure that only products meeting the structured constraints are considered during retrieval, we develop an IVF-Flat index (FAISS Team, ) for product embeddings, where all products are stored according to their metadata. During retrieval, structured constraints extracted from the query are applied using FAISS’s ID filtering mechanism.

Specifically, the IVF-Flat index partitions the embedding space into a set of inverted lists, with the number of partitions determined heuristically based on the corpus size. After training the index from a random sample of product embeddings, all product embeddings are added to it along with deterministic integer IDs corresponding to their row indices in the product metadata table. This one-to-one mapping ensures stable alignment between each FAISS vector and its associated metadata. For retrieval, with the structured constraints extracted from the query, we preselect the set of product IDs whose metadata satisfy these constraints, and pass them to FAISS to restrict the search space to only the matching subset of vectors. FAISS then performs similarity search (inner product) between the query embedding and the filtered vectors, retrieving the top-$k$ most relevant products. This two-stage approach combines the efficiency of FAISS’s partitioned search with exact filtering over structured attributes, ensuring that retrieved results are both semantically relevant and constraint-compliant.

### 4.2. User Query Understanding

Using the same fine-tuned sentence transformer, we can learn an embedding for each user query, and retrieve relevant products based on similarity scores. However, embeddings often struggle to represent numerical values and categorical information accurately (Wallace et al., 2019)(Kim et al., 2019). To fully capture user intent from their queries, we choose to combine the embedding with structured filters (LangChain Team, n.d.). Specifically, we fine-tune a language model, Flan-T5-small (Chung et al., 2022), to extract structured criteria from the numerical and categorical information of the queries. Flan-T5-small’s instruction-tuned design ensures consistent and structured outputs. The criteria include price, rating, number of reviews, and the subcategory of the target product, as these dimensions are available in the structured attributes of the review dataset. These criteria are used to filter the products before the retrieval based on similarity scores. Final ordering is then determined by dense retrieval scores.

#### 4.2.1. Structured filter generation for synthetic queries

Using the synthetic queries generated in Section 4.1.2, we further employ the Gemini Flash model to enrich the queries with preferences related to one or more of the criteria dimensions mentioned above, and extract and structure the corresponding filters. Table 1 shows an example. We further employ a chat-based generative model (ChatGPT) to synthesize queries that target certain underrepresented constraint styles, such as cell phones with both minimum and maximum price limits, thereby increasing coverage and lexical diversity in the constraint space.

*Table 1. A synthetic query is enriched with numerical features, and structured filters are extracted and formatted in JSON.*

| Original query: |

| smartphone with good battery life |

| Enriched query: |

| smartphone with good battery life, plenty of reviews and priced under $300 |

| Generated label: |

| "price": {"min": null, "max": 300}, "rating_number": {"min": "high", "max": null}, "average_rating": null |

| Structured filters: |

| {"price_min": null, "price_max": 300.0, "review_count_min": "high", "review_count_max": null, "average_rating_min": null, "average_rating_max": null, "subcategory": "Cell Phones"} |

The constraints can be quantitative or qualitative, depending on the description in the enriched query. For numerical values, such as “priced under $300” in the example, the model sets explicit numerical thresholds, i.e., setting price_max to be $300$. For qualitative descriptions, such as “plenty of reviews”, the model maps such expressions to standardized values of high, medium, or low. In the example, the label of review_count_min is set to be high. To map the values of high, medium, and low to the product information when searching, we adopt thresholds listed in Table 2. We map qualitative filters to numeric thresholds based on user expectations. This ensures that qualitative terms such as cheap, popular, or highly rated are applied consistently during retrieval, while also allowing flexibility to adjust these thresholds as needed. Note that these mappings are application-dependent and can be tuned per catalog, and our system supports dynamic threshold selection. In our experiments, however, we fix the thresholds to support training of the single-vector baseline and to ensure that different approaches are evaluated against the same targets. The thresholds used in this work were chosen based on experience and may not be optimal.

*Table 2. Thresholds for rating, review count, and price levels.*

| Metric | Low | Medium | High |

| Rating | [0, 4.0) | [4.0, 5] | [4.5, 5] |

$\infty$ $\infty$| #Reviews | [0, 100) | [100, +) | [1000, +) |

| Price | | | |

$\infty$| Cell Phones | [0, 100] | [100, 300] | [300, +) |

$\infty$| Cell Phone Accessories | [0, 15] | [15, 40] | [40, +) |

The resulting dataset contains 61,812 query–filters pairs. We randomly select 1,000 pairs for evaluation and filters in this set are manually validated and corrected as needed. The remaining pairs of this set is used for model fine-tuning.

#### 4.2.2. Model fine-tuning

To enable accurate extraction of structured filters, we adopt a sequence-to-sequence setup to generate a structured text output conditioned on the enriched input query. This formulation allows the model to learn how to map unstructured user intents to standardized filter representations. Specifically, we fine-tune the Flan-T5-small model using the enriched query-label pairs with a Seq2SeqTrainer (Hugging Face, n.d.), minimizing token-level cross-entropy loss over the target sequence (Jurafsky and Martin, 2025). At each decoding step $t$, the loss is computed based on the log-probability assigned to the correct next token $w_{t+1}$, as follows:

$\mathcal{L}_{\text{CE}}=-\log\hat{y}_{t}[w_{t+1}]$ | (2) | | | |

where $\hat{y}_{t}$ is the model’s predicted distribution over the vocabulary and $w_{t+1}$ is the correct next token. The total loss is averaged over all tokens in the target output. Training is performed using teacher forcing, where the model is conditioned on the correct sequence history rather than its own previous predictions (Jurafsky and Martin, 2025).

## 5. Evaluation

We then evaluate our framework and report its performance. First, we assess the user intent extraction component based on its accuracy in generating structured filters for queries. Then, we evaluate the entire framework based on its ability to retrieve relevant products.

### 5.1. Test Setting

*Table 3. Sample Test Queries.*

| Query # | Natural Language Query |

| 1 | 4G basic phones with keyboards |

| 2 | AT&T prepaid phones under $200 with 4+ stars. |

| 3 | Huawei P30 Pro unlocked. Maximum price: $300. |

| 4 | GSM unlocked flip phones with strong customer feedback |

| 5 | Show me 6-inch screen phones between $100 and $200 and rated 4.2+ stars from 250+ reviews. |

| 6 | Show me Alice in Wonderland iPhone 7 Plus cases with decent review count. |

| 7 | Anker 4-port USB charger averagely priced |

| 8 | I’m searching for a slim waterproof 40 mm Apple Watch Series 4 band with a regular buckle under $25 with strong ratings. |

| 9 | I’m looking for an athletic phone holder between $10 and $14. |

| 10 | I need a cheap and big iPhone SE case. |

Annotated relationships between queries and products is essential for evaluating a system’s ability to answer queries with high precision and recall. However, manually annotating whether a given product is a positive match for every query is infeasible when the product catalog contains 1.3 million items. To address this challenge, we leverage the Amazon ESCI (Shopping Queries) dataset (Reddy et al., 2022), which provides query-product relevance annotations with 130,652 unique queries and 2.62 million query–product judgments. From its extension version, ESCI-S, which augments the original product catalog with structured metadata (such as price, average rating, and review count) for approximately 1.66 million products, we select roughly 22,000 products belonging to the Cell Phones & Accessories category. This catalog serves as the testbed for our framework, on which we evaluate the performance of our proposed system and the baselines in retrieving matched products for given queries.

To create the test queries, for the 22,000 products, we extract query–product pairs that include these products and are labeled as exact matches in the original ESCI dataset. We then randomly sample 151 queries from this set and rewrite them into natural language queries enriched with additional constraints. These constraints deliberately narrow the set of products that satisfy each query, creating a more challenging evaluation set and enabling systematic assessment of our system’s precision and recall relative to the baseline model.

Most of the 151 queries are complex and attribute-rich to reflect realistic user intent involving multiple constraints (such as device type, specific technical specifications, and price). A few simpler queries, such as “Apple 11 Pro Max,” are included to assess the framework’s robustness in handling varying levels of query complexity. All queries are phrased in natural language and structured to require semantic understanding for effective retrieval. See examples in Table 3. Of these queries, 50 have exactly one match, 87% have up to four matches, and none has more than eight exact matches in the catalog of approximately 22,000 products. Therefore, we restrict our evaluation of top-$k$ retrieval effectiveness to values where $k$ $\leq$ 10. Metrics at $k$ = 10 evaluate the system’s ability to retrieve all relevant items, whereas metrics at smaller $k$ values (e.g., $k$ $\leq$ 5) are more indicative of ranking quality and thus more directly affect the user search experience.

### 5.2. Baselines

We compare our system with the following baselines.

$\bullet$ Pre-trained Sentence Transformer. In this setup, we use the pre-trained multi-qa-MiniLM-L6-cos-v1 model from the Sentence Transformers library without any task-specific fine-tuning. To introduce basic filter awareness, we append price, rating, and review count information directly to the beginning of the product metadata prior to embedding. Placing these attributes at the start helps ensure they are encoded as part of the product representation.

$\bullet$ Pre-trained Sentence Transformer + structured filter extraction. In this setup, we combine the same pre-trained Sentence Transformer with our fine-tuned Flan-T5-small filter extraction model. We extracted structured constraints directly from the query using Flan-T5-small and applied them as pre-filters to the retrieved results.

$\bullet$ Sentence Transformer fine-tuned with prepended metadata. This baseline uses the same sentence transformer architecture as our system but is fine-tuned differently. Specifically, price, rating, and review information are prepended to the product text during training, and the model embeds the full product description together with this metadata to provide a fair single-vector baseline for comparison with our multi-stage retrieval approach. While our system’s transformer is trained on approximately 6.5 million query–product pairs without explicit price, rating, or review constraints, this baseline is fine-tuned on a modified dataset that incorporates such constraints. Starting from the original dataset, we first remove around 487 thousand pairs containing qualitative keywords such as best, premium, and top-rated, which implicitly correspond to price, rating or review filters. We then add approximately 780 thousand additional query–product pairs in which the queries contain explicit price, rating, or review count constraints, and the paired products satisfy those constraints. To construct this augmented dataset, we began with roughly 1 million enriched queries generated using the Gemini API. We applied our filter extraction model to identify filter constraints (price, rating, and review count) in each query and separated pairs where product metadata met those constraints. For the remaining pairs, we retained only queries containing purely numeric constraints and programmatically replaced their constraint values with randomized but valid alternatives that matched the paired product metadata. This procedure increased the diversity of constraint formulations. We did not modify queries containing qualitative constraints (e.g., cheap, expensive), since these are mapped to “low,” “medium,” or “high” ranges by our extractor and the original string values cannot be deterministically recovered.

$\bullet$ Sentence Transformer fine-tuned with prepended metadata + structured filter extraction. In this baseline, we combine the fine-tuned Sentence Transformer with our fine-tuned Flan-T5-small filter extraction model similar to baseline 2.

$\bullet$ Sentence Transformer fine-tuned without prepended metadata. We also provide a baseline using the fine-tuned sentence transformer, identical to our complete system except for the absence of the structured filtering component.

For structured filter extraction, we do not include alternative models in the baseline settings but proceed directly with the fine-tuned Flan-T5-small model due to its strong performance, which can be directly evaluated using our held-out query-label dataset. We explicitly compare it with a NER model in Section 5.2.1, but readers may skip to the results in Section 5.3 if they are not interested in the details.

#### 5.2.1. Label extraction component evaluation

With the held-out set of 1,000 query-label pairs generated in Section 4.2.1, we are able to directly evaluate the label extraction component. Across all label dimensions (such as price_min and subcategory, see the label structure in Table 1) the fine-tuned Flan-T5-Small achieves strong results with an accuracy of 99.8% - 99.9%, and the overall match accuracy reaches 99.4%.

The high customizability of the Flan-T5-Small model may help explain its strong performance in this task. For example, it performs well when fine-tuned to map subjective phrases such as “great reviews” to a combination of average_rating_min = high and review_count_min = medium. Alternatively, the same phrase can be trained to correspond solely to average_rating_min = high, depending on the desired level of specificity. Similarly, phrases like “best” can be flexibly mapped to different filter combinations based on application needs.

We introduce a baseline to further demonstrate that the exceptionally high performance likely stems from the capabilities of the language model, rather than issues such as data bias.

$\bullet$ BERT-based NER Baseline. We use a rule-driven Named Entity Recognition (NER) baseline to compare its structured filter extraction ability with our LM-based method. We choose the bert-base-NER model (Devlin et al., 2018), a widely used BERT-based model fine-tuned on NER datasets, known for its strong out-of-the-box performance and ease of integration. To adapt it to our use case, we wrap the NER pipeline with domain-specific, rule-based logic. Specifically, for the product subcategory (i.e., Cell Phones or Cell Phone Accessories), we use a keyword-based strategy to determine it with a predefined list of accessory-related terms (such as “charger” and “screen protector”). For numerical information, related entities are interpreted using context-aware heuristics. For instance, if a numeric span is preceded by phrases like “under,” “less than,” or “stars,” we infer a corresponding constraint on price_max or average_rating_max. Additionally, we integrate qualitative cues using handcrafted keyword lists. Phrases such as “highly rated,” “cheap,” or “many reviews” are mapped to categorical thresholds (e.g., average_rating_min = high, price_max = low, review_count_min = high).

*Table 4. Comparison of Label Extraction Accuracy (%) Between NER Baseline and Fine-Tuned Flan-T5-Small.*

| Field | NER Baseline | Flan-T5-Small |

| price_min | 85.60 | 99.9 |

| price_max | 59.10 | 99.8 |

| review_count_min | 68.60 | 99.8 |

| review_count_max | 98.80 | 99.8 |

| average_rating_min | 70.70 | 99.8 |

| average_rating_max | 100.00 | 99.8 |

| subcategory | 84.70 | 99.9 |

| Overall exact match | 24.40 | 99.4 |

Table 4 shows the effectiveness of the NER model in extracting structured filters. Our model outperforms it across all dimensions except average_rating_max. Notably, only 24.4% of queries have all label dimensions correctly extracted. Inaccuracies in extracting these structured filters, which are used to filter out irrelevant products, can severely degrade overall retrieval performance. While such results could potentially be improved through keyword expansion or additional post-processing logic, these approaches introduce complexity and may still lack robustness in real-world deployment. We will use the fine-tuned Flan-T5-Small as the label extraction component in the system evaluation.

### 5.3. Results

The evaluation results are reported in Table 5. Overall, our system outperforms the baselines with substantial gains across all evaluation metrics.

*Table 5. Evaluation Results Measured by Average Precision and Recall at Top-$k$. When structured filtering is included (bottom table) or not (top table), the best results at each $k$ are shown in bold.*

| Pre-trained Sentence |

| Transformer (filters prepended) |

| Fine-tuned Sentence |

| Transformer (filters prepended) |

| Fine-tuned Sentence |

| Transformer (No filters prepended) |

k Precision@k Recall@k Precision@k Recall@k Precision@k Recall@k

1 0.11 0.06 0.14 0.08 0.13 0.08

2 0.08 0.09 0.13 0.14 0.12 0.13

3 0.08 0.13 0.12 0.19 0.11 0.19

5 0.08 0.20 0.10 0.25 0.10 0.24

10 0.05 0.26 0.07 0.36 0.08 0.36

| Pre-trained Sentence |

| Transformer (filters prepended) |

| & Fine-tuned Flan-T5-Small |

| Fine-tuned Sentence |

| Transformer (filters prepended) |

| & Fine-tuned Flan-T5-Small |

| Fine-tuned Sentence |

| Transformer (No filters prepended) |

| & Fine-tuned Flan-T5-Small (ours) |

k Precision@k Recall@k Precision@k Recall@k Precision@k Recall@k

1 0.22 0.11 0.30 0.16 0.32 0.16

2 0.19 0.18 0.29 0.30 0.29 0.28

3 0.17 0.23 0.25 0.37 0.25 0.36

5 0.14 0.32 0.20 0.47 0.20 0.44

10 0.09 0.39 0.13 0.57 0.13 0.57

#### 5.3.1. Effectiveness of structured filter extraction

Adding structured filter extraction (Flan-T5-small) consistently improves both precision and recall across all sentence transformer configurations (compare the top and bottom tables). For the pre-trained sentence transformer with prepended metadata, precision increases from 0.11 to 0.22 at $k{=}1$ and from 0.05 to 0.09 at $k{=}10$, while recall rises from 0.06 to 0.11 ($k{=}1$) and from 0.26 to 0.39 ($k{=}10$). When the model is fine-tuned with prepended metadata, precision improves from 0.14 to 0.30 at $k{=}1$ and from 0.07 to 0.13 at $k{=}10$, with recall gains from 0.08 to 0.16 ($k{=}1$) and from 0.36 to 0.57 ($k{=}10$). A similar pattern holds for the fine-tuned model without prepended metadata: precision rises from 0.13 to 0.32 at $k{=}1$ and from 0.08 to 0.13 at $k{=}10$, and recall improves from 0.08 to 0.16 ($k{=}1$) and from 0.36 to 0.57 ($k{=}10$). These gains indicate that accurate, query-specific pre-filtering substantially reduces irrelevance prior to ranking, yielding the largest absolute improvements in recall at higher $k$.

#### 5.3.2. Effectiveness of sentence transformer fine-tuning

Fine-tuning the sentence transformer improves over the pre-trained model in both settings (with and without structured filters). Without structured filters (top table), fine-tuning with prepended metadata improves precision from 0.11 to 0.14 at $k{=}1$ and recall from 0.26 to 0.36 at $k{=}10$. The variant without prepended metadata performs best at larger $k$, achieving the highest recall at $k{=}10$ (0.36) and the highest precision at $k{=}10$ (0.08). With structured filters (bottom table), both fine-tuned variants achieve the strongest results overall. The variant without prepended metadata attains the best $k{=}1$ scores (precision 0.32, recall 0.16) and the tied-best recall at $k{=}10$ (0.57), while the prepended-metadata variant remains very close (precision 0.30, recall 0.16 at $k{=}1$; recall 0.57 at $k{=}10$) and achieves the best or tied-best recall. Overall, fine-tuning yields consistent gains over the pre-trained configuration, and combining fine-tuning with structured filter extraction delivers the best precision and recall across all reported $k$.

#### 5.3.3. Does constraint-aware fine-tuning help?

Across our baselines, fine-tuning the sentence transformer specifically to encode price, rating, and review constraints yields limited gains relative to a constraint-agnostic fine-tuning objective. Without structured filters (top table), fine-tuning improves over the pre-trained model (e.g., precision at $k{=}1$: 0.11 $\rightarrow$ 0.14; recall at $k{=}10$: 0.26 $\rightarrow$ 0.36), but the two fine-tuned variants (with vs. without prepended metadata) are very close and often trade places. When structured filter extraction is enabled (bottom table), the two fine-tuned variants remain very close at all reported $k$ (e.g., at $k{=}1$: precision 0.30 vs. 0.32 and recall 0.16 vs. 0.16; at $k{=}10$: precision 0.13 for both and recall 0.57 for both), indicating that the improvement comes from accurate pre-filtering rather than from constraint-aware representation learning.

## 6. Discussion

$\bullet$ Effective handling of constraint information.The experiments suggest that structured filter extraction is the primary driver of improvements in both precision and recall, as the extracted constraints effectively narrow the search space before ranking. Incorporating constraint information through metadata prepending during fine-tuning, however, does not meaningfully alter the retrieval geometry once pre-filters are applied. In practice, the effective strategy is to use a generally fine-tuned encoder for retrieval and apply structured filters afterward to enforce price, rating, and review constraints prior to ranking. Moreover, preparing training data for constraint-aware fine-tuning is labor-intensive and less flexible. In particular, handling qualitative constraints (e.g., “cheap,” “premium”) requires ad hoc design decisions during data preparation and cannot be easily adapted post-training. By contrast, the filter extraction model converts qualitative constraints into normalized categories (high, medium, low), which can then be mapped to numerical boundaries that are adjustable at inference time. This provides a level of flexibility that is not feasible with fine-tuning alone, since any changes to constraint definitions in the fine-tuned setup would require retraining the model. Future work could explore objectives that explicitly penalize constraint violations during training, such as hard-negative mining with near-duplicate products that differ only in price or rating.

$\bullet$ Synthetic queries generated by LLMs. With the system effectiveness well evaluated, we now further discuss and highlight the role of the synthetic queries in our framework. Using LLMs, the synthetic queries were first generated to pair with each product and guide the fine-tuning of the sentence transformer, which is later used to compute embeddings for all products and real user queries. This process extremely contributes to the effectiveness of embedding, indicated by the performance gains described in Section 5.3.2. Additionally, still using LLMs, the synthetic queries were enriched with rich attributes that align with realistic needs in user searches, which were later converted to structured filters of numerical and categorical information. The enriched queries and the corresponding structured filters were used to guide the fine-tuning of the user intent extraction model, which was used to accurately generate structured filters from user queries. See the performance gains described in Section 5.3.1. Results suggest that leveraging LLMs to generate and enrich synthetic queries not only overcomes the limitations of manual annotation in large datasets like ours but also effectively harnesses the world knowledge and reasoning capabilities of LLMs. Furthermore, synthetic query generation can be used to address the challenge of edge cases in search systems, for example, by instructing an LLM to generate representative edge-case queries for each product to support model training or fine-tuning.

$\bullet$ Model efficiency. Our framework is designed with model efficiency in mind to reflect the real-time demands of practical e-commerce applications. Specifically, product embeddings are indexed using FAISS to enable fast similarity-based search, and a combination of lightweight models (i.e., multi-qa-MiniLM-L6-cos-v1 and Flan-T5-small) is selected to handle user queries.

$\bullet$ Architectural support for generalization. Our framework is designed with flexibility in mind as well, making it well-suited for future improvements. For example, a larger embedding model can be fine-tuned and deployed for product categories where smaller models underperform. The two-stage design supports a multi-model deployment strategy across multiple product categories, where embeddings are trained separately for each category and the user intent extraction component dynamically routes queries to the most appropriate category.

$\bullet$ Imperfect ground-truth labels. Reviewing the retrieval results suggests that the reported precision and recall may not fully reflect the true performance of the evaluated models, as we identified labeling inconsistencies in the ESCI dataset. Several retrieved products were exact matches both semantically and in metadata but were not labeled as such in the annotations. This behavior is expected in large-scale e-commerce datasets, where relevance judgments are often incomplete or inconsistent, highlighting the challenge of evaluating real-world retrieval systems using imperfect ground-truth labels.

$\bullet$ Limitation. A key limitation of our work lies in the quality of the labels used to train the filter extraction model, which require manual validation due to inconsistencies in LLM-generated outputs. Fine-tuning a language model may help address this issue in future work. Additionally, the implementation and evaluation of the proposed framework focus on the Cell Phones & Cell Phone Accessories category. How it performs on other categories, especially those with more noise or ambiguity, remains uncertain. Nevertheless, our work demonstrates the effectiveness of a domain-specific, fine-tuned embedding model (using synthetic queries) and structured filters in capturing users’ explicit and implicit intent, providing a solid foundation for solutions in more challenging domains.

## 7. Conclusion

In this paper, we presented an LLM-based framework capable of performing effective semantic search for conversational user queries in e-commerce. Using synthetic data generated by LLMs, we fine-tuned a Sentence Transformer for product and query embedding, as well as a sequence-to-sequence model for structured filter extraction. Experiments on a test set consisting of 151 manually crafted queries and a product catalog of roughly 22 thousand items demonstrate the effectiveness and superiority of our framework across queries with varying levels of complexity, compared to baseline methods. Future work includes fine-tuning language models to improve label generation for the training task, incorporating reranking models on top of the current pipeline, and validating the viability of the approach across other product categories.

## Appendix A Analysis of Synthetic Queries

We randomly sample 100 unique products (50 cell phones and 50 accessories) from the 6.5 million query–product pairs and manually screen the quality of the corresponding queries. Overall, the queries range from concise, name-based lookups to attribute-focused, conversational requests; they are relevant, diverse, and natural-sounding, and closely resemble real user behavior, with rare exceptions. We provide a few representative examples below.

Example 1:

Product: Samsung Galaxy A01 (LTE Verizon).

Product description (excerpt): Long lasting battery, versatile camera, 5.7 inch HD+ screen, expandable 16 GB storage, Android OS, Verizon compatibility.

Generated queries:

-

samsung galaxy a01 verizon 16gb

-

unlocked samsung galaxy a01 black

-

5.7 inch display phone for verizon

-

i need a phone with a long lasting battery

-

find a phone with a good camera and expandable storage

-

best budget phone for verizon with a large display

-

samsung galaxy a01 with 5.7" screen

Observation: The model mixes feature cues (battery, display, storage, and camera) with user intent (e.g., “unlocked” and “long lasting battery”) and applies general knowledge about Samsung’s budget line by identifying the product as a budget phone.

Example 2:

Product: Temdan for Samsung Galaxy Note 20 Ultra case.

Product description (excerpt): IP68 waterproof, built-in screen protector, full-body shockproof protection, compatible with Note 20 Ultra 5G.

Generated queries:

-

samsung galaxy note 20 ultra waterproof case

-

note 20 ultra case with screen protector

-

note 20 ultra case 360 degree protection

-

note 20 ultra case heavy duty

-

temdan note 20 ultra case

Observation: The queries vary meaningfully while consistently targeting salient attributes (IP rating, durability, and brand or model compatibility), and they reflect realistic shopping phrasing.

Example 3:

Product: Biter Galaxy S6/S6 Edge Case.

Product description (excerpt): Scratch resistant, three-in-one slim cover with electroplated frame, full-body protection; compatible with multiple Galaxy models.

Generated queries:

-

i need a galaxy s6/s6 edge case that is scratch resistant

-

find a galaxy s6/s6 edge case that offers full body protection

-

gold s6/s6 edge case

-

galaxy s6/s6 edge case with electroplated frame

-

galaxy s6/s6 edge case with three in one design

Observation: This example illustrates a rare failure case in which the generated queries exhibit limited diversity. All queries repeat the exact phrase “s6/s6 edge,” rather than producing distinct queries for each compatible device. In particular, no queries explicitly target “galaxy s6” or “galaxy s6 edge” individually, which reduces both query diversity and realism. In practice, users are more likely to search using a single model name rather than a combined string. This limitation could likely be mitigated by refining the prompts provided to the LLM to encourage model-specific phrasing.

That said, this behavior is not consistent across products. In several instances, the LLM correctly generated distinct queries for products compatible with multiple devices. For example, for a case compatible with both the iPhone 6 Plus and iPhone 6s Plus, the generated queries included:

-

iphone 6s plus case

-

iphone 6 plus case

-

iphone 6 plus / 6s plus case with mint color

Similarly, for a Samsung battery compatible with multiple models (e.g., SGH-T919 Behold, SGH-A797 Flight), the LLM produced model-specific queries such as:

-

samsung battery for sgh-t919 behold

-

samsung battery for sgh-a797 flight

-

replacement battery for samsung gravity 2

-

i need a battery for my samsung impression

-

looking for a compatible battery for my samsung sgh-a727

## References

- Aamir et al. (2024) F. Aamir, R. Sherafgan, T. Arbab, A. Jamil, F. N. Bhatti, and A. A. Hameed Deep learning-based semantic search techniques for enhancing product matching in e-commerce. In 2024 IEEE 3rd International Conference on Computing and Machine Intelligence (ICMI), Vol. , pp. 1–9. External Links: Document Cited by: §1.

- Bloomreach (2025) Bloomreach More than 60% of consumers have used conversational AI for shopping, new research from bloomreach finds. Note: https://www.bloomreach.com/en/news/2025/bloomreach-announces-findings-from-conversational-ai-shopping-study/Accessed: 2025-12-22 Cited by: §1.

- Chaudhary et al. (2023) A. Chaudhary, K. Raman, K. Srinivasan, K. Hashimoto, M. Bendersky, and M. Najork Exploring the viability of synthetic query generation for relevance prediction. arXiv preprint arXiv:2305.11944. External Links: Document, Link Cited by: §2.

- Cho et al. (2022) S. Cho, S. Jeong, W. Yang, and J. Park Query generation with external knowledge for dense retrieval. In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, E. Agirre, M. Apidianaki, and I. Vulić (Eds.), Dublin, Ireland and Online, pp. 22–32. External Links: Link, Document Cited by: §2.

- Chung et al. (2022) H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, E. Li, X. Wang, M. Dehghani, S. Brahma, A. Webson, S. S. Gu, Z. Dai, M. Suzgun, X. Chen, A. Chowdhery, S. Narang, G. Mishra, A. Yu, V. Zhao, Y. Huang, A. Dai, H. Yu, S. Petrov, E. H. Chi, J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V. Le, and J. Wei Scaling instruction-finetuned language models. arXiv. External Links: Document, Link Cited by: §4.2.

- Devlin et al. (2018) J. Devlin, M. Chang, K. Lee, and K. Toutanova BERT: pre-training of deep bidirectional transformers for language understanding. CoRR abs/1810.04805. External Links: Link, 1810.04805 Cited by: §2, §5.2.1.

- Douze et al. (2024) M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou The faiss library. arXiv preprint arXiv:2401.08281. External Links: 2401.08281 Cited by: §4.1.4.

- [8] FAISS Team FAISS wiki: faiss indexes. Note: https://github.com/facebookresearch/faiss/wiki/Faiss-indexesAccessed: 2025-10-08 Cited by: §4.1.4.

- Google AI for Developers (2025) Google AI for Developers Gemini models. Note: Accessed: 2025-05-24 External Links: Link Cited by: §3.

- Henderson et al. (2017) M. L. Henderson, R. Al-Rfou, B. Strope, Y. Sung, L. Lukács, R. Guo, S. Kumar, B. Miklos, and R. Kurzweil Efficient natural language response suggestion for smart reply. arXiv preprint arXiv:1705.00652. External Links: Document, Link Cited by: §4.1.3.

- Hou et al. (2024) Y. Hou, J. Li, Z. He, A. Yan, X. Chen, and J. McAuley Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952. Cited by: §1, §3.

- Hugging Face (n.d.) Hugging FaceTrainer — transformers documentation(Website) Note: Accessed: May 22, 2025 External Links: Link Cited by: §4.2.2.

- Jagatap et al. (2024) A. Jagatap, S. Merugu, and P. M. Comar Improving search for new product categories via synthetic query generation strategies. External Links: Link Cited by: §2.

- Jia et al. (2022) M. Jia, R. Liu, P. Wang, Y. Song, Z. Xi, H. Li, X. Shen, M. Chen, J. Pang, and X. He E-convrec: a large-scale conversational recommendation dataset for e-commerce customer service. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pp. 5787–5796. Cited by: §2.

- Jurafsky and Martin (2025) D. Jurafsky and J. H. Martin Speech and language processing: an introduction to natural language processing, computational linguistics, and speech recognition with language models. 3rd edition. Note: Online manuscript released January 12, 2025 External Links: Link Cited by: §4.2.2, §4.2.2.

- Kashyap (2023) K. KashyapConsumers exit online portals due to poor search experience(Website) External Links: Link Cited by: §1.

- Kim et al. (2019) J. Kim, R. K. Amplayo, K. Lee, S. Sung, M. Seo, and S. Hwang Categorical metadata representation for customized text classification. Transactions of the Association for Computational Linguistics 7, pp. 201–215. External Links: Link, Document Cited by: §4.2.

- LangChain Team (n.d.) LangChain TeamHow to do “self-querying” retrieval(Website) Note: Accessed: May 22, 2025 External Links: Link Cited by: §2, §4.2.

- Liu et al. (2020) R. Liu, M. Chen, H. Liu, L. Shen, Y. Song, and X. He Enhancing multi-turn dialogue modeling with intent information for e-commerce customer service. In Natural Language Processing and Chinese Computing: 9th CCF International Conference, NLPCC 2020, Zhengzhou, China, October 14–18, 2020, Proceedings, Part I 9, pp. 65–77. Cited by: §2.

- Liu et al. (2023) Y. Liu, W. Zhang, B. Dong, Y. Fan, H. Wang, F. Feng, Y. Chen, Z. Zhuang, H. Cui, Y. Li, et al. U-need: a fine-grained dataset for user needs-centric e-commerce conversational recommendation. In Proceedings of the 46th international ACM SIGIR conference on research and development in information retrieval, pp. 2723–2732. Cited by: §2.

- Loughnane et al. (2024) R. Loughnane, J. Liu, Z. Chen, Z. Wang, J. Giroux, T. Du, B. Schroeder, and W. Sun Explicit attribute extraction in e-commerce search. In Proceedings of the Seventh Workshop on e-Commerce and NLP @ LREC-COLING 2024, S. Malmasi, B. Fetahu, N. Ueffing, O. Rokhlenko, E. Agichtein, and I. Guy (Eds.), Torino, Italia, pp. 125–135. External Links: Link Cited by: §2.

- Nigam et al. (2019) P. Nigam, Y. Song, V. Mohan, V. Lakshman, W. Ding, A. Shingavi, C. H. Teo, H. Gu, and B. Yin Semantic product search. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’19), pp. 2876–2885. External Links: Document, Link Cited by: §2.

- Radlinski and Craswell (2017) F. Radlinski and N. Craswell A theoretical framework for conversational search. In Proceedings of the 2017 conference on conference human information interaction and retrieval, pp. 117–126. Cited by: §2.

- Reddy et al. (2022) C. K. Reddy, L. Màrquez, F. Valero, N. Rao, H. Zaragoza, S. Bandyopadhyay, A. Biswas, A. Xing, and K. Subbian Shopping queries dataset: a large-scale ESCI benchmark for improving product search. External Links: 2206.06588 Cited by: §1, §5.1.

- Reid (2025) E. Reid AI mode in google search: updates from google I/O 2025. Note: https://blog.google/products/search/google-search-ai-mode-update/#ai-mode-searchAccessed: 2025-12-22 Cited by: §1.

- Reimers and Gurevych (2019a) N. Reimers and I. Gurevych Sentence-bert: sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, External Links: Link Cited by: §2, §4.1.1.

- Reimers and Gurevych (2019b) N. Reimers and I. Gurevych Sentence-bert: sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992. Cited by: §1, §2.

- Sentence-Transformers Team (n.d.) Sentence-Transformers TeamLosses — Sentence Transformers Documentation(Website) Note: Accessed: May 22, 2025 External Links: Link Cited by: §4.1.3.

- Sentence-Transformers (2024) Sentence-Transformersmulti-qa-MiniLM-L6-cos-v1(Website) Hugging Face. Note: Accessed: May 22, 2025 External Links: Link Cited by: §4.1.1.

- Vaswani et al. (2017) A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin Attention is all you need. Advances in neural information processing systems 30. Cited by: §1, §2.

- Wallace et al. (2019) E. Wallace, Y. Wang, S. Li, S. Singh, and M. Gardner Do nlp models know numbers? probing numeracy in embeddings. Note: arXiv preprint arXiv:1909.07940 External Links: Document, Link Cited by: §1, §4.2.

- Yang et al. (2018) Y. Yang, Y. Gong, and X. Chen Query tracking for e-commerce conversational search: a machine comprehension perspective. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pp. 1755–1758. Cited by: §2.

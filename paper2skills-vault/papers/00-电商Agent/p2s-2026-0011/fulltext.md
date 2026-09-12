<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2608.20844
     paper_id : p2s-2026-0011
     source   : https://arxiv.org/html/2608.20844v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# TRACE: Agentic Catalog Enrichment with Multi-source Evidence Grounding

Rohan Kumar    Steven Xu    Kyle MacDonald    Matthew Long    Bernice Chow    Mac VanRenterghem    Sudeep Das

###### Abstract

Product catalogs underpin search, discovery, and recommendation in e-commerce, yet they are often attribute-sparse: the attributes shoppers and downstream systems rely on are either buried in unstructured content such as titles and images or missing from the catalog altogether. Manually enriching e-commerce catalogs is impractical given their scale and rapid growth. This paper introduces TRACE, a novel framework for automated catalog attribute enrichment using agentic Large Language Models (LLMs). A ScoutAgent triangulates multimodal evidence across merchant catalogs, syndicated feeds, and identity-matched web search to propose candidate attribute values with supporting evidence, while a JudgeAgent verifies the proposed value for each attribute value against its supporting evidence and decides whether to publish it or route it to human review. On an offline human evaluation dataset, TRACE’s proposed attribute values were 98.2% accurate at 74.7% attribute coverage. Deployed in production on an industry-scale catalog, TRACE increased impression-weighted enrichment coverage across four business verticals by 90.4%. An online experiment subsequently showed that surfacing the enriched attributes on the product detail page increased checkout conversion by 0.48%.

###### keywords

catalog enrichment ,multimodal LLM ,attribute extraction ,LLM-as-judge ,agentic search

## 1 Introduction

E-commerce and marketplace platforms rely on product catalogs that combine unstructured content (e.g., titles and images) with structured attributes (e.g., brand, size, and dietary tags). Structured attributes help shoppers filter and compare products, enable search systems to match queries to attribute intents Zhang et al. (2021); Nigam et al. (2019), and allow recommendation models to learn granular product affinities Wang et al. (2019). In practice, catalog data provided by sellers and merchants is often attribute-sparse Yang et al. (2022); Dong et al. (2020). Relevant values may be buried in unstructured content rather than represented as structured fields, or missing from the catalog altogether. As a result, shoppers may lack the information needed to make confident purchasing decisions or form accurate expectations about the products they receive, while downstream systems must rely on coarser product representations, limiting retrieval precision and the depth of personalization. Catalog enrichment seeks to close these gaps by extracting or sourcing missing attribute values and converting them into structured fields. Performing this work manually requires parsing free text, transcribing images, and triangulating evidence across data sources with heterogeneous formats and varying quality, making it error-prone and infeasible for large-scale, rapidly growing product catalogs.

Early work on product attribute extraction has relied on task-specific natural language processing (NLP) models trained to recover structured values from product text and images Zheng et al. (2018); Zhu et al. (2020). More recently, large language models (LLMs) have enabled more flexible approaches through their zero-shot and few-shot capabilities, reducing the need for labeled data and task-specific fine-tuning Brinkmann et al. (2023); Sinha and Gujral (2024). Despite this advancement, two practical challenges remain: (1) Some attribute values cannot be reliably inferred from owned data sources and must instead be sourced externally and verified against the exact product. (2) Accuracy estimated from a point-in-time catalog audit may become less representative as the catalog’s product mix evolves. As a result, a system that performs well on average may still publish unsupported or product-mismatched values, particularly for products and evidence patterns underrepresented in the evaluation sample. Missing or inaccurate attribute values can mislead shoppers and degrade fulfillment quality; for safety-sensitive attributes such as allergens or dietary restrictions, they can have especially serious consequences.

To address these challenges, we present TRACE (Figure 1), a novel framework for automated catalog attribute enrichment using agentic Large Language Models (LLMs). The workflow is divided among two specialized agents: a ScoutAgent that gathers grounded evidence from multiple sources, and a JudgeAgent that verifies proposed value before publication. Our main contributions are threefold: (1) We introduce an end-to-end agentic LLM framework for industry-scale product catalog enrichment. The workflow triangulates heterogeneous evidence across multiple data sources while preserving provenance. (2) We incorporate identity-matched search grounding to recover attribute values that cannot be reliably inferred from owned data sources. (3) We place a JudgeAgent in the serving path as a verify-before-write gate, applying a consistent evidence standard to each proposed value and making publication quality less sensitive to shifts in the catalog’s product mix. Through offline evaluation on datasets spanning four marketplace verticals, we show that TRACE can enrich attribute-sparse product catalogs with expert-level accuracy while achieving a scale and throughput unattainable through manual enrichment.

*Figure 1: TRACE’s architecture. The ScoutAgent receives the SKU record together with available catalog, syndicated, and image evidence. It may perform iterative, identity-matched web searches to resolve missing information and produces candidate attribute values with supporting evidence and provenance. The JudgeAgent independently verifies each candidate, optionally conducting additional web searches, and assigns a per-attribute verdict. The write gate then routes each proposed value to WRITE, BLOCK, or REVIEW.*

## 2 Related Work

Recovering structured attributes from unstructured text and images using generative models is a well-studied problem. Early approaches fine-tuned encoder-decoder models for attribute generation Khandelwal et al. (2023). More recent methods use foundation models through zero-shot or few-shot prompting Sinha and Gujral (2024), sometimes augmented with retrieval over similar catalog entries Zhang et al. (2025). Multi-agent systems are also emerging, with recent work using iterative collaboration to refine attribute predictions and cross-check extracted knowledge Huang and Caragea (2025); Lu and Wang (2025). These methods generally assume that the target value can be inferred from owned data sources, and the attribute of interest has already been extracted for similar products.

LLM-as-judge methods are widely used to evaluate model outputs Zheng et al. (2023); Liu et al. (2023), including in grounded settings Saad-Falcon et al. (2024); Es et al. (2024) and multi-model juries designed to reduce single-model bias Verga et al. (2024). They may serve as standalone evaluators or as critics within multi-agent workflows. However, little prior work has examined how an LLM judge can be integrated into a production catalog-enrichment pipeline to make per-value publication decisions under varying evidence quality and a continually changing product and attribute mix.

External search grounding helps LLMs access specialized or up-to-date knowledge beyond the provided data. Prior work retrieves real-time information to ground generation Shi et al. (2025) and uses agentic, tool-augmented search, in which a model interleaves reasoning with search actions to gather evidence and reduce hallucination Yao et al. (2023). These methods are not directly suited to catalog enrichment, where topical relevance alone is insufficient: a retrieved source may describe a closely related product or variant with different attribute values. Sourcing catalog attributes therefore requires explicit identity matching to ensure that the evidence applies to the exact product being enriched.

## 3 Methodology

### 3.1 Overview

Let $s$ denote a stock-keeping unit (SKU), $x_{s}$ its catalog record, and $c_{s}$ its leaf category. A category-specific template maps $c_{s}$ to a set of priority attributes:

$\mathcal{A}_{s}=\mathrm{Template}(c_{s};\mathcal{T}),$ | | | | (1) |

where $\mathcal{T}$ is the collection of attribute templates. These templates restrict enrichment to attributes that are meaningful for the product category.

For each attribute $a\in\mathcal{A}_{s}$, the objective is to produce either a grounded candidate value or an explicit abstention Wen et al. (2025). We represent a candidate as

$c_{a}=\left(a,\,v_{a},\,E_{a},\,\tau_{a},\,q_{a},\,z_{a}\right),$ | | | | (2) |

where $v_{a}$ is the proposed value, $E_{a}$ is its supporting evidence, $\tau_{a}$ records the evidence-source types, $q_{a}$ is a model-reported confidence score, and $z_{a}$ records the extraction status as one of extracted, not_found, not_applicable, ambiguous, or conflict. The final system output maps each candidate to one of three operational actions: WRITE, BLOCK, or REVIEW.

TRACE implements this process as a two-stage verify-before-write architecture. The ScoutAgent gathers evidence from multiple sources, verifies that externally retrieved evidence refers to the target product, and proposes grounded attribute values. The JudgeAgent then re-examines each candidate under a stricter verification policy and determines whether it is eligible for publication, should be blocked, or requires human review. Figure 1 shows the end-to-end workflow, and Algorithm 1 formalizes the procedure.

*Algorithm 1 TRACE enrichment for a SKU $s$.*

1: SKU $s$, category templates $\mathcal{T}$, threshold $\theta$

2: Candidate values with actions in $\{\textsc{write},\textsc{block},\textsc{review}\}$

3: $\mathcal{A}_{s}\leftarrow\mathrm{Template}(s,\mathcal{T})$

4: if $\mathcal{A}_{s}=\emptyset$ then

5:   return $\emptyset$

6: end if

7: $E\leftarrow\mathrm{Catalog}(s)\cup\mathrm{Syndicated}(s)\cup\mathrm{Image}(s)$

8: for all $a\in\mathcal{A}_{s}$ unresolved by $E$ do

9:   for all $p\in\mathrm{Search}(\mathrm{Query}(s,a))$ do

10:    if $\mathrm{IdentityMatch}(p,s)$ then

11:      $E\leftarrow E\cup\mathrm{ExtractEvidence}(p,a)$

12:    end if

13:   end for

14: end for

15: $C\leftarrow\mathrm{Reconcile}(\mathcal{A}_{s},E)$ $\triangleright$ normalize, merge, or abstain

16: $Y\leftarrow\mathrm{\texttt{JudgeAgent}}(C,E)$

17: $d\leftarrow\mathrm{WritePolicy}(Y,C,\mathrm{confidence},\theta)$

18: Apply($d,c$)

19: return $C$

### 3.2 The ScoutAgent

The ScoutAgent gathers and reconciles evidence for each target attribute in two stages. It first considers readily available, product-linked information, including textual fields and product images from seller-provided catalog data and syndicated product data. When this information is insufficient to determine a reliable value for the target attribute, the ScoutAgent uses web search to gather additional evidence.

-

Seller-provided catalog data. Structured and unstructured product information supplied by the seller, including textual fields (e.g., item name, description) and product images.

-

Syndicated product data. Product records supplied by commercial data syndicators. These records can provide authoritative specifications and product images, although their coverage and mapping quality vary.

-

Agentic web search. For attributes that remain unresolved, the ScoutAgent constructs targeted queries from the available product identifiers and the target attribute to retrieve additional evidence from the web.

Although product images originate from seller-provided or syndicated records, TRACE treats image evidence as a separate tier from textual evidence because visual attribute extraction is generally noisier than extraction from text. This separation allows the ScoutAgent to prioritize textual evidence while still using images for attributes expressed visually, such as material, certification marks, and on-package claims.

#### Identity-grounded web retrieval.

Web search may return pages that appear relevant to the query but refer to a different product or product variant. Query relevance alone is therefore insufficient for catalog enrichment, where the evidence must apply to the exact product being enriched. Before extracting an attribute value, the ScoutAgent verifies that the retrieved page refers to the target product. In our implementation, identity matching is performed within the ScoutAgent’s ReAct-style reasoning loop Yao et al. (2023) using the available product signals, including identifiers and descriptive metadata in the catalog record and retrieved page. In Algorithm 1, $\mathrm{IdentityMatch}(p,s)$ denotes this internal reasoning step for a retrieved page $p$ and SKU $s$, rather than a separate model call. Evidence from $p$ is used only when the ScoutAgent determines that the page describes the target product; otherwise, the page is discarded.

#### Evidence reconciliation and abstention.

After gathering evidence, the ScoutAgent normalizes and reconciles candidate values across sources. It maps benign variations to a common representation, such as “NiMH” and “nickel-metal hydride,” or “60 Hz” and “60Hz.” Evidence supporting the same normalized value is then consolidated into a single candidate while preserving source-level provenance. When the available evidence is insufficient, ambiguous, or conflicting, the ScoutAgent abstains rather than inferring a value from background knowledge. It records the outcome as one of extracted, not_found, not_applicable, ambiguous, or conflict. For each target attribute, the ScoutAgent outputs a candidate record containing the normalized value when available, supporting evidence, source types, model-reported confidence, and extraction status. The candidate record and its provenance are then passed to the JudgeAgent for verification.

The prompt template for ScoutAgent is provided in Figure 3 in Appendix.

### 3.3 The JudgeAgent

Before a candidate can be written to the catalog, it is adjudicated by the JudgeAgent. For each target attribute, the JudgeAgent receives the candidate record produced by the ScoutAgent, including the proposed value, supporting evidence, source types, and extraction status. It reassesses whether the evidence applies to the target product and whether it supports the proposed value. Similar to the ScoutAgent, the JudgeAgent may use web search to gather additional evidence when it determines the supplied evidence is insufficient for verification.

The JudgeAgent operates at the SKU level, allowing candidates to be evaluated in parallel across SKUs. It applies a stricter evidence policy than ScoutAgent, focusing on whether available evidence supports the proposed value for the target product.

#### Verdict taxonomy.

For each candidate, the JudgeAgent returns one of four verdicts:

-

PASS: The available evidence supports the proposed value for the target product.

-

FAIL: The proposed value is contradicted by the evidence or is not supported by it. The verdict includes a diagnostic subtype, such as HALLUCINATION or CONTRADICTION.

-

UNVERIFIED: The proposed value is not contradicted, but the available evidence does not directly confirm it for the target product. This verdict represents insufficient verification rather than an identified error.

-

UNCERTAIN: The evidence is conflicting or ambiguous, preventing the JudgeAgent from reaching a reliable determination.

The distinction between UNVERIFIED and UNCERTAIN separates a lack of confirming evidence from active disagreement among the available evidence. The empirical motivation for this distinction is discussed in Section 4.3.

#### From verdict to catalog action.

The JudgeAgent verdict is separated from the operational write policy. Candidates below the model-reported confidence threshold $\theta$ are blocked. Among the remaining candidates, those receiving PASS or UNVERIFIED are written, those receiving FAIL are blocked, and those receiving UNCERTAIN are routed to human review together with their evidence trail. Formally,

$d_{a}=\begin{cases}\texttt{BLOCK},&q_{a}<\theta\text{ or }y_{a}=\texttt{FAIL},\\
\texttt{REVIEW},&y_{a}=\texttt{UNCERTAIN},\\
\texttt{WRITE},&y_{a}\in\{\texttt{PASS},\texttt{UNVERIFIED}\},\end{cases}$ | | | | (3) |

where $y_{a}$ is the JudgeAgent verdict and $q_{a}$ is the ScoutAgent’s model-reported confidence. Separating the verdict from the write policy allows the publication rules to be adjusted without changing the JudgeAgent’s verdict taxonomy.

The prompt template for JudgeAgent is provided in Figure 4 in Appendix.

## 4 Experiments

We conduct comprehensive experiments to evaluate TRACE, our agentic framework for catalog enrichment, using data sampled from the production catalog of an e-commerce marketplace platform spanning multiple business verticals. We combine human evaluation with JudgeAgent-based adjudication to assess the quality of the enriched catalog.

Unless otherwise stated, all experimental results use Gemini 2.5 Comanici et al. (2025) Flash as the backbone for both the ScoutAgent and JudgeAgent. We compare alternative VLM backbones for ScoutAgent in Section 4.4.

### 4.1 Data Collection

We evaluate TRACE on products from four business verticals: Grocery, Alcohol, Electronics, and Home Improvement. The number of distinct target attributes ranges from 11 in Grocery to 409 in Home Improvement.

#### Grocery and Alcohol.

This dataset contains 500 SKUs and 2,497 target SKU–attribute pairs. Human annotators established the reference attribute values, and a separate group of auditors reviewed the values proposed by the ScoutAgent. We use this dataset to measure human-validated extraction quality and to analyze the behavior of the JudgeAgent against human judgments.

#### Electronics and Home Improvement.

This dataset contains 955 SKUs and 4,990 target SKU–attribute pairs. Because exhaustive human labeling was not available for these verticals, we use the JudgeAgent to adjudicate the complete dataset. The JudgeAgent results provide a scalable operational quality signal.

### 4.2 Evaluation Metrics

We evaluate the ScoutAgent using extraction accuracy and attribute coverage. Extraction accuracy is the fraction of extracted values judged correct. We report human-validated accuracy when correctness is determined by human review. Attribute coverage is the fraction of requested SKU–attribute pairs for which the ScoutAgent produces a nonempty value.

For datasets evaluated with the JudgeAgent, we report the distribution of PASS, UNVERIFIED, FAIL, and UNCERTAIN verdicts. In particular, we refer to the fraction receiving PASS or UNVERIFIED as the judge-supported rate. This metric measures compliance with the JudgeAgent’s evidence policy and is not interpreted as human-validated accuracy.

### 4.3 Evaluation Results

#### Grocery and Alcohol.

On the fully human-labeled Grocery and Alcohol dataset, the ScoutAgent achieved 98.2% extraction accuracy at 74.7% attribute coverage.

We additionally use the human labels to analyze JudgeAgent behavior on the values produced by the ScoutAgent. Unpopulated attributes are excluded because only proposed values enter the production verification and write gate. During early development, the JudgeAgent assigned each proposed value either PASS or FAIL. This binary formulation grouped together several distinct reasons for withholding approval, including explicit contradiction, insufficient evidence for verification, and conflicting or ambiguous evidence.

Among values assigned PASS, 98.4% were confirmed correct by human reviewers. Of the disagreements between the JudgeAgent and human reviewers, 87.8% were false rejections — values assigned FAIL but judged correct by humans — whereas 12.2% were false acceptances. The binary policy therefore achieved high precision among approved values but lower recall on correct values, reflecting an overly conservative rejection policy.

This asymmetry motivated the current four-verdict taxonomy. The evidence requirement for PASS remains unchanged, while the former rejection outcome is divided into FAIL for contradicted or unsupported values, UNVERIFIED for plausible values that cannot be directly confirmed, and UNCERTAIN for cases with conflicting or ambiguous evidence. Given the high precision observed among approved values, we use the current JudgeAgent as a scalable operational audit signal and report judge-based results separately from human-validated accuracy.

#### Electronics and Home Improvement.

On the Electronics and Home Improvement dataset, the ScoutAgent achieved 87.8% attribute coverage. Of the extracted values, 97.4% received a PASS or UNVERIFIED verdict from the JudgeAgent.

### 4.4 LLM Backbone Comparison

We compare three alternative VLM backbones for the ScoutAgent against the Gemini 2.5 Flash baseline, while holding the JudgeAgent and all other pipeline components fixed.

*Table 1: ScoutAgent backbone comparison on the Electronics and Home Improvement dataset, with the JudgeAgent fixed to Gemini 2.5 Flash. Judge-supported rate is the fraction of all extracted values receiving a PASS or UNVERIFIED verdict; UNCERTAIN and invalid responses remain in the denominator. Publication coverage is the fraction of requested attributes receiving one of these two verdicts. Costs are normalized to Gemini 2.5 Flash.*

| VLM backbone | Judge-supported rate | Attribute coverage | Publication coverage | Relative cost |

$1.00\times$| Gemini 2.5 Flash | 97.4% | 87.8% | 85.5% | |

$7.21\times$| Gemini 3.5 Flash | 92.7% | 88.3% | 81.9% | |

$1.93\times$| GPT-5.4 | 87.1% | 84.0% | 73.2% | |

$3.05\times$| Claude Sonnet 5 | 78.9% | 80.5% | 63.5% | |

As shown in Table 1, Gemini 2.5 Flash provides the strongest overall quality–coverage–cost trade-off. Although Gemini 3.5 Flash increases extraction coverage by $0.5$ percentage points, its lower judge-supported rate reduces publication coverage from 85.5% to 81.9%, while increasing inference cost by more than $7\times$. GPT-5.4 and Claude Sonnet 5 achieve still lower publication coverage, at 73.2% and 63.5%, respectively.

Error analysis shows that the lower judge-supported rates of the alternative backbones arise primarily from unsupported or partial extractions rather than explicit contradictions or hallucinations. These errors are concentrated in evidence-intensive attributes, including unit count and free-text descriptions. The results therefore show that backbone choice affects not only how often the ScoutAgent extracts a value, but also how often that value is sufficiently grounded for automatic publication.

Because all ScoutAgent variants are evaluated using the same fixed JudgeAgent, these results provide a controlled comparison of operational behavior rather than estimates of human-validated accuracy.

## 5 Deployment

We deployed TRACE in production and enriched 31 million SKUs across four business verticals. This increased impression-weighted enrichment coverage, defined as the share of customer impressions associated with product records carrying enriched attributes, by over 90% across these verticals.

*Figure 2: Illustrative product detail page (PDP) before and after catalog enrichment. Before enrichment, attributes were buried in unstructured text, making it difficult for shoppers to make a confident purchasing decision. TRACE generates structured attribute data using both internal and external data source.*

### 5.1 User Impact

Product Detail Page (PDP) presents the information shoppers use to evaluate a product before adding it to their cart or completing a purchase. When catalog attributes are missing, shoppers may have difficulty determining whether a product satisfies their needs or may form expectations that do not match the item ultimately received. We hypothesize that surfacing enriched attributes on the PDP reduces this information gap, leading to more confident purchase decisions and fewer post-purchase issues.

We evaluate this hypothesis through a five-week randomized A/B test with 90% of traffic assigned to treatment and 10% to holdout. In the treatment, shoppers were shown PDPs augmented with the attributes produced by TRACE; the holdout retained the existing PDP experience. An example PDP before and after the experiment is shown in Figure 2. The experiment therefore measures the end-to-end user impact of generating, validating, and surfacing enriched catalog information.

*Table 2: Online A/B test of surfacing enriched product detail pages (PDPs). Enriched PDPs improved shopping outcomes by increasing checkout conversion, with larger gains among power users, and reducing the rate of missing or incorrect items. Effects are reported as relative changes versus the control group, with 95% confidence intervals and $p$-values.*

| Metric | Effect (rel.) | 95% CI | p-value |

| Checkout conversion | +0.48% | [+0.04%, +0.92%] | 0.034 |

| Checkout conversion (power users) | +1.18% | [+0.24%, +2.12%] | 0.014 |

| Missing/incorrect item rate | −1.08% | [−2.04%, −0.13%] | 0.026 |

As shown in Table 2, the enriched PDP increased checkout conversion by $0.48\%$, with a larger $1.18\%$ increase among power users. It also reduced the missing/incorrect-item rate by $1.08\%$. These results are consistent with the hypothesis that richer product information helps shoppers make more informed purchase decisions and form more accurate expectations about the items they order.

## 6 Limitations

The Electronics and Home Improvement results use judge-supported rate as a scalable operational metric rather than as a substitute for human-validated precision. The JudgeAgent was calibrated on the Grocery and Alcohol human audit, while its transfer to other categories has received more limited human evaluation. Moreover, because the ScoutAgent and JudgeAgent use models from the same family in the primary configuration, they may exhibit correlated failure modes.

The online experiment evaluates the end-to-end effect of displaying enriched product pages. It therefore demonstrates the value of the deployed system as a whole, but does not isolate the contribution of the JudgeAgent or write-gating policy.

## 7 Conclusion

We presented TRACE, a multi-agent framework for evidence-grounded catalog attribute enrichment. TRACE separates candidate generation from verification: a ScoutAgent gathers and reconciles evidence from multiple sources, while a JudgeAgent applies a stricter evidence policy before proposed values are written to the catalog.

Experiments across multiple business verticals demonstrate that TRACE produces high-quality attribute values and can operate at production scale. On the human-annotated dataset, TRACE achieved 98.2% accuracy. A randomized online experiment showed that surfacing enriched attributes on product detail pages increased checkout conversion by 0.48% and reduced missing or incorrect item reports.

These findings show that evidence-grounded catalog enrichment can improve both catalog quality and downstream user experience. More broadly, they highlight the importance of grounding generated attribute values in product-specific evidence and verifying them before publication.

## Declaration on Generative AI

During the preparation of this work, the author(s) used generative AI tools in order to: Grammar and spelling check; and Paraphrase and reword. After using these tool(s)/service(s), the author(s) reviewed and edited the content as needed and take(s) full responsibility for the publication’s content.

## References

- Brinkmann et al. (2023) A. Brinkmann, R. Shraga, and C. Bizer ExtractGPT: exploring the potential of large language models for product attribute value extraction. Note: arXiv:2310.12537 External Links: 2310.12537 Cited by: §1.

- Comanici et al. (2025) G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, L. Marris, S. Petulla, C. Gaffney, A. Aharoni, N. Lintz, T. C. Pais, H. Jacobsson, I. Szpektor, N. Jiang, …, and W. Helmholz Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. External Links: 2507.06261, Link Cited by: §4.

- Dong et al. (2020) X. L. Dong, X. He, A. Kan, X. Li, Y. Liang, J. Ma, Y. E. Xu, C. Zhang, T. Zhao, G. Blanco Saldana, S. Deshpande, A. M. Manduca, J. Ren, S. P. Singh, F. Xiao, H. Chang, G. Karamanolakis, Y. Mao, Y. Wang, C. Faloutsos, A. McCallum, and J. Han AutoKnow: self-driving knowledge collection for products of thousands of types. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), External Links: Document, 2006.13473 Cited by: §1.

- Es et al. (2024) S. Es, J. James, L. Espinosa-Anke, and S. Schockaert RAGAs: automated evaluation of retrieval augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL): System Demonstrations, pp. 150–158. External Links: Document Cited by: §2.

- Huang and Caragea (2025) W. Huang and C. Caragea MADIAVE: multi-agent debate for implicit attribute value extraction. Note: arXiv:2510.05611 External Links: 2510.05611 Cited by: §2.

- Khandelwal et al. (2023) A. Khandelwal, H. Mittal, S. S. Kulkarni, and D. K. Gupta Large scale generative multimodal attribute extraction for e-commerce attributes. Note: arXiv:2306.00379 External Links: 2306.00379 Cited by: §2.

- Liu et al. (2023) Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu G-Eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2511–2522. External Links: Document Cited by: §2.

- Lu and Wang (2025) Y. Lu and J. Wang KARMA: leveraging multi-agent LLMs for automated knowledge graph enrichment. Note: arXiv:2502.06472 External Links: 2502.06472 Cited by: §2.

- Nigam et al. (2019) P. Nigam, Y. Song, V. Mohan, V. Lakshman, W. Ding, A. Shingavi, C. H. Teo, H. Gu, and B. Yin Semantic product search. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), External Links: Document, 1907.00937 Cited by: §1.

- Saad-Falcon et al. (2024) J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia ARES: an automated evaluation framework for retrieval-augmented generation systems. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Note: arXiv:2311.09476 External Links: 2311.09476 Cited by: §2.

- Shi et al. (2025) Y. Shi, T. Yang, C. Chen, Q. Li, T. Liu, X. Li, and N. Liu SearchRAG: can search engines be helpful for LLM-based medical question answering?. Note: arXiv:2502.13233 External Links: 2502.13233 Cited by: §2.

- Sinha and Gujral (2024) A. Sinha and E. Gujral PAE: LLM-based product attribute extraction for e-commerce fashion trends. Note: arXiv:2405.17533 External Links: 2405.17533 Cited by: §1, §2.

- Verga et al. (2024) P. Verga, S. Hofstätter, S. Althammer, Y. Su, A. Piktus, A. Arkhangorodsky, M. Xu, N. White, and P. Lewis Replacing judges with juries: evaluating LLM generations with a panel of diverse models. Note: arXiv:2404.18796 External Links: 2404.18796 Cited by: §2.

- Wang et al. (2019) X. Wang, X. He, Y. Cao, M. Liu, and T. Chua KGAT: knowledge graph attention network for recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), External Links: Document, 1905.07854 Cited by: §1.

- Wen et al. (2025) B. Wen, J. Yao, S. Feng, C. Xu, Y. Tsvetkov, B. Howe, and L. L. Wang Know your limits: a survey of abstention in large language models. External Links: 2407.18418, Link Cited by: §3.1.

- Yang et al. (2022) L. Yang, Q. Wang, Z. Yu, A. Kulkarni, S. Sanghai, B. Shu, J. Elsas, and B. Kanagal MAVE: a product dataset for multi-source attribute value extraction. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining (WSDM), pp. 1256–1265. External Links: Document, 2112.08663 Cited by: §1.

- Yao et al. (2023) S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao ReAct: synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations (ICLR), Note: arXiv:2210.03629 External Links: 2210.03629 Cited by: §2, §3.2.

- Zhang et al. (2025) B. Zhang, S. A. Khan, and S. Walter Leveraging product catalog patterns for multilingual e-commerce product attribute prediction. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 267–275. External Links: Document Cited by: §2.

- Zhang et al. (2021) D. Zhang, Z. Li, T. Cao, C. Luo, T. Wu, H. Lu, Y. Song, B. Yin, T. Zhao, and Q. Yang QUEACO: borrowing treasures from weakly-labeled behavior data for query attribute value extraction. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM), External Links: Document, 2108.08468 Cited by: §1.

- Zheng et al. (2018) G. Zheng, S. Mukherjee, X. L. Dong, and F. Li OpenTag: open attribute value extraction from product profiles. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), External Links: Document Cited by: §1.

- Zheng et al. (2023) L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track, Note: arXiv:2306.05685 External Links: 2306.05685 Cited by: §2.

- Zhu et al. (2020) T. Zhu, Y. Wang, H. Li, Y. Wu, X. He, and B. Zhou Multimodal joint attribute prediction and value extraction for E-commerce product. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2129–2139. External Links: Document Cited by: §1.

## Appendix A Condensed Agent Prompt Templates

The following provider-neutral templates reproduce the instruction contracts used in the final configuration. They are condensed rather than verbatim: repeated prose, concrete ontology values, SKU content, and provider-specific tool schemas are omitted for space. Angle-bracketed fields are populated for each SKU. TRACE makes one ScoutAgent call and one JudgeAgent call per eligible SKU; each call returns a map of per-attribute outputs.

*Figure 3: Condensed ScoutAgent prompt template for the final TRACE configuration. Provider-specific message wrappers, tool schemas, concrete ontology values, and SKU data are represented by placeholders.*

*Figure 4: Condensed JudgeAgent prompt template for the final TRACE configuration. Provider-specific message wrappers, tool schemas, concrete ontology values, and SKU data are represented by placeholders.*

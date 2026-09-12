<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2608.27006
     paper_id : p2s-2026-0014
     source   : https://arxiv.org/html/2608.27006v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval

Conference: 20th ACM Conference on Recommender Systems; September 27-October 02, 2026; Minneapolis, MN, USA20th ACM Conference on Recommender Systems (RecSys ’26), September 27-October 02, 2026, Minneapolis, MN, USADOI: 10.1145/3773078.3841297ISBN: 979-8-4007-2284-4/2026/09CCS: Information systems Recommender systemsCCS: Information systems Information retrieval
Ante Kapetanović Affiliation: Infobip, Split, Croatia email: ante.kapetanovic@infobip.com , Tomislav Đuričić Affiliation: Infobip, Zagreb, Croatia email: tomislav.duricic@infobip.com , Dionizije Fa Affiliation: Infobip, Osijek, Croatia email: dionizije.fa@infobip.com , Andro Merćep Affiliation: Infobip, Zagreb, Croatia email: andro.mercep@infobip.com and Emanuel Lacić Affiliation: Infobip, Zagreb, Croatia email: emanuel.lacic@infobip.com

© cc

###### Abstract.

Conversational recommender systems based on large language models (LLMs) are usually evaluated on static, pre-indexed item collections, yet e-commerce catalogues change continuously as products are added or removed, repriced, and restocked. We present a merchant-agnostic, multi-turn conversational shopping assistant that operates over such live catalogues. Its central component is a self-refreshing retriever that ingests a merchant product feed, enriches the records, and synchronizes them into a vector index. On each run, per-item hashes identify which products are new, changed, deleted, or unchanged, so only the delta is processed rather than rebuilding the whole catalogue. A controller-based dialogue layer consumes this index, using an LLM only for intent classification and preference elicitation while retrieval, reranking, and diversity selection run as dedicated functions. Our demonstration is a WhatsApp shopping assistant in which catalogue changes reach the recommendations after the next successful sync. A live chatbot, documentation, and a recorded walkthrough are available at [https://github.com/infobip/infobip-agentic-crs].

###### Keywords:

Generative Conversational Recommendation Systems, Incremental Indexing, E-Commerce

## 1. Introduction

Most large language model (LLM)-based conversational recommender systems (CRSs) are evaluated over fixed benchmark collections (He et al., 2023; Jannach et al., 2021), but in production the catalogue is a live object, continually updated. Re-indexing the whole catalogue on every change is wasteful, yet letting the index drift degrades recommendations and surfaces out-of-stock or discontinued items.

Our work sits within LLM-era CRS (Jannach et al., 2021; Kolb et al., 2025): retrieval-augmented CRSs ground recommendations in an item corpus (Yang and Chen, 2024; Kemper et al., 2024; Lewis et al., 2020), agentic designs give the LLM tools and control flow (Huang et al., 2025; Yao et al., 2023; Schick et al., 2023), and memory-enhanced systems enrich dialogue context (Xi et al., 2024), but deployed LLM-driven CRSs remain rare (Kunstmann et al., 2026). Our emphasis is orthogonal to model quality. We treat catalogue freshness—keeping the index consistent with a live assortment—as the engineering problem that makes such systems production-viable, complementing work on adapting LLM recommenders to refreshed indices (He et al., 2025). The engine is merchant-agnostic, reusing an existing product feed.

*Figure 1. Example multi-turn WhatsApp interaction showing preference elicitation and product recommendation.Screenshot of a multi-turn WhatsApp chat in which the assistant asks clarifying questions about the shopper's preferences and then replies with a short list of recommended products and links.*

We demonstrate a conversational shopping assistant built around one contribution: a self-refreshing retriever that re-embeds only new or semantically changed products, keeping synchronization proportional to the changed subset. A dialogue pipeline consumes the refreshed index independently of the vector store, model provider, and channel, reaching WhatsApp through Infobip Answers.

## 2. System Overview

The engine has three subsystems (Appendix A): a catalogue pipeline that ingests and indexes products, a conversation pipeline that handles multi-turn dialogue, and a storage layer, written by the former and read by the latter, providing vector search, user profiles, and session state. This shared storage decouples catalogue synchronization from dialogue, allowing each pipeline to run independently. All generative, embedding, and reranking calls use a single proxy, making model choices configuration rather than code. Our proof of concept uses ChromaDB through a swappable VectorStore interface.

### 2.1. Self-Refreshing Retriever

The retriever refreshes a searchable vector index from a merchant product feed. Each manual or scheduled run compares the latest catalogue snapshot with the index and applies only the difference; it does not monitor the feed continuously.

#### Fetch and parse.

A run reuses a fresh cached feed; otherwise, it streams XML to disk and retries failed downloads with exponential backoff, deleting an incomplete file after the final failure. An incremental parser bounds memory use, strips HTML, normalises prices, and converts availability to a boolean.

#### IDs, hashes, and embeddings.

A stable product ID links snapshots and drives exact updates and deletions, but cannot answer natural-language queries. A full hash detects any feed-field change; a semantic hash over name, description, brand, and category identifies changes requiring re-embedding. The resulting vectors make products retrievable; the generative LLM is only an enrichment fallback.

#### Change classes.

Comparing IDs and hashes yields five disjoint classes. New and semantically changed records are enriched, embedded, and upserted; enrichment resolves category paths and extracts attributes by rule, with generative fallback. Metadata-only changes, such as price or stock, retain the vector while updating the record and filters. Deleted records are removed, and unchanged records are skipped.

### 2.2. Conversational Pipeline

The conversation pipeline follows an orchestrator-as-controller pattern (Yao et al., 2023; Schick et al., 2023; Huang et al., 2025): a generative model classifies messages into eight intents, composes replies, and uses an elicitor sub-agent to ask one to three clarifying questions when preferences are vague (Shimazu, 2001; Sun and Zhang, 2018). Recommendation uses content-based semantic retrieval: query and product text share one embedding space, metadata filters restrict candidates, an optional non-generative model reranks them (Yang and Chen, 2024; Kemper et al., 2024), and a greedy selector adds brand and category variety. This generation-free path keeps cost predictable, although embedding and reranking still call external models (Kolb et al., 2025); the pipeline detects the user’s language, retrieves in English, and replies in that language.

## 3. Demonstration

Users access the live demonstration on any smartphone through WhatsApp, with no application to install (Figure 1). Sessions may be anonymous or personalised from prior purchases; the assistant elicits preferences, searches the live catalogue, and returns diverse in-stock products with links, and each successful sync exposes catalogue changes.

*Table 1. Incremental synchronization of an anonymized 500-record catalogue (medians over three or five runs; full rebuild: 2.914 s). ✓/✗ indicate enrichment (E), embedding (Emb), metadata update (M), and deletion (D).Rows show five catalogue change scenarios. Columns indicate whether each scenario triggers enrichment, embedding, metadata update, or deletion, followed by runtime and percentage of full-rebuild time.*

| Change | E | Emb | M | D | Time (s) | Full (%) |

| None | ✗ | ✗ | ✗ | ✗ | 0.053 | 1.8 |

| Add product | ✓ | ✓ | ✗ | ✗ | 0.321 | 11.0 |

| Price/stock | ✗ | ✗ | ✓ | ✗ | 0.072 | 2.5 |

| Description/category | ✓ | ✓ | ✗ | ✗ | 0.357 | 12.3 |

| Delete product | ✗ | ✗ | ✗ | ✓ | 0.062 | 2.1 |

Table 1 confirms the intended change classification: price or stock changes update only metadata, description or category changes re-enrich and re-embed the record, and all ID, hash, feed-field, and embedding-call checks passed.

The prototype accepts Google Merchant Center Atom feeds directly; other catalogue sources or messaging channels require adapters to the product schema or engine API, while filtering, batching, consistency, and operations remain backend concerns behind the VectorStore interface. Synchronization rejects parser failures, invalid or duplicate IDs, and incompatible index versions, and rejects empty or incomplete snapshots to prevent inferred mass deletion; it prepares enrichment and embeddings before writing, then applies upserts and metadata updates before deletions. Writes are non-transactional, so production requires truncated-feed detection, staging or rollback, and abnormal-delta monitoring; we make no tens-of-millions-scale claim. Documentation, engine excerpts, a demo video, and a runnable synthetic sync are public; the full engine, catalogue, and Infobip integration remain private.

## 4. Concluding Remarks

We presented a conversational shopping assistant whose self-refreshing retriever keeps a vector index consistent with a live catalogue, processing only new, changed, or removed products on each sync; a measured case study confirms the classification and operation counts behave as intended, and the assistant is deployed as a WhatsApp demo. The evaluation covers synchronization not ranking quality; an offline relevance study and a live user study of recommendation quality and cost remain future work, alongside freshness-aware retrieval and hybrid ranking.

###### Acknowledgements.

This research was supported in part by the project Infobip Global Communication Platform (PK.1.1.07.0001), part of the Important Project of Common European Interest on Next Generation Cloud Infrastructure and Services (IPCEI-CIS) consortium. Generative AI tools were used in a supporting capacity: Claude Code (Opus 4.7) for code implementation and data analysis, Chat-GPT 5.5 for LaTeX editing and grammar checking. All AI-assisted outputs were reviewed and approved by the authors, who take full responsibility for the content of this publication.

*Figure 2. Engine architecture. Top: the self-refreshing retriever turns a merchant product feed into a vector index. It fetches and parses the feed, then classifies each item by its stable ID and its full and semantic content hashes. New and semantically changed items are enriched, embedded, and upserted. Metadata-only changes update the stored record while keeping its vector. Deleted items are removed, and unchanged items are skipped. Bottom: the conversation pipeline serves shoppers over WhatsApp. The orchestrator classifies intent and dispatches either to the elicitor or to the recommend path (search $\to$ rerank $\to$ diversity), calling a generative model only for intent and elicitation. Center: a shared storage layer (vector store behind a VectorStore interface, user profiles, sessions) is written by the retriever and read by the conversation pipeline.Block diagram of the engine's three subsystems. The top band shows the self-refreshing retriever as a pipeline: product feed, fetch and cache, streaming parse, and a classify stage that compares each item's stable ID and its full and semantic hashes. New and semantically changed items continue through enrich and embed stages and are upserted into storage. Metadata-only changes update the stored record and keep its vector. Deleted items are removed and unchanged items are skipped. The center band is a shared storage layer holding the vector store, user profiles, and session state, written by the retriever and read by the conversation pipeline. The bottom band shows the conversation pipeline: a user on WhatsApp talks to an orchestrator that dispatches to an elicitor or to the search, rerank, and diversity recommend path. Enrichment fallback, embedding, and reranking calls route through a single proxy.*

## Appendix A Engine Architecture

Figure 2 details the engine’s three subsystems and how they share state. The self-refreshing retriever (top) turns a merchant feed into a vector index, classifying each item by its stable ID and its full and semantic hashes so that only new or semantically changed items are enriched and embedded. The conversation pipeline (bottom) serves WhatsApp shoppers through an orchestrator that dispatches to the elicitor or to the recommend path (search $\to$ rerank $\to$ diversity). Both meet at a shared storage layer, holding vectors, profiles, and sessions (retriever writes, conversation pipeline reads).

## References

- He et al. (2023) Z. He, Z. Xie, R. Jha, H. Steck, D. Liang, Y. Feng, B. P. Majumder, N. Kallus, and J. McAuley Large language models as zero-shot conversational recommenders. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM ’23), New York, NY, USA, pp. 720–730. Cited by: §1.

- He et al. (2025) Z. He, Z. Xie, H. Steck, D. Liang, R. Jha, N. Kallus, and J. McAuley Reindex-then-adapt: improving large language models for conversational recommendation. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining (WSDM ’25), New York, NY, USA, pp. 866–875. Cited by: §1.

- Huang et al. (2025) X. Huang, J. Lian, Y. Lei, J. Yao, D. Lian, and X. Xie Recommender AI agent: integrating large language models for interactive recommendations. ACM Transactions on Information Systems 43 (4). Cited by: §1, §2.2.

- Jannach et al. (2021) D. Jannach, A. Manzoor, W. Cai, and L. Chen A survey on conversational recommender systems. ACM Computing Surveys 54 (5). Cited by: §1, §1.

- Kemper et al. (2024) S. Kemper, J. Cui, K. Dicarlantonio, K. Lin, D. Tang, A. Korikov, and S. Sanner Retrieval-augmented conversational recommendation with prompt-based semi-structured natural language state tracking. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, New York, NY, USA, pp. 2786–2790. Cited by: §1, §2.2.

- Kolb et al. (2025) T. E. Kolb, A. Wagne, A. Banerjee, F. Nazary, J. Neidhardt, Y. Deldjoo, and T. Di Noia A tutorial on recent advances in generative conversational recommender systems. In Proceedings of the 19th ACM Conference on Recommender Systems, New York, NY, USA, pp. 1420–1422. Cited by: §1, §2.2.

- Kunstmann et al. (2026) H. Kunstmann, J. Ollier, J. Persson, and F. von Wangenheim EventChat: implementation and user-centric evaluation of a large language model- driven conversational recommender system for exploring leisure events in an SME context. ACM Transactions on Recommender Systems Just Accepted. Cited by: §1.

- Lewis et al. (2020) P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, Vol. 33, Red Hook, NY, USA, pp. 9459–9474. Cited by: §1.

- Schick et al. (2023) T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom Toolformer: language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, Vol. 36, Red Hook, NY, USA, pp. 68539–68551. Cited by: §1, §2.2.

- Shimazu (2001) H. Shimazu ExpertClerk: navigating shoppers’ buying process with the combination of asking and proposing. In Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI ’01), Vol. 2, San Francisco, CA, USA, pp. 1443–1450. Cited by: §2.2.

- Sun and Zhang (2018) Y. Sun and Y. Zhang Conversational recommender system. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR ’18), New York, NY, USA, pp. 235–244. Cited by: §2.2.

- Xi et al. (2024) Y. Xi, W. Liu, J. Lin, B. Chen, R. Tang, W. Zhang, and Y. Yu MemoCRS: memory-enhanced sequential conversational recommender systems with large language models. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), New York, NY, USA, pp. 2585–2595. Cited by: §1.

- Yang and Chen (2024) T. Yang and L. Chen Unleashing the retrieval potential of large language models in conversational recommender systems. In Proceedings of the 18th ACM Conference on Recommender Systems (RecSys ’24), New York, NY, USA, pp. 43–52. Cited by: §1, §2.2.

- Yao et al. (2023) S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao ReAct: synergizing reasoning and acting in language models. Cited by: §1, §2.2.

<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2504.17304
     paper_id : 2504.17304
     source   : paper2skills-vault/papers/07-NLP-VOC/2504.17304/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

arXiv:2504.17304v1 [cs.IR] 24 Apr 2025

You Are What You Bought: Generating Customer Personas for E-commerce Applications Yimin Shi

Yang Fei

Shiqi Zhang∗

yiminshi@u.nus.edu National University of Singapore Singapore, Singapore

yfei11@u.nus.edu National University of Singapore Singapore, Singapore

shiqi@pyrowis.ai National University of Singapore PyroWis AI Singapore, Singapore

Haixun Wang

Xiaokui Xiao

haixun@gmail.com EvenUp San Francisco, United States

xkxiao@nus.edu.sg National University of Singapore Singapore, Singapore

Abstract

CCS Concepts

In e-commerce, user representations are essential for various applications. Existing methods often use deep learning techniques to convert customer behaviors into implicit embeddings. However, these embeddings are difficult to understand and integrate with external knowledge, limiting the effectiveness of applications such as customer segmentation, search navigation, and product recommendations. To address this, our paper introduces the concept of the customer persona. Condensed from a customer’s numerous purchasing histories, a customer persona provides a multi-faceted and human-readable characterization of specific purchase behaviors and preferences, such as Busy Parents or Bargain Hunters.
This work then focuses on representing each customer by multiple personas from a predefined set, achieving readable and informative explicit user representations. To this end, we propose an effective and efficient solution GPLR. To ensure effectiveness, GPLR leverages pre-trained LLMs and few-shot learning to infer personas for customers. To reduce overhead, GPLR applies LLMbased labeling to only a fraction of users and utilizes a random walk technique to predict personas for the remaining customers. To further enhance efficiency, we propose an approximate solution called RevAff for this random walk-based computation. RevAff provides an absolute error 𝜖 guarantee while improving the time complexity 

• Information systems → Electronic commerce; Recommender systems; Data mining; Personalization; • Computing methodologies → Natural language generation; • Mathematics of computing → Graph algorithms; Approximation algorithms.

𝜖 · |𝐸 |𝑁

of the exact solution by a factor of at least 𝑂 |𝐸 |+𝑁 log 𝑁 , where 𝑁 represents the number of customers and products, and 𝐸 represents the interactions between them. We evaluate the performance of our persona-based representation in terms of accuracy and robustness for recommendation and customer segmentation tasks using three real-world e-commerce datasets. Most notably, we find that integrating customer persona representations improves the state-of-the-art graph convolution-based recommendation model by up to 12% in terms of NDCG@K and F1-Score@K.
∗ Shiqi Zhang is the corresponding author.

This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’25, Padua, Italy © 2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1592-1/2025/07 https://doi.org/10.1145/3726302.3730118

Keywords Persona, Large Language Model, Random Walk, Recommendation ACM Reference Format:
Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang, and Xiaokui Xiao. 2025.
You Are What You Bought: Generating Customer Personas for E-commerce Applications. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’25), July 13–18, 2025, Padua, Italy. ACM, New York, NY, USA, 14 pages. https://doi.org/10.
1145/3726302.3730118

1

Introduction

Large Language Models (LLMs), such as GPT-4 [1], hold substantial potential for various e-commerce applications. For example, in digital bookstores, LLMs can recommend books tailored to a customer’s specific interests in various genres and authors, while also explaining the rationale behind each recommendation. Furthermore, LLMs can comprehend literary nuances and character relationships within each book, aiding in the construction of a knowledge base. When combined with retrieval-augmented generation techniques, LLMs can leverage this knowledge base to deliver more personalized and detailed product QA for the books.
The aforementioned applications of LLMs, however, require explicit customer representations that characterize the customers in natural language (e.g., “Bargain Hunter,” “Audiobook Listeners”), so as to provide LLMs with useful information regarding potential customer preferences. In contrast, existing e-commerce recommendation methods (e.g., [13, 30, 35]) predominantly rely on implicit customer representations. In particular, these methods map each customer’s personal data (e.g., browsing histories, clicks, and purchases) to a fixed-size numeric vector, which cannot be easily understood by LLMs or even human experts. One possible solution is to conduct user surveys to gather explicit user preferences, but such surveys are time-consuming and unrepeatable, and often result in incomplete data. Some recent studies [19, 32, 33] propose representing customers based on their preferred product categories.

SIGIR ’25, July 13–18, 2025, Padua, Italy

Nevertheless, these representations are insufficiently informative as they fail to capture detailed purchase behaviors.
To address the above deficiencies, we introduce the concept of Personas as a new dimension to describe customers’ purchase behaviors and product preferences. For instance, on an e-commerce platform, customer personas may include Bargain Hunter, Brand Loyalist, Health Enthusiast, and Tech Savvy, among others. Each persona is accompanied by a detailed definition that provides insights into a customer’s purchasing characterization. This allows each persona to be multi-faceted and not confined to product category granularity. For example, a Health Enthusiast, whose core pursuit is achieving and maintaining good health, may have a subtle preference toward different products in the fruits category: kiwis are favored for their rich vitamin content, while high-calorie options such as bananas are avoided. Unlike event-driven purchase intentions that capture a customer’s immediate motivations [32, 33], personas are more durable and reusable, as they often relate to personal interests, habits, and lifestyles that do not frequently change.
Therefore, assigning each customer multiple personas based on her historical purchase behaviors provides a powerful explicit representation that enhances the capabilities of LLMs in e-commerce.
To generate personas for each customer, we propose utilizing LLMs themselves. By instructing LLMs to summarize the customers’ purchase behaviors using natural language, we can effectively map each customer to her persona labels, leveraging the LLMs’ knowledge and reasoning capabilities. However, scalability presents a substantial challenge if we are to repeatedly label numerous users on a real-world e-commerce platform, in response to the continuous update of user engagement data. For an e-commerce platform with 10 million active customers, relabeling all users monthly based on their dynamic purchase statistics with GPT-4 costs approximately 2.4 million dollars per year.
To address this scalability challenge, we propose an efficient and cost-effective solution, dubbed GPLR, for generating customer persona representations on real-world e-commerce platforms. GPLR first leverages LLMs to label a small, carefully selected user set within a limited budget. LLMs are instructed to assign personas for each customer from a platform-specific prototype persona set. Subsequently, GPLR utilizes random walks to analyze purchase behavior similarities among users, based on which it infers the personas for unlabeled users through a weighted aggregation of prototype persona labels. We further propose RevAff to reduce the time complexity of random walk-based computations by approximating exact random walk probabilities with a theoretical error tolerance.
Given that personalized product recommendation is a key downstream application of the explicit customer representation, we describe a methodology for integrating customer personas as an additional partition into existing graph convolution-based methods in Section 6. Based on this methodology, we integrate personas with two state-of-the-art recommendation models [31, 35], and extensively compare them with six cutting-edge recommendation models. Specifically, we find that persona-enhanced models can outperform competitors on three real-world datasets with up to 13 million user-item interactions. Notably, personas can improve upon the original model by up to 12% in NDCG@K and F1-Score@K.
Besides recommendation, we further demonstrate that personas outperform existing explicit customer representations in customer

Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang and Xiaokui Xiao

segmentation in terms of robustness and cluster quality. In addition, extensive experiments show that RevAff can process large-scale datasets in tens of seconds, achieving empirical errors significantly lower than the theoretical bound.
To summarize, we make the following contributions:
• We propose representing customers through personas, enabling the integration of external agents’ knowledge and reasoning.
• We propose GPLR, which significantly reduces the cost of using LLMs when generating effective customer personas.
• We enhance the scalability of GPLR by proposing RevAff to approximate its random walk-related computations efficiently.
• We present a methodology for seamlessly combining personas with graph convolution-based recommendation models.
• We conduct comprehensive experiments to show the superiority of the proposed solutions in product recommendation and customer segmentation.

2

Problem Formulation

This section first introduces the bipartite graph representation of customer purchase histories, and then proposes the concept of personas and defines our representation generation problem, after which we discuss three important downstream applications.

2.1

Preliminaries

Purchase histories. We represent purchase histories of users on an e-commerce platform as a bipartite graph G = (𝑈 , 𝑉 , 𝐸), where 𝑈 and 𝑉 are sets of nodes representing users and products on the e-commerce platform, respectively. The edge set 𝐸 contains the purchase history 𝑒𝑖 𝑗 = (𝑢𝑖 , 𝑣 𝑗 ) ∈ 𝐸, representing that the user 𝑢𝑖 ∈ 𝑈 purchased the product 𝑣 𝑗 ∈ 𝑉 in the past. For an edge 𝑒𝑖 𝑗 = (𝑢𝑖 , 𝑣 𝑗 ) ∈ 𝐸, we say 𝑢𝑖 and 𝑣 𝑗 are neighbors. We denote 𝑁 (𝑢𝑖 )
(resp. 𝑁 (𝑣 𝑗 )) as the neighbor set of 𝑢𝑖 (resp. 𝑣 𝑗 ) and represent its degree as 𝑑𝑖 (resp. 𝑑 𝑗 ).
Notations. Throughout this paper, we denote matrices in bold uppercase, e.g., M. We use M[𝑣𝑖 ] to denote the 𝑣𝑖 -th row vector of M, and M[:, 𝑣 𝑗 ] to denote the 𝑣 𝑗 -th column vector of M. In addition, we use M[𝑣𝑖 , 𝑣 𝑗 ] to denote the element at the 𝑣𝑖 -th row and 𝑣 𝑗 -th column of M. Given an index set 𝑆, we let M[𝑆] (resp. M[:, 𝑆]) be the matrix block of M that contains the row (resp. column) vectors of the indices in 𝑆. Table 1 lists the frequently used notations.

2.2

Customer Personas

The concept persona refers to the social face that an individual presents to the world from a psychological perspective [16]. In this work, we introduce the concept of customer persona for customers on e-commerce platforms. Specifically, each customer persona summarizes a characterization of a customer’s specific purchase behaviors and product preferences, such as Bargain Hunters, Health Enthusiasts, Tech Savvy, and Busy Parents. Additionally, each customer persona is associated with a detailed explanation. For example, the Busy Parents persona describes an individual who frequently purchases kid-friendly products, diapers, baby food, and other family necessities, often seeking convenience through pre-made meals and time-saving products. The Bargain Hunters represents customers who are always looking for the best deals and discounts, buying

You Are What You Bought: Generating Customer Personas for E-commerce Applications

Table 1: Frequently used notations.
Notation G = (𝑈 , 𝑉 , 𝐸 )
𝑅 𝑈𝑟 , 𝑈𝑠 𝜏 Φ Ψ ℓˆ 𝜖

Description A bipartite graph with user node set 𝑈 , item node set 𝑉 and historical purchase edge set 𝐸.
Predefined persona set.
Unlabeled user set and prototype user set.
LLM query budgets and 𝜏 = |𝑈𝑟 |.
Customer representation matrix by personas.
The user-persona affinity matrix in Eq. (4).
Random walk step in Eq. (2).
The absolute error in Definition 2.

sale products with coupons and purchasing in bulk to save money.
Advantages of the customer persona are three-fold.
Informativeness. In contrast to conventional features such as demographic information, the customer persona demonstrates high informativeness by seamlessly integrating multi-dimensional information from the user’s purchase preference. For example, the Busy Parent persona captures the information from the perspectives of baby needs, household, and convenience.
Readability. Although deep learning techniques can condense information into embeddings, such representations often lack humanreadable dimensions that convey clear and understandable messages. In contrast, customer personas, along with their corresponding detailed explanations, are easy for both humans and LLMs to read and comprehend.
Robustness. The customer persona is tied to a user’s values, habits, and living conditions, which do not change frequently. This makes it a suitable choice as a relatively long-standing label for customers.
Notably, in this work, we assume that the e-commerce platform has predefined a set of customer personas, denoted as 𝑅. To establish a set of comprehensive and representative customer personas, the platform can leverage LLMs associated with prompt engineering techniques∗ . Specifically, the service provider supplies the LLM with a portion of real-world customer purchase histories to generate an initial set of persona candidates, which are further refined and deduplicated by the LLM. Additionally, the predefined personas can be validated based on customer feedback on the platform, finally yielding a proprietary persona set for the platform.

2.3

Customer Representation by Personas

To personalize each customer with better informativeness, humanreadability, and robustness, we target to assign each customer multiple personas from the predefined persona set that align with her historical purchase behaviors. We formalize the major problem to solve in this paper as follows.
Goal. Given a bipartite graph G = (𝑈 , 𝑉 , 𝐸) representing user purchase histories and a predefined customer persona set 𝑅, this work focuses on finding a customer representation matrix Φ ∈ {0, 1} |𝑈 | × |𝑅 | . Φ[𝑢𝑖 ] represents the persona representation of user 𝑢𝑖 , where Φ[𝑢𝑖 , 𝑟𝑘 ] = 1 indicates that the customer 𝑢𝑖 possesses the persona 𝑟𝑘 ; otherwise, Φ[𝑢𝑖 , 𝑟𝑘 ] = 0. In other words, the core of this work lies in effectively and efficiently assigning personas to each customer.
∗ All related prompts in this work will be provided in Appendix B.

SIGIR ’25, July 13–18, 2025, Padua, Italy

2.4

Downstream Applications

The persona-based representation can be applied across a range of downstream tasks. For service providers and retailers, personas offer a new and informative perspective for customer segmentation.
For the customer experience, the personas not only serve as filter labels in searches but also facilitate product recommendations. In what follows, we will illustrate these application scenarios in detail.
Customer segmentation [22, 29]. Customer segmentation is the process of dividing customers into distinct groups based on similarities in specific attributes such as demographic features and interests.
The proposed personas offer a valuable method for categorizing customers according to their spending habits and purchasing behavior patterns, thereby enhancing data analytics in advertising, marketing, and customer relationship management. Furthermore, the customer persona set 𝑅 provides 2 |𝑅 | distinct combinations of persona for intersecting or uniting customer groups, allowing analytics to vary from the coarsest to the finest granularity. In addition to enhancing analytics, this persona-based approach also facilitates the design of personalized strategies to increase customer engagement and business profitability.
Customer-centric search navigation [32, 33]. On the e-commerce platform, customers search for products by entering concepts or keywords. In response, the platform returns a list of relevant products along with a navigation bar that enables customers to explore these products based on a taxonomy focused on the catalog and metadata of the products. This taxonomy can be enhanced by incorporating the proposed personas as an additional layer at the highest level, creating a new taxonomy that integrates both customercentric persona concepts and the product-focused hierarchy. For example, the persona Busy Parent could serve as a supernode in the new taxonomy, encompassing three departments: Baby, Health & Household, and Home & Kitchen, based on the Amazon catalog.
This structure allows the taxonomy to reflect both customer preferences and product categories, facilitating a more intuitive and targeted shopping experience.
Product recommendation [13, 31]. Customer representation by personas can enhance the effectiveness of personalized product recommendations. Specifically, these representations can serve as |𝑅| additional input features for well-adopted recommendation models, such as those based on graph neural networks. In contrast to merely considering historical transactions, incorporating personas provides more aggregated insights w.r.t. each customer. Moreover, as comprehensible features, they can be integrated with external knowledge, such as the favorite products of each persona, thereby improving the performance of recommender systems. In addition, personas can also alleviate the cold-starting problem. To explain, new customers can select personas that fit their own profiles when they first join, and retailers of new products can choose personas that align with the target customers for these products. Hence, new items can be recommended to users with matching personas, and vice versa.

3

Related Work

In this section, we briefly review existing works focused on generating implicit and explicit user representations.

SIGIR ’25, July 13–18, 2025, Padua, Italy

Most previous methods represent users as low-dimensional embeddings optimized according to specific objective functions (see Ref. [17] for the latest survey). Among them, numerous solutions [8, 13, 14, 18, 27, 30, 31, 36, 37] are based on collaborative filtering, ranging from traditional matrix factorization solution [18] to the modern approaches based on graph neural networks [7, 9, 13, 27, 31, 34–37].
However, these embeddings are implicit representations, meaningful only within the recommender system, and difficult for humans or LLMs to understand. Similar situations happen in other personalized e-commerce applications, such as customer segmentation [22, 29] and product search [3, 4, 9, 32, 33]. In addition, although some works [5, 6, 10] utilize the concept of personas in their modeling process, they still represent each persona as a set of implicit features instead of a human-readable definition.
Some works attempt explicit customer representation by utilizing the concept of persona, but fundamentally differ from our approach. For example, Li et al. [19] posteriorly assigns each user with a single persona label based on product categories rather than customer-centric characterization. Elad et al. [12] models customer personalities that are not directly related to e-commerce. Additionally, another line of related works, such as FolkScope [33] and COSMO [32], leverage LLMs to construct a commonsense knowledge graph for applications in the e-commerce field. In this graph, the head entity represents a pair of co-buy or view-buy items from the same category, while the tail entity explains the possible reason for this co-occurrence. In other words, this knowledge graph can currently serve only as a textual description for item pairs from the same category. However, it remains an open question how to convert this into an explicit and extensive representation for each item and, further, for each customer.

4

Solution Framework

In this section, we introduce the framework of GPLR, which Generates customers’ Persona representation matrix Φ through leveraging Large language models and Random walk-based affinities.

4.1

Main Idea

Benefiting from the finding that LLMs like GPT-4 are reliable in understanding and answering questions related to the shopping domain [28], a straightforward and effective approach to generate Φ[𝑢𝑖 ] for a user 𝑢𝑖 is to serialize all purchase histories of 𝑢𝑖 and ask a pre-trained LLM which personas 𝑢𝑖 belongs to. However, due to their large-scale model structure and autoregressive generation mechanism, LLMs incur a prohibitively immense computational overhead when generating personas for millions of customers on real-world e-commerce platforms, regardless of whether using online APIs or deploying locally. Furthermore, the need to repeatedly relabel each customer due to the dynamic nature of their personas further amplifies this problem.
To be more cost-effective, we propose the solution GPLR, whose idea is to sample a small fraction of users 𝑈𝑠 as prototype users and generate their customer representations using LLMs, then infer personas for remaining unlabelled users, 𝑈 \ 𝑈𝑠 , based on their proximity to the prototype users within G. The pseudocode of GPLR is illustrated in Algorithm 1. In particular, GPLR takes as inputs the

Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang and Xiaokui Xiao

Algorithm 1: GPLR Input: Purchase histories G, persona set 𝑅, budget constant 𝜏, number of iterations 𝑇 and a cutoff constant 𝑘 Output: Customer representation by personas Φ |𝑈 | × |𝑅 | ; 𝑈 ← ∅;
1 Φ, Ψ ← {0} 𝑠 2 for 𝑡 ← 1 to 𝑇 do 3 𝑈𝑡 ← DUSample(G, Φ, Ψ, 𝜏/𝑇 , 𝑡);
4 Φ[𝑈𝑡 ] ← LLMAnswer(𝑈𝑡 , 𝑅);
5 𝑈𝑠 ← 𝑈𝑠 ∪ 𝑈 𝑡 ;
6 Ψ ← AffinityCompute(G, Φ, 𝑈𝑠 );
For each 𝑢𝑖 ∈ 𝑈 \ 𝑈𝑠 , set each Φ[𝑢𝑖 , 𝑟 𝑤 ] to 1 where 𝑟 𝑤 is among the top-𝑘 personas of Ψ[𝑢𝑖 ];
8 return Φ;

7

purchase histories G, the persona set 𝑅, a budget constant 𝜏 indicating the cardinality of the sampled user set |𝑈𝑠 |, the number of iterations 𝑇 , and a cutoff constant 𝑘. Algorithm 1 initializes Φ and Ψ to zero matrices and 𝑈𝑠 to an empty set. Ψ ∈ R |𝑈 | × |𝑅 | is called the user-persona affinity matrix, which is the core of the subsequent subroutines. (For ease of presentation, we defer the formal definition and computation of Ψ to Section 5.) Intuitively, Ψ[𝑢𝑖 , 𝑟 𝑗 ] is a weighted aggregation of the label associated with 𝑟 𝑗 w.r.t. prototype users in 𝑢𝑖 ’s vicinity. In Lines 2-6, Algorithm 1 repeats over 𝑇 times.
In each iteration 𝑡, it invokes a sampling strategy called DUSample to return a set of users 𝑈𝑡 with |𝑈𝑡 | = 𝜏/𝑇 , followed by a subroutine called LLMAnswer that labels 𝑈𝑡 to personas in 𝑅 based on LLMs.
After that, GPLR updates the prototype user set 𝑈𝑠 and recomputes Ψ based thereon by invoking AffinityCompute. In Line 7, for each remaining user 𝑢𝑖 ∈ 𝑈 \𝑈𝑠 , we select the 𝑘 personas with the largest affinity scores from the latest Ψ[𝑢𝑖 ] as the persona representation of 𝑢𝑖 .

4.2

Subroutine Descriptions

In what follows, we elaborate on the three subroutines DUSample, LLMAnswer, and AffinityCompute in GPLR.
DUSample. We first introduce the Diversity-Uncertainty (DU) sampling approach, dubbed as DUSample, which considers both persona diversity and user uncertainty based on the current persona affinity scores, with the following motivations. From the perspective of diversity, we observe that the distribution of persona labels is often severely biased in real-world datasets, i.e., the majority of users are associated with a small set of dominating personas. For example, in the Instacart dataset where customer representations are labeled from 51 expert-designed personas using LLM, the two most popular personas are assigned to 76.2% and 75.8% of users, respectively, while the 20 least common personas together cover only 7.4% of users. In other words, randomly selecting users for the LLM labeling inherits the bias in the collected labels, leading to false positive associations for the remaining users. To mitigate this bias, we consider labeling users more likely to contain less common personas, ensuring better diversity in the collected persona labels. Regarding uncertainty, similar to the labeling process in active learning [23], we want to label the users who are more uncertain in the following AffinityCompute routine, i.e., having

You Are What You Bought: Generating Customer Personas for E-commerce Applications

higher probabilities of being assigned with wrong persona labels.
Instead of postponing the decision-making, directly labeling these users with LLMs increases the accuracy in expectation.
Based on the aforementioned insights, DUSample randomly selects 𝜏/𝑇 users from 𝑈 in the initial step (𝑡 = 1) to form 𝑈 1 . At each subsequent time step (𝑡 > 1), we compute the DU score for all 𝑢𝑖 ∈ 𝑈𝑟 as follows, and selects the top-(𝜏/𝑇 ) users with the highest DU scores among all unlabeled users.
 ˆ  Í ˆ 𝑄 (𝑟 )
𝑠 (𝑢𝑖 ) = 𝑄 (𝑟𝑚 ) · log 𝑄 (𝑟𝑚 ) − 𝑄𝑖 (𝑟𝑚 ) · log 𝑄𝑖 (𝑟𝑚 ) (1)
𝑖

𝑟𝑚 ∈𝑅

𝑚

In Eq. (1), 𝑄ˆ is defined as the persona label distribution in the currently collected Φ as follows:
Í

𝑄ˆ (𝑟𝑚 ) =

Φ[𝑢𝑖 ,𝑟𝑚 ]

𝑢𝑖 ∈𝑈𝑠 Í

Φ[𝑢𝑖 ,𝑟𝑛 ] ,

𝑢𝑖 ∈𝑈𝑠 ,𝑟𝑛 ∈𝑅

and 𝑄𝑖 represents the normalized user-persona affinity distribution of user 𝑢𝑖 in the current iteration, defined as:
Ψ[𝑢𝑖 ,𝑟𝑚 ]
.
𝑄𝑖 (𝑟𝑚 ) = Í Ψ[𝑢 𝑖 ,𝑟 𝑛 ]
𝑟𝑛 ∈𝑅

The first term of DU score is actually the KL divergence between 𝑄ˆ and 𝑄𝑖 . Selecting users with higher divergence values is expected to enhance the persona diversity in the collected prototype labels.
The intuition is that if 𝑢𝑖 ’s user-persona affinity distribution significantly differs from the currently collected persona distribution, it indicates that 𝑢𝑖 ’s purchase behavior is more likely to differ from the majority of users. The second term represents the entropy of 𝑄𝑖 . Labeling users with high entropy is expected to improve the accuracy of the persona assignment. The intuition here is that a lower entropy suggests that 𝑢𝑖 ’s affinities on different personas are similar, indicating that the persona information about this user is still ambiguous and has high uncertainty in determining her labels.
LLMAnswer. Subsequently, we send these users’ purchase logs and predefined persona set 𝑅 to the LLM and update its association results in Φ. We denote this approach as LLMAnswer, which takes as inputs the bipartite graph G and the persona set 𝑅. Specifically, we first employ serialization to transform each customer’s purchase logs into a natural language format. We then input the transformed result, a predefined persona set 𝑅, instruction prompts, and inprompt examples into the LLM. These in-prompt examples are defined to improve the generation quality and specify the output format. With few-shot learning, the pre-trained LLM can leverage its extensive real-world knowledge and reasoning abilities to associate the user with the relevant personas by referencing her purchased product names and amounts.
AffinityCompute. Due to the update of prototype user set 𝑈𝑠 and their persona representations Φ[𝑈𝑠 ], we conduct the random walkbased solution AffinityCompute to recompute the user-persona affinity matrix Ψ. Based on the homophily principle [11], we assume that a user is likely to share identical personas with other users who exhibit similar purchasing behaviors, referred to as neighbor users.
To compute Ψ, for each user 𝑢𝑖 , our main idea is to identify the labeled neighbor users in the vicinity and generate Ψ[𝑢𝑖 ] according to the aggregation of their persona representations.

SIGIR ’25, July 13–18, 2025, Padua, Italy

5

User-Persona Affinity Computation

This section first describes the original computation process of the user-persona affinity matrix Ψ. It then introduces RevAff as an efficient approximate solution based on reverse updating.

5.1

Exact Solution

To capture structural closeness between users, we define the attention as the mean of probabilities of a random walk, starting from Í ℓ-steps or fewer, i.e., 1ˆ ℓ ≤ ℓˆ 𝜋ℓ (𝑢𝑖 , 𝑢𝑘 ). In 𝑢𝑖 reaching 𝑢𝑘 within ˆ ℓ particular, for a given 𝑢𝑖 , an ℓ-step random walk starts from 𝑢𝑖 , and at each step (≤ ℓ), it first navigates to a random product purchased by the current user node and then randomly jumps to a user node that purchased this product. The attention matrix Π ∈ R |𝑈 | × |𝑈 | is Í Í Πℓ = 1ˆ I · (P · P′ ) ℓ , (2)
Π = 1ˆ ℓ

ℓ ≤ ℓˆ

ℓ

ℓ ≤ ℓˆ

where Πℓ [𝑢𝑖 , 𝑢 𝑗 ] = 𝜋ℓ (𝑢𝑖 , 𝑢 𝑗 ) for every 𝑢𝑖 , 𝑢 𝑗 ∈ 𝑈 . In Eq. (2), P ∈ R |𝑈 | × |𝑉 | and P′ ∈ R |𝑉 | × |𝑈 | are two transition matrices of G, 1 1 where P[𝑢𝑖 , 𝑣 𝑗 ] = |𝑁 (𝑢 and P′ [𝑣 𝑗 , 𝑢𝑖 ] = |𝑁 (𝑣 if 𝑒𝑖 𝑗 ∈ 𝐸, and 𝑖)| 𝑗 )| ′ P[𝑢𝑖 , 𝑣 𝑗 ] = P [𝑣 𝑗 , 𝑢𝑖 ] = 0 otherwise.
After that, we define a matrix L ∈ R |𝑈 | × |𝑅 | to measure the relative importance of each persona w.r.t. each user node. Specifically, !𝛽 ˆ 𝑐Í ·Φ[𝑢𝑖 ,𝑟𝑚 ]
L[𝑢𝑖 , 𝑟𝑚 ] = 𝑚 Φ[𝑢𝑖 ,𝑟𝑛 ] ,

min 𝑄 (𝑟𝑛 )

𝑐𝑚 =

𝑟𝑛 ∈𝑅

𝑟𝑛 ∈𝑅

𝑄ˆ (𝑟𝑚 )

(3)

if 𝑢𝑖 ∈ 𝑈𝑠 is a prototype user and L[𝑢𝑖 , 𝑟𝑚 ] = 0 otherwise. In Eq. (3), a coefficient 𝑐𝑚 and a hyperparameter 𝛽 are considered to deal with the persona distribution bias, such that the signal of minor personas can be emphasized. Finally, the user-persona affinity matrix Ψ is defined as:
Ψ = Π · L, (4)
where Ψ[𝑢𝑖 , 𝑟𝑚 ] represents the user-persona affinity from user 𝑢𝑖 ∈ 𝑈 to persona 𝑟𝑚 ∈ 𝑅.
Based on the aforementioned definition, the exact solution for computing the matrix Ψ involves first calculating the matrices Π and L, and then determining Ψ according to Eq. (4). Within Ψ, a submatrix Ψ[𝑈 \𝑈𝑠 ] is utilized for Algorithm 1. The time complexity of this exact computation is as follows † .
Theorem 1. The time complexity for computing the exact Ψ by Eq. (4) is 𝑂 (|𝐸| · |𝑈 | + ( ℓˆ − 1)|𝑈 | 3 + |𝑅| · |𝑈 | 2 ).
According to Theorem 1, this exact solution fails to handle large G for two main reasons. First, due to the sparsity of L, only attention values to prototype users (i.e., Π[:, 𝑈𝑠 ]) are utilized in computing Ψ, however, 𝑈𝑠 only represents a small fraction of 𝑈 , e.g., 𝜏 = 10%, incurring a large number of excessive computations. Second, due to the high connectivity in G, Πℓ is diminishing rapidly as the step ℓ increases, resulting in a multitude of tiny attention values that can be disregarded.

5.2

Fast Approximation

To address the efficiency issues mentioned above, we propose an approximation method called RevAff. At a high level, for a given persona 𝑟𝑚 , RevAff performs ℓ-step random walk in a reverse † Detailed proofs will be provided in Appendix A.

SIGIR ’25, July 13–18, 2025, Padua, Italy

Algorithm 2: RevAff Input: Purchase histories G, target persona 𝑟𝑚 , de-bias coefficient 𝑐𝑚 , prototype representation matrix Φ, and error tolerance 𝜖 Output: Ψ̂[:, 𝑟𝑚 ]
1 Ψ̂𝑡 [:, 𝑟𝑚 ], q𝑡 ← 0, ∀𝑡 ∈ {2ℓˆ};
2 Ψ̂0 [:, 𝑟𝑚 ], q0 ← L[:, 𝑟𝑚 ] in Eq. (3);
3 while ∃ ˆ 𝑤, ∀𝑡 ≤ 2ℓ,ˆ q𝑡 [ ˆ 𝑤] ≥ 𝜖/2ℓ,ˆ do 4 𝑡, 𝑤ˆ ← arg max𝑡,𝑤ˆ ∈𝑈 ∪𝑉 q𝑡 [ ˆ 𝑤];
5 for 𝑤 ∈ 𝑁 ( ˆ 𝑤) do q [ 𝑤ˆ ]
6 Δ ← |𝑁𝑡 (𝑤 ) | ;
Ψ̂𝑡 +1 [𝑤, 𝑟𝑚 ] ← Ψ̂𝑡 +1 [𝑤, 𝑟𝑚 ] + Δ;
q𝑡 +1 [𝑤] ← q𝑡 +1 [𝑤] + Δ;

7 8 9

q𝑡 [ ˆ 𝑤] ← 0;

10

return Ψ̂[:, 𝑟𝑚 ] ← 1ˆ ℓ

Í

ℓ ≤ ℓˆ Ψ̂2ℓ [𝑈 , 𝑟𝑚 ];

manner from every prototype user to compute Πℓ [:, 𝑈𝑠 ], which avoids the materialization of the entire Πℓ . Due to the existence of massive tiny attention values, RevAff resorts to computing 𝜖approximate user-persona affinities Ψ̂[:, 𝑟𝑚 ] for an input persona 𝑟𝑚 ∈ 𝑅, which is defined as follows.
Definition 2 (𝜖-approximate user-persona affinity). Given an absolute error threshold 𝜖, a persona 𝑟𝑚 , and a prototype user set 𝑈𝑠 , for any 𝑢𝑖 ∈ 𝑈𝑠 , Ψ̂[𝑢𝑖 , 𝑟𝑚 ] is an 𝜖-approximation of Ψ[𝑢𝑖 , 𝑟𝑚 ] if it satisfies |Ψ[𝑢𝑖 , 𝑟𝑚 ] − Ψ̂[𝑢𝑖 , 𝑟𝑚 ]| ≤ 𝜖.
For ease of presentation, we only illustrated the single persona case in the sequel. To compute Ψ̂, we can repeatedly invoke RevAff by taking as an input every persona 𝑟𝑚 ∈ 𝑅.
Given a target persona node 𝑟𝑚 , the main idea of RevAff is to reversely conduct a deterministic graph traversal and compute the estimated Ψ[:, 𝑟𝑚 ]. We use Ψ̂𝑡 [:, 𝑟𝑚 ] ∈ R |𝑈 |+|𝑉 | to record the estimation for every node 𝑤 ∈ 𝑈 ∪𝑉 to 𝑟𝑚 in the hop 𝑡 = 1, 2, ..., 2ℓ,ˆ where a hop means a user node jumps to an item node in a specific step, or vice versa. Following the previous work [20], we have the following recurrence formula for designing RevAff:
  𝑡 = 0 and 𝑤 ∈ 𝑈  L[𝑤, 𝑟𝑚 ],  Í Ψ𝑡 −1 [ ˆ 𝑤,𝑟𝑚 ]
.
, otherwise   𝑤ˆ ∈𝑁 (𝑤 ) |𝑁 (𝑤 ) |  Based on Eq. (2) and Eq. (4), we derive that the target Ψ is given Í by 1ˆ ℓ ≤ ℓˆ Ψ2ℓ . The pseudocode of RevAff is shown in Algorithm 2.
Ψ𝑡 [𝑤, 𝑟𝑚 ] =

ℓ

In Lines 1-2, we first initialize Ψ̂0 and its corresponding temporary vector q0 to zero vectors, and then set Ψ̂0 [𝑢𝑘 , 𝑟𝑚 ] = q0 [𝑢𝑘 , 𝑟𝑚 ] = L[𝑢𝑘 , 𝑟𝑚 ] for all 𝑢𝑘 ∈ 𝑈𝑠 . In subsequent Lines 3-9, we iteratively estimate the desired values of each hop in an asynchronous manner.
Specifically, in each iteration, we select the largest value in all q𝑡 vectors with 𝑡 = 1, 2, ..., 2ℓ,ˆ and denote the corresponding node and ˆ we inhop as 𝑤ˆ and 𝑡, respectively. For every neighbor 𝑤 ∈ 𝑁 (𝑤), q [ 𝑤ˆ ]
crease the value of Ψ̂𝑡 +1 [𝑤, 𝑟𝑚 ] and q𝑡 +1 [𝑤] by a term Δ = |𝑁𝑡 (𝑤 ) | .
After that, we reset q𝑡 +1 [ ˆ 𝑤] to 0 since all its accumulated updates have been propagated backward. Algorithm 2 repeats this process until the values in all q𝑡 vectors with 𝑡 = 1, 2, ..., 2ℓˆ are lower than a

Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang and Xiaokui Xiao

predefined threshold 𝜖/2ℓ.ˆ At last, we assign 1ˆ ℓ

Í

ℓ ≤ ℓˆ Ψ̂2ℓ [𝑈 , 𝑟𝑚 ] to

Ψ̂[:, 𝑟𝑚 ] as the output. The following theorems show the theoretical guarantee and time complexity that RevAff provides.
Theorem 3. Given a persona 𝑟𝑚 and an error 𝜖, Ψ̂[:, 𝑟𝑚 ] generated by RevAff is an 𝜖-approximate user-persona affinity.
Theorem 4. Given a graph G, a persona set 𝑅, and an error 𝜖, the time complexity of estimating Ψ by invoking RevAff from every   1 2 ˆ ˆ ) + |𝐸| , where 𝑁 = |𝑈 | + |𝑉 |.
𝑟𝑚 ∈ 𝑅 is 𝑂 𝜖 ℓ 𝑁 log( ℓ𝑁 By Theorems 1 and 4, RevAff improves the running time of  𝜖 · |𝐸 |𝑁 the exact solution by a factor of 𝑂 |𝐸 |+𝑁 log 𝑁 for ℓˆ = 1 and   3 𝑂 ˆ2 𝜖 ·𝑁 for ℓˆ > 1, demonstrating the efficiency of RevAff.
ℓ ( |𝐸 |+𝑁 log 𝑁 )

6

Product Recommendation

In this section, we first take graph convolution-based collaborative filtering algorithms [13, 35] as an example and review their general framework. We then present our approach that leverages the proposed users’ persona-based representations to enhance these methods for product recommendation.
Graph convolution-based methods. Given the bipartite graph G, these methods represent all user nodes and item nodes as trainable embeddings X ∈ R𝑁 ×𝑑 , with embedding dimension 𝑑. With G’s adjacency matrix A ∈ {0, 1}𝑁 ×𝑁 , where A[𝑢𝑖 , 𝑣 𝑗 ] = A[𝑣 𝑗 , 𝑢𝑖 ] = 1 if 𝑒𝑖 𝑗 ∈ 𝐸 and A[𝑢𝑖 , 𝑣 𝑗 ] = 0 otherwise, the graph convolution process represents nodes as Z = ℎ(A, X), where ℎ is a given graph convolution function. They frequently apply ranking losses for the optimization, e.g., Bayesian Personalized Ranking (BPR) [24], which optimizes each user 𝑢𝑖 to have a higher inner product between Z[𝑢𝑖 ]
and Z[𝑣 𝑗 ], where 𝑣 𝑗 is the item purchased by 𝑢𝑖 . It leads to a user having a similar representation vector with her neighbor users who share many identical purchased products, as well as their other purchased products. In this way, her neighbors’ other purchases will also have a larger chance of being recommended to this user.
Recommend with personas. In addition to the observed purchases in G, persona-based representations provide external information for each user. Leveraging LLM’s real-world knowledge and reasoning abilities, these additional features include deep insights.
Associating personas with their highly interested products allows for a more comprehensive consideration of the candidate items. For example, items with sparse purchases can attract sufficient attention if they align well with certain persona combinations. On the other hand, users with minor personas will avoid overemphasizing the purchases of mainstream users and popular items. Here we propose a straightforward method to integrate persona-based representations with graph convolution-based algorithms. First, we transform the original bipartite graph G to a new tripartite graph G′ = (𝑈 , 𝑉 , 𝑅, 𝐸 ′ ) by adding persona nodes as a new partition. In G′ , the edge set 𝐸 ′ contains the original edges in 𝐸 and the edges between persona nodes and the associated user and item nodes. Specifically, the edge between persona and user nodes can be constructed using the proposed representation Φ. Furthermore, the edges between persona and item nodes can be determined by querying an LLM to identify which personas an item pertains to, leveraging the LLM’s reasoning capabilities and the description of

You Are What You Bought: Generating Customer Personas for E-commerce Applications

Table 2: Statistics of datasets.
Dataset OnlineRetail Instacart Instacart Full

User# 4,297 20,620 206,209

Item# 3,846 41,521 49,677

Interaction# 263,267 1,333,805 13,307,953

Sparsity 98.4070% 99.8442% 99.8701%

each persona. We extend trainable embeddings to include persona ′ nodes as X′ ∈ R𝑁 ×𝑑 and reconstruct the adjacency matrix based on G′ . Without changing the graph convolution function ℎ, the representation vectors on the tripartite graph are given by Z′ . Except for adding an external set of trainable embeddings for persona nodes, only the propagation procedure changes due to modifying the adjacency matrix. This facilitates the easy migration of mainstream graph convolution-based algorithms, such as LGCN [35]
and AFDGCF [31], from the bipartite graph to the tripartite graph with personas by simply replacing the A with A′ , even without explicit modification of the implementation.

7

Experiments

In this section, we evaluate our proposed GPLR, mainly focusing on its application to personalized recommendations on real-world ecommerce datasets. We aim at the following five research questions:
• RQ1: How do persona-based representations improve the effectiveness of the personalized recommendation?
• RQ2: How does the choice of LLM, sampling budget, and random walk length affect GPLR’s performance in the recommendation?
• RQ3: How do persona-based representations outperform existing explicit customer representations in customer segmentation?
• RQ4: How much speedup does the reverse approximate solution yield in affinity computation?
All experiments are conducted on a Linux machine with 12 Intel(R)
Xeon(R) CPU @ 2.20GHz, 52GB of RAM and an NVIDIA L4 (24GB)
GPU.

7.1

Experimental Settings

Dataset description. To evaluate how integrating customer representations by personas improves the e-commerce personalized recommendation, we conduct experiments on the following publicly accessible real-world datasets that vary in size and sparsity:
OnlineRetail, containing sales data from a European retailer in 2010 [2]; Instacart, consisting of randomly sampled 20,620 anonymized users and their purchase data grocery orders in 2017 [15]; Instacart Full, the full version of Instacart dataset. The statistics of these datasets are summarized in Table 2. For each dataset, we randomly select approximately 80% of each user’s historical purchases as the training set to construct G, leaving the remaining 20% as the test set.
Model configurations. To evaluate the performance of personabased representation in the recommendation, we select two stateof-the-art models, LGCN [35] and AFDGCF [31]. Specifically, LGCN is an efficient and scalable graph convolution-based (GCN) model that is capable of processing large graphs with more than tens of millions of interaction edges. AFDGCF improves the recommendation accuracy by adding a de-correlation loss term to GCN models,

SIGIR ’25, July 13–18, 2025, Padua, Italy

including LightGCN [13] and LGCN. We improve LGCN by integrating personas using the procedure explained in Section 6 and name the enhanced model LGCN3. Due to the scalability issue of the default LightGCN model in AFDGCF, we choose to further improve LGCN3 by adding AFDGCF’s de-correlation loss and named this model A-LGCN3. To ensure a fair comparison, we use the same embedding size and fine-tuned the hyperparameters for all improved models and compared baselines respectively.
For GPLR, we set with the cutoff constant to 5, the number of iterations as 10 and employ OpenAI’s GPT-4 API [1] with its default settings to generate Φ and conduct LLMAnswer. For computing the user-persona affinity matrix, we set the random walk length ℓ = 1 and de-bias hyperparameter 𝛽 = 0.5. In addition, we use RevAff with 𝜖 = 0.001 for Instacart Full while using the exact solution for other datasets. All implementation details including datasets, prompt templates, and algorithms are available at: https:
//github.com/Hanc1999/YouAreWhatYouBought.
Evaluation metric. We employ NDCG@K and F1-Score@K as evaluation metrics on the recommendation results with K from {2, 5, 10, 20, 50, 100} to ensure a comprehensive comparison, where DCG@K NDCG@K = IDCG@K and F1-Score@K = 2 × Precision@K×Recall@K Precision@K+Recall@K .
We repeat each experiment three times and report the average performance.

7.2

Recommendation Performance (RQ1)

Overall evaluations. We compare the performance of LGCN3 and A-LGCN3 with six well-adopted baselines: MF [18], GCMC [7], LCFN [34], LightGCN [13], LGCN [35] and AFDGCF [31]. Table 3 presents the main performance results of MF, LightGCN (Light), LGCN, AFDGCF (AFD), LGCN3 and A-LGCN3 in terms of NDCG@K and F1-Score@K, with the best method highlighted in bold. Due to the space constraints, we omit LCFN and GCMC, which perform lower on most metrics than LightGCN. As demonstrated in Table 3, we observe that after integrating LGCN with persona-based customer representations generated by GPLR, LGCN3 significantly improves NDCG@K and F1-Score@K metrics by up to 10.4%, 11.7%, and 8.5% across three real-world datasets. Meanwhile, LGCN3 maintains LGCN’s efficiency and scalability. We further conduct paired t-tests between LGCN and LGCN3 across all three datasets. The average p-values for F1-Score@K and NDCG@K are 0.027 and 0.012, suggesting the improvement is statisticatlly meaningful. After introducing the same de-correlation loss term from the AFDGCF framework to LGCN3, the recommendation performance of A-LGCN3 is further improved on OnlineRetail and Instacart datasets. It outperforms AFDGCF in most metrics by up to 6.4 % without affecting its efficiency. In addition, when processing the largest dataset Instacart Full, LGCN, LGCN3, and A-LGCN3 demonstrate better scalability, whereas LCFN, LightGCN, and AFDGCF exceed the RAM limit.
Case study. In Table 4, the first column lists the test set products for User 156246 (unordered). The following columns display the top-10 recommendations returned by LGCN and LGCN3, ranked in descending order based on the inner product of the corresponding user and item embedding in LGCN. A recommended item is bolded if it appears in the test set. This user’s 39 historical purchases in the training set include nine organic foods, six fruits, six vegetables, four baby foods, and some healthy items like yogurts and chicken

SIGIR ’25, July 13–18, 2025, Padua, Italy

Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang and Xiaokui Xiao

Table 3: Performance evaluation in NDCG@K (N@K) and F1-Score@K (F@K).
OnelineRetail Instacart Instacart Full MF Light LGCN AFD LGCN3 A-LGCN3 MF Light LGCN AFD LGCN3 A-LGCN3 MF Light/AFD LGCN LGCN3 A-LGCN3 N@2 0.2391 0.2801 0.2686 0.2898 0.2933 0.2940 0.1166 0.1477 0.1405 0.1535 0.1570 0.1634 0.1149 OOM 0.1485 0.1540 0.1533 N@5 0.2143 0.2497 0.2356 0.2578 0.2602 0.2549 0.1006 0.1273 0.1208 0.1308 0.1319 0.1357 0.0972 OOM 0.1248 0.1300 0.1296 N@10 0.2104 0.2443 0.2274 0.2489 0.2503 0.2497 0.0916 0.1180 0.1106 0.1205 0.1197 0.1230 0.0879 OOM 0.1126 0.1187 0.1182 N@20 0.2221 0.2575 0.2383 0.2595 0.2577 0.2617 0.0934 0.1198 0.1117 0.1222 0.1206 0.1242 0.0879 OOM 0.1130 0.1210 0.1200 OOM 0.1363 0.1474 0.1463 N@50 0.2612 0.2942 0.2772 0.2978 0.2940 0.2996 0.1150 0.1448 0.1351 0.1476 0.1451 0.1491 0.1052 N@100 0.3011 0.3337 0.3167 0.3346 0.3325 0.3392 0.1389 0.1733 0.1618 0.1761 0.1732 0.1766 0.1268 OOM 0.1634 0.1773 0.1758 F@2 0.0859 0.1080 0.0955 0.1066 0.1052 0.1092 0.0326 0.0415 0.0375 0.0427 0.0411 0.0433 0.0310 OOM 0.0376 0.0396 0.0392 F@5 0.1123 0.1333 0.1207 0.1348 0.1332 0.1349 0.0478 0.0609 0.0564 0.0629 0.0611 0.0628 0.0457 OOM 0.0570 0.0604 0.0599 F@10 0.1225 0.1416 0.1352 0.1436 0.1431 0.1432 0.0560 0.0716 0.0672 0.0730 0.0722 0.0738 0.0539 OOM 0.0684 0.0727 0.0722 F@20 0.1239 0.1398 0.1341 0.1393 0.1406 0.1415 0.0602 0.0752 0.0710 0.0768 0.0758 0.0780 0.0571 OOM 0.0726 0.0779 0.0775 F@50 0.1081 0.1181 0.1159 0.1186 0.1201 0.1210 0.0565 0.0684 0.0644 0.0692 0.0681 0.0701 0.0525 OOM 0.0661 0.0713 0.0709 F@100 0.0886 0.0955 0.0934 0.0945 0.0967 0.0976 0.0470 0.0562 0.0528 0.0568 0.0557 0.0570 0.0434 OOM 0.0545 0.0587 0.0583

breasts. Accordingly, GPLR assigned her with personas Organic Foodie, Health Enthusiast, and Baby Care Provider. LGCN, without persona considerations, mostly recommended popular fruits and vegetables, with all recommendations being among the top 100 most popular out of 41,521 products, including six in the top 20.
However, it ignores the user’s preference for baby foods, resulting in only one product (Banana) overlapping with the test set, which contains six other baby foods. In contrast, with the persona-based representation, LGCN3 prioritizes the baby food category, which is much less popular than categories like fruits and vegetables in this dataset due to data bias. It recommends two relatively popular items in the baby food category and puts them in high-ranking positions, successfully hitting two additional products in the test set.
Furthermore, LGCN3 removed three top-20 popular false positives from LGCN’s results, indicated by the under-wave. In this way, personas help LGCN3 to reduce overemphasis on highly popular items, enabling fairer consideration of other products that align with the user’s personas.
Personas vs. categorical features. To demonstrate the effectiveness of personas compared to traditional categorical features, we introduce LGCNL as another extension of LGCN, which replaces personas with category-based representations [19] and follows the procedure in Section 6. Due to the lack of category information in OnlineRetail, we report LGCNL’s performance on Instacart.
As shown in Table 5, LGCN3 consistently outperforms LGCNL across all NDCG@K and F1-Score@K metrics. For instance, LGCN3 improves upon LGCNL by up to 3.6% in NDCG@K and 2.0% in F1-Score@K. This demonstrates that our generated personas are more informative and effective in describing a customer’s purchase preference, leading to more improvement in the recommendation.

7.3

Ablation Study (RQ2)

LLMs. To explore the performance of LGCN3 with different LLMs, we replace the default GPT-4-Turbo model with Llama-3-70B [26]
in GPLR and report the results on OnlineRetail in Table 6. As shown, there is no significant difference in their performance on NDCG@K and F1-Score@K. Specifically, LGCN3 with GPT-4-Turbo improves the NDCG@K and F1-Score@K on the OnlineRetail dataset by up to 10.4% and 10.3%. LGCN3 with Llama-3-70B improves these metrics by up to 8.9% and 12.2%. This indicates that the quality of persona representations returned by GPLR is not sensitive to the LLMs.

Table 4: Case study for User 156246.
Test Set Products # LGCN LGCN3 Babyfood (Broccoli) 1 Banana Banana 2 Strawberries Strawberries Pasta Sauce Babyfood (Pumpkin) 3 Limes Limes 4 Red Onion Babyfood (Spinach)
Cheese Slices Babyfood (Mighty) 5 Yellow Onions Babyfood (Carrot)
:::::::::
Banana 6 Bunched Cilantro Spinach Organic ::::::::
Strawberries Babyfood (Spinach) 7 ::::::
Red Onion Bunched Cilantro Babyfood (Carrot) 8 Whole Wheat Bread Babyfood (Beet)
9 Cucumber Kirby Red Peppers :::::::::::
Green Onions Organic Fuji Apple Organic Cucumbers 10

Table 5: LGCN3 with different personas on Instacart.
K 2 5 10 20 50 100

NDCG@K LGCN3 LGCNL 0.1570 0.1516 0.1319 0.1282 0.1197 0.1173 0.1206 0.1191 0.1451 0.1438 0.1732 0.1707

F1-Score@K LGCN3 LGCNL 0.0411 0.0403 0.0611 0.0602 0.0722 0.0713 0.0758 0.0749 0.0681 0.0673 0.0557 0.0549

Table 6: LGCN3 with various LLMs on OnlineRetail.
NDCG@K F1-Score@K Llama-3-70B GPT-4-Turbo Llama-3-70B GPT-4-Turbo 2 0.2925 0.2933 0.1050 0.1052 5 0.2553 0.2601 0.1355 0.1332 0.2455 0.2503 0.1440 0.1431 10 20 0.2578 0.2578 0.1428 0.1406 50 0.2965 0.2940 0.1217 0.1201 100 0.3330 0.3325 0.0972 0.0967 K

Sampling budget. To evaluate LGCN3 with different LLM budgets 𝜏, we run GPLR by setting 𝜏 ∈ {5%, 10%, 20%, 100%} × |𝑈 |, As reported in Figure 1, there are no significant differences in recommendation performance across sample rates for either metric. Even with a sample rate of 5%, the results on both datasets nearly match the performance achieved when all persona representations are generated by the LLM. This indicates that our random walk-based method effectively infers personas for unlabeled users, reducing

You Are What You Bought: Generating Customer Personas for E-commerce Applications

K=2

K=5

K=10

K=20

NDCG

F1-Score

0.3

0.15

0.2

0.1

0.1

0.05

0

K=50

SIGIR ’25, July 13–18, 2025, Padua, Italy

Table 8: Robustness on OnlineRetail.

K=100

Method # of Consistent Customers RFM 4 (1.3%)
Persona 54 (18%)

Table 9: Silhouette score on OnlineRetail with varying Clusters# (Larger is better).

0 5%

10%

20%

sample rate

100%

5%

10%

20%

Clusters# 5 15 25 35 RFM 0.366 0.404 0.431 0.445 Persona 0.451 0.671 0.771 0.788

100%

sample rate

Figure 1: LGCN3 with different sample rates on OnlineRetail.
Table 7: LGCN3 with various ℓˆ on OnlineRetail.

Table 10: RevAff on Instacart Full with varying 𝜖.

NDCG@K F1-Score@K LGCN3 (ℓˆ = 1) LGCN3 (ℓˆ = 2) LGCN3 (ℓˆ = 1) LGCN3 (ℓˆ = 2)
2 0.2899 0.2938 0.1062 0.1059 5 0.2557 0.2562 0.1341 0.1333 10 0.2511 0.2444 0.1436 0.1425 20 0.2604 0.2557 0.1412 0.1416 0.2988 0.2941 0.1207 0.1213 50 0.3349 0.3327 0.0966 0.0968 100

𝜖 0.0 0.02 0.05 0.1 AAE (e-3) 0.0 0.92 1.59 1.73 Time (s) 268.8 77.7 49.79 47.86

K

LLM usage costs without sacrificing downstream performance. Consistent patterns emerge across all sample rates for both metrics on OnlineRetail. The F1-Score@K starts low, rises with 𝐾 up to around 10–20, then declines as recall improves at low 𝐾 but precision drops at higher values. In contrast, NDCG@K dips near 𝐾 = 10 before increasing, as iDCG stabilizes after initially rising to match the average test set size, allowing DCG growth to drive NDCG@K upward.
Random walk length. To evaluate the impact of ℓ,ˆ we choose LGCN3 with a 20% sampling rate and set ℓˆ to 1 and 2, respectively.
We then evaluate their performance on NDCG@K and F1-Score@K on the OnlineRetail dataset. As reported in Table 7, while both settings show advantages across different metrics, the average improvement over LGCN for ℓˆ = 1 is 8.3%, which is higher than the 7.3% for ℓˆ = 2. This might be because the attention score becomes less discernible as the random walk length increases.

7.4

Customer Segmentation (RQ3)

In this set of experiments, we evaluate the quality of personas generated by GPLR against existing explicit customer representations in customer segmentation, which is another e-commerce application introduced in Section 2.4. Specifically, the quality is measured by (i) their robustness over time and (ii) their clustering quality.
Robustness. To evaluate the robustness over time, we first randomly sample 300 customers from the OnlineRetail dataset with more than 10 transactions over a year and then divide their transaction data into the first and second six-month periods. We include the well-adopted RFM model [29] in customer segmentation, which calculates users’ Recency, Frequency, and Monetary values as their representations. For persona representation, we apply the LLM (GPT-4-Turbo) to select the three most dominant personas for each user from the previously generated 20-persona set. To ensure a fair comparison, each dimension of the RFM representation is discretized into 10 quantile-based baskets. We call a user a consistent customer if their representation does not change between the two periods. We use the number and fraction of consistent customers to

measure the representation’s robustness over time, with the results shown in Table 8. We can observe that the fraction of consistent customers for persona representation is 13.8× higher than the RFM model, demonstrating its superior robustness.
Cluster quality. To demonstrate the effectiveness of the persona in customer segmentation, we compare it with the competing RFM model. Specifically, we encode the persona representation using one-hot encoding and reduce it to three dimensions via PCA [21]
to align with the RFM, with both representations L2-normalized.
Using the user representations from the first six months of data in OnlineRetail, we perform K-means clustering with varying numbers of clusters {5, 15, 25, 35} and evaluate clustering quality using the Silhouette Score [25]. As demonstrated in Table 9, the proposed persona representation outperforms the state-of-the-art competitor in the customer segmentation performance by an average of 61.3%.

7.5

Approximate Solution Evaluation (RQ4)

For large-scale e-commerce datasets such as Instacart Full, the original exact solution based on the matrix computation in Eq. (2) is impractical. The intermediate product P̂ = P · P′ can easily exceed the RAM limits. For instance, with over 200 thousand users in the Instacart Full dataset, the constructed P̂ contains more than 40 billion floating point numbers, which need 160GB storage. To evaluate the empirical efficiency and accuracy of the proposed RevAff method, we compute the user-persona affinities with various error tolerance settings ranging from 𝜖 = 0 to 𝜖 = 0.2. We execute RevAff five times for each setting and record the average running time. The average absolute error (AAE) between each user’s approximated user-persona affinities and the exact solution is calculated to represent the empirical error, defined as: 𝐴𝐴𝐸 (𝑢𝑖 ) = 1 Í |𝑅 | 𝑟𝑚 |Ψ[𝑢𝑖 , 𝑟𝑚 ] − Ψ̂[𝑢𝑖 , 𝑟𝑚 ]|. To simplify the comparison, we remove the de-bias coefficients in Eq. (4). Table 10 reports the results, showing that empirical error is significantly smaller than the theoretical error tolerance 𝜖. For example, with 𝜖 = 0.02, the empirical AAE is only 0.00092, about 20 times smaller. Regarding the time cost, computing exact solutions with RevAff (𝜖 = 0) takes around 268 seconds. While approximating the affinities with an empirical AAE smaller than 2 × 10 −3 takes about 51 seconds, which is more than five times faster. This demonstrates the efficiency of our proposed RevAff method on large-scale datasets.

SIGIR ’25, July 13–18, 2025, Padua, Italy

8

Conclusion

In this work, we introduce the concept of customer personas and propose an explicit customer representation through these personas. We then present GPLR, an effective and efficient method for generating customer personas for e-commerce applications. By leveraging random walks to infer customer personas from a small prototype user set, our approach significantly reduces the cost of full LLM labeling. Additionally, the proposed RevAff algorithm enables fast user-persona affinity computations, easily handling large-scale datasets in seconds. Integrating persona representations with state-of-the-art recommendation models, we demonstrate the superiority of our solution, which is also shown in customer segmentation. For future work, we plan to explore the effectiveness of our method in further e-commerce applications and a wider range of user interaction scenarios.

Acknowledgments This research is supported by the Ministry of Education, Singapore, under its MOE AcRF TIER 3 Grant (MOE-MOET32022-0001).

References [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).
[2] Aslan Ahmedov. 2010. Market Basket Analysis. https://www.kaggle.com/ datasets/aslanahmedov/market-basket-analysis Accessed: 2024-08-07.
[3] Qingyao Ai, Daniel N Hill, SVN Vishwanathan, and W Bruce Croft. 2019. A zero attention model for personalized product search. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 379–388.
[4] Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, and W Bruce Croft. 2017.
Learning a hierarchical embedding model for personalized product search. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. 645–654.
[5] Joydeep Banerjee, Gurulingesh Raravi, Manoj Gupta, Sindhu Ernala, Shruti Kunde, and Koustuv Dasgupta. 2016. CAPReS: context aware persona based recommendation for shoppers. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 30.
[6] Oren Barkan, Yonatan Fuchs, Avi Caciularu, and Noam Koenigstein. 2020. Explainable recommendations via attentive multi-persona collaborative filtering. In Proceedings of the 14th ACM Conference on Recommender Systems. 468–473.
[7] Rianne van den Berg, Thomas N Kipf, and Max Welling. 2017. Graph convolutional matrix completion. arXiv preprint arXiv:1706.02263 (2017).
[8] Yizhou Chen, Guangda Huzhang, Anxiang Zeng, Qingtao Yu, Hui Sun, Heng-Yi Li, Jingyi Li, Yabo Ni, Han Yu, and Zhiming Zhou. 2023. Clustered Embedding Learning for Recommender Systems. In Proceedings of the ACM Web Conference 2023. 1074–1084.
[9] Nurendra Choudhary, Edward W Huang, Karthik Subbian, and Chandan K Reddy.
2024. An interpretable ensemble of graph and language models for improving search relevance in e-commerce. In Companion Proceedings of the ACM on Web Conference 2024. 206–215.
[10] Yang Deng, Yaliang Li, Wenxuan Zhang, Bolin Ding, and Wai Lam. 2022. Toward personalized answer generation in e-commerce via multi-perspective preference modeling. ACM Transactions on Information Systems (TOIS) 40, 4 (2022), 1–28.
[11] David Easley, Jon Kleinberg, et al. 2010. Networks, crowds, and markets. Vol. 8.
Cambridge university press Cambridge.
[12] Guy Elad, Ido Guy, Slava Novgorodov, Benny Kimelfeld, and Kira Radinsky. 2019.
Learning to generate personalized product descriptions. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management.
389–398.
[13] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.

Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang and Xiaokui Xiao

[14] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web. 173–182.
[15] Instacart. 2017. Instacart Market Basket Analysis. https://www.kaggle.com/c/ instacart-market-basket-analysis/data Accessed: 2024-08-07.
[16] Carl Gustav Jung. 2014. Two essays on analytical psychology. Routledge.
[17] Shima Khoshraftar and Aijun An. 2024. A survey on graph representation learning methods. ACM Transactions on Intelligent Systems and Technology 15, 1 (2024), 1–55.
[18] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009), 30–37.
[19] Kang Li, Vinay Deolalikar, and Neeraj Pradhan. 2015. Mining lifestyle personas at scale in e-commerce. In 2015 IEEE International Conference on Big Data (Big Data). IEEE, 1254–1261.
[20] Peter Lofgren and Ashish Goel. 2013. Personalized pagerank to a target node.
arXiv preprint arXiv:1304.4658 (2013).
[21] Andrzej Maćkiewicz and Waldemar Ratajczak. 1993. Principal components analysis (PCA). Computers & Geosciences 19, 3 (1993), 303–342.
[22] Sumit Pai, Fiona Brennan, Adrianna Janik, Teutly Correia, and Luca Costabello.
2022. Unsupervised Customer Segmentation with Knowledge Graph Embeddings.
In Companion Proceedings of the Web Conference 2022. 157–161.
[23] Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij B Gupta, Xiaojiang Chen, and Xin Wang. 2021. A survey of deep active learning. ACM computing surveys (CSUR) 54, 9 (2021), 1–40.
[24] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme.
2009. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. 452–461.
[25] Peter J Rousseeuw. 1987. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics 20 (1987), 53–65.
[26] Shamane Siriwardhana, Mark McQuade, Thomas Gauthier, Lucas Atkins, Fernando Fernandes Neto, Luke Meyers, Anneketh Vij, Tyler Odenthal, Charles Goddard, Mary MacCarthy, et al. 2024. Domain Adaptation of Llama3-70BInstruct through Continual Pre-Training and Model Merging: A Comprehensive Evaluation. arXiv preprint arXiv:2406.14971 (2024).
[27] Xiran Song, Jianxun Lian, Hong Huang, Zihan Luo, Wei Zhou, Xue Lin, Mingqi Wu, Chaozhuo Li, Xing Xie, and Hai Jin. 2023. xgcn: An extreme graph convolutional network for large-scale social link prediction. In Proceedings of the ACM Web Conference 2023. 349–359.
[28] Yushi Sun, Hao Xin, Kai Sun, Yifan Ethan Xu, Xiao Yang, Xin Luna Dong, Nan Tang, and Lei Chen. 2024. Are Large Language Models a Good Replacement of Taxonomies? Proceedings of the VLDB Endowment 17, 11 (2024), 2919–2932.
[29] Shicheng Wan, Jiahui Chen, Zhenlian Qi, Wensheng Gan, and Linlin Tang. 2022.
Fast RFM model for customer segmentation. In Companion Proceedings of the Web Conference 2022. 965–972.
[30] Tian Wang, Yuri M Brovman, and Sriganesh Madhvanath. 2021. Personalized embedding-based e-commerce recommendations at ebay. arXiv preprint arXiv:2102.06156 (2021).
[31] Wei Wu, Chao Wang, Dazhong Shen, Chuan Qin, Liyi Chen, and Hui Xiong.
2024. Afdgcf: Adaptive feature de-correlation graph collaborative filtering for recommendations. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1242–1252.
[32] Changlong Yu, Xin Liu, Jefferson Maia, Yang Li, Tianyu Cao, Yifan Gao, Yangqiu Song, Rahul Goutam, Haiyang Zhang, Bing Yin, et al. 2024. COSMO: A largescale e-commerce common sense knowledge generation and serving system at Amazon. In Companion of the 2024 International Conference on Management of Data. 148–160.
[33] Changlong Yu, Weiqi Wang, Xin Liu, Jiaxin Bai, Yangqiu Song, Zheng Li, Yifan Gao, Tianyu Cao, and Bing Yin. 2023. FolkScope: Intention Knowledge Graph Construction for E-commerce Commonsense Discovery. Findings of the Association for Computational Linguistics: ACL 2023 (2023).
[34] Wenhui Yu and Zheng Qin. 2020. Graph convolutional network for recommendation with low-pass collaborative filters. In International Conference on Machine Learning. PMLR, 10936–10945.
[35] Wenhui Yu, Zixin Zhang, and Zheng Qin. 2022. Low-pass graph convolutional network for recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 36. 8954–8961.
[36] Jiahao Zhang, Rui Xue, Wenqi Fan, Xin Xu, Qing Li, Jian Pei, and Xiaorui Liu.
2024. Linear-Time Graph Neural Networks for Scalable Recommendations. In Proceedings of the ACM on Web Conference 2024. 3533–3544.
[37] Wei Zhang, Dai Li, Chen Liang, Fang Zhou, Zhongke Zhang, Xuewei Wang, Ru Li, Yi Zhou, Yaning Huang, Dong Liang, et al. 2024. Scaling User Modeling:
Large-scale Online User Representations for Ads Personalization in Meta. In Companion Proceedings of the ACM on Web Conference 2024. 47–55.

You Are What You Bought: Generating Customer Personas for E-commerce Applications

A Appendix A Proof of Theorems A.1 Proof of Theorem 1

SIGIR ’25, July 13–18, 2025, Padua, Italy

For 𝑡 = 1, we bound 𝐸 1∗ as the following:

Theorem 1. The time complexity for computing the exact Ψ by Eq. (4) is 𝑂 (|𝐸| · |𝑈 | + ( ℓˆ − 1)|𝑈 | 3 + |𝑅| · |𝑈 | 2 ).

𝐸 1∗ = |𝑠 1 (𝑤 1∗, 𝑟𝑚 ) − Ψ̂1 [𝑤 1∗, 𝑟𝑚 ]| ∑︁ 1 |𝑠 0 (𝑤, 𝑟𝑚 ) − Ψ̂0 [𝑤, 𝑟𝑚 ] + q0 (𝑤)| = ∗ |𝑁 (𝑤 1 )| ∗ 𝑤 ∈𝑁 (𝑤1 )

Proof. First, we need to compute P · P′ in Eq. (2) where both P and P′ are sparse matrices. We have 𝑑𝑖 = 𝑛𝑛𝑧 (P[𝑢𝑖 , :]) = 𝑛𝑛𝑧 (P′ [:
Í , 𝑢𝑖 ]) and 𝑖 ∈𝑈 𝑑𝑖 = |𝐸|. We calculate the computational complexity of this step by summing up the computations needed for each of its output elements as follows:
∑︁

!
= 𝑂 |𝑈 | ·

𝑑𝑖 + |𝑈 | · |𝐸|

𝑖 ∈𝑈

= 𝑂 (|𝑈 | · |𝐸|)
We now focus on the computation of Πℓˆ, as it naturally provides the values for all Πℓ with ℓ < ℓ,ˆ as well as for Π. Denote P · P′ as P̂, the first step involves computing P̂ℓ . Since P̂ ∈ R |𝑈 | × |𝑈 | is a dense matrix, the computational complexity of this step is 𝑂 (( ℓˆ − 1)|𝑈 | 3 ).
For the computation of user-persona affinities in Eq. (4), we apply Π · L where Π is also a dense matrix. Assuming that each user’s associated persona number is proportional to |𝑅|, the L is also dense. Consequently, the computational complexity of this step is 𝑂 (|𝑅| · |𝑈 | 2 ). Summing up these complexities derives that the overall algorithm computational complexity is 𝑂 (|𝐸| · |𝑈 | + (ℓ − 1)|𝑈 | 3 + |𝑅| · |𝑈 | 2 ).
□

A.2

1 |𝑁 (𝑤 1∗ )|

|𝑠 0 (𝑤, 𝑟𝑚 ) − Ψ̂0 [𝑤, 𝑟𝑚 ]| + |q0 (𝑤)|

𝑤 ∈𝑁 (𝑤1∗ )



∑︁

𝐸 0∗ +

𝑤 ∈𝑁 (𝑤1∗ )

 𝜖 𝜖 = 2ℓˆ 2ℓˆ

𝑤 ∈𝑁 (𝑤2 )

𝑖 ∈𝑈

∑︁

≤

∑︁

Similarly, we have the following bound for 𝐸 2∗ :
 ∑︁  𝜖 1 2𝜖 ∗ 𝐸 + 𝐸 2∗ ≤ ≤ 1 |𝑁 (𝑤 2∗ )| 2ℓˆ 2ℓˆ ∗

∑︁ ª © 𝑂 (𝑑𝑖 + 𝑑 𝑗 ) = 𝑂 ­ |𝑈 | · 𝑑𝑖 + 𝑑𝑗 ® 𝑖 ∈𝑈 𝑗 ∈𝑈 𝑖 ∈𝑈 « 𝑗 ∈𝑈 ¬ ∑︁ = 𝑂 (|𝑈 | · 𝑑𝑖 + |𝐸|)
∑︁ ∑︁

1 ≤ |𝑁 (𝑤 1∗ )|

Inductively, we have the following bound for 𝐸 ∗ ˆ:
2ℓ   ∑︁ ˆ 2 ℓ𝜖 1 =𝜖 𝐸∗ ˆ ≤ 2ℓ |𝑁 (𝑤 ∗ ˆ)| 2ℓˆ ∗ 2ℓ

2ℓ

Finally, for any 𝑢𝑖 ∈ 𝑈 we have the following bound on the estimation error:
Ψ[𝑢𝑖 , 𝑟𝑚 ] −

𝐸 0∗ = |𝑠 0 (𝑤 0∗, 𝑟𝑚 ) − Ψ̂0 [𝑤 0∗, 𝑟𝑚 ]| = |𝜋 0 (𝑤 0∗, 𝑤 0∗ ) · L[𝑤 0∗, 𝑟𝑚 ] − L[𝑤 0∗, 𝑟𝑚 ]| = 0

1 ∑︁ Ψ̂2ℓ [𝑢𝑖 , 𝑟𝑚 ]
ℓˆ ˆ ℓ ≤ℓ

1 ∑︁ 1 ∑︁ 𝑠 2ℓ (𝑢𝑖 , 𝑟𝑚 ) − Ψ̂2ℓ [𝑢𝑖 , 𝑟𝑚 ]
ˆ ℓ ˆ ℓˆ ˆ ℓ ≤ℓ ℓ ≤ℓ 1 ∑︁ ∗ ≤ 𝐸 2ℓ ℓˆ =

ℓ ≤ ℓˆ

≤𝜖

Proof of Theorem 3

Theorem 3. Given a persona 𝑟𝑚 and an error 𝜖, the Ψ̂[:, 𝑟𝑚 ]
generated by RevAff is an 𝜖-approximate user-persona affinity.
Í Proof. We denote 𝑠𝑡 (𝑤, 𝑟𝑚 ) = 𝑢𝑘 ∈𝑈𝑠 𝜋𝑡 (𝑤, 𝑢𝑘 ) ·L[𝑢𝑘 , 𝑟𝑚 ] and 𝐸𝑡∗ = max𝑤 ∈𝑈 ∪𝑉 |𝑠𝑡 (𝑤, 𝑟𝑚 ) − Ψ̂𝑡 [𝑤, 𝑟𝑚 ]| with the maximizer 𝑤𝑡∗ .
For 𝑡 = 0, by definition we have

𝑤 ∈𝑁 (𝑤 ˆ )

□

A.3

Proof of Theorem 4

Theorem 4. Given a graph G, a persona set 𝑅, and an error 𝜖, the time complexity of estimating Ψ by invoking RevAff from every   1 2 ˆ ˆ ) + |𝐸| , where 𝑁 = |𝑈 | + |𝑉 |.
𝑟𝑚 ∈ 𝑅 is 𝑂 𝜖 ℓ 𝑁 log( ℓ𝑁 Proof. Denote 𝑐 ∗ = max𝑚 𝑐𝑚 . During the execution for a specific persona 𝑟𝑚 , the maximal time to pop a node 𝑤 from PQ for Í its 𝑡-th reverse updating is 𝑢𝑘 ∈𝑈𝑠 (𝜋𝑡 (𝑤, 𝑢𝑘 ) · L[𝑢𝑘 , 𝑟𝑚 ])/(𝜖/(2 ˆ ℓ)).
Because each time we process on 𝑤 will refresh its priority by at least 𝜖/(2 ˆ ℓ), and its overall gained priority will not exceed the numerator term. In each process of 𝑤, the time to update Ψ̂𝑡 [:, 𝑟𝑚 ]
is 𝑑 𝑤 steps since each of its in-neighbors receives some of its updates. The maintenance of PQ (poping and updating) each time costs 𝑂 (log(2 ˆ ℓ𝑁 )) work. Thus the running time for a single 𝑤 for updating all its hops is less than the following.
∑︁ ∑︁ 1 ˆ ) + 𝑑𝑤 )
· (𝜋𝑡 (𝑤, 𝑢𝑘 ) · L[𝑤, 𝑟𝑚 ]) · 𝑂 (log(2ℓ𝑁 ˆ 𝜖/(2ℓ) 𝑡 𝑢 ∈𝑈 𝑘

𝑠

SIGIR ’25, July 13–18, 2025, Padua, Italy

Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang and Xiaokui Xiao

Then for all personas and all nodes, the overall cost can be bounded engineering techniques. Using the OnlineRetail dataset as an examas the following:
ple, we provide the templates in Figure 2.a, 2.b and 2.c. In the first ∑︁ ∑︁ ∑︁ ∑︁ step, we extract the purchase histories from 100 randomly selected 1 (𝜋𝑡 (𝑤, 𝑢𝑘 ) · L[𝑤, 𝑟𝑚 ]) · 𝑂 (log(2 ˆ ℓ𝑁 ) + 𝑑 𝑤customers )
and instruct the LLM to generate 20 persona candidates, 𝜖/(2 ˆ ℓ) 𝑟 ∈𝑅 𝑤 ∈𝑈 ∪𝑉 𝑡 𝑢 ∈𝑈 𝑚 𝑠 𝑘 each with a description. We then repeat this process 40 times to create 40 sets of persona candidates. In the second step, we sample ∑︁ ©∑︁ ∑︁ ∑︁ 1 ª 𝜋𝑡 (𝑤, 𝑢𝑘 )
= L[𝑤, 𝑟𝑚 ] ® · 𝑂 (log(2 ˆ ℓ𝑁 ) + 𝑑 𝑤 )5 of these 40 sets and prompt the LLM to provide an improved set ˆ 𝜖/(2ℓ) 𝑤 ∈𝑈 ∪𝑉 𝑡 𝑢 ∈𝑈 𝑟𝑚 ∈𝑅 𝑠 𝑘 of 20 personas. This process is repeated 8 times, producing 160 « ¬ ∑︁ 1 polished persona candidates. Finally, we feed these candidates into ∗ ≤ 2ℓˆ · 𝑐 · 𝑂 (log(2 ˆ ℓ𝑁 ) + 𝑑 𝑤 )
ˆ the third template to obtain the final set of 20 personas.
𝜖/(2ℓ)
𝑤 ∈𝑈 ∪𝑉 !
Persona labeling. To illustrate the LLM-based persona labeling, ∑︁ ∑︁ 𝑐 ∗ 4ℓˆ2 we present the detailed prompts in Figures 3.a, 3.b, and 3.c. First, = 𝑂 (log(2 ˆ ℓ𝑁 )) + 𝑂 (𝑑 𝑤 )
𝜖 we instruct the LLM to specify the requirements and format for this 𝑤 ∈𝑈 ∪𝑉 𝑤 ∈𝑈 ∪𝑉 task.
Next, for each prototype user, we serialize their historically ∗ 2 ˆ  𝑐 4ℓ purchased items and purchase frequencies into natural language 𝑁 · 𝑂 (log(2 ˆ ℓ𝑁 )) + 𝑂 (|𝐸|)
= 𝜖 and input them into the template. Notably, this template reminds  ∗ 2   𝑐 ℓˆ the LLM to select personas only from the provided list, resulting =𝑂 𝑁 log(2 ˆ ℓ𝑁 ) + |𝐸| 𝜖 in the persona representations shown in the last subfigure. To evaluate labeling consistency, we perform three independent rounds ∗ of labeling on customers in OnlineRetail using GPT-4-Turbo with Since  𝑐 ≤ 1, we have the  eventual computational complexity the default temperature and the same persona set. On average, 33% 1 2 𝑂 𝜖 ℓˆ 𝑁 log( ˆ ℓ𝑁 ) + |𝐸| .
□ of a user’s assigned personas appear in all three runs, while 42% appear only once. This suggests that the labeling process contains noise, and more robust downstream algorithms may help better B Details of Persona Generation and Labelling leverage these persona labels.
Persona generation. To establish a comprehensive and representative set of customer personas, we leverage LLMs with prompt

Market Basket Analysis (MBA)
System Prompt: You are an assistant skilled at summarizing, capable of deducing high-level consumer keywords based on a user’s purchases.
User Prompt: Take a deep breath and work according to the instructions step by step. Now you will conduct a series of analyses on the Market Basket Analysis (MBA) dataset. This dataset contains data from a retailer, where each user’s purchasing transactions and the bought items are recorded. From this dataset, I will provide you the purchase information from about 100 (2.5 percent)
users, per-user’s purchasing data has been grouped by their ID and transferred to a natural language description for your better understanding of their purchasing behaviors.
Your task is to generate representative and accurate 20 user personas according to these users’ purchasing patterns. Please notice that we give you 2 important targets you should consider and optimize:
• High Coverage: We hope that your generated persona set can cover as many users as possible. We define the ‘coverage’ as the total number of the users which can be labeled with at least one of your generated persona set.
• High Accuracy: We hope that each of your generated persona has a precise definition. An ambiguous or sweeping persona definition should be avoided.
Repeat your task one more time, the goal is to generate a proper set of 20 representative and accurate user personas existing in the data subset and explain them quantitatively.
For each persona, you should write a corresponding definition, an output example:
1. Home Comforts Enthusiast - Buys items focused on creating a cozy and inviting home atmosphere, such as wicker hearts, chalkboards, vintage decorative pieces, and heart-shaped ornaments.
2. Craft and DIY Hobbyist - Often purchases crafting materials, DIY kits, sewing items, plush toys, and bespoke stationery sets for personal projects or to entertain children.
... (20 personas in total)
Step 1 Now considering the user purchasing data given below: ...
Figure 2.a: Case study on initial persona set generation (Take MBA as an example) - Step 1.

You Are What You Bought: Generating Customer Personas for E-commerce Applications

SIGIR ’25, July 13–18, 2025, Padua, Italy

System Prompt: You are an assistant skilled at reading, observing and summarizing, capable of finding similar or repeated descriptions of user personas, and good at finding the most representative ones.
User Prompt: Take a deep breath and work according to the instructions step by step. Now we have 40 persona_sets, each containing 20 personas, and I will randomly select five persona_sets, each containing 20 personas, for a total of 100 personas. Your task is to select the 20 most representative personas from these 100 personas and output the results. If you find that the content of a certain group is not 20 personas but less than 20, or even irrelevant information, you should ignore this group of information and only refer to personas in other groups. Note that you may find that the personas you read have some similarities or even some duplications.
You need to find these similar or duplicate personas and select the 20 most representative personas accordingly. You should not refer to any information related to the number of occurrences in these personas, as this information is very likely to be unreasonable. You should ignore the occurrence times information and select the 20 most representative personas based only on their descriptions.
Here are the five sets of results that make up the 100 personas you need to choose from: ...
Step 2 Figure 2.b: Case study on initial persona set generation (Take MBA as an example) - Step 2.

System Prompt: You are an assistant skilled at reading, observing and summarizing, capable of finding similar or repeated description of user personas, and good at finding the most representative ones.
User Prompt: Take a deep breath and work according to the instructions step by step. Now we have 8 persona sets, each containing 20 personas, for a total of 160 personas. Your task is to select the 20 most representative personas from these 160 personas and output the results. Note that you may find that the personas you read have some similarities or even some duplications. You need to find these similar or duplicate personas and select the 20 most representative personas accordingly, that is to say, these 20 personas occurs most times and can cover most of them.
Step 3 Here are the eight sets of personas that make up the 160 personas you need to choose from: ...
Figure 2.c: Case study on initial persona set generation (Take MBA as an example) - Step 3.
User 12358 in MBA System Prompt: Now you are an intelligent e-commerce domain assistant. You are skilled at summarizing, and capable of assigning high-level consumer personas based on a user’s purchase behavior.
User Prompt: Take a deep breath and work according to the instructions step by step.
Your goal is to identify users’ shopping behaviors based on products they have bought and label them with a given set of personas.
You need to select at least one persona, at most 5 personas from our given persona list. But make sure that for each assignment you should find strong evidence in their purchase transactions. Please keep the procedure as accurate as possible.
Please provide the output in json format. Prefer to return arrays instead of comma separated strings. The following is an explanation of your return format:
{"user_number": ["Persona1", "Persona2", "Persona3"]} {"user_number": ["Persona1", "Persona2", "Persona3"]} And here is a specific example:
{"12346": [ "Vegan/Vegetarian", "High-Protein Shopper", "Pet Owner"]} In the case that you feel there does not exist any suitable persona from the given list that can properly describe a user’s purchasing behavior, you can label the user as an ’unrepresentable’ user as the following example:
{"12999": ["Unrepresentable"]} Instruction

Figure 3.a: Case study on user persona generation (take user 12358 in MBA as an example) - Instruction.

SIGIR ’25, July 13–18, 2025, Padua, Italy

Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang and Xiaokui Xiao

Here is the persona list you should choose from: [PERSONA LIST]
Remember that the user number (i.e., “user_number” in the example) should be exactly from the given transaction data, do not make it wrong since it is crucial.
Here is the data of user 12358’s transaction data for you to analyze:
The user 12358 has totally purchased 13 unique products, we show each product name followed by its purchased times: he bought:
FAIRY CAKE DESIGN UMBRELLA, 4 times; CERAMIC STRAWBERRY DESIGN MUG, 24 times; CERAMIC CAKE STAND + HANGING CAKES, 2 times; CERAMIC CAKE DESIGN SPOTTED PLATE, 12 times; DOORMAT FAIRY CAKE, 2 times; EDWARDIAN PARASOL PINK, 12 times; EDWARDIAN PARASOL NATURAL, 24 times; EDWARDIAN PARASOL RED, 24 times; EDWARDIAN PARASOL BLACK, 24 times; STRAWBERRY CERAMIC TRINKET BOX, 12 times; CERAMIC BOWL WITH STRAWBERRY DESIGN, 6 times; POSTAGE, 4 times; CERAMIC STRAWBERRY CAKE MONEY BANK, 36 times. Remind one more time that you can only select from the given 20 personas’ list and only use the exactly given persona, you cannot use other words to describe. You do not Input Prompt need to explain how you get the result, so please respond no more than the required format.
Figure 3.b: Case study on user persona generation (take user 12358 in MBA as an example) - Input Prompt.

{"12358": ["Home Decor Aficionado", "Vintage and Retro Enthusiast"]}

Generated Result Figure 3.c: Case study on user persona generation (take user 12358 in MBA as an example) - Generated Result.


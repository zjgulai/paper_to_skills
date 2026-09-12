<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2402.09176
     paper_id : 2402.09176
     source   : https://arxiv.org/html/2402.09176v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Large Language Model Interaction Simulator for Cold-Start Item Recommendation

DOI: XXXXXXX.XXXXXXXConference: Make sure to enter the correct conference title from your rights confirmation emai; June 03–05, 2018; Woodstock, NYPrice: 15.00ISBN: 978-1-4503-XXXX-X/18/06CCS: Information systems Collaborative filteringCCS: Information systems Recommender systemsCCS: Information systems Retrieval models and ranking
Feiran Huang Affiliation: Jinan University, Guangzhou, China email: huangfr@jnu.edu.cn , Zhenghang Yang Note: All these student authors contributed equally to this research. Affiliation: Jinan University, Guangzhou, China email: yangzhenghang@stu2022.jnu.edu.cn , Junyi Jiang Affiliation: Jinan University, Guangzhou, China email: jjy0116@stu2022.jnu.edu.cn , Yuanchen Bei Affiliation: Zhejiang University, Hangzhou, China email: yuanchenbei@zju.edu.cn , Yijie Zhang Affiliation: Jinan University, Guangzhou, China email: wingszhangyijie@gmail.com and Hao Chen Affiliation: The Hong Kong Polytechnic University, Hong Kong, China email: sundaychenhao@gmail.com

2018

###### Abstract.

Recommending cold items is a long-standing challenge for collaborative filtering models because these cold items lack historical user interactions to model their collaborative features. The gap between the content of cold items and their behavior patterns makes it difficult to generate accurate behavioral embeddings for cold items. Existing cold-start models use mapping functions to generate fake behavioral embeddings based on the content feature of cold items. However, these generated embeddings have significant differences from the real behavioral embeddings, leading to a negative impact on cold recommendation performance. To address this challenge, we propose an LLM Interaction Simulator (LLM-InS) to model users’ behavior patterns based on the content aspect. This simulator allows recommender systems to simulate vivid interactions for each cold item and transform them from cold to warm items directly. Specifically, we outline the designing and training process of a tailored LLM-simulator that can simulate the behavioral patterns of users and items. Additionally, we introduce an efficient “filtering-and-refining” approach to take full advantage of the simulation power of the LLMs. Finally, we propose an updating method to update the embeddings of the items. we unified trains for both cold and warm items within a recommender model based on the simulated and real interactions. Extensive experiments using real behavioral embeddings demonstrate that our proposed model, LLM-InS, outperforms nine state-of-the-art cold-start methods and three LLM models in cold-start item recommendations.

###### Keywords:

recommender systems, cold-start item recommendation, collaborative filtering, large language models

## 1. Introduction

Collaborative Filtering (CF) is essential for billion-scale recommender systems to filter the most interesting items for users from billions of candidates. CF models (He et al., 2020; He et al., 2017; Wang et al., 2019; Xu et al., 2022a) can train behavioral embeddings for each user and item based on their historical interactions. The recommender system can then efficiently filter the most desired items for each user by multiplying the user’s embeddings with all item embeddings (He et al., 2020; Huang et al., 2023; Bei et al., 2023a; Chen et al., 2024). However, CF models face significant challenges due to the constant influx of new items being uploaded, including new products, videos (Liao et al., 2013; Jiang et al., 2019; Gill et al., 2007), and livestreams (Lu et al., 2018; Tang et al., 2016; Bei et al., 2023b). Unlike “warm” items with a history of user interactions, these “cold” items lack available data for training behavior embeddings. As a result, these cold items cannot be effectively recommended to users, which hampers the overall ecosystem and revenue of the recommender system. Therefore, it is necessary to encourage cold items to accumulate user interactions as quickly as possible in order to transition into warm items.

*Figure 1. (a) A brief comparison of the embedding distribution gap between generative/dropout embedding simulation models and LLM-InS on CiteULike. (b) The seesaw problem in both overall, cold, and warm recommendation performance in embedding simulation models.*

Since generating interactions for cold items is challenging, the current cold-start method primarily relies on an “embedding simulating” approach (Chen et al., 2022a; Wei et al., 2021; Bei et al., 2023c; Huang et al., 2023). This involves training mapping functions to generate embeddings for each cold item based on their content feature. DeepMusic (Van den Oord et al., 2013) achieves this by minimizing the difference between the generated embeddings and the actual behavioral embeddings. Building on this idea, GAR (Chen et al., 2022a) employs a generative-adversarial mechanism to ensure that the generated embeddings have the same distribution as the real behavioral embeddings. ALDI (Huang et al., 2023) takes a different approach by using real behavioral embeddings as teachers to distill their knowledge into the generated embeddings. Additionally, some methods utilize the dropout strategy (Srivastava et al., 2014) to enhance the robustness of recommender systems. However, they still rely on a mapping function to map the content features to the behavioral embeddings, which can also be categorized as “embedding simulating”.

Unfortunately, the solution that simulates embeddings faces a gap between the actual behavioral embeddings and the generated ones. This is demonstrated in Figure 1-(a), where the distribution of the generated embeddings differs significantly from the distribution of the real behavioral embeddings. As a result, the recommendation of cold items is negatively impacted, as shown in Figure 1-(b). This highlights the limitations of the embedding simulating models. This phenomenon is mainly caused by the following reasons:
(1) Different embedding logic: The generated embeddings are created from content features, while the behavioral embeddings are trained from user-item interactions. This fundamental difference means that the embedding simulating models can only minimize the vector differences, but they are not optimized using real users’ behavioral embeddings.
(2) Insufficient content modeling: Existing cold start models often use simple NLP approaches like word2vec (Mikolov et al., 2013) and MLPs (Zhao et al., 2022; Sedhain et al., 2015) to model the content features of cold items. However, these simple NLP approaches cannot fully utilize the content features like more advanced models such as Berts (Devlin et al., 2018; Liu et al., 2019) with larger parameters. Both of these limitations hinder existing cold-start models from effectively bridging the gap between content and behaviors.

To better address the aforementioned issues, we conceived the idea of using Large Language Models (LLMs) (Lin et al., 2023) to simulate interactions for the behavioral embeddings and to model content embeddings more effectively. LLMs have made significant advancements in understanding and generating natural languages. However, their utilization in cold-start recommendations is still undefined and not well-established. The application of LLMs in cold-start item recommendation faces several challenges:

1. Co-recommending warm and cold items: In real recommender systems, both warm and cold items need to be recommended to users simultaneously. This implies that cold items should be recommended in the same way and with the same rating distribution as warm items (Chen et al., 2022a; Huang et al., 2023).

2. Behavioral embedding generation: Embedding-based recommendation is widely used in recommendation systems. However, it is challenging to enforce LLMs to generate embeddings that are aligned with the recommender system (Bao et al., 2023).

In this paper, we propose a new approach to address the mentioned challenges. we introduce the LLM Interaction Simulator (LLM-InS) to simulate user interactions for each cold item. Figure 1. (a) shows that the cold items and warm items have similar embedding distribution, while Figure 1. (b) demonstrates that the cold item recommendation performance is significantly better than existing cold-start models like generative models and dropout models. We describe the design of the LLM-simulator, including modeling user intent towards items in terms of content aspect. Additionally, we modify the structure of the LLM to enable the simulator to evaluate user intent in a coarse and detailed manner. We also present a filtering-and-refining approach to effectively utilize the simulator for quickly filtering and refining potential user candidates. Finally, we propose an updating method to update the embeddings of items based on a combination of simulated and real interactions, ensuring a comprehensive and effective update process. The main contributions of this paper are as follows:

-

We highlight that traditional embedding simulating models often encounter a significant decline in performance when recommending cold items, due to the noticeable difference between the generated embeddings and the warm embeddings. Based on this observation, we propose a novel solution called LLM-InS for cold-start item recommendation.

-

We modify the structure and training objective of large language models to not only provide inference capabilities for user and item embeddings but also accurately simulate pairwise interactions. We introduce an efficacious “filtering-and-refining” approach to simulate interactions for cold items.

-

Extensive offline experiments on two benchmark datasets demonstrate that LLM-InS outperforms nine state-of-the-art cold-start and three LLM models in overall, cold, and warm recommendation performance.

## 2. Related Works

### 2.1. Cold-Start Item Recommendation

Cold-start item recommendations refer to recommending newly occurred items to users. It presents a long-term challenge for recommendation systems due to the lack of behavioral interactions to model these items (Huang et al., 2023; Wei et al., 2021).

Current models map the content of cold items to an embedding, which is then aligned with the behavioral embedding obtained from warm items. Based on the direction of alignment, one category of these methods is the embedding dropout model, which aims to align the behavioral embedding of warm items with the content-generated embedding of cold items through a dropout approach (Volkovs et al., 2017; Zhu et al., 2020; Du et al., 2020; Wei et al., 2021; Shi et al., 2019; Xu et al., 2022b). Another category is the embedding generative model, where the goal is to align the embedding mapped from the content of cold items towards the direction of warm behavioral embedding (Van den Oord et al., 2013; Pan et al., 2019; Chen et al., 2022a; Huang et al., 2023). This is done in an effort to closely mimic behavioral embeddings through content embeddings, which we summarize as the “embedding simulato”.

### 2.2. Recommendation with LLMs

Recently, large language models (LLMs) have excelled in understanding and generating human-like text, leveraging vast amounts of data to provide accurate, context-aware responses. Due to the abundance of natural language descriptions of content in recommender systems, many studies have begun to focus on the capabilities of large language models in the field of recommendation (Li et al., 2023; Wei et al., 2023; Sanner et al., 2023; Bao et al., 2023; Wang and Lim, 2023).

Specifically, TALLRec (Bao et al., 2023) proposes a framework designed to efficiently align LLMs with recommendation tasks, which incorporates two stages of tuning: alpaca tuning and rec-tuning. The study demonstrates that TALLRec improves recommendation capabilities and robust cross-domain recommendation tasks. LLMRec (Wei et al., 2023) enhances recommender systems by employing three large language model (LLM)-based graph augmentation strategies. These strategies focus on reinforcing user-item interaction edges, enhancing item node attributes, and conducting user node profiling from a natural language perspective.

*Figure 2. The framework of the LLM-InS. In part (a), we present the overall architecture of our LLM-Simulator. Part (b) primarily illustrates the structure of our Hierarchical Interaction Simulator, where the upper part is Embedding-based Filtering, and the remaining parts involve Prompt-based Refining. Part (c) outlines our Embedding Updating process.*

## 3. Preliminaries

### 3.1. Notations

The recommended sets of users and items are denoted as ${\mathcal{U}}$ and ${\mathcal{I}}$, respectively. The set representing interactions between users and items is represented by ${\mathcal{H}}$. Specifically, items with a historical interaction count greater than zero are referred to as warm items and denoted as ${\mathcal{I}}_{w}$. For items with zero historical interactions, they are termed cold items and represented as ${\mathcal{I}}_{c}$. For warm items with historical interactions, behavior embeddings can be learned from interactions using models like MF (Rendle et al., 2009), LightGCN (He et al., 2020), etc. The learned user embedding is denoted as ${\bm{E}}_{\mathcal{U}}$, and the embedding for warm items is denoted as ${\bm{E}}_{{\mathcal{I}}_{w}}$. Since cold items lack interaction behavior, they can only have content embeddings generated based on features such as titles, item descriptions, etc. We represent these content embeddings as ${\bm{E}}_{{\mathcal{I}}_{c}}$.

The core problem of the cold items is these items have no historical interactions. To address this, we utilize an interaction simulator ${\mathcal{S}}_{im}$ to generate simulated interactions. For each cold item $i\in{\mathcal{I}}_{c}$, we use the interaction simulator ${\mathcal{S}}_{im}$ to generate simulated interactions ${\mathcal{I}}_{n}$ for that cold item. It’s impractical to compute interactions for all users with each cold item. To quickly generate interactions, we need to precompute a user candidate set ${\mathcal{C}}_{u}$, $u\in{\mathcal{U}}$, consisting of users who are most likely to have potential interactions with the cold item. During the process of simulating interactions, we need to train the LLM. We denote the original LLM as ${\bm{L}}_{o}$ and the trained LLM as ${\bm{L}}_{t}$.

### 3.2. Problem Definition

##### Cold-start Recommendation.

The task of a recommendation system is to recommend items to users. Specifically, this involves calculating similarity scores between user and item embeddings. Then recommend the top-$k$ items most similar to the current user involved. In the context of cold-start scenarios, where items can be categorized as cold and warm. To measure the recommendation capability in this scenario, we followed ALDI (Huang et al., 2023) and defined three tasks: recommendation tasks for all items, cold items, and warm items. Overall task $R_{all}$ refers to the recommendation task where predictions are made for all items based on their prediction scores.It can be represented as:

$R_{all}={\mathcal{P}}(\hat{y}_{u,i},\forall i\in({\mathcal{I}}_{w}\cup{\mathcal{I}}_{c})).$ | (1) | | | |

Warm recommendation $R_{w}$ refers to the recommendation task where predictions are made for warm items.

$R_{w}={\mathcal{P}}(\hat{y}_{u,i},\forall i\in{\mathcal{I}}_{w}\}).$ | (2) | | | |

Cold recommendation $R_{c}$ refers to the recommendation task where predictions are made for cold items.

$R_{c}={\mathcal{P}}(\hat{y}_{u,i},\forall i\in{\mathcal{I}}_{c}\}).$ | (3) | | | |

Here, ${\mathcal{P}}$ represents the prediction scores for the users and items, then returns the indices of the top-$K$ items ranked by prediction scores from high to low.

##### Interaction Simulation.

The interaction simulator can play the part of a given user, collecting the historical interactions of the given user to simulate whether this user may interact with the given cold item. Specifically, given user $u$ with behaviors $B_{u}=\{{\bm{c}}_{u,1},\cdots,{\bm{c}}_{u,|B_{u}|}\}$, the simulator ${\mathcal{S}}_{im}$ can be present as follows,

$\hat{I}_{u,i}={\mathcal{S}}_{im}(\{{\bm{c}}_{u,1},\cdots,{\bm{c}}_{u,|B_{u}|}\},{\bm{c}}_{i}),$ | (4) | | | |

Where $\hat{I}_{u,i}$ represents the predicted interaction of whether user $u$ will interact with item $i$. For cold items without interactions, we need to simulate sufficient interactions. Therefore, for each user $u\in{\mathcal{C}}_{u}$, the interaction simulator ${\mathcal{S}}_{im}$ needs to simulate whether an interaction would occur. After simulating interactions for all users in the candidate set, the users simulated to have interactions are combined to form the interaction set ${\mathcal{I}}_{n}$.

${\mathcal{I}}_{n}=\hat{I}_{u,i}(u\in{\mathcal{C}}_{u},i\in{\mathcal{I}}_{c}).$ | (5) | | | |

## 4. Methodology

In this section, we divide our LLM-InS framework into three parts. We first outline the role of the LLM-simulator, which can providing a natural language understanding perspective to simulate interactions and generate user interaction sets for cold items. Then leveraging the trained LLM-simulator, we introduce the Hierarchical Interaction Simulator, which mimics potential user interactions for each cold item. Lastly, utilizing the simulated user interactions, cold and warm items can be trained simultaneously, allowing for sufficient training in the same collaborative space and eliminating the giant gap between cold and warm embeddings. The overall architecture of LLM-InS is illustrated in Figure 2.

### 4.1. LLM-Simulator

Previous cold start methods often used simple models like word2-vec(Mikolov et al., 2013) to model the content information of cold items. These models solely relying on the semantic space of recommendation data are insufficient. This also makes their modeling of content information not precise and effective enough. Thanks to the extensive training of large language models in natural language processing, we leverage the excellent semantic understanding of large language models to simulate interactions between items and users through prompt-based training.

Furthermore, while previous models (Liu et al., 2023) focused on constructing item interaction sets for users, we propose to construct user interaction sets for items. This simple transformation shifts the focus of the problem to the construction of interactions for cold items. Instead of recommending items to users, we concentrate on the cold items and construct interactions that cover all cold items to the fullest extent possible, thereby facilitating their transition to warm items. In training the LLM, we followed the training approach of TALLRec (Bao et al., 2023), which involves constructing three parts: instruction, input, and output. By utilizing autoregressive loss and employing Lora (Hu et al., 2021) for parameter updates, we ensured that our trained LLM ${\bm{L}}_{t}$ adheres to the predefined output format and maximizes the authenticity of simulated interactions. The optimization objectives can be represented by the following formula:

$\begin{split}min\big(-\sum\limits_{(x,y)\in\mathcal{Z}}\sum\limits_{t=1}^{\left|y\right|}&log((P_{LLM}+P_{Lora})(y_{t}|x,y_{<t}))\big),\\
&max\quad{\mathcal{R}}\big({\mathcal{S}}_{im}({\mathcal{I}}_{c},{\mathcal{C}}_{u})\big),\end{split}$ | (6) | | | |

where $x$ represents the input, $y$ represents the output, $\mathcal{Z}$ represents the training set. $y_{t}$ denotes the t-th token in the output. $y_{<t}$ represents all tokens before the t-th token. ${\mathcal{R}}$ represents the Realpair rate of the interactions generated by the simulator. More details of the training of the LLM simulator are provided in Appendix A.1.

### 4.2. Hierarchical Interaction Simulator

Unlike conventional “embedding simulation” methods, we propose an “interaction simulation” approach. Obviously, predicting interactions for all users in large datasets is impractical. Therefore, filtering and selecting a limited number of potential user candidates is a more practical approach. Thus, we introduce an Embedding-based Filtering Simulator to solve this problem. Our method considers both semantic and collaborative information at the embedding level, allowing us to simultaneously leverage semantic and collaborative spaces to select candidate users. This ensures that our candidate user selection aligns more closely with real-world scenarios. In addition, to better leverage the capabilities of LLMs, we have designed a Prompt-based Refining Simulator, refining our interaction simulation through design prompts and ask LLMs. The following will provide explanations for each of these aspects.

#### 4.2.1. Embedding-based Filtering Simulator

In order to better achieve filtering at the embedding level, we further propose Llama Subtower, which can represent a better semantic space modeling and a Collaborative Subtower that uses the collaborative space to perform embedding filtering. Finally, we merge the embeddings of both sub-towers for the overall filtering.

##### Llama Subtower

In the training of the Llama Subtower, we represent the LLM trained through the LLM-Simulator as ${\bm{L}}_{t}$, which is equipped with recommendation capabilities. Since not all datasets include user content information, we uniformly use the set of historical items interacted with by the user to represent their content information. For both cold items and user content information, after inputting them into the ${\bm{L}}_{t}$, we calculate the embedding representation by summing and averaging the embeddings of all tokens in each content information, as expressed in equation 7,

$\bm{E}_{llm}=\frac{1}{n}\sum_{1}^{n}\bm{E}_{token},$ | (7) | | | |

where $n$ represents the number of tokens. $\bm{E}_{token}$ represents the pretrained tokens by ${\bm{L}}_{t}$. Then, we design a dual-tower model structure like DSSM (Huang et al., 2013), where each tower is an MLP projector, which utilizes the token embeddings $\bm{E}_{llm}$ from original LLM embeddings for further pre-training with recommendation tasks:

${\bm{E}}_{Lu}=\text{UserTower}({\bm{E}}_{llm}),$ | (8) | | | |

${\bm{E}}_{Li}=\text{ItemTower}({\bm{E}}_{llm}),$ | (9) | | | |

where $\bm{E}_{Lu}$ and $\bm{E}_{Li}$ are the projected user/item embeddings, respectively. To train and optimize the projector parameters, we separately calculate the inner product between the user embedding and the item embedding after passing through the MLP. This is then processed through a Sigmoid function, compared with the actual click situation (label) using BCE loss for training. Formally, for each user-item pair $(u,i)$ in training set ${\mathcal{T}}{\mathcal{S}}$, the adopted objective function can be expressed as:

${\mathcal{L}}_{bce}=-\frac{1}{|{\mathcal{T}}{\mathcal{S}}|}\sum_{(u,i)\in{\mathcal{T}}{\mathcal{S}}}y_{u,i}\log(\hat{y}_{u,i})+(1-y_{u,i})\log(1-\hat{y}_{u,i}),$ | (10) | | | |

$\hat{y}_{u,i}=\sigma(\bm{E}_{li}\cdot\bm{E}_{lu}),$ | (11) | | | |

where $\hat{y}_{u,i}$ is the predicted label and $y_{u,i}$ is the ground-truth label.

##### Collaborative Subtower

The Llama subtower aims to model the recommendation task from the natural language space. Therefore, we aim to obtain more reasonable collaborative embeddings in this collaborative subtower. Specifically, To obtain embeddings for cold items with collaborative information, we leverage the collaborative information of warm items. Initially, we learn the embeddings of warm items based on their existing interaction information. Subsequently, we employ a loss function to narrow the embedding gap between the embeddings of cold items without interaction information and the well-trained embeddings of warm items with interaction information. We utilize Bayesian Personalized Ranking (BPR) loss (Rendle et al., 2009) to enhance the consistency between corresponding embeddings on the collaborative side and non-collaborative side. Therefore, the scores for corresponding positive and negative samples should ideally be as consistent as possible. We have designed a rating distance loss $\mathcal{L}_{dis}$ to achieve this

$\mathcal{L}_{dis}=\frac{1}{\left|\mathcal{B}\right|}\sum\limits_{(u,i,j)\in\mathcal{B}}\left(\left|\widehat{y}_{ui}^{(co)}-\widehat{y}_{ui}^{(-co)}\right|^{2}+\left|\widehat{y}_{uj}^{(co)}-\widehat{y}_{uj}^{(-co)}\right|^{2}\right),$ | (12) | | | |

where $\mathcal{B}$ represent a batch of data. $u,i,j$ respectively represent the user ID, positive item ID, and negative item ID. $\widehat{y}^{co}$ represents the inner product score on the collaborative side (with collaborative information),$\widehat{y}^{-co}$ and represents the inner product score on the non-collaborative side (without collaborative information).

In addition to score consistency, on the embedding distance side, we incorporate a consistency distance loss to minimize the distance between positive and negative samples on the non-collaborative side and their counterparts on the collaborative side. To optimize computational resources, we represent the negative item embeddings by their mean, and the corresponding loss can be expressed as:

$\mathcal{L}_{ide}=-\sum\limits_{i\in\mathcal{B}_{I}}{(\overline{d}_{i}^{(co)}\ln\overline{d}_{i}^{(-co)}+(1-\overline{d}_{i}^{(co)})\ln(1-\overline{d}_{i}^{(-co)}))},$ | (13) | | | |

$\overline{d}_{i}^{(co)}=\sigma\big({\bm{e}}_{i}^{\top}\cdot({\bm{e}}_{i}-\frac{1}{\left|\mathcal{B}\right|}\sum\limits_{j\in\mathcal{B}_{J}}{\bm{e}}_{j})\big),$ | (14) | | | |

where the $\mathcal{B}$ is the batch size of the negative item embeddings, ${\bm{e}}_{j}$ represent the negative item’s embedding. The $\overline{d}_{i}^{(-co)}$ is the same operation on the non-collaborative side. To ultimately enable the side without collaborative information to learn information related to the recommendation task, we use $\mathcal{L}_{BPR}$ as the recommendation loss.Therefore, our overall loss is represented as:

$\min\limits_{\theta f}\mathcal{L}_{BPR}+\alpha\mathcal{L}_{dis}+\beta\mathcal{L}_{ide},$ | (15) | | | |

where $\alpha$ is a learnable parameter representing the weight of the recommendation loss, and $\beta$ represents the weight of other losses.

##### Merge Embedding

After training, we obtain the embeddings for users and items on the Collaborative Subtower side. Then we concatenate the obtained embeddings from the Llama Subtower, denoted as $E_{Lu}$ and $E_{Li}$ , and the embeddings obtained based on the Collaborative Subtower, denoted as $E_{Cu}$ and $E_{Ci}$, as shown in the following formula:

${\bm{E}}_{LTi}={\bm{E}}_{Li}\|{\bm{E}}_{Ci},\quad{\bm{E}}_{LTu}={\bm{E}}_{Lu}\|{\bm{E}}_{Cu}.$ | (16) | | | |

To ensure that the semantic space and collaborative space embeddings do not interfere with each other, we perform inner product operations using the concatenated embeddings. Inner product operations ensure that embeddings in the same space are multiplied, while embeddings in different spaces do not affect each other. The embeddings obtained after the inner product operation are denoted as $E_{LTu}$ and $E_{LTi}$ . Subsequently, for each cold item, we calculate its top-K user candidates $C_{cold_{i}}$ by multiplying its inner product with all users, as shown in the following formula:

$I_{sim}=E_{LTi}\cdot{E_{LTu}^{T}},\quad C_{cold_{i}}=\text{Topk}(I_{sim}).$ | (17) | | | |

#### 4.2.2. Prompt-based Refining Simulator

The user set from the candidate set $C_{cold_{i}}$ is utilized as the interaction set generated for the cold item during the Prompt-based Refining Simulator. To activate LLM’s ability to answer recommendation questions through prompt-based querying, we designed the Prompt-based Refining Simulator. By constructing prompts, we design interactions between the content information of cold items and user content information, simulating interactions in a question-and-answer format. Specifically, we use the user’s historical interactions as the user’s content information, and the item’s information is represented by the item’s content. The prompt can be formulated as follows:

$prompt={\mathcal{G}}_{u,i}({\mathcal{H}}_{u},{\mathcal{I}}_{t}|{\mathcal{I}}_{t}\in{\mathcal{I}}_{c}),$ | (18) | | | |

where ${\mathcal{G}}_{u,i}$ represents the generated function of prompt, ${\mathcal{H}}_{u}$ represents the user’s historical interaction item content, ${\mathcal{I}}_{t}$ represents the target cold item. After inputting the prompt into LLM, the output obtained serves as the label for this prompt, indicating whether it is a click or not. This process leverages the knowledge learned by LLM during the fine-tuning phase to refine the interactions generated in the Embedding-based Filtering Simulator. We denote the user set obtained after the Prompt-based Refining Simulator as ${\mathcal{C}}_{f}$.This process can be expressed using the following formula:

${\mathcal{C}}_{f}={\bm{L}}_{t}(prompt|{\bm{A}}_{ns}=yes),$ | (19) | | | |

where ${\bm{A}}_{ns}$ represents the answer of the LLM.

### 4.3. Embedding Updating

After obtaining the final user set ${\mathcal{C}}_{f}$, all the cold items now can get the interactions ${\mathcal{I}}_{n_{c}}$ with user set${\mathcal{C}}_{f}$. They possess information in the LLM semantic space and the collaborative information on the Collaborative Subtower. We merge the refined interactions with the historical interaction set of warm items ${\mathcal{I}}_{n_{w}}$ to obtain the final interaction set ${\mathcal{I}}_{n_{f}}={\mathcal{I}}_{n_{c}}\cup{\mathcal{I}}_{n_{w}}$. During the Embedding Updating phase, we uniformly traind final interactions ${\mathcal{I}}_{n_{f}}$. This approach aims to bring the post-training distributions of cold and warm items closer. At the same time, based on our design, the interactions generated for cold items have introduced useful information into the LLM to some extent. This not only generates interactions for cold items that have no interaction but also refines the overall embedding distribution more accurately. We then process the post-interaction data through the MF (Rendle et al., 2009), LightGCN (He et al., 2020), and NGCF (Wang et al., 2019) models, measuring their performance on cold, warm, and all items using the Recall and NDCG metrics. The early stopping criterion is based on the optimal performance across all items. This process results in the final user and item embeddings used for recommendations.

## 5. Experiments

In this section, we conduct comprehensive experiments on benchmark cold-start recommendation datasets, aiming to answer the following research questions.

-

RQ1: Does LLM-InS outperform contemporary state-of-the-art cold-start recommendation models in overall, warm, and cold recommendations?

-

RQ2: What is the effect of different components in LLM-InS?

-

RQ3: Can LLM-InS achieve superior performance than current representative LLMs for recommendations?

We also supplement the parameter experiments and the experiment of content recommendation and adoptadion experiment in the content of the appendix.

*Table 1. Overall, cold and warm recommendation performance comparison over three backbone models (MF, NGCF, LightGCN). The best and second-best results in each column are highlighted in bold font and underlined.*

| Method | Overall Recommendation | Cold Recommendation | Warm Recommendation |

| CiteULike | MovieLens | CiteULike | MovieLens | CiteULike | MovieLens |

| Recall | NDCG | Recall | NDCG | Recall | NDCG | Recall | NDCG | Recall | NDCG | Recall | NDCG |

| MF | Backbone | 0.0776 | 0.0647 | 0.0932 | 0.1377 | 0.0056 | 0.0031 | 0.0251 | 0.0276 | 0.2838 | 0.1933 | 0.2076 | 0.1819 |

| DropoutNet | 0.0794 | 0.0670 | 0.0646 | 0.1127 | 0.2268 | 0.1356 | 0.0671 | 0.0808 | 0.1343 | 0.0792 | 0.1391 | 0.1386 |

| MTPR | 0.1060 | 0.0810 | 0.0739 | 0.1055 | 0.2496 | 0.1476 | 0.0745 | 0.0811 | 0.1728 | 0.0998 | 0.1628 | 0.1388 |

| CLCRec | 0.1269 | 0.0992 | 0.0784 | 0.1358 | 0.2295 | 0.1347 | 0.0744 | 0.0726 | 0.1898 | 0.1167 | 0.1699 | 0.1696 |

| DeepMusic | 0.0956 | 0.0789 | 0.0933 | 0.1377 | 0.2141 | 0.1262 | 0.0327 | 0.0391 | 0.2838 | 0.1933 | 0.2076 | 0.1819 |

| MetaEmb | 0.0972 | 0.0804 | 0.0933 | 0.1377 | 0.2232 | 0.1306 | 0.0432 | 0.0468 | 0.2838 | 0.1933 | 0.2076 | 0.1819 |

| GPatch | 0.1568 | 0.1305 | 0.0813 | 0.1253 | 0.2107 | 0.1193 | 0.0683 | 0.0704 | 0.2838 | 0.1993 | 0.2076 | 0.1819 |

| GAR | 0.1440 | 0.1132 | 0.0462 | 0.0812 | 0.2453 | 0.1479 | 0.0348 | 0.0510 | 0.2272 | 0.1438 | 0.1003 | 0.1003 |

| ALDI | 0.1618 | 0.1204 | 0.0914 | 0.1355 | 0.2684 | 0.1550 | 0.0431 | 0.0464 | 0.2838 | 0.1993 | 0.2076 | 0.1819 |

| LLM-InS | 0.2213 | 0.1599 | 0.0976 | 0.1473 | 0.3335 | 0.1944 | 0.1225 | 0.1182 | 0.3058 | 0.2061 | 0.2129 | 0.1924 |

| %Improv. | 36.77% | 22.52% | 4.61% | 6.98% | 24.25% | 25.41% | 64.43% | 45.75% | 7.75% | 3.41% | 2.56% | 5.78% |

| NGCF | Backbone | 0.1105 | 0.0951 | 0.1393 | 0.2265 | 0.0064 | 0.0029 | 0.0253 | 0.0275 | 0.2347 | 0.1485 | 0.3076 | 0.2922 |

| DropoutNet | 0.0813 | 0.0656 | 0.1144 | 0.1935 | 0.2211 | 0.1278 | 0.0214 | 0.0223 | 0.1416 | 0.0842 | 0.2517 | 0.2457 |

| MTPR | 0.1006 | 0.0769 | 0.1132 | 0.1818 | 0.2479 | 0.1391 | 0.0749 | 0.0894 | 0.1753 | 0.0977 | 0.2509 | 0.2329 |

| CLCRec | 0.1201 | 0.0920 | 0.1270 | 0.2042 | 0.2093 | 0.1188 | 0.0694 | 0.0777 | 0.1886 | 0.1136 | 0.2807 | 0.2627 |

| DeepMusic | 0.1269 | 0.1043 | 0.1393 | 0.2265 | 0.1980 | 0.1152 | 0.0409 | 0.0493 | 0.2347 | 0.1485 | 0.3076 | 0.2922 |

| MetaEmb | 0.1119 | 0.0957 | 0.1393 | 0.2265 | 0.2830 | 0.1664 | 0.0247 | 0.0245 | 0.2347 | 0.1485 | 0.3076 | 0.2922 |

| GPatch | 0.1449 | 0.1099 | 0.1381 | 0.2234 | 0.2318 | 0.1371 | 0.0589 | 0.0646 | 0.2347 | 0.1485 | 0.3076 | 0.2922 |

| GAR | 0.1144 | 0.0909 | 0.0098 | 0.0174 | 0.2099 | 0.1251 | 0.0166 | 0.0178 | 0.1850 | 0.1143 | 0.2788 | 0.2721 |

| UCC | 0.1094 | 0.0990 | 0.0981 | 0.1523 | 0.0019 | 0.0008 | 0.0063 | 0.0066 | 0.2402 | 0.1582 | 0.2191 | 0.1999 |

| ALDI | 0.1541 | 0.1141 | 0.1393 | 0.2206 | 0.2466 | 0.1399 | 0.1022 | 0.1113 | 0.2347 | 0.1485 | 0.3076 | 0.2922 |

| LLM-InS | 0.2211 | 0.1653 | 0.1447 | 0.2372 | 0.3454 | 0.2025 | 0.1670 | 0.1662 | 0.2992 | 0.1958 | 0.3186 | 0.3052 |

| %Improv. | 43.48% | 44.87% | 3.88% | 4.72% | 22.05% | 21.69% | 63.41% | 49.33% | 24.56% | 23.76% | 3.58% | 4.45% |

| LightGCN | Backbone | 0.0812 | 0.0622 | 0.1418 | 0.2330 | 0.0041 | 0.0019 | 0.0177 | 0.0190 | 0.2528 | 0.1541 | 0.3130 | 0.3008 |

| DropoutNet | 0.0883 | 0.0639 | 0.1165 | 0.1978 | 0.2309 | 0.1312 | 0.0340 | 0.0373 | 0.1175 | 0.0692 | 0.2560 | 0.2518 |

| MTPR | 0.1001 | 0.0753 | 0.1011 | .0.1551 | 0.2585 | 0.1454 | 0.0779 | 0.0802 | 0.1753 | 0.0697 | 0.2247 | 0.2009 |

| CLCRec | 0.1293 | 0.0965 | 0.1253 | 0.2037 | 0.2435 | 0.1425 | 0.0677 | 0.0816 | 0.2149 | 0.1302 | 0.2764 | 0.2612 |

| DeepMusic | 0.0985 | 0.0745 | 0.1418 | 0.2330 | 0.2239 | 0.1259 | 0.0635 | 0.0719 | 0.2528 | 0.1541 | 0.3130 | 0.3008 |

| MetaEmb | 0.0924 | 0.0714 | 0.1418 | 0.2330 | 0.2252 | 0.1295 | 0.0248 | 0.0244 | 0.2528 | 0.1541 | 0.3130 | 0.3008 |

| GPatch | 0.1609 | 0.1197 | 0.1293 | 0.2106 | 0.2606 | 0.1532 | 0.0771 | 0.0760 | 0.2528 | 0.1541 | 0.3130 | 0.3008 |

| GAR | 0.1357 | 0.1062 | 0.0106 | 0.0195 | 0.2539 | 0.1489 | 0.0110 | 0.0130 | 0.2339 | 0.1455 | 0.2873 | 0.2794 |

| UCC | 0.1374 | 0.1260 | 0.1277 | 0.2020 | 0.0020 | 0.0011 | 0.0063 | 0.0073 | 0.3002 | 0.2010 | 0.2830 | 0.2641 |

| ALDI | 0.1626 | 0.1201 | 0.1428 | 0.2316 | 0.2692 | 0.1539 | 0.1229 | 0.1295 | 0.2528 | 0.1541 | 0.3130 | 0.3008 |

| LLM-InS | 0.2285 | 0.1747 | 0.1506 | 0.2468 | 0.3601 | 0.2126 | 0.1759 | 0.1762 | 0.3252 | 0.2156 | 0.3314 | 0.3186 |

| %Improv. | 40.52% | 38.65% | 5.46% | 5.92% | 33.76% | 38.14% | 43.12% | 36.06% | 8.33% | 7.26% | 5.97% | 5.92% |

### 5.1. Experimental Setup

#### 5.1.1. Datasets

We conduct comprehensive experiments on two widely used benchmark datasets: CiteULike (Wang et al., 2013) and MovieLens (Harper and Konstan, 2015) to verify the effectiveness of LLM-InS. CiteULike contains 5,551 users, 16,980 articles, and 204,986 interactions. The articles are represented by 300-dimensional vectors as item content features. MovieLens comprises 6,040 users, 3,883 items, and 1,000,210 interactions. In this study, the content features of items are also represented using 300-dimensional vectors. For each dataset, 20% of items are designated as cold-start items, with interactions split into a cold validation set and testing set (1:1 ratio). Records of the remaining 80% of items are divided into training, validation, and testing sets, using an 8:1:1 ratio.

#### 5.1.2. Compared Baselines

To assess the effectiveness and universality of LLM-InS, we conducted a comparative analysis with nine leading-edge models in the domain of cold-start recommendations. This comparison was carried out across two distinct datasets. The models we benchmarked against include two main groups: (i) Dropout-based embedding simulators: DropoutNet (Volkovs et al., 2017), MTPR (Du et al., 2020), and CLCRec (Wei et al., 2021). (ii) Generative embedding simulators: DeepMusic (Van den Oord et al., 2013), MetaEmb (Pan et al., 2019), GPatch (Chen et al., 2022b), GAR (Chen et al., 2022a), UCC (Liu et al., 2023), and ALDI (Huang et al., 2023). Furthermore, to verify the capability of our fine-tuned LLMs, we also conduct experiments with LLMs specific for recommendations: ChatGPT, TALLRec (Bao et al., 2023), and LLMRec (Wei et al., 2023).

#### 5.1.3. Hyperparameter Setting

In the Embedding-based filtering phase, we utilized AdamW as the optimizer with a chosen learning rate of 1e-5, and set the batch size for each training batch to 128. We opted for a top-k value of 20. In the Propmt-based refining phase, the learning rate was adjusted to 5e-5. In the Embedding updating phase, we implemented the baselines using their officially provided implementations. The dimension of the embeddings was standardized to 200 for all models. We employed the Adam optimizer with a learning rate of 0.001 and applied early stopping by monitoring NDCG on the validation set. To ensure fair comparisons, we adopted the same options and adhered to the designs outlined in their respective papers for all baselines.

#### 5.1.4. Evaluation Metrics

Our evaluation encompasses the overall, warm, and cold recommendation performance, adopting a full-ranking evaluation approach (Wang et al., 2019; He et al., 2020). Consistent with the definitions provided in Section 3, we assess these three facets of recommendation quality. To evaluate the effectiveness of top-ranked articles, we employ Recall@K and NDCG@K as our primary metrics. By convention, we set K to 20 and report the average values obtained across all users in the test set. Note that we run all the experiments five times with different random seeds and report the average results to prevent extreme cases.

*Table 2. Ablation study results with LLM-InS variants.*

| Variant | Overall | Cold | Warm |

| Recall | NDCG | Recall | NDCG | RealPair | Recall | NDCG |

| Filtering | random | 0.1816 | 0.1449 | 0.2556 | 0.1628 | 0.0009 | 0.2889 | 0.1923 |

| w/o CS | 0.2012 | 0.1543 | 0.3105 | 0.1825 | 0.0436 | 0.3067 | 0.1986 |

| w/o LS | 0.2070 | 0.1633 | 0.3175 | 0.1906 | 0.0401 | 0.3019 | 0.1970 |

| Refining | w/o refining | 0.2186 | 0.1668 | 0.3429 | 0.2020 | 0.0544 | 0.3124 | 0.2044 |

| random + refining | 0.1866 | 0.1507 | 0.2759 | 0.1574 | 0.0082 | 0.3189 | 0.2145 |

| w/o CS + refining | 0.2170 | 0.1696 | 0.3313 | 0.1967 | 0.0523 | 0.3102 | 0.2007 |

| w/o LS + refining | 0.2147 | 0.1661 | 0.3271 | 0.1976 | 0.0515 | 0.316 | 0.2073 |

| LLM-InS (Ours) | 0.2285 | 0.1747 | 0.3601 | 0.2126 | 0.0615 | 0.3252 | 0.2156 |

### 5.2. Main Results (RQ1)

The performance comparison of overall, warm, and cold recommendations between LLM-InS and other baselines on benchmark datasets is presented in Table 1. To evaluate the generalizability of LLM-InS, we conduct cold-start experiments with representative recommender backbones: MF (Rendle et al., 2009), NGCF (Wang et al., 2019), and LightGCN (He et al., 2020) models as representative recommendation models. The improvements are calculated by comparing LLM-InS with the best-performed baseline for each backbone. From the results, we can have the following observations:

-

LLM-InS excels beyond all embedding simulation baseline models in both overall, cold, and warm recommendation performance, consistently demonstrating superiority across various datasets and backbones. This remarkable achievement can be ascribed to the effective interaction simulation by LLM-InS, a key factor that significantly enhances its recommendation capabilities for both cold and warm items simultaneously.

-

The generative embedding simulation models generally perform better in warm and overall recommendation than dropout embedding simulation models. This indicates that forcing the warm embedding and cold embedding layers to approach each other through the embedding layer will lead to poor performance of the warm embedding recommendation. The interaction simulation allows both cold and warm products to be adequately trained within a unified recommender.

*Table 3. Comparison results of LLM-InS with recommendation LLM models.*

| Task | LLM | CiteULike | MovieLens |

| Recall | NDCG | Recall | NDCG |

| Overall | ChatGPT | 0.2054 | 0.1641 | 0.1396 | 0.2267 |

| TALLRec | 0.2141 | 0.1661 | 0.1428 | 0.2324 |

| LLMRec | 0.1983 | 0.1535 | 0.1372 | 0.2257 |

| LLM-InS | 0.2285 | 0.1747 | 0.1461 | 0.2368 |

| %Improv. | 6.73% | 5.18% | 2.31% | 1.90% |

| Cold | ChatGPT | 0.3477 | 0.2079 | 0.1480 | 0.1556 |

| TALLRec | 0.3352 | 0.1990 | 0.1374 | 0.1379 |

| LLMRec | 0.3453 | 0.2076 | 0.1425 | 0.1508 |

| LLM-InS | 0.3601 | 0.2126 | 0.1563 | 0.1566 |

| %Improv. | 3.57% | 2.27% | 5.61% | 0.64% |

| Warm | ChatGPT | 0.3001 | 0.1933 | 0.3074 | 0.2921 |

| TALLRec | 0.3102 | 0.2037 | 0.3172 | 0.3016 |

| LLMRec | 0.2613 | 0.1645 | 0.3022 | 0.2902 |

| LLM-InS | 0.3252 | 0.2156 | 0.3217 | 0.3060 |

| %Improv. | 4.84% | 5.84% | 1.42% | 1.46% |

### 5.3. Ablation Study (RQ2)

We conduct an ablation study of our proposed LLM-Ins approach to validate its key components and present the results in Table 2.

#### 5.3.1. Effectiveness of filtering phase

-

From the data in Table 2, it can be seen that in the case of random selection, this means that we did not perform any filtering and refining. We just simply randomly selected a certain number of users to construct interactions with cold items. In this case, both Recall and NDCG metrics experience a significant decrease. This suggests that without the filtering and refining phases, LLM-InS fails to obtain relatively authentic and reliable user-item pair interactions. It indicates that LLM-InS can acquire potential knowledge during the filtering and refining phases to construct more reasonable and reliable interactions.

-

In cases like w/o CS and w/o LS where there is only one subtower in the filtering phase and no refining is performed, metrics in all scenarios show a varying degree of decrease. This indicates that having only one subtower can capture partial information from either the semantic or collaborative space, and the embedding vectors are not comprehensive. This precisely underscores that LLM-InS, during the filtering phase, can learn different embedding representations from various sub towers. Ultimately, these representations are concatenated to obtain a more comprehensive set of potential information from different spaces. The RealPair ratio on the cold side indicates the proportion of generated interaction pairs to real interaction pairs, reflecting the effectiveness of different sub-tasks filtering.

#### 5.3.2. Effectiveness of refining phase

-

When there is only the filtering phase without the refining phase like w/o refining, there is an improvement in metrics compared to having no filtering phase. However, relative to LLM-InS with refining, the metrics decrease. This suggests that the refining phase effectively refines the user candidate set through the LLM-Simulator, improving the ratio of RealPair.

-

In comparison to the three ablation experiments in the filtering phase without refining like random + refining, w/o CS + refining, and w/o LS + refining, the metrics of the ablation from table 2 show improvement after refining. This indicates that the refining phase, incorporating semantic space refinement through prompts, is effective in enhancing the overall performance. Simultaneously, the increase in the RealPair ratio on the cold side also indicates that refining can enhance the authenticity of generated simulated interactions.

### 5.4. Large Language Model Comparison (RQ3)

Due to the LLM being included in LLM-InS for cold-start recommendation specific design, we also compare LLM-InS with the current start-of-the-art LLM recommendation models: ChatGPT, TALLRec (Bao et al., 2023), and LLMRec (Wei et al., 2023). The comparison results are illustrated in Table 3. From these results, we can find that LLM-InS generally outperforms all current LLM models in overall, cold, and warm recommendations. This demonstrates that through our specialized LLM-Simulator training and the filtering and refining stages, we can effectively integrate the recommendation task with LLM. It also emphasizes the effectiveness of our proposed LLM-InS in cold-start item recommendation.

## 6. Conclusion

In this paper, aiming at the giant embedding gap between cold items and warm items in cold-start recommendation scenarios, we propose a novel LLM-InS that overcomes the limitations of conventional cold-start models - "embedding simulators," by leveraging patterns from warm users and items to simulate realistic interactions for cold items. The LLM-InS’s innovative workflow comprises a large language model-powered sequence of filtering, refining, and updating stages. It begins by co-modeling the user-item relationship in both content and collaborative spaces to identify potential user candidates for cold items. This is followed by a refining process using a fine-tuned LLM to finalize virtual interactions. Finally, in the updating stage, LLM-InS trains both cold and warm items within a unified recommender model based on these simulated and real interactions. Extensive experiments demonstrate LLM-InS’s superior performance over existing methods.

## References

- Bao et al. (2023) Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. Tallrec: An effective and efficient tuning framework to align large language model with recommendation. arXiv preprint arXiv:2305.00447 (2023).

- Bei et al. (2023a) Yuanchen Bei, Hao Chen, Shengyuan Chen, Xiao Huang, Sheng Zhou, and Feiran Huang. 2023a. Non-Recursive Cluster-Scale Graph Interacted Model for Click-Through Rate Prediction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management. 3748–3752.

- Bei et al. (2023b) Yuanchen Bei, Hao Xu, Sheng Zhou, Huixuan Chi, Mengdi Zhang, Zhao Li, and Jiajun Bu. 2023b. CPDG: A Contrastive Pre-Training Method for Dynamic Graph Neural Networks. arXiv preprint arXiv:2307.02813 (2023).

- Bei et al. (2023c) Yuanchen Bei, Sheng Zhou, Qiaoyu Tan, Hao Xu, Hao Chen, Zhao Li, and Jiajun Bu. 2023c. Reinforcement Neighborhood Selection for Unsupervised Graph Anomaly Detection. In 2023 IEEE International Conference on Data Mining (ICDM). IEEE, 11–20.

- Chen et al. (2024) Hao Chen, Yuanchen Bei, Qijie Shen, Yue Xu, Sheng Zhou, Wenbing Huang, Feiran Huang, Senzhang Wang, and Xiao Huang. 2024. Macro Graph Neural Networks for Online Billion-Scale Recommender Systems. arXiv preprint arXiv:2401.14939 (2024).

- Chen et al. (2022a) Hao Chen, Zefan Wang, Feiran Huang, Xiao Huang, Yue Xu, Yishi Lin, Peng He, and Zhoujun Li. 2022a. Generative adversarial framework for cold-start item recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2565–2571.

- Chen et al. (2022b) Hao Chen, Zefan Wang, Yue Xu, Xiao Huang, and Feiran Huang. 2022b. GPatch: Patching Graph Neural Networks for Cold-Start Recommendations. In 4th Workshop on Deep Learning Practice and Theory for High-Dimensional Sparse and Imbalanced Data with KDD.

- Devlin et al. (2018) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

- Du et al. (2020) Xiaoyu Du, Xiang Wang, Xiangnan He, Zechao Li, Jinhui Tang, and Tat-Seng Chua. 2020. How to learn item representation for cold-start multimedia recommendation?. In Proceedings of the 28th ACM International Conference on Multimedia. 3469–3477.

- Gill et al. (2007) Phillipa Gill, Martin Arlitt, Zongpeng Li, and Anirban Mahanti. 2007. Youtube traffic characterization: a view from the edge. In Proceedings of the 7th ACM SIGCOMM conference on Internet measurement. 15–28.

- Harper and Konstan (2015) F Maxwell Harper and Joseph A Konstan. 2015. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis) 5, 4 (2015), 1–19.

- He et al. (2020) Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.

- He et al. (2017) Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web. 173–182.

- Hu et al. (2021) Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).

- Huang et al. (2023) Feiran Huang, Zefan Wang, Xiao Huang, Yufeng Qian, Zhetao Li, and Hao Chen. 2023. Aligning Distillation For Cold-Start Item Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1147–1157.

- Huang et al. (2013) Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2333–2338.

- Jiang et al. (2019) Qing-Yuan Jiang, Yi He, Gen Li, Jian Lin, Lei Li, and Wu-Jun Li. 2019. SVD: A large-scale short video dataset for near-duplicate video retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 5281–5289.

- Li et al. (2023) Lei Li, Yongfeng Zhang, Dugang Liu, and Li Chen. 2023. Large language models for generative recommendation: A survey and visionary discussions. arXiv preprint arXiv:2309.01157 (2023).

- Liao et al. (2013) Hank Liao, Erik McDermott, and Andrew Senior. 2013. Large scale deep neural network acoustic modeling with semi-supervised training data for YouTube video transcription. In 2013 IEEE Workshop on Automatic Speech Recognition and Understanding. IEEE, 368–373.

- Lin et al. (2023) Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, et al. 2023. How Can Recommender Systems Benefit from Large Language Models: A Survey. arXiv preprint arXiv:2306.05817 (2023).

- Liu et al. (2023) Taichi Liu, Chen Gao, Zhenyu Wang, Dong Li, Jianye Hao, Depeng Jin, and Yong Li. 2023. Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2466–2470.

- Liu et al. (2019) Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 (2019).

- Lu et al. (2018) Zhicong Lu, Haijun Xia, Seongkook Heo, and Daniel Wigdor. 2018. You watch, you give, and you engage: a study of live streaming practices in China. In Proceedings of the 2018 CHI conference on human factors in computing systems. 1–13.

- Mikolov et al. (2013) Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781 (2013).

- Pan et al. (2019) Feiyang Pan, Shuokai Li, Xiang Ao, Pingzhong Tang, and Qing He. 2019. Warm up cold-start advertisements: Improving ctr predictions via learning to learn id embeddings. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 695–704.

- Rendle et al. (2009) Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. 452–461.

- Sanner et al. (2023) Scott Sanner, Krisztian Balog, Filip Radlinski, Ben Wedin, and Lucas Dixon. 2023. Large language models are competitive near cold-start recommenders for language-and item-based preferences. In Proceedings of the 17th ACM conference on recommender systems. 890–896.

- Sedhain et al. (2015) Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th international conference on World Wide Web. 111–112.

- Shi et al. (2019) Shaoyun Shi, Min Zhang, Xinxing Yu, Yongfeng Zhang, Bin Hao, Yiqun Liu, and Shaoping Ma. 2019. Adaptive feature sampling for recommendation with missing content feature values. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1451–1460.

- Srivastava et al. (2014) Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research 15, 1 (2014), 1929–1958.

- Tang et al. (2016) John C Tang, Gina Venolia, and Kori M Inkpen. 2016. Meerkat and periscope: I stream, you stream, apps stream for live streams. In Proceedings of the 2016 CHI conference on human factors in computing systems. 4770–4780.

- Van den Oord et al. (2013) Aaron Van den Oord, Sander Dieleman, and Benjamin Schrauwen. 2013. Deep content-based music recommendation. In Advances in neural information processing systems, Vol. 26.

- Volkovs et al. (2017) Maksims Volkovs, Guangwei Yu, and Tomi Poutanen. 2017. Dropoutnet: Addressing cold start in recommender systems. In Advances in neural information processing systems, Vol. 30.

- Wang et al. (2013) Hao Wang, Binyi Chen, and Wu-Jun Li. 2013. Collaborative topic regression with social regularization for tag recommendation. In Twenty-Third International Joint Conference on Artificial Intelligence.

- Wang and Lim (2023) Lei Wang and Ee-Peng Lim. 2023. Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. arXiv preprint arXiv:2304.03153 (2023).

- Wang et al. (2019) Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval. 165–174.

- Wei et al. (2023) Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2023. Llmrec: Large language models with graph augmentation for recommendation. arXiv preprint arXiv:2311.00423 (2023).

- Wei et al. (2021) Yinwei Wei, Xiang Wang, Qi Li, Liqiang Nie, Yan Li, Xuanping Li, and Tat-Seng Chua. 2021. Contrastive learning for cold-start recommendation. In Proceedings of the 29th ACM International Conference on Multimedia. 5382–5390.

- Xu et al. (2022b) Xiaoxiao Xu, Chen Yang, Qian Yu, Zhiwei Fang, Jiaxing Wang, Chaosheng Fan, Yang He, Changping Peng, Zhangang Lin, and Jingping Shao. 2022b. Alleviating Cold-start Problem in CTR Prediction with A Variational Embedding Learning Framework. In Proceedings of the ACM Web Conference 2022. 27–35.

- Xu et al. (2022a) Yue Xu, Hao Chen, Zengde Deng, Yuanchen Bei, and Feiran Huang. 2022a. Flattened Graph Convolutional Networks For Recommendation. In 4th Workshop on Deep Learning Practice and Theory for High-Dimensional Sparse and Imbalanced Data with KDD.

- Zhao et al. (2022) Xu Zhao, Yi Ren, Ying Du, Shenzheng Zhang, and Nian Wang. 2022. Improving item cold-start recommendation via model-agnostic conditional variational autoencoder. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2595–2600.

- Zhu et al. (2020) Ziwei Zhu, Shahin Sefati, Parsa Saadatpanah, and James Caverlee. 2020. Recommendation for new users and new items via randomized training and mixture-of-experts transformation. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1121–1130.

## Appendix A Experimental details

### A.1. Training Set of LLM-Simulator

This research aims to enable LLM to learn the behavior of predicting whether a user will click on a target item based on items they have previously interacted with. In constructing the dataset, we align the instruction, input, and output as follows, corresponding to the task definition, user information, target item content information, and the expected output.

For the instruction content, we define it as follows: “Given the user’s interaction paper set, determine whether the user will like the target paper by answering Yes or No.”

In the input content, we use natural language to describe the content on which the recommendation task depends, incorporating user content information and the content information of the target item. Since not all datasets include user content information, for broader applicability, we represent user content information by constructing a set of user interactions. We randomly sample 20 interactions (or fewer, based on interaction length - 1 if needed) as the representation of user content information. Additionally, we randomly select an item that the user has interacted with but is not included in the user content information as the positive target item. For positive target items, the output is set to Yes. To prevent LLM from simply answering Yes, we sample and construct negative target items, where the user has not interacted with these items, and set the output to No. The ratio of negative to positive target items is 1:1, and after shuffling, they are structured into the training set. Therefore, the training set can be represented as follows:

instruction: “Given the user’s interaction paper set, determine whether the user will like the target paper by answering Yes or No.”

input: “User preference: [The collection of papers/movies that users have interacted with], Whether the user will like the target paper/movie [target paper/movie content info]?”

output:“Yes”/“No”

Upon completing training and during inference, the instruction and input are merged into a prompt, and the output generated by LLM is taken as the label for the user-item pair.

### A.2. Details of baseline models

Baselines. To evaluate LLM-InS’s effectiveness and universality, we compare it to eight state-of-the-art cold-start recommendation models across two datasets:

-

DeepMusic (Van den Oord et al., 2013) utilizes deep neural networks to minimize the Mean Squared Error (MSE) between generated embeddings and warm embeddings.

-

MetaEmb (Pan et al., 2019) trains a generator based on meta-learning principles to achieve rapid convergence.

-

GAR (Chen et al., 2022a) generates embeddings by engaging in a generative adversarial process with the warm recommendation model.

-

DropoutNet (Volkovs et al., 2017) enhances cold-start resilience by randomly discarding embeddings.

-

MTPR (Du et al., 2020) produces counterfactual cold embeddings by incorporating dropout and BPR ranking.

-

CLCRec (Wei et al., 2021) approaches cold-start recommendation by leveraging contrastive learning from an information-theoretic perspective.

-

GPatch (Chen et al., 2022b) introduces a universal cold-start framework, solving GCN models’ cold-start issue, facilitating hybrid and warm recommendations.

-

ALDI (Huang et al., 2023) uses distillation learning to narrow the distribution gap between cold and hot items to generate embeddings for cold items.

-

UCC (Liu et al., 2023) addresses the cold-start problem by generating interactions for cold items through computing correlations on a graph, thus heating up the cold items and training the model

## Appendix B Additional Experiments

### B.1. Parameter Study

The figures in Figure 3 show the effects of two hyperparameters, the learning rate and the embedding dimension, on the recommendation performance for the CiteULike and MovieLens datasets with the LightGCN model. For the CiteULike dataset, the best results were achieved in all three test scenarios when the learning rate was set to 1e-3 and the embedding dimension to 200. For the MovieLens dataset, on the other hand, the best results were obtained in all three test scenarios when the learning rate was set to 1e-4 and the embedding dimension to 300. The results indicate that the moderate intensity of the learning rate is crucial and plays a decisive role for optimal recommendation performance.

*Figure 3. Parameter study on the updating learning rate and embedding dimension.*

### B.2. Content Recommendation

In this section, we replaced our Llama Subtower representing content with content recommendation models, DeepMusic and ID.vs Mo Rec, and compared the results. The experimental results are presented in Table 4, the experimental outcomes indicate that our approach outperforms content recommendation in cold, warm, and overall scenarios.

*Table 4. Content recommendation models v.s. LLM-InS.*

| model | Overall | Cold | Warm |

| Recall | NDCG | Recall | NDCG | Recall | NDCG |

| DeepMusic | 0.2094 | 0.1672 | 0.3383 | 0.2073 | 0.2978 | 0.1925 |

| IDvsMO.rec | 0.1948 | 0.1577 | 0.266 | 0.1634 | 0.3207 | 0.2088 |

| LLM-InS | 0.2285 | 0.1747 | 0.3601 | 0.2126 | 0.3252 | 0.2156 |

### B.3. Additional Experiments

The adoption rate represents the number of samples adopted in the refining stage out of the candidate set after the filtering stage. The experimental results are presented in Table 5. Our experiments indicate that the adoption rate is highest under the LLM-InS framework, demonstrating that LLM-InS can accurately simulate interactions.

*Table 5. Adoption rate comparison results.*

| model | adoption rate |

| random | 0.0818 |

| ALDI | 0.7507 |

| w/o CS | 0.8133 |

| w/o LS | 0.8045 |

| LLM-InS | 0.8596 |

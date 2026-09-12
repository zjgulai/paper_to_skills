<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 1811.00855
     paper_id : 1811.00855
     source   : https://arxiv.org/html/1811.00855v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Session-based Recommendation with Graph Neural Networks

Shu Wu Affiliation: Center for Research on Intelligent Perception and ComputingNational Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences Email: shu.wu@nlpr.ia.ac.cn    Yuyuan Tang Affiliation: School of Computer and Communication Engineering, University of Science and Technology Beijing Email: tangyyuanr@gmail.com    Yanqiao Zhu Affiliation: School of Software Engineering, Tongji University Email: sxkdz@tongji.edu.cn    Liang Wang Affiliation: Center for Research on Intelligent Perception and ComputingNational Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences Email: wangliang@nlpr.ia.ac.cn    Xing Xie Affiliation: Microsoft Research Asia Email: xing.xie@microsoft.com    Tieniu Tan Affiliation: Center for Research on Intelligent Perception and ComputingNational Laboratory of Pattern Recognition, Institute of Automation, Chinese Academy of Sciences Email: tnt@nlpr.ia.ac.cn

###### Abstract

The problem of session-based recommendation aims to predict users’ actions based on anonymous sessions. Previous methods on the session-based recommendation most model a session as a sequence and capture users’ preference to make recommendations. Though achieved promising results, they fail to consider the complex items transitions among all session sequences, and are insufficient to obtain accurate users’ preference in the session. To better capture the structure of the user-click sessions and take complex transitions of items into account, we propose a novel method, i.e. Session-based Recommendation with Graph Neural Networks, SR-GNN for brevity. In the proposed method, session sequences are aggregated together and modeled as graph-structure data. Based on this graph, GNN can capture complex transitions of items, which are difficult to be revealed by the conventional sequential methods. Each session is then represented as the composition of the global preference and current interests of the session using an attention network. Extensive experiments conducted on two real datasets show that SR-GNN evidently outperforms the state-of-the-art session-based recommendation methods and always obtain stable performance with different connection schemes, session representations, and session lengths.

## Introduction

With the rapid growth of information on the internet, recommendation systems become fundamental for helping users alleviate the problem of information overload and select interesting information in many Web applications, e.g., search, e-commerce, media streaming sites. Most of the existing recommendation systems assume that user’s profiles and past activities are constantly recorded. However, in many recent services, user identification may be unknown and only the user behavior history during an ongoing session is available. It is of great importance to model limited behaviors in one session and generate the recommendation. Conventional recommendation methods dealing with adequate user-item interactions have problems in yielding accurate results under this situation.

Due to the highly practical value, an increasing interest in this problem can be observed, and many kinds of proposals for session-based recommendation have been developed. Based on Markov chain, some works [2002, 2010] predict the user’s next behavior based on the previous one. With a strong independence assumption, independent combinations of the past components confine the prediction accuracy.

In recent years, the majority of research [2016a, 2016, 2017, 2017a] applies Recurrent Neural Networks (RNNs) for session-based recommendation systems and obtains promising results. The work [2016a] proposes a recurrent neural network approach first, then the model is enhanced by considering data augmentation and temporal shift of user behavior [2016]. Recently, NARM [2017a] designs a global and local RNN recommender to capture user’s sequential behaviors and main purpose simultaneously. Similar to NARM, STAMP [2018] also captures users’ general interests and current interests, by employing simple MLP networks and an attentive net.

*Figure 1: Workflow of the session-based recommendation with graph neural network. All session sequences are aggregated together and model as an item graph, then node vectors can be obtained through a gated graph neural network. After that, each session is represented as the composition of the global preference and current interests of this session by using an attention net. Finally, we predict the probability of each item that will appear to be the next-click one for each session.*

Although the methods above achieve satisfactory results and become the state-of-the-arts, they still have some limitations. At first, without adequate user behaviors in one session, these RNN-based methods have difficulty in well-estimating user representations. Usually, the hidden vectors of these RNN methods are treated as the user representations, such that recommendations can be generated based on these representations, e.g., in the global recommender of NARM. In session-based recommendation systems, however, sessions are mostly anonymous and numerous, and user behaviors in each session are very limited. It is difficult to well-estimate the representation of each user from each session. Besides, those RNN models use a hidden vector representing the user’s general interest, which perform badly when the session is long, due to the presence of interest drift. Secondly, previous work reveals that patterns of item transition are important and can be used as a local factor [2017a, 2018] in the session-based recommendation. But these significant patterns can not always be well captured by previous methods. For example, NARM implicitly models the effect of item transitions by using user representations, STAMP treats previous clicked items as a whole, and considers the transition between the whole part and a next item. Finally, implicit item transitions are difficult to be inferred and used in previous work. The existing methods tend to treat each session as a separated item chain, and can not well-identify the complex item transitions among these sessions.

To overcome the limitations mentioned above, we propose a novel method for session-based recommendation with graph neural networks, SR-GNN for brevity, to generate accurate latent vectors of items and explore rich transitions among items. Graph Neural Networks (GNN) [2009, 2015] are designed for generating representations for directed graphs. Recently, it has been successfully employed to model graph-structural dependency for natural language process and computer vision applications, e.g., script event prediction [2018], situation recognition [2017b], and image classification [2017]. For the session-based recommendation, we first construct a directed item graph based on all historical session sequences. Based on this graph, GNN can capture complex transitions of items, which are difficult to be revealed by the conventional sequential methods, like MC-based and RNN-based methods. Then, using GNN, rich item transitions can be fully explored and the next-click item can be inferred based on the item graph structure and embeddings. Under this model, SR-GNN does not rely on user representations to reveal the session preference and it represents a session only using the items in that session.

Figure 1 illustrates the workflow of the proposed SR-GNN, i.e. session-based recommendation with graph neural networks. First, all session sequences are aggregated together and modeled as a directed item graph, where each session sequence can be treated as a subgraph. Then, we can learn the latent vectors for all nodes involved in each subgraph through a gated graph neural network. After that, the session can be represented as a composition of the global preference and the current interest of the user in that session. These global and local session embeddings are both composed by the latent vectors of nodes. Finally, for each session, we predict the probability of each item that will appear to be the next click. Extensive experiments conducted on real-world representative datasets demonstrate the effectiveness of the proposed method over the state-of-arts. The main contributions of this work are listed as follows:

-

We aggregate separated session sequences into graph-structure data and use graph neural network to capture complex items transitions. It presents a novel perspective on modeling of session-based recommendation.

-

To generate session-based recommendations, we do not rely on latent vectors of users, but use the session embedding, which can be obtained merely based on latent vectors of nodes involved in the session subgraph.

-

Extensive experiments conducted on real-world datasets show that SR-GNN evidently outperforms the state-of-art methods and obtains stable performance under different experimental setting.

The rest of this paper is organized as follows. We review prior related work in Section II. Section III presents the proposed method of session-based recommendation with graph neural networks. Detailed experiment results and analysis are shown in Section IV. Finally, we conclude this paper in Section V.

## Related Work

In this section, we review some related work on session-based recommendation systems, including conventional methods, sequential methods based on Markov chains, and RNN-based mthods, then we introduce the neural networks on graphs.

Conventional recommendation methods. Matrix factorization [2007, 2009, 2011] is a general approach to recommender systems. The basic objective is to factorize a user-item rating matrix into two low-rank matrices, each of which represents the latent factors of users or items. It is not very suitable for the session-based recommendation, because the user preference is only provided by some positive clicks. The item-based neighborhood methods [2001] is a natural solution, in which item similarities are calculated on the co-occurrence in the same session. These methods have difficulty in considering the sequential order of items and generate prediction merely based on the last click.

Then the sequential methods based on Markov chain are proposed, which predict users’ next behaviors based on the previous ones. Treating recommendation generation as a sequential optimization problem, ? (?) employ Markov decision processes (MDPs) for the solution. Via factorization of the personalized probability transition matrices of users, FPMC [2010] models sequential behaviors between every two adjacent clicks and provides a more accurate prediction for each sequence. However, the main drawback of Markov chain based models is a independent combination of the past components, which lies in a strong independence assumption and confines the prediction accuracy.

Deep learning based methods. Recently, some prediction models, especially language models [2013] are proposed based on neural networks. Among numerous language models, RNN has been the most successful one in modeling sentences [2010]. It has successfully applied in various natural language processing tasks, such as machine translation [2014], conversation machine [2016] and image caption [2015]. RNN also have been applied successfully in numerous applications, such as the sequential click prediction [2014], location prediction [2016], and next basket recommendation [2016].

For session-based recommendation, the work of [2016a] proposes the recurrent neural network approach, and then extends to an architecture with parallel RNNs [2016b] which can model sessions based on the clicks and features of the clicked items. Then several works are proposed based on these RNN methods. ? (?) enhance the performance of recurrent model by using proper data augmentation techniques and accounting for temporal shifts in user behavior. ? (?) combine the recurrent method and the neighborhood-based method together to mix the sequential patterns and co-occurrence signals. ? (?) combine session clicks and content features such as item descriptions and item categories to generate recommendations by using 3-dimensional convolutional neural networks. A list-wise deep neural network [2017] models the limited user behaviors within each session, and uses a list-wise ranking model to generate the recommendation for each session. A neural attentive recommendation machine with an encoder-decoder architecture, i.e. NARM [2017a] employs the attention mechanism on RNN to capture sequential behavior feature and main purpose feature of users. Then, a short-term attention priority model (STAMP) [2018] using simple MLP networks and an attentive net, is proposed to efficiently capture users’ general interests and current interests.

Neural network on graphs. Recently, neural network has been employed for generating representation for graph-structured data, e.g., social network and knowledge bases. Extending the word2vec [2013], a unsupervised algorithm deepwalk [2014] is designed to learn representations of graph nodes based on random walk. Following deepwalk, unsupervised network embedding algorithms LINE [2015] and node2vec [2016] are most representative methods. On the another hand, the classical neural network CNN and RNN are also deployed on graph-structured data. [2015] introduces a convolutional neural network that operates directly on graphs of arbitrary size and shape. A scalable approach [2016] chooses convolutional architecture via a localized approximation of spectral graph convolutions, which is an efficient variant and can operate directly on graphs. These methods only can be implemented on undirected graph. Previously, in form of recurrent neural networks, Graph Neural Networks (GNN) [2005, 2009] are proposed to operate on directed graphs. As a modification of GNN, gated GNN [2015] uses gated recurrent units and employs backpropagation through time to compute gradients. Recently, GNN is broadly applied for the different tasks, e.g., script event prediction [2018], situation recognition [2017b], image classification [2017].

## The Proposed Method

Here, we introduce the proposed SR-GNN applying graph neural networks into session-based recommendation. We formulate the problem first, then show how to construct the graph from sessions, and finally describe SR-GNN thoroughly.

### Notations

Session-based recommendation aims to predict which item a user will click next, solely based on the user’s current sequential session data without accessing to the long-term preference profile. Here we will give a formulation of this problem.

In session-based recommendation, let $V=\{v_{1},v_{2},\dots,v_{m}\}$ denotes the set consisting of all unique items involved in all the sessions. An anonymous session sequence $s$ can be represented by a list $s$ $=$ $[v_{s,1},v_{s,2},\dots,v_{s,n}]$ ordered by timestamps, where $v_{s,i}\in V$ represents a clicked item of the user within the session $s$. The goal of the session-based recommendation is to predict the next click (i.e., sequence label) $v_{s,n+1}$ for the session $s$. Under a session-based recommendation model, for the session $s$, we output probabilities $\hat{\mathbf{y}}$ for all possible items, where an element value of vector $\hat{\mathbf{y}}$ is the recommendation score of the corresponding item. The items with top-$K$ values in $\hat{\mathbf{y}}$ will be the candidate items for recommendation.

### Constructing session subgraphs and the global graph

Each session sequence $s$ can be modeled as a directed subgraph $\mathcal{G}_{s}=(\mathcal{V}_{s},\mathcal{E}_{s})$. In this session subgraph, each node represents an item $v_{s,i}\in V$. Each edge $(v_{s,i-1},v_{s,i})\in\mathcal{E}_{s}$ means that a user clicks item $v_{s,i}$ after $v_{s,i-1}$ in the session $s$. The node vector $\mathbf{v}_{s,i}\in\mathbb{R}^{d}$ indicates the latent vector of item $v_{s,i}$ learned via graph neural networks, where $d$ is the dimensionality. Based on node vectors, each session $s$ can be represented by an embedding $\mathbf{s}$, which is composed of node vectors in that subgraph.

Considering there are different kinds of item connections in different datasets, we further aggregate all session sequences together and model them as a directed whole item graph, which is termed as the global graph hereafter. The global graph will help establish connections between items in each subgraph and further facilitate GNN to attain accurate node vectors. We use $\mathcal{G}=(\mathcal{V},\mathcal{E})$ to denote the global graph, where $\mathcal{V}$ is the node set and $\mathcal{E}$ is the edge set. In the global graph $\mathcal{G}$, each node denotes a unique item, and each edge denotes a directed transition from one item to another.

For SR-GNN, we consider constructing a global graph with binary edges, where each edge is a boolean value indicating whether there exists a transition from one item to another in all subgraphs. In other words, the global graph with binary edges can be regarded as the aggregation of all subgraphs with duplicate edges removed. Please note that the proposed method has the flexibility to model various kinds of connections between nodes. More variants of connection schemes are discussed and analyzed in Section IV.

### Implementing graph neural networks

Here, we show how to obtain latent vectors of nodes in the global graph via graph neural networks. The vanilla graph neural network is proposed by ? (?), extending neural network methods for processing the graph-structured data. ? (?) further introduce gated recurrent units and backpropagation through time to GNN and proposes gated GNN. Graph neural networks are well-suited for session-based recommendation, because it can automatically extract features of session subgraphs with considerations of rich node connections. We first demonstrate the learning process of node vectors in a session subgraph. Formally, for the node $v_{s,i}$ of subgraph $\mathcal{G}_{s}$, the update functions are given as follows:

$\displaystyle\mathbf{a}^{t}_{s,i}$ $\displaystyle=\mathbf{A}_{s}^{\top}\left[\mathbf{v}^{t-1}_{1},\dots,\mathbf{v}^{t-1}_{m}\right]^{\top}+\mathbf{b},$ | | | | | (1) |

$\displaystyle\mathbf{z}^{t}_{s,i}$ $\displaystyle=\sigma\left(\mathbf{W}_{z}\mathbf{a}^{t}_{s,i}+\mathbf{U}_{z}\mathbf{v}^{t-1}_{s,i}\right),$ | | | | | (2) |

$\displaystyle\mathbf{r}^{t}_{s,i}$ $\displaystyle=\sigma\left(\mathbf{W}_{r}\mathbf{a}^{t}_{s,i}+\mathbf{U}_{r}\mathbf{v}^{t-1}_{s,i}\right),$ | | | | | (3) |

$\displaystyle\widetilde{\mathbf{v}^{t}_{s,i}}$ $\displaystyle=\tanh\left(\mathbf{W}_{o}\mathbf{a}^{t}_{s,i}+\mathbf{U}_{o}\left(\mathbf{r}^{t}_{s,i}\odot\mathbf{v}^{t-1}_{s,i}\right)\right),$ | | | | | (4) |

$\displaystyle\mathbf{v}^{t}_{s,i}$ $\displaystyle=\left(1-\mathbf{z}^{t}_{s,i}\right)\odot\mathbf{v}^{t-1}_{s,i}+\mathbf{z}^{t}_{s,i}\odot\widetilde{\mathbf{v}^{t}_{s,i}},$ | | | | | (5) |

where $\mathbf{z}_{s,i}$ and $\mathbf{r}_{s,i}$ are the reset and update gates respectively, $\sigma(\cdot)$ is the sigmoid function, and $\odot$ is the element-wise multiplication operator. $\mathbf{v}_{s,i}\in\mathbb{R}^{d}$ indicates the latent vector of node $v_{s,i}$. The connection matrix $\mathbf{A}_{s}$ determines how nodes in the subgraph communicate with each other based on the global graph, $\left[\mathbf{v}^{t-1}_{1},\dots,\mathbf{v}^{t-1}_{m}\right]$ is the list of all node vectors.

Here $\mathbf{A}_{s}$ is defined as a concatenation of two adjacency matrices $\mathbf{A}_{s}^{\text{(out)}}$ and $\mathbf{A}_{s}^{\text{(in)}}$, which represents outgoing and incoming edges in the subgraph. The proposed SR-GNN uses the global graph with binary edges. For example, consider a session $s=[v_{1},v_{2},v_{4},v_{3}]$, the corresponding subgraphs $\mathcal{G}_{s}$ and the matrix $\mathbf{A}_{s}$ are shown in Figure 2. Please note that SR-GNN can support different connection matrices $\mathbf{A}_{s}$ for various kinds of constructed global graphs as mentioned earlier. If different strategies of constructing the global graph are used, the connection matrix $\mathbf{A}_{s}$ will be changed accordingly. Moreover, when there exists content features of node, such as descriptions and category information, the method can be further generalized, e.g., concatenating features with node vector, to deal with this kinds of information.

For each session subgraph $\mathcal{G}_{s}$, the gated graph neural network proceeds nodes at the same time. Eq. (1) is used for information propagation between different nodes, under restrictions given by the matrix $\mathbf{A}_{s}$. Specifically, it extracts the latent vectors of neighborhoods and feeds them as input into the graph neural network. Then, two gates, i.e. update and reset gate, decide what information to be preserved and discarded respectively. After that, Eq. (4) constructs the candidate state by the previous state, the current state, and the reset gate. The final state is then the combination of the previous hidden state and the candidate state, under the control of the update gate. After updating all nodes in session subgraphs until convergence, we can obtain the final node vectors.

*Figure 2: A example of subgraph and its corresponding connection matrix $\mathbf{A}_{s}$*

### Generating session embeddings

Previous session-based recommendation methods always assume there exists a distinct latent vector of user for each session. The proposed SR-GNN method does not make any assumptions on that vector. Instead, a session is represented directly by nodes involved in that session. To better predict the users’ next clicks, we plan to develop a strategy to combine long-term preference and current interests of the session, and use this combined embedding as the session embedding. After feeding all session subgraphs into the gated graph neural networks, we obtain the vectors of all nodes. Then, to represent each session as an embedding vector $\mathbf{s}\in\mathbb{R}^{d}$, we first consider the local embedding $\mathbf{s}_{l}$ of session $s$. For session $s=[v_{s,1},v_{s,2},\dots,v_{s,n}]$, the local embedding can be simply defined as $\mathbf{v}_{s,n}$ of the last clicked item $v_{s,n}$, i.e. $\mathbf{s}_{l}=\mathbf{v}_{s,n}$.

Then, we consider the global embedding $\mathbf{s}_{g}$ of the session subgraph $\mathcal{G}_{s}$ by aggregating all node vectors. Consider information in these embedding may have different levels of priority, we further adopt the soft-attention mechanism to better represent the global session behaviors:

$\displaystyle\alpha_{i}$ $\displaystyle=\mathbf{q}^{\top}\,\sigma(\mathbf{W}_{1}\mathbf{v}_{s,n}+\mathbf{W}_{2}\mathbf{v}_{s,i}+\mathbf{b}),$ | | | | | (6) |

$\displaystyle\mathbf{s}_{g}$ $\displaystyle=\sum\limits_{i=1}^{m}{\alpha_{i}\mathbf{v}_{s,i}},$ | | | | |

where parameters $\mathbf{q}\in\mathbb{R}^{d}$ and $\mathbf{W}_{1},\mathbf{W}_{2}\in\mathbb{R}^{d\times d}$ control the weight of item embeddings.

Finally, we compute the hybrid embedding $\mathbf{s}_{h}$ by taking linear transformation over the concatenation of the local and global embedding vectors:

$\mathbf{s}_{h}=\mathbf{W}_{3}\left[\mathbf{s}_{l};\mathbf{s}_{g}\right],$ | | | | (7) |

where matrix $\mathbf{W}_{3}\in\mathbb{R}^{d\times 2d}$ compresses two combined embedding vectors into the latent space $\mathbb{R}^{d}$.

### Loss function

After obtained the embedding of each session, we compute the score $\hat{\mathbf{z}_{i}}$ for each candidate item $v_{i}\in V$ by multiplying its embedding $\mathbf{v}_{i}$ by session representation $\mathbf{s}_{h}$, which can be defined as:

$\hat{\mathbf{z}_{i}}=\mathbf{s}_{h}^{\top}\,\mathbf{v}_{i}.$ | | | | (8) |

Then we apply a softmax function to get the output vector of the model $\hat{\mathbf{y}}$:

$\hat{\mathbf{y}}={softmax}\left(\hat{\mathbf{z}}\right),$ | | | | (9) |

where $\hat{\mathbf{z}}\in\mathbb{R}^{m}$ denotes the recommendation scores over all candidate items and $\hat{\mathbf{y}}\in\mathbb{R}^{m}$ denotes the probabilities of nodes appearing to be the next click in the session $s$.

For each session subgraph, the loss function is defined as the cross-entropy of the prediction and the ground truth. It can be written as follows:

$\mathcal{L}(\hat{\mathbf{y}})=-\sum_{i=1}^{m}\mathbf{y}_{i}\log{(\hat{\mathbf{y}_{i}})}+(1-\mathbf{y}_{i})\log{(1-\hat{\mathbf{y}_{i}})},$ | | | | (10) |

where $\mathbf{y}$ denotes the one-hot encoding vector of the ground truth item. Finally, we use the Back-Propagation Through Time (BPTT) algorithm to train the proposed SR-GNN model until convergence.

## Experiments and Analysis

In this section, we first describe the datasets, compared methods and evaluation metrics in the experiments. Then, we compare the proposed SR-GNN with other comparative methods. Finally, we make some detailed analysis of SR-GNN under different experimental settings.

### Datasets

We evaluate the proposed method on two real-world representative datasets, i.e. Yoochoose and Diginetica. The Yoochoose dataset is obtained from the RecSys Challenge 2015, which contains a stream of user clicks on an e-commerce website within 6 months. The Diginetica dataset comes from CIKM Cup 2016, where only its transactional data is used.

*Table 1: Statistics of datasets used in the experiments*

| Statistics | Yoochoose 1/64 | Yoochoose 1/4 | Diginetica |

| # of clicks | 557248 | 8326407 | 982961 |

| # of training sessions | 369859 | 5917745 | 719470 |

| # of test sessions | 55898 | 55898 | 60858 |

| # of items | 16766 | 29618 | 43097 |

| average length | 6.16 | 5.71 | 5.12 |

For fair comparison, following [2017a, 2018], we filter out all sessions of length 1 and items appearing less than 5 times in both datasets. The remaining 7981580 sessions and 37483 items constitute the Yoochoose dataset, while 204771 sessions and 43097 items construct the Diginetica dataset. Furthermore, similar to [2016], we generate sequences and corresponding labels by splitting the input sequence. To be specific, for an input session $s=[v_{s,1},v_{s,2},\dots,v_{s,n}]$, we generate a series of sequences and labels $([v_{s,1}],v_{s,2}),([v_{s,1},v_{s,2}],v_{s,3}),\dots,$ $([v_{s,1},v_{s,2},\dots,v_{s,n-1}],v_{s,n})$. For example, in $([v_{s,1},v_{s,2},\dots,v_{s,n-1}],v_{s,n})$, $[v_{s,1},v_{s,2},\dots,v_{s,n-1}]$ is the generated sequence, and $v_{s,n}$ denotes the next-clicked item, i.e. the label of the sequence. Following [2017a, 2018], we also use the most recent fractions 1/64 and 1/4 of the training sequences of Yoochoose. The statistics of datasets are summarized in Table 1.

### Baseline Algorithms

To evaluate the performance of the proposed SR-GNN, we compare our method with the following representative methods:

-

POP and S-POP recommend the top-$N$ frequent items in the training set and in the current session respectively.

-

Item-KNN [2001] recommends items similar to the previously clicked item in the session, where similarity is defined as the cosine similarity between the vector of sessions.

-

BPR-MF [2009] optimizes a pairwise ranking objective function via stochastic gradient descent.

-

FPMC [2010] is a sequential prediction method based on markov chain.

-

GRU4REC [2016a] uses RNNs to model user sequences for the session-based recommendation.

-

NARM [2017a] employs RNNs with attention mechanism to capture the user’s main purpose and sequential behavior.

-

STAMP [2018] captures users’ general interests of the current session and current interests of the last click.

### Evaluation Metrics

Following metrics are used to evaluate compared methods.

P@20 (Precision) is widely used as a measure of predictive accuracy. It represents the proportion of correctly recommended items amongst the top-$20$ items.

MRR@20 (Mean Reciprocal Rank) is the average of reciprocal ranks of the correctly-recommended items. The reciprocal rank is set to 0 when the rank exceeds 20. The MRR measure considers the order of recommendation ranking, where large MRR value indicates that correct recommendations in the top of the ranking list.

### Parameter Setup

Following previous methods [2017a, 2018], we set the dimensionality of latent vectors $d=100$ for both datasets. Besides, we select other hyper-parameters on a validation set which is a random $10\%$ subset of the training set. All parameters are initialized with a Gaussian distribution with a mean of 0 and a standard deviation 0.1. The mini-batch Adam optimizer is exerted to optimize these parameters, where the learning rate is set to 0.01. Moreover, the batch size is set to 100 and the weight decay of the L2 norm is set to $10^{-5}$, respectively.

### Comparison among baseline methods

To demonstrate the overall performance of the proposed model, we compare it with other state-of-art session-based recommendation methods. The overall performance in terms of P@20 and MRR@20 is shown in Table 2, with the best results highlighted in boldface. Please note that, as in [2017a], due to insufficient memory to initialize FPMC, the performance on Yoochoose 1/4 is not reported.

*Table 2: The performance of SR-GNN with other baseline methods over three datasets*

| Algorithm | Yoochoose 1/64 | Yoochoose 1/4 | Diginetica |

| P@20 | MRR@20 | P@20 | MRR@20 | P@20 | MRR@20 |

| POP | 6.71 | 1.65 | 1.33 | 0.30 | 0.91 | 0.23 |

| S-POP | 30.44 | 18.35 | 27.08 | 17.75 | 21.07 | 14.69 |

| Item-KNN | 51.60 | 21.81 | 52.31 | 21.70 | 28.35 | 9.45 |

| BPR-MF | 31.31 | 12.08 | 3.40 | 1.57 | 15.19 | 8.63 |

| FPMC | 45.62 | 15.01 | – | – | 31.55 | 8.92 |

| GRU4REC | 60.64 | 22.89 | 59.53 | 22.60 | 43.82 | 15.46 |

| NARM | 68.32 | 28.63 | 69.73 | 29.23 | 62.58 | 27.35 |

| STAMP | 68.74 | 29.67 | 70.44 | 30.00 | 62.03 | 27.38 |

| SR-GNN | 70.57 | 30.94 | 71.36 | 31.89 | 63.03 | 27.42 |

SR-GNN aggregates separated session sequences into graph-structure data. In this model, we jointly consider the global session preference as well as the local interests. According to the experiments, it is obvious that the proposed SR-GNN method achieves the best performance among all methods on the three datasets in terms of P@20 and MRR@20. This verifies the effectiveness of the proposed method.

Regarding those traditional algorithms like POP and S-POP, their performance is relatively poor. Such simple models make recommendations solely based on repetitive co-occurred items or successive items, which is problematic in session-based recommendation scenarios. Even so, the S-POP still outperforms its opponents such as POP, BPR-MF, and FPMC, demonstrating the importance of session contextual information. Item-KNN achieves better results than FPMC based on Markov chains. Please note that, Item-KNN utilizes only the similarity between items without considerations of sequential information. This indicates that the assumption on the independence of successive items, which traditional MC-based methods mostly rely on, is not realistic.

Neural-network-based methods, such as NARM and STAMP, outperform the conventional methods, demonstrating the power of adopting deep learning in this domain. Short/long-term memory models, like GRU4REC and NARM, use recurrent units to capture a user’s general interest while STAMP improves the short-term memory by utilizing the last-clicked item. Those methods explicitly model the users’ global behavioral preferences and consider transitions between users’ previous actions and the next click, leading to superior performance against these traditional methods. However, their performance is still inferior to that of the proposed method. Compared with the state-of-art methods like NARM and STAMP, SR-GNN further considers transitions between items in a session and thereby models every session as a subgraph, which can capture more complex and implicit connections between user clicks. Whereas in NARM and GRU4REC, they explicitly model each user and obtain the user representations through separated session sequences, with possible interactive relationships between items ignored. Therefore, the proposed model is more powerful to model session behaviors.

Besides, SR-GNN adopts the soft-attention mechanism to generate a session representation which can automatically select the most significant item transitions, and neglect noisy and ineffective user actions in the current session. On the contrary, STAMP only uses the transition between the last-clicked item and previous actions, which may not be sufficient. Other RNN models, such as GRU4REC and NARM, fail to select impactful information during the propagation process as well. They use all previous items to obtain a vector representing the user’s general interest. When a user’s behavior is aimless, or his interests drift quickly in the current session, conventional models are ineffective to cope with noisy sessions.

### Comparison with different connection schemes

The proposed SR-GNN method is flexible in constructing connecting relationships between items in the graph. Here we propose another two connection schemes:

-

SR-GNN with full connections (SR-GNN-FC) explicitly models all high-order relationships between items as direct connections, represented by boolean weights.

-

SR-GNN with normalized full connections (SR-GNN-N) further normalizes each edge weight on the basis of SR-GNN. The edge weight is computed as the occurrence of the edge divided by the outdegree of the edge start node.

The results of different connection schemes are shown in Figure 3. From the figures, it is seen that all three connection schemes achieve better or almost the same performance as the state-of-the-art STAMP and NARM methods, confirming the usefulness of modeling sessions as graphs.

For SR-GNN and SR-GNN-FC, the former one only models the exact relationship between consecutive items, and the latter one further explicitly regards all high-order relationships as direct connections. It is reported that SR-GNN-FC performs worse than SR-GNN, though the experimental results of the two methods are not much different. Such a small difference in results suggests that in most recommendation scenarios, not every high-order transitions can be directly converted to straight connections and intermediate stages between high-order items are still necessities. For instance, consider that the user has viewed the following pages when browsing a website: $A\rightarrow B\rightarrow C$. Recommending page $C$ directly after $A$ without intermediate page $B$ is not appropriate, due to the lack of a direct connection between $A$ and $C$.

Compared with SR-GNN, for each session, SR-GNN-N takes the impact of other sessions into considerations in addition to items in the current session, which subsequently reduces the influence of edges that are connected to nodes with high degree within the current session subgraph. Such a fusion method notably affects the integrity of the current session, especially when the weight of the edge in the graph varies. Therefore, SR-GNN-N is not suitable for the current training strategy, which learns a subgraph once at a time. However, if adopting some other parallel learning strategies, such as mini-batch, which feeds a batch of subgraphs into GNN once, SR-GNN-N may be able to outperform other methods.

*(a) P@20*

*(b) MRR@20*

*Figure 3: The performance of different connection schemes*

### Comparison with different session representations

We compare the session embedding strategy with the following three approaches: (1) local embeddings only (SR-GNN-L), (2) global embeddings with average pooling (SR-GNN-AVG), and (3) global embeddings with the attention mechanism (SR-GNN-ATT). The results of methods with three different embedding strategy are given in Figure 4.

*(a) P@20*

*(b) MRR@20*

*Figure 4: The performance of different session representations*

From the figures, it can be observed that the hybrid embedding method SR-GNN achieves best results on all three datasets, which validates the importance of explicitly incorporating current session interests with the long-term preference. Furthermore, the figures show that SR-GNN-ATT performs better than SR-GNN-AVG with average pooling on three datasets. It indicates that the session maybe contains some noisy behaviors, these behaviors can not be treated as independent. Besides, attention mechanisms is very helpful in extracting the significant behaviors from the session data to construct the long-term perference.

Please note that SR-GNN-L, as a downgraded version of SR-GNN, still outperforms SR-GNN-AVG and achieves almost the same performance as of SR-GNN-ATT, supporting that both the current interest and long-term perference are crucial for the session-based recommendation.

### Analysis on session sequence lengths

We further analyze the capability of different models to cope with sessions of different lengths. For comparison, we partition sessions of Yoochoose 1/64 and Diginetica into two groups, where “Short” indicates that the length of sessions is less than or equal to 5, while each session has more than 5 items in “Long”. The pivot value 5 is chosen because it is the closest integer to the average length of total sessions in all datasets. The percentages of session belonging to short group and long group are 0.701 and 0.299 on the Yoochoose data, and 0.764 and 0.236 on the Diginetica data. For each method, we demonstrate the results evaluated by P@20 in Table 3.

Our proposed SR-GNN and variants perform stably on two datasets with different categories. It demonstrates the superior performance of proposed graph neural network in adaptability among different datasets. On the contrary, the performance of STAMP changes greatly in short and long groups. STAMP [2018] explains such a difference according to replicated actions. It adopts the attention mechanism, so replicated items can be ignored when obtaining user representations. Similar to STAMP, on Yoochoose, NARM achieves good performance on the short group, but the performance decreases very quickly with the length of the sessions increasing. This is partially because RNN models have difficulty to cope with long sequences.

Then we analyze the performance of SR-GNN-L, SR-GNN-ATT, and SR-GNN with different session representations. These three methods achieve promising results comparing with STAMP and NARM. It is probably because that based on the learning framework of graph neural network, our methods can attain more accurate node vectors. Such node embedding not only captures the latent features of nodes but also models the node connections in the global graph. On such basis, the performance is stable among variants of SR-GNN, while the performance of two state-of-art methods fluctuate considerably on short and long datasets. Moreover, the table shows that SR-GNN-L can also achieve good results, although this variant only uses local session embeddings. It is maybe because that SR-GNN-L also implicitly considers the properties of the first-order and higher-order nodes in subgraphs. Such results are also validated by Figure 4, where both SR-GNN-L and SR-GNN-ATT achieve the close-to-optimal performance.

*Table 3: The performance of different methods with different session lengths evaluated in terms of P@20*

| Method | Yoochoose 1/64 | Diginetica |

| Short | Long | Short | Long |

| NARM | 71.44 | 60.79 | 62.04 | 64.33 |

| STAMP | 70.69 | 64.73 | 59.91 | 64.58 |

| SR-GNN-L | 70.11 | 69.73 | 59.2 | 60.34 |

| SR-GNN-ATT | 70.31 | 70.64 | 62.72 | 63.55 |

| SR-GNN | 70.47 | 70.70 | 62.97 | 63.92 |

## Conclusion

Session-based recommendation is indispensable where users’ preference and historical records are hard to obtain. This paper presents a novel architecture for session-based recommendations that incorporates graph models into representing session sequences. The proposed method not only considers complex structure and transition between items of session sequences, but also develops a strategy to combine long-term preference and current interests of sessions to better predict the users’ next actions. Comprehensive experiments confirm that the proposed algorithm can consistently outperform other state-of-art methods.

## References

- [2014] Cho, K.; Van Merriënboer, B.; Gulcehre, C.; Bahdanau, D.; Bougares, F.; Schwenk, H.; and Bengio, Y. 2014. Learning phrase representations using rnn encoder-decoder for statistical machine translation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing 1724–1734.

- [2015] Duvenaud, D.; Maclaurin, D.; Aguilera-Iparraguirre, J.; Gómez-Bombarelli, R.; Hirzel, T.; Aspuru-Guzik, A.; and Adams, R. P. 2015. Convolutional networks on graphs for learning molecular fingerprints. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2, NIPS’15.

- [2005] Gori, M.; Monfardini, G.; and Scarselli, F. 2005. A new model for learning in graph domains. In Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005., volume 2, 729–734 vol. 2.

- [2016] Grover, A., and Leskovec, J. 2016. Node2vec: Scalable feature learning for networks. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, 855–864. New York, NY, USA: ACM.

- [2016a] Hidasi, B.; Karatzoglou, A.; Baltrunas, L.; and Tikk, D. 2016a. Session-based recommendations with recurrent neural networks. In Proceedings of the 2016 International Conference on Learning Representations, ICLR ’16.

- [2016b] Hidasi, B.; Quadrana, M.; Karatzoglou, A.; and Tikk, D. 2016b. Parallel recurrent neural network architectures for feature-rich session-based recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems, RecSys ’16, 241–248. New York, NY, USA: ACM.

- [2017] Jannach, D., and Ludewig, M. 2017. When recurrent neural networks meet the neighborhood for session-based recommendation. In Proceedings of the Eleventh ACM Conference on Recommender Systems, RecSys ’17, 306–310. New York, NY, USA: ACM.

- [2016] Kipf, T. N., and Welling, M. 2016. Semi-supervised classification with graph convolutional networks. In Proceedings of the 2016 International Conference on Learning Representations, ICLR ’16.

- [2011] Koren, Y., and Bell, R. 2011. Advances in collaborative filtering. In Recommender Systems Handbook. Springer. 145–186.

- [2009] Koren, Y.; Bell, R.; and Volinsky, C. 2009. Matrix factorization techniques for recommender systems. Computer 42(8):30–37.

- [2015] Li, Y.; Tarlow, D.; Brockschmidt, M.; and Zemel, R. S. 2015. Gated graph sequence neural networks. In Proceedings of the 2015 International Conference on Learning Representations, volume abs/1511.05493 of ICLR ’15.

- [2017a] Li, J.; Ren, P.; Chen, Z.; Ren, Z.; Lian, T.; and Ma, J. 2017a. Neural attentive session-based recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, CIKM ’17, 1419–1428. New York, NY, USA: ACM.

- [2017b] Li, R.; Tapaswi, M.; Liao, R.; Jia, J.; Urtasun, R.; and Fidler, S. 2017b. Situation recognition with graph neural networks. In 2017 IEEE International Conference on Computer Vision (ICCV), 4183–4192.

- [2018] Li, Z.; Ding, X.; and Liu, T. 2018. Constructing narrative event evolutionary graph for script event prediction.

- [2016] Liu, Q.; Wu, S.; Wang, L.; and Tan, T. 2016. Predicting the next location: A recurrent model with spatial and temporal contexts. In AAAI Conference on Artificial Intelligence, 194–200.

- [2018] Liu, Q.; Zeng, Y.; Mokhosi, R.; and Zhang, H. 2018. Stamp: Short-term attention/memory priority model for session-based recommendation. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’18, 1831–1839. New York, NY, USA: ACM.

- [2015] Mao, J.; Xu, W.; Yang, Y.; Wang, J.; Huang, Z.; and Yuille, A. 2015. Deep captioning with multimodal recurrent neural networks (m-rnn). International Conference on Learning Representations.

- [2017] Marino, K.; Salakhutdinov, R.; and Gupta, A. 2017. The more you know: Using knowledge graphs for image classification. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), volume 00, 20–28.

- [2010] Mikolov, T.; Karafiát, M.; Burget, L.; Cernockỳ, J.; and Khudanpur, S. 2010. Recurrent neural network based language model. In INTERSPEECH, volume 2, 3.

- [2013] Mikolov, T.; Sutskever, I.; Chen, K.; Corrado, G. S.; and Dean, J. 2013. Distributed representations of words and phrases and their compositionality. In Annual Conference on Neural Information Processing Systems, 3111–3119.

- [2007] Mnih, A., and Salakhutdinov, R. 2007. Probabilistic matrix factorization. In Advances in neural information processing systems, 1257–1264.

- [2014] Perozzi, B.; Al-Rfou, R.; and Skiena, S. 2014. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’14, 701–710. New York, NY, USA: ACM.

- [2009] Rendle, S.; Freudenthaler, C.; Gantner, Z.; and Schmidt-Thieme, L. 2009. Bpr: Bayesian personalized ranking from implicit feedback. In UAI, 452–461.

- [2010] Rendle, S.; Freudenthaler, C.; and Schmidt-Thieme, L. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th international conference on World wide web, 811–820. ACM.

- [2001] Sarwar, B.; Karypis, G.; Konstan, J.; and Riedl, J. 2001. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th International Conference on World Wide Web, WWW ’01.

- [2009] Scarselli, F.; Gori, M.; Tsoi, A. C.; Hagenbuchner, M.; and Monfardini, G. 2009. The graph neural network model. IEEE Transactions on Neural Networks 20(1):61–80.

- [2016] Serban, I. V.; Sordoni, A.; Bengio, Y.; Courville, A.; and Pineau, J. 2016. Building end-to-end dialogue systems using generative hierarchical neural network models. In Proceedings of the 30th AAAI Conference on Artificial Intelligence, 3776–3784.

- [2002] Shani, G.; Brafman, R. I.; and Heckerman, D. 2002. An mdp-based recommender system. In Proceedings of the Eighteenth Conference on Uncertainty in Artificial Intelligence, UAI’02, 453–460. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.

- [2016] Tan, Y. K.; Xu, X.; and Liu, Y. 2016. Improved recurrent neural networks for session-based recommendations. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, DLRS 2016, 17–22. New York, NY, USA: ACM.

- [2015] Tang, J.; Qu, M.; Wang, M.; Zhang, M.; Yan, J.; and Mei, Q. 2015. Line: Large-scale information network embedding. In Proceedings of the 24th International Conference on World Wide Web, WWW ’15, 1067–1077. Republic and Canton of Geneva, Switzerland: International World Wide Web Conferences Steering Committee.

- [2017] Tuan, T. X., and Phuong, T. M. 2017. 3d convolutional networks for session-based recommendation with content features. In Proceedings of the Eleventh ACM Conference on Recommender Systems, RecSys ’17, 138–146. New York, NY, USA: ACM.

- [2017] Wu, C., and Yan, M. 2017. Session-aware information embedding for e-commerce product recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, CIKM ’17, 2379–2382. New York, NY, USA: ACM.

- [2016] Yu, F.; Liu, Q.; Wu, S.; Wang, L.; and Tan, T. 2016. A dynamic recurrent basket recommendation model. In Proceedings of the 39nd international ACM SIGIR conference on Research and development in information retrieval. ACM.

- [2014] Zhang, Y.; Dai, H.; Xu, C.; Feng, J.; Wang, T.; Bian, J.; Wang, B.; and Liu, T.-Y. 2014. Sequential click prediction for sponsored search with recurrent neural networks. In AAAI Conference on Artificial Intelligence, 1369–1376.

<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2102.07831
     paper_id : 2102.07831
     source   : paper2skills-vault/papers/recommendation/NeuralNDCG-2102.07831.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

NeuralNDCG: Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting

arXiv:2102.07831v2 [cs.IR] 22 May 2021

Przemyslaw Pobrotyn? and Radoslaw Bialobrzeski?
ML Research at Allegro.pl mlr@allegro.pl

Abstract. Learning to Rank (LTR) algorithms are usually evaluated using Information Retrieval metrics like Normalised Discounted Cumulative Gain (NDCG) or Mean Average Precision. As these metrics rely on sorting predicted items’ scores (and thus, on items’ ranks), their derivatives are either undefined or zero everywhere. This makes them unsuitable for gradient-based optimisation, which is the usual method of learning appropriate scoring functions. Commonly used LTR loss functions are only loosely related to the evaluation metrics, causing a mismatch between the optimisation objective and the evaluation criterion. In this paper, we address this mismatch by proposing NeuralNDCG, a novel differentiable approximation to NDCG. Since NDCG relies on the nondifferentiable sorting operator, we obtain NeuralNDCG by relaxing that operator using NeuralSort, a differentiable approximation of sorting. As a result, we obtain a new ranking loss function which is an arbitrarily accurate approximation to the evaluation metric, thus closing the gap between the training and the evaluation of LTR models. We introduce two variants of the proposed loss function. Finally, the empirical evaluation shows that our proposed method outperforms previous work aimed at direct optimisation of NDCG and is competitive with the state-of-the-art methods.
Keywords: Learning to Rank · ranking metric optimisation · NDCG approximation

1

Introduction

Ranking is the problem of optimising, conditioned on some context, the ordering of a set of items in order to maximise a given metric. The metric is usually an Information Retrieval (IR) criterion chosen to correlate with user satisfaction.
Learning to Rank (LTR) is a machine learning approach to ranking, concerned with learning the function which optimises the items’ order from supervised data. In this work, without loss of generality, we assume our set of items are search results and the context in which we want to optimise their order is the user query.
?

Equal contribution.

2

P. Pobrotyn and R. Bialobrzeski.

Essentially, one would like to learn a function from search results into permutations. Since the space of all permutations grows factorially in the size of the search results set, the task of learning such a function directly becomes intractable. Thus, most common LTR algorithms resort to the approach known as score & sort. That is, instead of directly learning the correct permutation of the search results, one learns a scoring function which predicts relevancies of individual items, in the form of real-valued scores. Items are then sorted in the descending order of the scores and thus produced ordering is evaluated using an IR metric of choice. Typically, scoring functions are implemented as either gradient boosted trees [14] or Multilayer Perceptrons (MLP) [25]. Recently, there has been work in using the Transformer [28] architecture as a scoring function [21]. In order to learn a good scoring function, one needs a tagged dataset of query-search results pairs together with ground truth relevancy of each search result (in the context of a given query), as well as a loss function. There has been extensive research into constructing appropriate loss functions for LTR (see [19] for an overview of the field). Such loss functions fall into one of three categories: pointwise, pairwise or listwise. Pointwise approaches treat the problem as a simple regression or classification of the ground truth relevancy for each individual search result, foregoing possible interactions between items. In pairwise approaches, pairs of items are considered as independent variables and the function is learned to correctly indicate the preference among the pair. Examples include RankNet [5], LambdaRank [6] or LambdaMART [7]. However, IR metrics consider entire search results lists at once, unlike pointwise and pairwise algorithms. This mismatch has motivated listwise approaches, which compute the loss based on the scores of the entire list of search results. Two popular listwise approaches are ListNet [8] and ListMLE [29].
What these loss functions have in common is that they are either not connected or only loosely connected to the IR metrics used in the evaluation. The performance of LTR models is usually assessed using Normalised Discounted Cumulative Gain (NDCG) [16] or Mean Average Precision (MAP) [1]. Since such metrics rely on sorting the ground truth labels according to the scores predicted by the scoring function, they are either not differentiable or flat everywhere and thus cannot be used for gradient-based optimisation of the scoring function. As a result, there is a mismatch between objectives optimised by the aforementioned pairwise or listwise losses and metrics used for the evaluation, even though it can be shown that some of such losses provide upper bounds of IR measures [31], [30].
On the other hand, as demonstrated in [23], under certain assumptions on the class of the scoring functions, direct optimisation of IR measures on a large training set is guaranteed to achieve high test performance on the same IR measure. Thus, attempts to bridge the gap between LTR optimisation objectives and discontinuous evaluation metrics are an important research direction.
In this work, we propose a novel approach to directly optimise NDCG by approximating the sorting operator with NeuralSort [15]. Since the sorting operator is the source of discontinuity in NDCG (and other IR metrics), by substituting it with a differentiable approximation we obtain a smooth variant of the metric.

NeuralNDCG: Direct Optimisation of Ranking Metrics...

3

The main contributions of the paper are:
– We introduce NeuralNDCG, a novel smooth approximation of NDCG based on differentiable relaxation of the sorting operator. The variants of the proposed loss are discussed in Section 4.
– We evaluate a Context-Aware Ranker [21] trained with NeuralNDCG loss on Web30K [22] and Istella [12] datasets. We demonstrate favourable performance of NeuralNDCG as compared to baselines. In particular, NeuralNDCG outperforms ApproxNDCG [23], a competing method for direct optimisation of NDCG.
– We provide an open-source Pytorch [20] implementation allowing for the reproduction of our results available as part of the open-source allRank framework1 .
The rest of the paper is organised as follows. In Section 2, we review the related literature. In Section 3 we formalise the problem of LTR. In Section 4, we recap NeuralSort and demonstrate how it can be used to construct a novel loss function, NeuralNDCG. In Section 5 we discuss our experimental setup and results. In the final Section 6 we summarise our findings and discuss the possible future work.

2

Related work

As already mentioned in the introduction, most LTR approaches can be classified into one of three categories: pointwise, pairwise or listwise. For a comprehensive overview of the field and most common approaches, we refer the reader to [19].
In this work, we are concerned with the direct optimisation of non-smooth IR measures. Methods for optimisation of such metrics can be broadly grouped into two categories. The methods in the first category try to optimise the upper bounds of IR metrics as surrogate loss functions. Examples include SVMmap [31]
and SVMNDCG [9] which optimise upper bounds on 1 − MAP and 1 − NDCG, respectively. On the other hand, ListNet was originally designed to minimise cross-entropy between predicted and ground truth top-one probability distributions, and as such its relation to NDCG was ill-understood. Only recently was it shown to bound NDCG and Mean Reciprocal Rank (MRR) for binary labels [3]. Further, a modification to ListNet was proposed in [2] for which it can be shown that it bounds NDCG also for the graded relevance labels. Popular methods like LambdaRank and LambdaMART forgo explicit formulation of the loss function and instead heuristically formulate the gradients based on NDCG considerations. Since the exact loss function is unknown, its theoretical relation to NDCG is difficult to analyse.
The second category of methods aims to approximate an IR measure with a smooth function and directly optimise resulting surrogate function. Our method falls into this category. We propose to smooth-out NDCG by approximating 1

https://github.com/allegro/allRank

4

P. Pobrotyn and R. Bialobrzeski.

non-continuous sorting operator used in the computation of that measure. Recent works proposing continuous approximation to sorting are already mentioned NeuralSort, SoDeep [13] and smooth sorting as an Optimal Transport problem [11]. We use NeuralSort for its firm mathematical foundation, the possibility to control the degree of approximation and ability to generalise beyond the maximum list length seen in training. SoDeep uses a deep neural network (DNN) and synthetic data to learn to approximate the sorting operator and as such lacks the aforementioned properties. Smooth sorting as Optimal Transport reports similar performance to NeuralSort at benchmark tasks and we aim to explore the use of it in NeuralNDCG in the future. By replacing the sorting operator with its continuous approximation, we obtain NeuralNDCG, a differentiable approximation of the IR measure. Other notable methods for direct optimisation of NDCG include:
– ApproxNDCG in which authors reformulated NDCG formula to involve summation over documents, not their ranks. As a result, they introduce a nondifferentiable position function, which they approximate using a sigmoid.
This loss has been recently revisited in a DNN setting in [4].
– SoftRank [27], where authors propose to smooth scores returned by the scoring function with equal variance Gaussian distributions: thus deterministic scores become means of Gaussian score distributions. Subsequently, they derive an O(n3 ) algorithm to compute Rank-Binomial rank distributions using the smooth scores. Finally, NDCG is approximated by taking its expectation w.r.t. the rank distribution.

3

Preliminaries

In this section, we formalise the problem and introduce the notation used throughout the paper. Let (x, y) ∈ X n ×Zn≥0 be a training example consisting of a vector x of n items xi , 1 ≤ i ≤ n, and a vector y of corresponding non-negative integer relevance labels. Note that each item xi is itself a d-dimensional vector of numerical features, and should be thought of as representing a query-document pair.
The set X is the space of all such vectors xi . Thus, a pair (x, y) represents a list of vectorised search results for a given query together with the corresponding ground truth relevancies. The dataset of all such pairs is denoted Ψ . The goal of LTR is to find a scoring function f : X n → Rn that maximises the chosen IR metric on ΨP . The scoring function is learned by minimising the empirical risk L(f ) = |Ψ1 | (x,y)∈Ψ `(y, s) where `(·) is a loss function and s = f (x) is the vector of predicted scores. As discussed earlier, in most LTR approaches there is a mismatch between the loss function ` and the evaluation metric, causing a discrepancy between the learning procedure and its assessment. In this work, we focus on NDCG as our metric of choice and propose a new loss called NeuralNDCG, which bridges the gap between the training and the evaluation. Before we introduce NeuralNDCG, recall the definition of NDCG.
Definition 1. Let (x, y) ∈ X n × Zn≥0 be a training example and assume the documents in x have been ranked in the descending order of the scores computed

NeuralNDCG: Direct Optimisation of Ranking Metrics...

5

using some scoring function f . Let rj denote the relevance of the document ranked at j-th position, g(·) denote a gain function and d(·) denote a discount function. Then, the Discounted Cumulative Gain at k-th position (k ≤ n) is defined as k X DCG@k = g(rj )d(j)
(1)
j=1

and Normalised Discounted Cumulative Gain at k is defined as NDCG@k =

1 DCG@k maxDCG@k

(2)

where maxDCG@k is the maximum possible value of DCG@k, computed by ordering the documents in x by their decreasing ground truth relevancy.
Note that, typically, the discount function d(·) and the gain function g(·) are 1 given by d(j) = log (j+1)
and g(rj ) = 2rj − 1, respectively.
2

4

Loss formulation

In this section we define NeuralNDCG, a novel differentiable approximation to NDCG. It relies on NeuralSort, a smooth relaxation of the sorting operator. We begin by recalling NeuralSort, proceed to define NeuralNDCG and discuss its possible variants.
4.1

Sorting relaxation

Recall that sorting a list of scores s is equivalent to left-multiplying a column vector of scores by the permutation matrix Psort(s) induced by permutation sort(s) sorting the scores. Thus, in order to approximate the sorting operator, it is enough to approximate the induced permutation matrix. In [15], the permutation matrix is approximated via a unimodal row stochastic matrix Pbsort(s) (τ )
given by:
Pbsort(s) [i, :](τ ) = softmax[((n + 1 − 2i)s − As 1)/τ ]
(3)
where As is the matrix of absolute pairwise differences of elements of s such that As [i, j] = |si − sj |, 1 denotes the column vector of all ones and τ > 0 is a temperature parameter controlling the accuracy of approximation. For brevity, for the remainder of the paper we refer to Pbsort(s) (τ ) simply as Pb.
Note that the temperature parameter τ allows to control the trade-off between the accuracy of the approximation and the variance of the gradients.
Generally speaking, the lower the temperature, the better the approximation at the cost of a larger variance in the gradients. In fact, it is not difficult to demonstrate that:
lim Pbsort(s) (τ ) = Psort(s)
(4)
τ →0

(see [15] for proof). This fact will come in handy once we define NeuralNDCG.

6

P. Pobrotyn and R. Bialobrzeski.

An approximation of a permutation matrix by Equation 3 is a deterministic function of the predicted scores. Authors of NeuralSort proposed also a stochastic version, by deriving a reparametrised sampler for a Plackett-Luce family of distributions. Essentially, they propose to perturb scores s with a vector g of i.i.d.
Gumbel perturbations with zero mean and a fixed scale β to obtain perturbed scores s̃ = β log s + g. Perturbed scores are then used in place of deterministic scores in the formula for Pb.
We experimented with both deterministic and stochastic approximations to sorting and found them to yield similar results. Thus, for brevity, in this work we focus on the deterministic variant.

Table 1: Approximate sorting with NeuralSort. Given ground truth y = [4, 2, 1, 0, 4, 3] and predicted scores s = [0.5, 0.2, 0.1, 0.01, 0.65, 0.3], y is sorted by Pb for different values of τ . Exact sorting is shown in the first row.
Quasi-sorted ground truth Sum after sorting limτ →0 4 4 3 2 1 0 14 4 3 2 0.99992 0.00012339 14.00004339 τ = 0.01 4 τ = 0.1 3.9995 3.8909 2.8239 1.9730 0.9989 0.3136 13.9998 τ =1 3.3893 2.9820 2.4965 2.0191 1.6097 1.2815 10.388

4.2

NeuralNDCG

If the ground truth permutation is known, one could minimise the cross-entropy loss between the ground truth permutation matrix and its approximation given by Pb, as done in the experiments section in [15]. However, for many applications, including ranking, the exact ground truth permutation is not known. Relevance labels of individual items produce many possible valid ground truth permutation matrices. Thus, instead of optimising the cross-entropy loss, we use NeuralSort to introduce NeuralNDCG, a novel loss function appropriate for LTR.
Given a list of documents x, its corresponding vector of scores s = f (x) and the ground truth labels y we first find the approximate permutation matrix Pb induced by the scores s using Equation 3. We then apply the gain function g to the vector of ground truths y and obtain a vector g(y) of gains per document.
We then left-multiply the column vector g(y) of gains by Pb and obtain an “approximately” sorted version of the gains, d g(y). Another way to think of that approximate sorting is that the k-th row of Pb gives weights of documents xi in the computation of gain at rank k after sorting. Gain at rank k is then the weighted sum of ground truth gains, weighted by the entries in the k-th row of Pb. Note that after the approximate sorting the actual integer values of ground truth gains become ”distorted” and are not necessarily integers anymore (See d may Table 1 for example). In particular, the sum of quasi-sorted gains g(y)

NeuralNDCG: Direct Optimisation of Ranking Metrics...

7

differ from that of the original vector g(y). This leads to a peculiar behaviour of NeuralNDCG near the discontinuities of true NDCG (Figure 1), which may be potentially harmful for optimisation using Stochastic Gradient Descent [24].
Since Pb is row-stochastic but not-necessarily column-stochastic (i.e. each column does not necessarily sum to one), an individual ground truth gain g(y)j may have corresponding weights in the rows of Pb that do not sum to one (and, in particular, may sum to more than one), so it will overcontribute to the total sum d To alleviate that problem, we additionally perform Sinkhorn scaling [26]
of g(y).
on Pb (i.e. we iteratively normalize all rows and columns until convergence2 )
before using it for quasi-sorting. This way, the columns also sum to one and the approximate sorting is smoothed-out (again, see Figure 1). The remaining steps are identical to the computation of NDCG@k, with the exception that the gain of the relevance function rj is replaced with the j-th coordinate of quasi-sorted d For the discount function d, we use the usual inverse logarithmic gains g(y).
discount and for the gain function g we used the usual power function. For the computation of the maxDCG, we use original ground truth labels y. To find NeuralNDCG at rank k, we simply truncate quasi-sorted gains to the k-th position and compute the maxDCG at k-th rank.
We thus obtain the following formula for NeuralNDCG:
NeuralNDCGk (τ )(s, y) = Nk−1

k X (scale(Pb) · g(y))j · d(j)

(5)

j=1

where Nk−1 is the maxDCG at k-th rank, scale(·) is Sinkhorn scaling and g(·)
and d(·) are their gain and discount functions. Note that the summation is over the first k ranks.
Finally, since the popular autograd libraries provide means to minimise a given loss functions, we use (−1) × NeuralNDCG for optimisation.
4.3

NeuralNDCG Transposed

In the above formulation of NeuralNDCG, the summation is done over the ranks and gain at each rank is computed as a weighted sum of all gains, with weights given by the rows of Pb. We now provide an alternative formulation of NeuralNDCG, called NeuralNDCG Transposed (NeuralNDCGT for short), where the summation is done over the documents, not their ranks.
As previously, let x be a list of documents with corresponding scores s and ground truth relevancies y. We begin by finding the approximate permutation matrix Pb. Since we want to sum over the documents and not their ranks, we need to find the weighted average of discounts per document, not the weighted average of gains per rank as before. To this end, we transpose Pb to obtain an approximation PbT of the inverse of the permutation matrix corresponding to sorting the 2

We stop the procedure after 30 iterations or when the maximum difference between row or column sum and one is less than 10−6 , whatever happens first.

8

P. Pobrotyn and R. Bialobrzeski.

1.00 0.95

NDCG

0.90 0.85 0.80 0.75

True NDCG With Sinkhorn scaling Without Sinkhorn scaling

0.70 0.65

0

1

2

x

3

4

5

Fig. 1: Given ground truth y = [2, 1, 0, 0, 0] and a list of scores s = [4, 1, 0, 0, x], we vary the value of the score x and plot resulting NDCG induced by the scores along with NeuralNDCG (τ = 1.0) with and without Sinkhorn scaling of Pb.

documents x by their corresponding scores y. Thus, PbT can be thought of as an approximate unsorting matrix - when applied to sorted documents (ranks), it will (approximately) recover their original ordering. Since Pb is row-stochastic, PbT is column-stochastic. As we want to apply it by left-multiplication, we want it to be row-stochastic. Thus, similarly to before, we perform Sinkhorn scaling of PbT . After Sinkhorn scaling, the k-th row of PbT can be thought of as giving the weights of different ranks when computing the discount of the k-th document. We can now find the vector of the weighted averages of discounts per document by computing PbT d, where d is the vector of logarithmic discounts per rank (dj = d(j)). Note that since we want to perform summation over the documents, not ranks, it is not enough to sum the first k elements to truncate NDCG to the k-th position. Instead, the entries of the discounts vector d corresponding to ranks j > k are set to 0. This way, the documents which would end up at ranks j > k after sorting end up having weighted discounts being close to 0, and equal to 0 in the limit of the temperature τ . Thus, even though the summation is done over all documents, we still recover NDCG@k.
Hence, NeuralNDCGT is given by the following formula:

NeuralNDCGT k (τ )(s, y) = Nk−1

n X

g(yi ) · (scale(PbT ) · d)i

(6)

i=1

where Nk−1 is the maxDCG at k-th rank, scale(·) is Sinkhorn scaling, g(·) is the gain function, d is the vector of logarithmic discounts per rank set to 0 for ranks j > k, and the summation is done over all n documents.

NeuralNDCG: Direct Optimisation of Ranking Metrics...

4.4

9

Properties of NeuralNDCG

By Equation 4, in the limit of the temperature, the approximate permutation matrix Pb becomes the true permutation matrix P . Thus, as the temperature approaches zero, NeuralNDCG approaches true NDCG in both its variants. See Figure 2 for examples of the effect of the temperature on the accuracy of the approximation.
Comparing to ApproxNDCG, our proposed approximation to NDCG showcases more favourable properties. We can easily compute NDCG at any rank position k, whereas in ApproxNDCG, one needs to further approximate the truncation function using an approximation of the position function. This approximation of an approximation leads to a compounding of errors. We deal away with that problem by using a single approximation of the permutation matrix. Furthermore, the approximation of the position function in ApproxNDCG is done using a sigmoid function, which may lead to the vanishing gradient problem.
SoftRank suffers from a high computational complexity of O(n3 ): in order to compute all the derivatives required by the algorithm, a recursive computation is necessary. Authors relieve that cost by approximating all but a few of the RankBinomial distributions used, but at a cost of the accuracy of their solution. On the other hand, computation of Pb is of O(n2 ) complexity.

1.00

True NDCG =0.01 =0.1 =1.0 =10.0

0.95

NDCG

0.90 0.85 0.80 0.75 0.70

0

1

2

x

3

4

5

Fig. 2: Given ground truth y = [1, 2, 3, 4, 5] and a list of scores s = [1, 2, 3, 4, x], we vary the value of the score x and plot resulting NDCG induced by the scores along with Deterministic NeuralNDCG for different temperatures τ .

10

P. Pobrotyn and R. Bialobrzeski.

5

Experiments

This section describes the experimental setup used to empirically verify the proposed loss functions.
5.1

Datasets

Experiments were conducted on two datasets: Web30K and Istella 3 . Both datasets consists of queries and associated search results. Each query-document pair is represented by a real-valued feature vector and has an associated graded relevance on the scale from 0 (irrelevant) to 4 (highly relevant). For both datasets, we standardise the features, log-transforming selected ones, before feeding them to the learning algorithm. Since the lengths of search results lists in the datasets are unequal, we padded or subsampled to equal length for training, but used the full list length for evaluation. Web30K comes split into five folds. However, following the common practice in the field, we report results obtained on Fold 1 of the data. We used 60% of the data for training, 20% for validation and hyperparameter tuning and the remaining 20% for testing. Istella datasets comes partition into a training and a test fold according to a 80%-20% schema. We additionally split the training data into training and validation data to obtain a 60%/20%/20% split, similarly to Web30K. We tune the hyperparameters of our models on the validation data and report performance on the test set, having trained the best models on the full training fold. In both datasets there are a number of queries for which the associated search results list contains no relevant documents (i.e. all documents have label 0). We refer to these queries as empty queries. For such queries, the NDCG of their list of results can be arbitrarily set to either 0 or 1. To allow for a fair comparison with the current state of the art, we followed LightGBM [17] implementation of setting NDCG of such lists to 1 during the evaluation. Table 2 summaries the characteristics of the datasets used.

Table 2: Dataset statistics Dataset Features Queries in training Queries in test Empty queries Web30K 136 18919 6306 982 Istella 220 23219 9799 50

5.2

Scoring function

For the scoring function f , we used the Context-Aware Ranker, a ranking model based on the Transformer architecture. The model can be thought of as the 3

There are a few variants of this dataset, we used the Istella full dataset.

NeuralNDCG: Direct Optimisation of Ranking Metrics...

11

encoder part of the Transformer, taking raw features of items present in the same list as input and outputting a real-valued score for each item. Given the ubiquity of Transformer-based models in the literature, we refer to reader to [21] for the details of the architecture used. Compared to the original network described in [21], we used smaller architectures. For both datasets, we used an architecture consisting of 2 encoder blocks of a single attention head each, with a hidden dimension of 384. The dimensionality of initial fully-connected layer was set to 96 for models trained on Web30K and 128 for models trained on Istella. We did not apply any activation on the output except for NeuralNDCG and NeuralNDCGT . It exhibited suboptimal performance without any nonlinear output activation function and, in this case, we applied Tanh to the output. For both datasets, the same architectures were used across all loss functions.
5.3

Training hyperparameters

In all cases, we used Adam [18] optimiser and set the learning rate to 0.001.
The batch size was set to 64 (Web30K) or 110 (Istella) and search results lists were truncated or padded to the length of 240 when training. We trained the networks for 100 epochs, decaying the learning rate by the factor of 0.1 after 50 epochs.
5.4

Loss functions

We compared the performance of variants of NeuralNDCG against the following loss functions. For a pointwise baseline, we used a simple RMSE of predicted relevancy. Specifically, the output of the network f is passed through a sigmoid function and then multiplied by the number of relevance levels. The root mean squared difference of this score and the ground truth relevance is the loss. Pairwise losses we compared with consist of RankNet and LambdaRank. Similarly to NeuralNDCG, these losses support training with a specific rank cutoff. We thus train models with these losses at ranks 5, 10 and at the maximum rank.
The two most popular listwise losses are ListNet and ListMLE, and we, too, included them in our evaluation. Finally, the other method of direct optimisation of NDCG which we compared with was ApproxNDCG. We did not compare with SoftRank, as its O(n3 ) complexity proved prohibitive. We tuned ApproxNDCG and NeuralNDCG smoothness hyperparameters for optimal performance on the test set. Both ApproxNDCG’s α and NeuralNDCG’s τ parameter were set to 1 as other values in the [0.01; 100] interval did not show any improvement.
5.5

Results

For both datasets, we report models performance in terms of NDCG@5 and NDCG@10. Results are collected in Table 3. Both NeuralNDCG variants in every rank cutoff setting outperform ApproxNDCG on both datasets in all metrics reported. Moreover, NeuralNDCG variants with specific rank cutoffs provide

12

P. Pobrotyn and R. Bialobrzeski.

Table 3: Test NDCG on Web30K and Istella. Boldface is the best performing loss column-wise.
WEB30K Istella NDCG@5 NDCG@10 NDCG@5 NDCG@10 NeuralNDCG@5 50.32 52.01 65.32 69.97 NeuralNDCG@10 50.89 52.77 65.65 70.68 NeuralNDCG@max 51.56 53.46 65.69 70.55 NeuralNDCGT @5 50.50 52.14 65.46 69.95 NeuralNDCGT @10 50.85 52.70 66.02 71.02 53.49 65.60 70.53 NeuralNDCGT @max 51.45 ApproxNDCG 49.07 50.90 63.14 67.94 ListNet 50.75 52.80 65.62 70.70 ListMLE 49.81 51.82 59.85 66.24 RankNet@5 49.14 50.75 64.45 68.74 RankNet@10 50.95 52.69 65.75 70.68 RankNet@max 49.84 51.82 64.57 70.37 LambdaRank@5 48.70 50.10 63.50 67.75 49.66 51.34 65.21 69.82 LambdaRank@10 LambdaRank@max 51.55 53.47 65.90 71.09 RMSE 50.51 52.46 65.62 70.76 XGBoost 46.80 49.17 61.04 65.74 Loss

the best performance among all losses in both metrics on the WEB30K dataset and NDCG@5 on the Istella dataset. For reference, we also report the results of a GBDT model trained with XGBoost [10] and objective rank:pairwise (as rank:ndcg is known to yield suboptimal results4 ).

6

Conclusions

In this work we introduced NeuralNDCG, a novel differentiable approximation of NDCG. By substituting the discontinuous sorting operator with NeuralSort, we obtain a robust, efficient and arbitrarily accurate approximation to NDCG.
Not only does it enjoy favourable theoretical properties, but also proves to be effective in empirical evaluation, yielding competitive performance, on par with LambdaRank. This work can easily be extended to other rank-based metrics like MAP; a possibility we aim to explore in the future. Another interesting extension of this work would be the substitution of NeuralSort with another method of approximation of the sorting operator, most notably the method treating sorting as an Optimal Transport problem [11].
4

For details, please visit https://github.com/dmlc/xgboost/issues/6352.

NeuralNDCG: Direct Optimisation of Ranking Metrics...

13

References 1. Baeza-Yates, R.A., Ribeiro-Neto, B.: Modern Information Retrieval. AddisonWesley Longman Publishing Co., Inc., Boston, MA, USA (1999)
2. Bruch, S.: An alternative cross entropy loss for learning-to-rank (2019)
3. Bruch, S., Wang, X., Bendersky, M., Najork, M.: An analysis of the softmax cross entropy loss for learning-to-rank with binary relevance. In: Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval. p. 75–78. ICTIR ’19, Association for Computing Machinery, New York, NY, USA (2019). https://doi.org/10.1145/3341981.3344221, https://doi.org/ 10.1145/3341981.3344221 4. Bruch, S., Zoghi, M., Bendersky, M., Najork, M.: Revisiting approximate metric optimization in the age of deep neural networks. In: Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 1241–1244. SIGIR’19, Association for Computing Machinery, New York, NY, USA (2019). https://doi.org/10.1145/3331184.3331347, https://doi.org/10.1145/3331184.3331347 5. Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., Hullender, G.: Learning to rank using gradient descent. In: Proceedings of the 22Nd International Conference on Machine Learning. pp. 89–96. ICML ’05, ACM, New York, NY, USA (2005). https://doi.org/10.1145/1102351.1102363, http://doi.
acm.org/10.1145/1102351.1102363 6. Burges, C.J., Ragno, R., Le, Q.V.: Learning to rank with nonsmooth cost functions.
In: Schölkopf, B., Platt, J.C., Hoffman, T. (eds.) Advances in Neural Information Processing Systems 19, pp. 193–200. MIT Press (2007), http://papers.nips.cc/ paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf 7. Burges, C.J.C.: From RankNet to LambdaRank to LambdaMART: An overview.
Tech. rep., Microsoft Research (2010), http://research.microsoft.com/en-us/ um/people/cburges/tech_reports/MSR-TR-2010-82.pdf 8. Cao, Z., Qin, T., Liu, T.Y., Tsai, M.F., Li, H.: Learning to rank: From pairwise approach to listwise approach. In: Proceedings of the 24th International Conference on Machine Learning. pp. 129–136. ICML ’07, ACM, New York, NY, USA (2007). https://doi.org/10.1145/1273496.1273513, http://doi.acm.org/10.1145/ 1273496.1273513 9. Chakrabarti, S., Khanna, R., Sawant, U., Bhattacharyya, C.: Structured learning for non-smooth ranking losses. In: Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. p. 88–96. KDD ’08, Association for Computing Machinery, New York, NY, USA (2008). https://doi.org/10.1145/1401890.1401906, https://doi.org/ 10.1145/1401890.1401906 10. Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 785–794. KDD ’16, ACM, New York, NY, USA (2016). https://doi.org/10.1145/2939672.2939785, http://doi.acm.org/10.1145/ 2939672.2939785 11. Cuturi, M., Teboul, O., Vert, J.P.: Differentiable ranking and sorting using optimal transport. In: Wallach, H., Larochelle, H., Beygelzimer, A., d Alche-Buc, F., Fox, E., Garnett, R. (eds.) Advances in Neural Information Processing Systems 32, pp. 6858–6868. Curran Associates, Inc. (2019), http://papers.nips.cc/paper/ 8910-differentiable-ranking-and-sorting-using-optimal-transport.pdf

14

P. Pobrotyn and R. Bialobrzeski.

12. Dato, D., Lucchese, C., Nardini, F.M., Orlando, S., Perego, R., Tonellotto, N., Venturini, R.: Fast ranking with additive ensembles of oblivious and non-oblivious regression trees. ACM Trans. Inf. Syst. 35(2) (Dec 2016).
https://doi.org/10.1145/2987380, https://doi.org/10.1145/2987380 13. Engilberge, M., Chevallier, L., Perez, P., Cord, M.: Sodeep: A sorting deep net to learn ranking loss surrogates. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)
14. Friedman, J.H.: Greedy function approximation: A gradient boosting machine.
Annals of Statistics 29, 1189–1232 (2000)
15. Grover, A., Wang, E., Zweig, A., Ermon, S.: Stochastic optimization of sorting networks via continuous relaxations. In: International Conference on Learning Representations (2019), https://openreview.net/forum?id=H1eSS3CcKX 16. Järvelin, K., Kekäläinen, J.: Cumulated gain-based evaluation of ir techniques. ACM Trans. Inf. Syst. 20(4), 422–446 (Oct 2002).
https://doi.org/10.1145/582415.582418, http://doi.acm.org/10.1145/582415.
582418 17. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y.: Lightgbm: A highly efficient gradient boosting decision tree. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems 30, pp. 3146–3154. Curran Associates, Inc. (2017), http://papers.nips.cc/paper/ 6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf 18. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization (2014), http:
//arxiv.org/abs/1412.6980, cite arxiv:1412.6980Comment: Published as a conference paper at the 3rd International Conference for Learning Representations, San Diego, 2015 19. Liu, T.Y.: Learning to rank for information retrieval. Found. Trends Inf. Retr.
3(3), 225–331 (Mar 2009). https://doi.org/10.1561/1500000016, http://dx.doi.
org/10.1561/1500000016 20. Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., Lerer, A.: Automatic differentiation in PyTorch. In:
NIPS Autodiff Workshop (2017)
21. Pobrotyn, P., Bartczak, T., Synowiec, Mikolaj an Bialobrzeski, R., Bojar, J.:
Context-aware learning to rank with self-attention. In: SIGIR eCom ’20. Virtual Event, China. (2020)
22. Qin, T., Liu, T.M.: Introducing letor 4.0 datasets. ArXiv abs/1306.2597 (2013)
23. Qin, T., Liu, T.Y., Li, H.: A general approximation framework for direct optimization of information retrieval measures. Inf. Retr. 13, 375–397 (08 2010).
https://doi.org/10.1007/s10791-009-9124-x 24. Robbins, H., Monro, S.: A stochastic approximation method. Annals of Mathematical Statistics 22, 400–407 (1951)
25. Rumelhart, D.E., Hinton, G.E., Williams, R.J.: Learning internal representations by error propagation. Tech. rep., California Univ San Diego La Jolla Inst for Cognitive Science (1985)
26. Sinkhorn, R.: A relationship between arbitrary positive matrices and doubly stochastic matrices. Ann. Math. Statist. 35(2), 876–879 (06 1964).
https://doi.org/10.1214/aoms/1177703591, https://doi.org/10.1214/aoms/ 1177703591 27. Taylor, M., Guiver, J., Robertson, S., Minka, T.: Softrank: optimizing non-smooth rank metrics. pp. 77–86 (01 2008). https://doi.org/10.1145/1341531.1341544

NeuralNDCG: Direct Optimisation of Ranking Metrics...

15

28. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems 30, pp. 5998–6008. Curran Associates, Inc.
(2017), http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf 29. Xia, F., Liu, T.Y., Wang, J., Zhang, W., Li, H.: Listwise approach to learning to rank: Theory and algorithm. In: Proceedings of the 25th International Conference on Machine Learning. pp. 1192–1199. ICML ’08, ACM, New York, NY, USA (2008). https://doi.org/10.1145/1390156.1390306, http://doi.acm.org/10.
1145/1390156.1390306 30. Xu, J., Li, H.: Adarank: a boosting algorithm for information retrieval. In: Proceedings of the 30th annual ACM SIGIR conference. pp. 391–398. ACM, Amsterdam, The Netherlands (2007). https://doi.org/10.1145/1277741.1277809 31. Yue, Y., Finley, T., Radlinski, F., Joachims, T.: A support vector method for optimizing average precision. In: Proceedings of the 30th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 271–278. SIGIR ’07, Association for Computing Machinery, New York, NY, USA (2007). https://doi.org/10.1145/1277741.1277790, https://doi.
org/10.1145/1277741.1277790


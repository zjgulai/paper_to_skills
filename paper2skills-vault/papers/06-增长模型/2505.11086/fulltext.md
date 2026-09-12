<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2505.11086
     paper_id : 2505.11086
     source   : paper2skills-vault/papers/growth_model/2505.11086/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations for Sequential Data arXiv:2505.11086v1 [cs.AI] 16 May 2025

Keita Kinjo∗ Faculty of Business, Kyoritsu Women’s University 2-2-1, Hitotsubashi, Chiyoda-ku, Tokyo, 101-8437, Japan

Abstract Recently, the proliferation of omni-channel platforms has attracted interest in customer journeys, particularly regarding their role in developing marketing strategies.
However, few efforts have been taken to quantitatively study or comprehensively analyze them owing to the sequential nature of their data and the complexity involved in analysis. In this study, we propose a novel approach comprising three steps for analyzing customer journeys. First, the distance between sequential data is defined and used to identify and visualize representative sequences. Second, the likelihood of purchase is predicted based on this distance. Third, if a sequence suggests no purchase, counterfactual sequences are recommended to increase the probability of a purchase using a proposed method, which extracts counterfactual explanations for sequential data. A survey was conducted, and the data were analyzed; the results revealed that typical sequences could be extracted, and the parts of those sequences important for purchase could be detected. We believe that the proposed approach can support improvements in various marketing activities.

Keywords: counterfactual explanation, customer journeys, sequential data, clustering, Levenshtein distance, visualization, marketing research

1

Introduction

Consumer environments have become increasingly complex in recent years, with various options being offered for gaining increased product awareness, gathering information, making purchases, and sharing product reviews. Awareness can be raised via not only mass media but also web advertisements and social media. Physical stores, social media, and comparison sites are used for information gathering. Consumers perform purchases through multiple ∗

Email: kkinjo@kyoritsu-wu.ac.jp; ORCID: 0000-0001-5907-3501

1

channels and omni-channels, including online e-commerce sites and physical stores [3, 22].
Subsequently, they commonly share product reviews on social media and online sites.
A customer journey is a method for analyzing consumer perceptions and behavior [18].
This is a typology of a particular sequence of consumer cognition and behavior that companies use in marketing strategies [11, 30, 23]. By analyzing customer journeys, companies can better understand consumers and design consumer experiences more efficiently [17]. Specific patterns in customer journeys have been frequently extracted through qualitative research.
However, the number of studies that quantitatively analyze or formalize customer journeys remains limited.
One of the few to quantitatively analyze customer journeys, Rosenbaum et al. (2017)
investigated the frequency of consumer cognition and behavior at each stage of consumption (pre-purchase, purchase, post-purchase) [25]. There are several studies similar to these [30]. However, the length of the customer journey of each individual is different, and the corresponding data should be sequential in several stages. However, since sequential data are difficult to handle in tabular data analysis, there has been limited related research. For example, Herhausen et al. simply counted customer journey patterns [13].
One of the few to formalize customer journeys, Bernard and Andritsos (2017) did so (defining the data structure using trees) in a process mining framework [4]. Other studies have employed Markov models, mixed Markov models, edit distances between sequences, or genetic algorithms to detect patterns [1, 12, 5, 6, 33].
Several studies have employed sequence analysis methods based on inter-sequence distance measures. For example, Topaloglu, Oztaysi, and Dogan (2024) analyzed more than 20 million rows of clickstream data collected from a real e-commerce site and proposed a four-stage framework for extracting and interpreting visitor journeys. One of these stages involved clustering based on Levenshtein distance. Their findings revealed, among other insights, that journeys characterized by exploratory behavior exhibited significantly lower conversion rates than other behavioral patterns [29].
In another study, Bobroske et al. (2020) converted sequences of actions from 10,000 patients into strings and customized the Levenshtein edit-distance algorithm to evaluate similarities between those sequences [7]. Using this distance-based approach for clustering, they identified twelve representative patient-journey patterns.
Disadvantageously, existing research cannot suitably address the following issues. First, it is difficult to extract typical cases from sequential data that have a structured nature—such as the stages of consumption in a customer journey—which differs from general, unstructured sequences. Furthermore, in such cases, it is not only important to extract typical patterns, but also to compare them with other sequences. To enable such comparisons, the position of each pattern within the overall sequence space should be visualized and interpreted before being applied. Marketers need to identify and visualize both typical and representative customer journeys in order to gain insights into customer needs, understand their perceptions, and develop effective strategies [8].
In the context of marketing applications, it is necessary not only to extract typical patterns using sequence-based distance measures but also to predict outcomes such as whether a purchase will occur. For companies, whether a purchase ultimately takes place is a critical factor [29]. This is because such predictions can help identify which types of sequences are more likely to lead to purchases and can also be used to estimate outcomes if the sequences 2

are modified. To achieve this, the application of machine learning methods based on sequence distance is essential. However, this topic has received limited attention in existing research.
In addition, it is important to propose how a non-purchasing sequence could be improved to lead to a purchase. Such improved sequences can help uncover the factors that drive purchasing decisions. In recent years, as many machine learning models—including those using distance-based approaches—have been criticized for being difficult to interpret or functioning as “black boxes,” the field of explainable artificial intelligence (XAI) or interpretable machine learning has gained significant attention [21, 10]. Numerous techniques have been developed, including LIME and SHAP [24, 20]. Among these, counterfactual explanation (CE) has emerged as a promising approach [15, 31]. CE identifies the minimal changes required in a given instance to alter the prediction of a machine learning model. By analyzing the parts of an instance that must be changed to yield a different outcome, CE enables instance-level interpretation of relevant factors. Today, CE methods are widely used in fields such as image analysis and time series prediction. In the context of time series data, some studies have applied distance-based methods such as Dynamic Time Warping (DTW) to generate counterfactuals that are not only interpretable but also closer to typical patterns, thereby ensuring realism [14].
However, these methods are designed for numerical time series data. Although a few attempts have been made to apply CE to symbolic sequence data like the kind addressed in this study, they typically rely on mixed-integer linear programming (MILP), which limits their applicability to specific models [28]. To date, no method has been developed that both supports a wide range of models and generates plausible counterfactual sequences.
Therefore, this study proposes an approach with the following three steps. (1) First, the distances between sequential data are calculated using weighted Levenshtein distance, and prototypes (typical patterns) are extracted by clustering using k-medoids and then visualized. (2) Next, using these distances, predictions are made using nonparametric regression techniques such as k-nearest neighbor (k-NN). (3) Finally, based on these predictions, measures are proposed to improve specific sequences using counterfactual explanation. The effectiveness of both steps was evaluated using actual survey data; based on this, a new survey method was proposed for obtaining sequential data.
The rest of the manuscript is organized as follows. Section 2 describes the proposed method. Then, Section 3 describes the experimental details and the results, and Section 4 discusses the results with respect to the aim of the study and presents the limitations.
Finally, Section 5 provides the concluding remarks.

2

Method

Section 2.1 introduces the problem to be solved in the study, and Section 2.2 describes using the proposed method to solve it.

3

2.1

Definition and Problem Setting

We propose the following definition with reference to customer journey studies such as those of Bernard and Andritsos (2019) and the serial data mining work of Zhang et al. (2001)
[5, 6, 32].
First, let touchpoints be defined as T P = {tp 1 , tp 2 , . . . , tp e }, where ”touchpoints” refers to the means and places (e.g., product reviews, social media, e-commerce sites, in-store)
where consumers interact with a product.
Furthermore, let A = {a1 , . . . , ah } denote h actions (including no action), such as performing searches, making purchase or non-purchase decisions, and writing product reviews.
Let ST = hst 1 , . . . , st m i be a set of m ordered stages, where ”stage” refers to steps in the consumption process, namely the pre-purchase, purchase, and post-purchase stages.
Then, the qth data point of individual i is represented as Xqi ∈ (tp, a; st), where tp ∈ T P, a ∈ A, and st ∈ ST .
The sequential data belonging to individual i is denoted by the ordered dataset si = i hX1 , . . . , Xni i i, where ni is the total number of observations for individual i.
The subsequence in which each pair (tp, a) belongs to the gth stage st g is denoted as si (st g ).
Based on the above definitions, let S = {s1 , s2 , . . . , sr } be a set of sequences for r individuals. Let S(st g ) be the set of subsequences si (st g ) belonging to stage st g .
Finally, let y = {y1 , y2, . . . , yr } represent the variables to be predicted, where each yi can be derived from (tp, a; st), conveying information such as whether a purchase is made at the purchase stage.
Problem: Given the sequence set S, this study aims to (1) extract k prototypes based on the distances between sequences, and (2) predict the outcome variable y using these distances.

2.2

Proposed Method

2.2.1

Definition of the Distances Between Sequences

This section defines the distance between the sequences of S. The Levenshtein distance (LD), a type of edit distance, is used. The data Xqi are not usually interchanged across stages [9, 19]. The importance of each stage should be considered; for example, the presence or absence of purchases can result in large differences. However, conventional LD does not account for such importance, and thus fails to capture these differences appropriately.
To include such a condition, the distance between the two sequences si and sj is defined as the weighted sum of the distances at each stage, as follows:
Dl (si , sj ) =

m X

wg dl (si (stg ), sj (stg ))

(1)

g=1

where wg ≥ 0 is the weight, and dl is the Levenshtein distance. These distances are computed by considering each pair (tp, a) in si (stg ) as a single symbol.

4

The Levenshtein distance represents the number of operations (insertions, deletions, substitutions) required to transform one string into another. It satisfies properties such as the triangle inequality [2], and the above formulation inherits this property.
P Proposition. If Dl (si , sj ) = m g=1 wg dl (si (stg ), sj (stg )), wg ≥ 0, and each dl (si (stg ), sj (stg ))
satisfies the triangle inequality, then Dl (si , sj ) also satisfies the triangle inequality.
Proof. To simplify notation, let Dl (X, Y ) = components of sequences X, Y . Let X = (x1 , x2 ),

Pm

g=1 wg dl (xg , yg ), where xg , yg are the gth

Y = (y1, y2 ),

Z = (z1 , z2 )

Define:
Dl (X, Y ) = w1 dl (x1 , y1 ) + w2 dl (x2 , y2 )
Dl (X, Z) = w1 dl (x1 , z1 ) + w2 dl (x2 , z2 )
Dl (Y, Z) = w1 dl (y1 , z1 ) + w2 dl (y2, z2 )

(2)
(3)
(4)

Assuming that dl satisfies the triangle inequality, we have:
dl (x1 , z1 ) ≤ dl (x1 , y1) + dl (y1 , z1 )
dl (x2 , z2 ) ≤ dl (x2 , y2) + dl (y2 , z2 )

(5)
(6)

Multiplying (6) by w2 ≥ 0:
w2 dl (x2 , z2 ) ≤ w2 dl (x2 , y2 ) + w2 dl (y2 , z2 )

(7)

Adding w1 dl (x1 , z1 ) to both sides of (7):
w1 dl (x1 , z1 ) + w2 dl (x2 , z2 ) ≤ w1 dl (x1 , z1 ) + w2 dl (x2 , y2 ) + w2 dl (y2 , z2 )

(8)

From (5), we have:
w1 dl (x1 , z1 ) ≤ w1 dl (x1 , y1 ) + w1 dl (y1 , z1 )

(9)

Substituting (9) into (8), we get:
w1 dl (x1 , z1 ) + w2 dl (x2 , y2) + w2 dl (y2 , z2 ) ≤ w1 dl (x1 , y1 ) + w1 dl (y1 , z1 )
+ w2 dl (x2 , y2 ) + w2 dl (y2 , z2 )
Hence, Dl (X, Z) ≤ Dl (X, Y ) + Dl (Y, Z)
This conclusion also holds for the general case, where Dl (X, Y ) =

m X

wg d(xg , yg )

⇐⇒

Dl (si , sj ) =

g=1

m X g=1

 wg dl si (stg ), sj (stg ) .

Therefore, Dl (si , sj ) satisfies the triangle inequality and thus defines a valid distance metric over the space of staged sequences.
 5

The costs of replacement, insertion, and deletion operations can be modified. However, using LD with arbitrary weights or normalization may violate the distance axioms. Therefore, we adopt the above weighted LD formulation, which respects distance properties. Notably, the total computation cost is reduced by calculating distances separately for each stage stg .
2.2.2

Prototype Detection and Visualization

This section describes how the proposed weighted Levenshtein distance is used to extract prototypes. K-medoids, a clustering method, is employed for this purpose [16]. K-medoids identifies data points, called medoids, that minimize the total dissimilarity to all other data in the same cluster. These medoids serve as the prototypes. Since only pairwise distances between sequences are available, this method enables prototype detection without requiring data in vector format, which is commonly assumed in conventional clustering methods.
Other methods such as K-means are not suitable in this context, as they require the computation of mean vectors, which is not possible when only pairwise sequence distances are provided.
Visualization. Let D be a matrix whose entries represent the pairwise distances Dl . Each sequence is visualized as a point in a two-dimensional space based on the concept of multidimensional scaling (MDS) [27]. Although other dimensionality reduction techniques such as principal component analysis (PCA) exist, MDS is appropriate here because it requires only a distance matrix and not feature vectors.
Let I be an identity matrix and 1r an r-dimensional column vector of ones. The procedure for MDS-based visualization is as follows:
1. Represent the centering matrix:
1 Dce = − CD 2 C, 2

1 where C = I − 1r 1′r r

(10)

2. Perform eigenvalue decomposition of Dce :
Dce = V ΛV −1  λ1 0  0 λ2 where Λ = 

0 of eigenvectors.

0

(11)



0 0  contains the eigenvalues and V = [~v1 , ~v2 , . . .] is the matrix ..
.

3. Compute the coordinate matrix:
X ∗ = V Λ1/2 assuming only the top two eigenvalues λ1 and λ2 are nonzero.
4. Place each data point using X ∗ .
The resulting two-dimensional coordinates represent each sequence, and the Euclidean distances between the points approximate the original sequence distances. This visualization supports interpretation of the cluster structure and the distribution of prototype sequences.
6

2.2.3

Prediction and Counterfactual Explanation

This section explains how prediction was performed using the proposed Levenshtein distance.
In this study, the k-nearest neighbors (k-NN) method was employed. This is a nonparametric regression approach that leverages distance, and its use here is consistent with earlier sections: since only pairwise distances between sequences and corresponding target values y are available, conventional regression or machine learning models that require feature vectors cannot be applied.
To avoid confusion with k in k-medoids clustering, we use k ′ to denote the number of neighbors in k-NN.
The prediction for a given sequence si is computed as:
ŷ(si ) =

1 X yj k ′ s ∈N j

(12)

i

where Ni is the set of k ′ nearest sequences to si under the proposed distance measure.
Counterfactual Explanation. Additionally, a new type of counterfactual explanation is proposed as an interpretability method in machine learning [15, 31]. For example, this method suggests an alternative customer journey that changes a non-purchase prediction into a purchase outcome, thereby highlighting factors that influence consumer decisions.
Specifically, the counterfactual sequence cfsi for a given si is defined as:
cfsi = arg min {loss(yobj , yj ) + λ · D ′ (si , sj )} sj ∈S\{si }

(13)

where:
• S \ {si } is the set of all sequences excluding si , • yobj is the desired prediction outcome (e.g., purchase), • λ is a hyperparameter that balances prediction loss and similarity, • D ′ is a potentially modified distance between sequences (possibly restricted to some stages stg ).
A novel aspect of this method is that it does not generate new (synthetic) sequences;
rather, it selects the most plausible counterfactual sequence from the existing data. This contributes to producing more realistic and actionable explanations, which has not been widely addressed in prior work.
The core idea is to detect an existing sequence that achieves the desired target yobj and is also close to si under the defined distance metric. In practice, the process computes the objective value for all candidate sequences and selects the one with the smallest score.
The change from the original si to the selected counterfactual is analyzed to identify which components (e.g., touchpoints or actions) differ. Based on this insight, specific suggestions can be made to improve customer journeys.

7

3

Experimental

Section 3.1 describes the data used in the analysis, and Section 3.2 discusses the results.

3.1

Data

Although automatically collected log data from e-commerce sites can be used to analyze customer journeys, acquiring such data for general product purchases presents practical challenges. Therefore, we developed and employed a new questionnaire-based method that allowed the number of recorded actions to vary by individual.
The survey was conducted online and focused on the most recently purchased type of cosmetics—eye shadows. It was administered to 127 female university students in Japan over a two-day period from November 30 to December 1, 2022. Participants who indicated an interest in eye shadow were first asked: “How did you come to know about eye shadow?” Subsequent questions such as “Which of the following did you do next?” were then presented, with responses recorded up to 10 times per participant.
The questionnaire consisted of 13 predefined combinations of touchpoints and actions:
a. Knowing the product through TV commercials [awareness]
b. Knowing the product through direct word-of-mouth communication [awareness]
c. Understanding the product through word-of-mouth communication on social media [awareness]
d. Knowing the product through other methods [awareness]
e. Comparison and confirmation through various websites by searching [information gathering]
f. Comparison and confirmation through e-commerce sites [information gathering]
g. Comparison and confirmation through stores [information gathering]
h. Comparison and confirmation through social media [information gathering]
i. Performing purchases on e-commerce sites [purchase]
j. Performing purchases in stores [purchase]
k. Not performing purchases [non-purchase]
l. Writing product reviews on social media [post-purchase]
m. Other

8

Hereafter, each symbol corresponds to the meanings described below and is also consistent with the formal representation introduced in Section 2.1. The symbols a to h represent the subset of combinations of (tp, a), where tp ∈ {TV, person, social media, e-commerce site, storefront} and a ∈ {cognition, information gathering} (corresponding to the options described above).
Here, i and j correspond to (tp = ”e-commerce site”, a = ”purchase”) and (tp = ”storefront”, a = ”purchase”), respectively. Finally, purchases in k are represented by “1” and non-purchases by “0”.
The sequences were classified into three stages: st1 (pre-purchase: first recognition), st2 (pre-purchase: awareness and information gathering), and st3 (purchase stage: purchase and non-purchase). Post-purchase actions (such as items l and m) were excluded from the analysis.
Sequences that did not include either a purchase or non-purchase action were omitted, as were sequences with logically inconsistent stage transitions (e.g., taking a pre-purchase action after a purchase action). Only valid paths were retained, such as: st1 → st1 (multiple awareness), st1 → st2 (consideration after awareness), st2 → st2 (multiple considerations), st2 → st1 (awareness after consideration), st1 → st3 (purchase after awareness), and st2 → st3 (purchase after consideration).
As a result, 104 valid samples were obtained. Of these, 86 resulted in a purchase and 18 in a non-purchase.
To facilitate the recall of recent consumption behavior, respondents were first asked to describe specific brand names and shopping contexts. However, this approach may introduce recall bias—a limitation that is acknowledged in the design of survey-based research.

3.2

Results

Section 3.2.1 presents descriptive statistics, Section 3.2.2 discusses the results of prototype detection and visualization, and Section 3.2.3 reports the results of prediction and counterfactual explanation.
3.2.1

Descriptive Statistics

First, the descriptive statistics for each stage of the obtained data are presented.
For the first stage, st1 , that is, the pre-purchase (first recognition) phase, the results are shown in Table 1.
Table 1: Descriptive statistics of items in Stage 1

The results show that item c, “Understanding the product through word-of-mouth communication on social media,” appears most frequently, followed by item d, “Knowing the 9

product through other methods.” This indicates that recognition through social media is the most common. Item d is considered to include recognition through offline sources such as in-store exposure.
For the next stage, st2 , the pre-purchase phase (awareness and information gathering), the results are shown in Table 2 and Table 3.
Table 2: Descriptive statistics of items in Stage 2

Table 3: Per-sample item count statistics

The results show that item g, “Comparison and confirmation through stores,” was the most frequent, followed by item e, “Comparison and confirmation through various websites by searching,” and item h, “Comparison and confirmation through social media.” In this stage, the number of actions ranged from a minimum of 0 to a maximum of 6, with an average of 1.32 items. In particular, it was revealed that individuals typically performed 1 to 3 actions during this phase. On the other hand, a value of 0 indicates that the purchase was made immediately after recognition, suggesting behavior similar to impulse buying. Taken together, these results suggest that after recognizing a product, consumers generally tend to compare or confirm it in physical stores.
For the final stage, st3 , the number of purchases and non-purchases was 86 and 18, respectively. This indicates that most respondents chose to make a purchase.
10

Finally, all stages were integrated, and a matrix of item pairs was created with consideration of the sequence order. The result was visualized, as shown in Figure 1. For example, for a sequence such as a, e, i, the transitions a → e and e → i were each counted. Only item pairs that satisfy the stage transition constraints described in Section 3.1 (e.g., from st1 to st2 , or from st2 to st3 ) were used.

Figure 1: Sequential Co-occurrence Matrix (Y-axis: From Item, X-axis: To Item)
The results show that in transitions from Stage st1 (a, b, c, d) to the next stage, the most frequent item pairs were c → e, c → g, c → i, and d → g. This suggests that after recognizing a product through social media, consumers often proceed to consider it through websites or in physical stores. In transitions from Stage st2 (e, f, g, h) to the next stage, g → i was by far the most frequent, followed by e → g and h → i. This indicates that many consumers make a purchase after considering the product in stores.
However, it should be noted that these results are based solely on pairwise frequency.
To investigate customer journeys involving more than two consecutive actions, full sequence analysis is necessary.
3.2.2

Results of Prototype Detection and Visualization

The analysis of the results is presented here. First, the clustering results are described. Four different distance measures were considered: the standard Levenshtein distance (LD), the Damerau-Levenshtein distance, and two weighted versions with weights w1 = 2, w2 = 1, w3 = 1 and w1 = 2, w2 = 1, w3 = 10, respectively. Clustering was evaluated using the Silhouette Coefficient (SC) for cluster sizes k ∈ {2, 3, 4, 5, 6, 7, 8} [26]. Initial values were set using the 11

k-medoids++ algorithm, and random seeds were fixed to ensure reproducibility. Higher SC values indicate better clustering performance. In general, SC can be high for smaller k, but higher values may be preferred depending on the application.
Table 4: Clustering results (Silhouette Coefficient for different settings)

Table 4 shows that SC values increased when higher weights were assigned to w3 . This reflects the greater influence of differences in the purchase stage st3 under such settings. The typical sequences extracted under the setting w1 = 2, w2 = 1, w3 = 10 with k = 6 are shown below:
1: [c, e, g, 1]
2: [c, e, g, 0]
3: [c, g, 1]
4: [d, c, 1]
5: [c, b, e, g, 1]
6: [d, 1]
The cluster sizes were as follows: Cluster 1: 14, Cluster 2: 18, Cluster 3: 35, Cluster 4:
9, Cluster 5: 7, and Cluster 6: 20. The most common pattern was Pattern 3, which represented customers who recognized a product through social media, checked it in stores, and then purchased it. The next most common was Pattern 6, which represented customers who purchased a product after recognizing it through other means—possibly even prior familiarity. Pattern 2 and Pattern 5 represent customers who performed online searches, confirmed products in stores, and either made a purchase or did not, depending on the situation.
Next, the visualization results are presented in Figure 2. Although the positions are relative, patterns like 1 and 5 appeared in the lower right, suggesting more careful judgment.
Conversely, patterns like 3 and 6 occupied more central or upper positions, indicating more direct decision-making. The non-purchase group also tended to cluster in areas suggesting more deliberative behavior.
3.2.3

Results of Prediction and Counterfactual Explanation

Then, the k-NN algorithm was used to predict y. Specifically, in this study, S(st1 ) and S(st2 )
were used as explanatory variables, and the purchase (“1”) or non-purchase (“0”) decision in S(st3 ) was used as the target variable y. Regarding the data, 80% and 20% were used as training and test data, respectively. The prediction was performed using k ′ ∈ {1, 2, 3, 4, 5}, 12

Figure 2: Visualization of the sequence clusters in 2D space

Table 5: Comparison of accuracy and F1-measure values

13

and each setting was repeated 100 times to evaluate the mean and variance of both accuracy and F1-measure. These results are shown in Table 5.
Comparing the values of each model, the accuracy for k ′ = 5 was approximately 0.81, indicating that it would be possible to predict whether a product will be purchased based on the sequence.
The results of the counterfactual explanation for the k-NN model using sequential data are presented below. Specifically, four cases are shown in which the original sequence had a label of y = 0 (non-purchase), and an alternative sequence improved the prediction to ŷ = 1 (purchase).
(Base sequence and counterfactual sequence: 1)
Base sequence: ["c", "c", "e", "g"] : y = 0 Counterfactual sequence: ["c", "b", "e", "g"]: ŷ = 1 The improved sequence was derived by replacing c: “Knowing about the product through word-of-mouth communication on social media” with b: “Knowing about the product through direct word-of-mouth communication,” indicating that the respondent might purchase the product. This suggested the importance of direct word-of-mouth communication over communication on social media with respect to purchasing decisions.
(Base sequence and counterfactual sequence: 2)
Base sequence: ["d", "e", "f"] : y = 0 Counterfactual sequence: ["c", "e", "f"]: ŷ = 1 The improved sequence was derived by replacing d: “Knowing the product through other methods” with c: “Knowing the product through word-of-mouth communication on social media,” indicating that the respondent might purchase the product. This suggested the importance of social media communication over more ambiguous or less direct information sources.
(Base sequence and counterfactual sequence: 3)
Base sequence: ["d", "g", "h"] : y = 0 Counterfactual sequence: ["d", "g", "e"]: ŷ = 1 The improved sequence was derived by replacing h: “Comparison and confirmation through social media” with e: “Comparison and confirmation through various websites by searching,” indicating that the respondent might purchase the product. This suggested that broader web-based comparison may be more effective than relying solely on social media content.
(Base sequence and counterfactual sequence: 4)
Base sequence: ["b"]: y = 0 Counterfactual sequence: ["c"]: ŷ = 1 The improved sequence was derived by replacing b: “Knowing the product through direct word-of-mouth communication” with c: “Understanding the product through word-of-mouth communication on social media,” indicating that the respondent might purchase the product.
This suggested that communication on social media might have a broader or more persuasive influence compared to direct word-of-mouth interactions.
This analysis demonstrated that even minor changes in the sequence of information acquisition and evaluation can influence purchase intentions. The following key findings emerged:
14

1. The mode of word-of-mouth communication (direct vs. social media-based) significantly impacts purchasing decisions, with effectiveness varying by context.
2. Social media-based information tends to be more persuasive than impersonal or unspecified sources.
3. During the comparison and confirmation stage, web searches offering diverse information may be more effective than social media-based content.
These findings suggest that strategically designing information channels is essential for encouraging consumer purchases. Moreover, these results clearly showed that it was possible to propose ways to change a particular preselected sequence such that it would lead to a purchase. Furthermore, they indicated the factors that were important for performing purchases in each sequence.

4

Discussion

This study was conducted to formally and quantitatively analyze customer journeys. The differences between previous studies and the proposed study are discussed below.
Unlike studies that use mixed Markov models, our study specifically detects typical cases [12]. Then, there are several studies that have applied k-medoids to edit distances such as LD [5]. Our study differs from those in that it defines a new distance for more general customer journey data with categories such as stages. Since this distance satisfies the axiom of distance, it can easily be extended to other methods.
In addition, this study proposed a visualization method using MDS based on the proposed distance. Unlike previous research, the distance used here is not a simple Levenshtein distance but is designed for marketing-specific applications. This approach not only enables the extraction of prototypes, but also allows for identifying their relative positioning, which facilitates interpretation as prototypes and evaluation of their validity.
Furthermore, our study proposes not only clustering but also a method to predict and derive counterfactual hypothetical cases. This approach also contributes to the research field of interpretable counterfactual explanations—or algorithmic recourse—from the following perspectives.
First, (i) plausibility: Each suggested sequence corresponds to an actual customer journey observed in the dataset. Counterfactual recommendations are generated based on sequences that were empirically observed.
Second, the method improves (ii) managerial interpretability: Since each edit operation can be directly associated with a specific marketing intervention (e.g., replacing the first touchpoint from “direct word-of-mouth” to “social media-based word-of-mouth”). Recent studies have shown that the usefulness of counterfactual explanations heavily depends on such validity.
Finally, it introduces (iii) a novel approach for sequential data: While robustness in structured data has been extensively studied, for sequential data, only recently have methods begun to address symbolic constraints on allowable edits. This study offers one such method.

15

However, establishing robustness guarantees for the proposed approach remains an important topic for future work.
Moreover, we propose a new survey method and a method for cleansing data rather than just providing a formal definition of data. These two are novelties and contributions in this study.
Regarding the shortcomings of the proposed study, they are presented below.
1. First, a customer journey not only includes different stages but also the actual behaviors (search, purchase, and non-purchase) and psychological states of consumers (such as perceived needs) and more specific information at touchpoints [4, 33]. Therefore, it is necessary to incorporate and extend this information. Since these pieces of information may have complex structures, distances between sequences can be obtained by using approaches like tree structures and graphs.
2. Another issue is estimating the level of awareness of consumers regarding their behavior and changes in their awareness.
3. Finally, regarding problems such as recall bias in data collection, a method that enables automatic data collection should be considered.

5

Conclusions

The overall contributions of the study are listed below.
1. A method to formally define data sequences and the distances between them using Levenshtein distance (LD) was proposed.
2. Typical cases were extracted using k-medoids. Furthermore, a method for visualizing these data was presented.
3. Both a method for predicting purchases using k-NN with the proposed distance and a method for extracting counterfactual explanations to interpret the prediction results were proposed.
4. A new survey method to obtain the data used in the study was proposed.
The survey-based data analysis revealed typical interpretable patterns. Furthermore, through visualization, the positioning of the data could be understood, predictions could be made regarding whether purchases would be performed, and counterfactual hypothetical cases and important elements could be detected.
The analysis and research methods used in this study will be useful for understanding the future behavior of consumers and designing consumer experiences [17].

Acknowledgments I would like to thank Yui Nakazawa (Faculty of Business, Kyoritsu Women’s University) for her help in designing and collecting the data used in the study. This study was supported by a JSPS Grant-in-Aid for Scientific Research (25K05381).
16

References [1] E. Anderl, I. Becker, F. von Wangenheim, and J. H. Schumann. Mapping the customer journey: Lessons learned from graph-based online attribution modeling. International Journal of Research in Marketing, 33(3):457–474, 2016.
[2] G. V. Bard. Spelling-error tolerant, order-independent pass-phrases via the dameraulevenshtein string-edit distance metric. In Proceedings of the Fifth Australasian Symposium on ACSW Frontiers, volume 68, pages 117–124. Australian Computer Society, 2007.
[3] N. Barwitz and P. Maas. Understanding the omnichannel customer journey: determinants of interaction choice. Journal of Interactive Marketing, 43(1):116–133, 2018.
[4] G. Bernard and P. Andritsos. A process mining based model for customer journey mapping. In Proceedings of the Forum and Doctoral Consortium Papers Presented at the 29th International Conference on Advanced Information Systems Engineering (CAiSE 2017), volume 1848, pages 49–56, 2017.
[5] G. Bernard and P. Andritsos. Contextual and behavioral customer journey discovery using a genetic approach. In Advances in Databases and Information Systems: 23rd European Conference, ADBIS 2019, volume 11695 of Lecture Notes in Computer Science, pages 251–266. Springer, 2019.
[6] G. Bernard and P. Andritsos. Discovering customer journeys from evidence: A genetic approach inspired by process mining. In Information Systems Engineering in Responsible Information Systems: CAiSE Forum 2019, volume 350 of Lecture Notes in Business Information Processing, pages 36–47. Springer, 2019.
[7] K. Bobroske, C. Larish, A. Cattrell, M. V. Bjarnadóttir, and L. Huan. The bird’seye view: a data-driven approach to understanding patient journeys from claims data.
Journal of the American Medical Informatics Association, 27(7):1037–1045, 2020.
[8] C. Bradley, L. Oliveira, S. Birrell, and R. Cain. A new perspective on personas and customer journey maps: Proposing systemic ux. International Journal of Human-Computer Studies, 148:102583, 2021.
[9] F. J. Damerau. A technique for computer detection and correction of spelling errors.
Communications of the ACM, 7(3):171–176, 1964.
[10] R. Dwivedi, D. Dave, H. Naik, S. Singhal, R. Omer, P. Patel, B. Qian, Z. Wen, T. Shah, G. Morgan, and R. Ranjan. Explainable ai (xai): Core ideas, techniques, and solutions.
ACM Computing Surveys, 55(9):1–33, 2023.
[11] A. Følstad and K. Kvale. Customer journeys: A systematic literature review. Journal of Service Theory and Practice, 28(2):196–227, 2018.

17

[12] M. Harbich, G. Bernard, P. Berkes, B. Garbinato, and P. Andritsos. Discovering customer journey maps using a mixture of markov models. In Proceedings of the International Symposium on Data-Driven Process Discovery and Analysis (SIMPDA 2017), pages 3–7, 2017.
[13] D. Herhausen, K. Kleinlercher, P. C. Verhoef, O. Emrich, and T. Rudolph. Loyalty formation for different customer journey segments. Journal of Retailing, 95(3):9–29, 2019.
[14] P. Hosseinzadeh, P. Li, O. Bahri, S. F. Boubrahimi, and S. M. Hamdi. Acts: Adaptive counterfactual explanations for time series data using barycenters. In 2024 IEEE International Conference on Big Data (BigData), pages 1327–1332. IEEE, 2024.
[15] A. H. Karimi, G. Barthe, B. Schölkopf, and I. Valera. A survey of algorithmic recourse:
Contrastive explanations and consequential recommendations. ACM Computing Surveys, 55(5):1–29, 2022.
[16] L. Kaufman and P. J. Rousseeuw. Partitioning around medoids (program pam). In Finding Groups in Data: An Introduction to Cluster Analysis, pages 68–125. Wiley, 1990.
[17] C. Kuehnl, D. Jozic, and C. Homburg. Effective customer journey design: Consumers’ conception, measurement, and consequences. Journal of the Academy of Marketing Science, 47:551–568, 2019.
[18] K. N. Lemon and P. C. Verhoef. Understanding customer experience throughout the customer journey. Journal of Marketing, 80(6):69–96, 2016.
[19] V. I. Levenshtein. Binary codes capable of correcting deletions, insertions, and reversals.
Soviet Physics Doklady, 10(8):707–710, 1966.
[20] S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions.
In Advances in Neural Information Processing Systems, volume 30, pages 4765–4774, 2017.
[21] C. Molnar. Interpretable Machine Learning. Lulu.com, 2020.
[22] S. A. Neslin. The omnichannel continuum: Integrating online and offline channels along the customer journey. Journal of Retailing, 98(1):111–132, 2022.
[23] M. Palazón, M. López, M. Sicilia, and I. López. The customer journey: A proposal of indicators to evaluate integration and customer orientation. Journal of Marketing Communications, 28(5):528–559, 2022.
[24] M. T. Ribeiro, S. Singh, and C. Guestrin. ”why should i trust you?”: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1135–1144, 2016.

18

[25] M. S. Rosenbaum, M. L. Otalora, and G. C. Ramı́rez. How to create a realistic customer journey map. Business Horizons, 60(1):143–150, 2017.
[26] P. J. Rousseeuw. Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20:53–65, 1987.
[27] N. Saeed, H. Nam, M. I. U. Haq, and D. B. Muhammad Saqib. A survey on multidimensional scaling. ACM Computing Surveys, 51(3):1–25, 2018.
[28] M. Shvo, A. C. Li, R. T. Icarte, and S. A. McIlraith. Interpretable sequence classification via discrete optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 9647–9656, 2021.
[29] B. Topaloglu, B. Oztaysi, and O. Dogan. Cart-state-aware discovery of e-commerce visitor journeys with process mining. Journal of Theoretical and Applied Electronic Commerce Research, 19(4):2851–2879, 2024.
[30] Y. Tueanrat, S. Papagiannidis, and E. Alamanos. Going on a journey: A review of the customer journey literature. Journal of Business Research, 125:336–353, 2021.
[31] S. Verma, V. Boonsanong, M. Hoang, K. Hines, J. Dickerson, and C. Shah. Counterfactual explanations and algorithmic recourses for machine learning: A review. ACM Computing Surveys, 56(12):1–42, 2024.
[32] M. Zhang, B. Kao, C. L. Yip, and D. Cheung. A gsp-based efficient algorithm for mining frequent sequences. In Proceedings of the 2001 International Conference on Artificial Intelligence, pages 497–503. CSREA Press, 2001.
[33] S. Zhao, W.-L. Hsu, G. Ma, T. Xu, G. Jacobson, and R. Rustamov. Characterizing and learning representation on customer contact journeys in cellular services. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3252–3260. ACM, 2020.

19


<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2408.05353
     paper_id : 2408.05353
     source   : paper2skills-vault/papers/07-NLP-VOC/2408.05353/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

arXiv:2408.05353v2 [cs.IR] 20 May 2025

IntentRec: Predicting User Session Intent with Hierarchical Multi-Task Learning Sejoon Oh

Moumita Bhattacharya

sejoono@netflix.com Netflix Los Gatos, United States

mbhattacharya@netflix.com Netflix Los Gatos, United States

Yesu Feng

Sudarshan Lamkhede

yfeng@netflix.com Netflix Los Gatos, United States

slamkhede@netflix.com Netflix Los Gatos, United States

ABSTRACT Recommender systems have played a critical role in diverse digital services such as e-commerce, streaming media, social networks, etc. If we know what a user’s intent is in a given session (e.g. do they want to watch short videos or a movie or play games; are they shopping for a camping trip), it becomes easier to provide high-quality recommendations. In this paper, we introduce IntentRec, a novel recommendation framework based on hierarchical multi-task neural network architecture that tries to estimate a user’s latent intent using their short- and long-term implicit signals as proxies and uses the intent prediction to predict the next item user is likely to engage with. By directly leveraging the intent prediction, we can offer accurate and personalized recommendations to users. Our comprehensive experiments on Netflix user engagement data show that IntentRec outperforms the state-of-the-art nextitem and next-intent predictors. We also share several findings and downstream applications of IntentRec.

1

Figure 1: Overview of user engagement data in Netflix. User intent can be associated with several interaction metadata. We leverage various implicit signals to predict user intent and next-item.

(e.g., discovering new content vs continue watching, etc.), Genre (e.g., horror, thriller, drama, etc.), movie or TV show preference, etc. While predicting the next item ID is the most important task, anticipating the user’s future intent (e.g., action type prediction) is also crucial, as it can enhance the next-item prediction.
To predict some form of a user’s session intent, previous studies have proposed simple multi-task learning (MTL) that adds several intent prediction heads to the next-item prediction model. However, existing intent prediction models have two major limitations. First, they lack a hierarchical prediction scheme where the intent prediction result can directly affect the next-item prediction. The hierarchical learning is preferred to the simple MTL or standalone intent prediction, as next-item prediction can be enhanced and personalized by leveraging the user intent as one of the input features [6, 8, 10, 27]. Second, they cannot incorporate the short- and long-term interests of a user into the intent and item prediction.
A user’s latest interest is crucial for next-item prediction and can be significantly different from their long-term preferences; e.g., a user might want to watch horror movies with friends even if they do not usually watch horror movies alone. Modeling such shortand long-term interests separately is a challenging task.
To address the above issues, we propose a novel sequential recommender system: IntentRec that can predict the next item, while also capturing the user’s intent and balancing their short- and long-term preferences. IntentRec consists of three major components: input feature constructor, user intent predictor, and next-item predictor. As one can imagine exact user intent is often not known, we leverage various proxy implicit signals (e.g., previous browsed shows/movies and genre preferences)

INTRODUCTION

Sequential recommender systems, specifically next-item prediction systems, are one of the most useful applications of machine learning in industry [3, 4, 11, 50, 51]. A well-designed recommender system can drive a lot of product and business impact by surfacing the relevant items to a member at the right time [13, 18]. For instance, in streaming services like Netflix, recommendations have been employed in diverse situations (e.g., discovering or searching shows) [1, 4, 13, 41] to maximize users’ satisfaction. Recently, predicting a user’s future intent in an online platform has gained attention [6, 8, 27, 50, 51], since such latent intent can lead to more accurate and curated recommendations.
The exact definition of a user’s intent varies across diverse applications and is often hidden. In Netflix, we identify several interaction metadata that can be associated with the user intent.
Specifically, we leverage the type of action a member takes on the product as a direct reflection of what they intend to do on the platform. For example, when a member plays a follow-up episode or scene of something they were already watching previously can be categorized as “continue watching” intent. Additionally, intents can also be whether a member wants to watch a movie or TV show, or is in the mood for a short-watch session or a long continue-watching session. As shown in Fig. 1, we have different metadata of user interactions that can be mapped to intents including, Action Type 1

Conference’17, July 2017, Washington, DC, USA

Oh, et al.

that we collect based on user interactions in Netflix. As will be described later, an input feature constructor combines those proxy implicit signals to estimate the member’s latent intent on Netflix.
We explicitly model the short-term interest of a user using implicit signals happening within a certain time threshold (e.g., one week) and incorporate it while constructing the input feature sequence, while the long-term interest of a user will be modeled via a Transformer [47] later. While IntentRec is designed for Netflix, IntentRec can be easily adapted to other domains by redefining user intent (e.g., genre → category) and choosing proper implicit signals for intent and item predictions.
We feed the input feature sequence of a user to a Transformer intent encoder to predict the user intent at each position in the sequence. The output sequence of the Transformer will be used for each intent prediction task (e.g., action type), and all the individual predictions are transformed into embeddings via projection layers. Finally, the projected embedding sequences are mixed by an attention layer and form the final user intent embedding sequence.
The intent embedding sequence will be combined with the input feature sequence to predict the next item of a user accurately.
The aforementioned intent-aware feature sequence is fed to a Transformer item encoder to predict the next item at each position in the sequence. Unlike the conventional next-item prediction, IntentRec utilizes hierarchical multi-task learning, where we conduct the intent prediction first and use the intent prediction output for the next-item prediction. After all predictions, their loss functions are combined with weights and jointly optimized together.
Extensive experiments on Netflix user engagement data demonstrate that IntentRec outperforms state-of-the-art user intent and next-item prediction models. Ablation studies show the effectiveness of the hierarchical multi-task learning of IntentRec and the contribution of each intent prediction task. We also find unique and meaningful clusters of users employing the predicted intent embedding and suggest downstream applications of IntentRec in Netflix.
The main contributions of our work are summarized as follows.

have gained popularity because of their ability to handle long sequences and position-awareness. Most of these methods cannot predict a user’s next intent and next item simultaneously.
User Intent Predictions. Estimating a user’s next intent has been investigated actively in diverse domains [6, 8, 22, 27–29, 44, 50, 51], as the user intent is a direct indicator of the user’s future interaction and leads to several downstream applications such as personalized and real-time recommendations.
The definition of user intent varies across papers as the user intent highly depends on the specific domain (e.g., e-commerce [21]
vs social media [51]). For instance, Fan et al. [10] predict the user intent (represented by a sentence query) among 𝑄 possible queries using a metapath-guided GNN model on Alibaba E-commerce. In session-based recommendations [14, 19, 24, 31, 34, 43, 48, 50, 52], the user intent in a given session can be defined as a high-level summary of the session (e.g., searching for new shoes) or a real-time transitional interest (e.g., add-to-cart → purchase). Xia et al. [51] is the state-of-the-art intent prediction model developed by Pinterest, where the user intent is defined by the potential action (e.g., click, repin, hide) for a given pin (or an item). Yet, all the above models have limitations that they neither predict the next item and intent of a user at the same time nor model short- and long-term interests of a user together.
Hierarchical Multi-task Learning. Multi-task learning (MTL)
has improved the generalization capability of deep neural networks used in computer vision, natural language processing (NLP), and recommender systems [36, 54]. Particularly, in recommender systems [12, 15, 38], MTL has enhanced the next-item prediction performance by sharing the knowledge obtained between auxiliary tasks (e.g., category prediction) and the main task (e.g., item prediction). We can further enhance the performance of MTL by setting a hierarchy between prediction tasks, where low-level tasks are conducted first, and the high-level task exploits the outputs of lowlevel ones as input features. We call this hierarchical MTL (H-MTL).
H-MTL has been widely adopted in computer vision [9, 30, 35], NLP [39, 40, 49], and only few in recommender systems [25, 32].
Our paper is the first H-MTL framework that can predict the user intent using both short- and long-term interests of a user.

(1) We propose a novel recommendation framework that can capture a user’s intent on the online platform and enhance the next-item prediction using the user intent.
(2) We introduce hierarchical multi-task learning for intent and item predictions from the short- and long-term interests of a user and show its effectiveness.

2

3 PROPOSED METHOD 3.1 Overview Our proposed recommender: IntentRec infers a user’s next session intent and leverages such intent predictions to enhance the nextitem recommendations in a hierarchical manner. As touched upon earlier, exact user intent is typically unknown, hence we use a few implicit and explicit signals such as action type, genre preference, among others to estimate the latent user intent. To this end, we define the user intent by a mixture of 4 different key metadata in Netflix (see Section 4.1.1 for details): Action Type, Genre Preference, Movie/Show, and Time-since-release (e.g., new content vs oldies). These proxies can be extended to include any number of other proxies, and we have chosen four for brevity. IntentRec also incorporates short-term interests of a user (i.e., interactions that happened within the last 𝐻 hours) into the model as input features.
Fig. 2 illustrates a high-level overview of IntentRec.

RELATED WORK

Sequential Recommendations. Sequential recommenders have been employed widely in industry including Spotify [11], Pinterest [51], Amazon [50], and Netflix [4]. Sequential recommenders are trained on historical user-item interactions and predict the next item given a sequence of observed items of a user [33]. Recurrent neural networks (RNNs) have been the main architecture for sequential recommendations recently due to their capability to process arbitrary sequences of inputs [53]. Earlier methods [3, 17] had used Long short-term memory (LSTM) [16] and Gated Recurrent Unit [7]. Recently, self-attention [11, 20, 23, 50] and Transformer [26, 42, 47, 51]
2

IntentRec: Predicting User Session Intent with Hierarchical Multi-Task Learning

Conference’17, July 2017, Washington, DC, USA

Figure 3: Given a high-level interaction sequence of a user, an input feature sequence is constructed by a concatenation of an interaction feature sequence and a short-term interest feature sequence.
Figure 2: An architectural illustration of our hierarchical multi-task learning model IntentRec for user intent and item predictions.

3.2

feature S𝑘 is defined as follows.
S𝑘 ∈ R𝑑short = 𝐸𝑛𝑐 (F𝑝𝑜𝑠 , . . . , F𝑘 ), 𝑝𝑜𝑠 = arg min (T𝑘 − T𝑖 ≤ 𝐻 ),

Input Feature Sequence Formation

1≤𝑖 ≤𝑘

We assume user engagement data consists of historical interactions (e.g. clicks, plays, etc.) that users have made in Netflix. We use the latest 𝑛 interactions of each user for training and testing. A user 𝑢’s engagement is represented as a temporal interaction sequence {𝑖𝑛𝑡 1, . . . , 𝑖𝑛𝑡𝑛 } (latest at the end).
Each interaction 𝑖𝑛𝑡𝑘 , 1 ≤ 𝑘 ≤ 𝑛 is associated with various metadata. We convert all categorical metadata of an interaction 𝑖𝑛𝑡𝑘 into embedding using embedding layers. Regarding numerical metadata, they are normalized first (e.g., between 0 and 1) and become 1-dimensional features. Some interaction metadata are used as both intent prediction labels and input features. Different interaction metadata would be used in other datasets/applications.
Input feature F𝑘 of an interaction 𝑖𝑛𝑡𝑘 is a concatenation of all categorical and numerical features of the interaction. i.e., F𝑘 ∈ R𝑑full = 𝐸𝑖I ⊕𝐸𝑐C𝑘 ⊕· · ·⊕𝑒𝑘 ⊕· · · , where ⊕ is a concatenation operator, 𝑘 𝐸 indicates a trainable embedding layer, and 𝑑 full is a dimension of the input feature F𝑘 . The interaction feature sequence {F1, . . . , F𝑛 } will be used later for intent and item-ID predictions of a user 𝑢.
A user’s short-term interest is crucial for accurate next-intent and next-item predictions [26, 27, 51]. Assuming we are given interactions {𝑖𝑛𝑡 1, . . . , 𝑖𝑛𝑡𝑘 } of a user 𝑢 and want to predict the next intent and item (e.g., 𝑖𝑘+1 ) of this user, a naive way to define the short-term interest is aggregating the recent 𝐿 interactions for each user (e.g., {𝑖𝑛𝑡𝑘 −𝐿+1, . . . , 𝑖𝑛𝑡𝑘 }) using an off-the-shelf encoder such as Transformer. However, as different users may have very different viewing patterns, the number of interactions that corresponds to short-term for some members might not be the same for some other members, hence we propose a personalized approach to define the short-term interest of a user.
We use a timestamp-based definition of the short-term interest.
Specifically, we set a time window hyperparameter 𝐻 (e.g., 1 week, 1 day, 1 hour, etc.) and treat the recent interactions happening within the window 𝐻 as the short-term interest of a user. Formally, given interactions {𝑖𝑛𝑡 1, . . . , 𝑖𝑛𝑡𝑘 } of a user 𝑢, the short-term interest

(1)
where 𝐸𝑛𝑐 (·) is an encoder (e.g., Transformer), and T indicates timestamps of interactions. In this way, our short-term interest features are more personalized compared to the interaction-based definition and aligned well with the business consideration. The short-term interest feature sequence {S1, . . . , S𝑛 } is concatenated with the interaction feature sequence {F1, . . . , F𝑛 } to form the final input feature sequence {F1 ⊕ S1, . . . , F𝑛 ⊕ S𝑛 }. The overall process is summarized in Fig. 3.

3.3

User Intent Prediction

In this paper, a user intent is represented as a mixture of distinctive labels in the user engagement data: Action type (e.g., discover new content, continue watching, etc.), Genre preference (e.g., thriller), Movie/Show type preference, Time-since-release information (e.g., within a week), Language preference, Expected session duration, Date/time behavioral pattern, etc.1 . Note that the intent definition varies across datasets/applications (e.g., genre in streaming vs category in E-commerce). Detailed descriptions of some labels used for experiments are provided in Section 4.1.1. Predictions of such user intents can serve as prior knowledge to the next-item predictor, and they can enhance the next-item prediction accuracy.
We leverage the input feature sequence {F1 ⊕ S1, . . . , F𝑛 ⊕ S𝑛 } of all users from Section 3.2 to train a user intent encoder and employ the trained encoder to predict the future intents of all users. Before the input sequence is fed to the Transformer, it goes through a fully connected layer for dimensionality reduction and normalization. Among various encoders, we use a Transformer encoder [47] to effectively model the long-term interest of a user via multi-head attention. We use timestamp embeddings (e.g., 𝐸𝑡T𝑘 ) as positional encoding in the Transformer. The Transformer generates an intent encoding sequence {𝐸 1intent, . . . , 𝐸𝑛intent } given the input 1 More labels can be also included as a user intent, but we leave it for future work.

3

Conference’17, July 2017, Washington, DC, USA

Oh, et al.

Figure 5: Given an intent-aware feature sequence of a user, a nextitem prediction vector for each sequence position is found by a Transformer encoder and a fully-connected layer.

𝛼 intent𝑀 ]). One may argue why we should use Z𝑘 instead of 𝐸𝑘intent from the Transformer as the final user intent embedding. The main advantage of Z𝑘 tells us the importance weight of each prediction head (e.g., action type) for each user profile, so that the model developers can know which prediction they should prioritize in the future and investigate the relations between importance weights and user attributes. However, we cannot conduct the aforementioned analyses if we simply use 𝐸𝑘intent without attention weights.
The above process is illustrated in Fig. 4.

Figure 4: Given an input feature sequence of a user, a user intent embedding sequence is constructed by an attention-based aggregation of auxiliary prediction (e.g., Action Type and Genre) results. We use ground-truth intent and item-ID labels to optimize predictions.

feature sequence. We add a causal mask to the Transformer that prevents the encoder from attending to future interactions.
Based on the intent encoding from the Transformer, we conduct multiple predictions of the aforementioned labels. For each prediction task, we transform the intent encoding to a prediction score vector via a fully-connected layer. For instance, given a user’s previous 𝑘 interactions {𝑖𝑛𝑡 1, . . . , 𝑖𝑛𝑡𝑘 } and the current intent encoding 𝐸𝑘intent , the 𝑖-th intent prediction vector at position 𝑘 is defined as

3.4

follows: 𝑝𝑘intent𝑖 ∈ R𝑑𝑖 = 𝜎 (𝐹𝐶𝑖 (𝐸𝑘intent )), where 𝑑𝑖 is the number of unique labels for the 𝑖-th intent, 𝐹𝐶𝑖 is a fully-connected layer, and 𝜎 is the Softmax function.
The final step is deriving a comprehensive intent embedding Z𝑘 that encompasses all the individual prediction vectors 𝑝𝑘intent𝑖 .
For that, all prediction vectors go through projection layers to have unified dimensionality, and they are added together using an attention layer. Attention weights are trainable and computed as follows: 𝛼 intent𝑖 = 𝐹𝐶 att (Proj𝑖 (𝑝𝑘intent𝑖 )), where Proj𝑖 : R𝑑𝑖 −→ R𝑑proj is a projection layer, and 𝐹𝐶 att ∈ R𝑑proj is a fully-connected layer. With attention weights 𝛼, Z𝑘 is computed as follows.
Z𝑘 ∈ R𝑑proj =

𝑀 ∑︁

𝜎 (𝛼 intent𝑖 )Proj𝑖 (𝑝𝑘intent𝑖 )

Next-item Prediction and Hierarchical MTL

We perform next-item prediction using the input feature sequence and user intent embedding obtained in previous steps. First, we concatenate the input feature sequence {F1 ⊕ S1, . . . , F𝑛 ⊕ S𝑛 } and intent embedding sequence {Z1, . . . , Z𝑛 } for each user. Then, the intent-aware feature sequence {F1 ⊕ S1 ⊕ Z1, . . . , F𝑛 ⊕ S𝑛 ⊕ Z𝑛 } again goes through the FC and normalization layer, and the output is fed to another Transformer encoder optimized for next-item prediction, whose architecture is similar to the intent encoder. Note that all Transformers used in the paper use timestamp embeddings as positional encoding. For instance, given each position 𝑘 in the sequence, the encoder produces an optimized representation 𝐸𝑘item for next-item prediction. The next-item prediction score vector 𝑝𝑘item ∈ R𝑑 |I| = 𝜎 (𝐹𝐶 item (𝐸𝑘item )) for a position 𝑘 of a user 𝑢 is calculated by feeding the Transformer encoding 𝐸𝑘item to a fullyconnected layer. Fig. 5 describes this next-item prediction process.
Notably, we found separate Transformer encoders for intent and item predictions empirically outperform the shared Transformer architecture between prediction tasks.

(2)

𝑖=1

where 𝑀 is the number of distinct intents we are predicting, and 𝜎 is a Softmax function across all attention weights (e.g., [𝛼 intent1 , . . . , 4

IntentRec: Predicting User Session Intent with Hierarchical Multi-Task Learning

Conference’17, July 2017, Washington, DC, USA

Regarding the overall training procedure of IntentRec, it utilizes a hierarchical multi-task learning paradigm where losses of multiple prediction tasks are jointly optimized, and there are hierarchical relationships between these prediction tasks [35, 37, 45, 49].
In this hierarchy, the main prediction task exploits the outputs from other auxiliary tasks as input features. In our case, we have several user intent prediction tasks as well as the next-item prediction.
Between the prediction tasks, intent predictions are conducted first, and the item-ID prediction comes at the end. We note that the nextitem prediction is not solely dependent on intent predictions, due to the presence of other input features (F𝑘 and S𝑘 ), and it indicates that the next-item recommendation accuracy will not be heavily affected by the inaccurate intent predictions.
The loss function of IntentRec is summed over all training minibatches of users and all positions in a user interaction sequence.
For each position 𝑘 in a user profile, our goal is to predict the next intent (e.g., intent𝑘+1 ) and next item (e.g., 𝑖𝑘+1 ) precisely. In addition, each interaction has different weights proportional to their duration during the training; we prioritize interactions with long duration since they have higher business values. For example, given the current mini-batch B, we use the following weighted Cross-Entropy loss for next-item prediction.
Litem = −

𝑛 ∑︁ ∑︁ 𝑢 ∈ B 𝑘=1

𝑑𝑘

|I| ∑︁

𝑦𝑘I [𝑖] · log(𝑝𝑘item [𝑖]),

4 key metadata (Action type, Genre preference, Movie/Show type preference, and Time-since-release) to include as intent prediction labels. Note that these intent prediction labels are subject to change depending on datasets/applications.
4.1.2 Baselines. We use the following state-of-the-art sequential recommendation models as baselines. We have excluded baselines [5]
having similar or older architectures than our current baselines or that are not sequential recommenders [55]. Note that we added additional fully-connected layers to LSTM [16], GRU [7], Transformer [47] baselines in order to predict user intent, while we used original implementations for other baselines.
(1) LSTM [16], GRU [7], Transformer [47]: We modify the original LSTM, GRU, and Trasnformer architectures for multitask learning. Specifically, for intent predictions, we add multiple intent prediction heads to each enconder.
(2) SASRec [20]: a self-attention-based recommender that can compute the relevance of each item in the sequence to the next item prediction and utilize these importance weights to predict the next item.
(3) BERT4Rec [42]: a BERT-like recommender that predicts the masked items in the sequence using their left and right context with bidirectional Transformer encoders.
(4) TransAct [51]: a Transformer-based recommender that extracts users’ short-term preferences from their real-time activities. As it is designed to predict a user’s next action given an item, we modify the architecture to predict the next item and action together. This is the state-of-the-art model for user intent (or action) predictions.
(5) IntentRec-V0: a strong production model and a simplified version of IntentRec that does not incorporate short-term interest features and intent prediction heads for next-item prediction. Thus, it can predict the next item only.

(3)

𝑖=1

where 𝑑𝑘 is the duration weight of the 𝑘-th interaction of a user 𝑢, and 𝑦𝑘I is a one-hot vector indicating the ground-truth next item 𝑖𝑘+1 , 𝑥 [𝑖] means the 𝑖-th element of a vector 𝑥. Intent prediction losses such as Lintent𝑖 are defined similarly to Eq. (3). If the intent prediction can have multiple ground-truth labels (e.g., an intent = “romance” + “comedy”), we modify the loss Eq. (3) to Binary CrossEntropy where all the ground-truth labels become positive labels.
Note that we do not utilize negative labels for Binary Cross-Entropy.
The complete loss function of IntentRec is defined below.
LIntentRec = Litem + 𝜆

𝑀 ∑︁

Lintent𝑖 ,

4.1.3 Evaluation Metrics. We use various standard ranking metrics typically used in recommender systems for evaluation; for brevity, our “accuracy” metric is one of the standard ranking metrics (e.g., MRR, Recall, NDCG, AUC). Due to the proprietary nature of the paper, we only provide relative % improvements of accuracy of a method compared to the state-of-the-art baseline: TransAct [51].

(4)

𝑖=1

where 𝜆 is an intent prediction coefficient (hyperparameter). Note that it is possible to assign trainable weights or business-based manual weights to intent prediction heads, but they show similar prediction performance compared to our current formula.

4.2

Next Item and Intent Prediction Accuracy

The golden metric to measure the effectiveness of IntentRec is how much it improves the next-item prediction accuracy via utilizing intent prediction results, compared to baselines.
As shown in Table 1, IntentRec presents the highest next-item prediction accuracy across all baselines. Remarkably, IntentRec outperforms the best baselines: TransAct and IntentRec-V0; for instance, IntentRec shows 7.4% accuracy improvement compared to TransAct with statistical significance (p-values from Student’s t-test < 0.01). Regarding baselines, they all exhibit limited next-item prediction performance as they cannot predict users’ intents (marked as N/A in the table) or incorporate the user intents to the next-item prediction directly. TransAct [51] and IntentRecV0 show relatively higher accuracy than other baselines as they can leverage interaction metadata, while other baselines do not incorporate interaction metadata due to their architectural limitations.

4 EXPERIMENTS 4.1 Experimental Setup 4.1.1 Dataset. We use proprietary user engagement data collected in Netflix. We preprocess raw engagement sequences (i.e., every interactions a user has made on Netflix) of users. The preprocessing is based on some assumptions gathered from internal research to create higher-level engagements. We then randomly sample users from all users to generate training/validation/test data. Rich metadata of each interaction is available in the Netflix dataset; e.g., ‘Action Type’ information indicates a category of an interaction such as continuing to watch something the user has previously started watching, discovering content that the member has never seen on Netflix before, and others. Among all metadata, we select 5

Conference’17, July 2017, Washington, DC, USA

Oh, et al.

Table 1: Next-item and next-intent prediction results of baselines and our proposed method IntentRec on the Netflix user engagement dataset. All the metrics are represented as relative % improvements compared to the TransAct [51] baseline. N/A indicates that a model is not capable of predicting a certain intent (e.g., Action Type) of a user. The best baseline results are colored gray. For the next-item prediction task, IntentRec presents 7.4% accuracy improvement compared to the best baseline, with statistical significance.
Models \ Metrics

Item-ID Prediction Accuracy

Action Type Prediction Genre Prediction Accuracy Accuracy Baseline Methods -7.51% -2.48% -7.40% -2.30% -5.33% -1.56% N/A N/A N/A N/A +0.00% +0.00%

Movie/Show Prediction Accuracy

Time-since-release Prediction Accuracy

LSTM GRU Transformer SASRec BERT4Rec TransAct IntentRec-V0 (production model)

-25.2% -21.9% -17.5% -20.3% -25.0% +0.00%

-0.93% -0.92% -0.77% N/A N/A +0.00%

-0.58% -0.51% -0.41% N/A N/A +0.00%

-0.82%

N/A

N/A

N/A

N/A

IntentRec

+7.40%

Proposed Method +3.31% +2.78%

+0.41%

+0.84%

Table 2: Ablation study of different model architectures for IntentRec on the Netflix user engagement dataset. All the metrics are represented as relative % improvements compared to the V1 baseline. N/A indicates that a model is not capable of predicting a certain intent of a user. Our proposed hierarchical learning with short-term modeling shows the best performance.
Models \ Metrics V0: next-item prediction only V1: simple multi-task learning V2: hierarchical multi-task learning

Item-ID Prediction Accuracy

Action Type Prediction Genre Prediction Accuracy Accuracy Variants of IntentRec

Movie/Show Prediction Accuracy

Time-since-release Prediction Accuracy

-0.06%

N/A

N/A

N/A

N/A

+0.00%

+0.00%

+0.00%

+0.00%

+0.00%

+4.53%

+1.93%

+0.57%

+0.33%

+0.53%

Proposed Method V3-last-1-month:
short-term-as-input V3-last-1-week:
short-term-as-input

+8.56%

+3.09%

+2.85%

+0.80%

+0.72%

+8.28%

+3.43%

+2.49%

+0.38%

+1.03%

Table 3: Ablation study of each prediction head of IntentRec on the Netflix user engagement dataset. All the metrics are represented as relative % improvements compared to the V0 baseline for item-ID prediction and each variant for intent prediction. N/A indicates that a model is not capable of predicting a certain intent of a user. Action Type prediction is the most important one for next-item prediction.
Models \ Metrics V0: next-item prediction only V3-only-ActionType Prediction V3-only-Genre Prediction V3-only-Movie/ Show Prediction V3-only-Time-since -release Prediction

Item-ID Prediction Accuracy

Action Type Prediction Genre Prediction Accuracy Accuracy Variants of IntentRec

Movie/Show Prediction Accuracy

Time-since-release Prediction Accuracy

+0.00%

N/A

N/A

N/A

N/A

+7.73%

+0.00%

N/A

N/A

N/A

+6.50%

N/A

+0.00%

N/A

N/A

+6.52%

N/A

N/A

+0.00%

N/A

+7.18%

N/A

N/A

N/A

+0.00%

-0.39%

+0.23%

+0.63%

Proposed Method V3-all:
all prediction heads

+8.35%

+0.11%

IntentRec also shows superior prediction performance of user intents among all methods; IntentRec shows the highest accuracy across all intent prediction tasks (e.g., Action Type). However, the intent prediction accuracy improvements are smaller than the

next-item prediction ones as the hierarchical learning of IntentRec prioritizes to optimize next-item prediction over next-intent predictions; in other words, the intent predictor of IntentRec can be fine-tuned to find the optimal solutions for next-item prediction, 6

IntentRec: Predicting User Session Intent with Hierarchical Multi-Task Learning

Conference’17, July 2017, Washington, DC, USA

not for next-intent prediction. This trade-off can be moderated by changing the intent prediction coefficient 𝜆 during the training (i.e., larger 𝜆 focuses more on intent prediction).

4.3

Ablation Studies of IntentRec

We conduct ablation studies of IntentRec with respect to its model architecture and different intent prediction heads.
Table 2 shows how each architectural component of IntentRec improves the next-item and next-intent prediction performance.
The first version (V1) is extending IntentRec-V0 to multi-task learning setting using extra intent prediction heads. While it exhibits competitive intent prediction accuracy, its item-ID prediction accuracy is almost close to the existing model. One potential reason is that intent predictions do not affect the next-item prediction directly. Those predictions are connected together via the shared Transformer encoder; however, intent prediction results should be directly fed to the next-item predictor to improve the prediction accuracy. Based on this intuition, the second version (V2) uses two Transformer encoders to perform intent and item predictions separately. The V2 model also leverages hierarchical learning which performs the intent prediction first and item prediction next with the intent prediction results. It leads to significant improvements in next-item prediction. The final version (V3) adds short-term interest features to the V2 model, where we define the short-term as interactions happening within 1-week or 1-month from the current timestamp. While both thresholds are reasonable as per prediction accuracy, we choose the 1-week threshold considering the business value and consistency with the current company policy.
Table 3 indicates the contribution of an individual intent prediction head used in IntentRec to its prediction performance. While all variants outperform IntentRec-V0, predicting the Action Type label is the most important task among all, as per next-item prediction accuracy. It is intuitive since the user interactions are classified into 11 unique “Action Type” labels by a business-aware heuristic, which can be a direct translation of a user’s latent intent. Timesince-release prediction is also crucial since certain users tend to engage with newly released shows/movies more frequently than other users. Genre and Movie/Show predictions are less helpful than the others, but they still have downstream applications and business values. Using all prediction heads together (V3-all), it leads to the best performance with respect to intent and item predictions.

4.4

Figure 6: K-means++ (K=10) clustering of user intent embeddings found by IntentRec; T-SNE [46] is used for visualization. IntentRec finds unique clusters of users that share the similar intent.

Fig. 6 represents 10 unique clusters of user intent embeddings obtained by IntentRec. We use T-SNE [46] algorithm to visualize the high-dimensional intent embeddings to a two-dimensional image. Each colored dot in the figure indicates a randomly-sampled user close to each cluster center. We can find unique and meaningful user clusters that share a similar intent. For instance, there are two distinctive user groups that enjoy discovering new content vs continue-watching recent/favorite content. There is also an anime/kids genre enthusiast group where most of their interactions are from anime and kids genres. These examples imply user intent embeddings obtained by IntentRec are accurate.

4.5

Qualitative Analysis: Attention Weights

The attention layer in our intent predictor (Fig. 4) generates importance weights of each intent prediction head (i.e., 𝛼 intent𝑖 ), given a user’s historical interactions. These weights are personalized since they are computed based on the interaction feature sequence of each user. We can define a user’s primary intent by investigating the highest value of attention weights of this user.
Fig. 7 shows two user profiles whose primary intents are “fantasy genre” and “old-fashioned content”, respectively. The first user’s attention weight for genre prediction (0.52) is the highest among all weights, which indicates this user’s predicted intent is mainly related to watching specific genres. As expected, the user’s interactions mostly consist of fantasy shows/movies (Teen Wolf and Harry Potter), and IntentRec provides relevant fantasy-genre recommendations by capturing the primary intent of the user. On the other hand, IntentRec interprets the second user’s primary intent as watching old-fashioned content, since her attention weight for the time-since-release prediction is the highest. This intent prediction also aligns well with the historical interactions of the second user.

Qualitative Analysis: User Intent Clustering

Given a user and her previous interactions, IntentRec is able to predict this user’s current intent and generate an intent embedding by aggregating auxiliary prediction results (e.g., Action Type, Genre) via an attention layer. We conduct a qualitative analysis of user intent embeddings to validate their quality and accuracy.
Specifically, we apply K-means++ [2] clustering algorithm on intent embeddings of all users in the training data where the number of clusters (K) is set to 10. For each found cluster, we randomly sample a few users (e.g., 10) close to the cluster center on the embedding space. After that, we manually investigate the similarities between chosen users and determine the concept of the intent cluster (e.g., Rewatchers) by the commonalities between users.
7

Conference’17, July 2017, Washington, DC, USA

Oh, et al.

originally wanted to watch movies but got drawn to the latest TV show and ended up watching the TV show.
Downstream Application. One of the advantages of our proposed user session intent model is that the intents themselves are interpretable and can be directly associated with a member’s need.
Hence, it opens up a plethora of applications, such as User Interface (UI) optimization, analytics, and signals into various downstream ML models for personalization and recommendations. For example, we can use IntentRec to nudge members with explicit UI interventions to hone in or pivot a user based on the model’s prediction of user session intent. We can also directly use this model to replace the current next-item prediction model, as we have shown in the results that our approach outperforms the current model.

6 Figure 7: Attention weights of intent prediction heads for two distinctive users. Based on a user’s profile, IntentRec finds personalized attention weights and provides relevant recommendations.

We note that these personalized attention weights can be updated real-time in online recommendation setup using the inference mode of the latest trained model of IntentRec, while the full model training can be done regularly (e.g., every week). These real-time attention weights can be used as numerical evidence for explaining our intent prediction results to users and other downstream applications (e.g., search optimization).

5

CONCLUSION

We proposed a novel recommendation framework: IntentRec that can predict a user’s latent session intent and enhance next-item prediction by leveraging the intent prediction result. The hierarchical multi-task learning allows IntentRec to predict and exploit the user intent efficiently for personalized and accurate recommendations. Our extensive experiments on the Netflix user engagement dataset demonstrate the usefulness of user intent prediction for various applications.
Future work of IntentRec includes (1) scaling up and optimizing the training of IntentRec in multi-GPU and distributed computing settings, (2) real-time updates of user intent and item embeddings, and (3) incorporating large language models (LLMs) into IntentRec for more comprehensive next intent and item predictions.

DISCUSSION REFERENCES

Generalizability on Other Domains. Although IntentRec is tailored to Netflix, the user intent definition and prediction can be adjusted for other domains such as E-commerce. We design our methodology to be highly generalizable so that it can be applied to diverse domains. For instance, IntentRec can employ both common information (e.g., timestamp) and domain-specific metadata (e.g., genre or item taxonomy) as features, while leveraging the corresponding engagements in the product (e.g. click or purchase for e-commerce). Moreover, the user intent can be defined in coarsegrained (i.e., fewer intent labels) and fine-grained ways (i.e., many intent labels) depending on the domain needs. Exploring the empirical transferability of IntentRec to other domains will be the future work. Additionally, getting access to such rich information about other domains is challenging, due to obvious data privacy reasons.
Hence, we share the performance of our proposed approach only on one domain but as mentioned above, nothing in the model architecture makes any domain specific assumption, hence, it should be extensible to other domains.
Contradictory Predictions between Intent and Item Predictors. One might ask what if the predictions of the intent and item predictions tasks are contradictory. For instance, the result of intent prediction is ‘watching a movie’, whereas the predicted next item is a TV show. We conducted the prediction alignment analysis (e.g., see Fig. 4), and we found that user intent clusters are aligned well with the users’ interactions (e.g., next item) in most cases. Even if the predictions do not align, it can be legitimate since the user preference can be suddenly shifted. For instance, on Netflix, a user

[1] Xavier Amatriain and Justin Basilico. 2015. Recommender systems in industry:
A netflix case study. In Recommender systems handbook. Springer, 385–419.
[2] David Arthur and Sergei Vassilvitskii. 2007. K-means++ the advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms. 1027–1035.
[3] Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li, Vince Gatto, and Ed H Chi. 2018. Latent cross: Making use of context in recurrent recommender systems.
In Proceedings of the eleventh ACM international conference on web search and data mining. 46–54.
[4] Moumita Bhattacharya and Sudarshan Lamkhede. 2022. Augmenting Netflix Search with In-Session Adapted Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems. 542–545.
[5] Tong Chen, Hongzhi Yin, Hongxu Chen, Rui Yan, Quoc Viet Hung Nguyen, and Xue Li. 2019. Air: Attentional intention-aware recommender systems. In 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 304–315.
[6] Yongjun Chen, Zhiwei Liu, Jia Li, Julian McAuley, and Caiming Xiong. 2022.
Intent contrastive learning for sequential recommendation. In Proceedings of the ACM Web Conference 2022. 2172–2182.
[7] Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.
In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 1724–1734.
[8] Yujuan Ding, Yunshan Ma, Wai Keung Wong, and Tat-Seng Chua. 2021. Modeling instant user intent and content-level transition for sequential fashion recommendation. IEEE Transactions on Multimedia 24 (2021), 2687–2700.
[9] Jianping Fan, Tianyi Zhao, Zhenzhong Kuang, Yu Zheng, Ji Zhang, Jun Yu, and Jinye Peng. 2017. HD-MTL: Hierarchical deep multi-task learning for large-scale visual recognition. IEEE transactions on image processing 26, 4 (2017), 1923–1938.
[10] Shaohua Fan, Junxiong Zhu, Xiaotian Han, Chuan Shi, Linmei Hu, Biyu Ma, and Yongliang Li. 2019. Metapath-guided heterogeneous graph neural network for intent recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2478–2486.
[11] Ziwei Fan, Zhiwei Liu, Yu Wang, Alice Wang, Zahra Nazari, Lei Zheng, Hao Peng, and Philip S Yu. 2022. Sequential recommendation via stochastic self-attention.
In Proceedings of the ACM Web Conference 2022. 2036–2047.
8

IntentRec: Predicting User Session Intent with Hierarchical Multi-Task Learning

Conference’17, July 2017, Washington, DC, USA

[34] Zhiqiang Pan, Fei Cai, Yanxiang Ling, and Maarten de Rijke. 2020. An intentguided collaborative machine for session-based recommendation. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval. 1833–1836.
[35] Homin Park, Homanga Bharadhwaj, and Brian Y Lim. 2019. Hierarchical multitask learning for healthy drink classification. In 2019 International Joint Conference on Neural Networks (IJCNN). IEEE, 1–8.
[36] Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098 (2017).
[37] Victor Sanh, Thomas Wolf, and Sebastian Ruder. 2019. A hierarchical multi-task approach for learning embeddings from semantic tasks. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 6949–6956.
[38] Walid Shalaby, Sejoon Oh, Amir Afsharinejad, Srijan Kumar, and Xiquan Cui.
2022. M2TRec: Metadata-aware Multi-task Transformer for Large-scale and Cold-start free Session-based Recommendations. In Proceedings of the 16th ACM Conference on Recommender Systems. 573–578.
[39] Minguang Song and Yunxin Zhao. 2022. Enhance Rnnlms with Hierarchical Multi-Task Learning for ASR. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 6102–6106. https://doi.org/ 10.1109/ICASSP43922.2022.9747525 [40] Wei Song, Ziyao Song, Lizhen Liu, and Ruiji Fu. 2020. Hierarchical Multi-task Learning for Organization Evaluation of Argumentative Student Essays. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, Christian Bessiere (Ed.). International Joint Conferences on Artificial Intelligence Organization, 3875–3881. https://doi.org/10.24963/ijcai.2020/536 [41] Harald Steck, Linas Baltrunas, Ehtsham Elahi, Dawen Liang, Yves Raimond, and Justin Basilico. 2021. Deep learning for recommender systems: A Netflix case study. AI Magazine 42, 3 (2021), 7–18.
[42] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang.
2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.
[43] Zhu Sun, Hongyang Liu, Xinghua Qu, Kaidong Feng, Yan Wang, and Yew Soon Ong. 2024. Large Language Models for Intent-Driven Session Recommendations.
In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 324–334.
[44] Md Mehrab Tanjim, Congzhe Su, Ethan Benjamin, Diane Hu, Liangjie Hong, and Julian McAuley. 2020. Attentive sequential models of latent intent for next item recommendation. In Proceedings of The Web Conference 2020. 2528–2534.
[45] Bing Tian, Yong Zhang, Jin Wang, and Chunxiao Xing. 2019. Hierarchical InterAttention Network for Document Classification with Multi-Task Learning.. In IJCAI. 3569–3575.
[46] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE.
Journal of machine learning research 9, 11 (2008).
[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).
[48] Shoujin Wang, Liang Hu, Yan Wang, Quan Z Sheng, Mehmet Orgun, and Longbing Cao. 2019. Modeling multi-purpose sessions for next-item recommendations via mixture-channel purpose routing networks. In International Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence.
[49] Xinyi Wang, Guangluan Xu, Zequn Zhang, Li Jin, and Xian Sun. 2021. Endto-end aspect-based sentiment analysis with hierarchical multi-task learning.
Neurocomputing 455 (2021), 178–188.
[50] Yu Wang, Zhengyang Wang, Hengrui Zhang, Qingyu Yin, Xianfeng Tang, Yinghan Wang, Danqing Zhang, Limeng Cui, Monica Cheng, Bing Yin, et al. 2023.
Exploiting intent evolution in e-commercial query recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
5162–5173.
[51] Xue Xia, Pong Eksombatchai, Nikil Pancha, Dhruvil Deven Badani, Po-Wei Wang, Neng Gu, Saurabh Vishwas Joshi, Nazanin Farahpour, Zhiyuan Zhang, and Andrew Zhai. 2023. TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 5249–5259.
[52] Peiyan Zhang, Jiayan Guo, Chaozhuo Li, Yueqi Xie, Jae Boum Kim, Yan Zhang, Xing Xie, Haohan Wang, and Sunghun Kim. 2023. Efficiently leveraging multilevel user intent for session-based recommendation via atten-mixer network.
In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining. 168–176.
[53] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. ACM computing surveys (CSUR)
52, 1 (2019), 1–38.
[54] Yu Zhang and Qiang Yang. 2018. An overview of multi-task learning. National Science Review 5, 1 (2018), 30–43.
[55] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In Proceedings of the AAAI conference on artificial intelligence, Vol. 33.
5941–5948.

[12] Chen Gao, Xiangnan He, Dahua Gan, Xiangning Chen, Fuli Feng, Yong Li, TatSeng Chua, and Depeng Jin. 2019. Neural multi-task recommendation from multi-behavior data. In 2019 IEEE 35th international conference on data engineering (ICDE). IEEE, 1554–1557.
[13] Carlos A Gomez-Uribe and Neil Hunt. 2015. The netflix recommender system:
Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS) 6, 4 (2015), 1–19.
[14] Jiayan Guo, Yaming Yang, Xiangchen Song, Yuan Zhang, Yujing Wang, Jing Bai, and Yan Zhang. 2022. Learning multi-granularity consecutive user intent unit for session-based recommendation. In Proceedings of the fifteenth ACM International conference on web search and data mining. 343–352.
[15] Guy Hadash, Oren Sar Shalom, and Rita Osadchy. 2018. Rank and rate: multi-task learning for recommender systems. In Proceedings of the 12th ACM Conference on Recommender Systems. 451–454.
[16] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.
[17] Jin Huang, Wayne Xin Zhao, Hongjian Dou, Ji-Rong Wen, and Edward Y Chang.
2018. Improving sequential recommendation with knowledge-enhanced memory networks. In The 41st international ACM SIGIR conference on research & development in information retrieval. 505–514.
[18] Dietmar Jannach and Michael Jugovac. 2019. Measuring the business value of recommender systems. ACM Transactions on Management Information Systems (TMIS) 10, 4 (2019), 1–23.
[19] Di Jin, Luzhi Wang, Yizhen Zheng, Guojie Song, Fei Jiang, Xiang Li, Wei Lin, and Shirui Pan. 2023. Dual intent enhanced graph neural network for sessionbased new item recommendation. In Proceedings of the ACM Web Conference 2023.
684–693.
[20] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 197–206.
[21] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In Proceedings of the 28th ACM international conference on information and knowledge management.
2615–2623.
[22] Haoyang Li, Xin Wang, Ziwei Zhang, Jianxin Ma, Peng Cui, and Wenwu Zhu. 2021.
Intention-aware sequential recommendation with structured intent transition.
IEEE Transactions on Knowledge and Data Engineering 34, 11 (2021), 5403–5414.
[23] Jiacheng Li, Yujie Wang, and Julian McAuley. 2020. Time interval aware selfattention for sequential recommendation. In Proceedings of the 13th international conference on web search and data mining. 322–330.
[24] Yinfeng Li, Chen Gao, Hengliang Luo, Depeng Jin, and Yong Li. 2022. Enhancing hypergraph neural networks with intent disentanglement for session-based recommendation. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 1997–2002.
[25] Nicholas Lim, Bryan Hooi, See-Kiong Ng, Yong Liang Goh, Renrong Weng, and Rui Tan. 2022. Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval.
[26] Qiao Liu, Yifu Zeng, Refuoe Mokhosi, and Haibin Zhang. 2018. STAMP: shortterm attention/memory priority model for session-based recommendation. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1831–1839.
[27] Zhaoyang Liu, Haokun Chen, Fei Sun, Xu Xie, Jinyang Gao, Bolin Ding, and Yanyan Shen. 2021. Intent preference decoupling for user representation on online recommender system. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence. 2575–2582.
[28] Zhiwei Liu, Xiaohan Li, Ziwei Fan, Stephen Guo, Kannan Achan, and S Yu Philip.
2020. Basket recommendation with multi-intent translation graph neural network.
In 2020 IEEE International Conference on Big Data (Big Data). IEEE, 728–737.
[29] Jianxin Ma, Chang Zhou, Hongxia Yang, Peng Cui, Xin Wang, and Wenwu Zhu.
2020. Disentangled self-supervision in sequential recommenders. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 483–491.
[30] Duy-Kien Nguyen and Takayuki Okatani. 2019. Multi-Task Learning of Hierarchical Vision-Language Representation. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019), 10484–10493.
[31] Sejoon Oh, Ankur Bhardwaj, Jongseok Han, Sungchul Kim, Ryan A Rossi, and Srijan Kumar. 2022. Implicit session contexts for next-item recommendations. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 4364–4368.
[32] Sejoon Oh, Walid Shalaby, Amir Afsharinejad, and Xiquan Cui. 2023. Hierarchical Multi-Task Learning Framework for Session-based Recommendations. arXiv preprint arXiv:2309.06533 (2023).
[33] Sejoon Oh, Berk Ustun, Julian McAuley, and Srijan Kumar. 2022. Rank list sensitivity of recommender systems to interaction perturbations. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management.
1584–1594.
9


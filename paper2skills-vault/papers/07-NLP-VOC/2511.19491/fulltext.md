<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2511.19491
     paper_id : 2511.19491
     source   : https://arxiv.org/html/2511.19491v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# $OpenCML$: End-to-End Framework of Open-world Machine Learning to Learn Unknown Classes Incrementally

Jitendra Parmar Email: jitendra.parmar@juetguna.in Affiliation: Computer Science and Engineering, Jaypee University of Engineering and Technology, Agra-Mumbai Hwy, Guna, 473226, Madhya Pradesh, India    Praveen Singh Thakur Email: praveen.thakur@nmims.edu Affiliation: Computer Engineering, SVKM’S NMIMS STME, Super Corridor Rd, Indore, 453112, Madhya Pradesh, India

###### Abstract

Open-world machine learning is an emerging technique in artificial intelligence, where conventional machine learning models often follow closed-world assumptions, which can hinder their ability to retain previously learned knowledge for future tasks. However, automated intelligence systems must learn about novel classes and previously known tasks. The proposed model offers novel learning classes in an open and continuous learning environment. It consists of two different but connected tasks. First, it discovers unknown classes in the data and creates novel classes; next, it learns how to perform class incrementally for each new class. Together, they enable continual learning, allowing the system to expand its understanding of the data and improve over time. The proposed model also outperformed existing approaches in open-world learning. Furthermore, it demonstrated strong performance in continuous learning, achieving a highest average accuracy of 82.54% over four iterations and a minimum accuracy of 65.87%.

###### keywords

Continual Machine Learning, Open-world machine learning, Class incremental Learning, Lifelong Machine Learning, Text classification, Knowledge Discovery

## 1 Introduction

Human beings learn from childhood by experiencing and analysing things; they build a knowledge base (memories). It is an incremental learning process; generally, they do not forget this learning easily and utilise it for a long time in their future tasks. This phenomenon of incremental learning reduces effort, as there is no need to repeat similar tasks. The classical machine learning technique does not follow incremental learning techniques; for this reason, it must repeatedly relearn old tasks, which decreases system performance. It is challenging to develop classical or traditional machine learning models that can learn incrementally, as humans do. Here, in the case of traditional machine learning, we have to train the machine every time it encounters unknown data. This type of data is referred to as open-world data, which does not appear during training. In contrast, classical machine learning uses a closed-world assumption, where each testing data instance is available during the training.

Let us consider a dataset of 10,000 labelled text inputs, where each input is assigned to one of five intents: Transfer, Distance, Rewards Balance, Travel, and Utility. The model is trained on this dataset to classify the intents of new text inputs. However, new intents may become apparent over time, missing from the initial data set. For example, users might start questioning regional events or cafe requests. We can use incremental class learning to update the model with new labelled data when available( to manage these unknown intents). For example, if we receive 100 new inputs that belong to an unknown intent, such as ”events,” we can provide these new data attributes to the model and modify its parameters accordingly. In this way, the model can adapt these unknown intentions without retraining the entire model for all data; also, it will not forget the previous knowledge. By merging incremental class learning with OWML, the model can learn continuously and enhance its ability to determine unknown intents and classify data more accurately. It can benefit domains and applications where unknown data is continuously growing and varying, such as chatbots or voice assistants.

The literature contains numerous attempts to address the issue of open-world machine learning. learning McCloskey and Cohen (1989); Kirkpatrick et al. (2017); Fei and Liu (2016); Shu et al. (2017); Prakhya et al. (2017); Lin and Xu (2019a); Vedula et al. (2019); Vedula et al. (2020). Similarly, for class incremental learning Javed and White (2019); Gallardo et al. (2021); Mehta et al. (2021); Wu et al. (2022); Madaan et al. (2021); Purushwalkam et al. (2022); Fini et al. (2022); Cha et al. (2021); Robbins and Monro (1951); Kiefer and Wolfowitz (1952); Bottou et al. (2018); Lopez-Paz and Ranzato (2017); Chaudhry et al. (2018b); Tang et al. (2021); Riemer et al. (2018); Kirkpatrick et al. (2017); Schwarz et al. (2018); Zenke et al. (2017); Aljundi et al. (2018); Chaudhry et al. (2018a); Benzing (2022); Riemer et al. (2018); Chaudhry et al. (2019); Lopez-Paz and Ranzato (2017); Rebuffi et al. (2017); Caccia et al. (2020); Belouadah and Popescu (2019); Gong et al. (2022); Ebrahimi et al. (2020); Saha et al. (2020) Among them, perhaps the most effective strategy is to keep a memory buffer that stores part of the previous knowledge for the rehearsal Robins (1993); Robins (1995).Therefore, it needs a model to effectively integrate novel concepts without omitting the prior information, also known as the stability-plasticity dilemma. Excess plasticity usually yields a significant performance degradation of the old classes, which is referred to as catastrophic forgetting French and Chater (2002). Although samples are saved in memory, they have many challenges in CML, which can be understood from the representation and classifier learning views Serra et al. (2018). Representation learning refers to the process of learning data representations. Due to limited memory, classifier learning often requires improving the class imbalance between prior and novel classes. A restricted memory size typically leads to an imbalance between prior and novel classes.

To the best of our knowledge, no framework exists that can integrate both open-world learning and incremental class learning for text classification. In this paper, we propose OpenCML, an end-to-end framework for open-world machine learning that learns unknown classes incrementally, addressing both open-world learning and incremental class learning. Specifically, we address the following research questions using the proposed OpenCML framework.

-

$RQ1$: How can we deploy and scale CIL techniques in real-world applications in OWML?

-

$RQ2$: How can we develop cost-efficient memory management and storage techniques for CIL in OWML?

-

$RQ3$: How can we design methods to recognize and mitigate catastrophic forgetting in CML in OWML?

-

$RQ4$: How can we integrate previous knowledge to enhance CIL in OWML?

The proposed framework can identify unknown instances in the test data using a Convolutional Neural Network (CNN)- based model, which employs a 1-vs-rest approach to distinguish instances from existing ones. From known classes, we used examples to create an exemplar. The instances that our model rejects are further used to create novel classes. We applied Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) to identify the optimal clusters among the unknown instances. After forming clusters of unknown instances, the singleRank keyword extraction technique is used to generate new labels for the newly formed classes. We employed a custom loss function to prevent forgetting while learning classes incrementally.

*Table 1: List of variables used in the methodology and experiments*

| Notation | Meaning | Notation | Meaning |

$d_{im}$ $T_{h}$ | | World Vector | | Threshold |

$F_{m}$ $L(\theta)$ | | Feature Map | | Custom Loss |

$f$ $L_{Ds}(\theta)$ | | Non-leaner Features | | Distillation loss |

$b$ $L_{Ce}(\theta)$ | | Bias | | cross-entropy loss |

$S_{cn}$ $g_{i}$ | | Number of known Classes | | Ground Truth |

$I$ $T$ | | Indicator Function | | distillation parameter |

$I_{c}$ $cl$ | | Number of Instances | | Classification Layer |

$C_{f}$ $M$ | | Cluster Features | | Exemplar Memory |

$N_{d}$ | | Data Points | | |

$L_{s}$ | | Linear Sum | | |

$S_{s}$ | | Squared Sum | | |

$\beta_{f}$ | | Branching Factor | | |

## 2 Literature Reviews

Continual machine learning or lifelong machine learning was first introduced in the late $90$’s Thrun (1995). In the last decade, continuous machine learning has drawn significant attention in the deep learning and Natural Language Processing (NLP) communities, and it is also known as lifelong machine learning Chen and Liu (2018). The issue is that when a neural network is used to memorize a series of tasks, remembering the subsequent tasks may hinder the execution of the models learned for the primary tasks. However, the human brain retains an extraordinary proficiency to perform various assignments simultaneously without negatively hindering one another. Continual learning algorithms attempt to perform this identical capacity for neural networks and to solve the catastrophic forgetting problem McCloskey and Cohen (1989); Kirkpatrick et al. (2017).

### 2.1 Open-world machine learning

Traditional machine learning approaches have produced promising results for decades in all domains of data analysis. However, it has some limitations Bottou (2014); Burkart and Huber (2021); Kotsiantis et al. (2007); it works with isolated data and learns without utilizing prior knowledge. The trained model can only function with the input instances for which identical samples have been used for training purposes.

In Fei and Liu (2016), the authors presented a space learning technique, the Centre-based Similarity (CBS), to determine text in the open world. Each feature in the mean of the positive class and the feature vector of the document are transformed separately by CBS in the document space vector. In Shu et al. (2017), the authors introduced Deep Open Classification (DOC) as a method for identifying unclassified classes that may not be available in existing training classes. The classifier is designed to accurately classify both documents belonging to the known training classes and those that are unknown. This approach is known as open classification. Its multi-layer architecture is based on CNN. In Prakhya et al. (2017), the authors introduced another technique based on convolutional neural networks that incorporates feature extraction methods. It involves converting the document into a vector using the Word2Vec approach to extract features and calculate the cosine similarity between the entire document vector and the computed document vector using a naïve approach.

In Lin and Xu (2019a), the authors presented a softmax-based model to determine the profound novelty and detection of novel instances. It is called Softmax and Deep Novelty (SMDN). It utilises a Softmax and Local Outlier Factor (LOF) approach to identify new instances and can be implemented with various models without changing their architecture. In Vedula et al. (2019), the authors introduced a two-phase mechanism model that predicts the statement’s intent and then tags it in the input statement for open intent detection. The model comprises a Bidirectional Long Short-Term Memory (BiLSTM) and a Conditional Random Field (CRF) that uses adversarial training to enhance its robustness and execution across various domains. It can automatically detect a user’s intent in natural language without requiring prior knowledge. The methodology begins with identifying any pre-existing open intents, which are then labelled with corresponding actions and objectives in the input words. If no actions or objectives are associated with a noticed intent, it is labelled as ”none”.

In Vedula et al. (2020), the authors presented ADVIN, the automatic discovery of novel intents and domains; it can discover novel domains and intents from anonymous data. ADVIN operates in three phases: first, the identification of unknown domains; second, knowledge transfer; and finally, tagging of intents to their affiliated novel domains. It utilises BERT and multi-class classifiers to identify unknown intents. The DOC is used for determining unknown intents with hierarchical clustering to determine unknown classes of intents.

### 2.2 Continual machine learning

To effectively handle the complexities of the natural world, an intelligent system must continuously acquire, revise, retain, and apply knowledge to its existence. This capability, comprehended as continual learning, provides a basis for artificial intelligence systems to design themselves adaptively. The ability to employ continuous learning, known as continual learning, functions as a foundation for artificial intelligence systems to tailor their designs dynamically. Overall, the concept of continuous learning is typically associated with a phenomenon known as catastrophic forgetting, whereby the acquisition of new or distinct knowledge often results in a notable decrease in performance for previously learned tasks. Moreover, various developments have emerged over the last decade that have expanded the understanding and application of continuous learning.

There are numerous approaches to achieving continual learning, such as Representation, Regularisation, Optimisation, and Replay-based approaches.

#### 2.2.1 Representation-based CML

In addition to earlier research on acquiring sparse representations via meta-training Javed and White (2019), present researchers have attempted to integrate the edges of self-supervised learning Gallardo et al. (2021), and large-scale pre-training Mehta et al. (2021); Wu et al. (2022) to enhance the representations in initialisation and CML. These two approaches are nearly coupled since the pre-training data is usually of a massive portion and without detailed labels. At the same time, the execution of self-supervised learning is primarily evaluated through the fine-tuning of downstream tasks. The pre-training needs unsupervised learning or self-supervised learning to process extensive portions of data without detailed labels. The approach is designed to perform self-supervised learning, primarily using the contrastive loss for CML, which is generally based on the idea of contrastive learning. Regarding those self-supervised representations that are crucial to mitigating catastrophic forgetting, in Madaan et al. (2021), the authors presented a Lifelong Unsupervised Mixup that achieves further advancements by interpolating between examples of the old and novel tasks.

In Purushwalkam et al. (2022), the authors presented a Minimum redundancy-based approach that further encourages the variousness of knowledge replay by de-correlating the accumulated previous training examples. In Fini et al. (2022), the authors presented a practical and straightforward framework for Continual Self-Supervised Learning. The framework transforms the self-supervised loss into a compression technique by mapping the current representation phase to its prior phase, which aims to enhance the representations of Self-Supervised Learning models in a continual learning scenario. In Cha et al. (2021), the authors presented Contrastive Continual Learning; this approach utilises a contrastive loss to preserve individual tasks and a self-supervised loss to filter information between the previous and present examples.

#### 2.2.2 Optimization-based CML

Optimisation-based CML is a paradigm for training ML models on sequential data that constantly arrives over time. The model is updated iteratively by utilising stochastic gradient descent (SGD) or similar optimisation algorithms Robbins and Monro (1951); Kiefer and Wolfowitz (1952); Bottou et al. (2018). At each iteration, the model is trained on a tiny subset of the sequential data, and the model’s parameters are updated based on the error between the model’s predictions and the actual weights of the data.

Some optimization methods also use replay-based approaches, such as Gradient episodic memory (GEM), Averaged-GEM, Layer-wise optimization by gradient decomposition, and Maximizing Transfer and Minimizing Interference Lopez-Paz and Ranzato (2017); Chaudhry et al. (2018b); Tang et al. (2021); Riemer et al. (2018). These techniques ensure the prior input and gradient space conservation via previous training examples.

#### 2.2.3 Regularization-based CML

Regularisation-based CML algorithms generally employ two methods: first, regularisation, which enables the model to retain previous learning while learning new tasks, and second, distillation, which utilises prior models as trainers to guide the learning of new models. Weight regularisation is utilised to regulate the variation of network parameters selectively.

In Kirkpatrick et al. (2017)The authors proposed elastic weight consolidation (EWC), a solution to overcome catastrophic forgetting in neural networks. The algorithm decreases the learning rate on specific weights according to their importance in earlier learned tasks. The significance of this method is evaluated in both supervised learning and reinforcement learning contexts, where numerous tasks can be learned sequentially without interfering with prior learning.

In Schwarz et al. (2018), the authors proposed a method that utilises a knowledge base (KB) to train a competent model to solve previously encountered concerns, which is related to an active column used to learn the current task efficiently. After learning a new task, the active column is refined into the KB, protecting any earlier obtained skills. This active learning process, followed by compression, needs no architecture development, access to or accumulation of earlier data, and no specific parameters related to the task. It is the online version of EWC. It is based on the Fisher information matrix(FIM); the FIM is revised recursively without requiring access to the task label. Some other weight regularisation methods are there, such as synaptic intelligence (SI), Memory aware synapses, Riemannian walk for incremental learning, FIM Zenke et al. (2017); Aljundi et al. (2018); Chaudhry et al. (2018a); Benzing (2022)

#### 2.2.4 Replay-based CML

Replay-based CML works by accumulating and replaying earlier data examples or pieces of knowledge to the model during training. The model is trained on both recent data and a subset of previously known data, which prevents catastrophic forgetting of earlier learned knowledge. There are many techniques to execute the replay-based CML, such as Reservoir Sampling Riemer et al. (2018); Chaudhry et al. (2019), which randomly accumulates a specified number of training examples from each input set. Second, Ring Buffer Lopez-Paz and Ranzato (2017) further provides an equivalent number of previous training examples randomly selected per class. Third, Mean-of-Feature Rebuffi et al. (2017) specifies an equivalent number of previous training examples that are most comparable to the characteristic mean of the individual class.

The concept of Online Continual Compression involves the concurrent learning of compression processes and the storage of representative data from a non-independent and identically distributed (non-i.i.d.) data stream. To overcome this problem, in Caccia et al. (2020), the authors suggested adaptive quantisation modules that enhance continual online compression and preserve compact data for replay. Furthermore, the authors propose various methods to maintain prior learning, including supplementary knowledge with class statistics and minimal storage requirements. The methods employed are Class Incremental Learning with Dual Memory, Syntax-Aware Memory Network, Explanations that Reduce Catastrophic Forgetting, and Gradient Projection Memory for Continual Learning Belouadah and Popescu (2019); Gong et al. (2022); Ebrahimi et al. (2020); Saha et al. (2020).

*Table 2: Summarised Literature Surveys*

| Author | OTC | CML | Author | OTC | CML |

$\surd$ $\surd$| Robbins and Monro (1951) | - | | Rebuffi et al. (2017) | - | |

$\surd$ $\surd$| Kiefer and Wolfowitz (1952) | - | | Bottou et al. (2018) | - | |

$\surd$ $\surd$| Fei and Liu (2016) | | - | Chaudhry et al. (2018b) | - | |

$\surd$ $\surd$| Shu et al. (2017) | | - | Riemer et al. (2018) | - | |

$\surd$ $\surd$| Prakhya et al. (2017) | | - | Schwarz et al. (2018) | - | |

$\surd$ $\surd$| Lopez-Paz and Ranzato (2017) | - | | Aljundi et al. (2018) | - | |

$\surd$ $\surd$| Kirkpatrick et al. (2017) | - | | Chaudhry et al. (2018a) | - | |

$\surd$ $\surd$| Zenke et al. (2017) | - | | Riemer et al. (2018) | - | |

$\surd$ $\surd$| Lin and Xu (2019a) | | - | Chaudhry et al. (2019) | - | |

$\surd$ $\surd$| Vedula et al. (2019) | | - | Belouadah and Popescu (2019) | - | |

$\surd$ $\surd$| Vedula et al. (2020) | | - | Caccia et al. (2020) | - | |

$\surd$ $\surd$| Madaan et al. (2021) | - | | Ebrahimi et al. (2020) | - | |

$\surd$ $\surd$| Purushwalkam et al. (2022) | - | | Saha et al. (2020) | - | |

$\surd$ $\surd$| Fini et al. (2022) | - | | Tang et al. (2021) | - | |

$\surd$ $\surd$| Cha et al. (2021) | - | | Benzing (2022) | - | |

$\surd$ $\surd$| Lopez-Paz and Ranzato (2017) | - | | Gong et al. (2022) | - | |

$\surd$ $\surd$ | OpenCML | | | | | |

Abbreviation: OTC: Open-text classification CML: Continual Machine Learning

## 3 Proposed Methodology

To answer $RQ1$, this section presents the functionality and provides a detailed description of OpenCML. The functionality of OpenCML can be broken down into four stages, each with its own set of tasks and functions. The first stage uses BERT (Bidirectional Encoder Representations from Transformers) to preprocess the input data. The second stage identifies unknown instances, which is based on CNN (Section 3.1). It discovers novel instances from test data and stores them in a separate memory. The third stage creates clusters from unknown instances discovered by stage 2, using the balanced iterative reducing and clustering using hierarchies (BIRCH) algorithm. Next, it creates novel classes using the key extraction method and labels these classes of unknown test data instances (Section 3.2). Finally, it learns novel classes incrementally from the second iteration onward. OpenCML utilises cross-distillation loss to prevent forgetting and learn classes incrementally (Section 3.3).

*Figure 1: This figure explains the functioning of the OpenCML Framework. The Custom Loss $L(\theta)$ formulation, used to retain previously learned knowledge, is explained in a red dotted line box. Where $L_{Ds}(\theta)$ is distillation loss applied on old classification layer (on known classes), and $L_{Ce}(\theta)$ is classification loss which is applied on both new and old classification layer (on known and novel classes) *

### 3.1 Open Text Classification

Module one classifies unknown and known data using convolutional neural networks, which consist of three essential components for data classification. The first component of the data processing unit uses embedding for data embedding. Next, it consists of a convolutional network with max-pooling; the third component is the output unit. It is a fully connected layer. This module comprises convolution layers with a middle Rectified Linear Unit (ReLU) function to normalise the convolutional output. Next, the Max-pooling layer is used to decrease the dimensions of the convolution layer’s output. Next, it consists of two fully interconnected layers with ReLU functions in between. The output layer uses a 1-vs-rest sigmoid layer. The description in detail is provided below.

Let $d_{im}$ is world vector (dimensional) $s_{i}\in\mathbb{R}^{d}_{im}$ corresponding to the $i^{th}$ word. Now let us assume length of sentence with padding is $l_{p}$. Now, the concatenation for every word vector is as follows:

$s_{i:l_{p}}=s_{1}\mathbin{\|}s_{2}\mathbin{\|}s_{3}\mathbin{\|},\ldots,s_{l_{p}}$ | | | | (1) |

Concatenation is denote by ”$\mathbin{\|}$”, therefore the general representation of ”$\mathbin{\|}$” well be $s_{i:i+j}$ that is concatenations of $s_{i},\ s_{i}+1,\ldots s_{i+j}$. Now we apply the convolution filters $c\in\mathbb{R}^{w{d_{im}}}$ on word window with size $w$ and generates the feature map $F_{m}$ from the $s_{i:1+w-1}$ word window, that is represented as:

${F_{m}}_{i}=f(c.s_{i:1+w-1}+b)$ | | | | (2) |

Where the non-leaner features represented as $f$ and bias as $b$, that is $b\in\mathbb{R}$. we get the possible word window $s_{1:w},s_{2:w},\ldots s_{{l_{p}}-w+1:l}$. After, applying filters $f$ on wold windows we get the feature map $F_{m}$, that is $F_{m}\in\mathbb{R}^{{l_{p}}-{w+1}}$.

$F_{m}=(F_{1},F_{2},\ldots,F_{{{l_{p}}-w+1}})$ | | | | (3) |

After every convolution there is ReLU activation function to normalise the output of convolution layer. To reduce the $F_{m}$ (feature vector) it uses $1D$-max pool after convolution layer. It takes maximum value of $\hat{F_{m}}$ form convolution layer that is $\hat{F_{m}}=Max(F_{m})$

To segregate the unknown instances from testing data, it uses the 1-vs-rest method. It can classify unknown samples using a 1-vs-rest layer. The conventional functions, such as softmax and other similar ones, are proven effective for multi-class classification, but they cannot reject unknown data instances.

Here, we used $S_{cn}$ (number of known classes ) sigmoid in the last layer. The $i^{th}$ sigmoid function for $\alpha_{i}$ classes. Now model assigned classes as positive classes if $\beta=\alpha_{i}$ and classes denoted as negative for all $\beta\neq\alpha_{i}$.

$\hat{\beta}=\begin{Bmatrix}reject,\ ifsigmoid(\kappa_{i})<\gamma_{i},\ \forall\ \alpha_{i}\ in\ \beta_{i}\\
argmax_{\alpha_{i}\in\beta}\ sigmoid(\kappa_{i}),\ Otherwise\end{Bmatrix}$ | | | | (4) |

Where $I$ is the Indicator function, j= 1 to $I_{c}$ (Number of instances) and probability output of $S_{cn}$ sigmoid for $j^{th}$ input of $i^{th}$ dimension of $\kappa$ is $p(\beta_{j}=\alpha_{i})=sigmoid(\kappa_{j,i})$

### 3.2 Discovery and Identification of Novel Classes

To make classes from unknown data, we applied the Balanced Iterative Reducing and Clustering using Hierarchies (BIRCH) algorithm. We have $N_{d}$ data points with $d$ dimensions, each represented as a vector. Now, it calculates the cluster features $C_{f}$ for all $C_{j}={x_{i}\dots x_{N_{d}}}$. To calculate the cluster feature, $C_{f}$ consists of three parameters: $N_{d}$, $L_{s}$, and $S_{s}$. Where $N_{d}$ is the number of data points with d-dimensions, $L_{s}$ is the linear sum, and $S_{s}$ is the squared sum. The $C_{f}$, $L_{s}$, and $S_{s}$ can be calculated as:

$C_{f}=\left(N_{d},\vec{L_{s}},S_{s}\right)$ | | | | (5) |

$\vec{L_{s}}=\sum_{i=1}^{N_{d}}\vec{x_{i}}$ | | | | (6) |

$S_{s}=\sum_{i=1}^{N_{d}}(\vec{x_{i}})^{2}$ | | | | (7) |

The cluster features are systematised in the tree, called a feature tree; it is a height-balanced tree. It has two parameters $\beta_{f}$ (branching factor) and $T_{h}$ (threshold).

Every non-lead node holds at most $\beta_{f}$, from $[C_{f_{i}},Ch_{i}]$, where $Ch_{i}$ is a child node, and it is a pointer to its $i^{th}$ child node and $C_{f_{i}}$ defines the associated sub-cluster. A leaf node contains at most $L$ accesses individually of the form $C_{f_{i}}$. Two pointers, previous and next, are used to chain all leaf nodes. The height of the tree relies on the threshold $T_{h}$. Next, the algorithm reviews all the leaf accesses in the initial $C_{f}$ tree to reconstruct a smaller $C_{f}$ tree while extracting outliers and setting dense sub-clusters into bigger ones. For all leaf entries, an agglomerative hierarchical clustering algorithm is used directly to define sub-clusters based on their $C_{f}$ vectors.

To avoid minor and localised inaccuracies, we redistribute the data points to their closest seeds (calculate the centroid), which provides a new set of clusters. The data points which are distant from the seeds are considered outliers. The centroid can be computed as:

$Centroid(C)=\frac{\sum_{i=1}^{N_{d}}\vec{x_{i}}}{N_{d}}=\frac{\vec{L_{s}}}{N_{d}}$ | | | | (8) |

### 3.3 Continual Machine Learning

To continually acquire knowledge, OpenCML employed the cross-distillation loss method Castro et al. (2018). OpenCML utilises exemplar memory to store samples from previously learned classes (section 3.3.1). For the subsequent output, it constructs training data (section 3.3.2). In the next iteration, when new classes are added to the input data, it applies cross-distillation loss, distillation loss on the classification layer (for old classes), and multi-class cross-entropy loss for all classification layers (section 3.3.3). Next, it again updates the exemplar memory (section 3.3.4).

#### 3.3.1 Exemplar Memory

To answer $RQ2$, the proposed approach utilises techniques that efficiently manage memory for storing and managing previous knowledge. We tackle the challenge of class incremental learning by first classifying both known and unknown instances from the data and creating novel classes from the unknown data. To further enhance learning efficiency, we selectively store a subset of the most representative examples from known classes for future use. We employ a memory module with a restricted capacity of $K$ examples to achieve this, ensuring we are conscious of the memory limitations. As more class instances are accumulated, the number of examples per class decreases. The number of examples per class, denoted by $n$, is determined by $n=[K/c]$, where c represents the number of classes stored in memory, and K represents memory capacity. This careful selection process of memory enables us to retain diverse examples from known classes while still being mindful of memory limitations. This makes our approach efficient and effective for incremental learning tasks in classes. We employed the herding technique for sample selection in our approach.

Herding Welling (2009) is a powerful method for selecting exemplars in class incremental learning tasks. This technique involves iteratively selecting the sample closest to the current centroid of the selected samples. In other words, the aim is to find a set of K exemplars $Y={y_{1},\dots,y_{K}}$ that maximise the sum of cosine similarity between the exemplars and the mean vector of the selected samples. This similarity score is computed using the cosine similarity function $cos(y_{j},x_{i})$, where $y_{j}$ is an exemplar and $x_{i}$ is a sample. The mean vector of the chosen samples up to the $i^{th}$ iteration is computed as $\frac{1}{i}\sum_{k=1}^{i}x_{k}$. Herding is highly efficient and effective for selecting representative exemplars in class incremental learning tasks.

#### 3.3.2 Construction of the Training Data

The Input for the second iteration onward was created by integrating the Exemplar memory and new data (Input). When new Input is shown, the system integrates the stored exemplars with the new Input to generate a representation, which is then compared to the stored exemplars to identify the closest match. This process enables the accurate classification of new instances, as the variability and context-specific features are essential. By integrating exemplar memory and new input data, the system can identify new instances of a class based on their resemblance to prior experiences, resulting in greater flexibility and accuracy in classification.

#### 3.3.3 Custom Loss

To answer $RQ3$ and $RQ-4$, we used the custom loss Castro et al. (2018), which was created by combining the distillation loss Hinton et al. (2015) and cross-entropy loss. The distillation loss retrieves the knowledge from prior learned classes, and the cross-entropy loss learns to classify the recent classes. We employ cross-entropy loss for all classification layers, whereas distillation loss is only applied to the classification layers of the older classes. This enables the model to adjust its decision boundaries and improve its performance. By integrating these two loss functions, our method offers a more effective and efficient solution for training deep neural networks on classification tasks. The custom loss function $L(\theta)$ can be defined as:

$L(\theta)=L_{Ce}(\theta)+\sum_{Cl=1}^{co}L_{{Ds}_{cl}}(\theta),$ | | | | (9) |

Where $L_{Ce}(\theta)$ is cross-entropy loss, which is applied on all the classes (old + new), $L_{{Ds}_{cl}}$ is the distillation loss function of classification layer $cl$, and $co$ number of classification layers for the old classes. The $L_{Ce}(\theta)$ can be defined as:

$L_{Ce}(\theta)=-\frac{1}{I_{c}}\sum_{i=1}^{I_{c}}\sum_{j=1}^{C}g_{ij}\log s_{ij}$ | | | | (10) |

Where $s_{i}$ is a score acquired by using a sigmoid function on the logits of a classification layer, for instance, $i$, $g_{i}$ is the ground truth, $i$, and $I_{c}$ and $C$ denote the number of instances and classes, respectively. The distillation loss $L_{Ds}(\theta)$ can be defined as:

$L_{Ds}(\theta)=-\frac{1}{I_{c}}\sum_{i=1}^{I_{c}}\sum_{j=1}^{C}gdst_{ij}\log qsdst_{ij},$ | | | | (11) |

where, $gdst_{i}$ and $sdst_{i}$ are revised interpretations of $g_{i}$ and $s_{i}$, respectively. They are acquired by increasing $g_{i}$ and $s_{i}$ to the exponent $1/T$, as illustrated in Hinton et al. (2015), where $T$ is the distillation parameter.

#### 3.3.4 Updating the Exemplar Memory

As memory is limited, we update the memory to incorporate samples from new classes after training has occurred. We performed this step after training, which concerns extracting examples from the end of the sample set of each class. Since the examples are stored in a sorted list, this process takes little effort and can be performed easily. It should be noted that the samples that are removed during this step are never used again. This approach can optimise memory usage in the proposed framework and encourage the inclusion of novel classes in the model.

### 3.4 Use case Scenario

*Figure 2: Use Case Scenario*

To illustrate the methodology used in our research, let us consider an example of four distinct classes. However, only the classes $A$, $B$, $C$, and $D$ were initially known, while $E$ and $F$ remained unidentified. Initially, our model performed class segregation by distinguishing between known and unknown classes. This segregation allowed us to classify the classes $A,B,C$ and $D$ as known, while E and F were classified as unknown. Furthermore, we implemented a memory mechanism to store a fixed-size set of exemplars in the future. This memory served as a repository to retain valuable information from the initial process, ensuring its availability for subsequent classifications. As our model progressed, it could identify additional novel classes, specifically classes $E$ and $F$, which were previously unknown. Subsequently, our model incorporated new data alongside the exemplars stored in memory. Additionally, it considered all six classes, $A,B,C,D,E$ and $F$, to refine its learning and classification capabilities. By incorporating these newly discovered classes, our model evolved and adapted to perform the overall classification task more effectively. This iterative process, involving the integration of new data, the utilisation of exemplars from memory, and the continuous refinement of class identification, forms a crucial aspect of our research methodology. It enables the model to incrementally enhance performance and expand its understanding of the underlying data.

## 4 Experiments and Results

### 4.1 Experimental setup

The proposed work’s experiments were conducted on a computer equipped with an Intel Core i5-2410 M CPU and 8 GB of DDR3 RAM. The computer operated on a 64-bit version of Windows 10 with a 64-bit processor. All the experiments were implemented using Python 3.0. To classify known and unknown classes, the threshold value $t_{h}=0.5$ is used for open text classification. We use Adam as the optimiser, and the weight decay for all parameters is set to 0.01. Each training step consists of 50 epochs with a learning rate of 0.02. Due to computational limitations, we have fixed four memory sizes, ranging from 250 to 1500, for storing examples. We conducted all experiments with up to four iterations.

### 4.2 Datasets

The evaluation was conducted using the text datasets that are publicly accessible, as listed below.

-

BANKING77 (DS-1): A dataset discussed by, is explicitly designed with banking sector domains. It covers a wide range of 77 distinct categories, serving as a valuable resource for classification tasks. The dataset is divided into three segments: the training dataset, encompassing 10,003 utterances; the validation dataset, containing 1,000 utterances; and the test dataset, which holds 3,080 utterances Casanueva et al. (2020).

-

CLINC150 (DS-2) : A dataset is specifically created for out-of-domain (OOD) detection tasks. It encompasses a diverse collection of 150 distinct classes drawn from 10 different domains. Within this dataset, there are 22,500 in-domain (IND) utterances representing the primary domain and an additional 1,200 out-of-domain (OOD) utterances, which are instances intended to challenge and evaluate the model’s ability to detect OOD samples. Larson et al. (2019).

-

StackOverflow (DS-3): A dataset, originally introduced by Xu et al. (2015), this dataset is designed around 20 distinct categories and is thoughtfully segregated into three subsets to facilitate comprehensive model training and assessment. The training dataset comprises a substantial 12,000 data points, while the validation dataset, consisting of 2,000 data points, serves as a means for fine-tuning the model. Lastly, the test dataset includes 6,000 data points.

-

DBPedia Classes Dataset (DS-4) (DS-4): This dataset includes more than 300k hierarchically labeled Wikipedia reports. It has three groups with 9, 70, and 219 classes Auer et al. (2007).

### 4.3 Performance metric

To evaluate the discovery of unknown classes, we employed several performance metrics, including Accuracy (Acc), F1-score, Matthews Correlation Coefficient (MCC), G-mean1 (GM1), and G-mean2 (GM2). Subsequently, to assess the effectiveness of labeling and keyword extraction for novel classes, we utilized metrics such as Accuracy (Acc), Adjusted Rand Score (ARS), Normalized Mutual Information (NMI), Fowlkes-Mallows Score (FMS), and F1-score.

Assuming standard terminology, True Positive ($T_{p_{os}}$), False Positive ($F_{p_{os}}$), True Negative ($T_{n_{eg}}$), False Negative ($F_{n_{eg}}$), Precision ($P_{re}$), Recall ($R_{e}$), & Specificity ($S_{pe}$). The formulas of performance matrices are as follows.

*Table 3: Performance metrics *

| Parameter | Formula |

$\frac{T_{p_{os}}+T_{n_{eg}}}{T_{p_{os}}+T_{n_{eg}}+F_{p_{os}}+F_{n_{eg}}}$| Accuracy | = |

$2\times\frac{P_{re}\times R_{e}}{P_{re}+R_{e}}$| F1-score | = |

$\frac{(T_{p_{os}}\times T_{n_{eg}})-(F_{p_{os}}\times F_{n_{eg}})}{\sqrt{(T_{p_{os}}+F_{p_{os}})(T_{p_{os}}+f_{n_{eg}})(T_{n_{eg}}+F_{p_{os}})(T_{n_{eg}}+F_{n_{eg}})}}$| MCC | = |

$\sqrt{P_{re}\times R_{e}}$| G-mean 1 (GM1) | = |

$\sqrt{R_{e}\times S_{pe}}$| G-mean 2 (GM2) | = |

$\frac{(RI-ExpectedRI)}{(max(RI)-ExpectedRI)}$| ARS | = |

$NMI(g_{i},g_{j}^{\prime\prime})$$\sum_{i=1}^{|g_{i}|}\sum_{j=1}^{|g_{j}^{\prime\prime}|}\frac{|g_{i}\cap g{}^{\prime\prime}_{j}|}{N}\log\frac{N|g_{i}\cap g{}^{\prime\prime}_{j}|}{|g_{i}||g{}^{\prime\prime}_{j}|}$| NMI | = = |

$\frac{{T_{p_{os}}}}{\sqrt{(T_{p_{os}}+F_{p_{os}})*(T_{p_{os}}+F_{n_{eg}})}}$| FMS | = |

### 4.4 Performance Analysis

The performance analysis conducted in this study encompasses three main aspects. Firstly, we examine the performance of class increment learning, which involves evaluating the accuracy across multiple rounds for each dataset. This analysis provides insights into the model’s ability to learn new classes and adapt to evolving data incrementally. The results show variations in performance across the datasets. Second, we analyse the model’s performance in the context of open-text classification during class incremental learning. This evaluation provides valuable insights into how the model performs open text classification tasks, particularly in the context of incremental learning. Considering these two aspects, we gain a comprehensive understanding of the model’s overall performance and capabilities in class increment learning and open-text classification. Next, we conduct an ablation study.

#### 4.4.1 Class incremental learning

The performance was measured using accuracy metrics: Accuracy of each iteration and Average accuracy. The experiment results are shown in Table 4.

The analysis of incremental accuracy with varying memory bucket sizes (k) reveals significant insights into the model’s performance across different iterations (I) of incremental learning. Specifically, the data shows that as the memory bucket size increases from k=250 to k=1500, the average incremental accuracy increases consistently. For DS-1, at k=250, the model acquires an average accuracy of 65.87%, which increases to 68.34% at k=500, 70.49% at k=1000, and 71.99% at k=1500. Notably, the incremental accuracy at the initial iteration (I-1, 14 classes) remains constant at 89.589 across all memory bucket sizes, indicating that the initial learning performance is unaffected by memory size. However, in succeeding iterations (I-2 to I-4), the model’s accuracy declines as the number of classes increases, with a more prominent drop observed at smaller memory bucket sizes. For example, at k=250, the accuracy drops from 71.185 (I-2) to 42.453 (I-4), whereas at k=1500, it drops from 76.485 (I-2) to 54.748 (I-4).

For DS-2, there are four incremental Iterations, I-1 to I-4, corresponding to an increasing number of classes (4, 6, 8, and 1-1 classes, respectively). For all memory bucket sizes, the model maintains a consistent accuracy of 91.898% at the initial stage (I-1, four classes), indicating that the initial learning performance is independent of the memory size. With k=250, the accuracy drops from 80.785% (I-2) to 60.254% (I-4). In contrast, at k=1500, the accuracy decreases from 84.999% (I-2) to 65.789The average incremental accuracy shows a positive correlation with the size of the memory bucket. Specifically, the average accuracy improves from 75.847% (k=250) to 77.104% (k=500), 78.308% (k=1000), and 79.606% (k=1500).

The DS-3 is evaluated across four incremental iterations, I-1 to I-4, corresponding to an increasing number of classes (3, 5, 7, and 9 classes, respectively). The memory bucket sizes under consideration are k=250, k=500, k=1000, and k=1500. For each memory bucket size, the model’s incremental accuracy at the initial stage (I-1, three classes) remains constant at 89.987%, indicating that the initial learning performance is not affected by the memory size, with k=250, the accuracy drops from 82.754% (I-2) to 66.742% (I-4), whereas with k=100, the accuracy decreases from 89.168% (I-2) to 71.265% (I-4). The average incremental accuracy shows a positive correlation with the size of the memory bucket. Specifically, the average accuracy increases from 77.985% (k=250) to 79.217% (k=500), 80.178% (k=1000), and 82.541% (k=1500).

The DS-4 is evaluated over four incremental Iterations, I-1 to I-4, which correspond to an increasing number of classes (5, 7, 9, and 11 classes, respectively). The memory bucket sizes considered are k=250, k=500, k=1000, and k=1500. At the initial stage (I-1, five classes), the model consistently executes an incremental accuracy of 90.956% across all memory bucket sizes, indicating that the initial performance is unaffected by the size of the memory bucket, with k=250, the accuracy drops from 72.58% (I-2) to 51.316% (I-4), whereas with k=1500, the accuracy decreases from 81.889% (I-2) to 59.124% (I-4). The average incremental accuracy demonstrates a positive correlation with the size of the memory bucket. Specifically, the average accuracy improves from 68.828% (k=250) to 71.593% (k=500), 73.019% (k=1000), and 74.810% (k=1500).To the best of our knowledge, there is no work available that contains both open-text classification and continual learning. Therefore, we present a summarized evaluation in Table 7. Table 7 clearly states that only a few existing works are available in this domain for natural language processing (NLP) in continual machine learning. Moreover, based on our extensive review and knowledge in the field, we have yet to find research that comprehensively addresses the combined challenges of open-text classification and continual machine learning. This research gap underscores the need for novel approaches to tackle these challenges.

*Table 4: Performance analysis of OpenCML with four benchmark datasets. It shows the average results of four rounds (Iterations, that is indicated by I, $I_{1}$ to $I_{3}$) with M@250 to M@1500, where M= exemplar memory*

| DS-1 |

| Memory |

| Bucket |

| I-1 |

| (14-classes) |

| I-2 |

| (18-classes) |

| I-3 |

| (19-classes) |

| I-4 |

| (20-classes) |

| Avg. Incremental |

| Accuracy |

k=250 89.589 71.185 60.248 42.4527 65.868

k=500 89.589 73.145 61.164 49.475 68.343

k=1000 89.589 75.963 64.968 51.427 70.488

k=1500 89.589 76.485 67.154 54.748 71.994

DS-2

| Memory |

| Bucket |

| I-1 |

| (4-classes) |

| I-2 |

| (6-classes) |

| I-3 |

| (8-classes) |

| I-4 |

| (11-classes) |

| Avg. Incremental |

| Accuracy |

k=250 91.898 80.785 70.452 60.254 75.847

k=500 91.898 81.988 72.784 61.745 77.103

k=1000 91.898 83.745 73.965 63.625 78.308

k=1500 91.898 84.999 75.737 65.789 79.606

DS-3

| Memory |

| Bucket |

| I-1 |

| (3-classes) |

| I-2 |

| (5-classes) |

| I-3 |

| (7-classes) |

| I-4 |

| (9-classes) |

| Avg. Incremental |

| Accuracy |

k=250 89.987 82.754 72.457 66.742 77.985

k=500 89.987 83.412 75.634 67.835 79.217

k=1000 89.987 86.879 74.498 69.348 80.178

k=1500 89.987 89.168 79.743 71.265 82.541

DS-4

| Memory |

| Bucket |

| I-1 |

| (5-classes) |

| I-2 |

| (7-classes) |

| I-3 |

| (9-classes) |

| I-4 |

| (11-classes) |

| Avg. Incremental |

| Accuracy |

k=250 90.956 72.58 60.458 51.316 68.827

k=500 90.956 77.826 62.625 54.963 71.592

k=1000 90.956 79.245 65.731 56.145 73.019

k=1500 90.956 81.889 67.269 59.124 74.809

*Table 5: Incremental Open Classification Performance Analysis of Proposed Model*

| | | Banking77 | | | CLINC 150 | | | StackOverFlow | | | DBPedia Classes | | |

| | | ACC-ALL | F1-OOD | F1-KNOW | ACC-ALL | F1-OOD | F1-KNOW | ACC-ALL | F1-OOD | F1-KNOW | ACC-ALL | F1-OOD | F1-KNOW |

| | MSP | 26.72 | 9.03 | 50.09 | 30.54 | 24.25 | 39.55 | 42.45 | 44.43 | 42.95 | 44.89 | 46.76 | 45.7 |

| | DOC | 29.86 | 14.92 | 45.62 | 45.56 | 48.81 | 47 | 26.93 | 8.89 | 44.16 | 29.34 | 6.95 | 40.89 |

| | OpenMax | 81.69 | 87.11 | 72.72 | 89.79 | 93.42 | 79.69 | 91.53 | 94.41 | 83.18 | 88.28 | 91.43 | 85.92 |

| | Softmax | 79.42 | 84.87 | 72.06 | 89.78 | 93.34 | 81.74 | 89.76 | 93.6 | 73.62 | 92.74 | 90.32 | 75.89 |

| 25% | LMCL | 29.06 | 13.83 | 50.66 | 32.38 | 27.45 | 41.05 | 58 | 64.81 | 51.8 | 61.36 | 61.05 | 53.97 |

| | SEG | 33.38 | 22.14 | 48.67 | 50.67 | 55.8 | 49.54 | 27.23 | 9.66 | 47.61 | 30.29 | 11.79 | 45.23 |

| | ADB | 85.32 | 89.8 | 78.06 | 90.61 | 93.95 | 81.36 | 93.2 | 95.58 | 85.3 | 96.17 | 98.24 | 88.79 |

| | KNN | 87.41 | 91.52 | 77.7 | 92.71 | 95.42 | 83.81 | 92.04 | 94.76 | 84.29 | 89.61 | 92.22 | 86.32 |

| I-1 | Ours | 89.73 | 78.18 | 93.85 | 85.69 | 91.45 | 62.22 | 95.47 | 83.58 | 95.86 | 93.18 | 81.22 | 92.14 |

| I-2 | Ours | 91.65 | 78.15 | 93.8 | 77.56 | 55.45 | 86.88 | 81.56 | 58.91 | 92.38 | 84.42 | 55.6 | 89.24 |

| I-3 | Ours | 92.89 | 79.68 | 93.98 | 88.69 | 72.59 | 92.56 | 87.28 | 58.67 | 92 | 83.69 | 60.77 | 94.89 |

| I-4 | Ours | 91.37 | 80.67 | 94.85 | 90.86 | 77.68 | 94.86 | 86.27 | 63.89 | 92.67 | 89.88 | 60.77 | 90.09 |

| | | | | | | | | | | | | | |

| | MSP | 47.89 | 4.37 | 68.49 | 42.51 | 12.89 | 62.36 | 60 | 52.92 | 68.87 | 57.99 | 49.53 | 66.13 |

| | DOC | 49.98 | 11.82 | 68.26 | 55.44 | 43.8 | 67.19 | 47.92 | 7.93 | 65.49 | 50.5 | 5.39 | 68.51 |

| | OpenMax | 80.9 | 81.32 | 81.79 | 88.61 | 90.62 | 86.52 | 88.52 | 89.57 | 87.13 | 90.74 | 92.75 | 83.97 |

| | Softmax | 80.32 | 80.57 | 81.5 | 87.91 | 89.71 | 87.03 | 83.47 | 85.48 | 80.31 | 86.7 | 87.93 | 77.63 |

| 50% | LMCL | 50.45 | 12.63 | 69.56 | 46.53 | 24.26 | 63.6 | 63.25 | 58.03 | 72.9 | 60.59 | 60.38 | 75.19 |

| | SEG | 50.61 | 12.33 | 70 | 58.19 | 49.04 | 68.72 | 49.13 | 12.11 | 67.56 | 46.02 | 15.75 | 63.85 |

| | ADB | 81.86 | 81.51 | 83.9 | 89.5 | 91.4 | 87.49 | 89.45 | 90.46 | 88.47 | 86.29 | 93.17 | 92.4 |

| | KNN | 81.98 | 81.65 | 84.03 | 89.96 | 91.72 | 88.15 | 88.92 | 89.69 | 88.09 | 92.46 | 86.59 | 90.95 |

| I-1 | Ours | 84.98 | 77.92 | 89.57 | 88.78 | 82.98 | 91.78 | 92.67 | 88.3 | 94.89 | 96.3 | 90.6 | 98.2 |

| I-2 | Ours | 87.78 | 84.4 | 93.59 | 82.11 | 76 | 85.8 | 78.77 | 60.67 | 86.58 | 82.12 | 58.3 | 90.11 |

| I-3 | Ours | 92.73 | 87.79 | 94.97 | 89.62 | 84.78 | 92.28 | 77.24 | 59.8 | 85.78 | 79.42 | 62.11 | 88.38 |

| I-4 | Ours | 91.81 | 89.47 | 94.78 | 89.89 | 84.12 | 91.98 | 84.96 | 76.56 | 90.39 | 88.61 | 74.11 | 88.38 |

| | | | | | | | | | | | | | |

| | MSP | 72.31 | 13.32 | 83.53 | 57.82 | 7.47 | 75.24 | 69.58 | 16.52 | 80.86 | 23.89 | 6.14 | 53.62 |

| | DOC | 71.48 | 7.97 | 82.92 | 69.54 | 47.04 | 80.29 | 68.81 | 5.39 | 80 | 26.84 | 16.99 | 42.11 |

| | OpenMax | 82.79 | 71.95 | 87.2 | 87.7 | 85.86 | 89.33 | 83.75 | 75.21 | 87.66 | 85.51 | 84.34 | 70.39 |

| | Softmax | 82.06 | 65.6 | 87.68 | 88.59 | 87.18 | 89.68 | 83.44 | 73.62 | 87.53 | 82.03 | 81.44 | 69.06 |

| 75% | LMCL | 73.64 | 19.58 | 84.61 | 59.7 | 13.06 | 76.19 | 71.07 | 20.09 | 81.61 | 26.74 | 11.34 | 54.46 |

| | SEG | 72.23 | 10.97 | 83.68 | 71.6 | 52.2 | 81.51 | 69.51 | 8.24 | 81.23 | 36.64 | 19.96 | 44.78 |

| | ADB | 83.3 | 69.03 | 88.09 | 88.59 | 86.45 | 90.6 | 84.53 | 73.75 | 88.35 | 88.03 | 93.16 | 80.51 |

| | KNN | 84.47 | 72.64 | 88.66 | 89.88 | 88.21 | 91.41 | 85 | 75.76 | 88.71 | 91.01 | 89 | 74.08 |

| I-1 | Ours | 81.58 | 83.9 | 87.48 | 85.29 | 80.77 | 86.79 | 93.78 | 94.78 | 94.6 | 93.4 | 74.52 | 91.32 |

| I-2 | Ours | 88.85 | 85.7 | 90.67 | 84.33 | 81.2 | 85.57 | 70.87 | 59.79 | 79.87 | 95.14 | 81.92 | 97.69 |

| I-3 | Ours | 92.25 | 91.47 | 93.87 | 86.57 | 84.9 | 84.2 | 71.58 | 59.85 | 80.77 | 96.57 | 77.66 | 96.84 |

| I-4 | Ours | 93.38 | 92.28 | 96.91 | 88.65 | 86.25 | 89.44 | 61.84 | 69.58 | 83.37 | 94.45 | 83.55 | 98.31 |

#### 4.4.2 Open-text classification

In our study, we conducted experiments with our model at three different levels of openness: 25%, 50%, and 75%. Our preliminary research assessed our model’s performance in open-text classification scenarios. To provide a comprehensive evaluation and benchmark the effectiveness of our model, we compared it with several methods widely used in open-text classification, such as; MSP Hendrycks and Gimpel (2016), DOC Shu et al. (2017), OpenMax Bendale and Boult (2016), Softmax Zhang et al. (2021), LMCL Lin and Xu (2019b), SEG Yan et al. (2020), ADB Zhang et al. (2021), SCL Zeng et al. (2021) SCL with Gaussian discriminant analysis (GDA) and SCL with Local Outlier Factor (LOF) these are proposed in KNN Zhou et al. (2022).

The objective of comparing our model with these existing approaches is to understand how it performs compared to established techniques and to identify any potential improvements or advantages it may offer. The existing methods selected for comparison are recognised for their applicability in open-text classification tasks, serving as valuable reference points for evaluating the capabilities of our model. These comparisons enabled us to gain meaningful insights into the strengths and weaknesses of our model in various scenarios of openness. By comparing our results with existing methods, we can evaluate whether our model’s overall performance is improved despite introducing new data in each iteration. The choice of openness levels (25%, 50%, and 75%) reflects the varying degrees of openness typically encountered in real-world text classification problems. This multilevel assessment allowed us to evaluate the adaptability and robustness of our model across a spectrum of open-text classification challenges, ranging from relatively constrained to highly open scenarios.

Table 5 Show the incremental open classification performance analysis of the proposed model with Banking77, CLINC 150, StackOverFlow, and DBPedia classes.

Three Levels of Openness: 25%, 50%, and 75%: The analysis is carried out at different levels of openness. In this context, openness probably refers to the proportion of novel or previously unseen data introduced in each analysis. Initially, the openness is 25%; that is, 25% of the data introduced to the model is unknown and is not part of the training data. This increases to 50% and 75% openness in subsequent iterations.

Incremental Open Classification Performance Analysis: This analysis involves gradually introducing more challenging open data into the model and observing how it performs as the level of openness increases. Open classification typically refers to a scenario in which the model needs to classify known or familiar classes (in-distribution data) and detect and handle previously unseen or novel classes (out-of-distribution data). This incremental approach helps assess the model’s ability to adapt and generalize to new and unexpected information.

The finding is that the model’s performance improves despite the inclusion of new data after every iteration. It indicates that the model is learning to adapt to new and unexpected data, a critical capability in many real-world applications where data are dynamic and ever-changing. The incremental open classification performance analysis demonstrates that the proposed model is becoming more robust and effective at handling new data and increasing openness to data. This ability to adapt and improve its performance despite encountering novel data is a positive sign of the model’s versatility and generalisation capabilities.

#### 4.4.3 Ablation Study

A study on the ablation of clustering and keyword extraction has been conducted to validate the used techniques. The detailed study and evaluation are available in the supplementary file.

#### 4.4.4 performance analysis for clustering technique

We evaluated the effectiveness of our clustering method by comparing it with the well-known k-means clustering algorithm. We compared the cluster sizes for K = 2, K = 3, and K = 4. Our evaluation metrics included Completeness, Homogeneity, and v-measure. Completeness measures the fraction of data points correctly assigned to the same cluster by the clustering algorithm and accurate labelling. Homogeneity measures the fraction of data points that belong to the same cluster in both the clustering result and precise labelling. V-measure is the harmonic mean of Completeness and Homogeneity, providing an overall measure of how well the clustering result matches the actual labelling.

With the CLINC-150 dataset, the analysis reveals that BIRCH outperforms K-means in terms of Homogeneity, Completeness, and V-measure for all cluster sizes. For K = 2, K-means has a higher homogeneity score of 0.773 and a lower completeness score of 0.636, while BIRCH has a lower homogeneity score of 0.859 and a slightly higher completeness score of 0.651. The V-measure score for K-means is 0.528, while the V-measure score for BIRCH is 0.562. For K=3, K-means has a higher homogeneity score of 0.810 and a lower completeness score of 0.491, while BIRCH has a slightly higher homogeneity score of 0.842 and a higher completeness score of 0.718. The V-measure score for K-means is 0.555, while the V-measure score for BIRCH is 0.596. For K = 4, K-means has a lower homogeneity score of 0.793 and a higher completeness score of 0.729, whereas BIRCH has a higher homogeneity score of 0.845 and a slightly higher completeness score of 0.751. The V-measure score for K-means is 0.614, while the V-measure score for BIRCH is 0.633.

With the SNIPS dataset, for K = 2, the K-means algorithm achieves a homogeneity score of 0.666 and a completeness score of 0.630. In contrast, the BIRCH algorithm obtains a homogeneity score of 0.730 and a completeness score of 0.695. For the V-measure, the K-means algorithm yields a score of 0.422, whereas the BIRCH algorithm achieves a score of 0.520. For K=3, the K-means algorithm has a homogeneity score of 0.491, completeness score of 0.512, and V-measure score of 0.554, while the BIRCH algorithm has a homogeneity score of 0.543, completeness score of 0.636, and V-measure score of 0.583. For K=4, the K-means algorithm achieves a homogeneity score of 0.760, a completeness score of 0.788, and a V-measure score of 0.616. In contrast, the BIRCH algorithm obtains a homogeneity score of 0.789, a completeness score of 0.807, and a V-measure score of 0.668. Overall, BRICH outperforms.

With GOOGLE SNIPPETS. For K=2, K-means has a homogeneity score of 0.367 and a completeness score of 0.530, while BIRCH has a homogeneity score of 0.516 and a completeness score of 0.736. The V-measure score for K-means is 0.521, while the V-measure score for BIRCH is 0.642. For K=3, K-means has a higher homogeneity score of 0.591 and a completeness score of 0.651, while BIRCH has a lower homogeneity score of 0.658 and a higher completeness score of 0.765. The V-measure score for K-means is 0.694, while the V-measure score for BIRCH is 0.673. For K=4, K-means has a higher homogeneity score of 0.687 and a lower completeness score of 0.602, while BIRCH has a lower homogeneity score of 0.698 and a higher completeness score of 0.800. The V-measure score for K-means is 0.737, while the V-measure score for BIRCH is 0.891.

With DBPedia, we observed that for K = 2, K-means has a higher homogeneity score of 0.627 and a lower completeness score of 0.530. In contrast, BIRCH has a lower homogeneity score of 0.681 and a higher completeness score of 0.695. The V-measure score for K-means is 0.521, while the V-measure score for BIRCH is 0.620. For K = 3, K-means has a lower homogeneity score of 0.629 and a higher completeness score of 0.637. BIRCH has a slightly higher homogeneity score of 0.641 and a marginally higher completeness score of 0.664. The V-measure score for K-means is 0.678, while the V-measure score for BIRCH is 0.765. For K=4, K-means has a higher homogeneity score of 0.743 and a slightly lower completeness score of 0.754, while BIRCH has a lower homogeneity score of 0.784 and a higher completeness score of 0.805. The V-measure score for K-means is 0.789, while the V-measure score for BIRCH is 0.855.

Based on the clustering performance evaluation using the homogeneity, completeness, and V-measure metrics for different values of K (2, 3, and 4), it can be concluded that the BIRCH algorithm outperforms the K-means algorithm for the given experiment. The results demonstrate that BIRCH achieves higher homogeneity, completeness, and V-measure scores across all values of K. Therefore, based on these findings, using the BIRCH clustering algorithm for this particular experiment is recommended. These results offer valuable insights into selecting the most suitable clustering algorithm for similar experiments in the future.

*Table 6: Performance analysis of clustering with K=2, K=3, and K=4. Where K is number of clusters.*

| DS-1 |

| | K=2 | K=3 | K=4 |

| | K-means | BIRCH | K-means | BIRCH | K-means | BIRCH |

| Homogeneity | 0.773 | 0.859 | 0.810 | 0.842 | 0.793 | 0.845 |

| Completeness | 0.636 | 0.651 | 0.491 | 0.718 | 0.729 | 0.751 |

| V-Measure | 0.528 | 0.562 | 0.555 | 0.596 | 0.614 | 0.633 |

| DS-2 |

| Homogeneity | 0.666 | 0.730 | 0.491 | 0.543 | 0.760 | 0.789 |

| Completeness | 0.630 | 0.695 | 0.512 | 0.636 | 0.788 | 0.807 |

| V-Measure | 0.422 | 0.520 | 0.554 | 0.583 | 0.616 | 0.668 |

| DS-3 |

| Homogeneity | 0.367 | 0.516 | 0.591 | 0.658 | 0.687 | 0.698 |

| Completeness | 0.530 | 0.736 | 0.651 | 0.765 | 0.602 | 0.800 |

| V-Measure | 0.521 | 0.642 | 0.694 | 0.673 | 0.737 | 0.891 |

| DS-4 |

| Homogeneity | 0.627 | 0.681 | 0.629 | 0.641 | 0.743 | 0.784 |

| Completeness | 0.530 | 0.695 | 0.637 | 0.664 | 0.754 | 0.805 |

| V-Measure | 0.521 | 0.620 | 0.678 | 0.765 | 0.789 | 0.855 |

#### 4.4.5 performance analysis for different Labelling techniques

The technique used to discover unknown examples and identify novel classes has employed BRICH and the single-rank method. However, we evaluated the OpenCML framework with different approaches to validate the performance. We conducted an ablation study to justify the use of particular techniques.

For keyword extraction, we employed the Single Rank method, which outperforms contemporary methods and KeyBERT, a recently proposed method. We have compared six different keyword extraction methods, as evaluated by several metrics, including Accuracy, Adjusted Rand Score (ARS), Normalised Mutual Information (NMI), Fowlkes-Mallows Score (FMS), and F1-score. For DS-1, it is clear from the graph that the SingleRank method outperforms all other methods in terms of Accuracy, NMI, FMS, and F1-score, with an accuracy score of 0.879 and an NMI score of 0.682. TextRank also performs well, with an accuracy score of 0.856 and an NMI score of 0.525. YAKE and KeyBERT demonstrate similar performance, with YAKE achieving higher scores in ARS, NMI, and FMS, while KeyBERT exhibits a higher accuracy score. Kea presents the lowest overall performance, with insufficient Accuracy, ARS, and F1-score scores. Overall, Singlerank performed better than other methods.

For DS-2, SingleRank achieves the highest F1 score of 0.989, the highest among all the algorithms. Additionally, SingleRank achieves the highest scores for Accuracy and NMI, at 0.979 and 0.917, respectively. SingleRank’s score for ARS is slightly lower than YAKE, but it still achieves a high score of 0.952, indicating its effectiveness for automated keyword extraction. In comparison, other algorithms, such as TFIDF and Kea, have significantly lower scores across all metrics, indicating their lower effectiveness compared to SingleRank. It indicates that SingleRank is highly effective and outperforms other algorithms across all the evaluation metrics.

For DS-3, SingleRank achieves the highest score for ARS with a score of 0.682, the highest among all algorithms. SingleRank also achieves the highest F1-score, with a score of 0.888, surpassing YAKE, TextRank, Kea, and KeyBERT. Additionally, SingleRank achieves a higher score than TextRank and Kea in terms of Accuracy, with a score of 0.899. However, YAKE has the highest accuracy score, at 0.936. In contrast, TFIDF and Kea have the lowest scores across most metrics. The YAKE performance is slightly better here, but the Singlerank is given a better F1-score.

For DS-4, Singlerank achieves the highest Accuracy, ARS, F1-score, and FMS scores, with scores of 0.742, 0.512, 0.712, and 0.678, respectively. While its score for NMI is the same as that of YAKE and Kea, it still outperforms TextRank and KeyBERT in this metric. In comparison, YAKE and Kea have the lowest scores for Accuracy, with scores of 0.562. Similarly, TextRank has the lowest score for ARS and the second-lowest score for Accuracy. KeyBERT also has a lower score for ARS than SingleRank. Overall, SingleRank is the most effective and outperforms other algorithms across most metrics.

*(a) DS-1*

*(b) DS-2*

*(c) DS-3*

*(d) DS-4*

*Figure 3: A performance analysis of keyword extraction techniques for effective label detection for unknown classes in OpenCML*

*Table 7: Comparison of OpenCML with other recent proposed works of continual machine learning for natural language processing, none of them offers continual learning with open-text classification *

| | | | | | |

| Authors | Dataset | OTC | CML |

| | | | Task | Number of increments/tasks | Proposed result |

| Huang et al. (2021) | bpedia, yahoo, ag, amazon, yelp | NA | TIL | 5 task | Avg Acc 73.19 |

| Ke et al. (2021) | Amazon Revies | NA | TIL | 24 task | Avg Acc 0.8524 |

| Ermis et al. (2022) | Arxiv Papers, Reuters, Wiki-30K | NA | CIL | 5 task | Avg Acc 0.88 |

| | twitter data | NA | CIL | 5 rounds | Avg Acc 76.8 |

| OpenCML | DS1, DS2 ,DS3, DS4 | Avg Acc 0.902 & F1-score 0.831 | CIL | 4 rounds | Avg Acc 0.598 & F1-score 0.709 |

## 5 Conclusion and Future Work

OpenCML, which can discover and identify unknown novel classes, is promising for applications in different domains, including computer vision, natural language processing, and speech recognition. It enables the system to continuously learn from new knowledge without forgetting prior knowledge. The model’s capacity to discover new classes without retraining is especially helpful in dynamic environments where new data arrive frequently. Nonetheless, some constraints and challenges still need to be handled. One area for future research and essential direction is the integration of knowledge transfer and reinforcement learning into the incremental learning framework. Moreover, the interpretability and explainability of the models should also be considered to secure clarity and accountability. Overall, OpenCML has significant potential and can play an increasingly significant part in future machine learning and AI applications.

#### Acknowledgements

Not applicable

## Declarations

-

Funding: Not applicable

-

Conflict of interest/Competing interests: There is a conflict of Interest

-

Ethics approval and consent to participate: Not applicable

-

Consent for publication: Yes

-

Data availability : Not applicable

-

Materials availability: Not applicable

-

Code availability : Yes

-

Author contribution: All authors contributed equally to this work

## References

- Aljundi et al. (2018) R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars Memory aware synapses: learning what (not) to forget. In Proceedings of the European Conference on Computer Vision, pp. 139–154. Cited by: §1, §2.2.3, Table 2.

- Auer et al. (2007) S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, and Z. Ives Dbpedia: a nucleus for a web of open data. In The semantic web, pp. 722–735. Cited by: 4th item.

- Belouadah and Popescu (2019) E. Belouadah and A. Popescu Il2m: class incremental learning with dual memory. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 583–592. Cited by: §1, §2.2.4, Table 2.

- Bendale and Boult (2016) A. Bendale and T. E. Boult Towards open set deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1563–1572. Cited by: §4.4.2.

- Benzing (2022) F. Benzing Unifying importance based regularisation methods for continual learning. In International Conference on Artificial Intelligence and Statistics, pp. 2372–2396. Cited by: §1, §2.2.3, Table 2.

- Bottou et al. (2018) L. Bottou, F. E. Curtis, and J. Nocedal Optimization methods for large-scale machine learning. SIAM review 60 (2), pp. 223–311. Cited by: §1, §2.2.2, Table 2.

- Bottou (2014) L. Bottou From machine learning to machine reasoning: an essay. Machine learning 94, pp. 133–149. Cited by: §2.1.

- Burkart and Huber (2021) N. Burkart and M. F. Huber A survey on the explainability of supervised machine learning. Journal of Artificial Intelligence Research 70, pp. 245–317. Cited by: §2.1.

- Caccia et al. (2020) L. Caccia, E. Belilovsky, M. Caccia, and J. Pineau Online learned continual compression with adaptive quantization modules. In International Conference on Machine Learning, pp. 1240–1250. Cited by: §1, §2.2.4, Table 2.

- Casanueva et al. (2020) I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, and I. Vuli’c Efficient intent detection with dual sentence encoders. In proceedings of the 28th International Conference on Computational Linguistics, pp. 38–45. Cited by: 1st item.

- Castro et al. (2018) F. M. Castro, M. J. Marín-Jiménez, N. Guil, C. Schmid, and K. Alahari End-to-end incremental learning. In Proceedings of the European conference on computer vision (ECCV), pp. 233–248. Cited by: §3.3.3, §3.3.

- Cha et al. (2021) H. Cha, J. Lee, and J. Shin Co2l: contrastive continual learning. In Proceedings of the IEEE/CVF International conference on computer vision, pp. 9516–9525. Cited by: §1, §2.2.1, Table 2.

- Chaudhry et al. (2018a) A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. Torr Riemannian walk for incremental learning: understanding forgetting and intransigence. In Proceedings of the European Conference on Computer Vision, pp. 532–547. Cited by: §1, §2.2.3, Table 2.

- Chaudhry et al. (2018b) A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny Efficient lifelong learning with a-gem. In International Conference on Learning Representations, Cited by: §1, §2.2.2, Table 2.

- Chaudhry et al. (2019) A. Chaudhry, M. Rohrbach, M. Elhoseiny, T. Ajanthan, P. K. Dokania, P. H. Torr, and M. Ranzato On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486. Cited by: §1, §2.2.4, Table 2.

- Chen and Liu (2018) Z. Chen and B. Liu Lifelong machine learning. Synthesis Lectures on Artificial Intelligence and Machine Learning 12 (3), pp. 1–207. Cited by: §2.

- Ebrahimi et al. (2020) S. Ebrahimi, S. Petryk, A. Gokul, W. Gan, J. E. Gonzalez, M. Rohrbach, et al. Remembering for the right reasons: explanations reduce catastrophic forgetting. In International Conference on Learning Representations, Cited by: §1, §2.2.4, Table 2.

- Ermis et al. (2022) B. Ermis, G. Zappella, M. Wistuba, A. Rawal, and C. Archambeau Memory efficient continual learning with transformers. Advances in Neural Information Processing Systems 35, pp. 10629–10642. Cited by: Table 7.

- Fei and Liu (2016) G. Fei and B. Liu Breaking the closed world assumption in text classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 506–514. Cited by: §1, §2.1, Table 2.

- Fini et al. (2022) E. Fini, V. G. T. Da Costa, X. Alameda-Pineda, E. Ricci, K. Alahari, and J. Mairal Self-supervised models are continual learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9621–9630. Cited by: §1, §2.2.1, Table 2.

- French and Chater (2002) R. M. French and N. Chater Using noise to compute error surfaces in connectionist networks: A novel means of reducing catastrophic forgetting. Neural Comput.. Cited by: §1.

- Gallardo et al. (2021) J. Gallardo, T. L. Hayes, and C. Kanan Self-supervised training enhances online continual learning. arXiv preprint arXiv:2103.14010. Cited by: §1, §2.2.1.

- Gong et al. (2022) Z. Gong, K. Zhou, W. X. Zhao, J. Sha, S. Wang, and J. Wen Continual pre-training of language models for math problem understanding with syntax-aware memory network. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5923–5933. Cited by: §1, §2.2.4, Table 2.

- Hendrycks and Gimpel (2016) D. Hendrycks and K. Gimpel A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136. Cited by: §4.4.2.

- Hinton et al. (2015) G. Hinton, O. Vinyals, and J. Dean Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531. Cited by: §3.3.3, §3.3.3.

- Huang et al. (2021) Y. Huang, Y. Zhang, J. Chen, X. Wang, and D. Yang Continual learning for text classification with information disentanglement based regularization. arXiv preprint arXiv:2104.05489. Cited by: Table 7.

- Javed and White (2019) K. Javed and M. White Meta-learning representations for continual learning. Advances in Neural Information Processing Systems 32. Cited by: §1, §2.2.1.

- Ke et al. (2021) Z. Ke, B. Liu, H. Wang, and L. Shu Continual learning with knowledge transfer for sentiment classification. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2020, Ghent, Belgium, September 14–18, 2020, Proceedings, Part III, pp. 683–698. Cited by: Table 7.

- Kiefer and Wolfowitz (1952) J. Kiefer and J. Wolfowitz Stochastic estimation of the maximum of a regression function. The Annals of Mathematical Statistics, pp. 462–466. Cited by: §1, §2.2.2, Table 2.

- Kirkpatrick et al. (2017) J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114 (13), pp. 3521–3526. Cited by: §1, §2.2.3, Table 2, §2.

- Kotsiantis et al. (2007) S. B. Kotsiantis, I. Zaharakis, P. Pintelas, et al. Supervised machine learning: a review of classification techniques. Emerging artificial intelligence applications in computer engineering 160 (1), pp. 3–24. Cited by: §2.1.

- Larson et al. (2019) S. Larson, A. Mahendran, J. J. Peper, C. Clarke, A. Lee, P. Hill, J. K. Kummerfeld, K. Leach, M. A. Laurenzano, L. Tang, and J. Mars An evaluation dataset for intent classification and out-of-scope prediction. In International Joint Conference on Natural Language Processing, pp. 1311–1316. Cited by: 2nd item.

- Lin and Xu (2019a) T. Lin and H. Xu A post-processing method for detecting unknown intent of dialogue system via pre-trained deep neural network classifier. Knowledge-Based Systems 186, pp. 104979. Cited by: §1, §2.1, Table 2.

- Lin and Xu (2019b) T. Lin and H. Xu Deep unknown intent detection with margin loss. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 5491–5496. Cited by: §4.4.2.

- Lopez-Paz and Ranzato (2017) D. Lopez-Paz and M. Ranzato Gradient episodic memory for continual learning. Advances in Neural Information Processing Systems 30. Cited by: §1, §2.2.2, §2.2.4, Table 2, Table 2.

- Madaan et al. (2021) D. Madaan, J. Yoon, Y. Li, Y. Liu, and S. J. Hwang Representational continuity for unsupervised continual learning. In International Conference on Learning Representations, Cited by: §1, §2.2.1, Table 2.

- McCloskey and Cohen (1989) M. McCloskey and N. J. Cohen Catastrophic interference in connectionist networks: the sequential learning problem. In Psychology of learning and motivation, Vol. 24, pp. 109–165. Cited by: §1, §2.

- Mehta et al. (2021) S. V. Mehta, D. Patil, S. Chandar, and E. Strubell An empirical investigation of the role of pre-training in lifelong learning. arXiv preprint arXiv:2112.09153. Cited by: §1, §2.2.1.

- Prakhya et al. (2017) S. Prakhya, V. Venkataram, and J. Kalita Open set text classification using convolutional neural networks. In International Conference on Natural Language Processing, 2017, Cited by: §1, §2.1, Table 2.

- Purushwalkam et al. (2022) S. Purushwalkam, P. Morgado, and A. Gupta The challenges of continuous self-supervised learning. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVI, pp. 702–721. Cited by: §1, §2.2.1, Table 2.

- Rebuffi et al. (2017) S. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert Icarl: incremental classifier and representation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2001–2010. Cited by: §1, §2.2.4, Table 2.

- Riemer et al. (2018) M. Riemer, I. Cases, R. Ajemian, M. Liu, I. Rish, Y. Tu, and G. Tesauro Learning to learn without forgetting by maximizing transfer and minimizing interference. In International Conference on Learning Representations, Cited by: §1, §2.2.2, §2.2.4, Table 2, Table 2.

- Robbins and Monro (1951) H. Robbins and S. Monro A stochastic approximation method. The annals of mathematical statistics, pp. 400–407. Cited by: §1, §2.2.2, Table 2.

- Robins (1993) A. V. Robins Catastrophic forgetting in neural networks: the role of rehearsal mechanisms. In International Two-Stream Conference on Artificial Neural Networks and Expert Systems, ANNES, Cited by: §1.

- Robins (1995) A. V. Robins Catastrophic forgetting, rehearsal and pseudorehearsal. Connect. Sci.. Cited by: §1.

- Saha et al. (2020) G. Saha, I. Garg, and K. Roy Gradient projection memory for continual learning. In International Conference on Learning Representations, Cited by: §1, §2.2.4, Table 2.

- Schwarz et al. (2018) J. Schwarz, W. Czarnecki, J. Luketina, A. Grabska-Barwinska, Y. W. Teh, R. Pascanu, and R. Hadsell Progress & compress: a scalable framework for continual learning. In International conference on machine learning, pp. 4528–4537. Cited by: §1, §2.2.3, Table 2.

- Serra et al. (2018) J. Serra, D. Suris, M. Miron, and A. Karatzoglou Overcoming catastrophic forgetting with hard attention to the task. In International Conference on Machine Learning, pp. 4548–4557. Cited by: §1.

- Shu et al. (2017) L. Shu, H. Xu, and B. Liu DOC: deep open classification of text documents. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2911–2916. Cited by: §1, §2.1, Table 2, §4.4.2.

- Tang et al. (2021) S. Tang, D. Chen, J. Zhu, S. Yu, and W. Ouyang Layerwise optimization by gradient decomposition for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9634–9643. Cited by: §1, §2.2.2, Table 2.

- Thrun (1995) S. Thrun Is learning the n-th thing any easier than learning the first?. Advances in neural information processing systems 8. Cited by: §2.

- Vedula et al. (2020) N. Vedula, R. Gupta, A. Alok, and M. Sridhar Automatic discovery of novel intents & domains from text utterances. arXiv preprint arXiv:2006.01208. Cited by: §1, §2.1, Table 2.

- Vedula et al. (2019) N. Vedula, N. Lipka, P. Maneriker, and S. Parthasarathy Towards open intent discovery for conversational text. arXiv preprint arXiv:1904.08524. Cited by: §1, §2.1, Table 2.

- Welling (2009) M. Welling Herding dynamical weights to learn. In Proceedings of the 26th Annual International Conference on Machine Learning, pp. 1121–1128. Cited by: §3.3.1.

- Wu et al. (2022) T. Wu, G. Swaminathan, Z. Li, A. Ravichandran, N. Vasconcelos, R. Bhotika, and S. Soatto Class-incremental learning with strong pre-trained models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9601–9610. Cited by: §1, §2.2.1.

- Xu et al. (2015) J. Xu, P. Wang, G. Tian, B. Xu, J. Zhao, F. Wang, and H. Hao Short text clustering via convolutional neural networks. In Proceedings of the 1st Workshop on Vector Space Modeling for Natural Language Processing, pp. 62–69. Cited by: 3rd item.

- Yan et al. (2020) G. Yan, L. Fan, Q. Li, H. Liu, X. Zhang, X. Wu, and A. Y. Lam Unknown intent detection using gaussian mixture model with an application to zero-shot intent classification. In proceedings of the 58th annual meeting of the association for computational linguistics, pp. 1050–1060. Cited by: §4.4.2.

- Zeng et al. (2021) Z. Zeng, K. He, Y. Yan, Z. Liu, Y. Wu, H. Xu, H. Jiang, and W. Xu Modeling discriminative representations for out-of-domain detection with supervised contrastive learning. In proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pp. 870–878. Cited by: §4.4.2.

- Zenke et al. (2017) F. Zenke, B. Poole, and S. Ganguli Continual learning through synaptic intelligence. In International Conference on Machine Learning, pp. 3987–3995. Cited by: §1, §2.2.3, Table 2.

- Zhang et al. (2021) H. Zhang, H. Xu, and T. Lin Deep open intent classification with adaptive decision boundary. In proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35, pp. 14374–14382. Cited by: §4.4.2.

- Zhou et al. (2022) Y. Zhou, P. Liu, and X. Qiu KNN-contrastive learning for out-of-domain intent classification. In proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5129–5141. Cited by: §4.4.2.

<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2311.11250
     paper_id : 2311.11250
     source   : https://arxiv.org/html/2311.11250v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

﻿

∎

# A Comprehensive Review on Sentiment Analysis: Tasks, Approaches and Applications

Sudhanshu Kumar∗1    Partha Pratim Roy1    Debi Prosad Dogra 2    Byung-Gyu Kim 3 E-mail: skumar2@cs.iitr.ac.in E-mail: partha@cs.iitr.ac.in E-mail: dpdogra@iitbbs.ac.in E-mail: bg.kim@sookmyung.ac.kr Affiliation: Sudhanshu Kumar

Partha Pratim Roy

Debi Prosad Dogra

Byung-Gyu Kim

1Department of Computer Science and Engineering, IIT Roorkee, 247667, India
2School of Electrical Sciences, IIT Bhubaneswar, Odisha 752050, India.
3Department of IT Engineering, Sookmyung Women’s University, Seoul 04310, South Korea

∗Corresponding Author

Received: date / Accepted: date

###### Abstract

Sentiment analysis (SA) is an emerging field in text mining. It is the process of computationally identifying and categorizing opinions expressed in a piece of text over different social media platforms. Social media plays an essential role in knowing the customer mindset towards a product, services, and the latest market trends. Most organizations depend on the customer’s response and feedback to upgrade their offered products and services. SA or opinion mining seems to be a promising research area for various domains. It plays a vital role in analyzing big data generated daily in structured and unstructured formats over the internet. This survey paper defines sentiment and its recent research and development in different domains, including voice, images, videos, and text. The challenges and opportunities of sentiment analysis are also discussed in the paper.

###### Keywords:

Sentiment Analysis, Machine Learning, Lexicon-based approach, Deep Learning, Natural Language Processing

## 1 Introduction

Sentiment Analysis is the computational study of people’s opinions, attitudes, and emotions in the form of different modalities (text, image, and speech) toward an entity that represents topics, events, issues, products, services, and organizations. SA is the branch of many fields such as machine learning, data mining, natural language processing, and computational linguistics. Natural Language Processing (NLP) generally started back in the 1950s; little attention was paid by researchers to people’s opinions and sentiment analysis until 2005. With the advancement of web 2.0, web 3.0, web 4.0, social media thrusts SA’s development. Social media propels the growth of sentiment analysis. Most of the literature is on the English language, but many publications currently tackle the multilingual issue. SA is a suitcase research problem Cambria et al (2017) that is the combination of NLP tasks such as named entity recognition (Ma et al, 2016), concept extraction (Cambria et al, 2016), sarcasm detection (Poria et al, 2016a), aspect extraction (Ma et al, 2018), and subjectivity detection. Subjective information indicates the opinions of opinion holders, while objective texts show some objective facts. For example, ”The food is great and delicious.” These opinion words are subjective. Subjective texts can have a positive or negative sentiment.

SA classification process, as shown in Figure 1 and 2 uses any classification model to classify the reviews into positive, negative and neutral classes. There are three levels of SA such as document level, sentence level, and aspect level. In the document level, the whole document expresses a positive or negative opinion. For Example, the product reviews document has either positive or negative opinions for a product. It represents a single opinion for a document, so it comes under the document level. Sentence level is the second category widely used in e-commerce sites in which each sentence classifies into positive, negative, and neutral opinions. Aspect level sentiment analysis is also called feature-based analysis. In this type of analysis, each review categorizes into aspects and their target opinions. This level shows more insights about the opinion that it is positive or negative for which aspect. For Example, ‘The Food was very good at the hotel.’ It is an aspect-based SA where food is one aspect of the review.

*Figure 1: The process to classify the review into positive, negative, neutral using Machine learning.*

*Figure 2: The process to classify the review into positive, negative and neutral using Deep learning techniques.*

Different sentiment classification techniques are shown in Figure 3. It is divided into two categories, i.e., lexicon-based approach and machine learning approach. The Lexicon-based approach uses the dictionaries of words annotated with their semantic orientation, classified into the dictionary and corpus-based approach. The second category is the machine learning approach based on different types of learning like supervised, unsupervised, and semi-supervised. Depending on the nature of the data, these learning techniques are used and predict the result. Different deep learning-based and machine learning-based techniques are the most popular ones. The total number of research publications year-wise is as shown in Figure 4. It shows that as the advancement of industry 4.0 and now it’s 5.0, the numbers of research papers are increasing year by year.

*Figure 3: All sentiment analysis classification techniques from traditional to the latest one have been shown in this figure. Initially, It is divided into machine learning and lexicon-based approach, which further divide into different algorithms.*

*Figure 4: The number of research papers with the Scopus index are published in the last ten years from 2011 to Mid 2020. The number of publications is increasing yearly as the advancement in technology and the evolution of industry 4.0.*

The contributions of the paper are as follows:

-

A large number of literature has been reviewed in sentiment analysis process from multiple domains and identify the pros and cons of all approaches.

-

Summarizing each of the surveyed articles in detail, including the problems addressed, dataset details and methods.

-

Analyses of existing applications in order to determine which one is most suitable for certain application.

-

Discussing the challenges and application of sentiment analysis in order to keep up the current research trends.

The rest of the paper is organized as follows. In Section 2, we discuss the state of the art discussion on SA. Detailed discussion on the existing work, open issues, and possible applications of sentiment analysis is presented in Section 3 and Section 4. Finally, in the last Section 5 the research work has been concluded.

## 2 Terminology and background concepts

Opinion, views, and feeling are often used interchangeably in the different literature Munezero et al (2014). SA is also related to many terms such as emotions, moods, and feelings, which sometimes confuse the reader with opinion or SA. Emotion is related to the perception of the stimulus and the triggering of the bodily response. For example, one person shows an angry response when they lose the job, and in another case, they feel joy in the same situation. Many authors (Munezero et al, 2014; Scherer, 2005) showed that emotion is short term, while the mood is a long-term phenomenon. They differentiate on both terms basis on their duration.

A schematic representation of opinion and sentiment are given in Figure 5.

*Figure 5: Schematic structure of sentiments (Munezero et al, 2014), which further divide into sentiment holder, emotional disposition and object of the review. The emotion is short-term, while the mood is long-term disposition.*

SA has raised a growing interest in financial and political forecasting, e-health, e-tourism, and dialogue systems. The authors (Gallege and Raje, 2016) proposed trust-based ranking and the recommendation tool to improve online software services recommendation. This system enhances the existing recommendation system (content-based and collaborative filtering based) algorithm by considering the external attributes. The proposed system result was evaluated on the Amazon marketplace review dataset and showed a better ranking.

SA used many libraries such as TextBlob and naive Bayes to classify the content based on polarity score and subjectivity. In (Fang and Zhan, 2015), the authors proposed a sentiment polarity categorization process, which was the fundamental problem of sentiment earlier. The amazon product reviews dataset’s experimental results achieved an F1 score of 0.8 and .0.73 for sentence-level and review-level categorization, respectively. The polarity shift problem is one of the challenges in sentiment analysis to predict user reviews. Xia et al. (Xia et al, 2015) proposed dual sentiment analysis model to address the polarity shift problem. The model trained for sentiment-reversed review for both training and testing. The result of the multi-domain and Chinese datasets showed the effectiveness of the model.

In (Dasgupta et al, 2015), the authors proposed a map reducing paradigm to collect the user’s data from Facebook to understand brand reviews. They refined their approach through the iterative process of data pre-processing. In (Anto et al, 2016), the authors proposed an automatic feedback technique based on Twitter data. Different classifiers like SVM, Naive Bayes, and maximum entropy are used on Twitter comments. Out of these classifiers, SVM-based performance was the highest. In (Tan and Wu, 2011), the authors proposed a random walk algorithm for domain-oriented sentiment lexicon based on utilizing sentiment words and documents from both the old and target domains. The proposed algorithm reflects four kinds of relationships (words to documents, words to words, documents to words, and documents to documents) between words and documents. Experimental results indicate improvements in identifying the polarities of sentiment words. Day et al. (Day and Lee, 2016) presented that analytical methods use deep learning in financial news sources to forecast stock price trends. The authors found that financial news media sources can reveal investment information. Sentiment analysis aims to classify text in positive and negative polarity scores useful in quantifying different affective states of a user (Poria et al, 2017). Cambria et al. (Cambria et al, 2017) have developed an NLP approach that leverages both data and theory-driven methods to understand natural language.
Many approaches have simple categorization problems; however, sentiment analysis is a big suitcase problem requiring multiple polarity detection tasks. NLP problems divide into three layers: syntactic, semantics, and pragmatics. Each layer has a different subtask to process each layer’s text output as input for the next layer. In (Poria et al, 2016a), the authors have developed a pre-trained model for extracting emotion, sentiment, and personality features from sarcastic tweets using CNN. Experiments were conducted on a dataset consisting of both sarcastic and non-sarcastic tweets. Results computed on three datasets with $F_{1}$ scores of 87%, 92.32%, and 93.30%, respectively.

### 2.1 Sentiment in Text

Text is mainly an important medium to express the user’s state of mind in reviews and comments on the internet. It was the primary mode of communication in early 1990 when e-commerce company amazon was the first company to do business online. In 1992, the authors (Hearst, 1992) proposed an approach based on the sentence’s directionality. The approach is based on the semantic orientation of the sentence to determine the directionality of the text. Another researcher (Sack, 1994) whose theory is based on the information’s subjective point of view. In the early 2000s, many researchers (Das and Chen, 2001; Morinaga et al, 2002; Pang et al, 2002; Tong, 2001; Turney and Littman, 2003; Wiebe et al, 2000) worked on sentiments analysis and opinions mining. Nasukawa et al. (Nasukawa and Yi, 2003) showed the high precision result on customer reviews and news articles available over web pages. They classified the specific subjects from a document in positive or negative polarity. This paper’s result rise in interest to other researchers in this domain. The influential 2008 review of Pang and Lee (Pang and Lee, 2008) covers techniques and approaches that promise to directly enable opinion-oriented information-seeking systems on benchmark datasets in recent research. Here, we discuss sentiment analysis in NLP, including its different methods, such as supervised and unsupervised.

#### 2.1.1 Supervised Approach

It is based on the annotated dataset (labeled data) to build a prediction model. This approach builds a feature vector of the text, either aspect or word frequency, then the model learns (training) on the dataset and gives prediction for unseen data in testing. The first paper (Wiebe et al, 1999) used this approach to classify the text as subjective or objective on the gold standard dataset. It achieves 81.5% accuracy on the probabilistic classifier. There are different approaches in machine learning, like the supervised and unsupervised approaches. The supervised approach was used on stock trading (Das and Chen, 2007) domain to find the sentiment analysis. Further development focus on user comments available on the E-commerce site. Many machine learning algorithms like SVM, Naive Bayes, and linear regression solve the problem related to a different domain. SVM was the most suitable model for product reviews in supervised sentiment analysis.

The authors (Devlin et al, 2018) proposed BERT (bidirectional encoder representations from transformers) model designed to pre-trained deep bidirectional from the unlabelled text by jointly conditioning on both left and right context. Bidirectional means that BERT learns information from both the left and the right side of a token’s context during the training phase. The model is implemented on eleven NLP tasks such as GLUE, MultiNLI, SWAG, SQuAD v1.1, etc., and shown the impressive state-of-the-art result in the SA field. Unlike recent language representation models (Peters et al, 2018; Radford et al, 2018), this model is used for a wide range of tasks like question answering and language inference.

#### 2.1.2 Unsupervised Learning Approach

In this approach, the labeled data is not present, allowing an estimation based on expert knowledge. The most popular method in unsupervised learning is cluster analysis to find the data’s hidden pattern. In sentiment analysis, the lexicon plays an essential role in classifying the text into positive, negative, and neutral depending on the lexicon method, a combination of words or phrases. The most popular lexicon is General Inquirer (STONE, 1997), it is a corpus of positive and negative terms.

The aim to improve sentence-level classification, recently few methods are performing well such as SentiStrength (Thelwall et al, 2010), Valence Aware Dictionary and sEntiment Reasoner (VADER) (Hutto and Gilbert, 2014), and Umigon (Levallois, 2013). VADER is a lexicon and rule-based sentiment analysis method used to find the sentiment of the reviews available on different social media platforms. The reviews are shared by a user from different age groups and gender, so these reviews are not available in the form that can be directly processed by any method. It converted into a normalized form after pre-processing of these reviews. VADER evaluates the words and their context on pre-processed reviews based on a predefined dictionary with many words (sentiment lexicon) and their corresponding numeric score. The method produces four metrics for each review; the first three are positive, neutral, and negative. The last metric is a compound score used to identify these reviews’ sentiment. VADER is popular among other methods to analyze social media posts with slang, emoticons, and acronyms.

Du et al. (Du and Huang, 2018) proposed an attention mechanism for news categories in Chinese (NLPCC201) and English (REV1-v) datasets. The experimental result shows that this mechanism is beneficial to assign a score for keywords. The keywords that have a higher score in the corpus mean that these keywords are more important to the dataset than non-key words, so it improves the classifier’s accuracy compared to recurrent neural network (Mikolov et al, 2010) and long short term memory (Hochreiter and Schmidhuber, 1997). This mechanism showed an effective result in many research papers (Zadeh et al, 2018; Yang et al, 2016; Yan et al, 2019) for a different domain such as document classification (yelp reviews, IMDB reviews, Yahoo answers, and amazon reviews), understand human communication (language, vision, and acoustic modality), and video captioning, etc.

#### 2.1.3 Word Embedding

The development of deep learning techniques in sentiment analysis shows promising results in most real-world problems. Word embedding is the dominant approach in NLP problems compare to one-hot encoding. If the words are present in the vocabulary in one-hot encoding, then assign one else zero. The issue in one hot encoding is a computational issue. When you increase your vocabulary by size n, the feature size vector also increases by length n, requiring more computational time to train the model. A word embedding is a learned representation for text data where words or phrases with the same meaning have a similar representation mapped further either in vector or real numbers. The strategy typically includes a mathematic concept from a high-dimensional vector space to a lower-dimensional vector space. The vectors encoding is related to linguistic regularities and patterns, each dimension related to the word’s feature. The learning of word embedding is done by neural network (Bengio et al, 2003) from the text.
The most common word embedding system is word2vec, in which the words related to each other, like king-queen and man-women, are represented in the vector space near each other. The word2vec model approach is based on two models i.e. continuous bag-of-words (Mikolov et al, 2013a) and skip-gram model (Mikolov et al, 2013b). Another frequent word embedding technique is Glove Vector (Pennington et al, 2014) (GloVe), which utilizes both global statistics and local statistics to train word vector fast and scalable. The word2vec captures local statistics to do works well on analogy tasks. The author(Araque et al, 2017) proposed ensemble techniques that were the combination of word embeddings and a linear algorithm on seven public datasets extracted from the microblogging and movie reviews domain. This paper showed that word embedding techniques enhance the proposed model’s performance and work well in a smaller dataset. The deep learning algorithm does not perform very well when the dataset is small because it requires a large amount of data to train the model, so the word embedding algorithm is used in this case. Pre-trained word embedding is used to solve many research problems (Ren et al, 2016; Tang et al, 2014; Giatsoglou et al, 2017).

#### 2.1.4 Others Techniques

Qazi et al. (Qazi et al, 2017) proposed assessing users’ opinions on multiple topics like a social get-together, promoting efforts, and item inclinations. This study aims to find the users’ expectations and satisfaction at the post-purchase stage. They surveyed a questionnaire comprising seven sections, and the data was collected through LinkedIn and the university mail servers. The authors utilized a disconfirmation hypothesis, a set of seven theories, confirmatory factor analysis, and primary conditions to break down the users’ information and assess the model. The model’s consequences demonstrated that regular, comparative, and interesting assessments positively raise users’ desires. The author presumed that a wide range of sentiments is a rich wellspring of data that at last influences the customer loyalty level. Wang et al. (Wang et al, 2018) proposed a SentiRelated algorithm to fill the gap between different domains. The traditional supervised classification algorithm is performed well for a given domain but does not work well on different domains. The SentiRelated algorithm is based on the Sentiment Related Index to improve the model’s performance when tested on other domains. This algorithm was validated on two datasets with different domains such as a computer, Education, Hotel, Movie, Music, and Book reviews and showed 80% accuracy for short texts. Social media platforms like Twitter are trending to become a common platform for exchanging raw data and online text, providing a vast platform for sentiment analysis.
The author proposed (Pandey et al, 2017) a novel metaheuristic method based on Cuckoo Search and K-means (called CSK). It enlightens the clustering-based methods for analysing Twitter tweets to find the user’s viewpoints and the sentiment pertained while making such a tweet. The method proposed outlines to find the optimum cluster-heads from the Twitter dataset’s sentimental contents. The model tested its efficacy on various Twitter datasets and then compared it with the existing methods such as particle swarm optimization, differential evolution, cuckoo search, improved cuckoo search, etc. This research work performed a basis for designing a system that quickly provides conclusive reviews on any social issues.
The authors (Saif et al, 2016) discussed an approach where a publicized tweets from the Twitter site are processed and classified based on their polarity. In this paper, a new tool is called ”SENTICIRCLE” is used a lexicon-based approach. The word’s semantics are extracted from its co-occurrence pattern, and the strength is updated in the lexicon accordingly. The basic idea of the approach is that the group of word accompanying it decides the semantic of the word in any text. The force of movement from the static word sentiment orientation approach to this contextual approach derived from the dictum ”YOU SHALL KNOW THE WORD BY THE COMPANY IT KEEPS.” It is different from the traditional lexicon, where the words are given fixed static semantics regardless of the context.

#### 2.1.5 Microblogging Data of Non-English Text

Many studies have been conducted in sentiment analysis on English texts, while other languages have less attention than Arabic, Hindi, Bangla, etc. Many researchers have worked on SA in different languages after the rise of Web 2.0. The author in (Al-Ayyoub et al, 2019) introduced an overview on Arabic assessment analysis (Badarneh et al, 2018; Al-Radaideh and Al-Qudah, 2017; Socher et al, 2013) in which they examined various tools and applications pertinent to it. The study additionally included both corpus-based and dictionary-based approaches for different datasets. Microblogs like tweets are trending rapidly for online users to share their experiences and opinions daily. In contrast to the online reviews and blogs, these microblogs contain very dispersed and incomplete data. Unlike English-based microblogs, Chinese microblogs such as Sina Weibo have less sentiment analysis. The reason being that Chinese textual analysis is more challenging than English as its grammar of expression is different. The same length of Chinese sentences may contain more data than English, and the separation of words in those texts is relatively obscure. In totality, textually analyzing Chinese blogs has three primary research goals: First, the new words mining and their sentiment inference; second, how to extract other media modules and third, establish a hierarchical sentiment detection method based on Sina Weibo linguistics. The authors (Wang et al, 2014) proposed three primary goals for the analysis of these Chinese microblogs. They visualize the sentiment analysis’s result, depicting the relationship between social network sentiments and real-life events.
The researchers already working on Chinese microblogs tend to analyze the topic focussing on a single attribute while neglecting others. The model design is multilevel in single-level features keeping all the aspects under consideration. Chen et al. (Chen et al, 2017) used to extract the text’s sentiment using sentence-level sentiment analysis, but unlike other traditional approaches where the same technique was used in all types of sentences. The sentences are classified into three groups based upon their opinion targets. There are other ways to classify the sentences that have been previously used in other research papers. For example, The sentence can be subjective or objective based upon the subjectivity of the sentences. The subjective sentences express the opinions, while objective sentences implicate opinions or sentiments. The opinionated targets focused on the primary sentence classification. This opinion target can be any entity on which opinion is expressed. These opinionated sentences can give an opinion without mentioning the target on three different types of sentences: non-target, one-target, and multi-target. The Bi-LSTM and CNN deep learning approaches were used to classify the sentences and extract the text’s syntactic and semantic features.

### 2.2 Sentiment in Speech

Analysis of speech in search of emotional and affective cues has a comparably long tradition (Dellaert et al, 1996). This paper proposed statistical pattern recognition techniques to classify 1000 utterances according to their emotional content. Meanwhile, several kinds of literature have been established, including a range of recent surveys in emotions and affect in speech (Schuller et al, 2011). However, targeting sentiment explicitly exclusively from spoken utterances is a comparably new field than text-based sentiment analysis. Focusing on the acoustic side of spoken language, the border between sentiment and emotion analysis is often fragile, as discussed in (Crouch and Khosla, 2012). Mairesse et al. (Mairesse et al, 2012) focused on pitch-related features and observed that pitch contains information on sentiment without textual cues. The authors collected short-spoken reviews from 84 speakers, and the result outperformed a majority class baseline. This paper attracted other researchers to explore this area to solve real-world problems. The authors (Elmadany et al, 2018) created Arabic Speech Act and Sentiment (ArSAS) dataset. The dataset consisted of 21,064 tweets annotated for two tasks: speech act recognition and SA. Further, the tweets are annotated for four different sentiment categories: positive, negative, neutral, and mixed.
Ahmed et al. (Ahmed et al, 2016) showed the sentiment in phone calls by first using speech recognition to extract the text in the call and then use typical text-based SA techniques. The goal was to measure agent productivity in call centers.

### 2.3 Image based Sentiment Analysis

Vision-based emotion recognition (Zeng et al, 2008; Sariyanidi et al, 2014; Campos et al, 2017) is a relatively recent area of research. Users share millions of images and videos over social media platforms like Twitter, Tumblr, Flickr, and Instagram. These are the most popular sites where celebrities from sports, entertainment, and politics field share information in images. In the image-based sentiment analysis, opinions depict in the form of cartoons or memes. In most cases, the information conveyed through images is more effective compared to other modalities. Multiple techniques and algorithms such as SVM, naive Bayes, maximum entropy, and deep learning have been proposed in the image-based sentiment area to get significant results. The first work introduced by (Mikels et al, 2005) to classify the images into positive and negative. The author showed that there is a strong correlation between sentiment images and their visual content.
Further, the SentiWordNet lexicon (Ohana and Tierney, 2009) was used to find the text’s numerical scores associated with the image. This lexicon is used in WorldNet databases to identify the positive and negative sentiment of the word. Emotions are difficult to identify and pin down when discussing the state of the emotion that differentiates from other emotional states. To find the scientific approach regarding the emotional state of the human being. The database of different photos was collected and validated against the specific emotional response of the viewers. This database is called International Affective Picture System (IAPS). Mikels et al. (Mikels et al, 2005) studied eight emotion output categories: awe, anger, amusement, contentment, excitement, disgust, sadness, and fear. The author showed that each emotional state is different as different emotions have other cognitive and behavioral consequences. This paper adds some new dimensions of data for IAPS.

### 2.4 Multimodal Sentiment Analysis

Multimodal sentiment analysis (Lakomkin et al, 2019) performs sentiment analysis from multiple data such as audio, video, and text. It is the new dimension of traditional text-based sentiment analysis. Poria et al. (Poria et al, 2016b) proposed a new multimodal sentiment analysis methodology, which outperformed state of the art by more than 20%. The proposed system used feature-based fusion techniques on text, visual and audio data from the youtube dataset. Different classifiers such as Naive Bayes, SVM, and extreme learning machines are implemented on the youtube dataset. The results showed that the extreme learning classifier is better than other classifiers. Extreme learning classifiers have single layer or multiple layers of hidden nodes. In most cases, the weights of the hidden nodes are learned in a single step so the overall processing time to classifying the result is less.

Kumar et al. (Kumar et al, 2019) proposed a multimodal rating prediction framework for products to improve customer satisfaction. The forty participants were participated in this study to collect EEG data of the product. The text’s reviews from the product are processed through NLP techniques. The customer’s rating from EEG and the product’s reviews fused through optimization techniques. The experiment result showed that the ABC optimization approach was better than the unimodal scheme. There are many languages other than English, where researchers are working to predict the sentiment (Xu et al, 2019; Yang et al, 2020; Behera et al, 2021; Zhao et al, 2020; Williams et al, 2018). In (Khasawneh et al, 2015), the authors proposed a hybrid approach on the Arabic dataset (Text and audio). Two machine learning approaches were used on this dataset to find the polarity. The bagging and boosting algorithms were used to enhance the proposed system further. The summary of the reviewed articles is as shown in Table 1.

*Table 1: Summary of the publication’s details included author, approach, data set, and accuracy.*

| Author & Year | Approach | Dataset | Accuracy (%) |

| Hearst et al. [14], 1992 | Cognitive Linguistics | User’s Query | - |

| Wiebet et al. [21], 1999 | Probabilistic Classifier | Gold-standard | - |

| Camera reviews, |

| news articles |

75-95

Pang et al. [23], 2008

| Supervised and Unsupervised |

| approach (Survey paper) |

- -

Tan et al. [11], 2011

| Domain-Oriented |

| sentiment lexicon |

| Electronics reviews, |

| Stock reviews |

| and Hotel review |

82.9

Fang et al. [9], 2015

| sentiment polarity |

| categorization process |

| amazon product |

| reviews |

80

Gallege et al. [8], 2016

| Trust-based ranking |

| and the recommendation |

| Amazon |

| marketplace |

| review |

-

Xia et al. [10], 2015

| Naive Bayes, linear SVM, |

| logistic regression |

| Multi-Domain and |

| Chinese dataset |

90

Khasawneh et al. [12], 2015 Bagging and Boosting

| 1500 Arabic |

| comments and |

| Twitter reviews |

-

Campos et al. [65], 2017 CNN Twitter images -

Poria et al. [13], 2016 ELM classifier YouTube Dataset -

Cambria et al. [1], 2017 Top-Down and Bottom-Up

| Penn Treebank, |

| LIWC |

-

Kumar et al. [4], 2019 ABC optimization

| EEG data and |

| product reviews |

-

Qazi et al. [47], 2017

| Expectancy disconfirmation |

| theory and Confirmatory |

| factor analysis |

| LinkedIn and |

| the university mail |

| servers’ groups |

-

Wang et al. [48], 2018 SentiRelated

| Raw Data and |

| Douban Data |

80

Williams et al. [48], 2018

| intermediate-level |

| feature fusion |

| MOSI dataset |

74.0

Yang et al. [48], 2020 SLCABG

| book reviews |

| collected from |

| Dangdang dataset |

| Accuracy 93.5 |

| Precision 93 |

| Recall 93.6 |

| F1 93.3 |

Xu et al. [48], 2019 Seninfo+TF-IDF

| 15000 hotel |

| comment texts |

| Precision 91.54 |

| Recall 92.82 |

| F1 92.18 |

Lakomkin et al. [48], 2019 ASR model

| Multimodal Corpus of |

| Sentiment Intensity |

| 73.6 |

Guo et al. [48], 2022

| CNN-BiGRU-CTC + |

| ERNIE-BiLSTM |

| Aishell-1 |

| and NLPCC 2014 |

| 94.5 |

Kumar et al. [48], 2022

| BiLSTM + GloVe |

| IIT-R STSA |

| 92.83 |

Tian et al. [48], 2021

| BERT-LARGE + A-KVMN |

| with second-order |

| word dependencies |

| LAP14, REST14, |

| REST15, REST16 |

| and Twitter |

| 92.48 |

Behera et al. [48], 2021

| Co-LSTM model |

| Movie review, |

| Airline dataset |

| Self driving car GOP |

| 98.40 |

Zhao et al. [48], 2020

| Attention-based |

| LSTM model |

| Facebook corpus |

| containing user |

| personality tag |

| Precision 57.95 |

| Recall 65.78 |

| F1 72.2 |

Derakhshan et al. [48], 2019

| LDA-POS model |

| English and Persian |

| English dataset |

| average 56.24 |

| Persian dataset |

| average 55.33 |

There are many public datasets (Guo et al, 2022; Tian et al, 2021) available in SA. For different application such as text, images, audio and video, we use different datasets or create a dataset like Sanskrit dataset (Kumar et al, 2022). The tools and software library is also depend on the multimodal data. Open CV is a open source library used in computer vision tasks for object dection, face recognition and image segmentation. NLTK is a python library used for understanding the text or speech. Below is some popular database as shown in Table 2.

*Table 2: List of most popular public datasets available in the sentiment analysis field in different languages and modalities. The details included the source and uses of the dataset like the movie reviews, product reviews, social media data, etc.*

| Research Paper | Dataset | Use of this dataset | Volume |

| Bai et al. (Bai, 2011) | IMDB Movie | To analysis movie review | 50,000 movie reviews |

| To do the sentiment analysis |

| of tweets of different domain |

| Google 640917, |

| Microsoft 161292 |

| and Sony 141529 |

| tweets |

Araque et al. (Araque et al, 2017) Sentiment140

| Sentiment analysis of tweets |

| for a product or brand |

1.6 million tweets

Dredze et al. (Blitzer et al, 2007)

| Amazon Product |

| Reviews |

| To classify user review in |

| positive and negative |

142.8 million review

Qian et al. (Lei et al, 2016) Restaurant Reviews

| To find the aspect based |

| sentiment analysis |

| 3 million restaurant |

| reviews |

Karyotis et al. (Karyotis et al, 2018) Facebook

| To do the sentiment analysis |

| of Facebook post |

| million Facebook users |

| and their posts |

Chen et al. (Xu et al, 2014) Flicker images To classify the image

| 470 positive tweets |

| and 133 negative tweets, |

| Tumblr 1179 |

Yang et al. (Yang et al, 2018) IAPS, Instagram visual sentiment prediction

| IAPS 395, |

| Instagram 23308 |

Yang et al. (Yang et al, 2019) SemEval 2014 Aspect-based sentiment analysis

| Restaurants 3841, |

| Laptops 3845 |

Zhang et al. (Zhang et al, 2018) SemEval 2016 Sentiment analysis track

| Positive 3094, |

| Negative 2043 |

| and Neutral 863 tweets |

Schmitt et al. (Schmitt et al, 2018) SemEval 2017

| Detecting sentiment, |

| humour, and truth |

8000-10000 tweets

Joshi et al. (Joshi et al, 2010) Hindi Movie Reviews

| Sentiment analysis in Hindi |

250 Hindi
Movie Reviews

Xu et al. (Xu et al, 2019)

| Reviews of hotel |

| clothes, fruit |

| digital etc |

| Chinese Text Sentiment |

| Analysis |

2,50000 reviews

Stappen et al. (Stappen et al, 2021)

| MuSe-CAR |

| The Multimodal |

| Sentiment Analysis |

| in Car Reviews |

15 GB Audio, Video, Text

Latif et al. (Latif et al, 2018)

| URDU-Dataset |

| 4 emotions: angry, |

| happy, neutral, and sad. |

0.072 GB Audio

Duville et al. (Duville et al, 2021)

| MESD |

| 6 emotions provides |

| single-word utterances |

| for anger, disgust, fear |

| happiness, neutral, and sadness. |

0,097 GB Audio

## 3 Usage and Application of Sentiment Analysis

This section covers the wide range of applications of sentiment analysis in various emerging areas.

### 3.1 Reviews from E-commerce and Microblogging Sites

We have an extensive collection of data sets available on almost everything over the internet. It includes user comments, reviews, feedback on various topics, opinions drawn using surveys, products on e-commerce websites (Haque et al, 2018), customer services (Kang and Park, 2014), and recently Twitter data in the form of tweets on ongoing COVID-19 (Abd-Alrazaq et al, 2020; Manguri et al, 2020; Alamoodi et al, 2021; Chakraborty et al, 2020; Naseem et al, 2021) pandemic. Therefore, there is a severe demand for a system based on sentiment analysis that can extract sentiments about a particular product, item, or service. It will help us to automate the user feedback or customer rating for the given product, services, etc. This would help to improve the product and offered services and eventually serve both the buyer and seller’s requirements.

### 3.2 Business Intelligence

Nowadays, consumers are getting more intelligent (Sreesurya et al, 2020), quality-conscious, and technical savvy; therefore, they tend to seek out the reviews and ratings of online products and services before buying them. Many companies like Uber (Baj-Rogowska, 2017), Oyo (Shanmugam and Padmanaban, 2020), and zomato (Gupta et al, 2021) use digital transformation models to take feedback from the customer. The online customer opinion decides the success or failure of their offered services and products. The companies demand to extract sentiment from the online user reviews to enhance their offered products and services. It also helps companies to launch their new products and services in the new market for target customers. Therefore, It is evident that sentiment analysis plays a vital role in getting the customer and competition insights, which help companies make corrective and preventive actions to sustain and grow their businesses in the digital era.

### 3.3 Global Financial Market

SA is also helpful in the share market and the Federal open market committee (FOMC ) statement (Tadle, 2022; Bhandari, 2022; Doh et al, 2021) to extract the meaningful information for traders through which they understand the global financial markets . Some interesting trends reveal in Figure 6, Figure 7 and Figure 8 through sentiment analysis of FOMC statements.

*Figure 6: The federal open market committee (FOMC) controls the monetary policy of the central bank. The FOMC’s statement lexical frequency list (most popular word in the report) of July and September 2017.*

*Figure 7: In the list of positive and negative words in the FOMC’s statement of July and September 2017, the words in bold font denote the negative word while the other indicates the positive word.*

*Figure 8: Most commonly used words in the FOMC statements since 2012, where n is the occurrence of words. In this chart, the most frequent word has shown from top to bottom; the top word in the chart has a higher occurrence than the last word.*

### 3.4 Applications in Smart Homes

Smart homes are an emerging technology, and in the near future, the entire home will be more secure and better connected with other home appliances. The people would control and manage any part of the house using smart wearable devices such as apple watch, intelligent assistant devices such as Alexa (Bogdan et al, 2021; Gao et al, 2018), Google Home (Sánchez-Franco et al, 2021; Park and Kim, 2018), etc. Recently there has been a lot of research going on in the Internet of Things (IoT) and the SA. The SA also found its way in IoT, e.g., the connected home using smart devices such as smart bulbs, smart music devices could alter its ambiance to create a calming and comfortable environment based on the sentiment or emotion of the user.

### 3.5 Detection of Hate Speech

Bigotry speech (Gitari et al, 2015; Badjatiya et al, 2017; Rodriguez et al, 2019) is used to express repugnance towards a specifically intended community, group, or person that can cause a dangerous situation to the victim. It can also be used to demean or offend particular community members or groups on any social media. SA based detection system would help the social media companies such as Twitter (Jiang and Suzuki, 2019), Instagram (Naf’an et al, 2019), etc., instant messaging companies such as WhatsApp (Deb et al, 2020), Telegram and local enforcement and government to suppress hate speech and fake news towards a specific person, sex, religion, race, country, etc. which in turn improve their reputation and bring harmony in the community.

### 3.6 Emotion detection in suicide notes

In modern society, suicides are rising rapidly in recent times; it is critical to find a faster way to fine-grained emotion detection (Ghosh et al, 2022; Desmet and Hoste, 2013; Prasad et al, 2018) and anxiety in online posts, microblogging text in the form of tweets by these troubled individuals. The SA-based detection and analysis system may help to detect such tendencies upfront and prevent suicides.

### 3.7 Stress Detection

On the flip side of excessive competition, improving the living style in a fast-moving world, people typically face many changes from their work environment, eating habits, etc. The body reacts to these stress changes, influencing an individual’s emotional, mental, and physical health. The SA based detection system may help to detect stress symptoms (Wang et al, 2013; Jung et al, 2017) upfront and prevent any adverse impact due to this.

## 4 Challenges and Perspectives

The SA is a particularly challenging task for human behaviors and subjective sentiments. Below are a few of the challenges-

### 4.1 Recognizing Subjective Parts of The Phrase

The English language can sometimes be tricky. Homonyms, or multiple-meaning words, have the same spelling and usually sound alike but have different meanings. Subjective parts in the phrase or sentence epitomize sentiment-related content. The Homonyms in the phrase might be treated as subjective in one case or objective in some other. It brings it challenging to identify the subjective portions of the phrase. For example: 1. The new lamp had good light for reading. 2. Magnesium is a light metal. The word light is used to mean a particular quality or type of light in the first phase, whereas the light word objectively means having a relatively low density in the second phrase. Users share views or opinions over the internet on different social media platforms. Different age groups and gender share information or opinion in their way, recent study (Kumar et al, 2020) prove that older people people share their opinion in a better way instead of young ones.

### 4.2 Dependence on The Domains

The same phrase might have different interpretations in different domains in which it is being used. For Example, the word ’unpredictable’ is positive in entertainment and theater, etc., but if the same word is used in the context of an automobile’s break, it has a negative opinion. Still, this is challenging to identify the domain from which any word is related correctly. Different pre-trained word embedding corpus domains such as IMDB movie reviews corpus and customer reviews dataset classify the sentence correctly. This challenge is still not solved completely, and researchers are continuously working on this problem.

### 4.3 Detection of Sarcasm in The Phrase

Sarcastic sentences express a negative opinion about a person or thing using positive words in unique. Often, people use it to say the opposite of what’s true to make someone look or feel foolish. For Example: -” Good perfume. You must marinate in it for long”. The sentence has only positive words, but it expresses a negative sentiment.

### 4.4 Dependence on The Order

Discourse Structure analysis is essential for opinion mining and sentiment analysis. For Example, A is better than B conveys the exact opposite opinion from B is better than A. For finding SA for these kind of sentence is quite challenging.

### 4.5 Idioms

ML programs are designed so that they don’t understand a figure of speech. For example, language such as ”not my cup of tea” will disrupt the algorithm because it understands the things literally. When any user uses idioms in a comment or review, the sentence interpretation is not correctly map by the algorithm. The situation is even more difficult if the comment is multilingual.

### 4.6 Multilingual sentiment analysis

User share their opinion in different languages like Hinglish which is the combination of Hindi and English. Every language has its own lemmatizer, POS tagger and grammatical constructs so ML or Deep learning algorithm understand the context and classify the comment in positive and negative. The real challenges is that we can not translate multiple language into one base language. Usually in micro-blogging or chatting, user share their feeling in multilingual.

Despite different challenges in sentiment analysis still, it is an emerging field among customers for decision-making. Figure 9 and 10 presents users’ most popular topic and query search from 2004 to 2020.

*Figure 9: The top 25 topics search by the user worldwide from 2004 to 2020 in the sentiment analysis field. The most frequent topics search by the users is analysis and opinion.*

*Figure 10: The most frequently searched query worldwide from 2004 to 2020 in the sentiment analysis field. The sentiment, sentiment analysis, and Twitter sentiment are the top three search queries worldwide.*

## 5 Discussion Towards ML and DL Techniques on Sentiment Analysis Field

In the last decade, the paradigm shifted from machine learning to deep learning techniques. In-text data, the context problem is a big challenge to understand the sentence’s meaning through the ML algorithm correctly. This problem solves through pre-trained word embedding and the VADER approach even we have a smaller training dataset. However, the pre-trained word embedding corpus was trained on the google news dataset (100 billion words) and IMDB movie dataset. It shows a good result when the data are related to the pre-trained corpus domain; otherwise, it will not predict the result as expected. The BERT model is the start of the art model in NLP. It uses the bidirectional training of the input, which provides a more profound sense of the language context. However, it is very compute-intensive and takes time to predict the result. ML techniques are also predicted good results, depending on the dataset and nature of data. DL techniques predict a good outcome for a large dataset.

The present study covered different domains like text, speech, image, and video to analyze sentiment. In all domains, the start of the art algorithms and papers were discussed in the study.

## 6 Conclusion and Future Work

With the advancement of technology in machine learning and deep learning, the SA plays a vital role in analyzing data available on the internet in text, image, and speech. The SA is computationally identifying the polarity of text into a positive, negative, and neutral review. In this survey paper, we have investigated the history of the SA and its impact on the research community from the years 2000 to current trends. In the last five years, most articles are related to social media such as Facebook, Instagram, and Twitter. Most articles are related to the application area of health, restaurant, travel, spam, and politics. We have also included the top-cited paper and discuss the research challenges and perspectives suitable for new researchers who want to start research in the ML, NLP, and SA fields. We also cover in detail about global finacial market (FOMC), different languages like Sanskrit, Hindi, Arabic, Chinese etc and modalities in which many authors used SA. In future work, the SA is combined with network traffic to detect fake opinion or news, which creates a serious problem, resulting in mob violence. The method to do the SA will also improve with the continuous advancement of the NLP and ML fields.

## Compliance with ethical standards

Conflict of interest The authors declared that they have no conflicts of interest to this work.

## References

- Abd-Alrazaq et al (2020) Abd-Alrazaq A, Alhuwail D, Househ M, Hamdi M, Shah Z (2020) Top concerns of tweeters during the covid-19 pandemic: infoveillance study. Journal of medical Internet research 22(4):e19016

- Ahmed et al (2016) Ahmed A, Toral S, Shaalan K (2016) Agent productivity measurement in call center using machine learning. In: International Conference on Advanced Intelligent Systems and Informatics, Springer, pp 160–169

- Al-Ayyoub et al (2019) Al-Ayyoub M, Khamaiseh AA, Jararweh Y, Al-Kabi MN (2019) A comprehensive survey of arabic sentiment analysis. Information processing & management 56(2):320–342

- Al-Radaideh and Al-Qudah (2017) Al-Radaideh QA, Al-Qudah GY (2017) Application of rough set-based feature selection for arabic sentiment analysis. Cognitive Computation 9(4):436–445

- Alamoodi et al (2021) Alamoodi AH, Zaidan BB, Zaidan AA, Albahri OS, Mohammed K, Malik RQ, Almahdi EM, Chyad MA, Tareq Z, Albahri AS, et al (2021) Sentiment analysis and its applications in fighting covid-19 and infectious diseases: A systematic review. Expert systems with applications 167:114155

- Anto et al (2016) Anto MP, Antony M, Muhsina KM, Johny N, James V, Wilson A (2016) Product rating using sentiment analysis. In: Proceedings of the International Conference on Electrical, Electronics, and Optimization Techniques (ICEEOT), pp 3458–3462

- Araque et al (2017) Araque O, Corcuera-Platas I, Sánchez-Rada JF, Iglesias CA (2017) Enhancing deep learning sentiment analysis with ensemble techniques in social applications. Expert Systems with Applications 77:236–246

- Badarneh et al (2018) Badarneh O, Al-Ayyoub M, Alhindawi N, Jararweh Y, et al (2018) Fine-grained emotion analysis of arabic tweets: A multi-target multi-label approach. In: 2018 IEEE 12th International Conference on Semantic Computing (ICSC), IEEE, pp 340–345

- Badjatiya et al (2017) Badjatiya P, Gupta S, Gupta M, Varma V (2017) Deep learning for hate speech detection in tweets. In: Proceedings of the 26th international conference on World Wide Web companion, pp 759–760

- Bai (2011) Bai X (2011) Predicting consumer sentiments from online text. Decision Support Systems 50(4):732–742

- Baj-Rogowska (2017) Baj-Rogowska A (2017) Sentiment analysis of facebook posts: The uber case. In: 2017 Eighth International Conference on Intelligent Computing and Information Systems (ICICIS), IEEE, pp 391–395

- Behera et al (2021) Behera RK, Jena M, Rath SK, Misra S (2021) Co-lstm: Convolutional lstm model for sentiment analysis in social big data. Information Processing & Management 58(1):102435

- Bengio et al (2003) Bengio Y, Ducharme R, Vincent P, Jauvin C (2003) A neural probabilistic language model. Journal of machine learning research 3(Feb):1137–1155

- Bhandari (2022) Bhandari P (2022) Sentiment analysis of fomc meeting transcripts: Pre and post mexican pesos crisis

- Blitzer et al (2007) Blitzer J, Dredze M, Pereira F (2007) Biographies, bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In: Proceedings of the 45th annual meeting of the association of computational linguistics, pp 440–447

- Bogdan et al (2021) Bogdan R, Tatu A, Crisan-Vida MM, Popa M, Stoicu-Tivadar L (2021) A practical experience on the amazon alexa integration in smart offices. Sensors 21(3):734

- Cambria et al (2016) Cambria E, Poria S, Bajpai R, Schuller B (2016) Senticnet 4: A semantic resource for sentiment analysis based on conceptual primitives. In: Proceedings of COLING 2016, the 26th international conference on computational linguistics: Technical papers, pp 2666–2677

- Cambria et al (2017) Cambria E, Poria S, Gelbukh A, Thelwall M (2017) Sentiment analysis is a big suitcase. IEEE Intelligent Systems 32(6):74–80

- Campos et al (2017) Campos V, Jou B, Giro-i Nieto X (2017) From pixels to sentiment: Fine-tuning cnns for visual sentiment prediction. Image and Vision Computing 65:15–22

- Chakraborty et al (2020) Chakraborty K, Bhatia S, Bhattacharyya S, Platos J, Bag R, Hassanien AE (2020) Sentiment analysis of covid-19 tweets by deep learning classifiers—a study to show how popularity is affecting accuracy in social media. Applied Soft Computing 97:106754

- Chen et al (2017) Chen T, Xu R, He Y, Wang X (2017) Improving sentiment analysis via sentence type classification using bilstm-crf and cnn. Expert Systems with Applications 72:221–230

- Crouch and Khosla (2012) Crouch S, Khosla R (2012) Sentiment analysis of speech prosody for dialogue adaptation in a diet suggestion program. ACM SIGHIT Record 2(1):8–8

- Das and Chen (2001) Das S, Chen M (2001) Yahoo! for amazon: Extracting market sentiment from stock message boards. In: Proceedings of the Asia Pacific finance association annual conference (APFA), Bangkok, Thailand, vol 35, p 43

- Das and Chen (2007) Das SR, Chen MY (2007) Yahoo! for amazon: Sentiment extraction from small talk on the web. Management science 53(9):1375–1388

- Dasgupta et al (2015) Dasgupta SS, Natarajan S, Kaipa KK, Bhattacherjee SK, Viswanathan A (2015) Sentiment analysis of facebook data using hadoop based open source technologies. In: Proceedings of the IEEE International Conference on Data Science and Advanced Analytics (DSAA), pp 1–3

- Day and Lee (2016) Day MY, Lee CC (2016) Deep learning for financial sentiment analysis on finance news providers. In: Proceedings of the IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), pp 1127–1134

- Deb et al (2020) Deb K, Paul S, Das K (2020) A framework for predicting and identifying radicalization and civil unrest oriented threats from whatsapp group. In: Emerging Technology in Modelling and Graphics, Springer, pp 595–606

- Dellaert et al (1996) Dellaert F, Polzin T, Waibel A (1996) Recognizing emotion in speech. In: Proceeding of Fourth International Conference on Spoken Language Processing. ICSLP’96, IEEE, vol 3, pp 1970–1973

- Desmet and Hoste (2013) Desmet B, Hoste V (2013) Emotion detection in suicide notes. Expert Systems with Applications 40(16):6351–6358

- Devlin et al (2018) Devlin J, Chang MW, Lee K, Toutanova K (2018) Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:181004805

- Doh et al (2021) Doh T, Kim S, Yang SK (2021) How you say it matters: Text analysis of fomc statements using natural language processing. Economic Review-Federal Reserve Bank of Kansas City 106(1):25–40

- Du and Huang (2018) Du C, Huang L (2018) Text classification research with attention-based recurrent neural networks. International Journal of Computers Communications & Control 13(1):50–61

- Duville et al (2021) Duville MM, Alonso-Valerdi LM, Ibarra-Zarate DI (2021) The mexican emotional speech database (mesd): elaboration and assessment based on machine learning. In: 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), IEEE, pp 1644–1647

- Elmadany et al (2018) Elmadany A, Mubarak H, Magdy W (2018) Arsas: An arabic speech-act and sentiment corpus of tweets. OSACT 3:20

- Fang and Zhan (2015) Fang X, Zhan J (2015) Sentiment analysis using product review data. Journal of Big Data 2(1):5

- Gallege and Raje (2016) Gallege LS, Raje RR (2016) Towards selecting and recommending online software services by evaluating external attributes. In: Proceedings of the 11th Annual Cyber and Information Security Research Conference (CISRC), pp 23:1-23:4

- Gao et al (2018) Gao Y, Pan Z, Wang H, Chen G (2018) Alexa, my love: analyzing reviews of amazon echo. In: 2018 IEEE SmartWorld, Ubiquitous Intelligence & Computing, Advanced & Trusted Computing, Scalable Computing & Communications, Cloud & Big Data Computing, Internet of People and Smart City Innovation (SmartWorld/SCALCOM/UIC/ATC/CBDCom/IOP/SCI), IEEE, pp 372–380

- Ghosh et al (2022) Ghosh S, Ekbal A, Bhattacharyya P (2022) Deep cascaded multitask framework for detection of temporal orientation, sentiment and emotion from suicide notes. Scientific reports 12(1):1–16

- Giatsoglou et al (2017) Giatsoglou M, Vozalis MG, Diamantaras K, Vakali A, Sarigiannidis G, Chatzisavvas KC (2017) Sentiment analysis leveraging emotions and word embeddings. Expert Systems with Applications 69:214–224

- Gitari et al (2015) Gitari ND, Zuping Z, Damien H, Long J (2015) A lexicon-based approach for hate speech detection. International Journal of Multimedia and Ubiquitous Engineering 10(4):215–230

- Guo et al (2022) Guo H, Zhan X, Chi C (2022) Multiple scene sentiment analysis based on chinese speech and text. Journal of Computers 33(1):165–178

- Gupta et al (2021) Gupta R, Sameer S, Muppavarapu H, Enduri MK, Anamalamudi S (2021) Sentiment analysis on zomato reviews. In: 2021 13th International Conference on Computational Intelligence and Communication Networks (CICN), IEEE, pp 34–38

- Haque et al (2018) Haque TU, Saber NN, Shah FM (2018) Sentiment analysis on large scale amazon product reviews. In: 2018 IEEE international conference on innovative research and development (ICIRD), IEEE, pp 1–6

- Hearst (1992) Hearst MA (1992) Direction-based text interpretation as an information access refinement. Text-based intelligent systems: Current research and practice in information extraction and retrieval pp 257–274

- Hochreiter and Schmidhuber (1997) Hochreiter S, Schmidhuber J (1997) Long short-term memory. Neural computation 9(8):1735–1780

- Hutto and Gilbert (2014) Hutto CJ, Gilbert E (2014) Vader: A parsimonious rule-based model for sentiment analysis of social media text. In: Eighth international AAAI conference on weblogs and social media

- Jiang and Suzuki (2019) Jiang L, Suzuki Y (2019) Detecting hate speech from tweets for sentiment analysis. In: 2019 6th International Conference on Systems and Informatics (ICSAI), IEEE, pp 671–676

- Joshi et al (2010) Joshi A, Balamurali A, Bhattacharyya P, et al (2010) A fall-back strategy for sentiment analysis in hindi: a case study. Proceedings of the 8th ICON

- Jung et al (2017) Jung H, Park HA, Song TM, et al (2017) Ontology-based approach to social data sentiment analysis: detection of adolescent depression signals. Journal of medical internet research 19(7):e7452

- Kang and Park (2014) Kang D, Park Y (2014) based measurement of customer satisfaction in mobile service: Sentiment analysis and vikor approach. Expert Systems with Applications 41(4):1041–1050

- Karyotis et al (2018) Karyotis C, Doctor F, Iqbal R, James A, Chang V (2018) A fuzzy computational model of emotion for cloud based sentiment analysis. Information Sciences 433:448–463

- Khasawneh et al (2015) Khasawneh RT, Wahsheh HA, Alsmadi IM, AI-Kabi MN (2015) Arabic sentiment polarity identification using a hybrid approach. In: 2015 6th International Conference on Information and Communication Systems (ICICS), IEEE, pp 148–153

- Kumar et al (2022) Kumar P, Pathania K, Raman B (2022) Zero-shot learning based cross-lingual sentiment analysis for sanskrit text with insufficient labeled data. Applied Intelligence pp 1–18

- Kumar et al (2019) Kumar S, Yadava M, Roy PP (2019) Fusion of eeg response and sentiment analysis of products review to predict customer satisfaction. Information Fusion 52:41–52

- Kumar et al (2020) Kumar S, Gahalawat M, Roy PP, Dogra DP, Kim BG (2020) Exploring impact of age and gender on sentiment analysis using machine learning. Electronics 9(2):374

- Lakomkin et al (2019) Lakomkin E, Zamani MA, Weber C, Magg S, Wermter S (2019) Incorporating end-to-end speech recognition models for sentiment analysis. In: 2019 International Conference on Robotics and Automation (ICRA), IEEE, pp 7976–7982

- Latif et al (2018) Latif S, Qayyum A, Usman M, Qadir J (2018) Cross lingual speech emotion recognition: Urdu vs. western languages. In: 2018 International Conference on Frontiers of Information Technology (FIT), IEEE, pp 88–93

- Lei et al (2016) Lei X, Qian X, Zhao G (2016) Rating prediction based on social sentiment from textual reviews. IEEE transactions on multimedia 18(9):1910–1921

- Levallois (2013) Levallois C (2013) Sentiment analysis for tweets based on lexicons an heuristics

- Li and Li (2013) Li YM, Li TY (2013) Deriving market intelligence from microblogs. Decision Support Systems 55(1):206–217

- Ma et al (2016) Ma Y, Cambria E, Gao S (2016) Label embedding for zero-shot fine-grained named entity typing. In: Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pp 171–180

- Ma et al (2018) Ma Y, Peng H, Cambria E (2018) Targeted aspect-based sentiment analysis via embedding commonsense knowledge into an attentive lstm. In: Thirty-second AAAI conference on artificial intelligence

- Mairesse et al (2012) Mairesse F, Polifroni J, Di Fabbrizio G (2012) Can prosody inform sentiment analysis? experiments on short spoken reviews. In: 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, pp 5093–5096

- Manguri et al (2020) Manguri KH, Ramadhan RN, Amin PRM (2020) Twitter sentiment analysis on worldwide covid-19 outbreaks. Kurdistan Journal of Applied Research pp 54–65

- Mikels et al (2005) Mikels JA, Fredrickson BL, Larkin GR, Lindberg CM, Maglio SJ, Reuter-Lorenz PA (2005) Emotional category data on images from the international affective picture system. Behavior research methods 37(4):626–630

- Mikolov et al (2010) Mikolov T, Karafiát M, Burget L, Černockỳ J, Khudanpur S (2010) Recurrent neural network based language model. In: Eleventh annual conference of the international speech communication association

- Mikolov et al (2013a) Mikolov T, Chen K, Corrado G, Dean J (2013a) Efficient estimation of word representations in vector space. arXiv preprint arXiv:13013781

- Mikolov et al (2013b) Mikolov T, Sutskever I, Chen K, Corrado GS, Dean J (2013b) Distributed representations of words and phrases and their compositionality. In: Advances in neural information processing systems, pp 3111–3119

- Morinaga et al (2002) Morinaga S, Yamanishi K, Tateishi K, Fukushima T (2002) Mining product reputations on the web. In: Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pp 341–349

- Munezero et al (2014) Munezero MD, Montero CS, Sutinen E, Pajunen J (2014) Are they different? affect, feeling, emotion, sentiment, and opinion detection in text. IEEE transactions on affective computing 5(2):101–111

- Naf’an et al (2019) Naf’an MZ, Bimantara AA, Larasati A, Risondang EM, Nugraha NAS (2019) Sentiment analysis of cyberbullying on instagram user comments. Journal of Data Science and Its Applications 2(1):38–48

- Naseem et al (2021) Naseem U, Razzak I, Khushi M, Eklund PW, Kim J (2021) Covidsenti: A large-scale benchmark twitter data set for covid-19 sentiment analysis. IEEE Transactions on Computational Social Systems 8(4):1003–1015

- Nasukawa and Yi (2003) Nasukawa T, Yi J (2003) Sentiment analysis: Capturing favorability using natural language processing. In: Proceedings of the 2nd international conference on Knowledge capture, pp 70–77

- Ohana and Tierney (2009) Ohana B, Tierney B (2009) Sentiment classification of reviews using sentiwordnet. In: 9th. it & t conference, vol 13, pp 18–30

- Pandey et al (2017) Pandey AC, Rajpoot DS, Saraswat M (2017) Twitter sentiment analysis using hybrid cuckoo search method. Information Processing & Management 53(4):764–779

- Pang and Lee (2008) Pang B, Lee L (2008) Opinion mining and sentiment analysis. Foundations and trends in information retrieval 2(1-2):1–135

- Pang et al (2002) Pang B, Lee L, Vaithyanathan S (2002) Thumbs up?: sentiment classification using machine learning techniques. In: Proceedings of the ACL-02 conference on Empirical methods in natural language processing-Volume 10, Association for Computational Linguistics, pp 79–86

- Park and Kim (2018) Park H, Kim JH (2018) Perception of virtual assistant and smart speaker: Semantic network analysis and sentiment analysis. In: Proceedings of the Korean Institute of Information and Commucation Sciences Conference, The Korea Institute of Information and Commucation Engineering, pp 213–216

- Pennington et al (2014) Pennington J, Socher R, Manning CD (2014) Glove: Global vectors for word representation. In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp 1532–1543

- Peters et al (2018) Peters ME, Neumann M, Iyyer M, Gardner M, Clark C, Lee K, Zettlemoyer L (2018) Deep contextualized word representations. arXiv preprint arXiv:180205365

- Poria et al (2016a) Poria S, Cambria E, Hazarika D, Vij P (2016a) A deeper look into sarcastic tweets using deep convolutional neural networks. In: Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pp 1601–1612

- Poria et al (2016b) Poria S, Cambria E, Howard N, Huang GB, Hussain A (2016b) Fusing audio, visual and textual clues for sentiment analysis from multimodal content. Neurocomputing 174:50–59

- Poria et al (2017) Poria S, Cambria E, Bajpai R, Hussain A (2017) A review of affective computing: From unimodal analysis to multimodal fusion. Information Fusion 37:98–125

- Prasad et al (2018) Prasad DK, Liu S, Chen SHA, Quek C (2018) Sentiment analysis using eeg activities for suicidology. Expert Systems with Applications 103:206–217

- Qazi et al (2017) Qazi A, Tamjidyamcholo A, Raj RG, Hardaker G, Standing C (2017) Assessing consumers’ satisfaction and expectations through online opinions: Expectation and disconfirmation approach. Computers in Human Behavior 75:450–460

- Radford et al (2018) Radford A, Narasimhan K, Salimans T, Sutskever I (2018) Improving language understanding with unsupervised learning. Technical report, OpenAI

- Ren et al (2016) Ren Y, Wang R, Ji D (2016) A topic-enhanced word embedding for twitter sentiment classification. Information Sciences 369:188–198

- Rodriguez et al (2019) Rodriguez A, Argueta C, Chen YL (2019) Automatic detection of hate speech on facebook using sentiment and emotion analysis. In: 2019 international conference on artificial intelligence in information and communication (ICAIIC), IEEE, pp 169–174

- Sack (1994) Sack W (1994) On the computation of point of view. In: AAAI, p 1488

- Saif et al (2016) Saif H, He Y, Fernandez M, Alani H (2016) Contextual semantics for sentiment analysis of twitter. Information Processing & Management 52(1):5–19

- Sánchez-Franco et al (2021) Sánchez-Franco MJ, Arenas-Márquez FJ, Alonso-Dos-Santos M (2021) Using structural topic modelling to predict users’ sentiment towards intelligent personal agents. an application for amazon’s echo and google home. Journal of Retailing and Consumer Services 63:102658

- Sariyanidi et al (2014) Sariyanidi E, Gunes H, Cavallaro A (2014) Automatic analysis of facial affect: A survey of registration, representation, and recognition. IEEE transactions on pattern analysis and machine intelligence 37(6):1113–1133

- Scherer (2005) Scherer KR (2005) What are emotions? and how can they be measured? Social science information 44(4):695–729

- Schmitt et al (2018) Schmitt M, Steinheber S, Schreiber K, Roth B (2018) Joint aspect and polarity classification for aspect-based sentiment analysis with end-to-end neural networks. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp 1109–1114

- Schuller et al (2011) Schuller B, Batliner A, Steidl S, Seppi D (2011) Recognising realistic emotions and affect in speech: State of the art and lessons learnt from the first challenge. Speech Communication 53(9-10):1062–1087

- Shanmugam and Padmanaban (2020) Shanmugam S, Padmanaban I (2020) Twitter emotion analysis for brand comparison using naive bayes classifier. In: International Conference on Soft Computing and its Engineering Applications, Springer, pp 199–211

- Socher et al (2013) Socher R, Perelygin A, Wu J, Chuang J, Manning CD, Ng AY, Potts C (2013) Recursive deep models for semantic compositionality over a sentiment treebank. In: Proceedings of the 2013 conference on empirical methods in natural language processing, pp 1631–1642

- Sreesurya et al (2020) Sreesurya I, Rathi H, Jain P, Jain TK (2020) Hypex: A tool for extracting business intelligence from sentiment analysis using enhanced lstm. Multimedia Tools and Applications 79(47):35641–35663

- Stappen et al (2021) Stappen L, Baird A, Schumann L, Bjorn S (2021) The multimodal sentiment analysis in car reviews (muse-car) dataset: Collection, insights and improvements. IEEE Transactions on Affective Computing

- STONE (1997) STONE J (1997) Thematic text analysis-new agendas for analyzing text content. Test analysis for the social sciences-Methods for drawing statistical inferences from texts and transcripts pp 35–54

- Tadle (2022) Tadle RC (2022) Fomc minutes sentiments and their impact on financial markets. Journal of economics and business 118:106021

- Tan and Wu (2011) Tan S, Wu Q (2011) A random walk algorithm for automatic construction of domain-oriented sentiment lexicon. Expert Systems with Applications 38(10):12094–12100

- Tang et al (2014) Tang D, Wei F, Yang N, Zhou M, Liu T, Qin B (2014) Learning sentiment-specific word embedding for twitter sentiment classification. In: Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp 1555–1565

- Thelwall et al (2010) Thelwall M, Buckley K, Paltoglou G, Cai D, Kappas A (2010) Sentiment strength detection in short informal text. Journal of the American society for information science and technology 61(12):2544–2558

- Tian et al (2021) Tian Y, Chen G, Song Y (2021) Enhancing aspect-level sentiment analysis with word dependencies. In: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp 3726–3739

- Tong (2001) Tong RM (2001) An operational system for detecting and tracking opinions in on-line discussion. In: Working Notes of the ACM SIGIR 2001 Workshop on Operational Text Classification, vol 1

- Turney and Littman (2003) Turney PD, Littman ML (2003) Measuring praise and criticism: Inference of semantic orientation from association. ACM Transactions on Information Systems (TOIS) 21(4):315–346

- Wang et al (2018) Wang L, Niu J, Song H, Atiquzzaman M (2018) Sentirelated: A cross-domain sentiment classification algorithm for short texts through sentiment related index. Journal of Network and Computer Applications 101:111–119

- Wang et al (2013) Wang X, Zhang C, Ji Y, Sun L, Wu L, Bao Z (2013) A depression detection model based on sentiment analysis in micro-blog social network. In: Pacific-Asia Conference on Knowledge Discovery and Data Mining, Springer, pp 201–213

- Wang et al (2014) Wang Z, Yu Z, Chen L, Guo B (2014) Sentiment detection and visualization of chinese micro-blog. In: 2014 International Conference on Data Science and Advanced Analytics (DSAA), IEEE, pp 251–257

- Wiebe et al (1999) Wiebe J, Bruce R, O’Hara TP (1999) Development and use of a gold-standard data set for subjectivity classifications. In: Proceedings of the 37th annual meeting of the Association for Computational Linguistics, pp 246–253

- Wiebe et al (2000) Wiebe J, et al (2000) Learning subjective adjectives from corpora. Aaai/iaai 20(0):0

- Williams et al (2018) Williams J, Comanescu R, Radu O, Tian L (2018) Dnn multimodal fusion techniques for predicting video sentiment. In: Proceedings of grand challenge and workshop on human multimodal language (Challenge-HML), pp 64–72

- Xia et al (2015) Xia R, Xu F, Zong C, Li Q, Qi Y, Li T (2015) Dual sentiment analysis: Considering two sides of one review. IEEE transactions on knowledge and data engineering 27(8):2120–2133

- Xu et al (2014) Xu C, Cetintas S, Lee KC, Li LJ (2014) Visual sentiment prediction with deep convolutional neural networks. arXiv pp arXiv–1411

- Xu et al (2019) Xu G, Yu Z, Yao H, Li F, Meng Y, Wu X (2019) Chinese text sentiment analysis based on extended sentiment dictionary. IEEE Access 7:43749–43762

- Yan et al (2019) Yan C, Tu Y, Wang X, Zhang Y, Hao X, Zhang Y, Dai Q (2019) Stat: spatial-temporal attention mechanism for video captioning. IEEE transactions on multimedia

- Yang et al (2019) Yang C, Zhang H, Jiang B, Li K (2019) Aspect-based sentiment analysis with alternating coattention networks. Information Processing & Management 56(3):463–478

- Yang et al (2018) Yang J, She D, Sun M, Cheng MM, Rosin PL, Wang L (2018) Visual sentiment prediction based on automatic discovery of affective regions. IEEE Transactions on Multimedia 20(9):2513–2525

- Yang et al (2020) Yang L, Li Y, Wang J, Sherratt RS (2020) Sentiment analysis for e-commerce product reviews in chinese based on sentiment lexicon and deep learning. IEEE access 8:23522–23530

- Yang et al (2016) Yang Z, Yang D, Dyer C, He X, Smola A, Hovy E (2016) Hierarchical attention networks for document classification. In: Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies, pp 1480–1489

- Zadeh et al (2018) Zadeh A, Liang PP, Poria S, Vij P, Cambria E, Morency LP (2018) Multi-attention recurrent network for human communication comprehension. In: Thirty-Second AAAI Conference on Artificial Intelligence

- Zeng et al (2008) Zeng Z, Pantic M, Roisman GI, Huang TS (2008) A survey of affect recognition methods: Audio, visual, and spontaneous expressions. IEEE transactions on pattern analysis and machine intelligence 31(1):39–58

- Zhang et al (2018) Zhang Z, Zou Y, Gan C (2018) Textual sentiment analysis via three different attention convolutional neural networks and cross-modality consistent regression. Neurocomputing 275:1407–1415

- Zhao et al (2020) Zhao J, Zeng D, Xiao Y, Che L, Wang M (2020) User personality prediction based on topic preference and sentiment analysis using lstm model. Pattern Recognition Letters 138:397–402

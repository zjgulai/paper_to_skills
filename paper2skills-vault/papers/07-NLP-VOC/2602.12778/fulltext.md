<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2602.12778
     paper_id : 2602.12778
     source   : paper2skills-vault/papers/07-NLP-VOC/2602.12778/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

Aspect-Based Sentiment Analysis for Future Tourism Experiences: A BERT-MoE Framework for Persian User Reviews First Hamidreza Kazemi Taskooh1* and Second Taha Zare Harofte2

arXiv:2602.12778v1 [cs.CL] 13 Feb 2026

1 2

Industrial Engineering, IUST, Narmak, Tehran, 16846-13114, Iran.
Industrial Engineering, Organization, Street, Tehran, 16846-13114, Iran.

*Corresponding author(s). E-mail(s):
hamidreza kazemi83@ind.iust.ac.ir;
Contributing authors: taha zare@ind.iust.ac.ir;

This study contributes to the development of aspect-based sentiment analysis (ABSA)
in the tourism industry by creating a hybrid model designed for user reviews in the Persian language. It tackles the linguistic issues of low-resource languages to provide practical insights for ABSA in tourism to orient to the future in order to improve the personalization and sustainability of Iran’s digital tourism industry while considering the sustainability SDGs 9 and 12 of the UN. A multi-stage pipeline was designed:
at first, BERT for overall sentiment classification on 9,558 labeled reviews. After that, aspect extraction using a BERT encoder with sigmoid activation for six tourism aspects (host, price, location, amenities, cleanliness, connectivity) was offered, and at the end, ABSA via BERT was integrated with a hybrid architecture and Top-K routing to reduce routing collapse. The dataset comprises 58,473 preprocessed reviews from Jabama, an Iranian accommodation platform, annotated for aspects and sentiments. The model achieved a weighted F1-score of 90.6% for ABSA, outperforming the baseline BERT (89.25%) and the hybrid (85.7%). The hybrid’s dynamic routing can enable specialized sentiment detection, so we found out important aspects like cleanliness and amenities have high mention rates. Efficiency gains included a 39% lower GPU power consumption compared to dense BERT, which supports sustainable AI deployment. This is the first ABSA study for Persian tourism reviews, introducing a novel hybrid with Top-K routing and auxiliary losses for low-resource settings. The open-source dataset release fosters future multilingual NLP in tourism research.

1

1 Introduction Despite the rapid growth of online tourism platforms in Iran, there is still no large-scale aspect-based sentiment analysis (ABSA) system capable of understanding fine-grained user opinions in Persian. Sentiment analysis is now one of the NLP’s key elements. It can give a chance to businesses and platforms to collect and utilize valuable information from content created by users (Dashtipour et al. 2021). Traditional methods often only consider the general feeling. They fail to address specific opinions about different parts of the product or service(Liu et al. 2015). This is even more pronounced in the fields of tourism and hospitality, where the experiences have multiplicity such as cost, location, host attitude, cleanliness, etc. (Kwon 2025; Moreno-Ortiz et al. 2019). ABSA analyzes reviews at a detailed level and uses sentiment information to support better decision-making and improve service quality. To meet the demands of a more personalized, sustainable, and tech-driven tourism sector, aligning with UN SDGs 9 and , tools like ABSA are urgently needed (Khizar et al. 2023; Das et al. 2025). Geotagged social media data has proven effective in characterizing tourist flows and sentiments in real-world settings (Paolanti et al. 2021). This research will concentrate on the Persian language because of the lack of well-labeled data and weak models that hinder the linguistic environment. Persian sentiment analysis faces consistent challenges. These stem from inadequate preprocessing, culturally influenced perceptions, and a lack of standardized sentiment resources (Rajabi and Valavi 2021). In this study, we address these limitations by building an aspect-based sentiment analysis system specifically designed for Persian tourism reviews. To remedy the situation, we leverage a large dataset obtained from Jabama (www.jabama.com), one of the dominant accommodation booking sites in Iran, with 7 million users and 18,000 hosts across 769 cities. After preprocessing, we ended up with 58,473 high-quality reviews (from an initial set of 72,238), each annotated for six important aspects: the host, price, location, amenities, cleanliness, and connectivity. This dataset addresses one of the major missing pieces in Persian NLP and paves the way for predictive models that can genuinely help tourism stakeholders make better, evidence-based decisions. It can also predict and plan service delivery for a sustainable future. To address the language difficulties of Persian, given the computational requirements of ABSA and the textual content, we recommend a hybrid design that combines a Mixture of Experts (MoE) with BERT (Devlin et al.
2019; Shazeer et al. 2017). The BERT offers strong contextual comprehension because of its pretraining on massive multilingual data, while the MoE improves efficiency by routing. Compared to dense models, our three-stage method, which fine-tunes BERT for inputs to specialized sub-networks, results in a 39% reduction in GPU power use (Zeng et al. 2024). Our basic sentiment classification, aspect extraction with a BERT encoder, and ABSA using a hybrid expert-enhanced BERT model achieved a weighted F1-score of 90.6%, outperforming standalone BERT (89.25%) and a more advanced hybrid BERT model (BERT+MoE+LoRA) (85.7%) (Hu et al. 2021). This scalable method makes it a practical choice for real-time use of futuristic tourism platforms.
Our MoE design uses Top-K routing and auxiliary losses. This helps to minimize route failures, balance specialist use, and enable potential edge computing applications for mobile tourism. Despite all the advances in ABSA has seen, the literature keeps reminding us that serious problems remain, particularly for low-resource languages.
2

Persian is no exception—robust models and specialized datasets are still largely missing, a situation shared by many other less-studied languages (Ataei et al. 2019).
Our study contributes by releasing this annotated Jabama dataset as an open-source resource, fostering advancements in multilingual tourism NLP. While our experiments are limited to Persian, the architecture itself is generalizable and can be adapted for other languages, especially those facing similar resource constraint. Predictive insights help travel companies to better meet user needs, therefore generating more connection (SDG 9) and green infrastructure (SDG 12). The main contributions of this work are as follows: (1) we introduce the first large-scale Persian ABSA dataset for tourism, consisting of 58,473 annotated reviews across six key aspects; (2) we propose a threestage hybrid BERT–MoE architecture with Top-K routing that reduces GPU power consumption by 39% while improving F1 performance; and (3) we provide an efficient and generalizable ABSA framework suitable for low-resource languages and real-time tourism platforms. The rest of this paper is organized as follows. Section 2 reviews the related work on ABSA, with a focus on Persian and low-resource languages as well as tourism applications. Section 3 describes the Jabama dataset, preprocessing steps, and the proposed three-stage hybrid BERT-MoE model with Top-K routing. Section 4 presents the experimental results, showing that our model achieves a weighted F1score of 90.6% on the ABSA task while reducing GPU power consumption by 39% compared to dense BERT. Finally, Sections 5 and 6 conclude the paper and introduce future work.

List of Abbreviations The following compact model names are used only in tables and figures to save space.

Abbreviation

Full name

BERT+MoE BERT+MoE+LoRA

BERT with integrated Mixture-of-Experts BERT+MoE with additional LoRA adapters

2 Literature Review Aspect-Based Sentiment Analysis (ABSA), which has been established, can enable extracting subtle feelings from user comments to develop a diverse range of tourist experiences as a fundamental element of NLP (Sahin and Eyupoglu 2025; Kwon 2025).
Contrary to holistic sentiment categorization, which considers variables like facilities or location (Xu et al. 2024) that support personalization and sustainability, contributing to predictive modeling for tourism futures under UN SDGs 9 (industry innovation) and 12 (responsible consumption) (Kwon 2025; Li et al. 2023). This assessment combines over 30 studies (2017–2025) from the body, organized by methodological paradigms, to track ABSA’s shift from rule-based systems to hybrid transformers (Guidotti et al.
2025). It points out the need for efficient, scalable models—the driving force behind tourism—by drawing attention to deficiencies in datasets and low-resource languages like Persian (Nooraee et al. 2025). Motivation behind our hybrid expert-enhanced

3

BERT framework for applications with a future orientation (Jiang et al. 2024; Farahani et al. 2021).

Language model and transformer-based strategies.
Transformers outperform in solving challenges based on ABSA because of their bidirectional contextualization (Farahani et al. 2021). According to studies conducted from 2021 to 2025, BERT variants have been the most effective models for the ABSA task, with tuned models for specific tasks such as Instruct-DeBERTa (Mewada and Dewang 2022) and enhanced SBERT (Guidotti et al. 2025) being useful for customer-centered tourism. Persian adaptations such as ParsBERT (Farahani et al. 2021; Ataei et al.
2019), AriaBERT (Ghafouri et al. 2023), and Tiny-ParsBERT (Nooraee et al. 2025)
and the high accuracy attained for mobile tourism apps overcome the challenges of low-resource languages. Multitask learning (Zhao et al. 2023; Li et al. 2023) and Urdu SA (Khan et al. 2025) also support resource-limited contexts. MoE models (Jiang et al. 2024; Zeng et al. 2024; Shazeer et al. 2017) are essential for SDG 12 forecasting, but their high computational costs demand more efficient alternatives.
Traditional and Deep Learning Techniques.
Older rule-based methods still perform well across different topics (Liu et al. 2015;
Poria et al. 2014) and handle Persian data (Afzaal et al. 2019). Traditional methods combined with deep learning work well for analyzing tourist feedback. For example, some models accurately identify aspects like taste in reviews (Mewada and Dewang 2022; Li et al. 2023). Other approaches using word embeddings understand meaning but miss subtle details compared to modern models (Park and Jeon 2022). In Persian, models analyze movie and literary reviews effectively (Rajabi and Valavi 2021; Khodaei et al. 2022). Ethical models support fair tourism solutions (Park and Jeon 2022), but Persian text is tricky. Newer transformer models improve results (Zeng et al. 2024).
Zero-Shot Models and Ontology.
Zero-shot learning and ontologies make it easier to work with limited data. For example, zero-shot models group TripAdvisor reviews based on things like weather or location, which can improve results (Xu et al. 2024). Using ontologies with ABSA helps find hidden tourism details accurately (Nandwani and Verma 2021). Zero-shot combinations like BART-DeBERTa-RoBERTa, tested on hotel ratings, reach good accuracy for COVID-related features(Kwon 2025) and can adapt to Persian SDG 9 infrastructure goals (Guidotti et al. 2025). Future research could explore Large Language Models (LLMs) for zero-shot keyword and sub-aspect extraction from Persian tourism reviews, reducing annotation costs and enhancing scalability (Guidotti et al.
2025). Models like ParsBERT-mBERT with SHAP provide clear explanations on DariFarsi texts from ArmanEmo (Ghafouri et al. 2023; Muradi et al. 2025). Approaches using WordNet get high accuracy across different areas (Nandwani and Verma 2021), and better annotation methods improve tourist planning (Moreno-Ortiz et al. 2019).
Working with little data is still tough, but tools like Kano-SHAP help sort satisfaction levels, such as focusing on essential cleanliness for SDG 12 (Park and Jeon 2022;
Das et al. 2025).
4

Systematic Reviews and Meta-studies.
Recent studies on aspect-based sentiment analysis (ABSA) show some exciting progress in different methods. One survey about sentiment analysis in Persian looked at ways like lexicon-based, machine learning, and deep learning approaches, and noted that limited resources are a big challenge (Rajabi and Valavi 2021). Another review checked out custom tools that mix lexicon, machine learning, and deep learning methods (Moreno-Ortiz et al. 2019). Some researchers explored hybrid models that combine data augmentation with pre-trained systems (PourMostafa Roshan Sharami et al.
2020). When comparing these methods, they found different results across areas like computers and restaurants (Mewada and Dewang 2022; Li et al. 2023). Studies on Persian movie sentiment analysis used special datasets and deep learning models, getting excellent results but still facing issues with language and data variety (Khodaei et al. 2022; Dashtipour et al. 2021). Another study on hospitality sentiment analysis pointed out that linguistic limits are still a problem, even with recent improvements (Sahin and Eyupoglu 2025).
Resources and Datasets in Several Languages.
Despite the scarcity of Persian tourism datasets, aspect-based sentiment analysis (ABSA) relies heavily on such data (Ataei et al. 2019). A Persian dataset with thousands of targets established a baseline using TD-LSTM models (Jafarian et al. 2020;
Ataei et al. 2019). A German restaurant dataset derived from TripAdvisor reviews has been used with semantic clustering to enhance tourism recommendation systems (Abbasi-Moud et al. 2021). Multilingual approaches and hybrid models were tested using an Urdu review dataset (Khodaei et al. 2022). Topic modeling on TripAdvisor hotel data enabled focused sentiment-oriented summarization (Sahin and Eyupoglu 2025; Akhtar et al. 2017). Comparative methods were also used to assess the quality of Amazon reviews through aspect-based sentiment analysis (Mewada and Dewang 2022).
Limitations and Areas for Investigation.
ABSA faces challenges with uneven data and high demands for resources and time (Sahin and Eyupoglu 2025). To deal with this, experts suggest using simpler and more efficient language models (Nooraee et al. 2025). For Persian, problems like regional biases and spelling differences get worse because of limited data (Farahani et al. 2021;
Ataei et al. 2019). Zero-shot methods also struggle to understand hidden feelings (Kwon 2025). In tourism, better routing methods exist but don’t clearly connect to sustainability goals (Khizar et al. 2023). Even though recent studies give powerful insights, they often lack simple, affordable solutions for areas with few resources.
Research Gap and Contribution.
Overall, the reviewed studies reveal several research gaps, particularly regarding Persian tourism-oriented ABSA. This appears to be the first standalone research focusing on Aspect-Based Sentiment Analysis (ABSA) within the field of tourism and the Persian language. This research intends to address a gap in the NLP literature concerning under-resourced languages (Rajabi and Valavi 2021). Previous studies concentrated 5

on general sentiment analysis within the Persian language, analyzing domains like cinema and document-level sentiment (Dashtipour et al. 2021; Kaveh and Safa 2025).
There has been little to no work on the more nuanced and difficult task of aspectsentiment extraction on real-world, applied, and domain-specific datasets (Ataei et al.
2019; Moreno-Ortiz et al. 2019; Jafarian et al. 2020; Mewada and Dewang 2022).
For the greater research aims in the field, we plan to publish the annotated corpus for wider public access. This is to promote research in Persian NLP and to stimulate creating tourism-focused downstream applications. Besides its scholarly impact, this study paves the way for several practical applications, just as ranking services at the aspect level, customizing travel suggestions and automatic feedback summarization systems. The advancements improve user experience and also encourage Persian language businesses to grow in the tourism sector.

3 Methodology This section outlines the dataset, preprocessing steps, model architecture, training procedure, and evaluation strategy used in this study. This research presents a multi-stage model for ABSA, for the Aspect Category Detection (ACD) subtask, which is aimed at recognizing and categorizing key aspects of the Jabama platform (www.jabama.com)
Persian user reviews. Our method includes gathering and preparing Persian data, creating the system architecture (Maroof et al. 2024), training the model, and performing a comprehensive evaluation.

Fig. 1 Workflow of data collection, preprocessing, model training and evaluation.

Dataset Collection and Preprocessing.
A dataset of 72,238 user reviews was collected from Jabama, a leading Iranian tourism platform that serves over seven million users and 18,000 hosts across 769 cities. Because of the irregular characters, inconsistent half-spaces, varying forms of orthography,

6

and general differences regarding the spelling of words in the language, preprocessing became necessary (Ghafouri et al. 2023; Rajabi and Valavi 2021; Nandwani and Verma 2021). Standardizing characters as well as removing emojis, ensuring uniform application of half spaces, correcting over a hundred common spelling errors, unifying vocabularies, splitting concatenated words, and removing irrelevant spam were some tasks in the preprocessing pipeline.

Fig. 2 Distribution of 9,558 data points used for training the BERT base model.

58,473 high-quality reviews were kept. Each review was assigned a sentiment polarity and classified into six major categories: host, price, location, amenities, cleanliness, and connectivity. The categories were inspired by sentiment ABSA schemas designed for Persian (Moreno-Ortiz et al. 2019; Ataei et al. 2019; Afzaal et al. 2019), ensuring precise and high-quality annotations for subsequent modeling tasks. Similar domainspecific annotation frameworks have been validated for tourism reviews to improve inter-annotator agreement and schema reliability (Moreno-Ortiz et al. 2019).

Model Development.
The proposed model was developed in three stages (Figure 1):
1. Basic Sentiment Analysis For fundamental sentiment analysis, we adjusted a BERT Base model (Devlin et al.
2019) to categorize review sentiments into positive, negative, and neutral classes. To address limited data, we used a semi-supervised active learning technique, whereby the model was repeatedly used on the unlabeled data. To begin, we evaluated and manually incorporated 1,800 high-confidence predictions into the training data.
Subsequently, we integrated the rest of the high-confidence predictions automatically, resulting in the final labeled data set consisting of 9,558 samples. The modified model achieved an F1 score of 93.3% (with a learning rate of 2 × 10−5 , batch size of 32, and 4 epochs).
2. Aspect Category Detection (ACD)
A modified BERT encoder with a sigmoid activation function (Figure 5) was trained to identify six aspects: host, price, location, amenities, cleanliness, and connectivity. While achieving a weighted F1 score of 89.69% during training, the following 7

Table 1 This section shows a comparison of Persian sentiment analysis models. Considering the different datasets utilized, the comparison is indirect. It is only a general guide to compare performance(PourMostafa Roshan Sharami et al. 2020;
Farahani et al. 2021)
Model BERT fine-tuned ParsBERT v2 DeepSentiPers

Dataset

F1-score (%)

Jabama SentiPers (Binary Class)
SentiPers (Binary Class)

93.3% 92.42% 91.98%

Fig. 3 Precision–Recall curves for the baseline BERT model. Overall classification performance is summarized with the micro-average, which is located in the top left corner. Per-class curves show the results for the negative, neutral, and positive sentiment classes. The neutral class exhibits greater fluctuations than the other classes due to imbalance. In contrast, the positive and negative classes display more stable behavior.

hyperparameters were configured: (1.7e-5 learning rate, 8 batch size and 4 epochs).
Two domain-aware annotators maintained consistency and clarity throughout the annotation process, as for every aspect, a set of predefined semantic definitions was provided. The six aspect categories are host, location, amenities, connectivity, cleanliness, and price. The host aspect accounts for statements regarding the demeanor of the owner or the reception staff. Location refers to comments regarding geographic accessibility, views, and positioning in general. Amenities cover facilities within the dwelling, including the kitchen, swimming pool, systems for heat regulation (packaged units), ventilation, and security. Comments regarding the access to the internet or the mobile signal are addressed in connectivity (e.g., “no signal,” “weak Wi-Fi”). The cleanliness aspect refers to comments related to the hygiene and tidiness of the rooms. Last, price refers to comments about value for money, cost, and the fairness of pricing. Reviews could be tagged with multiple annotations. A single review might comprise multiple category labels. Implicit sentiment

8

was shown in phrases like ‘wish it was cleaner.’ Reviews that didn’t show the needed points were removed to keep things clear and good. Tabel 4 in the results section shows how the aspect labels are spread. Of the attributes, cleanliness and amenities were noted as two of the more prominent. This observation directly reflects the distribution shown in Figure 4. This careful labeling helped train the multi-label classifier and made the ACD stage more reliable.

Fig. 4 The distribution of labels for aspects.

Fig. 5 Architecture of the BERT-based model used for Aspect Category Detection (ACD). A sigmoid activation function is applied to the output layer to allow multi-label classification across six aspect categories: host, price, location, amenities, cleanliness, and connectivity.

3. Aspect-Based Sentiment with a hybrid BERT model To address the aspect-based sentiment classification (ABSA) task, we designed a Mixture-of-Experts (MoE) architecture atop a fine-tuned BERT model. The intuition behind this design is that different experts can specialize in different sentiment patterns across aspects such as cleanliness, price, location, and others, while a learned gating mechanism determines the contribution of each expert for a given

9

input. Our MoE model consists of a BERT base encoder (bert-base-multilingualcased) fine-tuned on domain-specific Airbnb review data (Devlin et al. 2019), with the [CLS] token embedding (dimension: 768) serving as input to six feed-forward neural network experts. Each expert comprises a linear layer (768→256), a ReLU activation, and a second linear layer (256→3) for the three sentiment classes (positive, neutral, negative). A gating network, receiving the aspect term embedding, outputs a softmax distribution over the experts, producing weights for a batchwise einsum operation to aggregate expert outputs. This specialized routing follows the Mixture-of-Experts paradigm (Shazeer et al. 2017) and incorporates recent advances in top-k routing (Zeng et al. 2024).
Table 2 Performance Comparison of ABSA Models.
Model

F1-score (%)

Standalone BERT BERT+MoE BERT+MoE+LoRA(Hard gate)
BERT+MoE+LoRA

89.25 89.43 85.7 85.7

Table 3 Performance and Routing Analysis of MoE Variants.
Variant BERT+MoE (Baseline)
BERT+MoE + Aux Loss v1 BERT+MoE + Aux Loss v2 (MSE)

F1-score (%)

COV²

89.43 93.03 93.36

1.5856 2.1406 2.0900

This architecture achieved an overall F1-score of 90.6% (learning rate = 1.8552 ×10−5 , batch size = 8, epochs = 3), which is greater than the scores of BERT and advanced hybrid BERT model (BERT+MoE+LoRA). Compared to other models, the hybrid expert-enhanced BERT has about 164 million parameters and can balance complexity and performance. We used two MoE approaches, the first being a hard mapping variant where experts were fixed and assigned to specific aspects (F1 = 87.23%), and the second being a dynamic routing variant where the gate leans into weighted aspect embeddings for the best score (F1 = 90.80%). The dynamic routing facilitated expert specialization, as seen in the expert weight heatmap (Figure 6). Other than the F1-score, which we weighted primarily because of imbalanced classes, the model used categorical cross-entropy loss. To better fine-tune hyperparameters, the dataset was divided into 80% training and 10% each for validation and testing. Future work can consider the fusion of sentence-aspect in the gate for greater interpretability, as well as attention visualization.

10

Loss Formulations and Routing Mechanisms.
Our model is trained by minimizing the standard categorical cross-entropy (CCE) loss:
N

C

1 XX LCE = − yi,c log(ŷi,c ), N i=1 c=1

(1)

where N is the batch size, C =3 is the number of sentiment classes (positive, neutral, negative), yi,c is the ground-truth one-hot label, and ŷi,c is the predicted probability from the softmax layer.
To counteract routing collapse, we incorporate an auxiliary importance loss inspired by GShard (Lepikhin et al. 2020) and later adopted in Switch Transformer (Fedus et al. 2022). The complete training objective is

L = LCE + λaux Laux + Lmse ,

(2)

with λaux = 0.011822. Full mathematical formulations of all three loss terms are provided in Appendix A.

Fig. 6 Heatmap of gate-assigned expert weights across aspect types (before applying rectification techniques), illustrating emergent specialization.

Enhanced Expert Utilization.
We employed a Top-K routing mechanism (K =3) with a capacity factor of 1.8, combined with two rectification techniques to mitigate routing collapse. Intra-GPU Rectification (IR) reassigns dropped tokens (due to capacity overflow) to the highestscoring local expert on the same GPU rather than discarding them. Fill-in Rectification (FR) fills padding positions in under-utilized experts with the (k +1)-th highest-scoring token candidates. During training, noisy Top-K gating was applied by adding Gumbeldistributed noise (scaled by 0.098323) to the gate logits. These modifications reduced the squared coefficient of variation (COV2 ) of expert utilization from 1.5856 (baseline softmax routing) to 0.0109, achieving near-uniform load across all six experts (ideal 11

balanced activity ≈ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]). Specialization patterns before and after applying the rectification techniques are shown in Figures 6 and 7, respectively. Full mathematical details of IR and FR, straight-through gradient handling, and complete pseudocode are provided in Appendix A.

Fig. 7 Heatmap of gate-assigned expert weights across aspect types, illustrating improved specialization after Top-K routing implementation.

The Top-K routing approach successfully resolved the routing collapse issue, enabling full and fair utilization of all experts in the MoE architecture. This significantly improved the model’s generalization on unseen samples, as shown with the validation metrics in Figure 8.
The model’s success in handling the inherent class imbalance in the dataset is evident from the consistent and high validation accuracy, as well as recall and precision metrics across all sentiment categories. The class imbalance, particularly the dominance of the cleanliness and amenities aspects, is often difficult to manage with standard classifiers. The weighted F1 score, which accounts for class distribution in its computation, provides further evidence of the proposed model’s superiority, showing a robust precision-recall trade-off that ensures balanced performance across both majority and minority classes.

Implementation Algorithm.
The complete pseudocode for the MoE-based ABSA algorithm, including the Top-K routing with rectification mechanisms, is provided in Appendix B. Persian language, stem from the language’s morphological and syntactic complexity and differing standards of orthography. Other issues with available labeled data and learning reliable representations exacerbates the problem, making the development of Persian NLP systems that perform at the level of English NLP systems almost impossible. In fact, the Top-K routed BERT with Mixture of Experts models integration you proposed could perform the functions of the BERT in outperforming language and NLP processing tasks in English and allow Persian NLP systems of comparable performance to be developed. The use of adaptive noise scaling, designed to temporally regulate

12

Fig. 8 Validation performance metrics for the modified Top-K routing the hybrid expert-enhanced BERT model. From left to right: (a) Validation accuracy, showing stable convergence; (b) Validation precision, highlighting improved precision across sentiment classes; (c) Validation recall, illustrating robust performance despite class imbalance; (d) Weighted F1-score, reflecting superior overall performance.

Fig. 9 The precision-recall curves were generated to assess the performance of the hybrid expertenhanced BERT model on the validation set across each sentiment class (Class 0 - negative, Class 1 - neutral, Class 2 - positive). Among the three, Class 1 (neutral) curves stand out as the most irregular with pronounced fluctuations, while the curves about to the negative and positive classes exhibit greater stability. The micro-average curve indicates the model’s overall performance, while the class-average curves delineate performance regarding each specific sentiment.

the tradeoff between exploitation and exploration, is likely to be the most important for controlling training noise. The proposed technique may contribute to the quest for expert diversification, without over-specialization, to improve robustness for which more abstract objectives are often proposed. Plans that seek to enhance the self-organizing nature of the model, such as dynamically controlling capacity and expanding the model based on the complexity of the input, will increase the model’s robustness and computational efficiency for cross-domain tasks. The collaboration of active attention control in models, as proposed, will enhance model interpretability.
Describing how the gating system selects various experts for the various segments of the task can elucidate how the model arrives at its decisions, thus offering some degree of explainability for the model itself. This model explainability is beneficial to both the user and the researcher. All of the aforementioned can increase the applicability of Top-K routing the hybrid expert-enhanced BERT frameworks for actual tourism platforms and also for numerous low-resource languages.

4 Results The outcomes related to the proposed models during all three stages of development can be found in Table 1. The sentiment analysis BERT model completed the Basic Sentiment Analysis stage, having scored 93.3% weighted F1. This shows the model

13

Fig. 10 Results of the Validation Aspect Category Detection (ACD) Model. The weighted precisionrecall curve indicates the adjustments in the precision-recall tradeoffs for all aspects according to class size. The improvement of the weighted F1-score during training shows a constant increase in overall model performance. The primary increase in weighted precision and recall further indicates the model’s capability in handling and processing the challenges presented by unbalanced aspect categories.

successfully classified the reviews as positive, negative, or neutral with consistent high accuracy. In the Aspect Category Detection (ACD) stage, the BERT encoder with a sigmoid activation function scored 88.0% F1 score across the 6 predefined aspects, which were host, price, location, amenities, cleanliness and connectivity. The hybrid expert-enhanced BERT model achieved the highest performance in aspect-based sentiment analysis (ABSA) because it surpassed a 90.6% weighted F1 score, which was also better than the scores of its standalone BERT (a weighted F1 score of 89.25%)
and advanced hybrid BERT model (BERT+MoE+LoRA) (a weighted F1 score of 85.7%). This can be found in Table 2. The precision–recall dynamics across distinct sentiment classes, as well as the overall microaverage performance, is evaluated in Figures 3, 9, and 10. The analysis BERT model scored high on precision and recall stability in positive and negative classes, while the neutral class showed greater fluctuations in performance stability because of class imbalance. For the ACD model, the micro-averaged PR curve shows powerful performance across all aspects, and the macro metrics show that it also performs well on less frequent categories. The hybrid expert-enhanced model achieved an excellent trade-off between precision and recall across the various sentiment classes, and the dynamic routing mechanism was key to making consistent gains over the baseline. A major difficulty in this study was the imbalance in the distribution of aspect labels, as detailed in Table 4. Cleanliness and amenities were mentioned more in reviews than connectivity and the host, which could lead the model to be biased. However, the BERT model used for baseline sentiment analysis pivoted around this using contextual embeddings to detect sentiment in a much more complex way, which achieved an impressive weighted F1 score of 93.3%.
The next hybrid expert-enhanced model architecture focused on this issue by using aspect-focused specialized experts and a gating mechanism that dynamically allocates weights to experts. The heatmap (Figure 6) shows that some experts focused mainly on specific aspects. This made the model perform better on datasets with imbalances.
The results indicate that the proposed model could cope with the challenges of the Persian language, achieving excellent results with high F1-scores. This may indicate the continuation of NLP research on the Persian language and potentially other under-resourced languages. The architecture of the hybrid expert-enhanced model, with its modularity and dynamic routing, should be able to be generalized to other 14

Table 4 Distribution of aspect categories and sentiment labels in the dataset. Sentiment labels: Negative, Neutral, Positive.
Aspect Price Amenities Host Location Cleanliness Connectivity

Total 693 2202 1267 608 1228 749

Negative 579 1508 188 187 359 580

Neutral 40 73 15 10 30 40

Positive 128 621 1064 411 839 129

datasets, particularly within the tourism domain. Due to domain-specific attributes like host, price, and location, if enough labeled data is provided, the model can be used for other related platforms, such as international booking services. This model could help tourism platforms perform user feedback analysis in a more meaningful way. For example, service providers can detect what needs improvement, like cleanliness or internet access, and travelers can choose better based on detailed feedback.
This helps users have a better experience and improves the connection between hosts and guests in Iran’s growing digital tourism sector. Although the overall weighted F1 improvement appears modest (+1.35 percentage points over the dense BERT baseline), the proposed MoE architecture delivers two decisive advantages that strongly justify its added complexity:

• Energy efficiency: A 39% reduction in GPU power consumption compared to dense BERT (Figure 12), directly supporting UN SDG 12 on responsible consumption and enabling cost-effective, sustainable deployment on tourism platforms in developing regions.
• Training stability and scalability: Near-elimination of routing collapse (COV2 reduced from 1.5856 to 0.0109) ensures stable long-term training and straightforward horizontal scaling—critical limitations that have historically hindered practical adoption of MoE models in real-world, low-resource settings.
These benefits make the architecture particularly well-suited for applications where sustainability, operational cost, and reliable scaling are prioritized alongside predictive performance.

15

Fig. 11 Schematic of the the hybrid expert-enhanced BERT architecture, showing sentence embedding flow into experts and soft routing based on aspect embeddings.

16

Fig. 12 Comparative GPU performance metrics between MoE+BERT (green) and BERT (blue)
architectures. Key findings show: (1) 39% lower power consumption (116W vs 191W), etc.

Implementation and Reproducibility Details All experiments were conducted in Kaggle notebooks using publicly available GPU resources. The final hybrid expert-enhanced architecture (BERT+MoE models), along with all reported results, were trained on two NVIDIA Tesla T4 GPUs (16 GB VRAM each) using PyTorch Automatic Mixed Precision. Early-stage experiments and baseline BERT models were partially trained on a single NVIDIA Tesla P100 GPU (16 GB VRAM). Hyperparameter optimization was performed using the Optuna framework (Akiba et al. 2019) with the Tree-structured Parzen Estimator (TPE) sampler (Bergstra et al. 2011). The search space included batch sizes of {8, 16, 32} per GPU and 3–6 training epochs, while the learning rate was tested in the range of 1 × 10−5 to 3 × 10−5 during hyperparameter search. The optimization process aimed to maximize the weighted F1-score on the validation set. The best configuration identified by Optuna—a batch size of 8 per GPU and 3–4 epochs, depending on the training stage—was used for all final models.
Energy Efficiency and Hardware Performance.
Hybrid expert-enhanced architecture is efficient with regard to hardware. Dynamic expert routing homes in on just the most important segments of the model for each input. This approach cuts energy consumption by nearly 39%. For tourism platforms in Iran and similar emerging markets that process thousands of reviews daily, this directly translates into significant monthly savings in cloud and electricity costs (Qudrat-Ullah 2025). This shows that sparsely activated AIs significantly mitigate the growing energy consumption attributed to artificial intelligence. The energy savings are remarkable since the model shows performance reliability. The model can still provide stable clock speeds with active memory, which is contrary to many dense transformer models that offer little efficiency with their speed and reliability. This is the working efficiency we look for in Mixture-of-Experts designs. This working efficiency extends to mobile, hybrid expert-enhanced models since their energy costs are operational, ensuring efficient target natural language processing. This is the first of many steps we expect in energy-sustainable AIs—targeting energy consumption while maintaining model accuracy. Overall, the architecture performs strongly in ABSA for Persian tourism

17

reviews, despite the complexities of the language and data imbalance, aiding tourism institutions to analyze reviews better for the users. This improves tourism services.

5 Conclusion This study introduced a three-stage ABSA framework for Persian tourism reviews and released the 58,473-review Jabama dataset. The proposed hybrid BERT–MoE model achieved a weighted F1-score of 90.6%, outperforming baseline architectures. TopK routing and rectification techniques ensured stable expert utilization and reduced GPU power consumption by 39%. These results demonstrate the model’s suitability for scalable, energy-efficient ABSA in low-resource languages.

6 Future Work The new hybrid expert-enhanced model performs well on aspect-based sentiment analysis (ABSA) for Persian tourism reviews; however, there are still opportunities for advancement. Detecting sub-aspects of reviews (e.g., ‘kitchen facilities’ or security systems under ‘amenities’) using BiO tagging is one way to improve the accuracy of sentiment analysis, ultimately assisting tourism services on Jabama and similar platforms. Focus on hyperparameter tuning in the next work to enhance the model and enable better performance in various tourism contexts. The other opportunity we have is the integration of the model with Interpretability frameworks. Classification of attributes using the Kano model (i.e., basic, performance, excitement) or sentiment shifting feature analysis using SHAP may enhance model transparency and usability, improving the overall user experience.

References Abbasi-Moud Z, Vahdat-Nejad H, Sadri J (2021) Tourism recommendation system based on semantic clustering and sentiment analysis. Expert Systems with Applications 168:114324. https://doi.org/10.1016/j.eswa.2020.114324 Afzaal M, Usman M, Fong A (2019) Tourism mobile app with aspect-based sentiment classification framework for tourist reviews. IEEE Transactions on Consumer Electronics pp 233–242. https://doi.org/10.1109/TCE.2019.2908944 Akhtar N, Zubair N, Kumar A, et al (2017) Aspect based sentiment oriented summarization of hotel reviews. Procedia Computer Science 115:563–571. https://doi.org/ https://doi.org/10.1016/j.procs.2017.09.115 Akiba T, Sano S, Yanase T, et al (2019) Optuna: A next-generation hyperparameter optimization framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. Association for Computing Machinery, p 2623–2631, https://doi.org/10.1145/3292500.3330701

18

Ataei TS, Darvishi K, Minaei-Bidgoli B, et al (2019) Pars-absa: An aspect-based sentiment analysis dataset for persian. https://doi.org/10.48550/arXiv.1908.01815 Bergstra J, Bardenet R, Bengio Y, et al (2011) Algorithms for hyperparameter optimization. In: Shawe-Taylor J, Zemel R, Bartlett P, et al (eds) Advances in Neural Information Processing Systems, vol 24. Curran Associates, Inc., URL https://proceedings.neurips.cc/paper files/paper/2011/file/ 86e8f7ab32cfd12577bc2619bc635690-Paper.pdf Das P, Mandal S, Nedungadi P, et al (2025) Unveiling sustainable tourism themes with machine learning based topic modeling. Discover Sustainability 6(1). https:
//doi.org/10.1007/s43621-025-01065-4 Dashtipour K, Gogate M, Adeel A, et al (2021) Sentiment analysis of persian movie reviews using deep learning. Entropy 23(5):596. https://doi.org/10.3390/e23050596 Devlin J, Chang MW, Lee K, et al (2019) Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long and Short Papers). Association for Computational Linguistics, Minneapolis, Minnesota, pp 4171–4186, https://doi.
org/10.18653/v1/N19-1423 Farahani M, Gharachorloo M, Farahani M, et al (2021) Parsbert: Transformer-based model for persian language understanding. Neural Processing Letters 53(5):3831– 3847. https://doi.org/10.1007/s11063-021-10528-4 Fedus W, Zoph B, Shazeer N (2022) Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. arXiv preprint arXiv:210103961 https:
//doi.org/10.48550/arXiv.2101.03961 Ghafouri A, Abbasi MA, Naderi H (2023) Ariabert: A pre-trained persian bert model for natural language understanding. Research Square preprint, https://doi.org/10.
21203/rs.3.rs-3558473/v1 Guidotti D, Pandolfo L, Pulina L (2025) Discovering sentiment insights: streamlining tourism review analysis with large language models. Information Technology & Tourism 27:227–261. https://doi.org/10.1007/s40558-024-00309-9, URL https:
//doi.org/10.1007/s40558-024-00309-9 Hu EJ, Shen Y, Wallis P, et al (2021) Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:210609685 https://doi.org/10.48550/arXiv.
2106.09685 Jafarian H, Taghavi A, Javaheri A, et al (2020) Exploiting bert to improve aspectbased sentiment analysis performance on persian language. 2021 7th International Conference on Web Research (ICWR) pp 5–8. https://doi.org/10.5121/ijwest.2020.

19

11401 Jiang AQ, Sablayrolles A, Roux A, et al (2024) Mixtral of experts. arXiv preprint arXiv:240104088 https://doi.org/10.48550/arXiv.2401.04088 Kaveh S, Safa R (2025) Advancing natural language processing for persian movie review analysis: Roadmap and opportunities. Computational Algorithms and Numerical Dimensions 4(1):34–47. https://doi.org/10.22105/cand.2024.493386.1168 Khan L, Qazi A, Chang HT, et al (2025) Empowering urdu sentiment analysis: an attention-based stacked cnn-bi-lstm dnn with multilingual bert. Complex Intelligent Systems https://doi.org/10.1007/s40747-024-01631-9 Khizar HMU, Younas A, Kumar S, et al (2023) The progression of sustainable development goals in tourism: A systematic literature review of past achievements and future promises. Journal of Innovation & Knowledge 8(4):100442. https://doi.org/ 10.1016/j.jik.2023.100442 Khodaei A, Bastanfard A, Saboohi H, et al (2022) Deep emotion detection sentiment analysis of persian literary text. https://doi.org/10.21203/rs.3.rs-1796157/v1 Kwon W (2025) Aspect-based sentiment analysis through zero-shot text classification and impact-asymmetry analysis. International Journal of Hospitality Management 133:104397. https://doi.org/10.1016/j.ijhm.2025.104397 Lepikhin D, Lee H, Xu Y, et al (2020) Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:200616668 https://doi.
org/10.48550/arXiv.2006.16668 Li H, Yu BXB, Li G, et al (2023) Restaurant survival prediction using customergenerated content: An aspect-based sentiment analysis of online reviews. Tourism Management 96:104707. https://doi.org/10.1016/j.tourman.2022.104707 Liu Q, Gao Z, Liu B, et al (2015) Automated rule selection for aspect extraction in opinion mining. In: Proceedings of the 24th International Conference on Artificial Intelligence. AAAI Press, p 1291–1297, https://doi.org/10.5555/2832415.2832429 Maroof A, Wasi S, Jami SI, et al (2024) Aspect based sentiment analysis for service industry. IEEE Access 12:1–1. https://doi.org/10.1109/ACCESS.2024.3440357 Mewada A, Dewang RK (2022) Sa-asba: a hybrid model for aspect-based sentiment analysis using synthetic attention in pre-trained language bert model with extreme gradient boosting. The Journal of Supercomputing 79(5):5516–5551. https://doi.
org/10.1007/s11227-022-04881-x

20

Moreno-Ortiz A, Salles-Bernal S, Orrequia-Barea A (2019) Design and validation of annotation schemas for aspect-based sentiment analysis in the tourism sector. Information Technology & Tourism 21(4):535–557. https://doi.org/10.1007/ s40558-019-00155-0, URL https://doi.org/10.1007/s40558-019-00155-0 Muradi M, Hussain B, Rhythm ER, et al (2025) A comparative study of parsbert and mbert in emotion recognition for dari-farsi text with explainable ai. Association for Computing Machinery, https://doi.org/10.1145/3723178.3723231 Nandwani P, Verma R (2021) A review on sentiment analysis and emotion detection from text. Social Network Analysis and Mining 11:81. https://doi.org/10.1007/ s13278-021-00776-6 Nooraee M, Ghaffari H, Kermani FZ (2025) Tiny-parsbert: an optimized hybrid model for efficient sentiment analysis in persian texts. The Journal of Supercomputing https://doi.org/10.1007/s11227-025-07297-5 Paolanti M, Mancini A, Frontoni E, et al (2021) Tourism destination management using sentiment analysis and geo-location information: a deep learning approach. Information Technology & Tourism 23(2):241–264. https://doi.org/10.
1007/s40558-021-00196-4, URL https://doi.org/10.1007/s40558-021-00196-4 Park H, Jeon H (2022) The dynamics of customer satisfaction dimension based on bert, shap, and kano model. IFAC-PapersOnLine 55(10):2384–2389. https://doi.org/10.
1016/j.ifacol.2022.10.065 Poria S, Cambria E, Ku LW, et al (2014) A rule-based approach to aspect extraction from product reviews. https://doi.org/10.3115/v1/W14-5905 PourMostafa Roshan Sharami J, Abbasi Sarabestani P, Mirroshandel SA (2020)
Deepsentipers: Novel deep learning models trained over proposed augmented persian sentiment corpus. arXiv preprint arXiv:200405328 https://doi.org/10.48550/ arXiv.2004.05328 Qudrat-Ullah H (2025) A thematic review of ai and ml in sustainable energy policies for developing nations. Energies 18(9):2239. https://doi.org/10.3390/en18092239 Rajabi Z, Valavi MR (2021) A survey on sentiment analysis in persian: a comprehensive system perspective covering challenges and advances in resources and methods.
Cognitive Computation 13:882–902. https://doi.org/10.1007/s12559-021-09886-x Sahin GG, Eyupoglu C (2025) Aspect-based sentiment analysis for hospitality industry applications: A systematic literature review. Advances in Computational Science and Computing 30(1). https://doi.org/10.2478/acss-2025-0007 Shazeer N, Mirhoseini A, Maziarz K, et al (2017) Outrageously large neural networks:
The sparsely-gated mixture-of-experts layer. https://doi.org/10.48550/arXiv.1701.

21

06538, arXiv preprint arXiv:1701.06538 Xu C, Wang M, Ren Y, et al (2024) Enhancing aspect-based sentiment analysis in tourism using large language models and positional information. https://doi.org/ 10.48550/arXiv.2409.14997 Zeng Z, Guo Q, Fei Z, et al (2024) Turn waste into worth: Rectifying top-k router of moe. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Miami, Florida, pp 9363–9375, https://doi.org/10.18653/v1/2024.emnlp-main.739 Zhao G, Luo Y, Chen Q, et al (2023) Aspect-based sentiment analysis via multitask learning for online reviews. Knowledge-Based Systems 264:110326. https://doi.org/ https://doi.org/10.1016/j.knosys.2023.110326

22

Appendix A A.1

Detailed Loss Formulations and Routing Mechanisms

Auxiliary Losses

In MoE models, a gating network chooses which experts handle each input. If we don’t guide it, only a few experts do most of the work while others sit idle. This makes the model less capable and reduces specialization. To counteract this, we incorporate an auxiliary importance loss inspired by load-balancing objectives proposed in GShard (Lepikhin et al. 2020) and later adopted in Switch Transformer (Fedus et al. 2022).
Var(u)
, λaux = 0.011822 (A1)
Mean(u)2 where Var(0) and Mean(0) compute the variance and mean across experts, and λaux controls the strength of the regularization. Minimizing this makes the experts share work more equally and become more specialized.
Another approach clearly punishes variations from the perfect uniform distribution, therefore strongly promoting consistent usage of knowledge.

Laux = λaux ·

uuniform =

1 · 1E E

uuniform,e =

1 , E

e = 1, 2, . . . , E.

(A2)

This is achieved by introducing a mean squared error (MSE) regularization term:

2 E  1 1 X ue − , LMSE = λMSE · E e=1 E

(A3)

where λMSE is the weight of the MSE term.

A.2

Evaluation of Expert Utilization

In order to evaluate the equity of expert usage in the MoE model, we calculated the squared Coefficient of Variation (COV2 ) (Shazeer et al. 2017) over the routing distributions for 10 batches of the test set. The baseline softmax routing yielded COV2 = 1.5856, signifying extreme disparity: experts 0, 1, and 4 were entirely sidelined while others monopolized the routing. Even with auxiliary losses alone, COV2 reached 2.1406 and 2.0900, indicating catastrophic routing collapse.

A.3

Intra-GPU Rectification (IR)

The Intra-GPU Rectification (IR) focuses on dropped tokens that occur when the number of tokens assigned to an expert surpasses the expert’s capacity limit. Instead of discarding these tokens or routing them across GPUs (which incurs high communication costs), IR reroutes them to the optimal expert within the same GPU. For a token xi dropped (k − |Ri |) times (where Ri is the set of successfully routed experts

23

from the initial top-k routing), IR assigns it to the highest-scoring local expert h based on routing scores aih = wh⊤ xi . The combined output oi is then computed as

P oi =

j∈Ri e

aij

P

Ej (xi ) + (k − |Ri |)eaih Eh (xi )

j∈Ri e

aij + (k − |R |)eaih i

,

(A4)

where Ej (xi ) and Eh (xi ) are the outputs from the initial top-k experts and the IR expert, respectively. The scaling factor (k − |Ri |) enhances the IR contribution in case of multiple drops. IR mitigates routing collapse by maintaining local balance, since dropped tokens per GPU are mostly equitable due to data parallelism.

A.4

Fill-in Rectification (FR)

Under-utilized experts are originally filled with zero-padding to sustain balanced workloads across GPUs, resulting in redundant computation. Fill-in Rectification (FR)
replaces this padding with high-scoring tokens that were not selected in the initial top-k routing. For each token xi , FR identifies the (k + 1)-th highest-scoring expert as a candidate. Among tokens selecting the same expert, those with the highest routing scores ai(k+1) are prioritized to occupy the padding positions, effectively extending top-k to top-(k + 1) while keeping fixed capacity. To prevent vanishing gradients for inactivated experts, we adopt the straight-through estimator during back-propagation by treating the normalization denominator as constant:
P ∂ ( j gij ) ∂gij ∂L ∂L ≡P · P · , (A5)
∂aij ∂aij j gij j gij where gij = eaij /

A.5

P

aim

me

.

Final Effect of IR + FR

After incorporating both IR and FR (with capacity factor 1.8 and K =3), COV2 decreased dramatically to 0.0109, indicating an almost uniform distribution of expert utilization (balanced activity ≈ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]). Figure 7 illustrates the improved specialization and confirms that no single expert dominates while all six sub-networks actively contribute.

24

Appendix B

Pseudocode for the MoE-based ABSA Algorithm

Algorithm 1 Pseudocode for Aspect-Based Sentiment Analysis using MoE with BERT Require: Pre-trained fine-tuned BERT model path, Test data Excel file Step 1: Load Model and Data 1: Load tokenizer and BERT model from the specified path 2: Move model to device (GPU if available) and set to evaluation mode 3: Read test data from Excel file Step 2: Preprocess Data 4: for each row in data do 5:
sentence ← row[’review’]
6:
aspect ← row[’Category’]
7:
label ← row[’sentiment’]
8:
Encode sentence and aspect to obtain input ids, attention mask, and aspect embedding 9:
Append to respective lists 10: end for 11: Stack inputs into tensors Step 3: Create and Split Dataset 12: Create TensorDataset from processed tensors 13: Split into training (80%), validation (10%), and test (10%) sets Step 4: Define Gate Module 14: Compute logits and probabilities for expert selection using linear layers and softmax Step 5: Top-K Dispatch 15: Select top-k experts based on gate scores 16: Dispatch tokens with capacity limits 17: Combine weighted expert outputs Step 6: Input Rectification (IR)
18: Reassign dropped tokens to the best local expert 19: Adjust weights and update outputs Step 7: Fill-In Rectification (FR)
20: Fill empty expert slots with top-(k + 1) candidates 21: Compute and add outputs Step 8: Define MoE Model 22: Use BERT to obtain CLS embedding 23: Apply gate for expert selection 24: Perform top-k dispatch, IR, and FR to produce final logits

25


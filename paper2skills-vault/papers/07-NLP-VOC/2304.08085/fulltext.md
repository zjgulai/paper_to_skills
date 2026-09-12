<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2304.08085
     paper_id : 2304.08085
     source   : https://arxiv.org/html/2304.08085v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction

Xiao Wang, Weikang Zhou , Can Zu, Han Xia, Tianze Chen Affiliation: School of Computer Science, Fudan University, Shanghai, China Email: xiaowang20@fudan.edu.cn    Yuansen Zhang, Rui Zheng, Junjie Ye, Qi Zhang†, Tao Gui , Affiliation: School of Computer Science, Fudan University, Shanghai, China Affiliation: Institute of Modern Languages and Linguistics, Fudan University, Shanghai, China Email: qz@fudan.edu.cn    Jihua Kang, Jingsheng Yang, Siyuan Li, Chunsai Du, Affiliation: ByteDance Inc. Email: tgui@fudan.edu.cn

###### Abstract

Large language models have unlocked strong multi-task capabilities from reading instructive prompts. However, recent studies have shown that existing large models still have difficulty with information extraction tasks. For example, gpt-3.5-turbo achieved an F1 score of 18.22 on the Ontonotes dataset, which is significantly lower than the state-of-the-art performance. In this paper, we propose InstructUIE, a unified information extraction framework based on instruction tuning, which can uniformly model various information extraction tasks and capture the inter-task dependency. To validate the proposed method, we introduce IE INSTRUCTIONS, a benchmark of 32 diverse information extraction datasets in a unified text-to-text format with expert-written instructions. Experimental results demonstrate that our method achieves comparable performance to Bert in supervised settings and significantly outperforms the state-of-the-art and gpt3.5 in zero-shot settings.

## 1 Introduction

Large language models (LLMs) Brown et al. (2020); Ouyang et al. (2022); OpenAI (2023) show tremendous promise in generalization within the set of observed tasks through multi-task training and unified encoding Mishra et al. (2022); Wang et al. (2022c); Longpre et al. (2023). Recent research has revealed a significant performance gap in LLMs when it comes to information extraction (IE) tasks Ye et al. (2023); Chen et al. (2023). For instance, gpt-3.5-turbo achieves an 18.22 F1 score on the Ontonotes dataset, which is far from satisfactory. Therefore, it is necessary to explore how to build a unified information extraction (UIE) model with LLMs.

*Figure 1: Illustration of 3 different paradigms for solving unified information extraction task.*

Recently, Lu et al. (2022) proposed UIE, which uniformly encodes different extraction structures via a structured extraction language, and captures the common IE abilities via a large-scale pretrained text-to-structure model (shown in Figure 1a). However, UIE requires separate finetune for different downstream tasks. This lead to the poor performance of UIE in low resource settings or facing new label schema, which greatly restricts the application of UIE in real scenarios. Lou et al. (2023) proposed USM, which decouple IE into two basic tasks, token-token linking to extract label-agnostic substructures, and label-token linking to attach substructures to corresponding semantic concepts (shown in Figure 1b). However, USM presents two major limitations. Firstly, it converts IE into a semantic matching task, which makes it difficult to integrate with generative language model. Secondly, the method requires semantic matching for each word, which leads to a significant increase in training and inference time.

*Figure 2: The overview framework of InstructUIE. The input consists of task instructions, options, and text. The output is a more understandable sentence converted from the original label structures.*

In this work, we introduce a unified information extraction framework based on multi-task instruction tuning, named InstructUIE (shown in Figure 1c). Specifically, we reformulate IE tasks as a natural language generation problem. For the source sentence, we design descriptive instructions to enable the model to understand different tasks and employ an option mechanism including all candidate categories as constraints of output space. Then, a pre-trained language model is required to generate the target structure and the corresponding type in the form of natural language. We believe that unrestricted decoding would stimulate the latent knowledge of LLMs to complete IE tasks to a larger extent. We further propose auxiliary tasks, which enable the model to capture common structure information and deepen the understanding of diverse semantics. Specifically, we introduce entity span extraction task and entity typing task for named entity recognition (NER) task, entity pair extraction task and entity pair relationship identification task for relation extraction (RE) task, and trigger extraction task and argument extraction task for event extraction (EE) task.

To evaluate the effectiveness of the proposed model, we have developed a new benchmark called IE INSTRUCTIONS. The benchmark consists of 32 diverse information extraction datasets that have been unified into a text-to-text format, allowing for a consistent and standardized evaluation of various IE tasks . Based on the benchmark, we conduct experiments on three main IE tasks under the supervised and zero-shot settings.

Our main contributions are summarized as follows:

-

We propose an end-to-end framework for universal information extraction – InstructUIE, which leverages natural language instructions to guide large language models for IE tasks.

-

We introduce IE INSTRUCTIONS, a benchmark of 32 diverse information extraction datasets in a unified text-to-text format with expert-written instructions.

-

Experimental results demonstrate that InstructUIE achieves comparable performance to Bert in a supervised setup. Notably, our method significantly outperforms the current state-of-the-art and GPT-3.5 in a zero-shot setup.

## 2 Methodology

In this section, we first briefly introduce the setup of instruction tuning. Then, we discuss the task meta-information schema and how IE tasks are mapped into our schema. Next, we discuss the framework of InstructUIE, which consists of two major parts: task schema and auxiliary tasks. Finally, we explain how IE INSTRUCTION is constructed.

### 2.1 Instruction Tuning Background

Instruction tuning is a multi-task learning framework that enables the use of human-readable instructions to guide the output of LLMs. Given a source text and task-specific instructions, the model is trained to generate a sequence of tokens representing the desired output structure and its corresponding labels.

In a supervised setup, the instructions are provided during training for all tasks, and the model is fine-tuned on a set of labeled data for each task. This allows the model to learn task-specific features and optimize for each task. In a zero-shot setup, the instructions are only provided for a subset of tasks during training, and the model is evaluated on unseen tasks without additional fine-tuning. This requires the model to generalize across tasks and use the shared features learned from the instruction tuning framework to infer the output structures for new tasks.

### 2.2 Framework

In this section, we discuss the task meta-information schema and how IE tasks are mapped into our schema. Next, we propose auxiliary tasks, which enable the model to capture common structure information and deepen the understanding of diverse semantics.

#### 2.2.1 Task Schema

To better transfer and utilize the knowledge learned in pre-trained language models, we reformulate the IE tasks to the seq2seq form and solve it through fine-tuning LLMs, as shown in Figure 2. Every task instance is formatted with four properties: task instruction, options, text, and output.

Task Instruction provides a detailed guide on how to extract the relevant information from the input text and produce the desired output structure. It includes information such as the type of information to be extracted, the format of the output structure, and any additional constraints or rules that need to be followed during the extraction process. The task instruction acts as a bridge between the raw input text and the structured output representation, enabling the model to understand the extraction task and generate accurate and meaningful output. In Table 8 in the Appendix we present the list of instructions for each task.

Options are the output label constraints for a task, which represent the set of possible outputs that can be generated by the model for a given input. These label constraints are specific to each task and provide information on how to map the predicted outputs to the corresponding semantic concepts. For instance, in NER, options could be entity tags such as person, organization, location, or miscellaneous. Similarly, in RE, options could represent the types of relations that can be extracted, such as "works for", "born in", "married to", and so on. In EE, options could represent the event tags that correspond to different types of events, such as "beginning", "end", "occurring", "ceasing", and so on. The options provide a structured output space for the model, allowing it to generate outputs that are consistent with the underlying semantic structure of the task.

Text is the input sentence of a task instance. This sequence is then fed into the pre-trained language model along with the task instruction and options, enabling the model to generate the desired output sequence for the given task.

Output is the sentence converted from the original tags of the sample. Specifically, for NER, the output format is "entity tag: entity span". For RE, the output format is "relationship: head entity, tail entity". For EE, the output format is "event tag: trigger word, argument tag: argument span". In cases where the input does not contain structural information that matches any of the provided options, we assign a value of "None" to the corresponding output sentence.

#### 2.2.2 Auxiliary Tasks

To boost the performance in a more fine-grained level, we further design auxiliary tasks to be trained in conjunction with the main task. The auxiliary tasks provide additional information that complements the main task, enabling the model to capture common structures better and deepen the understanding of diverse semantics.

For the named entity recognition task, we introduce a span extraction task and an entity typing task. The span extraction task is designed to extract the entity span from the input sentence, while the entity typing task is aimed at identifying the type of entity.

For the relation extraction task, we have introduced an entity pair extraction task and a relation classification task. The entity pair extraction task aims to extract the entity pairs involved in the relationship, while the relation classification task is designed to classify the type of relationship between the entity pairs.

For the event extraction task, we have introduced a trigger extraction task and an argument extraction task. The trigger extraction task is designed to extract the trigger word that triggers the event, while the argument extraction task aims to extract the associated arguments.

### 2.3 IE INSTRUCTIONS

IE INSTRUCTIONS collects 32 publicly available datasets covering three types of IE tasks: NER, RE, and EE. To ensure the diversity of the datasets, we include corpora from various domains, such as science, healthcare, social media, and transportation, in addition to general-domain sources, such as news and Wikidata. Figure 3 shows the breakdown of the benchmark by task, domain, and size. For detailed dataset statistics and train/test split methods, please refer to Appendix Table 7.

We carry out the following data processing steps: (1) To address the issue of inconsistent label schemas across different tasks, we unify the names of labels with identical semantics but different names in various datasets. (2) To better test the semantic understanding capabilities of the LLM, we convert labels with underscores, abbreviations, or special formats into natural language formats. For example, we renamed the label "people person place_of_birth" to "place of birth." (3) Following the guidelines outlined in the section 2.2.1, we transform all datasets into a text-to-text format, which ensures a consistent representation of the input-output pairs across all tasks.

Our benchmark provides a standardized evaluation platform for LLMs’ performance on IE tasks. This will facilitate a more accurate comparison of various models and contribute to the development of more effective and robust models for IE tasks.

*Figure 3: Overview of IE INSTRUCTIONS.*

## 3 Experiments

This section conducted extensive experiments under supervised and zero-shot settings to validate the effectiveness of InstructUIE. We select 11B FlanT5 Chung et al. (2022) as our backbone model because prior research Longpre et al. (2023) has demonstrated that models fine-tuned on instruction-based tasks offer a computationally efficient starting point for new tasks. The details of the experimental setup, datasets, and comparison methods are described in the following parts.

Dataset UIE USM Bert-base Ours ACE2005 85.78 87.14 87.30 86.66 AnatEM - - 85.82 90.89 bc2gm - - 80.90 85.16 bc4chemd - - 86.72 90.30 bc5cdr - - 85.28 89.59 broad twitter - - 58.61 83.14 CoNLL2003 92.99 93.16 92.40 92.94 FabNER - - 64.20 76.20 FindVehicle - - 87.13 89.47 GENIA-Ent - - 73.3 74.71 HarveyNER - - 82.26 88.79 MIT Movie - - 88.78 89.01 MIT Restaurant - - 81.02 82.55 multiNERD - - 91.25 92.32 ncbi-disease - - 80.20 90.23 Ontonotes - - 91.11 90.19 polyglot-NER - - 75.65 70.15 tweetNER7 - - 56.49 64.97 wikiann - - 70.60 85.13 wikineural - - 82.78 91.36 Avg - - 80.09 85.19

*Table 1: Overall results of InstructUIE on NER task. The evaluation metric is Entity F1. For 20 NER datasets, InstructUIE outperforms the Bert model on 17 of them. *

### 3.1 Experiments on Supervised Settings

#### 3.1.1 Dataset

We conduct supervised experiments on IE INSTRUCTIONS, including three tasks (named entity extraction, relation extraction, and event extraction). Details of the dataset splitting methods and statistics can be found in Appendix 6.1.

To balance the dataset, we apply a sampling strategy Poolsawad et al. (2014). Specifically, we sample 10,000 examples for each dataset and include all examples for datasets with fewer than 10,000 samples.

#### 3.1.2 Baselines

We compare the proposed InstructUIE with the following strong baseline models:

-

UIE Lu et al. (2022) is a unified text-to-structure generation framework that can universally model different IE tasks and adaptively generate targeted structures;

-

USM Lou et al. (2023) is a unified IE tasks framework, which converts IE tasks to a semantic matching problem;

-

Bert Devlin et al. (2019), which are widely used as text encoders for various tasks.

#### 3.1.3 Evaluation Metrics

We use span-based offset Micro-F1 as the primary metric to evaluate the model. For NER task, we follow a span-level evaluation setting, where the entity boundary and entity type must be correctly predicted. For RE task, a relation triple is correct if the model correctly predicts the boundaries of the subject entity, the object entity, and the entity relation. For EE task, we report two evaluation metrics: (1) Event Trigger: an event trigger is correct if the event type and the trigger word are correctly predicted. (2) Event Argument: an event argument is correct if its role type and event type match a reference argument mention.

#### 3.1.4 Results

Dataset UIE USM Ours ADE corpus - - 82.31 CoNLL2004 75.00 78.84 78.48 GIDS - - 81.98 kbp37 - - 36.14 NYT - - 90.47 NYT11 HRL - - 56.06 SciERC 36.53 37.36 45.15 semeval RE - - 73.23 Avg - - 67.98

*Table 2: Overall results of InstructUIE on RE task. The evaluation metric is Relation Strict F1. Our model reaches an average F1 of 67.98% on the eight datasets of the RE task and is comparable to the baseline.*

| Dataset | UIE | USM | Bert-base | Ours |

| ACE2005 | 73.36 | 72.41 | 72.5 | 77.13 |

| CASIE | 69.33 | 71.73 | 68.98 | 67.80 |

| PHEE | - | - | - | 70.14 |

| Avg | - | - | - | 71.69 |

*a. Event Trigger F1*

| Dataset | UIE | USM | Bert-base | Ours |

| ACE2005 | 54.79 | 55.83 | 59.9 | 72.94 |

| CASIE | 61.30 | 63.26 | 60.37 | 63.53 |

| PHEE | - | - | - | 62.91 |

| Avg | - | - | - | 66.46 |

*b. Event Argument F1*

*Table 3: Overall results of InstructUIE on EE task. The evaluation metric is Event Trigger F1 and Event Argument F1. Our model outperformed USM and UIE on some datasets.*

Tabel 1, Tabel 2 and 3 show the performance of different models for the NER, RE, and EE tasks.

##### Named Entity Recognition

Our model achieves an average F1 score of 85.19% on 20 NER datasets, surpassing Bert’s 80.09%. The best performance is on the CoNLL2003 dataset, where InstructUIE achieved an F1 score of 92.94%. For 20 NER data sets, InstructUIE outperforms the Bert model on 17 of them. Among them, our model outperforms Bert by more than 5 points on eight datasets. The dataset with the biggest gap is the broad twitter dataset, where InstructUIE outperforms Bert by about 25 points.

In the ACE2005, Ontonotes, and Polyglot-NER datasets, our model performs slightly worse than Bert. We speculate that this is due to our strategy of sampling only 10,000 training examples for each dataset. The original corpora for these three datasets contain a larger number of training examples, such as 420,000 for Polyglot-NER, of which we only used around 20%. The detailed number of training sets for all datasets can be seen in the appendix.

Compared with UIE and USM, our model also achieves comparable results on ACE2005 and CoNLL2003, which are two commonly used datasets. Due to the UIE and USM only test their modelson a small number of commonly used datasets, we are unable to compare our model with these two models on other datasets.

| Model | Movie | Restaurant | AI | Literature | Music | Politics | Science |

| USM | 37.73 | 14.73 | 28.18 | 56.00 | 44.93 | 36.10 | 44.09 |

| InstructUIE | 63.00 | 20.99 | 49.00 | 47.21 | 53.16 | 48.15 | 49.30 |

*Table 4: Micro-F1 scores of zero-shot NER on 7 datasets. The best results are in bold. InstructUIE outperforms SOTA by a wide margin on most datasets ranging from 5.21% to 25.27%.*

| | Model | FewRel | Wiki-ZSL |

$ZETT_{T5-small}$ | Baselines | | 30.53 | 31.74 |

$ZETT_{T5-base}$ | | 33.71 | 31.17 |

| Ours | InstructUIE | 39.55 | 35.20 |

*Table 5: Micro-F1 scores of zero-shot RE on FewRel and Wiki-ZSL. The best results are in bold. InstructUIE outperforms SOTA on both datasets.*

##### Relational Extraction

Our model reaches an average F1 of 67.98% on the eight datasets of the RE task, among which the NYT data set reaches 90.47% F1 score. Among the eight datasets, CoNLL2004 and SciERC datasets are also tested by UIE and USM models. We focus on the analysis of the results of these two datasets. For the SciERC dataset, InstructUIE significantly outperforms UIE and USM by 8.62% and 7.79% respectively. For the CoNLL2004 dataset, InstructUIE outperforms UIE by more than three points, and lag USM by less than 0.5%. Moreover, noted that as BERT is usually used for relation classification tasks rather than relation extraction. Therefore, we did not use this baseline in the RE task.

##### Event Extraction

Our model achieve sota on all datasets except for the Event Trigger F1 metric of the CASIE dataset. On the Event Trigger F1 metric, InstructUIE reaches an average of 71.69% on these three datasets, with ACE2005 reaching 77.13%, significantly surpassing UIE’s 73.36%, USM’s 72.41% and Bert’s 72.5%. On the Event Argument F1 metric, InstructUIE beats three baseline models to reach sota on all three datasets. In particular, ACE2005 dataset reaches 72.94%, 18 points higher than the UIE and 17 points higher than the USM.

### 3.2 Experiments on Zero-shot Settings

#### 3.2.1 Dataset

To evaluate InstructUIE’s zero-shot performance, we train the model on 18 NER datasets and 6 RE datasets and test it on 7 NER datasets and 2 RE datasets. Specifically, we eliminate the datasets for zero-shot experimental testing during the training phase. For the NER task, We use five CrossNER subsets(AI, literature, music, politics, science) Liu et al. (2020), MIT Movie Review, and MIT Restaurant Review Liu et al. (2019) to test the zero-shot capability of the model. For RE task, we test the zero-shot capability on FewRel Han et al. (2018) and Wiki-ZSL Chen and Li (2021). For FewRel and Wiki-ZSL data sets, we follow the previous work Chia et al. (2022) and randomly select 5 unseen labels which do not appear in the training set as the test set. In order to reduce the effect of experimental noise, the unseen label selection process is repeated for five different random seeds to produce the test set.

Since the training and testing tasks do not overlap at all and across various domains as well, this setting is challenging.

#### 3.2.2 Baselines

For zero-shot Named Entity Recognition and Relational Extraction, we compare InstructUIE with the following strong baselines:

-

ZETTKim et al. (2022) is a novel framework based on end-to-end generative transformers and outperform previous state-of-the-art models;

-

ChatGPT Ouyang et al. (2022) is also called GPT-3.5-turbo, which is the most capable GPT-3.5 model and optimized for chat;

-

UIE and USM have been introduced in 3.1.2.

#### 3.2.3 Results

| Model | Movie | Restaurant | AI | Literature | Music | Politics | Science | FewRel | Wiki-ZSL |

| davinci | 0.84 | 2.94 | 2.97 | 9.87 | 13.83 | 18.42 | 10.04 | 0.00 | 0.00 |

| chatgpt | 41.00 | 37.76 | 54.40 | 54.07 | 61.24 | 59.12 | 63.00 | 9.96 | 13.14 |

*Table 6: Micro-F1 scores of davinci and chatgpt under zero-shot setting.*

Table 4 and Table 5 show the performance of NER and RE tasks under the zero-shot setting. For the NER task, we can observe that InstructUIE outperforms the current sota model USM in Micro-F1 score on all the datasets except CrossNER_Literature, ranging from 5.21% to 25.27%. For example, compared with the USM model, InstructUIE performs over 20 points better on the MIT Movie Review dataset and the CrossNER_AI dataset. Noted that USM is trained on the same task corpus and tested on the label held out, while our model has never seen the task corpus. For the RE task, under the setting of 5 unseen labels, InstructUIE outperforms the current sota model ZETT on both the FewRel and Wiki-ZSL datasets by 5.84% and 3.46% respectively.

When compared to the GPT series model, InstructUIE significantly outperforms Davinci for the NER task but still falls some way short of Chatgpt’s results for the NER task. However, for the RE task, our model performs much better than these two GPT series models. Both Davinci and Chatgpt perform poorly, especially with Davinci completely unable to output correct results.

It is worth mentioning that since Chatgpt is not open source, we have no way of knowing whether the model has seen the two data sets used by the zero-shot setting during training, and we think the huge difference in results for NER and RE tasks may be due to this reason.

## 4 Related Work

### 4.1 Instruction Tuning

Instruction tuning Mishra et al. (2022); Wang et al. (2022c); Longpre et al. (2023), a novel paradigm that leverages natural language instructions to guide large language models for downstream tasks, shows tremendous promise in generalization within the set of observed tasks. Most recent work Wang et al. (2022c); Longpre et al. (2023) on instruction tuning has focused on general NLP tasks such as question answering and text classification, but not specifically on IE tasks. While some work such as Wang et al. (2022a); Parmar et al. (2022) includes a few IE tasks, those tasks do not provide good coverage of IE tasks and domains. No prior work has examined how training a model on a wide range of IE tasks with various instructions. In this paper, we propose a unified framework for information extraction that involves auxiliary task design as well as specific tuning methods.

### 4.2 Information Extraction

Information extraction is fundamental in natural language processing systems, aiming to extract structured information from unstructured or semi-structured data sources automatically. Traditional methods Wang et al. (2022b); Yan et al. (2021); Huguet Cabot and Navigli (2021); Xie et al. (2021) for IE typically require the design of specific architectures for different IE tasks, and the models are trained separately. However, training dedicated models for different IE tasks requires a significant amount of labeled data, which can be costly and time-consuming to obtain. Secondly, knowledge learned from one IE task cannot be easily applied to another task, even if the tasks have similar characteristics. Recently, Lu et al. (2022) proposed UIE, which uniformly encodes different extraction structures via a structured extraction language and captures the common IE abilities via a large-scale pre-trained text-to-structure model. However, UIE requires separate finetune for different downstream tasks. This lead to the poor performance of UIE in low resource settings or facing new label schema. Lou et al. (2023) proposed USM, which decouples IE into two basic tasks, token-token linking and label-token linking. Unfortunately, USM requires semantic matching for each word, which leads to a significant increase in training and inference time. InstructUIE addresses these challenges by utilizing instructive guidance to direct pre-trained large models toward the task, facilitating the efficient and adaptive generation of target structures.

## 5 Conclusion

In this paper, we propose an end-to-end framework for universal information extraction – InstructUIE, which leverages natural language instructions to guide large language models for IE tasks. We further introduce a new benchmark dataset. The benchmark consists of 32 diverse information extraction datasets that have been unified into a text-to-text format, allowing for a consistent and standardized evaluation of various IE tasks. Experimental results demonstrate that InstructUIE achieves state-of-the-art results under supervised and zero settings and solves massive tasks using a single multi-task model.

## References

- Al-Rfou et al. (2014) Rami Al-Rfou, Vivek Kulkarni, Bryan Perozzi, and Steven Skiena. 2014. POLYGLOT-NER: massive multilingual named entity recognition. CoRR, abs/1410.3791.

- Brown et al. (2020) Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. ArXiv, abs/2005.14165.

- Chen and Li (2021) Chih-Yao Chen and Cheng-Te Li. 2021. Zs-bert: Towards zero-shot relation extraction with attribute representation learning.

- Chen et al. (2022a) Pei Chen, Haotian Xu, Cheng Zhang, and Ruihong Huang. 2022a. Crossroads, buildings and neighborhoods: A dataset for fine-grained location recognition. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3329–3339, Seattle, United States. Association for Computational Linguistics.

- Chen et al. (2022b) Pei Chen, Haotian Xu, Cheng Zhang, and Ruihong Huang. 2022b. Crossroads, buildings and neighborhoods: A dataset for fine-grained location recognition. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3329–3339, Seattle, United States. Association for Computational Linguistics.

- Chen et al. (2023) Xuanting Chen, Junjie Ye, Can Zu, Nuo Xu, Rui Zheng, Minlong Peng, Jie Zhou, Tao Gui, Qi Zhang, and Xuanjing Huang. 2023. How robust is gpt-3.5 to predecessors? a comprehensive study on language understanding tasks. arXiv preprint arXiv:2303.00293.

- Chia et al. (2022) Yew Ken Chia, Lidong Bing, Soujanya Poria, and Luo Si. 2022. Relationprompt: Leveraging prompts to generate synthetic data for zero-shot relation triplet extraction.

- Chung et al. (2022) Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416.

- Derczynski et al. (2016) Leon Derczynski, Kalina Bontcheva, and Ian Roberts. 2016. Broad Twitter corpus: A diverse named entity recognition resource. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 1169–1179, Osaka, Japan. The COLING 2016 Organizing Committee.

- Devlin et al. (2019) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

- Dogan et al. (2014) Rezarta Islamaj Dogan, Robert Leaman, and Zhiyong Lu. 2014. Ncbi disease corpus: A resource for disease name recognition and concept normalization. Journal of biomedical informatics, 47:1–10.

- Guan (2022) Runwei Guan. 2022. Findvehicle and vehiclefinder: A ner dataset for a text-image cross-modal vehicle retrieval system.

- Gurulingappa et al. (2012) Harsha Gurulingappa, Abdul Mateen Rajput, Angus Roberts, Juliane Fluck, Martin Hofmann-Apitius, and Luca Toldo. 2012. Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports. Journal of Biomedical Informatics, 45(5):885–892. Text Mining and Natural Language Processing in Pharmacogenomics.

- Han et al. (2018) Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018. Fewrel: A large-scale supervised few-shot relation classification dataset with state-of-the-art evaluation.

- Hendrickx et al. (2010) Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid Ó Séaghdha, Sebastian Padó, Marco Pennacchiotti, Lorenza Romano, and Stan Szpakowicz. 2010. Semeval-2010 task 8: Multi-way classification of semantic relations between pairs of nominals. In *SEMEVAL.

- Hovy et al. (2006) Eduard H. Hovy, Mitchell P. Marcus, Martha Palmer, Lance A. Ramshaw, and Ralph M. Weischedel. 2006. Ontonotes: The 90% solution. In North American Chapter of the Association for Computational Linguistics.

- Huguet Cabot and Navigli (2021) Pere-Lluís Huguet Cabot and Roberto Navigli. 2021. REBEL: Relation extraction by end-to-end language generation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2370–2381, Punta Cana, Dominican Republic. Association for Computational Linguistics.

- Jat et al. (2018) Sharmistha Jat, Siddhesh Khandelwal, and Partha Pratim Talukdar. 2018. Improving distantly supervised relation extraction using word and entity based attention. ArXiv, abs/1804.06987.

- Kim et al. (2022) Bosung Kim, Hayate Iso, Nikita Bhutani, Estevam Hruschka, and Ndapa Nakashole. 2022. Zero-shot triplet extraction by template infilling.

- Kim et al. (2003a) Jin-Dong Kim, Tomoko Ohta, Yuka Tateisi, and Junichi Tsujii. 2003a. Genia corpus - a semantically annotated corpus for bio-textmining. Bioinformatics, 19 Suppl 1:i180–2.

- Kim et al. (2003b) Jin-Dong Kim, Tomoko Ohta, Yuka Tateisi, and Jun’ichi Tsujii. 2003b. Genia corpus—a semantically annotated corpus for bio-textmining. Bioinformatics (Oxford, England), 19 Suppl 1:i180–2.

- Kocaman and Talby (2020a) Veysel Kocaman and David Talby. 2020a. Biomedical named entity recognition at scale. In ICPR Workshops.

- Kocaman and Talby (2020b) Veysel Kocaman and David Talby. 2020b. Biomedical named entity recognition at scale. CoRR, abs/2011.06315.

- Kocaman and Talby (2020c) Veysel Kocaman and David Talby. 2020c. Biomedical named entity recognition at scale. CoRR, abs/2011.06315.

- Kocaman and Talby (2022a) Veysel Kocaman and David Talby. 2022a. Accurate clinical and biomedical named entity recognition at scale. Software Impacts, 13:100373.

- Kocaman and Talby (2022b) Veysel Kocaman and David Talby. 2022b. Accurate clinical and biomedical named entity recognition at scale. Software Impacts, 13:100373.

- Krallinger et al. (2015) Martin Krallinger, Obdulia Rabal, Florian Leitner, Miguel Vazquez, David Salgado, Zhiyong lu, Robert Leaman, Yanan Lu, Donghong Ji, Daniel Lowe, Roger Sayle, Riza Batista-Navarro, Rafal Rak, Torsten Huber, Tim Rocktäschel, Sérgio Matos, David Campos, Buzhou Tang, Wang Qi, and Alfonso Valencia. 2015. The chemdner corpus of chemicals and drugs and its annotation principles. Journal of Cheminformatics, 7:S2.

- Kumar and Starly (2021a) Aman Kumar and Binil Starly. 2021a. “fabner”: information extraction from manufacturing process science domain literature using named entity recognition. Journal of Intelligent Manufacturing, 33:2393 – 2407.

- Kumar and Starly (2021b) Aman Kumar and Binil Starly. 2021b. “fabner”: information extraction from manufacturing process science domain literature using named entity recognition. Journal of Intelligent Manufacturing, 33.

- Lai et al. (2018) Sunny Lai, Kwong Sak Leung, and Yee Leung. 2018. SUNNYNLP at SemEval-2018 task 10: A support-vector-machine-based method for detecting semantic difference using taxonomy and word embedding features. In Proceedings of the 12th International Workshop on Semantic Evaluation, pages 741–746, New Orleans, Louisiana. Association for Computational Linguistics.

- Li et al. (2016) Jiao Li, Yueping Sun, Robin J. Johnson, Daniela Sciaky, Chih-Hsuan Wei, Robert Leaman, Allan Peter Davis, Carolyn J. Mattingly, Thomas C. Wiegers, and Zhiyong Lu. 2016. Biocreative v cdr task corpus: a resource for chemical disease relation extraction. Database: The Journal of Biological Databases and Curation, 2016.

- Li et al. (2019) Xiaoya Li, Xiaofei Sun, Yuxian Meng, Junjun Liang, Fei Wu, and Jiwei Li. 2019. Dice loss for data-imbalanced NLP tasks. CoRR, abs/1911.02855.

- Liu et al. (2019) Yijin Liu, Fandong Meng, Jinchao Zhang, Jinan Xu, Yufeng Chen, and Jie Zhou. 2019. GCDT: A global context enhanced deep transition architecture for sequence labeling. CoRR, abs/1906.02437.

- Liu et al. (2020) Zihan Liu, Yan Xu, Tiezheng Yu, Wenliang Dai, Ziwei Ji, Samuel Cahyawijaya, Andrea Madotto, and Pascale Fung. 2020. Crossner: Evaluating cross-domain named entity recognition.

- Longpre et al. (2023) S. Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. 2023. The flan collection: Designing data and methods for effective instruction tuning. ArXiv, abs/2301.13688.

- Lou et al. (2023) Jie Lou, Yaojie Lu, Dai Dai, Wei Jia, Hongyu Lin, Xianpei Han, Le Sun, and Hua Wu. 2023. Universal information extraction as unified semantic matching.

- Lu et al. (2021) Yaojie Lu, Hongyu Lin, Jin Xu, Xianpei Han, Jialong Tang, Annan Li, Le Sun, M. Liao, and Shaoyi Chen. 2021. Text2event: Controllable sequence-to-structure generation for end-to-end event extraction. ArXiv, abs/2106.09232.

- Lu et al. (2022) Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun, and Hua Wu. 2022. Unified structure generation for universal information extraction.

- Luan et al. (2018) Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi. 2018. Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction.

- Mishra et al. (2022) Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022. Cross-task generalization via natural language crowdsourcing instructions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470–3487, Dublin, Ireland. Association for Computational Linguistics.

- OpenAI (2023) OpenAI. 2023. Gpt-4 technical report. ArXiv, abs/2303.08774.

- openbiocorpora (2015) openbiocorpora. 2015. openbiocorpora anatem.

- Ouyang et al. (2022) Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. 2022. Training language models to follow instructions with human feedback. ArXiv, abs/2203.02155.

- Pan et al. (2017a) Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017a. Cross-lingual name tagging and linking for 282 languages. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1946–1958, Vancouver, Canada. Association for Computational Linguistics.

- Pan et al. (2017b) Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017b. Cross-lingual name tagging and linking for 282 languages. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1946–1958, Vancouver, Canada. Association for Computational Linguistics.

- Parmar et al. (2022) Mihir Parmar, Swaroop Mishra, Mirali Purohit, Man Luo, Murad Mohammad, and Chitta Baral. 2022. In-BoXBART: Get instructions into biomedical multi-task learning. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 112–128, Seattle, United States. Association for Computational Linguistics.

- Poolsawad et al. (2014) N Poolsawad, C Kambhampati, and JGF Cleland. 2014. Balancing class for performance of classification with a clinical dataset. In proceedings of the World Congress on Engineering, volume 1, pages 1–6.

- Riedel et al. (2010) Sebastian Riedel, Limin Yao, and Andrew McCallum. 2010. Modeling relations and their mentions without labeled text. In ECML/PKDD.

- Roth and tau Yih (2004) Dan Roth and Wen tau Yih. 2004. A linear programming formulation for global inference in natural language tasks. In Conference on Computational Natural Language Learning.

- Sang and Meulder (2003) Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: Language-independent named entity recognition.

- Schweter and Akbik (2020) Stefan Schweter and Alan Akbik. 2020. FLERT: document-level features for named entity recognition. CoRR, abs/2011.06993.

- Sun et al. (2022) Zhao-Li Sun, Jiazheng Li, Gabriele Pergola, Byron C. Wallace, Bino John, Nigel Greene, Joseph Kim, and Yulan He. 2022. Phee: A dataset for pharmacovigilance event extraction from text. ArXiv, abs/2210.12560.

- Takanobu et al. (2018) Ryuichi Takanobu, Tianyang Zhang, Jiexi Liu, and Minlie Huang. 2018. A hierarchical framework for relation extraction with reinforcement learning. In AAAI Conference on Artificial Intelligence.

- Tang et al. (2022) Minghao Tang, Peng Zhang, Yongquan He, Yongxiu Xu, Chengpeng Chao, and Hongbo Xu. 2022. DoSEA: A domain-specific entity-aware framework for cross-domain named entity recogition. In Proceedings of the 29th International Conference on Computational Linguistics, pages 2147–2156, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

- Tedeschi et al. (2021) Simone Tedeschi, Valentino Maiorca, Niccolò Campolungo, Francesco Cecconi, and Roberto Navigli. 2021. WikiNEuRal: Combined neural and knowledge-based silver data creation for multilingual NER. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2521–2533, Punta Cana, Dominican Republic. Association for Computational Linguistics.

- Tedeschi and Navigli (2022a) Simone Tedeschi and Roberto Navigli. 2022a. MultiNERD: A multilingual, multi-genre and fine-grained dataset for named entity recognition (and disambiguation). In Findings of the Association for Computational Linguistics: NAACL 2022, pages 801–812, Seattle, United States. Association for Computational Linguistics.

- Tedeschi and Navigli (2022b) Simone Tedeschi and Roberto Navigli. 2022b. MultiNERD: A multilingual, multi-genre and fine-grained dataset for named entity recognition (and disambiguation). In Findings of the Association for Computational Linguistics: NAACL 2022, pages 801–812, Seattle, United States. Association for Computational Linguistics.

- Theodoropoulos and Moens (2023) Christos Theodoropoulos and Marie-Francine Moens. 2023. An information extraction study: Take in mind the tokenization!

- Ushio and Camacho-Collados (2021) Asahi Ushio and Jose Camacho-Collados. 2021. T-NER: An all-round python library for transformer-based named entity recognition. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations. Association for Computational Linguistics.

- Ushio et al. (2022a) Asahi Ushio, Leonardo Neves, Vitor Silva, Francesco Barbieri, and Jose Camacho-Collados. 2022a. Named entity recognition in twitter: A dataset and analysis on short-term temporal shifts.

- Ushio et al. (2022b) Asahi Ushio, Leonardo Neves, Vitor Silva, Francesco Barbieri, and Jose Camacho-Collados. 2022b. Named entity recognition in twitter: A dataset and analysis on short-term temporal shifts.

- Walker and Consortium (2005) C. Walker and Linguistic Data Consortium. 2005. ACE 2005 Multilingual Training Corpus. LDC corpora. Linguistic Data Consortium.

- Wang et al. (2023) Chenguang Wang, Xiao Liu, Zui Chen, Haoyun Hong, Jie Tang, and Dawn Song. 2023. Deepstruct: Pretraining of language models for structure prediction.

- Wang et al. (2022a) Liwen Wang, Rumei Li, Yang Yan, Yuanmeng Yan, Sirui Wang, Wei Yu Wu, and Weiran Xu. 2022a. Instructionner: A multi-task instruction-based generative framework for few-shot ner. ArXiv, abs/2203.03903.

- Wang et al. (2022b) Xiao Wang, Shihan Dou, Limao Xiong, Yicheng Zou, Qi Zhang, Tao Gui, Liang Qiao, Zhanzhan Cheng, and Xuanjing Huang. 2022b. MINER: Improving out-of-vocabulary named entity recognition from an information theoretic perspective. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5590–5600, Dublin, Ireland. Association for Computational Linguistics.

- Wang et al. (2021a) Xinyu Wang, Yong Jiang, Nguyen Bach, Tao Wang, Zhongqiang Huang, Fei Huang, and Kewei Tu. 2021a. Automated concatenation of embeddings for structured prediction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2643–2660, Online. Association for Computational Linguistics.

- Wang et al. (2021b) Yijun Wang, Changzhi Sun, Yuanbin Wu, Hao Zhou, Lei Li, and Junchi Yan. 2021b. UniRE: A unified label space for entity relation extraction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 220–231, Online. Association for Computational Linguistics.

- Wang et al. (2022c) Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. 2022c. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

- Xie et al. (2021) Chenhao Xie, Jiaqing Liang, Jingping Liu, Chengsong Huang, Wenhao Huang, and Yanghua Xiao. 2021. Revisiting the negative data of distantly supervised relation extraction. CoRR, abs/2105.10158.

- Yan et al. (2021) Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, and Zheng Zhang. 2021. A unified generative framework for aspect-based sentiment analysis. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2416–2429, Online. Association for Computational Linguistics.

- Ye et al. (2021) Deming Ye, Yankai Lin, and Maosong Sun. 2021. Pack together: Entity and relation extraction with levitated marker. CoRR, abs/2109.06067.

- Ye et al. (2023) Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, et al. 2023. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. arXiv preprint arXiv:2303.10420.

- Zhang and Wang (2015) Dongxu Zhang and Dong Wang. 2015. Relation classification via recurrent neural network.

- Zhang et al. (2023) Sheng Zhang, Hao Cheng, Jianfeng Gao, and Hoifung Poon. 2023. Optimizing bi-encoder for named entity recognition via contrastive learning.

- Zhong and Chen (2020) Zexuan Zhong and Danqi Chen. 2020. A frustratingly easy approach for joint entity and relation extraction. CoRR, abs/2010.12812.

## 6 Appendix

### 6.1 Data Details

IE INSTRUCTIONS collects 32 publicly available datasets covering three IE tasks: NER, RE, and EE. For NER(named entity extraction) task, the 21 used datasets includes ACE2004, ACE2005Walker and Consortium (2005), broad_twitter_corpusDerczynski et al. (2016), CoNLL2003Sang and Meulder (2003), multiNERDTedeschi and Navigli (2022a), OntonotesHovy et al. (2006), polyglot-NERAl-Rfou et al. (2014), tweetNER7Ushio et al. (2022a), wikiannPan et al. (2017a), wikineuralTedeschi et al. (2021), AnatEMopenbiocorpora (2015), bc2gmKocaman and Talby (2020a), bc4chemdKrallinger et al. (2015), bc5cdrLi et al. (2016), FabNERKumar and Starly (2021a), FindVehicleGuan (2022), GENIAKim et al. (2003b), HarveyNERChen et al. (2022a), MIT Movie ReviewLiu et al. (2019), MIT Restaurant ReviewLiu et al. (2019) and ncbi-diseaseDogan et al. (2014). For RE(relation extraction) task, we use 10 datasets including ADE corpusGurulingappa et al. (2012), CoNLL2004Roth and tau Yih (2004), GIDSJat et al. (2018), kbp37Zhang and Wang (2015), NYTRiedel et al. (2010), NYT11 HRLTakanobu et al. (2018), SciERCLuan et al. (2018), semeval REHendrickx et al. (2010), FewRelHan et al. (2018) and Wiki-ZSLChen and Li (2021). For task EE(event extraction), ACE2005Walker and Consortium (2005), CASIELu et al. (2021), GENIAKim et al. (2003a) and PHEESun et al. (2022) are used.

For the data set with only training set as the original data, we divided it into training set, validation set and test set according to the ratio of 8:1:1. For the data set with only training set and validation set as the original data, we randomly select half of the data in the validation set as the test set and the other half as the new validation set. For other datasets, we adopt the official split.

Tabel 7 shows detailed datasets statistics. NER refers to Named Entity Recognition task, RE refers to Relation Extraction task, and EE refers to Event Extraction task. |Labels| indicates the number of labels, and $\#$ is the number of sentences in the specific subset. For the |Labels| of event extraction, the number outside the parenthesis indicates the number of event types and the number inside the parenthesis indicates the number of argument types.

| Task | Dataset | |labels| | #Train | #Val | #Test |

| NER | ACE2004 | 7 | 6202 | 745 | 812 |

| ACE2005 | 7 | 7299 | 971 | 1060 |

| broad_twitter_corpus | 3 | 5334 | 2000 | 2001 |

| CoNLL2003 | 4 | 14041 | 3250 | 3453 |

| multiNERD | 16 | 134144 | 10000 | 10000 |

| Ontonotes | 18 | 59924 | 8528 | 8262 |

| polyglot-NER | 3 | 393982 | 10000 | 10000 |

| tweetNER7 | 7 | 7111 | 886 | 576 |

| wikiann | 3 | 20000 | 10000 | 10000 |

| wikineural | 3 | 92720 | 11590 | 11597 |

| AnatEM | 1 | 5861 | 2118 | 3830 |

| bc2gm | 1 | 12500 | 2500 | 5000 |

| bc4chemd | 1 | 30682 | 30639 | 26364 |

| bc5cd | 2 | 4560 | 4581 | 4797 |

| CrossNER_AI | 14 | 100 | 350 | 431 |

| CrossNER_literature | 12 | 100 | 400 | 416 |

| CrossNER_music | 13 | 100 | 380 | 465 |

| CrossNER_politics | 9 | 199 | 540 | 650 |

| CrossNER_science | 17 | 200 | 450 | 543 |

| FabNER | 12 | 9435 | 2182 | 2064 |

| FindVehicle | 21 | 21565 | 20777 | 20777 |

| GENIA | 5 | 15023 | 1669 | 1854 |

| HarveyNER | 4 | 3967 | 1301 | 1303 |

| MIT Movie Review | 12 | 9774 | 2442 | 2442 |

| MIT Restaurant Review | 8 | 7659 | 1520 | 1520 |

| ncbi-disease | 1 | 5432 | 923 | 940 |

| RE | ADE corpus | 1 | 3417 | 427 | 428 |

| CoNLL2004 | 5 | 922 | 231 | 288 |

| GIDS | 4 | 8526 | 1417 | 4307 |

| kbp37 | 18 | 15917 | 1724 | 3405 |

| NYT | 24 | 56196 | 5000 | 5000 |

| NYT11 HRL | 12 | 62648 | 149 | 369 |

| SciERC | 7 | 1366 | 187 | 397 |

| semeval RE | 10 | 6507 | 1493 | 2717 |

| EE | ACE2005 | 33(22) | 3342 | 327 | 293 |

| CASIE | 5(26) | 3751 | 788 | 1500 |

| GENIA | 5(0) | 15023 | 1669 | 1854 |

| PHEE | 2(16) | 2898 | 961 | 968 |

*Table 7: Detailed datasets statistics.*

### 6.2 Instruction Details

Table 8 shows prompts for different tasks. NER refers to the named entity recognition task, the object of which is the entity in the output sentence and its corresponding entity type. RE refers to the relation extraction task, the object of which is to extract the relation triplet in the sentence, including the relation name, the head entity and the tail entity. EE refers to the event extraction task. The task objective is to extract the event types, trigger word and arguments in the sentence. ES refers to entity span, the task target is given sentence and entity category options, and output entities that conform to the entity category, but there is no need to output the entity type of each entity; ET refers to entity type identification. The task target is a given sentence, which contains entity and entity category options, and outputs the entity category corresponding to each entity. EP refers to entity pair identification (entity pair). The task target is given sentence and relation category options, and output entity pairs that conform to relation category, but do not need to output its relation category; EPR refers to entity pair relationship identification. The task target is a given sentence, which contains entity pair and relationship category options, and outputs the corresponding relationship category for each entity pair. ES and ET are auxiliary tasks of NER, EP and EPR are auxiliary tasks of RE, and EEA and EET are auxiliary tasks of EE.

| Task | Prompts |

| NER | Please list all entity words in the text that fit the category. Output format is "type1: word1; type2: word2". |

| | |

| | Please find all the entity words associated with the category in the given text. Output format is "type1: word1; type2: word2". |

| | |

| | Please tell me all the entity words in the text that belong to a given category. Output format is "type1: word1; type2: word2". |

| RE | Given a phrase that describes the relationship between two words, extract the words and the lexical relationship between them. The output format should be "relation1: word1, word2; relation2: word3, word4". |

| | |

| | Find the phrases in the following sentences that have a given relationship. The output format is "relation1: word1, word2; relation2: word3, word4". |

| | |

| | Given a sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of "relation1: word1, word2; relation2: word3, word4". |

| EE | Locate the role in the text that participated in the event based on the event type and return it in the event list. |

| | |

| | Extract the event information in the text and return them in the event list. |

| ES | Please list all entity words in the text that fit the category. Output format is word1, word2. |

| ET | Given options, please tell me the categories of all the listed entity words.Output format is "type1: word1; type2: word2". |

| EP | Please list all entity pairs containing a certain relationship in the given options.Output format is "word1, word2; word3, word4". |

| EPR | Given options, please tell me the relationships of all the listed entity pairs.Output format is "relation1: word1, word2; relation2: word3, word4". |

| EEA | Given event type and trigger, please tell me the arguments of all the listed option. Output format is "name: role". |

| EET | Please tell me event type and its trigger word from given type options. Output format is "event type: trigger". |

*Table 8: Instructions for different tasks. *

| Dataset | Metric | Task-specific SOTA Methods |

| ACE2004 | Entity F1 | Zhong and Chen (2020) | 90.3 |

| ACE2005-Ent | Entity F1 | Zhong and Chen (2020) | 90.9 |

| AnatEM | Entity F1 | Kocaman and Talby (2022a) | 91.65 |

| bc2gm | Entity F1 | Kocaman and Talby (2020b) | 88.75 |

| bc4chemd | Entity F1 | Kocaman and Talby (2022b) | 94.39 |

| bc5cdr | Entity F1 | Zhang et al. (2023) | 91.9 |

| broad_twitter_corpus | Entity F1 | Wang et al. (2021b) | 74.70 |

| CoNLL2003 | Entity F1 | Wang et al. (2021a) | 94.60 |

| FabNER | Entity F1 | Kumar and Starly (2021b) | 88 |

| FindVehicle | Entity F1 | Schweter and Akbik (2020) | 80.9 |

| GENIA-Ent | Entity F1 | Wang et al. (2023) | 80.80 |

| HarveyNER | Entity F1 | Chen et al. (2022b) | 68.97 |

| MIT Movie Review | Entity F1 | Tang et al. (2022) | 87.31 |

| MIT Restaurant Review | Entity F1 | Ushio and Camacho-Collados (2021) | 79.6 |

| multiNERD | Entity F1 | Tedeschi and Navigli (2022b) | 85.0 |

| ncbi-disease | Entity F1 | Kocaman and Talby (2020c) | 90.48 |

| Ontonotes | Entity F1 | Li et al. (2019) | 92.07 |

| polyglot-NER | Entity F1 | - | |

| tweetNER7 | Entity F1 | Ushio et al. (2022b) | 66 |

| wikiann | Entity F1 | Pan et al. (2017b) | 91.8 |

| wikineural | Entity F1 | - | |

| ADE corpus | Relation Strict F1 | Theodoropoulos and Moens (2023) | 83.9 |

| CoNLL2004 | Relation Strict F1 | Huguet Cabot and Navigli (2021) | 76.65 |

| GIDS | Relation Strict F1 | - | |

| kbp37 | Relation Strict F1 | - | |

| NYT | Relation Strict F1 | - | |

| NYT11 HRL | Relation Strict F1 | Xie et al. (2021) | 55.47 |

| SciERC | Relation Strict F1 | Ye et al. (2021) | 38.40 |

| semeval RE | Relation Strict F1 | Lai et al. (2018) | 76.00 |

| ACE2005 | Event Trigger F1 | - | - |

| ACE2005 | Event Argument F1 | - | - |

| CASIE | Event Trigger F1 | - | - |

| CASIE | Event Argument F1 | - | - |

| GENIA-Evt | Event Trigger F1 | - | 63.96 |

| GENIA-Evt | Event Argument F1 | - | - |

| PHEE | Event Trigger F1 | - | - |

| PHEE | Event Argument F1 | - | - |

*Table 9: Overall results of InstructUIE on different datasets. InstructUIE perform better or comparable than Bert on popular NER datasets like ACE2005, CoNLL2003, Ontonotes, and tweetNER7. In the RE task, InstructUIE achieved results comparable to the baseline on most datasets. In the EE task, our model outperformed USM, UIE or SOTA on some datasets. *

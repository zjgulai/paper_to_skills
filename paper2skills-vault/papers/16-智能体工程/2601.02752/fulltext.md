<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2601.02752
     paper_id : 2601.02752
     source   : https://arxiv.org/html/2601.02752v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# EComStage: Stage-wise and Orientation-specific Benchmarking for Large Language Models in E-commerce

Kaiyan Zhao† Affiliation: The University of Tokyo Email: 11pt12ptqcrcaoshaosheng@xiaohongshu.com    Zijie Meng† Affiliation: Zhejiang University    Zheyong Xie Affiliation: Xiaohongshu Inc.    Jin Duan Affiliation: Xiaohongshu Inc.    Yao Hu Affiliation: Xiaohongshu Inc.    Zuozhu Liu Affiliation: Zhejiang University    Shaosheng Cao‡ Affiliation: Xiaohongshu Inc.

###### Abstract

Large Language Model (LLM)-based agents are increasingly deployed in e-commerce applications to assist customer services in tasks such as product inquiries, recommendations, and order management. Existing benchmarks primarily evaluate whether these agents successfully complete the final task, overlooking the intermediate reasoning stages that are crucial for effective decision-making. To address this gap, we propose EComStage, a unified benchmark for evaluating agent-capable LLMs across the comprehensive stage-wise reasoning process: Perception (understanding user intent), Planning (formulating an action plan), and Action (executing the decision). EComStage evaluates LLMs through seven separate representative tasks spanning diverse e-commerce scenarios, with all samples human-annotated and quality-checked. Unlike prior benchmarks that focus only on customer-oriented interactions, EComStage also evaluates merchant-oriented scenarios, including promotion management, content review, and operational support relevant to real-world applications. We evaluate a wide range of over 30 LLMs, spanning from 1B to over 200B parameters, including open-source models and closed-source APIs, revealing stage/orientation-specific strengths and weaknesses. Our results provide fine-grained, actionable insights for designing and optimizing LLM-based agents in real-world e-commerce settings.

## 1 Introduction

Large Language Model (LLM)-based agents have demonstrated remarkable reasoning and decision-making abilities across a wide range of complex tasks (Yang et al., 2023; Luo et al., 2025). They have been widely adopted in real-world applications, such as code generation (Dong et al., 2025) and gaming (Hu et al., 2025).

Among these domains, e-commerce stands out as one of the most promising and challenging areas for LLM-based agents. These agents are expected to enhance customers’ online shopping experiences by handling diverse and goal-oriented interactions Zeng et al. (2025). To achieve this, they must interpret user intents accurately while dynamically invoking a variety of tools such as databases and APIs to browse products, manage transactions, and resolve service issues Yao et al. (2024); Feng et al. (2024).

The practical value of LLM-based agents in e-commerce lies in their ability to automate large portions of customer interaction workflows. Major online retailers and service providers have begun integrating such agents into their platforms to assist with product search, order management, and customer support, demonstrating their growing practical impact Yao et al. (2022); Cheng et al. (2024); Zeng et al. (2025).

Given their growing importance in e-commerce applications, accurately evaluating the performance of LLM-based agents has become an emerging research focus. Several e-commerce–specific benchmarks have been proposed. For example, $\tau$-bench Yao et al. (2024) introduces evaluation scenarios in the retail and airline domains to assess agents’ capabilities in customer service assistance. ECom-Bench Wang et al. (2025a) emphasizes the evaluation of agents’ multimodal reasoning abilities, while Mix-Ecom Zhou et al. (2025) focuses on mixed-type e-commerce dialogues, such as multi-intent conversations involving refund requests followed by product recommendations.

*Figure 1: Stage-wise reasoning process of an LLM-based agent when handling an e-commerce requirement.*

However, existing e-commerce benchmarks share one common limitation: they primarily evaluate whether the agent successfully completes the final task, overlooking the intermediate stages that are crucial for effective decision-making. Before reaching a final decision, the backbone LLMs typically engage in a multi-stage reasoning process, and these stages play a vital role in overall performance. As illustrated in Figure 1, when given a single e-commerce requirement, we can decompose the agent’s reasoning process into three stages: (1) Perception: recognizing and summarizing the true intent of the customer; (2) Planning: reasoning and formulating a plan of action; and (3) Action: executing the appropriate decision to assist the customer. Each stage contributes a distinct aspect of reasoning, and together they enable the agent to achieve the final goal. Throughout this process, the agent must continuously interpret the customer’s current intent to decide the most appropriate next step. Unfortunately, existing e-commerce benchmarks lack a comprehensive evaluation of these reasoning stages, focusing instead solely on the final task success rates.

Moreover, understanding how agent-capable LLMs perform in these stages is crucial for improving real-world applications. For instance, failures in the Perception stage often lead to misunderstanding customer intent, causing irrelevant responses or poor recommendations; weaknesses in Planning can result in inefficient tool use or inconsistent task strategies; and errors in Action directly affect customer satisfaction and business outcomes. Therefore, evaluating LLMs that serve as the backbone of e-commerce agents across these stages provides actionable insights for improving user experience in real e-commerce applications.

While some prior work has emphasized the importance of these intermediate reasoning stages Liu et al. (2024); Isa et al. (2024), and others have considered Perception as a distinct step Huang et al. (2025), an integrated evaluation framework encompassing all three stages remains missing, as annotating and evaluating intermediate stages remains challenging and costly.

To this end, we propose EComStage, a comprehensive benchmark for evaluating LLMs that serve as the backbone of e-commerce agents across the full three-stage reasoning process. Unlike previous benchmarks that focus solely on end-to-end task outcomes, EComStage decomposes the e-commerce reasoning process into separate, stage-specific tasks, each designed to evaluate a distinct capability of the agent-capable LLM. Specifically, EComStage introduces seven representative tasks covering Perception, Planning, and Action abilities under diverse e-commerce settings, such as product inquiry, recommendation, and refund handling. This design enables a more fine-grained and interpretable evaluation, helping both researchers and practitioners identify stage-specific weaknesses and optimize agent behavior for real-world applications. To ensure reliability and alignment with real-world settings, all samples are human-annotated and quality-checked by professional annotators with e-commerce experience. In addition to traditional customer-service tasks, EComStage explicitly includes merchant-oriented operational scenarios, such as promotion management, content review, and refund handling for advertisers. This extension allows evaluation of LLM-based agents in supporting both customers and merchants, reflecting the full spectrum of real-world e-commerce interactions.

We evaluate a diverse set of over 30 agent-capable LLMs on EComStage, ranging from standard open-source LLMs such as LLaMA3 Grattafiori et al. (2024) and Qwen3 Yang et al. (2025) series to closed-source APIs including GPT-4o OpenAI et al. (2024) and Gemini 2.5 Pro Comanici et al. (2025), with model sizes spanning from 1B to over 200B parameters. By benchmarking this wide range of models, we aim to demonstrate both the generalizability of EComStage and the stage/orientation-specific strengths and weaknesses of current LLM-based agents in realistic e-commerce scenarios. Through extensive evaluation, our experiments reveal that no single model consistently excels across all stages (Perception, Planning, Action) or orientations (merchant-oriented, customer-oriented). This finding underscores the importance of a stage-wise and orientation-aware benchmark.

In summary, our paper makes the following contributions:

-

We propose EComStage, which provides a stage-wise evaluation framework for agent-capable LLMs in e-commerce, highlighting intermediate reasoning that is often overlooked in existing benchmarks.

-

EComStage introduces seven representative tasks covering diverse real-world scenarios including both customer and merchant orientation, with all samples human-annotated and quality-checked, enabling fine-grained, stage-specific evaluation.

-

By analyzing multiple models of varying sizes and capabilities, our benchmark offers valuable, actionable insights into agent performance, guiding both research and practical deployment in real-world e-commerce systems.

## 2 Related Works

| Benchmark | Scale | Tested Models | Customer-oriented | Merchant-oriented | Stage-wise Evaluation | Availability |

| ECom-Bench Wang et al. (2025a) | 53† | 7 | ✓ | ✗ | ✗ | ✓ |

| Mix-Ecom Zhou et al. (2025) | 4799 | 6 | ✓ | ✗ | ✗ | Not yet |

| EComStage (ours) | 4804 | 33 | ✓ | ✓ | ✓ | ✓ |

*Table 1: Comparison of Existing E-Commerce Benchmarks and EComStage. †Based on the released ECom-Bench repository, we observe only 53 samples.*

##### LLM-based Agents in E-commerce

LLM-based agents have recently gained widespread attention for their potential to automate complex, multi-step workflows across different domains Xi et al. (2023); Guo et al. (2024); Luo et al. (2025). By integrating the reasoning capabilities from LLMs, these agents can interact with users, retrieve external information, and execute tasks autonomously. In the e-commerce industry, LLM-based agents hold particular significance as they directly influence customer experience, operational efficiency, and seller profitability through intelligent service automation and personalized interactions Zeng et al. (2025).

##### Early E-commerce Datasets

Despite this growing importance, evaluation of LLM-based agents’ performance in e-commerce scenarios remains underexplored. Early e-commerce–related datasets often originate as subsets or adaptations of multi-domain task-oriented datasets, e.g., MultiWOZ Budzianowski et al. (2018), which primarily focuses on limited scenarios, such as booking and reservation rather than real e-commerce operations. Later e-commerce datasets begin to include domain-specific elements such as product recommendation Jia et al. (2022); Liu et al. (2023) and content detection Xu et al. (2025), yet they remain limited in scope, often modeling only one aspect of the shopping experience rather than the diverse interactions found in real platforms.

##### E-commerce Benchmarks

Recent works have begun to address these limitations with e-commerce–specific benchmarks. $\tau$-Bench Yao et al. (2024) simulates realistic tool–agent–user interactions in the retail and airline domains, while ECom-Bench Wang et al. (2025a) assesses agents’ capabilities in resolving customer support issues, including multimodal reasoning and interaction with structured data. Mix-Ecom Zhou et al. (2025) introduces mixed-type e-commerce dialogues, covering pre-sales, logistics, and after-sales scenarios. It evaluates agents’ ability to handle complex multi-turn interactions. While these benchmarks have advanced evaluation in e-commerce, they mainly focus on customer-oriented tasks and lack fine-grained analysis of intermediate reasoning stages. EComScriptBench Wang et al. (2025b) introduces step-wise evaluation in script planning, but its scope is restricted and does not cover broader agent behavior.

To ensure safe and reliable deployment, it is essential to evaluate not only whether agents complete a task, but also how effectively they reason through intermediate stages. Our EComStage enables a more comprehensive evaluation of the capabilities of LLMs that serve as the backbone of e-commerce agents, capturing performance from different stakeholder perspectives and supporting both customer- and merchant-oriented scenarios. We summarize the main differences between existing e-commerce benchmarks and EComStage in Table 1.

## 3 EComStage

### 3.1 Dataset Construction

To ensure both realism and safety, we construct EComStage through a multi-stage pipeline that integrates real-world e-commerce data collection, expert annotation, and rigorous multi-level filtering as shown in Figure 2.

*Figure 2: Pipeline for dataset construction.*

#### 3.1.1 Data Collection

We first collect business data from real e-commerce service scenarios, covering both customer and merchant orientations (e.g., customer inquiries, promotion management, advertising content review). These data are drawn from authentic operational contexts to ensure that each task reflects realistic e-commerce interaction patterns. From this data, we define seven representative tasks, chosen because they are the most frequently encountered and critical in real-world e-commerce workflows, ensuring that our benchmark evaluates capabilities relevant to practical applications.

#### 3.1.2 Human Annotation

All collected samples are manually annotated by professional annotators with e-commerce experience. Annotators identify the task types, understand user intents, determine correct actions, and write corresponding ground-truth responses following well-defined task guidelines on an online platform. This process ensures high-quality supervision and domain alignment for each task. Guidelines for human annotators are provided in Figure 4, Appendix.

#### 3.1.3 Task-specific Filtering

Data from each task undergoes an independent filtering phase to eliminate legal, security, or privacy risks through the following steps: (a) Removal of company identifiers and overly detailed internal operational procedures; (b) Rewriting or merging of text involving product rules or classification results to maintain generality; (c) Anonymization of sensitive user information, including names, phone numbers, addresses, ID numbers, order details, and product metadata; and (d) For samples containing images, all identifiable personal information is blurred or replaced.

#### 3.1.4 Global Filtering and Verification

After task-level cleaning, all tasks are processed through a unified multi-step quality and safety pipeline: (a) Answer–Question Consistency Check (LLM): An LLM evaluates whether each answer appropriately corresponds to its question; (b) Translation: All data are translated from Chinese into English to enhance accessibility and standardization; (c) Re-Consistency Check (LLM): The translated samples are re-evaluated for semantic consistency between question and answer; and (d) Security Screening: Both automated safety APIs and LLM-based evaluation again assess potential sensitive or risky content in both questions and answers.

All LLM-based filtering in the pipeline are performed using Qwen3-235B-A22B, selected for its high reasoning capability and multilingual understanding.

This multi-stage construction ensures that EComStage balances authenticity, safety, and reproducibility, making it suitable for both academic research and industrial deployment. Detailed prompts for dataset construction are provided in Appendix A.1.

### 3.2 Tasks

| Capability categories | Tasks | Orientation | Instances |

| Perception | Query Rewrite | Customer | 233 |

| Attitude Classification | Merchant | 424 |

| Query Match | Customer | 1927 |

| Intent Recognition | Customer | 1367 |

| Planning | Scenario Route | Merchant | 164 |

| Action | Solution Decision | Customer | 487 |

| RAG-QA | Both | 202 |

| Total instances | 4804 |

*Table 2: Statistics of EComStage.*

We categorize our benchmark tasks according to the three-stage reasoning framework introduced in former sections: Perception, Planning, and Action. Each task is designed to evaluate stage-specific abilities of agent-capable LLMs in realistic e-commerce scenarios.

#### 3.2.1 Perception

Tasks in this category measure the model’s ability to understand user intent and relevant context from both customer and merchant orientations.

-

Query Rewrite requires the agent to rewrite the user’s last utterance based on the conversation history, making it clearer and more complete while preserving the original meaning.

-

Attitude Classification asks the agent to identify whether the merchant’s current message contains a negative attitude based on chat history.

-

Query Match evaluates the agent’s ability to match a customer query to the most relevant entry in a provided question lists.

-

Intent Recognition asks the agent to identify the underlying intention of a customer message, including refund requests, complaints, product inquiries, and other issues.

#### 3.2.2 Planning

This stage assesses the model’s reasoning and action formulation capabilities.

-

Scenario Route requires the agent to determine the correct merchant-oriented workflow based on historical chat records given pre-defined scenarios, such as content rejection, promotion complaints, or refund requests.

This evaluates the agent’s ability to reason over multi-turn conversations and select the appropriate operational path.

#### 3.2.3 Action

Action tasks test the model’s decision-making and response execution.

-

Solution Decision requires the agent to generate the most appropriate response or solution for a customer query, referencing relevant response choices.

-

RAG-QA evaluates the agent’s ability to retrieve relevant reference knowledge and generate a context-aware, accurate answer, supporting both customer and merchant scenarios.

The statistics of EComStage, including task distribution, categories, and customer/merchant orientations after filtering, are summarized in Table 2. Note that for Query Match and Solution Decision, human annotators provide pre-defined question lists and response options tailored to each data sample. Specifically, our Planning set contains only 164 samples, but each sample spans multiple merchant scenarios, making it highly informative despite the smaller quantity. Overall, the benchmark includes five close-ended tasks and two open-ended generation tasks. The example prompts for each task are provided in Appendix A.2.

## 4 Experiments

### 4.1 Experimental Settings

#### 4.1.1 Evaluated Models

We evaluate a diverse set of agent-capable LLMs on EComStage. The selected models span multiple families, including general-purpose language models, instruction-tuned variants, MoE variants, and multimodal ones, reflecting the diversity of current research and industrial systems. We list the evaluated models as follows:

-

Closed-source APIs: GPT-4o OpenAI et al. (2024), Gemini 2.5-Pro Comanici et al. (2025), Claude 3.7 Anthropic (2025a), Claude Sonnet 4 Anthropic (2025b).

-

Open-source LLMs: LLaMA3.2 (3B, 11B), LLaMA3.3 (70B) Grattafiori et al. (2024), Qwen2.5-Instruct (3B, 7B, 14B, 32B, 72B) Qwen et al. (2025), Qwen3 (1.7B, 4B, 4B-Instruct, 8B, 14B, 30B-A3B, 30B-A3B-Instruct, 32B, 235B-A22B, 235B-A22B-Instruct) Yang et al. (2025), Phi-4-mini Abdin et al. (2024), GLM4 (9B, 32B) GLM et al. (2024), InternVL3 (8B, 14B) Zhu et al. (2025), MiniCPM-8B MiniCPM (2025), DeepSeek-V3 DeepSeek-AI et al. (2025b), DeepSeek-R1 DeepSeek-AI et al. (2025a), dots.llm1.inst Huo et al. (2025), gpt-oss-120B OpenAI (2025).

We choose this broad collection of models to assess the impact of model scale and cover multiple model families and architectures relevant for real-world e-commerce scenarios.

| Models | Perception | Planning | Action | Avg. |

| Query Rewrite | Attitude Classification | Query Match | Intent Recognition | Scenario Route | Solution Decision | RAG-QA |

| Closed-source APIs |

| GPT4o | 80.70 | 79.95 | 97.77 | 90.07 | 85.98 | 79.47 | 68.24 | 83.17 |

| Gemini2.5-Pro | 81.42 | 78.30 | 98.75 | 89.03 | 83.54 | 87.06 | 69.97 | 84.01 |

| Claude-3.7 | 81.99 | 77.59 | 98.39 | 88.00 | 82.32 | 75.56 | 69.38 | 81.89 |

| Claude Sonnet 4 | 81.81 | 83.49 | 98.44 | 90.42 | 88.41 | 74.74 | 72.13 | 84.21 |

| Open-source Models <7B |

| Qwen2.5-1.5B-Instruct | 74.72 | 64.62 | 93.93 | 43.96 | 61.59 | 78.85 | 64.85 | 68.93 |

| Qwen2.5-3B-Instruct | 76.19 | 70.99 | 95.12 | 68.98 | 53.66 | 77.62 | 64.72 | 72.47 |

| Qwen3-1.7B | 69.33 | 76.42 | 91.28 | 63.42 | 60.98 | 53.39 | 66.57 | 68.77 |

| Qwen3-4B | 73.26 | 78.07 | 97.09 | 80.83 | 71.95 | 78.23 | 67.47 | 78.13 |

| Qwen3-4B-Instruct | 79.97 | 82.78 | 96.94 | 84.20 | 78.66 | 84.80 | 68.45 | 82.26 |

| Llama3.2-3B | 76.99 | 69.58 | 89.93 | 44.48 | 77.44 | 15.61 | 63.73 | 62.54 |

| Phi-4-mini (4B) | 76.53 | 71.70 | 91.49 | 74.62 | 75.61 | 61.60 | 62.23 | 73.40 |

| Open-source Models <30B |

| Qwen2.5-7B-Instruct | 75.69 | 78.07 | 96.89 | 80.61 | 75.00 | 72.48 | 66.89 | 77.95 |

| Qwen2.5-14B-Instruct | 79.06 | 81.60 | 97.98 | 85.59 | 82.93 | 80.90 | 66.24 | 82.04 |

| Qwen3-8B | 75.40 | 79.25 | 97.30 | 83.61 | 78.05 | 85.01 | 68.48 | 81.01 |

| Qwen3-14B | 77.74 | 82.08 | 98.75 | 86.98 | 81.10 | 77.21 | 68.04 | 81.70 |

| Llama3.2-11B | 78.73 | 74.29 | 94.97 | 75.64 | 73.78 | 67.35 | 64.75 | 75.64 |

| GLM4-9B | 80.39 | 38.44 | 95.38 | 77.32 | 80.49 | 72.90 | 67.17 | 73.16 |

| InternVL3-8B | 77.73 | 81.60 | 97.87 | 84.64 | 78.05 | 68.38 | 66.09 | 79.19 |

| InternVL3-14B | 80.82 | 85.85 | 97.77 | 86.76 | 78.66 | 81.11 | 65.86 | 82.40 |

| MiniCPM-8B | 78.31 | 80.19 | 94.29 | 62.37 | 57.93 | 72.28 | 64.19 | 72.79 |

$\geq$| Open-source Models 30B |

| Qwen2.5-32B-Instruct | 79.74 | 79.95 | 98.50 | 88.15 | 85.37 | 76.59 | 65.88 | 82.03 |

| Qwen2.5-72B-Instruct | 80.28 | 83.02 | 98.70 | 89.83 | 89.02 | 87.06 | 67.06 | 85.00 |

| Qwen3-32B | 79.37 | 84.43 | 98.44 | 87.42 | 75.61 | 73.51 | 68.73 | 81.07 |

| Qwen3-30B-A3B | 70.07 | 82.08 | 97.98 | 85.30 | 89.02 | 81.72 | 68.21 | 82.05 |

| Qwen3-30B-A3B-Instruct | 78.53 | 78.30 | 97.92 | 86.32 | 82.32 | 94.25 | 67.75 | 83.63 |

| Qwen3-235B-A22B | 78.83 | 87.26 | 98.81 | 86.91 | 82.32 | 87.47 | 69.41 | 84.43 |

| Qwen3-235B-A22B-Instruct | 81.04 | 87.26 | 98.75 | 87.78 | 84.76 | 89.94 | 69.76 | 85.61 |

| DeepSeek-V3 | 82.49 | 88.68 | 98.34 | 88.59 | 85.37 | 77.00 | 69.17 | 84.23 |

| DeepSeek-R1 | 79.52 | 83.02 | 98.70 | 89.61 | 85.98 | 66.74 | 71.65 | 82.17 |

| GLM4-32B | 78.31 | 83.73 | 97.56 | 84.13 | 87.80 | 77.00 | 69.87 | 82.63 |

| LLama3.3-70B | 80.26 | 79.01 | 98.08 | 86.83 | 84.76 | 70.23 | 66.63 | 80.83 |

| dots.llm1.inst | 79.61 | 87.74 | 98.08 | 71.40 | 70.73 | 74.13 | 64.92 | 78.09 |

| GPT-OSS-120B | 78.38 | 73.58 | 98.39 | 88.59 | 77.44 | 78.23 | 70.04 | 80.66 |

*Table 3: Main evaluation results. The best results are highlighted in bold, and the second-best results are underlined.*

#### 4.1.2 Evaluation Metrics

For close-ended tasks, we report accuracy, while for open-ended generation tasks, we use cosine similarity to measure alignment with reference answers provided by human annotators. Specifically, we use Qwen3-Embedding-8B Zhang et al. (2025) to convert the generated texts and reference answers into embeddings of 4096 dimensions and then calculate the cosine similarity between them.

#### 4.1.3 Implementation Details

For all experiments, we set the batch size to 32 and limit the model input length to 4,096 tokens. During generation, we use a low temperature of 0.1 and a top-p sampling threshold of 0.001 to encourage deterministic outputs. To reduce repetitive responses, we apply a repetition penalty of 1.05. The maximum number of tokens generated per query is capped at 512. These settings are applied consistently across all evaluated models to provide a fair comparison and simulate practical deployment scenarios. All experiments are conducted on 8 NVIDIA H800 GPUs with a single run.

### 4.2 Main Experimental Results

We perform systematic evaluation of LLM-based agents’ abilities at each reasoning stage, from Perception and Planning to Action. The results across the diverse set of evaluated models are summarized in Table 3.

#### 4.2.1 Closed-source APIs

We first evaluate closed-source APIs, as shown at the top of Table 3. These models demonstrate strong overall performance across all tasks. Among them, Claude Sonnet 4 achieves the best average score (84.21), followed closely by Gemini 2.5-Pro (84.01). This finding is consistent with prior observations Zhou et al. (2025), where both models exhibit superior performance in e-commerce scenarios compared to GPT-4o. While all models perform consistently well on classification-style tasks such as Query Match and Intent Recognition, their performance varies more on merchant-oriented tasks like Attitude Classification and Scenario Route, highlighting the limitations of current evaluation practices. Claude Sonnet 4 shows stronger capability in handling merchant-oriented tasks, likely due to its optimization for complex reasoning and tool use Anthropic (2025b), making it a strong baseline for agent-oriented evaluation. In contrast, Gemini 2.5-Pro achieves the highest accuracy in Solution Decision, benefiting from its enhanced long-term planning ability Comanici et al. (2025).

#### 4.2.2 Open-source Models less than 7B

We next evaluate open-source models with fewer than 7B parameters. Among them, Qwen3-4B-Instruct achieves the highest overall score (82.26), showing strong generalization across both customer- and merchant-oriented tasks. It performs particularly well on Attitude Classification and Scenario Route, indicating improved instruction-following and reasoning capabilities. This strong performance aligns with its results on $\tau$-Bench Yao et al. (2024), suggesting that the model’s recent instruction tuning and alignment optimization substantially improve its robustness and domain adaptability in complex reasoning scenarios Yang et al. (2025). In contrast, smaller models such as Qwen2.5-1.5B-Instruct and Llama3.2-3B struggle on complex reasoning and decision-oriented tasks like Intent Recognition and Solution Decision, partly because they belong to relatively earlier generations of LLMs Zhao et al. (2025), developed prior to recent advances in multi-turn instruction tuning and reasoning optimization. Despite the smaller size, 4B models also demonstrate balanced accuracy across tasks, showing potential for lightweight deployment.

#### 4.2.3 Open-source Models less than 30B

Moving to mid-sized models (below 30B parameters), we observe that their overall performance is more competetive than smaller ones. InternVL3-14B achieves the highest overall score (82.40), followed closely by Qwen2.5-14B-Instruct (82.04). Models from the Qwen and InternVL families exhibit balanced capabilities across both customer- and merchant-oriented scenarios, reflecting their strong instruction alignment and task adaptability. In particular, InternVL3-14B benefits from extended training data that emphasize tool use, long-context reasoning and creative writing Zhu et al. (2025). It also employs Mixed Preference Optimization Wang et al. (2025c), which leverages additional supervision to better align model responses with ground-truth distributions. This dual-supervision strategy significantly enhances its reasoning and decision-making abilities across diverse e-commerce tasks. In contrast, Llama3.2-11B and MiniCPM-8B show noticeable performance drops in reasoning-intensive or decision-making tasks such as Scenario Route and Solution Decision, suggesting that these models may lack domain-specific optimization for structured reasoning in e-commerce contexts MiniCPM (2025). Notably, Qwen3-8B and Qwen3-14B fall behind Qwen3-4B-Instruct. This underscores the impact of post-training quality: Qwen3-4B-Instruct benefits from more recent instruction tuning and alignment optimization Yang et al. (2025), which significantly enhance its task-following and dialogue understanding capabilities in e-commerce scenarios.

#### 4.2.4 Open-source Models larger than 30B

Finally, we evaluate the largest models with over 30B parameters. In this group, Qwen3-235B-A22B-Instruct achieves the best overall performance (85.61), likely benefiting from both its large model capacity and the high-quality, diverse-coverage curated fine-tuning samples generated by DeepSeek-R1. Across these models, we also observe that more recent instruction-tuned variants such as Qwen3-30B-A3B-Instruct and Qwen3-235B-A22B-Instruct consistently outperform their earlier counterparts, indicating the importance of recent instruction tuning that enhance reasoning and task-following abilities Qwen et al. (2025). While most models excel in classification-style tasks like Intent Recognition, their performance in Solution Decision tasks remains unstable, possibly due to the task’s reliance on reasoning over multiple context turns. Deepseek V3 performs better than Deepseek R1, indicating V3’s stronger generalization ability to e-commerce scenarios DeepSeek-AI et al. (2025b). Regarding merchant-oriented tasks, dots.llm1.inst demonstrates strong performance in Attitude Classification, likely due to its training on fine-grained dialogue understanding and sentiment detection Huo et al. (2025).

These results can reveal several issues that are often overlooked in standard evaluations. The deficiencies found in different models highlight that even extremely large models can have stage-specific weaknesses that are not apparent when only measuring overall task success. Our stage-wise benchmark is crucial for uncovering these gaps, providing fine-grained insights into where models excel or fail, and guiding targeted improvements for real-world e-commerce applications.

### 4.3 Stage-wise and Side-wise Comparison

To better analyze the effect of stage-wise and orientation-wise evaluation, we choose and compare several models that exhibit strong overall performance including GPT4o, Claude-sonnet-4, Qwen3-235B-A22B-Instruct, Qwen2.5-72B-Instruct and Deepseek-V3. We illustrate the stage-wise and orientation-wise performance of them in Figure 3.

Across the different stages, all models maintain high performance on Perception, while their performance in other stages varies more noticeably. For example, Qwen2.5-72B-Instruct and Claude Sonnet 4 excel in Planning, benefiting from the enhanced multi-step reasoning and the ability to formulate coherent plans Qwen et al. (2025); Anthropic (2025b). Whereas Qwen3-235B-A22B-Instruct stands out in Action, likely owing to its markedly better alignment with user preferences in open-ended tasks Yang et al. (2025).

As for orientation-wise performance, Qwen3-235B-A22B-Instruct and Qwen2.5-72B-Instruct show strong results on customer-oriented tasks, while Deepseek-V3 exhibits the best performance in merchant-oriented tasks, likely because it has been specifically optimized for task-specific alignment, including structured reasoning and scenario planning. Notably, GPT-4o, previously one of the strongest LLMs, shows relative weakness on merchant-oriented tasks.

These observations highlight that overall average scores alone are insufficient to fully understand model behavior, as they can obscure critical weaknesses at specific stages or on certain sides. Our experiments reveal that no single model consistently excels across all tasks, stages, or orientations. Our benchmark therefore provides a more granular evaluation framework that captures the multi-dimensional capabilities of LLM-based agents in e-commerce scenarios. By breaking down model performance into Perception, Planning, Action, customer- and merchant-oriented dimensions, it allows researchers and practitioners to identify strengths and weaknesses more precisely and better guide model development and deployment, offering valuable insights for practical applications in both academia and industry.

*Figure 3: Stage-wise and orientation-wise comparison of LLM-based agents.*

## 5 Conclusion

In this work, we introduce a comprehensive stage-wise benchmark for evaluating agent-capable LLMs in e-commerce scenarios for both customer and merchant orientations with real-world e-commerce data. Specifically, we divide the reasoning process into Perception, Planning and Action. Our experiments reveal substantial differences in performance across model families, sizes, and instruction-tuning strategies. While most models perform well on Perception and classification-style tasks, significant variation remains in Planning, Action, and merchant-oriented tasks. We observe that no single model can excel across all tasks, underscoring the importance of a granular evaluation framework. Our benchmark enables researchers and practitioners to uncover model strengths and weaknesses that are hidden in success rates, providing actionable insights for model development and deployment.

## Limitations

Although EComStage decomposes e-commerce reasoning into Perception, Planning, and Action stages, these stages are evaluated through separate, stage-specific tasks. As a result, the benchmark does not capture error propagation across stages, which may occur in real-world deployments. Our benchmark focuses on representative but finite e-commerce scenarios, such as product inquiry, recommendation, complaint handling, and refund decisions. While these tasks cover common real-world interactions, they do not fully encompass all e-commerce domains, leaving room for future expansion.

## Ethics Statement

EComStage is constructed with careful and strict filtering to avoid introducing potential ethical concerns. Models for evaluation and automatic metrics are used in accordance with their respective licenses. We use OpenAI’s GPT-5 model for minor language editing and grammar polishing. The model is used to improve clarity and conciseness of writing. All technical content, data analysis, and conclusions are produced and validated by the authors.

## References

- Abdin et al. (2024) M. Abdin, J. Aneja, H. Behl, S. Bubeck, and R. E. et al Phi-4 technical report. External Links: 2412.08905, Link Cited by: 2nd item.

- Anthropic (2025a) Anthropic Claude 3.7 sonnet system card. External Links: Link Cited by: 1st item.

- Anthropic (2025b) Anthropic System card: claude opus 4 & claude sonnet 4. External Links: Link Cited by: 1st item, §4.2.1, §4.3.

- Budzianowski et al. (2018) P. Budzianowski, T. Wen, B. Tseng, I. Casanueva, S. Ultes, O. Ramadan, and M. Gašić MultiWOZ - a large-scale multi-domain Wizard-of-Oz dataset for task-oriented dialogue modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii (Eds.), Brussels, Belgium, pp. 5016–5026. External Links: Link, Document Cited by: §2.

- Cheng et al. (2024) Z. Cheng, W. Zhang, C. (. Chou, Y. Jau, A. Pathak, P. Gao, and U. Batur E-commerce product categorization with llm-based dual-expert classification paradigm. Cited by: §1.

- Comanici et al. (2025) G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, and N. S. et al Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. External Links: 2507.06261, Link Cited by: §1, 1st item, §4.2.1.

- DeepSeek-AI et al. (2025a) DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, and R. Z. et al DeepSeek-r1: incentivizing reasoning capability in llms via reinforcement learning. External Links: 2501.12948, Link Cited by: 2nd item.

- DeepSeek-AI et al. (2025b) DeepSeek-AI, A. Liu, B. Feng, B. Xue, B. Wang, and B. W. et al DeepSeek-v3 technical report. External Links: 2412.19437, Link Cited by: 2nd item, §4.2.4.

- Dong et al. (2025) Y. Dong, X. Jiang, J. Qian, T. Wang, K. Zhang, Z. Jin, and G. Li A survey on code generation with llm-based agents. External Links: 2508.00083, Link Cited by: §1.

- Feng et al. (2024) Z. Feng, Z. Meng, and Z. Liu EC-guide: a comprehensive e-commerce guide for instruction tuning and quantization. External Links: 2408.02970, Link Cited by: §1.

- GLM et al. (2024) T. GLM, A. Zeng, B. Xu, B. Wang, C. Zhang, and D. Y. et al ChatGLM: a family of large language models from glm-130b to glm-4 all tools. External Links: 2406.12793 Cited by: 2nd item.

- Grattafiori et al. (2024) A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, and A. K. et al. The llama 3 herd of models. External Links: 2407.21783, Link Cited by: §1, 2nd item.

- Guo et al. (2024) T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang Large language model based multi-agents: a survey of progress and challenges. External Links: 2402.01680, Link Cited by: §2.

- Hu et al. (2025) S. Hu, T. Huang, G. Liu, R. R. Kompella, F. Ilhan, S. F. Tekin, Y. Xu, Z. Yahn, and L. Liu A survey on large language model-based game agents. External Links: 2404.02039, Link Cited by: §1.

- Huang et al. (2025) Y. Huang, Y. Liu, R. Zhao, X. Zhong, X. Yue, and L. Jiang MemOrb: a plug-and-play verbal-reinforcement memory layer for e-commerce customer service. External Links: 2509.18713, Link Cited by: §1.

- Huo et al. (2025) B. Huo, B. Tu, C. Qin, D. Zheng, and D. Z. et al Dots.llm1 technical report. External Links: 2506.05767, Link Cited by: 2nd item, §4.2.4.

- Isa et al. (2024) N. A. N. M. Isa, S. N. A. Jawaddi, and A. Ismail Experimental evaluation of machine learning models for goal-oriented customer service chatbot with pipeline architecture. External Links: 2409.18568, Link Cited by: §1.

- Jia et al. (2022) M. Jia, R. Liu, P. Wang, Y. Song, Z. Xi, H. Li, X. Shen, M. Chen, J. Pang, and X. He E-ConvRec: a large-scale conversational recommendation dataset for E-commerce customer service. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, N. Calzolari, F. Béchet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, J. Odijk, and S. Piperidis (Eds.), Marseille, France, pp. 5787–5796. External Links: Link Cited by: §2.

- Liu et al. (2024) X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang, et al. AgentBench: evaluating llms as agents. Cited by: §1.

- Liu et al. (2023) Y. Liu, W. Zhang, B. Dong, Y. Fan, H. Wang, F. Feng, Y. Chen, Z. Zhuang, H. Cui, Y. Li, and W. Che U-need: a fine-grained dataset for user needs-centric e-commerce conversational recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, New York, NY, USA, pp. 2723–2732. External Links: ISBN 9781450394086, Link, Document Cited by: §2.

- Luo et al. (2025) J. Luo, W. Zhang, Y. Yuan, Y. Zhao, J. Yang, Y. Gu, B. Wu, B. Chen, Z. Qiao, Q. Long, R. Tu, X. Luo, W. Ju, Z. Xiao, Y. Wang, M. Xiao, C. Liu, J. Yuan, S. Zhang, Y. Jin, F. Zhang, X. Wu, H. Zhao, D. Tao, P. S. Yu, and M. Zhang Large language model agent: a survey on methodology, applications and challenges. External Links: 2503.21460, Link Cited by: §1, §2.

- MiniCPM (2025) T. MiniCPM Minicpm4: ultra-efficient llms on end devices. arXiv preprint arXiv:2506.07900. Cited by: 2nd item, §4.2.3.

- OpenAI et al. (2024) OpenAI, :, A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, and A. R. et al GPT-4o system card. External Links: 2410.21276, Link Cited by: §1, 1st item.

- OpenAI (2025) OpenAI Gpt-oss-120b & gpt-oss-20b model card. External Links: 2508.10925, Link Cited by: 2nd item.

- Qwen et al. (2025) Qwen, A. Yang, B. Yang, B. Zhang, B. Hui, and B. Z. et al Qwen2.5 technical report. External Links: 2412.15115, Link Cited by: 2nd item, §4.2.4, §4.3.

- Wang et al. (2025a) H. Wang, X. Peng, H. Cheng, Y. Huang, M. Gong, C. Yang, Y. Liu, and J. Lin ECom-bench: can LLM agent resolve real-world E-commerce customer support issues?. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, S. Potdar, L. Rojas-Barahona, and S. Montella (Eds.), Suzhou (China), pp. 276–284. External Links: Link, Document, ISBN 979-8-89176-333-3 Cited by: §1, §2, Table 1.

- Wang et al. (2025b) W. Wang, L. Cui, X. Liu, S. Nag, W. Xu, C. Luo, S. M. Sarwar, Y. Li, H. Gu, H. Liu, C. Yu, J. Bai, Y. Gao, H. Zhang, Q. He, S. Ji, and Y. Song EcomScriptBench: a multi-task benchmark for E-commerce script planning via step-wise intention-driven product association. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria, pp. 1–22. External Links: Link, Document, ISBN 979-8-89176-251-0 Cited by: §2.

- Wang et al. (2025c) W. Wang, Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, J. Zhu, X. Zhu, L. Lu, Y. Qiao, and J. Dai Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. External Links: 2411.10442, Link Cited by: §4.2.3.

- Xi et al. (2023) Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou, R. Zheng, X. Fan, X. Wang, L. Xiong, Y. Zhou, W. Wang, C. Jiang, Y. Zou, X. Liu, Z. Yin, S. Dou, R. Weng, W. Cheng, Q. Zhang, W. Qin, Y. Zheng, X. Qiu, X. Huang, and T. Gui The rise and potential of large language model based agents: a survey. External Links: 2309.07864, Link Cited by: §2.

- Xu et al. (2025) A. Xu, Z. Yang, J. Li, G. Yuan, L. Chen, L. Yan, J. Zhou, Z. Qin, H. Chang, H. Alinejad-Rokny, B. Zheng, and M. Yang EVADE: multimodal benchmark for evasive content detection in e-commerce applications. External Links: 2505.17654, Link Cited by: §2.

- Yang et al. (2025) A. Yang, A. Li, B. Yang, B. Zhang, and B. H. et al. Qwen3 technical report. External Links: 2505.09388, Link Cited by: §1, 2nd item, §4.2.2, §4.2.3, §4.3.

- Yang et al. (2023) H. Yang, S. Yue, and Y. He Auto-gpt for online decision making: benchmarks and additional opinions. External Links: 2306.02224, Link Cited by: §1.

- Yao et al. (2022) S. Yao, H. Chen, J. Yang, and K. Narasimhan WebShop: towards scalable real-world web interaction with grounded language agents. In NeurIPS, Cited by: §1.

- Yao et al. (2024) S. Yao, N. Shinn, P. Razavi, and K. Narasimhan $\tau$-Bench: a benchmark for tool-agent-user interaction in real-world domains. External Links: 2406.12045, Link Cited by: §1, §1, §2, §4.2.2.

- Zeng et al. (2025) J. Zeng, H. Liu, Z. Dai, X. Tang, C. Luo, S. Varshney, Z. Li, and Q. He Cite before you speak: enhancing context-response grounding in e-commerce conversational llm-agents. External Links: 2503.04830, Link Cited by: §1, §1, §2.

- Zhang et al. (2025) Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, and J. Zhou Qwen3 embedding: advancing text embedding and reranking through foundation models. External Links: 2506.05176, Link Cited by: §4.1.2.

- Zhao et al. (2025) W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, Y. Du, C. Yang, Y. Chen, Z. Chen, J. Jiang, R. Ren, Y. Li, X. Tang, Z. Liu, P. Liu, J. Nie, and J. Wen A survey of large language models. External Links: 2303.18223, Link Cited by: §4.2.2.

- Zhou et al. (2025) C. Zhou, X. Shi, H. Qiu, X. Zheng, H. Leng, Y. Jiang, S. Liu, T. Gao, and R. Ji Mix-ecom: towards mixed-type e-commerce dialogues with complex domain rules. External Links: 2509.23836, Link Cited by: §1, §2, Table 1, §4.2.1.

- Zhu et al. (2025) J. Zhu, W. Wang, Z. Chen, Z. Liu, and S. Y. et al InternVL3: exploring advanced training and test-time recipes for open-source multimodal models. External Links: 2504.10479, Link Cited by: 2nd item, §4.2.3.

*Figure 4: Guidelines for human annotators.*

## Appendix A Appendix

### A.1 Prompts for Dataset Construction

*Figure 5: Task-specific filtering prompt for removing sentitive information.*

*Figure 6: Global filtering prompt to ensure that no sensitive or risky content is involved. We only preserve data labeled as non-sensitive.*

*Figure 7: Prompt for consistency judgment. We only preserve data labeled as consistent.*

*Figure 8: Translation prompt for our data. We translate them from Chinese to English.*

We present detailed prompts used in the construction of EComStagein this section. Task-specific filtering prompt is presented in Figure 5, while global filtering prompt is provided in Figure 6. Figure 7 presents the prompt we used for consistency judgment and Figure 8 presents the prompt for translation.

### A.2 Examples for Tasks

*Figure 9: Example for Query Rewrite task (Customer-oriented).*

*Figure 10: Example for Attitude Classification task (Merchant-oriented).*

*Figure 11: Example for Query Match task (Customer-oriented).*

*Figure 12: Example for Intent Recognition task (Customer-oriented).*

*Figure 13: Example for Solution Decision task (Customer-oriented).*

*Figure 14: Example for Scenario Route task (Merchant-oriented).*

*Figure 15: Example for RAG-QA task (Customer- and Merchant-oriented).*

We present examples for our seven tasks in Figure 9, 10, 11, 12, 13, 14, and 15, separately.

### A.3 Human Annotators

All participants involved in the annotation and evaluation process are employees of our company with relevant experience in e-commerce operations. They are assigned with the tasks as part of their regular work responsibilities, and no additional payment is provided beyond their standard compensation. Given their professional expertise and familiarity with e-commerce scenarios, they are well-qualified to perform the tasks reliably.

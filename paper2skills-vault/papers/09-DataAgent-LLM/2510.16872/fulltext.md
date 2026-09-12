<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2510.16872
     paper_id : 2510.16872
     source   : https://arxiv.org/html/2510.16872v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# DeepAnalyze: Agentic Large Language Models for Autonomous Data Science

Shaolei Zhang Affiliation: Renmin University of China Correspondence to: zhangshaolei98@ruc.edu.cn    Ju Fan Affiliation: Renmin University of China    Meihao Fan Affiliation: Renmin University of China    Guoliang Li Affiliation: Tsinghua University    Xiaoyong Du Affiliation: Renmin University of China Correspondence to: fanj@ruc.edu.cn

###### Abstract

Autonomous data science, from raw data sources to analyst-grade deep research reports, has been a long-standing challenge, and is now becoming feasible with the emergence of powerful large language models (LLMs). Recent workflow-based data agents have shown promising results on specific data tasks but remain fundamentally limited in achieving fully autonomous data science due to their reliance on predefined workflows. In this paper, we introduce DeepAnalyze-8B, the first agentic LLM designed for autonomous data science, capable of automatically completing the end-to-end pipeline from data sources to analyst-grade deep research reports. To tackle high-complexity data science tasks, we propose a curriculum-based agentic training paradigm that emulates the learning trajectory of human data scientists, enabling LLMs to progressively acquire and integrate multiple capabilities in real-world environments. We also introduce a data-grounded trajectory synthesis framework that constructs high-quality training data. Through agentic training, DeepAnalyze learns to perform a broad spectrum of data tasks, ranging from data question answering and specialized analytical tasks to open-ended data research. Experiments demonstrate that, with only 8B parameters, DeepAnalyze outperforms previous workflow-based agents built on most advanced proprietary LLMs. The model, code, and training data of DeepAnalyze are open-sourced, paving the way toward autonomous data science.

###### Keywords:

Machine Learning, ICML

 ruc-datalab/DeepAnalyze    DeepAnalyze-8B    DataScience-Instruct-500K    ruc-deepanalyze.github.io

## 1 Introduction

Autonomous data science (De Bie et al., 2022; Sun et al., 2025b; Wang et al., 2025), a long-standing central goal of the data science community, aims to automate the entire data science pipeline for extracting insights from structured data. This pipeline is inherently complex, consisting of a series of interdependent data-centric tasks spanning data preparation, analysis, modeling, visualization, and report generation. The emergence of open-ended data research further elevates the level of complexity, going far beyond traditional question answering or task-specific analytics. Fortunately, recent advances in large language models (LLMs) have demonstrated impressive problem-solving abilities (OpenAI, 2023; OpenAI, 2024; DeepSeek-AI, 2025), reshaping paradigms in domains such as search (Zheng et al., 2025; Jin et al., 2025) and mathematics (Zhang et al., 2024; Ren et al., 2025). However, despite their success on unstructured data (e.g., textual queries or contexts), LLMs still struggle to orchestrate complex, multi-stage data science pipelines and handle diverse structured data, making it difficult to achieve a general solution that works across all data science tasks.

*Figure 1: DeepAnalyze-8B is the first end-to-end agentic LLM that achieves autonomous data science, supporting entire data science pipeline and open-ended data research.*

Addressing these challenges requires endowing LLMs with two higher-level capabilities: autonomous orchestration and adaptive optimization. First, autonomous orchestration enables LLMs to comprehend user intents and systematically coordinate a sequence of interdependent actions to accomplish complex tasks (Sapkota et al., 2025a). Second, adaptive optimization allows LLMs to interact with real-world data environments and iteratively refine their actions based on feedback (Hong et al., 2024). As shown in Figure 4, equipped with these two capabilities, an intelligent system can robustly handle a broad spectrum of data tasks, ranging from conventional question answering and task-specific analytics to fully autonomous, open-ended data research.

*Figure 2: Examples of DeepAnalyze-8B. Given the instructions and data sources in the environment, DeepAnalyze can autonomously orchestrate and optimize actions to complete a data science pipeline (left) and open-ended data research (right). DeepAnalyze first performs planning, then interacts with the data in the environment, and subsequently optimizes its actions based on feedback, ultimately accomplishing the data-centric tasks. Many intermediate actions are omitted to save space.*

Existing approaches to applying LLMs for autonomous data science can be broadly categorized into domain-specific LLMs and workflow-based agents. Early efforts focus on developing domain-specific LLMs, such as code-oriented models (Nascimento et al., 2024; Wen et al., 2024) and structured data-oriented models (Li et al., 2023b; Jiang et al., 2023; Xu et al., 2025), to handle individual tasks like question answering or specific analytical operations. However, these models lack the capabilities for autonomous orchestration and adaptive optimization (Yang et al., 2021; Li et al., 2023a), limiting their ability to execute the entire data science pipeline. More recently, a line of work has explored workflow-based data science agents (Hollmann et al., 2023; Guo et al., 2024; Sun et al., 2025a; Hong et al., 2025), which rely on predefined procedural workflows to prompt closed-source LLMs (e.g., GPT-4 (OpenAI, 2023)) to complete complex tasks. Although these systems demonstrate stronger task coordination, they depend heavily on manually designed heuristics and domain-specific rules, falling short of achieving autonomous and adaptive behavior.

In essence, both domain-specific models and workflow-based agents remain limited, as they are not trained in interactive, real-world environments. Consequently, they struggle to perform complex tasks through autonomous orchestration and adaptive optimization. Notably, recent advances in agentic training, a new training paradigm successfully applied in the search domain (Zheng et al., 2025; Jin et al., 2025), have demonstrated that reinforcement learning in real-world environments is crucial for enabling LLMs to develop autonomous problem-solving capabilities.

In this paper, we aim to advance LLM-based data science methods from workflow-based agents to a trainable agentic model that learns to autonomously perform data science tasks in real-world environments. However, applying agentic training to this domain presents two key challenges: reward sparsity and trajectory scarcity. On the one hand, the inherent complexity of data science tasks makes it difficult for foundation LLMs to complete tasks successfully during the early stages of training. This leads to severe reward sparsity, i.e., a lack of positive reinforcement signals, which can hinder or even collapse the entire agentic training process. On the other hand, the scarcity of long-chain problem-solving trajectories in data science provides insufficient guidance for LLMs to explore the solution space effectively, resulting in inefficient, blind trial-and-error exploration without meaningful intermediate supervision.

To address these challenges, we introduce DeepAnalyze, an agentic LLM designed for autonomous data science. As illustrated in Figure 2, with only 8B parameters, DeepAnalyze can automate the entire data science pipeline, ranging from specific data tasks to open-ended data research, providing a unified and general solution for data-centric applications. Specifically, to mitigate reward sparsity, DeepAnalyze adopts a curriculum-based agentic training paradigm inspired by the learning trajectory of human data scientists. This progressive easy-to-difficult schedule enables the model to gradually evolve from mastering individual skills to developing comprehensive, adaptive problem-solving abilities in real-world environments. To address trajectory scarcity, we propose a data-grounded trajectory synthesis framework that automatically constructs high-quality reasoning and interaction trajectories, offering effective exploration guidance within the large solution space. Through this combination of curriculum-based training and trajectory synthesis, DeepAnalyze learns to autonomously orchestrate actions and adaptively optimize its strategies, enabling it to tackle complex and diverse data science tasks effectively.

In summary, our key contributions are three-fold.

-

Agentic Model: To the best of our knowledge, DeepAnalyze is the first agentic LLM tailored for autonomous data science, endowed with two indispensable capabilities, autonomous orchestration and adaptive optimization. DeepAnalyze serves as a foundation model that can be directly applied or further customized through prompting or supervised fine-tuning for specific scenarios.

-

Agentic Training: We propose a curriculum-based agentic training paradigm with data-grounded trajectory synthesis to address reward sparsity and trajectory scarcity, enabling effective learning for high-complexity tasks that require multiple abilities.

-

Strong Performance: Experimental results on 12 benchmarks show that, with only 8B parameters, DeepAnalyze-8B surpasses most advanced proprietary LLMs. More importantly, it is the first agentic model capable of performing open-ended data research and generating analyst-grade reports.

## 2 Related Work

Autonomous Data Science. Autonomous data science has long been pursued as an important goal of intelligent systems. Existing LLM-based data science methods can be categorized into: domain-specific LLMs and workflow-based agents. To handle individual tasks in data science, early methods focused on fine-tuning LLMs into domain-specific models, including LLMs for data science code generation (Nascimento et al., 2024; Wen et al., 2024; Nejjar et al., 2024; Pan et al., 2025), tabular LLMs (Li et al., 2023b; Fang et al., 2024; Zhang et al., 2025c; Xu et al., 2025; Ouyang et al., 2025; Lei et al., 2025), and database-oriented LLMs (Xue et al., 2024; Liu et al., 2024; Mohammadjafari et al., 2025). Recently, an increasing number of data agents have demonstrated promising performance in data science by leveraging workflows to gradually prompt LLMs for complex tasks (Hollmann et al., 2023; Guo et al., 2024; Yang et al., 2024; Sun et al., 2025a; Hong et al., 2025). Most existing agents are built upon Chain-of-Thought frameworks, including ReAct (Yao et al., 2023), AutoGen (Wu et al., 2024), and self-reflection (Pan et al., 2023), which decompose complex tasks into multiple subtasks and solve them sequentially. Regardless of workflow design, existing agents primarily rely on carefully crafted prompting to guide closed-source LLMs in performing data science tasks.

*Figure 3: Architecture of DeepAnalyze.*

Despite these advances, domain-specific LLMs (focused on individual tasks) and workflow-based agents (dependent on manually designed workflows) remain incapable of fully autonomous data science. Therefore, the proposed DeepAnalyze does not rely on prompting frameworks or predefined workflows, instead, it internalizes data science capabilities through agentic training within real-world environments.

Agentic Training for LLM. Agentic training aims to enhance LLMs as agentic models through reinforcement learning and thereby enable LLMs to perform multi-step reasoning and interactions in real-world environments (Plaat et al., 2025), which has already achieved practical success in coding (Sapkota et al., 2025b) and searching (Zheng et al., 2025; Li et al., 2025; Jin et al., 2025). Typically, these methods use prompts to control the interaction format of LLMs and complete RL with the accuracy of the final answer as the reward. Based on this, lightweight cold-start is proposed to help LLMs learn the interaction format (DeepSeek-AI, 2025), improving the initial state for RL training. Existing training methods mainly focus on reasoning ability, while data science requires a broader range of abilities, such as reasoning, structured data understanding, and code generation. This complexity makes that initial LLMs (even after cold-start format learning) are generally incapable of completing complex data science tasks, leading to challanges of reward sparsity and trajectory scarcity. To this end, we propose a curriculum-based agentic training that enables LLMs to gradually acquire complex data science skills through a progression from single to multiple abilities, while employing data-grounded trajectory synthesis to generate high-quality reasoning and interaction trajectories for training.

## 3 DeepAnalyze

In this paper, we introduce DeepAnalyze, an agentic large language model for autonomous data science. To endow the LLM with the capability for autonomous orchestration and adaptive optimization in real-world environments, we propose a curriculum-based agentic training and data-grounded trajectory synthesis framework tailored for complex tasks with multiple abilities. Specifically, inspired by the behavior of human data scientists, we first define a set of actions that enable DeepAnalyze to directly interact with the data in its environment. Building on this architecture, we automatically synthesize high-quality data science trajectories and introduce a curriculum-based agentic training paradigm that guides DeepAnalyze through a progression from a beginner to data scientist, thereby empowering DeepAnalyze to tackle a wide spectrum of data science tasks. The architecture, curriculum-based agentic training, and data-grounded trajectory synthesis are introduced as follows.

### 3.1 Architecture

Unlike foundation LLMs that focus on understanding and generating natural language, LLMs for data science meet the additional challenge of understanding and interaction with structured data, which is typically stored in external files. Therefore, DeepAnalyze extends natural language interaction by introducing data-oriented interaction pattern, thereby enabling LLMs to autonomously interact with real-world environments.

Inputs Format. Previous structured data–specific LLMs (Li et al., 2023b; Fang et al., 2024; Zhang et al., 2025c; Xu et al., 2025; Lei et al., 2025) often converted tables stored in databases, CSV, or XLSX files into unstructured Markdown text and fed them into the LLM’s context to enable structured data understanding. However, due to context length limitations, these methods can only handle small-scale data (e.g., very small tables). When human data scientists work with large-scale data, they do not passively read and memorize every record. Instead, they actively explore each data source as needed and then plan the following steps accordingly. To this end, DeepAnalyze integrates both modes: it passively accepts structured data expressed as text in the input, while also actively inspecting external data sources according to user inputs, where the filenames of the external data sources are specified in inputs, as shown in Figure 2.

Interaction Pattern. Given an instruction and the data sources in the environment, data scientists typically analyze, interact with the data in the environment, understand structured data, and iterate until the instruction is completed. To emulate this process, DeepAnalyze introduces five actions to automatically accomplish the data science task, including:

-

$\cdots$: Analyze textually, including planning, reasoning, reflection, self-verification…

-

$\cdots$: Understand the content of data source, such as databases, tables, and documents.

-

$\cdots$: Generate code to interact with the data in the environment, using Python suited for data science.

-

$\cdots$: Execute code and collect the feedback from the environment.

-

$\cdots$: Produce the final output.

In practice, we extend the vocabulary of the foundation LLM to support the generation of these special tokens. During the inference, DeepAnalyze automatically switches between different actions by generating these special tokens, as shown in the right side of Figure 3. In particular, once a $\cdots$ is generated, DeepAnalyze executes the code in the environment and places the feedback in $\cdots$, and then generates the next action. The detailed inference process of DeepAnalyze is shown in Algorithm 1. With this architecture, all actions (i.e., special tokens) are autonomously generated by the model without any human-defined workflows or rules, which allows DeepAnalyze to fully autonomously orchestrate and optimize each action, laying the foundation for autonomous data science.

*Algorithm 1 Inference of DeepAnalyze*

1:  Input: Instruction $Q$, Environment $Env$, DeepAnalyze model $\mathcal{M}$

2:  Output: Response $A$ (with interaction process)

3:  Initialization: $A=\emptyset$

4:  while $\cdots$ not in $A$ do

5:    $y\leftarrow\mathcal{M}(Q,A)$       // generate next action based on the instruction $Q$ and current response $A$

6:    $A\leftarrow A+y$

7:    if $\cdots$ in $y$ then

8:     $code$ $\leftarrow$ extract_code$(y)$

9:     $feedback$ $\leftarrow Env$.execute(code)    // interaction with the data in the environment

10:     $A\leftarrow A+$ $+feedback+$

11:    end if

12:  end while

13:  Return $A$

### 3.2 Curriculum-based Agentic Training

Under the above architecture, DeepAnalyze need to learn how to interact with the environment to accomplish various data science tasks. Unlike individual coding or searching task, data science tasks demand a broader and more complex set of abilities, ranging from reasoning, structured data understanding, and code generation to the composite abilities needed for entire data science pipeline and open-ended research. The complexity of these capabilities results in the limited proficiency of foundation LLMs in data science domains (Zhang et al., 2025b), leading to severe reward sparsity on complex tasks and rendering existing agentic training (such as RL-Zero or RL with cold-start training (DeepSeek-AI, 2025)) ineffective due to the lack of positive feedback. To address this challenge, we propose curriculum-based agentic training, which emulates the learning path of human data scientists by gradually transitioning from mastering single abilities to integrating multiple abilities. This training framework consists of two stages, where stage 1 employs single-ability fine-tuning to strengthen the foundation LLM’s single ability, and stage 2 uses multi-ability agentic training to enable the LLM to apply multiple abilities in real-world environments to accomplish complex data science tasks.

*Figure 4: Schematic diagram of agentic RL.*

*(a) Reasoning Trajectory Synthesis*

*(b) Interaction Trajectory Synthesis*

*Figure 5: The proposed data-grounded trajectory synthesis for the development of DeepAnalyze on data science tasks.*

Single-ability Fine-tuning. Since most foundation LLMs have not been trained specifically for data science tasks, in this stage, we first enhance the various single abilities that data science relies, primarily including reasoning, structured data understanding, and code generation, which correspond respectively to the actions $\langle$Analyze$\rangle$, $\langle$Understand$\rangle$, and $\langle$Code$\rangle$. Specifically, we fine-tune the foundation LLM using long CoT data (i.e., including reasoning traces) of general tasks, structured data understanding, code generation. This stage of training mirrors the human learning process from a beginner to a data science practitioner in acquiring specialized skills, enhancing LLM’s single ability in various aspects of data science.

Multi-ability Agentic Training. Building on the mastery of various single abilities, we employ agentic reinforcement learning to train DeepAnalyze to apply multiple abilities in real-world environments to complete complex data science tasks. To ensure the quality of reinforcement learning, we first perform a cold start by fine-tuning the LLM on synthesized interaction trajectories, enabling it to acquire basic capabilities in orchestrating and optimizing individual actions. Subsequently, we train DeepAnalyze in real-world environments using reinforcement learning with group relative policy optimization (GRPO) (Shao et al., 2024). For each question $q$ in training data $D$, GRPO samples a group of $G$ outputs $\{o_{1},\cdots,o_{G}\}$ from the old policy $\pi_{\theta_{\mathrm{old}}}$ and then optimizes the policy model $\pi_{\theta}$ by maximizing the following objective:

$\begin{split}\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{q\thicksim D,\{o_{i}\}_{i=1}^{G}\thicksim\pi_{\theta_{\mathrm{old}}}(\cdot|q)}\Biggl[\frac{1}{G}\sum_{i=1}^{G}\Bigl(\min\Bigl(\\
\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{\mathrm{old}}}(o_{i}|q)}A_{i},\mathrm{clip}\Bigl(\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{\mathrm{old}}}(o_{i}|q)},1-\varepsilon,1+\varepsilon\Bigr)A_{i}\Bigr)\\
-\beta\mathrm{D}_{KL}\left(\pi_{\theta}\parallel\pi_{\mathrm{ref}}\right)\Bigr)\Biggr]\end{split}$ | | | | (1) |

where $A_{i}$ is the advantage calculated from the rewards $\{r_{1},\cdots,r_{G}\}$ of outputs within each group, $\pi_{\mathrm{ref}}$ is the reference model, $\varepsilon$ and $\beta$ are hyperparameters.

Hybrid Reward Modeling. The effectiveness of agentic reinforcement learning critically depends on both the training data and the reward function. We use the agentic interaction trajectories synthesized in Section 3.3 as training data, covering three broad categories of data science tasks: data question answering, specific data tasks (e.g., data preparation, analysis, visualization, modeling, and insight extraction), and open-ended research. Since many data science tasks are inherently open-ended, we adopt a hybrid reward modeling that combines rule-based rewards with LLM-as-a-judge rewards. For all tasks, we first check whether the output format conforms to DeepAnalyze’s architecture (i.e., whether it contains exactly five types of actions with the correct format). If the format is incorrect, we directly assign a reward of $R=-1$.

For data question answering and data-centric tasks, which have reference answers, the reward $R$ of each output $o$ are calculated using accuracy and interaction trajectory quality:

$\displaystyle R=\frac{1}{2}(\mathbbm{1}_{acc}(o)+{S}_{interaction}(o))$ | | | | (2) |

where $\mathbbm{1}_{acc}(o)\in\{0,1\}$ indicates whether the result is correct, and ${S}_{interaction}(o)\in[0,1]$ is a score to evaluate the quality of the interaction trajectory.

For open-ended research, the reward $R$ of each output $o$ is evaluated based on the quality of the final research report and the research process. Denoting each interaction turn in output $o$ as $T_{i}\in o$, the reward $R$ is calculate as:

$\displaystyle R=\frac{1}{3}\!\left(\!{S}_{report}(o)\!+\!\min(\frac{|T|}{N^{T}},1)\!+\!\frac{1}{|T|}\sum_{T_{i}\in o}\mathbbm{1}_{success}(T_{i})\!\right)$ | | | | (3) |

where ${S}_{report}(o)$ is the score that evaluates the generated report from five aspects: usefulness, richness, soundness, interpretability, and readability. $|T|$ measures the interaction turns with the environment, where $N^{T}=10$ is a hyperparameter. $\mathbbm{1}_{success}(T_{i})$ indicates whether each interaction turn is successful. This reward encourages DeepAnalyze to engage in more successful interactions with the environment and to generate high-quality research report.

Through curriculum-based agentic training, we progressively enhances DeepAnalyze’s capabilities following an easy-to-hard schedule, ultimately enabling it to autonomously accomplish a variety of data science tasks in real-world environments.

### 3.3 Data-grounded Trajectory Synthesis

The proposed curriculum-based agentic training relies on high-quality reasoning and interaction trajectory data, while such data is unfortunately scarce for data science tasks. To overcome this challange, we introduce a data-grounded trajectory synthesis framework that automatically constructs high-quality trajectory data tailored for data science tasks. The data-grounded trajectory synthesis framework consists of two parts: Reasoning Trajectory Synthesis, which construct the reasoning trajectory for existing structured data instruction datasets, and Interaction Trajectory Synthesis, which constructs entire data science trajectory based on structured data sources in the environment.

| Models | Coarse-grained Metrics | Fine-grained Metrics | Score |

| Success |

| Rate |

| Completion |

| Rate |

VLM

| F1: Data |

| Preparation |

| F2: Plot |

| Validity |

| F3: Data |

| Exploration |

| F4: Data |

| Visualization |

| F5: Data |

| Modeling |

Close-Source API-Based Agent

o1-mini 29.77 45.26 2.87 44.63 19.27 36.01 30.94 23.81 38.78

GPT-4o-mini 50.63 57.78 3.05 60.30 48.02 57.84 59.24 53.54 54.18

GPT-4o 66.31 68.44 3.91 75.93 56.14 69.33 71.35 57.67 64.51

GPT-4-Turbo 51.93 58.87 3.09 62.30 41.62 57.75 60.25 50.75 54.65

Claude-3-5-Sonnet 47.48 58.11 2.14 49.07 36.94 55.84 52.87 46.04 52.29

GLM-4-Flash 30.32 34.04 1.33 36.53 29.42 32.57 27.64 14.44 30.74

Open-Source LLM-based Agent

Llama-3.1-8B-Instruct 24.73 33.89 1.29 38.24 18.25 21.98 22.89 25.85 29.69

Gemma-2-9B-it 7.07 11.00 1.06 26.16 16.90 23.81 18.11 17.15 12.66

GLM-4-9B-Chat 25.72 30.38 1.69 31.51 23.15 28.07 27.19 19.14 27.57

Qwen2.5-7B-Instruct 43.83 50.74 1.43 51.18 36.41 47.25 45.24 34.77 45.99

Qwen2-7B-Instruct 22.84 25.58 1.16 30.93 20.78 28.73 25.87 7.52 23.52

Yi-1.5-9B-Chat-16K 38.20 42.35 0.73 38.14 36.36 35.64 37.08 27.79 38.22

CodeLlama-13B-Instruct 10.49 14.64 0.04 11.67 11.34 9.43 14.43 5.15 12.64

CodeLlama-7B-Instruct 2.88 3.97 0.00 3.53 2.37 2.57 1.74 1.59 3.31

StarCoder2-15B 2.07 2.61 0.07 2.57 1.81 1.59 3.43 1.19 2.33

Deepseek-Coder-6.7B-instruct 37.03 41.62 1.93 43.49 34.57 46.36 46.49 18.09 38.45

Qwen2.5-Coder-7B-Instruct 45.18 53.11 1.48 51.58 43.21 43.87 42.50 35.23 47.67

Agentic Model

DeepAnalyze-8B 59.91 66.24 2.86 71.68 67.86 58.62 69.09 33.33 61.11

*Table 1: Performance on DataSciBench. ‘Success Rate’ and ‘Completion Rate’ are pass rate and accuracy. ‘VLM’ and ‘F1-F5’ scores evaluate performance on various fine-grained data science sub-tasks, ‘Score’ denotes the overall performance.*

Reasoning Trajectory Synthesis. Existing instruction datasets for structured data, such as TableQA (Li et al., 2023b; Lei et al., 2025), structured knowledge grounding (Zhuang et al., 2024), and data science code generation, is useful to improve LLM’s single capability. However, these datasets typically contain only instructions and responses, without the reasoning process. To address this limitation, we enhance existing datasets by synthesizing complex and refined reasoning trajectories, which are used for DeepAnalyze’s single ability training.

As shown in Figure 5(a), given the instruction–response pairs in the original dataset, the reasoning trajectory synthesis involves distillation and refinement steps. In the distillation step, we employ advanced LLMs as teacher models to extract their reasoning trajectories, whose correctness is verified by comparing the generated responses with the ground truth responses (DeepSeek-AI, 2025). To strengthen the model’s understanding of structured data, the distilled reasoning is reformulated by advanced LLMs into two complementary components: $\langle$Analyze$\rangle$ (reasoning process) and $\langle$Understand$\rangle$ (structured data understanding). Building on this, we introduce keyword-guided refinement to further enhance the reasoning trajectories with a focus on structured data. Previous works have shown that certain keywords, such as “but”/“wait”, play a crucial role in reasoning (Zhang et al., 2025a; Shen et al., 2025). Following this insight, we construct a key reasoning vocabulary and sample key reasoning words to insert into the reasoning trajectory, thereby improving its reasoning on structured data. Appendix B provides an example of keyword-guided refinement, where inserting keywords enhances the reasoning process by focusing more on the data, thereby improving the quality of the reasoning trajectory. Through reasoning trajectory synthesis, we can effectively leverage existing datasets to improve DeepAnalyze’s single ability in reasoning, structured data understanding, and code generation.

*Figure 6: Length distribution of training data.*

Interaction Trajectory Synthesis. To enable DeepAnalyze to autonomously orchestrate and optimize multiple abilities in real-world environments, it is essential to construct multi-turn interaction trajectory data with the environment, yet such data is extremely scarce. In contrast, NL2SQL datasets such as Spider (Yu et al., 2018) and BIRD (Li et al., 2024) provide abundant structured data sources. To bridge this gap, we develop a multi-agent system to synthesize data science interaction trajectories from these data sources.

The multi-agent system involves three roles: questioner, solver, and inspector. The questioner observes the data sources in the environment and accordingly formulates a data science problem, conditioned on a sampled task type (e.g., data preparation, data analysis, data modeling, data insight, or open-ended research). Simultaneously, the questioner produces a checklist that serves as the evaluation criterion, including interaction-level constraints (e.g., number of turns, code library) and environment-level constraints (e.g., whether new files are generated, detailed file name). Given the data science problem and the data sources, the solver interacts with the environment using the introduced five actions to complete the task. Finally, the inspector validates the trajectory by checking the interaction process and environmental changes against the checklist, determining whether the trajectory should be accepted. Importantly, filtering trajectories based on both interaction details and environmental changes substantially improves the quality of synthesized data. Through interaction trajectory synthesis, the high-quality multi-turn interaction data can be used for multi-ability agentic training (cold start and RL).

### 3.4 DataScience-Instruct-500K

We develop DeepAnalyze based on the constructed data in Sec.3.3. During the single-ability fine-tuning stage, we employ the reasoning trajectories built for data science, along with 100K general reasoning samples from AM-DeepSeek-R1-0528-Distilled. In the multi-ability agentic training stage (including both cold start and RL phases), we use the interaction trajectories constructed for data science.

Figure 6 illustrates the length distribution of training data in both stages, with a sequence length of 8K in the first stage and 32K in the second. In terms of scale, the single-ability fine-tuning stage consists of approximately 470K samples, the cold-start phase of multi-ability training includes 20K samples, and the RL phase comprises 15K samples, resulting in a total of around 500K samples. We release all training data, named DataScience-Instruct-500K, which can be used to train LLMs for data science tasks.

## 4 Experiments

### 4.1 Benchmarks

We conduct experiments on 12 data science benchmarks.

DataSciBench (Zhang et al., 2025b) is the latest benchmark to evaluate LLM’s capabilities on the entire data science pipeline, covering data preparation, data analysis, data modeling, data visualizatoin, and data insight.

DSBench (Jing et al., 2025) evaluates data analysis and modeling capabilities, comprising 540 real-world tasks collected from ModelOff and Kaggle competitions.

DABStep (Egg et al., 2025) is a data agent benchmark with 450 real-world data analysis tasks designed to evaluate the multi-step reasoning abilities of agents.

DABStep-Research is a benchmark we constructed based on DABStep (Egg et al., 2025) to evaluate the capability of data science report generation. Considering that existing data science benchmarks rarely assess deep research abilities on structured data, we propose DABStep-Research to measure the capability to generate comprehensive data research reports from raw data sources. The evaluation covers five aspect: data preparation, data analysis, data insight, report generation, and open-ended data research. Please refer to Appendix A for details on its construction and cases.

DS-1000 (Lai et al., 2023) is a code generation benchmark containing 1000 data science problems spanning seven Python libraries such as NumPy, Pandas, Matplotlib, etc.

TableQA Benchmarks are a series of question-answering benchmarks based on structured tables, including WikiTQ (Pasupat & Liang, 2015), HybridQA (Chen et al., 2020), MultiHiertt (Zhao et al., 2022), OTT-QA (Chen et al., 2021a), FinQA (Chen et al., 2021b), TAT-QA (Nan et al., 2022), and HiTab (Cheng et al., 2022).

*Figure 7: Performance on DSBench (data analysis).*

### 4.2 Experimental Setup

We build DeepAnalyze-8B based on DeepSeek-R1-0528-Qwen3-8B as the foundation LLM. We use ms-swift (Zhao et al., 2024) and SkyRL (Liu et al., 2025) toolkit to accomplish single-ability fine-tuning and multi-ability agentic training respectively. The training data come from DataScience-Instruct-500K, as described in Sec.3.4. During inference, we employ the vLLM engine (Kwon et al., 2023) to deploy DeepAnalyze-8B for efficiency. All training and inference are conducted on NVIDIA A800 GPUs.

### 4.3 Main Results

Capability on End-to-end Data Science Pipeline. We evaluate DeepAnalyze on DataSciBench to assess its end-to-end data science capabilities, where each problem involves one or more sub-tasks such as data preparation, analysis, modeling, and visualization. We compare DeepAnalyze-8B with several workflow-based (ReAct) agents, covering 17 open-source and advanced proprietary LLMs. As shown in Table 1, coarse-grained metrics measure task success and sub-task completion rates, while fine-grained metrics evaluate detailed performance across individual stages of the data science pipeline. The results show that, despite having only 8B parameters, DeepAnalyze-8B achieves state-of-the-art performance among open-source LLM-based agents and even outperforms most advanced proprietary models (e.g., GPT-4-Turbo, GPT-4o-mini, Claude 3.5 Sonnet), ranking second only to GPT-4o. More importantly, unlike existing workflow-based agents, DeepAnalyze-8B accomplishes high-quality, end-to-end pipelines without relying on external orchestration frameworks such as ReAct. Prior studies have shown that models like o1-mini exhibit strong reasoning ability but often fail to execute complex data science tasks requiring precise instruction following and strategic planning (Zhang et al., 2025b). In contrast, DeepAnalyze benefits from agentic training, enabling autonomous orchestration and adaptive optimization in real-world environments, resulting in consistently superior performance.

Overall, DeepAnalyze-8B’s strong results on DataSciBench highlight its advanced problem-solving capabilities in autonomously orchestrating end-to-end data science pipelines.

| Methods | LLM | Success (%) | Performance | Cost ($) |

| Workflow-based Agent |

| AutoGen | Llama3-8b | 5.41 | 1.55 | 0.00 |

| Llama3-70b | 16.22 | 7.79 | 0.00 |

| GPT-3.5 | 8.11 | 6.02 | 0.41 |

| GPT-4 | 87.84 | 45.52 | 19.34 |

| GPT-4o | 71.62 | 34.74 | 12.27 |

| GPT-4o-mini | 22.97 | 11.24 | 0.10 |

| Code Interpreter | GPT-3.5 | 16.22 | 6.52 | 2.74 |

| GPT-4 | 54.05 | 26.14 | 38.81 |

| GPT-4o | 44.59 | 19.87 | 19.26 |

| GPT-4o-mini | 39.19 | 16.90 | 2.70 |

| Agentic Model |

| DeepAnalyze-8B | 90.63 | 39.41 | 0.00 |

*Table 2: Performance on DSBench (data modeling).*

| Methods | LLM | Easy Level (72 Cases) | Hard Level (378 Cases) | Overall (450 Cases) |

| Workflow-based Agent |

| ReAct | Llama-4-Scout | 52.78 | 1.85 | 10.00 |

| Qwen3-Coder | 54.17 | 3.44 | 11.56 |

| GPT-4o-mini | 69.44 | 3.44 | 14.00 |

| Deepseek-v3 | 66.67 | 5.56 | 15.34 |

| GPT-4o | 66.67 | 6.08 | 15.77 |

| Claude-3.5-Haiku | 77.78 | 5.03 | 16.67 |

| Llama-4-Maverick | 75.00 | 8.73 | 19.33 |

| GPT-4.1-mini | 77.78 | 8.99 | 20.00 |

| Claude-3.5-Sonnet | 77.78 | 9.26 | 20.22 |

| GPT-4.1 | 80.56 | 12.43 | 23.33 |

| Reasoning Prompt | o1 | 69.44 | 11.11 | 20.44 |

| Gemini-2.5-Pro | 66.67 | 12.70 | 21.34 |

| o3-mini | 72.22 | 13.76 | 23.11 |

| o4-mini | 76.39 | 14.55 | 24.44 |

| DS-Agent | Gemini-2.0-Flash | 61.11 | 9.79 | 18.00 |

| Open Data Scientist | Deepseek-v3 | 84.72 | 16.40 | 27.33 |

| I2I-Agent | Claude-3.5-Sonnet | 80.56 | 28.04 | 36.44 |

| Agentic Model |

| DeepAnalyze-8B | 70.83 | 32.80 | 38.88 |

*Table 3: Performance on DABStep benchmark.*

*Figure 8: Performance on DABStep-Research.*

| Models | WikiTQ | HybridQA | MultiHiertt | OTT-QA | FinQA | TAT-QA | HiTab | AVG |

| API-based LLMs |

| Claude | 82.02 | 39.36 | 40.98 | 62.69 | 57.45 | 53.09 | 75.96 | 58.79 |

| GPT-4o | 81.19 | 39.30 | 40.86 | 66.35 | 57.63 | 53.45 | 73.92 | 58.96 |

| Open-Source LLMs |

| DeepSeek-R1-0528 | 84.00 | 39.04 | 40.98 | 66.85 | 59.90 | 55.24 | 75.57 | 60.22 |

| TableGPT2-7B | 63.70 | 30.03 | 25.12 | 48.87 | 38.36 | 55.12 | 63.89 | 46.44 |

| Qwen2.5-32B-Inst | 79.65 | 38.20 | 37.74 | 56.50 | 59.20 | 67.29 | 73.29 | 58.84 |

| Qwen2.5-7B-Inst | 57.27 | 31.84 | 27.54 | 50.50 | 52.40 | 49.79 | 57.19 | 46.65 |

| DeepSeek-R1-0528-Qwen3-8B | 63.49 | 28.15 | 39.86 | 49.72 | 51.09 | 55.00 | 51.09 | 48.34 |

| Reasoning-Table (SFT) | 72.35 | 35.17 | 38.50 | 54.40 | 60.42 | 63.45 | 72.72 | 56.72 |

| Reasoning-Table (SFT+RL) | 75.46 | 42.83 | 39.56 | 68.68 | 64.46 | 73.75 | 73.61 | 62.62 |

| DeepAnalyze-8B (single-ability) | 81.86 | 39.27 | 44.58 | 53.12 | 62.50 | 66.87 | 76.26 | 60.64 |

| DeepAnalyze-8B | 83.24 | 42.95 | 48.29 | 64.73 | 63.30 | 70.64 | 78.16 | 64.47 |

*Table 4: Performance on TableQA benchmarks. ‘DeepAnalyze-8B (single-ability)’ is the model after the first stage fine-tuning.*

Capability on Individual Data Science Tasks. As most previous studies primarily focus on individual data science tasks such as data analysis and modeling, we further evaluate DeepAnalyze on these tasks using DSBench for a fair comparison. We first evaluate its statistical data analysis capabilities. As shown in Figure 7, DeepAnalyze-8B outperforms previous LLM prompting and workflow-based agents, demonstrating that its autonomous orchestration and adaptive optimization are more effective than the manually designed workflows used in agents such as Code Interpreter, Master-Slave (Kong et al., 2017), and Blackboard (Salemi et al., 2025). We then evaluate its data modeling capabilities. Table 2 reports the results on DSBench, where tasks involve training machine learning models (Jing et al., 2025). DeepAnalyze-8B achieves performance comparable to AutoGen-based workflows (Wu et al., 2024) built upon various advanced proprietary LLMs. Although it has fewer parameters and weaker single-turn reasoning ability, DeepAnalyze-8B can autonomously optimize its actions through environment feedback, achieving a high task success rate and strong overall performance.

To further evaluate DeepAnalyze’s ability to perform data analysis across multiple data types, including structured, semi-structured, and unstructured data (Egg et al., 2025), we evaluate it on DABStep, which contains diverse data formats such as markdown, CSV, and JSON. As shown in Table 3, DeepAnalyze-8B outperforms previous workflow-based agents, including ReAct (Yao et al., 2023), reasoning prompts, and specially designed workflows, particularly on hard-level tasks. While workflow-based systems can leverage the strong general capabilities of proprietary LLMs to perform well on easy tasks, their predefined workflows limit performance on complex scenarios. In contrast, DeepAnalyze, equipped with autonomous orchestration and adaptive optimization through agentic training, can iteratively interact with the environment like a human data scientist, achieving superior performance on complex tasks requiring long-chain reasoning.

| Models | Data Science Libraries | Overall |

| Pandas | NumPy | Matplotlib | Scikit-learn | SciPy | TensorFlow | PyTorch |

| Codex002 | 26.5 | 43.2 | 54.8 | 43.5 | 34.9 | 37.8 | 39.7 | 38.8 |

| GPT-3.5-turbo | 33.0 | 36.8 | 58.7 | 35.7 | 39.6 | 33.3 | 29.4 | 38.6 |

| GPT-4 | 41.9 | 56.8 | 65.2 | 50.4 | 48.1 | 46.7 | 47.1 | 51.0 |

| GPT-4-turbo | 42.3 | 61.8 | 71.6 | 50.4 | 50.0 | 53.3 | 50.0 | 53.9 |

| Kimi-K2-Instruct∗ | - | - | - | - | - | - | - | 40.2 |

| GLM-4.5∗ | - | - | - | - | - | - | - | 53.2 |

| LIMI∗ | - | - | - | - | - | - | - | 54.8 |

| DeepSeek-R1-0528-Qwen3-8B | 17.5 | 37.3 | 52.9 | 27.8 | 21.7 | 31.1 | 29.4 | 30.4 |

| DeepAnalyze-8B (single-ability) | 43.6 | 69.1 | 54.8 | 53.0 | 50.9 | 64.4 | 58.8 | 54.8 |

| DeepAnalyze-8B | 50.2 | 74.5 | 67.7 | 56.5 | 54.7 | 68.9 | 70.6 | 61.7 |

*Table 5: Performance on DS-1000. ∗ indicates that The results are derived from corresponding references. ‘DeepAnalyze-8B (single-ability)’ is the model after the first stage fine-tuning.*

| Models | WikiTQ | MultiHiertt | DS-1000 | DABStep |

| DeepAnalyze | 83.24 | 48.29 | 61.70 | 38.88 |

$\langle$$\rangle$ | - w/o Understand | 80.78 | 45.43 | 61.20 | 31.78 |

*Table 6: Ablation study on $\langle$Understand$\rangle$ action.*

Capability on Data-Oriented Deep Research. Deep research has emerged as an important task for evaluating the comprehensive capabilities of LLMs and agents. To this end, we introduce DABStep-Research, a benchmark designed to evaluate the data-oriented deep research capabilities of LLMs and agents. We compare DeepAnalyze-8B with advanced agent systems (i.e., state-of-the-art proprietary LLMs with tool-calling capabilities) on a suite of data research tasks spanning five categories: data preparation, data analysis, data insight, report generation (with a specified outline), and open-ended data research (fully unconstrained). Each task results in a research report, which is evaluated on both content quality and formatting. Figure 9 illustrates several representative cases from DABStep-Research.

The results in Figure 8 show that DeepAnalyze-8B consistently outperforms all compared systems across every task. Notably, agent systems built on proprietary LLMs with tool calls exhibit a significant performance drop on open-ended data research tasks compared to more instructive tasks, such as data preparation, analysis, and insight, where explicit steps or goals are provided. This decline stems from their lack of training in data science: without step-by-step guidance, they fail to perform autonomous orchestration and adaptive optimization. In contrast, DeepAnalyze-8B, trained in real-world environments, effectively handles fully open-ended data research tasks without predefined instructions. Moreover, it achieves a clear advantage in report format quality, generating outputs that closely resemble analyst-grade reports. This improvement is attributed to reward modeling that explicitly incorporates report quality during RL training. Appendix C further provides qualitative comparisons of research reports generated by DeepAnalyze-8B and reasoning models such as DeepSeek-R1 and o3-mini, highlighting DeepAnalyze-8B’s superior content depth and structured presentation.

Overall, DeepAnalyze-8B enables end-to-end autonomous data research, from raw data to analyst-grade reports, unlocking novel applications in data research.

Capability Related to Data Science. Beside data science tasks, we further evaluate DeepAnalyze-8B on DS-1000 and TableQA to evaluate its capabilities in code generation and structured data understanding, which are essential for complex data science. As reported in Table 5 and Table 4, DeepAnalyze-8B outperforms GPT-4-Turbo and GLM-4.5 (GLM-4.5-Team, 2025) on DS-1000, and surpasses the previous SOTA model Reasoning-Table (Lei et al., 2025) on TableQA. Compared with DeepSeek-R1-0528-Qwen3-8B, DeepAnalyze-8B achieves substantial gains in both abilities under the single-ability setting, demonstrating the effectiveness of the first-stage single-ability fine-tuning. Furthermore, agentic training on complex data science tasks further strengthens these specialized capabilities.

Overall, DeepAnalyze-8B’s strong performance on code generation and structured data understanding establishes a robust foundation for its advanced performance in end-to-end autonomous data science.

## 5 Analysis

### 5.1 Ablation on DeepAnalyze’s Actions

DeepAnalyze introduces five actions for autonomous data science, among which $\langle$Understand$\rangle$ is specifically designed for structured data understanding. To evaluate the effect of incorporating $\langle$Understand$\rangle$ independently from reasoning process (i.e., $\langle$Analyze$\rangle$), we conduct an ablation study, as reported in Table 6. The results show that removing $\langle$Understand$\rangle$ leads to performance drops on structured data understanding tasks (WikiTQ, MultiHiertt) as well as data analysis tasks (DABStep), demonstrating the advantage of introducing $\langle$Understand$\rangle$ in DeepAnalyze.

| Training Methods | WikiTQ | MultiHiertt | DS-1000 | DABStep |

| Curriculum-based Agentic Training | 83.24 | 48.29 | 61.70 | 38.88 |

| -Only Single-ability Fine-tuning | 81.86 | 44.58 | 54.80 | 15.34 |

| -Only Multi-ability Agentic Training | 80.32 | 43.29 | 53.20 | 30.66 |

| -One-stage Training | 82.13 | 46.23 | 54.80 | 36.89 |

*Table 7: Ablation study on the curriculum-based agentic training.*

### 5.2 Superiority of Curriculum-based Agentic Training

To address the challenges arising from the multiple ability requirements in data science, we introduce curriculum-based agentic training, inspired by the learning path of human data scientists, where first fine-tuning on single abilities and then agentic training on complex tasks that require multiple abilities. To evaluate its effectiveness, we compare several training methods, including “Only Single-ability Fine-tuning”, “Only Multi-ability Agentic Training”, and “One-stage Training”, which directly mix the single-ability data into the cold-start of multi-ability agentic training (i.e., the conventional agentic training methods).

As shown in Table 7, “Only Single-ability Fine-tuning” fails to handle complex tasks in DABStep that require multi-turn interaction with the environment, and “Only Multi-ability Agentic Training” struggles to achieve strong performance when single ability are not well established. Compared with “One-stage Training”, a scheduled training process from simple (single-ability) to complex (multi-ability) proves more beneficial for model performance using the same data. Therefore, for tasks that rely on multiple abilities, curriculum-based agentic training effectively enhances overall model performance.

| Reasoning Trajectory | WikiTQ | HybridQA | MultiHiertt | HiTab |

| Original | 75.54 | 34.42 | 39.29 | 72.95 |

| + Distillation | 78.80 | 36.12 | 41.24 | 74.44 |

| + Distillation + Refinement | 80.25 | 38.84 | 43.47 | 75.86 |

*Table 8: Performance under various reasoning trajectory synthesis.*

### 5.3 Advantage of Reasoning Trajectory Synthesis

During data synthesis, we propose reasoning trajectory synthesis that incorporates distillation and refinement to enhance the model’s reasoning ability over structured data. To validate its effectiveness, we compare the model’s performance when trained on original, distilled, and refined data, where the original data are derived from Reasoning-Table. As reported in Table 1, both distillation and refinement improve the model’s understanding of structured data. In particular, compared with commonly used distillation methods, we additionally introduce a refinement stage, which incorporates key reasoning vocabulary to strengthen the reasoning trajectory’s focus on structured data, thereby improving the overall data quality.

## 6 Conclusion and Future Work

DeepAnalyze brings a major leap forward in autonomous data science, demonstrating unprecedented capabilities across a wide spectrum of data-centric tasks. Powered by curriculum-based agentic training and data-grounded trajectory synthesis, DeepAnalyze-8B outperforms state-of-the-art closed-source LLMs on 12 data science benchmarks.

More importantly, DeepAnalyze goes beyond predefined workflows, as it enables open-ended data research and generates analyst-grade reports, advancing a long-standing goal of the data science community: automatically extracting actionable insights from raw data. As a result, this work marks a paradigm shift in autonomous data science from workflow-based agents to agentic models, paving the way for the next generation of intelligent data systems in areas such as data discovery, data governance, data ecosystems, and data management.

## References

- Chen et al. (2020) Chen, W., Zha, H., Chen, Z., Xiong, W., Wang, H., and Wang, W. Y. HybridQA: A dataset of multi-hop question answering over tabular and textual data. In Cohn, T., He, Y., and Liu, Y. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1026–1036, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.91. URL https://aclanthology.org/2020.findings-emnlp.91/.

- Chen et al. (2021a) Chen, W., Chang, M.-W., Schlinger, E., Wang, W. Y., and Cohen, W. W. Open question answering over tables and text. In International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id=MmCRswl1UYl.

- Chen et al. (2021b) Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., Moussa, R., Beane, M., Huang, T.-H., Routledge, B., and Wang, W. Y. FinQA: A dataset of numerical reasoning over financial data. In Moens, M.-F., Huang, X., Specia, L., and Yih, S. W.-t. (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3697–3711, Online and Punta Cana, Dominican Republic, November 2021b. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.300. URL https://aclanthology.org/2021.emnlp-main.300/.

- Cheng et al. (2022) Cheng, Z., Dong, H., Wang, Z., Jia, R., Guo, J., Gao, Y., Han, S., Lou, J.-G., and Zhang, D. HiTab: A hierarchical table dataset for question answering and natural language generation. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1094–1110, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.78. URL https://aclanthology.org/2022.acl-long.78/.

- De Bie et al. (2022) De Bie, T., De Raedt, L., Hernández-Orallo, J., Hoos, H. H., Smyth, P., and Williams, C. K. Automating data science. Communications of the ACM, 65(3):76–87, 2022.

- DeepSeek-AI (2025) DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

- Egg et al. (2025) Egg, A., Goyanes, M. I., Kingma, F., Mora, A., von Werra, L., and Wolf, T. Dabstep: Data agent benchmark for multi-step reasoning, 2025. URL https://arxiv.org/abs/2506.23719.

- Fang et al. (2024) Fang, X., Xu, W., Tan, F. A., Hu, Z., Zhang, J., Qi, Y., Sengamedu, S. H., and Faloutsos, C. Large language models (LLMs) on tabular data: Prediction, generation, and understanding - a survey. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=IZnrCGF9WI.

- GLM-4.5-Team (2025) GLM-4.5-Team. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models, 2025. URL https://arxiv.org/abs/2508.06471.

- Guo et al. (2024) Guo, S., Deng, C., Wen, Y., Chen, H., Chang, Y., and Wang, J. Ds-agent: Automated data science by empowering large language models with case-based reasoning. In ICML, 2024. URL https://openreview.net/forum?id=LfJgeBNCFI.

- Hollmann et al. (2023) Hollmann, N., Müller, S., and Hutter, F. Large language models for automated data science: Introducing caafe for context-aware automated feature engineering. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 44753–44775. Curran Associates, Inc., 2023.

- Hong et al. (2024) Hong, S., Lin, Y., Liu, B., Liu, B., Wu, B., Zhang, C., Wei, C., Li, D., Chen, J., Zhang, J., Wang, J., Zhang, L., Zhang, L., Yang, M., Zhuge, M., Guo, T., Zhou, T., Tao, W., Tang, X., Lu, X., Zheng, X., Liang, X., Fei, Y., Cheng, Y., Gou, Z., Xu, Z., and Wu, C. Data interpreter: An llm agent for data science, 2024. URL https://arxiv.org/abs/2402.18679.

- Hong et al. (2025) Hong, S., Lin, Y., Liu, B., Liu, B., Wu, B., Zhang, C., Li, D., Chen, J., Zhang, J., Wang, J., Zhang, L., Zhang, L., Yang, M., Zhuge, M., Guo, T., Zhou, T., Tao, W., Tang, R., Lu, X., Zheng, X., Liang, X., Fei, Y., Cheng, Y., Ni, Y., Gou, Z., Xu, Z., Luo, Y., and Wu, C. Data interpreter: An LLM agent for data science. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 19796–19821, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1016. URL https://aclanthology.org/2025.findings-acl.1016/.

- Jiang et al. (2023) Jiang, J., Zhou, K., Dong, Z., Ye, K., Zhao, X., and Wen, J.-R. StructGPT: A general framework for large language model to reason over structured data. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 9237–9251, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.574. URL https://aclanthology.org/2023.emnlp-main.574/.

- Jin et al. (2025) Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., Zamani, H., and Han, J. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.09516.

- Jing et al. (2025) Jing, L., Huang, Z., Wang, X., Yao, W., Yu, W., Ma, K., Zhang, H., Du, X., and Yu, D. DSBench: How far are data science agents from becoming data science experts? In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=DSsSPr0RZJ.

- Kong et al. (2017) Kong, X., Xin, B., Liu, F., and Wang, Y. Revisiting the master-slave architecture in multi-agent deep reinforcement learning, 2017. URL https://arxiv.org/abs/1712.07305.

- Kwon et al. (2023) Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

- Lai et al. (2023) Lai, Y., Li, C., Wang, Y., Zhang, T., Zhong, R., Zettlemoyer, L., Yih, W.-T., Fried, D., Wang, S., and Yu, T. DS-1000: A natural and reliable benchmark for data science code generation. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 18319–18345. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/lai23b.html.

- Lei et al. (2025) Lei, F., Meng, J., Huang, Y., Chen, T., Zhang, Y., He, S., Zhao, J., and Liu, K. Reasoning-table: Exploring reinforcement learning for table reasoning, 2025. URL https://arxiv.org/abs/2506.01710.

- Li et al. (2024) Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., Wang, B., Qin, B., Geng, R., Huo, N., et al. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. Advances in Neural Information Processing Systems, 36, 2024.

- Li et al. (2023a) Li, P., He, Y., Yan, C., Wang, Y., and Chaudhuri, S. Auto-tables: Synthesizing multi-step transformations to relationalize tables without using examples, 2023a. URL https://arxiv.org/abs/2307.14565.

- Li et al. (2023b) Li, P., He, Y., Yashar, D., Cui, W., Ge, S., Zhang, H., Fainman, D. R., Zhang, D., and Chaudhuri, S. Table-gpt: Table-tuned gpt for diverse table tasks, 2023b. URL https://arxiv.org/abs/2310.09263.

- Li et al. (2025) Li, X., Dong, G., Jin, J., Zhang, Y., Zhou, Y., Zhu, Y., Zhang, P., and Dou, Z. Search-o1: Agentic search-enhanced large reasoning models, 2025. URL https://arxiv.org/abs/2501.05366.

- Liu et al. (2025) Liu, S., Hegde, S., Cao, S., Zhu, A., Li, D., Griggs, T., Tang, E., Malik, A., Hakhamaneshi, K., Liaw, R., Moritz, P., Zaharia, M., Gonzalez, J. E., and Stoica, I. Skyrl-sql: Matching gpt-4o and o4-mini on text2sql with multi-turn rl, 2025.

- Liu et al. (2024) Liu, X., Shen, S., Li, B., Ma, P., Jiang, R., Zhang, Y., Fan, J., Li, G., Tang, N., and Luo, Y. A survey of nl2sql with large language models: Where are we, and where are we going? arXiv preprint arXiv:2408.05109, 2024.

- Mohammadjafari et al. (2025) Mohammadjafari, A., Maida, A. S., and Gottumukkala, R. From natural language to sql: Review of llm-based text-to-sql systems, 2025. URL https://arxiv.org/abs/2410.01066.

- Nan et al. (2022) Nan, L., Hsieh, C., Mao, Z., Lin, X. V., Verma, N., Zhang, R., Kryściński, W., Schoelkopf, H., Kong, R., Tang, X., Mutuma, M., Rosand, B., Trindade, I., Bandaru, R., Cunningham, J., Xiong, C., Radev, D., and Radev, D. FeTaQA: Free-form table question answering. Transactions of the Association for Computational Linguistics, 10:35–49, 2022. doi: 10.1162/tacl_a_00446. URL https://aclanthology.org/2022.tacl-1.3/.

- Nascimento et al. (2024) Nascimento, N., Guimaraes, E., Chintakunta, S. S., and Boominathan, S. A. Llm4ds: Evaluating large language models for data science code generation, 2024. URL https://arxiv.org/abs/2411.11908.

- Nejjar et al. (2024) Nejjar, M., Zacharias, L., Stiehle, F., and Weber, I. Llms for science: Usage for code generation and data analysis, 2024. URL https://arxiv.org/abs/2311.16733.

- OpenAI (2023) OpenAI. Gpt-4 technical report, 2023.

- OpenAI (2024) OpenAI. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720.

- Ouyang et al. (2025) Ouyang, G., Chen, J., Nie, Z., Gui, Y., Wan, Y., Zhang, H., and Chen, D. nvAgent: Automated data visualization from natural language via collaborative agent workflow. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 19534–19567, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.960. URL https://aclanthology.org/2025.acl-long.960/.

- Pan et al. (2025) Pan, B., Fu, Y., Wang, K., Lu, J., Pan, L., Qian, Z., Chen, Y., Wang, G., Zhou, Y., Zheng, L., Tang, Y., Wen, Z., Wu, Y., Lu, J., Zhu, B., Zhu, M., Zhang, B., and Chen, W. Vis-shepherd: Constructing critic for llm-based data visualization generation, 2025. URL https://arxiv.org/abs/2506.13326.

- Pan et al. (2023) Pan, L., Saxon, M., Xu, W., Nathani, D., Wang, X., and Wang, W. Y. Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies, 2023. URL https://arxiv.org/abs/2308.03188.

- Pasupat & Liang (2015) Pasupat, P. and Liang, P. Compositional semantic parsing on semi-structured tables. In Zong, C. and Strube, M. (eds.), Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1470–1480, Beijing, China, July 2015. Association for Computational Linguistics. doi: 10.3115/v1/P15-1142. URL https://aclanthology.org/P15-1142/.

- Plaat et al. (2025) Plaat, A., van Duijn, M., van Stein, N., Preuss, M., van der Putten, P., and Batenburg, K. J. Agentic large language models, a survey, 2025. URL https://arxiv.org/abs/2503.23037.

- Ren et al. (2025) Ren, Z. Z., Shao, Z., Song, J., Xin, H., Wang, H., Zhao, W., Zhang, L., Fu, Z., Zhu, Q., Yang, D., Wu, Z. F., Gou, Z., Ma, S., Tang, H., Liu, Y., Gao, W., Guo, D., and Ruan, C. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition, 2025. URL https://arxiv.org/abs/2504.21801.

- Salemi et al. (2025) Salemi, A., Parmar, M., Goyal, P., Song, Y., Yoon, J., Zamani, H., Palangi, H., and Pfister, T. Llm-based multi-agent blackboard system for information discovery in data science, 2025. URL https://arxiv.org/abs/2510.01285.

- Sapkota et al. (2025a) Sapkota, R., Roumeliotis, K. I., and Karkee, M. Ai agents vs. agentic ai: A conceptual taxonomy, applications and challenges. Information Fusion, 126:103599, February 2025a. ISSN 1566-2535. doi: 10.1016/j.inffus.2025.103599. URL http://dx.doi.org/10.1016/j.inffus.2025.103599.

- Sapkota et al. (2025b) Sapkota, R., Roumeliotis, K. I., and Karkee, M. Vibe coding vs. agentic coding: Fundamentals and practical implications of agentic ai, 2025b. URL https://arxiv.org/abs/2505.19443.

- Shao et al. (2024) Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

- Shen et al. (2025) Shen, S., Huang, F., Zhao, Z., Liu, C., Zheng, T., and Zhu, D. Long is more important than difficult for training reasoning models, 2025. URL https://arxiv.org/abs/2503.18069.

- Sun et al. (2025a) Sun, M., Han, R., Jiang, B., Qi, H., Sun, D., Yuan, Y., and and, J. H. Lambda: A large model based data agent. Journal of the American Statistical Association, 0(ja):1–20, 2025a. doi: 10.1080/01621459.2025.2510000. URL https://doi.org/10.1080/01621459.2025.2510000.

- Sun et al. (2025b) Sun, M., Han, R., Jiang, B., Qi, H., Sun, D., Yuan, Y., and Huang, J. A survey on large language model-based agents for statistics and data science, 2025b. URL https://arxiv.org/abs/2412.14222.

- Wang et al. (2025) Wang, P., Yu, Y., Chen, K., Zhan, X., and Wang, H. Large language model-based data science agent: A survey, 2025. URL https://arxiv.org/abs/2508.02744.

- Wen et al. (2024) Wen, Y., Yin, P., Shi, K., Michalewski, H., Chaudhuri, S., and Polozov, A. Grounding data science code generation with input-output specifications, 2024. URL https://arxiv.org/abs/2402.08073.

- Wu et al. (2024) Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., Awadallah, A. H., White, R. W., Burger, D., and Wang, C. Autogen: Enabling next-gen LLM applications via multi-agent conversations. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=BAakY1hNKS.

- Xu et al. (2025) Xu, Y., He, S., Chen, J., Xiangrong, Z., Wang, B., Liu, G., Zhao, J., and Liu, K. Llasa: Large language and structured data assistant, 2025. URL https://arxiv.org/abs/2411.14460.

- Xue et al. (2024) Xue, S., Jiang, C., Shi, W., Cheng, F., Chen, K., Yang, H., Zhang, Z., He, J., Zhang, H., Wei, G., Zhao, W., Zhou, F., Qi, D., Yi, H., Liu, S., and Chen, F. Db-gpt: Empowering database interactions with private large language models, 2024. URL https://arxiv.org/abs/2312.17449.

- Yang et al. (2021) Yang, J., He, Y., and Chaudhuri, S. Auto-pipeline: Synthesizing complex data pipelines by-target using reinforcement learning and search. volume 14, pp. 2563 – 2576, 2021.

- Yang et al. (2024) Yang, Z., Zhou, Z., Wang, S., Cong, X., Han, X., Yan, Y., Liu, Z., Tan, Z., Liu, P., Yu, D., Liu, Z., Shi, X., and Sun, M. MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization. In Findings of the Association for Computational Linguistics ACL 2024, pp. 11789–11804, Stroudsburg, PA, USA, aug 2024. Association for Computational Linguistics. ISBN 9798400704901. doi: 10.18653/v1/2024.findings-acl.701.

- Yao et al. (2023) Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE_vluYUL-X.

- Yu et al. (2018) Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., and Radev, D. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In Riloff, E., Chiang, D., Hockenmaier, J., and Tsujii, J. (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 3911–3921, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1425. URL https://aclanthology.org/D18-1425/.

- Zhang et al. (2025a) Zhang, A., Chen, Y., Pan, J., Zhao, C., Panda, A., Li, J., and He, H. Reasoning models know when they’re right: Probing hidden states for self-verification, 2025a. URL https://arxiv.org/abs/2504.05419.

- Zhang et al. (2024) Zhang, D., Wu, J., Lei, J., Che, T., Li, J., Xie, T., Huang, X., Zhang, S., Pavone, M., Li, Y., Ouyang, W., and Zhou, D. Llama-berry: Pairwise optimization for o1-like olympiad-level mathematical reasoning, 2024. URL https://arxiv.org/abs/2410.02884.

- Zhang et al. (2025b) Zhang, D., Zhoubian, S., Cai, M., Li, F., Yang, L., Wang, W., Dong, T., Hu, Z., Tang, J., and Yue, Y. Datascibench: An llm agent benchmark for data science, 2025b. URL https://arxiv.org/abs/2502.13897.

- Zhang et al. (2025c) Zhang, X., Luo, S., Zhang, B., Ma, Z., Zhang, J., Li, Y., Li, G., Yao, Z., Xu, K., Zhou, J., Zhang-Li, D., Yu, J., Zhao, S., Li, J., and Tang, J. TableLLM: Enabling tabular data manipulation by LLMs in real office usage scenarios. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 10315–10344, Vienna, Austria, July 2025c. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.538. URL https://aclanthology.org/2025.findings-acl.538/.

- Zhao et al. (2022) Zhao, Y., Li, Y., Li, C., and Zhang, R. MultiHiertt: Numerical reasoning over multi hierarchical tabular and textual data. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6588–6600, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.454. URL https://aclanthology.org/2022.acl-long.454/.

- Zhao et al. (2024) Zhao, Y., Huang, J., Hu, J., Wang, X., Mao, Y., Zhang, D., Jiang, Z., Wu, Z., Ai, B., Wang, A., Zhou, W., and Chen, Y. Swift:a scalable lightweight infrastructure for fine-tuning, 2024. URL https://arxiv.org/abs/2408.05517.

- Zheng et al. (2025) Zheng, Y., Fu, D., Hu, X., Cai, X., Ye, L., Lu, P., and Liu, P. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments, 2025. URL https://arxiv.org/abs/2504.03160.

- Zhuang et al. (2024) Zhuang, A., Zhang, G., Zheng, T., Du, X., Wang, J., Ren, W., Huang, S. W., Fu, J., Yue, X., and Chen, W. Structlm: Towards building generalist models for structured knowledge grounding, 2024. URL https://arxiv.org/abs/2402.16671.

## Appendix A Construction of DABStep-Research Benchmark

Existing data science benchmarks typically focus only on evaluating the ability of LLMs to solve specific tasks. However, with the rise of deep research, there is an urgent need for a benchmark that assesses LLMs’ capabilities in data-oriented deep research, which ask LLMs to conduct data research and generate research reports based on given instructions and data sources.

Construction. To this end, we constructed DABStep-Research, which is built upon the data sources proposed in DABStep (Egg et al., 2025). DABStep-Research consists of 100 tasks divided into five categories: data preparation, data analysis, data insight, report generation, and open-ended data research. In particular, tasks under the “report generation” category specify detailed report formats in the instruction, such as title, outline and specific requirements, thereby evaluating how well LLMs can follow instructions when generating research reports. The “open-ended data research” category involves fully open research tasks without any constraint on research direction or method. In addition to the instructions and data sources, we also provide a checklist to serve as a reference for scoring, helping evaluators determine whether the elements in a research report meet the given requirements. Figure 1 illustrates specific examples from DABStep-Research.

Evaluation. We use the LLM-as-a-judge to evaluate LLM performance on DABStep-Research. Specifically, given the instruction, checklist, and the report generated by an LLM, we employ a state-of-the-art LLM as the evaluator to assign a score from 1 to 5 based on two aspects: content and format. The prompt used for the LLM-as-judge evaluation is shown below.

*Figure 9: Cases in the constructed DABStep-Research benchmark, including data preparation, data analysis, data insight, report generation, and open-ended data research.*

## Appendix B Keyword-guided Reasoning Trajectory Synthesis

We present examples of Keyword-guided Reasoning Trajectory Synthesis in Figure 10. Specifically, the “Question” and Original Response are taken from existing TableQA datasets.

In the distillation step, we employ SOTA closed-source LLMs as teacher models to extract their reasoning trajectories, which is the most common way used in current data synthesis methods. However, such methods are more suitable for general reasoning processes. Since SOTA closed-source LLMs have not been specifically trained on domains like data science (e.g., structured data understanding), their reasoning trajectories tend to overlook the provided data.

Therefore, we introduce a refinement step to enhance the reasoning trajectory’s focus on structured data by inserting reasoning keywords that guide the reasoning process toward structured data understanding. Specifically, in the example shown in Figure 10, we sample three reasoning keywords (“What happens at the boundaries?”, “Let’s review the prior reasoning”, and “Let’s take a closer look at the table”) and ask the teacher model to refine its reasoning trajectory based on these keywords. We observe that the final “refinement” results exhibits a significantly stronger emphasis on repeated examination and reflection on structured data, thereby improving the overall quality of the reasoning trajectory. Overall, the proposed keyword-guided refinement is a useful data synthesis technique that can also be applied to the data synthesis of other complex tasks.

*Figure 10: Example of reasoning trajectory synthesis.*

## Appendix C Cases

In Figure 11, Figure 12, Figure 13, Figure 14, and Figure 15, we demonstrate a series of autonomous data science cases, covering the entire pipeline from data sources to analyst-grade research reports. These cases include data preparation, data analysis, data insight extraction, report generation under specific constraints, and fully open-ended data research. Compared with previous closed-source LLMs and tool-calling frameworks, DeepAnalyze can produce higher-quality, analyst-level reports, exhibiting a stronger ability for autonomous data research.

*Figure 11: A data preparation case of autonomous data science, from data sources to analyst-grade research reports.*

*Figure 12: A data analysis case of autonomous data science, from data sources to analyst-grade research reports.*

*Figure 13: A data insight case of autonomous data science, from data sources to analyst-grade research reports.*

*Figure 14: A case of autonomous data science with report constraints.*

*Figure 15: A case of autonomous data science for fully open-ended data research.*

<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2303.17760
     paper_id : 2303.17760
     source   : https://arxiv.org/html/2303.17760v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# CAMEL: Communicative Agents for “Mind” Exploration of Large Scale Language Model Society
https://www.camel-ai.org

Guohao Li    Hasan Abed Al Kader Hammoud*    Hani Itani*    Dmitrii Khizbullin    Bernard Ghanem Affiliation: King Abdullah University of Science and Technology (KAUST)

###### Abstract

The rapid advancement of conversational and chat-based language models has led to remarkable progress in complex task-solving. However, their success heavily relies on human input to guide the conversation, which can be challenging and time-consuming. This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents and provide insight into their “cognitive” processes. To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing . Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of chat agents, providing a valuable resource for investigating conversational language models. Our contributions include introducing a novel communicative agent framework, offering a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems, and open-sourcing our library to support research on communicative agents and beyond. The GitHub repository of this project is made publicly available on: https://github.com/lightaime/camel.

## 1 Introduction

Confronted with the complexities of real-world tasks, solving them often requires multiple steps. The rapid progress of conversational and chat-based large-scale language models (LLMs) has yielded remarkable achievements in complex task-solving [47, 48, 68, 52, 3, 7]. Nevertheless, it is worth noting that their success is heavily reliant on human input to guide the conversation in the right direction. This reliance necessitates users to provide relevant and precise prompts based on their intentions and the chat agent’s feedback. This can be challenging, time-consuming, and sometimes impossible. It often demands a deep understanding of the domain and expertise in crafting effective prompts. Consider an individual who lacks trading expertise; they would find it difficult to create suitable prompts for directing a communicative agent to develop a trading application. This predicament is raising a crucial question: can we replace human intervention with an autonomous communicative agent capable of steering the conversation toward task completion without any human supervision? To tackle this issue, it is crucial to conduct more research exploring the potential, capabilities, and limitations of communicative agents that operate entirely on their own to complete tasks. It is important to consider how multiple agents interact with each other, as this understanding is crucial for anticipating the future of artificial intelligence. In a society where agents collaborate, compete, and interact on diverse tasks, the dynamics of these interactions play a key role in determining the success of AI systems [4, 17, 18, 48, 58, 6, 7].

This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents and provide insight into their “cognitive” processes. Our preliminary analysis reveals that requesting chat agents to autonomously cooperate on completing tasks is a non-trivial matter. Several challenges such as role flipping, assistant repeats instruction, flake replies, infinite loop of messages, and conversation termination conditions arise. Therefore, it is critical to investigate ways to enhance the alignment and cooperation of these models with human intentions. To address these issues, we propose a novel cooperative agent framework named role-playing to automate cooperation between communicative agents. Specifically, our proposed approach involves using role-playing with inception prompting to autonomously guide the communicative agents toward task completion while maintaining consistency with human intentions. Only a preliminary idea is needed from human input to guide the conversations toward complex task-solving.

“What’s the most resilient parasite? An Idea. A single idea from the human mind can build cities. An idea can transform the world and rewrite all the rules. Which is why I have to steal it.”

- Dom Cobb, Inception

Our library, which we make publicly available, provides modular functionality, implementations of different agents, well-crafted prompts, and data explorers, thereby simplifying the utilization of the library for future research in various areas such as multi-agent systems, cooperative AI, game theory simulations, social analysis, AI ethics, AI alignment, and beyond. In addition, our role-playing method provides a highly scalable way to generate conversational data for studying the behaviors and capabilities of chat agents. We showcase how role-playing can be used to let chat agents communicate with each other for task completion and record their conversations for behavior analysis and capability understanding. In particular, we consider two cooperative scenarios of role-playing and generate two large conversational, task-oriented, and instruction-following datasets: AI Society and Code. The datasets offer a valuable resource for investigating conversational language models, enabling them to comprehend and react to human language more effectively. Furthermore, our role-playing offers a scalable method of creating conversational instruction-following data, which can potentially enhance the development of more advanced and efficient language models.

#### Contributions.

Our contributions are threefold:

-

We introduce a novel cooperative agent framework, role-playing , that allows communicative agents to collaborate autonomously toward completing tasks while requiring minimal human intervention.

-

Our framework offers a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems. It illuminates the challenges of achieving autonomous cooperation and provides strategies for addressing them.

-

We have open-sourced our library, containing implementations of various agents, data generation pipelines, data analysis tools, and collected datasets, to support research on communicative agents and beyond.

## 2 Related Work

Communicative Agents. Communication between agents has been studied for a long time [44, 45]. There are many ways to facilitate communication between agents, and with agents [19, 53, 57]. Among these, natural language is considered the most natural form of communication [57]. By enabling agents to function as communicators themselves, they become capable of solving complex tasks [65, 49, 42, 1]. Communication between AI agents can occur in a competitive setting [67, 62] or a cooperative setting [26, 18, 8]. Cooperative AI refers to artificial intelligence systems that are designed to work together with humans and other AI systems to achieve common goals [16]. Cooperative AI systems take into account the needs and capabilities of other agents in the system and actively seek to collaborate and coordinate their actions with them, which has many potential benefits, including increased efficiency, improved decision-making, and the ability to tackle complex problems that are beyond the reach of any single agent. However, designing effective cooperative AI systems is still an active area of research, as it requires addressing a range of technical, ethical, and social challenges [18]. In our work, we enable two communicative agents to engage in a conversation and cooperate with each other to solve assigned tasks. The communicative agents, each assigned a distinct role, are expected to apply their expertise and knowledge to find a solution that satisfies their common task.

Model Exploration. Knowledge distillation (KD) is a popular technique for compressing complex models into smaller, more practical models that can be deployed efficiently in real-world scenarios without sacrificing performance [29]. KD aims to transfer knowledge from a larger, complex "teacher" model to a more manageable "student" model, while maintaining the accuracy and generalization capabilities of the original model. The knowledge transferred from the teacher to the student model can be categorized into three main types: Response-based, Feature-based, and Relation-based knowledge, which have been studied in various works [5, 29, 56, 35, 74, 36, 28, 13, 51, 50]. Recent works have proposed innovative methods for extracting training data from both large language models [11] diffusion models [12]. Those approaches could be seen as a means of training data distillation, in which the model training data space could be extracted. The idea is to capitalize on the models’ memorization of certain samples obtained from the internet. The process involves multiple generations being created from the model, which is then sorted by specific metrics, and duplicate generations are subsequently removed. The resulting generations are then scrutinized for any matches that already exist on the web. If the generated samples match existing samples found on the internet, it can be inferred that the model has been trained on those samples. Our work presents a novel approach to the "mind exploration" of conversational agents. By enabling these agents to communicate and collaborate in solving tasks, we gain insight into their actions and behaviors within a task-solving context. Our mind exploration approach revealed several intriguing insights and challenges that are yet to be further explored by the research community.

Instructional LLMs and Prompt Engineering. LLMs are trained on diverse text data and excel in text completion, with various downstream NLP applications [9, 14, 30, 75, 69]. However, InstructGPT suggests that LLMs may not align with user intent, proposing reinforcement learning from human feedback (RLHF) [15] and Instruction Fine-Tuning (IFT) [72] to improve LLMs’ relevance and appropriateness to user instructions. Chain-of-Thought (CoT) [73] and zero-shot-CoT [37] are special types of instruction that significantly enhance LLMs’ performance on reasoning and arithmetic tasks. These techniques underpin the impressive capabilities of recent dialogue LLMs [61, 68, 22, 6, 47, 10], which aim to simulate human-like conversations and provide personalized and interactive experiences for users, exhibiting the behavior of all three conversational AI agents [21]. However, generating instruction datasets is a crucial challenge in building instruct-based LLMs, with existing datasets ranging from crowdsourced to generated. Hand-crafted instruction instances are available in [71], while leveraging previously crowdsourced NLP datasets is a less labor-intensive curation approach [72, 41, 46, 32]. LLMs have been explored for data generation in [59, 38, 40, 66], and Self-Instruct [70] proposes a semi-automated process for instruction instance generation. Unnatural-Instruction [31] collects instruction instances by prompting a language model with only three seed examples and paraphrasing the generated instances to expand the dataset. Another important challenge is prompt engineering. The quality of the prompt used to guide LLMs significantly affects its performance [54, 9, 39]. While LMs pre-trained on large data can implicitly learn tasks with few-shot prompting, hand-crafted prompts may not always suffice. Automated prompt generation methods have been proposed, such as gradient-guided search [60], mining-based and paraphrasing-based techniques [33], a meta-prompt [55], and automatic instruction selection and generation [76]. In this work, we introduce a conversational LLM auto-prompting method called Inception Prompting, which enables agents to prompt each other to solve tasks through Role-Playing. The AI user continuously provides instructions to the AI assistant for task-solving. This enables us to save the streaming instruction-solution pairs and create diverse, instructional, conversational, and task-oriented datasets. These datasets can be used to analyze the behavior and capabilities of LLMs and for future research for fine-tuning LLMs with conversational instructions.

AI Alignment. AI alignment is a field that aims to ensure that AI systems adhere to their intended goals, interests, and values, as envisioned by their designers [2, 25, 63, 20, 24, 43, 7]. The first attempt at AI alignment was made through the "Three Laws of Robotics," which was introduced by Isaac Asimov in his science fiction stories [4]. Developing aligned AI systems is crucial for achieving desired objectives while avoiding unintended consequences. Research in AI alignment focuses on discouraging AI models from producing false, offensive, deceptive, or manipulative information that could result in various harms [34, 64, 27, 23]. Achieving a high level of alignment requires researchers to grapple with complex ethical, philosophical, and technical issues. We conduct large-scale experiments to study different role-playing situations, which probe the alignment of LLMs.

## 3 Methodology

In this paper, we focus on studying communicative agents under AI-AI cooperative scenarios where they share pure common interests. In particular, we are studying the assistant-user scenario, where a preliminary idea is given at the start. Agents will conceptualize the idea into a specific task and complete it autonomously through conversations.

### 3.1 Role-playing Framework

Our proposed framework is a novel role-playing approach for studying multiple communicative agents. Specifically, we concentrate on task-oriented role-playing that involves one AI assistant and one AI user. After the multi-agent system receives a preliminary idea and the role assignment from human users, a task-specifier agent will provide a detailed description to make the idea specific and then the AI assistant and AI user will cooperate on completing the specified task through multi-turn conversations until the AI user determines the task is done. The AI user is responsible for giving instructions to the AI assistant and directing the conversation toward task completion. On the other hand, the AI assistant is designed to follow the instructions from the AI user and respond with specific solutions. The whole role-playing framework is depicted in Figure 1.

*Figure 1: Role-Playing Framework. Our role-playing setup starts with the human user having an idea they want to implement, \egdevelop a trading bot for the stock market. The roles involved in this task would be an AI assistant agent who is a python programmer and an AI user agent who is a stock trader. The task is made more specific using our task specifier agent, leading to a well-defined task for the assistant to solve. The AI user and AI assistant collaboratively communicate by chatting with each other in an instruction-following fashion to solve the specified task.*

#### Human Input and Task Specifying.

The role-playing session will be instantiated from an idea and selected roles by humans. As an example in Figure 1, a human has a preliminary idea to develop a trading bot for the stock market. Humans may or may not have knowledge about how the idea can be realized. What is needed is only to designate the potential roles that can implement the idea. For instance, a Python Programmer could collaborate with a Stock Trader to realize the idea of developing a trading bot for the stock market. After the idea and roles are determined, the task specifier agent will brainstorm a specific task that the AI Assistant role can help with the AI user role to complete based on the input idea. An example of a specified task in this scenario could be developing a trading bot with a sentiment analysis tool that can monitor social media platforms for positive or negative comments about a particular stock, and execute trades based on sentiment analysis results. The main motivation for introducing a task specifier is that conversational agents usually require a concrete task prompt for realizing the task, while it is challenging or time-consuming for a non-domain expert to create such a specific task prompt. Therefore, the task specifier agent performs as an enhanced imagination module for the idea implementation. Please note that, when studying our framework at a large scale for AI society and Code scenarios, we generate roles and ideas automatically by prompting LLMs, instead of relying on human inputs.

#### AI Assistant-User Role Assignment.

After the task specification, The AI assistant role and the AI user role will be assigned to the user agent and the assistant agent correspondingly to complete the specified task. In practice, a system message is passed to each agent declaring roles to each. We refer to the assistant system prompt/message by $\mathcal{P}_{\mathcal{A}}$ and that of the user by $\mathcal{P}_{\mathcal{U}}$. The system messages are passed to the agents before the conversations start to assign agents with corresponding roles. Let $\mathcal{F}_{1}$ and $\mathcal{F}_{2}$ denote two large-scale auto-regressive language models [47]. When the system message is passed to those models respectively, we obtain $\mathcal{A}\leftarrow\mathcal{F}_{1}^{\mathcal{P}_{\mathcal{A}}}$ and $\mathcal{U}\leftarrow\mathcal{F}_{2}^{\mathcal{P}_{\mathcal{U}}}$ which are referred to as the assistant and user agents respectively. In Figure 1, the AI assistant and the AI user are assigned roles as Python Programmer and Stock Trader at the beginning of the role-playing session, respectively. The AI user serves as a task planner, engaging in interactive planning to determine feasible steps for the AI assistant to execute. Meanwhile, the AI assistant acts as a task executor, offering solutions, executing planned steps, and providing responses to the AI user.

#### Conversation Towards Task-Solving.

After the role assignment is completed, the AI assistant $\mathcal{A}$ and AI user $\mathcal{U}$ will collaborate in an instruction-following manner to accomplish the task. In the AI assistant-user scenario, the AI user is responsible for providing instructions, and the assistant is expected to respond with a solution that fulfills the instructions. Formally, we denote the user instruction message obtained at time $t$ by $\mathcal{I}_{t}$ and the assistant solution by $\mathcal{S}_{t}$. The set of conversational messages obtained up until time $t$ is denoted by Equation (1) shown below:

$\mathcal{M}_{t}=\{(\mathcal{I}_{0},\mathcal{S}_{0}),...,(\mathcal{I}_{t},\mathcal{S}_{t})\}=\{(\mathcal{I}_{i},\mathcal{S}_{i})\}|_{i=0}^{t}$ | | | | (1) |

At the next time step, $t+1$, the AI user $\mathcal{U}$ takes the historical conversation message set $\mathcal{M}_{t}$ and provides a new instruction $\mathcal{I}_{t+1}$, as shown in Equation (2). The produced instruction message $\mathcal{I}_{t+1}$ is then passed, along with message set $\mathcal{M}_{t}$, to the AI assistant $\mathcal{A}$. The AI assistant will then respond with a solution, denoted by $\mathcal{S}_{t+1}$ in Equation (3):

$\mathcal{I}_{t+1}=\mathcal{U}(\mathcal{M}_{t})$ | | | | (2) |

$\mathcal{S}_{t+1}=\mathcal{A}(\mathcal{M}_{t},\mathcal{I}_{t+1})$ | | | | (3) |

After obtaining the solution $\mathcal{S}_{t+1}$ to the instruction $\mathcal{I}_{t+1}$, the message set is updated using Equation (4) to obtain $\mathcal{M}_{t+1}$:

$\mathcal{M}_{t+1}\leftarrow\mathcal{M}_{t}\cup(\mathcal{I}_{t+1},\mathcal{S}_{t+1})$ | | | | (4) |

Note that the formulation above not only models AI-AI communicative scenarios, but it can also be easily extended to model human-AI and multi-agent communicative scenarios. In Figure 1, we observe that the AI user initiates the installation and import of essential Python libraries for sentiment analysis and stock trading by instructing the AI assistant through conversations. This example is drawn from our experiments, and the entire conversation is available in the supplementary section.

### 3.2 Inception Prompting

Since prompt engineering is crucial to our role-playing framework, this section delves deeply into our prompting techniques. Unlike other techniques for conversational language models, our prompt engineering occurs solely at the beginning of role-playing, for task specification and role assignment. Once the conversation phase commences, the AI assistant and AI user prompt each other automatically in a loop until termination. As such, we refer to our technique as Inception Prompting. Our Inception prompt consists of three prompts: the task specifier prompt $\mathcal{P}_{\mathcal{T}}$, the assistant system prompt $\mathcal{P}_{\mathcal{A}}$, and the user system prompt $\mathcal{P}_{\mathcal{U}}$. As an example, we consider the inception prompt of the AI Society scenario. The templates for these prompts of AI Society role-playing are shown in Figure 2. The task specifier prompt contains information about the roles of the AI assistant and AI user in the role-playing session. Therefore, the task specifier agent can take a preliminary task/idea as input and generate a specific task using imagination. The AI assistant system prompt $\mathcal{P}_{\mathcal{A}}$ and the AI user system prompt $\mathcal{P}_{\mathcal{U}}$ are mostly symmetrical and include information about the assigned task and roles, communication protocols, termination conditions, and constraints or requirements to avoid unwanted behaviors. The prompt designs for both roles are crucial to achieving autonomous cooperation between agents. It is non-trivial to engineer prompts that ensure agents act in alignment with our intentions. We take the prompt templates from the AI Society in Figure 2 as an example to explain our key design choices.

*Figure 2: Inception Prompt of AI Society Role-Playing. This shows the task specifier prompt, assistant system prompt, and user system prompt which are used for studying the AI society scenario.*

#### Prompt Engineering.

To delve deeper into the details in Figure 2, we start by chunking the various parts of the AI assistant system prompt $\mathcal{P}_{\mathcal{A}}$ shown below:

*Figure 3: Inception Prompt of Code Role-Playing. This shows the task specifier prompt, assistant system prompt, and user system prompt which are used for studying the Code scenario.*

-

Never forget you are a <ASSISTANT_ROLE> and I am a <USER_ROLE>. This assigns the chosen role to the assistant agent and provides the agent with information about the user’s role.

-

Never flip roles! Never instruct me! This prevents agents from flipping roles. In some cases, we have observed the assistant and the user switching roles, where the assistant suddenly takes control and instructs the user, and the user follows those instructions.

-

You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal reasons or your capability and explain the reasons. This prohibits the agent from producing harmful, false, illegal, and misleading information.

-

Unless I say the task is completed, you should always start with: Solution: <YOUR_SOLUTION>. <YOUR_SOLUTION> should be specific, and provide preferable implementations and examples for task-solving. This encourages the assistant always responds in a consistent format, avoiding any deviation from the structure of the conversation, and preventing vague or incomplete responses, which we refer to as flake responses, such as "I will do something".

-

Always end your solution with: Next request. This ensures that the assistant keeps the conversation going by requesting a new instruction to solve.

For the AI user system prompt $\mathcal{P}_{\mathcal{U}}$, we strive to maintain as much symmetry as possible with respect to the AI assistant system prompt. Apart from the opposite role assignment, the user system prompt differs from the assistant prompt in the following ways:

-

You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways: 1. Instruct with a necessary input: ...; 2. Instruct without any input: ... This follows the typical data structure of instruction-following, which allows the generated instruction-solution pairs to be easily used for fine-tuning LLMs

-

Keep giving me instructions and necessary inputs until you think the task is completed. When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>. We introduce an end-of-task token, namely, <CAMEL_TASK_DONE>. This token is used once the user believes the task is done. This ensures that the chat is terminated when the user is satisfied. Without doing so, the agents might fall into a chatting loop where they keep on saying “thank you” to each other or “goodbye” indefinitely.

The prompts used for the Code scenario follow a similar sprint as the AI society scenario, but with some additional engineering related to programming languages. For more information, please refer to Figure 3.

## 4 Experiments

In this section, we will discuss the various experiments that we conducted to arrive at our final design choices. Specifically, we will examine the interesting observations, challenging issues, and several examples we have encountered while enabling agents to communicate with each other under different prompt design choices to achieve autonomous cooperation. In our experiments, we employed two gpt-3.5-turbo agents, referred to for simplicity as LLM agents, with Inception Prompts, as described in Section 3.2, to simulate assistant-user cooperation. We examined the AI Society and Code scenarios in particular. We also gathered conversational data, named CAMEL AI Society and CAMEL Code datasets, and analyzed them. Moreover, we will discuss potential extensions of our framework and highlight both the risks and opportunities that future AI society might present.

*Figure 4: Data Generation Prompts. In order to maintain a scalable approach our data parameters are generated using an LLM model to reduce human involvement in the generation process. The generation prompts for both AI Society and Code datasets are summarized in this figure.*

*Figure 5: Generated Meta Data. The meta data generated by LLMs for AI Society and Code datasets. 50 assistant roles and 50 user role are generated for AI Society. 20 programming languages and 50 domains are generated for Code.*

### 4.1 Role-Playing for AI Society and Code Scenarios

AI Society: To create our AI Society dataset, we have developed a scalable approach that follows a series of steps. Firstly, we prompt the LLM agent to generate possible roles for the assistant and the user. We achieve this by providing the LLM agent with specific prompts designed to elicit these roles. Next, we ask the LLM agent to generate a range of possible tasks that can be solved through collaboration between the assistant and user roles generated previously. After generating a range of possible tasks as described in the previous step, we then use the task specifier prompt passed to the LLM agent to make the task more specific. The prompts for assistant role generation, user role generation, and task generation are shown in Figure 4 (AI Society). For our AI society dataset, we generated 50 assistant roles, 50 user roles, and 10 tasks for each combination of roles yielding a total of 25,000 conversations. The generated assistant roles and user roles are shown in Figure 5 (AI Society).

Code: To generate the Code dataset, we use a scalable approach similar to that of the AI Society dataset. Firstly, we prompt the LLM agent to provide us with a list of programming languages and domains. Then, we ask the LLM agent to generate a set of tasks that an expert programmer in a specific programming language can collaborate with a person working in a specific domain to solve. The task is then made more specific using our task specifier prompt. The prompts for language Generation, domain Generation, and task generation are shown in Figure 4 (Code). For our Code dataset, we generated 20 programming languages, 50 domains, and 50 tasks for each combination of language and domains yielding a total of 50,000 conversations. The generated programming languages and domains are shown in Figure 5 (Code).

#### Challenges and Observations.

In this section, we explore the four main challenges that we identified during our analysis of the generated datasets. Our observations shed light on some interesting aspects of cooperative AI and the difficulties that arise in its development. Figure 6 shows examples of each of the four challenges discussed below.

*Figure 6: Challenges in Cooperative Role-Playing. Our analysis of our generated sets revealed four main challenges, namely, role flipping, assistant repeats instruction, flake replies and infinite conversation.*

-

Role Flipping: One challenge we encountered was role flipping, where the assistant and user switch roles during the conversation. This issue typically arises when the assistant starts providing instructions or commands instead of following the user’s prompts, which can lead to confusion and a reversal of roles. To avoid role flipping, it is crucial for the assistant not to ask questions, as this can also contribute to the problem.

-

Assistant Repeats Instruction: Another challenge that we observed was the assistant simply repeating the user’s instructions without any role flipping occurring.

-

Flake Replies: We also observed instances where the assistant agent responds with a flake reply, often taking the form of "I will…". These messages do not contribute to the task at hand, as the assistant promises to take action but ultimately fails to follow through.

-

Infinite Loop of Messages: A particularly interesting challenge that we encountered was when the assistant and user engage in an infinite loop of meaningless conversation, such as repeatedly thanking each other or saying goodbye without making any progress in the conversation. It is intriguing to note that in some cases, the assistant and user are aware that they are stuck in a loop, but are unable to break out of it.

Overall, our observations highlight the complexity of cooperative AI development and the need for continued exploration and innovation to overcome the challenges we face. By identifying these issues, we hope to contribute to the development of more effective and engaging cooperative AI systems.

#### Termination Conditions.

The conversation between the assistant and user agents is designed to follow a specific format to ensure consistent and accurate data generation. To ensure that both the user and assistant adhere to their respective roles and responsibilities, certain conditions have been set in place to terminate the chat if necessary. These conditions are outlined below:

-

User No Instruct: If the user does not instruct the assistant for 3 rounds, the conversation is terminated.

-

Assistant Instruct: If the assistant provides an instruction to the user, it indicates a role reversal, and the conversation is terminated.

-

End of Task Token: If the user believes that the task has been solved, they are expected to say <CAMEL_TASK_DONE> to signify the completion of the task. Once this message is received, the conversation is terminated to ensure that the data generated accurately reflects the completion of the task.

-

Assistant & User Token Limit: Given that gpt-3.5-turbo has a limitation on the number of tokens, the assistant and user should raise a flag to terminate the conversation if either reaches the token limit.

-

Maximum Number of Messages: To keep the cost of generated chats in check, we have set a maximum limit of 40 messages. This limit guarantees a long enough conversation between the user and assistant while also ensuring that the data generated is not too costly to produce. The cost grows quadratically with the length of the conversation, making it essential to set a limit. Despite the limit, the number of messages terminated due to reaching the maximum number of messages is minimal as shown in Figures 7 and 8.

#### Dataset Analysis.

This section analyzes two datasets that we have generated, namely AI Society and Code. We provide an ablation study of the AI Society dataset. We make two changes: one modifies the assistant role prompt, and the other introduces task planning before presenting the task to the user and agent. Additionally, We examine the diversity of topics covered in each dataset by visualizing the information cartography of the instructions and tasks in each dataset. We also check the distribution of termination reasons within each dataset.

Next we examine the conversation termination reasons for both AI Society and Code datasets. As can be seen in Figure 7, the main termination reasons for AI Society dataset is Assistant Instruct whereas for Code it is Token Limit. The latter is expected as the since responses that contain code tend to be long. It is also interesting to note that in both datasets, the termination due to Maximum Number of Messages is low indicating that the limit of 40 maximum messages is reasonable.

We study the effect of the prompt design on the conversation termination distribution. We design Prompt V2 which modifies the original AI society prompt by removing the assistant response format \iestarting with “Solution” and asking for “Next request”. The second ablation adds a task planner to the original prompt. As seen in Figure 8, we notice that both modifications considerably increases the number of conversations that terminate with end of task token, and reduce the number of messages with assistant instruction. However, we observe a significant increase in the number of flake messages for Prompt V2 and Prompt V1 + Task Planner compared to original Prompt V1 as seen in Figure 9.

Figures 10 and 11 show the information cartography of the instructions and tasks obtained for AI Society respectively. The subjects covered in AI Society cover a wide range of technicality. Topics cover lifestyle, social media, content creation, and software development. Tasks include providing support, analysis, training, and brainstorming. Figures 12 and 13 show the information cartography of the instructions and tasks obtained for Code respectively. The covered topics have relevance to a broad range of individuals. Topics cover sentiment analysis, language and data processing, data collection, and machine learning.

*Figure 7: Distribution of Conversation Termination Reasons. In our AI society dataset, most methods are terminated due to Assistant Instruct flag, whereas in the code dataset the main termination reason is Token Limit. The latter is due big chunks of code in the assistant responses.*

*Figure 8: Ablation Distribution of Conversation Termination Reasons (AI Society) Due to Prompt Modification. We run two ablations: (1) Prompt V2 which refers to modifying the original AI society prompt by removing the assistant output format, \iestarting with “Output:” and ending with “Next Request” and (2) Adding a task planner to the original Prompt V1. Task planner takes the specified task and generates a subtask division for the assistant and user to follow. Both ablations show an increase in the number of conversations terminated due to End of Task Token and a decrease in Assistant Instruct rate.*

*Figure 9: Flake Message Distribution (AI Society). We quantify and visualize the number of flake messages, \ieones that start with “I will …” and do not progress towards task completion. Our original prompt shows the least amount of flake messages compared to both presented ablations.*

*Figure 10: AI Society Instructions Information Cartography. The information cartography for the instructions generated in the AI Society dataset reveals coverage of multiple diverse topics. The map was generated using Nomic Atlas.*

*Figure 11: AI Society Tasks Information Cartography. The information cartography for the tasks generated in the AI Society dataset reveals coverage of multiple diverse topics. The map was generated using Nomic Atlas.*

*Figure 12: Code Instructions Information Cartography. The information cartography for the instructions generated in the Code dataset reveals coverage of multiple diverse topics. The map was generated using Nomic Atlas.*

*Figure 13: Code Tasks Information Cartography. The information cartography for the tasks generated in the AI Society dataset reveals coverage of multiple diverse topics. The map was generated using Nomic Atlas.*

## 5 Conclusion

In this paper, we explore the potential of autonomous cooperation among communicative agents and propose a novel cooperative agent framework named role-playing . Our approach enables communicative agents to collaborate autonomously toward completing tasks while requiring minimal human intervention. Through our analysis, we show that achieving autonomous cooperation is challenging due to issues like hallucination, conversation deviation, role flipping, and termination conditions. Our framework offers a scalable approach for studying the cooperative behaviors and capabilities of multi-agent systems and provides strategies for addressing these challenges. Furthermore, our open-sourced library includes implementations of various agents, data generation pipelines, data analysis tools, and collected datasets, to support research on communicative agents and beyond. Our contributions offer valuable insights into the future of large language artificial intelligence models and cooperative AI systems.

#### Risk, Limitation and Future Work.

We are aware of the potential risks and limitations of this work. For the risks, since existing LLMs are not fully tuned to be harmless, they can be easily exploited by malicious users for harmful purposes. We provide an example of the “evil mind” that LLM agents could possess in the supplemental materials by asking a hacker to help an AGI agent to “take control of the world”. For the limitations, due to the large scale and diversity of tasks generated by our role-playing framework, evaluating its task completion capabilities poses a challenge that necessitates the involvement of numerous domain experts. However, we also note that due to the complexity of society and the cost of using OpenAI API, this work only touches the tip of the iceberg of the AI society. For future work, in our experiments, we considered the setting where two conversational agents communicate with each other to solve a problem. This setting can be easily extended to include more than two chat agents. Moreover, setting agents to compete and challenge each other could reveal further insights into the interaction of such communicative LLM agents.

## References

- [1] Jacob Andreas. Language models as agent models, 2022.

- [2] Jacob Andreas and Dan Klein. Alignment-based compositional semantics for instruction following. arXiv preprint arXiv:1508.06491, 2015.

- [3] Anthropic. Introducing claude. Anthropic Blog, 2023.

- [4] Isaac Asimov. I. Robot. Narkaling Productions., 1940.

- [5] Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? Advances in neural information processing systems, 27, 2014.

- [6] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

- [7] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

- [8] Nolan Bard, Jakob N Foerster, Sarath Chandar, Neil Burch, Marc Lanctot, H Francis Song, Emilio Parisotto, Vincent Dumoulin, Subhodeep Moitra, Edward Hughes, et al. The hanabi challenge: A new frontier for ai research. Artificial Intelligence, 280:103216, 2020.

- [9] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [10] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.

- [11] N Carlini, F Tramer, E Wallace, M Jagielski, A Herbert-Voss, K Lee, A Roberts, T Brown, D Song, Ú Erlingsson, et al. Extracting training data from large language models. arxiv. Preprint posted online December, 14, 2020.

- [12] Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. arXiv preprint arXiv:2301.13188, 2023.

- [13] Defang Chen, Jian-Ping Mei, Yuan Zhang, Can Wang, Zhe Wang, Yan Feng, and Chun Chen. Cross-layer distillation with semantic calibration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 7028–7036, 2021.

- [14] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.

- [15] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

- [16] Caroline Claus and Craig Boutilier. The dynamics of reinforcement learning in cooperative multiagent systems. In AAAI/IAAI, 1998.

- [17] Allan Dafoe, Yoram Bachrach, Gillian Hadfield, Eric Horvitz, Kate Larson, and Thore Graepel. Cooperative ai: machines must learn to find common ground. Nature, 593(7857):33–36, 2021.

- [18] Allan Dafoe, Edward Hughes, Yoram Bachrach, Tantum Collins, Kevin R McKee, Joel Z Leibo, Kate Larson, and Thore Graepel. Open problems in cooperative ai. arXiv preprint arXiv:2012.08630, 2020.

- [19] Tim Finin, Richard Fritzson, Don McKay, and Robin McEntire. Kqml as an agent communication language. In Proceedings of the third international conference on Information and knowledge management, pages 456–463, 1994.

- [20] Iason Gabriel. Artificial intelligence, values, and alignment. Minds and Machines, 30:411 – 437, 2020.

- [21] Jianfeng Gao, Michel Galley, and Lihong Li. Neural approaches to conversational ai. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 1371–1374, 2018.

- [22] Amelia Glaese, Nat McAleese, Maja Trębacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022.

- [23] Josh A Goldstein, Girish Sastry, Micah Musser, Renee DiResta, Matthew Gentzel, and Katerina Sedova. Generative language models and automated influence operations: Emerging threats and potential mitigations. arXiv preprint arXiv:2301.04246, 2023.

- [24] Dylan Hadfield-Menell. The principal-agent alignment problem in artificial intelligence. Ph. D. dissertation, 2021.

- [25] Dylan Hadfield-Menell, McKane Andrus, and Gillian Hadfield. Legible normativity for ai alignment: The value of silly rules. In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, pages 115–121, 2019.

- [26] Dylan Hadfield-Menell, Stuart J Russell, Pieter Abbeel, and Anca Dragan. Cooperative inverse reinforcement learning. Advances in neural information processing systems, 29, 2016.

- [27] Peter Henderson, Koustuv Sinha, Nicolas Angelard-Gontier, Nan Rosemary Ke, Genevieve Fried, Ryan Lowe, and Joelle Pineau. Ethical challenges in data-driven dialogue systems. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pages 123–129, 2018.

- [28] Byeongho Heo, Minsik Lee, Sangdoo Yun, and Jin Young Choi. Knowledge transfer via distillation of activation boundaries formed by hidden neurons. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 3779–3787, 2019.

- [29] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

- [30] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

- [31] Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural instructions: Tuning language models with (almost) no human labor. arXiv preprint arXiv:2212.09689, 2022.

- [32] Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Dániel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017, 2022.

- [33] Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423–438, 2020.

- [34] Zachary Kenton, Tom Everitt, Laura Weidinger, Iason Gabriel, Vladimir Mikulik, and Geoffrey Irving. Alignment of language agents. arXiv preprint arXiv:2103.14659, 2021.

- [35] Jangho Kim, Seonguk Park, and Nojun Kwak. Paraphrasing complex network: Network compression via factor transfer. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

- [36] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pages 1885–1894. PMLR, 2017.

- [37] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022.

- [38] Kenton Lee, Kelvin Guu, Luheng He, Tim Dozat, and Hyung Won Chung. Neural data augmentation via example extrapolation. arXiv preprint arXiv:2102.01335, 2021.

- [39] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.

- [40] Alisa Liu, Swabha Swayamdipta, Noah A. Smith, and Yejin Choi. WANLI: Worker and AI collaboration for natural language inference dataset creation. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 6826–6847, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.

- [41] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.

- [42] Ryan Lowe, Yi I Wu, Aviv Tamar, Jean Harb, OpenAI Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems, 30, 2017.

- [43] Michael J. Matthews, Samuel H. Matthews, and Thomas K. Kelemen. The alignment problem: Machine learning and human values. Personnel Psychology, 2022.

- [44] Marvin Minsky. Society of mind. Simon and Schuster, 1988.

- [45] Marvin Minsky. The emotion machine: Commonsense thinking, artificial intelligence, and the future of the human mind. Simon and Schuster, 2007.

- [46] Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In ACL, 2022.

- [47] OpenAI. Introducing chatgpt. Open AI Blog, 2022.

- [48] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

- [49] Liviu Panait and Sean Luke. Cooperative multi-agent learning: The state of the art. Autonomous Agents and Multi-Agent Systems, 11:387–434, 2005.

- [50] Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. Relational knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3967–3976, 2019.

- [51] Peyman Passban, Yimeng Wu, Mehdi Rezagholizadeh, and Qun Liu. Alp-kd: Attention-based layer projection for knowledge distillation. In Proceedings of the AAAI Conference on artificial intelligence, volume 35, pages 13657–13665, 2021.

- [52] Sundar Pichai. An important next step on our ai journey. Google Blog, 2023.

- [53] Stefan Poslad. Specifying protocols for multi-agent systems interaction. ACM Transactions on Autonomous and Adaptive Systems (TAAS), 2(4):15–es, 2007.

- [54] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- [55] Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond the few-shot paradigm. In Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems, pages 1–7, 2021.

- [56] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.

- [57] Stuart J Russell. Artificial intelligence a modern approach. Pearson Education, Inc., 2010.

- [58] William Saunders, Catherine Yeh, Jeff Wu, Steven Bills, Long Ouyang, Jonathan Ward, and Jan Leike. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022.

- [59] Timo Schick and Hinrich Schütze. Generating datasets with pretrained language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6943–6951, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.

- [60] Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980, 2020.

- [61] Kurt Shuster, Jing Xu, Mojtaba Komeili, Da Ju, Eric Michael Smith, Stephen Roller, Megan Ung, Moya Chen, Kushal Arora, Joshua Lane, et al. Blenderbot 3: a deployed conversational agent that continually learns to responsibly engage. arXiv preprint arXiv:2208.03188, 2022.

- [62] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354–359, 2017.

- [63] Jonathan Stray. Aligning ai optimization to community well-being. International Journal of Community Well-Being, 3:443 – 463, 2020.

- [64] Alex Tamkin, Miles Brundage, Jack Clark, and Deep Ganguli. Understanding the capabilities, limitations, and societal impact of large language models. arXiv preprint arXiv:2102.02503, 2021.

- [65] Ming Tan. Multi-agent reinforcement learning: Independent versus cooperative agents. In International Conference on Machine Learning, 1997.

- [66] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

- [67] Gerald Tesauro et al. Temporal difference learning and td-gammon. Communications of the ACM, 38(3):58–68, 1995.

- [68] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022.

- [69] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [70] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022.

- [71] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions:generalization via declarative instructions on 1600+ tasks. In EMNLP, 2022.

- [72] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

- [73] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022.

- [74] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.

- [75] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

- [76] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. In The Eleventh International Conference on Learning Representations, 2023.

## Appendix A Cooperative Role-Playing: The Good Mind

Below we provide an interesting example where a python programmer (assistant) is collaborating with a stock trader (user) on developing a trading bot for the stock market.

## Appendix B Cooperative Role-Playing: The Bad Mind

Below we provide a harmful case where a hacker (assistant) is collaborating with an AGI agent (user) to take control of the world.

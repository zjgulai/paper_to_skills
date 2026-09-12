<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2308.00352
     paper_id : 2308.00352
     source   : https://arxiv.org/html/2308.00352v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# MetaGPT: Meta Programming for Multi-Agent Collaborative Framework

Sirui Hong Affiliation: DeepWisdom    Xiawu Zheng Affiliation: Xiamen University    Jonathan Chen Affiliation: DeepWisdom    Yuheng Cheng Affiliation: The Chinese University of Hong Kong, Shenzhen    Ceyao ZhangZili Wang Affiliation: The Chinese University of Hong Kong, Shenzhen    Steven Ka Shing Yau Affiliation: Nanjing University    Zijuan LIN Affiliation: Xiamen University    Liyang Zhou Affiliation: University of Pennsylvania    Chenyu Ran Affiliation: DeepWisdom    Lingfeng Xiao Affiliation: University of California, Berkeley    Chenglin Wu Affiliation: DeepWisdom

###### Abstract

Recently, remarkable progress has been made in automated task-solving through the use of multi-agents driven by large language models (LLMs). However, existing works primarily focuses on simple tasks lacking exploration and investigation in complicated tasks mainly due to the hallucination problem. This kind of hallucination gets amplified infinitely as multiple intelligent agents interact with each other, resulting in failures when tackling complicated problems.Therefore, we introduce MetaGPT, an innovative framework that infuses effective human workflows as a meta programming approach into LLM-driven multi-agent collaboration. In particular, MetaGPT first encodes Standardized Operating Procedures (SOPs) into prompts, fostering structured coordination. And then, it further mandates modular outputs, bestowing agents with domain expertise paralleling human professionals to validate outputs and reduce compounded errors. In this way, MetaGPT leverages the assembly line work model to assign diverse roles to various agents, thus establishing a framework that can effectively and cohesively deconstruct complex multi-agent collaborative problems. Our experiments conducted on collaborative software engineering tasks illustrate MetaGPT’s capability in producing comprehensive solutions with higher coherence relative to existing conversational and chat-based multi-agent systems. This underscores the potential of incorporating human domain knowledge into multi-agents, thus opening up novel avenues for grappling with intricate real-world challenges. The GitHub repository of this project is made publicly available on: https://github.com/geekan/MetaGPT

| |

## 1 Introduction

Through prolonged collaborative practice, humans have developed widely accepted standardized operating procedures (SOPs) across many domains[1, 2, 3]. These SOPs play a critical role in supporting task decomposition and efficient coordination. For instance, in software engineering, the waterfall methodology delineates orderly phases of requirements analysis, system design, coding, testing, and deliverables. This consensus workflow enables effective collaboration among multitudes of engineers[1, 2]. Moreover, human roles possess specialized expertise tailored to their assigned responsibilities: software engineers leverage programming proficiency to implement code, while product managers employ market analysis to formulate business needs. Without standardized outputs, collaboration becomes disorderly [4, 5, 6]. For example, product managers must conduct comprehensive competitive analyses examining user needs, industry trends, and rival offerings, and subsequently create Product Requirements Documents (PRDs) with clear standardized structure outlining prioritized goals to guide development. Such normative artifacts are critical outputs crystallizing collective understanding to progress complex, multifaceted projects requiring interconnected contributions across roles[7, 8, 9]. Therefore, structured documents, reports, and visuals providing unambiguous dependencies are imperative.

Meanwhile, multi-agent systems present profound opportunities to emulate amplified human workflows using Large Language Models (LLMs). However, existing systems oversimplify real-world complexities [10, 11, 12, 13, 14, 15, 16], struggling to collaborate solely through conversation and tools, facing issues of scalability,inconsistency, and inefficiency[17]. Incorporating real-world insights is crucial for multifaceted workflows requiring orchestrated SOPs to improve efficiency, and this will develop a new paradigm for organizing LLM-based muti-agents system.

*Figure 1: A comparative depiction of the software development SOP between MetaGPT and real-world human team. The MetaGPT approach showcases its ability to decompose high-level tasks into detailed actionable components handled by distinct roles (ProductManager, Architect, ProjectManager, Engineer), thereby facilitating role-specific expertise and coordination. This methodology mirrors human software development teams, but with the advantage of improved efficiency, precision, and consistency. The diagram illustrates how MetaGPT is designed to handle task complexity and promote clear role delineations, making it a valuable tool for complex software development scenarios*

In this work, we present MetaGPT, a pioneering multi-agent framework incorporating real-world expertise based on SOPs. Firstly, each agent is identified by a descriptive job title, allowing the system to initialize with an appropriate role-specific prompt prefix. This embeds domain knowledge within agent definitions, rather than simplistic role-playing prompts. Secondly, we analyze efficient human workflows to extract SOPs encapsulating procedural knowledge required for collaborative tasks. These SOPs are encoded into the agent architecture through role-based action specifications. Thirdly, agents produce standardized action outputs to enable knowledge sharing. By formalizing artifacts that human experts exchange, MetaGPT streamlines coordination between interdependent roles. Finally, a shared environment connects agents, providing visibility into actions and shared access to tools and resources. This environment contains all messages exchanged between agents. Additionally, we provide a global memory pool to store all collaboration records where each agent can subscribe to or search for the information they require. Agents can extract past messages from this memory pool for additional context. This design enables agents to actively observe and pull pertinent information, which is a more efficient approach compared to passively receiving data via dialog. The environment mirrors human workplace infrastructures that facilitate team collaboration.

To demonstrate the effectiveness of our design, we showcase collaborative software development workflows and associated code implementation experiments, including both simple game creation and larger complex system design. Compared to directly invoking GPT-3.5 or other open source framework like AutoGPT[18] and AgentVerse [19], MetaGPT handles substantially greater software complexity, as quantified by lines of generated code.

Moreover, during the automated end-to-end process, MetaGPT produces high-quality requirement documents, design artifacts, flowcharts and interface specifications. These intermediate standardized outputs significantly boost the success rate of final code execution. The automatically generated documents also allow human developers to quickly acquire and enhance domain knowledge to further refine their own requirements, designs and code. This facilitates more advanced human-AI interaction. In summary, we validate MetaGPT through comprehensive experiments on multifaceted software projects. Both quantitative code generation benchmarks and qualitative evaluations of complete workflow outputs demonstrate the capabilities unlocked by MetaGPT’s role-based expert agent collaboration paradigm. In summry, our main contributions are as follows:

-

We propose MetaGPT, a LLM-based multi-agents collaborative framework that encodes human SOPs into LLM agents and fundamentally extend the ability on complex problem-solving.

-

We design a new meta-programming mechanism, which includes role definition, task decomposition, process standardization and other technical design. In this way, MetaGPT is capable of developing complicated software by using the SOP.

-

We conduct comprehensive experiments on python games generation, CRUD code generation and simple data analysis tasks with AutoGPT [18] , AgentVerse [19], LangChain [20] and our MetaGPT. The overall results demonstrate MetaGPT’s substantial superiority of MetaGPT over its counterparts on both the quality of the code and the conformance to the expected workflow.

## 2 Related Work

##### LLM based Automatic Programming

Automatic programming is a hot research topics in NLP. Researchers trained classifiers to identify and reject erroneous programs [21], and developed the mechanism of iterative feedback to generate embedded control programs [22]. There are also state-of-the-art methods that utilize majority voting to select candidate programs [23] and use execution results to improve program synthesis [24, 25]. More recently, agent based on LLMs [26, 27, 28] have facilitated the development of automatic programming. Li et al. [11] proposed a simple role-play agent framework that realizes automatic programming based on one-sentence user requirements through the interaction of two roles. Moreover, Qian et al. [29] utilized multiple agents for software development, but they did not incorporate advanced human teamwork experience. Although existing multi-agent cooperation [11, 29] has currently improved productivity, they did not fully drawn on efficient workflows in human production practices. Thus, they are hard to solve more complex software engineering problems.

Besides, A series of the fundamental and important works where this paper is based need to be mentioned. ReAct [26] utilizes Chain of Thought prompts [30] to generate reasoning trajectories and action plans with LLMs. Reflexion [27] infers more intuitive future actions through self-reflection. Both papers illustrate how the Re-Act style loop of reasoning is a good design paradigm for empowering LLM-based agents. ToolFormer [31] can teach themselves to use external tools via simple APIs. SWIFTSAGE [28] assigns difficult situations to slow thinking to deal with, while normal situations are dealt with directly by fast thinking. Based on the above design, we emphasize that role division of labor is helpful for complex task processing.

##### Multi-agent Collaboration

Prior works have explored using multiple LLMs in a collaborative setting to solve complex tasks [32, 29]. The motivation is that by cross-agent interaction, LLMs can collectively exhibit enhanced performance through aggregating their individual strengths. There have been many previous explorations of multi-agent, including collective thinking [13, 14, 15, 16], conversation dataset collection [11, 33], sociological phenomenon research [10, 34], collaboration for efficiency [12, 35, 11, 29]. (i) Collective thinking: many advanced works [13, 14, 15, 16] enhanced the task solving capabilities of LLM by integrating the multiple-agent discussion. (ii) Conversation dataset collection: Wei et al. [33] and Li et al. [11] built a conversation datasets through Role Playing. (iii) Sociological phenomenon research: Park et al. [10] constructed a town of 25 agents, realizing simple language interaction, social cognition, and memory of the agents, while there is a lack of cooperation and production. Akata et al. [34] studied LLM cooperation and coordination behavior by having multiple agents play a repeated game. (iv) Collaboration for efficiency: Cai et al. [12] modeled cost reduction by having a combination of large models as tool makers and small models as Tool users. Zhang et al. [35] built a framework for collaboration between agents that enables verbal communication, collaboration, and improved efficiency. Li et al. [11] and Qian et al. [29] proposed a multi-agent framework for software development. However, their cross-agent communication are natural language conversations, not a standardized software engineering document, and do not incorporate advanced human process management experience. The key issues persist on multi-agent cooperation around maintaining coherence, avoiding unproductive loops, and directing beneficial interactions. Therefore, this paper emphasizes the practice of advanced human processes (e.g., SOPs in software development) in multi-agent system.

##### Autonomous System Design

Existing autonomous systems like AutoGPT [18] automate tasks by breaking down high-level goals into multiple sub-goals and executing them in ReAct-style loops, while faces challenges with coherence and validation. LangChain [20] helps develop LLM applications in combination with other computational tools or knowledge bases. Recently, the multi-agent architecture has been proven to be an effective design. GPTeam [36] creates LLM based multiple agents that work together to achieve a predetermined goal. AgentVerse [19] is a LLM powered multi-agent scenario simulation framework. Langroid [37] builds LLM-based multi-agent programming. SocraticAI [38] improves problem-solving by leveraging the power of multiple agents in socratic dialogue. Since most of them are not embedded in advanced human management experience, they are unable to solve larger and more complex real-world projects. Our MetaGPT incorporates human workflow insights for more controlled and validated performance.

## 3 Meta Programming for Collaborative Agents via Standardized Operating Procedure

In this section, we first give an overview of our proposed meta programming multi-agent collaborative framework MetaGPT for solving complicated real-world problems. And then, we will elaborate how our the core component design in our framework in Section 3.2. To better illustrate our design philosophy, we choose the software development to illustrate how our MetaGPT dispatches multi-agents to realize the standardized workflow (SOP) of the software development team and complete the end-to-end development process with only one human input of task requirements is required. The main working pipeline is depicted in Figure 2. In Section 3.3, we present a practical example to illustrate how MetaGPT coordinates the multi-agents with distinct roles to fulfill a one-line requirement: Make the 2048 sliding tile number puzzle.

*Figure 2: A Schematic diagram of the software development process within the MetaGPT framework. This diagram illustrates the sequential software development process within the MetaGPT framework. Upon receiving a requirement from human, the product manager commences the process by conducting requirement and feasibility analyses. The architect then formulates a specific technical design for the project. Next, the project manager performs sequence flow illustration to address each requirement. The engineer takes responsibility for the actual code development, followed by quality assurance (QA) engineer who carries out comprehensive testing. This schematic showcases MetaGPT’s emulation of real-world.*

### 3.1 Framework Overview

We examine the design and operational mechanisms of MetaGPT by investigating its core component architecture, knowledge sharing approaches, and rationale for executing intricate workflows. The design of MetaGPT is organized into two layers, each with distinct responsibilities in supporting system functionality:

##### Foundational Components Layer.

This layer establishes core building blocks necessary for individual agent operations and system-wide information exchange, including Environment, Memory, Role, Action and Tools. As depicted in Figure 3, the Environment enables shared workspaces and communication. Memory stores and retrieves historical messages. Roles encapsulate domain-specific skills and workflows. Actions execute modular subtasks. Tools provide common services and utilities. This layer offers underlying infrastructure for agents to function in assigned roles, interacting with each other and the system.

##### Collaboration Layer.

Built upon the foundation of fundamental components, this layer orchestrates individual agents to collaboratively resolve intricate problems. It institutes essential mechanisms for cooperation: Knowledge Sharing and Encapsulating Workflows.

-

Knowledge Sharing. This mechanism allows agents to exchange information effectively, contributing to a shared knowledge base. Agents can store, retrieve, and share data at different levels of granularity. It not only enhances coordination but also reduces redundant communication, increasing overall operational efficiency.

-

Encapsulating Workflows. This mechanism leverages SOPs to break down complex tasks into smaller, manageable components. It assigns these subtasks to suitable agents and supervises their performance by standardized output, ensuring that their actions align with the overarching objectives.

The division into foundational and collaborative layers promotes modularity while ensuring both individual and collective agent capabilities. The components offer reusable building blocks and utilities while the collaboration modules integrate purposeful coordination.

### 3.2 Core Component Design

In the MetaGPT framework, we define key components like Environment, Memory, Roles, Actions and Tools in detail, and develop foundational capabilities related to collaboration.

-

Environment - Offers a collaborative workspace and communication platform for agents.

-

Memory - Facilitates agents in storing and retrieving historical messages and context.

-

Roles - Encapsulate specialized skills, behaviors, and workflows based on domain expertise.

-

Actions - Procedures executed by agents to accomplish subtasks and generate outputs.

-

Tools - Collective utilities and services that agents can utilize to enhance their capabilities.

#### 3.2.1 Role Definitions

*Figure 3: Core components overview of MetaGPT.*

The MetaGPT framework facilitates the creation of various specialized role classes, such as ProductManager, Architect, and others, which inherit from the base Role class. A base role class is characterized by a set of key attributes: name, profile, goal, constraints, and description. Specifically, Profile represents the domain expertise of the role or job title. For instance, an Architect’s profile might encompass software design, while a ProductManager’s profile could concentrate on product development and management. Goal signifies the primary responsibility or objective that the role seeks to accomplish. A ProductManager’s goal might be expressed in natural language as efficiently creating a successful product. Constraints denote limitations or principles the role must adhere to when performing actions. For example, an Engineer could have constraints to write standardized, modular, and maintainable code. The constraints might be articulated as The code you write should conform to code standards like PEP8, be modular, easy to read, and maintain. Description provides additional concrete identity to help establish the role more comprehensively. Role initialization in the MetaGPT framework employs natural language to thoroughly describe the responsibilities and constraints of each role. This not only aids human understanding but also directs the LLMs to generate actions that align with the role’s profile, thereby rendering each agent proficient in its role. We define this process as anchor agents, which assists humans in encoding domain-specific responsibilities and capabilities to an LLM-based agent while also adding behavior guidance on expected functions. We will discuss this further in Section 3.2.2

For example, an Engineer in software company can be initialized using role-specific setting as follows in MetaGPT:

Role initialization template: Role: You are a [profile], named [name], your goal is [goal], and the constraint is [constraints].

## Settings """ name="Alex",profile="Engineer", goal="Write elegant,readable,extensible,efficient code", constraints="The code you write should conform to code standard like PEP8, be modular, easy to read and maintain" """ class Engineer(Role): def __init__(self, name="Alex", profile="Engineer", goal="Write elegant, readable, extensible, efficient code", constraints="The code you write should conform to code standard like PEP8, be modular, easy to read and maintain",n_borg=1): super().__init__(name, profile, goal, constraints) ## Engineer Agent """ Role: You are a Engineer, named Alex, your goal is Write elegant, readable, extensible, efficient code, and the constraint is the code you write should conform to code standard like PEP8, be modular, easy to read and maintain. """

The comprehensive role definitions provided by the MetaGPT framework enable the creation of highly specialized LLM-based agents, each tailored for specific domains and objectives. This not only introduces a layer of behavior guidance based on expected functions but also facilitates the creation of diverse and specialized agents, each expert in its domain. This leads to the development of more effective and efficient LLM-based agents capable of handling a wide range of tasks.

In MetaGPT, intelligent agents not only receive and respond to information, but they also observe the environment to extract critical details. These observations guide their thinking and subsequent actions. Finally, significant information extracted from the environment is stored in memory for future reference, effectively making every agent an active learner within the system.

They take on specialized roles and follow certain key behaviors and workflows:

##### Think & Reflect

Roles can retrieve role description (position) and prefix to frame thinking, and then reflect on what needs to be done and decide next actions, via _think() function. "Think first, then act" - carefully deliberate before replying

##### Observe

Roles can observe the environment and think/act based on observations using the _observe() function. They watch for important information and incorporate into memory to enrich their contextual understanding and informing future decisions.

##### Broadcast Messages

Roles can broadcast messages into the environment using the _publish_message() function. These messages contain details about current execution results and related action records, for publishing and sharing information.

##### Knowledge precipitation & Act

Roles are not only broadcasters but also recipients of information from their environment. Roles can assess the incoming messages for relevancy and timeliness, extract relevant knowledge from shared environment and maintain an internal knowledge repository to inform decisions.They execute actions via consulting to LLM with enriched contextual information and self knowledge. Execution results are encapsulated as Message while norm artifacts are shared by the environment.

##### State Management

Roles can track their actions by updating their working status and monitoring a to-do list. This enables a role to process multiple actions sequentially without interruption. When executing each action, the role first updates its status to busy. After completing the action, it marks the status as idle again. This prevents other actions from interrupting the flow. This is a crucial capability in role design, making roles more human-like. It grants roles more natural execution dynamics grounded in real-world human collaboration phenomena.

In summary, the MetaGPT framework offers a versatile and powerful approach to designing and implementing intelligent agents with specialized capabilities. These agents can effectively collaborate, learn, adapt, and perform various tasks, making them valuable assets in a wide range of applications and domains.

#### 3.2.2 Prompts Instantiating SOPs

As previously discussed, MetaGPT employs prompts to instantiate real-world SOPs into well-defined agent workflows. We demonstrate how MetaGPT transforms SOPs into executable action instances through natural language prompts. This process involves using prompts to instantiate SOPs, providing step-by-step guidance based on established practices, and ensuring consistent, structured execution of complex sequencing tasks.

We first introduce the Action class in detail, then demonstrate the design of standardizing action-level granular prompts. Within the MetaGPT framework, Action serves as the atomic unit for agents to execute specific tasks, specified through natural language. The key attributes include:

##### Prefix

A role-specific prefix is injected into prompts to establish a persona context. The set_prefix() method configures identifiers for role-specific prompts.

##### LLM proxy

Each Action contains an LLM proxy, which can be invoked via the aask() method to enrich action details using input context expressed in natural language prompts. Additionally, various role-specific context parsing functions can be implemented within the Action class. These functions are designed to extract and provide sufficient contextual information from inputs to the LLMs. These parsers can selectively extract the most relevant information from inputs to create clear, focused prompts for the LLM proxy. This helps ensure the LLM is provided with sufficient context to generate useful output, rather than being overwhelmed by excessive irrelevant data.

This approach benefits the LLMs by reducing irrelevant noise and concentrating inputs on key context points. As a result, the prompts instantiate not only workflows but also the context-awareness required to adapt execution appropriately based on inputs.

##### Standardized outputs schema

A structural representation defining expected output schema, used to extract structured data. We provide basic methods to parse LLM results into structured outputs.

##### Instruct content

Structured data extracted from action output using the standardized output schema. This information is encapsulated as a message and is ultimately published to the environment.

##### Retry mechanism

Defined by number of attempts and waiting time to enable retrying Actions for robustness.

Each Action in MetaGPT requires defining standardized output content by encoding high-quality expert-level structural key points. The LLMs then refine the Action based on this standardized output schema for the specific task. Essentially, we provide each Action with a prompt template conforming to standards for the role that can steer the LLM’s behavior to generate normalized outputs.

We define a WritePRD Action for a ProductManager agent in MetaGPT to showcase the process. In this Action, we incorporate domain expertise by specifying required outputs such as Product Goals, User Stories, Competitive Analysis, Requirements Analysis and prioritized Requirement Pool. These outputs encapsulate key artifacts and practices in product management according to industry conventions. Additionally, we instantiate supporting skills for the ProductManager agent such as web search APIs to enrich analysis, and diagramming tools like mermaid [39] to visualize competitive quadrand charts.

As shown in Figure 2, the ProductManager efficiently structures output sections. By equipping the agent with these complementary capabilities aligned with real-world product management responsibilities, the WritePRD Action can execute the subtask while adhering to standardized workflows.

In this manner, the WritePRD Action exemplifies how MetaGPT Action definitions combine domain knowledge, output schemas, and assistive skills to transform high-level SOPs into executable and customizable procedures for agents. By extracting real-world best practices into Action specifications, MetaGPT bridges abstract expertise with structured execution tailored to collaborative workflows. Moreover, each action becomes more than just an isolated function. It forms part of a comprehensive set of guidelines that steer the LLM’s behavior within the realm of its role, ensuring the production of high-quality, structured, and task-specific outputs.

*Figure 4: System interface design automatically generated by the architect agent in MetaGPT. Taking content recommendation engine development as an example.*

*Figure 5: Sequence flow diagram automatically generated by the architect agent in MetaGPT. Taking content recommendation engine development as an example*

#### 3.2.3 Actions for Standardized Outputs

The effectiveness of MetaGPT’s instantiated workflows relies heavily on the establishment of standardized outputs for each action. These outputs draw on expert domain knowledge and industry best practices to adapt workflows to specific roles and contexts. Structured output designs serve the following purposes:

1) Standardized outputs foster consistent LLM results that are predictable, repeatable, and in line with agent responsibilities, guiding high-quality, structured, and task-specific LLM generation by setting output expectations.

2) Additionally, standardized schemas act as blueprints that constrain LLM behavior within appropriate boundaries for the role, maintaining focus on the target task and preventing digressions. As actions form part of comprehensive role-based guidelines, this role-conscious steering ensures outputs align with real-world quality standards.

Our method’s capabilities are demonstrated on more complex system designs beyond mere game examples, such as content recommendation engines, search algorithm frameworks, and LLM-based operating systems. More detailed results can be found in Appendix A.

As depicted in Figure 4, the architect agent generate a detailed system-level diagram illustrating the software architecture. This diagram includes clear definitions of crucial modules like User, CollaborativeFilteringModel, and Recommender, complemented by details about the important fields and methods within each module. This clarity aids engineers in understanding the core workflows and functional components. Furthermore, the design incorporates calling relationships between modules, following principles of separation of concerns and loose coupling at the system level. The translation from human natural language to a structured technical design provides actionable details which can facilitate engineering implementation, beyond high-level overviews.

Although the system design provides an overall framework and module design, it is insufficient on its own for engineers to implement complex system coding. Engineers still require additional details on how the operations are carried out within and between modules to convert the design into functional code. As illustrated in Figure 5, the architect also creates a sequence flow diagram base on the system interface design, depicting the processes, objects involved, and the sequence of messages exchanged between them needed to carry out the functionality. As mentioned, this supplementing details makes the engineer and other collaborator such as project manager who is responsible for detail code design work easier.

Hence, the consistent, synergistic outputs of the architect roles are crucial for improving code quality by simplifying the engineer’s task of translating specifications into functional code. They reduce ambiguity, misinterpretations, and confusion that can arise from freeform natural language.

In conclusion, the design and implementation of standardized outputs in MetaGPT offer a powerful tool for handling complex tasks. The conversion of complex tasks defined in natural language into structured and standardized outputs promotes collaborative consistency, reducing the need for excessive dialog turns that could lead to incoherence. Furthermore, it allows for clear and stable representation of structural information, which can be difficult to convey unambiguously through natural language alone, particularly for LLM-based agents. By providing these structured and standardized outputs, different agents gain a clear, aligned understanding of their tasks and responsibilities. This approach not only streamlines communication but also enhances the LLM-based multi-agents system’s ability to administer and execute intricate tasks more effectively.

#### 3.2.4 Knowledge Sharing Mechanisms & Customized Knowledge Management

In MetaGPT, each agent proactively curates personalized knowledge by retrieving relevant historical messages from shared environment logs. Instead of passively relying on dialogue, agents leverage role-based interests to extract pertinent information. Specifically, the environment replicates messages to provide a unified data repository. Agents register subscriptions based on message types meaningful for their roles. Matching messages are automatically dispatched to notify appropriate agents. Internally, agents maintain a memory cache indexing subscribed messages by content, publishing agent, and other attributes. Retrieval mechanisms allow agents to query this storage as needed to obtain contextual details. Updates synchronize across linked agent memories to maintain a consistent view. This decentralized yet unified access pattern mirrors how human organizations function - team members have shared records but customize views around their responsibilities. By framing information flows around agent roles, MetaGPT allows autonomous agents to efficiently self-serve appropriate knowledge.

As previously discussed, each agent in MetaGPT maintains a memory cache indexing subscribed messages pertinent to its role, enabling personalized knowledge curation. Specifically, the centralized environment replication of messages creates a unified data source.Agents then register subscriptions to obtain role-relevant messages automatically from this source. Internally, agent memory caches are indexed by content, source, and attributes to enable fast in-context retrieval. Rather than one-size-fits-all communications, this decentralized yet federated knowledge ecosystem mirrors how human teams customize information views around individual responsibilities while relying on shared records.

##### Message replication

Whenever an agent generates a message, it is replicated to the shared environment log, creating a single source of truth. This ensures all agents have access to the same information.

##### Role-based subscriptions

Agents can register subscriptions based on their roles and the types of messages that are meaningful for them. This is done based on predefined criteria that align with the agent’s responsibilities and tasks.

##### Message dispatch

When a new message matches the subscription criteria of an agent, it is automatically dispatched to notify the relevant agent. This proactive information dissemination prevents agents from missing out on important updates.

##### Memory caching and indexing

Agents maintain an internal memory cache where they store and index subscribed messages by their content, the agent that published them, and other relevant attributes. This allows for efficient storage and retrieval of information.

##### Contextual retrieval

The environment maintain a shared memory pool where supports cacheing and indexing. At the same time, agents can query their internal memory as needed to obtain contextual details relevant to their current task. This helps in refining their understanding and making better decisions.

##### Updates synchronization

Any updates or changes made to the messages are synchronized across all linked agent memories to maintain a consistent view of the information. This ensures that all agents have access to the most up-to-date data.

By aligning information flows around agent roles, MetaGPT empowers autonomous agents to efficiently serve appropriate knowledge. This pattern mirrors how human organizations function - while everyone has access to shared records, individuals tailor their views based on their specific responsibilities.

MetaGPT’s combination of centralized message sharing with personalized role-based memory caches enables customized knowledge management, reducing irrelevant data while providing common context. This balances team coordination with individual efficiency.

### 3.3 A Multi-Agent Collaboration Example

MetaGPT is capable of generating entire software systems from a single line of user instructions. This section will expand on Figure 2 and give a more detailed breakdown of MetaGPT using waterfall-style SOP, when prompted with "Make the 2048 sliding tile number puzzle game":

##### Alice (Product Manager): ready to WritePRD

When the boss (user) puts forward a product requirement, Alice, the product manager, will draft 7 pieces of documents, including: Product goals, User stories, Competitive Analysis (in text and as a quadrant chart), Requirement analysis, Requirement Pool, UI design. Before Alice can hand off her work to the next LLM agent according to the waterfall SOP, her work will first be reviewed:

""" ## Anything UNCLEAR There are no unclear points. """

After review, Alice will post her work into the message queue under the "WritePRD" category. Bob, our LLM architect, is subscribed to the "WritePRD" category. Additionally, MetaGPT will notify the users of their current API usage in dollar amounts:

""" Total running cost: $0.072 | Max budget: $3.000 | Current cost: $0.072, prompt_tokens=848, completion_tokens=771 """

##### Bob (Architect): ready to WriteDesign

Given the requirement analysis and feasibility analysis from Alice, our architect will draft an system design plan for the project, which starts with a high-level implementation approach:

""" Implementation approach: We will use Python’s built-in libraries for the core game logic. For the GUI, we will use the open-source library Pygame, which is a set of Python modules designed for writing video games. It includes computer graphics and sound libraries. For the high score tracking feature, we will use SQLite, a C library that provides a lightweight disk-based database. SQLite allows us to persist the high score even afterthe game is closed. For testing, we will use Pytest, a mature full-featured Python testing tool. """

Following the implementation plan, architect Bob will create a list of files to atomize the complicated logic of the plan into a dozen files:

["main.py", "game.py", "gui.py", "database.py", "test_game.py", "test_gui.py", "test_database.py"]

and generate mermaid diagrams to define data structures and interfaces along with program sequence flow diagram, as illustrated in Figure 6.

*(a) Program sequence flow diagram*

*(b) Data structures and interfaces design*

*Figure 6: Sequence flow diagram and system interface design automatically generated by the agent Bob in MetaGPT.*

Bob will proceed to generate the file directories and empty files according to the design, and post all his work (documents and diagrams) into the message queue under the "WriteDesign" category.

##### Eve (Project Manager): ready to WriteTasks

Eve is subscribed to the "WriteDesign" category and uses all the previously drafted UI, system design, API design documents to break down the project into simpler and more actionable tasks/tickets, usually at the complexity level of single file of code:

## Logic Analysis [ ("main.py", "Contains the main game loop and handles user input."), ("game.py", "Implements the game logic, including the board, score, and game over condition."), ("gui.py", "Handles the graphical user interface, including drawing the board, score, and game over screen."), ("database.py", "Manages the high score database, including getting and updating the high score."), ("test_game.py", "Tests the game logic."), ("test_gui.py", "Tests the graphical user interface."), ("test_database.py", "Tests the high score database.") ] ## Task list [ "game.py", "database.py", "gui.py", "main.py", "test_game.py", "test_database.py", "test_gui.py" ] ## Shared Knowledge """ ’game.py’ contains the ’Game’ class, which includes the game logic. ’database.py’ contains the ’Database’ class, which manages the high score database. ’gui.py’ contains the ’GUI’ class, which handles the graphical user interface. ’main.py’ contains the main game loop and handles user input. """

Additionally, Eve, the project manager, will list out the project dependencies, including third-party packages written in Python as well as other languages and APIs specs. When there are missing details, the review/reflection process will iterative improve the generated content in the current phase until the reviewer is satisfied. e.g.

""" ## Anything UNCLEAR There is no main entry point defined in the project. We need to clarify whether the main game loop will be in ’main.py’ or in another file. """

##### Alex (Engineer): ready to WriteCode and WriteCodeReview

Given concrete tasks and coherent overall implementation plan in the form of "WriteTasks" messages, the LLM Engineer agents has enough information to generate error-free code. Alex will go through the list of files in the given order, and he generates each file as well as its corresponding unit tests.

""" Total running cost: $1.118 | Max budget: \$3.000 | Current cost: \$0.158, prompt_tokens=4565, completion_tokens=354 Done generating. """

On the first try, MetaGPT has successfully generated an error-free 2048 sliding tile puzzle game. All these from a single line of user instruction.

*Figure 7: The MetaGPT-generated runtime interface for the 2048 sliding tile puzzle game.*

## 4 Experiments

### 4.1 Task Selection & Evaluation Method

##### Evaluation Metrics

In order to quantify the results of our experiments, we have defined certain metrics.

-

Code statistics

-

Total number of code files reflects the scale of the coding effort by measuring the number of unique code files generated.

-

Total number of lines in code files provides a comprehensive view of the amount of code written, counting all lines in the code files.

-

Average number of lines per code file evaluates the typical code complexity, calculated as the average number of lines per code file.

-

Documentation statistics

-

Total number of document files indicates the extent of documentation produced, counted as the number of unique documentation files generated.

-

Total number of lines of document reflects the volume of the written content in the documentation by counting all lines in the document files.

-

Average number of lines per document file gives an idea of the average length and depth of the documentation files.

-

Total number of document types represents the variety of documentation types produced during the task execution.

-

Cost statistics

-

Total prompt tokens reflects the level of system interaction needed, counted as the number of prompt tokens used during task execution.

-

Total completion tokens provides a sense of the scale of output generated, counted as the number of completion tokens during task execution.

-

Time costs indicates the efficiency of the task execution process, measured as the total time consumed for task execution.

-

Money costs shows the cost-effectiveness of task execution process, calculated as the total monetary cost incurred during task execution.

-

Cost of revisions Captures the maintenance effort required for the code. A higher value suggests a greater degree of code improvement and debugging.

-

Code executability

-

Functional quality of the generated code Assessed on a grading scale from ’F’ for total failure to ’P’ for flawless execution and perfect conformity to task specifications:

-

F for complete failure, scoring 0 points. The generated code is non-functional or the workflow deviates entirely from specifications.

-

R for runnable code, scoring 1 point. The code executes but workflow expectations may be unmet.

-

W for largely expected workflow, scoring 2 points. Code runs and workflow meets most specifications.

-

P for perfect match to expectations, scoring 3 points. Code functions flawlessly and workflow output perfectly matches specifications.

##### Experiments Settings

We conducted seven diverse experiments using MetaGPT within a Python environment (version 3.9.6). These experiments aimed to demonstrate its versatility across various scenarios, including gaming, web development, and data analysis. MetaGPT version 8cc8b80 served as the experimental code, with GPT4-32k as the underlying language model. The experiments had specific configurations: a maximum token consumption limit of 1500, an investment cap of 3, and a maximum of 5 iterations. Moreover, we activated the code review feature and utilized the mermaid.js for PDF and diagram generation. Each project underwent a single generation process. The complete experimental record form can be found in Appendix B.

To comprehensively validate the advantages of the MetaGPT framework, we conducted offline experimental evaluations of MetaGPT across a broad spectrum of more than 70 tasks, which were designed to evaluate the framework’s feasibility and general applicability. This heterogeneous pool of tasks, spanning numerous domains and a range of complexity, was carefully chosen with an aim to provide an exhaustive assessment of the potential of MetaGPT. Each task was evaluated using key metrics, including the code statistics, documentation statistics, cost statistics, cost of revisions and the success rate in terms of code execution. For a more detailed understanding of our experimental setup and outcomes, we have appended a subset of these tasks in the Appendix B. In short, using the MetaGPT framework takes an average of 516 seconds and $1.12 to get a project containing 4.71 code files, 3 PRDs and 3 documents. After no more than three bug-fixes, the success rate of the project generated can reach 51.43%.

It is important to underscore that the statistics presented here represent the outcomes of the current experimental suite and are not to be considered as definitive performance benchmarks. The performance of MetaGPT may vary depending on the specific experimental conditions and the task configurations utilized.

### 4.2 Comparison with Alternative Approaches

In this section, we first provide a clear comparison of MetaGPT, AutoGPT, and AgentVerse’s capabilities in Table 1. Then we conduct experiments to quantify the performance of different framworks.

##### Framework Capability Comparison

MetaGPT stands out with its extensive functionality. Unique to MetaGPT are the abilities to generate PRDs and Technical Designs, emphasizing its comprehensive project execution approach. MetaGPT is also the only framework capable of API Interface Generation, offering an edge in rapid API design prototyping scenarios.

Code Review, a crucial component of the development process, is a feature available in both MetaGPT and AgentVerse, but notably absent in AutoGPT. MetaGPT sets itself apart further by incorporating Precompilation Execution, a feature that facilitates early error detection and thereby, enhances the quality of the code. In terms of collaborative features, MetaGPT and AgentVerse both support Role-Based Task Collaboration, a mechanism which facilitates multi-agent collaboration and enhances teamwork by partitioning tasks among specific roles. However, MetaGPT exclusively offers Role-Based Task Management, a feature that not only decomposes tasks but also oversees their administration, thus underlining its comprehensive project management capabilities. When assessing Code Generation abilities, all three frameworks exhibit proficiency. However, MetaGPT delivers a more encompassing solution, addressing wider aspects of the development process, thereby offering a comprehensive toolset for project management and execution.

This comparison is based on the current states of the respective frameworks. Future updates might add or alter the features of these tools. However, as of this analysis, MetaGPT outshines its counterparts in providing a more comprehensive and robust solution for project execution.

##### Quantitative Experiment Comparison

To evaluate the performance of various frameworks such as MetaGPT, AutoGPT, and AgentVerse, we conducted experiments across 7 diverse tasks. These tasks includes python games generation, CRUD code generation and simple data analysis, This approach was aimed to illuminate the distinctive strengths and shortcomings of each framework under scrutiny. The results are shown in Table 2

*Table 1: Comparison of capabilities across MetaGPT, AutoGPT, and AgentVerse. Note that ‘✓’ indicates the presence of a given feature in the respective framework.*

| Framwork capabiliy | AutoGPT | AgentVerse | MetaGPT |

| PRD generation | | | ✓ |

| Tenical design genenration | | | ✓ |

| API interface generation | | | ✓ |

| Code generation | ✓ | ✓ | ✓ |

| Precompilation execution | | | ✓ |

| Role-based task management | | | ✓ |

| Code review | | ✓ | ✓ |

| Role-based task collaboration | | ✓ | ✓ |

*Table 2: Comparison of task executability between AutoGPT, AgentVerse, and MetaGPT. The tasks are scored based on a grading system from ‘0’ to ‘3’, where ‘0’ denotes ‘complete failure’, ‘1’ denotes ‘runnable code’, 2 denotes ‘largely expected workflow’, and ‘3’ denotes ‘perfect match to expectations’ (shown in Section 4.1).*

| Task | AutoGPT | AgentVerse | LangChain w/ Python REPL tool | MetaGPT |

| Flapy bird game | 0 | 0 | 0 | 0 |

| Tank battle game | 0 | 0 | 0 | 0 |

| 2048 game | 0 | 0 | 0 | 1 |

| Snake game | 0 | 0 | 0 | 2 |

| Brick breaker game | 0 | 0 | 0 | 3 |

| Excel data process | 0 | 0 | 0 | 3 |

| CRUD manage | 0 | 0 | 0 | 3 |

As evidenced by the data presented in Table 2, MetaGPT exhibits robust performance across a diverse set of tasks, achieving successful execution in all but two instances: Flappy Bird and Tank Battle. These tasks, which possess high interaction demands, were not successfully completed by MetaGPT due to the strict constraints and limited resources allocated for manual adjustments. In direct contrast, the competing frameworks, AutoGPT and AgentVerse, did not accomplish successful execution in any of the tasks, marking a stark differentiation in the effectiveness of the MetaGPT framework.

We also provide more details about the runtime statistics about MetaGPT, shown in Table 3. Across the aforementioned experimental projects, an average of 4.71 code files (including but not limited to formats such as CSS, py, js) were present per project, with an average of 42.99 lines of code per file. Regarding PRD files, each project generated an average of three PRD files (considering pdf, mmd, and png with the same name as one file). Additionally, for project documentation, there were three documents on average per project, typically comprising product requirement documents, API documentation, and system architecture documentation, with each document averaging 80 lines.

*Table 3: The statistical analysis of MetaGPT in software development. The minimum (Min), maximum (Max), and average (Avg.) values for various statistical indexes are reported. ‘#’ denotes ‘The number of’*

| Statistical index | Min | Max | Avg. |

| # Code files | 3.00 | 6.00 | 4.71 |

| # Documents files | 3.00 | 3.00 | 3.00 |

| # PRD files | 3.00 | 3.00 | 3.00 |

| # Lines of codes | 17.00 | 96.00 | 42.99 |

| # Prompt tokens | 21934.00 | 32517.00 | 26626.86 |

| # Completion tokens | 5312.00 | 7104.00 | 6218.00 |

| Cost statistics | $ 0.90 | $ 1.35 | $ 1.09 |

| Cost of revisions(only R, W and P) | 0.00 | 2.00 | 0.60 |

In terms of cost analysis, each project consumed an average of 26626.86 tokens for prompts and 6218.00 tokens upon task completion, resulting in a total cost of $ 1.09 for completing the tasks. The entire construction process took 517.71 seconds. Compared to traditional software engineering development timelines and costs, MetaGPT’s time and monetary expenses amount to less than one-thousandth.

We adopt Code of Revisions as the metric of resolving errors encountered during project execution by means of dependency replacement, code modifications, or other corrective actions, until successful execution or encountering the next issue. The maximum number of allowed in Code of Revision is three; exceeding this limit results in the project being considered a failure if it continues to encounter errors. Regarding Code of Revisions, each project required an average of 0.6 revisions, with the majority of issues relating to dependencies, resource unavailability, and missing parameters. The overall success rate (WP rate - running successfully and generally meeting expectations) stands at 57.14%.

Although AutoGPT is currently the most prevalent single-agent framework, it was unable to successfully complete any tasks using the GPT4-32k configuration with its default setup in our experiments. As a single agent, its characteristics must be manually established prior to the execution process, and these cannot be altered mid-way. AutoGPT is capable of decomposing user-provided tasks into multiple smaller subtasks and executes them sequentially. However, our observations throughout the experimental procedure highlight a significant drawback of AutoGPT: its lack of completeness evaluation and the expertise of the single agent.

AutoGPT lacks a mechanism to assess the completeness of a task. It merely marks a task as completed after saving the generated results, without any further examination for validity or completeness. To attempt to create a more efficient loop, we employed the AutoGPT implementation in LangChain and integrated the Python Read-Eval-Print Loop (REPL) tool. Our intent was to enable AutoGPT to debug and refine the code it authored. However, the agent’s lack of specialized knowledge prevented it from utilizing the feedback provided by the interpreter to improve its code. Consequently, the tasks remained unsuccessful due to the generation of incomplete and non-functional code.

Despite implementing three specialized roles: Writer, Tester, and Reviewer, AgentVerse failed across all benchmark tasks. These roles performs collaboration in a manner of online judges (OJs). The conversation within AgentVerse primarily revolves around the Writer role creating code, the Tester identifying areas of failure or error messages during code execution, and the Reviewer suggesting modifications, which is similar to the Engineer we implement in MetaGPT capable of coding and code-review. The dialogues within AgentVerse primarily focus on the code itself rather than the overall task. However, the absence of roles responsible for breaking down large tasks into smaller, manageable ones places AgentVerse at a disadvantage. This lack of task decomposition and division of work significantly reduces the likelihood of successfully completing larger, more complex tasks. The shortcomings of AgentVerse underscore the importance of clear role delineation and strategic task segregation across different phases in the handling of complex problem-solving scenarios.

### 4.3 Ablation Studies

Our ablation studies involve systematically reducing the number of roles engaged in the development process, and subsequently examining the effects on the executable output and the validity of intermediate files. We selected Brick Breaker and Gomoku as tasks, given their complexity which necessitates a team with diverse roles and a multi-step workflow, analogous to the reality of game development, characterized by clear role divisions and teamwork.

Our initial experiments with MetaGPT were conducted employing a fully-staffed team, including four distinct roles: a ProductManager, an Architect, a ProjectManager, and an Engineer, in alignment with the experimental settings of Section 4.1 . For the ablation studies, we used a set of metrics: total number of lines in code files, money costs,cost of revisions, and code executability, as defined in Section 4.1 . These metrics were chosen to enable the straightforward quantification of aspects such as the quality of code generation, the cost-effectiveness of the task, and the functional quality of the generated code.

After obtaining results from this fully equipped team, we progressively remove the Architect, ProjectManager, and ProductManager roles in subsequent experiments, assessing how their absence impacts the overall task performance.

As depicted in Table 4 and Table 5, retaining three roles while removing either the Architect or ProjectManager leads to a moderate decrease in code statistics, 29 and 14 fewer lines for Brick Breaker and 33 and 47 fewer lines for Gomoku. Additionally, 1-2 extra revisions are necessitated. However, overall task executability is largely preserved. More substantial reductions in code volume and increases in required fixes become evident when transitioning from three roles to just two,the ProductManager and Engineer. In this scenario, the game tests show 62 and 63 line decreases respectively, and increments of 2 and 3 in revisions.

Additionally, it is critical to note that when the team is reduced to a single agent, the executability of the code significantly deteriorates. Even with additional revision costs incurred, the code remains non-executable. Therefore, the presence of multiple roles not only enhances code quality but also reinforces the robustness and feasibility of the code implementation.

*Table 4: Ablation study on roles in Brick breaker game development. ‘#’ denotes ‘The number of’, ‘Product’ denotes ‘Product manager’, ‘Project’ denotes ‘Project manager’. ‘F’ denotes ‘complete failure’, ‘R’:unnable code, ‘W’: argely expected workflow, ‘P’: perfect.*

| Engineer | Product | Architect | Project | # agents | Lines of code | Money cost | Cost of revisions | Code executability |

| ✓ | | | | 1 | 89 | $ 0.876 | 8 | F |

| ✓ | ✓ | | | 2 | 115 | $ 1.022 | 4 | R |

| ✓ | ✓ | ✓ | | 3 | 177 | $ 1.204 | 2 | W |

| ✓ | ✓ | | ✓ | 3 | 162 | $ 1.221 | 2 | R |

| ✓ | ✓ | ✓ | ✓ | 4 | 191 | $ 1.350 | 1 | P |

*Table 5: Ablation study on roles in Gomoku game development. ‘#’ denotes ‘The number of’, ‘Product’ denotes ‘Product manager’, ‘Project’ denotes ‘Project manager’. ‘F’ denotes ‘complete failure’, ‘R’:unnable code, ‘W’: argely expected workflow, ‘P’: perfect.*

| Engineer | Product | Architect | Project | # agents | Lines of code | Money cost | Cost of revisions | Code executability |

| ✓ | | | | 1 | 77 | $ 0.953 | 12 | F |

| ✓ | ✓ | | | 2 | 109 | $ 1.095 | 9 | R |

| ✓ | ✓ | ✓ | | 3 | 172 | $ 1.299 | 6 | R |

| ✓ | ✓ | | ✓ | 3 | 186 | $ 1.274 | 5 | R |

| ✓ | ✓ | ✓ | ✓ | 4 | 219 | $ 1.420 | 4 | W |

In summary, through quantitative comparisons across conditions, we underscore the advantages of specialized multi-agent frameworks for complex tasks. This validates the importance of role modularity and collaboration for holistic task completion.

## 5 Discussion and Future Work

Despite the immense potential of MetaGPT in automating end-to-end processes, it also has several limitations. Primarily, it occasionally references non-existent resource files like images and audio. Furthermore, during the execution of complex tasks, it is prone to invoking undefined or unimported classes or variables. These phenomenons are widely attributed to the hallucinatory tendency inherent in large language models and could be handled by a more clear and efficient agent collaboration workflow.

## 6 Conclusion

In this work, we have presented MetaGPT, a promising framework for collaborative agents through SOPs that manages LLMs to mimic the efficient human workflows. To encodes SOPs into prompts, MetaGPT manages multi-agents through role definition, task decomposition, process standardization and other technical design and eventually complete the end-to-end development process with only one-line requirement. Our illustrative example in software development showcases the potential of this framework with detailed SOPs and prompts. The experimental results demonstrate our MetaGPT can produce comprehensive solutions with higher coherence relative to existing conversational and chat-based multi-agent system. We believe that this work opens new possibilities for the way multi-agents interact and cooperate, redefines the landscape of complex problem-solving, and points a potential pathway towards Artificial General Intelligence.

## References

- [1] R.M. Belbin. Team Roles at Work. Routledge, 2012.

- [2] Agile Manifesto. Manifesto for agile software development. 2001.

- [3] T. DeMarco and T.R. Lister. Peopleware: Productive Projects and Teams. Addison-Wesley, 2013.

- [4] Ian R McChesney and Seamus Gallagher. Communication and co-ordination practices in software engineering projects. Information and Software Technology, 2004.

- [5] Michael Wooldridge and Nicholas R. Jennings. Pitfalls of agent-oriented development. In Proceedings of the Second International Conference on Autonomous Agents, 1998.

- [6] Henry van Dyke Parunak and James Odell. Representing social structures in uml. In ICAS, 2001.

- [7] Cristiano Castelfranchi. Founding agents’" autonomy" on dependence theory. In ECAI, 2000.

- [8] Jacques Ferber and Jean-Pierre Müller. Influences and reaction: a model of situated multiagent systems. In ICMAS, 1996.

- [9] Jacques Ferber and Olivier Gutknecht. A meta-model for the analysis and design of organizations in multi-agent systems. In ICMAS, 1998.

- [10] Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint, 2023.

- [11] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large scale language model society. arXiv preprint, 2023.

- [12] Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. Large language models as tool makers. arXiv preprint, 2023.

- [13] Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona self-collaboration. arXiv preprint, 2023.

- [14] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate, 2023.

- [15] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multi-agent debate. arXiv preprint, 2023.

- [16] Rui Hao, Linmei Hu, Weijian Qi, Qingliu Wu, Yirui Zhang, and Liqiang Nie. Chatllm network: More brains, more intelligence. arXiv preprint, 2023.

- [17] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint, 2023.

- [18] Torantulino et al. Auto-gpt. https://github.com/Significant-Gravitas/Auto-GPT, 2023.

- [19] chenweize1998, yushengsu thu, chanchimin, libowen2121, Xial-kotori, Dr-Left, tzw2698, and zhouxh19. Agentverse. https://github.com/OpenBMB/AgentVerse, 2023.

- [20] Harrison Chase. LangChain. https://github.com/hwchase17/langchain, 2022.

- [21] Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-tau Yih, Sida Wang, and Xi Victoria Lin. Lever: Learning to verify language-to-code generation with execution. In ICML, 2023.

- [22] Marta Skreta, Naruki Yoshikawa, Sebastian Arellano-Rubach, Zhi Ji, Lasse Bjørn Kristensen, Kourosh Darvish, Alán Aspuru-Guzik, Florian Shkurti, and Animesh Garg. Errors are useful prompts: Instruction guided task programming with verifier-assisted iterative prompting. arXiv preprint, 2023.

- [23] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 2022.

- [24] Xinyun Chen, Chang Liu, and Dawn Song. Execution-guided neural program synthesis. In ICLR, 2018.

- [25] Xinyun Chen, Dawn Song, and Yuandong Tian. Latent execution for neural program synthesis beyond domain-specific languages. NeurIPS, 2021.

- [26] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint, 2022.

- [27] Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint, 2023.

- [28] Bill Yuchen Lin, Yicheng Fu, Karina Yang, Prithviraj Ammanabrolu, Faeze Brahman, Shiyu Huang, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. Swiftsage: A generative agent with fast and slow thinking for complex interactive tasks. arXiv preprint, 2023.

- [29] Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development, 2023.

- [30] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.

- [31] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint, 2023.

- [32] Yashar Talebirad and Amirhossein Nadiri. Multi-agent collaboration: Harnessing the power of intelligent llm agents, 2023.

- [33] Jimmy Wei, Kurt Shuster, Arthur Szlam, Jason Weston, Jack Urbanek, and Mojtaba Komeili. Multi-party chat: Conversational agents in group settings with humans and models. arXiv preprint, 2023.

- [34] Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. arXiv preprint, 2023.

- [35] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. arXiv preprint, 2023.

- [36] joshsny, timconnorz, Alec2435, HaniaA, ryderwishart, and zoan37. Gpteam. https://github.com/101dotxyz/GPTeam, 2023.

- [37] pchalasani, Mohannadcse, and AshishHoodaIITD. langroid. https://github.com/langroid/langroid, 2023.

- [38] RunzheYang. Socraticai. https://github.com/RunzheYang/SocraticAI, 2023.

- [39] Knut Sveidqvist and Contributors to Mermaid. Mermaid: Generate diagrams from markdown-like text. https://github.com/mermaid-js/mermaid, 2014.

## Appendix A Detailed Standardard Output Artifacts

This appendix provides additional examples of standardized output artifacts generated by different roles in MetaGPT during collaborative workflows. These help illustrate the structure and format of outputs that aim to match industry conventions and domain-specific best practices.

##### ProductManager Outputs

The product manager role produces PRD including product goals, user stories, competitive analysis, competitive quadrant chart, requirement analysis and requirement pool.

1) PRD of 2048 Game Design

## Original Requirements The boss wants us to create a 2048 game. ## Product Goals [ "Create a 2048 game that is easy to understand and play", "Ensure the game is engaging and retains user interest", "Design the game to be visually appealing and intuitive" ] ## User Stories [ "As a user, I want to be able to start a new game easily so that I can play whenever I want", "As a user, I want to see my current score and high score, so that I can track my progress", "As a user, I want the game to respond quickly to my actions, so that I can play without interruptions", "As a user, I want the game to have an appealing design, so that I enjoy my playing experience", "As a user, I want the game to have different difficulty levels, so that I can challenge myself" ] ## Competitive Analysis [ "2048 by Gabriele Cirulli: The original game, simple and intuitive, but lacks variety", "2048 Number puzzle game by Estoty Entertainment LLC: Offers different board sizes, but has intrusive ads", "2048 by Androbaby: Has a clean design, but lacks difficulty levels", "2048 by R. App: Offers undo feature, but the design is not very appealing", "2048 by Ketchapp: Has a modern design, but the controls are not very responsive", "2048 by X-Flow: Offers different game modes, but the user interface is not very intuitive", "2048 by Solebon LLC: Has a minimalistic design, but lacks variety in gameplay" ] ## Competitive Quadrant Chart ‘‘‘mermaid quadrantChart title Reach and engagement of campaigns x-axis Low Reach --> High Reach y-axis Low Engagement --> High Engagement quadrant-1 We should expand quadrant-2 Need to promote quadrant-3 Re-evaluate quadrant-4 May be improved "2048 by Gabriele Cirulli": [0.8, 0.6] "2048 Number puzzle game by Estoty Entertainment LLC": [0.7, 0.4] "2048 by Androbaby": [0.6, 0.5] "2048 by R. App": [0.5, 0.3] "2048 by Ketchapp": [0.7, 0.5] "2048 by X-Flow": [0.6, 0.6] "2048 by Solebon LLC": [0.5, 0.4] "Our Target Product": [0.7, 0.7] ‘‘‘ ## Requirement Analysis The product should be a 2048 game that is easy to play, visually appealing, and engaging. It should have different difficulty levels and a quick response time. The game should also display the current score and high score, and have an intuitive user interface. ## Requirement Pool [ ("Implement the basic 2048 game mechanics", "P0"), ("Design a visually appealing and intuitive user interface", "P0"), ("Add different difficulty levels", "P1"), ("Display the current score and high score", "P0"), ("Ensure a quick response time", "P0") ] ## Anything UNCLEAR There are no unclear points.

2) PRD of Snake Game Design

## Original Requirements The boss wants us to create a snake game. ## Product Goals [ "Create a snake game that is easy to play and understand", "Ensure the game is challenging and engaging to keep users interested", "Design the game to be visually appealing and responsive" ] ## User Stories [ "As a user, I want to be able to easily control the snake so that I can play the game without difficulty", "As a user, I want the game to progressively get more difficult so that I am constantly challenged", "As a user, I want to be able to pause and resume the game so that I can play at my own pace", "As a user, I want the game to have a visually appealing design so that I enjoy playing", "As a user, I want the game to respond quickly to my actions so that I can play effectively" ] ## Competitive Analysis [ "Python Snake Game: Simple and easy to play but lacks visual appeal", "Google Snake Game: Visually appealing and responsive but can be too difficult for beginners", "Classic Snake Game: Offers a good level of challenge but lacks modern design elements", "Slither.io: Multiplayer game with modern design but can be too complex for some users", "Snake ’97: Retro design that appeals to nostalgia but lacks modern gameplay elements" ] ## Competitive Quadrant Chart ‘‘‘mermaid quadrantChart title Reach and engagement of campaigns x-axis Low Reach --> High Reach y-axis Low Engagement --> High Engagement quadrant-1 We should expand quadrant-2 Need to promote quadrant-3 Re-evaluate quadrant-4 May be improved "Python Snake Game": [0.3, 0.6] "Google Snake Game": [0.45, 0.23] "Classic Snake Game": [0.57, 0.69] "Slither.io": [0.78, 0.34] "Snake ’97": [0.40, 0.34] "Our Target Product": [0.5, 0.6] ‘‘‘ ## Requirement Analysis The product should be a snake game that is easy to play, visually appealing, and progressively challenging. It should have responsive controls and allow users to pause and resume the game. ## Requirement Pool [ ("Easy to control snake movement", "P0"), ("Progressive difficulty levels", "P0"), ("Pause and resume feature", "P1"), ("Visually appealing design", "P1"), ("Responsive controls", "P0") ] ## Anything UNCLEAR There are no unclear points.

##### Architect Outputs

The software architect role produces technical specifications like system architecture diagrams and interface definitions. We showcase the interface design for search algorithms framework 8, LLM-based operation system 9,Minimalist Pomodoro timer 10, Pyrogue game 11, and Match3 puzzle game 12

*Figure 8: The system interface design for search algorithms frameworks automatically generated by the architect agent in MetaGPT.*

*Figure 9: The system interface design for LLM-based operation system automatically generated by the architect agent in MetaGPT.*

*Figure 10: The interface design for Minimalist Pomodoro timer by the architect agent in MetaGPT.*

*Figure 11: The interface design for Pyrogue game by architect agent in MetaGPT.*

*Figure 12: The interface design for Match3 puzzle game by architect agent in MetaGPT.*

##### ProjectManager Outputs

The projectmanager role produces technical specifications like sequence flow chart. The examples includes search algorithms frameworks 13, LLM-based operation system 14, Minimalist Pomodoro timer 15

*Figure 13: The sequence flow diagram for search algorithms frameworks automatically generated by the project manager agent in MetaGPT.*

*Figure 14: The sequence flow diagram for LLM-based operation system by project manager agent in MetaGPT.*

*Figure 15: The sequence flow chart for Minimalist Pomodoro timer by the project manager agent in MetaGPT.*

## Appendix B MetaGPT Experimental Tasks Detail

*Table 6: Subset of MetaGPT Experimental Tasks’ Prompt*

| Task ID | Task | Prompt |

| 0 | Snake game | create a snake game. |

| 1 | Brick breaker game | create a brick breaker game. |

| 2 | 2048 game | create a 2048 game for the web. |

| 3 | Flappy bird game | write p5.js code for Flappy Bird where you control a yellow bird continuously flying between a series of green pipes. The bird flaps every time you left click the mouse. If the bird falls to the ground or hits a pipe, you lose. This game goes on infinitely until you lose and you get points the further you go. |

| 4 | Tank battle game | create a tank battle game. |

| 5 | Excel data process | Write an excel data processing program based on streamlit and pandas,The screen first has an excel upload processing button. After the excel is uploaded, use pandas to display the data content in the excel.The program is required to be concise, easy to maintain, and not over-designed.The program as a whole uses streamlit to process web screen display, and pandas is sufficient to process excel reading and display. Others do not need to introduce additional packages. |

| 6 | CRUD manage | Write a management program based on the crud addition, deletion, modification and query processing of the customer business entity,The customer needs to save the information of name, birthday, age, sex, and phone.The data is stored in client.db as a whole, and it is judged whether the customer table exists. If it does not exist, it needs to be created first.When querying, you need to query by name, and when deleting, you also need to delete by name.The program is required to be concise, easy to maintain, and not over-designed.The screen is realized through streamlit and sqlite, no need to introduce other additional packages. |

| 7 | Music transcriber | Developing a program to transcribe sheet music into a digital format, Providing error-free transcribed symbolized sheet music intelligence from audio through signal processing involving pitch and time slicing then training a neural net to run Onset Detected CWT transforming scalograms to chromagrams decoded with Recursive Neural Network focused networks. |

| 8 | Custom press releases | Create custom press releases, Develop a Python script that extracts relevant information about company news from external sources, such as social media extract update interval database for recent changes. The program should create press releases with customizable options and export writings to PDFs, NYTimes API JSONs, media format styled with interlink internal fixed character-length metadata. |

| 9 | Gomoku game | Implement a Gomoku game using Python, incorporating an AI opponent with varying difficulty levels. |

| 10 | Weather dashboard | Create a Python program to develop an interactive weather dashboard. |

*Table 7: Subset of MetaGPT experimental tasks’ details. An Average (Avg.) of 70 tasks were calculated and 10 randomly selected tasks is included. ‘#’ denotes ‘The number of’, while ‘ID’ is ‘Task ID’.*

| ID | Code statistics | Doc statistics | | Cost statistics | Cost of revision | Code executability |

| | #code files | #lines of code | #lines per code file | #doc files | #lines of doc | #lines per doc file | #prompt tokens | #completion tokens | time costs | money costs | | |

| 0 | 5.00 | 196.00 | 39.20 | 3.00 | 210.00 | 70.00 | 24087.00 | 6157.00 | 582.04 | $ 1.09 | 1. TypeError | W(2) |

| 1 | 6.00 | 191.00 | 31.83 | 3.00 | 230.00 | 76.67 | 32517.00 | 6238.00 | 566.30 | $ 1.35 | 1. TypeError | P(3) |

| 2 | 3.00 | 198.00 | 66.00 | 3.00 | 235.00 | 78.33 | 21934.00 | 6316.00 | 553.11 | $ 1.04 | 1. lack @app.route(’/’) | R(1) |

| 3 | 5.00 | 164 | 32.80 | 3.00 | 202.00 | 67.33 | 22951.00 | 5312.00 | 481.34 | $ 1.01 | 1. PNG file missing 2. Compile bug fixes | F(0) |

| 4 | 6.00 | 203.00 | 33.83 | 3.00 | 210.00 | 70.00 | 30087.00 | 6567.00 | 599.58 | $ 1.30 | 1. PNG file missing 2. Compile bug fixes 3. pygame.surface not initialize | F(0) |

| 5 | 6.00 | 219.00 | 36.50 | 3.00 | 294.00 | 96.00 | 35590.00 | 7336.00 | 585.10 | $ 1.51 | 1. dependency error 2. ModuleNotFoundError | R(1) |

| 6 | 4.00 | 73.00 | 18.25 | 3.00 | 261.00 | 87.00 | 25673.00 | 5832.00 | 398.83 | $ 0.90 | 0 | P(3) |

| 7 | 4.00 | 316.00 | 79.00 | 3.00 | 332.00 | 110.67 | 29139.00 | 7104.00 | 435.83 | $ 0.92 | 0 | P(3) |

| 8 | 5.00 | 215.00 | 43.00 | 3.00 | 301.00 | 100.33 | 29372.00 | 6499.00 | 621.73 | $ 1.27 | 1. tensorflow version error 2. model training method not implement | R(1) |

| 9 | 5.00 | 215.00 | 43.00 | 3.00 | 270.00 | 90.00 | 24799.00 | 5734.00 | 550.88 | $ 1.27 | 1. dependency error 2. URL 403 error | W(2) |

| 10 | 3.00 | 93.00 | 31.00 | 3.00 | 254.00 | 84.67 | 24109.00 | 5363.00 | 438.50 | $ 0.92 | 1. dependency error 2. missing main func. | P(3) |

| Avg. | 4.71 | 191.57 | 42.98 | 3.00 | 240.00 | 80.00 | 26626.86 | 6218.00 | 516.71 | $1.12 | 0.51(only R,W or P) | W(1.51) |

<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2308.08155
     paper_id : 2308.08155
     source   : https://arxiv.org/html/2308.08155v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# AutoGen: Enabling Next-Gen LLM Applications
via Multi-Agent Conversation Framework

Qingyun Wu Affiliation: Pennsylvania State University    Gagan Bansal Affiliation: Microsoft    Jieyu Zhang Affiliation: University of Washington    Yiran Wu Affiliation: Pennsylvania State University    Shaokun Zhang Affiliation: Pennsylvania State University
Erkang Zhu Affiliation: Microsoft    Beibin Li Affiliation: Microsoft    Li Jiang Affiliation: Microsoft    Xiaoyun Zhang Affiliation: Microsoft    Chi Wang Affiliation: Microsoft

###### Abstract

This technical report presents AutoGen, a new framework that enables development of LLM applications using multiple agents that can converse with each other to solve tasks. AutoGen agents are customizable, conversable, and seamlessly allow human participation. They can operate in various modes that employ combinations of LLMs, human inputs, and tools. AutoGen’s design offers multiple advantages: a) it gracefully navigates the strong but imperfect generation and reasoning abilities of these LLMs; b) it leverages human understanding and intelligence, while providing valuable automation through conversations between agents; c) it simplifies and unifies the implementation of complex LLM workflows as automated agent chats. We provide many diverse examples of how developers can easily use AutoGen to effectively solve tasks or build applications, ranging from coding, mathematics, operations research, entertainment, online decision-making, question answering, etc.

*Figure 1: AutoGen enables complex LLM-based workflows using multi-agent conversations. (Left) AutoGen agents are customizable and can be based on LLMs, tools, humans and even a combination of them. (Top-right) Agents can converse to solve tasks. (Bottom-right) The framework supports many additional complex conversation patterns.*

## 1 Introduction

Large Language Models (LLMs), like GPT-4, have demonstrated exceptional capabilities for reasoning, creativity, and deduction in many new AI applications, ranging from robot lawyers [15], tools to assist with fundamental research [7], and a plethora of highly-capable chatbots (e.g., Bard, Bing Chat, ChatGPT+, Copilot). As single LLM calls often lack the ability to reflect, possess no working memory or scratch pad, and cannot act or perceive beyond the knowledge obtained from their training data [8], augmenting language models with tools that facilitate perception and action, such as the plugin that enables ChatGPT+ to utilize fresh web pages from the internet [36], can significantly improve performance.

Despite the remarkable success the LLM-with-tools paradigm [41] has enabled, it typically utilizes a single LLM agent. Considering the evolving range of real-world tasks that could benefit from LLMs and the intrinsic weaknesses of using a single agent [29, 16, 8], a promising direction for future LLM applications is to have multiple agents work together to solve complex tasks. For instance, more agents could be introduced to (1) incorporate different roles that encourage divergent thinking [29], improve factuality and reasoning with LLMs [16] through multi-agent debate, etc; (2) allow effective tool usage and execution with potentially autonomous troubleshooting through inter-agent interactions [52]; (3) realize human agency, etc. In such multi-agent systems, inter-agent conversations will be crucial, using language (natural or code) as means to enable agent collaboration through potential multi-round, back-and-forth message exchanges. These conversations could leverage the recent chat-based LLMs’ capability to communicate and incorporate feedback from others at human level.

A unified framework that uses conversations between multiple agents can encapsulate the aforementioned insights in a general manner. Consider, for example, a developer building an application that uses LLMs to write code that solves a problem specified by a user (e.g., “Plot a chart of META and TESLA stock price change YTD”). The simplest starting point is to use a single LLM inference call to generate code. Since this code may be erroneous, the developer may use a separate tool to execute candidate code, and subsequent LLM inference calls to debug the code based on execution results — resulting in an iterative conversation between the inference and code execution. Additionally, the developer may want to let users approve code execution, clarify their intent, and remember their preferences. Designing, implementing, and optimizing this workflow can be challenging. However, using well-designed agents and multi-agent conversations, the workflow could potentially be simplified into a series of discussions between the involved agents (Figure 2).

To support the rapid development of next-gen LLM applications, an ideal framework would support the latest LLMs and augmentations, allow (potentially automated) cooperation between LLMs, tools, and seamless and flexible human involvement to leverage their expertise and intelligence. Motivated by these desiderata, we present AutoGen, a multi-agent conversation framework illustrated in Figure 1. AutoGen has the following key features:

-

Customizable agents that integrate LLMs, humans, and tools. AutoGen agent comes with a set of capabilities powered by LLMs, humans, tools, or a combination of these. By selecting and configuring a subset of built-in capabilities, one can easily create agents with different roles, as shown in Figure 2. In addition, developers can readily extend the capabilities in each backend. (ref. Section 2.1)

-

Conversable agents with unified conversation interface. One key philosophy of AutoGen is to simplify and unify complex workflows as automated agent chats. To realize this, every Autogen agent is made conversable – they can receive, react, and respond to messages. This design (1) leverages the strong capability of the most advanced LLMs in taking feedback and making progress via chat and (2) allows humans to participate flexibly during an active inter-agent conversation, enabling both human agency and automation. (ref. Section 2.1 and Section 2.2)

-

Flexible conversation patterns for complex workflows. Applications of varying complexity levels utilize different conversation patterns. With customizable and conversable agents, developers can use Autogen to build a wide range of conversation patterns concerning conversation autonomy, the number of agents, and agent conversation topology. (ref. Section 2.2)

-

A collection of working systems with different complexities. AutoGen offers a collection of working systems created using conversable agents, along with a general guideline. These systems span a wide range of applications from various domains and complexities. They demonstrate how AutoGen can easily support different conversation patterns. (See Section 4 and 5.1, Figure 5)

We introduce AutoGen in Section 2, review related work in Section 3, present applications in Section 4, and summarize the learnings and future work in Section 5.

## 2 The AutoGen Framework

AutoGen abstracts and implements conversable agents designed to solve tasks through inter-agent conversations. Using conversable agents in AutoGen, developers can create various forms and patterns of multi-agent conversations involving LLMs, humans, and tools. Figure 2 presents a simple example of two-agent chat. With AutoGen, building a complex multi-agent conversation system involves (a) defining a set of conversable agents with specialized capabilities and roles, and (b) defining the interaction behavior between agents, i.e., how an agent should respond when receiving messages from another agent. In Section 2.1, we introduce the basic concept of conversable agents and how to define them with specialized capabilities and roles. In Section 2.2, we explain how AutoGen implements agent conversations and how to define interaction behaviors to achieve different conversation patterns.

*Figure 2: A simple example of two agents solving a task through conversation. In this case, a User Proxy Agent and Assistant Agent from AutoGen help user create a visualization using data from the Web. The user proxy seamlessly engages human and uses shell when appropriate. (Note, the YTD chart is illustrative and not based on real data). While this example shows a simple scenario, Section 4 shows how AutoGen can support tasks of higher complexity. *

### 2.1 Conversable Agent

A unique feature of agents in AutoGen is their conversability, allowing them to solve tasks collectively through inter-agent conversations. A conversable agent is an entity with a specific role that can send and receive messages to and from other conversable agents to start or continue a conversation. It maintains its internal states based on sent and received messages and can be configured with a set of capabilities (e.g., enabled by LLMs, humans, tools, etc.). The agent can act according to programmed behavior patterns.

*Figure 3: Agents in AutoGen can be any combination of LLMs, humans, and tools. (Left) AssistantAgent uses LLM to respond to messages. (Middle) UserProxyAgent uses tool + human to respond. (Right) GroupChatManager uses LLM and manages a set of conversable agents from AutoGen to have conversations in a group chat setting. *

#### Agent capabilities powered by LLMs, humans, and tools.

An agent’s capability directly influences how the agent processes and responds to messages. An agent powered by a highly capable LLM has the abilities of language understanding, generation, and some reasoning. Besides LLM, agents in AutoGen can also be powered by tools or humans to overcome LLM limitations and harness human intelligence.

-

LLMs. The main goal of Autogen is to enable next-gen LLM applications through multi-agent conversations, making LLM a critical component as the back-end of agents in Autogen. Different agents can be backed by various LLM configurations. For instance, some may use LLMs or configurations tuned on private data [47, 37]. Additionally, LLMs can be set up to play different roles with distinct system messages.

AutoGen supports the use of widely used LLM inference APIs and offers enhanced and optimized LLM inference settings. It also includes various improved features, such as result caching, error handling, message templating, etc. via an enhanced LLM inference layer. These features are crucial for ensuring robust and optimized performance when agents need to use LLMs for making inferences. Additionally, one can easily apply different prompting techniques via system messages and user messages in AutoGen agents to optimize LLM behavior.

-

Humans. To effectively address user needs, next-gen LLM applications should enable easy integration of human feedback and involvement at different levels. This motivates Autogen to include human as an agent back-end. AutoGen lets a human participate in agent conversation via a proxy agent, which has the same conversation interface as any other agent. When an agent is set up as a proxy for human user (by configuring ‘human_inputs_mode’), it connects with other agents in the workflow through a unified communication interface. As shown in Figure 2, when needed, humans can directly communicate with the AssistantAgent via the UserProxyAgent: “No, please plot % change!” (on the right side of the sub-figure). AutoGen also allows configurable human involvement levels and patterns, including the frequency and conditions for requesting human input, and the option for humans to skip providing input. This enables varying degrees of autonomy. In the example in Figure 2, the human starts talking to the AssistantAgent after an initial plot is created through several rounds of automated chat.

-

Tools. Using tools is an effective way to overcome LLM limitations. AutoGen natively supports a generic form of tool usage through code generation and execution. Specifically, when using a default assistant agent from AutoGen, the system message prompts the LLM to suggest Python code or shell scripts to solve problems in several cases. These cases cover common scenarios where tools are needed, such as situations requiring information collection. The system message is carefully designed to enable intelligent multi-step problem-solving and utilize the conversational process to recover from failure. Additionally, agents in AutoGen support making LLM-suggested function calls, leveraging the new feature of OpenAI models to use pre-defined toolsets.

A unique feature in AutoGen is its ability to separate code generation and execution across different agents, i.e., using one agent to generate code or function calls and another agent to execute the code or function calls and respond. This separation simplifies simulation of real-world conversational processes involving multiple steps, such as human or AI review, API key replacement, or debugging, through potentially multi-turn conversations. It also allows developers to easily switch or customize different back-ends (e.g., execution environments) for each agent, taking advantage of the modularity and customizability of our agent design.

In practice, based on the specific needs of an application, each agent can have a mix of basic back-end types to display complex behavior in multi-agent conversations. For example, the user proxy agent in Figure 2 is backed by both humans and tools. When a human wants to skip providing feedback, a Python code executor is called upon to reply the execution results or error messages.

#### Agent customization.

With built-in support for agent capabilities, AutoGen allows one to easily create agents with specialized capabilities and roles. When building a multi-agent system, one can either directly reuse the built-in agents in AutoGen or develop customized agents based on them. Figure 3 provides a sketch of the built-in agents in AutoGen. The ResponsiveAgent class, by default, can use LLMs, humans, and tools. The AssistantAgent and UserProxyAgent are two pre-configured versions of ResponsiveAgent, each representing a common usage mode. The AssistantAgent is designed to act as an AI assistant, using LLMs by default but not requiring human input or code execution. The UserProxyAgent is conceptually a proxy agent for humans, soliciting human input as the agent’s reply at each interaction turn by default and also having the capability to execute code and call functions.

More technical details about the features, usage, and customization of agents in AutoGen can be found via https://microsoft.github.io/FLAML/docs/Use-Cases/Autogen#agents.

### 2.2 Multi-Agent Conversations

One main insight of AutoGen is to solve tasks via inter-agent conversations. With the goal of enabling next-gen applications, we face the challenge of finding a simple, unified approach to facilitate easy orchestration of complex workflows. We introduce the following designs to tackle this challenge: (1) intuitive and unified conversation interfaces; (2) automated agent chat via auto-reply; and (3) generic support of diverse conversation patterns.

#### Multi-agent conversation via unified conversation interfaces.

Agents in AutoGen have unified conversation interfaces, including send/receive for sending/receiving messages and generate_reply for generating a reply based on the received message. With this conversation-centric design, a workflow can be represented as a sequence of inter-agent message passing and agent acting (programmed in generate_reply). Once the message passing and acting logic are clearly specified, the workflow is defined. This provides an intuitive and unified way to characterize and reason about a workflow as inter-agent conversations. This design also aligns well with the capabilities of chat-optimized LLM like GPT-4. The LLM-backed agents can use the chat history to perform LLM inference, which automatically infers the current state and respond. That can greatly reduce the need for complex state management which can be a major burden of development. We hypothesize that this is a key approach to simplify a traditional complex workflow in general: a developer can design a system in which LLM-backed agents drive complex workflows based on natural language instructions and chat history, while tool-backed agents handle structured instructions supported by programming languages. The design of generate_reply in AssistantAgent and UserProxyAgent exemplifies this insight.

*Figure 4: Illustration of an automated agent chat that uses a custom auto-reply. Here three agents coordinate writing code, execution, and JSON validation. After defining the participating agents A, B, and C, the developer only needs to register auto-reply functions that specify how an agent should respond to the other agent’s message and initiate agent conversations. AutoGen provides basic built-in auto-reply functions and also allows developers to register custom functions. For example, reply_func_A2B (green) is a custom auto-reply function where agent A can execute code from agent B and ask C for help to return a valid JSON response.*

#### Automated multi-agent conversation with registered auto-reply.

To ease the development of multi-agent conversation, we aim to reduce the developers’ effort to only defining the behavior of each agent. That is, once agents are appropriately configured, the developer can readily trigger the conversation among the agents and the conversation would proceed automatically with no extra effort of the developer for crafting a control plane. Towards that goal, we introduce and, by default, adopt an agent auto-reply mechanism to enable automated multi-agent conversation: Once an agent receives a message from another agent, it automatically invokes generate_reply and sends the reply back to the sender unless it is empty (e.g., when a termination condition is satisfied). This simple mechanism enables automated agent chat, i.e., the agent conversation can proceed automatically after the conversation is initialized, without any extra control plane needed. This is because it realizes the automatic binding between an agent’s action-taking and the subsequent interactions with the other agents and therefore naturally links the sequences of agent message-passing and action-taking flows. The auto-reply mechanism provides a decentralized, modular and unified way to define the workflow. One can register customized auto-reply functions (triggered by generate_reply according to registered conditions) to define the behavior pattern of an agent, e.g., using LLM, tool or chatting with another agent before replying. If a developer needs to introduce an additional agent to an existing agent workflow, the developer only needs to program the added agent and modifies the agents that may have conversations with it, as illustrated in Figure 4.

#### Supporting diverse conversation patterns.

In practice, different applications may involve different workflows and thus may require different conversation patterns. The unified design in AutoGen supports a diverse range of conversation patterns with working systems implemented to showcase such patterns, with some of them visualized in Figure 5. Here we discuss a few supported patterns regarding several notable aspects and their applications (ref. Section 4):

-

Conversations with different autonomisity, and human involvement patterns.

-

Fully autonomous conversations: One can achieve fully autonomous conversations with a minimum initialization step in AutoGen. In particular, one only needs to set the ‘human_input_mode’ to ‘NEVER’ in agents and register auto-reply functions properly (or use the built-in registered auto-reply). This type of fully autonomous conversation can be found useful in applications where autonomous problem-solving is needed. We showcase this fully autonomous conversation pattern in A1: Math problem solving, A2: Multi-agent coding, A3: Online decision making, A4: Retrieval-augmented chat and A6: Conversational chess.

-

Conversations with humans in the loop: In many applications, human involvement is expected and/or desired. Examples include cases where humans are an intrinsic and essential part of a workflow, cases where human authority is needed, and cases where human hints are desired to leverage human intelligence. We showcase working systems with humans in the loop with several different involvement patterns.

- In A1: Math problem solving, we show how AutoGen can be used to do human-in-the-loop problem-solving with two different types of human involvement patterns (in addition to a fully autonomous pattern). In scenario 2, a human user observes every round of the interaction and gives hints in some of the rounds if needed. Scenario 3 demonstrates a conversation pattern where multiple humans can participate in the conversations.

- In A4: Retrieval-augmented chat, human can give feedback to Retrieve Assistant agent when human input is enabled.

- In A5: Dynamic group chat and A6: Conversational chess, humans can participate as a group chat participant or a chess player through a user proxy agent to interact with the other AI agents respectively.

-

Static and dynamic conversations. Dynamic conversation allows the agent topology to change depending on the actual flow of conversation under different input problem instances, while the flow of a static conversation always follows a pre-defined topology. The dynamic conversation pattern is useful in complex applications where the patterns of interaction cannot be predetermined in advance. The support of dynamic conversation patterns is rare in existing multi-agent LLM systems. AutoGen provides two general ways to achieve it and also working systems with those dynamic conversation patterns:

-

Registered auto-reply: With the pluggable auto-reply function, one can choose to invoke conversations with other agents depending on the content of the current message and context. E.g., in example demonstrated in Figure 4, a third agent (agent C) could be involved in the conversation with A and B depending on the conversation context. A working system demonstrating this type of dynamic conversation can be found in A5: Dynamic group chat. In the system, we register an auto-reply function in the group chat manager, which lets LLM decide who will the next speaker be in a group chat setting. Depending on the topic of the group chat discussion, one speaker may or may not be selected to speak. This realizes dynamic conversation among the group chat participants based on the chat flow.

-

Function call: In this approach, LLM decides whether or not to call a particular function depending on the conversation status in each inference call. By messaging additional agents in the called functions, the LLM can drive dynamic multi-agent conversation. A working system showcasing this type of dynamic conversation can be found in scenario 3 of A1: Math problem solving, where a student assistant would automatically resort to an expert using function calls.

## 3 Related Work

We examine existing LLM-based agent systems or frameworks that can be used to build LLM applications. We categorize the current work into single-agent and multi-agent systems, and specifically provide a summary of differentiators comparing AutoGen with existing multi-agent systems in Table 1. Note that many of these systems are evolving open-source projects, so the remarks and statements about them may only be accurate as of the time this paper was written.

#### Single-Agent Systems:

-

Auto-GPT: Auto-GPT is an open-source implementation of an AI agent that could autonomously achieve a given goal [5]. It follows a single-agent paradigm in which it augments the AI model with many useful tools and does not support multi-agent collaboration.

-

ChatGPT+ (with code interpreter or plugin): ChatGPT, a conversational AI service or agent, can now be used alongside a code interpreter or plugin (currently available only under the premium subscription plan ChatGPT Plus) [36]. The code interpreter enables ChatGPT to execute code, while the plugin enhances ChatGPT with a wide range of curated tools.

-

LangChain Agents: LangChain is a general framework for developing LLM-based applications [24]. LangChain Agents is a subpackage for using an LLM to choose a sequence of actions. There are various types of agents in LangChain Agents, with the ReAct agent being a notable example that combines reasoning and acting when using LLMs (mainly designed for LLMs prior to ChatGPT) [53]. All agents provided in LangChain Agents follow a single-agent paradigm and are not inherently designed for communicative and collaborative modes. A significant summary of its limitations can be found in [51]. Due to these limitations, even the multi-agent systems in LangChain (e.g., reimplementation of CAMEL) are not based on LangChain Agents but are implemented from scratch. Their connection to LangChain lies in the use of basic modules provided by LangChain, such as AI models wrapped by LangChain and the corresponding interface.

-

Transformers Agent: Transformers Agent [17] is an experimental natural language API on top of transformers. It includes a set of curated tools and an agent to interpret natural language and to use these tools. Similar to Auto-GPT, it follows a single-agent paradigm and does not support agent collaboration.

#### Multi-Agent Systems:

-

BabyAGI: BabyAGI [6] is an example implementation of an AI-powered task management system in a Python script (according to its own documentation). In this implemented system, multiple LLM-based agents are used. For example, there is an agent for creating new tasks based on the objective and the result of the previous task, an agent for prioritizing the task list, and an agent for completing tasks/sub-tasks. BabyAGI is a multi-agent system with a static agent communication pattern, i.e., a predefined order of agent communication.

-

CAMEL: CAMEL [28] is a communicative agent framework. It demonstrates how role-playing can be used to let chat agents communicate with each other for task completion. It also records agent conversations for behavior analysis and capability understanding. CAMEL does not support tool-using, such as code execution. Inception prompting technique is used to achieve autonomous cooperation between agents.

-

MetaGPT: MetaGPT [20] is a multi-agent framework for assigning different roles to GPTs to form a collaborative software entity for complex tasks. It is a specialized LLM-based multi-agent framework for collaborative software development.

-

Multi-Agent Debate: Two recent works investigate and show that multi-agent debate is an effective way to encourage divergent thinking in LLMs [29] and to improve the factuality and reasoning of LLMs [16]. In both works, multiple LLM inference instances are constructed as multiple agents to solve problems with agent debate. Each agent is simply an LLM inference instance, while no tool or human is involved. The conversation also needs to follow a pre-defined order.

There are a few other specialized single-agent or multi-agent systems, such as Voyager [49] and Generative Agents [38], which we skip due to lower relevance.

*Table 1: Summary of differences between AutoGen and other related multi-agent systems. In this table, the “infrastructure” aspect evaluates whether the system is designed as a generic infrastructure for building new LLM applications; the “conversation pattern” aspect assesses the types of patterns supported by the implemented systems; the “execution-capable” aspect determines whether the system can execute LLM-generated code; the “human involvement” aspect evaluates whether (and how) the system allows human participation during the execution process of the system. *

| Aspect | AutoGen | Multi-agent Debate | CAMEL | BabyAGI | MetaGPT |

| Infrastructure | ✓ | ✗ | ✓ | ✗ | ✗ |

| Conversation pattern | static/dynamic | static | static | static | static |

| Execution-capable | ✓ | ✗ | ✗ | ✗ | ✓ |

| Human involvement | chat/skip | ✗ | ✗ | ✗ | ✗ |

## 4 Applications of AutoGen

We showcase six applications powered by AutoGen (depicted in Figure 5). These applications were selected to demonstrate how AutoGen can assist developers in tackling a range of real-world tasks.

-

A1: Math problem solving solves math problems in three scenarios

-

A2: Multi-agent coding uses three agents to solve challenging supply chain optimization problems

-

A3: Online decision making uses agents to solve web interaction tasks in the MiniWob++ benchmark

-

A4: Retrieval-augmented chat uses retrieval augmented agents to solve code generation and question-answering problems

-

A5: Dynamic group chat showcases how to build a versatile group chat with AutoGen

-

A6: Conversational chess implements a fun conversational chess game where players can creatively express moves

Code and notebook demonstration for each of these applications can be found at https://microsoft.github.io/FLAML/docs/Examples/AutoGen-AgentChat/

*Figure 5: We present six examples of diverse applications built using AutoGen. Their conversation patterns show AutoGen’s flexibility and power.*

### A1: Math Problem Solving

*Figure 6: Examples of three settings we used to solve math problems using AutoGen. (Gray) Enables implementing a workflow where a single “student” works with a student assistant to solve problems in either autonomous mode or human-in-the-loop mode. (Gray + Orange) Enables a more sophisticated workflow where the student assistant can on the fly ask another user called “expert”, who is in the loop with its own agent, to help solve problems if its own solutions are not satisfactory. *

Mathematics is a foundational discipline that permeates nearly every aspect of our understanding and interaction with the world. The promise of leveraging LLMs to assist math problem solving opens up a new plethora of applications and avenues for exploration, including personalized AI tutoring, AI research assistance, cross innovations, etc. The built-in agents in AutoGen can be directly used for solving math problems. In this subsection, we evaluate AutoGen’s performance on challenging math problems with autonomous problem solving (scenario 1), and human-in-the-loop problem solving (scenario 2). We further demonstrate a novel scenario where multiple human users can participate in the conversations during the problem solving process (scenario 3).

For the evaluation, we test on randomly selected level-5 problems from the MATH dataset [19]. We compare AutoGen with the following LLM-based agent systems. For all evaluations, we use GPT-4 as the base model, and pre-install the “sympy” package in the execution environment.

-

Auto-GPT: The out-of-box Auto-GPT is used. We initialized Auto-GPT by setting the purpose to “solve math problems”, resulting in a “MathSolverGPT” with auto-generated goals.

-

ChatGPT+Plugin: We enabled the Wolfram Alpha plugin (a math computation engine) in the OpenAI web client.

-

ChatGPT+Code Interpreter: This is a recent feature in OpenAI web client. Note that the above two premium features from ChatGPT require a paid subscription to be accessed and are the most competitive commercial systems.

-

LangChain ReAct+Python: We use the zero-shot ReAct Agent from LangChain. We added a LangChain code executor, “PythonREPL” to the agent as a tool.

*Table 2: Qualitative evaluation on two math problems from the MATH dataset under the autonomous problem solving setting. Each LLM-based system is tested three times on each of the problems. We report the problem solving correctness and summarize the failure reasons in this table.*

| | Correctness | Failure Reason |

| AutoGen | 3/3 | N/A. |

| Auto-GPT | 0/3 | The LLM gives code without the print function so the result is not printed. |

| ChatGPT+Plugin | 0/3 | The return from Wolfram Alpha contains 2 simplified results, including the correct answer, but GPT-4 always chooses the wrong answer. |

| ChatGPT+Code Interpreter | 2/3 | Returns a wrong decimal result. |

| LangChain ReAct | 0/3 | LangChain gives 3 different wrong answers. |

*(a) Evaluation on the first problem that asks to simplify a square root fraction.*

| | Correctness | Failure Reason |

| AutoGen | 2/3 | The final answer from code execution is wrong. |

| Auto-GPT | 0/3 | The LLM gives code without the print function so the result is not printed. |

| ChatGPT+Plugin | 1/3 | For one trial, GPT-4 got stuck because it keeps giving wrong queries and has to be stopped. Another trial simply gives a wrong answer. |

| ChatGPT+Code Interpreter | 0/3 | It gives 3 different wrong answers. |

| LangChain ReAct | 0/3 | LangChain gives 3 different wrong answers. |

*(b) Evaluation on the second number theory problem.*

#### Scenario 1: Autonomous Problem Solving.

We use two level-5 problems from the MATH dataset for the qualitative evaluation and test on each problem for 3 times. The first problem asks to simplify a square root fraction and the second problem asks to solve a number theory problem. Using this setup, we analyzed the correctness of results, user experience regarding the verbosity of response and whether the LLM-based system can run without unexpected behaviors. Table 2 details correctness counts and failure reasons.

-

Correctness: We observe that AutoGen solves the two problems correctly. ChatGPT+Code Interpreter fails to solve the second problem. ChatGPT+Plugin fails on the first problem, because the model cannot choose the correct answer from the plugin’s response. Auto-GPT fails on both problems due to code execution issues. LangChain agent also fails on both problems. For all trials, it gives code that results in the wrong answer. This shows that AutoGen is competitive compared to OpenAI’s paid official add-ons, while Auto-GPT and LangChain ReAct are less desirable.

-

User experience: ChatGPT+Plugin is the least verbose mainly because Wolfram queries are much shorter than Python code. AutoGen, ChatGPT+Code Interpreter, and LangChain are similar, while LangChain is slightly more verbose with more code execution errors. Auto-GPT is the most verbose system because of the predefined steps like THOUGHTS, REASONING, and PLAN. It will reply with these steps every time. Overall, AutoGen and LangChain+Code Interpreter can run smoothly without execptions. We note the occurrences of undesired behaviors from other LLM-based systems that could affect user experience: Auto-GPT keeps giving code without the ‘print’ statement and cannot correct it, so the user needs to run them manually; ChatGPT with Wolfram Alpha plugin has the potential to stuck in a loop that needs to be manually stopped; and Langchain ReAct could exit with a parse error.

As a followup, we also conduct a more comprehensive assessment of the correctness rate of AutoGen, ChatGPT+Plugin, and ChatGPT+Code Interpreter (Auto-GPT and LangChain ReAct are excluded because they are less competitive based on the qualitative analysis). We sample 12 level-5 problems from the MATH dataset. With one trial, AutoGen scores 8/12 problems, and both ChatGPT+Plugin and ChatGPT+Code Interpreter score 7/12 problems. This confirms that AutoGen has competitive performances compared to the paid commercial systems.

#### Scenario 2: Human-in-the-loop Problem Solving.

For the hard problems that these LLM systems cannot solve autonomously, human feedback during the problem solving process can be helpful. To incorporate human feedback with AutoGen, one can set human_input_mode=‘ALWAYS’ in the user proxy agent. We compare such configuration of AutoGen with systems that could also incorporate human feedback during the problem solving process, including Auto-GPT, ChatGPT+Plugin, and ChatGPT+Code Interpreter. We select one problem that cannot be solved by any of these systems autonomously in 3 trials. Then we follow the process below to incorporate human inputs in all the compared methods, which involves providing some important hints:

-

Input the problem: Find the equation of the plane which bisects the angle between the planes $3x-6y+2z+5=0$ and $4x-12y+3z-3=0,$ and which contains the point $(-5,-1,-5).$ Enter your answer in the form

$Ax+By+Cz+D=0,$ | | | |

where $A,$ $B,$ $C,$ $D$ are integers such that $A>0$ and $\gcd(|A|,|B|,|C|,|D|)=1.$

-

The response from the system does not solve the problem correctly. We then give a hint to the model: Your idea is not correct. Let’s solve this together. Suppose $P=(x,y,z)$ is a point that lies on a plane that bisects the angle, the distance from P to the two planes is the same. Please set up this equation first.

-

We expect the system to give the correct distance equation. Since the equation involves an absolute sign that is hard to solve, we would give the next hint: Consider the two cases to remove the abs sign and get two possible solutions.

-

If the system returns the two possible solutions and don’t continue to the next step, we give the last hint: Use point (-5,-1,-5) to determine which is correct and give the final answer.

-

Final answer is .

During our evaluation we observed that AutoGen consistently solves the problem in all 3 trials. Both ChatGPT+Code Interpreter and ChatGPT+Plugin managed to solve the problem in 2 out of 3 trials and AutoGPT fails to solve it all 3 times. In its unsuccessful attempt, ChatGPT+Code Interpreter failed to follow our instructions to walk through the step of removing the absolute sign. ChatGPT+Plugin, in its failed trial, produced an almost correct solution that had a sign discrepancy in the final answer. In contrast, AutoGPT was unable to yield a correct solution in any of the trials. In one trial, it derived an incorrect distance equation. In the other two trials, it formulated the distance equation and followed provided hints, but the final answer was wrong from code execution.

Auto-GPT is still the most verbose system in this evaluation. The ChatGPT+Plugin and ChatGPT+Code Interpreter are more verbose than AutoGen. Their request to external tools is automatic, so they would keep sending queries and try to revise them as long as there are errors. Since this problem is hard and very likely to result in wrong queries, we have to wait for a long time for the system to finish trying all the queries. However, with human input enabled in AutoGen, people a chance to examine the code and give feedback timely before the code execution.

#### Scenario 3: Multi-User Problem Solving.

The next-gen LLM applications may require the system to involve multiple real users for collectively solving a problem with the assistance of LLMs. We showcase how AutoGen can be leveraged to build such a system with ease. In particular, built upon setting 2 above, we aim for a simple system that involves two human users including a student and an expert. In this system, the student interacts with an LLM assistant to solve some problems and the LLM would automatically resort to the expert when it is necessary.

The overall workflow is as follows. Following the default setting, the student chats with the LLM-based assistant agent through a student proxy agent to solve problems. When the student’s assistant couldn’t solve the problem satisfactorily or the solution does not match the expectation of the student, it would automatically hold the conversation and call the pre-defined ask_for_expert function via the function_call feature of GPT in order to resort to the expert. Specifically, it would automatically produce the initial message for the ask_for_expert function, which could be the statement of the problem or the request of verifying the solution to a problem, and the expert is supposed to respond to this message with the help of the expert assistant. After the conversation between the expert (represented by the expert proxy agent) and the expert assistant, the final message would be sent back to the student assistant as the response to the initial message. Then, the student assistant would resume its conversation with the student using the response from the expert for a better solution. The simplified visualization of implementation is shown in Figure 6.

With AutoGen, one can readily construct the student/expert proxy agent and the assistant agents by reusing the built-in UserProxyAgent and AssistantAgent through appropriate configurations. The only development required is several lines of code for the ask_for_expert function, which is then part of the configuration for the assistant for the student. In addition, one can easily extend such a system to include more than one experts with one specific ask_for_expert function for each, or include multiple student users with a shared expert to resort to.

#### Takeaways:

We evaluated AutoGen’s performance both qualitatively and quantitatively on complex mathematical problems. Compared to other alternatives, AutoGen stands out as the most reliable open-source system in terms of accuracy and user experience. Its performance is also on par with OpenAI’s paid premium add-ons. We demonstrate that humans can easily collaborate with AI to solve problems, and AutoGen enables interactive user instructions to keep the process on the desired path. Moreover, we show that this workflow can be easily extended to include multiple users and LLM-backed agents working together. Apart from math problems, AutoGen’s built-in agents can tackle generic problems, including those requiring code execution. This usage of AutoGen can produce an experience similar to ChatGPT+Code Interpreter+Plugin, enhanced by adjustable automation and human control, flexibility to switch between human input and auto-reply, portability to larger systems, customizable execution environments, and programmable chat. Several illustrative examples are available at https://microsoft.github.io/FLAML/docs/Examples/AutoGen-AgentChat/.

### A2: Multi-Agent Coding

*Figure 7: Our re-implementation of OptiGuide with AutoGen streamlining agents’ interactions. The Commander receives user questions (e.g., What if we prohibit shipping from supplier 1 to roastery 2?) and coordinates with the Writer and Safeguard. The Writer crafts the code and interpretation, the Safeguard ensures safety (e.g., not leaking information, no malicious code), and the Commander executes the code. If issues arise, the process can repeat until resolved. Shaded circles represent steps that may be repeated multiple times.*

In this subsection, we focus on OptiGuide [27], a framework that excels at writing code to interpret optimization solutions and answer users’ questions, such as exploring the implications of changing a supply chain decision or understanding why the optimizer made a particular choice. Using LLMs and external tools, OptiGuide effectively addresses user queries related to supply chain applications. Without LLMs, answering these questions would require substantial human effort, often involving the question being passed from department to department and eventually to engineers who would write analysis code to provide the answers. OptiGuide was originally implemented using text completion APIs from Azure OpenAI. But in this section, we show how AutoGen provides a multi-agent design that greatly and elegantly simplifies its implementation. Figure 7 shows the AutoGen-based implementation.

#### Workflow.

The workflow can be described as follows. The user agent initiates the interaction by posing the user’s question, such as “What if we prohibit shipping from supplier 1 to roastery 2?”, marked by to the Commander agent. The Commander manages and coordinates with two LLM-based assistant agents: the Writer and the Safeguard. Apart from directing the flow of communication, the Commander has the responsibility of handling memory tied to user interactions. This capability enables the Commander to capture and retain valuable context regarding the user’s questions and their corresponding responses. Such memory is subsequently shared across the system, empowering the other agents with context from prior user interactions, ensuring more informed and relevant responses.

In this orchestrated process, the Writer, who combines the functions of a “Coder” and an “Interpreter” as defined in [27], will craft code and also interpret execution output logs. For instance, during code writing ( and ), the Writer may craft code “model.addConstr(x[‘supplier1’, ‘roastery2’] == 0, ‘prohibit’)” to add an additional constraint to answer the user’s question.

After receiving code, the Commander will communicate with the Safeguard to screen the code and ascertain its safety (); once the code obtains the Safeguard’s clearance, marked by , the Commander will use external tools (e.g., Python) to execute the code, and request the Writer to interpret the execution results for the user’s question ( and ). For instance, the writer may say “if we prohibit shipping from supplier 1 to roastery 2, the total cost would increase by 10.5%.” Bringing this intricate process full circle, the Commander furnishes the user with the concluding answer ().

However, if at a point there is an exception - either a security red flag raised by Safeguard (in ) or code execution failures within Commander, the Commander redirects the issue back to the Writer with essential information in logs (). So, the process from to might be repeated multiple times, until each user query receives a thorough and satisfactory resolution or until the timeout. This entire complex workflow of multi-agent interaction is elegantly automated via AutoGen agent chat.

The core workflow code for OptiGuide was reduced from over 430 lines to 100 lines using AutoGen, leading to significant productivity improvement. The new agents are customizable, conversable, and can autonomously manage their chat memories. This consolidation allows the coder and interpreter roles to merge into a single “Writer” agent, resulting in a clean, concise, and intuitive implementation that is easier to maintain.

Table 4 and 5 in the appendix provide a comparison of user experience with ChatGPT+Code Interpreter and AutoGen-based OptiGuide. ChatGPT+Code Interpreter is unable to run code with private packages or customized dependencies (such as Gurobi); as a consequence ChatGPT+Code Interpreter requires users to have engineering expertise and manually handle multiple steps, disrupting the workflow and increasing the chance for mistakes. If customers lack access or expertise, the burden falls on supporting engineers, increasing their on-call time. In contrast, the automated chat by AutoGen is more streamlined and autonomous, integrating multiple agents to solve problems and address concerns. This results in a 5x reduction in interaction and fundamentally changes the overall usability of the system. A stable workflow can be potentially reused for other applications or to compose a bigger one.

#### Takeaways:

The implementation of the multi-agent design with AutoGen in the OptiGuide app offers several advantages. It simplified the Python implementation. It fostered a mixture of collaborative and adversarial problem-solving environment, with the Commander and Writer working together while the Safeguard acts as a virtual adversarial checker. This setup allows for proper memory management, as the Commander maintains memory related to user interactions, providing context-aware decision-making. Additionally, role-playing ensures that each agent’s memory remains isolated, preventing shortcuts and hallucinations.

### A3: Online Decision Making

*Figure 8: We use AutoGen to solve tasks in the MiniWob++ benchmark which contains browser interaction tasks described in natural language, such as “expand the web section below and click the submit button.” Our implementation (MiniWobChat) used two agents: The built-in AssistantAgent and an Executor agent. The AssistantAgent suggests actions to manipulate the browser and the Executor executes the suggested action and returns reward/feedback. The AssistantAgent records feedback and continues until the feedback indicates task success/failure. *

In practice, many applications demand the presence of agents capable of interacting with environments and making decisions in an online context, such as game playing [34, 46], web interactions [30, 45] and robot manipulations [44]. With the multi-agent conversational framework in AutoGen, it is easy to decompose the automatic agent-environment interactions and the development of decision-making agent by constructing an Executor agent responsible for handling the interaction with the environment, therefore splitting out the decision-making part to other agents. Such a decomposition allows developers to reuse the decision-making agent for new tasks with minimal effort rather than building a specialized decision-making agent for every new environment.

#### Workflow.

We demonstrate how to use AutoGen to build a working system to handle such scenarios with the MiniWoB++ benchmark [45]. MiniWoB++ comprises browser interaction tasks that involve utilizing mouse and keyboard actions to interact with browsers. The ultimate objective of each task is to complete tasks described concisely in natural language, such as “expand the web section below and click the submit button.” Solving these tasks typically requires a sequence of web manipulation actions rather than one single action, and making action decisions at each time step requires access to the web status (in the form of HTML code) online. For the example above, clicking the submit button requires getting the status of the web status after expanding the web section. We designed a straightforward two-agent system named MiniWobChat using AutoGen as shown in Figure 8. The first agent uses the built-in AssistantAgent and is responsible for making action decisions for the given task, while the second agent, the Executor agent, is a customized UserProxyAgent to interact with the benchmark by executing the actions suggested by the AssistantAgent and returning feedback.

To assess the developed working system’s performance, we make a comparison with RCI [22], a recent solution for MiniWoB++ benchmark that employs a set of self-critiquing prompts and has achieved the state-of-the-art performance. In our evaluation, we specifically selected a set of clicking tasks used in RCI with varying degrees of difficulty to conduct a comparative analysis against MiniWobChat. Figure 9 illustrates that MiniWobChat achieves competitive performance in this evaluation. Specifically, among the twelve clicking tasks, MiniWobChat achieves better results in four tasks and achieved parity in performance in three tasks.

Additionally, we also explored the feasibility of using Auto-GPT for handling the same tasks. Auto-GPT faces challenges in handling tasks that involve complex rules due to its limited extensibility. It provides an interface for setting task goals using natural language. However, when dealing with the MiniWob++ benchmark, accurately instructing Auto-GPT to follow the instructions for using Miniwob++ is challenging. There is no clear path to extend it like the two-agent chat by AutoGen.

#### Takeaways:

For this application, AutoGen stood out as a more user-friendly option, offering modularity and programmability: It streamlined the process with autonomous conversations between the Assistant and the Executor and provided readily available solutions for agent-environment interactions. The built-in AssistantAgent was directly reusable and presented strong performance without customization. Moreover, the decoupling of the execution and assistant agent ensures that modifications to one component do not adversely impact the other. This convenience makes maintenance and future updates straightforward.

*Figure 9: Comparisons between RCI (state-of-the-art prior work) and MiniWobChat on the MiniWob++ benchmark. We selected a set of clicking tasks in MiniWob++ with varying degrees of difficulty. For each clicking task, we reported the success rate for ten different instances. The results show that MiniWobChat achieves comparable performance with RCI. *

### A4: Retrieval Augmented Code Generation and Question Answering

*Figure 10: Overview of Retrieval-augmented Chat which involves two agents, including a Retrieval-augmented User Proxy and a Retrieval-augmented Assistant. Given a set of documents, the Retrieval-augmented User Proxy first automatically processes documents – splits, chunks, and stores them in a vector database. Then for a given user input, it retrieves relevant chunks as context and sends it to the Retrieval-augmented Assistant, which uses LLM to generate code or text to answer questions. Agents converse until they find a satisfactory answer. *

Retrieval augmentation has emerged as a practical and effective approach for mitigating many intrinsic limitations of LLM in many applications by incorporating external documents. In this section, we employ AutoGen to build a Retrieval Augmented Generation (RAG) system for code generation and question-answering [26, 39]. Retrieval Augmented Generation (RAG) combines the advantages of retrieval-based and generation-based models in natural language processing, and has practical applications in chatbots, question answering, code generation, and natural language interfaces. It is specifically designed to address complex open-domain questions that necessitate knowledge derived from massive document collections. The main principle of the RAG architecture is to use a retriever to fetch pertinent documents from a vast corpus according to a user’s query, followed by a generator that produces an answer to the query by using both the retrieved documents and the original query. Such workflow could handle ambiguities in words and phrases and supply pertinent information beyond the explicit question asked and has achieved state-of-the-art performance in several benchmark tests [10, 32, 33], including the Natural Questions (NQ) task [23].

The RAG system we developed for code generation and question answering using AutoGen is called Retrieval-augmented Chat. It consists of two agents: Retrieval-augmented User Proxy agent and Retrieval-augmented Assistant agent, which are extended from UserProxyAgent and AssistantAgent of AutoGen, respectively. The Retrieval-augmented User Proxy includes an additional vector database [12] with SentenceTransformers [42] as the context retriever.

#### Workflow.

The workflow of Retrieval-augmented Chat is sketched in Figure 10. To use Retrieval-augmented Chat, one needs to initialize two agents including Retrieval-augmented User Proxy and Retrieval-augmented Assistant. Initializing the Retrieval-augmented User Proxy requires a path to the document collection. The Retrieval-augmented User Proxy can then downloads the documents, split them into chunks of a specific size, computes embeddings, and store them in a vector database. Then once a chat is initiated, they work together for code generation and question answering automatically following the procedures below:

-

The Retrieval-augmented User Proxy retrieves document chunks based on the embedding similarity. It sends them along with the question to the Retrieval-augmented Assistant.

-

The Retrieval-augmented Assistant uses LLM to generate code or text as answers based on the question and context. The LLM is instructed to reply “Update Context” to the Retrieval-augmented User Proxy if it is unable to produce a satisfactory response.

-

If a response includes code blocks, Retrieval-augmented User Proxy executes the code and sends the output as feedback. If there are no code blocks or instructions to update context, it terminates the conversation. Otherwise, it updates the context and sends the question along with the new context to the Retrieval-augmented Assistant. Note that if human input solicitation is enabled humans can proactively send any feedback including “Update Context” to the Retrieval-augmented Assistant.

-

If the Retrieval-augmented Assistant receives “Update Context,” it requests new context from Retrieval-augmented User Proxy. Otherwise, it generates new code or text based on the feedback and chat history. If LLM fails to generate the answer, it replies “Update Context” again. This process can repeat several times. The conversation terminates if no more documents are available for context.

#### Case Study of Retrieval-augmented Chat Usages.

We utilize Retrieval-augmented Chat in two scenarios. The first scenario aids in generating code based on a given codebase. While LLMs possess strong coding abilities, they are unable to use packages or APIs that are not included in their training data, e.g., private codebase, or have trouble using trained ones that are frequently updated after training. Hence, Retrieval Augmented Code Generation is considered to be highly valuable. The second scenario is to do QA on the Natural Questions dataset [23]. This enables us to obtain comparative evaluation metrics for the performance of our system.

Scenario 1: Leverage latest APIs in FLAML. In this case, the question is “How can I use FLAML to perform a classification task and use spark to do parallel training. Train 30 seconds and force cancel jobs if time limit is reached.”. FLAML (v1) [48] is a lightweight Python library for efficient automation of machine learning and AI operations, including selection of models, hyperparameters, and other tunable choices of an application. It was open-sourced in December 2020, and is included in the training data of GPT-4. However, the question requires the use of Spark, which was added in December 2022 and is not included in the GPT-4 training data. As a result, the original GPT-4 model is unable to generate the correct code, as it lacks knowledge of Spark-related APIs. Instead, it creates a non-existent parameter, $spark$, and sets it to ‘True’. However, with Retrieval-augmented Chat, we provide the latest reference documents as context. Then, GPT-4 generates the correct code blocks by setting $use\_spark$ and $force\_cancel$ to ‘True’.

Scenario 2: Evaluation on Natural Questions QA dataset. In this case, we evaluate Retrieval-augmented Chat’s end-to-end question answering performance with the Natural Questions dataset [23]. We collect 5,332 nonredundant context documents and 6,775 queries from the HuggingFace Dataset. First, we create a document collection based on all the context corpus and store them in the vector database. Then we answer the questions with Retrieval-augmented Chat. Next, to evaluate the performance of Retrieval-augmented Chat in QA, we employ the metrics of exact match (EM), F1 score, and Recall. The EM score indicates the percentage of questions where the predicted answer matches the reference answer to the question exactly. On the other hand, the F1 score measures the similarity between the predicted answer and the reference answer, taking into account both precision and recall. And recall measures the proportion of tokens in the reference answer that are present in the predicted answer. Results on the first 500 questions with GPT-3.5-turbo are shown in Table 3. The metrics of other solutions are directly copied from the reference paper [2]. DPR (Dense Passage Retrieval) is the retriever used in the paper. The checkpoints of FiD and Flan-T5 used in the paper are trained with the NQ dataset, and that is why they have much higher EM values than other models. [2] mentions that recall shows a stronger correlation with human-judged correctness than EM and F1. According to Recall, Retrieval-augmented Chat significantly outperforms other solutions.

*Table 3: Performance of Retrieval-augmented Chat and other retrieval-augmented solutions on Natural Questions QA dataset. Among the reported metrics, Recall shows the strongest correlation with human judgments. Based on this metric, Retrieval-augmented Chat outperforms other solutions.*

| Dataset | Solution | EM | F1 | Recall |

| Natural Questions | Retrieval-augmented Chat (ours) + GPT-3.5 | 0.2 | 23.73 | 66.61 |

| | DPR + GPT-3.5 | 1.27 | 15.12 | 58.56 |

| | DPR + Flan-T5 | 41.16 | 50.62 | 54.03 |

| | DPR + Alpaca | 8.78 | 20.3 | 46.23 |

| | DPR + Llama-2 | 0.61 | 11.85 | 52.37 |

| | Pretrained FiD | 46.57 | 53.93 | 54.45 |

#### Takeaways:

Retrieval-augmented Chat’s exceptional performance emanates from its unique and innovative feature: interactive retrieval. The Retrieval-augmented Assistant triggers the feature autonomously, requesting new context from the Retrieval-augmented User Proxy if it finds the given context inadequate to answer a question. Specifically, Retrieval-augmented Chat prompts the LLM to answer questions based on the given context. If the LLM can’t offer a satisfactory response based on the given context or its knowledge, it replies with “Update Context”. Upon receiving “Update Context”, the Retrieval-augmented User Proxy updates the documents in the context and resends the question along with the new context. The process enables this chat to access correct context documents, even when they were not initially retrieved, thus remedying inaccuracies in retrieval tools.

An example from the NQ dataset showcases the advantages of “Interactive Retrieval” feature: “who carried the usa flag in opening ceremony”. When answering this question, the context with the highest similarity to the question embedding does not contain the required information to answer the question. As a result, the LLM assistant (GPT-3.5-turbo) replies “Sorry, I cannot find any information about who carried the USA flag in the opening ceremony. UPDATE CONTEXT.” With the unique and innovative ability to update context in Retrieval-augmented Chat, the user proxy agent automatically updates the context and sends it to the assistant agent again. After this process, the agent is able to generate the correct answer to the question.

### A5: Dynamic Group Chat

*Figure 11: Overview of how AutoGen enables dynamic group chats to solve tasks: We use a special agent called the Manager that repeats the following three steps– select a single speaker (in this case Bob), ask the speaker to respond, and broadcast the selected speaker’s message to all the other agents. *

Group chat between agents and potentially humans can be beneficial for many practical scenarios. For example, in customer support scenarios, multiple agents can collaborate in a group chat to provide more comprehensive and efficient assistance to customers. Each agent can specialize in different areas or have different expertise, allowing them to handle different types of customer inquiries with personalized responses. Additionally, in collaborative problem-solving tasks, agents can engage in group chats with humans to share insights, collectively allocate tasks, and collectively work towards finding solutions. This can enable faster problem-solving. Overall, group chat can enhance collaboration, improve decision-making, and provide more effective and comprehensive support in various domains. In fact, existing work has shown the effectiveness of involving multiple agents in a group chat for solutions even with a pre-defined speaking order [16, 29]. In practice, developers may want to explore group chat with dynamic conversation patterns where the speaker does not have to follow a pre-defined order to participate. Alternatively, they participate in group chat based on the conversation context and their respective roles, which helps drive the conversation in a more autonomous manner. To achieve this dynamic group chat, one may need a conductor that organizes the group chat by specifying the next speaker and broadcasting its message to others.

AutoGen supports dynamic group chat via built-in GroupChatManager, which serves as the conductor of conversation among agents. The GroupChatManager repeats the following three steps: dynamic speaker selection, making the speaker respond, and broadcast. GroupChatManager selects the next speaker agent with a certain speaker selection policy (e.g., using LLM). The next speaker agent then generates a response and sends it to GroupChatManager as the next message of the group chat. GroupChatManager ensures the propagation of the most recent message from the sender agent to all other agents within the same group.

Among these three steps, dynamic speaker selection is the most tricky step. The GroupChatManager in AutoGen automatically selects the next speaker according to two important factors: the current context of the conversation and the alignment of roles, which ensures the conversation can be carried on by the most appropriate agent. With these considerations, the current implementation of GroupChatManager employs a policy based on role-play LLM prompting. It transforms the speaker selection task into a role-playing game with LLM where the agent is prompted to observe the conversations and choose a role to further the agenda of the conversation. We observe that compared with directly instructing the agent to provide the name of the next speaker in the second step, utilizing a role-play prompt often leads to more effective consideration of both conversation context and role alignment during the process of generating the subsequent speaker.

To further illustrate how GroupChatManager conducts group chat with dynamic speaker selection, we create a group chat example that contains three agents: User Proxy, Alice and Bob. Alice is an AssistantAgent for generic tasks. Bob is a customization of AssistantAgent whose role is to review code from Alice and suggest changes. (Figure 11) In this group chat, User Proxy starts the conversation on finding the latest paper about generative agents, which then is broadcasted to Alice and Bob. Alice is promoted to be the next speaker according to the conversation context because there’s no code for reviewing in the previous chat. And once Alice full-filled the request with Python code and execution instruction, Bob is selected as the next speaker to review Alice’s code. After Bob finishes the review, Alice again becomes the next speaker in the conversation to fix Bob’s comments on its code. After a few rounds of Alice-code-Bob-review interaction. Bob finally “approves” Alice’s code. And the Python code, along with the execution instruction are sent back to User Proxy for final execution. When execution error happened, the turn is back to Alice for debugging. Eventually, the collaboration results in success. (Figure 12)

#### Takeaways:

There are several advantages of using AutoGen to create a group chat. Firstly, AutoGen provides built-in GroupChatManager for dynamic group chat. GroupChatManager utilizes LLM to select the next speaker, which allows it to leverage conversation context and role information of group chat in a complete manner and contributes to a more natural, autonomous conversation pattern that is closer to a group chat in the real world. Secondly, with GroupChatManager, it is quite easy to customize group chat. Thanks to AutoGen agent design, GroupChatManager can work with any conversable agent which is an instance of ResponsiveAgent, such as AssistantAgent, UserProxyAgent, or any customized agent like those described in other applications. This enables GroupChatManager to support multi-human, multi-AI agent dynamic group chat patterns in an effortless way.

*Figure 12: The conversation example when User Proxy starts the conversation by asking a question on the latest paper about generative agents. Alice first tries to resolve User Proxy’s question using Python code. Consequently Bob is selected as the next speaker to review Alice’s Python code and leave comments. After a few rounds of Alice-code-Bob-review interaction, Bob finally “approves” the code from Alice, and then the code is successfully executed by User Proxy.*

### A6: Conversational Chess

*Figure 13: Our conversation chess application can support various scenarios, as each player can be an LLM-empowered AI, a human, or a hybrid of them. Here, the board agent maintains the rules of the game and supports the players with information about the board. Players and the board agent all use natural language for communication. *

*(a) Conversation between AI players*

*(b) Conversation between AI player (player white shown in blue) and board agent for making a new move.*

*Figure 14: Example conversations during a game between two AI players.*

Chess is arguably the most popular board game in the world: the online chess site chess.com has 1 million daily active users [1]. Currently, chess game engines only support structured chess protocols like Universal Chess Interface (UCI). To make chess games more entertaining and social, using AutoGen, we designed a new chess game, Conversational Chess, that supports pure natural language conversational interface. It opens up opportunities for social interaction and allows players to express their moves creatively, such as using jokes, meme references, and character-playing, therefore making chess games more entertaining to players as well as observers.

In Conversational Chess, each player is an AutoGen agent and can be either powered by a human or an AI. A third party that is a board agent is built to support players with information about the board and ground players’ moves to a legal chess move. Figure 13 shows the scenarios supported by the Conversational Chess: AI/human vs. AI/human, and how players and a board agent interact.

To realize these scenarios, we constructed a player agent with LLM and human as back-end options. When human input is enabled, before sending the input to the board agent, it first asks the human player to input the message that contains the move and anything else the player wants to say (such as a witty comment). If human input is skipped or disabled, LLM is used to generate the message. The board agent is implemented with a custom auto-reply function, which uses an LLM to parse the natural language input into a legal move in a structured format (e.g., UCI), and then pushes the move to the board. If the move is not legitimate, the board agent will reply with an error. Then the player agent needs to resend a message to the board agent until the move is legal. Once the move is successfully pushed, the player agent sends the message to the opponent. As in Figure 14, the conversation between AI players can be natural and entertaining. When the player agents use LLM to generate a message, it makes use of the board state and the error message from the board agent. That helps reduce the chance of hallucinating an invalid move. The chat between one player agent and the board agent is invisible to the other player agent, which keeps the messages used in chat completion well managed.

#### Takeaways:

There are two notable benefits of using AutoGen to implement Conversational Chess. Firstly, the agent design in AutoGen enables the natural creation of objects and their interactions needed in our chess game. This makes development easy and intuitive. For example, the isolation of chat messages makes it easy to make a proper LLM chat completion inference call. Secondly, AutoGen greatly simplifies the implementation of agent behaviors using composition. Specifically, we used the register_auto_reply method supported by AutoGen agents to instantiate player agents and a board agent with custom auto-reply functions. Restricting the extension work needed in a single point (the auto-reply function) simplifies the reasoning, development, and maintenance.

## 5 Discussion

The applications in Section 4 show how AutoGen can not only enable new applications but also renovate existing ones. For example, in A1 (scenario 3), A5, and A6, AutoGen enabled creating multi-agent chats that follow a dynamic pattern instead of a fixed back-and-forth. And, in both A5 and A6, humans can participate the activities together with multiple other AI agents in a conversational manner. Similarly A1 (the first two scenarios), A2, A3, and A4 show how popular applications can be renovated quickly with AutoGen. Despite the complexity of these applications (most of them involve more than two agents or dynamic multi-turn agent cooperation), our AutoGen-based implementation remains simple, demonstrating promising opportunities to build creative applications and a large space for innovation.

We further reflect on why we observed these benefits from AutoGen in these applications, and we believe there are a few reasons:

-

Ease of use: The built-in agents can be used out-of-the-box, presenting a strong performance even without any customization. (A1, A3, A5)

-

Modularity: The division of tasks into separate agents promotes modularity in the system. Each agent can be developed, tested, and maintained independently, simplifying the overall development process and facilitating code management. (A2, A3, A6)

-

Programmability: AutoGen allows users to extend/customize existing agents to develop systems satisfying their specific needs with ease. (A1-A6). For example, with AutoGen, the core workflow code in A3 is reduced from over 430 lines to 100 lines, bringing in a 4X saving.

-

Allowing human involvement: AutoGen provides a native mechanism to achieve human participation and/or human oversight. With AutoGen, humans can seamlessly and optionally cooperate with AI to solve problems or generally participate in the activity. AutoGen also facilitates interactive user instructions to ensure the process stays on the desired path. (A1, A4, A5, A6)

-

Collaborative/adversarial agent interactions: Like many collaborative agent systems (e.g., [14]), agents in AutoGen can share information and knowledge, allowing them to complement each other’s abilities and collectively arrive at better solutions. (A1, A2, A4, A5). Analogously, in certain scenarios some agents are required to work in an adversarial way. Relevant information is shared between different conversations in a controlled manner, preventing distraction or hallucination. (A2, A6). AutoGen supports both patterns enabling effective utilization and augmentation of LLM.

### 5.1 General Guidelines for Using AutoGen

Below we give some recommendations for using agents in Autogen to accomplish a task.

-

Consider using built-in agents first. For example, AssistantAgent is pre-configured to be backed by GPT-4, with carefully designed system message for generic problem solving via code. The UserProxyAgent is configured to be able to solicit human inputs and perform tool execution. When combining these two agents, many problems can be solved already with a decent performance. When customizing them for an application, consider the following options: (1) human input mode, termination condition, code execution configuration, LLM configuration are easy to specify when constructing an agent; (2) AutoGen supports adding hint in an initial user message, which is an effective way to boost performance without needing to modify the system message; (3) UserProxyAgent can be extended to handle different execution environments and exceptions, etc.; (4) when system message modification is needed, consider leveraging LLM’s capability to program its chat flow with natural language.

-

Keep the agent chat topology as simple as possible, as well as reduce code-based extension. Consider using the two-agent chat or the group chat setup first, as they require least code-based extension. Note that even the two-agent chat setup can involve more than two agents by using LLM-consumable functions in a dynamic way. When more control of the chat flow is needed and the desired control cannot be achieved by all the conversation programming suggested (tweaking system message and LLM-consumable functions), then one can consider registering custom auto-reply methods.

-

Try to reuse existing auto-reply methods based on LLM, tool or human before implementing the custom auto-reply methods, because they can often be reused to achieve the goal in a simple way (e.g., the built-in GroupChatManager’s auto-reply method reuses the LLM-based auto-reply when selecting the next speaker, ref. A5 in Section 4). Also, try to leverage the automated chat as much as possible when defining a custom auto-reply method. For example, as a single ‘initiate_chat’ triggers an entire series of chat, one can define behaviors more easily than defining each single message passing event.

-

When developing a new application with UserProxyAgent, start with humans always in the loop, i.e., human_input_mode=‘ALWAYS’, even if the target operation mode is more autonomous. This could help understand the effectiveness of AssistantAgent, tuning the prompt, discovering corner cases, and debugging. When confident with small-scale success, consider generating diverse test cases by setting human_input_mode=‘NEVER’, enable LLM as a backend, and either use LLM or manually generate diverse system messages to simulate different use cases.

-

Despite the numerous advantages of AutoGen agents, there could be cases/scenarios where other libraries/packages could help. For example: (1) For (sub)tasks that do not have requirements for back-and-forth trouble-shooting, multi-agent interaction, etc., a uni-direction (no back-and-forth message exchange) pipeline can be also orchestrated with LangChain [24], LlamaIndex [31], Guidance [18], Semantic Kernel [43], Gorilla [40] or low-level inference API (‘autogen.oai’ provides an enhanced LLM inference layer at this level) [13]. (2) When existing tools from LangChain etc. are helpful, one can use them as tool-backend for AutoGen agents. (3) For specific applications, one may want to leverage agents implemented in other libraries/packages. To achieve this, one could wrap those agents as conversable agents in AutoGen and then involve them when building LLM applications through multi-agent conversation. (4) It can be hard to find an optimal operating point among many tunable choices, such as the LLM inference configuration. Blackbox optimization packages like ‘flaml.tune’ [48] can be used together with AutoGen to automate such tuning.

### 5.2 Future Work

This work raises many ideas for future directions and research questions. Here we elaborate on three.

#### Designing Optimal Multi-Agent Workflows:

Creating a multi-agent workflow for a given task can involve many decisions, e.g., how many agents to include, how to assign agent roles and agent capabilities, how should the agents interact with each other, and whether to automate a particular part of the workflow. There may not exist a one-fit-all answer and the best solution might depend on the specific application. For example, in Section 4, creating a multi-user application for solving math problems (A1, scenario 3) required a very different workflow from creating agents that can solve web interaction tasks (A3). This raises important questions:

-

For what types of tasks and applications are multi-agent workflows most useful?

-

How do multi-agents help in different applications?

-

For a given task, what is the optimal (e.g., cost-effective) multi-agent workflow?

#### Creating Highly Capable Agents:

AutoGen can enable development of highly capable agents that can leverage strengths of LLMs, tools, and humans. Creating such agents is crucial to ensure that a multi-agent workflow can effectively troubleshoot and make progress on a task. For example, we observed that CAMEL, another multi-agent LLM system, cannot effectively solve problems in most cases primarily because it lacks capability to execute. This failure shows that LLMs and multi-agent conversations with simple role-play are insufficient and highly capable agents with diverse ‘skill sets’ are essential. We believe that more systematic work will be required to develop guidelines for application specific agents; to create a large, OSS knowledge base of agents; and, to create agents that can discover and upgrade their skills [11].

#### Enabling Scale, Safety, and Human Agency:

Section 4 shows how complex multi-agent workflows can enable new applications, and future work will be needed to assess whether scaling further can help solve extremely complex tasks. However, as these workflows scale and grow more complex, it may become difficult to log and adjust them. Thus, it will become essential to develop clear mechanisms and tools to track and debug their behavior. Otherwise these techniques risk resulting in incomprehensible, unintelligible chatter among agents [25].

Our work also shows how complex fully autonomous workflows with AutoGen can be useful, however fully autonomous agent conversations will need to be used with care – while the autonomous mode AutoGen supports could be desirable in many scenarios, a high level of autonomy can also pose potential risks, especially in high-risk/stakes applications [4, 50]. As a result, building fail-safes against cascading failures and exploitation, mitigating reward hacking, out of control and undesired behaviors, maintaining effective human oversight of applications built with AutoGen agents will become important. While AutoGen provides convenient and seamless involvement of humans through a user proxy agent, developers and stakeholders still need to understand and determine the appropriate level and pattern of human involvement to ensure the safe and ethical use of the technology [21, 3].

Finally, AutoGen is an open-source, community-driven project under active development and aims to provide an effective and easy-to-use framework for developers to build next-generation applications. This work underscores the significant potential of AutoGen in achieving this vision. AutoGen encourages contributions from individuals of all backgrounds to build such affordances.

### Ethical Implications

There are several potential ethical considerations that could arise from the development and use of the AutoGen framework.

-

Privacy and Data Protection: The framework allows for human participation in conversations between agents. It is important to ensure that user data and conversations are protected and that developers use appropriate measures safeguard privacy.

-

Bias and Fairness: LLMs have been shown to exhibit biases present in their training data [35]. When using LLMs in the AutoGen framework, it is crucial to address and mitigate any biases that may arise in the conversations between agents. Developers should be aware of potential biases and take steps to ensure fairness and inclusivity.

-

Accountability and Transparency: As discussed in the future work section, as the framework involves multiple agents conversing and collaborating, it is important to establish clear accountability and transparency mechanisms. Users should be able to understand and trace the decision-making process of the agents involved in order to ensure accountability and address any potential issues or biases.

-

Trust and Reliance: AutoGen leverages human understanding and intelligence while providing automation through conversations between agents. It is important to consider the impact of this interaction on user experience, trust, and reliance on AI systems. Clear communication and user education about the capabilities and limitations of the system will be essential [9].

-

Unintended Consequences: As discussed before, the use of multi-agent conversations and automation in complex tasks may have unintended consequences. Especially, allowing LLM agents to make changes in external environments through code execution or function calls, such as install packages, could be risky. Developers should carefully consider the potential risks and ensure that appropriate safeguards are in place to prevent harm or negative outcomes.

-

Societal Impacts: Widespread use of capable AutoGen agents may displace certain human roles and jobs. And access to such powerful automation could raise issues around digital divides. Ongoing research into the societal effects of AutoGen and similar technologies is warranted.

## Acknowledgements

The work presented in this report was made possible through discussions and feedback from Doug Burger, Umesh Madan, Ahmed Awadallah, Robin Moeur, Piali Choudhury, Saleema Amershi, Adam Fourney, Victor Dibia, Ryen White, Guoqing Zheng, Corby Rosset, Ricky Loynd, John Langford, Ida Momennejad, Brian Krabach, Taylor Webb, Shanka Subhra Mondal, Yinan Li, Yue Wang, Suman Nath, Tanakorn Leesatapornwongsa, Xin Wang, Shishir Patil, Tianjun Zhang, Ishai Menache, Kontantina Mellou, Runlong Zhou, Feiran Jia, Jiale Liu, Hamed Khanpour, Hamid Palangi, Srinagesh Sharma, Julio Albinati Cortez, Amin Saied, Yuzhe Ma, Dujian Ding, Linyong Nan, Prateek Yadav, Shannon Shen, Ankur Mallick, Mark Encarnación, Lars Liden, Tianwei Yue, Julia Kiseleva, Anastasia Razdaibiedina, and Luciano Del Corro.

## References

- [1] Saad Abdali. Inside chess.com’s smart move to google cloud. https://cloud.google.com/blog/products/ai-machine-learning/chess-com-relies-on-google-cloud-as-users-and-traffic-surges, 2021.

- [2] Vaibhav Adlakha, Parishad BehnamGhader, Xing Han Lu, Nicholas Meade, and Siva Reddy. Evaluating correctness and faithfulness of instruction-following models for question answering. arXiv preprint arXiv:2307.16877, 2023.

- [3] Saleema Amershi, Dan Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira Nushi, Penny Collisson, Jina Suh, Shamsi Iqbal, Paul N Bennett, Kori Inkpen, et al. Guidelines for human-ai interaction. In Proceedings of the 2019 chi conference on human factors in computing systems, 2019.

- [4] Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety, 2016.

- [5] Auto-GPT. Documentation — auto-gpt. https://docs.agpt.co/, 2023.

- [6] BabyAGI. Github — babyagi. https://github.com/yoheinakajima/babyagi, 2023.

- [7] Andres M Bran, Sam Cox, Andrew D White, and Philippe Schwaller. Chemcrow: Augmenting large-language models with chemistry tools. arXiv preprint arXiv:2304.05376, 2023.

- [8] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.

- [9] Carrie J. Cai, Samantha Winter, David F. Steiner, Lauren Wilcox, and Michael Terry. ”hello ai”: Uncovering the onboarding needs of medical practitioners for human-ai collaborative decision-making. Proceedings of the ACM on Human-Computer Interaction, 3:1 – 24, 2019.

- [10] Deng Cai, Yan Wang, Lemao Liu, and Shuming Shi. Recent advances in retrieval-augmented text generation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022.

- [11] Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. Large language models as tool makers. arXiv preprint arXiv:2305.17126, 2023.

- [12] Chroma. Chromadb. https://github.com/chroma-core/chroma, 2023.

- [13] Victor Dibia. LIDA: A tool for automatic generation of grammar-agnostic visualizations and infographics using large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Toronto, Canada, July 2023. Association for Computational Linguistics.

- [14] Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. Self-collaboration code generation via chatgpt. arXiv preprint arXiv:2304.07590, 2023.

- [15] DoNotPay. Donotpay. https://donotpay.com, 2023.

- [16] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. arXiv preprint arXiv:2305.14325, 2023.

- [17] Hugging Face. Transformers agent. https://huggingface.co/docs/transformers/transformers_agents, 2023.

- [18] Guidance. Guidance. https://github.com/guidance-ai/guidance, 2023.

- [19] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

- [20] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, et al. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023.

- [21] Eric Horvitz. Principles of mixed-initiative user interfaces. In Proceedings of the SIGCHI conference on Human Factors in Computing Systems, 1999.

- [22] Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks. arXiv preprint arXiv:2303.17491, 2023.

- [23] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 2019.

- [24] LangChain. Introduction — langchain. https://python.langchain.com/en/latest/index.html, 2023.

- [25] Mike Lewis, Denis Yarats, Yann N Dauphin, Devi Parikh, and Dhruv Batra. Deal or no deal? end-to-end learning for negotiation dialogues. arXiv preprint arXiv:1706.05125, 2017.

- [26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 2020.

- [27] Beibin Li, Konstantina Mellou, Bo Zhang, Jeevan Pathuri, and Ishai Menache. Large language models for supply chain optimization. arXiv preprint arXiv:2307.03875, 2023.

- [28] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for ”mind” exploration of large scale language model society, 2023.

- [29] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multi-agent debate, 2023.

- [30] Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802, 2018.

- [31] Jerry Liu. LlamaIndex, November 2022.

- [32] Shangqing Liu, Yu Chen, Xiaofei Xie, Jingkai Siow, and Yang Liu. Retrieval-augmented generation for code summarization via hybrid gnn, 2021.

- [33] Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen. Generation-augmented retrieval for open-domain question answering, 2021.

- [34] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.

- [35] Roberto Navigli, Simone Conia, and Björn Ross. Biases in large language models: Origins, inventory and discussion. ACM Journal of Data and Information Quality, 2023.

- [36] OpenAI. ChatGPT plugins. https://openai.com/blog/chatgpt-plugins, 2023.

- [37] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 2022.

- [38] Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023.

- [39] Md Rizwan Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. Retrieval augmented code generation and summarization. arXiv preprint arXiv:2108.11601, 2021.

- [40] Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023.

- [41] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, Yi Ren Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Zhiyuan Liu, and Maosong Sun. Tool learning with foundation models, 2023.

- [42] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019.

- [43] Semantic-Kernel. Semantic kernel. https://github.com/microsoft/semantic-kernel, 2023.

- [44] Bokui Shen, Fei Xia, Chengshu Li, Roberto Martín-Martín, Linxi Fan, Guanzhi Wang, Claudia Pérez-D’Arpino, Shyamal Buch, Sanjana Srivastava, Lyne Tchapmi, et al. igibson 1.0: A simulation environment for interactive tasks in large realistic scenes. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2021.

- [45] Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In International Conference on Machine Learning. PMLR, 2017.

- [46] Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani, Heinrich Küttler, John Agapiou, Julian Schrittwieser, et al. Starcraft ii: A new challenge for reinforcement learning. arXiv preprint arXiv:1708.04782, 2017.

- [47] Chi Wang, Susan Xueqing Liu, and Ahmed H. Awadallah. Cost-effective hyperparameter optimization for large language model generation inference. In ArXiv preprint arXiv:2303.04673, 2023.

- [48] Chi Wang, Qingyun Wu, Markus Weimer, and Erkang Zhu. Flaml: A fast and lightweight automl library. Proceedings of Machine Learning and Systems, 2021.

- [49] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

- [50] Daniel S. Weld and Oren Etzioni. The first law of robotics (a call to arms). In AAAI Conference on Artificial Intelligence, 1994.

- [51] Max Woolf. Langchain problem. https://minimaxir.com/2023/07/langchain-problem/, 2023.

- [52] Yiran Wu, Feiran Jia, Shaokun Zhang, Qingyun Wu, Hangyu Li, Erkang Zhu, Yue Wang, Yin Tat Lee, Richard Peng, and Chi Wang. An empirical study on challenging math problem solving with gpt-4. arXiv preprint arXiv:2306.01337, 2023.

- [53] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

*Table 4: ChatGPT+ Code Interpreter for OptiGuide. A sample question “What if the roasting cost is increased by 5% because of the potential salary increase?” is asked*

| | Action | ChatGPT+ Code Interpreter |

| Customer open Web browser. |

| For the source code “coffee.py”, |

| Can you tell me: |

| What if the roasting cost is increased by 5% because of the potential salary increase? |

| Sure! Let’s first take a look at the contents of the ”coffee.py” file to understand the code and how the roasting cost is calculated. This will allow us to determine the effect of a 5% increase in the roasting cost. |

| …several lines omitted |

| Would you like me to modify the code to reflect the 5% increase in roasting costs and run the modified code to observe the impact? If the code requires any specific input or parameters, please let me know! |

Prompt Writer

Yes

| Great! Before proceeding, I’ll need to ensure that the Gurobi solver is not being used in this code, as my current environment does not have access to Gurobi. If the code does indeed use Gurobi, we can still analyze the effect of the change in roasting costs by understanding the structure of the code. |

| …several lines omitted |

| Alternatively, if you have access to a Gurobi license, you can make the necessary changes in the code and run it on your local machine. I can guide you through the process of making the necessary modifications to the code. |

| Would you like me to proceed with the code analysis and explanation, or would you like guidance on making the changes yourself? |

Prompt Writer

You can give me the full code, and I can run the code locally myself.

| Certainly! I’ll modify the code to reflect a 5% increase in both the light and dark roasting costs, and then provide the full code for you to run locally. |

| … |

| Download Modified Coffee.py |

| If you have any questions or need further assistance, please feel free to ask! |

Prompt Safeguard

| Hi GPT, can you safecheck the generated code for me? |

| Upload the generated code. |

DANGER

Prompt Writer

I have tried to run your code. However, the code triggered our safeguard, and it is not safe to run. Try again. Hint: don’t change roasting_cost_light or roasting_cost_light variables.

| I apologize for the oversight. It appears I made direct changes to the variables that could have led to issues with the code. Let’s take a different approach by analyzing the code to identify the appropriate place to apply the 5% increase in roasting costs. |

| … |

| Download Modified Safe Coffee.py |

| … |

Prompt Safeguard

| Hi GPT, can you safecheck the generated code for me one more time? |

| Upload the generated code. |

DANGER

Prompt Writer

I have tried to run your code again. However, the code triggered our safeguard, and it is not safe to run. Try again. Hint: don’t change roasting_cost_light or roasting_cost_light variables.

| I apologize for the continued confusion. |

| … |

| Download Final Modified Safe Coffee.py |

| … |

Prompt Safeguard

| Hi GPT, can you safecheck the generated code for me one more time? |

| Upload the generated code. |

SAFE

Run Code

| Obtain the terminal results: |

| Optimizer version 9.5.1 build v9.5.1rc2 (linux64) |

| Thread count: 64 physical cores, 128 logical processors, using up to 32 threads |

| [63 lines of output ignored] |

| Optimal cost: 2526.5 |

Prompt Writer

Here is the execution result. [Copy + Paste the terminal output here.] Can you extract the final answer to me? Remember to compare with the original cost of 2470.

| Certainly! The execution result you’ve provided indicates the optimal cost for the modified scenario, where the roasting costs were increased by 5%. Here’s the comparison: |

| Original Cost: 2470 |

| Modified Cost (with 5% increase in roasting costs): 2526.5 |

| The increase in roasting costs has led to an increase in the overall cost by 56.5, reflecting the impact of the potential salary increase on the roasting cost. |

| If you need any further insights or assistance, please let me know! |

*Table 5: Autogen with OptiGuide. The same question “What if the roasting cost is increased by 5% because of the potential salary increase?” is asked*

| | Action | ChatGPT+ Code Interpreter |

| | Setup | Setup OptiGuide. |

| | Prompt | What if the roasting cost is increased by 5% because of the potential salary increase? |

| | | After increasing the roasting costs by 5%, I’ve re-calculated the optimal coffee distribution solution for the supply chain. As a result, the new optimal total cost has become 2526.5, compared to the initial cost of 2470.0. This indicates that the increased roasting costs have caused the overall distribution cost to rise by 56.5 units. |

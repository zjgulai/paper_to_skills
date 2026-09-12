<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2512.24565
     paper_id : 2512.24565
     source   : https://arxiv.org/html/2512.24565v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use

Wenrui Liu† Affiliation: Peking University Email: liuwenrui@pku.edu.cn    Zixiang Liu† Affiliation: Columbia University Email: zl3611@columbia.edu    Elsie Dai† Affiliation: Peking University Email: elsiedai@stu.pku.edu.cn    Wenhan Yu Affiliation: Peking University Email: yuwenhan@stu.pku.edu.cn    Lei Yu Affiliation: Peking University Email: yulei123@stu.pku.edu.cn    Tong Yang* Affiliation: Peking University Email: yangtong@pku.edu.cn

###### Abstract

Large Language Models (LLMs) are increasingly serving as autonomous agents, and their utilization of external tools via the Model Context Protocol (MCP) is considered a future trend. Current MCP evaluation sets suffer from issues such as reliance on external MCP services and a lack of difficulty awareness. To address these limitations, we propose MCPAgentBench, a benchmark based on real-world MCP definitions designed to evaluate the tool-use capabilities of agents. We construct a dataset containing authentic tasks and simulated MCP tools. The evaluation employs a dynamic sandbox environment that presents agents with candidate tool lists containing distractors, thereby testing their tool selection and discrimination abilities. Furthermore, we introduce comprehensive metrics to measure both task completion rates and execution efficiency. Experiments conducted on various latest mainstream Large Language Models reveal significant performance differences in handling complex, multi-step tool invocations. All code is open-source at Github (2025).

## 1 Introduction

Large Language Models (LLMs)(Vaswani (2017); Brown et al. (2020); Guo (2025); Yang et al. (2025)) achieve breakthrough progress in natural language processing and complex reasoning tasks. To further advance capabilities, the research community shifts focus toward Agent models. The Model Context Protocol (MCP)(Model Context Protocol Team (2025)) currently represents a crucial exploration aimed at unifying the interaction modality between Agents and external tools, defining a standard format for tool invocation. Agent utilization of MCP tools becomes a key approach for solving complex, real-world tasks. Consequently, establishing a comprehensive evaluation benchmark that assesses the Planning and Execution capabilities of Agents in invoking MCP tools is essential.

However, existing MCP capability assessment benchmarksLuo et al. (2025); Gao et al. (2025); Fan et al. (2025) suffer from several significant limitations. Firstly, a stability and dependency issue exists, as current benchmarks often rely on real, remote MCP servers, where service stability and availability heavily impact the reproducibility of testing results. Secondly, these benchmarks exhibit insufficient difficulty awareness, performing only simple task categorization and lacking granular observation of the invocation complexity level. Most importantly, models should be able to efficiently complete tasks, while existing frameworks lack metrics for task execution efficiency. To address these challenges, a new evaluation benchmark must incorporate: local MCP server deployment to ensure stability; comprehensive coverage of complex invocation scenarios, including single-step, serial, and parallel calls; and dedicated metrics for task execution efficiency.

This paper proposes MCPAgentBench, an evaluation benchmark specifically designed to assess the efficiency of Agent MCP tool invocation, aiming to resolve the aforementioned challenges. This evaluation work provides the following contributions:

-

Data and Instance Construction: We collect authentic 841 tasks and over 20000 MCP Tools from sources including MCP Marketplace(MCP Market (2025)), Github(GitHub (2025)), and HuggingFace(MCPHackathon (2025)). We perform simple local reconstruction of all MCP Tools and, through manual labeling and matching, ultimately construct 180 high-quality task instances.

-

Automation Framework: An automated evaluation framework implemented based on Autogen achieves dynamic loading of tasks and MCP Tools, which ensures the automation and scalability of the evaluation.

-

Efficiency Metrics: The Task Finish Score (TFS), Task Efficiency Finish Score (TEFS), Time Efficiency, and Token Efficiency metrics define the comprehensive evaluation of the Agent’s planning correctness, execution timing, and resource consumption.

MCPAgentBench incorporates complex tasks spanning daily and professional domains, rigorously distinguishing tasks based on serial and parallel invocation complexity. This benchmark allows for an in-depth evaluation and analysis of mainstream Large Language Models, revealing a common efficiency deficiency in handling complex tool calls, particularly in parallel tool invocations.

## 2 Related Work

Recent advances in agentic large language models (LLMs) have shifted evaluation from pure text generation toward tool-use reasoning, with a growing emphasis on when, how, and why agents invoke tools. The Model Context Protocol (MCP) has emerged as a unified substrate for such evaluation by enforcing schema-consistent communication and execution-grounded validation, enabling reproducible and protocol-aligned assessment of tool-augmented agents.

Before MCP, tool-use benchmarks were developed largely within heterogeneous API ecosystems. API-Bank Li et al. (2023) and ToolBench Qin et al. (2024) evaluated Plan–Retrieve-Call behaviors but lacked unified protocol abstraction and execution-level guaranties. Earlier paradigms such as ReAct Yao et al. (2023), Auto-GPT Richards (2023), and GAIA Mialon et al. (2023) explored the interaction between reasoning and acting, though often in synthetic or text-only environments. More recent MCP-based benchmarks represent a clear shift from simulated function calling to execution-verified, protocol-consistent evaluation. MCP-Universe Luo et al. (2025) evaluates agents against live MCP servers, MCP-RADAR Gao et al. (2025) introduces multi-dimensional evaluation metrics, MCPWorld Yan et al. (2025) supports hybrid API–GUI tasks, and MCPToolBench++ Fan et al. (2025) scales evaluation to thousands of MCP servers with a fine-grained error taxonomy.

While these benchmarks substantially advance the realism and coverage of MCP-based evaluation, they primarily focus on task correctness and protocol compliance. In contrast, MCPAgentBench is proposed as a complementary data set that targets a more fine-grained and decision-centric aspect of tool use: the efficiency with which agents select and invoke MCP tools to complete tasks.In terms of concreteness, MCPAgentBench differs from the existing MCP benchmarks in several key aspects. It employs an Autogen-based sandbox with locally maintained MCP servers to ensure stable and reproducible execution. Tasks are categorized according to the complexity of MCP tool invocation and task attributes, enabling structured analysis across difficulty levels. Moreover, MCPAgentBench uses authentic definitions and parameters of the MCP tool, building simulated MCP servers that strictly follow the standard MCP protocol, while introducing realistic distractors that test the’ robustness of agents in tool selection. Together, these design choices position MCPAgentBench as a complementary benchmark that emphasizes task completion efficiency, task authenticity, and robustness to interference, allowing fine-grained evaluation of agent tool-invocation capabilities beyond correctness alone.

## 3 MCPAgentBench Framework

This section introduces the architecture of MCPAgentBench, the benchmark construction process, the evaluation process, task classification, and evaluation metrics.

*Figure 1: MCPAgentBench Overview.*

### 3.1 Overview

MCPAgentBench employs a sandbox environment built upon the Autogen framework. For each task, the system dynamically loads a corresponding MCP tool list via Autogen’s tool interface to facilitate automated benchmark testing.

The overall architecture of the MCPAgentBench framework, illustrated in Figure 1, comprises three key components:

-

MCP Tool Collection: A repository of authentic MCP tools, collected and curated from GitHub and various MCP collection websites. MCPAgentBench extracts structured information for each tool, particularly its functional description and parameters.

-

Task Set: A diverse collection of tasks spanning daily life and professional domains. These tasks, originating from real-world datasets, have undergone meticulous manual review and curation to ensure the uniqueness of each task’s solution.

-

Automated Evaluation Sandbox: A sandbox environment implemented using the Autogen framework, designed for automated task execution and evaluation.

The evaluation process leverages this sandbox environment. For each task $T$, MCPAgentBench retrieves $n$ corresponding correct tools ($G$) from the main tool library and samples $K-n$ "distractor tools" ($F$) that are functionally unrelated or easily confused. Together, these form a dynamic candidate list $L$ containing $K$ tools (e.g., $K=20,30$), which MCPAgentBench provides to the agent under test at runtime.

The agent (driven by the LLM under test) must interpret task $T$, select the correct tool(s) from the distractor-filled list $L$, and generate compliant call parameters. The Autogen framework manages the agent-tool interaction and records every tool call. Finally, MCPAgentBench compares these captured calls against the pre-defined, unique solution and automatically computes a score based on the evaluation metrics. During comparison, MCPAgentBench prompt by default compares the name of the called MCP tool with the incoming parameters. For tools where parameters are not unique, MCPAgentBench prompt only compares whether the names are consistent.

This design not only tests the model’s fundamental tool-calling capabilities but also specifically assesses its tool discrimination and anti-interference abilities in a "needle in a haystack" scenario. Users can initiate the fully automated evaluation simply by providing an API key and model name in the configuration file.

### 3.2 Data Preprocessing

The quality of the benchmark hinges on the authenticity and rigor of its data. To construct high-quality test cases, MCPAgentBench employs a four-step data processing workflow designed to ensure task authenticity, tool representativeness, and solution uniqueness.

*Figure 2: The Data Preprocess of MCPAgentBench.*

Step 1: Raw Data Collection. As shown in Figure 2, the process begins by collecting raw data from two primary public channels. For MCP Tools, we gather authentic Model Context Protocol servers and tool definitions from various websites, including awesome-mcp-servers (2025), MCP Market (2025), mcp.so (2025), and MCPHackathon (2025). These definitions typically exist in the form of API documentation, JSON Schemas, or code comments. After deduplication, we obtain definitions for 9714 MCP servers and over 20000 MCP tools. For Tasks, we collect real-world user queries and task descriptions from the Hugging Face Datasets platform and other academic datasets like Infinity-Instruct (2025) and Schema-Guided Dialogue Dataset Rastogi et al. (2020). These tasks cover both daily and professional domains.

Step 2: Tool and Task Annotation. The collected raw tools and tasks are functionally disparate. To establish connections between them and ensure label standardization, MCPAgentBench utilizes a three-stage, LLM-based annotation process. First, this stage leverages the open-ended generation capability of an LLM to perform an initial, unconstrained, free-format annotation of all MCP tools and tasks, allowing the model to generate multiple descriptive labels for each item. Next, this is followed by manual integration and filtering of all generated tags. This process aims to merge synonyms, remove ambiguous labels, and establish a unified, standardized "Tag Set". Finally, an LLM is utilized again, but constrained to select labels only from this established Tag Set. The model selects the most appropriate label(s) for each MCP tool and task, ensuring consistent classification.

Step 3: Matching and Curation. This step is critical for constructing effective test cases. It identifies potential "Task-Tool" pairs by matching similar tags. However, original task descriptions and tool definitions seldom align perfectly. Therefore, strict manual curation is performed, aiming to align the Task and Tool to achieve a "unique solution" with minimal modifications.

Step 4: MCP Tool Code Generation. Finally, to enable automated evaluation in the sandbox, executable mock code for the MCP tools is required. LLM (e.g., GPT-4o) automatically generate Python stub functions based on the curated tool definitions (including tool name, description, and parameters). As automatically generated code may contain defects, an expert team reviews and modifies each function. This review ensures function signatures are identical to the definitions, the logic is correct, and the code executes safely within the sandbox.

This process yields a complete test case, which comprises three core components: (1) the task description, (2) the MCP tool definition and its verified mock code, and (3) the unique solution for the task.

### 3.3 Task Classification

To assess the model’s capabilities in various scenarios, we classified all tasks based on their complexity and domain specificity.

Task Domain: 1) General Tasks: Covers common scenarios such as daily life, entertainment, and office work. 2) Professional Tasks: Involves specific domains, such as academic research or software engineering.

Invocation Complexity: 1) Single-Tool Invocation: The task can be resolved by invoking only one MCP tool. This tests the Agent’s foundational ability to understand and select the correct tool. 2) Dual-Tool Parallel Invocation: The task requires the Agent to plan and invoke two independent tools concurrently. This assesses the Agent’s task decomposition and parallel planning capabilities. 3) Dual-Tool Serial Invocation: The task requires the Agent to invoke tools in two sequential steps, following a specific logical order. For example, the tool invocation in the second step may depend on the output from the first. This assesses the Agent’s capabilities for multi-step reasoning, planning, and state maintenance. 4) Multi-Tool Invocation: The task requires the Agent to invoke tools in multiple steps according to a logical sequence. This may involve a combination of parallel and serial invocations, representing a more complex tool-use scenario.

Every Domain contains 30 tasks of type Single-tool Invocation, plus 20 tasks of each other complex type of invocation, for a total of 180 tasks.

## 4 Evaluation

### 4.1 Evaluation Metrics

To evaluate the Agent’s capabilities from multiple dimensions, we define the following evaluation metrics, which cover task completion, execution efficiency, and resource consumption.

Let $N$ be the total number of tasks in the test set, and $T_{i}$ be the $i$-th task. Let $G_{i}$ be the solution for task $T_{i}$, defined as a sequence of $n_{i}$ standard tool invocations. The weight of task $T_{i}$ is its number of standard invocations, $|G_{i}|=n_{i}$. Let $P_{i}$ be the tool invocation sequence actually generated by the Agent for task $T_{i}$.

Task Finish Score (TFS): A task $T_{i}$ is considered "Finished" ($\text{IsFinished}(T_{i})=1$) if and only if the set of tool invocations generated by the Agent, $P_{i}$, is identical to the set of invocations in the golden solution $G_{i}$. This requires an exact match of all "tool names" and "parameters" (where applicable), but does not consider the invocation order. TFS is the weighted average score across all tasks.

$TFS=\frac{\sum_{i=1}^{N}\text{IsFinished}(T_{i})\cdot|G_{i}|}{\sum_{i=1}^{N}|G_{i}|}$ | | | |

Task Efficiency Finish Score (TEFS): A task $T_{i}$ is considered "Efficiently Finished" ($\text{IsEfficientlyFinished}(T_{i})=1$) if and only if two conditions are met: (1) The task is "Finished" ($\text{IsFinished}(T_{i})=1$), and (2) The Agent’s generated tool invocation sequence $P_{i}$ exactly matches the golden solution $G_{i}$ in its serial and parallel execution order. TEFS is the weighted average score across all tasks.

$TEFS=\frac{\sum_{i=1}^{N}\text{IsEfficientlyFinished}(T_{i})\cdot|G_{i}|}{\sum_{i=1}^{N}|G_{i}|}$ | | | |

Resource Efficiency: We also record the Agent’s resource overhead during task execution to evaluate its cost-effectiveness. The "Total Score" in these metrics refers to the total weighted score (e.g., the numerator in the TFS formula: $\sum\text{IsFinished}\cdot|G_{i}|$).

-

Token Efficiency: Measures the score obtained per 1k output tokens consumed.

$\text{Token Efficiency}=\frac{\text{Total Score}}{\sum_{i=1}^{N}\text{Output Tokens}_{i}}$ | | | |

-

Time Efficiency: Measures the score obtained per minute of execution time.

$\text{Time Efficiency}=\frac{\text{Total Score}}{\sum_{i=1}^{N}\text{Time}_{i}}$ | | | |

In subsequent tests, the "Total Score" used for efficiency calculations is the total weighted score derived from TEFS.

### 4.2 Main Results

We evaluate the performance of the following 11 mainstream models using the MCPAgentBench benchmark: Claude Sonnet 4.5 (Anthropic (2025)), DeepSeek V3.2 (DeepSeek-AI et al. (2025)), Gemini 3 Pro Preview (GoogleAI (2025)), gpt-5(OpenAI (2025a)), gpt-o3(OpenAI (2025b)), gpt-o4-mini(OpenAI (2025c)), grok-4(xAI (2025)), qwen3-235b-a22b-instruct-2507(Yang et al. (2025)), qwen3-235b-a22b-thinking-2507(Yang et al. (2025)), kimi-k2(Bai et al. (2025)), and glm-4.6(ZhipuAI (2025)). For the performance comparison experiments, the number of tool invocation lists, $N$, is set to 20.

*Figure 3: Evaluation Results of TFS and TEFS.*

Figure 3 presents the overall average TFS and TEFS scores (avg@4) for these 10 mainstream models evaluated on MCPAgentBench. Under the TFS metric, Claude Sonnet 4.5, o3, and glm-4.6 achieve the top three scores, demonstrating superior task completion, whereas Gemini 3 Pro Preview records the lowest score. When we assess performance using the stricter TEFS metric, Claude Sonnet 4.5, glm-4.6, and qwen3-235b-a22b-instruct-2507 secure the top three positions in execution efficiency, with Gemini 3 Pro Preview exhibiting the lowest efficiency score.

*Table 1: TFS avg@4 by Task Category*

| Models | Daily | Professional |

| Single | Dual Serial | Dual Parallel | Multi | Single | Dual Serial | Dual Parallel | Multi |

| claude-sonnet-4.5 | 96.67 | 86.25 | 93.75 | 67.50 | 90.00 | 58.75 | 68.75 | 40.28 |

| DeepSeek-V3.2 | 91.67 | 58.75 | 68.75 | 56.25 | 89.17 | 50.00 | 62.50 | 36.11 |

| gemini-3-pro-preview | 72.50 | 50.00 | 52.50 | 38.75 | 67.50 | 40.00 | 48.75 | 37.50 |

| glm-4.6 | 87.50 | 92.50 | 77.50 | 50.00 | 83.33 | 58.75 | 61.25 | 44.44 |

| gpt-5 | 90.83 | 77.50 | 73.75 | 38.75 | 82.50 | 48.75 | 56.25 | 41.67 |

| grok-4 | 93.33 | 55.00 | 67.50 | 40.00 | 83.33 | 43.75 | 62.50 | 37.50 |

| kimi-k2-thinking | 90.83 | 70.00 | 76.25 | 57.50 | 86.67 | 47.50 | 56.25 | 34.72 |

| o3 | 95.83 | 80.00 | 85.00 | 56.25 | 87.50 | 48.75 | 71.25 | 36.11 |

| o4-mini | 93.33 | 72.50 | 70.00 | 37.50 | 91.67 | 57.50 | 65.00 | 13.89 |

| qwen3-235b-a22b-instruct-2507 | 89.17 | 76.25 | 77.50 | 60.00 | 89.17 | 47.50 | 63.75 | 31.94 |

| qwen3-235b-a22b-thinking-2507 | 94.17 | 40.00 | 80.00 | 32.50 | 86.67 | 33.75 | 70.00 | 29.17 |

| Average | 91.04 | 70.73 | 75.10 | 49.90 | 85.90 | 47.71 | 61.98 | 34.95 |

*Table 2: TEFS avg@4 by Task Category*

| Models | Daily | Professional |

| Single | Dual Serial | Dual Parallel | Multi | Single | Dual Serial | Dual Parallel | Multi |

| claude-sonnet-4.5 | 96.67 | 51.25 | 93.75 | 55.00 | 90.00 | 33.75 | 68.75 | 15.28 |

| DeepSeek-V3.2 | 91.67 | 58.75 | 22.50 | 12.50 | 89.17 | 48.75 | 13.75 | 26.39 |

| gemini-3-pro-preview | 72.50 | 28.75 | 48.75 | 20.00 | 67.50 | 17.50 | 47.50 | 5.56 |

| glm-4.6 | 87.50 | 75.00 | 76.25 | 35.00 | 83.33 | 50.00 | 48.75 | 23.61 |

| gpt-5 | 90.83 | 77.50 | 0.00 | 10.00 | 82.50 | 48.75 | 0.00 | 30.56 |

| grok-4 | 93.33 | 31.25 | 67.50 | 27.50 | 83.33 | 20.00 | 62.50 | 5.56 |

| kimi-k2-thinking | 90.83 | 57.50 | 72.50 | 41.25 | 86.67 | 37.50 | 43.75 | 22.22 |

| o3 | 95.83 | 80.00 | 0.00 | 8.75 | 87.50 | 48.75 | 0.00 | 25.00 |

| o4-mini | 93.33 | 72.50 | 0.00 | 10.00 | 91.67 | 57.50 | 0.00 | 11.11 |

| qwen3-235b-a22b-instruct-2507 | 89.17 | 66.25 | 73.75 | 25.00 | 89.17 | 37.50 | 60.00 | 20.83 |

| qwen3-235b-a22b-thinking-2507 | 94.17 | 8.75 | 80.00 | 13.75 | 86.67 | 3.75 | 70.00 | 5.56 |

| Average | 91.04 | 57.81 | 47.08 | 24.06 | 85.90 | 36.77 | 36.46 | 17.59 |

Tables 1 and 2 present the TFS and TEFS scores, respectively, across different task categories. We observe that under the TFS metric, the average score for the Dual Parallel Tool Task is higher than that for the Dual Serial Tool Task. This suggests that, in terms of logical complexity and task completeness, parallel tasks are inherently easier to solve than serial tasks. However, a transition to the TEFS metric reveals a sharp and significant drop in the Dual Parallel Tool Task scores across the board. This substantial decline indicates a general deficiency in the models’ capability for correct parallel tool invocation. This inability is particularly evident in models from the OpenAI series (e.g., gpt-5), which record a TEFS score of 0 for the Dual Parallel Tool Task, demonstrating a complete failure to execute the required parallel tool calls efficiently or correctly.

*Figure 4: Evaluation Results of Token Efficiency.*

Figure 4 illustrates the results for Token Efficiency. qwen3-235b-a22b-instruct-2507 exhibits the highest Token Efficiency, significantly higher than Claude Sonnet 4.5 and glm-4.6, which rank second and third, respectively. This leading performance is attributed to qwen3-235b-a22b-instruct-2507’s highest score under the TEFS metric combined with its "NoThinking" design. Conversely, gpt-5 records the lowest Token Efficiency, suggesting that the excessive "thinking" tokens generated by gpt-5 do not translate into effective scores.

*Figure 5: Evaluation Results of Time Efficiency.*

Figure 5 illustrates the results for Time Efficiency. Claude Sonnet 4.5 achieves the highest Time Efficiency, with glm-4.6 and qwen3-235b-a22b-instruct-2507 ranking second and third, respectively. The lowest Time Efficiency is recorded by gpt-5, which is consistent with the Token Efficiency results.

### 4.3 Performance Analysis

We further investigate the influence of model size and the number of candidate tools on TEFS.

First, we examine the change in avg@4 TEFS for different size models within the Qwen2.5 and Qwen3 series, setting the number of candidate tools to 10. As shown in Figure 6(a), TEFS generally exhibits an upward trend as the model size increases. However, a noticeable dip in performance occurs at the Qwen 2.5 32B model, which may relate to its specific training methodology.

Next, we evaluate the impact of varying the number of candidate tools using Deepseek-V3.2, Kimi-K2-Thinking and Qwen3-235B. Figure 6(b) presents the results, although Deepseek-V3.2 shows a slight upward trend when the number of tools is 10 and 20, overall, as the number of alternative tools increases, the TEFS of all models show a slight downward trend.

(a) Model Size v.s. TEFS Score

(b) Tool Count v.s. TEFS Score

*Figure 6: The influence of model size and Tool Count on TEFS score*

## 5 Conclusion

In this paper, we propose MCPAgentBench, an Autogen-based evaluation framework designed to measure the efficiency of large language models’ MCP tool invocation for task completion. The framework constructs daily and professional tasks covering single-tool, dual-tool (serial or parallel), and multi-tool invocations by matching hundreds of collected tasks with over 20000 MCP tools. The design of novel task completion efficiency metrics achieves automated evaluation of model capabilities. The relevant code is open-source.

## References

- Anthropic (2025) Anthropic Claude sonnet 4.5 system card. External Links: Link Cited by: §4.2.

- awesome-mcp-servers (2025) awesome-mcp-servers Awesome-mcp-servers. Note: https://github.com/punkpeye/awesome-mcp-servers Cited by: §3.2.

- Bai et al. (2025) Y. Bai, Y. Bao, G. Chen, and et al. Kimi k2: open agentic intelligence. External Links: 2507.20534, Link Cited by: §4.2.

- Brown et al. (2020) T. B. Brown, B. Mann, M. Ryder, J. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. External Links: 2005.14165, Link Cited by: §1.

- DeepSeek-AI et al. (2025) DeepSeek-AI, A. Liu, A. Mei, and et al. DeepSeek-v3.2: pushing the frontier of open large language models. External Links: 2512.02556, Link Cited by: §4.2.

- Fan et al. (2025) S. Fan, X. Ding, L. Zhang, and L. Mo MCPToolBench++: a large scale ai agent model context protocol mcp tool use benchmark. External Links: 2508.07575, Link Cited by: §1, §2.

- Gao et al. (2025) X. Gao, S. Xie, J. Zhai, S. Ma, and C. Shen MCP-radar: a multi-dimensional benchmark for evaluating tool use capabilities in large language models. External Links: 2505.16700, Link Cited by: §1, §2.

- GitHub (2025) GitHub GitHub. External Links: Link Cited by: 1st item.

- Github (2025) MCPAgentBench External Links: Link Cited by: Abstract.

- GoogleAI (2025) GoogleAI Gemini 3 pro preview. Note: https://ai.google.dev/gemini-api/docs/gemini-3 Cited by: §4.2.

- Guo (2025) D. e. al. Guo DeepSeek-r1: incentivizing reasoning capability in llms via reinforcement learning. External Links: 2501.12948, Link Cited by: §1.

- Infinity-Instruct (2025) Infinity-Instruct Infinity-instruct dataset. ModelScope. Note: Accessed: 2025-12-27 External Links: Link Cited by: §3.2.

- Li et al. (2023) M. Li, Y. Zhao, B. Yu, F. Song, H. Li, H. Yu, Z. Li, F. Huang, and Y. Li API-bank: a comprehensive benchmark for tool-augmented LLMs. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali (Eds.), Singapore, pp. 3102–3116. External Links: Link, Document Cited by: §2.

- Luo et al. (2025) Z. Luo, Z. Shen, W. Yang, Z. Zhao, P. Jwalapuram, A. Saha, D. Sahoo, S. Savarese, C. Xiong, and J. Li MCP-universe: benchmarking large language models with real-world model context protocol servers. External Links: 2508.14704, Link Cited by: §1, §2.

- MCP Market (2025) MCP Market MCP market (mcpmarket.cn): collection of global model context protocol servers. External Links: Link Cited by: 1st item, §3.2.

- mcp.so (2025) mcp.so MCP.so. External Links: Link Cited by: §3.2.

- MCPHackathon (2025) MCPHackathon Agents mcphackathon tools list: dataset of mcp tools for agents hackathon. Hugging Face. External Links: Link Cited by: 1st item, §3.2.

- Mialon et al. (2023) G. Mialon, C. Fourrier, C. Swift, T. Wolf, Y. LeCun, and T. Scialom GAIA: a benchmark for general ai assistants. External Links: 2311.12983, Link Cited by: §2.

- Model Context Protocol Team (2025) Model Context Protocol Team What is the model context protocol (mcp)? - getting started. Note: Accessed: 2025-12-11; Defines MCP as an open-source standard for connecting AI applications to external data sources, tools, and workflows, analogous to a USB-C port for AIModel Context Protocol Official Documentation External Links: Link Cited by: §1.

- OpenAI (2025a) OpenAI Gpt-5: a team of ph.d. level experts in your pocket. External Links: Link Cited by: §4.2.

- OpenAI (2025b) OpenAI Introducing openai o3. External Links: Link Cited by: §4.2.

- OpenAI (2025c) OpenAI Introducing openai o4-mini. External Links: Link Cited by: §4.2.

- Qin et al. (2024) Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, S. Zhao, L. Hong, R. Tian, R. Xie, J. Zhou, M. Gerstein, D. Li, Z. Liu, and M. Sun ToolLLM: facilitating large language models to master 16000+ real-world apis. In Proceedings of the 12th International Conference on Learning Representations (ICLR), External Links: Link Cited by: §2.

- Rastogi et al. (2020) A. Rastogi, X. Zang, S. Sunkara, R. Gupta, and P. Khaitan Towards scalable multi-domain conversational agents: the schema-guided dialogue dataset. External Links: 1909.05855, Link Cited by: §3.2.

- Richards (2023) T. B. Richards Auto-gpt: an autonomous gpt-4 experiment. Note: https://github.com/Significant-Gravitas/Auto-GPTOpen-source software project Cited by: §2.

- Vaswani (2017) A. e. al. Vaswani Attention is all you need. External Links: 1706.03762, Link Cited by: §1.

- xAI (2025) xAI Grok 4. Note: https://x.ai/news/grok-4/ Cited by: §4.2.

- Yan et al. (2025) Y. Yan, S. Wang, J. Du, Y. Yang, Y. Shan, Q. Qiu, X. Jia, X. Wang, X. Yuan, X. Han, M. Qin, Y. Chen, C. Peng, S. Wang, and M. Xu MCPWorld: a unified benchmarking testbed for api, gui, and hybrid computer use agents. External Links: 2506.07672, Link Cited by: §2.

- Yang et al. (2025) A. Yang, A. Li, B. Yang, and et al. Qwen3 technical report. External Links: 2505.09388, Link Cited by: §1, §4.2.

- Yao et al. (2023) S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao ReAct: synergizing reasoning and acting in language models. In Proceedings of the 11th International Conference on Learning Representations (ICLR), External Links: Link Cited by: §2.

- ZhipuAI (2025) ZhipuAI GLM-4.6: advanced agentic, reasoning and coding capabilities. Note: https://z.ai/blog/glm-4.6 Cited by: §4.2.

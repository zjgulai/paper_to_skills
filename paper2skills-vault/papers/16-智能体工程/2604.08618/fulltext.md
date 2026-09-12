<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2604.08618
     paper_id : 2604.08618
     source   : https://arxiv.org/html/2604.08618v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# SkillForge: Forging Domain-Specific, Self-Evolving Agent Skills in Cloud Technical Support

Xingyan Liu*    Xiyue Luo    Linyu Li    Ganghong Huang    Jianfeng Liu*    Honglin Qiao* Affiliation: Alibaba Cloud Computing, Alibaba Group Affiliation: {wxw, luoxiyue.lxy, lly455313, huangganghong.hgh, jiawei.ljf, kenny.qhl}@alibaba-inc.com

###### Abstract

Deploying LLM-powered agents in enterprise scenarios such as cloud technical support demands high-quality, domain-specific skills. However, existing skill creators lack domain grounding, producing skills poorly aligned with real-world task requirements. Moreover, once deployed, there is no systematic mechanism to trace execution failures back to skill deficiencies and drive targeted refinements, leaving skill quality stagnant despite accumulating operational evidence. We introduce SkillForge, a self-evolving framework that closes an end-to-end creation–evaluation–refinement loop. To produce well-aligned initial skills, a Domain-Contextualized Skill Creator grounds skill synthesis in knowledge bases and historical support tickets. To enable continuous self-optimization, a three-stage pipeline -— Failure Analyzer, Skill Diagnostician, and Skill Optimizer —- automatically diagnoses execution failures in batch, pinpoints the underlying skill deficiencies, and rewrites the skill to eliminate them. This cycle runs iteratively, allowing skills to self-improve with every round of deployment feedback. Evaluated on five real-world cloud support scenarios spanning 1,883 tickets and 3,737 tasks, experiments show that: (1) the Domain-Contextualized Skill Creator produces substantially better initial skills than the generic skill creator, as measured by consistency with expert-authored reference responses from historical tickets; and (2) the self-evolution loop progressively improves skill quality from diverse starting points (including expert-authored, domain-created, and generic skills) across successive rounds, demonstrating that automated evolution can surpass manually curated expert knowledge.

| |

Keywords: LLM Agents, Agent Skills, Self-Evolution, Cloud Technical Support

## 1 Introduction

Enterprise technical support demands reliable procedural knowledge and organizational context beyond raw language modeling [1], and LLM-based agents are increasingly deployed in cloud operations—from automated incident root cause analysis to domain-specific diagnosis [2, 3, 4]. The Agent Skills concept formalizes this specialization: a portable, file-based package that encapsulates procedures, resources, and tool-use guidance so agents can be versioned and composed like software modules [5]. Several foundations now underpin skill-equipped agents: tool-augmented architectures enable robust API invocation and reasoning over results [6, 7, 8]; iterative self-improvement methods let LLMs refine outputs through feedback loops [9, 10, 11]; and automatic instruction optimization techniques can discover stronger agent directives [12, 13, 14]. However, these approaches operate at the prompt or single-task level rather than managing reusable skill artifacts that can be systematically created with domain knowledge and continuously improved from deployment feedback.

Producing and maintaining such high-quality skills in enterprise settings remains difficult due to two key challenges: (1) generic skill creators lack domain grounding, producing poorly aligned initial skills; and (2) no systematic mechanism exists to trace execution failures back to skill deficiencies and drive targeted refinements. SkillsBench, a recent benchmark of 86 tasks across 11 domains, confirms that curated skills improve success rates by 16.2pp on average while self-generated skills provide no benefit [15]. Concurrent work on continuous improvement of LLM-based customer support relies on human annotation as the feedback signal [16]; SkillForge, by contrast, directly targets the skill artifact itself—automating failure analysis and skill rewriting without requiring per-deployment human supervision. We present SkillForge, an end-to-end creation–evaluation–refinement loop that addresses both challenges: a Domain-Contextualized Skill Creator for high-quality initialization, and an automated diagnosis-optimization pipeline for continuous self-evolution. We validate the framework on five real-world cloud technical support scenarios.

## 2 The Self-Evolving Skill Framework

### 2.1 Application Context

This work focuses on customer service AI agents deployed in cloud technical support. The agent’s primary output is a customer-facing reply, which is presented to human support engineers for review. In production, the human agent decides whether to adopt the AI-generated response (with or without minor edits) before sending it to the customer. During the problem-solving process, the AI agent may invoke various tools—including knowledge retrieval, diagnostic APIs, and resource queries—to assist in formulating the response. This human-in-the-loop design enables the “consistency with expert reference” evaluation paradigm used throughout this paper: we compare the agent’s output against the actual solutions provided by human experts in historical tickets.

### 2.2 Anatomy of an Agent Skill

A standard Agent Skill is a self-contained package comprising SKILL.md (instructions and workflow logic), scripts/ (executable code), and references/ (domain documentation) [5]. In enterprise environments, however, allowing agents to execute arbitrary scripts poses security and stability risks. We therefore adopt a constrained definition: the skill excludes scripts/ and relies exclusively on pre-defined, verified system tools. We add references/tools.json to store schemas for these verified tools.

Correspondingly, all skill-bearing agents—including both business agents handling customer tickets and meta-agents (Diagnostician, Optimizer) managing skill evolution—interact with skill assets (SKILL.md, references/) exclusively through a Virtual File System (VFS) rather than executing arbitrary code. This design is motivated by several considerations: (1) the majority of high-frequency customer service tasks can be effectively addressed through instruction/knowledge injection and pre-defined tools without requiring dynamic code execution; (2) eliminating executable scripts significantly improves runtime stability in production; and (3) constraining the action space to text-based operations simplifies failure diagnosis and optimization. More complex scenarios requiring real-time sandbox script execution are left to future work.

### 2.3 Framework Overview

Our framework establishes a continuous improvement cycle inspired by recent work on self-evolving agents [17, 10, 18, 19], designed as a five-phase pipeline to ensure robustness and traceability (Figure 1):

-

Initialization: The Domain-Contextualized Skill Creator generates Skill_v0 based on domain data.

-

Execution & Monitoring: The agent executes tasks using Skill_v_n. Discrepancies between agent execution and reference behaviors are flagged as Bad Cases.

-

Phase 1: Multi-Dimensional Failure Analysis: The Failure Analyzer processes each bad case in parallel across four dimensions (Knowledge, Tool, Clarification, Style) to produce Structured Failure Records.

-

Phase 2: Aggregation: Individual failure records are aggregated by category to identify systemic patterns and select representative cases.

-

Phase 3: Diagnosis: The Skill Diagnostician analyzes the aggregated data to map failures to specific sections of the skill definition, producing a Diagnostic Report and Optimization Plan.

-

Phase 4: Optimization: The Skill Optimizer applies the plan to Skill_v_n via a safe Virtual File System, producing Skill_v_n+1.

*Figure 1: Overview of the SkillForge self-evolving skill framework. A Domain-Contextualized Skill Creator mines historical tickets and domain knowledge to produce an initial Skill_v0. An iterative loop of Failure Analysis, Root Cause Diagnosis, and Skill Optimization then continuously refines the skill based on accumulated bad cases from agent execution.*

### 2.4 Domain-Contextualized Skill Creator

The creator addresses the cold-start problem by generating a robust initial skill. Due to enterprise data privacy requirements, we use an internal LLM (Qwen3-Max) to process proprietary domain data.

-

Input: Task descriptions, historical ticket datasets, and technical documentation.

-

Process:

-

Workflow Mining: Extracts typical solution patterns (e.g., Clarify $\rightarrow$ Diagnose $\rightarrow$ Resolve) from historical ticket dialogues and operation sequences using LLMs.

-

Tool Mining: Identifies high-frequency tools used by human experts in historical tickets and extracts their schemas for inclusion in the skill definition.

-

Knowledge Extraction: Searches internal documentation and knowledge bases (or collects references cited in historical tickets) to extract domain-specific information required by the skill.

-

Skill Synthesis: Fills a pre-defined cloud service Skill template with the mined workflows, tools, and extracted knowledge.

-

Output: A structured, domain-rich initial skill (Skill_v0).

See Appendix A for details on each mining stage.

### 2.5 Agent Execution and Performance Monitoring

Agent Execution. The agent execution pipeline consists of: (1) identifying the appropriate skill for the current scenario; (2) dynamically loading the skill’s SKILL.md and references into context (treating the skill as a “Meta-tool” rather than polluting the global system prompt); (3) executing the task using the loaded instructions and available tools; and (4) returning the generated response. This is achieved via a dual-message mechanism: injecting visible user metadata alongside hidden API instructions that adjust tool permissions and model selection at runtime.

Performance Evaluation. The agent loads the skill to handle tasks. The final response is checked for consistency against a reference response (using an LLM-judge). Low consistency indicates a failure, triggering the bad-case analysis pipeline. Data sources include offline replays of historical tickets or online failed interactions (e.g., low consistency or rejection by human agents). See Appendix B for implementation details of the agent execution pipeline.

### 2.6 Automated Diagnosis

#### 2.6.1 Multi-Dimensional Failure Analysis (Failure Analyzer)

Instead of a single classification, the Failure Analyzer performs a parallel analysis of each Bad Case across four distinct dimensions, drawing on structured reasoning approaches [20, 21] to ensure comprehensive attribution:

-

Knowledge Analysis: Checks for missing, incorrect, or contradictory domain knowledge.

-

Tool Analysis: Evaluates tool calls for missing invocations, wrong parameters, or misinterpretation of results.

-

Clarification Analysis: Assesses whether the agent over-asked, under-asked, or asked irrelevant questions.

-

Style Analysis: Reviews the response tone, ensuring it is not robotic, overly verbose, or cold.

The results from these four dimensions are aggregated (via code logic or LLM) to determine the primary failure category and severity, producing a comprehensive JSON failure record (see Appendix C).

#### 2.6.2 Root Cause Diagnosis (Skill Diagnostician)

The diagnosis phase bridges the gap between individual failures and skill definition defects:

-

Aggregation (Phase 2): The system first aggregates failure records by category. It calculates statistics (e.g., severity distribution) and selects representative cases that best exemplify the systemic issue.

-

Diagnosis Agent (Phase 3): A ReAct-based Skill Diagnostician agent is invoked. It reads the aggregated report and the current SKILL.md (via a Virtual File System).

-

Process: The agent maps the failure patterns to specific sections of the skill (e.g., linking "Knowledge Missing" failures to a gap in the "Troubleshooting" section of SKILL.md).

-

Output: A structured Diagnostic Report and a machine-parsable Optimization Plan.

See Appendix E for the full diagnostic workflow and optimization execution details.

### 2.7 Automated Optimization (Skill Optimizer)

The Skill Optimizer applies the optimization plan through the VFS described in Section 2.2.

-

Knowledge Augmentation: The optimizer can perform knowledge searches to supplement missing high-frequency knowledge identified in the diagnostic report, enriching the skill’s reference materials.

-

Process: The agent executes the Optimization Plan, modifying SKILL.md or reference files. It follows a “minimal modification” principle to preserve existing correct behaviors.

-

Versioning: The modified VFS state is committed as Skill_v_n+1, ensuring full traceability.

## 3 Experiments and Results

### 3.1 Experimental Setup

Our evaluation methodology follows established agent benchmarking practices [22, 23].

#### 3.1.1 Scenarios and Dataset

We evaluate SkillForge on five representative cloud technical support scenarios from a major cloud provider, as summarized in Table 1. Each scenario encompasses multiple fine-grained skills; the skill-level labels are derived from online production tags originally obtained via ticket clustering and subsequent manual refinement. Tickets are grouped into scenarios by their top-level product category.

*Table 1: Evaluation Scenarios and Dataset Statistics.*

| | S1 | S2 | S3 | S4 | S5 | Total |

| Scenario | Account | Domain | DNS | OSS | ECS | – |

| #Tickets | 389 | 527 | 256 | 385 | 326 | 1883 |

| #Tasks | 706 | 1061 | 572 | 730 | 668 | 3737 |

All tickets are real-world, anonymized production tickets. We define a task as a single-turn dialogue within a ticket that can advance ticket resolution—the input is the ticket’s message history (including the current user query), and the agent returns a customer-facing reply. A single ticket may involve multiple tasks. The dataset is split into two disjoint sets: (1) a development set used during skill evolution iterations for bad-case analysis and optimization; and (2) a held-out evaluation set containing tickets never seen during the evolution process, used to assess the generalizability of the optimizations. Bad cases are identified through automated LLM-judge filtering.

#### 3.1.2 Evaluation Metrics

We employ an LLM-judged metric—Consistency Rate (CR)—for skill quality evaluation, comparing the agent’s response against expert reference responses from historical tickets. The LLM-judge classifies each response into one of three categories:

-

Consistent: The clarification questions and solution are aligned with the reference; minor phrasing differences do not affect problem resolution.

-

Partially Consistent: The response overlaps with the reference without contradiction, but may miss some details.

-

Inconsistent: The response lacks critical clarifications, misses core solution elements, or conflicts with the reference.

We report two variants: Strict CR (proportion classified as Consistent) and Lenient CR (proportion classified as Consistent or Partially Consistent). We validated the LLM-judge against human annotations on a sample subset, achieving over 90% agreement, confirming its reliability for automated evaluation (see Appendix D for evaluation criteria and output schema).

#### 3.1.3 Baselines and Variants

-

S_generic: Skills generated by a generic skill creator (Claude Code with Claude-Sonnet-4.5) augmented with domain-specific tool schemas mined from historical tickets, but without access to domain knowledge or historical ticket content due to enterprise data privacy constraints.

-

S_domain: Initial output of our Domain-Contextualized Skill Creator.

-

S_manual: Manually authored initial skill by human domain experts, encoding expert knowledge and best practices.

To evaluate the generality of our self-evolution loop, we use S_manual, S_domain, and S_generic as different starting points for iterative evolution (v1/v2/v3 denote successive evolution cycles).

#### 3.1.4 Implementation Details

All experiments use the latest version of Qwen3-Max as the backbone LLM. Each offline evaluation is repeated 3 times; we report the mean. Bad cases are dynamically identified by the LLM-judge after the agent executes each task with the current skill version.

### 3.2 RQ1: Efficacy of Domain-Contextualized Skill Creator

We compare the initial skill quality of our domain-aware generator (S_domain) against the generic baseline. Table 2 presents the results.

*Table 2: RQ1: Comparison of initial skill quality across scenarios. Format: Strict CR (Lenient CR). Best Strict CR in bold.*

| Method | S1 | S2 | S3 | S4 | S5 |

| S_generic | 58.0 (61.4) | 57.9 (64.4) | 63.0 (69.6) | 59.1 (66.3) | 43.4 (55.3) |

| S_domain | 64.0 (65.7) | 60.5 (68.8) | 65.4 (70.6) | 62.5 (72.7) | 50.6 (57.2) |

Analysis. S_domain outperforms S_generic across all scenarios, with average gains of +4.3pp Strict CR and +3.6pp Lenient CR. The improvement is consistent across all five scenarios, with the largest Strict CR gain in S5 (+7.20pp). Since S_generic already includes mined tool schemas, the gap confirms that domain-specific workflow knowledge and knowledge extraction provide additional value beyond tools alone.

### 3.3 RQ2: Effectiveness of the Self-Evolution Loop

To evaluate the generality of the self-evolution mechanism, we apply it from three distinct starting points—S_manual, S_domain, and S_generic—and track the improvement over three evolution cycles. Table 3 reports the relative gain ($\Delta$) in Strict CR and Lenient CR on the held-out evaluation set.

*Table 3: RQ2: Consistency Rate improvement ($\Delta$) across self-evolution iterations from different starting points on held-out evaluation set. Format: Strict CR (Lenient CR).*

| Iteration | S_manual | S_domain | S_generic |

$\Delta$ | v1 () | +4.09 (+5.39) | +2.36 (+0.24) | +7.70 (+1.60) |

$\Delta$ | v2 () | +9.64 (+9.94) | +7.31 (+4.13) | +8.40 (+4.60) |

$\Delta$ | v3 () | +10.99 (+12.21) | +9.23 (+8.00) | +11.60 (+4.90) |

Analysis. Several key observations emerge:

-

Universal Improvement: All three starting points benefit from the self-evolution loop, with Strict CR gains of +10.99, +9.23, and +11.60 after three iterations respectively. This confirms that the framework is effective regardless of the initial skill quality.

-

Monotonic Progress: Improvements accumulate steadily across iterations, with each cycle providing additional gains.

-

Starting-Point Effects: S_generic, starting from a weaker baseline, achieves the largest cumulative Strict CR gain (+11.60), suggesting that the evolution loop is especially effective at closing the gap for lower-quality initial skills. S_manual, despite being expert-authored, still benefits substantially (+10.99 Strict, +12.21 Lenient), indicating that automated evolution can surpass human-curated knowledge.

#### 3.3.1 Failure Category Reduction Analysis

To understand what the evolution loop fixes, we analyze bad-case counts by failure category. Tool and Style failures improve steadily across both iterations ($-$14.5%/$-$18.2% and $-$16.4%/$-$20.9% respectively), reflecting the optimizer’s ability to refine tool invocation instructions and response tone. Clarification failures also decline consistently ($-$13.1%/$-$16.4%). Knowledge failures, by contrast, plateau after v1 (0% further reduction in v2), suggesting that early iterations address the most salient knowledge gaps while remaining deficiencies may be constrained by the coverage of the underlying knowledge base and retrieval tools—a natural boundary for text-based skill optimization.

### 3.4 Comparison with Production Legacy System

We compare v3 against the production legacy system, which combines predefined decision-tree workflows with manually curated expert prompts tuned over an extended period. The skill-equipped agent achieves +13.76pp Strict CR over this legacy system on the same held-out set, confirming that domain-contextualized creation combined with automated self-evolution can surpass mature, human-engineered production systems. A concrete end-to-end illustration of one evolution cycle—from failure aggregation through diagnosis to the resulting optimization plan—is provided in Appendix F.

## 4 Related Work

### 4.1 LLM-based Agents and Tool Use

LLM-based autonomous agents have advanced rapidly, with architectures integrating planning, memory, and tool use into general-purpose problem solvers [1]. ReAct [6] established the dominant paradigm of interleaving reasoning traces with executable actions, while Toolformer [7] and ToolLLM [8] demonstrated that LLMs can learn to invoke external APIs at scale. DSPy [24] introduced declarative programming abstractions that compile LLM pipelines into self-improving systems. These works operate at the prompt or API-call level, optimizing individual model interactions rather than managing reusable, versionable skill artifacts that encapsulate domain procedures, tools, and knowledge as cohesive packages.

### 4.2 Agent Skill Construction and Management

The concept of agent skills—portable, file-based packages that bundle instructions, tool schemas, and reference materials—was formalized by Anthropic [5] and empirically validated by SkillsBench [15], which showed that curated skills improve agent success rates by 16.2pp while self-generated skills provide negligible benefit. Voyager [25] pioneered automatic skill acquisition through open-ended exploration in Minecraft, building a growing skill library via code-based verification. Recent work has expanded along several axes: SkillX [26] constructs hierarchical skill knowledge bases at strategic, functional, and atomic levels; AgentSkillOS [27] organizes large-scale skill ecosystems via capability trees and DAG-based orchestration; PolySkill [28] introduces polymorphic abstractions enabling cross-domain skill generalization; and AgentFactory [29] proposes a three-stage lifecycle for progressively accumulating executable sub-agents. TARSE [30] explicitly separates reusable skills from episodic experiences for test-time adaptation. Unlike these works, which focus on skill construction, organization, or generalization in open-ended or benchmark settings, SkillForge addresses the distinct challenge of grounding skill creation in enterprise domain knowledge and closing a deployment-feedback-driven self-evolution loop.

### 4.3 Self-Evolving Agent Systems

Self-improvement in LLM agents spans multiple granularities. At the output level, Self-Refine [9] enables iterative refinement through self-feedback, and Reflexion [10] maintains verbal experience for trial-and-error learning. ExpeL [11] extracts reusable insights from execution trajectories without parameter updates. At the prompt level, APE [12], OPRO [13], and TextGrad [14] optimize instructions via search or text-based gradients. At the agent-architecture level, Symbolic Learning [17] enables post-deployment self-evolution of agent components, ADAS [19] searches over agent designs, and Gödel Agent [31] proposes recursive self-improvement without pre-defined optimization algorithms. Among skill-oriented approaches, MemSkill [18] reconstructs memory operations as evolvable skills, EvoSkills [32] introduces co-evolutionary verification between skill generators and verifiers, Steve-Evolving [33] distills success trajectories into reusable skills and failure trajectories into executable guardrails, and AutoAgent [34] continuously evolves prompt-level cognition with elastic memory. SkillForge differs from these approaches in two key respects: (1) it targets the skill artifact itself—a structured package of instructions, tools, and knowledge—rather than raw prompts or agent architectures; and (2) it operates a structured three-stage pipeline (Failure Analyzer, Skill Diagnostician, Skill Optimizer) that traces deployment failures back to specific skill deficiencies, enabling targeted rather than holistic rewrites.

### 4.4 LLM in Enterprise Operations and Customer Support

LLMs are increasingly deployed in cloud and IT operations. RCACopilot [2] automates incident root cause analysis at Microsoft, RCAgent [3] enables autonomous cloud fault diagnosis with tool-augmented agents, and D-Bot [4] applies LLM-driven tree search for database anomaly diagnosis. Broader surveys of LLM-based AIOps highlight the growing adoption of these techniques in failure management [35]. In customer support, Agent-in-the-Loop [16] implements a continuous improvement flywheel driven by four types of human annotation feedback. SkillForge complements this line of work by automating the feedback loop at the skill level—replacing human annotation with LLM-judged failure analysis and automated skill rewriting—while targeting the specific challenges of cloud technical support where domain knowledge, tool usage, and procedural workflows must be jointly optimized within a single skill package.

## 5 Conclusion

We presented SkillForge, an end-to-end creation–evaluation–refinement framework for domain-specific, self-evolving enterprise agent skills. A Domain-Contextualized Skill Creator grounds initial skill synthesis in historical tickets and domain knowledge, while an automated three-stage pipeline (Failure Analyzer, Skill Diagnostician, Skill Optimizer) continuously diagnoses execution failures and rewrites the skill to eliminate them. Evaluated on five real-world cloud technical support scenarios, S_domain outperforms the generic baseline by +4.3pp Strict CR and +3.6pp Lenient CR, and the self-evolution loop delivers consistent Strict CR gains of 9–12pp across three iterations regardless of starting point—demonstrating that automated evolution driven by deployment feedback can systematically surpass manually curated expert knowledge.

## References

- [1] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents, 2024. Submitted Aug 2023, revised Jan 2024.

- [2] Yinfang Chen, Huaibing Xie, Minghua Ma, Yu Kang, Xin Gao, Liu Shi, Yunjie Cao, Xuedong Gao, Hao Fan, Ming Wen, Jun Zeng, Supriyo Mandal, Xiaohua Jing, Chenyu Zhao, Jiahao Li, Sheryn Tai, Jom Dora, Tingting Liu, Longfei Li, Guoyao Xu, Yunlong Zhang, Rodrigo Fonseca, Saravan Rajmohan, and Thomas Moscibroda. Automatic root cause analysis via large language models for cloud incidents. In Proceedings of the Nineteenth European Conference on Computer Systems (EuroSys), 2024. EuroSys 2024.

- [3] Zelin Wang, Zhaoyang Shen, Chao Ma, Jiaming Zhang, Fuyuan Zhou, Hongtao Zhang, Jianpeng Yao, Kai Liu, Kunyi Li, Qiyu Liao, Liuqing Shen, Jianhui Deng, Bing Guo, Ye Li, Hongfeng Jiang, Juntao Wang, Guangbo Yang, and Yang Chen. RCAgent: Cloud root cause analysis by autonomous agents with tool-augmented large language models. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM), 2024. CIKM 2024.

- [4] Xuanhe Zhou, Guoliang Li, Zhaoyan Sun, Zhiyuan Liu, Weize Chen, Jianming Wu, Jiesi Liu, Ruohang Feng, and Guoyang Zeng. D-Bot: Database diagnosis system using large language models. Proceedings of the VLDB Endowment, 17(11), 2024. VLDB 2024.

- [5] Barry Zhang, Keith Lazuka, and Mahesh Murag. Equipping agents for the real world with agent skills. https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills, October 2025. Published Oct 16, 2025.

- [6] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. Last revised Mar 10, 2023.

- [7] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. Submitted Feb 9, 2023.

- [8] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs, 2024. ICLR 2024 Spotlight.

- [9] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2023. NeurIPS 2023.

- [10] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. NeurIPS 2023.

- [11] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. ExpeL: LLM agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 2024. AAAI 2024.

- [12] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers, 2023. ICLR 2023.

- [13] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers, 2023. NeurIPS 2023.

- [14] Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. TextGrad: Automatic “differentiation” via text. 2024. Published in Nature, 2024.

- [15] Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han chung Lee. Skillsbench: Benchmarking how well agent skills work across diverse tasks, 2026. Submitted Feb 13, 2026.

- [16] Cen Zhao, Tiantian Zhang, Hanchen Su, Yufeng Zhang, Shaowei Su, Mingzhi Xu, Yu Liu, Wei Han, Jeremy Werner, Claire Na Cheng, and Yashar Mehdad. Agent-in-the-loop: A data flywheel for continuous improvement in LLM-based customer support. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1919–1930, 2025. EMNLP 2025 Industry Track.

- [17] Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, et al. Symbolic learning enables self-evolving agents, 2024. Submitted Jun 2024.

- [18] Viktor Axelsen et al. Memskill: Learning and evolving memory skills for self-evolving agents, 2026. Submitted Feb 2026.

- [19] Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems, 2024. ICLR 2025 Outstanding Paper.

- [20] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. NeurIPS 2022.

- [21] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023. NeurIPS 2024.

- [22] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, et al. AgentBench: Evaluating LLMs as agents, 2023. ICLR 2024.

- [23] Sayash Kapoor, Benedikt Ströbl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter, 2024. Submitted Jul 2024.

- [24] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, et al. DSPy: Compiling declarative language model calls into self-improving pipelines, 2023. ICLR 2024.

- [25] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023. NeurIPS 2023 Workshop.

- [26] Zhixin Zhang, Jian Yang, Yifan Yu, Jiayi Zhang, and Zhoujun Li. SkillX: Automatically constructing skill knowledge bases for complex agent tasks, 2026. Submitted Apr 2026.

- [27] Jinyu Xiang, Tao Wang, Qi Zhang, and Xuanjing Huang. AgentSkillOS: Organizing LLM-based agent skills via capability trees and DAG orchestration, 2026. Submitted Mar 2026.

- [28] Yixiao Wang, Qi Liu, and Enhong Chen. PolySkill: Polymorphic skill abstraction for cross-domain agent generalization, 2026. ICLR 2026.

- [29] Zihao Wang, Shaofei Cai, Anji Liu, and Yitao Liang. AgentFactory: Automatically accumulating executable sub-agents via progressive skill refinement, 2026. Submitted Mar 2026.

- [30] Yuxuan Jiang et al. TARSE: Test-time adaptation via retrievable skills and experiences for LLM agents, 2026. Submitted Mar 2026.

- [31] Xunjian Yin et al. Gödel agent: A self-referential framework for agents recursively self-improvement, 2024. Submitted Oct 2024.

- [32] Jialu Zhang, Xiangru Tang, Junyu Luo, Yilun Zhao, Arman Cohan, and Mark Gerstein. Self-evolving agent skills via co-evolutionary verification, 2026. Submitted Apr 2026.

- [33] Zihao Wang, Shaofei Cai, Anji Liu, and Yitao Liang. Dual-track knowledge distillation: Learning skills from success and guardrails from failure, 2026. Submitted Mar 2026.

- [34] Qiushi Sun et al. AutoAgent: Evolving cognition and elastic memory for LLM-based agents, 2026. Submitted Mar 2026.

- [35] Lingzhe Zhang, Tong Jia, Mengxi Jia, Yifan Wu, Aiwei Liu, Yong Yang, and Zhonghai Wu. A survey of AIOps for failure management in the era of large language models, 2024. Submitted Jun 2024.

## Appendix A Domain-Contextualized Skill Creator Details

The Domain-Contextualized Skill Creator addresses the cold-start problem by generating a robust initial skill (Skill_v0) from historical ticket data and domain documentation. Due to enterprise data privacy requirements, all mining steps use an internal LLM (Qwen3-Max) to process proprietary data. The Creator operates through four sequential stages: Workflow Mining, Tool Mining, Knowledge Extraction, and Skill Synthesis.

### Workflow Mining

Workflow Mining extracts typical solution patterns from historical ticket dialogues, transforming unstructured human-agent conversations into structured resolution traces that capture both explicit actions and implicit expert reasoning.

Input. The mining operates on individual ticket dialogue segments. Each segment contains the raw conversation between the customer and human agent, along with a coarse category tag (e.g., “DNS resolution failure”).

Process. An LLM is prompted to act as a process mining engine that reverse-engineers the human agent’s thought-action chain from each dialogue. Specifically, the LLM performs the following for every ticket:

-

Problem Identification: Extract a detailed, self-contained problem description that enriches the coarse category tag with contextual information from the dialogue, so the description can be understood independently.

-

Resolution Path Reconstruction: Reconstruct the complete handling workflow by identifying the key phases—clarification (how the agent pinpointed the actual problem), information gathering (what data the agent requested and why), diagnosis/execution (the reasoning chain and actions taken), and solution delivery (the final resolution or escalation).

-

Experience Distillation: Extract reusable lessons, explicitly labeled as positive patterns (effective strategies worth replicating) or negative warnings (pitfalls to avoid).

-

Exemplar Response Extraction: Select high-quality verbatim responses from the human agent that demonstrate effective communication at critical moments—such as de-escalating customer frustration, explaining a complex technical constraint, or guiding the customer through a multi-step operation. Each extracted response is annotated with its usage context (e.g., “soothing emotion while setting expectations”), enabling the skill to include scenario-specific response templates grounded in proven human phrasing rather than LLM-generated text.

To comply with data governance requirements, the prompt enforces anonymization rules that replace personally identifiable information (names, phone numbers, order IDs, domain names, etc.) with typed placeholders while preserving business semantics.

Output. Each ticket produces a structured JSON record containing: core_issue (enriched problem description), resolution_path (step-by-step narrative), accumulated_experience (list of labeled lessons), and exemplar_responses (annotated human agent responses with usage context). These records collectively form the input dataset for subsequent stages.

Aggregation. After processing all tickets, the individual records are grouped by topic. Statistical analysis identifies high-frequency problem patterns and recurring workflow structures (e.g., Clarify $\rightarrow$ Diagnose $\rightarrow$ Resolve). These aggregated patterns become the backbone of the skill’s scenario-specific handling procedures.

### Tool Mining

Tool Mining identifies the tools that human experts frequently invoke during ticket resolution, and extracts their schemas for inclusion in the skill definition.

Process. Historical tickets contain operation logs that record which internal tools (APIs, diagnostic utilities, configuration interfaces) human agents used during each resolution. The mining step:

-

Parses operation logs across all tickets in the target domain to build a tool invocation frequency table.

-

Applies a frequency threshold to select high-utility tools, filtering out rarely used or deprecated ones.

-

Extracts the schema for each selected tool (name, description, parameters, return values) from the internal tool registry.

-

Associates each tool with the scenarios in which it is most commonly used, based on co-occurrence statistics.

Output. A tools.json file containing the selected tool schemas, along with per-scenario tool usage annotations that inform the skill’s workflow steps about when and how to invoke each tool.

### Knowledge Extraction

Knowledge Extraction gathers domain-specific information that the agent needs beyond what is captured in resolution traces—product rules, technical specifications, and troubleshooting references.

Process. Knowledge is collected from two complementary sources:

-

Documentation Search: The system constructs search queries from the task description and identified scenarios, then retrieves relevant articles from the internal knowledge base and official product documentation sites. Retrieved content is filtered for relevance and condensed into concise reference documents.

-

Ticket-Cited References: Historical tickets often contain links to documentation articles that human agents consulted during resolution. These cited references are collected, deduplicated, and organized by scenario, providing a curated set of authoritative sources validated by actual usage.

Output. A set of reference documents under references/, each covering a specific knowledge area (e.g., product-specific configuration rules, common error codes and their causes). Core decision-critical knowledge is flagged for inclusion directly in SKILL.md, while detailed background material remains in reference files.

### Skill Synthesis

Skill Synthesis assembles the outputs of the three mining stages into a structured skill package by filling a pre-defined template.

Template Structure. The skill package follows a fixed directory layout:

skill_name/ |-- SKILL.md # Core instructions and workflows +-- references/ |-- tools.json # Tool schemas from Tool Mining |-- knowledge_*.md # Domain knowledge from Knowledge Extraction +-- ...

The SKILL.md document is organized into the following sections:

-

Background Knowledge: Clarifies concepts most commonly misunderstood by customers, sourced from Knowledge Extraction and cross-scenario analysis of mined workflows. This section is intentionally prioritized as the primary determinant of skill quality.

-

Scenario Triage: A decision tree derived from the scenario classification in Workflow Mining, enabling the agent to route incoming requests to the appropriate handling procedure based on observable signals.

-

Per-Scenario Handling (typically 4–8 scenarios): Each scenario includes an applicability description, a branching workflow reconstructed from mined resolution paths (not a flat procedure), tool invocation guidance from Tool Mining, specific failure causes paired with resolution steps, and an escalation fallback.

-

FAQ: Covers long-tail issues that appear in fewer than 20 tickets—too infrequent for a dedicated scenario but validated by real occurrence.

-

Reference Index: Pointers to the reference documents and tool schemas bundled in the skill package.

Synthesis Constraints. The Creator enforces several quality constraints during synthesis:

-

All procedures and explanations must trace to actual resolution paths in the mined data, not to general product knowledge.

-

Handling workflows must encode conditional branching logic rather than presenting linear sequences.

-

Error symptoms must be paired with specific root causes and resolution steps.

-

Every scenario must include an explicit escalation path for unresolvable cases.

-

When SKILL.md exceeds 500 lines or 10K characters, detailed content is offloaded to reference files while the main document retains decision logic and workflow steps.

## Appendix B Agent Execution Environment Details

SkillForge agents operate within a sandboxed Virtual File System (VFS) that provides complete file operations while ensuring execution safety and traceability.

### B.1 Virtual File System Design

The VFS is a pure in-memory file system abstraction implemented as a key-value store mapping absolute paths to file nodes. Each node contains metadata (name, type, timestamps, size) and content (for files). The system supports standard operations including read, write, delete, rename, copy, and directory manipulation (mkdir, list, chdir), plus Unix-like utilities (grep, head, tail, find).

All operations return a unified result structure {success, data, error, message} without throwing exceptions, simplifying error handling for agents. Path normalization handles relative paths and special directories (., ..) automatically.

### B.2 Integration with Agent Workflow

At session initialization, the VFS is populated with task-specific files (skill definitions, reference materials). Agents interact through standard tool interfaces (e.g., read_file, write_file), with all operations logged in the execution trace for debugging and failure analysis.

The VFS provides complete isolation from the host file system while maintaining familiar file operation semantics, enabling agents to work with complex file structures safely and efficiently.

## Appendix C Failure Analyzer Details

The Failure Analyzer performs systematic root cause diagnosis of agent failures through parallel multi-dimensional analysis.

### Four-Dimension Analysis Framework

Each bad case is analyzed concurrently across four dimensions, each producing structured output with severity assessment and issue categorization:

-

Style Analysis: Evaluates expression quality (robotic, verbose, cold, inappropriate tone). Style issues only matter when semantic content is correct.

-

Knowledge Analysis: Identifies knowledge-level problems including missing information, factual errors, contradictions, outdated content, misapplication, or failure to surface existing knowledge.

-

Tool Analysis: Examines tool invocation behavior for missed calls, wrong tool selection, incorrect parameters, repeated calls, result misinterpretation, or underutilization.

-

Clarification Analysis: Assesses information gathering strategy appropriateness (over-clarification, under-clarification, wrong clarification focus).

### Aggregation Mechanism

Results from the four dimensions are aggregated through deterministic code logic to produce:

-

failure_categories: List of dimensions with detected issues

-

overall_severity: Maximum severity across dimensions (high/medium/low/none)

-

overall_verdict: fail (high$\geq$1 or medium$\geq$2), marginal (medium=1 or low$\geq$1), or acceptable

-

primary_category: Highest severity dimension, with priority order knowledge $>$ tool $>$ clarification $>$ style when tied

An optional LLM aggregation step adds natural language summary (divergence_summary) and actionable diagnostic hints for skill improvement.

### Batch Result Aggregation

After analyzing multiple bad cases, individual results are aggregated by failure category to identify systematic issues. For each category (knowledge, tool, clarification, style), the aggregation computes:

-

Severity distribution (counts of high/medium/low cases)

-

Issue type frequencies (e.g., knowledge:missing appears 8 times)

-

Representative cases (top-k by severity and diversity)

-

Aggregated diagnostic hints (deduplicated across cases)

This category-level aggregation enables the Skill Diagnostician to identify patterns across failures rather than treating each case in isolation, facilitating more effective skill improvements.

## Appendix D LLM-Judge Evaluation Details

The LLM-Judge evaluates agent response quality by comparing against reference responses from human experts. It operates independently from the Failure Analyzer: the Judge determines whether a response is acceptable, while the Analyzer diagnoses why failures occur. This separation allows the Judge to serve as a proxy for business metrics (e.g., user satisfaction) while the Analyzer focuses on execution trace diagnosis.

The Judge receives four inputs for each evaluation: (1) a global ticket summary containing the customer’s core problem, interaction history, and final resolution; (2) the dialogue history between customer and agent up to the current turn; (3) the reference response from a human expert; and (4) the agent’s actual response to be evaluated. The Judge first extracts the core action from the reference response, filtering out boilerplate (e.g., greetings, closing remarks), and then compares the agent’s response against this core action using the three-tier consistency criteria defined in Section 3.1.2. Reference responses are treated as one acceptable solution, not the unique correct answer.

Each evaluation produces structured JSON output: {verdict, ref_core_action, actual_action, reason}, where verdict $\in$ {consistent, partial, inconsistent}, ref_core_action summarizes the substantive action in the reference, actual_action summarizes the agent’s action, and reason provides brief justification for human review. Cases with verdict = inconsistent or partial are forwarded to the Failure Analyzer as bad cases.

The full evaluation prompt is shown below.

## Appendix E Skill Diagnostician and Optimizer Details

This appendix describes the Skill Diagnostician and Optimizer, which translate failure analysis results into concrete skill improvements.

### E.1 Skill Diagnostician

The Diagnostician is implemented as a ReAct agent that traces aggregated failure patterns to specific skill defects.

#### Diagnostic Workflow

The Diagnostician operates through four stages:

-

Skill Understanding: Read and summarize the structure of SKILL.md and reference files

-

Evidence Collection: Parse aggregated FA results to extract representative cases and diagnostic hints

-

Root Cause Attribution: For each failure category, map issues to specific SKILL.md locations and classify defect types (missing, insufficient, incorrect)

-

Optimization Plan Generation: Produce prioritized, actionable modification recommendations with evidence support

#### Attribution Patterns

Common mappings from FA categories to skill defects include:

*Table 4: Failure Category to Skill Defect Mapping*

| FA Category:Type | Likely Skill Defect |

| knowledge:missing | SKILL.md lacks knowledge for this scenario |

| knowledge:not_surfaced | Knowledge exists but poorly organized |

| tool:missed_call | No guidance on when to call this tool |

| tool:wrong_params | Insufficient parameter documentation |

| clarification:over_asking | Information gathering rules too broad |

| style:verbose | Lacks concise response guidelines |

#### Diagnostic Report Structure

The output diagnostic report contains: (1) overview with top issues and category distribution, (2) per-category analysis linking evidence to skill locations, and (3) prioritized optimization plan. Each recommendation specifies the modification location, content changes, whether examples or knowledge search are needed, expected impact, and risk assessment.

### E.2 Skill Optimizer

The Optimizer executes modifications specified in the diagnostic report, following strict principles to ensure positive evolution.

#### Core Principles

-

Minimal Modification: Only change what’s necessary to address diagnosed issues

-

Do No Harm: Preserve existing correct behaviors through additive changes; never delete working content

-

Evidence-Based: Every modification must trace to specific FA evidence from bad cases

#### Optimization Workflow

The Optimizer processes the diagnostic report through these steps:

-

Read diagnostic report and identify similar recommendations for merging

-

Understand original SKILL.md structure to determine appropriate insertion points

-

Apply modifications by priority, consulting category analysis files for detailed evidence

-

Add examples when specified (using reference excerpts from FA results)

-

Perform deduplication check to remove redundant content

-

Verify changes meet safety criteria (additive, consistent, evidence-backed)

#### Content Placement Strategy

New content is inserted following the original SKILL.md structure: background knowledge goes in dedicated knowledge sections, tool call rules are embedded in relevant workflow steps, style guidelines appear near response templates, and examples immediately follow related rules. When SKILL.md exceeds size thresholds (500 lines or 10K characters), detailed content is moved to reference files while keeping decision logic in SKILL.md.

#### Knowledge Augmentation

When the diagnostic report indicates missing knowledge, the Optimizer uses a search tool to retrieve relevant information from the knowledge base. Retrieved content is filtered for relevance, then integrated into appropriate locations—core decision rules in SKILL.md, detailed background in reference files.

## Appendix F Case Study: Skill Self-Evolution

This appendix presents a concrete case study from the OSS (Object Storage Service) error diagnosis skill, illustrating how the SkillForge pipeline traces agent failures through aggregated analysis to a structured optimization plan.

### Skill Overview

The target skill handles customer tickets related to OSS error diagnosis. The agent’s workflow involves: (1) identifying whether the issue is an OSS error (upload failure, access denied, 5xx, etc.); (2) collecting key information (RequestID, bucket name, error messages); (3) invoking diagnostic tools (request log lookup, bucket info query, knowledge search); and (4) providing a precise resolution. The evaluation dataset contains 108 bad cases where the agent’s response was judged inconsistent with expert reference responses.

### Aggregated Failure Analysis Results

After the Failure Analyzer processes all 108 bad cases across four dimensions, the aggregated statistics reveal the following distribution:

*Table 5: Failure category distribution across 108 bad cases. Each case may have issues in multiple dimensions.*

| Category | Count | High | Medium | Top Issue Types |

| Style | 108 | 72 | 36 | verbose (78), robotic (25) |

| Clarification | 97 | 64 | 33 | over_clarification (59), under_clarification (31) |

| Knowledge | 76 | 48 | 28 | missing (48), not_surfaced (19), incorrect (13) |

| Tool | 59 | 43 | 16 | missing_call (34), tool_missing (21) |

Three systemic issues emerge from the aggregation: (1) verbose and robotic responses dominate the style dimension; (2) the agent over-clarifies by requesting information the customer has already provided; (3) critical domain knowledge is missing, causing the agent to give incorrect guidance.

### Representative Failure Cases

Below are three representative cases selected from the aggregated results, each illustrating a distinct failure pattern. For each case, we show the divergence summary produced by the Failure Analyzer (explaining how the agent’s response diverged from the expert reference) and the automatically generated diagnostic hints.

### Diagnostic Report and Optimization Plan

The Skill Diagnostician reads the aggregated failure analysis and the current SKILL.md, then maps failure patterns to specific skill defects. Below is the resulting optimization plan with three prioritized actions.

### Analysis

This case study illustrates several key properties of the SkillForge pipeline:

Multi-dimensional attribution. Each failure case is analyzed across four dimensions simultaneously. Case 1, for example, is primarily a knowledge gap, but it cascades into tool, clarification, and style failures—showing that a single root cause can manifest across multiple dimensions. The aggregation step correctly identifies knowledge:missing as the primary category.

Evidence-to-location mapping. The Diagnostician does not produce generic advice. Each optimization item specifies the exact file and line range to modify, the concrete content change, whether knowledge search or examples are needed, and the expected impact in terms of resolved cases. This structured format enables the downstream Skill Optimizer to execute modifications precisely.

Prioritized optimization. The three priorities are ordered by both impact (number of cases resolved) and risk. Knowledge gaps (Priority 1) are addressed first because they are the root cause of cascading failures; style and clarification issues (Priority 2) affect the largest number of cases but are lower risk; tool improvements (Priority 3) have medium risk due to external dependencies on tool availability.

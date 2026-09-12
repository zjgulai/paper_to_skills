<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2603.14864
     paper_id : 2603.14864
     source   : https://arxiv.org/html/2603.14864v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks

Zijian Yu*    Kejun Xiao*†    Huaipeng Zhao    Tao Luo    Xiaoyi Zeng Affiliation: Alibaba International Digital Commercial Group Affiliation: {yuzhan.yzj, xiaokejunkejun.xia}@alibaba-inc.com

###### Abstract

In e-commerce, LLM agents show promise for shopping tasks such as recommendations, budgeting, and bundle deals, where accurately capturing user preferences from long-term conversations is critical. However, two challenges hinder realizing this potential: (1) the absence of benchmarks for evaluating long-term preference-aware shopping tasks, and (2) the lack of end-to-end optimization due to existing designs that treat preference identification and shopping assistance as separate components. In this paper, we introduce a novel benchmark with a long-term memory setup, spanning two shopping tasks over 1.2 million real-world products, and propose Shopping Companion, a unified framework that jointly tackles memory retrieval and shopping assistance while supporting user intervention. To train such capabilities, we develop a dual-reward reinforcement learning strategy with tool-wise rewards to handle the sparse and discontinuous rewards inherent in multi-turn interactions. Experimental results demonstrate that even state-of-the-art models (such as GPT-5) achieve success rates under 70% on our benchmark, highlighting the significant challenges in this domain. Notably, our lightweight LLM, trained with Shopping Companion, consistently outperforms strong baselines, achieving better preference capture and task performance, which validates the effectiveness of our unified design.

## 1 Introduction

*Figure 1: An example of the Shopping Companion framework. Given a user instruction, the agent first performs Preference Identification (Stage 1) by invoking memory tools to retrieve relevant conversation history and extract preferences. These are presented to the user for confirmation, giving the user the ability to intervene. Then, in Shopping Assistance (Stage 2), the agent leverages the confirmed preferences to iteratively search for products and check constraints until the task is completed.*

| Benchmark | Task | LTM | Interv. |

| WebShop (Yao et al., 2022) | ✓ | ✗ | ✗ |

| LongMemEval (Wu et al., 2024) | ✗ | ✓ | ✗ |

| ShoppingBench (Wang et al., 2025a) | ✓ | ✗ | ✗ |

| ShopSimulator (Wang et al., 2026) | ✓ | ✗ | ✓ |

| Ours | ✓ | ✓ | ✓ |

*Table 1: Comparison of our benchmark with existing works. Task denotes real-world downstream tasks. LTM signifies long-term memory evaluation. Interv. refers to multi-turn user intervention.*

Large language model (LLM) agents are increasingly deployed for e-commerce tasks such as product recommendation, budget management, and bundle-deals. Unlike one-shot QA, shopping assistance is inherently action-centric: agents must iteratively query large product databases, apply structured constraints, and revise decisions based on intermediate results. A defining challenge is that success depends on long-term user preferences expressed implicitly across extended conversations (e.g., brand aversions, size/fit history, aesthetics).

Existing benchmarks, meanwhile, rarely couple cross-session preference memory with end-to-end task evaluation. As summarized in Table 1, WebShop (Yao et al., 2022) offers scalable e-commerce search but operates within a single session without long-term memory. LongMemEval (Wu et al., 2024) rigorously evaluates LTM across sessions but does not ground evaluation in downstream tasks. ShoppingBench (Wang et al., 2025a) broadens intent coverage but likewise lacks both LTM evaluation and interactive feedback. ShopSimulator (Wang et al., 2026) introduces multi-turn interaction but relies on static, expert-summarized preferences rather than a persistent memory store. To the best of our knowledge, no existing benchmark simultaneously satisfies long-term memory, real-world shopping tasks, and interactive user intervention within a unified evaluation framework.

To address these gaps, we introduce a novel benchmark incorporating the above elements and propose Shopping Companion, a unified framework that jointly optimizes long-term memory retrieval and shopping task execution while supporting user intervention. We develop a dual-reward RL strategy with tool-wise rewards to handle sparse feedback in multi-turn interactions. Experiments demonstrate that our lightweight model consistently outperforms baselines in preference capture and task success, and that user intervention—from low hints to detailed corrections—reliably improves final outcomes.

The contributions of this paper are summarized as follows:

-

We introduce a novel long-horizon e-commerce benchmark designed to evaluate preference-grounded shopping success across sessions.

-

We propose Shopping Companion, a unified framework that jointly optimizes long-term memory retrieval and shopping task execution, with explicit support for user intervention.

-

We develop a dual-reward RL training strategy with tool-wise rewards to handle sparse and discontinuous feedback in multi-turn tool-augmented interactions.

## 2 Related Work

##### Long-Term Memory.

Long-term memory is increasingly essential for LLM agents operating over extended horizons. A dominant paradigm externalizes memory into retrievable records injected into generation via retrieval-augmented pipelines (Lewis et al., 2020; Karpukhin et al., 2020; Guu et al., 2020). Recent systems further model memory as an explicit read/write subsystem with structured organization (Zhong et al., 2024; Packer et al., 2023; Xu et al., 2025; Chhikara et al., 2025; Rasmussen et al., 2025). Despite these advances, benchmarks such as LongMemEval (Wu et al., 2024) reveal persistent gaps in multi-session reasoning and context efficiency, suggesting the core limitation lies not in memory structure but in optimization: LTM modules remain post-hoc components never trained end-to-end with downstream task success. Most closely related, Agentic Memory (Yu et al., 2026) takes a step toward closing this gap by training memory operations as tool actions with step-wise GRPO. Our work extends this policy-level optimization to real-world shopping tasks, jointly optimizing preference-centric LTM with task execution.

##### Shopping Agent.

E-commerce assistants require grounded interaction with large product databases and multi-turn constraint satisfaction. Prior work on conversational product search and recommendation has jointly addressed clarification questioning and item ranking (Zhang et al., 2018; Bi et al., 2019; Zou et al., 2022), with datasets such as Wizard of Shopping (Li et al., 2025) further improving realism through structured search processes. Complementing these task-oriented efforts, benchmarks like ShoppingBench (Wang et al., 2025a), Shopping MMLU (Jin et al., 2024), and EcomScriptBench (Wang et al., 2025b) evaluate broader shopping competence spanning intent-grounded end-to-end evaluation, domain knowledge, and script-level planning. Despite this progress, no existing benchmark or method explicitly couples long-term preference memory across sessions with end-to-end shopping success—most systems treat memory as an isolated, upstream stage. We address this gap by proposing a benchmark that embeds shopping preferences within long-term general conversations, and by training a unified agent policy that integrates LTM retrieval with shopping assistance.

## 3 Problem Formulation

We formulate the task as a partially observable Markov decision process defined by the tuple $(\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{O},\mathcal{R})$, consisting of a state space $\mathcal{S}$, an action space $\mathcal{A}$, a transition function $\mathcal{T}$, an observation space $\mathcal{O}$, and a reward function $\mathcal{R}$.

At each time step $t$, the agent receives an observation derived from the underlying state $s_{t}\in\mathcal{S}$, composed of the conversation context $C_{t}$, the long-term memory store $\mathcal{M}_{t}$ capturing user preferences, and the natural language instruction $\mathcal{I}$:

$s_{t}=(C_{t},\;\mathcal{M}_{t},\;\mathcal{I}).$ | | | | (1) |

Given $s_{t}$ and LLM parameters $\theta$, the agent performs an action $a_{t}\in\mathcal{A}$, interacts with the environment, and updates the state:

$(s_{t+1},\;o_{t+1})=\mathcal{T}(a_{t}|s_{t};\theta).$ | | | | (2) |

Our unified framework treats memory-based preference capture and shopping assistance as integral, jointly optimized components, rather than separate modules. To evaluate the effectiveness of this joint optimization, at the terminal state $s_{T}$, the task is successful if the agent’s final recommendation: (1) satisfies all needs $n\in\mathcal{N}(\mathcal{I})$; (2) matches all preferences $p\in\mathcal{P}(\mathcal{M})$:

$C_{\mathcal{I}}=\bigwedge_{n\,\in\,\mathcal{N}(\mathcal{I})}\textsc{Satisfy}(s_{T},\,n),$ | | | | (3) |

$C_{\mathcal{M}}=\bigwedge_{p\,\in\,\mathcal{P}(\mathcal{M})}\textsc{Match}(s_{T},\,p),$ | | | | (4) |

$\textsc{Success}(s_{T})=\begin{cases}1,&\text{if }C_{\mathcal{I}}\wedge C_{\mathcal{M}},\\[6.0pt]
0,&\text{otherwise.}\end{cases}$ | | | | (5) |

## 4 Benchmark Construction

In this section, we detail the design of our benchmark, which comprises three key components: a shopping simulation environment, natural language user instructions paired with long-horizon user-assistant conversations, and evaluation methods.

### 4.1 Shopping Simulation Environment

We construct a large-scale shopping sandbox containing over 1.2 million real-world products to enable consistent evaluation. In this simulated environment, an LLM agent interacts with the product database to generate recommendations aligned with user preferences.

To support preference retrieval and task completion, we build two search engines: one over the long-term conversation memory and another over the product database. For memory search, we embed each conversation turn using all-MiniLM-L6-v2 (Wang et al., 2020) and retrieve relevant turns via cosine similarity. For product search, we employ Pyserini (Lin et al., 2021) with BM25 (Robertson et al., 2009) to construct a sparse retrieval index offline. We develop 5 tools (Figure 1) based on these engines.

To construct a verifiable evaluation dataset for these tasks, we employ a systematic five-step synthesis pipeline (Algorithm 1). The pipeline progressively constructs diverse, realistic instances by sampling products, generating user instructions via LLMs, embedding implicit preferences through multi-turn dialogues, interleaving preference-bearing sessions with unrelated conversations (needle-in-a-haystack paradigm (Grover, 1997)), and performing manual verification. Detailed methodology and dataset statistics are provided in Appendix A.1-A.2.

*Algorithm 1 User Instructions and Preferences Generation Pipeline*

1: Product database $\mathcal{P}$, LongMemEval corpus $\mathcal{C}$, task types $\mathcal{T}$

2: Dataset $\mathcal{D}=\{(\text{instr}_{i},\text{hist}_{i},\text{ans}_{i})\}_{i=1}^{1000}$

3: for $i=1$ to 1000 do

4:   Step 1: Sample products from $\mathcal{P}$ across categories, brands, and prices

5:   if task is Add-on Deals then

6:    Generate voucher rules and valid product combinations

7:   end if

8:   Step 2: Generate brief instruction $\text{instr}_{i}$ via GPT-5

9:   Step 3: Define preferences $\text{attr}_{i}$ and generate dialogue session $\text{sess}_{i}$ via GPT-5 self-play

10:   Step 4: Sample unrelated sessions from $\mathcal{C}$ and interleave with $\text{sess}_{i}$ to create $\text{hist}_{i}$

11:   Step 5: Verify $\text{instr}_{i}$ resolvable from $\text{hist}_{i}$ and record in $\mathcal{D}$

12: end for

13: return $\mathcal{D}$

### 4.2 Evaluation Methods

We adopt an LLM-as-Judge paradigm to evaluate the final recommendations. Our meta-evaluation study demonstrates that the GPT-5 evaluator achieves more than 90% agreement with human experts. To implement this paradigm, we design two task-specific evaluation prompts, which are presented in Appendix A.3.

## 5 Shopping Companion

As motivated in Sec. 3, the target behavior is a unified shopping agent whose actions must jointly satisfy instruction-driven needs $C_{\mathcal{I}}$ and memory-driven preferences $C_{\mathcal{M}}$ at the terminal state. Meanwhile, Sec. 4 instantiates this setting as a tool-interactive environment with two retrieval engines (memory and product) and five tools built on top of them (Figure 1). This section presents how Shopping Companion operationalizes this formulation as a two-stage agent trained end-to-end with reinforcement learning under sparse feedback.

### 5.1 Two-Stage Agentic Framework

Shopping Companion employs a two-stage architecture (illustrated in Figure 1):

Stage 1 (Preference Identification) retrieves relevant conversation history via memory tools and extracts implicit user preferences (e.g., brand aversions, size history). These are presented to the user for confirmation, enabling intervention before shopping proceeds.

Stage 2 (Shopping Assistance) leverages confirmed preferences to iteratively retrieve products and verify constraint satisfaction ($C_{\mathcal{I}}$ and $C_{\mathcal{M}}$) until task completion. The prompt requires a special format for final recommended product(s), shown in Appendix B.1.

### 5.2 Reward Function Design: Dual-Reward with Tool-Wise Supervision

Our agent uses dual-reward supervision aligned with the two stages, with each stage further differentiated by task type ($b=0$ for single-product, $b=1$ for add-on-deals).

##### Dual-reward setup.

Dual-reward refers to two reward branches aligned with the two stages. Let $z\in\{1,2\}$ denote the stage label. For each sample, we compute only the reward that matches its stage:

$R_{z}(\tau_{z})=\begin{cases}R_{1}(\tau_{1}),&z=1,\\[4.0pt]
R_{2}(\tau_{2}),&z=2.\end{cases}$ | | | | (6) |

Both $R_{1}$ and $R_{2}$ are computed via LLM-based reward function with structured outputs and normalized to be comparable across instances. The reward prompts are provided in Appendix C.1.

##### Stage-1 reward

Stage-1 is scored by how well the agent grounds preferences from long-term memory. Let $F$ denote the number of required preference attributes. For add-on-deals tasks, let $N$ denote the reference bundle size. The evaluator returns three normalized signals: query relevance $q_{1}\in\{0,1\}$, attribute match $m_{1}$, and (only when $b=1$) product-count $c_{1}$. We compute:

$R_{1}(\tau_{1})=\frac{q_{1}+m_{1}+b\,c_{1}}{1+F+b\,N}.$ | | | | (7) |

##### Stage-2 reward

Stage-2 is scored by whether the agent outputs a machine-extractable recommendation and satisfies mandatory constraints, while matching intent and preferences. The evaluator returns product-validity $p_{2}\in\{0,1\}$, relevance $q_{2}\in\{0,1\}$, and attribute match $m_{2}\in\{0,1\}$. For add-on-deals tasks ($b=1$), it additionally returns count feasibility $n_{2}$ and budget feasibility $u_{2}$. The reward is:

$R_{2}(\tau_{2})=\frac{p_{2}+q_{2}+m_{2}+b\,(n_{2}+u_{2})}{2+F+b\,(N+1)}.$ | | | | (8) |

##### Tool-wise reward $R_{\mathrm{tool}}(\tau)$.

The stage rewards above are trajectory-level signals. To improve credit assignment for intermediate tool decisions, we introduce a tool-wise reward that scores each tool invocation. Let $\mathcal{U}(\tau)$ be the set of tool calls in $\tau$, and let $r(u)\in[0,1]$ be the per-call score returned by the reward server for invocation $u$(detailed in Appendix C.2). We use the mean over tool calls:

$R_{\mathrm{tool}}(\tau)=\begin{cases}\frac{1}{|\mathcal{U}(\tau)|}\sum\limits_{u\in\mathcal{U}(\tau)}r(u),&|\mathcal{U}(\tau)|>0,\\[6.0pt]
0,&\text{otherwise}.\end{cases}$ | | | | (9) |

##### Protocol shaping and final reward.

We add a lightweight format reward to stabilize structured generation under tool-call protocols. Let $f_{\mathrm{ans}},f_{\mathrm{th}},f_{\mathrm{tc}},f_{\mathrm{rec}}\in\{0,1\}$ denote whether the final answer span is extractable, the thinking tags are well-formed, tool-call JSON is parsable, and whether a recommendation output is present and conforms to the required schema:

$R_{\mathrm{fmt}}(\tau)=\frac{f_{\mathrm{ans}}+f_{\mathrm{th}}+f_{\mathrm{tc}}+\mathbb{I}[z=2]\cdot f_{\mathrm{rec}}}{3+\mathbb{I}[z=2]}.$ | | | | (10) |

With unit weights, the final reward for a training sample of stage $z$ is:

$R(\tau)=R_{z}(\tau_{z})+R_{\mathrm{tool}}(\tau)+R_{\mathrm{fmt}}(\tau).$ | | | | (11) |

## 6 Experiments

| Category | Model | Single Product | Add-on Deals | Average |

| Acc.(%) | Succ.(%) | Acc.(%) | Succ.(%) | Acc.(%) | Succ.(%) |

| Closed | GPT-5 | 82.0 | 75.0 | 66.0 | 54.0 | 74.0 | 64.5 |

| GPT-4.1 | 88.0 | 78.0 | 39.0 | 24.0 | 63.5 | 51.0 |

| GPT-4o | 79.0 | 72.0 | 41.0 | 26.0 | 60.0 | 49.0 |

| Qwen3-Max | 80.0 | 72.0 | 35.0 | 24.0 | 57.5 | 48.0 |

| Open | Qwen3-Next-80B-A3B | 63.0 | 57.0 | 29.0 | 18.0 | 46.0 | 37.5 |

| Qwen3-30B-A3B | 60.0 | 53.0 | 21.0 | 13.0 | 40.5 | 33.0 |

| Qwen3-4B | 49.0 | 44.0 | 11.0 | 6.0 | 30.0 | 25.0 |

| Ours | Qwen3-4B-LoRA | 82.0 | 72.0 | 42.0 | 31.0 | 62.0 | 51.5 |

| Qwen3-4B-LoRA + RL |

| (Dual-reward) |

89.0 81.0 50.0 38.0 69.5 59.5

| Qwen3-4B-LoRA + RL |

| (Dual&Tool-wise Reward) |

90.0 84.0 55.0 43.0 72.5 63.5

*Table 2: Main results on the our benchmark. Acc. measures extracted user preferences from memory in Stage 1; Succ. measures final recommendation in Stage 2. The best and second-best results are marked.*

### 6.1 Experimental Setup

##### Dataset.

Our benchmark contains 1,000 instructions (500 per task) split into 800 training and 200 test examples. Each instruction has 15–50 turn conversation history with embedded preferences. Further details are provided in Appendix D.

##### Baselines.

We compare closed-source (GPT, Qwen3-Max) and open-source LLMs (Qwen3) under zero-shot settings with in-context memory. We report progressive improvements on Shopping Companion: (1) Qwen3-4B + LoRA fine-tuning, (2) Qwen3-4B + LoRA + Dual-reward RL, and (3) Qwen3-4B + LoRA + Dual-reward + Tool-wise reward.

##### Metrics.

We employ LLM-based evaluators to assess the performance:

-

Accuracy (Acc.) measures Stage-1 preference grounding: the fraction of reference preference attributes successfully retrieved by the agent.

-

Success Rate (Succ.) measures end-to-end task success, checking: (1) product count correctness, (2) explicit needs satisfaction, (3) preference match, and (4) budget feasibility (for add-on deals). Success rate is the fraction of test examples that pass all checks.

### 6.2 Main Results

The results on both single-product recommendation and add-on deals shown in Table 2.

##### Closed-source LLMs.

Closed-source models achieve strong performance on single-product tasks. However, performance drops substantially on add-on deals, with success rates between 24.0% and 54.0%, indicating that multi-product coordination and constraint satisfaction remain challenging even for large-scale models.

##### Open-source LLMs.

Open-source models exhibit a clear performance gap. Qwen3-4B achieves only 49.0% Acc. and 44.0% Succ. on single products and performs poorly on add-on deals (6.0% Succ.). Scaling to 30B and 80B improves results but still trails closed-source models by 10–15% on average accuracy.

##### Shopping Companion.

Our method yields consistent improvements. LoRA fine-tuning substantially boosts the 4B backbone, and dual-reward RL further improves preference grounding and end-to-end success. With dual & tool-wise rewards, the model reaches 90.0% Acc. and 84.0% Succ. on single products and 55.0% Acc. and 43.0% Succ. on add-on deals, outperforming open-source baselines and approaching closed-source models. These results validate the effectiveness of joint optimization over memory grounding and task execution.

### 6.3 Ablation Studies

##### Two-Stage Strategy.

We conduct ablation studies comparing five strategies (details in Appendix B.2). As shown in Table 3, One-Stage achieves only 52.5% average success, with severe degradation on Add-on Deals (32.0%), showing that conflating both tasks overwhelms the LLM. Two-Stage (None) recovers to 65.0% (+23.0% on Add-on), confirming that explicit separation improves preference extraction. Simulated user feedback yields progressive gains—Low and High Hint reach 68.5% and 70.0%, respectively, narrowing the Oracle gap to 9 points—validating that coarse corrections refine preferences. Two-Stage (High) reaches 80.0% on Single tasks (vs. Oracle 85.0%), suggesting the bottleneck lies in shopping assistance rather than preference extraction, while the persistent Add-on gap indicates multi-product coordination remains a promising direction.

##### Reward Function Design.

To analyze the effect of tool-wise supervision, we compare Dual-reward and Dual & Tool-wise reward training strategies from both trajectory-level and behavioral perspectives.

| Strategy | Single | Add-on | Avg. |

| Oracle | 85.0 | 73.0 | 79.0 |

| One-Stage | 73.0 | 32.0 | 52.5 |

| Two-Stage (None) | 75.0 | 55.0 | 65.0 |

| Two-Stage (Low) | 78.0 | 59.0 | 68.5 |

| Two-Stage (High) | 80.0 | 60.0 | 70.0 |

*Table 3: Success Rates (%) Across Different Strategies. The backbone model is GPT-5.*

*Figure 2: Tool-wise reward effects: (a) tool-wise score and (b) response length over training steps (solid=smoothed, translucent=raw).*

Tool utilization quality. As shown in Figure 2(a), incorporating tool-wise reward consistently increases the averaged tool-wise score throughout training, with a widening gap in later stages. This indicates improved credit assignment for intermediate tool decisions, encouraging more relevant memory and product retrieval behaviors.

Efficiency and verbosity control. Figure 2(b) shows that the tool-wise variant produces shorter responses and exhibits a clearer downward trend over training. This suggests that step-level supervision not only improves tool correctness but also reduces unnecessary long-form generations, leading to more efficient trajectories.

In addition, Table 4 compares behavioral statistics across strategies. The Dual & Tool-wise reward variant demonstrates fewer redundant turns, more targeted tool usage, and shorter responses, further validating that granular reward signals effectively shape agent behavior beyond terminal success optimization.

| Strategy | Turns | Tool Uses | Resp. Len. |

| Dual-Reward | 9.82 | 9.17 | 10485.39 |

| Dual & Tool-wise | 8.89 | 8.47 | 10068.83 |

*Table 4: Behavioral Metrics Comparison(Mean Values).*

## 7 Conclusion

We study preference-grounded shopping assistance under long-term memory, introducing a benchmark that combines cross-session conversational histories, two realistic shopping tasks, and multi-turn user intervention over 1.2 million real-world products. We propose Shopping Companion, a unified framework that jointly optimizes preference identification and shopping assistance via reinforcement learning with dual rewards and tool-wise supervision. Experiments show that the benchmark remains challenging even for strong LLMs, and that our lightweight model achieves consistent gains over competitive baselines in both preference capture and task success, validating that jointly optimizing memory retrieval and downstream shopping decisions is a practical route to reliable, preference-aware e-commerce agents.

## Limitations

While our work presents a comprehensive benchmark and a unified framework for preference-grounded shopping assistance, we acknowledge the following limitations:

##### Difficulty of budget-constrained multi-item tasks.

Although Shopping Companion consistently outperforms baselines across both tasks, performance on the add-on deals task remains relatively low for all evaluated models. This task requires the agent to jointly reason over budget constraints, inter-product compatibility, and diverse user preferences—a combinatorial challenge that proves substantially harder than single-item recommendation. The results suggest that current approaches, including ours, still have significant room for improvement in handling constrained, multi-product optimization within conversational settings.

##### Generalizability of tool-wise reward design.

Our dual-reward RL strategy with tool-wise rewards is designed around the specific tool set used in our shopping tasks. Extending this approach to other domains or broader tool-augmented agent settings may require non-trivial adaptation, as decomposing sparse task-level feedback into fine-grained, per-tool supervision depends on domain-specific considerations. How to systematically and scalably design tool-wise rewards in more general settings remains an open question.

## Ethical Considerations

This work studies e-commerce assistance with long-term preference memory, raising privacy and fairness concerns. First, cross-session memory may expose sensitive information if mishandled; deployments should minimize stored content, enforce retention policies, encrypt data, and provide user controls to inspect, correct, and delete memories. Second, preference inference can enable unwanted profiling or amplify biases; systems should avoid inferring sensitive attributes without explicit user intent and should be audited for disparate outcomes. Finally, recommendations can shape spending behavior; agents should communicate uncertainty, avoid manipulative framing, and prioritize user-aligned constraints (e.g., budget and safety).

## References

- Bi et al. (2019) K. Bi, Q. Ai, Y. Zhang, and W. B. Croft Conversational product search based on negative feedback. In Proceedings of the 28th acm international conference on information and knowledge management, pp. 359–368. Cited by: §2.

- Chhikara et al. (2025) P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav Mem0: building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413. Cited by: §2.

- Grover (1997) L. K. Grover Quantum mechanics helps in searching for a needle in a haystack. Physical review letters 79 (2), pp. 325. Cited by: §4.1.

- Guu et al. (2020) K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang Retrieval augmented language model pre-training. In International conference on machine learning, pp. 3929–3938. Cited by: §2.

- Jin et al. (2024) Y. Jin, Z. Li, C. Zhang, T. Cao, Y. Gao, P. Jayarao, M. Li, X. Liu, R. Sarkhel, X. Tang, et al. Shopping mmlu: a massive multi-task online shopping benchmark for large language models. Advances in Neural Information Processing Systems 37, pp. 18062–18089. Cited by: §2.

- Karpukhin et al. (2020) V. Karpukhin, B. Oguz, S. Min, P. S. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih Dense passage retrieval for open-domain question answering.. In EMNLP (1), pp. 6769–6781. Cited by: §2.

- Lewis et al. (2020) P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33, pp. 9459–9474. Cited by: §2.

- Li et al. (2025) X. Li, Z. Chen, J. I. Choi, N. Vedula, B. Fetahu, O. Rokhlenko, and S. Malmasi Wizard of shopping: target-oriented e-commerce dialogue generation with decision tree branching. arXiv preprint arXiv:2502.00969. Cited by: §2.

- Lin et al. (2021) J. Lin, X. Ma, S. Lin, J. Yang, R. Pradeep, and R. Nogueira Pyserini: a python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pp. 2356–2362. Cited by: §4.1.

- Packer et al. (2023) C. Packer, V. Fang, S. Patil, K. Lin, S. Wooders, and J. Gonzalez MemGPT: towards llms as operating systems.. Cited by: §2.

- Rasmussen et al. (2025) P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, and D. Chalef Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956. Cited by: §2.

- Robertson et al. (2009) S. Robertson H. Zaragoza et al. The probabilistic relevance framework: bm25 and beyond. Foundations and trends® in information retrieval 3 (4), pp. 333–389. Cited by: §4.1.

- Sheng et al. (2024) G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu HybridFlow: a flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256. Cited by: §D.2.

- Wang et al. (2025a) J. Wang, K. Xiao, Q. Sun, H. Zhao, T. Luo, J. D. Zhang, and X. Zeng ShoppingBench: a real-world intent-grounded shopping benchmark for llm-based agents. arXiv preprint arXiv:2508.04266. Cited by: Table 1, §1, §2.

- Wang et al. (2026) P. Wang, Y. Wu, X. Song, W. Wang, G. Chen, Z. Li, K. Yan, K. Deng, Q. Liu, S. Zhao, et al. ShopSimulator: evaluating and exploring rl-driven llm agent for shopping assistants. arXiv preprint arXiv:2601.18225. Cited by: Table 1, §1.

- Wang et al. (2025b) W. Wang, L. Cui, X. Liu, S. Nag, W. Xu, C. Luo, S. M. Sarwar, Y. Li, H. Gu, H. Liu, et al. EcomScriptBench: a multi-task benchmark for e-commerce script planning via step-wise intention-driven product association. arXiv preprint arXiv:2505.15196. Cited by: §2.

- Wang et al. (2020) W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou Minilm: deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems 33, pp. 5776–5788. Cited by: §4.1.

- Wu et al. (2024) D. Wu, H. Wang, W. Yu, Y. Zhang, K. Chang, and D. Yu Longmemeval: benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813. Cited by: Table 1, §1, §2.

- Xu et al. (2025) W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang A-mem: agentic memory for llm agents. arXiv preprint arXiv:2502.12110. Cited by: §2.

- Yao et al. (2022) S. Yao, H. Chen, J. Yang, and K. Narasimhan Webshop: towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems 35, pp. 20744–20757. Cited by: Table 1, §1.

- Yu et al. (2026) Y. Yu, L. Yao, Y. Xie, Q. Tan, J. Feng, Y. Li, and L. Wu Agentic memory: learning unified long-term and short-term memory management for large language model agents. arXiv preprint arXiv:2601.01885. Cited by: §2.

- Zhang et al. (2018) Y. Zhang, X. Chen, Q. Ai, L. Yang, and W. B. Croft Towards conversational search and recommendation: system ask, user respond. In Proceedings of the 27th acm international conference on information and knowledge management, pp. 177–186. Cited by: §2.

- Zheng et al. (2024) Y. Zheng, R. Zhang, J. Zhang, Y. Ye, Z. Luo, Z. Feng, and Y. Ma LlamaFactory: unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. External Links: Link Cited by: §D.2.

- Zhong et al. (2024) W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang Memorybank: enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, pp. 19724–19731. Cited by: §2.

- Zou et al. (2022) J. Zou, J. Huang, Z. Ren, and E. Kanoulas Learning to ask: conversational product search via representation learning. ACM Transactions on Information Systems 41 (2), pp. 1–27. Cited by: §2.

## Appendix A Supplemental Details For Our Benchmark

### A.1 Basic Statistics

As shown in Figure 5, our product database comprises a total of 1,298,797 unique products, encompassing a wide range of product categories. The distributions of session count and total token count per user in the long-term conversational memory are shown in Figure 5. The distribution of the number of implicitly desired preferences for each task type is shown in Figure 5.

*Figure 3: Distribution of Product Category (Top 20)*

*Figure 4: Distribution of Sessions and Total Tokens*

*Figure 5: Distribution of Wanted Features*

### A.2 Long-Term Memory Construction

The preferences-evidence dialogue session generation prompt is shown in Figure 6. The user instruction generation prompts for Single Product and Add-on Deals are shown in Figure 7 and Figure 8, respectively.

*Figure 6: Dialogue Generation Prompt*

*Figure 7: User Instruction Generation Prompt of Single Product Rec*

*Figure 8: User Instruction Generation Prompt of Add-On Deals*

### A.3 Evaluation Methods Building

To accurately evaluate the diverse responses of LLMs, we employ an expert-written prompt to instruct GPT-5 as the correctness judge. We present the full prompts in Figure 9 and Figure 10. Since our benchmark spans two shopping tasks over a large-scale product catalog, each task type involves distinct evaluation considerations; we therefore design separate prompts for each task to enable the model to handle detailed edge cases as expert evaluators would. Specifically, for single-product recommendation, the evaluator checks whether the recommended product is relevant to the user query and contains all wanted features, matching semantically rather than requiring exact wording. For bundle-deal scenarios involving multiple products, the evaluator additionally verifies that the recommended set covers all products specified in the user query, with each product satisfying its corresponding wanted features.

To ensure the prompt-based judge has high agreement with expert judgment, we sample 25 questions per problem type, collect the trajectories from GPT-5 and GPT-4.1, and report the judgment correctness by category. As shown in Table 5, the prompt-engineered GPT-5 judge achieves reliable performance in evaluating both GPT-5 and GPT-4.1 as the backbone of agent.

*Figure 9: Single Product Rec Evaluation Prompt*

*Figure 10: Add-On Deals Evaluation Prompt*

| LLM Agent | Task Type |

| Single Product | Add-on Deals |

| GPT-5 | 0.96 (24/25) | 0.92 (23/25) |

| GPT-4.1 | 0.92 (23/25) | 0.88 (22/25) |

| Average | 0.94 | 0.90 |

*Table 5: The meta-evaluation results of GPT-5 evaluator.*

## Appendix B Two-Stage Agentic Framework

### B.1 Two-Stage Agent Prompts

The system prompts for two-stage agent shown in Figure 11 and Figure 12.

Stage 1 – Preference Retrieval Agent: Given a product search query, this agent retrieves relevant memories from the user’s dialogue history (using memory search, view, and summarization tools) to identify the user’s purchase preferences. It reasons step-by-step with multi-turn tool calls, then outputs the identified preferences and asks the user for confirmation.

Stage 2 – Shopping Assistance Agent: Given the search query and the identified user preferences from Stage 1, this agent searches for products that exactly match those preferences. It uses product search/view tools and optional web search to verify product attributes, then produces an expert-level Markdown report explaining alignment with user preferences and provides a best-matching recommendation in a special format (@REC::product_id@).

### B.2 User Simulator

We conduct ablation studies comparing five strategies: Oracle, which directly provides ground-truth preference-relevant dialogue sessions as context; One-Stage, which prompts the LLM to perform preference identification and shopping assistance end-to-end without explicit stage separation; and three Two-Stage variants that explicitly decouple the pipeline into a preference identification stage and a shopping assistance stage with distinct system prompts. In the two-stage variants, the first stage outputs retrieved memories and identified preferences for user confirmation. We implement a GPT-5-based User Simulator that judges the correctness of identified preferences under three feedback granularities: No Hint (no additional information), Low Hint (indicates whether preferences contain omissions or errors without specifying which ones), and High Hint (specifies which preference dimensions are missing or erroneous without revealing their values). This design faithfully mirrors real-world user intervention patterns, endowing our framework with practical applicability.The user simulator prompts shown in Figure 13 and Figure 14.

*Figure 11: Stage 1 System Prompt*

*Figure 12: Stage 2 System Prompt*

*Figure 13: User Simulator Prompt for Low Hint*

*Figure 14: User Simulator Prompt for High Hint*

## Appendix C Reward of Reinforcement Learning

### C.1 Dual-reward

This appendix documents the LLM-as-judge protocol used to compute the stage rewards in Sec. 5.2. We employ four task-and-stage specific prompts, corresponding to two tasks (single-product vs. add-on-deals) and two stages (Stage-1 preference grounding vs. Stage-2 product matching). All prompts are instantiated with (i) the user instruction, (ii) the agent output to be evaluated, and (iii) reference information from our benchmark annotations and product index. The judge is required to return structured outputs in a strict JSON schema to enable deterministic parsing and reward computation.

#### C.1.1 Stage-1: Preference Grounding Reward Prompt

Stage-1 rewards evaluate whether the agent correctly grounds user preferences from long-term conversations before product retrieval. Given the user query and the agent’s intermediate response, the judge assesses (i) query relevance and (ii) preference extraction quality measured by how many annotated preference attributes are surfaced. For add-on-deals, the judge additionally evaluates whether the response identifies a correct number of products aligned with the reference bundle.

#### C.1.2 Stage-2: Product Matching Reward Prompt

Stage-2 rewards evaluate whether the agent’s recommended product(s) match the user intent and satisfy preference attributes, grounded in the provided product attributes/options. Given the user query, wanted features, and the retrieved product descriptions, the judge assesses (i) query-intent relevance and (ii) preference satisfaction based on feature matches. For add-on-deals, relevance is counted at the product level and feature matches are aggregated across the bundle.

### C.2 Tool-Wise Reward

To compute the per-tool reward signal $r(u)$ in Eq. 9, we implement a reward server that maintains indexed product and memory databases, and evaluates tool invocations against gold-standard annotations.

#### C.2.1 Memory Tool Rewards

For memory retrieval tools:

-

mem_search: Given a search query $q$ and top-$k$ retrieved session indices $I_{q}$, we compute:

$r(\text{mem\_search})=\frac{\#\{i\in I_{q}:\text{sess}(i)\in\mathcal{S}_{\text{gold}}\}}{|I_{q}|}$ | | | | (12) |

where $\mathcal{S}_{\text{gold}}$ is the set of gold preference sessions in the reference annotation.

-

mem_view: Given explicitly selected session indices $I_{\text{view}}$:

$r(\text{mem\_view})=\frac{\#\{i\in I_{\text{view}}:\text{sess}(i)\in\mathcal{S}_{\text{gold}}\}}{|I_{\text{view}}|}$ | | | | (13) |

Both rewards are normalized to $[0,1]$ and measure recall against reference preference sessions.

#### C.2.2 Product Tool Rewards

For product retrieval tools:

-

product_search: Given a query $q$ with optional filters (price, shop), retrieve top-50 product IDs $P_{q}$. Score:

$r(\text{product\_search})=\frac{\#\{p\in P_{q}:p\in\mathcal{P}_{\text{gold}}\}}{|P_{q}|}$ | | | | (14) |

where $\mathcal{P}_{\text{gold}}$ are the gold product IDs from the reference answer.

-

product_view: Given selected product IDs $P_{\text{view}}$:

$r(\text{product\_view})=\frac{\#\{p\in P_{\text{view}}:p\in\mathcal{P}_{\text{gold}}\}}{|P_{\text{view}}|}$ | | | | (15) |

When exact product IDs are unavailable, we use LLM-based semantic matching (via GPT-5) to determine if a retrieved product sufficiently matches the reference wanted features.

#### C.2.3 Reward Computation Pipeline

During RL training, after the agent executes a trajectory $\tau$:

-

Extract all tool invocations and their arguments.

-

Query the reward server to compute $r(u)$ for each tool call $u$.

-

Aggregate via Eq. 9 to get $R_{\mathrm{tool}}(\tau)$.

-

Combine with $R_{z}(\tau_{z})$, $R_{\mathrm{fmt}}(\tau)$ via Eq. 11.

This enables immediate per-tool feedback without waiting for terminal-state evaluation, significantly improving credit assignment and training efficiency.

## Appendix D Experimental Implementation

### D.1 Dataset Details

For supervised fine-tuning (SFT), we generate successful trajectories via GPT-4.1 rejection sampling, yielding 2,948 step-level examples used to initialize the model.

For reinforcement learning, since our framework operates in two stages with distinct system prompts and tool sets (Stage-1 for preference identification and Stage-2 for shopping assistance), each instruction generates two separate data instances—one for each stage. This results in 1,600 training instances (800 queries $\times$ 2 stages) and 400 test instances (200 queries $\times$ 2 stages), totaling 2,000 instances. We train for 5 epochs with a learning rate of $1\times 10^{-6}$.

### D.2 Implementation Details

##### SFT.

For supervised fine-tuning (SFT), we employ LLaMA-Factory Zheng et al. (2024) and 8$\times$H20 GPUs to fine-tune Qwen3-4B-Thinking-2507 with LoRA (rank $=64$), targeting the query, key, value, and output projection layers (q_proj, k_proj, v_proj, o_proj). Training is conducted for 3 epochs with a cosine learning rate schedule (peak learning rate $=5\times 10^{-5}$), using BF16 mixed precision with an effective batch size of 4 per device.

##### RL.

For reinforcement learning, we utilize the VeRL frameworkSheng et al. (2024) with GRPO algorithm. Key RL hyperparameters include: eight rollouts per sample ($n=8$), maximum output length of 32,768 tokens, maximum 20 assistant turns per trajectory, batch size of 16, mini-batch size of 8, temperature of 0.6, top-$k$ sampling with $k=20$ and top-$p$ with $p=0.95$. We train for 2.6 epochs with a learning rate of $1\times 10^{-6}$. The training is conducted on 8 NVIDIA H20 GPUs using FSDP for distributed training.

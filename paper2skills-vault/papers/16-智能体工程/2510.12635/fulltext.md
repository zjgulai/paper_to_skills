<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2510.12635
     paper_id : 2510.12635
     source   : https://arxiv.org/html/2510.12635v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks

Yuxiang Zhang Affiliation: School of Computer Science and Technology, Beijing Jiaotong University Email: yuxiangzhang@bjtu.edu.cn    Jiangming Shu Affiliation: School of Computer Science and Technology, Beijing Jiaotong University Email: jiangmingshu@bjtu.edu.cn    Ye Ma Affiliation: Hithink Research Email: jtsang@bjtu.edu.cn    Xueyuan Lin Affiliation: Hithink Research Email: maye@myhexin.com    Shangxi Wu & Jitao Sang Affiliation: School of Computer Science and Technology, Beijing Jiaotong University Affiliation: Huawei Noah’s Ark Lab Email: linxy59@mail2.sysu.edu.cn Email: wushangxi1@huawei.com

###### Abstract

Large Language Models face challenges in long-horizon agentic tasks as their constrained memory is easily overwhelmed by distracting or irrelevant context. Existing working memory methods typically rely on external, heuristic mechanisms that are decoupled from the agent’s core policy. In this work, we reframe working memory management as a learnable, intrinsic capability. We propose a novel framework, Memory-as-Action, where an agent actively manages its working memory by executing explicit editing operations as part of a unified policy. This formulation allows an agent, trained via reinforcement learning, to balance memory curation against long-term task objectives under given resource constraints. However, such memory editing actions break the standard assumption of a continuously growing prefix in LLM interactions, leading to what we call trajectory fractures. These non-prefix changes disrupt the causal continuity required by standard policy gradient methods, making those methods inapplicable. To address this, we propose a new algorithm, Dynamic Context Policy Optimization, which enables stable end-to-end reinforcement learning by segmenting trajectories at memory action points and applying trajectory-level advantages to the resulting action segments. Our results demonstrate that jointly optimizing for task reasoning and memory management in an end-to-end fashion not only reduces overall computational consumption but also improves task performance, driven by adaptive context curation strategies tailored to the model’s intrinsic capabilities.

| |

## 1 Introduction

For agentic tasks demanding long-horizon reasoning and complex tool use, such as deep research and software engineering agents (Wei et al., 2025; Jimenez et al., 2024), the effectiveness of a Large Language Model (LLM) is fundamentally constrained by what it can attend to. We define the substrate for this attention as working memory: the structured sequence of historical observations available within a session to drive decision-making. Left unmanaged, however, this working memory can quickly become saturated with irrelevant or outdated information. This accumulation of noise degrades the quality of reasoning and can derail the agent from its primary objective. Therefore, the critical bottleneck for long-horizon tasks shifts from merely expanding memory capacity to actively curating its contents. We term this new meta-task Context Curation—a general mechanism whereby the agent learns to autonomously manage its own working memory by strategically selecting, integrating, and pruning information to maintain a focused and goal-relevant reasoning trace.

Recent advances in long-context methods, enabled by techniques such as positional-encoding scaling, long-context sample synthesis and sparse/linear attention mechanisms, have successfully expanded the capacity of an agent’s working memory (Peng et al., 2023; DeepSeek-AI, 2025). However, simply increasing the context window does not guarantee improved reasoning performance. The effectiveness of long-context model is fundamentally determined by Context Engineering, which refers to the deliberate curation and structuring of information to ensure the most relevant evidence is accessible at the right time (Mei et al., 2025). The dominant approach to context engineering today relies on a workflow of external, rule-based operations (Packer et al., 2023; Jin et al., 2025; Li et al., 2025). Processes such as information retrieval, content summarization, and sliding-window overflow handling are typically managed by separate controllers or heuristics. This design decouples memory management from the agent’s core reasoning policy, making it impossible to learn a coherent strategy that holistically balances task performance against resource costs in an end-to-end manner (Yu et al., 2025; Zhou et al., 2025).

We propose a shift in this paradigm towards Context Curation, which targets an intrinsic and learnable capability of the agent itself. Context Curation is challenging because it requires agents to balance task rewards against diverse resource costs (e.g., tokens, latency, tool calls). We propose Memory-as-Action (MemAct), a framework that treats context curation as a sequence of learnable memory-editing operations. Rather than passively accumulating an ever-growing prefix, the agent learns to decide when to retain, compress, or discard segments of history, and may insert summaries to maintain coherence. These transformations are applied through explicit function-call actions, enabling the agent to develop memory strategies that improve reasoning efficiency (see Fig. 1 for a schematic overview).

The dynamic nature of these decisions makes them difficult to supervise with static labels, motivating an end-to-end reinforcement learning approach. However, the very flexibility that makes MemAct powerful introduces a fundamental challenge: by enabling the agent to edit its history, we break the linear, prefix-accumulating nature of standard LLM trajectories. This disruption of the trajectory’s linear structure is a fundamental challenge for policy optimization. Standard RL methods for LLMs (Ouyang et al., 2022; Shao et al., 2024) rely on the prefix accumulation property, which dictates that each context is an extension of the previous one, to consistently compute policy gradient loss. Once the history becomes editable, this core assumption no longer holds. Memory actions can overwrite or remove earlier content from the working memory, thereby breaking the causal continuity of the trajectory. Standard RL algorithms for LLMs operate on a single, continuous context sequence and are therefore not directly applicable to the trajectories produced by MemAct.

*Figure 1: Comparison between MemAct and conventional memory management. The left side illustrates a representative design in existing systems, where memory operations—such as selection, compression, and summarization—are governed by handcrafted heuristics or external controllers. These behaviors remain decoupled from the agent’s core decision-making process. In contrast, the Memory-as-Action (MemAct) framework integrates such operations into the policy itself, enabling the agent to learn when and how to edit its own working memory as part of a unified decision loop. This formulation supports goal-directed, policy-driven memory management. *

To address the problem of trajectory fracture, we further propose Dynamic Context Policy Optimization (DCPO), a novel RL learning algorithm that enables stable policy gradient estimation by dynamically segmenting the execution trajectory whenever a context curation action occurs. The implementation of DCPO is designed to be compatible with the GRPO (Shao et al., 2024) training pipeline, ensuring ease of integration. In summary, our core contributions are:

-

Framework: We propose the MemAct framework, which redefines context curation as intrinsic, learnable actions. This design enables an agent to use a unified policy to balance context management and task execution, achieving end-to-end optimization toward long-term goals.

-

Algorithm: We propose the DCPO algorithm to address the challenge that memory editing disrupts the linear accumulation of context, rendering standard policy gradient methods inapplicable.

-

Evaluation: We demonstrate that MemAct achieves performance competitive with much larger models while operating at a substantially lower token cost, learning memory strategies that are both generalizable to new tasks and adaptive to the base model’s capabilities.

## 2 Method

### 2.1 Overview

We introduce the Memory-as-Action framework, which enables an agent to actively edit its own working memory (see Algorithm 2 for the full execution loop). Importantly, because memory operations are implemented as standard function calls, this design supports recursive memory management—the agent can manage not only task-relevant content, but also records of prior memory actions, enabling meta-level reflection and refinement.

This flexible editability introduces a challenge: memory actions can overwrite or remove past context, breaking the common prefix assumption and resulting in trajectory fractures. To address this, we propose a policy optimization algorithm, DCPO, which enables stable learning by partitioning histories into causally consistent segments and computing trajectory-level advantages(See Algorithm 1).

### 2.2 Formalizing Context Management as a Decision Process

We model the agent’s interaction as a Markov Decision Process (MDP), allowing the policy to explicitly edit its working memory while pursuing the primary task. The MDP is defined as follows:

-

State: The state $s_{t}\in\mathcal{S}$ at timestep $t$ is the working memory $H_{t}$, a structured sequence of historical observations such as user instructions, tool calls, and their outputs.

-

Action Space: $\mathcal{A}=\mathcal{A}_{\text{task}}\cup\mathcal{A}_{\text{mem}}$, where $\mathcal{A}_{\text{task}}$ contains task-oriented actions that interact with the external environment, and $\mathcal{A}_{\text{mem}}$ contains memory actions that directly modify the working memory.

-

Transition: Task actions $a_{t}\in\mathcal{A}_{\text{task}}$ yield observations $o_{t}$, appended to history: $H_{t+1}=H_{t}\oplus(a_{t},o_{t})$. Memory actions $a_{t}\in\mathcal{A}_{\text{mem}}$ transform history: $H_{t+1}=a_{t}(H_{t})$, potentially overwriting prior content and breaking the append-only assumption.

-

Objective: Learn a policy $\pi_{\theta}$ that maximizes the expected cumulative reward $J(\theta)$. A sparse, terminal reward combines final task success with adherence to resource constraints.

### 2.3 Dynamic Context Policy Optimization

*Algorithm 1 DCPO Training Loop*

Input : Initial policy $\pi_{\theta}$, prompt dataset $\mathcal{D}$, environment $\mathcal{E}$, trajectories per prompt $N_{\text{traj}}$, segments to sample per prompt $N_{\text{seg}}$

Output : Optimized policy $\pi_{\theta}$

1 while not converged do

    2 Sample a batch of prompts $\mathcal{U}_{\text{batch}}\sim\mathcal{D}$

    3 $\mathcal{B}\leftarrow\emptyset$ // Initialize an empty batch for training

    4 $A_{\text{map}}\leftarrow\{\}$ // Initialize a map from trajectory to its advantage

    5 foreach $u\in\mathcal{U}_{\text{batch}}$ do

      // 1. Rollout

       6 $\mathcal{T}_{u}\leftarrow\emptyset$ // Collect trajectories for the current prompt $u$

       7 for $i=1$ to $N_{\text{traj}}$ do

          8 Generate trajectory $\tau$ from prompt $u$ and get its return $R(\tau)$

          9 Append $(\tau,R(\tau))$ to $\mathcal{T}_{u}$

      // 2. Advantage Estimation

       10 $\text{group\_advantages}\leftarrow\text{ComputeAdvantages}(\mathcal{T}_{u})$

       11 $A_{\text{map}}.\text{update}(\text{group\_advantages})$

      // 3. Segmentation and Per-Prompt Sampling

       12 $\Sigma_{u\text{\_pool}}\leftarrow\emptyset$ // Create a local segment pool for prompt $u$

       13 foreach $(\tau,\dots)$ in $\mathcal{T}_{u}$ do

          14 Extract memory-action indices $\{t^{\mathrm{mem}}_{k}\}$ from $\tau$; set $t^{\mathrm{mem}}_{0}\!=\!0$, $t^{\mathrm{mem}}_{K+1}\!=\!T$

          15 for $i=0$ to $K$ do

             16 $H_{\text{prefix}}\leftarrow H_{t^{\mathrm{mem}}_{i}}$

             17 $Y_{\text{gen}}\leftarrow(y_{t})_{t=t^{\mathrm{mem}}_{i}+1}^{t^{\mathrm{mem}}_{i+1}}$

             18 $\text{input\_ids}\leftarrow\mathrm{tokenize}(H_{\text{prefix}})\oplus Y_{\text{gen}}$

             19 $m^{\sigma_{i}}\leftarrow[0,\dots,0,1,\dots,1]$ where $|0|=|\mathrm{tokenize}(H_{\text{prefix}})|$

             20 Append $(\text{input\_ids},m^{\sigma_{i}},\tau.\text{id})$ to $\Sigma_{u\text{\_pool}}$

       21 $\mathcal{B}_{u}\leftarrow\text{Sample}(N_{\text{seg}},\Sigma_{u\text{\_pool}})$

       22 $\mathcal{B}\leftarrow\mathcal{B}\cup\mathcal{B}_{u}$ // Aggregate segments into the training batch

   // 4. Policy Update

    23 $\mathcal{L}\leftarrow\text{ComputePolicyLoss}(\mathcal{B},A_{\text{map}},\pi_{\theta})$

    24 Update policy $\pi_{\theta}$

25 return Optimized policy $\pi_{\theta}$

#### 2.3.1 Outcome Reward

We employ a sparse, terminal reward function $R(\tau)$ assigned based on the final outcome of a trajectory $\tau$. The reward is defined as follows:

$R(\tau)=\begin{cases}r_{\text{task}}&\text{if the task is successfully completed,}\\
r_{\text{pen}}&\text{if a resource constraint is violated,}\\
0&\text{otherwise.}\end{cases}$ | | | |

Here, $r_{\text{task}}>0$ is a fixed positive reward for success, and $r_{\text{pen}}<0$ is a fixed penalty. A penalty is applied if the agent violates predefined operational constraints, such as exceeding the maximum context length. All other terminal states, such as failing the task without any constraint violations, result in a zero reward. This sparse signal is designed to encourage the policy to learn complex behaviors that lead to successful outcomes while respecting resource limits.

#### 2.3.2 Trajectory Segmentation and Sampling

As the agent executes memory actions, the standard assumption of a continuously growing, append-only context is broken. This creates what we term a trajectory fracture—a point where the working memory $H_{t+1}$ is no longer a simple extension of $H_{t}$. Such non-prefix histories pose a significant challenge for policy optimization: using mismatched context for gradient computation can lead to incorrect updates and instability.

To address this, we introduce trajectory segmentation. Let $t^{\text{mem}}_{1},\ldots,t^{\text{mem}}_{K}$ be the timesteps where memory actions occur, augmented such that $t^{\text{mem}}_{0}=0$ (i.e., the beginning of the trajectory) and $t^{\text{mem}}_{K+1}=T$ (i.e., the final timestep). We define a segment $\sigma_{i}$ as the subsequence between two consecutive memory actions at $t^{\text{mem}}_{i}$ and $t^{\text{mem}}_{i+1}$, sharing a common prefix $H_{t^{\text{mem}}_{i}}$ and generating a new sequence over $(t^{\text{mem}}_{i},\ t^{\text{mem}}_{i+1}]$. This ensures that the gradient for each token is computed using the exact context under which it was generated.

During the training process, for each prompt, we generate $N_{\text{traj}}$ trajectories and sample $N_{\text{seg}}$ segments from them for training. The ratio $N_{\text{seg}}/N_{\text{traj}}$ is a tunable hyperparameter, whose optimal value depends on the task complexity and the frequency of memory operations. We sample segments using a trajectory-based round-robin strategy, drawing one unique segment per trajectory per epoch, and repeating the process until $N_{\text{seg}}$ segments are collected.

#### 2.3.3 Trajectory-Level Policy Optimization

To compute the policy gradient, we use the trajectory-level group-normalized advantage estimation method from GRPO (Shao et al., 2024). Each full trajectory $\tau$ is assigned a normalized advantage:

$A(\tau)=\frac{R(\tau)-\mu_{u}}{\sigma_{u}},$ | | | |

where $\mu_{u}$ and $\sigma_{u}$ are the mean and standard deviation of returns over all trajectories generated from the same prompt $u$.

The loss is then defined as:

$\mathcal{L}(\theta)=-\,\mathbb{E}_{u\sim\mathcal{D}}\!\left[\frac{1}{|\mathcal{G}(u)|}\sum_{\tau\in\mathcal{G}(u)}\ \sum_{\sigma_{i}\in\Sigma(\tau)}\ \sum_{t\in\sigma_{i}}\ {m}^{\sigma_{i}}_{t}\cdot A(\tau)\cdot\log\pi_{\theta}(y_{t}\mid H_{t})\right],$ | | | |

where $\Sigma(\tau)$ is the set of sampled segments from trajectory $\tau$, and ${m}^{\sigma_{i}}_{t}$ is a binary mask indicating newly generated tokens within each segment.

## 3 Experiments & Results

### 3.1 Datasets

To ensure a comprehensive evaluation of the proposed methods, this study utilizes both synthetic data and public benchmarks across different experimental stages, with each dataset tailored to specific objectives in training, validation, and testing.

We created the Multi-objective QA dataset from HotpotQA to evaluate an agent’s long-range reasoning and memory management. In each task, the agent must answer several independent questions to provide a single, consolidated answer. The training data is constrained to simpler tasks with two to four objectives, while evaluation is performed on more complex test sets with up to eight objectives to measure generalization. We also conducted experiments on a collection of Multi-hop QA datasets preprocessed by Asearcher (Gao et al., 2025). This benchmark suite includes 2WikiMultihopQA (Ho et al., 2020), Bamboogle (Press et al., 2022), HotpotQA (Yang et al., 2018), Musique Trivedi et al. (2022), and Frames (Krishna et al., 2024). These datasets provide a structured testbed to assess memory management under varying reasoning depths and context complexities.

For supervised fine-tuning, we observed that even advanced models such as OpenAI o3, DeepSeek-V3.1, and Qwen3-235B were unable to reliably learn memory editing behavior through prompting alone. A key failure point lay in their inability to correctly interpret updated working memory states. To address this, we prompted DeepSeek-V3.1 to emulate MemAct-style behavior, thereby generating a high-quality trajectory dataset for policy initialization. A staged prompting strategy was adopted: memory operations were softly suggested when context lengths ranged between 8K and 16K tokens, and strictly enforced beyond 16K. The resulting dataset includes 700 multi-hop and 100 multi-objective QA instances. From more than 800 successful trajectories, over 3,000 segments were extracted for fine-tuning.

For the reinforcement learning phase, we trained the agent on a dataset combining 8,000 multi-hop QA examples from Asearcher with 8,000 synthesized multi-objective tasks from HotpotQA. The training distribution was deliberately biased toward lower-complexity tasks (fewer than four objectives) to increase generalization pressure and encourage the emergence of robust memory management strategies.

### 3.2 Baselines

To assess the effectiveness of MemAct, we compare it against the following three categories of representative baselines:

-

Long-context language models without explicit memory mechanisms. We include Qwen3-235B-A22B-instruct-2507 and Qwen3-30B-A3B-instruct-2507, which represent the state of the art in long-context reasoning. These models leverage large context windows and internal knowledge to perform multi-hop inference without any memory editing capability.

-

Conventional memory management strategies based on external mechanisms. Two representative approaches are implemented:

-

Sliding Window: discards the earliest segments of the interaction history once the context exceeds 8K tokens.

-

Summarization: performs proactive compression when the context length approaches 8K tokens, condensing the oldest half of the history into a summary.

-

RL-base agent. We include Search-R1, which is retrained on the same supervised fine-tuning (SFT) dataset used for MemAct-SFT to ensure comparability. The retrained agent achieves stronger performance than previously reported and serves as a competitive reference point.

#### 3.2.1 Implementation Details

##### Memory Management Implementation

In our implementation, memory management is operationalized via a dedicated prune_context tool. To enable targeted pruning, every tool call’s output is assigned a unique, randomly generated ID that serves as a handle to its record in the history. When the agent determines that the context needs to be condensed, it invokes the prune_context tool with two arguments: a model-generated summary synthesizing the key information to be retained, and a list of ids_to_prune corresponding to the historical records to be deleted. The framework’s executor then processes this call by removing the specified records from the active context, while preserving the summary. This effectively replaces detailed, pruned records with a concise and useful piece of memory.

##### Cold-start via Segmented Supervised Finetuning

For the cold-start initialization of the policy model, we employ a Segmented Supervised Finetuning (SFT) phase to adapt it to the MemAct action space, which includes both task-specific tools and our prune_context memory action. This phase adopts both the segmentation strategy and the loss masking mechanism from DCPO to guide the model’s learning process. We perform training for 6 epochs with a batch size of 256. The learning rate is set to $5\times 10^{-5}$ and follows a cosine decay schedule with a warmup ratio of 0.1.

##### DCPO Hyperparameters

For the primary DCPO training phase, we use a batch size of 128. As described in Alg. 1, we generate $N_{\text{traj}}=8$ trajectories for each prompt and sample $N_{\text{seg}}=16$ segments for each policy update step. The policy is optimized with the AdamW optimizer using a learning rate of $1\times 10^{-6}$. Trajectories are terminated if they exceed a maximum of 35 tool-use turns (including both task and memory actions). All experiments were conducted on NVIDIA H100 GPUs.

##### Reward

The reward signal is determined by a gpt-oss (OpenAI, 2025) based evaluator, which assesses whether the agent’s final answer is consistent with the ground truth. Following the reward function defined in Section 2.3.1, we set the parameters to $r_{\text{task}}=+1.0$ and $r_{\text{pen}}=-0.1$. The penalty $r_{\text{pen}}$ is applied for execution failures, such as exceeding a 20,000 token context limit or producing an unparsable final answer.

### 3.3 Metrics

We evaluate our method on both task success and resource efficiency, assessed by an evaluator based on gpt-oss OpenAI (2025) that judges consistency with the ground truth.

Success Metrics. We define two metrics: 1) Accuracy: The percentage of queries where all sub-objectives are answered correctly. 2) Per-Objective Accuracy: The average proportion of correctly addressed sub-objectives.

Efficiency Metrics. We track two primary costs, excluding internal memory actions: 1) Function Calls: The number of external tool interactions per query. 2) Input Token Cost: The cumulative input tokens fed to the policy model for a query, which is reported as Total Tokens per Query (the total sum) and Tokens per Round (the total sum divided by the number of LLM calls).

### 3.4 Main Result

### Analysis of Multi-Objective QA Results

*Figure 2: Model performance on the Multi-Objective QA dataset. (a) Per-objective accuracy versus average input tokens per round. The top-left quadrant represents higher accuracy with greater context efficiency. (b) Total input tokens per query versus the average number of function calls, illustrating the end-to-end efficiency of different strategies. *

*Figure 3: Comparative performance on the Multi-Objective QA dataset. The left panel shows task accuracy and the right panel shows the average number of task-related tool calls, broken down by the number of objectives per query. Models are sorted by their overall average accuracy. Memory actions are excluded from the tool call count. *

The evaluation on the Multi-Objective QA dataset highlights the effectiveness of the MemAct framework in improving both accuracy and robustness. The detailed results are presented in Figure 2, which provides a macro-level overview of accuracy-efficiency trade-offs, and Figure 3, which offers a granular performance breakdown as task complexity increases.

Our MemAct-14B-RL model achieves a leading average accuracy of 59.1%, outperforming all baselines, including the much larger Qwen3-235B model. As shown in Figure 3, this performance is consistent across tasks of varying difficulty. Notably, its accuracy exhibits a graceful degradation as the number of objectives increases, demonstrating generalization to task complexities unseen during training, as the training set was limited to tasks with a maximum of four objectives. This underscores the model’s robustness. Furthermore, this level of performance is achieved with an average context of only 3,447 input tokens per round (Figure 2a), in sharp contrast to the Search-R1-14B agent, which requires a substantially larger context (8,625 tokens) for lower accuracy.

The detailed breakdown of tool usage in Figure 3 provides insight into different problem-solving strategies. Models with extensive internal knowledge, like Qwen3-235B, consistently use fewer function calls than agent-based models. The heatmap also exposes the failure mode of weaker models like Qwen2.5-7B. Its minimal tool usage correlates with its poor performance, which notably falls below even a baseline that answers without using any tools, indicating a failure to effectively engage with the task’s procedural requirements. In contrast, agents built upon smaller language models, such as our MemAct variants, engage in more frequent tool interactions to solve the tasks.

*Figure 4: Tool usage by objective count. (a) Average external-tool calls. (b) Average memory-management calls. Rows: models; columns: objectives. Independent color scales per panel. *

While the 7B and 14B models exhibit similar behavioral patterns after supervised fine-tuning (SFT), their strategies diverge significantly after RL optimization with the same reward signal, a trend suggested by Figure 3. The detailed breakdown of tool usage in Figure 4 further illuminates this divergence, revealing how the learned policies develop different emphases on external tool use versus internal memory management.

-

For the more capable 14B model, RL leads to an efficiency-oriented strategy. As shown in Figure 4a, the MemAct-14B-RL model consistently uses fewer external tools than its SFT counterpart across all levels of difficulty.

-

In contrast, for the smaller 7B model, RL promotes a strategy of extending the reasoning process. The model makes more external tool calls than its SFT version at all difficulty levels. Concurrently, its usage of memory-management tools increases to a level comparable to the 14B models (Figure 4). This suggests a compensatory policy: the model attempts to overcome its limited internal knowledge by gathering more external information for each sub-objective, which in turn necessitates more intensive memory management.

Crucially, as shown in Figure 2(b), both of these learned strategies are highly token-efficient. The 14B model becomes more direct without a significant token penalty, while the 7B model extends its reasoning at a far lower total token cost than the Search-R1-7B baseline.

These findings indicate that the MemAct framework does not enforce a single, rigid strategy. Instead, it provides the necessary mechanism for reinforcement learning to discover adaptive policies tailored to a model’s intrinsic capabilities. By supporting these varied strategies at a low computational cost, our approach eases the trade-off between reasoning depth and efficiency, enabling smaller models to effectively tackle more complex, long-horizon tasks.

#### Robustness Across Multi-hop QA Benchmarks

To assess robustness across tasks of varying reasoning complexity, we evaluated our models on five multi-hop QA benchmarks, with results presented in Table 1. The performance of both MemAct and Search-R1 is consistently higher than that of the base instruction-tuned model and standard sliding window methods, suggesting that structured agentic frameworks are beneficial for these tasks.

Our final model, MemAct-14B-RL, achieves an average score of 0.567, nearly on par with the Search-R1 baseline’s score of 0.572. This level of accuracy is achieved with greater token efficiency, as established in our analysis of Figure 2(b), indicating a favorable trade-off between task performance and computational cost for our framework.

Within our method, we also observe a consistent improvement from the SFT-trained model (0.555 avg.) to the final RL-tuned version (0.567 avg.). This gain is most pronounced on the Musique and Frames datasets. A possible explanation is that these benchmarks require more complex or longer reasoning chains. The reinforcement learning phase may be particularly effective at refining the agent’s policy for such long-horizon tasks, optimizing the sequence of tool and memory actions beyond what can be learned from static demonstrations in SFT.

*Table 1: Comparison of different context-management methods on top of the Qwen2.5-14B-Instruct backbone across five multi-hop QA benchmarks. Our MemAct variants are highly competitive with the Search-R1 (Cold-Start) baseline and substantially outperform standard sliding-window techniques.*

| Method | 2Wiki | Bamboogle | HotpotQA | Musique | Frames | Avg. |

| Base | 0.580 | 0.488 | 0.655 | 0.233 | 0.275 | 0.446 |

| + Sliding Window | 0.535 | 0.472 | 0.560 | 0.271 | 0.215 | 0.411 |

| + Sliding Window w/ Summary | 0.540 | 0.442 | 0.692 | 0.268 | 0.335 | 0.455 |

| + Search-R1 w/ Cold-Start | 0.775 | 0.624 | 0.723 | 0.364 | 0.376 | 0.572 |

| + MemAct (SFT) | 0.764 | 0.616 | 0.705 | 0.330 | 0.359 | 0.555 |

| + MemAct (RL) | 0.767 | 0.618 | 0.710 | 0.353 | 0.385 | 0.567 |

#### Training Efficiency

The reduction in token consumption enabled by memory pruning directly translates to improved training efficiency. For our 7B model, employing MemAct within the DCPO framework reduced the duration of the rollout phase by approximately 40% and the policy update phase by 25%. This comparison is against a no-memory baseline using identical hyperparameters, highlighting a practical benefit of our approach..

## 4 Conclusion

This work introduces Memory-as-Action, a framework that treats working memory management as an integral part of an agent’s decision-making process, rather than as an external module. By formalizing memory operations as explicit actions, a single policy can learn to interleave task reasoning with context curation. This approach creates a challenge—the violation of the prefix assumption in standard policy optimization—which we address with Dynamic Context Policy Optimization (DCPO). DCPO enables stable policy gradient training on non-monotonic histories by segmenting trajectories and ensuring correct gradient attribution. Our experiments on multi-objective and multi-hop question answering demonstrate two primary benefits. First, the method improves task success by maintaining a focused and relevant context. Second, it reduces computational costs by pruning unhelpful history and adapting tool-use strategies to the base model’s capabilities. The training pipeline is compatible with existing reinforcement learning practices and improves training efficiency, suggesting that learned memory policies can be a viable alternative to brute-force context expansion. In conclusion, the results indicate that end-to-end optimization of reasoning and memory yields agents that are both more effective and more efficient, learning adaptive strategies suited to their underlying capabilities.

## 5 Future Work and Limitations

The results presented here represent our initial findings on the efficacy of the MemAct framework. We are actively conducting further analyses, focusing on the emergent trade-offs between reasoning depth and computational cost, and how learned context management strategies adapt to a model’s intrinsic capabilities.

Future work will likely explore more complex memory operations and test the framework’s robustness on a broader range of long-horizon tasks. We will continue to update this paper with additional results and analyses as they become available.

## References

- DeepSeek-AI (2025) DeepSeek-AI. Deepseek-v3.2-exp: Boosting long-context efficiency with deepseek sparse attention, 2025.

- Gao et al. (2025) Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous rl, 2025. URL https://arxiv.org/abs/2508.07976.

- Ho et al. (2020) Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.

- Jimenez et al. (2024) Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. Swe-bench: Can language models resolve real-world github issues? In ICLR, 2024.

- Jin et al. (2025) Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yongkang Wu, Zhonghua Li, Qi Ye, and Zhicheng Dou. Hierarchical document refinement for long-context retrieval-augmented generation. In ACL Long, 2025.

- Krishna et al. (2024) Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation. arXiv preprint arXiv:2409.12941, 2024.

- Li et al. (2025) Z. Li et al. Memos: A memory os for ai system, 2025.

- Mei et al. (2025) Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi Li, Duzhen Zhang, et al. A survey of context engineering for large language models. arXiv preprint arXiv:2507.13334, 2025.

- OpenAI (2025) OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925.

- Ouyang et al. (2022) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- Packer et al. (2023) Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Memgpt: Towards llms as operating systems, 2023.

- Peng et al. (2023) Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

- Press et al. (2022) Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350, 2022.

- Shao et al. (2024) Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

- Trivedi et al. (2022) Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.

- Wei et al. (2025) Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

- Yang et al. (2018) Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.

- Yu et al. (2025) Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.

- Zhou et al. (2025) Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents, 2025. URL https://arxiv.org/abs/2506.15841.

## Appendix A Appendix

### A.1 Implementation Details of MemAct

#### Pseudocode

*Algorithm 2 MemAct Agent Execution Loop*

Input: Initial prompt $x$, environment $\mathcal{E}$, model policy $\pi_{\theta}$

// $\mathcal{M}$: memory tool module that applies $a_{t}$ on $H_{t-1}$ and outputs $(H_{t},o_{t})$

Output: Final task output $o_{\text{final}}$, trajectory $\tau$

1 Initialize working memory $H_{0}\leftarrow[x]$

2 Initialize empty trajectory buffer $\tau\leftarrow\emptyset$

3 for $t=1$ to $T_{\max}$ do

    4 Sample action $a_{t}\sim\pi_{\theta}(\cdot\mid H_{t-1})$

    5 if $a_{t}\in\mathcal{A}_{\text{task}}$ then

       6 Execute $a_{t}$ in environment: $o_{t}\leftarrow\mathcal{E}(a_{t})$

       7 Update memory: $H_{t}\leftarrow H_{t-1}\oplus(a_{t},o_{t})$

    8 else

      // Execute memory tool, returning both new memory and execution outcome

       9 $(H_{t},o_{t})\leftarrow\mathcal{M}(a_{t},H_{t-1})$

      // $o_{t}$ contains status indicators (e.g., success flag)

       10 Append $(a_{t},o_{t})$ to memory: $H_{t}\leftarrow H_{t}\oplus(a_{t},o_{t})$

    11 Append $(a_{t},H_{t})$ to trajectory buffer $\tau$

    12 if termination condition met then

       13 break

14 Return final output $o_{\text{final}}$, full trajectory $\tau$

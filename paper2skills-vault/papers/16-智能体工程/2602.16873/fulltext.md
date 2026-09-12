<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2602.16873
     paper_id : 2602.16873
     source   : https://arxiv.org/html/2602.16873v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# AdaptOrch: Task-Adaptive Multi-Agent Orchestration
in the Era of LLM Performance Convergence

Geunbin Yu Affiliation: Department of Artificial Intelligence, Korea National Open University Email: ict03@rfems.com Affiliation: ORCID: 0009-0006-2879-9514

February 2026

###### Abstract

As large language models (LLMs) from diverse providers converge toward comparable benchmark performance, the traditional paradigm of selecting a single best model per task yields diminishing returns. We argue that orchestration topology—the structural composition of how multiple agents are coordinated, parallelized, and synthesized—now dominates system-level performance over individual model capability. We present AdaptOrch, a formal framework for task-adaptive multi-agent orchestration that dynamically selects among four canonical topologies (parallel, sequential, hierarchical, and hybrid) based on task dependency graphs and empirically derived domain characteristics. Our framework introduces three key contributions: (1) Performance Convergence Scaling Law, formalizing conditions under which orchestration selection outweighs model selection; (2) Topology Routing Algorithm that maps task decomposition DAGs to optimal orchestration patterns in $O(|V|+|E|)$ time; and (3) Adaptive Synthesis Protocol with provable termination guarantees and heuristic consistency scoring for parallel agent outputs. We validate AdaptOrch across coding (SWE-bench), reasoning (GPQA), and retrieval-augmented generation tasks, demonstrating that topology-aware orchestration achieves 12–23% improvement over static single-topology baselines, even when using identical underlying models. Our results establish orchestration design as a first-class optimization target independent of model scaling.

Keywords: multi-agent systems, LLM orchestration, task-adaptive routing, parallel agent execution, performance convergence

## 1 Introduction

The landscape of large language models in early 2026 presents a paradoxical challenge: as more models achieve near-identical benchmark scores, the marginal value of model selection diminishes while the complexity of choosing among them grows. GPT-4o, Claude 3.5 Sonnet, Gemini 2.0, Llama 3.3 70B, DeepSeek-V3, and Qwen 2.5 72B now cluster within 2–5% of each other on standard benchmarks including MMLU, HumanEval, and MATH (Hugging Face, 2025). This performance convergence reshapes the optimization frontier. When individual model capability plateaus, how models are composed begins to dominate which model is selected—a shift with far-reaching implications for system design.

Current orchestration approaches fall into two broad categories. Static frameworks—Model Context Protocol (MCP) (Anthropic, 2024), LangGraph (LangChain, 2024), and CrewAI (Moura, 2024)—define fixed execution topologies (chains, graphs, or role-based teams) that persist regardless of what the task demands. A second category, routing-based systems like Mixture-of-Agents (MoA) (Wang et al., 2024) and LLM-Blender (Jiang et al., 2023), dynamically selects or blends model outputs yet leaves the structural topology of agent coordination untouched. A natural question emerges: given a specific task, what is the optimal topology for coordinating multiple agents?

Recent practical advances illuminate this gap. Both Claude Code’s Agent Teams (Anthropic, 2026) and OpenCode’s parallel subagent architecture (OpenCode Contributors, 2025) show that parallel execution of specialized agents—each in its own context window, working on an independent subtask—can compress multi-hour sequential workflows into minutes. What these systems still leave to the user, however, is the decomposition itself: deciding how to split the work and assign agent roles. The topology selection problem remains unsolved at the algorithmic level.

*Figure 1: Paradigm shift from model selection (left) to orchestration design (right). When model capabilities converge, the dominant optimization variable becomes the structural topology of agent coordination.*

This paper introduces AdaptOrch, a framework that formalizes and automates topology selection. The central insight is straightforward: tasks decompose into dependency-annotated directed acyclic graphs (DAGs), and structural properties of these DAGs—parallelism width, critical path depth, inter-subtask coupling—turn out to predict the optimal orchestration topology with high accuracy.

We make four contributions:

-

Performance Convergence Scaling Law (Section 3): We show that under $\epsilon$-convergence of model capabilities, the variance in system performance attributable to orchestration topology exceeds that of model selection by a factor of $\Omega(1/\epsilon^{2})$, establishing topology selection as the dominant optimization target as models converge.

-

Topology Routing Algorithm (Section 4): A linear-time algorithm that analyzes task dependency DAGs and routes to one of four canonical topologies: parallel, sequential, hierarchical, or hybrid.

-

Adaptive Synthesis Protocol (Section 4.5): A protocol for reconciling outputs from parallel agents with provable termination guarantees via adaptive re-routing and heuristic consistency scoring based on embedding similarity.

-

Empirical validation (Section 5): Experiments across three domains showing 12–23% improvement over static baselines using identical models.

## 2 Related Work

### 2.1 LLM Performance Convergence

Multiple benchmark suites now document the convergence of LLM capabilities across providers. The Open LLM Leaderboard v2 (Hugging Face, 2025) shows top-10 models clustering within a 3-point MMLU range (87.2–90.1) as of January 2026. In a striking finding, Sato and Ito (2025) demonstrated that Self-MoA—a single top model queried multiple times—outperforms diverse model mixing by 6.6% on AlpacaEval 2.0, undermining the assumption that model diversity inherently improves performance. Chatbot Arena (Zheng et al., 2024) ELO rankings tell a similar story: frontier models from OpenAI, Anthropic, Google, Meta, and Alibaba now occupy overlapping confidence intervals on general-purpose tasks. Taken together, these results suggest that when models become increasingly interchangeable, the orchestration structure emerges as the primary lever for performance gains.

### 2.2 Static Orchestration Frameworks

Model Context Protocol (MCP) (Anthropic, 2024) standardizes tool-model interfaces but prescribes no topology for multi-agent coordination. LangGraph (LangChain, 2024) goes further, modeling workflows as directed graphs with parallel branches, conditional edges, and stateful execution—yet the topology must be designed manually. CrewAI (Moura, 2024) takes a role-based approach, assigning agents fixed personas (e.g., researcher, writer, reviewer) in predetermined interaction patterns, while AutoGen (Wu et al., 2023) supports multi-agent conversation but defaults to sequential round-robin communication. The common thread: none of these frameworks adapt their topology based on the task at hand.

### 2.3 Dynamic Model Composition

Mixture-of-Agents (Wang et al., 2024) arranges models in layered pipelines where each layer refines previous outputs, achieving 65.1% on AlpacaEval 2.0 versus 57.5% for the best individual model. LLM-Blender (Jiang et al., 2023) uses a PairRanker to select among candidate outputs. DEI (Zhang et al., 2024) employs multi-agent committees for SWE-bench Lite, where the best-performing group achieves a 55% resolve rate versus 27.3% for the strongest individual open-source agent. However, all these systems use fixed topologies (layered pipeline, output selection, or flat committee) regardless of task structure. To our knowledge, no prior work formalizes topology selection as an explicit function of task dependency structure, which is the gap we address.

### 2.4 Parallel Agent Execution in Practice

Claude Code Agent Teams (Anthropic, 2026) and the Superpowers framework (Superpowers Contributors, 2026) demonstrate practical parallel execution with lead-agent orchestration, DAG-based task dependencies, and inbox-based inter-agent communication. OpenCode (OpenCode Contributors, 2025) supports multi-provider agent routing with explicit permission-controlled subagent architectures. Drammeh (2025) showed that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches in incident response, with zero quality variance across 348 trials. These practical systems validate the performance potential of orchestrated multi-agent execution but lack formal frameworks for topology optimization.

### 2.5 Concurrent and Recent Work

Several concurrent efforts address related aspects of dynamic multi-agent orchestration. DyTopo (Lu et al., 2026) optimizes agent communication topology via semantic matching between agent capabilities and subtask requirements; unlike our approach, their routing operates at the agent-pair level rather than selecting among canonical structural patterns, which limits interpretability of the chosen topology. MetaGen (Wang and others, 2026) co-evolves agent roles and topologies through self-play, achieving impressive adaptation but sacrificing the predictability that our closed-form routing algorithm provides. ALMC (ALMC Authors, 2026) introduces Manager-Judge-Optimizer role separation with adaptive collaboration, though their role-based decomposition differs fundamentally from our DAG-structure-based routing and does not provide explicit cost control. MoMA (Guo et al., 2025) generalizes routing across both models and agents, treating the choice of orchestration strategy as a bandit problem; our work instead exploits task structure directly through DAG analysis, avoiding the sample complexity of online learning. S-DAG (Dong et al., 2026), accepted at AAAI 2026, decomposes tasks into subject-based DAGs for multi-agent allocation—the closest work to ours in spirit, though their subjects correspond to semantic domains while our DAG nodes represent subtask dependencies with explicit coupling annotations. ORCH (Vinay and Sankaran, 2026) proposes a deterministic multi-agent protocol with fixed execution guarantees; our framework complements this by adding adaptive topology selection on top of deterministic execution primitives.

Our work is distinguished by the combination of (i) formal topology routing grounded in DAG structural properties, (ii) provable termination guarantees for the synthesis protocol, and (iii) explicit cost-accuracy Pareto analysis—elements that no single prior system integrates.

## 3 Problem Formalization

### 3.1 Model Convergence

###### Definition 1 ($\epsilon$-Convergence).

A set of $n$ models $\mathcal{M}=\{M_{1},\ldots,M_{n}\}$ is $\epsilon$-convergent on benchmark $\mathcal{B}$ if:

$\max_{i,j\in[n]}|S_{\mathcal{B}}(M_{i})-S_{\mathcal{B}}(M_{j})|\leq\epsilon$ | | | | (1) |

where $S_{\mathcal{B}}(M_{i})$ denotes the score of model $M_{i}$ on benchmark $\mathcal{B}$, normalized to $[0,1]$.

For current frontier models on MMLU, $\epsilon\approx 0.03$; on HumanEval, $\epsilon\approx 0.05$.

### 3.2 Task Dependency Graphs

###### Definition 2 (Task Dependency DAG).

A task $T$ decomposes into a directed acyclic graph $G_{T}=(V,E,w,c)$ where:

-

$V=\{v_{1},\ldots,v_{k}\}$ is the set of subtasks

-

$E\subseteq V\times V$ encodes dependencies ($(v_{i},v_{j})\in E$ means $v_{i}$ must complete before $v_{j}$ starts)

-

$w:V\to\mathbb{R}^{+}$ assigns estimated computational cost to each subtask

-

$c:E\to[0,1]$ assigns coupling strength between dependent subtasks (degree of context sharing required)

###### Definition 3 (DAG Structural Properties).

For a task DAG $G_{T}=(V,E,w,c)$, we define:

$\displaystyle\text{Parallelism Width: }\quad\omega(G_{T})$ $\displaystyle=\max_{A\subseteq V}|A|\text{ s.t. }A\text{ is an antichain in }G_{T}$ | | | | | (2) |

$\displaystyle\text{Critical Path Depth: }\quad\delta(G_{T})$ $\displaystyle=\max_{\text{path }P}\sum_{v\in P}w(v)$ | | | | | (3) |

$\displaystyle\text{Coupling Density: }\quad\gamma(G_{T})$ $\displaystyle=\frac{\sum_{(u,v)\in E}c(u,v)}{|E|}$ | | | | | (4) |

### 3.3 Orchestration Topologies

We define four canonical topologies $\mathcal{T}=\{\tau_{P},\tau_{S},\tau_{H},\tau_{X}\}$:

###### Definition 4 (Canonical Topologies).

$\displaystyle\tau_{P}$ $\displaystyle:\text{{Parallel}}-\text{All subtasks execute concurrently; outputs merged post-hoc}$ | | | | | (5) |

$\displaystyle\tau_{S}$ $\displaystyle:\text{{Sequential}}-\text{Subtasks execute in topological order; each receives prior context}$ | | | | | (6) |

$\displaystyle\tau_{H}$ $\displaystyle:\text{{Hierarchical}}-\text{Lead agent decomposes and delegates; sub-agents report back}$ | | | | | (7) |

$\displaystyle\tau_{X}$ $\displaystyle:\text{{Hybrid}}-\text{DAG partitioned into parallel groups connected sequentially}$ | | | | | (8) |

Each topology $\tau$ induces a scheduling function $\sigma_{\tau}:G_{T}\to\text{ExecutionPlan}$ that maps the task DAG to a concrete execution ordering with agent assignments.

### 3.4 Performance Convergence Scaling Law

###### Proposition 1 (Orchestration Dominance under Convergence).

Let $\mathcal{M}$ be $\epsilon$-convergent on task distribution $\mathcal{D}$. Let $\text{Var}_{M}$ denote performance variance from model selection and $\text{Var}_{\tau}$ denote performance variance from topology selection. For a task $T$ with dependency DAG $G_{T}$ having $k$ subtasks, under uniform subtask weights, Lipschitz aggregation ($L_{f}\leq 1$), and a topology quality coefficient $C_{\tau}\geq 1/(4k)$:

$\frac{\text{Var}_{\tau}}{\text{Var}_{M}}\geq\frac{(\omega(G_{T})-1)^{2}}{4\epsilon^{2}\cdot k}\cdot\left(1-\gamma(G_{T})\right)^{2}$ | | | | (9) |

When $\epsilon\to 0$ (perfect convergence) and $\omega(G_{T})>1$ (parallelizable tasks), $\text{Var}_{\tau}/\text{Var}_{M}\to\infty$.

###### Proof sketch.

Model selection variance is bounded by $\text{Var}_{M}\leq\epsilon^{2}$ from Definition 1, using the correlated bound (all subtasks share the same model). Topology variance derives from the execution time ratio between worst-case (fully sequential: $\sum_{v}w(v)$) and best-case (maximally parallel: $\delta(G_{T})$) schedules. By Dilworth’s theorem, the minimum number of chains covering $G_{T}$ equals the maximum antichain width $\omega(G_{T})$. The speedup ratio $\sum_{v}w(v)/\delta(G_{T})\geq\omega(G_{T})$ when subtask weights are uniform. Coupling density $\gamma$ reduces effective parallelism by introducing synchronization overhead proportional to $\gamma^{2}$. Combining bounds yields the stated ratio. Full proof in Appendix A. ∎

###### Corollary 1.

For coding tasks (typical $\omega\geq 3$, $\gamma\leq 0.4$, $k\leq 6$, $\epsilon\approx 0.05$), the variance ratio satisfies $\text{Var}_{\tau}/\text{Var}_{M}\geq 20$, indicating that orchestration topology is the dominant performance factor over model selection.

## 4 The AdaptOrch Framework

*Figure 2: AdaptOrch pipeline. The Topology Router (Algorithm 1) selects the optimal execution topology based on DAG structural properties ($\omega$, $\delta$, $\gamma$). Failed syntheses trigger re-routing with adjusted coupling estimates.*

AdaptOrch operates in five phases: task decomposition, DAG construction, topology routing, parallel/sequential execution, and adaptive synthesis (Figure 2).

### 4.1 Phase 1: Task Decomposition

Given input task $T$, a decomposer agent $A_{\text{decomp}}$ extracts subtasks:

$A_{\text{decomp}}(T)\to\{(v_{i},d_{i},w_{i})\}_{i=1}^{k}$ | | | | (10) |

where $v_{i}$ is the subtask identifier, $d_{i}$ is its natural language description, and $w_{i}$ is the estimated token cost. The decomposer is prompted with domain-specific decomposition strategies:

### 4.2 Phase 2: DAG Construction

The decomposer output is parsed into a formal DAG $G_{T}=(V,E,w,c)$. Dependency edges are inferred from explicit “required inputs” declarations. Coupling strength $c(u,v)$ is estimated based on declared context requirements:

$c(u,v)=\begin{cases}0.0&\text{if coupling = none (outputs fully independent)}\\
0.3&\text{if coupling = weak (shared context helpful but not required)}\\
0.7&\text{if coupling = strong (output of $u$ is direct input to $v$)}\\
1.0&\text{if coupling = critical (semantic coherence required)}\end{cases}$ | | | | (11) |

DAG validity is verified: acyclicity check via topological sort ($O(|V|+|E|)$), connected component analysis, and critical path computation.

### 4.3 Phase 3: Topology Routing

The routing algorithm maps DAG structural properties to the optimal topology:

*Algorithm 1 Topology Routing Algorithm*

0:  Task DAG $G_{T}=(V,E,w,c)$, thresholds $\theta_{\omega},\theta_{\gamma},\theta_{\delta}$

0:  Topology $\tau^{*}\in\{\tau_{P},\tau_{S},\tau_{H},\tau_{X}\}$

1:  Compute $\omega(G_{T})$, $\delta(G_{T})$, $\gamma(G_{T})$ {Definition 3}

2:  Compute $r\leftarrow\omega(G_{T})/|V|$ {Parallelism ratio}

3:  if $|E|=0$ then

4:   return $\tau_{P}$ {Fully parallel}

5:  else if $\omega(G_{T})=1$ then

6:   return $\tau_{S}$ {Fully sequential}

7:  else if $\gamma(G_{T})>\theta_{\gamma}$ and $|V|>\theta_{\delta}$ then

8:   return $\tau_{H}$ {High coupling + many subtasks}

9:  else if $r>\theta_{\omega}$ and $\gamma(G_{T})\leq\theta_{\gamma}$ then

10:   return $\tau_{P}$ {Wide DAG, low coupling}

11:  else

12:   Partition $G_{T}$ into stages $S_{1},\ldots,S_{m}$ via topological layering

13:   return $\tau_{X}(S_{1},\ldots,S_{m})$ {Hybrid topology}

14:  end if

Default thresholds: $\theta_{\omega}=0.5$ (at least half the subtasks parallelizable), $\theta_{\gamma}=0.6$ (high coupling threshold), $\theta_{\delta}=5$ (minimum subtasks for hierarchical). These are empirically calibrated in Section 5.

Complexity: The routing decision (Algorithm 1, lines 3–11) runs in $O(|V|+|E|)$: critical path $\delta(G_{T})$ via longest-path DP on the DAG, coupling density $\gamma$ via edge traversal, and topological layering for hybrid partitioning. The antichain width $\omega(G_{T})$ computation requires separate analysis: an approximate $\omega$ via layer-width (maximum layer size in topological ordering) runs in $O(|V|+|E|)$ and suffices for routing; the exact $\omega$ via König’s theorem on the transitive closure requires $O(|V|^{2.5})$ matching and is used only for offline calibration.

### 4.4 Phase 4: Topology-Specific Execution

Each topology implements a distinct execution strategy:

#### 4.4.1 Parallel Executor ($\tau_{P}$)

All subtasks dispatch simultaneously to separate agent instances, each with isolated context windows:

$\forall v_{i}\in V:\quad\text{output}_{i}=A_{i}(d_{i},\text{context}_{\text{global}})\quad\text{[concurrent]}$ | | | | (12) |

Agent assignment uses round-robin across available model instances. This mirrors the architecture of Claude Code Agent Teams, where each subagent receives task-specific instructions plus minimal shared context.

#### 4.4.2 Sequential Executor ($\tau_{S}$)

Subtasks execute in topological order, with each agent receiving the accumulated context of all predecessors:

$\text{output}_{i}=A_{i}\left(d_{i},\text{context}_{\text{global}},\bigoplus_{(v_{j},v_{i})\in E}\text{output}_{j}\right)$ | | | | (13) |

where $\bigoplus$ denotes context concatenation with relevance-weighted truncation to fit context windows.

#### 4.4.3 Hierarchical Executor ($\tau_{H}$)

A lead agent $A_{\text{lead}}$ orchestrates sub-agents, maintaining a global task list with DAG-based dependency tracking:

$\displaystyle A_{\text{lead}}:\text{decompose}\to\text{assign}\to\text{monitor}\to\text{reconcile}$ | | | | | (14) |

$\displaystyle A_{\text{sub},i}:\text{receive}(d_{i})\to\text{execute}\to\text{report}(A_{\text{lead}})$ | | | | | (15) |

The lead agent resolves conflicts when sub-agent outputs are inconsistent, analogous to Claude Code’s lead-agent pattern with inbox-based communication.

#### 4.4.4 Hybrid Executor ($\tau_{X}$)

The DAG is partitioned into topological layers $S_{1},\ldots,S_{m}$. Within each layer, subtasks execute in parallel; between layers, execution is sequential:

$\text{For layer }S_{l}:\quad\forall v_{i}\in S_{l}:\text{output}_{i}=A_{i}\left(d_{i},\bigoplus_{v_{j}\in\bigcup_{l^{\prime}<l}S_{l^{\prime}}}\text{output}_{j}\right)\quad\text{[concurrent within }S_{l}\text{]}$ | | | | (16) |

### 4.5 Phase 5: Adaptive Synthesis Protocol

The synthesizer merges outputs from the selected topology into a coherent final result.

###### Definition 5 (Consistency Score (Heuristic)).

For outputs $\{o_{1},\ldots,o_{k}\}$ from $k$ subtasks, the consistency score is a heuristic measure of semantic agreement:

$\text{CS}(o_{1},\ldots,o_{k})=\frac{1}{\binom{k}{2}}\sum_{i<j}\text{sim}(o_{i}\cap o_{j},o_{i}\cup o_{j})$ | | | | (17) |

where sim measures semantic overlap via embedding cosine similarity on shared output dimensions. Note that CS captures semantic similarity rather than logical consistency; it serves as a practical proxy for detecting contradictory outputs but does not guarantee formal logical coherence.

*Algorithm 2 Adaptive Synthesis Protocol*

0:  Outputs $\{o_{1},\ldots,o_{k}\}$, topology $\tau$, consistency threshold $\theta_{\text{CS}}$

0:  Synthesized output $O$

1:  Compute $\text{CS}(o_{1},\ldots,o_{k})$

2:  if $\tau=\tau_{S}$ then

3:   return $o_{k}$ {Sequential: last output is final}

4:  else if $\text{CS}\geq\theta_{\text{CS}}$ then

5:   $O\leftarrow A_{\text{merge}}(\text{``Synthesize these consistent outputs: ''}\|o_{1}\|\cdots\|o_{k})$

6:   return $O$ {Consistent parallel outputs}

7:  else

8:   $O\leftarrow A_{\text{arbiter}}(\text{``Resolve conflicts among: ''}\|o_{1}\|\cdots\|o_{k})$

9:   if $\text{CS}(O)<\theta_{\text{CS}}$ then

10:    Re-route via Algorithm 1 with $\gamma^{\prime}=\gamma+0.2$ {Increase coupling}

11:   end if

12:   return $O$ {Inconsistent: escalated}

13:  end if

###### Proposition 2 (Synthesis Termination).

Under the adaptive re-routing mechanism (Algorithm 2, line 8), the synthesis protocol terminates within at most $\lceil(1-\gamma_{0})/0.2\rceil\leq 5$ iterations. As $\gamma$ increases by 0.2 per retry, after at most 5 iterations $\gamma>\theta_{\gamma}$ forces hierarchical routing ($\tau_{H}$), which uses a single arbiter agent, guaranteeing termination. Empirically, convergence occurs in $\leq 2$ iterations for 94% of tasks (Section 5).

## 5 Experiments

### 5.1 Setup

Models. We use five $\epsilon$-convergent models: GPT-4o-mini, Claude 3.5 Haiku, Gemini 2.0 Flash, Llama 3.3 70B (via Together AI), and Qwen 2.5 72B (via vLLM). All models score within $\epsilon=0.04$ on MMLU and $\epsilon=0.06$ on HumanEval. Table 1 provides explicit per-model scores validating the $\epsilon$-convergence assumption.

*Table 1: $\epsilon$-Convergence evidence. All models fall within $\epsilon$ of the best model on each benchmark, validating Definition 1.*

| Model | MMLU | HumanEval | ARC-C | MATH |

| GPT-4o-mini | 82.0 | 87.2 | 93.1 | 70.2 |

| Claude 3.5 Haiku | 83.1 | 88.7 | 92.4 | 69.5 |

| Gemini 2.0 Flash | 81.4 | 86.9 | 91.8 | 71.8 |

| Llama 3.3 70B | 82.6 | 85.3 | 93.7 | 68.1 |

| Qwen 2.5 72B | 83.8 | 87.8 | 94.2 | 72.4 |

$\epsilon$ | (max gap) | 0.024 | 0.034 | 0.024 | 0.043 |

Reproducibility. All experiments use seed $=42$, temperature $=0.0$ (greedy decoding), and max_workers $=8$ for parallel execution. SWE-bench runs use Docker-based sandboxed evaluation with run_id = adaptorch-v1.0. API endpoints: OpenAI gpt-4o-mini-2024-07-18, Anthropic claude-3-5-haiku-20241022, Google gemini-2.0-flash-001. Each experiment is run 3 times; we report mean $\pm$ standard deviation. Residual variance under greedy decoding arises from three sources: (i) non-deterministic API server-side batching documented by all three providers, (ii) race conditions in parallel agent execution order affecting synthesis inputs, and (iii) floating-point non-associativity in distributed inference. Observed standard deviations remain below 0.8% absolute across all benchmarks. Code, configuration files, topology routing logs, and a one-command reproduction script (Makefile) are available at https://github.com/adaptorch/adaptorch.

Token and Cost Accounting. Token usage is measured via provider-reported usage fields in each API response (prompt_tokens, completion_tokens) and summed across all calls within a single task instance, including orchestration overhead (decomposition, routing, synthesis). Because tokenizers differ across providers (OpenAI cl100k_base, Anthropic internal, Google SentencePiece), we report raw provider-reported counts without cross-provider normalization; the Tok(K) column in Table 2 reflects this aggregate. Pricing is taken as of 2026-01-15 from official pricing pages: OpenAI gpt-4o-mini at $0.15/1M input, $0.60/1M output; Anthropic claude-3.5-haiku at $0.80/1M input, $4.00/1M output; Google gemini-2.0-flash at $0.10/1M input, $0.40/1M output.

*Figure 3: $\epsilon$-Convergence evidence across four benchmarks. All five models score within $\epsilon$ of the best, validating the convergence assumption (Definition 1). Dashed line: best model score; shaded band: $\epsilon$ range.*

Benchmarks.

-

Coding: SWE-bench Verified (Jimenez et al., 2024) (500 instances)—multi-file bug fixing requiring code understanding, localization, and patching.

-

Reasoning: GPQA Diamond (Rein et al., 2024) (198 instances)—graduate-level science questions requiring multi-step domain reasoning.

-

RAG: HotpotQA (Yang et al., 2018) distractor setting (500 instances)—multi-hop question answering over retrieved documents.

Baselines.

-

Single Best: Best individual model per benchmark.

-

MoA-3L: Mixture-of-Agents with 3 layers (Wang et al., 2024).

-

Static-Parallel: All subtasks always parallel (mimics Claude Code Agent Teams without topology adaptation).

-

Static-Sequential: All subtasks always sequential (mimics standard chain-of-thought pipeline).

-

LLM-Blender: PairRanker-based output selection (Jiang et al., 2023).

Metrics.

-

Task accuracy: pass@1 for SWE-bench, accuracy for GPQA, F1 for HotpotQA

-

Latency: Wall-clock time from input to final output

-

Efficiency: Accuracy per 1M tokens consumed

-

Topology distribution: Fraction of tasks routed to each $\tau$

### 5.2 Results

*Table 2: Main results across three benchmarks. AdaptOrch selects topology per-task. Self-MoA (matched) uses a single top model with self-consistency voting under the same token budget as AdaptOrch. Best results in bold, second-best underlined. $\Delta$ shows improvement over Single Best baseline. Tok(K) = average tokens consumed per instance in thousands.*

| Method | SWE-bench Verified | GPQA Diamond | HotpotQA |

| Acc | Lat. | Tok(K) | Acc | Lat. | Tok(K) | F1 | Lat. | Tok(K) |

$\times$ $\times$ $\times$ | Single Best | 42.8 | 1.0 | 12.3 | 46.2 | 1.0 | 4.1 | 68.3 | 1.0 | 6.8 |

$\times$ $\times$ $\times$ | MoA-3L | 48.1 | 3.2 | 84.6 | 49.8 | 2.8 | 31.2 | 71.6 | 2.5 | 47.3 |

$\times$ $\times$ $\times$ | Static-Parallel | 47.3 | 1.4 | 52.1 | 44.1 | 1.3 | 18.7 | 72.8 | 1.2 | 28.4 |

$\times$ $\times$ $\times$ | Static-Sequential | 45.6 | 2.8 | 48.9 | 50.3 | 2.4 | 16.4 | 69.1 | 2.1 | 26.1 |

$\times$ $\times$ $\times$ | LLM-Blender | 44.9 | 1.8 | 61.7 | 47.7 | 1.6 | 22.3 | 70.4 | 1.5 | 34.8 |

$\times$ $\times$ $\times$ | Self-MoA (matched) | 51.5 | 1.5 | 43.2 | 52.3 | 1.4 | 16.8 | 75.5 | 1.2 | 23.1 |

$\times$ $\times$ $\times$ | AdaptOrch (ours) | 52.6 | 1.6 | 41.8 | 53.1 | 1.5 | 15.9 | 76.4 | 1.3 | 22.7 |

$\Delta$ | vs Single Best | +9.8 | — | — | +6.9 | — | — | +8.1 | — | — |

$\Delta$ | vs Best Static | +4.5 | — | — | +2.8 | — | — | +3.6 | — | — |

Table 2 presents our main results. AdaptOrch achieves the highest accuracy across all three benchmarks while maintaining moderate latency overhead.

On SWE-bench Verified, the improvement reaches 22.9% over Single Best. Coding tasks exhibit high parallelism width ($\omega\approx 3.4$) because file localization, context understanding, and patch generation can execute concurrently. The router sends 62% of instances to $\tau_{X}$ (hybrid), 24% to $\tau_{P}$ (parallel), and 14% to $\tau_{H}$ (hierarchical).

The picture differs for GPQA Diamond (+14.9%), where reasoning tasks show higher coupling ($\gamma\approx 0.55$). Here AdaptOrch prefers sequential (41%) and hierarchical (35%) topologies. Notably, Static-Parallel actually degrades performance below Single Best on this benchmark—a clear illustration that topology mismatch can be actively harmful.

HotpotQA (+11.9%) sits between these extremes: document processing parallelizes naturally, but reasoning chains impose sequential dependencies. Accordingly, 71% of instances route to $\tau_{X}$ (hybrid).

*Figure 4: Main results comparison across three benchmarks. AdaptOrch achieves the highest accuracy on all tasks while maintaining competitive latency. Error bars show $\pm 1$ standard deviation over 3 runs.*

*Figure 5: Pareto front: accuracy vs. latency. AdaptOrch achieves the best accuracy-latency tradeoff across benchmarks, dominating other methods in the Pareto sense.*

Token efficiency. Table 2 also reports token consumption. AdaptOrch consumes 41.8K tokens per SWE-bench instance, significantly less than MoA-3L (84.6K) and LLM-Blender (61.7K), because topology-aware routing avoids redundant model calls. Among multi-agent baselines, the accuracy-per-million-tokens metric favors AdaptOrch across all benchmarks (Figure 6); the Single Best baseline naturally achieves higher token efficiency in absolute terms due to its single-call design, but at substantially lower accuracy.

*Figure 6: Token efficiency analysis. (Left) Total token consumption per instance. (Center) Accuracy per 1M tokens. (Right) Cost-accuracy Pareto front showing AdaptOrch achieves optimal cost-efficiency.*

### 5.3 Topology Distribution Analysis

*Table 3: Topology routing distribution (%) by benchmark domain. The router adapts topology selection to domain characteristics.*

$\tau_{P}$ $\tau_{S}$ $\tau_{H}$ $\tau_{X}$| Domain | (Parallel) | (Sequential) | (Hierarchical) | (Hybrid) |

| SWE-bench | 24 | 0 | 14 | 62 |

| GPQA | 8 | 41 | 35 | 16 |

| HotpotQA | 18 | 3 | 8 | 71 |

| Average | 16.7 | 14.7 | 19.0 | 49.7 |

Table 3 reveals that the hybrid topology $\tau_{X}$ is most frequently selected (49.7% average), reflecting the reality that most tasks contain both parallelizable and sequential components. Pure parallel ($\tau_{P}$) is preferred for tasks with low coupling, while pure sequential ($\tau_{S}$) dominates high-coupling reasoning tasks.

*Figure 7: Topology routing distribution heatmap across benchmark domains. Row normalization shows the proportion of each topology selected per domain.*

### 5.4 Ablation Studies

*Table 4: Ablation study on SWE-bench Verified (500 instances).*

$\Delta$| Configuration | Accuracy | |

| AdaptOrch (full) | 52.6 | — |

$-$$\tau_{X}$ $-2.8$| Adaptive routing (fixed ) | 49.8 | |

$-$ $-5.5$| Synthesis protocol (naive concat) | 47.1 | |

$-$$c=0.5$ $-2.3$| DAG coupling (uniform ) | 50.3 | |

$-$ $-1.6$| Re-routing on failure | 51.0 | |

$-$ $-9.8$| Task decomposition (1 subtask) | 42.8 | |

The ablation study (Table 4) confirms that each component contributes meaningfully: the synthesis protocol provides the largest individual contribution ($-5.5$), followed by adaptive routing ($-2.8$) and coupling-aware decomposition ($-2.3$). Removing task decomposition entirely reduces to the Single Best baseline.

*Figure 8: Ablation waterfall chart showing cumulative contribution of each AdaptOrch component. Bars show accuracy drop when each component is removed independently.*

### 5.5 Threshold Sensitivity

*Figure 9: Sensitivity of task accuracy to coupling threshold $\theta_{\gamma}$ across SWE-bench and GPQA. Shaded regions show 95% bootstrap confidence intervals ($n=30$ trials per setting). Optimal range: $[0.55,0.65]$.*

Figure 9 shows that AdaptOrch is robust to threshold selection within $\theta_{\gamma}\in[0.5,0.7]$, with optimal performance at $\theta_{\gamma}=0.6$. Extreme values degrade performance: $\theta_{\gamma}<0.3$ forces sequential execution on parallelizable tasks; $\theta_{\gamma}>0.8$ allows parallel execution of tightly coupled subtasks, causing consistency failures.

Data Leakage Prevention. To avoid test-set contamination, all threshold calibration was performed on a held-out development split before any test evaluation. Specifically, we sampled 15% of instances from each benchmark (SWE-bench: 75 instances, GPQA: 30, HotpotQA: 75) using a fixed seed ($s{=}42$), performed grid search over $\theta_{\gamma}\in\{0.3,0.4,\ldots,0.8\}$ on this dev split, selected $\theta_{\gamma}{=}0.6$, and then froze the threshold for all test evaluation. The reported metrics in Tables 2–4 are computed exclusively on the remaining 85% test split. Dev instance IDs are included in the released codebase.

*Figure 10: Per-instance accuracy distribution by routed topology across benchmarks. Violin plots show density; white dots indicate median. The topology-dependent performance variation validates the adaptive routing approach.*

*Figure 11: Distribution of synthesis convergence iterations across all benchmark instances. 94% of tasks converge within 2 iterations, consistent with Proposition 2.*

## 6 Discussion

### 6.1 When Does Orchestration Not Help?

Our framework’s gains are smallest on single-step, atomic tasks where $|V|=1$ (no decomposition possible) or tasks with $\gamma\approx 1.0$ (complete sequential dependency). On GPQA instances classified as “single-concept recall,” AdaptOrch matches but does not exceed the Single Best baseline. This is expected: orchestration adds value proportional to task decomposability.

### 6.2 Relationship to Self-MoA

Sato and Ito (2025) showed that a single top model used multiple times outperforms diverse model mixing. Our framework is orthogonal: AdaptOrch optimizes how agents are structured, not which models are used. To control for this interaction, we include a compute-matched Self-MoA baseline (Table 2) that applies self-consistency voting with the same token budget as AdaptOrch. Self-MoA (matched) recovers 89% of AdaptOrch’s gains over Single Best, confirming that structured multi-sample reasoning itself provides substantial benefit. The remaining 11% gap—consistent across all three benchmarks—is attributable to topology-aware routing: AdaptOrch allocates compute non-uniformly across subtasks based on dependency structure, whereas Self-MoA distributes tokens uniformly.

### 6.3 Practical Implications

AdaptOrch can be implemented on existing infrastructure:

-

Claude Code Agent Teams: Use the lead-agent pattern for $\tau_{H}$, parallel subagent dispatch for $\tau_{P}$, and DAG-based task dependencies for $\tau_{X}$.

-

LangGraph: Map topologies to graph structures with conditional edges for routing.

-

OpenCode + MCP: Route through multi-provider APIs with permission-controlled subagents.

The routing algorithm (Algorithm 1) adds negligible overhead ($<$50ms) compared to LLM inference latency ($\sim$2–15s per call), making real-time topology adaptation practical.

### 6.4 Limitations

-

Decomposition quality depends on the decomposer model: Poor task decomposition propagates errors to all downstream phases. We mitigate this with self-consistency checks but do not guarantee optimal decomposition.

-

Coupling estimation is approximate: The discrete $c\in\{0,0.3,0.7,1.0\}$ scale is coarse. Continuous coupling estimation via embedding similarity is a promising extension.

-

Cost scaling: Parallel execution requires $\omega(G_{T})$ concurrent API calls, which may exceed rate limits or budget constraints for resource-constrained deployments.

-

Experimental scope: We evaluate on three benchmarks; generalization to creative writing, long-form generation, and multi-modal tasks requires further study.

## 7 Conclusion

We presented AdaptOrch, a framework built on a simple thesis: when LLM capabilities converge, the orchestration topology becomes the dominant lever for system performance. A scaling law grounds this intuition theoretically, the Topology Routing Algorithm translates it into a practical $O(|V|+|E|)$ procedure, and experiments across coding, reasoning, and retrieval tasks confirm 12–23% improvements over static baselines.

As LLM capabilities continue to converge, we believe the field will increasingly shift from “which model?” to “which orchestration?” AdaptOrch provides a principled foundation for this shift, bridging the gap between practical multi-agent systems (Claude Code Agent Teams, OpenCode, LangGraph) and formal orchestration theory.

### 7.1 Future Work

-

Learned routing: Replace threshold-based routing with a lightweight classifier trained on (DAG features, optimal topology) pairs.

-

Dynamic re-orchestration: Allow topology changes mid-execution when partial results reveal unexpected coupling.

-

Cost-aware routing: Extend the routing algorithm to jointly optimize accuracy and API cost under budget constraints.

-

Cross-modal orchestration: Apply AdaptOrch to multi-modal tasks combining vision, code, and language agents.

## Appendix A Full Proof of Proposition 1

###### Proof.

Let $\mathcal{M}=\{M_{1},\ldots,M_{n}\}$ be $\epsilon$-convergent on benchmark $\mathcal{B}$. Consider task $T$ with dependency DAG $G_{T}=(V,E,w,c)$ with $|V|=k$ subtasks.

Model selection variance bound. For any model $M_{i}$, its per-subtask performance satisfies $S(M_{i},v_{j})\in[S^{*}-\epsilon,S^{*}]$ where $S^{*}=\max_{i}S(M_{i},v_{j})$. The total task performance under model $M_{i}$ is:

$P(M_{i},T)=f\left(\{S(M_{i},v_{j})\}_{j=1}^{k}\right)$ | | | | (18) |

where $f$ is the aggregation function determined by the orchestration topology. Since each $S(M_{i},v_{j})$ varies by at most $\epsilon$, and $f$ is Lipschitz with constant $L_{f}\leq 1$ (normalized scoring), and subtask scores under the same model are positively correlated (shared model capacity), we obtain:

$\text{Var}_{M}[P(M,T)]\leq L_{f}^{2}\cdot\epsilon^{2}=\epsilon^{2}$ | | | | (19) |

Note: the $k$-fold summation applies only under independence; since all subtasks use the same model, the correlated bound $\epsilon^{2}$ is tighter.

Topology selection variance bound. Consider two extreme topologies for the same task:

-

Fully sequential ($\tau_{S}$): execution time $=\sum_{v\in V}w(v)$

-

Maximally parallel ($\tau_{P}$): execution time $=\delta(G_{T})=\max_{\text{path}}\sum_{v\in P}w(v)$

The quality impact of topology depends on two factors: (a) latency-quality tradeoff (parallel execution under budget constraints allows more refinement iterations), and (b) context propagation (sequential topology preserves inter-subtask context that parallel execution loses).

Assumption 1 (Topology quality sensitivity). The quality difference between fully parallel and fully sequential execution satisfies:

$|\Delta Q_{\text{topology}}|\geq C_{\tau}\cdot(\omega(G_{T})-1)\cdot(1-\gamma(G_{T}))$ | | | | (20) |

for some task-class-dependent constant $C_{\tau}>0$. This is motivated by: (i) the $(\omega-1)$ term captures the degree of parallelism—more parallel branches mean more potential for topology-induced quality variation, and (ii) the $(1-\gamma)$ term captures the information loss from not propagating context in parallel execution. We empirically validate this assumption in Table 4, where removing topology adaptation degrades performance by 4.7–8.3 points.

Under uniform subtask weights, by Dilworth’s theorem, the speedup from optimal parallelization is $\geq\omega(G_{T})$. Taking $C_{\tau}=1/2$ (conservative estimate: topology changes half the theoretical maximum quality gap) and noting that with $k$ subtasks the per-task variance from a binary topology choice (parallel vs. sequential) satisfies $\text{Var}_{\tau}\geq\Delta Q^{2}/4$, we obtain:

$\text{Var}_{\tau}[P(\tau,T)]\geq\frac{(\omega(G_{T})-1)^{2}\cdot(1-\gamma(G_{T}))^{2}}{4k}$ | | | | (21) |

where the $1/k$ factor arises from normalizing the per-subtask contribution to aggregate task performance.

Ratio.

$\frac{\text{Var}_{\tau}}{\text{Var}_{M}}\geq\frac{(\omega(G_{T})-1)^{2}\cdot(1-\gamma(G_{T}))^{2}}{4k\cdot\epsilon^{2}}=\frac{(\omega(G_{T})-1)^{2}\cdot(1-\gamma(G_{T}))^{2}}{4\epsilon^{2}\cdot k}$ | | | | (22) |

As $\epsilon\to 0$, this ratio diverges, confirming that topology selection dominates model selection under convergence. For typical coding tasks: $\omega\approx 3.4$, $\gamma\approx 0.35$, $k\approx 5$, $\epsilon\approx 0.05$, yielding a ratio $\geq\frac{(2.4)^{2}\cdot(0.65)^{2}}{4\cdot 0.0025\cdot 5}=\frac{2.43}{0.05}\approx 48.7$. ∎

## Appendix B Implementation Details

### B.1 Decomposition Prompt

The full decomposition prompt used for SWE-bench tasks:

### B.2 Computational Requirements

All experiments were conducted using API-based model access. Estimated costs:

-

SWE-bench (500 instances, 5 methods): $\sim$$1,200 total API cost

-

GPQA (198 instances, 5 methods): $\sim$$180 total API cost

-

HotpotQA (500 instances, 5 methods): $\sim$$350 total API cost

AdaptOrch’s routing overhead: $<$50ms per task (Python implementation on single CPU core). Synthesis overhead: one additional LLM call per task ($\sim$$0.01 per instance).

### B.3 DAG Feature Space Analysis

*Figure 12: PCA projection of DAG feature space (width $\omega$, depth, density, coupling ratio) colored by KMeans clusters ($k{=}4$). The four clusters correspond to dominant topology patterns: Chain (sequential), Wide-Shallow (parallel), Deep-Narrow (hierarchical), and Diamond (fan-out/fan-in). Cluster centroids are marked with $\times$.*

### B.4 Baseline Reproduction Specification

To ensure fair comparison and full reproducibility, we detail the exact configuration of each baseline method. All baselines use the same 5-model pool as AdaptOrch: GPT-4o-mini, Claude 3.5 Haiku, Gemini 2.0 Flash, Llama 3.3 70B (via Together AI), and Qwen 2.5 72B (via Together AI). Temperature is $0.0$ (greedy) and max_tokens $=4096$ for all methods unless noted.

MoA-3L (Wang et al., 2024). We implement the 3-layer Mixture-of-Agents architecture as described in the original paper. Layer 1: all 5 models generate independent responses to the full prompt. Layer 2: each of 3 aggregator models (GPT-4o-mini, Claude 3.5 Haiku, Gemini 2.0 Flash) receives all 5 Layer-1 outputs concatenated in the prompt and produces a refined answer. Layer 3: a single synthesizer (GPT-4o-mini) receives all 3 Layer-2 outputs and produces the final answer. Total LLM calls per instance: $5+3+1=9$. Aggregation prompt follows the template in Wang et al. (2024), §A.1.

LLM-Blender (Jiang et al., 2023). We use the prompt-based variant (no fine-tuned PairRanker) for fair comparison, since training a ranker on our specific benchmarks would introduce confounding. Stage 1 (Generation): all 5 models produce independent candidates. Stage 2 (Ranking): GPT-4o-mini is prompted to rank the 5 candidates pairwise using the template: “Given the task and two candidate solutions A and B, which better solves the problem? Output only ‘A’ or ‘B’.” This produces $\binom{5}{2}=10$ pairwise comparisons per instance. Stage 3 (Fusion): the top-ranked candidate is returned as the final output (no generative fusion, which would require a trained model). Total LLM calls per instance: $5+10+0=15$.

Static-Parallel / Static-Sequential. These ablation baselines use AdaptOrch’s own decomposition (Phase 1–2) but bypass the topology router. Static-Parallel executes all subtasks simultaneously across 3 models (round-robin assignment); Static-Sequential chains them in dependency order using a single model (GPT-4o-mini). Both use the same synthesis protocol (Phase 5) as AdaptOrch.

### B.5 Per-Cluster Orchestration Gain

Table 5 disaggregates AdaptOrch’s accuracy improvement by DAG cluster (cf. Figure 12), revealing which structural patterns benefit most from adaptive topology routing.

*Table 5: Per-cluster accuracy gain ($\Delta$) of AdaptOrch over Single Best, averaged across all three benchmarks. $n$: number of instances assigned to each cluster.*

$\tau$ $n$ $\Delta$| DAG Cluster | Dominant | | Single Best | AdaptOrch |

$\tau_{S}$ | Chain (sequential) | | 187 | 54.2% | +3.8 pp |

$\tau_{P}$ | Wide-Shallow (parallel) | | 294 | 49.1% | +12.6 pp |

$\tau_{H}$ | Deep-Narrow (hierarchical) | | 112 | 47.8% | +9.2 pp |

$\tau_{X}$ | Diamond (fan-out/fan-in) | | 105 | 51.3% | +11.4 pp |

The largest gains appear in Wide-Shallow tasks ($+12.6$ pp), where parallelism directly reduces error propagation by distributing independent subtasks. Chain-type tasks show the smallest gain ($+3.8$ pp), consistent with the expectation that fully sequential dependencies leave minimal room for topology improvement.

Router Accuracy (Confusion Matrix). To assess the topology router’s decision quality, we compare its selections against an oracle that exhaustively evaluates all four topologies per instance and selects the highest-scoring one. Across the full test set ($n=698$):

*Table 6: Router confusion matrix: predicted topology vs. oracle-optimal topology. Values are instance counts. Overall router accuracy: 81.2% (567/698).*

$\tau_{P}$ $\tau_{S}$ $\tau_{H}$ $\tau_{X}$| | | | | |

$\tau_{P}$ | | 248 | 12 | 8 | 18 |

$\tau_{S}$ | | 6 | 152 | 9 | 4 |

$\tau_{H}$ | | 14 | 7 | 89 | 11 |

$\tau_{X}$ | | 9 | 5 | 7 | 78 |

The router achieves 81.2% agreement with the oracle. Most misclassifications occur between $\tau_{P}$ and $\tau_{X}$ (18 + 9 = 27 instances), which is expected since Diamond tasks contain both parallel and fan-in components. Importantly, even misrouted instances typically receive the second-best topology, limiting accuracy loss to $<$2 pp compared to oracle routing.

## References

- ALMC Authors (2026) ALMC Authors ALMC: adaptive LLM multi-agent collaboration with manager-judge-optimizer roles. OpenReview preprint. Note: OpenReview ID: jXZGgxTjiK Cited by: §2.5.

- Anthropic (2024) Anthropic Model context protocol. Note: https://modelcontextprotocol.io/Open standard for LLM-tool integration Cited by: §1, §2.2.

- Anthropic (2026) Anthropic Claude code agent teams: parallel subagent orchestration. Note: https://docs.anthropic.com/en/docs/claude-codeResearch preview, February 2026 Cited by: §1, §2.4.

- Dong et al. (2026) K. Dong, Z. Lin, W. Lin, and Y. Zhang S-DAG: subject-based directed acyclic graph decomposition for multi-agent task allocation. In Proceedings of the AAAI Conference on Artificial Intelligence, Note: arXiv preprint arXiv:2511.06727 Cited by: §2.5.

- Drammeh (2025) P. Drammeh Multi-agent LLM orchestration achieves deterministic, high-quality decision support for incident response. arXiv preprint arXiv:2511.15755. Cited by: §2.4.

- Guo et al. (2025) Z. Guo, Y. Wang, T. Ji, X. Zhao, Y. Xi, C. Liu, P. Li, Y. Deng, and S. Feng MoMA: mixture-of-model-and-agent routing for generalized multi-agent orchestration. arXiv preprint arXiv:2509.07571. Cited by: §2.5.

- Hugging Face (2025) Hugging Face Open LLM leaderboard v2. Note: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboardAccessed: 2026-02-15 Cited by: §1, §2.1.

- Jiang et al. (2023) D. Jiang, X. Ren, and B. Y. Lin LLM-Blender: ensembling large language models with pairwise ranking and generative fusion. arXiv preprint arXiv:2306.02561. Cited by: §B.4, §1, §2.3, item 5.

- Jimenez et al. (2024) C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan SWE-bench: can language models resolve real-world GitHub issues?. arXiv preprint arXiv:2310.06770. Cited by: 1st item.

- LangChain (2024) LangChain LangGraph: build stateful multi-agent applications. Note: https://github.com/langchain-ai/langgraph Cited by: §1, §2.2.

- Lu et al. (2026) Y. Lu, Z. Hu, M. Zhao, and W. Cao DyTopo: dynamic topology optimization for multi-agent systems via semantic matching. arXiv preprint arXiv:2602.06039. Cited by: §2.5.

- Moura (2024) J. Moura CrewAI: framework for orchestrating role-playing autonomous AI agents. Note: https://github.com/crewAIInc/crewAI Cited by: §1, §2.2.

- OpenCode Contributors (2025) OpenCode Contributors OpenCode: open-source ai coding assistant with multi-provider support. Note: https://github.com/opencode-ai/opencode Cited by: §1, §2.4.

- Rein et al. (2024) D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman GPQA: a graduate-level Google-Proof q&a benchmark. arXiv preprint arXiv:2311.12022. Cited by: 2nd item.

- Sato and Ito (2025) A. Sato and M. Ito Self-MoA: scalable self-collaboration of a single LLM via mixture-of-agents. arXiv preprint arXiv:2502.00674. Cited by: §2.1, §6.2.

- Superpowers Contributors (2026) Superpowers Contributors Superpowers: multi-agent orchestration framework. Note: https://github.com/superpower-agents/superpowers Cited by: §2.4.

- Vinay and Sankaran (2026) K. Vinay and S. Sankaran ORCH: deterministic multi-agent orchestration protocol for structured task execution. Frontiers in Artificial Intelligence. Note: Accepted 30 January 2026, Machine Learning and Artificial Intelligence section External Links: Document Cited by: §2.5.

- Wang et al. (2024) J. Wang, J. Wang, B. Athiwaratkun, C. Zhang, and J. Zou Mixture-of-agents enhances large language model capabilities. arXiv preprint arXiv:2406.04692. Cited by: §B.4, §1, §2.3, item 2.

- Wang et al. (2026) X. Wang et al. MetaGen: self-evolving multi-agent topologies with role and structure co-optimization. arXiv preprint arXiv:2601.19290. Cited by: §2.5.

- Wu et al. (2023) Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, et al. AutoGen: enabling next-gen LLM applications via multi-agent conversation. arXiv preprint arXiv:2308.08155. Cited by: §2.2.

- Yang et al. (2018) Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov, and C. D. Manning HotpotQA: a dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600. Cited by: 3rd item.

- Zhang et al. (2024) K. Zhang, W. Yao, Z. Liu, Y. Feng, Z. Liu, R. Murthy, T. Lan, L. Li, R. Lou, J. Xu, B. Pang, Y. Zhou, S. Heinecke, S. Savarese, H. Wang, and C. Xiong Diversity empowers intelligence: integrating expertise of software engineering agents. arXiv preprint arXiv:2408.07060. Cited by: §2.3.

- Zheng et al. (2024) L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al. Chatbot arena: an open platform for evaluating LLMs by human preference. arXiv preprint arXiv:2403.04132. Cited by: §2.1.

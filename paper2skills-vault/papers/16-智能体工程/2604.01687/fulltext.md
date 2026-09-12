<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2604.01687
     paper_id : 2604.01687
     source   : https://arxiv.org/html/2604.01687v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# EvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification

Hanrong Zhang Affiliation: University of Illinois Chicago Email: hzhan135@uic.edu    Shicheng Fan Affiliation: University of Illinois Chicago Email: psyu@uic.edu    Henry Peng Zou Affiliation: University of Illinois Chicago Email: xiaoxiao.li@ece.ubc.ca    Yankai Chen Affiliation: MBZUAI Affiliation: McGill University    Zhenting Wang Affiliation: MBZUAI    Jiayu Zhou Affiliation: Columbia University    Chengze Li Affiliation: University of Illinois Chicago    Wei-Chieh Huang Affiliation: University of Illinois Chicago    Yifei Yao Affiliation: Zhejiang University    Kening Zheng Affiliation: University of Illinois Chicago    Xue (Steve) Liu Affiliation: MBZUAI Affiliation: McGill University    Xiaoxiao Li Affiliation: University of British Columbia    Philip S. Yu Affiliation: University of Illinois Chicago

###### Abstract

Anthropic proposes the concept of skills for LLM agents to tackle multi-step professional tasks that simple tool invocations cannot address. A tool is a single, self-contained function, whereas a skill is a structured bundle of interdependent multi-file artifacts. Currently, skill generation is not only label-intensive due to manual authoring, but also may suffer from human–machine cognitive misalignment, which can lead to degraded agent performance, as evidenced by evaluations on SkillsBench. Therefore, we aim to enable agents to autonomously generate skills. However, existing self-evolving methods designed for tools cannot be directly applied to skills due to their increased complexity. To address these issues, we propose EvoSkills, a self-evolving skills framework that enables agents to autonomously construct complex, multi-file skill packages. Specifically, EvoSkills couples a Skill Generator that iteratively refines skills with a Surrogate Verifier that co-evolves to provide informative and actionable feedback without access to ground-truth test content. On SkillsBench, EvoSkills achieves the highest pass rate among five baselines on both Claude Code and Codex, and also exhibits strong generalization capabilities to six additional LLMs.

| |

## 1 Introduction

LLM agents have made rapid progress through tool use and API calling (Yao et al., 2022; Schick et al., 2023; Zhang et al., 2024; Qin et al., 2023; Patil et al., 2023). However, professional open-ended tasks, such as complex software repair, multi-step scientific analysis, and enterprise data pipeline orchestration, require far more than isolated tool invocations. Agents must orchestrate a coherent procedure across multiple steps and artifacts: decomposing goals, coordinating tools, recovering from failures, and validating intermediate outputs. This is profoundly challenging because decisions are long-horizon, instructions and scripts are tightly coupled, and reliable environmental feedback is often sparse or delayed.

To bridge the gap, Anthropic proposed the concept of agent skills (Anthropic, 2025a). Fig. 1 illustrates the difference between a tool and a skill: a tool is usually a simple function, whereas a skill is a structured package of workflow instructions, executable scripts, and domain references (Xu and Yan, 2026). According to the systematic evaluation presented in SkillsBench (Li et al., 2026b), equipping agents with well-crafted skills yields consistent performance gains across a broad spectrum of professional domains, including software engineering and scientific analysis etc., confirming that structured procedural guidance substantially augments task-solving capability beyond what bare tool access affords.

Despite the demonstrated utility of skills, the prevailing paradigm relies almost entirely on human authoring, a process that is both labor-intensive and difficult to scale, with no systematic guarantee of output quality. As demonstrated in Fig. 6, the SkillsBench evaluation (Li et al., 2026b) reveals that human-curated skills yield highly uneven gains: while certain domains benefit substantially, others, such as Natural Science, even exhibit degraded performance after skill integration. We hypothesize that a key driver of this inconsistency is human–machine cognitive misalignment: workflows and abstractions designed to be intuitive for human experts do not naturally match how LLM agents process context, reason, and act under execution constraints.

*Figure 1: Tool–skill difference illustration.*

To reduce manual effort, recent approaches have shifted from pre-defining static tools or APIs to self-evolve tools or tools libraries by the LLM agent itself (Chen et al., 2026; Li et al., 2026a; Lu et al., 2026; Wang et al., 2023; Xia et al., 2025). However, these methods suffer from a fundamental tool–skill gap: they are inherently designed for one-shot generation of simple, self-contained functions, and are inadequate for the creation of structured, multi-file skill packages that coordinate workflow instructions, executable scripts, and domain references across multiple artifacts. Moreover, the work Alzubi et al. (2026) relies heavily on ground-truth supervision for failure diagnosis, limiting applicability in real-world settings where such signals are unavailable.

*Figure 2: Skill quality improvement across 5 evolution rounds. EvoSkills surpasses human-curated skills within 5 evolution iterations.*

To address these challenges, we propose a self-evolving skills framework EvoSkills. To overcome the inherent unreliability of one-shot multi-file skill generation, we design a main component, the Skill Generator, to iteratively generate and refine skill bundles, with skill quality steadily improving across evolution rounds, as shown in Fig. 2. It also maintains a persistent conversation context that accumulates high-fidelity feedback from another main component, the Surrogate Verifier, across iterations. The Surrogate Verifier, a separate LLM session without inheriting the generator’s biases, is designed to address the lack of ground-truth feedback in the real world. It synthesizes test cases and scripts according to the task instructions and environment, and provides high-fidelity feedback to the Skill Generator to co-evolutionarily improve skill generation quality.

In summary, our key contributions are as follows: ❶ We introduce EvoSkills, a co-evolutionary framework for LLM agents to self-evolve robust multi-file skill packages. ❷ We demonstrate key insights on autonomous skill generation: (i) agents create better skills than human-curated ones by capturing the reasoning patterns and tool-use strategies that agents actually need; (ii) self-evolved skills are portable across different model families, as the evolved packages encode reusable task structure rather than model-specific artifacts. ❸ We conduct extensive experiments on SkillsBench, EvoSkills achieves the highest pass rate of 71.1% (+40.5pp over the no-skill baseline), substantially surpassing all five baselines. Furthermore, skills evolved by a single frontier LLM transfer effectively to six additional LLMs from five companies, yielding 35–45pp gains over their respective no-skill baselines.

## 2 Related Work

#### LLM Agent Skills.

Anthropic introduced Agent Skills (Anthropic, 2025a) as shown in Fig. 1. A recent systematization (Jiang et al., 2026) further distinguishes skills from atomic tools and one-off plans, defining them as reusable modules with explicit applicability and termination conditions. SkillsBench (Li et al., 2026b) provides the first systematic benchmark for evaluating agent skills, comprising 87 tasks across 11 domains with deterministic verifiers. Several learning-based approaches attempt to close this gap. SAGE (Wang et al., 2025) trains agents on chains of related tasks with a skill-integrated reward, yet the resulting skills remain single-file programmatic functions rather than structured multi-file packages. SkillRL (Xia et al., 2026) distills trajectories into a hierarchical skill bank via reinforcement learning, but relies on teacher-guided distillation and produces prompt-level heuristics rather than executable artifacts. EvoSkills instead generates structured, multi-file skill packages and evolves them through iterative verification.

#### Self-evolving LLM Agents.

A growing body of work automates the self-improvement of agent capabilities, yet existing methods exhibit two recurring shortcomings. First, most self-evolving pipelines produce only single tools or function APIs or just prompt heuristics (Wang et al., 2023; Li et al., 2026a; Xia et al., 2025; Lu et al., 2026; Chen et al., 2026), and none can construct the multi-file structure a full skill package demands. Moreover, AutoSkill (Yang et al., 2026) and AutoRefine (Qiu et al., 2026) extract reusable knowledge as prompt templates rather than executable packages, while SEAgent (Sun et al., 2025) internalizes capabilities into model weights, making them non-inspectable and non-transferable (Jiang et al., 2026). Second, some methods heavily rely on ground-truth signals for failure diagnosis, limiting applicability when such supervision is unavailable (Alzubi et al., 2026; Sun et al., 2025). EvoSkills addresses both limitations by iteratively generating and evolving structured, multi-file skill packages, and employs information-isolated surrogate verification to provide structured failure diagnostics and feedback instead of relying on ground-truth signals.

## 3 Method

### 3.1 Method Overview

To answer the research question posed in Sec. 1, two core difficulties must be overcome: (1) generating a multi-file skill bundle in a single pass is inherently unreliable; and (2) the agent lacks ground-truth feedback during self-evolution. EvoSkills addresses both challenges by co-evolving the agent’s skill and a corresponding surrogate verifier. Fig. 3 illustrates the overall framework, and Alg. 1 formalizes the complete co-evolutionary procedure. Given a task input, the Skill Generator produces candidate skills and executes them to obtain task outputs. An informationally isolated Surrogate Verifier then independently generates and evolves test assertions against those outputs, providing structured failure diagnostics back to the generator. The two components co-evolve through iterative generate–verify–refine cycles: whenever the surrogate tests pass, a ground-truth oracle test re-executes the skill in a fresh environment and returns only an opaque success/failure signal. If the oracle passes, the final evolved skill is deployed to the target LLM agent (e.g., Claude Code (Anthropic, 2025b) and Codex (OpenAI, 2025)); otherwise, the signal triggers a new co-evolution iteration, in which the verifier escalates its tests and the generator refines the skill accordingly. Next, we formalize the task setting as a partially observable Markov decision process (POMDP) in Sec. 3.2 and present the EvoSkills framework in detail in Sec. 3.3.

*Figure 3: Overview of the EvoSkills co-evolutionary framework. The Skill Generator and Surrogate Verifier co-evolve through iterative refinement. The verifier provides structured failure feedback to drive skill improvement, while a ground-truth oracle test returns only an opaque pass/fail signal, triggering test escalation and ensuring strict information isolation.*

### 3.2 Problem Formulation

#### Task Definition.

Since the LLM agent never knows what the held-out ground-truth tests actually check, i.e., the success criteria remain entirely hidden, we define the task environment as a POMDP $\mathcal{M}=\langle\mathcal{X},\mathcal{A},T,\mathcal{O},\Omega,\mathcal{R}\rangle$. Here, $\mathcal{X}$ is the underlying state space, i.e., the complete filesystem and processes, $\mathcal{A}$ comprises the agent’s actions, i.e., terminal commands and file edits, $T(x^{\prime}\mid x,a)$ is the deterministic state transition upon executing action $a$ in state $x$ and reaching successor state $x^{\prime}$, $\mathcal{O}$ is the observation space, i.e., command execution results, $\Omega(o\mid x,a)$ maps post-action states to partial observations, and $\mathcal{R}(x_{T})\in[0,1]$ evaluates the output files in final state $x_{T}$ against hidden ground-truth tests. Because the agent only receives partial observations $o_{t}\sim\Omega(\cdot\mid x_{t},a_{t})$, it acts based on the observation–action history $h_{t}=(o_{1},a_{1},\ldots,a_{t-1},o_{t})$.

#### Optimization Objective.

Unlike atomic tools that offer simple functional interfaces, a skill $\mathcal{S}$ (Anthropic, 2025a) is a structured bundle of domain-specific instructions, executable scripts, and reference materials that collectively guide an agent in navigating a task space. The skill conditions the agent’s policy:

$a_{t}\sim\pi_{\theta}(a_{t}\mid h_{t},\,\mathcal{S}),$ | | | | (1) |

where $\pi_{\theta}$ is an LLM policy. We define the expected terminal reward under skill $\mathcal{S}$ as:

$J(\mathcal{S})\;\triangleq\;\mathbb{E}_{\tau\sim P(\tau\mid\pi_{\theta},\,\mathcal{S},\,\mathcal{M})}\bigl[\mathcal{R}(x_{T})\bigr],$ | | | | (2) |

where $\tau=(o_{1},a_{1},\ldots,a_{T-1},o_{T})$ is the execution trajectory. Our objective is to discover an optimal skill $\mathcal{S}^{*}$ that maximizes $J$:

$\mathcal{S}^{*}=\arg\max_{\mathcal{S}}\;J(\mathcal{S}).$ | | | | (3) |

*Algorithm 1 EvoSkills co-evolution Algorithm*

0:  Instruction $I$, environment $\mathcal{E}$, meta-skill $\mathcal{S}_{\mathrm{meta}}$ (skill-creator)

0:  Evolution iters $N{=}5$, surrogate iters $M{=}15$, context cap $\beta{=}0.7$

0:  LLM policy $\pi_{\theta}$ (generator), independent verifier policy $\pi_{\theta}^{V}$

0:  Final evolved skill $\mathcal{S}^{*}$

1:  $C\leftarrow(I,\,\mathcal{S}_{\mathrm{meta}})$ // in: instruction $I$, meta-skill; out: initial context $C$

2:  $\mathcal{S}^{(0)}\sim\pi_{\theta}\!\left(\cdot\mid C\right)$ // in: context $C$; out: skill bundle $\mathcal{S}^{(0)}$ (code + SKILL.md)

3:  $\mathcal{V}^{(0)}\leftarrow\emptyset$ // surrogate verifier test suite

4:  $i\leftarrow 0$;  $j\leftarrow 0$;  $n\leftarrow 0$;  $r\leftarrow 0$ // skill version; test-suite version; evolution iter.; surrogate iter.

5:  $\mathcal{R}_{\mathrm{best}}\leftarrow 0$;  $\mathcal{S}^{*}\leftarrow\mathcal{S}^{(0)}$ // $\mathcal{R}_{\mathrm{best}}$: best oracle score so far

6:  while $n<N$ and $r<M$ do

7:   // — Skill Generator: execute and produce outputs —

8:   $x^{(i)}\leftarrow\Phi(\mathcal{S}^{(i)},\,\mathcal{E})$ // $\Phi$: execute skill in env and collect outputs; out: artifacts $x^{(i)}$

9:   if LLM context usage proportion $>\beta$ then

10:    break // prevent LLM context overflow

11:   end if

12:   // — Surrogate Verifier: evaluate and refine (Eq. 4, Eq. 5) —

13:   $\tilde{\mathcal{R}}^{(i,j)}\leftarrow\tilde{\mathcal{R}}(x^{(i)},\,\mathcal{V}^{(j)})$ // $\tilde{\mathcal{R}}$: surrogate reward; in: artifacts, tests; out: pass rate $\in[0,1]$

14:   if $\tilde{\mathcal{R}}^{(i,j)}<1$ then

15:    $\mathcal{F}^{(i,j)}\sim\pi_{\theta}^{V}\!\left(\cdot\mid I,\,x^{(i)},\,\mathcal{V}^{(j)}\right)$ // in: $I$, artifacts $x^{(i)}$, tests $\mathcal{V}^{(j)}$; out: diagnostic $\mathcal{F}$

16:    $C\leftarrow C\oplus\mathcal{F}^{(i,j)}$ // append error diagnostic $\mathcal{F}$ to generator context $C$

17:    $\mathcal{S}^{(i+1)}\sim\pi_{\theta}\!\left(\cdot\mid\mathcal{S}^{(i)},\;C\right)$ // skill refinement (Eq. 7)

18:    $i\leftarrow i{+}1$;  $r\leftarrow r{+}1$;  continue // evolve $\mathcal{S}$; $\mathcal{V}^{(j)}$ locked

19:   end if

20:   // — Ground-Truth Oracle Test: independent re-execution in fresh environment —

21:   $\hat{x}^{(i)}\leftarrow\Phi(\mathcal{S}^{(i)},\,\mathcal{E}^{\prime})$ // in: skill $\mathcal{S}^{(i)}$, fresh env $\mathcal{E}^{\prime}$; out: artifacts $\hat{x}^{(i)}$

22:   $\mathcal{R}^{(i)}\leftarrow\mathcal{R}(\hat{x}^{(i)})$;  $n\leftarrow n{+}1$ // $\mathcal{R}$: ground-truth oracle reward; out: score $\in[0,1]$

23:   if $\mathcal{R}^{(i)}=1$ then

24:    $\mathcal{S}^{*}\leftarrow\mathcal{S}^{(i)}$;  return $\mathcal{S}^{*}$ // early exit: perfect score

25:   else if $\mathcal{R}^{(i)}>\mathcal{R}_{\mathrm{best}}$ then

26:    $\mathcal{R}_{\mathrm{best}}\leftarrow\mathcal{R}^{(i)}$;  $\mathcal{S}^{*}\leftarrow\mathcal{S}^{(i)}$ // save best snapshot

27:   end if

28:   // — Co-evolution: test escalation (Eq. 6, Eq. 8) —

29:   $C\leftarrow C\oplus\mathbf{1}[\mathcal{R}^{(i)}{<}1]$ // $\mathbf{1}[\cdot]$: indicator fn; append oracle pass/fail bit to $C$ (no test content)

30:   $\mathcal{V}^{(j+1)}\sim\pi_{\theta}^{V}\!\left(\cdot\mid I,\,x^{(i)},\,\mathcal{V}^{(j)}\right)$ // verifier escalation (Eq. 8)

31:   $j\leftarrow j{+}1$ // $\mathcal{V}$ evolves; $\mathcal{S}^{(i)}$ re-evaluated next iteration

32:  end while

33:  return $\mathcal{S}^{*}$ // save best skill

### 3.3 EvoSkills Framework

However, directly optimizing $J(\mathcal{S})$ (Eq. 3) is intractable: ground-truth evaluation is computationally expensive and returns only an opaque pass/fail signal; no test content or failure details are revealed to the agent. To provide dense, actionable feedback, we introduce a surrogate verifier reward $\tilde{\mathcal{R}}(x,\mathcal{V})$, defined by a suite of deterministic test assertions $\mathcal{V}=\{e_{1},\ldots,e_{|\mathcal{V}|}\}$ generated by an independent verifier

$\tilde{\mathcal{R}}(x,\mathcal{V})\;\triangleq\;\frac{1}{|\mathcal{V}|}\sum_{k=1}^{|\mathcal{V}|}\mathbf{1}\!\left[e_{k}(x)\right]\;\in\;[0,\,1],$ | | | | (4) |

where $x$ denotes the output files produced by skill execution and $\mathbf{1}[e_{k}(x)]$ indicates whether assertion $e_{k}$ passes on $x$. However, the surrogate is only useful insofar as it faithfully approximates the hidden $\mathcal{R}$. This creates a coupled optimization problem: the skill must maximize a proxy that itself must be aligned with the hidden ground truth. Since the ground-truth (GT) oracle test reveals only a binary pass/fail signal $\mathbf{1}[\mathcal{R}(\hat{x}^{(i)})<1]$, where $\hat{x}^{(i)}$ denotes the oracle’s independent re-execution output, and no test content, we cannot directly optimize against $\mathcal{R}$. Instead, letting $I$ denote the task instruction, we define the rollout operator $\Phi(\mathcal{S},\mathcal{E})$ as the execution output obtained by rolling out $\pi_{\theta}(\cdot\mid h_{t},\mathcal{S})$ in environment $\mathcal{E}$, so that $x^{(i)}=\Phi(\mathcal{S}^{(i)},\mathcal{E})$ is the output of the $i$-th skill version and $\hat{x}^{(i)}=\Phi(\mathcal{S}^{(i)},\mathcal{E}^{\prime})$ is the output produced by an independent oracle re-execution in a fresh environment $\mathcal{E}^{\prime}$. $\mathcal{V}^{(j)}$ is the $j$-th version of the surrogate verifier test suite. EvoSkills proceeds via alternating refinement:

$\displaystyle\mathcal{S}^{(i+1)}\;\leftarrow\;\arg\max_{\mathcal{S}}\;\tilde{\mathcal{R}}\!\bigl(\Phi(\mathcal{S},\,\mathcal{E}),\;\mathcal{V}^{(j)}\bigr),$ | | Skill refinement: | | | (5) |

$\displaystyle\mathcal{V}^{(j+1)}\sim\pi_{\theta}^{V}\!\bigl(\,\cdot\mid I,\;x^{(i)},\;\mathcal{V}^{(j)}\bigr),\;\;\text{if }\mathbf{1}\!\left[\tilde{\mathcal{R}}(x^{(i)},\,\mathcal{V}^{(j)}){=}1\;\wedge\;\mathcal{R}(\hat{x}^{(i)}){<}1\right]$ | | Test escalation: | | | (6) |

In practice, the $\arg\max$ in Eq. 5 is approximated by iterative LLM sampling (Eq. 7). Skill refinement maximizes $\tilde{\mathcal{R}}$ under a fixed verifier test suite $\mathcal{V}^{(j)}$; test escalation is triggered only when the oracle’s binary signal exposes a gap between $\tilde{\mathcal{R}}$ and $\mathcal{R}$, forcing the verifier to independently strengthen its tests without any access to ground-truth test content. EvoSkills realizes this alternating optimization through three informationally isolated components: a Skill Generator and a Surrogate Verifier orchestrated in a co-evolutionary loop (Alg. 1).

#### Skill Generator.

A single-pass skill generation often produces bundles with coverage gaps and logical errors, because the agent lacks ground-truth feedback during generation. To enable iterative refinement, the Skill Generator maintains a persistent conversation context $C$, initialized as $C^{(0)}=(I,\mathcal{S}_{\mathrm{meta}})$, where $I$ is the task instruction and $\mathcal{S}_{\mathrm{meta}}$ is a domain-agnostic meta-skill (skill-creator) that teaches how to create skills. Each skill revision evolves from the previous version, the LLM $\pi_{\theta}$ (Eq. 1) reads the current skill $\mathcal{S}^{(i)}$ together with accumulated verification feedback, and produces an improved version (Alg. 1, lines 15–17):

$\mathcal{S}^{(i+1)}\sim\pi_{\theta}\!\left(\cdot\mid\mathcal{S}^{(i)},\;C^{(i+1)}\right),\quad C^{(i+1)}=C^{(i)}\oplus\mathcal{F}^{(i,j)},$ | | | | (7) |

where $\mathcal{S}^{(i)}$ is the $i$-th skill version, $\mathcal{F}^{(i,j)}$ is the failure diagnostic from the Surrogate Verifier after evaluating $\mathcal{S}^{(i)}$ under test suite $\mathcal{V}^{(j)}$, comprising failed test cases, root-cause analysis, and actionable revision suggestions, and $\oplus$ appends detailed feedback to the LLM context. The agent executes $\mathcal{S}^{(i)}$ via rollout $x^{(i)}=\Phi(\mathcal{S}^{(i)},\mathcal{E})$ (line 8), and the resulting outputs are passed to the Surrogate Verifier for evaluation.

#### Surrogate Verifier.

Since the ground-truth reward $\mathcal{R}$ returns only an opaque pass/fail signal, the Skill Generator lacks dense feedback to diagnose and correct errors. The Surrogate Verifier addresses this gap by serving as a proxy for $\mathcal{R}$, providing per-assertion failure diagnostics that the opaque oracle cannot. To further prevent the confirmation bias inherent in self-verification, the Surrogate Verifier operates in a completely independent LLM session $\pi_{\theta}^{V}$, observing only the task instruction $I$ and the output files $x^{(i)}$, remaining blind to the Skill Generator’s reasoning, code, and skill content. This information isolation ensures that the verifier’s test generation is conditionally independent of the generator’s internal state, preventing the verifier from inheriting the generator’s biases. The verifier generates a proxy verifier test suite $\mathcal{V}=\{e_{1},\ldots,e_{|\mathcal{V}|}\}$ of deterministic assertions, yielding the proxy reward $\tilde{\mathcal{R}}(x,\mathcal{V})$ defined in Eq. 4 (Alg. 1, line 13). The verifier iteratively refines this verifier test suite by reading the previous script $\mathcal{V}^{(j)}$ and current outputs $x^{(i)}$:

$\mathcal{V}^{(j+1)}\sim\pi_{\theta}^{V}\!\bigl(\,\cdot\mid I,\;x^{(i)},\;\mathcal{V}^{(j)}\bigr),$ | | | | (8) |

where the verifier conditions on the task instruction $I$, the agent’s current outputs $x^{(i)}$, and its own previous test script $\mathcal{V}^{(j)}$ to produce an improved verifier test suite $\mathcal{V}^{(j+1)}$. When the surrogate reward indicates failure ($\tilde{\mathcal{R}}<1$), the verifier additionally generates a structured failure diagnostic $\mathcal{F}^{(i,j)}\sim\pi_{\theta}^{V}(\cdot\mid I,\,x^{(i)},\,\mathcal{V}^{(j)})$, including per-assertion results, root-cause analysis, and actionable revision suggestions fed back to the Skill Generator (Eq. 7).

#### Co-evolution of Skill Generator and Surrogate Verifier.

The alternating optimization in Eq. 5–Eq. 6 couples two feedback pathways (Alg. 1). When the surrogate reward indicates failure ($\tilde{\mathcal{R}}<1$), the verifier test suite $\mathcal{V}$ is held fixed and the failure diagnostic $\mathcal{F}$ drives skill revision (Eq. 7). When the surrogate test passes but the ground-truth oracle fails, only an opaque pass/fail bit is returned, i.e., no test content or failure details to prevent the Skill Generator from overfitting to the held-out tests. The Surrogate Verifier must then independently escalate its test suite (Eq. 8) according to the updated skills and test output files. For example, it may generate more diverse, comprehensive and challenging test cases. Through this dual-feedback mechanism, the skill $\mathcal{S}$ improves under surrogate test pressure. Fig. 2 empirically confirms that this co-evolutionary loop converges within a small number of iterations.

## 4 Experiments

In this section, we aim to answer four Research Questions (RQ): (1) Are self-evolved skills useful, and can they outperform human-curated skills (Sec. 4.2)? (2) How do different components of the method influence overall performance? (Sec. 4.3)? (3) Can evolved skills be transferred across other LLM models or LLM agents from different companies (Sec. 4.4)? (4) How are performance gains distributed across professional domains (Sec. 4.5)?

### 4.1 Experimental Setup

We evaluate EvoSkills on SkillsBench (Li et al., 2026b). It contains 87 tasks across roughly 20 professional domains, providing broad coverage of the real-world distribution of skill-augmented tasks. Each task is paired with a deterministic verifier, enabling reproducible, binary pass/fail evaluation without subjective human judgment. We adopt SkillsBench as the sole evaluation suite because to the best of our knowledge, it is the only benchmark purpose-built for assessing the utility of agent skills. The primary metric is pass rate: the proportion of tasks with reward $=1.0$, i.e., all tests passed, otherwise reward $=0.0$. Unless otherwise stated, all conditions use the same instruction format. For conditions with pre-installed skills, the task instruction notes their availability.

*Figure 4: Skill quality comparisons with baselines on SkillsBench (Claude Opus 4.6 + Claude-Code). Error bars: $\pm$1 std over 5 runs.*

We compare six baselines versus EvoSkills. The No-Skill Baseline evaluates the agent’s performance when no skills are available. Self-Generated Skills replicates the one-pass self-generation condition from SkillsBench (Li et al., 2026b): the agent generates one to five skill documents in a single pass before solving the task, with no iterative evolution or verification (see prompt in Sec. F.4). CoT-Guided Self-Generation extends the Self-Generated Skills condition with a structured five-step chain-of-thought prompt (see prompt in Sec. F.5). Skill-Creator adopts Anthropic’s official skill-creator (Anthropic, 2025a). Since our evaluation is fully autonomous, we replace its human-interactive steps with autonomous equivalents (see Sec. F.3): a first session iteratively drafts, self-tests, and refines skills for at least three iterations, and a second session solves the task using the resulting skills. Human Curated Skills pre-installs the human-authored skill packages released with SkillsBench. For the evolution agent, we use Claude Opus 4.6 and GPT-5.2 as backbone models. We also provide the evolution agent with background context of the tasks. For baseline comparisons, each primary method is evaluated over 5 independent runs, and mean $\pm$ standard deviation is reported.

In addition, we also evaluate the transferability of the skills evolved by Claude Opus 4.6 on six additional models: GPT-5.2 (Singh et al., 2025), Claude Sonnet 4.5 (Anthropic, 2025d), Claude Haiku 4.5 (Anthropic, 2025c), Qwen3-Coder-480B (Yang et al., 2025), DeepSeek V3-671B (Liu et al., 2024) and Mistral Large 3-675B (Mistral AI, 2025). Each model is evaluated over 3 independent runs. Please refer to Appendix A for configuration details.

### 4.2 RQ1: Skill Quality Comparison

Fig. 4presents the core comparison on Claude Opus 4.6 with Claude-Code. EvoSkills reaches a 71.1% pass rate, exceeding the no-skill baseline (30.6%) by $+40.5$pp and human-curated skills (53.5%) by $+17.6$pp. The Skill-Creator baseline achieves only 34.1%, barely above the no-skill baseline; an even simpler variant that generates and uses skills within a single session yields just 32.4%. Two in-session self-generation baselines perform no better: the SkillsBench self-generated skills baseline (Li et al., 2026b) reaches 32.0% ($\pm$3.1), and a CoT-guided variant that follows a structured five-step chain-of-thought prompt reaches 30.7% ($\pm$5.2), both providing negligible improvement over the no-skill baseline. This confirms that skill generation without co-evolutionary verification is insufficient regardless of the generation strategy: the gains of EvoSkills originate from the iterative verification loop, not from the skill-creation prompt itself. A per-task breakdown is provided in Appendix D. We summarize the main insight as follows.

### 4.3 RQ2: Ablation Studies

Due to page limitations, we present the full ablation analysis in Appendix B. Key findings (Tab. B1): removing the surrogate verifier drops pass rate from 71.1% to 41.1%, and using background context yields only 48.6%. Both results confirm that iterative verification and structured packaging are essential.

### 4.4 RQ3: Skill Cross-Model Transferability

*Figure 5: Cross-model skill transferability on SkillsBench. Skills evolved by Claude Opus 4.6 are transferred to six additional models spanning five providers. Each pair of bars shows the no-skill baseline (red) and the with-skills pass rate (blue). Delta annotations indicate absolute improvement. All models benefit substantially (+36–44pp), confirming that the evolved skills encode reusable task structure rather than model-specific artifacts.*

Having established that EvoSkills produces high-quality skills on two primary LLMs, we next examine whether these skills generalize across LLM model families. As shown in Fig. 5, self-evolved skills yield substantial gains on both primary backbones: Claude Opus 4.6 (+40.5pp) and GPT-5.2 (+40.2pp). Transferring Opus-evolved skills to six additional models consistently improves performance over each no-skill baseline by +36 to +44pp, demonstrating that the distilled workflows are not tied to the originating model. Full numerical results are provided in Tab. A3 (Appendix A).

Interestingly, GPT-5.2 benefits from Opus-transferred skills (65.0%), yet its own self-evolved skills (69.8%) still outperform the transferred set by 4.8pp, indicating a modest but consistent advantage for model-matched evolution. We summarize the main takeaway as follows.

### 4.5 RQ4: Domain-Level Analysis

Beyond aggregate metrics, Fig. 6 reveals how gains are distributed across domains. Self-evolved skills outperform human-curated skills in 9 of 11 domains, with the largest margins in Finance ($+56.9$pp over human-curated) and Cybersecurity ($+23.2$pp). The pattern is non-uniform: domains where human-curated skills already perform well (Energy, Robotics) see diminishing returns from evolution, whereas domains where human curation provides little benefit see the largest gains. We summarize the main takeaway as follows.

*Figure 6: Per-domain pass rates on SkillsBench. Three conditions are compared using Claude Opus 4.6: no-skill baseline, human-curated skills, and EvoSkills self-evolved skills, across 11 professional domains. Numbers in parentheses indicate task counts. Self-evolved skills outperform human-curated skills in 9 of 11 domains. The arrow highlights Natural Science, where human-curated skills degrade performance, whereas self-evolved skills yield substantial gains, evidencing human–machine cognitive misalignment.*

### 4.6 Evolution Dynamics

Fig. 2traces the pass rate across evolution rounds against three static baselines: the no-skill baseline (30.6%), Anthropic’s naive skill-creator (34.1%), and human-curated skills (53.5%). At round 0 (one-shot generation without verification), EvoSkills performs on par with the no-skill baseline, but the pass rate climbs sharply once iterative verification begins, reaching 44% at round 2, surpassing human-curated skills at round 3 (63%), and converging at 75% by round 5. This confirms that the co-evolutionary loop, not the generation prompt, is the primary driver of skill quality, and that convergence within five rounds keeps the evolution cost practical. Appendix C provides the full distribution of verification cycles and oracle rounds across all 86 tasks: each task requires 4.1 verification cycles and 2.4 evolution iterations on average to achieve convergence. Moreover, Appendix E presents a detailed trace of a representative evolution trajectory.

## 5 Conclusion

We presented EvoSkills, a co-evolutionary framework for agent skill self-generation. This design overcomes both the unreliability of one-shot skill generation and the lack of ground-truth feedback in real-world settings. On SkillsBench, EvoSkills substantially outperforms human-curated skills and all self-generation baselines, while demonstrating strong transferability. Our experiments reveal a human–machine cognitive misalignment that co-evolutionary optimization can bridge. In the future, we plan to extend the framework to multi-model skill evolution.

## References

- Alzubi et al. (2026) S. Alzubi, N. Provenzano, J. Bingham, W. Chen, and T. Vu EvoSkill: automated skill discovery for multi-agent systems. arXiv preprint arXiv:2603.02766. Cited by: §1, §2.

- Anthropic (2025a) Anthropic Agent skills overview. Note: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overviewAccessed: 2026-03-30 Cited by: §1, §2, §3.2, §4.1.

- Anthropic (2025b) Anthropic Claude code: an agentic coding tool. Note: https://code.claude.com/docs Cited by: Table A2, Table A2, §3.1.

- Anthropic (2025c) Anthropic Claude haiku 4.5 system card. Note: https://www.anthropic.com/claude-haiku-4-5-system-card Cited by: §4.1.

- Anthropic (2025d) Anthropic Claude sonnet 4.5 system card. Note: https://www.anthropic.com/claude-sonnet-4-5-system-card Cited by: §4.1.

- Chen et al. (2026) S. Chen, J. Gai, R. Zhou, J. Zhang, T. Zhu, J. Li, K. Wang, Z. Wang, Z. Chen, K. Kaleb, et al. SkillCraft: can llm agents learn to use tools skillfully?. arXiv preprint arXiv:2603.00718. Cited by: §1, §2.

- Jiang et al. (2026) Y. Jiang, D. Li, H. Deng, B. Ma, X. Wang, Q. Wang, and G. Yu SoK: agentic skills–beyond tool use in llm agents. arXiv preprint arXiv:2602.20867. Cited by: §2, §2.

- Li et al. (2026a) H. Li, S. Yang, W. Qi, S. Zhao, R. Hua, M. Song, X. Yang, and C. Peng Yunjue agent tech report: a fully reproducible, zero-start in-situ self-evolving agent system for open-ended tasks. arXiv preprint arXiv:2601.18226. Cited by: §1, §2.

- Li et al. (2026b) X. Li, W. Chen, Y. Liu, S. Zheng, X. Chen, Y. He, Y. Li, B. You, H. Shen, J. Sun, et al. SkillsBench: benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670. Cited by: §F.4, §1, §1, §2, §4.1, §4.1, §4.2.

- Liu et al. (2024) A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437. Cited by: §4.1.

- Lu et al. (2026) J. Lu, Z. Kong, Y. Wang, R. Fu, H. Wan, C. Yang, W. Lou, H. Sun, L. Wang, Y. Jiang, et al. Beyond static tools: test-time tool evolution for scientific reasoning. arXiv preprint arXiv:2601.07641. Cited by: §1, §2.

- Merrill et al. (2026) M. A. Merrill, A. G. Shaw, N. Carlini, B. Li, H. Raj, I. Bercovich, L. Shi, J. Y. Shin, T. Walshe, E. K. Buchanan, et al. Terminal-bench: benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868. Cited by: Table A2, Table A2, Table A2, Table A2.

- Mistral AI (2025) Mistral AI Introducing mistral 3. Note: https://mistral.ai/news/mistral-3 Cited by: §4.1.

- OpenAI (2025) OpenAI Introducing Codex. Note: https://openai.com/index/introducing-codex/ Cited by: Table A2, §3.1.

- Patil et al. (2023) S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez Gorilla: large language model connected with massive apis, 2023. URL https://arxiv. org/abs/2305.15334. Cited by: §1.

- Qin et al. (2023) Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, et al. Toolllm: facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789. Cited by: §1.

- Qiu et al. (2026) L. Qiu, Z. Gao, J. Chen, Y. Ye, W. Huang, X. Xue, W. Qiu, and S. Tang AutoRefine: from trajectories to reusable expertise for continual llm agent refinement. arXiv preprint arXiv:2601.22758. Cited by: §2.

- Schick et al. (2023) T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom Toolformer: language models can teach themselves to use tools. Advances in neural information processing systems 36, pp. 68539–68551. Cited by: §1.

- Singh et al. (2025) A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow, A. Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267. Cited by: §4.1.

- Sun et al. (2025) Z. Sun, Z. Liu, Y. Zang, Y. Cao, X. Dong, T. Wu, D. Lin, and J. Wang Seagent: self-evolving computer use agent with autonomous learning from experience. arXiv preprint arXiv:2508.04700. Cited by: §2.

- Wang et al. (2023) G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar Voyager: an open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291. Cited by: §1, §2.

- Wang et al. (2025) J. Wang, Q. Yan, Y. Wang, Y. Tian, S. S. Mishra, Z. Xu, M. Gandhi, P. Xu, and L. L. Cheong Reinforcement learning for self-improving agent with skill library. arXiv preprint arXiv:2512.17102. Cited by: §2.

- Xia et al. (2025) C. S. Xia, Z. Wang, Y. Yang, Y. Wei, and L. Zhang Live-swe-agent: can software engineering agents self-evolve on the fly?. arXiv preprint arXiv:2511.13646. Cited by: §1, §2.

- Xia et al. (2026) P. Xia, J. Chen, H. Wang, J. Liu, K. Zeng, Y. Wang, S. Han, Y. Zhou, X. Zhao, H. Chen, et al. Skillrl: evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234. Cited by: §2.

- Xu and Yan (2026) R. Xu and Y. Yan Agent skills for large language models: architecture, acquisition, security, and the path forward. arXiv preprint arXiv:2602.12430. Cited by: §1.

- Yang et al. (2025) A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388. Cited by: §4.1.

- Yang et al. (2026) Y. Yang, J. Li, Q. Pan, B. Zhan, Y. Cai, L. Du, J. Zhou, K. Chen, Q. Chen, X. Li, et al. AutoSkill: experience-driven lifelong learning via skill self-evolution. arXiv preprint arXiv:2603.01145. Cited by: §2.

- Yao et al. (2022) S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao React: synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, Cited by: §1.

- Zhang et al. (2024) H. Zhang, J. Huang, K. Mei, Y. Yao, Z. Wang, C. Zhan, H. Wang, and Y. Zhang Agent security bench (asb): formalizing and benchmarking attacks and defenses in llm-based agents. arXiv preprint arXiv:2410.02644. Cited by: §1.

## Appendix A Experimental Configuration

Tab. A1lists the shared configuration for evolution and evaluation. Tab. A2 specifies the agent harness paired with each model during oracle evaluation. Tab. A3 reports the full cross-model transfer results.

*Table A1: Shared configuration for evolution and evaluation.*

| Component | Setting |

| Backbones | Claude Opus 4.6, GPT-5.2 |

| Surrogate verifier model | Same as evolution backbone (Claude Opus 4.6 / GPT-5.2) |

| Ground-truth oracle test agent | Claude-Code (Claude Opus 4.6) / Codex (GPT-5.2) |

$K{=}5$$M{=}15$| Evolution stage | oracle interventions, surrogate retries |

$5\times$| Evolution runtime | timeout multiplier (effective 3000s/task), 4 parallel workers |

| Evaluation stage | timeout 7200s/task, 10 parallel workers |

*Table A2: Agent harness used for each model in the ground-truth oracle evaluation stage.*

| Model | Agent harness |

| Claude Opus 4.6 | Claude-Code (Anthropic, 2025b) |

| GPT-5.2 | Codex (OpenAI, 2025) |

| Claude Sonnet 4.5 | Claude-Code (Anthropic, 2025b) |

| Claude Haiku 4.5 | Terminus-2 (Merrill et al., 2026) |

| Qwen3 Coder | Terminus-2 (Merrill et al., 2026) |

| DeepSeek V3 | Terminus-2 (Merrill et al., 2026) |

| Mistral Large 3 | Terminus-2 (Merrill et al., 2026) |

*Table A3: Cross-model skill transferability, pass rate (%).*

$\Delta$| Model | With skills | No skill | |

| Self-Evolved Skills |

| Claude Opus 4.6 (self-evolved) | 71.1 | 30.6 | +40.5 |

| GPT-5.2 (self-evolved) | 69.8 | 29.6 | +40.2 |

| Cross-Model Transfer (Opus 4.6 Evolved Skills) |

| GPT-5.2 | 65.0 | 29.6 | +35.4 |

| Claude Sonnet 4.5 | 63.1 | 20.0 | +43.1 |

| Claude Haiku 4.5 | 54.5 | 10.4 | +44.1 |

| Qwen3 Coder | 50.8 | 8.4 | +42.4 |

| DeepSeek V3 | 48.8 | 13.0 | +35.8 |

| Mistral Large 3 | 43.1 | 4.9 | +38.2 |

## Appendix B Ablation Studies

Tab. B1summarizes the ablation results. We examine the contribution of the surrogate verifier, background context and the evolution process. All ablation experiments use Claude Code with Claude Opus 4.6 as the underlying model. The four settings are:

-

EvoSkills (Full framework): the complete EvoSkills with iterative skill evolution and surrogate verification. The evolved skills are structured multi-file packages installed before agent test.

-

W/O surrogate verifier: skill evolution proceeds without the surrogate verifier. The generator produces a skill package with the background context, and then immediately submits it to the ground-truth oracle test. If the test fails, the generator evolves the skill using only the opaque pass/fail signal without synthesized diagnostic feedback from the verifier, for up to 5 evolution iterations.

-

W/O skill evolution: the surrogate generator and skill verifier are both removed. The agent reads the background context and then directly attempts the task without evolution.

-

No-Skill Baseline: the agent directly attempts each task with the raw task instruction and environment.

#### Ablation analysis.

First, removing the surrogate verifier drops the pass rate from 71.1% to 41.1% ($-30.0$pp). The generator still evolves skills for up to 5 iterations, but relies solely on the oracle’s opaque pass/fail signal. This demonstrates that without structured diagnostic feedback identifying specific failure causes, the generator cannot perform targeted repairs, leading to inefficient evolution. Second, providing only background context documents without any evolution yields 48.6% pass rate, above the no-skill baseline ($+18.0$pp) but well below EvoSkills ($-22.5$pp). This shows that unstructured knowledge alone is insufficient without structured packaging and iterative evolution. Finally, without any skills or evolution, the agent achieves only 30.6%. The $+40.5$pp gap to EvoSkills confirms that the full co-evolutionary framework is essential.

*Table B1: Ablation results on SkillsBench, pass rate (%). Claude Opus 4.6 + Claude-Code. Ablation rows are single runs due to computational cost.*

$\Delta$| Setting | Pass rate (%) | vs. Full |

| EvoSkills (Full framework) | 71.1 | — |

$-30.0$| W/O surrogate verifier | 41.1 | |

$-22.5$| W/O evolution | 48.6 | |

$-40.5$| No-Skill Baseline | 30.6 | |

## Appendix C Evolution Iteration Analysis

Fig. C1reports the distribution of verification cycles and Ground Truth Oracle rounds across 86 evolution tasks. The left panel shows the total number of verification cycles per task, encompassing all host interventions during the evolution loop: Surrogate Verifier failures (where the agent is returned to fix issues), surrogate passes followed by Ground Truth Oracle evaluation. Each task requires 4.1 verification cycles on average before convergence.

The right panel isolates the number of Ground Truth Oracle rounds, the subset of verification cycles in which the Surrogate Verifier fully passed and the Ground Truth Oracle was invoked to evaluate the evolved skill in a clean, independent execution. Over 60% of tasks converge within 2 Ground Truth Oracle rounds, with a mean of 2.4. The 10 tasks that failed to achieve a perfect Ground Truth Oracle score cluster at higher iteration counts (5 or more verification cycles), indicating that tasks requiring many iterations are also harder to solve: the evolution loop exhausts its budget without converging.

This distribution confirms two properties of the EvoSkills framework. First, the Surrogate Verifier absorbs most of the iteration cost: out of 4.1 average verification cycles, only 2.4 escalate to the Ground Truth Oracle, meaning approximately 40% of iterations are resolved by the surrogate verifier alone. Second, the evolution budget of $K{=}5$ oracle rounds and $M{=}15$ surrogate retries is sufficient for the majority of tasks, with diminishing returns beyond 3 oracle rounds.

*Figure C1: Distribution of verification cycles (left) and Ground Truth Oracle rounds (right) across 86 evolution tasks. Verification cycles include all host interventions (Surrogate Verifier failures, Ground Truth Oracle evaluations). Ground Truth Oracle rounds count only the subset where the surrogate verifier passed and the oracle verifier was invoked. Failed tasks (red) cluster at higher iteration counts.*

## Appendix D Per-Task Breakdown

Fig. D1shows the per-task pass rate across five conditions: no-skill baseline and self-evolved skills for both Claude Opus 4.6 and GPT-5.2, plus human-curated skills for Opus 4.6. Each row is one of the 87 SkillsBench tasks, sorted by no-skill baseline difficulty. Self-evolved skills recover many tasks that both the no-skill baseline and human-curated skills fail, while a small number of hard tasks remain unsolved across all conditions.

*Figure D1: Per-task pass rate heatmap across conditions. Tasks are sorted by no-skill baseline difficulty (top = easiest). Darker cells indicate higher pass rates.*

## Appendix E Case Study: Exoplanet Transit Period Detection

This appendix presents a detailed trace of how EvoSkills evolves a skill for the Exoplanet Transit Period Detection task. The task requires the agent to detect exoplanet orbital periods from Transiting Exoplanet Survey Satellite (TESS) lightcurve data that contains stellar variability (rotational modulation), outputting a period value accurate to 5 decimal places. The Ground Truth Oracle suite comprises 4 deterministic tests (period accuracy, format, precision, and alias correctness), all of which must pass for reward $=1.0$.

The evolution proceeds through 4 script rewrites across 6 host interventions, with Ground Truth Oracle scores progressing as $75\%\rightarrow 75\%\rightarrow 100\%$. Tab. E1 summarizes the full evolution trace including Surrogate Verifier and Ground Truth Oracle outcomes at each round. Note that in Round 2, the Surrogate Verifier passes all 15 tests, yet the system does not proceed to Ground Truth Oracle evaluation because the agent’s mandatory progress checklist is incomplete. This checklist encodes all steps we consider necessary for a well-formed evolution iteration (environment discovery, skill creation, self-reflection, task execution, and summary), and the orchestrator requires every step to be marked complete before escalating to the oracle. This mechanism prevents premature oracle consumption on skill packages that have not undergone the full evolution procedure.

*Table E1: Evolution trace for the exoplanet transit detection skill. Each round records the trigger type, Surrogate Verifier outcome, Ground Truth Oracle outcome, and key event.*

| Round | Exit condition | Surrogate Verifier | Ground Truth Oracle | Key event |

| 1 | Verifier fail | 0/15 (0%) | — | Initial skill has bugs |

| 2 | Checklist fail | 15/15 (100%) | — | Progress checklist incomplete |

| 3 | Verifier pass | 15/15 (100%) | 3/4 (75%) | Period precision insufficient |

| 4 | Verifier pass | 20/20 (100%) | 3/4 (75%) | Precision fixed, alias check fails |

| 5 | Verifier fail | 19/22 (86%) | — | Surrogate Verifier catches regression |

| 6 | Verifier pass | 22/22 (100%) | 4/4 (100%) | All tests pass |

#### Version 1: BLS with biweight detrending (Ground Truth Oracle: not yet evaluated).

The agent’s first attempt uses classical Box Least Squares (BLS) with biweight detrending. The transit duration search range is set to 0.01 to 0.2 days, where the lower bound is too small and generates noise in the periodogram from fitting micro-transit-like features. The Surrogate Verifier catches format and logic bugs (0/15 tests passed), and the agent never reaches Ground Truth Oracle evaluation.

#### Version 2: Optimized BLS with wider duration range (Ground Truth Oracle: 75%).

The agent widens the transit duration search range to 0.05 to 0.3 days (more physically realistic) and adds an end-to-end find_transit_period() pipeline function. The Ground Truth Oracle returns 3/4 (75%): the detected period is close to the true value but lacks 5-decimal precision because the BLS grid resolution is too coarse.

#### Version 3: Median filter detrending (Ground Truth Oracle: 75%).

The agent switches from biweight to median filter detrending (more robust to outliers) and tightens the maximum duration to 0.15 days. The Ground Truth Oracle again returns 3/4 (75%). The precision issue persists: the BLS grid resolution appears insufficient to achieve 5-decimal accuracy regardless of the detrending method. At this point, the agent recognizes that incremental parameter tuning within BLS is insufficient.

#### Version 4: TLS with two-stage refinement (Ground Truth Oracle: 100%).

The final version introduces four changes: (a) the search algorithm switches from BLS to Transit Least Squares (TLS), which uses a realistic limb-darkened transit model instead of a box approximation, producing more accurate period estimates; (b) the detrending method changes to Savitzky-Golay filtering, which better preserves transit shape; (c) a two-stage period search is added, consisting of a broad sweep (0.5 to 15 days) followed by a narrow refinement ($\pm$2% around the candidate) for 5-decimal precision; and (d) an alias check against period harmonics ($P/2$, $2P$) is added to avoid false periods. The Ground Truth Oracle returns 4/4 (100%).

#### The surrogate-GT gap.

This task provides a concrete illustration of why the Surrogate Verifier cannot replace the Ground Truth Oracle. In Round 3 (Tab. E1), all 15 Surrogate Verifier tests passed, yet the Ground Truth Oracle reported only 3/4 (75%). The Surrogate Verifier independently ran its own BLS analysis on the raw lightcurve and used a 1% tolerance for period matching. The Ground Truth Oracle test, however, required an exact 5-decimal match, a precision threshold the Surrogate Verifier could not infer without access to the hidden test code.

By Round 5, the Surrogate Verifier had escalated to 22 tests and added BLS cross-validation. This introduced a different problem: the Surrogate Verifier’s BLS yielded a period of 3.24158 days, while the agent’s TLS produced 3.24156 days. The Surrogate Verifier flagged this 0.00002-day discrepancy as a failure, even though the agent’s answer was more accurate (TLS uses a realistic transit model). This illustrates two structural limitations of surrogate verification: (1) the Surrogate Verifier cannot replicate the Ground Truth Oracle’s exact precision requirements, and (2) it cannot distinguish its own estimation error from the agent’s error. The Ground Truth Oracle remains necessary as the authoritative arbiter.

#### Evolution summary.

Tab. E2 summarizes the design decisions across versions. The agent required four versions to converge. The key insight, that BLS appears unable to achieve 5-decimal precision on this task, only emerged after two rounds of 75% Ground Truth Oracle feedback. Without this feedback, the agent would have continued tuning BLS parameters indefinitely.

*Table E2: Skill versions for the exoplanet transit detection task. Each row records the detrending method, search algorithm, precision strategy, and Ground Truth Oracle outcome.*

| Ver. | Detrending | Algorithm | Precision strategy | Ground Truth Oracle |

| V1 | Biweight | BLS | None | — |

| V2 | Biweight (opt.) | BLS | None | 75% |

| V3 | Median filter | BLS | None | 75% |

| V4 | Savitzky-Golay | TLS | Two-stage + alias | 100% |

This case study illustrates three properties of the EvoSkills verification architecture. First, the Surrogate Verifier catches implementation bugs early (Round 1: 0/15) and detects regressions from refactoring (Round 5: 19/22), preventing these issues from consuming Ground Truth Oracle budget. Second, the Ground Truth Oracle exposes a fundamental algorithmic limitation (BLS precision ceiling) that the Surrogate Verifier’s functional tests cannot detect, because the Surrogate Verifier only checks output format and pipeline correctness, not numerical precision against hidden ground truth. Third, the co-evolutionary loop enables a qualitative shift in approach: the agent transitions from parameter tuning within a fixed algorithm (Versions 1–3) to replacing the algorithm entirely (Version 4), a decision driven by repeated 75% Ground Truth Oracle feedback.

#### Human-curated vs. self-evolved skill comparison.

For this task, SkillsBench provides 5 human-curated skills totaling 1,096 lines of documentation. The evolution produced 1 unified skill with 64 lines of procedure document and 142 lines of executable Python. Tab. E3 summarizes the structural differences.

*Table E3: Structural comparison between human-curated and self-evolved skills for the exoplanet transit detection task.*

| Aspect | Human-curated (5 skills) | Self-evolved (1 skill) |

| Total size | 1,096 lines across 5 SKILL.md | 64 lines SKILL.md + 142 lines Python |

| Executable code | None (documentation only) | 9 callable functions |

| Algorithm guidance | Lists BLS, TLS, Lomb-Scargle equally | Prescribes TLS with justification |

$\pm$| Period refinement | 2-line tip: “refine candidates” | Two-stage search: broad then 2% narrow |

$P$$2P$$P/2$| Alias detection | “Check for aliasing” (1 sentence) | Function testing , , automatically |

| Precision handling | Not addressed | Enforces 5-decimal output formatting |

The evolution discovered several patterns absent from, or only implicit in, the human-curated skills. (1) Two-stage period search: human skills mention “broad search first, then refine” as a two-line tip buried in a 246-line document; the evolved skill implements this as two distinct functions with calibrated parameters, a pattern that emerged after Versions 2 and 3 both failed at 75%. (2) Sigma-clip ordering constraint: human skills mention outlier removal before and after detrending once in passing; the evolved skill enforces this as a hard pipeline constraint with an explicit 3$\sigma$ threshold, after the agent learned that 2$\sigma$ clips remove actual transit dips. (3) Algorithm prescription vs. description: human skills present BLS, TLS, and Lomb-Scargle as equal alternatives, leaving the agent to choose; the evolved skill prescribes TLS with a concrete justification derived from three failed BLS attempts. (4) Executable functions vs. prose instructions: human skills are documentation that the agent must interpret and re-implement, risking precision bugs at each trial; the evolved skill bundles tested, debugged functions that the agent imports directly.

The final evolved skill bundles a procedure document (64 lines encoding a 9-step pipeline with domain knowledge) and a utility module (142 lines, 9 functions). When pre-installed for a fresh Opus 4.6 agent with no evolution context, the skill achieves 100% pass rate across 5 independent trials. In comparison, human-curated skills achieve 53.5% (the agent inconsistently chooses among the three algorithms across trials), the Skill-Creator baseline achieves approximately 34% (generates BLS documentation without executable code), and the no-skill baseline achieves approximately 75% (the agent defaults to Lomb-Scargle, a sinusoidal model inappropriate for transit detection).

## Appendix F Key Prompts

This appendix shows the key prompts used in the EvoSkills framework. Each prompt is presented verbatim (with minor formatting adjustments for readability).

### F.1 Evolution Agent System Prompt

The Evolution Agent receives the following system-level instruction, which governs its three-phase workflow: evolve skills, execute the task using those skills, and summarize changes. The prompt enforces skill design, mandatory self-reflection, and the constraint that all task outputs must be produced by importing skill functions rather than writing standalone code.

### F.2 Skill Discovery Hint

The following instruction is appended to the task description when pre-installed skills are available (both evolved and human-curated conditions). Without this hint, agents frequently fail to discover installed skills because skill metadata descriptions alone are insufficiently salient.

### F.3 Skill-Creator Autonomous Mode Instruction

For the Skill-Creator baseline (Sec. 4.2), the original skill-creator tool requires human interaction at several steps (e.g., reviewing drafts, selecting test cases, approving iterations). Because every method in our evaluation including EvoSkills operates without any human involvement, retaining these interactive steps would introduce an inconsistency: Skill-Creator would be the only condition receiving human guidance. We therefore replace the interactive steps with autonomous equivalents, enabling fully unattended two-phase operation: a first session generates skills and a second session uses the pre-installed results. This ensures a fair, apples-to-apples comparison across all conditions under the same fully automated protocol.

### F.4 Self-Generated Skills Prompt (Self-Generated Skills Baseline)

This prompt replicates the self-generation condition from SkillsBench (Li et al., 2026b) (Appendix C.6). It is appended to the task instruction; the agent generates skills in-session before solving the task, with no external verification.

### F.5 CoT-Guided Self-Generation Prompt

This prompt extends the Self-Generated Skills baseline with a structured five-step chain-of-thought workflow. Despite the added structure, the agent still lacks external verification feedback, and this condition achieves only 30.7% pass rate (comparable to the no-skill baseline).

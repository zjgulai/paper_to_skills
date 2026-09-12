<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2605.02801
     paper_id : 2605.02801
     source   : https://arxiv.org/html/2605.02801v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Reinforcement Learning for LLM-based Multi-Agent Systems
through Orchestration Traces

Chenchen Zhang Affiliation: Independent Researcher Email: zcc1959339538@gmail.com

May 2026

###### Abstract

As large language model (LLM) agents evolve from isolated tool users into coordinated teams, reinforcement learning (RL) must optimize not only individual actions, but also how work is spawned, delegated, communicated, aggregated, and stopped. This paper studies RL for LLM-based multi-agent systems through orchestration traces: temporal interaction graphs whose events include sub-agent spawning, delegation, communication, tool use, return, aggregation, and stopping decisions. The trace view provides a common unit for auditing reward design, credit and signal assignment, and orchestration learning.

Using this lens, we identify three technical axes. First, reward design falls into eight families; orchestration rewards target system-level properties such as parallelism speedup, split correctness, and aggregation quality. Second, reward and credit signals attach to eight credit- or signal-bearing units from token to team; explicit counterfactual message-level credit remains especially sparse in our curated pool, while agent-, role-, turn-, and orchestrator-level signals are beginning to fill in. Third, orchestration learning decomposes into five sub-decisions (when to spawn, whom to delegate to, how to communicate, how to aggregate, when to stop); within our curated pool as of May 4, 2026, we found no explicit RL training method for the stopping decision.

We connect academic methods to public industrial evidence from Kimi Agent Swarm, OpenAI Codex, and Anthropic Claude Code. The resulting scale gap should be read as a gap between publicly reported deployment envelopes and open academic evaluation regimes, not as independent verification of industrial training traces: Kimi is the clearest public trained-orchestrator anchor, while Codex and Claude Code mainly document deployment shape and harness constraints. We release the artifact at https://github.com/xxzcc/awesome-llm-mas-rl, including an $84$-entry tagged paper pool, a $32$-record exclusion log, scripted corpus statistics, and a minimal JSON schema for replayable orchestration traces, then close with fifteen research directions spanning algorithms, rewards, systems, safety, and evaluation.

## 1  Introduction

Thesis Single-agent reinforcement learning (RL) for large language models (LLMs) optimizes trajectories: a sequence of tokens, tool calls, and environment observations produced by one policy. As LLM agents evolve from isolated tool users into coordinated teams, we use orchestration traces as a working abstraction for taxonomy and audit in addition to per-agent trajectories: a temporal interaction graph in which an orchestrator decides when to spawn sub-agents, whom to delegate to, how they should communicate, which tools they may call, and how their partial outputs are aggregated. This re-frames the central technical challenges as (i) reward design across team, individual, process, tool, and verifier signals; (ii) credit assignment across agents, turns, messages, tool calls, and orchestrator decisions; and (iii) learning the orchestration process itself.

*Figure 1: Paper map. Reading: the survey takes three input traditions (single-agent LLM RL, classical MARL, and industrial agent systems), foregrounds the orchestration trace as the shared object, and then organizes the literature into reward design, credit assignment, and orchestration learning. Benchmarks, safety, and open problems are downstream because they inherit the same trace structure.*

### 1.1  Why now: recent developments

Three concurrent signals make May 4, 2026 a useful cutoff for this paper.

Public industrial evidence exposes larger deployment envelopes. Moonshot’s Kimi K2.5 introduced an Agent Swarm trained with Parallel-Agent Reinforcement Learning (PARL), scaling to up to $100$ sub-agents and $1{,}500$ coordinated steps / tool calls as reported [28]; K2.6 expanded this to $300$ sub-agents and $4{,}000$ coordinated steps, adding a “Claw Groups” research preview of cross-vendor coordination [29]. We treat these numbers as a publicly reported deployment envelope rather than an independently reproduced training trace. Kimi PARL is the clearest public example in our pool of trained multi-agent orchestration. OpenAI’s Codex app is described in official materials as a command center managing parallel software-engineering agents [45], and Anthropic’s Claude Code ships built-in and user-defined sub-agents [3], with an engineering post-mortem of sixteen parallel Claudes jointly building a C compiler [2]; in both cases the public material documents the deployment form—parallel workflows, harness boundaries, dynamic spawn—without disclosing whether multi-agent coordination itself is an RL training target. We treat Kimi as the published-training anchor and Codex / Claude Code as deployment-shape and engineering-pressure evidence (§4).

Academic methods are catching up with the right primitives. In the window from 2025-Q2 through May 2026, the literature in our pool produced a systematic multi-agent RFT paradigm [32, 47, 37], a hierarchical GRPO decomposition for LLM teams [19], a single-LLM dual-role policy optimization with tool integration [43], a stability analysis of multi-agent GRPO [15], and credit-assignment methods targeting message-level counterfactuals [7] and Shapley-based agent-level credit [31]. A May 2026 coverage refresh added closely related OpenReview, arXiv, and project-page entries on meta-thinking and deliberation [58, 70], UI-agent credit re-assignment [18], interaction-derived rewards and self-evolution [69, 81, 54], planner/workforce optimization [22], and zero-supervision MAS design [26]. A May 2026 refresh added actor-critic decentralized collaboration [38], width-scaling search teams [68], communication/topology learning [23], language-space credit assignment [71], multi-agent self-search for code [61], GUI role orchestration [62], attacker–defender safety training [65], and self-play / hierarchical interaction entries from OpenReview submissions and proceedings [34, 75, 21, 1]. These are not isolated tricks—they collectively formalize LLM collaboration as cooperative MARL with new credit- and signal-bearing units. Figure 2 visualizes the corpus across an 18-month window.

*Figure 2: Timeline of selected representative LLM-MAS entries from Q4 2024 to Q2 2026, plotted by arXiv submission date and grouped vertically by the credit-bearing unit they target (§7.1). Nearly the entire corpus sits in an 18-month window, motivating the timing claim in §1.1. The orchestrator and message rows remain sparsely populated throughout; agent- and role-level credit has received the most attention.*

| Entry | Reward | Credit | Orch. | Eval. | Safety |

$\bullet$ $\bullet$ $\circ$ $\circ$ | MAGRPO / MARFT / MAPoRL | | | | | – |

$\bullet$ $\bullet$ $\bullet$ $\circ$ | M-GRPO / MATPO / MALT | | | | | – |

$\bullet$ $\bullet$ $\circ$ $\circ$ | Dr. MAS / C3 / SHARP | | | | | – |

$\bullet$ $\circ$ $\bullet$ $\circ$ | Puppeteer / HALO / ParaManager | | | | | – |

$\bullet$ $\bullet$ $\circ$ $\bullet$ | Agent Lightning / MarsRL | | | | | – |

$\bullet$ $\circ$ $\bullet$ $\circ$ | ReMA / Learning to Deliberate | | | | | – |

$\bullet$ $\bullet$ $\bullet$ $\bullet$ | CollabUIAgents / OWL | | | | | – |

$\bullet$ $\circ$ $\bullet$ $\circ$ | CoMAS / SiriuS / Multiagent FT | | | | | – |

$\bullet$ $\circ$ $\bullet$ $\circ$ | MAE / MAS-Zero | | | | | – |

$\bullet$ $\bullet$ $\bullet$ $\circ$ | WideSeek / Agent Q-Mix / LangMARL | | | | | – |

$\bullet$ $\bullet$ $\bullet$ $\circ$ $\circ$| MAGIC / SPIRAL / MARSHAL / DEPART | | | | | |

$\circ$ $\circ$ $\bullet$ $\circ$ $\circ$| Kimi / Codex / Claude Code | | | | | |

$\circ$ $\circ$ $\bullet$ $\bullet$| TAMAS / AgentDojo / WASP | – | | | | |

$\circ$ $\bullet$ $\circ$| SWE-Bench / WebArena / MultiAgentBench | – | – | | | |

*Figure 3: Compact coverage map for representative retained entries. $\bullet$ means the entry directly studies the dimension; $\circ$ means it supplies indirect evidence, a benchmark substrate, or a system constraint. The sparsity is intentional: it shows why the survey treats reward, credit, orchestration, evaluation, and safety as coupled but unevenly supported dimensions.*

| Facet | Dominant counts | Interpretation |

| Source category | 42 RL methods; 18 benchmarks; 10 classical-MARL foundations; 6 industrial cases | The pool is method-centered but includes benchmarks, systems, and classical primitives only when they are load-bearing for the taxonomy. |

| RL relevance | 49 yes; 9 partial; 26 no | Non-RL entries are retained only as contrastive foundations, benchmarks, safety cases, or industrial deployment anchors. |

| Reward family | 15 hybrid; 10 shared; 7 orchestration; 6 verifier; 33 NA | Explicit reward design is concentrated in the LLM-MAS RL and critic/process-supervision entries. |

| Signal / credit granularity | 23 agent; 10 role; 8 orchestrator; 5 turn; 2 message; 36 NA | Message-level signals are rare, and explicit counterfactual message-level credit is rarer still; orchestrator-level signals are growing but still narrow relative to agent- and role-level signals. |

| Orchestration form | 18 centralized; 13 hierarchical; 8 debate; 4 swarm; 3 harness; 34 NA | The retained methods emphasize centralized and hierarchical controllers; many supporting entries do not define an orchestration policy. |

*Table 1: Coverage statistics computed from the $84$ retained entries in the artifact repository. These counts support the qualitative claims about sparsity in message-level explicit credit, orchestrator-level signals, and MAS-native evaluation; they should not be read as field-wide prevalence estimates.*

| Reward family | NA | agent | msg. | orch. | role | turn |

| shared | 0 | 9 | 1 | 0 | 0 | 0 |

| hybrid | 0 | 6 | 0 | 1 | 4 | 4 |

| orchestration | 0 | 0 | 0 | 6 | 1 | 0 |

| individual | 0 | 4 | 0 | 0 | 0 | 0 |

| debate | 0 | 3 | 1 | 0 | 0 | 0 |

| process | 0 | 1 | 0 | 0 | 0 | 1 |

| role | 0 | 0 | 0 | 0 | 3 | 0 |

| verifier | 4 | 0 | 0 | 0 | 2 | 0 |

| NA | 32 | 0 | 0 | 1 | 0 | 0 |

*Table 2: Reward-family by signal/credit-granularity cross-tab generated from the retained-entry CSV in the artifact repository. The table makes the sparsity claim more concrete within the retained pool: the finest retained tag is message for only two entries, and only one of those (C3) explicitly estimates message-level counterfactual credit; orchestrator-level tags now include width-scaling and orchestration-reward entries, but remain far fewer than agent-level tags. The counts are reproducible with the statistics script in the accompanying artifact repository.*

Existing surveys cover pairwise intersections but not the triple. Surveys of LLM-based multi-agent systems [6] and their collaboration mechanisms [57] cover architectures and applications; the recent $500+$-paper agentic RL survey [80] and the LLM-lifecycle RL survey [35] cover single-agent agentic RL extensively; the agentic reasoning survey [64] covers reasoning agents broadly. None triangulates the three: multi-agent and RL/post-training and LLM agents. That is the gap this paper targets.

### 1.2  Scope and positioning

The contribution is a taxonomy paper with an explicit position: LLM-MAS RL is most usefully organized around the orchestration trace. The paper therefore does not try to be a neutral catalogue of all multi-agent LLM systems. It asks a narrower question: when LLM agents are trained or post-trained as teams, which parts of the interaction graph can be rewarded, credited, and learned? The benchmark requirements in §9.4 are consequently framed as reporting recommendations derived from gaps in the retained pool, not as a new benchmark release.

Relative to LLM-MAS architecture surveys. [6, 57] catalogue agent profiles, perception, action, and interaction mechanisms, but say comparatively little about how these components are trained. Our focus is the post-training stage only: given an architecture, what rewards drive it, how is credit assigned, and how is the orchestration process itself learned?

Relative to agentic RL surveys. [80] covers the full single-agent agentic-RL landscape; we cover the step that follows: what changes when the policy is no longer a single agent but an orchestrated team of them. Many primitives carry over (PPO, GRPO, verifiable rewards) and we do not re-derive them here.

Relative to classical MARL. We treat classical MARL as a conceptual toolkit (Dec-POMDP, CTDE, COMA, Shapley credit, VDN/QMIX, MAPPO/IPPO) rather than as a field summary. §2 covers only the MARL concepts that are load-bearing for later sections; we refer the reader elsewhere for comprehensive MARL treatment.

The scope is deliberately bounded. The retained pool is not an exhaustive list of every multi-agent LLM paper, a benchmark leaderboard, or a system-design manual. We curate $51$ focal LLM-MAS entries—RL methods, industrial cases, and directly adjacent surveys—that populate the three taxonomies that structure the paper, supplemented by $33$ classical-MARL, safety, single-agent-RL, benchmark, and critic / tool-use evaluation references ($84$ retained entries total). The accompanying artifact snapshot and repository (https://github.com/xxzcc/awesome-llm-mas-rl) contain the retained-entry CSV, the $32$-record exclusion log, the statistics script, and the trace-schema files. Together these expose $116$ audited records; Appendix C gives the search strings, screening stages, borderline examples, and tag definitions.

### 1.3  Corpus construction and evidence levels

The paper pool is deliberately curated, not exhaustive. We constructed it in four passes. First, we seeded the pool from adjacent surveys on LLM-based multi-agent systems, agentic RL, and RL for LLMs [6, 57, 80, 35]. Second, we searched arXiv, ACL Anthology, OpenReview, and official project pages for combinations of multi-agent LLM, reinforcement learning, post-training, credit assignment, orchestration, agent swarm, tool use, and prompt injection. Third, we added backward and forward citation links when they supplied a load-bearing concept for one of our three taxonomies. Fourth, we audited the resulting set against three inclusion rules: the work must either (i) train or post-train an LLM-MAS component, (ii) document an industrial system whose public interface constrains RL design, or (iii) provide a benchmark, safety case, or classical-MARL primitive used later in the paper.

We exclude papers that use multiple LLM calls only as an implementation detail but do not expose multi-agent interaction, reward design, credit assignment, or orchestration as a study object. Tags in the retained-entry CSV were assigned by manual reading of the abstract, method section, and public artifact when available. The CSV records an explicit verified field and source status through its category, venue, and notes fields; we do not claim formal inter-annotator agreement. Instead, we treat the 18-column schema as a structured taxonomy artifact whose entries can be corrected as the literature changes.

Because several load-bearing systems are industrial, we separate evidence levels throughout the paper. Peer-reviewed and arXiv methods are used for algorithmic claims; company technical reports are used only where they disclose training or evaluation details; product documentation and blogs are used as deployment-shape evidence unless they explicitly disclose a training mechanism. This distinction is made explicit in §4.2.

| Protocol item | This paper |

| Cutoff | Literature and public-system audit through May 4, 2026. |

| Search sources | arXiv, ACL Anthology, OpenReview, Semantic Scholar / citation links, and official company documentation or technical blogs for deployed systems. |

$\times$| Query families | multi-agent LLM {reinforcement learning, post-training, credit assignment, orchestration, agent swarm, tool use, prompt injection}. |

$116$$84$$32$| Screening outcome | Internal audit count after the May 2026 coverage refresh: candidate records were considered; were retained in the tagged pool and exclusion decisions were logged. The retained-entry CSV is the reusable taxonomy artifact; the exclusion log is a screening-decision log, not a retained-paper bibliography or a full systematic-review reproducibility package. |

| Inclusion rule | Include work that trains or post-trains an LLM-MAS component, documents a deployment interface that constrains RL design, or supplies a benchmark, safety case, or MARL primitive used later. |

| Tagging rubric | Each retained entry is tagged for source category, RL use, reward family, finest credit granularity, orchestration form, scenario, core/case/supporting status, verification status, and notes. |

*Table 3: Curation protocol for the paper pool. The counts are internal audit counts for this paper rather than a claim of exhaustive coverage or independently reproducible screening.*

*Figure 4: Corpus construction flow. Counts are internal audit counts after the journal-revision coverage audit, not a claim of exhaustive coverage or independently reproducible screening.*

### 1.4  Contributions

-

A unifying thesis. We argue that LLM-MAS RL is usefully analyzed through the orchestration trace, understood as an event graph, rather than only through per-agent trajectories; this reframing reorganizes a large fraction of the recent literature.

-

A lightweight taxonomy formalism. We extend the Dec-POMDP to a dynamic-Dec-POMDP that accommodates spawn and despawn actions (§3) and state two informal observations: credit diffusion under uniform credit, and non-identifiability of orchestrator spawn decisions. These organize the rest of the paper. The formalism is intended as an organizing abstraction for taxonomy and auditability, not a new MARL theory; concrete algorithmic forms and tight rates are open (§11).

-

Three taxonomies. We organize methods along (a) reward design across eight families (§6), (b) credit and signal assignment across eight credit- or signal-bearing units (§7), and (c) orchestration learning across five sub-decisions (§8).

-

An industrial–academic bridge. We connect open methods to Kimi PARL, OpenAI Codex, and Anthropic Claude Code (§4), identify which design choices in these systems have—and have not—been published, and characterize the gap between publicly reported industrial deployment envelopes and open academic evaluation regimes in rollout cost and trace length (§5).

-

An open, tagged paper pool. We release an $84$-entry curated pool ($51$ focal LLM-MAS entries plus $33$ supporting references) with $18$-column taxonomy tags, synchronised with the paper bibliography and summarised as a single table in Appendix B. The broader artifact contains $116$ audited records when the exclusion log is included (Appendix C). It is intended as a reusable taxonomy substrate that follow-up work can extend without re-curating from scratch.

-

Scripted corpus statistics and trace schema. The artifact includes a statistics script, a static statistics snapshot, a machine-readable orchestration-trace JSON Schema, a valid example trace, and a dependency-free trace validator (§13). These make the sparsity claims and benchmark-reporting recommendations mechanically inspectable.

-

Entry cards. Appendix A gives one-card summaries for thirteen core methods, frameworks, and industrial anchors under a uniform template, suitable as a quick reference complementing the main taxonomies.

-

Open problems. We identify fifteen open problems (§11), organized along algorithmic, reward, systems, safety, and evaluation axes.

### 1.5  Roadmap

§1.3 defines the corpus and evidence levels. §2 gives the minimal MARL and agentic-RL background; §3 extends the Dec-POMDP to the dynamic-agent setting needed for the rest of the paper. §4 covers industrial and academic system forms; §5 quantifies the engineering constraints (rollout cost, harness boundary, trace-length dependence) that discipline algorithm choice. §6–§8 are the three pillars of the thesis: reward design, credit assignment, and orchestration learning. §9 argues that current benchmarks fail to measure the very properties (parallelism efficiency, collaboration quality, error amplification) that LLM-MAS RL is supposed to optimize. §11 lists fifteen open problems and §14 returns to the thesis. Appendices A–B contain the method cards and the complete paper-pool summary table.

## 2  Background: From MARL to LLM-MARL

This section gives the minimal background needed for the rest of the survey. We cover classical MARL (§2.1) and single-agent LLM RL (§2.2) compactly, then spend the rest of this section on what makes LLM-MAS genuinely different from either (§2.3).

### 2.1  Classical MARL in one page

A Markov game [33] generalizes an MDP to $n$ agents: each agent $i$ has an action space $\mathcal{A}_{i}$, observation space $\mathcal{O}_{i}$, and policy $\pi_{i}$; transitions are driven by the joint action $(a_{1},\ldots,a_{n})$ and yield per-agent rewards $r_{i}$ (cooperative, competitive, or mixed-motive). When observations are partial, the setting is a decentralized partially-observable MDP (Dec-POMDP) [5].

Two design choices organize most classical MARL algorithms:

-

Centralized training, decentralized execution (CTDE). A central critic that sees the joint $(s,a_{1},\ldots,a_{n})$ is used only during training; at deployment each agent runs on its own observation $o_{i}$. VDN [56], QMIX [50], MADDPG [39], and MAPPO [74] all live in this family.

-

Value decomposition vs. counterfactual baselines. VDN/QMIX decompose a team value function into per-agent contributions additively or monotonically. COMA [16] replaces that with a counterfactual baseline: agent $i$’s advantage is the difference between the team return and the return under a counterfactual where $i$’s action is marginalized. Shapley-value credit [59] generalizes this to a fair marginal-contribution attribution over all subsets of agents; difference rewards [66] are the closely-related earlier formulation.

Two practical algorithms recur in LLM-MAS papers: IPPO [11] (independent PPO per agent, no centralized critic) and MAPPO [74] (shared policy with centralized critic). Dr. MAS [15] is the most visible recent paper to reopen the IPPO-vs-MAPPO-vs-GRPO question in the LLM-MAS setting; its central observation is that GRPO’s group-normalized advantage, borrowed unchanged from single-agent reasoning RL, becomes unstable at the agent level without explicit agent-wise normalization.

### 2.2  Single-agent LLM RL in one page

Single-agent LLM RL has evolved rapidly: RLHF [46] (preference rewards from human labels) $\to$ RLAIF (preference rewards from AI judges) $\to$ RLVR (verifiable rewards against ground truth) $\to$ Reasoning RL [13] (o1-/R1-style long-CoT with GRPO) $\to$ Agentic RL [73] (multi-turn tool use, web browsing, code execution).

Two axes organize this progression. Along the reward axis, the signal shifts from sparse preference (one label per rollout) to dense verifiable (per-step check) to hybrid. Along the credit axis, the unit shifts from trajectory-level PPO to token-level GAE to step- or turn-level process rewards (PRM). By the time one reaches agentic RL, the policy already produces actions at three natural granularities—token, action, tool call—and credit must be assigned across all three. The multi-agent extension adds further granularities above the single-agent trajectory (agent, role, orchestrator), which is the subject of §7.

Two representative methods are load-bearing below. PPO [52] and GRPO [53] are the dominant policy-optimization choices: PPO uses a learned value baseline, GRPO normalizes advantages within a group of $K$ rollouts from the same prompt, eliminating the value network. GRPO’s simplicity makes it the default in most multi-agent papers in our pool, but as Dr. MAS [15] documents, its group-normalization is what needs to change at the multi-agent level.

### 2.3  Why LLM-MAS is not classical MARL

Seven differences separate LLM-MAS from the classical MARL setting in §2.1. Each has direct consequences for algorithm design in later sections.

-

Action space is natural language. A sub-agent’s action is a generated message, a tool invocation, or a sub-agent spawn. This makes the action space combinatorial and ill-defined for classical MARL machinery (VDN’s additive decomposition, MADDPG’s continuous-control assumptions).

-

Observation is long and partially summarized. An agent may see a conversation transcript of thousands of tokens, a tool-returned document, or a summarized report from another agent. Observation shape varies within and across episodes; this is why orchestration traces are graph-structured rather than sequence-structured (§8).

-

Number of agents is dynamic and learnable. Kimi K2.5 discloses PARL training of an orchestrator that can spawn up to $100$ sub-agents; K2.6 extends the reported deployment envelope to $300$ sub-agents. We use the latter as a scale-pressure signal, not as independent evidence of a new RL-training objective. In the disclosed K2.5 setting the count is the output of a learned policy, not a fixed hyperparameter. Classical MARL fixes $n$ and trains with $n$ fixed; Shapley credit over a dynamic agent set is still open (§7.4).

-

Communication is free-form. Classical MARL communication is typically a small discrete or continuous channel. In LLM-MAS every message is a natural-language utterance. This both widens the channel (agents can transmit plans, critiques, counterfactuals) and creates a new signal/credit-assignment unit (message-level signal or credit, §7.1).

-

Episode length is long and asynchronous. Thousands of steps, hours of wall-clock time, parallel sub-agent execution. Rollout cost dominates RL wall-clock (§8), and the slowest sub-agent gates the whole trace.

-

Agents are heterogeneous by role. Planner / executor / critic / verifier / summarizer. Role-based heterogeneity introduces role-level credit (MALT [44], M-GRPO [19]) that has no clean counterpart in homogeneous MARL.

-

Credit- and signal-bearing units are new. Beyond (state, action) and agent, LLM-MAS introduces message, tool call, role, and orchestrator-decision as credit- and signal-bearing units (§7). This is the single most important structural difference.

Takeaway. Classical MARL gives the language (Dec-POMDP, CTDE, COMA, Shapley). Single-agent LLM RL gives the algorithms (PPO, GRPO, verifiable reward, agentic rollouts). LLM-MAS adds new credit- and signal-bearing units that neither body of work handles natively, and that is what the rest of this survey is about.

## 3  A Working Abstraction for the Orchestration Trace

The background in §2 kept the formalism deliberately classical. The rest of this paper rests on a thesis that does not fit the classical mould: LLM multi-agent RL is usefully analyzed through an orchestration trace, a temporal interaction graph whose vertices are events (orchestrator decisions, sub-agent invocations, tool calls, messages, summary returns, aggregations) and whose vertex set itself is determined by the policy. This section fixes the vocabulary for that object and states two informal observations that are referenced throughout §6–§7.

Scope. Our intent here is a taxonomy formalism for the survey, not a fully axiomatized new MARL framework. We introduce the minimum formal vocabulary needed to make subsequent taxonomy claims unambiguous, and we flag the technical gaps (off-policy evaluation of unrealized branches, exact value-function forms over variable-shape graphs) that a follow-up theory paper would need to address. We do not define a new solution concept, establish equivalence to existing dynamic-agent MARL formalisms, or prove new convergence / identifiability results.

### 3.1  Relation to existing formalisms

The abstraction above is closest to four existing families of formalisms, but does not reduce cleanly to any one of them. Dec-POMDPs and Markov games [5, 33] provide the fixed-agent cooperative setting, but assume a fixed agent index set and a joint action at each time step; LLM-MAS orchestration must additionally represent spawn/despawn, delegation, and aggregation events. Hierarchical RL and options-style controllers supply the idea of a high-level policy choosing temporally extended actions, but they usually treat options as predefined action abstractions rather than as natural-language sub-agents that communicate, call tools, and return summaries. Dynamic-population or open multi-agent systems address the changing agent set, but typically abstract away the language/tool event graph that determines where reward and safety failures occur. Graph-conditioned MARL critics and communication learning handle graph-structured interaction, but generally assume the graph is observed or learned as a communication topology rather than produced by an orchestrator whose actions create and remove subgraphs.

Table 4 makes the boundary explicit. Our working abstraction occupies a narrower role than these formalisms: it fixes the event vocabulary needed by this survey. The objects that matter for later taxonomy are not only states and joint actions, but credit- and signal-bearing events: spawn, message, tool, return, and aggregation nodes whose existence depends on earlier orchestrator decisions. This is why we call the section a working abstraction rather than a new solution concept.

| Formalism | What it handles | What is missing for this survey | Role here |

| Dec-POMDP / Markov game | Fixed-agent decentralized control | Spawn/despawn, variable joint-action shape, language/tool events | Classical baseline. |

| Hierarchical RL / options | Meta-actions and temporally extended skills | Natural-language sub-agents that communicate, call tools, and return summaries | Analogy for orchestration actions. |

| Open / dynamic-population MAS | Changing agent populations | Trace-level message, tool, return, and aggregation events | Closest dynamic-agent precedent. |

| Graph-conditioned MARL critic | Graph-structured state or communication | Topology created by the orchestrator rather than only observed | Critic architecture precedent. |

$\mathcal{M}^{+}$ | This paper’s | Dynamic agents plus language/tool event graph | A new optimality theory, convergence result, or solution concept | Bookkeeping abstraction for taxonomy. |

*Table 4: Relation between the working abstraction in this paper and existing formalisms. The purpose of $\mathcal{M}^{+}$ is to make the taxonomy auditable over event graphs; it is not proposed as a new solution concept.*

### 3.2  Orchestration trace as a Dec-POMDP extension

Definition 1 (Dec-POMDP, recap). A decentralized POMDP [5] over a Markov game [33] is a tuple $\mathcal{M}=(\mathcal{I},\mathcal{S},\{\mathcal{A}_{i}\}_{i\in\mathcal{I}},P,\{\mathcal{O}_{i}\}_{i\in\mathcal{I}},\Omega,r,\gamma)$, where $\mathcal{I}=\{1,\ldots,n\}$ is a fixed set of agents, $\mathcal{S}$ is the state space, $P(s^{\prime}\mid s,\mathbf{a})$ is the transition kernel under joint action $\mathbf{a}=(a_{1},\ldots,a_{n})$, $\Omega(o_{i}\mid s,i)$ is the observation model, $r:\mathcal{S}\times\prod_{i}\mathcal{A}_{i}\to\mathbb{R}$ is a shared reward, and $\gamma\in[0,1)$ is the discount.

Definition 2 (Dynamic-Dec-POMDP). We extend $\mathcal{M}$ to accommodate spawn / despawn dynamics:

$\mathcal{M}^{+}\;=\;\bigl(\mathcal{I}_{t},\,\mathcal{S},\,\mathcal{A}(\cdot),\,\mathcal{A}_{\mathrm{spawn}},\,P,\,\Omega,\,r,\,\gamma\bigr),$ | | | | (1) |

where $\mathcal{I}_{t}\subseteq\mathbb{N}$ is a time-indexed agent set, $\mathcal{A}(i,r_{i},h_{t})$ is the action space available to agent instance $i$ under role $r_{i}$ and trace history $h_{t}$, $\mathcal{A}_{\mathrm{spawn}}$ is a discrete action space over spawn$(\text{role},\text{context})$ and despawn$(i)$ operations, and the global state is augmented with the current agent count $N(s_{t})=|\mathcal{I}_{t}|$. The orchestrator at time $t$ is a privileged agent whose action lies in $\mathcal{A}_{\mathrm{spawn}}$; sub-agents act in their own $\mathcal{A}(i,r_{i},h_{t})$. This notation is deliberately permissive: tool permissions, role prompts, memory access, and harness-imposed constraints can all change the action set without requiring a new agent identity.

Definition 3 (Orchestration trace as event graph). An orchestration trace produced by a rollout under $\mathcal{M}^{+}$ is a rooted, edge-labelled, vertex-labelled temporal graph

$G\;=\;(V,E,\ell_{V},\ell_{E}),$ | | | | (2) |

whose components are:

-

$V=V_{\mathrm{orch}}\cup V_{\mathrm{spawn}}\cup V_{\mathrm{msg}}\cup V_{\mathrm{tool}}\cup V_{\mathrm{ret}}\cup V_{\mathrm{agg}}$: a set of events—orchestrator decisions, sub-agent spawns, inter-agent messages, tool calls, summary returns, and aggregation steps;

-

$E\subseteq V\times V$: a set of temporal/causal dependency edges (e.g., a tool-call event depends on the orchestrator-decision event that authorized it);

-

$\ell_{V}:V\to(\text{agent},\text{role},\text{content})$: a vertex label assigning each event to an agent instance (drawn from $\mathcal{I}_{t}$ at the relevant $t$), a role, and structured content;

-

$\ell_{E}:E\to\{\texttt{spawn},\,\texttt{msg},\,\texttt{return},\,\texttt{aggregate}\}$: an edge-type label.

A classical Dec-POMDP trajectory corresponds to the special case $|V_{\mathrm{spawn}}|=0$, $|\mathcal{I}_{t}|\equiv n$, and the event sequence linearizes into the standard $(s_{0},a_{0},s_{1},a_{1},\ldots)$ form. Definition 3.2 is the same object used in the visual schematic of Figure 14 and the credit hierarchy of §7.1: events, not agents, are the carriers of credit.

### 3.3  Value function under variable-shape traces

In the fixed-$n$ setting the value of a joint policy $\boldsymbol{\pi}$ is $V^{\boldsymbol{\pi}}(s)=\mathbb{E}\bigl[\sum_{t\geq 0}\gamma^{t}r_{t}\mid s_{0}=s\bigr]$, and CTDE methods such as MADDPG [39] and MAPPO [74] learn $V^{\boldsymbol{\pi}}(s)$ or $Q^{\boldsymbol{\pi}}(s,\mathbf{a})$ over a fixed-shape joint. Under $\mathcal{M}^{+}$ neither $s$ nor $\mathbf{a}$ has a fixed dimension: the set of active sub-agents, and hence the shape of $\mathbf{a}$, is itself a random variable. We therefore parameterize the value object by the trace prefix:

$V^{\boldsymbol{\pi}}(G_{\leq t})\;=\;\mathbb{E}_{\boldsymbol{\pi}}\!\left[\sum_{\tau\geq t}\gamma^{\tau-t}r_{\tau}\;\Big|\;G_{\leq t}\right],$ | | | | (3) |

where $G_{\leq t}$ is the sub-graph of $G$ induced by events with timestamp $\leq t$. For orientation, the corresponding one-step bookkeeping identity can be written in graph-conditioned form,

$V^{\boldsymbol{\pi}}(G_{\leq t})\;=\;\mathbb{E}_{\boldsymbol{\pi}}\!\bigl[r_{t}+\gamma\,V^{\boldsymbol{\pi}}(G_{\leq t+1})\,\big|\,G_{\leq t}\bigr],$ | | | | (4) |

whose essential difference from the standard Bellman equation is that $G_{\leq t+1}$ may contain a larger vertex set than $G_{\leq t}$ (when a spawn event fires): the expectation is taken over transitions that grow or shrink the event graph, not only over transitions in a fixed joint state space. Equation (4) is a trace-conditioned value identity that motivates the graph- or trace-conditioned critics in §7; it is not a new solution concept or convergence claim. Concrete algorithmic forms (which sub-graph features are sufficient, how to amortize $V$ across variable-shape inputs) are open and are discussed in §11.

### 3.4  Two organizing observations

The remainder of this paper invokes two informal observations about $\mathcal{M}^{+}$. They are intended to organize the taxonomy, not to function as theorem statements: each is supported by the qualitative argument below and by the empirical evidence cited, but neither is accompanied by a formal proof or rate in this paper.

Observation 1 (Credit diffusion under uniform credit). Under a shared terminal team reward $R$, uniform credit allocation across $n$ credit- or signal-bearing units of an orchestration trace (e.g., tokenwise GAE with a single team baseline, as in naive GRPO on the concatenated trace), and no structure-specific baseline, the effective per-decision signal available to any single unit tends to become less distinguishable as trace length grows.

Argument. A uniform allocation distributes the terminal reward equally over the $n$ units; the shared baseline removes the mean. The remaining per-unit signal is dominated by the residual variance of $R$ shared among $n$ units. As $n$ grows with trace length, the per-unit signal becomes increasingly difficult to distinguish from baseline-estimation noise. Concrete order-of-magnitude scaling depends on the exact noise model (we leave a precise rate to a follow-up theory paper); what matters here is the qualitative fragility: longer traces with unstructured credit make it harder to identify the contribution of any individual decision. This is the same pathology that motivated COMA-style counterfactual baselines [16], Shapley credit [59], and difference rewards [66] in classical MARL, and that motivates the credit-decomposition methods surveyed in §7. Empirically, Dr. MAS [15] documents the resulting training instability when GRPO is applied naively to multi-agent rollouts.

Observation 2 (Non-identifiability of same-prefix counterfactual orchestrator credit). Let $\pi_{\mathrm{orch}}$ be the orchestrator policy and let $d_{t}\in\{\texttt{spawn},\texttt{no-spawn}\}$ be its decision at time $t$. Without an off-policy evaluation mechanism, the same-prefix counterfactual effect $\mathbb{E}[R\mid G_{\leq t},\texttt{spawn}]-\mathbb{E}[R\mid G_{\leq t},\texttt{no-spawn}]$ is not identifiable from realized on-policy traces alone unless both branches have coverage or additional structural assumptions are made.

Argument. A marginal association such as $\mathbb{E}[R\mid d_{t}]$ can be estimated from logged on-policy data, but it mixes different trace prefixes and therefore does not isolate the causal contribution of the decision at a fixed prefix. Classical $Q$-learning identifies $Q(s,a)$ only when every action $a$ is occasionally sampled at comparable states $s$. For an orchestrator’s spawn decision, the un-taken branch (no-spawn, when spawn was chosen) produces a structurally different trace—no sub-graph is generated for it—so on-policy rollouts furnish no realizations of the counterfactual. Estimating the same-prefix contribution of the spawn decision therefore requires either an explicit off-policy mechanism (e.g., a learned counterfactual value function trained on alternative-branch data) or strong structural assumptions. This is the conceptual anchor of the “counterfactual ambiguity” discussion in §7.3 and of the open problem flagged in §11.

### 3.5  The reward–credit dual

The two claims above motivate a trade-off that organizes the next three sections. Each reward family in §6 picks a privileged layer in the credit hierarchy: outcome rewards privilege the terminal-trajectory layer; process rewards privilege the step or turn layer; orchestration rewards deliver a dense signal directly at the orchestrator-decision event. The denser the reward, the smaller the effective number of units over which the diffusion in Claim 3.4 acts, and the less the credit decomposition in §7 must do to recover a usable per-unit signal. Conversely, sparse terminal rewards shift the burden onto credit assignment—counterfactual baselines, role-level critics, Shapley-style attribution—because the reward does not itself pick a layer. This duality explains why two apparently opposite design choices (process rewards with weak credit decomposition, vs. sparse outcome rewards with strong credit machinery) can both be empirically competitive: they place the same total work on different sides of the reward–credit ledger.

Takeaway. The orchestration trace $G=(V,E,\ell_{V},\ell_{E})$ is an event graph drawn from the dynamic-Dec-POMDP $\mathcal{M}^{+}$ of Definition 3.2. Value functions are naturally conditioned on $G$ rather than on a fixed-shape state; uniform credit on long shared-reward traces can make per-unit signal fragile (Claim 3.4); and orchestrator spawn decisions are non-identifiable from on-policy rollouts alone (Claim 3.4). Dense rewards and strong credit decomposition are duals: entries in §6–§7 differ in where they place that burden. The framework here is intended as an organizing abstraction for the survey, not a new MARL theory; concrete algorithmic forms are open (§11).

## 4  System Forms: How LLM Agent Teams Are Organized

Before algorithms, we fix the system object: the concrete topologies in which LLM agents are organized, both in open literature and in deployed products. This ordering matters for two reasons. First, every RL method in later sections optimizes some system form; the form determines what can be rewarded and where credit can flow. Second, public industrial systems expose engineering pressures (rollout cost, asynchrony, harness design) that discipline which RL methods are practical—a constraint missing from most academic benchmarks.

### 4.1  A typology of agent-team topologies

Six patterns recur across the paper pool (Table 5). They are not mutually exclusive; production systems typically combine two or three.

| Topology | Defining feature | Representative instances |

| Centralized orchestrator + sub-agents | One orchestrator dispatches tasks to a pool of sub-agents and aggregates results. | Kimi Agent Swarm [28], M-GRPO main/sub [19], Puppeteer [10], WideSeek-R1 [68] |

| Planner–executor–critic | Three specialized roles with distinct rubrics; critic closes the loop. | MALT [44], MATPO [43], MAE [8] |

| Debate / committee | Multiple agents argue, a resolver decides; credit is message-level. | Debate-as-Reward [51], LatentMAS [84] |

| Parallel swarm | Many near-homogeneous agents run concurrently, then aggregate. | Kimi PARL [28, 29], Anthropic parallel Claudes [2] |

$k$$(k{+}1)$ | Hierarchical agents | Multi-level spawn; agents at level can spawn level-. | HALO [20], AgentSpawn [9], DEPART [21], LAMO [62] |

| Managed / harness-based | System harness wraps model, tools, prompts, execution; agents live inside. | OpenAI Codex [45], Claude Code [3], Agent Lightning [40] |

*Table 5: Six recurring agent-team topologies. Every method in our pool fits one or more; the topology constrains which reward families (§6) and credit levels (§7) are even definable.*

*Figure 5: Visual schematics of the six recurring LLM-MAS topologies catalogued in Table 5. Red ($\bigcirc$/box) = orchestrator or planner; blue ($\bigcirc$) = sub-agent or executor; orange (box) = critic; green diamond = debate resolver; dashed outer box = managed harness. Solid arrows = delegation; dashed arrows = voting / aggregation; double-headed arrows = bidirectional debate. The topology constrains which credit-bearing units (§7.1) are easiest to define: (a)/(d)/(e) make orchestrator-level credit most natural; (c) makes message-level credit a natural primary signal, although logged messages can be credited in other topologies; only (f) admits a harness-level boundary as a training-frozen interface.*

Two observations.

-

Topology determines credit affordance. Centralized orchestrator topologies make orchestrator-level credit easiest to define; debate topologies make message-level credit a natural primary signal. Methods that target these levels necessarily commit to a compatible topology.

-

Harness-based systems are a research-opaque majority. Codex and Claude Code are in our pool as cases, not as methods, because their RL training recipes are not publicly disclosed. What is disclosed is the harness: model $\oplus$ tools $\oplus$ prompts $\oplus$ execution logic. Any RL method that targets these systems must respect harness shape, even if the harness is not itself being trained.

### 4.2  Public industrial evidence and selection rule

The industrial discussion is not intended as a census of all deployed agent products. We retain only public industrial sources that satisfy at least one load-bearing criterion for this paper: they disclose a trained orchestration mechanism, expose a stable harness or sub-agent interface that constrains future RL design, or document long-running parallel workflows at a scale not represented in open academic benchmarks. Under this rule, the retained industrial anchors are Kimi Agent Swarm, OpenAI Codex, and Anthropic Claude Code / parallel-Claude engineering reports. They are representative of three evidence roles rather than of the whole commercial agent market: Kimi provides the public trained-orchestrator anchor; Codex provides the cloud harness and parallel software-agent workflow anchor; Claude Code provides the sub-agent interface and parallel-team workflow anchor.

Several well-known frameworks and products are therefore not treated as main industrial anchors. AutoGen, CAMEL, MetaGPT, CrewAI, LangGraph, Devin-related product material, and OpenAI Swarm are valuable context, but the screened public records either do not disclose RL/post-training mechanisms, do not provide enough stable technical detail for the evidence ledger, or are better used as background framework examples. Appendix C logs these borderline decisions. This selection rule is deliberately conservative: it reduces coverage breadth in exchange for a clearer claim boundary.

Table 6 records how we use industrial materials. This is important because blogs and documentation can support claims about deployment shape, scale, interfaces, and user workflow, but they do not by themselves make an algorithm reproducible.

| Source class | Examples | Used for | Not used for |

| Peer-reviewed / arXiv methods | MALT, MAPoRL, Puppeteer, Dr. MAS | Algorithmic mechanisms, training regimes, reported ablations | Claims beyond the paper’s evaluation setting |

| Company technical reports | Kimi K2.5 / PARL | Publicly disclosed training shapes, reward components, system scale | Full reproducibility when optimizer, data, or ablations are absent |

| Product docs / launch blogs | Codex, Claude Code, Kimi K2.6, Claw Groups | Deployment form, harness boundary, scale, user-facing affordances | Undisclosed RL objectives or optimizer details |

| Engineering case studies | Anthropic parallel Claudes C-compiler case | Long-running agent-team workflow, cost, harness design pressure | General claims about model training or benchmark superiority |

*Table 6: Evidence levels used for industrial systems. We distinguish deployment-shape evidence from reproducible algorithmic evidence; this prevents product documentation from being treated as equivalent to a peer-reviewed RL method.*

The more compact source-class matrix used during screening is moved to Appendix C; Table 6 and Table 7 are the load-bearing evidence controls in the main text.

| Claim used here | Primary source | Confidence | Boundary |

| Kimi K2.5 trains an orchestrator with PARL | Kimi technical report | high | Used as the only public industrial anchor in our pool that explicitly discloses RL training of the orchestrator. |

$100$$1{,}500$ | Kimi K2.5 reports up to sub-agents and coordinated steps / tool calls | Kimi technical report | high | Used for publicly reported deployment-envelope evidence; full optimizer/data/ablation details are not reproducible. |

$300$$4{,}000$ | Kimi K2.6 reports up to sub-agents and coordinated steps | official Kimi blog / product material | medium-high | Used as deployment-envelope evidence; not used as a reproducible training claim. |

| OpenAI Codex exposes parallel software-agent workflows and a harness boundary | OpenAI product material | high | Used as deployment-shape evidence; we do not claim a public multi-agent RL objective. |

| Claude Code exposes sub-agents and custom sub-agent interfaces | Anthropic documentation | high | Used as harness and spawn/delegation evidence; not as evidence of RL-trained orchestration. |

| Anthropic C-compiler project used parallel Claude Code sessions | Anthropic engineering case study | medium | Used as workflow-shape and cost-pressure evidence; not as a model-training result. |

*Table 7: Claim-confidence ledger for industrial evidence. This table makes explicit which claims are supported by public material and which claims are intentionally not made.*

#### 4.2.1  Kimi Agent Swarm (K2.5 / K2.6)

Moonshot’s Kimi K2.5 is the most openly documented industrial instance of trained orchestration in our pool. The K2.5 report introduces Parallel-Agent Reinforcement Learning (PARL) and describes a swarm scaling to up to $100$ sub-agents and $1{,}500$ coordinated steps / tool calls as reported [28]. The K2.6 product and technical materials scale the deployment envelope to $300$ sub-agents and $4{,}000$ coordinated steps and add Claw Groups, a research preview of cross-vendor and human-in-the-loop coordination [29].

What makes Kimi the main industrial reference point under this evidence boundary:

-

The orchestrator is a learned policy, not a prompt template. Sub-agent creation is an action in its action space.

-

The reward decomposes as $r_{\text{perf}}+\lambda_{1}r_{\text{parallel}}+\lambda_{2}r_{\text{finish}}$ (§6.2)—a published instance of an R7+R8 composition with staged annealing.

-

The “Critical-Steps” metric functions as an orchestrator-level credit signal (§7): it distinguishes real parallel progress from padded traces, penalizing pseudo-parallelism at the orchestrator level.

Kimi is therefore the industrial anchor for our thesis, but with an important evidence boundary: the public materials disclose enough to identify learned orchestration, reward shaping, and orchestrator-level signals, but not enough to reproduce the full training recipe. More importantly, the scale-gap argument is anchored by Kimi rather than established uniformly across industrial systems; Codex, Claude Code, and related public systems are evidence for harness and workflow shape, not for comparable disclosed RL trace scale.

#### 4.2.2  OpenAI Codex (app + harness)

Codex [45] is described in OpenAI’s launch material as a cloud-native parallel software-engineering agent orchestrated from a single “command center.” Two features matter for RL.

First, the harness—model $\oplus$ tools $\oplus$ prompts $\oplus$ execution logic—is the unit of deployment, not the model alone. Any RL training for a harness-hosted agent must treat the harness boundary as fixed during training, and as part of the observation and action interface at inference. Agent Lightning’s execution/training decoupling [40] is, in effect, an academic articulation of this constraint.

Second, the Codex UI makes parallel workflows and long-running tasks first-class. This is significant because long-horizon parallel rollouts are the most expensive rollouts to train on: a single rollout can be minutes of wall clock and hundreds of tool calls. RL algorithms that require many rollouts (GRPO in particular, with its group of $K$) become impractical without the kind of pipeline-parallel scheduling that MarsRL [36] targets.

Codex’s RL training recipe is not publicly disclosed; we cite it here as a system-design anchor, not as an algorithmic data point.

#### 4.2.3  Anthropic Claude Code (sub-agents + agent teams)

Claude Code documentation [3] specifies built-in sub-agents (Explore, Plan, general-purpose) and a user-facing API for custom sub-agents, with a lead agent that dispatches subtasks and aggregates results. Anthropic’s engineering case study of sixteen parallel Claudes jointly building a C compiler over roughly $2{,}000$ sessions and $\sim\!100{,}000$ lines of Rust [2] is the largest public multi-agent code-generation case study we are aware of.

Two aspects of Claude Code are load-bearing for the rest of this paper.

-

Sub-agent as a first-class object. Claude Code’s sub-agent API makes the spawn / delegate / aggregate pattern a concrete object that can be the subject of RL, not just a prompt-engineering convention.

-

Explicit steerability concern. Anthropic’s trustworthy-agents framework [4] explicitly flags that sub-agents make it harder for users to understand and steer a workflow mid-execution. This is a credit-assignment-shaped concern: where in the trace can the human intervene, and what are the downstream consequences? We return to this in §11.

### 4.3  What systems reveal that papers do not

Reading across the three systems against the paper pool surfaces three gaps between what is deployed and what is published.

*Figure 6: Industry–academia scale gap. Reading: blue points summarize the typical public evaluation regime of academic LLM-MAS RL methods, while red filled points mark Kimi reports that disclose both team size and long trace length. Hollow red points indicate industrial deployment-shape evidence where the public material is useful for harness and workflow analysis but does not disclose a comparable RL training scale. Positions are approximate and log-scaled; the figure is intended to show the regime gap, not a leaderboard.*

| Point in Fig. 6 | Public scale signal | Source status |

$10$$100$ | Academic RL methods | mostly –-step traces; small-to-moderate teams | Representative operating range read from public evaluations in the retained pool. WideSeek-R1 and MARTI-MARS2 begin to stress width/self-search scaling, but remain below the disclosed Kimi envelope; this row is a regime summary, not a leaderboard. |

$10$$100$ | Agent Lightning / harness frameworks | –-step traces; small teams | Public material is most useful for the training-harness boundary; comparable Kimi-reported training traces are not disclosed. |

$1{,}500$$100$ | Kimi K2.5 | coordinated steps / tool calls; up to sub-agents | Public company report used as the disclosed deployment-envelope anchor [28]. |

$4{,}000$$300$ | Kimi K2.6 | coordinated steps; up to sub-agents | Public company report used as the disclosed deployment-envelope extension [29]. |

*Table 8: Source status for the scale-gap figure. The figure compares operating regimes disclosed in public material; it does not claim that all industrial systems train with Kimi-reported RL traces.*

-

Rollout-cost realism. Academic methods mostly evaluate on $10$–$100$-step orchestration traces; newer width-scaling and self-search methods such as WideSeek-R1 [68] and MARTI-MARS2 [61] narrow the conceptual gap but do not disclose Kimi-reported training traces. Public Kimi reports disclose $1{,}500$–$4{,}000$-step traces, while other industrial reports expose the harness and workflow pressures without comparable training-scale disclosure. Credit diffusion (§7.3) becomes qualitatively worse in this regime, and within our curated pool no open academic method publicly reports training at the Kimi-reported trace lengths.

-

Harness as fixed context. All three industrial systems expose a stable harness boundary; for any RL or post-training layer, that boundary would constrain the learnable policy. The academic literature mostly does not make this boundary explicit; methods are typically trained in a bespoke rollout environment. Agent Lightning [40] is the clearest counterexample.

-

Steerability as an RL target. Industrial safety frameworks [4] treat mid-execution human intervention as a first-class concern. We found no paper in our pool that formulates steerability as an RL objective; this is a named open problem in §11.

Takeaway. Topology determines which rewards and credit assignments are even definable; public industrial evidence, most clearly the Kimi reports, discloses deployment envelopes and harness constraints beyond most open academic evaluation regimes. The gap is not primarily algorithmic—it is in what shape of rollout and harness academic methods train against.

## 5  Systems Engineering: Rollout Cost and Harness Boundary

The system forms cataloged in §4 and the event-graph formalism of §3 imply concrete engineering constraints that academic RL methods cannot ignore. Like §3, the material in this section is not an independent theory contribution: it supplies the operational back-pressure (rollout cost, harness shape, trace length) that disciplines which taxonomy cells in §6–§8 are actually reachable at industrial scale. This section discusses three such constraints—rollout cost (§5.1), the harness boundary (§5.2), and trace-length dependence (§5.3)—and relates each to a design choice that recurs in the paper pool.

### 5.1  Rollout cost dominates wall-clock training time

A single-agent RL rollout for reasoning-level tasks is typically $10^{2}$–$10^{3}$ tokens and one or two tool calls. A multi-agent rollout at industrial scale is substantially more expensive. For a back-of-envelope estimate in a centralized-orchestrator topology (§4), let the orchestrator spawn $K$ sub-agents. The $i$-th sub-agent consumes $L_{i}$ context/output tokens and issues $T_{i}$ tool calls. Assuming per-token inference cost $c_{\text{tok}}$ and per-tool latency $c_{\text{tool}}$, the expected rollout cost is

$C_{\text{rollout}}(G)\;\approx\;\sum_{i=1}^{K}\bigl(L_{i}c_{\text{tok}}+T_{i}c_{\text{tool}}\bigr)\;+\;C_{\text{orch}}(K,|G|),$ | | | | (5) |

where $G$ is the orchestration trace, $|G|$ is its event count, and $C_{\text{orch}}$ is the orchestrator’s own inference and aggregation cost. We write

$T_{\text{total}}\;=\;\sum_{i=1}^{K}T_{i}$ | | | | (6) |

to avoid double-counting: if a public report gives total coordinated steps or total tool calls, that number should be used as $T_{\text{total}}$, not multiplied again by $K$. Under a simple per-event proxy, substituting Kimi K2.6’s reported operating point ($K{=}300$ and roughly $4{,}000$ coordinated steps / tool calls) [29] yields a rollout that can be one to several orders larger than a short single-agent reasoning rollout, depending on token lengths and tool latencies. This schematic proxy is before the RL-standard group-of-$G$ multiplier: GRPO-style training with $G{=}8$ rollouts per prompt multiplies rollout collection by a further $8\times$.

| Regime / entry | Team size | Trace length / calls | Source type | How used |

$K{=}1$ $10$$10^{2}$ | Single-agent reasoning baseline | | – reasoning/tool steps | modelling baseline | Normalization point for the cost schematic; not a corpus claim. |

$10^{1}$$10^{2}$ | Academic LLM-MAS RL entries | small-to-moderate teams | usually –-scale traces in reported evaluations | arXiv / conference papers | Representative regime for retained methods; WideSeek-R1 and MARTI-MARS2 add width/self-search scaling evidence but remain below the disclosed Kimi envelope. |

$100$ $1{,}500$ | Kimi K2.5 Agent Swarm | up to sub-agents | up to coordinated steps / tool calls as reported | company technical report | Published-training anchor for learned orchestration and scale-gap evidence. |

$300$ $4{,}000$ | Kimi K2.6 / Claw Groups | up to sub-agents | coordinated steps as reported | official product / technical blog | Deployment-scale evidence; not a fully reproducible training recipe. |

$16$ $2{,}000$ | Anthropic C-compiler case | parallel Claudes | roughly sessions in a long eng. project | case study | Workflow-shape and harness-pressure evidence, not RL-training evidence. |

*Table 9: Scale and rollout evidence used for Figure 7. The table separates extracted public quantities from modelling assumptions; aggregate academic ranges are representative of the retained method pool and should not be read as a leaderboard.*

*Figure 7: Rollout cost across representative operating regimes, shown as a schematic relative-cost proxy rather than a calibrated dollar or latency estimate. The bars combine representative team size and total trace length / tool-call counts under the cost form in (5); exact ratios depend on token lengths, tool latencies, and harness overhead. The group-of-$G$ annotation shows the additional rollout-collection multiplier imposed by GRPO-style training. The visible gap between academic and industrial regimes is what disciplines the engineering interventions surveyed in §5.1: pipeline parallelism, execution–training decoupling, and context folding all target the cost axis directly.*

The consequences for algorithm design are blunt. Methods that require large $G$ are impractical at industrial rollout cost without engineering interventions. Three such interventions recur in our pool. MarsRL [36] uses agentic pipeline parallelism: different stages of different rollouts execute concurrently, amortizing sub-agent idle time. Agent Lightning [40] introduces execution–training decoupling: an inference harness produces rollouts asynchronously into a buffer consumed by the trainer, removing wall-clock coupling between rollout completion and gradient steps. Context-Folding [55] compresses sub-trajectories back into the main trace, reducing the effective $L$ at aggregation time. WideSeek-R1 [68] and MARTI-MARS2 [61] add a complementary pressure: they explicitly scale parallel sub-agents or multi-agent self-search, so their engineering bottleneck is not only depth but also the width of the rollout graph.

### 5.2  The harness boundary as a training-frozen interface

In production systems (OpenAI Codex, Anthropic Claude Code) the harness—the shell of model, tool registry, system prompt, and execution runtime—is the actual unit of deployment (§4.2.2–4.2.3). The model parameters $\theta$ are one component; the harness specifies the interface through which $\theta$ is accessed and the set of actions $\theta$ can issue. RL that targets a harness-hosted agent must therefore obey a constraint that most academic methods do not:

$\pi_{\theta}(\cdot\mid o)\;=\;\text{LLM}_{\theta}(\cdot\mid\mathrm{harness}(o)),\quad a\in\mathcal{A}_{\text{harness}},$ | | | | (7) |

where $\mathrm{harness}(\cdot)$ is a fixed-at-training-time map from raw observation to prompt, and $\mathcal{A}_{\text{harness}}$ is the (typically small, finite) set of tools / sub-agent-spawn verbs the harness exposes. Equation (7) says that the harness defines both the input distribution the policy sees and the output grammar it may emit; fine-tuning through a different harness produces a different policy in the relevant operational sense.

*Figure 8: The harness as a training-frozen interface. The harness (dashed box) wraps the trainable LLM $\pi_{\theta}$ with a prompt template, tool registry, and execution runtime; only $\theta$ receives gradients during RL. The harness defines both the input distribution $\mathrm{harness}(o)$ that the policy sees and the output grammar $\mathcal{A}_{\text{harness}}$ it may emit. A policy fine-tuned through a different harness is a different policy in the deployment sense. In our pool only Agent Lightning [40] formulates this boundary explicitly as a contract the trainer must respect.*

Only Agent Lightning [40] formulates the harness boundary explicitly as an RL-training contract in our pool; most methods train in bespoke environments and then deploy into a different runtime. This is the most under-addressed gap between academic RL and industrial deployment; we return to it in §11.

### 5.3  Trace length and credit fragility

Section 7.3 observed qualitatively, and §3.4 argued via Observation 3.4, that the effective per-decision signal under shared-reward uniform credit decreases with trace length. This section records the same effect from an engineering standpoint.

Observation (credit fragility with trace length). Consider a shared-reward orchestration trace of length $T$ decisions with terminal reward $R\in\{0,1\}$ under uniform credit assignment ($A_{t}=R-\bar{R}$ at every step). The per-decision advantage is bounded by the spread of $R$ about its mean, while the expected per-decision marginal contribution shrinks as the reward is smeared across more and more decisions. Consequently, under these assumptions, the following heuristic signal-to-noise proxy at any single decision can become harder to estimate as the trace length grows:

$\mathrm{SNR}(t)\;\equiv\;\frac{\big|\mathbb{E}[A_{t}\mid s_{t},a_{t}]\big|}{\sqrt{\mathrm{Var}[A_{t}]}}\quad\text{is a fragile proxy as $T$ increases.}$ | | | | (8) |

We deliberately avoid a precise rate in Equation (8): the exact dependence on $T$ is determined by the task-specific noise structure of $R$ and the baseline estimator, and giving a closed form would require assumptions beyond the scope of this survey (§3.4). The qualitative failure mode suffices to explain the empirical failure mode: Dr. MAS [15] documents that naïve GRPO becomes unstable at multi-agent scale under exactly this regime. Methods that target role-level (MALT [44]), message-level (C3 [7]), or orchestrator-level (Puppeteer [10]) credit effectively shrink the relevant $T$ for their decomposed sub-problem—which is the reward–credit dual of §3.5 viewed from the engineering side.

*Figure 9: Schematic per-step signal-to-noise under three credit schemes as trace length $T$ grows. The blue curve is not a proven rate; it visualizes the qualitative warning in (8): uniform terminal credit can become low-SNR on long shared-reward traces. Role- or message-level decomposition (dashed green) partitions the trace into shorter sub-problems; a learned orchestrator critic (dotted red) targets a smaller set of orchestrator decisions. Curves and thresholds are illustrative, not fitted empirical laws.*

The practical implication: the right value of $T$ at which to benchmark an LLM-MAS RL method is the $T$ at which the method will be deployed. Since academic pool entries are mostly trained at $T\lesssim 10^{2}$ while the Kimi-reported deployment envelope reaches $T\sim 10^{3}$–$10^{4}$ and other public industrial systems mainly expose harness pressure rather than comparable training-scale traces, current academic results may systematically overestimate credit-assignment effectiveness at deployment scale—the opposite of the usual generalization story.

Takeaway. Three engineering constraints discipline LLM-MAS RL in ways classical MARL does not: rollout cost scales as $\sum_{i}(L_{i}c_{\text{tok}}+T_{i}c_{\text{tool}})+C_{\text{orch}}(K,|G|)$ and makes large-$G$ algorithms infeasible at industrial scale; the harness is a training-frozen interface that most academic methods ignore; and per-decision signal-to-noise under uniform credit decreases with trace length, with the decrease most relevant at the long-horizon $T$ publicly reported by Kimi. Each constraint points to a specific open problem (§11).

## 6  Reward Design for LLM-based MAS

Reward design is the first choice any LLM-MAS RL practitioner must make, and it is upstream of credit assignment: what cannot be measured as a reward cannot be assigned as credit. This section surveys the design space along eight families and grounds each in representative entries from the paper pool.

### 6.1  Eight families of rewards

Table 10 organizes the design space along five axes: what signal the reward captures, at what granularity it is emitted, where the signal comes from, its dominant hacking risk, and representative entries. Rows follow the order in which a practitioner typically encounters them: start from a shared outcome (R1), then decompose per-agent (R2–R3), then densify with process signals (R4–R6), then add system-level incentives (R7), then combine (R8).

| ID | Family | Granularity | Source | Dominant hacking risk | Representative methods |

| R1 | Shared team / outcome | team (terminal) | verifier / ground truth | reward diffusion; free-riding | MAGRPO [37], MAPoRL [47], Dr. MAS [15], CoLLM-MAAC [38] |

| R2 | Individual agent | per-agent (terminal) | per-agent outcome | credit overfits solvable sub-tasks; lazy-agent | MARFT [32], Context-Folding [55] |

| R3 | Role-specific | per-role (terminal or per-turn) | role-specific rubric | rubric mismatch across roles | MALT [44], MATPO [43], LAMO [62], DEPART [21] |

| R4 | Process (PRM) | per-step / per-turn | trained PRM or heuristic | step-padding; PRM gaming | MALT role-PRM [44], MarsRL [36] |

| R5 | Tool-use | per-tool-call | tool execution signal | tool-spam; fabricated tool success | MATPO [43], Agent Lightning [40] |

| R6 | Debate / verifier | per-message / per-turn | LLM judge or debate resolution | verifier collusion; over-communication | Debate-as-Reward [51], MAE [8], MAGIC [65], CriticLean [48] |

| R7 | Orchestration | per-orchestrator-decision | system metrics (speedup, finish-rate) | pseudo-parallelism; reward-shape collapse | Kimi PARL [28], Puppeteer [10], ParaManager [76], WideSeek-R1 [68] |

| R8 | Hybrid local–global | mixed | weighted composition of R1–R7 | weight drift; signal drowning | SHARP [31], M-GRPO [19], HERA [30], LangMARL [71], Agent Q-Mix [23] |

*Table 10: Eight reward families for LLM-MAS RL. Each row names a distinct signal a practitioner can attach to a multi-agent rollout; rows are not mutually exclusive and are commonly combined through R8.*

Three observations follow from the table.

-

Terminal $\to$ process is a densification axis. R1–R3 emit one number at episode end; R4–R6 emit signals throughout the trace. The latter give stronger gradients but introduce new attack surfaces (PRM gaming, judge collusion).

-

R7 is newly central in LLM-MAS. Orchestration rewards have single-agent analogues in compute, tool-cost, and process shaping, but no close analogue for spawn / delegate / aggregate decisions over multiple agent instances. They reward system-level properties (wall-clock speedup, split correctness, finish-rate), not task-level correctness. This is where LLM-MAS RL departs most sharply from agentic RL.

-

R8 is the default in practice. The larger-scale or practically oriented entries we emphasize (Kimi PARL, M-GRPO, Context-Folding, SHARP, LangMARL, Agent Q-Mix, MARSHAL [75], and DEPART [21]) use an R8 composition rather than a single family. Figure 10 makes the composition pattern explicit. The open question is not which family to pick but how to weight them without one drowning the others—a point we return to in §11.

*Figure 10: Reward family composition. The seven primitive families R1–R7 (§6.1) group into four semantic tiers—outcome, structured, process, system—and are composed through an R8 hybrid weighting to produce method-specific reward shapes. Three representative compositions from our pool are shown on the right. The less-studied axis is schedule semantics: which terms are transient scaffolds, which terms define the primary objective, and which schedules are disclosed.*

### 6.2  The Kimi PARL reward decomposition (worked example)

Kimi’s PARL [28] is the clearest published instance of an R7+R8 composition and serves as our canonical worked example. Following the evidence convention in Table 6, we use Kimi here as a company-report anchor: the public material discloses the reward components and deployment scale, but not enough optimizer, data, and ablation detail to make PARL independently reproducible. The orchestrator reward takes the form

$r_{\text{orch}}\;=\;r_{\text{perf}}\;+\;\lambda_{1}\,r_{\text{parallel}}\;+\;\lambda_{2}\,r_{\text{finish}},$ | | | | (9) |

where $r_{\text{perf}}$ is the downstream task outcome (R1), $r_{\text{parallel}}$ rewards genuine speedup over a serial baseline (R7), and $r_{\text{finish}}$ rewards all spawned sub-agents reaching termination (R7, a shape against pseudo-parallelism). Crucially, the public Kimi K2.5 description states that the hyperparameters for both auxiliary rewards are annealed to zero over training so that the final policy optimizes the primary task objective. Early in training these terms scaffold exploration of parallel scheduling; late in training they are removed so the orchestrator cannot farm auxiliary metrics by over-spawning or padding parallel work. Figure 11 illustrates the schematic shape. This staged annealing is, to our knowledge, the clearest explicit acknowledgement within our curated pool that R7 rewards are inherently transient scaffolds—useful to escape the zero-gradient region where the orchestrator has not yet learned to spawn, but harmful at convergence.

*Figure 11: Schematic of Kimi PARL’s three-term reward $r_{\text{orch}}=r_{\text{perf}}+\lambda_{1}r_{\text{parallel}}+\lambda_{2}r_{\text{finish}}$ across training (§6.2). The task-outcome term $r_{\text{perf}}$ is the primary objective; both auxiliary orchestration-shaping terms are shown as transient scaffolds because the public Kimi K2.5 report states that their hyperparameters are annealed to zero over training. Curves are schematic; exact schedules are not disclosed in the public PARL report.*

### 6.3  Reward-hacking failure modes specific to MAS

Five failure modes recur across the pool. Each maps to one or more reward families, and each has been reported (or, in at least two cases, directly measured) in published work.

-

Pseudo-parallelism (R7). The orchestrator spawns sub-agents that do no useful work in order to maximize a naïve parallelism bonus. Mitigated in Kimi PARL by the $r_{\text{finish}}$ shape and by a Critical-Steps metric that distinguishes real parallel progress from padded traces.

-

Free-riding / lazy agent (R1). Under shared reward, one sub-agent contributes negligibly but absorbs equal credit. This is the direct LLM-MAS analogue of the lazy-agent problem in classical MARL. SHARP [31] targets this via Shapley marginal credit; Dr. MAS [15] targets the related gradient pathology via agent-wise normalization.

-

Communication padding (R6). When a judge or PRM scores messages, policies inflate message length or verbosity to farm partial credit. Observed in debate-style setups and implicated in Debate-as-Reward’s [51] design of resolution-based (not length-based) rewards.

-

Tool-spam (R5). When tool-call success is rewarded, policies call many redundant tools. MATPO [43] and Agent Lightning [40] handle this by conditioning tool reward on downstream task outcome rather than call-level success alone.

-

Verifier collusion (R6). When the verifier is an LLM from the same family as the policy, both drift together and the verifier reward becomes uninformative. Mitigations in the pool are mostly diagnostic rather than algorithmic—dedicated critic benchmarks [78, 79] surface when judges and policies have drifted together but do not by themselves prevent collusion. CriticLean [48] is one of the few methods to train a critic explicitly via RL on a formal verification signal (Lean 4 type-checking), grounding the critic in something other than another LLM’s preference; this remains open more broadly (§11).

### 6.4  Discussion: the unanswered weighting question

Three threads run through the families above and converge on a single question we cannot yet answer.

Densification appears to trade signal-to-noise for attack surface. Moving along the R1$\to$R4$\to$R6 axis adds gradient signal at every step, but each new family is a new hackable sub-problem. A reward composed of $r_{\text{perf}}+r_{\text{PRM}}+r_{\text{tool}}+r_{\text{judge}}$ gives the policy four levers it can pull instead of one, and only $r_{\text{perf}}$ is anchored in ground truth. The empirical pattern in our pool is suggestive rather than conclusive: methods with rich shaping (MALT [44], SHARP [31]) typically report smaller gains over their own baselines than methods with sparse shaping (Dr. MAS [15], Puppeteer [10]) report over theirs. One possible explanation is that richer shaping creates more opportunities for PRM or judge gaming, but direct comparison is impossible (§9.3); we therefore treat this as a hypothesis generated by the survey rather than as an established empirical law.

Interaction-derived reward is becoming a fourth route to self-evolution. The coverage audit added several entries that do not fit cleanly into the older outcome/process/tool/verifier split. CoMAS [69] constructs rewards from inter-agent discussion dynamics; SiriuS [81] and Multiagent Finetuning [54] turn successful or majority-supported interaction traces into reusable experience or fine-tuning data; MAS-Zero [26] uses meta-level feedback to refine MAS designs without outcome supervision. We tag these as R4/R6/R8-adjacent rather than as a new primitive family, because the reward signal is still mediated by process, verifier, or aggregation mechanisms. They nevertheless mark a distinct trend: the team interaction itself is increasingly used to manufacture the training signal.

R7 weights are not constants—and we have no theory of their schedule. Section 6.2 described $\lambda_{2}$ in Kimi PARL as “annealed toward zero,” which is correct but uninformative: when, by how much, on what schedule? Any orchestration-reward shape that successfully scaffolds spawning early is, by construction, also a shape the orchestrator will exploit at convergence—a multi-agent analogue of the potential-based shaping pathology in single-agent RL. A principled R7 schedule—ideally derived from a measurable convergence indicator rather than from training-step count—is, to our knowledge, absent from our pool.

The composition is left to the practitioner. Every R8 weighting in our pool is hand-tuned. We found no retained entry with an auto-balancing mechanism that adjusts $\lambda_{k}$ during training based on observed gradient magnitudes or reward-component variance. This is a clear and tractable target for follow-up work, particularly given that classical RL has well-understood gradient-balancing heuristics (PCGrad-style methods, GradNorm) that have not yet been adapted to the MAS-specific setting of agent-shared losses with per-component noise structures.

Takeaway. Reward design in LLM-MAS is not a choice among eight families but a weighting over R8 compositions of them. The family that becomes newly central relative to single-agent LLM RL is R7—orchestration reward—and its defining feature in the clearest public industrial example is that auxiliary orchestration-shaping weights are not constant: in Kimi K2.5, they scaffold early training and are reported as annealed to zero as training progresses. The composition weighting itself is currently hand-tuned in the entries we reviewed; auto-balancing R8 weights against measurable training diagnostics is a clear near-term target.

## 7  Credit Assignment in LLM-based MAS

Credit assignment is where LLM-MAS departs most sharply from both single-agent LLM RL and classical MARL. Single-agent RL must propagate credit backwards in time (token $\to$ step $\to$ trajectory). Classical MARL adds a spatial dimension (which agent). LLM-MAS adds a structural dimension: credit must flow through roles, messages, tool calls, and—uniquely—through the orchestrator’s decisions about whether and how to spawn agents in the first place. This section makes that structural dimension explicit and organizes the paper pool around it.

### 7.1  The credit- and signal-bearing unit hierarchy

We argue that a single final reward in an LLM-MAS trace has eight plausible units to which reward, credit, or design signals can attach, each one finer-grained than the next:

$\begin{array}[]{c}\underbrace{\text{team}}_{\text{outcome}}\;\to\;\underbrace{\text{orchestrator}}_{\text{spawn/delegate}}\;\to\;\underbrace{\text{role}}_{\text{planner/critic/exec}}\;\to\;\underbrace{\text{agent}}_{\text{which sub-agent}}\\[6.0pt]
\;\to\;\underbrace{\text{turn}}_{\text{which round}}\;\to\;\underbrace{\text{message}}_{\text{which utterance}}\;\to\;\underbrace{\text{tool}}_{\text{which call}}\;\to\;\underbrace{\text{token}}_{\text{which span}}\end{array}$ | | | |

The team and agent units are inherited from cooperative MARL; role credit has partial analogues in heterogeneous-agent MARL; and orchestrator-decision credit is the least classical because the decision changes the future agent set itself. The lower four units (turn, message, tool, token) correspond to the temporal decomposition within any one agent’s trajectory, but become more consequential when messages and tool calls mediate inter-agent information flow. Figure 12 visualizes the stack together with representative entries at each level. An RL-oriented entry is characterized not only by whether it does explicit counterfactual credit assignment, but also by which level(s) in this hierarchy carry reward, credit, or optimization signals, and by what mechanism.

*Figure 12: The eight credit-bearing units in LLM-MAS RL (§7.1), stacked from coarsest (team) to finest (token). Red-outlined levels—orchestrator, role, message—have no clean counterpart in classical MARL or single-agent LLM RL. Right-column labels list representative entries that assign credit at each level; the sparse levels (orchestrator, message) mark the most under-populated research territory.*

### 7.2  A two-dimensional taxonomy of entries

Table 11 is the central organizing device of this section. Rows are representative entries from the paper pool plus a shared-outcome baseline; these include methods, frameworks, and system anchors. Columns are the seven above-token credit- or signal-bearing units; the token level is treated as inherited within each agent. Each filled cell names the mechanism by which the entry attaches a reward, credit, or optimization signal at that level.

$n$ | Entry | team | orch. | role | agent | turn | msg. | tool | dyn. ? | mech. |

| Shared outcome (GRPO baseline) | direct | – | – | – | – | – | – | – | heuristic |

| MAGRPO [37] | direct | – | – | group adv. | – | – | – | no | group-norm |

| MAPoRL [47] | direct | – | – | broadcast | – | – | – | no | broadcast |

| MARFT [32] | direct | – | – | per-agent | – | – | – | partial | PPO-style |

| Dr. MAS [15] | direct | – | – | agent-norm | – | – | – | no | agent-norm |

| CoLLM-MAAC [38] | direct | – | – | critic | – | – | – | no | actor-critic |

| LangMARL [71] | direct | – | – | language credit | – | – | – | no | lang-credit |

| MALT [44] | direct | – | role-PRM | – | PRM | – | – | no | learned PRM |

| MATPO [43] | direct | – | dual-role | – | – | – | tool outcome | no | shared-wt |

| Puppeteer [10] | direct | critic | – | – | – | – | – | yes | critic |

| Agent Q-Mix [23] | direct | – | – | QMIX | – | graph | – | yes | CTDE |

| M-GRPO [19] | direct | – | hier. base | indep. adv. | – | – | – | no | hier. GRPO |

| MarsRL [36] | direct | – | – | – | pipeline | – | – | no | stage-wise |

| C3 [7] | direct | – | – | – | – | CF | – | no | CF |

| SHARP [31] | direct | – | – | Shapley | – | – | tool-proc. | no | Shapley |

| HERA [30] | direct | evolve | – | – | – | – | – | partial | evol. |

| Kimi PARL [28] | direct | Crit-Steps | – | per-sub | – | – | – | yes | heuristic |

| WideSeek-R1 [68] | direct | lead/sub | – | per-sub | – | – | – | yes | width-RL |

| Context-Folding [55] | direct | – | – | – | branch | – | – | partial | fold |

| MARSHAL [75] | direct | – | – | agent-norm | turn adv. | – | – | no | self-play |

| DEPART [21] | direct | – | HIMPO | – | – | – | – | no | hier. PO |

| Agent Lightning [40] | direct | – | – | framework | turn | – | tool | yes | harness-gen. |

*Table 11: Two-dimensional taxonomy of credit and signal assignment in LLM-MAS RL. Rows are representative entries plus a shared-outcome baseline; columns 2–8 are credit- or signal-bearing units above the token level (§7.1); the token level is omitted because all entries inherit standard token-level GAE within each agent. dyn. $n$? = does the method accommodate a time-varying agent count (as in Kimi Agent Swarm)? mechanism = a one-phrase summary of how credit is mechanically computed. Cell entries name the decomposition or optimization signal applied at each granularity; bold marks the method’s distinctive contribution. Not every filled cell is an explicit counterfactual credit-assignment method; C3 is the retained entry that estimates counterfactual message-level credit. “direct” = the team reward is applied without decomposition; “–” = the method does not operate at that granularity. Empty cells for a given method do not mean the method is incomplete—they mean the method’s novelty lives elsewhere in the hierarchy.*

Three patterns in the table are worth naming explicitly.

-

Most RL-oriented entries contribute at exactly one novel level. Reading down the “bold” cells: Dr. MAS, CoLLM-MAAC, LangMARL, and Agent Q-Mix at agent / topology-conditioned agent credit, MALT at role, Puppeteer at orchestrator, M-GRPO at role, MarsRL at turn, C3 at message, SHARP at agent, Kimi PARL and WideSeek-R1 at orchestrator, and MARSHAL at turn. Each entry introduces or documents a technique for one layer of the hierarchy and leaves the others to standard machinery (GRPO, GAE, broadcast). This is not a criticism—it is how a research community makes progress—but it does mean no single method yet covers the full hierarchy.

-

The orchestrator level is sparsely populated. The CSV tags eight retained entries at the orchestrator level, but most are design- or evolution-level orchestration signals. Puppeteer, Kimi PARL, and WideSeek-R1 are the clearest cases that explicitly attach an optimization signal to orchestrator decisions, and they do so by very different mechanisms (learned central critic vs. Critical-Steps heuristic). We read explicit orchestrator credit, rather than orchestration as a system form, as the underdeveloped column in the table.

-

The message level is even sparser. Classical MARL has a long tradition of communication-level credit (difference rewards on messages), but our curated LLM-MAS pool still contains only two entries tagged at the message level: Debate-as-Reward uses message-level debate outcomes as a reward signal, while C3 [7] is the only retained entry that explicitly estimates counterfactual message-level credit. This is a wide-open research direction.

A natural follow-up question is whether these mechanisms compose. Nothing in the taxonomy prevents stacking, e.g., Puppeteer’s orchestrator critic on top of C3’s message counterfactuals on top of Dr. MAS’s agent-wise normalization. The May 2026 additions make agent-, role-, and turn-level credit denser, but we are not aware of published work in our pool that composes explicit credit mechanisms across all these levels; we return to this in §11. For practitioners selecting among the methods, Figure 13 gives a first-pass decision heuristic keyed to four system-level properties.

*Figure 13: A decision-tree heuristic for selecting a credit-assignment mechanism from our pool, organized by four system-level questions: (i) whether the agent set is dynamic at inference, (ii) whether the orchestrator is the identified bottleneck, (iii) whether traces are long enough to suffer diffusion, and (iv) whether roles are heterogeneous or the structure is debate-shaped. Leaves name the method whose design target most closely matches the path; in practice, industrial systems composite multiple choices rather than pick exactly one.*

### 7.3  Why naive single-agent credit assignment fails

A tempting position is that team-level reward plus standard GAE within each agent is enough, and that the rest of Table 11 is decoration. Three failure modes observed in the pool argue against this position.

-

Reward diffusion. Kimi’s Agent Swarm traces reach up to $1{,}500$ coordinated steps / tool calls in the public K2.5 training anchor [28]. Even at that scale, a single terminal reward distributed over realized training decisions can make the per-decision learning signal fragile and low-SNR. K2.6 extends the public deployment envelope to $4{,}000$ coordinated steps [29]; we use this only as scale-pressure evidence, not as an independently disclosed RL-training trajectory. Dr. MAS [15] documents a related empirical symptom as training instability that is not cured by hyperparameter tuning alone.

-

Asymmetric contribution. A single critic message—“this plan will not work because $X$”—can flip a $1{,}500$-step trace from failure to success. Uniform credit over agents (or uniform advantage over messages) assigns this pivotal message the same weight as routine executor chatter. C3 [7] argues that this is the right setting for counterfactual message-level credit specifically because contributions are so heavy-tailed.

-

Counterfactual ambiguity about spawning. When the orchestrator spawns a sub-agent and the trace succeeds, was the sub-agent responsible, or would the orchestrator have succeeded without spawning at all? No reward defined over realized trajectories can answer this; it requires a counterfactual over an unrealized alternative trace. This is why orchestrator credit is fundamentally harder than agent credit—and why the orchestrator column in Table 11 is sparse.

### 7.4  Open algorithmic questions

-

Compositionality. Can the mechanisms in Table 11 be stacked (e.g., Puppeteer’s orchestrator critic $+$ C3’s message counterfactual $+$ Dr. MAS’s agent normalization) without their regularizing effects cancelling? SHARP’s Shapley+tool-process, MARSHAL’s turn-level estimator plus agent-specific normalization, and DEPART’s dense role-specific plus sparse task rewards are partial composites, but each remains limited to a small subset of the hierarchy.

-

Process vs. outcome balance. When a dense PRM signal (R4) is combined with a sparse team reward (R1), the dense signal typically dominates gradients and the policy drifts toward what the PRM rewards rather than what the task rewards. MALT [44] uses role-specific PRMs to reduce this; a general principle is missing.

-

Dynamic-agent Shapley. SHARP computes Shapley credit over a fixed agent set. In systems like Kimi Agent Swarm where the agent set is itself produced by a policy decision (spawn / despawn), classical Shapley axioms no longer hold. A Shapley-analogue for dynamic coalitions is open.

-

Credit for the decision not to spawn. The orchestrator’s policy space includes “do nothing.” This decision does not produce a realized sub-trace, and the entries in Table 11 do not address it. A principled treatment likely requires off-policy evaluation of unrealized branches—a direction for which we found no entry in the pool as of May 4, 2026.

### 7.5  Discussion: composing the sparse and dense levels

The taxonomy in Table 11 still reads mostly as a story about single-level interventions: most methods target one dominant level and inherit standard machinery on the others. Three synthesis observations follow.

Density and visibility trade off. The token level is the densest signal source available (every generated token gives a gradient through GAE) but is also the level most divorced from team outcome. The team level is the cleanest signal source but emits one number per trace. All other levels lie on a Pareto frontier between these endpoints—role and turn levels yielding medium density and medium outcome-attribution; orchestrator and message levels yielding low density (few decisions per trace) but high attribution (each decision is consequential). Newer entries such as DEPART [21] and MARSHAL [75] mix sparse task reward with denser role or turn signals, but an exact partition of credit across the full hierarchy remains absent.

Counterfactual-based methods are quadratically expensive and structurally fragile. Both C3 [7] and the spawn counterfactual hinted at by §7.3 require estimating the return of an alternative trace not actually produced. C3 handles this at the message level by sampling substitute messages; the cost is at least linear in number of messages times sample count, and grows quadratically if the substitution is itself contextual. SHARP’s Shapley sampling [31] is similarly Monte-Carlo expensive. Practical counterfactual credit at Kimi-reported long trace lengths ($T\sim 10^{3}$, §5.3) would require either a learned counterfactual estimator (a “what-would-have-happened model” trained from off-policy data) or some form of importance sampling over realized branches. Neither has a published instance in our pool.

Compositionality is not for free. Stacking, e.g., agent-wise normalization (Dr. MAS) below a learned orchestrator critic (Puppeteer) below a message-level counterfactual (C3) is algebraically possible but introduces a new failure mode: credit double-counting. If a pivotal message is rewarded once via C3 and again via Puppeteer’s critic for the orchestrator’s delegation that produced the message, the policy receives stronger gradient on that message than on others—a kind of credit collision that classical MARL avoids by design (each agent has exactly one credit channel via CTDE). A clean compositional framework would need to specify a partition of the team reward across credit channels and enforce that the partition is exact. We found no such framework in our pool.

Takeaway. The central technical claim of this survey is that LLM-MAS methods should be read along a credit- and signal-bearing-unit hierarchy, not a flat list of “multi-agent RL tricks.” Under that reading, the literature as of May 4, 2026 populates some columns densely (agent via MAGRPO / Dr. MAS / SHARP / LangMARL / CoLLM-MAAC, role via MALT / M-GRPO / DEPART / LAMO) and others sparsely (especially explicit counterfactual message credit, and still explicit orchestrator credit)—and the sparse columns are where the near-term research opportunity is concentrated. Composing across columns introduces credit double-counting risks that no published method in our pool has formally addressed.

## 8  Learning Orchestration: Trajectory $\to$ Orchestration Trace

Reward design (§6) and credit assignment (§7) answer the questions what to measure and where to assign it. This section answers what is being optimized: the orchestration trace. We make the object formal (§8.1), organize methods by which orchestration sub-decision they learn (§8.2), discuss training regimes (§8.3), survey engineering constraints (§8.4), and enumerate failure modes specific to orchestration learning (§8.5).

### 8.1  The orchestration trace as a first-class object

A trajectory in single-agent RL is a sequence $\tau=(s_{0},a_{0},r_{0},s_{1},a_{1},r_{1},\ldots,s_{T})$. An orchestration trace is a temporal interaction graph $G=(V,E,\ell)$, where:

-

$V$ is a set of events: orchestrator decisions, sub-agent invocations, tool calls, messages, summary returns, and aggregation points.

-

$E\subseteq V\times V$ is a set of temporal/causal dependencies: “this sub-agent was spawned by that orchestrator decision”, “this aggregator consumed those summaries”, “this tool call followed that planning message”.

-

$\ell:V\to(\text{agent},\text{role},\text{content})$ labels each event with the executing agent, its role, and structured content.

A trajectory is a linearly ordered special case of an orchestration trace ($|V|$ = episode length, $E$ = successor relation). The multi-agent case is genuinely graph-structured: branching (parallel spawn), joining (aggregation), and delegation (orchestrator $\to$ sub-agent) have no trajectory analogue. Figure 14 contrasts the two objects. Consequently, in our taxonomy, the optimization target is naturally defined over $G$:

$\max_{\theta}\;\mathbb{E}_{G\sim\pi_{\theta}}\!\big[\,R(G)\,\big],$ | | | |

where $R$ is the composite reward from Table 10 and $\pi_{\theta}$ is the joint policy (orchestrator + sub-agents + aggregation). For RL-oriented entries, we use this as the common comparison object, even when the original papers do not frame their objectives this way; framework, benchmark, and industrial-anchor entries are used only where their public material constrains this comparison.

*Figure 14: Optimization objects for single-agent LLM RL vs. LLM-MAS RL. (a) A trajectory $\tau$ is a linearly ordered sequence of $(s_{t},a_{t})$ pairs. (b) An orchestration trace $G=(V,E,\ell)$ is a temporal interaction graph: orchestrator decisions (red) spawn sub-agents (blue), which issue tool calls (orange) and return summaries that are aggregated (green diamond) before the next orchestrator decision. Credit-bearing units (§7) attach at distinct substructures of $G$ rather than to time-indexed states. Compared to (a), the trace in (b) has branching, joining, and variable shape across rollouts.*

### 8.2  Methods by orchestration sub-decision

An orchestration trace is produced by a sequence of sub-decisions, each of which can, in principle, be the target of a learned policy. We identify five (Figure 15). The remainder of this subsection expands each one: what the decision is, what signal would train it, what retained entries address it, and what remains open.

*Figure 15: The five orchestration sub-decisions O1–O5 (§8.2). An orchestrator policy makes some or all of these decisions per task; surveyed entries cover O1–O4 but not O5. The red dashed box marks “when to stop” as a named open problem (§11): in the entries we surveyed, termination is either externally signaled (ground-truth answer found) or triggered by a fixed step-count cap rather than explicitly trained as a stopping policy.*

#### 8.2.1  O1: When to spawn

Decision. Given the current partial trace, does the orchestrator issue a spawn action, and with what role / context? The policy’s support includes no-op; exercising spawn commits to downstream rollout cost (§5.1).

What signal would train it? Ideally, the counterfactual team return under spawn vs no-op. Because no-op is never actually rolled out once spawn is chosen, the counterfactual is unobserved—this is the non-identifiability argument stated as Claim 3.4 in §3.4.

Entries in our pool. Kimi PARL [28] makes spawning an action and gives it two reward shapes: $r_{\text{parallel}}$ (against serial collapse) and $r_{\text{finish}}$ (against spurious parallelism), with both auxiliary weights reported as annealed to zero over training (Fig. 11). AgentSpawn [9] triggers spawn via learned complexity estimators at runtime; HALO [20] applies MCTS over spawn decisions at a hierarchical level, treating spawn as planning rather than as an RL action.

What remains open. None of the three methods uses an explicit counterfactual estimator; all use $R7$ shaping or search heuristics as proxies. A principled off-policy evaluation of unrealized no-op branches is the obvious missing piece (§11, P4).

#### 8.2.2  O2: Whom to delegate to

Decision. Conditional on spawn being chosen, which agent among the currently instantiated pool $\mathcal{I}_{t}$ (or newly created agent of a given role) receives the next task chunk?

What signal would train it? A per-delegation return differential—which agent, in context, would have produced the best outcome. This is the classical setting for a centralized critic.

Entries in our pool. Puppeteer [10] trains exactly such a learned central critic in a CTDE style (§2.1), freezing sub-agents and updating only the orchestrator. ParaManager [76] generalizes the support: agent and tool dispatch share a unified action space $\mathcal{A}_{\text{delegate}}=\{\text{sub-agent}_{i}\}\cup\{\text{tool}_{j}\}$, which lets the orchestrator trade off between creating a sub-agent and directly calling a tool with no delegation overhead. WideSeek-R1 jointly optimizes a lead agent and parallel sub-agents for broad information seeking, making width scaling itself part of the learned delegation regime [68].

What remains open. These academic works still operate far below the largest disclosed industrial swarms. In industrial swarms (§4.2.1) the pool is dynamic and can grow to hundreds; scaling a learned dispatcher to that regime is unaddressed.

#### 8.2.3  O3: How to communicate

Decision. What is the content, length, and format of messages exchanged between orchestrator and sub-agents, and among sub-agents?

What signal would train it? A reward or credit signal for a specific message’s contribution to team outcome—the message-level unit of §7.1. Explicit counterfactual message-level credit is the sparsest column of Table 11.

Entries in our pool. Three retained RL entries engage O3 directly. Debate-as-Reward [51] rewards resolution-based messages, disincentivizing length-based farming. C3 [7] estimates counterfactual contribution per message via contextual intervention. Agent Q-Mix [23] learns decentralized communication/topology decisions with a QMIX-style CTDE objective, treating the round-wise communication graph as the object to optimize. LatentMAS [84] takes the opposite route: replace token-level messages with a continuous latent channel, eliminating the message-level credit-assignment problem by changing the communication medium altogether (and, empirically, gaining $+14.6\%$ without any training).

What remains open. The field now has several point-solutions (token counterfactual, learned topology, latent channel) but still no unified information-theoretic account. A principled treatment of which bits of information an orchestrator should exchange—the direct LLM analogue of Shannon-rate constraints in classical emergent communication—remains absent from our pool.

#### 8.2.4  O4: How to aggregate

Decision. When sub-agents return partial results, how does the orchestrator combine them into the trace state that gates the next decision? Summaries, votes, consensus, or structured merge.

What signal would train it? The aggregation step is itself a policy output; it can receive either team reward (slow-moving) or a per-aggregation proxy (e.g., whether aggregated output contains the key fact needed for the next sub-decision).

Entries in our pool. M-GRPO [19] formalizes aggregation as a separate main agent whose policy consumes sub-agent summaries and emits trajectory continuations. Context-Folding [55] treats aggregation as an explicit agent action, rewarding branch outcomes approximately as $r_{\text{branch}}\approx r_{\text{main}}\pm 0.2$ scope adjustment.

What remains open. Both methods aggregate via LLM summarization—lossy and uncalibrated. An aggregator that explicitly models the uncertainty of sub-agent claims (e.g., a Bayesian combiner) has no entry in our pool.

#### 8.2.5  O5: When to stop

Decision. At which point does the orchestrator halt the trace and emit the final answer?

What signal would train it? Expected marginal gain of one more orchestration step vs. the cost of that step (§5.1). A stopping policy that trades accuracy for cost is a natural objective.

Entries in our pool. We found no retained entry that trains this decision directly. Existing entries stop either externally (e.g., ground-truth answer verifier signals completion) or at a fixed step-count cap. The orchestrator’s stop action is, as far as we can tell from public material, not explicitly trained as an RL target in any entry in our curated pool.

What remains open. This is the sub-decision with the clearest shape of an open research direction: a small modification to any orchestrator policy that adds a stop action and trains it against a cost-adjusted return would be the first entry in this cell of the taxonomy.

### 8.3  Orchestrator training regimes

Three regimes appear repeatedly; Figure 16 shows the gradient-flow pattern of each.

*Figure 16: Three orchestrator training regimes (§8.3). (A) Frozen sub-agents: gradient flows only into the orchestrator; cheapest and most common. (B) Joint training with shared baseline and per-agent advantage: all policies update together; requires stabilization (Dr. MAS’s agent-wise normalization). (C) Fully decoupled per-policy training against a central critic $V_{\phi}$: most expressive but most engineering-heavy. Solid red arrows show gradient flow; dashed outlines indicate frozen components.*

-

(A) Frozen sub-agents, train only the orchestrator. Cheapest and the most common in practice. Kimi PARL’s first stage [28], Puppeteer [10], and Agent Q-Mix’s CTDE-style topology learner [23] all fit this family in different ways. It avoids joint-training instability and is the “credit-assignment safest” choice because only the orchestrator’s policy gradient flows.

-

(B) Joint training with shared baseline and per-agent advantage. Context-Folding [55] is the clearest example: orchestrator and sub-agents are trained together, share a team baseline, but compute advantage independently. Dr. MAS [15] is the stability analysis of why naïve joint training under GRPO fails and how agent-wise normalization fixes it. WideSeek-R1 [68] and MARTI-MARS2 [61] add newer examples where lead/sub-agent or heterogeneous multi-agent policies are optimized together.

-

(C) Fully decoupled per-agent training with a central critic. M-GRPO [19] is the canonical hierarchical instance: top-layer and bottom-layer receive separate advantage signals. MATPO [43] implements a single-LLM analog where the planner and worker share model weights but receive role-specific advantages. DEPART [21] alternates planner and executor optimization under dense role-specific and sparse task rewards; SPIRAL [34] and MARSHAL [75] show the same decoupling pressure in self-play settings.

Regimes (A)–(C) span a trade-off between training cost and expressivity. (A) is cheap but cannot update sub-agent skill; (B) updates everyone but is unstable without explicit normalization; (C) is expressive but requires separate replay buffers per role and is the most engineering-heavy.

### 8.4  Engineering: rollout topology and asynchrony

Multi-agent rollouts are cost-dominated by two factors: the slowest sub-agent and the inter-agent dependency graph. Four engineering techniques recur.

-

Pipeline parallelism. MarsRL [36] arranges reasoning agents into a pipeline so that different stages of different rollouts execute concurrently, amortizing rollout cost. WideSeek-R1 [68] and MARTI-MARS2 [61] add complementary width and self-search scaling evidence.

-

Execution–training decoupling. Agent Lightning [40] separates agent execution from the trainer; rollouts are produced by an inference harness and consumed asynchronously by the trainer. This matches the industrial harness boundary discussed in §4.2.2.

-

Variable-shape replay buffer. Orchestration traces have variable $|V|$, variable branching, and variable depth. No retained entry treats this as a first-class problem; most RL-oriented entries in our pool pad or truncate. This is a concrete open engineering problem (§11).

-

Reward normalization across trace shapes. Dr. MAS’s agent-wise normalization [15] addresses this at the advantage level; a trace-level analogue (normalize over graph depth / branching factor) is still missing.

### 8.5  Failure modes of orchestration learning

Five failure modes recur when orchestrators are trained directly; each maps to one of O1–O5.

-

Serial collapse (O1). As auxiliary orchestration rewards decay in Kimi PARL, a naive orchestrator can collapse to never spawning, regressing to a single-agent baseline. Mitigated by staged annealing rather than an abrupt removal.

-

One-dominant-agent collapse (O2). Under shared reward, the orchestrator routes nearly all delegations to a single sub-agent that happens to be slightly above-average. Population diversity can collapse. We found no general fix in our retained pool; debate-style topologies [51] partly sidestep by making diversity part of the reward.

-

Over-communication (O3). Orchestrators inflate message volume to farm message-level process rewards. Observed whenever the judge scores per-message.

-

Aggregation leakage (O4). Summary-return content is copied verbatim into the main trace, inflating apparent progress without real information gain; filtered by Context-Folding’s scope-adjustment reward.

-

Train–inference topology mismatch. Policies are trained at $k$-agent teams but deployed at $k^{\prime}$-agent teams (Kimi K2.5 discloses a $100$-sub-agent trained-orchestrator regime; K2.6 reports a $300$-sub-agent deployment envelope). Generalization across team size is under-studied and is an open problem (§11).

-

Adaptive deliberation outside the standard hierarchy. Learning to Deliberate [70] introduces decentralized meta-cognitive actions such as Persist, Refine, and Concede. These are not ordinary messages and are not purely central orchestrator decisions; they sit between turn-level credit and orchestration-policy credit. We keep them in the turn/orchestration region of the taxonomy, but they are evidence that future versions of the hierarchy may need an explicit meta-policy layer.

Takeaway. The trajectory-to-trace shift is not cosmetic: it changes the optimization target, the training regime choice (A/B/C), and the engineering stack (pipeline parallelism, decoupled harness, variable-shape replay). Recent width-scaling, topology-learning, and self-play entries fill in O2/O3 and role/turn credit, but do not close the stopping cell. Five sub-decisions (O1–O5) enumerate what a learnable orchestrator actually does; no single paper in our pool covers all five, and O5—when to stop—has no entry in our pool that trains it explicitly.

## 9  Benchmarks and Evaluation

A recurring pattern in our paper pool is that LLM-MAS methods report gains on benchmarks designed for single-agent evaluation. This can be methodologically hazardous: single-agent benchmarks measure task success, which any system with enough compute can improve; they do not measure whether the improvement came from genuine multi-agent coordination. This section uses an expanded evaluation surface to audit the current benchmark landscape (§9.1–§9.2) and then gives benchmark-design recommendations derived from the observed gaps (§9.4).

### 9.1  Four dimensions for auditing multi-agent evaluation

We use four dimensions to audit whether an LLM-MAS benchmark measures coordination rather than task success alone.

-

(E1) Task success / accuracy. The standard metric. Necessary but not sufficient.

-

(E2) Parallelism efficiency. Wall-clock speedup over a serial baseline; agent utilization (fraction of sub-agents doing task-relevant work); Critical-Steps-style metrics that distinguish real parallel progress from padded traces (§6.2).

-

(E3) Collaboration quality. Message redundancy, consensus quality, debate diversity, and—in the specific case of debate-like topologies—whether resolution is reached. LatentMAS [84] is the clearest evidence that much of (E3) can be achieved without natural-language messaging at all.

-

(E4) Protocol overhead. Token cost per delegation, error-amplification ratio (how a single bad message propagates through the trace), and safety-related properties such as prompt-injection flow.

A benchmark is MAS-native when it reports at least three of (E1)–(E4). By this criterion, most benchmarks in our pool are not MAS-native—they report (E1) only.

### 9.2  Benchmark landscape by domain

Table 12 organizes the benchmarks referenced in the pool by domain and MAS-nativeness.

| Domain | Benchmarks (cited examples) | MAS-native? | Measures (E1–E4)? |

| Coding | SWE-Bench [24], ArtifactsBench [79], CodeCriticBench [78] | No | E1 only |

| Web / browser | WebArena [82], BrowseComp [63] | No | E1 only |

| Research / search | GAIA [42] | No | E1; occasional E4 (token cost) |

$\tau$ | Tool use | ToolBench [49], -bench [72], MTU-Bench [60] | Partial | E1 + partial E4 (tool success) |

| Long-horizon OS | OSWorld [67] | No | E1 + wall-clock (partial E2) |

| MAS-oriented | MultiAgentBench [83], TAMAS [25]; reported internal: Kimi Swarm Bench [28] | Partial / closed | Open entries cover subsets; Kimi Swarm Bench is not coded as open evidence for E1–E4. |

*Table 12: Benchmark landscape for LLM-MAS evaluation, restricted to benchmarks with arXiv-cited entries. Almost all domain benchmarks report task success (E1) only; among MAS-native benchmarks in our pool, none covers all four dimensions (E1–E4) jointly.*

Two observations.

-

No benchmark in our pool covers all four dimensions in an open, auditable way. Kimi Swarm Bench [28] is treated as a reported internal benchmark, not as open evidence for E1–E4, because it is closed and unauditable. TAMAS [25] covers (E4) safety specifically and nothing else. MultiAgentBench [83] reports (E1) plus partial (E2)/(E3) depending on the task instance.

-

Cross-method comparability is limited. Because credit-assignment papers (§7) each pick a different benchmark to evaluate on—C3 on math collaboration, SHARP on tool-augmented tasks, M-GRPO on deep research—direct comparison of their credit-assignment mechanisms is not currently possible.

### 9.3  The benchmark gap

We argue that the shortage of MAS-native benchmarks is not merely inconvenient; it actively shapes which algorithms succeed. Three concrete consequences.

-

E1-only benchmarks reward compute, not coordination. A method that improves task success by spawning more sub-agents and trying them in parallel is indistinguishable on (E1) from a method that improves success by better credit assignment. The former is a more-compute scaling story; the latter is the claim this survey is organized around. Without (E2) this confound cannot be resolved.

-

Safety signals are underrepresented. Only TAMAS [25] reports adversarial robustness systematically. Inter-agent prompt injection, shared-memory poisoning, and tool-parameter escalation are present in deployed systems but absent from most eval suites.

-

Kimi-reported long traces are absent from open benchmarks. Kimi reports traces reaching $4{,}000$ steps in K2.6 [29]; no open benchmark evaluates at that trace length. The credit-diffusion failure mode (§7.3) is correspondingly invisible to academic evaluation.

### 9.4  What a good MAS-native benchmark would look like

The question is not only which existing benchmark to extend, but what design properties would close the gaps observed above. We sketch five recommendations derived from the gaps in §9.3 and from the engineering constraints of §5.

B1. Dimensional completeness. A MAS-native benchmark should report all four of (E1)–(E4): task accuracy, parallelism efficiency, collaboration quality, and protocol overhead. Without this, the compute-vs-coordination confound (§9.3) cannot be resolved on a per-instance basis.

| Metric | Operational definition | What it distinguishes |

$T_{\text{serial}}/T_{\text{parallel}}$ | Parallelism efficiency | or wall-clock serial baseline divided by MAS wall-clock | Real coordination speedup vs. merely spending more inference. |

| Useful-agent utilization | task-relevant sub-agent actions divided by total sub-agent actions | Productive decomposition vs. idle or redundant spawned agents. |

| Protocol overhead | orchestration, message, and tool-management tokens divided by total tokens | Coordination cost vs. task-solving content. |

| Message redundancy | semantically duplicate messages divided by total inter-agent messages | Useful communication vs. verbosity or padding. |

| Error amplification ratio | downstream corrupted events divided by the initial corrupted event | Whether one bad tool/message contaminates the trace. |

*Table 13: Operational metric definitions for MAS-native evaluation. Exact implementations will vary by benchmark, but reporting these quantities would make E1–E4 comparable across methods.*

B2. Trace-length stratification. Tasks should be grouped by expected trace length so that performance can be reported per $T\in\{10^{1},10^{2},10^{3}\}$ band, exposing the credit-diffusion behavior of §5.3. A method that excels at $T=50$ but degrades at $T=500$ is a qualitatively different beast from one that scales gracefully; current single-number benchmarks hide this distinction.

B3. Topology variability. Ideally, the same underlying task is instrumented for multiple topologies from Table 5—centralized, debate, swarm, hierarchical—so that orchestration choices can be ablated. The goal is not to crown a single best topology but to expose how much of the gain attributed to a credit-assignment method is actually attributable to its preferred topology.

B4. Adversarial control conditions. Each task should ship with controlled adversarial perturbations from the attack vectors of §10.2: indirect prompt injection in tool output (AV2), inter-agent message pollution (AV3), shared-memory poisoning (AV4). A method’s robustness margin under these perturbations becomes a first-class metric, not an afterthought.

B5. Open data, public leaderboard, replayable traces. Instances and orchestration traces are released; runs are reproducible; scores are reported per-band rather than as a single average. This addresses the cross-method comparability problem (§9.3): publication of full orchestration traces lets follow-up work re-evaluate without re-running rollouts.

Minimal trace reporting schema. To make B1–B5 operational, an evaluation should log the orchestration trace as a typed event graph rather than only a final answer and score. The following schema is minimal: it is not a benchmark proposal by itself, but it is the smallest artifact that would let later work recompute reward, credit, parallelism, and safety metrics over the same rollout. The same structure is provided as a machine-readable JSON Schema in the artifact repository, together with a minimal valid example trace. The accompanying Python validator is a lightweight structural checker for required fields, event types, edge references, duplicate event identifiers, and non-negative costs; it is not a full implementation of the JSON Schema standard.

{ "trace_id": "...", "task_id": "...", "events": [ {"id": "e1", "t": 0, "type": "spawn", "agent": "orchestrator", "role": "planner"}, {"id": "e2", "t": 1, "type": "message", "agent": "planner", "from": "planner", "to": "executor"}, {"id": "e3", "t": 2, "type": "tool_call", "agent": "executor", "tool": "browser"} ], "edges": [{"src": "e1", "dst": "e2", "type": "causal"}], "rewards": {"team": 1.0, "orchestration": 0.3, "tool": 0.1}, "costs": {"tokens": 12000, "wall_clock_s": 420} }

| Reporting item | Minimum information |

| Topology and roles | Active roles, spawn/despawn events, fixed vs. dynamic team size. |

| Trace scale | Number of events, messages, tool calls, sub-agents, wall-clock time, and token cost. |

| Reward channels | Team reward plus any process, tool, verifier, or orchestration rewards and whether auxiliary terms are annealed. |

| Credit unit | Finest unit receiving an advantage, value estimate, counterfactual score, Shapley score, or learned critic signal. |

| Safety instrumentation | Whether untrusted tool output, inter-agent messages, shared memory, and human interventions are separately logged. |

*Table 14: Minimal reporting checklist for MAS-native evaluation. This checklist is intended to make trace-level claims auditable without requiring a new benchmark suite.*

The closest non-open reference point is the internally reported Kimi Swarm Bench [28]; we treat it as context, not as an auditable benchmark row for the E1–E4 gap. The closest open approximation is MultiAgentBench [83], which covers a subset of (E1)–(E3) at small $T$. A benchmark satisfying B1–B5 jointly would directly address the main comparability bottleneck in this survey: without it, new credit-assignment methods cannot be evaluated against a shared coordination-sensitive target.

Takeaway. Within our retained pool as of May 4, 2026, we found no single open benchmark that reports (E1)–(E4) jointly at the Kimi-reported long-trace envelope. This is the most tractable near-term infrastructure gap: credit-assignment methods cannot be fairly compared until evaluation measures more than task success.

## 10  Safety and Adversarial Robustness in LLM-MAS

The benchmark gap (§9.3) intersects with a second underdeveloped axis: adversarial robustness. Single-agent agentic safety is itself unsolved [17, 12], but LLM-MAS introduces failure modes whose multi-agent propagation patterns have no close single-agent analogue and only a handful of dedicated benchmarks in our pool. This section keeps the focus narrow: attacks are organized by the same credit- and signal-bearing units used in §7.1, because the levels at which credit is assigned are also the levels at which adversarial influence can enter and propagate.

### 10.1  Threat model for LLM-MAS

Three properties introduce attack surfaces beyond those of a single tool-using agent. First, the attack surface scales with team size: every spawned sub-agent inherits tool access, every message is a potential injection point, and every shared-memory write is a potential poison; the number of inter-agent flows grows super-linearly in the number of nodes of the orchestration trace (§7.1). Second, information flows between LLMs that each treat the other’s output as trusted natural language; a compromised tool output that passes through one sub-agent’s summary becomes an instruction to the next sub-agent or to the orchestrator [17, 77]. Third, dynamic-spawn systems (Kimi PARL, AgentSpawn [9]) create sub-agents at runtime whose isolation cannot be audited in advance. The May 2026 refresh also adds a training-time safety method: MAGIC formulates attacker and defender LLMs as a co-evolving multi-agent RL game [65], which is useful evidence for adversarial safety training even though it does not by itself solve trace-level constrained optimization for deployed swarms.

We distinguish three threat actors. (i) External user input is the classical jailbreak/prompt-injection channel. (ii) Untrusted tool output is the channel exploited by indirect prompt injection: a web page, an email body, or a retrieved document contains adversarial text that is treated as instructions when it re-enters the LLM context [17, 12, 14]. (iii) Adversarial agent in the team is novel to MAS: either a member sub-agent has been compromised at spawn time, or a team-internal message has been poisoned at runtime, after which the contagion propagates through shared memory [41, 27, 25]. Classical MARL safety—reward hacking, shielding, safe exploration—is necessary but not sufficient: it does not address natural-language information flow between LLMs, which is where most LLM-MAS attacks land in our pool.

### 10.2  Attack taxonomy

Table 15 organizes attack vectors against the credit- and signal-bearing units of §7.1. The mapping is not coincidental: the same structural levels at which credit must be assigned are the levels at which an attacker can pivot.

| Attack vector | Threat actor | Affected level | Representative attack / benchmark; defense status |

| Direct prompt injection | user | token / turn | Long-studied single-agent threat; AgentDojo [12] covers within tool-use loops; defenses are input-side filters. |

| Indirect prompt injection | tool output | tool / message | Greshake et al. [17], InjecAgent [77], WASP [14]; defenses are tool-output sanitization and provenance tracking, both partial. |

| Inter-agent message pollution | adversarial agent | message | Agents Under Siege [27] optimizes permutation-invariant attacks across topologies; no published defense in our pool. |

| Shared-memory poisoning | adversarial agent | agent / team | TMCHT [41] demonstrates contagious jailbreak via memory writes that survive across turns and propagate to honest agents. |

| Tool-parameter escalation | user / tool output | tool | A compromised sub-agent issues tool calls outside its declared capability set; mitigated heuristically by per-sub-agent tool allowlists [3]. |

| Orchestrator misdirection | user / message | orchestrator | A poisoned summary causes the orchestrator to spawn the wrong sub-agent or route the wrong task; observed in TAMAS [25]; no published RL-time defense. |

| Reward hacking across agents | policy itself | agent / role | Pseudo-parallelism, communication padding, verifier collusion (§6.3); MAGIC [65] trains an attacker–defender game for safety alignment, but trace-level constrained optimization remains open. |

| Sandbox escape under spawn | adversarial agent | system | Per-sub-agent isolation in dynamic-spawn systems is not publicly specified; flagged as P14 in §11.4. |

*Table 15: Attack vectors against LLM-MAS, organized by the credit- or signal-bearing level at which the attack lands (§7.1). Most published attacks target message or tool levels; orchestrator and role attacks are barely studied in our pool, and sandbox/system attacks are not publicly documented at all.*

*Figure 17: Attack-surface map for an LLM-MAS orchestration trace. The substrate (orchestrator $\to$ sub-agents $\to$ tools $\to$ shared memory) is the same as Fig. 14; red labels mark eight attack vectors (AV1–AV8) cataloged in Table 15, each anchored to its locus of action. Right legend shows which credit-bearing unit (§7.1) is the locus of attack: this mapping is what makes credit-assignment progress and security progress mutually informative. AV1–AV2 derive from single-agent indirect prompt injection [17, 77, 12]; AV3–AV4 are MAS-specific [41, 27]; AV5/AV8 are tool-runtime concerns; AV6/AV7 emerge only in orchestrated systems and remain under-benchmarked.*

### 10.3  Defense landscape and benchmarks

Inference-time defenses in our pool are mostly ad hoc mitigations: input and tool-output sanitization [12, 17], tool-parameter allowlists [3], per-task or per-sub-agent sandboxing [45, 3], reward-model verification, heterogeneous verifiers, and organizational controls from trustworthy-agent frameworks (e.g., human checkpoints and scoped credentials) [4]. MAGIC [65] is the clearest retained training-time counterexample: it uses co-evolving attacker and defender agents to manufacture adversarial safety data and optimize the defender. Within our pool, we found no trace-level constrained-optimization formulation comparable to constrained MDPs or shielded RL. This is the safety-side analogue of the credit-assignment gap: the defense must decide which orchestrator, role, message, or tool event should be constrained, edited, or blamed.

The benchmark situation is similarly sparse. TAMAS [25] is the most MAS-specific benchmark in our pool; AgentDojo, InjecAgent, and WASP cover adjacent tool-use or web-agent injection settings [12, 77, 14]; and two ACL 2025 works move into multi-agent attack propagation through topology and shared memory [27, 41]. None of these benchmarks jointly reports safety with collaboration quality (E3) and parallelism efficiency (E2), so they cannot yet support the kind of reward–credit–safety comparison needed by this survey.

### 10.4  The under-addressed problem: steerability

Anthropic’s trustworthy-agents framework [4] isolates a property they call mid-trace steerability: the ability of a human supervisor to inspect a partially completed orchestration trace, intervene at a specific point, and have the intervention propagate sensibly forward. This is a credit-assignment-shaped problem in disguise. To intervene informatively the supervisor must answer the same questions the RL trainer asks: which earlier orchestrator decision is responsible for the current state, and which downstream sub-agent decisions will be invalidated by changing it? An intervention at the wrong decision is wasted (the system reverts) or destructive (downstream sub-agents continue on stale assumptions). The hierarchy that organizes credit (orchestrator, role, agent, message, tool; §7.1) thus also organizes intervention points, and the counterfactual machinery used by C3 [7] to attribute credit to messages could in principle attribute the consequences of a human edit to messages. We are not aware of a published RL formulation of steerability in our pool; the closest existing work treats it as a UI/HCI question rather than as an RL objective. This is the connection point to P13 in §11.4.

Takeaway. LLM-MAS safety inherits all of single-agent agentic safety and adds three structural threats whose propagation patterns are multi-agent-specific: inter-agent message pollution, shared-memory contagion, and orchestrator misdirection. The defense landscape is uniformly ad hoc at inference time, while MAGIC shows that adversarial attacker–defender RL is beginning to enter the training side. The benchmark coverage remains shallow (TAMAS, AgentDojo, InjecAgent, WASP, plus two ACL 2025 multi-agent attack papers), and steerability—one operationally important safety property—has not been formalized as an RL objective in any work we are aware of.

## 11  Open Problems

We close the survey with fifteen open problems, organized along five axes: algorithmic (P1–P5), reward (P6–P8), systems (P9–P11), safety (P12–P14), and evaluation (P15). Each problem is stated compactly with a pointer to where in the survey it was developed and to the closest published work (if any) that addresses it.

### 11.1  Algorithmic

P1. Credit diffusion under long traces. Terminal-only team reward over $10^{3}$–$10^{4}$ orchestration steps can make the per-decision signal fragile or low-SNR (§7.3). Dr. MAS [15] addresses a symptom (training instability) through agent-wise normalization, while LangMARL [71], CoLLM-MAAC [38], and MARSHAL [75] add denser language, critic, or turn-level signals. A principled account of how these signals scale to Kimi-reported long traces is still missing in our retained pool.

P2. Free-riding under shared reward. Under R1 shared reward, silent or near-silent sub-agents receive equal credit. SHARP [31] offers Shapley-based marginal credit; an open question is whether Shapley approximation remains tractable at production team sizes ($n\gtrsim 100$, §4.2.1).

P3. Coordination collapse and one-dominant-agent. Under joint training, population diversity often collapses (§8.5); the orchestrator routes most delegations to a single sub-agent. Within our retained LLM-MAS RL pool, we found no method that rewards agent diversity directly across dynamic swarms. MARTI-MARS2 [61] shows policy-diversity gains from heterogeneous self-search training, but diversity is not yet a general-purpose orchestration objective.

P4. Counterfactual credit over unrealized branches. The orchestrator’s policy includes “do not spawn.” No realized trace exists to attribute credit against. C3 [7] handles message-level counterfactuals within a realized trace, not across realized/unrealized alternatives. Off-policy evaluation of unrealized branches is an open direction.

P5. Train–inference topology mismatch. Methods are trained at $k$ agents but deployed at $k^{\prime}$: Kimi K2.5 discloses a trained-orchestrator regime up to $100$ sub-agents, while K2.6 reports a deployment envelope up to $300$ sub-agents [28, 29]. Whether a trained orchestrator policy generalizes across team size, and under what conditions, is under-studied.

### 11.2  Reward

P6. Reward hacking in tool environments. Tool-spam, fabricated tool success, and padding (§6.3). Agent Lightning [40] and MATPO [43] condition tool reward on downstream outcome; a general principle for pricing tool calls is absent in our retained pool.

P7. Verifier–policy collusion. When a verifier LLM is drawn from the same family as the policy, both drift together and the verifier reward becomes uninformative. We are not aware of a fix in our pool beyond using heterogeneous verifier families, which is brittle.

P8. Process–outcome reward balance. Dense PRM (R4) combined with sparse team outcome (R1) lets the dense signal dominate gradients, causing policy drift toward what the PRM rewards rather than what the task rewards. MALT [44] uses role-specific PRMs as a mitigation; a general principle is missing in our retained pool.

### 11.3  Systems and engineering

P9. Rollout cost dominance. Multi-agent rollouts are $10$–$100\times$ more expensive than single-agent and dominate wall-clock RL time (§8.4). Pipeline parallelism (MarsRL [36]) and execution–training decoupling (Agent Lightning [40]) are partial answers; further gains likely require hierarchical rollout scheduling.

P10. Variable-shape replay. Orchestration traces have variable $|V|$, branching, and depth. Standard replay buffers pad or truncate. A graph-native buffer and a matching advantage normalization (cf. Dr. MAS [15] at the agent level) are still missing in our retained pool.

P11. Straggler-robust training. The slowest sub-agent gates the whole trace. Bias correction for asynchronous rollouts—on-policy vs. near-on-policy—is not addressed in any retained LLM-MAS method we found.

### 11.4  Safety

P12. Inter-agent prompt injection. Untrusted tool output flows through the team; one compromised message can pivot the orchestrator. Foundational indirect prompt injection [17, 77] already afflicts single-agent settings; LLM-MAS compounds the problem by letting the injection propagate across sub-agents as trusted summaries (§10). TAMAS [25], AgentDojo [12], and WASP [14] establish benchmarks at different scales; MAGIC [65] adds a training-time attacker–defender RL countermeasure, but deployed trace-level defenses remain mostly ad hoc (§10.3).

P13. Mid-trace steerability. Anthropic’s framework [4] flags that humans cannot easily intervene mid-orchestration. This is credit-assignment-shaped: where in the trace can a human inject, and what are the downstream consequences? Multi-agent attacks that exploit this gap—including permutation-invariant topology attacks [27] and contagious memory poisoning [41]—are now documented; we found no paper in our pool that formalizes steerability as an RL objective.

P14. Sandbox-isolation under dynamic spawn. Each new sub-agent needs its own sandbox; failure modes scale with team size. Dynamic-spawn systems such as Kimi PARL and AgentSpawn [9] do not publicly discuss how per-sub-agent isolation is guaranteed.

### 11.5  Evaluation

P15. MAS-native benchmark at Kimi-reported trace lengths. As argued in §9.3, we found no open benchmark in our retained pool that covers (E1)–(E4) jointly at $\gtrsim 10^{3}$-step traces. WideSeek-R1 [68] and MARTI-MARS2 [61] make academic width/self-search scaling more serious, but they do not replace an open trace-level benchmark at industrial lengths. This is the single most tractable infrastructure gap: without it, credit-assignment methods cannot be fairly compared and scaling claims cannot be cross-validated. Likewise, no explicit RL training method in the retained pool targets the stopping decision as a learned O5 policy; this does not rule out heuristic, budgeted-inference, or non-RL halting mechanisms.

Takeaway. Of the fifteen problems, P1 (credit diffusion), P4 (unrealized-branch counterfactuals), and P15 (MAS-native benchmarks) are the most load-bearing: progress on them would unlock progress on many of the others. P5, P10, and P13 are the most deployment-relevant and the most under-published.

## 12  Limitations

This survey is intended as a curated taxonomy and position paper, not as an exhaustive systematic review. Four limitations are therefore important for interpreting its claims.

Curated rather than exhaustive corpus. The retained pool contains $84$ entries selected for their relevance to reward design, credit assignment, orchestration learning, systems constraints, benchmarks, or safety in LLM-MAS RL. It is not a complete bibliography of all LLM-MAS, agentic-RL, hierarchical-RL, or MARL work. The inclusion protocol and screening-decision log make the curation auditable at record level for this manuscript. They do not make the review fully reproducible in the PRISMA sense because we do not provide database-result exports, deduplication logs, or a multi-annotator screening protocol.

Single-author tagging. The taxonomy tags in the retained-entry CSV were assigned by manual reading of abstracts, methods, and public artifacts. We did not run a blinded multi-annotator protocol or report inter-annotator agreement. Borderline entries—for example surveys with RL relevance, industrial systems with undisclosed training details, and self-evolution frameworks that do not optimize a conventional RL objective—should be read as taxonomy judgements rather than objective labels.

Industrial sources are not reproducible algorithms. Kimi K2.5 is the only industrial source in our pool that explicitly discloses RL training of the orchestrator. Codex, Claude Code, Kimi K2.6, and Anthropic engineering case studies are used for deployment shape, scale, harness boundaries, and workflow pressure unless a public source explicitly discloses the training objective. We do not infer undisclosed multi-agent RL objectives from product behavior.

Formal claims are organizing arguments. The dynamic-Dec-POMDP and orchestration-trace definitions fix the vocabulary used by the survey. The two conceptual claims about credit diffusion and non-identifiability are not theorems, and Figure 9 is schematic rather than a fitted empirical law. A full theory would need explicit assumptions on noise, baselines, graph dynamics, off-policy branch coverage, and value-function approximation over variable-shape traces.

The literature is moving quickly. The cutoff for this version is May 4, 2026. New industrial reports, OpenReview submissions, benchmarks, and safety evaluations can change both the coverage map and the sparsity conclusions, especially in the message-credit, orchestrator-credit, adaptive-deliberation, and trace-level safety cells.

## 13  Reproducibility and Artifact Statement

The manuscript is accompanied by a supplementary artifact snapshot, mirrored in the artifact repository (https://github.com/xxzcc/awesome-llm-mas-rl) with repository-path normalization (for example, repository scripts/ and trace-schema/ paths correspond to the manuscript-bundle artifact/ paths), intended to make the taxonomy auditable. The artifact contains four components.

-

Corpus metadata. A retained-entry CSV records the $84$ retained entries with 18 controlled fields. An exclusion-log CSV records the $32$ screened-but-excluded decisions, each with a public identifier or documentation handle and URL.

-

Scripted statistics. A repository script regenerates the retained / excluded counts, controlled-field histograms, and cross-tabs used in Table 1, Table 2, and Table 22.

-

Trace schema. A machine-readable JSON Schema specifies typed orchestration traces, and a companion example provides a minimal valid trace.

-

Validation. A dependency-free structural checker verifies required fields, event types, edge references, duplicate event identifiers, and non-negative cost fields for a trace JSON file. It is intended as a lightweight sanity check, not as a complete JSON Schema implementation.

These files do not make the literature review exhaustive, nor do they replace a multi-annotator systematic review. They do make the central claims of the paper mechanically inspectable: readers can check which entries support a taxonomy cell, regenerate the sparsity counts, and test whether a new benchmark log satisfies the minimal orchestration trace schema.

## 14  Conclusion

We surveyed reinforcement learning and post-training for LLM-based multi-agent systems as of May 4, 2026, organized around a single thesis: the field is usefully analyzed through orchestration traces rather than only through per-agent trajectories. §3 formalized this object as a working abstraction for taxonomy and auditability, using an event graph drawn from a dynamic-Dec-POMDP extension and stated two informal observations—credit diffusion under uniform credit and non-identifiability of orchestrator spawn decisions—that organize the rest of the paper. We stress that these observations are motivating arguments, not formal theorems; tight rates and full proofs are deferred to follow-up work.

Three taxonomies operationalize the thesis. §6 partitions the reward design space into eight families, with orchestration reward (R7) identified as the family that most directly targets spawn / delegate / aggregate decisions over multiple agent instances, and with the defining property that its useful weight is non-constant over training. §7 organizes entries along an eight-level credit- or signal-bearing-unit hierarchy—team / orchestrator / role / agent / turn / message / tool / token—and shows that explicit counterfactual message-level credit remains especially sparse, while newer agent-, role-, turn-, and orchestrator-level entries have started to fill in the surrounding taxonomy cells. §8 decomposes orchestration learning into five sub-decisions (when to spawn, whom to delegate to, how to communicate, how to aggregate, when to stop), and finds that within our curated pool no method explicitly trains the when-to-stop decision as an RL target.

The industrial–academic bridge is asymmetric. Kimi Agent Swarm publicly trains an orchestrator (PARL); OpenAI Codex and Anthropic Claude Code publicly document their deployment shape (parallel workflows, harness boundaries, dynamic spawn) but—to our knowledge—not whether the orchestration itself is an RL training target. Section 5 identified three engineering constraints—rollout cost scaling as $\sum_{i}(L_{i}c_{\text{tok}}+T_{i}c_{\text{tool}})+C_{\text{orch}}(K,|G|)$, the harness as a training-frozen interface, and per-decision signal decay under long traces—that together explain why academic methods mostly evaluated at $T\lesssim 10^{2}$ cannot be assumed to transfer to the disclosed Kimi-reported deployment envelope at $T\sim 10^{3}$–$10^{4}$. Even taking only Kimi as the public trained-orchestrator anchor, the open literature is still typically evaluated with fixed or moderate-size teams rather than hundreds of sub-agents, although WideSeek-R1 and MARTI-MARS2 now make academic width scaling more concrete. Closing this gap is less an algorithmic challenge than an infrastructural one: variable-shape replay (P10), rollout cost (P9), and MAS-native benchmarks at Kimi-like scale (P15) are engineering problems without solutions in our pool.

Three directions follow most directly from this survey:

-

A unified credit-assignment formalism over orchestration graphs. Table 11’s sparse cells (especially explicit counterfactual message credit and explicit orchestrator credit) are the most tractable research targets; compositionality across cells is an open question (P1–P4).

-

Benchmarks that measure coordination, not just success. (E1) is a weak discriminator of whether gains come from compute or from coordination. An open MAS-native benchmark covering (E1)–(E4) at Kimi-reported trace lengths would allow credit-assignment methods to be fairly compared for the first time (P15).

-

Safe and steerable long-horizon orchestrators. Mid-trace human intervention (P13), inter-agent prompt injection (P12), and sandbox-isolation under dynamic spawn (P14) are deployment-shaped problems that academic work has only begun to address through training-time adversarial games such as MAGIC [65]. They will not stay deferrable: Claw Groups [29] and the Anthropic trustworthy-agents framework [4] already treat them as first-class.

The field is now in the window where useful abstractions—the orchestration trace, the credit- and signal-bearing-unit hierarchy, the five-way orchestration sub-decision—can still be chosen cleanly, before conventions calcify around the first generation of industrial systems. This survey is an argument for why these abstractions are a productive starting point.

## References

- [1] Anonymous (2026) SAGE: multi-agent self-evolution for LLM reasoning. Note: ACL Rolling Review January 2026 submission; challenger, planner, solver, and critic co-evolve from a shared LLM backbone; under review; accessed 2026-05-04 External Links: Link Cited by: §1.1.

- [2] Anthropic Engineering (2026) Building a C compiler with a team of parallel Claudes. Note: https://www.anthropic.com/engineering/building-c-compiler2026-02-05; 16 parallel Claudes; accessed 2026-04-27 Cited by: §1.1, §4.2.3, Table 5.

- [3] Anthropic (2025) Creating custom sub-agents (claude code docs). Note: https://docs.anthropic.com/en/docs/claude-code/sub-agentsDocumentation; accessed 2026-04-27 Cited by: §1.1, §10.3, Table 15, §4.2.3, Table 5.

- [4] Anthropic (2025) Our framework for developing safe and trustworthy agents. Note: https://www.anthropic.com/news/our-framework-for-developing-safe-and-trustworthy-agents2025-08-04; accessed 2026-04-27 Cited by: §10.3, §10.4, §11.4, item 3, 2nd item, 3rd item.

- [5] D. S. Bernstein, R. Givan, N. Immerman, and S. Zilberstein (2002) The complexity of decentralized control of Markov decision processes. Mathematics of Operations Research 27 (4), pp. 819–840. Note: Original Dec-POMDP formalism; proves NEXP-complete External Links: Link Cited by: §2.1, §3.1, §3.2.

- [6] S. Chen, W. Zhang, T. Liu, et al. (2024) A survey on LLM-based multi-agent system: recent advances and new frontiers in application. arXiv preprint arXiv:2412.17481. Note: v2 dated 2025-01-07 External Links: Link Cited by: §1.1, §1.2, §1.3.

- [7] Y. Chen et al. (2026) Contextual counterfactual credit assignment for multi-agent reinforcement learning in LLM collaboration. arXiv preprint arXiv:2603.06859. Note: Counterfactual causal credit assignment at message level External Links: Link Cited by: §A.5, §1.1, §10.4, §11.1, §5.3, 3rd item, 2nd item, §7.5, Table 11, §8.2.3.

- [8] Y. Chen et al. (2025) Multi-agent evolve: LLM self-improve through co-evolution. arXiv preprint arXiv:2510.23595. Note: Proposer-Solver-Judge co-evolution; UIUC ulab External Links: Link Cited by: Table 5, Table 10.

- [9] I. Costa (2026) AgentSpawn: adaptive multi-agent collaboration through dynamic spawning for long-horizon code generation. arXiv preprint arXiv:2602.07072. Note: Runtime dynamic spawn + memory transfer; sole-author External Links: Link Cited by: §10.1, §11.4, Table 5, §8.2.1.

- [10] Y. Dang et al. (2025) Multi-agent collaboration via evolving orchestration. In Advances in Neural Information Processing Systems (NeurIPS), Note: Puppeteer central orchestrator; Tsinghua/OpenBMB ChatDev team External Links: Link Cited by: §A.1, Table 5, §5.3, §6.4, Table 10, Table 11, 1st item, §8.2.2.

- [11] C. S. de Witt, T. Gupta, D. Makoviichuk, V. Makoviychuk, P. H. S. Torr, M. Sun, and S. Whiteson (2020) Is independent learning all you need in the StarCraft multi-agent challenge?. arXiv preprint arXiv:2011.09533. Note: IPPO; independent PPO competitive on SMAC External Links: Link Cited by: §2.1.

- [12] E. Debenedetti, J. Zhang, M. Balunović, L. Beurer-Kellner, M. Fischer, and F. Tramèr (2024) AgentDojo: a dynamic environment to evaluate prompt injection attacks and defenses for LLM agents. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, Note: 97 realistic tasks, 629 security test cases; ETH SPY Lab External Links: Link Cited by: Figure 17, §10.1, §10.3, §10.3, Table 15, §10, §11.4.

- [13] DeepSeek-AI (2025) DeepSeek-R1: incentivizing reasoning capability in LLMs via reinforcement learning. Nature 645, pp. 633–638. Note: Rule-based RL unlocks long-CoT reasoning; R1 & R1-Zero External Links: Link Cited by: §2.2.

- [14] I. Evtimov, A. Zharmagambetov, A. Grattafiori, C. Guo, and K. Chaudhuri (2025) WASP: benchmarking web agent security against prompt injection attacks. arXiv preprint arXiv:2504.18575. Note: Meta FAIR; end-to-end web-agent prompt-injection benchmark External Links: Link Cited by: §10.1, §10.3, Table 15, §11.4.

- [15] L. Feng et al. (2026) Dr. MAS: stable reinforcement learning for multi-agent LLM systems. arXiv preprint arXiv:2602.08847. Note: Diagnoses GRPO instability in MAS; agent-wise normalization External Links: Link Cited by: §A.3, §1.1, §11.1, §11.3, §2.1, §2.2, §3.4, §5.3, 2nd item, §6.4, Table 10, 1st item, Table 11, 2nd item, 4th item.

- [16] J. N. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson (2018) Counterfactual multi-agent policy gradients. In Proceedings of the AAAI Conference on Artificial Intelligence, Note: COMA; counterfactual baseline for per-agent credit External Links: Link Cited by: 2nd item, §3.4.

- [17] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz (2023) Not what you’ve signed up for: compromising real-world LLM-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec), Note: Foundational paper on indirect prompt injection External Links: Document, Link Cited by: Figure 17, §10.1, §10.1, §10.3, Table 15, §10, §11.4.

- [18] Z. He, Z. Liu, P. Li, Y. R. Fung, M. Yan, J. Zhang, F. Huang, and Y. Liu (2025) Advancing language multi-agent learning with credit re-assignment for interactive environment generalization. In Conference on Language Modeling (COLM), Note: Introduces CollabUIAgents and multi-agent credit re-assignment for interactive UI/web environments; accessed 2026-04-27 External Links: Link Cited by: §1.1.

- [19] H. Hong et al. (2025) Multi-agent deep research: training multi-agent systems with M-GRPO. arXiv preprint arXiv:2511.13288. Note: Hierarchical GRPO; Ant Group External Links: Link Cited by: §A.2, §1.1, item 6, Table 5, Table 10, Table 11, 3rd item, §8.2.4.

- [20] Z. Hou et al. (2025) HALO: hierarchical autonomous logic-oriented orchestration for multi-agent LLM systems. arXiv preprint arXiv:2505.13516. Note: MCTS-based three-layer hierarchical MAS External Links: Link Cited by: Table 5, §8.2.1.

- [21] H. Hsu, J. Xu, N. Vichare, F. Carbone, M. Pajic, and G. Carenini (2026) DEPART: hierarchical multi-agent system for multi-turn interaction. Note: OpenReview ICLR 2026 submission; introduces HIMPO for alternating planner/executor post-training with role-specific rewards; accessed 2026-05-04 External Links: Link Cited by: §1.1, Table 5, 3rd item, Table 10, §7.5, Table 11, 3rd item.

- [22] M. Hu, Y. Zhou, W. Fan, Y. Nie, Z. Ye, B. Xia, T. Sun, Z. Jin, Y. Li, Z. Zhang, Y. Wang, Q. Ye, B. Ghanem, P. Luo, and G. Li (2025) OWL: optimized workforce learning for general multi-agent assistance in real-world task automation. In Advances in Neural Information Processing Systems (NeurIPS), Note: NeurIPS 2025 poster; trains a domain-agnostic planner in a hierarchical Workforce architecture; accessed 2026-04-27 External Links: Link Cited by: §1.1.

- [23] E. H. Jiang, L. Li, R. Sun, X. Liang, Y. Li, Y. Wu, H. Luo, H. Li, Z. Zhang, Z. Kang, K. Chang, and Y. N. Wu (2026) Agent Q-Mix: selecting the right action for LLM multi-agent systems through reinforcement learning. arXiv preprint arXiv:2604.00344. Note: QMIX-style CTDE for decentralized communication and topology decisions; accessed 2026-05-04 External Links: Link Cited by: §1.1, Table 10, Table 11, 1st item, §8.2.3.

- [24] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan (2024) SWE-bench: can language models resolve real-world GitHub issues?. In International Conference on Learning Representations (ICLR), Note: 2294 real GitHub issues from 12 Python repos External Links: Link Cited by: Table 12.

- [25] I. Kavathekar et al. (2025) TAMAS: benchmarking adversarial risks in multi-agent LLM systems. In ICML 2025 Multi-Agent Systems Workshop, Note: First adversarial robustness benchmark for MAS External Links: Link Cited by: §10.1, §10.3, Table 15, §11.4, 1st item, 2nd item, Table 12.

- [26] Z. Ke, A. Xu, Y. Ming, X. Nguyen, C. Xiong, and S. Joty (2025) MAS-Zero: designing multi-agent systems with zero supervision. arXiv preprint arXiv:2505.14996. Note: Inference-time self-evolved MAS design through meta-level design feedback and self-verification; accessed 2026-04-27 External Links: Link Cited by: §1.1, §6.4.

- [27] R. M. S. Khan, Z. Tan, S. Chen, P. Foulds, S. Yong, H. Liu, and T. Chen (2025) Agents under siege: breaking pragmatic multi-agent LLM systems with optimized prompt attacks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), Note: Permutation-invariant adversarial attack on multi-agent LLM topologies under bandwidth/latency constraints External Links: Link Cited by: Figure 17, §10.1, §10.3, Table 15, §11.4.

- [28] Kimi Team (2026) Kimi K2.5: visual agentic intelligence. Note: https://www.kimi.com/blog/kimi-k2-5.htmlTechnical report by Moonshot AI; arXiv 2602.02276; introduces Agent Swarm + PARL; accessed 2026-04-27 Cited by: §A.1, §1.1, §11.1, §4.2.1, Table 5, Table 5, Table 8, §6.2, Table 10, 1st item, Table 11, 1st item, §8.2.1, 1st item, §9.4, Table 12.

- [29] Kimi Team (2026) Kimi K2.6 tech blog. Note: https://www.kimi.com/blog/kimi-k2-62026-04-20; 300-agent coordination, Claw Groups; accessed 2026-04-27 Cited by: §A.1, §1.1, §11.1, item 3, §4.2.1, Table 5, Table 8, §5.1, 1st item, 3rd item.

- [30] S. Li et al. (2026) Experience as a compass: multi-agent RAG with evolving orchestration and agent prompts. arXiv preprint arXiv:2604.00901. Note: HERA; evolving orchestration policy for MAS-RAG External Links: Link Cited by: Table 10, Table 11.

- [31] Y. Li et al. (2026) Who deserves the reward? SHARP: shapley credit-based optimization for multi-agent system. arXiv preprint arXiv:2602.08335. Note: Shapley-value-based hierarchical credit assignment External Links: Link Cited by: §A.3, §1.1, §11.1, 2nd item, §6.4, Table 10, §7.5, Table 11.

- [33] M. L. Littman (1994) Markov games as a framework for multi-agent reinforcement learning. In Proceedings of the International Conference on Machine Learning (ICML), pp. 157–163. Note: Foundational stochastic/Markov-game formulation for MARL External Links: Link Cited by: §2.1, §3.1, §3.2.

- [34] B. Liu, S. Yu, Z. Liu, L. Guertler, P. Qi, D. Balcells, M. Liu, C. Tan, W. Shi, M. Lin, W. S. Lee, and N. Jaques (2026) SPIRAL: self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning. In International Conference on Learning Representations (ICLR), Note: ICLR 2026 poster; role-conditioned advantage estimation for online multi-turn multi-agent self-play; accessed 2026-05-04 External Links: Link Cited by: §1.1, 3rd item.

- [35] K. Liu et al. (2025) Reinforcement learning meets large language models: a survey of advancements and applications across the LLM lifecycle. arXiv preprint arXiv:2509.16679. Note: Fudan/Tongji/CUHK MMLab External Links: Link Cited by: §1.1, §1.3.

- [36] S. Liu et al. (2025) MarsRL: advancing multi-agent reasoning system via reinforcement learning with agentic pipeline parallelism. arXiv preprint arXiv:2511.11373. Note: Agentic pipeline-parallel RL External Links: Link Cited by: §A.4, §11.3, §4.2.2, §5.1, Table 10, Table 11, 1st item.

- [37] S. Liu C. Amato et al. (2025) LLM collaboration with multi-agent reinforcement learning. arXiv preprint arXiv:2508.04652. Note: v7 dated 2025-12-09; introduces MAGRPO External Links: Link Cited by: §A.3, §1.1, Table 10, Table 11.

- [38] S. Liu, T. Chen, R. Amiri, and C. Amato (2026) Learning decentralized LLM collaboration with multi-agent actor critic. arXiv preprint arXiv:2601.21972. Note: Introduces CoLLM-CC and CoLLM-DC actor-critic variants for decentralized LLM collaboration; accessed 2026-05-04 External Links: Link Cited by: §1.1, §11.1, Table 10, Table 11.

- [39] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch (2017) Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems (NeurIPS), Note: MADDPG; centralized critic, decentralized actor External Links: Link Cited by: 1st item, §3.3.

- [40] X. Luo et al. (2025) Agent Lightning: train any AI agents with reinforcement learning. arXiv preprint arXiv:2508.03680. Note: Microsoft Research; decouples agent execution from RL training External Links: Link Cited by: §A.6, §11.2, §11.3, 2nd item, §4.2.2, Table 5, Figure 8, §5.1, §5.2, 4th item, Table 10, Table 11, 2nd item.

- [41] T. Men, P. Cao, Z. Jin, Y. Chen, K. Liu, and J. Zhao (2025) A troublemaker with contagious jailbreak makes chaos in honest towns. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), Note: Contagious jailbreak that propagates through agent memory across non-complete-graph topologies External Links: Link Cited by: Figure 17, §10.1, §10.3, Table 15, §11.4.

- [42] G. Mialon, C. Fourrier, C. Swift, T. Wolf, Y. LeCun, and T. Scialom (2023) GAIA: a benchmark for general AI assistants. arXiv preprint arXiv:2311.12983. Note: 466 tool-use-heavy real-world questions External Links: Link Cited by: Table 12.

- [43] Z. Mo et al. (2025) Multi-agent tool-integrated policy optimization. arXiv preprint arXiv:2510.04678. Note: Single-LLM dual-role planner+worker; +18.38% over single-agent External Links: Link Cited by: §A.2, §1.1, §11.2, Table 5, 4th item, Table 10, Table 10, Table 11, 3rd item.

- [44] S. R. Motwani et al. (2025) MALT: improving reasoning with multi-agent LLM training. In Conference on Language Modeling (COLM), Note: Generator-verifier-refiner training with role-PRM (+14.14%) External Links: Link Cited by: §A.2, §11.2, item 6, Table 5, §5.3, §6.4, Table 10, Table 10, 2nd item, Table 11.

- [45] OpenAI (2025) Introducing codex. Note: https://openai.com/index/introducing-codex/; https://openai.com/index/introducing-the-codex-app/2025-05-16 launch post plus Codex app materials; cloud-native parallel software-engineering agent; accessed 2026-04-27 Cited by: §1.1, §10.3, §4.2.2, Table 5.

- [46] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe (2022) Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), Note: InstructGPT; canonical RLHF three-stage pipeline External Links: Link Cited by: §2.2.

- [47] C. Park et al. (2025) MAPoRL: multi-agent post-co-training for collaborative large language models with reinforcement learning. In Annual Meeting of the Association for Computational Linguistics (ACL), Note: MIT; first explicit post-training RL for collaboration External Links: Link Cited by: §A.3, §1.1, Table 10, Table 11.

- [48] Z. Peng, Y. Yao, K. Ma, S. Guo, Y. Li, Y. Zhang, C. Zhang, Y. Zhang, Z. Yu, et al. (2025) CriticLean: critic-guided reinforcement learning for mathematical formalization. arXiv preprint arXiv:2507.06181. Note: Trains a critic via SFT+RL to score Lean 4 formalizations; concrete instance of verifier-as-reward (R6) External Links: Link Cited by: 5th item, Table 10.

- [49] Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, S. Zhao, L. Hong, R. Tian, R. Xie, J. Zhou, M. Gerstein, D. Li, Z. Liu, and M. Sun (2024) ToolLLM: facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations (ICLR) Spotlight, Note: ToolBench + ToolLLaMA; 16k+ RapidAPI tools External Links: Link Cited by: Table 12.

- [50] T. Rashid, M. Samvelyan, C. Schroeder de Witt, G. Farquhar, J. Foerster, and S. Whiteson (2018) QMIX: monotonic value function factorisation for deep multi-agent reinforcement learning. In Proceedings of the International Conference on Machine Learning (ICML), Note: QMIX; monotonic mixing network External Links: Link Cited by: 1st item.

- [51] M. Salimi et al. (2026) Debate as reward: a multi-agent reward system for scientific ideation via RL post-training. arXiv preprint arXiv:2604.16723. Note: Multi-agent debate as reward signal External Links: Link Cited by: Table 5, 3rd item, Table 10, 2nd item, §8.2.3.

- [52] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017) Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. Note: PPO; clipped surrogate objective External Links: Link Cited by: §2.2.

- [53] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo (2024) DeepSeekMath: pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300. Note: Original source of GRPO External Links: Link Cited by: §2.2.

- [54] V. Subramaniam, Y. Du, J. B. Tenenbaum, A. Torralba, S. Li, and I. Mordatch (2025) Multiagent finetuning: self improvement with diverse reasoning chains. In International Conference on Learning Representations (ICLR), Note: Finetunes a society of language models using diverse reasoning chains generated through multiagent interaction; accessed 2026-04-27 External Links: Link Cited by: §1.1, §6.4.

- [56] P. Sunehag, G. Lever, A. Gruslys, W. M. Czarnecki, V. Zambaldi, M. Jaderberg, M. Lanctot, N. Sonnerat, J. Z. Leibo, K. Tuyls, and T. Graepel (2018) Value-decomposition networks for cooperative multi-agent learning. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (AAMAS), Note: VDN; additive value decomposition External Links: Link Cited by: 1st item.

- [57] K. Tran B. O’Sullivan et al. (2025) Multi-agent collaboration mechanisms: a survey of LLMs. arXiv preprint arXiv:2501.06322. Note: UCC Ireland External Links: Link Cited by: §1.1, §1.2, §1.3.

- [58] Z. Wan, Y. Li, X. Wen, Y. Song, H. Wang, L. Yang, M. Schmidt, J. Wang, W. Zhang, S. Hu, and Y. Wen (2025) ReMA: learning to meta-think for LLMs with multi-agent reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), Note: NeurIPS 2025 poster; multi-agent RL framework for meta-thinking with high-level meta-thinking and low-level reasoning agents; accessed 2026-04-27 External Links: Link Cited by: §1.1.

- [59] J. Wang, Y. Zhang, T. Kim, and Y. Gu (2020) Shapley Q-value: a local reward approach to solve global reward games. In Proceedings of the AAAI Conference on Artificial Intelligence, Note: Shapley-value credit assignment for cooperative MARL External Links: Link Cited by: 2nd item, §3.4.

- [60] P. Wang, Y. Wu, Z. Wang, J. Liu, X. Song, Z. Peng, K. Deng, C. Zhang, J. Wang, et al. (2025) MTU-Bench: a multi-granularity tool-use benchmark for large language models. In International Conference on Learning Representations (ICLR), Note: Five-granularity tool-use benchmark covering single/multi-turn and single/multi-tool scenarios External Links: Link Cited by: Table 12.

- [61] S. Wang, P. Li, Y. Fu, K. Liu, F. Li, Y. Liu, X. Sun, Z. Li, S. Zhao, J. Zhao, K. Tian, D. Li, J. Gao, Y. Zhang, Y. Chen, Y. Li, Z. Li, W. Zhang, P. Ye, S. Hu, L. Bai, B. Zhou, K. Zhang, and B. Qi (2026) MARTI-MARS${}^{2}$: scaling multi-agent self-search via reinforcement learning for code generation. arXiv preprint arXiv:2602.07848. Note: Multi-agent reinforced training and self-search scaling for code generation; accessed 2026-05-04 External Links: Link Cited by: §1.1, §11.1, §11.5, 1st item, §5.1, 2nd item, 1st item.

- [62] Z. Wang, J. Zheng, L. Yang, S. Zhou, X. Tang, Z. Fang, Z. Liu, D. Chen, Y. Li, and J. Bu (2026) Towards scalable lightweight GUI agents via multi-role orchestration. arXiv preprint arXiv:2604.13488. Note: Findings of ACL 2026; multi-role orchestration and RL for role-oriented cooperative exploration; accessed 2026-05-04 External Links: Link Cited by: §1.1, Table 5, Table 10.

- [63] J. Wei, Z. Sun, S. Papay, S. McKinney, J. Han, I. Fulford, H. W. Chung, A. T. Passos, W. Fedus, and A. Glaese (2025) BrowseComp: a simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516. Note: OpenAI 2025-04-17; 1266 hard browsing questions External Links: Link Cited by: Table 12.

- [64] T. Wei H. Ji et al. (2026) Agentic reasoning for large language models. arXiv preprint arXiv:2601.12538. Note: UIUC; 29-author team External Links: Link Cited by: §1.1.

- [65] X. Wen, Z. He, H. Qi, Z. Wan, Z. Ma, Y. Wen, T. Zheng, X. Xu, C. Lu, and Q. Zhang (2026) MAGIC: a co-evolving attacker-defender adversarial game for robust LLM safety. arXiv preprint arXiv:2602.01539. Note: Multi-turn attacker-defender multi-agent RL for safety alignment; accessed 2026-05-04 External Links: Link Cited by: §1.1, §10.1, §10.3, Table 15, §11.4, item 3, Table 10.

- [66] D. H. Wolpert and K. Tumer (2001) Optimal payoff functions for members of collectives. Advances in Complex Systems 4 (2–3), pp. 265–279. Note: Difference rewards / Wonderful Life Utility; foundational credit assignment External Links: Link Cited by: 2nd item, §3.4.

- [67] T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. Jing Hua, Z. Cheng, D. Shin, F. Lei, Y. Liu, Y. Xu, S. Zhou, S. Savarese, C. Xiong, V. Zhong, and T. Yu (2024) OSWorld: benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, Note: 369 real-computer tasks across Ubuntu/Windows/macOS External Links: Link Cited by: Table 12.

- [68] Z. Xu, Z. Xu, R. Zhang, C. Zhu, S. Yu, W. Liu, Q. Zhang, W. Ding, C. Yu, and Y. Wang (2026) WideSeek-R1: exploring width scaling for broad information seeking via multi-agent reinforcement learning. arXiv preprint arXiv:2602.04634. Note: Lead-agent/subagent MARL for broad information seeking and width scaling; accessed 2026-05-04 External Links: Link Cited by: §1.1, §11.5, 1st item, Table 5, §5.1, Table 10, Table 11, 2nd item, 1st item, §8.2.2.

- [69] X. Xue, Y. Zhou, G. Zhang, Z. Zhang, Y. Li, C. Zhang, Z. Yin, P. Torr, W. Ouyang, and L. Bai (2026) CoMAS: co-evolving multi-agent systems via interaction rewards. In International Conference on Learning Representations (ICLR), Note: ICLR 2026 poster; self-evolution through interaction-derived rewards and LLM-as-judge reward construction; accessed 2026-04-27 External Links: Link Cited by: §1.1, §6.4.

- [70] W. Yang and J. Thomason (2025) Learning to deliberate: meta-policy collaboration for agentic LLMs with multi-agent reinforcement learning. arXiv preprint arXiv:2509.03817. Note: Introduces MPDF and SoftRankPO for decentralized meta-cognitive actions Persist, Refine, and Concede; accessed 2026-04-27 External Links: Link Cited by: §1.1, 6th item.

- [71] H. Yao, L. Da, X. Liu, C. Fleming, T. Chen, and H. Wei (2026) LangMARL: natural language multi-agent reinforcement learning. arXiv preprint arXiv:2604.00722. Note: Agent-level language credit assignment and policy-gradient evolution in language space; accessed 2026-05-04 External Links: Link Cited by: §1.1, §11.1, Table 10, Table 11.

- [72] S. Yao, N. Shinn, P. Razavi, and K. Narasimhan (2024) $\tau$-bench: a benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045. Note: Sierra Research; retail/airline domains with policy adherence External Links: Link Cited by: Table 12.

- [73] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao (2023) ReAct: synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), Note: Interleaved reasoning+acting; agentic origin External Links: Link Cited by: §2.2.

- [74] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu (2022) The surprising effectiveness of PPO in cooperative multi-agent games. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, Note: MAPPO; PPO with centralized value for cooperative MARL External Links: Link Cited by: 1st item, §2.1, §3.3.

- [75] H. Yuan, Z. Xu, Z. Tan, X. Yi, M. Guang, K. Long, H. Hui, B. Li, X. Chen, B. Zhao, X. Zhang, C. Yu, and Y. Wang (2026) MARSHAL: incentivizing multi-agent reasoning via self-play with strategic LLMs. In International Conference on Learning Representations (ICLR), Note: ICLR 2026 poster; turn-level advantage estimation and agent-specific normalization for strategic self-play; accessed 2026-05-04 External Links: Link Cited by: §1.1, §11.1, 3rd item, §7.5, Table 11, 3rd item.

- [76] W. Yuan et al. (2026) Small model as master orchestrator: learning unified agent-tool orchestration with parallel subtask decomposition. arXiv preprint arXiv:2604.17009. Note: ParaManager; lightweight orchestrator unifying agent-tool action space External Links: Link Cited by: Table 10, §8.2.2.

- [77] Q. Zhan, Z. Liang, Z. Ying, and D. Kang (2024) InjecAgent: benchmarking indirect prompt injections in tool-integrated large language model agents. In Findings of the Association for Computational Linguistics: ACL, Note: 1,054 IPI test cases against tool-integrated agents External Links: Link Cited by: Figure 17, §10.1, §10.3, Table 15, §11.4.

- [78] A. Zhang, M. Dong, J. Liu, W. Zhang, Y. Wang, J. Yang, G. Zhang, T. Liu, et al. (2025) CodeCriticBench: a holistic code critique benchmark for large language models. arXiv preprint arXiv:2502.16614. Note: Two-task / multi-difficulty critique benchmark with fine-grained checklists, anchoring LLM-as-critic reward design External Links: Link Cited by: 5th item, Table 12.

- [79] C. Zhang, Y. Li, C. Xu, J. Liu, A. Liu, C. Zhou, K. Deng, D. Wu, G. Huang, et al. (2025) ArtifactsBench: bridging the visual-interactive gap in LLM code generation evaluation. arXiv preprint arXiv:2507.04952. Note: v2 dated 2025-09-29; MLLM-as-Judge with temporal screenshots over 1,825 visual-interactive code tasks External Links: Link Cited by: 5th item, Table 12.

- [80] G. Zhang, P. Torr, J. Wang, et al. (2025) The landscape of agentic reinforcement learning for LLMs: a survey. Transactions on Machine Learning Research (TMLR). Note: Covers 500+ works; v5 2026-04-17 External Links: Link Cited by: §1.1, §1.2, §1.3.

- [81] W. Zhao, M. Yuksekgonul, S. Wu, and J. Zou (2025) SiriuS: self-improving multi-agent systems via bootstrapped reasoning. In Advances in Neural Information Processing Systems (NeurIPS), Note: Builds and refines an experience library from successful reasoning trajectories for self-improving MAS; accessed 2026-04-27 External Links: Link Cited by: §1.1, §6.4.

- [82] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, U. Alon, and G. Neubig (2024) WebArena: a realistic web environment for building autonomous agents. In International Conference on Learning Representations (ICLR), Note: Self-hostable web environment across four site categories External Links: Link Cited by: Table 12.

- [83] K. Zhu, H. Du, Z. Hong, X. Yang, S. Guo, Z. Wang, Y. Qian, X. Tang, Z. Zhang, J. Wang, L. Gu, T. Xie, H. Ji, and J. You (2025) MultiAgentBench: evaluating the collaboration and competition of LLM agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), Note: MARBLE framework; collab+competition scenarios External Links: Link Cited by: 1st item, §9.4, Table 12.

- [84] J. Zou et al. (2025) Latent collaboration in multi-agent systems. arXiv preprint arXiv:2511.20639. Note: Training-free latent-space MAS; +14.6% over baselines External Links: Link Cited by: Table 5, §8.2.3, 3rd item.

## Appendix A Entry Cards: Core RL Methods and Anchors for LLM-MAS

This appendix gives one-card summaries of thirteen core methods, frameworks, and industrial anchors that are most frequently referenced by the credit-assignment taxonomy (Table 11). Each card uses a uniform layout: (1) one-line claim, (2) reward shape, (3) credit-assignment mechanism, (4) orchestration form, (5) headline empirical result, (6) key limitation. Cards are ordered by credit- or signal-bearing unit, from team/orchestrator to message/tool, mirroring the hierarchy of §7.1.

### A.1  Orchestrator-level credit

Puppeteer [10] NeurIPS 2025; Tsinghua / OpenBMB
Claim. A learned central orchestrator chooses which sub-agent takes the next turn, treating delegation as a learnable action. Reward. Team outcome (R1) only; orchestrator credit comes from a learned central critic over orchestrator decisions. Credit assignment. Orchestrator-level: a centralized critic scores each delegation against the trace return; sub-agent policies are frozen during orchestrator training. Orchestration. Centralized orchestrator + frozen sub-agents (Regime A, §8.3). Headline result / limitation. Among entries in our pool, the earliest explicit treatment of the orchestrator as the unit of RL training. Limitation: sub-agent skill is held fixed, so gains are bounded by the existing sub-agent pool.

Kimi PARL [28, 29] K2.5 technical report plus K2.6 deployment-envelope extension; Moonshot AI
Claim. K2.5 reports PARL training of a learned orchestrator that spawns up to $100$ sub-agents and coordinates up to $1{,}500$ reported steps / tool-call events. K2.6 extends the public deployment envelope to $300$ sub-agents and $4{,}000$ reported coordinated steps, but is not used here as an independent RL-training claim. Reward. Composite R7+R8: $r_{\text{perf}}+\lambda_{1}r_{\text{parallel}}+\lambda_{2}r_{\text{finish}}$, with both auxiliary weights reported in K2.5 as annealed (Figure 11). Credit assignment. Critical-Steps metric distinguishes real parallel progress from padded traces; functions as orchestrator-level credit. Orchestration. Parallel swarm with dynamic spawn; staged training (frozen-then-joint). Headline result / limitation. Public trained-orchestrator evidence in K2.5, with K2.6 used only for deployment-envelope pressure. Limitation: full algorithmic details are not public; reproducibility is limited.

### A.2  Role-level credit

MALT [44] COLM 2025; University of Oxford
Claim. Train a generator–verifier–refiner role triple end-to-end on reasoning tasks, with role-specific PRMs. Reward. Role-specific process reward (R3+R4); each role has its own PRM rubric. Credit assignment. Role-level: per-role advantage from per-role PRM, summed into policy gradient per role. Orchestration. Planner–executor–critic with explicit role separation. Headline result / limitation. $+14.14\%$ over single-agent baseline on reasoning tasks. Limitation: role rubric design is hand-engineered per task.

M-GRPO [19] arXiv (Ant Group); Nov 2025
Claim. Hierarchical GRPO that decouples planner (main agent) and sub-agent advantage estimation. Reward. Hybrid R8: top-layer team reward, bottom-layer sub-task reward, with separate baselines. Credit assignment. Role-level: hierarchical baseline—top and bottom layer compute advantages independently against their own group baselines. Orchestration. Centralized orchestrator + sub-agents on deep-research tasks. Headline result / limitation. Reported AIME $86.5\to 93.3$ when used as multi-agent deep research training. Limitation: assumes a fixed two-level hierarchy.

MATPO [43] arXiv (NTU); Oct 2025
Claim. A single LLM plays both planner and worker roles with role-specific tool integration; the same weights are trained for both roles via shared rollouts. Reward. Role-specific reward (R3) plus tool-use reward (R5) conditioned on downstream success. Credit assignment. Role-level: dual-role advantage—each rollout contributes policy gradient under both role labels with role-specific shaping. Orchestration. Planner–executor with single-LLM weight sharing. Headline result / limitation. $+18.38\%$ over single-agent baseline. Limitation: dual-role training requires careful balancing to avoid one role dominating.

### A.3  Agent-level credit

MAGRPO [37] arXiv (Northeastern, with C. Amato); Aug 2025
Claim. Cast LLM collaboration as a cooperative MARL problem and propose a multi-agent, multi-turn variant of GRPO. Reward. Shared team reward (R1). Credit assignment. Agent-level: group-relative advantage computed per agent within multi-agent rollouts. Orchestration. Centralized: writing / coding collaboration tasks. Headline result / limitation. Among entries in our pool, the earliest systematic formalization of LLM collaboration as cooperative MARL with a matched RL algorithm. Limitation: shared-reward setting; free-riding not directly addressed.

MAPoRL [47] ACL 2025; MIT
Claim. First post-co-training paradigm explicitly training collaboration behavior in LLMs (rather than as an emergent property of prompting). Reward. Shared team outcome (R1) broadcast to all participating agents. Credit assignment. Agent-level: PPO-style updates with team reward broadcast; no per-agent decomposition. Orchestration. Centralized; multiple LLMs trained jointly on collaborative tasks. Headline result / limitation. Establishes the post-training framing now standard in the field. Limitation: predates the credit-assignment literature it inspired.

Dr. MAS [15] arXiv (NTU); Feb 2026
Claim. Diagnose multi-agent GRPO instability and propose agent-wise advantage normalization as the fix. Reward. Shared team reward (R1). Credit assignment. Agent-level: instead of group-normalizing advantages across the rollout group, normalize per-agent within each rollout to prevent cross-agent variance from poisoning gradients. Orchestration. Centralized; general MAS workloads. Headline result / limitation. Establishes that naïve GRPO is unstable when used unchanged in multi-agent settings; agent-wise normalization restores convergence. Limitation: the fix is empirical—no theoretical guarantee.

SHARP [31] arXiv; Feb 2026
Claim. Apply Shapley-value credit allocation to multi-agent LLM systems with tool-augmented agents. Reward. Hybrid R8: global team reward $+$ Shapley marginal credit per agent $+$ tool-process reward. Credit assignment. Agent-level (Shapley) and tool-level: marginal contribution of each agent computed via Shapley sampling; tools receive process-style rewards. Orchestration. Hierarchical; tool-augmented multi-agent system. Headline result / limitation. Most principled credit-attribution method in the pool. Limitation: Shapley sampling cost grows combinatorially with agent set size; intractable at industrial team sizes (§11).

### A.4  Turn-level credit

MarsRL [36] arXiv; Nov 2025
Claim. Train a multi-reasoning-agent pipeline with RL using agentic pipeline parallelism so different rollout stages execute concurrently. Reward. Hybrid R8: per-stage rewards along the reasoning pipeline. Credit assignment. Turn-level (per-pipeline-stage): each stage has its own advantage signal. Orchestration. Hierarchical pipeline of reasoning agents. Headline result / limitation. Demonstrates that pipeline-parallel rollouts amortize the cost of multi-agent RL. Limitation: requires reasoning task to be cleanly factorable into stages.

Context-Folding [55] arXiv (ByteDance Seed / CMU); Oct 2025
Claim. Agent actively manages its own context by folding sub-trajectories back into the main trace, enabling long-horizon agent training. Reward. Hybrid R8: branch reward $\approx$ main reward $\pm 0.2$ scope adjustment per folded sub-trajectory. Credit assignment. Turn-level: each fold/unfold action receives advantage computed against a shared baseline with independent per-branch advantage. Orchestration. Hierarchical (orchestrator $\to$ branches $\to$ aggregation). Headline result / limitation. Shows that long-horizon training is feasible with explicit context management. Limitation: tied to an agent-specific harness; not framework-agnostic.

### A.5  Message-level credit

C3 [7] arXiv (HK PolyU); Mar 2026
Claim. The only retained entry in our pool that performs counterfactual message-level credit assignment for LLM multi-agent systems, using contextual counterfactual intervention. Reward. Shared team reward (R1). Credit assignment. Message-level: for each utterance, estimate counterfactual trace return under intervention (replacing or removing the message). Pivotal messages receive proportionally larger credit. Orchestration. Centralized; reasoning collaboration. Headline result / limitation. Only retained entry in our pool that explicitly estimates counterfactual message-level credit. Limitation: counterfactual estimation cost grows with trace length and message count.

### A.6  Framework-level (cross-cutting)

Agent Lightning [40] arXiv (Microsoft Research); Aug 2025
Claim. Generic RL training framework that decouples agent execution from the trainer; supports any agent harness. Reward. Hybrid R8: framework-level reward dispatch with per-agent and per-tool-call shaping. Credit assignment. Agent-level and tool-level: framework provides credit-assignment primitives; specific decomposition is application-defined. Orchestration. Harness-based; designed to wrap existing agent runtimes. Headline result / limitation. Aligns academic RL training with the harness boundary deployed in industrial systems (Codex, Claude Code). Limitation: a framework, not an algorithm; provides plumbing, not an answer to credit assignment per se.

## Appendix B Paper Pool Summary Table

Table 16 lists all $84$ entries in our paper pool, organized into manuscript evidence buckets rather than by the raw category field of the retained-entry CSV. These buckets separate focal LLM-MAS entries from supporting foundations, benchmarks, safety entries, and critic / verifier references so that the appendix follows the argument of the paper. The CSV remains the source of truth for machine-readable counts and controlled vocabulary; its category histogram is reported in Table 1. Columns abbreviate the taxonomy tags used throughout the paper; for the full $18$-column schema see the repository artifact. Entries are sorted within bucket by year, then by key.

Column legend. RL-rel: relevance to RL/post-training (Y = direct RL/post-training method, P = partial/framework/case-level relevance, N = not RL-centered). Reward: dominant family from Table 10 (shr = shared, ind = individual, role, proc = process, tool, dbt = debate, verif, orch, hyb = hybrid). Credit: finest credit level from §7.1 (tm, or, ro, ag, tn, msg, tool, tok). Orch: orchestration form (cntr = centralized, pec = planner-exec-critic, dbt = debate, swm = swarm, hier = hierarchical, hrn = harness). Scen: target scenario (cod = coding, web, rsh = research, mth = math, tl = tool use, gen = general).

*Table 16: Complete paper pool ($84$ entries), organized into eight manuscript evidence buckets. Entries carry one of three status labels in the Core? column: core marks entries central to the survey’s argument (including the thirteen method / framework / industrial-anchor cards in Appendix A, plus the most directly adjacent surveys); case marks industrial cases that motivate the survey’s framing but are not algorithmic contributions; supp marks entries that support the taxonomy (classical MARL, benchmarks, safety, single-agent foundations, critic / tool-use evaluations, remaining surveys) without being themselves central contributions. The core label is therefore a relevance flag for the survey’s argument, not a synonym for “RL method”—industrial cases are flagged separately, and several classical-MARL entries (e.g., COMA, Shapley-Q) remain supp despite being RL works. “–” denotes a field not applicable (e.g., credit granularity for pre-LLM classical works). The retained-entry CSV in the artifact repository is the authoritative source and carries nine additional columns omitted here for space; the bucket labels in this appendix are for readability and are not a replacement for the CSV’s controlled category field.*

| Key | Year / venue | RL-rel | Reward | Credit | Orch | Scen | Core? | One-liner |

| A. Focal LLM-MAS training, benchmark, and adjacent framework entries (40 entries) |

| magrpo | 2025 / arXiv | Y | shr | ag | cntr | gen | core | LLM collab as coop MARL; MAGRPO |

| marft | 2025 / arXiv | Y | hyb | ag | cntr | gen | core | Multi-agent reinforcement fine-tuning |

| m-grpo | 2025 / arXiv | Y | hyb | ro | hier | rsh | core | Hierarchical GRPO, decoupled planner/sub |

| agent-lightning | 2025 / arXiv | Y | hyb | ag | hrn | gen | core | Generic RL framework; exec–train decoupling |

| matpo | 2025 / arXiv | Y | role | ro | pec | tl | core | Dual-role planner+worker; tool-integrated PO |

| puppeteer | 2025 / NeurIPS | Y | orch | or | cntr | gen | core | Learned central orchestrator |

| halo | 2025 / arXiv | P | orch | ro | hier | gen | supp | MCTS-based 3-layer hierarchical MAS |

| marsrl | 2025 / arXiv | Y | hyb | tn | hier | mth | core | Agentic pipeline-parallel RL |

| malt | 2024 / COLM’25 | Y | role | ro | pec | mth | core | Generator-verifier-refiner w/ role-PRM |

| maporl | 2025 / ACL | Y | shr | ag | cntr | gen | core | Post-co-training RL for collaboration |

| latentmas | 2025 / arXiv | N | NA | NA | dbt | gen | supp | Training-free latent-space MAS |

| mae | 2025 / arXiv | Y | verif | ro | pec | gen | supp | Proposer-solver-judge co-evolution |

| context-folding | 2025 / ICLR’26s | Y | hyb | tn | hier | cod | core | Agent-managed context folding |

| dr-mas | 2026 / arXiv | Y | shr | ag | cntr | gen | core | Diagnose GRPO instability; agent-wise norm |

| c3 | 2026 / arXiv | Y | shr | msg | cntr | gen | core | Contextual counterfactual msg-level credit |

| sharp | 2026 / arXiv | Y | hyb | ag | hier | tl | core | Shapley-value credit allocation |

| debate-as-reward | 2026 / arXiv | Y | dbt | msg | dbt | rsh | core | Multi-agent debate as RL reward |

| paramanager | 2026 / arXiv | Y | orch | or | cntr | gen | core | Small-model master orchestrator |

| hera | 2026 / arXiv | P | orch | or | cntr | rsh | supp | Evolving orch policy for MAS-RAG |

| tamas | 2025 / ICML’25W | N | NA | NA | NA | gen | supp | Adversarial robustness benchmark for MAS |

| agentspawn | 2026 / arXiv | P | orch | or | swm | cod | supp | Runtime dynamic spawn + memory transfer |

| rema2025 | 2025 / NeurIPS | Y | hyb | ro | hier | mth | core | Meta-thinking + reasoning agents via MARL |

| collabuiagents2025 | 2025 / COLM | Y | proc | ag | cntr | web | core | Credit re-assignment for UI/web generalization |

| comas2026 | 2026 / ICLR | Y | dbt | ag | dbt | gen | core | Co-evolution via interaction rewards |

| owl2025 | 2025 / NeurIPS | Y | hyb | or | hier | rsh | core | Optimized Workforce planner learning |

| sirius2025 | 2025 / NeurIPS | P | proc | tn | hier | gen | supp | Bootstrapped reasoning experience library |

| multiagent-finetuning2025 | 2025 / ICLR | P | dbt | ag | dbt | mth | supp | Multiagent self-improvement via diverse chains |

| mas-zero2025 | 2025 / arXiv | N | NA | or | hier | gen | supp | Zero-supervision inference-time MAS design |

| learning-to-deliberate2025 | 2025 / arXiv | Y | hyb | tn | dbt | mth | core | Meta-policy deliberation actions + SoftRankPO |

| collm-maac2026 | 2026 / arXiv | Y | shr | ag | cntr | gen | core | Actor-critic decentralized LLM collaboration |

| wideseek-r1-2026 | 2026 / arXiv | Y | orch | or | hier | rsh | core | Width scaling with lead/subagent MARL |

| magic2026 | 2026 / arXiv | Y | dbt | ag | dbt | gen | core | Attacker-defender MARL for safety |

| marti-mars2-2026 | 2026 / arXiv | Y | hyb | ag | hier | cod | core | Multi-agent self-search RL for code |

| spiral2026 | 2026 / ICLR | Y | hyb | ro | dbt | gen | core | Online self-play with role-conditioned advantage |

| marshal2026 | 2026 / ICLR | Y | hyb | tn | dbt | gen | core | Strategic self-play with turn-level advantage |

| depart2026 | 2026 / OpenReview | Y | hyb | ro | hier | web | core | HIMPO planner/executor post-training |

| agent-qmix2026 | 2026 / arXiv | Y | hyb | ag | cntr | gen | core | QMIX topology and communication learning |

| langmarl2026 | 2026 / arXiv | Y | hyb | ag | cntr | gen | core | Language-space agent credit assignment |

| lamo2026 | 2026 / ACL Findings | Y | role | ro | hier | web | core | Lightweight GUI multi-role orchestration |

| sage2026 | 2026 / ARR | Y | verif | ro | pec | mth | core | Challenger-planner-solver-critic co-evolution |

| B. Related surveys used for gap analysis (5 entries) |

| survey-mas | 2024 / arXiv | P | NA | NA | NA | gen | core | LLM-MAS architecture survey |

| survey-collab | 2025 / arXiv | N | NA | NA | NA | gen | core | MAS collaboration mechanisms |

| survey-rl-meets-llm | 2025 / arXiv | Y | NA | NA | NA | gen | supp | RL across LLM lifecycle |

| survey-agentic-rl | 2025 / TMLR | Y | NA | NA | NA | gen | core | 500+ works on agentic RL |

| survey-agentic-reasoning | 2026 / arXiv | P | NA | NA | NA | gen | supp | Agentic reasoning roadmap |

| C. Industrial systems (cases) (6 entries) |

| kimi-k2-5 | 2026 / Tech rep. | Y | orch | or | swm | rsh | case | Agent Swarm + PARL |

| kimi-k2-6 | 2026 / Tech blog | P | NA | NA | swm | cod | case | Deployment-scale evidence; Claw Groups |

| openai-codex | 2025 / Blog | P | NA | NA | hrn | cod | case | Cloud-parallel SE agent |

| claude-code-subagents | 2025 / Docs | N | NA | NA | hrn | cod | case | Claude Code sub-agent API |

| anthropic-trustworthy | 2025 / Blog | N | NA | NA | NA | gen | case | Safe and trustworthy agent framework |

| anthropic-c-compiler | 2026 / Eng.blog | N | NA | NA | swm | cod | case | 16 parallel Claudes build C compiler |

| D. Classical MARL (conceptual toolkit) (10 entries) |

| markov-games1994 | 1994 / ICML | Y | ind | ag | NA | gen | supp | Markov games as MARL framework |

| difference-rewards2001 | 2001 / ACS | – | NA | – | NA | gen | supp | Difference rewards / WLU |

| dec-pomdp2002 | 2002 / MOR | N | NA | – | NA | gen | supp | Complexity of decentralized control |

| maddpg2017 | 2017 / NeurIPS | Y | ind | ag | NA | gen | supp | Multi-agent DDPG (CTDE) |

| coma2018 | 2018 / AAAI | Y | shr | ag | NA | gen | supp | Counterfactual multi-agent PG |

| vdn2018 | 2018 / AAMAS | Y | shr | ag | NA | gen | supp | Value decomposition networks |

| qmix2018 | 2018 / ICML | Y | shr | ag | NA | gen | supp | Monotonic value factorisation |

| ippo2020 | 2020 / arXiv | Y | ind | ag | NA | gen | supp | Independent PPO in StarCraft |

| shapley-q2020 | 2020 / AAAI | Y | shr | ag | NA | gen | supp | Shapley Q-value |

| mappo2022 | 2022 / NeurIPS D&B | Y | shr | ag | NA | gen | supp | Surprising effectiveness of MAPPO |

| E. Benchmarks cited in Sec. 9 (8 entries) |

| gaia2023 | 2023 / arXiv | N | NA | NA | NA | rsh | supp | General AI assistants benchmark |

| toolbench2023 | 2023 / ICLR’24 | N | NA | NA | NA | tl | supp | Tool-learning 16k+ APIs |

| swe-bench2024 | 2024 / ICLR | N | NA | NA | NA | cod | supp | Real-world GitHub issues |

| webarena2024 | 2024 / ICLR | N | NA | NA | NA | web | supp | Realistic web agent environment |

| tau-bench2024 | 2024 / arXiv | N | NA | NA | NA | tl | supp | Tool-agent-user interaction |

| osworld2024 | 2024 / NeurIPS D&B | N | NA | NA | NA | cod | supp | Multimodal computer-use tasks |

| browsecomp2025 | 2025 / arXiv | N | NA | NA | NA | web | supp | OpenAI browsing benchmark |

| multiagentbench2025 | 2025 / ACL | N | NA | NA | NA | gen | supp | MAS collab+competition benchmark |

| F. Single-agent RL & LLM-RL foundations (5 entries) |

| ppo2017 | 2017 / arXiv | Y | NA | NA | NA | gen | supp | PPO (Schulman et al.) |

| instructgpt2022 | 2022 / NeurIPS | Y | NA | NA | NA | gen | supp | RLHF / InstructGPT |

| react2023 | 2023 / ICLR | N | NA | NA | NA | gen | supp | Reasoning + acting prompting |

| deepseekmath2024 | 2024 / arXiv | Y | NA | NA | NA | mth | supp | DeepSeekMath; introduces GRPO |

| deepseek-r12025 | 2025 / Nature | Y | NA | NA | NA | gen | supp | Incentivizing reasoning via RL |

| G. Safety / Adversarial Robustness (cited in Sec. 10) (6 entries) |

| greshake2023 | 2023 / AISec | N | NA | NA | NA | gen | supp | Indirect prompt injection (foundational) |

| agentdojo2024 | 2024 / NeurIPS’24 D&B | N | NA | NA | hrn | tl | supp | 97 tasks + 629 security cases |

| injecagent2024 | 2024 / ACL Findings | N | NA | NA | NA | tl | supp | 1,054 IPI test cases |

| agents-under-siege2025 | 2025 / ACL | N | NA | NA | NA | gen | supp | Permutation-invariant topology attacks |

| tmcht2025 | 2025 / ACL | N | NA | NA | cntr | gen | supp | Contagious jailbreak via memory poisoning |

| wasp2025 | 2025 / arXiv | N | NA | NA | hrn | web | supp | Web-agent prompt-injection benchmark |

| H. Critic / tool-use evaluation cited in Sec. 6–9 (4 entries) |

| mtu-bench2025 | 2025 / ICLR | N | NA | NA | NA | tl | supp | 5-granularity tool-use benchmark |

| codecriticbench2025 | 2025 / arXiv | N | verif | NA | NA | cod | supp | Code critique benchmark; checklists |

| artifactsbench2025 | 2025 / arXiv | N | verif | NA | NA | cod | supp | MLLM-as-Judge over 1,825 visual code tasks |

| criticlean2025 | 2025 / arXiv | Y | verif | NA | NA | mth | supp | RL-trained critic for Lean 4 formalization |

Counts by manuscript evidence bucket. A: 40 entries (focal LLM-MAS training, benchmark, and adjacent framework entries); B: 5 entries (related surveys); C: 6 entries (industrial cases); D: 10 entries (classical MARL); E: 8 entries (benchmarks); F: 5 entries (single-agent foundations); G: 6 entries (safety / adversarial robustness); H: 4 entries (critic / tool-use evaluation). These buckets intentionally differ from the CSV category histogram because some entries play a different evidence role in the manuscript than their coarse source type would suggest. For machine-readable counts, use the artifact script and the CSV fields directly.

## Appendix C Artifact, Search Protocol, and Trace Schema

This appendix records the manuscript-level evidence protocol behind the paper pool. It is included to make the taxonomy easier to audit and extend. The protocol is quasi-systematic: it exposes the search families, screened records, exclusion stages, tag schema, and borderline decisions, but it does not claim PRISMA-level reproducibility because raw database exports, deduplication logs, and multi-annotator agreement are not provided.

The supplementary artifact snapshot and repository (https://github.com/xxzcc/awesome-llm-mas-rl), with repository-path normalization between the manuscript bundle (artifact/) and the GitHub layout (scripts/, trace-schema/, and docs/), turn the appendix into a reusable object rather than only prose. They contain a retained-entry CSV, an exclusion-log CSV, a statistics script that regenerates all corpus counts and cross-tabs, a machine-readable orchestration-trace schema used in §9.4, a minimal valid trace, and a dependency-free trace validator.

### C.1  Search strings and screening counts

The search cutoff for this version is May 4, 2026. Searches were performed over arXiv, ACL Anthology, OpenReview, Semantic Scholar / citation links, official project pages, company technical reports, and product documentation. The representative query families were:

-

multi-agent LLM $\times$ reinforcement learning

-

multi-agent LLM $\times$ post-training

-

multi-agent LLM $\times$ credit assignment

-

LLM agent $\times$ orchestration / agent swarm / dynamic spawn

-

tool-use LLM $\times$ reinforcement learning

-

LLM agent $\times$ prompt injection / safety / jailbreak

-

multi-agent reinforcement learning $\times$ communication / credit assignment

| Screening stage | Records | Notes |

| Candidate records considered | 116 | Union of seed surveys, keyword search, citation following, and public-system audit. |

| Retained entries | 84 | Tagged in the retained-entry CSV with 18 fields and cited in the manuscript. |

| Excluded records | 32 | Logged in the exclusion-log CSV; all rows include a public identifier or documentation handle and URL. |

| Abstract-screen exclusions | 16 | Excluded because the abstract/public page did not expose LLM-MAS RL, credit, orchestration, benchmark, safety, or load-bearing formalism. |

| Full-text-screen exclusions | 9 | Read beyond abstract but not retained as taxonomy evidence. |

| Artifact-screen exclusions | 6 | Documentation, code, or product pages without enough methodological detail for coding. |

| Duplicate / overlapping signal | 1 | Excluded because the retained pool already contained the load-bearing benchmark/framework signal. |

*Table 17: Screening counts for the $116$ audited records. Counts are manuscript audit counts, not raw search-engine result counts.*

### C.2  Tag schema

Each retained row in the retained-entry CSV has 18 fields: key, title, first_author, affiliation, year, arxiv_id, venue, url, category, is_rl, reward_type, credit_granularity, orchestration_form, scenario, is_core, one_liner, verified, and notes. The most important controlled fields are:

| Field | Controlled values / interpretation |

| category | rl_method, survey, benchmark, framework, industry, classical_marl. |

| is_rl | yes for explicit RL/post-training, partial for self-evolution, RL-adjacent optimization, or industrial deployment evidence that constrains RL design without independently disclosing a training objective, no for background/safety/benchmark/system entries. |

| reward_type | One of the reward families R1–R8, or NA when the entry is not a reward-design method. |

| credit_granularity | Finest level at which the entry exposes a reward, credit, or design signal: token, turn, message, tool, agent, role, orchestrator, team, or NA. Explicit credit-assignment mechanisms are a narrower subset discussed in §7; for example, a message-level debate reward is tagged at the message level, while C3 is the retained entry with counterfactual message-level credit. |

| orchestration_form | centralized, planner–executor–critic, debate, swarm, hierarchical, harness, or NA. |

| verified | yes when bibliographic metadata and public artifact were checked; partial when the entry depends on a submission/project page or evolving public material. |

*Table 18: Tagging schema used by the retained pool. Ambiguous entries are intentionally marked partial rather than forced into binary labels.*

##### Partial and evolving evidence.

Eight retained entries have verified=partial: ReMA, CollabUIAgents, CoMAS, OWL, SiriuS, Multiagent Finetuning, DEPART, and SAGE. These entries were added during the coverage audit from OpenReview or project-page material and are used to populate adjacent taxonomy cells, not as sole support for the paper’s central claims. Kimi K2.6 is instead verified=yes but is_rl=partial: the public material is stable enough to support deployment-envelope claims, while its row is not used as an independent RL-training claim.

### C.3  Borderline decisions

| Record | Decision | Reason |

| AutoGen / CAMEL / MetaGPT | exclude | Important LLM-MAS frameworks, but the screened records do not provide RL/post-training or credit-assignment mechanisms used as load-bearing evidence here. |

| MAS-Zero | retain | Not RL in the strict sense, but directly relevant to zero-supervision MAS design and orchestrator search. |

| SiriuS / Multiagent Finetuning | retain | Self-evolution and interaction-generated training signals are adjacent to RL and affect the reward taxonomy. |

| Kimi K2.5 | retain | Public industrial source explicitly discloses RL training of the orchestrator; reproducibility remains limited. |

| Codex / Claude Code | retain as cases | Used only for deployment-shape and harness evidence, not as public multi-agent RL training evidence. |

| AgentBench / AgentBoard | exclude | Strong agent benchmarks, but less directly tied to MAS-native E2–E4 instrumentation than retained benchmark entries. |

*Table 19: Representative borderline inclusion/exclusion decisions. These examples are included so that future versions can revise the pool consistently rather than silently changing the taxonomy boundary.*

### C.4  Evidence matrix

| Source class | Alg. | Train | Deploy | Scale | Reprod. |

| Peer-reviewed / arXiv methods | yes | partial | partial | partial | partial |

| OpenReview submissions / posters | partial | partial | partial | partial | limited |

| Company technical reports | partial | partial | yes | yes | limited |

| Product docs / launch blogs | no | no | yes | partial | no |

| Engineering case studies | no | no | yes | partial | no |

| Benchmarks / leaderboards | no | no | partial | partial | partial |

*Figure 18: Evidence-level matrix used when interpreting the corpus. “Alg.” denotes support for algorithmic mechanism claims; “Train” denotes public training-objective or post-training evidence; “Deploy” denotes deployment-shape evidence; “Scale” denotes public scale or horizon evidence. This matrix prevents product documentation from being treated as equivalent to reproducible algorithmic evidence.*

### C.5  Claim-to-artifact ledger

Table 20 records how the main empirical claims in the survey can be checked against the artifact. The ledger is intentionally narrow: it ties claims to retained-pool fields, scripted counts, or explicitly bounded industrial evidence, and it does not turn the curated pool into a field-wide prevalence estimate.

| Claim | Artifact check | Boundary |

| Message-level credit is sparse | Message-level tag count is two; only C3 is counterfactual message credit | Tag count includes message-level reward/signal, not only explicit credit mechanisms. |

| Orchestrator-level credit is sparse | Orchestrator-level tag count is eight after K2.6 is treated as deployment evidence and WideSeek-R1 is added | Explicit RL credit mechanisms are narrower than orchestrator-level design or evolution signals. |

| Kimi provides the public trained-orchestrator anchor | K2.5 is tagged as explicit RL with orchestration reward; K2.6 is tagged as partial / NA | K2.6 is used for deployment-envelope pressure, not as an independent training claim. |

| Open MAS-native evaluation remains incomplete | Benchmark rows and Table 12; trace schema and reporting checklist in §9.4 | The claim is restricted to open, auditable retained entries under the stated protocol. |

| Trace reporting is mechanically inspectable | Trace schema, example trace, and structural checker | The checker validates core structural constraints, not the full JSON Schema standard. |

*Table 20: Claim-to-artifact ledger. Each row identifies the artifact object that supports a central survey claim and the boundary on that claim.*

Because credit_granularity records the finest level at which an entry exposes a reward, credit, or design signal, it is broader than “explicit counterfactual credit.” Table 21 therefore lists the retained rows behind the two sparsest credit cells.

| Tagged row | Tag rationale | Mechanism boundary |

| puppeteer | Learned central critic over orchestrator delegation. | Explicit RL credit. |

| paramanager | Unified agent/tool orchestration action space. | Orchestrator-level design signal; not counterfactual credit. |

| hera | Evolving orchestration policy and prompts. | Evolution signal over orchestration choices. |

| agentspawn | Runtime spawn decisions and memory transfer. | Runtime design signal; no disclosed RL credit estimator. |

| kimi-k2-5 | PARL with Critical-Steps reward for Agent Swarm. | Explicit public training signal; full traces are not released. |

| owl2025 | Planner/workforce optimization for modular agents. | Planner-level training signal. |

| mas-zero2025 | Meta-level MAS design feedback and verification. | Non-RL design-search signal. |

| wideseek-r1-2026 | Lead-agent/subagent width scaling with MARL. | Explicit orchestration-training signal, but not dynamic-spawn counterfactual credit. |

| c3 | Counterfactual causal credit at message level. | Explicit counterfactual message credit. |

| debate-as-reward | Debate messages supply a reward signal. | Message-level reward signal; not counterfactual credit. |

*Table 21: Rationale for sparse credit-granularity tags. The table makes the distinction between tag level and explicit credit mechanism auditable for the eight orchestrator-tagged rows and two message-tagged rows.*

### C.6  Scripted meta-analysis

The headline counts in Table 1 and the reward–credit cross-tab in Table 2 are generated from the CSV artifact rather than hand-entered during analysis. Running the repository statistics script prints retained/excluded counts, controlled-field histograms, and three cross-tabs: reward_type $\times$ credit_granularity, orchestration_form $\times$ credit_granularity, and category $\times$ verified. The first cross-tab is reproduced in the introduction because it directly supports the paper’s claim that message-level and orchestrator-level tags are sparse, and that explicit message/orchestrator credit mechanisms are rarer still.

| Orchestration form | NA | agent | msg. | orch. | role | turn |

| centralized | 0 | 14 | 1 | 3 | 0 | 0 |

| hierarchical | 0 | 2 | 0 | 3 | 5 | 3 |

| debate | 1 | 3 | 1 | 0 | 1 | 2 |

| swarm | 2 | 0 | 0 | 2 | 0 | 0 |

| harness | 2 | 1 | 0 | 0 | 0 | 0 |

| planner–executor–critic | 0 | 0 | 0 | 0 | 4 | 0 |

| NA | 31 | 3 | 0 | 0 | 0 | 0 |

*Table 22: Orchestration-form by credit-granularity cross-tab generated from the artifact. Centralized entries dominate agent-level credit, while swarm entries split between deployment-shape evidence and orchestrator-level signals. This supports the paper’s claim that topology constrains which credit unit can be made explicit.*

### C.7  Machine-readable trace schema

The JSON Schema in the artifact repository encodes the minimal trace object used by §9.4. It requires trace_id, task_id, events, edges, rewards, and costs. Event types include orchestrator_decision, spawn, despawn, message, tool_call, tool_result, return, aggregate, human_intervention, and safety_event. Edges distinguish temporal, causal, spawn, message, tool-dependency, return, aggregation, and safety-flow relations.

The schema deliberately stores prompt/tool content through content_ref rather than requiring raw content. This lets a benchmark release replayable trace topology, reward channels, cost metadata, and safety-flow information while redacting private prompts, credentials, or tool outputs. The goal is not to standardize every agent framework, but to define the minimum object needed to recompute the evaluation quantities in Table 13 and the reporting checklist in Table 14. The example trace can be checked with the repository validator.

### C.8  Artifact update protocol

Future updates should add new records in three steps: (1) append the candidate to the exclusion-log CSV with identifier, URL, screening stage, and exclusion reason; (2) promote it to the retained-entry CSV only if it changes a reward, credit, orchestration, evaluation, safety, industrial-evidence, or formalism cell; (3) update Figure 3, Table 1, Table 2, and the appendix summary table by rerunning the repository statistics script if the promoted entry changes a controlled tag count. This keeps the pool curated while making curation decisions auditable.

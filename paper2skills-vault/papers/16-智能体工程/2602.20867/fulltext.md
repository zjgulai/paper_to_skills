<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2602.20867
     paper_id : 2602.20867
     source   : https://arxiv.org/html/2602.20867v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# SoK: Agentic Skills — Beyond Tool Use in LLM Agents

Yanna Jiang1, Delong Li1, Haiyu Deng1, Baihe Ma1, Xu Wang1, Qin Wang1,2, Guangsheng Yu1 Affiliation: 1University of Technology Sydney $|$2CSIRO Data61

###### Abstract

Agentic systems increasingly rely on reusable procedural capabilities, a.k.a., agentic skills, to execute long-horizon workflows reliably. These capabilities are callable modules that package procedural knowledge with explicit applicability conditions, execution policies, termination criteria, and reusable interfaces. Unlike one-off plans or atomic tool calls, skills operate (and often do well) across tasks.

This paper maps the skill layer across the full lifecycle (discovery, practice, distillation, storage, composition, evaluation, and update) and introduces two complementary taxonomies. The first is a system-level set of seven design patterns capturing how skills are packaged and executed in practice, from metadata-driven progressive disclosure and executable code skills to self-evolving libraries and marketplace distribution. The second is an orthogonal representation $\times$ scope taxonomy describing what skills are (natural language, code, policy, hybrid) and what environments they operate over (web, OS, software engineering, robotics).

We analyze the security and governance implications of skill-based agents, covering supply-chain risks, prompt injection via skill payloads, and trust-tiered execution, grounded by a case study of the ClawHavoc campaign in which nearly 1,200 malicious skills infiltrated a major agent marketplace, exfiltrating API keys, cryptocurrency wallets, and browser credentials at scale. We further survey deterministic evaluation approaches, anchored by recent benchmark evidence that curated skills can substantially improve agent success rates while self-generated skills may degrade them. We conclude with open challenges toward robust, verifiable, and certifiable skills for real-world autonomous agents.

## I Introduction

Large language model (LLM) agents have advanced rapidly from single-turn question answering to multi-step autonomous systems that browse the web [1], write/debug software [2, 3], orchestrate tools in sequence [4, 5], and collaborate as multi-agent teams [6, 7]. Yet a fundamental inefficiency persists: each new task forces the agent to re-derive an execution strategy from scratch. A coding agent that has successfully debugged a null-pointer exception a hundred times still approaches the hundred-and-first as if it were novel. The procedural knowledge gained from experience disappears at the end of every context window.

This observation motivates the central abstraction of this paper: the agentic skills. We define a skill as a reusable, callable module that encapsulates a sequence of actions or policies enabling an agent to achieve a class of goals under recurring conditions. Skills differ from tools (atomic primitives with fixed interfaces), plans (one-time reasoning scaffolds), and episodic memories (stored observations) in that they are simultaneously executable, reusable, and governable. A skill carries its own applicability conditions, termination criteria, and callable interface, making it a first-class unit of procedural knowledge.

The notion is not new in isolation. Cognitive architectures such as ACT-R [8] and Soar [9] formalized procedural memory decades ago. Reinforcement learning (RL) has long studied option frameworks and hierarchical policies [10]. What is new is the convergence of these ideas in the LLM agent [11, 12]. Skills manifest in forms ranging from natural-language playbooks and executable Python scripts to marketplace-distributed plugins. The diversity of representations calls for systematization.

Existing surveys cover LLM agents broadly [13, 14, 15, 16, 17, 18, 19], focus on tool use [20, 21, 22], or address multi-agent coordination [23]. None adopt a skill-centric lens to trace the full lifecycle from acquisition to governance. We fill this gap.

Contributions. This Systematization of Knowledge (SoK) offers six contributions:

-

a unified definition of agentic skills (§II), formalized as $S=(C,\pi,T,R)$, with precise boundary conditions separating skills from tools, plans, and memory.

-

a skill lifecycle model (§IV) mapping the stages from discovery through evaluation and update, with a summary linking representative systems to lifecycle stages.

-

a seven-pattern design taxonomy (§V) for how skills are packaged, loaded, and executed in real systems.

-

an orthogonal representation $\times$ scope taxonomy (§V-J) describing what skills are and what environments they act over, integrated with the seven patterns.

-

a security and governance analysis (§VII) covering threat models, trust tiers, a pattern-specific risk matrix, and an anchor case study of the ClawHavoc marketplace supply-chain attack.

-

an evaluation framework (§VIII) with metrics, benchmark mapping, and an anchor case study demonstrating that curated skills outperform self-generated ones.

Reading map. §II defines the core abstraction. §III describes our systematic methodology. §IV introduces the lifecycle model. §V presents the seven design patterns and the representation$\times$scope taxonomy. §VI covers skill acquisition and composition. §VII analyzes security and governance, including the ClawHavoc case study. §VIII surveys evaluation. §IX discusses cross-cutting observations and limitations. §X outlines open challenges. §XI concludes our work.

## II What Is an Agentic Skill?

### II-A Formal Definition

We ground the concept of an agentic skill in a four-tuple formalization that captures the essential properties distinguishing skills from related abstractions.

###### Definition 1 (Agentic skills).

Let an agent interact with environment $E$ via action space $A$, observation space $O$, and goal space $G$. Let $H=(o_{1},a_{1},\ldots,o_{t-1},a_{t-1})$ denote the interaction history up to the current step. An agentic skill is a tuple

$S=(C,\pi,T,R)$ | | | | (1) |

where:

-

$C:O\times G\rightarrow\{0,1\}$ is the applicability condition, a predicate over observations and the agent’s current goal that determines whether the skill is appropriate for the current context;

-

$\pi:O\times H\rightarrow A\cup\Sigma$ is the executable policy, a mapping from observations and interaction history to actions or skill invocations from the skill library $\Sigma$, which may be implemented as natural-language instructions, executable code, a learned controller, or a hybrid thereof. When $\pi$ selects a skill $s\in\Sigma$ rather than a primitive action $a\in A$, hierarchical composition arises (§VI-F1), mirroring the option-subroutine structure in the RL options framework [10];

-

$T:O\times H\times G\rightarrow\{0,1\}$ is the termination condition, specifying when the skill has completed (successfully or not) relative to the current goal;

-

$R=(\textit{name},\textit{params},\textit{returns})$ is the reusable callable interface, a metadata and contract component specifying the skill’s callable signature (name, parameter schema, return type) for programmatic invocation by the agent, other skills, or external orchestrators.

$G$ may be encoded within $O$ (e.g., as a task prompt in the observation) or passed as an explicit parameter; we make it explicit here for clarity. Implementations often compute soft applicability scores $C:O\times G\rightarrow[0,1]$ and apply a threshold; we present the binary form as a simplifying convention that captures the essential gating logic. In orchestrator-managed architectures, $C$ and $T$ may be externally provided rather than skill-internal; the 4-tuple then describes the logical interface regardless of where each function is implemented.

We argue these four components form a useful minimal schema that captures the properties distinguishing skills from related abstractions. Removing $C$ yields a policy that cannot self-select; removing $T$ produces a policy that cannot compose (callers do not know when to resume); removing $R$ yields internal knowledge that cannot be invoked programmatically; and removing $\pi$ leaves metadata without executability. The formalization is deliberately representation-agnostic: $\pi$ can be a prompt template, a Python function, a reinforcement-learning policy, or a combination.

This formalization parallels the options framework $(I,\pi,\beta)$ of Sutton et al. [10], where our $C$ corresponds to the initiation set $I$ and $T$ to the termination condition $\beta$. The interface $R$ builds on the options framework by making skills explicitly invocable, which is necessary for runtime composition. RL options are instead chosen implicitly by a meta-policy, so they do not address this requirement. Fig.1 shows the resulting four-component architecture.

*Fig. 1: Internal anatomy of an agentic skill. Observations $O$ enter the applicability gate $C$; the policy $\pi$ produces actions $A$; the termination condition $T$ determines whether to continue or halt. The interface $R$ wraps the entire module as a callable API boundary. Goal $G$ is typically encoded in observations $O$ or passed as a separate task parameter; for visual simplicity, we show $O$ as the single input.*

### II-B Skills versus Related Abstractions

We compares agentic skills with four related concepts (Table I): unit of reuse, execution semantics, verification surface, composability, and governance surface.

*TABLE I: Concept unification: agentic skills versus related abstractions in LLM agent systems.*

| Abstraction | Unit of Reuse | Execution Semantics | Verification Surface | Composability | Governance Surface |

| Tool | Single API call | Stateless, single invocation | Input/output schema | Sequential chaining | Permission per tool |

| Plan | Task decomposition | One-time reasoning scaffold | Step consistency | Hierarchical decomposition | N/A (ephemeral) |

| Episodic memory | Stored observation | Retrieval, no direct execution | Relevance, recency | Indirect (informs reasoning) | Access control on store |

| Prompt template | Text fragment | Injected into context window | Output quality | String concatenation | Template authorship |

| Agentic skill | Procedural module | Callable workflow with termination | Outcome correctness, safety | Hierarchical, DAG, recursive | Trust tier, sandboxing, provenance |

Tools. A tool is an atomic primitive (e.g., a web-search API or a file-write function) with a fixed interface and no internal decision-making. Prior work such as Toolformer [22] shows that LLMs can learn to invoke tools autonomously, but such behavior typically remains at the level of single calls. A skill may invoke tools, but extends them with applicability logic, multi-step sequencing, and explicit termination criteria. Conceptually, the distinction resembles that between a system call and a library routine in software engineering.

Plans. A plan is a reasoning artifact produced by the agent to decompose a task into sub-goals. Plans are typically one-time, session-scoped, and not directly executable without further interpretation. Skills, by contrast, persist across sessions, carry executable policies, and expose callable interfaces. A plan may select skills, but a skill is not a plan.

Memory. Episodic and semantic memory systems store observations and facts for later retrieval [24, 25, 26]. Skills are a form of procedural memory: they encode how to act, not what happened. The relationship between declarative memory and procedural skills in LLM agents mirrors the distinction drawn in cognitive psychology between knowing-that and knowing-how [8].

Prompt templates. Prompt templates are static text fragments injected into the context window [27]. They lack applicability conditions, termination logic, and callable interfaces. A skill may contain a prompt template as part of its policy $\pi$, but a template alone does not constitute a skill.

Classical AI planning formalisms. The skill abstraction also connects to classical AI planning. In Hierarchical Task Networks (HTNs) [28], methods decompose tasks into sub-tasks with preconditions, mirroring our hierarchical composition (§VI-F1). BDI (Belief-Desire-Intention) architectures [29] use reusable plan recipes with context conditions, which align with $C$ and $\pi$. STRIPS/PDDL actions [30] make preconditions and effects explicit, which anticipates our applicability and termination conditions. The main difference is representational: LLM-based skills act on natural-language observations and can encode policies as NL instructions or hybrid artifacts, while classical formalisms assume symbolic state. We keep the formalization representation-agnostic to connect these lines of work.

### II-C Skills as Procedural Memory

Cognitive science provides a useful lens for understanding why skills matter. Anderson’s ACT-R theory [8] distinguishes declarative memory (facts and episodes) from procedural memory (production rules that encode condition–action pairs). Experts differ from novices less in what they know than in the richness of their procedural repertoire: action patterns that trigger automatically when conditions are met, freeing working memory for higher-level reasoning.

LLM agents face an analogous challenge. Without a skill layer, every task requires the agent to reason from first principles within a limited context window, consuming tokens to re-derive procedures that could be stored and retrieved. Skills serve as the procedural memory of the agent, compressing learned procedures into reusable modules that reduce the cognitive load on the model’s context window, analogous to how chunking in human expertise compresses multi-step procedures into single retrievable units [31].

This framing has a practical implication: the value of a skill is not merely convenience but reliability. A curated skill that has been verified across multiple contexts is more reliable than an ad-hoc plan generated on the fly, for the same reason that a tested library function is more reliable than inline code. Recent empirical evidence supports this intuition: the SkillsBench benchmark [32] demonstrates that curated skills raise agent pass rates by 16.2 percentage points on average, while self-generated skills degrade performance by 1.3 pp, encoding incorrect or overly specific heuristics. Notably, a smaller model equipped with curated skills can outperform a larger model operating without them. One interpretation is that procedural memory serves as an efficiency multiplier and partially substitute for model scale.

*Fig. 2: The agentic skill lifecycle. Solid arrows indicate the primary forward path; dashed arrows indicate feedback loops for refinement and retirement. Each stage corresponds to a body of research surveyed in this paper.*

## III Methodology

This section describes the systematic process used to collect and analyze the literature on agentic skills in LLM agent systems, and the methodology through which our taxonomies were developed.

### III-A Literature Search and Selection

We ran a structured search across six databases: Google Scholar, Semantic Scholar, DBLP, ACM Digital Library, IEEE Xplore, and arXiv. We used keyword queries (including agent skills, skill learning LLM, reusable agent behaviors, procedural knowledge agents, tool composition LLM, agent libraries, and hierarchical agent policies) and followed citations forward and backward from seed papers (Voyager [33], ReAct [34], Reflexion [35], SWE-agent [2]).

The search covered publications from January 2020 through February 2025 for LLM agent systems. We include one concurrent work, SkillsBench [32] (February 2026), as a notable exception given its direct relevance to skill evaluation; all other primary sources fall within the stated window. Foundational works from cognitive science [8, 31], cognitive architectures [9], and reinforcement learning [10] were included regardless of date to ground the skill abstraction in established theory.

Inclusion criteria. A paper was included if it satisfies at least one of the following: (i) it introduces, implements, or evaluates reusable procedural capabilities for LLM-based or language-conditioned agents; (ii) it addresses at least one lifecycle stage (discovery, refinement, distillation, storage, retrieval, execution, or evaluation) of agent procedural knowledge; or (iii) it provides a benchmark environment in which agent skills can be measured.

Exclusion criteria. We excluded works that focus exclusively on single-turn tool calling without procedural composition, pure prompt engineering without skill persistence or reuse, and multi-agent coordination papers that do not involve a skill abstraction. We also excluded papers focusing solely on fine-tuning for instruction following without an explicit skill representation.

### III-B Corpus and Analysis

The initial search yielded approximately 180 candidate papers. After applying the inclusion and exclusion criteria, we retained 65 papers for detailed analysis. Of these, 24 systems are analyzed in depth through the mapping tables (Tables II and V), and the remaining papers inform the analysis across lifecycle stages, design patterns, security, and evaluation. The corpus spans eight benchmark environments, seven design patterns, and five representation categories.

### III-C Taxonomy Development

Both taxonomies were developed through an iterative bottom-up process. We first compiled a feature matrix for each analyzed system, recording its skill representation, acquisition method, execution model, storage mechanism, and governance features. Recurring clusters in this matrix suggested the seven design patterns (§V) and the five representation categories (§V-J1).

We tested each candidate taxonomy against the full corpus, refining categories through three revision cycles until every deeply-analyzed system could be classified without forcing. The representation $\times$ scope taxonomy was developed orthogonally: scope categories emerged from the environments addressed by the analyzed systems. Their orthogonality with representation was validated by confirming that examples exist across most cells of the resulting matrix.

The design patterns are deliberately non-exclusive: real systems often combine multiple patterns (e.g., a marketplace-distributed plugin using metadata-driven loading with hybrid NL+code implementation). We treat composability as a feature of the pattern framework rather than a taxonomy deficiency, as mutually exclusive patterns would not reflect how skills are deployed in practice.

## IV Skill Lifecycle Model

We organize the literature around a lifecycle model that traces an agentic skill from initial formation to eventual retirement. Rather than viewing skills as static artifacts, this model treats them as evolving system components shaped by interaction, feedback, and deployment constraints. The lifecycle comprises seven stages, depicted in Fig.2:

-

Discovery: identifying recurring task patterns, failure modes, or workflow bottlenecks that justify encapsulating behavior into a reusable skill.

-

Practice/Refinement: iteratively improving candidate skills through trial-and-error execution, reflection, and external feedback, allowing policies and prompts to stabilize across repeated use.

-

Distillation: extracting a stable and generalizable procedure from trajectories or demonstrations and packaging it into the $(C,\pi,T,R)$ tuple together with descriptive metadata and usage constraints.

-

Storage: persisting the skill within a library or repository, accompanied by indexing, versioning, and metadata that enable efficient retrieval and governance.

-

Retrieval/Composition: selecting relevant skills at runtime and composing them into higher-level workflows, often requiring compatibility checks across interfaces, contexts, and dependencies.

-

Execution: running the skill policy within the agent’s action loop under sandboxing, permission controls, and resource constraints that bound potential side effects.

-

Evaluation/Update: monitoring performance after deployment, detecting drift or failure, and revising, replacing, or retiring skills as environments and requirements evolve.

The lifecycle is not strictly linear. Feedback loops connect evaluation back to practice (when a skill underperforms), retrieval back to storage (when indexing fails to surface relevant skills), and execution back to discovery (when runtime failures reveal the need for new skills). Table II maps representative systems to lifecycle contributions.

*TABLE II: Lifecycle mapping: representative systems and their primary contributions to skill lifecycle stages. “✓” = primary focus; “$\sim$” = partial coverage.*

| System/Paper | Environment | Signal | Discovery | Practice | Distillation | Storage | Retrieval | Execution | Evaluation | Representation |

$\sim$ | Voyager [33] | Minecraft | Self-verify | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | Code |

$\sim$ $\sim$ $\sim$ | JARVIS-1 [36] | Minecraft | Multimodal | | ✓ | | ✓ | ✓ | ✓ | | Hybrid |

$\sim$ $\sim$ $\sim$ | DEPS [37] | Minecraft | LLM planner | ✓ | ✓ | | | | ✓ | ✗ | NL |

$\sim$ $\sim$ | Reflexion [35] | Multi | Verbal RL | ✗ | ✓ | | | ✗ | ✓ | ✓ | NL |

$\sim$ | Skill-it! [38] | Language | Curriculum | ✓ | ✓ | ✓ | ✗ | ✗ | | ✓ | Latent |

$\sim$ | SWE-agent [2] | SWE | Execution | ✗ | | ✗ | ✗ | ✗ | ✓ | ✓ | Code |

$\sim$ | AppAgent [39] | Mobile | Demonstrations | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | Hybrid |

$\sim$ | WebArena agents [1] | Web | Task reward | ✗ | | ✗ | ✗ | ✗ | ✓ | ✓ | NL/Code |

| CRADLE [40] | Games | Multi-source | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Hybrid |

$\sim$ | MemGPT [24] | Multi | Self-edit | ✗ | ✗ | ✗ | ✓ | ✓ | | ✗ | NL |

$\sim$ $\sim$ | AgentTuning [41] | Multi | SFT | | | ✓ | ✗ | ✗ | ✓ | ✓ | Policy |

| CodeAct [42] | Multi | Code exec | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | Code |

$\sim$ | FireAct [43] | Multi | Trajectories | ✗ | | ✓ | ✗ | ✗ | ✓ | ✓ | Policy |

$\sim$ | HuggingGPT [4] | Multi | API routing | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | | Hybrid |

$\sim$ $\sim$ $\sim$ | TaskWeaver [44] | SWE | Code-first | ✗ | | | ✓ | ✓ | ✓ | | Code |

$\sim$ $\sim$ $\sim$ | SayCan [45] | Robotics | Affordance | ✓ | | | ✓ | ✓ | ✓ | | Hybrid |

$\sim$ $\sim$ $\sim$ | MetaGPT [6] | Multi | Role assign | | | ✓ | ✓ | ✓ | ✓ | | Code |

$\sim$ $\sim$ $\sim$ $\sim$ | Generative Agents [46] | Social | Self-observe | | | | ✓ | ✓ | ✓ | | NL |

$\sim$ | Eureka [47] | Robotics | Reward search | ✓ | ✓ | ✓ | | ✗ | ✓ | ✓ | Code |

### IV-A Discovery

Skill discovery is the process of identifying recurring task patterns that warrant encapsulation into a reusable module. In Voyager [33], discovery is driven by a curriculum mechanism that proposes increasingly complex tasks in Minecraft; when the agent succeeds at a novel task, the solution trajectory becomes a candidate skill. DEPS [37] discovers skills through plan decomposition: a high-level planner identifies sub-goals, and repeated sub-goal patterns are promoted to skills. AppAgent [39] discovers skills through user demonstrations on mobile interfaces, identifying reusable interaction patterns across applications.

In the robotics domain, SayCan [45] discovers executable skills by grounding language instructions in robot affordances: the system scores candidate skills by both language relevance and physical feasibility, effectively discovering which skills apply in a given context. DECKARD [48] uses language-guided world models to discover skills through embodied decision making, imagining plans before executing them.

A key open question is unsupervised discovery: identifying skill boundaries without human-provided task definitions or explicit success signals. Current systems rely on either pre-defined task curricula or human demonstrations to seed the discovery process.

### IV-B Practice, Refinement, and Distillation

Once a candidate skill is identified, it must be refined into a reliable procedure. Reflexion [35] demonstrates a verbal reinforcement learning loop where the agent reflects on failed attempts and generates textual feedback that guides subsequent trials. This reflection mechanism serves as a practice loop that improves skill reliability without parameter updates.

Distillation converts raw traces into compact, generalizable skill representations. AgentTuning [41] distills traces from GPT-4 into smaller models through supervised fine-tuning, producing agents with internalized skills. FireAct [43] fine-tunes agents on diverse ReAct-style traces, distilling multi-step reasoning patterns into model weights.

Inner Monologue [49] extends verbal feedback to embodied agents by using language-based scene descriptions and success signals to iteratively refine robotic action sequences. Eureka [47] shows that LLMs can autonomously design reward functions for robotic skill acquisition through evolutionary search, achieving human-level performance and effectively automating the practice-and-refine loop for physical skills.

The distinction between practice and distillation is important: practice improves a skill’s reliability through iteration, while distillation changes its representation (e.g., from a verbose trajectory to a compact code function or from prompt-based instructions to model weights).

### IV-C Storage and Retrieval

Skill storage requires indexing mechanisms that support efficient retrieval. Voyager [33] maintains a skill library indexed by natural-language descriptions, using embedding similarity to retrieve relevant skills for new tasks. CRADLE [40] extends this with multi-level memory that stores skills alongside episodic context, enabling retrieval based on both task similarity and environmental state.

The storage-retrieval interface is where skill systems intersect with memory architectures. MemGPT [24] provides a hierarchical memory system that could serve as infrastructure for skill libraries, with main memory (context window) and archival storage (external database) supporting different access patterns. Generative Agents [46] implement a memory architecture for simulated social agents where behavioral patterns, analogous to social skills, are stored and retrieved based on recency, importance, and relevance, providing a model for how skill libraries might integrate with broader memory systems. The challenge is designing retrieval policies that balance precision (returning the most applicable skill) with recall (not missing relevant skills in novel contexts).

### IV-D Execution and Evaluation

Execution is the stage where a skill’s policy $\pi$ is enacted within the agent’s action loop. The execution model varies substantially by skill representation: natural-language skills are injected into the context window, code skills are executed in sandboxed environments, and policy skills operate through learned parameters. CodeAct [42] demonstrates that representing agent actions as executable Python code, rather than as tool-calling JSON, improves both the expressiveness and the verifiability of skill execution.

Evaluation assesses whether a skill achieves its intended outcome reliably. Deterministic evaluation harnesses, where the environment itself provides ground-truth verification, are preferable to human grading for scalability. SkillsBench [32] operationalizes this principle by pairing each of its 86 tasks with a deterministic verifier that checks environment state against expected outcomes, enabling reproducible evaluation across 7,308 agent trajectories.

We discuss evaluation in depth in §VIII.

## V Design Patterns and Taxonomy

We classify the emerging skill landscape along two complementary dimensions. First, we identify seven design patterns that describe how skills are packaged, loaded, and executed at the system level. Second, we develop a representation $\times$ scope taxonomy (§V-J) that describes what skills are and where they operate. Fig.3 arranges the seven patterns along an autonomy axis. This section presents both, beginning with the design patterns summarized in Table III.

*Fig. 3: Seven design patterns for agentic skills arranged along an autonomy spectrum, from human-controlled metadata disclosure (P1) to fully autonomous meta-skills (P6). Marketplace distribution (P7) spans the full spectrum as a cross-cutting distribution mechanism. Dashed lines indicate commonly combined patterns.*

*TABLE III: Seven design patterns for agentic skills, with representative systems, strengths, and governance considerations.*

| # | Pattern | Representative Systems | Representation | Strength | Weakness | Primary Risk |

| 1 | Metadata-driven progressive disclosure | Claude Code, Semantic Kernel, LangChain | NL + metadata | Token efficiency; scales to large libraries | Retrieval quality depends on metadata | Metadata poisoning |

| 2 | Code-as-skill (executable scripts) | Voyager, CodeAct, SWE-agent | Code | Deterministic; testable; composable | Requires sandbox; brittle to API changes | Code injection |

| 3 | Workflow enforcement | TDD agents, LATS, systematic debuggers | NL + rules | Reliability through gating; auditable | Rigid; may over-constrain agent | Rule bypass via prompt injection |

| 4 | Self-evolving skill libraries | Voyager, DEPS, CRADLE | Code + NL | Adapts to new tasks; improves with use | Quality control of self-generated skills | Skill drift; poisoned distillation |

| 5 | Hybrid NL+code macros | Claude skills, ReAct prompts | NL + code + refs | Flexible; human readable yet executable | Ambiguity at NL/code boundary | Inconsistent interpretation |

| 6 | Meta-skills (skills that create skills) | Self-Instruct, skill generators | NL / hybrid | Scales skill library; reduces human effort | Bootstrapping quality ceiling | Recursive error amplification |

| 7 | Plugin / marketplace distribution | OpenAI GPT Store, MCP servers, ClawHub, npm/pip | Any (packaged) | Ecosystem growth; community contribution | Supply-chain trust; version compat | Malicious packages (cf. ClawHavoc) |

### V-A Why Design Patterns?

Software engineering has long treated recurring design patterns as worth documenting for both practice and research [50]. We follow that approach for agentic skills. A design pattern describes a solution shape that shows up across systems. Here, patterns are system-level: they describe how infrastructure manages skills. By contrast, the representation/scope taxonomy is skill-level. They complement each other: one pattern (e.g., marketplace distribution) can host skills with any representation and scope.

### V-B Pattern-1: Metadata-Driven Disclosure

In Pattern-1, skills are discovered through compact metadata summaries (name, description, trigger conditions) that occupy minimal context. The full instructions are loaded into the agent’s context window only when the skill is selected for execution. This two-phase loading strategy addresses a fundamental constraint of LLM agents: the finite context window cannot hold all available skills simultaneously.

Claude Code’s skill system exemplifies this pattern. Each skill is registered with a short description and a set of trigger phrases. When the agent determines that a skill is relevant to the current task, it loads the full skill specification, which may include multi-page instructions, reference documents, and execution scripts. The Semantic Kernel framework [51] implements a similar approach with its plugin discovery mechanism, where function metadata is registered and the full function implementation is invoked only on selection.

The main benefit is scale: an agent can know about hundreds of skills while spending context tokens only on the few it activates. The main risk is metadata quality. If descriptions are wrong or incomplete, retrieval can pick the wrong skill or miss a relevant one.

### V-C Pattern-2: Code-as-Skill (Executable Scripts)

Code-as-skill represents skills as executable programs (Python functions, shell scripts, or domain-specific language programs) that the agent invokes through a runtime interface. Voyager [33] generates JavaScript functions as skills for Minecraft, stores them in a library, and retrieves them by natural-language description. CodeAct [42] demonstrates that framing agent actions as executable Python code, rather than as structured JSON tool calls, enables more expressive and verifiable behavior. In robotics, Code as Policies [52] generates Python programs for robotic control, and ProgPrompt [53] creates situated task plans as executable programs, both treating generated code as reusable skills for physical manipulation.

The key advantage of code skills is determinism: given the same inputs, a code skill produces the same outputs, enabling traditional software testing and verification. Code skills also composed of function calls, imports, and control flow. The limitation is brittleness: code skills break when underlying APIs, UI elements, or environmental conditions change, necessitating maintenance and version management (§IV-D).

### V-D Pattern-3: Workflow Enforcement

Workflow-enforcement skills impose hard-gated processes on agent behavior, ensuring that the agent follows a prescribed methodology rather than improvising. A test-driven development (TDD) skill, for example, mandates that the agent write tests before implementation, run the test suite, and iterate until all tests pass. The agent cannot skip or reorder these steps.

LATS (Language Agent Tree Search) [54] enforces a tree-search workflow that combines planning, acting, and reflection in a structured loop. Systematic debugging skills enforce a diagnosis-before-fix methodology, requiring the agent to reproduce the bug, identify root causes, and verify the fix before declaring success.

This pattern sacrifices flexibility for reliability. By constraining the agent’s action space to a proven sequence, workflow enforcement reduces the probability of hallucination-driven shortcuts and provides a clear audit trail. We note that Pattern-3 operates at the controller level: it prescribes how the agent executes rather than constituting a reusable skill artifact itself. LATS exemplifies a workflow controller that can host skills from other patterns. The governance surface is the rule set itself: if an attacker can modify the workflow rules (e.g., through prompt injection), the enforcement mechanism is compromised.

### V-E Pattern-4: Self-Evolving Skill Libraries

Self-evolving skill libraries combine skill execution with automated quality assessment and library maintenance. After each task, the system evaluates whether the agent’s behavior produced a successful trajectory worthy of distillation into a new skill or refinement of an existing one.

Voyager [33] provides a canonical example: it generates code-based skills, validates them through in-game execution, and incorporates verified skills into a persistent library indexed by natural-language descriptions. CRADLE [40] extends this paradigm with explicit memory management, linking skills to episodic context to enable retrieval based on environmental similarity.

The central tension in self-evolving libraries is quality control. The SkillsBench benchmark [32] reports that self-generated skills average $-$1.3 pp relative to skill-free baselines, with only one of five tested configurations showing any improvement, indicating that zero-shot self-generation without iterative verification can degrade performance in open-ended settings. This contrasts with Voyager and Eureka, where self-generated skills succeed in constrained environments with deterministic execution verification, suggesting the viability of self-generation depends critically on domain specificity and the availability of automated verification. Without human oversight or robust verification, self-evolving libraries risk accumulating “skill debt” analogous to technical debt in software systems.

### V-F Pattern-5: Hybrid NL+Code Macros

Hybrid skills combine natural-language specifications with executable components within a single package. The natural-language component describes the skill’s purpose, applicability conditions, and high-level logic in human-readable form, while the executable component provides code snippets, reference documents, or tool-calling sequences that implement concrete steps.

This pattern appears in production agent systems where skills must be both human-auditable and machine-executable. Claude Code’s skill system, for example, defines skills as markdown documents that include natural-language instructions, code blocks, and references to external assets. The ReAct paradigm [34] represents a lightweight version: the agent alternates between natural-language reasoning (“I need to search for X”) and executable actions (search API call), with the interleaving serving as an implicit hybrid skill.

The advantage of hybrid skills is flexibility: the natural-language component provides context and handles edge cases through reasoning, while the code component provides determinism for well-understood steps. The risk is boundary ambiguity: when instructions conflict with code, the agent must decide which to follow, creating potential for inconsistent behavior.

### V-G Pattern-6: Meta-Skills

Meta-skills are skills whose purpose is to create, modify, or compose other skills. A meta-skill might analyze an agent’s task history to identify recurring patterns, generate candidate skills from those patterns, and test them against held-out tasks. Self-Instruct [55] can be viewed through this lens: the LLM generates new instruction-following examples that serve as training data for skill acquisition. CREATOR [56] takes this further by enabling LLMs to create new tools (i.e., code functions) on demand, disentangling abstract reasoning from concrete tool implementation. Eureka [47] generates reward functions that serve as parameterizations for robotic skills, effectively creating skill specifications through code.

We cite Self-Instruct and CREATOR as precursors: they show the training-time idea of meta-skills, but they are mostly offline. We reserve Pattern-6 in the strict sense for methods that act as runtime-callable generators. Discovery (§IV-A) makes the procedural gap explicit; meta-skills are the generative mechanism that fills it. These sit at different levels (lifecycle stage vs. design pattern): a meta-skill automates what would otherwise be manual discovery.

Meta-skills let a small seed set of skills grow into a broad library without requiring a matching amount of human work. The risk is recursive error amplification: if the meta-skill produces a flawed skill that is subsequently used as input for further skill generation, errors compound. Quality gates at each generation step are essential (§VII).

### V-H Pattern-7: Plugin/Marketplace Distribution

The marketplace pattern treats skills as versioned, distributable packages with explicit dependency, compatibility, and governance metadata. The OpenAI GPT Store distributes custom GPT configurations that function as packaged skills. Anthropic’s Model Context Protocol (MCP) [57] defines a standardized interface for tool and skill servers, enabling third-party skill distribution with authentication and permission boundaries. ToolLLM [58] demonstrates integration with over 16,000 real-world APIs, illustrating the scale that marketplace-style distribution can achieve.

The most striking example of marketplace-scale skill distribution is OpenClaw [59], a viral agent framework built on a four-tool core (read, write, edit, bash) that treats skills as the primary extensibility mechanism. OpenClaw’s community skill registry, ClawHub, grew from zero to over 10,700 published skills within weeks of launch, while the project itself surpassed 200,000 GitHub stars faster than any software repository in history [60]. OpenClaw’s design philosophy is particularly relevant to our taxonomy: it embraces self-generated skills (Pattern-4 $+$ Pattern-6) by encouraging agents to extend themselves through code rather than downloading pre-built extensions. When combined with community distribution (Pattern-7), this creates a dual-source skill library: human-authored community skills alongside agent-authored local skills, both executable with full system access.

In the software ecosystem, analogous patterns include npm packages for JavaScript, pip packages for Python, and plugin systems in IDEs. The marketplace pattern enables community-driven skill creation at scale but introduces supply-chain risk: a malicious or compromised skill package can execute arbitrary actions within the agent’s permission scope. OpenClaw’s explosive growth and the severity of its subsequent security incidents (§VII-F) provide a stark illustration of this risk. We analyze these risks in detail in §VII.

### V-I Pattern Trade-offs

Our patterns represent different points in a multi-dimensional trade-off space. Table IV summarizes the key dimensions: Context cost measures how many tokens a pattern consumes during active use. Determinism reflects the predictability of execution outcomes. Composability captures how easily skills following this pattern can be combined into larger workflows. Governance surface indicates how amenable the pattern is to auditing, permission control, and provenance tracking.

*TABLE IV: Pattern trade-off summary across four dimensions. H = High, M = Medium, L = Low.*

| Pattern | Context cost | Determinism | Composability | Governance |

| 1: Metadata | L | L | M | M |

| 2: Code-as-skill | L | H | H | H |

| 3: Workflow | M | H | M | H |

| 4: Self-evolving | M | M | M | L |

| 5: Hybrid macro | M | M | M | M |

| 6: Meta-skill | H | L | L | L |

| 7: Marketplace | L | varies | H | M–H |

Pattern co-occurrence. Systems in our corpus use a median of 2 patterns (range: 1–4). The most common combination is Patterns 1+7 (metadata + marketplace), appearing in 4 systems (HuggingGPT, MetaGPT, AutoGen, ToolLLM). Two systems (Claude Code and OpenClaw) use 4 patterns, representing outliers. Five systems use a single pattern, while twelve use exactly two. The modest co-occurrence rates suggest the patterns capture meaningfully distinct architectural choices rather than collapsing into a single cluster.

No single pattern dominates. Production systems typically combine patterns: a marketplace-distributed plugin (Pattern-7) might use metadata-driven loading (Pattern-1) with hybrid NL+code implementation (Pattern-5) and workflow enforcement for critical steps (Pattern-3).

Computational overhead. The skill layer imposes overhead: retrieval adds latency, instruction loading consumes context tokens, and multi-level composition multiplies both. Table IV captures this abstractly as “context cost,” but quantifying the latency-accuracy tradeoff of skill-based versus skill-free agents across deployment scenarios remains an open empirical question.

### V-J Representation $\times$ Scope Taxonomy

Complementing the system-level design patterns, we propose an intrinsic taxonomy along two orthogonal axes: representation (how the skill’s policy is encoded) and scope (what environment or task domain the skill operates over). While patterns describe how infrastructure manages skills, this taxonomy operates at the skill level.

#### V-J1 Representation Axis

We identify five representation categories, ordered roughly by increasing formality:

Natural-language skills. The policy $\pi$ is expressed entirely in natural language: step-by-step instructions, standard operating procedures (SOPs), or playbook entries. The agent interprets these instructions through its language understanding capabilities. Natural-language skills are easy to author and audit but are subject to interpretation ambiguity and cannot be verified through traditional testing.

Code-as-skill. The policy $\pi$ is an executable program: a Python function, a shell script, a domain-specific language program, or a Jupyter notebook cell. Code skills offer determinism and testability but require execution infrastructure and are brittle to environmental changes.

Tool macros. A skill defined as a structured sequence of tool calls with parameterization logic. Tool macros occupy a middle ground between natural language (interpreted) and code (executed): they are more constrained than free-form code but more expressive than single tool calls.

Policy-based skills. The policy $\pi$ is a learned parameterized function, i.e., a neural network fine-tuned on trajectories. Policy skills are opaque (hard to inspect/audit) but capture subtle behavioral patterns that resist explicit codification.

Hybrid representations. A skill that combines two or more of the above. For example, a hybrid skill might use natural-language instructions for high-level logic, code blocks for deterministic steps, and an embedding-based retrieval mechanism for contextual adaptation.

#### V-J2 Scope Axis

We identify six scope categories based on the environment and task domain:

Single-tool skills. Skills that orchestrate a single tool with sophisticated parameterization, error handling, and retry logic. These are the simplest scope but can still exhibit non-trivial procedural complexity (e.g., a database query skill that handles schema variation).

Multi-tool orchestration. Skills that coordinate multiple tools in sequence or parallel to accomplish a composite task (e.g., search $\rightarrow$ extract $\rightarrow$ summarize $\rightarrow$ store).

Web interaction. Skills for navigating web interfaces, filling forms, extracting information from web pages, and completing web-based workflows. Benchmarked by WebArena [1] and Mind2Web [61]. A challenge unique to web skills is UI fragility: interfaces change frequently, breaking skills that depend on specific element selectors or page layouts. Skills encoding high-level intent (“fill in the departure field”) are more resilient than those encoding low-level actions (“click the element with id=departure-input”).

OS/desktop workflows. Skills that operate across multiple desktop applications, managing windows, files, and system settings. Benchmarked by OSWorld [62].

Software engineering. Skills for code understanding, bug localization, patch generation, testing, and deployment. Benchmarked by SWE-bench [63].

Robotics/physical. Skills for controlling physical actuators, navigating physical spaces, and manipulating objects. While this SoK focuses primarily on digital agents, robotics skill libraries provide instructive parallels, particularly for hierarchical skill composition [64]. Recent work demonstrates diverse LLM-driven robotic skills: SayCan [45] grounds language instructions in affordance functions to select feasible skills, Code as Policies [52] generates executable robot programs from language, ProgPrompt [53] creates situated task plans as programs, and Inner Monologue [49] uses language feedback to refine robotic actions iteratively.

Scope and skill value. The scope axis interacts with skill utility in a non-obvious way. SkillsBench [32] reports that skills yield the largest improvements in healthcare (+51.9 pp) and manufacturing (+41.9 pp) but only +4.5 pp in software engineering and +6.0 pp in mathematics. This suggests that skills provide the most value in domains where the base model’s pretraining data is sparse or insufficiently procedural, while domains with abundant code and mathematical reasoning data in pretraining benefit less from external procedural knowledge.

#### V-J3 Mapping: Patterns $\times$ Representation $\times$ Scope

Table V maps representative systems to all three classification dimensions. The mapping reveals that most systems occupy a sparse region of the full space: code-as-skill representation with SWE or web scope using the self-evolving library pattern. Large regions remain unexplored, particularly policy-based skills with marketplace distribution and natural-language skills with workflow enforcement.

*TABLE V: Taxonomy master table: representative systems mapped to design pattern, representation, scope, and key characteristics.*

| System | Pattern(s) | Representation | Scope | Acquisition | Execution | Evaluation | Governance |

| Voyager [33] | 2, 4 | Code | Game/Robotics | Self-practice | Sandbox | Self-verify | None |

| SWE-agent [2] | 2, 3 | Code | SWE | Pre-defined | Shell exec | SWE-bench | Sandboxed |

| CodeAct [42] | 2 | Code | Multi | Pre-defined | Python exec | AgentBench | Sandboxed |

| CRADLE [40] | 4, 5 | Hybrid | Game | Self-evolving | Multi-source | Task reward | None |

| AppAgent [39] | 1, 5 | Hybrid | Mobile | Demonstrations | UI actions | Task success | None |

| WebArena agents [1] | 2, 3 | Code/NL | Web | Pre-defined | Browser | Task reward | Sandboxed |

| HuggingGPT [4] | 1, 7 | Hybrid | Multi | Pre-registered | API routing | Task output | API auth |

| TaskWeaver [44] | 2, 5 | Code | SWE | Human + gen | Python exec | Output verify | Plugin sys |

| Claude Code | 1, 3, 5, 7 | Hybrid | SWE/Multi | Human-authored | Multi-mode | User verify | Trust tiers |

| MemGPT [24] | 1 | NL | Multi | Human-defined | Context mgmt | N/A | Access ctrl |

| AgentTuning [41] | 4 | Policy | Multi | Distillation | Fine-tuned | AgentBench | None |

| LATS [54] | 3 | NL + rules | Multi | Pre-defined | Tree search | Task reward | None |

| Reflexion [35] | 3† | NL | Multi | Self-practice | Verbal RL | Task reward | None |

| MetaGPT [6] | 1, 7 | Code | Multi | Role generation | Multi-agent | Task output | Role-based |

| SayCan [45] | 1, 2 | Hybrid | Robotics | Affordance grounding | Grounded exec | Task success | None |

| AutoGen [7] | 1, 7 | Hybrid | Multi | Pre-defined + gen | Multi-agent | Conversation | Protocol |

| Generative Agents [46] | 1, 4‡ | NL | Social | Self-observed | Memory retrieval | Behavioral | None |

| ToolLLM [58] | 1, 7 | Hybrid | Multi | API crawling | API routing | ToolEval | API auth |

| OpenClaw [59] | 2, 4, 6, 7 | Code/Hybrid | Multi | Self-generated + community | Bash/code exec | User verify | ClawHub + VirusTotal |

| †Reflexion performs transient in-context refinement without persistent library updates; we classify it under Pattern-3 only. |

| ‡Generative Agents’ behavioral patterns are closer to episodic memory than to skills as formally defined; included as an illustrative boundary case. |

## VI Acquisition, Composition, Orchestration

We address two complementary questions: how agents acquire skills, and how they compose and orchestrate acquired skills at runtime. We begin with five acquisition modes, ordered from most to least human involvement.

### VI-A Human-Authored Skills

The simplest acquisition mode is human authorship. A domain expert writes a skill specification (e.g., a standard operating procedure, a code function, or a hybrid document) and registers it in the agent’s skill library. Many production systems (e.g., Claude Code and enterprise automation platforms) rely on human-authored skills because they are easier to validate and assign accountability for.

Human authorship scales poorly but produces high-reliability skills. The trade-off is explicit: each skill requires human labor to create, test, and maintain, but the resulting skills are grounded in domain expertise and can be audited before deployment.

### VI-B Demonstration Distillation

Demonstration distillation extracts reusable procedures from observed trajectories. The input may be human demonstrations [39], expert agent traces [41], or successful task completions from the agent itself [33]. The key challenge is generalization: a trajectory that solved one specific instance must be abstracted into a skill that handles the broader class.

AgentTuning [41] collects interaction trajectories from GPT-4 across diverse agent tasks and uses them to fine-tune Llama models, effectively distilling procedural knowledge into model weights. FireAct [43] fine-tunes language models on ReAct-style trajectories, distilling the reasoning-acting pattern into an internalized skill.

### VI-C Self-Practice and Exploration

Self-practice acquisition allows the agent to discover and refine skills through autonomous interaction with the environment. Voyager [33] implements this through a curriculum-driven exploration loop: the agent proposes tasks, attempts them, evaluates success, and stores verified solutions as skills.

Reflexion [35] refines agent behavior through verbal self-reflection: after a failed attempt, the agent generates a textual analysis of what went wrong and uses this analysis to guide the next attempt. While Reflexion does not explicitly produce persistent skills, the reflection mechanism can be viewed as transient skill refinement within an episode.

AutoGPT [65] popularized the paradigm of fully autonomous agents that set their own sub-goals and practice iteratively, though without explicit skill persistence across sessions. DECKARD [48] combines language-guided world models with embodied exploration, imagining and evaluating plans before executing them in game environments.

The self-practice mode enables continual learning [66] without human supervision but introduces quality risk. Without external verification, agents may converge on locally optimal but globally suboptimal procedures, or worse, on procedures that succeed through exploitation of environment quirks rather than genuine task completion.

### VI-D Curriculum and Feedback

Curriculum-based acquisition structures the skill learning process through progressively harder tasks. Skill-it! [38] provides a theoretical framework for curriculum design in skill learning, demonstrating that training on an ordered sequence of skills improves sample efficiency compared to random ordering.

Feedback signals can come from humans (corrections, preferences), AI judges (LLM-based evaluators), or reward models trained on human preferences. The choice of feedback signal affects both the quality and the scalability of skill acquisition: human feedback is high-quality but expensive, AI judges are scalable but may miss subtle errors, and reward models generalize from limited human data but can be exploited through reward hacking.

### VI-E Meta-Skills and Self-Evolving Libraries

The most autonomous acquisition mode uses meta-skills (Pattern-6) to generate new skills from existing ones. A meta-skill might analyze an agent’s failure cases, identify missing capabilities, and generate candidate skills to fill those gaps. Self-Instruct [55] demonstrates a related approach: using an LLM to generate new instruction-following examples from a seed set, effectively bootstrapping a skill library from a small initial collection. CREATOR [56] enables LLMs to create new tools on demand, and Eureka [47] generates reward functions that parameterize robotic skills, both exemplifying meta-skill acquisition at different levels of abstraction.

Self-evolving libraries combine meta-skill generation with automated quality assessment, creating a closed loop in which the skill library grows and improves without human intervention. The primary risk is the quality ceiling problem: without external grounding, the library cannot exceed the capability of the meta-skill itself, and errors in early generations may propagate through subsequent ones.

### VI-F Skill composition and orchestration.

Individual skills rarely suffice for complex tasks. Fig.4 illustrates the composition architecture. The remainder of this section addresses how skills are combined, routed, and managed during multi-step execution.

*Fig. 4: Skill composition and orchestration. Tasks are matched to skills via embedding-based retrieval or LLM-mediated routing. Selected skills decompose hierarchically into sub-skills. Dashed arrows indicate failure recovery paths that trigger re-retrieval or alternative skill selection.*

#### VI-F1 Hierarchical Skill Structures

Skills organize into hierarchies: a high-level skill (e.g., “deploy a web application”) invokes mid-level skills (“set up database,” “configure server,” “run tests”), which in turn invoke low-level skills (“execute SQL migration,” “write Nginx config”). This hierarchical structure mirrors the option framework in reinforcement learning [10], where temporally extended actions (options) compose atomic actions into reusable behavioral modules.

In the LLM agent context, hierarchical composition is typically managed through a planning layer that decomposes tasks and routes sub-tasks to appropriate skills. HuggingGPT [4] demonstrates this at the tool level, using an LLM planner to decompose requests into sub-tasks routed to specialized Hugging Face models. The same architecture applies to skills: a planner selects and sequences skills based on task requirements and skill metadata.

Runtime skill selection and routing. When multiple skills could apply to a given context, the agent must select the most appropriate one. Two routing strategies dominate:

-

Embedding-based retrieval The task description is embedded and compared against skill description embeddings. The top-$k$ matching skills are loaded into the context window for the agent to evaluate. Voyager [33] and AppAgent [39] use this approach.

-

LLM-mediated routing The agent itself reasons about which skill to invoke, based on skill metadata loaded through progressive disclosure (Pattern-1). This approach is more flexible than embedding retrieval but consumes additional inference tokens and is subject to the agent’s reasoning quality.

Hybrid strategies combine both: embedding retrieval narrows the candidate set, and the agent’s reasoning selects the final skill. This two-stage approach balances recall (embedding search surfaces relevant candidates) with precision (LLM reasoning evaluates fit).

Skill conflict resolution. When multiple skills are simultaneously applicable ($C_{1}(o,g)=1$ and $C_{2}(o,g)=1$), the agent requires a tie-breaking mechanism. Current systems typically rely on ranking heuristics such as embedding similarity or ad hoc LLM judgment, but they lack an explicit conflict-resolution policy. A principled approach, analogous to method specificity in HTNs [28] or rule priority in production systems, remains an open research problem.

Failure recovery. Failure recovery in skill-based agents can itself be modeled as a skill. When the termination condition $T$ signals failure, a recovery skill is invoked to diagnose the cause and decide whether to retry, backtrack to a prior state, or escalate to an alternative strategy.

LATS [54] implements recovery through tree search: when a branch fails, the system backtracks and explores alternative action sequences. Reflexion [35] uses verbal reflection as a recovery mechanism, generating natural-language analysis of failures that guides subsequent attempts.

Treating recovery as a first-class skill has governance implications: the recovery skill must be at least as trusted as the skill it is recovering, since it operates in the same execution context and may need to undo or compensate for the failed skill’s actions.

Multi-agent skill sharing. In multi-agent systems, skills can be shared across agents through common skill repositories. MetaGPT [6] assigns specialized roles (product manager, architect, engineer) to different agents, each equipped with role-specific skills that compose into a software development workflow. AutoGen [7] enables multi-agent conversations where agents with different skill profiles collaborate through structured dialogue protocols. ProAgent [67] builds proactive cooperative agents that anticipate teammates’ actions and adapt their skill execution accordingly. This enables division of labor: different agents specialize in different skill sets, and tasks are routed to the agent with the most relevant skills. However, shared skill repositories introduce cross-agent security concerns (§VII): a compromised skill in a shared repository affects all agents that consume it.

## VII Security, Trust, and Governance of Skills

The skill layer introduces a new attack surface for LLM agents [16]. Skills are code or instructions that influence agent behavior; a compromised skill can steer an agent toward malicious outcomes while appearing benign at the metadata level. This section systematizes threats, mitigations, and governance mechanisms specific to the skill layer.

### VII-A Threat Model

We identify six primary threat categories:

Poisoned skill retrieval. An attacker crafts skill metadata to cause the retrieval mechanism to surface a malicious skill in response to benign queries. This is analogous to SEO poisoning in web search. The attack exploits Pattern-1 (metadata-driven disclosure): if the retrieval mechanism relies solely on embedding similarity, adversarial metadata can manipulate ranking.

Malicious skill payloads. A skill’s policy $\pi$ contains instructions or code that perform unauthorized actions when executed. In code skills (Pattern-2), this resembles supply-chain attacks in traditional software [68]. In natural-language skills (Pattern-5), the payload is a form of prompt injection: instructions embedded within the skill text that redirect agent behavior.

Cross-tenant leakage. In multi-agent or multi-user systems with shared skill repositories, skills authored by one tenant may access data or resources belonging to another. This risk is acute in enterprise deployments where multiple teams share agent infrastructure: a skill authored by Team A should not inadvertently access Team B’s data, requiring per-tenant sandboxing with permission boundaries enforced by the execution runtime rather than by the skill itself.

Skill drift exploitation. Skills that were safe at authoring time may become unsafe as environment evolves. An attacker who controls part of the environment (e.g., a web page that a skill navigates) can manipulate environments to change the skill’s behavior without modifying the skill itself.

Confused deputy via environmental injection. An agent processing untrusted observations (e.g., web pages or user documents) may encounter adversarial instructions that coerce it into misusing an otherwise benign, privileged skill. The skill itself remains uncompromised; instead, the attack exploits the data–control boundary between the observation space $O$ and skill invocation. This vector differs from malicious skill payloads, where the attack resides within the skill itself, and is particularly dangerous because it bypasses skill-level trust verification entirely.

Applicability condition poisoning. An attacker manipulates the input to $C$ such that a malicious or inappropriate skill returns $C(o,g)=1$ universally, activating in contexts where it should not. This can occur through metadata poisoning (Pattern-1) or through adversarial environmental states that trigger overbroad applicability predicates. The formal model’s reliance on $C$ for skill selection makes this a direct attack on the skill abstraction itself.

### VII-B Trust Tiers and Progressive Disclosure

We propose a four-tier trust model for skills. Fig.5 depicts the nested trust boundaries alongside attack vectors and defense mechanisms.

-

Tier-1(metadata only): The agent sees only the skill name and description. No instructions or code are loaded. This tier supports skill discovery without execution risk.

-

Tier-2 (instruction access): The agent loads the skill’s natural-language instructions into its context window. The instructions may influence the agent’s reasoning. However, Tier-2 provides meaningful isolation only when the runtime enforces a read-only mode during instruction loading, with tool execution gated behind a separate approval channel. Without architectural separation between reasoning and action, Tier-2 instructions can indirectly induce tool invocations through the agent’s standard decision loop, effectively degrading to Tier-3.

-

Tier-3 (supervised execution): The skill can execute actions (tool calls, code execution) but each action requires user approval or runs within a constrained sandbox.

-

Tier-4 (autonomous execution): The skill executes without per-action approval, subject to pre-configured permission boundaries and monitoring.

Production systems should default to Tier-1 for untrusted skills and require explicit trust escalation, backed by provenance verification, for higher tiers. The trust tier should be sticky: once a skill demonstrates reliable behavior at Tier-3 over multiple invocations, it may be promoted to Tier-4, but a single safety violation should trigger demotion.

Privilege escalation. The trust tier model must also guard against escalation: a Tier-1 skill’s metadata could include instructions designed to trick the agent into loading it at a higher tier. Tier transitions should be enforced by the runtime, not by skill-provided metadata. Cross-referencing with prompt injection attacks [69], Tier-2 instruction access is particularly vulnerable when the loaded instructions contain embedded directives that cause the agent to invoke tools or escalate the skill’s own privileges.

*Fig. 5: Trust-tiered threat model for skill governance. Four nested privilege tiers (T1–T4) form concentric security boundaries. Red arrows show attack vectors targeting different tier boundaries; green labels indicate defense mechanisms between tiers.*

### VII-C Sandboxing and Permission Boundaries

Code skills (Pattern-2) require sandboxed execution environments that limit access to the file system, network, and system resources. Container-based sandboxes (e.g., Docker) and WebAssembly runtimes provide isolation with varying performance overhead. The key design question is granularity: should sandboxing be per-skill (each skill runs in its own sandbox), per-session (all skills in a session share a sandbox), or per-tier (sandboxing varies by trust level)?

Natural-language skills (Pattern-5) present a different sandboxing challenge: the “execution environment” is the agent’s context window, and the “sandbox” is the instruction-following boundary. Prompt injection attacks [69] demonstrate that this boundary is permeable. Architectural mitigations include separating skill instructions from user data, using structured input/output schemas, and employing output filtering to detect unauthorized actions.

### VII-D Skill Supply-Chain Governance

Marketplace-distributed skills (Pattern-7) face supply-chain risks analogous to those in package management ecosystems. We recommend four governance mechanisms:

Provenance signing. Each skill package includes a cryptographic signature from its author, enabling verification of authorship and integrity. This mirrors code signing in traditional software distribution.

Dependency auditing. Skills may depend on other skills, tools, or external services. A dependency graph should be maintained and audited for known vulnerabilities, similar to dependency scanning in npm or pip.

Continuous monitoring. Even after initial vetting, skills should be monitored for behavioral anomalies during execution. Unexpected tool calls, excessive resource consumption, or access to out-of-scope resources should trigger alerts and potential demotion to a lower trust tier.

Version pinning. Skill consumers should pin to specific versions rather than tracking “latest,” to prevent a compromised update from automatically propagating to all consumers.

### VII-E Pattern-Specific Risk Matrix

Different design patterns expose different attack surfaces. Table VI maps each pattern to its primary risks and recommended mitigations.

*TABLE VI: Pattern-specific security risk matrix.*

| # | Pattern | Primary Risks | Recommended Mitigations | Severity |

| 1 | Metadata progressive disclosure | Metadata poisoning; misleading descriptions | Metadata schema validation; human review for high-privilege skills | Medium |

| 2 | Code-as-skill | Code injection; sandbox escape; dependency vulnerabilities | Container sandboxing; static analysis; dependency scanning | High |

| 3 | Workflow enforcement | Rule bypass via prompt injection; overly rigid constraints | Input sanitization; rule integrity verification | Medium |

| 4 | Self-evolving libraries | Poisoned distillation; skill drift; quality degradation | Human-in-the-loop verification; regression testing; anomaly detection | High |

| 5 | Hybrid NL+code macros | Boundary ambiguity exploitation; conflicting instructions | Clear NL/code separation; instruction priority rules | Medium |

| 6 | Meta-skills | Recursive error amplification; adversarial skill generation | Generation caps; quality gates at each iteration; diversity checks | High |

| 7 | Marketplace distribution | Supply-chain attacks; malicious packages; version tampering | Provenance signing; continuous monitoring; version pinning | Critical |

| — | Confused deputy (cross-cutting) | Environmental injection coerces misuse of privileged skills (affects P1, P2, P5) | Data-flow tracking; capability confinement; input/output separation | High |

$C$ | — | -poisoning (cross-cutting) | Adversarial inputs cause inappropriate skill activation (affects P1, P4) | Adversarial testing of applicability predicates; input validation | Medium |

### VII-F Case Study: ClawHavoc Supply-Chain Attack

The ClawHavoc campaign against OpenClaw’s ClawHub skill registry [60] provides the first large-scale empirical evidence of skill supply-chain exploitation, concretizing every threat category in our model and revealing the severity of real-world consequences.

Scale and attack surface. Within weeks of ClawHub’s launch, security researchers identified 1,184 malicious skills across the registry [60], while a separate Snyk audit found that 36.8% of all published skills contained at least one security flaw. The campaign involved 12 publisher accounts, with a single account responsible for 677 packages (57% of all malicious listings), while the platform’s most-downloaded skill (“What Would Elon Do”) contained 9 vulnerabilities including 2 critical ones, with its ranking artificially inflated through 4,000 faked downloads [60]. VirusTotal’s analysis of over 3,016 ClawHub skills confirmed that hundreds exhibited malicious characteristics [70]. Separately, a Snyk audit found that 283 of 3,984 skills (7.1%) exposed sensitive credentials in plaintext through LLM context windows and output logs. The attack surface was global: over 135,000 exposed OpenClaw instances were detected across 82 countries.

Severity of credential and asset theft. The consequences of malicious skill execution were not theoretical. The primary payload, Atomic macOS Stealer (AMOS), systematically harvested: (i) LLM API keys from .env files and OpenClaw configuration, enabling billing fraud and model abuse; (ii) cryptocurrency wallet keys across 60+ wallet types including Phantom, MetaMask, and Exodus, enabling irreversible asset theft; (iii) browser-stored passwords, credit card numbers, and autofill data across Chrome, Safari, Firefox, Brave, and Edge; (iv) SSH keys and Keychain credentials, granting persistent access to production infrastructure; and (v) Telegram sessions and local files from Desktop and Documents directories. Windows-targeted payloads delivered VMProtect-packed infostealers via password-protected archives, and 91% of malicious skills included prompt injection payloads that weaponized the agent itself as an accomplice, attacking both humans and AI simultaneously. Belgium’s Centre for Cybersecurity and China’s MIIT issued emergency advisories, while multiple South Korean technology companies blocked OpenClaw entirely.

Attack vector analysis through our pattern taxonomy. The ClawHavoc campaign instantiates multiple threat categories from §VII-A:

-

Poisoned skill retrieval: Attackers cloned popular legitimate skills under near-identical names, exploiting Pattern-1’s metadata-driven discovery to rank malicious versions alongside or above originals.

-

Malicious skill payloads: Skills included reverse shells, credential-exfiltration webhooks, and social-engineering “setup” instructions that told users to run curl | bash pipelines. These exploit Pattern-2’s code execution and Pattern-5’s ambiguity at the NL/code boundary.

-

Confused deputy: Prompt injection payloads in skill documentation coerced the agent into executing malicious commands using its legitimate tool access, bypassing any skill-level trust check.

-

$C$-poisoning: Overbroad skill descriptions ensured malicious skills activated across broad task categories (crypto, productivity, automation), maximizing the attack surface through Pattern-1 metadata manipulation.

Pattern-specific impact gradient. The severity varies by pattern. Pattern-7 (marketplace) is the most directly impacted: it is the distribution channel through which all attacks propagated, and the ClawHub registry’s initial absence of provenance signing, dependency auditing, or automated scanning enabled the 36.8% malicious-skill rate. Pattern-2 (code-as-skill) is the main execution vector: OpenClaw skills run code with the agent’s full system permissions, so one malicious skill can access local credentials such as API keys, wallets, browser vaults, and SSH keys. Pattern-1 (metadata) is the discovery vector: poisoned metadata enabled ranking manipulation and name-squatting. Pattern-5 (hybrid NL+code) is exploited through documentation-as-attack-surface: skill README files contained the actual social-engineering payload. Pattern-3 (workflow enforcement) is less exposed because hard-gated execution sequences constrain the agent’s action space. For example, a mandated test-before-deploy workflow is harder to bypass through prompt injection alone. Pattern-4 (self-evolving) presents a latent risk: if an agent’s self-generated skill library ingests a malicious community skill as a template, the poison propagates through the agent’s own generation loop.

Governance response. OpenClaw’s initial response was a partnership with VirusTotal [70] to scan published skills using SHA-256 fingerprints, Code Insight (LLM-based behavioral analysis), and daily re-scans. This matches several mechanisms in §VII-D: automated provenance checks, behavioral anomaly detection, and blocking known-bad versions by hash. OpenClaw also noted that VirusTotal scanning is “not a silver bullet”.

Why traditional scanners fail: tuple-level analysis. The limitations of traditional malware scanners become evident when viewed through our formal definition $S=(C,\pi,T,R)$ introduced in §II-A. Each component of the tuple exposes a distinct attack surface, yet conventional security tools cover only a small subset of them:

-

$R$ (interface): The callable interface (skill name, description, and parameter schema) serves as the first point of contact. Name squatting, misleading descriptions, and inflated download counts can manipulate $R$ and distort discovery. However, VirusTotal does not assess the semantics of skill metadata.

-

$C$ (applicability condition): Overbroad applicability predicates that return $C(o,g)=1$ for maximally many contexts increase the blast radius of a malicious skill. No current scanner audits whether a skill’s activation scope is proportionate to its stated purpose.

-

$\pi$ (policy): The policy component is where most attacks arise, yet $\pi$ in agentic skills is inherently heterogeneous: it may contain executable code (amenable to static analysis), natural-language instructions (largely invisible to binary scanners), or both. For example, a curl command hidden within a “stock tracking” skill’s setup instructions, or a prompt-injection directive (e.g., “ignore all previous safety guidelines”) embedded in the NL policy, can exfiltrate .env files and API keys to external servers. VirusTotal, which is designed to detect binary malware signatures, labels such payloads as Benign because they appear as syntactically valid text or harmless shell commands when analyzed in isolation.

-

$T$ (termination condition): A malicious $T$ can terminate the skill prematurely to evade logging (exfiltrate-then-exit-cleanly), or fail to terminate to enable persistent background access. Neither behavior triggers traditional antivirus heuristics.

Complementary skill auditing. Recognizing this gap, the community has developed skill-native auditing tools that operate at the tuple level rather than the binary level. Agent Skills Guard [71] and SkillGuard [72] exemplify a three-layer detection architecture mapped to our formalization:

-

Rule engine / AST analysis (auditing $\pi_{\text{code}}$ and $R$): Pattern rules and Abstract Syntax Tree analysis flag risky constructs in the executable part of $\pi$ (e.g., shell execution, eval(), reverse shells, credential access, destructive operations) and can also catch hardcoded secrets exposed via $R$’s metadata. This layer runs locally with low overhead, incurs no per-call cost, and can cover a broad set of attack patterns across multiple languages.

-

LLM semantic analysis (auditing $\pi_{\text{NL}}$ and $C$): An LLM reviews the natural-language part of $\pi$ for hidden intent (prompt injection [73], social-engineering directives, or instructions conflicting with the stated purpose) and checks whether $C$’s activation scope is appropriate. This catches attacks that rule-based or binary scanning misses, such as benign-looking NL instructions that steer the agent to exfiltrate data using otherwise legitimate tools.

-

Reputation scoring (aggregating across $C$, $\pi$, $T$, $R$): Signals from the layers are combined into a 0-100 reputation score. The tool uses threshold bands (e.g., above 80 as “safe” and below 30 as “malicious”). The authors report a controlled evaluation on 39 test cases, including 4 adversarial samples that VirusTotal marked as Benign, and no false positives on legitimate skills in that set [71].

These tools are implemented as agent skills (Pattern-2) and can be installed directly in the OpenClaw environment. This means an agent can use one skill to audit others. In practice, it shows that the skill abstraction can also encode governance checks within the $(C,\pi,T,R)$ framework. The ClawHavoc case demonstrates that skill marketplace governance requires defense in depth: binary scanning (VirusTotal) catches commodity malware targeting $\pi_{\text{code}}$, skill-native auditing catches NL-level attacks on $\pi_{\text{NL}}$ and $C$, and runtime behavioral monitoring is needed to detect attacks on $T$ and context-dependent exploits that only manifest during execution. No single layer suffices; the tuple-level decomposition provides the conceptual framework for understanding which defenses cover which attack surfaces.

## VIII Evaluating Agentic Skills

We evaluate the utility of agentic skills through a five-dimensional framework and map existing benchmarks to measurable skill properties.

### VIII-A Evaluation Dimensions

Correctness. Correctness measures whether a skill achieves its intended outcome. Evaluation relies on ground-truth annotations or deterministic verifiers. For code skills, unit tests provide direct verification, while for web interaction skills, environment state comparison (e.g., verifying whether a form was submitted correctly) serves as a practical proxy.

Robustness. Robustness captures a skill’s reliability under input variations, environment perturbations, and edge cases. A robust skill maintains consistent performance when confronted with minor deviations from the training distribution, such as handling both legacy and updated UI layouts.

Efficiency. Efficiency characterizes the resource cost of executing a skill. Relevant metrics include token consumption (for natural-language skills), wall-clock time, number of tool calls, and API costs. Efficiency directly affects deployment cost and composability, as inefficient sub-skills slow downstream workflows.

Generalization. Generalization evaluates whether a skill transfers to unseen tasks or domains. This dimension is challenging to measure because it requires out-of-distribution evaluation. Benchmarks such as cross-website generalization in Mind2Web [61] and cross-application evaluation in OSWorld [62] provide partial evidence.

Safety. Safety assesses whether a skill avoids harmful actions, respects permission boundaries, and handles failures gracefully. Evaluation commonly involves adversarial testing, red-teaming, and runtime monitoring for unauthorized or unsafe behaviors.

### VIII-B Deterministic Evaluation Harnesses

Human evaluation of agent skills does not scale. We advocate for deterministic evaluation harnesses: benchmark environments where success is measured automatically by checking environment state against expected outcomes. This approach provides low-cost reproducible evaluation that can be integrated into skill development pipelines.

The key design principle is outcome-based verification: rather than judging the quality of intermediate reasoning or the elegance of the skill’s approach, the harness checks whether the intended outcome was achieved. This aligns with the pragmatic nature of skills as procedural modules valued for their effects, not their form.

### VIII-C Benchmark-to-Skill Mapping

Table VII maps major agent benchmarks to the skill dimensions they assess. No single benchmark covers all dimensions; a comprehensive skill evaluation requires combining multiple benchmarks.

*TABLE VII: Benchmark-to-skill-dimension mapping. ✓ = primary assessment; $\sim$ = partial assessment; empty = not assessed.*

| Benchmark | Environment | Correctness | Robustness | Efficiency | Generalization | Safety | Skill Scope Assessed |

$\sim$ | SkillsBench [32] | Multi | ✓ | | ✓ | ✓ | | Skill utility, composition, domain |

$\sim$ $\sim$ | WebArena [1] | Web | ✓ | | | | | Web navigation, UI grounding |

| Mind2Web [61] | Web | ✓ | | | ✓ | | Cross-site generalization |

$\sim$ | OSWorld [62] | Desktop | ✓ | | | ✓ | | Multi-application workflows |

$\sim$ $\sim$ | SWE-bench [63] | SWE | ✓ | | | | | Code understanding, patch generation |

| GAIA [74] | Multi | ✓ | | | ✓ | | General assistant capability |

$\sim$ $\sim$ | AgentBench [75] | Multi | ✓ | | | ✓ | | Cross-environment performance |

$\sim$ | AndroidWorld [76] | Mobile | ✓ | | | | | Mobile interaction skills |

### VIII-D Anchor Case Study: SkillsBench

The SkillsBench benchmark [32] provides the most direct evidence to date for the value of curated skills. We note that the quantitative findings in this subsection derive primarily from a single, non-peer-reviewed benchmark. While the scale (86 tasks, 7,308 trajectories) and methodological rigor of SkillsBench provide useful evidence, independent replication across additional benchmarks is needed to confirm these patterns. The benchmark evaluates 86 tasks across 11 domains (healthcare, manufacturing, cybersecurity, natural science, energy, finance, office work, media, robotics, mathematics, and software engineering) using 7 agent-model configurations over 7,308 trajectories. Each task is assessed under three conditions: no skills, curated skills, and self-generated skills, with deterministic verifiers ensuring objective evaluation.

Curated skills provide substantial, quantifiable improvement. Across all configurations, curated skills raise the average pass rate by 16.2 percentage points (from 24.3% to 40.6%). The effect varies dramatically by domain: healthcare sees +51.9 pp, manufacturing +41.9 pp, and cybersecurity +23.2 pp, while software engineering gains only +4.5 pp and mathematics +6.0 pp. This domain variance is consistent with the hypothesis that skills help most where the base model’s pretraining data provides insufficient procedural grounding, which is directly relevant to the scope axis of our taxonomy (§V-J2). Domain variance may also reflect confounders including task construction, verifier strictness, and skill authoring quality differences across domains.

Self-generated skills provide no benefit. Self-generated skills average $-$1.3 pp relative to the no-skills baseline, suggesting that models cannot yet reliably author the procedural knowledge they benefit from consuming in open-ended settings. Only one configuration (Claude Opus 4.6) showed a modest +1.4 pp, while Codex + GPT-5.2 degraded by $-$5.6 pp. This finding is consistent with the quality concerns raised for self-evolving libraries (Pattern-4, §V-E).

Skill quantity and complexity matter. Focused skills with 2–3 modules yield optimal improvement (+18.6 pp), while 4+ skills show diminishing returns (+5.9 pp). “Detailed” skills (moderate-length, focused guidance) improve by +18.8 pp, whereas “comprehensive” skills (exhaustive documentation) degrade performance by $-$2.9 pp. This pattern is consistent with Pattern-1 (metadata-driven progressive disclosure, §V-B): loading focused procedural instructions outperforms loading comprehensive reference material.

Skills as compute equalizers. Smaller models equipped with curated skills can match or exceed larger models without skills. Claude Haiku 4.5 with skills (27.7%) outperforms Claude Opus 4.5 without skills (22.0%), suggesting that skill libraries may serve as a practical cost-reduction mechanism.

Negative-delta tasks. 16 of 84 tasks show performance degradation with skills, with the worst case ($-$39.3 pp) occurring for tasks where the base model already performs well and skills introduce conflicting guidance. This highlights the importance of the applicability condition $C$ in our formalization (§II-A): a skill should activate only when its procedural knowledge is beneficial.

We present these as interpretive hypotheses grounded in the benchmark data, not causal conclusions; Voyager’s self-verification success rate and AgentBench [75] cross-environment results provide partial corroboration from independent sources, even if they do not directly measure the curated-vs-self-generated comparison. These findings underscore the importance of distinguishing between skill availability (having relevant skills) and skill quality (having skills that actually help). The skill lifecycle model (§IV) addresses both: discovery and storage ensure availability, while practice, evaluation, and update ensure quality.

## IX Discussion and Limitations

### IX-A Cross-Cutting Observations

Several patterns emerge from the systematization that are not visible from any individual system.

Representation–governance coupling. More formal skill representations admit stronger governance. Code skills (Pattern-2) support static analysis, unit testing, and sandboxed execution; natural-language skills resist all three. This creates a tension: the representations that are easiest to author (NL) are hardest to govern, while those amenable to formal verification (code, policy) require specialized authoring expertise. No existing system resolves this tension fully; hybrid representations (Pattern-5) attempt a compromise but introduce boundary ambiguity.

Sparsity of the design space. Table V reveals that most systems cluster in a narrow region: code-as-skill representation with self-evolving library patterns in game or SWE environments. Large regions of the representation $\times$ scope $\times$ pattern space remain unexplored, particularly policy-based skills with marketplace distribution and NL skills with formal workflow enforcement. These unexplored regions represent both opportunity and risk: they may be inherently difficult (explaining the sparsity) or simply underexplored.

Marketplace growth outpaces governance. TThe OpenClaw experience (§VII-F) shows that when skill ecosystems grow quickly, governance mechanisms can lag behind. ClawHub’s 36.8% malicious-skill rate at its peak is orders of magnitude worse than the point-in-time malware rates observed in mature package registries such as npm, reflecting the absence of even basic supply-chain protections (package signing, automated scanning, reputation scoring) that took traditional package ecosystems years to develop. The subsequent VirusTotal partnership reduced the threat surface but was reactive; proactive governance (pre-publication scanning, behavioral sandboxing, capability confinement) is necessary for Pattern-7 systems that distribute skills with full system access. This observation reinforces the Pattern-3 advantage in our taxonomy: workflow enforcement, which constrains execution sequences before skills run, is inherently more resilient to supply-chain compromise than patterns that grant broad execution permissions and attempt to detect misuse after the fact.

The curation-scalability tradeoff. The SkillsBench evidence (§VIII-D) quantifies a fundamental tradeoff: curated skills improve pass rates by +16.2 pp on average, while self-generated skills degrade them by $-$1.3 pp. Self-evolving libraries (Pattern-4) are the most scalable acquisition mechanism but produce skills that hurt performance; human curation yields the most reliable skills but does not scale. SkillsBench further shows that focused skills with 2–3 modules outperform comprehensive documentation, suggesting that the quality problem is not just accuracy but also conciseness: effective skills must distill procedural knowledge rather than dump reference material. The tension is not absolute. Verification-gated self-generation (Voyager [33], Eureka [47]) succeeds in constrained environments with deterministic execution feedback; the SkillsBench evidence indicates this success does not yet generalize to open-ended, multi-domain settings without execution-verified practice loops. Closing the tradeoff likely requires combining autonomous generation with automated verification pipelines and length-constrained distillation.

### IX-B Limitations of This Systematization

Corpus recency. The LLM agent skill ecosystem is recent: the majority of systems analyzed were published in 2023–2024. While we ground the skill abstraction in decades of cognitive science and RL, the LLM-specific literature may be too nascent for patterns identified here to be stable. Taxonomies may require revision as the field matures.

Corpus coverage. Our analysis examines 24 systems from a retained set of 65 papers. Despite systematic search procedures (§III), we may miss relevant work, particularly from industry systems with limited public documentation, non-English-language publications, and concurrent preprints.

Taxonomy validation. The seven design patterns were derived bottom-up from the analyzed systems but have not been validated through external expert surveys or formal concept analysis. The non-exclusivity of patterns (systems combine multiple patterns) complicates categorical analysis and may limit the taxonomy’s discriminative power for future systems that combine patterns in novel ways.

Production and safety coverage. Our corpus emphasizes research systems with published evaluations. Several production frameworks (e.g., LangChain/LangGraph for skill composition, DSPy for declarative skill compilation) and safety-focused benchmarks (e.g., AgentHarm, InjectAgent) are relevant but under-represented in our analysis due to limited peer-reviewed documentation. We focus on systems with sufficient published detail for rigorous classification.

Benchmark reliance. Our evaluation analysis relies on published benchmark results, which may not reflect real-world skill utility. Production deployments involve longer time horizons, messier environments, and adversarial conditions not captured by existing benchmarks.

## X Open Problems and Research Roadmap

Existing skill-based agents still expose several unresolved tensions that limit reliable deployment at scale. We highlight several directions.

### X-A Verified Autonomous Skill Generation

A central tension revealed throughout our analysis is the trade-off between scalability and reliability in skill construction. Systems that allow skills to evolve autonomously can expand capability libraries rapidly, yet empirical evidence shows that automatically generated skills may occasionally degrade downstream performance. In contrast, human-curated skills remain more dependable but introduce a clear scalability bottleneck, as manual validation cannot keep pace with growing agent deployments.

This shows that the key obstacle is no longer skill generation itself, but verification at the point of admission into the skill library. A promising direction is to treat skills similarly to software artifacts in continuous integration pipelines: newly generated skills would be evaluated against held-out task distributions before becoming reusable components. For code-centric skills, formal or semi-formal verification techniques may provide guarantees about behavior, while natural-language or hybrid skills likely require behavioral testing and regression-style evaluation. Progress in this area would enable self-evolving skill libraries that improve over time without accumulating hidden performance regressions.

### X-B Unsupervised Skill Discovery

Another limitation concerns how new skills are discovered in the first place. Although many existing systems advertise autonomous learning, most still rely heavily on external scaffolding such as predefined curricula, demonstrations, or explicit reward signals. Our lifecycle survey shows that fully autonomous discovery remains rare: even systems designed for exploration typically depend on some form of human guidance to define progress.

Achieving open-ended capability growth therefore requires moving beyond supervised discovery. One possible path is adapting unsupervised skill discovery techniques from reinforcement learning to LLM-based agents, allowing reusable behaviors to emerge directly from interaction traces. Signals such as repeated trajectory patterns, attention regularities, or recurring subgoal structures may serve as implicit indicators of skill boundaries. An agent capable of extracting reusable competencies solely from its own experience would fundamentally change how agent capabilities scale, shifting learning from instruction-driven expansion toward self-organizing behavior.

### X-C Formal Verification Across Representations

A practical governance challenge arises from the diversity of skill representations. Skills expressed as executable code benefit from decades of software assurance techniques, including testing, static analysis, and sandboxing. In practice, however, many deployed skill libraries rely heavily on natural-language or policy-style skills because they are easier to author and distribute. Unfortunately, these representations are significantly harder to audit rigorously, creating a mismatch between expressive convenience and verifiability.

This gap becomes particularly visible in safety-sensitive deployments, where auditing requirements extend beyond simple correctness checks. Emerging approaches suggest combining multiple lightweight verification layers: rule-based analysis for executable components, semantic inspection for language-based policies, and reputation or behavioral monitoring across executions. The longer-term challenge is moving from static, pre-deployment inspection toward runtime verification capable of detecting context-dependent failures or delayed activation attacks that only appear under specific environmental conditions.

### X-D Robustness Under Environmental Drift

Even correctly implemented skills may fail over time as their operating environments evolve. Changes in APIs, tools, data formats, or surrounding workflows can gradually invalidate assumptions embedded in a skill, producing unintended behavior without any modification to the skill itself. This form of environmental drift creates an attack surface that operates indirectly: adversaries can manipulate external conditions rather than the skill artifact.

Despite its practical importance, proactive drift detection remains largely absent from current systems. Future work may focus on continuous monitoring mechanisms that track execution statistics, detect deviations from historical behavior, and correlate failures with environmental change signals. Such systems would treat skills as living components requiring maintenance, enabling automatic adaptation or retirement once reliability deteriorates. Addressing drift will likely become essential as agents transition from experimental settings to long-lived production deployments.

### X-E Governance Economics and Liability

Finally, the emergence of marketplace-style skill distribution introduces economic and governance questions that remain largely unexplored. Open skill ecosystems create strong incentives for contribution and innovation, but simultaneously expand the supply-chain attack surface. Our survey indicates that existing platforms rarely provide clear mechanisms for assigning responsibility when third-party skills cause harm, nor do they offer credible certification processes that align incentives with reliability.

Progress here requires integrating technical and economic design. Liability models must clarify responsibility among skill authors, platform operators, and users, while certification mechanisms should reward dependable skills and discourage risky ones. Understanding these dynamics may require agent-based economic modeling alongside empirical platform studies. As skill marketplaces mature, governance frameworks that combine accountability, certification, and incentive alignment will likely become as important as technical advances themselves.

## XI Conclusion

Agentic skills are reusable procedural modules for LLM agents. We structure the design space, analyze security risks, and show that skill quality critically affects agent performance. We close by outlining open challenges in discovery, verification, and governance for reliable skill-based agents.

## References

- [1] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, U. Alon, and G. Neubig (2024) WebArena: a realistic web environment for building autonomous agents. In International Conference on Learning Representations (ICLR), Note: arXiv:2307.13854 Cited by: §I, TABLE II, §V-J2, TABLE V, TABLE VII.

- [2] J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press (2024) SWE-agent: agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems (NeurIPS), Note: arXiv:2405.15793 Cited by: §I, §III-A, TABLE II, TABLE V.

- [3] Z. Ji, D. Wu, W. Jiang, P. Ma, Z. Li, and S. Wang (2025) Measuring and augmenting large language models for solving capture-the-flag challenges. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 603–617. Cited by: §I.

- [4] Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang (2023) HuggingGPT: solving AI tasks with ChatGPT and its friends in Hugging Face. In Advances in Neural Information Processing Systems (NeurIPS), Note: arXiv:2303.17580 Cited by: §I, TABLE II, TABLE V, §VI-F1.

- [5] C. Xie, C. Chen, F. Jia, Z. Ye, S. Lai, K. Shu, J. Gu, A. Bibi, Z. Hu, D. Jurgens, et al. (2024) Can large language model agents simulate human trust behavior?. Advances in Neural Information Processing Systems (NeurIPS) 37, pp. 15674–15729. Cited by: §I.

- [6] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, J. Wang, C. Zhang, Z. Wang, S. Yau, Z. Lin, L. Zhou, C. Ran, L. Xiao, C. Wu, and J. Schmidhuber (2024) MetaGPT: meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations (ICLR), Note: arXiv:2308.00352 Cited by: §I, TABLE II, TABLE V, §VI-F1.

- [7] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, A. H. Awadallah, R. W. White, D. Burger, and C. Wang (2024) AutoGen: enabling next-gen LLM applications via multi-agent conversation. In Conference on Language Modeling (COLM), Note: arXiv:2308.08155 Cited by: §I, TABLE V, §VI-F1.

- [8] J. R. Anderson, D. Bothell, M. D. Byrne, S. Douglass, C. Lebiere, and Y. Qin (2004) An integrated theory of the mind.. Psychological Review 111 (4), pp. 1036–1060. External Links: Document Cited by: §I, §II-B, §II-C, §III-A.

- [9] J. E. Laird (2012) The Soar cognitive architecture. MIT Press. Cited by: §I, §III-A.

- [10] R. S. Sutton, D. Precup, and S. Singh (1999) Between MDPs and semi-MDPs: a framework for temporal abstraction in reinforcement learning. Artificial Intelligence 112 (1–2), pp. 181–211. External Links: Document Cited by: §I, 2nd item, §II-A, §III-A, §VI-F1.

- [11] G. Zhang, H. Geng, X. Yu, Z. Yin, Z. Zhang, Z. Tan, H. Zhou, Z. Li, X. Xue, Y. Li, et al. The landscape of agentic reinforcement learning for llms: a survey. Transactions on Machine Learning Research (TMLR). Cited by: §I.

- [12] C. Qian, E. C. Acikgoz, Q. He, H. WANG, X. Chen, D. Hakkani-Tür, G. Tur, and H. Ji ToolRL: reward is all tool learning needs. In Annual Conference on Neural Information Processing Systems (NeurIPS), Cited by: §I.

- [13] H. Gao, J. Geng, W. Hua, M. Hu, X. Juan, H. Liu, S. Liu, J. Qiu, X. Qi, Q. Ren, et al. A survey of self-evolving agents: what, when, how, and where to evolve on the path to artificial super intelligence. Transactions on Machine Learning Research (TMLR). Cited by: §I.

- [14] X. Ma, Y. Gao, Y. Wang, R. Wang, X. Wang, Y. Sun, Y. Ding, H. Xu, Y. Chen, Y. Zhao, et al. (2026) Safety at scale: a comprehensive survey of large model and agent safety. Foundations and Trends in Privacy and Security 8 (3-4), pp. 1–240. Cited by: §I.

- [15] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J. Wen (2024) A survey on large language model based autonomous agents. Frontiers of Computer Science 18 (6), pp. 186345. Note: Extended from arXiv:2308.11432 External Links: Document Cited by: §I.

- [16] A. Shahriar, M. N. Rahman, S. Ahmed, F. Sadeque, and M. R. Parvez (2025) A survey on agentic security: applications, threats and defenses. arXiv preprint arXiv:2510.06445. Cited by: §I, §VII.

- [17] X. Huang, W. Liu, X. Chen, X. Wang, H. Wang, D. Lian, Y. Wang, R. Tang, and E. Chen (2024) Understanding the planning of LLM agents: a survey. arXiv preprint arXiv:2402.02716. Cited by: §I.

- [18] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang (2024) Large language model based multi-agents: a survey of progress and challenges. arXiv preprint arXiv:2402.01680. Cited by: §I.

- [19] A. Yehudai, L. Eden, A. Li, G. Uziel, Y. Zhao, R. Bar-Haim, A. Cohan, and M. Shmueli-Scheuer (2025) Survey on evaluation of LLM-based agents. arXiv preprint arXiv:2503.16416. Cited by: §I.

- [20] F. X. Fan, C. Tan, R. Wattenhofer, and Y. Ong (2026) Information fidelity in tool-using llm agents: a martingale analysis of the model context protocol. arXiv preprint arXiv:2602.13320. Cited by: §I.

- [21] Y. Qin, S. Hu, Y. Lin, W. Chen, N. Ding, G. Cui, Z. Zeng, X. Zhou, Y. Huang, C. Xiao, C. Han, Y. R. Fung, Y. Su, H. Wang, C. Qian, R. Tian, K. Zhu, S. Liang, X. Shen, B. Xu, Z. Zhang, Y. Ye, B. Li, et al. (2025) Tool learning with foundation models. ACM Computing Surveys (CSUR) 57 (4), pp. 101:1–101:40. External Links: Document Cited by: §I.

- [22] T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom (2023) Toolformer: language models can teach themselves to use tools. In Advances in Neural Information Processing Systems (NeurIPS), Note: arXiv:2302.04761 Cited by: §I, §II-B.

- [23] T. Guo, X. Chen, Y. Wang, R. Chang, S. Pei, N. V. Chawla, O. Wiest, and X. Zhang (2024) Large language model based multi-agents: a survey of progress and challenges. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI), pp. 8048–8057. Note: Survey track External Links: Document Cited by: §I.

- [24] C. Packer, V. Fang, S. G. Patil, K. Lin, S. Wooders, and J. E. Gonzalez (2023) MemGPT: towards LLMs as operating systems. arXiv preprint arXiv:2310.08560. Cited by: §II-B, §IV-C, TABLE II, TABLE V.

- [25] K. Hatalis, D. Christou, J. Myers, S. Jones, K. Lambert, A. Amos-Binks, Z. Dannenhauer, and D. Dannenhauer (2023) Memory matters: the need to improve long-term memory in llm-agents. In Proceedings of the AAAI Symposium Series (AAAI), Vol. 2, pp. 277–280. Cited by: §II-B.

- [26] B. Ma, Y. Jiang, X. Wang, G. Yu, Q. Wang, C. Sun, C. Li, X. Qi, Y. He, W. Ni, et al. (2025) SoK: semantic privacy in large language models. arXiv preprint arXiv:2506.23603. Cited by: §II-B.

- [27] J. White, Q. Fu, S. Hays, M. Sandborn, C. Olea, H. Gilbert, A. Elnashar, J. Spencer-Smith, and D. C. Schmidt (2023) A prompt pattern catalog to enhance prompt engineering with chatgpt. arXiv preprint arXiv:2302.11382. Cited by: §II-B.

- [28] D. S. Nau, T. Au, O. Ilghami, U. Kuter, J. W. Murdock, D. Wu, and F. Yaman (2003) SHOP2: an HTN planning system. Journal of Artificial Intelligence Research 20, pp. 379–404. External Links: Document Cited by: §II-B, §VI-F1.

- [29] A. S. Rao and M. P. Georgeff (1995) BDI agents: from theory to practice. In Proceedings of the First International Conference on Multi-Agent Systems (ICMAS), pp. 312–319. Cited by: §II-B.

- [30] R. E. Fikes and N. J. Nilsson (1971) STRIPS: a new approach to the application of theorem proving to problem solving. Artificial Intelligence 2 (3–4), pp. 189–208. External Links: Document Cited by: §II-B.

- [31] W. G. Chase and H. A. Simon (1973) Perception in chess. Cognitive Psychology 4 (1), pp. 55–81. External Links: Document Cited by: §II-C, §III-A.

- [32] X. Li, W. Chen, Y. Liu, S. Zheng, X. Chen, Y. He, Y. Li, B. You, H. Shen, J. Sun, et al. (2026) SkillsBench: benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670. Cited by: §II-C, §III-A, §IV-D, §V-J2, §V-E, §VIII-D, TABLE VII.

- [33] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar (2024) Voyager: an open-ended embodied agent with large language models. Transactions on Machine Learning Research (TMLR). Note: arXiv:2305.16291 Cited by: §III-A, §IV-A, §IV-C, TABLE II, §V-C, §V-E, TABLE V, 1st item, §VI-B, §VI-C, §IX-A.

- [34] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao (2023) ReAct: synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), Note: arXiv:2210.03629 Cited by: §III-A, §V-F.

- [35] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao (2023) Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), Note: arXiv:2303.11366 Cited by: §III-A, §IV-B, TABLE II, TABLE V, §VI-C, §VI-F1.

- [36] Z. Wang, S. Cai, A. Liu, Y. Jin, J. Hou, B. Zhang, H. Lin, Z. He, Z. Zheng, Y. Yang, X. Ma, and Y. Liang (2025) JARVIS-1: open-world multi-task agents with memory-augmented multimodal language models. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 47 (3), pp. 1894–1907. Note: Extended from arXiv:2311.05997 Cited by: TABLE II.

- [37] Z. Wang, S. Cai, A. Liu, X. Ma, and Y. Liang (2023) Describe, explain, plan and select: interactive planning with large language models enables open-world multi-task agents. In Advances in Neural Information Processing Systems (NeurIPS), Note: arXiv:2302.01560 Cited by: §IV-A, TABLE II.

- [38] M. F. Chen, N. Roberts, K. Bhatia, J. Wang, C. Zhang, F. Sala, and C. Ré (2023) Skill-it! a data-driven skills framework for understanding and training language models. In Advances in Neural Information Processing Systems (NeurIPS), Note: arXiv:2307.14430 Cited by: TABLE II, §VI-D.

- [39] C. Zhang, Z. Yang, J. Liu, Y. Han, X. Chen, Z. Huang, B. Fu, and G. Yu (2025) AppAgent: multimodal agents as smartphone users. In Proceedings of the CHI Conference on Human Factors in Computing Systems (CHI), pp. 70:1–70:20. Note: Extended from arXiv:2312.13771 Cited by: §IV-A, TABLE II, TABLE V, 1st item, §VI-B.

- [40] W. Tan, W. Zhang, X. Xu, H. Xia, Z. Ding, B. Li, B. Zhou, et al. (2025) Cradle: empowering foundation agents towards general computer control. In International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, Vol. 267, pp. 58658–58725. Note: Extended from arXiv:2403.03186 Cited by: §IV-C, TABLE II, §V-E, TABLE V.

- [41] A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y. Dong, and J. Tang (2024) AgentTuning: enabling generalized agent abilities for LLMs. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 3053–3077. Note: arXiv:2310.12823 External Links: Document Cited by: §IV-B, TABLE II, TABLE V, §VI-B, §VI-B.

- [42] X. Wang, Y. Chen, L. Yuan, Y. Zhang, Y. Li, H. Peng, and H. Ji (2024) Executable code actions elicit better LLM agents. In International Conference on Machine Learning (ICML), Note: arXiv:2402.01030 Cited by: §IV-D, TABLE II, §V-C, TABLE V.

- [43] B. Chen, C. Shu, E. Shareghi, N. Collier, K. Narasimhan, and S. Yao (2023) FireAct: toward language agent fine-tuning. arXiv preprint arXiv:2310.05915. Cited by: §IV-B, TABLE II, §VI-B.

- [44] B. Qiao, L. Li, X. Zhang, S. He, Y. Kang, C. Zhang, F. Yang, H. Dong, J. Zhang, L. Wang, M. Ma, P. Zhao, S. Qin, X. Qin, C. Du, Y. Xu, Q. Lin, S. Rajmohan, and D. Zhang (2023) TaskWeaver: a code-first agent framework. arXiv preprint arXiv:2311.17541. Cited by: TABLE II, TABLE V.

- [45] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gober, K. Hausman, et al. (2022) Do as i can, not as i say: grounding language in robotic affordances. In Conference on Robot Learning (CoRL), Note: arXiv:2204.01691 Cited by: §IV-A, TABLE II, §V-J2, TABLE V.

- [46] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein (2023) Generative agents: interactive simulacra of human behavior. In ACM Symposium on User Interface Software and Technology (UIST), Note: arXiv:2304.03442 Cited by: §IV-C, TABLE II, TABLE V.

- [47] Y. J. Ma, W. Liang, G. Wang, D. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar (2024) Eureka: human-level reward design via coding large language models. In International Conference on Learning Representations (ICLR), Note: arXiv:2310.12931 Cited by: §IV-B, TABLE II, §V-G, §VI-E, §IX-A.

- [48] K. Nottingham, P. Ammanabrolu, A. Suhr, Y. Choi, H. Hajishirzi, S. Singh, and R. Fox (2023) Do embodied agents dream of pixelated sheep: embodied decision making using language guided world modelling. In International Conference on Machine Learning (ICML), Note: arXiv:2301.12050 Cited by: §IV-A, §VI-C.

- [49] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, et al. (2022) Inner monologue: embodied reasoning through planning with language models. In Conference on Robot Learning (CoRL), Note: arXiv:2207.05608 Cited by: §IV-B, §V-J2.

- [50] E. Gamma, R. Helm, R. Johnson, and J. Vlissides (1994) Design patterns: elements of reusable object-oriented software. Addison-Wesley Professional. Cited by: §V-A.

- [51] Microsoft (2023) Semantic Kernel: a lightweight SDK for AI agent development. Note: https://github.com/microsoft/semantic-kernelAccessed: 2026-02-21 Cited by: §V-B.

- [52] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng (2023) Code as policies: language model programs for embodied control. In IEEE International Conference on Robotics and Automation (ICRA), Note: arXiv:2209.07753 Cited by: §V-J2, §V-C.

- [53] I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg (2023) ProgPrompt: generating situated robot task plans using large language models. In IEEE International Conference on Robotics and Automation (ICRA), Note: arXiv:2209.11302 Cited by: §V-J2, §V-C.

- [54] A. Zhou, K. Yan, M. Shlapentokh-Rothman, H. Wang, and Y. Wang (2024) Language agent tree search unifies reasoning, acting, and planning in language models. In International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, Vol. 235, pp. 62138–62160. Note: arXiv:2310.04406 Cited by: §V-D, TABLE V, §VI-F1.

- [55] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi (2023) Self-instruct: aligning language models with self-generated instructions. In Annual Meeting of the Association for Computational Linguistics (ACL), pp. 13484–13508. External Links: Document Cited by: §V-G, §VI-E.

- [56] C. Qian, C. Han, Y. R. Fung, Y. Qin, Z. Liu, and H. Ji (2023) CREATOR: tool creation for disentangling abstract and concrete reasoning of large language models. In Findings of the Association for Computational Linguistics (EMNLP), Note: arXiv:2305.14318 Cited by: §V-G, §VI-E.

- [57] Anthropic (2024) Introducing the model context protocol. Note: https://www.anthropic.com/news/model-context-protocolAccessed: 2026-02-21 Cited by: §V-H.

- [58] Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, et al. (2024) ToolLLM: facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations (ICLR), Note: arXiv:2307.16789 Cited by: §V-H, TABLE V.

- [59] OpenClaw Project (2026) OpenClaw: personal ai assistant. Note: https://github.com/openclaw/openclawOfficial repository (216k stars at access time). Accessed: 2026-02-22 Cited by: §V-H, TABLE V.

- [60] Alex and Oren Yomtov (2026) ClawHavoc: 341 malicious clawed skills found by the bot they were targeting. Note: https://www.koi.ai/blog/clawhavoc-341-malicious-clawedbot-skills-found-by-the-bot-they-were-targetingKoi Research blog post; update dated Feb 16, 2026 reports 824 malicious skills. Accessed: 2026-02-22 Cited by: §V-H, §VII-F, §VII-F.

- [61] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su (2023) Mind2Web: towards a generalist agent for the web. In Advances in Neural Information Processing Systems (NeurIPS), Note: Spotlight. arXiv:2306.06070 Cited by: §V-J2, §VIII-A, TABLE VII.

- [62] T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, Y. Liu, Y. Xu, S. Zhou, S. Savarese, C. Xiong, V. Zhong, and T. Yu (2024) OSWorld: benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems (NeurIPS), Note: Datasets and Benchmarks track. arXiv:2404.07972 Cited by: §V-J2, §VIII-A, TABLE VII.

- [63] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan (2024) SWE-bench: can language models resolve real-world GitHub issues?. In International Conference on Learning Representations (ICLR), Note: arXiv:2310.06770 Cited by: §V-J2, TABLE VII.

- [64] H. Ravichandar, A. S. Polydoros, S. Chernova, and A. Billard (2020) Recent advances in robot learning from demonstration. Annual Review of Control, Robotics, and Autonomous Systems 3, pp. 297–330. Cited by: §V-J2.

- [65] Significant Gravitas (2023) AutoGPT: an autonomous GPT-4 experiment. Note: https://github.com/Significant-Gravitas/AutoGPTAccessed: 2026-02-21 Cited by: §VI-C.

- [66] L. Wang, X. Zhang, H. Su, and J. Zhu (2024) A comprehensive survey of continual learning: theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 46 (8), pp. 5362–5383. Cited by: §VI-C.

- [67] C. Zhang, K. Yang, S. Hu, Z. Wang, G. Li, Y. Sun, C. Zhang, Z. Zhang, A. Liu, S. Zhu, X. Chang, J. Zhang, F. Yin, Y. Liang, and Y. Yang (2024) ProAgent: building proactive cooperative agents with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), Vol. 38, pp. 17591–17599. Note: arXiv:2308.11339 Cited by: §VI-F1.

- [68] P. Ladisa, H. Plate, M. Martinez, and O. Barais (2023) SoK: taxonomy of attacks on open-source software supply chains. In IEEE Symposium on Security and Privacy (SP), pp. 1509–1526. Cited by: §VII-A.

- [69] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz (2023) Not what you’ve signed up for: compromising real-world LLM-integrated applications with indirect prompt injection. In ACM Workshop on Artificial Intelligence and Security (AISec), Note: arXiv:2302.12173 Cited by: §VII-B, §VII-C.

- [70] B. Quintero (2026) From automation to infection: how OpenClaw AI agent skills are being weaponized. Note: https://blog.virustotal.com/2026/02/from-automation-to-infection-how.htmlVirusTotal Blog, February 2, 2026. Accessed: 2026-02-22 Cited by: §VII-F, §VII-F.

- [71] B. Van (2026) Agent skills guard. Note: https://github.com/brucevanfdm/agent-skills-guardDesktop scanner/manager; README reports 8 risk categories and 22 hard-trigger rules. Accessed: 2026-02-22 Cited by: 3rd item, §VII-F.

- [72] G. Singh (2026) SkillGuard: AI agent security scanner. Note: https://skillgaurd.up.railway.app/Website and linked source repo describe AST analysis for JS/TS, 9-language coverage, and 20+ attack patterns. Accessed: 2026-02-22 Cited by: §VII-F.

- [73] Y. Liu, G. Deng, Z. Xu, Y. Li, Y. Zheng, Y. Zhang, L. Zhao, T. Zhang, K. Wang, and Y. Liu (2023) Jailbreaking chatgpt via prompt engineering: an empirical study. arXiv preprint arXiv:2305.13860. Cited by: 2nd item.

- [74] G. Mialon, C. Fourrier, T. Wolf, Y. LeCun, and T. Scialom (2024) GAIA: a benchmark for general AI assistants. In International Conference on Learning Representations (ICLR), Note: Poster. arXiv:2311.12983 Cited by: TABLE VII.

- [75] X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang, S. Zhang, X. Deng, A. Zeng, Z. Du, C. Zhang, S. Shen, T. Zhang, Y. Su, H. Sun, M. Huang, Y. Dong, and J. Tang (2024) AgentBench: evaluating LLMs as agents. In International Conference on Learning Representations (ICLR), Note: arXiv:2308.03688 Cited by: §VIII-D, TABLE VII.

- [76] C. Rawles, S. Clinckemaillie, Y. Chang, J. Waltz, G. Lau, M. Fair, A. Li, W. E. Bishop, W. Li, F. Campbell-Ajala, D. K. Toyama, R. J. Berry, D. Tyamagundlu, T. P. Lillicrap, and O. Riva (2025) AndroidWorld: a dynamic benchmarking environment for autonomous agents. In International Conference on Learning Representations (ICLR), Note: arXiv:2405.14573 Cited by: TABLE VII.

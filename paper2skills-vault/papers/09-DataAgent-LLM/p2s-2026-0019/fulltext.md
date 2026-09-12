<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2607.22115
     paper_id : p2s-2026-0019
     source   : https://arxiv.org/html/2607.22115v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Benchmarking Text-to-SQL under Role-Based Access Control

Yang Fei Affiliation: National University of Singapore email: yfei11@u.nus.edu , Yangfan Jiang Affiliation: National University of Singapore email: jyangfan@u.nus.edu , Yin Yang Affiliation: Hamad Bin Khalifa University email: yyang@hbku.edu.qa and Xiaokui Xiao Affiliation: National University of Singapore email: xkxiao@nus.edu.sg

###### Abstract.

Given a database $\mathcal{S}$ and a natural language question $Q$, text-to-SQL systems aim to generate an SQL query that correctly answers $Q$ when executed against $\mathcal{S}$. Currently, popular text-to-SQL benchmarks mostly assume unrestricted access to $\mathcal{S}$; in practice, however, user access is often restricted, e.g., through role-based access control (RBAC) policies. This leads to a potential disconnect between benchmarking results and real-world performance: an LLM with high benchmark scores might perform poorly in an access-controlled environment, by frequently violating RBAC, or rejecting a query $q$ that could be answered with only permitted data in $\mathcal{S}$.

Motivated by this, we present a comprehensive text-to-SQL benchmarking framework with realistic RBAC constraints, which features an LLM-assisted workflow that augments existing text-to-SQL benchmarks with plausible user roles and access policies. To do so, we formulate the problem of role synthesis as a structured reasoning process over the database schema, in which the LLM first infers the application context from the schema, and then derives role responsibilities and access scopes consistent with this context. This process is audited by human-in-the-loop quality control, in which domain experts perform metric-guided screening on the generated roles. Besides the augmented dataset, the proposed framework also contains evaluation metrics that identify RBAC-specific failure modes, and disentangle SQL utility from access-control compliance.

We apply the proposed framework to several widely-used benchmarks, and conduct a systematic empirical study of state-of-the-art text-to-SQL systems. The results show that many solutions (especially open-weight LLMs) with high benchmarking scores under an unrestricted setting suffer sharp performance degradation once access constraints are in place, due to frequent RBAC violations.

## 1. Introduction

Structured query language (SQL) has long served as the lingua franca for database interactions. However, for many users, the complexity of SQL remains a significant barrier to effective data access (Li and Jagadish, 2014; Kim et al., 2020). Text-to-SQL systems aim to eliminate this barrier by allowing users to express their intent in natural language and translating it into SQL. Recent advances in large language models (LLMs), with strong coding and reasoning capabilities, have enabled substantial progress in text-to-SQL performance (Li et al., 2023; Gao et al., 2024; Luoma and Kumar, 2025; Xie et al., 2025; Chen et al., 2025).

State-of-the-art LLM-based text-to-SQL solutions typically combine prompt engineering and schema-aware reasoning to achieve high execution accuracy (Zhou et al., 2024). To evaluate and compare different approaches, several benchmarks have been established, including Spider (Yu et al., 2018; Lei et al., 2025), BIRD (Li et al., 2023), and LiveSQLBench (BIRD-SQL Team, 2025). On these benchmarks, recent systems report strong performance under the assumption that generated SQL queries are executed directly by the DBMS without additional constraints.

### 1.1. Text-to-SQL in the Wild

In real-world DBMS environments, database access is rarely unrestricted. To satisfy security, privacy, and regulatory requirements, production DBMSs enforce access control policies that determine which data a user is authorized to access. One of the most widely adopted mechanisms is role-based access control (RBAC), under which users are assigned roles with permissions over specific tables, attributes, operations, or views (Sandhu et al., 1996; Ferraiolo et al., 2003). In such settings, an unauthorized SQL query could trigger an execution error at the DBMS layer (e.g., MySQL’s error 1142), or silent result suppressions (e.g., Snowflake’s dynamic data masking) that are non-trivial to debug.

This exposes a fundamental gap between real-world deployments and current text-to-SQL evaluation practices. RBAC enforcement should remain deterministic at the DBMS layer; however, modern text-to-SQL systems typically generate SQL outside the DBMS and must therefore make access-sensitive decisions before execution. In practice, this separation can lead to repeated execution failures, brittle system behavior, and subtle mismatches between user intent and the query results returned to the user, often requiring costly manual intervention. A DBMS-side checker can reject unauthorized SQL at execution time, but by that point the upstream text-to-SQL workflow may already have failed to produce a usable response.

To illustrate this issue, consider the following example.

###### Example 0.

Consider an enterprise analytics application in which a text-to-SQL framework translates user requests into SQL queries executed by a backend DBMS. Suppose a user with the “Sales Analyst” role issues the natural language query: “Which departments have the highest average sales revenue per employee?” A vanilla text-to-SQL system may generate the following SQL query:

⬇

SELECT e.department, AVG(s.amount) AS avg_revenue

FROM employees e

JOIN sales s ON e.emp_id = s.emp_id

GROUP BY e.department;

Under a typical RBAC policy, however, the “Sales Analyst” role may be permitted to access the sales table but not sensitive employee information such as identifiers or department assignments. Although the generated query is syntactically valid and semantically aligned with the user’s intent, it violates the RBAC policy and is rejected by the DBMS at execution time. As a result, even if a deterministic checker blocks the query, the application-level workflow fails to complete: the text-to-SQL component neither produces an authorized query nor correctly refuses the request before execution. The system is therefore safe at the DBMS layer but not useful at the application layer, leaving the request unresolved and requiring retries or manual intervention. More generally, such a checker only determines whether a generated SQL query is authorized; it does not reveal whether the system should have refused earlier, whether it missed a policy-compliant SQL alternative, or whether it over-refused an answerable request.

Despite the practical importance of access control, most existing text-to-SQL benchmarks still focus on translation accuracy under unrestricted database access. While a few recent efforts (Weng et al., 2026; Subramaniam and Krishnan, 2024; Klisura et al., 2025) have begun to consider access control, they mainly target coarse-grained settings (e.g., database- or table-level permissions) and lack systematic benchmarking pipelines or comprehensive empirical analysis, offering limited insight into practical text-to-SQL deployments under RBAC. To our knowledge, established benchmarks and evaluation protocols (Yu et al., 2018; Li et al., 2023; BIRD-SQL Team, 2025; Ren et al., 2024; Gao et al., 2024; Zhang et al., 2024; Luoma and Kumar, 2025; Yang et al., 2025; Xie et al., 2025; Chen et al., 2025; Fan et al., 2024a; Fu et al., 2023; Chung et al., 2025) do not account for RBAC. This leaves an overlooked failure mode that we term an RBAC-rejected success, where a generated SQL query is syntactically correct and would be judged correct under unrestricted-access evaluation, yet is rejected at execution time due to RBAC violations. Since existing benchmarks assume unrestricted access, standard metrics such as execution accuracy (EX) fail to capture such failures. As a result, current benchmarks and metrics do not measure a text-to-SQL system’s ability to generate authorized SQL queries under realistic access control constraints, leaving a practically important class of errors largely invisible.

As we shall demonstrate in Section 5, RBAC-rejected successes are in fact common. Systems that achieve high EX scores on existing benchmarks may frequently fail once RBAC rules are enforced, since simply prompting LLMs with access policies does not reliably prevent unauthorized SQL generation. Moreover, heuristic remedies such as schema filtering, prompt engineering, and fine-tuning are insufficient; LLMs may still hallucinate relationships or make flawed assumptions, leading to invalid SQL and failed executions. These observations indicate that RBAC alignment must be treated as a first-class concern in text-to-SQL systems rather than an afterthought handled solely at execution time. Accordingly, a critical mismatch remains between the strong performance reported on existing benchmarks and the requirements of practical, access-controlled deployments in real DBMS environments.

### 1.2. Contributions

To fill this gap, we present a comprehensive benchmarking framework for evaluating the RBAC-enforcing capabilities of text-to-SQL systems, as illustrated in Figure 1. In a nutshell, the proposed framework takes as input an existing text-to-SQL dataset, augments the data with plausible roles and access policies generated with an LLM with human-in-the-loop validation, and defines new evaluation metrics that evaluate both SQL correctness and RBAC compliance. We apply this framework to three widely-used benchmarks, namely Spider (Yu et al., 2018; Lei et al., 2025), BIRD (Li et al., 2023), and LiveSQLBench (BIRD-SQL Team, 2025), producing RBAC-augmented datasets that support the evaluation of both access-policy compliance and SQL utility, and conduct a systematic empirical study that reveals previously unexamined behaviors of text-to-SQL systems under access control.

Our goal is not to replace deterministic DBMS-side enforcement with an LLM, but to evaluate the upstream text-to-SQL problem. We study whether a system can make the correct allow/deny decision given the natural language query and role policy and, when allowed, generate correct, policy-compliant SQL. A DBMS-side checker remains complementary, but does not resolve this evaluation problem. It can reject unauthorized SQL, yet does not indicate whether the system should have refused earlier, generated a policy-compliant alternative, or how much utility is lost to over-refusal.

Challenges. Designing an RBAC-aware text-to-SQL benchmark faces four challenges. First, synthesized roles and fine-grained policies must be semantically meaningful: they should reflect plausible application contexts while avoiding trivial or redundant permission patterns. Second, reliable ground truth requires precise permission verification, as each reference SQL must be analyzed to identify all required permissions over tables, columns, and operations. Third, automatically generated roles and policies may be unrealistic or imbalanced, requiring human validation of semantic plausibility, permission coverage, and role distinctness. Finally, existing text-to-SQL metrics do not capture RBAC-specific failures, motivating new metrics that jointly measure SQL utility and policy compliance.

*Figure 1. Overview of the proposed RBAC-aware dataset construction and benchmarking pipeline.*

Framework overview. At the core of the proposed framework is an automated and auditable pipeline that augments existing text-to-SQL benchmarks with realistic RBAC settings while preserving their original linguistic diversity and query semantics. For each database, the pipeline uses LLMs to synthesize plausible user roles and fine-grained access policies. To avoid degenerate role definitions, role synthesis follows a structured reasoning process tailored to text-to-SQL under RBAC constraints. Specifically, the model first infers an application context from the database schema, and then derives role responsibilities and access policies consistent with the schema structure. The resulting policies operate at the column-operation level, enabling evaluation under restricted access.

We further incorporate human-in-the-loop validation to ensure that the synthesized roles and policies are meaningful and faithful to practical settings. LLM-generated role configurations are first screened using interpretable quality metrics, including denial rate, coverage balance across operations, role distinctness, and semantic alignment with the database schema, and are then reviewed by human evaluators for plausibility. Configurations that fail these checks are rejected, with failure signals fed back to the synthesis stage to refine the role definitions and access policies. For permission assignment, each reference SQL query is parsed using SQLGlot (Mao, 2023) to extract the required permissions, which are then validated against each role’s access policy. This process produces RBAC-aware ground truth that distinguishes between allowed queries, proper refusals, and unsafe cases in which a semantically correct query would violate access constraints.

Finally, we introduce evaluation metrics that explicitly disentangle SQL utility from policy compliance. These metrics capture not only whether a generated query produces the correct result under full access, but also whether it respects user permissions and avoids RBAC-rejected successes, namely cases in which syntactically correct SQL violates access policies.

Benchmarking results and analysis. We apply this framework to several widely used text-to-SQL benchmarks, resulting in large-scale evaluation resources spanning 53 databases, 399 tables, and 3,353 columns, with a total of 21,502 RBAC-annotated query instances. Using these benchmarks, we conduct a systematic empirical study of state-of-the-art text-to-SQL systems and observe persistent RBAC-related failures. Failure analysis reveals the refusal-cliff phenomenon, i.e., an LLM may correctly follow RBAC rules during reasoning, yet still output a final SQL query that violates access control policies. Meanwhile, common mitigation measures, including supervised fine-tuning and in-context learning, show limited effectiveness in reducing RBAC failures. These findings highlight a substantial gap between strong benchmark performance under unrestricted settings and the requirements of access-controlled deployments, motivating text-to-SQL systems that explicitly reason about RBAC constraints.

We release the complete benchmarking pipeline, evaluation toolkit, and the resulting RBAC-augmented datasets in a public repository (Yang et al., 2026), to support reproducibility and future research.

## 2. Problem Setting

This section formalizes the RBAC-compliant text-to-SQL task studied in this paper, including its objectives and schema exposure mechanisms. We then describe the LLM-based role (profile) synthesis problem used to construct our benchmark.

### 2.1. Text-to-SQL under RBAC

LLM-based text-to-SQL. Let $\mathcal{S}=(\mathcal{T},\mathcal{C},\mathcal{R})$ denote a relational database schema, where $\mathcal{T}$ is the set of tables, $\mathcal{C}$ the set of columns, and $\mathcal{R}$ the set of foreign-key relations. Given a natural-language question $Q$ and a schema text $\Sigma$ describing $\mathcal{S}$, the goal of text-to-SQL is to generate a SQL program $Y$ whose execution result matches that of a canonical gold standard query $Y^{\star}$.

An LLM-based system forms a prompt $\mathcal{P}(Q,\Sigma)$ and decodes an SQL program token by token using a language model $M$:

$P_{M}(Y\mid\mathcal{P}(Q,\Sigma))=\prod_{i=1}^{|Y|}P_{M}\!\left(y_{i}\,\middle|\,\mathcal{P}(Q,\Sigma),\,y_{<i}\right).$ | | | |

Through the model $M$, a candidate program $\hat{Y}$ is obtained by greedy or sampling-based decoding, and then executed for evaluation.

Text-to-SQL with RBAC. We extend the standard text-to-SQL setting by introducing role-based access control. Each role $r$ is associated with an access policy $\Pi_{r}\subseteq\mathcal{T}\times\mathcal{C}\times\mathcal{O}$, where $\mathcal{O}$ denotes the set of SQL operations (e.g., SELECT, INSERT, UPDATE, DELETE). A permission $(t,c,o)\in\Pi_{r}$ authorizes role $r$ to apply operation $o$ to column $c$ of table $t$. Given a gold SQL $Y^{\star}$, we extract the required permission set $\mathsf{Perm}(Y^{\star})\subseteq\mathcal{T}\times\mathcal{C}\times\mathcal{O}$, corresponding to all $(\mathsf{table},\mathsf{column},\mathsf{operation})$ triples exercised by $Y^{\star}$. The ground-truth access decision for an instance $(Q,\mathcal{S},r)$ is defined as:

$y\;=\;\begin{cases}\textsf{allow},&\text{if }\;\mathsf{Perm}(Y^{\star})\subseteq\Pi_{r},\\[2.0pt]
\textsf{deny},&\text{otherwise}.\end{cases}$ | | | |

This construction follows the standard assumption adopted in text-to-SQL benchmarks that the gold SQL, authored by domain experts, accurately captures the data access and operations required to answer the query. Accordingly, in our benchmark, we treat the gold SQL as a correct and complete operational specification of the user’s intended data access behavior.

A system receives $(Q,\Sigma,r,\Pi_{r})$, where $\Pi_{r}$ is the serialized role policy provided in the prompt, and returns either a refusal symbol $\bot$ or a SQL program $\hat{Y}$. Emitting $\bot$ induces $\hat{y}=\textsf{deny}$; emitting $\hat{Y}$ induces $\hat{y}=\textsf{allow}$, after which $\hat{Y}$ is evaluated along two dimensions: (i) RBAC compliance, by checking whether $\mathrm{Perm}(\hat{Y})\subseteq\Pi_{r}$, and (ii) execution correctness, by comparing its execution result with that of the gold SQL $Y^{\star}$. Notably, even when $Y^{\star}$ is authorized under the policy $\Pi_{r}$, the generated SQL $\hat{Y}$ may still violate RBAC by invoking unnecessary or unauthorized tables, columns, or operations.

We consider two modes for the schema available: (i) $\Sigma^{\mathrm{full}}$, which describes all information in $\mathcal{S}$; (ii) $\Sigma^{\mathrm{role}}(r)$, which describes only schema elements permitted by $\Pi_{r}$. This controls the information exposed at prompt time without changing the underlying database.

Task objective. For each instance, the model must produce the correct access decision $y$ under the role’s RBAC policy. Conditional on both the ground-truth and predicted decision being allow, the model must further generate a SQL program $\hat{Y}$ that is execution-correct with respect to $Y^{\star}$ and compliant with the role’s fine-grained permissions. This formulation naturally induces a two-stage task structure, separating policy compliance from SQL generation quality while remaining compatible with execution-based evaluation.

Scope and limitations. Our benchmark focuses on explicit RBAC compliance: whether the system output references only the tables, columns, and operations authorized by the role policy $\Pi_{r}$. It does not model inference-based leakage, such as inferring salary ranges from authorized columns hourly_wage and hours_worked when salary is denied. Such indirect leakage requires policy models beyond standard RBAC, such as inference control or semantic privacy, and is left as future work in Section 8.

### 2.2. RBAC Role Synthesis

To construct a realistic benchmark for RBAC-compliant text-to-SQL, an important task is to systematically synthesize a set of plausible role profiles for a given database schema $\mathcal{S}$. We refer to this process as role synthesis. Given $\mathcal{S}$, the objective is to generate a role set $\mathbf{R}_{\mathcal{S}}=\{r_{1},r_{2},\dots,r_{k}\}$, where each role $r_{i}$ is defined as a tuple $(n_{i},d_{i},\Pi_{r_{i}})$ consisting of a role name $n_{i}$, a natural-language description $d_{i}$, and a fine-grained RBAC policy $\Pi_{r_{i}}$.

*Table 1. Databases, tables, columns, and query counts per data source with and without role design.*

| Data Source | DBs | Tables | Columns | w/o role | w/ role |

| Spider | 20 | 80 | 439 | 1,034 | 6,926 |

| BIRD | 11 | 75 | 798 | 1,534 | 10,175 |

| LiveSQLBench | 22 | 244 | 2,116 | 592 | 4,401 |

| Total | 53 | 399 | 3,353 | 3,160 | 21,502 |

An ideal role set should satisfy the following key properties. First, the synthesized roles must be semantically plausible and reflect real-world job functions and responsibilities that are logically grounded in the application domain of the database $\mathcal{S}$. For example, in an enterprise data analytics setting such as in Example 1.1, roles such as Sales Analyst and HR Admin are coherent, whereas Student and Teacher are not. Second, the roles should represent a reasonable division of responsibilities. The objective is not to create a strict partition (e.g., creating a separate role for every single table), as real-world roles often have overlapping permissions, but to avoid trivial or redundant role definitions. Ideally, generated roles should be semantically distinct, and reflect a realistic organizational structure.

Formally, automated role synthesis seeks to construct a role set $\mathbf{R}_{\mathcal{S}}$ that balances semantic plausibility and role distinctness. In our framework, this synthesis process is implemented using an LLM as a generative backend, and is realized through a structured pipeline that combines constrained generation with human-in-the-loop screening and iterative refinement, ensuring that the resulting role set $\mathbf{R}_{\mathcal{S}}$ satisfies the above criteria.

## 3. RBAC-Aware Benchmark Construction

We design a three-step, automated pipeline for constructing RBAC-aware text-to-SQL datasets, illustrated in Figure 1. The pipeline augments existing text-to-SQL benchmarks with role information and access-control semantics by leveraging state-of-the-art LLMs for role synthesis and structured analysis for permission verification. In what follows, Section 3.1 describes the source text-to-SQL datasets, Section 3.2 presents our role generation framework, and Section 3.3 details the RBAC dataset construction process.

### 3.1. Data Sources and Pre-Processing

We ground our study on three well-established text-to-SQL corpora: (i) Spider, a benchmark for complex cross-domain semantic parsing, equipped with a public evaluation framework and curated SQLite resources (Yu et al., 2018); (ii) BIRD, which incorporates large-scale, real-world databases from diverse professional domains, thereby strengthening realism in data noise, external knowledge dependency, and query complexity (Li et al., 2023); and (iii) LiveSQLBench-Base-Full-v1, which simulates end-to-end workloads from enterprise practices, supplemented with auditable executing and testing scripts (BIRD-SQL Team, 2025). For simplicity, hereafter we use the original names Spider, BIRD, and LiveSQLBench to denote the filtered source datasets.

To achieve a unified RBAC-aware evaluation while preserving the original design intents of the source datasets, we apply minimal and auditable preprocessing. Specifically, we adopt the official development splits of Spider and BIRD to support reproducible experiments without relying on closed-source test servers. For LiveSQLBench, we keep analytical queries together with administrative management operations to broaden the task surface across all queries. We also utilize the official database backend from each data source, SQLite execution for Spider and BIRD, and PostgreSQL for LiveSQLBench. We further develop a general-purpose evaluator that supports execution with values based on these backends. The resulting RBAC-aware evaluation pipeline preserves the original question statements, gold SQL answers, data content, and official database backends. Table 1 reports the affected counts alongside corpus statistics.

*Figure 2. Role Coverage and Semantic Alignment. Each row (e.g., wta_1) corresponds to a database. Each circular marker represents a synthesized role, with size indicating the proportion of accessible columns and horizontal axis representing semantic similarity to the schema (purple: low, yellow: high). Red triangles denote DataOperator roles.*

### 3.2. Automated Role and Policy Synthesis

As described in Section 2.2, given a database $\mathcal{S}$, we synthesize a set of user roles $\mathbf{R}_{\mathcal{S}}$ using an automated role synthesis pipeline with an LLM and human-in-the-loop validation. The LLM is provided with structured schema information, including table names, column attributes, data types, primary keys, and foreign key relationships, which together define the logical organization of the database.

A naive instantiation of this idea is to present the schema as plain text and directly prompt the model to generate roles. In practice, however, this approach often produces incoherent or overly fragmented role sets that fail to capture implicit domain relationships and realistic access patterns. For example, consider a school database containing tables StudentGrade, TeacherSalary, and CampusBuilding. When prompted directly, the model may generate isolated roles such as Student with access only to StudentGrade, Teacher with access only to TeacherSalary, and Architect with access only to CampusBuilding. While such roles may satisfy the basic schema coverage constraint in Section 2.2, they fail to reflect realistic role responsibilities. In practice, teachers require access to student grades for assessment, and both students and teachers typically share access to campus facilities.

Context-aware role and policy construction. To address these limitations, we adopt a structured role synthesis strategy that constrains the model’s inference process using schema semantics, rather than treating role generation as a free-form text completion task. Specifically, we decompose role synthesis and policy construction into a sequence of semantically grounded reasoning steps that guide the model to reason about the database at multiple levels of abstraction.

As illustrated in Step 1 of Figure 1, the model is first guided to infer a coherent application context implied by the database schema. This step requires the model to reason over table semantics, attribute names, and foreign key relationships to identify high-level usage scenarios, such as grading, enrollment management, or administrative workflows in an academic database. The inferred context serves as an intermediate semantic representation that anchors subsequent role construction. Conditioned on the inferred application context, the model then derives a small set of user roles together with their intended responsibilities. Each role is required to correspond to a meaningful organizational function, rather than an arbitrary subset of tables and columns. The synthesis process is explicitly structured to allow realistic overlap and dependency among roles, reflecting common access patterns in real DBMSs.

After synthesizing the role set, the model proceeds to derive a corresponding access control policy through a context-aware reasoning stage. Rather than assigning permissions independently, the model derives fine-grained access scopes that are consistent with the role’s responsibilities, including column-level and operation-level permissions, depending on the target benchmark. By separating context inference, role responsibility definition, and permission derivation, this structured reasoning process reduces degenerate role definitions and encourages semantically coherent and internally consistent RBAC policies. Finally, we synthesize scoped administrator roles to better reflect practical DBMS deployments. Each database is assigned 2 or 3 DataOperator roles with overlapping but incomplete access scopes. These roles provide broad schema coverage while still creating non-trivial deny cases for evaluation. Details are deferred to Appendix B.

Automated screening and human validation. To ensure that the synthesized roles and their corresponding access policies are meaningful and faithful to practical settings, we incorporate a human-in-the-loop validation stage into the role synthesis pipeline. The validation process follows a reject-regenerate paradigm that combines automatic quality screening with targeted human inspection, while maintaining explicit records of validation signals and regeneration decisions for auditability purposes.

We first perform an automatic quality check on each synthesized role configuration using a set of interpretable metrics, including denial rate, coverage balance across query operations, role distinctness measured via permission overlap, and semantic alignment between role descriptions and the database schema; details of the metrics definitions are deferred to Appendix A. These checks are fully automated and deterministic, producing explicit pass or fail signals together with diagnostic metrics that characterize the quality of the generated roles. When a role configuration fails any check, the specific failure reasons and associated metric diagnostics are recorded and packaged as structured feedback to the role synthesis stage. The context-grounded role and policy generation process is then re-executed using this feedback, enabling iterative refinement in a transparent and reproducible manner. The automatic reject-regenerate loop iterates until all quality checks are satisfied.

Once a role configuration passes all automatic checks, it is passed to human validation. Four annotators with database and access-control expertise first complete a lightweight calibration on held-out cases to align the review criteria. They then review each synthesized role configuration, including its roles and policies, for semantic plausibility and consistency with the inferred application context. Each configuration receives a binary accept/reject judgment and is accepted only if at least three of the four annotators approve it; otherwise, it is rejected and regenerated using the collected rejection reasons as structured feedback. Only role configurations that pass both automatic screening and human validation are retained for subsequent permission verification and dataset construction. Note that human validation is performed at the database-level role configuration level (53 in total), not at the expanded role-query instance level. Once a configuration is accepted, allow/deny labels for all role-query pairs are produced automatically via structured SQL analysis and policy matching. Detailed validation statistics, including first-pass acceptance, regeneration, manual revision, and total annotation time, are reported in Appendix A.3.4.

Validation via semantic alignment. We further assess whether synthesized roles are semantically aligned with the underlying database schema. For each role, we concatenate its name and description and embed the resulting text using a sentence embedding model (OpenAI/text-embedding-3-small in our experiments). We similarly embed a schema description formed from the database’s table and column names. We compute cosine similarity between the role and schema embeddings as a lightweight proxy for semantic alignment. For example, FinancialAuditor should be closer to a database with transactions and accounts tables than to one with players and matches. This check helps verify that LLM-generated roles are coherent with their assigned databases, rather than arbitrary.

Figure 2 reports the results on the three source datasets. Across all datasets, most synthesized roles show high semantic alignment with the schema while covering only a subset of the database access space. In contrast, DataOperator roles show lower semantic similarity, consistent with their design as scoped administrators. This pattern indicates that ordinary synthesized roles capture role-specific database semantics, while administrators mainly serve as structural safeguards rather than semantically specialized roles.

Human experts inspect the visualized results, identify outlier roles with low semantic similarity to the application context (e.g., the leftmost role in the singer database of Spider), and provide feedback to the LLM for regeneration.

*Figure 3. Query distribution and difficulty. Stacked bars show allowed (green) vs. denied (orange) query execution rates for each dataset and difficulty level/task category.*

### 3.3. RBAC-Aware Instance Construction

As shown in Figure 1, after obtaining a validated role set $\mathbf{R_{S}}$ and the corresponding access policies for each database, the dataset construction phase expands each source text-to-SQL instance into role-conditioned evaluation instances. In particular, for each example $(\mathcal{S},Q,Y^{\star})$ in a source dataset, where $Q$ is the natural language question and $Y^{\star}$ is the gold SQL, the framework pairs it with every synthesized role $r\in\mathbf{R_{S}}$ associated with the same database. Each resulting instance inherits the original database identifier, question, and gold SQL, and is augmented with role-specific information, including the role description and the corresponding access policy.

A critical component of this phase is the permission verification mechanism, which deterministically labels each role-query pair as allowed or denied. We perform structured SQL analysis to extract the permissions required by the gold SQL $Y^{\star}$. Specifically, we parse $Y^{\star}$ using SQLGlot (Mao, 2023) to deterministically recover the set of referenced base tables and accessed columns, and map $Y^{\star}$ to its CRUD operation type. To obtain a complete and accurate dependency set, we analyze clause constructs like JOIN, GROUP BY, ORDER BY, and predicate filters, which may introduce additional column references in a complex SQL. We then validate these extracted requirements against the role’s access policy to decide whether the query should be allowed. The extracted requirement set is then matched against the role’s fine-grained access policy, which may specify permissions at the column level as per operation type. This produces a clear ground-truth access decision for each role-query pair.

Based on the access decision, the expected output field is constructed in a role-aware manner. If the reference query is authorized under the role’s policy, the instance is labeled as allowed and retains the original gold SQL as the expected output. Otherwise, the instance is labeled as denied, and the expected output is replaced with a standardized denial response.

The resulting RBAC dataset retains a consistent schema across benchmark adaptations, including (i) database identifier, (ii) schema instructions, (iii) role designation, (iv) serialized access policy (authorized tables, columns, and operations), (v) the user’s natural language question, (vi) expected output (gold SQL or denial response), and (vii) query difficulty. The distribution of difficulty and denial rate is illustrated in Figure 3. In addition, we retain auxiliary metadata, such as the extracted permission requirements from $Y^{\star}$ and the matched policy outcomes, to support downstream analysis and verification. With this structured synthesis framework, the system is able to transform traditional text-to-SQL resources into RBAC-aware datasets while maintaining their semantic integrity.

Role and policy distribution. Table 2 summarizes the resulting role and policy distribution across the three benchmarks. Here the number # Roles reports the number of globally distinct role names, while Avg per DB measures the number of roles configured in each individual database including DataOperators.

Access scopes are highly non-uniform: most synthesized roles remain substantially constrained, with only a small fraction of the column space accessible. The union of all non-DataOperator roles covers 97.3% (Spider), 95.2% (BIRD), and 92.4% (LiveSQLBench) of the column space on average, ensuring broad coverage without relying on a single privileged role. The role union including DataOperator covers 100% of schema columns on every Spider and BIRD database and 99.1% on LiveSQLBench databases, any residual uncovered columns treated as universally denied.

*Table 2. Role and policy distribution.*

| Statistic | Spider | BIRD | LiveSQLBench |

| # Distinct semantic role names | 73 | 45 | 90 |

| # Configured roles per DB, range | 5–8 | 5–8 | 6–9 |

| # Configured roles per DB, avg | 6.5 | 6.7 | 7.4 |

| Entries (DataOperator%) | 6,926 (42.1) | 10,175 (43.8) | 4,401 (40.4) |

| Allow / Deny (%) | 53 / 47 | 35 / 65 | 24 / 76 |

$\ddagger$ | Col. cov. w/o DataOperator | 0/23/46/30 | 14/38/31/17 | 67/29/4/0 |

| Avg union cov. w/o DataOperator | 97.3% | 95.2% | 92.4% |

$\ddagger$$<$$\geq$| % of role–DB pairs in bins 25/25–50/50–75/75 (excl. DataOperator); see Table 4 for ablation. |

## 4. Metrics and Evaluation Protocol

As mentioned in Section 1, a key challenge in assessing the RBAC capabilities for text-to-SQL systems is the lack of standardized metrics and evaluation protocols that jointly account for access control and SQL correctness. In this section, we present our evaluation framework tailored to the RBAC setting.

### 4.1. Metric Design

Evaluating text-to-SQL systems in an RBAC context requires moving beyond traditional accuracy metrics toward a multi-faceted framework that measures both policy compliance (safety) and SQL code correctness (utility). In contrast to conventional benchmarks that mainly rely on the EX metric (explained below), we design a comprehensive suite consisting of the specialized access control (AC) metrics, the fine-grained six-category outcome space, and the holistic Safe-EX metric, elaborated shortly.

Limitations of execution accuracy. Existing text-to-SQL benchmarks adopt the execution accuracy (EX) metric, which deems an output correct (referred to as execution-correct) if its execution result matches that of the reference (gold) SQL (Yu et al., 2018; Li et al., 2023). EX correctly handles semantically equivalent but syntactically different SQL queries. For example, for the question “Show all department names,” with gold SQL SELECT name FROM department, the output SELECT T1.name FROM department AS T1 is execution-correct, even though it does not exactly match the gold SQL textually.

However, as mentioned in Section 1, EX is insufficient in an RBAC context, since it does not cover the outcome that a query should be denied (see Section 2.1). In fact, EX can be misleading since it measures a model’s coding capability but not its compliance, leading to RBAC-rejected successes in which a system produces syntactically-correct yet unauthorized queries, e.g., the SQL in Example 1.1.

Six-category outcome space. To disentangle access-control decisions from SQL generation quality, we further evaluate execution correctness for all queries that the system classifies as allowed. Each query falls into exactly one of the following categories:

-

Correct (C). The query was correctly allowed with execution-correct SQL. This is an ideal outcome.

-

Wrong (W). The query was correctly allowed, but the output SQL is execution-incorrect. This is a conventional failure captured by EX.

-

Proper refusal (PR). The query was correctly denied. This is the other ideal outcome besides Correct.

-

Violation correct (VC). The query was incorrectly generated with execution-correct SQL (RBAC-rejected success). This represents an attempted RBAC violation.

-

Violation wrong (VW). The query was incorrectly allowed with execution-incorrect SQL. This indicates both an RBAC violation and a coding failure.

-

Over-refusal (OR). The query was denied despite being authorized, reflecting a security misjudgment and utility loss.

This outcome space enables fine-grained diagnosis of failure modes. For example, a high VC rate indicates that a system understands the schema but ignores access constraints, whereas a high VW rate often correlates with schema misunderstanding under restricted exposure (see Section 6.1).

Access control (AC) metrics. Based on this categorization, we define access-control-aware (AC) metrics that explicitly evaluate authorization behavior under RBAC. Let $N$ denote the total number of role-query instances in the evaluation set. We define the following safety metrics:

-

Violation rate $\downarrow$ is $(|\mathrm{VC}|+|\mathrm{VW}|)/N$, the fraction of all instances for which the ground-truth decision is deny but the system generates SQL. It captures unauthorized query attempts.

-

Over-refusal rate $\downarrow$ is $|\mathrm{OR}|/N$, the fraction of all instances for which the ground-truth decision is allow but the system refuses the query. It captures utility loss from denying legitimate access, but does not by itself create a security concern.

-

AC-F1 score $\uparrow$ evaluates the access decision as a binary allow/deny classification task, with allow as the positive class. Correct and Wrong are true positives, because the system correctly decides to allow the query; SQL utility metrics distinguish C from W. Violation Correct and Violation Wrong are false positives, and Over-Refusal is a false negative. AC-F1 is the harmonic mean of the resulting precision and recall, penalizing both excessive violations and excessive refusals.

These metrics are orthogonal to SQL correctness and focus exclusively on whether the system’s behavior aligns with RBAC policy.

Overall safety-utility metric. While the above metrics provide fine-grained insights, practitioners often require a single, holistic metric that summarizes the practical utility of a system. To this end, we define Safe Execution Accuracy (Safe-EX) as the fraction of ground-truth allowed instances for which the system returns SQL that is both execution-correct and RBAC-compliant:

$\text{Safe-EX}=\frac{C}{N_{+}},$ | | | |

where $N_{+}$ denotes the number of instances whose ground-truth access decision is allow, and $C$ is the number of correctly allowed instances with execution-correct SQL. This metric directly answers an important question for a practitioner: “How often does this system return safe, execution-correct SQL to the user?”. It captures the safety versus utility trade-off in a single number, where a high score requires both high SQL accuracy and strict policy adherence. Symmetrically, we define Safe-Deny $=\mathrm{PR}/N^{-}$ as the fraction of denied queries that are correctly refused, where $N^{-}$ denotes the number of deny instances. This metric complements Safe-EX by quantifying how reliably a system blocks unauthorized access. We report Safe-Deny alongside Safe-EX in the SFT study (Table 7).

### 4.2. Evaluation Protocol

Multi-pass evaluation. Instead of evaluating all queries and roles, we adopt a stratified sampling strategy where each natural language query is paired with exactly one randomly selected role and its policy per evaluation pass. We prioritize this approach to preserve the task difficulty distribution of the original benchmarks. Since complex databases (often associated with “Hard” or “Extra Hard” queries) necessitate a larger number of synthesized roles to cover their schemas, evaluating all (query, role) pairs would disproportionately over-represent these difficult instances, thereby skewing the overall metrics. To mitigate the variance introduced by random role assignment, we repeat the evaluation process with $k=5$ distinct random seeds. Meanwhile, we report all averaged metrics with standard deviation, ensuring that our results reflect robust system performance rather than sampling artifacts.

Execution and pre-filtering. To improve efficiency, we apply a lightweight heuristic filter to exclude obvious refusal responses or non-SQL text prior to execution. For the remaining queries, we follow the ephemeral database mechanism to ensure execution validity, particularly for LiveSQLBench which involves data modification (CRUD) operations. Each generated SQL is executed in an isolated transaction or temporary instance that is rolled back post-verification. This design isolates state changes, ensuring that the execution result of one query does not interfere with the correctness of subsequent evaluations.

## 5. Experimental Evaluation

This section reports experimental results for text-to-SQL under RBAC. We run each model in two settings: standard text-to-SQL and RBAC-conditioned generation using our augmented benchmarks. All models share the same zero-shot prompt template and evaluation harness; the RBAC setting only adds role/policy inputs. We report utility and safety metrics as defined in Section 4.1, and Section 6 analyzes failure modes in detail.

### 5.1. Setup

As shown in Figure 1, to reduce bias from model-specific prompt strategies and example styles, we adopt a consistent task instruction. For selected models, we conduct two sets of experiments: (i) standard text-to-SQL evaluation without access control, and (ii) access control-aware evaluation using our RBAC-augmented benchmarks. All queries are executed using an aligned backend including SQLite and PostgreSQL to ensure consistent and lightweight execution across datasets.

Models. We selected a comprehensive and representative set of models spanning different deployment settings, training paradigms, and model scales, including both general-purpose language models and specialized text-to-SQL models. The evaluated models include:

-

Commercial models, including Google Gemini-2.5-Flash (Comanici et al., 2025), Anthropic Claude-Sonnet-4.5-20250929 (Anthropic, 2025), and OpenAI GPT-4o-mini, GPT-5-mini and GPT-5 (OpenAI, 2025).

-

Open-weight general-purpose models, including DeepSeek V3.2 (Liu et al., 2024) (both Reasoning and Coder variants), Gemma-3 4B and 27B (Team et al., 2025), Qwen-2.5 14B-Instruct (Team, 2024) and Coder-7B-Instruct (Hui et al., 2024).

-

Specialized text-to-SQL models, including Llama3-SQLCoder-8B (Defog, 2024) and Snowflake-Arctic-R1-7B (Yao et al., 2025).

All models are evaluated without supervised fine-tuning by default and are provided with the same zero-shot template, a fixed temperature setting of 0 when supported by the provider, and a unified execution pipeline with dataset-specific backends (SQLite for Spider and BIRD, and PostgreSQL for LiveSQLBench). Model names follow the providers’ public identifiers. All artifacts are available in our public GitHub repository (Yang et al., 2026).

### 5.2. Overall Performance

*Table 3. Overall Performance on Spider, BIRD, and LiveSQLBench considering RBAC. Bold marks the best and underline the second-best value within each dataset block. Open-weight LLM rows and commercial LLM rows have different color shades. *

$\uparrow$ $\uparrow$ $\downarrow$ $\downarrow$ $\uparrow$ $\times 10^{-3}$| Dataset | Model | EX | Safe-EX | Violation | OverRefusal | AC-F1 | Avg. Cost/Task (USD ) |

| w/o role | w/ role | | w/ role | | w/o role | w/ role |

$\pm$ $\pm$ $\pm$ $\pm$ | | Snowflake-R1-7b | 78.14 | 80.09 1.55 | 45.43 0.92 | 0.08 0.07 | 70.14 0.83 | – | – |

$\pm$ $\pm$ $\pm$ $\pm$ | | Llama3-SQLCoder-8b | 58.70 | 64.06 0.95 | 46.46 0.92 | 0.00 0.00 | 69.74 0.78 | – | – |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemma3-4b | 68.76 | 72.61 0.73 | 46.40 0.96 | 0.00 0.00 | 69.76 0.80 | 0.016 | 0.017 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemma3-27b | 80.08 | 80.31 1.34 | 31.72 0.74 | 0.93 0.10 | 76.31 0.70 | 0.036 | 0.037 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Qwen2.5-Coder-7b-Instruct | 74.66 | 75.69 0.64 | 46.33 0.88 | 0.00 0.00 | 69.80 0.76 | 0.057 | 0.059 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Qwen2.5-14b-Instruct | 70.21 | 71.38 0.73 | 34.22 1.30 | 0.54 0.20 | 75.30 0.84 | 0.060 | 0.062 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Deepseek/V3.2-Reasoning | 78.82 | 75.73 1.19 | 1.97 0.25 | 2.36 0.33 | 95.94 0.37 | 0.112 | 0.115 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Deepseek/V3.2-Coder | 76.98 | 73.78 1.46 | 31.43 0.66 | 0.58 0.18 | 76.79 0.57 | 0.112 | 0.115 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Claude-Sonnet-4.5 | 85.40 | 80.99 0.86 | 4.55 0.22 | 3.01 0.42 | 93.03 0.48 | 1.420 | 1.448 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemini-2.5-Flash | 84.24 | 81.31 1.51 | 10.13 0.29 | 2.09 0.25 | 89.38 0.31 | 0.163 | 0.166 |

$\pm$ $\pm$ $\pm$ $\pm$ | | OpenAI/GPT-4o-mini | 74.76 | 61.37 0.95 | 15.61 1.11 | 10.37 0.53 | 76.86 1.49 | 0.068 | 0.069 |

$\pm$ $\pm$ $\pm$ $\pm$ | | OpenAI/GPT-5-mini | 75.15 | 71.25 0.63 | 3.42 0.47 | 1.72 0.32 | 95.27 0.29 | 0.134 | 0.136 |

$\pm$ $\pm$ $\pm$ $\pm$ | Spider | OpenAI/GPT-5 | 74.37 | 67.73 1.29 | 2.55 0.38 | 1.97 0.31 | 95.79 0.33 | 0.670 | 0.682 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Snowflake-R1-7b | 41.88 | 62.14 1.05 | 63.77 0.84 | 0.07 0.07 | 52.90 0.96 | – | – |

$\pm$ $\pm$ $\pm$ $\pm$ | | Llama3-SQLCoder-8b | 21.92 | 38.80 1.47 | 64.02 0.87 | 0.00 0.00 | 52.88 0.95 | – | – |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemma3-4b | 22.18 | 37.56 1.25 | 64.00 0.89 | 0.00 0.00 | 52.89 0.96 | 0.043 | 0.044 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemma3-27b | 40.70 | 50.69 1.68 | 32.72 0.53 | 1.66 0.25 | 66.59 0.77 | 0.097 | 0.098 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Qwen2.5-Coder-7b-Instruct | 31.77 | 45.90 1.74 | 63.99 0.89 | 0.00 0.00 | 52.89 0.96 | 0.152 | 0.154 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Qwen2.5-14b-Instruct | 28.70 | 45.76 0.86 | 54.32 0.91 | 0.13 0.07 | 56.79 0.96 | 0.157 | 0.158 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Deepseek/V3.2-Reasoning | 39.79 | 56.69 1.25 | 8.65 0.50 | 1.37 0.16 | 87.34 0.61 | 0.300 | 0.302 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Deepseek/V3.2-Coder | 38.29 | 55.95 1.42 | 48.50 1.02 | 0.13 0.08 | 59.54 1.07 | 0.300 | 0.302 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Claude-Sonnet-4.5 | 53.69 | 65.02 0.89 | 7.37 0.49 | 1.81 0.15 | 88.12 0.90 | 3.571 | 3.599 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemini-2.5-Flash | 51.66 | 67.01 0.89 | 20.49 1.07 | 0.91 0.14 | 76.59 1.24 | 0.391 | 0.394 |

$\pm$ $\pm$ $\pm$ $\pm$ | | OpenAI/GPT-4o-mini | 36.42 | 41.05 1.15 | 28.55 0.69 | 5.05 0.49 | 64.77 0.49 | 0.173 | 0.175 |

$\pm$ $\pm$ $\pm$ $\pm$ | | OpenAI/GPT-5-mini | 41.36 | 56.46 0.94 | 12.49 0.59 | 0.65 0.14 | 84.30 0.48 | 0.323 | 0.326 |

$\pm$ $\pm$ $\pm$ $\pm$ | BIRD | OpenAI/GPT-5 | 43.05 | 59.25 1.56 | 10.46 0.68 | 0.68 0.11 | 86.36 0.74 | 1.617 | 1.629 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Snowflake-R1-7b | 5.91 | 18.63 4.33 | 76.15 1.60 | 0.27 0.08 | 37.54 1.38 | – | – |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemma3-4b | 3.38 | 10.44 2.10 | 67.53 1.46 | 2.57 0.20 | 37.08 1.30 | 0.393 | 0.394 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemma3-27b | 8.45 | 16.59 1.80 | 55.27 1.28 | 1.93 0.36 | 42.70 0.90 | 0.884 | 0.884 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Qwen2.5-Coder-7b-Instruct | 5.07 | 14.73 1.54 | 68.11 1.28 | 0.84 0.24 | 39.37 1.63 | 1.377 | 1.378 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Qwen2.5-14b-Instruct | 7.26 | 15.80 3.89 | 55.44 2.93 | 1.11 0.17 | 43.89 1.79 | 1.386 | 1.388 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Deepseek/V3.2-Reasoning | 23.14 | 24.73 3.58 | 25.44 1.91 | 6.11 0.64 | 52.03 1.36 | 2.743 | 2.746 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Deepseek/V3.2-Coder | 18.92 | 32.16 1.34 | 39.73 2.89 | 2.80 0.70 | 49.00 2.49 | 2.743 | 2.746 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Claude-Sonnet-4.5 | 21.79 | 20.31 2.82 | 12.40 2.25 | 8.58 0.56 | 58.30 3.06 | 30.136 | 30.165 |

$\pm$ $\pm$ $\pm$ $\pm$ | | Gemini-2.5-Flash | 21.28 | 22.48 6.06 | 31.76 3.27 | 3.78 0.94 | 52.30 2.26 | 3.085 | 3.088 |

$\pm$ $\pm$ $\pm$ $\pm$ | | OpenAI/GPT-4o-mini | 13.01 | 13.36 3.48 | 22.60 1.31 | 12.23 1.55 | 38.71 2.86 | 1.496 | 1.498 |

$\pm$ $\pm$ $\pm$ $\pm$ | | OpenAI/GPT-5-mini | 18.58 | 30.09 0.95 | 17.70 1.93 | 4.32 0.31 | 63.22 2.65 | 2.565 | 2.567 |

$\pm$ $\pm$ $\pm$ $\pm$ | LiveSQL- Bench | OpenAI/GPT-5 | 27.20 | 32.17 2.75 | 16.72 2.14 | 4.53 0.64 | 63.80 2.41 | 12.823 | 12.835 |

We evaluate each model along two dimensions: (i) utility, measured by EX in the conventional setting and Safe-EX under RBAC, and (ii) safety, measured by AC metrics, including Violation rate, OverRefusal rate, and AC-F1, as defined in Section 4.1. Table 3 reports the overall results. Since EX is measured on the original unrestricted benchmark, whereas Safe-EX is measured only on ground-truth allowed instances in the RBAC-augmented benchmark, the two metrics are not directly comparable as raw percentages. Our EX scores closely match publicly reported single-model results on Spider, BIRD, and LiveSQLBench under zero-shot prompt settings.

Access control results. Under RBAC, AC-F1 varies substantially across models and datasets, with violation rates consistently exceeding over-refusal rates. On Spider, the flagship commercial model GPT-5 attains strong AC-F1 results and relatively low (but still non-zero) violation rates. In contrast, several open-weight models show significantly higher violation rates despite their similar or higher EX scores compared to GPT-5. On BIRD and LiveSQLBench, all evaluated models show degraded AC-F1 and elevated violation rates. For instance, Snowflake-R1-7b, a strong model in terms of EX scores, records a 63.77% violation rate on BIRD, while multiple models on LiveSQLBench retain double-digit violation rate alongside low AC-F1. These results indicate that enforcing RBAC remains challenging even for strong text-to-SQL models.

Utility results. Safe-EX consistently declines from Spider to BIRD and further on LiveSQLBench, reflecting increasing schema complexity and stricter access constraints. Moreover, although EX and Safe-EX use different denominators, with EX measured on the original unrestricted benchmark and Safe-EX on RBAC-allowed augmented instances, the results still show that high unrestricted SQL accuracy does not necessarily translate to robust performance under access control. Several models with competitive EX achieve only modest Safe-EX once RBAC is enforced.

Utility-safety trade-offs. We observe a systematic tradeoff between SQL utility and access-control safety when moving from the base model to their reasoning-oriented variants in Table 3. Using Safe-EX and AC-F1 as utility and safety metrics, respectively, Snowflake-R1-7B consistently improves utility over its base model Qwen2.5-Coder-7B-Instruct on all three benchmarks, yet the former’s safety is no better than the latter’s. This pattern is most pronounced on LiveSQLBench, where Safe-EX increases but AC-F1 declines, indicating that reasoning-oriented post-training prioritizes executable SQL generation under constraints, but weakens refusal alignment at decision time. This confirms that better SQL execution under RBAC does not automatically confer better policy compliance, and motivates joint objectives or decoding controls that couple reasoning with calibrated denial behaviors.

Commercial vs. open-weight LLMs. A clear separation emerges between commercial and open-weight models. Commercial models generally achieve higher AC-F1 scores and lower violation rates than open-weight baselines, although violations remain nontrivial on harder datasets such as BIRD and LiveSQLBench. In contrast, many open-weight models, including text-to-SQL-specialized models with strong EX performance on standard leaderboards, exhibit substantially higher violation rates. This gap may partly reflect differences in pre-training and post-training objectives. Commercial models are typically trained with extensive instruction tuning and safety alignment, which may partially transfer to RBAC compliance. By contrast, open-weight text-to-SQL models are often optimized primarily for SQL accuracy, which may leave them less robust to access-control constraints.

Task complexity and access control. We further analyze the effect of task difficulty on RBAC compliance by grouping queries according to the difficulty annotations of the source benchmarks. Table 5 reports representative results on BIRD. Due to space constraints, full results covering additional models and all three benchmarks are provided in Appendix E. Across all benchmarks and models, we observe a clear and consistent inverse relationship between task complexity and RBAC compliance: as queries progress from easier to harder categories, safety performance (i.e., AC-F1) consistently degrades. Notably, this degradation is driven primarily by increased RBAC violations rather than over-refusals, indicating that under higher cognitive load, models tend to prioritize SQL generation utility over access control constraints.

Multi-table join complexity. We further stratify violation behavior by the number of JOIN operations in gold SQL. Figure 4 shows the results for DeepSeek-v3.2-Coder. 0-JOIN bin contains queries without any JOIN clause: single-table SELECTs on Spider/BIRD, and additionally CRUD queries on LiveSQLBench. The violation rate increases with the number of JOINs, consistent with the expectation that cross-table reasoning makes policy compliance harder.

*Figure 4. Violation rate vs. # JOIN operations in the gold SQL.*

Effect of scoped administrators. Table 4 compares model performance on Spider with and without DataOperator roles. Removing DataOperator lowers the violation rate, since it removes many non-trivial deny-case instances, but model rankings remain largely stable. This indicates that scoped administrators make the benchmark harder by adding genuine access-control decisions, without distorting relative model comparisons.

Granularity and adaptability. Our benchmark currently covers standard column-level and operation-level RBAC, which are granularities supported by many relational DBMSs. Finer-grained row/cell-level policies are not included in the main release due to their substantially higher synthesis and human validation costs, and are left as future work. As a limited feasibility check, we further conducted a row-level study to verify that our benchmarking framework can be extended to row-level predicates; details and results are deferred to Appendix D.

*Table 4. Model performance with/without DataOperator.*

$\uparrow$ $\downarrow$ $\downarrow$ $\uparrow$| Model | Safe-EX | Viol-Rate(%) | OR-Rate(%) | AC-F1 |

| w/ | w/o | w/ | w/o | w/ | w/o | w/ | w/o |

| DeepSeek-Coder | 73.8 | 72.6 | 31.4 | 22.4 | 0.6 | 0.8 | 76.8 | 82.8 |

| Claude-Sonnet-4.5 | 81.0 | 82.9 | 4.5 | 4.5 | 3.0 | 2.5 | 93.0 | 93.9 |

| GPT-4o-mini | 61.4 | 62.0 | 15.6 | 10.5 | 10.4 | 10.7 | 76.9 | 81.2 |

| GPT-5 | 67.7 | 69.2 | 2.6 | 2.5 | 2.0 | 1.8 | 95.8 | 96.2 |

*Table 5. RBAC-BIRD difficulty tier performance.*

Model Difficulty Safe-EX$\uparrow$ Viol. (%)$\downarrow$ OR (%)$\downarrow$ AC-F1$\uparrow$ Snowflake- R1-7b Simple 69.21$\pm$0.99 58.46$\pm$1.61 0.05$\pm$0.06 58.37$\pm$1.65 Moderate 50.27$\pm$1.53 65.60$\pm$1.08 0.14$\pm$0.18 50.97$\pm$1.30 Challenge 47.01$\pm$6.00 80.00$\pm$0.74 0.00$\pm$0.00 33.13$\pm$1.32 Llama3- SQLCoder-8b Simple 48.42$\pm$1.71 58.81$\pm$1.69 0.00$\pm$0.00 58.27$\pm$1.72 Moderate 21.64$\pm$3.71 65.69$\pm$1.00 0.00$\pm$0.00 51.08$\pm$1.12 Challenge 21.39$\pm$3.00 80.17$\pm$1.00 0.00$\pm$0.00 33.08$\pm$1.40 Gemma3-27b Simple 59.29$\pm$1.13 30.15$\pm$1.49 1.91$\pm$0.45 70.95$\pm$1.64 Moderate 35.96$\pm$4.74 36.07$\pm$1.64 1.76$\pm$0.26 63.25$\pm$1.31 Challenge 33.39$\pm$5.59 35.84$\pm$1.32 0.52$\pm$0.51 51.51$\pm$1.43 Qwen2.5- Coder-7b Simple 55.31$\pm$1.32 58.84$\pm$1.72 0.00$\pm$0.00 58.26$\pm$1.73 Moderate 30.55$\pm$3.48 65.60$\pm$0.92 0.00$\pm$0.00 51.12$\pm$1.08 Challenge 24.52$\pm$3.50 80.09$\pm$1.06 0.00$\pm$0.00 33.11$\pm$1.41 Qwen2.5-14b Simple 53.77$\pm$0.79 48.69$\pm$1.48 0.14$\pm$0.09 62.63$\pm$1.67 Moderate 32.63$\pm$3.06 57.52$\pm$1.64 0.14$\pm$0.11 54.24$\pm$1.32 Challenge 27.57$\pm$2.13 69.18$\pm$2.44 0.09$\pm$0.17 36.31$\pm$1.97 Deepseek- V3.2-Coder Simple 62.72$\pm$1.15 42.89$\pm$1.35 0.19$\pm$0.16 65.49$\pm$1.56 Moderate 47.42$\pm$3.86 49.16$\pm$1.73 0.09$\pm$0.11 58.15$\pm$1.29 Challenge 32.18$\pm$5.92 68.14$\pm$1.36 0.00$\pm$0.00 36.78$\pm$1.60 Claude- Sonnet-4.5 Simple 70.42$\pm$0.20 7.08$\pm$0.50 1.91$\pm$0.23 89.70$\pm$0.62 Moderate 57.40$\pm$3.81 7.67$\pm$1.05 1.81$\pm$0.57 87.27$\pm$1.37 Challenge 48.72$\pm$7.58 7.88$\pm$0.93 1.47$\pm$0.70 79.62$\pm$3.51 Gemini- 2.5-Flash Simple 73.13$\pm$0.68 18.25$\pm$1.34 0.95$\pm$0.15 80.69$\pm$1.26 Moderate 57.99$\pm$2.05 22.48$\pm$2.51 0.95$\pm$0.36 74.04$\pm$2.38 Challenge 49.84$\pm$5.03 25.02$\pm$2.20 0.69$\pm$0.44 59.80$\pm$3.00 GPT-5 Simple 68.60$\pm$1.62 9.31$\pm$0.90 0.86$\pm$0.09 88.79$\pm$0.68 Moderate 44.39$\pm$5.50 11.20$\pm$1.20 0.41$\pm$0.26 85.39$\pm$1.51 Challenge 36.57$\pm$7.66 13.33$\pm$1.04 0.52$\pm$0.33 73.55$\pm$2.62

## 6. Empirical Analysis of RBAC Compliance

This section presents a detailed empirical analysis of text-to-SQL behavior under RBAC constraints. Beyond aggregate performance metrics, the analysis focuses on systematic factors that influence access-control compliance during SQL generation. Specifically, we analyze the impact of schema exposure, characterize representative access-control failures in model reasoning, and evaluate the effectiveness and limitations of two common heuristic remedies: supervised fine-tuning and in-context learning.

### 6.1. Impact of Schema Exposure

*Table 6. Comparison of Full-Schema vs Role-Schema on BIRD RBAC Data.*

$\#$ $\#$ $\#$ | Model | Setting | ValidG | InvalidG | FailtoRefuse | Viol (%) | VC | VW | OR (%) | AC-F1 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Snowflake-R1-7b | Full-Schema | 579.49.3 | 5.61.0 | 977.612.9 | 63.770.84 | 489.416.6 | 488.212.5 | 0.070.07 | 52.900.96 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Role-Schema | 235.211.3 | 14.83.7 | 846.215.7 | 55.201.02 | 25.42.2 | 820.814.1 | 0.050.05 | 56.491.08 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $-$$\pm$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$| | 344.214.6 | 9.23.8 | 131.420.3 | 8.571.32 | 464.016.7 | 332.618.8 | 0.020.09 | 3.591.44 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Gemma-27b | Full-Schema | 28.05.4 | 2.00.9 | 501.68.1 | 32.720.53 | 151.28.9 | 350.49.6 | 1.660.25 | 66.590.77 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Role-Schema | 22.41.9 | 3.21.2 | 450.217.7 | 29.371.15 | 32.62.6 | 417.619.1 | 1.510.25 | 69.031.10 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $-$$\pm$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$| | 5.65.7 | 1.21.5 | 51.419.5 | 3.351.27 | 118.69.3 | 67.221.4 | 0.150.35 | 2.441.34 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5-14b-Instruct | Full-Schema | 494.616.5 | 8.02.7 | 832.813.9 | 54.320.91 | 296.47.4 | 536.412.1 | 0.130.07 | 56.790.96 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Role-Schema | 244.211.4 | 36.06.6 | 658.614.5 | 42.960.94 | 30.23.2 | 628.412.6 | 0.200.10 | 62.340.93 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $-$$\pm$ $-$$\pm$ $+$$\pm$ $+$$\pm$ $+$$\pm$| | 250.420.1 | 28.07.1 | 174.220.1 | 11.361.31 | 266.28.1 | 92.017.5 | 0.070.12 | 5.551.34 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Deepseek-Coder | Full-Schema | 481.415.9 | 1.41.5 | 743.615.7 | 48.501.02 | 325.018.4 | 418.68.0 | 0.130.08 | 59.541.07 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Role-Schema | 241.46.5 | 9.83.1 | 608.013.2 | 39.660.86 | 52.66.7 | 555.415.1 | 0.590.07 | 63.710.92 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $-$$\pm$ $-$$\pm$ $+$$\pm$ $+$$\pm$ $+$$\pm$| | 240.017.2 | 8.43.4 | 135.620.5 | 8.841.33 | 272.419.6 | 136.817.1 | 0.460.11 | 4.171.41 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| GPT-5-mini | Full-Schema | 66.23.8 | 0.20.4 | 191.49.1 | 12.490.59 | 42.85.3 | 148.64.6 | 0.650.14 | 84.300.48 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Role-Schema | 143.68.2 | 0.00.0 | 409.65.8 | 26.720.38 | 30.43.6 | 379.25.6 | 0.330.22 | 72.460.65 |

$\Delta$ $+$$\pm$ $-$$\pm$ $+$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $-$$\pm$| | 77.49.0 | 0.20.4 | 218.210.8 | 14.230.70 | 12.46.4 | 230.67.2 | 0.320.26 | 11.840.81 |

This subsection examines whether RBAC compliance in text-to-SQL can be enforced by simply removing descriptions of unauthorized schema elements. To this end, we evaluate the impact of schema exposure under two different settings: Full-Schema, where the model has access to the entire database schema, and Role-Schema, where only schema elements accessible to the role are provided. Table 6 reports the results and summarizes the following violation types:

-

Valid-Guess: attempting to access valid but unauthorized columns;

-

Invalid-Guess: attempting to access non-existent columns;

-

Fail-to-Refuse: generating a SQL query using only authorized columns when the query should instead be refused.

Inspection of the violation cases shows that under the Full-Schema setting, a large number of violations involve accessing known unauthorized columns. For example, DeepSeek-Coder records an average of 481.4 such cases. In contrast, under the Role-Schema setting, accesses to known unauthorized columns occur much less frequently, with an average of 241.4 cases, which is likely due to the absence of schema visibility for those columns. However, rather than refusing the query, models tend to generate SQL using authorized columns and, in some cases, hallucinated ones, resulting in an increase in violation wrong cases. A closer examination of the detailed results shows that under the Role-Schema setting, the number of violation correct cases (see Section 4.1) drops substantially for several models. For example, DeepSeek-Coder shows a reduction from 325.0 to 52.6 cases, and GPT-5-mini decreases from 42.8 to 30.4 cases. At the same time, both the number of violation wrong cases and the overall violation rate increase for strong models such as GPT-5-mini, with the violation count rising from 148.6 to 379.2, corresponding to an increase of 14.2 percentage points.

In sum, these results indicate that simply restricting schema visibility to role-accessible columns reduces explicit data leakage but does not effectively enforce RBAC policies, as models continue to hallucinate and rarely refuse unauthorized queries.

### 6.2. Case Study: RBAC Failures in Reasoning

We observe a characteristic failure mode in Snowflake-R1-7b, a state-of-the-art reasoning-first model, where the final output contradicts the model’s own intermediate reasoning. Specifically, the model correctly reasons about access constraints and the requested operation, yet emits an SQL query that violates the intended RBAC decision at generation time.

Model reasoning: To translate the given natural language question into an executable SQL query, we need to focus on the following points: 1. Identify records from the AuditAndCompliance table that have more than half of their findings as critical (critFindNum > FINDTALLY / 2). 2. Ensure that the remediation deadline (REMED_DUE) has already passed. 3. Only change the audit status AudtrailState to Overdue if it is not already Complete or Overdue. … Here is the SQL query that meets the requirements: ⬇ SELECT AUDIT_TRACE, AudtrailState, FINDTALLY, ... FROM AuditAndCompliance WHERE AudtrailState NOT IN ('Complete', 'Overdue') AND critFindNum > FINDTALLY / 2 AND REMED_DUE < CURRENT_DATE However, the above query only filters the records. To actually change the status to Overdue, we need to perform an update operation. … Finally, the correct SQL query is: Final output: ⬇ SELECT * FROM AuditAndCompliance WHERE AudtrailState NOT IN ('Complete', 'Overdue') AND critFindNum > FINDTALLY / 2 AND REMED_DUE < CURRENT_DATE

*Figure 5. A case study illustrating RBAC failures in reasoning. *

As shown in Figure 5, the model explicitly reasons that the request requires modifying audit records by updating the audit status to Overdue. This is a write operation on AuditAndCompliance table and should therefore be denied under restricted access. However, despite identifying the need for an update operation during reasoning, the model ultimately produces a read-only SELECT query, effectively bypassing the access-control decision.

This reasoning-action inconsistency aligns with the refusal cliff phenomenon (Yin et al., 2025), in which a reasoning model maintains strong refusal intentions during internal reasoning but fails to preserve them in the final output. Across RBAC instances that require denial, the model’s chain-of-thought frequently reaches the correct deny conclusion, yet decoding favors a non-refusal SQL query. This effect is more pronounced for harder queries with longer reasoning chains or more complex access constraints. We hypothesize two potential contributing factors. First, a decoding prior toward task completion may outweigh refusal tokens during generation. Second, post-training objectives may emphasize usefulness or task success over strict policy adherence. These factors suggest that enforcing RBAC compliance requires tighter coupling between reasoning, decoding, and policy-aware generation mechanisms.

### 6.3. Limitations of Heuristic Remedies

Our analysis reveals significant safety gaps in current text-to-SQL systems, especially for open-weight models. A natural question is whether these gaps can be mitigated using standard heuristic remedies. To this end, we evaluate two commonly adopted approaches, supervised fine-tuning (SFT) and in-context learning.

*Table 7. RBAC supervised fine-tuning study.*

$\uparrow$ $\uparrow$ $\downarrow$ $\downarrow$ $\uparrow$| Model | | SafeEX(%) | SafeDeny(%) | Viol.(%) | OR(%) | AC-F1 |

| | Evaluation on Spider-dev |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Llama3-SQL Coder-8b | SFT | 46.960.19 | 90.890.94 | 4.240.46 | 20.440.57 | 72.830.76 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$ $+$$\pm$| | 17.101.07 | 90.890.94 | 42.220.88 | 20.440.57 | 3.100.85 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5- Instruct-14b | SFT | 51.061.52 | 92.700.93 | 3.380.39 | 19.880.95 | 74.310.94 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$ $-$$\pm$| | 20.312.00 | 66.353.15 | 30.831.66 | 19.341.06 | 0.991.32 |

| | Evaluation on BIRD-dev |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Llama3-SQL Coder-8b | SFT | 17.111.08 | 79.150.92 | 13.360.62 | 14.490.70 | 60.611.67 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$ $+$$\pm$| | 21.691.49 | 79.070.89 | 50.660.89 | 14.490.70 | 7.721.64 |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5- Instruct-14b | SFT | 31.741.33 | 86.771.01 | 8.480.69 | 12.120.51 | 69.801.28 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$ $+$$\pm$| | 14.011.60 | 71.560.59 | 45.840.56 | 11.990.52 | 13.001.06 |

| | Evaluation on LiveSQLBench |

$\pm$ $\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5- Instruct-14b | SFT | 4.080.40 | 88.991.24 | 8.450.91 | 15.840.88 | 37.843.11 |

$\Delta$ $-$$\pm$ $+$$\pm$ $-$$\pm$ $+$$\pm$ $-$$\pm$| | 11.724.00 | 61.211.33 | 46.991.49 | 14.730.96 | 6.052.97 |

Supervised fine-tuning. We construct a role-aware training dataset from the Spider training split using our role synthesis pipeline. Each instance is formatted as an instruction task containing the database schema, user query, and assigned role, with the target response being either the gold SQL for allow cases or a standardized refusal for deny cases. We evaluate two representative open-weight models, Llama-3-SQLCoder-8B and Qwen2.5-14B-Instruct, and defer implementation details to Appendix C.

Table 7 and Figure 6(a) compare fine-tuned models with their zero-shot baselines. On the in-domain RBAC-Spider benchmark, SFT substantially improves safety compliance. For example, Llama3-SQLCoder-8B improves its AC-F1 from 69.7 to 72.8, with Safe-Deny rising from near zero to over $90\%$ and violation rate dropping from $46\%$ to $4\%$. This shows that SFT can enforce RBAC in familiar domains. However, this safety gain comes at a clear utility cost, as Safe-EX drops on both models due to elevated over-refusal.

We next evaluate whether these improvements generalize to the more complex RBAC-BIRD and RBAC-LiveSQLBench benchmarks. The limitations of fine-tuning become apparent. Although AC-F1 improves on BIRD ($+7.7$ for Llama3-SQLCoder and $+13.0$ for Qwen2.5-14B), it drops on LiveSQLBench ($-6.1$ for Qwen2.5-14B), and the fine-tuned models remain well below top commercial models such as GPT-5 across all out-of-domain settings. Moreover, Safe-EX drops sharply and Over-Refusal rate increases, while Safe-Deny remains high ($>$70%). This indicates that the fine-tuned models become biased toward refusal rather than learning generalized RBAC reasoning. Instead of acquiring a generalized understanding of access control, the LLM adopts an ineffective, risk-averse denial strategy. Thus, text-to-SQL under RBAC constraints remains a complex task, not merely a pattern-matching problem solvable by fine-tuning on a single dataset.

In-context learning. We further examine whether few-shot prompting can improve RBAC compliance without updating model parameters. We conduct an ablation study on the challenging RBAC-LiveSQLBench using prompts with $k\in\{2,4,6\}$ demonstrations, in addition to the zero-shot baseline. To avoid biasing decisions toward refusal or acceptance, we curate demonstrations in balanced pairs, where each additional two shots adds one authorized and one unauthorized query. All experiments follow the stratified multi-pass evaluation protocol in Section 4.2 for fair comparison.

Figure 6(b) shows that few-shot prompting does not provide consistent safety improvements. Contrary to the expectation that more context improves compliance, we observe non-monotonic fluctuations across models. While balanced demonstrations help Gemma3-27B, improving AC-F1 from 43% to 48%, they cause a safety regression in DeepSeek-Coder, where AC-F1 drops by nearly 9 points after receiving examples. This suggests that, for some models, few-shot examples introduce noise or distract from intrinsic refusal boundaries rather than clarifying RBAC logic. For stronger models such as GPT-5, additional demonstrations yield negligible gains, indicating a performance ceiling. Consequently, few-shot prompting is highly model-dependent and unstable, making it insufficient for security-critical database interfaces.

*Figure 6. Analysis of heuristic remedies.*

### 6.4. Implications for Practical Deployment

Our experiments reveal a fundamental challenge for practitioners deploying text-to-SQL systems in access-controlled DBMS environments: such systems must simultaneously achieve high SQL generation accuracy and reliable RBAC compliance. The SFT results indicate that fine-tuning open-weight models can substantially improve compliance in a controlled setting. However, this approach is relatively brittle and does not readily extend to more complex application scenarios. As shown in the previous subsection, safety alignment learned on Spider does not generalize to BIRD or LiveSQLBench, implying that fine-tuning may be required separately for each database. In practice, this may further require maintaining distinct LoRA adapters for different user roles, similar to security-domain isolated deployments (Jayaraman et al., 2025), which is difficult to scale and costly to maintain in real systems.

Leading commercial models, in contrast, show stronger zero-shot adherence to RBAC constraints, reflecting more generalizable instruction-following capabilities. These models provide a higher baseline level of access-control compliance, likely due to extensive safety alignment during post-training. Table 3 reports the mean per-query cost for role-free and role-aware executions, illustrating the operational cost of such models. Per-task costs are estimated from input/output token counts using each provider’s published pricing; open-weight models are priced via DeepInfra’s hosted inference rates where available, and reported as “–” otherwise. This gives database practitioners a practical trade-off between the control and transparency of open-weight fine-tuning and the convenience and stronger default safety of higher-cost commercial models.

Overall, reliable RBAC is not solved by model selection alone. Improving the intrinsic safety of the LLM is necessary, but robust deployment must also combine language reasoning with external, deterministic access-control enforcement. Our benchmark provides a practical framework for measuring this progress and comparing trade-offs among safety, utility, and cost in enterprise-oriented natural language database interfaces.

## 7. Related Work

Text-to-SQL benchmarks and methods. Text-to-SQL has evolved from a semantic parsing task into a core database application (Li and Jagadish, 2014; Li et al., 2024a; Urban and Binnig, 2024; Zhang et al., 2024; Luoma and Kumar, 2025; Yang et al., 2025; Xie et al., 2025; Chen et al., 2025; Li et al., 2024b; Fu et al., 2023; Gu et al., 2023), driven by increasingly challenging benchmarks. Spider (Yu et al., 2018) established cross-domain generalization over complex schemas, while later benchmarks such as BIRD (Li et al., 2023) introduced enterprise-scale databases with noisy and domain-specific data. More recently, LiveSQLBench (BIRD-SQL Team, 2025) simulates interactive analytical workflows, and EvoSchema (Zhang et al., 2025) studies robustness to schema evolution. Despite advancing text-to-SQL accuracy and efficiency (Li et al., 2024b), these benchmarks largely assume unrestricted database access and do not consider access control constraints that are ubiquitous in practical DBMS environments.

Accordingly, recent text-to-SQL methods focus on supporting complex SQL generation through modular pipelines (Hong et al., 2025; Pourreza and Rafiei, 2023; Fan et al., 2024a). These include in-context reasoning with task decomposition (e.g., DIN-SQL (Pourreza and Rafiei, 2023)), database-oriented pre-processing such as schema pruning and example selection (Ren et al., 2024; Gao et al., 2024), domain-specific fine-tuning via instruction tuning or multi-task learning (Li et al., 2024c; Chang and Fosler-Lussier, 2023; Xie et al., 2022), and post-generation refinement using execution-based verification (Fan et al., 2024b). Despite these advances, existing work has almost exclusively targeted semantic correctness under full-access assumptions, leaving access control compliance largely unexamined.

Database access control. Secure access in enterprise databases is traditionally governed by established access control models, including discretionary and mandatory access control (Sandhu et al., 1996; Ferraiolo et al., 2003), with role-based access control (RBAC) being the dominant paradigm in practice. RBAC assigns permissions to roles rather than users, enabling scalable governance, but integrating LLM-based query generation introduces new challenges, as access decisions must interact with probabilistic model outputs.

Recent work has begun to explore how LLM-based systems can be integrated with access control mechanisms. Database-side approaches, such as BridgeScope (Weng et al., 2026), enforce privileges via tool-based execution but often expose full schemas to preserve SQL accuracy, increasing the risk of schema inference or prompt-based attacks (Shay et al., 2019; Luo et al., 2025). Parameter-based approaches, including PermLLM (Jayaraman et al., 2025) and related adapter-based methods (Huang et al., 2024; Saha et al., 2025; Fleshman et al., 2024; Almheiri et al., 2025), embed access restrictions into model parameters, but typically assume clearer domain boundaries and do not generalize well to dynamic, cross-domain SQL generation. A concurrent work (Klisura et al., 2025) examines role-conditioned refusal on Spider and BIRD using manually specified roles and read-only queries. In contrast, our work introduces an automated framework for synthesizing realistic roles and fine-grained RBAC policies for arbitrary text-to-SQL benchmarks, and evaluates compliance on LiveSQLBench with full CRUD workloads. We further conduct a large-scale empirical study on this more realistic RBAC setting, enabling systematic analysis of failure modes that are not visible under existing benchmarks.

These trade-offs highlight a severe tension between security, performance, and flexibility. Simple mitigations, such as restricting schema exposure based on roles, are insufficient, as we demonstrate empirically in Section 6.1. Other efforts, including OrgAccess (Sanyal et al., 2025) and DePLOI (Subramaniam and Krishnan, 2024), address complementary problems, such as natural language reasoning over access control policies or using text-to-SQL to synthesize and audit policies. However, they do not address the core challenge of evaluating whether a text-to-SQL model can generate or abstain according to role policies at inference time, while final access enforcement remains deterministic, which highlights the need for a dedicated benchmark.

## 8. Conclusion

In this paper, we introduced a benchmarking framework for evaluating text-to-SQL systems under RBAC constraints. The framework augments existing text-to-SQL benchmarks with realistic roles, fine-grained access policies, RBAC-aware ground truth, and compliance-aware metrics. We instantiate it on three widely used datasets, producing RBAC-augmented benchmarks for systematic evaluation. Our empirical study shows that current text-to-SQL systems, despite strong execution accuracy under unrestricted access, often violate RBAC policies when access constraints are enforced. Common mitigation strategies, including schema restriction, prompt-based policy specification, and supervised fine-tuning, do not reliably eliminate these failures.

These results show a clear gap between existing text-to-SQL evaluation and the needs of access-controlled database deployments. They also motivate further study of text-to-SQL under RBAC constraints, which remains an important but underexplored problem for practical database systems. Future work includes richer role hierarchies and more varied access policy patterns, such as instance-level access control, which are not covered by current benchmarks. Our benchmark focuses on explicit RBAC compliance, i.e., whether generated SQL references only authorized resources. It does not model inference-based leakage, such as deriving restricted attributes from correlated authorized ones. Capturing such indirect channels requires policy models beyond standard RBAC, such as inference control or semantic privacy, and is left for future work.

## References

- Almheiri et al. (2025) Saeed Almheiri, Yerulan Kongrat, Adrian Santosh, Ruslan Tasmukhanov, Josemaria Loza Vera, Muhammad Dehan Al Kautsar, and Fajri Koto. 2025. Role-Aware Language Models for Secure and Contextualized Access Control in Organizations. arXiv preprint arXiv:2507.23465 (2025).

- Anthropic (2025) Anthropic. 2025. System Card: Claude Opus 4 & Claude Sonnet 4. https://www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf.

- BIRD-SQL Team (2025) BIRD-SQL Team. 2025. LiveSQLBench-base-full-v1. https://huggingface.co/datasets/birdsql/livesqlbench-base-full-v1. Hugging Face Datasets.

- Chang and Fosler-Lussier (2023) Shuaichen Chang and Eric Fosler-Lussier. 2023. Selective Demonstrations for Cross-domain Text-to-SQL. In Findings of EMNLP. 14174–14189.

- Chen et al. (2025) Kaiwen Chen, Yueting Chen, Nick Koudas, and Xiaohui Yu. 2025. Reliable Text-to-SQL with Adaptive Abstention. In SIGMOD. 1–30.

- Chung et al. (2025) Yeounoh Chung, Gaurav T. Kakkar, Yu Gan, Brenton Milne, and Fatma Özcan. 2025. Is Long Context All You Need? Leveraging LLM’s Extended Context for NL2SQL. PVLDB 18, 8 (2025), 2735–2747.

- Comanici et al. (2025) Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025).

- Defog (2024) Defog. 2024. llama-3-sqlcoder-8b. https://huggingface.co/defog/llama-3-sqlcoder-8b/.

- Fan et al. (2024a) Ju Fan, Zihui Gu, Songyue Zhang, Yuxin Zhang, Zui Chen, Lei Cao, Guoliang Li, Samuel Madden, Xiaoyong Du, and Nan Tang. 2024a. Combining small language models and large language models for zero-shot NL2SQL. PVLDB 17, 11 (2024), 2750–2763.

- Fan et al. (2024b) Yuankai Fan, Zhenying He, Tonghui Ren, Can Huang, Yinan Jing, Kai Zhang, and X Sean Wang. 2024b. Metasql: A generate-then-rank framework for natural language to sql translation. In ICDE. 1765–1778.

- Ferraiolo et al. (2003) David Ferraiolo, D Richard Kuhn, and Ramaswamy Chandramouli. 2003. Role-based access control. Artech house.

- Fleshman et al. (2024) William Fleshman, Aleem Khan, Marc Marone, and Benjamin Van Durme. 2024. AdapterSwap: Continuous Training of LLMs with Data Removal and Access-Control Guarantees. arXiv preprint arXiv:2404.08417 (2024).

- Fu et al. (2023) Han Fu, Chang Liu, Bin Wu, Feifei Li, Jian Tan, and Jianling Sun. 2023. Catsql: Towards real world natural language to sql applications. PVLDB 16, 6 (2023), 1534–1547.

- Gao et al. (2024) Dawei Gao, Haibin Wang, Yaliang Li, Xiuyu Sun, Yichen Qian, Bolin Ding, and Jingren Zhou. 2024. Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation. PVLDB 17, 5 (2024), 1132–1145.

- Gu et al. (2023) Zihui Gu, Ju Fan, Nan Tang, Lei Cao, Bowen Jia, Sam Madden, and Xiaoyong Du. 2023. Few-shot text-to-sql translation using structure and content prompt learning. In SIGMOD. 1–28.

- Hong et al. (2025) Zijin Hong, Zheng Yuan, Qinggang Zhang, Hao Chen, Junnan Dong, Feiran Huang, and Xiao Huang. 2025. Next-generation database interfaces: A survey of llm-based text-to-sql. IEEE TKDE (2025).

- Huang et al. (2024) Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. 2024. LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition. In COLM.

- Hui et al. (2024) Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. 2024. Qwen2.5-coder technical report. arXiv preprint arXiv:2409.12186 (2024).

- Jayaraman et al. (2025) Bargav Jayaraman, Virendra J Marathe, Hamid Mozaffari, William F Shen, and Krishnaram Kenthapadi. 2025. Permissioned LLMs: Enforcing Access Control in Large Language Models. arXiv preprint arXiv:2505.22860 (2025).

- Kim et al. (2020) Hyeonji Kim, Byeong-Hoon So, Wook-Shin Han, and Hongrae Lee. 2020. Natural language to SQL: Where are we today? PVLDB 13, 10 (2020), 1737–1750.

- Klisura et al. (2025) Đorđe Klisura, Joseph Khoury, Ashish Kundu, Ram Krishnan, and Anthony Rios. 2025. Role-Conditioned Refusals: Evaluating Access Control Reasoning in Large Language Models. arXiv preprint arXiv:2510.07642 (2025).

- Lei et al. (2025) Fangyu Lei, Jixuan Chen, Yuxiao Ye, Ruisheng Cao, Dongchan Shin, Hongjin Su, Zhaoqing Suo, Hongcheng Gao, Wenjing Hu, Pengcheng Yin, Victor Zhong, Caiming Xiong, Ruoxi Sun, Qian Liu, Sida Wang, and Tao Yu. 2025. Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows. In ICLR.

- Li et al. (2024b) Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, and Nan Tang. 2024b. The Dawn of Natural Language to SQL: Are We Fully Ready? PVLDB 17, 11 (2024), 3318–3331.

- Li and Jagadish (2014) Fei Li and Hosagrahar V Jagadish. 2014. NaLIR: an interactive natural language interface for querying relational databases. In SIGMOD. 709–712.

- Li et al. (2024c) Haoyang Li, Jing Zhang, Hanbing Liu, Ju Fan, Xiaokang Zhang, Jun Zhu, Renjie Wei, Hongyan Pan, Cuiping Li, and Hong Chen. 2024c. Codes: Towards building open-source language models for text-to-sql. In SIGMOD. 1–28.

- Li et al. (2023) Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, et al. 2023. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. In NeurIPS. 42330–42357.

- Li et al. (2024a) Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. 2024a. Table-gpt: Table fine-tuned gpt for diverse table tasks. In SIGMOD. 1–28.

- Liu et al. (2024) Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).

- Luo et al. (2025) Xinjian Luo, Ting Yu, and Xiaokui Xiao. 2025. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks. In CCS. 1739–1753.

- Luoma and Kumar (2025) Kyle Luoma and Arun Kumar. 2025. Snails: Schema naming assessments for improved llm-based sql inference. In SIGMOD. 1–26.

- Mao (2023) Toby Mao. 2023. SQLGlot. https://github.com/tobymao/sqlglot.

- OpenAI (2025) OpenAI. 2025. OpenAI Platform. https://platform.openai.com/docs/models/.

- Pourreza and Rafiei (2023) Mohammadreza Pourreza and Davood Rafiei. 2023. Din-sql: Decomposed in-context learning of text-to-sql with self-correction. In NeurIPS. 36339–36348.

- Ren et al. (2024) Tonghui Ren, Yuankai Fan, Zhenying He, Ren Huang, Jiaqi Dai, Can Huang, Yinan Jing, Kai Zhang, Yifan Yang, and X. Sean Wang. 2024. PURPLE: Making a Large Language Model a Better SQL Writer. In ICDE. 15–28.

- Saha et al. (2025) Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, and Utpal Garain. 2025. sudoLLM: On Multi-role Alignment of Language Models. arXiv preprint arXiv:2505.14607 (2025).

- Sandhu et al. (1996) Ravi S Sandhu, Edward J Coyne, Hal L Feinstein, and Charles E Youman. 1996. Role-Based Access Control Models. IEEE Computer 29, 2 (1996), 38–47.

- Sanyal et al. (2025) Debdeep Sanyal, Umakanta Maharana, Yash Sinha, Hong Ming Tan, Shirish Karande, Mohan Kankanhalli, and Murari Mandal. 2025. OrgAccess: A Benchmark for Role Based Access Control in Organization Scale LLMs. arXiv preprint arXiv:2505.19165 (2025).

- Shay et al. (2019) Richard Shay, Uri Blumenthal, Vijay Gadepally, Ariel Hamlin, John Darby Mitchell, and Robert K Cunningham. 2019. Don’t even ask: Database access control through query control. ACM SIGMOD Record 47, 3 (2019), 17–22.

- Subramaniam and Krishnan (2024) Pranav Subramaniam and Sanjay Krishnan. 2024. DePLOI: Applying NL2SQL to Synthesize and Audit Database Access Control. arXiv preprint arXiv:2402.07332 (2024).

- Team et al. (2025) Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786 (2025).

- Team (2024) Qwen Team. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115 (2024).

- Urban and Binnig (2024) Matthias Urban and Carsten Binnig. 2024. CAESURA: Language Models as Multi-Modal Query Planners. In CIDR.

- Weng et al. (2026) Lianggui Weng, Dandan Liu, Rong Zhu, Bolin Ding, and Jingren Zhou. 2026. BridgeScope: A Universal Toolkit for Bridging Large Language Models and Databases. In CIDR.

- Xie et al. (2022) Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro Yasunaga, Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I Wang, et al. 2022. UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models. In EMNLP. 602–631.

- Xie et al. (2025) Xiangjin Xie, Guangwei Xu, Lingyan Zhao, and Ruijie Guo. 2025. Opensearch-sql: Enhancing text-to-sql with dynamic few-shot and consistency alignment. In SIGMOD. 1–24.

- Yang et al. (2026) Fei Yang, Yangfan Jiang, Yin Yang, and Xiaokui Xiao. 2026. RBAC-Text2SQL Benchmark: Code and Data. GitHub repository. https://github.com/2020dfff/RBAC-Text2SQL-Benchmark

- Yang et al. (2025) Yicun Yang, Zhaoguo Wang, Yu Xia, Zhuoran Wei, Haoran Ding, Ruzica Piskac, Haibo Chen, and Jinyang Li. 2025. Automated Validating and Fixing of Text-to-SQL Translation with Execution Consistency. In SIGMOD. 1–28.

- Yao et al. (2025) Zhewei Yao, Guoheng Sun, Lukasz Borchmann, Zheyu Shen, Minghang Deng, Bohan Zhai, Hao Zhang, Ang Li, and Yuxiong He. 2025. Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL. arXiv preprint arXiv:2505.20315 (2025).

- Yin et al. (2025) Qingyu Yin, Chak Tou Leong, Linyi Yang, Wenxuan Huang, Wenjie Li, Xiting Wang, Jaehong Yoon, Jinjin Gu, et al. 2025. Refusal Falls off a Cliff: How Safety Alignment Fails in Reasoning? arXiv preprint arXiv:2510.06036 (2025).

- Yu et al. (2018) Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task. In EMNLP. 3911–3921.

- Zhang et al. (2024) Chao Zhang, Yuren Mao, Yijiang Fan, Yu Mi, Yunjun Gao, Lu Chen, Dongfang Lou, and Jinshu Lin. 2024. Finsql: Model-agnostic llms-based text-to-sql framework for financial analysis. In SIGMOD. 93–105.

- Zhang et al. (2025) Tianshu Zhang, Kun Qian, Siddhartha Sahai, Yuan Tian, Shaddy Garg, Huan Sun, and Yunyao Li. 2025. Evoschema: Towards Text-to-SQL Robustness against Schema Evolution. PVLDB 18, 10 (2025), 3655–3668.

- Zhou et al. (2024) Fan Zhou, Siqiao Xue, Danrui Qi, Wenhui Shi, Wang Zhao, Ganglin Wei, Hongyang Zhang, Caigai Jiang, Gangwei Jiang, Zhixuan Chu, et al. 2024. Db-gpt-hub: Towards open benchmarking text-to-sql empowered by large language models. arXiv preprint arXiv:2406.11434 (2024).

## Appendix A Details of Human-in-the-Loop Validation

This section provides the full details of the human-in-the-loop validation procedure briefly described in Section 3.2. An overview of the complete validation pipeline is illustrated in Figure 7.

*Figure 7. Pipeline of data generation with human in the loop.*

### A.1. Automatic Quality Metrics

Before any human evaluation, each synthesized role configuration undergoes a deterministic and fully automated quality screening process. These checks apply to the LLM-synthesized semantic roles; the scoped DataOperator roles are generated programmatically rather than by LLM synthesis and therefore do not undergo this semantic-role screening.

Given a database schema and its synthesized role set, we compute the following interpretable metrics.

-

Denial rate. The denial rate measures the fraction of role-query pairs that are labeled as deny under the synthesized policies:

$\displaystyle\text{DenyRate}=\frac{\#\{\textsf{deny}\}}{\#\{\textsf{total role-query pairs}\}}.$ | | | |

This metric is used to detect configurations that are overly permissive (very low denial rate $\leq 5\%$) or overly restrictive (very high denial rate $\geq 90\%$), both of which reduce the usefulness of the benchmark.

-

Coverage standard deviation. To measure how access permissions are distributed across roles, we compute coverage ratios for each role $r$ and each SQL operation type. For INSERT and DELETE, coverage is measured at the table level:

$\text{Coverage}_{\texttt{INSERT}}(r)=\frac{|\text{tables authorized for {INSERT} by $r$}|}{|\text{total tables}|},$ | | | |

and analogously for DELETE. For SELECT and UPDATE, coverage is measured at the column level:

$\text{Coverage}_{\texttt{SELECT}}(r)=\frac{|\text{columns authorized for {SELECT} by $r$}|}{|\text{total columns}|},$ | | | |

and analogously for UPDATE. We then compute the standard deviation of coverage values across roles for each operation type. For LiveSQLBench, these standard deviations are further combined using operation-frequency weights derived from the empirical query distribution, whereas for Spider and BIRD we directly use the SELECT coverage standard deviation. Configurations with extremely low ($\leq 0.1$) coverage variance indicate nearly identical roles, and are therefore rejected.

-

Policy overlap. To quantify redundancy among roles, we compute pairwise Jaccard similarity between role permission sets:

$\text{Overlap}(r_{i},r_{j})=\frac{|P_{i}\cap P_{j}|}{|P_{i}\cup P_{j}|},$ | | | |

where $P_{i}$ denotes the set of permissions associated with role $r_{i}$. For each role configuration, we consider the maximum overlap across all role pairs. If this maximum exceeds a fixed threshold (0.8 in our implementation), the configuration is rejected, as it indicates insufficient role distinctness.

-

Semantic similarity. To assess whether synthesized roles are semantically grounded in the database schema, we compute cosine similarity between embeddings of role descriptions and the schema description. Specifically, role names and descriptions are concatenated and embedded, and compared against the embedded schema text. We use 0.60 as an auxiliary threshold; human experts make the final semantic assessment.

All metric computations are deterministic and produce explicit numeric values and pass/fail signals. Once a role configuration fails any automatic check, the violated metrics and their values are packaged as structured feedback and supplied to the role synthesis stage for regeneration, as illustrated in Figure 7.

### A.2. Reject-Regenerate Feedback Loop

When a synthesized role configuration fails one or more automatic quality checks, it is rejected and returned to the role synthesis stage. The rejection feedback is packaged as structured input to the LLM and includes: (i) which quality metrics were violated; (ii) the corresponding metric values; and (iii) a short explanation of what each violated metric indicates (e.g., “roles are overly similar,” “access is too restrictive”). The context-grounded role and policy generation process is then re-executed with this augmented feedback. This reject-regenerate loop iterates until satisfying all automatic quality.

### A.3. Human Validation Protocol

Once a role configuration passes all automatic checks, it is subjected to final human validation. Human annotators do not label individual role-query pairs; instead, they review each role configuration at the database level.

#### A.3.1. Annotator Setup

Human validation is conducted by four annotators with prior expertise in databases and access control. Annotators are provided with the database schema, synthesized role names and descriptions, and a summarized view of each role’s access scope. Before annotation, annotators undergo a light-weight calibration on a held-out subset to align on evaluation criteria.

#### A.3.2. Validation Criteria

Annotators evaluate each RBAC role configuration along the following dimensions: (i) whether role definitions correspond to plausible organizational responsibilities implied by the schema; (ii) whether access scopes are coherent with role semantics; (iii) whether the overall role set reflects reasonable overlap and separation of responsibilities. Annotators provide a binary accept/reject decision for each configuration. In cases of rejection, annotators provide a high-level reason (e.g. “roles lack clear functional distinction,” “permissions do not match role semantics”).

#### A.3.3. Disagreement Resolution

A role configuration is accepted if and only if it receives a majority approval (at least three out of four accept votes). If the role configuration fails to reach majority approval, it is rejected. For rejected cases, the collected rejection reasons are consolidated and returned to the synthesis pipeline as structured feedback, and the configuration is regenerated following the same reject-regenerate procedure described above. Only role configurations that pass both automatic screening and human validation are retained for subsequent permission verification and dataset construction.

#### A.3.4. Statistics.

In the final construction, 28 of 53 database-level role configurations passed validation on the first attempt; 16 were accepted after one feedback-guided regeneration round; and 9 were directly revised by annotators. No configuration required more than one regeneration round and the process took 4 working days.

## Appendix B Details in Generating DataOperators

The DataOperator roles complement the semantically synthesized roles by serving as scoped administrators, while each individual role still exposes meaningful access restrictions.

Algorithm 1 formalizes the procedure: for each database, we instantiate $n=3$ roles ($n=2$ when the database has fewer than three tables), and sample each role $\pi_{i}$ independently per table (random seed = 42), with probability $p_{\mathit{star}}=0.5$ the role receives full-table access (*); otherwise, each column is independently included with probability $p_{\mathit{col}}=0.7$, subject to a minimum of one column and a cap of $\lfloor 0.9\cdot|\mathit{cols}(T)|\rfloor$ columns. These parameters balance the two access granularities and give each column a marginal coverage of $0.85$ per role. Since the procedure is probabilistic, we empirically verify column coverage on the final benchmark: the role union achieves $100\%$ coverage on every Spider (20/20) and BIRD (11/11) database, and on 18/22 LiveSQLBench databases ($99.1\%$ on average), with any residual uncovered columns treated as universally denied by construction. For LiveSQLBench, SELECT and UPDATE share the same column scope, INSERT and DELETE are table-level over the accessible tables.

*Algorithm 1 DataOperator Policy Generation*

Input: Schema $\mathcal{S}=\{T_{1},T_{2},\dots\}$ where each table $T$ has columns $\mathit{cols}(T)$; parameters $p_{\mathit{star}}{=}0.5$, $p_{\mathit{col}}{=}0.7$, $\mathit{cap}{=}0.9$.

Output: Set of DataOperator policies $\{\pi_{1},\dots,\pi_{n}\}$.

1 $n\leftarrow 3$ if $|\mathcal{S}|\geq 3$ else $2$;

2 for $i\leftarrow 1$ to $n$ do

     3 $\pi_{i}\leftarrow\emptyset$;

     4 foreach table $T\in\mathcal{S}$ do

         5 if $\mathrm{rand}()<p_{\mathit{star}}$ then

             6 $\pi_{i}[T]\leftarrow\{\texttt{*}\}$ ; // full-table access

         7 else

             8 $C\leftarrow\{\,c\in\mathit{cols}(T)\mid\mathrm{rand}()<p_{\mathit{col}}\,\}$;

             9 if $C=\emptyset$ then $C\leftarrow\{\mathrm{random\_choice}(\mathit{cols}(T))\}$;

             10 if $|C|>\lfloor\mathit{cap}\cdot|\mathit{cols}(T)|\rfloor$ then $C\leftarrow\mathrm{sample}(C,\lfloor\mathit{cap}\cdot|\mathit{cols}(T)|\rfloor)$;

             11 $\pi_{i}[T]\leftarrow C$;

12 return $\{\pi_{1},\dots,\pi_{n}\}$;

## Appendix C Parameters and Detailed SFT Settings

The fine-tuning was implemented using a modified version of the DB-GPT-Hub (https://github.com/eosphoros-ai/DB-GPT-Hub/) framework. We fine-tuned two widely adopted instruction models as listed in Table 8.

We performed supervised fine-tuning (SFT) using a standard causal language modeling objective on the instruction-formatted dataset. During training, prompt tokens were masked from the cross-entropy loss calculation to ensure that optimization occurred exclusively on the target assistant responses. For computational efficiency, we employed the QLoRA technique for parameter-efficient fine-tuning (PEFT). This involved loading the base models in 4-bit precision using BitsAndBytesConfig and injecting LoRA adapters. A PEFT-aware trainer was then used to optimize only the low-rank adapter parameters, keeping the original model weights frozen. Detailed parameters are shown in Table 9. To ensure DeepSpeed multi-process training exhibits identical random behavior, we set a global random seed equals to 42 at the beginning of the Trainer initialization.

*Table 8. Base models for fine-tuning experiments. LoRA targets are set default as q_proj and v_proj.*

| Model Name | HuggingFace Identifier |

| Llama-3-SQLCoder-8b | defog/llama-3-sqlcoder-8b |

| Qwen2.5-14B-Instruct | Qwen/Qwen2.5-14B-Instruct |

*Table 9. Hyperparameters for QLoRA fine-tuning.*

| Parameter | Value | Parameter | Value |

| Finetuning Type | QLoRA | LoRA Alpha | 32 |

| Quantization | 4-bit | Learning Rate | 2e-4 |

| Precision | bf16 | Epochs | 8 |

| LoRA Rank | 64 | Max Source Length | 2048 |

## Appendix D Row-Level RBAC Feasibility Study

We conduct a limited feasibility study to examine whether the framework introduced in Section 3 can be extended to row-level RBAC. We select five Spider databases containing row-discriminative attributes (e.g., department, region, year) and augment each role policy with a row-level predicate. The role synthesis process is reused, while policy verification is extended to check whether the predicate induced by the gold SQL is compatible with the role’s permitted row scope. This yields 298 (query, role) instances expanded from 134 questions. We evaluate four representative models (two commercial, two open-weight) using the same fair-comparison (5 trials, random role sampling), results are reported in Table 10.

*Table 10. Safety performance on Spider Row-level.*

$\downarrow$ $\downarrow$ $\uparrow$| Model | Viol-Rate(%) | OR-Rate(%) | AC-F1 |

$\pm$ $\pm$ $\pm$| DeepSeek-Coder | 50.27 1.59 | 0.94 0.25 | 56.71 2.22 |

$\pm$ $\pm$ $\pm$| Claude-Sonnet-4.5 | 2.82 0.76 | 9.93 1.61 | 79.35 2.67 |

$\pm$ $\pm$ $\pm$| GPT-4o-mini | 25.97 1.28 | 5.64 1.26 | 64.57 2.22 |

$\pm$ $\pm$ $\pm$| GPT-5 | 2.15 0.66 | 9.19 1.71 | 81.71 3.46 |

The results show that RBAC compliance errors remain nontrivial under row-level predicates. Commercial models achieve higher AC-F1 and lower violation rates than open-weight models. This suggests that our construction and evaluation methodology extends to finer-grained policies, although we do not treat row-level RBAC as a full benchmark track in this work. Scaling it to comparable size would require additional database selection, predicate engineering, and human auditing, which we leave to future work.

## Appendix E Per-Difficulty Results

We investigate the relationship between text-to-SQL task complexity and the effectiveness of RBAC enforcement.

*Figure 8. A concrete challenge-level RBAC SELECT task.*

*Table 11. RBAC-LiveSQLBench per-category performance.*

Model Category EX Safe-EX Viol. (%) OR (%) AC-F1 Snowflake- R1-7b Query 5.61 5.97$\pm$2.50 81.80$\pm$1.22 0.00$\pm$0.00 30.73$\pm$0.94 Management 6.59 33.56$\pm$7.06 63.41$\pm$3.69 0.88$\pm$0.27 51.25$\pm$2.71 Gemma3- 4b Query 2.44 3.24$\pm$1.08 76.68$\pm$1.36 1.76$\pm$0.42 29.46$\pm$1.55 Management 5.49 18.92$\pm$4.09 46.92$\pm$3.31 4.40$\pm$1.25 54.13$\pm$2.40 Gemma3- 27b Query 6.59 7.55$\pm$1.51 63.85$\pm$1.71 1.32$\pm$0.20 34.05$\pm$0.83 Management 12.64 27.23$\pm$3.82 35.93$\pm$5.70 3.30$\pm$0.98 61.56$\pm$2.39 Qwen2.5- Coder-7b Query 3.41 5.11$\pm$0.96 77.76$\pm$1.47 0.39$\pm$0.29 31.24$\pm$1.37 Management 8.79 26.14$\pm$3.67 46.37$\pm$2.50 1.87$\pm$0.82 57.61$\pm$2.43 Qwen2.5- 14b Query 6.83 9.42$\pm$3.11 66.34$\pm$2.26 1.07$\pm$0.33 33.62$\pm$1.65 Management 8.24 23.39$\pm$6.12 30.88$\pm$5.26 1.21$\pm$0.41 67.61$\pm$2.67 Deepseek- Reasoner Query 20.19 25.91$\pm$4.01 29.95$\pm$1.67 3.27$\pm$0.57 47.25$\pm$0.57 Management 15.91 23.43$\pm$7.44 15.27$\pm$4.31 12.53$\pm$1.41 61.40$\pm$3.01 Deepseek- Coder Query 25.00 25.32$\pm$3.64 48.59$\pm$3.76 2.44$\pm$0.64 38.14$\pm$2.54 Management 18.75 40.33$\pm$2.81 19.78$\pm$4.02 3.63$\pm$1.08 72.59$\pm$1.81 Claude- Sonnet-4.5 Query 24.63 23.35$\pm$4.33 15.95$\pm$3.08 4.68$\pm$0.73 56.71$\pm$3.79 Management 15.38 16.64$\pm$7.31 4.40$\pm$2.15 17.36$\pm$1.92 61.39$\pm$4.24 Gemini- 2.5-Flash Query 23.90 19.36$\pm$4.15 41.37$\pm$3.65 1.41$\pm$0.39 43.92$\pm$1.71 Management 15.38 26.34$\pm$13.33 10.11$\pm$3.08 9.12$\pm$3.18 72.60$\pm$4.47 GPT- 4o-mini Query 12.50 8.97$\pm$3.41 29.17$\pm$1.19 8.24$\pm$0.94 34.56$\pm$2.89 Management 14.20 18.61$\pm$4.12 7.80$\pm$2.22 21.21$\pm$3.29 48.20$\pm$4.40 GPT- 5-mini Query 20.24 27.73$\pm$2.29 22.39$\pm$2.28 2.49$\pm$0.32 55.77$\pm$2.24 Management 14.84 32.89$\pm$1.77 7.14$\pm$1.68 8.46$\pm$1.08 77.01$\pm$3.23 GPT-5 Query 25.12 28.99$\pm$3.65 20.34$\pm$2.84 2.54$\pm$0.53 57.80$\pm$3.56 Management 31.87 35.95$\pm$3.41 8.57$\pm$2.17 9.01$\pm$2.73 74.46$\pm$2.48

Difficulty annotations are present for both Spider and BIRD. Spider categorized task difficulty into four levels: easy, medium, hard, and extra hard, based on the complexity of SQL components, such as the number of SELECT columns, WHERE conditions, use of GROUP BY, nested subqueries, and advanced operations like EXCEPT or INTERSECT. BIRD provides a difficulty label that includes simple, moderate, and challenge in their dataset. However, LiveSQLBench only provides operation labels as Query or Management rather than difficulty labels, we follow this taxonomy as another distinguishing way between them. Figure 8 shows a concrete challenge-level task, on which almost all models commit RBAC violations.

Table 11 shows that in RBAC-LiveSQLBench, as task category changes, text-to-SQL models struggle more with access control compliance. Specifically, we observe that AC-F1 score consistently decreases from management to more complex query tasks across all models. This decline indicates that models become more prone to access control errors as task difficulty increases. A closer look reveals for most LLMs, RBAC violations are more common than over-refusals, which suggests that models prioritize generating executable SQL queries over adhering to access control constraints.

The degradation in access control performance is most pronounced on LiveSQLBench, where nearly all models, particularly open-weight ones, exhibit exceptionally high violation rates. For instance, Gemma-3-27B’s violation rate surges to 55.27% , a sharp increase from its performance on Spider and BIRD. This is likely attributable to the benchmark’s design, which mirrors complex enterprise environments with long and detailed schema descriptions. The extensive context can overwhelm the model’s attention mechanisms, distracting it from the specific RBAC constraints outlined in the prompt. Table 12 shows the per-difficulty results for all tested models on BIRD while Table 13 shows per-difficulty results on Spider, which lead to similar conclusions.

*Table 12. RBAC-BIRD difficulty tier performance.*

| Model | Difficulty | EX | Safe-EX | Viol. (%) | OR (%) | AC-F1 |

$\pm$ $\pm$ $\pm$ $\pm$| Snowflake- R1-7b | Simple | 52.74 | 69.210.99 | 58.461.61 | 0.050.06 | 58.371.65 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 32.73 | 50.271.53 | 65.601.08 | 0.140.18 | 50.971.30 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 19.05 | 47.016.00 | 80.000.74 | 0.000.00 | 33.131.32 |

$\pm$ $\pm$ $\pm$ $\pm$| Llama3- SQLCoder- 8b | Simple | 32.13 | 48.421.71 | 58.811.69 | 0.000.00 | 58.271.72 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 10.38 | 21.643.71 | 65.691.00 | 0.000.00 | 51.081.12 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 6.06 | 21.393.00 | 80.171.00 | 0.000.00 | 33.081.40 |

$\pm$ $\pm$ $\pm$ $\pm$| Gemma3-4b | Simple | 32.01 | 46.022.01 | 58.791.74 | 0.000.00 | 58.281.74 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 11.74 | 24.033.34 | 65.691.00 | 0.000.00 | 51.081.12 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 5.63 | 16.922.96 | 80.171.00 | 0.000.00 | 33.081.40 |

$\pm$ $\pm$ $\pm$ $\pm$| Gemma3- 27b | Simple | 53.20 | 59.291.13 | 30.151.49 | 1.910.45 | 70.951.64 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 29.80 | 35.964.74 | 36.071.64 | 1.760.26 | 63.251.31 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 15.15 | 33.395.59 | 35.841.32 | 0.520.51 | 51.511.43 |

$\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5- Coder-7b- Instruct | Simple | 43.66 | 55.311.32 | 58.841.72 | 0.000.00 | 58.261.73 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 19.64 | 30.553.48 | 65.600.92 | 0.000.00 | 51.121.08 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 10.82 | 24.523.50 | 80.091.06 | 0.000.00 | 33.111.41 |

$\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5- 14b- Instruct | Simple | 40.40 | 53.770.79 | 48.691.48 | 0.140.09 | 62.631.67 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 16.48 | 32.633.06 | 57.521.64 | 0.140.11 | 54.241.32 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 8.66 | 27.572.13 | 69.182.44 | 0.090.17 | 36.311.97 |

$\pm$ $\pm$ $\pm$ $\pm$| Deepseek- V3.2- Reasoner | Simple | 51.22 | 63.861.10 | 7.820.72 | 1.260.27 | 89.770.46 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 30.70 | 46.234.60 | 8.800.84 | 1.670.46 | 86.151.76 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 14.72 | 36.215.23 | 11.430.97 | 1.210.50 | 74.622.91 |

$\pm$ $\pm$ $\pm$ $\pm$| Deepseek- V3.2- Coder | Simple | 49.94 | 62.721.15 | 42.891.35 | 0.190.16 | 65.491.56 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 28.44 | 47.423.86 | 49.161.73 | 0.090.11 | 58.151.29 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 13.85 | 32.185.92 | 68.141.36 | 0.000.00 | 36.781.60 |

$\pm$ $\pm$ $\pm$ $\pm$| Claude- Sonnet-4.5 | Simple | 65.19 | 70.420.20 | 7.080.50 | 1.910.23 | 89.700.62 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 44.92 | 57.403.81 | 7.671.05 | 1.810.57 | 87.271.37 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 27.71 | 48.727.58 | 7.880.93 | 1.470.70 | 79.623.51 |

$\pm$ $\pm$ $\pm$ $\pm$| Gemini- 2.5-Flash | Simple | 61.35 | 73.130.68 | 18.251.34 | 0.950.15 | 80.691.26 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 44.92 | 57.992.05 | 22.482.51 | 0.950.36 | 74.042.38 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 28.57 | 49.845.03 | 25.022.20 | 0.690.44 | 59.803.00 |

$\pm$ $\pm$ $\pm$ $\pm$| GPT-4o- mini | Simple | 46.45 | 47.521.25 | 24.421.18 | 6.540.76 | 69.050.86 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 24.60 | 31.262.77 | 32.551.00 | 3.020.31 | 63.731.54 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 12.55 | 23.570.76 | 36.191.61 | 3.380.57 | 45.392.60 |

$\pm$ $\pm$ $\pm$ $\pm$| GPT-5- mini | Simple | 53.43 | 63.960.91 | 10.711.14 | 0.790.22 | 87.510.92 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 31.15 | 44.503.30 | 13.591.50 | 0.410.22 | 82.891.69 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 16.02 | 38.607.02 | 16.971.63 | 0.610.35 | 68.593.42 |

$\pm$ $\pm$ $\pm$ $\pm$| GPT-5 | Simple | 55.88 | 68.601.62 | 9.310.90 | 0.860.09 | 88.790.68 |

$\pm$ $\pm$ $\pm$ $\pm$| Moderate | 32.51 | 44.395.50 | 11.201.20 | 0.410.26 | 85.391.51 |

$\pm$ $\pm$ $\pm$ $\pm$| Challenge | 15.58 | 36.577.66 | 13.331.04 | 0.520.33 | 73.552.62 |

*Table 13. RBAC-Spider difficulty tier performance.*

| Model | Difficulty | EX | Safe-EX | V (%) | OR (%) | AC-F1 |

$\pm$ $\pm$ $\pm$ $\pm$| Snowflake-R1-7b | Easy | 89.75 | 92.401.39 | 29.842.28 | 0.240.32 | 82.031.72 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 83.50 | 80.651.52 | 49.151.32 | 0.000.00 | 66.841.20 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 74.66 | 70.653.34 | 48.971.68 | 0.110.23 | 67.121.36 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 60.40 | 60.924.30 | 55.062.76 | 0.000.00 | 61.782.52 |

$\pm$ $\pm$ $\pm$ $\pm$| Llama3-SQLCoder-8b | Easy | 75.00 | 82.462.48 | 30.972.35 | 0.000.00 | 81.651.64 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 61.66 | 65.961.58 | 50.451.38 | 0.000.00 | 66.261.22 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 46.55 | 44.882.40 | 49.771.52 | 0.000.00 | 66.861.36 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 39.16 | 38.204.87 | 55.422.56 | 0.000.00 | 61.622.43 |

$\pm$ $\pm$ $\pm$ $\pm$| Gemma3-4b | Easy | 87.50 | 86.430.79 | 30.722.45 | 0.000.00 | 81.771.68 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 72.20 | 77.171.24 | 50.451.38 | 0.000.00 | 66.261.22 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 57.47 | 60.381.70 | 49.771.52 | 0.000.00 | 66.861.36 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 43.37 | 41.184.57 | 55.422.56 | 0.000.00 | 61.622.43 |

$\pm$ $\pm$ $\pm$ $\pm$| Gemma3-27b | Easy | 93.15 | 90.070.35 | 15.321.47 | 3.070.20 | 87.741.32 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 85.20 | 85.051.33 | 33.681.71 | 0.130.11 | 74.501.48 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 71.26 | 70.854.12 | 40.811.03 | 0.690.23 | 70.471.00 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 56.02 | 54.484.94 | 41.453.98 | 0.120.24 | 68.133.07 |

$\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5-Coder-7b-Instruct | Easy | 91.53 | 91.360.30 | 30.732.48 | 0.000.00 | 81.771.70 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 80.94 | 77.651.52 | 50.271.39 | 0.000.00 | 66.341.22 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 66.67 | 67.482.85 | 49.771.52 | 0.000.00 | 66.861.36 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 40.96 | 43.054.58 | 55.422.56 | 0.000.00 | 61.622.43 |

$\pm$ $\pm$ $\pm$ $\pm$| Qwen2.5-14b-Instruct | Easy | 86.29 | 82.652.85 | 19.032.20 | 1.450.66 | 86.811.82 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 76.91 | 76.381.46 | 36.231.46 | 0.450.14 | 72.801.17 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 59.77 | 60.411.00 | 40.122.92 | 0.000.00 | 71.481.79 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 39.16 | 43.025.49 | 45.303.20 | 0.000.00 | 66.272.76 |

$\pm$ $\pm$ $\pm$ $\pm$| Deepseek-V3.2-Reasoner | Easy | 90.57 | 85.620.65 | 1.130.90 | 3.870.70 | 96.291.15 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 82.49 | 80.010.82 | 2.020.58 | 1.620.52 | 96.340.56 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 60.27 | 65.262.95 | 1.610.67 | 2.300.73 | 96.070.72 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 64.80 | 52.188.51 | 3.491.50 | 2.171.35 | 93.612.87 |

$\pm$ $\pm$ $\pm$ $\pm$| Deepseek-V3.2-Coder | Easy | 90.98 | 84.930.14 | 17.981.52 | 1.530.53 | 87.331.57 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 82.49 | 80.641.66 | 32.241.03 | 0.360.11 | 75.110.83 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 65.07 | 54.922.02 | 38.052.01 | 0.340.28 | 72.211.57 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 69.20 | 49.786.52 | 42.412.60 | 0.000.00 | 67.722.46 |

$\pm$ $\pm$ $\pm$ $\pm$| Claude-Sonnet-4.5 | Easy | 91.53 | 88.991.17 | 1.940.82 | 3.790.32 | 95.780.84 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 89.91 | 81.492.12 | 4.980.17 | 2.600.59 | 92.510.76 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 82.18 | 79.671.81 | 5.060.76 | 3.220.93 | 91.911.33 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 67.47 | 62.275.94 | 6.751.68 | 2.771.35 | 89.653.10 |

$\pm$ $\pm$ $\pm$ $\pm$| Gemini-2.5-Flash | Easy | 94.26 | 88.071.52 | 2.820.72 | 3.710.54 | 95.220.90 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 89.85 | 82.932.38 | 9.730.52 | 1.880.48 | 89.120.76 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 86.99 | 73.893.35 | 13.220.81 | 1.610.43 | 86.751.05 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 64.00 | 69.137.91 | 18.921.18 | 0.720.59 | 81.670.81 |

$\pm$ $\pm$ $\pm$ $\pm$| GPT-4o-mini | Easy | 90.98 | 66.821.58 | 5.000.91 | 18.312.05 | 81.302.13 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 79.70 | 67.421.84 | 14.621.01 | 8.520.79 | 77.981.04 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 61.64 | 50.973.58 | 22.991.50 | 7.130.59 | 74.091.87 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 58.80 | 42.725.86 | 26.382.99 | 6.871.89 | 69.264.18 |

$\pm$ $\pm$ $\pm$ $\pm$| GPT-5-mini | Easy | 90.16 | 79.300.93 | 1.290.59 | 3.150.97 | 96.711.15 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 77.66 | 73.151.27 | 4.210.69 | 1.120.42 | 94.770.62 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 68.49 | 65.664.47 | 2.070.78 | 1.720.36 | 96.250.94 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 60.40 | 53.385.17 | 5.901.04 | 1.200.76 | 92.361.75 |

$\pm$ $\pm$ $\pm$ $\pm$| GPT-5 | Easy | 89.75 | 74.521.47 | 1.370.41 | 3.630.57 | 96.300.63 |

$\pm$ $\pm$ $\pm$ $\pm$| Medium | 77.41 | 68.861.04 | 3.000.61 | 1.260.54 | 95.770.60 |

$\pm$ $\pm$ $\pm$ $\pm$| Hard | 68.49 | 62.642.97 | 1.490.46 | 1.950.46 | 96.550.65 |

$\pm$ $\pm$ $\pm$ $\pm$| Extra | 58.00 | 54.447.63 | 4.221.26 | 1.450.98 | 93.742.34 |

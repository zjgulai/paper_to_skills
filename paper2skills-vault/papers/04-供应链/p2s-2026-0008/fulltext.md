<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2606.29366
     paper_id : p2s-2026-0008
     source   : https://arxiv.org/html/2606.29366v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Solver-Verified Formulation Generation and Selection for Multi-Warehouse Inventory Allocation Using Large Language Models

Journal: European Journal of Operational Research
Jintao Xu Email: xujintao.3014@jd.com Affiliation: Supply Chain Tech Team Y, JD.com    Yingzheng Ma Email: mayingzheng.1@jd.com Affiliation: Supply Chain Tech Team Y, JD.com    Jiong Dong Email: dongjiong.1@jd.com Affiliation: Supply Chain Tech Team Y, JD.com    Yongzhi Qi Email: qiyongzhi1@jd.com Corresponding author: Corresponding author. Affiliation: Supply Chain Tech Team Y, JD.com    Jianshen Zhang Email: zhangjianshen@jd.com Affiliation: Supply Chain Tech Team Y, JD.com    Dongyang Geng Email: gengdongyang@jd.com Affiliation: Supply Chain Tech Team Y, JD.com    Anni Zhang Email: zhanganni3@jd.com Affiliation: Supply Chain Tech Team Y, JD.com

###### Abstract

Balance-oriented multi-warehouse inventory allocation is a recurring decision problem in large-scale e-commerce supply chains, in which a fixed replenishment quantity is distributed across warehouses to balance post-allocation inventory coverage while accounting for demand forecasts and heterogeneous allocation constraints. In practice, allocation requirements are often scenario-dependent and expressed in semi-structured or natural-language form rather than as ready-to-solve operations research (OR) formulations. We propose an OR-guided Large Language Model (LLM) for Allocation (ORLA) that uses solver feedback to generate, verify, and select OR formulations. ORLA integrates automatic “Problem–Model–Code (PMC)” generation, learning-based formulation selection, and feasibility restoration. We develop three complementary mixed-integer programming formulation families based on deviation minimization, soft band compliance, and knapsack-inspired allocation, together with solver-ready mixed-integer linear programming reformulations, modular constraint extensions, and a penalty-based relaxation mechanism for infeasible cases. The LLM component generates candidate formulations and executable solver code from textual or semi-structured specifications, while the solver provides verification signals for executability, feasibility, and solution quality. To address instance heterogeneity, ORLA estimates the expected quality of candidate formulations, selects promising candidates, and combines their outputs through score-aware aggregation. Experimental results on 29 production evaluation batches from JD.com show that the best single OR formulation improves allocation accuracy by 3.4 percentage points over the incumbent approach, while the full ORLA framework achieves a 4.5 percentage-point overall improvement and improves allocation accuracy in 26 of the 29 evaluation batches.

###### Keywords:

Multi-warehouse inventory allocation , AI+OR , Large language model , Supply chain management , E-commerce

## 1 Introduction

Multi-warehouse inventory allocation is a fundamental decision problem in operations management, supply chain planning, and operations research (OR). Given a fixed replenishment quantity, current on-hand inventory, forecast demand, and a collection of domain-specific allocation rules, the decision maker must determine how many units to allocate to each warehouse. This problem is closely related to classical topics such as inventory control, replenishment planning, and supply chain operations (Silver et al., 1998; Zipkin, 2000; Axsäter, 2015; Howard and Marklund, 2011; Simchi-Levi et al., 2008).

In this paper, a balance-oriented multi-warehouse inventory allocation problem under heterogeneous problem-specific side constraints is studied. The central modeling objective is to allocate a fixed replenishment quantity across warehouses so that post-allocation inventory coverage remains balanced, while hard constraints are respected whenever feasible. We measure inventory coverage through Target Inventory Days (TID), defined as the ratio between available inventory and forecast demand, and use TID-based balance as the organizing principle of the optimization problem.

From a modeling perspective, different allocation scenarios motivate different optimization formulation views. Some scenarios are naturally expressed by minimizing global deviations from a common coverage target. Others are better captured through target-band compliance with controlled violations. Still others require selective allocation under tightly coupled budget and service constraints. These differences suggest the need for a family of complementary formulations rather than a single universal model.

Another challenge is that heterogeneous allocation requirements are frequently specified in semi-structured or natural-language form rather than as ready-to-solve optimization models. This challenge is related to recent work on using large language models (LLMs) to translate natural-language descriptions into mathematical formulations and/or solver-compatible code pipelines (Ramamonjison et al., 2022; Astorga et al., 2025; Jiang et al., 2025; Huang et al., 2025; Ahmaditeshnizi et al., 2024). In contrast to this general line of work, this paper focuses on multi-warehouse inventory allocation problems.

We propose an OR-guided LLM for Allocation (ORLA), a solver-verified methodology that combines exact mixed-integer programming (MIP) modeling, automated formulation generation, feasibility relaxation, and predictive formulation selection. On the OR side, three complementary model families are developed: a sum-of-deviations formulation that directly minimizes aggregate imbalance, a soft band-constrained formulation that penalizes violations of target coverage intervals, and a knapsack-inspired formulation that captures budget-coupled allocation preferences under selective service logic. These formulations are further reformulated as mixed-integer linear programs (MILPs) and extended through a modular library of heterogeneous side constraints. To handle the case in which strict requirements render the OR model infeasible, a penalty-based relaxation mechanism is further introduced.

On the artificial intelligence (AI) side, each decision instance is represented as a Problem–Model–Code (PMC) triple, consisting of a textual problem specification, an optimization model, and executable solver code. LLMs are attractive in this setting because instruction-tuned and feedback-aligned models have shown strong instruction-following capabilities (Ouyang et al., 2022; Bai et al., 2022). This representation supports an end-to-end pipeline in which an LLM generates candidate optimization formulations and corresponding code from natural-language or semi-structured allocation requirements.

After that, the generated code is executed to call a MILP solver, and the resulting status and solution are used to verify executability, feasibility, and solution quality. These outputs are further used to detect invalid formulations, construct solver-grounded training signals, and keep the generated decisions auditable. To improve robustness, supervised fine-tuning (SFT) is adopted, followed by preference optimization based on binary desirability signals (Ouyang et al., 2022; Ethayarajh et al., 2024).

To adapt to instance-level heterogeneity, we propose a formulation-selection framework that predicts the expected quality of candidate LLM experts for each instance. Guided by the predicted scores, the framework selects the most promising candidates, and combines their outputs through score-aware aggregation.

Our AI+OR methodology is evaluated on real-world multi-warehouse inventory allocation instances from JD.com. The computational study examines the reliability of solver-oriented code generation, the behavior of feasibility restoration under conflicting constraints, and the generalization of the automated modeling pipeline. Furthermore, on real-world allocation scenarios from JD.com, ORLA improves TID-based allocation accuracy by 4.5 percentage points overall, with gains observed in 26 of the 29 evaluation batches.

The remainder of this paper is organized as follows. In Section 2, we review related work on LLM-driven automated OR modeling, post-training methods for LLMs, and MIP modeling. Section 3 introduces the balance-oriented multi-warehouse inventory allocation problem and provides an overview of the ORLA framework. Section 4 presents the three OR formulation families. Section 5 describes the PMC generation pipeline and the formulation-selection mechanism. Section 6 reports methodological evaluations of code-generation reliability and feasibility relaxation. Section 7 presents real-world validation results. Finally, we conclude in Section 8.

## 2 Related Work

### 2.1 LLM-Driven OR Modeling

Recent work has begun to systematically study how LLMs can support OR workflows, especially in translating textual problem descriptions into solver-ready optimization artifacts. The NL4Opt competition formalized this direction by evaluating models on entity extraction and intermediate-representation generation for optimization problems from natural-language descriptions (Ramamonjison et al., 2022). Moving beyond problem understanding, several works have investigated end-to-end autoformulation, in which an LLM generates optimization models, and in some cases executable code (Astorga et al., 2025; Jiang et al., 2025). ORLM studies customized training pipelines for automated optimization modeling (Huang et al., 2025), while OptiMUS highlights scalable model generation with solver-in-the-loop verification (Ahmaditeshnizi et al., 2024). From the data-generation perspective, MILP-Evolve demonstrates that LLM-guided evolution can be used to generate diverse MILP classes at scale, thereby alleviating dataset scarcity in optimization-oriented model training (Li et al., 2025).

### 2.2 Post-Training Methods for LLMs

SFT is the standard starting point for instruction-following behavior and structured output generation (Ouyang et al., 2022). Beyond SFT, alignment methods such as reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Ziegler et al., 2019; Ouyang et al., 2022) and more lightweight offline preference objectives, including direct preference optimization (DPO) (Rafailov et al., 2023), identity preference optimization (IPO) (Gheshlaghi Azar et al., 2024), simple preference optimization (SimPO) (Meng et al., 2024), and Kahneman-Tversky optimization (KTO) (Ethayarajh et al., 2024), have been widely used to improve response quality under task-specific desirability criteria.

### 2.3 MIP Modeling

The broad applicability of MIP across production-inventory planning, network design, routing, and facility location has made it one of the dominant modeling paradigms in OR (Eskandarpour et al., 2015; Baldacci et al., 2012; Melo et al., 2009; Kasirzadeh et al., 2017). Three ideas are particularly relevant to our formulation design: Big-$M$ modeling for conditional logic (Williams, 2013), knapsack-style modeling for capacity-coupled allocation (Kellerer et al., 2004), and equivalent linearization for handling absolute values within MILP models (Williams, 2013; Bertsimas and Tsitsiklis, 1997).

## 3 Problem Setup and Methodology Overview

### 3.1 Problem Setup

A balance-oriented multi-warehouse inventory allocation problem under heterogeneous side constraints is studied in this paper. Consider a set of warehouses indexed by $k\in\{1,\ldots,n\}$. For a given stock keeping unit (SKU), the input consists of: (i) the on-hand inventory $I_{k}\geq 0$ at warehouse $k$, (ii) the forecasted daily demand $D_{k}>0$, and (iii) the total replenishment quantity $R\in\mathbb{Z}_{\geq 0}$ to be allocated across warehouses.

The decision is an integer allocation plan $x_{k}\in\mathbb{Z}_{\geq 0}$ for each warehouse $k$, satisfying the conservation constraint $\sum_{k=1}^{n}x_{k}=R$. The goal is to produce a feasible and high-quality allocation plan that balances inventory coverage across warehouses while satisfying heterogeneous allocation constraints.

For each warehouse $k$, the target inventory days (TID) and target inventory levels (TIL) are defined as

$\tau_{k}:=\frac{I_{k}+x_{k}}{D_{k}},~~~T_{k}:=\tau_{\mathrm{all}}\,D_{k},$ | | | |

respectively, and the system-level TID is defined as

$\tau_{\mathrm{all}}:=\frac{\sum_{k=1}^{n}(I_{k}+x_{k})}{\sum_{k=1}^{n}D_{k}}.$ | | | |

To measure the balance quality for each allocation plan, we say that warehouse $k$ is $(\underline{\ell},\overline{\ell})$-accurately allocated if its TID lies within a relative band around $\tau_{\mathrm{all}}$, i.e.,

$\underline{\ell}\,\tau_{\mathrm{all}}\leq\tau_{k}\leq\overline{\ell}\,\tau_{\mathrm{all}},$ | | | |

where $0<\underline{\ell}\leq 1\leq\overline{\ell}$ are given control parameters. Based on these notions, the multi-warehouse allocation accuracy (or balance rate) is defined as

$\mathrm{Acc}_{\underline{\ell},\,\overline{\ell}}(x_{1},\ldots,x_{n}):=\frac{\sum_{k:\ \tau_{k}\in[\underline{\ell}\,\tau_{\mathrm{all}},\,\overline{\ell}\,\tau_{\mathrm{all}}]}x_{k}}{\sum_{k=1}^{n}x_{k}}.$ | | | |

In this paper, the objective is to optimize allocation quality under heterogeneous side constraints, with TID-based balance serving as the central organizing principle of the problem, as shown schematically in Figure 1.

*Figure 1: Illustration of TID-based balance in multi-warehouse inventory allocation.*

### 3.2 Methodology Overview

ORLA addresses this problem through a solver-verified methodology that combines exact optimization modeling, automatic formulation generation, learning-based formulation-selection, and feasibility restoration. Figure 2 summarizes the overall framework.

*Figure 2: Overview of ORLA framework.*

Accordingly, three complementary mixed-integer formulation families are developed in Section 4: a sum-of-deviations formulation, a soft band-constrained formulation with controlled slack, and a knapsack-inspired formulation for budget-coupled allocation under selective service logic. Together, these models provide complementary views of the same underlying allocation problem.

To enable automatic generation from textual or semi-structured specifications, we represent each instance as a Problem–Model–Code (PMC) triple and use an LLM to map problem descriptions into candidate model–code pairs. The external solver then verifies feasibility and solution quality, serving both as an execution engine and as a source of correctness feedback.

Furthermore, a formulation-selection mechanism is introduced that estimates the quality of candidate LLM experts, selects promising candidates, and combines their outputs.

## 4 A Family of OR Formulations

In this section, three complementary mixed-integer formulations are developed, and we further enrich these base models through a modular constraint library and a penalty-based relaxation mechanism for instances in which strict allocation constraints render the original formulation infeasible. These components form the exact OR backbone of ORLA, as illustrated in Figure 3.

*Figure 3: Three complementary formulation families with shared constraints and relaxation modules.*

### 4.1 Sum-of-Deviations Formulation

Basic formulation. The sum-of-deviations (SD) MIP problem is formulated as follows.

$\displaystyle\min_{\boldsymbol{x}}$ $\displaystyle\sum_{k=1}^{n}\left|\frac{I_{k}+x_{k}}{D_{k}}-\tau_{\rm all}\right|$ | | | | |

$\displaystyle\text{s.t.}$ $\displaystyle\sum_{k=1}^{n}x_{k}=R,$ | | | | |

$\displaystyle x_{k}\in\mathbb{Z}_{\geq 0},\quad k=1,\ldots,n,$ | | | | |

where $\boldsymbol{x}=(x_{1},\ldots,x_{n})^{\top}$. The objective minimizes the sum of absolute deviations between each warehouse’s TID and the global TID benchmark, so that all warehouses stay as close as possible to a common turnover target. In addition, the constraints enforce total-quantity conservation and integrality/non-negativity of allocation decisions, ensuring implementable integer shipment plans in practice.

Linearized model reformulation. The basic formulation above contains absolute-value terms $\lvert\cdot\rvert$ in the objective, which are inconvenient for direct optimization in standard MILP solvers (e.g., SCIP). Therefore, an equivalent linear reformulation is derived as below.

$\displaystyle\min_{{\boldsymbol{x},\,\boldsymbol{\delta}}}$ $\displaystyle\sum_{k=1}^{n}\frac{1}{D_{k}}\,\delta_{k}$ | | | | |

$\displaystyle\text{s.t.}$ $\displaystyle\delta_{k}\geq I_{k}+x_{k}-T_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle\delta_{k}\geq-I_{k}-x_{k}+T_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle\sum_{k=1}^{n}x_{k}=R,$ | | | | |

$\displaystyle x_{k}\in\mathbb{Z}_{\geq 0},~~\delta_{k}\geq 0,\quad k=1,\ldots,n,$ | | | | |

where $\boldsymbol{x}=(x_{1},\ldots,x_{n})^{\top}$, $\boldsymbol{\delta}=(\delta_{1},\ldots,\delta_{n})^{\top}$. We refer to this reformulation as the linearized sum-of-deviations (LSD) problem.

###### Theorem 1 (Equivalence of SD and LSD).

Assume $D_{k}>0$ for all $k=1,\ldots,n$. Then SD and its linearized LSD are equivalent.

See A for the proof of Theorem 1.

### 4.2 Soft Band-Constrained Formulation

The core idea of the following soft band (SB) MILP problem is to keep each warehouse’s TID within a target interval $[\underline{\ell}\,\tau_{\mathrm{all}},\,\overline{\ell}\,\tau_{\mathrm{all}}]$ while still guaranteeing global replenishment feasibility. Instead of enforcing this interval as a hard requirement, nonnegative slack variables are introduced to measure lower-side and upper-side violations, and minimize their total magnitude.

$\displaystyle\min_{\boldsymbol{{\rm\xi^{+}}},\,\boldsymbol{{\rm\xi^{-}}},\boldsymbol{x}}$ $\displaystyle\sum_{k=1}^{n}\xi_{k}^{+}+\sum_{k=1}^{n}\xi_{k}^{-}$ | | | | |

$\displaystyle\text{s.t.}$ $\displaystyle\underline{\ell}\,\tau_{\rm all}-\xi_{k}^{-}\leq\frac{I_{k}+x_{k}}{D_{k}}\leq\overline{\ell}\,\tau_{\rm all}+\xi_{k}^{+},k=1,\ldots,n,$ | | | | |

$\displaystyle\sum_{k=1}^{n}x_{k}=R,$ | | | | |

$\displaystyle\xi_{k}^{+},\;\xi_{k}^{-}\geq 0,\;x_{k}\in\mathbb{Z}_{\geq 0},k=1,\ldots,n,$ | | | | |

where $\boldsymbol{\xi^{+}}=(\xi_{1}^{+},\ldots,\xi_{n}^{+})^{\top}$, $\boldsymbol{\xi^{-}}=(\xi_{1}^{-},\ldots,\xi_{n}^{-})^{\top}$, $\boldsymbol{x}=(x_{1},\ldots,x_{n})^{\top}$.

### 4.3 Knapsack-Inspired Formulation

Basic formulation. Inspired by classic knapsack modeling, we incorporate a TID-based balance band through a binary indicator $z_{k}$ and Big-$M$ constraints, so that the model can explicitly trade off between allocating more units and satisfying balanced allocation requirements. This formulation is referred to as the knapsack-inspired (KI) MIP problem.

$\displaystyle\max_{\boldsymbol{x},\,\boldsymbol{y}\,,\boldsymbol{z}}\quad$ $\displaystyle\lambda_{1}\sum_{k=1}^{n}y_{k}-\lambda_{2}\sum_{k=1}^{n}\left|I_{k}+x_{k}-T_{k}\right|$ | | | | |

$\displaystyle\sum_{k=1}^{n}x_{k}=R,$ | | s.t. | | |

$\displaystyle I_{k}+x_{k}\geq\underline{\ell}\,T_{k}-M(1-z_{k}),\quad k=1,\ldots,n,$ | | | | |

$\displaystyle I_{k}+x_{k}\leq\overline{\ell}\,T_{k}+M(1-z_{k}),\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\leq x_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\leq R\,z_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\geq x_{k}-R(1-z_{k}),\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\geq 0,\quad x_{k}\in\mathbb{Z}_{\geq 0},\quad z_{k}\in\{0,1\},\quad k=1,\ldots,n.$ | | | | |

where $\boldsymbol{x}=(x_{1},\ldots,x_{n})^{\top}$, $\boldsymbol{y}=(y_{1},\ldots,y_{n})^{\top}$, $\boldsymbol{z}=(z_{1},\ldots,z_{n})^{\top}$.

The second and third constraints activate a TID-based balance requirement when $z_{k}=1$, forcing warehouse $k$’s post-allocation inventory level to lie within a relative range around the target. When $z_{k}=0$, the Big-$M$ terms relax this requirement. Finally, the auxiliary variable $y_{k}$ is used to reward allocating to warehouses that satisfy the balance band: the lower bound links $y_{k}$ to $x_{k}$ when $z_{k}=1$ and suppresses it when $z_{k}=0$. A weighted objective is adopted, where the first term maximizes the amount of replenishment assigned to warehouses whose allocations are counted as band-compliant, whereas the second term penalizes aggregate deviation from the target inventory levels.

Linearized model reformulation. Similarly, nonnegative deviation variables are introduced to obtain a fully solver-ready linearized knapsack-inspired (LKI) MILP problem as follows.

$\displaystyle\max_{\boldsymbol{x},\,\boldsymbol{y},\,\boldsymbol{z},\,\boldsymbol{\delta}}\quad$ $\displaystyle\lambda_{1}\sum_{k=1}^{n}y_{k}-\lambda_{2}\sum_{k=1}^{n}\delta_{k}$ | | | | |

$\displaystyle\sum_{k=1}^{n}x_{k}=R,$ | | s.t. | | | (C1) |

$\displaystyle I_{k}+x_{k}\geq\underline{\ell}\,T_{k}-M(1-z_{k}),\quad k=1,\ldots,n,$ | | | | | (C2) |

$\displaystyle I_{k}+x_{k}\leq\overline{\ell}\,T_{k}+M(1-z_{k}),\quad k=1,\ldots,n,$ | | | | | (C3) |

$\displaystyle\delta_{k}\geq I_{k}+x_{k}-T_{k},\quad k=1,\ldots,n,$ | | | | | (C4) |

$\displaystyle\delta_{k}\geq T_{k}-I_{k}-x_{k},\quad k=1,\ldots,n,$ | | | | | (C5) |

$\displaystyle y_{k}\leq x_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\leq R\,z_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\geq x_{k}-R(1-z_{k}),\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\geq 0,\quad\delta_{k}\geq 0,\quad x_{k}\in\mathbb{Z}_{\geq 0},\quad z_{k}\in\{0,1\},\quad k=1,\ldots,n,$ | | | | |

where $\boldsymbol{\delta}=(\delta_{1},\ldots,\delta_{n})^{\top}$. With similar arguments as in the proof of Theorem 1, we know that the above two models are equivalent.

###### Theorem 2 (Equivalence of KI and linearized LKI).

Assume $\lambda_{1},\lambda_{2}>0$, and $M\geq\max\limits_{k=1,\ldots,n}\{\underline{\ell}\,T_{k}-I_{k},\,I_{k}+R-\overline{\ell}\,T_{k},\,0\}$. Then KI and the linearized MILP formulation LKI are equivalent.

### 4.4 Modular Heterogeneous Constraints

In addition to the balance-oriented requirements captured by the base formulations, real-world decision instances often involve heterogeneous operational rules, such as ratio constraints, lower and upper allocation bounds, case-pack restrictions, group-level service requirements, and cardinality controls. Rather than treating these rules as ad hoc modifications for isolated scenarios, we model them as a modular constraint library that can be attached systematically to the base formulation families.

The extension patterns of the three formulation families are largely analogous. For clarity, we use the KI formulation as a representative example and introduce a collection of optional constraint modules, denoted by OC1–OC9 as shown in Table 1.

*Table 1: Heterogeneous problem-specific side constraints OC1–OC9.*

| Constraint | Meaning | Mathematical form |

$x_{k}\geq\alpha_{k}\,R,\ \forall k\in\mathcal{C}^{\geq}_{\text{ratio}}$| OC1 | Ratio lower-bound | |

$x_{k}\leq\beta_{k}\,R,\hskip 8.50012pt\forall k\in\mathcal{C}^{\leq}_{\text{ratio}}$| OC2 | Ratio upper-bound | |

$x_{k}=\gamma_{k}\,R,\ \forall k\in\mathcal{C}^{=}_{\text{ratio}}$| OC3 | Ratio equality | |

$x_{k}\geq d_{k},\forall k\in\mathcal{C}^{\geq}$| OC4 | Quantity lower-bound | |

$x_{k}\leq e_{k},\forall k\in\mathcal{C}^{\leq}$| OC5 | Quantity upper-bound | |

$x_{k}=c_{k},\forall k\in\mathcal{C}^{=}$| OC6 | Quantity equality | |

$p_{k}\in\mathbb{Z}_{>0}$ $x_{k}=p_{k}\,q_{k},q_{k}\in\mathbb{Z}_{\geq 0},\hskip 8.50012ptk=1,\ldots,n$| OC7 | Case-pack (size: ) | |

$\sum_{l\in\mathcal{S}_{k}}x_{l}\geq\eta_{k}\,R,\hskip 8.50012ptk=1,\ldots,K$| OC8 | Group-level minimum share | |

$\sum_{k=1}^{n}s_{k}\leq m,x_{k}\leq Ms_{k},s_{k}\in\{0,1\}$| OC9 | Served warehouse limit | |

Different combinations of these modules generate a family of extended formulations, see Table 2 for details.

*Table 2: Constraint-combination variants built on the base formulation KI. Each variant is obtained by augmenting KI with a subset of additional constraints (OC1–OC9), defined in Table 1.*

| Model | OC1 | OC2 | OC3 | OC4 | OC5 | OC6 | OC7 | OC8 | OC9 |

| KI | | | | | | | | | |

| KIV1 | ✓ | | | | | | | | |

| KIV2 | | ✓ | | | | | | | |

| KIV3 | | | ✓ | | | | | | |

| KIV4 | ✓ | ✓ | | | | | | | |

| KIV5 | ✓ | | ✓ | | | | | | |

| KIV6 | | ✓ | ✓ | | | | | | |

| KIV7 | ✓ | ✓ | ✓ | | | | | | |

| KIV8 | | | | ✓ | | | | | |

| KIV9 | | | | | ✓ | | | | |

| KIV10 | | | | | | ✓ | | | |

| KIV11 | | | | ✓ | ✓ | | | | |

| KIV12 | | | | ✓ | | ✓ | | | |

| KIV13 | | | | | ✓ | ✓ | | | |

| KIV14 | | | | ✓ | ✓ | ✓ | | | |

| KIV15 | | | | | | | ✓ | | |

| KIV16 | | | | | | | | ✓ | |

| KIV17 | | | | | | | | | ✓ |

As an example, KIV1 is given as follows:

$\displaystyle\max_{\boldsymbol{x},\,\boldsymbol{y},\,\boldsymbol{z}}\quad$ $\displaystyle\lambda_{1}\sum_{k=1}^{n}y_{k}-\lambda_{2}\sum_{k=1}^{n}\left|I_{k}+x_{k}-T_{k}\right|$ | | | | |

$\displaystyle\sum_{k=1}^{n}x_{k}=R,$ | | s.t. | | |

$\displaystyle I_{k}+x_{k}\geq\underline{\ell}\,T_{k}-M(1-z_{k}),\quad k=1,\ldots,n,$ | | | | |

$\displaystyle I_{k}+x_{k}\leq\overline{\ell}\,T_{k}+M(1-z_{k}),\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\leq x_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\leq R\,z_{k},\quad k=1,\ldots,n,$ | | | | |

$\displaystyle y_{k}\geq x_{k}-R(1-z_{k}),\quad k=1,\ldots,n,$ | | | | |

$\displaystyle\boldsymbol{x_{k}\geq\alpha_{k}\,R,\ \forall k\in\mathcal{C}^{\geq}_{\text{ratio}},}$ | | | | |

$\displaystyle y_{k}\geq 0,\quad x_{k}\in\mathbb{Z}_{\geq 0},~~z_{k}\in\{0,1\},~~k=1,\ldots,n.$ | | | | |

### 4.5 Penalty-Based Relaxation

A natural consequence of heterogeneous side constraints is that strict formulations may become infeasible. Infeasibility often results from mutually inconsistent lower bounds, equality constraints, or group-share constraints. To address this issue, we introduce a penalty-based relaxation framework in which selected hard constraints are softened through slack variables and associated violation costs in the objective. This relaxation scheme is formulation-agnostic and can be applied to all model families introduced above. Table 3 summarizes the corresponding relaxation patterns.

*Table 3: Relaxation schemes for constraints, including the corresponding slack-variable modeling and penalty forms added to the objective. *

| Constraints | Relaxations | Slack variables | Penalties |

$I_{k}+x_{k}+u_{k}^{-}\geq\underline{\ell}\,T_{k}-M(1-z_{k}),\ k=1,\ldots,n,\ u_{k}^{-}\geq 0$ $\{u_{k}^{-}\}_{k=1}^{n}$ $-\rho\sum_{k=1}^{n}u_{k}^{-}$| C2 | | | |

$I_{k}+x_{k}-u_{k}^{+}\leq\overline{\ell}\,T_{k}+M(1-z_{k}),\ k=1,\ldots,n,\ u_{k}^{+}\geq 0$ $\{u_{k}^{+}\}_{k=1}^{n}$ $-\rho\sum_{k=1}^{n}u_{k}^{+}$| C3 | | | |

$x_{k}+u_{k}^{-}\geq\alpha_{k}\,R,\ \forall k\in\mathcal{C}^{\geq}_{\text{ratio}},\ u_{k}^{-}\geq 0$ $\{u_{k}^{-}\}_{k\in\mathcal{C}^{\geq}_{\text{ratio}}}$ $-\rho\sum_{k\in\mathcal{C}^{\geq}_{\text{ratio}}}u_{k}^{-}$| OC1 | | | |

$x_{k}-u_{k}^{+}\leq\beta_{k}\,R,\ \forall k\in\mathcal{C}^{\leq}_{\text{ratio}},\ u_{k}^{+}\geq 0$ $\{u_{k}^{+}\}_{k\in\mathcal{C}^{\leq}_{\text{ratio}}}$ $-\rho\sum_{k\in\mathcal{C}^{\leq}_{\text{ratio}}}u_{k}^{+}$| OC2 | | | |

$\left|x_{k}-\gamma_{k}\,R\right|\leq u_{k},\ \forall k\in\mathcal{C}^{=}_{\text{ratio}},\ u_{k}\geq 0$ $\{u_{k}\}_{k\in\mathcal{C}^{=}_{\text{ratio}}}$ $-\rho\sum_{k\in\mathcal{C}^{=}_{\text{ratio}}}u_{k}$| OC3 | | | |

$x_{k}+u_{k}^{-}\geq d_{k},\ \forall k\in\mathcal{C}^{\geq},\ u_{k}^{-}\geq 0$ $\{u_{k}^{-}\}_{k\in\mathcal{C}^{\geq}}$ $-\rho\sum_{k\in\mathcal{C}^{\geq}}u_{k}^{-}$| OC4 | | | |

$x_{k}-u_{k}^{+}\leq e_{k},\ \forall k\in\mathcal{C}^{\leq},\ u_{k}^{+}\geq 0$ $\{u_{k}^{+}\}_{k\in\mathcal{C}^{\leq}}$ $-\rho\sum_{k\in\mathcal{C}^{\leq}}u_{k}^{+}$| OC5 | | | |

$\left|x_{k}-c_{k}\right|\leq u_{k},\ \forall k\in\mathcal{C}^{=},\ u_{k}\geq 0$ $\{u_{k}\}_{k\in\mathcal{C}^{=}}$ $-\rho\sum_{k\in\mathcal{C}^{=}}u_{k}$| OC6 | | | |

$\left|x_{k}-p_{k}\,q_{k}\right|\leq u_{k}$$q_{k}\in\mathbb{Z}_{\geq 0}$$u_{k}\geq 0$$k=1,\ldots,n$ $\{u_{k}\}_{k=1}^{n}$ $-\rho\sum_{k=1}^{n}u_{k}$| OC7 | , , , | | |

$\sum_{l\in\mathcal{S}_{k}}x_{l}+u_{k}^{-}\geq\eta_{k}\,R,\ u_{k}^{-}\geq 0,k=1,\ldots,K$ $\{u_{k}^{-}\}_{k=1}^{K}$ $-\rho\sum_{k=1}^{K}u_{k}^{-}$| OC8 | | | |

$\sum_{k=1}^{n}s_{k}\leq m+u^{+}$$x_{k}\leq Ms_{k}$$s_{k}\in\{0,1\}$$u^{+}\geq 0$ $u^{+}$ $-\rho u^{+}$| OC9 | , , , | | |

## 5 LLM-Driven Formulation Generation and Selection

### 5.1 Problem–Model–Code Representation

Each decision instance is represented as a “Problem–Model–Code (PMC)” triple: a textual problem specification (Problem, P), an exact OR model (Model, M), and executable solver code (Code, C). The LLM reads the problem description, generates candidate OR formulations and code, and then relies on an external MILP solver for verification.

A key design choice is that the solver serves both as an execution engine and as a correctness filter. An infeasibility-detect–relax–resolve loop is integrated. The hard-constraint model is solved first. If the solver reports infeasibility, the corresponding relaxation model is activated and re-solved to obtain a feasible allocation plan. This loop is also central to post-training: it provides structured supervision for valid formulation/code generation and binary desirability signals for preference optimization. Figure 4 illustrates the aforementioned pipeline.

*Figure 4: LLM-driven PMC pipeline for multi-warehouse inventory allocation.*

Solver-verified supervised fine-tuning. Model $\pi_{\rm SFT}$ is obtained through SFT so that it can generate a MIP formulation together with an executable Python script that calls SCIP, from a natural-language multi-warehouse inventory allocation specification. Our SFT dataset contains solver-verifiable PMC triples paired with structured prompts, covering the base formulation and 17 real-world constraint extensions. It also includes positive PMC samples from classical MIP problems such as Max-Cut. Furthermore, we include targeted negative instances (e.g., non-linear modeling hallucinations, wrong variable types, missing key constraints) alongside positive instances to improve format robustness and error awareness.

Solver-grounded preference optimization. Following the KTO pipeline, a binary-labeled preference dataset from solver verification is constructed. Concretely, we sample a diverse set of multi-warehouse inventory allocation prompts and use the SFT policy $\pi_{\rm SFT}$ to generate a batch of outputs under a high-temperature decoding setting. For each generated candidate, we execute the produced solver script and compare its solution against both (i) the theoretical ground-truth plan derived from the reference MIP and (ii) the required replenishment quantity.

If the generated code is executable, and its executed allocation matches the ground-truth plan, we label it as a positive sample. Otherwise, it is labeled as a negative sample. Finally, we perform $1\!:\!1$ downsampling between positive and negative samples to obtain a balanced dataset in the standard binary-label format for KTO training. In practice, it is often difficult to obtain high-quality paired preference samples, while it is relatively easy to accumulate single-label positive or negative samples through continuous annotation and solver-based verification. This data characteristic naturally supports an iterative self-enhancement loop, and is one of the main reasons we adopt KTO as our preference optimization objective.

### 5.2 Learning-Based Formulation-Selection

A learning-based formulation-selection mechanism over multiple LLM experts is introduced, as shown in Figure 2. For each instance, a set of regressors estimates the expected quality of candidate experts, and the most promising Top-$K$ candidates are selected. Their outputs are then combined through score-aware weighting, and feasibility is restored when aggregation introduces mild violations.

Predictive estimation. For each candidate expert indexed by $i\in\{1,\ldots,E\}$, an expert-specific LightGBM (Ke et al., 2017) regressor $\mathrm{LightGBM}_{i}$ is trained to predict the realized allocation accuracy from instance-level features such as

$s_{ij}\leftarrow\mathrm{LightGBM}_{i}(\boldsymbol{\phi}_{j}),\quad i=1,\ldots,E,$ | | | |

where $\boldsymbol{\phi}_{j}=(\phi_{j_{1}},\ldots,\phi_{j_{l}})^{\top}$ correspond to the features of the inventory allocation instance $j$, and $s_{ij}$ is the predicted score for expert $i$ produced by the trained regressor.

Top-$\boldsymbol{K}$ selection and score-weighted combination. After generating $E$ scores, we first select the top-$K$ experts. Assume that experts $1,\ldots,K$ are selected. Let $s_{ij}$ denote the predicted score of selected expert $i\in\{1,\ldots,K\}$ for instance $j$. Softmax weights with temperature $\kappa>0$ are computed as below:

$\theta_{ij}=\frac{\exp(s_{ij}/\kappa)}{\sum_{\ell=1}^{K}\exp(s_{\ell j}/\kappa)},\qquad i=1,\ldots,K.$ | | | |

Let $x_{ij}$ denote the allocation plan produced by selected expert $i$. The fused allocation is computed as

$x_{j}=\sum_{i=1}^{K}\theta_{ij}\,x_{ij}.$ | | | |

## 6 Computational Evaluation

This section evaluates the ORLA framework from a methodological perspective. The main questions are: (i) whether post-training improves the reliability of generated code, and (ii) whether the relaxation layer recovers valid solutions under conflicting constraints. The proposed methodology is evaluated on real-world multi-warehouse inventory allocation instances from JD.com supply-chain operations. For LLM training, we fine-tune the Qwen3-8B base model by minimizing the standard token-level cross-entropy loss under teacher forcing, with AdamW (Loshchilov and Hutter, 2019) as the optimizer and NEFTune (Jain et al., 2024) applied during training. The generated MILP problems are solved using SCIP.

### 6.1 Data Pools for Post-Training

SFT data pool. We build PMC-style SFT data for all 18 formulations reported in Table 2, covering the base MIP formulation and its 17 variants. To balance canonical-pattern learning and variant coverage, we set the sampling ratio across positive multi-warehouse inventory allocation samples to $34:1:\cdots:1$. Negative multi-warehouse inventory allocation SFT samples are further constructed by injecting 6 common solver-script error types with approximately uniform frequency across categories. To improve robustness and cross-task generalization, we additionally include PMC samples from 6 classical MIP problems including Max-Cut. In the final mixture, the proportions of multi-warehouse inventory allocation positive samples, multi-warehouse inventory allocation negative samples, and classical MIP samples are 85%, 10%, and 5%, respectively.

KTO data pool. Starting from $\pi_{\rm SFT}$, we construct binary preference data for offline KTO using solver-verifiable signals. In particular, each candidate is labeled as desirable or undesirable based on executability and solution-quality criteria under the MILP solver, as discussed in Section 5.1.

### 6.2 Code Generation Reliability

In this subsection, two post-training variants are compared, namely SFT-only and SFT+KTO, with a focused evaluation on the code-generation component. Our goal is to quantify how preference optimization affects practical code quality. Specifically, three complementary metrics are reported in Table 4: code syntax accuracy, code execution success rate, and code execution failure rate. The evaluation is conducted on a held-out test set containing more than 10,000 real-world multi-warehouse allocation instances.

*Table 4: Code quality comparison between SFT and SFT+KTO versions.*

| Evaluation Metric | SFT | SFT+KTO |

| Code Syntax Accuracy | 99.9% | 100% |

| Code Execution Success Rate | 99% | 99.9% |

| Code Execution Failure Rate | 1% | 0.1% |

On the held-out test set, the SFT+KTO variant attains 100% code syntax accuracy and reduces the code execution failure rate from 1.0% to 0.1% relative to the SFT-only variant. It is worth noting that these results are obtained under a constrained PMC protocol, where the MILP modeling and solver APIs schema are included during post-training. We observe that preference optimization improves the reliability of the generated solver code on the held-out test set. In the following experiments, “LLM” refers to the post-trained model obtained via SFT+KTO.

### 6.3 Relaxation

Two multi-warehouse inventory allocation tasks are presented to illustrate how relaxation strategies are applied in practice when hard constraints render the LLM-generated OR model infeasible, as discussed in Section 4.5. To fully present the details of model relaxation, we use the linearized model form, i.e., MILP, in this section.

Group minimum share task. In this task, each predefined warehouse group is required to receive at least a minimum fraction of the total replenishment quantity. Formally, for each group $k$, the model enforces a lower-bound share constraint of the form $\sum_{l\in\mathcal{S}_{k}}x_{l}\geq\eta_{k}\,R,~k=1,\ldots,K$. Infeasibility can occur when multiple group-level minimum-share constraints are jointly over-restrictive relative to the fixed total budget $R$ and other hard bounds. A typical conflicting case arises when the groups are disjoint and their required minimum shares satisfy $\eta_{1}+\eta_{2}>1$. In our task, we set $R=252$, $(D_{1},\ldots,D_{8})=(18.1,9.7,20.3,44.7,17.4,11.5,35.2,8.9)$, $(I_{1},\ldots,I_{8})=(33,13,34,102,28,21,66,14)$, $\eta_{1}=0.6$, $\eta_{2}=0.5$, $\underline{\ell}=0.8$, $\overline{\ell}=1.2$, $(\rho_{1},\rho_{2},\rho_{3})=\text{(200, 200, 10000)}$, and $M=1563$. We then relax the corresponding hard constraints and obtain the following relaxation model.

Allocation quantity constraints task. In this task, selected warehouses are required to satisfy explicit quantity constraints, including lower-bound constraints $x_{k}\geq d_{k}$ for $k\in\mathcal{C}^{\geq}$ and equality constraints $x_{k}=c_{k}$ for $k\in\mathcal{C}^{=}$. Typical conflicting cases include over-constrained equality assignments, incompatible lower bounds across multiple warehouses, or the aggregate required quantity exceeding the available replenishment budget. The concrete parameter values used in this case are listed as follows: $d_{5}=45$ for $k\in\mathcal{C}^{\geq}=\{5\}$, $c_{6}=33$ for $k\in\mathcal{C}^{=}=\{6\}$, $R=72$, $\underline{\ell}=0.8$, $\overline{\ell}=1.2$, and penalty weights $(\rho_{1},\rho_{2})=(10000,10000)$. Under the above setting, the original MILP can become infeasible. Then we relax the hard constraints and obtain the following relaxation model.

## 7 Real-World Validation

We now move from computational evaluation to real-world validation. Section 7.1 reports TID-based allocation accuracy on 29 production evaluation batches from JD.com, comparing the incumbent allocation procedure with the ORLA framework. Section 7.2 further demonstrates the generalization capability on task categories not included in post-training, using solver verification of generated code and validity checks of the resulting allocation plans.

### 7.1 Production-Batch Allocation Accuracy Evaluation

We evaluate the production performance of ORLA on 29 real-world evaluation batches from JD.com. The incumbent allocation procedure is used as the benchmark. We compare it with two OR-driven alternatives. The best-performing single formulation: KI modeling, and the full learning-based predictive formulation-selection method.

Figure 5 reports the batch-level changes in TID-based allocation accuracy relative to the incumbent. The fixed KI formulation improves 26 of the 29 production batches and achieves an overall improvement of 3.4 percentage points (pp). After applying the learning-based formulation-selection method, the overall improvement increases to 4.5 pp. Thus, both OR-driven approaches improve allocation accuracy in most production batches, and the formulation-selection method provides a larger aggregate gain than the best fixed formulation.

*Figure 5: TID-based allocation accuracy changes across 29 evaluation batches. Dashed lines indicate the overall improvements of the KI expert and the formulation-selection (FS) method.*

Detailed batch-level changes are reported in Table 5. The formulation-selection method outperforms the fixed KI formulation in 19 of these 26 batches, indicating that its advantage is not driven by a small number of isolated cases but is reflected at the batch level. A closer inspection shows that the gain from the formulation-selection method is particularly pronounced when the fixed KI formulation provides only moderate improvements. For example, in B19 and B20, the KI formulation improves accuracy by 0.8 and 2.9 pp, respectively, whereas the formulation-selection method increases the corresponding gains to 5.9 and 7.2 pp. This suggests that the formulation-selection method can compensate for cases in which a fixed formulation does not adequately match the structure of a production batch. The formulation-selection method also preserves and amplifies the high-gain cases of the KI formulation. For all batches in which KI improves accuracy by more than 8 pp, the formulation-selection method achieves larger gains, with additional improvements over KI ranging from 0.2 to 1.6 pp. Moreover, the number of batches with gains above 5 pp increases from 11 under KI to 17 under the formulation-selection method. These results show that the benefit of the formulation-selection method is not limited to correcting weaker KI cases. It also strengthens performance in batches where the best single formulation is already highly effective. The downside relative to KI is limited in both frequency and magnitude. KI outperforms the formulation-selection method in 7 of the 26 improved batches, and only two of these cases show a gap of at least one percentage point. Overall, these results show that learning-based formulation-selection increases the upside of the allocation decision while introducing only moderate downside relative to the best fixed formulation.

*Table 5: Batch-level allocation accuracy gains of KI and the formulation-selection (FS) method for the 26 production batches with positive gains relative to the incumbent.*

| Batch ID | KI (pp) | FS (pp) | Batch ID | KI (pp) | FS (pp) |

| B1 | 3.7 | 3.1 | B17 | 5.1 | 5.3 |

| B3 | 0.3 | 2.1 | B18 | 3.2 | 3.0 |

| B6 | 2.1 | 1.3 | B19 | 0.8 | 5.9 |

| B7 | 13.0 | 13.6 | B20 | 2.9 | 7.2 |

| B8 | 4.0 | 6.3 | B21 | 0.7 | 0.8 |

| B9 | 4.6 | 5.2 | B22 | 7.1 | 6.9 |

| B10 | 5.5 | 4.5 | B23 | 3.8 | 6.6 |

| B11 | 8.2 | 9.2 | B24 | 5.9 | 6.9 |

| B12 | 2.9 | 4.9 | B25 | 7.2 | 5.6 |

| B13 | 4.0 | 5.5 | B26 | 8.6 | 9.2 |

| B14 | 9.6 | 9.8 | B27 | 7.7 | 10.0 |

| B15 | 8.3 | 9.9 | B28 | 1.7 | 1.6 |

| B16 | 4.1 | 6.6 | B29 | 4.5 | 4.6 |

Figure 6 presents a paired comparison between the KI formulation and the formulation-selection method over 26 improved batches, focusing on the batch-level response pattern of the formulation-selection method. The two curves exhibit broadly similar movements across production batches, indicating that the formulation-selection method preserves the main allocation behavior induced by the KI formulation. This pattern suggests that the KI formulation captures part of the common structure underlying TID-based accuracy improvements, whereas the formulation-selection method serves as an adaptive refinement mechanism for instance-level heterogeneity. In this sense, the formulation-selection maintains the stable batch-wise behavior of the fixed formulation while selectively modifying allocation decisions when the predicted formulation quality supports such adaptation.

*Figure 6: Paired comparison of batch-level allocation accuracy gains between KI and the formulation-selection (FS) method over the 26 improved production batches. *

### 7.2 Generalization Evaluation

Beyond the production-batch evaluation, we further evaluate the generalization performance of LLM on unseen task categories. For 5 task categories not included in post-training, we run inference with LLM and invoke the solver to verify the generated code, while checking the rationality of the resulting inventory allocation plans. Table 6 reports results from two dimensions: code syntax accuracy (CSA) and allocation validity (AV).

*Table 6: Generalization on unseen multi-warehouse inventory allocation tasks*

| Task Name | Task Description | CSA | AV |

| Equal replenishment | Enforce equal replenishment across warehouses | ✓ | ✓ |

| Minimum order quantity | Enforce minimum order quantity per warehouse | ✓ | ✓ |

| Mutually exclusive replenishment | Enforce mutual exclusivity among specified warehouses | ✓ | ✓ |

| Fixed allocation ratio | Enforce a fixed allocation ratio between two warehouses | ✓ | ✗ |

| Minimum TID requirement | Enforce minimum TID | ✓ | ✗ |

Our LLM generates syntactically correct and executable solver code for all 5 scenarios. Moreover, for 3 of the 5 scenarios, the generated allocation plans are valid and satisfy all hard constraints. The generalization evaluation provides additional evidence beyond the production-batch results. ORLA improves the TID-based allocation accuracy in most real-world evaluation batches, and the LLM component retains partial generalization ability when tested on allocation requirements not included in post-training.

## 8 Conclusion

This paper develops ORLA, a solver-verified and LLM-driven AI+OR methodology for multi-warehouse inventory allocation under heterogeneous domain-specific allocation rules. Three complementary MIP formulations, together with exact MILP reformulations, a modular constraint library, and a formulation-agnostic relaxation scheme, are proposed. Furthermore, we show how solver verification can support structured generation not only at execution time but also during post-training for LLMs through correctness-oriented supervision and preference signals. Empirically, high reliability in solver-oriented code generation, effective restoration of feasibility under conflicting constraints, and generalization are demonstrated. The real-world case study further shows that the formulation-family approach yields consistent gains over an incumbent allocation process, with additional benefits from instance-dependent formulation selection.

Several directions remain open. First, end-to-end autonomous infeasibility handling can be strengthened by integrating relaxation and repair strategies directly into post-training, so that the model can proactively diagnose conflict sources and generate minimal-impact relaxations. Second, the integration of formulation selection logic into model weights can be explored to enable native formulation-selection capability in LLMs. Third, the framework can be extended from single-period allocation to multi-period and network-level planning.

## Appendix A Proof of Theorem 1

Let $\Delta_{k}=I_{k}+x_{k}-T_{k}$ and $\delta_{k}=\left|\Delta_{k}\right|,k=1,\ldots,n$ for any feasible allocation $\{x_{k}\}_{k=1}^{n}$ of SD problem. Then the two inequalities in LSD, $\delta_{k}\geq\Delta_{k}$ and $\delta_{k}\geq-\Delta_{k}$ are both satisfied, so $(\{x_{k}\}_{k=1}^{n},\{\delta_{k}\}_{k=1}^{n})$ is feasible for LSD. In this construction, the objective value is exactly the same as that of SD formulation.

Conversely, for any feasible $(\{x_{k}\}_{k=1}^{n},\{\delta_{k}\}_{k=1}^{n})$ of LSD problem, the two inequalities imply $\delta_{k}\geq\left|\Delta_{k}\right|$ for each $k$. Since $\delta_{k}\geq 0$, minimizing $\sum_{k=1}^{n}\frac{1}{D_{k}}\delta_{k}$ forces, at optimum, the smallest feasible value $\delta_{k}=\left|\Delta_{k}\right|$. Therefore, the two formulations have the same optimal objective value and the same set of optimal allocation solutions.

## Declaration of competing interests

The authors declare the following financial interests/personal relationships which may be considered as potential competing interests: Yongzhi Qi reports financial support was provided by JD.com Inc. Jintao Xu, Yingzheng Ma, Jiong Dong, Jianshen Zhang, Dongyang Geng, Anni Zhang report financial support was provided by JD.com Inc. The views and opinions expressed in this paper are solely those of the authors and do not necessarily represent those of JD.com Inc.

## References

- Silver et al. (1998) Edward A. Silver, David F. Pyke, and Rein Peterson. Inventory Management and Production Planning and Scheduling. Wiley, 3 edition, 1998. ISBN 978-0471119470.

- Zipkin (2000) Paul H. Zipkin. Foundations of Inventory Management. McGraw-Hill, 2000. ISBN 978-0256113792.

- Axsäter (2015) Sven Axsäter. Inventory Control. Springer, 3 edition, 2015. ISBN 978-3319157290.

- Howard and Marklund (2011) Christian Howard and Johan Marklund. Evaluation of stock allocation policies in a divergent inventory system with shipment consolidation. European Journal of Operational Research, 211(2):298–309, 2011. doi:https://doi.org/10.1016/j.ejor.2010.11.030.

- Simchi-Levi et al. (2008) David Simchi-Levi, Philip Kaminsky, and Edith Simchi-Levi. Designing and Managing the Supply Chain: Concepts, Strategies, and Case Studies. McGraw-Hill, 3 edition, 2008. ISBN 978-0072357561.

- Ramamonjison et al. (2022) Rindranirina Ramamonjison, Timothy Yu, Raymond Li, Haley Li, Giuseppe Carenini, Bissan Ghaddar, Shiqi He, Mahdi Mostajabdaveh, Amin Banitalebi-Dehkordi, Zirui Zhou, and Yong Zhang. NL4Opt competition: Formulating optimization problems based on their natural language descriptions. In Marco Ciccone, Gustavo Stolovitzky, and Jacob Albrecht, editors, Proceedings of the NeurIPS 2022 Competitions Track, volume 220 of Proceedings of Machine Learning Research, pages 189–203. PMLR, 2022. URL https://proceedings.mlr.press/v220/ramamonjison23a.html.

- Astorga et al. (2025) Nicolás Astorga, Tennison Liu, Yuanzhang Xiao, and Mihaela Van Der Schaar. Autoformulation of mathematical optimization models using LLMs. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 1864–1886. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/astorga25a.html.

- Jiang et al. (2025) Caigao Jiang, Xiang Shu, Hong Qian, Xingyu Lu, Jun Zhou, Aimin Zhou, and Yang Yu. LLMOPT: Learning to define and solve general optimization problems from scratch. In The Thirteenth International Conference on Learning Representations (ICLR 2025), 2025. URL https://openreview.net/forum?id=9OMvtboTJg.

- Huang et al. (2025) Chenyu Huang, Zhengyang Tang, Shixi Hu, Ruoqing Jiang, Xin Zheng, Dongdong Ge, Benyou Wang, and Zizhuo Wang. ORLM: A customizable framework in training large models for automated optimization modeling. Operations Research, 73(6):2986–3009, 2025. doi:https://doi.org/10.1287/opre.2024.1233.

- Ahmaditeshnizi et al. (2024) Ali Ahmaditeshnizi, Wenzhi Gao, and Madeleine Udell. OptiMUS: Scalable optimization modeling with (MI)LP solvers and large language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 577–596. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html.

- Ouyang et al. (2022) Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc., 2022. ISBN 9781713871088.

- Bai et al. (2022) Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. Technical report, arxiv preprint, arxiv, 2022. doi: https://doi.org/10.48550/arXiv.2204.05862.

- Ethayarajh et al. (2024) Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Model alignment as prospect theoretic optimization. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 12634–12651. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ethayarajh24a.html.

- Li et al. (2025) Sirui Li, Janardhan Kulkarni, Ishai Menache, Cathy Wu, and Beibin Li. Towards foundation models for mixed integer linear programming. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=6yENDA7J4G.

- Christiano et al. (2017) Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS 2017), page 4302–4310, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964. URL https://papers.neurips.cc/paper/7017-deep-reinforcement-learning-from-human-preferences.pdf.

- Ziegler et al. (2019) Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. Technical report, arxiv preprint, arxiv, 2019. doi: https://doi.org/10.48550/arXiv.1909.08593.

- Rafailov et al. (2023) Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 53728–53741. Curran Associates, Inc., 2023. URL https://openreview.net/forum?id=HPuSIXJaa9.

- Gheshlaghi Azar et al. (2024) Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li, editors, Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 4447–4455. PMLR, 2024. URL https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html.

- Meng et al. (2024) Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 124198–124235. Curran Associates, Inc., 2024. URL https://openreview.net/forum?id=3Tzcot1LKb.

- Eskandarpour et al. (2015) Majid Eskandarpour, Pierre Dejax, Joe Miemczyk, and Olivier Péton. Sustainable supply chain network design: An optimization-oriented review. Omega, 54:11–32, 2015. doi:https://doi.org/10.1016/j.omega.2015.01.006.

- Baldacci et al. (2012) Roberto Baldacci, Aristide Mingozzi, and Roberto Roberti. Recent exact algorithms for solving the vehicle routing problem under capacity and time window constraints. European Journal of Operational Research, 218(1):1–6, 2012. doi:https://doi.org/10.1016/j.ejor.2011.07.037.

- Melo et al. (2009) M.T. Melo, S. Nickel, and F. Saldanha-da Gama. Facility location and supply chain management – A review. European Journal of Operational Research, 196(2):401–412, 2009. doi:https://doi.org/10.1016/j.ejor.2008.05.007.

- Kasirzadeh et al. (2017) Atoosa Kasirzadeh, Mohammed Saddoune, and François Soumis. Airline crew scheduling: models, algorithms, and data sets. EURO Journal on Transportation and Logistics, 6(2):111–137, 2017. doi:https://doi.org/10.1007/s13676-015-0080-x.

- Williams (2013) H. Paul Williams. Model Building in Mathematical Programming. Wiley, 5 edition, 2013. ISBN 978-1118443330.

- Kellerer et al. (2004) Hans Kellerer, Ulrich Pferschy, and David Pisinger. Knapsack Problems. Springer, 2004. ISBN 978-3540402862.

- Bertsimas and Tsitsiklis (1997) Dimitris Bertsimas and John N. Tsitsiklis. Introduction to Linear Optimization. Athena Scientific, 1997. ISBN 978-1886529199.

- Ke et al. (2017) Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: A highly efficient gradient boosting decision tree. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf.

- Loshchilov and Hutter (2019) Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR 2019), 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

- Jain et al. (2024) Neel Jain, Ping yeh Chiang, Yuxin Wen, John Kirchenbauer, Hong-Min Chu, Gowthami Somepalli, Brian R. Bartoldson, Bhavya Kailkhura, Avi Schwarzschild, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. NEFTune: Noisy embeddings improve instruction finetuning. In The Twelfth International Conference on Learning Representations (ICLR 2024), 2024. URL https://openreview.net/forum?id=0bMmZ3fkCk.

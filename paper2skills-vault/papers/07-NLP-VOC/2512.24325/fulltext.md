<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2512.24325
     paper_id : 2512.24325
     source   : paper2skills-vault/papers/07-NLP-VOC/2512.24325/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

arXiv:2512.24325v1 [cs.IR] 30 Dec 2025

MaRCA: Multi-Agent Reinforcement Learning for Dynamic Computation Allocation in Large-Scale Recommender Systems Wan Jiang

Xinyi Zang

Yudong Zhao

jiangwan1@jd.com JD.com Beijing, China

zangxinyi3@jd.com JD.com Beijing, China

zhaoyudong10@jd.com JD.com Beijing, China

Yusi Zou

Yunfei Lu

Junbo Tong

zouyusi1@jd.com JD.com Beijing, China

luyunfei1@jd.com JD.com Beijing, China

tjb21@mails.tsinghua.edu.cn Tsinghua University Beijing, China

Yang Liu

Ming Li

Jiani Shi

liuyang123@jd.com JD.com Beijing, China

liming666@jd.com JD.com Beijing, China

shijiani@jd.com JD.com Beijing, China

Xin Yang yangxin81@jd.com JD.com Beijing, China

Abstract

1

Modern recommender systems face significant computational challenges due to growing model complexity and traffic scale, making efficient computation allocation critical for maximizing business revenue. Existing approaches typically simplify multi-stage computation resource allocation, neglecting inter-stage dependencies, thus limiting global optimality. In this paper, we propose MaRCA, a multi-agent reinforcement learning framework for end-to-end computation resource allocation in large-scale recommender systems. MaRCA models the stages of a recommender system as cooperative agents, using Centralized Training with Decentralized Execution (CTDE) to optimize revenue under computation resource constraints. We introduce an AutoBucket TestBench for accurate computation cost estimation, and a Model Predictive Control (MPC)based Revenue-Cost Balancer to proactively forecast traffic loads and adjust the revenue-cost trade-off accordingly. Since its endto-end deployment in the advertising pipeline of a leading global e-commerce platform in November 2024, MaRCA has consistently handled hundreds of billions of ad requests per day and has delivered a 16.67% revenue uplift using existing computation resources.

Modern recommender systems analyze user behavior and contextual information to filter relevant items from large candidate pools and generate a ranked list of recommendations through multi-stage processing pipelines [8, 47]. These systems have become crucial infrastructure for e-commerce platforms [37].
Contemporary industrial recommender systems typically adopt a cascaded architecture comprising three stages: retrieval, preranking, and ranking [19, 46]. Each stage of recommender systems involves models of different complexity and is subject to specific computational constraints. However, most early academic research on recommender systems aims to maximize profit under the assumption of abundant computation resources [10, 43, 45], failing to address resource allocation constraints.
With the continued advancement of deep learning-driven recommender systems, the tension between computational demand and available machine resources has become increasingly significant.
In industrial recommender systems, traffic volume varies significantly across different periods [17], and the value of requests differs across media platforms and user demographics [28]. An effective computation allocation strategy should dynamically adapt to fluctuating traffic conditions while maintaining system stability and high recommendation quality within limited machine resources.
Figure 1 illustrates an overview of such a multi-stage recommender system.
To address these issues, recent efforts in Dynamic Computation Allocation (DCA) have explored two major categories of approaches:
optimization-based methods and reinforcement learning (RL)-based methods. Optimization-based methods, such as linear programming and heuristic scheduling, allocate computation resources based on predefined constraints and business rules. These methods offer

CCS Concepts • Information systems → Recommender systems; Computational advertising.

Keywords Resource Allocation, Deep Reinforcement Learning, Recommender System, Cooperative Multi-Agent Systems, Model Predictive Control

Introduction

Wan Jiang et al.

2

Related Work

Dynamic computation allocation (DCA) in large-scale recommender systems has been studied under paradigms ranging from deterministic optimization to learning-based methods.

2.1 Figure 1: Overview of the general architecture of recommender systems, highlighting the key stages and components involved in processing user requests.

fast and interpretable decision-making but often struggle to adapt to dynamic traffic fluctuations and evolving system conditions.
In contrast, RL-based methods frame resource allocation as a sequential decision-making problem, enabling flexible strategies by learning from historical data and real-time interactions. However, many reinforcement learning methods have adopted either a singleagent paradigm or centralized multi-agent architectures, which tend to overlook the localized operational constraints inherent in distributed recommender systems.
In industrial settings, key stages of the recommendation pipeline are deployed across different data centers, each operating under distinct constraints and relying on localized observations. Centralized approaches combine all decisions into a single prediction, which may overlook localized factors. To overcome these constraints, we propose MaRCA, a cooperative multi-agent framework that models each stage as an autonomous agent. Using Centralized Training with Decentralized Execution (CTDE), each agent makes decisions using local information while benefiting from coordinated global learning. This approach leverages cooperative agent collaboration to maximize business revenue under limited computation resources.
Extensive offline experiments of MaRCA and a four-week online A/B test on a major e-commerce platform demonstrate that MaRCA achieves a 16.67% increase in advertising revenue without additional computation cost (see more details in Section 4).
Our main contributions are summarized as follows:
(1) We propose a novel collaborative multi-agent framework that leverages the Adaptive Weighting Recurrent QMixer (AWRQ-Mixer), which integrates an adaptive weighting recurrent Q for sequential decision-making with a mixing network to foster cross-stage coordination. By employing CTDE, our framework achieves globally optimal resource allocation.
(2) We develop an MPC-based Revenue-Cost Balancer that optimally allocates resources by forecasting traffic fluctuations. This approach enables real-time, proactive decisionmaking, and therefore enhances system stability and efficiency under dynamic resource constraints.
(3) We design an AutoBucket TestBench for computation cost estimation (see more details in Appendix B). Our framework employs automated testing and intelligent bucketing to address the absence of explicit cost labels.

Deterministic Optimizers for Computation Allocation

Early solutions to manage computational load relied on heuristicbased elastic degradation or static rules (e.g., disabling heavy models or truncating result lists) [5, 35], while robust, they are valueagnostic and struggle with dynamic traffic.
The first value-aware solutions formulate DCA as a constrained optimization problem. DCAF [15] casts resource allocation as a 0–1 knapsack that maximizes revenue under a global computation budget. CRAS [40] augments this idea with a PID feedback controller [2], improving stability during bursts. SACA [34] simultaneously employs an elastic queue and elastic model, enabling the incorporation of different action types within a single module. However, its binary-search tuning lags behind rapid traffic shifts. GreenFlow [21] is a learning-augmented deterministic optimizer that combines reward prediction with a dynamic primal–dual solver to allocate multi-stage action chains under a global computation budget.
These works validate the feasibility of DCA but were founded on a key simplification that recommendation stages are independent and that their costs are static. Meanwhile, purely reactive control can lead to oscillations. These limit their ability to capture the complex, non-stationary inter-stage dependencies of a live recommender system.

2.2

Learning-Based Approaches for Computation Allocation

To overcome the limitations of deterministic optimizers, researchers recast DCA as a sequential decision process and applied reinforcement learning (RL). Classic RL variants such as DQN [26], DDQN [38], and DRQN [13] address high-dimensional, partially observable environments. Averaged Ensemble-DQN [3] improves the accuracy of Q-value estimation by averaging the outputs of multiple Q-networks, while REM [1] improves generalization via random value mixing, at the cost of additional noise.
A representative learning-based baseline of DCA is RL-MPCA [48], which treats the multi-stage pipeline as a weakly coupled MDP [25] and lets a single agent coordinate all stages. While representing a significant advance over deterministic methods, centralized singleagent methods do not align well with the decentralized reality of industrial recommender systems, where stages run on separate services [9, 46].
These considerations naturally lead to multi-agent reinforcement learning (MARL) formulations [12, 14, 29]. The Centralized Training with Decentralized Execution (CTDE) paradigm [30] allows agents to learn a globally coordinated policy while executing actions based on local observations. Within the CTDE paradigm, MARL methods split into centralized-critic and value-decomposition families [11].
Centralized-critic algorithms [20, 41] must model an exponential joint-action space and ingest the full global state, limiting scalability in complex industry settings. Value-decomposition approaches,

MaRCA: Multi-Agent Reinforcement Learning for Dynamic Computation Allocation in Large-Scale Recommender Systems

such as VDN [36] and QMIX [33], factorize the global value into per-agent utilities, achieving better scalability.
MARL has been applied to various areas in recommender systems, including multi-stage recommendation coordination [18, 42, 44], ad slot ranking [31], ad bidding [7, 16]. Separately, RL-based resource scheduling has been extensively explored in infrastructure management [23, 27]. However, using MARL for resource allocation within recommender systems remains underexplored. To our knowledge, MaRCA is the first fully cooperative MARL framework for end-to-end computation allocation in a recommender system, bridging the gap between RL and industrial DCA.

3 Methodology 3.1 Problem Formulation To address the challenges outlined in the previous sections, we formulate the multi-stage recommendation process as a constrained sequential decision-making problem that aims to maximize overall business revenue while adhering to strict computation resource constraints. We formally define the following:
• State Space S. At each step 𝑡, the system observes a state 𝑠𝑡 ∈ S, encapsulating user profile features, real-time traffic patterns, and resource utilization metrics.
• Action Space A. At each step 𝑡, the joint action combination is a = (𝑎 1, . . . , 𝑎𝑛 ) ∈ A. The joint action space is A = A1 × · · · × A𝑛 . The action combination a collectively specifies the decisions across multiple stages, including retrieval channels to activate, switch modules to enable, and queue truncation lengths.
• Action Value 𝑄 (𝑠𝑡 , 𝑎𝑡 ) . Given the current state 𝑠𝑡 and action 𝑎𝑡 , 𝑄 (𝑠𝑡 , 𝑎𝑡 ) denotes the expected business revenue.
• Computation Cost 𝐶 (𝑠𝑡 , 𝑎𝑡 ). 𝐶 (𝑠𝑡 , 𝑎𝑡 ) represents the computation resources required to execute the action 𝑎𝑡 given the current state 𝑠𝑡 . We define:
𝐶 (𝑠𝑡 , 𝑎𝑡 ) = 𝐶ˆ (𝑠𝑡 , 𝑎𝑡 ) + 𝑓 (𝐷𝑡 )

(1)

where 𝐶ˆ (·) is the computation cost predicted by the AutoBucket TestBench (see more details in Appendix B) and 𝐷𝑡 is the elastic degradation level mapped into equivalent computation cost through 𝑓 (·) (calibrated in isolated load tests).
• Reward 𝑅(𝑠𝑡 , a). The reward function is designed as business revenue. Unlike conventional step-wise rewards, our system can only observe business revenue after the entire action sequence for a request is completed.
Consider a batch of 𝑀 user requests indexed by 𝑖. For a request 𝑖, the system assigns a binary decision variable 𝑥𝑖,a ∈ {0, 1} indicating whether a specific action combination a is selected (𝑥𝑖,a = 1) or not (𝑥𝑖,a = 0). We impose a computation resource budget 𝐶𝑚 to limit the overall computation cost over all requests. Following [15], we formulate the constrained optimization problem as stated in Eqs.
(2)–(5).

max 𝑥𝑖,a

s.t.

𝑀 ∑︁ ∑︁

𝑥𝑖,a 𝑄 (𝑠𝑡 , a)

(2)

𝑥𝑖,a 𝐶 (𝑠𝑡 , a) ≤ 𝐶𝑚

(3)

𝑖=1 a∈ A 𝑀 ∑︁ ∑︁ 𝑖=1 a∈ A

∑︁

𝑥𝑖,a = 1,

∀ 𝑖 = 1, . . . , 𝑀

(4)

∀ 𝑖 = 1, . . . , 𝑀, a ∈ A

(5)

a∈ A

𝑥𝑖,a ∈ {0, 1},

We enforce a one-hot structure over the joint action space. For each request 𝑖, exactly one composite action is executed.
To satisfy the budget constraint in Eqs. (3)–(5) while maximizing total revenue in Eq. (2), we adopt a Lagrangian relaxation approach, as in [15]. The complete derivation is provided in Appendix A. This yields the following request-level decision rule:
a∗ = arg max (𝑄 (𝑠𝑡 , a) − 𝜆𝐶 (𝑠𝑡 , a))

(6)

a∈ A Here, a∗ represents the chosen action combination that maxi-

mizes the net benefit, measured by the revenue 𝑄 (𝑠𝑡 , a) minus the 𝜆-weighted cost 𝐶 (𝑠𝑡 , a). Through MaRCA, we can learn appropriate policy parameters and dynamically adapt 𝜆 based on real-time load.

3.2

System Design

As illustrated in Figure 2, the system follows a collaborative multiagent framework, where the AWRQ-Mixer and AutoBucket TestBench feed their computed metrics into the MPC-based Balancer, which then orchestrates the final action selection. The AWRQ-Mixer assesses the expected business revenue 𝑄 (𝑠𝑡 , a) under various states and expected action combinations. Meanwhile, the AutoBucket TestBench (see more details in Appendix B) processes trace-log data and predicted action outcomes to estimate the computation cost 𝐶 (𝑠𝑡 , a) for each action combination. Subsequently, the MPCbased revenue-cost balancer dynamically selects optimal actions by balancing 𝑄 (𝑠𝑡 , a) and 𝐶 (𝑠𝑡 , a), guided by real-time resource utilization.

3.3

Adaptive Weighting Recurrent Q-Mixer

The action value estimation module, AWRQ-Mixer, predicts the expected revenue of each request by jointly encoding user attributes, contextual information, and the inter-stage dependencies in the recommendation stages.
To illustrate why modeling these interdependencies matters, consider that the stages of a recommender pipeline are highly interdependent, with decisions made upstream directly constraining what can be achieved downstream. For example, changes in retrieval actions can alter the candidate pool and thus the final revenue, even when the ranking actions remain the same. However, due to the independent operation of these stages in separate service clusters with resource constraints [48] [6], it is necessary to model their interdependencies while maintaining their ability to independently manage computation costs.
AWRQ-Mixer meets this requirement through three innovations:
(1) an adaptive weighting recurrent Q ensemble that dynamically

Wan Jiang et al.

 2 L𝑘,𝑡 = 𝑟𝑡 + 𝛾𝑄𝑔𝑘′ (𝑜𝑡 +1, 𝑎𝑡 +1 ) − 𝑄𝑔𝑘 (𝑜𝑡 , 𝑎𝑡 )

(8)

where 𝛾 is the discount factor, and 𝑄𝑔′ (𝑜𝑡 +1, 𝑎𝑡 +1 ) is the target Qvalue for the next state-action pair, calculated by the target network.
The individual losses are normalized to compute the adaptive weight 𝜂𝑘,𝑡 for each Q-head:
L𝑘,𝑡 𝜂𝑘,𝑡 = Í𝐾 𝑘=1 L𝑘,𝑡 Finally, the agent’s Q-value is taken as the weighted sum:

Figure 2: MaRCA system architecture: multi-agent collaborative decision flow with Adaptive Weighting Recurrent Q-Mixer, AutoBucket TestBench, and MPC-Based RevenueCost Balancer.

integrates multiple Q-value estimators; (2) a variance-guided credit assignment mechanism to allocate reward among actions; and (3) a softplus-based monotonicity constraint that ensures cooperative aggregation of agent-level values. The following subsections will detail each of these innovations.
3.3.1 Adaptive Weighting Recurrent Q (AWRQ). Traditional DQN performs well in fully observable environments but is limited in partially observable environments due to incomplete state information. In such contexts, historical observations become crucial for accurately inferring the underlying state. Therefore, we employ DRQN to process sequences of observations over time:
 ℎ𝑡 = GRU 𝑜𝑡 , ℎ𝑡 −1 , 𝑄 (𝑜𝑡 , 𝑎𝑡 ) = MLP(ℎ𝑡 )
(7)
where 𝑜𝑡 is the observation, and ℎ𝑡 is the hidden state capturing historical context.
However, a single Q-value estimator in DRQN is insufficient to capture the multi-faceted, cross-stage decision process. Inspired by ensemble learning principles, we extend DRQN by introducing parallel recurrent Q-value estimators to address uncertainty in the estimation process. Each agent instantiates a recurrent ensemble of 𝐾 heads. For each head 𝑘 ∈ {1, . . . , 𝐾 }, it outputs 𝑄𝜃𝑘𝑔 (𝑠𝑡 , 𝑎𝑡 ), where 𝑔 ∈ {1, . . . , 𝑛} indexes the agents. For brevity, 𝑄𝜃𝑔 is henceforth written as 𝑄𝑔 whenever parameters are clear from context. Rather than averaging, we dynamically weight ensemble outputs according to their temporal difference (TD) errors, and we call this method Adaptive Weighting (AW).
At each training step, each Q-head’s TD error is recorded. If a head exhibits a larger error on a given mini-batch, it is assigned a correspondingly larger weight among the heads. This design intentionally emphasises under-performing heads, ensuring they receive greater focus during training and can be corrected more quickly.
Empirically, this weighting scheme accelerates convergence and enhances robustness.
Consider a mini-batch in which each sample contains the state 𝑠𝑡 , the action 𝑎𝑡 , and the next state 𝑠𝑡 +1 . The loss for each Q-head is then computed as follows:

𝑄𝑔 =

𝐾 ∑︁

𝜂𝑘,𝑡 𝑄𝑔𝑘 (𝑜𝑡 , 𝑎𝑡 )

(9)

(10)

𝑘=1

3.3.2 Softplus-Based Monotonicity Constraints (SMC) for Cooperative Agents. In multi-agent recommendation pipelines, the joint action-value 𝑄 tot must be non-decreasing in each agent’s value 𝑄𝑔 to reflect their cooperative contribution. To enforce this, we define a mixing network M that aggregates individual Q-values 𝑄 1, 𝑄 2, . . . , 𝑄𝑛 from multiple agents into a joint Q-value 𝑄 tot :
𝑄𝑡𝑜𝑡 = M (𝑄 1, 𝑄 2, . . . , 𝑄𝑛 , 𝑠𝑡 )

(11)

The parameters of the mixing network are generated by a hypernetwork ℎ𝜓 by taking the state 𝑠𝑡 as input:
( 𝑊˜ 1, 𝑏 1, 𝑊˜ 2, 𝑏 2 ) = ℎ𝜓 (𝑠𝑡 )

(12)

Because the stages cooperate, the joint value must be monotonic non-decreasing in every agent’s value. We enforce this by applying the Softplus transform to ensure non-negativity of all weight matrices and thus ensure 𝑄 tot is a monotonic function of each 𝑄𝑔 .
Softplus(𝑊 ) = ln(1 + 𝑒𝑊 )
𝑊𝑖 = Softplus(𝑊˜ 𝑖 ),

𝑖 ∈ {1, 2}

(13)
(14)

The mixing network is trained by minimizing L (𝜃 tot ) = (𝑟𝑡 + 𝛾𝑄 tot ′ (𝑠𝑡 +1, a′ ) − 𝑄 tot (𝑠𝑡 , a)) 2

(15)

3.3.3 Variance-Guided Credit Assignment ( VGCA). In environments with sparse or delayed rewards, such as large-scale recommendation systems, it’s crucial to determine how each action contributed to the final reward. To address this, we introduce an auxiliary reward signal that mitigates sparse rewards and enhances training stability. The key insight is using variance across candidate actions to guide their contribution to the final reward.
  𝑤𝑡 = Var 𝑗 ∈ A𝑡 E[ 𝑅 | 𝑎𝑡 = 𝑗 ]

(16)

where 𝑟𝑡 is the reward and A𝑡 is the discrete action space of 𝑎𝑡 . During TD updates, the reward 𝑟𝑡 of each action is scaled by 𝑤𝑡 , amplifying learning signals for high-impact dimensions. This adjustment ensures that agents whose actions induce greater variance, and therefore have a greater potential impact on revenue, receive proportionally stronger learning signals. This auxiliary signal serves as an additional guide to the model during training, helping it better identify and prioritize the most relevant actions,

MaRCA: Multi-Agent Reinforcement Learning for Dynamic Computation Allocation in Large-Scale Recommender Systems

even in the absence of frequent immediate feedback. Algorithm 1 summarizes the overall training procedure for AWRQ-Mixer.
Bringing these elements together, AWRQ-Mixer extends a DRQN backbone with (1) Adaptive Weighting Recurrent Q, (2) VarianceGuided Credit Assignment, and (3) Softplus-Based Monotonicity Constraints. As illustrated in Figure 3, the framework models the recommendation stages as cooperative agents through AWRQ. The mixing network aggregates Q-values from AWRQ while enforcing monotonicity, guided by state information 𝑠𝑡 to dynamically tailor mixing weights using a hypernet. At training time, we adopt centralized optimization of all agents, allowing us to capture global dependencies. At inference time, each agent can operate independently in a separate cluster, thus enabling scalable decentralized execution without extra cross-agent communication overhead. This design is particularly critical in production environments where stages such as retrieval and ranking often run in physically separate machine clusters.

and RL-MPCA [48] utilize feedback-based methods to adjust 𝜆 for dynamic adaptation.
However, such feedback-based control inherently incurs onestep latency in responding to traffic fluctuations, leading to oscillatory resource misallocation when compensating for sudden traffic changes. To address this, we design a Model Predictive Control (MPC) [24] framework that performs rolling-horizon optimization.
By continuously predicting incoming traffic patterns and precomputing optimal 𝜆 trajectories, our approach enables proactive stabilization.
MPC optimizes computation allocation over a finite time horizon 𝑁 by continuously solving an optimization problem that balances business performance, latency constraints, and hardware efficiency.
The optimization process aims to ensure that the system computation resource utilization remains close to the computation budget and minimizes fluctuations to maintain stability.
The optimization goal at time 𝑡 is defined as:

Algorithm 1 Offline Training of AWRQ-Mixer Input: Dataset D, #iterations 𝐼 , mini-batch size 𝑁 , agent set G, ensemble heads 𝐾, discount 𝛾, hypernet ℎ𝜓 , target update frequency 𝜏.
1: Initialize AWRQ {𝑄𝑔 }𝑔∈ G (each with 𝐾 heads) and targets 𝑄𝑔′ for each agent, initialize mixing network hypernet ℎ𝜓 .
2: for iter = 1 to 𝐼 do Sample a mini-batch B of transitions {(𝑠𝑡 , 𝑜𝑡 , 𝑎𝑡 , 𝑅, 𝑠𝑡 +1 )} 3:
from D.
4:
for t=1,. . . ,T do   5:
∀𝑡: 𝑤𝑡 = Var 𝑗 ∈ A𝑡 E[ 𝑅 | 𝑎𝑡 = 𝑗 ]
// VGCA, Eq. (16)
6:
𝑟𝑡 = 𝑤𝑡 𝑅 7:
end for 8:
for 𝑡 = 1 to 𝑇 do 9:
AWRQ outputs 𝐾 Q-values 𝑄𝑔𝑘 (𝑜𝑡 , 𝑎𝑡 ).
10:
From Eq. (8) and Eq. (16):
 2 L𝑘,𝑡 = 𝑟𝑡 + 𝛾 𝑄𝑔𝑘′ (𝑜𝑡 +1, 𝑎𝑡 +1 ) − 𝑄𝑔𝑘 (𝑜𝑡 , 𝑎𝑡 )
Í𝐾 11:
𝜂𝑘,𝑡 = L𝑘,𝑡 / 𝑘=1 L𝑘,𝑡 // AW, Eq. (9)
Í𝐾 12:
𝑄𝑔 = 𝑘=1 𝜂𝑘,𝑡 𝑄𝑔𝑘 (𝑜𝑡 , 𝑎𝑡 )
13:
𝜃𝑔 ← arg min𝜃 g L𝑘,𝑡 14:
𝜃𝑔′ ← 𝜃𝑔 if iter mod 𝜏 = 0 15:
end for  16:
(𝑊1, 𝑏 1,𝑊2, 𝑏 2 ) ← ℎ𝜓 𝑠𝑡 , a with 𝑊1 = softplus(·), 𝑊2 = softplus(·)
// SMC, Eq. (14)
 17:
𝑄 tot = ReLU [𝑄 1, . . . , 𝑄𝑛 ] 𝑊1 + 𝑏 1 𝑊2 + 𝑏 2 18:
Ltot = (𝑅 + 𝛾𝑄 tot′ (𝑠𝑡 +1, a′ ) − 𝑄 tot (𝑠𝑡 , a)) 2 19:
𝜓 ← arg min𝜓 Ltot 20: end for Output: Trained {𝜃𝑔 } and 𝜓 .

𝐽 =

𝜆𝑡 +𝑖 0≤𝑖 ≤𝑁 −1

MPC-Based Revenue-Cost Balancer

The MPC-Based Revenue-Cost Balancer determines the optimal a∗ that maximizes the expected reward in a given state 𝑠 while keeping the computation resource utilization within the computation budget 𝐶𝑚 . In prior works such as DCAF [15] and SACA [34], binary search is employed to determine 𝜆. Moreover, CRAS [40]

𝑁 ∑︁

𝛼 𝑖 𝐶ˆ𝑡 +𝑖 − 𝐶𝑚

2

𝑖=0

( +

(17)
if 𝐶ˆ𝑡 +𝑖 < 𝐶𝑚 ,

0, 𝛽

Í𝑁

𝑖=0 𝛼

𝑖

2 𝐶ˆ𝑡 +𝑖 − 𝐶ˆ𝑡 +𝑖 −1 ,

otherwise.

s.t. 𝐶ˆ𝑡 +𝑖+1 = 𝑔(𝐶ˆ𝑡 +𝑖 , 𝑠𝑡 +𝑖 , 𝜆𝑡 +𝑖 )
0≤𝑖 ≤𝑁 −1

(18)

Here 𝛼 𝑖 ∈ (0, 1] is a decay weighting factor that reduces the weight of longer-term predictions, thus mitigating long-horizon errors. 𝛽 is an oscillation damping factor penalizing abrupt computation resource utilization changes. 𝑔(·) is a learned system model mapping computation resource states 𝐶ˆ𝑡 +𝑖 , environment states 𝑠𝑡 +𝑖 , and the revenue-cost balancer 𝜆𝑡 +𝑖 to future CPU loads.
By optimizing 𝐽 , this objective ensures that computation resource utilization near 𝐶𝑚 while reducing fluctuations. Only the first element of the computed optimal sequence {𝜆𝑡∗, ..., 𝜆𝑡∗+𝑁 −1 } is applied at time 𝑡, ensuring proactive and stable resource allocation.

4 Experiments 4.1 Offline Experiments Our evaluation focuses on two core modules: the AWRQ-Mixer and the MPC-based Revenue–Cost Balancer. We evaluate the AWRQMixer using both model metrics and simulated revenue derived from real-world logs, demonstrating its prediction accuracy and revenue uplift. For the MPC-based Balancer, we introduce utilization and overutilization rates as evaluation metrics to quantify its scheduling capability.
4.1.1

3.4

min

AWRQ-Mixer Experiment Results.

Implementation details. The AWRQ-Mixer relies on several hyperparameters. We used grid search [4] to identify the optimal hyperparameter values. A list of these parameters and their values is provided in Appendix D.
Model Evaluation Metrics. To quantify the monotonic relationship between the model’s predicted rankings and the true rankings,

Wan Jiang et al.

Figure 3: Adaptive Weighting Recurrent Q-Mixer (AWRQ-Mixer) Architecture in MaRCA.
we use Spearman’s Rank Correlation (𝑟𝑠 ):

Return Percentage:
Í

cov(R(𝑋 ), R(𝑌 ))
𝑟𝑠 = 𝜎R(𝑋 ) 𝜎R(𝑌 )

(19)

where cov(R(𝑋 ), R(𝑌 )) denotes the covariance between the ranks of variables 𝑋 and 𝑌 , and 𝜎rank(𝑋 ) , 𝜎rank(𝑌 ) are the corresponding standard deviations. Higher values of 𝑟𝑠 indicate stronger positive correlations, reflecting high agreement between the predicted and actual orderings.
Training Stability Metrics. To quantify optimisation stability we report two extra metrics:
• Convergence. The number of environment steps (in millions) required for a model to reach 95% of its final validation Return% (averaged over 5 random seeds). A smaller value indicates faster sample efficiency.
• Gradient-variance. 𝑔𝑡 = ∥∇𝜃 L𝑡 ∥ 2 be the L2-norm of the joint-Q network gradient. For a sliding window of 1000 updates, we compute the variance and then average across all windows and seeds. Lower variance implies smoother gradient flow and more stable learning.
Revenue Simulation. Evaluating new models in a live environment is often risky and resource-intensive. To enable fair and controlled comparisons under consistent computation budgets, we simulate revenue through four steps:
• Ground Truth Estimation. Train an ensemble model on historical data, including train and test data, to approximate true revenue (with 𝑟𝑠 = 0.95).
• Uniform Computation Cost. Derive action distributions from the test data that reflect the total computation cost, and use them as the action quota.
• Action Allocation. Train the baseline models on the training dataset, and compute the action values 𝑄ˆ (𝑠, a) for all actionstate pairs in the test dataset. Sort all (𝑠, a) pairs by ˆ 𝑄, then iteratively assign the highest-valued action without exceeding the action quota.
• Revenue Evaluation. Calculate total expected revenue using the ensemble model and assess performance using Relative

Return% = Í

𝑄 model (𝑠, 𝑎)
× 100% 𝑄 ground_truth (𝑠, 𝑎)

(20)

where 𝑄 model and 𝑄 ground_truth represent predicted revenues from experiment and ground_truth models respectively.
We verified that Return% and 𝑟𝑠 correlate strongly (r = 0.93)
and both have the same ordering as online Revenue, which shows the offline metrics have strong validity.
Baselines. We compare AWRQ-Mixer against several baseline models.
• DQN [26]: Employs a single deep network to approximate Q-values in high-dimensional state spaces.
• DRQN [13]: Extends DQN with a recurrent mechanism to handle partial observability.
• DDQN [38]: Decouples action selection from evaluation to mitigate Q-value overestimation.
• Averaged Ensemble-DQN [3]: Aggregates multiple Q-networks to reduce variance and enhance stability.
• REM [1]: Combines Q-networks through random convex mixtures for richer exploration.
• VDN [36]: Decomposes the global Q-value into a sum of per-agent Q-values for cooperative policies.
• QMIX [33]: Employs a mixing network to ensure monotonic relationships among individual Q-values, thus improving coordination efficiency in multi-agent systems.
• Weighted QMIX [32]: Uses dynamic weighting in QMIX to emphasize heterogeneous agent contributions.
Experiment Results. We comprehensively analyze the realworld logs from the display advertising system. As shown in Table 1, AWRQ-Mixer achieves significant improvements on both evaluation metrics. Compared to REM, the single-agent model used in RL-MPCA [48], AWRQ-Mixer boosts 𝑟𝑠 by 3.4% and Return% by 8.0%. Multi-agent methods generally outperform single-agent ones, underlining the benefits of collaborative modeling.
Ablation Study. We assess the contributions of three key components in AWRQ-Mixer by removing each one separately and comparing these variants to the full model. Table 2 presents the results. The full model attains the highest 𝑟𝑠 (0.911) and Return%

MaRCA: Multi-Agent Reinforcement Learning for Dynamic Computation Allocation in Large-Scale Recommender Systems

Table 1: Comparison of offline evaluation results across different baseline models.
𝑟𝑠

Return% 𝜈=

Single-Agent DQN DDQN DRQN Averaged Ensemble-DQN REM (RL-MPCA)

0.859(±0.025)
0.860(±0.019)
0.862(±0.023)
0.870(±0.015)
0.881(±0.014)

82.59(±4.22)
85.46(±3.18)
86.82(±3.67)
87.48(±2.79)
89.47(±2.63)

Multi-Agent VDN Weighted QMIX QMIX AWRQ-Mixer (MaRCA)

0.896(±0.018)
0.900(±0.015)
0.902(±0.017)
0.911(±0.009)

95.10(±1.94)
95.65(±1.68)
96.00(±1.63)
97.26(±1.01)

𝑇 1 ∑︁ max(𝐶ˆ𝑡 , 𝐶𝑚 ) − 𝐶𝑚 𝑇 𝑖=1 𝐶𝑚

(22)

Balancing 𝜇 and 𝜈 is crucial for ensuring efficiency and stability.
While increasing 𝜇 raises the risk of exceeding limits, reducing 𝜈 leads to conservative resource utilization.
Offline Experiment Results. We compared the performance of the feedback-based and MPC-based revenue-cost balancers. Table 3 shows that the MPC-based approach not only improves overall computation resource usage but also substantially lowers the risk of exceeding the computation budget.

(97.26%), while converging in the fewest environment steps (1.1 M)
and exhibiting the lowest gradient variance (0.18).
• AW. Removing AW decreases 𝑟𝑠 and Return% only marginally (–0.1% and –0.09%), while the gradient variance triples (0.54 vs 0.18) and the number of steps to reach the same validation loss increases by 27%. The larger variance indicates that, without the TD-error–driven head re-weighting, noisy or mis-calibrated heads exert disproportionate influence, producing erratic updates and slowing convergence.
• SMC. Replacing Softplus with an absolute-value operator leads to higher gradient variance (0.41) and a 20% slower convergence, accompanied by a further drop in both 𝑟𝑠 and Return%. Although absolute value enforces monotonicity, its discontinuity at zero amplifies gradient fluctuations around the decision boundary. The measured variance confirms this analytic expectation and explains the observed degradation.
• VGCA. Eliminating VGCA yields the largest decline in ranking quality (–0.51% 𝑟𝑠 ) and offline return (–0.55%), together with the slowest convergence (1.7 M steps). By reallocating TD-errors according to action variance, VGCA accelerates credit propagation in sparse-reward regions. Without VGCA, the agents require more samples to achieve the same validation criterion, even though the raw gradient variance remains moderate (0.27).
Across all metrics, each component contributes additively. Their combination yields a consistently more stable, accurate, and fasterconverging learner.
4.1.2

• Overutilization Rate (𝜈): This metric measures the risk of exceeding the computation budget. A lower 𝜈 indicates fewer stability violations and reduced risk of service degradation.

Hyperparameter Analysis. We investigate three key hyperparameters: decay-weighting factor 𝛼, oscillation damping factor 𝛽, and prediction horizon 𝑁 .
• Decay-Weighting Factor 𝛼. A small 𝛼 prioritizes current returns, which can lead to more aggressive resource utilization but can intensify fluctuations. Conversely, a large 𝛼 may cause underutilization by overemphasizing future risks. As shown in Figure 4a, the overutilization rate stabilizes around 𝛼 = 0.4, indicating balanced short/long-term trade-offs. We choose 𝛼 = 0.4 as the parameter value.

(a) Decay-Weighting Factor 𝛼

(b) Oscillation Damping Factor 𝛽

MPC-based Revenue-Cost Balancer Experiment Results.

Evaluation Metrics. We introduce two complementary metrics that capture distinct aspects of computation allocation: utilization rate 𝜇 and overutilization rate 𝜈.
• Utilization Rate (𝜇): This metric quantifies the effective usage of available resources. A higher 𝜇 indicates better computation resource utilization within capacity limits.
1 ∑︁ min(𝐶ˆ𝑡 , 𝐶𝑚 )
𝑇 𝑖=1 𝐶𝑚 𝑇

𝜇=

(21)

(c) Prediction Horizon 𝑁

Figure 4: Hyperparameter analysis results for 𝛼, 𝛽 and 𝑁 .
• Oscillation Damping Factor 𝛽. 𝛽 is employed to limit the fluctuation of computation resource utilization. When 𝛽 is too high, the system becomes overly conservative, lowering utilization. Conversely, an extremely small 𝛽 can result in unstable spikes. Based on the results in Figure 4b, we choose

Wan Jiang et al.

Table 2: Ablation study results demonstrating the impact of core innovations: adaptive weighting (AW), softplus-based monotonic constraints (SMC), and variance-guided credit assignment (VGCA).

AWRQ-Mixer (MaRCA)
AWRQ-Mixer w/o AW AWRQ-Mixer w/o SMC AWRQ-Mixer w/o VGCA

𝑟𝑠

Return%

Convergence (M steps)

Gradient-variance

0.911 (±0.009)
0.910 (±0.010)
0.908 (±0.009)
0.906 (±0.011)

97.26 (±1.01)
97.17 (±1.10)
96.92 (±1.18)
96.71 (±1.37)

1.1 (±0.10)
1.4 (±0.20)
1.3 (±0.15)
1.7 (±0.20)

0.18 (±0.05)
0.54 (±0.19)
0.41 (±0.12)
0.27 (±0.08)

Table 4: Online A/B test results comparing Static, RL-MPCA, MaRCA (Feedback-Based), and MaRCA (MPC-Based).

Static DCAF RL-MPCA MaRCA-Feedback MaRCA-MPC

Revenue +0.00% +3.67%(±0.20%)
+12.16%(±0.28%)
+14.93%(±0.43%)
+16.67%(±0.24%)

GMV +0.00% +6.16%(±3.68%)
+13.78%(±4.03%)
+15.65%(±5.39%)
+18.18%(±3.95%)

Impressions +0.00% +4.92%(±0.04%)
+9.07%(±0.03%)
+11.67%(±0.04%)
+14.24%(±0.03%)

Table 3: Comparison of load utilization and overutilization rates between feedback-based and MPC-based revenue-cost balancer.
Method Feedback-based MPC-based

Utilization Rate 92.33%(±0.40%)
95.29% (±0.23%)

Overutilization Rate 2.91%(±0.43%)
0.64%(±0.10%)

𝛽 = 8 to achieve a trade-off between resource utilization and stability.
• Prediction Horizon N. While a larger 𝑁 often improves longterm optimization, it simultaneously increases forecasting errors and latency overhead. Conversely, a smaller 𝑁 may fail to capture future dynamics. Figure 4c indicates that 𝑁 = 10 provides an effective balance between predictive accuracy and runtime efficiency.

4.2

Online A/B test Results

We conducted a four-week online A/B test comparing five strategies: a static method, DCAF [15], RL-MPCA [48], MaRCA with a feedback-based Revenue-Cost Balancer, and MaRCA with an MPC-based balancer. The static approach uses fixed allocation rules through stress testing and practical experience, with predefined downgrades to manage traffic spikes. Table 4 reports impressions, clicks, revenue, gross merchandise volume (GMV), return on investment (ROI), click-through rate (CTR), and cost per mille (CPM), with revenue and GMV as our primary metrics. Here, ROI is defined as ROI = GMV/Spend, where Spend denotes advertiser spend (i.e., the platform’s revenue in our setting). Our near-real-time deployment adds virtually no additional latency.
MaRCA achieved statistically significant improvements across all key metrics while operating within existing computation resource constraints. After the four-week A/B test, MaRCA was rolled out to 100% of production traffic, where it now processes hundreds of billions of requests each day and has since supported multiple

Clicks +0.00% +5.38%(±0.09%)
+15.37%(±0.10%)
+17.79%(±0.13%)
+19.51%(±0.07%)

ROI +0.00% +2.40%(±3.69%)
+0.55%(±4.04%)
+0.64%(±5.41%)
+1.29%(±3.96%)

CTR +0.00% +0.69%(±0.10%)
+4.85%(±0.10%)
+5.58%(±0.14%)
+5.22%(±0.08%)

large-scale sales events. Its stability has also mitigated the need for continuous on-call support.

5

Conclusion

In this paper, we propose MaRCA to address the challenge of maximizing business revenue in large-scale recommender systems under computation resource constraints. By modeling recommendation stages as cooperative agents and integrating Centralized Training with Decentralized Execution, MaRCA effectively captures crossstage dependencies while preserving independent decision-making.
Additionally, we introduce an MPC-based revenue-cost balancer that proactively adjusts resource allocation, ensuring system stability under dynamic traffic conditions. Our extensive offline experiments and large-scale online deployment demonstrate that MaRCA significantly improves business revenue, achieving a 16.67% revenue increase with no additional computation resource. Future work may focus on enhancing model capabilities, expanding the action space, and exploring cross-domain applications.

References [1] Rishabh Agarwal, Dale Schuurmans, and Mohammad Norouzi. 2020. An Optimistic Perspective on Offline Reinforcement Learning. In Proceedings of the 37th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 119), Hal Daumé III and Aarti Singh (Eds.). PMLR, 104–114.
https://proceedings.mlr.press/v119/agarwal20c.html [2] Kiam Heong Ang, G. Chong, and Yun Li. 2005. PID control system analysis, design, and technology. IEEE Transactions on Control Systems Technology 13, 4 (2005), 559–576. doi:10.1109/TCST.2005.847331 [3] Oron Anschel, Nir Baram, and Nahum Shimkin. 2017. Averaged-DQN: variance reduction and stabilization for deep reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning - Volume 70 (Sydney, NSW, Australia) (ICML’17). JMLR.org, 176–185.
[4] James Bergstra and Yoshua Bengio. 2012. Random search for hyper-parameter optimization. J. Mach. Learn. Res. 13, null (Feb. 2012), 281–305.
[5] Betsy Beyer, Chris Jones, Jennifer Petoff, and Niall Richard Murphy. 2016. Site Reliability Engineering: How Google Runs Production Systems (1st ed.). O’Reilly Media, Inc.
[6] Vanessa Cai, Pradeep Prabakar, Manuel Serrano Rebuelta, Lucas Rosen, Federico Monti, Katarzyna Janocha, Tomo Lazovich, Jeetu Raj, Yedendra Shrinivasan, Hao Li, and Thomas Markovich. 2023. TwERC: High Performance Ensembled Candidate Generation for Ads Recommendation at Twitter. In Proceedings of the Workshop on Data Mining for Online Advertising (AdKDD 2023) (CEUR Workshop Proceedings, Vol. 3556), Abraham Bagherjeiran, Nemanja Djuric, Kuang-Chih Lee,

MaRCA: Multi-Agent Reinforcement Learning for Dynamic Computation Allocation in Large-Scale Recommender Systems

Linsey Pang, Vladan Radosavljevic, and Suju Rajan (Eds.). CEUR-WS.org, Aachen, Germany. https://ceur-ws.org/Vol-3556/adkdd23-cai-twerc-ceur-paper.pdf [7] Chen Chen, Gao Wang, Baoyu Liu, Siyao Song, Keming Mao, Shiyu Yu, and Jingyu Liu. 2025. Real-time bidding with multi-agent reinforcement learning in multi-channel display advertising. Neural Comput. Appl. 37, 1 (January 2025), 499–511. https://doi.org/10.1007/s00521-024-10649-6 [8] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah.
2016. Wide & Deep Learning for Recommender Systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems (Boston, MA, USA)
(DLRS 2016). Association for Computing Machinery, New York, NY, USA, 7–10.
doi:10.1145/2988450.2988454 [9] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems (Boston, Massachusetts, USA) (RecSys ’16). Association for Computing Machinery, New York, NY, USA, 191–198. doi:10.1145/2959100.
2959190 [10] Yang Deng, Yaliang Li, Fei Sun, Bolin Ding, and Wai Lam. 2021. Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, Canada) (SIGIR ’21). Association for Computing Machinery, New York, NY, USA, 1431–1441.
doi:10.1145/3404835.3462913 [11] Mohamad A. Hady, Siyi Hu, Mahardhika Pratama, Zehong Cao, and Ryszard Kowalczyk. 2025. Multi-agent reinforcement learning for resources allocation optimization: a survey. Artificial Intelligence Review 58, 11 (2025), 354. doi:10.
1007/s10462-025-11340-5 [12] Jianye Hao, Tianpei Yang, Hongyao Tang, Chenjia Bai, Jinyi Liu, Zhaopeng Meng, Peng Liu, and Zhen Wang. 2024. Exploration in Deep Reinforcement Learning:
From Single-Agent to Multiagent Domain. IEEE Transactions on Neural Networks and Learning Systems 35, 7 (July 2024), 8762–8782. doi:10.1109/tnnls.2023.3236361 [13] Matthew Hausknecht and Peter Stone. 2015. Deep recurrent q-learning for partially observable mdps. In 2015 aaai fall symposium series.
[14] Dom Huh and Prasant Mohapatra. 2024. Multi-agent Reinforcement Learning:
A Comprehensive Survey. arXiv:2312.10256 [cs.MA] https://arxiv.org/abs/2312.
10256 [15] Biye Jiang, Pengye Zhang, Rihan Chen, Binding Dai, Xinchen Luo, Yin Yang, Guan Wang, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2020. DCAF: A Dynamic Computation Allocation Framework for Online Serving System. In Proceedings of the 2nd Workshop on Deep Learning Practice for High-Dimensional Sparse Data with KDD 2020 (DLP-KDD’20). Association for Computing Machinery, San Diego, CA, USA. Best Paper Runner-Up.
[16] Junqi Jin, Chengru Song, Han Li, Kun Gai, Jun Wang, and Weinan Zhang. 2018.
Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management (CIKM ’18). ACM, 2193–2201. doi:10.1145/3269206.
3272021 [17] Yehuda Koren. 2009. Collaborative filtering with temporal dynamics. Proceedings of the 15th 53, 447–456. doi:10.1145/1557019.1557072 [18] Kaiyuan Li, Pengfei Wang, and Chenliang Li. 2022. Multi-Agent RL-based Information Selection Model for Sequential Recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (Madrid, Spain) (SIGIR ’22). Association for Computing Machinery, New York, NY, USA, 1622–1631. doi:10.1145/3477495.3532022 [19] Shichen Liu, Fei Xiao, Wenwu Ou, and Luo Si. 2017. Cascade Ranking for Operational E-commerce Search. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’17). ACM.
doi:10.1145/3097983.3098011 [20] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch.
2017. Multi-agent actor-critic for mixed cooperative-competitive environments.
In Proceedings of the 31st International Conference on Neural Information Processing Systems (Long Beach, California, USA) (NIPS’17). Curran Associates Inc., Red Hook, NY, USA, 6382–6393.
[21] Xingyu Lu, Zhining Liu, Yanchu Guan, Hongxuan Zhang, Chenyi Zhuang, Wenqi Ma, Yize Tan, Jinjie Gu, and Guannan Zhang. 2023. GreenFlow: a computation allocation framework for building environmentally sound recommendation system. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (Macao, P.R.China) (IJCAI ’23). Article 677, 9 pages.
doi:10.24963/ijcai.2023/677 [22] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018.
Modeling Task Relationships in Multi-task Learning with Multi-gate Mixtureof-Experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (London, United Kingdom) (KDD ’18).
Association for Computing Machinery, New York, NY, USA, 1930–1939. doi:10.
1145/3219819.3220007 [23] Hongzi Mao, Mohammad Alizadeh, Ishai Menache, and Srikanth Kandula. 2016.
Resource Management with Deep Reinforcement Learning. In Proceedings of the

15th ACM Workshop on Hot Topics in Networks (Atlanta, GA, USA) (HotNets ’16).
Association for Computing Machinery, New York, NY, USA, 50–56. doi:10.1145/ 3005745.3005750 [24] D.Q. Mayne, J.B. Rawlings, C.V. Rao, and P.O.M. Scokaert. 2000. Constrained model predictive control: Stability and optimality. Automatica 36, 6 (2000), 789–814. doi:10.1016/S0005-1098(99)00214-9 [25] Nicolas Meuleau, Milos Hauskrecht, Kee-Eung Kim, Leonid Peshkin, Leslie Pack Kaelbling, Thomas Dean, and Craig Boutilier. 1998. Solving very large weakly coupled Markov decision processes. In Proceedings of the Fifteenth National/Tenth Conference on Artificial Intelligence/Innovative Applications of Artificial Intelligence (Madison, Wisconsin, USA) (AAAI ’98/IAAI ’98). American Association for Artificial Intelligence, USA, 165–172.
[26] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. 2015. Human-level control through deep reinforcement learning.
nature 518, 7540 (2015), 529–533.
[27] Navid Naderializadeh, Jaroslaw J. Sydir, Meryem Simsek, and Hosein Nikopour.
2021. Resource Management in Wireless Networks via Multi-Agent Deep Reinforcement Learning. IEEE Transactions on Wireless Communications 20, 6 (June 2021), 3507–3523. doi:10.1109/twc.2021.3051163 [28] Nicola Neophytou, Bhaskar Mitra, and Catherine Stinson. 2022. Revisiting Popularity and Demographic Biases in Recommender Evaluation and Effectiveness. In Advances in Information Retrieval: 44th European Conference on IR Research, ECIR 2022, Stavanger, Norway, April 10–14, 2022, Proceedings, Part I (Stavanger, Norway).
Springer-Verlag, Berlin, Heidelberg, 641–654. doi:10.1007/978-3-030-99736-6_43 [29] Zepeng Ning and Lihua Xie. 2024. A survey on multi-agent reinforcement learning and its application. Journal of Automation and Intelligence (02 2024).
doi:10.1016/j.jai.2024.02.003 [30] Zhou Qin, Kai Yuan, Pratik Lahiri, and Wenyang Liu. 2024. Cooperative MultiAgent Deep Reinforcement Learning In Content Ranking Optimization. In Proceedings of the ACM SIGIR Workshop on eCommerce 2024 co-located with the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024), Washington D.C., USA, July 18, 2024 (CEUR Workshop Proceedings, Vol. 3843), Surya Kallumadi, Yubin Kim, Tracy Holloway King, Maarten de Rijke, and Vamsi Salaka (Eds.). CEUR-WS.org. https://ceurws.org/Vol-3843/paper_15.pdf [31] Zhou Qin, Kai Yuan, Pratik Lahiri, and Wenyang Liu. 2024. Cooperative Multi-Agent Deep Reinforcement Learning in Content Ranking Optimization.
arXiv:2408.04251 [cs.LG] https://arxiv.org/abs/2408.04251 [32] Tabish Rashid, Gregory Farquhar, Bei Peng, and Shimon Whiteson. 2020.
Weighted QMIX: expanding monotonic value function factorisation for deep multi-agent reinforcement learning. In Proceedings of the 34th International Conference on Neural Information Processing Systems (Vancouver, BC, Canada) (NIPS ’20). Curran Associates Inc., Red Hook, NY, USA, Article 855, 12 pages.
[33] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. 2020. Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. Journal of Machine Learning Research 21, 178 (2020), 1–51. http://jmlr.org/papers/v21/20-081.html [34] Shunhui, Jiahong, Songwei, Guoliang, Qianlong, and Lebin. 2021. SingleAction Computation Allocation. https://tech.meituan.com/2021/06/17/waimaiai-advertisement.html.
[35] Santhosh Kumar Somarapu. 2024. Autoscaling Strategies for Stateful Stream Operators under Bursty Workloads. International Journal of Communication Networks and Information Security (IJCNIS) 16, 1 (Jan. 2024), 428–448. https:
//ijcnis.org/index.php/ijcnis/article/view/8368 [36] Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore Graepel. 2018. Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (Stockholm, Sweden) (AAMAS ’18). International Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 2085–2087.
[37] Farah Tawfiq, Abdul Monem Rahma, and Hala Abdul wahab. 2021. Recommendation Systems For E-commerce Systems An Overview. Journal of Physics:
Conference Series 1897 (05 2021), 012024. doi:10.1088/1742-6596/1897/1/012024 [38] Hado van Hasselt, Arthur Guez, and David Silver. 2016. Deep Reinforcement Learning with Double Q-Learning. Proceedings of the AAAI Conference on Artificial Intelligence 30, 1 (Mar. 2016). doi:10.1609/aaai.v30i1.10295 [39] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. In Proceedings of the ADKDD’17 (Halifax, NS, Canada)
(ADKDD’17). Association for Computing Machinery, New York, NY, USA, Article 12, 7 pages. doi:10.1145/3124749.3124754 [40] Xun Yang, Yunli Wang, Cheng Chen, Qing Tan, Chuan Yu, Jian Xu, and Xiaoqiang Zhu. 2021. Computation Resource Allocation Solution in Recommender Systems.
arXiv:2103.02259 [eess.SY] https://arxiv.org/abs/2103.02259 [41] Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. 2022. The surprising effectiveness of PPO in cooperative multi-agent games. In Proceedings of the 36th International Conference on Neural Information

Wan Jiang et al.

Processing Systems (New Orleans, LA, USA) (NIPS ’22). Curran Associates Inc., Red Hook, NY, USA, Article 1787, 14 pages.
[42] Gengrui Zhang, Yao Wang, Xiaoshuang Chen, Hongyi Qian, Kaiqiao Zhan, and Ben Wang. 2024. UNEX-RL: reinforcing long-term rewards in multi-stage recommender systems with unidirectional execution. In Proceedings of the ThirtyEighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence (AAAI’24/IAAI’24/EAAI’24). AAAI Press, Article 1035, 9 pages. doi:10.1609/aaai.v38i8.28783 [43] Xiangyu Zhao, Changsheng Gu, Haoshenglun Zhang, Xiwang Yang, Xiaobing Liu, Jiliang Tang, and Hui Liu. 2021. DEAR: Deep Reinforcement Learning for Online Advertising Impression in Recommender Systems. Proceedings of the AAAI Conference on Artificial Intelligence 35, 1 (May 2021), 750–758. doi:10.1609/ aaai.v35i1.16156 [44] Xiangyu Zhao, Long Xia, Lixin Zou, Hui Liu, Dawei Yin, and Jiliang Tang.
2020. Whole-Chain Recommendations. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM ’20). ACM, 1883–1891. doi:10.1145/3340531.3412044 [45] Xiangyu Zhao, Xudong Zheng, Xiwang Yang, Xiaobing Liu, and Jiliang Tang.
2020. Jointly Learning to Recommend and Advertise. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (Virtual Event, CA, USA) (KDD ’20). Association for Computing Machinery, New York, NY, USA, 3319–3327. doi:10.1145/3394486.3403384 [46] Zhishan Zhao, Jingyue Gao, Yu Zhang, Shuguang Han, Siyuan Lou, Xiang-Rong Sheng, Zhe Wang, Han Zhu, Yuning Jiang, Jian Xu, and Bo Zheng. 2023. COPR:
Consistency-Oriented Pre-Ranking for Online Advertising. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (Birmingham, United Kingdom) (CIKM ’23). Association for Computing Machinery, New York, NY, USA, 4974–4980. doi:10.1145/3583780.3615465 [47] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for ClickThrough Rate Prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (London, United Kingdom)
(KDD ’18). Association for Computing Machinery, New York, NY, USA, 1059–1068.
doi:10.1145/3219819.3219823 [48] Jiahong Zhou, Shunhui Mao, Guoliang Yang, Bo Tang, Qianlong Xie, Lebin Lin, Xingxing Wang, and Dong Wang. 2023. RL-MPCA: A Reinforcement Learning Based Multi-Phase Computation Allocation Approach for Recommender Systems.
In Proceedings of the ACM Web Conference 2023 (WWW ’23). ACM, 3214–3224.
doi:10.1145/3543507.3583313

MaRCA: Multi-Agent Reinforcement Learning for Dynamic Computation Allocation in Large-Scale Recommender Systems

A

Lagrangian Relaxation Details

To satisfy the budget constraint in Eqs. (3)–(5) while maximizing total revenue in Eq. (2), we adopt a Lagrangian relaxation approach:

L=

𝑀 ∑︁ ∑︁

𝑥𝑖,a 𝑄 (𝑠𝑡 , a) − 𝜆𝑡

𝑖=1 a∈ A

𝑀 ∑︁ ∑︁

(𝑥𝑖,a 𝐶 (𝑠𝑡 , a) − 𝐶𝑚 ))

(23)

𝑖=1 a∈ A

Where Revenue-Cost Balancer (𝜆𝑡 ) serves as a trade-off parameter balancing business revenue and computation cost. The constant term 𝜆𝑡 𝐶𝑚 can be omitted in the maximization as it does not affect the optimization. Substituting the Lagrangian utility in Eq. (23) back into the original objective of Eq. (2) absorbs the cost constraint into the multiplier 𝜆𝑡 . The global problem becomes:
max 𝑥𝑖,a

𝑀 ∑︁ ∑︁

𝑥𝑖,a 𝑄 (𝑠𝑡 , a) − 𝜆𝑡 𝐶 (𝑠𝑡 , a)



(24)

Figure 5: AutoBucket TestBench enables multi-stage computation cost estimation through traffic simulation, regression analysis, and sequence-aware modeling.

𝑖=1 a∈ A

Because 𝑥𝑖,a is one-hot for each request, the double sum decomposes into independent per-request arg-max operations. Following the approach in [48], each user request effectively solves a local decision subproblem:
a∗ = arg max (𝑄 (𝑠𝑡 , a) − 𝜆𝐶 (𝑠𝑡 , a))
a∈ A

to measure computation cost, and then fit a regression model.
(1) AutoBucket. Simulation traffic is grouped into buckets based on request value and queue length to capture the variability in computation cost across diverse traffic conditions.
(2) TestBench. The TestBench processes requests in each bucket at a fixed queries-per-second (QPS). The TestBench is deployed on 𝑛 machines, each with 𝑁 cores computation resource cores, and the computation resource utilization 𝑝% is recorded during these tests.
The computation cost per request for each bucket is computed as:

(25)

which is Eq. (6) in the main text.

B

AutoBucket TestBench

Accurately estimating computation costs is essential for resource allocation in large-scale recommender systems. However, real-world cost labels are scarce, and the variability in action results across multiple stages complicates direct cost measurement at the request level. We propose a two-phase approach that predicts action results and derives their associated costs via cost mapping. This method leverages the observation that computation costs remain consistent for identical action results, even when the underlying requests differ. Since computation cost is primarily dictated by code logic, model complexity, and retrieval processes.
Figure 5 presents the core architecture of our AutoBucket TestBench. Traffic logs from multiple sources are processed and subjected to simulated load testing, during which computation resource utilization is recorded. The resulting labeled samples (e.g., queue length vs. cost) are aggregated and used to fit a regression model, producing a cost-mapping function. Combined with predicted action outcomes, this module enables accurate computation cost estimations.
Action Results Prediction. Given user and contextual features, we employ a DCN [39]+MMoE [22] hybrid architecture to predict action results. The Deep & Cross Network (DCN) captures nonlinear feature interactions, while the Multi-gate Mixture of Experts (MMoE) layer enables parameter sharing among predictions. This multi-task learning framework operates within the 𝑆 × 𝐴 space, leveraging abundant logged interaction data for supervised training.
Computation Cost Estimation. With predicted action results, we estimate the corresponding computation cost via a cost mapping.
• Elastic Queue. For actions determining queue truncation lengths, we use empirical tests under varied queue lengths

Computation Cost per Bucket =

𝑝% × 𝑛 × 𝑁 cores 𝑄𝑃𝑆

(26)

A monotonic regression model is then used to fit the relationship between queue lengths and computation cost.
• Elastic Model. For actions that involve switching between different models, computation costs are derived from independent measurements of each switch configuration. Similar to Elastic Queue, Elastic Models are tested under controlled conditions to measure their computation cost.
• Elastic Channel. The computation cost is derived as the cumulative sum of the computation costs across all selected actions.

C

Hyperparameters Sensitivity Analysis

We conduct an offline sensitivity analysis for two key hyperparameters in AWRQ-Mixer: discount factor 𝛾 (see Table 4) and the number of ensemble heads 𝐾 (see Table 5). Each configuration is evaluated over 5 random seeds.
• Discount Factor 𝛾. We selected a 𝛾 value of 0.9. In reinforcement learning, 𝛾 represents the trade-off between long-term and immediate rewards, determining how much importance the model places on future rewards. From Table 4, we observe that when 𝛾 is set to 0.90, the model exhibits the highest stability and performance. In contrast, other values

Wan Jiang et al.

Table 6: Hyperparameter settings for MaRCA.
Hyperparameters

Value

AWRQ-Mixer Learning rate GRU hidden size Discount factor 𝛾 Target-network update frequency 𝜏 Ensemble size 𝐾 𝜀-greedy exploration rate Size of hidden layer in the network Optimizer Dropout rate Weight initializer Batch size

0.01 256 0.9 100 20 0.05 [512, 256]
Adam 0.2 glorot uniform 2048

MPC-based Revenue-Cost Balancer Decay-Weighting Factor 𝛼 Oscillation Damping Factor 𝛽 Prediction Horizon 𝑁

0.4 8 10

• Number of ensemble heads 𝐾. Table 5 illustrates the effects of different ensemble head numbers on performance. As the value of 𝐾 increases, we see a gradual improvement in both 𝑟𝑠 and Return%. Specifically, when 𝐾 is set to 200, the 𝑟𝑠 value reaches 0.9120, with a Return% of 97.30. However, while 𝐾 = 200 yields the best performance, increasing the number of ensemble heads significantly boosts computational complexity. After considering the trade-off between computational resources and model effectiveness, we selected 𝐾 = 20, which provided nearly optimal results while minimizing unnecessary computational overhead.
Table 5: Ensemble Head 𝐾 Sensitivity 𝐾

𝑟𝑠

Return%

1 5 20 100 200

0.9065(±0.0095)
0.9081(±0.0089)
0.9112(±0.0086)
0.9118(±0.0087)
0.9120(±0.0085)

96.78(±1.11)
96.95(±1.02)
97.26(±1.01)
97.29(±1.01)
97.30(±1.00)

Algorithm 3 Online Serving of MaRCA Input: Trained networks {𝑄𝑔 }, cost estimator 𝐶 (·), MPC RevenueCost Balancer that outputs 𝜆𝑡 1: if periodic update moment, 𝜆𝑡 ← MPC Revenue-Cost Balancer 2: for each incoming request do 3:
Observe state 𝑠𝑡 4:
a∗ ← arg maxa∈ A (𝑄 (𝑠𝑡 , a) − 𝜆𝐶 (𝑠𝑡 , a))
5:
Execute joint action a∗ 6: end for

D

Hyperparameters

The following Table 6 lists the hyperparameters we used in experiments.

E

Online Serving of AWRQ–Mixer

Algorithm 2 shows AWRQ–Mixer’s online serving process for a given request.
Algorithm 2 Online Serving of AWRQ–Mixer

such as 0.70 and 0.99, show slightly lower results. Therefore, we ultimately chose 0.9 for 𝛾 to ensure a balance of strong performance and system stability.
Table 4: Discount Factor 𝛾 Sensitivity 𝛾

𝑟𝑠

Return%

0.70 0.80 0.90 0.99

0.9096(±0.0076)
0.9107(±0.0083)
0.9112(±0.0086)
0.9105(±0.0109)

97.18(±0.97)
97.22(±1.01)
97.26(±1.01)
97.21(±1.05)

Input: Trained networks {𝑄𝑔𝑘 }.
1: for each incoming request do 2:
Observe state features 𝑜𝑡 3:
AWRQ outputs 𝐾 Q-values 𝑄𝑔𝑘 (𝑜𝑡 , 𝑎𝑡 )
Í𝐾 𝑄𝑔𝑘 4:
𝑄𝑔 ← 𝐾1 𝑘=1 5: end for

F

Online Serving of MaRCA

Algorithm 3 shows MaRCA’s online serving process for a given request.


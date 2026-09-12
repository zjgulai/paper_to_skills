<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2202.03867
     paper_id : 2202.03867
     source   : https://arxiv.org/html/2202.03867v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Offline Reinforcement Learning for Mobile Notifications

Yiping Yuan Affiliation: LinkedIn Corporation
Mountain View, CA, USA
ypyuan@linkedin.com    Ajith Muralidharan Affiliation: LinkedIn Corporation
Mountain View, CA, USA
amuralidharan@linkedin.com    Preetam Nandy Affiliation: LinkedIn Corporation
Mountain View, CA, USA
pnandy@linkedin.com    Miao Cheng Affiliation: LinkedIn Corporation
Mountain View, CA, USA
miacheng@linkedin.com    Prakruthi Prabhakar Affiliation: LinkedIn Corporation
Mountain View, CA, USA
paprabhakar@linkedin.com Affiliation:

###### Abstract

Mobile notification systems have taken a major role in driving and maintaining user engagement for online platforms. They are interesting recommender systems to machine learning practitioners with more sequential and long-term feedback considerations. Most machine learning applications in notification systems are built around response-prediction models, trying to attribute both short-term impact and long-term impact to a notification decision. However, a user’s experience depends on a sequence of notifications and attributing impact to a single notification is not always accurate, if not impossible. In this paper, we argue that reinforcement learning is a better framework for notification systems in terms of performance and iteration speed. We propose an offline reinforcement learning framework to optimize sequential notification decisions for driving user engagement. We describe a state-marginalized importance sampling policy evaluation approach, which can be used to evaluate the policy offline and tune learning hyperparameters. Through simulations that approximate the notifications ecosystem, we demonstrate the performance and benefits of the offline evaluation approach as a part of the reinforcement learning modeling approach. Finally, we collect data through online exploration in the production system, train an offline Double Deep Q-Network and launch a successful policy online. We also discuss the practical considerations and results obtained by deploying these policies for a large-scale recommendation system use-case.

###### Index Terms:

Reinforcement learning, offline evaluation, Mobile notifications

## I Introduction

As online services and applications provide more and more content and functionality, communications with users are increasingly crucial for them to keep users informed and engaged. Mobile notifications are a major channel that services use to highlight important and timely content to the users. With the right content at the right time, notifications can inform users of important activity and bring more value to users. Since users have limited attention, notifications can help remind users of important values that they would need to be aware of to increase engagement with the platform. There are mainly two categories of responses from sending a notification: content engagement responses (e.g., clicks, dismisses) and site engagement responses (e.g., user visits, notification disables). Typical recommender systems usually care more about the content engagement responses, while for notification systems, site engagement responses are as important if not more. Unlike content engagement responses, user engagement may not be attributed to a single notification, but rather a sequence of notifications, presenting an attribution challenge for modeling. Another challenge for modeling site engagement responses is that the short-term (within a few hours) impact and long-term (over a week or longer) impact may diverge. Although intrusive and frequent notifications can bring users back to site, they could create notification fatigue or cause notification disablement, which hurts user engagement in the long run [1, 2]. These unique challenges give rise to an interesting application area to machine learning practitioners with more sequential and long-term considerations.

Most notification systems [3, 4, 5, 6, 7] are built around response prediction models. To overcome the attribution challenge of the site engagement responses, a state-transition model is proposed to predict the additional user visits attributed to a single notification in [5]. The volume optimization framework proposed in [7] attributes site engagement responses to a weekly notification count rather than to a single notification. These response predictions are then compared with optimal thresholds from online or offline threshold search based on multi-objective optimizations [8]. While such systems have demonstrated good empirical performance over CTR-based systems, they could be sub-optimal in decision making. First, the attribution is still approximate and cannot fully capture the sequential impact. For example, the volume optimization framework in [7] assumes the site engagement response only depends on the volume sent to a user without considering the spacing of notification deliveries under the same volume. Secondly, the online or offline threshold tuning is often heuristic, and may not achieve the optimality defined by the multi-objective optimization. Practically, response prediction model improvement (e.g., in terms of offline AUC) may not necessarily lead to better online performance when couple with the threshold tuning. While there are efforts to automate and optimize this tuning [9], it could slow down model iteration by weeks from trained response models to a fully deployed system. And the model iteration speed is a very important consideration to real world systems that need to be improved and updated constantly.

In comparison, reinforcement learning is a principled approach to optimize for a sequence of well-coordinated notification decisions with respect to the defined objective. The attribution challenge comes down to the definition of the rewards. If rewards from the environment are defined properly, the aggregated rewards (the total return) will be consistent with the business objective. We explain our reward definition in Section III-B. Reinforcement learning is a superior framework to emphasize long-term impact with the aggregated rewards over a long or infinite horizon. Moreover, the offline reinforcement learning framework we propose comes with efficient offline policy evaluation, which provide consistency between offline evaluation and online performance. It could also avoid costly online tuning and speed up model iteration as described in Section IV-B and Section V.

Applying reinforcement learning to a large-scale online system faces several challenges. Online reinforcement learning training with online exploration may not be feasible due to high infrastructure costs, unknown time to converge, and unbounded risks of deteriorating user experience. The fact that a large proportion of reinforcement learning algorithms and research are more focused towards online learning paradigm is also one of the biggest obstacles to their widespread adoption[10]. Alternatively, offline reinforcement learning [11, 12, 13, 14, 15, 10, 16] has started to draw more research attention in recent years due to its well-controlled risk and a smoother fit into existing machine learning infrastructure. There are theoretical and practical challenges in efficient offline policy learning, and accurate offline policy evaluation [17, 18, 19, 20] due to the notorious Deadly Triad problem (i.e., the problem of instability and divergence arising when combining function approximation, bootstrapping and offline training) [21, 22]. Compared with classical control problems, the low signal-to-noise ratio and potential non-linearity in user behavior require thoughtful Markov Decision Process (MDP) formulation, adequate function approximation, and exploratory behavior policy design to learn effective policies offline. Additionally, it is important to have a reliable offline off-policy evaluation algorithm to ensure safe and efficient policy iterations [20].

In this paper, we propose an offline reinforcement learning approach to optimize for site engagement in notification systems. We summarize our contribution

-

We formulate a MDP to model the notification system and specify an offline learning approach based on the (Double) Deep Q-Network.

-

We propose a state-marginalized importance sampling algorithm for offline evaluation to reduce the high variance of the existing importance sampling based algorithms.

-

We evaluate our approach using a simplified simulation setup that helps mimic the online process we optimize. We use this simulation environment to validate and benchmark offline evaluation methods.

-

We present a real-world fully-deployed application to demonstrate how such a reinforcement learning paradigm can improve site user engagement and achieve better performance than the supervised approach.

The rest of this paper is organized as follows. Section II reviews related work. In Section III, we introduce the problem of notification delivery time optimization and its Markov Decision Process formulation. Section IV introduces the underlying methodology for offline training, offline evaluation and how we build up a simulation environment to mimic the real-world application. Section V carries out both simulated and real-world experiments to demonstrate how the proposed framework works. Finally, Section VI concludes this work and discusses our future work.

## II Related Work

Early work in [3, 4] proposed a volume optimization framework based on supervised model predictions. The framework was originally designed for emails and was later extended to notification applications [6, 7] with more considerations on real-time relevance and machine learning infrastructures. Other related work [23, 24, 25] focused on improving the prediction with supervised learning. A survival-based state-transition model [5] was proposed to drive user engagement through mobile notifications with heuristic global trade-offs between short-term and long-term. A bandit-based solution [26] was proposed to improve long-term user engagement in a recommender system. These approaches worked well in practice but could be suboptimal in sequential decision making. We argue in this paper that a reinforcement learning framework is a better fit to notification systems.

In terms of notification fatigue and unnecessary interruptions, several notification systems [27, 28, 29, 30] have been proposed for detecting opportune timings for delivery of notifications, leveraging various types of sensing and machine learning technologies. Such an adaptive notification system was evaluated in the product environment and showed impressive click-through rate increase when powered by supervised machine learning [31, 2]. We introduce a similar production system at LinkedIn in Section III-A, which we call it “Notification Spacing System”. A state-of-art survival model [5] is used in this system as the supervised benchmark for our proposed reinforcement learning. This survival model was the best performing production model at LinkedIn before we introduce this work. In this paper, we illustrate how reinforcement learning can be applied to such large-scale production systems and is compared against the supervised benchmark.

Recently, Chen et al. [13] applied a Policy Gradient learning in YouTube recommender system with Off-policy correction for offline reinforcement learning. Ie et al. [14] proposed an offline reinforcement learning for slate-based recommender systems, where the large action space can become intractable for many reinforcement learning algorithms. Zou et al. [32] designed an offline framework to learn a Q-network and a separate S-network from a simulated environment to assist the Q-network. In comparison, our simulated environment is only used for validation with ground truth, and the deployed policy takes no information from the simulated environment to avoid unknown bias. While there are a lot of efforts to apply reinforcement learning to real-world applications [33, 34, 35], driving long-term engagement through notification systems presents its unique challenges and opportunities for reinforcement learning due to its sequential planning and short-term long-term trade-off.

Offline evaluation is another crucial research area for real-world applications. Various importance sampling methods [19, 17, 18, 36] have been applied to correct the mismatch in the distributions under the behavior policy and evaluated policy. While these importance sampling based estimators are either unbiased or have little bias, the variances of them tend to be high for a long sequence. A marginalized importance sampling [20] was recently proposed to reduce variances for long-horizon MDP environments, with extensive theoretical and empirical studies. We choose this method for the offline evaluation and apply some practical modifications, namely a novel dimension reduction technique and discretization.

## III Notification delivery time optimization

Online platforms use mobile notifications to communicate timely, important, and actionable content to users. Some notifications, due to their nature, user expectation, or product constraints, need to be delivered in near real-time. Examples of these include content notifications described in [6] and notifications about messages sent to users. There are other notifications, which are not time-sensitive, and they would be relevant if they are delivered within a predefined time window. Examples of these notifications include events from your network, such as your colleague’s work anniversary and birthday, or aggregate notifications about activities that you may be interested in. Figure 1 gives such an example of work anniversary notification. Such time-insensitive notifications provide more opportunities for online platforms to optimize for site engagement through delivery time optimization. In this paper, we focus our discussions on applying reinforcement learning to such time-insensitive notifications to determine the best delivery times towards long-term engagement. We describe such a notification interaction environment through the notification spacing system at LinkedIn. This system ensures that users do not receive all the notifications at the same time but receive them over the course of multiple days or weeks, providing users a holistic and engaging experience over time.

*Fig. 1: An example of time-insensitive notifications *

### III-A Notification Spacing System

The notification spacing system in Figure 2 consists of a queuing system (one for each user), into which time-insensitive notification candidates for the user are queued. At fixed time intervals, we choose whether to send the top-ranked notification in the queue. Additionally, at each such time step, there may be notifications that would expire after that time. The notification spacing system also determines which of these expiring notifications to send to the user and which ones to drop.

*Fig. 2: Illustration of the notification spacing system *

We now briefly describe the baseline policy in this system, which is based on the work presented in [5] (Section 5.2). This is the supervised approach that delivers the best empirical performance against other supervised approaches at LinkedIn. An Accelerated Failure Time survival model is trained using the user interaction data. The model is then used to predict 1) the probability of a user’s visit within the next $T$ time after a notification delivery, denoted as $P_{T}(\text{visit}|\text{send})$; 2) the probability of a user’s organic visit within the next $T$ time without a notification delivery, denoted as $P_{T}(\text{visit}|\text{not send})$. This baseline model chooses to send the top ranked notification if

$\frac{P_{T}(\text{visit}|\text{send})-P_{T}(\text{visit}|\text{not send})}{P_{T}(\text{visit}|\text{not send})}>\tau,$ | | | | (1) |

where $\tau$ is the threshold to heuristically control the trade-off between short-term and long-term rewards. $P_{T}(\text{visit}|\text{send})-P_{T}(\text{visit}|\text{not send})$ is the uplift estimate of the short-term impact on the user engagement by a notification delivery. The extra denominator $P_{T}(\text{visit}|\text{not send})$ helps to normalize with respect to a user’s activity level. The policy makes a send decision when the short-term uplift exceeding the threshold $\tau$. Note that $\tau=0$ leads to a greedy action, that is, as long as the short-term uplift is positive, a notification will be delivered to a user, which will almost surely result in annoying notification experience and hurt the long-term site engagement. A policy with a larger $\tau$ will withhold some notifications in the hope that there will be more opportune moment in the future. One drawback of this approach is that $\tau$ is hyper-parameter outside of the supervised learning, thus cannot be learned with the model training. Instead, $\tau$ is usually tuned through grid search using online A/B test. This is the case for supervised approaches for notification systems in general, since there is often a gap between the model’s prediction and the optimal notification decision.

### III-B Markov Decision Process for Notification Spacing

In this section, we formulate the notification delivery time optimization in the notification spacing system as a Markov Decision Process (MDP), represented by $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma)$, where $\mathcal{S}$ is the environment’s state space, $\mathcal{A}$ is the action space, $\mathcal{P}:\mathcal{S}\times\mathcal{A}\rightarrow\mathcal{S}$ is the state transition model, $\mathcal{R}:\mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$ is the reward function and $\gamma$ is the discount factor of the cumulative reward. Reinforcement learning systems learn the optimal action given the state to maximize a objective defined as a cumulative discounted reward over time. In a typical setting, an agent receives the environment’s state and uses it to choose an action based on its policy. In response, the system makes a transition to a new state and provides the reward, and the cycle is repeated. The problem is then to learn an optimal policy for the agent to maximize the total reward over a finite or infinite time horizon in the future.

The key concepts for the notification decision problem are described below.

Actions. $a$ denotes an action in the action space $\mathcal{A}$. We consider a discrete action space consisting of two actions - SEND (send the notification candidate to the user) and NOT-SEND (the notification candidate is put back in the notification queue for further considerations). Note that every notification candidate has its validity time window ranging from a few hours to a few days. Once a notification candidate reaches its expiry time, it will be either sent or dropped based on its quality. Since this is controlled by an independent logic, it is abstracted out as part of the environment.

States. $s$ denotes a state in the state space $\mathcal{S}$. A state represents a situation in the environment and summarizes all useful historical information. A state $s_{t}$ at time $t$, has the Markov property, if and only if,

$P(s_{t+1}|s_{1},s_{2},..,s_{t})=P(s_{t+1}|s_{t}).$ | | | |

Since MDP is the foundation for reinforcement learning algorithms, states must be defined properly to ensure and Markovian system. In our problem setting, we use a plethora of features, including

-

user’s profile features such as locale and network size.

-

dynamic state features such as badge count, number of notifications in the queue, number of notifications sent in the past 24 hours.

-

user’s activity features such as user’s last visit time, the number of site visits over the past week.

In this manner, we allow the users to be part of the environment and represent their interests and context using a rich state representation.

Environment. In standard reinforcement learning, an agent interacts with an environment over a number of discrete time steps. At every time step $t$, the agent receives a state $s_{t}$ and chooses an action $a_{t}$. In return, the agent receives the next state $s_{t+1}$ and a scalar reward $r_{t}$. In our problem, the environment is made up of all users’ interests and interactions. A single episode corresponds to a sampled user and their interaction sequence. It consists of all the time steps at which the agent evaluates whether to send the top-ranked notification in the queue over a finite time horizon.

Reward. $r_{t}$ denotes an immediate reward collected between time $t$ and $t+1$. In this paper, we use a user visit to the platform within the next time step as a reward. The total return $R_{t}=\sum_{k=t}^{\infty}\gamma^{k}r_{k}$ represents the time-discounted total number of site visits by a user. Here, $\gamma\in(0,1]$ is the discount factor, which controls the trade-offs between the short-term and long-term rewards. The goal of the agent is to maximize this total return to encourage long-term site engagement. The reward can also be defined as notification clicks, or notification disables as negative rewards or a linear combination of them.

Policy. A policy $\pi$ is a mapping from the state space to the action space. In this setting, it makes SEND or NOT-SEND decisions given the state features. A policy can be either deterministic or stochastic. For every Markov decision process, there exists an optimal deterministic policy $\pi^{*}$, which maximizes the total return from any initial state.

## IV Methodology

We now introduce our offline training framework and offline evaluation method. We also show how setting up a simulated environment can help validate offline training and offline evaluation.

### IV-A Offline Training

One main challenge in applying reinforcement learning to any real-world online recommender systems is the high cost of exploration, which would harm the user experience. That is, if we train a reinforcement learning agent in an online fashion [37], our users can suffer from an exploratory yet bad policy, which we try to avoid.

Offline Reinforcement Learning, also known as Batch Reinforcement Learning [11] in literature, is a variant of reinforcement learning that the agent learns from a fixed batch of data [10]. This variant is suitable for large-scale user-platform interactive applications due to its control over exploration risks, and it naturally fits into existing machine learning infrastructures compared to online reinforcement learning. Our proposed offline solution is a combination of Offline Deep Q-Network (DQN) and data collection with well-controlled online exploration.

Q-learning algorithms are good candidates for offline reinforcement learning attributable to their off-policy nature, that is, they can learn the value of the optimal policy independently of the agent’s actions. On the contrary, on-policy learner learns the value of the policy being carried out by the agent, including the exploration steps. While most on-policy algorithms can have their off-policy versions, they require non-trivial importance sampling to adjust the off-policy bias. Importance sampling based estimation can be of very high variance, especially in an offline paradigm.

Among Q-learning algorithms, Deep Q-Network is using Deep Neural Network that takes a state and approximates Q-values for each action based on that state, which has been proved to be successful in tackling high-dimensional state space and gain wide popularity in the recent industry and research advancements [37, 38]. Therefore, we choose Offline Deep Q-Network algorithms for our use-case to learn from the data generated by the complex user behaviors that happened in the real world.

The Q-value function which takes two inputs state ($s$) and action ($a$) under policy $\pi$ is defined as

$\displaystyle Q^{\pi}(s_{t},a_{t})$ $\displaystyle{}={}$ $\displaystyle E_{\pi}[R_{t}|s_{t}=s,a_{t}=a]$ | | | | | | (2) |

$\displaystyle{}={}$ $\displaystyle E_{\pi}[{\sum_{k=0}^{\infty}\gamma^{k}}r_{t+k+1}|s_{t}=s,a_{t}=a],$ | | | | | |

where $\gamma$ is the discount factor, at each time step $t$ the agent with state $s_{t}$ selects an action $a_{t}$, observes a reward $r_{t}$, and $R_{t}$ is the cumulative long-term reward.

We define the optimal Q-value function

$Q^{{\pi}^{*}}(s_{t},a_{t})=\max_{\pi}E_{\pi}(R_{t}|s_{t}=s,a_{t}=a),$ | | | | (3) |

as the maximum expected return achievable across all policies. An optimal policy is easily derived from the optimal $Q^{{\pi}^{*}}(s,a)$ by selecting the highest-valued action in each state. This optimal $Q^{{\pi}^{*}}(s,a)$ obeys the Bellman equation:

$Q^{{\pi}^{*}}(s,a)=E\left(r+\gamma\max_{a^{\prime}}Q^{{\pi}^{*}}(s^{\prime},a^{\prime})|s,a\right).$ | | | | (4) |

The Deep Q-Network (DQN) learns a parameterized action-value function $Q(s,a;\boldsymbol{\theta})$ as a neural network. From Equation 4, the Q-Network can be trained by minimizing the following loss function:

$\displaystyle L(\boldsymbol{\theta})$ $\displaystyle=$ $\displaystyle E_{s_{t},a_{t},r_{t},s_{t+1}~\mathcal{B}}\left((y_{t}-Q(s,a;\boldsymbol{\theta}))^{2}\right),$ | | | | | | (5) |

$\displaystyle y_{t}$ $\displaystyle=$ $\displaystyle r_{t}+\gamma\max_{a}Q\left(s_{t+1},a;\boldsymbol{\theta}_{t}\right),$ | | | | | |

where $\mathcal{B}$ is an offline data batch which contains transition tuples of $\{s_{t},a_{t},r_{t},s_{t+1}\}$. The function approximation $\boldsymbol{\theta}$ can be designed according to characteristics of the application. In this work, we use a fully-connected network, which proved to be sufficient to capture the user interactions. Practically, a few techniques can be employed to improve the learning performance, such as setting up a separate target network [37], tuning hyper-parameters, using more advanced DQN variants (for example, Double DQN [38] and dueling DQN [39]). Our presented results are based on Double DQN, in which a second network $\boldsymbol{\theta}_{t}^{-}$ is introduced to stabilize the target action-value estimation and reduce the over-estimation well known for the vanilla DQN. The Double DQN can be trained by minimizing the following loss function:

$\displaystyle L(\boldsymbol{\theta})$ $\displaystyle{}={}$ $\displaystyle E_{s_{t},a_{t},r_{t},s_{t+1}~\mathcal{B}}\left((y_{t}^{\mathrm{DoubleQ}}-Q(s,a;\boldsymbol{\theta}))^{2}\right),$ | | | | | |

$\displaystyle y_{t}^{\mathrm{DoubleQ}}$ $\displaystyle{}={}$ $\displaystyle r_{t}+\gamma Q\left(s_{t+1},\operatorname{argmax}_{a}Q\left(s_{t+1},a;\boldsymbol{\theta}_{t}\right),\boldsymbol{\theta}_{t}^{-}\right).$ | | | | | |

These DQN variants were originally designed for online off-policy learning [37]. If we only look at the training algorithm, the offline version above is almost the same except that the mini-batch is sampled from an offline data batch instead of an experience buffer in [37]. However, the fundamental difference is that the offline training has no control over the behavior policy. Nor can it explore unseen action trajectories in the batch through interactions with the environment, which is safe for our scenario (to protect user experience). Blending with exploration is outside the scope of the algorithm we are using, as it is both theoretically and practically challenging to guarantee the performance of a learned policy.

Fortunately, we have the control over how the offline data are collected to certain extent, that is, we need to ensure enough exploration in the offline data. We deploy an $\epsilon-$greedy exploration strategy [40] on the baseline policy described in Section III-A,

$\pi^{\epsilon}(s)=\begin{cases}\pi_{0}(s)&\text{with probability }1-\epsilon,\\
a\in\text{Unif}(A)&\text{with probability }\epsilon,\end{cases}$ | | | | (6) |

where $\pi_{0}$ is the baseline policy, $\text{Unif}(A)$ is a uniform distribution over all possible actions and $\epsilon$ controls the exploration rate.

Not only is the exploration in data collection important to the error bound of offline learning, it is also critical to the offline policy evaluation, that

-

Data used for offline policy evaluation has to be collected from a stochastic policy with non-zero probability coverage on the state-action space.

-

The action probabilities have to be correctly recorded.

We will elaborate the discussion in the next section.

### IV-B Offline Evaluation

It is crucial to evaluate the performance of the learned policy before risking deployment. Furthermore, we often have more than one algorithm and corresponding hyper-parameter settings, making the offline evaluation an indispensable component of a reinforcement learning training pipeline. The offline evaluation of a reinforcement learning agent requires the estimation of a counterfactual metric of interest from the data collected from an arbitrary but known policy. Model-based approaches for evaluating a reinforcement learning agent from such off-policy data can induce large bias, while the classical importance weighting based non-parametric methods tend to exhibit a high variance for long-term evaluations. We propose a class of importance weighting based methods that can be tuned to obtain a desirable bias-variance trade-off depending on the environment. To this end, we first describe the importance weighting strategy.

Given $N$ i.i.d. trajectory observations from time $1$ to $T$, $\{(s_{1,i},a_{1,i},r_{1,i},\ldots,s_{T,i},a_{T,i},r_{T,i})\}_{i=1}^{N}$ based on a policy $\pi$ from the joint distribution $p_{\pi}(\cdot)$, we aim to estimate the total expected reward $\theta(\pi^{*})=\sum_{t=1}^{T}E_{\pi^{*}}[r_{t}]$ corresponding to a target policy $\pi^{*}$. A estimator of $\theta(\pi^{*})$ is given by

$\hat{\theta}(\pi^{*})=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}r_{t,i}~w_{t,i}$ | | | |

where $w_{t,i}$ denotes the importance weights adjusted for the mismatch in the distributions under policies $\pi$ and $\pi^{*}$.

Now it is easy to show that $\hat{\theta}(\pi^{*})$ is an unbiased estimator of $\theta(\pi^{*})$ for

$\displaystyle w_{t,i}=\frac{p_{\pi^{*}}(s_{t,i},a_{t,i})}{p_{\pi}(s_{t,i},a_{t,i})},$ | | | | (7) |

given that the data generating policy is a stochastic policy $\pi(a\mid s)>0$ for all $a\in\mathcal{A}$ and $s\in\mathcal{S}$.

In most cases, the functional form of the distributions $p_{\pi}(\cdot)$ and $p_{\pi^{*}}(\cdot)$ are unknown and hence $w_{t,i}$ needs to be computed/estimated from the data. There are two main obstacles in constructing reliable estimates of $\theta(\pi^{*})$ based on $w_{t,i}$: (i) the curse of dimensionality of the state space and (ii) the curse of the horizon. While the former is a well-known problem in supervised learning, the latter is tied to reinforcement learning problems with a long time horizon $T$.

Action Trajectory Based Weighting [17]: The following estimator avoids the curse of dimensionality of the state space by factorizing $p_{\pi}(\cdot)$ and $p_{\pi^{*}}(\cdot)$ using the Markov property.

$w_{t,i}=\frac{p_{\pi^{*}}(s_{1,i})~\prod_{j=1}^{t}\pi^{*}(a_{j,i}\mid s_{j,i})}{p_{\pi}(s_{1,i})~\prod_{j=1}^{t}\pi(a_{j,i}\mid s_{j,i})}=\frac{\prod_{j=1}^{t}\pi^{*}(a_{j,i}\mid s_{j,i})}{\prod_{j=1}^{t}\pi(a_{j,i}\mid s_{j,i})},$ | | | |

where the last equality follows from the fact that the distribution of $S_{1}$ does not depend on the underlying policy. The corresponding estimator of $\theta(\pi^{*})$ assign weights to each sample $(s_{1},a_{1},\ldots,s_{t},a_{t})$ according to the probability of observing the action trajectory $(a_{1},\ldots,a_{t})$ under $\pi^{*}$. Thus, we refer to this method as Action Trajectory Based Weighting. Note that if $\pi^{*}$ is a deterministic policy, i.e. $\pi^{*}(s|a)\in\{0,1\}$, then this method would assign zero weights to all trajectory that are not feasible under $\pi^{*}$.

This strategy would suffer from the curse of the horizon, i.e., the variance of the corresponding estimator of $\theta(\pi^{*})$ would increase exponentially with the time horizon $T$.

State Marginalized Weighting [20]: Again using the Markov property, an alternative way of factorizing $w_{t,i}$ in (7) is as follows.

$w_{t_{i}}=\frac{p_{\pi^{*}}(s_{t,i})~\pi^{*}(a_{t,i}\mid s_{t,i})}{p_{\pi}(s_{t,i})~\pi(a_{t,i}\mid s_{t,i})}.$ | | | |

First, we consider the case when the state space is discrete and finite. In this case, $p_{\pi}(s_{t})$ can be estimated as

$\hat{p}_{\pi}(s_{t})=\frac{1}{N}\sum_{i=1}^{n}1_{\{s_{t,i}=s_{t}\}}.$ | | | |

Next, we estimate $p_{\pi^{*}}(s_{t}\mid s_{1})$ recursively as follows.

$\hat{p}_{\pi^{*}}(s_{t})=\begin{cases}\frac{1}{N}\sum_{i=1}^{n}1_{\{s_{t,i}=s_{t}\}}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\text{if $t=1$};\\
\frac{1}{N}\sum_{i=1}^{n}1_{\{s_{t,i}=s_{t}\}}\frac{p_{\pi^{*}}(s_{t-1,i})~\pi^{*}(a_{t-1,i}\mid s_{t-1,i})}{p_{\pi}(s_{t-1,i})~\pi(a_{t-1,i}\mid s_{t-1,i})}\\
\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\text{if $t>1$}.\end{cases}$ | | | |

This strategy avoids the curse of the horizon by marginalizing over the state dimension. In fact, [20] showed that the variance of the corresponding estimator of $\theta(\pi^{*})$ is $O(T^{3})$. However, the estimations of $p_{\pi}(s_{t})$ and $p_{\pi^{*}}(s_{t})$ suffers from the curse of state dimensionality.

In the case of a large state space with multiple features (some/all of which can be continuous), we apply two strategies for dimension reduction:

-

We remove the state features from offline evaluation that are not influenced by the action (e.g., static features). More precisely, we can work with a reduced state space $h(s_{t})$ as long as we have

$\displaystyle\frac{p_{\pi^{*}}(s_{t})}{p_{\pi^{*}}(s_{t})}=\frac{p_{\pi^{*}}(h(s_{t}))}{p_{\pi^{*}}(h(s_{t}))}.$ | | | | (8) |

It is to show that following is a sufficient condition for having (8):

$p(s_{t}\mid h(s_{t}),a_{1},\ldots,a_{t})=p(s_{t}\mid h(s_{t})),$ | | | |

which states that all the features that are not in $h(s_{t})$ are conditionally independent of the actions given $h(s_{t})$. Note that these features (that are not in $h(s_{t})$) cannot be removed from model training since they can have an influence on the reward function.

-

We discretize each feature (that is left after the first step) into a fixed number of bins, where the bin size controls the bias-variance trade-off (a larger number of bins would lead to a smaller bias but a larger variance).

One-Step Correction based Weighting [13]: An easy way to construct a biased but low variance estimator is to use the following weights.

$w_{t,i}=\frac{\pi^{*}(s_{t,i}\mid a_{t,i})}{\pi(s_{t,i}\mid a_{t,i})}$ | | | |

This strategy avoids both the curse of dimensionality of the state space and the curse of the long horizon. The corresponding estimator that only corrects for the one-step policy mismatch is no longer unbiased. However, when the variances of both action trajectory based estimator and the state marginalized estimator are too high, the one-step correction might be the only reliable method.

### IV-C Simulation Environment

In order to evaluate and benchmark offline evaluation methods and offline training algorithms with known ground truth, we build a simplified simulation environment using Open AI GYM [41], to mimic the online notification spacing system. The reason we build such a simulation environment instead of using existing ones in GYM is that learning algorithms and offline evaluation methods that work well in one environment may not work as well in another environment. The closer we build a simulation environment to the real notification environment, the more accurate and useful we get from simulation studies. Section V-A gives a simulated study using this simulation environment.

The Markov decision process, which is implemented in the simulator, is described below.

-

Environment: The simulator mimics the notification queue of a user along with the environment. The environment generates notification candidates to be sent to users. We simulate user visits based on two factors - badge count (the number of unseen notifications awaiting the user on the app) and their activeness on the app. We assume that once a user visits the app, the notifications previously sent to them are seen by the user, and hence we reset the badge count for the user.

-

Transition process: The time step for the simulator is set to 4 hours. At each time step, the simulator generates new notifications to arrive in the queue. This is done by sampling a Poisson process, which is scaled by the day and time of the week as well as the demand patterns at that time. Each notification added to the queue consists of a relevance score (used to indicate the rank of the notification in the queue) as well as the time of expiry. Since users also receive near-real-time notifications outside of the queue, their visits and badge counts are affected by those notifications. To get the simulation environment closer to the real-world scenario, we similarly simulate new notifications directly sent to the user outside of the queue and update their badge count as part of the environment. Finally, we check for any expiring notifications available in the queue and remove them from the queue by either sending them to the user with a $50\%$ probability or dropping them altogether.

-

State: We use a simple four-dimensional state comprising of badge count, number of candidate notifications in the queue, time since the start of the week, and user activeness.

-

Actions: Consistent with the production system, there are two actions, SEND and NOT-SEND.

-

Reward: The numerical reward is 1 if a user visits in the next four hours. We use a reward model, which is a function of user activeness and badge count, which is fit to qualitatively represent actual user performance that is learned from our visit state transition models describe in [5].

The simulator roughly captures the notification spacing system, with a time-varying demand generation and user visits to the app. The queuing and de-queuing logic is consistent with the actual notification spacing system. For demand generation and user visits, we qualitatively capture the characteristics of our production models in the simulator.

Once the offline training algorithm and offline evaluation method is chosen, the offline training of the real-world data is independent and takes no information from this simulation environment. Therefore, any simplification and bias of the simulation environment from the real notification system is not a concern. Such an environment setup can be a low-cost alternative to simulation environments for hybrid learning [42, 32].

## V Experiments

We present both a simulation study and a real-world online experiment to demonstrate the contribution of the proposed framework.

### V-A Offline Evaluation in Simulation Environment

First, we show why offline evaluation is crucial to the offline reinforcement learning framework. Without true interaction with the environment, offline training can be unstable in terms of convergence and policy performance. We generated both training and validation data using an exploration policy describe in Equation 6 in the notification simulation environment described in Section IV-C. We then trained 128 Double DQN policies using the same training data under various hyper parameters (batch size, learning rate, number of batches with a fixed target network, network layer size, etc.). Figure 3 shows the online evaluation distribution in the simulation environment of the 128 policies, which can be regarded as the ground truth of the policy performance. The red vertical line is the performance of the behavior exploration policy. Only 38 out of 128 policies delivered total rewards exceeding that of the behavior policy. This demonstrates that in the offline learning setting, the learned policy is not guaranteed to perform better than the behavior policy and that hyper-parameter tuning and offline evaluation is necessary. For real-world applications, online evaluation can be risky and expensive, and offline evaluation is desired.

*Fig. 3: Distribution of policy performance with respect to the baseline in red *

Next, we show how the proposed state-marginalized importance sample provides better bias-variance trade-offs compared with trajectory matching and one-step matching described in Section IV-B. We evaluate the same 128 policies above using the three offline evaluation methods, where we used the self-normalized version of importance weights, i.e. $w_{t,i}/(\frac{1}{N}\sum_{i=1}^{N}w_{t,i})$, which is known to be more stable in practice.

Figure 4 shows the box plot of the error between offline and online estimates of the cumulative reward across all learned policies, obtained from one-step, action-trajectory and state-marginalized importance weighting techniques. We observe that action trajectory based importance weighting technique provides the lowest biased estimate of the cumulative reward but has the highest variance. One step importance weighting technique provides the highest biased estimate of the cumulative reward but has the lowest variance. The state-marginalized importance weighting estimate has a better balance between bias and variance of the estimate. In this simulation, we chose the number of bins to be 10 for discretizing the states. The bias of the estimate from this technique can be lowered by increasing the number of bins but at the expense of higher variance. However, the variance can be easily estimated in the production-setting, thereby allowing us to optimally choose the number of bins to control bias-variance trade-offs in our estimation.

*Fig. 4: Bias-variance comparison between three methods*

*Fig. 5: System architecture*

### V-B Online Experiments in Notification Spacing

Figure 5 illustrates the system architecture of this application. All policies, regardless of supervised or reinforcement learning, are served in a nearline Samza service [43]. Various notification providers send notification candidates to the Samza service and are queued in its data store for future evaluations. Every user’s notification queue is evaluated by a policy every few hours for a send-or-not decision for its top scored notification. A policy takes online features, makes decisions, and then snapshots its decisions and features to a Hadoop Distributed File System (HDFS) [44]. Additional offline features can be used for offline training and pushed to Samza data stores for nearline serving.

Under this architecture, we collected one-week snapshot data from a small percentage of LinkedIn users using an epsilon-greedy behavior policy $\pi^{\epsilon}$ on top of the baseline policy described in III-A. The data was then joined with other log data to construct the tuples $(s_{t},a_{t},s_{t+1},r_{t})$ for offline training. The state features we use are the same as the features in the baseline supervised model to allow a fair comparison. Important state features include the app badge count, the user’s last visit time, the number of notifications received over the past week and other user profile features. We then train the Double DQN models described in Section IV-A using a fully-connected 3-layer neural network with different hyper-parameters (number of inner nodes, learning rate, batch size, number of batches between target network updates, etc.). We apply the state marginalized weighting to estimate the expected total return of the learned policies. Given the large continuous state space, we apply the dimension reduction and discretization strategies described in Section IV-B for the state marginalized weighting to further reduce the estimation variance. The top performer (or performers) selected by offline evaluation was deployed in the Samza service on a certain percentage of total users for an online A/B test compared with the baseline policy. We are interested in user engagement and notification interactions, which can be characterized by the following metrics.

-

Sessions: A session is a collection of full-page views made by a single user on the same device type. Two sessions are separated by $30$ minutes of zero activity. This is a widely used metric on user engagement across social networks.

-

Notification cards: the total number of notification cards served to users after removing potential duplicates. This is a metric capturing notification volume.

-

Notification CTR: This metric measures the average click-through-rate of notifications sent to a user in a day. This is a metric capturing notification interactions.

-

Notification unfollow total: the total number of notification unfollow actions taken by users. This is negative feedback from users.

*TABLE I: Online A/B results for delivery time optimization*

| Metric | vs. Baseline policy |

| Sessions | +0.30% |

| Notification cards | -3.49% |

| Notification CTR | +4.53% |

| Notification unfollow total | -4.37% |

Table I shows the full-week A/B test results, and the numbers are all statistically significant (with p-value $<0.05$). Compared with the baseline policy, the new policy from offline reinforcement learning increased the total sessions by $0.3\%$, which is considered a moderate gain in a volume neutral iteration, but very impressive given that the total notification volume is reduced by $3.49\%$. The $4.53\%$ increase in notification CTR and $4.37\%$ decrease in notification unfollow total are mainly driven by the reduction in notification volume. These numbers demonstrate the business impact of our proposed framework and suggest that notifications were delivered at better timing and lower frequency, suggesting more optimality over the supervised approach.

In addition to business impact, we would like to point out that the offline reinforcement learning training and evaluation framework provides a significant increase in iteration speed in our ecosystem. At LinkedIn, we constantly test new features and new learning algorithms, which requires a modeling iteration cycle consisting of offline training and evaluation, online tuning (if needed) before we take it to the online A/B test to conclude whether the new model is better than its baseline. Therefore, we measure the iteration speed by the time it takes for each steps. Although reinforcement learning algorithms in general take more computation resources and longer time to converge than most supervised learning models, the difference in training and evaluation (in hours) is negligible when we take the online tuning into consideration. As described in Section III-A, $\tau$ has to be tuned online for a trade-off between short-term and long-term engagement. In fact, many supervised model-based frameworks typically resort to online tuning like the approach used in [9], as there is usually a gap between the model predictions and the service decisions/actions. The tuning typically takes 1-3 weeks for notifications as site engagement responses takes days to show up. This leads to tedious and time-consuming efforts for many practitioners. In contrast, the offline evaluation system, deployed as a part of the reinforcement learning model, does not require an online tuning cycle after the policy selection done in the offline evaluation. The selected policy can go directly to the online A/B test to draw a conclusion on a modeling iteration.

## VI Discussion

In this paper, we propose an offline reinforcement learning framework covering data collection, offline learning, offline evaluation, assisting simulation environment, and other practical considerations. We argue and demonstrate the benefits of such a framework as a more principled paradigm to optimize notification decisions over supervised learning approaches. Practically, it shortens the modeling iteration cycle for real-world systems that constantly improve with new features and retraining.

One of the limitations of our presented results is that we trained and tested this framework in a one-week frame. It would be interesting to see whether the framework can improve even longer-term engagement. This is limited by the measurement cost of truly long-term engagement, say a one-year scope.

Based on its positive impact and iteration speed gains, the reinforcement learning application in Section V-B was fully deployed at LinkedIn. We hope our work could motivate wider adoption of reinforcement learning for the mainstream recommender systems. We will continue to work on how to learn the optimal policy more efficiently offline and how to evaluate more accurately offline.

## Acknowledgment

We are thankful to Shaunak Chatterjee, Yan Gao, Cyrus DiCiccio, Bee-Chung Chen, Deepak Agawal, Matthew Walker, Romer rosales, Shipeng Yu and Mohsen Jamali for their detailed and insightful feedback during the development of this work.

## References

- [1] M. Pielot, K. Church, and R. De Oliveira, “An in-situ study of mobile phone notifications,” in Proceedings of the 16th international conference on Human-computer interaction with mobile devices & services. ACM, 2014, pp. 233–242.

- [2] T. Okoshi, K. Tsubouchi, and H. Tokuda, “Real-world product deployment of adaptive push notification scheduling on smartphones,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2792–2800.

- [3] R. Gupta, G. Liang, H.-P. Tseng, R. K. Holur Vijay, X. Chen, and R. Rosales, “Email volume optimization at linkedin,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016, pp. 97–106.

- [4] R. Gupta, G. Liang, and R. Rosales, “Optimizing email volume for sitewide engagement,” in Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 2017, pp. 1947–1955.

- [5] Y. Yuan, J. Zhang, S. Chatterjee, S. Yu, and R. Rosales, “A state transition model for mobile notifications via survival analysis,” in Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, 2019, pp. 123–131.

- [6] Y. Gao, V. Gupta, J. Yan, C. Shi, Z. Tao, P. Xiao, C. Wang, S. Yu, R. Rosales, A. Muralidharan et al., “Near real-time optimization of activity-based notifications,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 283–292.

- [7] B. Zhao, K. Narita, B. Orten, and J. Egan, “Notification volume control and optimization system at pinterest,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 1012–1020.

- [8] D. Agarwal, B.-C. Chen, P. Elango, and X. Wang, “Click shaping to optimize multiple objectives,” in Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2011, pp. 132–140.

- [9] K. Basu, C. DiCiccio, B. Gavin, V. Gupta, and Y. Ouyang, “Bayesian optimization for balancing metrics in recommender systems,” International Joint Conference on Artificial Intelligence, 2020. [Online]. Available: https://sites.google.com/view/ijcai2020-linkedin-bayesopt/home

- [10] S. Levine, A. Kumar, G. Tucker, and J. Fu, “Offline reinforcement learning: Tutorial, review, and perspectives on open problems,” arXiv preprint arXiv:2005.01643, 2020.

- [11] S. Lange, T. Gabel, and M. Riedmiller, “Batch reinforcement learning,” in Reinforcement learning. Springer, 2012, pp. 45–73.

- [12] R. Agarwal, D. Schuurmans, and M. Norouzi, “Striving for simplicity in off-policy deep reinforcement learning,” 2019.

- [13] M. Chen, A. Beutel, P. Covington, S. Jain, F. Belletti, and E. H. Chi, “Top-k off-policy correction for a reinforce recommender system,” in Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, 2019, pp. 456–464.

- [14] E. Ie, V. Jain, J. Wang, S. Narvekar, R. Agarwal, R. Wu, H.-T. Cheng, M. Lustman, V. Gatto, P. Covington et al., “Reinforcement learning for slate-based recommender systems: A tractable decomposition and practical methodology,” arXiv preprint arXiv:1905.12767, 2019.

- [15] S. Fujimoto, D. Meger, and D. Precup, “Off-policy deep reinforcement learning without exploration,” in International Conference on Machine Learning. PMLR, 2019, pp. 2052–2062.

- [16] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative q-learning for offline reinforcement learning,” arXiv preprint arXiv:2006.04779, 2020.

- [17] A. R. Mahmood, H. Van Hasselt, and R. S. Sutton, “Weighted importance sampling for off-policy learning with linear function approximation.” in NIPS, 2014, pp. 3014–3022.

- [18] N. Jiang and L. Li, “Doubly robust off-policy value evaluation for reinforcement learning,” in International Conference on Machine Learning. PMLR, 2016, pp. 652–661.

- [19] P. Thomas and E. Brunskill, “Data-efficient off-policy policy evaluation for reinforcement learning,” in International Conference on Machine Learning. PMLR, 2016, pp. 2139–2148.

- [20] T. Xie, Y. Ma, and Y.-X. Wang, “Towards optimal off-policy evaluation for reinforcement learning with marginalized importance sampling,” arXiv preprint arXiv:1906.03393, 2019.

- [21] R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction. MIT press Cambridge., 1998.

- [22] H. Van Hasselt, Y. Doron, F. Strub, M. Hessel, N. Sonnerat, and J. Modayil, “Deep reinforcement learning and the deadly triad,” arXiv preprint arXiv:1812.02648, 2018.

- [23] A. Mehrotra, M. Musolesi, R. Hendley, and V. Pejovic, “Designing content-driven intelligent notification mechanisms for mobile applications,” in Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing. ACM, 2015, pp. 813–824.

- [24] M. Pielot, R. de Oliveira, H. Kwak, and N. Oliver, “Didn’t you see my message?: predicting attentiveness to mobile instant messages,” in Proceedings of the SIGCHI Conference on Human Factors in Computing Systems. ACM, 2014, pp. 3319–3328.

- [25] M. Pielot, B. Cardoso, K. Katevas, J. Serrà, A. Matic, and N. Oliver, “Beyond interruptibility: Predicting opportune moments to engage mobile phone users,” Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, vol. 1, no. 3, pp. 1–25, 2017.

- [26] Q. Wu, H. Wang, L. Hong, and Y. Shi, “Returning is believing: Optimizing long-term user engagement in recommender systems,” in Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, 2017, pp. 1927–1936.

- [27] T. Okoshi, J. Ramos, H. Nozaki, J. Nakazawa, A. K. Dey, and H. Tokuda, “Attelia: Reducing user’s cognitive load due to interruptive notifications on smart phones,” in 2015 IEEE International Conference on Pervasive Computing and Communications (PerCom). IEEE, 2015, pp. 96–104.

- [28] ——, “Reducing users’ perceived mental effort due to interruptive notifications in multi-device mobile environments,” in Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing, 2015, pp. 475–486.

- [29] V. Pejovic and M. Musolesi, “Interruptme: designing intelligent prompting mechanisms for pervasive applications,” in Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing, 2014, pp. 897–908.

- [30] M. Pielot, T. Dingler, J. S. Pedro, and N. Oliver, “When attention is not scarce-detecting boredom from mobile phone usage,” in Proceedings of the 2015 ACM international joint conference on pervasive and ubiquitous computing, 2015, pp. 825–836.

- [31] T. Okoshi, K. Tsubouchi, M. Taji, T. Ichikawa, and H. Tokuda, “Attention and engagement-awareness in the wild: A large-scale study with adaptive notifications,” in 2017 ieee international conference on pervasive computing and communications (percom). IEEE, 2017, pp. 100–110.

- [32] L. Zou, L. Xia, Z. Ding, J. Song, W. Liu, and D. Yin, “Reinforcement learning to optimize long-term user engagement in recommender systems,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2810–2818.

- [33] X. Zhao, L. Zhang, Z. Ding, L. Xia, J. Tang, and D. Yin, “Recommendations with negative feedback via pairwise deep reinforcement learning,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 1040–1048.

- [34] J. W. Hughes, K.-h. Chang, and R. Zhang, “Generating better search engine text advertisements with deep reinforcement learning,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2269–2277.

- [35] P. Wang, K. Liu, L. Jiang, X. Li, and Y. Fu, “Incremental mobile user profiling: Reinforcement learning with spatial knowledge graph for modeling event streams,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020, pp. 853–861.

- [36] A. Swaminathan, A. Krishnamurthy, A. Agarwal, M. Dudík, J. Langford, D. Jose, and I. Zitouni, “Off-policy evaluation for slate recommendation,” arXiv preprint arXiv:1605.04812, 2016.

- [37] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing atari with deep reinforcement learning,” arXiv preprint arXiv:1312.5602, 2013.

- [38] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 30, no. 1, 2016.

- [39] Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot, and N. Freitas, “Dueling network architectures for deep reinforcement learning,” in Proceedings of The 33rd International Conference on Machine Learning, 2016, pp. 1995–2003.

- [40] T. W. Sandholm and R. H. Crites, “Multiagent reinforcement learning in the iterated prisoner’s dilemma,” Biosystems, vol. 37, no. 1-2, pp. 147–166, 1996.

- [41] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” CoRR, vol. abs/1606.01540, 2016. [Online]. Available: http://arxiv.org/abs/1606.01540

- [42] J.-C. Shi, Y. Yu, Q. Da, S.-Y. Chen, and A.-X. Zeng, “Virtual-taobao: Virtualizing real-world online retail environment for reinforcement learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, 2019, pp. 4902–4909.

- [43] S. A. Noghabi, K. Paramasivam, Y. Pan, N. Ramesh, J. Bringhurst, I. Gupta, and R. H. Campbell, “Samza: stateful scalable stream processing at linkedin,” Proceedings of the VLDB Endowment, vol. 10, no. 12, pp. 1634–1645, 2017.

- [44] K. Shvachko, H. Kuang, S. Radia, and R. Chansler, “The hadoop distributed file system,” in 2010 IEEE 26th symposium on mass storage systems and technologies (MSST). Ieee, 2010, pp. 1–10.

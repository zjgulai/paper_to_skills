<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2307.09702
     paper_id : 2307.09702
     source   : https://arxiv.org/html/2307.09702v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# Efficient Guided Generation for LLMs

Brandon T. Willard Affiliation: Normal Computing    Rémi Louf Affiliation: Normal Computing

2023-07-14

###### Abstract

In this article we describe an efficient approach to guiding language model text generation with regular expressions and context-free grammars. Our approach adds little to no overhead to the token sequence generation process, and makes guided generation feasible in practice. An implementation is provided in the open source Python library Outlines (Louf and Willard, ).

## 1 Introduction

We are concerned with the problem of generating sequences of tokens from a large language model (LLM) (Vaswani et al., 2017; Radford et al., 2019) that conform to regular expressions or context-free grammars (CFGs). This kind of guided LLM generation is used to make LLM model output usable under rigid formatting requirements that are either hard or costly to capture through fine-tuning alone (Beurer-Kellner et al., 2023; Scholak et al., 2021; Poesia et al., 2022; Rabinovich et al., 2017; Weng, 2021). Such features are being generalized in prompting libraries and interfaces (Microsoft, 2023; Beurer-Kellner et al., 2023; Rickard, 2023a; Rickard, 2023b), but their applicability is marred by their current cost.

Most implementations of guided generation bias the score values used to determine the probabilities of the tokens in an LLM’s vocabulary. A complete but naive approach would scan the entire vocabulary to determine which tokens are valid given the previously sampled tokens and set the probabilities of some tokens to zero when they would violate the constraints. This approach is naive because it scales as $\mathcal{O}(N)$, where $N$ is the size of the LLM’s vocabulary.

We propose an approach that uses the standard finite state machine (FSM) formulation of regular expressions to both arbitrarily start and stop guided generation and allow the construction of an ”index” with which the set of non-zero-probability tokens can be obtained efficiently at each step. The result is an algorithm that scales as $\mathcal{O}(1)$ on average.

Our FSM approach can also be extended to CFGs and $\operatorname{LALR}(1)$ parsers to allow for efficient guided generation according to popular data formats and programming languages (e.g. JSON, Python, SQL, etc.).

## 2 LLM Sampling and Guided Generation

In the following, we summarize the LLM token sampling process and guided generation.

Let $\boldsymbol{\Sigma}=(t_{1},\dots,t_{i})$ represent a sequence of $i$ tokens, we can define the next token $t_{i+1}\sim T_{i+1}$ as a sample from the LLM-generated random variable $T_{i+1}$:

$\displaystyle\boldsymbol{\alpha}$ $\displaystyle=\operatorname{LM}(\boldsymbol{t},\boldsymbol{\theta})$ | | | | |

$\displaystyle T_{i+1}$ $\displaystyle\sim\operatorname{Categorical}({\boldsymbol{\alpha}})$ | | | | |

where $\boldsymbol{\theta}$ is the set of trained parameters. The variable $\boldsymbol{\alpha}$ and the support of $T_{i+1}$ span the LM’s vocabulary, $\mathcal{V}$. The $\operatorname{LM}$ function refers to a deep neural network trained on next-token-completion tasks.

The vocabularies, $\mathcal{V}$, are composed of strings from a fixed alphabet (Sennrich et al., 2015), and–for our purposes–we can consider the exact strings as more or less random. We will denote the size of a vocabulary with $\lvert\mathcal{V}\rvert=N$, where, in some cases, $N$ is on the order of $10^{4}$.

The elements of the set of multi-token strings $\mathcal{S}\in\mathcal{V}^{*}$ are thus random variables, and each sequence has probability:

$\operatorname{P}\left(\Sigma=(t_{1},\dots,t_{i})\right)=\operatorname{P}\left(t_{1}\mid\emptyset\right)\prod_{n=1}^{i}\operatorname{P}\left(t_{n}\mid t_{1},\dots,t_{n-1}\right)$ | | | | (1) |

In the language model setting, we are interested in the subset $\mathcal{F}\subset\mathcal{S}$ of sequences that end with a special token $\text{EOS}\in\mathcal{V}$, the <EOS> token. Elements of $\mathcal{F}$ are random variables with support $\mathcal{V}^{\tau}$ where $\tau$ is also a random variable. Processes that draw samples from this distribution are therefore stop processes.

*Algorithm 1 Basic LLM token sampling*

1: function sample_tokens

2:   $\boldsymbol{t}\leftarrow()$

3:   for $i\leftarrow 1,M$ do

4:    $\boldsymbol{\alpha}\leftarrow$ $\operatorname{LM}$($\boldsymbol{t}$, $\boldsymbol{\theta}$)

5:    Sample $t\sim\operatorname{Categorical}({\boldsymbol{\alpha}})$

6:    if $t=\text{EOS}$ then

7:      break

8:    end if

9:    $\boldsymbol{t}\leftarrow$ $\operatorname{append}$($\boldsymbol{t}$, $t$)

10:   end for

11:   return $\boldsymbol{t}$

12: end function

The procedure in Algorithm 1 describes one such process, often called multinomial sampling, that samples elements of $\boldsymbol{F}$ by iteratively sampling new tokens until the <EOS> token is generated. Other methods have been used to generate samples from this distribution: greedy decoding consists in chosing the most probable token at each step, beam search is a heuristic to find the mode of the distribution. More recently, SMC sampling has been used to sample from this distribution (Lew et al., 2023).

### 2.1 Guided generation

The newer generation of Large Language Models has proven efficient at generating structured outputs, for instance code or outputs in JSON format, by conditioning on the prompt alone. The validity of the output is however influenced heavily by subtle differences in prompting, and even the best known prompting techniques do not systematically result in valid outputs. This is exemplified by the existence of libraries that validate the outputs and re-prompt the model when it is invalid (Rajpal, 2023).

The methods exposed here guarantee the validity of the output. They guide the sequence generation process by complementing it with a deterministic monitoring process. This process keeps track of the generated tokens, and manipulates the logits at each step of the process.

We can indeed derive other random variables from the next-token distribution by manipulating the output logits $\boldsymbol{\alpha}$, randomly or deterministically. More specifically, we can construct unnormalized conditional distributions. Since we’re dealing with a simple discrete case, the general approach can be described by the application of boolean mask $m$ that restricts the support of the distribution like so:

$\displaystyle\boldsymbol{\alpha}$ $\displaystyle=\operatorname{LM}(\boldsymbol{t},\boldsymbol{\theta})$ | | | | |

$\displaystyle\tilde{\boldsymbol{\alpha}}$ $\displaystyle=m\odot\boldsymbol{\alpha}$ | | | | |

$\displaystyle\tilde{T}_{i+1}$ $\displaystyle\sim\operatorname{Categorical}({\tilde{\boldsymbol{\alpha}}})$ | | | | |

The resulting conditional distribution implied by $\tilde{T}_{i+1}$ may encode constraints on the support of $T_{i+1}$. For instance, $\tilde{T}_{i+1}$ could be used to represent:

-

digit samples

-

the exclusion of <EOS> (end-of-sequence) tokens

-

strings that match the regular expression [a-zA-Z]

-

strings that parse according to a specified grammar (e.g. Python, SQL, etc.)

This method applies equally to every generation method mentioned above.

In Algorithm 2 we augment Algorithm 1 so that it masks the logits.

*Algorithm 2 LLM token sampling with masking*

1: function sample_tokens

2:   $\boldsymbol{t}\leftarrow()$

3:   for $i\leftarrow 1,M$ do

4:    $\boldsymbol{\alpha}\leftarrow$ $\operatorname{LM}$($\boldsymbol{t}$, $\boldsymbol{\theta}$)

5:    Construct the mask $m$

6:    $\tilde{\boldsymbol{\alpha}}\leftarrow m\odot\boldsymbol{\alpha}$

7:    Sample $\tilde{t}\sim\operatorname{Categorical}({\tilde{\boldsymbol{\alpha}}})$

8:    if $\tilde{t}=\text{EOS}$ then

9:      break

10:    end if

11:    $\boldsymbol{t}\leftarrow$ $\operatorname{append}$($\boldsymbol{t}$, $\tilde{t}$)

12:   end for

13:   return $\boldsymbol{t}$

14: end function

The computation corresponding to line $\ref{alg:llm-sequence-sampling-with-mask}.\ref{construct-m}$ is the step that requires some form of evaluation over all the elements of $\mathcal{V}$. In other words, aside from computing $\boldsymbol{\alpha}$, the biggest cost is in determining the support of $\tilde{T}_{i+1}$ on each iteration. In the case of regular expression-guided masking–and cases more sophisticated than that–the support and, thus, $m$ will depend on the current token sequence $(t_{1},\dots,t_{i})$. This means that guided generation is ultimately an ”iterative” matching and/or parsing problem, because we are not given the entire string upfront. This leads us to the main question of this work: how can we efficiently determine the support/mask $m$ at each iteration of Algorithm 2? We could always iterate through the vocabulary at each step and determine which strings are acceptable according to the guiding regular expression; however, that becomes impractical for large vocabularies, since it scales as $\mathcal{O}(N)$ and needs to be performed on each iteration.

### 2.2 Examples

In this section we use GPT2-medium (355M parameters) to illustrate how guided generation works in practice. We use the library Outlines to generate them:

⬇

import outlines.models as models

import outlines.text.generate as generate

model = models.transformers(”gpt2-medium”)

prompt = ”Is 1+1=2?”

unguided = generate.continuation(model, max_tokens=30)(prompt)

guided = generate.regex(model, r”\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)”, max_tokens=30)(

prompt

)

print(unguided)

# Is 1+1=2?

#

# This is probably the most perplexing question. As I said in one of my articles describing how I call 2 and 1, there isn’t

print(guided)

# Is 1+1=2?Yes

⬇

prompt = ”In what year was Noam Chomsky born?\n”

unguided = generate.continuation(model, max_tokens=30)(prompt)

guided = generate.regex(model, r”\s*19[0-9]{2}”, max_tokens=30)(prompt)

print(unguided)

# In what year was Noam Chomsky born?

#

# Professor Chomsky was born in about 1895 in Mille Medad, near Paris. Like others Chomsky does not know the details of the birth weight of

print(guided)

# In what year was Noam Chomsky born?1952

⬇

prompt = ”What is the IP address of the Google DNS servers?”

unguided = generate.continuation(model, max_tokens=30)(prompt)

guided = generate.regex(

model,

r”((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)”,

max_tokens=30,

)(prompt)

print(unguided)

# What is the IP address of the Google DNS servers?

#

# Passive DNS servers are at DNS servers that are private. In other words, both IP servers are private. The database does not contain Chelsea Manning

print(guided)

# What is the IP address of the Google DNS servers?

# 2.2.6.1

## 3 Iterative FSM Processing and Indexing

For the case of regular expression-guided generation, we frame the problem in terms of state machines. This framing allows us to specify exactly how regular expression matching can be arbitrarily ”started” and ”stopped”, so that it can be easily and efficiently continued between samples of $\tilde{T}_{i+1}$.

To be precise, we consider regular expressions in 5-tuple finite automaton form (Sipser, 1996, Definition 1.5):

###### Definition 1 (Finite Automaton).

A finite automaton is given by $(Q,\Sigma,\delta,q_{0},F)$, where $Q$ is a finite set of states, $\Sigma$ a finite alphabet, $\delta:Q\times\Sigma\to Q$ the transition function, $q_{0}\in Q$ the start state, and $F\in Q$ the set of accept states.

This formulation allows us to determine the exact states in $Q$ in which the guiding regular expression’s FSM stops after sampling a single vocabulary token $\tilde{t}_{i+1}$. These FSM states can then be tracked during the LLM token sampling procedure above and used to efficiently continue and restart the state machine for the purpose of determining the support.

We can efficiently process a sequence of tokens by simply walking through the FSM steps between token sampling, but looping through the vocabulary is still the issue. For that, we pre-process the vocabulary using the regular expression’s FSM. The important part is that we do this in a way that considers starting in every possible FSM state, because the strings in the vocabulary could be arbitrary parts of the regular expression.

A procedure for producing matches starting at any point in the FSM is given in Algorithm 3. The result is a list of sub-sequences detailing the states through which the FSM traverses when/if it accepts the provided string.

*Algorithm 3 Find sub-sequences of the FSM $M$ that accept the string $\boldsymbol{v}$*

1: function find_sub_sequences($M$, $\boldsymbol{v}$)

2:   $M=(Q,\Sigma,\delta,q_{0},F)$

3:   $res\leftarrow()$

4:   for $r\in\delta^{-1}(\cdot,v_{0})$ do $\triangleright$ Loop through states that read $v_{0}$

5:    $p\leftarrow(r)$

6:    for $i\leftarrow 1,\lvert\boldsymbol{v}\rvert-1$ do $\triangleright$ Walk the FSM

7:      if $\delta(r,v_{i})=\emptyset$ then $\triangleright$ The FSM does not read $v_{i}$

8:       $p\leftarrow()$

9:       break $\triangleright$ Stop walking and try the next start state

10:      end if

11:      $r\leftarrow\delta(r,v_{i})$

12:      $p\leftarrow$ $\operatorname{append}$($p$, $r$)

13:    end for

14:    $res\leftarrow$ $\operatorname{concat}$($res$, $p$)

15:   end for

16:   return $res$

17: end function

By matching the starting states of these sub-sequences to the last FSM state arrived at in Algorithm 1, we can efficiently index the vocabulary by creating a map, $\sigma:Q\to\mathcal{P}(\mathcal{V})$, connecting FSM states and sets of elements of the vocabulary that will be accepted by the FSM at those states.

Algorithm 4 describes the construction of $\sigma$.

*Algorithm 4 Construct a map from FSM states to subsets of $\mathcal{V}$*

1: function map_states_to_vocab($M$, $\mathcal{V}$)

2:   $M=(Q,\Sigma,\delta,q_{0},F)$

3:   Initialize the map $\sigma$ with empty sets for each element in $Q$

4:   for $v\in\mathcal{V}$ do $\triangleright$ Loop through the vocabulary

5:    $Z\leftarrow$ $\operatorname{find\_sub\_sequences}$($M$, $v$)

6:    for $z\in Z$ do $\triangleright$ Loop through state sequences accepting $v$

7:      $\sigma(z[0])\leftarrow\sigma(z[0])\cup v$

8:    end for

9:   end for

10:   return $\sigma$

11: end function

Using a hash-map for $\sigma$ can make the $m$ step in Algorithm 2 scale as $\mathcal{O}(1)$ on average. Furthermore, since $\sigma$ is constructed outside of the token sampling procedure, its run-time cost is effectively irrelevant, although it theoretically requires memory proportional to the number of states in the FSM (i.e. $\lvert Q\rvert$). Fortunately, for non-pathological combinations of regular expressions and vocabularies, not every string in the vocabulary will be accepted by the FSM. Likewise, the maps can be ”post-processed” so that the values in the map share pointers to the same subsets of the vocabulary, reducing the memory footprint when some FSM states accept all characters.

### 3.1 Illustration

To contextualize the effect of $\mathcal{O}(N)$ scaling versus the $\mathcal{O}(1)$ approach described here, and implemented in Outlines, we perform a simple comparison with the Guidance library.

The guidance code and prompt used for this comparison are as follows:

⬇

import guidance

llm = guidance.llms.Transformers(

”gpt2”,

token_healing=False,

device=”cuda”,

temperature=0.1,

)

program = guidance(

f”””What is a good Python variable name?{{{{gen temperature=0.1 max_tokens={max_tokens} pattern=”[^\W\d]\w*”}}}}”””,

llm=llm,

caching=False,

async_mode=False,

stream=False,

log=False,

silent=True,

)

# Generate the token sequence.

# Only this call is timed.

program().text

The corresponding Outlines code is as follows:

⬇

from outlines import disable_cache

import outlines.models as models

import outlines.text.generate as generate

disable_cache()

model = models.transformers(”gpt2”, device=”cuda”, temperature=0.1)

prompt = ”What is a good Python variable name?”

guided_continuation = generate.regex(

model,

r”[^\W\d]\w*”,

max_tokens=max_tokens,

)

def reset_continuation():

# This allows us to sample new sequences on each call

guided_continuation.pstates = []

return guided_continuation(prompt)

# Generate the token sequence.

# Only this call is timed.

reset_continuation()

The value of max_tokens is varied and the timings are recorded with timeit for a single loop and single repeat value (i.e. only one sample is collected for each value of max_tokens). The results are plotted in Figure 1.

Barring any configuration oversights that might be creating a large run-time discrepancy, the nearly linear scaling in the maximum number of sampled tokens is striking. The vocabulary size is 50,257, so this scaling should probably not be all that surprising, especially since Guidelines must iterate through the entire vocabulary each time it draws a sample.

*Figure 1: Run-time measurements for regex-guided sequence generation across maximum sequence length settings.*

## 4 Extension to Iterative Parsing

In this section, we move our focus to general parser-guided generation and start with a simple walkthrough for a Python-like grammar provided as a CFG.

Consider a vocabulary consisting of strings like ”d” and ”ef” that can be combined to produce Python-like syntax, and assume that these strings are sequentially sampled and concatenated according to a process like Algorithm 1.

Furthermore, consider a terminal symbol DEF in the CFG that corresponds to the string ”def” (i.e. the trivial regular expression def), and a NAME symbol given by the regular expression [^\W\d]\w* (e.g. Python identifiers). We want to iteratively lex/parse streams of strings sampled from the aforementioned vocabulary.

For example, the following is such a stream: [”d”, ”ef”, ”␣f”, ”oo(”, ”):”, ”␣”, ”pass”]. Concatenating the stream produces ”def␣foo():␣pass”, which is a valid sequence of Python tokens defining a function. In the situation we’re considering, we will have observed all the tokens up to a certain point and know nothing about the ones after that point.

For instance, at the third observation in the example stream, we have the concatenated string ”def␣f”. If we were to lex/parse this string a traditional approach would return the token sequence DEF NAME, which misidentifies the ”f” as a complete NAME token. As we can see from the rest of the stream, the correct NAME token will be ”foo”.

In general, the next valid strings that can be sampled from the vocabulary are ones that either

-

continue expanding/advancing the NAME currently starting with ”f” (as the full stream in our example does), and/or

-

anything that begins with ”(”, i.e. an LPAR token with regular expression (, and proceeds to specify a valid argument signature.

In the first case, the ”f” can be seen as a partially matched NAME token in Python, and–recalling that its regular expression is [^\W\d]\w*–we can say that it has been matched by the first sub-pattern (i.e. [^\W\d]) in the regular expression. Using Python indexing, we denote these ”partial” tokenizations with tuples like (”NAME”, (0, 1)): i.e. tuples containing the token names and their sub-pattern ranges.

The lex/parse state up to this point in our example ends with the partial token (”NAME”, (0, 1)), and, according to 1., it can be expanded by observing a string starting with any of the following partial matches: (”NAME”, (0, 1)), (”NAME”, (1, 2)), or (”NAME”, (0, 2)). According to 2., the next string could also start with or contain an LPAR. In our example, the next valid vocabulary strings are at least ”d”, ”ef”, ”pass”, ”oo(”, because all of those strings would expand the partially matched NAME, and the last one would also progress the parse state to LPAR. Any other string from the subset of the vocabulary we’ve observed in the complete example stream would result in invalid syntax according to the Python grammar we’re considering.

As we saw in Section 3, the regular expressions used to lex terminal symbols in our example can be viewed as FSMs, and the partial tokens can be mapped to states in the FSM. These connections provide a basis for extending the lexing steps of many standard parsers–e.g. $\operatorname{LALR}(1)$–so that the parsing process can handle sequentially sampled input. The same vocabulary indexing approach used for regular expressions can be extended to this situation by accounting for the conditional use of specific subsets of regular expressions for the lexing in each distinct parse state.

### 4.1 Pushdown Automata Formulation

We want to extend the state-based indexing provided above for regular expressions and their FSMs to CFGs. This can be done using pushdown automata (PDA).

The 6-tuple representation of PDAs (Sipser, 1996, Definition 2.13) is as follows:

###### Definition 2 (Pushdown Automaton).

A pushdown automaton is given by $(Q,\Sigma,\Gamma,\delta,q_{0},F)$, where $Q$, $\Sigma$, $\Gamma$, and $F$ are all finite sets, $\Gamma$ is the stack alphabet, $\delta:Q\times\Sigma_{\epsilon}\times\Gamma_{\epsilon}\to\mathcal{P}\left(Q\times\Gamma_{\epsilon}\right)$, $\Gamma_{\epsilon}=\Gamma\cup\epsilon$, $\epsilon$ is the empty character, and the remaining symbols retain their meanings from the finite automaton definition.

In this case, the vocabulary indexing approach is essentially the same, except that the index now maps states and stack elements to subsets of $\mathcal{V}$: $\sigma:Q\times\Gamma_{\epsilon}\to\mathcal{P}(\mathcal{V})$.

The important algorithmic change to Algorithm 3 occurs on Line $\ref{alg:fsm-sub-sequences}.\ref{delta-inv}$, where the preimage of the transition function is now

$\delta^{-1}(\cdot,v,\cdot)\equiv\left\{(r,g):\delta(r,v,g)\in\mathcal{P}\left(Q\times\Delta_{\epsilon}\right)\right\}.$ | | | |

## 5 Discussion

The vocabulary indexing introduced in this paper removes a prohibitive run-time scaling barrier in guided generation. Naturally, it makes a trade-off between processing and memory, but we believe that the memory costs are relatively low on average and–when not–can be reduced through conventional means. For instance, the (implicitly) definite FSMs used here do not seem entirely necessary, so when the exact representation of the state machine becomes a memory problem, it’s possible that other state machine formulations with better memory requirements could suffice.

It may also be possible to ”lift” the masks computed by this approach into the model $\operatorname{LM}$ in order to avoid some of the costly matrix products common to the deep neural networks and transformer models in general. Basically, the masks are telling us which computations to not perform, but our current formulation applies the masks at the lowest level. By lifting the mask-determining logic further up the forward-pass logic of the model, we may be able to modulate which slices of the model parameters are needed before unnecessarily performing the associated forward-pass operations. This has the potential to save a considerable amount of memory and computation.

## References

- Beurer-Kellner et al. [2023] Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Prompting is programming: A query language for large language models. Proceedings of the ACM on Programming Languages, 7(PLDI):1946–1969, 2023.

- Lew et al. [2023] Alexander K. Lew, Tan Zhi-Xuan, Gabriel Grand, and Vikash K. Mansinghka. Sequential Monte Carlo Steering of Large Language Models using Probabilistic Programs. arXiv preprint arXiv:2306.03081, 2023.

- [3] Rémi Louf and Brandon T. Willard. Outlines: Generative Model Programming. URL https://github.com/normal-computing/outlines.

- Microsoft [2023] Microsoft. Guidance. Microsoft, July 2023. URL https://github.com/microsoft/guidance.

- Poesia et al. [2022] Gabriel Poesia, Oleksandr Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, and Sumit Gulwani. Synchromesh: Reliable code generation from pre-trained language models. arXiv preprint arXiv:2201.11227, 2022.

- Rabinovich et al. [2017] Maxim Rabinovich, Mitchell Stern, and Dan Klein. Abstract syntax networks for code generation and semantic parsing. arXiv preprint arXiv:1704.07535, 2017.

- Radford et al. [2019] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- Rajpal [2023] Shreya Rajpal. Guardrails, July 2023. URL https://github.com/ShreyaR/guardrails.

- Rickard [2023a] Matt Rickard. parserLLM, July 2023a. URL https://github.com/r2d4/parserllm.

- Rickard [2023b] Matt Rickard. R2d4/rellm: Exact structure out of any language model completion., 2023b. URL https://github.com/r2d4/rellm.

- Scholak et al. [2021] Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. arXiv preprint arXiv:2109.05093, 2021.

- Sennrich et al. [2015] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

- Sipser [1996] Michael Sipser. Introduction to the Theory of Computation. ACM Sigact News, 27(1):27–29, 1996.

- Vaswani et al. [2017] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, \Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- Weng [2021] Lilian Weng. Controllable Neural Text Generation, January 2021. URL https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/.

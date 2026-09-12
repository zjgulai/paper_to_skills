<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2603.28677
     paper_id : 2603.28677
     source   : paper2skills-vault/papers/07-NLP-VOC/2603.28677/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

Enhancing User-Feedback Driven Requirements Prioritization Aurek Chattopadhyaya , Nan Niub , Hui Liuc , Jianzhang Zhangd

arXiv:2603.28677v1 [cs.SE] 30 Mar 2026

a

University of Cincinnati, Cincinnati, OH, USA University of North Florida, Jacksonville, FL, USA c Beijing Institute of Technology, Beijing, China d Hangzhou Normal University, Hangzhou, Zhejiang, China b

Abstract Context: Requirements prioritization is a challenging problem that is aimed to deliver the most suitable subset from a pool of candidate requirements.
The problem is NP-hard when formulated as an optimization problem. Feedback from end users can offer valuable support for software evolution, and ReFeed represents a state-of-the-art in automatically inferring a requirement’s priority via quantifiable properties of the feedback messages associated with a candidate requirement.
Objectives: In this paper, we enhance ReFeed by shifting the focus of prioritization from treating requirements as independent entities toward interconnecting them. Additionally, we explore if interconnecting requirements provides additional value for search-based solutions.
Methods: We leverage user feedback from mobile app store to group requirements into topically coherent clusters. Such interconnectedness, in turn, helps to auto-generate additional “requires” relations in candidate requirements. These “requires” pairs are then integrated into a search-based software engineering solution.
Results: The experiments on 94 requirements prioritization instances from four real-world software applications show that our enhancement outperforms ReFeed. In addition, we illustrate how incorporating interconnectedness among requirements improves search-based solutions.
Conclusion: Our findings show that requirements interconnectedness improves user feedback driven requirements prioritization, helps uncover additional “requires” relations in candidate requirements, and also strengthens search-based release planning.

Keywords: strategic release planning, software evolution and maintenance, CrowdRE, requirements interconnectedness, search-based software engineering 1. Introduction Deciding which requirements really matter is a challenging task. Tight time-to-market constraints and practical budget limitations demand effective and efficient requirements prioritization solutions. Modern software industry invests heavily in strategic release planning [1, 2, 3, 4], where a special case is known as the next release problem (NRP). Given a set of candidate requirements, the NRP is concerned with prioritizing a subset of requirements over the others, so that the prioritized part gets implemented and delivered, whereas the remaining requirements would not be included in the upcoming version of the software.
The literature has offered much search-based software engineering (SBSE)
support for the NRP [5, 6, 7, 8, 9, 10, 11, 12]. Assuming there are n candidate requirements, the NRP solutions are bound in a space of 2n . Thus, metaheuristic search guided by some objective function could identify high quality but possibly suboptimal solutions in a computationally tractable way.
The objective function is defined by commonly referring to some attributes of the requirements. For example, Bagnall et al. [5] aimed to maximize the value of different stakeholders’ desired requirements while ensuring that the implementation cost would not exceed a certain limit.
An important attribute relates to the feedback from the end users of the software application. This data source is valuable for understanding what users request, how they like or dislike the software, etc. In a seminal paper, Kifetew et al. [13] introduced ReFeed, a user-feedback driven requirements prioritization method. ReFeed associates user-feedback to requirements, and then computes the requirements priorities based on the extracted properties of the associated feedback. Fig. 1-a illustrates ReFeed which we review in more detail later.
Inspired by ReFeed, we propose a novel approach to enhance user-feedback driven requirements prioritization. Fig. 1-b highlights the key change. Rather than treating requirements as independent entities, we leverage user-feedback messages to group the requirements into topically coherent clusters. In this way, the messages’ influences on the priorities are no longer done at each 2

candidate requirements r1 r2 .
.
.
rn

F1 F2

user-feedback messages

candidate requirements

m m m

r1 .
.
.

...
m

m

rn

(a) ReFeed

m m m

FC1

r2

m m

Fn

user-feedback messages

m m

...
FCk

m m

m m

(b) iReFeed

Figure 1: Compared with ReFeed [13] that associates user-feedback messages to a single requirement, our approach of iReFeed associates messages to a cluster of interconnected requirements. The “i” in iReFeed emphasizes this interconnectedness.

requirement’s level, but at a granularity of interconnected requirements. We call our approach iReFeed to reflect the interconnectedness of the requirements in the same topic cluster.
Recognizing the requirements’ interconnectedness is important for prioritization. Karim and Ruhe [12], for instance, introduced “theme” where the requirements within a theme would be preferably delivered or postponed together. Aydemir et al. [14] further extended it to goal modeling, e.g., by including all the leaf goals of a theme in the next release or excluding them altogether. While theme models a “requires” relation symmetrically (i.e., “req1 requires req2 , and req2 requires req1 ”) [15], the asymmetric “requires” is more common and practically influential to prioritization. Such one-way “requires” manifests itself in different ways, including functional dependency [15], temporal relation [16], and refinement structure [14].
To explore the extent to which iReFeed is conducive to “requires” identifications, we exploit large language models (LLMs)—OpenAI’s ChatGPT 4.5 and 4o in particular—to investigate whether the topic clusters resulted from our approach would lead to the generation of more “requires” pairs.
Surprisingly, compared to feeding ChatGPT with all the requirements once, focusing on the requirements within iReFeed’s topic cluster uncovered additional “requires” pairs. These pairs, in turn, helped improve a state-of-the-art SBSE solution to the NRP.
This paper makes three main contributions.
• We improve ReFeed by incorporating user-feedback from mobile app 3

Table 1: Snippet of the Word Processor Data (adopted from Karim and Ruhe’s work [12]
with the data linked in [17])
Requirement r1 : Create a new file r2 : Open an existing file r3 : Close current file ...

...

...

r49 : Load help file r50 : Search a text in the help file

Stakeholder1 8 8 8

Value Scores Stakeholder2 . . .
9 ...
9 ...
9 ...

Stakeholder4 9 9 1

...
...
...
...

Resource Estimates Design Development QA / Testing 17 22 12 20 25 13 5 1 2

...

...

...

...

...

...

...

...

2

7

...

6

...

3

3

3

1

7

...

6

...

2

2

2

store in interconnecting requirements, and the experiments on four realworld applications show that our approach consistently achieves better prioritization results than ReFeed.
• To examine the usefulness of our approach’s requirements interconnectedness, we prompt ChatGPT on the Word Processor NRP benchmark data [12] and reveal that additional “requires” pairs could be generated with the help of our topic-cluster results.
• We demonstrate the added value of the “requires” pairs through a wellsuited SBSE NRP solution, namely NSGA-II, on the Word Processor benchmark.
We share our study materials and results publicly at https://doi.org/ 10.5281/zenodo.18881986 to facilitate replication. The rest of the paper is organized as follows. Section 2 describes the context of related work in which the current paper is located. Section 3 presents our iReFeed approach.
Section 4 evaluates iReFeed against ReFeed. Section 5 reports our ChatGPT experiments on identifying “requires” pairs. Section 6 illustrates the incorporation of the “requires” pairs into the NSGA-II algorithm. Section 7 draws the conclusion and points out future work directions.
2. Background and Related Work 2.1. SBSE for Automating Requirements Prioritization Deciding which, among a set of requirements, are to be considered first is a strategic process in software development [18, 19]. Traditionally, requirements prioritization is done manually, and hence is effective for only a small number of requirements. The manual approaches include win-win negotiation for deriving a final requirements ranking from an agreement among 4

subjective evaluations by different stakeholders [20], the 100-point method that lets stakeholders distribute their points individually before aggregating the points for ranking the requirements [21], and the analytic hierarchy process (AHP) in which the decision maker compares every pair of requirements in terms of value and cost, feeding into the eigenvalue calculations in order to determine the requirements priorities [22]. Karlsson et al. [23] compared several traditional methods, and showed that AHP was the most promising comthough limited in scalability. For n requirements, AHP requires n×(n−1)
2 parisons in one dimension (e.g., value). Perini et al. [24] reported that AHP suffers scalability issues with requirements sets larger than about 50.
In pursuit of scalable solutions, researchers have explored computational search, especially metaheuristic search. Bagnall et al. [5] were among the first to connect requirements prioritization with SBSE by defining the NRP, i.e., the problem of selecting an optimal set of requirements to be delivered in the software system’s next stable version. Here, optimality means satisfying the stakeholder demands as much as possible, and meanwhile ensuring that there are enough resources to undertake the necessary development. This optimization problem is shown to be NP-hard [5]. To illustrate Bagnall et al.’s NRP formulation, we show in Table 1 a snippet of the Word Processor data [12].
This benchmark contains 50 requirements, some of which are listed in the left column of Table 1. The solution to the NRP is given by the decision vector: ⃗x={x1 , x2 , . . . , x50 }, xi ∈ {0, 1} for 1≤i≤50. In this vector, xi =1 if requirement ri is selected to be part of the next release, and xi =0 otherwise.
To determine if a requirement should be selected or not, one shall consider some critical attributes and their values. In Table 1, columns 2–5 illustrate the stakeholder value scores, and columns 6–9 display the implementationrelated resource Bagnall et al. [5]Pdefined the search objective Pestimates.
50 to: Maximize i=1 xi · value(ri ), subject to 50 i=1 xi · cost(ri ) ≤ B, where value(ri ) is the (weighted) sum of stakeholders’ values placed on ri , cost(ri )
is the total resource estimates of implementing ri , and B is some bound.
Thus, SBSE techniques like hill climbing and simulated annealing could be used to find a high quality, but possibly suboptimal, solution ⃗x [5].
Although Bagnall et al. [5] originally presented a single objective formulation of the NRP, others have adopted multi-objective optimization approaches [8, 10, 12, 7]. Karim and Ruhe [12] showed that maximizing individual requirement’s value and theme-based coherence are conflicting objectives.
Similarly, Finkelstein et al. [8] revealed the trade-offs between maximizing 5

the total value and being fair (e.g., reducing the variance of the number of implemented requirements for each stakeholder).
In a multi-objective optimization problem, we are often interested in finding Pareto-optimal solutions. Suppose there are M objective functions. A decision vector ⃗x is Pareto-optimal if there is no ⃗y that is better than ⃗x in at least one i=1, 2, . . . , M , and no worse than ⃗x in the others. Let us consider two simple solutions about the Word Processor: α selects only r1 to release, and β selects only r2 . With the data given in Table 1, α is better than β in resource consumption (i.e., it costs less to implement r1 than r2 ), and meanwhile, α is no worse than β in stakeholder values. Therefore, in light of α, β is not a Pareto-optimal solution. Among the many search algorithms investigated in the NRP literature, NSGA-II is well-suited [12, 10, 7], exhibits the best run time [25, 26], and performs the best in finding Pareto-optimal solutions [8, 9, 26].
In summary, SBSE offers strategic and automatic ways to solve the NRP.
Extensive empirical studies show that one of the best search-based NRP algorithms is NSGA-II, which we integrate with the knowledge obtained from our iReFeed approach.
2.2. Interrelated Requirements in Prioritization In addition to stakeholder values and effort estimates, the requirements interrelations are among the most influential factors of release planning models used in industry [27, 28]. Carlshamre et al. [15] observed several relation types, such as “AND” and “requires”. The distinction is that “AND” suggests a requires-relation in both ways whereas “requires” indicates the relation in only one way. For example, a printer requires a driver to function, and the driver also requires the printer to function [15]. This is an instance of “AND” relation.
In contrast, emailing a scanned document requires a network connection, but not the other way around [15]. Here, a “requires” relation exists between the pair of requirements. Notably, “requires” embodies other forms than functional dependency, e.g., “auto-logout requires a detected inactivity period, but not the opposite” is a temporal relation [16], and “freight management improvement” is refined by “freight order history logging” [14]. In all these cases, we use ra → rb to denote “ra requires rb ”. The implication is that rb shall not be prioritized lower or later than ra . In case of “AND”, i.e., ra → rb and rb → ra , both requirements share the same priority.
6

Referring to the Word Processor requirements of Table 1, we have r3 → r2 .
Now consider three decisions: β = {r2 }, γ = {r3 }, and δ = {r2 , r3 }, which would respectively release r2 only, r3 only, and both together. Here, γ is not congruent with r3 → r2 , and hence should not be regarded as a valid NRP solution. One can impose the “requires” constraints in a post-processing step of SBSE, or encode them as part of the SMT/OMT solving [14].
In short, “requires” represents a common and critical relation that influences the quality of prioritization outcomes. However, such relations were identified manually in prior studies [15, 14]. We exploit ChatGPT for automating the “requires” identification, and further investigate our approach’s impact on uncovering the “ra → rb ” pairs.
2.3. CrowdRE and ReFeed Not only are requirements themselves interrelated, but they also relate to other data and specifically to the feedback from the end users of the software. Groen et al. [29] defined crowd-based requirements engineering (RE), or CrowdRE, as a semi-automated approach for obtaining and analyzing any kind of user feedback from a pool of current and potential stakeholders. Issue reports, forum discussion, and app reviews are among the common data sources for CrowdRE.
An earlier tool, ChangeAdvisor [30], experimented three topic modeling algorithms (namely LDA, LDA-GA, and HDP) for clustering user feedback into groups expressing similar needs. ChangeAdvisor adopted HDP as it provided a good trade-off between quality and execution time. A subsequent tool, CLAP [31], trained a random forest classifier to differentiate user reviews into categories like feature and bug. CLAP further involved clustering, e.g., by using DBScan to group the user reviews reporting the same bug in one cluster. CLAP finally employed another random forest to classify whether a cluster would be high priority or low priority. Recently, Radiation [32] was introduced to recommend the removal of certain user interface functionality.
Radiation applied HDP to cluster user reviews, followed by the use of a random forest classifier to determine what to delete based on quantifiable properties such as review numbers, ratings, and sentiments.
Although other CrowdRE methods exist, the above tools illustrate that topic modeling and clustering are effective in handling the high volume of feedback data and that the review aggregates are valuable for software evolution [33]. Nevertheless, ChangeAdvisor traces user feedback to source code, 7

CLAP prioritizes user reviews instead of candidate requirements, and Radiation recommends to remove existing user interface features but falls short of potentially enriching the software with better features.
Kifetew et al. [13] pioneered the use of feedback in automatically prioritizing requirements, which inspires our research. As outlined in Fig. 1-a, ReFeed comprises of three major steps: associating feedback with each requirement, extracting feedback properties, and inferring the priority of requirements. The feedback-requirement associations are established on the basis of textual similarity. ReFeed computes the Jaccard similarity between each feedback message and every requirement, and then forms associations if the similarity is greater than a threshold. Similarity-based associations are common in contemporary CrowdRE tools [30, 32].
ReFeed extracts quantifiable properties from user feedback that are relevant for prioritizing the requirements. These properties are resulted mainly from sentiment analysis and speech-acts analysis [13]. Let N be the total number of sentences in a feedback message, and Npos (or Nneg ) be the number of sentences with positive (or negative) sentiment. Due to neutral sentiment, Npos + Nneg does not always equal to N . GivenPa user-feedback message, neg P N

SentimentScore

N

(Sentences[n])

SentimentScore

(Sentences[n])

pos neg and pos = n=1 = n=1 Nneg Npos quantify the message’s negative and positive sentiments respectively. ReFeed uses StanfordCoreNLP to compute SentimentScore at the sentence level [13].
The speech-acts analysis is aimed to calculate the extent a feedback message conveys feature request as opposed to other intentions like bug fix. To that end, a message’s intention score, int, is calculated as the average of the intentions of its sentences. An int=1 implies the message is a feature request, whereas a lower int score suggests that the feedback is less about making a feature request.
Having associated each requirement r via F with a set of feedback messages, Kifetew et al. [13] inferred the requirement’s priority P in ReFeed as1 :

P|F | Pr =

i=1 [sim(r, F [i]) ∗ (negF [i] + posF [i] + intF [i] )]

|F |

1

(1)

We ignore the severity measure from [13] as it is based purely on the negative sentiment of feedback. As can be seen in equation (1), the negative sentiment has already been considered in ReFeed.

8

where F [i] denotes the feedback message at position i in the set mapped by F , and sim(r, F [i]) represents the textual similarity value between the requirement r and the message F [i]. If |F |=0, then no feedback is associated with r and that requirement’s priority is set to be 0. Intuitively, equation (1)
assigns a higher priority to r if its feedback is more similar to the requirement, embodies stronger sentiment, and has a greater intention to convey feature request.
In essence, ReFeed offers a fully automatic way to prioritize requirements based on user feedback. Improving ReFeed with requirements interconnectedness is precisely the focus of our research.
3. iReFeed We present in Fig. 2 an overview of our iReFeed approach. The figure elaborates our newly proposed steps of performing topic modeling on user feedback and then grouping the requirements into topically coherent clusters. Even though iReFeed shares with ReFeed the overall steps of associating feedback with requirements, extracting feedback properties, and inferring requirements priorities, important differences exist. We discuss these differences, together with key design rationales and implementation details, in this section.
Step 1: Our construction of clusters of interconnected requirements consists of topic modeling of user feedback and then grouping the candidate requirements by using the resulting topics. The primary reason of applying topic modeling to feedback is due to the quantity of the data. Typically, there

user feedback

candidate requirements

1a. apply

topic topic topic

topic modeling

1b. group requirements into topic clusters

2. associate feedback with req.s 3. extract feedback properties 4. infer req.s priorities cluster cluster cluster

Figure 2: Overview of iReFeed processing pipeline.

9

are tens to hundreds of candidate requirements, which can be inadequate for insightful topics to emerge in the latent space. In contrast, tens or hundreds of thousands of feedback messages can be readily collected for a software application, creating a sizeable critical mass for applying topic modeling.
We implement two topic modeling methods in iReFeed: LDA and BERTopic, due to their viable applications in CrowdRE [34, 35]. A salient distinction is that LDA uses the corpus from the tokens of the user feedback data, while BERTopic exploits a pre-trained transformer model with data crawled from external sources like Wikipedia. The literature has not recommended any common number of topics to be selected [34]. Thus, we experimented with several numbers of topics, and selected 20 as an appropriate one to use in our current work. Previous studies [36, 37] have also found 20 topics to be appropriate. We trained LDA on user reviews with 15 passes to ensure convergence, and then converted each candidate requirement into a bag-of-words representation to cluster them based on the highest topic distributions. For BERTopic, we applied UMAP (number of components=5, number of neighbors=15, minimum distance=0) for dimensionality reduction and HDBSCAN (minimum number of samples=10) for clustering user feedback into topics.
We then converted the requirements into sentence-BERT embeddings and assigned them to the most relevant topic cluster based on the embeddings’ cosine similarities. Note that the feedback topics are important intermediate results, because they serve as a means to form clusters of interconnected requirements.
Step 2: As in [13, 32], we exploit textual similarity to associate feedback with requirements. In [38], for instance, Palomba et al. used the cosine similarity threshold of 0.6 for the purpose of associations; yet, Nayebi et al [32] adjusted the threshold to 0.65 in order to achieve a more accurate matching level.
Similarly, we slightly increased the Jaccard similarity threshold of 0 [13]
to the cosine similarity of 0.1 to obtain associations in iReFeed and in our ReFeed implementation.
Different from ReFeed, the associations are established at a requirements cluster level in our work. Consider an example of two requirements and three feedback messages. If their cosine similarities are: sim(r1 , m1 )=0.15, sim(r1 , m2 )=0.2, sim(r1 , m3 )=0.06, sim(r2 , m1 )=0.07, sim(r2 , m2 )=0.22, and sim(r2 , m3 )=0.13, then under a 0.1 threshold, ReFeed would associate {r1 } with {m1 , m2 } and associate {r2 } with {m2 , m3 }. In iReFeed, if r1 and r2 are in the same cluster, then we link {r1 , r2 } with {m1 , m2 , m3 } by taking the 10

union of the messages associated with the cluster’s constituent requirements.
In this way, for instance, the influence of m1 on prioritizing r2 is incorporated in iReFeed but neglected in ReFeed. The influence, as will be seen later, does get moderated by the similarity score.
Step 3: To extract quantifiable properties of the feedback messages, we perform sentiment analysis via the Stanza library from the Stanford NLP package [39]. Similar to ReFeed, we compute negative sentiment and positive sentiment separately. These computations are done first at a sentence level, and then aggregated to a message level.
Our calculation of an intention score adopts a review classification method, rather than the user feedback ontology used in [13]. While the ontology may need to be updated from one domain to another, a classifier trained on diverse review data can offer an automatic alternative to recognize the type of feedback. To that end, we built a random forest classifier on top of the datasets shared by Scalabrino and colleagues [31]. The datasets contain 725 reviews from 14 apps, and the reviews are labeled with six classes: feature, bug, performance, usability, security, and energy. Most of the reviews belong to the categories of bug or feature. We developed the random forest with a train-test split ratio of 80-20, and achieved an accuracy of 79%. This performance is comparable to the random forest classifiers used in the CrowdRE literature [31, 32]. Note that we applied the same sentiment analysis and content classification to ReFeed and iReFeed, so there is no difference in extracting feedback properties.
Step 4: The chief difference in inferring requirements priority is illustrated in Fig. 1-b. In iReFeed, the feedback mapping, FC , is no longer from a single requirement, but from a cluster. Therefore, a priority of a requirement r in iReFeed is defined as:
P|FC | Pr =

i=1 [sim(r, FC [i]) ∗ (negFC [i] + posFC [i] + intFC [i] )]

|FC |

(2)

where FC represents the feedback mapping from the cluster that r is in, and the rest of the notations have the same interpretations in equation (1).
Orthogonal to LDA and BERTopic, we design a mechanism to weigh in the coherence of iReFeed’s requirements clusters. We call it LDA-C or BERTopicC, and define a requirement’s priority in the C variants as:

11

Table 2: Characteristics of RQ1 Datasets app [source of release notes]
Discord [40]
Microsoft 365 Word [41]
Webex [42]
Zoom [43]

# of prioritization instances 18 12 29 35

# of requirements 373 295 360 951

requirements time period July 2020–March 2025 July 2018–Dec 2024 Feb 2022–July 2024 Feb 2022–March 2025

# of userfeedback messages 368,367 143,547 17,829 62,074

feedback time period Jan 2020–March 2025 Jan 2018–Dec 2024 Jan 2022–July 2024 Jan 2022–March 2025

P|FC | Pr =

i=1 [α(FC ) ∗ sim(r, FC [i]) ∗ (negFC [i] + posFC [i] + intFC [i] )]

|FC |

(3)

where the newly added term α(FC ) = min(1, average pairwise similarity of ∀ri , rj ∈ FC ). This factor boosts the priorities of the requirements in an internally coherent cluster; however, due to the min( ) function, α(FC ) does not penalize the poorly coherent clusters. In fact, when α(FC )=1, equation (3) is reduced to equation (2), implying that requirements cluster’s coherence is expected to influence the prioritization results in a more general sense.
4. Evaluating Prioritization Results The research question that we investigate in this section is RQ1 : “How does iReFeed compare to ReFeed in delivering requirements prioritization results?” We treat ReFeed as a state-of-the-art method of using feedback to prioritize candidate requirements, and compare ReFeed with the four variants of iReFeed: LDA, BERTopic, LDA-C, and BERTopic-C.
4.1. Datasets and Metrics To answer RQ1 , we constructed 94 requirements prioritization instances from four real-world software applications: Discord, Microsoft 365 Word, Webex, and Zoom. Table 2 lists some characteristics of our evaluation datasets.
We relied on the release notes maintained by the software vendors themselves to identify requirements. In addition, we collected the release time of each requirement. Finally we combined the requirements from two consecutive release periods to form a prioritization instance.
One instance from the Zoom dataset is shown in Table 3. This instance has 71 total requirements: 19 were released in February 2025 and the other 52 were released in March 2025. The release timestamps give rise to the prioritization’s ground truth in an authoritative way. When the 71 requirements of Table 3 are prioritized, the ground truth is to select the 19 requirements.
Thus, we note the size of the ground truth here as k=19 and the size of 12

Table 3: Excerpts from One Prioritization Instance of the Zoom Dataset (the top 19 requirements were released in February 2025, and the bottom 52 requirements were released in March 2025)
# Feature Description In Ground Truth?
1 Meeting participants can request access to recordings directly from the meeting card or recording Yes link without having to send separate messages through chat, email, or calls. Meeting hosts receive these requests through in-product notifications and emails, which direct them to a viewing page where they can manage access permissions. Hosts can view all requesters on the share modal, grant or deny access, and manage existing permissions for specific users. The feature respects all security settings, including authentication requirements and password protection.
2 After a meeting ends, a dynamic pop-up appears, based on the assets utilized during the meeting, Yes directing users to the meeting details page. Here, they can access available meeting assets such as meeting recording, summary, continuous chat log, as well as any content shared during the session, such as whiteboards, links, and notes.
...
...
...
...
...
...
...
...
...
19 Users can view a complete list of meeting participants directly within the primary meeting card interface.
Yes The participant list appears in the full meeting page, providing clear visibility of all attendees.
1 Hosts and co-hosts can move participants directly from the waiting room to designated breakout No rooms without requiring them to enter the main meeting room first. Additionally, they can lock breakout rooms to prevent participants from returning to the main session. This feature works for both signed-in Zoom users and guests.
2 Webinar hosts can access AI-generated high-level summaries of webinar discussions, serving No as an automated note-taking solution for both live and recorded sessions. Hosts can distribute these summaries through the webinar’s follow-up email workflow or their preferred communication channels. Account owners and admins can control AI summary settings, including usage permissions and notification preferences at both account and group levels.
...
...
...
...
...
...
...
...
...
52 The Cobrowse feature now supports web chat engagements, enabling agents to assist customers No more efficiently while maintaining privacy. When customers need help with website forms or navigation, agents can request permission to view only the specific webpage where the issue occurs. This enhancement allows for more precise, real-time support while protecting sensitive information through built-in privacy safeguards. This feature must be enabled by Zoom.

the instance as n=71. This instance’s size (71>50) illustrates the scalability challenge of manually prioritizing requirements [24], echoing the importance of automatic solutions in industrial settings.
Our collection of user feedback data was done through a Google play store scraper [44]. Specifically, we extracted the reviews for the four software applications in the same time ranges as the release periods of the requirements.
However, the reviews were collected one release cycle prior to the requirements, as shown in Table 2. For example, we extracted a total of 62,074 user reviews for Zoom from Jan 2022 to March 2025, but the Zoom requirements were collected from Feb 2022 to March 2025. This allowed us to ensure time sensitivity in evaluating ReFeed and iReFeed. In particular, we used the reviews up until the beginning of a prioritization instance. Additionally, to ensure relevant feedback, we used at most the two years of reviews preceding each prioritization instance. When fewer than two years of reviews were available, we used all available reviews. For example, we used the Zoom reviews from Jan 2023 to Jan 2025 for the instance shown in Table 3, because the instance considered releasing the requirements in Feb 2025 and assuming 13

Legends:

Figure 3: Answering RQ1 with average performances of ReFeed and the four variants of iReFeed: LDA, LDA-C, BERTopic, and BERTopic-C.

the availability of the user reviews up till Jan 2025 was sensible to us.
Given the ground truth, we measure the qualities of ReFeed and iReFeed prioritization results by computing recall (R), precision (P), F1 -score (F1 ), and F2 -score (F2 ). These metrics quantify how much a prioritization solution overlaps the ground truth. Intuitively, recall indicates how complete the prioritization solution is whereas precision implies how noisy it is. Although F1 = 2·R·P symmetrically represents both recall and precision in one metric, R+P Berry et al. [45] argue that, for automated tools supporting RE tasks in the context of large-scale software development, recall is more important than 5·R·P which weighs recall twice as precision. Therefore, we also consider F2 = 4·P +R 14

important as precision.
As shown in equations (1)–(3), ReFeed and iReFeed rank candidate requirements in an ordered list according to requirements’ priority scores. We thus investigate the qualities of the ordered list at five cutoff points: k, k ± (10% · n), and k ± (20% · n). Recall that k is the ground truth size (e.g., k=19 in Table 3) and n is the instance size (e.g., n=71 in Table 3). While top-k represents a cutoff in experimental settings where the exact prioritization number is known, the others mimic practical scenarios. Taking Table 3’s instance as an example, the selections of the top-ranked 5 (i.e., 19 − 20% · 71)
or 12 (i.e., 19 − 10% · 71) requirements reflect aggressive prioritization decisions, whereas relaxed decisions are represented by choosing the top-ranked 26 (i.e., 19 + 10% · 71) or 33 (i.e., 19 + 20% · 71) requirements.
To evaluate statistical significance, we performed the Wilcoxon signedrank test across all cutoff points for each dataset. For each metric and iReFeed variant, we compared the scores against ReFeed. The resulting Wilcoxon p-values of the four iReFeed variants were then combined at each cutoff point using Fisher’s method.
4.2. Results and Analysis We plot the performances of ReFeed and iReFeed in Fig. 3. The plots are organized horizontally by the four performance evaluation metrics, and vertically by the datasets. Each sub-figure shows the averages across the prioritization instances in that dataset. The combined significance levels are also reported in Fig. 3. Overall, a clear trend observed in Fig. 3 is that iReFeed consistently outperforms ReFeed. This shows that clustering requirements into topically coherent groups positively influences prioritization, compared with treating requirements as being independent from each other in the prioritization process.
Using ReFeed as the baseline, we notice from Fig. 3 that the improvements of recall are more prominent in Discord, and the precision improvements are more salient in Discord and Webex. We speculate a possible reason might be due to the user feedback’s accurate matching and broad coverage of the candidate requirements in these datasets; however, testing the hypothesis requires future work. Nevertheless, a general trend of Fig. 3 is that LDA-C performs better than LDA, and BERTopic-C performs better than BERTopic. Therefore, our results suggest that integrating cluster’s internal coherence into ranking requirements further enhances prioritization qualities.
15

When paying attention to different cutoff points, we identify the precision changes in Fig. 3 as encouraging for iReFeed. As more top-ranked requirements are analyzed, it is not surprising that recall keeps increasing.
For iReFeed, precision peaks at top-k in most cases. If the prioritization decision is to continue including more requirements, then iReFeed drops precision more than ReFeed’s precision decreases. This shows that the user feedback’s influences on prioritization are more accurate to the top-k ranked requirements than to further ranked requirements. In another word, iReFeed achieves a good and balanced performance when the number of prioritized requirements is near the ground-truth size k.
Taking the cutoff at k and the weighted F2 as a performance indicator, we note that LDA-C performs better than BERTopic-C in Microsoft 365 Word. LDA-C’s performances in Discord, Webex, and Zoom are comparable to BERTopic-C’s. Thus, among the four variants of iReFeed, we recommend LDA-C, though it is somewhat surprising that the pre-trained BERT model with extensive external data does not definitively outperform a locally operated LDA in user-feedback driven requirements prioritization. Through the above analyses, we conclude that iReFeed consistently outperforms ReFeed.
We provide an example below to shed light on the differences between ReFeed and iReFeed. Consider two requirements and three feedback messages of a Microsoft 365 Word instance.
• r1 : “Use @mentions in comments to let co-workers know when you need their input.” • r2 : “Tired of being locked out of your document with macros? Now your docm files on OneDrive for Business allow simultaneous editing by multiple authors.” • m1 : “This app is super easy for reading and sharing word files. It has all the important features required for writing but steps for using it should be mentioned or outline of all the editing options should be mentioned for beginners. For me, it saved a lot of time. Good app. Thank you” • m2 : “Works great with Office 2016 and OneDrive across multiple devices/opearting systems. Can’t do without it.” • m3 : “I absolutely love Microsoft’s Word app. I’m able to do so many important things like create documents that are truly professional, use 16

r1

m1

r1

ReFeed r2

m3

m2

r2

iReFeed

m1 m3

m2

Figure 4: Examples from a Microsoft 365 Word instance illustrating the differences between ReFeed (left) and iReFeed (right).

the many templates to make the perfect letter or resume, upload document files in Word format even from the internet and edit, and if unlocked files I can highlight text and add notes. It allows me to be the author of my own document creations and either lock it, share it via other applications like Facebook or Twitter, and gives me the option to allow for others to participate in the writing and editing, for instance, a petition, and so much more.” Fig. 4 depicts the associations established in ReFeed and iReFeed for the above entities. Although lexical clues can be directly spotted, e.g., “mention” between r1 and m1 , iReFeed is able to recognize deeper semantic connections.
For example, m1 touches upon editing options and hence is linked to the simultaneous editing of r2 . Similarly, m3 advocates collaborative writing which is what r1 tries to facilitate. In fact, the cluster containing r1 and r2 resulted from iReFeed renders collaborative editing options as a topic. Consequently, iReFeed establishes richer associations that are missed in ReFeed.
4.3. Threats to Validity The core constructs of our experiments reported in this section are requirements prioritization instance and solution quality. Each instance contains the requirements that shall be prioritized to release and those that shall not. The software projects’ own release histories allow for the ground truth to be defined authoritatively. Although the instance can be easily extended to a series of multiple releases, we are interested in only two consecutive release periods, as this is also the focus of ReFeed. We measure the prioritization solution’s quality with well-known metrics at various cutoff points. We thus believe the threats to construct validity are low.
17

The internal validity of experiments could be affected by our choice of extracting review data with a one-release-cycle buffer ahead of requirements data. This avoids solving a prioritization instance without any review data.
A related decision is our use of cumulative reviews, rather than selecting only recent ones. While our rationale is to maintain a broad time range to capture both short-term and long-term trends of requirements evolution and integration of user feedback, determining the optimal amount of review data to exploit in prioritization is beyond the scope of our current work. Keep in mind that, for each prioritization instance, our experiments of ReFeed and iReFeed have used the same amounts of review and requirements data.
As a result, the comparisons reported in this section reflect the differences between ReFeed and iReFeed.
The results are based on the four real-world datasets that we have curated. Due to the threats to external validity, it would be too strong to claim iReFeed’s better performance over ReFeed on other prioritization instances (e.g., the software applications outside the video conferencing, word processing, and social networking domains). In terms of transparency and reliability, we share our datasets, along with our implementations and experimental results, in https://doi.org/10.5281/zenodo.18881986 to facilitate replication and expansion.
5. Uncovering “Requires” Pairs Our RQ2 examines: “To what extent is iReFeed conducive to identifying the ‘requires’ relations?” As discussed in Section 2.2, “requires” relations greatly shape prioritization decisions. However, identifying such relations remains largely a manual task [15, 14]. With the advent of LLMs, researchers have exploited them to automate a wide variety of RE tasks. Specifically, LLMs like OpenAI’s ChatGPT have shown promises in detecting the relationships among requirements: inconsistency (e.g., “the machine shall always offer coffee” is in tension with “the machine shall not offer Cappuccino as a beverage option”) [46], traceability (e.g., an “error decoding” requirement is traced to a “control and monitoring” feature) [47, 48], satisfaction (e.g., whether a mobile app’s specification satisfies GDPR’s data withdrawal guidance) [49], just to name a few.
Motivated by this thrust of research, we are interested in LLMs’ capabilities of automatically uncovering “requires” relations for the requirements prioritization task. We chose the Word Processor benchmark to experiment 18

because the 65 “ra → rb ” pairs defined on top of the 50 requirements [17]
would serve as the ground truth. In terms of LLMs, we tested ChatGPT 4.5 and ChatGPT 4o because these are the latest models from OpenAI, though ChatGPT 4.5 is generally considered an improvement over ChatGPT 4o with improved reasoning and reduced hallucination rates [50]. We developed a direct prompt and ran it through the LLMs’ web interfaces in May 2025: A “requires” relation between two requirements (req_x and req_y) is defined as:
req x requires req y for the purpose of software release, but not vice versa.
Identify and output all the “requires” pairs from the requirements provided below, using the format: req_x –> req_y. {requirements} In this prompt the objective of “software release” is made explicit, the instruction of “identifying all the ‘requires’ pairs” is given, and the output format of “req_x → req_y” is specified.
Identifying requires relations involves understanding the context of different requirements. Although LLMs such as ChatGPT have large context windows, analyzing all requirements at once can still confuse the model with too many competing semantic cues, especially when the requirements are topically diverse. This motivates us to utilize clustering to improve focus by grouping requirements that are topically similar based on the user feedback, allowing the LLM to reason more effectively within a focused group of interconnected requirements. We fed all the 50 Word Processor {requirements} once as the baseline. The iReFeed prompt, however, would feed only the {requirements} of one cluster at a time. To produce the clusters for the Word Processor requirements, we first applied LDA-C to 146,310 reviews (Jan 2018–March 2025) of Microsoft 365 Word, as the answers to RQ1 showed that LDA-C was among the best to implement iReFeed. We then used the user-review topics to group the 50 Word Processor requirements into clusters. This resulted in 7 clusters, and hence we ran the iReFeed prompt 7 times, each time with the {requirements} from one and only one cluster.
We made sure to clear ChatGPT’s history while prompting. Therefore, the baseline and iReFeed promptings were independent from each other, and so were the 7 times that the iReFeed prompts were executed. Finally, we treated iReFeed as a supplement to the baseline prompting rather than its replacement. The reason was that iReFeed prompt did not cross cluster boundaries, so the “requires” of two requirements in different clusters would not be detected by the iReFeed prompts. We aggregated iReFeed prompts’ results into the baseline results by eliminating duplicates, and referred to 19

Table 4: Answering RQ2 with Automatically Identified “Requires” Pairs for the Word Processor Benchmark [17]

LLM

prompting

ChatGPT 4.5 ChatGPT 4o

baseline iReFeed combined baseline iReFeed combined

# of distinct pairs 18 26 37 32 32 56

R

P

F1

F2

0.08 0.11 0.17 0.08 0.11 0.17

0.28 0.27 0.30 0.16 0.22 0.20

0.12 0.15 0.22 0.10 0.14 0.18

0.09 0.12 0.19 0.09 0.12 0.17

the aggregated “requires” pairs as the combined results.
The results’ accuracies are summarized in Table 4 where the best performances are displayed in boldface. While the complete results are shared in https://doi.org/10.5281/zenodo.18881986, we make a couple of observations. First, ChatGPT 4.5 generally outperforms ChatGPT 4o, which is not surprising given the advanced reasoning capabilities of ChatGPT 4.5 [50].
Second, and more importantly, regardless of the LLMs, iReFeed does help uncover additional “requires” pairs that the baseline prompting fails to identify. This is encouraging in that, by zooming in only the requirements inside a single cluster, subtle relationships could be recognized. In a way, iReFeed prompting decomposes a requirements set into smaller chucks, which slows down ChatGPT’s reasoning and improves identification accuracies. We conclude that iReFeed adds value in ChatGPT’s auto-generation of “requires” pairs.
Admittedly, the overall accuracies of Table 4 are low. On one hand, interrelating requirements for prioritization may be a difficult task that requires LLM fine-tuning or advanced prompting like few-shot and chain-of-thought.
On the other hand, the data leakage concerns appear to be alleviated. Data leakage means the LLM merely memorizes the data instead of doing the reasoning. Given that the Word Processor dataset [17] was released in 2016 and ChatGPT 4.5 and 4o were released in February 2025 and May 2024 respectively, little leakage seemed to have occurred in our RQ2 experiments.
6. Integrating iReFeed into SBSE We explore in RQ3 : “How could iReFeed be integrated into SBSE approaches to solving the NRP?” To that end, we illustrate in this sec20

tion a concrete way to fuse the “requires” pairs uncovered by iReFeed with NSGA-II, a well-suited SBSE solution to the NRP. Our illustration builds on the Word Processor dataset [17]. In particular, we take the 50 requirements and use NSGA-II to search for solutions for a bi-objective optimization problem: Maximize the weighted sum of all the stakeholders’ value scores (cf.
columns 2–5 of Table 1) while at the same time minimizing the development resource estimates (i.e., column 8 of Table 1).
Our baseline implementation of NSGA-II follows the same algorithmic tuning as introduced by Finkelstein et al. [8]:
• The initial population is set to 200.
• The experimental execution of NSGA-II is terminated after 50 generations, i.e., after 10,000 evaluations.
• The genetic approach uses the tournament selection (with tournament size of 5), single-point crossover, and bitwise mutation.
• The probability of the crossover operator being applied is set to Pc =0.8, 1 and the probability of the mutation operator is set to Pm = # of requirements 1 = 50 =0.02.
The result of such a search is a Pareto front. Each element on this front is a candidate solution to the NRP. All solutions on the Pareto front are non-dominated: no other solution on the front is better according to both objectives. The Pareto front thus represents a set of “best compromises” between the objectives that can be found by the search-based algorithm. As an illustration, we show the search results of the baseline NSGA-II run in Fig. 5 in which no “requires” pairs are taken into account. In this figure, the “×” and “−” nodes denote the Pareto-optimal solutions found by the baseline NSGA-II algorithm.
To integrate iReFeed into NSGA-II, we introduce dependency value (Dvalue). We utilize the automatically identified requires pairs from ChatGPT 4.5 combined results in RQ2 to compute the D-value. Let D be the set of all the requires pairs identified by ChatGPT 4.5 in RQ2, where each pair “ra → rb ” indicates that requirement a requires requirement b. Let counti be defined as the total number of requires relations where i appears on the right hand side of a requires pair (counti = |{ (ra → rb ) ∈ D, where b=i}|).
The D-value for a requirement i is defined as follows:
21

counti D D-Value for iReFeed is incorporated as a third objective of maximizing Dvalue. We maximize D-value as an objective in iReFeed giving importance to the requirements with higher dependencies for selection. The D values have a short range and are skewed in nature, so we apply different transformations and normalization techniques to explore alternative representations of the dependency values. We introduce five variants: D-value, log transformed D value, power transformed D-value, z-score normalized D-value, and inverse D-value. The formulas for each of the variants are reported in Table 5. For the log transformation, we apply loge (1 + Di ) to avoid undefined values when Di = 0. When applying the z-score normalization, we shift each value by the magnitude of the most negative z-score to ensure that all the values are non-negative, while preserving the relative ordering between the values. The inverse D-value serves as a contrasting variant that prioritizes requirements with fewer dependencies. This enables us to evaluate how the search is D-valuei =

Figure 5: One run of RQ3 with search results from the baseline NSGA-II and the iReFeed D-value variant NSGA-II.

22

Table 5: Dependency Value (D-value) Variants

Variant D-value (Di )
Log-transformed D-value Inverse D-value Power-transformed D-value Z-score normalized D-value

Formula counti |D| loge (1 + Di )
1 − Di Di0.5 Di − µ σ

Table 6: RQ3: average across 10 runs of percentage of solutions in reference pareto front

Variants Baseline NSGA-II D-value 80.65 Log transformed D-value 85.17 Inverse D-value 95.98 Power transformed D-value 83.84 Z-score normalized D-value 82.72

iReFeed NSGA-II 98.18 94.28 66.28 98.1 97.45

affected by the dependency values.
In Fig. 5, the “+” and “−” nodes denote the Pareto-optimal solutions found by the iReFeed NSGA-II search for one of the runs (run 5) of the D value variant . The shared solutions are marked by “−”. To evaluate the solutions found by baseline NSGA-II and iReFeed NSGA-II, we construct a reference Pareto front informed by the work of Finkelstein et al. [8]. In a nutshell, the reference Pareto front is the result of merging the solutions from different search algorithms and then excluding the solutions that are dominated by some other merged solutions. Since we are solving for the bi-objective optimization problem, we only consider the value and cost pairs as solutions when evaluating the baseline NSGA-II and iReFeed NSGA-II solutions. In Fig. 5, for example, although the solution A: (value=4097, cost=175) is Paretooptimal when only the baseline NSGA-II is considered, A becomes dominated by one of the iReFeed NSGA-II solutions B: (value=4713, cost=167). The reason is that B’s value is higher than A’s value, and at the same time, B’s cost is lower than A’s cost. In the face of B, A is no longer on the reference Pareto front. Therefore, the reference Pareto front denotes the best available approximation to the real Pareto front [8].
23

(a) D-value

(b) Log transformed D-value

(c) Inverse D-value

(d) Power Transformed D-value

(e) Z-score normalized D-value

Figure 6: RQ3: Comparison of each iReFeed variant against the Baseline across 10 runs

The measure used by Finkelstein et al. [8] to compare different search algorithms is to count the number of individual algorithm’s solutions on the reference Pareto front, namely the solutions that are not dominated by the reference Pareto front. We report the average results for each iReFeed variant against the baseline across 10 runs in Table 6. The results across each run for all the iReFeed variants against baseline is shown in Figure 6.
We observe from Table 6 and Figure 6 that for all iReFeed variants except the inverse D-value variant, iReFeed search solutions represent a higher 24

proportion of solutions in the reference pareto front. The greater share, according to Finkelstein et al. [8], suggests the better performance of iReFeed NSGA-II compared to baseline NSGA-II. Thus, we conclude positive findings of RQ3 with iReFeed’s superiority over the baseline SBSE solution. We attribute this to the baseline’s obliviousness to requirements interconnectedness.
The inverse D-value variant is particularly interesting. By inversing the importance of dependency values, this variant priorities requirements with less dependency on other requirements. As a result, its performance drops significantly compared to baseline. This variant further strengthes the importance of integrating D-value into the iReFeed search.
Our study presented in this section has several limitations. From the construct validity perspective, we use an individual search algorithm’s solutions shared within the reference Pareto front to evaluate that algorithm’s performance. While Finkelstein et al. [8] have used the same evaluative construct, there exist other measures, such as convergence and hypervolume [26], which could be considered in future assessments. We believe the internal validity is high, as the search results reported in this section are based on the same parameter setting for both the baseline and the iReFeed NSGA-II algorithms. Thus, the different shares of the reference Pareto front must be caused by the “requires” pairs exploited in iReFeed NSGA-II. Our results may not generalize to other datasets or SBSE algorithms—a threat to the external validity. With the encouraging initial results obtained here, we are positive that iReFeed is amenable to be integrated into other metaheuristic search techniques like the two-archive algorithm [8] or even hyper-heuristic search methods [26].
7. Conclusion Modern software engineering often injects agile’s iterative and incremental development. Requirements are continuously delivered, though challenges exist in terms of lack of long-term planning, poor resource management, and scope creep [51]. User feedback provides an invaluable source for supporting agile software project’s strategic releases. In this paper, we have presented a novel approach to enhancing user-feedback driven requirements prioritization. Our main novelty lies in the clustering of requirements based on the topics emerged from user reviews. Our evaluations show that iReFeed not only outperforms ReFeed in making prioritization decisions, but also is con25

ducive to enabling ChatGPT to identify requirements dependencies. The integration of iReFeed and NSGA-II, to the best of our knowledge, is the first to synthesize CrowdRE and SBSE for requirements prioritization.
Our future work includes carrying out experimentation on more datasets, investigating the optimal amount of feedback data to use, testing advanced prompting methods like few-shot and chain-of-thought, and guiding metaheuristic or hyper-heuristic search proactively with the “requires” pairs.
References [1] K. Marner, S. Wagner, G. Ruhe, Release planning patterns for the automotive domain, Computers 11 (6) (2022) 89:1–89:26.
URL https://doi.org/10.3390/computers11060089 [2] D. Spinellis, The strategic importance of release engineering, IEEE Software 32 (2) (2015) 3–5.
URL https://doi.org/10.1109/MS.2015.54 [3] M. Nayebi, G. Ruhe, Analytical product release planning, in: C. Bird, T. Menzies, T. Zimmermann (Eds.), The Art and Science of Analyzing Software Data, Morgan Kaufmann, 2015, pp. 555–589.
URL https://doi.org/10.1016/b978-0-12-411519-4.00019-7 [4] G. Zorn-Pauli, B. Paech, T. Beck, H. Karey, G. Ruhe, Analyzing an industrial strategic release planning process - a case study at Roche Diagnostics, in: Proceedings of the 19th International Working Conference on Requirements Engineering: Foundation for Software Quality (REFSQ), Essen, Germany, 2013, pp. 269–284.
URL https://doi.org/10.1007/978-3-642-37422-7\_19 [5] A. J. Bagnall, V. J. Rayward-Smith, I. M. Whittley, The next release problem, Information & Software Technology 43 (14) (2001) 883–890.
URL https://doi.org/10.1016/S0950-5849(01)00194-X [6] M. S. Feather, T. Menzies, Converging on the optimal attainment of requirements, in: Proceedings of the 10th IEEE Joint International Conference on Requirements Engineering (RE), Essen, Germany, 2002, pp.
263–272.
URL https://doi.org/10.1109/ICRE.2002.1048537 26

[7] Y. Zhang, M. Harman, S. A. Mansouri, The multi-objective next release problem, in: Proceedings of the 9th Annual Conference on Genetic and Evolutionary Computation (GECCO), London, UK, 2007, pp. 1129– 1137.
URL https://doi.org/10.1145/1276958.1277179 [8] A. Finkelstein, M. Harman, S. A. Mansouri, J. Ren, Y. Zhang, A search based approach to fairness analysis in requirement assignments to aid negotiation, mediation and decision making, Requirements Engineering 14 (4) (2009) 231–245.
URL https://doi.org/10.1007/s00766-009-0075-y [9] J. J. Durillo, Y. Zhang, E. Alba, M. Harman, A. J. Nebro, A study of the bi-objective next release problem, Empirical Software Engineering 16 (1) (2011) 29–60.
URL https://doi.org/10.1007/s10664-010-9147-3 [10] X. Cai, O. Wei, Z. Huang, Evolutionary approaches for multi-objective next release problem, Computing and Informatics 31 (4) (2012) 847–875.
[11] J. Xuan, H. Jiang, Z. Ren, Z. Luo, Solving the large scale next release problem with a backbone-based multilevel algorithm, IEEE Transactions on Software Engineering 38 (5) (2012) 261–284.
URL https://doi.org/10.1109/TSE.2011.92 [12] M. R. Karim, G. Ruhe, Bi-objective genetic search for release planning in support of themes, in: Proceedings of the 6th International Symposium on Search-Based Software Engineering (SSBSE), Fortaleza, Brazil, 2014, pp. 123–137.
URL https://doi.org/10.1007/978-3-319-09940-8\_9 [13] F. M. Kifetew, A. Perini, A. Susi, A. Siena, D. M. nante, I. MoralesRamirez, Automating user-feedback driven requirements prioritization, Information & Software Technology 138 (2021) 106635:1–106635:16.
URL https://doi.org/10.1016/j.infsof.2021.106635 [14] F. B. Aydemir, F. Dalpiaz, S. Brinkkemper, P. Giorgini, J. Mylopoulos, The next release problem revisited: A new avenue for goal models, in:
Proceedings of the 26th IEEE International Requirements Engineering

27

Conference (RE), Banff, Canada, 2018, pp. 5–16.
URL https://doi.org/10.1109/RE.2018.00-56 [15] P. Carlshamre, K. Sandahl, M. Lindvall, B. Regnell, J. N. och Dag, An industrial survey of requirements interdependencies in software product release planning, in: Proceedings of the 5th IEEE International Symposium on Requirements Engineering (RE), Toronto, Canada, 2001, pp.
84–91.
URL https://doi.org/10.1109/ISRE.2001.948547 [16] W. N. Robinson, S. D. Pawlowski, V. Volkov, Requirements interaction management, ACM Computing Surveys 35 (1) (2003) 132–190.
URL https://doi.org/10.1145/857076.857079 [17] M. R. Karim, G. Ruhe, Datasets of “bi-objective genetic search for release planning in support of themes”, https://sites.google.com/sit e/mrkarim/data-sets, Last accessed: March 31, 2026(2016).
[18] A. Perini, A. Susi, P. Avesani, A machine learning approach to software requirements prioritization, IEEE Transactions on Software Engineering 39 (4) (2013) 445–461.
URL https://doi.org/10.1109/TSE.2012.52 [19] G. Ruhe, Product Release Planning - Methods, Tools and Applications, CRC Press, 2010.
[20] G. Ruhe, A. Eberlein, D. Pfahl, Quantitative WinWin: A new method for decision support in requirements negotiation, in: Proceedings of the 14th International Conference on Software Engineering and Knowledge Engineering, SEKE’02, Ischia, Italy, 2002, pp. 159–166.
doi:https://doi.org/10.1145/568760.568789.
[21] D. Leffingwell, D. Widrig, Managing Software Requirements: A Use Case Approach, Addison-Wesley, 2003.
URL https://dl.acm.org/doi/10.5555/829554 [22] J. Karlsson, K. Ryan, A cost-value approach for prioritizing requirements, IEEE Software 14 (5) (1997) 67–74.
URL https://doi.org/10.1109/52.605933

28

[23] J. Karlsson, C. Wohlin, B. Regnell, An evaluation of methods for prioritizing software requirements, Information & Software Technology 39 (1415) (1998) 939–947.
URL https://doi.org/10.1016/S0950-5849(97)00053-0 [24] A. Perini, A. Susi, F. Ricca, C. Bazzanella, An empirical study to compare the accuracy of ahp and cbranking techniques for requirements prioritization, in: Proceedings of the 5th International Workshop on Comparative Evaluation in Requirements Engineering (CERE), New Delhi, India, 2007, pp. 23–35.
URL https://doi.org/10.1109/CERE.2007.1 [25] I. Rahimi, A. H. Gandomi, M. R. Nikoo, F. Chen, A comparative study on evolutionary multi-objective algorithms for next release problem, Applied Soft Computing 52 (3) (2023) 110472:1–110472:13.
doi:https://doi.org/10.1016/j.asoc.2023.110472.
[26] Y. Zhang, M. Harman, G. Ochoa, G. Ruhe, S. Brinkkemper, An empirical study of meta- and hyper-heuristic search for multi-objective release planning, ACM Transactions on Software Engineering and Methodology 27 (1) (2012) 3:1–3:32.
URL https://doi.org/10.1145/3196831 [27] M. Svahnberg, T. Gorschek, R. Feldt, R. Torkar, S. B. Saleem, M. U.
Shafique, A systematic review on strategic release planning models, Information & Software Technology 52 (3) (2010) 237–248.
URL https://doi.org/10.1016/j.infsof.2009.11.006 [28] D. Ameller, C. Farré, X. Franch, G. Rufián, A survey on software release planning models, in: Proceedings of the 17th International Conference on Product-Focused Software Process Improvement (PROFES), Trondheim, Norway, 2016, pp. 48–65.
URL https://doi.org/10.1007/978-3-319-49094-6\_4 [29] E. C. Groen, J. Dörr, S. Adam, Towards crowd-based requirements engineering a research preview, in: Proceedings of the 21st International Working Conference on Requirements Engineering: Foundation for Software Quality (REFSQ), Essen, Germany, 2015, pp. 247–253.
URL https://doi.org/10.1007/978-3-319-16101-3\_16 29

[30] F. Palomba, P. Salza, A. Ciurumelea, S. Panichella, H. C. Gall, F. Ferrucci, A. De Lucia, Recommending and localizing change requests for mobile apps based on user reviews, in: Proceedings of the 39th IEEE/ACM International Conference on Software Engineering (ICSE), Buenos Aires, Argentina, 2017, pp. 106–117.
URL https://doi.org/10.1109/ICSE.2017.18 [31] S. Scalabrino, G. Bavota, B. Russo, M. Di Penta, R. Oliveto, Listening to the crowd for the release planning of mobile apps, IEEE Transactions on Software Engineering 45 (1) (2019) 68–86.
URL https://doi.org/10.1109/TSE.2017.2759112 [32] M. Nayebi, K. Kuznetsov, A. Zeller, G. Ruhe, User driven functionality deletion for mobile apps, in: Proceedings of the 31st IEEE International Requirements Engineering Conference (RE), Hannover, Germany, 2023, pp. 6–16.
URL https://doi.org/10.1109/RE57278.2023.00011 [33] G. H. Strønstad, I. Gerostathopoulos, E. Guzmán, What’s next in my backlog? Time series analysis of user reviews, in: Proceedings of the 8th International Workshop on Empirical Requirements Engineering (EmpiRE), Hannover, Germany, 2023, pp. 154–161.
URL https://doi.org/10.1109/REW57809.2023.00032 [34] C. M. C. Silva, M. Galster, F. Gilson, Topic modeling in software engineering research, Empirical Software Engineering 26 (6) (2021) 120:1– 120:62.
URL https://doi.org/10.1007/s10664-021-10026-0 [35] M. Sihag, Z. S. Li, A. Dash, N. N. Arony, K. Devathasan, N. A. Ernst, A. B. Albu, D. E. Damian, A data-driven approach for finding requirements relevant feedback from TikTok and YouTube, in: Proceedings of the 31st IEEE International Requirements Engineering Conference (RE), Hannover, Germany, 2023, pp. 111–122.
URL https://doi.org/10.1109/RE57278.2023.00020 [36] A. Hindle, C. Bird, T. Zimmermann, N. Nagappan, Do topics make sense to managers and developers?, Empirical Software Engineering 20 (2)
(2015) 479–515. doi:10.1007/s10664-014-9312-1.
URL https://doi.org/10.1007/s10664-014-9312-1 30

[37] R. Tiarks, W. Maalej, How does a typical tutorial for mobile development look like?, in: Proceedings of the 11th Working Conference on Mining Software Repositories (MSR 2014), Association for Computing Machinery, New York, NY, USA, 2014, pp. 272–281.
doi:10.1145/2597073.2597106.
URL https://doi.org/10.1145/2597073.2597106 [38] F. Palomba, M. L. Vásquez, G. Bavota, M. D. Rocco Oliveto, D. Poshyvanyk, A. De Lucia, User reviews matter! Tracking crowdsourced reviews to support evolution of successful apps, in: Proceedings of the 31st IEEE International Conference on Software Maintenance and Evolution (ICSME), Bremen, Germany, 2015, pp. 291–300.
URL https://doi.org/10.1109/ICSM.2015.7332475 [39] P. Qi, Y. Zhang, Y. Zhang, J. Bolton, C. D. Manning, Stanza: A Python natural language processing toolkit for many human languages, CoRR (March 2020).
URL https://doi.org/10.48550/arXiv.2003.07082 [40] Discord, Change Log, https://discord.com/developers/docs/chan ge-log, Last accessed: March 31, 2026(2026).
[41] Microsoft, Microsoft 365 Apps for Windows: Archived Release Notes, https://learn.microsoft.com/en-us/officeupdates/monthly-cha nnel-archived, Last accessed: March 31, 2026(2026).
[42] Cisco, Webex App: What’s New, https://help.webex.com/en-us/ar ticle/8dmbcr/Webex-App-|-What’s-New, Last accessed: March 31, 2026(2026).
[43] Zoom, Release Notes for Windows, https://support.zoom.us/hc /en-us/articles/201361953-Release-notes-for-Windows, Last accessed: March 31, 2026(2026).
[44] Python Software Foundation, Google-Play-Scraper, https://pypi.o rg/project/google- play- scraper/, Last accessed: March 31, 2026(2026).
[45] D. M. Berry, J. Cleland-Huang, A. Ferrari, W. Maalej, J. Mylopoulos, D. Zowghi, Panel: Context-dependent evaluation of tools for NL RE 31

tasks: Recall vs. precision, and beyond, in: Proceedings of the 25th IEEE International Requirements Engineering Conference (RE), Lisbon, Portugal, 2017, pp. 570–573.
URL https://doi.org/10.1109/RE.2017.64 [46] A. Fantechi, S. Gnesi, L. C. Passaro, L. Semini, Inconsistency detection in natural language requirements using ChatGPT: A preliminary evaluation, in: Proceedings of the 31st IEEE International Requirements Engineering Conference (RE), Hannover, Germany, 2023, pp. 335–340.
URL https://doi.org/10.1109/RE57278.2023.00045 [47] A. D. Rodriguez, K. R. Dearstyne, J. Cleland-Huang, Prompts matter: Insights and strategies for prompt engineering in automated software traceability, in: Proceedings of the 11th International Workshop on Software and Systems Traceability (SST), Hannover, Germany, 2023, pp. 455–464.
URL https://doi.org/10.1109/REW57809.2023.00087 [48] A.-R. Preda, C. Mayr-Dorn, A. Mashkoor, A. Egyed, Supporting highlevel to low-level requirements coverage reviewing with large language models, in: Proceedings of the 14th International Conference on Software Engineering and Knowledge Engineering, SEKE’02, Ischia, Italy, 2024, pp. 159–166. doi:https://doi.org/10.1145/568760.568789.
[49] S. Santos, T. D. Breaux, T. B. Norton, S. Haghighi, S. Ghanavati, Interlinking user stories and GUI prototyping: A semi-automatic LLM-based approach, in: Proceedings of the 32nd IEEE International Requirements Engineering Conference (RE), Reykjavik, Iceland, 2024, pp. 380–388.
URL https://doi.org/10.1109/RE59067.2024.00045 [50] G. Mori, GPT-4.5 vs GPT-4o: Comparing OpenAI’s Latest AI Models, https://giancarlomori.substack.com/p/gpt-45-vs-gpt-4o-com paring-openais, Last accessed: March 31, 2026(2025).
[51] D. Raymond, Top 10 Cons or Disadvantages of Agile Methodology, ht tps://projectmanagers.net/top-10-cons-or-disadvantages-o f-agile-methodology/, Last accessed: March 31, 2026(2023).

32


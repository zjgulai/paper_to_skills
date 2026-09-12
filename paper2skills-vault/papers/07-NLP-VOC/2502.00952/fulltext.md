<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2502.00952
     paper_id : 2502.00952
     source   : paper2skills-vault/papers/07-NLP-VOC/2502.00952/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities Dora Zhao

Diyi Yang∗

Michael S. Bernstein∗

dorothyz@stanford.edu Stanford University Stanford, California, USA

diyiy@cs.stanford.edu Stanford University Stanford, California, USA

msb@cs.stanford.edu Stanford University Stanford, California, USA

Survey of r/politics members (N=218)

arXiv:2502.00952v2 [cs.SI] 16 Mar 2026

POLITICAL IDEOLOGIES

10.0

24.8

6.7

26.2

32.4

URLs posted to r/politics (N=29,589)

5.4

9.7

10.5

63.6

10.8

1

2

3

4

5

1

Strongly Conservative

2

Conservative

3

Center / Independent

4

Liberal

5

Strongly Liberal

Figure 1: The political orientation of content posted on r/politics is not representative of its members’ ideologies. Since submissions on r/politics must contain URLs to articles from approved news and media organizations [1], we analyze the political leanings of the linked-to news sources. Compared to the surveyed political orientations of r/politics members with the ideologies expressed in the links shared to the subreddit, there is 1.3 times more liberal-leaning links than active liberal members on r/politics (see Appendix A.1 for methodological details).

Abstract We often treat social media as a lens onto society. How might that lens distort the popularity of political and social viewpoints? We examine discrepancies between publicly posted and privately surveyed opinions within communities, contributing a measurement of the “spiral of silence” theory; the theory posits people are less likely to voice opinions when they believe they hold minority views, creating a reinforcing cycle where these opinions are expressed less.
We surveyed members of politically-oriented Reddit communities about their willingness to post on contentious topics, yielding 439 responses across twelve subreddits. 72.1% of participants who perceive themselves in the minority remain silent and are half as likely to post compared to those who believe their opinion is in the majority. Community design factors, such as perceived diversity, are associated with less self-silencing. We provide recommendations for counteracting self-silencing at the community level (e.g., positive reinforcement, more transparent moderation). Overall, these ∗ Both authors co-advised.

This work is licensed under a Creative Commons Attribution 4.0 International License.
CHI ’26, Barcelona, Spain © 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2278-3/2026/04 https://doi.org/10.1145/3772318.3790361

results reveal gaps between online discourse and broader public opinion.

CCS Concepts • Human-centered computing → Empirical studies in collaborative and social computing.

Keywords spiral of silence, online communities ACM Reference Format:
Dora Zhao, Diyi Yang, and Michael S. Bernstein. 2026. Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities. In Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI ’26), April 13–17, 2026, Barcelona, Spain. ACM, New York, NY, USA, 26 pages. https://doi.org/10.1145/3772318.3790361

1

Introduction

Content shared on social media is often treated as a reflection of broader public opinion [3, 4, 67]. But is this view distorted?
For example, in 2019, a racist photo of the governor of Virginia surfaced online. As reported in The New York Times, this event led to universal condemnation of the governor on social media, leading major politicians to demand that he resign [20]. However, a poll of Virginians showed that most people, and especially Black constituents, were more in favor of the governor remaining in

CHI ’26, April 13–17, 2026, Barcelona, Spain

office [41]. Such voices were less vocal and less visible on social media, despite their prevalence.
Divergence between what we see on social media and what the public believes is not uncommon. For example, comparing the content submitted to the subreddit r/politics — a massive online community for sharing and discussing US political news — to the surveyed ideologies of active members, there is a large distortion (details in Appendix A.1): 74.4% of the content shared on the subreddit has a liberal view, compared to only 58.6% of what active members self-report (Fig. 1). This distortion can arise in liberal, conservative, and mixed spaces. What causes this distortion? While this could be an artifact of users’ preferences for posting online on certain topics, or perhaps a product of social loafing [46], we explore an alternative: users may refrain from sharing their opinion when they perceive themselves as being in the minority. The spiral of silence theory [74] explains this behavior as a response to the fear of social isolation, where individuals self-silence — or decide not to share their opinion — rather than risk social disapproval.
Self-silencing then leads to less of the opinion being expressed in the group, making the viewpoint appear even further minoritized.
This reaction produces a downward spiral as people become more likely to self-silence. Since being proposed, the spiral of silence has been empirically studied in face-to-face communication across multiple issue types and participant populations [65, 92]. As online communication forms like social media proliferate, it raises the question of how the spiral of silence might affect user behavior in this setting.
Measuring the spiral of silence on social media, especially across diverse online communities, is challenging. Not all topics are likely to trigger the spiral of silence [56], and identifying such topics may require insider knowledge of community norms and practices [14, 38]. Furthermore, even with a set of topics in hand, the process of measuring rates of self-silencing requires capturing user attitudes, beliefs, and behaviors that cannot be observed from online content itself. This traditional approach of analyzing existing posts online is self-defeating, since it excludes viewpoints that users feel uncomfortable sharing online.
In our work, we develop a human-plus-algorithm pipeline to generate potential topics that community members feel are appropriate and in-bounds for a community to discuss, but which also likely to spark internal disagreement. Then, drawing on survey methodologies common in empirical studies of the spiral of silence [65], we measure whether the spiral of silence occurs on those topics in those communities. Further, we evaluate whether online community factors such as moderation and diversity mitigate or amplify the spiral. We apply this method to study the spiral of silence on Reddit — a popular social media platform in the United States — across a number of subcommunities (“subreddits”) focused on political and social issues but with varying norms and values.
Importantly, our analyses focus on topics that community members agree are acceptable to discuss within the subreddit but are still hesitant to engage with, and exclude topics considered inappropriate or in violation of community norms.
We find that participants who believe their opinion is in the minority remain silent 72.1% of the time, and these opinions are only half as likely to be posted compared to those in the majority.

Zhao et al.

Participants are also less likely to upvote posts sharing minoritized opinions. However, we identify two community design factors that are linked to variations in the rate of self-silencing. We test these results through a series of three mixed-effects models, finding that participants report a higher likelihood of sharing minoritized opinions in subreddits they perceive as more diverse, but a lower likelihood of sharing in subreddits with more stringent content moderation.1 Our findings illustrate how, within the politically-oriented subreddits we study, the distribution of viewpoints being shared online can misrepresent community members’ actual opinions, systematically marginalizing minority perspectives. There are certainly some topics where minority opinions are normatively undesirable to represent, even if a portion of the population agrees with them, such as, hate speech or misinformation. However, for many other topics where the minority voice instead reflects under-represented groups or moderate views from people who believe themselves to be in the minority, communities may want to mitigate the spiral and hear from a more representative set of voices [10, 26]. Based on our findings, we surface tangible actions that online moderators and platform designers can take, such as providing positive feedback on posts expressing diverse viewpoints and increasing moderation transparency, to help mitigate the spiral of silence.

2

How the Spiral of Silence Shapes Social Media

Social media is often viewed as a mirror of society, or at the very least, one of our best available instruments for understanding public opinion. Social media data has been used for a range of tasks from forecasting box-office performances [4] to understanding political support for candidates [67]. Furthermore, social media content can actively shape people’s opinions and influence their actions [54].
However, what we see online may not be a complete or accurate picture of public opinion. This is partly because online content originates from a small fraction of platform users [88]. The majority of online communities consist of “lurkers” or individuals who visit the community but choose to remain silent [100]. There are many reasons why people may choose to stay as lurkers. In some cases, users may just want to learn about a topic or are not motivated to post if they see their opinion reflected in existing content [77].
Other times, however, users may want to share their opinions but are uncomfortable doing so. The disproportionate silencing of viewpoints may make our online environments appear more polarized than they actually are.
One explanation for this phenomenon is Noelle-Neumann [74]’s spiral of silence theory. The theory posits that, when people perceive their viewpoints as being incongruent, or different from the majority of the group, they are less likely to engage in opinion expression, or share their viewpoint. Since fewer people express the opinion, the viewpoint appears to be even further in the minority, triggering a spiral effect as the growing silence of incongruent opinions compounds upon itself.
The spiral of silence is an empirical effect, not an unavoidable constant applying to all people: there are “avant-garde” and “hard core” individuals who stand up for their viewpoints even when they are in the minority [75]. Through empirical studies, work has 1 Code available at https://github.com/StanfordHCI/spiral-of-silence.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

proposed factors that modulate the spiral of silence. Individuallevel antecedents, including personality traits [37, 72], opinion certainty [65], and issue type [31] can influence opinion expression.
In social media settings, prior work has focused on identifying platform affordances that influence self-silencing behavior [81, 108]. For example, greater perceived anonymity can dampen the effects as the potential social consequences of speaking out are diluted [108].
Even the means by which opinions are shared, such as commenting versus liking content, can impact the degree of self-silencing [81].
While we have insight into how platform-level design decisions influence self-silencing, platforms can house many online communities each with their own rules, norms, and values. Given the unique configuration of each online community, it is not surprising that users’ behaviors, such as conversation patterns [17] or selfdisclosure [110], will differ across communities. At the moment, the impact of these community-level design decisions has not been factored into our understanding of how the spiral of silence manifests. To better understand how these design decisions are linked to self-silencing, this work focuses on Reddit, which is comprised of many distinct subreddits — each of which has its own governance structures, moderation practices, values, topical interests, and norms [27, 57, 104] — providing an ideal context for studying community-level design variation. This leads us to our research questions:
RQ1. How does users’ opinion expression on political subreddits differ when they perceive their viewpoint as being in the majority versus the minority?
RQ2. How do online community design factors on political subreddits influence opinion expression of incongruent viewpoints?

Hypothesis 1: The likelihood of opinion expression is negatively associated with viewpoint incongruency (i.e., when users believe the majority of community members disagree with them).

3

Factors Influencing Opinion Expression

In this section, we explore how the spiral of silence and community design decisions can shape opinion expression on social media and introduce our hypotheses.

3.1

Opinion Incongruency

The spiral of silence theory provides an explanation for viewpoint sharing patterns on social media platforms. Previous studies [30, 36, 40, 49, 59, 81] on social media platforms, such as Facebook and Twitter, found evidence that users are less likely to express incongruent viewpoints, or opinions that differ from what they perceive the majority of other members believe. We expect a similar phenomenon on Reddit. Since subreddits also offer a sense of community and socializing opportunities [69], community members may still experience a fear of social isolation. This fear can be triggered on Reddit by potential social repercussions, such as being downvoted or facing personal attacks from others in the community [15, 71]. Especially when discussing stigmatized topics, users also have expressed concerns about exposure, including having sensitive content linked to their public profile or facing offline repercussions from people who know about their online account [55, 85]. These consequences encourage users to self-silence when they have an opinion that they think differs from the majority. Given this evidence, we expect our study will yield results in accordance with the spiral of silence:

3.2

Community Inclusion and Diversity

How online communities are designed will influence how users behave, especially regarding sharing their opinions publicly. A greater sense of belonging can lead users to be more active as they feel more included and trusting of others in the group. Similarly, feeling psychologically safe can foster confidence and empower users to voice their opinions [25].
Feeling a sense of belonging is an essential human need that applies not only to offline but also online settings [52, 63]. Hagerty et al. proposed two key dimensions that define a sense of belonging:
(1) having involvement in a group be valued, and (2) feeling a sense of fit or congruency with other group members. When these criteria are met, people are more likely to be involved in a group and find this involvement to be more meaningful [35]. The same principles apply to belonging in online communities [53]. Users with strong shared identities or bonds are more likely to be active in online groups [87]. When people feel included, they are more likely to have a sense of familiarity and trust other community members [113].
This encourages users to share more freely, as they feel less at risk of being judged when sharing their viewpoints. Thus, we expect that:
Hypothesis 2a: The perceived inclusivity of a community is more positively associated with the likelihood of opinion expression when users hold incongruent viewpoints.
Who is part of the community also matters. Users frequently mention the fear of being personally attacked as a primary reason for not posting minority opinions [71]. In both online and offline settings, existing research has found that people are more secure when sharing ideas in psychologically safe communities [73]. That is, when people foresee fewer negative interpersonal consequences, they are more willing to take potentially controversial actions.
Community diversity is an important antecedent for psychological safety [73]. Emphasizing diversity can foster a positive climate in which community members, regardless of their background, more strongly identify as members of the group and feel psychologically safe [68]. In this setting, group members are more willing to engage in potentially risky behaviors, such as sharing opinions that may deviate from what is popular. In a similar vein, prior work on the spiral of silence has suggested that those with more bridging social capital that connects them to heterogeneous networks on social media platforms, are more willing to speak up online [94]. From this literature, we hypothesize the following:
Hypothesis 2b: The diversity of community members is more positively associated with the likelihood of opinion expression when users hold incongruent viewpoints.
To conceptualize inclusion and diversity, we draw on work that has sought to quantify the values of online communities [103]. We utilize the instrument proposed by Weld et al. [104] to capture users’ perceived sense of inclusion and diversity across different subreddits. Perceived inclusion is defined as how “included and able to contribute new and existing member feel” while perceived

CHI ’26, April 13–17, 2026, Barcelona, Spain

diversity refers to how diverse people in the community are thought to be.

3.3

Community Moderation

Moderation practices frequently shape the dynamics of online communities. Moderators can set the tone of the online community by encouraging other users to imitate their actions and by curbing instances of behavior that they deem to fall outside community norms [93]. Removing or prescreening content that does not adhere to the community’s norms is an effective way of reducing “bad behavior” and can positively impact community health [48, 93]. By filtering out potentially harassing or offensive submissions, moderators can assuage members’ fear of being personally attacked and facilitate more open opinion-sharing [32]. For example, Wadden et al. [101] studied mental health discussions in moderated and unmoderated spaces. They found that users were not only more likely to post in moderated spaces, but also that they were more willing to share vulnerable messages in these discussions. In a setting more similar to opinion-sharing on Reddit, Wise et al. [106]
found that users were more willing to participate in a moderated online political community compared to an unmoderated space. From these results, we expect that users feel more comfortable sharing minoritized opinions in communities with more active moderation.
Thus, we hypothesize:
Hypothesis 3: Moderation activity is more positively associated with the likelihood of opinion expression when users hold incongruent viewpoints.
In this work, we focus on moderation through content removal.
We acknowledge that content removal represents only a subset of moderation actions available on Reddit, but choose to focus on content removal because it is one of the most common moderation interventions and has been shown to influence community members’ participation [58].

4

Method

Our goal is to measure the spiral of silence across topics and communities. We first need a set of controversial topics that may trigger the spiral of silence. This analysis requires us to identify topics that community members agree are permissible to discuss in their community in principle, but may not feel comfortable sharing in practice. Crawling existing posts from these communities, unfortunately, is insufficient: these posts will not reflect the opinions that users may be hesitant to share in the first place. In other words, starting with existing topics runs the risk of only identifying topics that are so safe as to appear commonly in the community. Alternatively, while asking participants to list instances when they have self-silenced is possible, this elicitation process may be prone to recall or desirability bias and yield idiosyncratic or off-scope topics falling outside the theory’s bounds.
Thus, we use a hybrid human-AI method that draws on large language models (LLMs) to nominate potential controversial topics that are appropriate to each community, even if those topics are not posted in the community, and validate those proposed topics with active subreddit members. We then survey community members to capture the rate of self-silencing on those topics. In this section, we

Zhao et al.

start by discussing the scope of our study and our definition of selfsilencing. Then, we provide an overview of our method, describing the topic generation process and survey instrument. This study was approved by Stanford University’s Institutional Review Board.
Scope of the Study. One important distinction we make in this work is between when community members self-silence because they are uncomfortable sharing an opinion versus when an opinion is not considered acceptable to be shared within a community. Our study is constrained only to the former. In other words, we focus only on perspectives that lead to disagreement while still falling within the bounds of the community norms. We do not seek to make any normative claims about what should or should not be shared to a subreddit. Instead, we are interested in what is not being shared, even when most community members agree the opinion should be allowed to be discussed.

4.1

Topic Generation

Our methodological approach involves using an LLM to propose controversial topics and opposing viewpoints for each topic, then surveying actual community members to select viewpoints for our study, which provides several advantages for studying the spiral of silence in our setting. First, while manually selecting topics is the de facto approach when studying the spiral of silence across more than one issue type [31, 56], manual selection may introduce experimenter bias and may be impractical across a large number of diverse yet specific communities on Reddit, each with its own rules and norms [27]. Deciding what topics are controversial will be imprecise for those without specific knowledge of the community [14, 38]. Second, directly looking at Reddit data is not adequate in this scenario either, as we are interested in the opinions that may be less likely to be shared to the community, rather than those that are already present. Thus, using mined data from online only gives us a window into the viewpoints that community members feel comfortable sharing. By contrast, using an LLM allows us to nominate potential topics that are relevant to the community and might result in self-silencing, including perspectives that may be underrepresented or missing in the current discourse online. Prior work has also shown that LLM-based topic generation produces more coherent and interpretable topics than traditional approaches such as LDA [47, 82]. We then ask active community members to validate the relevancy and appropriateness of these proposals, selecting topics from this filtered set to include in our study.
4.1.1 Proposing Controversial Topics and Viewpoints. Topic generation is a three-step process (see Fig. 2). We provide a community name, description, and rules as input to an LLM (GPT-4) to propose 20 potentially controversial topics for the provided community. We provide in our prompt that we define “controversial” as being likely to cause disagreement between community members. Then, for each proposed topic, the LLM generates two viewpoints representing different sides of the issue using few-shot prompting techniques (see Appendix A.2 for prompts). The viewpoints must follow the rules of the subreddit, which are again provided to the model. Finally, as we discuss in detail in Sec. 4.2, viewpoints are manually validated to ensure that they are comprehensible and relevant to the respective subreddit.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

PROPOSE TOPICS

ENUMERATE VIEWPOINTS

The role of NATO in maintaining global peace.

"Russia's state-sponsored cyber attacks threatening global security require international action."

INPUT NAME r/worldnews DESCRIPTION A place for major news from around the world…

CHI ’26, April 13–17, 2026, Barcelona, Spain

The role of Russia in global X cyber warfare.X

FILTER TOPICS The role of Russia in global cyber warfare.
Affirmative action

X X

RULES 1. No US Internal News or Politics…

The future of the European Union without the United Kingdom.

Provide community name, description, and community rules as input to an LLM

Generate topics that will lead to disagreement amongst community members

"Blaming only Russia for cyber warfare oversimplifies a global issue involving many nations."

For each topic, generate and then shorten comments representing all sides of the argument

The future of the EU without the UK

Community members manually select proposed topics that are relevant to their community

Figure 2: We generate controversial viewpoints specific to a provided community. Using the community name, description, and rules, we prompt an LLM to propose controversial topics. For each topic, we enumerate statements representing different stances on the topic. Finally, the generated topics and viewpoints are validated by community members to ensure they are relevant and fall within the norms of what is accepted in the community.
4.1.2 Selecting Political Subreddits. To run our topic generation pipeline, we must select which subreddits to provide as input to the LLM. For this work, we focus on subreddits that focus on political and social issues, since Noelle-Neumann [74] posits that morally laden subjects typically trigger the spiral of silence. To identify these communities, we start with the 2,040 subreddits on r/ListOfSubreddits and sample the 50 hottest submissions in each as of January 2024 accessed via the Reddit API. We then filter for the following criteria: (1) the majority of submissions are in English as classified by langdetect [96]; (2) ≥ 80% of submissions are classified to contain political content using an LLM classifier on the post title following an approach from prior work [83] (see Appendix A.3 for prompt); and (3) the subreddit has over 100K members. Filtering leaves us with 23 political subreddits.
Since we seek to compare variation in opinion expression on topics based on community-level differences, we want to identify a single, static set of topics that are likely to be relevant across all 23 subreddits. To ensure broad applicability of the topics, we selected r/politics and r/worldnews as input to the LLM for topic generation based on their size and generality, rather than using more specialized subreddits that might yield niche topics with limited relevance across communities. Using our pipeline, we generate 40 topics (i.e., 20 per subreddit) with opposing viewpoints for each topic. As shown in Table 1, topics covered a range of political and social issues, including abortion rights, military spending, and affirmative action. Each identified topic contains two viewpoints that articulate two main clusters of opinion on the topic: for example, with affirmative action, one viewpoint states, “I believe affirmative action counters systemic biases and fosters a diverse, inclusive society,” and the other viewpoint states, “I believe affirmative action could unintentionally cause reverse discrimination and undermine merit, potentially increasing societal division.” For the full list of topics and viewpoints, refer to Appendix Table 5.

4.2

Validating Generated Topics and Viewpoints

Once we have our generated topics, how do we ensure that they fall within the bounds of what is acceptable for a given subreddit? We conduct three checks to validate and filter our generations. First,

we manually inspect each of the generations for coherence. Then, we survey active community members to certify that the generated topics are relevant and appropriate to the subreddit. Finally, we conduct an “intrusion test” also with active community members to verify that our generated topics could plausibly appear in posts on the subreddit. We apply our checks in a sequential order, meaning that only the topics that pass the coherence check are included in the community-specific relevance check and so on.
4.2.1 Coherence Check. We start by manually filtering the generations for comprehension. Using the 40 topics from the prior step, we check that the generated outputs contained opposing viewpoints, covered controversial topics, and were worded in a manner that participants would understand. From this step, we filtered out seven topics: four had incomplete viewpoints (i.e., the model only provided viewpoints representing one side) and three were either worded confusingly or presented viewpoints that did not present opposite stances, leaving us with 33 topics.
4.2.2 Community-Specific Relevance Check. Next, after ensuring that generations are coherent, we need to validate that our generated topics and viewpoints are not only relevant, but also follow the community norms for the respective subreddit. The first way we do this is by having active members of the subreddits (r/politics and r/worldnews) manually review the topics. We recruited Prolific participants who self-reported as active members. All participants first completed a screening quiz to confirm their knowledge about the subreddits in which they were presented a set of three post titles—two from their selected subreddit and one from another political subreddit (which we manually verify is irrelevant to the selected subreddit)—and asked to select which titles were likely to appear in their selected subreddit (see Appendix A.4 for more details and examples). If they identified all the correct post titles, participants qualified for the survey. In total, 46 participants passed our screening quiz and completed our community-specific topic relevance check. They were then shown a list of generated topics and viewpoints and asked whether they were relevant to the subreddit.
Participants also had the option to response “I don’t know” for topics. We removed two topics that many respondents reported not

CHI ’26, April 13–17, 2026, Barcelona, Spain

having knowledge on and one topic that was commonly marked as not relevant to the community. Ultimately, from this filtered set of 30 topics, we randomly selected 11 for our final study — five generated with r/politics as the input and six with r/worldnews — to ensure there are an adequate number of responses per topic for our analysis.

Zhao et al.

4.3

Opinion Expression Survey

Next, we use the validated viewpoints from Sec. 4.2 to gauge the rate of self-silencing among active community members.2 Since these self-silenced viewpoints will not be posted on Reddit, we directly survey community members to measure this phenomenon.
The goal of our survey is to identify how willing an individual is to share their opinion on a topic to a subreddit. Specifically, we are 4.2.3 Intrusion Testing. While our coherence and community-relevance interested in the following dimensions: (1) differences in sharing behavior when individuals view themselves as being in the majority checks ensure generated topics are understandable and broadly versus minority; and (2) how self-silencing differs across subreddits.
aligned with subreddit norms, they do not tell us whether these topics feel authentic to active community members. A topic may 4.3.1 Survey Instrument. For each topic, our survey captures the be technically relevant but still strike community members as out likelihood that participants would share their viewpoint on each of place. To examine whether our topics meet this bar, we consubreddit. As shown in Fig. 3, participants are first asked to select duct an intrusion test, a method of outlier detection widely used in up to three subreddits of which they are an active member. For each natural language processing as a quantitative evaluation for topic of the subreddits they selected, participants chose up to five topics modeling [8, 11, 18].
they thought were relevant to each subreddit from the total pool of eleven topics. We include this step as an additional check — on Task Setup. We recruited a new set of active community memtop of the validation described in Sec. 4.2 — to ensure that we are bers (i.e., Prolific participants who passed our screening quiz for surveying participants about topics they believe are appropriate the respective subreddit) to participate in our intrusion testing task.
and relevant to the community. For example, for r/combatfootage, These participants were shown the generated topic and four controselected topics included “ethics of drone warfare in the Middle East,” versial topics mined from the subreddit that are semantically similar “military spending,” and “the role of NATO in maintaining global to the generated topic. To obtain the controversial topics mined peace” whereas in r/feminism, relevant topics included “abortion from the subreddit, we start with Pushshift dumps of all posts made rights” and “universal healthcare.” in r/politics and r/worldnews between January 2020 and DeFor each topic, participants indicated the viewpoint they agreed cember 2023 [5]. Then, following the pipeline from Hessel and Lee with. They then rated the extent to which they agreed with the [38], we first discard posts that have less than 30 comments and sort viewpoint, how much they believed the majority of subreddit memposts by their upvote ratio. After removing posts with a negative bers agreed with the viewpoint, and whether they thought the score, we randomly select 500 posts in the bottom quartile of scores, viewpoint should be allowed to be posted on the subreddit. For which contain topics that subreddit members want to engage with all viewpoints the participant agreed with and thought should be (as indicated by the number of comments under the post) but lead to allowed on the subreddit, we asked about their likelihood of sharing high amounts of disagreement (as indicated by the close number of that viewpoint on the subreddit and their likelihood of upvoting upvotes and downvotes.). Finally, we extract common controversial that viewpoint if someone else shared it on the subreddit. If the topics using TopicGPT [82]. For each generated topic, we select participant responded that they were unwilling to comment their the four most semantically similar mined topics as determined by opinion, we asked what alternative actions (e.g., read discussion the cosine similarity between sentence embeddings [86]. For exambut not post, share to a different subreddit) they would take. The ple, for the generated topic of “abortion rights”, the mined topics order in which the topics were presented for each subreddit was were “impact of Roe v. Wade’s overturning”, “Amy Coney Barett’s randomized.
confirmation”, “freedom of protest rights”, and “government regu4.3.2 Participants. Survey participants were recruited via Prolific.
lation of personal freedoms and public health.” Participants then To qualify, participants were required to live in the United States, identified the “intruder” topic, by selecting which of the five topics be at least 18 years old, and identify as an active member of at is least likely to appear in the subreddit. If our generated topic is least one political subreddit included in our list. This list contains inappropriate to a subreddit, then it should be consistently selected r/politics and r/worldnews as well as the 21 additional political as the intruder.
subreddits from Section 4.1.2. We include these additional subreddits to obtain a larger pool of participants.
Results. In total, 63 participants completed our task – 32 from Participants selected up to three subreddits they were an active r/politics and 31 from r/worldnews. Across the eleven topics, member of from the list of 23 subreddits used in the pre-screener.
our generated topic was selected as the intruder only 18.8% of the Participants could, if desired, provide up to one additional political time. A one-proportion 𝑧-test comparing this intrusion detection subreddit not included in our list. We verify participants are active rate against random guessing (20.0% selection rate) was not statistimembers of the subreddit using the same knowledge quiz on identically significant (𝑧 = −0.41, 𝑝 = 0.68), indicating that community fying relevant post titles from the topic generation process. A total members were no better than chance at identifying our generated of 489 participants completed the pre-screener; 337 self-reported as topic. This result suggests that the generated topics are likely to members of a listed subreddit; and 290 passed the knowledge quiz.
appear in the subreddit according to community members, and for nine of the eleven topics, they are rated as more likely to appear in the subreddit than topics from mined conversation data.

2 Our study is pre-registered on OSF at this link.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 1: Examples of topics, and their respective viewpoints, generated for r/politics and r/worldnews.
Topic

Viewpoint 1

Viewpoint 2

Universal Healthcare

I believe in Universal Healthcare because everyone deserves access to good health, funded by the government.

I believe market competition and individual insurance plans are superior to Universal Healthcare.

Abortion Rights

As a pro-choice supporter, I believe women’s bodily autonomy and reproductive choices are crucial for gender equality.

As a pro-life advocate, I believe every life from conception deserves legal protection.

Drone warfare

Drone warfare is a necessary evil for global security due to its precision, efficiency, and safety for soldiers.

Drone warfare inevitably causes collateral damage, violates human rights, and induces terror.

STEP 1

Select ≤3 subreddits you are an active member of ✓ r/politics ✓ r/worldnews ✓ r/economics

STEP 2

STEP 3

Select ≤5 relevant topics and rate community values

Select the viewpoint you agree with more and report likelihood of opinion expression

✓ Abortion Election Reform ✓ Drone Warfare ✓ Impact of Brexit

I believe women’s bodily autonomy and reproductive choices are crucial for gender equality I believe every life from conception deserves legal protection.
I don’t know

PER SUBREDDIT

PER TOPIC X SUBREDDIT

Figure 3: We measure self-silencing behavior across topics and subreddits using a survey. First, participants select up to three subreddits of which they are an active member. For every selected subreddit, the participant answers questions about the community’s values and chooses up to five topics relevant to the community. For each topic, we ask the participants which viewpoint they agree with more. Finally, for the viewpoint that they agree with, participants answers how much they agree with the viewpoint, how much they believe others in the subreddit agree with the viewpoint, and their likelihood of sharing the viewpoint.
We conduct a Monte-Carlo power analysis to determine the sample size required to detect significance using a 𝑧-test for the fixed effect of Incongruency at 𝛼 = 0.05 and assuming a coefficient of 𝛽 = −0.6. Based on this power analysis, we needed to collect 270 responses (participants × topics × subreddit) to achieve 90% power, which is approximately 54 participants. In our final analysis, we include 439 responses from 58 participants, covering twelve subreddits.
4.3.3 Data Collection Process. To arrive at this sample, we initially collected 906 responses from 125 participants and after filtering arrive at our final sample of 439 responses (see Fig. 4). First, since we focus on political topics, we recruited a balanced sample of participants by political leaning. We asked for participant’s political leaning in our screening quiz and stratified recruitment, resulting in 40 Democrats, 38 Republicans, 40 Independents, and 7 participants with other political ideologies. Then, for quality assurance, we removed 46 responses from 10 participants who failed our attention checks. We also removed individual responses if: (1) the participant felt the viewpoint should not be shared on the subreddit (N=74); (2) the participant selected “don’t know” when asked to pick a viewpoint for a topic (N=122); or (3) the participant later disagreed with the viewpoint they selected as representing their stance (N=31). Finally, due to the observed sparsity in our crossed

random effects structure, we restricted the dataset to subreddits with ≥ 10 responses and participants with ≥ 5 responses based on selections from prior work to ensure the mixed-effects models we use in our analysis have stable estimates [61]. Over half of the subreddits from the initial responses had only one participant responding, which can lead to convergence issues and unreliable variance estimates in mixed-effects models [6]. To verify that this filtering did not introduce systematic bias, we conducted statistical tests comparing the filtered and unfiltered datasets, finding no significant differences in the distributions (see Appendix B.1 for full results). This process results in our final dataset of 439 responses from 58 participants, who consist of 21 Democrats, 21 Republicans, 14 Independents, and 2 with other political ideologies.

4.4

Measures

We detail the measures used in our study. Our dependent variable is the likelihood of opinion expression. We introduce four independent variables: opinion incongruence, community inclusion, community diversity, and content removal rate. Finally, we include controls for individual-level factors that can influence opinion expression.
4.4.1 Likelihood of Opinion Expression. Our main dependent variable is a participant’s likelihood of opinion expression. In our survey,

How di

How in contrib membe

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

Total responses

Responses after

Responses meeting

collected


quality filters


analysis criteria


(N = 906)

(N = 633)

(N = 439)

EXCLUDED

EXCLUDED

Failed attention checks (N=46)


Participants with <5 responses (N=113)


Viewpoint marked as not suitable for subreddit (N=74)

Viewpoint marked as “don’t know”

Subreddits with <10 responses (N=81)


(N=122)

Viewpoint later disagreed with (N=31)

Figure 4: We filter the responses — where an individual response denotes a participant’s likelihood of speaking out on a specific topic in a subreddit — collected to those that pass our quality checks and meet our analysis criteria. From our initial pool of 906 responses, our filtering process yield in 439 responses that we use in our analyses.
we follow prior work [60, 71, 112] and present the participants with a hypothetical scenario. Participants are asked to imagine they see a post on their selected subreddit related to the presented controversial topic, and that no comments with the following viewpoint have been raised yet. Then, we ask their likelihood of sharing the viewpoint, Share Likelihood, under their main account reported on a scale from 1 to 7 (i.e., “Rate the likelihood you would share this viewpoint on the selected subreddit under your main account”).
We also ask participants to provide the likelihood of upvoting the viewpoint (Upvote Likelihood) on a seven-point scale (i.e., “Rate the likelihood you would upvote someone else’s comment expressing this viewpoint, if the comment was present”).
4.4.2 Opinion Incongruency. For a given viewpoint, Incongruency is a binary variable indicating whether a participant believes their opinion is held by the majority (0) or minority (1) of other subreddit members. Following prior work [16, 72], we operationalize incongruency as the difference between how much a participant agrees with a viewpoint and how much they believe the majority of subreddit members agree. Both agreement scores are measured on a 7-point scale. Consistent with prior work that empirically measures the spiral of silence [31, 66, 72], we then convert this difference into a binary variable. We indicate all instances when the participant agrees with the viewpoint, and they believe the majority of subreddit members also agree as 0. Conversely, if the participant agrees with the viewpoint but believes the majority of subreddit members are neutral or disagree, incongruency is marked as 1.3 4.4.3 Community Inclusion and Diversity. We collect self-perceived community inclusivity and diversity from participants using Weld et al. [103]’s instrument for measuring community values across subreddits. For a given subreddit, participants report their perception of the current state of Inclusion (i.e., “How included and able to contribute do new and existing members feel?”) and Diversity (i.e., “How diverse are the people in the community?”). The response 3 As a robustness check, we report results using an alternative dichotomization for

Incongruency, finding qualitatively similar results (see Appendix B.2).

was recorded on an 11-point scale (from 1-11) for consistency with prior work [103].
4.4.4 Community Moderation. We use content removal as our main measure of community moderation. Using Pushshift data from Jan. 2022 - Dec. 2023, we compute Content Removal Rate as the percentage of submissions removed by moderators within a subreddit [43]. In total, there were 856, 980 submissions across twelve subreddits. The removal rate ranged from 16.0% (r/conspiracy) to 41.5% (r/changemyview), with a median of 31.9% percent per subreddit (see Appendix A.6 for disaggregated removal rates). We apply a logarithmic transformation with base 2 after adding a start-value of 1 to the computed values.
4.4.5 Controls. Prior work on the spiral of silence has surfaced individual-level factors that influence a person’s willingness to voice their viewpoint [21, 66, 89]. First, we include a measure for a participant’s willingness to self-censor (WTSC) to account for individual personality differences that could impact the likelihood of sharing. Following prior work [13, 29, 95], we use the composite measure from Hayes et al. [37], which consists of eight questions that capture an individual’s likelihood of withholding their opinion when they sense others disagree. A larger WTSC value means the individual is more disposed to self-censoring. Second, following Matthes et al. [66]’s finding that how strongly an individual agrees with a viewpoint impacts the chance that the spiral of silence occurs, we include a control for Agreement Intensity measured on a threepoint scale. Finally, we include controls for user demographics, as prior work [23, 84] has found differences in sharing behavior across identity groups. We include three binary variables for gender, race, and political leaning.
We also include controls related to the participant’s Reddit usage.
Importantly, we expect that many participants are lurkers, who are unlikely to post regardless of the topic [33, 76]. To control for the different base rates of posting frequency, we include a measure, Posting, which captures a participant’s self-reported likelihood of actively participate in a community (as opposed to lurking) on a 5-point scale. Finally, we expect newcomers may exhibit different

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

sharing behavior compared to platform veterans [109]. Thus, we include the control Account Tenure that measures the age of the participant’s Reddit account in years.

outside of posting frequency and agreement intensity, we do not find any statistically significant differences for our control variables.

5

Results

From our survey, we find that participants are less likely to voice incongruent viewpoints across topics. Among responses where the reported viewpoint aligns with the majority opinion in a subreddit, 47.2% indicate a high likelihood of sharing their viewpoint (Share Likelihood > 4). In contrast, only 27.9% of responses expressing incongruent viewpoints report a high likelihood of sharing. Looking across topics, congruent viewpoints are 2.03 ± 0.98 times more likely to be shared compared to incongruent viewpoints. The difference in sharing is most stark for the topic of affirmative action, where congruent viewpoints are 4.09 times more likely to be shared compared to incongruent viewpoints. We observe only one topic (“ethics of drone warfare”) in which participants are more likely to share incongruent viewpoints compared to congruent ones.
In this section, we formalize these analyses using linear mixedeffect models. We start by presenting the model we used for analysis. Next, we compare opinion expression in congruent versus incongruent opinion climates. Then, we explore the association of community-level factors, such as inclusivity and moderation practices, with self-silencing. We conclude by analyzing what factors are associated with variation in upvoting behavior.

5.1

Method of Analysis

To analyze the results of our survey, we use linear mixed-effect models, with subreddit and participant as crossed random effects.
We introduce a series of three nested models. As shown in Table 2, we first have a model using only our control variables as predictors.
We then add additional measures for viewpoint incongruency, community inclusivity, diversity, and moderation. Finally, we include interaction effects between incongruency and our measurements for community values and moderation to our model. We standardize all continuous independent variables and controls. For our first set of models (Models 1a-1c), the dependent variable is the selfreported likelihood of sharing the viewpoint on a main account (Share Likelihood). In Models 2a-2c, we use the likelihood of upvoting the viewpoint (Upvote Likelihood) as our dependent variable, which we report in Table 3. The dependent variables are left in raw units.

5.2

Analyzing Opinion Sharing Behavior

5.2.1 Participants are willing to speak up for congruent viewpoints.
We start by examining the relationship between the likelihood of sharing a viewpoint and our control variables. In Model 1a, the intercept is 4.56, representing the likelihood of sharing a congruent viewpoint for a non-Male, non-White, and non-Democrat individual who posts on Reddit rarely to occasionally, has an account age of 6.2 years, and mean WTSC (3.68 out of 8). The intercept value falls between “neither likely nor unlikely” and “slightly likely” on the 7-point scale from the survey. As expected, posting frequency is positively associated with opinion expression (𝛽 = 0.70, 𝑝 = 0.01).
In addition, participants who agree with a viewpoint more strongly are also more likely to comment (𝛽 = 0.42, 𝑝 < 0.001). However,

5.2.2 Incongruent viewpoints are shared less often. Participants were less likely to voice incongruent viewpoints across all topics despite agreeing with the stance. Approximately half (52.8%) of participants responded that they are likely to share their viewpoint (Share Likelihood > 4) when they believe themselves to agree with the majority (see Fig. 7). In contrast, the proportion of participants likely to share incongruent viewpoints is lower at 27.9%, although the percentage ranges across topics from 13.3% to 50.0%. On average, across all topics, participants are 2.04 times more likely to share a viewpoint they perceive as being in the majority compared to viewpoints they believe are in the minority.
This result is encoded in Model 1b, which includes a fixed effect for viewpoint incongruency, allowing us to compare the likelihood of sharing a viewpoint that is perceived to be in the majority versus minority. A likelihood ratio test (ANOVA) confirms that adding these fixed effect significantly improves the explanatory power from Model 1a, which only includes demographics and other control variables as predictors, to 1b (𝜒 2 = 31.4, 𝑝 < 0.001). In line with the spiral of silence theory, we observe a negative relationship between opinion incongruency and the likelihood of sharing (𝛽 = −0.76, 𝑝 < 0.001), indicating that participants may be less inclined to express their viewpoint when they believe their opinion differs from that of the majority. On average, when participants perceive themselves as being in the minority, they are less likely to share their viewpoint (𝑀 = 3.01) compared to those who view themselves as in line with the majority (𝑀 = 4.15).
While participants view themselves as being congruent with the majority for most viewpoints, 72.4% (N=42) of participants hold at least one incongruent viewpoint. Given that users must selfselect into being members of different communities on Reddit, it is unlikely that they will be completely incongruent across all topics.
Although community members are not typically in the opinion minority, they are likely to feel incongruent on at least a small selection of topics discussed on the subreddit.
5.2.3 Community diversity is positively associated with incongruent sharing behavior. In addition to incongruency, we include measures of participants’ perceived inclusivity and diversity of the subreddit. The main effects of perceived diversity (𝛽 = −0.08, 𝑝 = 0.42)
and inclusivity (𝛽 = 0.03, 𝑝 = 0.77) are not statistically significant.
However, we find a significant interaction between diversity and incongruency (𝛽 = 0.29, 𝑝 = 0.03), supporting H2b that higher perceived community diversity is associated with a greater likelihood of sharing incongruent viewpoints. These results come from Model 1c, which includes interaction effects between incongruency and our community design factors.4 Specifically, for incongruent viewpoints, a one-standard-deviation increase in perceived community diversity is associated with a 0.29-point increase in the likelihood of commenting, whereas diversity shows a negligible relationship for congruent viewpoints (𝛽 = −0.08). This result suggests that when individuals perceive communities as being more heterogeneous, 4We confirm that adding these additional predictors significantly improves model fit

compared to Model 1b ( 𝜒 2 = 13.3, 𝑝 = 0.004).

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

Figure 5: Across all topics, the average percentage of incongruent viewpoints likely to be shared is 27.9%. On the left, we report the percentage of incongruent viewpoints willing to be shared (i.e., Share Likelihood > 4) out of all incongruent viewpoints for the given topic. On the right, we do the same for congruent viewpoints.

B

Comment Likelihood

A

Diversity

Content Removal Rate Congruent

Incongruent

Figure 6: Community design factors relate to participants’ likelihood of sharing incongruent viewpoints. The left panel (A)
shows the marginal effects of community diversity on opinion expression for incongruent and congruent viewpoints. The right panel (B) shows the marginal effects of content removal rate on opinion expression for incongruent and congruent viewpoints.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 2: Although incongruent viewpoints are reported less often than congruent ones, communities with greater perceived diversity are associated with lower levels of self-silencing. We present the results of a linear mixed-effect model that predicts a participant’s likelihood of sharing their opinion as a hierarchical regression. Model 1a contains only our controls.
Model 1b includes the controls and fixed effects for opinion incongruency, community values, and community moderation.
Model 1c adds interaction effects. Note: ∗ p<0.05; ∗∗ p<0.01; ∗∗∗ p<0.001.
Model 1a Coef.
(Intercept)
Account Tenure Posting Male White Democrat WTSC Agreement

4.56 0.09 0.70 −0.70 0.11 −0.66 −0.27 0.48

Model 1b

SE

𝑝

0.60 0.24 0.24 0.56 0.54 0.50 0.25 0.06

0.00∗∗∗

Coef.

0.71 0.01∗∗ 0.21 0.83 0.19 0.28 0.00∗∗∗

Incongruency Inclusion Diversity Content Removal Rate

SE

𝑝

Coef.

SE

𝑝

4.78 0.09 0.64 −0.75 0.12 −0.74 −0.30 0.41

0.59 0.24 0.24 0.55 0.53 0.50 0.25 0.06

0.00∗∗∗ 0.72 0.01∗∗ 0.17 0.82 0.14 0.22 0.00∗∗∗

4.83 0.10 0.64 −0.77 0.12 −0.76 −0.33 0.42

0.59 0.24 0.24 0.54 0.52 0.49 0.24 0.06

0.00∗∗∗ 0.68 0.01∗∗ 0.16 0.82 0.13 0.19 0.00∗∗∗

−0.76 0.09 −0.02 −0.05

0.14 0.11 0.10 0.08

0.00∗∗∗ 0.42 0.85 0.53

−0.66 0.03 −0.08 0.04

0.14 0.11 0.10 0.09

0.00∗∗∗ 0.77 0.42 0.69

0.23 0.29 −0.31

0.14 0.14 0.14

0.12 0.03∗ 0.03∗

Incongruency × Inclusion Incongruency × Diversity Incongruency × Content Removal Rate Marginal 𝑅 2 Conditional 𝑅 2

0.219 0.799

this perception is associated with a higher likelihood of sharing incongruent viewpoints.
We do not find evidence at our level of statistical power to support a claim that more inclusive communities are associated with a greater likelihood of incongruent opinion-sharing. The interaction effect between Incongruency and Inclusion indicates a positive but non-significant relationship between perceived community inclusion and participants’ willingness to share minoritized viewpoints (𝛽 = 0.23, 𝑝 = 0.12). It is possible that in politics-oriented subreddits, community inclusion is less of a concern to users. When comparing the relative importance of different values across subreddits, Weld et al. [104] found that subreddits focused on news, media, and discussion, such as those included in our analysis, placed lower priority on inclusion as a community value.
5.2.4 Content removal has a negative relationship with incongruent viewpoint sharing behavior. Finally, we look at our measure Content Removal Rate which captures the amount of moderation within a given subreddit. While the main effect for content removal rate is not statistically significant (𝛽 = 0.04, 𝑝 = 0.69), there is a significant interaction between removal rate and incongruency (𝛽 = −0.31, 𝑝 = 0.03). Thus, the association between moderation and participants’ likelihood of commenting depends on how incongruent their viewpoint is. This result contradicts our initial hypothesis that moderation activity would bolster viewpoint sharing by promoting a more open environment. In practice, moderation may be associated with the opposite: seeing more comments being removed could exacerbate participants’ fear of social isolation, or reflect more

Model 1c

0.244 0.810

0.254 0.810

heavy-handed ideological moderation, correlating with more selfsilencing. To pursue this question further, we explored whether there is a quadratic relationship between moderation and participation, since draconian moderation policies could stifle people’s desire to speak up [48] (see Appendix B.4 for full results). While there may be weak evidence that such trend exists for incongruent viewpoints (𝛽 = −0.32), the relationship is non-significant (𝑝 = 0.07) at our level of power. It is likely that the subreddits in our sample are not overly moderated to the point that participation is diminished or subreddit members are unaware of the extent of content removal within a community [42].

5.3

Analyzing Upvoting Behavior

So far, we have focused on posting as the primary way users express their opinions, but participants can signal their viewpoints in other ways. We also examine upvoting as an alternative form of opinion expression. Unlike posting or commenting, upvoting is anonymous.
From our survey, we find that participants are likely to upvote 65.4% of incongruent viewpoints, compared to 83.0% of congruent viewpoints. Consistent with our findings on sharing (Sec. 5.2), when disaggregated by topics, congruent viewpoints are still more likely to be upvoted. However, this difference is smaller, with congruent viewpoints only 1.33 ± 0.37 times more likely to receive an upvote.
To study the relationship between upvoting and our selected predictors, we again use a set of nested linear mixed-effects model with subreddit and participant as crossed random effects. Our dependent variable is the likelihood of upvoting variable (measured on a 7

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

Figure 7: Across all topics, the average percentage of incongruent viewpoints likely to be upvoted is 65.4%. On the left, we report the percentage of incongruent viewpoints that participants are willing upvote (i.e., Upvote Likelihood > 4) out of all incongruent viewpoints for the given topic. On the right, we do the same for congruent viewpoints.
point Likert scale). In our models, we use the same control variables and predictors from our previous analysis on sharing likelihood (see Sec. 5.2). The following analysis was not pre-registered and should be considered post-hoc.

5.3.1 Upvoting is more common than commenting. We begin by examining how likely participants are to upvote comments across subreddits. As seen in Table 3, the intercept is 5.63 for Model 2a.
This indicates that it is “slightly” to “moderately” likely that a non-Male, non-White, and non-Democrat individual who posts on Reddit rarely to occasionally, has an account age of 6.2 years, and WTSC of 3.68 out of 8 would upvote a congruent viewpoint. Similar to our results in Sec. 5.2, we find a significant positive correlation between posting frequency and upvoting (𝛽 = 0.57, 𝑝 < 0.001) as well as agreement intensity and upvoting (𝛽 = 0.48, 𝑝 < 0.001).
Overall, participants are more likely to upvote a comment (𝑀 = 5.47) compared to posting the comment on their main account (𝑀 = 3.87).
5.3.2 Incongruent viewpoints are less likely to be upvoted, even when people agree with the stance. Similar to participants’ behavior with sharing viewpoints on Reddit, we observe that participants are less likely to upvote incongruent comments (𝑀 = 4.93) compared to congruent ones (𝑀 = 5.63). While we also observe a negative relationship between Incongruency and the likelihood of upvoting

(𝛽 = −0.22 in Model 2b), this association is not statistically significant, likely due to lack of statistical power (𝑝 = 0.07). Furthermore, unlike with commenting, we do not observe any interaction effects between our community design factors (i.e., inclusion, diversity, and moderation) and opinion congruency. This result indicates that these factors are not associated with participants’ decision to upvote a congruent viewpoint compared to an incongruent one.
Following Noelle-Neumann’s theory [74], we do expect the spiral of silence to be less pronounced for upvoting compared to posting or commenting as it is completely anonymous, making it less likely that participants anticipate social isolation or negative reactions as a consequence of their actions.

5.3.3 Upvoting provides an alternative for sharing otherwise selfsilenced opinions. Overall, upvoting provides a valuable avenue for users to express their opinion. We compare the number of viewpoints that users report that they are not willing to comment on but are willing to upvote. In congruent conditions, 65.8% of viewpoints that would not be posted (Share Likelihood ≤ 4) would be upvoted (Upvote Likelihood > 4). While this percentage of viewpoints is lower in comparison for incongruent viewpoints, for half of the incongruent viewpoints (53.3%), participants are still likely to use upvoting as a mechanism for expressing their opinions, even when they are unwilling to post them.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 3: Incongruent viewpoints are less likely to be upvoted compared to congruent viewpoints. We present the results of a linear mixed-effect model that predicts the likelihood of upvoting as a hierarchical regression. Model 2a only includes the controls; Model 2b has the controls and fixed effects for Incongruency, community values, and community moderation; and Model 2c adds interaction effects. Note: ∗ p<0.05; ∗∗ p<0.01; ∗∗∗ p<0.001 Model 2a Coef.
(Intercept)
Controls WTSC Agreement Intensity Male (0/1)
White (0/1)
Democrat (0/1)
Account Tenure Posting

Model 2b

SE

𝑝

5.63

0.43

0.00∗∗∗

0.07 0.48 −0.12 −0.29 0.31 −0.02 0.57

0.18 0.05 0.40 0.39 0.36 0.17 0.17

0.68 0.00∗∗∗ 0.76 0.46 0.38 0.89 0.00∗∗

Independent Variables Incongruency (0/1)
Inclusion Diversity Content Removal Rate

Coef.

SE

𝑝

Coef.

SE

𝑝

5.72

0.42

0.00∗∗∗

5.73

0.42

0.00∗∗∗

0.07 0.46 −0.15 −0.27 0.24 −0.03 0.55

0.18 0.05 0.39 0.38 0.35 0.17 0.17

0.69 0.00∗∗∗ 0.71 0.48 0.51 0.88 0.00∗∗

0.07 0.46 −0.15 −0.27 −0.23 −0.03 0.55

0.18 0.05 0.39 0.38 0.35 0.17 0.17

0.70 0.00∗∗∗ 0.71 0.48 0.52 0.88 0.00∗∗

−0.22 0.09 −0.00 0.06

0.12 0.08 0.07 0.06

0.07 0.27 0.99 0.37

−0.20 −0.08 −0.02 0.08

0.12 0.08 0.08 0.07

0.10 0.32 0.83 0.28

0.02 0.08 −0.08

0.12 0.12 0.12

0.89 0.50 0.50

Incongruency × Inclusion Incongruency × Diversity Incongruency × Content Removal Rate Marginal 𝑅 2 Conditional 𝑅 2

6

0.216 0.736

Discussion

In our work, we seek to measure the extent of self-silencing behavior across different political subreddits. Through our analysis, we find evidence that confirms the spiral of silence theory and uncovers community design factors that can help counteract these silencing effects on Reddit. A summary of our hypotheses and findings can be found in Table 4. In this section, we cover both the theoretical implications for understanding the spiral of silence on social media and complementary design recommendations.

6.1

Community Factors and the Spiral of Silence

We identify community design factors on Reddit that can inform design decisions to counteract silencing effects. Previous works [28, 70, 107, 108] have focused on how platform-level designs influences the spiral of silence. For example, Neubaum [70] studied how perceptions of message persistence decrease the likelihood that users will share incongruent viewpoints. In practice, it is difficult for social media users and moderators to change platform design. Intervening at the community design level provides a more actionable alternative. From our analysis, we found that perceived diversity is positively associated with the likelihood of sharing when individuals consider their opinion to be in the minority, whereas perceived content removal rates are negatively associated with sharing incongruent viewpoints. Since moderators have more power to influence the values and norms of online communities, this finding suggest

Model 2c

0.226 0.735

0.227 0.733

that there are feasible interventions to mitigate self-silencing within a community without having to change platform-level design.
In line with Hypothesis 2b, we observe a positive correlation between participants’ perceptions of community diversity (i.e., how diverse they believe people in the subreddit are) and their willingness to post incongruous viewpoints on politically-oriented subreddits. While this result suggests there are actionable changes to community design factors that can reduce self-silencing, we also acknowledge that fostering a hetereogenous subreddit poses a challenging task for moderators. As prior work has found, there are increasing amounts of fragmentation on the platform with smaller, more thematically specified subreddits forming and pulling away users from larger, more aggregated communities [97]. On one hand, this diversification means that users are likely to find a community where their viewpoints are not minoritized. However, this behavior may lead to more homogeneity within subreddits, discouraging community members from sharing viewpoints they perceive as being in the minority. One path forward, however, can be to encourage new users to join and participate within a community. As Duguay [24] found, long-term active users within a subreddit express express more opinion homogeneity whereas heterogeneity within communities can be linked to new users joining. Thus, moderators can employ design interventions that encourage new joiners within the community [48]. Examples that have been helpful on Reddit and online peer-production communities include having beginner FAQs [104], introducing automated recommendations for topics

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

Table 4: Summary of hypotheses and findings. ✓ indicates the hypothesis is supported and “NS” indicates the hypothesis is not supported.
Hypothesis

Result

H1: The likelihood of opinion expression is negatively associated with viewpoint incongruency.

✓

H2a: The perceived inclusivity of a community is more positively associated with the likelihood of opinion expression when users hold incongruent viewpoints.

NS

H2b: The diversity of community members is more positively associated with the likelihood of opinion expression when users hold incongruent viewpoints.

✓

H3: Moderation activity is more positively associated with the likelihood of opinion expression when users hold incongruent viewpoints.

NS

new members may be interested in [111], and using flairs / badges for newcomers [90]. Another lightweight intervention is to highlight diversity through positive reinforcement. Prior work [50, 51]
found that positive feedback on Reddit, such as awarding gold, can help set community norms and encourage participation — a practice that especially benefits newcomers. A similar mechanism could be implemented by moderators to showcase diverse viewpoints in the community, helping combat the spiral of silence by demonstrating that this type of opinion-sharing is not only invited but also actively encouraged.
Contrary to Hypothesis 3, we find that higher content removal rates within subreddits are associated with a decrease in participants’ willingness to comment incongruent viewpoints. We initially expected that more content removal would assuage users’ fear of retaliation or social isolation when sharing unpopular opinions.
However, one explanation for this negative relationship is that users in subreddits with stricter moderation may be worried that their own comments will be removed by moderators, and this negative reaction is associated with a lower likelihood of speaking out [30].
A known issue with content removal is that users often find the process to be opaque [43, 45]. This lack of transparency leads users’ to develop their own folk theories as to why their content was removed, often attributing it to them posting unpopular opinions or the moderators’ own political biases [43]; in practice, it may be that the posts were violating community rules. Increasing transparency, even in the form of lightweight explanations as to why the removal occurred, can help users understand moderators’ decisions. Otherwise, users may believe that expressing minoritized viewpoints will engender social sanctions, leading them to remain silent.

6.2

Platform Affordances and the Spiral of Silence

Prior work has compared how affordances including network association, social presence, and message persistence influence the degree of self-silencing across social media platforms, including Facebook and Twitter [70, 80, 98]. Reddit’s distinct affordances offer further insight into two platform characteristics that impact the spiral of silence. First, since Reddit consists of multiple online communities, the porous boundaries offer users more flexibility in choosing spaces where they feel their opinions are more in line

with the majority. Second, Reddit offers users a greater degree of perceived anonymity in comparison to other social media platforms, such as Facebook, where the spiral of silence has been previously studied [9, 30, 40]. Within Reddit, users are also afforded varying degrees of anonymity depending on their choice of commenting or upvoting. We build on prior work [28, 107, 108], which has explored how perceived anonymity influences self-silencing behaviors.
6.2.1 Users have options to express their opinions across multiple communities. Belonging to multiple communities on one platform give users more flexibility to find a space where they may feel comfortable expressing their opinion. On Reddit, users can easily join and navigate across many subreddits, and, in fact, 93.1% (N=54) of surveyed participants reported being active members of more than one community. This functionality is important to take into account because whether a viewpoint is incongruent varies depending on the community. For example, we find that a pro-life viewpoint is in the minority for many of the subreddits we examine (e.g., r/politics, r/news, etc.), but it is the majority for r/conservative. Albeit rare, we also found instances where participants report being more willing to share the same viewpoint in a different subreddit where their stance is aligned with the majority versus when they view themselves as being in the minor.
Of note, we also found cases of the opposite — where participants were more likely to share their viewpoint to a subreddit where it would be in the minority. We only observed this behavior for the subreddits r/changemyview and r/conspiracy where it is possible that sharing minoritized viewpoints may be more accepted.
Overall, this malleability can be beneficial since it means users are likely to find outlets where they feel comfortable freely sharing their opinions. Alternatively, there is a concern that this behavior can contribute to an “echo chamber” [19] effect, as users only participate in communities where they know others will agree with them.
6.2.2 Self-silencing persists under pseudoanonymity. As prior work has pointed out [28, 64], how identifiable participants perceive themselves to be has an influence on their willingness to share their opinion. In contrast to other social media platforms, such as Facebook, where the spiral of silence has been studied before [30, 40], Reddit provides a greater sense of perceived anonymity to users [9].

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

Yet, even under these conditions, our results indicate that participants are still less willing to align themselves with incongruent opinions. While Reddit accounts are not directly linked to users’ offline identities, users may not want certain content linked to their main account or long-term identity on the platform [55]. This rationale explains why users may choose to make “throwaway accounts” — a temporary identity created under a different pseudonym — on Reddit, especially when discussing stigmatized topics [2, 22]. In fact, when we surveyed participants what other actions they would take if they were not willing to comment an incongruent viewpoint under their main account, participants selected posting from a throwaway account as the preferred course of action for eleven viewpoints. The persistence of content on social media — in comparison to offline communication where there is no permanent trace of what people have said — can exacerbate self-silencing behaviors, especially on platforms with higher degrees of association, as users may worry that people they know will find or look through prior posts [28, 98].
We also observe that participants are less likely to upvote incongruent opinions compared to congruent ones. Given that upvoting is completely anonymous, fear of social isolation is likely to be less pressing of a concern for users. This finding is also consistent with results from Woong Yun and Park [107] who found that participants were unwilling to share opinions they perceived to be in the minority even in a completely anonymous online forum, suggesting that despite platform affordances, people may feel a fear of isolation which leads to self-silencing. Another explanation for this behavior may be that upvoting does not solely indicate agreement but also serves as an indication that the content is worth sharing in the community [69]. Since participants believe incongruent viewpoints are only held by a minority of subreddit members, they might also view the content as less worthwhile to share, decreasing the likelihood of upvoting.

video games and sports — r/gaming and r/nba), respectively (see Appendix A.2 for details). Future work can adapt this method to explore other online communities and topics.

6.3

Confirming and Measuring the Spiral of Silence Online

We find robust evidence that participants are less likely to share their viewpoint when they perceive their opinion to be in the minority. Corroborating Gearhart and Zhang [31]’s findings, we observe that the rate of self-silencing differs across issue types, although as a whole participants are consistently less willing to express incongruent viewpoints. The impact of the spiral of silence also differs across communities. For example, on r/conservative, the mean likelihood of sharing a congruent opinion is 3.98 ± 0.30 versus 3.10 ± 0.64 for an incongruent opinion. However, for r/politics, the gap is much larger between sharing a congruent (3.15 ± 0.24)
versus incongruent viewpoint (1.68±0.42). Overall, there is a consistent pattern that participants are less willing to express incongruent viewpoints.
We also introduce a method for identifying controversial viewpoints within a community and then measuring the extent of the spiral of silence. This method provides a flexible tool for generating topics that can be tailored to a specified community. In this work, we focused on identifying political and social issues, but our method can be applied to other topic areas. For example, we can generate plausible viewpoints even when using communities that focus on

6.4

When (or When Not) to Reduce Self-Silencing

As stated in Sec. 4, the scope of our work is intentionally limited to viewpoints that plausibly fall within the bounds of a community while still eliciting meaningful disagreement. We provide design interventions in Sec.6.1 to mitigate the spiral of silence for such opinions, as suppressing these viewpoints can distort broader understandings of public opinion and prevent communities from engaging with the full spectrum of legitimate viewpoints. However, it is important to acknowledge that reducing self-silencing is not universally desirable. Encouraging the expression of harmful content, such as hate speech or misinformation, or viewpoints that violate community norms can be counterproductive, as toxicity often amplifies rather than reduces self-silencing among other users [30, 79].
Prior work has shown that elevated toxicity in online political conversations leads many to withhold participation to avoid harassment or conflict [44, 62]. Moreover, the spiral of silence may operate differently for harmful content. For example, Chaudhry and Gruzd [12] found that Facebook users remained willing to express racist discourse, despite it constituting a minority opinion, suggesting that perceived social sanctions do not uniformly suppress all types of opinion expression. These findings underscore an important caveat: interventions to reduce self-silencing must carefully distinguish between encouraging legitimate minority viewpoints be heard versus moderating harmful content within a community.

6.5

Limitations and Future Work

6.5.1 Use of Self-Reported Measures. Our work provides a generalizable method for generating controversial topics and then measuring rates of self-silencing in communities using survey measures.
Following prior work [16, 29, 31], we use self-reported measures of opinion expression for sharing and upvoting to capture the spiral of silence. This approach may lack ecological validity compared to studying sharing behavior directly on the platform, as participants’ reported willingness to share may differ from their situated behaviors. Prior work [91] has also critiqued the use of a hypothetical scenario when measuring likelihood of opinion expression as it may not faithfully capture how participants would act in a realistic setting. Furthermore, by only studying self-reports, we are unable to analyze what participants would say if they were to express their opinion; our current measure is unable to capture the nuances between participants boldly stating their stance on a topic versus making a hedging or lukewarm statement. Nonetheless, our approach remains well validated in prior work [16, 29, 31].
A fruitful avenue for future work is to develop new methods for measuring participants’ opinion expression behavior beyond relying on hypothetical self-reports. For example, building systems that can directly measure self-silencing on the social media platform would provide more realistic accounts of user behavior compared to surveys. Another option could be to collect observed behavior traces (e.g., past comments and upvotes on Reddit) or account information (e.g., degree of anonymity on their profile) that can be linked to

CHI ’26, April 13–17, 2026, Barcelona, Spain

self-report data. This information allows us to better model users’ behavior on the platform, providing a more holistic picture about the users’ self-silencing behaviors.
6.5.2 Study Methodology. In addition, there are limitations with our methodology that can be explored in future studies. First, we take an automated approach to generating topics and viewpoints.
While we include multiple validation steps to ensure that these topics are relevant and plausible for a given subreddit, this method prioritizes precision over recall; we are not guaranteed to capture the full range of controversial topics within a community. For example, there may be niche topics specific to certain online communities that our current approach misses. Second, we recruited our survey participants using the Prolific platform, which can introduce selection bias. This recruitment method also limited the amount of information we could collect about participants (e.g., Reddit username), meaning that although we screened participants using our knowledge quiz to validate subreddit activity, we are still relying on self-report data. Third, in our survey, we ask participants to select from two opposing viewpoints, which may not fully capture the spectrum of stances people have on a topic. We note that in empirical studies of the spiral of silence, it is common to operationalize a topic into a binary choice for survey respondents [16, 31, 66, 72].
Expanding the range of viewpoints considered could offer a more nuanced understanding of how different minority perspectives interact within online discussions or shed insight into whether the spiral of silence manifests when there is not a prevailing perceived majority opinion on a controversial topic. Our work also only provides insights on correlations between incongruency and opinion sharing; we are not able to make causal claims about this relationship.
There are many extensions to the study methodology that would provide a richer understanding of the spiral of silence. For example, future work can explore how different operationalizations of measurements in our study, such as treating incongruency as continuous or ordinal rather than binary, or having different definitions of community design factors are related to self-silencing behavior.
We can also consider new methods for obtaining surveyed topics;
one possible extension is to augment the generated topics with crowdsourced viewpoints from community members, although particular care must be taken to ensure the responses themselves are not skewed by the silencing effects already occurring within the community. Finally, to better understand causality, future work can draw on experimental methods to test whether placing users in incongruent versus congruent environment shapes their likelihood of sharing opinion in a controlled fashion [108, 112].
6.5.3 Generalizability of Results. Finally, it is important to note that our analyses are centered on politically-oriented communities on Reddit. This decision raises two limitations. First, we consider a limited subset of subreddits on Reddit. We chose to study politicallyoriented communities, as Noelle-Neumann [74] asserts that the spiral of silence occurs for morally-laden topics, such as those related to politics or social issues. Focusing on these communities allows us to study contexts where individuals are more likely to experience pressure to self-censor, making the phenomenon more observable.
However, given the diversity of community values and norms on Reddit [103], this focus limits the extent to which our findings may

Zhao et al.

apply to subreddits centered on neutral or less value-laden topics.
Second, we focus our study on a single platform, Reddit, to explore how community design factors are associated with the spiral of silence. However, prior work has shown how the spiral of silence differs across social media platforms (e.g., Facebook compared to Twitter [80]). Since platform affordances can mediate how the spiral of silence manifests, this may limit the transferability of the results from this work to other platforms. Overall, future exploration on other topic areas, communities, and platforms can help bolster the generalizability of these findings.

7

Conclusion

In this work, our goal is to measure the extent of the spiral of silence across different online political communities. We also seek to identify community design factors that may amplify or mitigate self-silencing effects. Since directly measuring what is left unsaid using social media data is not possible, we propose a new method using LLMs to propose community-specific topics and viewpoints that are likely to be controversial. Then, we survey users’ likelihood of opinion expression, opinion incongruency, and perceived community values. In total, we collected 439 responses capturing subreddit community members’ likelihood of opinion expression across twelve subreddits and eleven topics. We find robust evidence that participants are less likely to share their viewpoint when they consider themselves to be in the minority. This finding corroborates that the spiral of silence manifests even on Reddit, which affords users a greater degree of perceived anonymity compared to other social media platforms.
Although self-silencing behavior is prevalent across the subreddits and topics covered in our study, we identify community-level design decisions that can help mitigate the spiral of silence. We find that perceived diversity is positively associated with sharing minority opinions, while higher content removal rates are related to a decrease in willingness to share. Moderators can foster more heterogeneous communities by encouraging newcomer participation, using positive reinforcement to highlight diverse viewpoints, and increasing transparency in content removal decisions to help users understand that moderation targets rule violations rather than unpopular opinions. These design interventions can help a foster an environment where users do not feel as if they will be penalized for speaking out, provide actionable levers for encouraging opinion-sharing.

Acknowledgments This work was funded by the Brown Institute for Media Innovation and Stanford HAI Seed Grant. Dora Zhao is also supported in part by the Paul and Daisy Soros Fellowship for New Americans. We thank Jordan Troutman, Tiziano Piccardi, Jacy Anthis, Lindsay Popowski, Omar Shaikh, and other members of Stanford HCI for their helpful comments and suggestions. We also thank Galen Weld for providing access to annotations from Media Bias/Fact Check.

References [1] [n. d.]. The Rules of /r/Politics:. https://www.reddit.com/r/politics/wiki/index/.
[2] Tawfiq Ammari, Sarita Schoenebeck, and Daniel Romero. 2019. Self-declared throwaway accounts on Reddit: How platform affordances and shared norms

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

enable parenting disclosure and support. Proc. ACM Hum.-Comput. Interact. 3, CSCW (2019).
[3] Nick Anstead and Ben O’Loughlin. 2015. Social media analysis and public opinion: The 2010 UK general election. Journal of Computer-Mediated Communication 20, 2 (2015), 204–220.
[4] Sitaram Asur and Bernardo A Huberman. 2010. Predicting the future with social media. In IEEE WIC ACM International Conference on Web Intelligence (WI), Vol. 1. IEEE.
[5] Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy Blackburn. 2020. The pushshift reddit dataset. (2020).
[6] Bethany A Bell, John M Ferron, and Jeffrey D Kromrey. 2008. Cluster size in multilevel models: The impact of sparse data structures on point and interval estimates in two-level models. JSM Proceedings, Section on Survey Research Methods (2008), 1122–1129.
[7] Sam Bestvater, Sono Shah, Gonzalo River, and Aaron Smith. 2022.
Politics on twitter: One-third of tweets from us adults are political.
(2022). https://www.pewresearch.org/politics/2022/06/16/politics-on-twitterone-third-of-tweets-from-u-s-adults-are-political/ [8] Shraey Bhatia, Jey Han Lau, and Timothy Baldwin. 2018. Topic intrusion for automatic topic model evaluation. In Conference on Empirical Methods in Natural Language Processing (EMNLP).
[9] Shelley Boulianne, Christian P Hoffmann, and Michael Bossetta. 2024. Social media platforms for politics: A comparison of Facebook, Instagram, Twitter, YouTube, reddit, snapchat, and WhatsApp. New Media & Society (2024).
[10] Frances Bowen and Kate Blackmon. 2003. Spirals of silence: The dynamic effects of diversity on organizational voice. Journal of Management Studies 40, 6 (2003), 1393–1417.
[11] Jonathan Chang, Sean Gerrish, Chong Wang, Jordan Boyd-Graber, and David Blei. 2009. Reading tea leaves: How humans interpret topic models. Advances in Neural Information Processing Systems (NeurIPS) (2009).
[12] Irfan Chaudhry and Anatoliy Gruzd. 2020. Expressing and challenging racist discourse on Facebook: How social media weaken the “spiral of silence” theory.
Policy & Internet 12, 1 (2020), 88–108.
[13] Hsuan-Ting Chen. 2018. Spiral of silence on social media and the moderating role of disagreement and publicness in the network: Analyzing expressive and withdrawal behaviors. New Media & Society 20, 10 (2018).
[14] Zoey Chen and Jonah Berger. 2013. When, why, and how controversy causes conversation. Journal of Consumer Research 40, 3 (2013), 580–593.
[15] Justin Cheng, Michael Bernstein, Cristian Danescu-Niculescu-Mizil, and Jure Leskovec. 2017. Anyone can become a troll: Causes of trolling behavior in online discussions. In ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW).
[16] Stella C Chia. 2014. How authoritarian social contexts inform individuals’ opinion perception and expression. International Journal of Public Opinion Research 26, 3 (2014), 384–396.
[17] Daejin Choi, Jinyoung Han, Taejoong Chung, Yong-Yeol Ahn, Byung-Gon Chun, and Ted Taekyoung Kwon. 2015. Characterizing conversation patterns in reddit:
From the perspectives of content properties and user participation behaviors.
In ACM Conference on Online Social Networks (COSN).
[18] Rob Churchill and Lisa Singh. 2022. The evolution of topic modeling. Comput.
Surveys 54, 10s (2022).
[19] Matteo Cinelli, Gianmarco De Francisci Morales, Alessandro Galeazzi, Walter Quattrociocchi, and Michele Starnini. 2021. The echo chamber effect on social media. Proceedings of the National Academy of Sciences 118, 9 (2021).
[20] Nate Cohn and Kevin Quealy. 2019.
The Democratic Electorate on Twitter Is Not the Actual Democratic Electorate. The New York Times (2019). https://www.nytimes.com/interactive/2019/04/08/upshot/democraticelectorate-twitter-real-life.html [21] Francis S Dalisay. 2012. The spiral of silence and conflict avoidance: Examining antecedents of opinion expression concerning the US military buildup in the Pacific island of Guam. Communication Quarterly 60, 4 (2012), 481–503.
[22] Munmun De Choudhury and Sushovan De. 2014. Mental health discourse on reddit: Self-disclosure, social support, and anonymity. In International AAAI Conference on Web and Social Media (ICWSM).
[23] Munmun De Choudhury, Sanket S Sharma, Tomaz Logar, Wouter Eekhout, and René Clausen Nielsen. 2017. Gender and cross-cultural differences in social media disclosures of mental illness. In ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW).
[24] Philippe A Duguay. 2022. Read it on Reddit: Homogeneity and ideological segregation in the age of social news. Social Science Computer Review 40, 5 (2022), 1186–1202.
[25] Amy C Edmondson and Zhike Lei. 2014. Psychological safety: The history, renaissance, and future of an interpersonal construct. Annual Review of Organizational Psychology and Organizational Behavior (2014).
[26] Mike Farjam and Karl Loxbo. 2024. Social conformity or attitude persistence?
The bandwagon effect and the spiral of silence in a polarized context. Journal of Elections, Public Opinion and Parties 34, 3 (2024), 531–551.

[27] Casey Fiesler, Jialun Jiang, Joshua McCann, Kyle Frye, and Jed Brubaker. 2018.
Reddit rules! characterizing an ecosystem of governance. In International AAAI Conference on Web and Social Media (ICWSM).
[28] Jesse Fox and Lanier Frush Holt. 2021. Fear of isolation and perceived affordances: The spiral of silence on social networking sites regarding police discrimination. In Social Media News and Its Impact. Routledge, 147–168.
[29] Sherice Gearhart and Weiwu Zhang. 2014. Gay bullying and online opinion expression: Testing spiral of silence in the social media environment. Social Science Computer Review 32, 1 (2014), 18–36.
[30] Sherice Gearhart and Weiwu Zhang. 2015. “Was it something I said?”“No, it was something you posted!” A study of the spiral of silence theory in social media contexts. Cyberpsychology, Behavior, and Social Networking 18, 4 (2015), 208–213.
[31] Sherice Gearhart and Weiwu Zhang. 2018. Same spiral, different day? Testing the spiral of silence across issue types. Communication Research 45, 1 (2018), 34–54.
[32] Anna Gibson. 2019. Free speech and safe spaces: How moderation policies shape online discussion spaces. Social Media + Society 5, 1 (2019).
[33] Wei Gong, Ee-Peng Lim, and Feida Zhu. 2015. Characterizing silent users in social media communities. In International AAAI Conference on Web and Social Media (ICWSM).
[34] Bonnie MK Hagerty, Judith Lynch-Sauer, Kathleen L Patusky, Maria Bouwsema, and Peggy Collier. 1992. Sense of belonging: A vital mental health concept.
Archives of Psychiatric Nursing 6, 3 (1992), 172–177.
[35] Bonnie M Hagerty, Reg A Williams, James C Coyne, and Margaret R Early.
1996. Sense of belonging and indicators of social and psychological functioning.
Archives of Psychiatric Nursing 10, 4 (1996), 235–244.
[36] Keither Hampton, Lee Rainie, Weixu Lu, Maria Dwyer, Inyoung Shin, and Kristen Purcell. 2014. Social Media and the ‘Spiral of Silence’. Technical Report.
Washington, D.C. https://www.pewresearch.org/internet/2014/08/26/socialmedia-and-the-spiral-of-silence/ [37] Andrew F Hayes, Carroll J Glynn, and James Shanahan. 2005. Willingness to self-censor: A construct and measurement tool for public opinion research.
International Journal of Public Opinion Research 17, 3 (2005), 298–323.
[38] Jack Hessel and Lillian Lee. 2019. Something’s Brewing! Early Prediction of Controversy-causing Posts from Discussion Features. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL).
[39] Shirley S Ho, Vivian Hsueh-Hua Chen, and Clarice C Sim. 2013. The spiral of silence: Examining how cultural predispositions, news attention, and opinion congruency relate to opinion expression. Asian Journal of Communication 23, 2 (2013), 113–134.
[40] Christian Pieter Hoffmann and Christoph Lutz. 2017. Spiral of silence 2.0: Political self-censorship among young Facebook users. In International Conference on Social Media & Society (SMSociety).
[41] Peter Jamison and Scott Clement. 2019. Virginians are split on governor’s fate amid blackface scandal, poll shows. The Washington Post (2019).
https://www.washingtonpost.com/local/virginia-politics/virginians-split-ongovernors-fate-amid-blackface-scandal-poll-shows/2019/02/09/93002e842bc1-11e9-b011-d8500644dc98_story.html?noredirect=on [42] Shagun Jhaver, Darren Scott Appling, Eric Gilbert, and Amy Bruckman. 2019.
“Did you suspect the post would be removed?” Understanding user reactions to content removals on Reddit. Proc. ACM Hum.-Comput. Interact. 3, CSCW (2019).
[43] Shagun Jhaver, Amy Bruckman, and Eric Gilbert. 2019. Does transparency in moderation really matter? User behavior after content removal explanations on reddit. Proc. ACM Hum.-Comput. Interact. 3, CSCW (2019).
[44] Gabriela Juncosa, Taha Yasseri, Julia Koltai, and Gerardo Iniguez. 2024. Toxic behavior silences online political conversations. arXiv preprint arXiv:2412.05741 (2024).
[45] Prerna Juneja, Deepika Rama Subramanian, and Tanushree Mitra. 2020. Through the looking glass: Study of transparency in Reddit’s moderation practices. Proc.
ACM Hum.-Comput. Interact. 4, GROUP, Article 17 (2020).
[46] Steven J Karau and Kipling D Williams. 1993. Social loafing: A meta-analytic review and theoretical integration. Journal of Personality and Social Psychology 65, 4 (1993), 681.
[47] Amandeep Kaur and James R Wallace. 2024. Moving Beyond LDA: A Comparison of Unsupervised Topic Modelling Techniques for Qualitative Data Analysis of Online Communities. arXiv preprint arXiv:2412.14486 (2024).
[48] Robert E Kraut and Paul Resnick. 2012. Building successful online communities:
Evidence-based social design. MIT Press.
[49] Matthew J Kushin, Masahiro Yamamoto, and Francis Dalisay. 2019. Societal majority, Facebook, and the spiral of silence in the 2016 US presidential election.
Social Media + Society 5, 2 (2019).
[50] Charlotte Lambert. 2024. Proactively Supporting Online Community Health Through Mechanisms of Positive Reinforcement. In Companion Publication of the ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW). 35–38.

CHI ’26, April 13–17, 2026, Barcelona, Spain

[51] Charlotte Lambert, Koustuv Saha, and Eshwar Chandrasekharan. 2025. Does Positive Reinforcement Work?: A Quasi-Experimental Study of the Effects of Positive Feedback on Reddit. In ACM CHI Conference on Human Factors in Computing Systems.
[52] Nathaniel M Lambert, Tyler F Stillman, Joshua A Hicks, Shanmukh Kamble, Roy F Baumeister, and Frank D Fincham. 2013. To belong is to matter: Sense of belonging enhances meaning in life. Personality and Social Psychology Bulletin 39, 11 (2013), 1418–1427.
[53] Cliff Lampe, Rick Wash, Alcides Velasquez, and Elif Ozkaya. 2010. Motivations to participate in online communities. In ACM CHI Conference on Human Factors in Computing Systems.
[54] Liliana Laranjo, Amaël Arguel, Ana L Neves, Aideen M Gallagher, Ruth Kaplan, Nathan Mortimer, Guilherme A Mendes, and Annie YS Lau. 2015. The influence of social networking sites on health behavior change: a systematic review and meta-analysis. Journal of the American Medical Informatics Association 22, 1 (2015), 243–256.
[55] Alex Leavitt. 2015. “This is a Throwaway Account” Temporary Technical Identities and Perceptions of Anonymity in a Massive Online Community. In ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW).
[56] Waipeng Lee, Benjamin H Detenber, Lars Willnat, Sean Aday, and Joseph Graf.
2004. A cross-cultural test of the spiral of silence theory in Singapore and the United States. Asian Journal of Communication 14, 2 (2004), 205–226.
[57] Leon Leibmann, Galen Weld, Amy X Zhang, and Tim Althoff. 2025. Reddit Rules and Rulers: Quantifying the Link Between Rules and Perceptions of Governance Across Thousands of Communities. In International AAAI Conference on Web and Social Media (ICWSM).
[58] Hanlin Li, Brent Hecht, and Stevie Chancellor. 2022. All that’s happening behind the scenes: Putting the spotlight on volunteer moderator labor in Reddit. In International AAAI Conference on Web and Social Media (ICWSM).
[59] Q Vera Liao, Wai-Tat Fu, and Markus Strohmaier. 2016. #Snowden: Understanding biases introduced by behavioral differences of opinion groups on social media. In ACM CHI Conference on Human Factors in Computing Systems.
[60] Yu Liu, Jian Raymond Rui, and Xi Cui. 2017. Are people willing to share their political opinions on Facebook? Exploring roles of self-presentational concern in spiral of silence. Computers in Human Behavior 76 (2017), 294–302.
[61] Cora JM Maas and Joop J Hox. 2005. Sufficient sample sizes for multilevel modeling. Methodology 1, 3 (2005), 86–92.
[62] Michalis Mamakos and Eli J Finkel. 2023. The social media discourse of engaged partisans is toxic even when politics are irrelevant. PNAS Nexus 2, 10 (2023).
[63] Abraham Harold Maslow. 1958. A Dynamic Theory of Human Motivation.
(1958), 26–47.
[64] Jörg Matthes and Andrew F Hayes. 2014. Methodological conundrums in spiral of silence research. In The Spiral of Silence. Routledge, 54–64.
[65] Jörg Matthes, Johannes Knoll, and Christian von Sikorski. 2018. The “spiral of silence” revisited: A meta-analysis on the relationship between perceptions of opinion support and political opinion expression. Communication Research 45, 1 (2018), 3–33.
[66] Jörg Matthes, Kimberly Rios Morrison, and Christian Schemer. 2010. A spiral of silence for some: Attitude certainty and the expression of political minority opinions. Communication Research 37, 6 (2010), 774–800.
[67] Shannon C McGregor. 2020. “Taking the temperature of the room” how political campaigns use social media to understand and represent public opinion. Public Opinion Quarterly 84, S1 (2020), 236–256.
[68] Patrick F McKay, Derek R Avery, Scott Tonidandel, Mark A Morris, Morela Hernandez, and Michelle R Hebl. 2007. Racial differences in employee retention:
Are diversity climate perceptions the key? Personnel Psychology 60, 1 (2007), 35–62.
[69] Carrie Moore and Lisa Chuang. 2017. Redditors revealed: Motivational factors of the Reddit community. In Hawaii International Conference on System Sciences.
[70] German Neubaum. 2022. “It’s going to be out there for a long time”: The influence of message persistence on users’ political opinion expression in social media. Communication Research 49, 3 (2022), 426–450.
[71] German Neubaum and Nicole C Krämer. 2018. What do we fear? Expected sanctions for expressing minority opinions in offline and online communication.
Communication Research 45, 2 (2018), 139–164.
[72] Kurt Neuwirth, Edward Frederick, and Charles Mayo. 2007. The spiral of silence and fear of isolation. Journal of Communication 57, 3 (2007), 450–468.
[73] Alexander Newman, Ross Donohue, and Nathan Eva. 2017. Psychological safety:
A systematic review of the literature. Human Resource Management Review 27, 3 (2017), 521–535.
[74] Elisabeth Noelle-Neumann. 1974. The spiral of silence a theory of public opinion.
Journal of Communication 24, 2 (1974), 43–51.
[75] Elisabeth Noelle-Neumann. 1977. Turbulences in the climate of opinion: Methodological applications of the spiral of silence theory. Public Opinion Quarterly 41, 2 (1977), 143–158.
[76] Blair Nonnecke and Jenny Preece. 2000. Lurker demographics: Counting the silent. In ACM CHI Conference on Human Factors in Computing Systems.

Zhao et al.

[77] Blair Nonnecke and Jenny Preece. 2001. Why lurkers lurk. (2001).
[78] Geoff Norman. 2010. Likert scales, levels of measurement and the “laws” of statistics. Advances in Health Sciences Education 15, 5 (2010), 625–632.
[79] Candi S Carter Olson and Victoria LaPoe. 2017.
“Feminazis,”“libtards,”“snowflakes,” and “racists”’: Trolling and the Spiral of Silence effect in women, LGBTQIA communities, and disability populations before and after the 2016 election. The Journal of Public Interest Communications 1, 2 (2017), 116–116.
[80] Mustafa Oz, Saif Shahin, and Scott B Greeves. 2024. Platform affordances and spiral of silence: How perceived differences between Facebook and Twitter influence opinion expression online. Technology in Society 76 (2024), 102431.
[81] Natalie Pang, Shirley S Ho, Alex MR Zhang, Jeremy SW Ko, WX Low, and Kay SY Tan. 2016. Can spiral of silence and civility predict click speech on Facebook?
Computers in Human Behavior 64 (2016), 898–905.
[82] Chau Pham, Alexander Hoyle, Simeng Sun, Philip Resnik, and Mohit Iyyer. 2024.
TopicGPT: A Prompt-based Topic Modeling Framework. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL).
[83] Tiziano Piccardi, Martin Saveski, Chenyan Jia, Jeffrey Hancock, Jeanne L. Tsai, and Michael S. Bernstein. 2025. Reranking partisan animosity in algorithmic social media feeds alters affective polarization. Science 390, 6776 (2025).
[84] Emma Pierson. 2015. Outnumbered but well-spoken: Female commenters in the New York Times. In ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW).
[85] Joseph Reagle. 2023. Even pseudonyms and throwaways delete their Reddit posts. First Monday (2023).
[86] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing (EMNLP-IJCNLP).
[87] Yuqing Ren, Robert Kraut, and Sara Kiesler. 2007. Applying common identity and bond theory to design of online communities. Organization Studies 28, 3 (2007), 377–408.
[88] Derek Ruths and Jürgen Pfeffer. 2014. Social media for large studies of behavior.
Science 346, 6213 (2014), 1063–1064.
[89] Charles T Salmon and Hayg Oshagan. 1990. Community size, perceptions of majority opinion, and opinion expression. Journal of Public Relations Research 2, 1-4 (1990), 157–171.
[90] Tiago Santos, Keith Burghardt, Kristina Lerman, and Denis Helic. 2020. Can badges foster a more welcoming culture on Q&A boards?. In International AAAI Conference on Web and Social Media (ICWSM).
[91] Dietram A Scheufele, James Shanahan, and Eunjung Lee. 2001. Real talk: Manipulating the dependent variable in spiral of silence research. Communication Research 28, 3 (2001), 304–324.
[92] Dietram A Scheufle and Patricia Moy. 2000. Twenty-five years of the spiral of silence: A conceptual review and empirical outlook. International Journal of Public Opinion Research 12, 1 (2000), 3–28.
[93] Joseph Seering, Robert Kraut, and Laura Dabbish. 2017. Shaping pro and antisocial behavior on twitch through moderation and example-setting. In ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW).
[94] Kim Sheehan. 2015. A change in the climate: Online social capital and the spiral of silence. First Monday (2015).
[95] Brett Sherrick and Jennifer Hoewe. 2018. The effect of explicit online comment moderation on three spiral of silence outcomes. New Media & Society 20, 2 (2018), 453–474.
[96] Nakatani Shuyo. 2010. Language Detection Library for Java. http://code.google.
com/p/language-detection/ [97] Philipp Singer, Fabian Flöck, Clemens Meinhart, Elias Zeitfogel, and Markus Strohmaier. 2014. Evolution of Reddit: from the front page of the internet to a self-referential community?. In International Conference on World Wide Web (Web).
[98] Elizabeth Stoycheff. 2016. Under surveillance: Examining Facebook’s spiral of silence effects in the wake of NSA internet monitoring. Journalism & Mass Communication Quarterly 93, 2 (2016), 296–311.
[99] Gail M Sullivan and Anthony R Artino Jr. 2013. Analyzing and interpreting data from Likert-type scales. Journal of Graduate Medical Education 5, 4 (2013), 541.
[100] Na Sun, Patrick Pei-Luen Rau, and Liang Ma. 2014. Understanding lurkers in online communities: A literature review. Computers in Human Behavior 38 (2014), 110–117.
[101] David Wadden, Tal August, Qisheng Li, and Tim Althoff. 2021. The effect of moderation on online mental health conversations. In International AAAI Conference on Web and Social Media (ICWSM).
[102] Galen Weld, Maria Glenski, and Tim Althoff. 2021. Political bias and factualness in news sharing across more than 100,000 online communities. In International AAAI Conference on Web and Social Media (ICWSM).
[103] Galen Weld, Amy X Zhang, and Tim Althoff. 2022. What makes online communities ‘better’? Measuring values, consensus, and conflict across thousands of subreddits. In International AAAI Conference on Web and Social Media (ICWSM).

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

[104] Galen Weld, Amy X Zhang, and Tim Althoff. 2024. Making online communities ‘better’: a taxonomy of community values on reddit. In International AAAI Conference on Web and Social Media (ICWSM).
[105] Lars Willnat, Waipeng Lee, and Benjamin H Detenber. 2002. Individual-level predictors of public outspokenness: A test of the spiral of silence theory in Singapore. International Journal of Public Opinion Research 14, 4 (2002), 391– 412.
[106] Kevin Wise, Brian Hamman, and Kjerstin Thorson. 2006. Moderation, response rate, and message interactivity: Features of online communities and their effects on intent to participate. Journal of Computer-Mediated Communication 12, 1 (2006), 24–41.
[107] Gi Woong Yun and Sung-Yeon Park. 2011. Selective posting: Willingness to post a message online. Journal of Computer-Mediated Communication 16, 2 (2011), 201–227.
[108] Tai-Yee Wu and David J Atkin. 2018. To comment or not to comment: Examining the influences of anonymity and social support on one’s willingness to express in online news discussions. New Media & Society 20, 12 (2018), 4512–4532.
[109] Diyi Yang, Robert Kraut, and John M Levine. 2017. Commitment of newcomers and old-timers to online health support communities. In ACM CHI Conference on Human Factors in Computing Systems.
[110] Diyi Yang, Zheng Yao, and Robert Kraut. 2017. Self-disclosure and channel difference in online health support groups. In International AAAI Conference on Web and Social Media (ICWSM).
[111] Ramtin Yazdanian, Leila Zia, Jonathan Morgan, Bahodir Mansurov, and Robert West. 2019. Eliciting new wikipedia users’ interests via automatically mined questionnaires: For a warm welcome, not a cold start. In International AAAI Conference on Web and Social Media (ICWSM).
[112] Thomas Zerback and Nayla Fawzi. 2017. Can online exemplars trigger a spiral of silence? Examining the effects of exemplar opinions on perceptions of public opinion and speaking out. New Media & Society 19, 7 (2017), 1034–1051.
[113] Ling Zhao, Yaobin Lu, Bin Wang, Patrick YK Chau, and Long Zhang. 2012.
Cultivating the sense of belonging and motivating user participation in virtual communities: A social capital perspective. International Journal of Information Management 32, 6 (2012), 574–588.

<name> ${SUBREDDIT} </name> <description> ${DESCRIPTION} </description> <rules> ${RULES} </rules> Provide ${NUMBER} issues that will be controversial in r/${SUBREDDIT }.
Return just the issue, no justification.

A Additional Method Details A.1 Comparing Political Leanings

We generate viewpoints representing different stances for each topic. Again, we provide the subreddit name with the description and community rules. We use the same set of hyperparameters from topic generation for this task.
You are a member of r/${SUBREDDIT}. Your task is to generate viewpoints that r/${SUBREDDIT} members would hold on a given topic.
<name> ${SUBREDDIT} </name> <description> ${DESCRIPTION} </description> <rules> ${RULES} </rules> For each side of the issue, write a 50-word Reddit comment from the perspective of that side that follows the rules of r/${SUBREDDIT}.
Issue: ${TOPIC} Return only the comments as a list, no justification. Example output:
["I am pro-choice.", "I am pro-life."]

We compare the political leanings of active r/politics members with the orientation of content posted to the subreddit. To measure the political leanings of active members, we released a survey to Prolific users. To qualify as an “active member” of r/politics, participants must first self-identify as an active member of the subreddit and pass our knowledge quiz. For those participants that qualify, we asked for their self-reported political affiliation and the valency of this affiliation. In total, we surveyed 550 Prolific users, with 218 passing our screening requirements.
To find the political orientation of content on r/politics, we leverage the fact that the subreddit’s rules require each submission consists of the headline from an article with the corresponding URL. Following prior works [19, 102], we use partisan ratings from Media Bias/Fact Check (MBFC) to label the URLs. In total, we analyze 50,000 posts from Jan. 2022 - Dec. 2023. 59.1% (N=29,589) of the URLs have corresponding MBFC ratings.

Finally, we shorten each of the viewpoints that was generated using the following prompt. We use a temperature of 0 for this task.

A.2

To find our list of political and social issues-oriented subreddits, we use the following process. Starting with 2,040 subreddits from r/ListOfSubreddits, we look at the 50 hottest submissions in each subreddit as of January 5, 2024. We select subreddits where the majority of submissions are in English – classified using langdetect [96]
– leaving us with 1,975 subreddits. Next, using GPT-3.5-Turbo we classify whether post titles are political (see below for prompt). We follow Piccardi et al. [83] and use the definition of political content proposed by Pew Research Center [7]. We choose subreddits where more than 80% of submissions are labelled as being political (N=48).
From the remaining subreddits, we select our final list of 23 based

Generating Controversial Viewpoints

A.2.1 Prompts. Our first step is to generate a list of topics that are likely to lead to disagreement. We provide the subreddit name, description of the subreddit, and community rules retrieved using PRAW. For our study, we generate 20 topics per subreddit using GPT-4 with temperature set to 1.
You are a member of r/${SUBREDDIT}. Your task is to generate controversial issues, aka issues that will lead to disagreement between r/${SUBREDDIT} members.

Shorten a statement while retaining the same viewpoint.
The statement should be phrased as an opinion.
Text: ${VIEWPOINT} Return just the shortened statement, no justification.

A.2.2 Generated Topics and Viewpoints. In Table 5, we provide the list of all eleven topics and their corresponding viewpoints. The first five topics were generated using r/politics as the seed subreddit and the subsequent six using r/worldnews. We also provide examples of applying our topic generation method to communities outside the realm of politics or social issues. In Table 6, we list five topics each from r/nba and r/gaming.

A.3

Identifying Relevant Subreddits

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

Table 5: The eleven topics, and their respective viewpoints, used in the opinion expression survey Topic Universal Healthcare

Abortion Rights

Election Reform and Voter ID Laws

Viewpoint 1. I believe in Universal Healthcare because everyone deserves access to good health, funded by the government.
2. I believe market competition and individual insurance plans are superior to Universal Healthcare.
1. As a pro-choice supporter, I believe women’s bodily autonomy and reproductive choices are crucial for gender equality.
2. As a pro-life advocate, I believe every life from conception deserves legal protection.
1. I believe in strict voter ID laws for a secure democracy and fewer fraud allegations.
2. Stringent voter ID laws marginalize minority and low-income communities. We should make voting more accessible, not harder, and aim for higher turnout, not suppression.

Military Spending

Affirmative Action

Impact of Brexit on the European Union

Israeli-Palestinian conflict and the recognition of Jerusalem as Israel’s capital

Role of NATO in maintaining global peace

Ethics of drone warfare in the Middle East

1. Increasing military spending is vital for national security, global presence, and economic growth.
2. We should cut military spending to fund education, healthcare, and infrastructure.
1. I believe affirmative action counters systemic biases and fosters a diverse, inclusive society.
2. I believe affirmative action could unintentionally cause reverse discrimination and undermine merit, potentially increasing societal division.
1. Brexit, economically, appears detrimental to the EU, potentially signaling a decline in internationalism.
2. Brexit could potentially boost EU cohesion as member states see the difficulties of exiting.
1. Recognizing Jerusalem as Israel’s capital disrupts the peace process by favoring Israel’s contested claims over Palestinian rights.
2. Recognizing Jerusalem as Israel’s capital acknowledges Jewish ties to the city, but doesn’t negate the need for fair negotiations.
1. NATO’s collective defense and strategic alliances are crucial for global peacekeeping and international security.
2. NATO’s actions can sometimes escalate global tension by infringing on sovereignty and threatening peace.
1. Drone warfare is a necessary evil for global security due to its precision, efficiency, and safety for soldiers.
2. Drone warfare inevitably causes collateral damage, violates human rights, and induces terror.

Role of social media platforms in spreading fake news

1. Social media’s lax approach has led to a fake news epidemic, undermining informed decision-making and threatening societal stability.
2. Blaming social media for fake news is misplaced; users should fact-check and stricter content control risks censorship.

Role of the United States in the Venezuelan political crisis

1. I believe US intervention is vital for Venezuela’s democratic restoration and humanitarian aid.
2. I view US involvement in Venezuela as neo-imperialism; each country should independently handle its internal affairs.

on subreddit size. In our final survey, participants also had the option to list an additional subreddit they were an active member of, bringing us to 33 subreddits in total.
Political content on Reddit is varied and can be about officials and activists, social issues, or news and current events. Looking at the following post title, would you categorize it as POLITICAL or NOT POLITICAL content?
Answer 1 if it is POLITICAL, 0 otherwise.
Post: ${POST} Answer:

A.4

one that came from the top 50 hottest submissions of a different political subreddit. We manually checked that the title from the other subreddit was not relevant to the selected subreddit. The titles used in the knowledge quiz are listed in Table 7.

A.5

Survey Details

Participants were compensated a prorated $15/hr for completing the survey with a total of $1,070 spent (including pilot studies, screenings, etc.).
We provide the survey instrument used in the study as follows:

Knowledge Quiz

To validate that participants were members of the subreddits they selected, they were given a knowledge screening quiz. In the quiz, the participants were shown three post titles, two of which belonged to the top 50 hottest submissions of their selected subreddit and

(1) In what year did you create your Reddit account?
(2) Typically, how often do you post or comment on Reddit versus browsing what others have submitted (lurking)?
(3) Select up to 3 subreddits that you are an active member of.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 6: Our viewpoint generation process can be used for non-political communities. We present a sample of topics, and their respective viewpoints, proposed for r/NBA and r/gaming.
Topic

Viewpoint

r/NBA LeBron James vs Michael Jordan

1. LeBron’s consistent dominance and achievements arguably make him the GOAT, surpassing MJ.
2. Despite LeBron’s feats, MJ’s NBA Finals record, killer instinct, and game-changing ability make him the GOAT in my opinion.
1. I believe the 82-game season should be shortened to maintain game quality and reduce player fatigue.
2. The 82-game NBA season is crucial for testing team endurance, reducing fluke performances, and providing ample basketball for fans.

NBA Season Length

1. I believe international NBA games disrupt the season’s rhythm, fatigue players with travel, and unfairly risk their health and team performance for sport’s geographical expansion.
2. I support international NBA games as they expand the audience, grow the brand, inspire youth, and boost the sport’s global popularity, despite logistical challenges.

International NBA games

1. I strongly favor abolishing the draft system for a free market system, allowing players to choose their teams and potentially balance league competition.
2. The NBA draft is crucial for maintaining balance between small and large market teams and preventing the formation of super-teams.

NBA Draft System

Golden State Warriors 2015–2019

1. The Warriors’ 2015–2019 run boosted the league’s competitiveness by forcing teams to adapt and improve.
2. The Warriors’ 2015–2019 dominance made the NBA predictable and potentially drove fans away.

r/gaming 1. As a console gamer, I value its simplicity, exclusives, and plug-and-play convenience for relaxation.
2. As a PC enthusiast, I believe PCs provide unmatched customization, performance, and game variety for in-depth gaming.

Console vs PC

Working conditions in gaming

Large vs indie game companies

1. Crunch culture in game development is harmful and unsustainable; companies should prioritize employee wellbeing.
2. Crunch may not be ideal, but without it, could top games be made? Many vocations are stressful, not just gaming.
1. Major game companies’ dominance stifles creativity; we need more space for indie developers to innovate.
2. Big companies like Nintendo prove that dominance doesn’t necessarily kill creativity, as they continually innovate and fund ambitious projects.
1. eSports, despite lacking physical exertion, is a sport due to its demand for strategy, teamwork, and skill.
2. Esports should be classified as competitive gaming, not traditional sports due to the lack of physical exertion.

eSports as a real sport

Objectification of female characters

1. I believe game developers overly sexualize female characters, neglecting character depth.
2. Video games are fantasy and idealized characters reflect artistic vision, not objectification.

Copyright issues and modding

1. I support modders as they express themselves by reshaping games, while respecting authorship.
2. I support developers because unauthorized mods can threaten their control over their costly game creations.

Subreddit

Title 1

Title 2

Title 3

worldnews

Mongolia Commits to Fighting Corruption With International Help Stopping the Cop Cities Countrywide

Finland votes: Stubb wins presidency

Already two ufo references in Super Bowl commercials lol Biden Docs Confirm Hunter’s Pay-ToPlay Was A Family Affair

socialism politics unitedkingdom

Trump Asks Supreme Court to Pause Ruling Denying Him Absolute Immunity Sinn Féin politician secretly attended son’s PSNI graduation

Historic Newton Teachers Strike Highlights Divided MA Democratic Party Right-wing judges flaunting their bias and conflicts threaten democracy HMS Prince of Wales sailed today to participate in NATO’s Exercise Steadfast Defender

Finland votes: Stubb wins presidency Mongolia Commits to Fighting Corruption With International Help

Table 7: Examples of post titles used in the knowledge quizzes. Participants must correctly select the post titles that belong to the subreddit (i.e., those in columns “Title 1” and “Title 2”).

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

including a quadratic term for Content Removal Rate. Finally, we report results using an ordinal mixed-effects model.

B.1

Figure 8: Disaggregated content removal rates for the twelve subreddits included in our analyses. Content removal rates range from 16.0% for r/conspiracy to 41.5% for r/changemyview.

(4) Are you an active member of any other subreddits that focus on political or social issues outside of those you have already selected?
(5) Provide the name of a subreddit that you are an active member of which focuses on political or social issues that is not one of the subreddits you have already selected.
(6) How diverse are the people for each subreddit?
(7) How included and able to contribute do new and existing members feel for each subreddit?
(8) Select up to 5 topics that are relevant to the subreddit.
(9) Which viewpoint on the topic do you agree with more?
(10) Indicate your level of agreement with the viewpoint.
(11) Indicate the level of agreement you think the majority of subreddit members have for the viewpoint.
(12) Disregarding your own stance on the topic, should members of the subreddit be able to share this viewpoint?
(13) Imagine that you are browsing posts, and you see a post related to the selected topic. You notice that in the comments the following viewpoint has not been raised yet. Rate the likelihood you would share this viewpoint on the subreddit under your main account.
(14) Rate the likelihood you would upvote someone else’s comment expressing this viewpoint, if the comment was present.

A.6

Descriptive Statistics

Finally, we provide the descriptive statistics of the variables we use in our linear mixed-effect models in Table 8 and the disaggregated content removal rates by subreddit (Fig. 8).

B

Robustness Checks

In this section, we conduct robustness checks on our analyses.
First, we present comparisons of our dataset post-filtering with the unfiltered dataset. Second, we show results using an alternative dichotomization of Incongruency. Then, we report using results with ModRatio for moderation. In our pre-registration, we planned on using both ModRatio and Content Removal Rate as measures of community moderation, but ultimately removed ModRatio as the two were correlated. Then, we provide results for our analysis,

Comparing filtered dataset

We compare the filtered dataset that we use in our analyses with the full, unfiltered dataset to examine if there are any participants or subreddits that are systematically excluded. As shown in Table 9, we conducted statistical analyses comparing the distributions of key variables between the full dataset and the filtered sample. Across all tests, we fail to reject the null hypothesis that filtered responses come from the same distribution as unfiltered responses. While there are no statistically significant differences between the filtered and unfiltered dataset, we do note the two following differences.
First, participants in the filtered dataset have longer account tenures (6.16 years in the filtered dataset compared to 5.78 for the unfiltered case). Second, subreddits in the filtered dataset have slightly higher content remove rates with a mean of 30.6% compared to 29.8%, although this difference is not statistically significant (𝑡 (993.9) = −1.65, 𝑝 = 0.10).

B.2

Using an alternative operationalization of incongruency

We demonstrate that our findings are robust to different definitions of our measurement for opinion incongruency. In Sec. 4.4, we define Incongruency as 1 if participants agree with the viewpoint and the majority of subreddit members are neutral or disagree; Incongruency is 0 if the participant agrees and the majority of the subreddit also agrees. In, our alternative dichotomization, Incongruency Alt , we compute the absolute difference between how much the participant agrees with a viewpoint, Agreement, and how much they believe the majority of subreddit members agree with the viewpoint. The continuous score ranges from 0 to 7. Absolute differences greater than 4 are coded as 1 (i.e., incongruent) and less than or equal to 4 are coded as 0. The correlation (Pearson’s 𝑟 ) between Incongruency and Incongruency Alt is 0.288. Using Incongruency results in a slightly better model fit than Incongruency Alt , as indicated by its lower AIC score (1462.7 vs. 1489.2).
As shown in Table 10, Incongruency Alt has a negative relationship with the likelihood of sharing (𝛽 = −1.21, 𝑝 = 0.002). Furthermore, we observe qualitatively similar results with the interaction effects between Incongruency Alt and Diversity (𝛽 = 0.78, 𝑝 = 0.044) as well as Incongruency Alt and Content Removal Rate (𝛽 = −0.97, 𝑝 = 0.059).

B.3

Replacing measures for moderation

We examine changes to the model when replacing Content Removal Rate with ModRatio as our measure of moderation. ModRatio is the ratio of the number of subscribers to number of moderators in a subreddit. In this case, a higher ModRatio would mean less active moderation as there are fewer moderators per subscribers. We apply a logarithmic transformation with base 2 after adding a start-value of 1 to ModRatio. As shown in Table 11, there is a significant negative interaction between Incongruency and ModRatio (𝛽 = −0.35, 𝑝 = 0.02), mirroring our results when using Content Removal Rates. The trends with our independent variables are similar to when we used Content Removal Rate.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 8: An overview of descriptive statistics for all introduced variables (min, max, mean, median).
Variable

Min

Max

Mean

Median

Dependent Variables

Share Likelihood Upvote Likelihood

1 1

7 7

3.87 5.47

4 6

Controls

WTSC Agreement Intensity Male (0/1)
White (0/1)
Democrat (0/1)
Account Tenure Posting

1.5 1 0 0 0 0 1

5.5 3 1 1 1 16 4

3.68 2.17 0.73 0.67 0.37 6.16 2.53

3.63 2 1 1 0 6 2

0 1 1 15.99

1 11 11 41.46

0.25 6.39 6.33 30.63

0 7 6 34.47

Incongruency (0/1)
Inclusion Diversity Content Removal Rate (%)

Independent Variables

Variable

Test

Comment Likelihood Upvote Likelihood WTSC Agreement Intensity Gender White (0/1)
Political Affiliation Account Tenure Posting Incongruency Diversity Inclusion Content Removal Rate

Welch’s t-test Welch’s t-test Welch’s t-test Welch’s t-test Chi-squared Chi-squared Chi-squared Welch’s t-test Welch’s t-test Chi-squared Welch’s t-test Welch’s t-test Welch’s t-test

Filtered Mean

Unfiltered Mean

Test Statistic

p-value

3.88 5.47 3.68 6.17 — — — 6.16 2.53 — 5.33 5.39 0.31

3.79 5.40 3.68 6.14 — — — 5.78 2.51 — 5.31 5.37 0.30

t(926.4)=-0.67 t(936.0)=-0.65 t(934.7)=-0.01 t(950.9)=-0.62 𝜒 2 (2) = 0.01 𝜒 2 (1) = 0.00 𝜒 2 (4) = 0.01 t(928.2)=-1.63 t(948.5)=-0.40 𝜒 2 (1) = 0.00 t(957.6)=-0.19 t(948.6)=-0.14 t(993.9)=-1.65

0.51 0.52 0.99 0.54 0.99 0.98 0.99 0.10 0.69 0.98 0.85 0.89 0.10

Table 9: We compare the filtered and unfiltered data, conducting t-test and 𝜒 2 differences for our independent and dependent variables. There are no statistically significant differences between our filtered and unfiltered data.

B.4

Examining quadratic terms for moderation

Treating sharing likelihood as ordinal

Our conclusions about direction are robust across continuous and ordinal specifications; however, statistical significance varies with modeling assumptions. First, we note there is a statistically significant negative relationship between Incongruency (𝛽 = −0.87, 𝑝 < 0.001 in Model 3b) and the likelihood of sharing a viewpoint.
Furthermore, we note a positive interaction effect between Diversity and opinion sharing (𝛽 = 0.26) as well as a negative interaction effect between Content Removal Rate and opinion sharing (𝛽 = −0.28) although the interaction effects are not significant at our level of statistical power (𝑝 = 0.11 for both).

In Table 13, we use an ordinal mixed-effect regression to predict Share Likelihood, which is a Likert-scale with values ranging from 1 to 7. In the main body of the paper, we report results treating Share Likelihood as continuous, as is consistent with other works measuring the spiral of silence [29, 30, 39, 105]. This decision is also consistent with prior work in statistics, demonstrating the appropriateness of such an approach [78, 99].

We pre-register our analyses on OSF. We also report the following deviations from our pre-registration. First, we change our statistical model to include a crossed random effect between subreddit and participant, as we are capturing multiple observations per subreddit per participant. We updated our power analysis with this

We also report results of our linear mixed-effects model after including a quadratic term for Content Removal Rate in Table 12. Again, we see similar trends between Incongruency and Share Likelihood.
There is no significant relationship between the quadratic term and our dependent variable.

B.5

C

Deviations from Pre-registration

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

Table 10: Our results are robust to alternative dichotomizations of opinion incongruency. We present the results of a linear mixed-effect model that predicts a participant’s likelihood of sharing their opinion to a subreddit using Incongruency Alt , which binarizes opinion incongruency based off a continuous score, as an independent variable and results after adding interaction effects.
Incongruency Alt

Incongruency Alt w/ Interactions

Coef.

SE

𝑝

Coef.

SE

𝑝

Fixed Effects (Intercept)

4.63

0.59

0.00∗∗∗

4.64

0.59

0.00∗∗∗

Controls WTSC Agreement Intensity Male (0/1)
White (0/1)
Democrat (0/1)
Account Tenure Posting

−0.28 0.50 −0.72 0.07 −0.71 0.08 0.68

0.25 0.06 0.55 0.53 0.49 0.24 0.24

0.27 0.00∗∗∗ 0.19 0.89 0.16 0.74 0.01∗∗

−0.29 0.50 −0.72 0.07 −0.71 0.09 0.68

0.25 0.06 0.55 0.53 0.50 0.24 0.24

0.25 0.00∗∗∗ 0.20 0.90 0.16 0.71 0.01∗∗

Independent Variables IncongruencyAlt (0/1)
Inclusion Diversity Content Removal Rate

−1.21 0.09 −0.01 −0.08

0.40 0.10 0.10 0.08

0.00∗∗ 0.37 0.93 0.35

−1.45 0.10 −0.04 −0.06

0.70 0.10 0.09 0.08

0.04∗ 0.32 0.64 0.48

−0.53 0.78 −0.97

0.56 0.39 0.51

0.35 0.04∗ 0.06

IncongruencyAlt × Inclusion IncongruencyAlt × Diversity IncongruencyAlt × Content Removal Rate Model Fit Marginal 𝑅 2 Conditional 𝑅 2

0.232 0.798

0.236 0.797

Note: ∗ p<0.05; ∗∗ p<0.01; ∗∗∗ p<0.001

updated model to reach our target sample size of 270 responses.
Second, we split community values into two separate hypotheses, as we found perceived inclusion and diversity were measuring distinct constructs. We also updated the wording on our hypotheses to mention incongruent viewpoints. Third, we changed how we measured Content Removal Rate. In the pre-registration, we had planned to use 500 posts sampled from each subreddit; however, during analysis, we found this method underreported the amount of content removal compared to results from prior work [43]. Thus, we decided to rely on Pushshift data [5], which gave us access to a larger corpus of posts. Fourth, we remove the variable Mod Ratio from our models since it was correlated with Content Removal Rate.
For robustness, we replicate our results using Mod Ratio as our

measure for community moderation, finding similar results. Finally, we conducted additional analyses using Upvote Likelihood as our dependent variable, which are presented as post-hoc analyses.

D

Other Techniques for Avoiding Self-Silencing

In our survey, we also ask participants what other actions they are likely to take for viewpoints they would not share under their main account. Participants (N=68) are most likely to “lurk,” or read discussion on a topic but not share their viewpoint. After lurking, the second most common option is to discuss the viewpoint offline with friends and family (N=34). Prior work [36] found that people are more likely to engage in conversation on controversial political topics in face-to-face settings rather than on social media platforms.

Mapping the Spiral of Silence: Surveying Unspoken Opinions in Online Communities

CHI ’26, April 13–17, 2026, Barcelona, Spain

Table 11: Including ModRatio does not substantially alter relationships between our independent variables and the likelihood of sharing a viewpoint. The results of a linear mixed-effect model that predicts a participant’s likelihood of sharing their opinion to a subreddit. We present results including ModRatio as an independent variable and results after adding interaction effects.
ModRatio

ModRatio w/ Interactions

Coef.

SE

𝑝

Coef.

SE

𝑝

Fixed Effects (Intercept)

4.75

0.59

0.00∗∗∗

4.81

0.58

0.00∗∗∗

Controls WTSC Agreement Intensity Male (0/1)
White (0/1)
Democrat (0/1)
Account Tenure Posting

−0.30 0.40 −0.71 0.11 −0.75 0.10 0.64

0.25 0.06 0.55 0.53 0.49 0.24 0.24

0.22 0.00∗∗∗ 0.20 0.83 0.13 0.67 0.01∗∗

−0.30 0.41 −0.83 0.19 −0.75 0.12 0.65

0.25 0.06 0.54 0.53 0.49 0.24 0.24

0.22 0.00∗∗∗ 0.13 0.72 0.13 0.63 0.01∗∗

Independent Variables Incongruency (0/1)
Inclusion Diversity ModRatio

−0.74 0.07 0.02 −0.15

0.14 0.10 0.10 0.10

0.00∗∗∗ 0.49 0.84 0.14

−0.57 0.03 −0.06 −0.07

0.15 0.11 0.10 0.10

0.00∗∗∗ 0.81 0.56 0.53

0.21 0.38 −0.35

0.14 0.15 0.16

0.14 0.01∗ 0.02∗

Incongruency × Inclusion Incongruency × Diversity Incongruency × ModRatio Model Fit Marginal 𝑅 2 Conditional 𝑅 2

0.25 0.81

0.26 0.81

Note: ∗ p<0.05; ∗∗ p<0.01; ∗∗∗ p<0.001

Table 12: The results of a linear mixed-effect model predicting a participant’s likelihood of sharing their opinion in a subreddit. We present results including a quadratic term for Content Removal Rate as an independent variable.
Coef.

SE

𝑝

Fixed Effects (Intercept)

4.71

0.60

0.00∗∗∗

Controls WTSC Agreement Intensity Male (0/1)
White (0/1)
Democrat (0/1)
Account Tenure Posting

−0.30 0.40 −0.69 0.10 −0.74 0.08 0.71

0.25 0.06 0.55 0.53 0.49 0.24 0.24

0.23 0.00∗∗∗ 0.22 0.86 0.14 0.75 0.01∗∗

Independent Variables Incongruency (0/1)
Content Removal Rate Content Removal Rate2 Incongruency ×Content Removal Rate Incongruency ×Content Removal Rate2

−1.09 0.06 0.04 −0.02 0.32

0.22 0.12 0.11 0.19 0.18

0.00∗∗∗ 0.62 0.70 0.90 0.07

Model Fit Marginal 𝑅 2 Conditional 𝑅 2

0.25 0.81

Note: ∗ p<0.05; ∗∗ p<0.01; ∗∗∗ p<0.001

CHI ’26, April 13–17, 2026, Barcelona, Spain

Zhao et al.

Table 13: We present results using an ordinal mixed-effect model. The direction of the relationships are robust to the model reported in Sec. 5.
Model 3a

Dependent Variable Share Likelihood ≤ 1 Share Likelihood ≤ 2 Share Likelihood ≤ 3 Share Likelihood ≤ 4 Share Likelihood ≤ 5 Share Likelihood ≤ 6 Controls WTSC Agreement Intensity Male (0/1)
White (0/1)
Democrat (0/1)
Account Tenure Posting Independent Variables Incongruency (0/1)
Inclusion Diversity Content Removal Rate Incongruency × Inclusion Incongruency × Diversity Incongruency × Content Removal Rate Note: ∗ p<0.05; ∗∗ p<0.01; ∗∗∗ p<0.001

Model 3b

Model 3c

Coef.

SE

𝑝

Coef.

SE

𝑝

Coef.

SE

−2.45 −1.20 −0.72 −0.17 0.83 1.97

0.68 0.68 0.68 0.68 0.68 0.68

0.00∗∗∗ 0.08 0.29 0.81 0.22 0.00∗∗

−2.84 −1.53 −1.02 −0.46 0.62 1.83

0.71 0.70 0.70 0.69 0.69 0.70

0.00∗∗∗ 0.03∗ 0.14 0.51 0.37 0.00∗∗∗

−2.96 −1.61 −1.10 −0.47 0.59 1.80

0.71 0.70 0.70 −0.52 0.70 0.70

0.00∗∗∗ 0.02∗ 0.00∗∗∗ 0.46 0.40 0.01∗∗

−0.42 0.60 −0.58 0.10 −0.77 0.06 0.80

0.29 0.08 0.63 0.61 0.57 0.27 0.28

0.15 0.00∗∗∗ 0.35 0.87 0.18 0.81 0.00∗∗

−0.46 0.53 −0.66 0.11 −0.87 0.05 0.77

0.29 0.08 0.64 0.62 0.58 0.28 0.29

0.12 0.00∗∗∗ 0.30 0.86 0.13 0.84 0.01∗∗

−0.47 0.57 −0.70 0.11 −0.89 0.07 0.80

0.29 0.08 0.64 0.62 0.58 0.28 0.29

0.11 0.00∗∗∗ 0.27 0.86 0.13 0.81 0.01∗∗

−0.87 0.10 0.02 −0.07

0.17 0.12 0.11 0.10

0.00∗∗∗ 0.41 0.88 0.50

−0.80 −0.02 −0.04 0.00

0.17 0.13 0.12 0.10

0.00∗∗∗ 0.88 0.76 0.99

0.49 0.26 −0.28

0.19 0.17 0.17

0.01∗ 0.11 0.11

𝑝


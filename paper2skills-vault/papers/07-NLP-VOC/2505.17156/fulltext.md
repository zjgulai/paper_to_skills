<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2505.17156
     paper_id : 2505.17156
     source   : paper2skills-vault/papers/07-NLP-VOC/2505.17156/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

P ERSONA BOT: B RINGING C USTOMER P ERSONAS TO L IFE WITH LLM S AND RAG ∗

arXiv:2505.17156v1 [cs.CL] 22 May 2025

Muhammed Rizwan1, 2 , Lars Carlsson1 , Mohammad Loni2, † 1

2

Department of Computer Science, Jönköping University, Jönköping, Sweden Department of Future Solutions, Volvo Construction Equipment, Eskilstuna, Sweden † Corresponding author: mohammad.loni@volvo.com

A BSTRACT The introduction of Large Language Models (LLMs) has significantly transformed Natural Language Processing (NLP) applications by enabling more advanced analysis of customer personas. At Volvo Construction Equipment (VCE), customer personas have traditionally been developed through qualitative methods, which are time-consuming and lack scalability. The main objective of this paper is to generate synthetic customer personas and integrate them into a Retrieval-Augmented Generation (RAG) chatbot to support decision-making in business processes. To this end, we first focus on developing a persona-based RAG chatbot integrated with verified personas. Next, synthetic personas are generated using Few-Shot and Chain-of-Thought (CoT) prompting techniques and evaluated based on completeness, relevance, and consistency using McNemar’s test. In the final step, the chatbot’s knowledge base is augmented with synthetic personas and additional segment information to assess improvements in response accuracy and practical utility. Key findings indicate that Few-Shot prompting outperformed CoT in generating more complete personas, while CoT demonstrated greater efficiency in terms of response time and token usage. After augmenting the knowledge base, the average accuracy rating of the chatbot increased from 5.88 to 6.42 on a 10-point scale, and 81.82% of participants found the updated system useful in business contexts.
Keywords Customer Persona · Market Analysis · LLM · RAG

1

Introduction

The advent of large language models (LLMs) has significantly advanced the field of natural language processing (NLP).
These models are capable of capturing complex linguistic patterns and are increasingly employed in diverse applications, including virtual assistance, text generation [25], and information extraction [21]. Their adoption across industries has enabled new levels of automation and insight, particularly in customer-facing domains such as marketing, customer support, and strategic planning [22].
Customer personas—detailed representations of user segments—play a pivotal role in enabling businesses to tailor their offerings and communication strategies. Traditionally, personas are crafted through qualitative methods such as interviews and surveys, which, while insightful, are time-intensive and difficult to scale [12]. More recently, datadriven approaches have emerged, leveraging statistical and machine learning techniques to streamline persona creation [13, 14, 15]. However, these methods often struggle to extract nuanced insights from unstructured text and adapt to evolving customer behavior in real time.
LLMs such as GPT-4 offer a new avenue for generating high-quality, structured customer personas directly from unstructured textual data, such as customer success stories [19, 17, 12]. Nevertheless, current research has primarily focused on using a single prompting method, without comparing the effectiveness of alternative strategies like few-shot or chain-of-thought (CoT) prompting [29, 28]. Moreover, the practical integration of generated personas into business ∗

This study was carried out as part of a Master’s thesis project at Volvo Construction Equipment.

PersonaBOT

workflows—especially through interactive systems like retrieval-augmented generation (RAG) chatbots—remains underexplored [11].
This paper addresses these gaps by: (1) evaluating the effectiveness of different prompting techniques for generating synthetic customer personas from publicly available texts; and (2) presenting a proof-of-concept chatbot system that enables users to interact with these personas through natural language queries. The study focuses on a use case from the construction industry, where customer segmentation is critical but existing persona development practices are resource-intensive and static.
By exploring both the generation and application of synthetic personas, this work contributes to the growing body of research at the intersection of LLMs, human-centered design, and business decision support. It provides practical insights into how organizations can adopt LLM-based tools for scalable, data-driven persona generation and utilization.

2

Related Work

2.1

Non-LLM Approaches for Creating Customer Personas

The traditional method of creating personas depended on qualitative data, such as interviews, observations, and survey data from target users [12]. There have been researches that explored data-driven approaches that improved efficiency, scalability, and reliability in creating personas. One such approach was introduced by McGinn et al. [13], where a survey was sent over to 1300 users. An exploratory factor analysis, a data reduction technique, was performed on the survey results. This analysis helped identify the groups based on the tasks performed. Stakeholders were involved throughout this process to ensure the relevance of personas. Instead of relying on survey data or user interviews, Zhang et al.[14] followed a two-step statistical machine-learning approach to create personas only based on user behavior.
In the first step, they analyzed 3.5 million clicks from 2400 users and clustered them into a common workflow using hierarchical clustering. In the second step, a mixed statistical model was used to create five personas.
Jung et al.[15] introduced Automatic Persona Generation (APG), a system that creates personas from real-time social media interactions on platforms like Facebook and YouTube. They processed tens of millions of interactions using non-negative matrix factorization. This automatically generated realistic and up-to-date personas from large-scale social media data. Similarly, Farseev et al.[16] introduced a framework named SOMONITOR that used X-Mean clustering with ADA embeddings to extract customer personas from digital marketing content. Unlike previous studies that relied on survey data or behavioral data, SOMONITOR clusters advertising content into distinct persona groups based on customer needs, interests, and aspirations. While these data-driven methods significantly improve persona creation, advancements in LLM offer further opportunities for automating and improving persona development.
2.2

LLM Approaches of Creating Personas

LLMs’ ability to generate structured text based on the input text provided using advanced natural language processing capabilities makes them a strong candidate for persona creation. This section reviews various approaches that utilized LLMs for persona creation.
One of the methods used to create personas is by providing LLM with structured prompts. This methodology was used in [17] to create 450 personas. In this study, they utilized these generated personas and investigated the bias and diversity in them. Their findings indicated that LLMs can create informative and relatable personas, but they exhibit a strong bias from specific countries. Similarly, Zhang et al. [18] introduced PersonaGen, a tool that used Generative Pre-trained Transformer (GPT)-4 [30] along with knowledge graphs to refine persona generation. The tool was developed to assist the agile software development process. The GPT-4 model analyzed the user feedback provided and generated high-quality and detailed persona content. This content was then used by the knowledge graphs to create personas. PersonaGen demonstrated that it improved accuracy in capturing user needs compared to independent human analysis. Although challenges remain in analyzing non-functional requirements.
Another method for persona generation using LLMs involves the use of thematic analysis. De Paoli et al. [19] proposed a workflow where LLMs analyze qualitative interview data to generate personas. This approach follows a structured methodology where LLMs first generate codes (such as behaviors, goals, etc.) in textual format. From these codes, emerging themes are identified. These themes, along with prompts, are then used to construct persona narratives. The advantage of this method lies in its ability to extract meaningful user traits from raw interview data without predefined coding schemes. An extension of this approach is found in Persona-L [12], a system that integrates LLMs with a RAG framework. By using specific types of datasets, this system enhances persona realism while addressing biases commonly found in LLM-generated content. The system was tested in creating personas that represent individuals with 2

PersonaBOT

complex needs. This study demonstrated that incorporating external data can improve both the diversity and contextual accuracy of the generated personas.
Beyond structured prompting and thematic analysis, there has been a study that used human-AI collaboration in persona generation. Goel et al. [20] conducted an exploratory study where novice designers used GPT-3 [28] to create personas through iterative refinement. The study found that personas generated with GPT-3 were comparable to those created manually, particularly when designers provided detailed prompts and engaged in multiple iterations. However, the study also highlighted challenges such as generic responses, inconsistencies, and stereotypical outputs. This makes it necessary for human intervention to refine and personalize the generated personas.
2.3

Non-LLM Approaches for Analyzing and Leveraging Customer Personas

There have been various techniques to analyze and utilize customer persona before the emergence of LLMs. These methods were based on statistical methods [2] and machine learning [1, 3] to extract insights from customer data. This section reviews these approaches, limitations, and the reasons for the shift towards using LLM-based methods.
One such approach is the use of Quantum artificial intelligence (QAI). QAI combines quantum computing and AI to process large datasets in parallel. This allows obtaining real-time updates of customer profiles in response to dynamic behaviors and preferences. A study by More et al. [1] discussed how QAI can improve sentiment analysis and predictive modeling using quantum machine learning. This methodology improves customer segmentation, recommendation engines, and consumer behavior prediction. Even though QAI is promising, it remains in the early stages of adoption, and its implementation is challenging due to limited computational feasibility and hardware availability.
Generative AI techniques, such as Generative Adversarial Nets (GANs) [31] and Variational Autoencoders (VAEs)
[32], have been explored for improving marketing applications using personas. [2] demonstrated that GANs and VAEs can improve customer profiling in social media marketing by generating personalized product recommendations and marketing content. The study found that generated content was able to significantly improve customer engagement, loyalty, and sales. However, generative models cause risks related to algorithmic bias, ethical concerns about privacy, and the chances of misleading or made-up customer insights [2].
Another approach includes utilizing Bayesian probabilistic models such as Latent Dirichlet Allocation(LDA) [33] and Structural Topic Models (STM) [34]. These models categorize textual data into topics based on word co-occurrence patterns, helping in customer segmentation and persona identification. However, Bayesian models are frequency-based models that rely on word frequency distribution. They struggle to capture the complicated and nuanced elements in the textual data. This is due to the lack of an attention mechanism, a key feature of modern transformer-based LLMs [3].
The challenges discussed above in [1, 2, 3] have led researchers and businesses to adopt LLMs, as these models show promise in contextual understanding and adaptability.
2.4

LLM Approaches for Analyzing and Leveraging Customer Personas

This section reviews studies that utilized LLMs for personas, such as (i) persona interpretation, (ii) personalization, (iii)
role-playing techniques, (iv) investigating bias and stereotypes, and (v) business insights.
(i) Persona Interpretation: LLMs are built upon large and diverse datasets, enabling them to interpret and generate user personas with high precision. Unlike traditional methods that need structured datasets and predefined heuristics, LLMs can extract persona-related attributes from conversational data, social media posts, and customer feedback. This information can then be used to understand the needs, motivations, and goals of the specific user. [4] examines how LLM interprets culturally specific personas, focusing on the Indian context. The research conducted both quantitative and qualitative analyses to assess how well LLMs understood personas within cultural contexts. The study revealed that LLMs exhibit high consistency and completeness in persona evaluation, but they struggle with credibility.
(ii) Personalization: One of the notable advancements in LLM-driven persona development is personalization. Zhang et al. [5] provides a detailed survey of how LLMs can be personalized. They propose a taxonomy of personalization levels in three categories: user-level, persona-level, and global preferences. This study highlights how techniques like RAG and prompt engineering can be used to tailor responses to user-specific needs.
There have been studies that have explored how LLMs can be customized for personalized interaction. One such application is CloChat [7], which allows users to tailor personas for various contexts and tasks. The end user of this application can choose to define personas by altering attributes like conversational style, emotions, areas of interest, and visual representations, thereby making interactions more human-like and relevant. To assess CloChat’s effectiveness, researchers conducted surveys and in-depth interviews by comparing it with ChatGPT. The findings indicated that CloChat significantly improves user engagement, trust, and emotional connection when compared with ChatGPT.
3

PersonaBOT

(iii) LLM Role-playing: Personas can be integrated with LLMs through two approaches: LLM Role-Playing and LLM Personalization. In LLM role-playing, LLMs are assigned personas (roles) and they adapt to specific environments and tasks. Whereas in LLM personalization, LLM is adapted to user-specific personas for customized responses. The techniques used in Role-Playing are prompt engineering, multi-agent frameworks, and emergent behaviors in specific domains. In the personalization techniques, user data is integrated by Reinforcement Learning from Human Feedback (RLHF), fine-tuning, and memory mechanisms. This study highlights several challenges associated with role-playing personas, including limited contextual understanding, the need for manual persona creation, and the static nature of personas, which prevents them from adapting to dynamic tasks [6]. To address these challenges, [39] proposes a pattern language for persona-based interactions. This pattern language contains a series of patterns, where each pattern identifies a specific problem and provides its associated solution in the form of a template. In this study, seven person-related patterns are introduced, which improved realism, adaptability, and specificity in LLM interactions, making them more effective for complex and evolving tasks.
(iv) Investigate bias and stereotypes: While persona-based LLMs improve customer engagement, they also can introduce biases and stereotypes which may affect customer insights and segmentations. Cheng et al. [8] introduced Marked Persona, a prompt-based framework that captures the patterns and stereotypes across the LLM outputs. Their study used GPT-3.5 and GPT-4 to generate personas across various demographic groups and analyzed how this output is different from human-written personas. The findings reveal that personas generated by LLM contain more stereotypes than the personas written by humans. These biases are a challenge for LLM driven customer analysis, as they can provide inaccurate customer information.
(v) Business insights: Understanding customer preferences and requirements has become important for businesses.Extracting and analyzing customer data manually is often a difficult and time-consuming task. Barandoni et al. [9] evaluate the ability of proprietary and open-source models, such as GPT-4, Gemini, and Mistral 7B, to extract customer needs from TripAdvisor forum posts. This study systematically compared two prompting techniques, such as CoT using various proprietary and open-source LLMs for customer needs extraction. However, the focus was on extracting short customer needs from forum posts, not on generating structured personas. Additionally, the study did not explore fine-tuning techniques, which could have further improved the model’s performance. In contrast, [3] used fine-tuning on different models to identify topics, emotions, and sentiments from TripAdvisor customer reviews. This study provides an alternative technique for improving LLM-driven customer insights extraction.
While persona-based LLMs help businesses, their ability to understand persona and generate meaningful insights needs to be researched. Jiang et al. [10], through a case study, investigate whether LLMs can generate content similar to assigned personas by simulating different personalities using the Big Five personality model. Their findings demonstrate that LLMs can adjust their output to match the behavior of assigned personas.

2.5

Positioning of this Research in the Context of Related Work

The reviewed literature from Sections 2.1 to 2.4 highlights major advancements in persona creation and usage. This includes both traditional methods that do not use LLMs and newer methods that use LLMs. However, some limitations make it difficult to apply these methods in real-world businesses, such as the construction equipment manufacturing industry.
While studies such as [9] have compared prompting methods such as few-shot and CoT reasoning for tasks like customer needs extraction, they did not focus on structured persona generation. Moreover, studies on persona generation such as [19, 17, 20] typically relied on a single prompting method without comparing multiple approaches. This brings a gap in understanding which prompting method works best when generating personas from qualitative data, such as customer success stories. Existing studies on using personas mainly focus on general use cases, such as how LLMs understand personas [4], identifying stereotypes in LLM responses [8], and the use of personas to improve personalization [7].
However, there is limited research on how customer personas can support businesses where the customer attributes, such as challenges and needs, vary widely. Finally, the reviewed studies do not explore the integration of customer personas with retrieval systems like RAG. The authors of [3] mention in their limitations and future directions that techniques like RAG could be beneficial for retrieving consumer data. However, the use of RAG for persona-based analysis and insight generation remains unexplored, especially in helping R&D engineers and stakeholders to interact efficiently with persona data.
This research addresses the identified limitations in existing studies by systematically comparing and evaluating different prompting methods using specific metrics. The evaluation will help determine the most effective prompting technique for persona generation. This will be further discussed in the methodology section. Additionally, integrating a RAG-based system with personas allows R&D engineers and stakeholders to interact with customer data more easily.
4

PersonaBOT

This ensures that the findings of this research are not only theoretically grounded but also practically applicable in real-world business scenarios.

3

Method And Implementation

3.1

Research Method

This research adopts the Design Science Research Methodology (DSRM) as the primary research framework [36].
DSRM is especially suitable for studies in computer science where the focus is on the development and evaluation of innovative artifacts to solve real-world problems. As described in [26], the DSRM framework includes five main activities:
1. Problem Explication 2. Requirements Definition 3. Design and Development 4. Demonstration 5. Evaluation To systematically address the research objectives, these activities were executed in three phases. Each iteration was built upon findings and evaluations from previous cycles, which improved the developed artifact. The artifact in this study is a conversational system designed to help stakeholders at VCE to query on customer persona. Figure 1 depicts the research method followed in the study.
Iteration 1: Initial Chatbot Development and Testing • Problem Explication: The initial problem identified was through a comprehensive literature review and discussions with stakeholders at VCE. The literature review explored existing research on customer personas, LLMs, RAG and prompting techniques. It helped providing a clear research gap regarding the integration of personas into RAG system and comparison of prompting method. The discussions with stakeholders highlighted the practical need for utilization of personas that can support various stakeholders to help make decisions faster.
• Requirements Definition: Based on insights gained from the review of the literature and discussion with stakeholders, the key requirements were identified. This also included collecting and preparing relevant data and defining the chatbot’s core functionalities.
• Design and Development: The first artifact developed was a RAG-based chatbot that was integrated with verified customer personas provided by the VCE’s Customer Experience team. Section 3.5 discusses the process involved in building the RAG system.
• Demonstration: The chatbot was deployed and demonstrated to a selected group of end-users. This enabled them to test and explore capabilities of the artifact.
• Evaluation: An evaluation form was sent out to the end users who had the chance to explore the chatbot. The process involved in the evaluation is discussed in Section 3.6 and the results of this section are discussed in Section 4.1. This feedback was then used as input for the second iteration.
Iteration 2: Persona Generation and Comparison • Problem Explication: Based on user feedback from Iteration 1, the problem identified was that the chatbot’s input data. The suggestion was to explore more data from customer success stories and more segment-specific information. Additional feedback included improved response accuracy and better handling of complex queries.
• Requirements Definition: As per the problem identified in the prior iteration, the requirement included gathering of additional personas, segment information and improvement of performance. In the literature study, there was also a gap in comparing the prompting technique for persona generation. This brings a requirement on studying how can different prompting techniques be used and evaluated.
• Design and Development: Personas were generated from customer success stories using two different prompting techniques. The development process involved in persona generation is elaborated in section 3.4.
• Demonstration: Personas created by both prompting techniques were clearly presented to evaluators alongside the original customer success stories. This helped them with a clear basis for comparison.
5

PersonaBOT

• Evaluation: A structured evaluation was conducted to statistically compare both prompting methods. The steps followed in the evaluation of personas are discussed in 3.7. The statistical analysis tested which of the prompting generates better personas in terms of metrics (accuracy, relevance, and consistency). Based on these results, the generated synthetic persona was used to augment the knowledge base of the chatbot in the third iteration.
Iteration 3: Improvement of the conversational system and the final evaluation • Problem Explication: Following the second iteration, the best-performing method of generating persona was determined. The other feedback from the first iteration included the addition of additional segment information.
• Requirements Definition: The requirement in this iteration was to improve the chatbot by updating the knowledge base. This was done by adding additional synthetic customer personas and additional segmentspecific information.
• Design and Development: The chatbot’s knowledge base was expanded and refined by incorporating new data based on the initial feedback from iteration 1 and results from iteration 2.
• Demonstration: The improved chatbot was redeployed and demonstrated to various end-users for testing and exploration.
• Evaluation: A second evaluation round was conducted using a similar evaluation form from the iteration 1.
This form assessed chatbot accuracy, usability, user satisfaction, and practical applicability. The purpose was to validate improvements from previous iterations and confirm the artifact effectively addressed the originally identified problem. The process involved in the evaluation is discussed in Section 3.8 and the results of this section are discussed in Section 4.3.
3.2

Overview of Data

Three types of data were used in this study:
1. Customer Success Stories: These are real-world narratives that illustrate how customers achieved positive outcomes using VCE’s products or services. This data is publicly available on the VCE website 2 . This contains text, images, and videos. The stories are categorized by product, application, or industry segment (e.g., agriculture, demolition, quarrying, and aggregates). In this study, only stories related to the quarrying and mining segments were used.
2. Verified Personas: These personas were developed by the Customer Experience team at VCE through direct interviews with customers from different regions. This data is internal and confidential. It is accessible only to VCE employees. Table 1 contains the information included in each persona.
3. General information about the Quarry, Mining, and Aggregates segments: This consists of textual data providing definitions, processes, and background information about the three industry segments. It is used as contextual knowledge to support understanding of the domain.
Table 1: Persona Attributes Attribute Narrated Video Name Role Number of Employees Fleet Size Short Story What is Important Challenges Expectations Buying Considerations 2

Description A video summarizing the persona’s story The customer’s name The job title or position Total employees in the customer’s organization Size of the equipment fleet A brief background or narrative Key priorities or values of the customer Main issues faced by the customer What the customer expects from VCE Factors that influence the customer’s decisions

https://www.volvoce.com/united-states/en-us/resources/customer-success-stories/

6

PersonaBOT

Figure 1: Research Method

3.3

Data Preparation

This section discusses the process involved in the data preparation. Each subsection describes the process involved for that specific data type.
3.3.1

Customer Success Story

The first type of data used was the Customer Success Stories, which served as input for generating synthetic customer personas. To extract these stories, web scraping was employed using the Python library Beautiful Soup3 . Manual 3

https://pypi.org/project/beautifulsoup4/

7

PersonaBOT

extraction of information from each webpage would have been time-consuming and error-prone. Hence, automated scraping was chosen to extract information.
The process began by inspecting the HTML structure to identify the relevant tags that contained the main story content.
Only the textual narrative was extracted. Non-relevant elements such as headings, image captions, videos, and figures were excluded. A CSV file containing the URLs of selected (mining and quarrying segment) success stories was used as input for web scraping. From each URL, the script fetched the page content and extracted all paragraph (<p>)
elements within a specific section of the webpage ( div class "newsArticle-2023"). The extracted text was then cleaned by removing extra spacing and manually adding the missing content that was not extracted during the scraping process.

3.3.2

Verified Personas

The second source of data is verified Personas provided by VCE. These personas are stored in an internal platform that is only accessible to VCE employees. Due to this restriction, web scraping was not a feasible option for this dataset.
Thus, the persona data was manually copied from the internal website. All the textual content from each of the personas was extracted, excluding video material. Once the textual data was collected, it was converted into structured JSON files using a Python script. The data was then converted to JSON as it provides a structured, machine-readable format that enables integration with retrieval systems. Each JSON file represented a single persona and included key-value pairs corresponding to the persona attributes such as name, role, challenges, and expectations.

3.3.3

General Information About the Quarry, Mining, and Aggregates Segments

The third type of data used in this study was general textual information related to quarrying, mining, and aggregates.
This content was used to provide contextual background for retrieval tasks. This will help the RAG system better understand industry-specific terminology and operations.
This data was completely textual content. To enhance readability and support LLM’s, the whole text was manually split into small meaningful chunks based on meaningful topics. Each chunk was then converted into Markdown format to introduce structure and hierarchy within the documents. Headings, subheadings, and bullet points were added to clearly distinguish between concepts, definitions, and processes. Converting into markdown will help the system better recognize the relationships between different pieces of information.

3.4

Generation Of Synthetic Customer Personas

This section details the process of generating customer personas using customer success stories as input. In this study, personas were created using two different prompting techniques: few-shot prompting and CoT prompting. GPT-4o Mini was selected as the language model for persona generation. The model received both the prompt and the success story as input. The prompt designs were developed based on OpenAI’s prompt engineering guidelines [23]. Multiple iterations of prompts were refined and tested to improve the output quality. The refinement process involved experimenting with wording and adjusting the level of detail provided in the prompts.
In the few-shot prompting technique, the model was provided with three verified personas as examples. The complete prompt included system instructions, a task definition, an output structure format, and three example personas. The benefit of this method is that by using examples, the model will be able to recognize patterns and relationships between persona attributes. This helps the model generate structured and coherent outputs. Figure 2 is depiction of an example persona generated by few-shot prompting technique.
The CoT prompting technique followed a different approach by guiding the model through an internal step-by-step reasoning process instead of directly generating persona attributes. The prompt included system instructions, a structured output format, and a reasoning process to improve information extraction. The model was instructed to first identify key details from the success story, then analyze the customer’s background and business context, extract challenges, expectations, buying considerations, and finally generate the structured persona. In this method, the model is encouraged to perform logical reasoning before output generation. Figure 3 is an illustration of a persona generated using this method.
For each of the personas generated by using the two prompting methods, the total time taken to generate personas in seconds and the total tokens consumed were also computed. The overall process involved in persona generation is illustrated in Figure 4 8

PersonaBOT

Figure 2: Example Synthetic Persona - Few Shot

Figure 3: Example Synthetic Persona - Cot

3.5

Building the RAG system

The role of the RAG system is to act as a conversational agent that allows users to query information based on customer persona data and general information about different segments. The system consists of two main components: (1)
Retrieval Component is responsible for storing, indexing, and retrieving relevant documents; and (2) Generation Component that is responsible for generating responses based on the retrieved content using LLM. The subsection below is a brief explanation of the design and implementation details of each component.
9

PersonaBOT

Figure 4: Persona Generation Process 3.5.1

Retrieval Component

The retrieval component was built using Azure AI Search4 that acts as a dedicated search engine and storage of data.
The implementation process involved in building this component included the following steps:
1. Creating the Search Index: The first step in constructing a retrieval system involves creating a search index. The index schema was designed to accommodate both structured persona data (*.JSON format) and unstructured general information (*.txt format). The schema contained the below fields:
• id- A unique identifier for each document.
• title- The name of the document.
• category- The type of document (e.g. "persona" or "general information" )
• content- The complete data in textual format that is to be searched and retrieved.
• content_vector- A high dimensional vector representation of the document for similarity-based retrieval.
While creating the search index, three search techniques were configured for efficient content retrievals:
• Keyword Search: This type of search performs lexical matching based on the exact words in the query.
It allows filtering and ranking documents using traditional search techniques.
• Semantic Search: This approach improves ranking by understanding the meaning of the query and prioritizing documents based on contextual relevance rather than just exact word matches. The content field was set as the primary ranking factor to ensure meaningful results.
• Vector Search: This type of search was implemented using the Hierarchical Navigable Small World (HNSW) algorithm for Approximate Nearest Neighbor (ANN) retrieval. It enables searching for semantically similar documents using vector embeddings, even when the query and the document do not share exact words.
2. Uploading Documents to the Index: After the index was created, the next step involved uploading the documents to the index that was created. This process began by loading the data and extracting textual content from the data. The textual content was then converted into embeddings using an embedding model named text-embedding-ada-002. These embeddings, along with the raw text, were then batch-uploaded into the index.
3.5.2

Generation Component

The Generation Component is responsible for utilizing the content retrieved by the retrieval component to generate output for the user. This component was developed using GPT-4o Mini in combination with a hybrid search approach.
Below are the implementation details of the processes involved in this component.
1. Integrating search index with hybrid search strategy: To improve the quality of responses, a hybrid search approach was employed. Hybrid search combines the capabilities of both keyword-based search and vector-based search techniques. According to experiments by Microsoft [24], hybrid search outperforms standalone keyword or vector-based search methods in retrieving relevant documents for question-answering systems. Due to this reason, the hybrid search was opted for this study.
4

https://azure.microsoft.com/en-us/products/ai-services/ai-search

10

PersonaBOT

When a user submits a query, the query is first converted into an embedding using the embedding model.
The hybrid search method is then applied to retrieve the top three most relevant documents from the search index. These documents are used as contextual input for the language model to perform the generation of an appropriate response.
2. System Message and Prompt Engineering: To ensure consistency, accuracy, and contextual relevance in the generated responses, a Prompty file 5 was created. This file contains a system message that defines specific role instructions, the tone for responses, and detailed guidelines for answering the questions.
3. Final Response Generation: After retrieving the relevant documents, the GPT-4o Mini model synthesizes the final response by integrating the retrieved documents, system message, and conversation history to ensure coherent and context-relevant output.
Figure 3.5.2 shows the overall process of the chatbot system, highlighting the flow from data indexing and retrieval to response generation using a hybrid search strategy.

3.6

Initial Evaluation of the Conversational System

This section describes the process involved in the initial user evaluation to assess the effectiveness of the persona-based chatbot system. The goal of this evaluation is to understand how the chatbot supports decision-making and contributes to automating customer-facing processes. For this stage of evaluation, the developed chatbot was integrated with verified customer personas and segment-specific information. The system was then deployed using Azure Web App for user interaction 6 . The participants included people from relevant business functions, such as R&D, marketing, and customer representatives. After interacting with the system, they were asked to provide feedback through an evaluation form.
3.6.1

Evaluation Design

To systematically assess the ability of the chatbot to support decision-making and process automation, five questions were opted. The type of questions includes multiple choice, Likert’s scale, and 1-10 rating scale. Table 2 presents the questions in the evaluation form along with their purposes and types of responses.
In addition to these questions, participants were also asked to provide open-ended comments to elaborate on their experience, share specific feedback, or suggest improvements. This combination of quantitative and qualitative feedback designed helped in providing insights into how well the system aligns with user expectations, business needs, and opportunities for automation. Data from these responses was analyzed using descriptive statistical analysis and qualitative thematic analysis. The findings are presented in section 4.1 in the chapter 4.
5 6

https://prompty.ai/ https://azure.microsoft.com/en-us/products/app-service/web

11

PersonaBOT

Table 2: Initial Evaluation Questions, Purpose and Type Question Purpose Response Type How would you rate the chatbot’s ability Evaluate the overall accuracy of the 1-10 scale to provide accurate answers?
chatbot in providing relevant answers Does the chatbot correctly interpret and Measures the chatbot’s capability to han- Likert Scale respond to complex queries (e.g., pro- dle complex queries.
viding details on customer personas)?
Does the chatbot provide clear and con- Assesses clarity in responses.
Likert Scale cise answers?
How well does the chatbot align with Evaluates the chatbot’s relevance and Likert Scale your business needs?
usefulness in business contexts.
Do you believe the chatbot has reduced Understand the impact of chatbot in au- Multiple Choice the workload for human support teams? tomation and improving efficiency.

3.7

Evaluation of Generated Personas

This section presents the methodology used to evaluate the two types of prompting techniques. The aim of this evaluation is to identify the optimal prompting technique to produce personas.
3.7.1

Evaluation Design

A total of 24 customer success stories were used to generate personas. To reduce the time and effort for evaluators, a random subset of five stories was selected for the evaluation. Each evaluator read the full customer success story before reviewing two anonymized personas. The order in which the personas were presented was randomized to minimize the bias. A Microsoft Form 7 was used to collect binary feedback (Yes/No) on each of the evaluation metrics.
3.7.2

Metrics Used

The metrics used in this study were adapted from literature [17, 12, 20] that evaluated personas. Each of the metrics was evaluated using binary response. The choice of binary metrics was to reduce ambiguity and speed up the evaluation process.
Table 3 presents the questionnaire used for evaluating each metric, along with a brief description of what each metric assesses.
Table 3: Metrics and Definition Metric Name Completeness

Relevance

Consistency

3.7.3

Questionnaire Does the persona include all the important details (like role, challenges, expectations etc.) from the customer success story to fully understand the customer?
Does the persona focus only on the relevant and important details from the customer success story?

Description Evaluates whether the persona captures all key customer insights needed for understanding.

Assess whether the persona includes only important details from the source story, avoiding any irrelevant or redundant information.
Does the persona add any incorrect or made- Checks if the persona introduced incorrect, up information that is not in the customer fabricated, or contradictory information.
success story?

Participants

The evaluation was conducted with professionals from VCE who are familiar with customers, products, and services.
The evaluators included Customer Solution Strategists, Research Engineers, and Project Managers.
7

https://forms.office.com/

12

PersonaBOT

3.7.4

Analysis Method

A formal hypothesis testing approach was adopted to determine whether differences between prompting methods were statistically significant. For each prompting method and for each metric, the following hypotheses were defined:
• Null Hypothesis (H0 ) : There is no significant difference between the two prompting methods in terms of the metrics used for evaluation.
• Alternative Hypothesis (H1 ): There is a significant difference between the two prompting methods in terms of the metrics used for evaluation.
The McNemar test [27] was selected because it is specifically designed for paired nominal (categorical) data. It is commonly used when the same subjects are exposed to two conditions, and their binary responses (e.g., Yes/No)
are analyzed for shifts between the two conditions [27]. In this research, it is used to determine if one prompting method significantly outperformed another across the three binary evaluation metrics. This test uses a 2×2 contingency table based on paired binary responses for each persona pair. Only the discordant pairs, where evaluators responded differently for the two methods are used to compute the test statistic. A separate contingency table is constructed for each evaluation metric. Table 4 presents an example contingency table.
Table 4: Example Contingency Table Method: Few-Shot - Yes Method: Few-Shot - No

Method: CoT - Yes a c

Method: CoT - No b d

a = Both methods are rated as ’Yes’ b = Method Few-Shot is rated as ’No’ and Method CoT is rated as ’Yes’ c = Method Few-Shot is rated as ’Yes’ and Method CoT is rated as ’No’ d = Both methods are rated as ’No’ 2

The McNemar test statistic is defined as χ2 = (b−c)
b+c . The test statistic value which we obtain follows a chi-square distribution with one degree of freedom. From this value, a p-value is calculated and used to assess whether the observed difference is statistically significant. A p-value below 0.05 indicates a significant difference between the two prompting methods for the given evaluation metric.
3.8

Evaluation of Augmented Chatbot with Synthetic Personas

This section discusses the methodology used to evaluate the impact of augmenting the chatbot’s knowledge base with synthetic personas and additional segment-specific information. The objective was to assess whether these enhancements improved the chatbot’s performance in terms of accuracy, usability, and decision-making capabilities.
System Updates and Evaluation Design Based on the insights from Section 3.6 (Initial Evaluation) and Section 3.7 (Persona Generation Evaluation), the conversational system was updated to improve its overall performance. "Feedback from the initial evaluation highlighted the need for additional segment data and more personas derived from customer success stories. As a result, the knowledge base was expanded with newly generated synthetic personas and additional segment-specific information.
Based on the findings in Section 3.7, the best-performing prompting technique for persona generation was selected. The personas generated by this method were then added to the Azure AI Search index, replacing the initial dataset. Plus, the system prompt that was used to guide the chatbot’s responses was also revised. This was to improve the accuracy, particularly for complex or context-rich queries.
Once these changes were implemented, the system was tested and redeployed. The updated version of the chatbot was made available for user testing. To ensure consistency and for direct performance comparison, the same evaluation method, participant group, and questionnaire from the initial evaluation were used again. The collected responses were analyzed using the same methods as in the initial evaluation. Descriptive statistical analysis was used to compare quantitative results. This approach enabled direct comparison between the initial chatbot (with verified personas) and the updated version (with synthetic personas). The findings from this evaluation are presented in Chapter 4.3.
13

PersonaBOT

4

Results

This section presents the results of this study, including the evaluation of generated personas and the performance of the persona-based chatbot. The results are organized based on the research questions.
4.1

Results for Research Question 1: Effectiveness of the Persona-Based Chatbot

This subsection presents the findings related to the evaluation of the persona-based chatbot conducted with eight stakeholders from relevant business functions such as R&D, marketing, and customer relations. The evaluation was focused on five aspects, such as accuracy of the answers, the ability to handle complex queries, clarity of the responses, alignment with business needs, and impact on workload reduction.
4.1.1

Quantitative Results

Participants were asked to provide an overall rating for the ability of the chatbot to provide accurate answers on a scale from 1 (Very Poor) to 10 (Excellent). The average rating across all evaluators was 5.88. The range of ratings from all evaluators was 4 to 10, with the majority giving a rating of 5. Figure 5 is a bar chart illustrating the distribution of accuracy ratings.

Figure 5: Distribution of the accuracy rating The ability of the chatbot to interpret and respond to complex queries such as providing details on customer persona was measured using a Likert scale. The majority of the participants indicated that the chatbot responded correctly "most of the time", while three participants selected "sometimes", and one participant reported "never". Figure 6 presents a bar chart depicting these findings.
Regarding the ability of the conversational system to provide clear and concise responses, 3 users reported that the system provides clear and concise results only sometimes. This indicates some inconsistency in the clarity and conciseness of the responses. Figure 7 shows the distribution of the responses in a bar chart.
The impact of the system in business contexts can be seen largely positive. Only one evaluator rated the system as not useful. The remaining participants responded positively. 62.5% users rated it as "somewhat needed", suggesting that while the chatbot addressed business needs to some extent, further alignment and improvements are required. Figure 8 is a bar chart showing the distribution of the various ratings regarding the alignment of business needs.
When evaluating the potential of the system to reduce the workforce, 75% of the participants believe that it will reduce their workload and improve automation. Figure 9 is a pie chart depicting this distribution.
4.1.2

Qualitative Results

In addition to the five evaluation questions, participants were asked to provide open-ended feedback on their experience with the persona-based chatbot. Three main key themes were identified from their responses.
14

PersonaBOT

Figure 6: Ability of the system to provide response to complex query

Figure 7: Ability of the system to provide clear and concise response Primarily, the participants emphasized the importance of improving the quality of the data used. They suggested that integrating customer success stories and incorporating a more diverse range of customer data could significantly enhance the chatbot’s utility. Another observation was that the responses sometimes felt too generalized and lacked specific insights, making them indistinguishable from publicly available information. This highlights that the system needs to provide deeper, more personalized answers based on additional customer datasets. Secondly, participants pointed out the necessity of improving segment-specific information within the chatbot’s knowledge base. Finally, the participants also identified several technical areas for improvement. These included better handling of complex queries, improved accuracy in responses, faster response times, and a more natural, human-like conversational style. Participants also suggested better integration with internal systems and the inclusion of sources for the information provided in responses.
4.1.3

Summary of Findings

The initial round evaluation of the persona-based chatbot demonstrated a moderate overall effectiveness in supporting decision-making and automation within the construction industry. From the quantitative results, the system received an average overall accuracy of 5.88 out of 10. While it generally aligned with business needs for most of the users, there 15

PersonaBOT

Figure 8: Alignment of system in business need

Figure 9: Workload reduction and improvement in automation

remains significant room for improvement. From the feedback on the ability of the system to reduce workload and automation, it can be concluded that the persona chatbot has contributed to reducing human workload. Along with these quantitative findings, the qualitative feedback highlighted the need for broader and more diverse data integration from customer success stories, enhanced segment-specific information, improved handling of complex queries, and more human-like interactions. Overall, these findings provided critical insights for guiding the refinement and improvement of the chatbot in the next iterations.
4.2

Results for Research Question 2: Persona Generation and Prompting Techniques

This section presents the results obtained from comparing synthetic personas generated by two prompting techniques.
16

PersonaBOT

4.2.1

Quantitative Results

In this research, the synthetic persona was generated using the customer success story as input. The process involved in the generation of the persona is described in the section 3.4. The evaluation was carried out by three expert evaluators (n=3), each of whom reviewed a total of 5 personas. Each anonymized persona was assessed using three binary metrics:
completeness, relevance, and consistency. The responses were analyzed using the McNemar’s test to identify if there are statistically significant differences between the two prompting methods. In addition, efficiency metrics such as average generation time (seconds) and token usage were recorded.
Completeness: The completeness metric evaluated whether the persona captured all important details (e.g., role, challenges, expectations) from the customer success story. The McNemar test produced a test statistic of 1.0 and a p-value of 0.0063, indicating a statistically significant difference between the two prompting methods. As shown in contingency table 5, in 11 cases evaluators rated the Few-Shot persona as complete and the CoT persona as not complete, and in only 1 case the opposite occurred. This result clearly indicates that Few-Shot prompting outperformed CoT prompting in terms of completeness. Figure 10 presents a bar chart that illustrates the distribution of the evaluator ratings for this metric.
Table 5: Contingency Table for Completeness Metric Few-Shot: Yes Few-Shot: No

CoT: Yes a=3 c=1

CoT: No b = 11 d=0

Figure 10: Comparison of the metrics- Completeness Relevance: Relevance was assessed to determine whether the persona focused only on the important and relevant details from the source material. The test yielded a p-value of 0.6250, suggesting that there is no significant difference between the two prompting techniques. Table 6 is the contingency table for the relevance metrics. While there were some variances in individual ratings, the differences were not statistically significant.
Table 6: Contingency Table for Relevance Metric Few-Shot: Yes Few-Shot: No

CoT: Yes a = 11 c=1

CoT: No b=3 d=0

Consistency: The consistency metric evaluated whether the persona introduced incorrect or fabricated information.
The test result for this metric was a p-value of 0.2500, indicating no significant difference between the two prompting approaches. Table 7 is the contingency table for this metric.
Efficiency Metrics: In addition to qualitative performance, the two prompting methods were evaluated based on average generation time and token usage. Few-Shot prompting had an average generation time of 3.66 seconds and used 17

PersonaBOT

Table 7: Contingency Table for Consistency Metric Few-Shot: Yes Few-Shot: No

CoT: Yes a=1 c=0

CoT: No b=3 d = 11

3505.91 tokens. While for CoT prompting, the average generation time was 2.79 seconds and the total average tokens consumed was 2064.2. Figure 11 and Figure 12 show the average time and average token usage across all personas.
These results indicate that CoT prompting outperformed Few-Shot as it required less time and fewer tokens, making it superior and computationally more efficient.

Figure 11: Average Time Taken

4.2.2

Summary of Findings

The evaluation revealed that Few-Shot prompting significantly outperformed CoT prompting in terms of completeness.
Evaluators generally rated Few-Shot personas as more complete than those generated using CoT. For the other two quality metrics, relevance and consistency, no statistically significant differences were found between the prompting methods. However, when evaluated from an efficiency perspective, CoT prompting proved to be both faster and more resource-efficient. The personas generated using the CoT method use fewer tokens and have shorter response times compared to those generated with Few-Shot prompting. Given its superiority in efficiency, CoT prompting was selected for the second iteration of the system. Table 8 summarizes the comparative performance of the two prompting methods in terms of quality and efficiency.
Table 8: Summary of Evaluation of Prompting Techniques Prompting Method Few-Shot CoT

Completeness

Relevance

Consistency

Statistically significant

Statistically insignificant

Statistically insignificant

18

Avg Time (s)
3.66 2.79

Avg Total Tokens 3505.91 2064.2

PersonaBOT

Figure 12: Average Tokens Consumed

4.3

Results for Research Question 3: Impact of Knowledge Base Augmentation

This section presents the results after augmenting the chatbot’s knowledge base with synthetic personas and segmentspecific information. 12 stakeholders from business functions such as R&D, marketing, and customer relations had participated in the evaluation. The evaluation was focused on three primary aspects: accuracy of the responses, ability to handle diverse queries, and overall usefulness of the chatbot in business needs.
4.3.1

Quantitative Results

Participants were asked to assess the impact of the augmented knowledge base on the accuracy of responses, using a scale from 1 (Very Poor) to 10 (Excellent). The average rating across all evaluators was 6.42. This shows a slight improvement from the previous round of chatbot evaluation where the average rating was 5.88. The range of ratings was from 4 to 8. Most of the evaluators gave ratings of 6 or above. This indicates more consistent performance. Figure 13 is a bar chart depicting the distribution of the accuracy rating post improvement.
To evaluate if the system was able to handle diverse and complex queries, participants were asked to assess the system’s ability to respond to complex persona-related queries after data augmentation. 6 participants selected "sometimes", 5 of them selected "most of the time", and 1 selected "always". While the system showed balanced performance in handling complex queries, the responses like "sometimes" indicate some inconsistencies in providing comprehensive and contextually relevant answers. Figure 14 is a bar graph that illustrates the response to handling complex queries.
Regarding the usefulness after the augmented knowledge base, participants provided varied responses on how well the chatbot aligned with business needs. Seven participants rated it as "somewhat", two as "mostly", one as "perfectly", one as "not at all", and one as "not well". This distribution indicates a moderate but mixed perception of the relevance of the augmented knowledge base to business needs. Figure 15 is a bar chart showing the distribution of the ratings regarding usefulness after the augmentation.
4.3.2

Summary of Findings

The evaluation results after updating the knowledge base indicate a moderate but noticeable improvement in the overall performance of the chatbot and practical utility. The integration of synthetic personas and segment-specific information 19

PersonaBOT

Figure 13: Distribution of the accuracy rating - post improvements

Figure 14: Ability of the system to provide response to complex query - post augmentation increased the average accuracy rating from 5.88 to 6.42.It can also be concluded that the system was able to handle complex queries, as none of the participants selected the "never" option.
Regarding practical utility, 81.82% of evaluators rated the augmented knowledge base as at least "somewhat useful", indicating a positive trend in the usefulness of the system in the business contexts. Despite this positive indication, there remains room for further refinement to fully align the system’s output with business needs and expectations. Overall, while the augmentation has contributed to improved performance, further refinement can be made to fully optimize the effectiveness and relevance of the chatbot to business needs.

5

Discussion

A comprehensive analysis of each of the research questions and the potential reasons for the observed outcomes is discussed in this section.
1. Effectiveness of Persona-Based Chatbot in Decision-Making The evaluation of the persona-based chatbot revealed moderate effectiveness in supporting decision-making and automating customer-facing processes. Even though the system was able to provide relevant responses, the average rating of 5.88/10 indicates that there were inconsistencies in providing accurate and contextually 20

PersonaBOT

Figure 15: Alignment of system in business need - post updation appropriate information. One of the reasons for these inconsistencies could be limited data used. When it comes to ability of the system to reduce workload, 75% of participants believed that the system had the potential to reduce workload and automate routine tasks.
2. Effectiveness of Prompting Techniques in Synthetic Persona Generation From the generated personas it can be observed when multiple customers were mentioned in a single customer success story, the LLM tend to focus on only one customer. The relevant information about others customers in the same story are often missed. This tendency to concentrate on a single person could potentially reduce the completeness of the persona generated. This suggests that further refinement in data processing or prompt engineering may be needed to make sure a comprehensive persona is generated in cases where multiple customer information are present in single stories.
The evaluation of the generated personas was conducted with a small sample size, consisting of only three evaluators and a limited number of personas. This limited scope may affect the generalizability of the findings.
Future evaluations with a larger sample could provide better insights. From the statistical test it is evident that the Few-Shot prompting produced more comprehensive personas. However, CoT prompting outperformed Few-Shot in terms of efficiency. This personas generated by this method took less time and consumed fewer tokens. This finding is particularly relevant in the context of real-world deployment, where response time and resource consumption are critical factors. The lack of significant differences in relevance and consistency between the two methods indicates that both prompting techniques was similar in terms factual accuracy and reducing fabricated content. This suggests that while Few-Shot may be preferable for completeness, CoT is an optimal choice for scenarios where response efficiency is prioritized.
3. Impact of Knowledge Base Augmentation The evaluation of the augmented chatbot was conducted with participants from different business functions, including R&D, customer relations, and IT. Their expectations and query types varied significantly, which may have influenced their assessment. For example, IT engineers might ask for more technical information, while customer experience staff might ask queries related to customer data. Addressing the chatbot functionalities for different users may help to improve overall satisfaction. Augmenting the knowledge base with synthetic personas and segment-specific information resulted in a slight increase in accuracy, with the average rating rising to 6.42. 81.82% of participants rated the augmented knowledge base as at somewhat useful. This highlights the need for further refinement in data selection and segmentation.
The analysis indicates that while the persona-based chatbot demonstrated some effectiveness in automating customerfacing processes and supporting decision-making, its overall performance was limited by data limitations and variability in response quality. The Few-Shot prompting method produced more complete personas, whereas CoT prompting was more efficient in terms of response time and token usage. Augmenting the knowledge base has shown slight 21

PersonaBOT

improvements in accuracy and perceived usefulness. Further data refinement and prompt engineering can optimize the system to align more closely with business objectives.

6

Conclusion, Limitations, and Future Work

6.1

Conclusion

In conclusion, this study demonstrates the potential of leveraging LLMs to generate synthetic customer personas and enhance business decision-making through a persona-based RAG chatbot. By comparing Few-Shot and Chain-ofThought prompting methods, the research highlights trade-offs between persona completeness and generation efficiency.
The integration of synthetic personas into the chatbot’s knowledge base led to measurable improvements in response accuracy and user-perceived utility, suggesting that such approaches can effectively complement traditional persona development methods and support scalable, data-driven strategies in industrial settings.
6.2

Limitations

The implementation of the conversational system and the persona generation system faced several limitations that impacted the overall scope and performance of the project.
1. Knowledge Base: The knowledge base used for the chatbot was limited to a small set of verified personas and segment information specifically on mining, quarrying, and aggregates. This limits the diversity of the information available to generate responses and potentially reduces the chatbot’s ability to provide comprehensive insights for other relevant segments.
2. Input Data For Synthetic Persona Generation: For synthetic persona generation, the data source was restricted to customer success stories, which mainly showcased positive customer experiences. These narratives often emphasized successful outcomes rather than describing the challenges faced or areas for improvement.
Consequently, the generated personas may lack a balanced perspective, as they primarily reflect favorable customer experiences with VCE products.
3. Evaluation: The evaluation process was subjective, as different evaluators might have a different opinion on the quality of the persona. This subjectivity could introduce potential biases and inconsistencies in the assessment results.
6.3

Future Work

To make the conversational system and the persona generation better, there are several areas to work on in the future.
1. Expand Data: Currently, data are limited to verified personas and segment-specific information focused on mining, quarrying, and aggregates. Future work could include additional data sources such as customer feedback, satisfaction surveys, and competitor analysis reports. This would provide a more comprehensive dataset that captures diverse customer experiences.
2. Persona Generation: Another area for further development is the refinement of persona generation techniques.
The present approach utilizes GPT-4o Mini with prompting methods. Future research could explore more advanced LLMs or using a fine-tuned LLM with information about VCE. This approach could potentially improve the contextual accuracy and relevance of the personas generated.
3. Real-Time Updation: As VCE customer needs and challenges change over time, implementing mechanisms to track changes in customer personas over time would be beneficial. This could be achieved by periodically updating the knowledge base with new customer success stories and feedback data, allowing the system to maintain relevance over time.
4. Chatbot Framework: When it comes to the chatbot, exploring advanced RAG frameworks such as GraphRAG [37] and Multi-Hop RAG [38] could improve the system. Graph-RAG introduces graph structures to link related personas, customer success stories, and other data points. This approach could capture more complex relationships between data elements, enabling richer and more context-aware retrieval. Similarly, Multi-Hop RAG [38] extends the RAG framework by retrieving and sequentially processing multiple data points to generate more comprehensive responses. Another promising direction includes improving the chatbot’s interaction to match specific user roles. For example, tailoring the conversation paths for specific roles, such as R&D Engineers or Customer Experience Teams. This could potentially improve the relevance of responses and provide more targeted insights.
22

PersonaBOT

5. Evaluation Metrics: Lastly, including automated metrics for evaluating the retrieval system and persona quality could streamline the evaluation process and reduce dependence on subjective human evaluation.

Acknowledgments The authors would like to express their sincere gratitude to the Department of Future Solutions, the Department of Brand Experience, the Department of Digital&IT, and the Department of Finance at Volvo Construction Equipment for their support and guidance throughout the project.

References [1] More, Pratik and Pothula, Shiva Sai Kiran (2025). Quantum Leap in Customer Persona Development: Enhancing Consumer Profiles and Experiences Using Quantum AI. In The Quantum AI Era of Neuromarketing, pp. 133–156.
[2] Morandé (2023). Digital Persona: Reflection on the Power of Generative AI for Customer Profiling in Social Media Marketing.
[3] Praveen, SV and Gajjar, Pranshav and Ray, Rajeev Kumar and Dutt, Ashutosh (2024). Crafting clarity: Leveraging large language models to decode consumer reviews. Journal of Retailing and Consumer Services, 81, 103975.
[4] Panda, Swaroop (2024). LLMs’ ways of seeing User Personas. arXiv preprint arXiv:2409.14858 [5] Zhang, Zhehao and Rossi, Ryan A and Kveton, Branislav and Shao, Yijia and Yang, Diyi and Zamani, Hamed and Dernoncourt, Franck and Barrow, Joe and Yu, Tong and Kim, Sungchul and others (2024). Personalization of large language models: A survey. arXiv preprint arXiv:2411.00027 [6] Tseng, Yu-Min and Huang, Yu-Chao and Hsiao, Teng-Yun and Hsu, Yu-Ching and Foo, Jia-Yin and Huang, ChaoWei and Chen, Yun-Nung (2024). Two tales of persona in llms: A survey of role-playing and personalization.
arXiv preprint arXiv:2406.01171 [7] Ha, Juhye and Jeon, Hyeon and Han, Daeun and Seo, Jinwook and Oh, Changhoon (2024). CloChat: Understanding How People Customize, Interact, and Experience Personas in Large Language Models. In Proceedings of the CHI Conference on Human Factors in Computing Systems, pp. 1–24.
[8] Cheng, Myra and Durmus, Esin and Jurafsky, Dan (2023). Marked Personas: Using Natural Language Prompts to Measure Stereotypes in Language Models. arXiv preprint arXiv:2305.18189 [9] Barandoni, Simone and Chiarello, Filippo and Cascone, Lorenzo and Marrale, Emiliano and Puccio, Salvatore (2024). Automating Customer Needs Analysis: A Comparative Study of Large Language Models in the Travel Industry. arXiv preprint arXiv:2404.17975 [10] Jiang, Hang and Zhang, Xiajie and Cao, Xubo and Kabbara, Jad (2023). PersonaLLM: Investigating the Ability of Large Language Models to Express Big Five Personality Traits. arXiv preprint arXiv, 2305 [11] Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33, 9459–9474.
[12] Sun, Lipeipei and Qin, Tianzi and Hu, Anran and Zhang, Jiale and Lin, Shuojia and Chen, Jianyan and Ali, Mona and Prpa, Mirjana (2024). Persona-L has Entered the Chat: Leveraging LLM and Ability-based Framework for Personas of People with Complex Needs. arXiv preprint arXiv:2409.15604 [13] McGinn, Jennifer and Kotamraju, Nalini (2008). Data-driven persona development. In Proceedings of the SIGCHI conference on human factors in computing systems, pp. 1521–1524.
[14] Zhang, Xiang and Brown, Hans-Frederick and Shankar, Anil (2016). Data-driven personas: Constructing archetypal users with clickstreams and user telemetry. In Proceedings of the 2016 CHI conference on human factors in computing systems, pp. 5350–5359.
[15] Jung, Soon-gyo and Salminen, Joni and Kwak, Haewoon and An, Jisun and Jansen, Bernard J (2018). Automatic persona generation (APG) a rationale and demonstration. In Proceedings of the 2018 conference on human information interaction & retrieval, pp. 321–324.
[16] Farseev, Aleksandr and Yang, Qi and Ongpin, Marlo and Gossoudarev, Ilia and Chu-Farseeva, Yu-Yi and Nikolenko, Sergey (2024). SOMONITOR: Combining Explainable AI & Large Language Models for Marketing Analytics. arXiv e-prints, arXiv–2407.
23

PersonaBOT

[17] Salminen, Joni and Liu, Chang and Pian, Wenjing and Chi, Jianxing and H. (2024). Deus ex machina and personas from large language models: investigating the composition of AI-generated persona descriptions. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pp. 1–20.
[18] Zhang, Xishuo and Liu, Lin and Wang, Yi and Liu, Xiao and Wang, Hailong and Ren, Anqi and Arora, Chetan (2023). Personagen: A tool for generating personas from user feedback. In 2023 IEEE 31st International Requirements Engineering Conference (RE), pp. 353–354.
[19] De Paoli, Stefano (2023). Improved prompting and process for writing user personas with LLMs, using qualitative interviews: Capturing behaviour and personality traits of users. arXiv preprint arXiv:2310.06391 [20] Goel, Toshali and Shaer, Orit and Delcourt, Catherine and Gu, Quan and Cooper, Angel (2023). Preparing future designers for human-ai collaboration in persona creation. In Proceedings of the 2nd Annual Meeting of the Symposium on Human-Computer Interaction for Work, pp. 1–14.
[21] Hadi, Muhammad Usman and Al-Tashi, Qasem and Qureshi, Rizwan and Shah, Abbas and Muneer, Amgad and Irfan, Muhammad and Zafar, Anas and Shaikh, Muhammad Bilal and Akhtar, Naveed and Al-Garadi, Mohammed Ali and others (n.d.). LLMs: A Comprehensive Survey of Applications, Challenges, Datasets, Models, Limitations, and Future Prospects.
[22] Cheung, Ming (2024). A Reality check of the benefits of LLM in business. arXiv preprint arXiv:2406.10249 [23] OpenAI (2025). Prompt Engineering – Enhance Results with Prompt Engineering Strategies.
[24] Microsoft Azure Team (2024). Azure AI Search: Outperforming Vector Search with Hybrid Retrieval and Reranking.
[25] Loni, Mohammad and Poursalim, Fatemeh and Asadi, Mehdi and Gharehbaghi, Arash (2024). A Review on Generative AI Models for Synthetic Medical Text, Time Series, and Longitudinal Data. arXiv preprint arXiv:2411.12274.
[26] Johannesson, Paul and Perjons, Erik and Johannesson, Paul and Perjons, Erik (2014). A method framework for design science research. An introduction to design science, 75–89.
[27] Pembury Smith, Matilda QR and Ruxton, Graeme D (2020). Effective use of the McNemar test. Behavioral Ecology and Sociobiology, 74, 1–9.
[28] Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and others (2020).
Language models are few-shot learners. Advances in neural information processing systems, 33, 1877–1901.
[29] Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Xia, Fei and Chi, Ed and Le, Quoc V and Zhou, Denny and others (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35, 24824–24837.
[30] Achiam, Josh and Adler, Steven and Agarwal, Sandhini and Ahmad, Lama and Akkaya, Ilge and Aleman, Florencia Leoni and Almeida, Diogo and Altenschmidt, Janko and Altman, Sam and Anadkat, Shyamal and others (2023). Gpt-4 technical report. arXiv preprint arXiv:2303.08774 [31] Goodfellow, Ian J and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua (2014). Generative adversarial nets. Advances in neural information processing systems, 27 [32] Kingma, Diederik P and Welling, Max and others (2013). Auto-encoding variational bayes. Banff, Canada [33] Blei, David M and Ng, Andrew Y and Jordan, Michael I (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993–1022.
[34] Roberts, Margaret E and Stewart, Brandon M and Tingley, Dustin and Lucas, Christopher and Leder-Luis, Jetson and Gadarian, Shana Kushner and Albertson, Bethany and Rand, David G (2014). Structural topic models for open-ended survey responses. American journal of political science, 58(4), 1064–1082.
[35] Peffers, Ken and Tuunanen, Tuure and Rothenberger, Marcus and Chatterjee, S. (2007). A design science research methodology for information systems research. Journal of Management Information Systems, 24, 45-77.
[36] Peffers, Ken and Tuunanen, Tuure and Rothenberger, Marcus A and Chatterjee, Samir (2007). A design science research methodology for information systems research. Journal of management information systems, 24(3), 45–77.
[37] Han, Haoyu and Wang, Yu and Shomer, Harry and Guo, Kai and Ding, Jiayuan and Lei, Yongjia and Halappanavar, Mahantesh and Rossi, Ryan A and Mukherjee, Subhabrata and Tang, Xianfeng and others (2024). Retrievalaugmented generation with graphs (graphrag). arXiv preprint arXiv:2501.00309 24

PersonaBOT

[38] Tang, Yixuan and Yang, Yi (2024). Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries. arXiv preprint arXiv:2401.15391 [39] Schreiber, William, White, Jules, and Schmidt, Douglas C. (2024). A Pattern Language for Persona-based Interactions with LLMs.

25


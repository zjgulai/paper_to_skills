<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py --pdf
     arxiv_id : 2206.07587
     paper_id : 2206.07587
     source   : paper2skills-vault/papers/07-NLP-VOC/2206.07587/paper.pdf
     fulltext : 是（本地 PDF 转换）
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

Cross-lingual AMR Aligner: Paying Attention to Cross-Attention Abelardo Carlos Martínez Lorenzo1,2∗ Pere-Lluís Huguet Cabot1,2∗ 2 Roberto Navigli 1 Babelscape, Italy 2 Sapienza NLP Group, Sapienza University of Rome {martinez,huguetcabot}@babelscape.com navigli@diag.uniroma1.it

arXiv:2206.07587v2 [cs.CL] 19 Jun 2023

Abstract This paper introduces a novel aligner for Abstract Meaning Representation (AMR) graphs that can scale cross-lingually, and is thus capable of aligning units and spans in sentences of different languages. Our approach leverages modern Transformer-based parsers, which inherently encode alignment information in their cross-attention weights, allowing us to extract this information during parsing. This eliminates the need for English-specific rules or the Expectation Maximization (EM) algorithm that have been used in previous approaches. In addition, we propose a guided supervised method using alignment to further enhance the performance of our aligner. We achieve state-of-the-art results in the benchmarks for AMR alignment and demonstrate our aligner’s ability to obtain them across multiple languages. Our code will be available at github.com/Babelscape/AMR-alignment.

1

Introduction

At the core of Natural Language Understanding lies the task of Semantic Parsing, aimed at translating natural language text into machine-interpretable representations. One of the most popular semantic formalisms is the Abstract Meaning Representation (Banarescu et al., 2013, AMR), which embeds the semantics of a sentence in a directed acyclic graph, where concepts are represented by nodes, such as time, semantic relations between concepts by edges, such as :beneficiary, and the co-references by reentrant nodes, such as r representing rose. In crosslingual AMR, the English AMR graph represents the sentence in different languages (see Figure 1).
To date, AMR has been widely used in Machine Translation (Song et al., 2019), Question Answering (Lim et al., 2020; Kapanipathi et al., 2021), Human-Robot Interaction (Bonial et al., 2020), Text Summarization (Hardy and Vlachos, 2018;
∗

Equal contributions.

Liao et al., 2018) and Information Extraction (Rao et al., 2017), among other areas.
The alignment between spans in text and semantic units in AMR graphs is an essential requirement for a variety of purposes, including training AMR parsers (Zhou et al., 2021), cross-lingual AMR parsing (Blloshmi et al., 2020), downstream task application (Song et al., 2019), or the creation of new semantic parsing formalisms (Navigli et al., 2022; Martínez Lorenzo et al., 2022).
Despite the emergence of various alignment generation approaches, such as rule-based methods (Liu et al., 2018) and statistical strategies utilizing Expectation Maximization (EM) (Pourdamghani et al., 2014; Blodgett and Schneider, 2021), these methods rely heavily on English-specific rules, making them incompatible with cross-lingual alignment. Furthermore, even though several attempts extend the alignment to non-English sentences and graphs (Damonte and Cohen, 2018; Uhrig et al., 2021), these efforts are inherently monolingual and therefore lack the connection to the richer AMR graph bank available in English, which can be exploited as a source of interlingual representations.
On the other hand, current state-of-the-art AMR parsers are auto-regressive neural models (Bevilacqua et al., 2021; Bai et al., 2022) that do not generate alignment when parsing the sentence to produce the graph. Therefore, to obtain both, one needs to i) predict the graph and then ii) generate the alignment using an aligner system that is based on language-specific rules.
This paper presents the first AMR aligner that can scale cross-lingually by leveraging the implicit information acquired in Transformer-based parsers (Bai et al., 2022). We propose an approach for extracting alignment information from crossattention, and a guided supervised method to enhance the performance of our aligner. We eliminate the need for language-specific rules and enable simultaneous generation of the AMR graph and

Figure 1: AMR graph (left) and its corresponding sentences in several languages (right). Colors represent alignment.

alignment. Our approach is efficient and robust, and is suitable for cross-lingual alignment of AMR graphs.
Our main contributions are: (i) we explore how Transformer-based AMR parsers preserve implicit alignment knowledge and how we can extract it;
(ii) we propose a supervised method using crossattention to enhance the performance of our aligner, (iii) we achieve state-of-the-art results along different alignment standards and demonstrate the effectiveness of our aligner across languages.

2

Related Work

AMR alignment Since the appearance of AMR as a Semantic Parsing formalism, several aligner systems have surfaced that provide a link between the sentence and graph units. JAMR (Flanigan et al., 2014) is a widely used aligner system that employs an ordered list of 14 criteria, including exact and fuzzy matching, to align spans to subgraphs.
However, this approach has limitations as it is unable to resolve ambiguities or learn novel alignment patterns. TAMR (Liu et al., 2018) extends JAMR by incorporating an oracle parser that selects the alignment corresponding to the highest-scored candidate AMR graph. ISI (Pourdamghani et al., 2014)
aligner utilizes an EM algorithm to establish alignment between words and graphs’ semantic units.
First, the graph is linearized, and then the EM algorithm is employed with a symmetrized scoring function to establish alignments. This method leads to more diversity in terms of alignment patterns, but fails to align easy-to-recognize patterns that could be aligned using rules. LEAMR (Blodgett and Schneider, 2021) is another aligner system that combines rules and EM. This approach aligns all the subgraph structures to any span in the sentence. However, it is based on language-specific rules, making it unsuitable for cross-lingual settings. Moreover, despite several attempts to extend the alignment to non-English languages An-

chiêta and Pardo (2020); Oral and Eryiğit (2022), these efforts are still monolingual since they rely on language-specific strategies. Consequently, in this paper we present an approach that fills this gap.
Cross-attention Most state-of-the-art systems for AMR parsing are based on Encoder-Decoder Transformers, specifically on BART (Lewis et al., 2020). These models consist of two stacks of Transformer layers, which utilize self- and crossattention as their backbone. The popularity of Transformer models has led to increased interest in understanding how attention encodes information in text and relates to human intuition (Vashishth et al., 2019) and definitions of explainability (Bastings and Filippova, 2020; Bibal et al., 2022). Research has been conducted on how attention operates, relates to preconceived ideas, aggregates information, and explains model behavior for tasks such as natural language inference(Stacey et al., 2021), Translation (Yin et al., 2021; Zhang and Feng, 2021; Chen et al., 2021), Summarization (Xu et al., 2020; Manakul and Gales, 2021) or Sentiment Analysis (Wu et al., 2020). Furthermore, there have been attempts to guide attention to improve interpretability or performance in downstream tasks (Deshpande and Narasimhan, 2020;
Sood et al., 2020). However, to the best of our knowledge, there has been no prior study on attention for AMR parsing. This paper fills this gap by investigating the role of attention in AMR parsing.

3

Method

Originally described by Vaswani et al. (2017) as “multi-head attention over the output of the Encoder”, and referred to as cross-attention in Lewis et al. (2020), it enables the Decoder to attend to the output of the Encoder stack, conditioning the hidden states of the autoregressive component on the input text. Self-attention and cross-attention

modules are defined as:
Attention(Q, K, V ) = att(Q, K)V QK T att(Q, K) = sof tmax( √ )
dk

of imposing sparsity by employing the scalar mixing approach introduced in ELMo (Peters et al., 2018). We learn a weighted mix of each head and obtain a single attention matrix:
ℓ

att = γ

CrossAtt(Q, K, V ) = Concat(head1 , ..., headH )W O headh = Attention(QWhQ , KWhK , V WhV )
where K, V = E ℓ ∈ Rne ×dk H and Q = Dℓ ∈ Rnd ×dk H are the Encoder and Decoder hidden states at layer ℓ, ne and nd are the input and output sequence lengths, H is the number of heads, WhQ , WhK and WhV ∈ Rdk H×dk are learned weights that project the hidden states to the appropriate dimensions, dk , for each head and W O ∈ Rdk H×dk H is a final learned linear projection. Therefore in each head h and layer ℓ we define the attention weights as attℓh = att(Dℓ WhQ , E ℓ WhK ) ∈ Rnd ×ne .
3.1

Unguided Cross-Attention

We argue that there is an intuitive connection between cross-attention and alignments. Under the assumption the Decoder will attend to the parts in the input that are more relevant to predicting the next token, we infer that, when decoding the tokens for a certain node in the graph, attention should focus on related tokens in the input, and therefore the words that align to that node. We will use the crossattention matrices (attℓh ) to compute an alignment between the input and the output.
3.2

Guided Cross-Attention

We also aim to explore whether cross-attention can be guided by the alignment between the words of the sentence and the nodes of the graph. To this end, we construct a sparse matrix align ∈ Rnd ×ne from the automatically-generated alignments:

align(i, j) =

  1 if

xi ∼ yj

0 if

xi ≁ yj



where ∼ indicates alignment between subword token xi and graph token yj .
However, even though there are sparse versions of attention (Martins and Astudillo, 2016), these did not produce successful alignments in our experiments. Hence we choose to alleviate the constraint

H−1 X

sℓh attℓh ∈ Rnd ×ne

(1)

h=0

where s = sof tmax(a) with scalar learnable parameters γ, a0 , . . . , aH .
The model has the flexibility to learn how to distribute weights such that certain heads give sparser attention similar to alignment, while others can encode additional information that is not dependent on alignment. In our experiments, we use the implementation of Bevilacqua et al. (2021, SPRING)
to train our parser but add an extra cross-entropy loss signal:
L=−

nd X

log pBART (yj | y<j , x)

j=1



 nd X ne X

ℓ

 eatt (i,j)
align(i, j)   log  nd n  P e P ℓ (i,k)
att align(k, j)
e Pj=1 i=1

−

i align(i,j)>0

3.3

k=1

k=1

Saliency Methods

A theoretical alternative to our reasoning about cross-attention is the use of input saliency methods.
These methods assign higher importance to the input tokens that correspond to a particular node in the graph or were more important in their prediction during decoding. To obtain these importance weights, we employ Captum (Kokhlikyan et al., 2020), an open-source library for model interpretability and understanding, which provides a variety of saliency methods, including gradientbased methods such as Integrated Gradients (IG), Saliency (Simonyan et al., 2014), and Input X Gradient (IxG), backpropagation-based methods such as Deeplift (Shrikumar et al., 2017) and Guided Backpropagation (GB) (Springenberg et al., 2015), and finally occlusion-based methods (Zeiler and Fergus, 2014).
We obtain a weight matrix sal ∈ Rnd ×ne with the same size as the cross-attention matrix and use it to extract alignments in the same fashion as the unguided cross-attention method. This approach allows us to explore the input tokens that have a greater impact on the decoding process and can aid in understanding the reasoning behind the alignments made by the model.

3.4

Alignment Extraction

Our algorithm1 to extract and align the input-output spans is divided into six steps:
1. Alignment score matrix: we create a matrix M ∈ Rnd ×ne , where ne is the number of tokens in the sentence and nd is the number of tokens in the linearized graph, using the crossattention weights (attℓh or attℓ ) as described in Section 3.
2. Span segmentation: For each sentence word, we sum the scores of tokens that belong to the same word column-wise in M . Then, for LEAMR alignments (see Section 4.2), the sentence tokens are grouped into spans using their span segmentation (see Appendix A).
3. Graph segmentation: We sum the score of tokens that belong to the same graph’s semantic unit row-wise in M .
4. Sentence graph tokens map: We iterate over all the graph’s semantic units and map them to the sentence span with highest score in M .
5. Special graph structures: We revise the mapping by identifying subgraphs that represent literal or matching spans – e.g., named entities, dates, specific predicates, etc. – and align them accordingly.
6. Alignment formatting: We extract the final alignments to the appropriate format using the resulting mapping relating graph’s semantic units to sentence spans.

4

Experimental Setup

4.1

Graph inventory

AMR 3.0 (LDC2020T02) consists of 59,255 sentence-graph pairs that are manually annotated.
However, it lacks alignment information between the nodes in the graphs and the concepts in the sentences. We use the train split for the guided approach and use the respective validation and test splits from the alignment systems. Additionally, to evaluate cross-lingual performance, we use the gold German, Italian, and Spanish sentences of “AMR 2.0 – Four Translation” (LDC2020T07)
which are human parallel translations of the test set 1

The pseudo-algorithm is described in the Appendix C.

in AMR 2.02 , paired with their English graphs from the AMR 3.0 test set. Despite this, as the graph inventory does not contain alignment information, it becomes necessary to access other repositories in order to obtain the alignment.
4.2

Alignment Standards

We propose an approach that is agnostic to different alignment standards and we evaluate it on two standards that are commonly used: ISI and LEAMR.
ISI The ISI standard, as described in (Pourdamghani et al., 2014), aligns single spans in the sentence to graphs’ semantic units (nodes or relations), and aligns relations and reentrant nodes when they appear explicitly in the sentence. The alignments are split into two sets of 200 annotations each, which we use as validation and test sets, updated to the AMR 3.0 formalism. For the cross-lingual alignment setup, we project English ISI graph-sentence alignments to the sentences in other languages, using the machine translation aligner (Dou and Neubig, 2021). This involves connecting the nodes in the graph to the spans in non-English sentences using the projected machine translation alignments between the English spans and the corresponding non-English sentence spans.
By leveraging this, we are able to generate a silver alignment for cross-lingual AMR, which enables us to validate the model’s performance in a cross-lingual setup and determine its scalability across-languages.
LEAMR The LEAMR standard differentiates among four different types of alignment: i) Subgraph Alignments, where all the subgraphs that explicitly appear in the sentence are aligned to a list of consecutive spans, ii) Duplicate Subgraph, where all the subgraphs that represent omit repeated concepts in the sentence are aligned, iii)
Relation Alignments, where all the relations that were not part of a previous subgraph structure are aligned, and iv) Reentrancy Alignments, where all the reentrant nodes are aligned. In contrast to ISI, all the semantic units in the graph are aligned to some list of consecutive spans in the text. We use 150 alignments as the validation set and 200 as the test set, which includes sentence-graph pairs from The Little Prince Corpus (TLP) complemented with randomly sampled pairs from AMR 3.0.
2

The sentences of AMR 2.0 are a subset of AMR 3.0.

Figure 2: Heatmap of Pearson’s r correlation to LEAMR validation set for unguided (left) and guided on half the heads in layer 3 (right) cross-attention weights, as well as saliency methods (bottom).

4.3

Model

We use SPRING (Bevilacqua et al., 2021) as our parsing model based on the BART-large architecture (Lewis et al., 2020) for English and SPRING based on mBART for non-English languages mBART (Liu et al., 2020) for the multilingual setting. We extract all attℓh matrices from a model trained on AMR 3.0 as in Blloshmi et al.
(2021) in order to perform our unguided crossattention analysis. For the guided approach we re-train using the same hyperparameters as the original implementation, but with an extra loss signal as described in Section 3.2 based on either LEAMR or ISI. When using LEAMR alignments, we restructure the training split in order to exclude any pair from their test and validation sets.

5

Experiments

5.1

Correlation

In this study, we investigate the correlation between cross-attention and alignment by computing the Pearson’s r correlation coefficient between the attℓh matrix and the LEAMR alignment matrix align.
To do so, we first flatten the matrices and remove any special tokens that are not relevant for alignment. As shown in Figure 2, there is a clear positive correlation between the two.
While we do not have a clear explanation for

why certain heads have a higher correlation than others, it is evident that there is a connection between cross-attention and alignment. For example, head 6 in layer 3 (i.e., att36 ) has a correlation coefficient of 0.635, approximately the same as the sum of the entire layer.
With regard to the saliency methods described in Section 3.3, the two most highly correlated methods were Saliency and GB, with a correlation coefficient of 0.575. Despite this result, we observe that saliency methods tend to focus more on essential parts of the sentence, such as the subject or predicate. These parts are usually aligned to more nodes and relations, which explains the high correlation, but they lack nuance compared to cross-attention.
Our best results were obtained by supervising layer 3 during training with the approach outlined in Section 3.2, using Cross-Entropy Loss on half of the heads (i.e., 3, 4, 5, 6, 7, 11, 12, and 15) that were selected based on their correlation on the validation set. This did not affect the performance of parsing.
When we looked at att3 using the learned weighted mix from Equation 1 with LEAMR alignments, the correlation reached 0.866, which is significantly higher than any other method. Figure 2 shows the impact of supervising half the heads on layer 3 and how it influences heads in other layers.
To gain a better understanding of these results, we present an example from the TLP corpus in Fig-

Figure 3: Unguided (left), saliency (center-left) and guided (center-right) alignment weights and LEAMR (right)
gold alignment for lpp_1943.1209. To explore all cross- attention weights interactively, please go here.

ure 3 to illustrate the different methods, including cross-attention and saliency methods. The left image shows the cross-attention values for att36 . Despite not having seen any alignment information, the model is able to correctly match non-trivial concepts such as "merchant" and "person". The center-left image illustrates how saliency methods focus on essential parts of the sentence, but lack nuance compared to cross-attention. The center-right image shows that supervising learning on layer 3 results in more condensed attention, which is associated with the improvement in correlation. However, it is important to note that the model can reliably attend to incorrect positions, such as aligning "pointer" to "merchant" instead of "sold".
5.2

Results

LEAMR Table 1 shows the performances of our two approaches on the LEAMR gold alignments compared to previous systems. We use the same evaluation setup as Blodgett and Schneider (2021), where the partial match assigns a partial credit from Jaccard indices between nodes and tokens. In both guided and unguided methods, we extract the score matrix for Algorithm 1 from the sum of the crossattention in the first four layers. We use a Wilcoxon signed-rank test (Wilcoxon, 1945) on the alignment matches per graph to check for significant differences. Both our approaches are significantly different compared to LEAMR (p=0.031 and p=0.007 respectively). However, we find no statistical difference between our unguided and guided approaches (p=0.481).

Our guided attention approach performs best, improving upon LEAMR on Subgraph (+0.5) and Relation (+2.6). For Reentrancy, performance is relatively low, and we will explore the reasons for this in Section 7. Perhaps most interesting is the performance of the unguided system using raw cross-attention weights from SPRING. The system remains competitive against the guided model without having access to any alignment information. It outperforms LEAMR which, despite being unsupervised with respect to alignments, relies on a set of inductive biases and rules based on alignments.
While we also draw on specific rules related to the graph structure in post-processing, we will need to investigate their impact in an ablation study.
Relations that are argument structures (i.e., :ARG and :ARG-of ) usually depend on the predictions for their parent or child nodes; hence their improvement would be expected to be tied to the Subgraph Alignment. The results in Table 2 reassure us that this intuition is correct. Notice how for Single Relations (such as :domain or :purpose in Figure 3) the performance by LEAMR was much lower, even worse than that of ISI: Blodgett and Schneider (2021) argued that this was due to the model being overeager to align to frequent prepositions such as to and of. On the other hand, our unguided method achieves 15 points over ISI and 20 over LEAMR, which hints at the implicit knowledge on alignment that cross-attention encodes. Our guided approach experiences a considerable drop for Single Relations since it was trained on data generated by LEAMR, replicating its faulty behavior albeit

Exact Alignment P R F1

Partial Alignment P R F1

Spans F1

Coverage

Subgraph Alignment (1707)

ISI JAMR TAMR LEAMR LEAMR † Ours - Unguided Ours - Guided - ISI Ours - Guided - LEAMR

71.56 87.21 85.68 93.91 93.74 94.11 89.87 94.39

68.24 83.06 83.38 94.02 93.91 94.49 91.97 94.67

69.86 85.09 84.51 93.97 93.82 94.30 90.91 94.53

78.03 90.29 88.62 95.69 95.51 96.03 92.11 96.62

74.54 85.99 86.24 95.81 95.68 96.42 94.27 96.90

76.24 88.09 87.41 95.75 95.60 96.26 93.18 96.76

86.59 92.38 94.64 96.05 95.54 95.94 93.69 96.40

78.70 91.10 94.90 100.00 100.00 100.00 100.00 100.00

Relation Alignment (1263)

ISI LEAMR LEAMR † Ours - Unguided Ours - Guided - ISI Ours - Guided - LEAMR

59.28 85.67 84.63 87.14 83.82 88.03

8.51 87.37 84.85 87.59 83.39 88.18

14.89 85.52 84.74 87.36 83.61 88.11

66.32 88.74 87.77 89.87 86.45 91.08

9.52 88.44 87.99 90.33 86.00 91.24

16.65 88.59 87.88 90.10 86.22 91.16

83.09 95.41 91.98 91.03 87.30 91.87

9.80 100.00 100.00 100.00 100.00 100.00

Reentrancy Alignment (293)

LEAMR LEAMR † Ours - Unguided Ours - Guided - ISI Ours - Guided - LEAMR

55.75 54.61 44.75 42.09 56.90

54.61 54.05 44.59 39.35 57.09

55.17 54.33 44.67 40.77 57.00

— — — — —

— — — — —

— — — — —

— — — — —

100.00 100.00 100.00 100.00 100.00

Duplicate Subgraph Alignment (17)

LEAMR LEAMR † Ours - Unguided Ours - Guided - ISI Ours - Guided - LEAMR

66.67 68.75 77.78 63.16 70.00

58.82 64.71 82.35 70.59 82.35

62.50 66.67 80.00 66.67 75.68

70.00 68.75 77.78 65.79 72.50

61.76 64.71 82.35 73.53 85.29

65.62 66.67 80.00 69.44 78.38

— — — — —

100.00 100.00 100.00 100.00 100.00

Table 1: LEAMR alignment results. Column blocks: models; Exact and Partial alignment scores; Span and Coverage measures. Row blocks: alignment types, number of instances in brackets. † indicates our re-implementation. Guided versions using ISI/LEAMR silver alignments. Bold is best.

AMR parser

P

R

F1

ALL

ISI LEAMR † Ours - Unguided Ous - Guided - LEAMR

59.3 84.6 87.1 88.0

08.5 84.9 87.6 88.2

14.9 84.7 87.4 88.1

Single Relations (121)

ISI LEAMR † Ours - Unguided Ous - Guided - LEAMR

82.9 64.8 79.5 77.5

52.1 55.7 79.5 64.8

64.0 59.9 79.5 70.5

Argument Structure (1042)

ISI LEAMR † Ours - Unguided Ous - Guided - LEAMR

39.6 86.6 87.9 89.0

03.5 88.2 88.4 90.8

06.4 87.4 88.2 89.9

Table 2: LEAMR results breakdown for Relation Alignment. Column blocks: relation type; models; scores.
Bold is best. † indicates our re-implementation.

being slightly more robust.
ISI When we test our systems against the ISI alignments, both our models achieve state-of-the-

art results, surpassing those of previous systems, including LEAMR. This highlights the flexibility of cross-attention as a standard-agnostic aligner (we provide additional information in Appendix B). Table 3 shows the performance of our systems and compares ones with the ISI alignment as a reference. We omit relations and Named Entities to focus solely on non-rule-based alignments and have a fair comparison between systems. Here, our aligner does not rely on any span-segmentation, hence nodes and spans are aligned solely based on which words and nodes share the highest crossattention values. Still, both our alignments outperform those of the comparison systems in English.
Moreover, only our approach achieves competitive results in Spanish, German and Italian – obtaining 40 points more on average above the second best model – while the other approaches are hampered by the use of English-specific rules. However, we found two reasons why non-English systems

JAMR TAMR LEAMR Unguided Guided

P

EN R

F1

P

DE R

F1

P

ES R

F1

P

IT R

F1

P

AVG R

F1

92.7 92.1 85.9 95.4 96.3

80.1 84.5 92.3 93.2 94.2

85.9 88.1 89.0 94.3 95.2

75.4 73.7 8.4 64.0 —

6.6 6.4 9.3 74.4 —

12.1 11.8 8.8 68.85 —

84.4 84.0 8.1 67.9 —

16.1 16.4 9.0 77.1 —

27.1 27.5 8.5 72.2 —

64.8 64.3 9.0 67.4 —

13.2 13.2 9.5 75.5 —

21.9 21.9 9.3 71.2 —

79.3 78.5 27.9 73.7 —

29.0 30.1 30.0 80.1 —

36.8 37.3 28.9 76.6 —

Table 3: ISI results. Column blocks: models, language.
GOLD

Sub.
Rel.
Reen.
Dupl.

Without Rules

Layers

LEAMR †

Ung.

Guided

LEAMR †

Ung.

Guided

Sal.

Unguided [0:4] [4:8] [8:12]

96.5 87.1 56.8 62.9

96.7 89.2 46.7 80.0

97.0 90.3 59.0 75.7

87.6 26.6 15.2 40.0

88.6 60.1 38.6 71.8

93.4 83.4 57.0 73.7

62.2 50.0 34.5 9.5

94.3 87.7 44.7 80.0

69.8 72.7 41.1 11.1

63.3 61.6 36.1 27.3

[0:12]

[0:4]

[4:8]

Guided [8:12] [0:12]

[3]

[3]*

87.7 84.5 41.9 64.3

94.5 88.1 57.0 75.9

74.4 73.8 39.2 30.0

66.3 62.5 33.0 27.3

93.7 86.2 52.7 70.3

93.7 85.9 53.4 70.3

93.2 87.9 51.0 66.7

Table 4: F1 results on Exact Alignment on ablation studies. Column blocks: alignment types; using gold spans;
removing rules from the models; by layers. Guided approach using LEAMR silver alignments. † indicates our re-implementation. [x:y] indicates sum from layer x to y. * indicates weighted head sum. Bold is best.

perform worse than in English: 1) linguistic divergences (as explained in Wein and Schneider (2021)), and ii) the machine translation alignment error.

6

Ablation Study

Gold spans LEAMR relies on a span segmentation phase, with a set of multiword expressions and Stanza-based named entity recognition. We use the same system in order to have matching sentence spans. However, these sometimes differ from the gold spans, leading to errors. Table 4 (left)
shows performance using an oracle that provides gold spans, demonstrating how our approach still outperforms LEAMR across all categories.
Rules All modern alignment systems depend on rules to some degree. For instance, we use the subgraph structure for Named Entities, certain relations are matched to their parent or child nodes, etc. (see Appendix A for more details). But what is the impact of such rules? As expected, both LEAMR and our unguided method see a considerable performance drop when we remove them.
For Relation, LEAMR drops by almost 60 points, since it relies heavily on the predictions of parent and child nodes to provide candidates to the EM model. Our unguided approach also suffers from such dependency, losing 25 points. However, our guided model is resilient to rule removal, dropping by barely one point on Subgraph and 5 points on Relation.
Layers Figure 2 shows how alignment acts

differently across heads and layers. We explore this information flow in the Decoder by extracting the alignments from the sum of layers at different depths. The right of Table 4 shows this for both our unguided and guided models, as well as the Saliency method. [3] indicates the sum of heads in the supervised layer, while [3]* is the learned weighted mix. From our results early layers seem to align more explicitly, with performance dropping with depth. This corroborates the idea that Transformer models encode basic semantic information early (Tenney et al., 2019). While layers 7 and 8 did show high correlation values, the cross-attention becomes more disperse with depth, probably due to each token encoding more contextual information.

7

Error Analysis

We identify two main classes of error that undermine the extraction of alignments.
Consecutive spans Because each subgraph in LEAMR is aligned to a list of successive spans, the standard cannot deal correctly with transitive phrasal verbs. For example, for the verb "take off"
the direct object might appear in-between ("He took his jacket off in Málaga"). Because these are not consecutive spans, we align just to "take" or "off".
Rules We have a few rules for recognizing subgraph structures, such as Named Entities, and align them to the same spans. However, Named Entity structures contain a placeholder node indicating the entity type; when the placeholder node appears explicitly in the sentence, the node should not be

Figure 4: AMR graph and its sentence from "AMR 2.0 – Four Translation”. Color represents the alignment.

part of the Named Entity subgraph. For example, when aligning ‘Málaga’, the city, the placeholder node should be aligned to city while our model aligned it to Málaga.

8

Cross-lingual Analysis: A Case Study

To investigate the potential causes of misalignment between English and non-English languages, we conduct a case study that qualitatively examines the differences in alignment generated by different systems and languages. Figure 4 illustrates the sentence, "why is it so hard to understand?" with its human translations in German, Spanish, and Italian, and its AMR. In the Italian translation, the subject of the verb is omitted, while in the Spanish translation the focus of the question is modified from asking the reason why something is difficult to understand to asking directly what is difficult to understand, making "qué" the subject. As a consequence, in both cases making it impossible to align "it" with any word in either the Italian or Spanish sentence by Machine Translation Alignment. Table 5 presents the alignments generated for the sentence in each language and with each model in ISI format. Although our model was able to align the node "it" by aligning it with the conjugated verb in the Italian sentence and with the word "qué"
in the Spanish sentence, which serves as the subject, this resulted in an error in our evaluation since the alignment of "it" was not projected in either Italian or Spanish. In addition, we also observed the performance of jamr and tamr, which are rulebased systems, and found that they were only able to align the word "so" in the German translation, as it shares the same lemma in English. In contrast, LEAMR was able to detect more alignments

En ref ours jamr tamr leamr De ref ours jamr tamr leamr Es ref ours jamr tamr leamr It ref ours jamr tamr leamr

Why 1.2 1.2.1 1.2 1.2.1 — — 1.2 Warum 1.2 1.2.1 1.2 1.2.1 — — 1 Qué 1.2 1.2.1 1.1.1 1.2.1 — — 1.1.1 1 Perché 1.2 1.2.1 1.2 1.2.1 — — 1

is — — — — 1.1.1 ist — — — — es — — — — 1.2 é — 1.1.1 — — 1.1

it 1.1.1 1.1.1 1.1.1 1.1.1 1.1.1 das 1.1.1 1.1.1 — — 1.2 — — — — — — — — — — — —

so 1.3 1.3 1.3 1.3 1.3 so 1.3 1.3 1.3 1.3 1.2 tan 1.3 1.2 1.3 — — — cosí 1.3 1.3 — — 1.3

hard 1 1 1 1 1.2 1 schwer 1 1 — — 1.3 díficil 1 1 — — 1.1 difficile 1 1 — — 1.1.1

to — — — — — zu — — — — 1.1.1 de — — — — — da — — — —

understand 1.1 1.1 1.1 1.1 1.1 verstehen 1.1 1.1 — — 1.2.1 entender 1.1 1.1 — — 1.3 capire 1.1 1.1 — — 1.2.1 1.2

?
— — — — 1.2.1 ?
— — — — — ?
— — — — 1.2.1 ?
— — — —

Table 5: Alignments between sentences and graph from Figure 4 across diferent system. "ref" is the reference alignment obtained by Machine Translation Aligment.

due to its requirement to align all nodes to a corresponding word in the target language. However, the alignments generated by LEAMR appeared to be almost entirely random.

9

Conclusion

In this paper, we have presented the first AMR aligner that can scale cross-lingually and demonstrated how cross-attention is closely tied to alignment in AMR Parsing. Our approach outperforms previous aligners in English, being the first to align cross-lingual AMR graphs. We leverage the cross-attention from current AMR parsers, without overhead computation or affecting parsing quality.
Moreover, our approach is more resilient to the lack of handcrafted rules, highlighting its capability as a standard- and language-agnostic aligner, paving the way for further NLP tasks. As a future direction, we aim to conduct an analysis of the attention heads that are not correlated with the alignment information in order to identify the type of information they capture, such as predicate identification, semantic relations, and other factors. Additionally, we plan to investigate how the alignment information is captured across different NLP tasks and languages in the cross-attention mechanism of sequence-tosequence models. Such analysis can provide insights into the inner workings of the models and improve our understanding of how to enhance their performance in cross-lingual settings.

10

Limitations

Despite the promising results achieved by our proposed method, there are certain limitations that need to be noted. Firstly, our approach relies heavily on the use of Transformer models, which can be computationally expensive to train and run. Additionally, the lower performance of our aligner for languages other than English is still a substantial shortcoming, which is discussed in Section 5.2.
Furthermore, our method is not adaptable to nonTransformer architectures, as it relies on the specific properties of Transformer-based models to extract alignment information.
Lastly, our method is based on the assumption that the decoder will attend to those input tokens that are more relevant to predicting the next one.
However, this assumption may not always hold true in practice, which could lead to suboptimal alignments.
In conclusion, while our proposed method presents a promising approach for cross-lingual AMR alignment, it is important to consider the aforementioned limitations when applying our method to real-world scenarios. Future research could focus on addressing these limitations and exploring ways to improve the performance of our aligner for languages other than English.

11

Ethics Statement

While our approach has shown itself to be effective in aligning units and spans in sentences of different languages, it is important to consider the ethical and social implications of our work.
One potential concern is the use of Transformerbased models, which have been shown to perpetuate societal biases present in the data used for training. Our approach relies on the use of these models, and it is therefore crucial to ensure that the data used for training is diverse and unbiased.
Furthermore, the use of cross-attention in our approach could introduce new ways to supervise a model in order to produce harmful or unwanted model predictions. Therefore, it is crucial to consider the ethical implications of any guidance or supervision applied to models and to ensure that any training data used to guide the model is unbiased and does not perpetuate harmful stereotypes or discrimination.
Additionally, it is important to consider the potential impact of our work on under-resourced languages. While our approach has shown to be ef-

fective in aligning units and spans in sentences of different languages, it is important to note that the performance gap for languages other than English still exists. Further research is needed to ensure that our approach is accessible and beneficial for under-resourced languages.

Acknowledgments The authors gratefully acknowledge the support of the European Union’s Horizon 2020 research project Knowledge Graphs at Scale (KnowGraphs) under the Marie Marie Skłodowska-Curie grant agreement No 860801.
The last author gratefully acknowledges the support of the PNRR MUR project PE0000013-FAIR.

References Rafael Anchiêta and Thiago Pardo. 2020. Semantically inspired AMR alignment for the Portuguese language.
In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1595–1600, Online. Association for Computational Linguistics.
Xuefeng Bai, Yulong Chen, and Yue Zhang. 2022.
Graph pre-training for AMR parsing and generation.
In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6001–6015, Dublin, Ireland.
Association for Computational Linguistics.
Laura Banarescu, Claire Bonial, Shu Cai, Madalina Georgescu, Kira Griffitt, Ulf Hermjakob, Kevin Knight, Philipp Koehn, Martha Palmer, and Nathan Schneider. 2013. Abstract Meaning Representation for sembanking. In Proceedings of the 7th Linguistic Annotation Workshop and Interoperability with Discourse, pages 178–186, Sofia, Bulgaria. Association for Computational Linguistics.
Jasmijn Bastings and Katja Filippova. 2020. The elephant in the interpretability room: Why use attention as explanation when we have saliency methods? In Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, pages 149–155, Online. Association for Computational Linguistics.
Michele Bevilacqua, Rexhina Blloshmi, and Roberto Navigli. 2021. One SPRING to rule them both: Symmetric AMR semantic parsing and generation without a complex pipeline. In Proceedings of AAAI.
Adrien Bibal, Rémi Cardon, David Alfter, Rodrigo Souza Wilkens, Xiaoou Wang, Thomas François, and Patrick Watrin. 2022. Is attention explanation? an

introduction to the debate. In Association for Computational Linguistics. Annual Meeting. Conference Proceedings.
Rexhina Blloshmi, Michele Bevilacqua, Edoardo Fabiano, Valentina Caruso, and Roberto Navigli. 2021.
SPRING Goes Online: End-to-End AMR Parsing and Generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 134–142, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
Rexhina Blloshmi, Rocco Tripodi, and Roberto Navigli. 2020. XL-AMR: Enabling cross-lingual AMR parsing with transfer learning techniques. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2487–2500, Online. Association for Computational Linguistics.
Austin Blodgett and Nathan Schneider. 2021. Probabilistic, structure-aware algorithms for improved variety, accuracy, and coverage of AMR alignments.
In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3310–3321, Online. Association for Computational Linguistics.
Claire Bonial, Lucia Donatelli, Mitchell Abrams, Stephanie M. Lukin, Stephen Tratz, Matthew Marge, Ron Artstein, David Traum, and Clare Voss. 2020.
Dialogue-AMR: Abstract Meaning Representation for dialogue. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 684– 695, Marseille, France. European Language Resources Association.
Chi Chen, Maosong Sun, and Yang Liu. 2021. Maskalign: Self-supervised neural word alignment. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4781– 4791, Online. Association for Computational Linguistics.

Zi-Yi Dou and Graham Neubig. 2021. Word alignment by fine-tuning embeddings on parallel corpora. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2112–2128, Online.
Association for Computational Linguistics.
Jeffrey Flanigan, Sam Thomson, Jaime Carbonell, Chris Dyer, and Noah A. Smith. 2014. A discriminative graph-based parser for the Abstract Meaning Representation. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1426–1436, Baltimore, Maryland. Association for Computational Linguistics.
Hardy Hardy and Andreas Vlachos. 2018. Guided neural language generation for abstractive summarization using Abstract Meaning Representation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 768–773, Brussels, Belgium. Association for Computational Linguistics.
Pavan Kapanipathi, Ibrahim Abdelaziz, Srinivas Ravishankar, Salim Roukos, Alexander Gray, Ramón Fernandez Astudillo, Maria Chang, Cristina Cornelio, Saswati Dana, Achille Fokoue, Dinesh Garg, Alfio Gliozzo, Sairam Gurajada, Hima Karanam, Naweed Khan, Dinesh Khandelwal, Young-Suk Lee, Yunyao Li, Francois Luus, Ndivhuwo Makondo, Nandana Mihindukulasooriya, Tahira Naseem, Sumit Neelam, Lucian Popa, Revanth Gangi Reddy, Ryan Riegel, Gaetano Rossiello, Udit Sharma, G P Shrivatsa Bhargav, and Mo Yu. 2021. Leveraging Abstract Meaning Representation for knowledge base question answering. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3884–3894, Online. Association for Computational Linguistics.
Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, and Orion Reblitz-Richardson. 2020.
Captum: A unified and generic model interpretability library for pytorch.

Marco Damonte and Shay B. Cohen. 2018. Crosslingual Abstract Meaning Representation parsing. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1146–1155, New Orleans, Louisiana. Association for Computational Linguistics.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020.
BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Ameet Deshpande and Karthik Narasimhan. 2020.
Guiding attention for self-supervised learning with transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4676– 4686, Online. Association for Computational Linguistics.

Kexin Liao, Logan Lebanoff, and Fei Liu. 2018. Abstract Meaning Representation for multi-document summarization. In Proceedings of the 27th International Conference on Computational Linguistics, pages 1178–1190, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Jungwoo Lim, Dongsuk Oh, Yoonna Jang, Kisu Yang, and Heuiseok Lim. 2020. I know what you asked:
Graph path learning using AMR for commonsense reasoning. In Proceedings of the 28th International Conference on Computational Linguistics, pages 2459–2471, Barcelona, Spain (Online). International Committee on Computational Linguistics.
Yijia Liu, Wanxiang Che, Bo Zheng, Bing Qin, and Ting Liu. 2018. An AMR aligner tuned by transitionbased parser. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2422–2430, Brussels, Belgium. Association for Computational Linguistics.
Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. Transactions of the Association for Computational Linguistics, 8:726–742.
Potsawee Manakul and Mark Gales. 2021. Long-span summarization via local attention and content selection. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6026–6041, Online. Association for Computational Linguistics.
Abelardo Carlos Martínez Lorenzo, Marco Maru, and Roberto Navigli. 2022. Fully-Semantic Parsing and Generation: the BabelNet Meaning Representation.
In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1727–1741, Dublin, Ireland.
Association for Computational Linguistics.
André F. T. Martins and Ramón F. Astudillo. 2016.
From softmax to sparsemax: A sparse model of attention and multi-label classification. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ICML’16, page 1614–1623. JMLR.org.
Roberto Navigli, Rexhina Blloshmi, and Abelardo Carlos Martinez Lorenzo. 2022. BabelNet Meaning Representation: A Fully Semantic Formalism to Overcome Language Barriers. Proceedings of the AAAI Conference on Artificial Intelligence, 36.
K. Elif Oral and Gülşen Eryiğit. 2022. AMR alignment for morphologically-rich and pro-drop languages. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages 143–152, Dublin, Ireland.
Association for Computational Linguistics.
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 2227–2237,

New Orleans, Louisiana. Association for Computational Linguistics.
Nima Pourdamghani, Yang Gao, Ulf Hermjakob, and Kevin Knight. 2014. Aligning English strings with Abstract Meaning Representation graphs. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 425–429, Doha, Qatar. Association for Computational Linguistics.
Sudha Rao, Daniel Marcu, Kevin Knight, and Hal Daumé III. 2017. Biomedical event extraction using Abstract Meaning Representation. In BioNLP 2017, pages 126–135, Vancouver, Canada,. Association for Computational Linguistics.
Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. 2017. Learning important features through propagating activation differences. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17, page 3145–3153.
JMLR.org.
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. 2014. Deep inside convolutional networks:
Visualising image classification models and saliency maps. CoRR, abs/1312.6034.
Linfeng Song, Daniel Gildea, Yue Zhang, Zhiguo Wang, and Jinsong Su. 2019. Semantic neural machine translation using AMR. Transactions of the Association for Computational Linguistics, 7:19–31.
Ekta Sood, Simon Tannert, Philipp Mueller, and Andreas Bulling. 2020. Improving natural language processing tasks with human gaze-guided neural attention. In Advances in Neural Information Processing Systems, volume 33, pages 6327–6341. Curran Associates, Inc.
Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin A. Riedmiller. 2015. Striving for simplicity: The all convolutional net. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Workshop Track Proceedings.
Joe Stacey, Yonatan Belinkov, and Marek Rei. 2021. Supervising model attention with human explanations for robust natural language inference.
Ian Tenney, Dipanjan Das, and Ellie Pavlick. 2019.
BERT rediscovers the classical NLP pipeline. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4593– 4601, Florence, Italy. Association for Computational Linguistics.
Sarah Uhrig, Yoalli Garcia, Juri Opitz, and Anette Frank.
2021. Translate, then parse! a strong baseline for cross-lingual AMR parsing. In Proceedings of the 17th International Conference on Parsing Technologies and the IWPT 2021 Shared Task on Parsing into Enhanced Universal Dependencies (IWPT 2021), pages 58–64, Online. Association for Computational Linguistics.

Shikhar Vashishth, Shyam Upadhyay, Gaurav Singh Tomar, and Manaal Faruqui. 2019. Attention interpretability across nlp tasks. CoRR, abs/1909.11218.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.
Shira Wein and Nathan Schneider. 2021. Classifying divergences in cross-lingual AMR pairs. In Proceedings of The Joint 15th Linguistic Annotation Workshop (LAW) and 3rd Designing Meaning Representations (DMR) Workshop, pages 56–65, Punta Cana, Dominican Republic. Association for Computational Linguistics.

A

The LEAMR standard has some predefined strategies for alignments that were followed during their annotation, as well as fixed in their alignment pipeline along EM. We kept a few of these strategies when extracting the alignment, just those related to the structure of the graph, but not those concerning token matching between the sentence and the graph.
A.1

Zhengxuan Wu, Thanh-Son Nguyen, and Desmond Ong.
2020. Structured self-AttentionWeights encode semantics in sentiment analysis. In Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, pages 255–264, Online. Association for Computational Linguistics.

• Similarly for Named Entities, we align the whole subgraph structure based on its child nodes which indicate its surfaceform. However this leads to some errors as described in Section 7.
• We align node amr-unknown to the question mark if it appears in the sentence.
A.2

• :purpose is aligned with to when in the sentence.
• :ARGX relations are aligned to the same span as the parent node, while :ARGX-of to that of the child, since they share the alignment of the predicate they are connected to.

Matthew D. Zeiler and Rob Fergus. 2014. Visualizing and understanding convolutional networks. In ECCV.

Jiawei Zhou, Tahira Naseem, Ramón Fernandez Astudillo, Young-Suk Lee, Radu Florian, and Salim Roukos. 2021.
Structure-aware fine-tuning of sequence-to-sequence transformers for transitionbased AMR parsing. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6279–6290, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Relations

• For the relation :condition we align it to the word if when it appears in the sentence.

Kayo Yin, Patrick Fernandes, Danish Pruthi, Aditi Chaudhary, André F. T. Martins, and Graham Neubig. 2021. Do context-aware translation models pay the right attention? In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1:
Long Papers), pages 788–801, Online. Association for Computational Linguistics.

Shaolei Zhang and Yang Feng. 2021. Modeling concentrated cross-attention for neural machine translation with Gaussian mixture model. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 1401–1411, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Subgraph

• Nodes have-org-role-91 and have-rel-role-91 follow a fixed structure related to a person ie.
the sentence word enemy is represented as person → have-rel-role-91 → enemy, therefore for such subgraphs we use the alignment from the child node.

Frank Wilcoxon. 1945. Individual comparisons by ranking methods. Biometrics Bulletin, 1(6):80–83.

Song Xu, Haoran Li, Peng Yuan, Youzheng Wu, Xiaodong He, and Bowen Zhou. 2020. Self-attention guided copy mechanism for abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1355–1362, Online. Association for Computational Linguistics.

LEAMR Alignment Rules

• For :mod and :duration we use the alignment from the child node.
• For :domain and :opX we use the alignment from the parent node.

B

Extra Results

B.1

LEAMR Results

We explore the variance with different seeds when guiding cross-attention. Table 1 reports on a single seed selected at random. Table 6 shows the results for five different seeds as well as the average and standard deviation. We observe some variance, especially for those alignment types with fewer

Exact Alignment P R F1 Subgraph Alignment (1707)

Relation Alignment (1263)

Reentrancy Alignment (293)

Duplicate Subgraph Alignment (17)

Partial Alignment P R F1

Spans F1

Run 1 Run 2 Run 3 Run 4 Run 5

94.39 93.79 94.26 94.20 93.81

94.67 93.85 94.32 94.26 94.14

94.53 93.82 94.29 94.23 93.98

96.62 96.22 96.60 96.47 95.81

96.90 96.27 96.66 96.53 96.14

96.76 96.25 96.63 96.50 95.97

96.40 96.05 96.34 96.22 95.73

Average Std

94.09 0.27

94.25 0.30

94.17 0.28

96.34 0.34

96.50 0.30

96.42 0.32

96.15 0.27

Run 1 Run 2 Run 3 Run 4 Run 5

88.03 87.90 88.61 88.39 88.59

88.18 88.36 88.61 88.61 88.44

88.11 88.13 88.61 88.50 88.52

91.08 90.71 91.44 91.02 91.24

91.24 91.18 91.44 91.25 91.08

91.16 90.95 91.44 91.14 91.16

91.87 91.87 91.95 91.66 91.86

Average Std

88.30 0.32

88.44 0.18

88.37 0.28

91.10 0.27

91.24 0.13

91.17 0.17

91.84 0.05

Run 1 Run 2 Run 3 Run 4 Run 5

56.90 56.23 57.24 55.56 55.22

57.09 56.42 57.43 55.74 55.41

57.00 56.32 57.34 55.65 55.31

— — — — —

— — — — —

— — — — —

— — — — —

Average Std

56.23 0.86

56.42 0.86

56.32 0.86

— —

— —

— —

— —

Run 1 Run 2 Run 3 Run 4 Run 5

70.00 65.00 70.00 73.68 70.00

82.35 76.47 82.35 82.35 82.35

75.88 70.27 75.68 77.78 75.68

72.50 67.50 70.00 76.32 70.00

85.29 79.41 82.35 85.29 82.35

78.38 72.97 75.68 80.56 75.68

— — — — —

Average Std

69.74 3.09

81.17 2.63

75.06 2.82

71.26 3.33

82.94 2.46

76.65 2.90

— —

Table 6: Results on the LEAMR alignment for 5 seeds on the guided approach. Column blocks: runs; measures.
Row blocks: alignment types; average and standard deviation (std). Bold is best.

elements; however, average performance is always higher than previous approaches.

C

Alignment Extraction Algorithm

Algorithm 1 shows the procedure for extracting the alignment between spans in the sentence and the semantic units in the graphs, using a matrix that weights Encoder tokens with the Decoder tokens

D

AMR parsing

Since our guided approach was trained with a different loss than the SPRING model, it could influence the performance in the Semantic Parsing task.

Therefore, we also tested our model in the AMR parsing task using the test set of AMR 2.0 and AMR 3.0. Table 7 shows the result, where we can observe how our model preserves the performance on parsing.

SPRING Ours - Guided - ISI Ours - Guided - Leamr

AMR 2.0

AMR 3.0

84.3 84.3 84.3

83.0 83.0 83.0

Table 7: AMR parsing Results.

Algorithm 1 Procedure for extracting the alignment between spans in the sentence and the semantic units in the graphs, using a matrix that weights Encoder tokens with the Decoder tokens.
1: function E XTRACTA LIGNMENTS(encoderT okens, DecoderT okens, scoreM atrix)
2:
alignmentM ap ← dict()
3:
spansList ← S PANS(encoderT okens)
▷ Extract sentence spans as in LEAMR 4:
spanP osM ap ← TOK 2 SPAN(encoderT okens)
▷ Map input tokens to spans 5:
graphP osM ap ← TOK 2 NODE(DecoderT okens)
▷ Map output tokens to graph unit 6:
C OMBINE S UBWORD T OKENS(scoreM atrix)
7:
for DecoderT okenP os, GraphU nit in graphP osM ap do 8:
encoderT okensScores ← scoreM atrix[DecoderT okenP os]
9:
maxScoreP os ← ARGMAX(encoderT okensScores)
10:
alignmentM ap[GraphU nit] ← S ELECT S PAN(spansList, maxScoreP os)
11:
end for 12:
f ixedM atches ← GET F IXED M ATCHES(graphP osM ap)
▷ Look for rule based matches 13:
alignmentM ap ← APPLY F IXED M ATCHES(alignmentM ap, f ixedM atches)
14:
alignments ← F ORMATA LIGNMENT(alignmentM ap)
15:
return alignments 16: end function

E

Hardware

Experiments were performed using a single NVIDIA 3090 GPU with 64GB of RAM and Intel® Core™ i9-10900KF CPU.
Training the model took 13 hours, 30 min per training epoch while evaluating on the validation set took 20 min at the end of each epoch. We selected the best performing epoch based on the SMATCH metric on the validation set.

F

Data

The AMR data used in this paper is licensed under the LDC User Agreement for Non-Members for LDC subscribers, which can be found here. The The Little Prince Corpus can be found here from the Information Science Institute of the University of Southern California.

G

Limitations

Even though our method is an excellent alternative to the current AMR aligner system, which is standard and task-agnostic, we notice some drawbacks when moving to other autoregressive models or languages:
Model In this work, we studied how Cross Attention layers retain alignment information between input and output tokens in auto-regressive models. In Section 5.1, we examined which layers in state-of-the-art AMR parser models based on BART-large best preserve this information. Unfortunately, we cannot guarantee that these layers are

optimal for other auto-regressive models, and so on. As a result, an examination of cross-attention across multiple models should be done before developing the cross-lingual application of this approach.
Sentence Segmentation It is necessary to apply LEAMR’s Spam Segmentation technique to produce the alignment in LEAMR format (Section 3.4). However, this segmentation method has several flaws: i) As stated in Section 7, this approach does not deal appropriately with phrasal verbs and consecutive segments; ii) the algorithm is Englishspecific; it is dependent on English grammar rules that we are unable to project to other languages.
Therefore we cannot extract the LEAMR alignments in a cross-lingual AMR parsing because we lack a segmentation procedure. However, although LEAMR alignment has this constraint, ISI alignment does not require any initial sentence segmentation and may thus be utilized cross-lingually.


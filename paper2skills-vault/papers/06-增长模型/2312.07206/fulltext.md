<!-- 自动生成 by paper2skills-research/scripts/fetch_fulltext.py
     arxiv_id : 2312.07206
     paper_id : 2312.07206
     source   : https://arxiv.org/html/2312.07206v1
     fulltext : 是
     用途     : evidence.md 的 `> 原文:"..."` 引用块的出处核验底本
-->

# A churn prediction dataset from the telecom sector:
a new benchmark for uplift modeling

Théo Verhelst    Denis Mercier    Jeevan Shrestha    Gianluca Bontempi

###### Abstract

Uplift modeling, also known as individual treatment effect (ITE) estimation, is an important approach for data-driven decision making that aims to identify the causal impact of an intervention on individuals. This paper introduces a new benchmark dataset for uplift modeling focused on churn prediction, coming from a telecom company in Belgium, Orange Belgium. Churn, in this context, refers to customers terminating their subscription to the telecom service. This is the first publicly available dataset offering the possibility to evaluate the efficiency of uplift modeling on the churn prediction problem. Moreover, its unique characteristics make it more challenging than the few other public uplift datasets.

## 1 Introduction

Uplift modeling, often called the conditional average treatment effect, has become a crucial tool for data-driven decision making. This modeling technique estimates the effect that a particular intervention or treatment has on individuals, enabling the selection of only those individuals who are likely to have a positive reaction to the action. Although the methodology of uplift modeling has witnessed substantial development and diversification [9], a notable constraint remains: the low number of publicly available datasets designed specifically for uplift modeling. A recent uplift benchmark conducted by [14] listed only 4 public uplift datasets: Criteo [4], Hillstrom [10], Starbucks and Lenta. Furthermore, despite the fact that customer churn is often cited as a common application for uplift modeling, none of these public datasets are concerned with churn. To address this issue, this paper introduces a new churn dataset for uplift modeling, coming from a major telecom company in Belgium, Orange Belgium. This dataset offers researchers and practitioners a new resource to evaluate strategies aimed at reducing churn and increasing customer retention within the telecommunications industry. In Section 2 we present the marketing campaigns that form the basis of this dataset, and we compare the characteristics of our dataset to two other public datasets in Sections 3 and 4. Then, in Sections 5 and 6 we evaluate the performance of three models on these datasets, and finally we conclude by highlighting its potential to foster innovation and progress within the uplift modeling domain in Section 7. The dataset is available on the OpenML platform and the benchmark code is available on GitHub.

## 2 Churn campaigns

The dataset comes from a series of three marketing campaigns conducted between September and December 2020. The campaign pipeline is represented in Fig. 1. During each campaign, the probability of churn for each customer was estimated using a predictive model and the riskiest customers were selected. A subset of these high-risk customers was randomly assigned to the control group, while the remaining customers formed the target group. The list of customers in the target group was shared with a call center tasked with contacting each customer and presenting them with a marketing offer or recommending a new tariff plan based on their individual history. The churn outcome is determined in a two-month window following the campaign, and any subsequent churn is not attributed to this specific campaign. The data of this campaign and the churn outcome are then recorded in the historical database, and the same campaign process is repeated the next month.

*Figure 1: Schematic representation of the churn retention campaign pipeline.*

## 3 Description

*Table 1: Description of the churn dataset and two other uplift datasets.*

| Name | Features | Samples | Control response rate (%) | Target response rate (%) | Treatment rate (%) |

$178$ $11true896$ $3.6$ $3.4$ $75.74$| Churn | | | | | |

$15$ $42true693$ $10.62$ $15.14$ $66.71$| Hillstrom | | | | | |

$12$ $25true309true483$ $4.2$ $4.9$ $84.6$| Criteo | | | | | |

The characteristics of the dataset are summarized in Table 1, as well as the same characteristics of two popular uplift datasets, the Criteo dataset [4], and the Hillstrom dataset [10]. We report the number of features, the number of samples, the response rate in the control and target groups, and the treatment rate.

The churn dataset consists of 11,896 samples, a relatively small number compared to other publicly available uplift datasets. However, it has a larger number of features, totaling 178. These features encompass a diverse range of customer attributes, including demographics (e.g., region of residence, age), usage patterns (e.g., data consumption, number of calls), and subscription details (e.g., price of the tariff plan). The dataset comprises features of various types, including discrete numerical, continuous, and categorical variables, each exhibiting diverse distributions. To ensure privacy and data confidentiality, the dataset is anonymized by using a Principal Component Analysis (PCA) projection of the numerical features, allowing for effective analysis and modeling while protecting sensitive information. Adopting this strategy has proven effective in preserving predictive accuracy while safeguarding privacy in the domain of fraud detection [3]. All categorical features and their levels are anonymized by giving them generic names.

One distinctive aspect of the dataset is the inherent difficulty in accurately predicting the churn outcome. The complex dynamics of churn in the telecom sector make it a challenging task, requiring advanced modeling techniques to capture the underlying patterns and factors influencing customer behavior. Uplift modeling is even more difficult than predicting the outcome probability alone due to the relatively small effect of the treatment. To quantify this aspect, we estimate the mutual information $I(\bm{x};\bm{y}_{t})$ (for $t=0,1$), which represents the difficulty in predicting the binary outcome $\bm{y}_{t}$ from the set of features $\bm{x}$ [2]. It is estimated using the formula

$I(\bm{x};\bm{y}_{t})=H(\bm{y}_{t})-H(\bm{y}_{t}\mid\bm{x})\approx H(\bm{y}_{t})-\frac{1}{N}\sum_{i=1}^{N}H\left(\bm{y}_{t}\mid\bm{x}=x^{(i)}\right)$ | | | |

where the term $H(\bm{y}_{t})$ is estimated from the prior distribution of $\bm{y}_{t}$, and a T-learner uplift model (details of the experimental setup are presented in Section 5) provides the necessary probability estimates to compute

$H(\bm{y}_{t}\mid x)=P(\bm{y}_{t}=0\mid x)\log P(\bm{y}_{t}=0\mid x)+P(\bm{y}_{t}=1\mid x)\log P(\bm{y}_{t}=1\mid x).$ | | | |

*Table 2: Estimates of the mutual information between the features and the outcomes.*

$H(\bm{y}_{0})$ $H(\bm{y}_{1})$ ${\hat{I}(\bm{x};\bm{y}_{0})}$ ${\hat{I}(\bm{x};\bm{y}_{1})}$ ${\frac{\hat{I}(\bm{x};\bm{y}_{0})}{H(\bm{y}_{0})}}$ ${\frac{\hat{I}(\bm{x};\bm{y}_{1})}{H(\bm{y}_{1})}}$| | | | | | | |

$0.16$ $0.15$ $0.0008$ $0.0025$ $0.54$ $1.71$| Churn | | | | | % | % |

$0.34$ $0.43$ $0.0112$ $0.0123$ $3.32$ $2.90$| Hillstrom | | | | | % | % |

$0.16$ $0.20$ $0.0429$ $0.0573$ $24.63$ $29.32$| Criteo | | | | | % | % |

The estimates of the mutual information are given in Table 2. In the last two columns, the mutual information estimate is divided by the entropy of the prior distribution, indicating the proportion of uncertainty of the outcome explained by the features. Our dataset has a low outcome probability similar to that of the Criteo dataset, while also having a very low mutual information, like the Hillstrom dataset. This represents a unique contribution to the uplift ecosystem, where there are no other datasets that come from a small-scale marketing campaign with these particular characteristics.

To better characterize the differences in the outcome distribution among the three datasets, we use the counterfactual point estimator proposed in [15]. It estimates the probabilities of the joint distribution of the potential outcomes $\bm{y}_{0},\bm{y}_{1}$, even though this distribution cannot be observed directly. We use the following formula, also based on the probability estimates given by the T-learner:

$P(\bm{y}_{0}=y_{0},\bm{y}_{1}=y_{1})\approx\frac{1}{N}\sum_{i=1}^{N}P\left(\bm{y}_{0}=y_{0}\mid\bm{x}^{(i)}\right)P\left(\bm{y}_{1}=y_{1}\mid\bm{x}^{(i)}\right).$ | | | | (1) |

for $y_{0},y_{1}\in\{0,1\}$.

*Table 3: Estimated distribution of counterfactuals.*

| Formula | Name (in churn) | Name (in retail) | Churn | Hillstrom | Criteo |

${P(\bm{y}_{0}=0,\bm{y}_{1}=0)}$ $93.1$ $76.0$ $90.8$| | Sure thing | Lost cause | % | % | % |

${P(\bm{y}_{0}=1,\bm{y}_{1}=0)}$ $3.5$ $8.9$ $2.8$| | Persuadable | Do-not-disturb | % | % | % |

${P(\bm{y}_{0}=0,\bm{y}_{1}=1)}$ $3.2$ $13.4$ $4.8$| | Do-not-disturb | Persuadable | % | % | % |

${P(\bm{y}_{0}=1,\bm{y}_{1}=1)}$ $0.1$ $1.8$ $1.7$| | Lost cause | Sure thing | % | % | % |

Estimated probabilities are reported in Table 3, along with the business name associated with the four counterfactuals. Note that, in the churn setting, the outcome $\bm{y}=1$ should be avoided, while in retail, its probability should be maximized. This implies that the two contexts associate different names to the same probabilities. We see that the churn dataset is characterized by a low probability that both potential outcomes are $1$ (lost cause customers, fourth row). This suggests that a negligible number of customers are likely to churn regardless of the targeted marketing action. This differs from the two other datasets, for which this probability is higher. We also observe that the churn dataset is more balanced between positive and negative causal effects (second and third rows), whereas, in both the Hillstrom and Criteo datasets, there is a larger proportion of individuals with a positive causal effect (persuadable customers, third row).

## 4 Randomization

Since the dataset comes from a randomized campaign, the treatment should be independent of the outcomes. To validate this independence, we performed the Classifier 2 Sample Test used in [4]. We trained a classifier to predict the treatment indicator and compared its Hamming loss with the loss distribution obtained under the null hypothesis, sampled by training models to predict random splits. The treatment predictor has a loss of 23.82% (close to the proportion of control samples, 24.26%), which corresponds to a p-value of 0.26 under the null hypothesis. This result is shown in Fig. 2. This indicates that the treatment cannot be predicted based on available features, hence the randomization of treatment assignment can be considered appropriate and unbiased.

*Figure 2: Distribution of the loss under the hypothesis that the treatment is randomized.*

## 5 Benchmark experimental setup

We conducted an experimental benchmark on the churn, Hillstrom and Criteo datasets. We used the classical random forest (RF) model [1], the T-learner uplift model [11], and the uplift random forest [8]. The classical RF model, which we call outcome RF, was trained to predict churn with control samples, therefore, without explicitly considering the individual treatment effect. This serves as a baseline for evaluating the performance of uplift models. This is especially relevant in sectors such as telecoms, where such predictive models are often used instead of uplift models because of their simplicity and sufficient performance. The T-learner used a random forest as base learner. All three models used 100 trees, a maximum depth of 20, and a minimum of 10 samples per leaf. To tackle class imbalance, the EasyEnsemble strategy [13] was applied with 8 folds. For each fold, a new model was trained using all positive samples and an equally sized set of randomly selected negative samples. This approach helps mitigate the adverse impact of class imbalance. The predictions of the eight models were then averaged, effectively reducing the potential biases caused by undersampling the negative samples in individual folds. K-fold cross-validation with $k=3$ was used to obtain training and test splits of each dataset. Finally, the whole experiment was repeated 10 times to obtain a more robust estimation of the performance, as well as an estimation of its variability. The performance of each model was estimated in terms of the area under the uplift curve (AUUC) [7].

## 6 Results

The AUUC is reported in Table 4. To evaluate the impact of the PCA projection, we performed the same experiment on the original, non-anonymized churn dataset. It appears that the performance is only slightly lower on the anonymized dataset than on the original, and, given the high uncertainty in the AUUC, we cannot exclude that this difference is due to random variations in the benchmark sampling. We also observe that the performance of all models is highest on the Hillstrom dataset and lowest on the churn dataset. We attribute this difference to the fact that the outcome is balanced in the Hillstrom dataset, whereas it is unbalanced and more difficult to predict in the churn dataset. Interestingly, the performance of the outcome RF model is consistently the highest, showing that the uplift approach is not always preferable, as discussed in [5, 6].

*Table 4: Mean and standard deviation of the area under the uplift curve (AUUC) in the benchmark.*

| | Churn (%) | Churn (not anonymized) (%) | Hillstrom (%) | Criteo (%) |

$0.26\pm 0$ $0.33\pm 0$ $2.20\pm 0$ $1.01\pm 0$| Outcome RF | .47) | .37) | .32) | .25) |

$0.19\pm 0$ $0.22\pm 0$ $2.19\pm 0$ $0.89\pm 0$| Uplift RF | .37) | .29) | .27) | .23) |

$0.25\pm 0$ $0.33\pm 0$ $2.72\pm 0$ $0.86\pm 0$| T-learner RF | .38) | .39) | .28) | .18) |

The estimator variance likely plays an important role in determining when classical predictive modeling outperforms uplift modeling [5, 6]. To evaluate this possibility, we computed the variance of the predicted probability estimates of each model on each data sample. This was achieved by considering the 10 different predictions generated for each sample during the repeated 3-fold cross-validation procedure. The values reported in Table 5 represent the variance averaged across all samples in the dataset. We observe that the two uplift models in this benchmark suffer from a higher variance than the outcome RF, especially on the Criteo dataset.

*Table 5: Variance of the predictions averaged over the dataset.*

| | Churn | Hillstrom | Criteo |

$2.07\text{\times}{10}^{-3}$ $3.49\text{\times}{10}^{-3}$ $1.22\text{\times}{10}^{-3}$| Outcome RF | | | |

$3.06\text{\times}{10}^{-3}$ $4.33\text{\times}{10}^{-4}$ $2.05\text{\times}{10}^{-3}$| Uplift RF | | | |

$3.78\text{\times}{10}^{-3}$ $7.59\text{\times}{10}^{-3}$ $1.94\text{\times}{10}^{-3}$| T-learner RF | | | |

## 7 Conclusion

The primary objective of this new dataset is to facilitate the evaluation and comparison of uplift modeling techniques, with a focus on customer churn prediction in the telecom sector. More generally, researchers and practitioners can leverage this dataset to develop and benchmark new algorithms, feature engineering approaches, and model evaluation metrics tailored to uplift modeling in difficult settings characterized by a low information rate, a low outcome probability, and a small number of samples. This is especially interesting for smaller companies aiming to initiate personalized marketing campaigns, but which have limited historical data to train uplift models. While large-scale benchmarks such as the Criteo dataset are crucial for evaluating the performance of uplift models on a large sample, small-scale datasets are more representative of some practical applications. This dataset also provides an opportunity to assess other causal inference methods such as counterfactual estimation [12, 15]. Finally, we observed in a benchmark experiment that classical predictive modeling is more effective than uplift modeling [5, 6]. This has also been observed in practice by our industrial partner. In future work, we intend to investigate this question from a theoretical perspective with the hope of gaining a deeper understanding of the critical factors that impact the performance of both predictive and uplift approaches.

## References

- [1] Leo Breiman “Random forests” Publisher: Springer In Machine learning 45.1, 2001, pp. 5–32

- [2] Thomas. Cover and Joy. Thomas “Elements of information theory” Publication Title: Elements of Information Theory John Wiley & Sons, 1991 DOI: 10.1002/0471200611

- [3] Andrea Dal, Olivier Caelen, Reid Johnson and Gianluca Bontempi “Calibrating probability with undersampling for unbalanced classification” In 2015 IEEE Symposium Series on Computational Intelligence IEEE, 2015, pp. 159–166

- [4] Betlei Diemert, Christophe Renaudin and Amini Massih-Reza “A Large Scale Benchmark for Uplift Modeling” In Proceedings of the AdKDD and TargetAd Workshop, KDD, London,United Kingdom, August, 20, 2018 ACM, 2018

- [5] Carlos Fernández-Loria and Foster Provost “Causal Classification: Treatment Effect Estimation vs. Outcome Prediction” In Journal of Machine Learning Research 23.59, 2022, pp. 1–35

- [6] Carlos Fernández-Loria and Foster Provost “Causal decision making and causal effect estimation are not the same… and why it matters” Publisher: INFORMS In INFORMS Journal on Data Science, 2022

- [7] Robin Gubela and Stefan Lessmann “Uplift modeling with value-driven evaluation metrics” Publisher: Elsevier In Decision Support Systems, 2021, pp. 113648

- [8] Leo Guelman, Montserrat Guillén and Ana. Pérez-Marín “Uplift random forests” Publisher: Taylor & Francis In Cybernetics and Systems 46.3-4, 2015, pp. 230–248 DOI: 10.1080/01969722.2015.1012892

- [9] Pierre Gutierrez and Jean-Yves Gérardy “Causal Inference and Uplift Modelling: A Review of the Literature” Series Title: Proceedings of Machine Learning Research In Proceedings of The 3rd International Conference on Predictive Applications and APIs 67 Microsoft NERD, Boston, USA: PMLR, 2016, pp. 1–13 URL: http://proceedings.mlr.press/v67/gutierrez17a.html

- [10] Kevin Hillstrom “The minethatdata e-mail analytics and data mining challenge, 2008” In URL https://blog. minethatdata. com/2008/03/minethatdata-e-mail-analytics-and-data. html, 2008

- [11] Sören. Künzel, Jasjeet. Sekhon, Peter. Bickel and Bin Yu “Metalearners for estimating heterogeneous treatment effects using machine learning” arXiv: 1706.03461 Publisher: National Acad Sciences In Proceedings of the National Academy of Sciences of the United States of America 116.10, 2019, pp. 4156–4165 DOI: 10.1073/pnas.1804597116

- [12] Ang Li and Judea Pearl “Unit Selection Based on Counterfactual Logic” In IJCAI International Joint Conferences on Artificial Intelligence Organization, 2019, pp. 1793–1799 DOI: 10.24963/ijcai.2019/248

- [13] Xu-Ying Liu, Jianxin Wu and Zhi-Hua Zhou “Exploratory undersampling for class-imbalance learning” Publisher: IEEE In IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 39.2, 2009, pp. 539–550 DOI: 10.1109/tsmcb.2008.2007853

- [14] Jannik Rößler and Detlef Schoder “Bridging the gap: A systematic benchmarking of uplift modeling and heterogeneous treatment effects methods” Publisher: SAGE Publications Sage CA: Los Angeles, CA In Journal of Interactive Marketing 57.4, 2022, pp. 629–650

- [15] Théo Verhelst, Denis Mercier, Jeevan Shrestha and Gianluca Bontempi “Partial counterfactual identification and uplift modeling: theoretical results and real-world assessment” In Machine Learning, 2023 DOI: 10.1007/s10994-023-06317-w

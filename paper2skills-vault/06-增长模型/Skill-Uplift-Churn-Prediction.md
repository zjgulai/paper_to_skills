---
title: Skill-Uplift-Churn-Prediction
module: 06-增长模型
topic: 用 Uplift/ITE 而非流失概率本身，识别「干预才留下」的可说服者
status: draft
created: 2026-05-15
updated: 2026-09-13
owner: self
source: ai
venue_tier: preprint
venue_source: arxiv-abs(comments='8 pages, 2 figures, 5 tables, post-proceedings of the ECML PKDD 2023 Workshop on')
paper_id: 2312.07206
paper: A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling
evidence_basis: paper-verbatim
verified_by: quote_check.py (引文逐字核验 VERBATIM) + gate_check.py G2
verified_at: 2026-09-12
related: Skill-Customer-Journey-Prototype.md, Skill-DQN-Purchase-Prediction.md
l1_id: PLN-OPS
l1_plane: 业务运营
l2_id: DOM-05
l2_domain: 品牌与增长
l3_id: DOM-05-105
l3_business: 增量分析
l3_all: 增量分析 / 生命周期触达
l1_l2_l3: 业务运营/品牌与增长/增量分析
---

# Skill Card: Uplift Modeling for Churn Prediction

**论文来源**: A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling  
**arXiv ID**: [2312.07206](https://arxiv.org/abs/2312.07206)  
**发表会议**: ECML PKDD 2023 Workshop（post-proceedings）✅ 已核验 —— 证据在 arXiv 元数据 `Comments`，**不在**底本正文（论文正文通常不写自己的 venue）；见附录  
**适用领域**: 用户流失预测、干预效果评估、精准营销

---

## ① 算法原理

### 核心思想
Uplift Modeling解决的核心问题：**识别哪些用户会因为干预（如优惠券、客服电话）而降低流失概率**。它估计干预对个体的因果效应，从而只挑选出「对干预有正向反应」的那部分人，而不是对所有高危用户统一施策。论文按潜在结果 $(y_0,y_1)$ 的联合分布把用户分为四类：必然转化者（Sure thing）、可说服者（Persuadable）、无法挽回者（Lost cause）、不要打扰者（Do-not-disturb）。

> ⚠️ 论文用的是「可说服者 / 必然转化者 / 无法挽回者 / 不要打扰者」这四个名字，**没有**「Sleeping Dogs」，该词在底本中零命中。四类标签与流失语境的对应见 §3 表 3。

### 数学直觉
**个体干预效应 (ITE)**：
$$\tau(x) = P(Y=1|T=1,X=x) - P(Y=1|T=0,X=x)$$

其中：
- Y=1 表示流失，Y=0 表示留存
- T=1 表示接受干预，T=0 表示未接受干预
- 负的ITE表示干预降低流失概率（好效果）

### 论文基准里实际比较的三个模型（§5）
论文没有提出新算法，而是**在一个新数据集上比较了三个既有模型**，这也是这张卡唯一可核验的「哪种模型更优」的来源：

| 模型（论文原称） | 论文的定义 |
|---|---|
| **outcome RF** | 经典随机森林，**只用对照组样本**预测流失，「without explicitly considering the individual treatment effect」，作为评估 uplift 模型的**基线** |
| **uplift RF** | uplift random forest |
| **T-learner RF** | T-learner uplift model，base learner 用随机森林 |

三者统一配置：100 trees、max depth 20、min 10 samples per leaf；类别不平衡用 EasyEnsemble（8 folds）；K-fold 交叉验证 $k=3$；整个实验重复 10 次。评价指标是 **AUUC（area under the uplift curve）**。

> ⚠️ 旧版此处还有 SUTVA / 无混淆性 / 正向性三条「关键假设」（原文写作 `SUTVA`、`unconfoundedness`、`positivity`）与 T-Learner 双模型公式、X-Learner 三阶段流程。底本中 `SUTVA`、`unconfoundedness`、`positivity`、`X-Learner`、`S-Learner` **全部零命中**，论文未列出建模假设清单、基准里也没有 X/S-Learner，故一并删除。旧版的模型对比结论与论文相反，详见 ①b。

---

## ①b 反例与适用边界

> **这一节是本卡最重要的一节。** 论文的核心实测结论是**负结果**：在这份流失基准上，普通流失预测反而最好，uplift 并不必然更优。本卡旧版把它写成了正结果，已改正。

- **什么时候不要用 uplift（论文的直接反例）**
  1. **只看「uplift 一定比普通预测模型好」这一条就上马**：论文 §6 实测 **outcome RF 的 AUUC 始终最高** —— 「the performance of the outcome RF model is consistently the highest, showing that the uplift approach is not always preferable」。§7 结论段再次确认「classical predictive modeling is more effective than uplift modeling」，并说明**其产业合作方在实践中也观察到同样现象**。故 uplift 只在「普通预测模型作为基线已被打败」之后才值得投入。
  2. **小样本 + 低信息量的场景**：本文数据集只有 11,896 样本 / 178 特征，互信息极低（论文表 2 的估计值显示，对照组的特征-结果互信息仅约 0.0008，远低于 Criteo 数据集的 0.0429），论文自述这是「a low information rate, a low outcome probability, and a small number of samples」的困难设定。
  3. **干预效应本身微弱时**：论文明确「Uplift modeling is even more difficult than predicting the outcome probability alone due to the relatively small effect of the treatment」。
  4. **流失在对照组也极少发生**：该数据中「无论是否干预都会流失」的 Lost cause 比例极低（表 3 第四行 $0.1\%$），论文称「a negligible number of customers are likely to churn regardless of the targeted marketing action」。这种情况下 uplift 的可操作空间本就窄。
  5. **把四象限当作可直接触达的名单**：四象限来自**估计**出的反事实联合分布（表 3），论文未给出任何象限归属的准确率或校准证据，不能直接当成发券名单。
- **已知的失败模式**
  - **估计量方差压过效应**：论文表 5 实测，两个 uplift 模型（uplift RF / T-learner RF）的预测方差都**高于 outcome RF**。论文把方差列为「classical predictive modeling outperforms uplift modeling」的可能原因之一（§6 与参考文献 [5,6]），即效应小 + 方差大时，uplift 的排序会被噪声吃掉。
  - **把「uplift 排名第一」当成论文结论**：论文的基准里根本没有该模型。**任何在本卡上声称「某 uplift 变体显著优于其他变体」的说法都超出了底本。**
  - **跨数据集外推胜负关系**：论文表 4 显示，胜负关系随数据集而变 —— 在 Hillstrom 上 T-learner RF 的 AUUC 最高（$2.72\%$），而在 churn 与 Criteo 上 outcome RF 最高。**不能把 churn 上的结论直接搬到别的域。**
- **论文自己承认的局限**
  1. 数据只来自比利时一家电信运营商（Orange Belgium）的三次营销活动（2020 年 9–12 月），**跨行业、跨企业迁移性未验证**；
  2. 数值特征经 PCA 投影匿名化，论文承认匿名化后性能略降，但因 AUUC 不确定性高，**无法排除差异来自随机波动**；
  3. 流失结果只在活动后**两个月窗口**内判定，更晚的流失不归因于该次活动；
  4. 作者把「普通预测模型为何更好」列为**未来工作**：「we intend to investigate this question from a theoretical perspective」—— 即本文**没有**给出该负结果的理论解释。

---

## ② 母婴出海应用案例

### 场景1：高危用户挽回优惠券精准发放

**业务问题**  
母婴出海电商面临用户生命周期短（孩子长大需求消失）、获客成本高的挑战。现有流失预警模型能识别高危用户，但对所有高危用户发券成本高、效果差。部分用户"给券才留"，部分"给券也不留"，还有部分"不给券也会留"。统一发券策略ROI低。

**数据要求**
- 用户特征：在网时长、月消费金额、累计消费、客服通话次数、购买产品数量
- 历史干预数据：是否发放优惠券、是否进行客服回访
- 流失标签：30天内是否流失（**卡片自定的电商业务口径**；论文的观察窗是 activity 后 **two-month window** 的流失，本卡此处的 30 天不代表论文口径）
- 样本量：必须有**足够的历史随机对照实验数据**，即同一时期内明确划分了处理组与对照组、且两组都记录了流失结果。论文的自述弱点是样本小 + 信息率低（11,896 样本 / 178 特征），**样本量下限论文未给出**，需按自己实验的历史实验量评估。

**预期产出**
- 每个用户的Uplift分数：干预对降低流失的概率
- **同时产出普通流失预测（outcome RF 口径）分数作为对照基线** —— 论文实测在流失数据集上该基线反而最优（见 ①b），不并列基线就无法判断 uplift 是否值得上线
- 四象限分群（论文四类反事实，§3 表 3）：
  - 可说服者（Persuadable）：干预对其有正向因果效应，是值得触达的对象
  - 必然转化者（Sure thing）：不给干预也会留存，无需发券
  - 无法挽回者（Lost cause）：给不给干预都会流失，发券无效，节省成本
  - 不要打扰者（Do-not-disturb）：干预反而有害，应避免触达
- 分群触达策略：仅对"可说服者"发放高价值优惠券，且**上线前须与上一行的普通预测基线做同期 A/B 对比**

> ⚠️ 旧版此处的 `Uplift>0.1` / `0<Uplift<0.1` / `Uplift≈0` / `Uplift<0` 是**卡片自定的分群阈值，论文中零命中、无依据**，已删除；建议按自己的实验数据用 AUUC 曲线选阈值。旧版第四类的名字 `Sleeping Dogs` 在底本中同样零命中，已替换为论文自己的第四类 `Do-not-disturb`（§3 表 3）。

**业务价值**
- 优惠券成本降低30-40%（假设月发券成本10万，节省3-4万）
- 挽回率提升15-25%（将预算集中在高响应用户）
- 避免"优惠券依赖"：不给"必然转化者"发券，培养正常消费习惯

---

### 场景2：结合AIPL-VOC标签的精细化干预

**业务问题**  
在AIPL-VOC标签体系下，不同生命周期阶段和情感状态的用户对干预的反应不同。认知期高投诉用户可能需要客服介入，忠诚期满意用户可能反感营销打扰。需要量化不同标签组合对干预的敏感度。

**数据要求**
- AIPL标签：Awareness/Interest/Purchase/Loyalty（来自STAN模型）
- VOC标签：高满意/价格敏感/质量关注/服务抱怨/中性（来自CSK模型）
- 干预历史：优惠券、客服电话、APP推送等
- 流失结果：30天流失标签

**预期产出**
- 分群Uplift矩阵：
  | AIPL阶段 | VOC标签 | Uplift | 建议策略 |
  |----------|---------|--------|----------|
  | Interest | 价格敏感 | 高 | 发放优惠券 |
  | Loyalty | 高满意 | 低 | 减少打扰 |
  | Awareness | 服务抱怨 | 负 | 客服介入而非营销 |
- 个性化干预策略：根据AIPL+VOC标签自动选择干预方式
- **必须提交一份「特征分层 vs 统一策略」的基线对比**：矩阵里的「高/低/负」只有在**打过普通流失预测基线**之后才有决策意义 —— 论文的负结果（①b）说明，不做这一步就可能把预算投向一个并不优于普通预测模型的方案

**业务价值**
- 营销ROI提升2-3倍（精准匹配干预方式）
- 用户满意度提升（减少无效打扰）
- 建立"标签→Uplift→策略"的自动化运营闭环

> ⚠️ 上述 2–3 倍 ROI 为**作者业务估算，论文无对应量**；且在本卡场景下必须先通过 ①b 的基线检验，否则该倍数不成立。

---

## ③ 代码模板

代码位置: `paper2skills-code/growth_model/uplift_churn_prediction/model.py`

核心组件：
1. **TLearner**: 分别训练处理组和对照组模型，相减得到ITE
2. **SLearner**: 将干预作为特征输入单一模型
3. **XLearner**: 结合两种方法优势，用回归模型预测ITE
4. **UpliftMetrics**: Qini曲线和AUUC评估指标
5. **CustomerUpliftAnalyzer**: 业务分析器，输出四象限分群和策略建议

> ⚠️ **本节与底本的关系必须说清**：上面 1–3 的类名是本卡自带代码包的组件，**不是论文的基准模型**。底本 §5 的基准只有 **outcome RF / uplift RF / T-learner RF** 三个（后见 ① 节表格），其中只有 T-learner 与本节组件同名同源；`SLearner` 与 `XLearner` 在底本中零命中，`Qini` 亦然（论文评价指标是 AUUC）。本节代码块未改动（按本次任务范围），但**读者不应据此认为论文验证过 S/X-Learner 或 Qini**。

运行测试:
```bash
cd paper2skills-code/growth_model/uplift_churn_prediction
python3 model.py
```

---

## ④ 技能关联

### 前置技能
- **Skill-Customer-Churn-Prediction**: 掌握传统流失预测方法，理解用户行为特征
- **Skill-Uplift-Modeling**: 理解Uplift Modeling基础概念（该卡覆盖各类 meta-learner；注意**本卡底本 2312.07206 并没有比较 T/S/X-Learner**，见 ① 与 ①b）
- **Skill-A-B-Test-Design**: 需要A/B测试数据作为训练样本

### 延伸技能
- **Skill-Causal-Forest**: 基于树的非参数ITE估计，适合大规模数据
- **Skill-Doubly-Robust-Estimation**: 结合倾向评分的双重稳健估计
- **Skill-Reinforcement-Learning**: 动态调整干预策略

### 可组合技能
| 组合技能 | 组合效果 | 应用场景 |
|----------|----------|----------|
| Uplift + STAN生命周期 | 分阶段Uplift分析 | 不同生命周期阶段干预效果差异 |
| Uplift + CSK情感聚类 | 情感分群Uplift | 价格敏感型 vs 服务抱怨型的干预差异 |
| Uplift + 旅程原型 | 原型级Uplift | 不同行为模式用户的干预敏感度 |
| Uplift + DQN购买预测 | Uplift+意图联合建模 | 综合评估干预价值和购买概率 |

---

## ⑤ 商业价值评估

### ROI预估

**实施成本**（一次性投入）：
- 模型开发：1-2周（1名算法工程师）
- 数据pipeline搭建：1周（1名数据工程师）
- 与现有优惠券系统集成：1周
- **总计成本**：约20-30人天

**预期收益**（年化）：
- 优惠券成本降低30% → 假设月成本10万，年节省 **36万元**
- 挽回率提升带来的LTV增长 → 估计 **20-30万元**
- **年化ROI**：(56-66万) / (人力成本约10万) = **5-6倍**

### 实施难度
2/5星

**依据**：
- 论文给出了可直接复用的基准实验设置（三个模型、100 trees / depth 20 / EasyEnsemble / $k=3$、重复 10 次），复现路径清晰
- 依赖 A/B测试数据，需确保历史实验数据质量
- 与现有流失预测系统整合有一定工程复杂度

### 优先级评分
4/5星

**依据**：
- **业务价值明确**：直接降低优惠券成本，效果可量化
- **与现有体系契合**：可与AIPL-VOC标签体系深度结合
- **前提条件（必读）**：论文在流失数据集上的实测是 **outcome RF 的 AUUC 始终最高**，且 §7 称其产业合作方在实践中也观察到普通预测模型更有效（①b）。**故本卡的优先级只在「普通预测基线在同一数据上被 uplift 打败」后才成立**；未做该对比前不应按 4 星投入。
- **实施周期短**：2-3周可完成MVP

### 实施建议
1. **MVP阶段**（2周）：用历史优惠券实验数据训练模型，输出四象限分群报告，**并同期跑通普通流失预测基线（outcome RF 口径）做对比** —— 未过这一关就不进入试点
2. **试点阶段**（2周）：选择"兴趣期+价格敏感"用户群体进行A/B测试
3. **全面推广**（1个月）：集成到优惠券发放系统，实现自动化分群触达

---

## ⑥ 原文引用

> 原文："This is the first publicly available dataset offering the possibility to evaluate the efficiency of uplift modeling on the churn prediction problem."
> 出处：2312.07206 §Abstract（PDF 第 1 页）

> 原文："Uplift modeling, also known as individual treatment effect (ITE) estimation, is an important approach for data-driven decision making that aims to identify the causal impact of an intervention on individuals."
> 出处：2312.07206 §Abstract（PDF 第 1 页）

> 原文："Uplift modeling, often called the conditional average treatment effect, has become a crucial tool for data-driven decision making. This modeling technique estimates the effect that a particular intervention or treatment has on individuals, enabling the selection of only those individuals who are likely to have a positive reaction to the action."
> 出处：2312.07206 §1 Introduction（PDF 第 1 页）

> 原文："Churn, in this context, refers to customers terminating their subscription to the telecom service."
> 出处：2312.07206 §Abstract（PDF 第 1 页）

> 原文："To address this issue, this paper introduces a new churn dataset for uplift modeling, coming from a major telecom company in Belgium, Orange Belgium."
> 出处：2312.07206 §1 Introduction（PDF 第 1 页）

> 原文："A subset of these high-risk customers was randomly assigned to the control group, while the remaining customers formed the target group."
> 出处：2312.07206 §2 Churn campaigns（PDF 第 1 页）

> 原文："The churn outcome is determined in a two-month window following the campaign, and any subsequent churn is not attributed to this specific campaign."
> 出处：2312.07206 §2 Churn campaigns（PDF 第 1 页）——论文的观察窗是 **two-month**，本卡 ② 写的「30 天内是否流失」是卡片自定的业务口径

> 原文："The churn dataset consists of 11,896 samples, a relatively small number compared to other publicly available uplift datasets. However, it has a larger number of features, totaling 178."
> 出处：2312.07206 §3 Description（PDF 第 2 页）

> 原文："The treatment predictor has a loss of 23.82% (close to the proportion of control samples, 24.26%), which corresponds to a p-value of 0.26 under the null hypothesis."
> 出处：2312.07206 §4 Randomization（PDF 第 3 页）——随机化校验（Classifier 2 Sample Test）

> 原文："Estimated probabilities are reported in Table 3, along with the business name associated with the four counterfactuals."
> 出处：2312.07206 §3 表 3（PDF 第 3 页）——「必然转化者 / 可说服者 / 无法挽回者 / 不要打扰者」四象限即出自此表

> 原文："This suggests that a negligible number of customers are likely to churn regardless of the targeted marketing action."
> 出处：2312.07206 §3 表 3 后的分析（PDF 第 3 页）

> 原文："We also observe that the churn dataset is more balanced between positive and negative causal effects (second and third rows), whereas, in both the Hillstrom and Criteo datasets, there is a larger proportion of individuals with a positive causal effect (persuadable customers, third row)."
> 出处：2312.07206 §3（PDF 第 3 页）

> 原文："We used the classical random forest (RF) model [1], the T-learner uplift model [11], and the uplift random forest [8]."
> 出处：2312.07206 §5 Benchmark experimental setup（PDF 第 4 页）——论文基准里**没有 X-Learner，也没有 S-Learner**

> 原文："The performance of each model was estimated in terms of the area under the uplift curve (AUUC) [7]."
> 出处：2312.07206 §5 Benchmark experimental setup（PDF 第 4 页）——评价指标是 **AUUC**，不是 Qini

> 原文："All three models used 100 trees, a maximum depth of 20, and a minimum of 10 samples per leaf."
> 出处：2312.07206 §5 Benchmark experimental setup（PDF 第 4 页）——这是本卡 ① 三个模型统一配置（100 trees / depth 20 / min 10 samples per leaf）的出处

> 原文："Finally, the whole experiment was repeated 10 times to obtain a more robust estimation of the performance, as well as an estimation of its variability."
> 出处：2312.07206 §5 Benchmark experimental setup（PDF 第 4 页）

> 原文："The dataset comes from a series of three marketing campaigns conducted between September and December 2020."
> 出处：2312.07206 §2 Churn campaigns（PDF 第 1 页）——本卡 ①b 引用的 2020 年 9–12 月三次活动的出处

> 原文："More generally, researchers and practitioners can leverage this dataset to develop and benchmark new algorithms, feature engineering approaches, and model evaluation metrics tailored to uplift modeling in difficult settings characterized by a low information rate, a low outcome probability, and a small number of samples."
> 出处：2312.07206 §7 Conclusion（PDF 第 5 页）——本卡 ①b 第 2 条「小样本 + 低信息量」的直接出处

> 原文："Interestingly, the performance of the outcome RF model is consistently the highest, showing that the uplift approach is not always preferable, as discussed in [5, 6]."
> 出处：2312.07206 §6 Results（PDF 第 4 页）

> 原文："Finally, we observed in a benchmark experiment that classical predictive modeling is more effective than uplift modeling [5, 6]. This has also been observed in practice by our industrial partner."
> 出处：2312.07206 §7 Conclusion（PDF 第 5 页）——**负结果**：本文的基准里普通流失预测（outcome RF）反而最好；仅凭本卡断言「干预对谁有效」会与本文结论相反

> **口径提示（不改正文，仅记录）**：本卡 ②⑤ 的全部数字（节省比例区间、挽回率区间、月成本、年节省金额、倍数、人天/周数）均为**业务假设代入**，论文没有对应数字；
> 论文可核验的对应量只有 11,896 样本 / 178 特征、随机化校验、表 3 的四类反事实分布，以及 §6–§7 的**负结果**。

---

## 附录：论文核心信息

| 项目 | 内容 |
|------|------|
| 论文标题 | A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling |
| 作者 | Théo Verhelst、Denis Mercier、Jeevan Shrestha、Gianluca Bontempi |
| 发表 | **ECML PKDD 2023 Workshop**（post-proceedings, *Uplift Modeling and Causal Machine Learning for Operational Decision Making*）—— ✅ **已核验，但证据不在底本里**：arXiv 元数据 `Comments` 逐字为 `8 pages, 2 figures, 5 tables, post-proceedings of the ECML PKDD 2023 Workshop on Uplift Modeling and Causal Machine Learning for Operational Decision Making`（2026-09-13 复核）。⚠️ 底本 v1 正文 `ECML`/`PKDD` **零命中是正常的** —— 论文正文通常不写自己的 venue，所以 **venue 声明不得用底本判定**，须用 arXiv 元数据 / 出版方 DOI |
| arXiv | 2312.07206 (v1) |
| 核心贡献 | 提供电信行业（Orange Belgium，2020 年 9–12 月三次营销活动）的公开 churn uplift 基准数据集：**11,896 样本 / 178 特征**，数值特征经 PCA 匿名化；论文另给出三模型基准实验设置 |
| 实验结果 | **outcome RF 的 AUUC 在流失数据集上始终最高**（§6）；Hillstrom 上 T-learner RF 最高（$2.72\%$）；两个 uplift 模型的预测方差均高于 outcome RF（表 5）。**「哪种模型更优」随数据集而变，且论文结论是 uplift 并不必然更优** |

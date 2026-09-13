---
title: Skill-User-Lifecycle-STAN
module: 06-增长模型
topic: 用可学习的用户生命周期阶段表示自适应调权，改进多任务推荐
status: draft
created: 2026-05-15
updated: 2026-09-13
owner: self
source: ai
paper_id: 2306.12232
paper: "STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation"
evidence_basis: paper-verbatim
verified_by: quote_check.py (引文逐字核验 VERBATIM) + gate_check.py G2
verified_at: 2026-09-12
related: Skill-Customer-Journey-Prototype.md, Skill-Uplift-Churn-Prediction.md
---

# Skill Card: STAN 用户生命周期自适应建模

**论文来源**: STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation  
**arXiv ID**: [2306.12232](https://arxiv.org/abs/2306.12232)  
**发表会议**: RecSys 2023  
**适用领域**: 用户增长、生命周期运营、生命周期阶段标签（论文阶段名为 New / Wander / Stick / Loyal；**AIPL 为本项目映射，非论文概念**）

---

## ① 算法原理

### 核心思想
用户在不同生命周期阶段对推荐任务有不同偏好，且这些偏好随时间变化。STAN 先从用户行为中推断其对各任务的偏好，再以隐式的阶段表示（latent stage representation）为各任务损失加权，从而自适应地调整多任务学习的关注点。

### 数学直觉
- **偏好学习**（§3.2）：先对用户特征矩阵做自注意力得到 U′_i（式 6/7），再由第二个注意力单元得到任务特定用户表示 s_k^i = U′_i ⊙ Softmax(W_k · U′_i)（式 8），经单层 MLP_k 与 Sigmoid 输出该任务的预测值 ỹ_i^k（式 9）。
- **隐式阶段表示**（§3.3）：ỹ_k 被假设服从 Beta 分布，即 ỹ_k ∼ Beta(α_k, β_k)（式 12）。α_k 是"用户执行了任务 k 对应动作"的试验次数、β_k 是"没有执行"的次数；二者在训练中按 Algorithm 1 累积更新（α_k += ỹ·c_u，β_k += (1−ỹ)·c_u），再从该分布**采样**得到 γ_k —— 论文称其为该任务的 latent stage representation（Algorithm 1 的 Output 即如此命名）。
- **阶段自适应**（§3.4）：γ_k 只用于给任务 k 的预测损失加权，总目标为 L = Σ_k ( γ_k · L_tk + L_sk )（式 13）；用户在某任务上兴趣越低，该任务损失在反向传播中获得的关注越少（§3.3）。

**⚠️ 与门控网络的区分**：论文中的 gating network（g_B）**不属于阶段模块**，而在 §3.1 的 backbone 多任务预测网络里，负责对 shared / task-specific 专家输出做加权选择（式 1–3；§3.1 自述该 backbone 沿用 PLE [20] 的思路——限制共享专家的使用）。阶段表示不是门控网络的输出，而是上一条 Beta 采样的结果。

**直观解释**：α_k / β_k 随样本累积使偏好估计更可靠；样本少时 Beta 先验起平滑作用，避免仅凭少量行为就判定偏好（对应 §4.2 的 STAN w/o Beta 消融：去掉 Beta 后性能波动更大）。

### 关键假设
1. 用户对任务的偏好会随生命周期推进而变化，需被动态跟踪（论文 §4.4.2 观察到同一批用户一个月内从 New 转移到 Wander / Stick）；短期行为抖动由伪标签取历史均值来平滑（式 10）
2. 同一阶段用户对不同任务的偏好相似
3. 生命周期阶段可以从用户行为中推断（§3.2：论文自述"we can only infer users' stage information from their behaviors"）

---

## ② 母婴出海应用案例

### 场景1：生命周期阶段标签驱动的精准触达（AIPL 为本项目映射，非论文概念）

**业务问题**  
母婴出海电商用户决策周期长（孕期到育儿多阶段），不同阶段用户需求差异巨大。新客需要品牌认知教育，老客需要复购推荐。统一策略导致新客流失快、老客触达疲劳。

**数据要求**
- 用户行为日志：页面浏览、商品点击、加购、下单、复购（最近90天）
- 商品属性：品类（奶粉/尿布/辅食）、阶段（0-6月/6-12月/1-3岁）、品牌
- 时间特征：距首次访问天数、距上次购买天数

**预期产出**
- 每个用户的生命周期阶段标签（**论文原名 New / Wander / Stick / Loyal**：New 为低 CTR 新客；Wander 停留时长短；Stick 高 CTR、高停留但低 CVR，只逛不买；Loyal 稳定点击、停留与购买）
- 阶段转移概率矩阵（论文 §4.4.2 观察到用户会在阶段间转移，如 New → Wander / Stick）
- 分阶段推荐策略配置（各阶段任务权重自动调整）

**业务价值**
- 新客30日留存率提升 3-5%（**本项目假设**；论文线上 A/B 的对应口径是**人均停留时长 +3.05%**，论文未给留存率数字）
- 老客转化率提升 0.8-1.2%（参考论文线上 A/B：CVR +0.88%）
- 触达疲劳投诉下降 15-20%（本项目假设）

---

### 场景2：跨品类生命周期价值预测

**业务问题**  
母婴用户生命周期天然受限（孩子长大需求消失）。需要在有限窗口内最大化LTV。传统LTV预测忽略用户所处阶段，导致对"早期高潜力用户"和"晚期衰退用户"的预算分配失衡。

**数据要求**
- 用户跨品类购买序列（奶粉→尿布→辅食→玩具→童装）
- 品类间转移时间间隔
- 营销活动触达记录

**预期产出**
- 用户当前所处生命周期阶段及预计剩余价值
- 最优触达时机预测（在兴趣高峰期推送优惠券）
- 跨品类推荐优先级（根据阶段预测下一步需求品类）

**业务价值**
- 营销ROI提升 10-15%（避免对衰退用户浪费预算）
- 品类渗透率提升 5-8%（精准预测下一需求品类）

---

## ③ 代码模板

代码位置: `paper2skills-code/growth_model/user_lifecycle_stan/model.py`

核心组件（括注说明各组件与论文机制的对应关系）：
1. **LifecycleStageEncoder**: 阶段编码器，用自注意力从行为序列学表示（**实现侧差异**：论文 §3.2 的注意力作用在用户特征矩阵 U_i 上，作用对象不是行为序列）
2. **TaskAdaptiveHead**: 任务自适应头，输出各任务的 softmax 权重（**实现侧差异**：论文 §3.4 的阶段自适应发生在**损失层**——用阶段表示 γ_k 加权 L_tk，见式 13，而不是给任务头加权）
3. **STANLifecycleModel**: 完整模型整合
4. **AIPLLabelSystem**: 标签体系实现，将模型输出映射到业务标签（**命名警示**：论文的阶段名是 New / Wander / Stick / Loyal，**AIPL 为本项目映射，非论文概念**）

运行测试:
```bash
cd paper2skills-code/growth_model/user_lifecycle_stan
python3 model.py
```

---

## ④ 技能关联

### 前置技能
- **Skill-Customer-Churn-Prediction**: 理解用户流失预测基础，掌握用户行为序列特征工程
- **Skill-Time-Series-Forecasting**: 理解时间序列建模方法，用户行为具有时序特性
- **Skill-VOC-Aspect-Extraction**: VOC情感分析结果可作为生命周期阶段的辅助特征

### 延伸技能
- **Skill-LTV-Prediction**: 生命周期阶段可作为LTV预测的重要特征
- **Skill-Multi-Armed-Bandit**: 结合STAN阶段输出，实现分阶段的个性化定价/优惠券策略
- **Skill-Causal-Uplift-Modeling**: 评估不同生命周期阶段的干预效果

### 可组合技能
| 组合技能 | 组合效果 | 应用场景 |
|----------|----------|----------|
| STAN + VOC情感分析 | 生命周期标签+情绪标签 | 精准识别需要客服介入的高价值用户 |
| STAN + 推荐系统 | 分阶段推荐策略 | 论文线上 A/B 实测 STAN 自身 CTR +3.94%；组合后的增益论文未给 |
| STAN + A/B测试 | 分阶段实验设计 | 避免新客和老客的策略冲突 |

---

## ⑤ 商业价值评估

### ROI预估

**实施成本**（一次性投入）：
- 模型开发：2-3周（1名算法工程师）
- 数据pipeline搭建：1-2周（1名数据工程师）
- 与现有推荐系统集成：1周
- **总计成本**：约30-40人天

**预期收益**（年化）：
- 转化率提升 0.88% → GMV 1亿 × 0.88% × 20%毛利 = **176万元**
- 停留时间提升 3.05% → 间接价值 **50-100万元**
- **年化ROI**：(226-276万) / (人力成本约15万) = **15-18倍**

### 实施难度
3/5星

**依据**：
- 论文已有开源思想，实现路径清晰
- 依赖用户行为序列数据，需确保埋点完整
- 需要与现有推荐系统集成，有一定工程复杂度

### 优先级评分
5/5星

**依据**：
- **战略契合度**：生命周期阶段运营是母婴出海核心运营框架（本项目按 AIPL 落地，AIPL 非论文概念），STAN 提供阶段表示的数据支撑
- **业务价值量化明确**：论文A/B测试数据可靠
- **技术成熟度**：RecSys顶会验证
- **复用性高**：一套模型可支撑 New / Wander / Stick / Loyal 全阶段运营（本项目按 AIPL 认知/兴趣/购买/忠诚 做映射落地）

### 实施建议
1. **MVP阶段**（2周）：用历史数据离线训练模型，输出用户阶段分布报告
2. **试点阶段**（2周）：选择 Wander → Loyal 的转化场景做 A/B 测试（对应本项目 AIPL 的"兴趣期→购买期"）
3. **全面推广**（1个月）：全量上线，集成到推荐系统和营销自动化平台

---

## ⑥ 原文引用

> 原文："Existing methods generally formulate the optimization of these evaluation metrics as a multitask learning problem, but often overlook the fact that user preferences for different tasks are personalized and change over time. Identifying and tracking the evolution of user preferences can lead to better user retention."
> 出处：2306.12232 §Abstract（PDF 第 1 页）

> 原文："To address this issue, we introduce the concept of “user lifecycle,” consisting of multiple stages characterized by users’ varying preferences for different tasks."
> 出处：2306.12232 §Abstract（PDF 第 1 页）

> 原文："We propose a novel Stage-Adaptive Network (STAN) framework for modeling user lifecycle stages. STAN first identifies latent user lifecycle stages based on learned user preferences, and then employs the stage representation to enhance multi-task learning performance."
> 出处：2306.12232 §Abstract（PDF 第 1 页）

> 原文："It dynamically adjusts its focus on tasks according to the user’s stage, which is modeled by the representation of their preferences."
> 出处：2306.12232 §1 Introduction（PDF 第 2 页）

> 原文："we introduce the latent user stage representation module to adjust 𝑦˜𝑘 and generate a reliable preference"
> 出处：2306.12232 §3 阶段自适应模块（PDF 第 8 页）

> 原文："Note that the four discrete stages in Fig. 1 are merely examples for visualization purposes, and the actual stages in our model are represented by continuous vectors."
> 出处：2306.12232 §1 Introduction（PDF 第 2 页）——**边界**：论文的阶段是连续向量；本卡 ② 展示的离散标签用的是**论文自己**的 New / Wander / Stick / Loyal 命名（仅作业务解释），**AIPL 为本项目映射，非论文概念**

> 原文："We collected one month of user behavior data from an e-commerce platform, which records users’ clicks, staytime, and purchase actions."
> 出处：2306.12232 §2.1 数据分析（PDF 第 3 页）

> 原文："We randomly selected 50,000 users’ actions over three days for data analysis."
> 出处：2306.12232 §2.1 数据分析（PDF 第 3 页）

> 原文："The stages are named New, Wander, Stick and Loyal, as shown in Fig. 3. Note that one user could not belong to multiple stages."
> 出处：2306.12232 §2.1 数据分析（PDF 第 3 页）——论文自己的阶段命名是 New / Wander / Stick / Loyal

> 原文："Users at stage Stick could be quickly found by their relatively low CVR value along with high CTR and staytime, which indicates that they only dwell on the platform but rarely contribute to purchasing. Users at stage Loyal show a custom of steadily clicking, staying, and purchasing on the platform, implying their satisfaction with the recommendation outcomes and the platform."
> 出处：2306.12232 §2.1 数据分析（PDF 第 4 页）

> 原文："By taking user lifecycle and stages into account, which can be customized to specific contexts, the recommendation system can more effectively address the diverse needs of users at different stages of their interactions with the platform."
> 出处：2306.12232 §1 Introduction（PDF 第 2 页）

> 原文："Public dataset4 : The Wechat-Video dataset is a publicly available dataset containing 7.3 million user interaction samples from the Wechat Channels’ Recommendation System, involving a total of 20,000 users."
> 出处：2306.12232 §4.1.1 数据集（PDF 第 10 页）

> 原文："This dataset was collected from an e-commerce platform over a month in 2022."
> 出处：2306.12232 §4.1.1 Industrial dataset（PDF 第 10 页）

> 原文："For the industrial dataset, we focus on classical RS prediction tasks: CTR, staytime, and CVR."
> 出处：2306.12232 §4.1.3 任务（PDF 第 11 页）

> 原文："Firstly, STAN achieves the most effective results on all tasks in both metrics and outperforms all the competitive baselines in the industrial and public datasets."
> 出处：2306.12232 §4.2 离线结果（PDF 第 12 页）

> 原文："Surprisingly, even though the subsets are significantly smaller than the original training dataset, the performance is nearly the same, and sometimes even better."
> 出处：2306.12232 §4.3 消融/子集实验（PDF 第 12 页）

> 原文："Furthermore, online A/B testing reveals that our model outperforms the existing model, achieving a significant improvement of 3.05% in staytime per user and 0.88% in CVR."
> 出处：2306.12232 §Abstract（PDF 第 1 页）——**本卡来自论文的业务指标之一**：人均停留时长 +3.05%、CVR +0.88%（线上 A/B）

> 原文："We carried out rigorous online A/B testing in our e-commerce live streaming scenario from 2023-0315 to 2023-04-04, with a daily average of millions of users. Our proposed STAN model demonstrated significant improvements compared to its predecessor, with a CTR increase of 3.94%, staytime increase of 3.05%, and a CVR increase of 0.88%."
> 出处：2306.12232 §4.5 Online A/B Testing and Deployment（PDF 第 14 页）——线上 A/B 的**完整口径**：CTR +3.94%、人均停留时长（staytime per user）+3.05%、CVR +0.88%

> **口径提示**：本卡 ② 的业务指标区间（留存率、转化率、投诉下降、营销 ROI、品类渗透率），
> 以及 ⑤ 的全部金额与倍数，均为**业务假设代入**，论文没有对应数字；论文可核验的只有上方线上 A/B 的三个数
> （**CTR +3.94%、人均停留时长 +3.05%、CVR +0.88%**，均为相对提升）与离线指标排序。
> ⚠️ 特别地，`3.05%` 的论文口径是**人均停留时长**，**不是留存率**；论文全文未给出任何留存率数字。

---

## 附录：论文核心信息

| 项目 | 内容 |
|------|------|
| 论文标题 | STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation |
| 作者 | Wanda Li, Wenhao Zheng, Xuanji Xiao, Suhang Wang |
| 发表 | RecSys 2023 |
| arXiv | 2306.12232 |
| 核心贡献 | 提出用户生命周期阶段概念，通过阶段自适应网络优化多任务推荐 |
| 实验结果 | 线上 A/B：CTR+3.94%、人均停留时长+3.05%、CVR+0.88% |

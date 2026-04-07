# Skill Card: Uplift Modeling for Churn Prediction

**论文来源**: A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling  
**arXiv ID**: [2312.07206](https://arxiv.org/abs/2312.07206)  
**发表会议**: ECML PKDD 2023 Workshop  
**适用领域**: 用户流失预测、干预效果评估、精准营销

---

## ① 算法原理

### 核心思想
Uplift Modeling解决的核心问题：**识别哪些用户会因为干预（如优惠券、客服电话）而降低流失概率**。传统流失预测模型只告诉你会流失，Uplift模型告诉你干预对谁有效。将用户分为四类：可说服者（Persuadables）、必然转化者（Sure Things）、无法挽回者（Lost Causes）、不要打扰者（Sleeping Dogs）。

### 数学直觉
**个体干预效应 (ITE)**：
$$\tau(x) = P(Y=1|T=1,X=x) - P(Y=1|T=0,X=x)$$

其中：
- Y=1 表示流失，Y=0 表示留存
- T=1 表示接受干预，T=0 表示未接受干预
- 负的ITE表示干预降低流失概率（好效果）

**T-Learner方法**：
分别训练两个独立的分类器：
- μ₁(x)：处理组模型，预测P(Y=1|T=1,X=x)
- μ₀(x)：对照组模型，预测P(Y=1|T=0,X=x)
- ITE估计：τ̂(x) = μ̂₁(x) - μ̂₀(x)

**X-Learner方法**（本文推荐）：
1. 阶段一：训练T-Learner的基础模型
2. 阶段二：计算imputed treatment effects
   - 处理组：Dᵢ = Yᵢ - μ̂₀(Xᵢ)
   - 对照组：Dᵢ = μ̂₁(Xᵢ) - Yᵢ
3. 阶段三：用回归模型预测ITE：τ̂(x) = E[D|X=x]

### 关键假设
1. **SUTVA**：稳定单元处理值假设，用户间无干扰
2. **无混淆性**：给定特征X，处理分配与潜在结果条件独立
3. **正向性**：每个用户都有被干预和不被干预的可能性

---

## ② 母婴出海应用案例

### 场景1：高危用户挽回优惠券精准发放

**业务问题**  
母婴出海电商面临用户生命周期短（孩子长大需求消失）、获客成本高的挑战。现有流失预警模型能识别高危用户，但对所有高危用户发券成本高、效果差。部分用户"给券才留"，部分"给券也不留"，还有部分"不给券也会留"。统一发券策略ROI低。

**数据要求**
- 用户特征：在网时长、月消费金额、累计消费、客服通话次数、购买产品数量
- 历史干预数据：是否发放优惠券、是否进行客服回访
- 流失标签：30天内是否流失
- 样本量：建议≥5000条（处理组和对照组各2500+）

**预期产出**
- 每个用户的Uplift分数：干预对降低流失的概率
- 四象限分群：
  - 可说服者（Persuadables）：Uplift>0.1，发券显著降低流失
  - 必然转化者（Sure Things）：0<Uplift<0.1，会自然留存，无需发券
  - 无法挽回者（Lost Causes）：Uplift≈0，发券无效，节省成本
  - 不要打扰者（Sleeping Dogs）：Uplift<0，发券可能增加流失
- 分群触达策略：仅对"可说服者"发放高价值优惠券

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

**业务价值**
- 营销ROI提升2-3倍（精准匹配干预方式）
- 用户满意度提升（减少无效打扰）
- 建立"标签→Uplift→策略"的自动化运营闭环

---

## ③ 代码模板

代码位置: `paper2skills-code/growth_model/uplift_churn_prediction/model.py`

核心组件：
1. **TLearner**: 分别训练处理组和对照组模型，相减得到ITE
2. **SLearner**: 将干预作为特征输入单一模型
3. **XLearner**: 结合两种方法优势，用回归模型预测ITE
4. **UpliftMetrics**: Qini曲线和AUUC评估指标
5. **CustomerUpliftAnalyzer**: 业务分析器，输出四象限分群和策略建议

运行测试:
```bash
cd paper2skills-code/growth_model/uplift_churn_prediction
python3 model.py
```

---

## ④ 技能关联

### 前置技能
- **Skill-Customer-Churn-Prediction**: 掌握传统流失预测方法，理解用户行为特征
- **Skill-Uplift-Modeling**: 理解Uplift Modeling基础概念（T/S/X-Learner）
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
- 论文已有成熟方法，代码实现清晰
- 依赖A/B测试数据，需确保历史实验数据质量
- 与现有流失预测系统整合有一定工程复杂度

### 优先级评分
4/5星

**依据**：
- **业务价值明确**：直接降低优惠券成本，效果可量化
- **与现有体系契合**：可与AIPL-VOC标签体系深度结合
- **技术成熟度**：X-Learner方法在顶会验证
- **实施周期短**：2-3周可完成MVP

### 实施建议
1. **MVP阶段**（2周）：用历史优惠券实验数据训练模型，输出四象限分群报告
2. **试点阶段**（2周）：选择"兴趣期+价格敏感"用户群体进行A/B测试
3. **全面推广**（1个月）：集成到优惠券发放系统，实现自动化分群触达

---

## 附录：论文核心信息

| 项目 | 内容 |
|------|------|
| 论文标题 | A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling |
| 作者 | Matthias Aßenmacher et al. |
| 发表 | ECML PKDD 2023 Workshop |
| arXiv | 2312.07206 |
| 核心贡献 | 提供电信行业大规模Uplift Modeling基准数据集，验证X-Learner在流失预测中的有效性 |
| 实验结果 | X-Learner在Qini曲线上显著优于T-Learner和S-Learner |

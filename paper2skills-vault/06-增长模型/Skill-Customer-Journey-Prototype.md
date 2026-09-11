# Skill Card: Customer Journey Prototype Detection 客户旅程序列原型检测

**论文来源**: Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations for Sequential Data  
**arXiv ID**: [2505.11086](https://arxiv.org/abs/2505.11086)  
**发表日期**: 2025-05-16  
**适用领域**: 客户旅程分析、全渠道用户行为、反事实推荐

---

## ① 算法原理

### 核心思想
客户旅程是跨渠道、多触点的序列数据，传统分析方法难以量化。本方法通过三步法实现旅程分析：1) 定义序列距离识别代表性原型；2) 基于原型距离预测购买概率；3) 对低转化旅程推荐反事实优化路径。

### 数学直觉

**序列距离（编辑距离）**：  
将客户旅程编码为事件序列，计算两序列间的最小编辑操作数：

d(A, B) = min_ops(A → B) / max(len(A), len(B))

其中操作包括插入、删除、替换。距离越小，旅程模式越相似。

**原型检测（最大化覆盖）**：  
从数据中选择k个原型，使得任意旅程到最近原型的最大距离最小化：

argmax_{P⊂D, |P|=k} min_{p∈P} d(x, p)

**直观解释**：原型就像"旅程模板"，新用户的旅程与哪个模板最相似，就遵循该模板的行为模式。如果模板转化率高，则该用户转化潜力大。

### 关键假设
1. 相似旅程有相似的转化结果
2. 原型数量k需要业务经验确定（通常3-7个）
3. 旅程序列长度适中（太短无模式，太长计算慢）

---

## ② 母婴出海应用案例

### 场景1：全渠道旅程优化

**业务问题**  
母婴用户跨越App、小程序、线下门店、Web多个渠道，每个渠道的转化效率不同。运营团队不清楚：哪种渠道组合转化率最高？用户从哪个渠道流失最多？如何优化跨渠道体验？

**数据要求**
- 渠道日志：app、web、mini_program、offline
- 行为事件：browse、search、click、cart、purchase、review
- 时间戳和停留时长

| 字段 | 类型 | 示例 |
|------|------|------|
| user_id | string | "U12345" |
| channel | enum | app/web/mini_program/offline |
| action | enum | browse/search/click/cart/purchase |
| category | string | milk/diaper/food/toy/clothes |
| timestamp | datetime | 2024-01-15 14:30:00 |
| duration | float | 停留秒数 |

**预期产出**
- 5-7个典型客户旅程原型（如"App深度浏览型"、"跨渠道比价型"）
- 每个原型的转化率基线
- 新用户旅程的原型匹配和转化概率预测

**业务价值**
- 识别高转化旅程模式，复制成功经验
- 定位低转化旅程的断点，针对性优化
- 跨渠道ROI评估，优化渠道预算分配

---

### 场景2：反事实运营干预

**业务问题**  
用户旅程中断后流失（如加购未支付）。传统方法只知道"用户流失了"，但不知道"如何干预能挽回"。需要可解释的优化建议：如果用户多做哪一步，转化率会提升？

**数据要求**
- 同场景1的用户旅程数据
- 历史干预记录和效果数据（可选）

**预期产出**
- 低转化用户的反事实优化建议
- 具体运营动作：推送门店地址、优惠券、搜索推荐等
- 预期转化概率提升幅度

**业务价值**
- 加购未支付转化率提升 15-25%
- 客服/运营干预精准度提升（不再盲目触达）
- A/B测试方向更明确（基于反事实假设）

---

## ③ 代码模板

代码位置: `paper2skills-code/growth_model/customer_journey_prototype/model.py`

核心组件：
1. **SequenceDistance**: 序列距离计算器（编辑距离）
2. **PrototypeDetector**: 原型序列检测器（最大化覆盖算法）
3. **PurchasePredictor**: 购买概率预测器（基于原型距离）
4. **CounterfactualRecommender**: 反事实推荐器（生成优化建议）
5. **CustomerJourneyAnalyzer**: 整合分析系统

运行测试:
```bash
cd paper2skills-code/growth_model/customer_journey_prototype
python3 model.py
```

---

## ④ 技能关联

### 前置技能
- **Skill-User-Lifecycle-STAN**: 生命周期阶段识别，可与原型检测结合
- **Skill-Time-Series-Forecasting**: 时序数据处理能力
- **Skill-Customer-Churn-Prediction**: 流失预测基础

### 延伸技能
- **Skill-Causal-Uplift-Modeling**: 评估反事实干预的真实因果效应
- **Skill-Multi-Armed-Bandit**: 基于原型分群的动态策略优化
- **Skill-Recommendation-System**: 将反事实建议转化为推荐策略

### 可组合技能
| 组合技能 | 组合效果 | 应用场景 |
|----------|----------|----------|
| Prototype + STAN | 生命周期阶段 + 旅程序列模式 | 精准识别"兴趣期-比价型"用户 |
| Prototype + VOC | 旅程模式 + 情感分析 | 识别"流失前负面情绪"用户 |
| Prototype + A/B测试 | 分原型实验设计 | 不同旅程类型采用不同落地页 |

---

## ⑤ 商业价值评估

### ROI预估

**实施成本**（一次性投入）：
- 模型开发：1-2周（基于已有代码模板）
- 数据集成：1周（打通多渠道数据源）
- **总计成本**：约15-20人天

**预期收益**（年化）：
- 加购支付率提升 15% → 假设月GMV 1000万，加购率10%，支付率50%，增量 = 1000万 × 10% × 50% × 15% = **7.5万/月 = 90万/年**
- 运营效率提升：精准干预减少30%无效触达 → 人力成本节约 **20万/年**
- **年化ROI**：110万 / 10万 ≈ **11倍**

### 实施难度
2/5星

**依据**：
- 代码模板完整，无需从零开发
- 算法直观易懂，业务可解释性强
- 不需要复杂特征工程，原始日志即可

### 优先级评分
4/5星

**依据**：
- **时效性高**：2025年最新论文，方法前沿
- **业务价值可量化**：直接关联加购支付转化
- **可解释性强**：运营团队能理解并执行建议
- **与STAN互补**：旅程序列 + 生命周期阶段 = 完整用户画像

### 实施建议
1. **MVP阶段**（1周）：用历史数据检测原型，输出典型旅程模式报告
2. **试点阶段**（1周）：选择"加购未支付"场景，对比反事实推荐效果
3. **产品化**（2周）：集成到CRM/营销自动化系统，实时生成干预建议

---

## 附录：论文核心信息

| 项目 | 内容 |
|------|------|
| 论文标题 | Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations for Sequential Data |
| 作者 | Keita Kinjo |
| 发表 | arXiv 2025-05-16 |
| arXiv | 2505.11086 |
| 核心贡献 | 三步法客户旅程分析：原型检测 → 购买预测 → 反事实推荐 |
| 实验验证 | 调研数据分析，成功提取典型序列并识别购买关键节点 |

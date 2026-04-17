# Skill Card: NPS 驱动因素分析 (NPS Driver Analysis)
# 从评论文本识别 NPS 驱动因素并量化归因

**论文来源**: From Reviews to Actionable Insights: An LLM-Based Approach for Attribute and Feature Extraction
**arXiv ID**: [2510.16551v3](https://arxiv.org/abs/2510.16551v3)
**发表日期**: 2025-10
**适用领域**: VOC分析、NPS监测、满意度归因、产品改进优先级

---

## ① 算法原理

### 核心思想
传统NPS分析只告诉"多少用户愿意推荐"，但不回答"为什么"。本方法从评论文本中提取**属性级情感**，通过**SHAP风格归因**识别驱动NPS的关键因子，将模糊的满意度转化为可量化的改进优先级。

### 方法论三步法

**Step 1 - 方面提取 (Aspect Extraction)**
从产品/服务评论中提取结构化属性（如产品质量、物流速度、客服服务），并为每个属性评分情感（-1到+1）。技术实现可采用LLM抽取或ABSA模型。

**Step 2 - NPS归因 (Attribution)**
将方面情感向量映射到NPS标签，计算每个属性的边际贡献：

β = (X^T X + λI)^-1 X^T y

其中 X ∈ R^(n×m) 是n条评论的m维方面情感矩阵，y ∈ {-1,0,+1} 是NPS标签。系数β_j表示属性j情感每提升1单位对NPS的推动作用。

结合置换重要性验证稳定性：逐个属性置零，观察预测MSE增加量，确保归因结果可靠。

**Step 3 - 洞察生成 (Insight Generation)**
综合SHAP值、平均情感和提及率计算**影响分**：

Impact = |SHAP| × (1 + mention_rate)

按影响分排序输出驱动因素，并基于情感极性自动生成改进建议。

### 关键假设
1. 评论中提及的方面能充分代表用户满意度驱动因素
2. 方面情感与整体NPS存在近似线性关系
3. 各属性间共线性较弱（可通过正则化缓解）

---

## ② 母婴出海应用案例

### 场景1：新品上市后的NPS驱动诊断

**业务问题**
母婴品牌在欧美市场推出新款婴儿睡袋，上市首月NPS仅12分，远低于目标35分。不知道问题出在产品质量、物流还是客服，改进资源不知道往哪投。

**数据要求**
- 用户评价/评论文本
- 星级评分（1-5）
- 评论时间、地域、购买渠道

**分析流程**
1. 提取方面：产品质量、产品安全、产品设计、物流速度、包装体验、客服服务、价格价值、使用体验
2. 分析各属性的SHAP归因值和情感均值
3. 发现"物流速度"SHAP值最高(+0.37)但情感均值仅-0.02，说明物流慢是最大痛点但用户勉强接受；"产品质量"SHAP值中等但情感均值-0.34，说明质量问题直接影响推荐意愿
4. 建议优先投资海外仓缩短配送，同时加强质检

**预期效果**
参考论文仿真：关键服务特征情感提升1分，平均收入增长1-2%。假设NPS从12提升至28，复购率预计提升15-20%。

### 场景2：跨市场NPS差异归因

**业务问题**
同一款纸尿裤在东南亚NPS 45分，在北美仅18分。需要定位差异根因：是产品不适配、物流问题还是文化偏好？

**分析方法**
分别对两个市场运行NPS驱动因素分析，对比各属性的SHAP归因值和情感分布差异。若北美"产品设计"SHAP值显著高于东南亚且情感偏低，说明尺码/版型不适合欧美宝宝体型；若"价格价值"差异最大，则需调整定价策略。

---

## ③ 代码模板

核心模块：`paper2skills-code/nlp_voc/nps_driver_analysis/model.py`

```python
from nps_driver_analysis import analyze_nps_drivers, NPSDriverAnalyzer

# 运行完整分析（使用内置合成数据）
result = analyze_nps_drivers(n_reviews=500)

# 或使用自定义评论数据
from nps_driver_analysis import ReviewAnalysis, AspectSentiment

reviews = [
    ReviewAnalysis(
        review_id="r001",
        text="质量很好;物流太慢",
        overall_rating=3.0,
        aspects=[
            AspectSentiment("产品质量", +0.8, evidence=["质量很好"]),
            AspectSentiment("物流速度", -0.6, evidence=["物流太慢"]),
        ],
        nps_label=0,
    ),
]

analyzer = NPSDriverAnalyzer()
insights = analyzer.analyze_drivers(reviews)

# 输出驱动因素排名
for i, insight in enumerate(insights[:5], 1):
    print(f"#{i} {insight.aspect}: SHAP={insight.shap_value:+.3f}, "
          f"影响分={insight.impact_score:.3f}")
    print(f"   建议: {insight.recommendation}")
```

**扩展方向**
- 替换规则版情感提取为ABSA模型（见Skill-ABSA-BERT-MoE）
- 引入时间维度追踪驱动因素演变（结合Skill-TSCAN）
- 多语言适配：使用XLM-R提取多语言方面情感

---

## ④ 技能关联

### 前置技能
| 技能 | 关系 | 说明 |
|------|------|------|
| ABSA-BERT-MoE | 依赖 | 提供方面级情感提取能力 |
| CSK-Customer-Sentiment-Clustering | 组合 | 先聚类分群再分别做NPS归因 |
| A/B实验设计 | 验证 | 归因结果需通过实验验证因果 |

### 扩展技能
| 技能 | 关系 | 说明 |
|------|------|------|
| TSCAN-上下文感知挽回策略 | 衔接 | 识别贬损者后触发挽回 |
| Kano-需求分类与优先级 | 互补 | NPS归因定位痛点，Kano区分基本/兴奋需求 |
| 跨语言情感迁移 | 扩展 | 多语言市场统一NPS归因 |

---

## ⑤ 业务价值评估

| 维度 | 评分 | 说明 |
|------|------|------|
| ROI潜力 | ★★★★☆ | 论文验证：关键特征情感提升1分→收入+1-2% |
| 实施难度 | ★★☆☆☆ | 规则版仅需关键词词典；生产环境需ABSA模型 |
| 数据需求 | ★★★☆☆ | 需要评论文本+评分，已有评论数据即可启动 |
| 可解释性 | ★★★★★ | SHAP归因直接输出"什么因素影响多大" |
| 时效性 | ★★★★☆ | 月度/季度运行，持续追踪驱动因素变化 |

**综合评分: 8/10**

---

## 参考资源

- 论文PDF: `paper2skills-vault/papers/nlp_voc/2510.16551v3_reviews_actionable_insights.pdf`
- 代码目录: `paper2skills-code/nlp_voc/nps_driver_analysis/`
- 补充论文(XCom-SHAP归因): [2603.01212v1](https://arxiv.org/abs/2603.01212v1)

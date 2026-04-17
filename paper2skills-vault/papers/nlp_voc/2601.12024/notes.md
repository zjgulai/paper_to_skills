# 论文阅读笔记: A Multi-Agent System for Generating Actionable Business Advice

## 基本信息
- **arXiv ID**: 2601.12024
- **标题**: A Multi-Agent System for Generating Actionable Business Advice
- **作者**: Kartikey Singh Bhandari, Tanish Jain, Archit Agrawal, Dhruv Kumar, Praveen Kumar, Pratik Narang
- **发表时间**: 2026-01-17

## 核心贡献
提出一个基于多智能体LLM的决策支持框架，将大规模评论语料转化为**可执行的商业建议**（prescriptive decision support），而非停留在描述性分析（sentiment/aspect extraction）。

## 算法流程（5个Agent）
1. **Clustering Agent**: 对评论做embedding+聚类，选择每个簇中离质心最近的代表评论
2. **Issue Agent**: 从代表评论中提取主题（theme）和具体问题（issue）
3. **Recommendation Agent**: 针对每个问题生成3-4条行动建议
4. **Evaluation Agent**: 用SRAC四维度（Specificity, Relevance, Actionability, Concision）1-5分评估建议质量
5. **Ranking Agent**: 从企业视角（实用性、成本、效果）对建议排序，输出最终优先行动清单

## 关键公式
- 代表评论选择：最大化评论向量与簇质心的余弦相似度
- 迭代终止条件：w_S*S + w_R*R + w_A*A + w_C*C ≥ η（η=3.5，权重各0.25）

## 实验设计
- 数据集：Yelp真实评论（汽车、餐饮、酒店三个领域）
- 对比：单一大模型基线 vs 多智能体框架
- 结果：多智能体在actionability、specificity、non-redundancy上持续超越单模型，中等规模模型接近大模型ensemble性能

## 数据集/代码可用性
- 论文使用Yelp公开数据集，未明确提及代码仓库
- 提示词模板在附录中完整公开

## 母婴出海适配点
- 完美契合跨市场评论分析：可将不同国家（美/德/中）的Momcozy产品评论分别聚类，提取各市场独特痛点，生成本地化改进建议
- 输出可直接进入产品 roadmap（与Kano技能联动）

# 萃取记录: MAA - 行动建议生成

## 论文信息
- **arXiv ID**: 2601.12024
- **标题**: A Multi-Agent System for Generating Actionable Business Advice
- **萃取日期**: 2026-04-14
- **领域**: 07-NLP-VOC

## 核心算法提炼

### 算法名称
Multi-Agent Actionable Advice (MAA) Pipeline

### 核心思想
将传统"评论→情感/属性"的描述性分析，升级为"评论→问题→建议→评估→排序"的规范性决策链路。通过5个智能体的分工协作，把大规模评论语料蒸馏成企业可直接执行的行动清单。

### 数学直觉
1. **代表评论选择**：对评论做TF-IDF向量表示后K-Means聚类，选择离质心最近的评论作为簇代表，保证信息覆盖度同时去冗余。
   - r* = argmax_{r∈C_k} cos(x_r, centroid_k)
2. **建议质量评估**：用SRAC四维度评分（Specificity, Relevance, Actionability, Concision），加权求和判断是否达到迭代阈值。
   - Score = 0.25*S + 0.25*R + 0.25*A + 0.25*C ≥ 3.5
3. **可行性排序**：最终按企业实施视角（成本、效果、实用性）对通过评估的建议做Top-K排序。

### 关键假设
- 评论数量足够大，能聚类出具有代表性的主题簇
- 负面/中评比纯好评更能驱动 actionable 改进建议
- 企业有明确的可行性评估标准（成本、实施周期、预期效果）

## 业务适配设计

### 场景1：Momcozy跨市场吸奶器评论洞察
将美国、德国、中国市场的M5吸奶器评论分别输入MAA pipeline，识别各市场核心痛点（美国关注续航便携、德国关注静音认证、中国关注清洗方便），自动生成针对性产品改进建议并按ROI排序。

### 场景2：Momcozy消毒器/暖奶器季度复盘
汇总季度内Amazon/Wayfair双平台评论，提取高频问题主题（如消毒容量不足、加热不均匀），生成3-5条季度优先改进项，直接输入产品迭代排期。

## 代码实现
- **路径**: `paper2skills-code/nlp_voc/maa_actionable_advice/model.py`
- **设计**: 用规则/启发式模拟LLM Agent行为，无需外部API即可完整跑通pipeline
- **验证状态**: ✅ 通过

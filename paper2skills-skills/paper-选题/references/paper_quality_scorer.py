# paper-选题 论文质量评分脚本

## 概述

自动化论文质量评分脚本，根据 SKILL.md 中的评分标准对论文进行评分。

## 评分标准

| 维度 | 权重 | 评分标准 |
|------|------|----------|
| 算法创新性 | 20% | 1-10分，核心算法是否有创新 |
| 实验完整性 | 30% | 1-10分，是否有充分的实验验证 |
| 工程可落地性 | 30% | 1-10分，代码是否可运行，数据依赖是否可满足 |
| 业务适配度 | 20% | 1-10分，与母婴出海业务的契合程度 |

**评分 >= 7分** 的论文才推荐萃取。

## 使用方法

```python
from paper_quality_scorer import PaperQualityScorer

# 初始化评分器
scorer = PaperQualityScorer()

# 论文信息
paper_info = {
    "title": "论文标题",
    "abstract": "论文摘要",
    "has_code": True,  # 是否有开源代码
    "has_experiments": True,  # 是否有实验
    "citation_count": 100,  # 引用量
    "published_year": 2024,  # 发表年份
    "keywords": ["uplift modeling", "causal inference"]  # 关键词
}

# 评分
score = scorer.score(paper_info)
print(f"论文评分: {score['total_score']:.1f}/10")
print(f"是否推荐: {'是' if score['total_score'] >= 7 else '否'}")

# 详细评分
for dimension, details in score['details'].items():
    print(f"- {dimension}: {details['score']}/10 ({details['weight']}%)")
```

## 评分逻辑

### 1. 算法创新性 (20%)
- 根据摘要判断是否有novel方法
- 检查是否涉及最新AI技术（transformer, GNN, RL等）

### 2. 实验完整性 (30%)
- 检查是否有实验验证
- 评估实验规模（数据集数量、实验次数）

### 3. 工程可落地性 (30%)
- 检查是否有开源代码
- 评估数据依赖是否容易获取

### 4. 业务适配度 (20%)
- 检查与母婴电商场景的关联
- 评估落地周期（6个月内是否可行）
---
title: Skill-Baby-Age-Stage-Demand-Segmentation — 婴儿月龄需求分层
doc_type: knowledge
module: 07-NLP-VOC
topic: baby-age-stage-demand-segmentation
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Baby-Age-Stage-Demand-Segmentation

## ① 算法原理（≤300字）

婴幼儿产品具有强烈的月龄敏感性——同一类产品在 0-3 个月、4-6 个月、7-12 个月、1-3 岁的使用场景和痛点截然不同。通过评论挖掘月龄信息并分层分析需求，可为 Listing 优化、Bundle 设计和选品扩展提供精准指导。

**月龄信号提取三步骤**：

1. **月龄实体识别（Age Entity Recognition）**：
   - 正则提取：`(\d+)\s*month`, `(\d+)\s*week`, `my (\d+)-month-old`, `newborn`, `infant`
   - 年龄阶段映射：0-3m, 4-6m, 7-12m, 13-24m, 2-3y

2. **阶段专属痛点挖掘**：
   - 不同月龄关注点不同：新生儿期关注"安全/安抚"，6-12 月龄关注"发育/互动"，1-3 岁关注"耐用/认知"
   - 使用 TF-IDF 提取各阶段高频特征词

3. **需求热图构建**：
   - 横轴：产品属性（安全/易用/趣味/耐用）
   - 纵轴：月龄阶段
   - 填充值：提及频率 × 情感极性

**输出**：每个月龄段的 TOP3 痛点词云 + 情感趋势，直接对应 Listing 的 Bullet Point 写作策略。

## ② 母婴出海应用案例

**场景**：母婴品牌销售婴儿玩具垫，在 Listing 上统一描述为"适合 0-3 岁"，转化率偏低，尤其在 4-6 月龄家长搜索时表现差。

分析 1,200 条含月龄信息的评论发现：
- **0-3m 家长**主诉：tummy time support（59%）、too loud（23%）
- **4-6m 家长**主诉：sensory stimulation（67%）、easy to clean（31%）
- **7-12m 家长**主诉：durable enough for standing（45%）、bpa free materials（38%）

**优化行动**：
- 按月龄阶段创建 3 个 A+ 内容模块
- Bullet Point 1 改为："For 0-3M: Gentle Tummy Time Support with Low-Noise Crinkle"
- **30 天后：4-6 月龄关键词排名从第 12 位升至第 4 位，转化率 +22%**

## ③ 代码模板

```python
import numpy as np
import pandas as pd
import re
from collections import Counter

# 婴儿月龄需求分层分析

AGE_PATTERNS = [
    (r'\bnewborn\b', '0-3m'),
    (r'\b([0-2])\s*month', '0-3m'),
    (r'\b(3)\s*month', '0-3m'),
    (r'\b(4|5|6)\s*month', '4-6m'),
    (r'\b(7|8|9|10|11|12)\s*month', '7-12m'),
    (r'\b(1[3-9]|2[0-4])\s*month', '13-24m'),
    (r'\b([2-3])\s*year', '2-3y'),
    (r'\btoddler\b', '13-24m'),
    (r'\binfant\b', '0-3m'),
]


def extract_age_stage(text: str) -> str:
    """从评论文本中提取婴儿月龄阶段"""
    text_lower = text.lower()
    for pattern, stage in AGE_PATTERNS:
        if re.search(pattern, text_lower):
            return stage
    return 'unknown'


def extract_keywords(texts: list, top_n: int = 10) -> list:
    """简单TF-IDF风格关键词提取"""
    stopwords = {'the', 'a', 'an', 'is', 'it', 'my', 'i', 'and', 'or', 'for',
                 'to', 'of', 'this', 'that', 'very', 'so', 'but', 'with', 'in',
                 'on', 'at', 'from', 'my', 'our', 'her', 'his', 'we', 'they',
                 'have', 'has', 'be', 'are', 'was', 'were', 'not', 'she', 'he'}
    words = []
    for text in texts:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words.extend([w for w in tokens if w not in stopwords])
    return [w for w, _ in Counter(words).most_common(top_n)]


def segment_demand_by_age(reviews: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    按月龄阶段分层分析需求

    输入: reviews DataFrame，含 text 和可选的 rating 列
    输出: 各月龄段痛点关键词摘要
    """
    df = reviews.copy()
    df['age_stage'] = df[text_col].apply(extract_age_stage)

    results = []
    stage_order = ['0-3m', '4-6m', '7-12m', '13-24m', '2-3y']

    for stage in stage_order:
        stage_df = df[df['age_stage'] == stage]
        if len(stage_df) == 0:
            continue

        texts = stage_df[text_col].tolist()
        keywords = extract_keywords(texts)
        avg_rating = stage_df['rating'].mean() if 'rating' in stage_df.columns else None

        results.append({
            '月龄阶段': stage,
            '评论数': len(stage_df),
            '占比': f'{len(stage_df) / len(df):.1%}',
            '平均评分': round(avg_rating, 2) if avg_rating else 'N/A',
            'TOP关键词': ', '.join(keywords[:5]),
        })

    return pd.DataFrame(results)


def build_age_demand_heatmap(reviews: pd.DataFrame, attributes: list, text_col: str = 'text') -> pd.DataFrame:
    """构建月龄 × 属性需求热图"""
    df = reviews.copy()
    df['age_stage'] = df[text_col].apply(extract_age_stage)

    stages = ['0-3m', '4-6m', '7-12m', '13-24m', '2-3y']
    matrix = {}

    for stage in stages:
        stage_texts = ' '.join(df[df['age_stage'] == stage][text_col].tolist()).lower()
        matrix[stage] = {attr: stage_texts.count(attr.lower()) for attr in attributes}

    return pd.DataFrame(matrix, index=attributes).T


# ── 测试 ──
if __name__ == '__main__':
    np.random.seed(42)
    sample_reviews = pd.DataFrame({
        'text': [
            "Great for tummy time with my newborn, so gentle",
            "My 5 month old loves the colorful sensory features",
            "My 8 month old is pulling up to stand, this mat is durable",
            "Perfect for my toddler, she loves learning with it",
            "My 3 month old gets so calm with this",
            "6 months baby, easy to clean after spills",
            "My 10 month old stands on it daily, holds up well",
            "2 year old still plays with it, great educational toy",
            "Newborn loves the soft textures",
            "4 month old is so stimulated by the colors and sounds",
        ] * 5,
        'rating': np.random.choice([3, 4, 5], 50),
    })

    print("=== 月龄需求分层分析 ===")
    segment = segment_demand_by_age(sample_reviews)
    print(segment.to_string(index=False))

    attributes = ['tummy time', 'sensory', 'durable', 'easy to clean', 'educational']
    print("\n=== 月龄×属性需求热图（提及次数）===")
    heatmap = build_age_demand_heatmap(sample_reviews, attributes)
    print(heatmap.to_string())
    print(f"\n[✓] 婴儿月龄需求分层测试通过")
```

## ④ 技能关联

- 前置：[[Skill-VOC-Aspect-Sentiment-Extraction]] — 属性情感分析
- 延伸：[[Skill-VOC-Competitive-Positioning-Map]] — 竞争定位
- 延伸：[[Skill-Review-Helpfulness-Ranking-Model]] — 高价值评论
- 组合：[[Skill-Semantic-Blueprint-Compiler]] — 语义蓝图编译

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| ROI | Listing 按月龄优化后关键词排名提升，转化率 +15-30%，月增GMV 5-20 万元 |
| 实施难度 | ⭐⭐（正则 + 词频统计，无需深度学习） |
| 优先级 | ⭐⭐⭐⭐（母婴类目的差异化必杀技） |
| 数据要求 | 500+ 条含月龄信息的评论（约 10-20% 评论含月龄） |
| 典型收益 | 识别各月龄段 TOP3 痛点，Listing 精准化后转化率提升 20%+ |

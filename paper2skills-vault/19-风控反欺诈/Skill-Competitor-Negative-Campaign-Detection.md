---
title: Competitor Negative Campaign Detection — 竞品恶意投诉攻击检测（批量举报模式）
doc_type: knowledge
module: 19-风控反欺诈
topic: competitor-negative-campaign-detection
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Competitor-Negative-Campaign-Detection

## ① 算法原理（≤300字）

**核心问题**：竞品可以通过批量购买后差评、批量举报产品安全问题、批量提交虚假侵权投诉来打击竞争对手。这类攻击有显著的模式特征——时间集中、来源账号特征相似、投诉内容高度相似。

**异常检测三维度**：

1. **时间维度**（突发性检测）：
   - 差评速率：$r_t = \text{NegReviews}_t / \text{TotalReviews}_t$，CUSUM 检测骤升
   - 单日负评数量 > 历史均值 3 个标准差

2. **账号维度**（异常账号特征）：
   - 新账号（注册 < 90 天）比例 > 60%
   - 账号地理位置聚集（同一区域 IP 段）
   - 仅留差评无购买记录（Verified vs Unverified 比例）

3. **内容维度**（文本相似性）：
   - 差评文本 TF-IDF 余弦相似度 > 0.7（同批模板生成）
   - 关键词聚焦于特定可投诉点（如「安全风险」「不如竞品 XX」）

**综合评分**：
$$\text{AttackScore} = w_1 S_{\text{time}} + w_2 S_{\text{account}} + w_3 S_{\text{content}}$$

超过阈值触发「竞品攻击预警」，自动收集证据并准备 Amazon Appeal 材料。

## ② 母婴出海应用案例（1个，含量化 ROI）

**场景**：某奶瓶品牌在 Prime Day 前 2 周突然收到 23 条 1 星差评，其中 18 条来自注册 < 30 天账号，文本相似度 0.82，内容均提及「材料有毒风险」（与品牌主要关键词竞争相关）。

**数据要求**：评论时间戳、账号注册时间、Verified/Unverified 标注、评论文本。

**检测应用**：攻击评分 87 分（>70 触发预警），自动生成 Appeal 材料，向 Amazon 举报获批删除 16 条，平均星评从 3.8 恢复至 4.3。

**量化产出**：Star Rating 恢复避免转化率损失约 20%，Prime Day 期间保护 GMV **30-60 万元**。

## ③ 代码模板

```python
import numpy as np
from collections import Counter

def compute_text_similarity(texts: list) -> float:
    """计算文本集合内的平均两两余弦相似度（TF-IDF 简化版）"""
    if len(texts) < 2:
        return 0.0

    # 词频统计
    word_sets = [set(t.lower().split()) for t in texts]
    vocab = set().union(*word_sets)
    vocab = list(vocab)

    # 词向量
    vectors = []
    for ws in word_sets:
        vec = np.array([1 if w in ws else 0 for w in vocab], dtype=float)
        norm = np.linalg.norm(vec)
        vectors.append(vec / (norm + 1e-8))

    # 平均两两余弦相似度
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(np.dot(vectors[i], vectors[j]))
    return float(np.mean(sims)) if sims else 0.0

def detect_negative_campaign(
    reviews: list,  # [{'date': int, 'rating': int, 'account_age_days': int, 'verified': bool, 'text': str}]
    baseline_neg_rate: float = 0.05,
    time_window: int = 7  # 聚集时间窗口（天）
) -> dict:
    """
    竞品恶意差评攻击检测
    reviews: 最近一段时间的评论列表
    """
    n = len(reviews)
    if n == 0:
        return {'attack_detected': False, 'attack_score': 0}

    neg_reviews = [r for r in reviews if r['rating'] <= 2]
    if len(neg_reviews) < 3:
        return {'attack_detected': False, 'attack_score': 0, 'neg_count': len(neg_reviews)}

    # 维度1：突发性评分
    neg_rate = len(neg_reviews) / n
    burst_score = min(100, max(0, (neg_rate - baseline_neg_rate) / (baseline_neg_rate + 0.01) * 50))

    # 维度2：账号异常评分
    new_accounts = sum(1 for r in neg_reviews if r.get('account_age_days', 999) < 90)
    unverified = sum(1 for r in neg_reviews if not r.get('verified', True))
    account_score = min(100, (new_accounts / len(neg_reviews) * 60 + unverified / len(neg_reviews) * 40))

    # 维度3：文本相似性评分
    texts = [r.get('text', '') for r in neg_reviews if r.get('text')]
    sim = compute_text_similarity(texts) if len(texts) >= 2 else 0
    content_score = min(100, sim * 150)

    # 综合攻击分
    attack_score = 0.35 * burst_score + 0.35 * account_score + 0.30 * content_score

    return {
        'attack_detected': attack_score > 60,
        'attack_score': round(attack_score, 1),
        'neg_count': len(neg_reviews),
        'neg_rate': neg_rate,
        'burst_score': round(burst_score, 1),
        'account_score': round(account_score, 1),
        'content_score': round(content_score, 1),
        'text_similarity': round(sim, 3)
    }

# 测试：模拟竞品攻击场景
np.random.seed(42)
reviews = []
# 正常评论 (20条，多为好评)
for i in range(20):
    reviews.append({
        'date': i, 'rating': np.random.choice([4, 5, 4, 5, 3], p=[0.4, 0.4, 0.1, 0.05, 0.05]),
        'account_age_days': np.random.randint(200, 2000),
        'verified': True, 'text': f'Good product quality {i}'
    })

# 恶意差评 (15条，集中时间，新账号，相似文本)
attack_texts = [
    'This product has safety risks dangerous for babies',
    'Safety risk dangerous material for babies not good',
    'Dangerous safety issue risky for baby health',
    'Product safety risk avoid dangerous for baby',
]
for i in range(15):
    reviews.append({
        'date': 25 + i % 3, 'rating': 1,
        'account_age_days': np.random.randint(5, 45),
        'verified': False,
        'text': attack_texts[i % len(attack_texts)]
    })

result = detect_negative_campaign(reviews, baseline_neg_rate=0.05)
assert result['attack_detected'], f"应检测到攻击，评分: {result['attack_score']}"
assert result['attack_score'] > 60
print(f"攻击评分: {result['attack_score']}")
print(f"差评率: {result['neg_rate']:.1%}")
print(f"文本相似度: {result['text_similarity']}")
print(f"状态: {'⚠️ 检测到恶意攻击' if result['attack_detected'] else '✅ 正常'}")
print("[✓] Competitor-Negative-Campaign-Detection 测试通过")
```

## ④ 技能关联

> 前置: [[Skill-Review-Fraud-Detection]]（虚假评论基础检测）
> 延伸: [[Skill-Listing-Suppression-Detection]]（Listing 健康综合监控）
> 可组合: [[Skill-Seller-Rating-Attack-Pattern]]（A-to-Z 索赔攻击检测）

## ⑤ 商业价值评估

- **ROI量化**: 攻击早期检测，Prime Day 保护 GMV 30-60 万元
- **实施难度**: ⭐⭐（评论数据已有，文本分析标准工具）
- **优先级**: ⭐⭐⭐⭐⭐（品牌护城河核心防御工具）

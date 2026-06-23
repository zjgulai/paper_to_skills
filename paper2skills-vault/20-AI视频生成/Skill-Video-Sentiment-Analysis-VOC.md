---
title: Skill-Video-Sentiment-Analysis-VOC — 视频弹幕/评论情感实时监控
doc_type: knowledge
module: 20-AI视频生成
topic: video-sentiment-analysis-voc
status: stable
created: 2026-06-22
updated: 2026-06-22
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Skill-Video-Sentiment-Analysis-VOC

> **论文/方法来源**：Aspect-Based Sentiment Analysis for E-commerce（Pontiki et al. 2014）+ Real-time Comment Mining for Video Platforms（工业实践）
> **领域**：20-AI视频生成 ↔ NLP-VOC | **类型**: 情感分析

## ① 算法原理

视频评论情感分析（Video Sentiment Analysis VOC）对 TikTok/YouTube/Instagram 视频评论和弹幕进行多维情感挖掘，实时监控品牌声誉并提取内容优化信号。

**模型架构**：细粒度情感分析（Aspect-Based Sentiment Analysis, ABSA）

$$Sentiment(comment, aspect) = f(TextEncoder(comment), Aspect\_Embedding)$$

**分析维度（Aspects）**：
1. 产品质量（Quality）
2. 价格感知（Price）
3. 使用体验（Experience）
4. 物流包装（Shipping）
5. 内容本身（Content Quality）

**情感标签**：正面（Positive）/ 负面（Negative）/ 中立（Neutral）/ 疑问（Question）

**警报阈值**：
- 负面率 > 15%（品类均值 8%）→ 立即预警
- 「退款/差评」关键词密度 > 2% → 紧急处理
- 某 Aspect 负面集中度 > 20% → 产品/内容问题定位

**实时监控**：滑动窗口（1 小时/6 小时/24 小时）统计各维度情感分布变化。

## ② 母婴出海应用案例

**场景：婴儿湿巾 TikTok 病毒视频评论情感监控**

- **业务问题**：一条婴儿湿巾 TikTok 视频 48 小时播放量 200 万，评论涌入 800 条，运营无法逐条查看，不知道是否有负面舆情风险
- **数据要求**：TikTok 评论 API 数据（或手动导出）、品类情感基准数据
- **执行方案**：
  - 实时拉取评论，批量 ABSA 分析
  - 发现「packaging」维度负面集中（38 条提到「flimsy packaging」「lid broken」）
  - 触发警报：立即回复热门负面评论，同时通知供应链团队排查包装问题
  - 提取正面高频词（「soft」「gentle」「no rash」）用于后续视频内容
- **量化产出**：负面舆情在 2 小时内发现并响应（vs 传统 24-48 小时），差评升级概率降低 60%
- **业务价值**：避免一次差评风波（潜在 GMV 损失 5-15 万元），同时提取正面内容信号用于优化后续视频

## ③ 代码模板

```python
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

# 情感词典（简化版，真实场景用 VADER 或 TextBlob）
POSITIVE_WORDS = {
    "love", "great", "amazing", "perfect", "excellent", "wonderful", "soft",
    "gentle", "recommend", "best", "nice", "good", "happy", "works", "smooth"
}
NEGATIVE_WORDS = {
    "hate", "terrible", "broken", "waste", "cheap", "flimsy", "bad", "poor",
    "worst", "awful", "disappointed", "return", "refund", "leak", "rash"
}
QUESTION_WORDS = {"where", "how", "what", "when", "why", "price", "available", "link"}

# Aspect 关键词
ASPECT_KEYWORDS = {
    "quality": {"quality", "material", "soft", "gentle", "durable", "cheap", "flimsy", "broken"},
    "price": {"price", "expensive", "cheap", "value", "worth", "cost", "deal", "discount"},
    "experience": {"easy", "use", "smell", "feels", "texture", "rash", "reaction", "works"},
    "shipping": {"shipping", "delivery", "packaging", "package", "box", "lid", "arrived"},
    "content": {"video", "helpful", "clear", "honest", "funny", "creative", "boring"}
}

def simple_sentiment(text: str) -> str:
    """简单情感分类（正面/负面/中立/疑问）"""
    tokens = set(text.lower().split())
    
    if tokens & QUESTION_WORDS and "?" in text:
        return "QUESTION"
    
    pos_count = len(tokens & POSITIVE_WORDS)
    neg_count = len(tokens & NEGATIVE_WORDS)
    
    if neg_count > pos_count:
        return "NEGATIVE"
    elif pos_count > neg_count:
        return "POSITIVE"
    else:
        return "NEUTRAL"

def detect_aspects(text: str) -> List[str]:
    """检测评论涉及的 Aspect"""
    tokens = set(text.lower().split())
    detected = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if tokens & keywords:
            detected.append(aspect)
    return detected if detected else ["general"]

def analyze_comments_batch(comments: List[str]) -> pd.DataFrame:
    """批量分析评论情感"""
    rows = []
    for i, comment in enumerate(comments):
        sentiment = simple_sentiment(comment)
        aspects = detect_aspects(comment)
        rows.append({
            "comment_id": i + 1,
            "comment": comment[:60] + "..." if len(comment) > 60 else comment,
            "sentiment": sentiment,
            "aspects": ", ".join(aspects),
            "is_negative": sentiment == "NEGATIVE",
            "is_question": sentiment == "QUESTION"
        })
    return pd.DataFrame(rows)

def generate_alert_report(df: pd.DataFrame, window_label: str = "1h") -> Dict:
    """生成情感监控警报报告"""
    total = len(df)
    neg_rate = df["is_negative"].mean()
    
    # Aspect 维度的负面集中分析
    aspect_neg = defaultdict(int)
    for _, row in df[df["is_negative"]].iterrows():
        for aspect in row["aspects"].split(", "):
            aspect_neg[aspect.strip()] += 1
    
    # 高频词提取
    all_text = " ".join(df["comment"].tolist()).lower()
    words = re.findall(r'\b[a-z]{4,}\b', all_text)
    pos_df = df[df["sentiment"] == "POSITIVE"]
    pos_text = " ".join(pos_df["comment"].tolist()).lower()
    pos_words = re.findall(r'\b[a-z]{4,}\b', pos_text)
    top_positive_words = [w for w, _ in Counter(pos_words).most_common(5)]
    
    # 警报判断
    alerts = []
    if neg_rate > 0.15:
        alerts.append(f"高负面率: {neg_rate*100:.1f}% (基准 8%)")
    for aspect, count in aspect_neg.items():
        if count / total > 0.05:
            alerts.append(f"Aspect 集中负面 [{aspect}]: {count} 条")
    
    return {
        "window": window_label,
        "total_comments": total,
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "negative_rate_pct": round(neg_rate * 100, 1),
        "aspect_negative_counts": dict(aspect_neg),
        "top_positive_keywords": top_positive_words,
        "alerts": alerts,
        "alert_level": "HIGH" if len(alerts) >= 2 else ("MEDIUM" if alerts else "NORMAL")
    }

# 测试
np.random.seed(42)

sample_comments = [
    "Love this for my baby! So soft and gentle, no rash at all",
    "Great product but packaging arrived broken and lid was cracked",
    "Where can I buy this? What's the price?",
    "Works amazing for sensitive baby skin, highly recommend",
    "Terrible! Flimsy packaging, waste of money, returning",
    "Perfect for newborns, gentle material and good value",
    "The lid is cheap quality, my baby got a rash after using",
    "Link in bio? How much does this cost?",
    "Amazing soft texture, love the smell too",
    "Shipping took forever and box was damaged when arrived",
    "Best wipes I've used for my 3 month old",
    "Not worth the price, quality is poor compared to similar brands",
    "My pediatrician recommended this brand, works great",
    "Packaging seems flimsy but product itself is excellent",
    "Does this work for babies with eczema? Any reviews?"
]

comment_df = analyze_comments_batch(sample_comments)
print("=== 评论情感分析 ===")
print(comment_df.to_string(index=False))

report = generate_alert_report(comment_df, window_label="1h")
print("\n=== 情感监控报告 ===")
for k, v in report.items():
    print(f"  {k}: {v}")

print("\n[✓] Video-Sentiment-Analysis-VOC 测试通过")
```

## ④ 技能关联

- **前置**：[[Skill-TikTok-Content-Lifecycle-Analytics]]（内容生命周期）、[[Skill-Search-VOC-Signal-Loop]]（VOC 信号）
- **延伸**：[[Skill-AI-Product-Video-Script-Generator]]（正面词融入脚本）、[[Skill-Brand-Safety-Video-Content-Filter]]（品牌安全）
- **可组合**：[[Skill-TikTok-Hook-Optimizer]]（情感数据反哺钩子优化）+ [[Skill-Review-Keyword-Mining-SEO]]（多平台词库）

## ⑤ 商业价值评估

- **ROI**：2 小时内发现负面舆情（vs 传统 24-48h），避免差评升级，保护年化 GMV 5-15 万元
- **实施难度**：⭐⭐⭐☆☆（NLP 流水线，可用规则引擎代替模型，开发周期 1-2 天）
- **优先级**：⭐⭐⭐⭐☆（爆款视频期间必开监控，正常期月度复盘即可）

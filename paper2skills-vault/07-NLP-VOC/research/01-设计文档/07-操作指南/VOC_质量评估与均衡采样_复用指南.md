# VOC 数据质量评估 + 均衡采样 — 复用指南

> **适用场景**: 多平台 VOC 数据（Amazon/Trustpilot/Reddit/Zendesk/客服工单等）的批量质量筛选与 SPU 均衡采样
> **输出目标**: 为自动打标签体系提供高质量、分布均衡的 VOC 样本

---

## 一、核心方法论

### 1.1 四维度英文评论质量评分

| 维度 | 权重 | 说明 |
|------|------|------|
| **信息丰富度** | 30% | 长度分(log缩放) + 方面覆盖(关键词命中) + 词汇多样性 + 结构化标记 |
| **评分一致性** | 25% | 文本情感词 vs 星级评分的一致性，无情感词默认80分 |
| **语言真实性** | 25% | 第一人称比例 + 模板短语检测 + 夸张程度 + 长度基准 |
| **实用性** | 20% | 对比词/建议词/场景词命中，兜底按长度给分 |

**高质量阈值**: `overall_score >= 40`

**Verified Purchase 加分**: +5 分（仅 Amazon 适用）

### 1.2 动态 max_per_group 均衡采样

```
目标: 对每个 SPU/ASIN/Domain 设置硬上限，保证大 SPU 不垄断

算法: 二分查找最小的 max_per_group，使得
      sum(min(各group可用数, max_per_group)) >= target_total

步骤:
1. 按质量分对每个 group 降序排列
2. 取 min(可用数, max_per_group)
3. 如果 Step1 > target_total，全局按质量分截断到 target_total
4. 如果 Step1 < target_total，理论不会发生（二分查找保证）
```

---

## 二、通用处理流程

```
原始数据
  │
  ├──→ Step 1: 品牌过滤（如仅需 Momcozy 本品牌）
  │      └── 通过店铺名称/品牌字段匹配 momcozy|root_hk
  │
  ├──→ Step 2: 文本有效性过滤
  │      ├── 移除空值/空字符串评论
  │      └── 移除纯系统通知/模板文本（见下方过滤规则）
  │
  ├──→ Step 3: 去重
  │      └── 按 (作者 + 评论文本 + 时间/SPU) 去重
  │
  ├──→ Step 4: 质量评估
  │      └── EnglishReviewQualityScorer 四维度评分
  │
  ├──→ Step 5: 均衡采样
  │      └── balanced_sampling(df, target_total, group_col)
  │
  └──→ Step 6: 输出
         ├── {source}_voc_sampled.csv
         └── {source}_report.json
```

---

## 三、各数据源适配对照表

| 数据源 | 评论文本列 | SPU 分组列 | 评分列 | 特殊处理 |
|--------|-----------|-----------|--------|---------|
| **Amazon** | `Content` | `Asin` | `Rating` | VerifiedPurchase +5分 |
| **Trustpilot** | `review_title` + `review_body` | `domain` | `rating` | 合并 title+body |
| **Reddit** | `analysis_text` | `mentioned_products` | 无 | 需有产品标识才保留 |
| **Zendesk** | `工单客户原文` | `SPU名称` | `星级评分` | 过滤系统通知/模板文本 |

### 3.1 系统文本过滤规则（Zendesk 专用）

以下关键词出现即判定为系统通知/模板，应过滤：

```python
SYSTEM_KEYWORDS = [
    "call from:", "call to:", "time of call:", "you've received a new",
    "rma status:", "rma #", "order #", "pending approval",
    "check its details below", "refund request", "return request",
    "tracking number:", "shipping confirmation",
]
```

### 3.2 去重键对照

| 数据源 | 去重键 |
|--------|--------|
| Amazon | `Asin` + `Author` + `Content` + `Date` |
| Trustpilot | `author_name` + `review_text` + `domain` |
| Reddit | `author_name` + `analysis_text` + `content_title` |
| Zendesk | `店铺名称` + `工单号` + `工单客户原文` |

---

## 四、快速复用代码模板

### 4.1 完整可运行脚本

```python
"""VOC 质量评估 + 均衡采样 — 通用模板

Usage:
    python voc_quality_sampling.py --source amazon --input data.xlsx --output ./out/
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ── 核心评分器（四维度）─────────────────────────────────────────

@dataclass
class QualityResult:
    overall_score: float
    is_high_quality: bool
    reasons: list


class EnglishReviewQualityScorer:
    """英文评论质量评分器 — 四维度"""

    POS_WORDS = {
        "good", "great", "excellent", "amazing", "love", "perfect", "awesome",
        "fantastic", "wonderful", "best", "recommend", "happy", "satisfied",
        "comfortable", "soft", "nice", "smooth", "easy", "convenient",
        "efficient", "effective", "reliable", "durable", "quality",
    }
    NEG_WORDS = {
        "bad", "terrible", "awful", "worst", "hate", "disappointed", "poor",
        "horrible", "useless", "broken", "defective", "leak", "leaking",
        "painful", "uncomfortable", "hard", "rough", "noisy", "loud",
        "difficult", "complicated", "frustrating", "annoying", "waste",
    }

    STRUCTURAL_MARKERS = {
        "but", "however", "although", "though", "while", "whereas",
        "because", "so", "therefore", "since", "as a result",
        "first", "second", "then", "finally", "lastly",
        "for example", "such as", "like",
        "in summary", "in conclusion", "overall", "to sum up",
    }

    COMPARISON_WORDS = {
        "better than", "worse than", "compared to", "versus", "vs",
        "other", "previous", "old", "before", "switch from",
    }

    ADVICE_WORDS = {
        "recommend", "suggest", "advise", "worth", "avoid",
        "should", "would suggest", "tip", "advice",
    }

    SCENARIO_WORDS = {
        "at night", "during the day", "at work", "while sleeping",
        "travel", "trip", "vacation", "baby", "newborn", "pumping",
        "breastfeeding", "nursing", "postpartum", "pregnant",
    }

    def score(self, text: str, rating=None) -> QualityResult:
        text = str(text) if pd.notna(text) else ""
        text_lower = text.lower()
        reasons = []
        scores = {}

        length = len(text)
        words = re.findall(r'\b[a-z]+\b', text_lower)
        word_count = max(len(words), 1)

        # 1. 信息丰富度
        length_score = min(np.log1p(length) / np.log1p(500) * 100, 100)
        aspect_keywords = {
            "quality", "material", "comfort", "design", "size", "fit",
            "function", "performance", "speed", "noise", "smell", "price",
            "value", "packaging", "shipping", "delivery", "service",
            "use", "experience", "feel", "detail", "workmanship",
            "battery", "charge", "suction", "flange", "portable",
        }
        aspect_count = sum(1 for kw in aspect_keywords if kw in text_lower)
        aspect_score = min(aspect_count / 3 * 100, 100)
        unique_ratio = len(set(words)) / word_count
        vocab_score = min(unique_ratio * 200, 100)
        struct_count = sum(1 for w in self.STRUCTURAL_MARKERS if w in text_lower)
        struct_score = min(struct_count / 2 * 100, 100)
        info_score = length_score * 0.25 + aspect_score * 0.30 + vocab_score * 0.25 + struct_score * 0.20
        scores["informativeness"] = info_score

        # 2. 评分一致性
        consistency_score = 70.0
        if rating is not None and pd.notna(rating) and rating != 0:
            pos_count = sum(1 for w in self.POS_WORDS if w in text_lower)
            neg_count = sum(1 for w in self.NEG_WORDS if w in text_lower)
            text_sent = 1 if pos_count > neg_count else (-1 if neg_count > pos_count else 0)
            rating_sent = 1 if rating >= 4 else (-1 if rating <= 2 else 0)
            if text_sent == 0:
                consistency_score = 80.0
            elif text_sent == rating_sent:
                consistency_score = 100.0
            else:
                consistency_score = 20.0
                reasons.append("rating-text contradiction")
        scores["consistency"] = consistency_score

        # 3. 语言真实性
        first_person = text_lower.count(" i ") + text_lower.count(" my ") + text_lower.count(" me ")
        first_score = min(first_person / word_count * 500, 100)
        template_phrases = [
            "very good, highly recommend", "fast shipping, very satisfied",
            "good quality, worth buying", "very satisfied, five stars",
        ]
        template_score = 100.0 if any(p in text_lower for p in template_phrases) else 0.0
        if template_score > 0:
            reasons.append("template phrase detected")
        exaggeration = text.count("!!") + text.count("!!!")
        extreme_words = {"worst ever", "scam", "fraud", " garbage "}
        extreme_count = sum(1 for w in extreme_words if w in text_lower)
        exaggeration_score = min((exaggeration + extreme_count) * 20, 100)
        auth_score = (first_score * 0.25 + (100 - template_score) * 0.30
                      + (100 - exaggeration_score) * 0.25 + min(length / 10, 100) * 0.20)
        scores["authenticity"] = auth_score

        # 4. 实用性
        has_comparison = any(w in text_lower for w in self.COMPARISON_WORDS)
        has_advice = any(w in text_lower for w in self.ADVICE_WORDS)
        has_scenario = any(w in text_lower for w in self.SCENARIO_WORDS)
        useful_score = (30 if has_comparison else 0) + (35 if has_advice else 0) + (20 if has_scenario else 0)
        if not has_comparison and not has_advice and not has_scenario:
            useful_score = 10 if length < 20 else (25 if length < 50 else 35)
        scores["usefulness"] = useful_score

        # 综合
        weights = {"informativeness": 0.30, "consistency": 0.25, "authenticity": 0.25, "usefulness": 0.20}
        overall = sum(scores[k] * weights[k] for k in weights)
        if length < 10:
            overall *= 0.5
            reasons.append("too short")
        elif length < 25:
            overall *= 0.8

        is_high_quality = overall >= 40.0
        return QualityResult(overall_score=round(overall, 1), is_high_quality=is_high_quality, reasons=reasons)


# ── 均衡采样 ─────────────────────────────────────────────────────

def balanced_sampling(df: pd.DataFrame, target_total: int, group_col: str) -> pd.DataFrame:
    """动态 max_per_group 均衡采样"""
    group_counts = df[group_col].value_counts()
    n_groups = len(group_counts)
    base_quota = target_total // n_groups

    low, high = base_quota, int(group_counts.max())
    while low < high:
        mid = (low + high) // 2
        total = sum(min(c, mid) for c in group_counts)
        if total >= target_total:
            high = mid
        else:
            low = mid + 1

    max_per_group = low
    print(f"  Unique {group_col}: {n_groups}")
    print(f"  Target per group: ~{base_quota}")
    print(f"  Max per group (dynamic): {max_per_group}")

    sampled = []
    for gval, group in df.groupby(group_col):
        group_sorted = group.sort_values("_quality_score", ascending=False)
        n_take = min(len(group_sorted), max_per_group)
        sampled.append(group_sorted.head(n_take))

    result = pd.concat(sampled)
    print(f"  Step1 (capped): {len(result):,} 条")
    if len(result) > target_total:
        result = result.sort_values("_quality_score", ascending=False).head(target_total)
        print(f"    -> truncated to {target_total:,}")
    return result.reset_index(drop=True)


# ── 完整处理示例（以 Amazon 为例）────────────────────────────────

def process_amazon_example(input_path: str, output_dir: str):
    """Amazon VOC 处理示例"""
    df = pd.read_excel(input_path)
    print(f"原始数据: {len(df):,} 条")

    # Step 1: 去重
    before = len(df)
    dedup_cols = ["Asin", "Author", "Content", "Date"]
    available = [c for c in dedup_cols if c in df.columns]
    for c in available:
        df[c] = df[c].fillna("").astype(str)
    df = df.drop_duplicates(subset=available, keep="first")
    print(f"去重: {before:,} -> {len(df):,}")

    # Step 2: 质量评估
    scorer = EnglishReviewQualityScorer()
    scores, flags = [], []
    for _, row in df.iterrows():
        result = scorer.score(row.get("Content", ""), row.get("Rating", None))
        scores.append(result.overall_score)
        flags.append(result.is_high_quality)
    df["_quality_score"] = scores
    df["_is_high_quality"] = flags
    print(f"高质量: {sum(flags):,} / {len(df):,} ({sum(flags)/len(df)*100:.1f}%)")

    # Step 3: 均衡采样 -> 20万
    sampled = balanced_sampling(df, 200_000, "Asin")

    # Step 4: 输出
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sampled.to_csv(f"{output_dir}/amazon_voc_200k.csv", index=False)
    print(f"输出: {output_dir}/amazon_voc_200k.csv ({len(sampled):,} 条)")

    return sampled


if __name__ == "__main__":
    # 示例调用
    process_amazon_example(
        input_path="/path/to/amazon_reviews.xlsx",
        output_dir="/path/to/output/"
    )
```

### 4.2 新增数据源适配模板

新增一个数据源时，只需修改以下 4 处：

```python
def process_new_source(input_path: str, output_dir: str):
    df = pd.read_excel(input_path)  # 或 read_csv

    # 1. 【改】文本列名和合并逻辑
    df["_review_text"] = df["评论列A"].fillna("") + "\n" + df["评论列B"].fillna("")
    df = df[df["_review_text"].str.strip() != ""]

    # 2. 【改】系统文本过滤（可选）
    SYSTEM_KEYWORDS = ["模板词1", "模板词2"]
    df = df[~df["_review_text"].str.lower().str.contains("|".join(SYSTEM_KEYWORDS))]

    # 3. 【改】去重键
    df = df.drop_duplicates(subset=["作者列", "文本列", "时间列"], keep="first")

    # 4. 【改】分组列和采样目标
    # 质量评估（通用）
    scorer = EnglishReviewQualityScorer()
    df["_quality_score"] = [scorer.score(t, r).overall_score for t, r in zip(df["_review_text"], df.get("评分列", [None]*len(df)))]
    df["_is_high_quality"] = df["_quality_score"] >= 40

    # 均衡采样（通用）
    sampled = balanced_sampling(df, target_total=50000, group_col="SPU列名")

    sampled.to_csv(f"{output_dir}/new_source_sampled.csv", index=False)
    return sampled
```

---

## 五、参数配置速查

| 参数 | 默认值 | 说明 | 调节建议 |
|------|--------|------|---------|
| `quality_threshold` | 40 | 高质量阈值 | 中文数据可降至 35；英文短评场景可维持 40 |
| `max_per_group` | 动态计算 | 每组上限 | 数据极度不均衡时，手动指定更严格的值 |
| `min_per_group` | 不适用 | 最低保障 | 当前算法通过保留全部数据自然保障（无硬性丢弃） |
| `target_total` | 按需 | 采样目标总量 | Amazon 20万 / Trustpilot 10万 / Zendesk 5万 |

---

## 六、历史执行记录

### 2026-04-22 执行批次

| 数据源 | 原始数据 | 有效数据 | 高质量率 | 采样后 | SPU/Group 数 | 平均质量分 |
|--------|---------|---------|---------|--------|-------------|-----------|
| Amazon | 664,317 | 437,582 | 95.6% | 200,000 | 6,775 ASINs | 66.6 |
| Trustpilot | 204,425 | 193,248 | 99.6% | 100,000 | 109 domains | 63.0 |
| Reddit | 13,234 | 6,297 | 98.7% | 6,213 | 571 SPU | 61.2 |
| Zendesk Momcozy | 353,092 | 48,900 | 98.2% | 48,005 | 297 SPU | 60.8 |
| **合计** | — | — | — | **354,218** | — | — |

---

## 七、常见问题

**Q: 质量分普遍偏高（>95% 高质量），阈值是否太宽松？**
> 英文 VOC 数据本身质量较高，且评分器针对亚马逊评论做了适配（短评不严厉惩罚）。如需更严格筛选，可将阈值提高到 50 或 55。

**Q: 如何处理中文 VOC 数据？**
> 当前评分器基于英文特征词。中文数据需要：
> 1. 将 POS/NEG/方面关键词替换为中文同义词
> 2. 使用 `jieba` 替代 `re.findall(r'\b[a-z]+\b')` 进行分词
> 3. 结构化标记替换为中文连接词（但是/然而/因为/所以/首先/其次）

**Q: SPU 分组列有大量空值怎么办？**
> 先填充空值为 "UNKNOWN" 或直接过滤掉空值行。过滤前检查空值占比，如果超过 30% 可能需要从其他列推导 SPU（如从标题/内容中提取产品型号）。

**Q: 动态 max_per_group 计算出的上限仍然很大？**
> 说明数据极度不均衡（少数 group 占绝大多数）。可以：
> 1. 手动设置更严格的 `max_per_group`（如 target_total / n_groups * 2）
> 2. 或舍弃样本过少的 group（<5 条的 group 直接过滤）

---

*文档版本: 2026-04-22*
*关联脚本: `merge_and_filter.py`, `sample_reddit_trustpilot.py`, `sample_zendesk_momcozy.py`*

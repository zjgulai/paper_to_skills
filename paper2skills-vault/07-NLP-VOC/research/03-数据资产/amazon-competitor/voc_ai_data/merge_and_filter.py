"""亚马逊 VOC 数据合并 + 质量筛选 + SPU 均衡采样

流程:
1. 遍历所有 xlsx，识别评论型文件（含 Asin/Content/Rating 列）
2. 合并所有评论数据
3. 去重（Asin + Author + Content + Date）
4. ReviewQuality 质量评估（英文适配，使用 Content 列）
5. SPU 均衡采样 → 20 万条高质量评论
6. 输出合并结果到 /tmp/amazon_voc_merged/
"""

import glob
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. 文件扫描与合并
# ---------------------------------------------------------------------------

REVIEW_SIGNATURE_COLS = {"Asin", "Content", "Rating"}


def is_review_file(df: pd.DataFrame) -> bool:
    """判断是否为评论型文件"""
    return REVIEW_SIGNATURE_COLS.issubset(set(df.columns))


def scan_and_merge(base_dir: str) -> pd.DataFrame:
    """扫描目录，合并所有评论型 xlsx"""
    all_files = sorted(glob.glob(os.path.join(base_dir, "*.xlsx")))
    print(f"扫描到 {len(all_files)} 个 xlsx 文件")

    review_files = []
    skip_files = []
    total_rows = 0

    for f in all_files:
        try:
            df = pd.read_excel(f)
            if is_review_file(df):
                basename = os.path.basename(f).replace(".xlsx", "")
                df["_source_file"] = basename
                review_files.append(df)
                total_rows += len(df)
            else:
                skip_files.append(os.path.basename(f))
        except Exception as e:
            print(f"  跳过 {os.path.basename(f)}: {e}")
            skip_files.append(os.path.basename(f))

    print(f"评论型文件: {len(review_files)} 个, 跳过: {len(skip_files)} 个")
    print(f"总评论数(去重前): {total_rows:,}")

    if not review_files:
        return pd.DataFrame()

    all_cols = set()
    for df in review_files:
        all_cols.update(df.columns)

    aligned = []
    for df in review_files:
        for col in all_cols:
            if col not in df.columns:
                df[col] = None
        aligned.append(df[list(all_cols)])

    merged = pd.concat(aligned, ignore_index=True)
    print(f"合并后行数: {len(merged):,}")
    return merged


# ---------------------------------------------------------------------------
# 2. 去重
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    dedup_cols = ["Asin", "Author", "Content", "Date"]
    available_cols = [c for c in dedup_cols if c in df.columns]

    if len(available_cols) < 2:
        print(f"警告: 去重列不足，可用: {available_cols}")
        return df

    for c in available_cols:
        df[c] = df[c].fillna("").astype(str)

    df_dedup = df.drop_duplicates(subset=available_cols, keep="first")
    after = len(df_dedup)
    print(f"去重: {before:,} → {after:,} (去重率 {(before - after) / before * 100:.1f}%)")
    return df_dedup.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 质量评估（英文适配版，针对亚马逊评论优化）
# ---------------------------------------------------------------------------

@dataclass
class QualityResult:
    overall_score: float
    is_high_quality: bool
    reasons: list


class EnglishReviewQualityScorer:
    """英文评论质量评分器 — 针对亚马逊评论优化

    亚马逊评论特点:
    - 原文就是英文，用 Content 列
    - 大量短评（"Great product!"）也是有效反馈
    - Rating 与文本情感不一定强相关（有些用户只写简短好评）
    """

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

    TEMPLATE_PHRASES = [
        "very good, highly recommend",
        "fast shipping, very satisfied",
        "good quality, worth buying",
        "very satisfied, five stars",
        "product is good, shipping is fast",
    ]

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

        # 1. 信息丰富度 (0-100)
        # 长度分（对数缩放）
        length_score = min(np.log1p(length) / np.log1p(500) * 100, 100)

        # 方面覆盖
        aspect_keywords = {
            "quality", "material", "comfort", "design", "size", "fit",
            "function", "performance", "speed", "noise", "smell", "price",
            "value", "packaging", "shipping", "delivery", "service",
            "use", "experience", "feel", "detail", "workmanship",
            "battery", "charge", "suction", "flange", "portable",
        }
        aspect_count = sum(1 for kw in aspect_keywords if kw in text_lower)
        aspect_score = min(aspect_count / 3 * 100, 100)

        # 词汇多样性
        unique_ratio = len(set(words)) / word_count
        vocab_score = min(unique_ratio * 200, 100)

        # 结构化标记
        struct_count = sum(1 for w in self.STRUCTURAL_MARKERS if w in text_lower)
        struct_score = min(struct_count / 2 * 100, 100)

        info_score = length_score * 0.25 + aspect_score * 0.30 + vocab_score * 0.25 + struct_score * 0.20
        scores["informativeness"] = info_score

        # 2. 评分一致性 (0-100) — 放宽判断
        consistency_score = 70.0  # 默认中等偏上（避免短评被误判）
        if rating is not None and pd.notna(rating):
            pos_count = sum(1 for w in self.POS_WORDS if w in text_lower)
            neg_count = sum(1 for w in self.NEG_WORDS if w in text_lower)

            if pos_count > neg_count:
                text_sent = 1
            elif neg_count > pos_count:
                text_sent = -1
            else:
                text_sent = 0  # 无明显情感词

            if rating >= 4:
                rating_sent = 1
            elif rating <= 2:
                rating_sent = -1
            else:
                rating_sent = 0

            if text_sent == 0:
                # 无明显情感词 → 中性，不扣分
                consistency_score = 80.0
            elif text_sent == rating_sent:
                consistency_score = 100.0
            else:
                # 有明确矛盾
                consistency_score = 20.0
                reasons.append("rating-text contradiction")

        scores["consistency"] = consistency_score

        # 3. 语言真实性 (0-100)
        template_score = 0.0
        for phrase in self.TEMPLATE_PHRASES:
            if phrase in text_lower:
                template_score = 100.0
                reasons.append("template phrase detected")
                break

        first_person = text_lower.count(" i ") + text_lower.count(" my ") + text_lower.count(" me ")
        first_ratio = first_person / word_count
        first_score = min(first_ratio * 500, 100)

        exaggeration = text.count("!!") + text.count("!!!")
        extreme_words = {"worst ever", "scam", "fraud", " garbage "}
        extreme_count = sum(1 for w in extreme_words if w in text_lower)
        exaggeration_score = min((exaggeration + extreme_count) * 20, 100)

        auth_score = first_score * 0.25 + (100 - template_score) * 0.30 + (100 - exaggeration_score) * 0.25 + min(length / 10, 100) * 0.20
        scores["authenticity"] = auth_score

        # 4. 实用性 (0-100)
        has_comparison = any(w in text_lower for w in self.COMPARISON_WORDS)
        has_advice = any(w in text_lower for w in self.ADVICE_WORDS)
        has_scenario = any(w in text_lower for w in self.SCENARIO_WORDS)

        useful_score = (30 if has_comparison else 0) + (35 if has_advice else 0) + (20 if has_scenario else 0)
        if not has_comparison and not has_advice and not has_scenario:
            if length < 20:
                useful_score = 10
            elif length < 50:
                useful_score = 25
            else:
                useful_score = 35
        scores["usefulness"] = useful_score

        # 综合质量分
        weights = {"informativeness": 0.30, "consistency": 0.25, "authenticity": 0.25, "usefulness": 0.20}
        overall = sum(scores[k] * weights[k] for k in weights)

        # 极短文本适度惩罚（不要太严厉）
        if length < 10:
            overall *= 0.5
            reasons.append("too short")
        elif length < 25:
            overall *= 0.8

        # Verified Purchase 加分
        # (在调用处处理)

        is_high_quality = overall >= 40.0

        return QualityResult(
            overall_score=round(overall, 1),
            is_high_quality=is_high_quality,
            reasons=reasons,
        )


# ---------------------------------------------------------------------------
# 4. SPU 均衡采样
# ---------------------------------------------------------------------------

def balanced_sampling(df: pd.DataFrame, target_total: int = 200000) -> pd.DataFrame:
    """按 SPU (Asin) 均衡采样

    策略:
    - 动态计算每 ASIN 上限: 二分查找最小的 max_per_asin，使得
      sum(min(各ASIN可用数, max_per_asin)) >= target_total
    - 这样每个 ASIN 有硬上限，同时保证总行数达到 target_total
    - 截断时全局按质量分排序，优先保留高质量评论
    """
    asin_counts = df["Asin"].value_counts()
    n_asins = len(asin_counts)
    base_quota = target_total // n_asins

    # 二分查找: 最小的 max_per_asin 使得 Step1 总行数 >= target_total
    low, high = base_quota, int(asin_counts.max())
    while low < high:
        mid = (low + high) // 2
        total = sum(min(c, mid) for c in asin_counts)
        if total >= target_total:
            high = mid
        else:
            low = mid + 1

    max_per_asin = low

    print(f"Unique ASINs: {n_asins}")
    print(f"Target per ASIN: ~{base_quota}")
    print(f"Max per ASIN (dynamic): {max_per_asin}")
    print(f"ASIN distribution: min={asin_counts.min()}, max={asin_counts.max()}, median={asin_counts.median()}")

    sampled = []
    for asin, group in df.groupby("Asin"):
        group_sorted = group.sort_values("_quality_score", ascending=False)
        n_take = min(len(group_sorted), max_per_asin)
        sampled.append(group_sorted.head(n_take))

    result = pd.concat(sampled)
    step1_total = len(result)
    print(f"Step1 (per-ASIN capped): {step1_total:,} 条")

    # 截断到 target_total（全局按质量分，不会超过 per-ASIN 上限）
    if step1_total > target_total:
        result = result.sort_values("_quality_score", ascending=False).head(target_total)
        print(f"  → truncated to {target_total:,} (by quality score)")
    elif step1_total < target_total:
        # 理论上不会发生（二分查找保证 >= target_total）
        print(f"  WARNING: Step1 {step1_total:,} < target {target_total:,}")

    print(f"采样后: {len(result):,} 条")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def main():
    base_dir = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/voc_ai_data"
    output_dir = "/tmp/amazon_voc_merged"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("亚马逊 VOC 数据合并 + 质量筛选 + SPU 均衡采样")
    print("=" * 70)

    # Step 1: 合并
    print("\n--- Step 1: 合并所有评论文件 ---")
    df = scan_and_merge(base_dir)
    if df.empty:
        print("无评论数据！")
        return

    core_cols = ["Asin", "Title", "English Title", "Content", "English Content",
                 "Verified Purchase", "Model", "Rating", "Helpful", "Author",
                 "Nation", "Date", "_source_file"]
    keep_cols = [c for c in core_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Step 2: 去重
    print("\n--- Step 2: 去重 ---")
    df = deduplicate(df)

    # Step 3: 质量评估
    print("\n--- Step 3: 质量评估 ---")
    scorer = EnglishReviewQualityScorer()

    # 使用 Content 列（原文就是英文）
    text_col = "Content"

    quality_scores = []
    quality_flags = []
    for idx, row in df.iterrows():
        text = row.get(text_col, "")
        rating = row.get("Rating", None)
        result = scorer.score(text, rating)

        # Verified Purchase 加分
        verified = row.get("Verified Purchase", "")
        if str(verified).lower() in {"true", "yes", "1", "verified"}:
            result.overall_score = min(100, result.overall_score + 5)
            if result.overall_score >= 40:
                result.is_high_quality = True

        quality_scores.append(result.overall_score)
        quality_flags.append(result.is_high_quality)

    df["_quality_score"] = quality_scores
    df["_is_high_quality"] = quality_flags

    high_quality = df[df["_is_high_quality"]].copy()
    print(f"高质量评论: {len(high_quality):,} / {len(df):,} ({len(high_quality)/len(df)*100:.1f}%)")

    # 质量分布
    print(f"质量分分布: mean={df['_quality_score'].mean():.1f}, median={df['_quality_score'].median():.1f}")
    for low, high in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        cnt = sum((df['_quality_score'] >= low) & (df['_quality_score'] < high))
        print(f"  {low}-{high}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # Step 4: SPU 均衡采样 → 20 万
    print("\n--- Step 4: SPU 均衡采样 → 200,000 ---")

    # 从全部数据中按质量排序采样
    sampled = balanced_sampling(df, 200000)

    # Step 5: 输出
    print("\n--- Step 5: 输出结果 ---")

    # 5.1 全量数据
    df.to_csv(f"{output_dir}/amazon_voc_all_deduped.csv", index=False)
    print(f"全量去重数据: {output_dir}/amazon_voc_all_deduped.csv ({len(df):,} 条)")

    # 5.2 高质量数据
    high_quality.to_csv(f"{output_dir}/amazon_voc_high_quality.csv", index=False)
    print(f"高质量数据: {output_dir}/amazon_voc_high_quality.csv ({len(high_quality):,} 条)")

    # 5.3 采样数据（20万）
    sampled.to_csv(f"{output_dir}/amazon_voc_200k_balanced.csv", index=False)
    print(f"均衡采样数据: {output_dir}/amazon_voc_200k_balanced.csv ({len(sampled):,} 条)")

    # 5.4 统计报告
    report = {
        "total_files": len(glob.glob(os.path.join(base_dir, "*.xlsx"))),
        "total_reviews_deduped": len(df),
        "high_quality_count": int(len(high_quality)),
        "high_quality_rate": round(len(high_quality) / len(df) * 100, 1),
        "sampled_count": int(len(sampled)),
        "unique_asins": int(df["Asin"].nunique()),
        "sampled_asins": int(sampled["Asin"].nunique()),
        "avg_quality_score": round(float(sampled["_quality_score"].mean()), 1),
        "rating_distribution": {str(k): int(v) for k, v in sampled["Rating"].value_counts().sort_index().items()} if "Rating" in sampled.columns else {},
        "asin_distribution": {
            "min": int(sampled["Asin"].value_counts().min()),
            "max": int(sampled["Asin"].value_counts().max()),
            "median": float(sampled["Asin"].value_counts().median()),
            "mean": round(float(sampled["Asin"].value_counts().mean()), 1),
        },
    }

    with open(f"{output_dir}/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n统计报告: {output_dir}/report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("处理完成 ✓")
    print("=" * 70)

    return df, high_quality, sampled


if __name__ == "__main__":
    main()

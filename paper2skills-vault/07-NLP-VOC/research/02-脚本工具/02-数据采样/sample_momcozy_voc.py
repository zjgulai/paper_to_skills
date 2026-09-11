"""Momcozy VOC 数据高质量抽样

对 Momcozy 亚马逊商品评论进行质量评估 + 均衡采样，
输出高质量样本到 03-数据资产/高质量数据源。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class QualityResult:
    overall_score: float
    is_high_quality: bool
    reasons: list


class EnglishReviewQualityScorer:
    """英文评论质量评分器"""

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

        info_score = (
            length_score * 0.25
            + aspect_score * 0.30
            + vocab_score * 0.25
            + struct_score * 0.20
        )
        scores["informativeness"] = info_score

        # 2. 评分一致性
        consistency_score = 70.0
        if rating is not None and pd.notna(rating):
            pos_count = sum(1 for w in self.POS_WORDS if w in text_lower)
            neg_count = sum(1 for w in self.NEG_WORDS if w in text_lower)

            if pos_count > neg_count:
                text_sent = 1
            elif neg_count > pos_count:
                text_sent = -1
            else:
                text_sent = 0

            if rating >= 4:
                rating_sent = 1
            elif rating <= 2:
                rating_sent = -1
            else:
                rating_sent = 0

            if text_sent == 0:
                consistency_score = 80.0
            elif text_sent == rating_sent:
                consistency_score = 100.0
            else:
                consistency_score = 20.0
                reasons.append("rating-text contradiction")

        scores["consistency"] = consistency_score

        # 3. 语言真实性
        template_score = 0.0
        for phrase in self.TEMPLATE_PHRASES:
            if phrase in text_lower:
                template_score = 100.0
                reasons.append("template phrase detected")
                break

        first_person = (
            text_lower.count(" i ")
            + text_lower.count(" my ")
            + text_lower.count(" me ")
        )
        first_ratio = first_person / word_count
        first_score = min(first_ratio * 500, 100)

        exaggeration = text.count("!!") + text.count("!!!")
        extreme_words = {"worst ever", "scam", "fraud", " garbage "}
        extreme_count = sum(1 for w in extreme_words if w in text_lower)
        exaggeration_score = min((exaggeration + extreme_count) * 20, 100)

        auth_score = (
            first_score * 0.25
            + (100 - template_score) * 0.30
            + (100 - exaggeration_score) * 0.25
            + min(length / 10, 100) * 0.20
        )
        scores["authenticity"] = auth_score

        # 4. 实用性
        has_comparison = any(w in text_lower for w in self.COMPARISON_WORDS)
        has_advice = any(w in text_lower for w in self.ADVICE_WORDS)
        has_scenario = any(w in text_lower for w in self.SCENARIO_WORDS)

        useful_score = (
            (30 if has_comparison else 0)
            + (35 if has_advice else 0)
            + (20 if has_scenario else 0)
        )
        if not has_comparison and not has_advice and not has_scenario:
            if length < 20:
                useful_score = 10
            elif length < 50:
                useful_score = 25
            else:
                useful_score = 35
        scores["usefulness"] = useful_score

        weights = {
            "informativeness": 0.30,
            "consistency": 0.25,
            "authenticity": 0.25,
            "usefulness": 0.20,
        }
        overall = sum(scores[k] * weights[k] for k in weights)

        if length < 10:
            overall *= 0.5
            reasons.append("too short")
        elif length < 25:
            overall *= 0.8

        is_high_quality = overall >= 40.0

        return QualityResult(
            overall_score=round(overall, 1),
            is_high_quality=is_high_quality,
            reasons=reasons,
        )


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
    print(
        f"  Group distribution: min={group_counts.min()}, max={group_counts.max()}, median={group_counts.median()}"
    )

    sampled = []
    for gval, group in df.groupby(group_col):
        group_sorted = group.sort_values("_quality_score", ascending=False)
        n_take = min(len(group_sorted), max_per_group)
        sampled.append(group_sorted.head(n_take))

    result = pd.concat(sampled)
    step1_total = len(result)
    print(f"  Step1 (per-group capped): {step1_total:,} 条")

    if step1_total > target_total:
        result = result.sort_values("_quality_score", ascending=False).head(
            target_total
        )
        print(f"    -> truncated to {target_total:,}")

    print(f"  采样后: {len(result):,} 条")
    return result.reset_index(drop=True)


SYSTEM_KEYWORDS = [
    "call from:", "call to:", "time of call:", "you've received a new",
    "rma status:", "rma #", "order #", "pending approval",
    "check its details below", "refund request", "return request",
    "tracking number:", "shipping confirmation",
]


def is_system_text(text: str) -> bool:
    text_lower = str(text).lower()
    return any(kw in text_lower for kw in SYSTEM_KEYWORDS)


def process_momcozy_voc(input_path: str, output_dir: str):
    print("=" * 70)
    print("Momcozy VOC 数据质量评估 + 均衡采样")
    print("=" * 70)

    df = pd.read_excel(input_path, sheet_name="VOC明细")
    print(f"\n原始数据: {len(df):,} 条")

    # Step 1: 提取有效文本（买家评论）
    df["_review_text"] = df["买家评论"].fillna("").astype(str).str.strip()
    df = df[df["_review_text"] != ""].copy()
    print(f"有评论文本: {len(df):,} 条")

    # Step 2: 过滤系统通知/模板文本
    before = len(df)
    df = df[~df["_review_text"].apply(is_system_text)].copy()
    after = len(df)
    print(f"过滤系统文本: {before:,} -> {after:,} (过滤率 {(before - after) / before * 100:.1f}%)")

    # Step 3: 去重（ASIN + 买家评论）
    before = len(df)
    dedup_cols = ["ASIN", "买家评论"]
    for c in dedup_cols:
        df[c] = df[c].fillna("").astype(str)
    df = df.drop_duplicates(subset=dedup_cols, keep="first")
    after = len(df)
    print(f"去重: {before:,} -> {after:,} (去重率 {(before - after) / before * 100:.1f}%)")

    # Step 4: 质量评估
    print("\n--- 质量评估 ---")
    scorer = EnglishReviewQualityScorer()

    quality_scores = []
    quality_flags = []
    for _, row in df.iterrows():
        text = row.get("_review_text", "")
        rating = row.get("星级评分", None)
        if rating == 0 or pd.isna(rating):
            rating = None
        result = scorer.score(text, rating)
        quality_scores.append(result.overall_score)
        quality_flags.append(result.is_high_quality)

    df["_quality_score"] = quality_scores
    df["_is_high_quality"] = quality_flags

    high_quality = df[df["_is_high_quality"]].copy()
    print(f"高质量评论: {len(high_quality):,} / {len(df):,} ({len(high_quality) / len(df) * 100:.1f}%)")
    print(f"质量分分布: mean={df['_quality_score'].mean():.1f}, median={df['_quality_score'].median():.1f}")
    for low, high_b in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        cnt = sum((df["_quality_score"] >= low) & (df["_quality_score"] < high_b))
        print(f"  {low}-{high_b}: {cnt:,} ({cnt / len(df) * 100:.1f}%)")

    # Step 5: 按品类均衡采样 -> 目标 2 万条
    target = min(20000, len(high_quality))
    print(f"\n--- 品类均衡采样 -> {target:,} ---")

    # 使用四级类目名称作为分组
    df_with_category = df[df["四级类目名称"].notna() & (df["四级类目名称"] != "")].copy()
    sampled = balanced_sampling(df_with_category, target, "四级类目名称")

    # Step 6: 输出
    print("\n--- 输出结果 ---")
    output_path = Path(output_dir) / "momcozy_voc_high_quality_sampled.csv"
    sampled.to_csv(output_path, index=False)
    print(f"Momcozy 高质量采样数据: {output_path} ({len(sampled):,} 条)")

    # 统计信息
    print(f"\n=== 采样统计 ===")
    print(f"品类覆盖: {sampled['四级类目名称'].nunique()} / {df['四级类目名称'].nunique()}")
    print(f"SPU覆盖: {sampled['SPU名称'].nunique()} / {df['SPU名称'].nunique()}")
    print(f"国家覆盖: {sampled['国家'].nunique()} / {df['国家'].nunique()}")
    print(f"平均质量分: {sampled['_quality_score'].mean():.1f}")
    print(f"评分分布:")
    for star in sorted(sampled["星级评分"].unique()):
        cnt = (sampled["星级评分"] == star).sum()
        print(f"  {star}星: {cnt:,} ({cnt / len(sampled) * 100:.1f}%)")

    report = {
        "source": "momcozy_voc",
        "input_file": input_path,
        "total_raw": int(len(df) + (before - after)),
        "has_text": int(len(df)),
        "total_deduped": int(after),
        "high_quality_count": int(len(high_quality)),
        "high_quality_rate": round(len(high_quality) / len(df) * 100, 1),
        "sampled_count": int(len(sampled)),
        "target_count": target,
        "unique_categories": int(sampled["四级类目名称"].nunique()),
        "unique_spus": int(sampled["SPU名称"].nunique()),
        "unique_countries": int(sampled["国家"].nunique()),
        "avg_quality_score": round(float(sampled["_quality_score"].mean()), 1),
        "rating_distribution": {
            str(k): int(v)
            for k, v in sampled["星级评分"].value_counts().sort_index().items()
        },
        "category_distribution": {
            str(k): int(v)
            for k, v in sampled["四级类目名称"].value_counts().head(20).items()
        },
        "country_distribution": {
            str(k): int(v)
            for k, v in sampled["国家"].value_counts().head(10).items()
        },
    }

    report_path = Path(output_dir) / "momcozy_voc_sampling_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告: {report_path}")
    return sampled


def main():
    input_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/momcozy/momcozy_voc_data/Momcozy_VOC明细2504-2604.xlsx"
    output_dir = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    process_momcozy_voc(input_path, output_dir)

    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

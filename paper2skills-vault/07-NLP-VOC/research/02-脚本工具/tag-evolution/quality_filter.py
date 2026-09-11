"""质量筛选（Phase 1.2）

复用 ReviewQualityScorer，对统一 VOCRecord 进行质量评估，
输出高质量子集用于后续萃取打标。
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
    """英文评论质量评分器（复用自 sample_momcozy_voc.py）"""

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


def process_quality_filter():
    print("=" * 70)
    print("Phase 1.2: 质量筛选")
    print("=" * 70)

    input_path = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling/phase1_1_unified_voc_records.jsonl"
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取统一 VOC 记录
    print("\n--- 读取统一 VOC 记录 ---")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    print(f"  总计: {len(records):,} 条")

    # 质量评估
    print("\n--- 质量评估 ---")
    scorer = EnglishReviewQualityScorer()

    has_existing_score = 0
    scored_from_scratch = 0
    high_quality_count = 0
    quality_scores = []

    for i, rec in enumerate(records):
        existing_score = rec.get("_quality_score")
        existing_flag = rec.get("_is_high_quality")

        if existing_score is not None and existing_flag is not None:
            # 复用已有质量分
            has_existing_score += 1
            rec["_quality_score"] = float(existing_score)
            rec["_is_high_quality"] = bool(existing_flag)
        else:
            # 重新评分（主要是 Reddit）
            result = scorer.score(rec["text"], rec.get("rating"))
            rec["_quality_score"] = result.overall_score
            rec["_is_high_quality"] = result.is_high_quality
            scored_from_scratch += 1

        if rec["_is_high_quality"]:
            high_quality_count += 1
        quality_scores.append(rec["_quality_score"])

    print(f"  复用已有质量分: {has_existing_score:,}")
    print(f"  重新评分: {scored_from_scratch:,}")
    print(f"  高质量: {high_quality_count:,} / {len(records):,} ({high_quality_count/len(records)*100:.1f}%)")

    # 质量分分布
    scores = pd.Series(quality_scores)
    print(f"\n  质量分分布: mean={scores.mean():.1f}, median={scores.median():.1f}")
    for low, high in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]:
        cnt = sum((scores >= low) & (scores < high))
        print(f"    {low}-{high}: {cnt:,} ({cnt/len(scores)*100:.1f}%)")

    # 保存高质量子集
    print("\n--- 输出高质量子集 ---")
    high_quality = [r for r in records if r["_is_high_quality"]]

    output_path = output_dir / "phase1_2_high_quality_voc.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in high_quality:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  输出: {output_path} ({len(high_quality):,} 条)")

    # 各数据源高质量占比
    print(f"\n  各数据源高质量占比:")
    source_stats = {}
    for r in records:
        src = r["data_source"]
        if src not in source_stats:
            source_stats[src] = {"total": 0, "hq": 0}
        source_stats[src]["total"] += 1
        if r["_is_high_quality"]:
            source_stats[src]["hq"] += 1

    for src, stat in sorted(source_stats.items()):
        rate = stat["hq"] / stat["total"] * 100
        print(f"    {src}: {stat['hq']:,} / {stat['total']:,} ({rate:.1f}%)")

    # 审计报告
    audit = {
        "phase": "1.2",
        "total_records": len(records),
        "has_existing_score": has_existing_score,
        "scored_from_scratch": scored_from_scratch,
        "high_quality_count": len(high_quality),
        "high_quality_rate": round(len(high_quality) / len(records) * 100, 1),
        "quality_score_mean": round(float(scores.mean()), 1),
        "quality_score_median": round(float(scores.median()), 1),
        "source_breakdown": {
            src: {"total": s["total"], "hq": s["hq"], "rate": round(s["hq"]/s["total"]*100, 1)}
            for src, s in source_stats.items()
        },
        "output_path": str(output_path),
    }
    audit_path = output_dir / "phase1_2_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 1.2 完成")
    print("=" * 70)


if __name__ == "__main__":
    process_quality_filter()

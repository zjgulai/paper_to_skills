"""增量打标（Phase 1.3 - Step 2）

对 v3.3 未覆盖的数据（Momcozy + 未匹配记录）使用关键词匹配进行增量打标。
加载 V3.0 标签字典（409 标签），执行关键词匹配 + 否定词检测。
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd


# ── 标签加载 ──────────────────────────────────────────────────────

def parse_keywords(keyword_str: str) -> list[str]:
    """解析关键词字符串（逗号/换行分隔）"""
    if pd.isna(keyword_str) or not str(keyword_str).strip():
        return []
    result = []
    for part in re.split(r"[,;\n]", str(keyword_str)):
        kw = part.strip().lower()
        if kw and len(kw) >= 2:
            result.append(kw)
    return result


def load_tag_dictionary() -> list[dict]:
    """从 V3.0 Excel 加载全部标签"""
    path = (
        Path(__file__).parent.parent.parent
        / "03-数据资产/原种子标签/SGCS_标签字典_VOC标签大宽表V3.0_v1.xlsx"
    )

    sheets = {
        "01_通用标签主表": "general",
        "02_吸奶器": "breast_pump",
        "03_内衣服饰": "innerwear",
        "04_家居家纺": "home_textile",
        "05_母婴综合护理": "baby_care",
        "06_喂养电器": "feeding_appliance",
        "07_智能母婴电器": "smart_baby",
    }

    all_tags = []
    for sheet_name, sheet_key in sheets.items():
        df = pd.read_excel(path, sheet_name=sheet_name)
        for _, row in df.iterrows():
            tag_id = str(row.get("标签ID", "")).strip()
            if not tag_id:
                continue

            tag_en = str(row.get("VOC标签（英文）", "")).strip()
            tag_cn = str(row.get("VOC标签（中文）", "")).strip()
            aipl = str(row.get("AIPL节点", "")).strip()
            sentiment = str(row.get("情感极性", "")).strip().lower()

            kws = parse_keywords(row.get("英文关键词/典型表达", ""))
            consumer_kws = parse_keywords(row.get("消费者习惯关键词/原话短语", ""))

            # 适用品线/品类
            lines = parse_keywords(row.get("适用产品品线", ""))
            categories = parse_keywords(row.get("适用产品品类", ""))

            all_tags.append(
                {
                    "tag_id": tag_id,
                    "tag_en": tag_en,
                    "tag_cn": tag_cn,
                    "aipl_node": aipl,
                    "sentiment_preset": sentiment if sentiment in ["positive", "negative", "neutral"] else "neutral",
                    "keywords": kws,
                    "consumer_keywords": consumer_kws,
                    "all_keywords": kws + consumer_kws,
                    "applicable_lines": lines,
                    "applicable_categories": categories,
                    "sheet": sheet_key,
                }
            )

    print(f"  加载标签: {len(all_tags)} 个")
    return all_tags


# ── 关键词匹配引擎 ────────────────────────────────────────────────

NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing",
    "n't", "dont", "doesnt", "didnt", "wouldnt", "couldnt",
    "shouldnt", "wont", "cant", "isnt", "arent", "wasnt",
    "werent", "hasnt", "havent", "hadnt",
}


def _precompile_patterns(tags: list[dict]) -> dict[str, list[tuple[str, Optional[re.Pattern]]]]:
    """预编译所有标签关键词的正则表达式（大幅提升速度）"""
    compiled = {}
    for tag in tags:
        patterns = []
        for kw in tag["all_keywords"]:
            kw_lower = kw.lower()
            if len(kw_lower) >= 4:
                patterns.append((kw_lower, re.compile(r"\b" + re.escape(kw_lower) + r"\b")))
            else:
                patterns.append((kw_lower, None))
        compiled[tag["tag_id"]] = patterns
    return compiled


def _check_negation_fast(text_lower: str, keyword: str, window: int = 15) -> bool:
    """快速否定词检测"""
    idx = text_lower.find(keyword.lower())
    if idx < 0:
        return False
    prefix = text_lower[max(0, idx - window) : idx]
    return any(neg in prefix for neg in NEGATION_WORDS)


def match_tags(text: str, tags: list[dict], compiled_patterns: Optional[dict] = None) -> list[dict]:
    """对单条文本匹配所有标签（支持预编译模式）"""
    text_lower = text.lower()
    matches = []

    for tag in tags:
        patterns = compiled_patterns.get(tag["tag_id"], []) if compiled_patterns else []
        if not patterns:
            # 回退：动态编译
            patterns = []
            for kw in tag["all_keywords"]:
                kw_lower = kw.lower()
                if len(kw_lower) >= 4:
                    patterns.append((kw_lower, re.compile(r"\b" + re.escape(kw_lower) + r"\b")))
                else:
                    patterns.append((kw_lower, None))

        matched = False
        match_kw = ""

        for kw_lower, pattern in patterns:
            if pattern is None:
                if kw_lower in text_lower:
                    if not _check_negation_fast(text_lower, kw_lower):
                        matched = True
                        match_kw = kw_lower
                        break
            else:
                if pattern.search(text_lower):
                    if not _check_negation_fast(text_lower, kw_lower):
                        matched = True
                        match_kw = kw_lower
                        break

        if matched:
            confidence = min(0.3 + len(match_kw) * 0.05, 0.9)
            matches.append(
                {
                    "tag_id": tag["tag_id"],
                    "tag_en": tag["tag_en"],
                    "tag_cn": tag["tag_cn"],
                    "aipl_node": tag["aipl_node"],
                    "sentiment_preset": tag["sentiment_preset"],
                    "sentiment_calibrated": 1.0 if tag["sentiment_preset"] == "positive" else (-1.0 if tag["sentiment_preset"] == "negative" else 0.0),
                    "confidence": round(confidence, 2),
                }
            )

    return matches


# ── 辅助推断 ──────────────────────────────────────────────────────

def derive_aipl_stage(labels: list[dict]) -> str:
    """从标签推断主 AIPL 阶段"""
    if not labels:
        return ""
    nodes = [l["aipl_node"] for l in labels if l["aipl_node"]]
    if not nodes:
        return ""
    # 取频率最高的节点
    counter = Counter(nodes)
    return counter.most_common(1)[0][0]


def derive_sentiment(labels: list[dict], rating: Optional[float]) -> tuple[float, str]:
    """推断情感极性和校准方式"""
    if not labels:
        if rating is not None:
            if rating >= 4:
                return 1.0, "rating_derived"
            elif rating <= 2:
                return -1.0, "rating_derived"
            else:
                return 0.0, "rating_derived"
        return 0.0, "unknown"

    # 基于标签预设情感
    pos = sum(1 for l in labels if l["sentiment_preset"] == "positive")
    neg = sum(1 for l in labels if l["sentiment_preset"] == "negative")

    if pos > neg:
        return 1.0, "preset"
    elif neg > pos:
        return -1.0, "preset"
    else:
        return 0.0, "preset"


def derive_proxy_nps(labels: list[dict], sentiment: float, rating: Optional[float]) -> str:
    """推断 Proxy NPS"""
    if rating is not None:
        if rating >= 4:
            return "promoter"
        elif rating <= 2:
            return "detractor"
        else:
            return "passive"

    # 基于情感
    if sentiment >= 0.5:
        return "promoter"
    elif sentiment <= -0.5:
        return "detractor"
    return "passive"


def derive_persona(text: str, labels: list[dict], sentiment: float) -> str:
    """推断用户画像"""
    text_lower = text.lower()

    # 简单规则推断
    tag_ens = [l["tag_en"] for l in labels]

    if any(t in tag_ens for t in ["price_concern", "discount_hunting", "value_for_money"]):
        return "value_driven"
    if any(t in tag_ens for t in ["community_recommendation", "social_proof", "influencer_driven"]):
        return "community_driven"
    if any(t in tag_ens for t in ["quality_focus", "detailed_review", "comparison_shopping"]):
        return "quality_explorer"
    if any(t in tag_ens for t in ["gift_purchase", "first_time_buyer", "brand_loyal"]):
        return "systematic_planner"

    # 基于文本特征
    if "recommend" in text_lower or "suggest" in text_lower:
        if len(text) > 200:
            return "quality_explorer"
    if "price" in text_lower or "cheap" in text_lower or "expensive" in text_lower:
        return "value_driven"
    if "compare" in text_lower or "versus" in text_lower or "vs" in text_lower:
        return "quality_explorer"

    return "unclassified"


def detect_brands(text: str) -> tuple[list[str], bool]:
    """检测品牌提及"""
    text_lower = text.lower()

    brand_keywords = {
        "momcozy": ["momcozy"],
        "elvie": ["elvie"],
        "medela": ["medela"],
        "spectra": ["spectra"],
        "willow": ["willow"],
        "lansinoh": ["lansinoh"],
        "haakaa": ["haakaa"],
        "philips avent": ["philips avent", "avent"],
        "dr brown's": ["dr brown"],
        "tommee tippee": ["tommee tippee"],
    }

    mentions = []
    for brand, kws in brand_keywords.items():
        if any(kw in text_lower for kw in kws):
            mentions.append(brand)

    comparison = any(w in text_lower for w in ["compared to", "versus", "vs", "better than", "worse than", "switch from", "other brand"])

    return mentions, comparison


# ── 主流程 ────────────────────────────────────────────────────────

def process_incremental_labeling():
    print("=" * 70)
    print("Phase 1.3 Step 2: 增量打标（关键词匹配）")
    print("=" * 70)

    # 1. 加载标签字典
    print("\n--- 加载标签字典 ---")
    tags = load_tag_dictionary()

    # 2. 加载待打标记录
    print("\n--- 加载待打标记录 ---")
    hq_path = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling/phase1_2_high_quality_voc.jsonl"
    unmatched_path = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling/phase1_3_unmatched_samples.jsonl"

    # Momcozy 数据
    momcozy_records = []
    unmatched_records = []

    with open(hq_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec["data_source"] == "momcozy":
                momcozy_records.append(rec)

    # 未匹配记录（Amazon + Reddit）
    if unmatched_path.exists():
        with open(unmatched_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                unmatched_records.append(rec)

    # 合并待打标
    to_label = momcozy_records + unmatched_records
    print(f"  Momcozy: {len(momcozy_records):,}")
    print(f"  未匹配: {len(unmatched_records):,}")
    print(f"  总计待打标: {len(to_label):,}")

    # 3. 预编译正则并执行打标
    print("\n--- 预编译正则表达式 ---")
    compiled = _precompile_patterns(tags)
    print(f"  预编译完成: {len(compiled)} 个标签")

    print("\n--- 增量打标 ---")
    labeled = []
    tag_counts = Counter()
    zero_tag = 0

    for i, rec in enumerate(to_label):
        if (i + 1) % 5000 == 0:
            print(f"  进度: {i + 1:,} / {len(to_label):,}")

        labels = match_tags(rec["text"], tags, compiled)
        n_tags = len(labels)

        if n_tags == 0:
            zero_tag += 1

        for l in labels:
            tag_counts[l["tag_id"]] += 1

        sentiment, sentiment_calib = derive_sentiment(labels, rec.get("rating"))
        proxy_nps = derive_proxy_nps(labels, sentiment, rec.get("rating"))
        persona = derive_persona(rec["text"], labels, sentiment)
        aipl_stage = derive_aipl_stage(labels)
        brand_mentions, brand_comparison = detect_brands(rec["text"])

        result = {
            "review_id": rec["review_id"],
            "text": rec["text"],
            "source_type": rec["source_type"],
            "platform": rec["platform"],
            "spu_code": rec.get("spu_code"),
            "asin": rec.get("asin"),
            "product_line": rec.get("product_line"),
            "category": rec.get("category"),
            "rating": rec.get("rating"),
            "language": rec.get("language", "en"),
            "data_source": rec["data_source"],
            "_quality_score": rec.get("_quality_score"),
            "labels": labels,
            "n_tags": n_tags,
            "aipl_stage": aipl_stage,
            "persona_derived": persona,
            "sentiment_polarity": sentiment,
            "sentiment_calibration": sentiment_calib,
            "proxy_nps": proxy_nps,
            "brand_mentions": brand_mentions,
            "brand_comparison": brand_comparison,
            "label_source": "incremental_keyword",
        }
        labeled.append(result)

    # 4. 输出
    print(f"\n--- 打标统计 ---")
    print(f"  总计打标: {len(labeled):,}")
    print(f"  零标签: {zero_tag:,} ({zero_tag/len(labeled)*100:.1f}%)")
    print(f"  有标签: {len(labeled) - zero_tag:,} ({(len(labeled)-zero_tag)/len(labeled)*100:.1f}%)")
    print(f"  命中标签种类: {len(tag_counts)} / {len(tags)}")

    # 标签频率 Top 20
    print(f"\n  高频标签 Top 20:")
    for tag_id, cnt in tag_counts.most_common(20):
        tag_info = next((t for t in tags if t["tag_id"] == tag_id), None)
        tag_name = tag_info["tag_en"] if tag_info else "?"
        print(f"    {tag_id} ({tag_name}): {cnt:,}")

    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling"
    output_path = output_dir / "phase1_3_incremental_labeled.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  输出: {output_path}")

    # 5. 合并 v3.3 转录 + 增量打标
    print(f"\n--- 合并全部打标结果 ---")
    transcribed_path = output_dir / "phase1_3_v33_transcribed.jsonl"

    all_labeled = []
    with open(transcribed_path, "r", encoding="utf-8") as f:
        for line in f:
            all_labeled.append(json.loads(line.strip()))
    all_labeled.extend(labeled)

    merged_path = output_dir / "phase1_3_all_sources_labeled.jsonl"
    with open(merged_path, "w", encoding="utf-8") as f:
        for r in all_labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  合并输出: {merged_path} ({len(all_labeled):,} 条)")

    # 6. 审计报告
    v33_count = sum(1 for r in all_labeled if r.get("label_source") == "v3.3_transcribed")
    inc_count = sum(1 for r in all_labeled if r.get("label_source") == "incremental_keyword")
    total_zero = sum(1 for r in all_labeled if r["n_tags"] == 0)
    total_with_tags = len(all_labeled) - total_zero

    source_stats = defaultdict(lambda: {"total": 0, "with_tags": 0, "zero": 0})
    for r in all_labeled:
        ds = r["data_source"]
        source_stats[ds]["total"] += 1
        if r["n_tags"] > 0:
            source_stats[ds]["with_tags"] += 1
        else:
            source_stats[ds]["zero"] += 1

    audit = {
        "phase": "1.3",
        "description": "萃取引擎打标（v3.3 转录 + 增量关键词匹配）",
        "total_labeled": len(all_labeled),
        "v3.3_transcribed": v33_count,
        "incremental_keyword": inc_count,
        "zero_tag_count": total_zero,
        "zero_tag_rate": round(total_zero / len(all_labeled) * 100, 1),
        "coverage_rate": round(total_with_tags / len(all_labeled) * 100, 1),
        "source_breakdown": {
            ds: {
                "total": s["total"],
                "with_tags": s["with_tags"],
                "zero": s["zero"],
                "coverage": round(s["with_tags"] / s["total"] * 100, 1) if s["total"] > 0 else 0,
            }
            for ds, s in source_stats.items()
        },
        "tag_hit_distribution": {
            str(k): v for k, v in sorted(Counter(r["n_tags"] for r in all_labeled).items())
        },
        "output_path": str(merged_path),
    }
    audit_path = output_dir / "phase1_3_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 1.3 完成")
    print("=" * 70)


if __name__ == "__main__":
    process_incremental_labeling()

"""VOC 标签自动优化闭环

自动化流程：
1. 零标签文本主题聚类 → 发现新主题
2. 关键词自动扩展推荐 → 基于同义词和共现词
3. 覆盖率监控 → 追踪各维度覆盖率趋势
4. 生成优化建议报告

Usage:
    python auto_optimize_loop.py --input labeling_output_v3.3 \
        --dict SGCS_VOC标签字典_V3.3.1_universal.xlsx \
        --output optimization_report.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_labeled_data(input_dir: Path):
    """加载打标结果"""
    records = []
    zero_label_texts = []

    for src_dir in sorted(input_dir.iterdir()):
        if not src_dir.is_dir() or src_dir.name.endswith(".json"):
            continue
        for batch_file in sorted(src_dir.glob("batch_*.jsonl")):
            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    records.append(r)
                    if r["n_tags"] == 0 and r.get("text_preview"):
                        zero_label_texts.append({
                            "text": r["text_preview"],
                            "source": src_dir.name,
                        })

    return records, zero_label_texts


def cluster_zero_label_topics(zero_texts: list[dict], top_n: int = 30):
    """零标签文本主题聚类（基于高频实词）"""
    # 停用词
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "and", "but", "or",
        "yet", "so", "if", "because", "although", "while", "where", "when",
        "that", "which", "who", "whom", "what", "this", "these", "those",
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you",
        "your", "yours", "yourself", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them",
        "their", "theirs", "themselves", "very", "really", "quite", "pretty",
        "fairly", "rather", "just", "only", "even", "also", "too", "well",
        "so", "still", "already", "yet", "always", "never", "often", "sometimes",
        "great", "good", "nice", "awesome", "amazing", "love", "loved", "like",
        "liked", "bad", "terrible", "horrible", "awful", "worst", "hate",
        "recommend", "recommended", "happy", "disappointed", "thanks", "thank",
        "amazon", "order", "ordered", "purchase", "purchased", "buy", "bought",
        "shipping", "shipped", "delivery", "delivered", "arrived", "package",
        "item", "product", "products", "company", "brand", "seller", "store",
        "review", "reviews", "star", "stars", "rating", "money", "dollar",
        "price", "cost", "paid", "pay", "return", "returned", "refund",
        "mom", "mother", "mum", "mama", "mommy", "dad", "father", "parent",
        "parents", "baby", "child", "children", "kid", "kids", "son", "daughter",
        "family", "day", "days", "week", "weeks", "month", "months", "year",
        "years", "time", "hour", "hours", "today", "yesterday", "tomorrow",
    }

    # 提取词
    word_counts = Counter()
    bigram_counts = Counter()

    for item in zero_texts:
        text = item["text"].lower()
        words = re.findall(r"[a-z][a-z'-]*[a-z]", text)
        valid_words = [w for w in words if w not in stop_words and len(w) >= 4]

        for w in valid_words:
            word_counts[w] += 1

        for i in range(len(valid_words) - 1):
            bigram = f"{valid_words[i]} {valid_words[i+1]}"
            bigram_counts[bigram] += 1

    # 按来源分组的高频词
    source_words = defaultdict(Counter)
    for item in zero_texts:
        text = item["text"].lower()
        words = re.findall(r"[a-z][a-z'-]*[a-z]", text)
        for w in words:
            if w not in stop_words and len(w) >= 4:
                source_words[item["source"]][w] += 1

    # 主题聚类（简单规则）
    theme_clusters = {
        "comfort_sleep": ["comfortable", "comfort", "sleep", "sleeping", "asleep", "rest", "restful", "cozy"],
        "pain_support": ["pain", "painful", "sore", "hurt", "aching", "back", "hip", "neck", "shoulder", "support"],
        "noise_sound": ["noise", "noisy", "quiet", "silent", "loud", "sound", "whisper", "hum", "buzz"],
        "suction_pumping": ["suction", "suck", "pull", "strength", "power", "pump", "pumping", "extract"],
        "battery_charge": ["battery", "charge", "charging", "charged", "cord", "plug", "outlet", "electric"],
        "clean_wash": ["clean", "cleaning", "wash", "washing", "sterilize", "sanitize", "dishwasher"],
        "portable_travel": ["portable", "travel", "compact", "small", "lightweight", "carry", "bag"],
        "size_fit": ["size", "fit", "fitting", "tight", "loose", "small", "large", "band", "strap"],
        "material_fabric": ["fabric", "material", "cotton", "silicone", "soft", "smooth", "stretchy"],
        "leak_spill": ["leak", "leaking", "spill", "spilling", "drip", "dripping"],
        "temperature_heat": ["hot", "heat", "warm", "cold", "cool", "burning", "overheat", "temperature"],
        "price_value": ["price", "expensive", "cheap", "affordable", "value", "worth", "budget", "cost"],
        "design_look": ["design", "color", "look", "appearance", "cute", "pretty", "ugly", "aesthetic"],
        "shipping_delivery": ["shipping", "delivery", "arrived", "package", "box", "fast", "slow"],
        "customer_service": ["service", "support", "contact", "response", "helpful", "rude", "help"],
    }

    cluster_results = []
    for theme, keywords in theme_clusters.items():
        total = sum(word_counts.get(w, 0) for w in keywords)
        if total > 0:
            top_words = [(w, word_counts.get(w, 0)) for w in keywords if word_counts.get(w, 0) > 0]
            top_words.sort(key=lambda x: -x[1])
            cluster_results.append({
                "theme": theme,
                "total_mentions": total,
                "top_words": top_words[:5],
                "zero_label_count": sum(1 for item in zero_texts if any(kw in item["text"].lower() for kw in keywords)),
            })

    cluster_results.sort(key=lambda x: -x["total_mentions"])

    return {
        "total_zero_labels": len(zero_texts),
        "top_unigrams": word_counts.most_common(top_n),
        "top_bigrams": bigram_counts.most_common(20),
        "theme_clusters": cluster_results[:10],
        "source_distribution": {src: len([x for x in zero_texts if x["source"] == src]) for src in set(x["source"] for x in zero_texts)},
    }


def recommend_keyword_expansions(zero_label_analysis: dict, existing_tags: dict) -> list[dict]:
    """基于零标签分析推荐关键词扩展"""
    recommendations = []

    # 1. 高频率但未被覆盖的词 → 建议映射到现有标签或创建新标签
    top_words = dict(zero_label_analysis["top_unigrams"][:50])

    # 2. 主题聚类结果 → 建议新标签或扩展现有标签
    for cluster in zero_label_analysis.get("theme_clusters", []):
        theme = cluster["theme"]
        total = cluster["total_mentions"]

        if total < 50:  # 忽略太小的主题
            continue

        # 查找最接近的现有标签
        matching_tags = []
        for tag_name, tag_info in existing_tags.items():
            all_keywords = " ".join(tag_info.get("keywords", [])).lower()
            score = sum(1 for w, _ in cluster["top_words"] if w in all_keywords)
            if score > 0:
                matching_tags.append((tag_name, score))

        matching_tags.sort(key=lambda x: -x[1])

        recommendations.append({
            "theme": theme,
            "mentions": total,
            "zero_label_count": cluster["zero_label_count"],
            "top_words": [w for w, _ in cluster["top_words"]],
            "suggested_action": "expand_existing_tag" if matching_tags else "create_new_tag",
            "matching_tags": [t[0] for t in matching_tags[:3]],
            "priority": "高" if total > 500 else "中" if total > 200 else "低",
        })

    return recommendations


def calculate_coverage_trends(records: list[dict]) -> dict:
    """计算覆盖率趋势"""
    source_stats = defaultdict(lambda: {"total": 0, "hit": 0, "tag_counts": Counter()})

    for r in records:
        src = r.get("_source", "unknown")
        source_stats[src]["total"] += 1
        if r["n_tags"] > 0:
            source_stats[src]["hit"] += 1
        for tag in r.get("aipl_tags", []):
            source_stats[src]["tag_counts"][tag["tag_en"]] += 1

    trends = {}
    for src, stats in source_stats.items():
        total = stats["total"]
        trends[src] = {
            "total": total,
            "coverage_rate": round(stats["hit"] / total * 100, 1) if total else 0,
            "unique_tags": len(stats["tag_counts"]),
            "top_tags": stats["tag_counts"].most_common(10),
            "zero_tags": [tag for tag, cnt in stats["tag_counts"].items() if cnt == 0],
        }

    # 全局
    global_total = len(records)
    global_hit = sum(1 for r in records if r["n_tags"] > 0)
    trends["global"] = {
        "total": global_total,
        "coverage_rate": round(global_hit / global_total * 100, 1) if global_total else 0,
    }

    return trends


def generate_optimization_report(input_dir: Path, output_path: Path):
    """生成完整的优化报告"""
    print("=" * 70)
    print("VOC 标签自动优化闭环报告")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/4] 加载打标数据...")
    records, zero_texts = load_labeled_data(input_dir)
    print(f"  总记录: {len(records):,}")
    print(f"  零标签: {len(zero_texts):,} ({len(zero_texts)/len(records)*100:.1f}%)")

    # 2. 零标签主题聚类
    print("\n[2/4] 零标签主题聚类...")
    zero_analysis = cluster_zero_label_topics(zero_texts)
    print(f"  Top 10 未覆盖主题:")
    for cluster in zero_analysis["theme_clusters"][:10]:
        print(f"    {cluster['theme']}: {cluster['total_mentions']} 提及, {cluster['zero_label_count']} 条文本")

    # 3. 覆盖率趋势
    print("\n[3/4] 覆盖率趋势...")
    trends = calculate_coverage_trends(records)
    for src, stats in trends.items():
        if src != "global":
            print(f"  {src}: {stats['coverage_rate']:.1f}% | {stats['unique_tags']} 标签种类")

    # 4. 生成建议
    print("\n[4/4] 生成优化建议...")
    existing_tags = {}  # 简化版，实际应从字典加载
    recommendations = recommend_keyword_expansions(zero_analysis, existing_tags)

    print(f"\n  生成 {len(recommendations)} 条建议:")
    for rec in recommendations[:10]:
        action = "扩展现有标签" if rec["suggested_action"] == "expand_existing_tag" else "创建新标签"
        tags = f" → {', '.join(rec['matching_tags'])}" if rec["matching_tags"] else ""
        print(f"    [{rec['priority']}] {rec['theme']}: {rec['mentions']} 提及 | {action}{tags}")

    # 保存报告
    report = {
        "generated_at": str(Path().absolute()),
        "summary": {
            "total_records": len(records),
            "zero_label_count": len(zero_texts),
            "zero_label_rate": round(len(zero_texts) / len(records) * 100, 1),
            "global_coverage": trends["global"]["coverage_rate"],
        },
        "zero_label_analysis": zero_analysis,
        "coverage_trends": trends,
        "recommendations": recommendations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  报告已保存: {output_path}")

    print("\n" + "=" * 70)
    print("优化闭环完成")
    print("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser(description="VOC 标签自动优化闭环")
    parser.add_argument("--input", required=True, help="打标结果目录")
    parser.add_argument("--output", default="optimization_report.json", help="输出报告路径")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    generate_optimization_report(input_dir, output_path)


if __name__ == "__main__":
    main()

"""VOC 打标结果看板生成 + 逆向完善分析

从各数据源 JSON Lines 加载打标结果，输出：
1. 全局/各数据源看板指标 (AIPL漏斗、Proxy NPS、驱动因素、画像洞察)
2. 零标签 VOC 高频词分析 → 标签关键词逆向完善清单
3. 标签-数据源交叉分析 → 优化建议
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# 停用词（内置，避免 nltk 依赖）
STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "and", "but", "or", "yet",
        "so", "if", "because", "although", "though", "while", "where",
        "when", "that", "which", "who", "whom", "whose", "what", "this",
        "these", "those", "i", "me", "my", "mine", "myself", "we", "us",
        "our", "ours", "ourselves", "you", "your", "yours", "yourself",
        "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "one", "ones", "one's", "oneself", "am", "it", "its",
        "itself", "they", "them", "their", "theirs", "themselves", "one",
        "ones", "one's", "oneself", "i", "me", "my", "mine", "myself",
    }

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/00-归档资料/labeling-outputs/v3.3")


def load_jsonl_batches(source_dir: Path) -> list[dict]:
    """加载一个数据源的所有 batch JSONL"""
    results = []
    for bf in sorted(source_dir.glob("batch_*.jsonl")):
        with open(bf, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
    return results


def calc_proxy_nps(records: list[dict]) -> dict:
    """计算 Proxy NPS"""
    total = len(records)
    if total == 0:
        return {"proxy_nps": 0.0, "promoters": 0, "passives": 0, "detractors": 0}
    promoters = sum(1 for r in records if r["proxy_nps"] == "promoter")
    passives = sum(1 for r in records if r["proxy_nps"] == "passive")
    detractors = sum(1 for r in records if r["proxy_nps"] == "detractor")
    proxy_nps = (promoters / total * 100) - (detractors / total * 100)
    return {
        "proxy_nps": round(proxy_nps, 1),
        "promoters": promoters,
        "passives": passives,
        "detractors": detractors,
        "promoter_pct": round(promoters / total * 100, 1),
        "detractor_pct": round(detractors / total * 100, 1),
    }


def calc_aipl_funnel(records: list[dict]) -> dict:
    """计算 AIPL 旅程漏斗"""
    node_counts = Counter(r["aipl_stage"] for r in records)
    node_themes = defaultdict(Counter)
    for r in records:
        for tag in r["aipl_tags"]:
            theme = tag.get("theme") or tag.get("tag_en", "unknown")
            node_themes[r["aipl_stage"]][theme] += 1

    funnel = {}
    for node in ["A", "I", "P1", "P2", "L1", "L2", "L3"]:
        count = node_counts.get(node, 0)
        top_themes = [{"theme": t, "count": c} for t, c in node_themes[node].most_common(3)]
        funnel[node] = {"count": count, "top_themes": top_themes}
    return funnel


def calc_driver_analysis(records: list[dict]) -> dict:
    """驱动因素分析"""
    theme_sentiments = defaultdict(list)
    theme_counts = Counter()

    for r in records:
        for tag in r["aipl_tags"]:
            theme = tag.get("theme") or tag.get("tag_en", "unknown")
            theme_sentiments[theme].append(tag["sentiment_calibrated"])
            theme_counts[theme] += 1

    total = len(records) if records else 1
    theme_stats = []
    for theme, sentiments in theme_sentiments.items():
        avg_sent = sum(sentiments) / len(sentiments)
        mention_rate = theme_counts[theme] / total
        theme_stats.append({
            "theme": theme,
            "mention_rate": round(mention_rate, 4),
            "avg_sentiment": round(avg_sent, 2),
            "count": theme_counts[theme],
            "nps_contribution": (
                "promoter_driver" if avg_sent > 0.3
                else "detractor_driver" if avg_sent < -0.3
                else "neutral"
            ),
        })

    theme_stats.sort(key=lambda x: x["mention_rate"], reverse=True)
    detractors = [t for t in theme_stats if t["nps_contribution"] == "detractor_driver"]
    promoters = [t for t in theme_stats if t["nps_contribution"] == "promoter_driver"]
    return {
        "top_detractor_themes": detractors[:5],
        "top_promoter_themes": promoters[:5],
        "all_themes": theme_stats[:15],
    }


def calc_persona_insights(records: list[dict]) -> dict:
    """画像洞察分析"""
    by_persona = defaultdict(list)
    for r in records:
        by_persona[r["persona_derived"]].append(r)

    total = len(records) if records else 1
    insights = {}
    for persona, items in by_persona.items():
        penetration = len(items) / total
        sentiments = [r["sentiment_polarity"] for r in items]
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
        theme_counts = Counter()
        for r in items:
            for tag in r["aipl_tags"]:
                theme = tag.get("theme") or tag.get("tag_en", "unknown")
                theme_counts[theme] += 1
        top_themes = [t for t, _ in theme_counts.most_common(5)]
        insights[persona] = {
            "penetration": round(penetration, 4),
            "count": len(items),
            "avg_sentiment": round(avg_sent, 2),
            "top_themes": top_themes,
            "proxy_nps": calc_proxy_nps(items),
        }
    return insights


def calc_brand_analysis(records: list[dict]) -> dict:
    """品牌分析"""
    mentions = Counter()
    comparisons = 0
    total = len(records)
    for r in records:
        for brand in r.get("brand_mentions", []):
            mentions[brand] += 1
        if r.get("brand_comparison"):
            comparisons += 1
    return {
        "total_mentions": sum(mentions.values()),
        "unique_brands": len(mentions),
        "brand_distribution": dict(mentions.most_common(10)),
        "comparison_count": comparisons,
        "comparison_rate": round(comparisons / total, 4) if total else 0,
    }


def calc_tag_coverage(records: list[dict]) -> dict:
    """标签覆盖率统计"""
    total = len(records)
    matched = sum(1 for r in records if r["n_tags"] > 0)
    all_tag_ids = set()
    for r in records:
        for tag in r["aipl_tags"]:
            all_tag_ids.add(tag["tag_id"])
    return {
        "total_voc": total,
        "matched_voc": matched,
        "unmatched_voc": total - matched,
        "coverage_rate": round(matched / total, 4) if total else 0,
        "unique_tags_matched": len(all_tag_ids),
    }


def analyze_zero_label_texts(records: list[dict]) -> dict:
    """分析零标签 VOC 的文本高频词"""
    zero_label_texts = [r["text_preview"] for r in records if r["n_tags"] == 0 and r.get("text_preview")]
    if not zero_label_texts:
        return {"total_zero": 0, "top_words": []}

    # 提取有效词
    word_counts = Counter()
    for text in zero_label_texts:
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        for w in words:
            if w not in STOP_WORDS and len(w) > 3:
                word_counts[w] += 1

    return {
        "total_zero": len(zero_label_texts),
        "top_words": word_counts.most_common(50),
    }


def build_dashboard(records: list[dict]) -> dict:
    """构建完整看板"""
    return {
        "proxy_nps": calc_proxy_nps(records),
        "aipl_funnel": calc_aipl_funnel(records),
        "driver_analysis": calc_driver_analysis(records),
        "persona_insights": calc_persona_insights(records),
        "brand_analysis": calc_brand_analysis(records),
        "tag_coverage": calc_tag_coverage(records),
    }


def generate_reverse_improvement_plan(all_sources: dict[str, list[dict]]) -> dict:
    """生成逆向完善清单"""
    # 1. 全局高频零标签词
    global_zero_words = Counter()
    for records in all_sources.values():
        zero_texts = [r["text_preview"] for r in records if r["n_tags"] == 0 and r.get("text_preview")]
        for text in zero_texts:
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            for w in words:
                if w not in STOP_WORDS:
                    global_zero_words[w] += 1

    # 2. 各数据源标签交叉对比 — 哪些标签只在特定数据源命中
    tag_by_source = defaultdict(set)
    source_tag_counts = defaultdict(Counter)
    for source, records in all_sources.items():
        for r in records:
            for tag in r["aipl_tags"]:
                tag_by_source[tag["tag_en"]].add(source)
                source_tag_counts[tag["tag_en"]][source] += 1

    # 数据源独占标签
    exclusive_tags = {
        tag: list(sources)
        for tag, sources in tag_by_source.items()
        if len(sources) == 1
    }

    # 3. 冲突率最高的标签（全局）
    tag_conflict_stats = defaultdict(lambda: {"hits": 0, "conflicts": 0})
    for records in all_sources.values():
        for r in records:
            for tag in r["aipl_tags"]:
                tag_conflict_stats[tag["tag_en"]]["hits"] += 1
                if r["sentiment_calibration"] == "conflict":
                    tag_conflict_stats[tag["tag_en"]]["conflicts"] += 1

    high_conflict = []
    for tag, stats in tag_conflict_stats.items():
        if stats["hits"] >= 20:
            rate = stats["conflicts"] / stats["hits"] * 100
            if rate > 20:
                high_conflict.append({
                    "tag": tag,
                    "hits": stats["hits"],
                    "conflicts": stats["conflicts"],
                    "conflict_rate": round(rate, 1),
                })
    high_conflict.sort(key=lambda x: -x["conflict_rate"])

    return {
        "zero_label_top_words": global_zero_words.most_common(30),
        "exclusive_tags": exclusive_tags,
        "high_conflict_tags": high_conflict,
        "suggestions": [
            {
                "priority": "高",
                "issue": "零标签VOC高频词未被覆盖",
                "evidence": [f"{w}({c})" for w, c in global_zero_words.most_common(10)],
                "action": "将高频词映射到现有标签或创建新标签",
            },
            {
                "priority": "中",
                "issue": "高冲突标签需优化关键词或情感校准策略",
                "evidence": [f"{t['tag']}({t['conflict_rate']}%冲突)" for t in high_conflict[:5]],
                "action": "增加否定/正面语境排除规则，或拆分标签",
            },
            {
                "priority": "低",
                "issue": "数据源独占标签",
                "evidence": f"{len(exclusive_tags)} 个标签仅出现在单一数据源",
                "action": "确认是否为数据源特性，如是则标记为 exclusive",
            },
        ],
    }


def main():
    print("=" * 70)
    print("VOC 打标结果看板生成 + 逆向完善分析")
    print("=" * 70)

    all_sources = {}
    for src_dir in sorted(OUTPUT_BASE.iterdir()):
        if not src_dir.is_dir() or src_dir.name.endswith(".json"):
            continue
        records = load_jsonl_batches(src_dir)
        if not records:
            continue
        all_sources[src_dir.name] = records
        print(f"\n--- {src_dir.name.upper()} 加载完成: {len(records):,} 条 ---")

    # 全局汇总
    all_records = []
    for records in all_sources.values():
        all_records.extend(records)

    print(f"\n{'=' * 70}")
    print(f"全局看板 ({len(all_records):,} 条)")
    print(f"{'=' * 70}")

    # 生成全局看板
    global_dashboard = build_dashboard(all_records)
    print(f"\n--- Proxy NPS (全局) ---")
    nps = global_dashboard["proxy_nps"]
    print(f"  Proxy NPS: {nps['proxy_nps']:.1f}")
    print(f"  推荐者: {nps['promoters']:,} ({nps['promoter_pct']}%)")
    print(f"  贬损者: {nps['detractors']:,} ({nps['detractor_pct']}%)")

    print(f"\n--- AIPL 旅程漏斗 (全局) ---")
    for node, info in global_dashboard["aipl_funnel"].items():
        if info["count"] > 0:
            print(f"  {node}: {info['count']:,} 条")
            for t in info["top_themes"]:
                print(f"    - {t['theme']}: {t['count']}")

    print(f"\n--- 驱动因素分析 (全局) ---")
    drivers = global_dashboard["driver_analysis"]
    print(f"  Promoter 驱动主题:")
    for t in drivers["top_promoter_themes"][:3]:
        print(f"    {t['theme']}: 提及率 {t['mention_rate']:.1%}, 情感 {t['avg_sentiment']:+.2f}")
    print(f"  Detractor 驱动主题:")
    for t in drivers["top_detractor_themes"][:3]:
        print(f"    {t['theme']}: 提及率 {t['mention_rate']:.1%}, 情感 {t['avg_sentiment']:+.2f}")

    print(f"\n--- 画像洞察 (全局) ---")
    for persona, info in global_dashboard["persona_insights"].items():
        if not persona:
            continue
        print(f"  {persona}: 渗透率 {info['penetration']:.1%}, NPS={info['proxy_nps']['proxy_nps']:.1f}")

    print(f"\n--- 品牌分析 (全局) ---")
    brand = global_dashboard["brand_analysis"]
    print(f"  总品牌提及: {brand['total_mentions']}")
    print(f"  独特品牌: {brand['unique_brands']}")
    print(f"  竞品对比: {brand['comparison_count']} ({brand['comparison_rate']:.1%})")
    for b, c in brand["brand_distribution"].items():
        print(f"    {b}: {c}")

    # 各数据源看板
    source_dashboards = {}
    for source, records in all_sources.items():
        source_dashboards[source] = build_dashboard(records)

    # 零标签分析
    print(f"\n{'=' * 70}")
    print(f"零标签 VOC 高频词分析")
    print(f"{'=' * 70}")
    for source, records in all_sources.items():
        zero = analyze_zero_label_texts(records)
        print(f"\n--- {source.upper()} ---")
        print(f"  零标签: {zero['total_zero']:,} 条")
        print(f"  Top 10 未覆盖词:")
        for w, c in zero["top_words"][:10]:
            print(f"    {w}: {c}")

    # 逆向完善清单
    print(f"\n{'=' * 70}")
    print(f"标签体系逆向完善清单")
    print(f"{'=' * 70}")
    plan = generate_reverse_improvement_plan(all_sources)

    print(f"\n--- 全局零标签 Top 30 高频词 ---")
    for w, c in plan["zero_label_top_words"][:30]:
        print(f"  {w}: {c}")

    print(f"\n--- 高冲突标签 (冲突率 > 20%) ---")
    for t in plan["high_conflict_tags"][:10]:
        print(f"  {t['tag']}: {t['hits']} 命中, {t['conflicts']} 冲突 ({t['conflict_rate']}%)")

    print(f"\n--- 数据源独占标签 ({len(plan['exclusive_tags'])} 个) ---")
    for tag, sources in list(plan["exclusive_tags"].items())[:10]:
        print(f"  {tag} → 仅 {sources}")

    print(f"\n--- 完善建议 ---")
    for s in plan["suggestions"]:
        print(f"\n  [{s['priority']}优先级] {s['issue']}")
        print(f"    证据: {s['evidence']}")
        print(f"    行动: {s['action']}")

    # 保存完整报告
    report = {
        "global_dashboard": global_dashboard,
        "source_dashboards": source_dashboards,
        "reverse_improvement_plan": plan,
    }
    report_path = OUTPUT_BASE / "dashboard_and_insights.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告已保存: {report_path}")

    print(f"\n{'=' * 70}")
    print("分析完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

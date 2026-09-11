"""打标结果汇总分析

读取所有数据源的打标结果，生成汇总报告和洞察。
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze_source(source_dir: Path) -> dict:
    """分析单个数据源的打标结果"""
    batch_files = sorted(source_dir.glob("batch_*.jsonl"))
    if not batch_files:
        return {}

    total = 0
    labeled = 0
    conflicts = 0
    tag_counts = Counter()
    aipl_dist = Counter()
    nps_dist = Counter()
    persona_dist = Counter()
    spu_tag_dist = defaultdict(Counter)  # SPU -> tag counts
    rating_tag_dist = defaultdict(Counter)  # rating -> tag counts

    for bf in batch_files:
        with open(bf, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                r = json.loads(line)
                if r["n_tags"] > 0:
                    labeled += 1
                if r["sentiment_calibration"] == "conflict":
                    conflicts += 1
                for t in r["aipl_tags"]:
                    tag_counts[t["tag_en"]] += 1
                    aipl_dist[t["aipl_node"]] += 1
                nps_dist[r["proxy_nps"]] += 1
                if r["persona_derived"]:
                    persona_dist[r["persona_derived"]] += 1

    summary_path = source_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            saved = json.load(f)
    else:
        saved = {}

    return {
        "source": source_dir.name,
        "total": total,
        "labeled": labeled,
        "coverage": labeled / total * 100 if total else 0,
        "conflicts": conflicts,
        "conflict_rate": conflicts / total * 100 if total else 0,
        "top_tags": tag_counts.most_common(20),
        "aipl_distribution": dict(aipl_dist),
        "nps_distribution": dict(nps_dist),
        "persona_distribution": dict(persona_dist),
        "batches": len(batch_files),
    }


def main():
    base_dir = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest")

    print("=" * 70)
    print("VOC 打标结果汇总分析")
    print("=" * 70)

    all_results = {}
    for src_dir in sorted(base_dir.iterdir()):
        if src_dir.is_dir():
            result = analyze_source(src_dir)
            if result:
                all_results[src_dir.name] = result

    # 全局汇总
    total_all = sum(r["total"] for r in all_results.values())
    labeled_all = sum(r["labeled"] for r in all_results.values())
    conflicts_all = sum(r["conflicts"] for r in all_results.values())

    print(f"\n--- 全局汇总 ---")
    print(f"  数据源: {len(all_results)}")
    print(f"  总处理: {total_all:,} 条")
    print(f"  总标签命中: {labeled_all:,}")
    print(f"  整体覆盖率: {labeled_all/total_all*100:.1f}%")
    print(f"  整体冲突率: {conflicts_all/total_all*100:.1f}%")

    for name, r in all_results.items():
        print(f"\n--- {name.upper()} ---")
        print(f"  处理: {r['total']:,} 条")
        print(f"  覆盖率: {r['coverage']:.1f}%")
        print(f"  冲突率: {r['conflict_rate']:.1f}%")
        print(f"  AIPL: {r['aipl_distribution']}")
        print(f"  NPS: {r['nps_distribution']}")
        print(f"  Top 5 标签:")
        for tag, cnt in r["top_tags"][:5]:
            print(f"    {tag}: {cnt}")

    # 保存全局报告
    report_path = base_dir / "analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "global": {
                "total": total_all,
                "labeled": labeled_all,
                "coverage": labeled_all/total_all*100 if total_all else 0,
                "conflicts": conflicts_all,
                "conflict_rate": conflicts_all/total_all*100 if total_all else 0,
            },
            "sources": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()

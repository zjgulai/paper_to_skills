"""零标签 VOC 提取（Phase 2.1）

从统一打标结果中提取 n_tags == 0 的记录，按品类分组统计，
输出各品类零标签率和样本。
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def extract_zero_labels():
    print("=" * 70)
    print("Phase 2.1: 零标签 VOC 提取")
    print("=" * 70)

    input_path = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling/phase1_3_all_sources_labeled.jsonl"
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/tag_gap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取全部打标结果
    print("\n--- 读取打标结果 ---")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    print(f"  总计: {len(records):,} 条")

    # 筛选零标签
    zero_label = [r for r in records if r["n_tags"] == 0]
    print(f"  零标签: {len(zero_label):,} 条 ({len(zero_label)/len(records)*100:.1f}%)")

    # 按品类分组统计
    print("\n--- 按品类统计 ---")
    category_stats = defaultdict(lambda: {"total": 0, "zero": 0, "samples": []})

    for r in records:
        cat = r.get("category") or "未分类"
        category_stats[cat]["total"] += 1
        if r["n_tags"] == 0:
            category_stats[cat]["zero"] += 1
            category_stats[cat]["samples"].append(r)

    # 按零标签率排序
    sorted_cats = sorted(
        category_stats.items(),
        key=lambda x: x[1]["zero"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True,
    )

    report = []
    for cat, stat in sorted_cats:
        zero_rate = stat["zero"] / stat["total"] * 100 if stat["total"] > 0 else 0
        report.append(
            {
                "category": cat,
                "total": stat["total"],
                "zero_count": stat["zero"],
                "zero_rate": round(zero_rate, 1),
            }
        )
        print(f"  {cat}: {stat['zero']:,} / {stat['total']:,} ({zero_rate:.1f}%)")

    # 保存零标签率报告
    report_path = output_dir / "zero_label_by_category.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {report_path}")

    # 保存每个品类的零标签样本（最多 100 条/品类）
    print("\n--- 提取样本 ---")
    all_samples = []
    for cat, stat in category_stats.items():
        samples = stat["samples"][:100]
        for s in samples:
            all_samples.append(
                {
                    "review_id": s["review_id"],
                    "text": s["text"],
                    "data_source": s["data_source"],
                    "category": cat,
                    "rating": s.get("rating"),
                    "platform": s["platform"],
                }
            )

    samples_df = pd.DataFrame(all_samples)
    samples_path = output_dir / "zero_label_samples.csv"
    samples_df.to_csv(samples_path, index=False)
    print(f"  样本: {samples_path} ({len(all_samples):,} 条, {len(category_stats)} 个品类)")

    # 按数据源统计零标签率
    print("\n--- 按数据源统计 ---")
    source_stats = defaultdict(lambda: {"total": 0, "zero": 0})
    for r in records:
        ds = r["data_source"]
        source_stats[ds]["total"] += 1
        if r["n_tags"] == 0:
            source_stats[ds]["zero"] += 1

    for ds, stat in sorted(source_stats.items()):
        rate = stat["zero"] / stat["total"] * 100
        print(f"  {ds}: {stat['zero']:,} / {stat['total']:,} ({rate:.1f}%)")

    # 审计
    audit = {
        "phase": "2.1",
        "total_records": len(records),
        "zero_label_count": len(zero_label),
        "zero_label_rate": round(len(zero_label) / len(records) * 100, 1),
        "category_count": len(category_stats),
        "sample_count": len(all_samples),
        "source_breakdown": {
            ds: {"total": s["total"], "zero": s["zero"], "rate": round(s["zero"]/s["total"]*100, 1)}
            for ds, s in source_stats.items()
        },
        "report_path": str(report_path),
        "samples_path": str(samples_path),
    }
    audit_path = output_dir / "phase2_1_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 2.1 完成")
    print("=" * 70)


if __name__ == "__main__":
    extract_zero_labels()

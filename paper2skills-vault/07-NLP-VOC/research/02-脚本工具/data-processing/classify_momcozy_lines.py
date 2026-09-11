"""对 Momcozy 抽样数据运行品线分类

复用 generate_product_line_spu_matrix.py 中的 classify_product_line 函数，
统计 momcozy 数据的品线分布和 other 率，提取 other 文本供零标签挖掘。
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# 将原脚本目录加入路径
# 引用同级目录的 generate_product_line_spu_matrix
sys.path.insert(0, str(Path(__file__).parent.parent / "data-processing"))

from generate_product_line_spu_matrix import classify_product_line


def classify_momcozy(input_csv: Path, output_dir: Path):
    print("=" * 70)
    print("Momcozy 数据品线分类")
    print("=" * 70)

    print(f"\n[1/3] 加载数据: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  总记录: {len(df):,}")

    print(f"\n[2/3] 运行品线分类...")
    lines = []
    other_texts = []
    for idx, row in df.iterrows():
        text = str(row.get("text", ""))
        category_lv4 = str(row.get("category_lv4", "")) if pd.notna(row.get("category_lv4")) else None
        line = classify_product_line(text, category_lv4=category_lv4)
        lines.append(line)

        if line == "other":
            other_texts.append({
                "review_id": row.get("review_id", ""),
                "text": text[:300],
                "rating": row.get("rating", ""),
                "spu_name": row.get("spu_name", ""),
                "category_lv4": row.get("category_lv4", ""),
            })

        if (idx + 1) % 2000 == 0:
            print(f"  已处理 {idx + 1:,} / {len(df):,}")

    df["product_line"] = lines

    print(f"\n[3/3] 统计与导出...")
    line_counts = Counter(lines)
    total = len(lines)
    other_count = line_counts.get("other", 0)
    other_rate = other_count / total * 100

    print(f"\n  品线分布:")
    for line, cnt in line_counts.most_common():
        pct = cnt / total * 100
        marker = "  ⚠️" if line == "other" else ""
        print(f"    {line:30s}: {cnt:5,} ({pct:5.1f}%){marker}")

    print(f"\n  other 率: {other_rate:.1f}% ({other_count:,} / {total:,})")

    # 导出分类结果
    output_dir.mkdir(parents=True, exist_ok=True)

    # 完整分类结果
    full_path = output_dir / "momcozy_classified.csv"
    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"\n  分类结果: {full_path}")

    # 导出 other 文本供零标签挖掘
    other_path = output_dir / "momcozy_other_texts.json"
    with open(other_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_other": len(other_texts),
            "other_rate": round(other_rate, 2),
            "samples": other_texts,
        }, f, ensure_ascii=False, indent=2)
    print(f"  other 文本: {other_path} ({len(other_texts):,} 条)")

    # 导出摘要
    summary = {
        "total_records": total,
        "other_count": other_count,
        "other_rate": round(other_rate, 2),
        "line_distribution": dict(line_counts.most_common()),
        "by_rating": {},
        "by_category": {},
    }

    # 按星级统计 other 率
    for rating in sorted(df["rating"].unique()):
        sub = df[df["rating"] == rating]
        other_in_rating = (sub["product_line"] == "other").sum()
        summary["by_rating"][str(rating)] = {
            "total": len(sub),
            "other": int(other_in_rating),
            "other_rate": round(other_in_rating / len(sub) * 100, 1),
        }

    # 按品类统计 other 率（Top 20）
    for cat in df["category_lv4"].value_counts().head(20).index:
        sub = df[df["category_lv4"] == cat]
        other_in_cat = (sub["product_line"] == "other").sum()
        summary["by_category"][cat] = {
            "total": len(sub),
            "other": int(other_in_cat),
            "other_rate": round(other_in_cat / len(sub) * 100, 1),
        }

    summary_path = output_dir / "classification_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  统计摘要: {summary_path}")

    print("\n" + "=" * 70)
    print("品线分类完成")
    print("=" * 70)

    return df, other_texts


if __name__ == "__main__":
    input_csv = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/00-归档资料/runtime-outputs/sampled/momcozy_sampled_full.csv")
    output_dir = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/00-归档资料/runtime-outputs/classified")
    classify_momcozy(input_csv, output_dir)

"""快速诊断 — 验证关键词优化后的效果"""
import json
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, "/Users/pray/project/paper_to_skills/paper2skills-code/nlp_voc/proxy_nps_aipl_workflow")
from unified_label_extraction import TagSeedDictionary, VOCLabelExtractor, VOCRecord


def main():
    print("=" * 70)
    print("快速诊断 — 验证高冲突标签优化效果")
    print("=" * 70)

    # 加载标签字典
    xlsx_path = "/Users/pray/project/sgcs/20_insights/22_insight_reports/03_voc分析/01_内部VOC精细化运营洞察框架/SGCS_VOC标签字典_喂养电器V3.2_final.xlsx"
    tag_dict = TagSeedDictionary.from_xlsx(xlsx_path)

    # 按 review 数据源筛选
    review_tags = tag_dict.filter_by_source("review")
    print(f"\nAmazon review 适配标签: {len(review_tags)} / {len(tag_dict.get_all())}")

    # 创建萃取器（使用 review 标签子集）
    extractor = VOCLabelExtractor(tag_dict=tag_dict)

    # 加载 1000 条样本
    csv_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv"
    df = pd.read_csv(csv_path, nrows=1000)

    results = []
    for _, row in df.iterrows():
        text = str(row.get("Content", "") or row.get("English Content", ""))
        if not text or text == "nan":
            continue
        rating = row.get("Rating")
        try:
            rating = float(rating) if pd.notna(rating) else None
        except:
            rating = None

        voc = VOCRecord(
            review_id=f"{row.get('Asin')}_{_}",
            text=text,
            source_type="review",
            platform="amazon",
            spu_code=str(row.get("Asin", "")),
            product_line="breast_pump",
            category="unknown",
            rating=rating,
        )
        result = extractor.extract(voc)
        results.append({
            "n_tags": len(result.aipl_tags),
            "tags": [t.tag_en for t in result.aipl_tags],
            "is_conflict": result.sentiment_calibration == "conflict",
            "rating": rating,
        })

    # 统计
    has_tags = sum(1 for r in results if r["n_tags"] > 0)
    conflicts = sum(1 for r in results if r["is_conflict"])

    # 高冲突标签命中统计
    tag_stats = Counter()
    conflict_tag_stats = Counter()
    for r in results:
        for t in r["tags"]:
            tag_stats[t] += 1
            if r["is_conflict"]:
                conflict_tag_stats[t] += 1

    print(f"\n--- 整体指标 ---")
    print(f"  样本: {len(results)}")
    print(f"  覆盖率: {has_tags}/{len(results)} ({has_tags/len(results)*100:.1f}%)")
    print(f"  冲突: {conflicts}/{len(results)} ({conflicts/len(results)*100:.1f}%)")

    print(f"\n--- Top 10 命中标签 ---")
    for tag, cnt in tag_stats.most_common(10):
        c_cnt = conflict_tag_stats.get(tag, 0)
        rate = c_cnt / cnt * 100 if cnt > 0 else 0
        print(f"  {tag}: {cnt} 次, 冲突 {c_cnt} ({rate:.1f}%)")

    # 对比优化前的高冲突标签
    high_conflict_before = [
        "general_core_product_performance_issue",
        "instruction_user_manual",
        "size_runs_large",
        "size_runs_small",
        "delivery_too_slow",
        "price_concern",
        "burnt_smell",
        "too_noisy",
    ]
    print(f"\n--- 优化前后对比 (Top 高冲突标签) ---")
    for tag in high_conflict_before:
        hits = tag_stats.get(tag, 0)
        c = conflict_tag_stats.get(tag, 0)
        rate = c / hits * 100 if hits > 0 else 0
        print(f"  {tag}: 命中 {hits}, 冲突 {c} ({rate:.1f}%)")

    print("\n✓ 快速诊断完成")


if __name__ == "__main__":
    main()

"""快速验证扩充字典的覆盖率提升效果

对 Amazon 前 5000 条数据分别用旧字典和新字典打标，对比覆盖率。
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/pray/project/paper_to_skills/paper2skills-code/nlp_voc/proxy_nps_aipl_workflow")

from unified_label_extraction import TagSeedDictionary, VOCLabelExtractor

OLD_DICT = "/Users/pray/project/sgcs/20_insights/22_insight_reports/03_voc分析/01_内部VOC精细化运营洞察框架/SGCS_VOC标签字典_喂养电器V3.2_final.xlsx"
NEW_DICT = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest/SGCS_VOC标签字典_V3.3_expanded.xlsx"
CSV_PATH = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv"


def test_dict(tag_dict_path: str, label: str, texts: list[str], sample_size: int = 2000):
    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"{'='*60}")

    tag_dict = TagSeedDictionary.from_xlsx(tag_dict_path)
    extractor = VOCLabelExtractor(tag_dict)

    hit_count = 0
    zero_count = 0
    tag_dist = {}
    new_tag_hits = {}  # 仅用于新字典：新增通用标签的命中

    start = time.time()
    for text in texts[:sample_size]:
        voc = type('VOCRecord', (), {
            'review_id': 'test',
            'text': text,
            'source_type': 'review',
            'platform': 'amazon',
            'spu_code': '',
            'product_line': '',
            'category': '',
            'rating': None,
            'order_id': None,
            'user_id': None,
            'timestamp': None,
            'classification_tag': None,
            'cn_level1': None,
            'cn_level2': None,
            'cn_level3': None,
        })()
        result = extractor.extract(voc)
        n_tags = len(result.aipl_tags)

        if n_tags > 0:
            hit_count += 1
            for tag in result.aipl_tags:
                tag_dist[tag.tag_en] = tag_dist.get(tag.tag_en, 0) + 1
                if label == "新字典" and tag.tag_id.startswith("TAG_GEN_"):
                    new_tag_hits[tag.tag_en] = new_tag_hits.get(tag.tag_en, 0) + 1
        else:
            zero_count += 1

    elapsed = time.time() - start
    coverage = hit_count / sample_size * 100

    print(f"  样本: {sample_size}")
    print(f"  命中: {hit_count} ({coverage:.1f}%)")
    print(f"  零标签: {zero_count} ({zero_count/sample_size*100:.1f}%)")
    print(f"  耗时: {elapsed:.1f}s ({sample_size/elapsed:.0f} 条/s)")
    print(f"  唯一标签命中: {len(tag_dist)}")

    if new_tag_hits:
        print(f"\n  新增通用标签命中:")
        for tag, cnt in sorted(new_tag_hits.items(), key=lambda x: -x[1]):
            print(f"    {tag}: {cnt}")

    return coverage, tag_dist


def main():
    print("=" * 60)
    print("扩充字典覆盖率对比验证")
    print("=" * 60)

    # 加载文本
    print("\n加载 Amazon 数据...")
    df = pd.read_csv(CSV_PATH, nrows=5000)
    texts = df["Content"].dropna().astype(str).tolist()
    print(f"  加载: {len(texts)} 条")

    # 旧字典
    old_cov, old_dist = test_dict(OLD_DICT, "旧字典 (V3.2)", texts, 2000)

    # 新字典
    new_cov, new_dist = test_dict(NEW_DICT, "新字典 (V3.3)", texts, 2000)

    # 对比
    print(f"\n{'='*60}")
    print("对比结果")
    print(f"{'='*60}")
    print(f"  旧覆盖率: {old_cov:.1f}%")
    print(f"  新覆盖率: {new_cov:.1f}%")
    print(f"  提升: +{new_cov - old_cov:.1f} 个百分点")
    print(f"  相对提升: {(new_cov - old_cov) / old_cov * 100:.0f}%")

    # Top 命中差异
    print(f"\n  新字典 Top 10 命中标签:")
    for tag, cnt in sorted(new_dist.items(), key=lambda x: -x[1])[:10]:
        old_cnt = old_dist.get(tag, 0)
        delta = f" (+{cnt - old_cnt})" if cnt > old_cnt else ""
        print(f"    {tag}: {cnt}{delta}")


if __name__ == "__main__":
    main()

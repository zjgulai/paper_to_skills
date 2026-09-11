"""快速覆盖率验证

用扩充后的标签字典对少量样本打标，验证覆盖率提升。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/pray/project/paper_to_skills/paper2skills-code/nlp_voc/proxy_nps_aipl_workflow")

from unified_label_extraction import VOCLabelExtractor, TagSeedDictionary

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest")
EXPANDED_XLSX = OUTPUT_BASE / "SGCS_VOC标签字典_V3.3_expanded.xlsx"


def test_expanded_dictionary(sample_size: int = 2000):
    """快速验证扩充字典的覆盖率"""
    print("=" * 70)
    print("扩充字典覆盖率快速验证")
    print("=" * 70)

    # 1. 加载扩充后的字典
    print("\n加载扩充后的标签字典...")
    tag_dict = TagSeedDictionary.from_xlsx(str(EXPANDED_XLSX))
    print(f"  标签总数: {len(tag_dict.get_all())}")

    # 2. 创建提取器
    extractor = VOCLabelExtractor(tag_dict)

    # 3. 加载样本数据
    samples = []
    for src in ["amazon", "trustpilot", "reddit", "zendesk"]:
        import glob
        files = sorted(glob.glob(str(OUTPUT_BASE / src / "batch_*.jsonl")))
        for f in files[:1]:
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    samples.append({
                        "text": r.get("text_preview", ""),
                        "source": src,
                        "original_n_tags": r["n_tags"],
                    })
                    if len(samples) >= sample_size // 4:
                        break
            if len(samples) >= sample_size // 4:
                break
        if len(samples) >= sample_size:
            break

    print(f"\n测试样本: {len(samples)} 条")

    # 4. 用新字典打标
    hit_count = 0
    zero_count = 0
    new_hits = 0  # 原来零标签现在命中
    tag_dist = {}

    for s in samples:
        # 创建 VOCRecord 对象
        voc = type('VOCRecord', (), {
            'review_id': 'test',
            'text': s["text"],
            'source_type': 'review' if s["source"] == "amazon" else s["source"],
            'platform': s["source"],
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
        n_tags = len(result.tags)

        if n_tags > 0:
            hit_count += 1
            for tag in result.tags:
                tag_dist[tag.tag_en] = tag_dist.get(tag.tag_en, 0) + 1
            if s["original_n_tags"] == 0:
                new_hits += 1
        else:
            zero_count += 1

    coverage = hit_count / len(samples) * 100
    original_coverage = sum(1 for s in samples if s["original_n_tags"] > 0) / len(samples) * 100

    print(f"\n{'=' * 70}")
    print("验证结果")
    print(f"{'=' * 70}")
    print(f"  样本数: {len(samples)}")
    print(f"  原覆盖率: {original_coverage:.1f}%")
    print(f"  新覆盖率: {coverage:.1f}%")
    print(f"  覆盖率提升: +{coverage - original_coverage:.1f}个百分点")
    print(f"  零标签 → 命中: {new_hits} 条")

    print(f"\n{'=' * 70}")
    print("Top 20 命中标签")
    print(f"{'=' * 70}")
    for tag, cnt in sorted(tag_dist.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag}: {cnt}")

    print(f"\n{'=' * 70}")
    print("新增通用标签命中情况")
    print(f"{'=' * 70}")
    new_tags = ["ease_of_use", "product_quality_perception", "comfort_experience",
                "product_functionality", "design_appearance", "material_texture",
                "size_accuracy", "portability_convenience", "cleaning_maintenance",
                "noise_level_acceptable", "general_dissatisfaction", "difficult_to_use",
                "durability_concern", "positive_customer_service", "fast_shipping_delivery",
                "strong_recommendation", "gift_purchase_intent", "packaging_quality"]
    for tag in new_tags:
        cnt = tag_dist.get(tag, 0)
        print(f"  {tag}: {cnt}")

    return coverage, original_coverage


if __name__ == "__main__":
    test_expanded_dictionary(sample_size=2000)

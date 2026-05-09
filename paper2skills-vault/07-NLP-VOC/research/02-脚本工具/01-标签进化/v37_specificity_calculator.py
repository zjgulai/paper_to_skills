"""v3.7 品类特异性指数计算器

从 VOC 打标数据（phase3_p3_labeled.jsonl）统计每个标签在各品线的命中分布，
计算品类特异性指数、共性/特性分类、主导品类。

输出: tag_en -> {specificity_index, commonality_class, dominant_category} 映射表
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# 6 大品线
SIX_LINES = ["家居家纺", "智能母婴电器", "吸奶器", "内衣服饰", "母婴综合护理", "喂养电器"]

# 细粒度品类 -> 6 大品线 映射规则（按优先级排序，先匹配的先应用）
LINE_MAPPING_RULES = [
    # 吸奶器
    (
        "吸奶器",
        [
            "吸奶器",
            "储奶",
            "防溢乳垫",
            "背奶",
            "哺乳内衣",
            "哺乳背心",
            "哺乳巾",
            "吸奶器内衣",
            "吸奶器背心",
            "吸奶器湿巾",
        ],
    ),
    # 内衣服饰
    (
        "内衣服饰",
        [
            "内衣",
            "内裤",
            "塑身",
            "孕妇裤",
            "哺乳睡衣",
            "哺乳睡裙",
            "孕妇枕",
            "抱枕",
            "托腹",
            "收腹",
            "压力袜",
            "产妇袜",
            "常规内衣",
            "夹腿枕",
        ],
    ),
    # 家居家纺
    (
        "家居家纺",
        [
            "枕",
            "被",
            "毯",
            "床笠",
            "睡袋",
            "襁褓",
            "包巾",
            "浴巾",
            "方巾",
            "凉感被",
            "凉感毯",
            "凉感头枕套",
        ],
    ),
    # 母婴综合护理
    (
        "母婴综合护理",
        [
            "摇椅",
            "推车",
            "背巾",
            "背带",
            "腰凳",
            "围栏",
            "门栏",
            "辅食",
            "牙胶",
            "玩具",
            "体温计",
            "温度计",
            "体重秤",
            "湿巾",
            "奶瓶",
            "清洁刷",
            "磨甲器",
            "吸鼻器",
            "洗护套装",
            "护理套装",
            "牙刷",
            "沐浴",
            "应急垫",
            "收纳包",
            "待产包",
            "运动垫",
            "安抚巾",
            "车挂包",
            "纸尿裤",
            "隔尿垫",
            "尿布膏",
            "保温包",
            "妈咪包",
            "储奶壶",
            "清洁液",
            "清洁喷雾",
            "辅食袋",
            "辅食装袋器",
            "防挤压支架",
        ],
    ),
    # 喂养电器
    (
        "喂养电器",
        ["消毒器", "暖奶器", "温奶器"],
    ),
    # 智能母婴电器
    (
        "智能母婴电器",
        ["白噪音", "空气净化器"],
    ),
]


def map_to_six_lines(product_line: str) -> Optional[str]:
    """将细粒度品类映射到 6 大品线，无法映射则返回 None"""
    if not product_line or product_line == "unknown":
        return None
    if product_line in SIX_LINES:
        return product_line
    for line, keywords in LINE_MAPPING_RULES:
        for kw in keywords:
            if kw in product_line:
                return line
    return None


def classify_specificity(index: float) -> str:
    """基于品类特异性指数分类"""
    if index < 0.3:
        return "强共性标签"
    elif index < 0.5:
        return "共性标签"
    elif index < 0.8:
        return "半共性标签"
    else:
        return "特性标签"


def normalize_class(cls: str) -> str:
    """统一共性/特性分类值域"""
    mapping = {
        "共性": "共性标签",
        "特性": "特性标签",
    }
    return mapping.get(cls, cls)


def main():
    jsonl_path = (
        Path(__file__).parent.parent.parent
        / "04-输出结果/unified_labeling/phase3_p3_labeled.jsonl"
    )
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "08-辅助数据/v37_tag_specificity_map.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v3.7 品类特异性指数计算")
    print("=" * 70)
    print(f"\n输入: {jsonl_path}")

    # 统计每个 tag_en 在各品线的命中次数
    tag_line_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tag_total_counts: dict[str, int] = defaultdict(int)
    line_totals: dict[str, int] = defaultdict(int)
    unknown_lines: dict[str, int] = defaultdict(int)

    processed = 0
    mapped_records = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            raw_pl = record.get("product_line") or record.get("category") or "unknown"
            pl = map_to_six_lines(raw_pl)

            if pl:
                line_totals[pl] += 1
                mapped_records += 1
            else:
                unknown_lines[raw_pl] += 1

            for lbl in record.get("labels", []):
                tag_en = lbl.get("tag_en")
                if not tag_en:
                    continue
                tag_total_counts[tag_en] += 1
                if pl:
                    tag_line_counts[tag_en][pl] += 1

            processed += 1
            if processed % 50000 == 0:
                print(f"  已处理 {processed:,} 条...")

    print(f"\n处理完成: {processed:,} 条")
    print(f"  成功映射到 6 大品线: {mapped_records:,} 条")
    print(f"  无法映射: {processed - mapped_records:,} 条")
    print(f"  唯一标签数: {len(tag_total_counts)}")

    print(f"\n6 大品线分布:")
    for line in SIX_LINES:
        print(f"  {line}: {line_totals[line]:,}")

    # 计算三字段
    print("\n--- 计算品类特异性指数 ---")
    result = {}
    class_dist = defaultdict(int)

    for tag_en, total in sorted(tag_total_counts.items()):
        counts = tag_line_counts[tag_en]
        line_vals = [counts.get(l, 0) for l in SIX_LINES]
        six_total = sum(line_vals)

        if six_total == 0:
            # 该标签在所有已知品线中无命中，跳过
            continue

        max_count = max(line_vals)
        specificity_index = round(max_count / six_total, 3)
        commonality_class = classify_specificity(specificity_index)
        class_dist[commonality_class] += 1

        # 主导品类：命中最多的品线
        dominant = SIX_LINES[line_vals.index(max_count)]

        result[tag_en] = {
            "total_hits": total,
            "six_line_hits": six_total,
            "specificity_index": specificity_index,
            "commonality_class": commonality_class,
            "dominant_category": dominant,
            "line_distribution": {l: counts.get(l, 0) for l in SIX_LINES},
        }

    # 保存映射表
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n输出: {output_path}")
    print(f"  映射标签数: {len(result)}")

    print(f"\n共性/特性分类分布:")
    for cls in ["强共性标签", "共性标签", "半共性标签", "特性标签"]:
        cnt = class_dist[cls]
        print(f"  {cls}: {cnt} ({cnt / len(result) * 100:.1f}%)")

    # 展示示例
    print("\n--- 示例标签 ---")
    examples = [
        "general_positive",
        "comfort_experience",
        "noise_level_acceptable",
        "size_runs_small",
    ]
    for tag in examples:
        if tag in result:
            r = result[tag]
            print(f"\n  {tag}:")
            print(f"    总命中: {r['total_hits']:,} (6大品线: {r['six_line_hits']:,})")
            print(f"    特异性指数: {r['specificity_index']}")
            print(f"    分类: {r['commonality_class']}")
            print(f"    主导品类: {r['dominant_category']}")
            print(f"    分布: {r['line_distribution']}")

    print("\n" + "=" * 70)
    print("计算完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

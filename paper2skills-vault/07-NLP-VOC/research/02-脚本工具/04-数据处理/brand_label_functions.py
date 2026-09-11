"""品牌识别 Label Functions（Phase 4 T1.2）

将品牌关键词库包装为 VOC 打标流水线兼容的 Label Functions。
输出格式与通用标签器保持一致，可直接注入标签列表。

品牌标签设计：
- tag_id: BRAND_{品牌名}
- tag_en: 品牌英文名
- tag_cn: 品牌中文名（同英文）
- aipl_node: B1（品牌认知）
- sentiment_preset: own_brand→neutral, competitor→negative
- brand_type: own_brand | direct_competitor | indirect_competitor
"""

from typing import Optional

from brand_keyword_library import (
    BRAND_KEYWORD_LIBRARY,
    identify_brand,
    get_brand_type,
    get_brand_categories,
)


# ── 品牌标签元数据 ─────────────────────────────────────────────────

BRAND_TAG_TEMPLATE = {
    "tag_id_prefix": "BRAND_",
    "source": "brand_label",
    "aipl_node": "B1",  # 品牌认知节点
}

# 品牌类型 → 情感预设映射
BRAND_TYPE_SENTIMENT = {
    "own_brand": "neutral",
    "direct_competitor": "negative",
    "indirect_competitor": "negative",
}


# ── Label Function 核心 ────────────────────────────────────────────

def apply_brand_labels(text: str) -> list[dict]:
    """对单条文本应用品牌标签

    Returns: 品牌标签列表（流水线标准格式）
    """
    brand_results = identify_brand(text)
    labels: list[dict] = []

    for result in brand_results:
        brand_name = result["brand"]
        brand_type = result["brand_type"]
        matched_kw = result["matched_kw"]
        confidence = result["confidence"]

        config = BRAND_KEYWORD_LIBRARY.get(brand_name)
        if not config:
            continue

        sentiment = BRAND_TYPE_SENTIMENT.get(brand_type, "neutral")
        labels.append({
            "tag_id": f"{BRAND_TAG_TEMPLATE['tag_id_prefix']}{brand_name.replace(' ', '_')}",
            "tag_en": brand_name.lower().replace(" ", "_"),
            "tag_cn": brand_name,
            "aipl_node": BRAND_TAG_TEMPLATE["aipl_node"],
            "sentiment_preset": sentiment,
            "sentiment_calibrated": 1.0 if sentiment == "positive" else (-1.0 if sentiment == "negative" else 0.0),
            "confidence": round(confidence, 2),
            "source": BRAND_TAG_TEMPLATE["source"],
            "matched_keyword": matched_kw,
            "brand_type": brand_type,
            "brand_categories": config.categories,
        })

    return labels


def filter_competitor_voc(labels: list[dict], keep_categories: Optional[list[str]] = None) -> bool:
    """判断竞品 VOC 是否应保留

    规则：
    1. 若 keep_categories 为 None → 保留所有（仅用于品牌识别，不过滤）
    2. 若 keep_categories 不为 None → 仅当竞品品牌所属品类与 keep_categories 有交集时保留

    Returns: True=保留, False=剔除
    """
    if keep_categories is None:
        return True

    for lbl in labels:
        if lbl.get("brand_type") in ("direct_competitor", "indirect_competitor"):
            brand_cats = set(lbl.get("brand_categories", []))
            if brand_cats & set(keep_categories):
                return True
            return False  # 竞品但品类不匹配 → 剔除

    return True  # 无竞品标签 → 保留


def resolve_brand_conflicts(labels: list[dict]) -> list[dict]:
    """解决品牌标签冲突

    规则：
    1. 同一品牌多次命中 → 去重，取最高置信度
    2. 自有品牌 + 竞品同时命中 → 保留两者（用于竞品对比场景）
    """
    seen: dict[str, dict] = {}
    for lbl in labels:
        tag_id = lbl["tag_id"]
        if tag_id not in seen or lbl["confidence"] > seen[tag_id]["confidence"]:
            seen[tag_id] = lbl

    return list(seen.values())


# ── 快捷查询 ───────────────────────────────────────────────────────

def has_own_brand(labels: list[dict]) -> bool:
    """是否包含自有品牌标签"""
    return any(lbl.get("brand_type") == "own_brand" for lbl in labels)


def has_direct_competitor(labels: list[dict]) -> bool:
    """是否包含直接竞品标签"""
    return any(lbl.get("brand_type") == "direct_competitor" for lbl in labels)


def get_mentioned_brands(labels: list[dict]) -> list[str]:
    """获取提及的所有品牌名"""
    return [lbl["tag_cn"] for lbl in labels if lbl.get("source") == "brand_label"]


# ── 自证测试 ───────────────────────────────────────────────────────

def _test():
    """品牌 Label Functions 自证测试"""
    print("=" * 70)
    print("品牌 Label Functions 自证测试")
    print("=" * 70)

    test_cases = [
        # (文本, 期望标签数, 期望品牌列表, 描述)
        ("I love my Momcozy pump", 1, ["Momcozy"], "自有品牌-吸奶器"),
        ("Spectra vs Medela comparison", 2, ["Spectra", "Medela"], "双竞品"),
        ("The Willow pump is better than Elvie", 2, ["Willow", "Elvie"], "竞品对比"),
        ("Boppy pillow is comfortable", 1, ["Boppy"], "间接竞品"),
        ("No brand mentioned here", 0, [], "无品牌"),
        ("I switched from Haakaa to Momcozy", 2, ["Haakaa", "Momcozy"], "竞品→自有"),
        ("avent bottles are cheaper", 1, ["Philips Avent"], "短品牌名"),
        ("The owlet smart sock is great", 1, ["Owlet"], "智能母婴竞品"),
        ("Momcozy is much better than Spectra", 2, ["Momcozy", "Spectra"], "自有+竞品"),
        ("Kindred Bravely bra review", 1, ["Kindred Bravely"], "带空格品牌名"),
    ]

    passed = 0
    failed = 0

    for text, expected_count, expected_brands, desc in test_cases:
        labels = apply_brand_labels(text)
        labels = resolve_brand_conflicts(labels)
        actual_count = len(labels)
        actual_brands = sorted(get_mentioned_brands(labels))
        expected_brands_sorted = sorted(expected_brands)

        count_ok = actual_count == expected_count
        brands_ok = actual_brands == expected_brands_sorted
        status = "PASS" if count_ok and brands_ok else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {desc}")
        if status == "FAIL":
            print(f"    文本: '{text}'")
            if not count_ok:
                print(f"    标签数: 期望 {expected_count}, 实际 {actual_count}")
            if not brands_ok:
                print(f"    品牌: 期望 {expected_brands_sorted}, 实际 {actual_brands}")
            for lbl in labels:
                print(f"      -> {lbl['tag_id']} ({lbl['tag_cn']}): type={lbl['brand_type']}, conf={lbl['confidence']}")

    print(f"\n测试结果: {passed}/{passed + failed} 通过 ({passed / (passed + failed) * 100:.1f}%)")

    # 品类过滤测试
    print("\n--- 品类过滤测试 ---")
    labels = apply_brand_labels("Spectra pump review")
    keep = filter_competitor_voc(labels, keep_categories=["吸奶器"])
    print(f"  Spectra(吸奶器) in [吸奶器] → {'保留' if keep else '剔除'} {'PASS' if keep else 'FAIL'}")

    labels = apply_brand_labels("BabyBjorn carrier review")
    keep = filter_competitor_voc(labels, keep_categories=["吸奶器"])
    print(f"  BabyBjorn(母婴综合) in [吸奶器] → {'保留' if keep else '剔除'} {'FAIL' if keep else 'PASS'}")

    # 情感审计
    print("\n--- 品牌情感审计 ---")
    for bt in ["own_brand", "direct_competitor", "indirect_competitor"]:
        sample_text = {
            "own_brand": "Momcozy is great",
            "direct_competitor": "Spectra is noisy",
            "indirect_competitor": "BabyBjorn is bulky",
        }[bt]
        lbls = apply_brand_labels(sample_text)
        if lbls:
            lbl = lbls[0]
            print(f"  {bt}: sentiment_preset={lbl['sentiment_preset']}, calibrated={lbl['sentiment_calibrated']}")

    print("=" * 70)
    return passed, failed


if __name__ == "__main__":
    _test()

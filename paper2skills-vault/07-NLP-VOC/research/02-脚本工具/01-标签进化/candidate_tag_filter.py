"""候选标签过滤（Phase 2.3-2.4 简化版）

对 gap_detector 生成的原始候选标签进行质量过滤：
1. 去除噪音（系统标记、URL、无意义组合）
2. 合并同类标签
3. 保留高频、有明确语义的候选
4. 输出验证后的候选标签
"""

import json
from collections import Counter
from pathlib import Path


# 噪音模式
NOISE_PATTERNS = [
    "soon_possible", "important_margin", "kleanpal_pro", "yiv_", "https_",
    "url_", "com_", "width_", "height_", "list_", "items_", "item_",
    "upn_", "msn_", "vous_", "nous_", "les_", "des_", "pour_", "est_",
    "pas_", "sur_", "que_", "une_", "qui_", "dans_", "plus_", "fait_",
    "tout_", "avec_", "son_", "sont_", "comme_", "tous_", "deux_",
    "please_", "thank_", "regards_", "sincerely_", "dear_", "kindly_",
    "note_", "attached_", "attachment_", "enclosure_", "click_", "link_",
    "page_", "website_", "site_", "online_", "web_", "image_", "photo_",
    "picture_", "video_", "file_", "document_",
]

# 合法产品相关词根（保留包含这些词的候选）
VALID_PRODUCT_ROOTS = [
    "bath", "bottle", "breast", "pump", "milk", "baby", "food", "mattress",
    "pillow", "blanket", "towel", "diaper", "wipe", "warmer", "sterilizer",
    "stroller", "carrier", "sling", "wrap", "band", "belt", "bra", "nursing",
    "sleep", "bag", "pacifier", "teether", "nasal", "aspirator", "thermometer",
    "scale", "monitor", "noise", "purifier", "fan", "heater", "cooler",
    "chair", "highchair", "table", "tray", "cup", "spoon", "bowl", "plate",
    "mat", "pad", "cover", "sheet", "case", "bag", "pack", "pouch",
    "nipple", "shield", "cream", "lotion", "oil", "soap", "shampoo",
    "brush", "comb", "clipper", "trimmer", "nail", "tooth", "oral",
    "massage", "pump", "suction", "flange", "tubing", "valve", "membrane",
    "motor", "battery", "charger", "cable", "cord", "plug", "adapter",
    "strap", "buckle", "clip", "hook", "loop", "velcro", "zipper",
]


def is_noise(tag_en: str) -> bool:
    """判断是否为噪音标签"""
    # 包含噪音模式
    for noise in NOISE_PATTERNS:
        if noise in tag_en:
            return True

    # 纯数字或太短
    if tag_en.replace("_", "").isdigit():
        return True
    if len(tag_en) < 4:
        return True

    # 检查是否包含至少一个有效产品词根
    has_valid_root = any(root in tag_en for root in VALID_PRODUCT_ROOTS)

    # 如果不包含任何有效词根，且长度较短，可能是噪音
    if not has_valid_root and len(tag_en) < 8:
        return True

    return False


def merge_similar_tags(candidates: list[dict]) -> list[dict]:
    """合并相似标签"""
    # 按 tag_en 分组，保留 support_count 最大的
    groups = {}
    for c in candidates:
        key = c["tag_en"]
        if key not in groups:
            groups[key] = c
        elif c["support_count"] > groups[key]["support_count"]:
            groups[key] = c

    return list(groups.values())


def main():
    print("=" * 70)
    print("Phase 2.3-2.4: 候选标签过滤与合并")
    print("=" * 70)

    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/tag_gap_analysis"

    # 加载原始候选
    raw_path = output_dir / "candidate_tags_raw.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"\n  原始候选: {len(candidates)} 个")

    # 过滤噪音
    filtered = [c for c in candidates if not is_noise(c["tag_en"])]
    removed = len(candidates) - len(filtered)
    print(f"  过滤噪音: {removed} 个 -> 剩余 {len(filtered)} 个")

    # 合并相似
    merged = merge_similar_tags(filtered)
    print(f"  合并相似: {len(merged)} 个")

    # 按 support_count 排序
    merged.sort(key=lambda x: x["support_count"], reverse=True)

    # 保存
    filtered_path = output_dir / "candidate_tags_filtered.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n  过滤后候选: {filtered_path}")

    # 打印 Top 30
    print(f"\n--- Top 30 候选标签 ---")
    for c in merged[:30]:
        print(f"  [{c['applicable_category']}] {c['tag_en']} ({c['support_count']}) -> {c['suggested_aipl']}")

    # 审计
    audit = {
        "phase": "2.3-2.4",
        "description": "候选标签过滤与合并",
        "raw_count": len(candidates),
        "filtered_count": len(filtered),
        "removed_count": removed,
        "merged_count": len(merged),
        "top_20": [
            {"tag_en": c["tag_en"], "category": c["applicable_category"], "count": c["support_count"]}
            for c in merged[:20]
        ],
        "output_path": str(filtered_path),
    }
    audit_path = output_dir / "phase2_3_4_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 2.3-2.4 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""品牌关键词库（Phase 4 T1）

构建 Momcozy 自有品牌 + 竞品品牌的分层关键词库。
用于品牌识别、竞品 VOC 分层打标。

品牌分层：
- own_brand: 自有品牌（Momcozy）
- direct_competitor: 直接竞品（核心品类竞争品牌：吸奶器/哺乳内衣/消毒器）
- indirect_competitor: 间接竞品（相关品类品牌）

设计原则：
1. 品牌关键词覆盖常见拼写变体（大小写、空格、连字符）
2. 每个品牌标注所属品类，用于品线推断
3. 品牌识别与产品关键词推断解耦
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BrandConfig:
    """品牌配置"""
    name: str
    brand_type: str  # own_brand | direct_competitor | indirect_competitor
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    priority: int = 1
    notes: Optional[str] = None


# ── 品牌关键词库 ───────────────────────────────────────────────────

BRAND_KEYWORD_LIBRARY: dict[str, BrandConfig] = {
    # === 自有品牌 ===
    "Momcozy": BrandConfig(
        name="Momcozy",
        brand_type="own_brand",
        keywords=[
            "momcozy", "mom cozy", "momcozy.com",
        ],
        categories=["全部品线"],
        priority=1,
    ),

    # === 直接竞品：吸奶器/哺乳内衣核心竞品 ===
    "Spectra": BrandConfig(
        name="Spectra",
        brand_type="direct_competitor",
        keywords=[
            "spectra", "spectra s1", "spectra s2", "spectra pump",
            "spectra s1 plus", "spectra s2 plus", "spectra synergy gold",
            "spectra 9 plus",
        ],
        categories=["吸奶器", "吸奶器配件"],
        priority=2,
    ),
    "Medela": BrandConfig(
        name="Medela",
        brand_type="direct_competitor",
        keywords=[
            "medela", "medela pump", "medela freestyle", "medela symphony",
            "medela swing", "medela pump in style", "medela sonata",
            "freestyle flex", "pump in style",
        ],
        categories=["吸奶器", "吸奶器配件", "哺乳内衣"],
        priority=2,
    ),
    "Willow": BrandConfig(
        name="Willow",
        brand_type="direct_competitor",
        keywords=[
            "willow", "willow pump", "willow go", "willow 3.0",
            "willow hands-free", "willow wearable", "willow hands free",
        ],
        categories=["吸奶器"],
        priority=2,
    ),
    "Elvie": BrandConfig(
        name="Elvie",
        brand_type="direct_competitor",
        keywords=[
            "elvie", "elvie pump", "elvie stride", "elvie curve",
            "elvie wearable", "elvie breast pump", "elvie hands free",
        ],
        categories=["吸奶器"],
        priority=2,
    ),
    "Lansinoh": BrandConfig(
        name="Lansinoh",
        brand_type="direct_competitor",
        keywords=[
            "lansinoh", "lansinoh pump", "lansinoh breast pump",
            "lansinoh nursing", "lansinoh breastfeeding",
            "lansinoh signature pro",
        ],
        categories=["吸奶器", "哺乳内衣", "吸奶器配件"],
        priority=2,
    ),
    "Haakaa": BrandConfig(
        name="Haakaa",
        brand_type="direct_competitor",
        keywords=[
            "haakaa", "haakaa pump", "haakaa collector",
            "haakaa silicone", "haakaa manual",
        ],
        categories=["吸奶器配件"],
        priority=2,
    ),
    "Philips Avent": BrandConfig(
        name="Philips Avent",
        brand_type="direct_competitor",
        keywords=[
            "philips avent", "avent", "avent pump", "avent sterilizer",
            "avent bottles", "philips avent breast pump",
            "avent double", "avent comfort",
        ],
        categories=["吸奶器", "喂养电器"],
        priority=2,
    ),

    # === 间接竞品：母婴综合品类 ===
    "Comfelie": BrandConfig(
        name="Comfelie",
        brand_type="indirect_competitor",
        keywords=[
            "comfelie", "comfelie bra", "comfelie nursing",
        ],
        categories=["内衣服饰"],
        priority=3,
    ),
    "Kindred Bravely": BrandConfig(
        name="Kindred Bravely",
        brand_type="indirect_competitor",
        keywords=[
            "kindred bravely", "kindred bravely bra", "kindredbravely",
        ],
        categories=["内衣服饰"],
        priority=3,
    ),
    "BabyBjorn": BrandConfig(
        name="BabyBjorn",
        brand_type="indirect_competitor",
        keywords=[
            "babybjorn", "baby bjorn", "babybjorn carrier", "babybjorn bouncer",
        ],
        categories=["母婴综合护理"],
        priority=3,
    ),
    "Ergobaby": BrandConfig(
        name="Ergobaby",
        brand_type="indirect_competitor",
        keywords=[
            "ergobaby", "ergo baby", "ergobaby carrier", "ergo 360",
            "ergobaby omni", "ergo baby carrier",
        ],
        categories=["母婴综合护理"],
        priority=3,
    ),
    "Dr. Brown's": BrandConfig(
        name="Dr. Brown's",
        brand_type="indirect_competitor",
        keywords=[
            "dr. brown", "dr browns", "dr brown's bottle", "dr browns bottles",
            "dr brown's sterilizer", "dr brown warmer",
        ],
        categories=["喂养电器"],
        priority=3,
    ),
    "Tommee Tippee": BrandConfig(
        name="Tommee Tippee",
        brand_type="indirect_competitor",
        keywords=[
            "tommee tippee", "tommee tippee bottle", "tommee tippee sterilizer",
            "tommee tippee warmer", "tomme tippee",
        ],
        categories=["喂养电器"],
        priority=3,
    ),
    "Nanit": BrandConfig(
        name="Nanit",
        brand_type="indirect_competitor",
        keywords=[
            "nanit", "nanit monitor", "nanit camera", "nanit plus",
            "nanit pro",
        ],
        categories=["智能母婴电器"],
        priority=3,
    ),
    "Owlet": BrandConfig(
        name="Owlet",
        brand_type="indirect_competitor",
        keywords=[
            "owlet", "owlet sock", "owlet monitor", "owlet dream",
            "owlet camera", "owlet dream sock", "owlet smart sock",
        ],
        categories=["智能母婴电器"],
        priority=3,
    ),
    "Wabi Baby": BrandConfig(
        name="Wabi Baby",
        brand_type="indirect_competitor",
        keywords=[
            "wabi baby", "wabi sterilizer", "wabibaby",
        ],
        categories=["喂养电器"],
        priority=3,
    ),
    "Boppy": BrandConfig(
        name="Boppy",
        brand_type="indirect_competitor",
        keywords=[
            "boppy", "boppy pillow", "boppy lounger", "boppy nursing",
        ],
        categories=["母婴综合护理"],
        priority=3,
    ),
}


# ── 快捷查询表 ─────────────────────────────────────────────────────

OWN_BRAND_KEYWORDS: set[str] = {
    kw for name, config in BRAND_KEYWORD_LIBRARY.items()
    if config.brand_type == "own_brand"
    for kw in config.keywords
}

DIRECT_COMPETITOR_KEYWORDS: set[str] = {
    kw for name, config in BRAND_KEYWORD_LIBRARY.items()
    if config.brand_type == "direct_competitor"
    for kw in config.keywords
}

INDIRECT_COMPETITOR_KEYWORDS: set[str] = {
    kw for name, config in BRAND_KEYWORD_LIBRARY.items()
    if config.brand_type == "indirect_competitor"
    for kw in config.keywords
}

ALL_BRAND_KEYWORDS: set[str] = {
    kw for config in BRAND_KEYWORD_LIBRARY.values()
    for kw in config.keywords
}

OWN_BRAND_NAMES: list[str] = [
    name for name, config in BRAND_KEYWORD_LIBRARY.items()
    if config.brand_type == "own_brand"
]

DIRECT_COMPETITOR_NAMES: list[str] = [
    name for name, config in BRAND_KEYWORD_LIBRARY.items()
    if config.brand_type == "direct_competitor"
]

INDIRECT_COMPETITOR_NAMES: list[str] = [
    name for name, config in BRAND_KEYWORD_LIBRARY.items()
    if config.brand_type == "indirect_competitor"
]


# ── 品牌识别函数 ───────────────────────────────────────────────────

def identify_brand(text: str) -> list[dict]:
    """识别文本中的品牌

    Returns: [{"brand": str, "brand_type": str, "matched_kw": str, "confidence": float}]
    """
    text_lower = text.lower()
    results: list[dict] = []
    matched_brands: set[str] = set()

    for brand_name, config in BRAND_KEYWORD_LIBRARY.items():
        if brand_name in matched_brands:
            continue
        for kw in config.keywords:
            kw_lower = kw.lower()
            # 品牌名通常>=5字符，直接子串匹配
            if len(kw_lower) >= 5 and kw_lower in text_lower:
                confidence = 0.99 if len(kw_lower) >= 8 else 0.95
                results.append({
                    "brand": brand_name,
                    "brand_type": config.brand_type,
                    "matched_kw": kw,
                    "confidence": confidence,
                })
                matched_brands.add(brand_name)
                break
            # 短品牌名（如"avent"）用整词边界
            elif len(kw_lower) < 5:
                import re
                pattern = re.compile(r'\b' + re.escape(kw_lower) + r'\b')
                if pattern.search(text_lower):
                    results.append({
                        "brand": brand_name,
                        "brand_type": config.brand_type,
                        "matched_kw": kw,
                        "confidence": 0.90,
                    })
                    matched_brands.add(brand_name)
                    break

    return results


def get_brand_type(brand_name: str) -> str:
    """获取品牌类型"""
    config = BRAND_KEYWORD_LIBRARY.get(brand_name)
    return config.brand_type if config else "unknown"


def get_brand_categories(brand_name: str) -> list[str]:
    """获取品牌所属品类"""
    config = BRAND_KEYWORD_LIBRARY.get(brand_name)
    return config.categories if config else []


# ── 审计函数 ───────────────────────────────────────────────────────

def self_audit() -> dict:
    """品牌关键词库自我审计"""
    audit = {
        "total_brands": len(BRAND_KEYWORD_LIBRARY),
        "own_brands": len(OWN_BRAND_NAMES),
        "direct_competitors": len(DIRECT_COMPETITOR_NAMES),
        "indirect_competitors": len(INDIRECT_COMPETITOR_NAMES),
        "total_keywords": len(ALL_BRAND_KEYWORDS),
        "own_brand_keywords": len(OWN_BRAND_KEYWORDS),
        "direct_competitor_keywords": len(DIRECT_COMPETITOR_KEYWORDS),
        "indirect_competitor_keywords": len(INDIRECT_COMPETITOR_KEYWORDS),
        "brand_list": list(BRAND_KEYWORD_LIBRARY.keys()),
        "coverage_categories": sorted({
            cat for config in BRAND_KEYWORD_LIBRARY.values()
            for cat in config.categories
        }),
    }
    return audit


# ── 测试 ───────────────────────────────────────────────────────────

def _test():
    """快速测试"""
    test_cases = [
        ("I love my Momcozy pump, much better than my Spectra", ["Momcozy", "Spectra"]),
        ("The Medela Freestyle is so quiet", ["Medela"]),
        ("Willow pump vs Elvie stride comparison", ["Willow", "Elvie"]),
        ("Boppy pillow is comfortable", ["Boppy"]),
        ("No brand mentioned here", []),
        ("I switched from Haakaa to Momcozy", ["Haakaa", "Momcozy"]),
        ("avent bottles are cheaper", ["Philips Avent"]),
        ("The owlet smart sock is great", ["Owlet"]),
    ]

    print("=" * 70)
    print("品牌关键词库自证测试")
    print("=" * 70)

    passed = 0
    failed = 0
    for text, expected_brands in test_cases:
        results = identify_brand(text)
        actual = sorted([r["brand"] for r in results])
        expected = sorted(expected_brands)
        status = "PASS" if actual == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] '{text[:50]}...'")
        if status == "FAIL":
            print(f"    期望: {expected}")
            print(f"    实际: {actual}")
            print(f"    详情: {results}")

    print(f"\n测试结果: {passed}/{passed + failed} 通过 ({passed / (passed + failed) * 100:.1f}%)")
    print("=" * 70)

    # 审计报告
    audit = self_audit()
    print("\n品牌关键词库审计报告:")
    print(f"  品牌总数: {audit['total_brands']}")
    print(f"  自有品牌: {audit['own_brands']}")
    print(f"  直接竞品: {audit['direct_competitors']}")
    print(f"  间接竞品: {audit['indirect_competitors']}")
    print(f"  关键词总数: {audit['total_keywords']}")
    print(f"  覆盖品类: {', '.join(audit['coverage_categories'])}")

    return passed, failed


if __name__ == "__main__":
    _test()

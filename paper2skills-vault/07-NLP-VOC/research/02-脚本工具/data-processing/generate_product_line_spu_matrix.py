"""品线 × SPU × 品牌 三维标签宽表生成器

从 V3.3 打标结果 + 品线分类 + SPU提取，生成业务级宽表：
1. 品线 × 标签 宽表
2. SPU × 标签 宽表（产品系列级别）
3. 品牌 × 品线 × 标签 三维透视表

输出: CSV 格式，可直接导入 BI 工具
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/00-归档资料/labeling-outputs/v3.3")
EXPORT_DIR = INPUT_DIR / "product_line_spu_matrices"

# 动态导入多语言关键词（momcozy_integration 子目录）
_MOMCOZY_DIR = Path(__file__).parent / "momcozy_integration"
if str(_MOMCOZY_DIR) not in sys.path:
    sys.path.insert(0, str(_MOMCOZY_DIR))

try:
    from multilingual_keywords import MULTILINGUAL_KEYWORDS, get_all_keywords
    _HAS_MULTILANG = True
except ImportError:
    _HAS_MULTILANG = False


# ──────────────────────────────────────────────
# 品线分类规则（复用 product_line_classifier.py）
# ──────────────────────────────────────────────

# ── 品线分类规则（五级匹配策略）──────────────────────────────────
#
# Level -1: 四级类目直接映射（最高优先级，仅 momcozy 数据可用）
# Level 0:  客服/物流文本快速分流
# Level 1:  高置信度完整关键词 → 直接匹配
# Level 2:  中置信度 → 核心词 + 上下文推断
# Level 3:  品牌强关联 → 品牌名 + 品类上下文
# Level 4:  标签辅助推断 → 兜底

# ── 词干化工具 ──────────────────────────────────────────────────

def simple_stem(word: str) -> str:
    """简化词干化：去除常见后缀"""
    w = word.lower()
    # 顺序很重要：先长后缀
    for suffix in ["ing", "ings", "ed", "edly", "er", "ers", "est", "ly", "ness", "ment", "ments"]:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[:-len(suffix)]
    # 复数
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def stem_text(text: str) -> set[str]:
    """将文本转为词干集合"""
    words = re.findall(r"[a-z][a-z'\-\/]*[a-z]", text.lower())
    return {simple_stem(w) for w in words if len(w) >= 3}


# ── 强特征词：单独出现即足以推断品线 ─────────────────────────────
STRONG_FEATURE_WORDS = {
    # 品线: [(词或词组, 权重), ...]
    "sound_machine": [
        ("white noise", 10), ("brown noise", 10), ("pink noise", 10),
        ("lullaby", 8), ("nature sounds", 8), ("sound machine", 10),
        ("sleep machine", 10), ("white noise machine", 10),
        ("hatch", 6), ("dohm", 6), ("noise machine", 8),
    ],
    "crib_playard": [
        ("bassinet", 10), ("playard", 10), ("playpen", 10),
        ("travel crib", 10), ("pack n play", 10), ("pack and play", 10),
        ("co sleeper", 8), ("co-sleeper", 8), ("bedside sleeper", 8),
        ("mini crib", 8), ("convertible crib", 8),
    ],
    "air_purifier": [
        ("hepa", 10), ("air purifier", 10), ("air purifiers", 10),
        ("air cleaner", 8), ("allergen", 6), ("air filtration", 8),
    ],
    "sterilizer": [
        ("uv sterilizer", 10), ("steam sterilizer", 10), ("bottle sterilizer", 10),
        ("pacifier sterilizer", 10), ("microwave sterilizer", 10),
        ("sterilization", 8), ("sterilizing", 8),
    ],
    "red_light_therapy": [
        ("red light therapy", 10), ("light therapy", 8), ("infrared therapy", 10),
        ("led therapy", 8), ("phototherapy", 10), ("redlight", 8),
    ],
    "baby_monitor": [
        ("baby monitor", 10), ("video monitor", 10), ("audio monitor", 10),
        ("wifi monitor", 10), ("breathing monitor", 10), ("movement monitor", 10),
        ("nanit", 10), ("owlet", 10), ("miku", 10), ("cocoon cam", 10),
    ],
    "humidifier": [
        ("humidifier", 10), ("humidifiers", 10), ("vaporizer", 10),
        ("cool mist", 8), ("warm mist", 8), ("ultrasonic humidifier", 10),
    ],
    "postpartum_recovery": [
        ("faja", 10), ("waist trainer", 10), ("shapewear", 10),
        ("compression garment", 10), ("compression band", 10), ("belly binder", 10),
        ("postpartum belly", 10), ("postpartum girdle", 10), ("c section recovery", 10),
        ("c-section recovery", 10), ("recovery belt", 10),
        ("body shaper", 10), ("tummy control", 10), ("snatched", 8),
    ],
    "nursing_bra": [
        ("pumping bra", 10), ("pumping bras", 10), ("hands free bra", 10),
        ("hands-free bra", 10), ("hands-free pumping bra", 10),
        ("breast pump bra", 10), ("nursing cami", 10), ("nursing tank", 10),
    ],
    "nipple_cream": [
        ("lanolin", 10), ("nipple butter", 10), ("nipple balm", 10),
        ("nipple ointment", 10), ("nipple cream", 10),
    ],
    "breast_pad": [
        ("nursing pad", 10), ("breast pad", 10), ("bra pad", 10),
        ("leak pad", 10), ("disposable pad", 8), ("washable pad", 8),
    ],
    "breast_milk_storage": [
        ("milk bag", 10), ("milk bags", 10), ("storage bag", 8),
        ("milk freezer", 10), ("milk stash", 10), ("freeze milk", 10),
        ("frozen milk", 10), ("milk container", 10),
    ],
    "bottle_washer": [
        ("bottle washer", 10), ("bottle cleaner", 10), ("brush cleaner", 8),
    ],
    "wearable_breast_pump": [
        ("wearable breast pump", 10), ("wearable pump", 10),
        ("hands free pump", 10), ("hands-free pump", 10),
        ("in-bra pump", 10), ("in bra pump", 10), ("wireless pump", 10),
    ],
    "car_seat": [
        ("infant car seat", 10), ("convertible car seat", 10), ("booster seat", 10),
        ("5 point harness", 10), ("rear facing", 8), ("forward facing", 8),
    ],
    "baby_carrier": [
        ("ring sling", 10), ("soft structured carrier", 10), ("ssc", 8),
        ("moby wrap", 10), ("solly wrap", 10), ("front carrier", 8), ("back carrier", 8), ("hip carrier", 8),
    ],
    "baby_bottle": [
        ("sippy cup", 10), ("training cup", 10), ("straw cup", 10), ("transition cup", 10),
        ("bottle nipple", 10), ("slow flow", 8), ("medium flow", 8), ("fast flow", 8),
    ],
    "stroller": [
        ("jogging stroller", 10), ("travel stroller", 10), ("umbrella stroller", 10),
        ("double stroller", 10), ("twin stroller", 10), ("stroller frame", 10),
    ],
    "diaper_bag": [
        ("diaper tote", 10), ("changing bag", 10), ("diaper bag backpack", 10),
    ],
}


# ── 词干扩展的关键词匹配 ─────────────────────────────────────────
# 对 Level 1 关键词做自动词干扩展（复数、ing、ed 等）

def expand_keywords(keywords: list[str]) -> list[str]:
    """为关键词列表生成常见变体"""
    expanded = set(keywords)
    for kw in keywords:
        if " " not in kw:  # 只扩展单字词
            # 复数
            if not kw.endswith("s"):
                expanded.add(kw + "s")
            if kw.endswith("y"):
                expanded.add(kw[:-1] + "ies")
            if kw.endswith("e"):
                expanded.add(kw + "s")
            else:
                expanded.add(kw + "es")
            # ing
            expanded.add(kw + "ing")
            # ed
            expanded.add(kw + "ed")
            # er
            expanded.add(kw + "er")
            # 's
            expanded.add(kw + "'s")
    return list(expanded)


# Level 1: 完整关键词匹配（含自动扩展）
PRODUCT_LINE_RULES = {
    "pregnancy_pillow": ["pregnancy pillow", "body pillow", "wedge pillow", "maternity pillow", "pregnant pillow", "boppy pillow", "c pillow", "u shaped pillow", "u-shaped pillow", "full body pillow"],
    "nursing_bra": ["nursing bra", "breastfeeding bra", "pumping bra", "nursing bras", "breastfeeding bras", "pumping bras", "maternity bra", "nursing tank", "nursing cami", "breast pump bra", "hands free bra", "nursing top"],
    "wearable_breast_pump": ["wearable pump", "wearable breast pump", "hands free pump", "hands-free pump", "handsfree pump", "wireless pump", "portable pump", "discreet pump", "in-bra pump", "in bra pump"],
    "breast_pump": ["breast pump", "breastpump", "pump and dump", "pumping at work", "pumping session", "pump parts", "pump flange", "pump tubing", "pump valve"],
    "bottle_warmer": ["bottle warmer", "wipe warmer", "portable bottle warmer", "baby bottle warmer", "milk warmer", "formula warmer"],
    "sterilizer": ["sterilizer", "steriliser", "uv sterilizer", "steam sterilizer", "bottle sterilizer", "pacifier sterilizer", "baby bottle sterilizer", "microwave sterilizer", "electric sterilizer"],
    "humidifier": ["humidifier", "vaporizer", "cool mist humidifier", "warm mist humidifier", "ultrasonic humidifier", "baby humidifier", "nursery humidifier"],
    "air_purifier": ["air purifier", "air purifiers", "hepa filter", "air cleaner", "air quality", "air filtration", "room purifier"],
    "sound_machine": ["sound machine", "white noise", "white noise machine", "noise machine", "baby sound machine", "sleep machine", "soothing sounds", "lullaby machine", "nature sounds"],
    "red_light_therapy": ["red light therapy", "red light", "light therapy", "redlight therapy", "led light therapy", "infrared therapy", "phototherapy", "light therapy lamp", "red light device"],
    "stroller": ["stroller", "pram", "pushchair", "buggy", "baby stroller", "jogging stroller", "travel stroller", "umbrella stroller", "double stroller", "twin stroller", "stroller frame"],
    "car_seat": ["car seat", "infant car seat", "convertible car seat", "booster seat", "carseat", "infant carseat", "baby car seat", "5 point harness", "rear facing", "forward facing"],
    "diaper_bag": ["diaper bag", "nappy bag", "diaper backpack", "diaper tote", "changing bag", "baby bag", "diaper bag backpack"],
    "baby_bottle": ["baby bottle", "feeding bottle", "sippy cup", "training cup", "straw cup", "transition cup", "nipple bottle", "slow flow", "medium flow", "fast flow", "bottle nipple", "biberon", "biberons"],
    "baby_monitor": ["baby monitor", "video monitor", "audio monitor", "wifi monitor", "smart baby monitor", "breathing monitor", "movement monitor", "nanit", "owlet", "miku", "cocoon cam"],
    "baby_carrier": ["baby carrier", "wrap carrier", "ring sling", "soft structured carrier", "ssc", "ergo baby", "tula", "lillebaby", "moby wrap", "solly wrap", "baby wrap", "front carrier", "back carrier", "hip carrier"],
    "breast_milk_storage": ["milk storage", "breast milk storage", "milk bag", "milk bags", "storage bag", "storage bags", "milk container", "breast milk bag", "milk freezer", "milk stash", "freeze milk", "frozen milk"],
    "postpartum_recovery": ["postpartum", "postpartum recovery", "postpartum belly", "postpartum girdle", "c section recovery", "c-section recovery", "after birth", "post birth", "post delivery", "recovery belt", "belly wrap", "postpartum wrap", "compression garment", "faja", "waist trainer", "shapewear", "compression band", "belly binder", "body shaper", "tummy control", "belly control", "midsection", "body suit", "bodysuit"],
    "baby_clothing": ["baby clothes", "baby clothing", "onesie", "onesies", "footie", "footies", "newborn clothes", "infant clothes", "baby outfit", "dress", "dresses", "jumper", "jumpers"],
    # 新增品线
    "bottle_washer": ["bottle washer", "bottle cleaner", "brush cleaner", "bottle brush"],
    "nipple_cream": ["nipple cream", "nipple butter", "lanolin", "nipple balm", "nipple ointment"],
    "breast_pad": ["nursing pad", "breast pad", "bra pad", "disposable pad", "washable pad", "leak pad"],
    "baby_wipe": ["baby wipe", "baby wipes", "wet wipe", "wet wipes", "diaper wipe"],
    "crib_playard": ["crib", "playard", "playpen", "travel crib", "pack n play", "pack and play", "bassinet", "co sleeper", "co-sleeper", " bedside sleeper", "mini crib", "convertible crib"],
}

# ── 融合多语言关键词 ──────────────────────────────────────────────
# 将 multilingual_keywords.py 中的多语言词并入 PRODUCT_LINE_RULES
if _HAS_MULTILANG:
    for line, lang_dict in MULTILINGUAL_KEYWORDS.items():
        if line in PRODUCT_LINE_RULES:
            for lang, kws in lang_dict.items():
                if lang != "en":  # 英语已在基础规则中
                    for kw in kws:
                        if kw not in PRODUCT_LINE_RULES[line]:
                            PRODUCT_LINE_RULES[line].append(kw)
    print(f"[INFO] 已融合多语言关键词，覆盖 {len(MULTILINGUAL_KEYWORDS)} 个品线")

# Level 2: 核心词 + 上下文推断
CONTEXT_RULES = {
    "breast_pump": {
        "keywords": ["pump", "pumping", "milchpumpe"],
        "context": ["suction", "flange", "milk", "letdown", "let down", "oversupply", "low supply",
                    "spectra", "medela", "lansinoh", "elvie", "willow", "ameda", "hygeia",
                    "tubing", "valve", "diaphragm", "backflow", "express", "expressed"],
    },
    "wearable_breast_pump": {
        "keywords": ["pump", "pumping", "milchpumpe"],
        "context": ["wearable", "hands free", "hands-free", "wireless", "in bra", "in-bra",
                    "discreet", "portable", "elvie", "willow", "momcozy s12", "momcozy m", "wearables"],
    },
    "pregnancy_pillow": {
        "keywords": ["pillow"],
        "context": ["pregnancy", "pregnant", "maternity", "boppy", "wedge", "c shaped", "u shaped",
                    "body pillow", "side sleeper", "belly support", "back support",
                    "comfy", "deliver", "trimester", "side sleeping", "stomach sleeper", "back sleeper", "hole"],
    },
    "baby_bottle": {
        "keywords": ["bottle", "bottles"],
        "context": ["nipple", "nipples", "slow flow", "medium flow", "fast flow", "dr brown",
                    "avent", "tommee tippee", "mam bottle", "comotomo", "formula", "feeding"],
    },
    "nursing_bra": {
        "keywords": ["bra", "bras"],
        "context": ["nursing", "breastfeeding", "pumping", "hands free", "clip down", "nursing clip",
                    "kindred bravely", "cake maternity", "sublime"],
    },
    "baby_monitor": {
        "keywords": ["monitor"],
        "context": ["baby", "nursery", "cry", "crying", "sleep", "breathing", "movement",
                    "nanit", "owlet", "miku", "wifi", "wi-fi", "camera", "connect", "password", "internet"],
    },
    "sterilizer": {
        "keywords": ["sterilize", "sterilise", "sanitize"],
        "context": ["bottle", "pacifier", "uv", "steam", "microwave", "wabi baby", "papablic"],
    },
    "bottle_warmer": {
        "keywords": ["warmer"],
        "context": ["bottle", "milk", "formula", "wipe", "dr brown", "avent"],
    },
    "stroller": {
        "keywords": ["stroller", "pram", "pushchair"],
        "context": ["buggy", "jogging", "travel", "umbrella", "double", "uppababy", "baby jogger",
                    "bugaboo", "nuna", "cybex", "fold", "unfold", "recline"],
    },
    "car_seat": {
        "keywords": ["car seat", "carseat"],
        "context": ["infant", "convertible", "booster", "5 point", "rear facing", "forward facing",
                    "britax", "graco", "chicco", "nuna", "cybex", "base", "install", "harness", "buckle", "weight", "height"],
    },
    "humidifier": {
        "keywords": ["humidifier"],
        "context": ["mist", "vapor", "ultrasonic", "nursery", "cool mist", "warm mist", "filter", "humidity", "bedroom", "water", "leaked"],
    },
    "air_purifier": {
        "keywords": ["purifier"],
        "context": ["air", "hepa", "filter", "cleaner", "allergen", "dust", "pollen", "smoke", "dogs", "cats", "pet", "smell", "odor", "room"],
    },
    "sound_machine": {
        "keywords": ["noise", "sound"],
        "context": ["white noise", "brown noise", "lullaby", "nature", "rain", "ocean", "shush",
                    "sleep", "baby", "nursery", "hatch", "dohm", "alarm", "display"],
    },
    "baby_carrier": {
        "keywords": ["carrier", "wrap", "sling"],
        "context": ["baby", "ergo", "tula", "lillebaby", "moby", "solly", "front carry", "back carry",
                    "newborn", "toddler", "hip seat"],
    },
    "postpartum_recovery": {
        "keywords": ["postpartum", "faja", "compression", "waist", "shaper", "snatched"],
        "context": ["belly", "c section", "c-section", "recovery", "girdle", "wrap", "belt", "shape", "stomach", "wear", "binder", "tummy", "snatched", "rolls", "roll down", "rolldown", "smooth", "sucked", "midsection", "compressing", "compression", "belly control", "size down", "opening", "bone"],
    },
    "baby_wipe": {
        "keywords": ["wipe", "wipes"],
        "context": ["baby", "diaper", "rash", "sensitive", "soft", "thick", "wet"],
    },
}

# Level 3: 品牌强关联推断
BRAND_PRODUCT_LINE = {
    "spectra": "breast_pump",
    "medela": "breast_pump",
    "lansinoh": "breast_pump",
    "elvie": "wearable_breast_pump",
    "willow": "wearable_breast_pump",
    "ameda": "breast_pump",
    "hygeia": "breast_pump",
    "nanit": "baby_monitor",
    "owlet": "baby_monitor",
    "miku": "baby_monitor",
    "wabi baby": "sterilizer",
    "papablic": "sterilizer",
    "dr brown": "baby_bottle",
    "comotomo": "baby_bottle",
    "mam": "baby_bottle",
    "avent": "baby_bottle",
    "tommee tippee": "baby_bottle",
    "uppababy": "stroller",
    "baby jogger": "stroller",
    "bugaboo": "stroller",
    "britax": "car_seat",
    "graco": "car_seat",
    "chicco": "car_seat",
    "kindred bravely": "nursing_bra",
    "boppy": "pregnancy_pillow",
    "ergo baby": "baby_carrier",
    "tula": "baby_carrier",
    "moby": "baby_carrier",
    "hatch": "sound_machine",
    "dohm": "sound_machine",
}

# 专属品牌：该品牌只做一种产品，品牌名出现即足够推断品线
EXCLUSIVE_BRANDS = {"elvie", "willow", "nanit", "owlet", "miku", "wabi baby", "papablic", "hatch", "dohm"}

LINE_PRIORITY = [
    "breast_pump", "red_light_therapy",
    "sound_machine", "air_purifier", "humidifier", "baby_monitor",
    "sterilizer", "bottle_warmer", "bottle_washer", "pregnancy_pillow", "nursing_bra",
    "wearable_breast_pump", "baby_carrier", "stroller", "car_seat", "diaper_bag", "baby_bottle",
    "breast_milk_storage", "postpartum_recovery", "baby_clothing",
    "nipple_cream", "breast_pad", "baby_wipe", "crib_playard", "customer_service",
]


# ── 客服/物流相关关键词 ────────────────────────────────────────────
SERVICE_KEYWORDS = [
    "customer service", "conversation with", "web user", "follow-up", "follow up",
    "shipping", "delivery", "delivered", "arrived", "package", "parcel",
    "return", "refund", "exchange", "warranty", "replacement",
    "order", "ordered", "purchase", "bought",
    "schnelle lieferung", "livraison rapide", "fast shipping", "quick delivery",
    # 德语物流/客服
    "gut geklappt", "super geklappt", "sehr zufrieden", "alles ok", "gut verpackt",
    "beschädigt", "zustellung", "bestellung", "lieferung", "schnelle",
    # 法语物流/客服
    "envoie rapide", "produit de qualité", "commander", "colis est perdu",
    "bonne expérience", "livraison rapide", "service client",
    # 意大利语物流/客服
    "spedizione rapida", "imballaggio accurato", "spedizione prime",
    # 西班牙语物流/客服
    "envío rápido", "producto de calidad", "entrega rápida",
]

# ── 标签 → 品线 辅助推断 ─────────────────────────────────────────
TAG_PRODUCT_LINE_HINTS = {
    "white_noise_effectiveness": "sound_machine",
    "nature_sounds_variety": "sound_machine",
    "volume_control_range": "sound_machine",
    "timer_auto_shutoff": "sound_machine",
    "hepa_filter_effectiveness": "air_purifier",
    "filter_replacement_cost": "air_purifier",
    "allergen_dust_removal": "air_purifier",
    "humidity_level_control": "humidifier",
    "tank_capacity_runtime": "humidifier",
    "cool_mist_vs_warm_mist": "humidifier",
    "camera_image_quality_night": "baby_monitor",
    "range_connectivity_issues": "baby_monitor",
    "battery_life_monitor": "baby_monitor",
    "breathing_movement_detection": "baby_monitor",
    "two_way_audio_intercom": "baby_monitor",
    "sterilization_cycle_time": "sterilizer",
    "bottle_capacity_load": "sterilizer",
    "drying_function_quality": "sterilizer",
    "uv_vs_steam_method": "sterilizer",
    "heating_speed_consistency": "bottle_warmer",
    "temperature_accuracy": "bottle_warmer",
    "defrost_breast_milk": "bottle_warmer",
    "fold_unfold_ease": "stroller",
    "maneuverability_steering": "stroller",
    "recline_positions": "stroller",
    "car_seat_compatibility": "stroller",
    "installation_difficulty": "car_seat",
    "safety_certification": "car_seat",
    "head_support_newborn": "car_seat",
    "rear_facing_duration": "car_seat",
    "clip_down_nursing_access": "nursing_bra",
    "support_fullness_large_chest": "nursing_bra",
    "hands_free_pumping_compatible": "nursing_bra",
    "belly_support_side_sleeping": "pregnancy_pillow",
    "back_hip_relief": "pregnancy_pillow",
    "detachable_sections": "pregnancy_pillow",
    "cover_washability": "pregnancy_pillow",
    "ergonomic_hip_positioning": "baby_carrier",
    "back_support_parent": "baby_carrier",
    "forward_inward_facing": "baby_carrier",
    "nipple_flow_rate": "baby_bottle",
    "gas_colic_prevention": "baby_bottle",
    "baby_acceptance_latch": "baby_bottle",
    "skin_improvement_results": "red_light_therapy",
    "treatment_time_frequency": "red_light_therapy",
    "pain_inflammation_relief": "red_light_therapy",
    "c_section_incision_healing": "postpartum_recovery",
    "compression_support_level": "postpartum_recovery",
    "leak_proof_seal": "breast_milk_storage",
    "freezer_space_efficient": "breast_milk_storage",
    "suction_strength_weak": "breast_pump",
    "suction_too_strong_painful": "breast_pump",
    "pump_parts_wear_tear": "breast_pump",
    "flange_size_fit_issue": "breast_pump",
    "discreet_wear_under_clothes": "wearable_breast_pump",
    "leakage_while_wearing": "wearable_breast_pump",
    "bulky_visible_under_clothing": "wearable_breast_pump",
}


def _match_keyword(text_lower: str, kw: str) -> bool:
    """模糊关键词匹配：精确匹配 + 词边界 + 词干扩展 + 连字符变体"""
    if " " in kw:
        # 多词短语：精确子串匹配
        if kw in text_lower:
            return True
        # 连字符变体（如 "hands free" 也匹配 "hands-free"）
        hyphen_version = kw.replace(" ", "-")
        if hyphen_version in text_lower:
            return True
        return False
    # 单字词：词边界匹配，并检查词干
    if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
        return True
    # 词干扩展匹配
    stem = simple_stem(kw)
    text_stems = stem_text(text_lower)
    if stem in text_stems:
        return True
    return False


def classify_product_line(text: str, tags=None, category_lv4: str = None) -> str:
    """六级品线分类：类目映射 → 客服分流 → 强特征词 → 关键词(含词干) → 加权上下文 → 品牌 → 标签辅助

    Args:
        text: 评论文本
        tags: 标签列表（可选）
        category_lv4: 四级类目名（可选），若提供则作为 Level -1 最高优先级
    """
    # ── Level -1: 四级类目直接映射（最高优先级）───────────────────
    if category_lv4:
        # 动态导入 momcozy 类目映射（避免循环依赖）
        try:
            from momcozy_integration.category_to_product_line_mapping import classify_by_category
            cat_line = classify_by_category(category_lv4)
            if cat_line:
                return cat_line
        except ImportError:
            pass  # 无映射表时降级为文本分类

    text_lower = text.lower()
    matches = []

    # ── Level 0: 客服/物流文本快速分流 ────────────────────────────
    service_score = 0
    for kw in SERVICE_KEYWORDS:
        if kw in text_lower:
            service_score += 2 if " " in kw else 1
    if service_score >= 3:
        has_product_kw = any(
            kw in text_lower
            for line_kws in PRODUCT_LINE_RULES.values()
            for kw in line_kws[:3]
        )
        if not has_product_kw:
            return "customer_service"

    # ── Level 1: 强特征词直接推断 ─────────────────────────────────
    # 某些词/词组单独出现即足以推断品线（高特异性）
    strong_votes = Counter()
    for line, features in STRONG_FEATURE_WORDS.items():
        for phrase, weight in features:
            if phrase in text_lower:
                strong_votes[line] += weight
    if strong_votes:
        best_line, best_score = strong_votes.most_common(1)[0]
        if best_score >= 8:
            return best_line

    # ── Level 2: 关键词匹配（含词干扩展）───────────────────────────
    for line, keywords in PRODUCT_LINE_RULES.items():
        for kw in keywords:
            if _match_keyword(text_lower, kw):
                matches.append(line)
                break

    # ── Level 3: 加权上下文推断 ───────────────────────────────────
    if not matches:
        for line, rule in CONTEXT_RULES.items():
            # 排除规则：wearable_breast_pump 不应包含 bra 相关文本
            if line == "wearable_breast_pump":
                if "bra" in text_lower or "bras" in text_lower:
                    continue
            score = 0
            for kw in rule["keywords"]:
                if _match_keyword(text_lower, kw):
                    score += 3
            for ctx in rule["context"]:
                if ctx in text_lower:
                    score += 2
            # 不同品线有不同的阈值
            threshold = 5
            if line == "wearable_breast_pump":
                threshold = 7  # 可穿戴吸奶器需要更高置信度
            elif line == "pregnancy_pillow":
                threshold = 4  # 孕妇枕上下文信号较弱，降低阈值
            elif line == "postpartum_recovery":
                threshold = 4  # 产后恢复词较分散，降低阈值
            if score >= threshold:
                matches.append(line)

    # ── Level 4: 品牌强关联推断 ──────────────────────────────────
    if not matches:
        for brand, line in BRAND_PRODUCT_LINE.items():
            if brand in text_lower:
                # 专属品牌：该品牌只做一种产品，直接命中
                if brand in EXCLUSIVE_BRANDS:
                    matches.append(line)
                    break
                # 非专属品牌：需要品类上下文验证
                if line in CONTEXT_RULES:
                    ctx_words = CONTEXT_RULES[line].get("context", []) + CONTEXT_RULES[line].get("keywords", [])
                    if any(c in text_lower for c in ctx_words):
                        matches.append(line)
                        break
                else:
                    matches.append(line)
                    break

    # ── Level 5: 标签辅助推断 ────────────────────────────────────
    if not matches and tags:
        tag_votes = Counter()
        for tag in tags:
            tag_en = tag.get("tag_en", "")
            if tag_en in TAG_PRODUCT_LINE_HINTS:
                tag_votes[TAG_PRODUCT_LINE_HINTS[tag_en]] += 3
        if tag_votes:
            best_line, best_score = tag_votes.most_common(1)[0]
            if best_score >= 3:
                return best_line

    if not matches:
        return "other"
    for priority_line in LINE_PRIORITY:
        if priority_line in matches:
            return priority_line
    return matches[0]


def extract_spu(text: str, product_line: str) -> str:
    """从文本中提取 SPU（产品型号/系列）"""
    text_lower = text.lower()

    # 品牌+型号模式
    spu_patterns = {
        "breast_pump": [
            (r'momcozy\s+s12\s*(?:pro|plus)?', 'momcozy_s12'),
            (r'momcozy\s+m[0-9]', 'momcozy_m_series'),
            (r'spectra\s+s[0-9]', 'spectra_s_series'),
            (r'spectra\s+synergy', 'spectra_synergy'),
            (r'medela\s+pump', 'medela_pump'),
            (r'medela\s+freestyle', 'medela_freestyle'),
            (r'elvie\s+(?:pump|stride)', 'elvie_pump'),
            (r'willow\s+3\.0', 'willow_3'),
            (r'willow\s+go', 'willow_go'),
            (r'lansinoh', 'lansinoh_pump'),
        ],
        "wearable_breast_pump": [
            (r'momcozy\s+s12\s*(?:pro|plus)?', 'momcozy_s12'),
            (r'elvie\s+(?:pump|stride)', 'elvie_pump'),
            (r'willow\s+3\.0', 'willow_3'),
            (r'willow\s+go', 'willow_go'),
        ],
        "pregnancy_pillow": [
            (r'phar[m]?e[cs]y', 'pharmacy_pillow'),
            (r'boppy', 'boppy_pillow'),
            (r'c\s*shaped', 'c_shaped_pillow'),
            (r'u\s*shaped', 'u_shaped_pillow'),
        ],
        "nursing_bra": [
            (r'sublime', 'sublime_bra'),
            (r'kindred\s+bravely', 'kindred_bravely'),
            (r'cake', 'cake_bra'),
        ],
        "bottle_warmer": [
            (r'dr\s*brown', 'dr_brown_warmer'),
            (r'philips\s+avent', 'avent_warmer'),
        ],
        "sterilizer": [
            (r'wabi\s+baby', 'wabi_baby_sterilizer'),
            (r'papablic', 'papablic_sterilizer'),
        ],
        "stroller": [
            (r'baby\s+jogger', 'baby_jogger_stroller'),
            (r'uppababy', 'uppababy_stroller'),
            (r'bugaboo', 'bugaboo_stroller'),
        ],
        "baby_monitor": [
            (r'nanit', 'nanit_monitor'),
            (r'owlet', 'owlet_monitor'),
            (r'miku', 'miku_monitor'),
        ],
    }

    patterns = spu_patterns.get(product_line, [])
    for pattern, spu_name in patterns:
        if re.search(pattern, text_lower):
            return spu_name

    # 如果没有匹配到具体型号，返回品线作为 SPU
    return product_line


def extract_brand(text: str) -> str:
    """提取品牌"""
    text_lower = text.lower()
    brand_map = {
        "momcozy": ["momcozy"],
        "elvie": ["elvie"],
        "spectra": ["spectra"],
        "medela": ["medela"],
        "tommee_tippee": ["tommee tippee", "tommeetippee"],
        "avent": ["philips avent", "avent"],
        "dr_brown": ["dr brown", "drbrown", "dr. brown"],
        "willow": ["willow"],
        "lansinoh": ["lansinoh"],
        "haakaa": ["haakaa", "haaka"],
        "fridababy": ["fridababy", "frida baby"],
        "boppy": ["boppy"],
        "dockatot": ["dockatot"],
        "nanit": ["nanit"],
        "owlet": ["owlet"],
        "kindred_bravely": ["kindred bravely"],
    }
    for brand, patterns in brand_map.items():
        for p in patterns:
            if p in text_lower:
                return brand
    return "other"


def load_and_enrich_records():
    """加载打标结果并添加品线、SPU、品牌信息"""
    print("=" * 70)
    print("加载并富集打标数据")
    print("=" * 70)

    records = []
    for src in ["amazon", "trustpilot", "reddit", "zendesk"]:
        src_dir = INPUT_DIR / src
        if not src_dir.exists():
            continue
        for batch_file in sorted(src_dir.glob("batch_*.jsonl")):
            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    text = r.get("text_preview", "")
                    tags = r.get("aipl_tags", [])
                    r["_product_line"] = classify_product_line(text, tags)
                    r["_spu"] = extract_spu(text, r["_product_line"])
                    r["_brand"] = extract_brand(text)
                    r["_source"] = src
                    records.append(r)

    print(f"  总记录: {len(records):,}")
    return records


def build_product_line_matrix(records: list[dict]) -> pd.DataFrame:
    """品线 × 标签 宽表"""
    print("\n" + "=" * 70)
    print("构建 品线 × 标签 宽表")
    print("=" * 70)

    line_stats = defaultdict(lambda: {
        "total_voc": 0,
        "tag_hits": Counter(),
        "promoters": 0,
        "detractors": 0,
        "passives": 0,
        "sentiment_sum": 0.0,
        "brands": Counter(),
    })

    all_tags = set()

    for r in records:
        line = r["_product_line"]
        line_stats[line]["total_voc"] += 1
        line_stats[line]["sentiment_sum"] += r.get("sentiment_polarity", 0)
        line_stats[line]["brands"][r["_brand"]] += 1

        nps = r.get("proxy_nps", "passive")
        if nps == "promoter":
            line_stats[line]["promoters"] += 1
        elif nps == "detractor":
            line_stats[line]["detractors"] += 1
        else:
            line_stats[line]["passives"] += 1

        for tag in r.get("aipl_tags", []):
            tag_en = tag["tag_en"]
            line_stats[line]["tag_hits"][tag_en] += 1
            all_tags.add(tag_en)

    all_tags_sorted = sorted(all_tags)
    rows = []

    for line, stats in sorted(line_stats.items(), key=lambda x: -x[1]["total_voc"]):
        total = stats["total_voc"]
        nps_val = (stats["promoters"] / total * 100) - (stats["detractors"] / total * 100) if total else 0
        top_brand = stats["brands"].most_common(1)[0][0] if stats["brands"] else "other"

        row = {
            "品线": line,
            "总VOC": total,
            "覆盖率": round(sum(1 for t in stats["tag_hits"].values() if t > 0) / len(all_tags) * 100, 1) if all_tags else 0,
            "Proxy_NPS": round(nps_val, 1),
            "推荐者": stats["promoters"],
            "贬损者": stats["detractors"],
            "平均情感": round(stats["sentiment_sum"] / total, 2) if total else 0,
            "Top品牌": top_brand,
        }

        for tag in all_tags_sorted:
            row[f"{tag}_count"] = stats["tag_hits"].get(tag, 0)
            row[f"{tag}_rate"] = round(stats["tag_hits"].get(tag, 0) / total * 100, 2) if total else 0

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  品线数: {len(df)}")
    print(f"  标签数: {len(all_tags_sorted)}")
    return df


def build_spu_matrix(records: list[dict]) -> pd.DataFrame:
    """SPU × 标签 宽表"""
    print("\n" + "=" * 70)
    print("构建 SPU × 标签 宽表")
    print("=" * 70)

    spu_stats = defaultdict(lambda: {
        "total_voc": 0,
        "tag_hits": Counter(),
        "promoters": 0,
        "detractors": 0,
        "product_line": "",
        "top_brand": "",
    })

    all_tags = set()

    for r in records:
        spu = r["_spu"]
        line = r["_product_line"]
        brand = r["_brand"]

        spu_stats[spu]["total_voc"] += 1
        spu_stats[spu]["product_line"] = line
        if not spu_stats[spu]["top_brand"]:
            spu_stats[spu]["top_brand"] = brand

        nps = r.get("proxy_nps", "passive")
        if nps == "promoter":
            spu_stats[spu]["promoters"] += 1
        elif nps == "detractor":
            spu_stats[spu]["detractors"] += 1

        for tag in r.get("aipl_tags", []):
            tag_en = tag["tag_en"]
            spu_stats[spu]["tag_hits"][tag_en] += 1
            all_tags.add(tag_en)

    # 只保留有一定样本量的 SPU（≥20条）
    min_samples = 20
    spu_stats = {k: v for k, v in spu_stats.items() if v["total_voc"] >= min_samples}

    all_tags_sorted = sorted(all_tags)
    rows = []

    for spu, stats in sorted(spu_stats.items(), key=lambda x: -x[1]["total_voc"]):
        total = stats["total_voc"]
        nps_val = (stats["promoters"] / total * 100) - (stats["detractors"] / total * 100) if total else 0

        row = {
            "SPU": spu,
            "品线": stats["product_line"],
            "品牌": stats["top_brand"],
            "总VOC": total,
            "Proxy_NPS": round(nps_val, 1),
            "推荐者": stats["promoters"],
            "贬损者": stats["detractors"],
        }

        for tag in all_tags_sorted:
            row[f"{tag}_count"] = stats["tag_hits"].get(tag, 0)
            row[f"{tag}_rate"] = round(stats["tag_hits"].get(tag, 0) / total * 100, 2) if total else 0

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  SPU数: {len(df)} (≥{min_samples}条)")
    print(f"  标签数: {len(all_tags_sorted)}")
    return df


def build_brand_product_line_matrix(records: list[dict]) -> pd.DataFrame:
    """品牌 × 品线 交叉矩阵"""
    print("\n" + "=" * 70)
    print("构建 品牌 × 品线 交叉矩阵")
    print("=" * 70)

    brand_line = defaultdict(lambda: defaultdict(int))
    brand_nps = defaultdict(lambda: {"promoters": 0, "detractors": 0})

    for r in records:
        brand = r["_brand"]
        line = r["_product_line"]
        brand_line[brand][line] += 1

        nps = r.get("proxy_nps", "passive")
        if nps == "promoter":
            brand_nps[brand]["promoters"] += 1
        elif nps == "detractor":
            brand_nps[brand]["detractors"] += 1

    # 获取所有品线
    all_lines = sorted(set(line for lines in brand_line.values() for line in lines))

    rows = []
    for brand in sorted(brand_line.keys(), key=lambda b: -sum(brand_line[b].values())):
        total = sum(brand_line[brand].values())
        nps_val = (brand_nps[brand]["promoters"] / total * 100) - (brand_nps[brand]["detractors"] / total * 100) if total else 0

        row = {
            "品牌": brand,
            "总VOC": total,
            "Proxy_NPS": round(nps_val, 1),
            "推荐者": brand_nps[brand]["promoters"],
            "贬损者": brand_nps[brand]["detractors"],
        }

        for line in all_lines:
            row[line] = brand_line[brand].get(line, 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  品牌数: {len(df)}")
    print(f"  品线数: {len(all_lines)}")
    return df


def main():
    print("=" * 70)
    print("品线 × SPU × 品牌 三维标签宽表生成")
    print("=" * 70)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载并富集数据
    records = load_and_enrich_records()

    # 1. 品线 × 标签 宽表
    line_df = build_product_line_matrix(records)
    line_path = EXPORT_DIR / "product_line_tag_matrix.csv"
    line_df.to_csv(line_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {line_path}")

    # 2. SPU × 标签 宽表
    spu_df = build_spu_matrix(records)
    spu_path = EXPORT_DIR / "spu_tag_matrix.csv"
    spu_df.to_csv(spu_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {spu_path}")

    # 3. 品牌 × 品线 交叉矩阵
    brand_line_df = build_brand_product_line_matrix(records)
    brand_line_path = EXPORT_DIR / "brand_product_line_matrix.csv"
    brand_line_df.to_csv(brand_line_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {brand_line_path}")

    print("\n" + "=" * 70)
    print(f"三维宽表生成完成: {EXPORT_DIR}")
    print("=" * 70)

    # 打印摘要
    print("\n" + "=" * 70)
    print("品线分布摘要")
    print("=" * 70)
    for _, row in line_df.head(10).iterrows():
        print(f"  {row['品线']:25s}: {row['总VOC']:6,} VOC | NPS={row['Proxy_NPS']:5.1f} | 覆盖率={row['覆盖率']:.1f}%")

    print("\n" + "=" * 70)
    print("Top 10 SPU")
    print("=" * 70)
    for _, row in spu_df.head(10).iterrows():
        print(f"  {row['SPU']:25s} ({row['品线']:15s}): {row['总VOC']:5,} VOC | NPS={row['Proxy_NPS']:.1f}")


if __name__ == "__main__":
    main()

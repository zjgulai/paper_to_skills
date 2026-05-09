"""通用情感/体验/属性标签打标器（Phase 3 P3）

为 zero-label VOC 补充通用体验标签：
- 情感/体验维度：易用性、质量感知、外观设计、耐用性、性能、性价比
- 场景维度：工作、夜间、旅行、推荐、礼品
- 产品属性维度：颜色、材质、无线、便携、静音

设计原则：
1. 只打 zero-label 记录（已有标签的不重复打）
2. 通用标签，不依赖品线
3. 带否定词检测和情感校准
"""

import json
import re
from pathlib import Path
from collections import Counter


# ── 通用标签定义 ───────────────────────────────────────────────────

GENERAL_TAGS = [
    {
        "tag_id": "TAG_GEN_E001",
        "tag_en": "ease_of_use",
        "tag_cn": "易用性",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "easy to use", "easy to", "so easy", "very easy", "super easy",
            "simple to", "straightforward", "user friendly", "user-friendly",
            "intuitive", "hassle free", "hassle-free", "no hassle",
            "convenient", "effortless", "not complicated",
        ],
        "negation_keywords": ["not easy", "not simple", "not intuitive", "complicated", "difficult to use", "hard to use", "confusing"],
    },
    {
        "tag_id": "TAG_GEN_E002",
        "tag_en": "comfort_experience",
        "tag_cn": "舒适体验",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "comfortable", "comfort", "cozy", "soft", "cushiony", "snug",
            "relaxing", "gentle on", "doesn't hurt", "doesnt hurt", "no pain",
            "comfy", "so comfy", "very comfortable", "extremely comfortable",
        ],
        "negation_keywords": ["uncomfortable", "not comfortable", "hurts", "painful", "rough", "scratchy", "itchy"],
    },
    {
        "tag_id": "TAG_GEN_E003",
        "tag_en": "quality_perception",
        "tag_cn": "质量感知",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "good quality", "great quality", "high quality", "well made", "well-made",
            "sturdy", "solid build", "premium", "excellent quality", "superior quality",
            "nicely made", "decent quality", "quality product",
            # German
            "sehr zufrieden", "sehr gut", "alles bestens", "alles super",
            "top qualität", "super qualität", "gute qualität", "gute qualitaet",
            # French
            "très satisfait", "très satisfaite", "super produit", "bonne qualité",
            "bonne qualite", "excellente qualité",
        ],
        "negation_keywords": ["poor quality", "bad quality", "cheaply made", "flimsy", "low quality", "terrible quality"],
    },
    {
        "tag_id": "TAG_GEN_E004",
        "tag_en": "appearance_design",
        "tag_cn": "外观设计",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "cute", "pretty", "beautiful", "stylish", "sleek", "elegant",
            "nice design", "great design", "love the design", "aesthetic",
            "looks nice", "look nice", "looks good", "look good", "looks great",
            "attractive", "modern design", "clean design",
        ],
        "negation_keywords": ["ugly", "cheap looking", "looks cheap", "dated design", "old fashioned"],
    },
    {
        "tag_id": "TAG_GEN_E005",
        "tag_en": "durability",
        "tag_cn": "耐用性",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "durable", "long lasting", "long-lasting", "holds up well", "lasted",
            "still works", "years later", "months later", "after a year", "after two years",
            "well built", "built to last", "robust", "reliable",
        ],
        "negation_keywords": ["broke", "broken", "fall apart", "fell apart", "worn out", "didnt last", "didn't last", "stopped working after"],
    },
    {
        "tag_id": "TAG_GEN_E006",
        "tag_en": "performance_satisfaction",
        "tag_cn": "性能满意",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "works great", "works well", "works perfectly", "working great",
            "effective", "efficient", "powerful", "strong", "does the job",
            "exactly what i needed", "exactly what we needed", "does what it says",
        ],
        "negation_keywords": ["doesn't work", "doesnt work", "not working", "stopped working", "barely works", "ineffective", "weak"],
    },
    {
        "tag_id": "TAG_GEN_E007",
        "tag_en": "price_value_positive",
        "tag_cn": "性价比正面",
        "aipl": "P1",
        "sentiment_base": "positive",
        "keywords": [
            "worth the money", "worth every penny", "great value", "good value",
            "affordable", "reasonable price", "great price", "good price",
            "cheap for the quality", "bang for buck", "great deal", "good deal",
        ],
        "negation_keywords": ["overpriced", "too expensive", "not worth", "waste of money", "rip off", "over priced"],
    },
    {
        "tag_id": "TAG_GEN_E008",
        "tag_en": "size_fit_positive",
        "tag_cn": "尺码合身",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "fits well", "fit well", "fits perfectly", "fit perfectly",
            "true to size", "accurate size", "right size", "perfect fit",
            "fits great", "fit great", "good fit", "snug fit", "comfortable fit",
        ],
        "negation_keywords": ["too big", "too small", "too tight", "too loose", "doesn't fit", "doesnt fit", "wrong size", "runs small", "runs large"],
    },
    {
        "tag_id": "TAG_GEN_S001",
        "tag_en": "work_scenario",
        "tag_cn": "工作场景",
        "aipl": "L1",
        "sentiment_base": "neutral",
        "keywords": [
            "at work", "while working", "working mom", "working mother",
            "office", "wfh", "work from home", "back to work", "return to work",
            "pumping at work", "at my desk", "during work",
        ],
        "negation_keywords": [],
    },
    {
        "tag_id": "TAG_GEN_S002",
        "tag_en": "night_use_scenario",
        "tag_cn": "夜间使用",
        "aipl": "L1",
        "sentiment_base": "neutral",
        "keywords": [
            "at night", "night time", "nighttime", "overnight", "while sleeping",
            "sleep", "sleeping", "bedtime", "middle of the night", "late night",
            "dark", "dim light", "night feed", "night feeding",
        ],
        "negation_keywords": [],
    },
    {
        "tag_id": "TAG_GEN_S003",
        "tag_en": "travel_scenario",
        "tag_cn": "旅行场景",
        "aipl": "L1",
        "sentiment_base": "neutral",
        "keywords": [
            "travel", "traveling", "traveling with", "on a trip", "trip",
            "vacation", "on vacation", "flying", "airport", "road trip",
            "away from home", "hotel", "packing", "in my bag", "car ride",
        ],
        "negation_keywords": [],
    },
    {
        "tag_id": "TAG_GEN_S004",
        "tag_en": "strong_recommendation",
        "tag_cn": "强烈推荐",
        "aipl": "L3",
        "sentiment_base": "positive",
        "keywords": [
            "highly recommend", "strongly recommend", "definitely recommend",
            "would recommend", "recommend to", "recommend this", "recommend it",
            "buy again", "repurchase", "will buy again", "purchased again",
            "told my friend", "told my sister", "told everyone",
            "love this", "love the", "love it", "i love",
            # German
            "gerne wieder", "immer wieder", "weiterempfehlen", "empfehle gerne",
            # French
            "je recommande", "vivement recommandé", "vivement recommande",
        ],
        "negation_keywords": ["would not recommend", "wouldn't recommend", "dont recommend", "don't recommend", "cannot recommend"],
    },
    {
        "tag_id": "TAG_GEN_A001",
        "tag_en": "color_mention",
        "tag_cn": "颜色提及",
        "aipl": "L1",
        "sentiment_base": "neutral",
        "keywords": [
            "black", "white", "pink", "blue", "grey", "gray", "beige", "nude",
            "red", "green", "purple", "yellow", "brown", "tan", "ivory", "cream",
            "navy", "teal", "mint", "rose", "mauve", "blush", "taupe",
        ],
        "negation_keywords": [],
    },
    {
        "tag_id": "TAG_GEN_A002",
        "tag_en": "material_mention",
        "tag_cn": "材质提及",
        "aipl": "L1",
        "sentiment_base": "neutral",
        "keywords": [
            "silicone", "cotton", "fabric", "cloth", "plastic", "metal",
            "stainless steel", "mesh", "foam", "memory foam", "rubber",
            "velcro", "elastic", "spandex", "nylon", "polyester", "bamboo",
        ],
        "negation_keywords": [],
    },
    {
        "tag_id": "TAG_GEN_A003",
        "tag_en": "wireless_handsfree",
        "tag_cn": "无线免提",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "wireless", "hands free", "hands-free", "bluetooth", "cordless",
            "no cord", "no wires", "without wires", "freedom to move",
        ],
        "negation_keywords": [],
    },
    {
        "tag_id": "TAG_GEN_A004",
        "tag_en": "portable_lightweight",
        "tag_cn": "便携轻量",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "portable", "lightweight", "light weight", "compact", "small enough",
            "fits in my bag", "fits in purse", "easy to carry", "convenient to carry",
        ],
        "negation_keywords": ["too heavy", "heavy", "bulky", "cumbersome"],
    },
    {
        "tag_id": "TAG_GEN_A005",
        "tag_en": "quiet_operation",
        "tag_cn": "静音运行",
        "aipl": "L1",
        "sentiment_base": "positive",
        "keywords": [
            "quiet", "silent", "barely hear", "hardly hear", "not loud",
            "so quiet", "very quiet", "super quiet", "whisper quiet",
        ],
        "negation_keywords": ["loud", "noisy", "too loud", "very loud", "annoying noise"],
    },
    {
        "tag_id": "TAG_GEN_D001",
        "tag_en": "delivery_satisfaction",
        "tag_cn": "配送满意",
        "aipl": "P2",
        "sentiment_base": "positive",
        "keywords": [
            # German
            "schnelle lieferung", "schneller versand", "schnell geliefert",
            "schnelle lieferung", "schneller versand", "schnell geliefert",
            "schnell und zuverlässig", "schnell und zuverlaessig",
            # French
            "livraison rapide", "livraison très rapide", "reçu rapidement",
            "recu rapidement", "livraison express", "livraison rapide et",
            # English
            "fast shipping", "quick delivery", "arrived quickly", "arrived fast",
            "delivered quickly", "shipped fast",
        ],
        "negation_keywords": [
            "slow delivery", "late delivery", "delayed shipping", "never arrived",
            # German
            "langsame lieferung", "verspätete lieferung", "nicht angekommen",
            "nie angekommen", "verspätung", "lieferung dauert zu lange",
            "lieferung langsam", "verspäteter versand",
            # French
            "livraison lente", "livraison retardée", "pas arrivé",
            "jamais arrivé", "retard de livraison", "délai trop long",
            "livraison trop lente", "expédition retardée",
        ],
    },
    {
        "tag_id": "TAG_GEN_C001",
        "tag_en": "service_satisfaction",
        "tag_cn": "服务满意",
        "aipl": "L3",
        "sentiment_base": "positive",
        "keywords": [
            # German
            "guter service", "super service", "freundlich", "hilfsbereit",
            "sehr freundlich", "ausgezeichneter service", "toller service",
            # French
            "bon service", "service client", "très professionnel", "tres professionnel",
            "excellent service", "service impeccable",
            # English
            "great service", "excellent customer service", "helpful staff",
            "friendly support", "quick response",
        ],
        "negation_keywords": [
            "terrible service", "awful service", "rude staff", "no response",
            # German
            "schlechter service", "unfreundlich", "keine antwort",
            "mieser service", "schlechte kundenbetreuung",
            "unfreundlicher service", "schlechter kundenservice",
            # French
            "mauvais service", "service horrible", "pas de réponse",
            "service client nul", "personnel impoli",
            "service client terrible", "aucune réponse",
        ],
    },
    {
        "tag_id": "TAG_GEN_P001",
        "tag_en": "general_positive",
        "tag_cn": "总体正面",
        "aipl": "L3",
        "sentiment_base": "positive",
        "keywords": [
            "thank you", "thanks for", "amazing", "awesome", "fantastic",
            "wonderful", "perfect", "brilliant", "outstanding", "super",
            "top", "great", "happy with", "very happy", "so happy",
        ],
        "negation_keywords": ["not amazing", "not awesome", "not fantastic", "not perfect"],
    },
    # ── 通用负面标签（Phase 4 新增）─────────────────────────────────
    {
        "tag_id": "TAG_GEN_N001",
        "tag_en": "difficult_to_use",
        "tag_cn": "使用困难",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "difficult to use", "hard to use", "complicated", "confusing",
            "not intuitive", "steep learning curve", "frustrating to use",
            "difficult to figure out", "hard to operate", "not user friendly",
            "hard to understand", "difficult to assemble", "hard to set up",
            # German
            "schwierig zu bedienen", "kompliziert", "nicht intuitiv",
            "schwer zu verstehen", "nicht benutzerfreundlich",
            # French
            "difficile à utiliser", "compliqué", "pas intuitif",
            "difficile à comprendre", "pas convivial",
        ],
        "negation_keywords": [
            "not difficult", "not hard", "easy to use", "simple to use",
            "not complicated", "not confusing",
        ],
        "counterpart": "TAG_GEN_E001",  # 与 ease_of_use 互斥
    },
    {
        "tag_id": "TAG_GEN_N002",
        "tag_en": "uncomfortable",
        "tag_cn": "不舒适",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "uncomfortable", "painful", "hurts", "rough", "scratchy",
            "digs in", "pinches", "too tight", "squeezes", "chafes",
            "irritating", "rubbing", "blisters", "sore", "aching",
            "not comfortable", "not comfy", "no comfort",
            # German
            "unbequem", "schmerzhaft", "zu eng", "drückt", "reizt",
            "nicht bequem", "schmerzend",
            # French
            "inconfortable", "douloureux", "trop serré", "frotte",
            "irritant", "pas confortable", "fait mal",
        ],
        "negation_keywords": [
            "not uncomfortable", "comfortable", "comfy", "soft",
        ],
        "counterpart": "TAG_GEN_E002",
    },
    {
        "tag_id": "TAG_GEN_N003",
        "tag_en": "poor_quality",
        "tag_cn": "质量差",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "poor quality", "bad quality", "cheap", "flimsy", "cheaply made",
            "poor construction", "falls apart", "not well made", "shoddy",
            "crappy", "junky", "garbage", "trash", "piece of junk",
            "low quality", "inferior quality", "subpar",
            # German
            "schlechte qualität", "billig", "mindere qualität",
            "minderwertig", "schlecht verarbeitet", "billig gemacht",
            "plastikmüll", "schlecht",
            # French
            "mauvaise qualité", "pas cher", "qualité médiocre",
            "fait cheap", "mauvaise fabrication", "bas de gamme",
            "qualité inférieure", "très mauvaise qualité",
        ],
        "negation_keywords": [
            "good quality", "great quality", "high quality", "well made",
        ],
        "counterpart": "TAG_GEN_E003",
    },
    {
        "tag_id": "TAG_GEN_N004",
        "tag_en": "poor_design",
        "tag_cn": "设计差",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "poor design", "bad design", "ugly", "cheap looking", "dated",
            "awkward design", "poorly designed", "doesn't look good",
            "weird design", "strange design", "clunky", "bulky design",
            "not well designed", "design flaw", "design issue",
            # German
            "schlechtes design", "hässlich", "billig aussehend",
            "nicht gut designt", "designfehler",
            # French
            "mauvais design", "moche", "laid", "pas beau",
            "design raté", "défaut de design",
        ],
        "negation_keywords": [
            "nice design", "great design", "love the design", "beautiful",
        ],
        "counterpart": "TAG_GEN_E004",
    },
    {
        "tag_id": "TAG_GEN_N005",
        "tag_en": "not_durable",
        "tag_cn": "不耐用",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "broke quickly", "fell apart", "didn't last", "not durable",
            "poor durability", "wore out fast", "lasted only", "cheap material",
            "broke after", "stopped working after", "died after", "failed after",
            "falling apart", "coming apart", "tearing", "ripping",
            "only lasted", "barely lasted", "lasted less than",
            # German
            "schnell kaputt", "hält nicht lange", "nicht haltbar",
            "fiel auseinander", "ging kaputt", "nicht langlebig",
            # French
            "s'est cassé vite", "pas durable", "ne dure pas",
            "tombé en morceaux", "pas solide", "pas résistant",
        ],
        "negation_keywords": [
            "durable", "long lasting", "holds up well", "still works",
        ],
        "counterpart": "TAG_GEN_E005",
    },
    {
        "tag_id": "TAG_GEN_N006",
        "tag_en": "poor_performance",
        "tag_cn": "性能差",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "poor performance", "weak", "ineffective", "doesn't work well",
            "not powerful enough", "underperforms", "disappointing performance",
            "not effective", "barely works", "hardly works", "doesn't do much",
            "not strong enough", "too weak", "lacks power", "insufficient power",
            "not working properly", "not functioning", "malfunctioning",
            # German
            "schlechte leistung", "schwach", "nicht effektiv",
            "funktioniert nicht gut", "unzureichende leistung",
            # French
            "mauvaises performances", "faible", "pas efficace",
            "ne fonctionne pas bien", "performances décevantes",
        ],
        "negation_keywords": [
            "works great", "works well", "effective", "powerful",
        ],
        "counterpart": "TAG_GEN_E006",
    },
    {
        "tag_id": "TAG_GEN_N007",
        "tag_en": "poor_value",
        "tag_cn": "性价比差",
        "aipl": "P1",
        "sentiment_base": "negative",
        "keywords": [
            "overpriced", "not worth", "waste of money", "too expensive",
            "poor value", "not worth the price", "overpriced for what it is",
            "rip off", "rip-off", "money down the drain",
            "expensive for", "too much money", "cost too much",
            "not worth it", "not worth the money", "not worth buying",
            "could be cheaper", "should be cheaper", "way too expensive",
            # German
            "zu teuer", "nicht sein geld wert", "überteuert",
            "schlechtes preis-leistungs-verhältnis", "geldverschwendung",
            # French
            "trop cher", "pas worth", "mauvais rapport qualité prix",
            "trop coûteux", "pas rentable", "gaspillage d'argent",
        ],
        "negation_keywords": [
            "worth the money", "worth every penny", "great value", "good value",
        ],
        "counterpart": "TAG_GEN_E007",
    },
    {
        "tag_id": "TAG_GEN_N008",
        "tag_en": "size_fit_negative",
        "tag_cn": "尺码不合",
        "aipl": "L1",
        "sentiment_base": "negative",
        "keywords": [
            "too tight", "too loose", "doesn't fit", "runs small", "runs large",
            "wrong size", "size off", "sizing is wrong", "fits poorly",
            "doesn't fit right", "fit is wrong", "fit is off",
            "too big", "too small", "way too", "unfortunately too",
            "sizing is off", "not true to size", "inaccurate sizing",
            "ordered my size but", "size chart is wrong",
            # German
            "zu klein", "zu groß", "passt nicht", "fällt klein aus",
            "fällt groß aus", "falsche größe", "größe passt nicht",
            # French
            "trop petit", "trop grand", "ne correspond pas", "taille incorrecte",
            "taille ne correspond pas", "fausse taille",
        ],
        "negation_keywords": [
            "fits well", "fits perfectly", "true to size", "right size",
        ],
        "counterpart": "TAG_GEN_E008",
    },
]


# ── 互斥对映射 ─────────────────────────────────────────────────────

POS_TO_NEG: dict[str, str] = {
    "TAG_GEN_E001": "TAG_GEN_N001",  # ease_of_use vs difficult_to_use
    "TAG_GEN_E002": "TAG_GEN_N002",  # comfort_experience vs uncomfortable
    "TAG_GEN_E003": "TAG_GEN_N003",  # quality_perception vs poor_quality
    "TAG_GEN_E004": "TAG_GEN_N004",  # appearance_design vs poor_design
    "TAG_GEN_E005": "TAG_GEN_N005",  # durability vs not_durable
    "TAG_GEN_E006": "TAG_GEN_N006",  # performance_satisfaction vs poor_performance
    "TAG_GEN_E007": "TAG_GEN_N007",  # price_value_positive vs poor_value
    "TAG_GEN_E008": "TAG_GEN_N008",  # size_fit_positive vs size_fit_negative
}


# ── 否定词检测 ─────────────────────────────────────────────────────

NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing",
    "n't", "dont", "doesnt", "didnt", "wouldnt", "couldnt",
    "shouldnt", "wont", "cant", "isnt", "arent", "wasnt",
    "werent", "hasnt", "havent", "hadnt",
}


def detect_negation(text_lower: str, keyword: str) -> bool:
    """检测关键词前20字符内是否有否定词"""
    idx = text_lower.find(keyword)
    if idx < 0:
        return False
    prefix = text_lower[max(0, idx - 25):idx]
    return any(neg in prefix for neg in NEGATION_WORDS)


def detect_negation_in_phrases(text_lower: str, neg_phrases: list[str]) -> bool:
    """检测文本是否包含否定短语"""
    return any(phrase in text_lower for phrase in neg_phrases)


def resolve_exclusive_pairs(labels: list[dict], text_lower: str) -> list[dict]:
    """解决互斥标签对冲突

    当正面对应负面标签同时命中时，根据否定词检测结果决定保留哪个。
    规则：
    1. 若文本含否定词 → 保留负面标签（否定语境）
    2. 若文本无否定词 → 保留正面标签（默认正面）
    3. 若负面标签自身关键词含否定词（如"not comfortable"）→ 保留负面标签
    """
    tag_ids = {lbl["tag_id"] for lbl in labels}
    to_remove: set[str] = set()

    for pos_tag, neg_tag in POS_TO_NEG.items():
        if pos_tag not in tag_ids or neg_tag not in tag_ids:
            continue

        # 检查文本中是否有全局否定词
        has_global_negation = any(neg in text_lower for neg in NEGATION_WORDS)

        # 检查负面标签是否由自身否定短语触发（如"not comfortable"）
        neg_def = next((t for t in GENERAL_TAGS if t["tag_id"] == neg_tag), None)
        neg_triggered_by_phrase = False
        if neg_def and neg_def.get("negation_keywords"):
            # 如果负面标签的 negation_keywords 在文本中命中
            # 说明是通过否定短语触发的，应保留负面标签
            for np in neg_def["negation_keywords"]:
                if np in text_lower:
                    neg_triggered_by_phrase = True
                    break

        # 检查正面标签是否由自身否定短语触发
        pos_def = next((t for t in GENERAL_TAGS if t["tag_id"] == pos_tag), None)
        pos_triggered_by_phrase = False
        if pos_def and pos_def.get("negation_keywords"):
            for np in pos_def["negation_keywords"]:
                if np in text_lower:
                    pos_triggered_by_phrase = True
                    break

        if neg_triggered_by_phrase:
            # 负面标签由否定短语触发，保留负面
            to_remove.add(pos_tag)
        elif pos_triggered_by_phrase:
            # 正面标签由否定短语触发，保留正面
            to_remove.add(neg_tag)
        elif has_global_negation:
            # 有全局否定词 → 保留负面（否定语境）
            to_remove.add(pos_tag)
        else:
            # 无否定词 → 保留正面（默认）
            to_remove.add(neg_tag)

    return [lbl for lbl in labels if lbl["tag_id"] not in to_remove]


# Pre-compile regex patterns for single-word keywords per tag
_WORD_BOUNDARY_CACHE = {}


def _keyword_in_text(text_lower: str, kw: str) -> bool:
    """关键词匹配：多词用子串，单词用整词边界"""
    if " " in kw:
        return kw in text_lower
    # Single word: use word boundary to avoid "red" matching "ordered"
    cache_key = kw
    if cache_key not in _WORD_BOUNDARY_CACHE:
        _WORD_BOUNDARY_CACHE[cache_key] = re.compile(r'\b' + re.escape(kw) + r'\b')
    return _WORD_BOUNDARY_CACHE[cache_key].search(text_lower) is not None


def apply_general_tag(text: str, tag_def: dict) -> tuple[bool, float, str]:
    """对单条文本应用单个通用标签

    Returns: (matched, confidence, sentiment_calibrated)
    """
    text_lower = text.lower()
    keywords = tag_def["keywords"]
    neg_keywords = tag_def.get("negation_keywords", [])
    sentiment_base = tag_def["sentiment_base"]

    # 1. 匹配关键词（单词用整词边界，避免 "red" 匹配 "ordered"）
    matched_kw = None
    for kw in keywords:
        if _keyword_in_text(text_lower, kw):
            matched_kw = kw
            break

    if not matched_kw:
        return False, 0.0, sentiment_base

    # 2. 否定短语检测（更强否定信号）
    if neg_keywords and detect_negation_in_phrases(text_lower, neg_keywords):
        # 检查否定短语是否紧邻匹配词
        # 简化：如果同时存在匹配词和否定短语，视为否定
        # 但排除 "not easy" 匹配 "easy" 的情况——这种情况由下面处理
        for neg_kw in neg_keywords:
            if neg_kw in text_lower:
                # 如果否定短语包含匹配词本身（如 "not easy" 包含 "easy"），
                # 且匹配词就是从这个否定短语来的，则否定
                if matched_kw in neg_kw:
                    return False, 0.0, sentiment_base
        # 否定短语存在但不包含匹配词——可能是对别的东西的否定
        # 继续检查局部否定

    # 3. 局部否定词检测（匹配词前25字符）
    if detect_negation(text_lower, matched_kw):
        # 反转为负面情感
        if sentiment_base == "positive":
            return True, 0.55, "negative"
        elif sentiment_base == "negative":
            return True, 0.55, "positive"
        else:
            return True, 0.5, "negative"

    # 4. 置信度计算
    # 长关键词 = 更高置信度
    kw_len = len(matched_kw.split())
    confidence = min(0.5 + kw_len * 0.05, 0.75)
    # 情感强烈的词额外加分
    if any(w in matched_kw for w in ["love", "perfect", "great", "amazing", "highly"]):
        confidence = min(confidence + 0.05, 0.8)

    return True, round(confidence, 2), sentiment_base


def label_record(text: str, existing_labels: list[dict]) -> list[dict]:
    """对单条记录应用所有通用标签"""
    new_labels = []
    for tag_def in GENERAL_TAGS:
        matched, conf, sentiment = apply_general_tag(text, tag_def)
        if matched:
            new_labels.append({
                "tag_id": tag_def["tag_id"],
                "tag_en": tag_def["tag_en"],
                "tag_cn": tag_def["tag_cn"],
                "aipl_node": tag_def["aipl"],
                "sentiment_preset": tag_def["sentiment_base"],
                "sentiment_calibrated": 1.0 if sentiment == "positive" else (-1.0 if sentiment == "negative" else 0.0),
                "confidence": conf,
                "source": "general_tag",
            })
    # ── 否定翻转：正面标签被否定 → 替换为对应负面标签 ──
    to_remove: set[str] = set()
    to_add: list[dict] = []
    existing_ids = {lbl["tag_id"] for lbl in new_labels}

    for lbl in new_labels:
        tag_id = lbl["tag_id"]
        # 正面标签被否定（sentiment_calibrated < 0）且存在对应负面标签
        if tag_id in POS_TO_NEG and lbl["sentiment_calibrated"] < 0:
            neg_tag_id = POS_TO_NEG[tag_id]
            to_remove.add(tag_id)
            # 若对应负面标签尚未独立命中，则添加之
            if neg_tag_id not in existing_ids:
                neg_def = next(t for t in GENERAL_TAGS if t["tag_id"] == neg_tag_id)
                to_add.append({
                    "tag_id": neg_def["tag_id"],
                    "tag_en": neg_def["tag_en"],
                    "tag_cn": neg_def["tag_cn"],
                    "aipl_node": neg_def["aipl"],
                    "sentiment_preset": neg_def["sentiment_base"],
                    "sentiment_calibrated": -1.0,
                    "confidence": lbl["confidence"],
                    "source": "general_tag",
                })

    new_labels = [lbl for lbl in new_labels if lbl["tag_id"] not in to_remove] + to_add

    # 互斥冲突解决
    text_lower = text.lower()
    new_labels = resolve_exclusive_pairs(new_labels, text_lower)

    return new_labels


def _test():
    """通用标签器自证测试（Phase 4 扩展版）"""
    print("=" * 70)
    print("通用标签器自证测试")
    print("=" * 70)

    test_cases = [
        # ── 正面标签 ──
        ("This pump is so easy to use", ["TAG_GEN_E001"], "正面-易用性"),
        ("Very comfortable to wear", ["TAG_GEN_E002"], "正面-舒适"),
        ("Good quality product", ["TAG_GEN_E003"], "正面-质量"),
        ("Beautiful design", ["TAG_GEN_E004"], "正面-外观"),
        ("Very durable and reliable", ["TAG_GEN_E005"], "正面-耐用"),
        ("Works great and powerful", ["TAG_GEN_E006", "TAG_GEN_P001"], "正面-性能+总体正面"),
        ("Worth every penny", ["TAG_GEN_E007"], "正面-性价比"),
        ("Fits perfectly and true to size", ["TAG_GEN_E008"], "正面-尺码"),

        # ── 负面标签 ──
        ("This pump is difficult to use and confusing", ["TAG_GEN_N001"], "负面-使用困难"),
        ("Very uncomfortable and painful", ["TAG_GEN_N002"], "负面-不舒适"),
        ("Poor quality and cheaply made", ["TAG_GEN_N003"], "负面-质量差"),
        ("Ugly design and cheap looking", ["TAG_GEN_N003", "TAG_GEN_N004"], "负面-设计差+质量差"),
        ("Broke after two weeks", ["TAG_GEN_N005"], "负面-不耐用"),
        ("Poor performance and weak", ["TAG_GEN_N006"], "负面-性能差"),
        ("Overpriced and not worth it", ["TAG_GEN_N007"], "负面-性价比差"),
        ("Too tight and wrong size", ["TAG_GEN_N002", "TAG_GEN_N008"], "负面-尺码不合+不舒适"),

        # ── 否定词反转 ──
        ("not easy to use at all", ["TAG_GEN_N001"], "否定反转-易用性"),
        ("not comfortable at all", ["TAG_GEN_N002"], "否定反转-舒适"),
        ("not good quality", ["TAG_GEN_N003"], "否定反转-质量"),
        ("doesn't fit well, too small", ["TAG_GEN_N008"], "否定反转-尺码"),

        # ── 互斥处理 ──
        ("easy to use but poor quality", ["TAG_GEN_E001", "TAG_GEN_N003"], "互斥-不同维度"),

        # ── 德/法多语言 ──
        ("schwierig zu bedienen", ["TAG_GEN_N001"], "德语-使用困难"),
        ("difficile à utiliser", ["TAG_GEN_N001"], "法语-使用困难"),
        ("schlechte qualität", ["TAG_GEN_N003"], "德语-质量差"),
        ("mauvaise qualité", ["TAG_GEN_N003"], "法语-质量差"),
        ("zu teuer", ["TAG_GEN_N007"], "德语-性价比差"),
        ("trop cher", ["TAG_GEN_N007"], "法语-性价比差"),
        ("pas confortable", ["TAG_GEN_N002"], "法语-不舒适"),
        ("unbequem", ["TAG_GEN_N002"], "德语-不舒适"),

        # ── 德/法配送负面 ──
        ("schnelle lieferung", ["TAG_GEN_D001"], "德语-配送正面"),
        ("livraison rapide", ["TAG_GEN_D001"], "法语-配送正面"),
    ]

    passed = 0
    failed = 0

    for text, expected_ids, desc in test_cases:
        labels = label_record(text, [])
        actual_ids = sorted([l["tag_id"] for l in labels])
        expected = sorted(expected_ids)

        status = "PASS" if actual_ids == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {desc}")
        if status == "FAIL":
            print(f"    文本: '{text}'")
            print(f"    期望: {expected}")
            print(f"    实际: {actual_ids}")
            for l in labels:
                print(f"      -> {l['tag_id']} ({l['tag_en']}): conf={l['confidence']}")

    print(f"\n测试结果: {passed}/{passed + failed} 通过 ({passed / (passed + failed) * 100:.1f}%)")

    # 标签覆盖审计
    pos_tags = [t for t in GENERAL_TAGS if t["sentiment_base"] == "positive" and t["tag_id"].startswith("TAG_GEN")]
    neg_tags = [t for t in GENERAL_TAGS if t["sentiment_base"] == "negative" and t["tag_id"].startswith("TAG_GEN")]
    neutral_tags = [t for t in GENERAL_TAGS if t["sentiment_base"] == "neutral" and t["tag_id"].startswith("TAG_GEN")]

    print(f"\n标签覆盖审计:")
    print(f"  正面标签: {len(pos_tags)} 个")
    print(f"  负面标签: {len(neg_tags)} 个")
    print(f"  中性标签: {len(neutral_tags)} 个")
    print(f"  总计: {len(GENERAL_TAGS)} 个")
    print(f"  互斥对: {len(POS_TO_NEG)} 对")

    # 多语言覆盖审计
    has_multilingual = 0
    for tag in GENERAL_TAGS:
        for kw in tag["keywords"]:
            if any(c in kw for c in "äöüßéèàù"):
                has_multilingual += 1
                break
    print(f"  含多语言关键词标签: {has_multilingual}/{len(GENERAL_TAGS)}")

    print("=" * 70)
    return passed, failed


def main():
    print("=" * 70)
    print("Phase 3 P3: 通用情感/体验/属性标签部署")
    print("=" * 70)

    input_path = Path("04-输出结果/unified_labeling/phase3_p2_labeled.jsonl")
    output_path = Path("04-输出结果/unified_labeling/phase3_p3_labeled.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计
    total = 0
    zero_before = 0
    newly_tagged = 0
    tag_counter = Counter()
    source_breakdown = Counter()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            rec = json.loads(line)
            total += 1

            existing_labels = rec.get("labels", [])
            is_zero = len(existing_labels) == 0

            if is_zero:
                zero_before += 1
                text = rec.get("text", "")
                new_labels = label_record(text, existing_labels)

                if new_labels:
                    newly_tagged += 1
                    rec["labels"] = new_labels
                    rec["n_tags"] = len(new_labels)
                    for lbl in new_labels:
                        tag_counter[lbl["tag_id"]] += 1
                    source_breakdown[rec.get("data_source", "unknown")] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total % 50000 == 0:
                print(f"  已处理 {total:,} 条...")

    # 报告
    coverage_before = (total - zero_before) / total * 100
    zero_after = zero_before - newly_tagged
    coverage_after = (total - zero_after) / total * 100

    print(f"\n--- 执行结果 ---")
    print(f"  总记录: {total:,}")
    print(f"  原零标签: {zero_before:,}")
    print(f"  新打标: {newly_tagged:,} ({newly_tagged/zero_before*100:.1f}%)")
    print(f"  覆盖率: {coverage_before:.2f}% → {coverage_after:.2f}% (+{coverage_after-coverage_before:.2f}%)")

    print(f"\n--- 按数据源新打标 ---")
    for src, cnt in source_breakdown.most_common():
        print(f"  {src}: {cnt:,}")

    print(f"\n--- 按标签分布 ---")
    for tag_id, cnt in tag_counter.most_common():
        tag_def = next(t for t in GENERAL_TAGS if t["tag_id"] == tag_id)
        print(f"  {tag_id} ({tag_def['tag_en']}): {cnt:,}")

    # 审计
    audit = {
        "phase": "3_P3",
        "general_tags": len(GENERAL_TAGS),
        "total_records": total,
        "zero_before": zero_before,
        "newly_tagged": newly_tagged,
        "zero_tag_rate_before": round(zero_before / total * 100, 2),
        "zero_tag_rate_after": round(zero_after / total * 100, 2),
        "coverage_before": round(coverage_before, 2),
        "coverage_after": round(coverage_after, 2),
        "coverage_improvement": round(coverage_after - coverage_before, 2),
        "tag_breakdown": dict(tag_counter),
        "source_breakdown": dict(source_breakdown),
        "output_path": str(output_path),
    }
    audit_path = Path("04-输出结果/04-审计数据/phase3_p3_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 3 P3 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

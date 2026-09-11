"""零标签文本高频实词挖掘

从所有零标签 VOC 中提取有意义的实词，分析主题分布，
为标签关键词扩充提供数据输入。
"""

import json
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest")

# 扩展停用词（覆盖更全面的常见无意义词）
STOP_WORDS = {
    # 基础停用词
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "and", "but", "or", "yet", "so", "if", "because",
    "although", "though", "while", "where", "when", "that", "which", "who",
    "whom", "whose", "what", "this", "these", "those",
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "one", "ones", "oneself",
    "am", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "one", "ones", "oneself", "i", "me", "my", "mine", "myself",
    # 情感停用词（这些词太泛，不表达具体主题）
    "great", "good", "nice", "awesome", "amazing", "fantastic", "wonderful",
    "excellent", "perfect", "best", "love", "loved", "loving", "like", "liked",
    "enjoy", "enjoyed", "recommend", "recommended", "happy", "pleased",
    "satisfied", "glad", "thank", "thanks", "appreciate", "bad", "terrible",
    "horrible", "awful", "worst", "hate", "hated", "dislike", "disappointed",
    "disappointing", "unhappy", "upset", "angry", "frustrated", "annoyed",
    "very", "really", "quite", "pretty", "fairly", "rather", "somewhat",
    "extremely", "incredibly", "absolutely", "totally", "completely",
    "definitely", "certainly", "probably", "maybe", "perhaps", "almost",
    "just", "only", "even", "also", "too", "enough", "well", "so", "still",
    "already", "yet", "always", "never", "often", "sometimes", "usually",
    "really", "truly", "actually", "literally", "honestly", "seriously",
    "definitely", "surely", "certainly", "obviously", "clearly",
    # 通用代词/限定词
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "than", "too",
    "very", "just", "now", "then", "here", "there", "when", "where", "why",
    "how", "what", "which", "who", "whom", "whose", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "ought",
    # 人称相关
    "mom", "mother", "mum", "mummy", "mama", "mommy", "dad", "father",
    "parent", "parents", "baby", "child", "children", "kid", "kids",
    "son", "daughter", "family", "husband", "wife", "partner",
    # 时间/频率
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "time", "times", "hour", "hours", "minute", "minutes", "second", "seconds",
    "today", "yesterday", "tomorrow", "morning", "afternoon", "evening",
    "night", "daily", "weekly", "monthly", "yearly", "every", "once", "twice",
    "first", "second", "third", "last", "next", "previous", "ago", "since",
    # 购物通用词
    "amazon", "order", "ordered", "purchase", "purchased", "buy", "bought",
    "shipping", "shipped", "delivery", "delivered", "arrived", "package",
    "box", "item", "product", "products", "thing", "things", "stuff",
    "company", "brand", "seller", "store", "website", "online",
    "review", "reviews", "star", "stars", "rating", "ratings",
    "money", "dollar", "dollars", "price", "cost", "paid", "pay",
    "return", "returned", "exchange", "refund", "warranty", "guarantee",
    # 其他泛词
    "way", "ways", "part", "parts", "piece", "pieces", "lot", "lots",
    "bit", "little", "much", "many", "few", "several", "various", "different",
    "same", "other", "another", "every", "each", "whole", "entire", "full",
    "half", "quarter", "double", "single", "multiple", "various",
    "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone", "noone", "somebody", "anybody", "everybody", "nobody",
    "place", "places", "area", "areas", "space", "room", "spot", "location",
    "point", "points", "line", "lines", "side", "sides", "end", "ends",
    "top", "bottom", "front", "back", "left", "right", "center", "middle",
    "inside", "outside", "within", "without", "above", "below", "over",
    "under", "between", "among", "across", "through", "into", "onto",
    "upon", "within", "throughout", "along", "around", "behind", "beside",
    "beyond", "inside", "outside", "underneath", "upon", "versus", "vs",
    "etc", "ie", "eg", "ok", "okay", "yes", "no", "yeah", "nah", "yep",
    "nope", "uh", "um", "oh", "ah", "wow", "oops", "hey", "hi", "hello",
    "bye", "goodbye", "please", "sorry", "excuse", "pardon", "welcome",
    "congratulations", "thanks", "thankyou", "cheers", "byebye",
    "im", "ive", "id", "ill", "youre", "youve", "youd", "youll",
    "hes", "hes", "shes", "shes", "its", "its", "were", "weve", "wed",
    "well", "theyre", "theyve", "theyd", "theyll", "isnt", "arent",
    "wasnt", "werent", "havent", "hasnt", "hadnt", "dont", "doesnt",
    "didnt", "wont", "wouldnt", "couldnt", "shouldnt", "mightnt",
    "mustnt", "cant", "cannot", "shant", "neednt", "darent", "oughtnt",
    "let", "lets", "let's", "got", "get", "gets", "getting", "gotten",
    "come", "comes", "coming", "came", "go", "goes", "going", "went",
    "gone", "give", "gives", "giving", "gave", "given", "take", "takes",
    "taking", "took", "taken", "make", "makes", "making", "made",
    "put", "puts", "putting", "set", "sets", "setting", "keep", "keeps",
    "keeping", "kept", "leave", "leaves", "leaving", "left",
    # 数字相关
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "twenty", "thirty", "hundred", "thousand",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "1st", "2nd", "3rd", "4th", "5th",
    # 格式词
    "etc", "ie", "eg", "vs", "aka", "btw", "fyi", "faq", "diy",
    "dont", "doesnt", "didnt", "wont", "wouldnt", "couldnt", "shouldnt",
    "cant", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "let's", "lets", "that's", "thats", "there's", "theres",
    "here's", "heres", "who's", "whos", "what's", "whats", "where's",
    "wheres", "when's", "whens", "why's", "whys", "how's", "hows",
    "it's", "its", "he's", "hes", "she's", "shes", "we're", "were",
    "they're", "theyre", "i'm", "im", "i've", "ive", "i'd", "id",
    "i'll", "ill", "you're", "youre", "you've", "youve", "you'd", "youd",
    "you'll", "youll", "we've", "weve", "we'd", "wed", "we'll", "well",
    "they've", "theyve", "they'd", "theyd", "they'll", "theyll",
}

# 产品相关的核心实词白名单（这些词即使频率高也应该保留分析）
PRODUCT_WORDS = {
    "pump", "pumping", "pumped", "breast pump", "breastpump",
    "pillow", "pregnancy pillow", "body pillow", "wedge pillow",
    "bra", "nursing bra", "breastfeeding bra", "pumping bra",
    "warmer", "bottle warmer", "wipe warmer",
    "sterilizer", "steriliser", "uv sterilizer", "steam sterilizer",
    "bottle", "bottles", "nipple", "nipples", "teat", "teats",
    "flange", "flanges", "shield", "shields",
    "tubing", "tube", "tubes", "hose", "hoses",
    "valve", "valves", "membrane", "membranes", "diaphragm", "diaphragms",
    "motor", "battery", "charging", "charger", "cord", "cable", "usb",
    "suction", "suction power", "suction strength", "vacuum", "vacuum strength",
    "mode", "modes", "setting", "settings", "level", "levels", "speed", "speeds",
    "cycle", "cycles", "expression", "expression mode", "massage", "massage mode",
    "noise", "noisy", "quiet", "silent", "loud", "sound", "volume",
    "comfort", "comfortable", "uncomfortable", "comfortably", "discomfort",
    "pain", "painful", "painless", "sore", "soreness", "hurt", "hurts", "hurting",
    "leak", "leaking", "leaked", "leaks", "spill", "spills", "spilling", "spilled",
    "clog", "clogged", "clogging", "clogs", "duct", "ducts", "mastitis",
    "milk", "breastmilk", "breast milk", "supply", "oversupply", "low supply",
    "letdown", "let down", "flow", "drip", "dripping", "drips",
    "portable", "portability", "compact", "small", "lightweight", "light", "heavy",
    "travel", "traveling", "travelled", "trip", "car", "flight", "airport",
    "charge", "charged", "charging", "battery life", "battery powered",
    "clean", "cleaning", "cleaned", "wash", "washing", "washed", "wipe", "wiping",
    "sterilize", "sterilizing", "sterilized", "sanitize", "sanitizing", "sanitized",
    "dishwasher", "dishwasher safe", "microwave", "boil", "boiling",
    "size", "sizes", "sizing", "fit", "fits", "fitting", "fitted", "tight", "loose",
    "small", "large", "medium", "xl", "xxl", "xs", "adjustable", "adjust",
    "band", "bands", "strap", "straps", "clip", "clips", "clasp", "clasps",
    "fabric", "material", "cotton", "silicone", "plastic", "rubber",
    "soft", "hard", "smooth", "rough", "thick", "thin", "stretchy", "stiff",
    "instruction", "instructions", "manual", "guide", "direction", "directions",
    "assembly", "assemble", "assembling", "setup", "set up", "install", "installation",
    "warranty", "guarantee", "return", "returning", "returned", "refund", "exchange",
    "customer service", "support", "contact", "email", "call", "phone", "chat",
    "shipping", "delivery", "delivered", "arrived", "arrive", "package", "box",
    "price", "cost", "expensive", "cheap", "affordable", "value", "worth", "money",
    "discount", "sale", "coupon", "deal", "promo", "promotion", "code",
    "gift", "gifted", "registry", "baby shower", "present",
    "recommend", "recommended", "recommending", "suggest", "suggested",
    "buy", "bought", "purchase", "purchased", "ordering", "ordered", "order",
    "use", "using", "used", "wear", "wearing", "wore", "work", "works", "working",
    "worked", "try", "tried", "trying", "attempt", "attempted",
    "help", "helps", "helping", "helped", "fix", "fixes", "fixed", "fixing",
    "issue", "issues", "problem", "problems", "defect", "defective", "faulty",
    "broken", "broke", "break", "breaking", "crack", "cracked", "cracking",
    "damage", "damaged", "dent", "dented", "scratch", "scratched",
    "stain", "stained", "discolor", "discolored", "fading", "faded", "fade",
    "smell", "smells", "smelling", "odor", "scent", "fragrance", "stink", "stinky",
    "burn", "burning", "burnt", "burned", "overheat", "overheating", "overheated",
    "hot", "heat", "heating", "heated", "warm", "warming", "warmed", "cool", "cooling",
    "melting", "melted", "melt", "warpage", "warp", "warped", "deform", "deformed",
}


def extract_meaningful_words(text: str) -> list[str]:
    """提取有意义的实词（排除停用词）"""
    text_lower = text.lower()
    # 提取单词（支持带连字符和撇号的词）
    words = re.findall(r"[a-z][a-z'-]*[a-z]|[a-z]", text_lower)
    # 过滤停用词和过短词
    return [w for w in words if w not in STOP_WORDS and len(w) >= 3]


def analyze_zero_label_texts(max_samples: int = 100000):
    """分析零标签文本的高频实词"""
    print("=" * 70)
    print("零标签文本高频实词挖掘")
    print("=" * 70)

    # 收集零标签文本
    zero_texts = []
    total_records = 0
    zero_count = 0

    for src in ["amazon", "trustpilot", "reddit", "zendesk"]:
        src_dir = OUTPUT_BASE / src
        if not src_dir.exists():
            continue
        batch_files = sorted(src_dir.glob("batch_*.jsonl"))
        for bf in batch_files:
            with open(bf, "r", encoding="utf-8") as f:
                for line in f:
                    total_records += 1
                    r = json.loads(line)
                    if r["n_tags"] == 0 and r.get("text_preview"):
                        zero_count += 1
                        zero_texts.append((src, r["text_preview"]))

                    if len(zero_texts) >= max_samples:
                        break
                if len(zero_texts) >= max_samples:
                    break
            if len(zero_texts) >= max_samples:
                break

    print(f"\n总记录: {total_records:,}")
    print(f"零标签: {zero_count:,} ({zero_count/total_records*100:.1f}%)")
    print(f"分析样本: {len(zero_texts):,}")

    # 统计高频词
    word_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()
    src_word_dist = defaultdict(Counter)

    for src, text in zero_texts:
        words = extract_meaningful_words(text)
        for w in words:
            word_counts[w] += 1
            src_word_dist[src][w] += 1

        # 2-gram
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            bigram_counts[bigram] += 1

        # 3-gram
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            trigram_counts[trigram] += 1

    # 输出结果
    print(f"\n{'=' * 70}")
    print("Top 100 高频实词 (单字)")
    print(f"{'=' * 70}")
    for w, c in word_counts.most_common(100):
        marker = " 🏷️" if w in PRODUCT_WORDS else ""
        print(f"  {w:20s} {c:6d}{marker}")

    print(f"\n{'=' * 70}")
    print("Top 50 高频 2-gram")
    print(f"{'=' * 70}")
    for bg, c in bigram_counts.most_common(50):
        print(f"  {bg:40s} {c:6d}")

    print(f"\n{'=' * 70}")
    print("Top 30 高频 3-gram")
    print(f"{'=' * 70}")
    for tg, c in trigram_counts.most_common(30):
        print(f"  {tg:50s} {c:6d}")

    print(f"\n{'=' * 70}")
    print("产品相关高频词（白名单命中）")
    print(f"{'=' * 70}")
    product_hits = [(w, c) for w, c in word_counts.most_common() if w in PRODUCT_WORDS]
    for w, c in product_hits[:30]:
        print(f"  {w:20s} {c:6d}")

    # 保存完整结果
    result = {
        "total_records": total_records,
        "zero_count": zero_count,
        "zero_rate": round(zero_count / total_records, 4),
        "sample_size": len(zero_texts),
        "top_words": word_counts.most_common(200),
        "top_bigrams": bigram_counts.most_common(100),
        "top_trigrams": trigram_counts.most_common(50),
        "product_words": [(w, c) for w, c in word_counts.most_common() if w in PRODUCT_WORDS][:50],
    }

    output_path = OUTPUT_BASE / "zero_label_mining.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {output_path}")

    return result


def suggest_keyword_expansions(mining_result: dict):
    """基于挖掘结果，建议关键词扩展"""
    print(f"\n{'=' * 70}")
    print("关键词扩展建议")
    print(f"{'=' * 70}")

    # 高频但可能未被标签覆盖的词
    top_words = dict(mining_result["top_words"][:100])

    suggestions = []

    # 按主题聚类建议
    theme_clusters = {
        "舒适/疼痛": ["pain", "comfortable", "comfort", "sore", "hurt", "aching", "relief", "support", "back", "hip", "neck", "shoulder"],
        "噪音": ["noise", "noisy", "quiet", "silent", "loud", "sound", "whisper", "hum", "buzz"],
        "吸力": ["suction", "suck", "pull", "strength", "power", "strong", "weak", "gentle"],
        "电池/充电": ["battery", "charge", "charging", "charged", "cord", "plug", "outlet", "electric", "power"],
        "清洁": ["clean", "cleaning", "wash", "washing", "sterilize", "sanitize", "dishwasher", "boil"],
        "便携/旅行": ["portable", "travel", "compact", "small", "lightweight", "bag", "carrying"],
        "尺码/合身": ["size", "fit", "fitting", "tight", "loose", "small", "large", "band", "strap"],
        "材质": ["fabric", "material", "cotton", "silicone", "plastic", "soft", "smooth", "stretchy"],
        "泄漏": ["leak", "leaking", "spill", "spilling", "drip", "dripping"],
        "使用体验": ["easy", "easily", "simple", "difficult", "hard", "tricky", "convenient", "hassle"],
        "设计/外观": ["design", "color", "look", "appearance", "cute", "pretty", "ugly", "aesthetic"],
        "温度": ["hot", "heat", "warm", "cold", "cool", "burning", "overheat", "temperature"],
        "价格/价值": ["price", "expensive", "cheap", "affordable", "value", "worth", "budget", "cost"],
        "客服/售后": ["service", "support", "contact", "response", "helpful", "rude", "help"],
        "物流": ["shipping", "delivery", "arrived", "package", "box", "fast", "slow", "quick"],
    }

    for theme, words in theme_clusters.items():
        found = [(w, top_words.get(w, 0)) for w in words if w in top_words]
        if found:
            found.sort(key=lambda x: -x[1])
            total = sum(c for _, c in found)
            print(f"\n【{theme}】总出现: {total}")
            for w, c in found[:8]:
                print(f"  - {w}: {c}")
            suggestions.append({
                "theme": theme,
                "words": found,
                "total": total,
            })

    return suggestions


def main():
    result = analyze_zero_label_texts(max_samples=150000)
    suggest_keyword_expansions(result)
    print(f"\n{'=' * 70}")
    print("分析完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

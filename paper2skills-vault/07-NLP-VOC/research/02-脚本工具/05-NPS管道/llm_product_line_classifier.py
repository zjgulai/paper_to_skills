"""LLM 品线分类器（采样版）

对规则分类困难的 "other" 文本采样，用 Claude API 做品线分类。
分类结果用于提取关键词、优化规则。

Usage:
    export ANTHROPIC_API_KEY=sk-xxx
    python llm_product_line_classifier.py --sample-size 500 --output llm_classified.json
"""

import argparse
import json
import os
import random
from pathlib import Path

from anthropic import Anthropic

INPUT_DIR = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/00-归档资料/labeling-outputs/v3.3")

PRODUCT_LINES = [
    "breast_pump", "wearable_breast_pump", "pregnancy_pillow", "nursing_bra",
    "bottle_warmer", "sterilizer", "humidifier", "air_purifier", "sound_machine",
    "red_light_therapy", "stroller", "car_seat", "diaper_bag", "baby_bottle",
    "baby_monitor", "baby_carrier", "breast_milk_storage", "postpartum_recovery",
    "baby_clothing", "bottle_washer", "nipple_cream", "breast_pad", "baby_wipe",
    "crib_playard", "customer_service", "other",
]

CLASSIFICATION_PROMPT = """You are a product line classifier for a mother & baby cross-border e-commerce company.

Task: Classify the following customer review/product mention into exactly one product line.

Available product lines:
- breast_pump: Traditional electric/manual breast pump
- wearable_breast_pump: Hands-free, in-bra wearable pump (Elvie, Willow, Momcozy S12/M series)
- pregnancy_pillow: C-shaped, U-shaped, wedge pillows for pregnant women
- nursing_bra: Bras designed for breastfeeding/pumping
- bottle_warmer: Device to warm baby bottles
- sterilizer: UV or steam device to sterilize baby bottles/pacifiers
- humidifier: Adds moisture to air (cool mist, warm mist, ultrasonic)
- air_purifier: Cleans air with HEPA filter
- sound_machine: White noise, nature sounds for baby sleep
- red_light_therapy: LED/infrared light therapy device
- stroller: Baby pram, pushchair, buggy
- car_seat: Infant/convertible/booster car seat
- diaper_bag: Bag for carrying baby essentials
- baby_bottle: Feeding bottle, sippy cup, training cup
- baby_monitor: Video/audio/wifi monitor for baby
- baby_carrier: Wrap, ring sling, soft structured carrier
- breast_milk_storage: Milk bags, containers for freezing breast milk
- postpartum_recovery: Belly wrap, compression garment, recovery belt
- baby_clothing: Onesies, sleepers, rompers, bodysuits
- bottle_washer: Electric/manual device to wash baby bottles
- nipple_cream: Cream/balm for sore nursing nipples
- breast_pad: Nursing pads for leak protection
- baby_wipe: Wet wipes for diaper changes
- other: Cannot determine from text

Rules:
1. Respond ONLY with the product line name (lowercase with underscores)
2. If the text is primarily about customer service, shipping, delivery, returns, or refunds, respond "customer_service"
3. If the text is too vague/generic or about a product not in the list, respond "other"
4. If multiple products are mentioned, pick the MAIN product being reviewed
5. Important distinction:
   - "nursing_bra" = bras for breastfeeding/pumping (hands-free pumping bra, clip-down nursing bra)
   - "wearable_breast_pump" = actual breast pump device that goes in the bra (Elvie, Willow, Momcozy S12)
   - If text mentions "pumping bra" or "hands-free bra" → nursing_bra
   - If text mentions "wearable pump" or "in-bra pump" → wearable_breast_pump

Text to classify:
"""


def load_other_samples(sample_size: int) -> list[dict]:
    """加载规则分类为 other 的文本样本"""
    other_records = []

    # 简化的规则分类（与 generate_product_line_spu_matrix.py 五级分类一致）
    def classify(text: str) -> str:
        text_lower = text.lower()

        # customer_service 分流
        service_score = 0
        for kw in ['customer service', 'conversation with', 'web user', 'follow-up', 'shipping', 'delivery', 'return', 'refund', 'warranty', 'replacement', 'schnelle lieferung', 'livraison rapide']:
            if kw in text_lower:
                service_score += 2 if ' ' in kw else 1
        if service_score >= 3:
            has_product = any(kw in text_lower for kw in ['breast pump','pregnancy pillow','nursing bra','stroller','bottle warmer','wearable pump','baby monitor','sterilizer','humidifier','air purifier','sound machine','red light','car seat','diaper bag','baby bottle','baby carrier','milk storage','postpartum','baby clothes','bottle washer','nipple cream','breast pad','baby wipe','crib','playard','bassinet'])
            if not has_product:
                return 'customer_service'

        # 强特征词
        strong_features = {
            'sound_machine': ['white noise', 'brown noise', 'pink noise', 'lullaby', 'nature sounds', 'sound machine', 'sleep machine', 'hatch', 'dohm'],
            'crib_playard': ['bassinet', 'playard', 'playpen', 'travel crib', 'pack n play', 'pack and play', 'co sleeper', 'co-sleeper', 'bedside sleeper'],
            'air_purifier': ['hepa', 'air purifier', 'air cleaner', 'allergen'],
            'sterilizer': ['uv sterilizer', 'steam sterilizer', 'bottle sterilizer', 'pacifier sterilizer'],
            'red_light_therapy': ['red light therapy', 'infrared therapy', 'led therapy', 'phototherapy'],
            'baby_monitor': ['baby monitor', 'video monitor', 'audio monitor', 'breathing monitor', 'movement monitor', 'nanit', 'owlet', 'miku'],
            'humidifier': ['humidifier', 'vaporizer', 'cool mist', 'warm mist'],
            'postpartum_recovery': ['faja', 'waist trainer', 'shapewear', 'compression garment', 'compression band', 'belly binder', 'postpartum belly', 'postpartum girdle', 'c section recovery', 'c-section recovery', 'recovery belt'],
            'nipple_cream': ['lanolin', 'nipple butter', 'nipple balm', 'nipple ointment'],
            'breast_pad': ['nursing pad', 'breast pad', 'bra pad', 'leak pad'],
            'breast_milk_storage': ['milk bag', 'milk freezer', 'milk stash', 'freeze milk', 'frozen milk'],
            'bottle_washer': ['bottle washer', 'bottle cleaner'],
            'wearable_breast_pump': ['wearable pump', 'wearable breast pump', 'hands free pump', 'hands-free pump', 'in-bra pump', 'in bra pump', 'wireless pump'],
            'car_seat': ['infant car seat', 'convertible car seat', 'booster seat', '5 point harness'],
            'baby_carrier': ['ring sling', 'soft structured carrier', 'ssc', 'moby wrap', 'solly wrap'],
            'baby_bottle': ['sippy cup', 'training cup', 'straw cup', 'transition cup', 'bottle nipple', 'slow flow', 'medium flow', 'fast flow'],
            'stroller': ['jogging stroller', 'travel stroller', 'umbrella stroller', 'double stroller', 'twin stroller'],
            'diaper_bag': ['diaper tote', 'changing bag', 'diaper bag backpack'],
        }
        for line, phrases in strong_features.items():
            for phrase in phrases:
                if phrase in text_lower:
                    return line

        # Level 1 关键词
        for line, kws in {
            'pregnancy_pillow': ['pregnancy pillow','body pillow','wedge pillow','maternity pillow','pregnant pillow','boppy pillow','c pillow','u shaped pillow','u-shaped pillow','full body pillow'],
            'nursing_bra': ['nursing bra','breastfeeding bra','pumping bra','nursing bras','breastfeeding bras','pumping bras','maternity bra','nursing tank','nursing cami','breast pump bra','hands free bra','hands-free bra','hands-free pumping bra'],
            'wearable_breast_pump': ['wearable pump','wearable breast pump','hands free pump','hands-free pump','handsfree pump','wireless pump','portable pump','discreet pump','in-bra pump','in bra pump'],
            'breast_pump': ['breast pump','breastpump','pump and dump','pumping at work','pumping session','pump parts','pump flange','pump tubing','pump valve'],
            'bottle_warmer': ['bottle warmer','wipe warmer','portable bottle warmer','baby bottle warmer','milk warmer','formula warmer'],
            'sterilizer': ['sterilizer','steriliser','uv sterilizer','steam sterilizer','bottle sterilizer','pacifier sterilizer','baby bottle sterilizer','microwave sterilizer','electric sterilizer'],
            'humidifier': ['humidifier','vaporizer','cool mist humidifier','warm mist humidifier','ultrasonic humidifier','baby humidifier','nursery humidifier'],
            'air_purifier': ['air purifier','air purifiers','hepa filter','air cleaner','air quality','air filtration','room purifier'],
            'sound_machine': ['sound machine','white noise','white noise machine','noise machine','baby sound machine','sleep machine','soothing sounds','lullaby machine','nature sounds'],
            'red_light_therapy': ['red light therapy','red light','light therapy','redlight therapy','led light therapy','infrared therapy','phototherapy','light therapy lamp','red light device'],
            'stroller': ['stroller','pram','pushchair','buggy','baby stroller','jogging stroller','travel stroller','umbrella stroller','double stroller','twin stroller','stroller frame'],
            'car_seat': ['car seat','infant car seat','convertible car seat','booster seat','carseat','infant carseat','baby car seat','5 point harness','rear facing','forward facing'],
            'diaper_bag': ['diaper bag','nappy bag','diaper backpack','diaper tote','changing bag','baby bag','diaper bag backpack'],
            'baby_bottle': ['baby bottle','feeding bottle','sippy cup','training cup','straw cup','transition cup','nipple bottle','slow flow','medium flow','fast flow','bottle nipple'],
            'baby_monitor': ['baby monitor','video monitor','audio monitor','wifi monitor','smart baby monitor','breathing monitor','movement monitor','nanit','owlet','miku','cocoon cam'],
            'baby_carrier': ['baby carrier','wrap carrier','ring sling','soft structured carrier','ssc','ergo baby','tula','lillebaby','moby wrap','solly wrap','baby wrap','front carrier','back carrier','hip carrier'],
            'breast_milk_storage': ['milk storage','breast milk storage','milk bag','milk bags','storage bag','storage bags','milk container','breast milk bag','milk freezer','milk stash','freeze milk','frozen milk'],
            'postpartum_recovery': ['postpartum','postpartum recovery','postpartum belly','postpartum girdle','c section recovery','c-section recovery','after birth','post birth','post delivery','recovery belt','belly wrap','postpartum wrap','compression garment','faja','waist trainer','shapewear','compression band','belly binder'],
            'baby_clothing': ['baby clothes','baby clothing','onesie','onesies','footie','footies','newborn clothes','infant clothes','baby outfit'],
            'bottle_washer': ['bottle washer','bottle cleaner','brush cleaner','bottle brush'],
            'nipple_cream': ['nipple cream','nipple butter','lanolin','nipple balm','nipple ointment'],
            'breast_pad': ['nursing pad','breast pad','bra pad','disposable pad','washable pad','leak pad'],
            'baby_wipe': ['baby wipe','baby wipes','wet wipe','wet wipes','diaper wipe'],
            'crib_playard': ['crib','playard','playpen','travel crib','pack n play','pack and play','bassinet','co sleeper','co-sleeper','bedside sleeper','mini crib','convertible crib'],
        }.items():
            for kw in kws:
                if kw in text_lower:
                    return line

        # 加权上下文（排除 bra 的 wearable_breast_pump）
        ctx_rules = {
            'breast_pump': (['pump','pumping','milchpumpe'], ['suction','flange','milk','letdown','spectra','medela','lansinoh','tubing','valve','diaphragm','backflow']),
            'wearable_breast_pump': (['pump','pumping','milchpumpe'], ['wearable','hands free','hands-free','wireless','in bra','in-bra','discreet','portable','elvie','willow','momcozy s12','momcozy m']),
            'pregnancy_pillow': (['pillow'], ['pregnancy','pregnant','maternity','boppy','wedge','c shaped','u shaped','body pillow','side sleeper','belly support']),
            'baby_bottle': (['bottle','bottles'], ['nipple','nipples','slow flow','medium flow','fast flow','dr brown','avent','tommee tippee','formula','feeding']),
            'nursing_bra': (['bra','bras'], ['nursing','breastfeeding','pumping','hands free','clip down','nursing clip','kindred bravely']),
            'baby_monitor': (['monitor'], ['baby','nursery','cry','crying','sleep','breathing','movement','nanit','owlet','miku','wifi','wi-fi','camera','connect']),
            'sterilizer': (['sterilize','sterilise','sanitize'], ['bottle','pacifier','uv','steam','microwave','wabi baby','papablic']),
            'bottle_warmer': (['warmer'], ['bottle','milk','formula','wipe','dr brown','avent']),
            'stroller': (['stroller','pram','pushchair'], ['buggy','jogging','travel','umbrella','double','uppababy','baby jogger','bugaboo','fold','unfold','recline']),
            'car_seat': (['car seat','carseat'], ['infant','convertible','booster','5 point','rear facing','forward facing','britax','graco','chicco','harness','buckle']),
            'humidifier': (['humidifier'], ['mist','vapor','ultrasonic','nursery','cool mist','warm mist','filter','humidity','bedroom']),
            'air_purifier': (['purifier'], ['air','hepa','filter','cleaner','allergen','dust','pollen','smoke','dogs','cats','pet','smell','odor','room']),
            'sound_machine': (['noise','sound'], ['white noise','brown noise','lullaby','nature','rain','ocean','shush','sleep','baby','nursery','hatch','dohm','alarm']),
            'baby_carrier': (['carrier','wrap','sling'], ['baby','ergo','tula','lillebaby','moby','solly','front carry','back carry','newborn','toddler']),
            'postpartum_recovery': (['postpartum','faja','compression','waist'], ['belly','c section','c-section','recovery','girdle','wrap','belt','shape','stomach','wear','binder']),
            'baby_wipe': (['wipe','wipes'], ['baby','diaper','rash','sensitive','soft','thick']),
        }
        for line_name, (kws, ctxs) in ctx_rules.items():
            if line_name == 'wearable_breast_pump' and ('bra' in text_lower or 'bras' in text_lower):
                continue
            score = 0
            for kw in kws:
                if kw in text_lower:
                    score += 3
            for ctx in ctxs:
                if ctx in text_lower:
                    score += 2
            threshold = 7 if line_name == 'wearable_breast_pump' else 5
            if score >= threshold:
                return line_name

        return 'other'

    for src in ["amazon", "trustpilot", "reddit", "zendesk"]:
        src_dir = INPUT_DIR / src
        if not src_dir.exists():
            continue
        for batch_file in sorted(src_dir.glob("batch_*.jsonl")):
            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    text = r.get("text_preview", "")
                    if classify(text) == "other":
                        other_records.append({
                            "review_id": r["review_id"],
                            "text": text,
                            "source": src,
                            "tags": [t["tag_en"] for t in r.get("aipl_tags", [])],
                        })

    # 随机采样
    if len(other_records) > sample_size:
        random.seed(42)
        other_records = random.sample(other_records, sample_size)

    print(f"  other 文本总数: {len([r for r in other_records]):,}")
    print(f"  采样数: {len(other_records):,}")
    return other_records


def classify_with_llm(records: list[dict], batch_size: int = 20) -> list[dict]:
    """用 Claude API 批量分类"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return []

    client = Anthropic(api_key=api_key)
    results = []

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        # 构建批量提示
        texts = []
        for j, r in enumerate(batch):
            texts.append(f"[{j+1}] {r['text'][:300]}")

        prompt = CLASSIFICATION_PROMPT + "\n\n".join(texts) + """

Respond with exactly one product line per item, in this format:
[1]: product_line_name
[2]: product_line_name
..."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            output = response.content[0].text

            # 解析输出
            lines = output.strip().split("\n")
            for j, r in enumerate(batch):
                predicted = "other"
                for line in lines:
                    if line.startswith(f"[{j+1}]:"):
                        predicted = line.split(":", 1)[1].strip().lower().replace(" ", "_")
                        if predicted not in PRODUCT_LINES:
                            predicted = "other"
                        break

                results.append({
                    **r,
                    "llm_predicted": predicted,
                })

            print(f"  批次 {i//batch_size + 1}/{(len(records)-1)//batch_size + 1}: {len(batch)} 条已分类")

        except Exception as e:
            print(f"  批次 {i//batch_size + 1} 失败: {e}")
            for r in batch:
                results.append({**r, "llm_predicted": "error"})

    return results


def analyze_results(results: list[dict]):
    """分析 LLM 分类结果，提取关键词"""
    from collections import Counter, defaultdict

    predicted_dist = Counter(r["llm_predicted"] for r in results if r["llm_predicted"] != "error")
    print("\n" + "=" * 70)
    print("LLM 分类分布")
    print("=" * 70)
    for line, cnt in predicted_dist.most_common():
        print(f"  {line}: {cnt} ({cnt/len(results)*100:.1f}%)")

    # 每个品线的高频词
    print("\n" + "=" * 70)
    print("各品线高频关键词 (Top 5)")
    print("=" * 70)
    line_words = defaultdict(Counter)
    for r in results:
        pl = r["llm_predicted"]
        if pl in ("other", "error"):
            continue
        words = r["text"].lower().split()
        for w in words:
            w = "".join(c for c in w if c.isalpha())
            if len(w) >= 4 and w not in {
                "this", "that", "with", "from", "they", "have", "been", "were", "when", "than", "them",
                "would", "could", "should", "there", "their", "about", "after", "before", "very",
                "really", "just", "also", "only", "even", "much", "more", "most", "some", "many",
                "well", "good", "great", "nice", "love", "like", "back", "time", "year", "years",
                "month", "months", "week", "weeks", "day", "days", "hour", "hours", "first", "last",
            }:
                line_words[pl][w] += 1

    for line in sorted(line_words.keys()):
        words = line_words[line].most_common(5)
        print(f"  {line}: {', '.join(f'{w}({c})' for w, c in words)}")

    return predicted_dist


def main():
    parser = argparse.ArgumentParser(description="LLM 品线分类器（采样版）")
    parser.add_argument("--sample-size", type=int, default=500, help="采样数量")
    parser.add_argument("--batch-size", type=int, default=20, help="每批请求条数")
    parser.add_argument("--output", default="llm_classified.json", help="输出文件")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM 品线分类")
    print("=" * 70)

    # 1. 加载样本
    print("\n[1/3] 加载 other 文本样本...")
    samples = load_other_samples(args.sample_size)

    # 2. LLM 分类
    print(f"\n[2/3] LLM 分类 ({args.sample_size} 条)...")
    results = classify_with_llm(samples, batch_size=args.batch_size)

    # 3. 分析结果
    print("\n[3/3] 分析结果...")
    analyze_results(results)

    # 保存
    output_path = INPUT_DIR / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()

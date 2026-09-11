"""品线分类器（规则 + LLM 混合）

基于用户选择：
- 先用规则对有明确产品关键词的文本快速分类
- 对模糊文本用 LLM 精准分类

品线类别：
- breast_pump, wearable_breast_pump, bottle_warmer, sterilizer
- pregnancy_pillow, nursing_bra, baby_carrier, stroller, car_seat
- baby_monitor, humidifier, air_purifier, sound_machine
- red_light_therapy, diaper_bag, baby_bottle, breast_milk_storage
- postpartum_recovery, baby_clothing, other
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

from anthropic import Anthropic

# 品线规则分类器（快速、免费）
PRODUCT_LINE_RULES = {
    "pregnancy_pillow": [
        "pregnancy pillow", "body pillow", "wedge pillow", "pregnancy wedge",
        "maternity pillow", "pregnant pillow", "boppy pillow", "c pillow",
        "u shaped pillow", "u-shaped pillow", "full body pillow",
    ],
    "nursing_bra": [
        "nursing bra", "breastfeeding bra", "pumping bra", "nursing bras",
        "breastfeeding bras", "pumping bras", "maternity bra", "nursing tank",
        "nursing cami", "breast pump bra", "hands free bra",
    ],
    "wearable_breast_pump": [
        "wearable pump", "wearable breast pump", "hands free pump",
        "hands-free pump", "handsfree pump", "wireless pump",
        "portable pump", "discreet pump", "in-bra pump", "in bra pump",
    ],
    "breast_pump": [
        "breast pump", "breastpump", "pump and dump", "pumping at work",
        "spectra", "medela", "elvie pump", "willow pump", "lansinoh",
        "ameda", "hygeia", "motif", "evenflo pump", "pumping session",
        "pump parts", "pump flange", "pump tubing", "pump valve",
        "suction", "let down", "letdown", "milk supply", "oversupply",
        "low supply", "pumping schedule", "exclusive pumping", "ep",
    ],
    "bottle_warmer": [
        "bottle warmer", "wipe warmer", "portable bottle warmer",
        "baby bottle warmer", "milk warmer", "formula warmer",
    ],
    "sterilizer": [
        "sterilizer", "steriliser", "uv sterilizer", "steam sterilizer",
        "bottle sterilizer", "pacifier sterilizer", "baby bottle sterilizer",
        "microwave sterilizer", "electric sterilizer",
    ],
    "humidifier": [
        "humidifier", "vaporizer", "cool mist humidifier", "warm mist humidifier",
        "ultrasonic humidifier", "baby humidifier", "nursery humidifier",
    ],
    "air_purifier": [
        "air purifier", "air purifiers", "hepa filter", "air cleaner",
        "air quality", "air filtration", "room purifier",
    ],
    "sound_machine": [
        "sound machine", "white noise", "white noise machine", "noise machine",
        "baby sound machine", "sleep machine", "soothing sounds",
        "lullaby machine", "nature sounds",
    ],
    "red_light_therapy": [
        "red light therapy", "red light", "light therapy", "redlight therapy",
        "led light therapy", "infrared therapy", "phototherapy",
        "light therapy lamp", "red light device",
    ],
    "stroller": [
        "stroller", "pram", "pushchair", "buggy", "baby stroller",
        "jogging stroller", "travel stroller", "umbrella stroller",
        "double stroller", "twin stroller", "stroller frame",
    ],
    "car_seat": [
        "car seat", "infant car seat", "convertible car seat",
        "booster seat", "carseat", "infant carseat", "baby car seat",
        "5 point harness", "rear facing", "forward facing",
    ],
    "diaper_bag": [
        "diaper bag", "nappy bag", "diaper backpack", "diaper tote",
        "changing bag", "baby bag", "diaper bag backpack",
    ],
    "baby_bottle": [
        "baby bottle", "feeding bottle", "sippy cup", "training cup",
        "straw cup", "transition cup", "nipple bottle", "slow flow",
        "medium flow", "fast flow", "bottle nipple",
    ],
    "baby_monitor": [
        "baby monitor", "video monitor", "audio monitor", "wifi monitor",
        "smart baby monitor", "breathing monitor", "movement monitor",
        "nanit", "owlet", "miku", "cocoon cam",
    ],
    "baby_carrier": [
        "baby carrier", "wrap carrier", "ring sling", "soft structured carrier",
        "ssc", "ergo baby", "tula", "lillebaby", "moby wrap", "solly wrap",
        "baby wrap", "front carrier", "back carrier", "hip carrier",
    ],
    "breast_milk_storage": [
        "milk storage", "breast milk storage", "milk bag", "milk bags",
        "storage bag", "storage bags", "milk container", "breast milk bag",
        "milk freezer", "milk stash", "freeze milk", "frozen milk",
    ],
    "postpartum_recovery": [
        "postpartum", "postpartum recovery", "postpartum belly",
        "postpartum girdle", "c section recovery", "c-section recovery",
        "after birth", "post birth", "post delivery", "recovery belt",
        "belly wrap", "postpartum wrap", "compression garment",
    ],
    "baby_clothing": [
        "baby clothes", "baby clothing", "onesie", "onesies", "sleeper",
        "sleepers", "footie", "footies", "romper", "rompers", "bodysuit",
        "bodysuits", "newborn clothes", "infant clothes", "baby outfit",
    ],
}

# 品线优先级（一个文本匹配多个规则时，按优先级选择）
LINE_PRIORITY = [
    "wearable_breast_pump",  # 更具体的吸奶器
    "breast_pump",
    "red_light_therapy",     # 特定治疗设备
    "sound_machine",
    "air_purifier",
    "humidifier",
    "baby_monitor",
    "sterilizer",
    "bottle_warmer",
    "pregnancy_pillow",
    "nursing_bra",
    "baby_carrier",
    "stroller",
    "car_seat",
    "diaper_bag",
    "baby_bottle",
    "breast_milk_storage",
    "postpartum_recovery",
    "baby_clothing",
]


class RuleBasedClassifier:
    """基于规则的品线分类器"""

    def __init__(self):
        self.rules = PRODUCT_LINE_RULES

    def classify(self, text: str) -> "str | None":
        """规则分类，返回品线或 None"""
        text_lower = text.lower()
        matches = []

        for line, keywords in self.rules.items():
            for kw in keywords:
                # 支持词组匹配（带空格）和单词匹配
                if " " in kw:
                    if kw in text_lower:
                        matches.append(line)
                        break
                else:
                    # 单词边界匹配
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, text_lower):
                        matches.append(line)
                        break

        if not matches:
            return None

        # 按优先级选择第一个
        for priority_line in LINE_PRIORITY:
            if priority_line in matches:
                return priority_line

        return matches[0]


class LLMProductLineClassifier:
    """基于 LLM 的品线分类器"""

    def __init__(self):
        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )
        self.model = "claude-sonnet-4-6"  # 使用快速模型

    def classify_batch(self, texts: list[str], batch_size: int = 20) -> list[str]:
        """批量分类"""
        # 构建提示
        lines_desc = "\n".join(
            f"- {k}: {', '.join(v[:3])}"
            for k, v in PRODUCT_LINE_RULES.items()
        )

        prompt = f"""You are a product category classifier for mother & baby e-commerce reviews.

Classify each review into ONE of these product lines. If unclear, use "other".

Available product lines:
{lines_desc}
- other: None of the above or unclear

Rules:
1. Return ONLY a JSON array of strings, one per review
2. Each string must be exactly one of the product line keys above
3. Base your decision on explicit product mentions in the text
4. If multiple products are mentioned, pick the primary one being reviewed

Reviews to classify:
"""
        for i, text in enumerate(texts[:batch_size], 1):
            # 截断文本到前 200 字符
            truncated = text[:200].replace('"', '"')
            prompt += f"\n[{i}] {truncated}"

        prompt += "\n\nReturn ONLY a JSON array like [\"breast_pump\", \"other\", \"pregnancy_pillow\", ...]:"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()

            # 提取 JSON
            # 尝试直接解析
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return result[:len(texts)]
            except json.JSONDecodeError:
                pass

            # 尝试从代码块中提取
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_str)
                if isinstance(result, list):
                    return result[:len(texts)]

            # 尝试从方括号中提取
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                result = json.loads(content[start:end+1])
                if isinstance(result, list):
                    return result[:len(texts)]

        except Exception as e:
            print(f"LLM classification error: {e}")

        # 失败时全部返回 other
        return ["other"] * len(texts)


class HybridProductLineClassifier:
    """混合分类器：规则优先，LLM兜底"""

    def __init__(self, use_llm: bool = True, llm_batch_size: int = 20):
        self.rule_classifier = RuleBasedClassifier()
        self.use_llm = use_llm
        if use_llm:
            self.llm_classifier = LLMProductLineClassifier()
        self.llm_batch_size = llm_batch_size

    def classify_single(self, text: str) -> str:
        """单条分类"""
        # 先尝试规则
        result = self.rule_classifier.classify(text)
        if result:
            return result

        # 规则失败，用 LLM（单条）
        if self.use_llm:
            try:
                results = self.llm_classifier.classify_batch([text], batch_size=1)
                return results[0] if results else "other"
            except Exception:
                pass

        return "other"

    def classify_batch(self, texts: list[str]) -> list[str]:
        """批量分类"""
        results = []
        llm_texts = []
        llm_indices = []

        # 第一轮：规则分类
        for i, text in enumerate(texts):
            rule_result = self.rule_classifier.classify(text)
            if rule_result:
                results.append((i, rule_result))
            else:
                results.append((i, None))
                llm_texts.append(text)
                llm_indices.append(i)

        # 第二轮：LLM 分类（对规则失败的）
        if self.use_llm and llm_texts:
            llm_results = []
            for i in range(0, len(llm_texts), self.llm_batch_size):
                batch = llm_texts[i:i + self.llm_batch_size]
                batch_results = self.llm_classifier.classify_batch(batch)
                llm_results.extend(batch_results)

            for idx, llm_result in zip(llm_indices, llm_results):
                results[idx] = (idx, llm_result)

        # 排序并返回
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


def analyze_data_sources():
    """分析各数据源的品线分布"""
    import pandas as pd

    OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest")

    configs = {
        "amazon": ("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv", "Content"),
        "trustpilot": ("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/trustpilot_voc_100k_balanced.csv", "Content"),
        "reddit": ("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/reddit_voc_sampled.csv", "Content"),
        "zendesk": ("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/zendesk_momcozy_voc_sampled.csv", "Content"),
    }

    classifier = HybridProductLineClassifier(use_llm=False)

    print("=" * 70)
    print("品线分布分析（规则分类）")
    print("=" * 70)

    for source, (csv_path, text_col) in configs.items():
        print(f"\n--- {source.upper()} ---")
        df = pd.read_csv(csv_path, nrows=5000)
        texts = df[text_col].dropna().astype(str).tolist()

        lines = Counter()
        unclassified = 0

        for text in texts:
            line = classifier.rule_classifier.classify(text)
            if line:
                lines[line] += 1
            else:
                unclassified += 1

        print(f"  样本: {len(texts)}")
        print(f"  已分类: {len(texts) - unclassified} ({(len(texts) - unclassified)/len(texts)*100:.1f}%)")
        print(f"  未分类: {unclassified} ({unclassified/len(texts)*100:.1f}%)")
        print("  品线分布:")
        for line, cnt in lines.most_common():
            print(f"    {line}: {cnt} ({cnt/len(texts)*100:.1f}%)")


if __name__ == "__main__":
    analyze_data_sources()

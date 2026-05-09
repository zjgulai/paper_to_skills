"""品线/品类推断（Phase 1.4-1.5）

对无 SPU 映射的数据（Amazon/Trustpilot/Reddit）基于文本关键词推断品线和品类，
更新统一打标结果中的 product_line 和 category 字段。
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ── 品线推断规则 ──────────────────────────────────────────────────

LINE_KEYWORDS = {
    "吸奶器": {
        "keywords": [
            "pump", "pumping", "breast pump", "wearable pump", "suction", "flange",
            "express", "extract", "milk", "breast milk", "lactation", "letdown",
            "spectra", "medela", "elvie", "willow", "momcozy pump", "m5", "m6", "m9",
            "hands free", "portable pump", "electric pump", "manual pump",
        ],
        "categories": ["吸奶器", "一体穿戴式吸奶器", "分离穿戴式吸奶器", "双边穿戴式吸奶器"],
    },
    "内衣服饰": {
        "keywords": [
            "bra", "nursing bra", "maternity bra", "breastfeeding bra", "pumping bra",
            "underwear", "panty", "brief", "shapewear", "bodysuit", "compression",
            "postpartum belly", "belly band", "belly wrap", "waist trainer",
            "sling", "carrier", "wrap", "babywearing", "hip seat",
            "sock", "tights", "legging", "pressure sock",
            "clothing", "apparel", "wear", "fabric", "lace", "cotton",
        ],
        "categories": ["文胸", "常规内衣", "常规内裤", "塑身衣", "塑身裤", "塑身内裤", "收腹带", "托腹带", "孕妇内裤", "孕妇裤", "压力袜", "背奶包", "产妇袜", "运动内衣", "运动长裤", "船袜"],
    },
    "喂养电器": {
        "keywords": [
            "sterilizer", "sterilise", "sterilization", "uv sterilizer", "steam sterilizer",
            "bottle warmer", "warmer", "formula maker", "mixer", "blender",
            "food processor", "baby food", "puree", "steamer", "baby bottle",
            "nipple", "pacifier", "teether", "sippy cup", "straw cup",
            "bottle brush", "cleaning brush", "drying rack", "bottle dryer",
            "wipe warmer", "wipe dispenser",
        ],
        "categories": ["消毒器", "调奶器", "辅食机", "奶瓶清洗机", "奶瓶沥干支架", "电动奶瓶清洁刷", "硅胶奶瓶清洁刷", "湿巾加热盒", "清洁液", "清洁喷雾", "牙刷", "牙胶组合", "全硅胶款牙胶", "注水款牙胶", "辅食装袋器", "辅食袋"],
    },
    "智能母婴电器": {
        "keywords": [
            "thermometer", "temperature", "fever", "forehead thermometer", "ear thermometer",
            "scale", "weight", "baby scale", "monitor", "baby monitor", "camera",
            "white noise", "sound machine", "night light", "humidifier", "purifier",
            "air purifier", "fan", "heater", "cooler",
            "nasal aspirator", "nose sucker", "snot sucker", "bulb syringe",
            "nail clipper", "nail trimmer", "electric nail", "baby nail",
            "massager", "massage", "lactation massager", "breast massager",
            "high chair", "booster seat", "feeding chair", "stroller", "pram", "buggy",
            "bassinet", "playpen", "playard", "rocker", "bouncer", "swing",
        ],
        "categories": ["温度计", "儿童体重秤", "婴儿监视器", "贴片式体温计配件", "BN02婴儿吸鼻器配件", "BN007吸鼻器配件", "分体式吸鼻器", "手持式吸鼻器", "磨甲器", "电动磨甲器配件", "lupantte磨甲器配件", "NC02电动磨甲器配件", "乳房按摩仪", "乳房按摩乳垫", "乳房按摩仪配件", "常规乳房按摩仪", "滚珠按摩仪", "儿童温度计", "电动摇椅", "3D摇电动摇椅配件", "成长型餐椅", "餐椅配件", "推车", "轻型婴儿车", "推车通用配件", "车挂包", "ST01多功能婴儿标准车配件", "桌面空气净化器", "台式白噪音机器", "便携白噪音机器", "奶袋", "储奶袋", "旋钮奶袋", "硅胶奶袋", "储奶壶", "GP01储奶壶配件"],
    },
    "家居家纺": {
        "keywords": [
            "pillow", "pregnancy pillow", "body pillow", "nursing pillow", "boppy",
            "mattress", "crib mattress", "mattress pad", "sheet", "crib sheet",
            "blanket", "quilt", "comforter", "duvet", "swaddle", "sleep sack",
            "towel", "washcloth", "bib", "burp cloth", "changing pad",
            "play mat", "activity mat", "tummy time", "gym", "rug",
            "curtain", "blind", "shade",
        ],
        "categories": ["夹腿枕", "记忆棉哺乳枕", "U型哺乳枕", "U型双边孕妇枕", "U型双边孕妇枕套", "U型哺乳枕套", "J型单边孕妇枕", "J型二代单边孕妇枕", "J型单边孕妇枕套", "G型双边孕妇枕", "W型孕妇枕", "哺乳枕", "腰凳", "腰凳配件", "腰凳背带", "圆环背巾", "插扣背巾", "包巾", "襁褓巾", "安抚巾", "浴巾", "方巾", "床笠", "凉感被", "凉感毯"],
    },
    "母婴综合护理": {
        "keywords": [
            "diaper", "nappy", "changing", "wipe", "wet wipe", "baby wipe",
            "lotion", "cream", "ointment", "rash cream", "diaper rash",
            "shampoo", "body wash", "soap", "bubble bath", "bath bomb",
            "sunscreen", "spf", "insect repellent", "thermometer patch",
            "nursing pad", "breast pad", "nipple cream", "lanolin",
        ],
        "categories": ["沐浴炸弹", "沐浴片", "应急垫", "常规运动垫", "围栏款运动垫", "双布套运动垫", "便携隔尿垫", "便携隔尿垫包", "一次性防溢乳垫", "可水洗防溢乳垫", "防挤压支架", "吸奶器内衣", "吸奶器背心", "吸奶器湿巾", "保温包", "妈咪包", "收纳包", "待产包", "沐浴露", "洗发水", "屁屁膏/尿布膏", "尿布膏喷雾", "生理盐水湿巾", "纯水湿巾", "乳液湿巾", "洗护套装", "护理套装", "婴儿无袖睡袋", "柔软纸尿裤", "竹纤维纸尿裤", "大环腰拉拉裤", "伸缩门栏", "跳舞毯", "其他玩具", "单边音乐毯", "婴儿纱布四季盖毯"],
    },
}


def infer_line(text: str) -> tuple[str, str]:
    """基于文本推断品线和品类"""
    text_lower = text.lower()

    # 统计每个品线的关键词命中数
    line_scores = {}
    for line, info in LINE_KEYWORDS.items():
        score = sum(1 for kw in info["keywords"] if kw in text_lower)
        line_scores[line] = score

    if not line_scores or max(line_scores.values()) == 0:
        return "", ""

    # 取最高分品线
    best_line = max(line_scores, key=line_scores.get)
    best_score = line_scores[best_line]

    if best_score == 0:
        return "", ""

    # 在最佳品线中进一步推断品类
    categories = LINE_KEYWORDS[best_line]["categories"]
    cat_scores = {}
    for cat in categories:
        # 用品类名作为关键词
        cat_kw = cat.lower().replace("/", " ").replace("\\", " ")
        # 简单子串匹配
        if any(c in text_lower for c in cat_kw.split() if len(c) >= 3):
            cat_scores[cat] = 1

    best_cat = max(cat_scores, key=cat_scores.get) if cat_scores else ""

    return best_line, best_cat


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 1.4-1.5: 品线/品类推断 + 输出审阅")
    print("=" * 70)

    input_path = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling/phase1_3_all_sources_labeled.jsonl"
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取打标结果
    print("\n--- 读取打标结果 ---")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    print(f"  总计: {len(records):,} 条")

    # 统计当前 product_line/category 缺失情况
    no_line = sum(1 for r in records if not r.get("product_line"))
    no_cat = sum(1 for r in records if not r.get("category"))
    print(f"  缺品线: {no_line:,} ({no_line/len(records)*100:.1f}%)")
    print(f"  缺品类: {no_cat:,} ({no_cat/len(records)*100:.1f}%)")

    # 推断品线/品类
    print("\n--- 品线/品类推断 ---")
    inferred_count = 0
    line_stats = defaultdict(int)
    cat_stats = defaultdict(int)

    for i, r in enumerate(records):
        if (i + 1) % 50000 == 0:
            print(f"  进度: {i + 1:,} / {len(records):,}")

        # 只对无品线的记录进行推断
        if not r.get("product_line"):
            line, cat = infer_line(r["text"])
            if line:
                r["product_line"] = line
                r["_inferred_line"] = True
                line_stats[line] += 1
                inferred_count += 1
            if cat:
                r["category"] = cat
                r["_inferred_category"] = True
                cat_stats[cat] += 1

    print(f"  推断成功: {inferred_count:,} 条")
    print(f"  品线分布:")
    for line, cnt in sorted(line_stats.items(), key=lambda x: -x[1]):
        print(f"    {line}: {cnt:,}")

    # 保存更新后的结果
    print("\n--- 保存更新结果 ---")
    output_path = output_dir / "phase1_5_all_sources_labeled_final.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  输出: {output_path}")

    # 审阅统计
    print("\n--- 审阅统计 ---")
    final_no_line = sum(1 for r in records if not r.get("product_line"))
    final_no_cat = sum(1 for r in records if not r.get("category"))
    print(f"  最终缺品线: {final_no_line:,} ({final_no_line/len(records)*100:.1f}%)")
    print(f"  最终缺品类: {final_no_cat:,} ({final_no_cat/len(records)*100:.1f}%)")

    # 覆盖率按数据源
    print(f"\n  各数据源覆盖率:")
    source_coverage = defaultdict(lambda: {"total": 0, "with_tags": 0})
    for r in records:
        ds = r["data_source"]
        source_coverage[ds]["total"] += 1
        if r["n_tags"] > 0:
            source_coverage[ds]["with_tags"] += 1

    for ds, stat in sorted(source_coverage.items()):
        rate = stat["with_tags"] / stat["total"] * 100
        print(f"    {ds}: {stat['with_tags']:,} / {stat['total']:,} ({rate:.1f}%)")

    # 标签分布
    tag_dist = Counter(r["n_tags"] for r in records)
    print(f"\n  标签数分布:")
    for n in sorted(tag_dist.keys())[:8]:
        print(f"    {n} 个标签: {tag_dist[n]:,} ({tag_dist[n]/len(records)*100:.1f}%)")

    # 审计报告
    audit = {
        "phase": "1.4-1.5",
        "total_records": len(records),
        "inferred_count": inferred_count,
        "missing_line_before": no_line,
        "missing_line_after": final_no_line,
        "missing_category_before": no_cat,
        "missing_category_after": final_no_cat,
        "line_breakdown": dict(line_stats),
        "source_coverage": {
            ds: {"total": s["total"], "with_tags": s["with_tags"], "rate": round(s["with_tags"]/s["total"]*100, 1)}
            for ds, s in source_coverage.items()
        },
        "output_path": str(output_path),
    }
    audit_path = output_dir / "phase1_5_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 1.4-1.5 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

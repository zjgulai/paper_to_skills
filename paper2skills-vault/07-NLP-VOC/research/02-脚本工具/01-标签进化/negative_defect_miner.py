"""负面缺陷字典扩展工具（Phase 4 T2）

目标：从 Momcozy 自有数据的零标签样本中，挖掘负面缺陷描述，
生成候选标签并更新标签字典。

输入：Phase 3 最终打标结果（phase3_p3_labeled.jsonl）
输出：候选标签 JSON + 标签字典更新脚本

步骤：
1. 加载零标签 Momcozy 样本
2. 负面情感筛选（基于负面词典）
3. 高频缺陷描述聚类
4. 候选标签生成（tag_id / tag_en / tag_cn / keywords）
5. 标签字典更新
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# ── 负面情感词典（用于筛选负面零标签）───────────────────────────────

NEGATIVE_SENTIMENT_WORDS = {
    # 基础负面
    "bad", "terrible", "horrible", "awful", "worst", "poor", "disappointing",
    "disappointed", "frustrated", "frustrating", "annoyed", "annoying",
    "unhappy", "unsatisfied", "regret", "useless", "worthless", "useless",
    # 缺陷描述
    "broken", "broke", "damaged", "defective", "faulty", "malfunction",
    "failed", "failure", "error", "issue", "problem", "problems",
    "doesn't work", "doesnt work", "not working", "stopped working",
    "won't turn on", "wont turn on", "not charging", "not charge",
    "leaking", "leak", "leaks", "leaked", "spill", "spills", "spilling",
    "crack", "cracked", "cracking", "scratch", "scratched", "dented",
    "stain", "stained", "discolor", "discolored", "fading", "faded",
    "smell", "smells", "smelly", "odor", "stink", "stinky", "burning smell",
    "overheat", "overheating", "overheated", "too hot", "burning",
    "melt", "melted", "melting", "warp", "warped", "deform", "deformed",
    "loose", "loosen", "fall off", "falls off", "falling off", "detach",
    "detached", "come off", "comes off", "coming off", "peel", "peeling",
    "tear", "tears", "tearing", "torn", "rip", "rips", "ripped", "ripping",
    "wear", "worn", "wearing out", "wore out", "fray", "fraying", "frayed",
    # 功能失效
    "no suction", "weak suction", "low suction", "suction weak",
    "not strong", "too weak", "no power", "noisy", "loud", "squeak",
    "squeaking", "rattling", "vibrating", "buzzing", "clicking",
    # 体验负面
    "hurt", "hurts", "hurting", "pain", "painful", "sore", "soreness",
    "uncomfortable", "itchy", "scratchy", "rough", "irritating",
    "too tight", "too loose", "doesn't fit", "doesnt fit", "wrong size",
    "too small", "too big", "too large", "runs small", "runs large",
    # 其他负面
    "missing", "not included", "incomplete", "no adapter", "no cable",
    "no charger", "no manual", "no instructions",
    "difficult", "hard to", "complicated", "confusing", "not intuitive",
    "waste", "waste of", "overpriced", "expensive", "not worth",
    # 德/法负面
    "kaputt", "defekt", "nicht funktioniert", "schlecht", "mangelhaft",
    "trop cher", "mauvais", "défectueux", "cassé", "ne fonctionne pas",
}


def is_negative_text(text: str) -> tuple[bool, list[str]]:
    """判断文本是否为负面描述，返回 (是否负面, 命中词列表)"""
    text_lower = text.lower()
    matched = []
    for neg_word in NEGATIVE_SENTIMENT_WORDS:
        if " " in neg_word:
            if neg_word in text_lower:
                matched.append(neg_word)
        else:
            if re.search(r'\b' + re.escape(neg_word) + r'\b', text_lower):
                matched.append(neg_word)
    return len(matched) > 0, matched


# ── 缺陷主题聚类 ───────────────────────────────────────────────────

DEFECT_CLUSTERS = {
    "功能失效": {
        "keywords": [
            "doesn't work", "doesnt work", "not working", "stopped working",
            "won't turn on", "wont turn on", "no power", "dead",
            "malfunction", "faulty", "defective", "broken", "broke",
            "not charging", "not charge", "no suction", "weak suction",
            "low suction", "not strong", "too weak", "no response",
            "kaputt", "defekt", "nicht funktioniert",
            "ne fonctionne pas", "défectueux", "cassé",
        ],
        "suggested_tag": {
            "tag_en": "functional_failure",
            "tag_cn": "功能失效",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
    "泄漏问题": {
        "keywords": [
            "leaking", "leak", "leaks", "leaked", "spill", "spills",
            "spilling", "drip", "dripping", "drips",
        ],
        "suggested_tag": {
            "tag_en": "leakage_issue",
            "tag_cn": "泄漏问题",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
    "材质/表面损伤": {
        "keywords": [
            "crack", "cracked", "cracking", "scratch", "scratched",
            "dented", "stain", "stained", "discolor", "discolored",
            "fading", "faded", "peel", "peeling", "tear", "torn",
            "rip", "ripped", "fray", "fraying", "frayed",
        ],
        "suggested_tag": {
            "tag_en": "surface_damage",
            "tag_cn": "表面损伤",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
    "异味/过热": {
        "keywords": [
            "smell", "smells", "smelly", "odor", "stink", "stinky",
            "burning smell", "chemical smell", "plastic smell",
            "overheat", "overheating", "overheated", "too hot", "burning",
            "melt", "melted", "melting", "warp", "warped",
        ],
        "suggested_tag": {
            "tag_en": "odor_overheating",
            "tag_cn": "异味过热",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
    "结构松动": {
        "keywords": [
            "loose", "loosen", "fall off", "falls off", "falling off",
            "detach", "detached", "come off", "comes off", "coming off",
            "wobble", "wobbly", "not secure", "not stable",
        ],
        "suggested_tag": {
            "tag_en": "structural_looseness",
            "tag_cn": "结构松动",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
    "噪音问题": {
        "keywords": [
            "noisy", "loud", "squeak", "squeaking", "rattling",
            "vibrating", "buzzing", "clicking", "grinding", "whining",
        ],
        "suggested_tag": {
            "tag_en": "noise_issue",
            "tag_cn": "噪音问题",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
    "缺少配件": {
        "keywords": [
            "missing", "not included", "incomplete", "no adapter",
            "no cable", "no charger", "no manual", "no instructions",
            "parts missing", "missing part",
        ],
        "suggested_tag": {
            "tag_en": "missing_parts",
            "tag_cn": "缺少配件",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
    "磨损老化": {
        "keywords": [
            "wear", "worn", "wearing out", "wore out", "deteriorate",
            "deteriorated", "aging", "aged", "old", "not durable",
        ],
        "suggested_tag": {
            "tag_en": "wear_aging",
            "tag_cn": "磨损老化",
            "aipl": "L1",
            "sentiment": "negative",
        },
    },
}


# ── 核心挖掘函数 ───────────────────────────────────────────────────

def mine_negative_defects(
    input_path: Path,
    output_dir: Path,
    source_filter: Optional[str] = None,
    min_samples: int = 10,
    max_records: int = 500000,
) -> dict:
    """挖掘负面缺陷描述，生成候选标签

    Args:
        input_path: Phase 3 打标结果路径
        output_dir: 输出目录
        source_filter: 数据源过滤（如 "momcozy"）
        min_samples: 生成候选标签的最小样本数
        max_records: 最大读取记录数

    Returns: 挖掘结果字典
    """
    print("=" * 70)
    print("Phase 4 T2: 负面缺陷字典扩展")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("\n--- 步骤1: 加载零标签样本 ---")
    zero_label_records = []
    total = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            rec = json.loads(line)
            total += 1

            # 筛选零标签
            n_tags = rec.get("n_tags", len(rec.get("labels", [])))
            if n_tags != 0:
                continue

            # 筛选数据源
            if source_filter:
                src = rec.get("data_source", "").lower()
                if source_filter.lower() not in src:
                    continue

            zero_label_records.append(rec)

    print(f"  总记录: {total:,}")
    print(f"  零标签样本: {len(zero_label_records):,}")

    if not zero_label_records:
        print("  ⚠️ 无零标签样本，终止")
        return {}

    # 2. 负面情感筛选
    print("\n--- 步骤2: 负面情感筛选 ---")
    negative_records = []
    for rec in zero_label_records:
        text = rec.get("text", "")
        is_neg, matched = is_negative_text(text)
        if is_neg:
            rec["negative_matched"] = matched
            negative_records.append(rec)

    print(f"  负面样本: {len(negative_records):,} ({len(negative_records)/len(zero_label_records)*100:.1f}%)")

    if not negative_records:
        print("  ⚠️ 无负面样本，终止")
        return {}

    # 3. 缺陷聚类
    print("\n--- 步骤3: 缺陷主题聚类 ---")
    cluster_results = {}

    for cluster_name, cluster_def in DEFECT_CLUSTERS.items():
        matched_records = []
        keyword_hits = Counter()

        for rec in negative_records:
            text_lower = rec.get("text", "").lower()
            hit_kws = []
            for kw in cluster_def["keywords"]:
                if " " in kw:
                    if kw in text_lower:
                        hit_kws.append(kw)
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                        hit_kws.append(kw)

            if hit_kws:
                rec_copy = dict(rec)
                rec_copy["cluster_keywords"] = hit_kws
                matched_records.append(rec_copy)
                for kw in hit_kws:
                    keyword_hits[kw] += 1

        cluster_results[cluster_name] = {
            "count": len(matched_records),
            "records": matched_records,
            "keyword_hits": dict(keyword_hits.most_common(20)),
            "suggested_tag": cluster_def["suggested_tag"],
        }

        print(f"  {cluster_name}: {len(matched_records):,} 条")

    # 4. 生成候选标签
    print("\n--- 步骤4: 生成候选标签 ---")
    candidate_tags = []
    tag_id_counter = 1

    for cluster_name, result in cluster_results.items():
        if result["count"] < min_samples:
            continue

        tag_def = result["suggested_tag"]
        # 从实际命中的关键词中选择高频词作为标签关键词
        top_kws = [kw for kw, _ in Counter(result["keyword_hits"]).most_common(15)]

        candidate = {
            "tag_id": f"TAG_DEF_N{tag_id_counter:03d}",
            "tag_en": tag_def["tag_en"],
            "tag_cn": tag_def["tag_cn"],
            "aipl": tag_def["aipl"],
            "sentiment": tag_def["sentiment"],
            "category": "缺陷",
            "cluster": cluster_name,
            "sample_count": result["count"],
            "keywords": top_kws,
            "keyword_coverage": result["keyword_hits"],
        }
        candidate_tags.append(candidate)
        tag_id_counter += 1

    print(f"  候选标签数: {len(candidate_tags)}")
    for ct in candidate_tags:
        print(f"    {ct['tag_id']} {ct['tag_cn']} ({ct['cluster']}): {ct['sample_count']} 样本, {len(ct['keywords'])} 关键词")

    # 5. 未聚类负面样本分析
    print("\n--- 步骤5: 未聚类负面样本分析 ---")
    clustered_ids = set()
    for result in cluster_results.values():
        for rec in result["records"]:
            clustered_ids.add(id(rec))

    unclustered = [r for r in negative_records if id(r) not in clustered_ids]
    print(f"  未聚类负面样本: {len(unclustered):,}")

    # 高频未覆盖词
    unclustered_words = Counter()
    for rec in unclustered:
        text = rec.get("text", "").lower()
        words = re.findall(r'[a-z]{4,}', text)
        for w in words:
            if w not in {
                "this", "that", "with", "from", "have", "been", "were",
                "they", "them", "their", "there", "when", "what", "than",
                "just", "only", "also", "very", "really", "would", "could",
            }:
                unclustered_words[w] += 1

    print(f"  高频未覆盖词 Top 10:")
    for w, c in unclustered_words.most_common(10):
        print(f"    {w}: {c}")

    # 6. 保存结果
    result_dict = {
        "total_zero_label": len(zero_label_records),
        "negative_samples": len(negative_records),
        "negative_rate": round(len(negative_records) / len(zero_label_records), 4) if zero_label_records else 0,
        "candidate_tags": candidate_tags,
        "cluster_stats": {k: {
            "count": v["count"],
            "keyword_hits": v["keyword_hits"],
        } for k, v in cluster_results.items()},
        "unclustered_count": len(unclustered),
        "unclustered_top_words": unclustered_words.most_common(30),
    }

    # 保存候选标签
    candidate_path = output_dir / "negative_defect_candidates.json"
    with open(candidate_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"\n  候选标签已保存: {candidate_path}")

    # 保存样本（每个聚类最多50条）
    samples_path = output_dir / "negative_defect_samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for cluster_name, result in cluster_results.items():
            for rec in result["records"][:50]:
                f.write(json.dumps({
                    "cluster": cluster_name,
                    "text": rec.get("text", "")[:200],
                    "keywords": rec.get("cluster_keywords", []),
                }, ensure_ascii=False) + "\n")
    print(f"  样本已保存: {samples_path}")

    print("\n" + "=" * 70)
    return result_dict


# ── 标签字典更新函数 ───────────────────────────────────────────────

def generate_label_functions(candidates: list[dict]) -> str:
    """从候选标签生成 Python Label Functions 代码"""
    code_lines = ['"""负面缺陷 Label Functions（Phase 4 T2 自动生成）"""', ""]
    code_lines.append("NEGATIVE_DEFECT_TAGS = [")

    for cand in candidates:
        kw_str = ", ".join(f'"{kw}"' for kw in cand["keywords"])
        code_lines.append(f"    {{")
        code_lines.append(f'        "tag_id": "{cand["tag_id"]}",')
        code_lines.append(f'        "tag_en": "{cand["tag_en"]}",')
        code_lines.append(f'        "tag_cn": "{cand["tag_cn"]}",')
        code_lines.append(f'        "aipl": "{cand["aipl"]}",')
        code_lines.append(f'        "sentiment": "{cand["sentiment"]}",')
        code_lines.append(f'        "keywords": [{kw_str}],')
        code_lines.append(f"    }},")

    code_lines.append("]")
    code_lines.append("")
    code_lines.append("""
def apply_negative_defect_tags(text: str) -> list[dict]:
    \"\"\"应用负面缺陷标签\"\"\"
    text_lower = text.lower()
    labels = []
    for tag in NEGATIVE_DEFECT_TAGS:
        for kw in tag["keywords"]:
            if " " in kw:
                if kw in text_lower:
                    labels.append({
                        "tag_id": tag["tag_id"],
                        "tag_en": tag["tag_en"],
                        "tag_cn": tag["tag_cn"],
                        "aipl_node": tag["aipl"],
                        "sentiment_preset": tag["sentiment"],
                        "sentiment_calibrated": -1.0,
                        "confidence": 0.75,
                        "source": "negative_defect_label",
                    })
                    break
            else:
                if re.search(r'\\b' + re.escape(kw) + r'\\b', text_lower):
                    labels.append({
                        "tag_id": tag["tag_id"],
                        "tag_en": tag["tag_en"],
                        "tag_cn": tag["tag_cn"],
                        "aipl_node": tag["aipl"],
                        "sentiment_preset": tag["sentiment"],
                        "sentiment_calibrated": -1.0,
                        "confidence": 0.75,
                        "source": "negative_defect_label",
                    })
                    break
    return labels
""")

    return "\n".join(code_lines)


# ── 自证测试 ───────────────────────────────────────────────────────

def _test():
    """负面缺陷挖掘工具自证测试"""
    print("=" * 70)
    print("负面缺陷挖掘工具自证测试")
    print("=" * 70)

    # 测试负面情感检测
    test_texts = [
        ("The pump stopped working after 2 weeks", True, "功能失效"),
        ("Great product, love it!", False, "正面文本"),
        ("Leaking everywhere, terrible quality", True, "泄漏+质量"),
        ("Just a normal question about usage", False, "中性文本"),
        ("It's too noisy and gets hot", True, "噪音+过热"),
        ("Missing the charger cable", True, "缺少配件"),
    ]

    print("\n--- 负面情感检测测试 ---")
    passed = 0
    for text, expected, desc in test_texts:
        is_neg, matched = is_negative_text(text)
        status = "PASS" if is_neg == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"  [{status}] {desc}: neg={is_neg} (命中: {matched})")
    print(f"  情感检测: {passed}/{len(test_texts)} 通过")

    # 测试聚类
    print("\n--- 聚类匹配测试 ---")
    cluster_tests = [
        ("The pump stopped working", "功能失效"),
        ("It's leaking milk everywhere", "泄漏问题"),
        ("The plastic is cracked", "材质/表面损伤"),
        ("Burning smell when charging", "异味/过热"),
        ("The flange keeps falling off", "结构松动"),
        ("Too noisy at night", "噪音问题"),
        ("Missing the power adapter", "缺少配件"),
    ]

    cluster_passed = 0
    for text, expected_cluster in cluster_tests:
        text_lower = text.lower()
        matched_clusters = []
        for cluster_name, cluster_def in DEFECT_CLUSTERS.items():
            for kw in cluster_def["keywords"]:
                if " " in kw:
                    if kw in text_lower:
                        matched_clusters.append(cluster_name)
                        break
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                        matched_clusters.append(cluster_name)
                        break

        status = "PASS" if expected_cluster in matched_clusters else "FAIL"
        if status == "PASS":
            cluster_passed += 1
        print(f"  [{status}] '{text[:40]}...' → {matched_clusters} (期望: {expected_cluster})")
    print(f"  聚类匹配: {cluster_passed}/{len(cluster_tests)} 通过")

    # 测试候选标签生成
    print("\n--- 候选标签生成测试 ---")
    mock_candidates = [
        {
            "tag_id": "TAG_DEF_N001",
            "tag_en": "functional_failure",
            "tag_cn": "功能失效",
            "aipl": "L1",
            "sentiment": "negative",
            "keywords": ["stopped working", "doesn't work", "not working"],
        },
        {
            "tag_id": "TAG_DEF_N002",
            "tag_en": "leakage_issue",
            "tag_cn": "泄漏问题",
            "aipl": "L1",
            "sentiment": "negative",
            "keywords": ["leaking", "leak", "spill"],
        },
    ]
    code = generate_label_functions(mock_candidates)
    has_import = "def apply_negative_defect_tags" in code
    has_tags = "NEGATIVE_DEFECT_TAGS" in code
    gen_status = "PASS" if has_import and has_tags else "FAIL"
    print(f"  [{gen_status}] 代码生成: has_func={has_import}, has_tags={has_tags}")

    total_passed = passed + cluster_passed + (1 if gen_status == "PASS" else 0)
    total_tests = len(test_texts) + len(cluster_tests) + 1
    print(f"\n测试结果: {total_passed}/{total_tests} 通过 ({total_passed/total_tests*100:.1f}%)")

    print("=" * 70)
    return total_passed, total_tests


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _test()
    else:
        # 默认运行模式：需要实际数据文件
        input_path = Path("04-输出结果/unified_labeling/phase3_p3_labeled.jsonl")
        output_dir = Path("04-输出结果/negative_defect_mining")

        if not input_path.exists():
            print(f"⚠️ 输入文件不存在: {input_path}")
            print("运行自证测试: python negative_defect_miner.py --test")
            _test()
        else:
            mine_negative_defects(input_path, output_dir, source_filter="momcozy")

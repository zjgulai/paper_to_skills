"""标签体系逆向完善分析器

基于诊断数据，分析标签质量问题并输出优化建议。

核心问题: 标签字典覆盖多场景(售后/客服/社媒/电商评论)，
但 Amazon 评论仅覆盖其中一小部分。
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/Users/pray/project/paper_to_skills/paper2skills-code/nlp_voc/proxy_nps_aipl_workflow")

from unified_label_extraction import TagSeedDictionary


def analyze_zero_hit_tags(tag_dict, hit_tags_set):
    """分析零命中标签的特征和原因"""
    zero_hit = []
    for tag in tag_dict.get_all():
        if tag.tag_en not in hit_tags_set:
            zero_hit.append(tag)

    # 按 AIPL 节点分布
    by_aipl = Counter(t.aipl_node for t in zero_hit)
    # 按主题分布
    by_theme = Counter(t.theme for t in zero_hit)
    # 按数据源适用性
    by_source = Counter()
    for t in zero_hit:
        src = tuple(sorted(t.applicable_source)) if t.applicable_source else ("any",)
        by_source[src] += 1

    # 关键词特征
    short_kw_tags = []  # 只有短关键词（<4字符）
    no_kw_tags = []  # 无关键词
    for t in zero_hit:
        kws = t.keywords + t.consumer_keywords
        if not kws:
            no_kw_tags.append(t.tag_en)
        elif all(len(kw) < 4 for kw in kws):
            short_kw_tags.append(t.tag_en)

    return {
        "total": len(zero_hit),
        "by_aipl": dict(by_aipl),
        "by_theme": dict(by_theme.most_common(20)),
        "by_source": {str(k): v for k, v in by_source.most_common()},
        "no_keywords": no_kw_tags,
        "only_short_keywords": short_kw_tags,
        "examples": [
            {
                "tag_en": t.tag_en,
                "tag_cn": t.tag_cn,
                "aipl": t.aipl_node,
                "theme": t.theme,
                "keywords": t.keywords + t.consumer_keywords,
                "applicable_source": t.applicable_source,
            }
            for t in zero_hit[:30]
        ],
    }


def analyze_high_conflict_tags(tag_dict, diagnostic_details):
    """分析高冲突标签 — 哪些标签频繁导致情感冲突"""
    # 统计每个标签的命中次数和冲突次数
    tag_stats = defaultdict(lambda: {"hits": 0, "conflicts": 0, "examples": []})

    for d in diagnostic_details:
        for tag in d["tags"]:
            tag_stats[tag]["hits"] += 1
            if d["is_conflict"]:
                tag_stats[tag]["conflicts"] += 1
            # 保存示例（限制数量）
            if len(tag_stats[tag]["examples"]) < 3:
                tag_stats[tag]["examples"].append({
                    "text": d["text_preview"],
                    "rating": d["rating"],
                })

    # 计算冲突率并排序
    conflict_analysis = []
    for tag, stats in tag_stats.items():
        if stats["hits"] >= 3:  # 至少命中3次才分析
            rate = stats["conflicts"] / stats["hits"] * 100
            conflict_analysis.append({
                "tag": tag,
                "hits": stats["hits"],
                "conflicts": stats["conflicts"],
                "conflict_rate": rate,
                "examples": stats["examples"],
            })

    conflict_analysis.sort(key=lambda x: -x["conflict_rate"])
    return conflict_analysis


def analyze_over_matching(diagnostic_details, tag_by_en):
    """分析过度匹配 — 哪些标签关键词太宽泛导致命中过多"""
    tag_counts = Counter()
    for d in diagnostic_details:
        for tag in d["tags"]:
            tag_counts[tag] += 1

    # 命中数 > 20 的标签可能存在过度匹配
    over_matching = []
    for tag, count in tag_counts.most_common():
        if count < 10:
            break
        seed = tag_by_en.get(tag)
        if seed:
            over_matching.append({
                "tag": tag,
                "hit_count": count,
                "keywords": seed.keywords + seed.consumer_keywords,
                "sentiment_preset": seed.sentiment_preset,
            })

    return over_matching


def generate_improvement_plan(tag_dict, report_path, detail_path):
    """生成逆向完善计划"""
    # 建立英文标签名到 TagSeed 的快速查找
    tag_by_en = {}
    for t in tag_dict.get_all():
        tag_by_en[t.tag_en] = t

    def get_seed_by_en(tag_en):
        return tag_by_en.get(tag_en)
    """生成逆向完善计划"""
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(detail_path, "r", encoding="utf-8") as f:
        details = json.load(f)

    hit_tags = set(t for t, _ in report.get("top_hit_tags", []))

    print("=" * 70)
    print("标签体系逆向完善分析报告")
    print("=" * 70)

    # 1. 零命中标签分析
    print("\n--- 一、零命中标签分析 ---")
    zero_analysis = analyze_zero_hit_tags(tag_dict, hit_tags)
    print(f"零命中标签总数: {zero_analysis['total']} / 212 ({zero_analysis['total']/212*100:.1f}%)")

    print(f"\n按 AIPL 节点分布:")
    for node, cnt in sorted(zero_analysis["by_aipl"].items()):
        print(f"  {node}: {cnt} 个")

    print(f"\n按数据源适用性分布:")
    for src, cnt in zero_analysis["by_source"].items():
        print(f"  {src}: {cnt} 个")

    print(f"\n无关键词标签: {len(zero_analysis['no_keywords'])} 个")
    if zero_analysis["no_keywords"]:
        print(f"  示例: {zero_analysis['no_keywords'][:10]}")

    print(f"\n仅短关键词标签: {len(zero_analysis['only_short_keywords'])} 个")

    # 2. 高冲突标签分析
    print("\n--- 二、高冲突标签分析 ---")
    conflict_analysis = analyze_high_conflict_tags(tag_dict, details)
    high_conflict = [c for c in conflict_analysis if c["conflict_rate"] > 50]
    print(f"冲突率 > 50% 的标签: {len(high_conflict)} 个")
    for c in high_conflict[:15]:
        print(f"\n  {c['tag']}:")
        print(f"    命中: {c['hits']}, 冲突: {c['conflicts']}, 冲突率: {c['conflict_rate']:.1f}%")
        seed = get_seed_by_en(c['tag'])
        if seed:
            print(f"    情感预设: {seed.sentiment_preset}")
            print(f"    关键词: {seed.keywords + seed.consumer_keywords}")
        for ex in c["examples"]:
            print(f"    示例(评分{ex['rating']}): {ex['text'][:100]}")

    # 3. 过度匹配分析
    print("\n--- 三、过度匹配分析 ---")
    over_matching = analyze_over_matching(details, tag_by_en)
    print(f"命中 > 10 次的标签: {len(over_matching)} 个")
    for om in over_matching[:10]:
        print(f"\n  {om['tag']}: {om['hit_count']} 次")
        print(f"    关键词: {om['keywords']}")
        print(f"    情感预设: {om['sentiment_preset']}")

    # 4. 逆向完善建议
    print("\n" + "=" * 70)
    print("四、逆向完善建议")
    print("=" * 70)

    print("""
【问题根因】
标签字典覆盖全旅程(A→I→P1→P2→L1→L2→L3)和多数据源，
但 Amazon 评论主要集中在 L1(使用体验)和少量 A(品牌认知)。
大量标签为售后/客服/DTC/社媒场景设计，在 Amazon 评论中自然极少出现。

【这不是"关键词不够"的问题，而是"标签与数据源不匹配"的问题】

【建议方案】

1. 按数据源分层打标（立即执行）
   - Amazon: 仅用 applicable_source 包含 "review" 或空的标签
   - Zendesk: 用售后/客服类标签 (L2/L3 节点为主)
   - Trustpilot: 用品牌认知/推荐类标签 (A/L3 节点)
   - Reddit: 用社媒种草/对比类标签 (A/I 节点)

2. 为 Amazon 场景补充体验标签（高优先级）
   当前 Amazon 缺少的关键体验维度:
   - 舒适度/人体工学 (comfort, ergonomic)
   - 便携性 (portable, travel-friendly)
   - 续航能力 (battery life, charge)
   - 清洁便利性 (easy to clean, dishwasher safe)
   - 噪音控制 (quiet, loud, noisy)
   - 吸力强度 (suction strength, weak suction)
   - 配件/零件 (parts, accessories, flange)
   - 性价比 (worth the money, overpriced)

3. 优化高冲突标签（中优先级）
   - general_core_product_performance_issue: 关键词过于宽泛("product", "work")
     → 应拆分为具体性能维度
   - instruction_user_manual: 关键词 "instruction", "manual" 过于通用
     → 增加上下文限定（如 "manual" + "confusing/unclear"）
   - size_runs_large/small: 需排除 "large amount", "small space" 等非尺寸语境

4. 零命中标签分类处理（低优先级）
   - 售后类(80+个): 标记为 "zendesk_only"，不在 Amazon 使用
   - 社媒类(20+个): 标记为 "social_only"，不在 Amazon 使用
   - 安全类(10+个): 保留但补充同义词（如 "burn" → "got hot", "overheated"）

5. 建立标签-数据源映射矩阵
   输出: tag_applicability_matrix.csv
   列: tag_id, tag_en, amazon_ok, zendesk_ok, trustpilot_ok, reddit_ok
   基于: applicable_source + 实际命中数据验证
""")

    # 5. 输出优化后的标签字典
    print("\n--- 五、输出优化建议文件 ---")
    output_dir = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果"

    # 高冲突标签清单
    high_conflict_list = [
        {
            "tag_en": c["tag"],
            "hits": c["hits"],
            "conflicts": c["conflicts"],
            "conflict_rate": round(c["conflict_rate"], 1),
            "suggestion": "关键词过于宽泛，需增加限定词或上下文"
        }
        for c in conflict_analysis if c["conflict_rate"] > 30
    ]

    # 零命中标签清单（分类）
    zero_hit_list = []
    for ex in zero_analysis["examples"]:
        category = "unknown"
        if any(kw in ex["tag_en"] for kw in ["return", "refund", "exchange", "cancel"]):
            category = "售后场景"
        elif any(kw in ex["tag_en"] for kw in ["support", "service", "response", "help"]):
            category = "客服场景"
        elif any(kw in ex["tag_en"] for kw in ["influencer", "content", "review_fake", "sponsored"]):
            category = "社媒场景"
        elif any(kw in ex["tag_en"] for kw in ["cannot_", "malfunction", "broken", "leakage", "damage"]):
            category = "故障场景"
        elif any(kw in ex["tag_en"] for kw in ["shipping", "delivery", "package", "logistics"]):
            category = "物流场景"
        zero_hit_list.append({
            "tag_en": ex["tag_en"],
            "tag_cn": ex["tag_cn"],
            "aipl": ex["aipl"],
            "theme": ex["theme"],
            "category": category,
            "keywords": ex["keywords"],
        })

    improvement_plan = {
        "summary": {
            "total_tags": 212,
            "zero_hit": zero_analysis["total"],
            "high_conflict": len(high_conflict_list),
            "over_matching": len(over_matching),
        },
        "high_conflict_tags": high_conflict_list,
        "zero_hit_tags": zero_hit_list,
        "over_matching_tags": over_matching,
    }

    plan_path = f"{output_dir}/tag_improvement_plan.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(improvement_plan, f, ensure_ascii=False, indent=2)
    print(f"  优化计划已保存: {plan_path}")

    print("\n" + "=" * 70)
    print("逆向完善分析完成")
    print("=" * 70)

    return improvement_plan


def main():
    xlsx_path = "/Users/pray/project/sgcs/20_insights/22_insight_reports/03_voc分析/01_内部VOC精细化运营洞察框架/SGCS_VOC标签字典_喂养电器V3.2_final.xlsx"
    tag_dict = TagSeedDictionary.from_xlsx(xlsx_path)

    report_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/06-诊断报告/tag_diagnostic_report.json"
    detail_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/06-诊断报告/tag_diagnostic_details.json"

    generate_improvement_plan(tag_dict, report_path, detail_path)


if __name__ == "__main__":
    main()

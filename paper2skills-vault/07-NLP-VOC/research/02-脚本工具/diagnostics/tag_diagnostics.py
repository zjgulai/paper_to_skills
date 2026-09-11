"""标签质量诊断工具

运行 1000 条 Amazon VOC，收集每个标签的命中/冲突详情，
输出诊断报告指导逆向完善。

策略: 最大化覆盖率(不过滤停用词+子串匹配)，完整记录诊断数据。
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/pray/project/paper_to_skills/paper2skills-code/nlp_voc/proxy_nps_aipl_workflow")

from unified_label_extraction import (
    TagSeedDictionary,
    VOCLabelExtractor,
    VOCRecord,
)


def load_tag_dict():
    """加载真实标签字典"""
    xlsx_path = "/Users/pray/project/sgcs/20_insights/22_insight_reports/03_voc分析/01_内部VOC精细化运营洞察框架/SGCS_VOC标签字典_喂养电器V3.2_final.xlsx"
    return TagSeedDictionary.from_xlsx(xlsx_path)


def load_amazon_sample(n=1000):
    """加载 Amazon VOC 样本"""
    csv_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv"
    df = pd.read_csv(csv_path, nrows=n)
    vocs = []
    for _, row in df.iterrows():
        text = str(row.get("Content", "") or row.get("English Content", ""))
        if not text or text == "nan":
            continue
        rating = row.get("Rating")
        try:
            rating = float(rating) if pd.notna(rating) else None
        except (ValueError, TypeError):
            rating = None
        vocs.append(VOCRecord(
            review_id=str(row.get("Asin", "")) + "_" + str(row.get("Author", "")) + "_" + str(_),
            text=text,
            source_type="review",
            platform="amazon",
            spu_code=str(row.get("Asin", "")),
            product_line="breast_pump",
            category="unknown",
            rating=rating,
        ))
    return vocs


class DiagnosticExtractor(VOCLabelExtractor):
    """扩展萃取器，收集诊断数据 — 使用宽松匹配策略最大化覆盖率"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics: list[dict] = []
        # 匹配详情：每条VOC每个标签的匹配信息
        self.match_details: list[dict] = []

    def _match_aipl_tags(self, text_lower, candidate_tags):
        """宽松匹配：全部子串匹配，不过滤停用词，最大化覆盖率用于诊断"""
        from unified_label_extraction import AIPLTagMatch

        matches = []
        for tag in candidate_tags:
            all_keywords = tag.keywords + tag.consumer_keywords
            matched = False
            match_kw = ""

            for kw in all_keywords:
                kw_lower = kw.lower()
                # 宽松策略：全部用子串匹配，不过滤任何关键词
                if kw_lower in text_lower:
                    # 检查否定词
                    idx = text_lower.find(kw_lower)
                    prefix = text_lower[max(0, idx - 15):idx]
                    negators = {"not", "no", "never", "none", "nobody", "nothing",
                                "neither", "nowhere", "hardly", "barely", "rarely",
                                "without", "dont", "doesnt", "didnt", "wont",
                                "wouldnt", "cant", "cannot", "isnt", "arent",
                                "wasnt", "werent", "hasnt", "havent", "hadnt"}
                    if any(neg in prefix for neg in negators):
                        continue
                    matched = True
                    match_kw = kw_lower
                    break

            if matched:
                confidence = min(len(match_kw) / 20.0, 1.0)
                sentiment = self.sentiment_calibrator.calibrate(
                    text_lower, tag.sentiment_preset, None
                )
                matches.append(AIPLTagMatch(
                    tag_id=tag.tag_id,
                    tag_en=tag.tag_en,
                    tag_cn=tag.tag_cn,
                    theme=tag.theme,
                    aipl_node=tag.aipl_node,
                    sentiment_preset=tag.sentiment_preset,
                    sentiment_calibrated=sentiment.polarity,
                    confidence=confidence,
                ))
                # 记录匹配详情
                self.match_details.append({
                    "tag_id": tag.tag_id,
                    "tag_en": tag.tag_en,
                    "matched_keyword": match_kw,
                    "keyword_source": "keywords" if match_kw in [k.lower() for k in tag.keywords] else "consumer",
                })

        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def extract(self, voc):
        result = super().extract(voc)

        # 收集诊断
        diag = {
            "review_id": voc.review_id,
            "text_preview": voc.text[:200],
            "rating": voc.rating,
            "n_tags": len(result.aipl_tags),
            "tags": [t.tag_en for t in result.aipl_tags],
            "aipl_stage": result.aipl_stage,
            "sentiment_polarity": result.sentiment_polarity,
            "proxy_nps": result.proxy_nps_contribution,
            "is_conflict": result.sentiment_calibration == "conflict",
            "persona": result.persona_derived,
        }
        self.diagnostics.append(diag)
        return result


def run_diagnostics(n=1000):
    """运行诊断并输出报告"""
    print("=" * 70)
    print(f"标签质量诊断 — {n} 条 Amazon VOC 样本")
    print("=" * 70)

    # 1. 加载标签字典
    print("\n[1/5] 加载标签字典...")
    tag_dict = load_tag_dict()
    all_tags = tag_dict.get_all()
    print(f"  总标签数: {len(all_tags)}")

    # 统计关键词信息
    kw_stats = Counter()
    single_word_tags = []
    multi_word_tags = []
    for tag in all_tags:
        kws = tag.keywords + tag.consumer_keywords
        kw_stats[len(kws)] += 1
        if kws:
            has_single = any(len(kw.split()) == 1 for kw in kws)
            if has_single:
                single_word_tags.append(tag.tag_en)
            else:
                multi_word_tags.append(tag.tag_en)

    print(f"  关键词分布: {dict(kw_stats.most_common())}")
    print(f"  含单字/单词关键词的标签: {len(single_word_tags)} 个")
    print(f"  仅多词关键词的标签: {len(multi_word_tags)} 个")

    # 2. 加载 VOC 样本
    print(f"\n[2/5] 加载 {n} 条 VOC 样本...")
    vocs = load_amazon_sample(n)
    print(f"  有效样本: {len(vocs)} 条")

    # 3. 运行萃取（诊断模式）
    print("\n[3/5] 运行标签萃取...")
    extractor = DiagnosticExtractor(tag_dict=tag_dict)
    for voc in vocs:
        extractor.extract(voc)

    # 4. 分析诊断数据
    print("\n[4/5] 分析诊断数据...")
    diags = extractor.diagnostics

    # 覆盖率
    has_tags = sum(1 for d in diags if d["n_tags"] > 0)
    coverage = has_tags / len(diags) * 100

    # 平均标签数
    avg_tags = sum(d["n_tags"] for d in diags) / len(diags)

    # 冲突率
    conflicts = sum(1 for d in diags if d["is_conflict"])
    conflict_rate = conflicts / len(diags) * 100

    # AIPL 分布
    aipl_counts = Counter(d["aipl_stage"] for d in diags)

    # 标签命中分布
    tag_hits = Counter()
    for d in diags:
        for t in d["tags"]:
            tag_hits[t] += 1

    # 冲突按星级分布
    conflict_by_rating = defaultdict(int)
    for d in diags:
        if d["is_conflict"]:
            conflict_by_rating[d["rating"]] += 1

    # 高频冲突标签 — 从冲突 VOC 的标签中统计
    conflict_voc_tags = Counter()
    for d in diags:
        if d["is_conflict"]:
            for t in d["tags"]:
                conflict_voc_tags[t] += 1
    top_conflict_tags = conflict_voc_tags.most_common(20)

    # 高命中标签（检查是否有标签过度匹配）
    top_hit_tags = tag_hits.most_common(30)

    # NPS 分布
    nps_dist = Counter(d["proxy_nps"] for d in diags)

    # 画像分布
    persona_dist = Counter(d["persona"] for d in diags if d["persona"])

    # 5. 输出报告
    print("\n" + "=" * 70)
    print("诊断报告")
    print("=" * 70)

    print(f"\n--- 整体指标 ---")
    print(f"  样本数: {len(diags)}")
    print(f"  覆盖率(至少1个标签): {has_tags}/{len(diags)} ({coverage:.1f}%)")
    print(f"  平均标签数/VOC: {avg_tags:.2f}")
    print(f"  冲突数: {conflicts}/{len(diags)} ({conflict_rate:.1f}%)")

    print(f"\n--- AIPL 阶段分布 ---")
    for stage, cnt in sorted(aipl_counts.items(), key=lambda x: -x[1]):
        print(f"  {stage}: {cnt} ({cnt/len(diags)*100:.1f}%)")

    print(f"\n--- Proxy NPS 分布 ---")
    for nps, cnt in nps_dist.most_common():
        print(f"  {nps}: {cnt} ({cnt/len(diags)*100:.1f}%)")

    print(f"\n--- Top 30 命中标签 ---")
    for tag, cnt in top_hit_tags:
        print(f"  {tag}: {cnt} 次")

    print(f"\n--- Top 20 冲突标签 ---")
    for tag, cnt in top_conflict_tags:
        print(f"  {tag}: {cnt} 次")

    print(f"\n--- 冲突按星级分布 ---")
    for rating in sorted(conflict_by_rating.keys()):
        cnt = conflict_by_rating[rating]
        print(f"  星级 {rating}: {cnt} 次冲突")

    print(f"\n--- 画像分布 ---")
    for persona, cnt in persona_dist.most_common():
        print(f"  {persona}: {cnt} 次")

    # 6. 零命中标签（潜在覆盖缺口）
    all_tag_names = {t.tag_en for t in all_tags}
    hit_tag_names = set(tag_hits.keys())
    zero_hit = all_tag_names - hit_tag_names
    print(f"\n--- 零命中标签 ({len(zero_hit)} 个) ---")
    for tag in sorted(zero_hit):
        print(f"  {tag}")

    # 7. 保存详细诊断数据
    print("\n[5/5] 保存诊断数据...")
    output_dir = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果"
    report = {
        "sample_size": len(diags),
        "total_tags": len(all_tags),
        "coverage": {"has_tags": has_tags, "total": len(diags), "rate": coverage},
        "avg_tags_per_voc": avg_tags,
        "conflict": {"count": conflicts, "rate": conflict_rate},
        "aipl_distribution": dict(aipl_counts),
        "nps_distribution": dict(nps_dist),
        "top_hit_tags": top_hit_tags,
        "top_conflict_tags": top_conflict_tags,
        "zero_hit_tags": sorted(zero_hit),
        "persona_distribution": dict(persona_dist),
        "single_word_tags_count": len(single_word_tags),
        "multi_word_tags_count": len(multi_word_tags),
    }

    report_path = f"{output_dir}/tag_diagnostic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}")

    # 保存逐条诊断（用于人工复核）
    detail_path = f"{output_dir}/tag_diagnostic_details.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(diags, f, ensure_ascii=False, indent=2)
    print(f"  详情已保存: {detail_path}")

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_diagnostics(n=1000)

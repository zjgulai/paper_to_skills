"""v3.3 打标结果转录（Phase 1.3 - Step 1）

将 v3.3 已打标数据（Amazon 200k + Trustpilot 10k + Reddit 6.2k + Zendesk 48k）
转录为统一格式，与 unified high-quality VOC 记录合并。

匹配策略：
- 从 unified review_id 提取 raw_id（ASIN/content_id/工单号）
- 用 (data_source, raw_id, text[:120]) 作为匹配键
- v3.3 text_preview 截断 200 字符，统一用前 120 字符匹配
"""

import json
from collections import defaultdict
from pathlib import Path


def extract_raw_id(review_id: str, data_source: str) -> str:
    """从 unified review_id 提取 v3.3 格式的 raw_id"""
    if data_source == "amazon_competitor":
        # amz_B01KG84CLI_0 -> B01KG84CLI
        parts = review_id.split("_")
        if len(parts) >= 2 and parts[0] == "amz":
            return "_".join(parts[1:-1])  # handle ASINs with underscores
        return review_id
    elif data_source == "trustpilot":
        # tp_6345ebb09b12f67dfede8bc8_0 -> 6345ebb09b12f67dfede8bc8
        parts = review_id.split("_")
        if len(parts) >= 3 and parts[0] == "tp":
            return "_".join(parts[1:-1])
        return review_id
    elif data_source == "reddit":
        # rd_xxx_0 -> xxx
        parts = review_id.split("_")
        if len(parts) >= 3 and parts[0] == "rd":
            return "_".join(parts[1:-1])
        return review_id
    elif data_source == "zendesk":
        # zd_xxx_0 -> xxx
        parts = review_id.split("_")
        if len(parts) >= 3 and parts[0] == "zd":
            return "_".join(parts[1:-1])
        return review_id
    return review_id


def make_match_key(data_source: str, raw_id: str, text: str) -> str:
    """生成匹配键"""
    text_norm = text[:120].lower().replace(" ", "").replace("\n", "")
    return f"{data_source}::{raw_id}::{text_norm}"


def load_v33_labeled() -> dict[str, dict]:
    """加载全部 v3.3 打标结果，建立匹配索引"""
    v33_dir = Path(__file__).parent.parent.parent / "00-归档资料/labeling-outputs/v3.3"

    # 数据源映射
    source_batches = {
        "amazon_competitor": (v33_dir / "amazon", "batch_*.jsonl"),
        "trustpilot": (v33_dir / "trustpilot", "batch_*.jsonl"),
        "reddit": (v33_dir / "reddit", "batch_*.jsonl"),
        "zendesk": (v33_dir / "zendesk", "batch_*.jsonl"),
    }

    index = {}  # match_key -> v33 record
    stats = defaultdict(int)

    for ds_name, (dir_path, pattern) in source_batches.items():
        batch_files = sorted(dir_path.glob(pattern))
        print(f"\n  加载 {ds_name}: {len(batch_files)} batches")

        for batch_file in batch_files:
            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line.strip())
                    raw_id = rec["review_id"]
                    text = rec.get("text_preview", "")
                    key = make_match_key(ds_name, raw_id, text)

                    if key in index:
                        # 同一个 key 出现多次（同一 ASIN 多条相似评论）
                        # 保留第一个，记录冲突
                        stats[f"{ds_name}_dup_keys"] += 1
                        continue

                    index[key] = rec
                    stats[f"{ds_name}_loaded"] += 1

    print("\n  v3.3 加载统计:")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v:,}")

    return index


def transcribe_v33_to_unified(v33_rec: dict, uni_rec: dict) -> dict:
    """将 v3.3 格式转录为统一输出格式"""
    return {
        "review_id": uni_rec["review_id"],
        "text": uni_rec["text"],
        "source_type": uni_rec["source_type"],
        "platform": uni_rec["platform"],
        "spu_code": uni_rec.get("spu_code"),
        "asin": uni_rec.get("asin"),
        "product_line": uni_rec.get("product_line"),
        "category": uni_rec.get("category"),
        "rating": uni_rec.get("rating"),
        "language": uni_rec.get("language", "en"),
        "data_source": uni_rec["data_source"],
        "_quality_score": uni_rec.get("_quality_score"),

        # 标签结果（转录自 v3.3）
        "labels": [
            {
                "tag_id": t["tag_id"],
                "tag_en": t["tag_en"],
                "tag_cn": t["tag_cn"],
                "aipl_node": t["aipl_node"],
                "sentiment_preset": t["sentiment_preset"],
                "sentiment_calibrated": t["sentiment_calibrated"],
                "confidence": t["confidence"],
            }
            for t in v33_rec.get("aipl_tags", [])
        ],
        "n_tags": v33_rec.get("n_tags", 0),
        "aipl_stage": v33_rec.get("aipl_stage", ""),
        "persona_derived": v33_rec.get("persona_derived", ""),
        "sentiment_polarity": v33_rec.get("sentiment_polarity", 0.0),
        "sentiment_calibration": v33_rec.get("sentiment_calibration", ""),
        "proxy_nps": v33_rec.get("proxy_nps", ""),
        "brand_mentions": v33_rec.get("brand_mentions", []),
        "brand_comparison": v33_rec.get("brand_comparison", False),

        # 标记来源
        "label_source": "v3.3_transcribed",
    }


def main():
    print("=" * 70)
    print("Phase 1.3 Step 1: v3.3 打标结果转录")
    print("=" * 70)

    # 1. 加载 v3.3 索引
    print("\n--- 加载 v3.3 打标结果 ---")
    v33_index = load_v33_labeled()
    print(f"  总索引数: {len(v33_index):,}")

    # 2. 遍历 unified high-quality 记录，匹配 v3.3
    hq_path = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling/phase1_2_high_quality_voc.jsonl"
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- 匹配转录 ---")
    print(f"  输入: {hq_path}")

    transcribed = []
    unmatched = []
    stats = defaultdict(int)

    with open(hq_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            ds = rec["data_source"]
            stats[f"total_{ds}"] += 1

            if ds == "momcozy":
                # Momcozy 无 v3.3 打标，留待增量打标
                stats["momcozy_skipped"] += 1
                continue

            raw_id = extract_raw_id(rec["review_id"], ds)
            key = make_match_key(ds, raw_id, rec["text"])

            if key in v33_index:
                v33_rec = v33_index[key]
                unified_rec = transcribe_v33_to_unified(v33_rec, rec)
                transcribed.append(unified_rec)
                stats[f"matched_{ds}"] += 1
            else:
                # 未匹配：可能是 quality filter 过滤掉的，或文本差异
                unmatched.append(rec)
                stats[f"unmatched_{ds}"] += 1

    # 3. 输出统计
    print(f"\n--- 转录统计 ---")
    for ds in ["amazon_competitor", "trustpilot", "reddit", "zendesk", "momcozy"]:
        total = stats.get(f"total_{ds}", 0)
        matched = stats.get(f"matched_{ds}", 0)
        unmatched_cnt = stats.get(f"unmatched_{ds}", 0)
        skipped = stats.get(f"{ds}_skipped", 0)
        if total > 0:
            print(f"  {ds}: 总计 {total:,}, 匹配 {matched:,} ({matched/total*100:.1f}%), 未匹配 {unmatched_cnt:,}, 跳过 {skipped:,}")

    print(f"\n  转录总计: {len(transcribed):,} 条")
    print(f"  未匹配: {len(unmatched):,} 条")

    # 4. 保存转录结果
    output_path = output_dir / "phase1_3_v33_transcribed.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in transcribed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  输出: {output_path}")

    # 5. 保存未匹配样本（用于分析）
    if unmatched:
        unmatched_path = output_dir / "phase1_3_unmatched_samples.jsonl"
        with open(unmatched_path, "w", encoding="utf-8") as f:
            for r in unmatched[:1000]:  # 最多保存 1000 条
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  未匹配样本: {unmatched_path} ({min(len(unmatched), 1000):,} 条)")

    # 6. 审计报告
    audit = {
        "phase": "1.3-step1",
        "description": "v3.3 打标结果转录",
        "total_hq_records": sum(stats.get(f"total_{ds}", 0) for ds in ["amazon_competitor", "trustpilot", "reddit", "zendesk", "momcozy"]),
        "transcribed_count": len(transcribed),
        "unmatched_count": len(unmatched),
        "momcozy_to_label": stats.get("momcozy_skipped", 0),
        "source_breakdown": {
            ds: {
                "total": stats.get(f"total_{ds}", 0),
                "matched": stats.get(f"matched_{ds}", 0),
                "unmatched": stats.get(f"unmatched_{ds}", 0),
            }
            for ds in ["amazon_competitor", "trustpilot", "reddit", "zendesk"]
            if stats.get(f"total_{ds}", 0) > 0
        },
        "output_path": str(output_path),
    }
    audit_path = output_dir / "phase1_3_step1_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"  审计: {audit_path}")

    # 7. 标签覆盖率统计
    print(f"\n--- 标签覆盖统计 ---")
    tag_counts = defaultdict(int)
    zero_tag = 0
    for r in transcribed:
        n = r["n_tags"]
        if n == 0:
            zero_tag += 1
        tag_counts[n] += 1

    print(f"  零标签: {zero_tag:,} / {len(transcribed):,} ({zero_tag/len(transcribed)*100:.1f}%)")
    print(f"  标签数分布:")
    for n in sorted(tag_counts.keys())[:10]:
        print(f"    {n} 个标签: {tag_counts[n]:,} ({tag_counts[n]/len(transcribed)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Phase 1.3 Step 1 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""VOC 大规模分批自动打标

对 4 个数据源分别运行适配标签子集的批量打标，
每批 10000 条，输出 JSON Lines 格式中间结果，支持断点续跑。

Usage:
    python batch_labeling.py --source amazon --batch-size 10000
    python batch_labeling.py --source all
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, "/Users/pray/project/paper_to_skills/paper2skills-code/nlp_voc/proxy_nps_aipl_workflow")

from unified_label_extraction import (
    TagSeedDictionary,
    UnifiedLabelingPipeline,
    VOCRecord,
)


# ── 数据源配置 ────────────────────────────────────────────────────

@dataclass
class SourceConfig:
    name: str
    csv_path: str
    text_col: str
    rating_col: Optional[str]
    id_col: str
    spu_col: str
    source_type: str
    platform: str
    batch_size: int = 10000


SOURCE_CONFIGS: dict[str, SourceConfig] = {
    "amazon": SourceConfig(
        name="amazon",
        csv_path="/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv",
        text_col="Content",
        rating_col="Rating",
        id_col="Asin",
        spu_col="Asin",
        source_type="review",
        platform="amazon",
        batch_size=10000,
    ),
    "trustpilot": SourceConfig(
        name="trustpilot",
        csv_path="/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/trustpilot_voc_100k_balanced.csv",
        text_col="_review_text",
        rating_col="rating",
        id_col="review_id",
        spu_col="domain",
        source_type="trustpilot",
        platform="trustpilot",
        batch_size=10000,
    ),
    "reddit": SourceConfig(
        name="reddit",
        csv_path="/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/reddit_voc_sampled.csv",
        text_col="analysis_text",
        rating_col=None,
        id_col="data_id",
        spu_col="_spu",
        source_type="review",
        platform="reddit",
        batch_size=10000,
    ),
    "zendesk": SourceConfig(
        name="zendesk",
        csv_path="/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/zendesk_momcozy_voc_sampled.csv",
        text_col="_review_text",
        rating_col="星级评分",
        id_col="工单号",
        spu_col="SPU名称",
        source_type="ticket",
        platform="zendesk",
        batch_size=10000,
    ),
}


TAG_DICT_PATH = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest/SGCS_VOC标签字典_V3.3.1_universal.xlsx"
OUTPUT_BASE = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest"


# ── 核心处理 ──────────────────────────────────────────────────────

def load_tag_dict(source_type: str) -> TagSeedDictionary:
    """加载标签字典并按数据源筛选"""
    tag_dict = TagSeedDictionary.from_xlsx(TAG_DICT_PATH)
    filtered = tag_dict.filter_by_source(source_type)
    print(f"  标签字典: {len(tag_dict.get_all())} → 筛选后 {len(filtered)} 条")
    return tag_dict


def row_to_voc(row: pd.Series, cfg: SourceConfig) -> Optional[VOCRecord]:
    """将 DataFrame 行转换为 VOCRecord"""
    text = str(row.get(cfg.text_col, ""))
    if not text or text.lower() in ("nan", "none", ""):
        return None

    review_id = str(row.get(cfg.id_col, ""))
    if not review_id:
        review_id = f"{cfg.name}_{hash(text) & 0xFFFFFFFF}"

    rating = None
    if cfg.rating_col:
        raw = row.get(cfg.rating_col)
        if pd.notna(raw):
            try:
                rating = float(raw)
                # 星级评分 0 视为缺失
                if rating == 0:
                    rating = None
            except (ValueError, TypeError):
                pass

    spu = str(row.get(cfg.spu_col, "")) if cfg.spu_col else ""
    if not spu or spu.lower() == "nan":
        spu = "unknown"

    return VOCRecord(
        review_id=review_id,
        text=text,
        source_type=cfg.source_type,
        platform=cfg.platform,
        spu_code=spu,
        product_line="breast_pump",  # 默认品线
        category="unknown",
        rating=rating,
    )


def process_batch(
    pipeline: UnifiedLabelingPipeline,
    vocs: list[VOCRecord],
) -> list[dict[str, Any]]:
    """处理一批 VOC 记录"""
    results = pipeline.process(vocs)
    output = []
    for voc, result in zip(vocs, results):
        output.append({
            "review_id": result.review_id,
            "text_preview": voc.text[:200] if voc.text else "",
            "rating": result.rating,
            "aipl_stage": result.aipl_stage,
            "aipl_tags": [
                {
                    "tag_id": t.tag_id,
                    "tag_en": t.tag_en,
                    "tag_cn": t.tag_cn,
                    "aipl_node": t.aipl_node,
                    "sentiment_preset": t.sentiment_preset,
                    "sentiment_calibrated": t.sentiment_calibrated,
                    "confidence": round(t.confidence, 3),
                }
                for t in result.aipl_tags
            ],
            "n_tags": len(result.aipl_tags),
            "persona_derived": result.persona_derived,
            "sentiment_polarity": round(result.sentiment_polarity, 2),
            "sentiment_calibration": result.sentiment_calibration,
            "proxy_nps": result.proxy_nps_contribution,
            "brand_mentions": result.brand_mentions,
            "brand_comparison": result.brand_comparison,
            "quality_score": round(result.quality_score, 1) if result.quality_score else None,
            "is_suspicious": result.is_suspicious,
        })
    return output


def label_source(cfg: SourceConfig, output_dir: str, resume: bool = True):
    """对单个数据源执行分批打标"""
    print("=" * 70)
    print(f"数据源: {cfg.name.upper()}")
    print(f"  输入: {cfg.csv_path}")
    print(f"  批次大小: {cfg.batch_size}")
    print("=" * 70)

    # 加载标签字典
    print("\n[1/3] 加载标签字典...")
    tag_dict = load_tag_dict(cfg.source_type)

    # 创建流水线
    pipeline = UnifiedLabelingPipeline(tag_dict=tag_dict)

    # 准备输出目录
    source_out = Path(output_dir) / cfg.name
    source_out.mkdir(parents=True, exist_ok=True)

    # 检查断点
    existing_batches = sorted(source_out.glob("batch_*.jsonl"))
    start_batch = len(existing_batches) if resume else 0
    if start_batch > 0:
        print(f"  发现 {start_batch} 个已完成批次，断点续跑")

    # 统计总行数
    total_rows = sum(1 for _ in open(cfg.csv_path, "r", encoding="utf-8")) - 1
    print(f"\n[2/3] 总记录: {total_rows:,} 条")

    # 分批次处理
    print(f"\n[3/3] 开始打标 (从 batch_{start_batch:04d} 开始)...")
    batch_idx = start_batch
    total_processed = 0
    total_labeled = 0
    total_conflicts = 0
    tag_distribution: dict[str, int] = {}

    chunk_iter = pd.read_csv(cfg.csv_path, chunksize=cfg.batch_size, dtype=str)
    for chunk in chunk_iter:
        if batch_idx < start_batch:
            batch_idx += 1
            continue

        t0 = time.time()

        # 转换为 VOCRecord
        vocs = []
        for _, row in chunk.iterrows():
            voc = row_to_voc(row, cfg)
            if voc:
                vocs.append(voc)

        if not vocs:
            batch_idx += 1
            continue

        # 打标
        results = process_batch(pipeline, vocs)

        # 保存
        batch_path = source_out / f"batch_{batch_idx:04d}.jsonl"
        with open(batch_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 统计
        labeled = sum(1 for r in results if r["n_tags"] > 0)
        conflicts = sum(1 for r in results if r["sentiment_calibration"] == "conflict")
        for r in results:
            for t in r["aipl_tags"]:
                tag_distribution[t["tag_en"]] = tag_distribution.get(t["tag_en"], 0) + 1

        total_processed += len(results)
        total_labeled += labeled
        total_conflicts += conflicts

        elapsed = time.time() - t0
        print(f"  batch_{batch_idx:04d}: {len(results):,} 条 | "
              f"标签命中 {labeled:,} | 冲突 {conflicts} | "
              f"耗时 {elapsed:.1f}s | 累计 {total_processed:,}/{total_rows:,}")

        batch_idx += 1

    # 保存汇总
    summary = {
        "source": cfg.name,
        "total_processed": total_processed,
        "total_labeled": total_labeled,
        "coverage_rate": round(total_labeled / total_processed * 100, 1) if total_processed else 0,
        "total_conflicts": total_conflicts,
        "conflict_rate": round(total_conflicts / total_processed * 100, 1) if total_processed else 0,
        "tag_distribution": dict(sorted(tag_distribution.items(), key=lambda x: -x[1])),
        "batches": batch_idx,
    }

    summary_path = source_out / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"{cfg.name.upper()} 打标完成")
    print(f"  处理: {total_processed:,} 条")
    print(f"  标签命中: {total_labeled:,} ({summary['coverage_rate']:.1f}%)")
    print(f"  冲突: {total_conflicts} ({summary['conflict_rate']:.1f}%)")
    print(f"  批次: {batch_idx}")
    print(f"  输出: {source_out}")
    print(f"{'=' * 70}")

    return summary


# ── 入口 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VOC 大规模分批自动打标")
    parser.add_argument("--source", default="all", help="数据源: amazon|trustpilot|reddit|zendesk|all")
    parser.add_argument("--batch-size", type=int, default=10000, help="每批处理条数")
    parser.add_argument("--output", default=OUTPUT_BASE, help="输出目录")
    parser.add_argument("--no-resume", action="store_true", help="不续跑，从头开始")
    args = parser.parse_args()

    sources = list(SOURCE_CONFIGS.keys()) if args.source == "all" else [args.source]

    all_summaries = {}
    for src_name in sources:
        if src_name not in SOURCE_CONFIGS:
            print(f"未知数据源: {src_name}")
            continue

        cfg = SOURCE_CONFIGS[src_name]
        cfg.batch_size = args.batch_size

        summary = label_source(cfg, args.output, resume=not args.no_resume)
        all_summaries[src_name] = summary

    # 全局汇总
    if len(sources) > 1:
        print("\n" + "=" * 70)
        print("全局汇总")
        print("=" * 70)
        total_all = sum(s["total_processed"] for s in all_summaries.values())
        labeled_all = sum(s["total_labeled"] for s in all_summaries.values())
        conflicts_all = sum(s["total_conflicts"] for s in all_summaries.values())
        print(f"  总处理: {total_all:,} 条")
        print(f"  总标签命中: {labeled_all:,}")
        print(f"  总冲突: {conflicts_all}")
        print(f"  平均覆盖率: {labeled_all/total_all*100:.1f}%")
        print(f"  平均冲突率: {conflicts_all/total_all*100:.1f}%")

        global_summary = {
            "total_processed": total_all,
            "total_labeled": labeled_all,
            "total_conflicts": conflicts_all,
            "sources": all_summaries,
        }
        global_path = Path(args.output) / "global_summary.json"
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)
        print(f"  全局汇总已保存: {global_path}")


if __name__ == "__main__":
    main()

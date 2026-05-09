"""统一 VOCRecord 输入格式

将各数据源映射到标准 VOCRecord 格式，建立统一输入层。
"""

import json
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class VOCRecord:
    review_id: str          # 评论唯一标识（全局唯一）
    text: str               # VOC 文本
    source_type: str        # review / ticket / trustpilot / reddit
    platform: str           # amazon / dtc / trustpilot / reddit / zendesk
    spu_code: Optional[str] # SPU 编码（Momcozy 数据有，竞品无）
    asin: Optional[str]     # ASIN（Amazon 数据有）
    product_line: Optional[str]  # VOC 品线
    category: Optional[str]      # VOC 品类（四级类目）
    rating: Optional[float]      # 1-5 星评分
    language: str = "en"    # 语言标识
    data_source: str = ""   # 原始数据源标识
    _quality_score: Optional[float] = None  # 质量分（如有）
    _is_high_quality: Optional[bool] = None  # 是否高质量（如有）


# ── 产品主数据映射 ────────────────────────────────────────────────

def load_product_master() -> dict:
    """加载 SPU → 品线/品类映射"""
    path = Path(__file__).parent.parent.parent / "03-数据资产/产品主数据/产品主数据_VOC维度关联.xlsx"
    df = pd.read_excel(path, sheet_name="Sheet1")
    mapping = {}
    for _, row in df.iterrows():
        spu_code = str(row.get("SPU编码", "")).strip()
        if spu_code:
            mapping[spu_code] = {
                "spu_name": str(row.get("SPU名称", "")),
                "product_line": str(row.get("VOC品线", "")),
                "category": str(row.get("VOC品类", "")),
            }
    print(f"  产品主数据映射: {len(mapping)} SPU")
    return mapping


# ── 各数据源转换器 ────────────────────────────────────────────────

def convert_amazon(df: pd.DataFrame) -> list[VOCRecord]:
    """Amazon 竞品数据转换"""
    records = []
    for idx, row in df.iterrows():
        # 生成唯一 review_id（避免 ASIN 冲突）
        raw_id = str(row.get("Asin", ""))
        unique_id = f"amz_{raw_id}_{idx}"

        # 文本优先 English Content，否则 Content
        text = str(row.get("English Content", "")).strip()
        if not text or text == "nan":
            text = str(row.get("Content", "")).strip()

        # 语言推断（English Content 非空 → en，否则基于内容推断）
        lang = "en"  # Amazon 数据基本为英文

        records.append(VOCRecord(
            review_id=unique_id,
            text=text,
            source_type="review",
            platform="amazon",
            spu_code=None,
            asin=raw_id if raw_id != "nan" else None,
            product_line=None,  # 后续推断
            category=None,      # 后续推断
            rating=float(row["Rating"]) if pd.notna(row.get("Rating")) else None,
            language=lang,
            data_source="amazon_competitor",
            _quality_score=float(row["_quality_score"]) if "_quality_score" in row and pd.notna(row["_quality_score"]) else None,
            _is_high_quality=bool(row["_is_high_quality"]) if "_is_high_quality" in row and pd.notna(row["_is_high_quality"]) else None,
        ))
    return records


def convert_momcozy(df: pd.DataFrame, spu_mapping: dict) -> list[VOCRecord]:
    """Momcozy 自有数据转换"""
    records = []
    for idx, row in df.iterrows():
        spu_code = str(row.get("SPU编码", "")).strip()
        asin = str(row.get("ASIN", "")).strip()
        raw_id = str(row.get("工单号", "")).strip() or asin
        unique_id = f"mz_{raw_id}_{idx}" if raw_id else f"mz_idx_{idx}"

        # 文本
        text = str(row.get("_review_text", "")).strip()
        if not text or text == "nan":
            text = str(row.get("买家评论", "")).strip()

        # 映射品线/品类
        mapped = spu_mapping.get(spu_code, {})
        product_line = mapped.get("product_line") if mapped else None
        category = mapped.get("category") if mapped else None
        # 兜底：用四级类目名称
        if not category:
            category = str(row.get("四级类目名称", "")).strip() or None

        # 平台名称映射
        platform_raw = str(row.get("平台名称", "")).strip()
        platform_map = {"亚马逊": "amazon", "独立站": "dtc", "国内": "domestic"}
        platform = platform_map.get(platform_raw, platform_raw.lower())

        # 来源类型映射
        source_raw = str(row.get("VOC来源", "")).strip()
        source_map = {"商品评论": "review", "退货留言": "return_note", "客服工单": "ticket"}
        source_type = source_map.get(source_raw, source_raw.lower())

        # 语言（Momcozy 评论基本为英文，但有其他语言混入）
        lang = "en"  # 简化为英文，后续可用 langdetect

        records.append(VOCRecord(
            review_id=unique_id,
            text=text,
            source_type=source_type,
            platform=platform,
            spu_code=spu_code if spu_code != "nan" else None,
            asin=asin if asin != "nan" else None,
            product_line=product_line if product_line and product_line != "nan" else None,
            category=category if category and category != "nan" else None,
            rating=float(row["星级评分"]) if pd.notna(row.get("星级评分")) else None,
            language=lang,
            data_source="momcozy",
            _quality_score=float(row["_quality_score"]) if "_quality_score" in row and pd.notna(row["_quality_score"]) else None,
            _is_high_quality=bool(row["_is_high_quality"]) if "_is_high_quality" in row and pd.notna(row["_is_high_quality"]) else None,
        ))
    return records


def convert_trustpilot(df: pd.DataFrame) -> list[VOCRecord]:
    """Trustpilot 数据转换"""
    records = []
    for idx, row in df.iterrows():
        raw_id = str(row.get("review_id", "")).strip()
        unique_id = f"tp_{raw_id}_{idx}" if raw_id else f"tp_idx_{idx}"

        text = str(row.get("_review_text", "")).strip()
        if not text or text == "nan":
            text = str(row.get("review_body", "")).strip()

        # 国家映射到语言
        country = str(row.get("author_country", "")).strip()
        lang_map = {"de": "de", "fr": "fr", "es": "es", "it": "it", "pl": "pl"}
        lang = lang_map.get(country.lower(), "en")

        records.append(VOCRecord(
            review_id=unique_id,
            text=text,
            source_type="trustpilot",
            platform="trustpilot",
            spu_code=None,
            asin=None,
            product_line=None,
            category=None,
            rating=float(row["rating"]) if pd.notna(row.get("rating")) else None,
            language=lang,
            data_source="trustpilot",
            _quality_score=float(row["_quality_score"]) if "_quality_score" in row and pd.notna(row["_quality_score"]) else None,
            _is_high_quality=bool(row["_is_high_quality"]) if "_is_high_quality" in row and pd.notna(row["_is_high_quality"]) else None,
        ))
    return records


def convert_reddit(df: pd.DataFrame) -> list[VOCRecord]:
    """Reddit 数据转换"""
    records = []
    for idx, row in df.iterrows():
        raw_id = str(row.get("content_id", "")).strip()
        unique_id = f"rd_{raw_id}_{idx}" if raw_id else f"rd_idx_{idx}"

        text = str(row.get("analysis_text", "")).strip()
        if not text or text == "nan":
            text = str(row.get("content_title", "")).strip()

        # Reddit 无评分
        records.append(VOCRecord(
            review_id=unique_id,
            text=text,
            source_type="reddit",
            platform="reddit",
            spu_code=None,
            asin=None,
            product_line=None,
            category=None,
            rating=None,
            language="en",
            data_source="reddit",
            _quality_score=float(row["_quality_score"]) if "_quality_score" in row and pd.notna(row["_quality_score"]) else None,
            _is_high_quality=bool(row["_is_high_quality"]) if "_is_high_quality" in row and pd.notna(row["_is_high_quality"]) else None,
        ))
    return records


def convert_zendesk(df: pd.DataFrame, spu_mapping: dict) -> list[VOCRecord]:
    """Zendesk 数据转换"""
    records = []
    for idx, row in df.iterrows():
        raw_id = str(row.get("工单号", "")).strip()
        unique_id = f"zd_{raw_id}_{idx}" if raw_id else f"zd_idx_{idx}"

        text = str(row.get("_review_text", "")).strip()
        if not text or text == "nan":
            text = str(row.get("工单客户原文", "")).strip()
        if not text or text == "nan":
            text = str(row.get("买家评论", "")).strip()

        # 映射品线/品类
        spu_code = str(row.get("SPU编码", "")).strip()
        mapped = spu_mapping.get(spu_code, {})
        product_line = mapped.get("product_line") if mapped else None
        category = mapped.get("category") if mapped else None
        if not category:
            category = str(row.get("四级类目名称", "")).strip() or None

        records.append(VOCRecord(
            review_id=unique_id,
            text=text,
            source_type="ticket",
            platform="zendesk",
            spu_code=spu_code if spu_code != "nan" else None,
            asin=None,
            product_line=product_line if product_line and product_line != "nan" else None,
            category=category if category and category != "nan" else None,
            rating=float(row["星级评分"]) if pd.notna(row.get("星级评分")) else None,
            language="en",
            data_source="zendesk",
            _quality_score=float(row["_quality_score"]) if "_quality_score" in row and pd.notna(row["_quality_score"]) else None,
            _is_high_quality=bool(row["_is_high_quality"]) if "_is_high_quality" in row and pd.notna(row["_is_high_quality"]) else None,
        ))
    return records


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent.parent.parent / "03-数据资产/高质量数据源"
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1.1: 统一 VOCRecord 输入格式")
    print("=" * 70)

    # 1. 加载产品主数据映射
    print("\n--- 加载产品主数据映射 ---")
    spu_mapping = load_product_master()

    # 2. 逐数据源转换
    all_records = []
    stats = {}

    # Amazon
    print("\n--- Amazon 竞品数据 ---")
    amz_path = base_dir / "amazon_voc_200k_balanced.csv"
    amz_df = pd.read_csv(amz_path)
    amz_records = convert_amazon(amz_df)
    all_records.extend(amz_records)
    stats["amazon"] = len(amz_records)
    print(f"  转换: {len(amz_records):,} 条")

    # Momcozy
    print("\n--- Momcozy 自有数据 ---")
    mz_path = base_dir / "momcozy_voc_high_quality_sampled.csv"
    mz_df = pd.read_csv(mz_path)
    mz_records = convert_momcozy(mz_df, spu_mapping)
    all_records.extend(mz_records)
    stats["momcozy"] = len(mz_records)
    print(f"  转换: {len(mz_records):,} 条")

    # Trustpilot
    print("\n--- Trustpilot 数据 ---")
    tp_path = base_dir / "trustpilot_voc_100k_balanced.csv"
    tp_df = pd.read_csv(tp_path)
    tp_records = convert_trustpilot(tp_df)
    all_records.extend(tp_records)
    stats["trustpilot"] = len(tp_records)
    print(f"  转换: {len(tp_records):,} 条")

    # Reddit
    print("\n--- Reddit 数据 ---")
    rd_path = base_dir / "reddit_voc_sampled.csv"
    rd_df = pd.read_csv(rd_path)
    rd_records = convert_reddit(rd_df)
    all_records.extend(rd_records)
    stats["reddit"] = len(rd_records)
    print(f"  转换: {len(rd_records):,} 条")

    # Zendesk
    print("\n--- Zendesk 数据 ---")
    zd_path = base_dir / "zendesk_momcozy_voc_sampled.csv"
    zd_df = pd.read_csv(zd_path)
    zd_records = convert_zendesk(zd_df, spu_mapping)
    all_records.extend(zd_records)
    stats["zendesk"] = len(zd_records)
    print(f"  转换: {len(zd_records):,} 条")

    # 3. 统计与审计
    print("\n--- 审计报告 ---")
    print(f"总计: {len(all_records):,} 条")
    for source, count in stats.items():
        print(f"  {source}: {count:,} ({count/len(all_records)*100:.1f}%)")

    # SPU 映射覆盖率
    has_spu = sum(1 for r in all_records if r.spu_code)
    has_category = sum(1 for r in all_records if r.category)
    print(f"\n有 SPU: {has_spu:,} ({has_spu/len(all_records)*100:.1f}%)")
    print(f"有品类: {has_category:,} ({has_category/len(all_records)*100:.1f}%)")

    # 4. 保存中间文件
    output_jsonl = output_dir / "phase1_1_unified_voc_records.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"\n  输出: {output_jsonl} ({len(all_records):,} 条)")

    # 审计报告
    audit = {
        "phase": "1.1",
        "total_records": len(all_records),
        "source_breakdown": stats,
        "spu_mapped_rate": round(has_spu / len(all_records) * 100, 1),
        "category_mapped_rate": round(has_category / len(all_records) * 100, 1),
        "output_path": str(output_jsonl),
    }
    audit_path = output_dir / "phase1_1_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 1.1 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

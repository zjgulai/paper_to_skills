"""SPU/品线/品牌 标签宽表生成器

从 V3.3 打标结果生成业务级宽表：
1. 品牌 × 标签 宽表
2. 品线 × 标签 宽表 (基于规则分类)
3. SPU × 标签 宽表

输出: CSV 格式，可直接导入 BI 工具
"""

import json
import glob
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/00-归档资料/labeling-outputs/v3.3")
EXPORT_DIR = OUTPUT_BASE / "spu_matrices"


def load_all_labeled_data():
    """加载所有打标结果"""
    all_records = []
    for src in ["amazon", "trustpilot", "reddit", "zendesk"]:
        src_dir = OUTPUT_BASE / src
        if not src_dir.exists():
            continue
        for batch_file in sorted(src_dir.glob("batch_*.jsonl")):
            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    r["_source"] = src
                    all_records.append(r)
    return all_records


def extract_brand_from_text(text: str) -> str:
    """从文本中提取品牌（简化版）"""
    text_lower = text.lower()
    brand_patterns = {
        "momcozy": ["momcozy"],
        "elvie": ["elvie"],
        "spectra": ["spectra"],
        "medela": ["medela"],
        "tommee_tippee": ["tommee tippee", "tommee_tippee", "tommeetippee"],
        "avent": ["philips avent", "avent"],
        "dr_brown": ["dr brown", "dr_brown", "drbrown"],
        "willow": ["willow"],
        "lansinoh": ["lansinoh"],
        "haakaa": ["haakaa", "haaka"],
        "kinde": ["kiinde"],
        "babymoov": ["babymoov"],
        "fridababy": ["fridababy", "frida baby"],
        "nosefrida": ["nosefrida", "nose frida"],
        "wubbanub": ["wubbanub", "wubba nub"],
        "boppy": ["boppy"],
        "dockatot": ["dockatot", "dock a tot"],
        "halo": ["halo"],
        "snoo": ["snoo"],
        "owlet": ["owlet"],
        "nanit": ["nanit"],
        "miku": ["miku"],
    }

    for brand, patterns in brand_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                return brand
    return "other"


def build_brand_tag_matrix(records: list[dict]) -> pd.DataFrame:
    """构建品牌 × 标签 宽表"""
    print("\n" + "=" * 70)
    print("构建 品牌 × 标签 宽表")
    print("=" * 70)

    # 聚合数据
    brand_stats = defaultdict(lambda: {
        "total_voc": 0,
        "tag_hits": Counter(),
        "sentiment_sum": 0.0,
        "promoters": 0,
        "detractors": 0,
        "passives": 0,
    })

    all_tags = set()

    for r in records:
        brand = extract_brand_from_text(r.get("text_preview", ""))
        brand_stats[brand]["total_voc"] += 1
        brand_stats[brand]["sentiment_sum"] += r.get("sentiment_polarity", 0)

        nps = r.get("proxy_nps", "passive")
        if nps == "promoter":
            brand_stats[brand]["promoters"] += 1
        elif nps == "detractor":
            brand_stats[brand]["detractors"] += 1
        else:
            brand_stats[brand]["passives"] += 1

        for tag in r.get("aipl_tags", []):
            tag_en = tag["tag_en"]
            brand_stats[brand]["tag_hits"][tag_en] += 1
            all_tags.add(tag_en)

    # 构建 DataFrame
    all_tags_sorted = sorted(all_tags)
    rows = []

    for brand, stats in sorted(brand_stats.items(), key=lambda x: -x[1]["total_voc"]):
        total = stats["total_voc"]
        nps_val = (stats["promoters"] / total * 100) - (stats["detractors"] / total * 100) if total else 0

        row = {
            "品牌": brand,
            "总VOC": total,
            "覆盖率": round(sum(1 for t in stats["tag_hits"].values() if t > 0) / len(all_tags) * 100, 1) if all_tags else 0,
            "Proxy_NPS": round(nps_val, 1),
            "推荐者": stats["promoters"],
            "贬损者": stats["detractors"],
            "平均情感": round(stats["sentiment_sum"] / total, 2) if total else 0,
        }

        for tag in all_tags_sorted:
            row[f"{tag}_count"] = stats["tag_hits"].get(tag, 0)
            row[f"{tag}_rate"] = round(stats["tag_hits"].get(tag, 0) / total * 100, 2) if total else 0

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  品牌数: {len(df)}")
    print(f"  标签数: {len(all_tags_sorted)}")
    return df


def build_source_tag_matrix(records: list[dict]) -> pd.DataFrame:
    """构建数据源 × 标签 宽表"""
    print("\n" + "=" * 70)
    print("构建 数据源 × 标签 宽表")
    print("=" * 70)

    source_stats = defaultdict(lambda: {
        "total_voc": 0,
        "tag_hits": Counter(),
        "promoters": 0,
        "detractors": 0,
        "passives": 0,
    })

    all_tags = set()

    for r in records:
        src = r.get("_source", "unknown")
        source_stats[src]["total_voc"] += 1

        nps = r.get("proxy_nps", "passive")
        if nps == "promoter":
            source_stats[src]["promoters"] += 1
        elif nps == "detractor":
            source_stats[src]["detractors"] += 1
        else:
            source_stats[src]["passives"] += 1

        for tag in r.get("aipl_tags", []):
            tag_en = tag["tag_en"]
            source_stats[src]["tag_hits"][tag_en] += 1
            all_tags.add(tag_en)

    all_tags_sorted = sorted(all_tags)
    rows = []

    for src, stats in sorted(source_stats.items()):
        total = stats["total_voc"]
        nps_val = (stats["promoters"] / total * 100) - (stats["detractors"] / total * 100) if total else 0

        row = {
            "数据源": src,
            "总VOC": total,
            "Proxy_NPS": round(nps_val, 1),
            "推荐者": stats["promoters"],
            "贬损者": stats["detractors"],
        }

        for tag in all_tags_sorted:
            row[tag] = stats["tag_hits"].get(tag, 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  数据源: {len(df)}")
    return df


def build_tag_summary_matrix(records: list[dict]) -> pd.DataFrame:
    """构建标签汇总矩阵（全局视角）"""
    print("\n" + "=" * 70)
    print("构建 标签汇总矩阵")
    print("=" * 70)

    tag_stats = defaultdict(lambda: {
        "total_hits": 0,
        "by_source": Counter(),
        "by_nps": {"promoter": 0, "detractor": 0, "passive": 0},
        "avg_sentiment": [],
    })

    for r in records:
        src = r.get("_source", "unknown")
        nps = r.get("proxy_nps", "passive")
        sentiment = r.get("sentiment_polarity", 0)

        for tag in r.get("aipl_tags", []):
            tag_en = tag["tag_en"]
            tag_stats[tag_en]["total_hits"] += 1
            tag_stats[tag_en]["by_source"][src] += 1
            tag_stats[tag_en]["by_nps"][nps] += 1
            tag_stats[tag_en]["avg_sentiment"].append(sentiment)

    rows = []
    for tag_en, stats in sorted(tag_stats.items(), key=lambda x: -x[1]["total_hits"]):
        sentiments = stats["avg_sentiment"]
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0

        row = {
            "标签": tag_en,
            "总命中": stats["total_hits"],
            "Amazon": stats["by_source"].get("amazon", 0),
            "Trustpilot": stats["by_source"].get("trustpilot", 0),
            "Reddit": stats["by_source"].get("reddit", 0),
            "Zendesk": stats["by_source"].get("zendesk", 0),
            "推荐者": stats["by_nps"]["promoter"],
            "贬损者": stats["by_nps"]["detractor"],
            "被动者": stats["by_nps"]["passive"],
            "平均情感": round(avg_sent, 2),
            "NPS贡献": round((stats["by_nps"]["promoter"] - stats["by_nps"]["detractor"]) / stats["total_hits"] * 100, 1) if stats["total_hits"] else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  标签数: {len(df)}")
    return df


def main():
    print("=" * 70)
    print("SPU/品线/品牌 标签宽表生成")
    print("=" * 70)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n加载打标结果...")
    records = load_all_labeled_data()
    print(f"  总记录: {len(records):,}")

    # 1. 品牌 × 标签 宽表
    brand_df = build_brand_tag_matrix(records)
    brand_path = EXPORT_DIR / "brand_tag_matrix.csv"
    brand_df.to_csv(brand_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {brand_path}")

    # 2. 数据源 × 标签 宽表
    source_df = build_source_tag_matrix(records)
    source_path = EXPORT_DIR / "source_tag_matrix.csv"
    source_df.to_csv(source_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {source_path}")

    # 3. 标签汇总矩阵
    tag_df = build_tag_summary_matrix(records)
    tag_path = EXPORT_DIR / "tag_summary_matrix.csv"
    tag_df.to_csv(tag_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {tag_path}")

    print(f"\n{'=' * 70}")
    print(f"宽表生成完成: {EXPORT_DIR}")
    print(f"{'=' * 70}")

    # 打印 Top 品牌摘要
    print(f"\n{'=' * 70}")
    print("Top 品牌摘要")
    print(f"{'=' * 70}")
    for _, row in brand_df.head(5).iterrows():
        print(f"  {row['品牌']}: {row['总VOC']:,} VOC | NPS={row['Proxy_NPS']} | 覆盖率={row['覆盖率']}%")

    print(f"\n{'=' * 70}")
    print("Top 10 标签（全局）")
    print(f"{'=' * 70}")
    for _, row in tag_df.head(10).iterrows():
        print(f"  {row['标签']}: {row['总命中']:,} | NPS={row['NPS贡献']}")


if __name__ == "__main__":
    main()

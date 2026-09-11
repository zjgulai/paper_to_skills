"""Momcozy 品牌数据高质量抽样

抽样策略：
1. 分层抽样：按四级类目 × 星级评分 分层，确保各类目和情绪分布均衡
2. 负面优先：1-3 星评论抽样率更高（负面声量更有分析价值）
3. 长文本优先：文本长度 > 100 字符的优先纳入（信息密度更高）
4. 多语言覆盖：确保非英语评论有代表性样本

输出格式与现有打标流程兼容（CSV → batch_labeling.py 可处理）
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def detect_language(text: str) -> str:
    """简单语种检测"""
    t = str(text).lower()
    if re.search(r'\b(der|die|das|ist|nicht|sehr|gut|für|und|ein|war|haben|waren)\b', t):
        return "de"
    if re.search(r'\b(el|la|los|las|es|muy|bueno|para|y|un|una|este|esta|muy|bien)\b', t):
        return "es"
    if re.search(r'\b(le|la|les|est|très|bon|pour|et|un|une|ce|cette|très|bien)\b', t):
        return "fr"
    if re.search(r'\b(il|la|gli|è|molto|buono|per|e|un|una|questo|questa)\b', t):
        return "it"
    return "en"


def quality_score(row: pd.Series) -> float:
    """计算评论质量分，用于抽样权重"""
    text = str(row.get("买家评论", ""))
    rating = int(row.get("星级评分", 3))
    text_len = len(text)

    score = 0.0

    # 文本长度权重（100-500 字最佳）
    if 100 <= text_len <= 500:
        score += 3.0
    elif 50 <= text_len < 100:
        score += 2.0
    elif text_len >= 500:
        score += 1.5
    else:
        score += 0.5

    # 负面情绪权重（1-2 星更高）
    if rating <= 2:
        score += 3.0
    elif rating == 3:
        score += 1.5
    elif rating == 4:
        score += 0.5

    # 多语言奖励（非英语样本珍贵）
    lang = detect_language(text)
    if lang != "en":
        score += 1.0

    # 已有 AI 打标奖励（可对比验证）
    if pd.notna(row.get("classification-tag打标")):
        score += 0.5

    return score


def stratified_sample(df: pd.DataFrame, target_total: int = 10000) -> pd.DataFrame:
    """分层抽样

    按 (四级类目 × 星级评分) 分层，每层按比例分配样本数，
    但在每层内按 quality_score 排序取 top。
    """
    # 计算质量分
    df = df.copy()
    df["quality_score"] = df.apply(quality_score, axis=1)
    df["lang"] = df["买家评论"].apply(detect_language)

    # 过滤：必须有评论文本
    df = df[df["买家评论"].notna() & (df["买家评论"].str.len() > 10)]

    # 分层：四级类目 + 星级评分（1-2星合并为"负面"，4-5星合并为"正面"）
    df["sentiment_group"] = df["星级评分"].apply(
        lambda x: "negative" if x <= 2 else "neutral" if x == 3 else "positive"
    )

    # 统计各层数量
    stratify_counts = df.groupby(["四级类目名称", "sentiment_group"]).size().reset_index(name="count")
    total_available = len(df)

    print(f"可用评论数: {total_available:,}")
    print(f"目标抽样数: {target_total:,}")
    print()

    # 分配样本：按层大小比例分配，但确保每层至少有一定样本
    stratify_counts["sample_size"] = (
        stratify_counts["count"] / total_available * target_total
    ).clip(lower=5).astype(int)

    # 调整使总和精确等于 target_total
    diff = target_total - stratify_counts["sample_size"].sum()
    if diff > 0:
        # 从最大层补充
        idx = stratify_counts["count"].idxmax()
        stratify_counts.loc[idx, "sample_size"] += diff

    # 执行抽样
    sampled = []
    for _, row in stratify_counts.iterrows():
        cat = row["四级类目名称"]
        sent = row["sentiment_group"]
        n = row["sample_size"]
        avail = row["count"]

        layer = df[(df["四级类目名称"] == cat) & (df["sentiment_group"] == sent)]
        # 按质量分排序，取 top n
        layer_sorted = layer.sort_values("quality_score", ascending=False)
        actual_n = min(n, len(layer_sorted))
        picked = layer_sorted.head(actual_n)
        sampled.append(picked)

        if avail > 0:
            print(f"  {cat:20s} | {sent:8s} | 取 {actual_n:4d} / {avail:4d}")

    result = pd.concat(sampled, ignore_index=True)
    print(f"\n实际抽样数: {len(result):,}")
    return result


def export_for_labeling(df_sample: pd.DataFrame, output_dir: Path):
    """导出为打标流程兼容格式"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 转换为与现有 batch_labeling.py 兼容的 CSV 格式
    # 字段映射：
    #   工单号 → review_id
    #   买家评论 → Content/_review_text
    #   星级评分 → Rating
    #   SPU名称 → SPU
    #   四级类目名称 → category
    export_df = pd.DataFrame({
        "review_id": df_sample["工单号"],
        "text": df_sample["买家评论"],
        "rating": df_sample["星级评分"],
        "spu_name": df_sample["SPU名称"],
        "spu_code": df_sample["SPU编码"],
        "category_lv4": df_sample["四级类目名称"],
        "country": df_sample["国家"],
        "language": df_sample["lang"],
        "quality_score": df_sample["quality_score"],
        "ai_tag": df_sample.get("classification-tag打标", ""),
        "data_source": "momcozy_amazon",
    })

    # 保存完整抽样
    full_path = output_dir / "momcozy_sampled_full.csv"
    export_df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"完整抽样已保存: {full_path} ({len(export_df):,} 条)")

    # 按情绪分层保存（方便分别分析）
    for sent_group, label in [("negative", "1-2星"), ("neutral", "3星"), ("positive", "4-5星")]:
        mask = df_sample["sentiment_group"] == sent_group
        subset = export_df[mask]
        path = output_dir / f"momcozy_sampled_{sent_group}.csv"
        subset.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {label}: {len(subset):,} 条 → {path.name}")

    # 保存多语言子集
    non_en = export_df[export_df["language"] != "en"]
    if len(non_en) > 0:
        path = output_dir / "momcozy_sampled_non_english.csv"
        non_en.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  非英语: {len(non_en):,} 条 → {path.name}")

    # 保存统计摘要
    summary = {
        "total_sampled": len(export_df),
        "by_rating": export_df["rating"].value_counts().sort_index().to_dict(),
        "by_category": export_df["category_lv4"].value_counts().head(20).to_dict(),
        "by_language": export_df["language"].value_counts().to_dict(),
        "by_country": export_df["country"].value_counts().head(10).to_dict(),
        "avg_text_length": export_df["text"].str.len().mean(),
    }
    summary_path = output_dir / "sample_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"统计摘要: {summary_path}")

    return export_df


def main():
    parser = argparse.ArgumentParser(description="Momcozy 数据高质量抽样")
    parser.add_argument("--input", required=True, help="Momcozy VOC Excel 文件路径")
    parser.add_argument("--output", default="./momcozy_sampled", help="输出目录")
    parser.add_argument("--n", type=int, default=10000, help="目标抽样数")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    print("=" * 70)
    print("Momcozy 品牌数据高质量抽样")
    print("=" * 70)

    print(f"\n[1/3] 加载数据...")
    df = pd.read_excel(input_path)
    print(f"  原始数据: {len(df):,} 条, {len(df.columns)} 列")

    print(f"\n[2/3] 分层抽样 (目标 {args.n:,} 条)...")
    df_sample = stratified_sample(df, target_total=args.n)

    print(f"\n[3/3] 导出打标兼容格式...")
    export_for_labeling(df_sample, output_dir)

    print("\n" + "=" * 70)
    print("抽样完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

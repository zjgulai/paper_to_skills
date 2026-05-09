"""Active-Learning 质量把关（Phase 2.5 简化版）

对每个候选标签，从支持样本中筛选不确定性最高的 Top-K 样本，
输出供人工审核的清单。

不确定性指标：
- 情感极性混合度（样本中同时出现正面/负面词）
- 关键词歧义度（匹配词在多种上下文中出现）
- 标签冲突度（样本同时匹配其他不相关标签）
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


NEG_WORDS = {"bad", "terrible", "awful", "worst", "hate", "disappoint", "poor",
             "horrible", "useless", "broken", "defective", "leak", "pain",
             "uncomfortable", "hard", "rough", "noisy", "loud", "difficult",
             "complicated", "frustrat", "annoy", "waste", "not work", "doesnt work",
             "didnt work", "stopped working", "return", "refund", "money back"}
POS_WORDS = {"good", "great", "excellent", "amazing", "love", "perfect", "awesome",
             "fantastic", "wonderful", "best", "happy", "satisf", "comfortable",
             "soft", "nice", "smooth", "easy", "convenient", "efficient",
             "reliable", "durable", "quality", "recommend", "highly"}


def compute_uncertainty(text: str, keyword: str) -> float:
    """计算单条样本的不确定性分数（越高越需要审核）"""
    text_lower = text.lower()

    score = 0.0

    # 1. 情感混合度
    has_pos = any(w in text_lower for w in POS_WORDS)
    has_neg = any(w in text_lower for w in NEG_WORDS)
    if has_pos and has_neg:
        score += 0.4  # 情感矛盾

    # 2. 关键词前是否有否定词
    idx = text_lower.find(keyword.lower())
    if idx > 0:
        prefix = text_lower[max(0, idx - 20):idx]
        negations = {"not", "no", "never", "none", "n't", "dont", "doesnt", "didnt",
                     "isnt", "wasnt", "wont", "cant", "couldnt", "wouldnt"}
        if any(n in prefix for n in negations):
            score += 0.3  # 可能被否定

    # 3. 文本长度极短（信息量不足）
    if len(text) < 50:
        score += 0.2

    # 4. 包含极端词（可能情绪化）
    extreme = {"scam", "fraud", "garbage", "trash", "rip off", "worst ever",
               "never again", "complete waste", "total disappointment"}
    if any(w in text_lower for w in extreme):
        score += 0.1

    return round(min(score, 1.0), 2)


def main():
    print("=" * 70)
    print("Phase 2.5: Active-Learning 质量把关")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent / "04-输出结果"

    # 1. 加载候选标签
    print("\n--- 加载候选标签 ---")
    candidates_path = base_dir / "tag_gap_analysis/candidate_tags_filtered.json"
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # 只保留高频候选
    candidates = [c for c in candidates if c["support_count"] >= 20]
    print(f"  高频候选: {len(candidates)} 个")

    # 2. 加载零标签样本
    print("\n--- 加载零标签样本 ---")
    zero_samples_path = base_dir / "tag_gap_analysis/zero_label_samples.csv"
    zero_df = pd.read_csv(zero_samples_path)
    print(f"  样本数: {len(zero_df):,}")

    # 建立品类 -> 样本索引
    cat_samples = defaultdict(list)
    for _, row in zero_df.iterrows():
        cat = str(row.get("category", "")).strip()
        if cat:
            cat_samples[cat].append({
                "review_id": row["review_id"],
                "text": str(row.get("text", "")),
                "data_source": row.get("data_source", ""),
            })

    # 3. 对每个候选标签采样不确定性最高的样本
    print("\n--- 不确定性采样 ---")
    audit_results = []
    top_k = 10  # 每个候选标签采样 10 条

    for i, cand in enumerate(candidates):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1} / {len(candidates)}")

        cat = cand["applicable_category"]
        phrase = cand["source_phrase"].lower()

        samples = cat_samples.get(cat, [])
        matched = []

        for s in samples:
            if phrase in s["text"].lower():
                uncertainty = compute_uncertainty(s["text"], phrase)
                matched.append({
                    "review_id": s["review_id"],
                    "text_preview": s["text"][:200],
                    "uncertainty": uncertainty,
                    "data_source": s["data_source"],
                })

        # 按不确定性排序，取 Top-K
        matched.sort(key=lambda x: x["uncertainty"], reverse=True)
        top_samples = matched[:top_k]

        avg_uncertainty = sum(s["uncertainty"] for s in matched) / len(matched) if matched else 0

        audit_results.append({
            "tag_en": cand["tag_en"],
            "tag_cn": cand["tag_cn"],
            "category": cat,
            "suggested_aipl": cand["suggested_aipl"],
            "suggested_sentiment": cand["suggested_sentiment"],
            "support_count": cand["support_count"],
            "avg_uncertainty": round(avg_uncertainty, 2),
            "needs_review": avg_uncertainty > 0.3,
            "audit_samples": top_samples,
            "total_matched": len(matched),
        })

    # 4. 分类结果
    needs_review = [r for r in audit_results if r["needs_review"]]
    auto_approve = [r for r in audit_results if not r["needs_review"]]

    print(f"\n--- 审核结果 ---")
    print(f"  需要人工审核: {len(needs_review)} / {len(audit_results)}")
    print(f"  自动通过: {len(auto_approve)} / {len(audit_results)}")

    # 5. 输出审核报告
    output_dir = base_dir / "tag_gap_analysis"
    report_path = output_dir / "active_learning_audit_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_candidates": len(audit_results),
                "needs_review": len(needs_review),
                "auto_approve": len(auto_approve),
            },
            "needs_review": needs_review,
            "auto_approve": auto_approve,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  审核报告: {report_path}")

    # 6. 输出人工审核清单（CSV，便于阅读）
    review_rows = []
    for r in needs_review:
        for s in r["audit_samples"]:
            review_rows.append({
                "候选标签(英)": r["tag_en"],
                "候选标签(中)": r["tag_cn"],
                "品类": r["category"],
                "AIPL": r["suggested_aipl"],
                "情感": r["suggested_sentiment"],
                "支持数": r["support_count"],
                "不确定度": s["uncertainty"],
                "样本预览": s["text_preview"],
                "数据源": s["data_source"],
                "审核意见": "",
            })

    if review_rows:
        review_df = pd.DataFrame(review_rows)
        review_csv_path = output_dir / "manual_review_checklist.csv"
        review_df.to_csv(review_csv_path, index=False)
        print(f"  人工审核清单: {review_csv_path} ({len(review_rows)} 条)")

    # 7. 输出自动通过的候选标签（可直接进入字典）
    approved = [
        {
            "tag_en": r["tag_en"],
            "tag_cn": r["tag_cn"],
            "category": r["category"],
            "aipl": r["suggested_aipl"],
            "sentiment": r["suggested_sentiment"],
            "support_count": r["support_count"],
        }
        for r in auto_approve
    ]
    approved_path = output_dir / "auto_approved_candidates.json"
    with open(approved_path, "w", encoding="utf-8") as f:
        json.dump(approved, f, ensure_ascii=False, indent=2)
    print(f"  自动通过候选: {approved_path} ({len(approved)} 个)")

    # 8. 审计
    audit = {
        "phase": "2.5",
        "total_candidates": len(audit_results),
        "needs_review_count": len(needs_review),
        "auto_approve_count": len(auto_approve),
        "review_samples": len(review_rows),
        "report_path": str(report_path),
        "review_csv_path": str(review_csv_path) if review_rows else None,
        "approved_path": str(approved_path),
    }
    audit_path = output_dir / "phase2_5_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 2.5 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""轻量版缺口检测（Phase 2.2）

对零标签 VOC 进行高频词/主题分析，与现有标签关键词对比，
识别未覆盖主题，生成候选标签。
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ── 文本预处理 ────────────────────────────────────────────────────

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "among", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "am", "so", "very", "just", "now", "then", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "not", "only", "own",
    "same", "than", "too", "also", "as", "until", "while", "because",
    "if", "unless", "since", "although", "though", "once",
    # 噪音词（系统标记、URL、HTML、外语等）
    "yiv", "https", "http", "url", "com", "www", "html", "width", "height",
    "list", "items", "item", "vous", "nous", "les", "des", "pour", "est",
    "pas", "sur", "que", "une", "qui", "dans", "plus", "fait", "tout",
    "avec", "son", "sont", "comme", "tous", "deux", "upn", "msn",
    "please", "thank", "thanks", "regards", "sincerely", "dear",
    "kindly", "note", "attached", "attachment", "enclosure",
    "click", "link", "page", "website", "site", "online", "web",
    "image", "photo", "picture", "video", "file", "document",
    # 通用评价词（避免生成过于泛化的标签）
    "item", "order", "buy", "bought", "purchase", "purchased",
    "got", "get", "really", "definitely", "absolutely", "highly",
    "recommend", "recommended", "great", "good", "nice", "love",
    "loved", "like", "liked", "use", "used", "using",
    "would", "will", "one", "two", "first", "last", "time",
    "way", "thing", "things", "something", "anything", "everything",
    "product", "customer", "service", "support", "help", "helped",
    "made", "make", "makes", "making", "say", "said", "says",
    "tell", "told", "think", "thought", "know", "knew", "see",
    "saw", "seen", "look", "looked", "looking", "want", "wanted",
    "needs", "need", "needed", "came", "come", "comes", "coming",
    "went", "go", "goes", "going", "take", "took", "taken", "taking",
    "give", "gave", "given", "giving", "put", "puts", "putting",
    "find", "found", "finding", "work", "worked", "working", "works",
    "try", "tried", "trying", "tries", "start", "started", "starting",
    "keep", "kept", "keeping", "let", "lets", "letting", "left",
    "leave", "leaving", "feels", "feel", "felt", "feeling",
    "thoughts", "opinion", "experience", "experiences", "review",
    "reviews", "update", "updated", "month", "months", "week", "weeks",
    "day", "days", "year", "years", "today", "yesterday", "tomorrow",
}


def tokenize(text: str) -> list[str]:
    """提取有意义的词（去除停用词、短词）"""
    text_lower = text.lower()
    # 提取 3+ 字母的英文单词
    words = re.findall(r"[a-z]{3,}", text_lower)
    return [w for w in words if w not in STOP_WORDS and len(w) >= 3]


def extract_ngrams(text: str, n: int = 2) -> list[str]:
    """提取 n-gram"""
    words = tokenize(text)
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


# ── 缺口检测 ──────────────────────────────────────────────────────

def load_existing_keywords() -> set[str]:
    """加载现有标签的所有关键词"""
    path = (
        Path(__file__).parent.parent.parent
        / "03-数据资产/原种子标签/SGCS_标签字典_VOC标签大宽表V3.0_v1.xlsx"
    )

    sheets = [
        "01_通用标签主表", "02_吸奶器", "03_内衣服饰", "04_家居家纺",
        "05_母婴综合护理", "06_喂养电器", "07_智能母婴电器",
    ]

    all_keywords = set()
    for sheet in sheets:
        df = pd.read_excel(path, sheet_name=sheet)
        for col in ["英文关键词/典型表达", "消费者习惯关键词/原话短语"]:
            if col in df.columns:
                for val in df[col].dropna():
                    for kw in re.split(r"[,;\n]", str(val)):
                        kw = kw.strip().lower()
                        if kw and len(kw) >= 3:
                            all_keywords.add(kw)
                            # 也加入单个词
                            for w in kw.split():
                                if len(w) >= 3:
                                    all_keywords.add(w)

    print(f"  现有标签关键词库: {len(all_keywords)} 个")
    return all_keywords


def detect_gaps_by_category(
    zero_label_samples: list[dict],
    existing_keywords: set[str],
    top_k: int = 20,
) -> list[dict]:
    """按品类检测关键词缺口"""

    # 按品类分组
    by_category = defaultdict(list)
    for s in zero_label_samples:
        by_category[s["category"]].append(s["text"])

    results = []

    for cat, texts in sorted(by_category.items()):
        if len(texts) < 10:
            continue

        # 提取所有词和 bigram
        all_words = Counter()
        all_bigrams = Counter()

        for text in texts:
            words = tokenize(text)
            bigrams = extract_ngrams(text, 2)
            all_words.update(words)
            all_bigrams.update(bigrams)

        # 过滤已在关键词库中的词
        novel_words = [
            (w, c) for w, c in all_words.most_common(top_k * 3)
            if w not in existing_keywords and c >= 5
        ][:top_k]

        novel_bigrams = [
            (bg, c) for bg, c in all_bigrams.most_common(top_k * 3)
            if not any(kw in bg for kw in existing_keywords if len(kw) >= 5)
            and c >= 3
        ][:top_k]

        if novel_words or novel_bigrams:
            results.append({
                "category": cat,
                "zero_label_count": len(texts),
                "novel_words": [{"word": w, "count": c} for w, c in novel_words],
                "novel_bigrams": [{"phrase": bg, "count": c} for bg, c in novel_bigrams],
            })

    return results


def suggest_tags_from_gaps(gap_results: list[dict]) -> list[dict]:
    """基于缺口结果生成候选标签建议"""
    candidates = []

    for gap in gap_results:
        cat = gap["category"]

        # 从 bigram 生成候选标签
        for bg_info in gap["novel_bigrams"][:5]:
            phrase = bg_info["phrase"]
            count = bg_info["count"]

            # 简单推断 AIPL 节点（基于关键词）
            aipl = "L1"  # 默认 L1（使用体验）
            if any(w in phrase for w in ["buy", "purchase", "order", "shopping", "decide", "choose"]):
                aipl = "P1"
            elif any(w in phrase for w in ["recommend", "suggest", "gift", "share", "review", "rate", "star"]):
                aipl = "L3"
            elif any(w in phrase for w in ["ship", "delivery", "arrive", "package", "logistics", "fast", "slow"]):
                aipl = "P2"
            elif any(w in phrase for w in ["search", "find", "look", "compare", "research", "information"]):
                aipl = "I"

            # 推断情感
            neg_words = {"bad", "terrible", "awful", "worst", "hate", "disappoint", "poor", "horrible",
                         "useless", "broken", "defective", "leak", "pain", "uncomfortable", "hard",
                         "rough", "noisy", "loud", "difficult", "complicated", "frustrat", "annoy",
                         "waste", "not work", "doesnt work", "didnt work", "stopped working",
                         "break", "leak", "leaking", "painful", "hurt", "uncomfortable"}
            pos_words = {"good", "great", "excellent", "amazing", "love", "perfect", "awesome",
                         "fantastic", "wonderful", "best", "happy", "satisf", "comfortable",
                         "soft", "nice", "smooth", "easy", "convenient", "efficient", "reliable",
                         "durable", "quality", "recommend"}

            sentiment = "neutral"
            if any(w in phrase for w in neg_words):
                sentiment = "negative"
            elif any(w in phrase for w in pos_words):
                sentiment = "positive"

            candidates.append({
                "tag_en": phrase.replace(" ", "_"),
                "tag_cn": f"[{cat}] {phrase}",
                "suggested_aipl": aipl,
                "suggested_sentiment": sentiment,
                "applicable_category": cat,
                "support_count": count,
                "source_phrase": phrase,
                "source": "gap_detection",
            })

    return candidates


# ── 主流程 ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase 2.2: 缺口检测（轻量版）")
    print("=" * 70)

    output_dir = Path(__file__).parent.parent.parent / "04-输出结果/tag_gap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载现有关键词库
    print("\n--- 加载现有标签关键词 ---")
    existing_keywords = load_existing_keywords()

    # 2. 加载零标签样本
    print("\n--- 加载零标签样本 ---")
    samples_path = output_dir / "zero_label_samples.csv"
    df = pd.read_csv(samples_path)
    samples = df.to_dict("records")
    print(f"  样本数: {len(samples):,}")

    # 3. 按品类检测缺口
    print("\n--- 按品类检测缺口 ---")
    gap_results = detect_gaps_by_category(samples, existing_keywords, top_k=15)
    print(f"  发现缺口品类: {len(gap_results)}")

    # 4. 生成候选标签
    print("\n--- 生成候选标签 ---")
    candidates = suggest_tags_from_gaps(gap_results)
    print(f"  候选标签: {len(candidates)}")

    # 去重
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = (c["tag_en"], c["applicable_category"])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    print(f"  去重后: {len(unique_candidates)}")

    # 5. 保存结果
    gap_path = output_dir / "gap_analysis.json"
    with open(gap_path, "w", encoding="utf-8") as f:
        json.dump(gap_results, f, ensure_ascii=False, indent=2)
    print(f"\n  缺口报告: {gap_path}")

    candidates_path = output_dir / "candidate_tags_raw.json"
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(unique_candidates, f, ensure_ascii=False, indent=2)
    print(f"  候选标签: {candidates_path}")

    # 6. 审计
    audit = {
        "phase": "2.2",
        "description": "轻量版缺口检测（关键词频率对比）",
        "existing_keywords": len(existing_keywords),
        "sample_count": len(samples),
        "gap_categories": len(gap_results),
        "candidate_tags": len(unique_candidates),
        "top_candidates": [
            {"tag_en": c["tag_en"], "category": c["applicable_category"], "count": c["support_count"]}
            for c in sorted(unique_candidates, key=lambda x: x["support_count"], reverse=True)[:20]
        ],
        "gap_path": str(gap_path),
        "candidates_path": str(candidates_path),
    }
    audit_path = output_dir / "phase2_2_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"  审计: {audit_path}")

    # 打印 Top 候选
    print(f"\n--- Top 20 候选标签 ---")
    for c in sorted(unique_candidates, key=lambda x: x["support_count"], reverse=True)[:20]:
        print(f"  [{c['applicable_category']}] {c['tag_en']} ({c['support_count']}) -> {c['suggested_aipl']}")

    print("\n" + "=" * 70)
    print("Phase 2.2 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

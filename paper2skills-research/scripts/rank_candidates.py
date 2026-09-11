#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
近三个月论文候选 → 业务推荐清单 打分器

输入:  data/arxiv_candidates.json （arxiv_harvest.py 产出）
输出:  data/recommendations.json   全量打分结果
       data/recommendations.csv    排序清单
       data/shortlist.md           按领域分组的候选短名单（人读）

打分维度（总分 100）:
  A. 发表层级 venue_tier   0-30  —— 顶会/顶刊正式接收 > workshop > arXiv only
  B. 工程可得性 code       0-15  —— 有开源链接 / 有系统实现描述
  C. 业务相关性 business   0-20  —— 母婴出海电商场景关键词命中
  D. 方法可萃取性 method   0-20  —— 是否有明确算法/实验/数据集（排除纯综述、纯立场）
  E. 时效 freshness        0-8   —— 越新越高
  F. 匹配缺口缺口 gap      0-7   —— 命中项目中尚无 Skill 卡片的空白方向
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ---------- A. 发表层级 ----------
TIER_A = {
    "neurips": "NeurIPS", "nips": "NeurIPS", "icml": "ICML", "iclr": "ICLR",
    "kdd": "KDD", "sigkdd": "KDD", "sigir": "SIGIR", "www": "WWW", "thewebconf": "WWW",
    "wsdm": "WSDM", "recsys": "RecSys", "cikm": "CIKM", "acl": "ACL", "emnlp": "EMNLP",
    "naacl": "NAACL", "aaai": "AAAI", "ijcai": "IJCAI", "aamas": "AAMAS", "colm": "COLM",
    "uai": "UAI", "aistats": "AISTATS", "mlsys": "MLSys", "cvpr": "CVPR", "iccv": "ICCV",
    "eccv": "ECCV", "vldb": "VLDB", "sigmod": "SIGMOD", "icde": "ICDE", "tkde": "TKDE",
    "tois": "TOIS", "tist": "TIST", "management science": "Management Science",
    "marketing science": "Marketing Science", "operations research": "Operations Research",
    "m&som": "M&SOM", "msom": "M&SOM", "mis quarterly": "MIS Quarterly",
    "information systems research": "ISR",
}
WORKSHOP_HINTS = ("workshop", "tutorial", "findings", "demo track", "industry track", "challenge")
NEG_HINTS = ("arxiv preprint", "under review", "preprint", "submitted to", "to appear")

# ---------- C. 业务相关性 ----------
BUSINESS_STRONG = [
    "e-commerce", "ecommerce", "online retail", "cross-border", "marketplace", "seller",
    "shopping", "product listing", "assortment", "merchandis", "catalog",
    "advertis", "bidding", "sponsored search", "roas", "budget allocation", "campaign",
    "recommendation system", "recommender", "search ranking", "ranking system",
    "demand forecasting", "inventory", "replenishment", "supply chain", "fulfillment",
    "pricing", "promotion", "discount", "coupon", "subscription",
    "customer churn", "lifetime value", "retention", "conversion", "funnel", "uplift",
    "a/b test", "ab test", "experiment platform", "bandit",
    "customer review", "user feedback", "voice of customer", "sentiment", "aspect-based",
    "logistics", "warehouse", "shipping", "return",
]
BUSINESS_WEAK = [
    "user behavior", "click-through", "ctr", "personaliz", "cold-start", "session",
    "anomaly detection", "knowledge graph", "text-to-sql", "data analysis agent",
    "marketing", "consumer", "brand",
]
# 母婴品类锚点（命中则强加权）
CATEGORY_ANCHORS = [
    "baby", "infant", "mother", "maternal", "toddler", "nursery", "stroller",
    "breast pump", "diaper", "formula", "feeding", "carseat", "car seat",
]

# ---------- D. 方法可萃取性 ----------
METHOD_POS = [
    "we propose", "we introduce", "we present", "our method", "our approach", "our framework",
    "algorithm", "framework", "architecture", "model we", "we develop", "we design",
    "experiments", "experimental results", "benchmark", "dataset", "ablation",
    "outperform", "state-of-the-art", "sota", "baseline", "evaluation on",
]
METHOD_NEG = [
    "survey", "systematic review", "literature review", "meta-analysis", "position paper",
    "we argue", "call for", "roadmap", "tutorial", "overview of", "taxonomy of existing",
    "research agenda", "perspective paper", "soK:", "sok ",
]

GAP_KEYWORDS = {
    "广告归因/增量": ["attribution", "incrementality", "geo experiment", "marketing mix", "mmm"],
    "实验平台/序贯检验": ["sequential test", "always valid", "switchback", "crossover", "variance reduction", "cuped"],
    "选品/组合优化": ["assortment", "product selection", "new product", "category management"],
    "供应链-LLM": ["supply chain", "procurement", "supplier", "lead time"],
    "推荐-生成式": ["generative recommendation", "semantic id", "llm ranker", "reranker"],
    "Agent技能/上下文": ["agent skill", "skill library", "context compression", "context engineering", "memory"],
    "评价/自动化评测": ["llm-as-a-judge", "judge", "rubric", "evaluation harness", "benchmark"],
    "跨境合规/多语言": ["multilingual", "cross-lingual", "compliance", "regulation", "customs"],
}


def venue_of(item: dict) -> tuple[int, str]:
    """返回 (tier 分, 命中venue名)"""
    blob = f"{item.get('comment','')} {item.get('journal_ref','')}"
    low = blob.lower()
    hits: list[str] = []
    for key, name in TIER_A.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            hits.append(name)
    if not hits:
        # 摘要里的“published at”类描述
        return (0, "")
    is_ws = any(h in low for h in WORKSHOP_HINTS)
    top = {"NeurIPS", "ICML", "ICLR", "KDD", "SIGIR", "WWW", "WSDM", "RecSys", "ACL",
           "EMNLP", "NAACL", "AAAI", "IJCAI", "AAMAS", "COLM", "VLDB", "SIGMOD",
           "Management Science", "Marketing Science", "Operations Research", "M&SOM",
           "MIS Quarterly", "ISR", "TKDE", "TOIS", "TIST", "UAI", "AISTATS", "MLSys"}
    best = sorted(hits, key=lambda h: (h not in top, h))[0]
    if best in top:
        return (18 if is_ws else 30), best
    return (10 if is_ws else 18), best


def score(item: dict) -> dict:
    text = f"{item['title']} . {item['abstract']} . {item.get('comment','')}"
    low = text.lower()
    title_low = item["title"].lower()

    tier, venue = venue_of(item)

    # B. 代码
    code = 0
    if re.search(r"github\.com|huggingface\.co|code is available|our code|code will be|open-source", low):
        code += 11
    if re.search(r"open[- ]sourc|we release|publicly available|repository", low):
        code += 4
    code = min(code, 15)

    # C. 业务
    strong_hits = [k for k in BUSINESS_STRONG if k in low]
    weak_hits = [k for k in BUSINESS_WEAK if k in low]
    anchors = [k for k in CATEGORY_ANCHORS if k in low]
    business = min(14, 3.5 * len(set(strong_hits)))
    business += min(6, 1.2 * len(set(weak_hits)))
    if anchors:
        business += 4
    business = min(20, round(business, 1))

    # D. 方法
    pos = sum(1 for k in METHOD_POS if k in low)
    neg = sum(1 for k in METHOD_NEG if k in low)
    method = min(20, 3.0 * pos)
    if neg:
        method = max(0, method - 8 * neg)
    if re.search(r"\babstract\b.{0,80}\b(survey|review)\b", low) or title_low.startswith(("a survey", "survey", "a review", "systematic review")):
        method = min(method, 4)
    method = round(method, 1)

    # E. 时效
    try:
        pub = datetime.fromisoformat(item["published"].replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - pub).days
    except Exception:
        days = 92
    fresh = round(max(0, 8 - days / 12), 1)

    # F. 缺口
    gap_hits = [g for g, kws in GAP_KEYWORDS.items() if any(k in low for k in kws)]
    gap = min(7, 2.5 * len(gap_hits))

    total = round(tier + code + business + method + fresh + gap, 1)
    return {
        "score": total,
        "s_venue": tier, "venue": venue,
        "s_code": code,
        "s_business": business,
        "s_method": method,
        "s_fresh": fresh,
        "s_gap": round(gap, 1),
        "gap_hits": gap_hits,
        "strong_hits": sorted(set(strong_hits))[:8],
        "anchors": sorted(set(anchors)),
        "days_old": days,
    }


def main() -> None:
    src = json.loads((DATA / "arxiv_candidates.json").read_text(encoding="utf-8"))
    items = src["items"]
    rows = []
    for it in items:
        s = score(it)
        rows.append({**it, **s})
    rows.sort(key=lambda r: -r["score"])

    (DATA / "recommendations.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                    "window": src["window"], "count": len(rows), "items": rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    cols = ["score", "s_venue", "s_code", "s_business", "s_method", "s_gap", "venue",
            "arxiv_id", "published", "domains", "title", "journal_ref", "strong_hits", "url"]
    with (DATA / "recommendations.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["score"], r["s_venue"], r["s_code"], r["s_business"], r["s_method"], r["s_gap"],
                r["venue"], r["arxiv_id"], r["published"][:10], "|".join(r["query_groups"]),
                r["title"], r["journal_ref"], "|".join(r["strong_hits"]), r["url"],
            ])

    # 短名单 markdown
    order = ["16-智能体工程", "00-电商Agent", "05-推荐系统", "13-广告分析", "02-A_B实验",
             "01-因果推断", "03-时间序列", "04-供应链", "06-增长模型", "07-VOC舆情",
             "08-知识图谱", "09-DataAgent", "10-MAS", "14-用户分析", "15-营销投放",
             "12-ML基础", "11-AI人文"]
    lines = ["# 近三个月（2026-06-12 → 2026-09-11）候选论文短名单", "",
             f"候选总数 {len(rows)}；下表为各领域综合得分 Top 10。", ""]
    for dom in order:
        sub = [r for r in rows if dom in r["query_groups"]][:10]
        if not sub:
            continue
        lines += [f"## {dom}（命中 {sum(1 for r in rows if dom in r['query_groups'])} 篇，列 Top {len(sub)}）", "",
                  "| 分 | 会议/期刊 | arXiv | 日期 | 标题 |", "|---|---|---|---|---|"]
        for r in sub:
            lines.append(f"| {r['score']} | {r['venue'] or '—'} | {r['arxiv_id']} | {r['published'][:10]} | {r['title'][:95]} |")
        lines.append("")
    (DATA / "shortlist.md").write_text("\n".join(lines), encoding="utf-8")

    print("Top 40 总榜")
    print(f"{'分':>5} {'venue':<10} {'arxiv':<12} {'日期':<11} title")
    for r in rows[:40]:
        print(f"{r['score']:>5} {r['venue'] or '—':<10} {r['arxiv_id']:<12} {r['published'][:10]} {r['title'][:78]}")
    print()
    print("venue 命中统计:", dict(Counter(r["venue"] for r in rows if r["venue"]).most_common(20)))
    print("分数分布:", {k: sum(1 for r in rows if r["score"] >= t) for k, t in
                    [("≥70", 70), ("≥60", 60), ("≥50", 50), ("≥40", 40), ("≥30", 30)]})


if __name__ == "__main__":
    main()

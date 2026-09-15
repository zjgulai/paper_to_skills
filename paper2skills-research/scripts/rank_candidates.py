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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CONFIG_PATH = ROOT.parent / "paper2skills-vault" / "07-资源库" / "scoring_config.json"



# ---------------------------------------------------------------------------
# 域名守卫（PHASE6 P1）：本文件用到的每个域标签都必须是**规范名**
# ---------------------------------------------------------------------------
# 本仓库实测过一次「同一件事 7 处各写一份名字」造成的静默失效：
# `arxiv_harvest` 写旧名、`candidate_filter` 写新名 ⇒ 1046 篇候选里 141 篇
# 的负向词与约束词**一条都没生效**（且没有任何读数）。此处照 `arxiv_harvest.py`
# 的办法，在**自己这一侧** fail loud：认不出的名字当场炸，不留给下游去猜。
# (sys.path 见下)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import domains as _domains  # noqa: E402


def _assert_canonical(labels, where: str) -> None:
    bad = sorted(set(labels) - set(_domains.CANONICAL))
    if bad:
        hint = {b: _domains.RETIRED[b] for b in bad if b in _domains.RETIRED}
        raise SystemExit(
            f"🔴 {where} 里出现非规范域名：{bad}\n"
            + (f"   其中这些是**已退休的旧名**，规范名是：{hint}\n" if hint else "")
            + f"   规范名以 `domains.py` 的 CANONICAL 为准（{len(_domains.CANONICAL)} 个）。")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """读取外置评分配置。

    ⚠️ 设计约束：**配置文件缺失时必须完全退回内置默认值**，且默认值等于本文件
    原本硬编码的那一套。这样「引入配置文件」这件事本身不改变任何一条既有排序 ——
    否则无法区分「调参导致的排序变化」与「重构引入的回归」。
    实测验收方式：重构前后 recommendations.json 的 md5 必须一致。
    """
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                                   # 配置坏了不能静默用空值
        print(f"⚠️  评分配置 {path} 解析失败（{exc}），本轮使用内置默认值")
        return {}


CFG = load_config()
DIM = CFG.get("dimensions", {})

_assert_canonical(
    CFG.get("shortlist_domain_order") or SHORTLIST_ORDER,
    "rank_candidates 的短名单域顺序")
THRESH = CFG.get("thresholds", {})

#: 短名单的域顺序（`scoring_config.json` 可覆盖）。⚠️ 顺序 = 展示序，
#: 但**每个名字都必须是规范名** —— 守卫在下面。
SHORTLIST_ORDER = ["16-智能体工程", "00-电商Agent", "05-推荐系统", "13-广告分析", "02-A_B实验",
                   "01-因果推断", "03-时间序列", "04-供应链", "06-增长模型", "07-NLP-VOC",
                   "08-知识图谱", "09-DataAgent-LLM", "10-MAS", "14-用户分析", "15-营销投放分析",
                   "12-ML基础", "11-AI人文"]

# ---------- A. 发表层级 ----------
TIER_A_DEFAULT = {
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
TIER_A = DIM.get("venue", {}).get("venue_aliases", TIER_A_DEFAULT)
WORKSHOP_HINTS = tuple(DIM.get("venue", {}).get(
    "workshop_hints",
    ["workshop", "tutorial", "findings", "demo track", "industry track", "challenge"]))
NEG_HINTS = ("arxiv preprint", "under review", "preprint", "submitted to", "to appear")

# ---------- C. 业务相关性 ----------
BUSINESS_STRONG_DEFAULT = [
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
BUSINESS_WEAK_DEFAULT = [
    "user behavior", "click-through", "ctr", "personaliz", "cold-start", "session",
    "anomaly detection", "knowledge graph", "text-to-sql", "data analysis agent",
    "marketing", "consumer", "brand",
]
# 母婴品类锚点（命中则强加权）
CATEGORY_ANCHORS_DEFAULT = [
    "baby", "infant", "mother", "maternal", "toddler", "nursery", "stroller",
    "breast pump", "diaper", "formula", "feeding", "carseat", "car seat",
]

# ---------- D. 方法可萃取性 ----------
METHOD_POS_DEFAULT = [
    "we propose", "we introduce", "we present", "our method", "our approach", "our framework",
    "algorithm", "framework", "architecture", "model we", "we develop", "we design",
    "experiments", "experimental results", "benchmark", "dataset", "ablation",
    "outperform", "state-of-the-art", "sota", "baseline", "evaluation on",
]
METHOD_NEG_DEFAULT = [
    "survey", "systematic review", "literature review", "meta-analysis", "position paper",
    "we argue", "call for", "roadmap", "tutorial", "overview of", "taxonomy of existing",
    "research agenda", "perspective paper", "soK:", "sok ",
]

GAP_KEYWORDS_DEFAULT = {
    "广告归因/增量": ["attribution", "incrementality", "geo experiment", "marketing mix", "mmm"],
    "实验平台/序贯检验": ["sequential test", "always valid", "switchback", "crossover", "variance reduction", "cuped"],
    "选品/组合优化": ["assortment", "product selection", "new product", "category management"],
    "供应链-LLM": ["supply chain", "procurement", "supplier", "lead time"],
    "推荐-生成式": ["generative recommendation", "semantic id", "llm ranker", "reranker"],
    "Agent技能/上下文": ["agent skill", "skill library", "context compression", "context engineering", "memory"],
    "评价/自动化评测": ["llm-as-a-judge", "judge", "rubric", "evaluation harness", "benchmark"],
    "跨境合规/多语言": ["multilingual", "cross-lingual", "compliance", "regulation", "customs"],
}


# ---------- 配置解析后的生效常量（默认值 = 重构前硬编码的那一套）----------
BUSINESS_STRONG = DIM.get("business", {}).get("strong_keywords", BUSINESS_STRONG_DEFAULT)
BUSINESS_WEAK = DIM.get("business", {}).get("weak_keywords", BUSINESS_WEAK_DEFAULT)
CATEGORY_ANCHORS = DIM.get("business", {}).get("category_anchors", CATEGORY_ANCHORS_DEFAULT)
METHOD_POS = DIM.get("method", {}).get("positive", METHOD_POS_DEFAULT)
METHOD_NEG = DIM.get("method", {}).get("negative", METHOD_NEG_DEFAULT)
GAP_KEYWORDS = DIM.get("gap", {}).get("keywords", GAP_KEYWORDS_DEFAULT)

TOP_VENUES = set(DIM.get("venue", {}).get("top_venues", [
    "NeurIPS", "ICML", "ICLR", "KDD", "SIGIR", "WWW", "WSDM", "RecSys", "ACL",
    "EMNLP", "NAACL", "AAAI", "IJCAI", "AAMAS", "COLM", "VLDB", "SIGMOD",
    "Management Science", "Marketing Science", "Operations Research", "M&SOM",
    "MIS Quarterly", "ISR", "TKDE", "TOIS", "TIST", "UAI", "AISTATS", "MLSys"]))

V = DIM.get("venue", {})
C = DIM.get("code", {})
B = DIM.get("business", {})
M = DIM.get("method", {})
F = DIM.get("freshness", {})
G = DIM.get("gap", {})


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
    best = sorted(hits, key=lambda h: (h not in TOP_VENUES, h))[0]
    if best in TOP_VENUES:
        return (V.get("tier_top_workshop", 18) if is_ws
                else V.get("tier_top", 30)), best
    return (V.get("tier_other_workshop", 10) if is_ws
            else V.get("tier_other", 18)), best


def score(item: dict) -> dict:
    text = f"{item['title']} . {item['abstract']} . {item.get('comment','')}"
    low = text.lower()
    title_low = item["title"].lower()

    tier, venue = venue_of(item)

    # B. 代码
    code = 0
    if re.search(C.get("strong_pattern",
                       r"github\.com|huggingface\.co|code is available|our code|code will be|open-source"), low):
        code += C.get("strong_points", 11)
    if re.search(C.get("weak_pattern", r"open[- ]sourc|we release|publicly available|repository"), low):
        code += C.get("weak_points", 4)
    code = min(code, C.get("max", 15))

    # C. 业务
    strong_hits = [k for k in BUSINESS_STRONG if k in low]
    weak_hits = [k for k in BUSINESS_WEAK if k in low]
    anchors = [k for k in CATEGORY_ANCHORS if k in low]
    business = min(B.get("strong_cap", 14),
                   B.get("strong_points_each", 3.5) * len(set(strong_hits)))
    business += min(B.get("weak_cap", 6),
                    B.get("weak_points_each", 1.2) * len(set(weak_hits)))
    if anchors:
        business += B.get("category_anchor_bonus", 4)
    business = min(B.get("max", 20), round(business, 1))

    # D. 方法
    pos = sum(1 for k in METHOD_POS if k in low)
    neg = sum(1 for k in METHOD_NEG if k in low)
    method = min(M.get("max", 20), M.get("positive_points_each", 3.0) * pos)
    if neg:
        method = max(0, method - M.get("negative_penalty_each", 8.0) * neg)
    if re.search(r"\babstract\b.{0,80}\b(survey|review)\b", low) or \
            title_low.startswith(tuple(M.get("survey_title_prefixes",
                                              ["a survey", "survey", "a review", "systematic review"]))):
        method = min(method, M.get("survey_cap", 4))
    method = round(method, 1)

    # E. 时效
    try:
        pub = datetime.fromisoformat(item["published"].replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - pub).days
    except Exception:
        days = F.get("fallback_days", 92)
    fresh = round(max(0, F.get("base", 8) - days / F.get("days_divisor", 12)), 1)

    # F. 缺口
    gap_hits = [g for g, kws in GAP_KEYWORDS.items() if any(k in low for k in kws)]
    gap = min(G.get("max", 7), G.get("points_each", 2.5) * len(gap_hits))

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
    order = CFG.get("shortlist_domain_order", SHORTLIST_ORDER)
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
    print("分数分布:", {f"≥{t}": sum(1 for r in rows if r["score"] >= t)
                      for t in (70, 60, 50, 40, 30)})
    print(f"阈值（scoring_config.json）: P0≥{THRESH.get('p0', 60)} "
          f"P1≥{THRESH.get('p1', 40)} P2≥{THRESH.get('p2', 30)}"
          + ("" if CFG else "   ⚠️ 未找到配置文件，正在使用内置默认值"))


if __name__ == "__main__":
    main()

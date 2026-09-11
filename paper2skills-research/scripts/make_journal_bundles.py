#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""期刊候选 → 分组 bundle（供 LLM 精读）。"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

GROUPS = {
    "期刊A-广告投放与定价促销": ["advertis", "sponsor", "auction", "bidding", "attribution", "campaign",
                                  "pricing", "price", "discount", "promotion", "coupon", "subscription",
                                  "media mix", "marketing mix", "roas", "retarget"],
    "期刊B-库存履约与供应链": ["inventory", "stock", "replenish", "fulfill", "warehouse", "lead time",
                                 "assortment", "logistics", "shipping", "return", "supply chain", "procurement",
                                 "omnichannel", "newsvendor", "capacity"],
    "期刊C-推荐搜索与平台卖家": ["recommend", "ranking", "search", "retrieval", "personaliz", "cold-start",
                                   "cold start", "marketplace", "platform", "seller", "merchant", "e-commerce",
                                   "ecommerce", "online retail", "two-sided", "review", "rating", "reputation"],
    "期刊D-因果实验与增长留存": ["experiment", "a/b", "causal", "treatment effect", "uplift", "incrementality",
                                   "field experiment", "randomized", "churn", "retention", "customer lifetime",
                                   "clv", "ltv", "engagement", "loyalty", "referral", "nudge", "generative ai",
                                   "large language model", "llm", "agent"],
}
TOP_TIERS = ("UTD24", "FT50", "CCF-A", "顶刊")


def main() -> None:
    d = json.loads((DATA / "journal_candidates.json").read_text(encoding="utf-8"))
    items = [i for i in d["items"] if i["score_hint"] > 0 and not i["noise"]]

    # —— 修复：Crossref 多关键词抓取会重复计数（实测 120 条里 29 条重复 = 24%）
    # 先按 DOI 去重，再排序，否则同一篇论文会占掉多个分组的 Top 位
    dedup: dict[str, dict] = {}
    for i in items:
        key = (i.get("doi") or i["title"]).lower().strip()
        if key not in dedup or len(i["biz_hits"]) > len(dedup[key]["biz_hits"]):
            dedup[key] = i
    dup_removed = len(items) - len(dedup)
    items = list(dedup.values())

    # —— 修复：解析 DOI 中的投稿年份，"2026 年在线发表"常对应 2-3 年前投稿
    # 规则：DOI 年份 < 发表年份 - 1 时标记 stale_method，提示卡片写清"结论年份 vs 数据年份"
    doi_year_re = re.compile(r"/(?:mnsc|mksc|isre|msom|opre|jasa|jorm|jmkt|decz|ijec|mnre)\.[a-z]*\.?(20\d{2})\.|/(20\d{2})\.")
    for i in items:
        m = doi_year_re.search(i.get("doi", ""))
        y = int(m.group(1) or m.group(2)) if m else None
        pub_y = None
        try:
            pub_y = int(str(i.get("published", ""))[:4])
        except Exception:
            pass
        i["doi_year"] = y
        i["stale_method"] = bool(y and pub_y and pub_y - y >= 2)
    stale_n = sum(1 for i in items if i.get("stale_method"))
    out_dir = DATA / "bundles"
    out_dir.mkdir(exist_ok=True)
    index = {}
    for g, kws in GROUPS.items():
        sub = [i for i in items if any(k in i["title"].lower() for k in kws)]
        # 排序：UTD24/FT50/CCF-A 优先，再按 biz_hits 与日期
        sub.sort(key=lambda x: (not any(t in x["tier"] for t in TOP_TIERS), -x["score_hint"], x["published"]), reverse=False)
        top = [i for i in sub if any(t in i["tier"] for t in TOP_TIERS)]
        keep = (top + [i for i in sub if i not in top])[:30]
        payload = {
            "group": g, "window": d["window"], "total_hits": len(sub),
            "tier_a_hits": len(top), "listed": len(keep),
            "doi_duplicates_removed": dup_removed,
            "stale_method_hits": stale_n,
            "papers": [
                {
                    "journal": i["journal"], "tier": i["tier"], "published": i["published"],
                    "title": i["title"], "doi": i["doi"], "url": i["url"],
                    "doi_year": i.get("doi_year"), "stale_method": i.get("stale_method", False),
                    "biz_hits": i["biz_hits"], "authors": i["authors"][:3],
                }
                for i in keep
            ],
        }
        fp = out_dir / f"{g}.json"
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        index[g] = {"total_hits": len(sub), "tier_a_hits": len(top), "listed": len(keep), "file": fp.name}
    (out_dir / "_journal_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顶级期刊近 N 天论文收割器（Crossref 路线，已验证可用）

为什么需要它:
    paper2skills 原来只搜 arXiv。但"顶级期刊"（UTD24 / FT50 / CCF-A 期刊）里
    大量高业务价值论文（库存、定价、促销、因果、实验）不上 arXiv。
    Crossref 的 journals/{issn}/works + from-pub-date 过滤器可以精确拿到
    某个刊在指定时间窗内**正式在线发表**的全部文章，这是最可靠的期刊更新路线。

用法:
    python3 journal_harvest.py --days 92
输出:
    data/journal_candidates.json   全量（标题/期刊/日期/DOI/链接/命中关键词）
    data/journal_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UA = {"User-Agent": "paper2skills-research/1.0 (mailto:research@paper2skills.local)"}

# 期刊清单：领域 -> [(刊名, ISSN, 层级标签)]
JOURNALS: dict[str, list[tuple[str, str, str]]] = {
    "运营/供应链": [
        ("Management Science", "0025-1909", "UTD24/FT50"),
        ("M&SOM", "1523-4614", "UTD24"),
        ("Operations Research", "0030-364X", "UTD24"),
        ("Production and Operations Management", "1059-1478", "ABS4"),
        ("Journal of Operations Management", "0272-6963", "UTD24"),
        ("European Journal of Operational Research", "0377-2217", "ABS4/CCF-B"),
        ("International Journal of Forecasting", "0169-2070", "ABS3/预测顶刊"),
        ("Decision Support Systems", "0167-9236", "ABS3/CCF-B"),
    ],
    "营销/定价/促销": [
        ("Marketing Science", "0732-2399", "UTD24/FT50"),
        ("Journal of Marketing", "0022-2429", "UTD24/FT50"),
        ("Journal of Marketing Research", "0022-2437", "UTD24/FT50"),
        ("Journal of Retailing", "0022-4359", "ABS4"),
        ("Quantitative Marketing and Economics", "1570-7156", "ABS3"),
        ("Marketing Letters", "0923-0645", "ABS3"),
        ("Journal of Interactive Marketing", "1094-9968", "ABS3"),
        ("Journal of Business Research", "0148-2963", "ABS3"),
        ("Electronic Commerce Research", "1389-5753", "ABS2/电商专刊"),
    ],
    "信息系统/电商": [
        ("MIS Quarterly", "0276-7783", "UTD24/FT50"),
        ("Information Systems Research", "1047-7047", "UTD24/FT50"),
        ("ACM TOIS", "1046-8188", "CCF-A 期刊"),
        ("IEEE TKDE", "1041-4347", "CCF-A 期刊"),
        ("ACM TIST", "2157-6904", "CCF-B 期刊"),
        ("VLDB Journal", "1066-8888", "CCF-A 期刊"),
        ("Information Processing & Management", "0306-4573", "CCF-B 期刊"),
        ("IEEE TNNLS", "2162-237X", "CCF-B 期刊"),
    ],
    "统计/计量": [
        ("Journal of Econometrics", "0304-4076", "计量顶刊"),
        ("Annals of Statistics", "0090-5364", "统计顶刊"),
        ("JASA", "0162-1459", "统计顶刊"),
    ],
}

# 业务相关性关键词（标题命中即保留；未命中则进全量池，标记未命中）
BIZ_KEYWORDS = [
    "recommend", "ranking", "search", "retrieval", "personaliz", "cold start", "cold-start",
    "advertis", "sponsor", "auction", "bidding", "budget", "attribution", "campaign",
    "pricing", "price", "discount", "promotion", "coupon", "bundle", "subscription",
    "inventory", "stock", "replenish", "assortment", "supply chain", "logistic",
    "fulfill", "warehouse", "lead time", "demand forecast", "forecast",
    "churn", "retention", "lifetime value", "clv", "ltv", "conversion", "funnel",
    "experiment", "a/b", "ab test", "causal", "treatment effect", "uplift", "heterogeneous",
    "review", "sentiment", "consumer", "customer", "shopper", "platform", "marketplace",
    "e-commerce", "ecommerce", "online retail", "digital", "mobile", "app",
    "generative ai", "large language model", "llm", "agent", "machine learning",
    "deep learning", "graph neural", "bandit", "reinforcement learning",
]
# 反噪声：命中则降权（与业务无关的编辑部内容等）
NOISE = ["editorial board", "erratum", "corrigendum", "retraction", "in this issue",
         "call for papers", "acknowledgement to referees", "thanks to reviewers",
         "front matter", "back matter", "index", "table of contents", "president's message"]


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t or "")).strip()


def fetch_journal(name: str, issn: str, tier: str, frm: str, to: str, rows: int) -> list[dict]:
    url = f"https://api.crossref.org/journals/{issn}/works"
    params = {
        "filter": f"from-pub-date:{frm},until-pub-date:{to},type:journal-article",
        "rows": rows,
        "sort": "published",
        "order": "desc",
        "select": "title,container-title,published,DOI,URL,abstract,author,subject,page,volume,issue,type",
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=60, headers=UA)
            if r.status_code == 200:
                return r.json()["message"]["items"]
            print(f"    HTTP {r.status_code}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERR {exc}", file=sys.stderr)
        time.sleep(3 * (attempt + 1))
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=92)
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=1.2)
    args = ap.parse_args()

    now = datetime.now()
    frm = (now - timedelta(days=args.days)).date().isoformat()
    to = now.date().isoformat()
    print(f"期刊窗口: {frm} -> {to}")

    out: list[dict] = []
    for domain, jlist in JOURNALS.items():
        for name, issn, tier in jlist:
            items = fetch_journal(name, issn, tier, frm, to, args.rows)
            kept = 0
            for it in items:
                title = norm_title((it.get("title") or [""])[0])
                if not title:
                    continue
                tl = title.lower()
                hits = [k for k in BIZ_KEYWORDS if k in tl]
                noise = [k for k in NOISE if k in tl]
                doi = it.get("DOI", "")
                pub = (it.get("published", {}).get("date-parts") or [[None]])[0]
                pub_s = "-".join(str(x) for x in pub if x)
                out.append({
                    "source": "crossref",
                    "journal": name,
                    "issn": issn,
                    "tier": tier,
                    "domain": domain,
                    "title": title,
                    "doi": doi,
                    "url": it.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                    "published": pub_s,
                    "volume": it.get("volume", ""), "issue": it.get("issue", ""),
                    "page": it.get("page", ""),
                    "authors": [f"{a.get('given','')} {a.get('family','')}".strip()
                                for a in (it.get("author") or [])][:6],
                    "biz_hits": hits,
                    "noise": noise,
                    "score_hint": len(hits) - 5 * len(noise),
                    "abstract_present": bool(it.get("abstract")),
                })
                kept += 1
            print(f"  {domain:12s} {name:45s} {tier:12s} {kept:4d} 篇")
            time.sleep(args.sleep)

    out.sort(key=lambda x: (-x["score_hint"], x["journal"]))
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "journal_candidates.json").write_text(
        json.dumps({"generated_at": now.isoformat(timespec="seconds"),
                    "window": {"start": frm, "end": to}, "count": len(out), "items": out},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    with (DATA / "journal_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["score_hint", "journal", "tier", "domain", "published", "title", "biz_hits", "doi", "url"])
        for r in out:
            w.writerow([r["score_hint"], r["journal"], r["tier"], r["domain"], r["published"],
                        r["title"], "|".join(r["biz_hits"]), r["doi"], r["url"]])
    hit = [r for r in out if r["score_hint"] > 0]
    print(f"\n完成: {len(out)} 篇（其中标题命中业务关键词 {len(hit)} 篇）")
    print("Top 25 业务相关:")
    for r in hit[:25]:
        print(f"  [{r['tier'][:9]:<9}] {r['journal'][:32]:<32} {r['published']:<10} {r['title'][:72]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

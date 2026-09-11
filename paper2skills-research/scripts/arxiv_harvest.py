#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arXiv 近三个月候选论文收割器（paper2skills 项目专用）

用法:
    python3 arxiv_harvest.py [--days 90] [--max-per-query 60] [--sleep 3.0]

输出:
    data/arxiv_candidates.json   全量候选（含摘要、分类、日期、链接）
    data/arxiv_candidates.csv    精简清单（便于人工/表格筛选）

设计要点:
- 按 paper2skills 的 16 个业务领域 × 关键词分组构造查询
- 用 arXiv 官方 API 的 submittedDate 区间过滤（而不是只取最新 N 条再本地过滤）
- 请求间强制 sleep，避免 429
- 结果按 arxiv id 去重，并记录命中的查询组
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# 业务领域 -> 关键词组（每组一条查询，组内 AND，组间 OR 由 arXiv 语法决定）
QUERY_GROUPS: dict[str, list[str]] = {
    "01-因果推断": [
        'abs:"uplift modeling" OR abs:"heterogeneous treatment effect"',
        'abs:"causal inference" AND abs:"e-commerce"',
        'abs:"causal" AND abs:"marketing" AND abs:"incrementality"',
    ],
    "02-A_B实验": [
        'abs:"A/B testing" OR abs:"online controlled experiment"',
        'abs:"multi-armed bandit" AND abs:"recommendation"',
        'abs:"sequential testing" OR abs:"experiment assignment"',
    ],
    "03-时间序列": [
        'abs:"demand forecasting" AND (cat:cs.LG OR cat:stat.ML)',
        'abs:"time series" AND abs:"foundation model"',
        'abs:"intermittent demand" OR abs:"hierarchical forecasting"',
    ],
    "04-供应链": [
        'abs:"inventory management" AND (abs:"reinforcement learning" OR abs:"deep learning")',
        'abs:"supply chain" AND abs:"large language model"',
        'abs:"replenishment" OR abs:"newsvendor"',
    ],
    "05-推荐系统": [
        'abs:"recommender system" AND abs:"large language model"',
        'abs:"sequential recommendation" OR abs:"generative recommendation"',
        'abs:"cold-start" AND abs:"recommendation"',
    ],
    "06-增长模型": [
        'abs:"churn prediction" OR abs:"customer lifetime value"',
        'abs:"user growth" OR abs:"retention" AND abs:"uplift"',
        'abs:"conversion rate prediction" AND abs:"e-commerce"',
    ],
    "07-VOC舆情": [
        'abs:"aspect-based sentiment" OR abs:"opinion mining"',
        'abs:"customer review" AND (abs:"LLM" OR abs:"large language model")',
        'abs:"review summarization" OR abs:"user feedback" AND abs:"mining"',
    ],
    "08-知识图谱": [
        'abs:"knowledge graph" AND abs:"retrieval augmented"',
        'abs:"graph neural network" AND abs:"e-commerce"',
        'abs:"entity alignment" OR abs:"knowledge graph completion"',
    ],
    "09-DataAgent": [
        'abs:"data analysis agent" OR abs:"text-to-SQL"',
        'abs:"autonomous data science" OR abs:"insight generation"',
        'abs:"table" AND abs:"agent" AND abs:"reasoning"',
    ],
    "10-MAS": [
        'abs:"multi-agent" AND abs:"LLM" AND abs:"coordination"',
        'abs:"agent" AND abs:"simulation" AND abs:"consumer"',
        'abs:"multi-agent reinforcement learning" AND abs:"pricing"',
    ],
    "11-AI人文": [
        'abs:"continual learning" AND abs:"LoRA"',
        'abs:"cross-modal" AND abs:"transfer" AND abs:"humanities"',
    ],
    "12-ML基础": [
        'abs:"feature engineering" AND abs:"tabular"',
        'abs:"tabular foundation model" OR abs:"tabular deep learning"',
    ],
    "13-广告分析": [
        'abs:"advertising" AND (abs:"auction" OR abs:"bidding")',
        'abs:"attribution" AND abs:"marketing"',
        'abs:"budget allocation" AND abs:"ROI"',
    ],
    "14-用户分析": [
        'abs:"funnel" AND abs:"conversion"',
        'abs:"cohort" OR abs:"survival analysis" AND abs:"user"',
    ],
    "15-营销投放": [
        'abs:"marketing mix model" OR abs:"media mix"',
        'abs:"promotion" AND abs:"causal" AND abs:"retail"',
    ],
    "16-智能体工程": [
        'abs:"agent skills" OR abs:"skill library" AND abs:"LLM"',
        'abs:"context engineering" OR abs:"context compression" AND abs:"agent"',
        'abs:"model context protocol" OR abs:"agent2agent" OR abs:"tool use" AND abs:"benchmark"',
        'abs:"agent memory" AND abs:"long-term"',
    ],
    # 跨域高价值：电商/跨境/母婴
    "00-电商Agent": [
        'abs:"e-commerce" AND abs:"agent"',
        'abs:"shopping" AND abs:"LLM" AND abs:"agent"',
        'abs:"cross-border" AND abs:"commerce"',
        'abs:"product selection" OR abs:"assortment optimization"',
    ],
}

CATEGORY_FILTER = (
    "cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.IR OR cat:cs.DB OR "
    "cat:cs.MA OR cat:cs.SI OR cat:stat.ML OR cat:stat.AP OR cat:stat.ME OR "
    "cat:math.OC OR cat:econ.EM OR cat:q-fin.GN OR cat:cs.CY"
)


def build_query(group: str, start: str, end: str) -> str:
    return f'({group}) AND ({CATEGORY_FILTER}) AND submittedDate:[{start} TO {end}]'


def fetch(query: str, max_results: int, retries: int = 3) -> str | None:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "paper2skills-research/1.0"})
            if r.status_code == 200:
                return r.text
            print(f"    HTTP {r.status_code}, retry {attempt + 1}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"    ERR {exc}, retry {attempt + 1}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    return None


def parse(xml: str, group: str) -> list[dict]:
    out: list[dict] = []
    root = ET.fromstring(xml)
    for e in root.findall("atom:entry", NS):
        raw_id = e.findtext("atom:id", default="", namespaces=NS)
        m = re.search(r"abs/([0-9]{4}\.[0-9]{4,5})", raw_id or "")
        if not m:
            continue
        aid = m.group(1)
        title = " ".join((e.findtext("atom:title", default="", namespaces=NS) or "").split())
        summary = " ".join((e.findtext("atom:summary", default="", namespaces=NS) or "").split())
        published = e.findtext("atom:published", default="", namespaces=NS) or ""
        updated = e.findtext("atom:updated", default="", namespaces=NS) or ""
        authors = [
            a.findtext("atom:name", default="", namespaces=NS)
            for a in e.findall("atom:author", NS)
        ]
        cats = [c.attrib.get("term", "") for c in e.findall("atom:category", NS)]
        primary = e.find("arxiv:primary_category", NS)
        comment = e.findtext("arxiv:comment", default="", namespaces=NS) or ""
        jref = e.findtext("arxiv:journal_ref", default="", namespaces=NS) or ""
        out.append(
            {
                "arxiv_id": aid,
                "title": title,
                "abstract": summary,
                "published": published,
                "updated": updated,
                "authors": authors[:6],
                "n_authors": len(authors),
                "categories": cats,
                "primary_category": primary.attrib.get("term", "") if primary is not None else "",
                "comment": " ".join(comment.split()),
                "journal_ref": " ".join(jref.split()),
                "url": f"https://arxiv.org/abs/{aid}",
                "pdf": f"https://arxiv.org/pdf/{aid}",
                "query_groups": [group],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--max-per-query", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--out", default="arxiv_candidates")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=args.days)
    start = start_dt.strftime("%Y%m%d%H%M")
    end = now.strftime("%Y%m%d%H%M")
    print(f"窗口: {start} -> {end}  (近 {args.days} 天)")

    store: dict[str, dict] = {}
    total_queries = sum(len(v) for v in QUERY_GROUPS.values())
    n = 0
    for domain, groups in QUERY_GROUPS.items():
        for g in groups:
            n += 1
            q = build_query(g, start, end)
            print(f"[{n}/{total_queries}] {domain}: {g[:70]}")
            xml = fetch(q, args.max_per_query)
            if not xml:
                print("    -> 失败", file=sys.stderr)
                continue
            items = parse(xml, domain)
            new = 0
            for it in items:
                aid = it["arxiv_id"]
                if aid in store:
                    if domain not in store[aid]["query_groups"]:
                        store[aid]["query_groups"].append(domain)
                else:
                    store[aid] = it
                    new += 1
            print(f"    -> {len(items)} 条，新增 {new}，累计 {len(store)}")
            time.sleep(args.sleep)

    items = sorted(store.values(), key=lambda x: x["published"], reverse=True)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{args.out}.json").write_text(
        json.dumps(
            {
                "generated_at": now.isoformat(),
                "window": {"start": start_dt.date().isoformat(), "end": now.date().isoformat()},
                "count": len(items),
                "items": items,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    with (DATA / f"{args.out}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arxiv_id", "published", "primary_category", "domains", "title", "journal_ref", "url"])
        for it in items:
            w.writerow(
                [
                    it["arxiv_id"],
                    it["published"][:10],
                    it["primary_category"],
                    "|".join(it["query_groups"]),
                    it["title"],
                    it["journal_ref"],
                    it["url"],
                ]
            )
    print(f"\n完成: {len(items)} 篇候选 -> data/{args.out}.json / .csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

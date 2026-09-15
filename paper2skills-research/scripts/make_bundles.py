#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按业务领域生成候选 bundle（供 LLM 深度精读排序用）。"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOMAINS = ["16-智能体工程", "00-电商Agent", "05-推荐系统", "13-广告分析", "02-A_B实验",
           "01-因果推断", "03-时间序列", "04-供应链", "06-增长模型", "07-NLP-VOC",
           "08-知识图谱", "09-DataAgent-LLM", "10-MAS", "14-用户分析", "15-营销投放分析",
           "12-ML基础", "11-AI人文"]

TOPN_PER_DOMAIN = 22


def brief(it: dict) -> dict:
    abstract = re.sub(r"\s+", " ", it["abstract"])
    return {
        "arxiv_id": it["arxiv_id"],
        "title": it["title"],
        "date": it["published"][:10],
        "venue": it.get("venue") or "",
        "raw_venue_hint": (it.get("comment") or "")[:120] or (it.get("journal_ref") or "")[:120],
        "primary_category": it["primary_category"],
        "score": it["score"],
        "s_business": it["s_business"],
        "s_method": it["s_method"],
        "s_code": it["s_code"],
        "abstract": abstract[:900],
        "url": it["url"],
    }


def main() -> None:
    rec = json.loads((DATA / "recommendations.json").read_text(encoding="utf-8"))
    items = rec["items"]
    out_dir = DATA / "bundles"
    out_dir.mkdir(exist_ok=True)
    index = {}
    for dom in DOMAINS:
        sub = [i for i in items if dom in i["query_groups"]]
        sub.sort(key=lambda r: -r["score"])
        sel = sub[:TOPN_PER_DOMAIN]
        payload = {
            "domain": dom,
            "window": rec["window"],
            "hit_total": len(sub),
            "top_by_score": len(sel),
            "papers": [brief(i) for i in sel],
        }
        fp = out_dir / f"{dom}.json"
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        index[dom] = {"hit_total": len(sub), "file": str(fp.name), "size_kb": fp.stat().st_size // 1024}
    (out_dir / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

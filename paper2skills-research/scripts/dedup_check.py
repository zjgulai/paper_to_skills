#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
去重检查器：候选论文 vs 存量 Skill 卡片

三层去重（对应方案 Step 3）：
  L1 硬 ID 命中  —— 候选 arXiv ID 是否已在某张卡片中被引用过
  L2 方法名近重  —— 候选标题的方法名 token 是否与已有卡片名高度重合
  L3 语义簇提示  —— 候选标题/摘要的方法关键词是否落在已有卡片的主题簇里（需 LLM 二次确认）

输入:  data/recommendations.json（或 journal_candidates.json）、paper2skills-vault/**/Skill-*.md
输出:  data/dedup_report.json + 控制台摘要
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper2skills-research" / "data"
VAULT = ROOT / "paper2skills-vault"
OUT = DATA / "dedup_report.json"

ARXIV_RE = re.compile(r"arXiv[:\s]*([0-9]{4}\.[0-9]{4,5})|arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})")
DOI_RE = re.compile(r"10\.[0-9]{4,9}/[^\s)\]|]+")
STOP = {"skill", "card", "the", "for", "and", "with", "based", "learning", "model", "models",
        "analysis", "prediction", "optimization", "using", "via", "towards", "toward", "a", "an"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def tokens(s: str) -> set[str]:
    ws = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", s)
    return {w.lower().strip("-") for w in ws if w.lower() not in STOP}


def load_cards() -> list[dict]:
    cards = []
    seen = set()
    for p in sorted(VAULT.rglob("Skill-*.md")):
        if "node_modules" in p.parts or p.stem in seen:
            continue
        seen.add(p.stem)
        text = p.read_text(encoding="utf-8", errors="ignore")
        fm_paper = ""
        m = re.search(r"^paper:\s*(.+)$", text, flags=re.M)
        if m:
            fm_paper = m.group(1).strip().strip('"')
        ids = {a or b for a, b in ARXIV_RE.findall(text)}
        if fm_paper:
            ids |= {a or b for a, b in ARXIV_RE.findall(fm_paper)}
        cards.append({
            "file": str(p.relative_to(ROOT)),
            "stem": p.stem,
            "arxiv_ids": sorted(i for i in ids if i),
            "dois": sorted(set(DOI_RE.findall(text)))[:5],
            "name_tokens": tokens(p.stem.replace("Skill-", "")),
            "topic": (re.search(r"^topic:\s*(.+)$", text, flags=re.M) or [None, ""])[1].strip(),
        })
    return cards


def main() -> None:
    rec = json.loads((DATA / "recommendations.json").read_text(encoding="utf-8"))
    cands = rec["items"]
    cards = load_cards()
    card_ids = {i: c["file"] for c in cards for i in c["arxiv_ids"]}

    l1, l2 = [], []
    for it in cands[:400]:  # 只对前 400 名做近重检查，避免噪声
        aid = it["arxiv_id"]
        if aid in card_ids:
            l1.append({"arxiv_id": aid, "title": it["title"], "card": card_ids[aid]})
        ct = tokens(it["title"])
        for c in cards:
            inter = ct & c["name_tokens"]
            if len(inter) >= 3:
                l2.append({
                    "arxiv_id": aid, "title": it["title"], "card": c["file"],
                    "shared_tokens": sorted(inter), "overlap": round(len(inter) / max(len(c["name_tokens"]), 1), 2),
                })

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cards_scanned": len(cards),
        "cards_with_arxiv": sum(1 for c in cards if c["arxiv_ids"]),
        "candidates_scanned": len(cands),
        "l1_hard_id_collisions": l1,
        "l2_method_name_near_dup": l2,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"存量卡片: {len(cards)}（其中 {out['cards_with_arxiv']} 张带 arXiv ID）")
    print(f"候选(前400): L1 硬 ID 命中 {len(l1)} 条；L2 方法名近重 {len(l2)} 组")
    if l1:
        print("\nL1 硬命中（同一篇论文已有卡片）:")
        for x in l1[:20]:
            print(f"  {x['arxiv_id']} {x['title'][:60]}  ->  {x['card']}")
    if l2:
        print("\nL2 方法名近重（Top 20，需人工/LLM 确认是否为同一方法）:")
        for x in sorted(l2, key=lambda y: -y["overlap"])[:20]:
            print(f"  [{x['overlap']:.2f}] {x['arxiv_id']} {x['title'][:52]} ~ {Path(x['card']).stem}")
            print(f"         共享词: {', '.join(x['shared_tokens'])}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

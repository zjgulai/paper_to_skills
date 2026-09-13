#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE6 · S10 · venue 证据采集器（铁律 1 的仪器层）

采集两样东西，**都不看论文底本正文**：

  ① arXiv abs 页（`https://arxiv.org/abs/<id>`）的 `Comments:` / `journal_ref` / `DOI`
     —— 输出 `data/venue_sources/arxiv_abs.json`
  ② 出版方 DOI 在 Crossref 的解析（`container-title` / `event` / `page` /
     `published-online` / `published-print`）—— 输出 `data/venue_sources/crossref.json`

用法：
    python3 fetch_venue_sources.py --ids-from-cards        # 从卡片 paper_id 收集
    python3 fetch_venue_sources.py --ids 2608.22152 ...    # 显式给
    python3 fetch_venue_sources.py --refresh-crossref      # 只补 Crossref

礼貌与可复跑：
  · 每次请求间隔 ≥ 1.5s；arXiv 失败重试 3 次（指数退避），**不无限循环**
  · 输出**增量写盘**（每抓一条就落一次），中断后重跑只补缺的
  · 单条超时 30s（curl `--max-time`）

⚠️ arXiv API（`export.arxiv.org/api/query`）在本机实测返回 `Rate exceeded.`
   （HTTP 200 + 13 字节文本）——**它是 HTTP 200 的失败**，脚本若不检查正文会把
   这 13 字节当数据。故本采集器走**abs 页 HTML**，并对 `len(body) > 5000` 做硬校验。
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAULT = REPO / "paper2skills-vault"
OUTDIR = REPO / "paper2skills-research" / "data" / "venue_sources"
ARXIV_OUT = OUTDIR / "arxiv_abs.json"
CROSSREF_OUT = OUTDIR / "crossref.json"

UA = "Mozilla/5.0 (compatible; paper2skills-venue-backfill/1.0; +local research)"
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
DOI_IN_TEXT_RE = re.compile(r"10\.\d{4,9}/\S+")


def curl(url: str, *, tries: int = 3, timeout: int = 30, min_len: int = 0) -> tuple[int, str]:
    """带重试的 curl。返回 (rc, body)。**检查正文长度**，不只看状态码。"""
    last = ""
    for attempt in range(tries):
        p = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", str(timeout), "-A", UA, url],
            capture_output=True,
        )
        body = p.stdout.decode("utf-8", "replace")
        if p.returncode == 0 and len(body) >= max(min_len, 2):
            return 0, body
        last = body or p.stderr.decode("utf-8", "replace")
        time.sleep(2 + 2 * attempt)
    return 1, last


def _cell(t: str, cls: str) -> str:
    m = re.search(r'<td class="tablecell %s[^"]*">(.*?)</td>' % cls, t, re.S)
    if not m:
        return ""
    return html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", "", m.group(1)))).strip()


def parse_abs(t: str) -> dict:
    m = re.search(r"<title>(.*?)</title>", t, re.S)
    page_title = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip() if m else ""
    m = re.search(r'<meta name="citation_title" content="([^"]*)"', t)
    return {
        "page_title": page_title,
        "citation_title": html.unescape(m.group(1)) if m else "",
        "comments": _cell(t, "comments"),
        "jref": _cell(t, "jref"),
        "doi": _cell(t, "doi"),
        "authors": re.findall(r'<meta name="citation_author" content="([^"]*)"', t)[:8],
        "len": len(t),
    }


def fetch_arxiv(ids: list[str], *, refresh: bool = False) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(ARXIV_OUT.read_text(encoding="utf-8")) if ARXIV_OUT.exists() else {}
    if refresh:
        data = {}
    todo = [i for i in ids if i not in data]
    print(f"arXiv abs：共 {len(ids)} 个号，已有 {len(data)}，待抓 {len(todo)}", file=sys.stderr)
    for n, aid in enumerate(todo):
        rc, body = curl(f"https://arxiv.org/abs/{aid}", min_len=5000)
        if rc != 0 or len(body) < 5000:
            data[aid] = {"error": f"fetch failed len={len(body)}"}
        else:
            data[aid] = parse_abs(body)
        print(f"  [{n+1}/{len(todo)}] {aid} -> {data[aid].get('page_title','')[:64]}", file=sys.stderr)
        ARXIV_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(1.5)
    return data


def crossref_of(doi: str) -> dict:
    rc, body = curl(f"https://api.crossref.org/works/{doi}", min_len=50)
    if rc != 0:
        return {"error": f"crossref fetch failed: {body[:200]}"}
    try:
        msg = json.loads(body)["message"]
    except Exception as exc:  # noqa: BLE001
        return {"error": f"crossref json: {exc}"}
    keep = [
        "container-title", "event", "page", "published-online", "published-print",
        "published", "type", "title", "publisher", "ISSN", "volume", "issue", "DOI",
    ]
    return {k: msg.get(k) for k in keep if msg.get(k) is not None}


def fetch_crossref(dois: list[str], *, refresh: bool = False) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(CROSSREF_OUT.read_text(encoding="utf-8")) if CROSSREF_OUT.exists() else {}
    if refresh:
        data = {}
    todo = [d for d in dois if d not in data]
    print(f"Crossref：共 {len(dois)} 个 DOI，已有 {len(data)}，待抓 {len(todo)}", file=sys.stderr)
    for n, doi in enumerate(todo):
        data[doi] = crossref_of(doi)
        ct = (data[doi].get("container-title") or [""])[0]
        print(f"  [{n+1}/{len(todo)}] {doi} -> {str(ct)[:70]}", file=sys.stderr)
        CROSSREF_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(1.5)
    return data


def ids_from_cards() -> list[str]:
    ids: list[str] = []
    for p in sorted(VAULT.rglob("Skill-*.md")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"^paper_id:\s*(\S+)", text, re.M)
        if m and ARXIV_ID_RE.match(m.group(1)):
            ids.append(m.group(1))
    if not ids:
        raise SystemExit("一张卡的 paper_id 都没读到 —— 「没东西可查」不等于「查过了没问题」")
    return sorted(set(ids))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-from-cards", action="store_true")
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--refresh", action="store_true", help="忽略既有缓存重抓 arXiv")
    ap.add_argument("--refresh-crossref", action="store_true")
    ap.add_argument("--crossref-only", action="store_true")
    args = ap.parse_args(argv)

    ids = args.ids or (ids_from_cards() if args.ids_from_cards else [])
    if not args.crossref_only:
        if not ids:
            print("没给号，也没 --ids-from-cards", file=sys.stderr)
            return 2
        arxiv = fetch_arxiv(ids, refresh=args.refresh)
    else:
        arxiv = json.loads(ARXIV_OUT.read_text(encoding="utf-8")) if ARXIV_OUT.exists() else {}

    dois = []
    for v in arxiv.values():
        m = DOI_IN_TEXT_RE.search(v.get("doi") or "")
        if m:
            dois.append(m.group(0).rstrip("."))
        for m2 in DOI_IN_TEXT_RE.finditer(v.get("jref") or ""):
            dois.append(m2.group(0).rstrip("."))
    # 卡片 paper_id 本身是 DOI 的也要抓
    for p in sorted(VAULT.rglob("Skill-*.md")):
        m = re.search(r"^paper_id:\s*(10\.\d{4,9}/\S+)", p.read_text(encoding="utf-8"), re.M)
        if m:
            dois.append(m.group(1).rstrip("."))
    dois = sorted(set(dois))
    if dois:
        fetch_crossref(dois, refresh=args.refresh_crossref)
    else:
        print("没有 DOI 可查 Crossref（不是失败）", file=sys.stderr)

    print(f"\n✅ arXiv {len(arxiv)} 条 → {ARXIV_OUT}")
    print(f"✅ Crossref {len(dois)} 条 → {CROSSREF_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

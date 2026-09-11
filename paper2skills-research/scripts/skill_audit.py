#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill 卡片体检器 —— paper2skills 存量资产盘点（只读，不修改任何文件）

用途:
    在批量萃取新卡片前，量化存量卡片的规范程度与缺口，为方案提供事实依据。

输出:
    data/skill_audit.json   结构化体检结果
    （同时打印摘要到 stdout）

检查项:
    1. frontmatter 是否存在 & 关键字段（title/module/topic/status/created/updated/source/paper）
    2. 五模块是否齐全（① 算法原理 ② 应用案例 ③ 代码模板 ④ 技能关联 ⑤ 商业价值）
    3. 代码块数量与是否含 def/class（可运行性最低信号）
    4. 是否含论文溯源（arXiv ID / DOI / paper 字段）
    5. 是否含公式（$$ 或 $...$）
    6. 字数规模
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAULT = ROOT / "paper2skills-vault"
OUT = ROOT / "paper2skills-research" / "data" / "skill_audit.json"

MODULES = {
    "algorithm": r"①\s*算法原理|##\s*①|算法原理",
    "application": r"②\s*母婴|##\s*②|应用案例",
    "code": r"③\s*代码模板|##\s*③|代码模板",
    "relations": r"④\s*技能关联|##\s*④|技能关联",
    "business": r"⑤\s*商业价值|##\s*⑤|商业价值",
}
FM_KEYS = ["title", "module", "topic", "status", "created", "updated", "doc_type", "owner", "source", "paper"]
ARXIV_RE = re.compile(r"arXiv[:\s]*([0-9]{4}\.[0-9]{4,5})|arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})")
DOI_RE = re.compile(r"10\.[0-9]{4,9}/[^\s)\]]+")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end]
    fm: dict[str, str] = {}
    for line in block.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line.strip())
        if m:
            fm[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return fm, text[end + 4 :]


def audit_card(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    fm, body = parse_frontmatter(text)
    code_blocks = re.findall(r"```(\w*)\n(.*?)```", text, flags=re.S)
    py_blocks = [c for lang, c in code_blocks if lang.lower() in {"python", "py"}]
    arxiv = {a or b for a, b in ARXIV_RE.findall(text)}
    doi = set(DOI_RE.findall(text))
    missing = [k for k, pat in MODULES.items() if not re.search(pat, text)]
    return {
        "file": path.name,
        "rel": str(path.relative_to(ROOT)),
        "domain_dir": path.parent.name,
        "has_frontmatter": bool(fm),
        "fm_keys": sorted(fm),
        "fm_missing": [k for k in FM_KEYS if k not in fm],
        "missing_modules": missing,
        "n_code_blocks": len(code_blocks),
        "n_python_blocks": len(py_blocks),
        "has_def": bool(re.search(r"^\s*(def|class)\s+\w+", "\n".join(py_blocks), flags=re.M)),
        "arxiv_ids": sorted(arxiv),
        "dois": sorted(doi)[:3],
        "has_provenance": bool(arxiv or doi or "paper" in fm),
        "has_formula": bool(re.search(r"\$\$|\$[^$\n]{3,}\$", text)),
        "lines": text.count("\n") + 1,
        "chars": len(text),
        "status": fm.get("status", ""),
        "paper_field": fm.get("paper", ""),
    }


def main() -> None:
    cards = sorted(
        p
        for p in VAULT.rglob("Skill-*.md")
        if "node_modules" not in p.parts
    )
    results = [audit_card(p) for p in cards]
    n = len(results)

    def pct(k: int) -> str:
        return f"{k}/{n} ({k * 100 // max(n, 1)}%)"

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_cards": n,
        "by_domain": dict(Counter(r["domain_dir"] for r in results).most_common()),
        "has_frontmatter": pct(sum(r["has_frontmatter"] for r in results)),
        "has_provenance": pct(sum(r["has_provenance"] for r in results)),
        "has_paper_field": pct(sum(bool(r["paper_field"]) for r in results)),
        "has_arxiv_id": pct(sum(bool(r["arxiv_ids"]) for r in results)),
        "has_python_block": pct(sum(r["n_python_blocks"] > 0 for r in results)),
        "has_def_or_class": pct(sum(r["has_def"] for r in results)),
        "has_formula": pct(sum(r["has_formula"] for r in results)),
        "missing_any_module": pct(sum(bool(r["missing_modules"]) for r in results)),
        "module_gap_counts": dict(Counter(m for r in results for m in r["missing_modules"]).most_common()),
        "status_dist": dict(Counter(r["status"] or "(无)" for r in results).most_common()),
        "median_lines": sorted(r["lines"] for r in results)[n // 2] if n else 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": summary, "cards": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()

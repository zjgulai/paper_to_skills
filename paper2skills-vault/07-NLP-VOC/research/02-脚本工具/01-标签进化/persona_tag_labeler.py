"""Persona tag labeler — Phase 5 D6 T6.2

对输入 jsonl 记录的 `text` 字段应用 55 原子画像标签的关键词规则，为每条记录产出
命中的 persona_tags + 分维度聚合 + 置信度分桶。

规则来源：
  01-设计文档/02-工作流设计/persona_tags_55.json

匹配规则：
  - 用词边界正则 `\b<keyword>\b`（IGNORECASE）
  - 每条标签按命中关键词数量赋置信度：
    1 → 0.60 | 2 → 0.80 | ≥3 → 1.00；最终乘以 tag.weight
  - 默认只保留 confidence ≥ 0.60 的标签

输出字段（保留原有字段，追加）:
  persona_tags            : list[{tag_id, tag_en, confidence, n_hits, evidence}]
  persona_dimensions      : dict[dim → list[tag_en]]（dedupe）
  persona_tag_count       : int
  persona_has_any         : bool

用法：
  python persona_tag_labeler.py \
    --input test_set_5k_p5_llm.jsonl \
    --source-text test_set_5k_stratified.jsonl \
    --rules-json ../01-设计文档/02-工作流设计/persona_tags_55.json \
    --output test_set_5k_p5_persona.jsonl \
    --summary-out test_set_5k_p5_persona.summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def index_by_id(records: list[dict]) -> dict[str, dict]:
    return {r["review_id"]: r for r in records if r.get("review_id")}


def build_regex_for_keyword(kw: str) -> re.Pattern:
    """Word-boundary regex; multi-word keywords preserve spaces literally.

    Keyword 'first time mom' → \bfirst time mom\b
    Non-ASCII accents kept literal.
    """
    escaped = re.escape(kw.strip().lower())
    return re.compile(r"(?:^|\W)" + escaped + r"(?:$|\W)", re.IGNORECASE)


class PersonaRule:
    __slots__ = ("tag_id", "tag_en", "tag_cn", "dimension", "sub_dimension",
                 "keywords", "weight", "regexes")

    def __init__(self, d: dict):
        self.tag_id = d["tag_id"]
        self.tag_en = d["tag_en"]
        self.tag_cn = d.get("tag_cn", "")
        self.dimension = d["dimension"]
        self.sub_dimension = d.get("sub_dimension", "")
        self.keywords = list(d.get("keywords", []))
        self.weight = float(d.get("weight", 1.0))
        self.regexes = [build_regex_for_keyword(k) for k in self.keywords]

    def match(self, text: str) -> tuple[int, list[str]]:
        """Return (n_hits, list of matched keyword source strings)."""
        hits: list[str] = []
        for kw, rx in zip(self.keywords, self.regexes):
            if rx.search(text):
                hits.append(kw)
        return len(hits), hits


def confidence_from_hits(n: int, weight: float) -> float:
    base = 0.60 if n == 1 else (0.80 if n == 2 else 1.0)
    return round(min(1.0, base * weight), 3)


def load_rules(path: Path) -> list[PersonaRule]:
    return [PersonaRule(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def label_one(rec: dict, src: dict, rules: list[PersonaRule],
              min_conf: float = 0.60) -> dict:
    text = (rec.get("text") or src.get("text") or "").strip()
    if not text:
        out = dict(rec)
        out["persona_tags"] = []
        out["persona_dimensions"] = {}
        out["persona_tag_count"] = 0
        out["persona_has_any"] = False
        return out

    tags_out: list[dict] = []
    dim_map: dict[str, list[str]] = defaultdict(list)
    for r in rules:
        n, kws = r.match(text)
        if n == 0:
            continue
        conf = confidence_from_hits(n, r.weight)
        if conf < min_conf:
            continue
        evidence = kws[:3]
        tags_out.append({
            "tag_id": r.tag_id,
            "tag_en": r.tag_en,
            "dimension": r.dimension,
            "sub_dimension": r.sub_dimension,
            "confidence": conf,
            "n_hits": n,
            "evidence": evidence,
        })
        if r.tag_en not in dim_map[r.dimension]:
            dim_map[r.dimension].append(r.tag_en)
    tags_out.sort(key=lambda x: (-x["confidence"], x["dimension"], x["tag_id"]))
    out = dict(rec)
    if not rec.get("text") and text:
        out["text"] = text
    out["persona_tags"] = tags_out
    out["persona_dimensions"] = dict(dim_map)
    out["persona_tag_count"] = len(tags_out)
    out["persona_has_any"] = len(tags_out) > 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--source-text", type=Path, default=None,
                    help="Optional stratified jsonl to join text by review_id")
    ap.add_argument("--rules-json", type=Path,
                    default=Path(__file__).resolve().parents[2] /
                    "01-设计文档" / "02-工作流设计" / "persona_tags_55.json",
                    help="Canonical 55-tag JSON path")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--min-confidence", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}", file=sys.stderr); return 2
    if not args.rules_json.exists():
        print(f"❌ Rules JSON not found: {args.rules_json}", file=sys.stderr); return 2

    records = load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    rules = load_rules(args.rules_json)
    print(f"Loaded {len(records)} records, {len(rules)} persona rules")

    src_idx: dict[str, dict] = {}
    if args.source_text and args.source_text.exists():
        src_idx = index_by_id(load_jsonl(args.source_text))
        print(f"Joined source-text: {len(src_idx)} records")

    n_with_any = 0
    tag_hit_counter: Counter = Counter()
    dim_hit_counter: Counter = Counter()
    per_dim_coverage: dict[str, set[str]] = defaultdict(set)
    per_record_tag_count = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            src = src_idx.get(rec.get("review_id"), {})
            out = label_one(rec, src, rules, args.min_confidence)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if out["persona_has_any"]:
                n_with_any += 1
            per_record_tag_count.append(out["persona_tag_count"])
            for t in out["persona_tags"]:
                tag_hit_counter[t["tag_id"]] += 1
                dim_hit_counter[t["dimension"]] += 1
                per_dim_coverage[t["dimension"]].add(t["tag_en"])

    n = len(records) or 1
    penetration = n_with_any / n
    n_tags_with_hit = sum(1 for r in rules if tag_hit_counter.get(r.tag_id, 0) > 0)
    avg_tags_per_rec = sum(per_record_tag_count) / n

    print("\n" + "=" * 60)
    print("Persona Labeler Summary")
    print("=" * 60)
    print(f"  n_records               : {n}")
    print(f"  records_with_any_tag    : {n_with_any}  ({penetration*100:.2f}%)")
    print(f"  tags_with_at_least_1_hit: {n_tags_with_hit} / {len(rules)}")
    print(f"  avg_tags_per_record     : {avg_tags_per_rec:.2f}")
    print("\nPer-dimension hits (records):")
    for dim, c in dim_hit_counter.most_common():
        n_tags_in_dim = len(per_dim_coverage[dim])
        print(f"  {dim:10}  ×{c:5} records, covering {n_tags_in_dim} distinct tags")
    print("\nTop-20 tag hit counts:")
    for tid, c in tag_hit_counter.most_common(20):
        rule = next((r for r in rules if r.tag_id == tid), None)
        name = rule.tag_en if rule else "?"
        print(f"  {tid}  {name:28} ×{c}")

    if args.summary_out:
        summary = {
            "n_records": n,
            "penetration_rate": penetration,
            "n_records_with_any_tag": n_with_any,
            "tags_with_at_least_1_hit": n_tags_with_hit,
            "n_total_rules": len(rules),
            "avg_tags_per_record": avg_tags_per_rec,
            "per_dimension_hits": dict(dim_hit_counter),
            "per_dimension_coverage": {k: sorted(v) for k, v in per_dim_coverage.items()},
            "top_tag_hits": dict(tag_hit_counter.most_common(30)),
            "qa_pass_penetration_60": penetration >= 0.60,
            "qa_pass_coverage_45": n_tags_with_hit >= 45,
        }
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📄 Summary: {args.summary_out}")
    print(f"✅ Output : {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

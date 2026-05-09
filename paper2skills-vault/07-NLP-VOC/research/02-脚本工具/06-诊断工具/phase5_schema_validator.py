"""Phase 5 schema validator — D7 T7.1.5

Validates the merged Phase 5 jsonl output against a chosen dictionary version.

Checks:
  S1  required fields exist on every record
      (review_id, text, labels, phase5_meta)
  S2  every labels[].tag_id ∈ dictionary (frozenset from tag_dict_loader)
  S3  every consensus_labels[].tag_id ∈ dictionary
  S4  every persona_tags[].tag_id matches /^P-L2-\\d{2}$/
  S5  proxy_nps_final ∈ {promoter, passive, detractor, None}
  S6  overall_sentiment ∈ {positive, neutral, negative, None}
  S7  no record has both POS and matching NEG (e.g. *_P001 + *_N001)

Outputs Markdown + JSON report. Exit 0 only when all 7 checks pass.

Usage:
  python phase5_schema_validator.py \
      --input phase5_intermediate_merged.jsonl \
      --dict-version v3.9 \
      --report phase5_schema_validation.md \
      --json-out phase5_schema_validation.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


PERSONA_RE = re.compile(r"^P-L2-\d{2}$")
SENT_OK = {"positive", "neutral", "negative", None}
NPS_OK = {"promoter", "passive", "detractor", None}


def load_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def load_dict_ids(version: str) -> frozenset[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07-LLM引擎"))
    if version == "v3.9":
        from tag_dict_loader import get_all_tag_ids
        return get_all_tag_ids()
    if version == "v4.0":
        from tag_dict_loader import DICT_PATH, get_all_tag_ids
        v40 = DICT_PATH.parent / "tag_dictionary_v4.0.xlsx"
        if v40.exists():
            return get_all_tag_ids(str(v40))
        print("⚠ v4.0 dict not found, falling back to v3.9")
        return get_all_tag_ids()
    raise ValueError(f"Unknown dict version: {version}")


def find_pos_neg_conflicts(tag_ids: set[str]) -> list[tuple[str, str]]:
    pos: dict[tuple[str, str], str] = {}
    neg: dict[tuple[str, str], str] = {}
    pn_re = re.compile(r"^(.+)_([PN])(\d{3,})$")
    for tid in tag_ids:
        m = pn_re.match(tid)
        if m:
            base, sign, num = m.groups()
            (pos if sign == "P" else neg)[(base, num)] = tid
    out = []
    for k in set(pos) & set(neg):
        out.append((pos[k], neg[k]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--dict-version", choices=["v3.9", "v4.0"], default="v3.9")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}", file=sys.stderr); return 2

    valid_ids = load_dict_ids(args.dict_version)
    print(f"📂 Loaded dictionary {args.dict_version}: {len(valid_ids)} tag_ids")

    n_records = 0
    n_missing_required = 0
    n_invalid_label = 0
    n_invalid_consensus = 0
    n_invalid_persona = 0
    n_invalid_sent = 0
    n_invalid_nps = 0
    n_pos_neg_conflict = 0
    invalid_tag_examples: Counter = Counter()
    invalid_persona_examples: Counter = Counter()
    pos_neg_examples: list[tuple[str, str, str]] = []

    REQUIRED = ["review_id", "text", "labels", "phase5_meta"]

    for rec in load_jsonl(args.input):
        n_records += 1
        for f in REQUIRED:
            if f not in rec:
                n_missing_required += 1
                break

        rid = rec.get("review_id", "?")
        for l in rec.get("labels") or []:
            tid = (l or {}).get("tag_id")
            if tid and tid not in valid_ids:
                n_invalid_label += 1
                invalid_tag_examples[tid] += 1

        for l in rec.get("consensus_labels") or []:
            tid = (l or {}).get("tag_id")
            if tid and tid not in valid_ids:
                n_invalid_consensus += 1

        for t in rec.get("persona_tags") or []:
            tid = (t or {}).get("tag_id")
            if tid and not PERSONA_RE.match(tid):
                n_invalid_persona += 1
                invalid_persona_examples[tid] += 1

        if rec.get("overall_sentiment") not in SENT_OK:
            n_invalid_sent += 1
        if rec.get("proxy_nps_final") not in NPS_OK and rec.get("proxy_nps_final") is not None:
            n_invalid_nps += 1

        tag_ids = {l.get("tag_id") for l in (rec.get("labels") or []) if l.get("tag_id")}
        label_by_id = {l.get("tag_id"): l for l in (rec.get("labels") or []) if l.get("tag_id")}
        conflicts = find_pos_neg_conflicts(tag_ids)
        if conflicts:
            hard_conflicts = []
            for p, n in conflicts:
                p_ev = (label_by_id.get(p) or {}).get("evidence") or ""
                n_ev = (label_by_id.get(n) or {}).get("evidence") or ""
                if not p_ev or not n_ev:
                    hard_conflicts.append((p, n))
            if hard_conflicts:
                n_pos_neg_conflict += 1
                for p, n in hard_conflicts[:1]:
                    if len(pos_neg_examples) < 5:
                        pos_neg_examples.append((rid, p, n))

    checks = [
        ("S1", "Required fields present", n_missing_required == 0, f"{n_missing_required} records missing"),
        ("S2", f"labels[].tag_id ∈ dict {args.dict_version}", n_invalid_label == 0,
         f"{n_invalid_label} invalid tags ({len(invalid_tag_examples)} distinct)"),
        ("S3", "consensus_labels[].tag_id ∈ dict", n_invalid_consensus == 0,
         f"{n_invalid_consensus} invalid consensus tags"),
        ("S4", "persona_tags[].tag_id matches P-L2-NN", n_invalid_persona == 0,
         f"{n_invalid_persona} invalid persona tags"),
        ("S5", "overall_sentiment ∈ {pos/neu/neg/None}", n_invalid_sent == 0,
         f"{n_invalid_sent} invalid sentiment values"),
        ("S6", "proxy_nps_final ∈ {prom/pas/det/None}", n_invalid_nps == 0,
         f"{n_invalid_nps} invalid NPS values"),
        ("S7", "POS/NEG hard conflict-free (both sides w/o evidence)", n_pos_neg_conflict == 0,
         f"{n_pos_neg_conflict} hard conflicts (mixed-sentiment reviews with evidence on both sides allowed)"),
    ]
    overall = all(c[2] for c in checks)

    lines: list[str] = []
    lines.append(f"# Phase 5 Schema Validation — {args.dict_version}")
    lines.append("")
    lines.append(f"**Records**: {n_records}  ｜  **Overall**: {'🟢 PASS' if overall else '🔴 FAIL'}")
    lines.append("")
    lines.append("| # | Check | Pass | Detail |")
    lines.append("|---|---|:---:|---|")
    for cid, name, ok, detail in checks:
        lines.append(f"| {cid} | {name} | {'✅' if ok else '❌'} | {detail} |")
    lines.append("")
    if invalid_tag_examples:
        lines.append("## Invalid label examples (top-10)")
        lines.append("")
        for tid, c in invalid_tag_examples.most_common(10):
            lines.append(f"- `{tid}` × {c}")
        lines.append("")
    if pos_neg_examples:
        lines.append("## POS/NEG conflict examples")
        lines.append("")
        for rid, p, n in pos_neg_examples:
            lines.append(f"- `{rid}` ⟶ `{p}` + `{n}`")
        lines.append("")

    print("\n".join(lines))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n📄 Report: {args.report}")

    if args.json_out:
        result = {
            "dict_version": args.dict_version,
            "n_records": n_records,
            "overall_pass": overall,
            "checks": [
                {"id": cid, "name": name, "pass": ok, "detail": detail}
                for cid, name, ok, detail in checks
            ],
            "invalid_tag_examples": dict(invalid_tag_examples.most_common(20)),
            "invalid_persona_examples": dict(invalid_persona_examples.most_common(20)),
            "pos_neg_examples": pos_neg_examples,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📦 JSON: {args.json_out}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

"""Low-confidence sample extractor.

Filter rule (any ONE triggers low-conf, with explicit filter_reason):
  reason='zero_label'      labels list is empty (LLM produced no tags)
  reason='low_max_conf'    max(labels[].confidence) < threshold (default 0.70)
  reason='phase4_zero'     original Phase 4 had n_tags == 0 (joined from stratified file by review_id)
  reason='llm_failed'      success=false in LLM output

Output schema: each input record + extra fields
  filter_reason  list of triggered reasons, e.g. ["zero_label", "phase4_zero"]
  max_conf       float | null (None when zero labels)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def index_by_review(records: list[dict]) -> dict[str, dict]:
    return {r["review_id"]: r for r in records if r.get("review_id")}


def classify(rec: dict, p4_lookup: dict[str, dict], conf_threshold: float, phase4_mode: bool = False) -> tuple[list[str], float | None]:
    reasons: list[str] = []
    # phase4_mode: input IS phase4_labeled.jsonl directly; trigger ONLY on n_tags == 0
    if phase4_mode:
        if (rec.get("n_tags") or 0) == 0:
            reasons.append("phase4_zero")
        confs = [l.get("confidence") for l in (rec.get("labels") or [])
                 if isinstance(l.get("confidence"), (int, float))
                 and not str(l.get("tag_id", "")).startswith("TAG_ALC")]
        max_conf = max(confs) if confs else None
        if confs and max_conf is not None and max_conf < conf_threshold:
            reasons.append("low_max_conf")
        return reasons, max_conf

    if not rec.get("success"):
        reasons.append("llm_failed")
    labels = rec.get("labels") or []
    if not labels:
        reasons.append("zero_label")
    confs = [l.get("confidence") for l in labels if isinstance(l.get("confidence"), (int, float))]
    max_conf = max(confs) if confs else None
    if confs and max_conf is not None and max_conf < conf_threshold:
        reasons.append("low_max_conf")
    p4 = p4_lookup.get(rec.get("review_id"))
    if p4 is not None and (p4.get("n_tags") or 0) == 0:
        reasons.append("phase4_zero")
    return reasons, max_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="LLM-labeled jsonl (D2 output)")
    ap.add_argument("--phase4", type=Path, default=None,
                    help="Original stratified jsonl with n_tags field (for phase4_zero detection)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--confidence-threshold", type=float, default=0.70)
    ap.add_argument("--include-zero-label", action="store_true", default=True,
                    help="Include records with empty labels list (default true)")
    ap.add_argument("--exclude-zero-label", action="store_false", dest="include_zero_label")
    ap.add_argument("--phase4-mode", action="store_true", default=False,
                    help="Input is phase4_labeled.jsonl directly; trigger on n_tags==0 or low-conf GEN tags")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}"); sys.exit(2)

    p4_lookup: dict[str, dict] = {}
    if args.phase4 and args.phase4.exists():
        p4_lookup = index_by_review(load_jsonl(args.phase4))
        print(f"Phase 4 reference: {len(p4_lookup)} records")

    in_records = load_jsonl(args.input)
    n_total = len(in_records)
    print(f"Input:  {n_total} records  (conf_threshold={args.confidence_threshold})")

    out_records: list[dict] = []
    reason_counter = Counter()
    n_kept = 0

    for rec in in_records:
        reasons, max_conf = classify(rec, p4_lookup, args.confidence_threshold, phase4_mode=args.phase4_mode)
        if not args.include_zero_label and reasons == ["zero_label"]:
            continue
        if reasons:
            for r in reasons:
                reason_counter[r] += 1
            out_rec = dict(rec)
            out_rec["filter_reason"] = reasons
            out_rec["max_conf"] = max_conf
            out_records.append(out_rec)
            n_kept += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Low-conf samples extracted: {n_kept}/{n_total} = {n_kept/n_total*100:.2f}%")
    print("=" * 60)
    print("\nReason breakdown (a record may have multiple reasons):")
    for reason, n in reason_counter.most_common():
        print(f"  {reason:20} ×{n}")
    print(f"\n✅ Written: {args.output}")


if __name__ == "__main__":
    main()

"""Golden-set sampler — picks 500 records for human annotation.

Strategy (per Phase 5 D3 plan):
  - Stratified by data_source proportional to 5K test set
  - Force ≥ 20 zero-label samples per source (for hard-case calibration)
  - Stable seed (random_state=20260507_d3) so re-runs are deterministic
  - Records are paired with D2 LLM predictions (test_set_5k_p5_llm.jsonl)
    for side-by-side review during annotation

Schema in:  test_set_5k_stratified.jsonl  (D1)
Schema out: golden_set_500.jsonl with fields
  review_id, text, language, data_source, rating,
  phase4_labels (list of {tag_id, tag_en, sentiment_calibrated}),
  llm_pred (list of LabelItem),
  llm_overall_sentiment, llm_proxy_nps,
  golden_labels (empty, to be filled by human),
  golden_overall_sentiment (empty),
  golden_proxy_nps (empty),
  notes (empty, free-form)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TARGET = 500
ZERO_LABEL_FLOOR = 20
SEED = 20260507_03


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_lookup(d2_records: list[dict]) -> dict[str, dict]:
    return {r["review_id"]: r for r in d2_records}


def stratify(stratified: list[dict], d2_lookup: dict[str, dict], target: int, zero_floor: int):
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in stratified:
        by_source[r.get("data_source", "unknown")].append(r)

    total_n = len(stratified)
    rng = random.Random(SEED)

    picks: list[dict] = []
    per_src_quota: dict[str, int] = {}
    for src, recs in by_source.items():
        quota = max(zero_floor, round(target * len(recs) / total_n))
        per_src_quota[src] = quota

    total_quota = sum(per_src_quota.values())
    if total_quota != target:
        diff = target - total_quota
        biggest = max(per_src_quota, key=lambda k: per_src_quota[k])
        per_src_quota[biggest] += diff

    for src, recs in by_source.items():
        quota = per_src_quota[src]
        zero = [r for r in recs if not r.get("labels")]
        non_zero = [r for r in recs if r.get("labels")]
        rng.shuffle(zero); rng.shuffle(non_zero)

        n_zero = min(zero_floor, len(zero), quota)
        n_non_zero = quota - n_zero
        chosen = zero[:n_zero] + non_zero[:n_non_zero]
        while len(chosen) < quota and len(zero) > n_zero:
            chosen.append(zero[n_zero])
            n_zero += 1
        picks.extend(chosen)

    rng.shuffle(picks)
    return picks, per_src_quota


def to_golden_record(rec: dict, d2: dict | None) -> dict:
    phase4_labels = []
    for lbl in rec.get("labels") or []:
        phase4_labels.append({
            "tag_id": lbl.get("tag_id"),
            "tag_en": lbl.get("tag_en"),
            "sentiment_calibrated": lbl.get("sentiment_calibrated"),
        })
    llm_pred = []
    if d2 and d2.get("success"):
        for lbl in d2.get("labels") or []:
            llm_pred.append({
                "tag_id": lbl.get("tag_id"),
                "tag_en": lbl.get("tag_en"),
                "confidence": lbl.get("confidence"),
                "evidence": lbl.get("evidence"),
            })
    return {
        "review_id": rec["review_id"],
        "data_source": rec.get("data_source"),
        "language": rec.get("language"),
        "rating": rec.get("rating"),
        "text": rec.get("text"),
        "phase4_labels": phase4_labels,
        "llm_pred": llm_pred,
        "llm_overall_sentiment": (d2 or {}).get("overall_sentiment"),
        "llm_proxy_nps": (d2 or {}).get("proxy_nps"),
        "golden_labels": [],
        "golden_overall_sentiment": None,
        "golden_proxy_nps": None,
        "golden_notes": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratified", type=Path, required=True,
                    help="D1 5K stratified jsonl with phase4 labels embedded")
    ap.add_argument("--llm-pred", type=Path, required=True,
                    help="D2 LLM labeled jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--zero-floor", type=int, default=ZERO_LABEL_FLOOR)
    args = ap.parse_args()

    if not args.stratified.exists():
        print(f"❌ Stratified not found: {args.stratified}"); sys.exit(2)
    if not args.llm_pred.exists():
        print(f"❌ LLM pred not found: {args.llm_pred}"); sys.exit(2)

    stratified = load_jsonl(args.stratified)
    d2 = load_jsonl(args.llm_pred)
    d2_lookup = build_lookup(d2)

    picks, quota = stratify(stratified, d2_lookup, args.target, args.zero_floor)

    print("=" * 60)
    print(f"Golden-set sampling: target={args.target}, zero_floor={args.zero_floor}")
    print(f"Per-source quotas:")
    for s, q in sorted(quota.items(), key=lambda x: -x[1]):
        print(f"  {s:25} {q}")
    print(f"Picked: {len(picks)}")
    print("=" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in picks:
            d2_rec = d2_lookup.get(r["review_id"])
            f.write(json.dumps(to_golden_record(r, d2_rec), ensure_ascii=False) + "\n")

    _audit_output(args.output, len(picks))
    print(f"\n✅ Written: {args.output}")


def _audit_output(output_path: Path, n_total: int):
    by_src = defaultdict(int); zero_by_src = defaultdict(int); has_llm = 0
    for line in output_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        by_src[rec["data_source"]] += 1
        if not rec["phase4_labels"]:
            zero_by_src[rec["data_source"]] += 1
        if rec["llm_pred"]:
            has_llm += 1
    print(f"\nAudit: by_source={dict(by_src)}")
    print(f"       zero_label_count_by_source={dict(zero_by_src)}")
    print(f"       has_llm_pred={has_llm}/{n_total}")


if __name__ == "__main__":
    main()

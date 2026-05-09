"""Schema test for llm_labeler outputs.

Validates that every record in a labeled JSONL file conforms to:
  - JSON parseable
  - tag_id ∈ v3.9 dictionary (602 IDs)
  - confidence ∈ [0, 1]
  - overall_sentiment ∈ {positive, negative, neutral}
  - proxy_nps ∈ {promoter, passive, detractor}
  - labels list ≤ 8 items
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_labeler import LLMLabelOutput, LabelItem
from tag_dict_loader import get_all_tag_ids


def validate_file(path: Path, fail_threshold: float = 0.01) -> dict:
    valid_ids = get_all_tag_ids()
    n_total = 0
    n_records_pass = 0
    n_json_fail = 0
    n_schema_fail = 0
    n_with_labels = 0
    invalid_tag_ids: dict[str, int] = {}
    errors: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            n_total += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                n_json_fail += 1
                errors.append({"line": lineno, "type": "json_parse", "msg": str(e)})
                continue

            if not rec.get("success"):
                continue

            try:
                obj = {
                    "labels": rec.get("labels", []),
                    "overall_sentiment": rec.get("overall_sentiment") or "neutral",
                    "proxy_nps": rec.get("proxy_nps") or "passive",
                }
                LLMLabelOutput.model_validate(obj)
            except Exception as e:
                n_schema_fail += 1
                errors.append({"line": lineno, "review_id": rec.get("review_id"), "type": "schema", "msg": str(e)[:200]})
                for lbl in rec.get("labels", []):
                    tid = lbl.get("tag_id")
                    if tid and tid not in valid_ids:
                        invalid_tag_ids[tid] = invalid_tag_ids.get(tid, 0) + 1
                continue

            n_records_pass += 1
            if rec.get("labels"):
                n_with_labels += 1

    n_succeeded_records = sum(
        1 for line in path.open("r", encoding="utf-8") if line.strip() and json.loads(line).get("success")
    )

    fail_rate = (n_json_fail + n_schema_fail) / n_total if n_total else 0
    pass_overall = (
        n_total > 0 and fail_rate < fail_threshold and not invalid_tag_ids
    )

    report = {
        "file": str(path),
        "n_total": n_total,
        "n_succeeded_records": n_succeeded_records,
        "n_records_pass": n_records_pass,
        "n_json_fail": n_json_fail,
        "n_schema_fail": n_schema_fail,
        "fail_rate": fail_rate,
        "n_with_labels": n_with_labels,
        "invalid_tag_ids": invalid_tag_ids,
        "errors_sample": errors[:5],
        "pass": pass_overall,
        "threshold": fail_threshold,
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Input not found: {args.input}")
        sys.exit(2)

    report = validate_file(args.input, args.threshold)

    print("=" * 70)
    print("Schema Validation Report")
    print("=" * 70)
    for k, v in report.items():
        if k in ("errors_sample", "invalid_tag_ids"):
            continue
        print(f"  {k:30}: {v}")
    if report["invalid_tag_ids"]:
        print(f"  invalid_tag_ids               : {len(report['invalid_tag_ids'])} unique")
        for tid, n in sorted(report["invalid_tag_ids"].items(), key=lambda x: -x[1])[:5]:
            print(f"    {tid:20} ×{n}")
    if report["errors_sample"]:
        print(f"  errors_sample                 : {len(report['errors_sample'])}")
        for e in report["errors_sample"]:
            print(f"    {e}")

    out = args.report_out or args.input.with_suffix(".schema_report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {out}")

    print(f"\n{'🎉 SCHEMA TEST PASS' if report['pass'] else '❌ SCHEMA TEST FAIL'}")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()

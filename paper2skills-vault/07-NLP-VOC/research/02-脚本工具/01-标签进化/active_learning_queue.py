"""Active learning queue manager.

Post-processes the active_learning_queue.jsonl produced by llm_consensus.py:
  - Dedupes by review_id (keeps the most recent entry)
  - Sorts by queue_priority (high → medium → low) then by data_source
  - Optionally slices a subset for human triage (--slice-top N)
  - Reports queue health stats (size by priority, by source, by reason)

Output:
  --output       cleaned + sorted full queue (jsonl)
  --slice-top N  optional --slice-top-out subset for immediate triage
  --stats-out    JSON summary
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dedupe_keep_last(records: list[dict]) -> list[dict]:
    seen: OrderedDict[str, dict] = OrderedDict()
    for r in records:
        rid = r.get("review_id")
        if rid is None:
            continue
        seen[rid] = r
    return list(seen.values())


def sort_key(rec: dict) -> tuple[int, str, str]:
    return (
        PRIORITY_ORDER.get(rec.get("queue_priority", "low"), 99),
        rec.get("data_source") or "",
        rec.get("review_id") or "",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--slice-top", type=int, default=None,
                    help="Take top N high-priority entries to a separate file")
    ap.add_argument("--slice-top-out", type=Path, default=None)
    ap.add_argument("--stats-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}"); sys.exit(2)

    raw = load_jsonl(args.input)
    deduped = dedupe_keep_last(raw)
    deduped.sort(key=sort_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_priority = Counter(r.get("queue_priority", "low") for r in deduped)
    by_source = Counter(r.get("data_source") or "?" for r in deduped)
    by_reason = Counter()
    for r in deduped:
        for reason in (r.get("disagreement_reason") or "").split("; "):
            key = reason.split(":")[0].split("=")[0]
            if key:
                by_reason[key] += 1

    print(f"Loaded:  {len(raw)}")
    print(f"Deduped: {len(deduped)}")
    print(f"\nBy priority:  {dict(by_priority)}")
    print(f"By source:    {dict(by_source)}")
    print(f"By reason:")
    for reason, n in by_reason.most_common(10):
        print(f"  {reason:30} ×{n}")

    if args.slice_top and args.slice_top_out:
        slice_n = min(args.slice_top, len(deduped))
        with args.slice_top_out.open("w", encoding="utf-8") as f:
            for r in deduped[:slice_n]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n📌 Top-{slice_n} slice: {args.slice_top_out}")

    if args.stats_out:
        stats = {
            "n_total": len(deduped),
            "n_raw_input": len(raw),
            "by_priority": dict(by_priority),
            "by_source": dict(by_source),
            "by_reason": dict(by_reason),
            "n_dedupe_removed": len(raw) - len(deduped),
        }
        args.stats_out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📊 Stats: {args.stats_out}")

    print(f"\n✅ Queue: {args.output}")


if __name__ == "__main__":
    main()

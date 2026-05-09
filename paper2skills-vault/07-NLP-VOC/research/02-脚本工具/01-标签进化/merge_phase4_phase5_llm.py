"""D9 T9.6 — merge phase4_labeled.jsonl + phase5_full_labeled_llm.jsonl by review_id.

Merge rules:
  - Main key: review_id (both inputs guaranteed to cover 364,569 records)
  - Phase 4 record provides the FULL schema (text, meta, rating, etc.)
  - Phase 5 LLM labels are APPENDED to labels array with source="llm_v4flash"
    (NOT replacing phase4 labels — dual-source tagging preserves both)
  - phase5_meta attached with confidence / tokens / cache stats
  - overall_sentiment / proxy_nps: prefer phase5 when present and success=True,
    else fallback to phase4 value

Schema contract (output = phase5_intermediate_merged.jsonl):
  - All phase4 fields preserved
  - labels[] = phase4 labels (if any) + phase5 LLM labels (if any)
  - phase5_meta = {has_llm_label, llm_success, llm_cache_hit_tokens, ...}
  - label_sources[] = list of sources present (e.g. ["phase4_gen", "phase4_alchemist", "llm_v4flash"])

Post-merge QA:
  - Total records == 364,569
  - review_id unique (no dups)
  - Every phase5 record's labels merged correctly

Usage:
  python merge_phase4_phase5_llm.py \\
      --p4 research/04-输出结果/unified_labeling/phase4_labeled.jsonl \\
      --p5-llm research/04-输出结果/unified_labeling/phase5_full_labeled_llm.jsonl \\
      --output research/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl \\
      --audit-out research/04-输出结果/03-审计报告/phase5_d9_merge_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def index_by_review(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in iter_jsonl(path):
        rid = r.get("review_id")
        if rid is None:
            continue
        out[rid] = r
    return out


def merge_one(p4: dict, p5: dict | None) -> dict:
    """Merge a single phase4 record with optional phase5 record.

    Phase 4 is base. Phase 5 labels appended. Sentiment/NPS phase5 preferred when success.
    """
    out = dict(p4)

    p4_labels = list(p4.get("labels") or [])

    p5_labels: list[dict] = []
    if p5 and p5.get("success"):
        for l in (p5.get("labels") or []):
            if not isinstance(l, dict):
                continue
            new_l = dict(l)
            new_l["source"] = "llm_v4flash"
            p5_labels.append(new_l)

    # Dedupe by tag_id (keep phase4 version if both exist — phase4 has richer metadata like aipl_node)
    seen = {l.get("tag_id") for l in p4_labels if l.get("tag_id")}
    for l in p5_labels:
        tid = l.get("tag_id")
        if tid and tid not in seen:
            p4_labels.append(l)
            seen.add(tid)

    out["labels"] = p4_labels

    sources = set()
    for l in p4_labels:
        src = l.get("source") or l.get("label_source") or "phase4"
        sources.add(src)
    out["label_sources"] = sorted(sources)

    if p5 and p5.get("success"):
        p5_sent = p5.get("overall_sentiment")
        p5_nps = p5.get("proxy_nps")
        if p5_sent:
            out["overall_sentiment"] = p5_sent
        if p5_nps:
            out["proxy_nps_llm"] = p5_nps

    phase5_meta = {
        "has_llm_label": bool(p5_labels),
        "llm_success": bool(p5 and p5.get("success")),
        "llm_attempted": p5 is not None,
    }
    if p5:
        meta = p5.get("_meta") or {}
        phase5_meta.update({
            "llm_model": meta.get("model"),
            "llm_tokens_in": meta.get("tokens_in"),
            "llm_tokens_out": meta.get("tokens_out"),
            "llm_cache_hit_tokens": meta.get("cache_hit"),
            "llm_latency_ms": meta.get("latency_ms"),
            "llm_error": meta.get("error") if not p5.get("success") else "",
        })
    out["phase5_meta"] = phase5_meta

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", type=Path, required=True, help="phase4_labeled.jsonl (364K full)")
    ap.add_argument("--p5-llm", type=Path, required=True, help="phase5_full_labeled_llm.jsonl (D8 output ~87K)")
    ap.add_argument("--output", type=Path, required=True, help="Merged jsonl output")
    ap.add_argument("--audit-out", type=Path, default=None, help="Merge audit JSON report")
    args = ap.parse_args()

    for p in [args.p4, args.p5_llm]:
        if not p.exists():
            print(f"❌ Not found: {p}", file=sys.stderr)
            return 2

    print(f"Indexing phase5 LLM output by review_id ...", flush=True)
    p5_map = index_by_review(args.p5_llm)
    print(f"  phase5 LLM records: {len(p5_map)}")

    # Stream phase4 as base (large file, don't load all at once)
    print(f"Streaming phase4_labeled.jsonl → merging → writing {args.output.name} ...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_with_p5 = 0
    n_p5_success = 0
    n_p5_failed = 0
    n_p5_has_label = 0
    n_dup_reviews = 0
    seen_review_ids: set[str] = set()
    label_source_dist: Counter = Counter()
    llm_error_dist: Counter = Counter()

    with args.output.open("w", encoding="utf-8") as out_f:
        for p4 in iter_jsonl(args.p4):
            n_total += 1
            rid = p4.get("review_id")
            if rid in seen_review_ids:
                n_dup_reviews += 1
            else:
                seen_review_ids.add(rid)

            p5 = p5_map.get(rid)
            if p5:
                n_with_p5 += 1
                if p5.get("success"):
                    n_p5_success += 1
                    if p5.get("labels"):
                        n_p5_has_label += 1
                else:
                    n_p5_failed += 1
                    err = (p5.get("_meta") or {}).get("error") or ""
                    llm_error_dist[err[:60]] += 1

            merged = merge_one(p4, p5)
            for src in merged.get("label_sources") or []:
                label_source_dist[src] += 1

            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")

            if n_total % 50000 == 0:
                print(f"  progress: {n_total:,} records", flush=True)

    audit = {
        "n_total_merged": n_total,
        "n_duplicate_review_ids": n_dup_reviews,
        "n_with_phase5_record": n_with_p5,
        "n_phase5_success": n_p5_success,
        "n_phase5_failed": n_p5_failed,
        "n_phase5_has_label": n_p5_has_label,
        "label_source_distribution": dict(label_source_dist),
        "top_llm_errors": dict(llm_error_dist.most_common(10)),
    }

    print()
    print("=" * 60)
    print("Merge Summary")
    print("=" * 60)
    print(f"  Total records:              {n_total:,}")
    print(f"  Duplicate review_ids:       {n_dup_reviews}")
    print(f"  With phase5 record:         {n_with_p5:,} ({n_with_p5/n_total*100:.2f}%)")
    print(f"  Phase5 success:             {n_p5_success:,} ({n_p5_success/n_total*100:.2f}%)")
    print(f"  Phase5 failed:              {n_p5_failed}")
    print(f"  Phase5 produced ≥1 label:   {n_p5_has_label:,}")
    print()
    print(f"  Label source distribution (per-label counts):")
    for src, n in label_source_dist.most_common(10):
        print(f"    {src:30} {n:,}")

    if args.audit_out:
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📊 Audit: {args.audit_out}")

    ok = (n_total == 364569 and n_dup_reviews == 0)
    print(f"\n{'🎉 MERGE PASS' if ok else '⚠️  MERGE WARNING (expected 364569 records, 0 duplicates)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

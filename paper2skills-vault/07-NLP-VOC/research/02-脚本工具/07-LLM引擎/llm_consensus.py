"""LLM consensus arbitration — primary + fallback dual-model agreement.

Generalised from D3's consensus_prefill: takes low-confidence samples that
already have a primary LLM prediction, runs the fallback LLM on the same
records, then partitions by soft-consensus agreement.

Soft consensus rule (per Phase 5 D3 empirical finding):
  agreement = (shared_tags ≥ 1) AND (sentiment match) AND (NPS match)

Outputs:
  consensus_result.jsonl       records that reached agreement; carries
                                consensus_labels (intersection ≤ 3),
                                consensus_sentiment, consensus_nps,
                                source = "primary_kimi_consensus"
  active_learning_queue.jsonl  records that disagreed; original primary +
                                fallback predictions preserved + diagnostic
                                disagreement_reason for human review

Usage:
  python llm_consensus.py \\
      --input low_conf_samples.jsonl \\
      --primary deepseek-v4-flash --fallback kimi-k2.6 \\
      --output consensus_result.jsonl \\
      --unresolved-queue active_learning_queue.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient
from llm_labeler import label_one


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def primary_tags(rec: dict) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for lbl in rec.get("labels") or []:
        tid = (lbl or {}).get("tag_id")
        if tid and tid not in seen:
            out.append(tid); seen.add(tid)
    return out


def fallback_tags(parsed: dict | None) -> list[str]:
    if not parsed:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for lbl in parsed.get("labels") or []:
        tid = (lbl or {}).get("tag_id")
        if tid and tid not in seen:
            out.append(tid); seen.add(tid)
    return out


def soft_agree(primary_set: list[str], fallback_set: list[str],
               primary_sent: str | None, fallback_sent: str | None,
               primary_nps: str | None, fallback_nps: str | None) -> tuple[bool, list[str], list[str]]:
    pa, fa = set(primary_set), set(fallback_set)
    shared = [t for t in primary_set if t in fa]
    reasons: list[str] = []
    if not shared and (primary_set or fallback_set):
        reasons.append("no_shared_tags")
    if primary_sent != fallback_sent:
        reasons.append(f"sentiment_diff:{primary_sent}!={fallback_sent}")
    if primary_nps != fallback_nps:
        reasons.append(f"nps_diff:{primary_nps}!={fallback_nps}")
    if not primary_set and not fallback_set:
        reasons.append("both_zero_label")
    return (len(reasons) == 0), shared, reasons


async def run_fallback_on_records(client: LLMClient, records: list[dict],
                                  vendor: str, model: Optional[str]) -> dict[str, dict]:
    sem = client._get_semaphore(vendor)
    print(f"  Fallback concurrency: {sem._value}")

    async def one(rec):
        result = await label_one(client, rec, vendor=vendor, model=model)
        return rec.get("review_id"), result

    tasks = [asyncio.create_task(one(r)) for r in records]
    n_total = len(records)
    t0 = time.time()
    out: dict[str, dict] = {}
    n_done = n_success = 0
    for fut in asyncio.as_completed(tasks):
        rid, result = await fut
        n_done += 1
        if result.success:
            n_success += 1
        out[rid] = {
            "success": result.success,
            "labels": (result.parsed or {}).get("labels", []),
            "overall_sentiment": (result.parsed or {}).get("overall_sentiment"),
            "proxy_nps": (result.parsed or {}).get("proxy_nps"),
            "_meta": {
                "model": result.model_used,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cache_hit": result.cache_hit_tokens,
                "latency_ms": round(result.latency_ms, 1),
                "retries": result.retries,
                "error": result.error,
            },
        }
        if n_done % 100 == 0 or n_done == n_total:
            elapsed = time.time() - t0
            print(f"  Fallback [{n_done}/{n_total}] success={n_success} rate={n_done/elapsed:.2f}/s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="low_conf_samples.jsonl from low_conf_extractor")
    ap.add_argument("--source-text", type=Path, default=None,
                    help="Optional jsonl with original review text (joined by review_id) — required when input lacks 'text'")
    ap.add_argument("--primary", type=str, default="deepseek-v4-flash",
                    help="Primary model (informational; predictions already in input)")
    ap.add_argument("--fallback", type=str, default=None,
                    help="Fallback model name override; default uses client config default (Kimi: kimi-k2-turbo-preview)")
    ap.add_argument("--fallback-vendor", choices=["deepseek", "kimi"], default="kimi")
    ap.add_argument("--output", type=Path, required=True, help="consensus_result.jsonl")
    ap.add_argument("--unresolved-queue", type=Path, required=True, help="active_learning_queue.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}"); sys.exit(2)

    records = load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]

    needs_text = any(not (r.get("text") or "").strip() for r in records)
    if needs_text:
        if not args.source_text or not args.source_text.exists():
            print(f"❌ Input lacks 'text' field; please provide --source-text pointing to the original stratified jsonl"); sys.exit(2)
        text_idx = {r["review_id"]: r for r in load_jsonl(args.source_text) if r.get("review_id")}
        n_filled = 0
        for r in records:
            if not (r.get("text") or "").strip():
                src = text_idx.get(r.get("review_id"))
                if src:
                    r["text"] = src.get("text", "")
                    r["language"] = r.get("language") or src.get("language")
                    r["data_source"] = r.get("data_source") or src.get("data_source")
                    r["rating"] = r.get("rating") or src.get("rating")
                    n_filled += 1
        print(f"  Joined text from --source-text for {n_filled}/{len(records)} records")

    print(f"Input: {len(records)} records")
    print(f"Primary: {args.primary}  (predictions read from input.labels)")
    print(f"Fallback: {args.fallback_vendor} / {args.fallback or '(default)'}")

    client = LLMClient()
    fallback_map = asyncio.run(run_fallback_on_records(client, records, args.fallback_vendor, args.fallback))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.unresolved_queue.parent.mkdir(parents=True, exist_ok=True)
    n_consensus = n_unresolved = 0
    reason_counter = Counter()

    with args.output.open("w", encoding="utf-8") as f_ok, args.unresolved_queue.open("w", encoding="utf-8") as f_q:
        for rec in records:
            rid = rec.get("review_id")
            primary_set = primary_tags(rec)
            primary_sent = rec.get("overall_sentiment")
            primary_nps = rec.get("proxy_nps")
            fb = fallback_map.get(rid) or {}
            fb_set = fallback_tags(fb)
            fb_sent = fb.get("overall_sentiment")
            fb_nps = fb.get("proxy_nps")

            agree, shared, reasons = soft_agree(primary_set, fb_set, primary_sent, fb_sent, primary_nps, fb_nps)
            primary_lookup = {l.get("tag_id"): l for l in (rec.get("labels") or [])}

            if agree and fb.get("success", False):
                consensus_labels = [
                    {
                        "tag_id": t,
                        "tag_en": (primary_lookup.get(t) or {}).get("tag_en"),
                        "confidence": (primary_lookup.get(t) or {}).get("confidence"),
                    }
                    for t in shared[:3]
                ]
                out_rec = dict(rec)
                out_rec["consensus_labels"] = consensus_labels
                out_rec["consensus_sentiment"] = primary_sent
                out_rec["consensus_nps"] = primary_nps
                out_rec["consensus_source"] = f"{args.primary}+{args.fallback}_consensus"
                out_rec["fallback_pred"] = {
                    "labels": fb.get("labels"),
                    "overall_sentiment": fb_sent,
                    "proxy_nps": fb_nps,
                }
                f_ok.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_consensus += 1
            else:
                out_rec = dict(rec)
                out_rec["fallback_pred"] = {
                    "success": fb.get("success", False),
                    "labels": fb.get("labels"),
                    "overall_sentiment": fb_sent,
                    "proxy_nps": fb_nps,
                }
                out_rec["disagreement_reason"] = "; ".join(reasons) if reasons else "fallback_failed"
                out_rec["queue_priority"] = (
                    "high" if "no_shared_tags" in reasons else
                    "medium" if any(r.startswith(("sentiment_diff", "nps_diff")) for r in reasons) else
                    "low"
                )
                for r in reasons or ["fallback_failed"]:
                    reason_counter[r.split(":")[0].split("=")[0]] += 1
                f_q.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n_unresolved += 1

    n_total = len(records)
    print("\n" + "=" * 60)
    print(f"Consensus reached         : {n_consensus}/{n_total} = {n_consensus/n_total*100:.1f}%")
    print(f"Active-learning queued    : {n_unresolved}/{n_total} = {n_unresolved/n_total*100:.1f}%")
    print("=" * 60)
    print("\nDisagreement reasons (Top-10):")
    for reason, n in reason_counter.most_common(10):
        print(f"  {reason:30} ×{n}")
    print(f"\n✅ Consensus  : {args.output}")
    print(f"📥 Queue      : {args.unresolved_queue}")


if __name__ == "__main__":
    main()

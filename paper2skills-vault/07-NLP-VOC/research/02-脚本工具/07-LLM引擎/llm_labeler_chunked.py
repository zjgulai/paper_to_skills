"""D8 chunked runner for llm_labeler.py.

Avoids the asyncio.as_completed overhead on 87K tasks by running the underlying
labeler in sequential chunks of ~2000 records each, streaming results to a single
append-only output file.

Resumable: reads existing output, skips already-labeled review_ids.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import LLMClient
from llm_labeler import run_batch


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def existing_review_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("review_id")
            if rid:
                done.add(rid)
    return done


async def run_chunks(
    client: LLMClient,
    records: list[dict],
    output_path: Path,
    chunk_size: int,
    vendor: str,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = (len(records) + chunk_size - 1) // chunk_size
    overall_success = 0
    overall_total = 0
    overall_tokens_in = 0
    overall_cache_hit = 0
    t0 = time.time()

    for ci in range(total_chunks):
        chunk = records[ci * chunk_size:(ci + 1) * chunk_size]
        if not chunk:
            continue
        tmp_out = output_path.with_suffix(f".chunk{ci:04d}.jsonl")
        chunk_t0 = time.time()
        print(f"\n=== CHUNK {ci+1}/{total_chunks} "
              f"({len(chunk)} records, "
              f"overall progress {ci*chunk_size}/{len(records)}) ===",
              flush=True)
        summary = await run_batch(client, chunk, tmp_out, vendor=vendor)

        with tmp_out.open("r", encoding="utf-8") as src, \
             output_path.open("a", encoding="utf-8") as dst:
            for line in src:
                dst.write(line)
        tmp_out.unlink()

        overall_success += summary["n_success"]
        overall_total += summary["n_total"]
        overall_tokens_in += summary["tokens_in_total"]
        overall_cache_hit += summary.get("tokens_in_cache_hit_total", 0)
        elapsed = time.time() - t0
        rate = overall_total / elapsed if elapsed else 0
        eta = (len(records) - overall_total) / rate if rate else 0
        print(
            f"\n>>> CHUNK {ci+1} DONE | chunk time {time.time()-chunk_t0:.1f}s | "
            f"cumulative: {overall_total}/{len(records)} "
            f"succ={overall_success/overall_total*100:.1f}% "
            f"cache_hit={(overall_cache_hit/overall_tokens_in*100 if overall_tokens_in else 0):.1f}% "
            f"rate={rate:.2f}/s eta={eta/60:.1f}min",
            flush=True,
        )

    return {
        "n_total": overall_total,
        "n_success": overall_success,
        "success_rate": overall_success / overall_total if overall_total else 0.0,
        "tokens_in_total": overall_tokens_in,
        "tokens_in_cache_hit_total": overall_cache_hit,
        "cache_hit_rate": overall_cache_hit / overall_tokens_in if overall_tokens_in else 0.0,
        "elapsed_sec": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vendor", choices=["deepseek", "kimi"], default="deepseek")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Input not found: {args.input}", file=sys.stderr)
        return 2

    all_records = load_jsonl(args.input)
    if args.limit:
        all_records = all_records[: args.limit]

    done_ids = existing_review_ids(args.output)
    records = [r for r in all_records if r.get("review_id") not in done_ids]
    print(f"Loaded {len(all_records)} records, {len(done_ids)} already done, {len(records)} to label")

    if not records:
        print("✅ Nothing to do — output is complete.")
        return 0

    client = LLMClient()
    summary = asyncio.run(
        run_chunks(client, records, args.output, args.chunk_size, args.vendor)
    )

    summary_path = args.summary_out or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n🎉 DONE. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

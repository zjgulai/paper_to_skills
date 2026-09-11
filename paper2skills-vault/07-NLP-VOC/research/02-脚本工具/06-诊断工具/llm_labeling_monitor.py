"""D8 real-time monitor for streaming LLM labeling output.

Tails a jsonl file produced by llm_labeler.py and reports a sliding-window
rollup of: success rate, mean confidence, cache_hit ratio, throughput, latency.

Red lines (T8 / QA 场景 2):
  - success_rate >= 0.98
  - mean_confidence >= 0.70
  - cache_hit_ratio >= 0.85   (computed as tokens_in_cache_hit / tokens_in_total)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path


def follow(path: Path):
    """Generator yielding new lines as they're appended to path."""
    with path.open("r", encoding="utf-8") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            yield line


def load_existing(path: Path) -> list[dict]:
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


def rollup(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "msg": "empty window"}
    n_success = sum(1 for r in records if r.get("success"))
    confs: list[float] = []
    for r in records:
        for l in (r.get("labels") or []):
            c = l.get("confidence")
            if isinstance(c, (int, float)):
                confs.append(float(c))
    tokens_in = sum(int((r.get("_meta") or {}).get("tokens_in", 0)) for r in records)
    cache_hit = sum(int((r.get("_meta") or {}).get("cache_hit", 0)) for r in records)
    latencies = [float((r.get("_meta") or {}).get("latency_ms", 0)) for r in records]
    return {
        "n": n,
        "success_rate": n_success / n,
        "mean_confidence": (sum(confs) / len(confs)) if confs else 0.0,
        "cache_hit_ratio": (cache_hit / tokens_in) if tokens_in else 0.0,
        "tokens_in_total": tokens_in,
        "tokens_in_cache_hit_total": cache_hit,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
    }


def verdict(stats: dict, success_min: float, conf_min: float, cache_min: float) -> str:
    if stats.get("n", 0) == 0:
        return "—"
    flags = []
    if stats["success_rate"] < success_min:
        flags.append(f"success<{success_min:.2f}")
    if stats["mean_confidence"] < conf_min:
        flags.append(f"conf<{conf_min:.2f}")
    if stats["cache_hit_ratio"] < cache_min:
        flags.append(f"cache<{cache_min:.2f}")
    return "PASS" if not flags else "FAIL " + ",".join(flags)


def fmt_line(stats: dict, v: str) -> str:
    if stats.get("n", 0) == 0:
        return f"[--:--:--] empty"
    return (
        f"[{time.strftime('%H:%M:%S')}] "
        f"n={stats['n']:>5d} "
        f"succ={stats['success_rate']*100:5.1f}% "
        f"conf={stats['mean_confidence']:.2f} "
        f"cache={stats['cache_hit_ratio']*100:5.1f}% "
        f"lat={stats['mean_latency_ms']:6.0f}ms "
        f"| {v}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=Path, required=True, help="jsonl file to follow")
    ap.add_argument("--window", type=int, default=1000, help="sliding window size")
    ap.add_argument("--success-min", type=float, default=0.98)
    ap.add_argument("--conf-min", type=float, default=0.70)
    ap.add_argument("--cache-min", type=float, default=0.85)
    ap.add_argument("--report-out", type=Path, default=None,
                    help="Optional final-stats json (written on Ctrl-C)")
    ap.add_argument("--once", action="store_true",
                    help="Compute rollup over current file content and exit")
    ap.add_argument("--interval-sec", type=float, default=5.0,
                    help="How often to refresh stats while tailing")
    args = ap.parse_args()

    if not args.tail.parent.exists():
        print(f"❌ Parent dir not found: {args.tail.parent}", file=sys.stderr)
        sys.exit(2)

    if args.once:
        records = load_existing(args.tail)
        window = records[-args.window:] if args.window and args.window < len(records) else records
        stats = rollup(window)
        v = verdict(stats, args.success_min, args.conf_min, args.cache_min)
        print(fmt_line(stats, v))
        if args.report_out:
            args.report_out.write_text(
                json.dumps({"window_stats": stats, "verdict": v, "global_n": len(records)},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0 if v == "PASS" else 1

    print(f"Tailing {args.tail} (window={args.window}, interval={args.interval_sec}s)")
    print(f"Red lines: success>={args.success_min:.2f}, conf>={args.conf_min:.2f}, cache>={args.cache_min:.2f}")

    buf: deque[dict] = deque(maxlen=args.window)
    if args.tail.exists():
        existing = load_existing(args.tail)
        for r in existing[-args.window:]:
            buf.append(r)

    last_print = 0.0
    try:
        if not args.tail.exists():
            print("(file not yet created, waiting...)")
            while not args.tail.exists():
                time.sleep(1.0)
        for line in follow(args.tail):
            line = line.strip()
            if not line:
                continue
            try:
                buf.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            now = time.time()
            if now - last_print >= args.interval_sec:
                stats = rollup(list(buf))
                v = verdict(stats, args.success_min, args.conf_min, args.cache_min)
                print(fmt_line(stats, v), flush=True)
                last_print = now
    except KeyboardInterrupt:
        stats = rollup(list(buf))
        v = verdict(stats, args.success_min, args.conf_min, args.cache_min)
        print("\n=== final window ===")
        print(fmt_line(stats, v))
        if args.report_out:
            args.report_out.write_text(
                json.dumps({"window_stats": stats, "verdict": v}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0 if v == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)

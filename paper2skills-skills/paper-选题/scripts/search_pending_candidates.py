#!/usr/bin/env python3
"""Search papers for pending candidate queue items and write audit artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


BASE_DIR = Path(os.environ.get("PAPER2SKILLS_ROOT") or Path(__file__).resolve().parents[3])
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from paper2skills_common.candidates import DEFAULT_QUEUE_PATH
from paper2skills_common.paper_search import (
    DEFAULT_SEARCH_RUN_DIR,
    build_search_run,
    enrich_queue_with_search,
    fetch_arxiv_results,
    load_candidate_queue,
    write_json_document,
    write_search_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search arXiv papers for pending paper2skills candidates")
    parser.add_argument("--queue", type=Path, default=BASE_DIR / DEFAULT_QUEUE_PATH)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / DEFAULT_SEARCH_RUN_DIR)
    parser.add_argument("--max-results", type=int, default=5, help="arXiv results to fetch per candidate")
    parser.add_argument("--limit", type=int, default=None, help="Limit pending candidates searched")
    parser.add_argument("--sleep-seconds", type=float, default=3.0, help="Delay between arXiv API requests")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for each arXiv API request")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing artifacts")
    args = parser.parse_args()

    queue = load_candidate_queue(BASE_DIR, args.queue)
    fetcher = lambda query, max_results: fetch_arxiv_results(
        query,
        max_results=max_results,
        timeout=args.timeout,
    )
    run = build_search_run(
        queue,
        fetcher=fetcher,
        limit=args.limit,
        max_results_per_candidate=args.max_results,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(run["summary"], ensure_ascii=False, indent=2))

    if args.dry_run:
        preview = [
            {
                "topic_id": item["topic_id"],
                "search_query": item["search_query"],
                "top_papers": item["papers"][:3],
                "error": item["error"],
            }
            for item in run["results"][:5]
        ]
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0

    run_path = write_search_run(BASE_DIR, run, args.output_dir)
    enriched_queue = enrich_queue_with_search(queue, run)
    write_json_document(enriched_queue, args.queue)
    print(f"已写入搜索审计: {run_path}")
    print(f"已更新候选队列: {args.queue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

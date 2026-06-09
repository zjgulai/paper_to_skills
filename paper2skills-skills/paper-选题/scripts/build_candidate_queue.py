#!/usr/bin/env python3
"""Build the derived paper candidate queue from graph gaps and roadmaps."""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import json
import os
import sys
from pathlib import Path


BASE_DIR = Path(os.environ.get("PAPER2SKILLS_ROOT") or Path(__file__).resolve().parents[3])
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from paper2skills_common.candidates import (
    DEFAULT_QUEUE_PATH,
    build_candidate_queue,
    carry_forward_candidate_search,
    write_candidate_queue,
)


def _load_graph_analyzer():
    script = BASE_DIR / "paper2skills-skills" / "paper-skills-graph" / "scripts" / "skills_graph_analyzer.py"
    spec = importlib.util.spec_from_file_location("skills_graph_analyzer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load graph analyzer from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_graph_gaps(limit: int) -> list[dict]:
    module = _load_graph_analyzer()
    graph = module.SkillsGraph(str(BASE_DIR / "paper2skills-vault"))
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_graph()
    return graph.find_knowledge_gaps()[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper candidate queue")
    parser.add_argument("--dry-run", action="store_true", help="Print preview without writing queue")
    parser.add_argument("--limit", type=int, default=100, help="Maximum candidates to keep")
    parser.add_argument("--graph-limit", type=int, default=30, help="Maximum graph gaps to carry into the queue")
    parser.add_argument("--output", type=Path, default=BASE_DIR / DEFAULT_QUEUE_PATH)
    args = parser.parse_args()

    graph_gaps = build_graph_gaps(args.graph_limit)
    document = build_candidate_queue(BASE_DIR, graph_gaps=graph_gaps, limit=args.limit)
    if args.output.exists():
        try:
            existing_document = json.loads(args.output.read_text(encoding="utf-8"))
            document = carry_forward_candidate_search(document, existing_document)
        except json.JSONDecodeError:
            pass
    print(json.dumps(document["summary"], ensure_ascii=False, indent=2))

    if args.dry_run:
        print(json.dumps(document["candidates"][:10], ensure_ascii=False, indent=2))
        return 0

    write_candidate_queue(document, args.output)
    print(f"已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

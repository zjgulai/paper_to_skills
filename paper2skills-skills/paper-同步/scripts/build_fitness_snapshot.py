#!/usr/bin/env python3
"""Build a derived static fitness snapshot for Skill assets.

This is a deterministic governance snapshot, not live production fitness.
Runtime usage/SLO data can be added later without changing the output shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(os.environ.get("PAPER2SKILLS_ROOT") or Path(__file__).resolve().parents[3])
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from paper2skills_common.assets import build_asset_inventory


DEFAULT_OUTPUT = BASE_DIR / "paper2skills-vault" / "00-项目管理" / "fitness-snapshot.json"


def _score(item: dict) -> float:
    score = 0.0
    if item["relations_status"] == "wikilinked":
        score += 0.45
    elif item["relations_status"] == "present_without_wikilinks":
        score += 0.25
    if item["code_path"]:
        score += 0.30
    if item["has_tests"]:
        score += 0.20
    if item["verification_status"] == "vault_only":
        score += 0.05
    return round(min(score, 1.0), 3)


def build_document() -> dict:
    inventory = build_asset_inventory(BASE_DIR)
    scored = [(item["skill_id"], _score(item)) for item in inventory]
    values = [score for _, score in scored]
    distribution = {
        "above_0_75": sum(1 for value in values if value > 0.75),
        "0_6_to_0_75": sum(1 for value in values if 0.6 <= value <= 0.75),
        "0_4_to_0_6": sum(1 for value in values if 0.4 <= value < 0.6),
        "below_0_4": sum(1 for value in values if value < 0.4),
    }
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generator": "paper2skills-skills/paper-同步/scripts/build_fitness_snapshot.py",
        "note": "Derived static governance score from relations/code/test coverage; not live usage fitness.",
        "total_skills": len(scored),
        "avg_fitness": round(sum(values) / max(len(values), 1), 3),
        "distribution": distribution,
        "top_skills": sorted(scored, key=lambda row: row[1], reverse=True)[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build derived static fitness snapshot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    document = build_document()
    print(
        "扫描完成: "
        f"{document['total_skills']} 个 Skill, avg_fitness={document['avg_fitness']}"
    )
    if args.dry_run:
        print(json.dumps(document, ensure_ascii=False, indent=2)[:1000] + "...")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


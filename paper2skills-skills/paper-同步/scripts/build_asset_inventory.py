#!/usr/bin/env python3
"""Build a derived Skill asset inventory.

The inventory is generated from the filesystem and CLAUDE.md domain registry.
It is not a hand-maintained source of truth.
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


DEFAULT_OUTPUT = BASE_DIR / "paper2skills-vault" / "00-项目管理" / "skill_asset_inventory.json"


def build_document() -> dict:
    inventory = build_asset_inventory(BASE_DIR)
    return {
        "_README": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "paper2skills-skills/paper-同步/scripts/build_asset_inventory.py",
            "note": "Derived snapshot. Regenerate from vault/code filesystem; do not hand edit.",
        },
        "summary": {
            "total_skills": len(inventory),
            "with_code": sum(1 for item in inventory if item["code_path"]),
            "with_tests": sum(1 for item in inventory if item["has_tests"]),
            "relations_missing": sum(1 for item in inventory if item["relations_status"] == "missing"),
        },
        "skills": inventory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build derived paper2skills asset inventory")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing output")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    document = build_document()
    summary = document["summary"]
    print(
        "扫描完成: "
        f"{summary['total_skills']} 个 Skill, "
        f"{summary['with_code']} 个有 code, "
        f"{summary['with_tests']} 个有测试"
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


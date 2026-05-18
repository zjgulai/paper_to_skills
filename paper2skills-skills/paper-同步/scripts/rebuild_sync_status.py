#!/usr/bin/env python3
"""
rebuild_sync_status.py

一次性扫描 paper2skills-vault/ 所有 Skill-*.md 文件，重建 sync_status.json
作为存档快照。配合 P0-7 的「停止手动维护」策略使用。

用法:
    python3 rebuild_sync_status.py [--dry-run] [--output path/to/file]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(os.environ.get("PAPER2SKILLS_ROOT") or Path(__file__).resolve().parents[3])
VAULT_DIR = BASE_DIR / "paper2skills-vault"
CODE_DIR = BASE_DIR / "paper2skills-code"
DEFAULT_OUTPUT = VAULT_DIR / "07-资源库" / "sync_status.json"


def find_all_skill_files() -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for domain_dir in sorted(VAULT_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        if domain_dir.name in {"papers", "07-资源库", "00-项目管理"}:
            continue
        for skill_file in sorted(domain_dir.glob("Skill-*.md")):
            results.append((domain_dir.name, skill_file))
    return results


def detect_code_path(skill_filename: str) -> Path | None:
    module_name = skill_filename.lower().replace("skill-", "").replace(".md", "").replace("-", "_")
    if not CODE_DIR.exists():
        return None
    for domain_dir in CODE_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        candidate = domain_dir / module_name
        if candidate.exists():
            return candidate.relative_to(BASE_DIR)
    return None


def build_status() -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    status: dict = {
        "_README": {
            "generated_at": now,
            "generator": "paper2skills-skills/paper-同步/scripts/rebuild_sync_status.py",
            "note": "本文件为基于 vault 文件系统的自动重建快照。手动维护已废弃；以 vault 真实文件存在性为准。",
            "ground_truth": "paper2skills-vault/{01-16}-*/Skill-*.md",
        }
    }

    for domain_name, skill_file in find_all_skill_files():
        skill_key = skill_file.name
        code_path = detect_code_path(skill_key)
        entry = {
            "vault": {
                "synced": True,
                "timestamp": now,
                "path": str(skill_file.relative_to(BASE_DIR)),
                "domain": domain_name,
            },
            "github": {
                "synced": code_path is not None,
                "timestamp": now,
                "code_path": str(code_path) if code_path else None,
            },
            "feishu": {
                "synced": False,
                "error": "not tracked (manual maintenance deprecated)",
            },
        }
        status[skill_key] = entry

    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild sync_status.json from vault file system")
    parser.add_argument("--dry-run", action="store_true", help="只打印不写文件")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出路径")
    args = parser.parse_args()

    status = build_status()
    skill_count = sum(1 for k in status.keys() if not k.startswith("_"))
    code_count = sum(1 for k, v in status.items() if not k.startswith("_") and v["github"]["synced"])

    print(f"扫描完成: {skill_count} 个 Skill 卡片, 其中 {code_count} 个有对应 code 目录")

    if args.dry_run:
        print("dry-run: 不写入文件")
        print(json.dumps(status, ensure_ascii=False, indent=2)[:500] + "...")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    print(f"已写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

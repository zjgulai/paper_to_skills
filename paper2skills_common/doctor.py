"""Project health checks for paper2skills."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any

from paper2skills_common.assets import build_asset_inventory, iter_skill_files
from paper2skills_common.domains import load_domain_registry, project_root_from


def check_markdown_utf8(root: str | Path | None = None) -> list[dict[str, Any]]:
    project_root = project_root_from(Path(root) if root is not None else None)
    issues: list[dict[str, Any]] = []
    paths = [
        *project_root.glob("*.md"),
        *project_root.glob("paper2skills-skills/**/*.md"),
        *project_root.glob("paper2skills-vault/**/*.md"),
    ]
    for path in sorted(set(paths)):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            issues.append({"path": str(path.relative_to(project_root)), "type": "decode_error", "error": str(exc)})
            continue
        bad_count = sum(1 for ch in text if (ord(ch) < 32 and ch not in "\n\r\t") or ch == "\ufffd")
        if bad_count:
            issues.append({"path": str(path.relative_to(project_root)), "type": "control_chars", "count": bad_count})
    return issues


def check_python_ast(root: str | Path | None = None) -> list[dict[str, Any]]:
    project_root = project_root_from(Path(root) if root is not None else None)
    issues: list[dict[str, Any]] = []
    for path in sorted([*project_root.glob("mas/**/*.py"), *project_root.glob("paper2skills-skills/**/*.py"), *project_root.glob("paper2skills-code/**/*.py")]):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            issues.append({"path": str(path.relative_to(project_root)), "type": type(exc).__name__, "error": str(exc)})
    return issues


def check_code_dependencies() -> list[dict[str, Any]]:
    required = {
        "matplotlib": "paper2skills-code tests with plots/dashboard agents",
        "sklearn": "scikit-learn based code templates",
        "openai": "DataAgent non-simulate/mock import paths",
    }
    return [
        {"module": module, "reason": reason}
        for module, reason in required.items()
        if importlib.util.find_spec(module) is None
    ]


def check_domain_registry(root: str | Path | None = None) -> list[dict[str, Any]]:
    project_root = project_root_from(Path(root) if root is not None else None)
    registry = load_domain_registry(project_root)
    issues: list[dict[str, Any]] = []
    vault = project_root / "paper2skills-vault"
    for entry in registry.entries:
        if entry.vault_dir and not (vault / entry.vault_dir).exists() and entry.key != "nlp_voc":
            issues.append({"domain": entry.key, "vault_dir": entry.vault_dir, "type": "missing_vault_dir"})
    if "compliance" not in registry.by_key:
        issues.append({"domain": "compliance", "type": "missing_required_domain"})
    return issues


def check_skill_code_alignment(root: str | Path | None = None) -> dict[str, Any]:
    project_root = project_root_from(Path(root) if root is not None else None)
    inventory = build_asset_inventory(project_root)
    return {
        "total_skills": len(inventory),
        "with_code": sum(1 for item in inventory if item["code_path"]),
        "with_tests": sum(1 for item in inventory if item["has_tests"]),
        "relations_missing": sum(1 for item in inventory if item["relations_status"] == "missing"),
    }


def run_checks(root: str | Path | None = None, checks: list[str] | None = None) -> dict[str, Any]:
    selected = checks or ["markdown", "ast", "domains", "deps", "alignment"]
    result: dict[str, Any] = {}
    if "markdown" in selected:
        result["markdown"] = check_markdown_utf8(root)
    if "ast" in selected:
        result["ast"] = check_python_ast(root)
    if "domains" in selected:
        result["domains"] = check_domain_registry(root)
    if "deps" in selected:
        result["deps"] = check_code_dependencies()
    if "alignment" in selected:
        result["alignment"] = check_skill_code_alignment(root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper2skills project health checks")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="append",
        choices=["markdown", "ast", "domains", "deps", "alignment"],
        help="Run only selected checks. May be passed multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    result = run_checks(args.root, args.check)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for name, value in result.items():
            print(f"{name}: {value}")

    hard_failures = []
    for key in ("markdown", "ast", "domains"):
        hard_failures.extend(result.get(key, []))
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Skill asset inventory helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper2skills_common.domains import DomainRegistry, load_domain_registry, project_root_from


EXCLUDED_VAULT_DIRS = {"papers", "07-资源库", "00-项目管理"}


def skill_id_from_path(path: Path) -> str:
    return path.stem


def module_name_for_skill(skill_id: str) -> str:
    return skill_id.lower().replace("skill-", "").replace("-", "_")


def iter_skill_files(root: str | Path | None = None, registry: DomainRegistry | None = None) -> list[Path]:
    project_root = project_root_from(Path(root) if root is not None else None)
    registry = registry or load_domain_registry(project_root)
    vault = project_root / "paper2skills-vault"
    files: list[Path] = []
    allowed = {entry.vault_dir for entry in registry.entries}
    for domain_dir in sorted(vault.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name in EXCLUDED_VAULT_DIRS:
            continue
        if domain_dir.name not in allowed:
            continue
        files.extend(sorted(domain_dir.glob("Skill-*.md")))
    return files


def detect_code_path(root: str | Path, skill_id: str, domain_key: str | None = None) -> Path | None:
    project_root = project_root_from(Path(root))
    registry = load_domain_registry(project_root)
    code = project_root / "paper2skills-code"
    module_name = module_name_for_skill(skill_id)

    candidates: list[Path] = []
    if domain_key and domain_key in registry.by_key:
        candidates.append(code / domain_key / module_name)

    candidates.extend(sorted(code.glob(f"*/{module_name}")))
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.relative_to(project_root)
    return None


def _has_tests(project_root: Path, relative_code_path: Path | None) -> bool:
    if relative_code_path is None:
        return False
    code_dir = project_root / relative_code_path
    return any(code_dir.glob("test_*.py")) or any(code_dir.glob("*_test.py"))


def _relations_status(skill_file: Path) -> str:
    content = skill_file.read_text(encoding="utf-8")
    if "技能关联" not in content:
        return "missing"
    if "[[Skill-" in content:
        return "wikilinked"
    return "present_without_wikilinks"


def build_asset_inventory(root: str | Path | None = None) -> list[dict[str, Any]]:
    project_root = project_root_from(Path(root) if root is not None else None)
    registry = load_domain_registry(project_root)
    inventory: list[dict[str, Any]] = []

    for skill_file in iter_skill_files(project_root, registry):
        domain_vault = skill_file.parent.name
        domain_key = registry.vault_to_key.get(domain_vault, "unknown")
        skill_id = skill_id_from_path(skill_file)
        code_path = detect_code_path(project_root, skill_id, domain_key)
        has_tests = _has_tests(project_root, code_path)
        if code_path is None:
            verification_status = "vault_only"
        elif has_tests:
            verification_status = "has_tests"
        else:
            verification_status = "code_without_tests"

        inventory.append(
            {
                "skill_id": skill_id,
                "domain": domain_key,
                "vault_path": str(skill_file.relative_to(project_root)),
                "code_path": str(code_path) if code_path else None,
                "has_tests": has_tests,
                "verification_status": verification_status,
                "relations_status": _relations_status(skill_file),
            }
        )
    return inventory

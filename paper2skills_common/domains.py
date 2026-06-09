"""Domain registry derived from CLAUDE.md.

CLAUDE.md remains the human-maintained source of truth. This module provides a
small structured view for scripts that previously carried their own domain maps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DomainEntry:
    key: str
    vault_dir: str
    description: str
    skill_count: str
    code_status: str


class DomainInferenceError(ValueError):
    def __init__(self, skill_name: str, candidates: Iterable[str]) -> None:
        known = ", ".join(sorted(candidates))
        super().__init__(f"Cannot infer domain for {skill_name!r}. Pass --domain explicitly. Known domains: {known}")


@dataclass(frozen=True)
class DomainRegistry:
    entries: tuple[DomainEntry, ...]

    @property
    def by_key(self) -> dict[str, DomainEntry]:
        return {entry.key: entry for entry in self.entries}

    @property
    def vault_to_key(self) -> dict[str, str]:
        return {entry.vault_dir: entry.key for entry in self.entries if entry.vault_dir}

    def known_keys(self) -> list[str]:
        return sorted(self.by_key)

    def vault_dir_for(self, domain_key: str) -> str:
        entry = self.by_key.get(domain_key)
        if entry is None:
            raise DomainInferenceError(domain_key, self.known_keys())
        return entry.vault_dir


def project_root_from(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "CLAUDE.md").exists() and (candidate / "paper2skills-vault").exists():
            return candidate
    raise FileNotFoundError("Could not locate paper2skills project root from CLAUDE.md")


def _clean_table_cell(value: str) -> str:
    value = value.strip()
    value = value.replace("`", "")
    value = value.replace("~~", "")
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    return value.strip()


def _parse_domain_rows(claude_md: str) -> list[DomainEntry]:
    entries: list[DomainEntry] = []
    in_domain_section = False

    for line in claude_md.splitlines():
        if line.startswith("## Domain Mapping"):
            in_domain_section = True
            continue
        if in_domain_section and line.startswith("## ") and not line.startswith("## Domain Mapping"):
            break
        if not in_domain_section or not line.startswith("|"):
            continue

        cells = [_clean_table_cell(c) for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        key, vault_dir, description, skill_count, code_status = cells[:5]
        if key in {"English Directory", "-------------------"}:
            continue
        if not re.fullmatch(r"[a-z0-9_]+", key):
            continue
        entries.append(
            DomainEntry(
                key=key,
                vault_dir=vault_dir,
                description=description,
                skill_count=skill_count,
                code_status=code_status,
            )
        )

    if not entries:
        raise ValueError("No domain mapping rows found in CLAUDE.md")
    return entries


@lru_cache(maxsize=8)
def _load_domain_registry_cached(root_str: str) -> DomainRegistry:
    root = Path(root_str)
    claude_path = root / "CLAUDE.md"
    entries = _parse_domain_rows(claude_path.read_text(encoding="utf-8"))
    return DomainRegistry(entries=tuple(entries))


def load_domain_registry(root: str | Path | None = None) -> DomainRegistry:
    project_root = project_root_from(Path(root) if root is not None else None)
    return _load_domain_registry_cached(str(project_root))


def find_existing_skill_domain(root: Path, skill_name: str, registry: DomainRegistry | None = None) -> str | None:
    registry = registry or load_domain_registry(root)
    name = skill_name[:-3] if skill_name.endswith(".md") else skill_name
    filename = f"{name}.md"
    vault_dir = root / "paper2skills-vault"

    for entry in registry.entries:
        domain_dir = vault_dir / entry.vault_dir
        if (domain_dir / filename).exists():
            return entry.key
    return None


def infer_domain_or_raise(root: Path, skill_name: str, explicit_domain: str | None = None) -> str:
    registry = load_domain_registry(root)
    if explicit_domain:
        if explicit_domain in registry.by_key:
            return explicit_domain
        if explicit_domain in registry.vault_to_key:
            return registry.vault_to_key[explicit_domain]
        raise DomainInferenceError(explicit_domain, registry.known_keys())

    existing = find_existing_skill_domain(root, skill_name, registry)
    if existing:
        return existing
    raise DomainInferenceError(skill_name, registry.known_keys())

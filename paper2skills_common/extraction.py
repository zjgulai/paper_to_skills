"""Extraction bundle manifests for paper-to-Skill conversion."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from paper2skills_common.assets import module_name_for_skill
from paper2skills_common.domains import load_domain_registry, project_root_from


WORKFLOW_STATES = ("candidate", "selected", "extracted", "code_created", "verified", "reviewed", "synced")
ALLOWED_STATUS_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "candidate": ("selected",),
    "selected": ("extracted",),
    "extracted": ("code_created",),
    "code_created": ("verified",),
    "verified": ("reviewed",),
    "reviewed": ("synced",),
    "synced": (),
}


def validate_status_transition(current: str, next_status: str) -> bool:
    return next_status in ALLOWED_STATUS_TRANSITIONS.get(current, ())


def _safe_paper_id(candidate: dict[str, Any]) -> str:
    raw = candidate.get("arxiv_id") or candidate.get("paper_id") or candidate.get("topic_id") or "manual-paper"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw)).strip("-") or "manual-paper"


def _skill_id(candidate: dict[str, Any]) -> str:
    raw = candidate.get("skill_id") or candidate.get("topic") or candidate.get("topic_id") or "Skill-New-Paper"
    raw = str(raw).strip()
    if raw.startswith("Skill-"):
        return raw
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-")
    return f"Skill-{slug or 'New-Paper'}"


def verification_status_for_state(status: str) -> str:
    if status in {"verified", "reviewed", "synced"}:
        return "verified"
    if status in {"selected", "extracted", "code_created"}:
        return "pending_verification"
    return "not_started"


def build_skill_bundle_manifest(
    root: str | Path | None,
    candidate: dict[str, Any],
    *,
    status: str = "candidate",
) -> dict[str, Any]:
    if status not in WORKFLOW_STATES:
        raise ValueError(f"Unknown extraction status: {status}")

    project_root = project_root_from(Path(root) if root is not None else None)
    registry = load_domain_registry(project_root)
    domain = str(candidate.get("domain") or "unknown")
    if domain not in registry.by_key:
        raise ValueError(f"Unknown domain for extraction bundle: {domain}")

    vault_dir = registry.vault_dir_for(domain)
    skill_id = _skill_id(candidate)
    module_name = module_name_for_skill(skill_id)
    paper_id = _safe_paper_id(candidate)
    paper_dir = project_root / "paper2skills-vault" / "papers" / vault_dir / paper_id

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "paper2skills_common.extraction",
        "topic_id": candidate.get("topic_id"),
        "status": status,
        "allowed_next_statuses": list(ALLOWED_STATUS_TRANSITIONS[status]),
        "verification_status": verification_status_for_state(status),
        "domain": domain,
        "vault_dir": vault_dir,
        "skill_id": skill_id,
        "paper_id": paper_id,
        "paper_url": candidate.get("paper_url"),
        "arxiv_id": candidate.get("arxiv_id"),
        "paper_pdf_path": str((paper_dir / "paper.pdf").relative_to(project_root)),
        "notes_path": str((paper_dir / "notes.md").relative_to(project_root)),
        "extract_path": str((paper_dir / "extract.md").relative_to(project_root)),
        "verification_report_path": str((paper_dir / "verification_report.md").relative_to(project_root)),
        "skill_path": str((project_root / "paper2skills-vault" / vault_dir / f"{skill_id}.md").relative_to(project_root)),
        "code_path": str((project_root / "paper2skills-code" / domain / module_name).relative_to(project_root)),
        "expected_code_files": ["__init__.py", "model.py", "example.py", "test_model.py"],
        "relations_status": "pending_review",
        "candidate": candidate,
    }

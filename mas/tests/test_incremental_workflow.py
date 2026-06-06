"""Regression tests for the incremental paper2skills workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_candidate_queue_scores_graph_gaps_and_keeps_audit_fields():
    from paper2skills_common.candidates import build_candidate_queue, score_candidate

    gaps = [
        {
            "type": "missing_prerequisite",
            "priority": "P0",
            "source_skill": "Skill-Downstream",
            "missing_skill": "Skill-Causal-Baseline",
            "description": "Skill-Downstream 依赖的 Skill-Causal-Baseline 尚未建立",
        },
        {
            "type": "missing_bridge",
            "priority": "P1",
            "domain_a": "causal_inference",
            "domain_b": "recommendation",
            "description": "causal_inference 与 recommendation 之间缺少桥梁连接",
        },
    ]

    queue = build_candidate_queue(ROOT, graph_gaps=gaps, roadmap_items=[], user_priorities=[])

    assert queue["summary"]["total_candidates"] == 2
    first = queue["candidates"][0]
    assert first["topic_id"].startswith("GAP-P0-")
    assert first["domain"] == "causal_inference"
    assert first["gap_type"] == "missing_prerequisite"
    assert first["decision"] == "pending"
    assert first["reason"]
    assert first["score"] >= score_candidate(gap_priority="P1")
    assert {"topic_id", "domain", "gap_type", "keywords", "score", "decision", "reason"} <= set(first)


def test_candidate_queue_resolves_aliases_before_marking_pending():
    from paper2skills_common.candidates import build_candidate_queue

    queue = build_candidate_queue(
        ROOT,
        graph_gaps=[],
        roadmap_items=[
            {"skill": "Skill-A-B-Test-Design", "paper_direction": "pseudo gap", "priority": "P1"},
            {"skill": "Skill-VOC-Aspect-Extraction", "paper_direction": "migrated gap", "priority": "P1"},
        ],
        user_priorities=[],
    )

    by_topic = {item["topic"]: item for item in queue["candidates"]}
    assert by_topic["Skill-A-B-Test-Design"]["decision"] == "already_exists"
    assert by_topic["Skill-A-B-Test-Design"]["resolved_existing_skill"] == "Skill-AB-Experimental-Design"
    assert by_topic["Skill-VOC-Aspect-Extraction"]["decision"] == "external_migrated"
    assert queue["summary"]["pending"] == 0


def test_extraction_bundle_manifest_requires_known_state_and_verification_status():
    from paper2skills_common.extraction import (
        ALLOWED_STATUS_TRANSITIONS,
        build_skill_bundle_manifest,
        validate_status_transition,
    )

    candidate = {
        "topic_id": "USER-P0-001",
        "domain": "compliance",
        "skill_id": "Skill-Category-Policy-Risk",
        "arxiv_id": "2503.23213",
        "paper_url": "https://arxiv.org/abs/2503.23213",
        "keywords": "ecommerce compliance risk",
    }

    manifest = build_skill_bundle_manifest(ROOT, candidate, status="selected")

    assert "candidate" in ALLOWED_STATUS_TRANSITIONS
    assert validate_status_transition("candidate", "selected") is True
    assert validate_status_transition("verified", "synced") is False
    assert manifest["status"] == "selected"
    assert manifest["verification_status"] == "pending_verification"
    assert manifest["domain"] == "compliance"
    assert manifest["skill_path"].endswith("paper2skills-vault/21-合规决策/Skill-Category-Policy-Risk.md")
    assert manifest["code_path"].endswith("paper2skills-code/compliance/category_policy_risk")
    assert manifest["verification_report_path"].endswith("verification_report.md")


def test_graph_gaps_are_prioritized_and_limited_with_compliance_bridge():
    script = ROOT / "paper2skills-skills" / "paper-skills-graph" / "scripts" / "skills_graph_analyzer.py"
    spec = importlib.util.spec_from_file_location("skills_graph_analyzer", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    graph = module.SkillsGraph(str(ROOT / "paper2skills-vault"))
    graph.build_graph()
    gaps = graph.find_knowledge_gaps()

    assert gaps
    assert {gap["priority"] for gap in gaps} <= {"P0", "P1", "P2"}
    bridge_gaps = [gap for gap in gaps if gap["type"] == "missing_bridge"]
    assert len(bridge_gaps) <= 30
    assert any(gap.get("domain") == "compliance" for gap in gaps)
    assert all("topic_id" in gap and "score" in gap for gap in gaps)


def test_workflow_dry_run_chains_stages_without_writing_run_file(tmp_path):
    from paper2skills_common.workflow import run_incremental_workflow

    result = run_incremental_workflow(
        ROOT,
        mode="one-topic",
        dry_run=True,
        write_run=False,
        run_output_dir=tmp_path,
        graph_limit=5,
    )

    assert result["dry_run"] is True
    assert result["status"] == "completed"
    assert [stage["name"] for stage in result["stages"]] == [
        "doctor",
        "graph_gaps",
        "candidate_queue",
        "extraction_bundle",
        "verification_plan",
        "sync_dry_run",
        "inventory_snapshot",
    ]
    assert result["summary"]["candidate_count"] >= 1
    assert list(tmp_path.glob("*.json")) == []

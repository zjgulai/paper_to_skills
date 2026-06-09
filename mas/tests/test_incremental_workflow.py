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


def test_candidate_queue_rebuild_carries_forward_search_audit_by_topic_id():
    from paper2skills_common.candidates import carry_forward_candidate_search

    rebuilt = {
        "candidates": [
            {"topic_id": "GAP-P2-001-DOMAIN-REVIEW", "decision": "pending"},
            {"topic_id": "GAP-P2-999-REMOVED", "decision": "pending"},
        ]
    }
    existing = {
        "candidates": [
            {
                "topic_id": "GAP-P2-001-DOMAIN-REVIEW",
                "decision": "selected",
                "workflow_status": "verified",
                "verification_status": "verified",
                "skill_id": "Skill-Compliance-Scored-Guardrail-Orchestration",
                "paper_search": {"last_run_id": "previous-run"},
                "selected_outputs": {"skill_path": "paper2skills-vault/21-合规决策/example.md"},
            },
            {
                "topic_id": "OLD-GAP",
                "decision": "pending",
                "paper_search": {"last_run_id": "old-run"},
            },
        ]
    }

    merged = carry_forward_candidate_search(rebuilt, existing)

    by_topic = {candidate["topic_id"]: candidate for candidate in merged["candidates"]}
    assert by_topic["GAP-P2-001-DOMAIN-REVIEW"]["paper_search"]["last_run_id"] == "previous-run"
    assert by_topic["GAP-P2-001-DOMAIN-REVIEW"]["decision"] == "selected"
    assert by_topic["GAP-P2-001-DOMAIN-REVIEW"]["workflow_status"] == "verified"
    assert by_topic["GAP-P2-001-DOMAIN-REVIEW"]["skill_id"] == "Skill-Compliance-Scored-Guardrail-Orchestration"
    assert by_topic["GAP-P2-001-DOMAIN-REVIEW"]["selected_outputs"]["skill_path"].endswith("example.md")
    assert "paper_search" not in by_topic["GAP-P2-999-REMOVED"]


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


def test_code_template_preflight_detects_missing_optional_dependency(tmp_path):
    conftest_path = ROOT / "paper2skills-code" / "conftest.py"
    spec = importlib.util.spec_from_file_location("paper2skills_code_conftest", conftest_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "model.py").write_text(
        "import definitely_missing_optional_dep\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "test_model.py"
    test_path.write_text("from .model import Something\n", encoding="utf-8")

    missing = module.missing_optional_dependencies_for_test_path(
        test_path,
        optional_modules=("definitely_missing_optional_dep",),
    )

    assert missing == ["definitely_missing_optional_dep"]


def test_paper_search_scores_candidate_matches_and_preserves_audit_fields():
    from paper2skills_common.paper_search import build_arxiv_query, enrich_queue_with_search, rank_papers_for_candidate

    candidate = {
        "topic_id": "GAP-P2-005-MISSING-BRIDGE",
        "domain": "recommendation",
        "gap_type": "missing_bridge",
        "keywords": "recommendation advertising cross-domain ecommerce",
        "decision": "pending",
    }
    papers = [
        {
            "arxiv_id": "2601.00001",
            "title": "Cross-Domain Recommendation and Advertising Optimization for E-Commerce",
            "abstract": "We present a practical benchmark, experiments, ablation, and implementation for recommendation advertising in ecommerce.",
            "published": "2026-01-04T00:00:00Z",
            "url": "https://arxiv.org/abs/2601.00001",
        },
        {
            "arxiv_id": "2401.00002",
            "title": "A Survey of Abstract Optimization",
            "abstract": "This survey reviews unrelated theory without experiments.",
            "published": "2024-01-04T00:00:00Z",
            "url": "https://arxiv.org/abs/2401.00002",
        },
    ]

    ranked = rank_papers_for_candidate(candidate, papers)
    query = build_arxiv_query(candidate)
    queue = {"candidates": [candidate.copy()]}
    run = {
        "run_id": "test-run",
        "results": [{"topic_id": candidate["topic_id"], "papers": ranked, "search_query": query}],
    }
    enriched = enrich_queue_with_search(queue, run)

    assert "all:recommendation" in query
    assert ranked[0]["arxiv_id"] == "2601.00001"
    assert ranked[0]["search_score"] > ranked[1]["search_score"]
    assert ranked[0]["recommendation"] == "candidate"
    assert enriched["candidates"][0]["decision"] == "pending"
    assert enriched["candidates"][0]["paper_search"]["last_run_id"] == "test-run"
    assert enriched["candidates"][0]["paper_search"]["top_papers"][0]["arxiv_id"] == "2601.00001"

    weak_bridge = {
        "topic_id": "GAP-P2-003-MISSING-BRIDGE",
        "domain": "logistics",
        "gap_type": "missing_bridge",
        "keywords": "logistics visual_content cross-domain ecommerce",
        "gap_ref": {"domain_a": "logistics", "domain_b": "visual_content"},
        "decision": "pending",
    }
    weak_ranked = rank_papers_for_candidate(
        weak_bridge,
        [
            {
                "arxiv_id": "2601.00003",
                "title": "Foundation Models for Medical Images",
                "abstract": "We present visual content benchmarks and implementation details for image understanding.",
                "published": "2026-01-04T00:00:00Z",
                "url": "https://arxiv.org/abs/2601.00003",
            }
        ],
    )

    assert weak_ranked[0]["recommendation"] != "candidate"

    preserved = enrich_queue_with_search(
        {
            "candidates": [
                {
                    **candidate,
                    "paper_search": {
                        "last_run_id": "previous-run",
                        "top_papers": [{"arxiv_id": "2601.00001", "title": "Preserved Paper"}],
                    },
                }
            ]
        },
        {
            "run_id": "failed-run",
            "source": "arxiv",
            "results": [
                {
                    "topic_id": candidate["topic_id"],
                    "search_query": query,
                    "attempted_queries": [query],
                    "error": "URLError: timeout",
                    "papers": [],
                }
            ],
        },
    )

    preserved_search = preserved["candidates"][0]["paper_search"]
    assert preserved_search["last_successful_run_id"] == "previous-run"
    assert preserved_search["last_error_run_id"] == "failed-run"
    assert preserved_search["top_papers"][0]["arxiv_id"] == "2601.00001"

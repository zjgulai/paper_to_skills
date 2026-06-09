from __future__ import annotations

from .model import ComplianceScoredGuardrailOrchestrator, baby_compliance_guardrails, demo_candidates


def test_orchestrator_selects_compliant_candidate():
    orchestrator = ComplianceScoredGuardrailOrchestrator(baby_compliance_guardrails(), threshold=0.88)
    result = orchestrator.select_best(demo_candidates())

    assert result.status == "accepted"
    assert result.selected is not None
    assert result.selected["title"] == "Compliance-gated listing response"
    assert result.compliance_score >= 0.88


def test_pii_and_policy_terms_reduce_score():
    orchestrator = ComplianceScoredGuardrailOrchestrator(baby_compliance_guardrails(), threshold=0.9)
    bad = demo_candidates()[0]
    scored = orchestrator.score(bad)

    assert scored["compliance_score"] < 0.75
    reasons = [reason for violation in scored["violations"] for reason in violation["reasons"]]
    assert "pii:email" in reasons
    assert any(reason.startswith("policy_term:") for reason in reasons)


def test_missing_schema_routes_to_review():
    orchestrator = ComplianceScoredGuardrailOrchestrator(baby_compliance_guardrails(), threshold=0.9)
    result = orchestrator.select_best(
        [
            {
                "title": "Incomplete response",
                "body": "Based on evidence, review CPSC and EU GPSR risk before publication.",
            }
        ]
    )

    assert result.status == "human_review_required"
    assert result.selected is not None
    assert result.compliance_score < 0.9

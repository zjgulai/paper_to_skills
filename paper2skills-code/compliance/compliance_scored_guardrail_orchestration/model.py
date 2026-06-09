"""Compliance-scored best-of-N guardrail orchestration.

This is a standard-library implementation of the core idea in arXiv:2606.01513:
score multiple generated candidates with weighted guardrails, return the best
candidate once it crosses a compliance threshold, and preserve decision
metadata for audit.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable


Candidate = dict[str, object]
Violation = dict[str, object]


@dataclass(frozen=True)
class GuardrailRule:
    """A weighted rule that scores one candidate in the [0, 1] interval."""

    name: str
    weight: float
    checker: Callable[[Candidate], tuple[float, list[str]]]
    blocking_threshold: float = 0.0

    def evaluate(self, candidate: Candidate) -> tuple[float, list[str], bool]:
        score, reasons = self.checker(candidate)
        score = max(0.0, min(1.0, float(score)))
        blocked = score <= self.blocking_threshold and bool(reasons)
        return score, reasons, blocked


@dataclass
class SelectionResult:
    selected: Candidate | None
    compliance_score: float
    status: str
    attempts: int
    elapsed_ms: float
    threshold: float
    violations: list[Violation] = field(default_factory=list)
    scored_candidates: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "compliance_score": round(self.compliance_score, 4),
            "status": self.status,
            "attempts": self.attempts,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "threshold": self.threshold,
            "violations": self.violations,
            "scored_candidates": self.scored_candidates,
        }


def _text(candidate: Candidate) -> str:
    fields = [
        str(candidate.get("title") or ""),
        str(candidate.get("body") or ""),
        str(candidate.get("evidence_summary") or ""),
        str(candidate.get("image_ocr") or ""),
    ]
    return "\n".join(fields)


def schema_rule(required_fields: Iterable[str]) -> Callable[[Candidate], tuple[float, list[str]]]:
    required = tuple(required_fields)

    def check(candidate: Candidate) -> tuple[float, list[str]]:
        missing = [field for field in required if not candidate.get(field)]
        if not missing:
            return 1.0, []
        return max(0.0, 1.0 - len(missing) / len(required)), [f"missing_field:{field}" for field in missing]

    return check


def pii_rule(candidate: Candidate) -> tuple[float, list[str]]:
    text = _text(candidate)
    patterns = {
        "email": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "card": r"\b(?:\d[ -]*?){13,16}\b",
    }
    hits = [name for name, pattern in patterns.items() if re.search(pattern, text, flags=re.IGNORECASE)]
    if not hits:
        return 1.0, []
    return max(0.0, 1.0 - 0.35 * len(hits)), [f"pii:{hit}" for hit in hits]


def moderation_rule(candidate: Candidate) -> tuple[float, list[str]]:
    banned = [
        "hide the defect",
        "ignore the recall",
        "guaranteed safe",
        "no certification needed",
        "bypass policy",
    ]
    lowered = _text(candidate).lower()
    hits = [term for term in banned if term in lowered]
    if not hits:
        return 1.0, []
    return max(0.0, 1.0 - 0.4 * len(hits)), [f"policy_term:{term}" for term in hits]


def baby_product_domain_rule(candidate: Candidate) -> tuple[float, list[str]]:
    text = _text(candidate).lower()
    required_concepts = {
        "market": ("us", "eu", "uk", "cpsc", "fda", "gpsr", "rapex", "ce"),
        "risk": ("risk", "hazard", "recall", "injury", "safety"),
        "action": ("review", "test", "certification", "evidence", "manual", "human"),
    }
    missing = [
        concept
        for concept, aliases in required_concepts.items()
        if not any(alias in text for alias in aliases)
    ]
    if not missing:
        return 1.0, []
    return max(0.0, 1.0 - len(missing) / len(required_concepts)), [f"missing_domain_concept:{item}" for item in missing]


def evidence_grounding_rule(candidate: Candidate) -> tuple[float, list[str]]:
    evidence = str(candidate.get("evidence_summary") or "")
    body = str(candidate.get("body") or "")
    if len(evidence.strip()) >= 20 and any(token in body.lower() for token in ("based on", "evidence", "record")):
        return 1.0, []
    if len(evidence.strip()) >= 20:
        return 0.75, ["evidence_not_explicitly_cited"]
    return 0.25, ["missing_evidence_summary"]


def baby_compliance_guardrails() -> list[GuardrailRule]:
    return [
        GuardrailRule("schema", 0.25, schema_rule(("title", "body", "evidence_summary", "recommended_action"))),
        GuardrailRule("pii", 0.25, pii_rule),
        GuardrailRule("moderation", 0.20, moderation_rule),
        GuardrailRule("baby_domain_rules", 0.20, baby_product_domain_rule),
        GuardrailRule("evidence_grounding", 0.10, evidence_grounding_rule),
    ]


class ComplianceScoredGuardrailOrchestrator:
    def __init__(self, rules: Iterable[GuardrailRule], threshold: float = 0.9, timeout_seconds: float = 20.0):
        self.rules = list(rules)
        if not self.rules:
            raise ValueError("At least one guardrail rule is required")
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds

    def score(self, candidate: Candidate) -> dict[str, object]:
        total_weight = sum(rule.weight for rule in self.rules)
        weighted_score = 0.0
        violations: list[Violation] = []
        blocked = False
        for rule in self.rules:
            rule_score, reasons, rule_blocked = rule.evaluate(candidate)
            weighted_score += rule.weight * rule_score
            if reasons:
                violations.append({"rule": rule.name, "score": rule_score, "reasons": reasons})
            blocked = blocked or rule_blocked

        compliance_score = weighted_score / total_weight if total_weight else 0.0
        return {
            "candidate": candidate,
            "compliance_score": compliance_score,
            "blocked": blocked,
            "violations": violations,
        }

    def select_best(self, candidates: Iterable[Candidate]) -> SelectionResult:
        start = time.perf_counter()
        best: dict[str, object] | None = None
        scored_candidates: list[dict[str, object]] = []
        attempts = 0

        for candidate in candidates:
            if time.perf_counter() - start > self.timeout_seconds:
                break
            attempts += 1
            scored = self.score(candidate)
            scored_candidates.append(
                {
                    "title": candidate.get("title"),
                    "compliance_score": round(float(scored["compliance_score"]), 4),
                    "blocked": scored["blocked"],
                    "violations": scored["violations"],
                }
            )
            if not scored["blocked"] and (
                best is None or float(scored["compliance_score"]) > float(best["compliance_score"])
            ):
                best = scored
            if best is not None and float(best["compliance_score"]) >= self.threshold:
                break

        elapsed_ms = (time.perf_counter() - start) * 1000
        if best is None:
            return SelectionResult(
                selected=None,
                compliance_score=0.0,
                status="human_review_required",
                attempts=attempts,
                elapsed_ms=elapsed_ms,
                threshold=self.threshold,
                violations=[violation for item in scored_candidates for violation in item["violations"]],
                scored_candidates=scored_candidates,
            )

        score = float(best["compliance_score"])
        status = "accepted" if score >= self.threshold else "human_review_required"
        return SelectionResult(
            selected=best["candidate"],
            compliance_score=score,
            status=status,
            attempts=attempts,
            elapsed_ms=elapsed_ms,
            threshold=self.threshold,
            violations=list(best["violations"]),
            scored_candidates=scored_candidates,
        )


def demo_candidates() -> list[Candidate]:
    return [
        {
            "title": "Unsafe launch copy",
            "body": "This baby monitor is guaranteed safe. No certification needed. Contact jane@example.com.",
            "evidence_summary": "Draft listing copy from supplier intake form.",
            "recommended_action": "publish",
        },
        {
            "title": "Compliance-gated listing response",
            "body": (
                "Based on certification evidence, route the US launch through CPSC safety review, "
                "confirm EU GPSR traceability, and send missing battery-test records to human review."
            ),
            "evidence_summary": "Supplier provided battery report, draft manual, and US/EU target-market plan.",
            "recommended_action": "human_review_before_publish",
        },
    ]


def main() -> None:
    orchestrator = ComplianceScoredGuardrailOrchestrator(baby_compliance_guardrails(), threshold=0.88)
    print(json.dumps(orchestrator.select_best(demo_candidates()).as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

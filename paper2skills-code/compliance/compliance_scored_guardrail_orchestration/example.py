from __future__ import annotations

from .model import ComplianceScoredGuardrailOrchestrator, baby_compliance_guardrails, demo_candidates


def run_example() -> dict:
    orchestrator = ComplianceScoredGuardrailOrchestrator(baby_compliance_guardrails(), threshold=0.88)
    return orchestrator.select_best(demo_candidates()).as_dict()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), ensure_ascii=False, indent=2))

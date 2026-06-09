#!/usr/bin/env python3
"""Playbook quality checks — run before build or in CI."""

import json, sys, re
from pathlib import Path

def check_dup_problem_solved(playbook_dir: Path) -> list[str]:
    data = json.loads((playbook_dir / "assets" / "playbook-data.json").read_text())
    issues = []
    for s in data["skills"]:
        ps = s.get("problem_solved", "")
        algo = s.get("algorithm_summary", "")
        if ps and algo and ps[:60] == algo[:60]:
            issues.append(f"dup_ps: {s['skill_id']}")
    return issues

def check_missing_biz_context(playbook_dir: Path) -> list[str]:
    data = json.loads((playbook_dir / "assets" / "playbook-data.json").read_text())
    return [
        f"missing_biz_role: {s['skill_id']}"
        for s in data["skills"]
        if not s.get("biz_role")
    ]

def check_workflow_yamls(wf_dir: Path) -> list[str]:
    import yaml
    issues = []
    for yf in wf_dir.glob("*.yaml"):
        try:
            data = yaml.safe_load(yf.read_text(encoding="utf-8"))
            if "id" not in data:
                issues.append(f"missing_id: {yf.name}")
            if "steps" not in data or not data["steps"]:
                issues.append(f"no_steps: {yf.name}")
            has_measure = any(
                "measure" in s.get("id","").lower() or "度量" in s.get("name","")
                for s in data.get("steps", [])
            )
            if not has_measure:
                issues.append(f"no_measurement_step: {data.get('id', yf.name)}")
        except Exception as e:
            issues.append(f"yaml_error: {yf.name}: {e}")
    return issues

def check_required_pages(playbook_dir: Path) -> list[str]:
    required = [
        "index.html", "agents.html", "ai-roadmap.html",
        "assets/style.css", "assets/graph-data.json", "assets/ego-graph.js",
    ]
    return [f"missing_page: {p}" for p in required
            if not (playbook_dir / p).exists()]

if __name__ == "__main__":
    root = Path(__file__).parents[3]
    playbook_dir = root / "playbook"
    wf_dir = root / "paper2skills-skills" / "paper-workflow" / "definitions"

    all_issues = []
    checks = [
        ("Duplicate problem_solved", check_dup_problem_solved(playbook_dir)),
        ("Missing biz_context", check_missing_biz_context(playbook_dir)),
        ("Workflow YAML issues", check_workflow_yamls(wf_dir)),
        ("Required pages", check_required_pages(playbook_dir)),
    ]

    for label, issues in checks:
        if issues:
            print(f"\n[WARN] {label} ({len(issues)} issues):")
            for issue in issues[:5]:
                print(f"  {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues)-5} more")
            all_issues.extend(issues)
        else:
            print(f"[OK] {label}")

    critical = [i for i in all_issues if i.startswith("missing_page") or i.startswith("yaml_error")]
    if critical:
        print(f"\n❌ {len(critical)} critical issues found")
        sys.exit(1)
    elif all_issues:
        print(f"\n⚠ {len(all_issues)} warnings (non-blocking)")
        sys.exit(0)
    else:
        print(f"\n✅ All checks passed")
        sys.exit(0)

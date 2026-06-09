"""Dry-run first orchestration for the incremental paper2skills workflow."""

from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from paper2skills_common.assets import build_asset_inventory
from paper2skills_common.candidates import build_candidate_queue
from paper2skills_common.doctor import run_checks
from paper2skills_common.domains import project_root_from
from paper2skills_common.extraction import build_skill_bundle_manifest


def _load_graph_analyzer(root: Path) -> Any:
    script = root / "paper2skills-skills" / "paper-skills-graph" / "scripts" / "skills_graph_analyzer.py"
    spec = importlib.util.spec_from_file_location("skills_graph_analyzer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load graph analyzer from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage(name: str, status: str = "completed", **data: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **data}


def run_incremental_workflow(
    root: str | Path | None = None,
    *,
    mode: str = "dry-run",
    dry_run: bool = True,
    write_run: bool = True,
    run_output_dir: str | Path | None = None,
    graph_limit: int = 30,
    batch_size: int = 5,
) -> dict[str, Any]:
    project_root = project_root_from(Path(root) if root is not None else None)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    stages: list[dict[str, Any]] = []

    doctor = run_checks(project_root)
    stages.append(
        _stage(
            "doctor",
            markdown_issues=len(doctor.get("markdown", [])),
            ast_issues=len(doctor.get("ast", [])),
            domain_issues=len(doctor.get("domains", [])),
            missing_deps=[item["module"] for item in doctor.get("deps", [])],
            alignment=doctor.get("alignment", {}),
        )
    )

    graph_module = _load_graph_analyzer(project_root)
    graph = graph_module.SkillsGraph(str(project_root / "paper2skills-vault"))
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_graph()
    gaps = graph.find_knowledge_gaps()[:graph_limit]
    stages.append(
        _stage(
            "graph_gaps",
            total_gaps=len(gaps),
            p0=sum(1 for gap in gaps if gap.get("priority") == "P0"),
            p1=sum(1 for gap in gaps if gap.get("priority") == "P1"),
            p2=sum(1 for gap in gaps if gap.get("priority") == "P2"),
            sample=gaps[:5],
        )
    )

    queue_limit = max(batch_size, 5) if mode in {"one-topic", "one-paper"} else max(batch_size, 1)
    queue = build_candidate_queue(project_root, graph_gaps=gaps, roadmap_items=[], user_priorities=[], limit=queue_limit)
    candidates = queue["candidates"]
    stages.append(_stage("candidate_queue", summary=queue["summary"], candidates=candidates))

    extraction_candidate = next((item for item in candidates if item.get("gap_type") != "domain_review"), None)
    if extraction_candidate is None and candidates:
        extraction_candidate = candidates[0]
    bundle = None
    if extraction_candidate:
        bundle = build_skill_bundle_manifest(project_root, extraction_candidate, status="selected")
    stages.append(_stage("extraction_bundle", bundle=bundle))

    verification_plan = {
        "required_before_sync": [
            "python3 -m paper2skills_common.doctor --json",
            "python3 -m pytest",
            "python3 -m pytest paper2skills-code -q",
        ],
        "code_template_dependency_note": "Install paper2skills-code/requirements-lock.txt before expecting template tests to execute.",
    }
    stages.append(_stage("verification_plan", plan=verification_plan))

    stages.append(
        _stage(
            "sync_dry_run",
            commands=[
                "python3 paper2skills-skills/paper-同步/scripts/rebuild_sync_status.py --dry-run",
                "python3 paper2skills-skills/paper-同步/scripts/build_asset_inventory.py --dry-run",
                "python3 paper2skills-skills/paper-同步/scripts/build_fitness_snapshot.py --dry-run",
            ],
        )
    )

    inventory = build_asset_inventory(project_root)
    stages.append(
        _stage(
            "inventory_snapshot",
            total_skills=len(inventory),
            with_code=sum(1 for item in inventory if item["code_path"]),
            with_tests=sum(1 for item in inventory if item["has_tests"]),
            relations_missing=sum(1 for item in inventory if item["relations_status"] == "missing"),
        )
    )

    result = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "dry_run": dry_run,
        "status": "completed",
        "stages": stages,
        "summary": {
            "candidate_count": len(candidates),
            "selected_topic_id": extraction_candidate["topic_id"] if extraction_candidate else None,
            "declared_generated_outputs": [
                "paper2skills-vault/00-项目管理/paper_candidate_queue.json",
                "paper2skills-vault/00-项目管理/workflow_runs/<run_id>.json",
            ],
        },
    }

    if write_run:
        output_dir = Path(run_output_dir) if run_output_dir is not None else project_root / "paper2skills-vault" / "00-项目管理" / "workflow_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{run_id}.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["run_path"] = str(output.relative_to(project_root) if output.is_relative_to(project_root) else output)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the incremental paper2skills workflow driver")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Keep dry-run mode. This is the default.")
    parser.add_argument("--apply", action="store_true", help="Mark the run as non-dry-run for future mutating stages.")
    parser.add_argument("--one-topic", action="store_true")
    parser.add_argument("--one-paper", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--graph-limit", type=int, default=30)
    parser.add_argument("--run-output-dir", type=Path, default=None)
    parser.add_argument("--no-write-run", action="store_true")
    args = parser.parse_args()

    mode = "dry-run"
    if args.one_topic:
        mode = "one-topic"
    elif args.one_paper:
        mode = "one-paper"
    elif args.batch:
        mode = "batch"

    result = run_incremental_workflow(
        args.root,
        mode=mode,
        dry_run=not args.apply,
        write_run=not args.no_write_run,
        run_output_dir=args.run_output_dir,
        graph_limit=args.graph_limit,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Darwin-style autoresearch evolution loops for paper2skills governance."""

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
from paper2skills_common.domains import project_root_from


DARWIN_MECHANISMS = ["mutation", "heredity", "selection", "competition", "niche"]
DEFAULT_OUTPUT_DIR = Path("paper2skills-vault/00-项目管理/darwin_evolution_runs")
DEFAULT_REPORT = Path("paper2skills-vault/00-项目管理/darwin_evolution_20loop_report.md")


def _load_graph_analyzer(root: Path) -> Any:
    script = root / "paper2skills-skills" / "paper-skills-graph" / "scripts" / "skills_graph_analyzer.py"
    spec = importlib.util.spec_from_file_location("skills_graph_analyzer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load graph analyzer from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph_state(root: Path, graph_limit: int) -> dict[str, Any]:
    module = _load_graph_analyzer(root)
    graph = module.SkillsGraph(str(root / "paper2skills-vault"))
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_graph()
    gaps = graph.find_knowledge_gaps()[:graph_limit]
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "density": round(len(graph.edges) / max(len(graph.nodes), 1), 3),
        "gaps": gaps,
        "p0": sum(1 for gap in gaps if gap.get("priority") == "P0"),
        "p1": sum(1 for gap in gaps if gap.get("priority") == "P1"),
        "p2": sum(1 for gap in gaps if gap.get("priority") == "P2"),
    }


def _domain_counts(inventory: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in inventory:
        counts[item["domain"]] = counts.get(item["domain"], 0) + 1
    return counts


def _fitness(candidate: dict[str, Any], domain_counts: dict[str, int]) -> float:
    score = min(int(candidate.get("score", 0)) / 140.0, 0.95)
    if candidate.get("source") == "skills_graph_gaps":
        score += 0.04
    if candidate.get("gap_type") == "missing_bridge":
        score += 0.06
    if candidate.get("gap_type") == "thin_domain":
        score += 0.08
    if candidate.get("gap_type") == "domain_review":
        score -= 0.14
    if candidate.get("decision") == "already_exists":
        score -= 0.42

    domain = str(candidate.get("domain", "unknown"))
    count = domain_counts.get(domain, 0)
    if 0 < count <= 5:
        score += 0.08
    elif count >= 30:
        score -= 0.06
    return round(max(0.0, min(score, 1.0)), 3)


def _gate(fitness: float) -> str:
    if fitness >= 0.7:
        return "inherit"
    if fitness >= 0.4:
        return "observe"
    return "reject"


def _action_for(candidate: dict[str, Any], gate: str) -> str:
    if candidate.get("decision") == "already_exists":
        return "competition_review_existing_skill"
    gap_type = candidate.get("gap_type")
    if gate == "reject":
        return "skip_or_manual_review"
    if gap_type == "missing_bridge":
        return "autoresearch_bridge_skill"
    if gap_type == "thin_domain":
        return "autoresearch_thin_domain_skill"
    if gap_type == "domain_review":
        return "domain_health_monitoring"
    if gap_type == "roadmap_candidate":
        return "autoresearch_paper_search"
    return "candidate_refinement"


def _candidate_for_loop(candidates: list[dict[str, Any]], loop_number: int) -> dict[str, Any]:
    if not candidates:
        return {
            "topic_id": f"SYNTHETIC-L{loop_number:02d}",
            "source": "synthetic",
            "domain": "unknown",
            "gap_type": "empty_queue",
            "topic": "No candidate available",
            "keywords": "paper2skills autoresearch backlog",
            "score": 0,
            "decision": "pending",
            "reason": "Candidate queue was empty",
        }
    return candidates[(loop_number - 1) % len(candidates)]


def _evolution_population(candidates: list[dict[str, Any]], loops: int) -> list[dict[str, Any]]:
    pending = [item for item in candidates if item.get("decision") == "pending"]
    base = pending or candidates
    if not base:
        return [_candidate_for_loop([], loop_number) for loop_number in range(1, loops + 1)]

    population = [dict(item) for item in base[:loops]]
    variant_index = 1
    while len(population) < loops:
        source = base[(len(population) - len(base)) % len(base)]
        variant = dict(source)
        variant_index += 1
        variant["topic_id"] = f"{source.get('topic_id', 'CANDIDATE')}-V{variant_index:02d}"
        variant["keywords"] = f"{source.get('keywords') or source.get('topic')} production benchmark implementation"
        variant["reason"] = f"Recombined mutation from {source.get('topic_id')} to keep all {loops} loops actionable."
        variant["recombined_from"] = source.get("topic_id")
        population.append(variant)
    return population[:loops]


def _loop_record(
    loop_number: int,
    candidate: dict[str, Any],
    *,
    fitness: float,
    domain_counts: dict[str, int],
    seen_topics: set[str],
) -> dict[str, Any]:
    gate = _gate(fitness)
    topic_id = str(candidate.get("topic_id", f"L{loop_number:02d}"))
    domain = str(candidate.get("domain", "unknown"))
    duplicate = topic_id in seen_topics or candidate.get("decision") == "already_exists"
    seen_topics.add(topic_id)

    return {
        "loop": loop_number,
        "candidate_id": topic_id,
        "candidate": candidate.get("topic") or candidate.get("skill_id") or topic_id,
        "mutation": {
            "source": candidate.get("source"),
            "autoresearch_query": candidate.get("keywords") or candidate.get("topic"),
            "paper_url": candidate.get("paper_url"),
            "arxiv_id": candidate.get("arxiv_id"),
        },
        "selection": {
            "fitness": fitness,
            "gate": gate,
            "action": _action_for(candidate, gate),
            "thresholds": {"inherit": 0.7, "observe": 0.4},
        },
        "niche": {
            "domain": domain,
            "domain_skill_count": domain_counts.get(domain, 0),
            "pressure": "thin_domain_boost" if 0 < domain_counts.get(domain, 0) <= 5 else "normal",
        },
        "competition": {
            "duplicate_or_existing": bool(duplicate),
            "decision": candidate.get("decision", "pending"),
            "policy": "retain_audit_lower_priority" if duplicate else "keep_in_population",
        },
        "heredity": {
            "template": "MasterPrompt 5-section Skill card + verification report",
            "inherits_structure": gate == "inherit",
            "lineage_source": "AutoSkill + EvoSC + EvoSkills local Skill cards",
        },
    }


def _stability_checkpoint(loop_number: int, graph_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "loop": loop_number,
        "nodes": graph_state["nodes"],
        "edges": graph_state["edges"],
        "density": graph_state["density"],
        "p0": graph_state["p0"],
        "p1": graph_state["p1"],
        "p2": graph_state["p2"],
        "mutation_paused": graph_state["p0"] > 0,
        "reason": "pause mutation if P0 appears; otherwise continue autoresearch evolution",
    }


def _escape_valves(loop_records: list[dict[str, Any]], graph_state: dict[str, Any]) -> dict[str, Any]:
    consecutive_low = 0
    max_consecutive_low = 0
    for record in loop_records:
        if record["selection"]["fitness"] < 0.6:
            consecutive_low += 1
        else:
            max_consecutive_low = max(max_consecutive_low, consecutive_low)
            consecutive_low = 0
    max_consecutive_low = max(max_consecutive_low, consecutive_low)
    triggered = max_consecutive_low >= 3 or graph_state["p0"] > 0
    return {
        "triggered": triggered,
        "consecutive_low_fitness": max_consecutive_low,
        "p0_gaps": graph_state["p0"],
        "policy": "switch_to_manual_or_relation_backfill" if triggered else "continue_autoresearch",
    }


def run_darwin_evolution(
    root: str | Path | None = None,
    *,
    loops: int = 20,
    dry_run: bool = True,
    write_run: bool = True,
    run_output_dir: str | Path | None = None,
    report_output: str | Path | None = None,
    graph_limit: int = 30,
    queue_limit: int = 100,
) -> dict[str, Any]:
    project_root = project_root_from(Path(root) if root is not None else None)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    graph_state = _graph_state(project_root, graph_limit)
    queue = build_candidate_queue(project_root, graph_gaps=graph_state["gaps"], limit=queue_limit)
    inventory = build_asset_inventory(project_root)
    counts = _domain_counts(inventory)

    candidates = queue["candidates"]
    population = _evolution_population(candidates, loops)
    seen_topics: set[str] = set()
    loop_records = []
    for loop_number in range(1, loops + 1):
        candidate = _candidate_for_loop(population, loop_number)
        fit = _fitness(candidate, counts)
        loop_records.append(
            _loop_record(
                loop_number,
                candidate,
                fitness=fit,
                domain_counts=counts,
                seen_topics=seen_topics,
            )
        )

    checkpoints = [
        _stability_checkpoint(loop_number, graph_state)
        for loop_number in range(5, loops + 1, 5)
    ]
    result = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "dry_run": dry_run,
        "loops_requested": loops,
        "summary": {
            "mechanisms": DARWIN_MECHANISMS,
            "autoresearch_sources": sorted({item.get("source", "unknown") for item in candidates}),
            "candidate_pool": queue["summary"],
            "inherited": sum(1 for record in loop_records if record["selection"]["gate"] == "inherit"),
            "observed": sum(1 for record in loop_records if record["selection"]["gate"] == "observe"),
            "rejected": sum(1 for record in loop_records if record["selection"]["gate"] == "reject"),
            "graph_state": {key: graph_state[key] for key in ("nodes", "edges", "density", "p0", "p1", "p2")},
        },
        "loops": loop_records,
        "stability_checkpoints": checkpoints,
        "escape_valves": _escape_valves(loop_records, graph_state),
    }

    if write_run:
        output_dir = Path(run_output_dir) if run_output_dir is not None else project_root / DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{run_id}.json"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["run_path"] = str(json_path.relative_to(project_root) if json_path.is_relative_to(project_root) else json_path)

        report_path = Path(report_output) if report_output is not None else project_root / DEFAULT_REPORT
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_darwin_markdown_report(result), encoding="utf-8")
        result["report_path"] = str(report_path.relative_to(project_root) if report_path.is_relative_to(project_root) else report_path)

    return result


def build_darwin_markdown_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# paper2skills Darwin Evolution 20-Loop Report",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Dry run: `{result['dry_run']}`",
        f"- Mechanisms: {', '.join(summary['mechanisms'])}",
        f"- Autoresearch mutation sources: {', '.join(summary['autoresearch_sources'])}",
        f"- Graph state: {summary['graph_state']['nodes']} nodes / {summary['graph_state']['edges']} edges / density {summary['graph_state']['density']}",
        f"- Selection gates: inherit={summary['inherited']}, observe={summary['observed']}, reject={summary['rejected']}",
        "",
        "Autoresearch mutation uses graph gaps and roadmap-derived paper topics; Darwin selection keeps all decisions auditable.",
        "",
        "## 20 Loop Table",
        "",
        "| Loop | Candidate ID | Candidate | Domain | Fitness | Gate | Action |",
        "|---|---|---|---|---:|---|---|",
    ]
    for record in result["loops"]:
        candidate = str(record["candidate"]).replace("|", "/")[:80]
        lines.append(
            f"| L{record['loop']:02d} | {record['candidate_id']} | {candidate} | {record['niche']['domain']} | "
            f"{record['selection']['fitness']:.3f} | {record['selection']['gate']} | {record['selection']['action']} |"
        )

    lines.extend(
        [
            "",
            "## Stability Checkpoints",
            "",
            "| Loop | Nodes | Edges | Density | P0 | P1 | P2 | Mutation Paused |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for checkpoint in result["stability_checkpoints"]:
        lines.append(
            f"| L{checkpoint['loop']:02d} | {checkpoint['nodes']} | {checkpoint['edges']} | "
            f"{checkpoint['density']} | {checkpoint['p0']} | {checkpoint['p1']} | {checkpoint['p2']} | "
            f"{checkpoint['mutation_paused']} |"
        )

    lines.extend(
        [
            "",
            "## Escape Valves",
            "",
            f"- Triggered: `{result['escape_valves']['triggered']}`",
            f"- Consecutive low fitness loops: `{result['escape_valves']['consecutive_low_fitness']}`",
            f"- Policy: `{result['escape_valves']['policy']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Darwin-style autoresearch evolution loops")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Mark run as dry-run. Default behavior is dry-run.")
    parser.add_argument("--apply", action="store_true", help="Mark run as non-dry-run metadata; still writes only audit artifacts.")
    parser.add_argument("--no-write-run", action="store_true")
    parser.add_argument("--run-output-dir", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--graph-limit", type=int, default=30)
    parser.add_argument("--queue-limit", type=int, default=100)
    args = parser.parse_args()

    result = run_darwin_evolution(
        args.root,
        loops=args.loops,
        dry_run=not args.apply,
        write_run=not args.no_write_run,
        run_output_dir=args.run_output_dir,
        report_output=args.report_output,
        graph_limit=args.graph_limit,
        queue_limit=args.queue_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

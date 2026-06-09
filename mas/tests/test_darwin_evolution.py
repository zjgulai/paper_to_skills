"""Regression tests for Darwin-style autoresearch evolution loops."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_darwin_evolution_runs_twenty_auditable_loops_without_writing(tmp_path):
    from paper2skills_common.evolution import run_darwin_evolution

    result = run_darwin_evolution(
        ROOT,
        loops=20,
        dry_run=True,
        write_run=False,
        run_output_dir=tmp_path,
        graph_limit=12,
        queue_limit=60,
    )

    assert result["status"] == "completed"
    assert result["loops_requested"] == 20
    assert len(result["loops"]) == 20
    assert {loop["loop"] for loop in result["loops"]} == set(range(1, 21))
    assert result["summary"]["mechanisms"] == [
        "mutation",
        "heredity",
        "selection",
        "competition",
        "niche",
    ]
    assert result["summary"]["autoresearch_sources"] == ["roadmap", "skills_graph_gaps"]
    assert result["stability_checkpoints"][0]["loop"] == 5
    assert result["stability_checkpoints"][-1]["loop"] == 20
    assert list(tmp_path.glob("*")) == []


def test_darwin_loop_records_selection_fitness_and_escape_valves():
    from paper2skills_common.evolution import run_darwin_evolution

    result = run_darwin_evolution(ROOT, loops=20, dry_run=True, write_run=False, graph_limit=12, queue_limit=60)

    first_loop = result["loops"][0]
    assert {"mutation", "selection", "niche", "competition", "heredity"} <= set(first_loop)
    assert first_loop["mutation"]["autoresearch_query"]
    assert 0.0 <= first_loop["selection"]["fitness"] <= 1.0
    assert first_loop["selection"]["gate"] in {"inherit", "observe", "reject"}
    assert first_loop["niche"]["domain"]
    assert isinstance(first_loop["competition"]["duplicate_or_existing"], bool)
    assert result["escape_valves"]["triggered"] in {True, False}
    assert "consecutive_low_fitness" in result["escape_valves"]


def test_darwin_markdown_report_contains_twenty_loop_table():
    from paper2skills_common.evolution import build_darwin_markdown_report, run_darwin_evolution

    result = run_darwin_evolution(ROOT, loops=20, dry_run=True, write_run=False, graph_limit=12, queue_limit=60)
    report = build_darwin_markdown_report(result)

    assert "# paper2skills Darwin Evolution 20-Loop Report" in report
    assert "| Loop | Candidate ID | Candidate | Domain | Fitness | Gate | Action |" in report
    assert report.count("| L") >= 20
    assert "Autoresearch mutation" in report

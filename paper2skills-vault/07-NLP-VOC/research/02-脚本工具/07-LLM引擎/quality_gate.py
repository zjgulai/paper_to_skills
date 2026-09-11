"""Quality Gate — Phase 5 D5 T5.3

Aggregates red-line judgments from existing artifact files and prints a
PASS/FAIL table. No new ML; pure aggregation/threshold logic.

Two gates supported:
  --gate week1   9 red lines (D7 boundary)
  --gate week2   7 red lines (D13 boundary)

Inputs (resolved per gate):
  week1:
    --eval-json     phase5 evaluation_suite three-way JSON
    --pred          5K LLM-labeled jsonl (for json-fail-rate, mutex check)
    --absa-summary  absa_500_pred.jsonl.summary.json
    --nps-pred      golden_500 NPS labeler output (for NPS agreement vs golden)
    --golden        golden_set_500_consensus.jsonl

  week2:
    --pred          full labeled jsonl (364K)
    --coverage-json dual_coverage_calculator output JSON
    --persona-pred  full persona-tagged jsonl
    --self-test-log phase5_unified_labeler self-test exit code (0=pass)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fmt_pass(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"


def gate_week1(args) -> tuple[list[dict], bool]:
    checks: list[dict] = []

    eval_json = load_json(args.eval_json) if args.eval_json and args.eval_json.exists() else {}
    llm = (eval_json.get("phase5_llm") or eval_json.get("llm") or eval_json.get("consensus") or {})

    pred_records = load_jsonl(args.pred) if args.pred and args.pred.exists() else []
    n_pred = len(pred_records)

    absa_summary = load_json(args.absa_summary) if args.absa_summary and args.absa_summary.exists() else {}

    nps_records = load_jsonl(args.nps_pred) if args.nps_pred and args.nps_pred.exists() else []
    golden_records = load_jsonl(args.golden) if args.golden and args.golden.exists() else []
    golden_idx = {r["review_id"]: r for r in golden_records if r.get("review_id")}

    top1_acc = 0.0
    top1_n = 0
    if pred_records and golden_idx:
        for rec in pred_records:
            g = golden_idx.get(rec.get("review_id"))
            if not g:
                continue
            golden_tags = {l.get("tag_id") for l in (g.get("golden_labels") or []) if l.get("tag_id")}
            if not golden_tags:
                continue
            pred_tags = [l.get("tag_id") for l in (rec.get("labels") or []) if l.get("tag_id")]
            if not pred_tags:
                top1_n += 1
                continue
            top1_n += 1
            if pred_tags[0] in golden_tags:
                top1_acc += 1
        top1_acc = top1_acc / max(top1_n, 1)
    checks.append({
        "id": 1, "name": "LLM Top-1 accuracy vs golden", "threshold": ">= 0.85",
        "value": round(top1_acc, 4), "pass": top1_acc >= 0.85 and top1_n > 0,
        "note": f"n_eval={top1_n}",
    })

    weighted_f1 = llm.get("tag_weighted_f1", 0.0)
    checks.append({
        "id": 2, "name": "Per-label F1 weighted (TOP-30)", "threshold": ">= 0.75",
        "value": round(weighted_f1, 4), "pass": weighted_f1 >= 0.75,
    })

    mean_jaccard = llm.get("mean_jaccard", 0.0)
    checks.append({
        "id": 3, "name": "Top-3 mean Jaccard (recall proxy)", "threshold": ">= 0.50",
        "value": round(mean_jaccard, 4), "pass": mean_jaccard >= 0.50,
    })

    sent_kappa = llm.get("sentiment_kappa", 0.0)
    checks.append({
        "id": 4, "name": "LLM sentiment Cohen κ", "threshold": ">= 0.65",
        "value": round(sent_kappa, 4), "pass": sent_kappa >= 0.65,
    })

    n_aspects = absa_summary.get("total_aspects", 0)
    n_total = absa_summary.get("n_total", 1)
    avg_aspects = absa_summary.get("avg_aspects_per_record", 0.0)
    checks.append({
        "id": 5, "name": "ABSA aspect/record in [1, 5]", "threshold": "1.0 <= x <= 5.0",
        "value": round(avg_aspects, 2), "pass": 1.0 <= avg_aspects <= 5.0,
        "note": f"total_aspects={n_aspects}, n_total={n_total}",
    })

    empty_pct = absa_summary.get("empty_pct", 1.0)
    checks.append({
        "id": 6, "name": "ABSA empty rate", "threshold": "< 0.10",
        "value": round(empty_pct, 4), "pass": empty_pct < 0.10,
    })

    nps_match = 0
    nps_n = 0
    if nps_records and golden_idx:
        for r in nps_records:
            g = golden_idx.get(r.get("review_id"))
            if not g:
                continue
            gv = g.get("golden_proxy_nps")
            pv = r.get("proxy_nps_final")
            if not gv:
                continue
            nps_n += 1
            if gv == pv:
                nps_match += 1
    nps_acc = nps_match / max(nps_n, 1)
    checks.append({
        "id": 7, "name": "Proxy NPS three-way agreement vs golden", "threshold": ">= 0.85",
        "value": round(nps_acc, 4), "pass": nps_acc >= 0.85 and nps_n > 0,
        "note": f"n_eval={nps_n}",
    })

    mutex_violations = 0
    for r in pred_records:
        tag_ids = {l.get("tag_id") for l in (r.get("labels") or []) if l.get("tag_id")}
        for t in tag_ids:
            if t and (t.endswith("N001") or "_N0" in t):
                base_pos = t.replace("_N0", "_P0").replace("N001", "P001")
                if base_pos in tag_ids:
                    mutex_violations += 1
                    break
    mutex_rate = mutex_violations / max(n_pred, 1)
    checks.append({
        "id": 8, "name": "Tag mutex (POS+NEG co-occurrence) rate", "threshold": "< 0.03",
        "value": round(mutex_rate, 4), "pass": mutex_rate < 0.03,
        "note": f"violations={mutex_violations}, n={n_pred}",
    })

    json_fail = sum(1 for r in pred_records if not r.get("success", True))
    json_fail_rate = json_fail / max(n_pred, 1)
    checks.append({
        "id": 9, "name": "JSON parse failure rate", "threshold": "< 0.01",
        "value": round(json_fail_rate, 4), "pass": json_fail_rate < 0.01,
        "note": f"failed={json_fail}, n={n_pred}",
    })

    overall = all(c["pass"] for c in checks)
    return checks, overall


def _stream_pred_stats(path: Path) -> tuple[int, int, float, int]:
    """Stream-aggregate pred jsonl → (n_pred, n_labels, sum_conf, n_with_nps)

    避免把 364K × 500M 记录一次性装进内存（原 load_jsonl 会 OOM）。
    """
    n_pred = 0
    n_labels = 0
    sum_conf = 0.0
    n_with_nps = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_pred += 1
            for l in r.get("labels") or []:
                c = l.get("confidence")
                if isinstance(c, (int, float)):
                    sum_conf += float(c)
                    n_labels += 1
            if r.get("proxy_nps_final") or r.get("proxy_nps"):
                n_with_nps += 1
    return n_pred, n_labels, sum_conf, n_with_nps


def _stream_persona_stats(path: Path) -> tuple[int, int]:
    n_total = 0
    n_hit = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            if r.get("persona_tags"):
                n_hit += 1
    return n_total, n_hit


def gate_week2(args) -> tuple[list[dict], bool]:
    checks: list[dict] = []

    coverage = load_json(args.coverage_json) if args.coverage_json and args.coverage_json.exists() else {}

    raw_cov = coverage.get("raw_coverage", 0.0)
    eff_cov = coverage.get("effective_coverage", 0.0)
    checks.append({"id": 10, "name": "Raw coverage", "threshold": ">= 0.88",
                   "value": round(raw_cov, 4), "pass": raw_cov >= 0.88})
    checks.append({"id": 11, "name": "Effective coverage", "threshold": ">= 0.94",
                   "value": round(eff_cov, 4), "pass": eff_cov >= 0.94})

    if args.pred and args.pred.exists():
        n_pred, n_labels, sum_conf, n_with_nps = _stream_pred_stats(args.pred)
    else:
        n_pred = n_labels = n_with_nps = 0
        sum_conf = 0.0
    avg_conf = (sum_conf / n_labels) if n_labels else 0.0
    checks.append({"id": 12, "name": "Average confidence", "threshold": ">= 0.75",
                   "value": round(avg_conf, 4), "pass": avg_conf >= 0.75,
                   "note": f"n_labels={n_labels}"})

    if args.persona_pred and args.persona_pred.exists():
        n_persona_total, n_persona_hit = _stream_persona_stats(args.persona_pred)
    else:
        n_persona_total = n_persona_hit = 0
    persona_rate = n_persona_hit / max(n_persona_total, 1)
    checks.append({"id": 13, "name": "Persona penetration", "threshold": ">= 0.60",
                   "value": round(persona_rate, 4), "pass": persona_rate >= 0.60,
                   "note": f"hit={n_persona_hit}, n={n_persona_total}"})

    nps_cov = n_with_nps / max(n_pred, 1)
    checks.append({"id": 14, "name": "Proxy NPS coverage", "threshold": ">= 0.95",
                   "value": round(nps_cov, 4), "pass": nps_cov >= 0.95,
                   "note": f"with_nps={n_with_nps}, n={n_pred}"})

    self_test_pass = bool(args.self_test_log and args.self_test_log.exists()
                          and args.self_test_log.read_text(encoding="utf-8").strip().splitlines()[-1:].count("0") > 0)
    checks.append({"id": 15, "name": "Self-test pass rate", "threshold": "= 1.00",
                   "value": "1.00" if self_test_pass else "<1.00",
                   "pass": self_test_pass,
                   "note": "Provide --self-test-log with `0` on its last line if passed"})

    bi_spec_ok = bool(args.bi_spec and args.bi_spec.exists())
    checks.append({"id": 16, "name": "BI dashboard spec exists (7-dept)", "threshold": "exists",
                   "value": "exists" if bi_spec_ok else "missing",
                   "pass": bi_spec_ok,
                   "note": str(args.bi_spec) if args.bi_spec else "not provided"})

    overall = all(c["pass"] for c in checks)
    return checks, overall


def render_md(gate: str, checks: list[dict], overall: bool) -> str:
    lines = [f"# Phase 5 Quality Gate — {gate.upper()}", "",
             f"**Overall**: {fmt_pass(overall)}", "",
             "| # | Check | Threshold | Value | Pass | Note |",
             "|---|---|---|---:|:---:|---|"]
    for c in checks:
        note = c.get("note", "")
        lines.append(f"| {c['id']} | {c['name']} | `{c['threshold']}` | {c['value']} | {fmt_pass(c['pass'])} | {note} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["week1", "week2"], required=True)
    ap.add_argument("--report", type=Path, help="Markdown report output")
    ap.add_argument("--json-out", type=Path, help="JSON output")

    ap.add_argument("--eval-json", type=Path)
    ap.add_argument("--pred", type=Path, help="LLM-labeled jsonl")
    ap.add_argument("--absa-summary", type=Path)
    ap.add_argument("--nps-pred", type=Path)
    ap.add_argument("--golden", type=Path)

    ap.add_argument("--coverage-json", type=Path)
    ap.add_argument("--persona-pred", type=Path)
    ap.add_argument("--self-test-log", type=Path)
    ap.add_argument("--bi-spec", type=Path)

    args = ap.parse_args()
    if args.gate == "week1":
        checks, overall = gate_week1(args)
    else:
        checks, overall = gate_week2(args)

    print(render_md(args.gate, checks, overall))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_md(args.gate, checks, overall), encoding="utf-8")
        print(f"\n📄 Report: {args.report}", file=sys.stderr)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"gate": args.gate, "overall_pass": overall, "checks": checks},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📦 JSON: {args.json_out}", file=sys.stderr)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

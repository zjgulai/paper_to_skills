"""Persona diagnostic — Phase 5 D6 T6.5

Reads a persona-labeled jsonl and produces:
  - Per-dimension / per-tag penetration rates
  - Penetration heatmap: data_source × dimension
  - Penetration heatmap: product_line × dimension (when available)
  - Dead-tag list (0 hits in this batch)
  - Markdown report + JSON stats

Usage:
  python persona_diagnostic.py \
    --input test_set_5k_p5_persona.jsonl \
    --rules-json ../01-设计文档/02-工作流设计/persona_tags_55.json \
    --report ../04-输出结果/03-审计报告/phase5_d6_persona_diagnostic.md \
    --json-out ../04-输出结果/03-审计报告/phase5_d6_persona_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_rules(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--source-text", type=Path, default=None,
                    help="Optional to join data_source/product_line when missing in input")
    ap.add_argument("--rules-json", type=Path,
                    default=Path(__file__).resolve().parents[1] /
                    "01-标签进化" / "persona_tags_55.json")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    recs = load_jsonl(args.input)
    if not args.rules_json.exists():
        alt = Path(__file__).resolve().parents[2] / "01-设计文档" / "02-工作流设计" / "persona_tags_55.json"
        if alt.exists():
            args.rules_json = alt
    rules = load_rules(args.rules_json)
    rule_lookup = {r["tag_id"]: r for r in rules}

    src_idx = {}
    if args.source_text and args.source_text.exists():
        src_idx = {r["review_id"]: r for r in load_jsonl(args.source_text) if r.get("review_id")}

    n = len(recs)
    print(f"Loaded {n} records, {len(rules)} rule tags")

    tag_hit: Counter = Counter()
    dim_hit: Counter = Counter()
    per_dim_tag_coverage: dict[str, set[str]] = defaultdict(set)
    source_dim_counter: dict[tuple[str, str], int] = defaultdict(int)
    source_record_counter: Counter = Counter()
    product_line_dim_counter: dict[tuple[str, str], int] = defaultdict(int)
    product_line_record_counter: Counter = Counter()
    per_record_tag_count: list[int] = []
    n_with_any = 0

    for rec in recs:
        rid = rec.get("review_id")
        src = src_idx.get(rid, {})
        data_source = rec.get("data_source") or src.get("data_source") or "unknown"
        product_line = rec.get("product_line") or src.get("product_line") or "unknown"
        source_record_counter[data_source] += 1
        product_line_record_counter[product_line] += 1

        ptags = rec.get("persona_tags") or []
        per_record_tag_count.append(len(ptags))
        if ptags:
            n_with_any += 1

        dims_in_rec: set[str] = set()
        for t in ptags:
            tid = t.get("tag_id")
            dim = t.get("dimension")
            en = t.get("tag_en")
            if tid:
                tag_hit[tid] += 1
                per_dim_tag_coverage[dim].add(en)
                dims_in_rec.add(dim)
        for dim in dims_in_rec:
            source_dim_counter[(data_source, dim)] += 1
            product_line_dim_counter[(product_line, dim)] += 1
            dim_hit[dim] += 1

    penetration = n_with_any / n if n else 0
    n_tags_with_hit = sum(1 for r in rules if tag_hit.get(r["tag_id"], 0) > 0)
    avg_tags_per_rec = sum(per_record_tag_count) / n if n else 0
    dead_tags = [r["tag_id"] for r in rules if tag_hit.get(r["tag_id"], 0) == 0]

    DIM_ORDER = ["WHO", "WHY", "WHAT", "WHEN", "HOW", "EMOTION", "LANGUAGE"]

    lines: list[str] = []
    lines.append("---")
    lines.append("name: phase5-d6-persona-diagnostic")
    lines.append("description: Phase 5 D6 画像标签诊断报告。涵盖 55 标签整体渗透率、各维度/数据源/品线渗透率热力表、死标签清单。当评估 D6 画像标签器效果、定位冷门标签、为字典进化 v4.1 选择候选时使用。")
    lines.append("title: Phase 5 D6 画像标签诊断报告")
    lines.append("doc_type: audit")
    lines.append("module: voc-nlp")
    lines.append("topic: persona-diagnostic")
    lines.append("status: final")
    lines.append("created: 2026-05-08")
    lines.append("updated: 2026-05-08")
    lines.append("owner: self")
    lines.append("source: ai")
    lines.append("---")
    lines.append("")
    lines.append("# Phase 5 D6 画像标签诊断报告")
    lines.append("")
    lines.append(f"**总记录数**：{n}")
    lines.append(f"**至少命中 1 个画像标签**：{n_with_any} / {n} = **{penetration*100:.2f}%**（红线 ≥ 60%）")
    lines.append(f"**命中至少 1 次的标签**：**{n_tags_with_hit} / {len(rules)}**（红线 ≥ 45）")
    lines.append(f"**平均每条命中**：{avg_tags_per_rec:.2f} 个标签")
    lines.append("")
    pass_pen = "✅" if penetration >= 0.60 else "❌"
    pass_cov = "✅" if n_tags_with_hit >= 45 else "❌"
    lines.append(f"**D6 场景 2 判定**：渗透率 {pass_pen}  ｜  标签覆盖 {pass_cov}")
    lines.append("")

    lines.append("## 一、各维度命中分布")
    lines.append("")
    lines.append("| 维度 | 命中记录数 | 记录占比 | 覆盖标签数 | 总标签数 |")
    lines.append("|---|---:|---:|---:|---:|")
    for dim in DIM_ORDER:
        hits = dim_hit.get(dim, 0)
        covered = len(per_dim_tag_coverage[dim])
        total = sum(1 for r in rules if r["dimension"] == dim)
        lines.append(f"| {dim} | {hits} | {hits/n*100:.2f}% | {covered} | {total} |")
    lines.append("")

    lines.append("## 二、数据源 × 维度 渗透率热力表")
    lines.append("")
    header = ["data_source (n)"] + DIM_ORDER + ["至少 1 个"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|---" * len(header) + "|")
    for src, tot in source_record_counter.most_common():
        row = [f"{src} ({tot})"]
        for dim in DIM_ORDER:
            c = source_dim_counter.get((src, dim), 0)
            row.append(f"{c/tot*100:.1f}%" if tot else "—")
        n_any = sum(1 for rec in recs
                    if (rec.get("data_source") or src_idx.get(rec.get("review_id"), {}).get("data_source")) == src
                    and rec.get("persona_has_any"))
        row.append(f"{n_any/tot*100:.1f}%" if tot else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if product_line_record_counter and list(product_line_record_counter.keys()) != ["unknown"]:
        lines.append("## 三、品线 × 维度 渗透率热力表")
        lines.append("")
        header = ["product_line (n)"] + DIM_ORDER
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---" * len(header) + "|")
        for pl, tot in product_line_record_counter.most_common():
            if pl == "unknown" and len(product_line_record_counter) > 1:
                continue
            row = [f"{pl} ({tot})"]
            for dim in DIM_ORDER:
                c = product_line_dim_counter.get((pl, dim), 0)
                row.append(f"{c/tot*100:.1f}%" if tot else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## 四、Top-20 标签命中")
    lines.append("")
    lines.append("| 排名 | tag_id | tag_en | dimension | 命中数 | 占比 |")
    lines.append("|---:|---|---|---|---:|---:|")
    for rank, (tid, c) in enumerate(tag_hit.most_common(20), 1):
        r = rule_lookup.get(tid, {})
        lines.append(f"| {rank} | {tid} | {r.get('tag_en','?')} | {r.get('dimension','?')} | {c} | {c/n*100:.2f}% |")
    lines.append("")

    lines.append("## 五、死标签（本批 0 命中）")
    lines.append("")
    if not dead_tags:
        lines.append("_无死标签，全部 55 标签至少命中 1 次。_")
    else:
        lines.append("| tag_id | tag_en | dimension | 首批关键词 | 建议 |")
        lines.append("|---|---|---|---|---|")
        for tid in dead_tags:
            r = rule_lookup.get(tid, {})
            kws = ", ".join(r.get("keywords", [])[:3])
            suggest = "采样稀缺，D7/D9 开集阶段可挖更多关键词或 LLM 兜底" if r.get("dimension") in ["WHO", "WHY"] else "可能语料不覆盖此场景"
            lines.append(f"| {tid} | {r.get('tag_en','?')} | {r.get('dimension','?')} | {kws} | {suggest} |")
    lines.append("")

    lines.append("## 六、观察结论")
    lines.append("")
    if penetration >= 0.60 and n_tags_with_hit >= 45:
        lines.append("✅ **D6 场景 2 全通过**：渗透率与标签覆盖双过红线，画像标签器可进 D7 集成。")
    else:
        lines.append("❌ **D6 场景 2 未全通过**：需要补关键词或 LLM 兜底。")
    lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"📄 Report: {args.report}")

    if args.json_out:
        stats = {
            "n_records": n,
            "n_with_any_tag": n_with_any,
            "penetration_rate": penetration,
            "tags_with_at_least_1_hit": n_tags_with_hit,
            "n_total_rules": len(rules),
            "avg_tags_per_record": avg_tags_per_rec,
            "dead_tags": dead_tags,
            "per_dimension_hits": dict(dim_hit),
            "per_dimension_coverage_count": {d: len(v) for d, v in per_dim_tag_coverage.items()},
            "top_tag_hits": dict(tag_hit.most_common(30)),
            "source_dim_counts": {f"{k[0]}|{k[1]}": v for k, v in source_dim_counter.items()},
            "qa_pass_penetration_60": penetration >= 0.60,
            "qa_pass_coverage_45": n_tags_with_hit >= 45,
        }
        args.json_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📦 JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

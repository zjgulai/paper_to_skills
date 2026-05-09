"""Phase 5 D3 evaluation suite.

Two modes:

1. --self-consistency A.jsonl B.jsonl
     Same operator's two passes on the same N records. Reports Cohen's κ on
     each tri-class axis (overall_sentiment, proxy_nps) plus tag-set Jaccard
     mean. Pass: κ ≥ 0.80 on each axis.

2. --golden golden.jsonl --pred-p4 p4.jsonl --pred-llm llm.jsonl --report out.md
     Multi-source three-way evaluation of label assignments against the
     human golden set.
       Per-tag binary metrics  (precision / recall / F1)
       Macro-averaged + weighted F1 (sklearn over the union tag space)
       Cohen's κ for sentiment and NPS tri-class
       Confusion matrices for sentiment and NPS (also written as JSON)

Inputs (jsonl):
  golden record fields used:  review_id, golden_labels[].tag_id,
                              golden_overall_sentiment, golden_proxy_nps
  P4 prediction fields:       review_id, labels[].tag_id, sentiment_polarity, proxy_nps
  LLM prediction fields:      review_id, labels[].tag_id, overall_sentiment, proxy_nps
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

SENTIMENT_LABELS = ["positive", "neutral", "negative"]
NPS_LABELS = ["promoter", "passive", "detractor"]


def normalize_sentiment(value) -> str | None:
    """Phase 4 emits numeric polarity (-1..1); LLM emits text. Normalize to text."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"positive", "neutral", "negative"}:
            return v
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x >= 0.34:
        return "positive"
    if x <= -0.34:
        return "negative"
    return "neutral"


def normalize_nps(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in {"promoter", "passive", "detractor"} else None


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def index_by_review(records: list[dict]) -> dict[str, dict]:
    return {r["review_id"]: r for r in records if r.get("review_id")}


def get_tag_set(rec: dict, key: str) -> set[str]:
    out: set[str] = set()
    for lbl in rec.get(key) or []:
        tid = lbl.get("tag_id") if isinstance(lbl, dict) else None
        if tid:
            out.add(tid)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def per_tag_metrics(golden_sets: list[set[str]], pred_sets: list[set[str]]) -> dict:
    universe = sorted({t for s in golden_sets + pred_sets for t in s})
    if not universe:
        return {"n_tags": 0, "macro_f1": 0.0, "weighted_f1": 0.0, "per_tag": []}
    y_true = [[1 if t in g else 0 for t in universe] for g in golden_sets]
    y_pred = [[1 if t in p else 0 for t in universe] for p in pred_sets]

    import numpy as np
    yt = np.array(y_true); yp = np.array(y_pred)
    p, r, f, sup = precision_recall_fscore_support(yt, yp, average=None, zero_division=0, labels=range(len(universe)))
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    weighted_f1 = f1_score(yt, yp, average="weighted", zero_division=0)

    per_tag = [
        {"tag_id": universe[i], "precision": float(p[i]), "recall": float(r[i]),
         "f1": float(f[i]), "support": int(sup[i])}
        for i in range(len(universe))
    ]
    per_tag.sort(key=lambda x: -x["support"])

    return {
        "n_tags": len(universe),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_tag": per_tag,
    }


def safe_kappa(y_true: list[str], y_pred: list[str], labels: list[str]) -> float | None:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t and p]
    if not pairs:
        return None
    yt = [t for t, _ in pairs]
    yp = [p for _, p in pairs]
    return float(cohen_kappa_score(yt, yp, labels=labels))


def confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t and p]
    if not pairs:
        return {"labels": labels, "matrix": [], "n": 0}
    yt = [t for t, _ in pairs]; yp = [p for _, p in pairs]
    cm = confusion_matrix(yt, yp, labels=labels)
    return {"labels": labels, "matrix": cm.tolist(), "n": len(pairs)}


def evaluate_pred(name: str, golden: list[dict], preds: dict[str, dict], pred_label_key: str,
                  pred_sent_key: str, pred_nps_key: str) -> dict:
    g_tags, p_tags, g_sent, p_sent, g_nps, p_nps, n_missing = [], [], [], [], [], [], 0
    for g in golden:
        rid = g["review_id"]
        if not g.get("golden_labels"):
            continue
        pred = preds.get(rid)
        if pred is None:
            n_missing += 1
            continue
        g_tags.append(get_tag_set(g, "golden_labels"))
        p_tags.append(get_tag_set(pred, pred_label_key))
        g_sent.append(normalize_sentiment(g.get("golden_overall_sentiment")))
        p_sent.append(normalize_sentiment(pred.get(pred_sent_key)))
        g_nps.append(normalize_nps(g.get("golden_proxy_nps")))
        p_nps.append(normalize_nps(pred.get(pred_nps_key)))

    metrics = per_tag_metrics(g_tags, p_tags)
    jaccards = [jaccard(g, p) for g, p in zip(g_tags, p_tags)]
    return {
        "name": name,
        "n_eval": len(g_tags),
        "n_missing": n_missing,
        "tag_macro_f1": metrics["macro_f1"],
        "tag_weighted_f1": metrics["weighted_f1"],
        "tag_universe_size": metrics["n_tags"],
        "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        "sentiment_kappa": safe_kappa(g_sent, p_sent, SENTIMENT_LABELS),
        "nps_kappa": safe_kappa(g_nps, p_nps, NPS_LABELS),
        "sentiment_confusion": confusion(g_sent, p_sent, SENTIMENT_LABELS),
        "nps_confusion": confusion(g_nps, p_nps, NPS_LABELS),
        "per_tag": metrics["per_tag"][:30],
    }


def render_confusion_md(cm: dict, title: str) -> str:
    if not cm["matrix"]:
        return f"_{title}: no overlap_"
    lines = [f"**{title}** (n={cm['n']})", "", "| true \\ pred | " + " | ".join(cm["labels"]) + " |"]
    lines.append("|" + "---|" * (len(cm["labels"]) + 1))
    for i, lab in enumerate(cm["labels"]):
        row = " | ".join(str(v) for v in cm["matrix"][i])
        lines.append(f"| **{lab}** | {row} |")
    return "\n".join(lines)


def render_report(result_p4: dict, result_llm: dict) -> str:
    lines = ["# Phase 5 D3 三方评估报告", ""]
    lines.append("## 1. 总览")
    lines.append("")
    lines.append("| 系统 | n_eval | tag macro-F1 | tag weighted-F1 | mean Jaccard | sentiment κ | NPS κ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in (result_p4, result_llm):
        sk = "—" if r["sentiment_kappa"] is None else f"{r['sentiment_kappa']:.3f}"
        nk = "—" if r["nps_kappa"] is None else f"{r['nps_kappa']:.3f}"
        lines.append(f"| {r['name']} | {r['n_eval']} | {r['tag_macro_f1']:.3f} | "
                     f"{r['tag_weighted_f1']:.3f} | {r['mean_jaccard']:.3f} | {sk} | {nk} |")
    lines.append("")

    for r in (result_p4, result_llm):
        lines.append(f"\n## 2. {r['name']} 详细指标\n")
        lines.append(f"- 评估样本: {r['n_eval']}（缺失 {r['n_missing']}）")
        lines.append(f"- tag 全集: {r['tag_universe_size']} 个")
        lines.append("")
        lines.append(render_confusion_md(r["sentiment_confusion"], "Sentiment 混淆矩阵"))
        lines.append("")
        lines.append(render_confusion_md(r["nps_confusion"], "Proxy NPS 混淆矩阵"))
        lines.append("")
        lines.append("**Top-30 高频 tag 表现（按 support 降序）**")
        lines.append("")
        lines.append("| tag_id | support | precision | recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for t in r["per_tag"]:
            lines.append(f"| {t['tag_id']} | {t['support']} | {t['precision']:.2f} | {t['recall']:.2f} | {t['f1']:.2f} |")

    return "\n".join(lines)


def cmd_three_way(args):
    golden_all = load_jsonl(args.golden)
    golden = [g for g in golden_all if g.get("golden_labels")]
    if args.golden_source_filter:
        wanted = set(args.golden_source_filter.split(","))
        before = len(golden)
        golden = [g for g in golden if g.get("golden_source", "human") in wanted]
        print(f"Filter golden_source ∈ {wanted}: {before} → {len(golden)}")
    if not golden:
        print(f"❌ No matching golden records (after filter)")
        sys.exit(2)
    print(f"Annotated golden records: {len(golden)}/{len(golden_all)}")

    src_breakdown = Counter(g.get("golden_source", "human") for g in golden)
    print(f"  by golden_source: {dict(src_breakdown)}")

    p4_idx = index_by_review(load_jsonl(args.pred_p4))
    llm_idx = index_by_review(load_jsonl(args.pred_llm))
    print(f"P4 preds:  {len(p4_idx)}")
    print(f"LLM preds: {len(llm_idx)}")

    result_p4 = evaluate_pred("Phase 4 (rule + ALCHEmist)", golden, p4_idx,
                              pred_label_key="labels",
                              pred_sent_key="sentiment_polarity",
                              pred_nps_key="proxy_nps")
    result_llm = evaluate_pred("Phase 5 D2 (DeepSeek-V4-Flash)", golden, llm_idx,
                               pred_label_key="labels",
                               pred_sent_key="overall_sentiment",
                               pred_nps_key="proxy_nps")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result_p4, result_llm), encoding="utf-8")
        print(f"\n📄 Report: {args.report}")

    if args.json_out:
        args.json_out.write_text(json.dumps({"phase4": result_p4, "phase5_llm": result_llm},
                                            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📦 JSON  : {args.json_out}")

    print("\n=== Summary ===")
    for r in (result_p4, result_llm):
        print(f"  {r['name']:35} F1_macro={r['tag_macro_f1']:.3f}  "
              f"F1_weighted={r['tag_weighted_f1']:.3f}  Jaccard={r['mean_jaccard']:.3f}")


def cmd_self_consistency(args):
    a = load_jsonl(args.first)
    b = load_jsonl(args.second)
    a_idx = {r["review_id"]: r for r in a if r.get("golden_labels")}
    b_idx = {r["review_id"]: r for r in b if r.get("golden_labels")}
    common = sorted(a_idx.keys() & b_idx.keys())
    if not common:
        print("❌ No overlapping annotated review_ids"); sys.exit(2)
    print(f"Overlap: {len(common)}")

    s_a, s_b, n_a, n_b, j_scores = [], [], [], [], []
    for rid in common:
        ra = a_idx[rid]; rb = b_idx[rid]
        s_a.append(normalize_sentiment(ra.get("golden_overall_sentiment")))
        s_b.append(normalize_sentiment(rb.get("golden_overall_sentiment")))
        n_a.append(normalize_nps(ra.get("golden_proxy_nps")))
        n_b.append(normalize_nps(rb.get("golden_proxy_nps")))
        j_scores.append(jaccard(get_tag_set(ra, "golden_labels"), get_tag_set(rb, "golden_labels")))

    sk = safe_kappa(s_a, s_b, SENTIMENT_LABELS)
    nk = safe_kappa(n_a, n_b, NPS_LABELS)
    j_mean = sum(j_scores) / len(j_scores) if j_scores else 0
    print(f"\n  sentiment κ : {sk}")
    print(f"  NPS κ       : {nk}")
    print(f"  tag Jaccard : {j_mean:.3f}")

    out = {
        "n": len(common),
        "sentiment_kappa": sk,
        "nps_kappa": nk,
        "tag_jaccard_mean": j_mean,
        "pass": (sk or 0) >= 0.80 and (nk or 0) >= 0.80,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n📦 JSON: {args.json_out}")

    print(f"\n{'🎉 SELF-CONSISTENCY PASS' if out['pass'] else '⚠ SELF-CONSISTENCY FAIL (target κ ≥ 0.80)'}")
    sys.exit(0 if out["pass"] else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("self-consistency")
    sc.add_argument("first", type=Path)
    sc.add_argument("second", type=Path)
    sc.add_argument("--json-out", type=Path)
    sc.set_defaults(fn=cmd_self_consistency)

    tw = sub.add_parser("three-way")
    tw.add_argument("--golden", type=Path, required=True)
    tw.add_argument("--pred-p4", type=Path, required=True)
    tw.add_argument("--pred-llm", type=Path, required=True)
    tw.add_argument("--report", type=Path)
    tw.add_argument("--json-out", type=Path)
    tw.add_argument("--golden-source-filter", type=str, default=None,
                    help="Comma-separated golden_source values to keep, e.g. 'human' or 'consensus_llm,human'")
    tw.set_defaults(fn=cmd_three_way)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

"""Consensus pre-fill — auto-fill golden_labels when DeepSeek and Kimi agree.

Two modes:
  --mode soft  (default, recommended)
    Auto-accept when ALL three hold:
      a. shared_tags = (deepseek_tags ∩ kimi_tags) is non-empty
      b. overall_sentiment is identical
      c. proxy_nps is identical
    Rationale: in 5K test, two LLMs agree fully (Jaccard ≥ 0.6) on only
    12% of records but share ≥ 1 tag in 70% — different Top-3 ordering is
    not a quality signal. Taking the intersection as golden labels keeps
    only what BOTH models endorsed, which is the highest-confidence subset.

  --mode strict
    Auto-accept only when Jaccard(tags) ≥ jaccard_min (default 0.6) AND
    sentiment AND NPS all match. Use this when you want maximum precision
    on the consensus golden subset and don't mind doing more human work.

Output schema (added on top of input):
  golden_labels                = ranked intersection (≤ MAX_GOLDEN_TAGS), keeping DeepSeek order
  golden_overall_sentiment     = consensus value
  golden_proxy_nps             = consensus value
  golden_source                = "consensus_llm" | "needs_human"
  consensus_meta               = {"jaccard": float, "shared_tags": [...], "mode": "soft"|"strict"}
  disagreement_reason          = combined reason string when needs_human
  kimi_pred / kimi_overall_sentiment / kimi_proxy_nps  always copied for CLI display
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

TAG_JACCARD_MIN = 0.6
MAX_GOLDEN_TAGS = 3


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tag_ids(labels: list[dict] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for lbl in labels or []:
        tid = (lbl or {}).get("tag_id")
        if tid and tid not in seen:
            out.append(tid); seen.add(tid)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def consensus_for_one(rec: dict, kimi: dict | None, mode: str = "soft", jaccard_min: float = TAG_JACCARD_MIN) -> dict:
    deepseek_tags = tag_ids(rec.get("llm_pred"))
    deepseek_sent = rec.get("llm_overall_sentiment")
    deepseek_nps = rec.get("llm_proxy_nps")

    kimi_tags = tag_ids((kimi or {}).get("labels"))
    kimi_sent = (kimi or {}).get("overall_sentiment")
    kimi_nps = (kimi or {}).get("proxy_nps")
    kimi_success = bool(kimi and kimi.get("success"))

    out = dict(rec)
    out["kimi_pred"] = kimi.get("labels") if kimi else None
    out["kimi_overall_sentiment"] = kimi_sent
    out["kimi_proxy_nps"] = kimi_nps

    set_a, set_b = set(deepseek_tags), set(kimi_tags)
    shared = [t for t in deepseek_tags if t in set_b]
    j = jaccard(set_a, set_b) if (set_a or set_b) else 1.0

    reasons: list[str] = []
    if not kimi_success:
        reasons.append("kimi_missing_or_failed")
    if not deepseek_tags and not kimi_tags:
        reasons.append("both_zero_label")

    if mode == "strict":
        if j < jaccard_min:
            reasons.append(f"tag_jaccard={j:.2f}<{jaccard_min}")
    else:
        if not shared and (deepseek_tags or kimi_tags):
            reasons.append("no_shared_tags")

    if deepseek_sent != kimi_sent:
        reasons.append(f"sentiment_diff:{deepseek_sent}!={kimi_sent}")
    if deepseek_nps != kimi_nps:
        reasons.append(f"nps_diff:{deepseek_nps}!={kimi_nps}")

    if not reasons and kimi_success:
        ds_pred_lookup = {l.get("tag_id"): l for l in (rec.get("llm_pred") or [])}
        out["golden_labels"] = [
            {"tag_id": t, "tag_en": (ds_pred_lookup.get(t) or {}).get("tag_en")}
            for t in shared[:MAX_GOLDEN_TAGS]
        ]
        out["golden_overall_sentiment"] = deepseek_sent
        out["golden_proxy_nps"] = deepseek_nps
        out["golden_source"] = "consensus_llm"
        out["consensus_meta"] = {"jaccard": round(j, 3), "shared_tags": shared, "mode": mode}
        out["disagreement_reason"] = ""
    else:
        out["golden_source"] = "needs_human"
        out["disagreement_reason"] = "; ".join(reasons) if reasons else "unknown"
        out["consensus_meta"] = {"jaccard": round(j, 3), "shared_tags": shared, "mode": mode}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True, help="golden_set_500.jsonl with llm_pred (DeepSeek)")
    ap.add_argument("--kimi", type=Path, required=True, help="golden_set_500_kimi_pred.jsonl from llm_labeler --vendor kimi")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["soft", "strict"], default="soft")
    ap.add_argument("--jaccard-min", type=float, default=TAG_JACCARD_MIN, help="only used when --mode strict")
    args = ap.parse_args()

    if not args.golden.exists():
        print(f"❌ Golden not found: {args.golden}"); sys.exit(2)
    if not args.kimi.exists():
        print(f"❌ Kimi pred not found: {args.kimi}"); sys.exit(2)

    golden = load_jsonl(args.golden)
    kimi_records = load_jsonl(args.kimi)
    kimi_idx = {r["review_id"]: r for r in kimi_records if r.get("review_id")}

    print(f"Golden: {len(golden)}  Kimi: {len(kimi_idx)}  mode={args.mode}")
    if args.mode == "strict":
        print(f"  jaccard_min: {args.jaccard_min}")

    out_records: list[dict] = []
    counter = Counter()
    reason_counter = Counter()
    for rec in golden:
        kimi = kimi_idx.get(rec["review_id"])
        merged = consensus_for_one(rec, kimi, mode=args.mode, jaccard_min=args.jaccard_min)
        counter[merged["golden_source"]] += 1
        if merged["golden_source"] == "needs_human":
            for r in (merged.get("disagreement_reason") or "").split("; "):
                key = r.split(":")[0].split("=")[0] if r else "other"
                reason_counter[key] += 1
        out_records.append(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Auto-filled (consensus_llm): {counter['consensus_llm']:>4}  ({counter['consensus_llm']/len(golden)*100:.1f}%)")
    print(f"Needs human triage         : {counter['needs_human']:>4}  ({counter['needs_human']/len(golden)*100:.1f}%)")
    print("=" * 60)
    print("\nDisagreement reasons (Top-10):")
    for reason, n in reason_counter.most_common(10):
        print(f"  {reason:30} ×{n}")
    print(f"\n✅ Written: {args.output}")


if __name__ == "__main__":
    main()

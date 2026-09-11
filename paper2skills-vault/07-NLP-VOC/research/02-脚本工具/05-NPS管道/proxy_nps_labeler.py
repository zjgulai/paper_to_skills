"""Proxy NPS labeler — Phase 5 D5 T5.2

Per-record three-way voting:
  Method 1 — Star rating mapping (when rating available):
    1-2 → detractor, 3 → passive, 4-5 → promoter
  Method 2 — Recommendation-intent keyword (en/de/fr; positive vs negation):
    strong recommend → promoter
    would-not / cannot recommend → detractor
    no signal → None (abstain)
  Method 3 — LLM proxy_nps field (read from already-labeled record;
    can be optionally re-queried on miss).

Voting:
  1. Drop None votes.
  2. If ≥ 2 votes agree → that class wins.
  3. If only 1 voter → that class wins (low confidence).
  4. If 3 voters all disagree (e.g., promoter / passive / detractor) →
     prefer LLM > rating > keyword (LLM is most context-aware).

Output: same JSONL, with extra fields
  proxy_nps_final         : 'promoter' | 'passive' | 'detractor'
  proxy_nps_method_votes  : {rating, keyword, llm}
  proxy_nps_confidence    : float in [0,1]   (1.0 = unanimous, 0.5 = 1-only,
                                              0.7 = 2/3 agreement, 0.4 = 3-way split)

Usage:
  python proxy_nps_labeler.py \
      --input  test_set_5k_p5_llm.jsonl \
      --source-text test_set_5k_stratified.jsonl \
      --output test_set_5k_p5_nps.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


def vote_by_rating(rating: Optional[float]) -> Optional[str]:
    if rating is None:
        return None
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return None
    if r >= 4:
        return "promoter"
    if r <= 2:
        return "detractor"
    if 2 < r < 4:
        return "passive"
    return None


PROMOTER_KEYWORDS = [
    # English
    "highly recommend", "strongly recommend", "definitely recommend",
    "would recommend", "recommend to", "recommend this", "recommend it",
    "10/10", "5 stars", "5/5", "five stars",
    "buy again", "repurchase", "will buy again", "purchased again",
    "told my friend", "told my sister", "told everyone",
    "best purchase", "best buy", "must buy", "must have",
    # German
    "gerne wieder", "immer wieder", "weiterempfehlen", "empfehle gerne",
    "kann ich empfehlen", "absolut empfehlenswert", "klare empfehlung",
    # French
    "je recommande", "vivement recommandé", "vivement recommande",
    "je le recommande", "à recommander", "a recommander",
]

DETRACTOR_KEYWORDS = [
    # English
    "would not recommend", "wouldn't recommend", "do not recommend",
    "don't recommend", "dont recommend", "cannot recommend", "can't recommend",
    "stay away", "waste of money", "complete waste", "huge disappointment",
    "do not buy", "don't buy", "dont buy", "avoid this", "avoid at all",
    "never again", "never buy", "0 stars", "1 star",
    # German
    "kann ich nicht empfehlen", "nicht empfehlen", "absolut nicht empfehlenswert",
    "finger weg", "geld verschwendet",
    # French
    "ne recommande pas", "à éviter", "a eviter", "perte d'argent",
    "déçue", "tres deçue", "très déçue",
]

PROMOTER_RX = re.compile(r"|".join(re.escape(k) for k in PROMOTER_KEYWORDS), re.IGNORECASE)
DETRACTOR_RX = re.compile(r"|".join(re.escape(k) for k in DETRACTOR_KEYWORDS), re.IGNORECASE)


def vote_by_keyword(text: str) -> Optional[str]:
    if not text:
        return None
    # Detractor takes priority — negation phrases like "would not recommend"
    # contain "recommend" and would falsely match the promoter rx otherwise.
    if DETRACTOR_RX.search(text):
        return "detractor"
    if PROMOTER_RX.search(text):
        return "promoter"
    return None


VALID_NPS = {"promoter", "passive", "detractor"}


def vote_by_llm(record: dict) -> Optional[str]:
    v = record.get("proxy_nps") or record.get("llm_proxy_nps")
    if isinstance(v, str) and v.lower() in VALID_NPS:
        return v.lower()
    return None


def vote(votes: dict[str, Optional[str]]) -> tuple[str, float]:
    """Return (final_class, confidence). 0.0 confidence means no voter active."""
    valid = {k: v for k, v in votes.items() if v in VALID_NPS}
    if not valid:
        return "passive", 0.0
    counter = Counter(valid.values())
    most_common = counter.most_common()
    top_class, top_n = most_common[0]
    n_voters = len(valid)
    if top_n == n_voters:
        return top_class, 1.0 if n_voters >= 2 else 0.5
    if top_n >= 2:
        return top_class, 0.7
    for k in ("llm", "rating", "keyword"):
        if k in valid:
            return valid[k], 0.4
    return top_class, 0.4


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def index_by_review(records: list[dict]) -> dict[str, dict]:
    return {r["review_id"]: r for r in records if r.get("review_id")}


def label_one(rec: dict, src: Optional[dict]) -> dict:
    text = (rec.get("text") or (src or {}).get("text") or "").strip()
    rating = rec.get("rating") if rec.get("rating") is not None else (src or {}).get("rating")

    votes = {
        "rating": vote_by_rating(rating),
        "keyword": vote_by_keyword(text),
        "llm": vote_by_llm(rec),
    }
    final_class, conf = vote(votes)
    out = dict(rec)
    out["proxy_nps_final"] = final_class
    out["proxy_nps_method_votes"] = votes
    out["proxy_nps_confidence"] = conf
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="JSONL with review_id and (optionally) proxy_nps + text + rating")
    ap.add_argument("--source-text", type=Path, default=None,
                    help="Optional stratified jsonl carrying text/rating; joined by review_id when --input lacks them")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}", file=sys.stderr)
        return 2

    records = load_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    print(f"Input: {len(records)} records")

    src_idx: dict[str, dict] = {}
    if args.source_text and args.source_text.exists():
        src_idx = index_by_review(load_jsonl(args.source_text))
        print(f"Joined source-text: {len(src_idx)} records available")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    class_counter: Counter = Counter()
    method_used: Counter = Counter()
    conf_buckets: Counter = Counter()
    n_active_voters = []
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for rec in records:
            src = src_idx.get(rec.get("review_id"))
            out = label_one(rec, src)
            class_counter[out["proxy_nps_final"]] += 1
            n_active = sum(1 for v in out["proxy_nps_method_votes"].values() if v in VALID_NPS)
            n_active_voters.append(n_active)
            for m, v in out["proxy_nps_method_votes"].items():
                if v in VALID_NPS:
                    method_used[m] += 1
            conf_buckets[round(out["proxy_nps_confidence"], 1)] += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    n = max(written, 1)
    print("\n" + "=" * 60)
    print(f"Output: {args.output}  ({written} records)")
    print("=" * 60)
    print("\nClass distribution:")
    for cls in ("promoter", "passive", "detractor"):
        c = class_counter.get(cls, 0)
        print(f"  {cls:10}  ×{c:6}  ({c/n*100:.1f}%)")
    print("\nMethod usage (records with that voter active):")
    for m in ("rating", "keyword", "llm"):
        c = method_used.get(m, 0)
        print(f"  {m:10}  ×{c:6}  ({c/n*100:.1f}%)")
    avg_voters = sum(n_active_voters) / n
    print(f"\nAvg active voters/record: {avg_voters:.2f}")
    print("Confidence distribution:")
    for k in sorted(conf_buckets.keys()):
        print(f"  {k}  ×{conf_buckets[k]}")
    n_high_conf = sum(v for k, v in conf_buckets.items() if k >= 0.7)
    print(f"\n≥0.7 confidence: {n_high_conf}/{n} = {n_high_conf/n*100:.1f}%")

    summary = {
        "n_total": written,
        "class_distribution": dict(class_counter),
        "method_usage": dict(method_used),
        "avg_active_voters": avg_voters,
        "confidence_buckets": {f"{k:.1f}": v for k, v in conf_buckets.items()},
        "high_confidence_rate": n_high_conf / n,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

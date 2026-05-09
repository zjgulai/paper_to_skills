"""Phase 5 unified labeler — D7 T7.1

Integrates D1-D6 outputs into a single record schema. Two operating modes:

  --mode merge   join already-produced D2-D6 jsonl files by review_id
                 (no LLM re-call; idempotent; for end-to-end smoke test)

  --mode stream  call label_single_record(record) for each input record,
                 chaining Phase 4 rules + (optional) LLM/ABSA/NPS/persona
                 hooks if the corresponding modules are wired up.

For D7 small-sample, --mode merge is the canonical run since D2-D6 artifacts
already exist. --mode stream remains a contract placeholder for D8 production.

Self-test (--self-test):
  Synthesises 30+ deterministic mini-records and runs label_single_record
  end-to-end. Asserts schema fields and exits 0 on full pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def index_by_id(records: list[dict]) -> dict[str, dict]:
    return {r["review_id"]: r for r in records if r.get("review_id")}


PHASE5_FIELDS = [
    "review_id", "text", "data_source", "language", "rating",
    "labels", "overall_sentiment", "proxy_nps",
    "consensus_labels", "consensus_sentiment", "consensus_nps", "consensus_source",
    "aspects",
    "nps_final", "nps_vote_method", "nps_agreement",
    "proxy_nps_final", "proxy_nps_method_votes", "proxy_nps_confidence",
    "persona_tags", "persona_dimensions", "persona_tag_count", "persona_has_any",
    "phase5_meta",
]


def merge_record(
    base: dict,
    consensus: Optional[dict],
    absa: Optional[dict],
    nps: Optional[dict],
    persona: Optional[dict],
    source: Optional[dict],
) -> dict:
    out: dict = {}
    rid = base.get("review_id") or (source or {}).get("review_id")
    out["review_id"] = rid

    text = base.get("text") or (source or {}).get("text") or (persona or {}).get("text") or ""
    out["text"] = text
    out["data_source"] = base.get("data_source") or (source or {}).get("data_source")
    out["language"] = base.get("language") or (source or {}).get("language")
    out["rating"] = base.get("rating") if base.get("rating") is not None else (source or {}).get("rating")

    out["labels"] = base.get("labels") or []
    out["overall_sentiment"] = base.get("overall_sentiment")
    out["proxy_nps"] = base.get("proxy_nps")

    if consensus:
        out["consensus_labels"] = consensus.get("consensus_labels") or []
        out["consensus_sentiment"] = consensus.get("consensus_sentiment")
        out["consensus_nps"] = consensus.get("consensus_nps")
        out["consensus_source"] = consensus.get("consensus_source")

    if absa:
        out["aspects"] = absa.get("aspects") or []

    if nps:
        out["proxy_nps_final"] = nps.get("proxy_nps_final")
        out["proxy_nps_method_votes"] = nps.get("proxy_nps_method_votes")
        out["proxy_nps_confidence"] = nps.get("proxy_nps_confidence")

    if persona:
        out["persona_tags"] = persona.get("persona_tags") or []
        out["persona_dimensions"] = persona.get("persona_dimensions") or {}
        out["persona_tag_count"] = persona.get("persona_tag_count") or 0
        out["persona_has_any"] = persona.get("persona_has_any") or False

    out["phase5_meta"] = {
        "has_llm_label": bool(base.get("labels")),
        "has_consensus": consensus is not None and bool(consensus.get("consensus_labels")),
        "has_absa": absa is not None and bool(absa.get("aspects")),
        "has_nps_vote": nps is not None and nps.get("proxy_nps_final") in {"promoter", "passive", "detractor"},
        "has_persona": persona is not None and persona.get("persona_has_any", False),
    }
    return out


def run_merge(args) -> dict:
    base = load_jsonl(args.llm_pred)
    base_idx = index_by_id(base)
    consensus_idx = index_by_id(load_jsonl(args.consensus)) if args.consensus and args.consensus.exists() else {}
    absa_idx = index_by_id(load_jsonl(args.absa)) if args.absa and args.absa.exists() else {}
    nps_idx = index_by_id(load_jsonl(args.nps)) if args.nps and args.nps.exists() else {}
    persona_idx = index_by_id(load_jsonl(args.persona)) if args.persona and args.persona.exists() else {}
    src_idx = index_by_id(load_jsonl(args.source_text)) if args.source_text and args.source_text.exists() else {}

    print(f"📂 Loaded base={len(base_idx)} consensus={len(consensus_idx)} absa={len(absa_idx)} "
          f"nps={len(nps_idx)} persona={len(persona_idx)} source={len(src_idx)}")

    n_total = len(base_idx)
    flag_counter: Counter = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for rid, rec in base_idx.items():
            merged = merge_record(
                base=rec,
                consensus=consensus_idx.get(rid),
                absa=absa_idx.get(rid),
                nps=nps_idx.get(rid),
                persona=persona_idx.get(rid),
                source=src_idx.get(rid),
            )
            for k, v in merged.get("phase5_meta", {}).items():
                if v:
                    flag_counter[k] += 1
            f.write(json.dumps(merged, ensure_ascii=False) + "\n")

    summary = {
        "n_total": n_total,
        "coverage": {k: f"{v}/{n_total} = {v/n_total*100:.2f}%" for k, v in flag_counter.items()},
        "flag_counts": dict(flag_counter),
    }
    print("\n" + "=" * 60)
    print("Merge Summary")
    print("=" * 60)
    for k, v in summary["coverage"].items():
        print(f"  {k:25}: {v}")
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📦 Summary: {args.summary_out}")
    print(f"✅ Output: {args.output}")
    return summary


def label_single_record(
    record: dict,
    phase4_label_fn=None,
    llm_label_fn=None,
    persona_label_fn=None,
) -> tuple[list[dict], list[dict], dict]:
    """Phase 4-compatible streaming entry for D8 production.

    Returns: (new_labels, all_labels, meta)
      Phase 4 returned (new_labels, all_labels). meta is the new third tuple
      member carrying llm/persona/consensus flags; downstream Phase 4 consumers
      can ignore meta to remain compatible.
    """
    text = (record.get("text") or "").strip()
    existing = record.get("labels") or []

    new_labels: list[dict] = []
    if phase4_label_fn is not None and text:
        try:
            p4_new, _ = phase4_label_fn(record)
            new_labels.extend(p4_new)
        except Exception as exc:
            new_labels.append({"_phase4_error": str(exc)})

    if llm_label_fn is not None and text and not existing:
        try:
            llm_pred = llm_label_fn(record)
            if isinstance(llm_pred, list):
                new_labels.extend(llm_pred)
        except Exception as exc:
            new_labels.append({"_llm_error": str(exc)})

    persona_tags: list[dict] = []
    if persona_label_fn is not None and text:
        try:
            persona_tags = persona_label_fn(record) or []
        except Exception as exc:
            persona_tags = [{"_persona_error": str(exc)}]

    seen = {l.get("tag_id") for l in existing if l.get("tag_id")}
    all_labels = list(existing)
    for l in new_labels:
        tid = l.get("tag_id")
        if tid and tid not in seen:
            all_labels.append(l)
            seen.add(tid)

    meta = {
        "n_new_labels": len([l for l in new_labels if l.get("tag_id")]),
        "n_persona_tags": len([t for t in persona_tags if t.get("tag_id")]),
        "had_existing_labels": bool(existing),
    }
    return new_labels, all_labels, meta


def run_self_test() -> tuple[int, int, list[str]]:
    """Synthesise 30+ deterministic test records and assert schema integrity."""
    cases: list[tuple[str, dict]] = []
    for i, (text, data_source, rating) in enumerate([
        ("This pump is comfortable and quiet at night.", "amazon_competitor", 5.0),
        ("Worst purchase ever. Will not recommend.", "trustpilot", 1.0),
        ("Highly recommend for working moms.", "trustpilot", 5.0),
        ("Bought as a baby shower gift for my sister.", "amazon_competitor", 4.0),
        ("Easy to clean, dishwasher safe, lightweight.", "momcozy", 5.0),
        ("Sehr empfehlenswert, schnelle Lieferung.", "trustpilot", 5.0),
        ("Je recommande vivement ce produit.", "trustpilot", 5.0),
        ("Stay away, complete waste of money.", "amazon_competitor", 1.0),
        ("First time mom struggling with low supply.", "reddit", None),
        ("My pediatrician recommended this brand.", "amazon_competitor", 5.0),
        ("I do all night feedings, exhausted.", "momcozy", 4.0),
        ("Hospital grade pump for NICU baby.", "zendesk", 0.0),
        ("Quiet enough not to wake the baby.", "amazon_competitor", 5.0),
        ("Hands free pumping while at work.", "amazon_competitor", 4.0),
        ("Saw it on TikTok and had to try.", "trustpilot", 4.0),
        ("Travel pump for our road trip.", "amazon_competitor", 4.0),
        ("Value for money, budget friendly.", "trustpilot", 4.0),
        ("Anxious first time mom, this helped.", "reddit", None),
        ("Disappointed, it broke after a week.", "amazon_competitor", 2.0),
        ("Empty rating short text", "zendesk", 0.0),
        ("So grateful for this purchase.", "trustpilot", 5.0),
        ("Postpartum recovery tool, comfortable.", "momcozy", 5.0),
        ("Brand loyal, always buy Momcozy.", "amazon_competitor", 5.0),
        ("Researched extensively before buying.", "amazon_competitor", 5.0),
        ("Got it as a gift, very happy.", "trustpilot", 5.0),
        ("My friend recommended this to me.", "amazon_competitor", 5.0),
        ("Pumping at work in break room.", "trustpilot", 4.0),
        ("Toddler still nursing, durable bra.", "amazon_competitor", 5.0),
        ("Single mom, need affordable options.", "reddit", None),
        ("Effortless setup, intuitive app.", "momcozy", 5.0),
        ("Premium quality, well made.", "amazon_competitor", 5.0),
        ("", "zendesk", 0.0),
    ]):
        cases.append((f"st_{i:03d}", {
            "review_id": f"st_{i:03d}",
            "text": text,
            "data_source": data_source,
            "rating": rating,
            "labels": [],
        }))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from phase4_unified_labeler import label_single_record as p4_fn
    except Exception as e:
        return 0, len(cases), [f"phase4 import failure: {e}"]

    persona_rules_path = Path(__file__).resolve().parents[1] / "01-设计文档" / "02-工作流设计" / "persona_tags_55.json"
    persona_label_fn = None
    if persona_rules_path.exists():
        from persona_tag_labeler import load_rules, label_one as persona_label_one
        rules = load_rules(persona_rules_path)

        def persona_label_fn(rec: dict) -> list[dict]:
            return persona_label_one(rec, {}, rules).get("persona_tags") or []

    n_pass = 0
    failures: list[str] = []
    for rid, rec in cases:
        try:
            new, all_l, meta = label_single_record(
                rec,
                phase4_label_fn=p4_fn,
                persona_label_fn=persona_label_fn,
            )
            assert isinstance(new, list), "new_labels not list"
            assert isinstance(all_l, list), "all_labels not list"
            assert isinstance(meta, dict), "meta not dict"
            assert "n_new_labels" in meta and "n_persona_tags" in meta
            n_pass += 1
        except AssertionError as exc:
            failures.append(f"{rid}: {exc}")
        except Exception as exc:
            failures.append(f"{rid}: unexpected {type(exc).__name__}: {exc}")

    return n_pass, len(cases), failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["merge", "self-test"], default="merge")

    ap.add_argument("--llm-pred", type=Path, help="D2 LLM-labeled jsonl (5K)")
    ap.add_argument("--consensus", type=Path, help="D4 consensus_result.jsonl")
    ap.add_argument("--absa", type=Path, help="D4 absa_500_pred.jsonl")
    ap.add_argument("--nps", type=Path, help="D5 *_nps_pred.jsonl")
    ap.add_argument("--persona", type=Path, help="D6 *_persona.jsonl")
    ap.add_argument("--source-text", type=Path, help="Stratified jsonl for text/rating join")
    ap.add_argument("--output", type=Path, help="Merged unified jsonl")
    ap.add_argument("--summary-out", type=Path)

    ap.add_argument("--self-test", action="store_true",
                    help="Run 30+ synthetic record self-test (overrides --mode)")

    args = ap.parse_args()

    if args.self_test or args.mode == "self-test":
        n_pass, n_total, failures = run_self_test()
        print(f"\nSelf-test: {n_pass}/{n_total} passed")
        for f in failures:
            print(f"  FAIL  {f}")
        ok = n_pass == n_total and n_total >= 30
        print("\n🎉 SELF-TEST PASS" if ok else "❌ SELF-TEST FAIL")
        return 0 if ok else 1

    if args.mode == "merge":
        if not args.llm_pred or not args.output:
            print("❌ merge mode requires --llm-pred and --output", file=sys.stderr)
            return 2
        run_merge(args)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

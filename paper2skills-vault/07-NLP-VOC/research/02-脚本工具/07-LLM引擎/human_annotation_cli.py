"""Human annotation CLI for golden_set_500.jsonl.

Workflow per record:
  1. Show text + Phase 4 labels + LLM predictions side by side
  2. Operator picks Top-3 tag_ids (by id, comma separated)
       - default: accept LLM picks via 'a' (all) or '1,2,3' (by index)
       - manual:  type tag_id directly (e.g. TAG_GEN_E001)
  3. overall_sentiment: p/n/x  (positive/negative/neutral)
  4. proxy_nps: m/p/d        (proMoter/Passive/Detractor)
  5. optional notes
  6. auto-save after every record (resumable)

Keys:
  Enter alone → accept LLM defaults
  s          → skip (mark for later, golden_labels stays empty)
  q          → quit (progress saved)
  b          → go back one record
  ?          → show help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

LANG_BANNER = {
    "en": "🇬🇧 EN",
    "zh": "🇨🇳 ZH",
    "es": "🇪🇸 ES",
    "de": "🇩🇪 DE",
    "fr": "🇫🇷 FR",
    "it": "🇮🇹 IT",
    "pt": "🇵🇹 PT",
}

SENTIMENT_MAP = {"p": "positive", "n": "negative", "x": "neutral"}
NPS_MAP = {"m": "promoter", "p": "passive", "d": "detractor"}


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_records(path: Path, records: list[dict]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def render_record(idx: int, total: int, rec: dict):
    print("\n" + "=" * 80)
    print(f"  [{idx + 1}/{total}]  review_id={rec['review_id']}")
    print(f"  source={rec['data_source']}   lang={LANG_BANNER.get(rec.get('language'), rec.get('language'))}   rating={rec.get('rating')}")
    print("-" * 80)
    text = rec.get("text") or ""
    print(text[:1500])
    if len(text) > 1500:
        print(f"  …(truncated, total {len(text)} chars)")
    print("-" * 80)

    p4 = rec.get("phase4_labels") or []
    print(f"  Phase 4 labels ({len(p4)}):")
    for lbl in p4:
        print(f"    [P4] {lbl.get('tag_id'):20} {lbl.get('tag_en')}  ({lbl.get('sentiment_calibrated')})")

    llm = rec.get("llm_pred") or []
    print(f"  DeepSeek pred ({len(llm)}):  overall={rec.get('llm_overall_sentiment')}  nps={rec.get('llm_proxy_nps')}")
    for i, lbl in enumerate(llm, 1):
        ev = (lbl.get("evidence") or "")[:60]
        print(f"    [{i}] {lbl.get('tag_id'):20} {lbl.get('tag_en')}  conf={lbl.get('confidence'):.2f}  ev={ev}")

    kimi = rec.get("kimi_pred") or []
    if kimi or rec.get("kimi_overall_sentiment"):
        print(f"  Kimi pred ({len(kimi)}):     overall={rec.get('kimi_overall_sentiment')}  nps={rec.get('kimi_proxy_nps')}")
        for i, lbl in enumerate(kimi, 1):
            ev = (lbl.get("evidence") or "")[:60]
            conf = lbl.get("confidence") or 0
            print(f"    [k{i}] {lbl.get('tag_id'):19} {lbl.get('tag_en')}  conf={conf:.2f}  ev={ev}")

    if rec.get("disagreement_reason"):
        print(f"  ⚠ Disagreement: {rec['disagreement_reason']}")
    if rec.get("golden_source") == "consensus_llm":
        print(f"  ✓ Auto-filled by LLM consensus (jaccard={rec.get('consensus_meta',{}).get('jaccard')})")

    print("-" * 80)
    if rec.get("golden_labels"):
        print(f"  ✓ CURRENT GOLDEN: {[l['tag_id'] for l in rec['golden_labels']]}  "
              f"sent={rec.get('golden_overall_sentiment')}  nps={rec.get('golden_proxy_nps')}  "
              f"src={rec.get('golden_source','manual')}")


def parse_label_input(text: str, llm_preds: list[dict]) -> list[dict] | None:
    text = text.strip()
    if text.lower() == "a":
        return [{"tag_id": l["tag_id"], "tag_en": l.get("tag_en")} for l in llm_preds[:3]]
    if not text:
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out: list[dict] = []
    for p in parts:
        if p.isdigit():
            i = int(p) - 1
            if 0 <= i < len(llm_preds):
                out.append({"tag_id": llm_preds[i]["tag_id"], "tag_en": llm_preds[i].get("tag_en")})
        else:
            out.append({"tag_id": p, "tag_en": None})
    return out[:3] if out else None


def prompt_value(prompt: str, mapping: dict[str, str], default: str | None) -> str | None:
    valid = "/".join(mapping.keys())
    suffix = f" (default {default})" if default else ""
    while True:
        raw = input(f"  {prompt} [{valid}]{suffix}: ").strip().lower()
        if not raw and default:
            return mapping.get(default, default)
        if raw in mapping:
            return mapping[raw]
        if raw == "":
            return None
        print(f"    ❌ unknown '{raw}' — please type one of {valid}")


def annotate_one(rec: dict) -> str:
    llm_preds = rec.get("llm_pred") or []
    raw = input("\n  Top-3 tag_ids (Enter=accept LLM, 'a'=all LLM, '1,3'=by index, 's'=skip, 'b'=back, 'q'=quit): ").strip()
    if raw.lower() == "q":
        return "quit"
    if raw.lower() == "b":
        return "back"
    if raw.lower() == "s":
        return "skip"
    if raw.lower() == "?":
        print(__doc__)
        return "again"

    if not raw:
        labels = parse_label_input("a", llm_preds)
    else:
        labels = parse_label_input(raw, llm_preds)
    if labels is None:
        print("    ⚠ no label parsed — try again or 's' to skip")
        return "again"

    default_sent = (rec.get("llm_overall_sentiment") or "neutral")
    default_sent_short = {v: k for k, v in SENTIMENT_MAP.items()}.get(default_sent)
    sentiment = prompt_value("overall_sentiment", SENTIMENT_MAP, default_sent_short) or default_sent

    default_nps = (rec.get("llm_proxy_nps") or "passive")
    default_nps_short = {v: k for k, v in NPS_MAP.items()}.get(default_nps)
    nps = prompt_value("proxy_nps", NPS_MAP, default_nps_short) or default_nps

    notes = input("  notes (Enter to skip): ").strip()

    rec["golden_labels"] = labels
    rec["golden_overall_sentiment"] = sentiment
    rec["golden_proxy_nps"] = nps
    rec["golden_notes"] = notes
    rec["golden_source"] = "human"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="golden_set_500.jsonl")
    ap.add_argument("--start", type=int, default=0, help="start index (0-based)")
    ap.add_argument("--only-pending", action="store_true", help="skip already-annotated records")
    ap.add_argument("--only-disagreement", action="store_true",
                    help="only show records where DeepSeek and Kimi disagreed (golden_source=needs_human)")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}"); sys.exit(2)

    records = load_records(args.input)
    total = len(records)
    n_done = sum(1 for r in records if r.get("golden_labels"))
    n_consensus = sum(1 for r in records if r.get("golden_source") == "consensus_llm")
    n_needs_human = sum(1 for r in records if r.get("golden_source") == "needs_human")
    print(f"\nLoaded {total} records from {args.input}")
    print(f"  Already annotated: {n_done} / {total}")
    print(f"    consensus_llm    : {n_consensus}")
    print(f"    needs_human      : {n_needs_human}")
    if args.only_disagreement:
        print(f"  Mode: --only-disagreement  (showing {n_needs_human} records)")

    i = args.start
    while 0 <= i < total:
        rec = records[i]
        if args.only_pending and rec.get("golden_labels"):
            i += 1
            continue
        if args.only_disagreement and rec.get("golden_source") != "needs_human":
            i += 1
            continue
        render_record(i, total, rec)
        result = annotate_one(rec)
        if result == "quit":
            save_records(args.input, records)
            print(f"\n💾 Saved. Progress: {sum(1 for r in records if r.get('golden_labels'))}/{total}")
            return
        if result == "back":
            i = max(0, i - 1)
            continue
        if result == "skip":
            i += 1
            continue
        if result == "again":
            continue
        save_records(args.input, records)
        i += 1

    save_records(args.input, records)
    n_done = sum(1 for r in records if r.get("golden_labels"))
    print(f"\n🎉 All done. Final: {n_done}/{total} annotated.")


if __name__ == "__main__":
    main()

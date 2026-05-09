"""ABSA (Aspect-Based Sentiment Analysis) extractor — LLM-driven.

Extracts a list of (aspect, sentiment, evidence_span, confidence) triples per
review. Uses the same DeepSeek-V4-Flash / Kimi-K2.6 client as Phase 5 D1-D3.

Output schema per record (jsonl, one record per input):
  review_id    : passthrough
  success      : bool
  aspects      : [
      {
        "aspect"          : "battery life",          # noun phrase in evidence_lang
        "sentiment"       : "positive"|"neutral"|"negative",
        "evidence_span"   : "after 6 months still holds charge well",
        "confidence"      : 0.0..1.0,
      }, ...
    ]
  _meta        : { model, tokens_in, tokens_out, cache_hit, latency_ms, retries, error }

Domain priors injected into the system prompt:
  - Mother & baby cross-border e-commerce context
  - Common aspect types: product quality, comfort, fit/sizing, price/value,
    delivery, customer service, packaging, durability, ingredient safety
  - Reject overly generic aspects ("good", "nice") — must be a concrete noun phrase
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07-LLM引擎"))

from llm_client import LLMClient
from pydantic import BaseModel, Field, ValidationError

MAX_ASPECTS = 5
DEFAULT_MAX_TOKENS = 800


SYSTEM_PROMPT = """You are an Aspect-Based Sentiment Analysis (ABSA) expert for mother & baby cross-border e-commerce reviews.

For each review, extract concrete aspect-sentiment triples. Each triple captures ONE specific attribute of the product or shopping experience that the reviewer commented on.

Rules:
1. Aspect MUST be a concrete noun phrase (2-4 words). Reject vague terms like "good", "nice", "love it".
   GOOD: "battery life", "shoulder strap", "delivery speed", "customer service rep", "pump suction"
   BAD : "good", "nice", "experience", "thing", "product"
2. Sentiment ∈ {positive, neutral, negative} — based on the reviewer's stance toward THAT aspect only.
3. evidence_span: copy the exact original text fragment (≤ 30 words) supporting this aspect-sentiment.
4. confidence: 0.0–1.0, calibrated. ≥0.9 = unambiguous; 0.7–0.9 = clear; 0.5–0.7 = inferred.
5. Output up to 5 aspects per review. If the review has no clear aspects, output empty list.
6. Prefer aspects in the reviewer's original language for the aspect string.
7. Domain hints: pumps, bottles, baby monitors, strollers, wipes, diapers, baby clothes, formula, breastfeeding, sleep accessories.

Output ONLY valid JSON in this exact schema, no markdown:
{
  "aspects": [
    {"aspect": "...", "sentiment": "positive|neutral|negative", "evidence_span": "...", "confidence": 0.85}
  ]
}
"""


class AspectItem(BaseModel):
    aspect: str = Field(min_length=2, max_length=80)
    sentiment: str = Field(pattern=r"^(positive|neutral|negative)$")
    evidence_span: str = Field(max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)


class ABSAOutput(BaseModel):
    aspects: list[AspectItem] = Field(default_factory=list, max_length=MAX_ASPECTS * 2)


def extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        s = m.group(1)
    if not (s.startswith("{") and s.rstrip().endswith("}")):
        m2 = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m2:
            s = m2.group(1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


async def extract_one(client: LLMClient, rec: dict, vendor: str, model: Optional[str]) -> dict:
    text = (rec.get("text") or "").strip()
    if not text:
        return {"review_id": rec.get("review_id"), "success": True, "aspects": [], "_meta": {"skipped": "empty_text"}}

    user_msg = f"Review (lang={rec.get('language','?')}, source={rec.get('data_source','?')}, rating={rec.get('rating','?')}):\n\n{text[:2000]}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    last_err = ""
    last_resp = None
    for attempt in range(3):
        try:
            resp = await client.chat_async(
                vendor=vendor,
                messages=messages,
                model=model,
                response_format={"type": "json_object"},
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0.1,
            )
            last_resp = resp
            parsed = extract_json(resp.content)
            if parsed is None:
                last_err = "json_parse_fail"
                messages = messages + [
                    {"role": "assistant", "content": resp.content[:500]},
                    {"role": "user", "content": "Your previous response was not valid JSON. Reply ONLY with the exact JSON schema, no markdown, no commentary."},
                ]
                continue
            try:
                validated = ABSAOutput.model_validate(parsed)
            except ValidationError as e:
                last_err = f"schema:{str(e)[:100]}"
                messages = messages + [
                    {"role": "assistant", "content": resp.content[:500]},
                    {"role": "user", "content": f"Schema error: {last_err}. Fix and reply ONLY with valid JSON."},
                ]
                continue
            aspects = [a.model_dump() for a in validated.aspects[:MAX_ASPECTS]]
            return {
                "review_id": rec.get("review_id"),
                "success": True,
                "aspects": aspects,
                "_meta": {
                    "model": resp.model_used,
                    "tokens_in": resp.tokens_in,
                    "tokens_out": resp.tokens_out,
                    "cache_hit": resp.cache_hit_tokens,
                    "latency_ms": round(resp.latency_ms, 1),
                    "retries": attempt,
                    "error": "",
                },
            }
        except Exception as e:
            last_err = str(e)[:200]
            await asyncio.sleep(1.0 * (attempt + 1))

    return {
        "review_id": rec.get("review_id"),
        "success": False,
        "aspects": [],
        "_meta": {
            "model": (last_resp.model_used if last_resp else (model or "")),
            "tokens_in": (last_resp.tokens_in if last_resp else 0),
            "tokens_out": (last_resp.tokens_out if last_resp else 0),
            "cache_hit": (last_resp.cache_hit_tokens if last_resp else 0),
            "latency_ms": (round(last_resp.latency_ms, 1) if last_resp else 0.0),
            "retries": 3,
            "error": last_err or "unknown",
        },
    }


async def run_batch(client: LLMClient, records: list[dict], output_path: Path,
                    vendor: str, model: Optional[str], progress_every: int = 50) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sem = client._get_semaphore(vendor)
    print(f"  Concurrency: {sem._value}")

    n_total = len(records)
    t0 = time.time()
    written = 0

    tasks = [asyncio.create_task(extract_one(client, r, vendor, model)) for r in records]
    f = output_path.open("w", encoding="utf-8")
    n_success = n_with_aspects = total_aspects = 0
    try:
        for fut in asyncio.as_completed(tasks):
            r = await fut
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if r.get("success"):
                n_success += 1
                if r.get("aspects"):
                    n_with_aspects += 1
                    total_aspects += len(r["aspects"])
            if written % progress_every == 0 or written == n_total:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                print(f"  [{written}/{n_total}] rate={rate:.2f}/s success={n_success} with_aspects={n_with_aspects} avg_aspects={total_aspects/max(n_with_aspects,1):.1f}")
    finally:
        f.close()

    elapsed = time.time() - t0
    return {
        "n_total": n_total,
        "n_success": n_success,
        "n_with_aspects": n_with_aspects,
        "total_aspects": total_aspects,
        "avg_aspects_per_record": total_aspects / max(n_with_aspects, 1),
        "empty_pct": (n_success - n_with_aspects) / max(n_success, 1),
        "elapsed_sec": elapsed,
        "rate_per_sec": n_total / elapsed if elapsed > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vendor", choices=["deepseek", "kimi"], default="deepseek")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}"); sys.exit(2)

    records = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]
    print(f"Input:  {len(records)} records")
    print(f"Vendor: {args.vendor}, Model: {args.model or '(default)'}")

    client = LLMClient()
    summary = asyncio.run(run_batch(client, records, args.output, args.vendor, args.model))

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 60)
    for k, v in summary.items():
        print(f"  {k:25}: {v}")
    print("=" * 60)
    print(f"\n✅ Output: {args.output}")
    print(f"📦 Summary: {summary_path}")


if __name__ == "__main__":
    main()

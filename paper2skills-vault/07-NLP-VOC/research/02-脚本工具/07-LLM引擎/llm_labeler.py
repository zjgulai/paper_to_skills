"""LLM Labeler - Phase 5 D2.T2.1

闭集多标签打标器。从 v3.9 字典 (602 唯一 tag_id) 中选 1-5 个标签 + overall_sentiment + proxy_nps。

输出 schema 与 v3.9 兼容（labels[] 内含 tag_id + sentiment_calibrated + confidence + source），
便于 phase4_unified_labeler 风格的下游消费。

设计要点：
- System prompt = 字典紧凑版（~7K tokens），DeepSeek V4-Flash cache hit 复用
- response_format JSON mode + Pydantic 校验：tag_id 必须 ∈ ALL_TAG_IDS
- 失败重试：JSON 解析失败/tag_id 越集 → 自动重试 1 次（更严格 prompt），仍失败标记 invalid
- 异步并发：通过 LLMClient 的 semaphore 控流
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import LLMClient, LLMResponse
from tag_dict_loader import build_compact_prompt, get_all_tag_ids


SCHEMA_INSTRUCTION = """# 任务
对一条母婴出海跨境电商场景的 VOC 评论，输出 JSON：

```json
{
  "labels": [
    {"tag_id": "TAG_L1_002", "confidence": 0.92, "evidence": "原文片段"}
  ],
  "overall_sentiment": "positive|negative|neutral",
  "proxy_nps": "promoter|passive|detractor"
}
```

# 规则（严格遵守）
1. tag_id **必须**是字典中已列出的 ID，禁止编造
2. labels 列表 1-5 项；按相关性降序；confidence < 0.5 不输出
3. evidence 是原文连续片段（≤80 字符），用于回溯
4. 若评论既正向又负向，使用 overall_sentiment="neutral" 或选主导极性
5. proxy_nps 三分类映射：极推荐→promoter / 中性满意→passive / 抱怨/差评→detractor
6. **只输出 JSON，无任何解释**
"""


def get_system_prompt() -> str:
    return build_compact_prompt() + "\n\n" + SCHEMA_INSTRUCTION


class LabelItem(BaseModel):
    tag_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""

    @field_validator("tag_id")
    @classmethod
    def tag_id_in_dict(cls, v: str) -> str:
        valid = get_all_tag_ids()
        if v not in valid:
            raise ValueError(f"tag_id '{v}' not in v3.9 dictionary (602 tags)")
        return v

    @field_validator("evidence")
    @classmethod
    def truncate_evidence(cls, v: str) -> str:
        return (v or "")[:120]


class LLMLabelOutput(BaseModel):
    labels: list[LabelItem] = Field(default_factory=list, max_length=8)
    overall_sentiment: str = "neutral"
    proxy_nps: str = "passive"

    @field_validator("overall_sentiment")
    @classmethod
    def check_sent(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"positive", "negative", "neutral"}:
            return "neutral"
        return v

    @field_validator("proxy_nps")
    @classmethod
    def check_nps(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"promoter", "passive", "detractor"}:
            return "passive"
        return v


@dataclass
class LabelingResult:
    review_id: str
    success: bool
    parsed: Optional[dict] = None
    raw_content: str = ""
    error: str = ""
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    json_parse_failed: bool = False
    schema_invalid: bool = False
    invalid_tag_ids: list[str] = field(default_factory=list)


def to_v39_label_dicts(parsed: LLMLabelOutput, source_tag: str = "llm_v4flash") -> list[dict]:
    out = []
    for li in parsed.labels:
        out.append(
            {
                "tag_id": li.tag_id,
                "confidence": round(li.confidence, 3),
                "evidence": li.evidence,
                "source": source_tag,
            }
        )
    return out


def _parse_json_lenient(raw: str) -> dict:
    """Tolerate markdown fences and trailing junk while keeping json mode strict.

    DeepSeek/Kimi occasionally emit ```json ... ``` despite response_format=json_object,
    or truncate the closing brace when max_tokens hits. We try the stdlib first, then
    fall back to extracting the longest balanced {...} substring.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        if first_nl > 0:
            raw = raw[first_nl + 1 :]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start < 0:
        raise json.JSONDecodeError("no '{' found", raw, 0)
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        last_brace = raw.rfind("}")
        if last_brace > start:
            return json.loads(raw[start : last_brace + 1] + "}" * max(0, depth))
        raise json.JSONDecodeError("unbalanced braces", raw, len(raw))
    return json.loads(raw[start:end])


async def label_one(
    client: LLMClient,
    record: dict,
    vendor: str = "deepseek",
    model: Optional[str] = None,
    max_retries_invalid: int = 1,
) -> LabelingResult:
    review_id = str(record.get("review_id", record.get("_idx", "")))
    text = (record.get("text") or "").strip()
    if not text:
        return LabelingResult(review_id=review_id, success=False, error="empty text")

    sys_prompt = get_system_prompt()
    user_prompt = f"评论：\n{text[:1500]}"

    attempts = 0
    last_raw = ""
    last_err = ""
    last_resp: Optional[LLMResponse] = None
    invalid_ids: list[str] = []

    while attempts <= max_retries_invalid:
        try:
            resp = await client.chat_async(
                vendor=vendor,
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1200,
                temperature=0.2,
            )
            last_resp = resp
            last_raw = resp.content
        except Exception as e:
            return LabelingResult(
                review_id=review_id,
                success=False,
                error=f"api_error: {e}",
                model_used=model or "",
            )

        try:
            raw_obj = _parse_json_lenient(last_raw)
        except json.JSONDecodeError as e:
            last_err = f"json_parse: {e}"
            attempts += 1
            user_prompt = (
                f"评论：\n{text[:1500]}\n\n"
                f"⚠ 上次输出 JSON 不合法（可能被截断），请控制 evidence ≤80 字符、labels ≤3 项，确保 600 tokens 内完成"
            )
            continue

        try:
            parsed = LLMLabelOutput.model_validate(raw_obj)
            return LabelingResult(
                review_id=review_id,
                success=True,
                parsed=parsed.model_dump(),
                raw_content=last_raw,
                model_used=last_resp.model_used,
                tokens_in=last_resp.tokens_in,
                tokens_out=last_resp.tokens_out,
                cache_hit_tokens=last_resp.cache_hit_tokens,
                latency_ms=last_resp.latency_ms,
                retries=last_resp.retries,
            )
        except ValidationError as ve:
            last_err = f"schema: {ve.error_count()} errors"
            invalid_ids = []
            for err in ve.errors():
                if err.get("type") == "value_error" and "tag_id" in str(err.get("loc", [])):
                    msg = str(err.get("msg", ""))
                    if "'" in msg:
                        try:
                            invalid_ids.append(msg.split("'")[1])
                        except IndexError:
                            pass
            attempts += 1
            invalid_part = f"\n以下 tag_id 不存在: {invalid_ids[:5]}" if invalid_ids else ""
            user_prompt = (
                f"评论：\n{text[:1500]}\n\n"
                f"⚠ 上次输出 schema 校验失败 ({last_err}){invalid_part}\n"
                f"重新输出，tag_id 必须严格选自字典"
            )
            continue

    return LabelingResult(
        review_id=review_id,
        success=False,
        error=last_err,
        raw_content=last_raw,
        json_parse_failed=("json_parse" in last_err),
        schema_invalid=("schema" in last_err),
        invalid_tag_ids=invalid_ids,
        model_used=last_resp.model_used if last_resp else (model or ""),
        tokens_in=last_resp.tokens_in if last_resp else 0,
        tokens_out=last_resp.tokens_out if last_resp else 0,
        cache_hit_tokens=last_resp.cache_hit_tokens if last_resp else 0,
        latency_ms=last_resp.latency_ms if last_resp else 0.0,
    )


async def run_batch(
    client: LLMClient,
    records: list[dict],
    output_path: Path,
    vendor: str = "deepseek",
    model: Optional[str] = None,
    progress_every: int = 50,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sem = client._get_semaphore(vendor)
    print(f"  并发上限: {sem._value}")

    results: list[LabelingResult] = []
    n_total = len(records)
    t0 = time.time()
    written = 0

    async def worker(rec: dict, idx: int) -> LabelingResult:
        rec.setdefault("_idx", idx)
        r = await label_one(client, rec, vendor=vendor, model=model)
        return r

    tasks = [asyncio.create_task(worker(r, i)) for i, r in enumerate(records)]

    f = output_path.open("w", encoding="utf-8")
    try:
        for fut in asyncio.as_completed(tasks):
            r = await fut
            results.append(r)
            line = {
                "review_id": r.review_id,
                "success": r.success,
                "labels": (r.parsed or {}).get("labels", []),
                "overall_sentiment": (r.parsed or {}).get("overall_sentiment"),
                "proxy_nps": (r.parsed or {}).get("proxy_nps"),
                "_meta": {
                    "model": r.model_used,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "cache_hit": r.cache_hit_tokens,
                    "latency_ms": round(r.latency_ms, 1),
                    "retries": r.retries,
                    "error": r.error if not r.success else "",
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            if written % progress_every == 0 or written == n_total:
                done_pct = written / n_total * 100
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (n_total - written) / rate if rate > 0 else 0
                ok = sum(1 for x in results if x.success)
                print(
                    f"  [{written}/{n_total} {done_pct:.1f}%] "
                    f"ok={ok} fail={written-ok} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s"
                )
    finally:
        f.close()

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok
    n_json_fail = sum(1 for r in results if r.json_parse_failed)
    n_schema_fail = sum(1 for r in results if r.schema_invalid)
    sum_in = sum(r.tokens_in for r in results)
    sum_out = sum(r.tokens_out for r in results)
    sum_cache = sum(r.cache_hit_tokens for r in results)
    n_with_label = sum(
        1 for r in results if r.success and r.parsed and r.parsed.get("labels")
    )
    coverage = n_with_label / n_total if n_total else 0

    summary = {
        "n_total": n_total,
        "n_success": n_ok,
        "n_failed": n_fail,
        "n_json_fail": n_json_fail,
        "n_schema_fail": n_schema_fail,
        "success_rate": n_ok / n_total if n_total else 0,
        "json_fail_rate": n_json_fail / n_total if n_total else 0,
        "n_with_label": n_with_label,
        "coverage": coverage,
        "elapsed_sec": round(elapsed, 1),
        "throughput_per_sec": round(n_total / elapsed, 2) if elapsed else 0,
        "tokens_in_total": sum_in,
        "tokens_out_total": sum_out,
        "tokens_in_cache_hit_total": sum_cache,
        "cache_hit_rate": (sum_cache / sum_in) if sum_in else 0,
        "avg_tokens_in": sum_in / n_ok if n_ok else 0,
        "avg_tokens_out": sum_out / n_ok if n_ok else 0,
        "avg_latency_ms": (sum(r.latency_ms for r in results) / n_ok) if n_ok else 0,
        "vendor": vendor,
        "model": model or "default",
    }

    print("\n" + "=" * 70)
    print("Batch Summary")
    print("=" * 70)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:30}: {v:.4f}")
        else:
            print(f"  {k:30}: {v}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Phase 5 LLM Labeler")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vendor", choices=["deepseek", "kimi"], default="deepseek")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Input not found: {args.input}")
        sys.exit(2)

    records = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if args.limit and len(records) >= args.limit:
                break

    print(f"Loaded {len(records)} records from {args.input}")
    print(f"Vendor: {args.vendor}, Model: {args.model or '(default)'}")

    client = LLMClient()
    summary = asyncio.run(
        run_batch(client, records, args.output, vendor=args.vendor, model=args.model)
    )

    summary_path = args.summary_out or args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSummary saved: {summary_path}")

    pass_d2 = (
        summary["success_rate"] >= 0.95
        and summary["json_fail_rate"] < 0.01
        and summary["coverage"] >= 0.92
        and summary["cache_hit_rate"] >= 0.90
    )
    print(f"\n{'🎉 D2.QA PASS' if pass_d2 else '⚠️  D2.QA FAIL'}")
    sys.exit(0 if pass_d2 else 1)


if __name__ == "__main__":
    main()

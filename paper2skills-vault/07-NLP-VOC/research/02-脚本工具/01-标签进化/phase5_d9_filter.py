"""D9 T9.3 — three-filter pipeline on candidate tags (frequency + Jaccard + LLM relevance).

Pipeline:
  F1 (frequency): support_count >= --min-support
  F2 (Jaccard dedupe): semantic distance to existing v3.9 tags < --jaccard-max → reject
  F3 (LLM relevance): batched LLM scores 1-5; keep >= --relevance-min

Output: auto_approved_candidates.json (input for alchemist_label_generator.py)

Usage:
  python phase5_d9_filter.py \\
      --input research/04-输出结果/tag_gap_analysis/candidate_tags_raw.json \\
      --dict-xlsx research/04-输出结果/01-字典版本/tag_dictionary_v3.9.xlsx \\
      --output research/04-输出结果/tag_gap_analysis/auto_approved_candidates.json \\
      --report research/04-输出结果/03-审计报告/phase5_d9_filter_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07-LLM引擎"))


def normalize_tag(s: str) -> set[str]:
    return {t for t in s.lower().replace("_", " ").replace("-", " ").split() if len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_existing_tags(xlsx_path: Path) -> list[tuple[str, set[str]]]:
    import pandas as pd
    xl = pd.ExcelFile(xlsx_path)
    candidate_columns = (
        "tag_en", "tag_cn",
        "VOC标签（英文）", "VOC标签（中文）",
        "英文关键词/典型表达", "消费者习惯关键词/原话短语",
        "VOC标签",
    )
    tags: list[tuple[str, set[str]]] = []
    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet)
        except Exception:
            continue
        for col in candidate_columns:
            if col in df.columns:
                for v in df[col].dropna().astype(str):
                    tokens = normalize_tag(v)
                    if tokens:
                        tags.append((v, tokens))
    return tags


def filter_frequency(candidates, min_support):
    return [c for c in candidates if (c.get("support_count") or 0) >= min_support]


def filter_jaccard(candidates, existing, max_jacc):
    kept, rejected = [], []
    for c in candidates:
        cand_tokens = normalize_tag(c.get("tag_en") or "")
        max_j = 0.0
        max_match = ""
        for name, tokens in existing:
            j = jaccard(cand_tokens, tokens)
            if j > max_j:
                max_j = j
                max_match = name
        c["_max_jaccard"] = round(max_j, 3)
        c["_nearest_existing"] = max_match
        (kept if max_j < max_jacc else rejected).append(c)
    return kept, rejected


def build_relevance_prompt(batch):
    lines = [
        "You are evaluating candidate NEW tags for a cross-border mother & baby e-commerce VOC system.",
        "Brand: Momcozy (breast pumps, nursing bras, baby pillows, bassinets, sterilizers).",
        "",
        "For each candidate, score BUSINESS RELEVANCE 1-5:",
        "  5 = clearly useful product/feature/sentiment tag",
        "  4 = relevant but may overlap with existing tag",
        "  3 = marginal — useful in some scenarios",
        "  2 = noisy / generic / not actionable",
        '  1 = garbage (email metadata, random phrases)',
        "",
        'Respond strict JSON: {"results":[{"idx":N,"score":1-5,"reason":"..."}]}',
        "",
        "Candidates:",
    ]
    for i, c in enumerate(batch):
        lines.append(f'  [{i}] tag_en="{c["tag_en"]}" category="{c.get("applicable_category", "?")}" support={c.get("support_count", 0)}')
    return "\n".join(lines)


async def score_relevance_batched(candidates, client, batch_size=40):
    import re as _re
    scores = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        prompt = build_relevance_prompt(batch)
        try:
            resp = await client.chat_async(
                vendor="deepseek",
                messages=[
                    {"role": "system", "content": "You rate VOC tag candidates for business relevance. Output strict JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=4000,
                temperature=0.1,
            )
            content = resp.content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                m = _re.search(r"\{.*\}", content, _re.DOTALL)
                parsed = json.loads(m.group(0)) if m else {"results": []}
            results = parsed.get("results") if isinstance(parsed, dict) else parsed
            if not isinstance(results, list):
                print(f"⚠ batch {start}: unexpected response", file=sys.stderr)
                continue
            for item in results:
                idx = item.get("idx") if isinstance(item, dict) else None
                if idx is not None and 0 <= idx < len(batch):
                    scores[start + idx] = {
                        "score": int(item.get("score") or 0),
                        "reason": str(item.get("reason") or "")[:120],
                    }
            print(f"  batch {start//batch_size + 1}/{(len(candidates) + batch_size - 1)//batch_size}: {len(results)} scored")
        except Exception as e:
            print(f"⚠ batch {start} failed: {e}", file=sys.stderr)
    return scores


def filter_llm_relevance(candidates, min_score):
    import asyncio
    from llm_client import LLMClient
    if not candidates:
        return [], []
    client = LLMClient()
    scores = asyncio.run(score_relevance_batched(candidates, client))
    kept, rejected = [], []
    for i, c in enumerate(candidates):
        s = scores.get(i, {"score": 0, "reason": "no_llm_response"})
        c["_llm_score"] = s["score"]
        c["_llm_reason"] = s["reason"]
        (kept if s["score"] >= min_score else rejected).append(c)
    return kept, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--dict-xlsx", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--jaccard-max", type=float, default=0.3)
    ap.add_argument("--relevance-min", type=int, default=3)
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Not found: {args.input}", file=sys.stderr); return 2
    if not args.dict_xlsx.exists():
        print(f"❌ Not found: {args.dict_xlsx}", file=sys.stderr); return 2

    candidates = json.loads(args.input.read_text(encoding="utf-8"))
    n0 = len(candidates)
    print(f"Loaded {n0} raw candidates")

    print(f"\nF1 frequency >= {args.min_support}")
    kept_f1 = filter_frequency(candidates, args.min_support)
    print(f"  {n0} -> {len(kept_f1)}")

    print(f"\nF2 Jaccard < {args.jaccard_max}")
    existing = load_existing_tags(args.dict_xlsx)
    print(f"  Existing pool: {len(existing)} tags")
    kept_f2, rejected_f2 = filter_jaccard(kept_f1, existing, args.jaccard_max)
    print(f"  {len(kept_f1)} -> {len(kept_f2)} ({len(rejected_f2)} dedup-rejected)")

    if args.skip_llm or not kept_f2:
        print(f"\nF3 SKIPPED")
        approved = kept_f2
        rejected_f3 = []
    else:
        print(f"\nF3 LLM relevance >= {args.relevance_min}/5")
        approved, rejected_f3 = filter_llm_relevance(kept_f2, args.relevance_min)
        print(f"  {len(kept_f2)} -> {len(approved)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(approved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ {len(approved)} approved → {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# D9 T9.3 候选标签三过滤报告",
            "",
            f"**输入**: {n0} 原始候选  ｜  **输出**: {len(approved)} 通过",
            "",
            "| 过滤器 | 阈值 | 拒绝 | 保留 |",
            "|---|---|---:|---:|",
            f"| F1 频率 | support >= {args.min_support} | {n0 - len(kept_f1)} | {len(kept_f1)} |",
            f"| F2 Jaccard 去重 | distance < {args.jaccard_max} | {len(rejected_f2)} | {len(kept_f2)} |",
            f"| F3 LLM 业务相关性 | score >= {args.relevance_min}/5 | {len(rejected_f3)} | {len(approved)} |",
            "",
            f"**最终通过**: **{len(approved)}**  目标范围 [20, 40]",
            "",
            "## 通过候选标签（按 support 降序）",
            "",
            "| tag_en | category | support | Jaccard 最近 | LLM | reason |",
            "|---|---|---:|---|---:|---|",
        ]
        for c in sorted(approved, key=lambda x: -(x.get("support_count") or 0))[:50]:
            lines.append(
                f"| `{c.get('tag_en', '')}` | {c.get('applicable_category', '')} | "
                f"{c.get('support_count', '')} | {c.get('_nearest_existing', '')[:30]} | "
                f"{c.get('_llm_score', '-')} | {c.get('_llm_reason', '')[:80]} |"
            )
        if rejected_f3:
            lines += [
                "",
                "## F3 LLM 拒绝（top 30 by lowest score）",
                "",
                "| tag_en | support | LLM | reason |",
                "|---|---:|---:|---|",
            ]
            for c in sorted(rejected_f3, key=lambda x: x.get("_llm_score", 0))[:30]:
                lines.append(
                    f"| `{c.get('tag_en', '')}` | {c.get('support_count', '')} | "
                    f"{c.get('_llm_score', 0)} | {c.get('_llm_reason', '')[:100]} |"
                )
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"📄 Report: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

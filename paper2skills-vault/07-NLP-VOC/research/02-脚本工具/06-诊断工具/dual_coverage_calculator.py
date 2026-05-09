"""Phase 5 D10 — 双覆盖率计算器

口径定义（来源 voc-tag-evolution-phase5-product-closed-loop-plan.md T10.2/T10.3）：

- 原始覆盖率 = with_label / total
    - with_label: n_tags >= 1
- 广义覆盖率 = (with_label OR brand_only) / total
    - brand_only: n_tags == 0 但 brand_mentions 非空（命中品牌词典）
- 业务有效覆盖率 = with_label / effective_denominator
    - effective_denominator = total - excluded
    - excluded（互斥三类，按优先级判定）：
        1. too_short：text.strip() 字符数 < 30（极短无意义）
        2. off_category：product_line 为空 且 无任何标签 且 无品牌 mentions（非品类）
        3. generic_only：n_tags==0 且 文本去停用词后有效 token 数 < 3（泛化评价/无信息量）

阈值（决策 4，T10 QA 场景 2）：
- 原始覆盖率 ≥ 0.88
- 业务有效覆盖率 ≥ 0.94

用法：
    # 完整报告（含排除追踪抽样 100 条）
    python dual_coverage_calculator.py \
        --input <vault>/research/04-输出结果/unified_labeling/phase5_intermediate_merged.jsonl \
        --report <vault>/research/04-输出结果/03-审计报告/phase5_dual_coverage_report.md \
        --with-exclusion-trace

    # 阈值自动判定（CI 友好）
    python dual_coverage_calculator.py \
        --input ...phase5_intermediate_merged.jsonl \
        --threshold-raw 0.88 --threshold-effective 0.94 \
        --json-output <vault>/research/04-输出结果/03-审计报告/phase5_dual_coverage_thresholds.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "to", "from", "in", "on", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "my", "your", "his", "her", "our", "their", "me", "him", "us",
    "them", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "just", "don", "should", "now",
    "really", "definitely", "absolutely", "highly", "recommend", "great",
    "good", "nice", "love", "loved", "like", "liked", "use", "used",
    "would", "will", "one", "two", "thing", "things", "something",
    "product", "item", "order", "buy", "bought", "purchase",
    "got", "get", "make", "made", "go", "going", "went", "come", "came",
}

GENERIC_TOKEN_THRESHOLD = 3
TOO_SHORT_CHARS = 30
SAMPLE_PER_BUCKET = 100


def meaningful_token_count(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"[a-z]{3,}", text.lower())
    return sum(1 for w in words if w not in STOP_WORDS)


def classify_exclusion(record: dict[str, Any]) -> str | None:
    """返回排除桶名 or None。互斥三类按优先级判定（顺序敏感，对应 T10.2）。

    口径修正：n_tags >= 1 的记录已有有效信号，不应进入业务有效分母的排除桶；
    排除桶仅针对 n_tags == 0 的"无信号"记录细分原因。
    """
    n_tags = record.get("n_tags") or 0
    if n_tags >= 1:
        return None

    text = (record.get("text") or "").strip()
    brand_mentions = record.get("brand_mentions") or []
    product_line = record.get("product_line")

    # 1. too_short
    if len(text) < TOO_SHORT_CHARS:
        return "too_short"

    # 2. off_category
    if not brand_mentions and (product_line is None or str(product_line).strip() == ""):
        return "off_category"

    # 3. generic_only
    if meaningful_token_count(text) < GENERIC_TOKEN_THRESHOLD:
        return "generic_only"

    return None


def calculate(
    input_path: Path,
    *,
    sample_seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(sample_seed)

    total = 0
    with_label = 0
    brand_only = 0

    excl_counts: Counter[str] = Counter()
    by_source = defaultdict(lambda: {"total": 0, "with_label": 0, "brand_only": 0,
                                     "too_short": 0, "off_category": 0, "generic_only": 0})

    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)

    def reservoir_add(bucket: str, item: dict[str, Any]) -> None:
        # 蓄水池采样：固定容量 SAMPLE_PER_BUCKET，每个桶独立计数
        seen[bucket] += 1
        if len(samples[bucket]) < SAMPLE_PER_BUCKET:
            samples[bucket].append(item)
        else:
            j = rng.randint(0, seen[bucket] - 1)
            if j < SAMPLE_PER_BUCKET:
                samples[bucket][j] = item

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            src = r.get("data_source") or "unknown"
            src_stat = by_source[src]
            src_stat["total"] += 1

            n_tags = r.get("n_tags") or 0
            brand_mentions = r.get("brand_mentions") or []

            if n_tags >= 1:
                with_label += 1
                src_stat["with_label"] += 1
            elif brand_mentions:
                brand_only += 1
                src_stat["brand_only"] += 1

            bucket = classify_exclusion(r)
            if bucket is not None:
                excl_counts[bucket] += 1
                src_stat[bucket] += 1
                reservoir_add(bucket, {
                    "review_id": r.get("review_id"),
                    "text_preview": (r.get("text") or "")[:200],
                    "n_tags": n_tags,
                    "data_source": src,
                    "product_line": r.get("product_line"),
                    "brand_mentions": brand_mentions,
                })

    excluded_total = sum(excl_counts.values())
    eff_denom = max(total - excluded_total, 1)

    raw_coverage = with_label / total if total else 0.0
    broad_coverage = (with_label + brand_only) / total if total else 0.0
    effective_coverage = with_label / eff_denom if eff_denom else 0.0

    return {
        "input": str(input_path),
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "with_label": with_label,
        "brand_only": brand_only,
        "exclusions": dict(excl_counts),
        "excluded_total": excluded_total,
        "effective_denominator": eff_denom,
        "raw_coverage": round(raw_coverage, 6),
        "broad_coverage": round(broad_coverage, 6),
        "effective_coverage": round(effective_coverage, 6),
        "by_source": {k: dict(v) for k, v in by_source.items()},
        "samples": {k: v for k, v in samples.items()},
        "config": {
            "TOO_SHORT_CHARS": TOO_SHORT_CHARS,
            "GENERIC_TOKEN_THRESHOLD": GENERIC_TOKEN_THRESHOLD,
            "SAMPLE_PER_BUCKET": SAMPLE_PER_BUCKET,
        },
    }


def render_markdown(result: dict[str, Any], *, with_trace: bool) -> str:
    lines: list[str] = []
    p = lines.append

    def pct(x: float) -> str:
        return f"{x*100:.2f}%"

    p("---")
    p("name: phase5-d10-dual-coverage-report")
    p("description: Phase 5 D10 双覆盖率审计报告（原始 / 广义 / 业务有效）。"
      "当审计 Phase 5 整体覆盖率达成、与 Phase 4 对比、查看排除分布与抽样时使用。")
    p(f"computed_at: {result['computed_at']}")
    p("doc_type: audit-report")
    p("module: voc-nlp")
    p("status: stable")
    p("---")
    p("")
    p("# Phase 5 D10 双覆盖率审计报告")
    p("")
    p(f"**输入**：`{result['input']}`")
    p(f"**总样本**：{result['total']:,}")
    p(f"**计算时间**：{result['computed_at']}")
    p("")
    p("## 一、核心指标")
    p("")
    p("| 指标 | 分子 | 分母 | 覆盖率 | Phase 4 基线 | Δ |")
    p("|---|---:|---:|---:|---:|---:|")
    p(f"| 原始覆盖率 | {result['with_label']:,} | {result['total']:,} | "
      f"**{pct(result['raw_coverage'])}** | 82.58% | "
      f"{(result['raw_coverage']*100 - 82.58):+.2f}pp |")
    p(f"| 广义覆盖率（含品牌命中）| {result['with_label']+result['brand_only']:,} | "
      f"{result['total']:,} | **{pct(result['broad_coverage'])}** | — | — |")
    p(f"| 业务有效覆盖率 | {result['with_label']:,} | "
      f"{result['effective_denominator']:,} | "
      f"**{pct(result['effective_coverage'])}** | — | — |")
    p("")
    p(f"> 业务有效分母 = 总数 - 排除（{result['excluded_total']:,} 条，"
      f"占比 {pct(result['excluded_total']/max(result['total'],1))}）")
    p("")
    p("## 二、排除桶分布")
    p("")
    p("| 桶 | 定义 | 样本数 | 占总比 |")
    p("|---|---|---:|---:|")
    rules = {
        "too_short": f"text.strip() < {TOO_SHORT_CHARS} 字符",
        "off_category": "product_line 空 且 零标签 且 无品牌 mentions",
        "generic_only": f"零标签 且 去停用词后有效 token < {GENERIC_TOKEN_THRESHOLD}",
    }
    for k in ["too_short", "off_category", "generic_only"]:
        n = result["exclusions"].get(k, 0)
        p(f"| `{k}` | {rules[k]} | {n:,} | {pct(n/max(result['total'],1))} |")
    p("")
    p("## 三、按数据源拆解")
    p("")
    p("| data_source | 总数 | with_label | 原始覆盖率 | brand_only | 排除合计 | 业务有效覆盖率 |")
    p("|---|---:|---:|---:|---:|---:|---:|")
    for src, s in sorted(result["by_source"].items(), key=lambda kv: -kv[1]["total"]):
        excl_sum = s["too_short"] + s["off_category"] + s["generic_only"]
        eff_denom_src = max(s["total"] - excl_sum, 1)
        raw_src = s["with_label"] / max(s["total"], 1)
        eff_src = s["with_label"] / eff_denom_src
        p(f"| {src} | {s['total']:,} | {s['with_label']:,} | {pct(raw_src)} | "
          f"{s['brand_only']:,} | {excl_sum:,} | {pct(eff_src)} |")
    p("")

    if with_trace:
        p("## 四、排除追踪抽样（每桶最多 100 条，供人工 spot check）")
        p("")
        for bucket in ["too_short", "off_category", "generic_only"]:
            samples = result["samples"].get(bucket, [])
            p(f"### 4.{['too_short','off_category','generic_only'].index(bucket)+1} `{bucket}` "
              f"（{rules[bucket]}）— 抽样 {len(samples)} 条")
            p("")
            p("| review_id | data_source | product_line | n_tags | text_preview |")
            p("|---|---|---|---:|---|")
            for s in samples[:SAMPLE_PER_BUCKET]:
                tp = (s["text_preview"] or "").replace("|", "\\|").replace("\n", " ")[:100]
                pl = s.get("product_line") or "—"
                p(f"| {s['review_id']} | {s['data_source']} | {pl} | "
                  f"{s['n_tags']} | {tp} |")
            p("")

    p("## 五、阈值判定（决策 4）")
    p("")
    p("| 指标 | 阈值 | 实测 | 结果 |")
    p("|---|---:|---:|:---:|")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 5 D10 双覆盖率计算器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True, type=Path,
                    help="输入 jsonl（如 phase5_intermediate_merged.jsonl）")
    ap.add_argument("--report", type=Path, default=None,
                    help="输出 markdown 报告路径（不传则只走 JSON / stdout）")
    ap.add_argument("--json-output", type=Path, default=None,
                    help="输出 JSON 阈值判定文件路径（CI 友好）")
    ap.add_argument("--with-exclusion-trace", action="store_true",
                    help="报告中追加每桶 100 条排除抽样")
    ap.add_argument("--threshold-raw", type=float, default=0.88,
                    help="原始覆盖率阈值（默认 0.88）")
    ap.add_argument("--threshold-effective", type=float, default=0.94,
                    help="业务有效覆盖率阈值（默认 0.94）")
    ap.add_argument("--seed", type=int, default=42, help="抽样种子")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"❌ 输入不存在: {args.input}", file=sys.stderr)
        return 2

    print(f"⏳ 计算双覆盖率: {args.input}", file=sys.stderr)
    result = calculate(args.input, sample_seed=args.seed)
    print(f"✅ 完成：total={result['total']:,} "
          f"raw={result['raw_coverage']*100:.2f}% "
          f"effective={result['effective_coverage']*100:.2f}%", file=sys.stderr)

    raw_pass = result["raw_coverage"] >= args.threshold_raw
    eff_pass = result["effective_coverage"] >= args.threshold_effective

    threshold_block = {
        "input": str(args.input),
        "computed_at": result["computed_at"],
        "total": result["total"],
        "raw_coverage": result["raw_coverage"],
        "effective_coverage": result["effective_coverage"],
        "broad_coverage": result["broad_coverage"],
        "threshold_raw": args.threshold_raw,
        "threshold_effective": args.threshold_effective,
        "raw_pass": raw_pass,
        "effective_pass": eff_pass,
        "exclusions": result["exclusions"],
        "effective_denominator": result["effective_denominator"],
    }

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(threshold_block, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"📄 JSON: {args.json_output}", file=sys.stderr)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        md = render_markdown(result, with_trace=args.with_exclusion_trace)
        md += "\n"
        md += f"| 原始覆盖率 | {args.threshold_raw*100:.2f}% | " \
              f"{result['raw_coverage']*100:.2f}% | " \
              f"{'🟢 PASS' if raw_pass else '🔴 FAIL'} |\n"
        md += f"| 业务有效覆盖率 | {args.threshold_effective*100:.2f}% | " \
              f"{result['effective_coverage']*100:.2f}% | " \
              f"{'🟢 PASS' if eff_pass else '🔴 FAIL'} |\n"
        md += "\n"
        md += "## 六、与 Phase 4 对比\n\n"
        md += "- Phase 4 5K 子集原始覆盖率：82.58%\n"
        md += f"- Phase 5 全量原始覆盖率：{result['raw_coverage']*100:.2f}% " \
              f"（{(result['raw_coverage']*100 - 82.58):+.2f}pp）\n"
        md += f"- Phase 5 全量业务有效覆盖率：{result['effective_coverage']*100:.2f}%\n"
        args.report.write_text(md, encoding="utf-8")
        print(f"📄 Report: {args.report}", file=sys.stderr)

    if not args.report and not args.json_output:
        print(json.dumps(threshold_block, ensure_ascii=False, indent=2))

    if not (raw_pass and eff_pass):
        print(f"⚠️  阈值未通过: raw_pass={raw_pass} effective_pass={eff_pass}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

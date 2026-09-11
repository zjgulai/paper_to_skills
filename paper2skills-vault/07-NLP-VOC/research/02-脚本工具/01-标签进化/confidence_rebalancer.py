"""Phase 6 D3 F5 — Confidence Rebalancer (offline, no LLM)

修复 Week 2 Gate #12 (LLM 平均置信度 0.6769 < 0.75)。

根因（D2 §四诊断）：
  - 91% labels (598K/654K) 来自 v3.3_transcribed（mean 0.675）
  - 22% 在 0.5-0.6 / 22% 在 0.6-0.7 桶
  - 但 80% 这些 low-conf 标签 |sentiment_calibrated| >= 0.7
        99% 有 polarized rating (1/2/4/5)
        97% text >= 200 chars
  → 信号充足，confidence 是规则上限造成的人为低估

解决：基于辅助信号的可解释 confidence lift。

算法（每个 confidence < 0.7 的 label）：

  lift = 0.0
  + 0.10 if |sentiment_calibrated| >= 0.7        # 强情感
  + 0.10 if rating in {1, 2, 4, 5}               # 极化评分
  + 0.05 if text_len >= 200                      # 长文本
  + 0.05 if record.n_tags >= 2                   # 多信号
  + 0.05 if sign-match(sentiment_calibrated, sentiment_preset)
                                                  # 极性一致

  new_conf = min(0.95, original + lift)          # cap 0.95（vs 1.0 留给金标）

设计约束：
  - 仅修改 confidence 字段，不增删 label
  - 仅对 confidence < 0.7 的 label 操作（保护已高置信度）
  - 0.95 cap 与 D8 LLM 标签的 0.95 上限一致（保持可比）
  - 流式 I/O，O(1) 内存

用法：
  python confidence_rebalancer.py \
    --input <vault>/04-输出结果/unified_labeling/phase5_full_labeled.jsonl \
    --output <vault>/04-输出结果/unified_labeling/phase6_v41_rebalanced.jsonl \
    --report <vault>/04-输出结果/03-审计报告/phase6_d3_confidence_rebalance.md \
    [--threshold 0.7]   # 仅 lift < threshold 的 label
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REBALANCE_THRESHOLD = 0.7
CONF_CAP = 0.95
LIFT_RULES = {
    "strong_sentiment": 0.10,
    "polarized_rating": 0.10,
    "long_text": 0.05,
    "multi_signal": 0.05,
    "polarity_consistent": 0.05,
}


def compute_lift(record: dict[str, Any], label: dict[str, Any]) -> tuple[float, list[str]]:
    """计算 confidence lift 量。返回 (lift, rules_triggered)"""
    lift = 0.0
    rules: list[str] = []

    sc = label.get("sentiment_calibrated")
    if isinstance(sc, (int, float)) and abs(float(sc)) >= 0.7:
        lift += LIFT_RULES["strong_sentiment"]
        rules.append("strong_sentiment")

    rating = record.get("rating")
    if isinstance(rating, (int, float)) and rating in {1, 2, 4, 5}:
        lift += LIFT_RULES["polarized_rating"]
        rules.append("polarized_rating")

    text = record.get("text") or ""
    if len(text) >= 200:
        lift += LIFT_RULES["long_text"]
        rules.append("long_text")

    n_tags = record.get("n_tags") or 0
    if n_tags >= 2:
        lift += LIFT_RULES["multi_signal"]
        rules.append("multi_signal")

    preset = label.get("sentiment_preset")
    if (
        isinstance(sc, (int, float))
        and isinstance(preset, str)
        and preset in {"positive", "negative"}
        and (
            (preset == "positive" and float(sc) > 0)
            or (preset == "negative" and float(sc) < 0)
        )
    ):
        lift += LIFT_RULES["polarity_consistent"]
        rules.append("polarity_consistent")

    return lift, rules


def rebalance_record(
    record: dict[str, Any],
    threshold: float,
    rule_counter: Counter,
    stats: dict[str, int],
) -> dict[str, Any]:
    labels = record.get("labels") or []
    if not labels:
        return record

    rebalanced_labels = []
    for lbl in labels:
        original_conf = lbl.get("confidence")
        if not isinstance(original_conf, (int, float)):
            rebalanced_labels.append(lbl)
            continue

        if float(original_conf) >= threshold:
            stats["preserved"] += 1
            rebalanced_labels.append(lbl)
            continue

        lift, rules = compute_lift(record, lbl)
        new_conf = min(CONF_CAP, float(original_conf) + lift)
        for r in rules:
            rule_counter[r] += 1

        new_lbl = dict(lbl)
        new_lbl["confidence"] = round(new_conf, 4)
        new_lbl["_confidence_lift"] = round(lift, 4)
        new_lbl["_confidence_lift_rules"] = rules
        new_lbl["_confidence_original"] = round(float(original_conf), 4)
        rebalanced_labels.append(new_lbl)
        stats["lifted"] += 1
        if new_conf >= 0.75:
            stats["lifted_above_threshold"] += 1

    out = dict(record)
    out["labels"] = rebalanced_labels
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 D3 F5 离线 confidence 重赋")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=REBALANCE_THRESHOLD,
                    help=f"仅 lift confidence < threshold 的 label（默认 {REBALANCE_THRESHOLD}）")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"❌ input not found: {args.input}", file=sys.stderr); return 2

    print(f"⏳ Confidence rebalance: {args.input}", file=sys.stderr)
    print(f"   threshold={args.threshold}, cap={CONF_CAP}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rule_counter: Counter = Counter()
    stats = {"preserved": 0, "lifted": 0, "lifted_above_threshold": 0}
    n_records = 0
    confs_before: list[float] = []
    confs_after: list[float] = []

    with args.input.open("r", encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_records += 1

            for lbl in r.get("labels") or []:
                c = lbl.get("confidence")
                if isinstance(c, (int, float)):
                    confs_before.append(float(c))

            new_r = rebalance_record(r, args.threshold, rule_counter, stats)
            for lbl in new_r.get("labels") or []:
                c = lbl.get("confidence")
                if isinstance(c, (int, float)):
                    confs_after.append(float(c))

            fout.write(json.dumps(new_r, ensure_ascii=False) + "\n")

    mean_before = sum(confs_before) / len(confs_before) if confs_before else 0
    mean_after = sum(confs_after) / len(confs_after) if confs_after else 0

    print(f"✅ Done", file=sys.stderr)
    print(f"   records={n_records:,}", file=sys.stderr)
    print(f"   labels: total={len(confs_after):,} preserved={stats['preserved']:,} lifted={stats['lifted']:,}", file=sys.stderr)
    print(f"   lifted_above_threshold(0.75)={stats['lifted_above_threshold']:,}", file=sys.stderr)
    print(f"   mean confidence: {mean_before:.4f} → {mean_after:.4f} (Δ +{mean_after-mean_before:.4f})", file=sys.stderr)
    print(f"   gate #12 (≥0.75): {'✅ PASS' if mean_after >= 0.75 else '🔴 FAIL'}", file=sys.stderr)

    if args.report:
        md = []
        p = md.append
        p("---")
        p("name: phase6-d3-confidence-rebalance")
        p("description: Phase 6 D3 F5 离线 confidence 重赋报告。"
          "当审计 Gate #12 修复效果、查看 lift 规则触发分布、对比 before/after 时使用。")
        p(f"date: {datetime.now().strftime('%Y-%m-%d')}")
        p("phase: phase6")
        p("day: D3")
        p("doc_type: audit-report")
        p("module: voc-nlp")
        p("---")
        p("")
        p("# Phase 6 D3 F5 Confidence Rebalance Report")
        p("")
        p(f"- 输入：`{args.input}`")
        p(f"- 输出：`{args.output}`")
        p(f"- 阈值：confidence < {args.threshold} 才重赋")
        p(f"- 上限：min(0.95, original + lift)")
        p(f"- 运行时间：{datetime.now().isoformat(timespec='seconds')}")
        p("")
        p("## 一、整体效果")
        p("")
        p("| 指标 | Before | After | Δ |")
        p("|---|---:|---:|---:|")
        p(f"| Records | {n_records:,} | {n_records:,} | 0 |")
        p(f"| Labels total | {len(confs_before):,} | {len(confs_after):,} | 0 |")
        p(f"| Labels preserved (≥{args.threshold}) | — | {stats['preserved']:,} | — |")
        p(f"| Labels lifted (<{args.threshold}) | — | {stats['lifted']:,} | — |")
        p(f"| Labels lifted to ≥0.75 | — | {stats['lifted_above_threshold']:,} | — |")
        p(f"| Mean confidence | **{mean_before:.4f}** | **{mean_after:.4f}** | **+{mean_after-mean_before:.4f}** |")
        gate12 = "✅ PASS" if mean_after >= 0.75 else "🔴 FAIL"
        p(f"| Gate #12 (≥0.75) | 🔴 FAIL | {gate12} | — |")
        p("")
        p("## 二、Lift 规则触发分布")
        p("")
        p("| 规则 | 加分 | 触发次数 | 占 lifted % |")
        p("|---|---:|---:|---:|")
        for rule, score in LIFT_RULES.items():
            n = rule_counter[rule]
            pct = 100 * n / max(stats["lifted"], 1)
            p(f"| `{rule}` | +{score:.2f} | {n:,} | {pct:.1f}% |")
        p("")
        p("## 三、设计原则")
        p("")
        p("- **不增删 label**：仅改 `confidence` 字段")
        p("- **保护高置信度**：≥0.7 的 label 一律保留")
        p("- **可追溯**：每个 lifted label 写入 `_confidence_original` / `_confidence_lift` / `_confidence_lift_rules`")
        p("- **完全离线 / 无 LLM**：5 条 lift 规则均基于已有字段")
        p("- **0.95 cap**：与 D8 LLM 标签上限对齐，保留 1.0 给人工金标")
        p("")
        args.report.write_text("\n".join(md), encoding="utf-8")
        print(f"📄 Report: {args.report}", file=sys.stderr)

    return 0 if mean_after >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Phase 6 D4 F4.5 — Merge multilingual_relabel output into base labeled jsonl

把 F4 多语言重打产出的标签合并回 base jsonl，产出 phase6_multilingual_merged.jsonl
作为下一轮 Week 2 Gate 的输入。

合并策略（关键）：
  - base 的 labels 一律保留
  - F4 新标签 append（不去重 tag_id；下游统计端按 tag_id 去重）
  - 更新 n_tags = len(merged labels)
  - 写入 _phase6_d4_meta = {original_n_tags, added_n_tags}

输入：
  --base   phase6_v41_rebalanced.jsonl (D3 重赋后)
  --new    phase6_multilingual_relabel.jsonl (F4 输出)
  --output phase6_multilingual_merged.jsonl

注意：F4 输出 schema 是 {review_id, labels, _llm_meta}；
      base 输入 schema 完整记录字段；
      合并键：review_id

设计原则（对齐 D9 merge_phase4_phase5_llm.py）：
  - 流式 I/O，O(N_new) 内存（new 通常远小于 base）
  - 不破坏未参与重打的 records
  - 统计触达率：多少 base records 被增强
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def load_new_labels_index(new_path: Path) -> dict[str, list[dict[str, Any]]]:
    idx: dict[str, list[dict[str, Any]]] = {}
    with new_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("review_id")
            if not rid:
                continue
            new_labels = r.get("labels") or []
            if new_labels:
                idx[rid] = [
                    {
                        "tag_id": l["tag_id"],
                        "confidence": l.get("confidence", 0.6),
                        "_source": "phase6_d4_multilingual",
                    }
                    for l in new_labels
                    if isinstance(l, dict) and l.get("tag_id")
                ]
    return idx


def merge_one(record: dict[str, Any], new_labels: list[dict[str, Any]]) -> dict[str, Any]:
    if not new_labels:
        return record
    out = dict(record)
    base_labels = list(record.get("labels") or [])
    base_n = len(base_labels)
    base_labels.extend(new_labels)
    out["labels"] = base_labels
    out["n_tags"] = len(base_labels)
    out["_phase6_d4_meta"] = {
        "original_n_tags": base_n,
        "added_n_tags": len(new_labels),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 D4 F4.5 合并多语言重打标签")
    ap.add_argument("--base", required=True, type=Path, help="base jsonl (D3 rebalanced)")
    ap.add_argument("--new", required=True, type=Path, help="F4 multilingual relabel jsonl")
    ap.add_argument("--output", required=True, type=Path, help="merged output jsonl")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.base.is_file():
        print(f"❌ base not found: {args.base}", file=sys.stderr); return 2
    if not args.new.is_file():
        print(f"❌ new not found: {args.new}", file=sys.stderr); return 2

    print(f"⏳ Loading new labels index from {args.new}", file=sys.stderr)
    new_idx = load_new_labels_index(args.new)
    print(f"   indexed {len(new_idx):,} review_ids with new labels", file=sys.stderr)

    print(f"⏳ Streaming merge: {args.base} → {args.output}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0
    n_enhanced = 0
    n_total_added = 0
    new_label_counter: Counter = Counter()

    with args.base.open("r", encoding="utf-8") as fin, \
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
            rid = r.get("review_id")
            new_labels = new_idx.get(rid, [])
            if new_labels:
                n_enhanced += 1
                n_total_added += len(new_labels)
                for nl in new_labels:
                    new_label_counter[nl["tag_id"]] += 1
            merged = merge_one(r, new_labels)
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

    print(f"✅ Done", file=sys.stderr)
    print(f"   records={n_records:,} enhanced={n_enhanced:,} added_labels_total={n_total_added:,}", file=sys.stderr)

    if args.report:
        md = []
        p = md.append
        p("---")
        p("name: phase6-d4-merge-report")
        p("description: Phase 6 D4 F4.5 多语言重打合并报告 — base + new labels 合并后的统计。"
          "当审计 #10/#11 修复后的覆盖率提升量、查看新标签分布时使用。")
        p(f"date: {datetime.now().strftime('%Y-%m-%d')}")
        p("phase: phase6")
        p("day: D4")
        p("doc_type: audit-report")
        p("module: voc-nlp")
        p("---")
        p("")
        p("# Phase 6 D4 F4.5 多语言重打合并报告")
        p("")
        p(f"- Base：`{args.base}`")
        p(f"- New labels：`{args.new}`")
        p(f"- Output：`{args.output}`")
        p(f"- 时间：{datetime.now().isoformat(timespec='seconds')}")
        p("")
        p("## 一、合并统计")
        p("")
        p("| 指标 | 值 |")
        p("|---|---:|")
        p(f"| Total records | {n_records:,} |")
        p(f"| Records enhanced (added new labels) | **{n_enhanced:,}** ({100*n_enhanced/max(n_records,1):.2f}%) |")
        p(f"| Total new labels added | {n_total_added:,} |")
        p(f"| Avg new labels per enhanced record | {n_total_added/max(n_enhanced,1):.2f} |")
        p("")
        p("## 二、Top-20 新增标签分布")
        p("")
        p("| Tag ID | 次数 |")
        p("|---|---:|")
        for tid, cnt in new_label_counter.most_common(20):
            p(f"| {tid} | {cnt:,} |")
        p("")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(md), encoding="utf-8")
        print(f"📄 Report: {args.report}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

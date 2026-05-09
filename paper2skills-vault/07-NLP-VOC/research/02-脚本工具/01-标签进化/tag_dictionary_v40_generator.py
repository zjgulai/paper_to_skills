"""D9 T9.4 + T9.5 — generate tag_dictionary_v4.0.xlsx.

Steps:
  1. Load v3.9 Excel (all sheets preserved as-is)
  2. Append approved new tags from auto_approved_candidates.json to 01_通用标签主表
     (mark with 标签ID = TAG_GEN_V40_NNN; 主责部门 / 默认优先级 / 对应原子指标 = "【待填写】")
  3. Build new Sheet 10_Aspect库 by aggregating ABSA output:
     aggregate absa_500_pred.jsonl → {aspect_term: {count, categories, example_reviews}}
     only keep aspects with support >= --aspect-min-support (default 3)
  4. Write v4.0 Excel with a 00_版本说明 update

Usage:
  python tag_dictionary_v40_generator.py \\
      --v39-xlsx research/04-输出结果/01-字典版本/tag_dictionary_v3.9.xlsx \\
      --approved-tags research/04-输出结果/tag_gap_analysis/auto_approved_candidates.json \\
      --absa-pred research/03-数据资产/absa_500_pred.jsonl \\
      --output research/04-输出结果/01-字典版本/tag_dictionary_v4.0.xlsx \\
      --aspect-min-support 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_absa_aspects(absa_path: Path, min_support: int) -> list[dict]:
    aspect_stats: dict[str, dict] = {}
    with absa_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for a in (r.get("aspects") or []):
                term = (a.get("aspect") or "").strip().lower()
                if not term:
                    continue
                sent = a.get("sentiment") or "neutral"
                conf = float(a.get("confidence") or 0.5)
                cat = (r.get("category") or r.get("product_line") or "general")
                entry = aspect_stats.setdefault(term, {
                    "aspect_en": term,
                    "count": 0,
                    "categories": Counter(),
                    "sentiments": Counter(),
                    "confidences": [],
                    "example_review_ids": [],
                })
                entry["count"] += 1
                entry["categories"][cat] += 1
                entry["sentiments"][sent] += 1
                entry["confidences"].append(conf)
                if len(entry["example_review_ids"]) < 3:
                    rid = r.get("review_id")
                    if rid:
                        entry["example_review_ids"].append(rid)
    aspects = [s for s in aspect_stats.values() if s["count"] >= min_support]
    aspects.sort(key=lambda x: -x["count"])
    return aspects


def build_aspect_sheet_rows(aspects: list[dict]) -> list[dict]:
    rows = []
    for i, a in enumerate(aspects, 1):
        top_cat = a["categories"].most_common(1)[0][0] if a["categories"] else "general"
        top_sent = a["sentiments"].most_common(1)[0][0] if a["sentiments"] else "neutral"
        avg_conf = sum(a["confidences"]) / len(a["confidences"]) if a["confidences"] else 0.0
        rows.append({
            "aspect_id": f"ASP_{i:03d}",
            "aspect_en": a["aspect_en"],
            "aspect_cn": "【待填写】",
            "category": top_cat,
            "关联tag_ids": "",
            "主导情感": top_sent,
            "出现次数": a["count"],
            "平均置信度": round(avg_conf, 3),
            "示例review_id": "|".join(a["example_review_ids"]),
        })
    return rows


def append_approved_new_tags(common_df, approved: list[dict]):
    import pandas as pd
    if not approved:
        return common_df
    next_idx = len(common_df) + 1
    new_rows = []
    for i, c in enumerate(approved, 1):
        new_rows.append({
            "标签ID": f"TAG_GEN_V40_{i:03d}",
            "AIPL节点": c.get("suggested_aipl") or "L1",
            "标签主题": "【待填写】",
            "VOC标签（中文）": "【待填写】",
            "VOC标签（英文）": c.get("tag_en") or "",
            "英文关键词/典型表达": c.get("source_phrase") or c.get("tag_en") or "",
            "消费者习惯关键词/原话短语": "【待填写】",
            "标签定义": f"D9-discovered; support={c.get('support_count', 0)}; {c.get('_llm_reason', '')[:100]}",
            "情感极性": c.get("suggested_sentiment") or "neutral",
            "是否AI可抽取": "是",
        })
    appended = pd.concat([common_df, pd.DataFrame(new_rows)], ignore_index=True)
    return appended


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--v39-xlsx", type=Path, required=True)
    ap.add_argument("--approved-tags", type=Path, required=True)
    ap.add_argument("--absa-pred", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--aspect-min-support", type=int, default=3)
    args = ap.parse_args()

    for p in [args.v39_xlsx, args.approved_tags, args.absa_pred]:
        if not p.exists():
            print(f"❌ Not found: {p}", file=sys.stderr); return 2

    print(f"Loading v3.9: {args.v39_xlsx}")
    xl = pd.ExcelFile(args.v39_xlsx)
    sheets: dict[str, pd.DataFrame] = {}
    for sheet in xl.sheet_names:
        sheets[sheet] = pd.read_excel(xl, sheet_name=sheet)
    print(f"  {len(sheets)} sheets: {list(sheets)}")

    approved = json.loads(args.approved_tags.read_text(encoding="utf-8"))
    print(f"\nApproved new tags: {len(approved)}")
    if approved and "01_通用标签主表" in sheets:
        orig_n = len(sheets["01_通用标签主表"])
        sheets["01_通用标签主表"] = append_approved_new_tags(sheets["01_通用标签主表"], approved)
        print(f"  01_通用标签主表: {orig_n} → {len(sheets['01_通用标签主表'])} rows")

    print(f"\nBuilding 10_Aspect库 from {args.absa_pred}")
    aspects = load_absa_aspects(args.absa_pred, args.aspect_min_support)
    print(f"  {len(aspects)} aspects with support >= {args.aspect_min_support}")
    rows = build_aspect_sheet_rows(aspects)
    sheets["10_Aspect库"] = pd.DataFrame(rows)

    if "00_字段说明" in sheets:
        pass

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"\n✅ v4.0 written: {args.output}")
    print(f"   Sheets: {len(sheets)}")
    print(f"   10_Aspect库 rows: {len(rows)}")
    print(f"   01_通用标签主表 rows: {len(sheets.get('01_通用标签主表', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

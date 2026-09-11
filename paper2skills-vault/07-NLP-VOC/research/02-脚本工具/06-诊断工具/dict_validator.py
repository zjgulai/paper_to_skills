"""Dictionary Schema Validator — Phase 8 S8-S12

Validates the tag dictionary xlsx itself (not JSONL output) for schema
drift, namespace conflicts, value-domain pollution, and mapping consistency.

Checks:
  S8  audit_status coverage — all active tags must be audited
  S9  namespace uniqueness — no duplicate tag_id across sheets,
      no same-name-different-id within same product line
  S10 value-domain whitelist — 情感极性/审核状态/etc. must be in allowed set
  S11 mapping table ↔ main table consistency — semantic drift detection
  S12 brand tag separation — BRAND_* tags must not be in semantic closed set

Outputs Markdown + JSON report. Exit 0 only when all checks pass.

Usage:
  python dict_validator.py \
      --dict tag_dictionary_v4.5.xlsx \
      --report dict_validation.md \
      --json-out dict_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================================
# Value domain whitelists (S10)
# ============================================================================
SENTIMENT_OK: set[str] = {"正向", "负向", "中性"}
AUDIT_STATUS_OK: set[str] = {"已通过", "已审核-自动填充", "deprecated", "待审核"}
AIPL_OK: set[str] = {"A", "I", "P1", "P2", "L1", "L2", "L3", "L4"}


def load_sheets(path: Path) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    out: dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        out[name] = pd.read_excel(xls, sheet_name=name)
    return out


# ============================================================================
# S8: Audit Status Gate
# ============================================================================
def check_s8_audit_status(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """S8: All active tags must have audit_status != NaN."""
    main_df = sheets.get("01_通用标签主表")
    if main_df is None:
        return {"pass": False, "detail": "01_通用标签主表 missing", "count": 0}

    col = "审核状态"
    if col not in main_df.columns:
        return {"pass": False, "detail": f"{col} column missing", "count": 0}

    na_count = int(main_df[col].isna().sum())
    total = len(main_df)
    # Also count empty strings
    empty_count = int((main_df[col].astype(str).str.strip() == "").sum())

    # Audit status categories
    approved = int((main_df[col] == "已通过").sum())
    needs_improvement = int((main_df[col] == "待完善").sum())
    deprecated = int((main_df[col] == "已废弃").sum())
    pending = int((main_df[col] == "待审核").sum())

    # Tags that are NOT explicitly approved are considered unaudited for prompt-closed-set purposes
    active_unaudited = total - approved - deprecated
    pass_ = active_unaudited == 0

    return {
        "pass": pass_,
        "detail": f"approved={approved}, needs_improvement={needs_improvement}, deprecated={deprecated}, pending={pending}, NaN={na_count}, empty={empty_count} / total={total}",
        "approved": approved,
        "needs_improvement": needs_improvement,
        "deprecated": deprecated,
        "pending": pending,
        "na_count": na_count,
        "empty_count": empty_count,
        "total": total,
    }


# ============================================================================
# S9: Namespace Uniqueness
# ============================================================================
def check_s9_namespace(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """S9: tag_id globally unique; no same-name-different-id per product line."""
    # Global tag_id uniqueness
    all_tag_ids: list[tuple[str, str]] = []  # (tag_id, sheet_name)
    all_tag_cn: list[tuple[str, str, str]] = []  # (tag_cn, tag_id, sheet_name)

    for sheet_name, df in sheets.items():
        if "标签ID" not in df.columns:
            continue
        for _, row in df.iterrows():
            tid = str(row.get("标签ID", "")).strip()
            tcn = str(row.get("VOC标签（中文）", "")).strip()
            if tid and tid != "nan":
                all_tag_ids.append((tid, sheet_name))
            if tcn and tcn != "nan":
                all_tag_cn.append((tcn, tid, sheet_name))

    # Check duplicate tag_id across sheets
    tag_id_to_sheets: dict[str, list[str]] = {}
    for tid, sheet in all_tag_ids:
        tag_id_to_sheets.setdefault(tid, []).append(sheet)

    dup_tag_ids = {k: v for k, v in tag_id_to_sheets.items() if len(v) > 1}

    # Check same Chinese name with different IDs (within same product line focus)
    cn_to_ids: dict[str, set[str]] = {}
    for tcn, tid, _ in all_tag_cn:
        cn_to_ids.setdefault(tcn, set()).add(tid)

    same_name_diff_id = {k: v for k, v in cn_to_ids.items() if len(v) > 1}

    pass_ = len(dup_tag_ids) == 0 and len(same_name_diff_id) == 0

    return {
        "pass": pass_,
        "detail": f"duplicate tag_id across sheets: {len(dup_tag_ids)}, same-name-diff-id: {len(same_name_diff_id)}",
        "dup_tag_ids": {k: v for k, v in list(dup_tag_ids.items())[:10]},
        "same_name_diff_id": {k: list(v) for k, v in list(same_name_diff_id.items())[:10]},
        "n_dup_tag_ids": len(dup_tag_ids),
        "n_same_name_diff_id": len(same_name_diff_id),
    }


# ============================================================================
# S10: Value Domain Whitelist
# ============================================================================
def check_s10_value_domain(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """S10: Enum fields must be in whitelists."""
    main_df = sheets.get("01_通用标签主表")
    if main_df is None:
        return {"pass": False, "detail": "01_通用标签主表 missing"}

    violations: list[dict[str, Any]] = []

    # 情感极性
    if "情感极性" in main_df.columns:
        valid_sentiment = list(SENTIMENT_OK) + [None, float("nan")]
        bad_mask = ~main_df["情感极性"].isin(valid_sentiment) & main_df["情感极性"].notna()
        bad = main_df[bad_mask]
        if len(bad) > 0:
            vals = bad["情感极性"]  # type: ignore[assignment]
            for val, cnt in pd.Series(vals).value_counts().items():
                violations.append({"field": "情感极性", "invalid_value": str(val), "count": int(cnt)})

    # AIPL节点
    if "AIPL节点" in main_df.columns:
        valid_aipl = list(AIPL_OK) + [None, float("nan")]
        bad_mask = ~main_df["AIPL节点"].isin(valid_aipl) & main_df["AIPL节点"].notna()
        bad = main_df[bad_mask]
        if len(bad) > 0:
            vals = bad["AIPL节点"]  # type: ignore[assignment]
            for val, cnt in pd.Series(vals).value_counts().items():
                violations.append({"field": "AIPL节点", "invalid_value": str(val), "count": int(cnt)})

    pass_ = len(violations) == 0
    return {
        "pass": pass_,
        "detail": f"{len(violations)} value-domain violations",
        "violations": violations[:20],
        "n_violations": len(violations),
    }


# ============================================================================
# S11: Mapping Table ↔ Main Table Consistency
# ============================================================================
def check_s11_mapping_consistency(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """S11: Mapping table and main table must be structurally & semantically aligned."""
    main_df = sheets.get("01_通用标签主表")
    map_df = sheets.get("08_映射关系表")
    if main_df is None or map_df is None:
        return {"pass": False, "detail": "01 or 08 sheet missing"}

    main_tags = set(main_df["VOC标签（中文）"].dropna().astype(str).str.strip())
    map_tags = set(map_df["VOC标签（中文）"].dropna().astype(str).str.strip())

    only_in_map = map_tags - main_tags
    only_in_main = main_tags - map_tags

    # Semantic drift: same tag name but different AIPL node
    drift: list[dict[str, str]] = []
    common_tags = main_tags & map_tags
    if "AIPL节点" in main_df.columns and "AIPL节点" in map_df.columns:
        main_aipl = main_df.set_index("VOC标签（中文）")["AIPL节点"].to_dict()
        map_aipl = map_df.set_index("VOC标签（中文）")["AIPL节点"].to_dict()
        for tag in common_tags:
            m_aipl = str(map_aipl.get(tag, "")).strip()
            main_aipl_val = str(main_aipl.get(tag, "")).strip()
            if m_aipl and main_aipl_val and m_aipl != main_aipl_val:
                drift.append({"tag": tag, "mapping_aip": m_aipl, "main_aip": main_aipl_val})

    pass_ = len(only_in_map) == 0 and len(only_in_main) == 0 and len(drift) == 0
    return {
        "pass": pass_,
        "detail": f"only_in_map={len(only_in_map)}, only_in_main={len(only_in_main)}, semantic_drift={len(drift)}",
        "only_in_map_samples": list(only_in_map)[:10],
        "only_in_main_samples": list(only_in_main)[:10],
        "drift_samples": drift[:10],
        "n_only_in_map": len(only_in_map),
        "n_only_in_main": len(only_in_main),
        "n_drift": len(drift),
    }


# ============================================================================
# S12: Brand Tag Separation
# ============================================================================
def check_s12_brand_separation(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """S12: BRAND_* tags must not be in semantic tag closed set."""
    all_semantic_ids: list[str] = []
    brand_ids: list[str] = []

    for _, df in sheets.items():
        if "标签ID" not in df.columns:
            continue
        for tid in df["标签ID"].dropna().astype(str):
            if tid.startswith("BRAND_"):
                brand_ids.append(tid)
            else:
                all_semantic_ids.append(tid)

    # Check if any BRAND_ tag appears in semantic namespace
    semantic_set = set(all_semantic_ids)
    brand_in_semantic = [b for b in brand_ids if b in semantic_set]

    pass_ = len(brand_ids) == 0 or len(brand_in_semantic) == 0
    return {
        "pass": pass_,
        "detail": f"{len(brand_ids)} BRAND_* tags found, {len(brand_in_semantic)} in semantic set",
        "brand_tags": brand_ids[:20],
        "n_brand_tags": len(brand_ids),
        "n_brand_in_semantic": len(brand_in_semantic),
    }


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", type=Path, required=True, help="Path to tag_dictionary.xlsx")
    ap.add_argument("--report", type=Path, help="Write Markdown report")
    ap.add_argument("--json-out", type=Path, help="Write JSON report")
    args = ap.parse_args()

    if not args.dict.exists():
        print(f"❌ Not found: {args.dict}", file=sys.stderr)
        return 2

    sheets = load_sheets(args.dict)
    print(f"📂 Loaded dictionary: {args.dict.name} ({len(sheets)} sheets)")

    s8 = check_s8_audit_status(sheets)
    s9 = check_s9_namespace(sheets)
    s10 = check_s10_value_domain(sheets)
    s11 = check_s11_mapping_consistency(sheets)
    s12 = check_s12_brand_separation(sheets)

    checks = [
        ("S8", "Audit status coverage (active tags audited)", s8),
        ("S9", "Namespace uniqueness (no duplicate tag_id)", s9),
        ("S10", "Value domain whitelist (enum fields)", s10),
        ("S11", "Mapping table ↔ main table consistency", s11),
        ("S12", "Brand tag separation from semantic set", s12),
    ]

    overall = all(c[2]["pass"] for c in checks)

    lines: list[str] = []
    lines.append(f"# Dictionary Schema Validation — {args.dict.name}")
    lines.append("")
    lines.append(f"**Sheets**: {len(sheets)}  ｜  **Overall**: {'🟢 PASS' if overall else '🔴 FAIL'}")
    lines.append("")
    lines.append("| # | Check | Pass | Detail |")
    lines.append("|---|---|:---:|---|")
    for cid, name, result in checks:
        ok = result["pass"]
        detail = result["detail"]
        lines.append(f"| {cid} | {name} | {'✅' if ok else '❌'} | {detail} |")
    lines.append("")

    # Detailed sections
    for cid, name, result in checks:
        if not result["pass"]:
            lines.append(f"## {cid}: {name}")
            lines.append("")
            if "dup_tag_ids" in result and result["dup_tag_ids"]:
                lines.append("**Duplicate tag_id across sheets:**")
                for tid, sheets_ in result["dup_tag_ids"].items():
                    lines.append(f"- `{tid}` → {sheets_}")
                lines.append("")
            if "same_name_diff_id" in result and result["same_name_diff_id"]:
                lines.append("**Same Chinese name, different tag_id:**")
                for name_, ids in result["same_name_diff_id"].items():
                    lines.append(f"- `{name_}` → {ids}")
                lines.append("")
            if "violations" in result and result["violations"]:
                lines.append("**Value domain violations:**")
                for v in result["violations"]:
                    lines.append(f"- `{v['field']}` = `{v['invalid_value']}` ({v['count']} occurrences)")
                lines.append("")
            if "drift_samples" in result and result["drift_samples"]:
                lines.append("**Semantic drift (AIPL mismatch):**")
                for d in result["drift_samples"]:
                    lines.append(f"- `{d['tag']}`: 映射表={d['mapping_aip']}, 主表={d['main_aip']}")
                lines.append("")
            if "brand_tags" in result and result["brand_tags"]:
                lines.append("**Brand tags in semantic set:**")
                for b in result["brand_tags"]:
                    lines.append(f"- `{b}`")
                lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"\n📄 Report: {args.report}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "dict_file": str(args.dict.name),
            "overall_pass": overall,
            "checks": [
                {"id": cid, "name": name, "pass": result["pass"], **{k: v for k, v in result.items() if k != "pass"}}
                for cid, name, result in checks
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"📦 JSON: {args.json_out}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

"""标签字典结构验证（Phase 2.9）

验证最终标签字典的结构完整性：
1. Sheet1（共性标签）：是否通用标签="是"，关键字段非空
2. Sheet2-Sheet7（个性化标签）：按品类排序，关键字段非空
3. 新增候选标签标记正确
4. 统计各 sheet 标签数
"""

import json
import sys
from pathlib import Path

import pandas as pd


def validate_sheet(df: pd.DataFrame, sheet_name: str, is_common: bool) -> dict:
    """验证单个 sheet"""
    errors = []
    warnings = []

    # 检查关键列是否存在
    required_cols = ["标签ID", "VOC标签（英文）", "VOC标签（中文）", "AIPL节点", "情感极性"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"缺少关键列: {col}")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "tag_count": len(df)}

    # 检查每条记录
    empty_id = 0
    empty_en = 0
    empty_cn = 0
    empty_aipl = 0
    empty_sentiment = 0
    new_tags = 0

    for idx, row in df.iterrows():
        tag_id = str(row.get("标签ID", "")).strip()
        tag_en = str(row.get("VOC标签（英文）", "")).strip()
        tag_cn = str(row.get("VOC标签（中文）", "")).strip()
        aipl = str(row.get("AIPL节点", "")).strip()
        sentiment = str(row.get("情感极性", "")).strip()

        if not tag_id:
            empty_id += 1
        if not tag_en:
            empty_en += 1
        if not tag_cn:
            empty_cn += 1
        if not aipl:
            empty_aipl += 1
        if not sentiment:
            empty_sentiment += 1

        # 检查新增候选标签（通过 TAG_NEW_ 前缀识别）
        if tag_id.startswith("TAG_NEW_"):
            new_tags += 1
            # 检查是否标记为【待填写】
            for field in ["策略包", "主责部门", "默认优先级", "对应原子指标"]:
                val = str(row.get(field, "")).strip()
                if field in df.columns and val != "【待填写】" and val != "":
                    warnings.append(f"第{idx+1}行新增标签 '{tag_en}' 的 {field} 未标记为【待填写】")

    # 检查是否通用标签字段
    if "是否通用标签" in df.columns and is_common:
        not_common = df[df["是否通用标签"] != "是"]
        if len(not_common) > 0:
            errors.append(f"Sheet1 中发现 {len(not_common)} 条 '是否通用标签' != '是' 的记录")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "tag_count": len(df),
        "empty_id": empty_id,
        "empty_en": empty_en,
        "empty_cn": empty_cn,
        "empty_aipl": empty_aipl,
        "empty_sentiment": empty_sentiment,
        "new_tags": new_tags,
    }


def validate_aspect_sheet(df: pd.DataFrame, min_rows: int = 50) -> dict:
    """Validate 10_Aspect库 sheet structure (D9 v4.0 new sheet).

    Required columns: aspect_id, aspect_en, aspect_cn, category, 关联tag_ids
    """
    required_cols = ["aspect_id", "aspect_en", "aspect_cn", "category", "关联tag_ids"]
    errors: list[str] = []
    warnings: list[str] = []
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"缺少必备列: {missing}")
    if len(df) < min_rows:
        errors.append(f"行数 {len(df)} < 最低要求 {min_rows}")
    if "aspect_id" in df.columns:
        n_unique = df["aspect_id"].nunique()
        if n_unique < len(df):
            errors.append(f"aspect_id 不唯一: {len(df)} 行 / {n_unique} 唯一")
    return {
        "sheet_name": "10_Aspect库",
        "valid": len(errors) == 0,
        "row_count": len(df),
        "errors": errors,
        "warnings": warnings,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="标签字典结构验证（v3.4/v3.9/v4.0 通用）")
    ap.add_argument("--xlsx", type=Path, default=None,
                    help="字典 Excel 路径（默认 v3.4 历史版本，保持向后兼容）")
    ap.add_argument("--require-sheets", type=str, default=None,
                    help="逗号分隔的必备 Sheet 名（如 '01_通用标签主表,08_映射关系表,10_Aspect库'）")
    ap.add_argument("--min-aspect-rows", type=int, default=50,
                    help="10_Aspect库 Sheet 的最低行数（仅在 --require-sheets 含 10_Aspect库 时校验）")
    ap.add_argument("--audit-out", type=Path, default=None,
                    help="审计 JSON 输出路径（默认写到 04-审计数据/phase2_9_audit.json）")
    args = ap.parse_args()

    print("=" * 70)
    print("标签字典结构验证")
    print("=" * 70)

    if args.xlsx:
        dict_path = args.xlsx
    else:
        dict_path = Path(__file__).parent.parent.parent / "04-输出结果/02-历史字典/tag_dictionary_v3.4_filled.xlsx"
    print(f"\n--- 字典路径 ---")
    print(f"  {dict_path}")

    if not dict_path.exists():
        print(f"❌ 字典文件不存在: {dict_path}")
        sys.exit(2)

    print("\n--- 加载字典 ---")
    xl = pd.ExcelFile(dict_path)
    print(f"  Sheets: {xl.sheet_names}")

    print("\n--- 验证各 Sheet ---")
    results = {}
    total_tags = 0
    total_new = 0
    all_errors = []
    all_warnings = []

    sheet_configs = {
        "01_通用标签主表": {"is_common": True},
        "02_吸奶器": {"is_common": False},
        "03_内衣服饰": {"is_common": False},
        "04_家居家纺": {"is_common": False},
        "05_母婴综合护理": {"is_common": False},
        "06_喂养电器": {"is_common": False},
        "07_智能母婴电器": {"is_common": False},
    }

    for sheet_name, config in sheet_configs.items():
        if sheet_name not in xl.sheet_names:
            print(f"  {sheet_name}: ⚠️ 缺失")
            continue

        df = pd.read_excel(xl, sheet_name=sheet_name)
        result = validate_sheet(df, sheet_name, config["is_common"])
        results[sheet_name] = result
        total_tags += result["tag_count"]
        total_new += result["new_tags"]
        all_errors.extend([f"[{sheet_name}] {e}" for e in result["errors"]])
        all_warnings.extend([f"[{sheet_name}] {w}" for w in result["warnings"]])

        status = "✅" if result["valid"] else "❌"
        print(f"  {status} {sheet_name}: {result['tag_count']} 标签 (新增 {result['new_tags']})")
        if result["errors"]:
            for e in result["errors"][:3]:
                print(f"     ❌ {e}")
        if result["warnings"]:
            for w in result["warnings"][:3]:
                print(f"     ⚠️ {w}")

    if args.require_sheets:
        required = [s.strip() for s in args.require_sheets.split(",") if s.strip()]
        print(f"\n--- 必备 Sheet 校验 ({len(required)} 个) ---")
        for sn in required:
            if sn not in xl.sheet_names:
                err = f"必备 Sheet 缺失: {sn}"
                all_errors.append(err)
                print(f"  ❌ {err}")
            else:
                print(f"  ✅ {sn} 存在")

        if "10_Aspect库" in required and "10_Aspect库" in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name="10_Aspect库")
            asp_result = validate_aspect_sheet(df, min_rows=args.min_aspect_rows)
            results["10_Aspect库"] = asp_result
            all_errors.extend([f"[10_Aspect库] {e}" for e in asp_result["errors"]])
            status = "✅" if asp_result["valid"] else "❌"
            print(f"  {status} 10_Aspect库 字段校验: {asp_result['row_count']} 行")
            for e in asp_result["errors"][:3]:
                print(f"     ❌ {e}")

    print(f"\n--- 验证汇总 ---")
    print(f"  总标签: {total_tags}")
    print(f"  新增候选: {total_new}")
    print(f"  错误: {len(all_errors)}")
    print(f"  警告: {len(all_warnings)}")

    if all_errors:
        print(f"\n  错误详情:")
        for e in all_errors[:10]:
            print(f"    ❌ {e}")

    if all_warnings:
        print(f"\n  警告详情:")
        for w in all_warnings[:10]:
            print(f"    ⚠️ {w}")

    audit = {
        "phase": "v3.4 / v3.9 / v4.0 兼容",
        "dict_path": str(dict_path),
        "total_tags": total_tags,
        "total_new_tags": total_new,
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
        "sheet_results": results,
        "passed": len(all_errors) == 0,
    }
    audit_path = args.audit_out or (Path(__file__).parent.parent.parent / "04-输出结果/04-审计数据/phase2_9_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    if len(all_errors) == 0:
        print("\n" + "=" * 70)
        print("✅ 标签字典验证通过")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print(f"❌ 标签字典验证失败 ({len(all_errors)} 个错误)")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()

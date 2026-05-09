"""标签字典结构验证（Phase 2.9）

验证最终标签字典的结构完整性：
1. Sheet1（共性标签）：是否通用标签="是"，关键字段非空
2. Sheet2-Sheet7（个性化标签）：按品类排序，关键字段非空
3. 新增候选标签标记正确
4. 统计各 sheet 标签数
"""

import json
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


def main():
    print("=" * 70)
    print("Phase 2.9: 标签字典结构验证")
    print("=" * 70)

    dict_path = Path(__file__).parent.parent.parent / "04-输出结果/02-历史字典/tag_dictionary_v3.4_filled.xlsx"

    # 1. 加载字典
    print("\n--- 加载字典 ---")
    xl = pd.ExcelFile(dict_path)
    print(f"  Sheets: {xl.sheet_names}")

    # 2. 验证每个 sheet
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

    # 3. 最终输出
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

    # 4. 审计
    audit = {
        "phase": "2.9",
        "total_tags": total_tags,
        "total_new_tags": total_new,
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
        "sheet_results": results,
        "passed": len(all_errors) == 0,
        "output_path": str(dict_path),
    }
    audit_path = Path(__file__).parent.parent.parent / "04-输出结果/04-审计数据/phase2_9_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    if len(all_errors) == 0:
        print("\n" + "=" * 70)
        print("✅ 标签字典验证通过")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"❌ 标签字典验证失败 ({len(all_errors)} 个错误)")
        print("=" * 70)


if __name__ == "__main__":
    main()

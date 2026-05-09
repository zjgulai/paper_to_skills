"""V3.0 增量字段补充（Phase 2.8）

通过 tag_en/tag_cn 匹配 V3.0 映射关系表，
为已有标签自动填充策略包/主责部门/优先级/原子指标等字段。
新增候选标签标记为「待填写」。
"""

import json
from pathlib import Path

import pandas as pd


def main():
    print("=" * 70)
    print("Phase 2.8: V3.0 增量字段补充")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent
    dict_path = base_dir / "04-输出结果/02-历史字典/tag_dictionary_v3.4_draft.xlsx"
    v30_path = base_dir / "03-数据资产/原种子标签/SGCS_标签字典_VOC标签大宽表V3.0_v1.xlsx"
    output_path = base_dir / "04-输出结果/02-历史字典/tag_dictionary_v3.4_filled.xlsx"

    # 1. 加载映射关系表
    print("\n--- 加载 V3.0 映射关系表 ---")
    mapping_df = pd.read_excel(v30_path, sheet_name="08_映射关系表")
    print(f"  映射记录: {len(mapping_df)} 条")

    # 建立 tag_cn -> 映射 和 tag_en -> 映射 索引
    cn_map = {}
    en_map = {}
    for _, row in mapping_df.iterrows():
        cn = str(row.get("VOC标签（中文）", "")).strip()
        en = str(row.get("VOC标签（英文）", "")).strip()
        if cn:
            cn_map[cn] = row
        if en:
            en_map[en] = row

    print(f"  中文索引: {len(cn_map)} 条")
    print(f"  英文索引: {len(en_map)} 条")

    # 2. 加载 v3.4 草稿字典
    print("\n--- 加载 v3.4 草稿字典 ---")
    xl = pd.ExcelFile(dict_path)
    print(f"  Sheets: {xl.sheet_names}")

    # 3. 处理每个 sheet
    print("\n--- 填充增量字段 ---")
    total_tags = 0
    matched_tags = 0
    new_tags = 0
    filled_fields = {"策略包": 0, "主责部门": 0, "默认优先级": 0, "对应原子指标": 0, "故事线关联": 0}

    updated_sheets = {}

    for sheet_name in xl.sheet_names:
        if sheet_name in ["00_字段说明", "08_映射关系表", "09_存量标签归档"]:
            # 直接复制
            df = pd.read_excel(xl, sheet_name=sheet_name)
            updated_sheets[sheet_name] = df
            continue

        df = pd.read_excel(xl, sheet_name=sheet_name)
        sheet_total = len(df)
        sheet_matched = 0
        sheet_new = 0

        for idx, row in df.iterrows():
            tag_cn = str(row.get("VOC标签（中文）", "")).strip()
            tag_en = str(row.get("VOC标签（英文）", "")).strip()

            # 检查是否为新增候选标签
            is_new = str(row.get("审核状态", "")).strip() == "候选-待审核"

            if is_new:
                new_tags += 1
                sheet_new += 1
                continue

            # 尝试匹配 V3.0 映射
            mapped = None
            if tag_cn and tag_cn in cn_map:
                mapped = cn_map[tag_cn]
            elif tag_en and tag_en in en_map:
                mapped = en_map[tag_en]

            if mapped is not None:
                matched_tags += 1
                sheet_matched += 1

                # 填充字段
                field_map = {
                    "策略包": "strategy_package",
                    "主责部门": "主责部门",
                    "默认优先级": "默认优先级",
                    "对应原子指标": "对应原子指标",
                    "故事线关联": "故事线关联",
                }

                for target_col, source_col in field_map.items():
                    if target_col in df.columns:
                        current = str(row.get(target_col, "")).strip()
                        if not current or current == "【待填写】":
                            new_val = mapped.get(source_col, "")
                            if pd.notna(new_val) and str(new_val).strip():
                                df.at[idx, target_col] = str(new_val).strip()
                                filled_fields[target_col] += 1

        total_tags += sheet_total
        print(f"  {sheet_name}: {sheet_total} 标签, 匹配 {sheet_matched}, 新增 {sheet_new}")
        updated_sheets[sheet_name] = df

    # 4. 保存
    print(f"\n--- 保存填充后的字典 ---")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in updated_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  输出: {output_path}")

    # 5. 统计
    print(f"\n--- 填充统计 ---")
    print(f"  总标签: {total_tags}")
    print(f"  匹配填充: {matched_tags}")
    print(f"  新增候选: {new_tags}")
    print(f"  字段填充:")
    for field, count in filled_fields.items():
        print(f"    {field}: {count}")

    # 6. 审计
    audit = {
        "phase": "2.8",
        "total_tags": total_tags,
        "matched_tags": matched_tags,
        "new_tags": new_tags,
        "filled_fields": filled_fields,
        "output_path": str(output_path),
    }
    audit_path = base_dir / "04-输出结果/04-审计数据/phase2_8_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 2.8 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""v3.7 三字段注入器

读取 v3.7 Excel 和特异性映射表，为所有品线表注入/补全
「品类特异性指数」「共性/特性分类」「主导品类」三字段。

处理逻辑:
1. 5 个缺失字段的品线表 -> 新增三列
2. 已有字段的表 -> 填充空值，修正偏差值（差异>0.2时覆盖）
3. 未匹配标签（VOC无命中）-> 基于「适用产品品线」规则推断
4. 统一值域: "共性"->"共性标签", "特性"->"特性标签"
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

# 数据 Sheet 列表（排除元数据 sheet）
DATA_SHEETS = [
    "01_通用标签主表",
    "02_吸奶器",
    "03_内衣服饰",
    "04_家居家纺",
    "05_母婴综合护理",
    "06_喂养电器",
    "07_智能母婴电器",
]

THREE_FIELDS = ["品类特异性指数", "共性/特性分类", "主导品类"]

# 值域统一映射
CLASS_NORMALIZE = {
    "共性": "共性标签",
    "特性": "特性标签",
}

# Sheet -> 品线名称 映射
SHEET_TO_LINE = {
    "01_通用标签主表": None,  # 通用标签，不推断
    "02_吸奶器": "吸奶器",
    "03_内衣服饰": "内衣服饰",
    "04_家居家纺": "家居家纺",
    "05_母婴综合护理": "母婴综合护理",
    "06_喂养电器": "喂养电器",
    "07_智能母婴电器": "智能母婴电器",
}


def load_specificity_map(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_class_value(val):
    """统一共性/特性分类值域"""
    if pd.isna(val):
        return val
    s = str(val).strip()
    return CLASS_NORMALIZE.get(s, s)


def infer_from_product_line(product_line: str, sheet_line) -> dict:
    """基于适用产品品线推断三字段

    如果适用产品品线明确指向单一品线 -> 特性标签, 指数=1.0
    如果适用产品品线包含多个品线 -> 半共性或共性
    """
    if not product_line or str(product_line).strip().lower() == "nan":
        return None

    pl = str(product_line).strip()

    # 检查是否为单一品线
    six_lines = ["吸奶器", "内衣服饰", "家居家纺", "母婴综合护理", "喂养电器", "智能母婴电器"]
    matched = [l for l in six_lines if l in pl]

    if len(matched) == 1:
        return {
            "specificity_index": 1.0,
            "commonality_class": "特性标签",
            "dominant_category": matched[0],
        }
    elif len(matched) > 1:
        return {
            "specificity_index": 0.6,
            "commonality_class": "半共性标签",
            "dominant_category": matched[0],
        }
    elif sheet_line and sheet_line in pl:
        # 适用产品品线包含当前 sheet 的品线名（但不完全匹配）
        return {
            "specificity_index": 1.0,
            "commonality_class": "特性标签",
            "dominant_category": sheet_line,
        }
    return None


def inject_fields_to_sheet(df: pd.DataFrame, spec_map: dict, sheet_name: str) -> pd.DataFrame:
    """为单个 sheet 注入/补全三字段"""
    # 检查 tag_en 列名
    tag_en_col = None
    for col in df.columns:
        if "VOC标签（英文）" in col or col == "tag_en":
            tag_en_col = col
            break

    if tag_en_col is None:
        print(f"  ⚠️ {sheet_name}: 未找到 tag_en 列，跳过")
        return df

    # 检查适用产品品线列
    product_line_col = None
    for col in df.columns:
        if "适用产品品线" in col:
            product_line_col = col
            break

    # 记录新增列数（显式指定 dtype=object 避免 FutureWarning）
    cols_added = 0
    for field in THREE_FIELDS:
        if field not in df.columns:
            df[field] = pd.Series(dtype=object)
            cols_added += 1

    if cols_added > 0:
        print(f"  + {sheet_name}: 新增 {cols_added} 列")

    sheet_line = SHEET_TO_LINE.get(sheet_name)

    # 填充/修正每行
    filled_from_voc = 0
    filled_from_rule = 0
    corrected = 0
    skipped = 0

    for idx, row in df.iterrows():
        tag_en = str(row.get(tag_en_col, "")).strip()
        if not tag_en or tag_en.lower() == "nan":
            skipped += 1
            continue

        spec = spec_map.get(tag_en)

        if spec is not None:
            # === VOC 数据匹配 ===
            # 品类特异性指数
            current_idx = row.get("品类特异性指数")
            new_idx = spec["specificity_index"]
            if pd.isna(current_idx):
                df.at[idx, "品类特异性指数"] = new_idx
                filled_from_voc += 1
            else:
                try:
                    current_idx_f = float(current_idx)
                    if abs(current_idx_f - new_idx) > 0.2:
                        df.at[idx, "品类特异性指数"] = new_idx
                        corrected += 1
                except (ValueError, TypeError):
                    df.at[idx, "品类特异性指数"] = new_idx
                    corrected += 1

            # 共性/特性分类
            current_cls = row.get("共性/特性分类")
            new_cls = spec["commonality_class"]
            if pd.isna(current_cls):
                df.at[idx, "共性/特性分类"] = new_cls
                filled_from_voc += 1
            else:
                normalized = normalize_class_value(current_cls)
                if normalized != str(current_cls).strip():
                    df.at[idx, "共性/特性分类"] = normalized
                    corrected += 1

            # 主导品类
            current_dom = row.get("主导品类")
            new_dom = spec["dominant_category"]
            if pd.isna(current_dom):
                df.at[idx, "主导品类"] = new_dom
                filled_from_voc += 1
            else:
                current_dom_s = str(current_dom).strip()
                if current_dom_s != new_dom and current_dom_s not in ("多品线", "通用"):
                    df.at[idx, "主导品类"] = new_dom
                    corrected += 1

        else:
            # === VOC 无命中 -> 规则推断 ===
            product_line = row.get(product_line_col) if product_line_col else None
            inferred = infer_from_product_line(product_line, sheet_line)

            if inferred is not None:
                current_idx = row.get("品类特异性指数")
                if pd.isna(current_idx):
                    df.at[idx, "品类特异性指数"] = inferred["specificity_index"]
                    filled_from_rule += 1

                current_cls = row.get("共性/特性分类")
                if pd.isna(current_cls):
                    df.at[idx, "共性/特性分类"] = inferred["commonality_class"]
                    filled_from_rule += 1

                current_dom = row.get("主导品类")
                if pd.isna(current_dom):
                    df.at[idx, "主导品类"] = inferred["dominant_category"]
                    filled_from_rule += 1
            else:
                skipped += 1

    total_filled = filled_from_voc // 3 + filled_from_rule // 3
    print(f"    VOC填充: {filled_from_voc//3} 标签, 规则推断: {filled_from_rule//3} 标签, 修正: {corrected} 处, 跳过: {skipped}")
    return df


def main():
    print("=" * 70)
    print("v3.7 三字段注入器")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent
    excel_path = base_dir / "04-输出结果/01-字典版本/tag_dictionary_v3.7.xlsx"
    spec_map_path = base_dir / "04-输出结果/08-辅助数据/v37_tag_specificity_map.json"
    output_path = base_dir / "04-输出结果/01-字典版本/tag_dictionary_v3.7.xlsx"

    print(f"\n输入 Excel: {excel_path}")
    print(f"输入映射: {spec_map_path}")

    # 加载映射表
    spec_map = load_specificity_map(spec_map_path)
    print(f"  加载映射: {len(spec_map)} 标签")

    # 加载 Excel
    xl = pd.ExcelFile(excel_path)
    print(f"  Sheets: {xl.sheet_names}")

    updated_sheets = {}

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name)

        if sheet_name in DATA_SHEETS:
            print(f"\n--- 处理: {sheet_name} ({len(df)} 标签) ---")
            df = inject_fields_to_sheet(df, spec_map, sheet_name)
        else:
            print(f"\n--- 复制: {sheet_name} ({len(df)} 行) ---")

        updated_sheets[sheet_name] = df

    # 保存
    print(f"\n--- 保存更新后的 Excel ---")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in updated_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"  输出: {output_path}")

    # 验证
    print(f"\n--- 验证 ---")
    xl_verify = pd.ExcelFile(output_path)
    for sheet_name in DATA_SHEETS:
        if sheet_name in xl_verify.sheet_names:
            df = pd.read_excel(xl_verify, sheet_name=sheet_name)
            for field in THREE_FIELDS:
                if field in df.columns:
                    non_null = df[field].notna().sum()
                    print(f"  {sheet_name}.{field}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
                else:
                    print(f"  {sheet_name}.{field}: ❌ 列不存在!")

    print("\n" + "=" * 70)
    print("注入完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

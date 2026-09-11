"""标签字典更新（Phase 2.7）

将验证通过的候选标签插入标签字典草稿：
- 高频候选标签（support_count >= 20）自动插入
- 标记为「候选-待审核」
- 策略包/主责部门/优先级/原子指标标记为「待填写」

输出：tag_dictionary_v3.4_draft.xlsx
"""

import json
from pathlib import Path

import pandas as pd


# 品类 -> 品线映射（从主数据提取）
CATEGORY_TO_LINE = {
    "Air 1吸奶器配件": "吸奶器",
    "M5吸奶器配件": "吸奶器",
    "M6吸奶器配件": "吸奶器",
    "M9吸奶器配件": "吸奶器",
    "S系列吸奶器通用配件": "吸奶器",
    "V系列吸奶器迭代款配件": "吸奶器",
    "V系列吸奶器通用配件": "吸奶器",
    "一体穿戴式吸奶器": "吸奶器",
    "乳房按摩乳垫": "智能母婴电器",
    "乳房按摩仪配件": "智能母婴电器",
    "储奶袋": "智能母婴电器",
    "儿童温度计": "智能母婴电器",
    "分离穿戴式吸奶器": "吸奶器",
    "双边穿戴式吸奶器": "吸奶器",
    "常规乳房按摩仪": "智能母婴电器",
    "常规内衣": "内衣服饰",
    "常规内裤": "内衣服饰",
    "旋钮奶袋": "智能母婴电器",
    "消毒器": "喂养电器",
    "清洗机": "喂养电器",
    "滚珠按摩仪": "智能母婴电器",
    "硅胶奶袋": "智能母婴电器",
    "船袜": "内衣服饰",
    "调奶器": "喂养电器",
    "辅食机": "喂养电器",
    "运动内衣": "内衣服饰",
    "运动长裤": "内衣服饰",
    # 新增映射（基于常识推断）
    "吸奶器": "吸奶器",
    "哺乳内衣": "内衣服饰",
    "哺乳背心": "内衣服饰",
    "哺乳睡裙": "内衣服饰",
    "哺乳睡衣套装": "内衣服饰",
    "哺乳巾": "内衣服饰",
    "文胸": "内衣服饰",
    "非文胸产品": "内衣服饰",
    "塑身衣": "内衣服饰",
    "塑身裤": "内衣服饰",
    "塑身内裤": "内衣服饰",
    "收腹带": "内衣服饰",
    "托腹带": "内衣服饰",
    "孕妇内裤": "内衣服饰",
    "孕妇裤": "内衣服饰",
    "压力袜": "内衣服饰",
    "一次性产妇内裤": "内衣服饰",
    "背奶包": "内衣服饰",
    "产妇袜": "内衣服饰",
    "奶袋": "智能母婴电器",
    "温度计": "智能母婴电器",
    "奶瓶清洗机": "喂养电器",
    "乳房按摩仪": "智能母婴电器",
    "乳房冷热敷垫": "智能母婴电器",
    "乳房热敷垫": "智能母婴电器",
    "乳房冰垫": "智能母婴电器",
    "乳房护理垫套装组合": "智能母婴电器",
    "乳头保护罩": "智能母婴电器",
    "常规乳头修护霜": "智能母婴电器",
    "电动乳头按摩霜": "智能母婴电器",
    "磨甲器": "智能母婴电器",
    "电动磨甲器配件": "智能母婴电器",
    "lupantte磨甲器配件": "智能母婴电器",
    "NC02电动磨甲器配件": "智能母婴电器",
    "儿童体重秤": "智能母婴电器",
    "贴片式体温计配件": "智能母婴电器",
    "婴儿监视器": "智能母婴电器",
    "BN02婴儿吸鼻器配件": "智能母婴电器",
    "BN007吸鼻器配件": "智能母婴电器",
    "分体式吸鼻器": "智能母婴电器",
    "手持式吸鼻器": "智能母婴电器",
    "辅食装袋器": "喂养电器",
    "辅食袋": "喂养电器",
    "奶袋": "智能母婴电器",
    "储奶壶": "智能母婴电器",
    "GP01储奶壶配件": "智能母婴电器",
    "消毒器": "喂养电器",
    "调奶器": "喂养电器",
    "辅食机": "喂养电器",
    "奶瓶沥干支架": "喂养电器",
    "电动奶瓶清洁刷": "喂养电器",
    "硅胶奶瓶清洁刷": "喂养电器",
    "湿巾加热盒": "喂养电器",
    "清洁液": "喂养电器",
    "清洁喷雾": "喂养电器",
    "牙刷": "喂养电器",
    "牙胶组合": "喂养电器",
    "全硅胶款牙胶": "喂养电器",
    "注水款牙胶": "喂养电器",
    "电动摇椅": "智能母婴电器",
    "3D摇电动摇椅配件": "智能母婴电器",
    "成长型餐椅": "智能母婴电器",
    "餐椅配件": "智能母婴电器",
    "推车": "智能母婴电器",
    "轻型婴儿车": "智能母婴电器",
    "推车通用配件": "智能母婴电器",
    "车挂包": "智能母婴电器",
    "ST01多功能婴儿标准车配件": "智能母婴电器",
    "桌面空气净化器": "智能母婴电器",
    "台式白噪音机器": "智能母婴电器",
    "便携白噪音机器": "智能母婴电器",
    "夹腿枕": "家居家纺",
    "记忆棉哺乳枕": "家居家纺",
    "U型哺乳枕": "家居家纺",
    "U型双边孕妇枕": "家居家纺",
    "U型双边孕妇枕套": "家居家纺",
    "U型哺乳枕套": "家居家纺",
    "J型单边孕妇枕": "家居家纺",
    "J型二代单边孕妇枕": "家居家纺",
    "J型单边孕妇枕套": "家居家纺",
    "G型双边孕妇枕": "家居家纺",
    "W型孕妇枕": "家居家纺",
    "哺乳枕": "家居家纺",
    "腰凳": "家居家纺",
    "腰凳配件": "家居家纺",
    "腰凳背带": "家居家纺",
    "圆环背巾": "家居家纺",
    "插扣背巾": "家居家纺",
    "包巾": "家居家纺",
    "襁褓巾": "家居家纺",
    "安抚巾": "家居家纺",
    "浴巾": "家居家纺",
    "方巾": "家居家纺",
    "床笠": "家居家纺",
    "凉感被": "家居家纺",
    "凉感毯": "家居家纺",
    "沐浴炸弹": "母婴综合护理",
    "沐浴片": "母婴综合护理",
    "应急垫": "母婴综合护理",
    "常规运动垫": "母婴综合护理",
    "围栏款运动垫": "母婴综合护理",
    "双布套运动垫": "母婴综合护理",
    "便携隔尿垫": "母婴综合护理",
    "便携隔尿垫包": "母婴综合护理",
    "一次性防溢乳垫": "母婴综合护理",
    "可水洗防溢乳垫": "母婴综合护理",
    "防挤压支架": "母婴综合护理",
    "吸奶器内衣": "母婴综合护理",
    "吸奶器背心": "母婴综合护理",
    "吸奶器湿巾": "母婴综合护理",
    "保温包": "母婴综合护理",
    "妈咪包": "母婴综合护理",
    "收纳包": "母婴综合护理",
    "待产包": "母婴综合护理",
    "沐浴露": "母婴综合护理",
    "洗发水": "母婴综合护理",
    "屁屁膏/尿布膏": "母婴综合护理",
    "尿布膏喷雾": "母婴综合护理",
    "生理盐水湿巾": "母婴综合护理",
    "纯水湿巾": "母婴综合护理",
    "乳液湿巾": "母婴综合护理",
    "洗护套装": "母婴综合护理",
    "护理套装": "母婴综合护理",
    "婴儿无袖睡袋": "母婴综合护理",
    "柔软纸尿裤": "母婴综合护理",
    "竹纤维纸尿裤": "母婴综合护理",
    "大环腰拉拉裤": "母婴综合护理",
    "伸缩门栏": "母婴综合护理",
    "跳舞毯": "母婴综合护理",
    "其他玩具": "母婴综合护理",
    "单边音乐毯": "母婴综合护理",
    "婴儿纱布四季盖毯": "母婴综合护理",
}

# 品线 -> Sheet 名称映射
LINE_TO_SHEET = {
    "吸奶器": "02_吸奶器",
    "内衣服饰": "03_内衣服饰",
    "家居家纺": "04_家居家纺",
    "母婴综合护理": "05_母婴综合护理",
    "喂养电器": "06_喂养电器",
    "智能母婴电器": "07_智能母婴电器",
}


def get_next_tag_id(existing_ids: set[str], prefix: str = "TAG_NEW_") -> str:
    """生成下一个标签ID"""
    max_num = 0
    for tid in existing_ids:
        if tid.startswith(prefix):
            try:
                num = int(tid[len(prefix):])
                max_num = max(max_num, num)
            except ValueError:
                pass
    return f"{prefix}{max_num + 1:03d}"


def main():
    print("=" * 70)
    print("Phase 2.7: 标签字典自动更新")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent / "03-数据资产/原种子标签"
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = base_dir / "SGCS_标签字典_VOC标签大宽表V3.0_v1.xlsx"
    output_path = output_dir / "tag_dictionary_v3.4_draft.xlsx"

    # 1. 加载候选标签
    print("\n--- 加载候选标签 ---")
    candidates_path = (
        Path(__file__).parent.parent.parent
        / "04-输出结果/tag_gap_analysis/candidate_tags_filtered.json"
    )
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # 只保留高频候选
    candidates = [c for c in candidates if c["support_count"] >= 20]
    print(f"  高频候选 (>=20): {len(candidates)} 个")

    # 2. 加载原字典
    print("\n--- 加载原标签字典 ---")
    xl = pd.ExcelFile(input_path)
    print(f"  Sheets: {xl.sheet_names}")

    # 3. 处理每个 sheet
    print("\n--- 更新各 Sheet ---")

    # 读取通用标签主表，收集现有 tag_id
    general_df = pd.read_excel(xl, sheet_name="01_通用标签主表")
    existing_ids = set(str(tid).strip() for tid in general_df["标签ID"].dropna())

    # 读取各品线 sheet
    line_sheets = {}
    for sheet_name in xl.sheet_names:
        if sheet_name not in ["00_字段说明", "08_映射关系表", "09_存量标签归档"]:
            line_sheets[sheet_name] = pd.read_excel(xl, sheet_name=sheet_name)
            for tid in line_sheets[sheet_name]["标签ID"].dropna():
                existing_ids.add(str(tid).strip())

    print(f"  现有标签ID: {len(existing_ids)} 个")

    # 4. 按品线分组候选标签
    line_candidates = {line: [] for line in LINE_TO_SHEET.keys()}
    unknown_candidates = []

    for c in candidates:
        cat = c["applicable_category"]
        line = CATEGORY_TO_LINE.get(cat)
        if line:
            line_candidates[line].append(c)
        else:
            unknown_candidates.append(c)

    print(f"  已映射品类: {sum(len(v) for v in line_candidates.values())} 个")
    print(f"  未映射品类: {len(unknown_candidates)} 个")

    # 5. 为每个品线 sheet 添加候选标签
    updated_sheets = {}
    for line, sheet_name in LINE_TO_SHEET.items():
        if sheet_name not in line_sheets:
            continue

        df = line_sheets[sheet_name].copy()
        added = []

        for c in line_candidates.get(line, []):
            tag_id = get_next_tag_id(existing_ids, "TAG_NEW_")
            existing_ids.add(tag_id)

            new_row = {
                "标签ID": tag_id,
                "VOC标签（英文）": c["tag_en"],
                "VOC标签（中文）": c["tag_cn"],
                "AIPL节点": c["suggested_aipl"],
                "情感极性": c["suggested_sentiment"],
                "适用产品品线": line,
                "适用产品品类": c["applicable_category"],
                "是否通用标签": "否",
                "标签主题": "自动发现-待审核",
                "策略包": "【待填写】",
                "主责部门": "【待填写】",
                "默认优先级": "【待填写】",
                "对应原子指标": "【待填写】",
                "英文关键词/典型表达": c["source_phrase"],
                "消费者习惯关键词/原话短语": "",
                "数据来源": f"gap_detection ({c['support_count']}次)",
                "审核状态": "候选-待审核",
            }
            added.append(new_row)

        if added:
            added_df = pd.DataFrame(added)
            # 对齐列
            for col in df.columns:
                if col not in added_df.columns:
                    added_df[col] = ""
            added_df = added_df[df.columns]
            df = pd.concat([df, added_df], ignore_index=True)
            print(f"  {sheet_name}: +{len(added)} 个候选标签")
        else:
            print(f"  {sheet_name}: 无新增")

        updated_sheets[sheet_name] = df

    # 6. 保存新 Excel
    print(f"\n--- 保存新字典 ---")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # 先复制原 sheet
        for sheet_name in xl.sheet_names:
            if sheet_name in updated_sheets:
                updated_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
            elif sheet_name == "01_通用标签主表":
                general_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                df = pd.read_excel(xl, sheet_name=sheet_name)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"  输出: {output_path}")

    # 7. 统计
    total_added = sum(len(line_candidates.get(line, [])) for line in LINE_TO_SHEET.keys())
    print(f"\n--- 更新统计 ---")
    print(f"  新增候选标签: {total_added} 个")
    print(f"  未映射候选: {len(unknown_candidates)} 个")
    for c in unknown_candidates[:10]:
        print(f"    - {c['tag_en']} ({c['applicable_category']})")

    # 8. 审计
    audit = {
        "phase": "2.7",
        "description": "标签字典自动更新",
        "original_sheets": len(xl.sheet_names),
        "total_existing_tags": len(existing_ids) - total_added,
        "added_candidates": total_added,
        "unknown_candidates": len(unknown_candidates),
        "output_path": str(output_path),
    }
    audit_path = output_dir / "phase2_7_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 2.7 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

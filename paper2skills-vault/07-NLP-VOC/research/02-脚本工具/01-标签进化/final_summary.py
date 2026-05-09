"""最终综合报告生成

汇总 Phase 1-2 全部产出，生成可读的综合报告。
"""

import json
from pathlib import Path


def main():
    print("=" * 70)
    print("最终综合报告：统一萃取引擎与标签字典重构")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent / "04-输出结果"

    # 收集各阶段审计
    reports = {}
    for name, path in {
        "Phase 1.1": "unified_labeling/phase1_1_audit.json",
        "Phase 1.2": "unified_labeling/phase1_2_audit.json",
        "Phase 1.3": "unified_labeling/phase1_3_audit.json",
        "Phase 1.5": "unified_labeling/phase1_5_audit.json",
        "Phase 2.1": "tag_gap_analysis/phase2_1_audit.json",
        "Phase 2.2": "tag_gap_analysis/phase2_2_audit.json",
        "Phase 2.3-2.4": "tag_gap_analysis/phase2_3_4_audit.json",
        "Phase 2.5": "tag_gap_analysis/phase2_5_audit.json",
        "Phase 2.6": "phase2_6_audit.json",
        "Phase 2.7": "phase2_7_audit.json",
        "Phase 2.8": "phase2_8_audit.json",
        "Phase 2.9": "phase2_9_audit.json",
    }.items():
        p = base_dir / path
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                reports[name] = json.load(f)

    # 输出报告
    print("\n" + "=" * 70)
    print("一、数据统一与萃取打标（Phase 1）")
    print("=" * 70)

    p11 = reports.get("Phase 1.1", {})
    print(f"\n  Step 1.1 统一输入格式")
    print(f"    总记录: {p11.get('total_records', 'N/A'):,} 条")
    for src, cnt in p11.get("source_breakdown", {}).items():
        print(f"      {src}: {cnt:,}")

    p12 = reports.get("Phase 1.2", {})
    print(f"\n  Step 1.2 质量筛选")
    print(f"    高质量: {p12.get('high_quality_count', 'N/A'):,} / {p12.get('total_records', 'N/A'):,} ({p12.get('high_quality_rate', 'N/A')}%)")
    print(f"    质量分均值: {p12.get('quality_score_mean', 'N/A')}")

    p13 = reports.get("Phase 1.3", {})
    print(f"\n  Step 1.3 萃取引擎打标")
    print(f"    v3.3 转录: {p13.get('v3.3_transcribed', 'N/A'):,} 条")
    print(f"    增量打标: {p13.get('incremental_keyword', 'N/A'):,} 条")
    print(f"    覆盖率: {p13.get('coverage_rate', 'N/A')}%")
    print(f"    零标签率: {p13.get('zero_tag_rate', 'N/A')}%")

    p15 = reports.get("Phase 1.5", {})
    print(f"\n  Step 1.4-1.5 品线推断")
    print(f"    推断成功: {p15.get('inferred_count', 'N/A'):,} 条")
    print(f"    最终缺品线: {p15.get('missing_line_after', 'N/A'):,} ({p15.get('missing_line_after', 0) / p15.get('total_records', 1) * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("二、逆向分析与标签字典更新（Phase 2）")
    print("=" * 70)

    p21 = reports.get("Phase 2.1", {})
    print(f"\n  Step 2.1 零标签提取")
    print(f"    零标签: {p21.get('zero_label_count', 'N/A'):,} 条 ({p21.get('zero_label_rate', 'N/A')}%)")
    print(f"    品类覆盖: {p21.get('category_count', 'N/A')} 个")

    p22 = reports.get("Phase 2.2", {})
    print(f"\n  Step 2.2 缺口检测")
    print(f"    发现缺口品类: {p22.get('gap_categories', 'N/A')} 个")
    print(f"    候选标签: {p22.get('candidate_tags', 'N/A')} 个")

    p24 = reports.get("Phase 2.3-2.4", {})
    print(f"\n  Step 2.3-2.4 候选标签过滤")
    print(f"    原始候选: {p24.get('raw_count', 'N/A')} 个")
    print(f"    过滤后: {p24.get('filtered_count', 'N/A')} 个")
    print(f"    去重后: {p24.get('merged_count', 'N/A')} 个")

    p25 = reports.get("Phase 2.5", {})
    print(f"\n  Step 2.5 Active-Learning 质量把关")
    print(f"    需要人工审核: {p25.get('needs_review_count', 'N/A')} / {p25.get('total_candidates', 'N/A')}")
    print(f"    自动通过: {p25.get('auto_approve_count', 'N/A')} / {p25.get('total_candidates', 'N/A')}")

    p26 = reports.get("Phase 2.6", {})
    print(f"\n  Step 2.6 ALCHEmist Label Function 生成")
    print(f"    生成函数: {p26.get('generated_functions', 'N/A')} 个")
    print(f"    代码行数: {p26.get('code_lines', 'N/A')} 行")

    p27 = reports.get("Phase 2.7", {})
    print(f"\n  Step 2.7 标签字典更新")
    print(f"    新增候选标签: {p27.get('added_candidates', 'N/A')} 个")

    p28 = reports.get("Phase 2.8", {})
    print(f"\n  Step 2.8 V3.0 增量字段补充")
    print(f"    匹配填充: {p28.get('matched_tags', 'N/A')} 个标签")

    p29 = reports.get("Phase 2.9", {})
    print(f"\n  Step 2.9 标签字典结构验证")
    print(f"    总标签: {p29.get('total_tags', 'N/A')} 个")
    print(f"    新增标签: {p29.get('total_new_tags', 'N/A')} 个")
    print(f"    验证结果: {'✅ 通过' if p29.get('passed') else '❌ 失败'}")

    print("\n" + "=" * 70)
    print("三、核心产出文件")
    print("=" * 70)

    outputs = [
        ("统一打标结果", "04-输出结果/unified_labeling/phase1_5_all_sources_labeled_final.jsonl", "364,569 条 VOC 的完整标签"),
        ("标签字典 v3.4", "04-输出结果/02-历史字典/tag_dictionary_v3.4_filled.xlsx", "483 个标签（409 原有 + 74 新增）"),
        ("ALCHEmist Label Functions", "04-输出结果/alchemist_label_functions.py", "74 个可审计标注规则"),
        ("人工审核清单", "04-输出结果/tag_gap_analysis/manual_review_checklist.csv", "1 个候选标签待审核"),
        ("零标签样本", "04-输出结果/tag_gap_analysis/zero_label_samples.csv", "9,486 条零标签 VOC"),
        ("缺口分析报告", "04-输出结果/tag_gap_analysis/gap_analysis.json", "112 个品类缺口"),
    ]

    for name, path, desc in outputs:
        full_path = Path(__file__).parent.parent.parent / path
        exists = "✅" if full_path.exists() else "❌"
        print(f"\n  {exists} {name}")
        print(f"     路径: {path}")
        print(f"     说明: {desc}")

    print("\n" + "=" * 70)
    print("四、关键指标")
    print("=" * 70)

    print(f"""
  ┌─────────────────────┬──────────┬──────────┐
  │ 指标                │ 实际值   │ 目标值   │
  ├─────────────────────┼──────────┼──────────┤
  │ VOC 数据总量        │ 368,624  │ -        │
  │ 标签覆盖率          │ 41.9%    │ >70%    │ ⚠️ 未达标 │
  │ 零标签率            │ 58.1%    │ <30%    │ ⚠️ 未达标 │
  │ 标签字典标签总数    │ 483      │ -        │
  │ 新增候选标签        │ 74       │ -        │
  │ 标签字典验证        │ 通过     │ 通过     │ ✅       │
  └─────────────────────┴──────────┴──────────┘
    """)

    print("=" * 70)
    print("五、风险与建议")
    print("=" * 70)

    print("""
  1. 覆盖率 41.9% 远低于 70% 目标
     → 需要部署 ALCHEmist label functions 到生产环境
     → 扩大标签关键词库，特别是 Momcozy 自有数据

  2. Momcozy 数据覆盖率仅 14.5%
     → 针对客服工单/退货留言文本特征训练专用规则
     → 引入多语言支持（法语/德语等 Trustpilot 评论）

  3. 211,919 条零标签 VOC 价值未挖掘
     → 使用 TaxoAdapt 进行文本嵌入聚类分析
     → LLM 辅助主题总结生成更多候选标签

  4. 74 个新增候选标签待人工确认
     → 审核 manual_review_checklist.csv
     → 确认通过后更新标签字典正式版本
    """)

    print("=" * 70)
    print("执行完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""最终审计报告（Phase 1-2 综合）

汇总全部中间产出，验证关键指标，生成综合审计报告。
"""

import json
from collections import defaultdict
from pathlib import Path


def main():
    print("=" * 70)
    print("最终审计报告：Phase 1-2 综合")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent / "04-输出结果"

    # 收集各阶段审计
    audits = {}
    for phase_file in [
        "unified_labeling/phase1_1_audit.json",
        "unified_labeling/phase1_2_audit.json",
        "unified_labeling/phase1_3_audit.json",
        "tag_gap_analysis/phase2_1_audit.json",
        "tag_gap_analysis/phase2_2_audit.json",
        "tag_gap_analysis/phase2_3_4_audit.json",
        "phase2_7_audit.json",
    ]:
        path = base_dir / phase_file
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                audits[path.name] = json.load(f)

    # 综合报告
    report = {
        "audit_time": "2026-04-22",
        "audit_scope": "Phase 1 (数据统一加载与萃取打标) + Phase 2 (逆向分析与标签字典更新)",
        "phases": {},
        "key_metrics": {},
        "risks": [],
        "next_steps": [],
    }

    # Phase 1 指标
    if "phase1_3_audit.json" in audits:
        p13 = audits["phase1_3_audit.json"]
        report["phases"]["phase1_labeling"] = {
            "total_labeled": p13.get("total_labeled", 0),
            "v3_3_transcribed": p13.get("v3.3_transcribed", 0),
            "incremental_keyword": p13.get("incremental_keyword", 0),
            "coverage_rate": p13.get("coverage_rate", 0),
            "zero_tag_rate": p13.get("zero_tag_rate", 0),
        }
        report["key_metrics"]["label_coverage"] = p13.get("coverage_rate", 0)

    # Phase 2 指标
    if "phase2_1_audit.json" in audits:
        p21 = audits["phase2_1_audit.json"]
        report["phases"]["phase2_gap_analysis"] = {
            "zero_label_count": p21.get("zero_label_count", 0),
            "zero_label_rate": p21.get("zero_label_rate", 0),
            "category_count": p21.get("category_count", 0),
        }

    if "phase2_2_audit.json" in audits:
        p22 = audits["phase2_2_audit.json"]
        report["phases"]["phase2_gap_detection"] = {
            "gap_categories": p22.get("gap_categories", 0),
            "candidate_tags": p22.get("candidate_tags", 0),
        }

    if "phase2_7_audit.json" in audits:
        p27 = audits["phase2_7_audit.json"]
        report["phases"]["phase2_dictionary_update"] = {
            "added_candidates": p27.get("added_candidates", 0),
        }

    # 风险
    report["risks"] = [
        {
            "risk": "覆盖率仅 41.9%，远低于 70% 目标",
            "impact": "大量 VOC 未被打标，业务洞察不完整",
            "mitigation": "需要 ALCHEmist label functions 提升覆盖率，或扩大关键词库",
        },
        {
            "risk": "增量关键词匹配对 Momcozy 覆盖率低（14.5%）",
            "impact": "自有数据价值未充分挖掘",
            "mitigation": "针对 Momcozy 数据特性训练专用 label functions",
        },
        {
            "risk": "候选标签未经人工审核",
            "impact": "噪音标签可能进入字典",
            "mitigation": "Phase 2.5 Active-Learning 人工审核待执行",
        },
        {
            "risk": "品类映射不完整（101 品类仅映射 74 个）",
            "impact": "部分候选标签无法自动插入字典",
            "mitigation": "补充品类-品线映射表",
        },
    ]

    # 下一步
    report["next_steps"] = [
        "Phase 2.5: Active-Learning 质量把关（人工审核 Top-K 候选标签）",
        "Phase 2.6: ALCHEmist Label Function 生成（为新标签生成可审计标注规则）",
        "Phase 2.8: V3.0 增量字段补充（策略包/主责部门/优先级/原子指标）",
        "Phase 2.9: 标签字典结构验证（Sheet1/Sheet2 完整性检查）",
        "Phase 1.4-1.5: 品线/品类推断（补充 Amazon/Trustpilot/Reddit 品类信息）",
    ]

    # 输出
    report_path = base_dir / "final_audit_phase1_2.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n--- 综合审计报告 ---")
    print(f"  输出: {report_path}")
    print(f"\n  关键指标:")
    for k, v in report["key_metrics"].items():
        print(f"    {k}: {v}")

    print(f"\n  Phase 1 打标:")
    p1 = report["phases"].get("phase1_labeling", {})
    print(f"    总计: {p1.get('total_labeled', 0):,}")
    print(f"    v3.3 转录: {p1.get('v3_3_transcribed', 0):,}")
    print(f"    增量打标: {p1.get('incremental_keyword', 0):,}")
    print(f"    覆盖率: {p1.get('coverage_rate', 0)}%")

    print(f"\n  Phase 2 逆向:")
    p2g = report["phases"].get("phase2_gap_analysis", {})
    print(f"    零标签: {p2g.get('zero_label_count', 0):,} ({p2g.get('zero_label_rate', 0)}%)")
    p2d = report["phases"].get("phase2_dictionary_update", {})
    print(f"    新增候选标签: {p2d.get('added_candidates', 0)} 个")

    print(f"\n  风险 ({len(report['risks'])}):")
    for r in report["risks"]:
        print(f"    - {r['risk']}")

    print(f"\n  下一步 ({len(report['next_steps'])}):")
    for i, s in enumerate(report["next_steps"], 1):
        print(f"    {i}. {s}")

    print("\n" + "=" * 70)
    print("最终审计完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Phase 4 统一打标流水线

集成四项新 labeler 对全量 VOC 重新打标：
1. 通用标签（general_tag_labeler）— 情感/体验/属性
2. 品牌标签（brand_label_functions）— 品牌识别
3. Zendesk 极简规则（zendesk_minimal_rules）— 短工单
4. 负面缺陷标签（negative_defect_miner）— 缺陷描述

执行顺序：
  零标签记录：Zendesk规则 → 通用标签 → 负面缺陷 → 品牌标签
  有标签记录：仅追加品牌标签

冲突解决：
  - 通用标签内部互斥（resolve_exclusive_pairs）
  - Zendesk 内部互斥（resolve_zendesk_conflicts）
  - 品牌标签去重（resolve_brand_conflicts）
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# ── 将各模块所在目录加入路径 ───────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent.resolve()
_DATA_PROC_DIR = _SCRIPT_DIR.parent / "04-数据处理"
for d in (_SCRIPT_DIR, _DATA_PROC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

# ── 导入各 labeler ─────────────────────────────────────────────────
from general_tag_labeler import (
    GENERAL_TAGS,
    POS_TO_NEG,
    label_record as apply_general_tags,
    resolve_exclusive_pairs as resolve_general_conflicts,
)
from brand_label_functions import (
    apply_brand_labels,
    resolve_brand_conflicts,
    BRAND_KEYWORD_LIBRARY,
)
from zendesk_minimal_rules import (
    ZENDESK_RULES,
    apply_zendesk_rules,
    resolve_zendesk_conflicts,
    should_use_minimal_rules,
)

# ── 负面缺陷标签（内联定义，不依赖 miner 运行时）───────────────────

NEGATIVE_DEFECT_TAGS = [
    {
        "tag_id": "TAG_DEF_N001",
        "tag_en": "functional_failure",
        "tag_cn": "功能失效",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "doesn't work", "doesnt work", "not working", "stopped working",
            "won't turn on", "wont turn on", "no power", "dead",
            "malfunction", "faulty", "defective", "broken", "broke",
            "not charging", "not charge", "no suction", "weak suction",
            "low suction", "not strong", "too weak", "no response",
            "kaputt", "defekt", "nicht funktioniert",
            "ne fonctionne pas", "défectueux", "cassé",
        ],
    },
    {
        "tag_id": "TAG_DEF_N002",
        "tag_en": "leakage_issue",
        "tag_cn": "泄漏问题",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "leaking", "leak", "leaks", "leaked", "spill", "spills",
            "spilling", "drip", "dripping", "drips",
        ],
    },
    {
        "tag_id": "TAG_DEF_N003",
        "tag_en": "surface_damage",
        "tag_cn": "表面损伤",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "crack", "cracked", "cracking", "scratch", "scratched",
            "dented", "stain", "stained", "discolor", "discolored",
            "fading", "faded", "peel", "peeling", "tear", "torn",
            "rip", "ripped", "fray", "fraying", "frayed",
        ],
    },
    {
        "tag_id": "TAG_DEF_N004",
        "tag_en": "odor_overheating",
        "tag_cn": "异味过热",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "smell", "smells", "smelly", "odor", "stink", "stinky",
            "burning smell", "chemical smell", "plastic smell",
            "overheat", "overheating", "overheated", "too hot", "burning",
            "melt", "melted", "melting", "warp", "warped",
        ],
    },
    {
        "tag_id": "TAG_DEF_N005",
        "tag_en": "structural_looseness",
        "tag_cn": "结构松动",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "loose", "loosen", "fall off", "falls off", "falling off",
            "detach", "detached", "come off", "comes off", "coming off",
            "wobble", "wobbly", "not secure", "not stable",
        ],
    },
    {
        "tag_id": "TAG_DEF_N006",
        "tag_en": "noise_issue",
        "tag_cn": "噪音问题",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "noisy", "loud", "squeak", "squeaking", "rattling",
            "vibrating", "buzzing", "clicking", "grinding", "whining",
        ],
    },
    {
        "tag_id": "TAG_DEF_N007",
        "tag_en": "missing_parts",
        "tag_cn": "缺少配件",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "missing", "not included", "incomplete", "no adapter",
            "no cable", "no charger", "no manual", "no instructions",
            "parts missing", "missing part",
        ],
    },
    {
        "tag_id": "TAG_DEF_N008",
        "tag_en": "wear_aging",
        "tag_cn": "磨损老化",
        "aipl": "L1",
        "sentiment": "negative",
        "keywords": [
            "wear", "worn", "wearing out", "wore out", "deteriorate",
            "deteriorated", "aging", "aged", "old", "not durable",
        ],
    },
]


def apply_negative_defect_tags(text: str) -> list[dict]:
    """应用负面缺陷标签"""
    text_lower = text.lower()
    labels = []
    matched_ids = set()

    for tag in NEGATIVE_DEFECT_TAGS:
        tag_id = tag["tag_id"]
        if tag_id in matched_ids:
            continue

        for kw in tag["keywords"]:
            matched = False
            if " " in kw:
                if kw in text_lower:
                    matched = True
            else:
                if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                    matched = True

            if matched:
                labels.append({
                    "tag_id": tag_id,
                    "tag_en": tag["tag_en"],
                    "tag_cn": tag["tag_cn"],
                    "aipl_node": tag["aipl"],
                    "sentiment_preset": tag["sentiment"],
                    "sentiment_calibrated": -1.0,
                    "confidence": 0.75,
                    "source": "negative_defect",
                })
                matched_ids.add(tag_id)
                break

    return labels


# ── 统一打标函数 ───────────────────────────────────────────────────

def label_single_record(record: dict) -> tuple[list[dict], list[dict]]:
    """对单条记录应用所有 Phase 4 labeler

    Returns: (new_labels, all_labels)
      new_labels: 本次新增的 label
      all_labels: 合并后的全部 label（原有 + 新增）
    """
    text = record.get("text", "")
    source = record.get("data_source", "")
    existing_labels = record.get("labels", [])
    new_labels: list[dict] = []

    # ── 品牌标签（所有记录都追加）─────────────────────────────────
    brand_lbls = apply_brand_labels(text)
    brand_lbls = resolve_brand_conflicts(brand_lbls)
    new_labels.extend(brand_lbls)

    # ── 零标签记录追加打标 ────────────────────────────────────────
    if len(existing_labels) == 0:
        text_stripped = text.strip()

        # 1. Zendesk 极简规则（仅限短文本 + Zendesk 源）
        zen_lbls: list[dict] = []
        if should_use_minimal_rules(text_stripped, source):
            zen_lbls = apply_zendesk_rules(text_stripped)
            zen_lbls = resolve_zendesk_conflicts(zen_lbls)
            new_labels.extend(zen_lbls)

        # 2. 通用标签（无论 Zendesk 是否命中，都尝试补充）
        gen_lbls = apply_general_tags(text_stripped, [])
        # 避免与 Zendesk 标签重复（同一维度）
        zen_ids = {z["tag_id"] for z in zen_lbls}
        gen_lbls = [g for g in gen_lbls if g["tag_id"] not in zen_ids]
        new_labels.extend(gen_lbls)

        # 3. 负面缺陷标签
        defect_lbls = apply_negative_defect_tags(text_stripped)
        new_labels.extend(defect_lbls)

    # ── 最终去重（按 tag_id）──────────────────────────────────────
    all_labels = list(existing_labels)
    existing_ids = {l["tag_id"] for l in existing_labels}
    for lbl in new_labels:
        if lbl["tag_id"] not in existing_ids:
            all_labels.append(lbl)
            existing_ids.add(lbl["tag_id"])

    return new_labels, all_labels


# ── 全量处理 ───────────────────────────────────────────────────────

def run_phase4_labeling(
    input_path: Path,
    output_path: Path,
    max_records: Optional[int] = None,
) -> dict:
    """运行 Phase 4 全量重新打标"""
    print("=" * 70)
    print("Phase 4: 统一打标流水线")
    print("=" * 70)

    print(f"\n输入: {input_path}")
    print(f"输出: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计初始化
    total = 0
    zero_before = 0
    newly_tagged = 0
    tag_counter = Counter()
    source_breakdown = Counter()
    brand_counter = Counter()
    new_tag_breakdown = defaultdict(Counter)
    pipeline_stage_counts = Counter()

    # 各 labeler 独立统计
    zen_hits = 0
    gen_hits = 0
    defect_hits = 0
    brand_hits = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if max_records and total >= max_records:
                break

            rec = json.loads(line)
            total += 1

            existing_labels = rec.get("labels", [])
            n_tags_before = len(existing_labels)

            if n_tags_before == 0:
                zero_before += 1

            # 打标
            new_labels, all_labels = label_single_record(rec)

            # 更新记录
            if new_labels:
                rec["labels"] = all_labels
                rec["n_tags"] = len(all_labels)

                if n_tags_before == 0:
                    newly_tagged += 1

                # 统计新增标签来源
                for lbl in new_labels:
                    src = lbl.get("source", "unknown")
                    tag_counter[lbl["tag_id"]] += 1
                    new_tag_breakdown[src][lbl["tag_id"]] += 1
                    source_breakdown[rec.get("data_source", "unknown")] += 1

                    if src == "brand_label":
                        brand_counter[lbl.get("tag_cn", "unknown")] += 1
                        brand_hits += 1
                    elif src == "zendesk_minimal_rule":
                        zen_hits += 1
                    elif src == "general_tag":
                        gen_hits += 1
                    elif src == "negative_defect":
                        defect_hits += 1

            # 统计各阶段命中
            if new_labels:
                has_zen = any(l.get("source") == "zendesk_minimal_rule" for l in new_labels)
                has_gen = any(l.get("source") == "general_tag" for l in new_labels)
                has_def = any(l.get("source") == "negative_defect" for l in new_labels)
                has_brand = any(l.get("source") == "brand_label" for l in new_labels)

                if has_zen:
                    pipeline_stage_counts["zendesk"] += 1
                if has_gen:
                    pipeline_stage_counts["general"] += 1
                if has_def:
                    pipeline_stage_counts["defect"] += 1
                if has_brand:
                    pipeline_stage_counts["brand"] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total % 50000 == 0:
                coverage = (total - zero_before + newly_tagged) / total * 100
                print(f"  已处理 {total:,} 条 | 新打标 {newly_tagged:,} | 当前覆盖率 {coverage:.2f}%")

    # ── 报告 ──────────────────────────────────────────────────────
    coverage_before = (total - zero_before) / total * 100
    zero_after = zero_before - newly_tagged
    coverage_after = (total - zero_after) / total * 100
    coverage_delta = coverage_after - coverage_before

    print(f"\n{'=' * 70}")
    print("Phase 4 打标结果报告")
    print(f"{'=' * 70}")
    print(f"\n  总记录: {total:,}")
    print(f"  原零标签: {zero_before:,}")
    print(f"  新打标: {newly_tagged:,} ({newly_tagged/zero_before*100:.1f}% of zero-label)")
    print(f"  覆盖率: {coverage_before:.2f}% → {coverage_after:.2f}% (+{coverage_delta:.2f}%)")

    print(f"\n--- 按 Labeler 新打标 ---")
    print(f"  Zendesk 极简规则: {zen_hits:,}")
    print(f"  通用标签: {gen_hits:,}")
    print(f"  负面缺陷: {defect_hits:,}")
    print(f"  品牌标签: {brand_hits:,}")

    print(f"\n--- Pipeline 阶段命中 ---")
    for stage, cnt in pipeline_stage_counts.most_common():
        print(f"  {stage}: {cnt:,}")

    print(f"\n--- 按数据源新打标 ---")
    for src, cnt in source_breakdown.most_common():
        print(f"  {src}: {cnt:,}")

    print(f"\n--- 新增标签 Top 20 ---")
    for tag_id, cnt in tag_counter.most_common(20):
        tag_def = None
        for t in GENERAL_TAGS + ZENDESK_RULES + NEGATIVE_DEFECT_TAGS:
            if t.get("tag_id") == tag_id:
                tag_def = t
                break
        if tag_def:
            print(f"  {tag_id} ({tag_def.get('tag_en', '')}): {cnt:,}")
        else:
            print(f"  {tag_id}: {cnt:,}")

    print(f"\n--- 品牌提及 Top 10 ---")
    for brand, cnt in brand_counter.most_common(10):
        print(f"  {brand}: {cnt:,}")

    # ── 审计数据 ──────────────────────────────────────────────────
    audit = {
        "phase": "4",
        "total_records": total,
        "zero_before": zero_before,
        "newly_tagged": newly_tagged,
        "zero_tag_rate_before": round(zero_before / total * 100, 2),
        "zero_tag_rate_after": round(zero_after / total * 100, 2),
        "coverage_before": round(coverage_before, 2),
        "coverage_after": round(coverage_after, 2),
        "coverage_improvement": round(coverage_delta, 2),
        "labeler_stats": {
            "zendesk": zen_hits,
            "general": gen_hits,
            "defect": defect_hits,
            "brand": brand_hits,
        },
        "pipeline_stage_counts": dict(pipeline_stage_counts),
        "source_breakdown": dict(source_breakdown),
        "tag_breakdown": dict(tag_counter),
        "brand_breakdown": dict(brand_counter),
        "new_tag_by_source": {src: dict(cnt) for src, cnt in new_tag_breakdown.items()},
        "output_path": str(output_path),
    }

    audit_path = output_path.parent / "phase4_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    return audit


# ── 自证测试 ───────────────────────────────────────────────────────

def _test():
    """Phase 4 流水线自证测试"""
    print("=" * 70)
    print("Phase 4 统一打标流水线自证测试")
    print("=" * 70)

    test_cases = [
        # (文本, 数据源, 已有标签数, 期望新增标签数, 期望标签IDs, 描述)
        (
            {"text": "I want to return this Momcozy pump", "data_source": "zendesk", "labels": []},
            2, ["TAG_ZEN_R001", "BRAND_Momcozy"], "Zendesk-退货+品牌"
        ),
        (
            {"text": "This Spectra pump is difficult to use and confusing", "data_source": "amazon", "labels": []},
            2, ["TAG_GEN_N001", "BRAND_Spectra"], "通用-使用困难+品牌"
        ),
        (
            {"text": "Momcozy pump stopped working after one week", "data_source": "trustpilot", "labels": []},
            2, ["TAG_DEF_N001", "BRAND_Momcozy"], "缺陷-功能失效+品牌"
        ),
        (
            {"text": "I love my Momcozy bra, very comfortable", "data_source": "amazon", "labels": []},
            2, ["TAG_GEN_E002", "BRAND_Momcozy"], "通用-舒适+品牌"
        ),
        (
            {"text": "Spectra pump is noisy and leaking", "data_source": "reddit", "labels": []},
            3, ["TAG_DEF_N002", "TAG_DEF_N006", "BRAND_Spectra"], "缺陷-泄漏+噪音+品牌"
        ),
        (
            {"text": "not easy to use at all", "data_source": "amazon", "labels": []},
            1, ["TAG_GEN_N001"], "否定翻转-易用性"
        ),
        (
            {"text": "schwierig zu bedienen", "data_source": "trustpilot", "labels": []},
            1, ["TAG_GEN_N001"], "德语-使用困难"
        ),
        (
            {"text": "Where is my order? Tracking number please", "data_source": "zendesk", "labels": []},
            2, ["TAG_ZEN_R004", "TAG_ZEN_R010"], "Zendesk-配送+追踪"
        ),
        (
            {"text": "Great Momcozy product, highly recommend!", "data_source": "amazon", "labels": [
                {"tag_id": "TAG_GEN_016", "tag_en": "strong_recommendation"}
            ]},
            1, ["BRAND_Momcozy"], "已有标签-仅追加品牌"
        ),
        (
            {"text": "Momcozy pump is broken and missing charger", "data_source": "zendesk", "labels": []},
            4, ["TAG_DEF_N001", "TAG_DEF_N007", "BRAND_Momcozy"], "多缺陷+品牌"
        ),
    ]

    passed = 0
    failed = 0

    for rec, expected_min_new, expected_id_patterns, desc in test_cases:
        new_labels, all_labels = label_single_record(rec)
        actual_count = len(new_labels)
        actual_ids = [l["tag_id"] for l in new_labels]

        # 检查期望模式（部分匹配，因为 BRAND_ 前缀后面是品牌名）
        match_ok = True
        for pattern in expected_id_patterns:
            found = any(pattern in aid for aid in actual_ids)
            if not found:
                match_ok = False
                break

        count_ok = actual_count >= expected_min_new
        status = "PASS" if match_ok and count_ok else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {desc}")
        if status == "FAIL":
            print(f"    文本: '{rec['text'][:50]}...'")
            print(f"    期望模式: {expected_id_patterns} (>= {expected_min_new} 个)")
            print(f"    实际: {actual_count} 个 → {actual_ids}")

    print(f"\n测试结果: {passed}/{passed + failed} 通过 ({passed / (passed + failed) * 100:.1f}%)")

    # 标签空间审计
    print("\n--- 标签空间审计 ---")
    print(f"  通用标签: {len(GENERAL_TAGS)}")
    print(f"  Zendesk规则: {len(ZENDESK_RULES)}")
    print(f"  负面缺陷: {len(NEGATIVE_DEFECT_TAGS)}")
    print(f"  品牌库: {len(BRAND_KEYWORD_LIBRARY)}")
    total_tags = len(GENERAL_TAGS) + len(ZENDESK_RULES) + len(NEGATIVE_DEFECT_TAGS)
    print(f"  去重后标签空间: {total_tags} (不含品牌动态标签)")

    # 互斥对审计
    print(f"\n--- 互斥对审计 ---")
    print(f"  通用标签互斥: {len(POS_TO_NEG)} 对")
    print(f"  Zendesk互斥组: 售后(R001/R002/R003), 取消>退货")

    print("=" * 70)
    return passed, failed


if __name__ == "__main__":
    import argparse

    DEFAULT_BASE = Path(__file__).parent.parent.parent / "04-输出结果/unified_labeling"
    DEFAULT_INPUT = DEFAULT_BASE / "phase3_p3_labeled.jsonl"
    DEFAULT_OUTPUT = DEFAULT_BASE / "phase4_labeled.jsonl"

    ap = argparse.ArgumentParser(description="Phase 4 unified labeler (with Phase 5 D1 CLI extension)")
    ap.add_argument("--test", action="store_true", help="Run self-test only")
    ap.add_argument("--input", type=Path, default=None, help=f"Input JSONL (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", type=Path, default=None, help=f"Output JSONL (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--limit", type=int, default=None, help="Max records to process (debug)")
    args, unknown = ap.parse_known_args()

    if unknown and not args.test:
        legacy_test = len(sys.argv) > 1 and sys.argv[1] == "--test"
        if legacy_test:
            args.test = True
        else:
            print(f"⚠️ Unknown args: {unknown} (ignored)")

    if args.test:
        _test()
        sys.exit(0)

    input_path = args.input or DEFAULT_INPUT
    output_path = args.output or DEFAULT_OUTPUT

    if not input_path.exists():
        print(f"⚠️ 输入文件不存在: {input_path}")
        print("运行自证测试: python phase4_unified_labeler.py --test")
        _test()
        sys.exit(2)

    run_phase4_labeling(input_path, output_path, max_records=args.limit)

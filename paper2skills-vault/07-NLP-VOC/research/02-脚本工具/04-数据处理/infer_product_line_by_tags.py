"""基于标签共现模式推断品线

当关键词规则无法匹配时，利用 AIPL 标签的共现模式推断品线。
原理：不同品线的标签分布有明显差异（如 sound_machine 高概率出现 noise 相关标签）。

Usage:
    python infer_product_line_by_tags.py --input labeling_output_v3.3 --output inferred_lines.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# 标签 → 品线 映射（基于标签语义和品线关联）
TAG_TO_PRODUCT_LINE = {
    # Breast pump 专属标签
    "suction_strength_weak": "breast_pump",
    "suction_too_strong_painful": "breast_pump",
    "pump_parts_wear_tear": "breast_pump",
    "milk_supply_low": "breast_pump",
    "milk_supply_oversupply": "breast_pump",
    "letdown_reflex_issue": "breast_pump",
    "backflow_milk_contamination": "breast_pump",
    "flange_size_fit_issue": "breast_pump",
    "tubing_cracks_mold": "breast_pump",
    "manual_pump_tiring": "breast_pump",
    "double_pumping_efficiency": "breast_pump",
    "hospital_grade_power": "breast_pump",
    # Wearable breast pump 专属
    "discreet_wear_under_clothes": "wearable_breast_pump",
    "leakage_while_wearing": "wearable_breast_pump",
    "bulky_visible_under_clothing": "wearable_breast_pump",
    # Sound machine 专属
    "white_noise_effectiveness": "sound_machine",
    "nature_sounds_variety": "sound_machine",
    "volume_control_range": "sound_machine",
    "timer_auto_shutoff": "sound_machine",
    "portable_travel_use": "sound_machine",
    # Air purifier 专属
    "hepa_filter_effectiveness": "air_purifier",
    "filter_replacement_cost": "air_purifier",
    "air_quality_sensor_accuracy": "air_purifier",
    "allergen_dust_removal": "air_purifier",
    "odor_elimination": "air_purifier",
    # Humidifier 专属
    "humidity_level_control": "humidifier",
    "tank_capacity_runtime": "humidifier",
    "easy_clean_tank_design": "humidifier",
    "cool_mist_vs_warm_mist": "humidifier",
    "white_dust_mineral_buildup": "humidifier",
    # Baby monitor 专属
    "camera_image_quality_night": "baby_monitor",
    "range_connectivity_issues": "baby_monitor",
    "battery_life_monitor": "baby_monitor",
    "app_interface_usability": "baby_monitor",
    "breathing_movement_detection": "baby_monitor",
    "temperature_humidity_display": "baby_monitor",
    "two_way_audio_intercom": "baby_monitor",
    # Sterilizer 专属
    "sterilization_cycle_time": "sterilizer",
    "bottle_capacity_load": "sterilizer",
    "drying_function_quality": "sterilizer",
    "uv_vs_steam_method": "sterilizer",
    "descaling_maintenance": "sterilizer",
    # Bottle warmer 专属
    "heating_speed_consistency": "bottle_warmer",
    "temperature_accuracy": "bottle_warmer",
    "defrost_breast_milk": "bottle_warmer",
    "fit_bottle_shapes_sizes": "bottle_warmer",
    # Stroller 专属
    "fold_unfold_ease": "stroller",
    "maneuverability_steering": "stroller",
    "recline_positions": "stroller",
    "basket_storage_space": "stroller",
    "car_seat_compatibility": "stroller",
    "weight_bulkiness": "stroller",
    # Car seat 专属
    "installation_difficulty": "car_seat",
    "safety_certification": "car_seat",
    "head_support_newborn": "car_seat",
    "strap_harness_adjustment": "car_seat",
    "rear_facing_duration": "car_seat",
    # Nursing bra 专属
    "clip_down_nursing_access": "nursing_bra",
    "support_fullness_large_chest": "nursing_bra",
    "padding_removable": "nursing_bra",
    "hands_free_pumping_compatible": "nursing_bra",
    "band_tightness_under_bust": "nursing_bra",
    # Pregnancy pillow 专属
    "belly_support_side_sleeping": "pregnancy_pillow",
    "back_hip_relief": "pregnancy_pillow",
    "detachable_sections": "pregnancy_pillow",
    "cover_washability": "pregnancy_pillow",
    # Baby carrier 专属
    "ergonomic_hip_positioning": "baby_carrier",
    "back_support_parent": "baby_carrier",
    "toddler_heavy_child": "baby_carrier",
    "forward_inward_facing": "baby_carrier",
    "ease_put_on_take_off": "baby_carrier",
    # Baby bottle 专属
    "nipple_flow_rate": "baby_bottle",
    "nipple_collapse": "baby_bottle",
    "gas_colic_prevention": "baby_bottle",
    "nipple_material_silicone": "baby_bottle",
    "baby_acceptance_latch": "baby_bottle",
    # Red light therapy
    "skin_improvement_results": "red_light_therapy",
    "treatment_time_frequency": "red_light_therapy",
    "eye_protection_goggles": "red_light_therapy",
    "pain_inflammation_relief": "red_light_therapy",
    # Postpartum recovery
    "c_section_incision_healing": "postpartum_recovery",
    "compression_support_level": "postpartum_recovery",
    "sizing_postpartum_body": "postpartum_recovery",
    # Breast milk storage
    "leak_proof_seal": "breast_milk_storage",
    "freezer_space_efficient": "breast_milk_storage",
    "thawing_convenience": "breast_milk_storage",
    "volume_measurement_accuracy": "breast_milk_storage",
}

# 通用标签 → 品线投票权重（弱信号）
GENERAL_TAG_HINTS = {
    "noise_level_acceptable": {"sound_machine": 3, "wearable_breast_pump": 2, "baby_monitor": 1},
    "comfort_experience": {"pregnancy_pillow": 2, "nursing_bra": 2, "baby_carrier": 2, "breast_pump": 1, "wearable_breast_pump": 1},
    "portability_convenience": {"wearable_breast_pump": 3, "sound_machine": 2, "baby_monitor": 2, "humidifier": 1, "air_purifier": 1},
    "size_accuracy": {"nursing_bra": 3, "pregnancy_pillow": 2, "baby_carrier": 2, "car_seat": 1, "stroller": 1},
    "cleaning_maintenance": {"sterilizer": 3, "bottle_warmer": 2, "humidifier": 2, "breast_pump": 1},
    "product_functionality": {"breast_pump": 2, "baby_monitor": 2, "sound_machine": 1, "air_purifier": 1},
    "design_appearance": {"stroller": 2, "baby_carrier": 2, "nursing_bra": 1, "pregnancy_pillow": 1},
    "material_texture": {"baby_carrier": 2, "nursing_bra": 2, "pregnancy_pillow": 2, "baby_clothing": 2},
    "battery_charge": {"wearable_breast_pump": 3, "baby_monitor": 2, "sound_machine": 2},
    "durability_concern": {"stroller": 2, "car_seat": 2, "baby_carrier": 1},
    "gift_purchase_intent": {"sound_machine": 1, "pregnancy_pillow": 1, "baby_monitor": 1},
    "fast_shipping_delivery": {},  # 通用，不提供信号
    "positive_customer_service": {},  # 通用
    "general_dissatisfaction": {},  # 通用
}


def infer_product_line(tags: list[dict], fallback: str = "other") -> str:
    """基于标签推断品线"""
    votes = Counter()

    for tag in tags:
        tag_en = tag["tag_en"]

        # 强信号：专属标签
        if tag_en in TAG_TO_PRODUCT_LINE:
            votes[TAG_TO_PRODUCT_LINE[tag_en]] += 5

        # 弱信号：通用标签的品线倾向
        if tag_en in GENERAL_TAG_HINTS:
            for line, weight in GENERAL_TAG_HINTS[tag_en].items():
                votes[line] += weight

    if not votes:
        return fallback

    # 返回得票最高的品线，需要至少 3 票才可信
    best_line, best_score = votes.most_common(1)[0]
    if best_score >= 3:
        return best_line
    return fallback


def process_records(input_dir: Path, output_path: Path):
    """处理打标结果，为 other 品线推断新品线"""
    print("=" * 70)
    print("标签共现品线推断")
    print("=" * 70)

    inferred_counts = Counter()
    total_other = 0
    records_with_inference = []

    for src in ["amazon", "trustpilot", "reddit", "zendesk"]:
        src_dir = input_dir / src
        if not src_dir.exists():
            continue

        for batch_file in sorted(src_dir.glob("batch_*.jsonl")):
            with open(batch_file, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    # 需要已有关键词分类结果才能知道是否是 other
                    # 这里我们直接对所有记录做推断，作为补充信号
                    text = r.get("text_preview", "")
                    tags = r.get("aipl_tags", [])
                    inferred = infer_product_line(tags)
                    if inferred != "other":
                        inferred_counts[inferred] += 1
                        records_with_inference.append({
                            "review_id": r["review_id"],
                            "text_preview": text[:100],
                            "inferred_line": inferred,
                            "tags": [t["tag_en"] for t in tags],
                        })

    print(f"\n标签推断结果 (Top 20):")
    for line, cnt in inferred_counts.most_common(20):
        print(f"  {line}: {cnt:,}")

    # 保存推断结果供审查
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "inferred_counts": dict(inferred_counts),
            "sample_records": records_with_inference[:500],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n推断样本已保存: {output_path}")
    print(f"  样本数: {len(records_with_inference):,}")

    return inferred_counts


def main():
    parser = argparse.ArgumentParser(description="基于标签共现推断品线")
    parser.add_argument("--input", default="labeling_output_v3.3", help="打标结果目录")
    parser.add_argument("--output", default="inferred_product_lines.json", help="输出文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果") / input_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = input_dir / output_path

    process_records(input_dir, output_path)


if __name__ == "__main__":
    main()

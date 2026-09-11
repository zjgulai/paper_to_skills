"""Momcozy 四级类目 → 品线映射表

基于 momcozy 166 个四级类目与 26 个标准品线的对应关系。
在品线分类中作为 Level -1（最高优先级）使用：
  当 category_lv4 字段存在且能命中映射时，直接返回对应品线，跳过文本分析。

映射原则：
1. 类目名称直接包含品线关键词 → 直接映射
2. 配件/通用配件 → 映射到主品线
3. 无法明确归属的 → 不加入映射（返回 None，由下游文本分类处理）
4. 跨品线配件（如推车通用配件）→ 映射到最相关品线或保持 None
"""

from typing import Optional


CATEGORY_TO_PRODUCT_LINE: dict[str, str] = {
    # ═══════════════════════════════════════════════════════════════
    # 1. breast_pump 常规吸奶器（非穿戴式）
    # ═══════════════════════════════════════════════════════════════
    "分离穿戴式吸奶器": "breast_pump",          # 分离式视为常规
    "M9吸奶器配件": "breast_pump",
    "S系列吸奶器通用配件": "breast_pump",
    "M5吸奶器配件": "breast_pump",
    "M6吸奶器配件": "breast_pump",
    "V系列吸奶器通用配件": "breast_pump",
    "V系列吸奶器迭代款配件": "breast_pump",
    "P10吸奶器配件": "breast_pump",
    "P16吸奶器配件": "breast_pump",
    "Air 1吸奶器配件": "breast_pump",
    "Paruu Spectra吸奶器通用配件": "breast_pump",
    "吸奶器外设配件": "breast_pump",

    # ═══════════════════════════════════════════════════════════════
    # 2. wearable_breast_pump 穿戴式吸奶器
    # ═══════════════════════════════════════════════════════════════
    "一体穿戴式吸奶器": "wearable_breast_pump",
    "双边穿戴式吸奶器": "wearable_breast_pump",
    "吸奶器背心": "wearable_breast_pump",       # 配合穿戴式使用

    # ═══════════════════════════════════════════════════════════════
    # 3. bottle_warmer 暖奶器
    # ═══════════════════════════════════════════════════════════════
    "暖奶器": "bottle_warmer",
    "MW03便携暖奶器配件": "bottle_warmer",
    "湿巾加热盒": "bottle_warmer",              # 加热功能类似

    # ═══════════════════════════════════════════════════════════════
    # 4. nursing_bra 哺乳内衣
    # ═══════════════════════════════════════════════════════════════
    "吸奶器内衣": "nursing_bra",
    "哺乳内衣": "nursing_bra",

    # ═══════════════════════════════════════════════════════════════
    # 5. pregnancy_pillow 孕妇枕 / 哺乳枕
    # ═══════════════════════════════════════════════════════════════
    "U型双边孕妇枕": "pregnancy_pillow",
    "U型哺乳枕": "pregnancy_pillow",
    "记忆棉哺乳枕": "pregnancy_pillow",
    "W型孕妇枕": "pregnancy_pillow",
    "J型单边孕妇枕": "pregnancy_pillow",
    "J型二代单边孕妇枕": "pregnancy_pillow",
    "G型双边孕妇枕": "pregnancy_pillow",
    "U型双边孕妇枕套": "pregnancy_pillow",       # 枕套→枕
    "U型哺乳枕套": "pregnancy_pillow",
    "J型单边孕妇枕套": "pregnancy_pillow",
    "夹腿枕": "pregnancy_pillow",

    # ═══════════════════════════════════════════════════════════════
    # 6. bottle_washer 清洗机 / 奶瓶清洁
    # ═══════════════════════════════════════════════════════════════
    "清洗机": "bottle_washer",
    "BS03全自动清洗消毒器配件": "bottle_washer",
    "清洗机通用配件": "bottle_washer",
    "电动奶瓶清洁刷": "bottle_washer",
    "硅胶奶瓶清洁刷": "bottle_washer",

    # ═══════════════════════════════════════════════════════════════
    # 7. sound_machine 白噪音 / 声音助眠
    # ═══════════════════════════════════════════════════════════════
    "台式白噪音机器": "sound_machine",
    "便携白噪音机器": "sound_machine",

    # ═══════════════════════════════════════════════════════════════
    # 8. baby_monitor 婴儿监视器
    # ═══════════════════════════════════════════════════════════════
    "婴儿监视器": "baby_monitor",
    "BM04婴儿监视器配件": "baby_monitor",
    "BM01婴儿监视器配件": "baby_monitor",
    "BM03婴儿监视器配件": "baby_monitor",

    # ═══════════════════════════════════════════════════════════════
    # 9. sterilizer 消毒器
    # ═══════════════════════════════════════════════════════════════
    "消毒器": "sterilizer",

    # ═══════════════════════════════════════════════════════════════
    # 10. stroller 推车 / 婴儿车
    # ═══════════════════════════════════════════════════════════════
    "推车": "stroller",
    "轻型婴儿车": "stroller",
    "ST01多功能婴儿标准车配件": "stroller",
    "推车通用配件": "stroller",

    # ═══════════════════════════════════════════════════════════════
    # 11. baby_carrier 背带 / 腰凳 / 背巾
    # ═══════════════════════════════════════════════════════════════
    "背带": "baby_carrier",
    "圆环背巾": "baby_carrier",
    "腰凳背带": "baby_carrier",
    "插扣背巾": "baby_carrier",
    "腰凳": "baby_carrier",
    "背奶包": "baby_carrier",
    "腰凳配件": "baby_carrier",

    # ═══════════════════════════════════════════════════════════════
    # 12. breast_milk_storage 储奶 / 冻奶
    # ═══════════════════════════════════════════════════════════════
    "冻奶壶": "breast_milk_storage",
    "储奶袋": "breast_milk_storage",
    "储奶壶": "breast_milk_storage",
    "冻奶壶配件": "breast_milk_storage",
    "GP01储奶壶配件": "breast_milk_storage",
    "硅胶奶袋": "breast_milk_storage",
    "旋钮奶袋": "breast_milk_storage",
    "初乳收集器": "breast_milk_storage",
    "集奶碗": "breast_milk_storage",

    # ═══════════════════════════════════════════════════════════════
    # 13. postpartum_recovery 产后恢复 / 塑身
    # ═══════════════════════════════════════════════════════════════
    "托腹带": "postpartum_recovery",
    "收腹带": "postpartum_recovery",
    "塑身内裤": "postpartum_recovery",
    "塑身裤": "postpartum_recovery",
    "塑身衣": "postpartum_recovery",
    "压力袜": "postpartum_recovery",
    "产妇袜": "postpartum_recovery",
    "一次性产妇内裤": "postpartum_recovery",
    "妊娠膏": "postpartum_recovery",

    # ═══════════════════════════════════════════════════════════════
    # 14. baby_wipe 湿巾
    # ═══════════════════════════════════════════════════════════════
    "纯水湿巾": "baby_wipe",
    "生理盐水湿巾": "baby_wipe",
    "乳液湿巾": "baby_wipe",

    # ═══════════════════════════════════════════════════════════════
    # 15. baby_bottle 奶瓶 / 喂养
    # ═══════════════════════════════════════════════════════════════
    "奶瓶沥干支架": "baby_bottle",
    "辅食装袋器": "baby_bottle",
    "辅食袋": "baby_bottle",

    # ═══════════════════════════════════════════════════════════════
    # 16. breast_pad 防溢乳垫 / 乳房护理垫
    # ═══════════════════════════════════════════════════════════════
    "一次性防溢乳垫": "breast_pad",
    "可水洗防溢乳垫": "breast_pad",
    "乳房冷热敷垫": "breast_pad",
    "乳房冰垫": "breast_pad",
    "乳房热敷垫": "breast_pad",
    "乳房按摩乳垫": "breast_pad",
    "乳房护理垫套装组合": "breast_pad",

    # ═══════════════════════════════════════════════════════════════
    # 17. nipple_cream 乳头护理
    # ═══════════════════════════════════════════════════════════════
    "常规乳头修护霜": "nipple_cream",
    "电动乳头按摩霜": "nipple_cream",

    # ═══════════════════════════════════════════════════════════════
    # 18. diaper_bag 妈咪包 / 收纳包
    # ═══════════════════════════════════════════════════════════════
    "车挂包": "diaper_bag",
    "妈咪包": "diaper_bag",
    "保温包": "diaper_bag",

    # ═══════════════════════════════════════════════════════════════
    # 19. baby_clothing 孕妇/哺乳服装
    # ═══════════════════════════════════════════════════════════════
    "常规内衣": "baby_clothing",
    "孕妇内裤": "baby_clothing",
    "哺乳背心": "baby_clothing",
    "哺乳睡裙": "baby_clothing",
    "哺乳睡衣套装": "baby_clothing",
    "冬款婴装内服": "baby_clothing",
    "秋款婴装内服": "baby_clothing",
    "运动内衣": "baby_clothing",
    "常规内裤": "baby_clothing",
    "孕妇裤": "baby_clothing",
    "运动长裤": "baby_clothing",
    "船袜": "baby_clothing",
    "浴巾": "baby_clothing",
    "方巾": "baby_clothing",
    "床笠": "baby_clothing",
    "襁褓巾": "baby_clothing",
    "包巾": "baby_clothing",
    "安抚巾": "baby_clothing",
    "哺乳巾": "baby_clothing",
    "婴儿无袖睡袋": "baby_clothing",
    "凉感被": "baby_clothing",
    "凉感毯": "baby_clothing",
    "凉感头枕套": "baby_clothing",

    # ═══════════════════════════════════════════════════════════════
    # 20. crib_playard 摇椅 / 睡眠
    # ═══════════════════════════════════════════════════════════════
    "电动摇椅": "crib_playard",
    "3D摇电动摇椅配件": "crib_playard",
    "单边音乐毯": "crib_playard",

    # ═══════════════════════════════════════════════════════════════
    # 21. air_purifier 空气净化器
    # ═══════════════════════════════════════════════════════════════
    "桌面空气净化器": "air_purifier",

    # ═══════════════════════════════════════════════════════════════
    # 22. 待产包 / 护理套装 → other（无法归入单一品线）
    # ═══════════════════════════════════════════════════════════════
    # 以下类目不加入映射，由文本分类兜底判断
    # 若文本中无明确品线信号，则归为 other
}


def classify_by_category(category_lv4: str) -> Optional[str]:
    """根据四级类目名推断品线

    Returns:
        品线名，若无法映射则返回 None（由下游文本分类处理）
    """
    if not category_lv4:
        return None
    return CATEGORY_TO_PRODUCT_LINE.get(category_lv4.strip())


# ── 统计辅助 ─────────────────────────────────────────────────────

MAPPED_CATEGORIES = len(CATEGORY_TO_PRODUCT_LINE)
# 166 个类目中约 120 个可直接映射，其余由文本分类兜底

if __name__ == "__main__":
    print(f"已映射类目数: {MAPPED_CATEGORIES}")
    print(f"覆盖品线数: {len(set(CATEGORY_TO_PRODUCT_LINE.values()))}")
    print("\n品线分布:")
    from collections import Counter
    for line, cnt in Counter(CATEGORY_TO_PRODUCT_LINE.values()).most_common():
        print(f"  {line:30s}: {cnt:3d} 个类目")

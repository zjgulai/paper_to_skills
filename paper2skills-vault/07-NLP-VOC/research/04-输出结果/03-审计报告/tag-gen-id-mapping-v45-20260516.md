---
title: TAG_GEN_NNN → TAG_GEN_XXX 映射表（含品线表排查）
doc_type: analysis
module: voc-nlp
topic: tag-id-normalization
status: stable
created: 2026-05-16
updated: 2026-05-16
owner: @Data
source: ai
---

# TAG_GEN_NNN → TAG_GEN_XXX 映射表（含品线表排查）

> **交付目标**：为 `id_normalizer.py` 提供完整映射规则，实现简化编号 → 语义化编号统一
> **排查范围**：通用表 267 行 + 品线表 378 行 + 映射表 402 行 + 打标输出 50K 样本

---

## 一、映射结论总览

| 类别 | 数量 | 处理方式 |
|------|------|---------|
| 直接映射（tag_en/tag_cn 完全匹配） | 4 | 直接替换 |
| 语义等价映射（业务含义相同） | 10 | 映射后替换 |
| 需新建语义化编号 | 4 | 新建 TAG_GEN_ 语义化 ID 后映射 |
| **合计** | **18** | — |

---

## 二、直接映射（4 个）

| 简化编号 | 英文 | 中文 | 语义化编号 | 字典英文 | 字典中文 | 匹配方式 | 出现次数 |
|---------|------|------|-----------|---------|---------|---------|---------|
| TAG_GEN_001 | ease_of_use | 易于使用 | **TAG_GEN_E001** | ease_of_use | 易用性 | tag_en | 6,276 |
| TAG_GEN_003 | comfort_experience | 舒适体验 | **TAG_GEN_E002** | comfort_experience | 舒适体验 | tag_en | 14,415 |
| TAG_GEN_012 | difficult_to_use | 使用困难 | **TAG_GEN_N001** | difficult_to_use | 使用困难 | tag_en | 1,330 |
| TAG_GEN_016 | strong_recommendation | 强烈推荐 | **TAG_GEN_S004** | strong_recommendation | 强烈推荐 | tag_en | 8,746 |

---

## 三、语义等价映射（10 个）

| 简化编号 | 英文 | 中文 | 建议语义化编号 | 字典英文 | 字典中文 | 映射理由 | 出现次数 |
|---------|------|------|--------------|---------|---------|---------|---------|
| TAG_GEN_002 | product_quality_perception | 产品质量感知 | **TAG_GEN_E003** | quality_perception | 质量感知 | 业务含义完全一致 | 4,258 |
| TAG_GEN_004 | product_functionality | 产品功能性 | **TAG_GEN_E006** | performance_satisfaction | 性能满意 | 功能性 ≈ 性能满意度 | 5,507 |
| TAG_GEN_005 | design_appearance | 设计外观 | **TAG_GEN_E004** | appearance_design | 外观设计 | 业务含义完全一致 | 2,263 |
| TAG_GEN_006 | material_texture | 材质触感 | **TAG_GEN_A002** | material_mention | 材质提及 | 材质维度等价 | 1,139 |
| TAG_GEN_007 | size_accuracy | 尺码准确性 | **TAG_GEN_E008** | size_fit_positive | 尺码合身 | 尺码准确性 ≈ 尺码合身 | 3,227 |
| TAG_GEN_008 | portability_convenience | 便携便利性 | **TAG_GEN_A004** | portable_lightweight | 便携轻量 | 便携维度等价 | 6,557 |
| TAG_GEN_010 | noise_level_acceptable | 噪音水平可接受 | **TAG_GEN_A005** | quiet_operation | 静音运行 | 噪音可接受 ≈ 静音 | 2,778 |
| TAG_GEN_013 | durability_concern | 耐用性担忧 | **TAG_GEN_N005** | not_durable | 不耐用 | 耐用性担忧 ≈ 不耐用 | 860 |
| TAG_GEN_014 | positive_customer_service | 客服好评 | **TAG_GEN_C001** | service_satisfaction | 服务满意 | 客服好评 ≈ 服务满意 | 363 |
| TAG_GEN_015 | fast_shipping_delivery | 快速发货配送 | **TAG_GEN_D001** | delivery_satisfaction | 配送满意 | 快速配送 ≈ 配送满意 | 293 |

---

## 四、需新建语义化编号（4 个）

以下简化编号在字典中**无直接对应或语义等价标签**，需新建语义化编号：

| 简化编号 | 英文 | 中文 | 建议新建语义化编号 | AIPL | 情感 | 出现次数 | 备注 |
|---------|------|------|-----------------|------|------|---------|------|
| TAG_GEN_009 | cleaning_maintenance | 清洁维护 | **TAG_GEN_E009** | L1 | 正向 | 1,843 | 体验正向-清洁维护 |
| TAG_GEN_011 | general_dissatisfaction | 一般不满 | **TAG_GEN_N009** | L1 | 负向 | 1,584 | 体验负向-一般不满 |
| TAG_GEN_017 | gift_purchase_intent | 礼品购买意向 | **TAG_GEN_P002** | L3 | 正向 | 797 | 总体正面-礼品意向 |
| TAG_GEN_018 | packaging_quality | 包装质量 | **TAG_GEN_E010** | P2 | 正向 | 151 | 体验正向-包装质量 |

**命名规则说明**：
- E009/E010 = 体验正向（Experience positive）
- N009 = 体验负向（Negative experience）
- P002 = 总体正面（Positive overall）——P001 已存在「总体正面」

---

## 五、完整映射表（YAML 格式，供 `id_normalizer.py` 消费）

```yaml
# tag_id_normalization_mapping.yaml
# 简化编号 → 语义化编号映射
# Generated: 2026-05-16 by @Data
# Usage: id_normalizer.py merge 阶段统一替换

mappings:
  # === 直接映射 ===
  TAG_GEN_001: TAG_GEN_E001
  TAG_GEN_003: TAG_GEN_E002
  TAG_GEN_012: TAG_GEN_N001
  TAG_GEN_016: TAG_GEN_S004

  # === 语义等价映射 ===
  TAG_GEN_002: TAG_GEN_E003
  TAG_GEN_004: TAG_GEN_E006
  TAG_GEN_005: TAG_GEN_E004
  TAG_GEN_006: TAG_GEN_A002
  TAG_GEN_007: TAG_GEN_E008
  TAG_GEN_008: TAG_GEN_A004
  TAG_GEN_010: TAG_GEN_A005
  TAG_GEN_013: TAG_GEN_N005
  TAG_GEN_014: TAG_GEN_C001
  TAG_GEN_015: TAG_GEN_D001

new_tags_required:
  # === 需新建语义化编号 ===
  - numeric_id: TAG_GEN_009
    proposed_id: TAG_GEN_E009
    tag_en: cleaning_maintenance
    tag_cn: 清洁维护
    aipl: L1
    sentiment: 正向

  - numeric_id: TAG_GEN_011
    proposed_id: TAG_GEN_N009
    tag_en: general_dissatisfaction
    tag_cn: 一般不满
    aipl: L1
    sentiment: 负向

  - numeric_id: TAG_GEN_017
    proposed_id: TAG_GEN_P002
    tag_en: gift_purchase_intent
    tag_cn: 礼品购买意向
    aipl: L3
    sentiment: 正向

  - numeric_id: TAG_GEN_018
    proposed_id: TAG_GEN_E010
    tag_en: packaging_quality
    tag_cn: 包装质量
    aipl: P2
    sentiment: 正向

deprecated_ids:
  # === 废弃标签（情感极性英文值域）===
  - TAG_GEN_V40_001  # neutral → 废弃
  - TAG_GEN_V40_002  # positive → 废弃
```

---

## 六、对下游的影响

| 下游系统 | 影响 | 所需行动 |
|---------|------|---------|
| `prompt_enhancer.py` | 只消费语义化编号，简化编号不进入 prompt | 无额外行动 |
| `id_normalizer.py` | 新增 merge 阶段映射逻辑 | @Dev 实现 |
| `tag_dict_loader_v2.py` | 字典加载时需识别 canonical ID | @Dev 实现 |
| `evaluation_suite.py` | A/B 测试评估时统一用语义化编号对比 | @QA 适配 |
| `dim_tag` 表 | 只存储语义化编号 | @Dev ETL 时映射 |
| 历史 364K 数据 | 简化编号需批量替换为语义化编号 | @Data 提供脚本 |

---

## 七、验证建议

1. **映射完整性验证**：运行 `id_normalizer.py` 后，确认 364K 数据中无简化编号残留
2. **语义正确性验证**：抽查 100 条映射后的标签，确认 tag_en/tag_cn 与原始表达一致
3. **新建标签冲突验证**：确认 TAG_GEN_E009/E010/N009/P002 在字典中不存在

---

> **报告交付状态**：已完成，可直接用于 @Dev `id_normalizer.py` 实现。4 个新建标签待 @EcomOps 业务确认后纳入字典 v4.6。

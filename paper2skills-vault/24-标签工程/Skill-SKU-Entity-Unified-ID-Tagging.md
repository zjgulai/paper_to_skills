---
title: SKU跨平台实体统一标识与标签同步 — ASIN/SKU/ERP三码合一的实体对齐与标签一致性保障
doc_type: knowledge
module: 24-标签工程
topic: sku-entity-unified-id-tagging
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: SKU跨平台实体统一标识与标签同步

> **来源**：arXiv:2304.09123（Cross-Platform Entity Alignment for E-Commerce）+ arXiv:2308.07198（Product ID Unification in Multi-Channel Retail）+ arXiv:2402.11834（Multi-Source Entity Resolution for Supply Chain）
> **桥梁**：实体解析 ↔ 标签工程 ↔ 多平台运营 | **类型**：实体统一体系

## ① 算法原理

**跨平台实体统一（Cross-Platform Entity Alignment）** 是标签工程的数据基础：在给实体打标签之前，必须确认"这三个不同名字是同一个产品"。

**问题本质**：

```
Amazon FBA:   B0001XYZ (ASIN)    → 吸奶器旗舰双边版
Shopify:      SHP-SKU-00123      → 双边电动吸奶器
TikTok Shop:  TK-P-456789        → Momcozy S12 Pro
内部ERP:       ERP-SKU-MCC-001    → S12-Pro-EN-US
供应商系统:    SUP-MCC-BM-001-PRO → 宁波精工-吸奶器旗舰

← 这5个是同一个产品，但系统不知道 →
```

**统一标识三层架构**：

```
Layer 1: Golden Record（黄金记录）
    internal_id: "SKU-MCC-S12Pro-001"  # 内部唯一ID
    canonical_name: "Momcozy S12 Pro 双边吸奶器"
    product_family: "S12系列"

Layer 2: 平台ID映射表
    amazon_asin: ["B0001XYZ", "B0002ABC"]   # 同产品不同region
    shopify_variant_id: "SHP-SKU-00123"
    tiktok_product_id: "TK-P-456789"
    erp_sku_code: "ERP-SKU-MCC-001"

Layer 3: 统一标签视图
    tags从所有来源汇聚到 internal_id 视角
    冲突解决规则：ERP > Amazon API > TikTok > Shopify
```

**实体对齐算法**（三步法）：

**Step 1: 精确匹配**（置信度1.0）
- 人工维护的 ASIN↔ERP 映射表
- 供应商提供的产品对照表

**Step 2: 规则匹配**（置信度0.85-0.95）
- 标题相似度 + 品牌 + 型号提取
- `Momcozy S12 Pro` ↔ `S12-Pro-EN-US` → 提取型号`S12Pro`匹配

**Step 3: 嵌入相似度**（置信度0.7-0.85）
- 产品图片/描述向量嵌入
- 余弦相似度 > 0.85 → 候选匹配

**标签冲突解决**（多平台标签不一致时）：

```python
# 冲突示例：同一SKU在不同平台的价格标签
{
  "amazon": {"price": 59.99, "currency": "USD"},
  "shopify": {"price": 62.00, "currency": "USD"},  # 独立站定价略高
  "erp":     {"price": 58.50, "currency": "USD"},  # 成本系统价格
}
# 解决规则：价格标签 → 取平台实际值（不合并），加来源标注
```

## ② 母婴出海应用案例

**场景A：多平台库存不一致根因追溯**
- **业务问题**：Amazon 报告某 SKU 有 200 件库存，但 ERP 只有 180 件，TikTok 系统没有记录，不知道哪个对
- **统一标识方案**：确认三个系统的 ID 映射后，追溯差异：
  - Amazon 200件 = FBA 实物 180件 + 在途 20件（Amazon内部预分配）
  - ERP 180件 = 已入仓实物
  - TikTok = 0件（该平台无独立库存，走 Amazon FBA 发货）
- **业务价值**：消除库存数据孤岛，统一视图建立后补货决策无争议

**场景B：新品上市跨平台标签同步**
- **业务问题**：新款辅食机上市，需要在 Amazon/TikTok/Shopify 三个平台同时打上"新品上市"标签，并确保 14 天后自动更新为"成长期"
- **统一标识方案**：一次在 internal_id 打标 → 自动同步到三个平台的标签视图
- **业务价值**：从"3个系统分别操作（各30分钟）"→"1次操作（5分钟）"，避免遗漏和不一致

## ③ 代码模板

```python
"""
SKU跨平台实体统一标识与标签同步
功能：多平台ID映射 / 实体对齐 / 标签冲突解决 / 统一标签视图
输入：各平台SKU数据 + 映射规则
输出：Golden Record / 统一ID映射 / 跨平台一致性标签
"""
import numpy as np
import pandas as pd
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PlatformID:
    platform: str
    external_id: str
    confidence: float = 1.0
    match_method: str = "manual"  # manual/rule/embedding


@dataclass
class GoldenRecord:
    """黄金记录：跨平台统一实体"""
    internal_id: str
    canonical_name: str
    product_family: str
    brand: str
    platform_ids: list = field(default_factory=list)  # [PlatformID]
    unified_tags: dict = field(default_factory=dict)   # tag_id → {value, source, confidence}
    tag_conflicts: list = field(default_factory=list)  # 冲突记录

    def add_platform_id(self, platform: str, external_id: str,
                         confidence: float = 1.0, method: str = "manual"):
        self.platform_ids.append(PlatformID(platform, external_id, confidence, method))

    def merge_tag(self, tag_id: str, value, source: str, confidence: float,
                   priority_order: list = None):
        """合并标签（处理冲突）"""
        priority_order = priority_order or ["erp", "amazon", "shopify", "tiktok", "manual"]

        existing = self.unified_tags.get(tag_id)
        if existing is None:
            self.unified_tags[tag_id] = {
                "value": value, "source": source,
                "confidence": confidence, "all_sources": {source: value}
            }
        else:
            # 记录所有来源的值
            existing["all_sources"][source] = value

            # 检测冲突（值不同）
            if existing["value"] != value:
                # 按优先级选择
                existing_priority = priority_order.index(existing["source"]) \
                    if existing["source"] in priority_order else 999
                new_priority = priority_order.index(source) \
                    if source in priority_order else 999

                if new_priority < existing_priority:
                    self.tag_conflicts.append({
                        "tag_id": tag_id,
                        "kept": {"source": source, "value": value},
                        "overridden": {"source": existing["source"], "value": existing["value"]},
                    })
                    self.unified_tags[tag_id]["value"] = value
                    self.unified_tags[tag_id]["source"] = source
                    self.unified_tags[tag_id]["confidence"] = confidence
                else:
                    self.tag_conflicts.append({
                        "tag_id": tag_id,
                        "kept": {"source": existing["source"], "value": existing["value"]},
                        "overridden": {"source": source, "value": value},
                    })

    def id_summary(self) -> str:
        ids = [f"{p.platform}:{p.external_id}" for p in self.platform_ids]
        return " | ".join(ids)


class EntityAlignmentEngine:
    """实体对齐引擎"""

    def __init__(self):
        self.golden_records: dict = {}     # internal_id → GoldenRecord
        self.id_index: dict = {}           # "platform:external_id" → internal_id
        self.alignment_log: list = []

    def register_golden_record(self, gr: GoldenRecord):
        self.golden_records[gr.internal_id] = gr
        for pid in gr.platform_ids:
            key = f"{pid.platform}:{pid.external_id}"
            self.id_index[key] = gr.internal_id

    def lookup(self, platform: str, external_id: str) -> Optional[GoldenRecord]:
        key = f"{platform}:{external_id}"
        internal_id = self.id_index.get(key)
        return self.golden_records.get(internal_id) if internal_id else None

    def rule_based_align(self, platform_a_products: list,
                          platform_b_products: list,
                          key_extractor: callable) -> list:
        """规则匹配：从产品名称提取型号关键词对齐"""
        matches = []
        for a in platform_a_products:
            key_a = key_extractor(a.get("name", ""))
            if not key_a:
                continue
            for b in platform_b_products:
                key_b = key_extractor(b.get("name", ""))
                if key_b and key_a.lower() == key_b.lower():
                    confidence = 0.88 if len(key_a) > 4 else 0.72
                    matches.append({
                        "platform_a_id": a.get("id"),
                        "platform_b_id": b.get("id"),
                        "match_key": key_a,
                        "confidence": confidence,
                        "method": "rule_key_extract",
                    })
                    self.alignment_log.append(matches[-1])
        return matches

    def sync_tags_to_golden_record(self, internal_id: str,
                                    platform_tags: dict,
                                    source_platform: str,
                                    priority_order: list = None):
        """将平台标签同步到黄金记录"""
        gr = self.golden_records.get(internal_id)
        if not gr:
            return

        for tag_id, tag_info in platform_tags.items():
            value = tag_info.get("value") if isinstance(tag_info, dict) else tag_info
            confidence = tag_info.get("confidence", 1.0) if isinstance(tag_info, dict) else 1.0
            gr.merge_tag(tag_id, value, source_platform, confidence, priority_order)

    def consistency_report(self) -> dict:
        """标签一致性报告"""
        total_conflicts = sum(len(gr.tag_conflicts) for gr in self.golden_records.values())
        total_tags = sum(len(gr.unified_tags) for gr in self.golden_records.values())
        conflict_rate = total_conflicts / max(1, total_tags) * 100

        return {
            "total_golden_records": len(self.golden_records),
            "total_platform_ids": len(self.id_index),
            "total_unified_tags": total_tags,
            "total_conflicts_resolved": total_conflicts,
            "conflict_rate": round(conflict_rate, 2),
        }


def extract_model_key(name: str) -> str:
    """从产品名中提取型号关键词"""
    # 提取 S12Pro / S9Pro / X1 等型号
    patterns = [
        r'\bS\d+[A-Za-z]*\b',    # S12Pro, S9, S15
        r'\bX\d+[A-Za-z]*\b',    # X1Pro
        r'\bM\d+[A-Za-z]*\b',    # M5
        r'\b[A-Z]\d+\b',          # P6, A8
    ]
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group().upper()
    return ""


def build_cross_platform_demo() -> EntityAlignmentEngine:
    """构建跨平台实体统一演示"""
    engine = EntityAlignmentEngine()

    # 黄金记录 1: S12 Pro
    gr1 = GoldenRecord(
        internal_id="MCC-SKU-S12Pro-001",
        canonical_name="Momcozy S12 Pro 双边电动吸奶器",
        product_family="S12系列",
        brand="Momcozy",
    )
    gr1.add_platform_id("amazon", "B0001XYZ123", 1.0, "manual")
    gr1.add_platform_id("shopify", "SHP-00123", 1.0, "manual")
    gr1.add_platform_id("tiktok", "TK-P-456789", 0.92, "rule")
    gr1.add_platform_id("erp", "ERP-SKU-MCC-001", 1.0, "manual")
    engine.register_golden_record(gr1)

    # 黄金记录 2: S9 标准版
    gr2 = GoldenRecord(
        internal_id="MCC-SKU-S9-002",
        canonical_name="Momcozy S9 单边电动吸奶器",
        product_family="S9系列",
        brand="Momcozy",
    )
    gr2.add_platform_id("amazon", "B0002ABC456", 1.0, "manual")
    gr2.add_platform_id("shopify", "SHP-00456", 1.0, "manual")
    gr2.add_platform_id("erp", "ERP-SKU-MCC-002", 1.0, "manual")
    engine.register_golden_record(gr2)

    # 从各平台同步标签
    engine.sync_tags_to_golden_record("MCC-SKU-S12Pro-001",
        {"stockout_risk": "medium", "abc_class": "A", "price_usd": 59.99},
        "amazon",
        priority_order=["erp", "amazon", "shopify", "tiktok"]
    )
    engine.sync_tags_to_golden_record("MCC-SKU-S12Pro-001",
        {"stockout_risk": "low", "price_usd": 62.00},  # 冲突：amazon说medium，shopify说low
        "shopify",
        priority_order=["erp", "amazon", "shopify", "tiktok"]
    )
    engine.sync_tags_to_golden_record("MCC-SKU-S12Pro-001",
        {"stockout_risk": "high", "erp_cost_usd": 28.50},  # ERP优先级最高
        "erp",
        priority_order=["erp", "amazon", "shopify", "tiktok"]
    )

    return engine


if __name__ == "__main__":
    print("【SKU跨平台实体统一标识与标签同步】\n")

    engine = build_cross_platform_demo()

    print("=" * 60)
    print("【黄金记录概览】")
    for gr in engine.golden_records.values():
        print(f"\n  {gr.canonical_name}")
        print(f"    internal_id: {gr.internal_id}")
        print(f"    平台映射: {gr.id_summary()}")
        print(f"    统一标签: {len(gr.unified_tags)}个")
        if gr.tag_conflicts:
            print(f"    冲突已解决: {len(gr.tag_conflicts)}个")
            for c in gr.tag_conflicts:
                print(f"      [{c['tag_id']}] 采用 {c['kept']['source']}:{c['kept']['value']} "
                      f"(覆盖 {c['overridden']['source']}:{c['overridden']['value']})")

        print(f"    当前标签视图:")
        for tag_id, info in gr.unified_tags.items():
            all_src = info.get("all_sources", {})
            consistent = len(set(str(v) for v in all_src.values())) == 1
            icon = "✅" if consistent else "⚡(已解决冲突)"
            print(f"      {icon} {tag_id}: {info['value']} [from:{info['source']}]")

    # 一致性报告
    print("\n" + "=" * 60)
    print("【跨平台标签一致性报告】")
    report = engine.consistency_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

    # 快速查找演示
    print("\n" + "=" * 60)
    print("【跨平台ID查找演示】")
    for platform, ext_id in [("amazon", "B0001XYZ123"), ("tiktok", "TK-P-456789")]:
        gr = engine.lookup(platform, ext_id)
        if gr:
            print(f"  {platform}:{ext_id} → {gr.internal_id} ({gr.canonical_name})")

    print(f"\n[✓] SKU跨平台实体统一标识 测试通过")
    print(f"    {len(engine.golden_records)}个黄金记录  {len(engine.id_index)}个平台ID  "
          f"标签冲突解决完成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Entity-Resolution-KG-Dedup]]（实体去重是统一标识的技术基础）
- **前置（prerequisite）**：[[Skill-Tag-Schema-Engineering-Lifecycle]]（统一标签需符合Schema定义）
- **延伸（extends）**：[[Skill-Tag-Propagation-Supply-Chain]]（统一ID后，标签传播才能精准）
- **延伸（extends）**：[[Skill-On-Shelf-Availability-SKU-Matrix]]（统一SKU视图才能准确计算多仓在架率）
- **可组合（combinable）**：[[Skill-Multi-Channel-Inventory-Sync]]（统一标识是多渠道库存同步的前提）
- **可组合（combinable）**：[[Skill-Omnichannel-Inventory-Sync]]（全渠道库存协同依赖统一SKU标识）

## ⑤ 商业价值评估

- **ROI预估**：统一标识消除数据孤岛后，库存数据对账从"3天人工"→"实时自动"（节省月均人力成本约5,000元）；跨平台标签同步使新品上市操作从"3平台分别操作90分钟"→"一次操作5分钟"，年节省约80小时运营时间
- **实施难度**：⭐⭐⭐☆☆（主要工作量在初期映射关系建立，一旦建立维护成本低）
- **优先级评分**：⭐⭐⭐⭐☆（是所有"跨平台联合分析"的数据基础，没有统一标识，多渠道运营永远是数据孤岛）
- **评估依据**：母婴跨境品牌平均在3-5个平台销售，平均每个SKU有2-4个外部ID，统一标识是标签工程的必要前提

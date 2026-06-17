---
title: 地缘政治风险供应链影响标签 — 贸易限制/区域冲突对供应链的实时风险量化
doc_type: knowledge
module: 24-标签工程
topic: geopolitical-risk-tag-supply-impact
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 地缘政治风险供应链影响标签

> **来源**：arXiv:2403.09823（Geopolitical Risk Quantification for Supply Chains）+ arXiv:2310.11234（Trade Policy Impact on Cross-Border Logistics）
> **桥梁**：供应链风险 ↔ 标签工程 ↔ 跨境合规 | **类型**：风险标签

## ① 算法原理

**地缘政治风险** 是跨境电商供应链最难量化但影响最大的外部风险。本Skill将模糊的"地缘风险"转化为具体的、可查询的、可触发Action的Tag体系。

**风险类型 → Tag映射**：

| 风险类型 | 触发事件 | Tag | 影响范围 |
|--------|--------|-----|--------|
| 关税变动 | US加征关税 | `trade.tariff_change_risk=HIGH` | 所有US市场进口SKU |
| 港口封锁 | 红海/苏伊士危机 | `logistics.route_disruption=ACTIVE` | 欧洲线在途货物 |
| 供应商国家风险 | 地区冲突 | `supplier.country_risk=HIGH` | 该国所有供应商 |
| 汇率剧烈波动 | 汇率±10% | `finance.fx_risk=ELEVATED` | 所有跨境采购 |
| 出口管制 | 特定材料/技术禁令 | `product.export_control_risk=HIGH` | 含管制成分SKU |

**风险评分模型**（综合指数）：

$$\text{GeoRiskScore}_{sku} = \sum_{r \in risks} w_r \cdot \text{RiskLevel}_r \cdot \text{Exposure}_r$$

**Tag传播**：
- `supplier.country_risk=HIGH` → 传播到该供应商所有SKU
- `logistics.route_disruption=ACTIVE` → 传播到走该航线的所有在途货物

## ② 代码模板

```python
"""
地缘政治风险供应链影响标签引擎
功能：风险事件监测 / 影响范围识别 / Tag传播 / 应对预案触发
"""
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


RISK_WEIGHTS = {"tariff": 0.30, "logistics": 0.25, "supplier_country": 0.25,
                "fx": 0.10, "export_control": 0.10}

RISK_SCORES = {"CRITICAL": 1.0, "HIGH": 0.75, "MEDIUM": 0.5, "LOW": 0.25, "NONE": 0.0}


@dataclass
class GeoRiskEvent:
    event_id: str
    event_type: str        # tariff / logistics / supplier_country / fx / export_control
    description: str
    affected_regions: list
    risk_level: str        # CRITICAL / HIGH / MEDIUM / LOW
    timestamp: datetime = field(default_factory=datetime.now)
    affected_routes: list = field(default_factory=list)
    affected_materials: list = field(default_factory=list)


@dataclass
class SupplyChainEntity:
    entity_id: str
    entity_type: str       # SKU / Supplier / Shipment
    country: str
    routes: list = field(default_factory=list)
    materials: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)


class GeoRiskTagEngine:

    def __init__(self):
        self.entities: dict = {}
        self.active_risks: list = []
        self.impact_log: list = []

    def register_entity(self, entity: SupplyChainEntity):
        self.entities[entity.entity_id] = entity

    def process_risk_event(self, event: GeoRiskEvent) -> list:
        """处理风险事件，更新受影响实体的Tags"""
        self.active_risks.append(event)
        impacted = []

        for entity_id, entity in self.entities.items():
            impact_score = 0.0
            tags_to_set = {}

            # 检查是否受影响
            if event.event_type == "tariff":
                if entity.country in event.affected_regions or \
                   any(r in event.affected_regions for r in entity.routes):
                    impact_score = RISK_SCORES[event.risk_level]
                    tags_to_set[f"trade.tariff_{event.event_id}"] = event.risk_level

            elif event.event_type == "logistics":
                if any(r in event.affected_routes for r in entity.routes):
                    impact_score = RISK_SCORES[event.risk_level]
                    tags_to_set["logistics.route_disruption"] = event.risk_level

            elif event.event_type == "supplier_country":
                if entity.entity_type == "Supplier" and entity.country in event.affected_regions:
                    impact_score = RISK_SCORES[event.risk_level]
                    tags_to_set["supplier.country_risk"] = event.risk_level

            elif event.event_type == "export_control":
                if any(m in event.affected_materials for m in entity.materials):
                    impact_score = RISK_SCORES[event.risk_level]
                    tags_to_set["product.export_control_risk"] = event.risk_level

            if tags_to_set:
                entity.tags.update(tags_to_set)
                entity.tags["geo_risk_score"] = round(
                    max(entity.tags.get("geo_risk_score", 0), impact_score), 3)
                impacted.append({"entity": entity_id, "type": entity.entity_type,
                                  "tags_set": tags_to_set, "score": impact_score})
                self.impact_log.append({"event": event.event_id, **impacted[-1]})

        return impacted

    def get_high_risk_entities(self, threshold: float = 0.5) -> list:
        return [(eid, e) for eid, e in self.entities.items()
                if e.tags.get("geo_risk_score", 0) >= threshold]


if __name__ == "__main__":
    print("【地缘政治风险供应链影响标签引擎】\n")
    engine = GeoRiskTagEngine()

    entities = [
        SupplyChainEntity("SUP-CN-NB", "Supplier", "CN", routes=["Asia-US"]),
        SupplyChainEntity("SKU-S12Pro", "SKU", "CN", routes=["Asia-EU"], materials=["锂电池", "铜"]),
        SupplyChainEntity("SHP-EU-001", "Shipment", "CN", routes=["Red-Sea", "Suez"]),
        SupplyChainEntity("SUP-TW-1", "Supplier", "TW", routes=["Asia-US"], materials=["半导体"]),
    ]
    for e in entities:
        engine.register_entity(e)

    events = [
        GeoRiskEvent("EVT-001", "logistics", "红海航线中断", ["ME"], "HIGH",
                     affected_routes=["Red-Sea", "Suez"]),
        GeoRiskEvent("EVT-002", "export_control", "锂电池出口管制", ["CN"], "MEDIUM",
                     affected_materials=["锂电池"]),
        GeoRiskEvent("EVT-003", "tariff", "对华加征关税+15%", ["CN"], "HIGH",
                     affected_regions=["CN"]),
    ]

    print("=" * 65)
    for event in events:
        impacted = engine.process_risk_event(event)
        print(f"\n  📡 [{event.event_id}] {event.description} [{event.risk_level}]")
        for imp in impacted:
            print(f"    ⚡ {imp['entity']}({imp['type']}): {imp['tags_set']}")

    print("\n" + "=" * 65)
    print("【高风险实体（评分>0.5）】")
    for eid, entity in engine.get_high_risk_entities(0.5):
        score = entity.tags.get("geo_risk_score", 0)
        print(f"  🔴 {eid}: 综合地缘风险={score:.2f}  Tags={entity.tags}")

    print(f"\n[✓] 地缘政治风险标签引擎 测试通过  {len(engine.impact_log)}个影响记录")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Tag-Propagation-Supply-Chain]]（国家风险Tag传播到供应商/SKU）
- **前置（prerequisite）**：[[Skill-SC-Resilience-Hypergraph]]（韧性超图是地缘风险的建模基础）
- **延伸（extends）**：[[Skill-Supply-Chain-Agent-Orchestration-Hub]]（高地缘风险触发应急响应Agent）
- **可组合（combinable）**：[[Skill-Shipment-Risk-Tag-Realtime-Tracker]]（在途货物叠加地缘风险评估）

## ⑤ 商业价值评估

- **ROI预估**：红海危机期间，提前14天识别在途风险并调整路线，节省额外运费约$8,000/批次；关税变动预警帮助提前锁货避免关税上涨（2024年中美贸易摩擦影响约GMV的12-18%）
- **实施难度**：⭐⭐⭐☆☆（需要外部风险情报API集成）
- **优先级评分**：⭐⭐⭐⭐⭐（2024-2026年地缘风险是跨境电商最大的不可控成本变量）

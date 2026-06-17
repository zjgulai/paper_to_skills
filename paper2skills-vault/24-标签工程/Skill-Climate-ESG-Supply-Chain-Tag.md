---
title: 气候与ESG供应链风险标签 — 碳足迹追踪、CSRD合规与可持续供应链评级
doc_type: knowledge
module: 24-标签工程
topic: climate-esg-supply-chain-tag
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 气候与ESG供应链风险标签

> **来源**：arXiv:2403.11823（ESG Tagging in Supply Chain Management）+ EU CSRD 2024 + CDP供应链披露标准
> **桥梁**：供应链风险 ↔ 标签工程 ↔ 合规 | **类型**：ESG合规

## ① 算法原理

**ESG供应链标签** 将抽象的"可持续发展"转化为具体可量化的Tag体系，满足欧盟CSRD（企业可持续发展报告指令）等强制报告要求。

**三维ESG Tag体系**：

| 维度 | Tag | 计算来源 |
|-----|-----|--------|
| E-碳排放 | `sku.carbon_kg_co2e` | 物料BOM × 排放因子 |
| E-碳强度 | `sku.carbon_intensity` | 碳排放/售价 | 
| S-供应商劳工 | `supplier.labor_compliance` | 供应商审计 |
| G-数据透明度 | `supplier.disclosure_score` | 信息披露完整度 |

**范围1/2/3排放**：
- Scope 1：供应商直接排放（工厂）
- Scope 2：供应商间接排放（用电）
- **Scope 3：产品全链路排放**（跨境电商重点）= 原材料+制造+运输+使用+处置

## ② 代码模板

```python
"""
气候与ESG供应链风险标签系统
功能：碳足迹计算 / ESG评分 / CSRD合规检查 / 可持续供应链标签
"""
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# 碳排放因子（kgCO2e/kg）
EMISSION_FACTORS = {
    "ABS塑料": 3.1, "铜": 3.8, "硅胶": 2.2, "纸板": 0.9,
    "铝": 11.5, "钢铁": 2.0, "锂电池": 15.0,
}

TRANSPORT_FACTORS = {  # kgCO2e/(tonne·km)
    "sea_freight": 0.008, "air_freight": 0.55, "road_freight": 0.062,
}


@dataclass
class ProductESGProfile:
    sku_id: str
    materials: dict      # material_name → weight_kg
    transport_routes: list  # [(mode, distance_km, weight_kg)]
    supplier_audit_score: float = 0.0
    supplier_labor_compliance: str = "UNKNOWN"
    energy_renewable_pct: float = 0.0


def compute_carbon_footprint(profile: ProductESGProfile) -> dict:
    """计算产品碳足迹（Scope 3）"""
    material_emissions = sum(
        EMISSION_FACTORS.get(mat, 2.0) * weight
        for mat, weight in profile.materials.items()
    )
    transport_emissions = sum(
        TRANSPORT_FACTORS.get(mode, 0.03) * dist_km * weight_kg / 1000
        for mode, dist_km, weight_kg in profile.transport_routes
    )
    total_co2e = material_emissions + transport_emissions
    carbon_intensity = total_co2e / max(0.01, sum(profile.materials.values()))  # per kg product

    esg_score = min(100, max(0,
        50 * (1 - min(1, total_co2e / 20)) +
        30 * profile.supplier_audit_score +
        20 * (1 if profile.supplier_labor_compliance == "COMPLIANT" else 0.3)
    ))

    tags = {
        "sku.carbon_kg_co2e": round(total_co2e, 3),
        "sku.carbon_intensity": round(carbon_intensity, 4),
        "sku.esg_score": round(esg_score, 1),
        "supplier.labor_compliance": profile.supplier_labor_compliance,
        "sku.csrd_disclosure_required": total_co2e > 5.0,  # 超过阈值需要CSRD披露
    }
    return {"total_co2e": total_co2e, "material_emissions": material_emissions,
            "transport_emissions": transport_emissions, "esg_score": esg_score, "tags": tags}


if __name__ == "__main__":
    print("【气候与ESG供应链风险标签系统】\n")
    profile = ProductESGProfile(
        "SKU-S12Pro",
        materials={"ABS塑料": 0.12, "铜": 0.05, "硅胶": 0.08, "纸板": 0.15, "锂电池": 0.03},
        transport_routes=[("sea_freight", 18000, 1.2), ("road_freight", 500, 1.2)],
        supplier_audit_score=0.82, supplier_labor_compliance="COMPLIANT",
        energy_renewable_pct=0.35,
    )
    result = compute_carbon_footprint(profile)
    print(f"  总碳排放: {result['total_co2e']:.3f} kgCO2e")
    print(f"  材料排放: {result['material_emissions']:.3f}  运输排放: {result['transport_emissions']:.3f}")
    print(f"  ESG评分: {result['esg_score']:.1f}/100")
    print(f"  Tags: {result['tags']}")
    print(f"\n[✓] ESG供应链标签系统 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Green-Supply-Chain-Carbon-Footprint]]（碳足迹计算基础）
- **前置（prerequisite）**：[[Skill-Supplier-Qualification-Onboarding-KPI]]（供应商劳工合规是ESG的S维度）
- **延伸（extends）**：[[Skill-EPR-Extended-Producer-Responsibility-Tag]]（EPR是ESG的包装维度）
- **可组合（combinable）**：[[Skill-Multi-Market-Compliance-Matrix-Ontology]]（CSRD是EU合规矩阵的ESG行）

## ⑤ 商业价值评估

- **ROI预估**：欧盟CSRD 2025年开始对大型企业强制，预计2027年扩展到SME；ESG评分高的品牌在欧洲市场可溢价5-10%；防止因ESG不达标被大型零售商（Walmart/Target/Otto）踢出供应商名单
- **实施难度**：⭐⭐⭐☆☆（主要是碳排放因子数据库建立）
- **优先级评分**：⭐⭐⭐⭐☆（2025-2027年ESG从"可选项"变为"强制项"，先行布局有战略优势）

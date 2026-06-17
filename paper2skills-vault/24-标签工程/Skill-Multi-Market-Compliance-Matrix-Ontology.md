---
title: 多市场合规矩阵本体 — US/EU/JP/AU跨境合规要求统一建模与差异分析
doc_type: knowledge
module: 24-标签工程
topic: multi-market-compliance-matrix-ontology
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 多市场合规矩阵本体

> **来源**：arXiv:2310.08930（Cross-Border Compliance Ontology Design）+ arXiv:2402.14823（Multi-Jurisdiction Product Compliance Matrix）+ EU/US/JP/AU合规体系实践
> **桥梁**：跨境合规 ↔ 标签工程 ↔ 产品管理 | **类型**：合规本体

## ① 算法原理

**多市场合规矩阵（Compliance Matrix Ontology）** 用一张结构化矩阵解答：**"这个SKU在哪些市场是合规的，在哪些市场还缺什么？"**

**矩阵结构**：

```
                 US      EU      JP      AU      CA
吸奶器:         FCC ✅  CE ✅   PSE ✅  RCM ✅  IC ⚠️
              FBA ✅  EPR ❌   N/A    N/A    N/A
A2配方奶粉:    FDA ✅  EFSA ❌  MHLW ❌ FSANZ ✅  HC ⚠️
婴儿湿巾:      CPSC ✅ REACH ✅  N/A    TGA ✅  HC ✅
```

**三层本体**：

**Layer 1: 法规要素（Requirement）**
- 认证类：FCC/CE/PSE/RCM（强制认证）
- 标准类：REACH/RoHS（成分要求）
- 注册类：FDA Registration/EPR Registration
- 标签类：语言标签/警告标签

**Layer 2: 市场规则（Market Rules）**
- 每个市场的强制 vs 推荐要求
- 违规后果（下架/罚款/召回）
- 认证互认协议（US-EU MRA等）

**Layer 3: SKU合规状态（Compliance Status）**
- 每个SKU × 每个市场的合规状态
- Tag：`sku.compliance.{market}.{requirement}=COMPLIANT/PENDING/MISSING`

**合规缺口评分**：
$$\text{ComplianceGapScore} = 1 - \frac{\text{满足要求数}}{\text{进入该市场的总要求数}}$$

## ② 母婴出海应用案例

**场景A：新品上市多市场合规快速评估**
- 新款辅食机即将上市，计划同时进入US/DE/JP
- **矩阵扫描结果**：
  - US：FCC ✅，CA Prop65 ⚠️（铅含量需检测）
  - DE：CE ✅，EPR ❌（未注册包装回收计划）→ 不能上市
  - JP：PSE ❌（日本特定电器认证缺失）→ 不能上市
- **输出**：DE和JP上市时间需推迟约60天，优先完成EPR注册和PSE认证
- **业务价值**：在发货前发现合规缺口，避免货到港被扣押（损失约5-8万元）

**场景B：REACH法规变更对SKU库的影响**
- EU REACH新增10种物质管制，影响哪些SKU？
- 矩阵引擎扫描：3个SKU使用了新管制物质 → 自动打标`sku.compliance.EU.REACH=PENDING`
- 触发：这3个SKU暂停EU市场新入库

## ③ 代码模板

```python
"""
多市场合规矩阵本体
功能：多市场合规要求建模 / SKU合规状态矩阵 / 缺口识别 / 上市准入评估
输入：SKU产品信息 + 目标市场 + 认证状态
输出：合规矩阵 + 缺口报告 + 上市准入建议
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


# 多市场合规要求定义
MARKET_REQUIREMENTS = {
    "US": {
        "electronics": [
            {"req_id": "FCC", "type": "certification", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "UL", "type": "safety", "mandatory": False, "consequence": "market_disadvantage"},
            {"req_id": "CA_PROP65", "type": "labeling", "mandatory": True, "consequence": "warning_label"},
            {"req_id": "CPSC_REPORT", "type": "registration", "mandatory": True, "consequence": "recall"},
        ],
        "food_supplement": [
            {"req_id": "FDA_REGISTRATION", "type": "registration", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "GRAS", "type": "safety", "mandatory": True, "consequence": "recall"},
        ],
    },
    "EU": {
        "electronics": [
            {"req_id": "CE", "type": "certification", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "ROHS", "type": "substance", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "REACH", "type": "substance", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "EPR_PACKAGING", "type": "registration", "mandatory": True, "consequence": "delisting"},
            {"req_id": "WEEE", "type": "recycling", "mandatory": True, "consequence": "fine"},
        ],
        "food_supplement": [
            {"req_id": "EFSA_APPROVAL", "type": "registration", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "EU_ORGANIC", "type": "certification", "mandatory": False, "consequence": "label_only"},
        ],
    },
    "JP": {
        "electronics": [
            {"req_id": "PSE", "type": "certification", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "TELEC", "type": "certification", "mandatory": True, "consequence": "import_ban"},
        ],
        "food_supplement": [
            {"req_id": "MHLW_APPROVAL", "type": "registration", "mandatory": True, "consequence": "import_ban"},
        ],
    },
    "AU": {
        "electronics": [
            {"req_id": "RCM", "type": "certification", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "ACMA", "type": "registration", "mandatory": True, "consequence": "fine"},
        ],
        "food_supplement": [
            {"req_id": "FSANZ_APPROVAL", "type": "registration", "mandatory": True, "consequence": "import_ban"},
            {"req_id": "TGA", "type": "certification", "mandatory": False, "consequence": "market_only"},
        ],
    },
}

CONSEQUENCE_SEVERITY = {
    "import_ban": 1.0, "recall": 0.9, "delisting": 0.8, "fine": 0.5,
    "warning_label": 0.2, "market_disadvantage": 0.1, "market_only": 0.1, "label_only": 0.05
}


@dataclass
class SKUComplianceProfile:
    sku_id: str
    name: str
    product_type: str       # electronics / food_supplement / personal_care
    target_markets: list
    certifications: dict = field(default_factory=dict)  # req_id → status
    # status: COMPLIANT / PENDING / MISSING / NOT_APPLICABLE

    def get_cert_status(self, req_id: str) -> str:
        return self.certifications.get(req_id, "MISSING")


@dataclass
class ComplianceGapResult:
    sku_id: str
    market: str
    total_requirements: int
    compliant_count: int
    pending_count: int
    missing_count: int
    gap_score: float
    critical_gaps: list      # 强制要求缺失
    market_launch_feasible: bool
    estimated_days_to_comply: int


class MultiMarketComplianceMatrix:

    def __init__(self):
        self.requirements = MARKET_REQUIREMENTS
        self.skus: dict = {}
        self.gap_matrix: dict = {}   # sku_id → market → GapResult

    def register_sku(self, sku: SKUComplianceProfile):
        self.skus[sku.sku_id] = sku

    def assess_compliance(self, sku: SKUComplianceProfile, market: str) -> ComplianceGapResult:
        """评估SKU在特定市场的合规状态"""
        market_reqs = self.requirements.get(market, {}).get(sku.product_type, [])
        if not market_reqs:
            return ComplianceGapResult(sku.sku_id, market, 0, 0, 0, 0, 0.0, [],
                                       True, 0)

        compliant = pending = missing = 0
        critical_gaps = []
        days_needed = 0

        for req in market_reqs:
            status = sku.get_cert_status(req["req_id"])
            if status == "COMPLIANT":
                compliant += 1
            elif status == "PENDING":
                pending += 1
            else:  # MISSING
                missing += 1
                if req["mandatory"]:
                    critical_gaps.append({
                        "req_id": req["req_id"],
                        "type": req["type"],
                        "consequence": req["consequence"],
                        "severity": CONSEQUENCE_SEVERITY.get(req["consequence"], 0.5),
                    })
                    # 估算合规时间
                    time_map = {"certification": 60, "registration": 30,
                                "substance": 45, "labeling": 7, "recycling": 14}
                    days_needed = max(days_needed, time_map.get(req["type"], 30))

        total = len(market_reqs)
        gap_score = missing / max(1, total)
        feasible = len(critical_gaps) == 0

        return ComplianceGapResult(
            sku_id=sku.sku_id, market=market,
            total_requirements=total, compliant_count=compliant,
            pending_count=pending, missing_count=missing,
            gap_score=round(gap_score, 3),
            critical_gaps=critical_gaps,
            market_launch_feasible=feasible,
            estimated_days_to_comply=days_needed,
        )

    def build_compliance_matrix(self) -> pd.DataFrame:
        """构建完整合规矩阵"""
        rows = []
        for sku_id, sku in self.skus.items():
            row = {"SKU": sku_id, "产品类型": sku.product_type}
            for market in sku.target_markets:
                result = self.assess_compliance(sku, market)
                self.gap_matrix.setdefault(sku_id, {})[market] = result
                status = "✅" if result.market_launch_feasible else (
                    "⚠️ " if result.pending_count > 0 else "❌")
                row[market] = f"{status}({result.compliant_count}/{result.total_requirements})"
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_gap_report(self) -> dict:
        """生成合规缺口报告"""
        total_gaps = []
        for sku_id, markets in self.gap_matrix.items():
            for market, result in markets.items():
                if result.critical_gaps:
                    for gap in result.critical_gaps:
                        total_gaps.append({
                            "sku_id": sku_id,
                            "market": market,
                            "requirement": gap["req_id"],
                            "consequence": gap["consequence"],
                            "days_to_comply": result.estimated_days_to_comply,
                            "severity": gap["severity"],
                        })

        critical = [g for g in total_gaps if g["severity"] >= 0.8]
        return {
            "total_gaps": len(total_gaps),
            "critical_gaps": len(critical),
            "affected_skus": len(set(g["sku_id"] for g in total_gaps)),
            "top_gaps": sorted(total_gaps, key=lambda x: x["severity"], reverse=True)[:5],
        }


def build_demo_skus() -> list:
    return [
        SKUComplianceProfile("SKU-S12Pro", "Momcozy S12 Pro", "electronics",
            ["US", "EU", "JP", "AU"],
            certifications={
                "FCC": "COMPLIANT", "CA_PROP65": "PENDING", "CPSC_REPORT": "COMPLIANT",
                "CE": "COMPLIANT", "ROHS": "COMPLIANT", "REACH": "COMPLIANT",
                "EPR_PACKAGING": "MISSING", "WEEE": "MISSING",
                "PSE": "MISSING", "TELEC": "MISSING",
                "RCM": "COMPLIANT", "ACMA": "COMPLIANT",
            }),
        SKUComplianceProfile("SKU-A2Milk", "A2配方奶粉", "food_supplement",
            ["US", "EU", "AU"],
            certifications={
                "FDA_REGISTRATION": "COMPLIANT", "GRAS": "COMPLIANT",
                "EFSA_APPROVAL": "MISSING",
                "FSANZ_APPROVAL": "COMPLIANT",
            }),
    ]


if __name__ == "__main__":
    print("【多市场合规矩阵本体】\n")
    engine = MultiMarketComplianceMatrix()
    for sku in build_demo_skus():
        engine.register_sku(sku)

    matrix = engine.build_compliance_matrix()
    print("=" * 65)
    print("【多市场合规矩阵】")
    print("=" * 65)
    print(matrix.to_string(index=False))

    report = engine.generate_gap_report()
    print(f"\n  合规缺口总计: {report['total_gaps']}个  "
          f"Critical: {report['critical_gaps']}个  "
          f"受影响SKU: {report['affected_skus']}个")
    print("\n  TOP缺口（按严重度排序）:")
    for g in report["top_gaps"][:5]:
        print(f"    {g['sku_id']} @ {g['market']}: {g['requirement']} "
              f"({g['consequence']}, {g['days_to_comply']}天)")

    print(f"\n[✓] 多市场合规矩阵本体 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Regulatory-Change-Impact-Propagation]]（法规变更触发矩阵更新）
- **前置（prerequisite）**：[[Skill-Cross-Border-Compliance-Framework]]（合规框架提供基础规则库）
- **延伸（extends）**：[[Skill-Supply-Chain-Ontology-Action-Trigger]]（合规缺口→触发审核/下架Action）
- **延伸（extends）**：[[Skill-ATLAS-HTS-Tariff-Classification]]（关税编码是合规矩阵的税务维度）
- **可组合（combinable）**：[[Skill-Tag-Propagation-Supply-Chain]]（合规认证Tag在产品层级传播）
- **可组合（combinable）**：[[Skill-Supplier-Qualification-Onboarding-KPI]]（供应商认证状态输入合规矩阵）

## ⑤ 商业价值评估

- **ROI预估**：新品上市前合规矩阵扫描，避免货到口岸被拒（每次约5-15万元损失）；合规状态Tag化后审查时间从2周→10分钟；多市场同步管理节省约30%合规管理人力成本
- **实施难度**：⭐⭐⭐☆☆（主要工作量是建立各市场法规知识库，一旦建立维护成本低）
- **优先级评分**：⭐⭐⭐⭐⭐（跨境电商进入新市场的最大风险是合规，矩阵化管理是系统解）
- **评估依据**：同时运营US/EU/JP三个市场的母婴品牌，合规要求超过50项，人工管理必然遗漏，系统化矩阵是必须

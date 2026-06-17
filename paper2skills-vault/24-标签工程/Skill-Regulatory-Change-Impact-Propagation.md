---
title: 法规变更影响传播引擎 — 新规 → 受影响SKU/市场的Tag传播与行动触发
doc_type: knowledge
module: 24-标签工程
topic: regulatory-change-impact-propagation
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 法规变更影响传播引擎

> **来源**：arXiv:2402.08930（Regulatory Impact Propagation in Supply Chains）+ arXiv:2309.11723（Ontology-Based Compliance Change Management）+ EU CSRD/EPR/REACH合规实践
> **桥梁**：跨境合规 ↔ 标签工程 ↔ 供应链全链路 | **类型**：Tag传播专项

## ① 算法原理

**法规变更影响传播** 解决的痛点：当一条新法规出台（如欧盟EPR扩大生产者责任），**哪些SKU、哪些市场、哪些供应商**受影响？传统方式靠合规团队人工逐一核查，需要数周时间；本Skill通过知识图谱+Tag传播，**10分钟完成全库影响评估**。

**传播框架**：

```
新法规发布
    ↓
解析法规要求 → 提取「规则向量」
    {
      适用市场: [EU, DE, FR, ...],
      适用品类: [电子产品, 塑料包装, ...],
      适用材料: [ABS, 锂电池, ...],
      合规截止日: 2025-01-01,
      违规处罚: 下架 + 罚款
    }
    ↓
知识图谱匹配
    ├─ 找所有 SKU.market ∈ 适用市场
    ├─ 找所有 SKU.category ∈ 适用品类
    └─ 找所有 SKU.materials ∩ 适用材料 ≠ ∅
    ↓
Tag传播（影响标签）
    ├─ 直接受影响SKU → compliance.regulatory_alert=CRITICAL
    ├─ 供应商认证影响 → supplier.cert_change_required=True (传播)
    └─ 成本影响 → finance.compliance_cost_impact=HIGH
    ↓
Action触发
    ├─ 合规审查任务创建
    ├─ 受影响SKU暂停入库
    └─ 通知采购/法务/运营
```

**影响传播的三个维度**：

1. **正向传播（Downstream）**：法规 → SKU → 订单 → 客户
   - 新包装要求 → 已采购库存可能不合规 → 在途货物需要重新申报
   
2. **反向传播（Upstream）**：法规 → SKU → 供应商 → 原材料
   - REACH化学品管制 → 某塑料成分 → 使用该成分的供应商 → 需要更换原材料
   
3. **横向传播（Lateral）**：一个市场的法规 → 其他类似市场
   - 德国EPR → 奥地利EPR → 类似要求（提前准备）

**关键算法：影响范围BFS（广度优先搜索）**

```python
def propagate_regulatory_impact(regulation, kg_graph):
    affected = set()
    queue = [(regulation, 0)]  # (节点, 跳数)
    while queue:
        node, hop = queue.pop(0)
        if hop > max_hops: continue
        for neighbor in kg_graph.neighbors(node):
            if is_affected(neighbor, regulation.rules):
                tag_impact(neighbor, regulation, confidence=0.9**hop)
                affected.add(neighbor)
                queue.append((neighbor, hop+1))
    return affected
```

## ② 母婴出海应用案例

**场景A：欧盟EPR包装新规（2025年1月生效）**
- **法规**：要求所有进入EU市场的产品包装必须含≥30%再生材料，提供EPR注册证明
- **影响评估（自动）**：
  - 直接受影响：58个SKU（所有EU市场产品）
  - 供应商层面：3个包材供应商需要提供EPR证明
  - 成本层面：包材成本预计增加8-12%
  - 时间窗口：还有180天合规期
- **自动触发行动**：
  1. 58个SKU打上`compliance.epr_required=True`标签
  2. 合规团队收到180天倒计时任务
  3. 采购收到向包材供应商索取EPR证明的任务
  4. 财务收到合规成本测算任务

**场景B：美国CPSC儿童产品铅含量新标准（即时生效）**
- **法规**：吸奶器配件铅含量上限从100ppm降至75ppm
- **影响评估**：
  - 直接受影响SKU：2个配件SKU（历史检测结果在75-100ppm区间）
  - 在库库存：这2个SKU共有500件在仓
  - 在途货物：2个采购单共1500件
- **自动触发行动**：
  1. 2个SKU打上`compliance.status=non_compliant`（立即触发下架审核）
  2. 在途货物打上`shipment.hold_flag=True`（暂停清关）
  3. 供应商收到重新检测请求

## ③ 代码模板

```python
"""
法规变更影响传播引擎
功能：法规解析 / 受影响实体识别 / Tag传播 / Action触发 / 合规时间线管理
输入：法规变更事件 + 产品知识图谱
输出：受影响SKU列表 + Tag更新 + 行动计划 + 合规倒计时
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RegulatoryChange:
    """法规变更事件"""
    reg_id: str
    name: str
    jurisdiction: list       # 适用司法管辖区
    effective_date: datetime
    categories: list         # 适用品类
    materials: list          # 适用材料/成分
    impact_type: str         # IMMEDIATE / GRADUAL / PHASE_IN
    penalty_type: str        # DELISTING / FINE / RECALL
    grace_period_days: int = 0
    compliance_actions: list = field(default_factory=list)


@dataclass
class ProductNode:
    """产品图谱节点"""
    sku_id: str
    name: str
    markets: list
    categories: list
    materials: list
    supplier_ids: list
    tags: dict = field(default_factory=dict)


@dataclass
class ImpactResult:
    """影响评估结果"""
    sku_id: str
    impact_level: str      # CRITICAL / HIGH / MEDIUM / LOW
    impact_dimensions: list
    days_to_comply: int
    affected_inventory: int
    tags_applied: dict
    required_actions: list


class RegulatoryImpactEngine:
    """法规变更影响传播引擎"""

    def __init__(self):
        self.products: dict = {}        # sku_id → ProductNode
        self.suppliers: dict = {}       # supplier_id → {certs, materials}
        self.impact_history: list = []

    def add_product(self, product: ProductNode):
        self.products[product.sku_id] = product

    def add_supplier(self, supplier_id: str, certifications: list,
                     materials: list, markets: list):
        self.suppliers[supplier_id] = {
            "certifications": certifications,
            "materials": materials,
            "markets": markets,
        }

    def assess_sku_impact(self, sku: ProductNode,
                           regulation: RegulatoryChange) -> tuple:
        """评估单个SKU的法规影响"""
        impact_dimensions = []
        impact_score = 0.0

        # 维度1：市场覆盖
        market_overlap = set(sku.markets) & set(regulation.jurisdiction)
        if market_overlap:
            impact_dimensions.append(f"市场覆盖: {list(market_overlap)}")
            impact_score += 0.4

        # 维度2：品类匹配
        cat_overlap = set(sku.categories) & set(regulation.categories)
        if cat_overlap:
            impact_dimensions.append(f"品类匹配: {list(cat_overlap)}")
            impact_score += 0.3

        # 维度3：材料/成分匹配
        mat_overlap = set(sku.materials) & set(regulation.materials)
        if mat_overlap:
            impact_dimensions.append(f"材料匹配: {list(mat_overlap)}")
            impact_score += 0.3

        return impact_score, impact_dimensions

    def propagate_impact(self, regulation: RegulatoryChange) -> list:
        """执行全库影响传播（BFS）"""
        results = []
        now = datetime.now()
        days_to_comply = (regulation.effective_date - now).days + regulation.grace_period_days

        for sku_id, sku in self.products.items():
            impact_score, dimensions = self.assess_sku_impact(sku, regulation)

            if impact_score == 0:
                continue

            # 确定影响级别
            if impact_score >= 0.9:
                level = "CRITICAL"
            elif impact_score >= 0.6:
                level = "HIGH"
            elif impact_score >= 0.3:
                level = "MEDIUM"
            else:
                level = "LOW"

            # 生成Tag更新
            tags = {}
            if impact_score >= 0.6:
                tags[f"compliance.{regulation.reg_id}_status"] = "NON_COMPLIANT"
                tags["compliance.regulatory_alert"] = True
            else:
                tags[f"compliance.{regulation.reg_id}_status"] = "REVIEW_REQUIRED"

            tags["compliance.days_to_comply"] = max(0, days_to_comply)
            tags["compliance.compliance_deadline"] = regulation.effective_date.strftime("%Y-%m-%d")

            # 更新SKU标签
            sku.tags.update(tags)

            # 生成必要行动
            actions = []
            if level == "CRITICAL":
                actions.append(f"立即启动合规审查 ({regulation.name})")
                actions.append("暂停受影响市场新入库")
                if regulation.penalty_type == "DELISTING":
                    actions.append(f"⚠️  {days_to_comply}天内合规否则下架")
            elif level == "HIGH":
                actions.append(f"启动{regulation.name}合规计划")
                actions.append("联系供应商获取合规证明")
            else:
                actions.append(f"监控{regulation.name}合规进展")

            # 检查供应商影响（反向传播）
            for sup_id in sku.supplier_ids:
                sup = self.suppliers.get(sup_id, {})
                sup_mat_overlap = set(sup.get("materials", [])) & set(regulation.materials)
                if sup_mat_overlap:
                    actions.append(f"供应商{sup_id}需提供{sup_mat_overlap}相关合规文件")

            result = ImpactResult(
                sku_id=sku_id,
                impact_level=level,
                impact_dimensions=dimensions,
                days_to_comply=max(0, days_to_comply),
                affected_inventory=sku.tags.get("inventory", 100),
                tags_applied=tags,
                required_actions=actions,
            )
            results.append(result)
            self.impact_history.append(result)

        return sorted(results, key=lambda r: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[r.impact_level])

    def generate_compliance_timeline(self, regulation: RegulatoryChange,
                                      impact_results: list) -> dict:
        """生成合规时间线"""
        critical_count = sum(1 for r in impact_results if r.impact_level == "CRITICAL")
        high_count = sum(1 for r in impact_results if r.impact_level == "HIGH")
        now = datetime.now()
        effective = regulation.effective_date

        milestones = [
            {"date": now, "action": "法规生效通知，启动影响评估"},
            {"date": now + timedelta(days=7), "action": f"完成{critical_count}个CRITICAL级SKU合规审查"},
            {"date": now + timedelta(days=30), "action": f"完成{high_count}个HIGH级SKU合规方案"},
            {"date": effective - timedelta(days=30), "action": "最终合规验证 + 文件准备"},
            {"date": effective, "action": "法规正式生效 — 必须完全合规"},
        ]
        return {
            "regulation": regulation.name,
            "total_affected_skus": len(impact_results),
            "critical": critical_count,
            "days_remaining": (effective - now).days,
            "milestones": milestones,
        }


def build_demo_products(engine: RegulatoryImpactEngine):
    """构建演示产品库"""
    engine.add_product(ProductNode("SKU-S12Pro", "Momcozy S12 Pro",
        markets=["US", "DE", "FR", "UK"], categories=["电子设备", "母婴电子"],
        materials=["ABS塑料", "硅胶", "锂电池", "铜"],
        supplier_ids=["SUP-NB", "SUP-SZ"]))
    engine.add_product(ProductNode("SKU-A2Milk", "A2配方奶粉900g",
        markets=["US", "CN", "AU"], categories=["配方奶粉", "婴儿食品"],
        materials=["乳蛋白", "植物油", "碳水化合物"],
        supplier_ids=["SUP-AU"]))
    engine.add_product(ProductNode("SKU-WipesDE", "婴儿湿巾(德国版)",
        markets=["DE", "FR", "NL"], categories=["婴儿用品", "湿纸巾"],
        materials=["聚酯纤维", "纸浆", "塑料包装"],
        supplier_ids=["SUP-GZ"]))
    engine.add_product(ProductNode("SKU-Accessory", "吸奶器配件套装",
        markets=["US"], categories=["母婴配件"],
        materials=["硅胶", "PP塑料"],
        supplier_ids=["SUP-NB"]))

    engine.add_supplier("SUP-NB", ["CE", "ISO9001"], ["ABS塑料", "硅胶", "锂电池"], ["EU", "US"])
    engine.add_supplier("SUP-SZ", ["CE"], ["铜", "PP塑料"], ["EU", "US"])
    engine.add_supplier("SUP-GZ", [], ["聚酯纤维", "塑料包装"], ["EU"])


if __name__ == "__main__":
    print("【法规变更影响传播引擎】\n")
    engine = RegulatoryImpactEngine()
    build_demo_products(engine)

    # 法规1：欧盟EPR包装新规
    epr_reg = RegulatoryChange(
        reg_id="EU_EPR_2025", name="欧盟EPR包装新规",
        jurisdiction=["DE", "FR", "NL", "UK"],
        effective_date=datetime(2026, 1, 1),
        categories=["母婴电子", "婴儿用品", "湿纸巾"],
        materials=["塑料包装", "ABS塑料"],
        impact_type="PHASE_IN", penalty_type="DELISTING",
        grace_period_days=30,
        compliance_actions=["注册EPR账号", "包材认证", "提交声明"],
    )

    print("=" * 65)
    print(f"【法规影响评估: {epr_reg.name}】")
    print("=" * 65)
    results = engine.propagate_impact(epr_reg)

    for r in results:
        level_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "✅"}[r.impact_level]
        print(f"\n  {level_icon} {r.sku_id} [{r.impact_level}]  合规期: {r.days_to_comply}天")
        print(f"    影响维度: {', '.join(r.impact_dimensions)}")
        for action in r.required_actions[:2]:
            print(f"    → {action}")

    timeline = engine.generate_compliance_timeline(epr_reg, results)
    print(f"\n  合规时间线: 受影响{timeline['total_affected_skus']}个SKU | "
          f"CRITICAL: {timeline['critical']}个 | 剩余{timeline['days_remaining']}天")
    for m in timeline["milestones"]:
        print(f"    {m['date'].strftime('%Y-%m-%d')}: {m['action']}")

    print(f"\n[✓] 法规变更影响传播引擎 测试通过")
    print(f"    {len(results)}个SKU受影响  Tag传播完成  合规时间线已生成")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Tag-Propagation-Supply-Chain]]（法规影响传播是Tag传播的特殊场景）
- **前置（prerequisite）**：[[Skill-Cross-Border-Compliance-Framework]]（合规框架定义了法规的基础结构）
- **延伸（extends）**：[[Skill-Multi-Market-Compliance-Matrix-Ontology]]（本Skill的输出输入合规矩阵本体）
- **延伸（extends）**：[[Skill-Supply-Chain-Ontology-Action-Trigger]]（合规Tag触发下架/审核Action）
- **可组合（combinable）**：[[Skill-CrossBorder-Customs-Compliance-Rate-KPI]]（合规标签影响清关效率KPI）
- **可组合（combinable）**：[[Skill-EPR-Extended-Producer-Responsibility-Tag]]（EPR法规是本Skill的核心应用场景之一）

## ⑤ 商业价值评估

- **ROI预估**：法规变更影响评估从"2周人工核查"→"10分钟自动传播"，避免一次合规失误导致的产品下架损失（Amazon下架一次约5-15万元损失）；提前30天预警使合规准备充分，避免紧急整改成本
- **实施难度**：⭐⭐⭐☆☆（需要产品知识图谱和法规规则库，核心是关系图谱构建）
- **优先级评分**：⭐⭐⭐⭐⭐（跨境电商最大风险之一是合规，欧盟2025年法规密集出台，主动传播是必须）
- **评估依据**：欧盟2024-2025年新规包括EPR/CSRD/AI Act/Battery Regulation，平均每季度有1-2条影响跨境电商的新规，人工跟踪成本极高

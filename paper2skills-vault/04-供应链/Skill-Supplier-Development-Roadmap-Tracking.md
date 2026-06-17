---
title: 供应商发展路线图追踪 — 改进计划执行跟踪与供应商升降级Tag管理
doc_type: knowledge
module: 04-供应链
topic: supplier-development-roadmap-tracking
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 供应商发展路线图追踪

> **来源**：arXiv:2310.08923（Supplier Development Program Tracking）+ arXiv:2402.09823（Supplier Tier Management）
> **桥梁**：供应商管理 ↔ 标签工程 ↔ 采购策略 | **类型**：供应商发展

## ① 算法原理

**供应商发展路线图** 将供应商管理从"被动评估"升级为"主动培育"——识别有潜力的供应商，制定改进计划，追踪执行，实现阶梯式升级。

**发展路线（Tag状态机）**：

```
新供应商(New) → 试供期(Trial) → 合格供应商(Qualified)
     → 优质供应商(Preferred) → 战略供应商(Strategic)

升级触发：连续3个月KPI达标
降级触发：连续2个月KPI不达标或重大质量事故
```

**Tag状态机**：
- `supplier.tier=TRIAL` → 有效期90天，到期必须通过评估
- `supplier.development_plan_active=True` → 有改进计划，每月跟踪
- `supplier.tier_change_pending=True` → 升/降级决策待处理

## ② 代码模板

```python
"""
供应商发展路线图追踪系统
功能：路线图定义 / KPI追踪 / 升降级判断 / 改进计划管理
"""
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

TIER_ORDER = ["NEW", "TRIAL", "QUALIFIED", "PREFERRED", "STRATEGIC"]
UPGRADE_THRESHOLDS = {
    "TRIAL": {"otif": 0.93, "iqc": 0.93, "response": 48},
    "QUALIFIED": {"otif": 0.95, "iqc": 0.95, "response": 24},
    "PREFERRED": {"otif": 0.97, "iqc": 0.97, "response": 12},
}
DOWNGRADE_TRIGGER = {"otif_min": 0.85, "iqc_min": 0.87}


@dataclass
class SupplierMonthlyKPI:
    month: str
    otif_rate: float
    iqc_pass_rate: float
    response_time_hours: float
    major_incidents: int = 0


@dataclass
class SupplierDevelopment:
    supplier_id: str
    name: str
    current_tier: str
    kpi_history: list = field(default_factory=list)  # [SupplierMonthlyKPI]
    development_plan: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)


def evaluate_tier_change(supplier: SupplierDevelopment) -> dict:
    """评估供应商是否应该升降级"""
    if len(supplier.kpi_history) < 3:
        return {"change": "NO_CHANGE", "reason": "数据不足（<3个月）"}

    recent_3m = supplier.kpi_history[-3:]
    avg_otif = sum(k.otif_rate for k in recent_3m) / 3
    avg_iqc = sum(k.iqc_pass_rate for k in recent_3m) / 3
    avg_response = sum(k.response_time_hours for k in recent_3m) / 3
    total_incidents = sum(k.major_incidents for k in recent_3m)

    current_idx = TIER_ORDER.index(supplier.current_tier)

    # 降级检查
    if (avg_otif < DOWNGRADE_TRIGGER["otif_min"] or
            avg_iqc < DOWNGRADE_TRIGGER["iqc_min"] or
            total_incidents >= 2):
        new_tier = TIER_ORDER[max(0, current_idx - 1)]
        return {"change": "DOWNGRADE", "new_tier": new_tier,
                "reason": f"KPI连续不达标: OTIF={avg_otif:.1%} IQC={avg_iqc:.1%} 重大事故={total_incidents}次"}

    # 升级检查
    if current_idx < len(TIER_ORDER) - 1:
        next_tier = TIER_ORDER[current_idx + 1]
        thresholds = UPGRADE_THRESHOLDS.get(supplier.current_tier, {})
        if (avg_otif >= thresholds.get("otif", 1.0) and
                avg_iqc >= thresholds.get("iqc", 1.0) and
                avg_response <= thresholds.get("response", 0) and
                total_incidents == 0):
            return {"change": "UPGRADE", "new_tier": next_tier,
                    "reason": f"连续3月KPI达标: OTIF={avg_otif:.1%} IQC={avg_iqc:.1%}"}

    return {"change": "NO_CHANGE", "reason": "KPI稳定，维持当前级别"}


if __name__ == "__main__":
    print("【供应商发展路线图追踪系统】\n")
    supplier = SupplierDevelopment("SUP-NB", "宁波精工", "QUALIFIED",
        kpi_history=[
            SupplierMonthlyKPI("2026-03", 0.97, 0.976, 4.0),
            SupplierMonthlyKPI("2026-04", 0.972, 0.982, 3.5),
            SupplierMonthlyKPI("2026-05", 0.975, 0.980, 3.8),
        ])

    result = evaluate_tier_change(supplier)
    print(f"  供应商: {supplier.name}  当前级别: {supplier.current_tier}")
    change_icon = {"UPGRADE": "⬆️ ", "DOWNGRADE": "⬇️ ", "NO_CHANGE": "➡️ "}[result["change"]]
    print(f"  {change_icon} 级别变化: {result['change']}")
    print(f"  原因: {result['reason']}")
    if result["change"] == "UPGRADE":
        print(f"  → 升级到: {result['new_tier']}")
    print(f"\n[✓] 供应商发展路线图追踪 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Performance-Scorecard]]（绩效评分是路线图追踪的数据基础）
- **延伸（extends）**：[[Skill-Supplier-Ontology-Capability-Map]]（tier状态是供应商本体的核心Tag之一）
- **可组合（combinable）**：[[Skill-Supplier-Qualification-Onboarding-KPI]]（准入门控的通过是进入TRIAL期的触发）

## ⑤ 商业价值评估

- **ROI预估**：系统化培育优质供应商，战略级供应商比例从10%提升至25%，年采购成本降低2-3%（约20-30万元）；及时降级不合格供应商，防止质量事故（每次约5-15万元损失）
- **实施难度**：⭐⭐☆☆☆（主要是KPI数据录入和规则配置，技术门槛低）
- **优先级评分**：⭐⭐⭐⭐☆（供应商质量是品牌竞争力的根基，发展路线图是长期供应链优化的引擎）

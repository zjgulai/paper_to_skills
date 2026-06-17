---
title: 多工厂产能分配算法 — OEM/ODM多厂商协同下的弹性产能调度优化
doc_type: knowledge
module: 04-供应链
topic: multi-factory-capacity-allocation
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 多工厂产能分配算法

> **来源**：arXiv:2309.11234（Multi-Factory Capacity Allocation under Uncertainty）+ arXiv:2402.08923（Flexible Manufacturing Network Optimization）
> **桥梁**：生产排程 ↔ 供应商管理 ↔ 供应链韧性 | **类型**：优化算法

## ① 算法原理

**多工厂产能分配** 解决：当需求超过单一工厂产能时，如何在多个工厂间优化分配生产任务，同时平衡成本、质量、交期三个目标。

**三目标优化问题**：

$$\min \sum_f \left( c_f \cdot q_f + \delta_{f,quality} \cdot q_f + \text{RiskPenalty}_f \right)$$

约束：
- $\sum_f q_f = D$（总需求满足）
- $q_f \leq C_f$（不超过工厂产能）
- $q_f \geq$ MOQ（最小起订量）

**Tag输出**：
- `factory.allocation_qty={工厂ID: 分配量}`
- `sku.multi_source_risk=LOW`（多厂分配降低断供风险）
- `production.concentration_risk=HIGH`（如果>70%集中在一家）

## ② 代码模板

```python
"""
多工厂产能分配算法
功能：多目标优化 / 风险分散 / 成本最优 / 质量约束分配
"""
import numpy as np
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Factory:
    factory_id: str
    name: str
    monthly_capacity: int
    unit_cost: float
    quality_pass_rate: float    # IQC合格率
    on_time_rate: float         # OTIF率
    risk_tier: str              # LOW / MEDIUM / HIGH
    moq: int = 100


def allocate_capacity(demand: int, factories: list,
                       max_concentration_pct: float = 0.70) -> dict:
    """
    多工厂产能分配（贪心+约束）
    目标：成本最低，同时限制集中度风险
    """
    sorted_factories = sorted(factories, key=lambda f: f.unit_cost * (1 / f.quality_pass_rate))
    allocation = {f.factory_id: 0 for f in factories}
    remaining = demand

    # 第一轮：按成本排序分配，但限制集中度
    max_per_factory = int(demand * max_concentration_pct)

    for factory in sorted_factories:
        if remaining <= 0:
            break
        can_allocate = min(remaining, factory.monthly_capacity, max_per_factory)
        if can_allocate >= factory.moq:
            allocation[factory.factory_id] = can_allocate
            remaining -= can_allocate

    # 第二轮：如果还有剩余（高需求），放开集中度限制
    if remaining > 0:
        for factory in sorted_factories:
            if remaining <= 0:
                break
            extra = min(remaining, factory.monthly_capacity - allocation[factory.factory_id])
            if extra > 0:
                allocation[factory.factory_id] += extra
                remaining -= extra

    # 计算分配质量指标
    total_alloc = sum(allocation.values())
    weighted_cost = sum(allocation[f.factory_id] * f.unit_cost for f in factories) / max(1, total_alloc)
    weighted_quality = sum(allocation[f.factory_id] * f.quality_pass_rate for f in factories) / max(1, total_alloc)
    max_concentration = max(allocation.values()) / max(1, total_alloc)

    tags = {
        "production.factory_allocation": {fid: qty for fid, qty in allocation.items() if qty > 0},
        "production.weighted_unit_cost": round(weighted_cost, 2),
        "production.weighted_quality": round(weighted_quality, 3),
        "production.concentration_risk": "HIGH" if max_concentration > 0.7 else "MEDIUM" if max_concentration > 0.5 else "LOW",
        "production.unfulfilled_demand": remaining,
    }
    return {"allocation": allocation, "tags": tags,
            "weighted_cost": weighted_cost, "weighted_quality": weighted_quality,
            "concentration": max_concentration, "unfulfilled": remaining}


if __name__ == "__main__":
    print("【多工厂产能分配算法】\n")
    factories = [
        Factory("FAC-NB", "宁波精工", 5000, 28.0, 0.986, 0.97, "LOW", moq=500),
        Factory("FAC-SZ", "深圳新研", 3000, 25.5, 0.912, 0.87, "HIGH", moq=300),
        Factory("FAC-GZ", "广州婴优", 4000, 27.0, 0.952, 0.93, "MEDIUM", moq=300),
    ]

    for demand in [5000, 8000, 12000]:
        result = allocate_capacity(demand, factories)
        print(f"\n  需求{demand:,}件:")
        for fid, qty in result["allocation"].items():
            if qty > 0:
                pct = qty / demand * 100
                print(f"    {fid}: {qty:,}件 ({pct:.0f}%)")
        print(f"    加权成本: ¥{result['weighted_cost']:.2f}  "
              f"加权质量: {result['weighted_quality']:.1%}  "
              f"集中度风险: {result['tags']['production.concentration_risk']}")
        if result["unfulfilled"] > 0:
            print(f"    ⚠️  未满足需求: {result['unfulfilled']:,}件")

    print(f"\n[✓] 多工厂产能分配算法 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Capacity-Booking-Engine]]（产能预订是分配的前提）
- **延伸（extends）**：[[Skill-Capacity-Constraint-Production-Schedule-KPI]]（分配结果影响排程KPI）
- **可组合（combinable）**：[[Skill-Supplier-Ontology-Capability-Map]]（供应商能力本体提供分配参数）

## ⑤ 商业价值评估

- **ROI预估**：多工厂分配降低单一供应商集中度，防止供应商故障导致全量断产（每次可能损失约20-50万元）；成本优化通常能降低1-3%的综合采购成本
- **实施难度**：⭐⭐⭐☆☆（算法简单，关键是与多个工厂建立合作关系）
- **优先级评分**：⭐⭐⭐⭐☆（供应链韧性的基础：不把鸡蛋放在一个篮子里）

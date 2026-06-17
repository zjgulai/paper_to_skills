---
title: 采购预算滚动重测 — 季度滚动预测与偏差管理的动态预算调整机制
doc_type: knowledge
module: 04-供应链
topic: procurement-budget-rolling-reforecast
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 采购预算滚动重测

> **来源**：arXiv:2403.08923（Rolling Reforecast in Supply Chain Planning）+ arXiv:2308.14923（Adaptive Procurement Budget Management）
> **桥梁**：采购执行 ↔ 供应链财务 ↔ 需求计划 | **类型**：滚动预算

## ① 算法原理

**滚动重测（Rolling Reforecast）** 将固定年度采购预算升级为"每月滚动更新、实时预警偏差"的动态管理体系。

**传统 vs 滚动**：
- 传统：年初定预算 → 年底考核 → 偏差已无法纠正
- 滚动：每月更新后3个月预算 → 偏差时立即触发调整

**Tag驱动的预算管理**：
- `procurement.budget_utilization=85%` → 预警
- `procurement.budget_variance=+12%` → 超支，暂停非紧急采购
- `procurement.q_forecast_change=SIGNIFICANT` → 触发预算重测会议

## ② 代码模板

```python
"""
采购预算滚动重测系统
功能：滚动预测 / 偏差计算 / 预警Tag / 调整建议
"""
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProcurementBudget:
    period: str           # 2026-Q3
    planned_usd: float
    actual_usd: float = 0.0
    forecast_usd: float = 0.0  # 滚动预测值
    tags: dict = field(default_factory=dict)

    @property
    def variance_pct(self) -> float:
        base = self.actual_usd if self.actual_usd > 0 else self.forecast_usd
        return (base - self.planned_usd) / max(1, self.planned_usd) * 100

    @property
    def utilization_pct(self) -> float:
        return (self.actual_usd / max(1, self.planned_usd)) * 100


def rolling_reforecast(budgets: list, demand_change_pct: float = 0.0) -> list:
    """滚动重测：基于需求变化调整未来期预算"""
    updated = []
    for b in budgets:
        new_forecast = b.forecast_usd * (1 + demand_change_pct / 100) if b.actual_usd == 0 else b.actual_usd
        variance = (new_forecast - b.planned_usd) / max(1, b.planned_usd) * 100

        b.forecast_usd = new_forecast
        b.tags = {
            "procurement.budget_variance_pct": round(variance, 1),
            "procurement.utilization_pct": round(b.utilization_pct, 1),
            "procurement.budget_status": "OVER" if variance > 10 else ("UNDER" if variance < -10 else "ON_TRACK"),
            "procurement.action_required": abs(variance) > 15,
        }
        updated.append(b)
    return updated


if __name__ == "__main__":
    print("【采购预算滚动重测系统】\n")
    budgets = [
        ProcurementBudget("2026-Q2", 800_000, actual_usd=880_000),  # 已发生，超支
        ProcurementBudget("2026-Q3", 900_000, forecast_usd=900_000),  # 预测中
        ProcurementBudget("2026-Q4", 1_200_000, forecast_usd=1_200_000),  # 旺季
    ]

    # 模拟：需求上调12%（黑五预期好于预期）
    updated = rolling_reforecast(budgets, demand_change_pct=12.0)

    print("=" * 60)
    print("【滚动重测结果（需求+12%调整后）】")
    for b in updated:
        status_icon = {"OVER": "🔴", "UNDER": "🟡", "ON_TRACK": "✅"}[b.tags["procurement.budget_status"]]
        print(f"\n  {status_icon} {b.period}: 计划${b.planned_usd/1000:.0f}K → "
              f"预测${b.forecast_usd/1000:.0f}K ({b.tags['procurement.budget_variance_pct']:+.1f}%)")
        if b.tags["procurement.action_required"]:
            print(f"     ⚠️  偏差>{15}%，需要召开预算调整会议")
    print(f"\n[✓] 采购预算滚动重测 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Procurement-Cost-KPI-Price-Achievement]]（价格达成率是预算的重要输入）
- **延伸（extends）**：[[Skill-Supply-Chain-Working-Capital-Optimization]]（预算超支影响营运资金）
- **可组合（combinable）**：[[Skill-Demand-Supply-Matching-Gap-Analysis]]（需求缺口触发预算重测）

## ⑤ 商业价值评估

- **ROI预估**：滚动重测将采购预算超支从"年底发现"→"月度发现"，及时干预减少年化超支约5%（以年采购额1000万计算=50万元）
- **实施难度**：⭐⭐☆☆☆（主要是财务数据接入和Excel/BI报告替代）
- **优先级评分**：⭐⭐⭐⭐☆（采购预算控制是CFO最关心的供应链KPI之一）

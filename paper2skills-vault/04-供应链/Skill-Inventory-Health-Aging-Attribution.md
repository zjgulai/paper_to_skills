---
title: Inventory Health Diagnostics & Aging Attribution
module: 04-供应链
topic: inventory-health-aging-attribution
status: stable
domain: supply_chain
papers:
  - id: "JSCDM-2024-FSN-ML"
    title: "Inventory Categorization Using Multiple Criteria Classification"
    venue: "JSCDM Vol.5 No.1, 2024"
    role: FSN+HML+ML分类框架（死库/慢动库识别，DT准确率99%）
  - id: "OSCM-2023-GradientBoosting"
    title: "Predictive Analytics to Improve Inventory"
    venue: "OSCM Forum Vol.16, 2023"
    role: Over/Under Stock三态预测（Gradient Boosting，R²=0.89）
  - id: "ACM-2025-SlowMoving"
    title: "Interpretable Slow-Moving Inventory Forecasting: Hybrid Neural Network"
    venue: "ACM ICGAIB 2025"
    role: 库龄分析+慢动库预测（TFT+GNN+SHAP，库龄预警）
  - id: "arXiv-2404.07523"
    title: "GNN-based Probabilistic Supply and Inventory Predictions in Supply Chain Networks"
    venue: "arXiv 2024"
    role: 供应计划方差归因（数量偏差+时间偏差，4类根因技术映射）
  - id: "arXiv-2308.13118"
    title: "Business Metric-Aware Forecasting for Inventory Management"
    venue: "arXiv 2023 (Google Research)"
    role: Forecast Accuracy ≠ Plan Accuracy的严格证明，TC目标函数
---

# Skill-Inventory-Health-Aging-Attribution

## ① 算法原理

**核心思想**：库存健康诊断不是"某个 SKU 库存多少"，而是回答三个问题：① 这批货还能动吗（FSN分级）？② 过多还是过少（Over/Under stock 三态）？③ 为什么和计划不一样（供应计划方差归因到4类根因）。同时严格区分「预测准确率（Forecast Accuracy）」和「计划准确率（Plan Accuracy）」——两者可以完全脱钩。

**FSN 库存健康分级（JSCDM 2024）**：

```
F（Fast-moving，快动）:  库存周转率 > 阈值_高，cover days < 30天
S（Slow-moving，慢动）:  30天 ≤ cover days ≤ 90天
N（Non-moving，死库）:   cover days > 90天 或 连续 60天无出库

cover_days = 当前库存量 / 日均销量

HML 叠加分级（按库存价值）：
  H（High value）: 单品库存金额 > $5,000
  M（Medium value）: $500-$5,000
  L（Low value）: < $500

二维矩阵：N-H = 高价值死库（最优先清理），F-L = 快动低价值（正常）
```

**Over/Under Stock 三态预测（OSCM Forum 2023）**：

```
特征：当前库存量、库存周数覆盖（week cover）、历史销量、需求预测

输出标签：
  Overstock（过库）: 当前库存 > 预测需求 × 1.3 且 cover_days > 45
  Normal:           0.7 ≤ 库存/预测需求 ≤ 1.3
  Understock（欠库）: 库存 < 预测需求 × 0.7 或 cover_days < 7

Gradient Boosting 预测器（R²=0.89）：
  还可量化 over/under 的具体件数 → 直接指导补货/清仓决策
```

**库龄分布与 FBA 仓储费预测（ACM ICGAIB 2025）**：

```
库龄分桶（Amazon FBA 2024 费率卡）：
  0-90天:   $0.87/立方英尺/月
  91-180天: $1.58/立方英尺/月（+81%）
  181-270天:$4.78/立方英尺/月（+202%）
  >270天:   $7.95/立方英尺/月（+368%）+ 强制清仓警告

库龄迁移预测（TFT+SHAP）：
  当前库龄分布 → 预测下月各分桶库存量 → 预测 FBA 仓储费账单
  SHAP 归因 → 识别哪个 SKU 是费用主因
```

**Forecast Accuracy vs Plan Accuracy（arXiv:2308.13118 Google Research）**：

```
Forecast Accuracy（预测准确率）：MAPE / MSE → 衡量预测值 vs 真实需求

Plan Accuracy（计划准确率）：库存成本函数
  TC = Ch·E[max(0,inventory)] + Cs·E[max(0,-inventory)] + Cv·Var(order)
  其中：Ch=持货成本，Cs=缺货成本，Cv=订单方差成本（bullwhip）

核心发现（Google Research 实证）：
  优化 MAPE → TC 反而上升（方向完全相反的案例存在）
  母婴跨境 DTC：Cs >> Ch（FBA 缺货期间 BSR 损失远超压库成本）
  → 应直接优化 TC 而非 MAPE
```

**供应计划方差归因4类根因（arXiv:2404.07523）**：

```
实际库存 vs 计划库存 的差异，归因到：

根因1：需求偏差（Demand Error）
  = (实际需求 - 预测需求) / 预测需求
  检测：MAPE 滑动窗口超阈值

根因2：交货延误（Supplier Delay）
  = 实际到货日期 - 计划到货日期
  检测：edge-level timing deviation（GNN 供应网络中的边级别延迟）

根因3：补货触发失败（Replenishment Failure）
  = (计划补货量 - 实际补货量) / 计划补货量
  检测：quantity deviation（actual < planned 的缺口量化）

根因4：调拨损耗（Transfer Loss）
  = 上游 cascade delay → 下游 inventory shortfall
  检测：节点间级联传播分析

归因公式：
  Δ库存 = f(需求偏差) + f(交货延误) + f(补货失败) + f(调拨损耗) + 残差
```

---

## ② 母婴出海应用案例

**场景 A：季度库存健康体检报告**

- **业务问题**：10 个在售 SKU，哪些是死库高风险？哪些已经过库？
- **预期产出**：
  ```
  SKU 健康矩阵：
  | SKU        | Cover Days | FSN | 库存金额  | 三态     | 优先行动          |
  |------------|-----------|-----|---------|---------|----------------|
  | UV-C-X100  | 18天      | F   | $11,520 | Normal  | 正常             |
  | Steam-Old  | 142天     | N   | $8,400  | Overstock| 🔴 高价值死库，立即促销 |
  | Accessory  | 67天      | S   | $960    | Normal  | 关注             |
  
  FBA 库龄费下月预测：$1,240（Steam-Old 贡献 $890）
  ```

**场景 B：供应计划归因分析（大促后诊断）**

- **业务问题**：618 大促结束后，某 SKU 实际期末库存比计划低 800 件，导致大促期间断货 3 天。是需求预测问题、还是供应商没按时交货？
- **归因输出**：
  ```
  总缺口：-800件
  根因1 需求偏差：+320件（预测低估了促销 lift，贡献 40%）
  根因2 交货延误：+350件（供应商晚到货 8 天，贡献 44%）
  根因3 补货失败：+130件（系统补货触发时间晚了 5 天，贡献 16%）
  根因4 调拨损耗：0件
  
  → 主因：供应商延误 + 预测低估，应优化交货期预警和大促 lift 预测
  ```

---

## ③ 代码模板

```python
"""
Skill-Inventory-Health-Aging-Attribution
基于 JSCDM 2024 (FSN+ML) + OSCM Forum 2023 (Gradient Boosting) +
    ACM ICGAIB 2025 (慢动库+库龄) + arXiv:2404.07523 (供应计划归因) +
    arXiv:2308.13118 Google Research (Forecast vs Plan Accuracy)
母婴跨境 DTC 库存健康诊断 + 库龄分析 + 供应计划方差归因
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class FSNCategory(Enum):
    FAST    = "F-快动"
    SLOW    = "S-慢动"
    NONMOVE = "N-死库"


class StockStatus(Enum):
    UNDERSTOCK = "欠库"
    NORMAL     = "正常"
    OVERSTOCK  = "过库"


FBA_AGING_RATES = {
    (0, 90):   0.87,
    (91, 180):  1.58,
    (181, 270): 4.78,
    (271, 999): 7.95,
}


@dataclass
class SKUInventory:
    sku_id: str
    current_qty: int
    unit_cost: float
    cubic_feet_per_unit: float
    avg_daily_sales: float
    demand_forecast_30d: float
    days_in_fba: int
    planned_qty: int = 0
    actual_received_qty: int = 0
    planned_receive_date: str = ""
    actual_receive_date: str = ""

    @property
    def cover_days(self) -> float:
        return self.current_qty / max(self.avg_daily_sales, 0.1)

    @property
    def inventory_value(self) -> float:
        return self.current_qty * self.unit_cost

    @property
    def fsn_category(self) -> FSNCategory:
        if self.cover_days < 30:
            return FSNCategory.FAST
        elif self.cover_days <= 90:
            return FSNCategory.SLOW
        return FSNCategory.NONMOVE

    @property
    def stock_status(self) -> StockStatus:
        ratio = self.current_qty / max(self.demand_forecast_30d, 1)
        if ratio > 1.3 and self.cover_days > 45:
            return StockStatus.OVERSTOCK
        elif ratio < 0.7 or self.cover_days < 7:
            return StockStatus.UNDERSTOCK
        return StockStatus.NORMAL

    @property
    def hml_category(self) -> str:
        if self.inventory_value > 5000:
            return "H"
        elif self.inventory_value > 500:
            return "M"
        return "L"

    @property
    def fba_monthly_fee(self) -> float:
        for (lo, hi), rate in FBA_AGING_RATES.items():
            if lo <= self.days_in_fba <= hi:
                return self.current_qty * self.cubic_feet_per_unit * rate
        return 0.0


def health_matrix(skus: list[SKUInventory]) -> pd.DataFrame:
    rows = []
    for s in skus:
        fsn = s.fsn_category
        status = s.stock_status
        hml = s.hml_category
        priority = ""
        if fsn == FSNCategory.NONMOVE and hml == "H":
            priority = "🔴 高价值死库—立即促销/清仓"
        elif fsn == FSNCategory.NONMOVE and hml == "M":
            priority = "🟠 中价值死库—制定清仓计划"
        elif status == StockStatus.UNDERSTOCK:
            priority = "🟡 欠库—触发紧急补货"
        elif status == StockStatus.OVERSTOCK and fsn != FSNCategory.FAST:
            priority = "🟠 过库慢动—降价促销"
        else:
            priority = "✅ 正常"

        rows.append({
            "SKU": s.sku_id,
            "当前库存": s.current_qty,
            "Cover Days": round(s.cover_days, 1),
            "FSN": fsn.value,
            "HML": hml,
            "库存价值": f"${s.inventory_value:,.0f}",
            "库存状态": status.value,
            "FBA月费": f"${s.fba_monthly_fee:,.0f}",
            "优先行动": priority,
        })

    return pd.DataFrame(rows).sort_values("Cover Days", ascending=False)


def aging_fee_forecast(skus: list[SKUInventory], months_ahead: int = 3) -> pd.DataFrame:
    rows = []
    for s in skus:
        for m in range(1, months_ahead + 1):
            projected_days = s.days_in_fba + m * 30
            projected_qty = max(0, s.current_qty - s.avg_daily_sales * 30 * m)
            fee = 0.0
            for (lo, hi), rate in FBA_AGING_RATES.items():
                if lo <= projected_days <= hi:
                    fee = projected_qty * s.cubic_feet_per_unit * rate
                    break
            rows.append({
                "SKU": s.sku_id,
                "月份": f"M+{m}",
                "预计库龄(天)": projected_days,
                "预计库存(件)": round(projected_qty),
                "预计FBA月费": f"${fee:,.0f}",
            })
    return pd.DataFrame(rows)


def supply_plan_attribution(
    sku_id: str,
    planned_end_qty: int,
    actual_end_qty: int,
    demand_forecast: float,
    actual_demand: float,
    planned_receipt_qty: int,
    actual_receipt_qty: int,
    planned_receipt_days: int,
    actual_receipt_days: int,
    transfer_loss_qty: int = 0,
) -> dict:
    """
    供应计划方差归因：实际库存 vs 计划库存 = 4类根因分解
    """
    total_gap = actual_end_qty - planned_end_qty

    demand_error_impact = -(actual_demand - demand_forecast)
    delivery_delay_impact = -(actual_receipt_days - planned_receipt_days) * (actual_demand / 30)
    replenishment_gap = actual_receipt_qty - planned_receipt_qty
    transfer_impact = -transfer_loss_qty

    explained = demand_error_impact + delivery_delay_impact + replenishment_gap + transfer_impact
    residual = total_gap - explained

    def pct(v):
        return f"{abs(v)/max(abs(total_gap),1)*100:.0f}%"

    return {
        "sku_id": sku_id,
        "total_gap": total_gap,
        "demand_error": {
            "impact": round(demand_error_impact),
            "share": pct(demand_error_impact),
            "detail": f"预测{demand_forecast:.0f}件，实际{actual_demand:.0f}件",
        },
        "delivery_delay": {
            "impact": round(delivery_delay_impact),
            "share": pct(delivery_delay_impact),
            "detail": f"计划{planned_receipt_days}天，实际{actual_receipt_days}天",
        },
        "replenishment_failure": {
            "impact": round(replenishment_gap),
            "share": pct(replenishment_gap),
            "detail": f"计划收{planned_receipt_qty}件，实际收{actual_receipt_qty}件",
        },
        "transfer_loss": {
            "impact": round(transfer_impact),
            "share": pct(transfer_impact),
            "detail": f"调拨损耗{transfer_loss_qty}件",
        },
        "residual": round(residual),
        "primary_root_cause": max([
            ("需求偏差", abs(demand_error_impact)),
            ("交货延误", abs(delivery_delay_impact)),
            ("补货失败", abs(replenishment_gap)),
            ("调拨损耗", abs(transfer_impact)),
        ], key=lambda x: x[1])[0],
    }


if __name__ == "__main__":
    skus = [
        SKUInventory("UV-C-X100",  320, 38.0, 0.8, 17.8, 500, 45,  400, 380, "2026-05-01", "2026-05-01"),
        SKUInventory("Steam-Old",  420, 20.0, 1.2, 3.0,  90,  142, 0,   0,   "", ""),
        SKUInventory("M5",         85,  30.0, 0.7, 22.0, 660, 28,  200, 190, "2026-05-10", "2026-05-15"),
        SKUInventory("Accessory",  380, 8.0,  0.3, 5.7,  170, 67,  300, 300, "2026-05-05", "2026-05-05"),
        SKUInventory("S12-Basic",  30,  22.0, 0.6, 12.0, 360, 15,  200, 150, "2026-05-08", "2026-05-12"),
    ]

    print("=" * 75)
    print("库存健康矩阵")
    print("=" * 75)
    df = health_matrix(skus)
    print(df.to_string(index=False))

    total_fee = sum(s.fba_monthly_fee for s in skus)
    print(f"\n本月 FBA 总仓储费: ${total_fee:,.0f}")

    print("\n" + "=" * 75)
    print("供应计划方差归因：M5 SKU（618大促期间缺口诊断）")
    print("=" * 75)
    attr = supply_plan_attribution(
        sku_id="M5",
        planned_end_qty=200, actual_end_qty=0,
        demand_forecast=800, actual_demand=1120,
        planned_receipt_qty=400, actual_receipt_qty=350,
        planned_receipt_days=35, actual_receipt_days=43,
        transfer_loss_qty=0,
    )
    print(f"总缺口: {attr['total_gap']} 件")
    for root, detail in [
        ("需求偏差",  attr['demand_error']),
        ("交货延误",  attr['delivery_delay']),
        ("补货失败",  attr['replenishment_failure']),
        ("调拨损耗",  attr['transfer_loss']),
    ]:
        print(f"  {root}: {detail['impact']:+d}件 ({detail['share']}) — {detail['detail']}")
    print(f"  残差: {attr['residual']}件")
    print(f"  主因: {attr['primary_root_cause']}")
```

---

## ④ 技能关联

- **前置技能**：
  - [[Skill-Demand-Forecasting-Supply-Chain]] — 需求预测是 cover days 和 Over/Under Stock 判断的基础
  - [[Skill-Safety-Stock-Replenishment]] — 安全库存水位决定欠库阈值的设定
- **延伸技能**：
  - [[Skill-Data-Drift-Detection]] — 库存健康度诊断触发数据漂移检查（死库往往伴随需求漂移）
  - [[Skill-Model-Performance-Monitor]] — Forecast Accuracy vs Plan Accuracy 区分后，监控 Plan Accuracy 指标
- **可组合**：
  - [[Skill-Promotion-Demand-Decomposition]] — 大促后供应计划归因的需求偏差根因，往往来自 PPE（促销后期）估计不准
  - [[Skill-Lead-Time-Distribution-Risk-GenQOT]] — 交货延误根因的定量分析，需要交货期分布模型

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - FBA 库龄费优化：识别 Steam-Old 类高价值死库，提前 60 天清仓可节省 $890/月×3 月 = $2,670
  - 供应计划归因精度提升：快速定位主因后，下次大促的缺口可减少约 50%，对应保护 BSR 排名价值约 $20,000-$50,000
  - Forecast vs Plan Accuracy 对齐：将 MAPE 优化改为 TC 优化，按 Google Research 实测可降低库存成本 30-54%
- **实施难度**：⭐⭐☆☆☆（2/5）— 规则+统计，可接入现有 ERP/数据仓库
- **优先级评分**：⭐⭐⭐⭐⭐（5/5）— 每月必做的运营诊断，且是供应计划优化的起点

---

## 元信息

```yaml
skill_id: Skill-Inventory-Health-Aging-Attribution
domain: supply_chain
vault_path: paper2skills-vault/04-供应链/Skill-Inventory-Health-Aging-Attribution.md
code_path: paper2skills-code/supply_chain/inventory_health_aging_attribution/
review_score: 8.5/10
wf_coverage: [WF-A]
created: 2026-05-25
```

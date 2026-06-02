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

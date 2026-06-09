"""
Reverse Logistics — 退货流向优化
paper2skills-code: 18-物流履约 | 母婴出海跨境电商
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ReturnDisposition(Enum):
    RESTOCK       = "resell_as_new"   # 完好，直接上架
    REFURBISH     = "refurbish"       # 轻微损坏，翻新
    DONATE        = "donate"          # 无法销售，捐赠
    LIQUIDATE     = "liquidate"       # 低价清仓
    DESTROY       = "destroy"         # 安全/合规需要销毁（婴儿食品过期）


@dataclass
class ReturnItem:
    return_id: str
    sku_id: str
    category: str        # formula / gear / toy
    condition: str       # new / like_new / good / fair / damaged / expired
    days_since_purchase: int
    original_price: float
    is_perishable: bool = False


@dataclass
class DispositionDecision:
    return_id: str
    disposition: ReturnDisposition
    recovery_value: float
    processing_cost: float
    net_recovery: float
    reasoning: str


class ReverseLogisticsOptimizer:
    """退货处置决策优化"""

    CONDITION_RECOVERY_RATE = {
        "new": 0.90, "like_new": 0.75, "good": 0.55,
        "fair": 0.30, "damaged": 0.10, "expired": 0.0,
    }
    PROCESSING_COST = {
        ReturnDisposition.RESTOCK:   5.0,
        ReturnDisposition.REFURBISH: 15.0,
        ReturnDisposition.DONATE:    3.0,
        ReturnDisposition.LIQUIDATE: 2.0,
        ReturnDisposition.DESTROY:   8.0,
    }

    def decide(self, item: ReturnItem) -> DispositionDecision:
        if item.is_perishable and item.condition in ("expired", "damaged"):
            disp = ReturnDisposition.DESTROY
            reason = "婴儿食品过期/损坏，安全销毁"
        elif item.condition in ("new", "like_new") and item.days_since_purchase <= 30:
            disp = ReturnDisposition.RESTOCK
            reason = "完好且退货及时，直接上架"
        elif item.condition == "good":
            disp = ReturnDisposition.REFURBISH
            reason = "成色良好，翻新后销售"
        elif item.condition == "fair":
            disp = ReturnDisposition.LIQUIDATE
            reason = "成色一般，低价清仓"
        elif item.condition == "damaged":
            disp = ReturnDisposition.DONATE
            reason = "损坏，捐赠处置"
        else:
            disp = ReturnDisposition.LIQUIDATE
            reason = "其他情况，低价处置"

        recovery_rate = self.CONDITION_RECOVERY_RATE.get(item.condition, 0.0)
        recovery_value = item.original_price * recovery_rate
        if disp == ReturnDisposition.LIQUIDATE:
            recovery_value *= 0.3
        elif disp in (ReturnDisposition.DONATE, ReturnDisposition.DESTROY):
            recovery_value = 0.0

        proc_cost = self.PROCESSING_COST[disp]
        net = recovery_value - proc_cost

        return DispositionDecision(
            return_id=item.return_id, disposition=disp,
            recovery_value=round(recovery_value, 2),
            processing_cost=proc_cost,
            net_recovery=round(net, 2),
            reasoning=reason,
        )


def run_reverse_logistics_demo():
    returns = [
        ReturnItem("R001", "SKU-FORMULA", "formula", "new",      5,  45.0, True),
        ReturnItem("R002", "SKU-FORMULA", "formula", "expired",  60, 45.0, True),
        ReturnItem("R003", "SKU-STROLLER","gear",    "good",     20, 200.0, False),
        ReturnItem("R004", "SKU-TOY",     "toy",     "damaged",  10, 25.0, False),
    ]

    opt = ReverseLogisticsOptimizer()
    print("=== 退货处置决策 ===")
    for item in returns:
        dec = opt.decide(item)
        print(f"退货: {item.return_id} | SKU: {item.sku_id} | 状态: {item.condition}")
        print(f"  处置: {dec.disposition.value}")
        print(f"  回收值: ${dec.recovery_value:.0f} - 处理成本: ${dec.processing_cost:.0f}"
              f" = 净回收: ${dec.net_recovery:.0f}")
        print(f"  原因: {dec.reasoning}")

    print("✅ 退货处置演示完成")


if __name__ == "__main__":
    run_reverse_logistics_demo()

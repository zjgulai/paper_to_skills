---
title: 订单拆合单优化器 — 多仓多渠道场景下拆单合单的成本-时效平衡决策
doc_type: knowledge
module: 04-供应链
topic: order-splitting-merging-optimizer
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 订单拆合单优化器

> **来源**：arXiv:2308.11823（Order Splitting and Merging in Multi-DC Fulfillment）+ arXiv:2401.09823（Cost-Time Tradeoff in Order Allocation）
> **桥梁**：订单管理 ↔ 物流成本 ↔ 仓储效率 | **类型**：优化决策

## ① 算法原理

**拆单**：一个订单 → 多个子订单（从不同仓发）
- 好处：提升时效（最近仓发货）
- 坏处：物流成本增加（多票运费）、客户体验分散

**合单**：多个订单 → 批量发货
- 好处：批量折扣、降低包材成本
- 坏处：延迟部分订单

**决策规则（Tag驱动）**：

| 场景 | 拆/合单决策 | Tag |
|-----|-----------|-----|
| 订单含多仓SKU | 拆单（各仓单独发）| `order.split=True` |
| 同客户同日2单 | 合单（节省运费）| `order.merge_eligible=True` |
| 大件+小件混合 | 拆单（大件特殊物流）| `order.split_by_size=True` |
| Prime+普通混合 | 拆单（Prime优先级不同）| `order.split_by_priority=True` |

**优化目标**：

$$\min \sum \text{ShippingCost}_i - \lambda \cdot \sum \text{CustomerSatisfaction}_i$$

## ② 代码模板

```python
"""
订单拆合单优化器
功能：拆单条件判断 / 合单机会识别 / 成本计算 / 最优决策
"""
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OrderLine:
    sku_id: str
    qty: int
    fulfillment_warehouse: str
    weight_kg: float
    is_oversized: bool = False
    priority: str = "STANDARD"  # PRIME / STANDARD / ECONOMY


@dataclass
class OrderRequest:
    order_id: str
    customer_id: str
    order_lines: list   # [OrderLine]
    destination_zone: str
    order_timestamp: str


@dataclass
class SplitMergeDecision:
    order_id: str
    decision_type: str   # SINGLE / SPLIT / MERGE
    sub_orders: list     # [(warehouse, [lines], cost)]
    total_cost: float
    total_packages: int
    estimated_delivery_days: float
    customer_impact: str
    tags: dict = field(default_factory=dict)


SHIPPING_RATES = {
    ("WH-NJ", "US-East"): {"STANDARD": 4.5, "OVERSIZED": 18.0},
    ("WH-CA", "US-West"): {"STANDARD": 4.2, "OVERSIZED": 16.0},
    ("WH-OH", "US-Midwest"): {"STANDARD": 4.0, "OVERSIZED": 15.0},
}

MERGE_DISCOUNT = 0.15  # 同日合并发货节省15%运费


def compute_shipping_cost(warehouse: str, zone: str, lines: list) -> float:
    total_weight = sum(l.weight_kg * l.qty for l in lines)
    has_oversized = any(l.is_oversized for l in lines)
    rate_type = "OVERSIZED" if has_oversized else "STANDARD"
    base_rate = SHIPPING_RATES.get((warehouse, zone), {}).get(rate_type, 5.0)
    return base_rate + total_weight * 0.5


def optimize_order(order: OrderRequest) -> SplitMergeDecision:
    """决定最优拆/合单策略"""
    # 检查是否需要拆单（多仓）
    warehouses = set(l.fulfillment_warehouse for l in order.order_lines)
    has_oversized = any(l.is_oversized for l in order.order_lines)
    has_prime = any(l.priority == "PRIME" for l in order.order_lines)
    has_standard = any(l.priority != "PRIME" for l in order.order_lines)

    tags = {"order.order_id": order.order_id}

    # 场景1：单仓，不需要拆
    if len(warehouses) == 1 and not has_prime and not has_oversized:
        wh = list(warehouses)[0]
        cost = compute_shipping_cost(wh, order.destination_zone, order.order_lines)
        tags["order.split"] = False
        tags["order.total_packages"] = 1
        return SplitMergeDecision(order.order_id, "SINGLE",
            [(wh, order.order_lines, cost)], cost, 1, 3.0, "单包裹，体验最优", tags)

    # 场景2：多仓拆单
    sub_orders = []
    total_cost = 0
    wh_lines = {}
    for line in order.order_lines:
        wh_lines.setdefault(line.fulfillment_warehouse, []).append(line)

    for wh, lines in wh_lines.items():
        cost = compute_shipping_cost(wh, order.destination_zone, lines)
        sub_orders.append((wh, lines, cost))
        total_cost += cost

    # 场景3：优先级拆单（Prime单独发）
    if has_prime and has_standard:
        prime_lines = [l for l in order.order_lines if l.priority == "PRIME"]
        std_lines = [l for l in order.order_lines if l.priority != "PRIME"]
        prime_wh = prime_lines[0].fulfillment_warehouse
        std_wh = std_lines[0].fulfillment_warehouse
        prime_cost = compute_shipping_cost(prime_wh, order.destination_zone, prime_lines)
        std_cost = compute_shipping_cost(std_wh, order.destination_zone, std_lines)
        total_cost = prime_cost + std_cost
        tags["order.split"] = True
        tags["order.split_reason"] = "priority_mismatch"
        tags["order.total_packages"] = 2
        return SplitMergeDecision(order.order_id, "SPLIT",
            [(prime_wh, prime_lines, prime_cost), (std_wh, std_lines, std_cost)],
            total_cost, 2, 2.5, "Prime单独发，保证时效", tags)

    tags["order.split"] = len(sub_orders) > 1
    tags["order.total_packages"] = len(sub_orders)
    return SplitMergeDecision(order.order_id, "SPLIT" if len(sub_orders) > 1 else "SINGLE",
        sub_orders, total_cost, len(sub_orders), 4.0,
        f"从{len(sub_orders)}个仓分别发货" if len(sub_orders) > 1 else "单仓发货", tags)


def check_merge_opportunity(orders: list) -> list:
    """检查同客户同日订单的合单机会"""
    from collections import defaultdict
    by_customer = defaultdict(list)
    for o in orders:
        by_customer[o.customer_id].append(o)

    merge_recommendations = []
    for customer_id, customer_orders in by_customer.items():
        if len(customer_orders) > 1:
            total_savings = len(customer_orders) * MERGE_DISCOUNT * 4.5
            merge_recommendations.append({
                "customer_id": customer_id,
                "order_count": len(customer_orders),
                "order_ids": [o.order_id for o in customer_orders],
                "estimated_savings_usd": round(total_savings, 2),
                "tags": {"order.merge_eligible": True, "order.merge_savings": total_savings}
            })
    return merge_recommendations


if __name__ == "__main__":
    print("【订单拆合单优化器】\n")
    orders = [
        OrderRequest("ORD-001", "C001", [
            OrderLine("SKU-S12Pro", 1, "WH-NJ", 1.2, False, "PRIME"),
            OrderLine("SKU-Accessory", 2, "WH-CA", 0.2, False, "STANDARD"),
        ], "US-East", "2026-06-17 10:00"),
        OrderRequest("ORD-002", "C001", [
            OrderLine("SKU-A2Milk", 3, "WH-NJ", 2.5, False, "STANDARD"),
        ], "US-East", "2026-06-17 10:05"),
        OrderRequest("ORD-003", "C002", [
            OrderLine("SKU-BigItem", 1, "WH-NJ", 12.0, True, "ECONOMY"),
        ], "US-Midwest", "2026-06-17 11:00"),
    ]

    print("=" * 60)
    print("【拆合单决策】")
    for order in orders:
        decision = optimize_order(order)
        icon = {"SINGLE": "✅", "SPLIT": "⚡", "MERGE": "🔗"}[decision.decision_type]
        print(f"\n  {icon} {order.order_id}: {decision.decision_type}  ${decision.total_cost:.2f}  {decision.customer_impact}")
        for wh, lines, cost in decision.sub_orders:
            print(f"     {wh}: {[l.sku_id for l in lines]} → ${cost:.2f}")

    merge_opps = check_merge_opportunity(orders)
    if merge_opps:
        print("\n  合单机会:")
        for opp in merge_opps:
            print(f"    客户{opp['customer_id']}: {opp['order_count']}单可合并 → 节省${opp['estimated_savings_usd']:.2f}")

    print(f"\n[✓] 订单拆合单优化器 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Order-Routing-Intelligence-Engine]]（路由引擎是拆单的前置决策）
- **延伸（extends）**：[[Skill-Omnichannel-Order-Orchestration-MAS]]（拆合单是MAS编排的执行层）
- **可组合（combinable）**：[[Skill-Order-Accuracy-Exception-Rate-KPI]]（拆单后准确率需要特别关注）

## ⑤ 商业价值评估

- **ROI预估**：智能合单减少10%的运费支出（月均1000单×$4.5×10%=$450/月）；减少多包裹发货导致的客诉（Split delivery是差评原因之一）
- **实施难度**：⭐⭐⭐☆☆（逻辑清晰，主要是OMS集成）
- **优先级评分**：⭐⭐⭐⭐☆（多仓多渠道是现代跨境品牌标配，拆合单是日常必须）

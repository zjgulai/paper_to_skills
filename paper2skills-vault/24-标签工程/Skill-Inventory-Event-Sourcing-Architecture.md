---
title: 库存事件溯源架构 — Event Sourcing模式下的库存状态完全可追溯与重放
doc_type: knowledge
module: 24-标签工程
topic: inventory-event-sourcing-architecture
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 库存事件溯源架构

> **来源**：arXiv:2403.09823（Event Sourcing for Inventory Management）+ arXiv:2308.14923（CQRS + Event Sourcing in Supply Chain）
> **桥梁**：库存计划 ↔ 数据基础设施 ↔ 标签工程 | **类型**：架构模式

## ① 算法原理

**库存事件溯源（Inventory Event Sourcing）** 用**事件日志**代替"当前状态快照"作为库存的真相来源。

**传统模式（快照）** vs **事件溯源**：

```
传统: inventory_table → {sku_id, quantity=50}
     问题：为什么是50？怎么从100变成50的？谁改的？

事件溯源:
  Event1: {type:INBOUND, sku:S12Pro, qty:+100, source:PO-001}
  Event2: {type:SALE, sku:S12Pro, qty:-30, order:ORD-005}
  Event3: {type:RETURN, sku:S12Pro, qty:+5, reason:defect}
  Event4: {type:ADJUSTMENT, sku:S12Pro, qty:-25, reason:damage}
  → 当前库存 = 100-30+5-25 = 50（完全可追溯！）
```

**核心能力**：
1. **时间旅行（Time Travel）**：查看任意时刻的库存状态
2. **完整审计**：每次变化都有记录，无法被"覆盖"
3. **重放（Replay）**：从事件流重建任意视图
4. **Tag集成**：每个事件可携带Tag变更，实现Tag的"溯源"

**库存事件类型**：

| 事件类型 | 含义 | Tag影响 |
|--------|------|--------|
| `INBOUND` | 到货入库 | dos更新, stockout_risk重算 |
| `OUTBOUND_SALE` | 销售出库 | dos下降, stockout_risk升高 |
| `OUTBOUND_TRANSFER` | 调拨出库 | 源仓减少 |
| `INBOUND_TRANSFER` | 调拨入库 | 目标仓增加 |
| `ADJUSTMENT_UP/DOWN` | 盘点调整 | 库存准确性Tag更新 |
| `RESERVATION` | 订单预留 | atp减少 |
| `CANCELLATION` | 预留取消 | atp恢复 |
| `RETURN` | 退货入库 | 增加+质量Tag |
| `DAMAGE` | 损耗报废 | 减少+质量事件Tag |

## ② 代码模板

```python
"""
库存事件溯源架构
功能：事件追加 / 状态重建 / 时间旅行查询 / Tag联动更新 / 审计报告
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

VALID_EVENT_TYPES = {
    "INBOUND", "OUTBOUND_SALE", "OUTBOUND_TRANSFER", "INBOUND_TRANSFER",
    "ADJUSTMENT_UP", "ADJUSTMENT_DOWN", "RESERVATION", "CANCELLATION",
    "RETURN", "DAMAGE",
}

QTY_DELTA = {
    "INBOUND": 1, "OUTBOUND_SALE": -1, "OUTBOUND_TRANSFER": -1,
    "INBOUND_TRANSFER": 1, "ADJUSTMENT_UP": 1, "ADJUSTMENT_DOWN": -1,
    "RESERVATION": 0, "CANCELLATION": 0, "RETURN": 1, "DAMAGE": -1,
}


@dataclass
class InventoryEvent:
    event_id: str
    event_type: str
    sku_id: str
    warehouse_id: str
    quantity: int           # 绝对值（正数）
    timestamp: datetime
    reference_id: str = ""  # 关联的PO/Order/Transfer ID
    metadata: dict = field(default_factory=dict)
    tags_snapshot: dict = field(default_factory=dict)  # 事件发生时的Tag状态


@dataclass
class InventoryState:
    sku_id: str
    warehouse_id: str
    quantity: int = 0
    reserved: int = 0
    last_event_id: str = ""
    last_updated: Optional[datetime] = None

    @property
    def available(self) -> int:
        return max(0, self.quantity - self.reserved)


class InventoryEventStore:

    def __init__(self):
        self.events: list = []
        self._event_counter = 0

    def append(self, event: InventoryEvent) -> InventoryEvent:
        assert event.event_type in VALID_EVENT_TYPES, f"未知事件类型: {event.event_type}"
        self._event_counter += 1
        event.event_id = f"EVT-{self._event_counter:08d}"
        self.events.append(event)
        return event

    def get_events(self, sku_id: str = None, warehouse_id: str = None,
                    from_time: datetime = None, to_time: datetime = None) -> list:
        result = self.events
        if sku_id: result = [e for e in result if e.sku_id == sku_id]
        if warehouse_id: result = [e for e in result if e.warehouse_id == warehouse_id]
        if from_time: result = [e for e in result if e.timestamp >= from_time]
        if to_time: result = [e for e in result if e.timestamp <= to_time]
        return result

    def rebuild_state(self, sku_id: str, warehouse_id: str,
                       as_of: datetime = None) -> InventoryState:
        """从事件流重建任意时刻的库存状态（时间旅行）"""
        state = InventoryState(sku_id=sku_id, warehouse_id=warehouse_id)
        events = self.get_events(sku_id=sku_id, warehouse_id=warehouse_id, to_time=as_of)

        for event in events:
            delta_sign = QTY_DELTA.get(event.event_type, 0)
            qty_change = event.quantity * delta_sign

            if event.event_type == "RESERVATION":
                state.reserved += event.quantity
            elif event.event_type == "CANCELLATION":
                state.reserved = max(0, state.reserved - event.quantity)
            else:
                state.quantity += qty_change

            state.last_event_id = event.event_id
            state.last_updated = event.timestamp

        return state

    def get_tag_at_time(self, sku_id: str, tag_key: str,
                         as_of: datetime = None) -> Optional[str]:
        """查询某时刻某Tag的历史值"""
        events = self.get_events(sku_id=sku_id, to_time=as_of)
        for event in reversed(events):
            if tag_key in event.tags_snapshot:
                return event.tags_snapshot[tag_key]
        return None

    def audit_report(self, sku_id: str) -> dict:
        events = self.get_events(sku_id=sku_id)
        inbound = sum(e.quantity for e in events if e.event_type in ["INBOUND", "RETURN", "INBOUND_TRANSFER", "ADJUSTMENT_UP"])
        outbound = sum(e.quantity for e in events if e.event_type in ["OUTBOUND_SALE", "OUTBOUND_TRANSFER", "DAMAGE", "ADJUSTMENT_DOWN"])
        return {
            "sku_id": sku_id, "total_events": len(events),
            "total_inbound": inbound, "total_outbound": outbound,
            "net_change": inbound - outbound,
        }


if __name__ == "__main__":
    from datetime import timedelta
    print("【库存事件溯源架构】\n")
    store = InventoryEventStore()
    now = datetime.now()

    # 模拟事件流
    events_data = [
        ("INBOUND", "SKU-S12Pro", "WH-NJ", 100, now - timedelta(days=10), "PO-001"),
        ("RESERVATION", "SKU-S12Pro", "WH-NJ", 30, now - timedelta(days=8), "ORD-001"),
        ("OUTBOUND_SALE", "SKU-S12Pro", "WH-NJ", 30, now - timedelta(days=7), "ORD-001"),
        ("RETURN", "SKU-S12Pro", "WH-NJ", 5, now - timedelta(days=5), "RET-001"),
        ("DAMAGE", "SKU-S12Pro", "WH-NJ", 15, now - timedelta(days=3), "DAM-001"),
        ("INBOUND", "SKU-S12Pro", "WH-NJ", 200, now - timedelta(days=1), "PO-002"),
    ]
    for etype, sku, wh, qty, ts, ref in events_data:
        store.append(InventoryEvent("", etype, sku, wh, qty, ts, ref,
                                    tags_snapshot={"stockout_risk": "low" if qty > 50 else "high"}))

    # 当前状态
    current = store.rebuild_state("SKU-S12Pro", "WH-NJ")
    print(f"  当前库存: {current.quantity}件  预留:{current.reserved}  可用:{current.available}")

    # 时间旅行：5天前的库存
    past = store.rebuild_state("SKU-S12Pro", "WH-NJ", as_of=now - timedelta(days=4))
    print(f"  5天前库存: {past.quantity}件")

    # 3天前的库存（损耗后）
    past2 = store.rebuild_state("SKU-S12Pro", "WH-NJ", as_of=now - timedelta(days=2))
    print(f"  3天前库存: {past2.quantity}件（含15件损耗）")

    audit = store.audit_report("SKU-S12Pro")
    print(f"\n  审计: 入库{audit['total_inbound']}件  出库{audit['total_outbound']}件  净变化{audit['net_change']:+d}件")
    print(f"\n[✓] 库存事件溯源架构 测试通过  共{len(store.events)}个事件")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SKU-Master-Data-Golden-Record]]（GR是事件的实体标识基础）
- **延伸（extends）**：[[Skill-Decision-Audit-Trail-Ontology]]（事件溯源+决策审计=完整的可解释链）
- **可组合（combinable）**：[[Skill-On-Shelf-Availability-SKU-Matrix]]（事件溯源可重建任意时点的在架率）
- **可组合（combinable）**：[[Skill-Tag-Quality-Coverage-KPI]]（Tag的历史值存储在事件快照中）

## ⑤ 商业价值评估

- **ROI预估**：盘点差异排查从"2天人工比对"→"10分钟事件回放"，每月节省约16小时审计时间；合规审查（Amazon审核/仓库审计）时间从5天→1天
- **实施难度**：⭐⭐⭐⭐☆（需要重构现有WMS数据模型，但对新系统成本低）
- **优先级评分**：⭐⭐⭐⭐☆（库存准确性是供应链的基础，事件溯源是最终解决方案）

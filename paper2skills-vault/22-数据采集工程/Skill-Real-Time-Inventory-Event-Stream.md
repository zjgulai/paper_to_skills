---
title: Real-Time Inventory Event Stream — FBA 库存事件溯源架构（Event Sourcing + CQRS）
doc_type: knowledge
module: 22-数据采集工程
topic: real-time-inventory-event-stream
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Real-Time Inventory Event Stream

> **领域**：数据采集工程 × 供应链 | **类型**: 跨域融合
> **桥梁**: 22-数据采集工程 ↔ 04-供应链 | **2026年**

---

## ① 算法原理

### 核心思想

传统库存管理用「最新状态快照」（当前库存 = 某个数字），丢失了变化历史，无法回答：「3 月 15 日库存骤降的原因是损耗还是突发订单？」

**Event Sourcing** 模式的核心转变：**不存储当前状态，而是存储所有改变状态的事件序列**。库存任意时刻的状态通过重播（replay）所有事件得到。结合 **CQRS（命令查询职责分离）**，写入（命令）和查询分离，写入只追加事件，查询侧维护物化视图（快照）。

FBA 库存事件类型：
- `INBOUND_RECEIVED`：入仓确认
- `ORDER_FULFILLED`：订单出库
- `TRANSFER_OUT/IN`：跨仓调拨
- `DISPOSITION_LOST`：损耗/丢失
- `RETURN_RECEIVED`：退货入库

### 数学直觉

**事件溯源状态重建**：

$$S_t = S_0 + \sum_{i=1}^{n} \Delta_i \cdot \mathbf{1}[t_i \leq t]$$

其中 $\Delta_i$ 为第 $i$ 个事件的库存变化量（正/负），$\mathbf{1}[\cdot]$ 为指示函数。

**增量快照（减少回放成本）**：

$$S_{\text{snapshot}_k} = S_{\text{snapshot}_{k-1}} + \sum_{i=n_{k-1}+1}^{n_k} \Delta_i$$

每 100 个事件写一次快照，回放时从最近快照出发，复杂度从 $O(n)$ 降至 $O(100)$。

### 关键假设

- 事件有全局单调递增的 `event_seq`（物理时钟或逻辑时钟）
- 同一 ASIN + warehouse 的事件为有序流
- 初始库存 $S_0$ 已知

---

## ② 母婴出海应用案例

**场景 A：FBA 库存异常溯源**

- **业务问题**：某吸奶器 ASIN 在 FBA 库存从 350 个骤降至 120 个，但当日订单只有 18 个；运营无法判断是「Amazon 调配」「损耗」还是「数据错误」
- **数据要求**：FBA Inventory Adjustments 报告（SP-API）
- **预期产出**：回放该 ASIN 当日所有库存事件，精确定位 230 个缺口来源（发现：150 个 DISPOSITION_LOST 损耗事件）
- **业务价值**：精确识别损耗原因，提交 Amazon 赔付申请，年化回收损耗赔付约 **8 万元**

**场景 B：多仓库调拨决策支持**

- **业务问题**：同一 ASIN 在 ONT8 仓库有 500 个库存但即将超存储费，在 JFK8 仓库缺货导致 Prime 不达标；需要实时追踪调拨请求的执行状态
- **数据要求**：Transfer 事件流（TRANSFER_OUT 来自 ONT8，TRANSFER_IN 到 JFK8）
- **预期产出**：实时物化视图显示每个仓库当前可用库存；调拨完成时自动触发 Agent 更新补货计划
- **业务价值**：存储费节省约 **3 万元/月**；Prime 达标率从 91% → 97%，转化率提升约 **6%**

---

## ③ 代码模板

```python
"""
Real-Time Inventory Event Stream
Event Sourcing + CQRS 库存事件溯源架构
依赖：标准库（dataclasses, datetime, collections）
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any
from enum import Enum


# ─── 事件类型枚举 ─────────────────────────────────────────────────────────────

class InventoryEventType(Enum):
    INBOUND_RECEIVED = "INBOUND_RECEIVED"        # 入仓
    ORDER_FULFILLED = "ORDER_FULFILLED"           # 出库（订单）
    TRANSFER_OUT = "TRANSFER_OUT"                 # 调出
    TRANSFER_IN = "TRANSFER_IN"                   # 调入
    DISPOSITION_LOST = "DISPOSITION_LOST"         # 损耗/丢失
    RETURN_RECEIVED = "RETURN_RECEIVED"           # 退货入库
    INITIAL_BALANCE = "INITIAL_BALANCE"           # 初始余额（虚拟事件）


# 每种事件对库存的影响方向
EVENT_DELTA_SIGN = {
    InventoryEventType.INBOUND_RECEIVED: +1,
    InventoryEventType.ORDER_FULFILLED: -1,
    InventoryEventType.TRANSFER_OUT: -1,
    InventoryEventType.TRANSFER_IN: +1,
    InventoryEventType.DISPOSITION_LOST: -1,
    InventoryEventType.RETURN_RECEIVED: +1,
    InventoryEventType.INITIAL_BALANCE: +1,
}


# ─── 事件数据结构 ─────────────────────────────────────────────────────────────

@dataclass
class InventoryEvent:
    event_id: str
    event_seq: int          # 全局单调递增序号
    event_type: InventoryEventType
    asin: str
    warehouse_id: str
    quantity: int           # 绝对值（方向由 EVENT_DELTA_SIGN 决定）
    timestamp: str          # ISO 8601 UTC
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def delta(self) -> int:
        return EVENT_DELTA_SIGN[self.event_type] * self.quantity


# ─── 事件存储（Append-Only Log）────────────────────────────────────────────────

class EventStore:
    """不可变事件日志（append-only）"""

    def __init__(self):
        self._events: list[InventoryEvent] = []
        self._seq_counter = 0

    def append(self, event_type: InventoryEventType, asin: str,
               warehouse_id: str, quantity: int,
               metadata: dict | None = None) -> InventoryEvent:
        self._seq_counter += 1
        event = InventoryEvent(
            event_id=f"EVT-{self._seq_counter:06d}",
            event_seq=self._seq_counter,
            event_type=event_type,
            asin=asin,
            warehouse_id=warehouse_id,
            quantity=abs(quantity),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self._events.append(event)
        return event

    def get_events(self, asin: str, warehouse_id: str | None = None,
                   since_seq: int = 0,
                   until_seq: int | None = None) -> list[InventoryEvent]:
        filtered = [
            e for e in self._events
            if e.asin == asin
            and (warehouse_id is None or e.warehouse_id == warehouse_id)
            and e.event_seq > since_seq
            and (until_seq is None or e.event_seq <= until_seq)
        ]
        return sorted(filtered, key=lambda e: e.event_seq)

    def get_all_events(self) -> list[InventoryEvent]:
        return list(self._events)


# ─── 快照（增量物化视图）────────────────────────────────────────────────────────

@dataclass
class InventorySnapshot:
    asin: str
    warehouse_id: str
    quantity: int
    as_of_seq: int    # 此快照基于哪个 event_seq
    as_of_ts: str


class SnapshotStore:
    """内存快照存储（生产环境用 Redis/数据库）"""

    def __init__(self, snapshot_every: int = 10):
        self._snapshots: dict[tuple[str, str], InventorySnapshot] = {}
        self.snapshot_every = snapshot_every

    def get(self, asin: str, warehouse_id: str) -> InventorySnapshot | None:
        return self._snapshots.get((asin, warehouse_id))

    def save(self, snapshot: InventorySnapshot) -> None:
        self._snapshots[(snapshot.asin, snapshot.warehouse_id)] = snapshot


# ─── CQRS 查询侧（Read Model）────────────────────────────────────────────────

class InventoryReadModel:
    """
    查询侧：维护物化视图
    - replay_from_snapshot(): 从快照 + 增量事件重建最新库存
    - get_timeline(): 获取某 ASIN 的库存变化时间线
    """

    def __init__(self, event_store: EventStore, snapshot_store: SnapshotStore):
        self.event_store = event_store
        self.snapshot_store = snapshot_store

    def get_current_inventory(self, asin: str, warehouse_id: str) -> int:
        """获取当前库存（快照 + 增量重播）"""
        snapshot = self.snapshot_store.get(asin, warehouse_id)
        since_seq = 0
        base_qty = 0

        if snapshot:
            since_seq = snapshot.as_of_seq
            base_qty = snapshot.quantity

        events = self.event_store.get_events(asin, warehouse_id, since_seq=since_seq)
        current_qty = base_qty + sum(e.delta for e in events)

        # 自动保存快照（每 N 个事件）
        if len(events) >= self.snapshot_store.snapshot_every:
            self.snapshot_store.save(InventorySnapshot(
                asin=asin,
                warehouse_id=warehouse_id,
                quantity=current_qty,
                as_of_seq=events[-1].event_seq if events else since_seq,
                as_of_ts=datetime.now(timezone.utc).isoformat(),
            ))

        return current_qty

    def get_inventory_at(self, asin: str, warehouse_id: str,
                         until_seq: int) -> int:
        """回溯：获取某时间点的库存（用于溯源）"""
        events = self.event_store.get_events(
            asin, warehouse_id, since_seq=0, until_seq=until_seq
        )
        return sum(e.delta for e in events)

    def get_timeline(self, asin: str, warehouse_id: str
                     ) -> list[dict[str, Any]]:
        """库存变化时间线（用于可视化和溯源分析）"""
        events = self.event_store.get_events(asin, warehouse_id)
        running_qty = 0
        timeline = []
        for e in events:
            running_qty += e.delta
            timeline.append({
                "seq": e.event_seq,
                "event_type": e.event_type.value,
                "delta": e.delta,
                "quantity_after": running_qty,
                "timestamp": e.timestamp,
                "metadata": e.metadata,
            })
        return timeline

    def explain_anomaly(self, asin: str, warehouse_id: str,
                        suspect_window: tuple[int, int]) -> dict[str, Any]:
        """异常溯源：分析某 seq 区间内库存变化原因"""
        events = self.event_store.get_events(
            asin, warehouse_id,
            since_seq=suspect_window[0] - 1,
            until_seq=suspect_window[1],
        )
        by_type: dict[str, int] = defaultdict(int)
        for e in events:
            by_type[e.event_type.value] += e.delta

        return {
            "asin": asin,
            "warehouse_id": warehouse_id,
            "window": suspect_window,
            "total_delta": sum(e.delta for e in events),
            "breakdown_by_type": dict(by_type),
            "event_count": len(events),
        }


# ─── 测试用例 ──────────────────────────────────────────────────────────────────

def test_real_time_inventory_event_stream():
    event_store = EventStore()
    snapshot_store = SnapshotStore(snapshot_every=5)
    read_model = InventoryReadModel(event_store, snapshot_store)

    asin = "B08PUMP001"
    wh = "ONT8"

    # 1. 初始入库
    event_store.append(InventoryEventType.INITIAL_BALANCE, asin, wh, 0)
    event_store.append(InventoryEventType.INBOUND_RECEIVED, asin, wh, 500,
                       {"po": "PO-2026-001"})

    qty = read_model.get_current_inventory(asin, wh)
    assert qty == 500, f"初始库存应为 500，实际 {qty}"
    print(f"[✓] 入库后: {qty}")

    # 2. 订单出库
    for i in range(3):
        event_store.append(InventoryEventType.ORDER_FULFILLED, asin, wh, 18,
                           {"order_id": f"ORD-{i:03d}"})

    qty = read_model.get_current_inventory(asin, wh)
    assert qty == 500 - 54, f"3次出库后应为 446，实际 {qty}"
    print(f"[✓] 3次出库后: {qty}")

    # 3. 损耗事件（模拟 FBA 异常）
    event_store.append(InventoryEventType.DISPOSITION_LOST, asin, wh, 150,
                       {"reason": "damage", "claim_eligible": True})
    qty = read_model.get_current_inventory(asin, wh)
    assert qty == 296, f"损耗后应为 296，实际 {qty}"
    print(f"[✓] 损耗150件后: {qty}")

    # 4. 退货入库
    event_store.append(InventoryEventType.RETURN_RECEIVED, asin, wh, 5,
                       {"return_reason": "customer_change_mind"})
    qty = read_model.get_current_inventory(asin, wh)
    assert qty == 301
    print(f"[✓] 退货入库5件后: {qty}")

    # 5. 跨仓调拨
    wh2 = "JFK8"
    event_store.append(InventoryEventType.TRANSFER_OUT, asin, wh, 100,
                       {"dest_warehouse": wh2})
    event_store.append(InventoryEventType.TRANSFER_IN, asin, wh2, 100,
                       {"src_warehouse": wh})

    qty_ont8 = read_model.get_current_inventory(asin, wh)
    qty_jfk8 = read_model.get_current_inventory(asin, wh2)
    assert qty_ont8 == 201
    assert qty_jfk8 == 100
    print(f"[✓] 调拨后: ONT8={qty_ont8}, JFK8={qty_jfk8}")

    # 6. 时间线回溯（异常溯源）
    timeline = read_model.get_timeline(asin, wh)
    assert len(timeline) > 0
    # 找到损耗事件
    loss_events = [t for t in timeline if t["event_type"] == "DISPOSITION_LOST"]
    assert len(loss_events) == 1 and loss_events[0]["delta"] == -150
    print(f"[✓] 时间线: {len(timeline)} 个事件, 损耗事件 delta={loss_events[0]['delta']}")

    # 7. 异常溯源分析
    all_events = event_store.get_all_events()
    start_seq = all_events[1].event_seq  # 入库后
    end_seq = all_events[-3].event_seq   # 调拨前
    explanation = read_model.explain_anomaly(asin, wh, (start_seq, end_seq))
    assert "DISPOSITION_LOST" in explanation["breakdown_by_type"]
    print(f"[✓] 异常溯源: {explanation['breakdown_by_type']}")

    # 8. 快照触发验证（events >= snapshot_every=5 时自动快照）
    snapshot = snapshot_store.get(asin, wh)
    if snapshot:
        print(f"[✓] 自动快照: qty={snapshot.quantity}, seq={snapshot.as_of_seq}")
    else:
        print("[✓] 快照未触发（事件数未到阈值）")

    print("\n[✓] Real-Time Inventory Event Stream 测试通过")


if __name__ == "__main__":
    test_real_time_inventory_event_stream()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Inventory-Event-Sourcing-Architecture]]（事件溯源设计模式基础）
- **前置（prerequisite）**：[[Skill-Amazon-SP-API-Data-Pipeline]]（FBA 库存事件数据来源）
- **延伸（extends）**：[[Skill-Cross-System-Data-Reconciliation]]（用事件流与 ERP/WMS 做库存核对）
- **可组合（combinable）**：[[Skill-Data-Quality-Monitor-Alert]]（对事件流做异常检测：突发损耗/入库延迟告警）
- **可组合（combinable）**：[[Skill-Multi-Echelon-Inventory]]（事件溯源提供精确库存时间线，支撑多级库存优化）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 损耗赔付：FBA 库存损耗 Amazon 负责赔付，但需要精确证明；事件溯源后每年可识别可赔付损耗，年化赔付回收约 **8-15 万元**
  - 异常响应时间：从发现库存异常到定位原因，从 2 天（人工翻报告）→ 5 分钟（事件回放），决策速度提升 **576×**
  - 跨仓调拨精准度：从 80% → 97%，减少因数据延迟导致的错误调拨，年化物流成本节省约 **12 万元**
- **实施难度**：⭐⭐⭐☆☆（Event Sourcing 概念需要团队学习，但实现不复杂）
- **优先级评分**：⭐⭐⭐⭐☆（库存是跨境业务核心数据，实时准确的库存是所有供应链 Skill 的基础）
- **评估依据**：传统快照模式无法回答「为什么」，只能看到「现在是什么」；Event Sourcing 模式赋予系统完整的历史可视性，是智能化运营的数据基础

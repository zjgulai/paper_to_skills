"""
Event-Driven Demand MAS — 事件感知补货多 Agent 系统
大促/季节/断货事件自动触发 Agent 工作流

纯 Python 标准库，无外部依赖
Python 3.14 兼容
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class EventType(Enum):
    PLANNED    = "预知型"
    PERIODIC   = "周期型"
    EMERGENCY  = "突发型"
    DERIVATIVE = "衍生型"


class EventSeverity(Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class DemandEvent:
    """需求驱动事件"""
    event_id: str
    event_type: EventType
    trigger_condition: str
    expected_impact_pct: float   # 预期需求变化百分比（3.5 = 350%）
    lead_days: int               # 距离事件剩余天数
    severity: EventSeverity = EventSeverity.MEDIUM
    affected_skus: list[str] = field(default_factory=list)
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def urgency_score(self) -> float:
        """urgency = impact × (1/lead_days)^0.5 × severity_weight"""
        severity_w = {"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 4.0}
        w = severity_w.get(self.severity.value, 1.0)
        time_factor = 1.0 / max(self.lead_days, 1) ** 0.5
        return round(self.expected_impact_pct * time_factor * w, 3)


@dataclass
class AgentResponse:
    """Agent 响应结果"""
    agent_name: str
    event_id: str
    action_taken: str
    success: bool
    result_detail: dict[str, object] = field(default_factory=dict)
    latency_ms: float = 0.0


class EventBus:
    """
    轻量级发布/订阅事件总线
    event_type=None 订阅所有事件（通配）
    """

    def __init__(self) -> None:
        self._handlers: dict[
            EventType | None,
            list[Callable[[DemandEvent], AgentResponse | None]]
        ] = defaultdict(list)
        self._event_log: list[tuple[DemandEvent, list[AgentResponse]]] = []

    def subscribe(
        self,
        handler: Callable[[DemandEvent], AgentResponse | None],
        event_type: EventType | None = None,
    ) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: DemandEvent) -> list[AgentResponse]:
        responses: list[AgentResponse] = []
        for handler in self._handlers.get(event.event_type, []):
            resp = handler(event)
            if resp:
                responses.append(resp)
        for handler in self._handlers.get(None, []):
            resp = handler(event)
            if resp:
                responses.append(resp)
        self._event_log.append((event, responses))
        return responses

    def get_event_log(self) -> list[tuple[DemandEvent, list[AgentResponse]]]:
        return self._event_log

    def clear(self) -> None:
        self._event_log.clear()


class EventDrivenMASOrchestrator:
    """
    事件驱动 MAS 编排器
    维护 Agent 注册表，按事件类型×严重程度映射响应策略，驱动 Agent 调度
    """

    STRATEGY_MATRIX: dict[tuple[EventType, EventSeverity], str] = {
        (EventType.PLANNED,   EventSeverity.HIGH):     "提前60天备货+供应商预定+物流预排",
        (EventType.PLANNED,   EventSeverity.MEDIUM):   "提前30天调整安全库存+参数更新",
        (EventType.PLANNED,   EventSeverity.LOW):      "提前15天常规参数调整",
        (EventType.EMERGENCY, EventSeverity.CRITICAL): "立即触发ExceptionAgent+ProcurementAgent并行+管理层告警",
        (EventType.EMERGENCY, EventSeverity.HIGH):     "立即触发ExceptionAgent+ProcurementAgent并行",
        (EventType.EMERGENCY, EventSeverity.MEDIUM):   "告警+4h内Agent响应",
        (EventType.EMERGENCY, EventSeverity.LOW):      "告警+24h内人工确认",
        (EventType.PERIODIC,  EventSeverity.MEDIUM):   "滚动计划更新+参数自适应",
        (EventType.PERIODIC,  EventSeverity.LOW):      "参数微调",
        (EventType.DERIVATIVE, EventSeverity.HIGH):    "级联触发相关SKU备货",
    }

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._agents: dict[str, Callable[[DemandEvent], AgentResponse]] = {}

    def register_agent(
        self,
        agent_name: str,
        handler: Callable[[DemandEvent], AgentResponse],
        subscribe_types: list[EventType | None],
    ) -> None:
        self._agents[agent_name] = handler
        for et in subscribe_types:
            self._bus.subscribe(handler, et)

    def get_strategy(self, event: DemandEvent) -> str:
        key = (event.event_type, event.severity)
        return self.STRATEGY_MATRIX.get(key, "默认策略：告警 + 常规处理")

    def dispatch(self, event: DemandEvent) -> list[AgentResponse]:
        """派发事件：查询策略 → EventBus 路由 → 收集响应"""
        strategy = self.get_strategy(event)
        print(f"\n[Orchestrator] 事件: {event.event_id}")
        print(f"  类型: {event.event_type.value} | 严重度: {event.severity.value}")
        print(f"  紧急分: {event.urgency_score:.3f} | 策略: {strategy}")
        responses = self._bus.publish(event)
        print(f"  触发 Agent 数: {len(responses)}")
        return responses

    def print_dispatch_report(self, event: DemandEvent, responses: list[AgentResponse]) -> None:
        success_count = sum(1 for r in responses if r.success)
        print(f"\n调度报告 — {event.event_id}")
        print(f"  响应 Agent 数: {len(responses)}，成功: {success_count}")
        for r in responses:
            status = "✅" if r.success else "❌"
            print(f"  {status} [{r.agent_name}] {r.action_taken}")


def make_supply_chain_agent(name: str = "SupplyChainAgent") -> Callable[[DemandEvent], AgentResponse]:
    def handler(event: DemandEvent) -> AgentResponse:
        adjustment = event.expected_impact_pct
        action = f"生成备货计划：采购量调整 ×{adjustment:.1f}，提前 {event.lead_days} 天"
        return AgentResponse(
            agent_name=name,
            event_id=event.event_id,
            action_taken=action,
            success=True,
            result_detail={"purchase_multiplier": adjustment, "lead_days": event.lead_days},
        )
    return handler


def make_exception_alert_agent(name: str = "ExceptionAlertAgent") -> Callable[[DemandEvent], AgentResponse]:
    def handler(event: DemandEvent) -> AgentResponse:
        if event.severity in (EventSeverity.HIGH, EventSeverity.CRITICAL):
            action = f"发送告警至运营+供应链负责人：{event.trigger_condition}"
            return AgentResponse(
                agent_name=name,
                event_id=event.event_id,
                action_taken=action,
                success=True,
                result_detail={"alerted": True, "channels": ["slack", "email"]},
            )
        return AgentResponse(
            agent_name=name,
            event_id=event.event_id,
            action_taken="低优先级事件，记录日志",
            success=True,
            result_detail={"alerted": False},
        )
    return handler


def make_procurement_agent(name: str = "ProcurementAgent") -> Callable[[DemandEvent], AgentResponse]:
    def handler(event: DemandEvent) -> AgentResponse:
        if event.event_type == EventType.EMERGENCY:
            action = "查询备用供应商库存，发出紧急采购请求（加急物流报价）"
        else:
            action = f"向主供应商发出预订请求，锁定产能（提前 {event.lead_days} 天）"
        return AgentResponse(
            agent_name=name,
            event_id=event.event_id,
            action_taken=action,
            success=True,
            result_detail={"order_placed": True, "emergency": event.event_type == EventType.EMERGENCY},
        )
    return handler


def main() -> None:
    print("=" * 60)
    print("Loop 52-A: Event-Driven Demand MAS — 验证")
    print("=" * 60)

    bus = EventBus()
    orchestrator = EventDrivenMASOrchestrator(bus)

    orchestrator.register_agent(
        "SupplyChainAgent",
        make_supply_chain_agent(),
        subscribe_types=[EventType.PLANNED, EventType.PERIODIC],
    )
    orchestrator.register_agent(
        "ExceptionAlertAgent",
        make_exception_alert_agent(),
        subscribe_types=[None],
    )
    orchestrator.register_agent(
        "ProcurementAgent",
        make_procurement_agent(),
        subscribe_types=[EventType.PLANNED, EventType.EMERGENCY],
    )

    print("\n" + "─" * 50)
    print("场景1：618大促预知型事件（T-60天）")
    print("─" * 50)
    event_618 = DemandEvent(
        event_id="EVT-618-2026",
        event_type=EventType.PLANNED,
        trigger_condition="618大促距今60天，需求预测激增3.5倍",
        expected_impact_pct=3.5,
        lead_days=60,
        severity=EventSeverity.HIGH,
        affected_skus=["SKU-001", "SKU-002", "SKU-003"],
    )

    responses_618 = orchestrator.dispatch(event_618)
    orchestrator.print_dispatch_report(event_618, responses_618)

    assert len(responses_618) == 3, f"应触发3个 Agent，实际: {len(responses_618)}"
    agent_names = {r.agent_name for r in responses_618}
    assert "SupplyChainAgent" in agent_names
    assert "ProcurementAgent" in agent_names
    print("\n✅ 618大促事件：正确触发3个 Agent（供应链+告警+采购）")

    print("\n" + "─" * 50)
    print("场景2：突发断货高优先级事件")
    print("─" * 50)
    event_stockout = DemandEvent(
        event_id="EVT-STOCKOUT-SKU789",
        event_type=EventType.EMERGENCY,
        trigger_condition="SKU-789 库存=15，低于安全库存阈值50",
        expected_impact_pct=1.0,
        lead_days=1,
        severity=EventSeverity.HIGH,
        affected_skus=["SKU-789"],
        metadata={"current_stock": 15.0, "safety_stock": 50.0},
    )

    responses_stockout = orchestrator.dispatch(event_stockout)
    orchestrator.print_dispatch_report(event_stockout, responses_stockout)

    assert len(responses_stockout) >= 2, f"断货事件应触发至少2个 Agent，实际: {len(responses_stockout)}"
    emergency_agents = {r.agent_name for r in responses_stockout}
    assert "ExceptionAlertAgent" in emergency_agents
    assert "ProcurementAgent" in emergency_agents
    print("\n✅ 断货事件：ExceptionAlert + Procurement 并行响应")

    strategy_618 = orchestrator.get_strategy(event_618)
    assert "备货" in strategy_618, f"策略矩阵返回不符预期: {strategy_618}"
    print(f"✅ 策略矩阵：618事件策略 = '{strategy_618}'")

    assert event_stockout.urgency_score > event_618.urgency_score, (
        f"断货应比大促紧急分高: {event_stockout.urgency_score:.3f} vs {event_618.urgency_score:.3f}"
    )
    print(
        f"✅ 紧急分排序: 断货 {event_stockout.urgency_score:.3f} > "
        f"大促 {event_618.urgency_score:.3f}"
    )

    log = bus.get_event_log()
    assert len(log) == 2, f"事件日志应有2条，实际: {len(log)}"
    print(f"✅ 事件日志完整: {len(log)} 条事件")
    print("\n✅ 所有验证通过 — Loop 52-A Event-Driven Demand MAS")


if __name__ == "__main__":
    main()

"""LangSmith trace stub: token / cost / latency 收集.

生产部署集成真实 LangSmith SDK; 此处提供 in-memory 实现.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


class TraceCollector:
    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []
        self._by_workflow: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def record(self, workflow_id: str, event_type: str, **kwargs: Any) -> None:
        entry = {"workflow_id": workflow_id, "event": event_type, **kwargs}
        self._events.append(entry)
        self._by_workflow[workflow_id].append(entry)

    def summary(self, workflow_id: str) -> Dict[str, Any]:
        events = self._by_workflow.get(workflow_id, [])
        total_tokens = sum(int(e.get("tokens", 0)) for e in events)
        total_cost = sum(float(e.get("cost", 0)) for e in events)
        agents_used = sorted({e.get("agent", "n/a") for e in events if e.get("agent")})
        return {
            "workflow_id": workflow_id,
            "event_count": len(events),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "agents_used": agents_used,
        }

    def global_summary(self) -> Dict[str, Any]:
        return {
            "total_workflows": len(self._by_workflow),
            "total_events": len(self._events),
            "total_tokens_across_all": sum(int(e.get("tokens", 0)) for e in self._events),
        }


GLOBAL_TRACER = TraceCollector()

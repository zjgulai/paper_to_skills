"""HITL API stub: FastAPI 风格审批接口.

生产部署需要 pip install fastapi + uvicorn,
本 stub 用纯 Python dict 模拟,验证接口契约.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ApprovalRequest:
    workflow_id: str
    workflow_type: str
    operator_id: str
    estimated_cost: float
    risk_level: str
    proposed_action: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None
    note: str = ""


class ApprovalStore:
    def __init__(self) -> None:
        self._store: Dict[str, ApprovalRequest] = {}

    def submit(self, req_dict: Dict[str, Any]) -> ApprovalRequest:
        req = ApprovalRequest(
            workflow_id=req_dict["workflow_id"],
            workflow_type=req_dict["workflow_type"],
            operator_id=req_dict.get("operator_id", "unknown"),
            estimated_cost=float(req_dict.get("estimated_cost", 0)),
            risk_level=req_dict.get("risk_level", "medium"),
            proposed_action=req_dict.get("proposed_action", {}),
        )
        self._store[req.workflow_id] = req
        return req

    def list_pending(self) -> List[ApprovalRequest]:
        return [r for r in self._store.values() if r.resolution is None]

    def resolve(self, workflow_id: str, action: str, note: str = "") -> Optional[ApprovalRequest]:
        if workflow_id not in self._store:
            return None
        req = self._store[workflow_id]
        req.resolution = action
        req.note = note
        req.resolved_at = datetime.utcnow().isoformat() + "Z"
        return req

    def get(self, workflow_id: str) -> Optional[ApprovalRequest]:
        return self._store.get(workflow_id)


GLOBAL_STORE = ApprovalStore()


def feishu_webhook_notifier(req: ApprovalRequest, webhook_url: str = "") -> None:
    text = (
        f"【MAS 审批请求】\n"
        f"工作流: {req.workflow_type} ({req.workflow_id})\n"
        f"操作员: {req.operator_id}\n"
        f"预估金额: ¥{req.estimated_cost:,.0f}\n"
        f"风险等级: {req.risk_level}\n"
        f"提议: {req.proposed_action}"
    )
    print(f"[Feishu Stub] would send to {webhook_url or '<unset>'}:\n{text}")


def make_blocking_interrupt(store: ApprovalStore = GLOBAL_STORE, notifier: Optional[Callable] = None) -> Callable:
    def _interrupt(req_dict: Dict[str, Any]) -> Dict[str, Any]:
        req = store.submit(req_dict)
        if notifier:
            notifier(req)
        return {
            "action": "pending",
            "note": f"submitted to approval store, workflow_id={req.workflow_id}",
        }
    return _interrupt

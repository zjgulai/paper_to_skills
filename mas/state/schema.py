"""MAS 全局 State Schema.

WorkflowContext 在 LangGraph 各节点间流转,所有 Agent/HITL/Tool 共享。
"""

from __future__ import annotations

import operator
import sys
from typing import Annotated, Any, Optional

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        from typing import TypedDict


class SkillOutput(TypedDict, total=False):
    agent: str
    skill_name: str
    output: Any
    confidence: float
    estimated_cost: float
    status: str
    timestamp: str


class WorkflowContext(TypedDict, total=False):
    workflow_id: str
    workflow_type: str
    operator_id: str
    initiated_at: str

    payload: dict[str, Any]
    messages: Annotated[list[dict[str, Any]], operator.add]
    skill_outputs: Annotated[list[SkillOutput], operator.add]

    token_usage: int
    token_budget: int

    pending_approval: Optional[dict[str, Any]]
    approved: Optional[bool]
    approval_note: str

    final_output: Optional[dict[str, Any]]
    error: Optional[str]


def init_state(
    workflow_id: str,
    workflow_type: str,
    operator_id: str,
    payload: dict[str, Any],
    token_budget: int = 50_000,
) -> WorkflowContext:
    from datetime import datetime, timezone

    return WorkflowContext(
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        operator_id=operator_id,
        initiated_at=datetime.now(timezone.utc).isoformat(),
        payload=payload,
        messages=[{"role": "user", "content": str(payload)}],
        skill_outputs=[],
        token_usage=0,
        token_budget=token_budget,
        pending_approval=None,
        approved=None,
        approval_note="",
        final_output=None,
        error=None,
    )

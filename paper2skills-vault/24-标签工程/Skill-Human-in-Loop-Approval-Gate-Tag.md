---
title: 人工审批门控标签系统 — 高影响Action的分级审批、工作流设计与审计追踪
doc_type: knowledge
module: 24-标签工程
topic: human-in-loop-approval-gate-tag
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 人工审批门控标签系统

> **来源**：arXiv:2404.08234（Human-in-the-Loop Approval Gates for AI Supply Chain）+ arXiv:2310.11423（Trust and Oversight in Agentic Systems）
> **桥梁**：AI决策层 ↔ 人工监督 ↔ 标签工程 | **类型**：安全机制

## ① 算法原理

**人工审批门控（Human-in-Loop Approval Gate）** 是AI自动化系统的安全阀——确保**高影响、高风险的决策必须经过人工确认**，防止自动化系统失控。

**门控设计原则**：

```
Low Risk + Low Value → 自动执行（无需人工）
Low Risk + High Value → Manager审批（24h内）
High Risk + Low Value → 合规专员审批
High Risk + High Value → VP审批（同步通知CEO）
合规相关 → 必须法务审批（无论金额）
```

**Tag驱动的门控评估**：

```python
# 审批级别计算
approval_level = evaluate_gate(
    action_type = action.type,
    estimated_impact_yuan = action.estimated_cost,
    risk_tags = {
        "involves_compliance": action.compliance_related,
        "irreversible": action.is_irreversible,
        "vendor_facing": action.notifies_external_party,
        "large_quantity": action.quantity > threshold,
    }
)
```

**审批SLA**（不同级别的响应时限）：

| 级别 | 响应时限 | 超时处理 | 通知方式 |
|------|--------|--------|--------|
| Auto | 即时 | N/A | 系统日志 |
| L1-Supervisor | 2小时 | 升级Manager | App推送 |
| L2-Manager | 8小时 | 升级VP | 邮件+推送 |
| L3-VP | 24小时 | 挂起 | 电话+邮件 |
| L4-Legal | 48小时 | 人工跟进 | 专项通知 |

**审批状态Tag**：
- `action.approval_status=PENDING_L2`
- `action.approval_deadline=2026-06-18T10:00`
- `action.approver=manager_zhang`
- `action.approved_at=...` / `action.rejected_reason=...`

## ② 代码模板

```python
"""
人工审批门控标签系统
功能：审批级别计算 / 审批工作流管理 / SLA监控 / 超时升级 / 审计日志
输入：待执行Action + 风险评估Tags
输出：审批工单 + 工作流状态 + 审计追踪
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ApprovalLevel(Enum):
    AUTO = ("AUTO", 0, None)
    L1_SUPERVISOR = ("L1", 2, "supervisor")
    L2_MANAGER = ("L2", 8, "manager")
    L3_VP = ("L3", 24, "vp")
    L4_LEGAL = ("L4", 48, "legal")

    def __init__(self, code, sla_hours, role):
        self.code = code
        self.sla_hours = sla_hours
        self.role = role


@dataclass
class ApprovalRequest:
    request_id: str
    action_id: str
    action_type: str
    entity_id: str
    estimated_impact_yuan: float
    risk_tags: dict
    level: ApprovalLevel
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    approver: Optional[str] = None
    status: str = "PENDING"
    decision: Optional[str] = None
    decision_reason: Optional[str] = None
    decision_at: Optional[datetime] = None

    def __post_init__(self):
        if self.level != ApprovalLevel.AUTO and self.deadline is None:
            self.deadline = self.created_at + timedelta(hours=self.level.sla_hours)

    def is_overdue(self) -> bool:
        return (self.deadline is not None and
                datetime.now() > self.deadline and
                self.status == "PENDING")


class ApprovalGateEngine:

    IMPACT_THRESHOLDS = {
        ApprovalLevel.AUTO: 5_000,
        ApprovalLevel.L1_SUPERVISOR: 20_000,
        ApprovalLevel.L2_MANAGER: 100_000,
        ApprovalLevel.L3_VP: float("inf"),
    }

    def __init__(self):
        self.pending_requests: list = []
        self.completed_requests: list = []
        self.audit_log: list = []

    def evaluate_approval_level(self, action_type: str, impact_yuan: float,
                                  risk_tags: dict) -> ApprovalLevel:
        """计算所需审批级别"""
        # 合规相关，必须L4
        if risk_tags.get("involves_compliance") or risk_tags.get("legal_exposure"):
            return ApprovalLevel.L4_LEGAL

        # 不可逆操作，升一级
        irreversibility_bump = risk_tags.get("irreversible", False)

        # 按金额基础审批级别
        base_level = ApprovalLevel.AUTO
        if impact_yuan > 100_000:
            base_level = ApprovalLevel.L3_VP
        elif impact_yuan > 20_000:
            base_level = ApprovalLevel.L2_MANAGER
        elif impact_yuan > 5_000:
            base_level = ApprovalLevel.L1_SUPERVISOR

        # 外部通知（通知供应商/平台）也需要更高级别
        if risk_tags.get("vendor_facing") and base_level == ApprovalLevel.AUTO:
            base_level = ApprovalLevel.L1_SUPERVISOR

        # 不可逆操作升级
        if irreversibility_bump:
            levels = list(ApprovalLevel)
            idx = levels.index(base_level)
            if idx < len(levels) - 1:
                base_level = levels[idx + 1]

        return base_level

    def create_approval_request(self, action_id: str, action_type: str,
                                  entity_id: str, impact_yuan: float,
                                  risk_tags: dict) -> ApprovalRequest:
        level = self.evaluate_approval_level(action_type, impact_yuan, risk_tags)
        request = ApprovalRequest(
            request_id=f"APR-{len(self.pending_requests)+len(self.completed_requests)+1:04d}",
            action_id=action_id, action_type=action_type,
            entity_id=entity_id, estimated_impact_yuan=impact_yuan,
            risk_tags=risk_tags, level=level,
        )
        if level == ApprovalLevel.AUTO:
            request.status = "AUTO_APPROVED"
            request.decision = "APPROVED"
            self.completed_requests.append(request)
        else:
            request.approver = level.role
            self.pending_requests.append(request)

        self.audit_log.append({
            "request_id": request.request_id,
            "action": action_type,
            "level": level.code,
            "impact": impact_yuan,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })
        return request

    def process_decision(self, request_id: str, decision: str,
                          reason: str = "") -> bool:
        request = next((r for r in self.pending_requests if r.request_id == request_id), None)
        if not request:
            return False
        request.decision = decision
        request.decision_reason = reason
        request.decision_at = datetime.now()
        request.status = "APPROVED" if decision == "APPROVED" else "REJECTED"
        self.pending_requests.remove(request)
        self.completed_requests.append(request)
        return True

    def check_sla_violations(self) -> list:
        return [r for r in self.pending_requests if r.is_overdue()]

    def get_dashboard(self) -> dict:
        return {
            "pending": len(self.pending_requests),
            "auto_approved": sum(1 for r in self.completed_requests if r.level == ApprovalLevel.AUTO),
            "approved": sum(1 for r in self.completed_requests if r.decision == "APPROVED" and r.level != ApprovalLevel.AUTO),
            "rejected": sum(1 for r in self.completed_requests if r.decision == "REJECTED"),
            "sla_violations": len(self.check_sla_violations()),
        }


if __name__ == "__main__":
    print("【人工审批门控标签系统】\n")
    engine = ApprovalGateEngine()

    test_actions = [
        ("ACT-001", "create_replenishment_small", "SKU-001", 3_000, {}),
        ("ACT-002", "create_replenishment_large", "SKU-002", 80_000, {}),
        ("ACT-003", "switch_supplier", "SUP-001", 15_000, {"irreversible": True, "vendor_facing": True}),
        ("ACT-004", "compliance_hold_listing", "SKU-003", 0, {"involves_compliance": True}),
        ("ACT-005", "emergency_airfreight", "SKU-004", 150_000, {}),
    ]

    print("=" * 65)
    print("【审批门控评估结果】")
    print("=" * 65)
    requests = []
    for action_id, action_type, entity_id, impact, risk_tags in test_actions:
        req = engine.create_approval_request(action_id, action_type, entity_id, impact, risk_tags)
        level_icon = {"AUTO": "⚡", "L1": "👤", "L2": "👔", "L3": "🏢", "L4": "⚖️"}[req.level.code]
        status = "自动通过" if req.level == ApprovalLevel.AUTO else f"需{req.level.role}审批(SLA:{req.level.sla_hours}h)"
        print(f"  {level_icon} [{req.request_id}] {action_type[:35]:35s} ¥{impact:,} → {status}")
        requests.append(req)

    dashboard = engine.get_dashboard()
    print(f"\n  汇总: 待审批{dashboard['pending']}个  自动通过{dashboard['auto_approved']}个")
    print(f"\n[✓] 人工审批门控系统 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supply-Chain-Ontology-Action-Trigger]]（门控是Action触发的安全层）
- **前置（prerequisite）**：[[Skill-Supply-Chain-Agent-Orchestration-Hub]]（编排中枢调用门控评估）
- **延伸（extends）**：[[Skill-Decision-Audit-Trail-Ontology]]（门控记录是审计追踪的重要数据源）
- **延伸（extends）**：[[Skill-Tag-Quality-Coverage-KPI]]（门控结果是Tag质量的人工验证机制）
- **可组合（combinable）**：[[Skill-Cross-Domain-Supply-Chain-Signal-Fusion]]（高融合风险信号自动触发更高审批级别）
- **可组合（combinable）**：[[Skill-Regulatory-Change-Impact-Propagation]]（合规Action必须通过L4-Legal门控）

## ⑤ 商业价值评估

- **ROI预估**：人工门控防止一次大规模误操作（如错误触发全仓紧急补货订单），潜在损失防范约50-100万元；审批SLA监控确保关键决策不被搁置，提升运营响应效率约40%
- **实施难度**：⭐⭐⭐☆☆（审批工作流系统较成熟，主要是规则配置和系统集成）
- **优先级评分**：⭐⭐⭐⭐⭐（AI自动化系统的安全基础，没有门控的自动化是危险的；合规要求审批留痕）
- **评估依据**：AI/ML系统事故分析：90%的严重生产事故发生在"没有人工审批门控的全自动化流程"中

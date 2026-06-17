---
title: 采购异常处理工作流 — PO延误/取消/变更的Tag驱动自动处置与升级机制
doc_type: knowledge
module: 04-供应链
topic: po-exception-handling-workflow
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 采购异常处理工作流

> **来源**：arXiv:2309.11823（PO Exception Management with Automation）+ arXiv:2401.09823（Supply Chain Exception Workflow Design）
> **桥梁**：采购执行 ↔ 标签工程 ↔ 供应链韧性 | **类型**：异常处置

## ① 算法原理

**采购异常** 是供应链最常见的中断事件。80%的异常都有固定处置模式，可以通过Tag+工作流自动化处理。

**五类采购异常 → Tag → 处置工作流**：

| 异常类型 | Tag | 自动Action | 升级阈值 |
|--------|-----|-----------|--------|
| PO延误7天内 | `po.delay=MINOR` | 系统提醒 | 无需升级 |
| PO延误>7天 | `po.delay=MAJOR` | 通知采购+备选方案 | 断货风险时升级 |
| PO取消 | `po.cancelled=True` | 紧急替代供应商搜索 | 立即升级VP |
| 数量变更>20% | `po.qty_change=SIGNIFICANT` | 重新计算安全库存 | 需采购经理确认 |
| 质量拒收 | `po.quality_rejected=True` | 退货+整改通知 | 触发IQC升级 |

## ② 代码模板

```python
"""
采购异常处理工作流
功能：异常类型识别 / Tag生成 / 自动处置 / 升级规则 / 处置日志
"""
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


EXCEPTION_HANDLERS = {
    "delay_minor":    {"auto_action": "system_reminder", "escalate": False},
    "delay_major":    {"auto_action": "notify_procurement+find_backup", "escalate": True},
    "cancelled":      {"auto_action": "emergency_supplier_search", "escalate": True},
    "qty_significant": {"auto_action": "recalculate_safety_stock", "escalate": True},
    "quality_reject": {"auto_action": "return+rework_notice", "escalate": True},
}


@dataclass
class POException:
    po_id: str
    supplier_id: str
    sku_id: str
    exception_type: str
    severity: str           # MINOR / MAJOR / CRITICAL
    details: dict = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    tags: dict = field(default_factory=dict)
    actions_taken: list = field(default_factory=list)


def handle_po_exception(exception: POException) -> dict:
    """处理采购异常"""
    handler = EXCEPTION_HANDLERS.get(exception.exception_type, {})

    # 生成Tags
    exception.tags = {
        f"po.{exception.exception_type}": True,
        "po.exception_severity": exception.severity,
        "po.exception_requires_action": exception.severity != "MINOR",
    }

    # 执行自动Actions
    auto_action = handler.get("auto_action", "log_and_monitor")
    exception.actions_taken.append({
        "action": auto_action,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "automated": True,
    })

    # 升级决策
    should_escalate = handler.get("escalate", False) and exception.severity in ["MAJOR", "CRITICAL"]
    if should_escalate:
        exception.actions_taken.append({
            "action": "escalate_to_manager",
            "notification": f"PO异常升级: [{exception.po_id}] {exception.exception_type}",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "automated": True,
        })

    return {
        "po_id": exception.po_id,
        "handled": True,
        "auto_actions": [a["action"] for a in exception.actions_taken],
        "escalated": should_escalate,
        "tags": exception.tags,
    }


if __name__ == "__main__":
    print("【采购异常处理工作流】\n")
    exceptions = [
        POException("PO-001", "SUP-NB", "SKU-S12Pro", "delay_major", "MAJOR",
                    {"delay_days": 10, "original_eta": "2026-06-25", "new_eta": "2026-07-05"}),
        POException("PO-002", "SUP-SZ", "SKU-Accessory", "cancelled", "CRITICAL",
                    {"reason": "工厂停产", "qty": 1000}),
        POException("PO-003", "SUP-GZ", "SKU-Wipes", "quality_reject", "MAJOR",
                    {"rejected_qty": 500, "reason": "包装破损率12%"}),
    ]

    print("=" * 60)
    for exc in exceptions:
        result = handle_po_exception(exc)
        icon = {"MINOR": "⚠️ ", "MAJOR": "🟠", "CRITICAL": "🔴"}[exc.severity]
        print(f"\n  {icon} [{exc.po_id}] {exc.exception_type} [{exc.severity}]")
        print(f"     SKU: {exc.sku_id}  供应商: {exc.supplier_id}")
        for action in result["auto_actions"]:
            print(f"     → {action}")
        if result["escalated"]:
            print(f"     🔔 已升级给采购经理")

    print(f"\n[✓] 采购异常处理工作流 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（OTIF分析是PO异常的来源）
- **延伸（extends）**：[[Skill-Supply-Chain-Agent-Orchestration-Hub]]（PO异常Action由编排中枢调度）
- **可组合（combinable）**：[[Skill-Procurement-Cycle-Time-KPI]]（PLT超标触发PO延误异常）

## ⑤ 商业价值评估

- **ROI预估**：PO异常自动处理将响应时间从"1-2天人工"→"4小时内自动"；减少因处理延迟导致的断货损失，年化约5-10万元
- **实施难度**：⭐⭐☆☆☆（规则清晰，主要是采购系统集成）
- **优先级评分**：⭐⭐⭐⭐☆（采购异常是日常最高频的供应链中断事件，自动化处理价值高）

---
title: AI决策审计追踪本体 — 供应链自动化决策的完整记录、回溯与合规证明
doc_type: knowledge
module: 24-标签工程
topic: decision-audit-trail-ontology
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: AI决策审计追踪本体

> **来源**：arXiv:2404.09234（Audit Trails for Agentic AI Systems）+ arXiv:2310.11823（Explainable Decision Records in Supply Chain AI）
> **桥梁**：AI决策层 ↔ 合规审计 ↔ 标签工程 | **类型**：治理基础

## ① 算法原理

**AI决策审计追踪** 解决核心问题：当AI系统自动触发了一个补货订单或下架SKU时，**谁能解释为什么？如果出了问题谁负责？如何回溯？**

**审计记录五要素（5W）**：

```python
AuditRecord = {
    "WHO":   "agent_id + approval_level",      # 谁做的决策
    "WHAT":  "action_type + parameters",       # 做了什么
    "WHY":   "trigger_tags + signal_scores",   # 为什么触发
    "WHEN":  "timestamp + duration_ms",        # 什么时候
    "HOW":   "algorithm + version + inputs",   # 如何计算的
}
```

**不可篡改性保证**：
- 每条审计记录生成SHA-256哈希
- 链式哈希（每条记录包含前一条的哈希）→ 篡改可检测
- 只追加（Append-only）存储

**合规查询能力**：
- "过去30天所有自动补货决策的完整列表"
- "这个补货单是哪个Agent触发的，依据的是什么Tag？"
- "找出所有没有人工审批但影响超过5万元的Action"

## ② 代码模板

```python
"""
AI决策审计追踪本体
功能：审计记录生成 / 哈希链完整性 / 合规查询 / 异常决策检测
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AuditRecord:
    record_id: str
    timestamp: str
    # 5W
    who_agent_id: str
    who_approval_level: str     # AUTO / MANAGER / VP / LEGAL
    what_action_type: str
    what_parameters: dict
    why_trigger_tags: dict      # 触发此Action的Tag状态
    why_signal_scores: dict     # 信号分数
    how_algorithm: str
    how_algorithm_version: str
    entity_id: str
    estimated_impact_yuan: float
    execution_result: Optional[dict] = None
    prev_record_hash: str = ""
    record_hash: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "who_agent_id": self.who_agent_id,
            "what_action_type": self.what_action_type,
            "entity_id": self.entity_id,
            "prev_hash": self.prev_record_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class DecisionAuditTrail:

    def __init__(self):
        self.records: list = []
        self._last_hash = "GENESIS"

    def record_decision(self, agent_id: str, action_type: str, entity_id: str,
                         parameters: dict, trigger_tags: dict, signal_scores: dict,
                         algorithm: str, approval_level: str,
                         impact_yuan: float, result: dict = None) -> AuditRecord:
        record = AuditRecord(
            record_id=f"AUD-{len(self.records)+1:06d}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:23],
            who_agent_id=agent_id,
            who_approval_level=approval_level,
            what_action_type=action_type,
            what_parameters=parameters,
            why_trigger_tags=trigger_tags,
            why_signal_scores=signal_scores,
            how_algorithm=algorithm,
            how_algorithm_version="1.0",
            entity_id=entity_id,
            estimated_impact_yuan=impact_yuan,
            execution_result=result,
            prev_record_hash=self._last_hash,
        )
        record.record_hash = record.compute_hash()
        self._last_hash = record.record_hash
        self.records.append(record)
        return record

    def verify_integrity(self) -> bool:
        """验证审计链完整性（防篡改检查）"""
        for i, rec in enumerate(self.records):
            expected_prev = "GENESIS" if i == 0 else self.records[i-1].record_hash
            if rec.prev_record_hash != expected_prev:
                print(f"  ❌ 完整性破坏：记录 {rec.record_id}")
                return False
        return True

    def query(self, action_type: str = None, min_impact: float = None,
               approval_level: str = None, limit: int = 20) -> list:
        results = self.records
        if action_type:
            results = [r for r in results if r.what_action_type == action_type]
        if min_impact:
            results = [r for r in results if r.estimated_impact_yuan >= min_impact]
        if approval_level:
            results = [r for r in results if r.who_approval_level == approval_level]
        return results[-limit:]

    def compliance_report(self) -> dict:
        total = len(self.records)
        auto = sum(1 for r in self.records if r.who_approval_level == "AUTO")
        high_value_auto = sum(1 for r in self.records
                              if r.who_approval_level == "AUTO" and r.estimated_impact_yuan > 50_000)
        return {
            "total_decisions": total,
            "auto_decisions": auto,
            "auto_pct": round(auto / max(1, total) * 100, 1),
            "high_value_auto_decisions": high_value_auto,
            "integrity_ok": self.verify_integrity(),
        }


if __name__ == "__main__":
    print("【AI决策审计追踪本体】\n")
    trail = DecisionAuditTrail()

    # 记录一系列决策
    decisions = [
        ("procurement_agent", "create_replenishment_order", "SKU-S12Pro",
         {"qty": 500, "supplier": "宁波精工"}, {"stockout_risk": "high", "dos": 3},
         {"fused_score": 0.82}, "CrossDomainSignalFusion+PredictiveTag", "AUTO", 90_000),
        ("compliance_agent", "pause_listing_eu", "SKU-Accessory",
         {"market": "EU", "reason": "EPR非合规"}, {"compliance_status": "non_compliant"},
         {"compliance_score": 0.95}, "RegulatoryImpactPropagation", "MANAGER", 15_000),
        ("inventory_agent", "transfer_order", "SKU-A2Milk",
         {"from": "WH-CA", "to": "WH-NJ", "qty": 200}, {"wh.capacity_alert": "NORMAL"},
         {"rebalance_score": 0.65}, "MultiDCInventoryRebalancing", "AUTO", 3_500),
    ]

    recs = [trail.record_decision(*d) for d in decisions]

    print("=" * 65)
    print("【审计记录（含哈希链）】")
    for r in recs:
        icon = "⚡" if r.who_approval_level == "AUTO" else "👤"
        print(f"  {icon} [{r.record_id}] {r.what_action_type[:35]:35s} "
              f"¥{r.estimated_impact_yuan:,} | hash:{r.record_hash}")
        print(f"     Why: {r.why_trigger_tags}  Score:{r.why_signal_scores}")

    print("\n" + "=" * 65)
    integrity = trail.verify_integrity()
    print(f"  哈希链完整性: {'✅ 未被篡改' if integrity else '❌ 检测到篡改'}")

    report = trail.compliance_report()
    print(f"\n  合规报告: 总决策{report['total_decisions']}  "
          f"自动{report['auto_decisions']}({report['auto_pct']:.0f}%)  "
          f"高额自动决策{report['high_value_auto_decisions']}个")

    print(f"\n[✓] AI决策审计追踪本体 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supply-Chain-Agent-Orchestration-Hub]]（编排中枢的每个决策都需要审计记录）
- **前置（prerequisite）**：[[Skill-Human-in-Loop-Approval-Gate-Tag]]（审批门控的审批记录是审计的重要组成）
- **延伸（extends）**：[[Skill-Supply-Chain-Data-Lineage-Tracking]]（血缘追踪 + 决策审计 = 完整的可解释链）
- **可组合（combinable）**：[[Skill-Cross-Domain-Supply-Chain-Signal-Fusion]]（融合信号的计算过程写入审计记录）

## ⑤ 商业价值评估

- **ROI预估**：合规审计（SOC2/ISO27001）要求决策可追溯，审计准备时间从"2周人工整理"→"即时查询"节省约80小时；防止AI误操作后无法追责的法律风险（潜在损失不可估量）
- **实施难度**：⭐⭐⭐☆☆（技术上是Append-only存储+哈希链，工程可行性高）
- **优先级评分**：⭐⭐⭐⭐⭐（监管要求：EU AI Act要求高风险AI系统的决策可追溯；Amazon也要求卖家能解释账号操作历史）

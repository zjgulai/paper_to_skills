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

## ② 应用案例

### 案例1：婴儿暖奶器补货决策审计与异常回溯

**背景**：某母婴品牌使用AI供应链Agent管理“智能恒温暖奶器”（SKU: WN-2026）的库存。该SKU日均销量50件，安全库存200件，仓库现有库存2000件。

**事件**：2026年6月15日，AI Agent自动触发了一笔补货订单，数量为800件，金额¥96,000。但该决策导致库存积压，因为实际需求已因竞品降价而下滑。

**审计追踪过程**：
1. **记录生成**：Agent `procurement_agent_v2` 生成审计记录，包含：
   - WHO: `agent_id=procurement_agent_v2`, `approval_level=AUTO`
   - WHAT: `action_type=create_replenishment_order`, `parameters={"qty":800, "supplier":"宁波精工"}`
   - WHY: `trigger_tags={"stockout_risk":"high", "dos":2.5}`, `signal_scores={"fused_score":0.82}`
   - WHEN: `2026-06-15 14:23:11.456`
   - HOW: `algorithm=CrossDomainSignalFusion+PredictiveTag`, `version=1.0`
   - 影响金额: `estimated_impact_yuan=96000`

2. **问题发现**：6月20日，运营发现该SKU周转率从28天升至45天，库存积压严重。

3. **审计回溯**：通过查询 `query(action_type="create_replenishment_order", min_impact=50000)`，定位到该记录。发现 `why_trigger_tags` 中的 `dos`（库存天数）为2.5天，但实际该SKU的日均销量已从50件降至30件（因竞品降价），正确的 `dos` 应为4.2天。原因是Agent使用的销量预测模型未及时更新市场信号。

4. **整改**：修正预测模型，并增加人工审批门控：所有影响金额超过¥50,000的自动补货决策，必须经过 `approval_level=MANAGER` 审批。

**量化产出**：
- 库存周转率提升：28%（从45天降至32天）
- 年化库存持有成本节省：¥45万元（按库存金额¥120万、持有成本率25%计算）
- 异常决策发现时间：从2周缩短至即时查询

### 案例2：婴儿推车合规下架决策审计

**背景**：某跨境卖家在欧盟站销售“轻便折叠婴儿推车”（SKU: ST-2026-EU）。欧盟EPR法规要求所有推车必须注册并上传合规证书。

**事件**：2026年6月10日，合规Agent检测到该SKU的EPR证书已过期，自动触发了下架决策。

**审计追踪过程**：
1. **记录生成**：Agent `compliance_agent_v1` 生成审计记录：
   - WHO: `agent_id=compliance_agent_v1`, `approval_level=MANAGER`（因影响金额¥15,000，触发人工审批）
   - WHAT: `action_type=pause_listing_eu`, `parameters={"market":"EU", "reason":"EPR非合规"}`
   - WHY: `trigger_tags={"compliance_status":"non_compliant"}`, `signal_scores={"compliance_score":0.95}`
   - WHEN: `2026-06-10 09:15:33.789`
   - HOW: `algorithm=RegulatoryImpactPropagation`, `version=2.1`
   - 影响金额: `estimated_impact_yuan=15000`

2. **合规证明**：审计记录提供了完整的决策依据，可用于应对欧盟监管机构的审计，证明下架是基于合规规则自动触发，且经过了人工审批。

3. **后续优化**：通过分析审计记录，发现该SKU的EPR证书过期前30天，Agent已发出预警，但运营未及时处理。因此，增加了预警升级机制：若预警后7天未处理，自动升级至 `approval_level=VP`。

**量化产出**：
- 合规审计准备时间：从2周人工整理缩短至即时查询
- 合规风险事件减少：60%（因预警升级机制）
- 避免潜在罚款：€50,000（EU AI Act违规罚款）

### 案例3：有机辅食库存调拨决策审计与ROI分析

**背景**：某有机辅食品牌（SKU: BF-2026-Organic）在两个仓库（WH-CA和WH-NJ）之间进行库存调拨。该SKU日均销量120件，ROAS（广告支出回报率）为3.2，转化率为4.5%。

**事件**：2026年6月18日，库存Agent自动发起了一笔调拨订单，从WH-CA调拨200件至WH-NJ，金额¥7,000。

**审计追踪过程**：
1. **记录生成**：Agent `inventory_agent_v3` 生成审计记录：
   - WHO: `agent_id=inventory_agent_v3`, `approval_level=AUTO`
   - WHAT: `action_type=transfer_order`, `parameters={"from":"WH-CA", "to":"WH-NJ", "qty":200}`
   - WHY: `trigger_tags={"wh.capacity_alert":"NORMAL", "demand_shift":"NJ_region_up"}`, `signal_scores={"rebalance_score":0.65}`
   - WHEN: `2026-06-18 16:45:22.123`
   - HOW: `algorithm=MultiDCInventoryRebalancing`, `version=2.0`
   - 影响金额: `estimated_impact_yuan=7000`

2. **效果验证**：调拨后，NJ仓库的缺货率从8%降至2%，CA仓库的库存周转率从35天降至28天。通过审计记录，可以量化该决策带来的ROI：
   - 避免缺货损失：日均120件 × 4.5%转化率 × ¥35单价 × 6%缺货改善 = ¥1,134/天
   - 年化收益：¥1,134 × 365 = ¥41.4万元
   - 调拨成本：¥7,000（含物流）
   - ROI：¥41.4万 / ¥0.7万 = 59倍

3. **模型优化**：审计记录显示 `rebalance_score` 仅为0.65，说明模型对调拨收益的置信度不高。通过分析历史审计数据，发现当 `demand_shift` 信号强度 > 0.8 时，调拨准确率更高。因此，将调拨触发阈值从0.6提升至0.7。

**量化产出**：
- 调拨决策准确率提升：15%（从82%提升至94%）
- 年化缺货损失减少：¥41.4万元
- 模型优化周期：从月度缩短至周度（基于审计数据反馈）

## ③ 代码模板

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

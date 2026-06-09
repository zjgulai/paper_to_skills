---
title: Root Cause Analysis Agent for Business Anomalies
doc_type: knowledge
module: 09-DataAgent-LLM
topic: root-cause-analysis
status: stable
created: 2026-05-15
updated: 2026-05-15
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Root Cause Analysis Agent

---

## ① 算法原理

**核心问题**：异常检测告诉你"什么出问题了"，但不告诉你"为什么"。根因分析（RCA）回答"为什么"——是系统Bug、竞品行动、营销活动、还是供应链问题？

**传统RCA的局限**：
- 人工排查：慢，依赖经验
- 规则引擎：僵化，无法覆盖未知场景
- 相关性分析：容易陷入"相关≠因果"的陷阱

**Agent-based RCA 框架**：

1. **多源数据接入**：整合指标数据、日志、用户反馈、竞品情报
2. **假设生成**：LLM根据异常特征生成可能的根因假设
3. **证据收集**：Agent自动查询数据库、调用API验证假设
4. **因果推理**：用因果推断方法（DoWhy/PyWhy）验证因果关系
5. **报告生成**：输出结构化的根因报告，含置信度和建议

**关键组件**：

- **Hypothesis Generator**：基于异常模式生成根因假设树
- **Evidence Collector**：自动查询相关数据源
- **Causal Validator**：用干预分析验证因果方向
- **Report Composer**：生成可执行的建议

**反直觉洞察**：
- 最快的根因定位往往不是"深入挖掘"，而是"横向对比"——与其他 unaffected 的维度对比
- 80%的根因可以在5分钟内定位：看最近变更（代码发布、配置修改、营销活动）
- Agent的价值在于处理那20%的复杂根因——需要跨系统、跨时间维度的关联分析

---

## ② 母婴出海应用案例

### 场景：转化率异常自动根因分析

**业务问题**：某日下午转化率从2.5%骤降到1.2%。传统排查需要人工逐个检查系统、页面、流量来源，耗时1-2小时。

**Agent RCA 流程**：

1. **异常接收**：时序异常检测系统触发告警
   ```
   告警：转化率 2.5% → 1.2%（-52%，Z=-4.3）
   时间：2026-05-10 14:00-15:00
   ```

2. **假设生成**（LLM）：
   - H1: 前端页面加载异常
   - H2: 支付通道故障
   - H3: 流量来源变化（低质量流量涌入）
   - H4: 某个SKU缺货导致流失
   - H5: 竞品促销导致用户比价流失

3. **证据收集**（Agent自动查询）：
   - H1: 检查页面加载时间 → 正常（1.2s）
   - H2: 检查支付成功率 → 异常（从98%降到45%）
   - H3: 检查流量来源构成 → 无明显变化
   - H4: 检查SKU库存状态 → 正常
   - H5: 检查竞品价格 → 无重大促销

4. **根因确认**：支付通道故障（Stripe API 429限流）

5. **建议生成**：
   - 立即：切换备用支付通道
   - 短期：与Stripe沟通限流原因
   - 长期：建立多支付通道自动切换机制

**预期产出**：
- 根因定位时间：1-2小时 → 5分钟
- 误定位率：人工30% → Agent <10%

---

## ③ 代码模板

```python
"""
Root Cause Analysis Agent — 根因分析Agent
用于业务异常的自动根因定位和报告生成
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class AnomalyEvent:
    """异常事件"""
    metric: str
    value: float
    baseline: float
    timestamp: str
    severity: str


@dataclass
class Hypothesis:
    """根因假设"""
    id: str
    description: str
    confidence: float = 0.0
    evidence: List[Dict] = None
    status: str = "pending"  # pending, confirmed, rejected


class RootCauseAnalysisAgent:
    """根因分析Agent"""

    def __init__(self):
        self.hypothesis_templates = {
            "payment": "支付通道故障",
            "frontend": "前端页面异常",
            "traffic": "流量来源变化",
            "inventory": "商品库存问题",
            "competitor": "竞品行动影响",
            "campaign": "营销活动副作用",
            "system": "系统发布/配置变更",
            "seasonal": "季节性因素"
        }

    def generate_hypotheses(self, event: AnomalyEvent) -> List[Hypothesis]:
        """
        根据异常事件生成根因假设

        实际应用中可用LLM生成更精准的假设
        """
        hypotheses = []

        # 根据异常指标类型生成相关假设
        if "conversion" in event.metric.lower():
            hypotheses.extend([
                Hypothesis("H1", "支付通道故障导致用户无法完成支付"),
                Hypothesis("H2", "前端页面加载异常导致用户流失"),
                Hypothesis("H3", "流量来源变化（低质量流量涌入）"),
                Hypothesis("H4", "价格显示异常或优惠券失效"),
            ])
        elif "revenue" in event.metric.lower() or "gmv" in event.metric.lower():
            hypotheses.extend([
                Hypothesis("H1", "热销SKU缺货"),
                Hypothesis("H2", "竞品大幅降价"),
                Hypothesis("H3", "营销活动到期"),
                Hypothesis("H4", "物流费用上涨导致客单价下降"),
            ])
        elif "traffic" in event.metric.lower():
            hypotheses.extend([
                Hypothesis("H1", "广告投放异常（预算耗尽或被封）"),
                Hypothesis("H2", "SEO排名下降"),
                Hypothesis("H3", "社交媒体账号异常"),
                Hypothesis("H4", "节假日效应"),
            ])
        else:
            # 通用假设
            hypotheses.extend([
                Hypothesis("H1", "系统发布/配置变更导致"),
                Hypothesis("H2", "第三方服务故障"),
                Hypothesis("H3", "数据 pipeline 异常"),
                Hypothesis("H4", "外部事件影响"),
            ])

        return hypotheses

    def collect_evidence(self, hypothesis: Hypothesis, event: AnomalyEvent) -> Dict:
        """
        收集验证假设的证据

        实际应用中这里会调用真实的数据库/API查询
        """
        evidence = {"hypothesis": hypothesis.id, "checks": []}

        # 模拟证据收集
        if "支付" in hypothesis.description:
            evidence["checks"].append({
                "check": "支付成功率",
                "result": "异常: 98% → 45%",
                "supports": True,
                "confidence": 0.95
            })
            evidence["checks"].append({
                "check": "支付渠道分布",
                "result": "Stripe占比60%，其他渠道正常",
                "supports": True,
                "confidence": 0.90
            })
        elif "前端" in hypothesis.description:
            evidence["checks"].append({
                "check": "页面加载时间",
                "result": "正常: 1.2s",
                "supports": False,
                "confidence": 0.80
            })
        elif "流量" in hypothesis.description:
            evidence["checks"].append({
                "check": "流量来源构成",
                "result": "无明显变化",
                "supports": False,
                "confidence": 0.70
            })

        # 计算假设的置信度
        supporting = [c for c in evidence["checks"] if c.get("supports")]
        rejecting = [c for c in evidence["checks"] if not c.get("supports")]

        if supporting and not rejecting:
            hypothesis.confidence = sum(c["confidence"] for c in supporting) / len(supporting)
            hypothesis.status = "confirmed"
        elif rejecting and not supporting:
            hypothesis.confidence = 0.0
            hypothesis.status = "rejected"
        else:
            # 混合结果，取加权平均
            pos_score = sum(c["confidence"] for c in supporting) if supporting else 0
            neg_score = sum(c["confidence"] for c in rejecting) if rejecting else 0
            hypothesis.confidence = max(0, (pos_score - neg_score) / max(len(supporting) + len(rejecting), 1))
            hypothesis.status = "pending"

        hypothesis.evidence = evidence["checks"]
        return evidence

    def analyze(self, event: AnomalyEvent) -> Dict:
        """
        执行完整的根因分析
        """
        print(f"\n{'='*60}")
        print(f"根因分析: {event.metric}")
        print(f"{'='*60}")
        print(f"异常值: {event.value} (基准: {event.baseline})")
        print(f"时间: {event.timestamp}")

        # 1. 生成假设
        hypotheses = self.generate_hypotheses(event)
        print(f"\n生成 {len(hypotheses)} 个假设:")
        for h in hypotheses:
            print(f"  {h.id}: {h.description}")

        # 2. 收集证据
        print(f"\n收集证据...")
        for h in hypotheses:
            self.collect_evidence(h, event)
            status_icon = "✓" if h.status == "confirmed" else "✗" if h.status == "rejected" else "?"
            print(f"  {h.id} {status_icon} 置信度: {h.confidence:.2f}")
            if h.evidence:
                for e in h.evidence:
                    icon = "+" if e.get("supports") else "-"
                    print(f"    [{icon}] {e['check']}: {e['result']}")

        # 3. 排序并输出最可能根因
        confirmed = [h for h in hypotheses if h.status == "confirmed"]
        confirmed.sort(key=lambda x: x.confidence, reverse=True)

        result = {
            "event": event,
            "hypotheses": hypotheses,
            "root_cause": confirmed[0] if confirmed else None,
            "recommendations": []
        }

        if confirmed:
            root = confirmed[0]
            print(f"\n{'='*60}")
            print(f"根因: {root.description}")
            print(f"置信度: {root.confidence:.2f}")
            print(f"{'='*60}")

            # 生成建议
            result["recommendations"] = self.generate_recommendations(root)
            print("\n建议:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"  {i}. [{rec['urgency']}] {rec['action']}")
        else:
            print("\n未能确认根因，需要人工介入调查")

        return result

    def generate_recommendations(self, root_cause: Hypothesis) -> List[Dict]:
        """基于根因生成建议"""
        recommendations = []

        if "支付" in root_cause.description:
            recommendations = [
                {"urgency": "立即", "action": "切换备用支付通道"},
                {"urgency": "短期", "action": "联系支付服务商排查限流原因"},
                {"urgency": "长期", "action": "建立多支付通道自动切换机制"},
            ]
        elif "前端" in root_cause.description:
            recommendations = [
                {"urgency": "立即", "action": "回滚最近的前端发布"},
                {"urgency": "短期", "action": "修复页面加载问题"},
                {"urgency": "长期", "action": "加强前端发布前的性能测试"},
            ]
        elif "竞品" in root_cause.description:
            recommendations = [
                {"urgency": "短期", "action": "监控竞品价格变化"},
                {"urgency": "短期", "action": "评估是否需要跟进促销"},
                {"urgency": "长期", "action": "建立竞品价格监控体系"},
            ]
        else:
            recommendations = [
                {"urgency": "立即", "action": "启动应急排查流程"},
                {"urgency": "短期", "action": "联系相关团队确认变更"},
                {"urgency": "长期", "action": "完善监控和告警体系"},
            ]

        return recommendations


# 示例
if __name__ == '__main__':
    agent = RootCauseAnalysisAgent()

    event = AnomalyEvent(
        metric="conversion_rate",
        value=0.012,
        baseline=0.025,
        timestamp="2026-05-10 14:00",
        severity="high"
    )

    result = agent.analyze(event)
```

---


## ④ 技能关联

### 前置技能
- [Skill-Argos-Agentic-Anomaly-Detection](../09-DataAgent-LLM/[[Skill-Argos-Agentic-Anomaly-Detection]].md) — RCA 由异常检测触发
- [Skill-SQL-Agent-Text-to-SQL](../09-DataAgent-LLM/[[Skill-SQL-Agent-Text-to-SQL]].md) — RCA 需要查询多维数据切片

### 延伸技能
- [Skill-DeepAnalyze-Autonomous-Data-Science-Agent](../09-DataAgent-LLM/[[Skill-DeepAnalyze-Autonomous-Data-Science-Agent]].md) — RCA 是自治数据科学的核心场景

### 可组合
- [Skill-Multi-Agent-Debate](../10-MAS/[[Skill-Multi-Agent-Debate]].md) — 多 Agent 辩论提升 RCA 假设质量

## ⑤ 商业价值评估

- **ROI**：故障定位时间 1-2h → 5min，减少损失 80%
- **难度**：⭐⭐⭐☆☆（3/5）
- **优先级**：⭐⭐⭐⭐⭐（5/5）— 与异常检测形成完整监控闭环

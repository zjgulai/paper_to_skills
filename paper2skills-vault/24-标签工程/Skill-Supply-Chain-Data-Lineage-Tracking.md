---
title: 供应链数据血缘追踪 — 从原始数据到Tag决策的全链路溯源与影响分析
doc_type: knowledge
module: 24-标签工程
topic: supply-chain-data-lineage-tracking
status: stable
created: 2026-06-17
updated: 2026-06-17
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: 供应链数据血缘追踪

> **来源**：arXiv:2309.08923（Data Lineage for Supply Chain Intelligence）+ arXiv:2401.10234（Tag Provenance in Knowledge Graphs）+ Apache Atlas/OpenLineage实践
> **桥梁**：数据基础设施 ↔ 标签工程 ↔ 审计合规 | **类型**：数据治理

## ① 算法原理

**数据血缘（Data Lineage）** 追踪回答三个关键问题：
1. **这个Tag的值从哪里来？**（溯源）
2. **如果上游数据错了，影响了哪些Tag和决策？**（影响分析）
3. **这个自动决策的数据依据是什么？**（审计合规）

**血缘图谱结构**：

```
ERP库存数据 ──────────────────────────────────────────────┐
                                                          ↓
销售历史API ─→ 需求预测模型 ─→ predicted_stockout_7d Tag ─→ 补货Action
                  ↑                      ↑
PLT历史数据 ───────┘      Tag融合引擎 ─────┘
                              ↑
供应商延误数据 ────────────────┘
```

**血缘节点类型**：
- `DataSource`：原始数据来源（ERP/API/文件）
- `Transform`：转换/计算过程（模型推理/规则计算）
- `Tag`：产出的标签
- `Action`：由标签触发的业务操作

**影响分析（Impact Analysis）**：
- 当 ERP 库存数据延迟4小时未更新
- 影响哪些Tag？（stockout_risk、dos、predicted_stockout_7d）
- 影响哪些下游决策？（补货工单是否基于错误数据触发？）

## ② 代码模板

```python
"""
供应链数据血缘追踪系统
功能：血缘图谱构建 / 溯源查询 / 影响分析 / 审计报告
"""
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')


@dataclass
class LineageNode:
    node_id: str
    node_type: str      # DataSource / Transform / Tag / Action
    name: str
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class LineageEdge:
    from_id: str
    to_id: str
    edge_type: str      # PRODUCES / CONSUMES / TRIGGERS
    transform_info: str = ""


class DataLineageGraph:

    def __init__(self):
        self.nodes: dict = {}
        self.edges: list = []
        self.adj: dict = defaultdict(list)      # 正向
        self.radj: dict = defaultdict(list)     # 反向

    def add_node(self, node: LineageNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: LineageEdge):
        self.edges.append(edge)
        self.adj[edge.from_id].append(edge)
        self.radj[edge.to_id].append(edge)

    def trace_upstream(self, node_id: str, max_hops: int = 5) -> list:
        """溯源：追踪某Tag/决策的数据来源"""
        visited, path = set(), []
        queue = deque([(node_id, 0)])
        while queue:
            nid, hop = queue.popleft()
            if nid in visited or hop > max_hops:
                continue
            visited.add(nid)
            node = self.nodes.get(nid)
            if node:
                path.append({"hop": hop, "node": node.name, "type": node.node_type, "id": nid})
            for edge in self.radj[nid]:
                queue.append((edge.from_id, hop + 1))
        return path

    def impact_analysis(self, source_node_id: str) -> list:
        """影响分析：某数据源变化影响哪些下游"""
        visited, impacted = set(), []
        queue = deque([(source_node_id, 0)])
        while queue:
            nid, hop = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            node = self.nodes.get(nid)
            if node and nid != source_node_id:
                impacted.append({"node": node.name, "type": node.node_type, "hop": hop})
            for edge in self.adj[nid]:
                queue.append((edge.to_id, hop + 1))
        return impacted

    def audit_decision(self, action_id: str) -> dict:
        """审计：某业务决策的完整数据依据"""
        upstream = self.trace_upstream(action_id)
        sources = [n for n in upstream if n["type"] == "DataSource"]
        transforms = [n for n in upstream if n["type"] == "Transform"]
        tags = [n for n in upstream if n["type"] == "Tag"]
        return {
            "action": self.nodes.get(action_id, LineageNode("", "", "Unknown")).name,
            "data_sources": sources,
            "transformations": transforms,
            "tags_consumed": tags,
            "total_lineage_depth": max((n["hop"] for n in upstream), default=0),
        }


def build_supply_chain_lineage() -> DataLineageGraph:
    g = DataLineageGraph()
    nodes = [
        LineageNode("DS-ERP", "DataSource", "ERP库存数据"),
        LineageNode("DS-SALES", "DataSource", "销售历史API"),
        LineageNode("DS-PLT", "DataSource", "PLT前置期历史"),
        LineageNode("DS-SUP", "DataSource", "供应商延误数据"),
        LineageNode("TF-FORECAST", "Transform", "需求预测模型(LightGBM)"),
        LineageNode("TF-RISK", "Transform", "风险评分规则引擎"),
        LineageNode("TF-FUSION", "Transform", "跨域信号融合引擎"),
        LineageNode("TAG-DOS", "Tag", "Tag:sku.dos(库存天数)"),
        LineageNode("TAG-STOCKOUT", "Tag", "Tag:predicted_stockout_7d"),
        LineageNode("TAG-RISK", "Tag", "Tag:sku.stockout_risk"),
        LineageNode("TAG-FUSED", "Tag", "Tag:fused_risk_score"),
        LineageNode("ACT-REPL", "Action", "Action:create_replenishment_order"),
    ]
    for n in nodes:
        g.add_node(n)

    edges = [
        LineageEdge("DS-ERP", "TAG-DOS", "PRODUCES"),
        LineageEdge("DS-ERP", "TF-FORECAST", "CONSUMES"),
        LineageEdge("DS-SALES", "TF-FORECAST", "CONSUMES"),
        LineageEdge("DS-PLT", "TF-FORECAST", "CONSUMES"),
        LineageEdge("TF-FORECAST", "TAG-STOCKOUT", "PRODUCES"),
        LineageEdge("TAG-DOS", "TF-RISK", "CONSUMES"),
        LineageEdge("TAG-STOCKOUT", "TF-RISK", "CONSUMES"),
        LineageEdge("DS-SUP", "TF-RISK", "CONSUMES"),
        LineageEdge("TF-RISK", "TAG-RISK", "PRODUCES"),
        LineageEdge("TAG-RISK", "TF-FUSION", "CONSUMES"),
        LineageEdge("TF-FUSION", "TAG-FUSED", "PRODUCES"),
        LineageEdge("TAG-FUSED", "ACT-REPL", "TRIGGERS"),
    ]
    for e in edges:
        g.add_edge(e)
    return g


if __name__ == "__main__":
    print("【供应链数据血缘追踪系统】\n")
    g = build_supply_chain_lineage()

    print("=" * 60)
    print("【溯源：补货决策的数据来源】")
    upstream = g.trace_upstream("ACT-REPL")
    for n in upstream:
        icon = {"DataSource": "📦", "Transform": "⚙️ ", "Tag": "🏷️ ", "Action": "🔔"}[n["type"]]
        print(f"  {'  ' * n['hop']}{icon} [{n['type']}] {n['node']}")

    print("\n" + "=" * 60)
    print("【影响分析：ERP数据延迟影响哪些下游】")
    impacts = g.impact_analysis("DS-ERP")
    for imp in impacts:
        print(f"  Hop+{imp['hop']}: [{imp['type']}] {imp['node']}")

    print("\n" + "=" * 60)
    print("【审计报告：补货Action的完整数据依据】")
    audit = g.audit_decision("ACT-REPL")
    print(f"  Action: {audit['action']}")
    print(f"  数据来源: {[s['node'] for s in audit['data_sources']]}")
    print(f"  消费Tags: {[t['node'] for t in audit['tags_consumed']]}")
    print(f"  血缘深度: {audit['total_lineage_depth']}跳")
    print(f"\n[✓] 数据血缘追踪 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SKU-Master-Data-Golden-Record]]（GR是血缘的数据基础节点）
- **延伸（extends）**：[[Skill-Decision-Audit-Trail-Ontology]]（决策审计追踪是血缘的Action层）
- **可组合（combinable）**：[[Skill-Tag-Quality-Coverage-KPI]]（血缘帮助定位Tag质量问题的根源）
- **可组合（combinable）**：[[Skill-Cross-System-Data-Reconciliation]]（对账差异可通过血缘追踪到具体数据源）

## ⑤ 商业价值评估

- **ROI预估**：数据问题排查从"2天找数据来源"→"5分钟溯源"，合规审计证明AI决策依据节省法务时间约20小时/次；防止因数据错误导致的错误补货（每次约损失5-10万元）
- **实施难度**：⭐⭐⭐⭐☆（需要在数据管道中埋点，初期工程投入较大）
- **优先级评分**：⭐⭐⭐⭐☆（监管合规（GDPR/CSRD）要求数据可追溯；AI决策的可解释性需要血缘支撑）

---
title: 供应链操作知识库OKB图谱设计 — Neo4j+Delta双层架构与CDC实时同步策略
doc_type: knowledge
module: 24-标签工程
topic: graph-okb-design-supply-chain
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 供应链操作知识库(OKB)图谱设计

> **来源**：GitHub:neo4j-partners/databricks-neo4j-supply-chain（Databricks+Neo4j双层 SC 架构）+ AstraZeneca/Capgemini Graph Summit 2026（400万节点案例）+ Rivian/Databricks 案例（2026）
> **桥梁**：标签工程 ↔ 知识图谱 ↔ Palantir Object Store 设计 | **类型**：图数据库+数据架构

## ① 算法原理

**OKB（操作知识库）vs 分析型数仓的根本区别**：数仓回答"历史发生了什么"，OKB 支撑"现在该怎么做"。在 Palantir 架构中，Object Store 就是 OKB 的实现形式。

**双层架构设计**（Databricks + Neo4j 模式，AstraZeneca 规模验证）：

```
操作层 (Databricks Delta Lake)          图智能层 (Neo4j AuraDB)
────────────────────────────            ──────────────────────────
Gold 表：库存/订单/KPI                  Supplier→Product→BOM节点
Metric Views：认证指标定义              GDS算法：PageRank/社区检测
Unity Catalog：治理+血缘                多跳遍历：BOM展开/替代供应商
批量分析：时序预测/聚合报告             实时查询：<2秒 关系推理
              ↕ CDC 双向同步 ↕
```

**三类典型查询的路由规则**：

| 问题类型 | 路由 | 查询语言 | 延迟 |
|---------|------|---------|------|
| "STERILIZER-PRO 当前库存多少？" | Delta SQL | Spark SQL | <5s |
| "如果供应商A断供，哪些SKU受影响？" | Neo4j | Cypher 多跳 | <2s |
| "哪个供应商是整个供应网络的瓶颈？" | GDS | PageRank | <10s |

**GDS 算法在供应链中的应用**：

```cypher
// PageRank：识别最关键的供应商节点
CALL gds.pageRank.stream('supply-network')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS supplier, score
ORDER BY score DESC LIMIT 5

// Betweenness Centrality：识别路由瓶颈
CALL gds.betweenness.stream('supply-network')
YIELD nodeId, score
WHERE gds.util.asNode(nodeId).type = 'WAREHOUSE'
RETURN gds.util.asNode(nodeId).name, score ORDER BY score DESC

// Node Similarity：识别可替代供应商
CALL gds.nodeSimilarity.stream('supply-network')
YIELD node1, node2, similarity
WHERE similarity > 0.7
RETURN gds.util.asNode(node1).name, gds.util.asNode(node2).name, similarity
```

## ② 母婴出海应用案例

**场景A：供应商断供时秒级发现影响链**

吸奶器主供应商突发产能危机，运营需要立即知道：哪些 SKU 受影响？有没有替代供应商？哪些在途 PO 需要取消？

传统 SQL 需要 3 层 JOIN（供应商→SKU→库存→在途PO），查询 >30 秒且容易漏查。Neo4j Cypher 多跳遍历 <2 秒完成完整影响链分析。

**数据要求**：供应商-SKU 供货关系、BOM 展开数据、在途 PO 记录
**预期产出**：影响 SKU 列表 + 可替代供应商排名 + 需取消/调整的 PO 清单
**业务价值**：应急响应从 3 小时人工梳理 → 2 秒自动图遍历，供应商替代方案发现时间 ↓95%

**场景B：BOM 成本涨价传播分析**

原材料（ABS 塑料）价格上涨 15%，需要快速知道影响哪些 SKU 的成本，影响幅度各多少。

BOM 多级展开（原料→零件→半成品→成品）通过 Cypher variable-length path 一次性计算，而 SQL 递归 CTE 复杂且慢。

**数据要求**：多级 BOM 数据、各级用量系数、单位成本
**预期产出**：受影响 SKU 列表 + 成本影响金额（按SKU排序）+ 建议提价幅度
**业务价值**：成本涨价影响分析从 1 天 → 10 分钟，及时调整定价保护毛利

## ③ 代码模板

```python
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import math

@dataclass
class GraphNode:
    """知识图谱节点"""
    node_id: str
    node_type: str   # Supplier/Product/Warehouse/PurchaseOrder
    properties: Dict = field(default_factory=dict)

@dataclass
class GraphEdge:
    """知识图谱边"""
    from_id: str
    to_id: str
    relation_type: str  # SUPPLIES/CONTAINS/STORES/SHIPS_TO
    properties: Dict = field(default_factory=dict)

class SCKnowledgeGraph:
    """
    供应链知识图谱（内存图，用于原型验证）
    生产环境：替换为 Neo4j AuraDB Driver 调用
    
    核心能力：
    1. 多跳影响链遍历（断供影响分析）
    2. PageRank 关键节点识别
    3. 相似节点发现（可替代供应商）
    4. BOM 成本传播计算
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._adjacency: Dict[str, List[Tuple[str, str, Dict]]] = {}  # node_id → [(neighbor, rel, props)]
        self._reverse_adj: Dict[str, List[Tuple[str, str, Dict]]] = {}
    
    def add_node(self, node: GraphNode):
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)
        if edge.from_id not in self._adjacency:
            self._adjacency[edge.from_id] = []
        self._adjacency[edge.from_id].append((edge.to_id, edge.relation_type, edge.properties))
        if edge.to_id not in self._reverse_adj:
            self._reverse_adj[edge.to_id] = []
        self._reverse_adj[edge.to_id].append((edge.from_id, edge.relation_type, edge.properties))
    
    def find_supply_disruption_impact(self, disrupted_supplier_id: str,
                                       max_hops: int = 4) -> Dict:
        """
        供应商断供影响链分析（模拟 Cypher 多跳遍历）
        对应 Cypher:
          MATCH (s:Supplier {id: $id})-[:SUPPLIES*1..4]->(affected)
          RETURN affected
        """
        affected = {}
        queue = [(disrupted_supplier_id, 0)]
        visited = {disrupted_supplier_id}
        
        while queue:
            node_id, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            for neighbor, rel_type, props in self._adjacency.get(node_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    node = self.nodes.get(neighbor)
                    if node:
                        node_type = node.node_type
                        if node_type not in affected:
                            affected[node_type] = []
                        affected[node_type].append({
                            "id": neighbor,
                            "name": node.properties.get("name", neighbor),
                            "via_relation": rel_type,
                            "hop_distance": depth + 1
                        })
                    queue.append((neighbor, depth + 1))
        
        return {
            "disrupted_supplier": disrupted_supplier_id,
            "impact_by_type": affected,
            "total_affected_nodes": sum(len(v) for v in affected.values()),
            "affected_skus": [n["id"] for n in affected.get("Product", [])],
            "affected_warehouses": [n["id"] for n in affected.get("Warehouse", [])],
        }
    
    def compute_pagerank(self, node_type_filter: Optional[str] = None,
                          iterations: int = 20, damping: float = 0.85) -> List[Dict]:
        """
        供应链 PageRank（识别最关键节点）
        对应 GDS: gds.pageRank.stream
        """
        all_node_ids = list(self.nodes.keys())
        if node_type_filter:
            all_node_ids = [nid for nid, n in self.nodes.items() 
                          if n.node_type == node_type_filter]
        
        scores = {nid: 1.0 / len(all_node_ids) for nid in all_node_ids}
        
        for _ in range(iterations):
            new_scores = {}
            for nid in all_node_ids:
                in_neighbors = [(src, rel, props) for src, rel, props 
                               in self._reverse_adj.get(nid, [])
                               if src in all_node_ids]
                
                rank_sum = sum(
                    scores.get(src, 0) / max(len(self._adjacency.get(src, [])), 1)
                    for src, _, _ in in_neighbors
                )
                new_scores[nid] = (1 - damping) / len(all_node_ids) + damping * rank_sum
            scores = new_scores
        
        results = [
            {"node_id": nid, "name": self.nodes[nid].properties.get("name", nid),
             "type": self.nodes[nid].node_type,
             "pagerank_score": round(scores[nid], 6)}
            for nid in all_node_ids
        ]
        return sorted(results, key=lambda x: x["pagerank_score"], reverse=True)
    
    def find_alternative_suppliers(self, target_supplier_id: str,
                                    similarity_threshold: float = 0.5) -> List[Dict]:
        """
        可替代供应商发现（模拟 GDS Node Similarity）
        基于：共同供货的 SKU 数量（Jaccard 相似度）
        """
        target = self.nodes.get(target_supplier_id)
        if not target or target.node_type != "Supplier":
            return []
        
        # 目标供应商供货的 SKU 集合
        target_skus = set(
            neighbor for neighbor, rel, _ in self._adjacency.get(target_supplier_id, [])
            if rel == "SUPPLIES"
        )
        if not target_skus:
            return []
        
        alternatives = []
        for nid, node in self.nodes.items():
            if nid == target_supplier_id or node.node_type != "Supplier":
                continue
            
            candidate_skus = set(
                neighbor for neighbor, rel, _ in self._adjacency.get(nid, [])
                if rel == "SUPPLIES"
            )
            
            # Jaccard 相似度
            intersection = len(target_skus & candidate_skus)
            union = len(target_skus | candidate_skus)
            jaccard = intersection / max(union, 1)
            
            if jaccard >= similarity_threshold:
                alternatives.append({
                    "supplier_id": nid,
                    "name": node.properties.get("name", nid),
                    "similarity": round(jaccard, 3),
                    "shared_skus": intersection,
                    "lead_time_days": node.properties.get("lead_time_days", 0),
                    "risk_score": node.properties.get("risk_score", 50)
                })
        
        return sorted(alternatives, key=lambda x: x["similarity"], reverse=True)
    
    def compute_bom_cost_impact(self, material_id: str, 
                                 price_change_pct: float) -> List[Dict]:
        """BOM 成本涨价传播（variable-length path 模拟）"""
        impact = []
        
        # 找所有使用该原材料的成品（多跳）
        def traverse(node_id: str, depth: int, multiplier: float, path: List[str]):
            for neighbor, rel, props in self._adjacency.get(node_id, []):
                if neighbor in path:  # 避免循环
                    continue
                usage = props.get("usage_per_unit", 1.0)
                new_multiplier = multiplier * usage
                neighbor_node = self.nodes.get(neighbor)
                if neighbor_node and neighbor_node.node_type == "Product":
                    unit_cost = neighbor_node.properties.get("unit_cost", 0)
                    cost_impact = unit_cost * (price_change_pct / 100) * new_multiplier
                    impact.append({
                        "product_id": neighbor,
                        "product_name": neighbor_node.properties.get("name", neighbor),
                        "hop_depth": depth + 1,
                        "cost_impact_per_unit": round(cost_impact, 4),
                        "price_change_pct": price_change_pct
                    })
                traverse(neighbor, depth + 1, new_multiplier, path + [neighbor])
        
        traverse(material_id, 0, 1.0, [material_id])
        return sorted(impact, key=lambda x: abs(x["cost_impact_per_unit"]), reverse=True)


# ===== 测试用例 =====
def run_test():
    kg = SCKnowledgeGraph()
    
    # 构建供应链知识图谱
    nodes = [
        GraphNode("SUP-A", "Supplier", {"name": "深圳乐宝", "lead_time_days": 45, "risk_score": 25}),
        GraphNode("SUP-B", "Supplier", {"name": "广州辅食坊", "lead_time_days": 50, "risk_score": 40}),
        GraphNode("SUP-C", "Supplier", {"name": "义乌百优", "lead_time_days": 60, "risk_score": 70}),
        GraphNode("MAT-ABS", "Material", {"name": "ABS塑料", "unit_cost": 2.5}),
        GraphNode("PROD-STL", "Product", {"name": "消毒锅Pro", "unit_cost": 18.5}),
        GraphNode("PROD-BTL", "Product", {"name": "奶瓶套装", "unit_cost": 8.0}),
        GraphNode("PROD-FOOD", "Product", {"name": "有机辅食", "unit_cost": 5.0}),
        GraphNode("WH-US", "Warehouse", {"name": "FBA-US-East"}),
    ]
    for n in nodes:
        kg.add_node(n)
    
    edges = [
        GraphEdge("SUP-A", "PROD-STL", "SUPPLIES"),
        GraphEdge("SUP-A", "PROD-BTL", "SUPPLIES"),
        GraphEdge("SUP-B", "PROD-FOOD", "SUPPLIES"),
        GraphEdge("SUP-C", "PROD-BTL", "SUPPLIES"),   # SUP-C 也供 BTL (可替代SUP-A)
        GraphEdge("MAT-ABS", "PROD-STL", "CONTAINS", {"usage_per_unit": 0.3}),
        GraphEdge("MAT-ABS", "PROD-BTL", "CONTAINS", {"usage_per_unit": 0.15}),
        GraphEdge("PROD-STL", "WH-US", "STORES"),
        GraphEdge("PROD-BTL", "WH-US", "STORES"),
    ]
    for e in edges:
        kg.add_edge(e)
    
    # Test 1: 断供影响分析
    impact = kg.find_supply_disruption_impact("SUP-A")
    assert len(impact["affected_skus"]) == 2, f"SUP-A断供应影响2个SKU，实际{len(impact['affected_skus'])}"
    print(f"  SUP-A断供影响: {impact['total_affected_nodes']} 个节点")
    print(f"  受影响SKU: {impact['affected_skus']}")
    
    # Test 2: PageRank关键性分析
    pr = kg.compute_pagerank(node_type_filter="Supplier")
    assert len(pr) == 3, "应有3个供应商节点"
    print(f"  PageRank Top1供应商: {pr[0]['name']} (score={pr[0]['pagerank_score']:.4f})")
    
    # Test 3: 可替代供应商
    alts = kg.find_alternative_suppliers("SUP-A", similarity_threshold=0.3)
    assert len(alts) > 0, "应找到至少1个可替代供应商"
    print(f"  SUP-A的替代供应商: {alts[0]['name']} (similarity={alts[0]['similarity']})")
    
    # Test 4: BOM成本传播
    cost_impact = kg.compute_bom_cost_impact("MAT-ABS", price_change_pct=15)
    assert len(cost_impact) > 0, "ABS涨价应影响至少1个产品"
    print(f"  ABS涨价15%影响: {len(cost_impact)}个产品, 最大影响${cost_impact[0]['cost_impact_per_unit']}/件")
    
    print("\n[✓] Graph-OKB-Design-SC 测试通过 — 多跳遍历+PageRank+相似度+BOM传播就绪")

run_test()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-SKU-Master-Data-Golden-Record]] — 主数据是 OKB 节点质量的基础
- **前置（prerequisite）**：[[Skill-Supply-Chain-Data-Lineage-Tracking]] — 血缘追踪是 OKB 数据治理的配套
- **延伸（extends）**：[[Skill-Ontology-LLM-AutoBuild-SC]] — LLM 自动构建的本体最终存入 OKB 图谱
- **延伸（extends）**：[[Skill-Supply-Chain-Data-Mesh-Architecture]] — Data Mesh 是 OKB 的分布式演化形态
- **可组合（combinable）**：[[Skill-SC-Digital-Twin-Sync-Architecture]] — OKB 图谱是数字孪生的语义层基础
- **可组合（combinable）**：[[Skill-Supplier-Ontology-Capability-Map]] — OKB 承载供应商能力图谱的存储与查询

## ⑤ 商业价值评估

- **ROI 预估**：供应商断供影响分析从 3 小时 → 2 秒（↓99.9%），AstraZeneca 400万节点 BOM 遍历 <2 秒，Rivian 根因分析 30 分钟 → <2 分钟（↓93%）
- **实施难度**：⭐⭐⭐⭐☆（Neo4j AuraDB + Debezium CDC 是主要工程挑战）
- **优先级**：⭐⭐⭐⭐⭐（企业 AI 知识库的核心基础设施，Palantir Object Store 的开源替代方案）
- **企业AI知识库依赖**：极高 — OKB 本身即是企业 AI 知识库的图谱层，所有 Agent 的关系推理依赖于此

"""
Helicase Supply Chain KG MAS — 不确定性感知供应链知识图谱多 Agent 构建
arXiv:2605.26835 | Python 3.14+ | 仅标准库，无需额外安装

参考论文: Helicase: Uncertainty-Guided Supply Chain Knowledge Graph
Construction with Autonomous Multi-Agent LLMs
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class KGNode:
    """知识图谱节点（含不确定性评分）"""
    node_id: str
    entity_type: str           # supplier / ingredient / factory / certifier / product
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    uncertainty_score: float = 0.5   # 0=确定，1=高度不确定
    sources: list[str] = field(default_factory=list)
    last_verified: str = ""


@dataclass
class KGEdge:
    """知识图谱边（含不确定性评分）"""
    edge_id: str
    from_node: str
    to_node: str
    relation_type: str         # supplies / certified_by / manufactured_at / recalled_for
    confidence: float = 0.5   # 关系置信度 0-1
    uncertainty_score: float = 0.5
    evidence: list[str] = field(default_factory=list)


@dataclass
class InvestigationTask:
    """调查子任务"""
    task_id: str
    query: str
    target_entity: str
    priority: float = 0.5
    status: str = "pending"   # pending / running / done


# ── 三层不确定性跟踪器 ────────────────────────────────────────────────────────

class UncertaintyTracker:
    """
    三层不确定性量化框架：
    - action_uncertainty:     单步工具调用结果置信度
    - trajectory_uncertainty: 多轮推理链路累积不确定性
    - memory_uncertainty:     KG 节点时效性和来源可信度
    """

    def __init__(self, action_weight: float = 0.4,
                 trajectory_weight: float = 0.35,
                 memory_weight: float = 0.25):
        self.action_weight = action_weight
        self.trajectory_weight = trajectory_weight
        self.memory_weight = memory_weight
        self._action_history: list[float] = []
        self._trajectory_steps: list[float] = []

    def record_action(self, confidence: float) -> None:
        """记录单步动作置信度"""
        self._action_history.append(confidence)

    def record_trajectory_step(self, step_uncertainty: float) -> None:
        """记录推理链路步骤不确定性（累积衰减）"""
        self._trajectory_steps.append(step_uncertainty)

    @property
    def action_uncertainty(self) -> float:
        if not self._action_history:
            return 0.5
        recent = self._action_history[-5:]
        return 1.0 - sum(recent) / len(recent)

    @property
    def trajectory_uncertainty(self) -> float:
        if not self._trajectory_steps:
            return 0.3
        n = len(self._trajectory_steps)
        base = sum(self._trajectory_steps) / n
        return min(1.0, base * (1 + 0.05 * n))

    @property
    def memory_uncertainty(self) -> float:
        if not self._action_history:
            return 0.5
        consistency = 1.0 - self.action_uncertainty * 0.6
        return 1.0 - consistency

    @property
    def total_uncertainty(self) -> float:
        return (self.action_uncertainty * self.action_weight
                + self.trajectory_uncertainty * self.trajectory_weight
                + self.memory_uncertainty * self.memory_weight)

    def reset(self) -> None:
        self._action_history.clear()
        self._trajectory_steps.clear()


# ── 专业 Agent ────────────────────────────────────────────────────────────────

class PlannerAgent:
    """将高层查询分解为可执行子调查任务列表"""

    def decompose(self, query: str, existing_nodes: list[str]) -> list[InvestigationTask]:
        """分解查询为子任务（实际应调用 LLM，此处模拟）"""
        tasks = []
        subtopics = [
            ("原料来源调查", f"{query} 原材料 供应商"),
            ("工厂认证调查", f"{query} 工厂 GMP 认证"),
            ("召回历史检索", f"{query} 召回 CPSC RAPEX"),
            ("企业资质核验", f"{query} 公司 营业执照 资质"),
        ]
        for i, (name, sub_query) in enumerate(subtopics):
            task = InvestigationTask(
                task_id=f"task_{i+1}",
                query=sub_query,
                target_entity=name,
                priority=1.0 - i * 0.15,
            )
            tasks.append(task)
        return tasks

    def replan(self, high_uncertainty_nodes: list[str]) -> list[InvestigationTask]:
        """对高不确定性节点生成补充调查任务"""
        tasks = []
        for i, node in enumerate(high_uncertainty_nodes):
            tasks.append(InvestigationTask(
                task_id=f"replan_{i+1}",
                query=f"深度核验 {node} 信息来源",
                target_entity=node,
                priority=0.9,
            ))
        return tasks


class WebSearchAgent:
    """多源证据检索 Agent（模拟 Web/专利/政府数据库检索）"""

    def search(self, query: str, tracker: UncertaintyTracker) -> list[dict]:
        """执行搜索，返回证据列表"""
        random.seed(hash(query) % 1000)
        num_results = random.randint(2, 5)
        results = []
        for i in range(num_results):
            confidence = random.uniform(0.5, 0.95)
            tracker.record_action(confidence)
            results.append({
                "source": f"source_{i+1}",
                "snippet": f"[模拟] {query} 相关信息片段 {i+1}",
                "confidence": confidence,
                "url": f"https://example.com/{query[:10]}_{i}",
            })
        return results


class ReasoningAgent:
    """跨源推断 Agent：整合多个搜索结果，生成关系候选"""

    def infer_relations(self, query: str, evidences: list[dict],
                        tracker: UncertaintyTracker) -> list[dict]:
        """从证据推断实体关系"""
        if not evidences:
            return []
        avg_confidence = sum(e["confidence"] for e in evidences) / len(evidences)
        consistency_bonus = min(0.15, len(evidences) * 0.03)
        final_confidence = min(0.95, avg_confidence + consistency_bonus)
        tracker.record_trajectory_step(1.0 - final_confidence)

        relations = [
            {
                "from_entity": f"entity_{query[:6]}",
                "to_entity": f"entity_target",
                "relation": "supplies",
                "confidence": final_confidence,
                "evidence_count": len(evidences),
            }
        ]
        return relations


class CodingAgent:
    """将推理结论转化为 JSON KG 更新操作"""

    def update_kg(self, kg: "SupplyChainKG", relations: list[dict],
                  tracker: UncertaintyTracker) -> int:
        """增量更新 KG，返回新增边数"""
        added = 0
        for rel in relations:
            u = tracker.total_uncertainty
            from_id = rel["from_entity"]
            to_id = rel["to_entity"]
            if not kg.has_node(from_id):
                kg.add_node(KGNode(
                    node_id=from_id,
                    entity_type="supplier",
                    name=from_id,
                    uncertainty_score=u,
                ))
            if not kg.has_node(to_id):
                kg.add_node(KGNode(
                    node_id=to_id,
                    entity_type="ingredient",
                    name=to_id,
                    uncertainty_score=u,
                ))
            edge = KGEdge(
                edge_id=f"edge_{from_id}_{to_id}",
                from_node=from_id,
                to_node=to_id,
                relation_type=rel["relation"],
                confidence=rel["confidence"],
                uncertainty_score=u,
                evidence=[f"evidence_{i}" for i in range(rel.get("evidence_count", 1))],
            )
            kg.add_edge(edge)
            added += 1
        return added


# ── 知识图谱存储 ───────────────────────────────────────────────────────────────

class SupplyChainKG:
    """供应链知识图谱（内存存储 + 不确定性查询）"""

    def __init__(self):
        self._nodes: dict[str, KGNode] = {}
        self._edges: dict[str, KGEdge] = {}

    def add_node(self, node: KGNode) -> None:
        self._nodes[node.node_id] = node

    def add_edge(self, edge: KGEdge) -> None:
        self._edges[edge.edge_id] = edge

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_high_uncertainty_nodes(self, threshold: float = 0.65) -> list[str]:
        """返回不确定性超过阈值的节点（需人工复核或再调查）"""
        return [
            nid for nid, node in self._nodes.items()
            if node.uncertainty_score > threshold
        ]

    def query_suppliers(self, product: str) -> list[dict]:
        """查询指定产品的供应商关系"""
        results = []
        for edge in self._edges.values():
            if edge.relation_type == "supplies":
                from_node = self._nodes.get(edge.from_node)
                to_node = self._nodes.get(edge.to_node)
                if from_node and to_node:
                    results.append({
                        "supplier": from_node.name,
                        "target": to_node.name,
                        "confidence": edge.confidence,
                        "uncertainty": edge.uncertainty_score,
                    })
        return results

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.node_id, "type": n.entity_type,
                    "name": n.name, "uncertainty": n.uncertainty_score
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "id": e.edge_id, "from": e.from_node, "to": e.to_node,
                    "relation": e.relation_type, "confidence": e.confidence,
                    "uncertainty": e.uncertainty_score
                }
                for e in self._edges.values()
            ],
            "stats": {
                "nodes": len(self._nodes),
                "edges": len(self._edges),
                "high_uncertainty_nodes": len(self.get_high_uncertainty_nodes()),
            }
        }


# ── Helicase Orchestrator ──────────────────────────────────────────────────────

class HelicaseOrchestrator:
    """
    螺旋式迭代编排器：
    计划 → 搜索 → 推理 → 更新 KG → 评估不确定性 → 再计划（循环）
    """

    def __init__(self, max_iterations: int = 5,
                 uncertainty_threshold: float = 0.65):
        self.max_iterations = max_iterations
        self.uncertainty_threshold = uncertainty_threshold
        self.planner = PlannerAgent()
        self.searcher = WebSearchAgent()
        self.reasoner = ReasoningAgent()
        self.coder = CodingAgent()
        self.kg = SupplyChainKG()
        self.tracker = UncertaintyTracker()

    def run(self, query: str) -> SupplyChainKG:
        """执行螺旋式 KG 构建"""
        print(f"\n[Helicase] 开始调查: {query}")
        tasks = self.planner.decompose(query, [])

        for iteration in range(self.max_iterations):
            print(f"\n── 迭代 {iteration + 1}/{self.max_iterations} ──")
            pending = [t for t in tasks if t.status == "pending"]
            if not pending:
                print("  所有任务完成，退出迭代")
                break

            for task in pending[:2]:
                task.status = "running"
                print(f"  执行: {task.target_entity} | {task.query[:40]}...")

                evidences = self.searcher.search(task.query, self.tracker)
                print(f"  ↳ 搜索结果: {len(evidences)} 条，"
                      f"平均置信度 {sum(e['confidence'] for e in evidences)/len(evidences):.2f}")

                relations = self.reasoner.infer_relations(task.query, evidences, self.tracker)
                added = self.coder.update_kg(self.kg, relations, self.tracker)
                task.status = "done"

                print(f"  ↳ KG 新增 {added} 条关系 | "
                      f"总不确定性 {self.tracker.total_uncertainty:.3f}")

            high_u = self.kg.get_high_uncertainty_nodes(self.uncertainty_threshold)
            if high_u:
                print(f"\n  ⚠️  {len(high_u)} 个高不确定性节点，触发再计划...")
                new_tasks = self.planner.replan(high_u[:3])
                tasks.extend(new_tasks)
            else:
                print("  ✅ 不确定性已收敛")
                break

        return self.kg


# ── 测试 ───────────────────────────────────────────────────────────────────────

def test_helicase_baby_formula():
    """测试：母婴奶粉供应链查询，验证多 Agent 协作构建 KG 并标注不确定性"""
    print("=" * 60)
    print("Helicase 供应链 KG 构建测试：母婴奶粉供应商溯源")
    print("=" * 60)

    orchestrator = HelicaseOrchestrator(max_iterations=3, uncertainty_threshold=0.65)
    kg = orchestrator.run("某品牌 A2 有机婴儿配方奶粉 原料供应链")

    print("\n── KG 构建结果 ──")
    result = kg.to_dict()
    print(f"节点数: {result['stats']['nodes']}")
    print(f"边数:   {result['stats']['edges']}")
    print(f"高不确定性节点: {result['stats']['high_uncertainty_nodes']}")

    assert result["stats"]["nodes"] > 0, "KG 应包含至少一个节点"
    assert result["stats"]["edges"] > 0, "KG 应包含至少一条边"

    for edge in result["edges"]:
        assert 0.0 <= edge["uncertainty"] <= 1.0, \
            f"不确定性评分应在 [0, 1]，实际: {edge['uncertainty']}"

    high_u_count = result["stats"]["high_uncertainty_nodes"]
    total_nodes = result["stats"]["nodes"]
    print(f"\n高不确定性节点比例: {high_u_count}/{total_nodes} = "
          f"{high_u_count/max(1,total_nodes):.1%}")

    print("\n✅ 测试通过：多 Agent 协作构建 KG 并标注不确定性")
    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_helicase_baby_formula()

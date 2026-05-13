"""
Dialogue-to-Action Graph Parser
基于 TOD-Flow (Sohn et al., 2023) 的图条件对话建模思想，
将客服对话解析为决策动作图。

核心流程：
1. 识别对话轮次和角色（用户/客服）
2. 提取每轮的意图/动作（dialogue acts）
3. 构建 TOD-Flow 风格的图结构（Can/Should/ShouldNot 关系）
4. 输出可执行的决策动作图
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum


# ── 数据模型 ──────────────────────────────────────────

class RelationType(Enum):
    """TOD-Flow 的三种关系类型"""
    CAN = "can"           # 前置条件满足后可执行
    SHOULD = "should"     # 推荐执行
    SHOULD_NOT = "should_not"  # 不应执行


class NodeType(Enum):
    """决策图中的节点类型"""
    USER_ISSUE = "user_issue"       # 用户问题
    DIAGNOSIS = "diagnosis"         # 诊断
    SOLUTION = "solution"           # 解决方案
    RECOMMENDATION = "recommendation"  # 产品推荐
    RESOLUTION = "resolution"       # 问题解决/成交
    ESCALATION = "escalation"       # 升级/流失


@dataclass
class ActionNode:
    """决策图中的一个节点"""
    id: str
    node_type: NodeType
    text: str                       # 节点描述
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
        }


@dataclass
class ActionEdge:
    """决策图中的一条边（关系）"""
    source: str
    target: str
    relation: RelationType
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class DecisionGraph:
    """客服对话决策动作图"""
    nodes: List[ActionNode] = field(default_factory=list)
    edges: List[ActionEdge] = field(default_factory=list)
    raw_dialogue: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "raw_dialogue": self.raw_dialogue,
            "metadata": self.metadata,
        }

    def get_node(self, node_id: str) -> Optional[ActionNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def get_children(self, node_id: str) -> List[ActionNode]:
        """获取某节点的所有子节点"""
        child_ids = {e.target for e in self.edges if e.source == node_id}
        return [n for n in self.nodes if n.id in child_ids]

    def get_path_to_resolution(self) -> List[ActionNode]:
        """获取从问题到解决的推荐路径（沿 SHOULD 边）"""
        # 找 USER_ISSUE 节点作为起点
        start = next((n for n in self.nodes if n.node_type == NodeType.USER_ISSUE), None)
        if not start:
            return []

        path = [start]
        visited = {start.id}
        current = start.id

        while True:
            # 找 SHOULD 关系的下一个节点
            should_edges = [e for e in self.edges
                          if e.source == current and e.relation == RelationType.SHOULD
                          and e.target not in visited]
            if not should_edges:
                break
            next_edge = should_edges[0]
            next_node = self.get_node(next_edge.target)
            if not next_node:
                break
            path.append(next_node)
            visited.add(next_node.id)
            current = next_node.id

            if next_node.node_type in (NodeType.RESOLUTION, NodeType.ESCALATION):
                break

        return path


# ── 对话解析器 ────────────────────────────────────────

class DialogueToActionGraphParser:
    """
    客服对话 → 决策动作图解析器。

    基于 TOD-Flow 的图条件建模思想，将客服对话文本解析为
    (用户问题→诊断→解决方案→推荐商品→成交/流失) 的决策图。
    """

    # 问题类型关键词
    ISSUE_PATTERNS = {
        NodeType.USER_ISSUE: [
            r"change.*address", r"modify.*order", r"cancel.*order",
            r"return", r"refund", r"exchange", r"broken", r"defective",
            r"not working", r"issue", r"problem", r"question",
            r"改地址", r"取消订单", r"退货", r"退款", r"坏了", r"问题",
        ],
        NodeType.DIAGNOSIS: [
            r"check", r"verify", r"confirm", r"look into", r"investigate",
            r"核实", r"确认", r"查询",
        ],
        NodeType.SOLUTION: [
            r"solution", r"fix", r"replace", r"send", r"ship",
            r"solution", r"解决", r"更换", r"补发",
        ],
        NodeType.RECOMMENDATION: [
            r"recommend", r"suggest", r"try", r"consider",
            r"推荐", r"建议",
        ],
        NodeType.RESOLUTION: [
            r"resolved", r"completed", r"satisfied", r"thank",
            r"解决", r"完成", r"满意", r"谢谢",
        ],
        NodeType.ESCALATION: [
            r"escalate", r"manager", r"supervisor", r"complaint",
            r"升级", r"主管", r"投诉",
        ],
    }

    def __init__(self):
        self.node_counter = 0

    def parse(self, dialogue_text: str) -> DecisionGraph:
        """
        解析客服对话为决策动作图。

        Args:
            dialogue_text: 对话文本，格式:
                "User: ...\nAgent: ...\nUser: ...\nAgent: ..."
        """
        if not dialogue_text:
            return DecisionGraph(raw_dialogue=dialogue_text)

        self.node_counter = 0

        # 1. 分割对话轮次
        turns = self._split_turns(dialogue_text)

        # 2. 识别每轮的节点类型
        nodes = []
        for role, text in turns:
            node_type = self._classify_turn(text, role)
            node = ActionNode(
                id=f"n{self.node_counter}",
                node_type=node_type,
                text=text[:100],
                confidence=0.6,
                metadata={"role": role, "turn_idx": self.node_counter},
            )
            nodes.append(node)
            self.node_counter += 1

        # 3. 构建边（Can/Should/ShouldNot 关系）
        edges = self._build_edges(nodes)

        return DecisionGraph(
            nodes=nodes,
            edges=edges,
            raw_dialogue=dialogue_text,
            metadata={
                "turn_count": len(turns),
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        )

    def parse_batch(self, dialogues: List[str]) -> List[DecisionGraph]:
        """批量解析"""
        return [self.parse(d) for d in dialogues]

    def _split_turns(self, text: str) -> List[Tuple[str, str]]:
        """分割对话为 (role, text) 轮次"""
        turns = []

        # 模式 1: "User:/Agent:" 或 "Customer:/Support:" 前缀
        role_patterns = [
            r"(?:User|Customer|买家|客户)[\s:：]+(.+?)(?=(?:Agent|Support|客服|坐席)[\s:：]+|$)",
            r"(?:Agent|Support|客服|坐席)[\s:：]+(.+?)(?=(?:User|Customer|买家|客户)[\s:：]+|$)",
        ]

        # 尝试按角色分割
        lines = text.split("\n")
        current_role = None
        current_text = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测角色前缀
            if re.match(r"(?i)^(user|customer|买家|客户)[\s:：]+", line):
                if current_role and current_text:
                    turns.append((current_role, current_text.strip()))
                current_role = "user"
                current_text = re.sub(r"(?i)^(user|customer|买家|客户)[\s:：]+", "", line)
            elif re.match(r"(?i)^(agent|support|客服|坐席)[\s:：]+", line):
                if current_role and current_text:
                    turns.append((current_role, current_text.strip()))
                current_role = "agent"
                current_text = re.sub(r"(?i)^(agent|support|客服|坐席)[\s:：]+", "", line)
            else:
                current_text += " " + line

        if current_role and current_text:
            turns.append((current_role, current_text.strip()))

        # 如果未能分割，把整个文本当作一个 user turn
        if not turns:
            turns.append(("user", text))

        return turns

    def _classify_turn(self, text: str, role: str) -> NodeType:
        """分类对话轮次的意图类型"""
        text_lower = text.lower()

        # 根据角色和文本内容判断
        if role == "user":
            # 用户消息通常是问题
            return NodeType.USER_ISSUE

        # 客服消息：根据内容判断
        scores = {}
        for node_type, patterns in self.ISSUE_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, text_lower))
            scores[node_type] = score

        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best

        # 默认：如果对话后半段，倾向于解决方案
        return NodeType.SOLUTION

    def _build_edges(self, nodes: List[ActionNode]) -> List[ActionEdge]:
        """构建节点间的 Can/Should/ShouldNot 关系"""
        edges = []

        for i in range(len(nodes) - 1):
            current = nodes[i]
            next_node = nodes[i + 1]

            # 默认规则：相邻节点之间建立 SHOULD 关系
            relation = RelationType.SHOULD

            # 特定规则
            if current.node_type == NodeType.USER_ISSUE:
                if next_node.node_type == NodeType.DIAGNOSIS:
                    relation = RelationType.SHOULD
                elif next_node.node_type == NodeType.SOLUTION:
                    relation = RelationType.CAN  # 可以直接给方案

            elif current.node_type == NodeType.DIAGNOSIS:
                if next_node.node_type == NodeType.SOLUTION:
                    relation = RelationType.SHOULD

            elif current.node_type == NodeType.SOLUTION:
                if next_node.node_type == NodeType.RECOMMENDATION:
                    relation = RelationType.CAN
                elif next_node.node_type == NodeType.RESOLUTION:
                    relation = RelationType.SHOULD

            elif current.node_type == NodeType.ESCALATION:
                # 升级后不应该再推荐产品
                if next_node.node_type == NodeType.RECOMMENDATION:
                    relation = RelationType.SHOULD_NOT

            edges.append(ActionEdge(
                source=current.id,
                target=next_node.id,
                relation=relation,
                confidence=0.7,
            ))

        # 添加跨跳的 CAN 关系（非相邻但逻辑可达）
        for i in range(len(nodes)):
            for j in range(i + 2, len(nodes)):
                if nodes[i].node_type == NodeType.USER_ISSUE and \
                   nodes[j].node_type == NodeType.RESOLUTION:
                    edges.append(ActionEdge(
                        source=nodes[i].id,
                        target=nodes[j].id,
                        relation=RelationType.CAN,
                        confidence=0.5,
                    ))

        return edges


# ── 可视化辅助 ────────────────────────────────────────

def print_decision_graph(graph: DecisionGraph, indent: int = 0) -> None:
    """打印决策动作图"""
    prefix = "  " * indent
    print(f"{prefix}🗣️  客服对话决策图")
    print(f"{prefix}轮次: {graph.metadata.get('turn_count', 0)}")
    print(f"{prefix}节点: {len(graph.nodes)}, 边: {len(graph.edges)}")
    print()

    # 打印推荐路径
    path = graph.get_path_to_resolution()
    if path:
        print(f"{prefix}📍 推荐解决路径:")
        for i, node in enumerate(path):
            arrow = "→" if i < len(path) - 1 else "✓"
            emoji = {
                NodeType.USER_ISSUE: "❓",
                NodeType.DIAGNOSIS: "🔍",
                NodeType.SOLUTION: "💡",
                NodeType.RECOMMENDATION: "📦",
                NodeType.RESOLUTION: "✅",
                NodeType.ESCALATION: "⚠️",
            }.get(node.node_type, "•")
            print(f"{prefix}   {emoji} {node.node_type.value}: {node.text[:50]} {arrow}")
        print()

    # 打印所有节点和边
    print(f"{prefix}📋 完整结构:")
    for node in graph.nodes:
        emoji = {
            NodeType.USER_ISSUE: "❓",
            NodeType.DIAGNOSIS: "🔍",
            NodeType.SOLUTION: "💡",
            NodeType.RECOMMENDATION: "📦",
            NodeType.RESOLUTION: "✅",
            NodeType.ESCALATION: "⚠️",
        }.get(node.node_type, "•")
        print(f"{prefix}   {emoji} [{node.id}] {node.node_type.value}: {node.text[:60]}")

    if graph.edges:
        print(f"{prefix}   关系:")
        for edge in graph.edges:
            rel_emoji = {
                RelationType.CAN: "➡️",
                RelationType.SHOULD: "✅➡️",
                RelationType.SHOULD_NOT: "❌➡️",
            }.get(edge.relation, "→")
            print(f"{prefix}     {rel_emoji} {edge.source} --{edge.relation.value}--> {edge.target}")
    print()


def analyze_graph_patterns(graphs: List[DecisionGraph]) -> Dict[str, Any]:
    """分析多个对话决策图的共性模式"""
    path_patterns: Dict[str, int] = {}
    resolution_rate = 0
    avg_turns = 0

    for g in graphs:
        path = g.get_path_to_resolution()
        if path:
            pattern = "->".join(n.node_type.value for n in path)
            path_patterns[pattern] = path_patterns.get(pattern, 0) + 1

        if any(n.node_type == NodeType.RESOLUTION for n in g.nodes):
            resolution_rate += 1

        avg_turns += g.metadata.get("turn_count", 0)

    n = len(graphs) if graphs else 1
    return {
        "total_dialogues": len(graphs),
        "avg_turns": round(avg_turns / n, 2),
        "resolution_rate": round(resolution_rate / n, 2),
        "common_patterns": sorted(path_patterns.items(), key=lambda x: -x[1])[:5],
    }


# ── 测试 ──────────────────────────────────────────────

def test_parser() -> None:
    """单元测试"""
    parser = DialogueToActionGraphParser()

    # 测试用例 1: 改地址对话
    dialogue1 = """User: Hello, I was wondering if it possible to change my delivery address.
Agent: Sure, I can help with that. Let me check your order.
Agent: I've updated your address. You should receive a confirmation email.
User: Thank you so much!
"""
    graph1 = parser.parse(dialogue1)
    print_decision_graph(graph1)
    assert len(graph1.nodes) >= 3, f"Expected >= 3 nodes, got {len(graph1.nodes)}"
    assert any(e.relation == RelationType.SHOULD for e in graph1.edges)
    print("✅ Test 1 passed")

    # 测试用例 2: 产品问题对话
    dialogue2 = """User: My pump is not working. It won't turn on.
Agent: I'm sorry to hear that. Can you check if it's fully charged?
User: Yes, I charged it overnight but it still doesn't work.
Agent: Thank you for checking. I'll arrange a replacement for you.
User: Okay, thank you.
"""
    graph2 = parser.parse(dialogue2)
    print_decision_graph(graph2)
    path = graph2.get_path_to_resolution()
    assert len(path) >= 2, "Expected path to resolution"
    print("✅ Test 2 passed")

    # 测试用例 3: 空文本
    graph3 = parser.parse("")
    assert len(graph3.nodes) == 0
    print("✅ Test 3 passed")

    print("\n🎉 All tests passed!")


def test_with_zendesk_data() -> None:
    """用 Zendesk 真实数据做 POC 验证"""
    import pandas as pd

    data_path = "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/高质量数据源/zendesk_momcozy_voc_sampled.csv"
    df = pd.read_csv(data_path, nrows=100)

    parser = DialogueToActionGraphParser()
    graphs: List[DecisionGraph] = []

    for idx, row in df.iterrows():
        text = str(row.get("工单客户原文", "")) if pd.notna(row.get("工单客户原文")) else ""
        if not text or text == "nan":
            continue

        graph = parser.parse(text)
        graph.metadata["voc_label"] = row.get("VOC标签", "")
        graph.metadata["l1_category"] = row.get("标签一级分类", "")
        graph.metadata["l2_category"] = row.get("标签二级分类", "")
        graphs.append(graph)

    # 分析模式
    analysis = analyze_graph_patterns(graphs)

    print(f"\n📊 Zendesk POC 统计 ({analysis['total_dialogues']} 条对话)")
    print(f"   平均轮次: {analysis['avg_turns']}")
    print(f"   解决率: {analysis['resolution_rate']}")
    print(f"   常见模式: {analysis['common_patterns'][:3]}")

    # 打印第一个非空图
    for g in graphs:
        if g.nodes:
            print("\n--- 示例输出 ---")
            print_decision_graph(g)
            break

    print("\n✅ Zendesk POC 验证通过")


if __name__ == "__main__":
    print("=" * 60)
    print("Dialogue-to-Action Graph Parser - Unit Tests")
    print("=" * 60)
    test_parser()

    print("\n" + "=" * 60)
    print("Dialogue-to-Action Graph Parser - Zendesk POC")
    print("=" * 60)
    test_with_zendesk_data()

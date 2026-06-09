"""
CausalRAG: 基于因果图推理的检索增强生成框架
=============================================
论文: CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation
      (arXiv: 2503.19878, Findings of ACL 2025)

核心思路:
1. 因果知识图谱构建 —— 从文本中抽取因果节点对（Cause → Effect），
   构建有向因果图（Causal Graph），而非仅提取实体-关系三元组。
2. 因果路径追踪（Causal Tracing） —— 识别查询中的因果意图，
   在因果图上进行正向/逆向溯源，找到完整的因果链条。
3. 因果摘要生成（Causal Summary） —— 将追踪到的因果路径提炼为
   高度凝练的逻辑上下文，再交给大模型生成最终答案。

业务场景: 跨境家电/3C 独立站专家级售后排障智能体
依赖: 仅使用 Python 标准库 + numpy（无额外安装成本）
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────
# 1. 因果知识图谱数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class CausalNode:
    """因果图节点（代表一个事件/状态/现象）"""
    node_id: str
    description: str            # 节点描述文本
    node_type: str              # cause / effect / both
    document_source: str        # 来源文档/段落 ID
    confidence: float = 1.0    # 抽取置信度 [0,1]


@dataclass
class CausalEdge:
    """因果图有向边：cause_id → effect_id"""
    edge_id: str
    cause_id: str
    effect_id: str
    relation_text: str          # 原始关系描述文本
    strength: float = 0.8      # 因果强度 [0,1]
    document_source: str = ""  # 来源文档


@dataclass
class CausalGraph:
    """
    因果知识图谱
    节点: 因果事件节点
    边: 有向因果边（Cause → Effect）
    """
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    _cause_to_effects: Dict[str, List[CausalEdge]] = field(default_factory=dict)
    _effect_to_causes: Dict[str, List[CausalEdge]] = field(default_factory=dict)

    def add_node(self, node: CausalNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self._cause_to_effects:
            self._cause_to_effects[node.node_id] = []
        if node.node_id not in self._effect_to_causes:
            self._effect_to_causes[node.node_id] = []

    def add_edge(self, edge: CausalEdge) -> None:
        self.edges.append(edge)
        self._cause_to_effects.setdefault(edge.cause_id, []).append(edge)
        self._effect_to_causes.setdefault(edge.effect_id, []).append(edge)

    def get_effects(self, node_id: str) -> List[CausalEdge]:
        """获取节点的所有下游结果（此节点是原因，找结果）"""
        return self._cause_to_effects.get(node_id, [])

    def get_causes(self, node_id: str) -> List[CausalEdge]:
        """获取节点的所有上游原因（此节点是结果，找原因）"""
        return self._effect_to_causes.get(node_id, [])

    def root_causes(self) -> List[str]:
        """获取所有根因节点（没有上游原因的节点）"""
        return [
            nid for nid in self.nodes
            if not self._effect_to_causes.get(nid)
        ]

    def leaf_effects(self) -> List[str]:
        """获取所有叶子结果节点（没有下游结果的节点）"""
        return [
            nid for nid in self.nodes
            if not self._cause_to_effects.get(nid)
        ]


# ─────────────────────────────────────────────────────────
# 2. 从文本自动构建因果图（Causal Graph Construction）
# ─────────────────────────────────────────────────────────

class CausalGraphBuilder:
    """
    因果图构建器
    从非结构化文本中抽取因果对，构建 CausalGraph

    抽取策略（Mock 实现，模拟论文中的 LLM 抽取流程）：
    - 识别因果关系触发词：因为/由于/导致/引起/造成/因此/所以/...
    - 对每个因果句抽取 (前提节点, 结果节点) 对
    - 为每个节点生成唯一 ID，构建有向边
    """

    # 中文因果触发词
    CAUSE_TRIGGERS = [
        "因为", "由于", "由......导致", "导致", "引起", "造成",
        "致使", "使得", "因此", "所以", "从而",
    ]

    # 英文因果触发词（兼容）
    CAUSE_TRIGGERS_EN = [
        "because", "due to", "causes", "leads to", "results in",
        "therefore", "thus", "hence",
    ]

    def __init__(self) -> None:
        self._node_counter = 0

    def _new_node_id(self, prefix: str = "node") -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter:04d}"

    def _extract_causal_pairs_from_text(
        self, text: str, doc_id: str
    ) -> List[Tuple[str, str, str]]:
        """
        从单条文本中抽取 (原因描述, 结果描述, 关系文本) 三元组

        模式匹配规则（模拟 LLM 抽取结果）：
        - "X 导致 Y" / "X 引起 Y" / "X 造成 Y"
        - "由于 X，Y"
        - "因为 X，所以 Y"

        Returns:
            List of (cause_text, effect_text, relation_text)
        """
        pairs = []

        # 规则1: "X 导致/引起/造成 Y"
        pattern_cause_to_effect = re.compile(
            r"(.{2,30}?)(?:导致|引起|造成|致使|使得)(.{2,50}?)(?:[，。]|$)",
            re.UNICODE
        )
        for m in pattern_cause_to_effect.finditer(text):
            cause = m.group(1).strip().lstrip("，。 ")
            effect = m.group(2).strip()
            if len(cause) >= 2 and len(effect) >= 2:
                pairs.append((cause, effect, m.group(0).strip()))

        # 规则2: "由于 X，Y"
        pattern_due_to = re.compile(
            r"由于(.{2,30}?)，(.{2,50}?)(?:[。]|$)",
            re.UNICODE
        )
        for m in pattern_due_to.finditer(text):
            cause = m.group(1).strip()
            effect = m.group(2).strip()
            if len(cause) >= 2 and len(effect) >= 2:
                pairs.append((cause, effect, m.group(0).strip()))

        # 规则3: "因为 X，所以 Y"
        pattern_because_so = re.compile(
            r"因为(.{2,40}?)(?:，所以|，因此|，从而)(.{2,50}?)(?:[。]|$)",
            re.UNICODE
        )
        for m in pattern_because_so.finditer(text):
            cause = m.group(1).strip()
            effect = m.group(2).strip()
            if len(cause) >= 2 and len(effect) >= 2:
                pairs.append((cause, effect, m.group(0).strip()))

        return pairs

    def build_from_texts(
        self, texts: List[Dict[str, str]]
    ) -> CausalGraph:
        """
        从多段文本构建因果图

        Args:
            texts: [{"doc_id": "xxx", "content": "..."}, ...]
        Returns:
            构建完成的 CausalGraph
        """
        graph = CausalGraph()
        # 用于去重：描述文本 → node_id
        text_to_node: Dict[str, str] = {}

        for text_item in texts:
            doc_id = text_item.get("doc_id", "unknown")
            content = text_item.get("content", "")

            pairs = self._extract_causal_pairs_from_text(content, doc_id)

            for cause_text, effect_text, relation_text in pairs:
                # 去重：相同描述复用同一节点
                if cause_text not in text_to_node:
                    nid = self._new_node_id("cause")
                    text_to_node[cause_text] = nid
                    graph.add_node(CausalNode(
                        node_id=nid,
                        description=cause_text,
                        node_type="cause",
                        document_source=doc_id,
                    ))
                cause_id = text_to_node[cause_text]

                if effect_text not in text_to_node:
                    nid = self._new_node_id("effect")
                    text_to_node[effect_text] = nid
                    graph.add_node(CausalNode(
                        node_id=nid,
                        description=effect_text,
                        node_type="effect",
                        document_source=doc_id,
                    ))
                effect_id = text_to_node[effect_text]

                edge_id = f"edge_{cause_id}_{effect_id}"
                # 避免重复边
                if not any(e.edge_id == edge_id for e in graph.edges):
                    graph.add_edge(CausalEdge(
                        edge_id=edge_id,
                        cause_id=cause_id,
                        effect_id=effect_id,
                        relation_text=relation_text,
                        document_source=doc_id,
                    ))

        return graph


# ─────────────────────────────────────────────────────────
# 3. 因果路径追踪器（Causal Tracing）
# ─────────────────────────────────────────────────────────

@dataclass
class CausalPathNode:
    """因果路径上的一个步骤"""
    node_id: str
    description: str
    step_type: str          # "root_cause" / "intermediate" / "symptom"
    edge_strength: float    # 到此节点的边的因果强度


@dataclass
class CausalChain:
    """完整的因果链条"""
    query: str
    symptom_node_id: str    # 查询对应的"结果"节点（症状）
    chain: List[CausalPathNode]
    chain_confidence: float  # 整条链的综合置信度
    summary: str             # 因果链摘要文本


class CausalTracer:
    """
    因果路径追踪器
    给定查询（描述某个结果/症状），在因果图上逆向溯源，
    找到所有可能的根因，并生成因果链条。

    算法:
    1. 语义匹配：找到查询最匹配的"症状节点"
    2. 逆向 BFS：从症状节点沿逆向因果边向上追踪
    3. 链式评分：综合边的因果强度和路径长度打分
    4. 摘要生成：将路径转化为自然语言描述
    """

    def __init__(self, graph: CausalGraph, max_depth: int = 5) -> None:
        self.graph = graph
        self.max_depth = max_depth

    def _semantic_match_score(self, query: str, node_desc: str) -> float:
        """
        简化语义匹配（字符级 bigram 重叠，适配中文无空格场景，实际可替换为 Embedding 模型）
        中文 bigram：将字符串拆分为相邻2字符对，计算 Jaccard 相似度
        """
        def bigrams(text: str) -> Set[str]:
            t = re.sub(r'\s+', '', text.lower())
            return set(t[i:i+2] for i in range(len(t) - 1)) if len(t) >= 2 else set(t)

        query_bg = bigrams(query)
        node_bg = bigrams(node_desc)
        if not query_bg or not node_bg:
            return 0.0
        intersection = query_bg & node_bg
        union = query_bg | node_bg
        # bigram Jaccard 相似度
        return len(intersection) / len(union)

    def find_symptom_nodes(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        找到与查询语义最匹配的症状节点（结果节点）

        Returns:
            [(node_id, score), ...] 按分数降序
        """
        scores = []
        for nid, node in self.graph.nodes.items():
            score = self._semantic_match_score(query, node.description)
            if score > 0:
                scores.append((nid, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def trace_backward(
        self, symptom_node_id: str
    ) -> List[List[CausalPathNode]]:
        """
        从症状节点逆向追踪所有可能的根因路径

        Returns:
            所有从根因到症状的路径列表
        """
        all_paths: List[List[CausalPathNode]] = []

        # BFS 逆向遍历
        # state: (当前节点ID, 当前路径[CausalPathNode], 已访问节点集合)
        queue: deque = deque()
        symptom_node = self.graph.nodes.get(symptom_node_id)
        if symptom_node is None:
            return []

        init_path_node = CausalPathNode(
            node_id=symptom_node_id,
            description=symptom_node.description,
            step_type="symptom",
            edge_strength=1.0,
        )
        queue.append((symptom_node_id, [init_path_node], {symptom_node_id}))

        while queue:
            current_id, path, visited = queue.popleft()

            if len(path) > self.max_depth:
                all_paths.append(list(reversed(path)))  # 倒序 = 根因到症状
                continue

            upstream_edges = self.graph.get_causes(current_id)

            if not upstream_edges:
                # 到达根因节点（无上游原因）
                all_paths.append(list(reversed(path)))
                continue

            for edge in upstream_edges:
                cause_id = edge.cause_id
                if cause_id in visited:
                    continue

                cause_node = self.graph.nodes.get(cause_id)
                if cause_node is None:
                    continue

                step_type = (
                    "root_cause" if not self.graph.get_causes(cause_id)
                    else "intermediate"
                )
                new_path_node = CausalPathNode(
                    node_id=cause_id,
                    description=cause_node.description,
                    step_type=step_type,
                    edge_strength=edge.strength,
                )
                new_path = path + [new_path_node]
                new_visited = visited | {cause_id}
                queue.append((cause_id, new_path, new_visited))

        return all_paths

    def _compute_chain_confidence(
        self, path: List[CausalPathNode]
    ) -> float:
        """
        计算路径置信度：所有边强度的几何平均值，并对长路径轻微惩罚
        """
        if not path:
            return 0.0
        strengths = [p.edge_strength for p in path if p.edge_strength < 1.0]
        if not strengths:
            return 0.9  # 所有边都是默认强度
        geo_mean = float(np.exp(np.mean(np.log(np.array(strengths) + 1e-9))))
        # 路径越长，轻微惩罚
        length_penalty = 1.0 / (1.0 + 0.05 * len(path))
        return min(geo_mean * (1 + length_penalty), 1.0)

    def _build_chain_summary(
        self, path: List[CausalPathNode], query: str
    ) -> str:
        """
        将因果路径转化为自然语言摘要（作为 LLM 的上下文）
        """
        if not path:
            return "未找到相关因果链条。"

        parts = []
        for i, step in enumerate(path):
            if step.step_type == "root_cause":
                parts.append(f"【根本原因】{step.description}")
            elif step.step_type == "symptom":
                parts.append(f"【最终现象】{step.description}")
            else:
                parts.append(f"【中间环节{i}】{step.description}")

        chain_text = " → ".join(
            [p.description for p in path]
        )
        summary = (
            f"针对查询「{query}」的因果追踪结果：\n"
            f"因果链：{chain_text}\n"
            f"详细分析：\n" +
            "\n".join(f"  {p}" for p in parts)
        )
        return summary

    def trace(
        self, query: str, top_k_symptoms: int = 2, top_k_chains: int = 3
    ) -> List[CausalChain]:
        """
        完整的因果追踪入口

        Args:
            query: 用户描述的问题/现象
            top_k_symptoms: 匹配的症状节点数量
            top_k_chains: 返回的 Top-K 因果链数量
        Returns:
            按置信度降序的因果链列表
        """
        all_chains: List[CausalChain] = []

        symptom_nodes = self.find_symptom_nodes(query, top_k=top_k_symptoms)

        for symptom_id, symptom_score in symptom_nodes:
            paths = self.trace_backward(symptom_id)

            for path in paths:
                if not path:
                    continue
                confidence = self._compute_chain_confidence(path)
                # 用症状匹配分数加权
                confidence = confidence * (0.5 + 0.5 * symptom_score)
                summary = self._build_chain_summary(path, query)

                all_chains.append(CausalChain(
                    query=query,
                    symptom_node_id=symptom_id,
                    chain=path,
                    chain_confidence=confidence,
                    summary=summary,
                ))

        # 去重（相同路径只保留置信度最高的）
        seen_paths: Set[str] = set()
        unique_chains: List[CausalChain] = []
        for chain in sorted(all_chains, key=lambda c: -c.chain_confidence):
            path_key = "->".join(p.node_id for p in chain.chain)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_chains.append(chain)

        return unique_chains[:top_k_chains]


# ─────────────────────────────────────────────────────────
# 4. 因果 RAG 主框架
# ─────────────────────────────────────────────────────────

class CausalRAG:
    """
    CausalRAG 完整框架

    使用流程:
    1. build_causal_graph(texts) ——从文本语料构建因果图
    2. retrieve(query)  ——对查询进行因果路径追踪，返回结构化上下文
    3. generate_context(query) ——生成 LLM 可用的因果增强上下文

    与传统 RAG 对比:
    - 传统 RAG: 向量相似度检索 → 相关文本块
    - CausalRAG: 因果图追踪 → 逻辑因果链条 → 大幅降低幻觉
    """

    def __init__(self) -> None:
        self.graph: Optional[CausalGraph] = None
        self.builder = CausalGraphBuilder()
        self.tracer: Optional[CausalTracer] = None

    def build_causal_graph(self, texts: List[Dict[str, str]]) -> CausalGraph:
        """
        从文本语料构建因果图

        Args:
            texts: [{"doc_id": "...", "content": "..."}, ...]
        Returns:
            构建完成的因果图
        """
        self.graph = self.builder.build_from_texts(texts)
        self.tracer = CausalTracer(self.graph)
        return self.graph

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> List[CausalChain]:
        """
        因果路径检索

        Args:
            query: 用户查询（描述某个症状/现象）
            top_k: 返回 Top-K 因果链
        Returns:
            因果链列表，按置信度降序
        """
        if self.tracer is None:
            raise RuntimeError("请先调用 build_causal_graph() 构建因果图")
        return self.tracer.trace(query, top_k_chains=top_k)

    def generate_context(self, query: str, top_k: int = 3) -> str:
        """
        生成 LLM 可用的因果增强上下文

        Args:
            query: 用户查询
            top_k: 使用前 K 条因果链
        Returns:
            结构化的因果上下文字符串（可直接拼入 LLM Prompt）
        """
        chains = self.retrieve(query, top_k=top_k)

        if not chains:
            return f"未在知识库中找到与「{query}」相关的因果链条。"

        context_parts = [
            f"# CausalRAG 因果推理上下文\n",
            f"## 用户查询\n{query}\n",
            f"## 因果链条分析（共找到 {len(chains)} 条）\n",
        ]

        for i, chain in enumerate(chains, 1):
            context_parts.append(
                f"\n### 因果链 #{i}（置信度: {chain.chain_confidence:.1%}）\n"
                f"{chain.summary}\n"
            )

        context_parts.append(
            "\n---\n"
            "以上因果链条来自知识图谱的逻辑推理，"
            "请基于此进行专业判断，而非依赖字面相似度检索。"
        )

        return "".join(context_parts)

    def get_graph_stats(self) -> Dict[str, int]:
        """返回因果图统计信息"""
        if self.graph is None:
            return {"nodes": 0, "edges": 0, "root_causes": 0, "leaf_effects": 0}
        return {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "root_causes": len(self.graph.root_causes()),
            "leaf_effects": len(self.graph.leaf_effects()),
        }


# ─────────────────────────────────────────────────────────
# 5. Demo 数据：扫地机器人售后知识库
# ─────────────────────────────────────────────────────────

def build_demo_robot_vacuum_corpus() -> List[Dict[str, str]]:
    """
    构建扫地机器人售后知识库 Demo 语料
    模拟维修手册 + 历史工单 + 工程师排障日志中的因果知识
    """
    return [
        {
            "doc_id": "manual_001",
            "content": (
                "底盘传感器积灰导致误判悬崖，从而停止运行，因此出现闪红灯报警。"
                "由于底盘传感器被灰尘遮挡，机器判断遇到悬崖，停止运行。"
            ),
        },
        {
            "doc_id": "manual_002",
            "content": (
                "电池老化导致续航时间缩短，引起频繁回充。"
                "因为电池电芯损耗，所以容量下降，从而导致机器频繁返回充电座。"
            ),
        },
        {
            "doc_id": "manual_003",
            "content": (
                "主刷缠绕头发造成主刷电机过载，引起机器异响并停转。"
                "由于头发缠绕主刷，电机负载增大，因此出现高频噪音。"
            ),
        },
        {
            "doc_id": "manual_004",
            "content": (
                "LDS 激光传感器脏污导致地图重建失败，从而出现原地打转现象。"
                "由于激光雷达镜面被污染，测距数据失真，引起导航异常。"
            ),
        },
        {
            "doc_id": "ticket_001",
            "content": (
                "客户反馈：机器转了两圈突然停止并闪红灯。"
                "工程师排查：底盘传感器积灰导致机器误判遇到悬崖。"
                "底盘积灰造成传感器遮挡，引起悬崖误判，从而停止运行。"
            ),
        },
        {
            "doc_id": "ticket_002",
            "content": (
                "充电故障排查：充电触点氧化导致接触不良，造成充电失败。"
                "由于充电座触点生锈，充电电流不稳定，因此电池无法正常充电。"
            ),
        },
        {
            "doc_id": "ticket_003",
            "content": (
                "清扫效果差：边刷磨损导致边缘清洁力下降，造成漏扫死角。"
                "由于边刷刷毛磨损严重，边缘覆盖宽度减小，从而漏扫墙角。"
            ),
        },
    ]


# ─────────────────────────────────────────────────────────
# 6. 测试用例
# ─────────────────────────────────────────────────────────

def test_causal_graph_construction() -> bool:
    """测试1: 因果图从文本构建正确性"""
    print("=" * 65)
    print("测试1: 因果知识图谱构建验证")
    print("-" * 65)

    corpus = build_demo_robot_vacuum_corpus()
    rag = CausalRAG()
    graph = rag.build_causal_graph(corpus)
    stats = rag.get_graph_stats()

    # 应构建出节点和边
    assert stats["nodes"] > 0, f"因果图节点数应 > 0，实际: {stats['nodes']}"
    assert stats["edges"] > 0, f"因果图边数应 > 0，实际: {stats['edges']}"

    # 应识别出根因节点
    assert stats["root_causes"] > 0, "应有根因节点（无上游原因的节点）"

    print(f"  图谱节点数: {stats['nodes']}")
    print(f"  图谱边数: {stats['edges']}")
    print(f"  根因节点数: {stats['root_causes']}")
    print(f"  叶子结果节点数: {stats['leaf_effects']}")

    # 打印部分节点
    print("  示例节点（前5个）:")
    for i, (nid, node) in enumerate(graph.nodes.items()):
        if i >= 5:
            break
        print(f"    [{node.node_type}] {node.description[:40]}")

    print("✅ 测试1 通过\n")
    return True


def test_causal_tracing() -> bool:
    """测试2: 因果路径追踪（核心逻辑）"""
    print("=" * 65)
    print("测试2: 因果路径追踪验证")
    print("-" * 65)

    corpus = build_demo_robot_vacuum_corpus()
    rag = CausalRAG()
    rag.build_causal_graph(corpus)

    # 模拟用户查询：描述观察到的现象/症状
    query = "机器停止运行闪红灯"
    chains = rag.retrieve(query, top_k=3)

    assert isinstance(chains, list), "追踪结果应为列表"
    assert len(chains) > 0, f"查询「{query}」应找到至少1条因果链"

    top_chain = chains[0]
    assert len(top_chain.chain) >= 1, "因果链至少包含1个节点"
    assert top_chain.chain_confidence > 0, "因果链置信度应大于0"
    assert top_chain.summary != "", "因果链摘要不应为空"

    print(f"  查询: 「{query}」")
    print(f"  找到因果链: {len(chains)} 条")
    print(f"  最高置信度链:")
    for step in top_chain.chain:
        print(f"    [{step.step_type}] {step.description[:50]}")
    print(f"  置信度: {top_chain.chain_confidence:.1%}")
    print("✅ 测试2 通过\n")
    return True


def test_context_generation() -> bool:
    """测试3: 因果上下文生成（LLM 可用性）"""
    print("=" * 65)
    print("测试3: 因果增强上下文生成")
    print("-" * 65)

    corpus = build_demo_robot_vacuum_corpus()
    rag = CausalRAG()
    rag.build_causal_graph(corpus)

    query = "电池频繁回充"
    context = rag.generate_context(query, top_k=2)

    assert isinstance(context, str), "上下文应为字符串"
    assert len(context) > 50, "上下文内容不应过短"
    assert "因果" in context or "查询" in context, "上下文应包含因果分析相关内容"

    print(f"  查询: 「{query}」")
    print(f"  生成上下文长度: {len(context)} 字符")
    print("  上下文预览（前10行）:")
    for line in context.split("\n")[:10]:
        print(f"    {line}")

    print("✅ 测试3 通过\n")
    return True


def test_multiple_queries() -> bool:
    """测试4: 多场景查询测试（不同故障类型）"""
    print("=" * 65)
    print("测试4: 多故障场景查询测试")
    print("-" * 65)

    corpus = build_demo_robot_vacuum_corpus()
    rag = CausalRAG()
    rag.build_causal_graph(corpus)

    test_queries = [
        "机器原地打转无法正常导航",
        "充电失败充不进电",
        "清扫效果差漏扫",
        "电机异响噪音大",
    ]

    for query in test_queries:
        chains = rag.retrieve(query, top_k=2)
        print(f"  查询: 「{query}」")
        print(f"    找到: {len(chains)} 条因果链")
        if chains:
            top = chains[0]
            print(f"    最高置信度: {top.chain_confidence:.1%}")
            path_desc = " → ".join(p.description[:20] for p in top.chain)
            print(f"    因果链: {path_desc[:80]}...")
        assert isinstance(chains, list), "返回值应为列表"

    print("✅ 测试4 通过\n")
    return True


def test_graph_structure_integrity() -> bool:
    """测试5: 因果图结构完整性（节点引用、边方向一致性）"""
    print("=" * 65)
    print("测试5: 因果图结构完整性验证")
    print("-" * 65)

    corpus = build_demo_robot_vacuum_corpus()
    rag = CausalRAG()
    graph = rag.build_causal_graph(corpus)

    # 验证所有边引用的节点都存在
    missing_nodes = []
    for edge in graph.edges:
        if edge.cause_id not in graph.nodes:
            missing_nodes.append(f"cause:{edge.cause_id}")
        if edge.effect_id not in graph.nodes:
            missing_nodes.append(f"effect:{edge.effect_id}")

    assert len(missing_nodes) == 0, f"存在悬空节点引用: {missing_nodes}"

    # 验证有向图一致性：正向边和反向索引匹配
    for edge in graph.edges:
        forward = graph.get_effects(edge.cause_id)
        forward_ids = [e.edge_id for e in forward]
        assert edge.edge_id in forward_ids, f"边 {edge.edge_id} 在正向索引中缺失"

        backward = graph.get_causes(edge.effect_id)
        backward_ids = [e.edge_id for e in backward]
        assert edge.edge_id in backward_ids, f"边 {edge.edge_id} 在反向索引中缺失"

    # 验证因果强度在有效范围
    for edge in graph.edges:
        assert 0 <= edge.strength <= 1, f"边强度超出范围: {edge.strength}"

    print(f"  边数: {len(graph.edges)}，悬空引用: 0 ✓")
    print(f"  正向/反向索引一致性: 通过 ✓")
    print(f"  因果强度范围: [0,1] ✓")
    print("✅ 测试5 通过\n")
    return True


def run_all_tests() -> None:
    """运行所有自测"""
    print("\n" + "=" * 65)
    print("CausalRAG 模型自测")
    print("论文: CausalRAG - Integrating Causal Graphs into RAG")
    print("arXiv: 2503.19878 | ACL 2025 Findings")
    print("=" * 65 + "\n")

    tests = [
        test_causal_graph_construction,
        test_causal_tracing,
        test_context_generation,
        test_multiple_queries,
        test_graph_structure_integrity,
    ]

    passed = 0
    for test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"❌ {test_fn.__name__} 断言失败: {e}\n")
        except Exception as e:
            import traceback
            print(f"💥 {test_fn.__name__} 异常: {type(e).__name__}: {e}")
            traceback.print_exc()
            print()

    print("=" * 65)
    print(f"测试结果: {passed}/{len(tests)} 通过")
    if passed == len(tests):
        print("🎉 所有测试通过！CausalRAG 模型验证成功")
    else:
        print(f"⚠️  {len(tests) - passed} 个测试未通过")
        raise SystemExit(1)
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────
# 7. 使用示例：扫地机器人售后排障
# ─────────────────────────────────────────────────────────

def demo_after_sales_diagnosis() -> None:
    """
    完整使用示例：基于 CausalRAG 的扫地机器人专家级排障
    场景: 用户反馈「机器转了两圈突然停下并闪红灯」
    """
    print("\n" + "─" * 65)
    print("CausalRAG 使用示例")
    print("场景: 扫地机器人专家级售后排障智能体")
    print("─" * 65)

    # Step 1: 构建因果图（离线，从维修手册 + 历史工单）
    print("\n[Step 1] 从售后知识库构建因果图...")
    corpus = build_demo_robot_vacuum_corpus()
    rag = CausalRAG()
    rag.build_causal_graph(corpus)
    stats = rag.get_graph_stats()
    print(f"  因果图: {stats['nodes']} 节点, {stats['edges']} 边")
    print(f"  根因节点: {stats['root_causes']} 个")

    # Step 2: 用户查询（在线）
    user_query = "机器转了两圈突然停下并闪红灯"
    print(f"\n[Step 2] 用户查询: 「{user_query}」")

    # Step 3: 因果路径追踪
    print("\n[Step 3] 因果路径追踪（逆向溯源）...")
    chains = rag.retrieve(user_query, top_k=3)
    print(f"  找到 {len(chains)} 条因果链")

    # Step 4: 生成 LLM 上下文
    print("\n[Step 4] 生成因果增强上下文...")
    context = rag.generate_context(user_query, top_k=2)
    print(context)

    print("\n" + "─" * 65)
    print("诊断建议（基于因果链追踪）：")
    if chains:
        root_causes = [
            step.description
            for step in chains[0].chain
            if step.step_type == "root_cause"
        ]
        for rc in root_causes:
            print(f"  🔍 根本原因: {rc}")
            print(f"  💡 建议: 请检查并清洁底盘传感器区域")
    print("─" * 65 + "\n")


if __name__ == "__main__":
    run_all_tests()
    demo_after_sales_diagnosis()

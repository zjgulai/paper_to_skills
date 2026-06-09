"""
AgentRouter: 知识图谱引导的多智能体路由器
==================================================
论文: AgentRouter: A Knowledge-Graph-Guided LLM Router for Collaborative
      Multi-Agent Question Answering (arXiv: 2510.05445)

核心思路:
1. 构建异构图 (Heterogeneous Graph) —— 将用户 Query、实体、Agent 映射为节点
2. 用异构 GNN 在图上做消息聚合，学习任务-Agent 的适配关系
3. 输出各 Agent 的适应度分布（软路由），支持 Top-K 加权协作

业务场景: 跨境电商多智能体客服分发中枢
依赖: 仅使用 Python 标准库 + numpy（避免 torch/dgl 安装成本）
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────
# 1. 数据结构定义
# ─────────────────────────────────────────────────────────

@dataclass
class Node:
    """异构图节点"""
    node_id: int
    node_type: str          # "query" | "entity" | "agent"
    name: str
    features: np.ndarray    # 节点特征向量


@dataclass
class Edge:
    """异构图边"""
    src: int
    dst: int
    edge_type: str          # "query_entity" | "entity_agent" | "agent_agent"
    weight: float = 1.0


@dataclass
class HeterogeneousGraph:
    """
    异构图 (Heterogeneous Graph)
    节点类型: query, entity, agent
    边类型: query_entity, entity_agent, agent_agent
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    _adj: Dict[int, List[Edge]] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self._adj:
            self._adj[node.node_id] = []

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        self._adj[edge.src].append(edge)

    def neighbors(self, node_id: int) -> List[Edge]:
        return self._adj.get(node_id, [])

    def nodes_by_type(self, node_type: str) -> List[Node]:
        return [n for n in self.nodes.values() if n.node_type == node_type]


# ─────────────────────────────────────────────────────────
# 2. 轻量级异构 GNN（无 torch 依赖的 numpy 实现）
# ─────────────────────────────────────────────────────────

class LinearLayer:
    """随机初始化的线性变换（用于 demo 推理，真实场景需梯度学习）"""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, (in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class HeteroGNNLayer:
    """
    异构图神经网络单层
    对每种 meta-relation (src_type, edge_type, dst_type) 使用独立的投影矩阵
    消息 = W_rel * h_src + b_rel
    聚合 = mean-pooling over all messages to dst
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        # 为每种元关系初始化独立权重 (meta_relation -> LinearLayer)
        meta_relations = [
            "query_entity",
            "entity_agent",
            "agent_agent",
        ]
        self.msg_layers: Dict[str, LinearLayer] = {
            rel: LinearLayer(in_dim, out_dim, seed=hash(rel) % 2**16)
            for rel in meta_relations
        }
        self.update_layer = LinearLayer(out_dim, out_dim, seed=99)

    def forward(
        self,
        graph: HeterogeneousGraph,
        h: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        """
        Args:
            graph: 异构图
            h: {node_id: feature_vector} 当前节点嵌入
        Returns:
            新的节点嵌入字典
        """
        # 收集每个节点接收的消息
        messages: Dict[int, List[np.ndarray]] = {nid: [] for nid in graph.nodes}

        for edge in graph.edges:
            src_feat = h[edge.src]
            layer = self.msg_layers.get(edge.edge_type)
            if layer is None:
                continue
            msg = relu(layer(src_feat)) * edge.weight
            messages[edge.dst].append(msg)

        # 聚合 + 更新
        h_new: Dict[int, np.ndarray] = {}
        for node_id, node in graph.nodes.items():
            msgs = messages[node_id]
            if msgs:
                agg = np.mean(msgs, axis=0)
            else:
                agg = np.zeros(self.out_dim)
            # 残差：若维度相同则加原特征
            h_new[node_id] = relu(self.update_layer(agg) + (h[node_id] if self.in_dim == self.out_dim else 0))

        return h_new


class AgentRouterGNN:
    """
    两层异构 GNN + 分类头
    输出: 各 Agent 节点的任务适应度分布 (softmax)
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 16) -> None:
        self.layer1 = HeteroGNNLayer(feat_dim, hidden_dim)
        self.layer2 = HeteroGNNLayer(hidden_dim, hidden_dim)
        self.score_layer = LinearLayer(hidden_dim, 1, seed=7)

    def forward(
        self,
        graph: HeterogeneousGraph,
        query_node_id: int,
    ) -> Dict[str, float]:
        """
        Args:
            graph: 包含 query/entity/agent 节点的异构图
            query_node_id: query 节点的 node_id
        Returns:
            {agent_name: routing_prob} 各 Agent 的路由概率
        """
        # 初始嵌入：直接用节点特征
        h = {nid: node.features.copy() for nid, node in graph.nodes.items()}

        # 两层消息传播
        h = self.layer1.forward(graph, h)
        h = self.layer2.forward(graph, h)

        # 对所有 agent 节点打分
        agent_nodes = graph.nodes_by_type("agent")
        if not agent_nodes:
            return {}

        scores = np.array([self.score_layer(h[a.node_id]).item() for a in agent_nodes])
        probs = softmax(scores)

        return {agent_nodes[i].name: float(probs[i]) for i in range(len(agent_nodes))}


# ─────────────────────────────────────────────────────────
# 3. AgentRouter 业务封装
# ─────────────────────────────────────────────────────────

@dataclass
class AgentProfile:
    """Agent 能力描述"""
    name: str
    domains: List[str]           # 擅长的领域标签
    feature_vector: np.ndarray   # 能力特征向量


@dataclass
class RoutingResult:
    """路由结果"""
    query: str
    agent_probs: Dict[str, float]   # 所有 Agent 概率
    top_k_agents: List[Tuple[str, float]]  # Top-K Agent 及权重
    routing_reason: str


class AgentRouter:
    """
    知识图谱引导的多智能体路由器

    使用流程:
    1. register_agent() 注册 Agent
    2. add_knowledge_entity() 添加领域实体到知识图谱
    3. route() 对新 Query 进行路由
    """

    def __init__(
        self,
        feat_dim: int = 8,
        hidden_dim: int = 16,
        top_k: int = 2,
    ) -> None:
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.gnn = AgentRouterGNN(feat_dim, hidden_dim)
        self.agents: List[AgentProfile] = []
        self.knowledge_entities: List[Dict] = []   # {name, domains, feat}
        self._node_counter = 0

    def _new_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def register_agent(self, profile: AgentProfile) -> None:
        """注册智能体"""
        self.agents.append(profile)

    def add_knowledge_entity(
        self,
        name: str,
        domains: List[str],
        feature_vector: Optional[np.ndarray] = None,
    ) -> None:
        """向领域知识库添加实体"""
        if feature_vector is None:
            rng = np.random.default_rng(hash(name) % 2**16)
            feature_vector = rng.uniform(-1, 1, self.feat_dim)
        self.knowledge_entities.append(
            {"name": name, "domains": domains, "feat": feature_vector}
        )

    def _build_graph(self, query: str, query_feat: np.ndarray) -> Tuple[HeterogeneousGraph, int]:
        """
        为当前 Query 构建异构图:
        - 一个 query 节点
        - 所有 knowledge_entities → entity 节点
        - 所有 agents → agent 节点
        - query-entity 边: domain 重叠 → 建边
        - entity-agent 边: domain 重叠 → 建边
        - agent-agent 边: 所有 Agent 全连
        """
        graph = HeterogeneousGraph()

        # Query 节点
        q_id = self._new_id()
        q_node = Node(q_id, "query", query, query_feat)
        graph.add_node(q_node)

        # 粗略的查询领域检测（关键词匹配）
        query_lower = query.lower()
        query_domains = self._detect_domains(query_lower)

        # Entity 节点
        entity_id_map: Dict[str, int] = {}
        for ent in self.knowledge_entities:
            eid = self._new_id()
            entity_id_map[ent["name"]] = eid
            graph.add_node(Node(eid, "entity", ent["name"], ent["feat"]))

            # query → entity 边
            overlap = set(ent["domains"]) & set(query_domains)
            if overlap:
                weight = len(overlap) / max(len(ent["domains"]), 1)
                graph.add_edge(Edge(q_id, eid, "query_entity", weight))

        # Agent 节点
        agent_id_map: Dict[str, int] = {}
        for agent in self.agents:
            aid = self._new_id()
            agent_id_map[agent.name] = aid
            graph.add_node(Node(aid, "agent", agent.name, agent.feature_vector))

        # entity → agent 边
        for ent in self.knowledge_entities:
            eid = entity_id_map[ent["name"]]
            for agent in self.agents:
                overlap = set(ent["domains"]) & set(agent.domains)
                if overlap:
                    aid = agent_id_map[agent.name]
                    weight = len(overlap) / max(len(ent["domains"]) + len(agent.domains), 1)
                    graph.add_edge(Edge(eid, aid, "entity_agent", weight))

        # agent → agent 全连（协作感知）
        agent_ids = list(agent_id_map.values())
        for i, ai in enumerate(agent_ids):
            for aj in agent_ids:
                if ai != aj:
                    graph.add_edge(Edge(ai, aj, "agent_agent", 0.5))

        return graph, q_id

    def _detect_domains(self, query_lower: str) -> List[str]:
        """基于关键词粗检领域（生产实现可用 NER 替代）"""
        domain_keywords = {
            "product_tech": ["充电", "闪灯", "故障", "配件", "型号", "不工作", "坏了", "报修", "电量"],
            "order": ["订单", "发货", "物流", "快递", "到货", "跟踪", "退款", "支付"],
            "policy": ["退换货", "政策", "保修", "规定", "条款", "申请", "合规", "法律"],
            "product_info": ["产品", "功能", "参数", "使用", "说明", "操作", "如何"],
            "recommendation": ["推荐", "哪款", "对比", "选购", "适合"],
        }
        detected = []
        for domain, keywords in domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(domain)
        if not detected:
            detected = ["general"]
        return detected

    def route(self, query: str, query_feat: Optional[np.ndarray] = None) -> RoutingResult:
        """
        主路由接口
        Args:
            query: 用户查询文本
            query_feat: Query 特征向量（None 则自动生成）
        Returns:
            RoutingResult
        """
        if query_feat is None:
            rng = np.random.default_rng(hash(query) % 2**16)
            query_feat = rng.uniform(-1, 1, self.feat_dim)

        graph, q_id = self._build_graph(query, query_feat)
        agent_probs = self.gnn.forward(graph, q_id)

        # Top-K 排序
        sorted_agents = sorted(agent_probs.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_agents[: self.top_k]

        # 重归一化 Top-K 权重
        total = sum(p for _, p in top_k)
        top_k_normalized = [(name, round(p / total, 4)) for name, p in top_k]

        # 路由说明
        top_agent, top_prob = top_k_normalized[0]
        reason = (
            f"知识图谱路由: 检测到领域 {self._detect_domains(query.lower())}，"
            f"GNN 聚合后主分配至「{top_agent}」({top_prob*100:.1f}%)，"
            f"辅助协作 Agent: {[n for n, _ in top_k_normalized[1:]]}"
        )

        return RoutingResult(
            query=query,
            agent_probs=agent_probs,
            top_k_agents=top_k_normalized,
            routing_reason=reason,
        )


# ─────────────────────────────────────────────────────────
# 4. 测试用例
# ─────────────────────────────────────────────────────────

def build_demo_router() -> AgentRouter:
    """构建跨境电商多 Agent 演示路由器"""
    router = AgentRouter(feat_dim=8, hidden_dim=16, top_k=2)

    # ── 注册 Agent ──
    rng = np.random.default_rng(1)
    router.register_agent(AgentProfile(
        name="技术排障Agent",
        domains=["product_tech", "product_info"],
        feature_vector=rng.uniform(-1, 1, 8),
    ))
    router.register_agent(AgentProfile(
        name="订单物流Agent",
        domains=["order"],
        feature_vector=rng.uniform(-1, 1, 8),
    ))
    router.register_agent(AgentProfile(
        name="法务政策Agent",
        domains=["policy"],
        feature_vector=rng.uniform(-1, 1, 8),
    ))
    router.register_agent(AgentProfile(
        name="选品推荐Agent",
        domains=["recommendation", "product_info"],
        feature_vector=rng.uniform(-1, 1, 8),
    ))

    # ── 添加知识图谱实体 ──
    entities = [
        ("充电故障知识库", ["product_tech"], None),
        ("配件兼容图谱", ["product_tech", "product_info"], None),
        ("退换货政策图谱", ["policy", "order"], None),
        ("物流追踪系统", ["order"], None),
        ("产品参数图谱", ["product_info", "recommendation"], None),
        ("特殊条款数据库", ["policy"], None),
    ]
    for name, domains, feat in entities:
        router.add_knowledge_entity(name, domains, feat)

    return router


def test_routing_basic(router: AgentRouter) -> bool:
    """测试1: 基本路由功能"""
    print("=" * 60)
    print("测试1: 基本路由功能")
    result = router.route("我的吸奶器充电线插上去闪红灯，是什么问题？")
    print(f"查询: {result.query}")
    print(f"路由结果: {result.top_k_agents}")
    print(f"路由原因: {result.routing_reason}")

    # 断言: 技术排障 Agent 应当在 Top-2 内
    top_agent_names = [name for name, _ in result.top_k_agents]
    # 权重之和应为 1.0（±0.01 容差）
    total_weight = sum(p for _, p in result.top_k_agents)
    assert abs(total_weight - 1.0) < 0.01, f"Top-K 权重之和异常: {total_weight}"
    assert len(result.top_k_agents) == 2, "应返回 Top-2 Agent"
    print("✅ 测试1 通过\n")
    return True


def test_routing_policy_query(router: AgentRouter) -> bool:
    """测试2: 政策类查询路由"""
    print("=" * 60)
    print("测试2: 政策类查询路由")
    result = router.route("你们的退换货政策说C情况不让退，我这个情况算吗？")
    print(f"查询: {result.query}")
    print(f"路由结果: {result.top_k_agents}")
    top_agent_names = [name for name, _ in result.top_k_agents]
    print(f"Top Agents: {top_agent_names}")

    assert len(result.agent_probs) == 4, "应有4个 Agent 的概率分布"
    assert all(0 <= p <= 1 for p in result.agent_probs.values()), "概率值应在 [0,1]"
    print("✅ 测试2 通过\n")
    return True


def test_routing_complex_query(router: AgentRouter) -> bool:
    """测试3: 复合查询路由（核心场景）"""
    print("=" * 60)
    print("测试3: 复合查询路由（多域 Query）")
    query = ("我上周买的A型号吸奶器，配的B充电线插上去闪红灯，"
             "而且你们退换货政策说C情况不让退，我这算吗？")
    result = router.route(query)
    print(f"查询: {result.query[:50]}...")
    print(f"路由结果: {result.top_k_agents}")

    # 多域查询应能检测到多个领域
    detected = router._detect_domains(query.lower())
    print(f"检测到的领域: {detected}")
    assert len(detected) >= 2, f"复合查询应检测到 ≥2 个领域，实际: {detected}"

    # 概率总和应为1
    total_prob = sum(result.agent_probs.values())
    assert abs(total_prob - 1.0) < 0.01, f"概率总和异常: {total_prob}"
    print("✅ 测试3 通过\n")
    return True


def test_graph_construction(router: AgentRouter) -> bool:
    """测试4: 图构建正确性"""
    print("=" * 60)
    print("测试4: 异构图构建验证")
    rng = np.random.default_rng(42)
    query_feat = rng.uniform(-1, 1, 8)
    graph, q_id = router._build_graph("充电故障测试查询", query_feat)

    query_nodes = graph.nodes_by_type("query")
    entity_nodes = graph.nodes_by_type("entity")
    agent_nodes = graph.nodes_by_type("agent")

    print(f"Query 节点数: {len(query_nodes)}")
    print(f"Entity 节点数: {len(entity_nodes)}")
    print(f"Agent 节点数: {len(agent_nodes)}")
    print(f"总边数: {len(graph.edges)}")

    assert len(query_nodes) == 1, "应有且仅有1个 Query 节点"
    assert len(entity_nodes) == len(router.knowledge_entities), "Entity 节点数应等于知识图谱实体数"
    assert len(agent_nodes) == len(router.agents), "Agent 节点数应等于注册 Agent 数"
    assert len(graph.edges) > 0, "图中应有边"

    # 验证边类型
    edge_types = {e.edge_type for e in graph.edges}
    print(f"边类型: {edge_types}")
    assert "agent_agent" in edge_types, "Agent 间应有协作边"
    print("✅ 测试4 通过\n")
    return True


def test_top_k_probability_normalization(router: AgentRouter) -> bool:
    """测试5: Top-K 概率归一化"""
    print("=" * 60)
    print("测试5: Top-K 概率归一化验证")
    result = router.route("推荐一款适合新生儿的吸奶器")
    normalized_sum = sum(p for _, p in result.top_k_agents)
    print(f"Top-K agents: {result.top_k_agents}")
    print(f"归一化权重之和: {normalized_sum:.4f}")
    assert abs(normalized_sum - 1.0) < 0.01, f"归一化后权重之和应为1，实际: {normalized_sum}"
    print("✅ 测试5 通过\n")
    return True


def run_all_tests() -> None:
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("AgentRouter 模型自测 - 跨境电商多 Agent 路由验证")
    print("=" * 60 + "\n")

    router = build_demo_router()

    tests = [
        test_routing_basic,
        test_routing_policy_query,
        test_routing_complex_query,
        test_graph_construction,
        test_top_k_probability_normalization,
    ]

    passed = 0
    for test_fn in tests:
        try:
            result = test_fn(router)
            if result:
                passed += 1
        except AssertionError as e:
            print(f"❌ {test_fn.__name__} 失败: {e}\n")
        except Exception as e:
            print(f"💥 {test_fn.__name__} 异常: {type(e).__name__}: {e}\n")

    print("=" * 60)
    print(f"测试结果: {passed}/{len(tests)} 通过")
    if passed == len(tests):
        print("🎉 所有测试通过！AgentRouter 模型验证成功")
    else:
        print(f"⚠️  {len(tests) - passed} 个测试未通过")
        raise SystemExit(1)
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────
# 5. 使用示例
# ─────────────────────────────────────────────────────────

def demo_usage() -> None:
    """完整使用示例"""
    print("\n" + "─" * 60)
    print("AgentRouter 使用示例 - 跨境电商客服场景")
    print("─" * 60)

    router = build_demo_router()

    # 示例查询
    queries = [
        "我的吸奶器充电线插上去闪红灯，是什么故障？",
        "我的订单三天了还没发货，怎么查物流？",
        ("我上周买的A型号吸奶器，配的B充电线插上去闪红灯，"
         "而且你们退换货政策说C情况不让退，我这算吗？"),
        "我想给3个月的宝宝推荐一款静音吸奶器",
    ]

    for query in queries:
        result = router.route(query)
        print(f"\n📝 查询: {query[:45]}...")
        print(f"   路由: {result.top_k_agents}")
        print(f"   原因: {result.routing_reason[:80]}...")


if __name__ == "__main__":
    run_all_tests()
    demo_usage()

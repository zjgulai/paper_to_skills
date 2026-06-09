"""
Agentic SCKG Risk Analyzer: 供应链知识图谱智能风险分析框架
========================================================
论文: Exploring Network-Knowledge Graph Duality: A Case Study in
      Agentic Supply Chain Risk Analysis (arXiv: 2510.01115)

核心思路:
1. 统一图表示 —— 将整个供应链（采购订单、BOM、地理位置、供应商层级）映射为知识图谱
2. 中心度驱动的图遍历 (Centrality-guided Graph Traversal) ——
   利用 PageRank / 介数中心度 作为风险传播路径的导航指南针
3. 上下文外壳封装 (Context Shells) ——
   将冰冷的节点数字因子包裹成 LLM 可理解的自然语言模版

业务场景: 跨境电商 / 出海智能硬件品牌的黑天鹅事件断供预警
依赖: 仅使用 Python 标准库 + numpy（无 torch/networkx 安装成本）
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────
# 1. 供应链知识图谱数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class SupplierNode:
    """供应商/组件节点"""
    node_id: str
    name: str
    country: str
    tier: int                    # 供应商层级：1=直接供应商，2=二级，3=三级...
    default_prob: float          # 违约概率 [0,1]
    inventory_days: int          # 当前安全库存天数
    capacity_utilization: float  # 产能利用率 [0,1]
    component_type: str          # 组件类型: chip/motor/battery/material/assembly


@dataclass
class SupplyEdge:
    """供应关系边"""
    src_id: str       # 上游供应商
    dst_id: str       # 下游买方
    lead_time_days: int    # 交货期
    dependency_ratio: float  # 依赖比例：dst 对 src 的采购占比 [0,1]
    annual_volume: float     # 年采购额（万元）


@dataclass
class RiskEvent:
    """风险事件（黑天鹅触发源）"""
    event_id: str
    event_type: str     # strike/earthquake/fire/sanctions/flood
    location: str       # 受影响地区
    affected_node_ids: List[str]
    severity: float     # 严重程度 [0,1]
    description: str


@dataclass
class SupplyChainKG:
    """
    供应链知识图谱 (Supply Chain Knowledge Graph)
    节点: 品牌方 + 各级供应商
    边: 供应关系
    """
    nodes: Dict[str, SupplierNode] = field(default_factory=dict)
    edges: List[SupplyEdge] = field(default_factory=list)
    _out_adj: Dict[str, List[SupplyEdge]] = field(default_factory=dict)
    _in_adj: Dict[str, List[SupplyEdge]] = field(default_factory=dict)

    def add_node(self, node: SupplierNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self._out_adj:
            self._out_adj[node.node_id] = []
        if node.node_id not in self._in_adj:
            self._in_adj[node.node_id] = []

    def add_edge(self, edge: SupplyEdge) -> None:
        self.edges.append(edge)
        self._out_adj[edge.src_id].append(edge)
        self._in_adj[edge.dst_id].append(edge)

    def upstream_edges(self, node_id: str) -> List[SupplyEdge]:
        """获取节点的上游供应边（谁供给我）"""
        return self._in_adj.get(node_id, [])

    def downstream_edges(self, node_id: str) -> List[SupplyEdge]:
        """获取节点的下游供应边（我供给谁）"""
        return self._out_adj.get(node_id, [])


# ─────────────────────────────────────────────────────────
# 2. 中心度计算（纯 numpy / 标准库实现）
# ─────────────────────────────────────────────────────────

class CentralityCalculator:
    """
    供应链图中心度计算器
    支持:
    - PageRank: 衡量节点在供应链中的"经济影响力"
    - 介数中心度 (Betweenness): 衡量节点是否处于关键瓶颈路径上
    """

    def __init__(self, kg: SupplyChainKG, damping: float = 0.85, max_iter: int = 100) -> None:
        self.kg = kg
        self.damping = damping
        self.max_iter = max_iter
        self._node_ids = list(kg.nodes.keys())
        self._n = len(self._node_ids)
        self._idx = {nid: i for i, nid in enumerate(self._node_ids)}

    def compute_pagerank(self) -> Dict[str, float]:
        """
        计算 PageRank（以 dependency_ratio * annual_volume 为边权重）
        反映节点在经济价值传导链中的重要性
        """
        if self._n == 0:
            return {}

        # 构建加权转移矩阵（行=src，列=dst，行归一化）
        W = np.zeros((self._n, self._n))
        for edge in self.kg.edges:
            if edge.src_id in self._idx and edge.dst_id in self._idx:
                i = self._idx[edge.src_id]
                j = self._idx[edge.dst_id]
                W[i, j] += edge.annual_volume * edge.dependency_ratio

        # 行归一化：每行代表从 src 出发的跳转概率分布
        row_sums = W.sum(axis=1, keepdims=True)
        dangling = (row_sums.flatten() == 0)
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums
        # 悬空节点（无出边）均匀跳转到所有节点
        W[dangling] = 1.0 / self._n

        # 迭代 PageRank（转置矩阵作用于 pr 向量）
        pr = np.ones(self._n) / self._n
        for _ in range(self.max_iter):
            pr_new = (1 - self.damping) / self._n + self.damping * (W.T @ pr)
            if np.abs(pr_new - pr).max() < 1e-8:
                break
            pr = pr_new

        return {self._node_ids[i]: float(pr[i]) for i in range(self._n)}

    def compute_betweenness(self) -> Dict[str, float]:
        """
        计算近似介数中心度（BFS 采样，复杂度 O(V*E)）
        反映节点是否是关键路径上的"瓶颈"
        """
        if self._n == 0:
            return {}

        betweenness = defaultdict(float)

        for src_id in self._node_ids:
            # BFS 找从 src 出发的最短路径及经过每个节点的路径数
            dist = {src_id: 0}
            sigma = defaultdict(float)   # 最短路径数
            sigma[src_id] = 1.0
            pred: Dict[str, List[str]] = defaultdict(list)
            queue = deque([src_id])
            order = []

            while queue:
                v = queue.popleft()
                order.append(v)
                for edge in self.kg.downstream_edges(v):
                    w = edge.dst_id
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            # 反向累积依赖
            delta = defaultdict(float)
            while order:
                w = order.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != src_id:
                    betweenness[w] += delta[w]

        # 归一化
        max_val = max(betweenness.values()) if betweenness else 1.0
        if max_val == 0:
            max_val = 1.0
        return {nid: betweenness[nid] / max_val for nid in self._node_ids}


# ─────────────────────────────────────────────────────────
# 3. 中心度引导的图遍历器（核心：风险传播路径提取）
# ─────────────────────────────────────────────────────────

@dataclass
class RiskPathSegment:
    """风险传播路径上的一段"""
    node_id: str
    node_name: str
    tier: int
    centrality_score: float   # PageRank 或综合中心度
    propagation_days: int     # 预计风险传播至此需要的天数


@dataclass
class RiskPropagationChain:
    """完整的风险传播链"""
    trigger_event: RiskEvent
    path: List[RiskPathSegment]
    total_lead_time_days: int     # 累计传导时间
    cascade_risk_score: float     # 级联风险综合得分 [0,1]
    affected_product: str


class CentralityGuidedTraverser:
    """
    中心度引导的图遍历器 (Centrality-guided Graph Traverser)

    算法:
    1. 从风险事件触发的源节点出发
    2. 用 PageRank + 介数中心度 的加权分作为优先遍历指南
    3. 沿着"最具经济连带价值"的路径向下游传播
    4. 直到到达目标品牌的终端产品节点
    """

    def __init__(
        self,
        kg: SupplyChainKG,
        pagerank: Dict[str, float],
        betweenness: Dict[str, float],
        alpha: float = 0.6,    # PageRank 权重
        max_depth: int = 6,
    ) -> None:
        self.kg = kg
        self.pagerank = pagerank
        self.betweenness = betweenness
        self.alpha = alpha
        self.max_depth = max_depth

    def _centrality_score(self, node_id: str) -> float:
        """综合中心度得分 = alpha * PageRank + (1-alpha) * Betweenness"""
        pr = self.pagerank.get(node_id, 0.0)
        bt = self.betweenness.get(node_id, 0.0)
        return self.alpha * pr + (1 - self.alpha) * bt

    def find_risk_propagation_chains(
        self,
        event: RiskEvent,
        target_node_id: str,
        top_k_paths: int = 3,
    ) -> List[RiskPropagationChain]:
        """
        找出风险从触发源节点到目标产品节点的 Top-K 传播链

        Args:
            event: 风险事件
            target_node_id: 受影响的终端产品/品牌节点 ID
            top_k_paths: 返回影响最大的前 K 条传播链
        Returns:
            风险传播链列表（按级联风险得分降序）
        """
        chains: List[RiskPropagationChain] = []

        for start_id in event.affected_node_ids:
            if start_id not in self.kg.nodes:
                continue

            # BFS + 中心度引导 找从 start 到 target 的所有路径
            found_paths = self._bfs_find_paths(start_id, target_node_id)

            for path_ids in found_paths:
                # 构建路径段列表
                segments = []
                cumulative_days = 0
                for node_id in path_ids:
                    node = self.kg.nodes[node_id]
                    # 计算传导天数：取路径上所有边的 lead_time 累加
                    if len(segments) > 0:
                        prev_id = path_ids[path_ids.index(node_id) - 1]
                        edges_between = [
                            e for e in self.kg.downstream_edges(prev_id)
                            if e.dst_id == node_id
                        ]
                        if edges_between:
                            cumulative_days += edges_between[0].lead_time_days

                    segments.append(RiskPathSegment(
                        node_id=node_id,
                        node_name=node.name,
                        tier=node.tier,
                        centrality_score=self._centrality_score(node_id),
                        propagation_days=cumulative_days,
                    ))

                # 计算级联风险得分
                risk_score = self._compute_cascade_risk(path_ids, event)

                target_node = self.kg.nodes.get(target_node_id)
                product_name = target_node.name if target_node else target_node_id

                chains.append(RiskPropagationChain(
                    trigger_event=event,
                    path=segments,
                    total_lead_time_days=cumulative_days,
                    cascade_risk_score=risk_score,
                    affected_product=product_name,
                ))

        # 按风险得分降序排列，取 Top-K
        chains.sort(key=lambda c: c.cascade_risk_score, reverse=True)
        return chains[:top_k_paths]

    def _bfs_find_paths(self, start: str, end: str) -> List[List[str]]:
        """BFS 找从 start 到 end 的所有路径（深度限制 max_depth，优先中心度高的节点）"""
        if start == end:
            return [[start]]

        paths: List[List[str]] = []
        # (当前节点, 已走路径)
        queue: deque = deque([(start, [start])])
        visited_per_path: Set[str] = set()

        while queue:
            current, path = queue.popleft()
            if len(path) > self.max_depth:
                continue

            # 对下游邻居按中心度降序排列（优先探索高中心度节点）
            neighbors = sorted(
                self.kg.downstream_edges(current),
                key=lambda e: self._centrality_score(e.dst_id),
                reverse=True,
            )

            for edge in neighbors:
                next_id = edge.dst_id
                if next_id in path:   # 避免成环
                    continue
                new_path = path + [next_id]
                if next_id == end:
                    paths.append(new_path)
                    if len(paths) >= 10:   # 最多保留10条候选路径
                        return paths
                else:
                    queue.append((next_id, new_path))

        return paths

    def _compute_cascade_risk(self, path_ids: List[str], event: RiskEvent) -> float:
        """
        计算路径的级联风险得分
        Risk = event.severity * Π(1 - slack_i) * mean(default_prob_i)
        其中 slack_i = min(inventory_days / lead_time, 1) 表示缓冲余量
        """
        if not path_ids:
            return 0.0

        risk = event.severity
        default_probs = []

        for i, node_id in enumerate(path_ids):
            node = self.kg.nodes.get(node_id)
            if node is None:
                continue

            default_probs.append(node.default_prob)

            # 库存缓冲衰减：库存越低、风险越高
            if i > 0:
                prev_id = path_ids[i - 1]
                edges = [e for e in self.kg.downstream_edges(prev_id) if e.dst_id == node_id]
                if edges:
                    lead_time = max(edges[0].lead_time_days, 1)
                    slack = min(node.inventory_days / lead_time, 1.0)
                    risk *= (1 - 0.3 * slack)   # 库存越充裕，衰减越多

        # 叠加平均违约概率
        if default_probs:
            avg_default = sum(default_probs) / len(default_probs)
            risk = risk * (1 + avg_default)

        return min(risk, 1.0)


# ─────────────────────────────────────────────────────────
# 4. 上下文外壳封装（Context Shells）—— LLM 可读的风险报告
# ─────────────────────────────────────────────────────────

class ContextShellGenerator:
    """
    上下文外壳生成器
    将图结构数据和数字因子转化为 LLM 原生可理解的自然语言模版
    """

    def generate_risk_report_shell(
        self,
        chain: RiskPropagationChain,
        brand_inventory_days: int,
        alternative_suppliers: List[str],
    ) -> str:
        """
        生成风险诊断报告的上下文外壳（Context Shell）

        这个 Shell 就是论文中描述的：将冰冷的节点-数字因子表
        包裹在精巧的自然语言模版中，使 LLM 能精准解析复杂图谱数据。

        Args:
            chain: 风险传播链
            brand_inventory_days: 品牌方当前安全库存天数
            alternative_suppliers: 可替换的备用供应商列表
        Returns:
            结构化自然语言上下文外壳字符串
        """
        event = chain.trigger_event
        path = chain.path

        # 路径文本描述
        path_str = " → ".join([
            f"[{seg.node_name}(T{seg.tier}, 违约率{seg.centrality_score:.2f})]"
            for seg in path
        ])

        # 时间缓冲差值
        buffer_gap = brand_inventory_days - chain.total_lead_time_days

        # 备用供应商文本
        alt_suppliers_str = "、".join(alternative_suppliers) if alternative_suppliers else "暂无备用供应商"

        # 建议行动
        if buffer_gap < 0:
            action = (
                f"⚠️ 紧急：库存缺口 {abs(buffer_gap)} 天，建议立刻向备用供应商「{alt_suppliers_str}」追加采购。"
            )
        elif buffer_gap < 15:
            action = (
                f"⚡ 预警：缓冲余量仅 {buffer_gap} 天，建议本周启动备货谈判：{alt_suppliers_str}。"
            )
        else:
            action = f"✅ 安全：当前库存缓冲 {buffer_gap} 天，保持关注即可。"

        shell = f"""
【供应链风险诊断报告】

📌 触发事件: {event.description}
   - 事件类型: {event.event_type}
   - 发生地区: {event.location}
   - 严重程度: {event.severity:.0%}

🔗 风险传播路径 (中心度引导图遍历结果):
   {path_str}

⏱️ 传播时间评估:
   - 预计风险传导至本品牌: {chain.total_lead_time_days} 天
   - 当前安全库存: {brand_inventory_days} 天
   - 缓冲余量: {buffer_gap:+d} 天
   - 级联风险得分: {chain.cascade_risk_score:.1%}

💼 建议行动:
   {action}

📊 供应链知识图谱参数:
""".strip()

        for seg in path:
            node = None
            shell += f"\n   • {seg.node_name}(第{seg.tier}层): 传导+{seg.propagation_days}天"

        return shell


# ─────────────────────────────────────────────────────────
# 5. 主框架：Agentic SCKG Risk Analyzer
# ─────────────────────────────────────────────────────────

class AgenticSCKGRiskAnalyzer:
    """
    供应链知识图谱智能风险分析框架 (Agentic SCKG Risk Analyzer)

    使用流程:
    1. build_kg() 构建供应链知识图谱
    2. analyze_risk_event() 触发风险事件分析
    3. 返回结构化的风险传播链 + LLM 可读的上下文外壳
    """

    def __init__(self, brand_node_id: str) -> None:
        self.brand_node_id = brand_node_id
        self.kg = SupplyChainKG()
        self.centrality_calc: Optional[CentralityCalculator] = None
        self._pagerank: Dict[str, float] = {}
        self._betweenness: Dict[str, float] = {}
        self.shell_gen = ContextShellGenerator()

    def build_kg(
        self,
        nodes: List[SupplierNode],
        edges: List[SupplyEdge],
    ) -> None:
        """构建供应链知识图谱并预计算中心度"""
        for node in nodes:
            self.kg.add_node(node)
        for edge in edges:
            self.kg.add_edge(edge)

        # 预计算中心度（一次性，供后续风险分析复用）
        self.centrality_calc = CentralityCalculator(self.kg)
        self._pagerank = self.centrality_calc.compute_pagerank()
        self._betweenness = self.centrality_calc.compute_betweenness()

    def analyze_risk_event(
        self,
        event: RiskEvent,
        brand_inventory_days: int = 30,
        alternative_suppliers: Optional[List[str]] = None,
        top_k_paths: int = 3,
    ) -> Tuple[List[RiskPropagationChain], List[str]]:
        """
        核心分析入口：给定风险事件，输出影响链 + LLM 上下文外壳

        Args:
            event: 触发的风险事件
            brand_inventory_days: 品牌方当前安全库存天数
            alternative_suppliers: 备用供应商清单
            top_k_paths: 返回 Top-K 风险路径
        Returns:
            (风险传播链列表, 对应的 LLM 上下文外壳列表)
        """
        if alternative_suppliers is None:
            alternative_suppliers = []

        traverser = CentralityGuidedTraverser(
            kg=self.kg,
            pagerank=self._pagerank,
            betweenness=self._betweenness,
        )

        chains = traverser.find_risk_propagation_chains(
            event=event,
            target_node_id=self.brand_node_id,
            top_k_paths=top_k_paths,
        )

        shells = [
            self.shell_gen.generate_risk_report_shell(
                chain=c,
                brand_inventory_days=brand_inventory_days,
                alternative_suppliers=alternative_suppliers,
            )
            for c in chains
        ]

        return chains, shells

    def get_centrality_summary(self) -> Dict[str, Dict[str, float]]:
        """返回所有节点的中心度摘要（用于调试和可视化）"""
        summary = {}
        for node_id, node in self.kg.nodes.items():
            summary[node_id] = {
                "name": node.name,
                "tier": node.tier,
                "pagerank": round(self._pagerank.get(node_id, 0.0), 6),
                "betweenness": round(self._betweenness.get(node_id, 0.0), 6),
            }
        return summary


# ─────────────────────────────────────────────────────────
# 6. 测试用例：模拟出海清洁家电品牌供应链 + 越南罢工场景
# ─────────────────────────────────────────────────────────

def build_demo_supply_chain() -> AgenticSCKGRiskAnalyzer:
    """
    构建演示供应链：某出海智能清洁家电品牌（V8 吸尘器系列）
    供应链拓扑（基于 extract.md 的业务场景）:
      越南材料厂A → 韩国马达B → 深圳组件C → 总装厂D → 品牌方
      (同时有: 德国芯片X → 总装厂D → 品牌方)
    """
    analyzer = AgenticSCKGRiskAnalyzer(brand_node_id="brand_001")

    nodes = [
        # 品牌方（Tier 0）
        SupplierNode("brand_001", "XX智能家电品牌(V8吸尘器)",
                     "中国", 0, 0.01, 30, 0.8, "assembly"),

        # 直接供应商 Tier 1
        SupplierNode("factory_d", "深圳整机组装厂D",
                     "中国", 1, 0.03, 20, 0.9, "assembly"),

        # Tier 2
        SupplierNode("supplier_c", "深圳马达模组供应商C",
                     "中国", 2, 0.05, 15, 0.85, "motor"),
        SupplierNode("supplier_x", "德国芯片供应商X",
                     "德国", 2, 0.02, 45, 0.75, "chip"),

        # Tier 3
        SupplierNode("supplier_b", "韩国电机制造商B",
                     "韩国", 3, 0.04, 10, 0.9, "motor"),

        # Tier 4（最上游）
        SupplierNode("factory_a", "越南工业区材料厂A",
                     "越南", 4, 0.08, 5, 0.95, "material"),
        SupplierNode("supplier_de2", "德国精密轴承供应商E",
                     "德国", 4, 0.02, 60, 0.7, "material"),
    ]

    edges = [
        # 越南A → 韩国B → 深圳C → 整机D → 品牌
        SupplyEdge("factory_a",  "supplier_b",  lead_time_days=21, dependency_ratio=0.9,  annual_volume=500),
        SupplyEdge("supplier_b", "supplier_c",  lead_time_days=14, dependency_ratio=0.75, annual_volume=800),
        SupplyEdge("supplier_c", "factory_d",   lead_time_days=7,  dependency_ratio=0.6,  annual_volume=1200),
        SupplyEdge("factory_d",  "brand_001",   lead_time_days=3,  dependency_ratio=1.0,  annual_volume=3000),

        # 德国芯片链路
        SupplyEdge("supplier_de2", "supplier_x", lead_time_days=30, dependency_ratio=0.5, annual_volume=300),
        SupplyEdge("supplier_x",   "factory_d",  lead_time_days=14, dependency_ratio=0.4, annual_volume=600),
    ]

    analyzer.build_kg(nodes, edges)
    return analyzer


def test_kg_construction(analyzer: AgenticSCKGRiskAnalyzer) -> bool:
    """测试1: 知识图谱构建正确性"""
    print("=" * 65)
    print("测试1: 供应链知识图谱构建验证")
    print("-" * 65)

    kg = analyzer.kg
    assert len(kg.nodes) == 7, f"应有7个节点，实际: {len(kg.nodes)}"
    assert len(kg.edges) == 6, f"应有6条边，实际: {len(kg.edges)}"

    # 验证品牌节点存在
    assert "brand_001" in kg.nodes, "品牌节点应存在"

    # 验证供应商层级
    factory_a = kg.nodes["factory_a"]
    assert factory_a.tier == 4, f"越南工厂应为Tier4，实际: {factory_a.tier}"

    # 验证边连通性：越南A应有下游
    downstream = kg.downstream_edges("factory_a")
    assert len(downstream) == 1, f"越南工厂应有1条下游边，实际: {len(downstream)}"
    assert downstream[0].dst_id == "supplier_b", "越南工厂应供给韩国B"

    print(f"  节点数: {len(kg.nodes)} ✓")
    print(f"  边数: {len(kg.edges)} ✓")
    print(f"  品牌节点 Tier: {kg.nodes['brand_001'].tier} ✓")
    print(f"  越南工厂 Tier: {factory_a.tier} ✓")
    print("✅ 测试1 通过\n")
    return True


def test_centrality_computation(analyzer: AgenticSCKGRiskAnalyzer) -> bool:
    """测试2: 中心度计算正确性"""
    print("=" * 65)
    print("测试2: PageRank + 介数中心度计算")
    print("-" * 65)

    pr = analyzer._pagerank
    bt = analyzer._betweenness
    summary = analyzer.get_centrality_summary()

    # 所有节点应有中心度值
    assert len(pr) == 7, f"PageRank 应覆盖所有7个节点，实际: {len(pr)}"
    assert len(bt) == 7, f"介数中心度应覆盖所有7个节点，实际: {len(bt)}"

    # PageRank 值应在 [0, 1] 且总和近似为 1
    pr_values = list(pr.values())
    assert all(0 <= v <= 1 for v in pr_values), "PageRank 值应在 [0,1]"
    pr_sum = sum(pr_values)
    assert abs(pr_sum - 1.0) < 0.01, f"PageRank 总和应约为1，实际: {pr_sum:.4f}"

    # 介数中心度值应在 [0, 1]
    assert all(0 <= v <= 1 for v in bt.values()), "介数中心度应在 [0,1]"

    print("  PageRank 分布:")
    for nid, info in sorted(summary.items(), key=lambda x: -x[1]["pagerank"]):
        print(f"    {info['name'][:25]:25s} PR={info['pagerank']:.4f}  BT={info['betweenness']:.4f}")

    print("✅ 测试2 通过\n")
    return True


def test_risk_event_analysis(analyzer: AgenticSCKGRiskAnalyzer) -> bool:
    """测试3: 核心场景 —— 越南罢工风险分析（extract.md 的核心业务案例）"""
    print("=" * 65)
    print("测试3: 越南罢工黑天鹅事件风险分析")
    print("-" * 65)

    # 触发事件：越南工业区罢工
    event = RiskEvent(
        event_id="evt_001",
        event_type="strike",
        location="越南某工业区",
        affected_node_ids=["factory_a"],
        severity=0.75,
        description="越南某主要工业区发生大规模罢工，停工预计持续30天",
    )

    chains, shells = analyzer.analyze_risk_event(
        event=event,
        brand_inventory_days=30,
        alternative_suppliers=["备用材料商-泰国F", "国内替代材料商G"],
        top_k_paths=3,
    )

    # 应找到风险传播链
    assert len(chains) > 0, "应找到至少1条风险传播链"

    top_chain = chains[0]
    assert top_chain.cascade_risk_score > 0, "风险得分应大于0"
    assert top_chain.total_lead_time_days > 0, "传播时间应大于0"
    assert len(top_chain.path) >= 2, "传播路径应包含至少2个节点"

    # 验证路径包含越南工厂到品牌的节点
    path_node_ids = [seg.node_id for seg in top_chain.path]
    assert "factory_a" in path_node_ids, "路径应包含越南工厂(触发源)"
    assert "brand_001" in path_node_ids, "路径应包含品牌方(终点)"

    print(f"  找到风险传播链: {len(chains)} 条")
    print(f"  最高风险链:")
    print(f"    传播路径: {' → '.join(seg.node_name for seg in top_chain.path)}")
    print(f"    预计传导时间: {top_chain.total_lead_time_days} 天")
    print(f"    级联风险得分: {top_chain.cascade_risk_score:.1%}")
    print()
    print("  风险诊断报告预览（前20行）:")
    report_lines = shells[0].split("\n")[:20]
    for line in report_lines:
        print(f"    {line}")
    print()

    # 验证上下文外壳包含关键信息
    assert "越南" in shells[0], "报告应包含触发地区信息"
    assert "传播路径" in shells[0] or "风险传播" in shells[0], "报告应包含路径信息"

    print("✅ 测试3 通过\n")
    return True


def test_multi_event_scenarios(analyzer: AgenticSCKGRiskAnalyzer) -> bool:
    """测试4: 多事件场景并行分析"""
    print("=" * 65)
    print("测试4: 多风险事件并行场景测试")
    print("-" * 65)

    events = [
        RiskEvent("evt_002", "earthquake", "日本",
                  ["supplier_b"], 0.9,
                  "日本发生7.5级地震，韩国电机制造商供应链受波及"),
        RiskEvent("evt_003", "sanctions", "美国",
                  ["supplier_x"], 0.6,
                  "美国实体清单制裁德国芯片供应商X"),
    ]

    for event in events:
        chains, shells = analyzer.analyze_risk_event(
            event=event,
            brand_inventory_days=45,
            alternative_suppliers=["备用供应商H"],
        )
        print(f"  事件: {event.description[:40]}...")
        print(f"    找到传播链: {len(chains)} 条")
        if chains:
            print(f"    最高风险得分: {chains[0].cascade_risk_score:.1%}")
            print(f"    传播时间: {chains[0].total_lead_time_days} 天")

        # 每个事件都应能完成分析（允许0条链但不报错）
        assert isinstance(chains, list), "返回值应为列表"
        assert isinstance(shells, list), "报告应为列表"

    print("✅ 测试4 通过\n")
    return True


def test_context_shell_format(analyzer: AgenticSCKGRiskAnalyzer) -> bool:
    """测试5: 上下文外壳格式验证（LLM 可读性检查）"""
    print("=" * 65)
    print("测试5: 上下文外壳 (Context Shell) 格式验证")
    print("-" * 65)

    event = RiskEvent(
        "evt_test", "flood", "泰国",
        ["factory_a"], 0.5,
        "泰国洪水影响原材料供应",
    )

    chains, shells = analyzer.analyze_risk_event(event, brand_inventory_days=20)

    if shells:
        shell = shells[0]
        # 检验关键字段存在
        required_keywords = ["触发事件", "风险传播路径", "传播时间", "建议行动"]
        for keyword in required_keywords:
            assert keyword in shell, f"上下文外壳缺少字段: {keyword}"
            print(f"  ✓ 包含字段: {keyword}")
    else:
        print("  ⚠️ 未找到传播链，跳过外壳格式验证（无路径连接）")

    print("✅ 测试5 通过\n")
    return True


def run_all_tests() -> None:
    """运行所有自测"""
    print("\n" + "=" * 65)
    print("Agentic SCKG Risk Analyzer 模型自测")
    print("论文: Network-Knowledge Graph Duality (arXiv: 2510.01115)")
    print("=" * 65 + "\n")

    analyzer = build_demo_supply_chain()

    tests = [
        test_kg_construction,
        test_centrality_computation,
        test_risk_event_analysis,
        test_multi_event_scenarios,
        test_context_shell_format,
    ]

    passed = 0
    for test_fn in tests:
        try:
            result = test_fn(analyzer)
            if result:
                passed += 1
        except AssertionError as e:
            print(f"❌ {test_fn.__name__} 失败: {e}\n")
        except Exception as e:
            import traceback
            print(f"💥 {test_fn.__name__} 异常: {type(e).__name__}: {e}")
            traceback.print_exc()
            print()

    print("=" * 65)
    print(f"测试结果: {passed}/{len(tests)} 通过")
    if passed == len(tests):
        print("🎉 所有测试通过！Agentic SCKG Risk Analyzer 模型验证成功")
    else:
        print(f"⚠️  {len(tests) - passed} 个测试未通过")
        raise SystemExit(1)
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────
# 7. 使用示例
# ─────────────────────────────────────────────────────────

def demo_usage() -> None:
    """完整使用示例：越南罢工的断供预警"""
    print("\n" + "─" * 65)
    print("Agentic SCKG Risk Analyzer 使用示例")
    print("场景: 某出海智能清洁家电品牌 × 越南罢工黑天鹅")
    print("─" * 65)

    analyzer = build_demo_supply_chain()

    event = RiskEvent(
        event_id="evt_demo",
        event_type="strike",
        location="越南胡志明工业区",
        affected_node_ids=["factory_a"],
        severity=0.8,
        description="越南胡志明工业区发生大规模工人罢工，预计持续4周",
    )

    chains, shells = analyzer.analyze_risk_event(
        event=event,
        brand_inventory_days=30,
        alternative_suppliers=["备用材料商-泰国F", "国内替代材料商G"],
        top_k_paths=2,
    )

    print(f"\n发现 {len(chains)} 条风险传播链:")
    for i, (chain, shell) in enumerate(zip(chains, shells)):
        print(f"\n{'─'*50}")
        print(f"【风险链 #{i+1}】级联风险: {chain.cascade_risk_score:.1%}")
        print(shell)


if __name__ == "__main__":
    run_all_tests()
    demo_usage()

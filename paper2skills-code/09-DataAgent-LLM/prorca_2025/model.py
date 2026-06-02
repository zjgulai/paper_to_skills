"""
ProRCA: Causal Root Cause Analysis for Business Anomalies
基于因果图的条件异常打分与深度路径溯源

论文: ProRCA: A Causal Python Package for Actionable Root Cause Analysis
      in Real-world Business Scenarios
arXiv: 2503.01475 (2025-03)

核心算法：
1. 构建业务指标因果依赖图 (DAG)
2. 条件异常打分 (Conditional Anomaly Scoring) -- 只在父节点状态给定后仍然异常才报警
3. 深度优先路径追踪 (DFS Path Exploration) -- 逆着因果图溯源到真正的根因

设计目标：模拟出海电商 GMV 暴跌场景的秒级自动溯源。
"""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class MetricNode:
    """业务指标节点"""
    name: str
    # 当前观测值（标准化 z-score）
    z_score: float
    # 原始值（可选，用于报告）
    raw_value: Optional[float] = None
    raw_baseline: Optional[float] = None

    @property
    def is_anomalous(self) -> bool:
        """绝对异常：|z| > 2（未考虑父节点状态）"""
        return abs(self.z_score) > 2.0


@dataclass
class RCAResult:
    """根因分析结果"""
    root_cause: str
    path: List[str]
    conditional_scores: Dict[str, float]
    final_score: float
    explanation: str


# ---------------------------------------------------------------------------
# 因果图
# ---------------------------------------------------------------------------

class CausalGraph:
    """
    有向无环图 (DAG) — 表示业务指标的因果依赖关系。
    边方向: 父节点 → 子节点 (原因 → 结果)
    """

    def __init__(self) -> None:
        # node_name -> MetricNode
        self._nodes: Dict[str, MetricNode] = {}
        # child -> list of parents
        self._parents: Dict[str, List[str]] = {}
        # parent -> list of children
        self._children: Dict[str, List[str]] = {}

    def add_node(self, node: MetricNode) -> None:
        self._nodes[node.name] = node
        self._parents.setdefault(node.name, [])
        self._children.setdefault(node.name, [])

    def add_edge(self, parent: str, child: str) -> None:
        """parent 是 child 的原因"""
        if parent not in self._nodes:
            raise ValueError(f"节点不存在: {parent}")
        if child not in self._nodes:
            raise ValueError(f"节点不存在: {child}")
        self._parents[child].append(parent)
        self._children[parent].append(child)

    def get_parents(self, name: str) -> List[str]:
        return self._parents.get(name, [])

    def get_children(self, name: str) -> List[str]:
        return self._children.get(name, [])

    def get_node(self, name: str) -> MetricNode:
        return self._nodes[name]

    def all_nodes(self) -> List[str]:
        return list(self._nodes.keys())


# ---------------------------------------------------------------------------
# 条件异常打分器
# ---------------------------------------------------------------------------

class ConditionalAnomalyScorer:
    """
    条件异常打分 (Conditional Anomaly Scoring)

    核心思想：
        如果上游（父节点）已经异常，下游跟着异常是"正常的传播"，不应该二次报警。
        只有在给定父节点当前状态后，节点的残差依然超出期望，才给高分。

    简化实现：
        noise_i = z_i - beta * avg(z_parents)
        conditional_score = |noise_i|

    其中 beta 是传导系数（默认 0.7），表示上游对下游的平均影响强度。
    """

    def __init__(self, propagation_beta: float = 0.7) -> None:
        self.beta = propagation_beta

    def score(self, node_name: str, graph: CausalGraph) -> float:
        """计算节点的条件异常分数（=去除父节点影响后的残差 |noise|）"""
        node = graph.get_node(node_name)
        parents = graph.get_parents(node_name)

        if not parents:
            # 根节点：无父节点，直接用绝对 z-score
            return abs(node.z_score)

        parent_avg_z = sum(
            graph.get_node(p).z_score for p in parents
        ) / len(parents)

        # 去除父节点的"正常传导"影响，只看残差
        residual = node.z_score - self.beta * parent_avg_z
        return abs(residual)

    def score_all(self, graph: CausalGraph) -> Dict[str, float]:
        """对图中所有节点打分"""
        return {name: self.score(name, graph) for name in graph.all_nodes()}


# ---------------------------------------------------------------------------
# 根因溯源追踪器
# ---------------------------------------------------------------------------

class RootCauseTracer:
    """
    深度优先路径追踪 (DFS Root Cause Tracing)

    从出问题的最终节点出发，逆着因果边向上追溯，
    沿着条件异常分数最高的父节点方向递归，
    直到找到"没有异常父节点"或"条件分数最高"的叶节点。
    """

    def __init__(
        self,
        scorer: ConditionalAnomalyScorer,
        min_score_threshold: float = 0.5,
        max_depth: int = 10,
    ) -> None:
        self.scorer = scorer
        self.threshold = min_score_threshold
        self.max_depth = max_depth

    def trace(
        self,
        start_node: str,
        graph: CausalGraph,
        conditional_scores: Optional[Dict[str, float]] = None,
    ) -> RCAResult:
        """
        从 start_node 开始，逆向追踪到根因节点。

        Returns:
            RCAResult 包含根因节点名称、完整路径、打分详情
        """
        if conditional_scores is None:
            conditional_scores = self.scorer.score_all(graph)

        path: List[str] = [start_node]
        current = start_node
        visited = {start_node}

        for _ in range(self.max_depth):
            parents = graph.get_parents(current)
            if not parents:
                break

            anomalous_parents = [
                p for p in parents
                if conditional_scores.get(p, 0) >= self.threshold and p not in visited
            ]

            if not anomalous_parents:
                break

            best_parent = max(anomalous_parents, key=lambda p: conditional_scores[p])
            current_score = conditional_scores.get(current, 0)
            best_parent_score = conditional_scores[best_parent]

            if best_parent_score <= current_score:
                break

            path.append(best_parent)
            visited.add(best_parent)
            current = best_parent

        root_cause = path[-1]
        root_node = graph.get_node(root_cause)
        final_score = conditional_scores.get(root_cause, 0)

        explanation = _build_explanation(path, graph, conditional_scores)

        return RCAResult(
            root_cause=root_cause,
            path=path,
            conditional_scores=conditional_scores,
            final_score=final_score,
            explanation=explanation,
        )


def _build_explanation(
    path: List[str],
    graph: CausalGraph,
    scores: Dict[str, float],
) -> str:
    """构建可读的因果链说明"""
    if len(path) == 1:
        node = graph.get_node(path[0])
        return (
            f"根因节点 [{path[0]}] 条件异常分数={scores[path[0]]:.2f}，"
            f"z-score={node.z_score:.2f}，无上游传导，直接异常。"
        )

    parts = []
    for i in range(len(path) - 1, -1, -1):
        name = path[i]
        node = graph.get_node(name)
        score = scores[name]
        if i == len(path) - 1:
            parts.append(f"【根因】{name}(z={node.z_score:.2f}, score={score:.2f})")
        elif i == 0:
            parts.append(f"【影响终点】{name}(z={node.z_score:.2f}, score={score:.2f})")
        else:
            parts.append(f"{name}(z={node.z_score:.2f}, score={score:.2f})")

    return " → ".join(parts)


# ---------------------------------------------------------------------------
# 高层 API：ProRCA Engine
# ---------------------------------------------------------------------------

class ProRCAEngine:
    """
    ProRCA 根因分析引擎 — 端到端 API

    Usage:
        engine = ProRCAEngine()
        engine.load_graph(nodes, edges)
        result = engine.analyze(trigger_node="GMV")
        print(result.explanation)
    """

    def __init__(
        self,
        propagation_beta: float = 0.7,
        score_threshold: float = 0.5,
        max_depth: int = 10,
    ) -> None:
        self.scorer = ConditionalAnomalyScorer(propagation_beta=propagation_beta)
        self.tracer = RootCauseTracer(
            scorer=self.scorer,
            min_score_threshold=score_threshold,
            max_depth=max_depth,
        )
        self.graph: Optional[CausalGraph] = None

    def load_graph(
        self,
        nodes: List[Dict],
        edges: List[Tuple[str, str]],
    ) -> None:
        """
        加载因果图。

        Args:
            nodes: [{"name": ..., "z_score": ..., "raw_value": ..., "raw_baseline": ...}, ...]
            edges: [("parent", "child"), ...]
        """
        g = CausalGraph()
        for n in nodes:
            g.add_node(MetricNode(
                name=n["name"],
                z_score=n["z_score"],
                raw_value=n.get("raw_value"),
                raw_baseline=n.get("raw_baseline"),
            ))
        for parent, child in edges:
            g.add_edge(parent, child)
        self.graph = g

    def analyze(self, trigger_node: str) -> RCAResult:
        """
        从 trigger_node（最终出问题的指标）开始溯源。

        Returns:
            RCAResult
        """
        if self.graph is None:
            raise RuntimeError("请先调用 load_graph() 加载因果图。")
        conditional_scores = self.scorer.score_all(self.graph)
        return self.tracer.trace(trigger_node, self.graph, conditional_scores)

    def summary(self, result: RCAResult) -> str:
        """输出给 LLM/报告系统的文本摘要"""
        lines = [
            "=" * 60,
            "ProRCA 根因分析报告",
            "=" * 60,
            f"根因节点   : {result.root_cause}",
            f"追踪路径   : {' → '.join(result.path)}",
            f"根因分数   : {result.final_score:.3f}",
            "",
            "因果链解释:",
            f"  {result.explanation}",
            "",
            "各节点条件异常分数 (Top 5):",
        ]
        top5 = sorted(
            result.conditional_scores.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for name, score in top5:
            node = self.graph.get_node(name)
            lines.append(f"  {name:30s}  score={score:.3f}  z={node.z_score:.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

def _make_ecommerce_graph() -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """
    模拟出海电商黑五大促 GMV 暴跌场景的因果图。

    指标因果链：
        广告流量 → 加购量 → 结账到达量 → PayPal支付成功率 → 最终GMV
        广告流量 → 加购量 → 结账到达量 → 信用卡支付成功率 → 最终GMV

    异常注入：PayPal 支付节点 502 错误，z-score = -4.5（极度异常）。
    其余节点因 PayPal 崩溃被连带影响，z-score 轻度负偏。
    """
    nodes = [
        # 真正的根因：PayPal 支付网关故障
        {"name": "PayPal支付成功率",  "z_score": -4.5,  "raw_value": 0.32, "raw_baseline": 0.97},
        # 被连带：结账到达量轻微下降（用户看到失败会放弃）
        {"name": "结账到达量",         "z_score": -2.1,  "raw_value": 1850, "raw_baseline": 2100},
        # 信用卡正常
        {"name": "信用卡支付成功率",   "z_score":  0.1,  "raw_value": 0.96, "raw_baseline": 0.97},
        # 加购量正常
        {"name": "加购量",             "z_score": -0.3,  "raw_value": 5100, "raw_baseline": 5200},
        # 广告流量正常
        {"name": "广告流量",           "z_score":  0.2,  "raw_value": 28000, "raw_baseline": 27500},
        # 最终 GMV 暴跌（触发节点）
        {"name": "GMV",                "z_score": -3.8,  "raw_value": 45000, "raw_baseline": 75000},
    ]
    edges = [
        ("广告流量",           "加购量"),
        ("加购量",             "结账到达量"),
        ("结账到达量",         "PayPal支付成功率"),
        ("结账到达量",         "信用卡支付成功率"),
        ("PayPal支付成功率",   "GMV"),
        ("信用卡支付成功率",   "GMV"),
    ]
    return nodes, edges


def test_conditional_anomaly_scoring() -> None:
    """测试 1：条件异常打分——被连带的节点分数应低于真正的根因"""
    print("\n[TEST 1] 条件异常打分")
    nodes, edges = _make_ecommerce_graph()
    g = CausalGraph()
    for n in nodes:
        g.add_node(MetricNode(**{k: v for k, v in n.items() if k in ("name", "z_score", "raw_value", "raw_baseline")}))
    for parent, child in edges:
        g.add_edge(parent, child)

    scorer = ConditionalAnomalyScorer(propagation_beta=0.7)
    scores = scorer.score_all(g)

    print("  节点条件异常分数：")
    for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:30s}  score={score:.3f}")

    # 断言：PayPal 的条件分数必须最高（它是根因，残差最大）
    assert scores["PayPal支付成功率"] == max(scores.values()), \
        f"期望 PayPal 分数最高，实际: {scores}"

    # 断言：广告流量正常，分数应 < 1
    assert scores["广告流量"] < 1.0, \
        f"期望广告流量分数 < 1，实际: {scores['广告流量']}"

    print("  ✅ 条件异常打分测试通过")


def test_root_cause_tracing() -> None:
    """测试 2：深度路径追踪——从 GMV 出发应该追溯到 PayPal"""
    print("\n[TEST 2] 根因路径追踪")
    nodes, edges = _make_ecommerce_graph()
    engine = ProRCAEngine(propagation_beta=0.7, score_threshold=0.5, max_depth=10)
    engine.load_graph(nodes, edges)

    result = engine.analyze(trigger_node="GMV")

    print(f"  追踪路径: {' → '.join(result.path)}")
    print(f"  根因: {result.root_cause}  (score={result.final_score:.3f})")

    # 断言：根因必须是 PayPal
    assert result.root_cause == "PayPal支付成功率", \
        f"期望根因为 PayPal支付成功率，实际: {result.root_cause}"

    # 断言：路径必须从 GMV 开始
    assert result.path[0] == "GMV", \
        f"期望路径起点为 GMV，实际: {result.path[0]}"

    print("  ✅ 根因追踪测试通过")


def test_no_anomaly_scenario() -> None:
    """测试 3：无异常场景——所有指标正常时，根因分数应很低"""
    print("\n[TEST 3] 无异常场景")
    nodes = [
        {"name": "广告流量",  "z_score": 0.1},
        {"name": "加购量",    "z_score": 0.2},
        {"name": "GMV",       "z_score": 0.3},
    ]
    edges = [("广告流量", "加购量"), ("加购量", "GMV")]

    engine = ProRCAEngine(propagation_beta=0.7, score_threshold=0.5)
    engine.load_graph(nodes, edges)

    result = engine.analyze(trigger_node="GMV")

    # 无异常时路径应该很短（当前节点即为终止），根因分数应 < 2
    assert result.final_score < 2.0, \
        f"无异常场景期望分数 < 2，实际: {result.final_score}"

    print(f"  根因: {result.root_cause}  分数: {result.final_score:.3f}")
    print("  ✅ 无异常场景测试通过")


def test_multi_hop_chain() -> None:
    """测试 4：多跳链路溯源——A→B→C→D→E，根因在 A"""
    print("\n[TEST 4] 多跳因果链溯源")
    nodes = [
        {"name": "A", "z_score": -5.0},
        {"name": "B", "z_score": -4.2},
        {"name": "C", "z_score": -3.5},
        {"name": "D", "z_score": -2.9},
        {"name": "E", "z_score": -2.4},
    ]
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]

    engine = ProRCAEngine(propagation_beta=0.7, score_threshold=0.1)
    engine.load_graph(nodes, edges)
    result = engine.analyze(trigger_node="E")

    print(f"  追踪路径: {' → '.join(result.path)}")
    print(f"  根因: {result.root_cause}  (score={result.final_score:.3f})")

    # A 是根节点（无父节点），路径必须追到 A
    assert result.root_cause == "A", \
        f"期望多跳根因为 A，实际: {result.root_cause}"

    assert len(result.path) >= 2, \
        f"期望路径长度 >= 2，实际: {len(result.path)}"

    print("  ✅ 多跳链路溯源测试通过")


def run_full_demo() -> None:
    """完整演示：出海电商黑五 GMV 暴跌溯源"""
    print("\n" + "=" * 60)
    print("ProRCA 演示：出海电商黑五大促 GMV 暴跌 15% 自动溯源")
    print("=" * 60)

    nodes, edges = _make_ecommerce_graph()
    engine = ProRCAEngine(propagation_beta=0.7, score_threshold=0.5)
    engine.load_graph(nodes, edges)

    result = engine.analyze(trigger_node="GMV")
    print(engine.summary(result))

    # LLM Agent 推送文案示例
    root_node = engine.graph.get_node(result.root_cause)
    drop_pct = (1 - root_node.raw_value / root_node.raw_baseline) * 100 if root_node.raw_baseline else 0
    print(
        f"\n【LLM Agent 推送文案示例】\n"
        f"【紧急】GMV 暴跌根因已定位：\n"
        f"因果追踪路径: {' → '.join(reversed(result.path))}\n"
        f"根因: 北美区 {result.root_cause} 出现异常（当前 {root_node.raw_value:.0%} vs 基线 {root_node.raw_baseline:.0%}，"
        f"下降 {drop_pct:.0f}%），\n"
        f"连带引发结账成功率下降，最终导致大盘 GMV 暴跌。\n"
        f"其余系统（广告流量、加购量、信用卡通道）运转正常。\n"
        f"请立即通知支付运维工程师！"
    )


if __name__ == "__main__":
    print("ProRCA 自测开始...")
    test_conditional_anomaly_scoring()
    test_root_cause_tracing()
    test_no_anomaly_scenario()
    test_multi_hop_chain()
    run_full_demo()
    print("\n✅ 所有测试通过！")

"""AdaptOrch: 任务自适应多智能体编排拓扑路由框架.

参考论文:Yu, G. (2026) AdaptOrch: Task-Adaptive Multi-Agent Orchestration.
arxiv:2602.16873.

本实现简化版,演示:
- TaskDAG + DependencyEdge + Subtask 数据结构
- DAGAnalyzer: 计算 ω(parallelism width), δ(critical path), γ(coupling density)
- TopologyRouter (Algorithm 1): O(|V|+|E|) 路由
- 4 种 Executor: Parallel / Sequential / Hierarchical / Hybrid
- AdaptiveSynthesizer (Algorithm 2): consistency + arbitration + re-routing
- 母婴客服工单 demo

生产环境:
- Decomposer 接 Claude/GPT LLM
- Executor 接 MCP/A2A 协议栈(P1-4)
- Embedding 用 text-embedding-3-small
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Topology enum ----------------------------------------------------------


class Topology(Enum):
    PARALLEL = "parallel"       # τ_P
    SEQUENTIAL = "sequential"   # τ_S
    HIERARCHICAL = "hierarchical"  # τ_H
    HYBRID = "hybrid"           # τ_X


# DAG data structures ----------------------------------------------------


@dataclass
class Subtask:
    """DAG 节点 v_i."""

    subtask_id: str
    description: str
    est_tokens: int = 100  # w(v): 估算 token 成本


@dataclass
class DependencyEdge:
    """DAG 边 (u, v), 带 coupling strength c(u,v)."""

    from_id: str
    to_id: str
    coupling: float = 0.0  # 0.0=none, 0.3=weak, 0.7=strong, 1.0=critical


@dataclass
class TaskDAG:
    """任务依赖 DAG: G_T = (V, E, w, c)."""

    subtasks: dict[str, Subtask] = field(default_factory=dict)
    edges: list[DependencyEdge] = field(default_factory=list)

    def add_subtask(self, subtask_id: str, description: str, est_tokens: int = 100) -> Subtask:
        st = Subtask(subtask_id=subtask_id, description=description, est_tokens=est_tokens)
        self.subtasks[subtask_id] = st
        return st

    def add_edge(self, from_id: str, to_id: str, coupling: float = 0.0) -> None:
        self.edges.append(DependencyEdge(from_id=from_id, to_id=to_id, coupling=coupling))

    def adjacency(self) -> dict[str, list[str]]:
        """出边邻接表."""
        adj: dict[str, list[str]] = {sid: [] for sid in self.subtasks}
        for e in self.edges:
            adj[e.from_id].append(e.to_id)
        return adj

    def reverse_adjacency(self) -> dict[str, list[str]]:
        """入边邻接表."""
        rev: dict[str, list[str]] = {sid: [] for sid in self.subtasks}
        for e in self.edges:
            rev[e.to_id].append(e.from_id)
        return rev

    def topological_order(self) -> list[str]:
        """Kahn 算法拓扑排序, 顺便检测环."""
        in_deg = {sid: 0 for sid in self.subtasks}
        for e in self.edges:
            in_deg[e.to_id] += 1
        q = deque([sid for sid, d in in_deg.items() if d == 0])
        order: list[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for e in self.edges:
                if e.from_id == u:
                    in_deg[e.to_id] -= 1
                    if in_deg[e.to_id] == 0:
                        q.append(e.to_id)
        if len(order) != len(self.subtasks):
            raise ValueError("Cycle detected in DAG")
        return order


# DAG Analyzer -----------------------------------------------------------


@dataclass
class DAGProperties:
    """Definition 3: DAG 结构属性."""

    omega: int          # parallelism width (最大反链大小)
    delta: int          # critical path depth
    gamma: float        # coupling density
    parallel_ratio: float  # r = ω / |V|


class DAGAnalyzer:
    """分析 DAG 结构属性 (论文 Definition 3)."""

    def analyze(self, dag: TaskDAG) -> DAGProperties:
        omega = self._parallelism_width(dag)
        delta = self._critical_path_depth(dag)
        gamma = self._coupling_density(dag)
        n = len(dag.subtasks)
        return DAGProperties(
            omega=omega,
            delta=delta,
            gamma=gamma,
            parallel_ratio=omega / max(1, n),
        )

    def _parallelism_width(self, dag: TaskDAG) -> int:
        """近似 ω: 拓扑分层后最大层宽 (Dilworth 定理: 层宽 ≤ ω).

        精确 ω 需匹配算法 O(|V|^2.5), 这里用近似 O(|V|+|E|).
        """
        adj = dag.adjacency()
        in_deg = {sid: 0 for sid in dag.subtasks}
        for e in dag.edges:
            in_deg[e.to_id] += 1

        # Kahn + 分层
        layer: dict[str, int] = {}
        q = deque()
        for sid, d in in_deg.items():
            if d == 0:
                q.append(sid)
                layer[sid] = 0

        processed = 0
        while q:
            u = q.popleft()
            processed += 1
            for v in adj[u]:
                layer[v] = max(layer.get(v, 0), layer[u] + 1)
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)

        if processed != len(dag.subtasks):
            raise ValueError("Cycle in DAG")

        layer_counts: dict[int, int] = defaultdict(int)
        for sid, l in layer.items():
            layer_counts[l] += 1
        return max(layer_counts.values()) if layer_counts else 1

    def _critical_path_depth(self, dag: TaskDAG) -> int:
        """最长路径(按节点数)."""
        order = dag.topological_order()
        dist = {sid: 1 for sid in dag.subtasks}  # 每个节点自身算 1
        rev = dag.reverse_adjacency()
        for sid in order:
            for pred in rev[sid]:
                dist[sid] = max(dist[sid], dist[pred] + 1)
        return max(dist.values()) if dist else 0

    def _coupling_density(self, dag: TaskDAG) -> float:
        """γ = Σc(u,v) / |E|."""
        if not dag.edges:
            return 0.0
        return sum(e.coupling for e in dag.edges) / len(dag.edges)


# Topology Router (Algorithm 1) ------------------------------------------


@dataclass
class RouterThresholds:
    """论文默认阈值."""

    theta_omega: float = 0.5   # 至少一半子任务可并行
    theta_gamma: float = 0.6   # 高耦合阈值
    theta_delta: int = 5       # 分层最小子任务数


class TopologyRouter:
    """Algorithm 1: O(|V|+|E|) 拓扑路由."""

    def __init__(self, thresholds: Optional[RouterThresholds] = None) -> None:
        self.thresholds = thresholds or RouterThresholds()

    def route(self, dag: TaskDAG) -> Topology:
        props = DAGAnalyzer().analyze(dag)
        n = len(dag.subtasks)
        m = len(dag.edges)

        # Line 3: 全独立
        if m == 0:
            return Topology.PARALLEL

        # Line 4: 全串行(无并行性)
        if props.omega == 1:
            return Topology.SEQUENTIAL

        # Line 5: 高耦合 + 多子任务 → 分层
        if props.gamma > self.thresholds.theta_gamma and n > self.thresholds.theta_delta:
            return Topology.HIERARCHICAL

        # Line 6: 宽 DAG + 低耦合 → 并行
        if props.parallel_ratio > self.thresholds.theta_omega and props.gamma <= self.thresholds.theta_gamma:
            return Topology.PARALLEL

        # Line 7-9: 否则混合
        return Topology.HYBRID

    def route_with_reason(self, dag: TaskDAG) -> tuple[Topology, str]:
        """返回拓扑 + 路由理由."""
        topo = self.route(dag)
        props = DAGAnalyzer().analyze(dag)
        reasons = {
            Topology.PARALLEL: f"宽DAG(r={props.parallel_ratio:.2f}),低耦合(γ={props.gamma:.2f})",
            Topology.SEQUENTIAL: f"全串行(ω={props.omega})",
            Topology.HIERARCHICAL: f"高耦合(γ={props.gamma:.2f})+多子任务({len(dag.subtasks)})",
            Topology.HYBRID: f"复杂DAG:ω={props.omega},r={props.parallel_ratio:.2f},γ={props.gamma:.2f}",
        }
        return topo, reasons[topo]


# Executors --------------------------------------------------------------


@dataclass
class ExecutionResult:
    """子任务执行结果."""

    subtask_id: str
    output: str


class ParallelExecutor:
    """τ_P: 所有子任务并发执行."""

    def execute(self, dag: TaskDAG) -> list[ExecutionResult]:
        results: list[ExecutionResult] = []
        for sid, st in dag.subtasks.items():
            # mock: 并行执行
            results.append(ExecutionResult(
                subtask_id=sid,
                output=f"[PARALLEL] {st.description} done",
            ))
        return results


class SequentialExecutor:
    """τ_S: 拓扑序串行, 累积上下文."""

    def execute(self, dag: TaskDAG) -> list[ExecutionResult]:
        order = dag.topological_order()
        results: list[ExecutionResult] = []
        context = []
        for sid in order:
            st = dag.subtasks[sid]
            # mock: 串行, 带上下文
            ctx = "; ".join(context)
            results.append(ExecutionResult(
                subtask_id=sid,
                output=f"[SEQUENTIAL] {st.description} (ctx: {ctx[:40]}...)",
            ))
            context.append(f"{sid} done")
        return results


class HierarchicalExecutor:
    """τ_H: Lead agent 分解+委派+仲裁."""

    def execute(self, dag: TaskDAG) -> list[ExecutionResult]:
        # mock: lead 先 "decompose", 然后 delegate 每个子任务
        results: list[ExecutionResult] = []
        results.append(ExecutionResult(
            subtask_id="lead",
            output=f"[HIERARCHY] Lead decomposed {len(dag.subtasks)} subtasks",
        ))
        for sid, st in dag.subtasks.items():
            results.append(ExecutionResult(
                subtask_id=sid,
                output=f"[HIERARCHY-SUB] {st.description} by sub-agent",
            ))
        results.append(ExecutionResult(
            subtask_id="lead_reconcile",
            output="[HIERARCHY] Lead reconciled all sub-agent outputs",
        ))
        return results


class HybridExecutor:
    """τ_X: 拓扑分层, 层内并行, 层间串行."""

    def _topological_layers(self, dag: TaskDAG) -> list[list[str]]:
        """返回拓扑分层."""
        in_deg = {sid: 0 for sid in dag.subtasks}
        for e in dag.edges:
            in_deg[e.to_id] += 1
        adj = dag.adjacency()

        layers: list[list[str]] = []
        remaining = set(dag.subtasks.keys())
        while remaining:
            layer = [sid for sid in remaining if in_deg[sid] == 0]
            if not layer:
                raise ValueError("Cycle detected")
            layers.append(layer)
            for sid in layer:
                remaining.remove(sid)
                for v in adj[sid]:
                    in_deg[v] -= 1
        return layers

    def execute(self, dag: TaskDAG) -> list[ExecutionResult]:
        layers = self._topological_layers(dag)
        results: list[ExecutionResult] = []
        for li, layer in enumerate(layers):
            # 层内并行
            for sid in layer:
                st = dag.subtasks[sid]
                results.append(ExecutionResult(
                    subtask_id=sid,
                    output=f"[HYBRID-L{li}] {st.description}",
                ))
        return results


# Synthesis --------------------------------------------------------------


class ConsistencyScore:
    """Definition 5: heuristic consistency score.

    简化版: 用字符串 overlap 代理 embedding cosine similarity.
    """

    def compute(self, outputs: list[str]) -> float:
        if len(outputs) < 2:
            return 1.0
        total = 0.0
        count = 0
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = self._string_similarity(outputs[i], outputs[j])
                total += sim
                count += 1
        return total / max(1, count)

    def _string_similarity(self, a: str, b: str) -> float:
        """简化: Jaccard similarity on words."""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / max(1, union)


class AdaptiveSynthesizer:
    """Algorithm 2: 自适应合成协议."""

    def __init__(self, theta_cs: float = 0.5) -> None:
        self.theta_cs = theta_cs
        self.max_retries = 5
        self.gamma_increment = 0.2

    def synthesize(
        self,
        topology: Topology,
        results: list[ExecutionResult],
        dag: TaskDAG,
    ) -> str:
        outputs = [r.output for r in results]

        # Sequential: 直接取最后
        if topology == Topology.SEQUENTIAL:
            return outputs[-1] if outputs else ""

        # 计算 consistency
        cs = ConsistencyScore().compute(outputs)

        if cs >= self.theta_cs:
            # 一致: merge
            return f"[MERGE] Consistent({cs:.2f}): " + " | ".join(outputs[:3])

        # 不一致: arbitration + re-routing
        arbiter_output = f"[ARBITER] Resolved conflicts among {len(outputs)} outputs (CS={cs:.2f})"

        # 模拟重路由(实际应更新 dag coupling 后重新路由)
        retry = 0
        current_gamma = self._estimate_gamma(dag)
        while cs < self.theta_cs and retry < self.max_retries:
            current_gamma += self.gamma_increment
            if current_gamma > 0.6:
                # 强制转 hierarchical
                return f"[REROUTE→HIERARCHY] After {retry + 1} retries, γ={current_gamma:.2f} > θ_γ. {arbiter_output}"
            retry += 1

        return arbiter_output

    def _estimate_gamma(self, dag: TaskDAG) -> float:
        if not dag.edges:
            return 0.0
        return sum(e.coupling for e in dag.edges) / len(dag.edges)


# Main orchestrator ------------------------------------------------------


@dataclass
class AdaptOrch:
    """完整 AdaptOrch pipeline (论文 Figure 2)."""

    router: TopologyRouter = field(default_factory=TopologyRouter)
    synthesizer: AdaptiveSynthesizer = field(default_factory=AdaptiveSynthesizer)

    def run(self, dag: TaskDAG) -> dict[str, Any]:
        # Phase 3: 路由
        topo, reason = self.router.route_with_reason(dag)

        # Phase 4: 执行
        executors = {
            Topology.PARALLEL: ParallelExecutor(),
            Topology.SEQUENTIAL: SequentialExecutor(),
            Topology.HIERARCHICAL: HierarchicalExecutor(),
            Topology.HYBRID: HybridExecutor(),
        }
        executor = executors[topo]
        results = executor.execute(dag)

        # Phase 5: 合成
        final_output = self.synthesizer.synthesize(topo, results, dag)

        return {
            "topology": topo.value,
            "reason": reason,
            "subtask_count": len(dag.subtasks),
            "edge_count": len(dag.edges),
            "results": [(r.subtask_id, r.output[:60]) for r in results],
            "final_output": final_output,
        }


# Demo -------------------------------------------------------------------


def _build_customer_service_dag() -> TaskDAG:
    """构建母婴客服工单 DAG (过敏退货+物流+关税+替代品)."""
    dag = TaskDAG()
    dag.add_subtask("v1", "过敏症状分类", 50)
    dag.add_subtask("v2", "订单状态查询", 50)
    dag.add_subtask("v3", "合规判定(CN/US)", 80)
    dag.add_subtask("v4", "退款流程初始化", 60)
    dag.add_subtask("v5", "物流拦截申请", 70)
    dag.add_subtask("v6", "替代品推荐", 40)

    dag.add_edge("v2", "v3", coupling=0.7)   # strong
    dag.add_edge("v3", "v4", coupling=0.5)   # medium (降低使 γ≤0.6)
    dag.add_edge("v2", "v5", coupling=0.7)   # strong
    dag.add_edge("v4", "v5", coupling=0.7)   # strong
    dag.add_edge("v1", "v6", coupling=0.3)   # weak
    return dag


def main() -> None:
    print("=== AdaptOrch Demo:跨境客服工单拓扑路由 ===\n")

    dag = _build_customer_service_dag()
    analyzer = DAGAnalyzer()
    props = analyzer.analyze(dag)
    print(f"DAG 属性:")
    print(f"  子任务数 |V| = {len(dag.subtasks)}")
    print(f"  边数 |E| = {len(dag.edges)}")
    print(f"  Parallelism width ω = {props.omega}")
    print(f"  Critical path depth δ = {props.delta}")
    print(f"  Coupling density γ = {props.gamma:.2f}")
    print(f"  Parallel ratio r = {props.parallel_ratio:.2f}\n")

    orch = AdaptOrch()
    result = orch.run(dag)

    print(f"路由结果: {result['topology']}")
    print(f"路由理由: {result['reason']}")
    print(f"\n子任务执行:")
    for sid, out in result["results"]:
        print(f"  {sid}: {out}")
    print(f"\n最终合成: {result['final_output']}\n")

    # 对比 4 种拓扑
    print("--- 四种拓扑对比 ---")
    for topo_name, topo in [
        ("Parallel", Topology.PARALLEL),
        ("Sequential", Topology.SEQUENTIAL),
        ("Hierarchical", Topology.HIERARCHICAL),
        ("Hybrid", Topology.HYBRID),
    ]:
        # 强制使用指定拓扑执行
        executors = {
            Topology.PARALLEL: ParallelExecutor(),
            Topology.SEQUENTIAL: SequentialExecutor(),
            Topology.HIERARCHICAL: HierarchicalExecutor(),
            Topology.HYBRID: HybridExecutor(),
        }
        exec_results = executors[topo].execute(dag)
        print(f"  {topo_name}: {len(exec_results)} execution steps")


def test_pipeline() -> None:
    """Sanity checks."""

    # 1) 空 DAG → parallel (|E|=0)
    empty_dag = TaskDAG()
    empty_dag.add_subtask("a", "task a")
    empty_dag.add_subtask("b", "task b")
    router = TopologyRouter()
    assert router.route(empty_dag) == Topology.PARALLEL, "无依赖应 parallel"

    # 2) 链式 DAG → sequential
    chain = TaskDAG()
    chain.add_subtask("a", "a")
    chain.add_subtask("b", "b")
    chain.add_subtask("c", "c")
    chain.add_edge("a", "b", 0.7)
    chain.add_edge("b", "c", 0.7)
    assert router.route(chain) == Topology.SEQUENTIAL, "链式应 sequential"

    # 3) 高耦合 + 多子任务 → hierarchical
    # 需要 ω>1 (有并行性) AND γ>0.6 AND |V|>5
    hier = TaskDAG()
    for i in range(6):
        hier.add_subtask(f"v{i}", f"task {i}")
    # 两层并行, 但层间高耦合
    hier.add_edge("v0", "v2", 1.0)
    hier.add_edge("v1", "v2", 1.0)
    hier.add_edge("v2", "v4", 1.0)
    hier.add_edge("v3", "v4", 1.0)
    hier.add_edge("v4", "v5", 0.8)
    # γ = (1.0+1.0+1.0+1.0+0.8)/5 = 0.96 > 0.6
    # ω = max layer width: layer0=[v0,v1], layer1=[v2,v3], layer2=[v4], layer3=[v5] → ω=2
    assert router.route(hier) == Topology.HIERARCHICAL, "高耦合+多任务应 hierarchical"

    # 4) 宽 DAG + 低耦合 → parallel
    wide = TaskDAG()
    for i in range(6):
        wide.add_subtask(f"v{i}", f"task {i}")
    wide.add_edge("v0", "v5", 0.3)
    wide.add_edge("v1", "v5", 0.3)
    assert router.route(wide) == Topology.PARALLEL, "宽+低耦合应 parallel"

    # 5) 复杂 DAG → hybrid
    hybrid_dag = _build_customer_service_dag()
    assert router.route(hybrid_dag) == Topology.HYBRID, "客服 DAG 应 hybrid"

    # 6) DAG 属性计算
    analyzer = DAGAnalyzer()
    props = analyzer.analyze(hybrid_dag)
    assert props.omega >= 2, f"客服 DAG ω 应 ≥2, got {props.omega}"
    assert props.gamma > 0, f"γ 应 >0, got {props.gamma}"
    assert 0 < props.gamma < 1, f"γ 应在 (0,1), got {props.gamma}"

    # 7) 拓扑分层
    layers = HybridExecutor()._topological_layers(hybrid_dag)
    assert len(layers) >= 2, f"应至少 2 层, got {len(layers)}"
    # 所有子任务应被覆盖
    all_in_layers = set()
    for layer in layers:
        all_in_layers.update(layer)
    assert all_in_layers == set(hybrid_dag.subtasks.keys())

    # 8) Consistency Score
    cs = ConsistencyScore()
    assert cs.compute(["hello world", "hello world"]) == 1.0
    assert cs.compute(["hello world", "goodbye world"]) < 1.0
    assert 0 <= cs.compute(["a", "b"]) <= 1.0

    # 9) Synthesis: sequential 直接取最后
    synth = AdaptiveSynthesizer()
    seq_results = [
        ExecutionResult("a", "step1"),
        ExecutionResult("b", "step2"),
        ExecutionResult("c", "step3"),
    ]
    seq_out = synth.synthesize(Topology.SEQUENTIAL, seq_results, hybrid_dag)
    assert "step3" in seq_out, "sequential 应取最后一步"

    # 10) Synthesis: parallel 不一致 → arbitration
    par_results = [
        ExecutionResult("a", "result A is completely different"),
        ExecutionResult("b", "result B says opposite things here"),
    ]
    par_out = synth.synthesize(Topology.PARALLEL, par_results, hybrid_dag)
    assert "ARBITER" in par_out or "REROUTE" in par_out, "不一致时应仲裁或重路由"

    # 11) 完整 pipeline
    orch = AdaptOrch()
    result = orch.run(hybrid_dag)
    assert result["topology"] == "hybrid"
    assert len(result["results"]) > 0
    assert result["final_output"]

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()

# AgentTrace 因果图根因分析 — 完整可运行实现
# 论文：arXiv:2603.14688 | ICLR 2026 AI-Wild Workshop
# 复现核心：DAG 构建 + 反向 BFS 根因定位，无需 LLM，确定性算法

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import time


# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ExecutionEvent:
    """单个 Agent 执行事件"""
    agent_id: str
    event_type: str          # INVOKE | RETURN | ERROR
    timestamp: float         # Unix epoch (秒)
    inputs: dict             # Agent 输入参数
    outputs: dict            # Agent 输出（含 token/ID 用于因果边推导）
    status: str              # SUCCESS | FAILED | PENDING
    trace_id: str = ""       # 分布式 trace ID（跨服务聚合用）
    error_msg: str = ""      # 错误信息（FAILED 时填写）


@dataclass
class RCAResult:
    """根因分析结果"""
    root_agent: str                  # 根因 Agent ID
    causal_path: list[str]           # 因果传播路径（根因 → 故障终点）
    confidence: float                # 置信度 [0, 1]
    diagnosis_time_ms: float         # 诊断耗时（毫秒）
    error_summary: str = ""          # 根因错误摘要


# ─── 因果 DAG ─────────────────────────────────────────────────────────────────

class CausalDAG:
    """有向无环因果图"""

    def __init__(self):
        self.nodes: dict[str, ExecutionEvent] = {}   # agent_id → event
        self.edges: dict[str, list[str]] = {}         # agent_id → [downstream agent_ids]
        self.reverse_edges: dict[str, list[str]] = {} # agent_id → [upstream agent_ids]

    def add_node(self, event: ExecutionEvent):
        self.nodes[event.agent_id] = event
        self.edges.setdefault(event.agent_id, [])
        self.reverse_edges.setdefault(event.agent_id, [])

    def add_edge(self, from_agent: str, to_agent: str):
        """添加因果边 from → to"""
        if to_agent not in self.edges.get(from_agent, []):
            self.edges[from_agent].append(to_agent)
        if from_agent not in self.reverse_edges.get(to_agent, []):
            self.reverse_edges[to_agent].append(from_agent)

    def get_failed_nodes(self) -> list[str]:
        return [aid for aid, ev in self.nodes.items() if ev.status == "FAILED"]

    def get_terminal_failure(self) -> Optional[str]:
        """获取最下游的失败节点（无下游失败节点的 FAILED 节点）"""
        failed = set(self.get_failed_nodes())
        for fid in failed:
            downstream_failed = [d for d in self.edges.get(fid, []) if d in failed]
            if not downstream_failed:
                return fid
        return None


# ─── DAG 构建器 ───────────────────────────────────────────────────────────────

class CausalDAGBuilder:
    """从执行日志构建因果 DAG"""

    def from_execution_log(self, events: list[ExecutionEvent]) -> CausalDAG:
        dag = CausalDAG()

        # Step 1：添加所有节点
        for ev in events:
            dag.add_node(ev)

        # Step 2：结构信号推导因果边
        # 若 B 的 inputs 中包含 A 的 outputs 中的任意 token/ID，则 A→B
        for ev_b in events:
            b_input_vals = self._flatten_values(ev_b.inputs)
            for ev_a in events:
                if ev_a.agent_id == ev_b.agent_id:
                    continue
                a_output_vals = self._flatten_values(ev_a.outputs)
                if a_output_vals & b_input_vals:
                    # Step 3：位置信号过滤（时序约束：A 必须在 B 之前完成）
                    if ev_a.timestamp < ev_b.timestamp:
                        dag.add_edge(ev_a.agent_id, ev_b.agent_id)

        return dag

    @staticmethod
    def _flatten_values(d: dict) -> set:
        """将嵌套 dict 的所有叶子值展开为字符串集合（用于 token 匹配）"""
        result = set()
        stack = list(d.values())
        while stack:
            v = stack.pop()
            if isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, (list, tuple)):
                stack.extend(v)
            elif v is not None:
                result.add(str(v))
        return result


# ─── 根因分析引擎 ──────────────────────────────────────────────────────────────

class RCAEngine:
    """因果图根因分析引擎 — 目标延迟 < 200ms"""

    def locate_root_cause(self, dag: CausalDAG) -> Optional[RCAResult]:
        t_start = time.perf_counter()

        terminal = dag.get_terminal_failure()
        if terminal is None:
            return None  # 无失败节点

        # 反向 BFS：从故障终点沿 reverse_edges 逆向遍历
        visited = set()
        queue = deque([terminal])
        causal_path = []
        root_agent = terminal

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            causal_path.append(node)

            upstream_failed = [
                u for u in dag.reverse_edges.get(node, [])
                if dag.nodes[u].status == "FAILED"
            ]
            if not upstream_failed:
                # 没有上游失败节点 → 当前节点即根因
                root_agent = node
                break
            queue.extend(upstream_failed)

        # 反转路径为 根因 → 终点 方向
        causal_path = list(reversed(causal_path))

        # 置信度：路径上全为 FAILED 节点时为 1.0，有跳跃时降权
        failed_in_path = sum(1 for n in causal_path if dag.nodes[n].status == "FAILED")
        confidence = failed_in_path / len(causal_path) if causal_path else 0.0

        t_end = time.perf_counter()
        diagnosis_ms = (t_end - t_start) * 1000

        root_event = dag.nodes[root_agent]
        return RCAResult(
            root_agent=root_agent,
            causal_path=causal_path,
            confidence=confidence,
            diagnosis_time_ms=diagnosis_ms,
            error_summary=root_event.error_msg or root_event.outputs.get("error", ""),
        )


# ─── 测试：模拟 WF-A 3-Agent 失败链路 ────────────────────────────────────────

def _make_wfa_failure_scenario() -> list[ExecutionEvent]:
    """
    WF-A 供应链 MAS 失败链路：
    DemandForecastAgent → ProcurementAgent → POOrderAgent
    根因：DemandForecastAgent（外部数据 API 超时）
    """
    # DemandForecastAgent 失败（API 超时）
    ev_demand = ExecutionEvent(
        agent_id="DemandForecastAgent",
        event_type="RETURN",
        timestamp=1000.0,
        inputs={"sku_ids": ["BW-001", "BW-002"], "horizon_days": 30},
        outputs={"forecast_token": "TOKEN-DEMAND-ERR", "error": "API timeout after 30s"},
        status="FAILED",
        trace_id="trace-wfa-001",
        error_msg="外部数据源 API 超时，无法获取历史销售数据",
    )

    # ProcurementAgent 失败（收到异常 forecast token）
    ev_procurement = ExecutionEvent(
        agent_id="ProcurementAgent",
        event_type="RETURN",
        timestamp=1001.0,
        inputs={"forecast_token": "TOKEN-DEMAND-ERR", "budget": 500000},
        outputs={"po_token": "TOKEN-PO-ERR", "error": "Invalid forecast data"},
        status="FAILED",
        trace_id="trace-wfa-001",
        error_msg="收到无效预测数据，无法生成采购决策",
    )

    # POOrderAgent 失败（收到异常 po token）
    ev_po = ExecutionEvent(
        agent_id="POOrderAgent",
        event_type="RETURN",
        timestamp=1002.0,
        inputs={"po_token": "TOKEN-PO-ERR", "supplier_id": "SUP-100"},
        outputs={"error": "Cannot place order with invalid PO data"},
        status="FAILED",
        trace_id="trace-wfa-001",
        error_msg="PO 下单失败：无效的采购决策数据",
    )

    return [ev_demand, ev_procurement, ev_po]


def main():
    print("=" * 60)
    print("AgentTrace 因果图根因分析 — WF-A 供应链失败场景")
    print("=" * 60)

    events = _make_wfa_failure_scenario()

    # 构建因果 DAG
    builder = CausalDAGBuilder()
    dag = builder.from_execution_log(events)

    print(f"\n[DAG] 节点数: {len(dag.nodes)}, 边数: {sum(len(v) for v in dag.edges.values())}")
    for agent_id, downstream in dag.edges.items():
        if downstream:
            print(f"  {agent_id} → {downstream}")

    # 执行根因分析
    engine = RCAEngine()
    result = engine.locate_root_cause(dag)

    if result:
        print(f"\n[RCA 结果]")
        print(f"  根因 Agent  : {result.root_agent}")
        print(f"  因果路径    : {' → '.join(result.causal_path)}")
        print(f"  置信度      : {result.confidence:.2f}")
        print(f"  诊断耗时    : {result.diagnosis_time_ms:.3f} ms")
        print(f"  错误摘要    : {result.error_summary}")

        # 验证约束
        assert result.root_agent == "DemandForecastAgent", \
            f"根因应为 DemandForecastAgent，实际: {result.root_agent}"
        assert result.diagnosis_time_ms < 200, \
            f"诊断耗时应 < 200ms，实际: {result.diagnosis_time_ms:.2f}ms"
        assert result.confidence == 1.0, \
            f"全 FAILED 路径置信度应为 1.0，实际: {result.confidence}"

        print("\n✅ 所有断言通过：根因定位正确，耗时 < 200ms")
    else:
        print("❌ 未检测到失败节点")

    print("=" * 60)


if __name__ == "__main__":
    main()

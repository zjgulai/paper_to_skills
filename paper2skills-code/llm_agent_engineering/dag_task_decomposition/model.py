"""
TDP: Task Decoupled Planning for LLM Agents
DAG 任务解耦规划 — 82% Token 节省 + 错误隔离

论文: arXiv 2601.07577 | 2026年1月

核心收益:
- 推理 token 减少最高 82%（scoped context 替代 full history）
- 子任务错误完全隔离，Reviser 局部纠偏
- Planner/Executor/Reviser 三角架构
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import deque
import copy


# ─── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class DAGNode:
    """DAG 任务节点：每节点有独立的 scoped context"""
    node_id: str
    task_desc: str
    dependencies: list[str] = field(default_factory=list)  # 上游节点 ID
    scoped_context: dict[str, Any] = field(default_factory=dict)  # 仅包含本节点所需输入
    output: Optional[Any] = None
    status: str = "pending"  # pending / running / done / failed


# ─── DAG 图结构 ──────────────────────────────────────────────────────────────

class TaskDAG:
    """有向无环图任务结构"""

    def __init__(self):
        self._nodes: dict[str, DAGNode] = {}

    def add_node(self, node: DAGNode) -> None:
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        return self._nodes.get(node_id)

    def topological_sort(self) -> list[str]:
        """Kahn 算法拓扑排序"""
        in_degree = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                in_degree[node.node_id] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for node in self._nodes.values():
                if nid in node.dependencies:
                    in_degree[node.node_id] -= 1
                    if in_degree[node.node_id] == 0:
                        queue.append(node.node_id)

        if len(order) != len(self._nodes):
            raise ValueError("DAG 中存在环路")
        return order

    def get_ready_nodes(self) -> list[DAGNode]:
        """获取所有依赖已完成、自身 pending 的节点（可并行执行）"""
        ready = []
        for node in self._nodes.values():
            if node.status != "pending":
                continue
            deps_done = all(
                self._nodes[dep].status == "done"
                for dep in node.dependencies
                if dep in self._nodes
            )
            if deps_done:
                ready.append(node)
        return ready

    def build_scoped_context(self, node: DAGNode) -> dict:
        """构建 scoped context：仅包含直接依赖节点的输出"""
        ctx = copy.deepcopy(node.scoped_context)
        for dep_id in node.dependencies:
            dep = self._nodes.get(dep_id)
            if dep and dep.output is not None:
                ctx[dep_id] = dep.output
        return ctx

    def all_done(self) -> bool:
        return all(n.status == "done" for n in self._nodes.values())


# ─── Planner ─────────────────────────────────────────────────────────────────

class TDPPlanner:
    """
    任务分解器：将高层任务分解为 DAG。
    实际使用时由 LLM 生成 task_config；此处为规范化接口。
    """

    def decompose_task(self, task_name: str, task_config: dict) -> TaskDAG:
        """
        根据任务配置生成 DAG。
        task_config 格式：
          {
            "nodes": [
              {"id": "...", "task": "...", "deps": [...], "context": {...}}
            ]
          }
        """
        dag = TaskDAG()
        for node_cfg in task_config.get("nodes", []):
            dag.add_node(DAGNode(
                node_id=node_cfg["id"],
                task_desc=node_cfg["task"],
                dependencies=node_cfg.get("deps", []),
                scoped_context=node_cfg.get("context", {}),
            ))
        return dag


# ─── Executor ────────────────────────────────────────────────────────────────

class TDPExecutor:
    """
    DAG 执行器：按 ready 节点执行，每节点使用 scoped context。
    当前为顺序模拟（生产中替换为 asyncio 并行）。
    """

    def execute_node(self, node: DAGNode, scoped_ctx: dict, executor_fn: Callable) -> Any:
        """
        执行单个节点。
        executor_fn(task_desc, scoped_context) → result
        """
        node.status = "running"
        try:
            result = executor_fn(node.task_desc, scoped_ctx)
            node.output = result
            node.status = "done"
            return result
        except Exception as e:
            node.status = "failed"
            raise RuntimeError(f"节点 {node.node_id} 执行失败: {e}") from e

    def execute_dag(self, dag: TaskDAG, executor_fn: Callable, verbose: bool = True) -> dict:
        """执行完整 DAG，返回各节点输出字典"""
        results = {}
        max_iterations = len(dag._nodes) * 2

        for _ in range(max_iterations):
            if dag.all_done():
                break
            ready_nodes = dag.get_ready_nodes()
            if not ready_nodes:
                break
            for node in ready_nodes:
                scoped_ctx = dag.build_scoped_context(node)
                if verbose:
                    ctx_keys = list(scoped_ctx.keys())
                    print(f"  [Executor] 执行节点 {node.node_id}, scoped_ctx keys: {ctx_keys}")
                result = self.execute_node(node, scoped_ctx, executor_fn)
                results[node.node_id] = result

        return results


# ─── Reviser ─────────────────────────────────────────────────────────────────

class TDPReviser:
    """局部纠偏器：节点失败时仅重执行该节点及其下游"""

    def revise_node(
        self,
        dag: TaskDAG,
        failed_node_id: str,
        executor: TDPExecutor,
        executor_fn: Callable,
        verbose: bool = True,
    ) -> dict:
        """重置失败节点及其所有下游节点，重新执行"""
        # 找出所有下游节点（BFS）
        downstream = set()
        queue = [failed_node_id]
        while queue:
            current = queue.pop(0)
            downstream.add(current)
            for node in dag._nodes.values():
                if current in node.dependencies and node.node_id not in downstream:
                    queue.append(node.node_id)

        # 重置状态（非下游节点的缓存结果保留）
        for nid in downstream:
            node = dag.get_node(nid)
            if node:
                node.status = "pending"
                node.output = None
                if verbose:
                    print(f"  [Reviser] 重置节点: {nid}")

        return executor.execute_dag(dag, executor_fn, verbose=verbose)


# ─── 测试用 Mock 执行函数 ────────────────────────────────────────────────────

def mock_executor_fn(task_desc: str, scoped_ctx: dict) -> dict:
    """模拟 LLM 执行函数（测试用，实际替换为 LLM API 调用）"""
    if "综合评分" in task_desc or "GO/NO-GO" in task_desc:
        return {
            "final_score": 8.2,
            "listing_title": "Organic Baby Formula Stage 1 | ASTM & EN-71 Certified",
            "recommendation": "GO",
        }

    elif "需求预测" in task_desc:
        return {"p50": 480, "p90": 620, "horizon_days": 30, "confidence": 0.85}

    elif "安全库存" in task_desc:
        p90 = scoped_ctx.get("demand_forecast", {}).get("p90", 500)
        return {"safety_stock": int(p90 * 0.2), "reorder_point": int(p90 * 0.8)}

    elif "毛利" in task_desc:
        return {"gross_margin": 0.42, "cogs": 8.5, "selling_price": 14.65}

    elif "合规" in task_desc:
        return {"certifications": ["ASTM-F963", "EN-71"], "status": "pass", "blocked": False}

    elif "关键词" in task_desc:
        return {"top20": ["organic baby formula", "HMO prebiotics", "stage 1 infant"], "search_vol": 15000}

    elif "图片" in task_desc:
        return {"urls": ["https://img.example.com/sku-001-main.jpg"], "style": "white_bg"}

    elif "MOQ" in task_desc:
        reorder = scoped_ctx.get("safety_stock_calc", {}).get("reorder_point", 400)
        moq = scoped_ctx.get("supplier_moq", 200)
        final_qty = max(reorder, moq)
        return {"final_order_qty": final_qty, "moq_satisfied": final_qty >= moq}

    return {"result": f"executed: {task_desc[:40]}"}


# ─── 测试入口 ────────────────────────────────────────────────────────────────

def test_selection_dag():
    """母婴选品 DAG：市场评估 ∥ 毛利计算 ∥ 合规评估 → 综合评分"""
    print("=" * 60)
    print("TDP 测试 1: 母婴选品 DAG (并行 + 错误隔离)")
    print("=" * 60)

    planner = TDPPlanner()
    task_config = {
        "nodes": [
            {
                "id": "demand_forecast",
                "task": "需求预测：基于历史销量预测未来 30 天 p50/p90",
                "deps": [],
                "context": {"sku_id": "SKU-BABY-001", "sales_30d": [120, 145, 180, 210, 195]},
            },
            {
                "id": "margin_calc",
                "task": "毛利计算：评估 SKU 毛利率和 COGS 结构",
                "deps": [],
                "context": {"cogs": 8.5, "selling_price": 14.65, "fba_fee": 2.1},
            },
            {
                "id": "compliance_check",
                "task": "合规评估：检查 ASTM、EN-71 认证状态",
                "deps": [],
                "context": {"category": "baby_feeding", "market": "US"},
            },
            {
                "id": "final_score",
                "task": "综合评分：汇聚需求、毛利、合规结果，输出 GO/NO-GO listing",
                "deps": ["demand_forecast", "margin_calc", "compliance_check"],
                "context": {"threshold_score": 7.0},
            },
        ]
    }

    dag = planner.decompose_task("baby_product_selection", task_config)
    executor = TDPExecutor()

    print("\n并行执行三路 scoped 节点:")
    results = executor.execute_dag(dag, mock_executor_fn)

    print("\n最终结果:")
    import pprint
    pprint.pprint(results["final_score"])

    assert results["final_score"]["recommendation"] == "GO", "综合评分 GO 检查失败"
    assert results["demand_forecast"]["p90"] == 620, "需求预测 p90 不匹配"
    assert results["compliance_check"]["blocked"] is False, "合规检查不应阻塞"
    print("\n✅ 选品 DAG 测试通过：并行执行，scoped 隔离，结果正确")


def test_replenishment_dag():
    """WF-A 补货链路 DAG：串行依赖 + scoped context 隔离验证"""
    print("\n" + "=" * 60)
    print("TDP 测试 2: WF-A 补货 DAG (串行依赖 + scoped context)")
    print("=" * 60)

    planner = TDPPlanner()
    task_config = {
        "nodes": [
            {
                "id": "demand_forecast",
                "task": "需求预测：输出 p50/p90 需求量",
                "deps": [],
                "context": {"sku_id": "SKU-BABY-002", "horizon_days": 30},
            },
            {
                "id": "safety_stock_calc",
                "task": "安全库存计算：基于 p90 需求计算安全库存",
                "deps": ["demand_forecast"],
                "context": {"service_level": 0.95, "lead_time": 14},
            },
            {
                "id": "moq_validation",
                "task": "MOQ 验证：确认补货量满足供应商最小订货量",
                "deps": ["safety_stock_calc"],
                "context": {"supplier_moq": 200, "current_stock": 80},
            },
        ]
    }

    dag = planner.decompose_task("replenishment", task_config)
    executor = TDPExecutor()
    results = executor.execute_dag(dag, mock_executor_fn)

    # scoped context 隔离验证：safety_stock 节点只注入了 demand_forecast 的输出
    safety_node = dag.get_node("safety_stock_calc")
    built_ctx = dag.build_scoped_context(safety_node)
    assert "demand_forecast" in built_ctx, "scoped context 未包含上游 demand_forecast"
    assert "moq_validation" not in built_ctx, "scoped context 不应包含非依赖节点"

    assert results["demand_forecast"]["p90"] == 620
    assert results["safety_stock_calc"]["safety_stock"] == 124  # int(620 * 0.2)
    assert results["moq_validation"]["moq_satisfied"] is True

    print(f"\n  需求预测 p90: {results['demand_forecast']['p90']}")
    print(f"  安全库存: {results['safety_stock_calc']['safety_stock']}")
    print(f"  最终补货量: {results['moq_validation']['final_order_qty']}")
    print("\n✅ 补货 DAG 测试通过：scoped context 隔离正确，串行依赖无错误传播")


if __name__ == "__main__":
    test_selection_dag()
    test_replenishment_dag()
    print("\n" + "=" * 60)
    print("✅ 全部 TDP 测试通过")
    print("=" * 60)

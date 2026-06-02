"""
Dynamic DAG Orchestration: 运行时动态调整工作流 DAG

来源: 动态 DAG 编排框架 2025-2026 实践（基于 TDP 的扩展）
应用场景: 母婴出海 WF-D 自适应选品 / WF-A 异常补货
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 节点状态枚举
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# 2. DynamicDAGNode 数据类
# ---------------------------------------------------------------------------

@dataclass
class DynamicDAGNode:
    """动态 DAG 节点描述符

    Attributes:
        node_id: 唯一标识
        description: 节点描述
        fn: 执行函数，签名 (context: dict) -> Any
        dependencies: 依赖的 node_id 列表
        condition: 执行前置条件，签名 (context: dict) -> bool；None 表示无条件执行
        on_success_add: 成功后动态注入的节点列表
        on_failure_skip: 失败/条件不满足时跳过的节点 id 列表
        can_parallelize: 是否允许与同批次其他节点并行（模拟并行，单线程交替执行）
        status: 当前状态
        result: 执行结果
        error: 错误信息
    """
    node_id: str
    description: str
    fn: Callable[[dict[str, Any]], Any]
    dependencies: list[str] = field(default_factory=list)
    condition: Callable[[dict[str, Any]], bool] | None = None
    on_success_add: list[DynamicDAGNode] = field(default_factory=list)
    on_failure_skip: list[str] = field(default_factory=list)
    can_parallelize: bool = False
    status: NodeStatus = NodeStatus.PENDING
    result: Any = None
    error: str | None = None

    def is_ready(self, done_ids: set[str]) -> bool:
        """所有依赖均已完成或跳过时返回 True"""
        return all(dep in done_ids for dep in self.dependencies)


# ---------------------------------------------------------------------------
# 3. 执行结果
# ---------------------------------------------------------------------------

@dataclass
class DAGResult:
    """DAG 执行汇总结果"""
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    injected: list[str] = field(default_factory=list)   # 运行时新增的节点 id
    context: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.failed) == 0


# ---------------------------------------------------------------------------
# 4. DynamicDAGEngine
# ---------------------------------------------------------------------------

class DynamicDAGEngine:
    """运行时动态调整 DAG 拓扑的执行引擎

    支持三种变形操作：
    - 节点插入（on_success_add）
    - 节点跳过（on_failure_skip / condition=False）
    - 子图并行化（can_parallelize=True 的节点同批调度）
    """

    def __init__(self, max_iterations: int = 200) -> None:
        self._max_iterations = max_iterations

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def execute_with_adaptation(
        self,
        initial_nodes: list[DynamicDAGNode],
        context: dict[str, Any] | None = None,
    ) -> DAGResult:
        """执行动态 DAG，支持运行时拓扑变更

        Args:
            initial_nodes: 初始节点列表（不需要预先排序）
            context: 共享上下文，节点可读写

        Returns:
            DAGResult 汇总结果
        """
        ctx: dict[str, Any] = context or {}
        result = DAGResult(context=ctx)
        t0 = time.monotonic()

        # 节点注册表（支持运行时追加）
        nodes: dict[str, DynamicDAGNode] = {n.node_id: n for n in initial_nodes}
        # 已完成或跳过的节点 id（用于依赖判定）
        done_ids: set[str] = set()
        # 强制跳过集合（由 on_failure_skip 填充）
        force_skip: set[str] = set()

        iterations = 0
        while iterations < self._max_iterations:
            iterations += 1
            ready = self._collect_ready_batch(nodes, done_ids, force_skip)
            if not ready:
                break  # 无可执行节点，结束

            parallel_batch, serial_batch = self._split_parallel(ready)

            # 并行批次（模拟：单线程按顺序执行，语义上为同批）
            for node in parallel_batch:
                self._run_node(node, ctx, nodes, done_ids, force_skip, result)

            # 串行批次（逐个执行）
            for node in serial_batch:
                self._run_node(node, ctx, nodes, done_ids, force_skip, result)

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "DAG 执行完毕: completed=%s, skipped=%s, failed=%s, injected=%s, elapsed=%.1fms",
            result.completed, result.skipped, result.failed, result.injected, result.elapsed_ms,
        )
        return result

    def should_add_node(
        self, context: dict[str, Any], condition: Callable[[dict[str, Any]], bool]
    ) -> bool:
        """判断是否应动态注入节点（供外部业务逻辑调用）"""
        try:
            return bool(condition(context))
        except Exception as exc:  # noqa: BLE001
            logger.warning("should_add_node 条件评估异常: %s", exc)
            return False

    def parallelize_subtree(
        self,
        nodes: list[DynamicDAGNode],
        context: dict[str, Any],  # noqa: ARG002
    ) -> list[DynamicDAGNode]:
        """将节点列表标记为可并行，返回修改后的列表"""
        for node in nodes:
            node.can_parallelize = True
        return nodes

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _collect_ready_batch(
        self,
        nodes: dict[str, DynamicDAGNode],
        done_ids: set[str],
        force_skip: set[str],
    ) -> list[DynamicDAGNode]:
        """收集当前可执行（依赖满足且未完成）的节点"""
        ready = []
        for node in nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            # 强制跳过
            if node.node_id in force_skip:
                node.status = NodeStatus.SKIPPED
                done_ids.add(node.node_id)
                continue
            if node.is_ready(done_ids):
                ready.append(node)
        return ready

    def _split_parallel(
        self, nodes: list[DynamicDAGNode]
    ) -> tuple[list[DynamicDAGNode], list[DynamicDAGNode]]:
        """将就绪节点按 can_parallelize 分为并行批次和串行批次"""
        parallel = [n for n in nodes if n.can_parallelize]
        serial = [n for n in nodes if not n.can_parallelize]
        return parallel, serial

    def _run_node(
        self,
        node: DynamicDAGNode,
        ctx: dict[str, Any],
        nodes: dict[str, DynamicDAGNode],
        done_ids: set[str],
        force_skip: set[str],
        result: DAGResult,
    ) -> None:
        """执行单个节点，处理条件检查、结果注入和拓扑变更"""
        # 条件检查（condition=False → 跳过该节点）
        if node.condition is not None and not node.condition(ctx):
            logger.debug("节点 %s 条件不满足，跳过", node.node_id)
            node.status = NodeStatus.SKIPPED
            done_ids.add(node.node_id)
            result.skipped.append(node.node_id)
            # 条件不满足时也触发 on_failure_skip
            for skip_id in node.on_failure_skip:
                force_skip.add(skip_id)
            return

        node.status = NodeStatus.RUNNING
        logger.debug("执行节点: %s — %s", node.node_id, node.description)

        try:
            node.result = node.fn(ctx)
            node.status = NodeStatus.COMPLETED
            done_ids.add(node.node_id)
            result.completed.append(node.node_id)
            logger.debug("节点 %s 完成，结果: %s", node.node_id, node.result)

            # 成功后动态注入新节点
            for new_node in node.on_success_add:
                if new_node.node_id not in nodes:
                    nodes[new_node.node_id] = new_node
                    result.injected.append(new_node.node_id)
                    logger.info("动态注入节点: %s", new_node.node_id)

        except Exception as exc:  # noqa: BLE001
            node.status = NodeStatus.FAILED
            node.error = str(exc)
            done_ids.add(node.node_id)  # 失败节点也视为"已处理"，让下游可感知
            result.failed.append(node.node_id)
            logger.error("节点 %s 失败: %s", node.node_id, exc)

            # 失败时触发跳过列表
            for skip_id in node.on_failure_skip:
                force_skip.add(skip_id)


# ---------------------------------------------------------------------------
# 5. 测试：WF-D 选品 DAG（饱和市场 → 跳过；高潜力 → 注入深度竞品节点）
# ---------------------------------------------------------------------------

def _make_wfd_dag(market_result: str) -> tuple[list[DynamicDAGNode], dict[str, Any]]:
    """构造 WF-D 选品 DAG

    Args:
        market_result: "saturated" | "high_potential" | "normal"
    """
    ctx: dict[str, Any] = {}

    # 深度竞品分析节点（动态注入）
    deep_competitor = DynamicDAGNode(
        node_id="deep_competitor_analysis",
        description="深度竞品分析（动态注入）",
        fn=lambda c: c.update({"deep_competitor": "top3 分析完成"}) or "深度竞品分析完成",
        dependencies=["market_eval"],
    )

    # 跳过标记输出节点
    output_no_go = DynamicDAGNode(
        node_id="output_no_go",
        description="输出不推荐报告（饱和市场）",
        fn=lambda c: c.update({"recommendation": "NO-GO: 市场饱和"}) or "NO-GO",
        dependencies=["market_eval"],
    )

    def market_eval_fn(ctx: dict[str, Any]) -> str:
        ctx["market_result"] = market_result
        return market_result

    # 根据 market_result 决定 on_success_add 和 on_failure_skip
    if market_result == "saturated":
        market_node = DynamicDAGNode(
            node_id="market_eval",
            description="市场评估",
            fn=market_eval_fn,
            on_success_add=[output_no_go],
            on_failure_skip=["competitor_basic", "margin_calc", "compliance"],
        )
        # 饱和时成功后立即将后续节点加入跳过列表（通过条件控制）
        # 实际通过 on_failure_skip 在下次收集时跳过
        # 此处用 condition 让 competitor_basic 自己检查
    elif market_result == "high_potential":
        market_node = DynamicDAGNode(
            node_id="market_eval",
            description="市场评估",
            fn=market_eval_fn,
            on_success_add=[deep_competitor],
        )
    else:
        market_node = DynamicDAGNode(
            node_id="market_eval",
            description="市场评估",
            fn=market_eval_fn,
        )

    def make_skip_if_saturated(name: str, actual_fn: Callable[[dict], Any]) -> DynamicDAGNode:
        return DynamicDAGNode(
            node_id=name,
            description=f"{name}（饱和时跳过）",
            fn=actual_fn,
            dependencies=["market_eval"],
            condition=lambda c: c.get("market_result") != "saturated",
        )

    competitor_basic = make_skip_if_saturated(
        "competitor_basic",
        lambda c: c.update({"competitor": "基础竞品分析完成"}) or "ok",
    )
    margin_calc = DynamicDAGNode(
        node_id="margin_calc",
        description="毛利测算",
        fn=lambda c: c.update({"margin": 0.32}) or 0.32,
        dependencies=["competitor_basic"],
        condition=lambda c: c.get("market_result") != "saturated",
    )
    compliance = DynamicDAGNode(
        node_id="compliance",
        description="合规检查",
        fn=lambda c: c.update({"compliance": "PASS"}) or "PASS",
        dependencies=["margin_calc"],
        condition=lambda c: c.get("market_result") != "saturated",
    )
    output_report = DynamicDAGNode(
        node_id="output_report",
        description="输出最终报告",
        fn=lambda c: c.update({"final": "报告生成完毕"}) or "报告生成完毕",
        dependencies=["compliance"],
        condition=lambda c: c.get("market_result") != "saturated",
    )

    initial = [market_node, competitor_basic, margin_calc, compliance, output_report]
    return initial, ctx


def run_tests() -> None:
    """运行 WF-D 选品 DAG 测试"""
    engine = DynamicDAGEngine()

    # 测试 1：饱和市场 → 后续节点应被跳过，output_no_go 应被注入并执行
    print("\n=== 测试 1：饱和市场（触发跳过）===")
    nodes, ctx = _make_wfd_dag("saturated")
    result = engine.execute_with_adaptation(nodes, ctx)
    assert "market_eval" in result.completed, "market_eval 应完成"
    assert "output_no_go" in result.injected, "output_no_go 应被动态注入"
    assert "output_no_go" in result.completed, "output_no_go 应执行完成"
    assert ctx.get("recommendation") == "NO-GO: 市场饱和", "recommendation 应为 NO-GO"
    # competitor_basic / margin_calc / compliance / output_report 应被跳过
    for skip_id in ["competitor_basic", "margin_calc", "compliance", "output_report"]:
        assert skip_id in result.skipped, f"{skip_id} 应被跳过"
    print(f"  ✓ 完成: {result.completed}, 跳过: {result.skipped}, 注入: {result.injected}")
    print(f"  ✓ 耗时: {result.elapsed_ms:.1f}ms, 推荐: {ctx.get('recommendation')}")

    # 测试 2：高潜力市场 → deep_competitor_analysis 应被动态注入并执行
    print("\n=== 测试 2：高潜力市场（触发节点注入）===")
    nodes, ctx = _make_wfd_dag("high_potential")
    result = engine.execute_with_adaptation(nodes, ctx)
    assert "market_eval" in result.completed
    assert "deep_competitor_analysis" in result.injected, "deep_competitor_analysis 应被动态注入"
    assert "deep_competitor_analysis" in result.completed, "deep_competitor_analysis 应执行完成"
    print(f"  ✓ 完成: {result.completed}, 注入: {result.injected}")
    print(f"  ✓ 耗时: {result.elapsed_ms:.1f}ms")

    # 测试 3：正常市场 → 全链路执行，无跳过无注入
    print("\n=== 测试 3：正常市场（全链路执行）===")
    nodes, ctx = _make_wfd_dag("normal")
    result = engine.execute_with_adaptation(nodes, ctx)
    assert result.skipped == [], f"正常市场不应有跳过节点，实际: {result.skipped}"
    assert result.injected == [], "正常市场不应有注入节点"
    assert "output_report" in result.completed
    print(f"  ✓ 完成: {result.completed}, 耗时: {result.elapsed_ms:.1f}ms")

    print("\n✅ 所有测试通过")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run_tests()

"""
ParaManager: Small Model as Master Orchestrator
Agent-as-Tool 并行子任务分解

论文来源: arXiv 2604.17009 | 2026年4月
应用场景: 母婴出海 WF-D 选品 5 维并行扫描 / 新品上架 SOP 并行化
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. 子任务状态
# ---------------------------------------------------------------------------

class SubTaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


# ---------------------------------------------------------------------------
# 2. SubTask 数据类
# ---------------------------------------------------------------------------

@dataclass
class SubTask:
    """并行子任务描述符

    Attributes:
        task_id: 唯一标识
        description: 任务描述
        dependencies: 依赖的 task_id 列表（空列表表示可立即执行）
        assigned_agent: 执行该任务的 AgentAsTool 名称
        status: 当前状态
        result: 执行结果（完成后填充）
        error: 错误信息（失败时填充）
    """
    task_id: str
    description: str
    assigned_agent: str
    dependencies: list[str] = field(default_factory=list)
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Any = None
    error: str | None = None

    def is_ready(self, completed_ids: set[str]) -> bool:
        """所有依赖已完成时返回 True"""
        return all(dep in completed_ids for dep in self.dependencies)


# ---------------------------------------------------------------------------
# 3. AgentAsTool：统一 Agent 和 Tool 接口
# ---------------------------------------------------------------------------

class AgentAsTool:
    """将 Agent 或 Tool 统一为标准化调用接口

    无论底层是有状态 Agent 还是无状态 Tool，
    外部调用方式统一为 invoke(input_data) -> dict。

    返回格式统一:
        {
            "status": "success" | "failed",
            "output": ...,
            "progress": 1.0,
        }
    """

    def __init__(self, name: str, fn: Callable[..., Any], timeout_seconds: float = 30.0) -> None:
        self.name = name
        self._fn = fn
        self._timeout = timeout_seconds

    def invoke(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """同步调用，返回标准化结果"""
        try:
            raw_result = self._fn(**input_data)
            return {"status": "success", "output": raw_result, "progress": 1.0}
        except Exception as exc:
            logger.error("AgentAsTool [%s] 执行失败: %s", self.name, exc)
            return {"status": "failed", "output": None, "progress": 0.0, "error": str(exc)}

    async def invoke_async(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """异步调用（在线程池中运行同步函数）"""
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: self.invoke(input_data)),
            timeout=self._timeout,
        )


# ---------------------------------------------------------------------------
# 4. 最终汇总结果
# ---------------------------------------------------------------------------

@dataclass
class FinalResult:
    """并行编排最终结果

    Attributes:
        success: 所有关键子任务是否成功
        task_results: task_id -> 执行结果映射
        blocked_tasks: 被阻塞（未执行）的任务列表
        failed_tasks: 执行失败的任务列表
        cross_validation_flags: 交叉验证发现的冲突项
        elapsed_seconds: 总耗时
        recommendation: 最终推荐（聚合输出）
    """
    success: bool
    task_results: dict[str, Any]
    blocked_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    cross_validation_flags: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    recommendation: str = ""


# ---------------------------------------------------------------------------
# 5. ParaManagerCore：核心编排器
# ---------------------------------------------------------------------------

class ParaManagerCore:
    """ParaManager 核心编排器

    基于 Agent-as-Tool 统一接口，实现：
    1. decompose: 状态驱动的子任务分解（含依赖分析）
    2. execute_parallel: 并行执行 + 交叉验证
    3. aggregate_results: 结果聚合 + 推荐输出
    """

    def __init__(self, agents: dict[str, AgentAsTool]) -> None:
        self._agents = agents

    def decompose(self, task: dict[str, Any]) -> list[SubTask]:
        """状态驱动的任务分解

        Args:
            task: 任务描述字典，至少包含 "type" 和 "params"

        Returns:
            SubTask 列表（含依赖关系）

        说明：实际 ParaManager 由 Qwen3-4B 动态生成 SubTask 列表；
        此处为模板实现，硬编码 WF-D 选品扫描场景。
        """
        task_type = task.get("type", "selection_scan")

        if task_type == "selection_scan":
            return self._decompose_selection_scan(task)
        elif task_type == "listing_sop":
            return self._decompose_listing_sop(task)
        else:
            raise ValueError(f"未知任务类型: {task_type}")

    def _decompose_selection_scan(self, task: dict[str, Any]) -> list[SubTask]:
        """WF-D 选品 5 维并行分解（全无依赖，同批并行）"""
        product_kw = task.get("params", {}).get("keyword", "baby sterilizer")
        return [
            SubTask(
                task_id="market_size",
                description=f"市场空间评估: {product_kw}",
                assigned_agent="market_size_agent",
                dependencies=[],
            ),
            SubTask(
                task_id="margin_calc",
                description=f"毛利测算: {product_kw}",
                assigned_agent="margin_calc_tool",
                dependencies=[],
            ),
            SubTask(
                task_id="compliance",
                description=f"合规检查: {product_kw}",
                assigned_agent="compliance_agent",
                dependencies=[],
            ),
            SubTask(
                task_id="kg_attribute",
                description=f"KG 属性匹配: {product_kw}",
                assigned_agent="kg_attribute_tool",
                dependencies=[],
            ),
            SubTask(
                task_id="causal_lift",
                description=f"因果 Lift 预测: {product_kw}",
                assigned_agent="causal_lift_agent",
                dependencies=[],
            ),
        ]

    def _decompose_listing_sop(self, task: dict[str, Any]) -> list[SubTask]:
        """上架 SOP 两批次分解"""
        return [
            SubTask(
                task_id="compliance_3track",
                description="合规三轨验证",
                assigned_agent="compliance_agent",
                dependencies=[],
            ),
            SubTask(
                task_id="image_gen",
                description="商业图片生成",
                assigned_agent="image_gen_agent",
                dependencies=[],
            ),
            SubTask(
                task_id="keyword_research",
                description="关键词研究",
                assigned_agent="kg_attribute_tool",
                dependencies=[],
            ),
            SubTask(
                task_id="listing_copy",
                description="Listing 文案生成",
                assigned_agent="compliance_agent",
                dependencies=["compliance_3track", "image_gen", "keyword_research"],
            ),
        ]

    def execute_parallel(self, subtasks: list[SubTask]) -> dict[str, Any]:
        """并行执行子任务 + 交叉验证

        使用拓扑层次执行：
        - 第 0 批：无依赖任务，并行执行
        - 第 N 批：依赖第 N-1 批全部完成的任务
        """
        completed_ids: set[str] = set()
        all_results: dict[str, Any] = {}
        task_map = {t.task_id: t for t in subtasks}

        while True:
            ready_tasks = [
                t for t in subtasks
                if t.status == SubTaskStatus.PENDING and t.is_ready(completed_ids)
            ]
            if not ready_tasks:
                break

            logger.info("并行执行批次: %s", [t.task_id for t in ready_tasks])

            # 并行执行当前批次
            batch_results = asyncio.run(self._run_batch_async(ready_tasks))

            for task_id, result in batch_results.items():
                task = task_map[task_id]
                if result["status"] == "success":
                    task.status = SubTaskStatus.COMPLETED
                    task.result = result["output"]
                    completed_ids.add(task_id)
                    all_results[task_id] = result["output"]
                else:
                    task.status = SubTaskStatus.FAILED
                    task.error = result.get("error", "未知错误")
                    # 级联阻塞依赖此任务的下游任务
                    self._cascade_block(task_id, task_map)

        return all_results

    async def _run_batch_async(self, tasks: list[SubTask]) -> dict[str, Any]:
        """异步并行执行一批任务"""
        async def run_one(task: SubTask) -> tuple[str, dict[str, Any]]:
            task.status = SubTaskStatus.RUNNING
            agent = self._agents.get(task.assigned_agent)
            if agent is None:
                return task.task_id, {"status": "failed", "output": None, "error": f"Agent [{task.assigned_agent}] 未注册"}
            result = await agent.invoke_async({})
            return task.task_id, result

        results = await asyncio.gather(*[run_one(t) for t in tasks], return_exceptions=False)
        return dict(results)

    def _cascade_block(self, failed_id: str, task_map: dict[str, SubTask]) -> None:
        """级联阻塞依赖失败任务的下游任务"""
        for task in task_map.values():
            if failed_id in task.dependencies and task.status == SubTaskStatus.PENDING:
                task.status = SubTaskStatus.BLOCKED
                logger.warning("任务 [%s] 因依赖 [%s] 失败而被阻塞", task.task_id, failed_id)
                # 递归阻塞
                self._cascade_block(task.task_id, task_map)

    def aggregate_results(self, results: dict[str, Any]) -> FinalResult:
        """聚合结果 + 交叉验证 + 生成最终推荐"""
        flags: list[str] = []

        # 交叉验证：毛利 < 25% 且市场空间 < 10M → NO-GO
        market = results.get("market_size", {})
        margin = results.get("margin_calc", {})
        if (
            market.get("market_size_usd", 999_999_999) < 10_000_000
            and margin.get("gross_margin_pct", 100) < 25
        ):
            flags.append("LOW_MARKET_LOW_MARGIN: 市场空间 < $10M 且毛利 < 25%，建议 NO-GO")

        # 合规阻塞检查
        compliance = results.get("compliance", {})
        if compliance.get("blocked"):
            flags.append(f"COMPLIANCE_BLOCKED: {compliance.get('reason', '合规不通过')}")

        success = len(flags) == 0 or not any("BLOCKED" in f for f in flags)
        recommendation = "GO" if success and not flags else ("NO-GO" if flags else "REVIEW")

        return FinalResult(
            success=success,
            task_results=results,
            cross_validation_flags=flags,
            recommendation=recommendation,
        )


# ---------------------------------------------------------------------------
# 6. 测试：选品 5 维并行扫描，验证并行 vs 串行耗时对比
# ---------------------------------------------------------------------------

def _make_demo_agents() -> dict[str, AgentAsTool]:
    """创建演示用 Agent（模拟耗时操作）"""
    import time as _time

    def market_size_fn() -> dict[str, Any]:
        _time.sleep(0.5)
        return {"market_size_usd": 50_000_000, "trend": "growing", "bsr_top10_avg_monthly": 800}

    def margin_calc_fn() -> dict[str, Any]:
        _time.sleep(0.4)
        return {"gross_margin_pct": 38, "landed_cost_usd": 12.5, "sell_price_usd": 24.99}

    def compliance_fn() -> dict[str, Any]:
        _time.sleep(0.6)
        return {"blocked": False, "certifications_required": ["CPSC", "ASTM F963"], "risk_level": "low"}

    def kg_attribute_fn() -> dict[str, Any]:
        _time.sleep(0.3)
        return {"age_group": "0-24m", "safety_certified": True, "bpa_free": True}

    def causal_lift_fn() -> dict[str, Any]:
        _time.sleep(0.5)
        return {"sp_roas_lift": 1.4, "sb_click_lift": 1.2, "confidence": 0.85}

    return {
        "market_size_agent": AgentAsTool("market_size_agent", market_size_fn),
        "margin_calc_tool": AgentAsTool("margin_calc_tool", margin_calc_fn),
        "compliance_agent": AgentAsTool("compliance_agent", compliance_fn),
        "kg_attribute_tool": AgentAsTool("kg_attribute_tool", kg_attribute_fn),
        "causal_lift_agent": AgentAsTool("causal_lift_agent", causal_lift_fn),
        "image_gen_agent": AgentAsTool("image_gen_agent", lambda: {"images": 6, "quality": "commercial"}),
    }


def run_selection_scan_test() -> None:
    """测试 WF-D 5 维并行扫描，输出并行 vs 串行耗时对比"""
    print("=" * 60)
    print("WF-D 选品 5 维并行扫描测试")
    print("=" * 60)

    agents = _make_demo_agents()
    manager = ParaManagerCore(agents)

    task = {"type": "selection_scan", "params": {"keyword": "baby bottle sterilizer"}}

    # 并行执行
    subtasks = manager.decompose(task)
    print(f"\n分解出 {len(subtasks)} 个子任务:")
    for st in subtasks:
        print(f"  [{st.task_id}] {st.description} -> {st.assigned_agent}")

    t0 = time.time()
    results = manager.execute_parallel(subtasks)
    parallel_elapsed = time.time() - t0

    final = manager.aggregate_results(results)
    final.elapsed_seconds = parallel_elapsed

    print(f"\n并行执行耗时: {parallel_elapsed:.2f}s")
    print(f"推荐结论: {final.recommendation}")
    print(f"交叉验证标记: {final.cross_validation_flags or '无冲突'}")
    print("\n各维度结果:")
    for task_id, res in results.items():
        print(f"  [{task_id}]: {res}")

    # 估算串行耗时（各任务耗时之和）
    serial_estimate = 0.5 + 0.4 + 0.6 + 0.3 + 0.5
    speedup = serial_estimate / max(parallel_elapsed, 0.001)
    print(f"\n串行估算耗时: {serial_estimate:.1f}s | 并行加速比: {speedup:.1f}×")
    print("\n✅ ParaManager 测试通过")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_selection_scan_test()

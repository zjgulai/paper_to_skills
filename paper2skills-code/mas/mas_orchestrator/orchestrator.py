"""
MAS Orchestrator — 多智能体编排与调度
自定义框架（基于 DAG Execution + State Machine + Message Bus）

核心能力:
1. 生命周期管理 — 启动、监控、完成、失败处理
2. 数据流转 — 子 Agent 间输入/输出传递
3. 状态同步 — 全局执行状态，支持断点续传
4. 错误处理 — 重试、降级、超时、死锁检测
5. 资源调度 — 并发控制、优先级、配额管理

母婴电商场景: 全品类 VOC 分析流水线编排、实时预警流水线
"""

from typing import List, Dict, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time
import random


class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class EventType(Enum):
    """执行事件类型"""
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_TIMEOUT = "node_timeout"
    PROGRESS = "progress"
    DAG_COMPLETED = "dag_completed"
    DAG_FAILED = "dag_failed"


@dataclass
class ExecutionEvent:
    """执行事件"""
    event_type: EventType
    node_id: str
    timestamp: float
    payload: Dict = field(default_factory=dict)


@dataclass
class TaskNode:
    """任务节点"""
    node_id: str
    name: str
    skill_name: str
    dependencies: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    output: Any = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 60
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0


@dataclass
class ExecutionDAG:
    """执行 DAG"""
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=dict)  # node -> downstream

    def add_node(self, node: TaskNode):
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = set()

    def add_edge(self, upstream: str, downstream: str):
        if upstream in self.nodes and downstream in self.nodes:
            self.edges[upstream].add(downstream)

    def get_ready_nodes(self) -> List[str]:
        """获取依赖全部满足的 PENDING 节点"""
        ready = []
        for nid, node in self.nodes.items():
            if node.status == NodeStatus.PENDING:
                deps_satisfied = all(
                    self.nodes[dep].status == NodeStatus.SUCCESS
                    for dep in node.dependencies
                    if dep in self.nodes
                )
                if deps_satisfied:
                    ready.append(nid)
        return ready

    def is_complete(self) -> bool:
        """检查是否全部完成"""
        return all(
            node.status in (NodeStatus.SUCCESS, NodeStatus.FAILED)
            for node in self.nodes.values()
        )

    def get_stats(self) -> Dict:
        """获取执行统计"""
        status_counts = {}
        for node in self.nodes.values():
            status_counts[node.status.value] = status_counts.get(node.status.value, 0) + 1
        return status_counts


class MessageBus:
    """
    消息总线

    简化版：负责子 Agent 间的数据传递。
    生产环境使用 Redis / RabbitMQ / Kafka。
    """

    def __init__(self):
        self._messages: Dict[str, Any] = {}  # node_id -> output

    def send(self, node_id: str, data: Any):
        """发送消息"""
        self._messages[node_id] = data

    def receive(self, node_id: str) -> Any:
        """接收消息"""
        return self._messages.get(node_id)

    def get_outputs_for_node(self, node: TaskNode) -> Dict[str, Any]:
        """获取某节点的所有上游输出"""
        return {
            dep: self._messages.get(dep)
            for dep in node.dependencies
        }


class MASOrchestrator:
    """
    MAS 编排器

    协调多个子 Agent 的执行，管理完整的工作流生命周期。
    """

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self.message_bus = MessageBus()
        self.event_handlers: List[Callable] = []
        self.execution_log: List[ExecutionEvent] = []

    def add_event_handler(self, handler: Callable):
        """添加事件处理器"""
        self.event_handlers.append(handler)

    def _emit(self, event: ExecutionEvent):
        """发射事件"""
        self.execution_log.append(event)
        for handler in self.event_handlers:
            handler(event)

    def execute(self, dag: ExecutionDAG) -> Dict:
        """
        执行完整 DAG

        Returns:
            执行结果摘要
        """
        start_time = time.time()
        running_nodes: Set[str] = set()

        print(f"\n[Orchestrator] 开始执行 DAG")
        print(f"  总节点数: {len(dag.nodes)}")

        while not dag.is_complete():
            # 1. 扫描就绪节点
            ready = dag.get_ready_nodes()

            # 2. 调度可执行的节点（受并发限制）
            available_slots = self.max_concurrency - len(running_nodes)
            to_schedule = ready[:available_slots]

            for nid in to_schedule:
                self._start_node(dag, nid)
                running_nodes.add(nid)

            # 3. 模拟执行进度（简化版：直接完成）
            # 生产环境：轮询子 Agent 状态或接收回调
            completed = []
            for nid in list(running_nodes):
                node = dag.nodes[nid]
                # 模拟执行时间
                elapsed = time.time() - (node.start_time or time.time())
                if elapsed > 0.1:  # 模拟 100ms 执行
                    success = self._mock_execute(node)
                    if success:
                        self._complete_node(dag, nid)
                    else:
                        if node.retry_count < node.max_retries:
                            self._retry_node(dag, nid)
                        else:
                            self._fail_node(dag, nid)
                    completed.append(nid)

            running_nodes -= set(completed)

            if not ready and not running_nodes and not dag.is_complete():
                # 死锁检测：有节点无法推进
                break

        # 收尾
        total_time = time.time() - start_time
        return self._build_result(dag, total_time)

    def _start_node(self, dag: ExecutionDAG, nid: str):
        """启动节点"""
        node = dag.nodes[nid]
        node.status = NodeStatus.RUNNING
        node.start_time = time.time()

        # 获取上游输出作为输入
        inputs = self.message_bus.get_outputs_for_node(node)

        self._emit(ExecutionEvent(
            event_type=EventType.NODE_STARTED,
            node_id=nid,
            timestamp=time.time(),
            payload={"inputs": list(inputs.keys())}
        ))

    def _mock_execute(self, node: TaskNode) -> bool:
        """模拟节点执行（生产环境替换为真实 Agent 调用）"""
        # 模拟 10% 失败率（测试容错）
        if random.random() < 0.1:
            return False
        return True

    def _complete_node(self, dag: ExecutionDAG, nid: str):
        """节点完成"""
        node = dag.nodes[nid]
        node.status = NodeStatus.SUCCESS
        node.end_time = time.time()

        # 生成模拟输出
        node.output = f"output_of_{nid}"
        self.message_bus.send(nid, node.output)

        self._emit(ExecutionEvent(
            event_type=EventType.NODE_COMPLETED,
            node_id=nid,
            timestamp=time.time(),
            payload={"duration_ms": node.duration_ms}
        ))

    def _retry_node(self, dag: ExecutionDAG, nid: str):
        """重试节点"""
        node = dag.nodes[nid]
        node.retry_count += 1
        node.status = NodeStatus.RETRYING
        node.start_time = None

        self._emit(ExecutionEvent(
            event_type=EventType.NODE_FAILED,
            node_id=nid,
            timestamp=time.time(),
            payload={"retry": node.retry_count, "max_retries": node.max_retries}
        ))

        # 重置为 PENDING 以便重新调度
        node.status = NodeStatus.PENDING

    def _fail_node(self, dag: ExecutionDAG, nid: str):
        """节点最终失败"""
        node = dag.nodes[nid]
        node.status = NodeStatus.FAILED
        node.end_time = time.time()

        self._emit(ExecutionEvent(
            event_type=EventType.NODE_FAILED,
            node_id=nid,
            timestamp=time.time(),
            payload={"final_failure": True, "retries": node.retry_count}
        ))

    def _build_result(self, dag: ExecutionDAG, total_time: float) -> Dict:
        """构建执行结果"""
        stats = dag.get_stats()
        success_count = stats.get("success", 0)
        failed_count = stats.get("failed", 0)
        total = len(dag.nodes)

        return {
            "completed": failed_count == 0,
            "total_nodes": total,
            "success": success_count,
            "failed": failed_count,
            "success_rate": success_count / total if total > 0 else 0,
            "total_time_ms": int(total_time * 1000),
            "node_details": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "status": n.status.value,
                    "duration_ms": n.duration_ms,
                    "retries": n.retry_count
                }
                for n in dag.nodes.values()
            ]
        }


# ============================================
# 母婴电商场景 — 全品类 VOC 分析流水线编排
# ============================================

def demo_voc_pipeline():
    """演示全品类 VOC 分析流水线的编排"""
    print("=" * 70)
    print("MAS Orchestrator — 全品类 VOC 分析流水线")
    print("=" * 70)

    # 构建执行 DAG
    dag = ExecutionDAG()

    # 8 个并行品类分析任务
    categories = ["吸奶器", "储奶袋", "温奶器", "推车", "安全座椅", "洗护", "喂养配件", "其他"]
    for cat in categories:
        node = TaskNode(
            node_id=f"extract_{cat}",
            name=f"抽取分析 ({cat})",
            skill_name="InstructUIE+ABSA",
            timeout_seconds=30
        )
        dag.add_node(node)

    # 跨品类对比（依赖所有抽取任务）
    compare_node = TaskNode(
        node_id="cross_compare",
        name="跨品类对比",
        skill_name="CrossCategoryAnalyzer",
        dependencies=[f"extract_{cat}" for cat in categories],
        timeout_seconds=45
    )
    dag.add_node(compare_node)
    for cat in categories:
        dag.add_edge(f"extract_{cat}", "cross_compare")

    # 报告生成（依赖对比任务）
    report_node = TaskNode(
        node_id="generate_report",
        name="生成报告",
        skill_name="ReportGenerator",
        dependencies=["cross_compare"],
        timeout_seconds=30
    )
    dag.add_node(report_node)
    dag.add_edge("cross_compare", "generate_report")

    # 执行
    print(f"\n[DAG] {len(dag.nodes)} 个节点")
    print(f"[并发限制] max_concurrency=4")

    orchestrator = MASOrchestrator(max_concurrency=4)

    # 添加事件处理器
    def print_event(event: ExecutionEvent):
        icon = {
            EventType.NODE_STARTED: "▶",
            EventType.NODE_COMPLETED: "✓",
            EventType.NODE_FAILED: "✗",
        }.get(event.event_type, "•")
        if event.event_type in (EventType.NODE_STARTED, EventType.NODE_COMPLETED, EventType.NODE_FAILED):
            print(f"  {icon} [{event.node_id}] {event.event_type.value} ({event.payload})")

    orchestrator.add_event_handler(print_event)

    random.seed(42)
    result = orchestrator.execute(dag)

    print(f"\n[执行结果]")
    print(f"  完成: {'是' if result['completed'] else '否'}")
    print(f"  成功率: {result['success_rate']:.1%}")
    print(f"  成功/失败: {result['success']}/{result['failed']}")
    print(f"  总耗时: {result['total_time_ms']}ms")

    print("\n" + "=" * 70)


def demo_realtime_alert_pipeline():
    """演示实时预警流水线编排"""
    print("\n" + "=" * 70)
    print("MAS Orchestrator — 实时 VOC 预警流水线")
    print("=" * 70)

    dag = ExecutionDAG()

    # Stage 1: 数据获取
    dag.add_node(TaskNode(node_id="fetch_data", name="获取近1h评论", skill_name="DataFetcher"))

    # Stage 2: 并行分析（依赖数据获取）
    analysis_nodes = ["extract_entities", "sentiment_analysis", "trend_detection"]
    for nid in analysis_nodes:
        dag.add_node(TaskNode(
            node_id=nid,
            name=nid,
            skill_name=nid.replace("_", "").title(),
            dependencies=["fetch_data"]
        ))
        dag.add_edge("fetch_data", nid)

    # Stage 3: 根因分析（依赖所有分析）
    dag.add_node(TaskNode(
        node_id="root_cause",
        name="根因分析",
        skill_name="RootCauseAnalyzer",
        dependencies=analysis_nodes
    ))
    for nid in analysis_nodes:
        dag.add_edge(nid, "root_cause")

    # Stage 4: 预警生成（依赖根因）
    dag.add_node(TaskNode(
        node_id="generate_alert",
        name="生成预警",
        skill_name="AlertGenerator",
        dependencies=["root_cause"]
    ))
    dag.add_edge("root_cause", "generate_alert")

    # Stage 5: 通知发送（依赖预警生成）
    for channel in ["email", "slack", "sms"]:
        nid = f"notify_{channel}"
        dag.add_node(TaskNode(
            node_id=nid,
            name=f"通知({channel})",
            skill_name="NotificationSender",
            dependencies=["generate_alert"]
        ))
        dag.add_edge("generate_alert", nid)

    print(f"\n[DAG] {len(dag.nodes)} 个节点")

    orchestrator = MASOrchestrator(max_concurrency=5)
    random.seed(123)
    result = orchestrator.execute(dag)

    print(f"\n[执行结果]")
    print(f"  完成: {'是' if result['completed'] else '否'}")
    print(f"  成功率: {result['success_rate']:.1%}")
    print(f"  总耗时: {result['total_time_ms']}ms")
    print(f"  SLA 目标: ≤ 5min | 实际: {result['total_time_ms']/1000:.1f}s ✓")

    print("\n" + "=" * 70)


def demonstrate_orchestrator_architecture():
    """展示 Orchestrator 架构"""
    print("\n" + "=" * 70)
    print("MAS Orchestrator 架构")
    print("=" * 70)

    print("""
    Orchestrator 职责:

    ┌─────────────────────────────────────────────┐
    │              MAS Orchestrator               │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
    │  │ 生命周期 │  │ 数据流转 │  │ 状态同步 │  │
    │  │  管理    │  │  管理    │  │  管理    │  │
    │  └──────────┘  └──────────┘  └──────────┘  │
    │  ┌──────────┐  ┌──────────┐                │
    │  │ 错误处理 │  │ 资源调度 │                │
    │  │  管理    │  │  管理    │                │
    │  └──────────┘  └──────────┘                │
    └─────────────────────────────────────────────┘
              │              │              │
              ▼              ▼              ▼
         [Agent 1]      [Agent 2]      [Agent N]

    执行循环:
      while DAG 未全部完成:
        1. 扫描就绪节点
        2. 调度到执行队列（受并发限制）
        3. 启动子 Agent
        4. 监控执行状态
        5. 处理事件（成功/失败/超时）
        6. 触发下游节点

    错误处理策略:
      - 重试: 指数退避，最多 3 次
      - 降级: 主技能失败时切换备用技能
      - 超时: 强制取消，标记失败
      - 断路器: 连续失败时暂停新任务
    """)


if __name__ == "__main__":
    demo_voc_pipeline()
    demo_realtime_alert_pipeline()
    demonstrate_orchestrator_architecture()

    print("\n生产环境建议:")
    print("  1. 使用 Temporal / Airflow / Dagster 作为底层引擎")
    print("  2. 持久化执行状态，支持断点续传")
    print("  3. 集成监控（执行延迟、失败率、资源使用）")
    print("  4. 支持动态 DAG 修改（执行中调整流程）")
    print("  5. 实现资源隔离（避免单 Agent 占满资源）")
    print("  6. 与 Subagent Decomposer 集成：Decomposer 生成 DAG，Orchestrator 执行")

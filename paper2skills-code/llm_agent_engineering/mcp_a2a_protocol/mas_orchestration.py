"""Orchestrated MAS: MCP + A2A 双协议栈 + 三类 Agent + 四层编排.

参考论文:Adimulam, A. et al. (2026) The Orchestration of Multi-Agent Systems:
Architectures, Protocols, and Enterprise Adoption. arxiv:2601.13671.

本实现是简化版:
- MCP/A2A 用 in-memory 实现替代 JSON-RPC / HTTP, 保留协议语义
- 4 层 orchestration unit + 3 类 specialized agent
- 适配跨境母婴客服场景:Worker(检索/抽取) + Service(合规/质检) + Support(监控)

生产环境:
- MCP 接官方 SDK (JSON-RPC stdio/SSE/HTTP)
- A2A 接 Google A2A spec (RSA/Ed25519 签名)
- StateUnit 用 PostgreSQL + Redis
- QualityUnit 调专门 LLM-judge service
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


# Enums --------------------------------------------------------------------


class AgentRole(Enum):
    WORKER = "worker"
    SERVICE = "service"
    SUPPORT = "support"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# MCP (Model Context Protocol) --------------------------------------------


@dataclass
class MCPToolSchema:
    name: str
    description: str
    parameters: dict[str, str]
    returns: str


@dataclass
class MCPRequest:
    tool_name: str
    args: dict[str, Any]
    requester_id: str
    session_id: str


@dataclass
class MCPResponse:
    success: bool
    data: Any = None
    error: Optional[str] = None


class MCPServer:
    """暴露 tool 给 agent 调用. 简化版用函数注册替代 JSON-RPC."""

    def __init__(self, server_id: str) -> None:
        self.server_id = server_id
        self._tools: dict[str, tuple[MCPToolSchema, Callable[[dict[str, Any]], Any]]] = {}
        self._access_log: list[dict] = []

    def register_tool(self, schema: MCPToolSchema, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._tools[schema.name] = (schema, handler)

    def list_tools(self) -> list[MCPToolSchema]:
        return [schema for schema, _ in self._tools.values()]

    def invoke(self, request: MCPRequest) -> MCPResponse:
        if request.tool_name not in self._tools:
            return MCPResponse(success=False, error=f"Tool {request.tool_name} not found")
        schema, handler = self._tools[request.tool_name]
        # Schema 校验:必填参数都得有
        for param in schema.parameters:
            if param not in request.args:
                return MCPResponse(success=False, error=f"Missing param: {param}")
        try:
            result = handler(request.args)
            self._access_log.append({
                "tool": request.tool_name,
                "requester": request.requester_id,
                "session": request.session_id,
                "timestamp": datetime.now().isoformat(),
            })
            return MCPResponse(success=True, data=result)
        except Exception as exc:
            return MCPResponse(success=False, error=str(exc))

    def audit_log(self) -> list[dict]:
        return list(self._access_log)


class MCPClient:
    """Agent 通过 MCPClient 调用 MCP server 上的 tool."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def call(self, server: MCPServer, tool_name: str, args: dict[str, Any], session_id: str = "default") -> MCPResponse:
        return server.invoke(MCPRequest(
            tool_name=tool_name,
            args=args,
            requester_id=self.agent_id,
            session_id=session_id,
        ))


# A2A (Agent-to-Agent Protocol) -------------------------------------------


@dataclass
class A2AMessage:
    from_agent: str
    to_agent: str
    message_type: str  # "request" / "response" / "broadcast" / "negotiation"
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    signature: str = ""  # 简化版,生产用真实签名


class A2ARouter:
    """A2A peer-to-peer 通信路由器. 经 orchestrator supervise."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._message_log: list[A2AMessage] = []

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_id] = agent

    def send(self, message: A2AMessage) -> Optional[Any]:
        if message.to_agent not in self._agents:
            return None
        # 简化版签名:用 from + msg_type + hash
        message.signature = f"sig_{message.from_agent}_{message.message_type}"
        self._message_log.append(message)
        target = self._agents[message.to_agent]
        return target.handle_a2a(message)

    def broadcast(self, message: A2AMessage, exclude: Optional[set[str]] = None) -> dict[str, Any]:
        exclude = exclude or set()
        results = {}
        for agent_id, agent in self._agents.items():
            if agent_id == message.from_agent or agent_id in exclude:
                continue
            msg_copy = A2AMessage(
                from_agent=message.from_agent,
                to_agent=agent_id,
                message_type=message.message_type,
                payload=dict(message.payload),
                metadata=dict(message.metadata),
            )
            self._message_log.append(msg_copy)
            results[agent_id] = agent.handle_a2a(msg_copy)
        return results

    def message_log(self) -> list[A2AMessage]:
        return list(self._message_log)


# Base agent ---------------------------------------------------------------


@dataclass
class BaseAgent:
    agent_id: str
    role: AgentRole
    mcp_client: MCPClient
    a2a_router: A2ARouter

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def handle_a2a(self, message: A2AMessage) -> Any:
        # 默认:不响应,生产 agent 应 override
        return {"received": True, "agent": self.agent_id}


# Specialized agents (demo: 跨境母婴客服) ----------------------------------


class RetrievalAgent(BaseAgent):
    """Worker:KB 检索."""

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        # 通过 MCP 调外部 RAG 服务
        if "mcp_server" not in task:
            return {"status": "error", "msg": "no MCP server provided"}
        server: MCPServer = task["mcp_server"]
        response = self.mcp_client.call(server, "kb_search", {"query": task.get("query", "")})
        return {"status": "ok" if response.success else "fail", "results": response.data}


class ExtractionAgent(BaseAgent):
    """Worker:订单/物流数据抽取."""

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        text = task.get("text", "")
        # 简化抽取:从文本找 ORD 编号
        import re
        order_ids = re.findall(r"ORD\d+", text)
        batch_ids = re.findall(r"BATCH\d+", text)
        return {"status": "ok", "order_ids": order_ids, "batch_ids": batch_ids}


class ComplianceAgent(BaseAgent):
    """Service:各国合规审查."""

    PROHIBITED_TERMS = {"US": ["miracle cure"], "EU": ["100% safe"], "CN": ["最好的"]}

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        country = task.get("country", "US")
        text = task.get("text", "")
        violations = [t for t in self.PROHIBITED_TERMS.get(country, []) if t in text]
        return {"status": "violation" if violations else "ok", "violations": violations}

    def handle_a2a(self, message: A2AMessage) -> Any:
        if message.message_type == "request" and message.payload.get("action") == "validate":
            return self.execute(message.payload)
        return super().handle_a2a(message)


class QAAgent(BaseAgent):
    """Service:输出质量校验."""

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        output = task.get("output", "")
        # 简化:长度阈值 + 关键词检查
        ok = 10 <= len(output) <= 500 and any(kw in output for kw in ["建议", "请", "您"])
        return {"status": "ok" if ok else "fail", "output_length": len(output)}


class MonitorAgent(BaseAgent):
    """Support:监控 + 异常检测."""

    def __init__(self, agent_id: str, role: AgentRole, mcp_client: MCPClient, a2a_router: A2ARouter) -> None:
        super().__init__(agent_id, role, mcp_client, a2a_router)
        self.metrics: dict[str, list[float]] = defaultdict(list)

    def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        metric_name = task.get("metric")
        value = task.get("value", 0.0)
        if metric_name:
            self.metrics[metric_name].append(value)
            # 简单异常检测:最近 5 次平均与历史平均偏差 > 50%
            history = self.metrics[metric_name]
            if len(history) >= 5:
                recent = sum(history[-5:]) / 5
                overall = sum(history) / len(history)
                if overall > 0 and abs(recent - overall) / overall > 0.5:
                    return {"status": "anomaly", "metric": metric_name, "recent": recent, "overall": overall}
        return {"status": "ok"}


# Orchestration layer (4 units) -------------------------------------------


@dataclass
class Task:
    task_id: str
    description: str
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class PlanningUnit:
    """V-A Planning & Policy Management."""

    policy_rules: dict[str, Any] = field(default_factory=dict)

    def decompose(self, goal: str) -> list[Task]:
        # 简化:按 goal 类型生成固定任务序列
        if "客服" in goal or "customer" in goal.lower():
            return [
                Task("t1", "Retrieve KB", inputs={"query": goal}),
                Task("t2", "Extract order info", inputs={"text": goal}, dependencies=["t1"]),
                Task("t3", "Generate response", inputs={"goal": goal}, dependencies=["t2"]),
                Task("t4", "Compliance check", inputs={"country": "US"}, dependencies=["t3"]),
                Task("t5", "QA review", inputs={}, dependencies=["t4"]),
            ]
        return [Task("t1", goal)]


class ExecutionUnit:
    """V-B Execution & Control Management."""

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self.agents = agents
        self.telemetry: list[dict] = []

    def run(self, task: Task) -> Task:
        if task.assigned_to and task.assigned_to in self.agents:
            agent = self.agents[task.assigned_to]
            task.status = TaskStatus.RUNNING
            try:
                output = agent.execute(task.inputs)
                task.outputs = output
                task.status = TaskStatus.COMPLETED
            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.outputs = {"error": str(exc)}
            self.telemetry.append({
                "task_id": task.task_id,
                "agent": task.assigned_to,
                "status": task.status.value,
                "timestamp": datetime.now().isoformat(),
            })
        return task


@dataclass
class StateUnit:
    """V-C State & Knowledge Management."""

    workflow_state: dict[str, Task] = field(default_factory=dict)
    knowledge_base: dict[str, Any] = field(default_factory=dict)

    def checkpoint(self, task: Task) -> None:
        self.workflow_state[task.task_id] = task

    def get(self, task_id: str) -> Optional[Task]:
        return self.workflow_state.get(task_id)


class QualityUnit:
    """V-D Quality & Operations Management."""

    def __init__(self) -> None:
        self.violations: list[dict] = []

    def validate(self, task: Task, expected_schema: Optional[dict] = None) -> bool:
        if task.status != TaskStatus.COMPLETED:
            return False
        if expected_schema:
            for key in expected_schema:
                if key not in task.outputs:
                    self.violations.append({"task": task.task_id, "missing": key})
                    return False
        return True


# Master orchestrator ------------------------------------------------------


@dataclass
class MASOrchestrator:
    planning: PlanningUnit
    execution: ExecutionUnit
    state: StateUnit
    quality: QualityUnit
    a2a_router: A2ARouter

    def assign_default(self, tasks: list[Task]) -> None:
        """简化版分配:按任务名找 agent."""
        assignment = {
            "Retrieve KB": "retrieval_agent",
            "Extract order info": "extraction_agent",
            "Generate response": "extraction_agent",  # 默认 worker
            "Compliance check": "compliance_agent",
            "QA review": "qa_agent",
        }
        for task in tasks:
            task.assigned_to = assignment.get(task.description)

    def run_workflow(self, goal: str) -> dict[str, Any]:
        # 1. Planning
        tasks = self.planning.decompose(goal)
        self.assign_default(tasks)

        # 2. Execute by dependency order
        for task in tasks:
            # 满足依赖:等所有 deps 完成
            deps_ok = all(
                self.state.get(d) and self.state.get(d).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
                for d in task.dependencies
            )
            if not deps_ok:
                task.status = TaskStatus.FAILED
                self.state.checkpoint(task)
                continue

            # 3. Run
            task = self.execution.run(task)

            # 4. Validate
            self.quality.validate(task)

            # 5. Checkpoint
            self.state.checkpoint(task)

        # 6. Return summary
        return {
            "goal": goal,
            "total_tasks": len(tasks),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "telemetry": self.execution.telemetry,
            "a2a_messages": len(self.a2a_router.message_log()),
        }


# Demo ---------------------------------------------------------------------


def _setup_demo() -> MASOrchestrator:
    a2a_router = A2ARouter()

    def make_agent(cls, agent_id: str, role: AgentRole) -> BaseAgent:
        agent = cls(agent_id, role, MCPClient(agent_id), a2a_router)
        a2a_router.register(agent)
        return agent

    agents = {
        "retrieval_agent": make_agent(RetrievalAgent, "retrieval_agent", AgentRole.WORKER),
        "extraction_agent": make_agent(ExtractionAgent, "extraction_agent", AgentRole.WORKER),
        "compliance_agent": make_agent(ComplianceAgent, "compliance_agent", AgentRole.SERVICE),
        "qa_agent": make_agent(QAAgent, "qa_agent", AgentRole.SERVICE),
        "monitor_agent": make_agent(MonitorAgent, "monitor_agent", AgentRole.SUPPORT),
    }

    return MASOrchestrator(
        planning=PlanningUnit(),
        execution=ExecutionUnit(agents),
        state=StateUnit(),
        quality=QualityUnit(),
        a2a_router=a2a_router,
    )


def _setup_mcp_server() -> MCPServer:
    server = MCPServer("ecom_platform")
    server.register_tool(
        MCPToolSchema(
            name="kb_search",
            description="Search internal KB for product / policy info",
            parameters={"query": "string"},
            returns="list[KBEntry]",
        ),
        handler=lambda args: [
            {"title": "Diaper sizing guide", "score": 0.9},
            {"title": "Return policy for allergy", "score": 0.85},
        ],
    )
    server.register_tool(
        MCPToolSchema(
            name="order_lookup",
            description="Lookup order by ID",
            parameters={"order_id": "string"},
            returns="dict",
        ),
        handler=lambda args: {"order_id": args["order_id"], "status": "shipped", "batch": "BATCH4"},
    )
    return server


def main() -> None:
    print("=== Orchestrated MAS (MCP + A2A) Demo:跨境母婴客服 ===\n")
    orch = _setup_demo()

    # MCP server 暴露 RAG / 订单查询 tool
    mcp_server = _setup_mcp_server()
    print(f"MCP server '{mcp_server.server_id}' tools:")
    for tool in mcp_server.list_tools():
        print(f"  - {tool.name}: {tool.description}")

    # MCP 调用 demo
    print("\n--- MCP 调用示例 ---")
    client = MCPClient("ad_hoc_test")
    response = client.call(mcp_server, "order_lookup", {"order_id": "ORD1001"})
    print(f"  MCP call 结果: {response.data}")

    # A2A 通信 demo
    print("\n--- A2A 通信示例 ---")
    compliance_agent = orch.execution.agents["compliance_agent"]
    msg = A2AMessage(
        from_agent="retrieval_agent",
        to_agent="compliance_agent",
        message_type="request",
        payload={"action": "validate", "country": "EU", "text": "100% safe for babies"},
    )
    result = orch.a2a_router.send(msg)
    print(f"  A2A request → compliance_agent: {result}")

    # 完整工作流 demo
    print("\n--- 完整客服工作流 ---")
    summary = orch.run_workflow("跨境客服:客户咨询纸尿裤过敏退货, 订单 ORD1001 批次 BATCH4")
    print(f"总任务: {summary['total_tasks']}, 完成: {summary['completed']}, 失败: {summary['failed']}")
    print(f"A2A 消息数: {summary['a2a_messages']}")
    print(f"Telemetry:")
    for log in summary["telemetry"]:
        print(f"  [{log['timestamp'][-8:]}] {log['agent']:20s} → {log['status']}")


def test_pipeline() -> None:
    # MCP server / client
    server = _setup_mcp_server()
    assert len(server.list_tools()) == 2

    client = MCPClient("test_agent")
    response = client.call(server, "order_lookup", {"order_id": "ORD999"})
    assert response.success
    assert response.data["order_id"] == "ORD999"

    # Missing param 应失败
    bad_response = client.call(server, "order_lookup", {})
    assert not bad_response.success
    assert "Missing" in (bad_response.error or "")

    # 不存在的 tool 应失败
    no_tool = client.call(server, "nonexistent", {})
    assert not no_tool.success

    # MCP audit log
    assert len(server.audit_log()) == 1  # 只有 ORD999 成功

    # A2A router
    orch = _setup_demo()
    msg = A2AMessage(
        from_agent="extraction_agent",
        to_agent="compliance_agent",
        message_type="request",
        payload={"action": "validate", "country": "US", "text": "miracle cure for babies"},
    )
    result = orch.a2a_router.send(msg)
    assert result["status"] == "violation", f"应检出 US prohibited term, got {result}"

    # 不存在的 agent 返回 None
    bad_msg = A2AMessage("a", "nonexistent_agent", "request", {})
    assert orch.a2a_router.send(bad_msg) is None

    # 完整工作流
    summary = orch.run_workflow("跨境客服:客户咨询订单 ORD1001 退货")
    assert summary["total_tasks"] == 5, f"expected 5 tasks, got {summary['total_tasks']}"
    assert summary["completed"] >= 1, "至少应完成第一个任务"

    # Specialized agents 各司其职
    ret_agent = orch.execution.agents["retrieval_agent"]
    assert ret_agent.role == AgentRole.WORKER
    assert orch.execution.agents["compliance_agent"].role == AgentRole.SERVICE
    assert orch.execution.agents["monitor_agent"].role == AgentRole.SUPPORT

    print("[PASS] all assertions")


if __name__ == "__main__":
    test_pipeline()
    print()
    main()

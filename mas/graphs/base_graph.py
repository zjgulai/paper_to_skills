"""通用工作流图模板.

使用纯 Python 实现 StateGraph + RetryPolicy + recursion_limit + interrupt 语义,
不强依赖 langgraph 包(因为本机环境受限). 接口对齐 LangGraph 1.x,
迁移时只需 from langgraph.graph import StateGraph 即可平移.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from mas.state.schema import WorkflowContext


START = "__start__"
END = "__end__"


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 30.0
    jitter: bool = False
    retry_on: Tuple[type, ...] = (TimeoutError, ConnectionError, RuntimeError)


class GraphRecursionError(RuntimeError):
    pass


class WorkflowInterrupt(RuntimeError):
    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__("workflow interrupted, pending human approval")
        self.payload = payload


@dataclass
class _Node:
    name: str
    fn: Callable[[WorkflowContext], Dict[str, Any]]
    retry: Optional[RetryPolicy] = None


@dataclass
class _CondEdge:
    src: str
    router: Callable[[WorkflowContext], str]
    branches: Dict[str, str]


def merge_state(base: WorkflowContext, delta: Dict[str, Any]) -> WorkflowContext:
    new_state = dict(base)
    for key, value in delta.items():
        if key in {"messages", "skill_outputs"} and isinstance(value, list):
            new_state[key] = list(new_state.get(key, [])) + value
        elif key == "token_usage" and isinstance(value, int):
            new_state[key] = int(new_state.get("token_usage", 0)) + value
        else:
            new_state[key] = value
    return new_state  # type: ignore[return-value]


@dataclass
class StateGraph:
    nodes: Dict[str, _Node] = field(default_factory=dict)
    edges: Dict[str, str] = field(default_factory=dict)
    cond_edges: List[_CondEdge] = field(default_factory=list)
    entry: Optional[str] = None

    def add_node(
        self,
        name: str,
        fn: Callable[[WorkflowContext], Dict[str, Any]],
        retry: Optional[RetryPolicy] = None,
    ) -> None:
        if name in {START, END}:
            raise ValueError(f"reserved node name: {name}")
        self.nodes[name] = _Node(name=name, fn=fn, retry=retry)

    def add_edge(self, src: str, dst: str) -> None:
        if src == START:
            self.entry = dst
            return
        self.edges[src] = dst

    def add_conditional_edges(
        self,
        src: str,
        router: Callable[[WorkflowContext], str],
        branches: Dict[str, str],
    ) -> None:
        self.cond_edges.append(_CondEdge(src=src, router=router, branches=branches))

    def _next_node(self, current: str, state: WorkflowContext) -> Optional[str]:
        for ce in self.cond_edges:
            if ce.src == current:
                branch = ce.router(state)
                target = ce.branches.get(branch)
                if target is None:
                    raise KeyError(f"router for node {current} returned unknown branch {branch}")
                return None if target == END else target
        target = self.edges.get(current)
        if target is None:
            return None
        return None if target == END else target

    def _run_node_with_retry(self, node: _Node, state: WorkflowContext) -> Dict[str, Any]:
        policy = node.retry or RetryPolicy(max_attempts=1)
        attempt = 0
        wait = policy.initial_interval
        last_exc: Optional[BaseException] = None
        while attempt < policy.max_attempts:
            try:
                return node.fn(state)
            except WorkflowInterrupt:
                raise
            except policy.retry_on as exc:
                last_exc = exc
                attempt += 1
                if attempt >= policy.max_attempts:
                    break
                time.sleep(wait)
                wait = min(wait * policy.backoff_factor, policy.max_interval)
        assert last_exc is not None
        raise last_exc

    def invoke(
        self,
        initial_state: WorkflowContext,
        *,
        recursion_limit: int = 50,
    ) -> WorkflowContext:
        if self.entry is None:
            raise ValueError("graph has no entry node, did you forget add_edge(START, ...)?")

        state = dict(initial_state)
        current: Optional[str] = self.entry
        steps = 0

        while current is not None:
            if steps >= recursion_limit:
                raise GraphRecursionError(f"recursion_limit={recursion_limit} reached, last node={current}")
            steps += 1

            node = self.nodes.get(current)
            if node is None:
                raise KeyError(f"unknown node: {current}")

            delta = self._run_node_with_retry(node, state)  # type: ignore[arg-type]
            state = merge_state(state, delta)  # type: ignore[assignment]

            current = self._next_node(current, state)  # type: ignore[arg-type]

        return state  # type: ignore[return-value]


def build_workflow_graph(
    agent_name: str,
    agent_fn: Callable[[WorkflowContext], Dict[str, Any]],
    approval_fn: Callable[[WorkflowContext], Dict[str, Any]],
    execute_fn: Callable[[WorkflowContext], Dict[str, Any]],
    reject_fn: Callable[[WorkflowContext], Dict[str, Any]],
    orchestrator_fn: Optional[Callable[[WorkflowContext], Dict[str, Any]]] = None,
    retry_attempts: int = 3,
) -> StateGraph:
    from mas.agents.orchestrator import orchestrator_route, route_after_approval

    graph = StateGraph()
    graph.add_node("orchestrator", orchestrator_fn or orchestrator_route)
    graph.add_node(
        agent_name,
        agent_fn,
        retry=RetryPolicy(max_attempts=retry_attempts, initial_interval=0.5, backoff_factor=2.0),
    )
    graph.add_node("human_approval", approval_fn)
    graph.add_node("execute_action", execute_fn)
    graph.add_node("rejected", reject_fn)

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", agent_name)
    graph.add_edge(agent_name, "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"execute": "execute_action", "rejected": "rejected", "pending": "human_approval"},
    )
    graph.add_edge("execute_action", END)
    graph.add_edge("rejected", END)

    return graph

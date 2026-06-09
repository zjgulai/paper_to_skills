import time
import threading
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import random


class Priority(Enum):
    CRITICAL = 0
    NORMAL = 1
    BACKGROUND = 2


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class AgentTask:
    task_id: str
    agent_fn: Callable
    priority: Priority = Priority.NORMAL
    budget_tokens: int = 4000
    deadline_seconds: float = 60.0
    created_at: float = field(default_factory=time.time)
    mlfq_level: int = 0


@dataclass
class ContextLayer:
    session: Dict[str, Any] = field(default_factory=dict)
    task: Dict[str, Any] = field(default_factory=dict)
    agent: Dict[str, Any] = field(default_factory=dict)

    def clear_task(self):
        self.task.clear()

    def clear_agent(self):
        self.agent.clear()


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def record_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def is_open(self) -> bool:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.cooldown:
                self.state = CircuitState.HALF_OPEN
                return False
            return True
        return False


class HiveMindProxy:
    def __init__(self, max_concurrent: int = 5, rate_limit_per_min: int = 60):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit_per_min
        self._semaphore = threading.Semaphore(max_concurrent)
        self._request_times: deque = deque()
        self._circuit = CircuitBreaker()
        self._current_concurrency = max_concurrent
        self._lock = threading.Lock()

    def _check_rate_limit(self) -> bool:
        now = time.time()
        while self._request_times and self._request_times[0] < now - 60:
            self._request_times.popleft()
        return len(self._request_times) < self.rate_limit

    def _aimd_decrease(self):
        with self._lock:
            self._current_concurrency = max(1, self._current_concurrency // 2)

    def _aimd_increase(self):
        with self._lock:
            self._current_concurrency = min(self.max_concurrent, self._current_concurrency + 1)

    def execute(self, fn: Callable, priority: Priority = Priority.NORMAL,
                token_budget: int = 4000) -> Any:
        if self._circuit.is_open():
            raise RuntimeError("Circuit breaker OPEN")
        if not self._check_rate_limit():
            time.sleep(60.0 / max(self.rate_limit, 1))
        with self._semaphore:
            try:
                result = fn()
                self._request_times.append(time.time())
                self._circuit.record_success()
                self._aimd_increase()
                return result
            except Exception:
                self._circuit.record_failure()
                self._aimd_decrease()
                raise


class MLFQScheduler:
    def __init__(self, levels: int = 3, starvation_threshold: float = 30.0):
        self.levels = levels
        self.starvation_threshold = starvation_threshold
        self.queues: List[deque] = [deque() for _ in range(levels)]
        self._lock = threading.Lock()

    def enqueue(self, task: AgentTask):
        with self._lock:
            level = min(task.priority.value, self.levels - 1)
            self.queues[level].append(task)

    def dequeue(self) -> Optional[AgentTask]:
        with self._lock:
            self._promote_starving()
            for q in self.queues:
                if q:
                    return q.popleft()
        return None

    def _promote_starving(self):
        now = time.time()
        for level in range(1, self.levels):
            remaining = deque()
            while self.queues[level]:
                task = self.queues[level].popleft()
                if now - task.created_at > self.starvation_threshold:
                    task.mlfq_level = 0
                    self.queues[0].append(task)
                else:
                    remaining.append(task)
            self.queues[level] = remaining

    def demote(self, task: AgentTask):
        with self._lock:
            new_level = min(task.mlfq_level + 1, self.levels - 1)
            task.mlfq_level = new_level
            self.queues[new_level].append(task)

    def pending_count(self) -> int:
        return sum(len(q) for q in self.queues)


class AgentContextManager:
    def __init__(self):
        self._contexts: Dict[str, ContextLayer] = {}

    def get_or_create(self, session_id: str) -> ContextLayer:
        if session_id not in self._contexts:
            self._contexts[session_id] = ContextLayer()
        return self._contexts[session_id]

    def begin_task(self, session_id: str) -> ContextLayer:
        ctx = self.get_or_create(session_id)
        ctx.clear_task()
        return ctx

    def end_agent(self, session_id: str):
        if session_id in self._contexts:
            self._contexts[session_id].clear_agent()

    def end_task(self, session_id: str):
        if session_id in self._contexts:
            self._contexts[session_id].clear_task()
            self._contexts[session_id].clear_agent()

    def end_session(self, session_id: str):
        self._contexts.pop(session_id, None)


class MCPPPlanner:
    def __init__(self, n_simulations: int = 200):
        self.n_simulations = n_simulations

    def plan(self, tasks: List[AgentTask], budget: float, deadline: float) -> Dict[str, Any]:
        results = []
        for _ in range(self.n_simulations):
            sim_cost = sum(
                random.gauss(t.budget_tokens * 0.002, t.budget_tokens * 0.0005)
                for t in tasks
            )
            sim_latency = sum(
                random.gauss(t.deadline_seconds * 0.4, t.deadline_seconds * 0.1)
                for t in tasks
            )
            results.append((sim_cost, sim_latency))

        both = sum(1 for c, l in results if c <= budget and l <= deadline)
        completion_rate = both / self.n_simulations
        critical = [t for t in tasks if t.priority == Priority.CRITICAL]
        normal = [t for t in tasks if t.priority == Priority.NORMAL]
        background = [t for t in tasks if t.priority == Priority.BACKGROUND]

        return {
            "completion_rate": round(completion_rate, 3),
            "p_within_budget": round(sum(1 for c, _ in results if c <= budget) / self.n_simulations, 3),
            "p_within_deadline": round(sum(1 for _, l in results if l <= deadline) / self.n_simulations, 3),
            "recommended_order": [t.task_id for t in critical + normal + background],
            "feasible": completion_rate >= 0.8,
        }


def test_hivemind_circuit_breaker():
    proxy = HiveMindProxy(max_concurrent=3, rate_limit_per_min=100)
    for _ in range(5):
        try:
            proxy.execute(lambda: (_ for _ in ()).throw(ConnectionError("429")))
        except Exception:
            pass
    assert proxy._circuit.state == CircuitState.OPEN
    print(f"[PASS] circuit_breaker: state={proxy._circuit.state.value} after 5 failures")


def test_hivemind_success_resets_circuit():
    proxy = HiveMindProxy(max_concurrent=3, rate_limit_per_min=100)
    proxy._circuit.record_failure()
    proxy._circuit.record_success()
    assert proxy._circuit.state == CircuitState.CLOSED
    print("[PASS] circuit_reset: success resets circuit to CLOSED")


def test_mlfq_priority_order():
    scheduler = MLFQScheduler()
    scheduler.enqueue(AgentTask("bg1", lambda: None, Priority.BACKGROUND))
    scheduler.enqueue(AgentTask("crit1", lambda: None, Priority.CRITICAL))
    scheduler.enqueue(AgentTask("norm1", lambda: None, Priority.NORMAL))
    first = scheduler.dequeue()
    assert first.task_id == "crit1", f"Expected crit1, got {first.task_id}"
    second = scheduler.dequeue()
    assert second.task_id == "norm1"
    print("[PASS] mlfq_priority: CRITICAL dequeued before NORMAL before BACKGROUND")


def test_context_manager_task_isolation():
    mgr = AgentContextManager()
    ctx = mgr.begin_task("sess_001")
    ctx.task["price"] = 45.0
    ctx.agent["temp"] = "working"
    assert ctx.task["price"] == 45.0

    mgr.end_task("sess_001")
    ctx2 = mgr.begin_task("sess_001")
    assert "price" not in ctx2.task, "Task context should be cleared between tasks"
    assert "temp" not in ctx2.agent
    print("[PASS] context_isolation: task context cleared between tasks")


def test_mcpp_feasibility():
    random.seed(42)
    tasks = [
        AgentTask("compliance", lambda: None, Priority.CRITICAL, budget_tokens=2000, deadline_seconds=10),
        AgentTask("trend", lambda: None, Priority.NORMAL, budget_tokens=3000, deadline_seconds=15),
        AgentTask("report", lambda: None, Priority.BACKGROUND, budget_tokens=1000, deadline_seconds=20),
    ]
    planner = MCPPPlanner(n_simulations=500)
    plan = planner.plan(tasks, budget=20.0, deadline=30.0)
    assert 0.0 <= plan["completion_rate"] <= 1.0
    assert plan["recommended_order"][0] == "compliance"
    print(f"[PASS] mcpp_plan: completion_rate={plan['completion_rate']}, feasible={plan['feasible']}")


if __name__ == "__main__":
    test_hivemind_circuit_breaker()
    test_hivemind_success_resets_circuit()
    test_mlfq_priority_order()
    test_context_manager_task_isolation()
    test_mcpp_feasibility()
    print("\n✅ All tests passed")

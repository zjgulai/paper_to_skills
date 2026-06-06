import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class ServiceType(Enum):
    MODEL = "model"
    AGENT = "agent"
    ENVIRONMENT = "environment"


@dataclass
class AgentCapability:
    agent_id: str
    task_success_counts: Dict[str, int] = field(default_factory=dict)
    task_total_counts: Dict[str, int] = field(default_factory=dict)
    is_familiarized: bool = False
    joined_at: float = field(default_factory=time.time)

    def success_rate(self, task_type: str) -> float:
        total = self.task_total_counts.get(task_type, 0)
        return self.task_success_counts.get(task_type, 0) / total if total else 0.0

    def total_tasks(self) -> int:
        return sum(self.task_total_counts.values())

    def update(self, task_type: str, success: bool):
        self.task_total_counts[task_type] = self.task_total_counts.get(task_type, 0) + 1
        if success:
            self.task_success_counts[task_type] = self.task_success_counts.get(task_type, 0) + 1


class MonoScaleRouter:
    def __init__(self, exploration_coeff: float = 1.0):
        self.beta = exploration_coeff
        self.agents: Dict[str, AgentCapability] = {}
        self.t: int = 0

    def register(self, agent_id: str, familiarization_tasks: Optional[List[Dict]] = None):
        cap = AgentCapability(agent_id=agent_id)
        if familiarization_tasks:
            for task in familiarization_tasks:
                cap.update(task["task_type"], task["success"])
            cap.is_familiarized = True
        self.agents[agent_id] = cap

    def route(self, task_type: str) -> Optional[str]:
        if not self.agents:
            return None
        self.t += 1
        best_agent, best_score = None, -1.0
        for agent_id, cap in self.agents.items():
            q = cap.success_rate(task_type)
            n = cap.task_total_counts.get(task_type, 0)
            score = q + self.beta * math.sqrt(math.log(self.t + 1) / (n + 1))
            if score > best_score:
                best_score, best_agent = score, agent_id
        return best_agent

    def record_outcome(self, agent_id: str, task_type: str, success: bool):
        if agent_id in self.agents:
            self.agents[agent_id].update(task_type, success)


class MegaFlowService:
    def __init__(self):
        self._model_handlers: Dict[str, Callable] = {}
        self._env_handlers: Dict[str, Callable] = {}
        self._task_queue: List[Dict] = []

    def register_model(self, model_name: str, handler: Callable):
        self._model_handlers[model_name] = handler

    def register_env_tool(self, tool_name: str, handler: Callable):
        self._env_handlers[tool_name] = handler

    def model_call(self, model_name: str, prompt: str) -> Any:
        handler = self._model_handlers.get(model_name)
        if not handler:
            raise ValueError(f"Model '{model_name}' not registered")
        return handler(prompt)

    def env_call(self, tool_name: str, **kwargs) -> Any:
        handler = self._env_handlers.get(tool_name)
        if not handler:
            raise ValueError(f"Tool '{tool_name}' not registered")
        return handler(**kwargs)

    def submit_agent_task(self, task_id: str, task_type: str, payload: dict) -> str:
        self._task_queue.append({"task_id": task_id, "task_type": task_type, "payload": payload})
        return task_id

    def pending_tasks(self) -> int:
        return len(self._task_queue)


class OrgAgentSystem:
    def __init__(self):
        self._governance: Optional[Callable] = None
        self._departments: Dict[str, Callable] = {}
        self._compliance: Optional[Callable] = None
        self._audit_log: List[Dict] = []

    def set_governance(self, agent_fn: Callable):
        self._governance = agent_fn

    def add_department(self, name: str, agent_fn: Callable):
        self._departments[name] = agent_fn

    def set_compliance(self, agent_fn: Callable):
        self._compliance = agent_fn

    def execute(self, user_request: str) -> Dict[str, Any]:
        if not self._governance:
            raise RuntimeError("Governance agent not set")
        plan = self._governance(user_request)
        results = {}
        for task in plan.get("tasks", []):
            dept = task.get("department")
            handler = self._departments.get(dept)
            if not handler:
                results[dept] = {"error": f"No department '{dept}'"}
                continue
            result = handler(task)
            results[dept] = result
            self._audit_log.append({"dept": dept, "task": task, "result": result, "ts": time.time()})
            if self._compliance:
                cr = self._compliance({"dept": dept, "result": result})
                if not cr.get("compliant", True):
                    results["compliance_flag"] = cr
        return {"plan": plan, "results": results}

    def get_audit_log(self) -> List[Dict]:
        return list(self._audit_log)


def test_monoscale_prefers_expert():
    router = MonoScaleRouter(exploration_coeff=0.5)
    router.register("expert_agent", familiarization_tasks=[
            {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
        {"task_type": "supplier_eval", "success": True},
    ])
    router.register("new_agent")

    chosen_counts = {"expert_agent": 0, "new_agent": 0}
    for _ in range(50):
        chosen = router.route("supplier_eval")
        chosen_counts[chosen] += 1
        router.record_outcome(chosen, "supplier_eval", True)

    assert chosen_counts["expert_agent"] > chosen_counts["new_agent"], \
        f"Expert should win: {chosen_counts}"
    print(f"[PASS] monoscale_routing: expert={chosen_counts['expert_agent']}, new={chosen_counts['new_agent']}")


def test_monoscale_explores_new_agent():
    router = MonoScaleRouter(exploration_coeff=2.0)
    router.register("agent_a", familiarization_tasks=[{"task_type": "t1", "success": True}] * 5)
    router.register("agent_b")

    chosen = {router.route("t1") for _ in range(10)}
    assert "agent_b" in chosen, "New agent should be explored"
    print("[PASS] monoscale_exploration: new agent gets explored")


def test_megaflow_service_isolation():
    svc = MegaFlowService()
    svc.register_model("gpt4", lambda prompt: f"response:{prompt}")
    svc.register_env_tool("db_query", lambda query: {"rows": [1, 2, 3]})

    model_result = svc.model_call("gpt4", "hello")
    assert model_result == "response:hello"

    env_result = svc.env_call("db_query", query="SELECT 1")
    assert env_result["rows"] == [1, 2, 3]

    try:
        svc.model_call("unknown_model", "test")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print("[PASS] megaflow_isolation: model/env services isolated correctly")


def test_orgagent_three_layers():
    org = OrgAgentSystem()
    org.set_governance(lambda req: {
        "goal": req,
        "tasks": [
            {"department": "ops", "action": "check_inventory"},
            {"department": "marketing", "action": "run_ad"},
        ]
    })
    org.add_department("ops", lambda task: {"status": "ok", "inventory": 500})
    org.add_department("marketing", lambda task: {"status": "ok", "impressions": 10000})
    org.set_compliance(lambda ctx: {"compliant": True})

    result = org.execute("prepare for 11.11 sale")
    assert "ops" in result["results"]
    assert "marketing" in result["results"]
    assert result["results"]["ops"]["inventory"] == 500
    assert len(org.get_audit_log()) == 2
    print(f"[PASS] orgagent_three_layers: {len(result['results'])} depts executed, {len(org.get_audit_log())} audit entries")


def test_orgagent_compliance_flag():
    org = OrgAgentSystem()
    org.set_governance(lambda req: {"tasks": [{"department": "risky_dept", "action": "risky_op"}]})
    org.add_department("risky_dept", lambda task: {"action": "done", "risk": "high"})
    org.set_compliance(lambda ctx: {"compliant": False, "reason": "high risk detected"})

    result = org.execute("risky operation")
    assert "compliance_flag" in result["results"]
    print(f"[PASS] compliance_flag: detected high-risk operation")


if __name__ == "__main__":
    test_monoscale_prefers_expert()
    test_monoscale_explores_new_agent()
    test_megaflow_service_isolation()
    test_orgagent_three_layers()
    test_orgagent_compliance_flag()
    print("\n✅ All tests passed")

import time
import uuid
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


class FailureType(Enum):
    INFINITE_LOOP = "infinite_loop"
    TOOL_CALL_FAILURE = "tool_call_failure"
    AGENT_DEPENDENCY_FAILURE = "agent_dependency_failure"
    SEMANTIC_ERROR = "semantic_error"
    TIMEOUT = "timeout"
    NULL_OUTPUT = "null_output"


@dataclass
class AgentSpec:
    agent_id: str
    required_output_keys: List[str]
    output_value_constraints: Dict[str, Tuple[Any, Any]]
    max_retries: int = 3
    timeout_seconds: float = 30.0


@dataclass
class FailureRecord:
    failure_type: FailureType
    agent_id: str
    message: str
    test_input: dict
    severity: str = "medium"


@dataclass
class AgentCall:
    agent_id: str
    input_data: dict
    output_data: Optional[dict]
    latency_ms: float
    success: bool
    tool_calls: List[str] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class ExecutionTrace:
    trace_id: str
    task: str
    agent_calls: List[AgentCall]
    total_latency_ms: float
    total_tokens: int
    success: bool
    failures: List[FailureRecord] = field(default_factory=list)

    @property
    def agent_call_graph(self) -> Dict[str, List[str]]:
        calls = [c.agent_id for c in self.agent_calls]
        return {calls[i]: [calls[i+1]] for i in range(len(calls)-1)}


class FLAREOracle:
    def __init__(self, specs: Dict[str, AgentSpec], loop_detection_threshold: int = 5):
        self.specs = specs
        self.loop_threshold = loop_detection_threshold
        self._state_history: Dict[str, List[str]] = {}

    def check_output(self, agent_id: str, output: Optional[dict]) -> List[FailureRecord]:
        failures = []
        if agent_id not in self.specs:
            return failures
        spec = self.specs[agent_id]

        if output is None:
            failures.append(FailureRecord(
                FailureType.NULL_OUTPUT, agent_id,
                f"{agent_id} returned None", {}, "critical"
            ))
            return failures

        for key in spec.required_output_keys:
            if key not in output:
                failures.append(FailureRecord(
                    FailureType.SEMANTIC_ERROR, agent_id,
                    f"Missing required key '{key}' in output", output, "high"
                ))

        for key, (lo, hi) in spec.output_value_constraints.items():
            if key in output and isinstance(output[key], (int, float)):
                val = output[key]
                if not (lo <= val <= hi):
                    failures.append(FailureRecord(
                        FailureType.SEMANTIC_ERROR, agent_id,
                        f"'{key}'={val} out of range [{lo}, {hi}]", output, "high"
                    ))
        return failures

    def check_loop(self, agent_id: str, state_signature: str) -> bool:
        history = self._state_history.setdefault(agent_id, [])
        history.append(state_signature)
        return history.count(state_signature) >= self.loop_threshold

    def reset(self):
        self._state_history.clear()


class FLAREMutator:
    BOUNDARY_MUTATIONS = {
        "string": ["", " " * 100, "a" * 10000, None, "!@#$%^&*()"],
        "number": [0, -1, -0.001, float("inf"), float("nan"), 999999],
        "list": [[], [None], [""] * 100],
    }

    def mutate(self, base_input: dict, mutation_rate: float = 0.3) -> List[dict]:
        mutated = []
        for key, val in base_input.items():
            if random.random() > mutation_rate:
                continue
            val_type = (
                "string" if isinstance(val, str) else
                "number" if isinstance(val, (int, float)) else
                "list" if isinstance(val, list) else None
            )
            if val_type and val_type in self.BOUNDARY_MUTATIONS:
                for mutation in self.BOUNDARY_MUTATIONS[val_type]:
                    new_input = dict(base_input)
                    new_input[key] = mutation
                    mutated.append(new_input)
        return mutated if mutated else [base_input]


class MAESTROTracer:
    def __init__(self):
        self.traces: List[ExecutionTrace] = []

    def new_trace(self, task: str) -> str:
        trace_id = str(uuid.uuid4())[:8]
        self.traces.append(ExecutionTrace(
            trace_id=trace_id, task=task,
            agent_calls=[], total_latency_ms=0.0,
            total_tokens=0, success=False
        ))
        return trace_id

    def record_agent_call(self, trace_id: str, agent_id: str,
                          input_data: dict, output_data: Optional[dict],
                          latency_ms: float, success: bool,
                          tool_calls: Optional[List[str]] = None,
                          tokens: int = 0, retry_count: int = 0):
        trace = self._get(trace_id)
        if not trace:
            return
        trace.agent_calls.append(AgentCall(
            agent_id=agent_id, input_data=input_data,
            output_data=output_data, latency_ms=latency_ms,
            success=success, tool_calls=tool_calls or [],
            retry_count=retry_count,
        ))
        trace.total_latency_ms += latency_ms
        trace.total_tokens += tokens

    def finalize(self, trace_id: str, success: bool,
                 failures: Optional[List[FailureRecord]] = None):
        trace = self._get(trace_id)
        if trace:
            trace.success = success
            trace.failures = failures or []

    def _get(self, trace_id: str) -> Optional[ExecutionTrace]:
        return next((t for t in self.traces if t.trace_id == trace_id), None)

    def aggregate_stats(self) -> Dict[str, Any]:
        if not self.traces:
            return {}
        latencies = sorted(t.total_latency_ms for t in self.traces)
        tokens = [t.total_tokens for t in self.traces]
        n = len(latencies)
        return {
            "total_runs": n,
            "success_rate": round(sum(1 for t in self.traces if t.success) / n, 3),
            "latency_p50_ms": latencies[int(n * 0.5)],
            "latency_p95_ms": latencies[min(int(n * 0.95), n - 1)],
            "latency_mean_ms": round(sum(latencies) / n, 1),
            "tokens_mean": round(sum(tokens) / n, 1),
            "total_failures": sum(len(t.failures) for t in self.traces),
            "failure_types": self._count_failure_types(),
        }

    def _count_failure_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for trace in self.traces:
            for f in trace.failures:
                counts[f.failure_type.value] = counts.get(f.failure_type.value, 0) + 1
        return counts

    def compare_frameworks(self, other: "MAESTROTracer") -> Dict[str, Any]:
        sa, sb = self.aggregate_stats(), other.aggregate_stats()
        if not sa or not sb:
            return {}
        return {
            "latency_p50_ratio": round(sb["latency_p50_ms"] / max(sa["latency_p50_ms"], 1), 3),
            "latency_p95_ratio": round(sb["latency_p95_ms"] / max(sa["latency_p95_ms"], 1), 3),
            "success_rate_delta": round(sb["success_rate"] - sa["success_rate"], 3),
            "token_efficiency_ratio": round(sb["tokens_mean"] / max(sa["tokens_mean"], 1), 3),
            "framework_a": sa,
            "framework_b": sb,
        }


class MASTestRunner:
    def __init__(self, agent_specs: Dict[str, AgentSpec]):
        self.oracle = FLAREOracle(agent_specs)
        self.mutator = FLAREMutator()
        self.tracer = MAESTROTracer()

    def fuzz_test(self, agent_runner: Callable[[str, dict], Optional[dict]],
                  agent_id: str, base_inputs: List[dict],
                  n_mutations: int = 20) -> List[FailureRecord]:
        all_failures = []
        test_cases = list(base_inputs)
        for base in base_inputs:
            test_cases.extend(self.mutator.mutate(base)[:n_mutations])

        for test_input in test_cases:
            try:
                output = agent_runner(agent_id, test_input)
                if self.oracle.check_loop(agent_id, str(sorted(str(output)))):
                    all_failures.append(FailureRecord(
                        FailureType.INFINITE_LOOP, agent_id,
                        "Repeated state detected", test_input, "critical"
                    ))
                all_failures.extend(self.oracle.check_output(agent_id, output))
            except Exception as e:
                all_failures.append(FailureRecord(
                    FailureType.TOOL_CALL_FAILURE, agent_id, str(e), test_input, "high"
                ))
        self.oracle.reset()
        return all_failures

    def run_with_tracing(self, pipeline: List[Tuple[str, Callable[[dict], Optional[dict]]]],
                         initial_input: dict, task: str = "unnamed_task") -> ExecutionTrace:
        trace_id = self.tracer.new_trace(task)
        current_input = initial_input
        all_failures = []
        overall_success = True

        for agent_id, agent_fn in pipeline:
            start = time.perf_counter()
            output = None
            success = False
            try:
                output = agent_fn(current_input)
                success = output is not None
                spec_failures = self.oracle.check_output(agent_id, output)
                all_failures.extend(spec_failures)
                if spec_failures:
                    overall_success = False
            except Exception as e:
                all_failures.append(FailureRecord(
                    FailureType.TOOL_CALL_FAILURE, agent_id, str(e), current_input
                ))
                overall_success = False

            latency = (time.perf_counter() - start) * 1000
            self.tracer.record_agent_call(trace_id, agent_id, current_input, output, latency, success)
            if output:
                current_input = output

        self.tracer.finalize(trace_id, overall_success, all_failures)
        return self.tracer._get(trace_id)


def test_flare_oracle_detects_missing_key():
    specs = {"trend_agent": AgentSpec(
        "trend_agent", ["trend_score", "top_keywords"], {"trend_score": (0.0, 1.0)}
    )}
    oracle = FLAREOracle(specs)
    failures = oracle.check_output("trend_agent", {"trend_score": 0.8})
    assert any(f.failure_type == FailureType.SEMANTIC_ERROR for f in failures)
    assert any("top_keywords" in f.message for f in failures)
    print(f"[PASS] oracle_missing_key: {len(failures)} failure(s) detected")


def test_flare_oracle_detects_value_violation():
    specs = {"profit_agent": AgentSpec("profit_agent", ["margin_rate"], {"margin_rate": (0.0, 1.0)})}
    oracle = FLAREOracle(specs)
    failures = oracle.check_output("profit_agent", {"margin_rate": -0.5})
    assert any(f.failure_type == FailureType.SEMANTIC_ERROR for f in failures)
    print("[PASS] oracle_value_violation: margin_rate=-0.5 correctly flagged")


def test_flare_loop_detection():
    specs = {"loop_agent": AgentSpec("loop_agent", [], {})}
    oracle = FLAREOracle(specs, loop_detection_threshold=3)
    for _ in range(2):
        assert not oracle.check_loop("loop_agent", "state_xyz")
    assert oracle.check_loop("loop_agent", "state_xyz")
    print("[PASS] loop_detection: 3-repeat state correctly flagged")


def test_mutator_generates_boundary_cases():
    mutator = FLAREMutator()
    base = {"query": "baby sterilizer", "price": 45.0}
    mutations = mutator.mutate(base, mutation_rate=1.0)
    assert len(mutations) > 0
    has_empty = any(m.get("query") == "" for m in mutations)
    has_boundary = any(m.get("price") in [0, -1, float("inf")] for m in mutations)
    assert has_empty or has_boundary
    print(f"[PASS] mutator: {len(mutations)} boundary cases generated")


def test_maestro_aggregate_stats():
    tracer = MAESTROTracer()
    for i in range(10):
        tid = tracer.new_trace(f"task_{i}")
        tracer.record_agent_call(tid, "agent_a", {}, {"result": i}, 100.0 + i * 10, True, tokens=300)
        tracer.finalize(tid, success=(i % 5 != 0))
    stats = tracer.aggregate_stats()
    assert stats["total_runs"] == 10
    assert 0.0 < stats["success_rate"] < 1.0
    print(f"[PASS] maestro_stats: success_rate={stats['success_rate']}, p50={stats['latency_p50_ms']}ms")


def test_end_to_end_pipeline():
    specs = {
        "trend": AgentSpec("trend", ["trend_score"], {"trend_score": (0.0, 1.0)}),
        "profit": AgentSpec("profit", ["margin_rate"], {"margin_rate": (0.0, 1.0)}),
    }
    runner = MASTestRunner(specs)
    trace = runner.run_with_tracing(
        pipeline=[
            ("trend", lambda inp: {"trend_score": 0.75, "category": inp.get("query", "")}),
            ("profit", lambda inp: {"margin_rate": 0.35}),
        ],
        initial_input={"query": "baby sterilizer"},
        task="selection_scan",
    )
    assert trace.success
    assert len(trace.agent_calls) == 2
    print(f"[PASS] e2e_pipeline: {len(trace.agent_calls)} agents, {trace.total_latency_ms:.1f}ms")


if __name__ == "__main__":
    random.seed(42)
    test_flare_oracle_detects_missing_key()
    test_flare_oracle_detects_value_violation()
    test_flare_loop_detection()
    test_mutator_generates_boundary_cases()
    test_maestro_aggregate_stats()
    test_end_to_end_pipeline()
    print("\n✅ All tests passed")

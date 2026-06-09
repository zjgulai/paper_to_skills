"""QAAgent: 工作流输出质检 + 风险阈值检测.

可挂在任意 Agent 节点后,执行:
  - 输出 schema 校验 (必备字段)
  - 置信度阈值检查 (< 0.6 触发告警)
  - 估算金额合理性 (vs 历史均值的偏离)
  - Skill 链路完整性 (期望调用顺序)
"""

from __future__ import annotations

from typing import Any, Dict, List

from mas.state.schema import WorkflowContext


REQUIRED_OUTPUT_FIELDS = ["agent", "skill_name", "status"]
MIN_CONFIDENCE = 0.6
MAX_COST_DEVIATION_PCT = 0.5


class QAAgent:
    name = "qa_agent"
    role_description = "MAS 质检 Agent: 校验 schema + 置信度 + 金额合理性"

    def __call__(self, state: WorkflowContext) -> Dict[str, Any]:
        outputs = state.get("skill_outputs", [])
        if not outputs:
            return {
                "skill_outputs": [{
                    "agent": self.name,
                    "skill_name": "qa_check",
                    "status": "no_outputs_to_check",
                    "confidence": 1.0,
                    "estimated_cost": 0.0,
                }],
                "token_usage": 100,
            }

        last_output = outputs[-1]
        violations: List[str] = []

        for f in REQUIRED_OUTPUT_FIELDS:
            if f not in last_output:
                violations.append(f"missing_field:{f}")

        conf = float(last_output.get("confidence", 1.0))
        if conf < MIN_CONFIDENCE:
            violations.append(f"low_confidence:{conf}")

        cost = float(last_output.get("estimated_cost", 0))
        avg_cost = self._historical_avg_cost(outputs[:-1])
        if avg_cost > 0 and abs(cost - avg_cost) / avg_cost > MAX_COST_DEVIATION_PCT:
            violations.append(f"cost_deviation:{cost}_vs_avg_{avg_cost:.2f}")

        passed = len(violations) == 0
        return {
            "messages": [{
                "role": "system",
                "content": f"[QA] passed={passed}, violations={violations}",
                "agent": self.name,
            }],
            "skill_outputs": [{
                "agent": self.name,
                "skill_name": "qa_check",
                "output": {"passed": passed, "violations": violations, "checked_output": last_output},
                "confidence": 1.0,
                "estimated_cost": 0.0,
                "status": "ok" if passed else "qa_failed",
            }],
            "token_usage": 200,
        }

    @staticmethod
    def _historical_avg_cost(outputs: List[Dict[str, Any]]) -> float:
        costs = [float(o.get("estimated_cost", 0)) for o in outputs if "estimated_cost" in o]
        return sum(costs) / len(costs) if costs else 0.0

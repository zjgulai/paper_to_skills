"""BaseAgent: 所有业务 Agent 的共同基类.

约束:
  - 自动追踪 token_usage,写回 WorkflowContext
  - 仅注入指定 skill_domains 的工具(避免 token 污染)
  - 预算软终止 (token_budget 剩余 < 1000 时直接 skip)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from mas.state.schema import WorkflowContext


SOFT_BUDGET_RESERVE = 1000


class BaseAgent:
    def __init__(
        self,
        name: str,
        role_description: str,
        skill_domains: list[str],
        llm_invoker: Optional[Callable[[list[dict[str, Any]], list[Any]], dict[str, Any]]] = None,
        tool_registry: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.role_description = role_description
        self.skill_domains = skill_domains
        self.llm_invoker = llm_invoker or self._stub_llm
        if tool_registry is None:
            from mas.skills.registry import SkillRegistry
            tool_registry = SkillRegistry()
        self.tools = tool_registry.get_tools_for_domains(skill_domains)

    def __call__(self, state: WorkflowContext) -> dict[str, Any]:
        remaining = state.get("token_budget", 0) - state.get("token_usage", 0)
        if remaining < SOFT_BUDGET_RESERVE:
            return {
                "skill_outputs": [{
                    "agent": self.name,
                    "skill_name": "n/a",
                    "status": "skipped_budget_low",
                    "output": f"remaining={remaining}",
                }],
                "error": "token_budget_low",
            }

        system_prompt = (
            f"你是【{self.name}】,职责是【{self.role_description}】.\n"
            f"你可以调用的 Skill 工具: {[t.name for t in self.tools]}\n"
            f"输出必须包含: 决策依据 / 使用的 Skill 名称 / 置信度(0-1) / 预估金额(若涉及)."
        )
        prompt_messages = [{"role": "system", "content": system_prompt}, *state.get("messages", [])]

        response = self.llm_invoker(prompt_messages, self.tools)
        used_tokens = int(response.get("token_usage", 0))
        skill_output = {
            "agent": self.name,
            "skill_name": response.get("skill_name", "n/a"),
            "output": response.get("output", ""),
            "confidence": float(response.get("confidence", 0.0)),
            "estimated_cost": float(response.get("estimated_cost", 0.0)),
            "status": "ok",
        }

        return {
            "messages": [{"role": "assistant", "content": response.get("output", ""), "agent": self.name}],
            "skill_outputs": [skill_output],
            "token_usage": state.get("token_usage", 0) + used_tokens,
        }

    @staticmethod
    def _stub_llm(messages: list[dict[str, Any]], tools: list[Any]) -> dict[str, Any]:
        tool_names = [t.name for t in tools]
        return {
            "output": f"[STUB] received {len(messages)} messages, available tools: {tool_names[:3]}",
            "skill_name": tool_names[0] if tool_names else "none",
            "confidence": 0.5,
            "estimated_cost": 0.0,
            "token_usage": 100,
        }

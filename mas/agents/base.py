"""BaseAgent: 所有业务 Agent 的共同基类.

约束:
  - 自动追踪 token_usage,写回 WorkflowContext
  - 仅注入指定 skill_domains 的工具(避免 token 污染)
  - 预算软终止 (token_budget 剩余 < 1000 时直接 skip)
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from mas.state.schema import WorkflowContext


SOFT_BUDGET_RESERVE = 1000


def _build_real_llm() -> Callable[[list[dict[str, Any]], list[Any]], dict[str, Any]]:
    """构建真实 LLM 调用器。无 API key 时回退到 stub。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("MAS_LLM_MODEL", "claude-3-5-haiku-20241022")
    provider = os.environ.get("MAS_LLM_PROVIDER", "anthropic")  # anthropic | openai

    if not api_key:
        return BaseAgent._stub_llm

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)

            def _invoke_anthropic(messages, tools):
                system_msg = ""
                user_msgs = []
                for m in messages:
                    if m.get("role") == "system":
                        system_msg = m.get("content", "")
                    else:
                        user_msgs.append({"role": "user", "content": m.get("content", "")})

                tool_descriptions = []
                for t in tools:
                    desc = getattr(t, "description", "") or getattr(t, "__doc__", "") or ""
                    params = getattr(t, "parameters", {}) if hasattr(t, "parameters") else {}
                    tool_descriptions.append(
                        f"Tool: {getattr(t, 'name', 'unknown')}\nDescription: {desc}\n"
                        f"Parameters: {json.dumps(params, ensure_ascii=False)}"
                    )

                tool_text = "\n\n".join(tool_descriptions)
                enhanced = (
                    f"{system_msg}\n\n可用工具:\n{tool_text}\n\n"
                    f"返回 JSON: {{\"skill_name\": \"工具名\", \"output\": \"分析/决策内容\", "
                    f"\"confidence\": 0.0-1.0, \"estimated_cost\": 金额}}\n\n"
                    f"用户消息: {user_msgs[-1]['content'] if user_msgs else '无'}"
                )

                resp = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system="你是母婴跨境电商数据分析Agent。只返回JSON。",
                    messages=[{"role": "user", "content": enhanced}],
                )
                text = resp.content[0].text if resp.content else "{}"
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    result = {"output": text, "skill_name": "analysis", "confidence": 0.5, "estimated_cost": 0.0}
                result.setdefault("token_usage", resp.usage.input_tokens + resp.usage.output_tokens if hasattr(resp, 'usage') else 500)
                return result

            return _invoke_anthropic
        except ImportError:
            pass

    return BaseAgent._stub_llm


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
        self.llm_invoker = llm_invoker or _build_real_llm()
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

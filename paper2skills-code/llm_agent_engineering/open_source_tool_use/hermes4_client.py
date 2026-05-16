"""Hermes 4 Tool Use Client — 开源混合推理模型工具调用演示.

参考论文: Teknium et al. (2025) Hermes 4 Technical Report. arxiv:2508.18255.

本实现演示:
- Hermes4Config: 14B/70B/405B 模型配置
- Hermes4Tokenizer: <think> / </think> / <tool_call> / <tool_response> 特殊标签解析
- ToolCallParser: JSON schema 验证与解析
- ToolUseClient: 统一接口(本地 vLLM/llama.cpp + 云端 Together/Fireworks)
- RejectionSampler: 简化版 rejection sampling(验证→奖励→过滤)
- 跨境母婴客服 demo: 订单查询 + 物流追踪 tool use 流程

生产环境:
- 接 vLLM / llama.cpp 做本地推理
- 接 Together AI / Fireworks / OpenRouter 做云端托管
- Tool schema 用 Pydantic 定义, 生成 OpenAPI 格式
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------


class ModelSize(Enum):
    SMALL = "14B"
    MEDIUM = "70B"
    LARGE = "405B"


@dataclass(frozen=True)
class Hermes4Config:
    """Hermes 4 模型配置."""

    size: ModelSize
    base_model: str
    context_length: int = 131_072
    max_think_tokens: int = 16_384
    temperature_reasoning: float = 0.6
    temperature_tool_use: float = 0.0

    @classmethod
    def hermes4_14b(cls) -> "Hermes4Config":
        return cls(ModelSize.SMALL, "Qwen3-14B", max_think_tokens=16_384)

    @classmethod
    def hermes4_70b(cls) -> "Hermes4Config":
        return cls(ModelSize.MEDIUM, "Llama-3.1-70B", max_think_tokens=16_384)

    @classmethod
    def hermes4_405b(cls) -> "Hermes4Config":
        return cls(ModelSize.LARGE, "Llama-3.1-405B", max_think_tokens=24_000)


# ---------------------------------------------------------------------------
# Special Token Handling
# ---------------------------------------------------------------------------


class SpecialToken(Enum):
    THINK_START = "<think>"
    THINK_END = "</think>"
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_RESPONSE_START = "<tool_response>"
    TOOL_RESPONSE_END = "</tool_response>"


@dataclass
class ParsedContent:
    """解析后的内容块."""

    reasoning: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    plain_text: Optional[str] = None


class Hermes4Tokenizer:
    """Hermes 4 特殊标签解析器.

    处理格式:
    <think>
    ...推理过程...
    </think>
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>
    """

    def parse(self, raw: str) -> ParsedContent:
        result = ParsedContent()

        # 提取 <think> 内容
        think_pattern = re.compile(
            re.escape(SpecialToken.THINK_START.value)
            + r"(.*?)"
            + re.escape(SpecialToken.THINK_END.value),
            re.DOTALL,
        )
        think_match = think_pattern.search(raw)
        if think_match:
            result.reasoning = think_match.group(1).strip()

        # 提取 <tool_call> JSON
        tc_start = SpecialToken.TOOL_CALL_START.value
        tc_end = SpecialToken.TOOL_CALL_END.value
        tc_pattern = re.compile(
            re.escape(tc_start) + r"(.*?)" + re.escape(tc_end), re.DOTALL
        )
        for match in tc_pattern.finditer(raw):
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    result.tool_calls.append(parsed)
                elif isinstance(parsed, list):
                    result.tool_calls.extend(parsed)
            except json.JSONDecodeError:
                continue

        # 提取纯文本(去掉特殊标签后的内容)
        text = raw
        for token in SpecialToken:
            text = text.replace(token.value, "")
        text = text.strip()
        if text:
            result.plain_text = text

        return result

    def build_prompt(
        self,
        system_msg: str,
        user_msg: str,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_reasoning: bool = True,
    ) -> str:
        """构建 Hermes 4 格式的 prompt."""
        lines: list[str] = []
        lines.append(f"<|im_start|>system\n{system_msg}")
        if enable_reasoning:
            lines.append(
                "\nWhen solving problems, wrap your reasoning in <think>...</think> tags."
            )
        if tools:
            lines.append(
                "\nYou have access to the following tools. Use <tool_call>{...}</tool_call> to invoke them:"
            )
            for tool in tools:
                func = tool.get("function", tool)
                name = func.get("name", func.get("function", "unknown"))
                desc = func.get("description", "")
                lines.append(f"\n- {name}: {desc}")
        lines.append("<|im_end|>")
        lines.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
        lines.append("<|im_start|>assistant\n")
        return "\n".join(lines)

    def build_tool_response(
        self, tool_call: dict[str, Any], result: Any
    ) -> str:
        """构建 tool execution 后的响应."""
        name = tool_call.get("name", "unknown")
        return (
            f"{SpecialToken.TOOL_RESPONSE_START.value}\n"
            f'{{"name": "{name}", "result": {json.dumps(result)}}}\n'
            f"{SpecialToken.TOOL_RESPONSE_END.value}"
        )


# ---------------------------------------------------------------------------
# Tool Schema & Parsing
# ---------------------------------------------------------------------------


@dataclass
class ToolSchema:
    """工具描述 schema."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openapi(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCallParser:
    """解析 <tool_call> JSON 并验证 schema."""

    def __init__(self, schemas: list[ToolSchema]) -> None:
        self._schemas = {s.name: s for s in schemas}

    def validate(self, tool_call: dict[str, Any]) -> tuple[bool, Optional[str]]:
        """验证 tool call 是否符合 schema.

        Returns: (is_valid, error_message)
        """
        if "name" not in tool_call:
            return False, "Missing 'name' field"

        name = tool_call["name"]
        if name not in self._schemas:
            return False, f"Unknown tool: {name}"

        args = tool_call.get("arguments", tool_call.get("parameters", {}))
        if not isinstance(args, dict):
            return False, "Arguments must be a dict"

        schema = self._schemas[name]
        required = schema.parameters.get("required", [])
        for req in required:
            if req not in args:
                return False, f"Missing required argument: {req}"

        return True, None

    def parse_single(self, raw_json: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                return None, "Tool call must be a JSON object"
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"


# ---------------------------------------------------------------------------
# Backend Interface
# ---------------------------------------------------------------------------


class InferenceBackend(Protocol):
    """推理后端协议."""

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        ...


@dataclass
class LocalBackend:
    """本地推理后端(vLLM / llama.cpp 占位)."""

    model_path: str
    config: Hermes4Config

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        # 生产环境: 接 vLLM / llama.cpp
        del prompt, temperature, max_tokens
        raise NotImplementedError("Use vLLM or llama.cpp in production")


@dataclass
class CloudBackend:
    """云端推理后端(Together AI / Fireworks 占位)."""

    api_base: str
    api_key: str
    model_name: str

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        # 生产环境: 接 Together AI / Fireworks API
        del prompt, temperature, max_tokens
        raise NotImplementedError("Use Together AI / Fireworks API in production")


# ---------------------------------------------------------------------------
# Tool Use Client
# ---------------------------------------------------------------------------


@dataclass
class ToolUseClient:
    """Hermes 4 Tool Use 统一客户端."""

    config: Hermes4Config
    tokenizer: Hermes4Tokenizer = field(default_factory=Hermes4Tokenizer)
    parser: Optional[ToolCallParser] = None
    backend: Optional[InferenceBackend] = None
    tools: list[ToolSchema] = field(default_factory=list)
    tool_registry: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register_tool(
        self, schema: ToolSchema, handler: Callable[..., Any]
    ) -> None:
        self.tools.append(schema)
        self.tool_registry[schema.name] = handler

    def chat(
        self,
        user_msg: str,
        system_msg: Optional[str] = None,
        max_tool_rounds: int = 5,
    ) -> str:
        """带 tool use 的多轮对话.

        Returns: 最终 assistant 回复文本.
        """
        if self.backend is None:
            raise RuntimeError("Backend not configured")

        system = system_msg or "You are a helpful assistant."
        tool_schemas = [t.to_openapi() for t in self.tools]
        prompt = self.tokenizer.build_prompt(
            system, user_msg, tool_schemas, enable_reasoning=True
        )

        for _ in range(max_tool_rounds):
            response = self.backend.generate(
                prompt,
                temperature=self.config.temperature_tool_use,
                max_tokens=4096,
            )
            parsed = self.tokenizer.parse(response)

            if not parsed.tool_calls:
                return parsed.plain_text or response

            # 执行 tool calls
            for tc in parsed.tool_calls:
                if self.parser:
                    valid, err = self.parser.validate(tc)
                    if not valid:
                        return f"[Tool Error] {err}"

                name = tc.get("name", "")
                args = tc.get("arguments", tc.get("parameters", {}))
                if name in self.tool_registry:
                    result = self.tool_registry[name](**args)
                    tool_resp = self.tokenizer.build_tool_response(tc, result)
                    prompt += f"\n{response}\n{tool_resp}"
                else:
                    return f"[Tool Error] Unknown tool: {name}"

        return "[Error] Max tool rounds exceeded"


# ---------------------------------------------------------------------------
# Rejection Sampling
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """验证结果."""

    is_valid: bool
    reward: float  # binary: 1.0 or 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    """验证器协议."""

    def verify(self, response: str, expected: Any) -> VerificationResult:
        ...


class ToolCallVerifier:
    """Tool Use 环境验证器: 验证 <tool_call> JSON 与 origin 完全匹配."""

    def __init__(self, parser: ToolCallParser) -> None:
        self.parser = parser

    def verify(self, response: str, expected: dict[str, Any]) -> VerificationResult:
        parsed = self.parser.parse_single(response)
        if parsed[0] is None:
            return VerificationResult(False, 0.0, {"error": parsed[1]})

        tool_call = parsed[0]
        valid, err = self.parser.validate(tool_call)
        if not valid:
            return VerificationResult(False, 0.0, {"error": err})

        # 与 expected 对比(字段层级 + 值)
        match = tool_call == expected
        return VerificationResult(
            match, 1.0 if match else 0.0, {"expected": expected, "got": tool_call}
        )


@dataclass
class RejectionSampler:
    """简化版 rejection sampling: 采样 → 验证 → 过滤."""

    client: ToolUseClient
    verifier: Verifier
    max_samples: int = 16

    def sample(
        self, prompt: str, expected: dict[str, Any]
    ) -> tuple[Optional[str], list[VerificationResult]]:
        """Rejection sampling 直到找到有效样本.

        Returns: (best_response, all_results)
        """
        results: list[VerificationResult] = []
        best_response: Optional[str] = None

        for _ in range(self.max_samples):
            if self.client.backend is None:
                raise RuntimeError("Backend not configured")
            response = self.client.backend.generate(
                prompt,
                temperature=self.client.config.temperature_reasoning,
                max_tokens=4096,
            )
            vr = self.verifier.verify(response, expected)
            results.append(vr)

            if vr.is_valid and best_response is None:
                best_response = response
                break  # 找到有效样本,提前退出

        return best_response, results


# ---------------------------------------------------------------------------
# Demo: 跨境母婴客服 Tool Use
# ---------------------------------------------------------------------------


def _demo_order_lookup(order_id: str) -> dict[str, Any]:
    """模拟订单查询 tool."""
    return {
        "order_id": order_id,
        "status": "delivered",
        "product": "baby_formula_stage2_800g",
        "quantity": 3,
        "delivery_date": "2026-05-10",
        "tracking": "SF1234567890",
    }


def _demo_logistics_track(tracking_number: str) -> dict[str, Any]:
    """模拟物流追踪 tool."""
    return {
        "tracking_number": tracking_number,
        "carrier": "SF Express",
        "status": "in_transit",
        "current_location": "Shanghai Distribution Center",
        "estimated_arrival": "2026-05-18",
        "history": [
            {"time": "2026-05-12 09:00", "location": "Shenzhen", "event": "picked_up"},
            {"time": "2026-05-13 14:30", "location": "Shenzhen Hub", "event": "departed"},
            {"time": "2026-05-14 08:00", "location": "Shanghai Distribution Center", "event": "arrived"},
        ],
    }


class MockBackend:
    """模拟推理后端(用于 demo)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self._idx = 0

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        del prompt, temperature, max_tokens
        resp = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return resp


def demo_cross_border_customer_service() -> None:
    """跨境母婴客服 Tool Use demo."""
    print("=== Hermes 4 Tool Use Demo: Cross-Border Customer Service ===\n")

    config = Hermes4Config.hermes4_14b()
    tokenizer = Hermes4Tokenizer()

    # 定义 tools
    order_schema = ToolSchema(
        name="order_lookup",
        description="查询订单状态",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "订单号"},
            },
            "required": ["order_id"],
        },
    )
    logistics_schema = ToolSchema(
        name="logistics_track",
        description="追踪物流信息",
        parameters={
            "type": "object",
            "properties": {
                "tracking_number": {"type": "string", "description": "物流单号"},
            },
            "required": ["tracking_number"],
        },
    )

    # 模拟 backend: 第一轮 tool call, 第二轮最终回复
    mock_response_1 = (
        "<think>\n"
        "用户询问订单 ORD1001 的状态。我需要先查询订单信息，然后追踪物流。\n"
        "</think>\n"
        "<tool_call>\n"
        '{"name": "order_lookup", "arguments": {"order_id": "ORD1001"}}\n'
        "</tool_call>"
    )
    mock_response_2 = (
        "<think>\n"
        "已获取订单信息: 已送达, 产品为 baby_formula_stage2_800g, 物流单号 SF1234567890。\n"
        "接下来追踪物流详细状态。\n"
        "</think>\n"
        "<tool_call>\n"
        '{"name": "logistics_track", "arguments": {"tracking_number": "SF1234567890"}}\n'
        "</tool_call>"
    )
    mock_response_3 = (
        "<think>\n"
        "物流信息显示包裹在上海分拨中心,预计 2026-05-18 到达。\n"
        "综合订单和物流信息,可以给用户完整答复。\n"
        "</think>\n"
        "您的订单 ORD1001 (baby formula stage 2, 3罐) 已于 2026-05-10 送达。"
        "物流追踪显示包裹目前在上海分拨中心,预计 2026-05-18 最终送达。"
    )

    backend = MockBackend([mock_response_1, mock_response_2, mock_response_3])

    parser = ToolCallParser([order_schema, logistics_schema])
    client = ToolUseClient(config=config, tokenizer=tokenizer, parser=parser, backend=backend)
    client.register_tool(order_schema, _demo_order_lookup)
    client.register_tool(logistics_schema, _demo_logistics_track)

    # 执行对话
    result = client.chat(
        user_msg="我的订单 ORD1001 现在什么状态？",
        system_msg="你是一个专业的跨境母婴客服助手，帮助用户查询订单和物流信息。",
    )
    print(f"最终回复: {result}\n")

    # 展示推理过程提取
    print("--- Reasoning Trace ---")
    for i, resp in enumerate([mock_response_1, mock_response_2, mock_response_3], 1):
        parsed = tokenizer.parse(resp)
        if parsed.reasoning:
            print(f"  Round {i}: {parsed.reasoning[:80]}...")
        if parsed.tool_calls:
            for tc in parsed.tool_calls:
                print(f"    → Tool: {tc.get('name')}({tc.get('arguments', {})})")
    print()


# ---------------------------------------------------------------------------
# Test Pipeline
# ---------------------------------------------------------------------------


def test_pipeline() -> None:
    """Sanity checks."""

    # 1) Config
    cfg14 = Hermes4Config.hermes4_14b()
    assert cfg14.size == ModelSize.SMALL
    assert cfg14.base_model == "Qwen3-14B"

    cfg70 = Hermes4Config.hermes4_70b()
    assert cfg70.size == ModelSize.MEDIUM

    cfg405 = Hermes4Config.hermes4_405b()
    assert cfg405.size == ModelSize.LARGE

    # 2) Tokenizer parse
    tok = Hermes4Tokenizer()
    raw = (
        "<think>\n推理内容\n</think>\n"
        "<tool_call>\n"
        '{"name": "test_tool", "arguments": {"x": 1}}\n'
        "</tool_call>\n"
        "Plain text here"
    )
    parsed = tok.parse(raw)
    assert parsed.reasoning == "推理内容"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["name"] == "test_tool"
    assert parsed.plain_text is not None
    assert "Plain text here" in parsed.plain_text

    # 3) Multiple tool calls
    raw_multi = (
        "<tool_call>\n"
        '{"name": "a", "arguments": {}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '[{"name": "b", "arguments": {}}, {"name": "c", "arguments": {}}]\n'
        "</tool_call>"
    )
    parsed_multi = tok.parse(raw_multi)
    assert len(parsed_multi.tool_calls) == 3
    names = [tc["name"] for tc in parsed_multi.tool_calls]
    assert names == ["a", "b", "c"]

    # 4) ToolCallParser
    schema = ToolSchema("lookup", "查询", {"required": ["id"]})
    parser = ToolCallParser([schema])

    valid_tc = {"name": "lookup", "arguments": {"id": "123"}}
    ok, err = parser.validate(valid_tc)
    assert ok and err is None

    invalid_tc = {"name": "lookup", "arguments": {}}
    ok2, err2 = parser.validate(invalid_tc)
    assert not ok2
    assert err2 and "Missing required" in err2

    # 5) ToolCallVerifier
    verifier = ToolCallVerifier(parser)
    vr = verifier.verify(json.dumps(valid_tc), valid_tc)
    assert vr.is_valid and vr.reward == 1.0

    vr_bad = verifier.verify('{"name": "unknown"}', valid_tc)
    assert not vr_bad.is_valid

    # 6) Prompt building
    prompt = tok.build_prompt(
        "You are helpful.",
        "Hello",
        tools=[{"name": "test", "description": "A test tool"}],
        enable_reasoning=True,
    )
    assert "<think>" in prompt
    assert "test" in prompt
    assert "<|im_start|>system" in prompt

    # 7) Tool response building
    resp = tok.build_tool_response({"name": "lookup"}, {"status": "ok"})
    assert SpecialToken.TOOL_RESPONSE_START.value in resp
    assert '"status": "ok"' in resp

    # 8) Special tokens enum completeness
    assert len(SpecialToken) == 6

    print("[PASS] all assertions")


def main() -> None:
    test_pipeline()
    print()
    demo_cross_border_customer_service()

    # 9) 展示 RejectionSampler 结构
    print("\n--- RejectionSampler Structure ---")
    print("RejectionSampler = sample(prompt, expected) -> (best, results)")
    print("  - max_samples: 16 (configurable)")
    print("  - reward: binary 1.0/0.0 (exact match)")
    print("  - stops at first valid sample")

    # 10) 展示模型家族对比
    print("\n--- Hermes 4 Model Family ---")
    for cfg in [
        Hermes4Config.hermes4_14b(),
        Hermes4Config.hermes4_70b(),
        Hermes4Config.hermes4_405b(),
    ]:
        print(
            f"  {cfg.size.value:>5s} ({cfg.base_model:20s}) "
            f"max_think={cfg.max_think_tokens} "
            f"temp_reasoning={cfg.temperature_reasoning} "
            f"temp_tool={cfg.temperature_tool_use}"
        )


if __name__ == "__main__":
    main()

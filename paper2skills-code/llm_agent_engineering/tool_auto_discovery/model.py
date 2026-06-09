"""
Tool Auto Discovery — Agent 工具自动发现：OpenAPI + MCP Schema 自注册
paper2skills-code: 16-智能体工程 | 母婴出海跨境电商

核心设计：
  - OpenAPIParser：解析 OpenAPI JSON → ToolDefinition 列表
  - MCPSchemaReader：读取 MCP list_tools 响应 → ToolDefinition 列表
  - ToolSimilarityChecker：Jaccard 相似度去重（阈值 0.85）
  - AutoDiscoveryRegistry：自动发现 + 注册 + 健康检查全流程
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

class ToolStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"     # 健康检查失败，暂停使用
    DEPRECATED = "deprecated"


@dataclass
class ParameterSchema:
    """工具参数 schema"""
    name: str
    param_type: str        # string / integer / number / boolean / array / object
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """标准工具定义（OpenAPI + MCP 统一格式）"""
    name: str
    description: str
    parameters: list[ParameterSchema] = field(default_factory=list)
    source: str = ""           # 来源标识（供应商名 / MCP server 名）
    endpoint: str = ""         # API endpoint（OpenAPI 场景）
    method: str = "POST"       # HTTP 方法
    success_rate: float = 1.0  # 历史成功率（初始默认 1.0）
    latency_ms: float = 300.0  # 平均延迟（毫秒）
    status: ToolStatus = ToolStatus.ACTIVE
    registered_at: float = field(default_factory=time.time)
    consecutive_failures: int = 0


# ─────────────────────────────────────────────
# OpenAPI 解析器
# ─────────────────────────────────────────────

class OpenAPIParser:
    """
    将 OpenAPI 3.x JSON schema 转换为 ToolDefinition 列表。

    支持路径：
      paths[path][method] → 提取 operationId / summary / parameters / requestBody
    """

    # OpenAPI type → Python type 映射
    _TYPE_MAP: dict[str, str] = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }

    def parse(self, schema: dict[str, Any], source: str = "") -> list[ToolDefinition]:
        """解析完整 OpenAPI schema，返回工具列表"""
        tools: list[ToolDefinition] = []
        paths = schema.get("paths", {})

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                    continue
                tool = self._parse_operation(path, method.upper(), operation, source)
                if tool:
                    tools.append(tool)

        return tools

    def _parse_operation(
        self,
        path: str,
        method: str,
        operation: dict[str, Any],
        source: str,
    ) -> ToolDefinition | None:
        """从单个 OpenAPI operation 生成 ToolDefinition"""
        # 工具名：优先用 operationId，否则组合 method_path
        name = operation.get("operationId") or f"{method.lower()}_{path.replace('/', '_').strip('_')}"
        description = operation.get("summary") or operation.get("description") or ""

        if not description:
            return None  # 无描述的工具跳过

        parameters: list[ParameterSchema] = []

        # 解析 query/path 参数
        for param in operation.get("parameters", []):
            schema_info = param.get("schema", {})
            parameters.append(ParameterSchema(
                name=param.get("name", ""),
                param_type=self._TYPE_MAP.get(schema_info.get("type", "string"), "string"),
                description=param.get("description", ""),
                required=param.get("required", False),
                default=schema_info.get("default"),
            ))

        # 解析 requestBody（JSON body 参数）
        request_body = operation.get("requestBody", {})
        content = request_body.get("content", {})
        json_content = content.get("application/json", {})
        body_schema = json_content.get("schema", {})
        body_props = body_schema.get("properties", {})
        required_body = body_schema.get("required", [])

        for prop_name, prop_info in body_props.items():
            parameters.append(ParameterSchema(
                name=prop_name,
                param_type=self._TYPE_MAP.get(prop_info.get("type", "string"), "string"),
                description=prop_info.get("description", ""),
                required=prop_name in required_body,
                default=prop_info.get("default"),
            ))

        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            source=source,
            endpoint=path,
            method=method,
        )


# ─────────────────────────────────────────────
# MCP Schema 读取器
# ─────────────────────────────────────────────

class MCPSchemaReader:
    """
    将 MCP Server 的 list_tools 响应转换为 ToolDefinition 列表。

    MCP list_tools 响应格式（标准）：
    {
        "tools": [
            {
                "name": "tool_name",
                "description": "...",
                "inputSchema": {
                    "type": "object",
                    "properties": { "param1": {"type": "string", "description": "..."} },
                    "required": ["param1"]
                }
            }
        ]
    }
    """

    def parse(self, list_tools_response: dict[str, Any], source: str = "") -> list[ToolDefinition]:
        """解析 MCP list_tools 响应"""
        tools: list[ToolDefinition] = []
        for tool_info in list_tools_response.get("tools", []):
            tool = self._parse_tool(tool_info, source)
            if tool:
                tools.append(tool)
        return tools

    def _parse_tool(self, tool_info: dict[str, Any], source: str) -> ToolDefinition | None:
        name = tool_info.get("name", "")
        description = tool_info.get("description", "")
        if not name or not description:
            return None

        input_schema = tool_info.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_fields = input_schema.get("required", [])

        parameters: list[ParameterSchema] = []
        for prop_name, prop_info in properties.items():
            parameters.append(ParameterSchema(
                name=prop_name,
                param_type=prop_info.get("type", "string"),
                description=prop_info.get("description", ""),
                required=prop_name in required_fields,
            ))

        return ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            source=source,
        )


# ─────────────────────────────────────────────
# 相似度检测（去重）
# ─────────────────────────────────────────────

class ToolSimilarityChecker:
    """
    基于 Jaccard 相似度检测重复工具。
    相似度 > threshold 则认为是重复，拒绝注册。
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold

    def is_duplicate(
        self,
        new_tool: ToolDefinition,
        existing_tools: list[ToolDefinition],
    ) -> tuple[bool, str]:
        """
        检查 new_tool 是否与 existing_tools 中某个工具重复。

        Returns:
            (is_dup, duplicate_name)：重复时返回 (True, 已有工具名)
        """
        new_tokens = self._tokenize(new_tool.description)
        for existing in existing_tools:
            similarity = self._jaccard(new_tokens, self._tokenize(existing.description))
            if similarity >= self.threshold:
                return True, existing.name
        return False, ""

    def _tokenize(self, text: str) -> set[str]:
        """简单分词：小写 + 按空格/标点切分"""
        import re
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return set(tokens)

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# 自动发现注册中心
# ─────────────────────────────────────────────

class AutoDiscoveryRegistry:
    """
    工具自动发现、注册、健康检查一体化注册中心。

    功能：
      - discover_from_schema()：从 OpenAPI schema dict 发现工具
      - discover_from_mcp()：从 MCP list_tools 响应发现工具
      - register()：注册单个工具（含去重）
      - get_active_tools()：查询活跃工具列表（按质量排序）
      - report_result()：上报工具调用结果（更新 success_rate）
      - health_check()：心跳检查（模拟 ping）
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._tools: dict[str, ToolDefinition] = {}   # name → ToolDefinition
        self._openapi_parser = OpenAPIParser()
        self._mcp_reader = MCPSchemaReader()
        self._similarity_checker = ToolSimilarityChecker(threshold=similarity_threshold)
        self._call_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}

    # ── 发现接口 ──────────────────────────────

    def discover_from_schema(
        self,
        openapi_schema: dict[str, Any],
        source: str = "",
    ) -> list[ToolDefinition]:
        """从 OpenAPI schema 自动发现并注册工具"""
        candidates = self._openapi_parser.parse(openapi_schema, source=source)
        return self._batch_register(candidates)

    def discover_from_mcp(
        self,
        list_tools_response: dict[str, Any],
        source: str = "",
    ) -> list[ToolDefinition]:
        """从 MCP list_tools 响应自动发现并注册工具"""
        candidates = self._mcp_reader.parse(list_tools_response, source=source)
        return self._batch_register(candidates)

    # ── 注册逻辑 ──────────────────────────────

    def register(self, tool: ToolDefinition) -> tuple[bool, str]:
        """
        注册单个工具。

        Returns:
            (success, reason)
        """
        # 名称冲突：同名工具已存在
        if tool.name in self._tools:
            return False, f"duplicate_name:{tool.name}"

        # 描述相似度过高：视为重复工具
        existing = list(self._tools.values())
        is_dup, dup_name = self._similarity_checker.is_duplicate(tool, existing)
        if is_dup:
            return False, f"similar_to:{dup_name}"

        self._tools[tool.name] = tool
        self._call_counts[tool.name] = 0
        self._success_counts[tool.name] = 0
        return True, "registered"

    def _batch_register(self, candidates: list[ToolDefinition]) -> list[ToolDefinition]:
        """批量注册，返回成功注册的工具列表"""
        registered: list[ToolDefinition] = []
        for tool in candidates:
            success, _ = self.register(tool)
            if success:
                registered.append(tool)
        return registered

    # ── 查询接口 ──────────────────────────────

    def get_active_tools(self, min_success_rate: float = 0.0) -> list[ToolDefinition]:
        """
        返回活跃工具列表，按质量得分排序（success_rate / latency_ms）。
        """
        active = [
            t for t in self._tools.values()
            if t.status == ToolStatus.ACTIVE and t.success_rate >= min_success_rate
        ]
        # 质量得分 = success_rate * 1000 / latency_ms
        active.sort(
            key=lambda t: t.success_rate * 1000 / max(t.latency_ms, 1),
            reverse=True,
        )
        return active

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    # ── 质量反馈 ──────────────────────────────

    def report_result(self, tool_name: str, success: bool, latency_ms: float) -> None:
        """上报工具调用结果，更新 success_rate 和 latency_ms（指数移动平均）"""
        if tool_name not in self._tools:
            return

        tool = self._tools[tool_name]
        self._call_counts[tool_name] += 1
        if success:
            self._success_counts[tool_name] += 1
        else:
            tool.consecutive_failures += 1
            # 连续失败 3 次 → 暂停
            if tool.consecutive_failures >= 3:
                tool.status = ToolStatus.PAUSED
        if success:
            tool.consecutive_failures = 0

        # EMA 更新 success_rate（α=0.1）
        total = self._call_counts[tool_name]
        tool.success_rate = self._success_counts[tool_name] / total

        # EMA 更新 latency（α=0.2）
        tool.latency_ms = 0.8 * tool.latency_ms + 0.2 * latency_ms

    def health_check(self, tool_name: str, ping_success: bool) -> None:
        """健康检查：ping 成功则恢复 ACTIVE，失败则累计 consecutive_failures"""
        if tool_name not in self._tools:
            return
        tool = self._tools[tool_name]
        if ping_success and tool.status == ToolStatus.PAUSED:
            tool.status = ToolStatus.ACTIVE
            tool.consecutive_failures = 0
        elif not ping_success:
            tool.consecutive_failures += 1
            if tool.consecutive_failures >= 3:
                tool.status = ToolStatus.PAUSED


# ─────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────

def _test_auto_discovery() -> None:
    """模拟 3 个 API schema，验证自动注册和去重"""

    registry = AutoDiscoveryRegistry(similarity_threshold=0.85)

    # ── Schema 1：供应商 MOQ 查询 API ──────────
    supplyx_schema: dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "SupplyX API", "version": "1.0"},
        "paths": {
            "/moq/query": {
                "post": {
                    "operationId": "moq_query",
                    "summary": "Query minimum order quantity for a product SKU",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "sku": {"type": "string", "description": "Product SKU"},
                                        "warehouse": {"type": "string", "description": "Warehouse code"},
                                    },
                                    "required": ["sku"],
                                }
                            }
                        }
                    },
                }
            },
            "/inventory/check": {
                "get": {
                    "operationId": "inventory_check",
                    "summary": "Check real-time inventory level for a product",
                    "parameters": [
                        {
                            "name": "sku",
                            "in": "query",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Product SKU to check",
                        }
                    ],
                }
            },
        },
    }

    tools1 = registry.discover_from_schema(supplyx_schema, source="supplyx")
    assert len(tools1) == 2, f"期望注册 2 个工具，实际 {len(tools1)}"
    print(f"[SupplyX] 注册 {len(tools1)} 个工具: {[t.name for t in tools1]}")

    # ── Schema 2：MCP 广告 API（Google Ads）──────
    google_ads_mcp: dict[str, Any] = {
        "tools": [
            {
                "name": "google_ads_create_campaign",
                "description": "Create a new Google Ads campaign with budget and targeting settings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "campaign_name": {"type": "string", "description": "Campaign display name"},
                        "budget_usd": {"type": "number", "description": "Daily budget in USD"},
                        "targeting_country": {"type": "string", "description": "Target country code (e.g. US)"},
                    },
                    "required": ["campaign_name", "budget_usd"],
                },
            },
            {
                "name": "google_ads_get_report",
                "description": "Get performance report for Google Ads campaigns including CTR and ROAS",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "campaign_id": {"type": "string", "description": "Campaign ID"},
                        "date_range": {"type": "string", "description": "Date range: last_7d / last_30d"},
                    },
                    "required": ["campaign_id"],
                },
            },
        ]
    }

    tools2 = registry.discover_from_mcp(google_ads_mcp, source="google_ads")
    assert len(tools2) == 2, f"期望注册 2 个工具，实际 {len(tools2)}"
    print(f"[Google Ads MCP] 注册 {len(tools2)} 个工具: {[t.name for t in tools2]}")

    # ── Schema 3：重复工具测试（Meta Ads，与 Google Ads 工具描述高度相似）──
    meta_ads_mcp_with_dup: dict[str, Any] = {
        "tools": [
            {
                # 完全不同的工具
                "name": "meta_ads_create_adset",
                "description": "Create a Meta Ads ad set with audience targeting and placement options",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "adset_name": {"type": "string", "description": "Ad set name"},
                        "audience_id": {"type": "string", "description": "Custom audience ID"},
                    },
                    "required": ["adset_name"],
                },
            },
            {
                # 与 google_ads_get_report 高度重复（刻意测试去重）
                "name": "google_ads_get_report_v2",
                "description": "Get performance report for Google Ads campaigns including CTR and ROAS",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
        ]
    }

    tools3 = registry.discover_from_mcp(meta_ads_mcp_with_dup, source="meta_ads")
    assert len(tools3) == 1, f"期望注册 1 个工具（去重后），实际 {len(tools3)}"
    assert tools3[0].name == "meta_ads_create_adset"
    print(f"[Meta Ads MCP] 注册 {len(tools3)} 个工具（去重 1 个重复工具）")

    # ── 验证活跃工具总数 ──────────────────────
    active = registry.get_active_tools()
    assert len(active) == 5, f"期望 5 个活跃工具，实际 {len(active)}"
    print(f"\n活跃工具总数: {len(active)}")
    for t in active:
        print(f"  [{t.source}] {t.name}: {t.description[:50]}...")

    # ── 测试质量反馈 ──────────────────────────
    registry.report_result("moq_query", success=True, latency_ms=120)
    registry.report_result("moq_query", success=True, latency_ms=110)
    registry.report_result("moq_query", success=False, latency_ms=5000)
    registry.report_result("inventory_check", success=False, latency_ms=9000)
    registry.report_result("inventory_check", success=False, latency_ms=9000)
    registry.report_result("inventory_check", success=False, latency_ms=9000)

    # inventory_check 连续失败 3 次，应被暂停
    inv_tool = registry.get_tool("inventory_check")
    assert inv_tool is not None
    assert inv_tool.status == ToolStatus.PAUSED, f"期望 PAUSED，实际 {inv_tool.status}"
    print(f"\ninventory_check 状态: {inv_tool.status}（连续失败 3 次后暂停）✓")

    # 活跃工具应减少到 4 个
    active_after = registry.get_active_tools()
    assert len(active_after) == 4, f"期望 4 个活跃工具，实际 {len(active_after)}"

    # ── 健康检查恢复 ──────────────────────────
    registry.health_check("inventory_check", ping_success=True)
    inv_tool_recovered = registry.get_tool("inventory_check")
    assert inv_tool_recovered is not None
    assert inv_tool_recovered.status == ToolStatus.ACTIVE
    print(f"inventory_check 恢复状态: {inv_tool_recovered.status} ✓")

    print("\n✅ 所有测试通过：自动注册 5 个工具，成功去重 1 个，健康检查暂停/恢复正常")


if __name__ == "__main__":
    _test_auto_discovery()

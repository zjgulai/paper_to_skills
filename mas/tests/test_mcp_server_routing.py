"""MCP Server 路由集成测试.

验证:
  - MultiServerMCPClient 能正确路由到各 domain server
  - 每个 server 的工具数量和 list_tools() 格式正确
  - call_tool 返回结构包含 skill 字段
  - 未知工具名返回 tool_not_found
  - SkillRegistry.get_mcp_client() 可用
"""

from __future__ import annotations

from mas.skills.mcp_servers.client import MultiServerMCPClient
from mas.skills.mcp_servers.supply_chain_server import SupplyChainServer
from mas.skills.mcp_servers.advertising_server import AdvertisingServer
from mas.skills.mcp_servers.customer_service_server import CustomerServiceServer
from mas.skills.mcp_servers.selection_server import SelectionServer
from mas.skills.registry import SkillRegistry


def test_mcp_client_total_tools():
    client = MultiServerMCPClient()
    assert client.total_tools >= 27


def test_mcp_client_list_all_tools_format():
    client = MultiServerMCPClient()
    tools = client.list_all_tools()
    assert len(tools) >= 27
    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "domain" in t


def test_mcp_supply_chain_server_tools():
    server = SupplyChainServer()
    assert len(server.list_tools()) >= 8
    result = server.call_tool("supply_demand_forecast",
                              sku_id="SKU-001",
                              history_daily_sales=[10.0, 12.0, 11.0, 13.0, 10.0])
    assert result.get("skill") == "supply_demand_forecast"
    assert "forecast_total" in result


def test_mcp_advertising_server_tools():
    server = AdvertisingServer()
    assert len(server.list_tools()) >= 7
    rows = [{"search_term": "baby formula", "clicks": 20, "orders": 0, "spend": 30.0, "revenue": 0.0}]
    result = server.call_tool("ad_negative_keywords", rows=rows)
    assert result.get("skill") == "ad_negative_keywords"
    assert result["negative_count"] >= 1


def test_mcp_customer_service_server_tools():
    server = CustomerServiceServer()
    assert len(server.list_tools()) >= 6
    result = server.call_tool("case_intent_classifier", text="我要退款")
    assert result.get("skill") == "case_intent_classifier"
    assert result["intent"] == "refund"


def test_mcp_selection_server_tools():
    server = SelectionServer()
    assert len(server.list_tools()) >= 6
    candidate = {
        "monthly_sales_usd": 80000, "review_count": 200,
        "avg_rating": 4.5, "bsr_trend_30d": 0.3,
    }
    result = server.call_tool("selection_market_space", candidate=candidate)
    assert result.get("skill") == "growth_new_product_opportunity"
    assert "market_score" in result


def test_mcp_tool_not_found():
    client = MultiServerMCPClient()
    result = client.call_tool("nonexistent_tool_xyz")
    assert result["status"] == "tool_not_found"
    assert result["tool"] == "nonexistent_tool_xyz"


def test_registry_get_mcp_client():
    reg = SkillRegistry()
    client = reg.get_mcp_client()
    assert isinstance(client, MultiServerMCPClient)
    assert client.total_tools >= 27
    result = client.call_tool("supply_safety_stock_replenishment",
                              sku_id="SKU-X", daily_mean=10.0,
                              daily_std=2.0, lead_time_days=30.0)
    assert "safety_stock" in result


def run_all():
    tests = [
        test_mcp_client_total_tools,
        test_mcp_client_list_all_tools_format,
        test_mcp_supply_chain_server_tools,
        test_mcp_advertising_server_tools,
        test_mcp_customer_service_server_tools,
        test_mcp_selection_server_tools,
        test_mcp_tool_not_found,
        test_registry_get_mcp_client,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ❌ {t.__name__}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nMCP Server 路由集成测试: {len(tests) - len(failures)}/{len(tests)} 通过")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run_all())

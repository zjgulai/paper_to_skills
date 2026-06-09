# MCP Servers

每个 domain server 遵循统一接口，通过 `MultiServerMCPClient` 路由调用。

## 接口规范

### Server 结构

```python
class BaseMCPServer:
    server_name: str          # 唯一标识，与 domain 对应
    domain: str               # SkillRegistry domain key
    tools: dict[str, dict]    # {tool_name: {"description": str, "fn": Callable}}

    def list_tools(self) -> list[dict]
    def call_tool(self, tool_name: str, **kwargs) -> dict
```

### 协议约定

- `list_tools()` 返回 `[{"name": str, "description": str, "domain": str}]`
- `call_tool()` 返回 `{"skill": str, "status": str, ...业务结果字段}`
- 工具名称与 `SkillRegistry._DOMAIN_TOOLS` 中的 `SkillTool.name` 保持一致
- 不存在的工具名返回 `{"status": "tool_not_found", "tool": tool_name}`

### 生产迁移路径

当前 stub 实现使用纯函数路由。生产替换步骤：

1. 安装 `mcp` SDK：`pip install mcp`
2. 将 `BaseMCPServer` 改为继承 `mcp.server.Server`
3. 用 `@server.tool()` 装饰器注册每个工具函数
4. `MultiServerMCPClient` 改为 `mcp.client.MultiServerMCPClient`
5. 37 个现有测试 + C-8 集成测试零修改通过 = 迁移完成

## 已实现 Servers

| Server 文件 | Domain | 工具数 | 对应工作流 |
|---|---|---|---|
| `supply_chain_server.py` | `supply_chain` | 8 | WF-A 智能补货 |
| `advertising_server.py` | `advertising` | 7 | WF-B 广告优化 |
| `customer_service_server.py` | `customer_service` | 6 | WF-C/E 客服+Review |
| `selection_server.py` | `selection` | 6 | WF-D 选品扫描 |

## Client 使用示例

```python
from mas.skills.mcp_servers.client import MultiServerMCPClient

client = MultiServerMCPClient()
tools = client.list_all_tools()
result = client.call_tool("supply_demand_forecast",
                          sku_id="SKU-001",
                          history_daily_sales=[10, 12, 11])
```

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mas.skills.mcp_servers.advertising_server import AdvertisingServer
from mas.skills.mcp_servers.base import BaseMCPServer
from mas.skills.mcp_servers.customer_service_server import CustomerServiceServer
from mas.skills.mcp_servers.selection_server import SelectionServer
from mas.skills.mcp_servers.supply_chain_server import SupplyChainServer


_DEFAULT_SERVERS: List[BaseMCPServer] = [
    SupplyChainServer(),
    AdvertisingServer(),
    CustomerServiceServer(),
    SelectionServer(),
]


class MultiServerMCPClient:
    def __init__(self, servers: Optional[List[BaseMCPServer]] = None) -> None:
        self._servers: Dict[str, BaseMCPServer] = {}
        for s in (servers or _DEFAULT_SERVERS):
            self._servers[s.server_name] = s

        self._tool_index: Dict[str, str] = {}
        for server_name, server in self._servers.items():
            for tool in server.list_tools():
                self._tool_index[tool["name"]] = server_name

    def list_all_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for server in self._servers.values():
            tools.extend(server.list_tools())
        return tools

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        server_name = self._tool_index.get(tool_name)
        if server_name is None:
            return {"status": "tool_not_found", "tool": tool_name}
        return self._servers[server_name].call_tool(tool_name, **kwargs)

    def get_server(self, server_name: str) -> Optional[BaseMCPServer]:
        return self._servers.get(server_name)

    @property
    def total_tools(self) -> int:
        return len(self._tool_index)

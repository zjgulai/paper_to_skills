from __future__ import annotations

from typing import Any, Callable, Dict, List


class BaseMCPServer:
    server_name: str = ""
    domain: str = ""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        raise NotImplementedError

    def _add(self, name: str, description: str, fn: Callable[..., Dict[str, Any]]) -> None:
        self._tools[name] = {"description": description, "domain": self.domain, "fn": fn}

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": k, "description": v["description"], "domain": v["domain"]}
            for k, v in self._tools.items()
        ]

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        entry = self._tools.get(tool_name)
        if entry is None:
            return {"status": "tool_not_found", "tool": tool_name}
        return entry["fn"](**kwargs)

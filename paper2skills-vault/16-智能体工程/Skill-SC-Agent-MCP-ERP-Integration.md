---
title: 供应链智能体MCP多ERP集成 — Model Context Protocol驱动的多系统双向协调模式
doc_type: knowledge
module: 16-智能体工程
topic: sc-agent-mcp-erp-integration
status: stable
created: 2026-06-18
updated: 2026-06-18
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 供应链智能体MCP多ERP集成

> **来源**：AWS + Elixir Claw 案例（SAP+Oracle+Neptune 多ERP MCP集成，采购自动化率30%→80%）+ MCP（Model Context Protocol）规范 v2025 + Palantir OSDK（Ontology Software Development Kit）设计原则
> **桥梁**：智能体工程 ↔ 供应链系统集成 ↔ Palantir OSDK/Writeback Layer | **类型**：智能体工程+系统集成

## ① 算法原理

**MCP（Model Context Protocol）**是 2024 年 Anthropic 提出的 AI Agent 与外部工具/系统交互的标准协议，解决了 Agent 集成碎片化的问题。在供应链场景，企业往往有 SAP ERP + Oracle 采购 + WMS + 物流 API 等多套系统，每套都需要不同集成方式。MCP 统一了这个层：**所有外部系统都变成 MCP Server，Agent 通过统一协议调用**。

**MCP 三层架构**：

```
┌─────────────────────────────────────────────────────────┐
│ Agent Layer (LLM + Tool Use)                            │
│ 供应链规划Agent / 补货Agent / 采购Agent                  │
└─────────────────────────────────────────────────────────┘
              ↕ MCP Protocol (JSON-RPC 2.0)
┌─────────────────────────────────────────────────────────┐
│ MCP Server Layer                                        │
│ ├─ SAP MCP Server: 采购订单 CRUD / 库存查询             │
│ ├─ Oracle MCP Server: 供应商管理 / 合同数据             │
│ ├─ WMS MCP Server: 仓储操作 / 入出库                    │
│ ├─ Neptune MCP Server: 供应链知识图谱查询               │
│ └─ Logistics MCP Server: 物流跟踪 / ETA查询             │
└─────────────────────────────────────────────────────────┘
              ↕ Native API (REST/SOAP/GraphQL)
┌─────────────────────────────────────────────────────────┐
│ Enterprise System Layer                                 │
│ SAP S/4HANA | Oracle SCM | Manhattan WMS | UPS API      │
└─────────────────────────────────────────────────────────┘
```

**MCP Tool 定义三要素**（Agent 调用时需要的信息）：

```json
{
  "name": "create_purchase_order",
  "description": "在SAP中创建采购订单，触发供应商确认流程",
  "inputSchema": {
    "type": "object",
    "properties": {
      "supplier_id": {"type": "string", "description": "SAP供应商编码"},
      "sku": {"type": "string", "description": "物料编码"},
      "quantity": {"type": "integer", "minimum": 1},
      "delivery_date": {"type": "string", "format": "date"},
      "po_type": {"type": "string", "enum": ["standard", "urgent", "blanket"]}
    },
    "required": ["supplier_id", "sku", "quantity", "delivery_date"]
  }
}
```

**Palantir OSDK 对比**：OSDK 是 Palantir 自研的 SDK，MCP 是开放标准，两者设计理念一致——都是"Agent 通过结构化接口操作业务对象"。中小企业可以用 MCP + 开源 ERP 替代 OSDK。

## ② 母婴出海应用案例

**场景A：补货 Agent 跨系统自动化采购**

补货 Agent 决策需要补货 2000 件吸奶器时，自动：
1. 查询 Oracle 供应商档案（MCP）→ 获取最优供应商
2. 查询 SAP 库存（MCP）→ 确认当前库存和在途
3. 在 SAP 创建 PO（MCP）→ 生成采购订单
4. 在 WMS 预约入库槽位（MCP）→ 确保仓库空间
5. 发送供应商确认邮件（MCP Email）→ 触发供应商响应

全程无人工介入，完成时间 <5 分钟（vs 传统流程 2 天）。

**数据要求**：SAP MCP Server、Oracle MCP Server、WMS MCP Server（均需提前配置）
**预期产出**：PO 号 + 供应商确认 + WMS 入库预约 + 完整审计日志
**业务价值**：采购自动化率从 30% → 80%，采购周期从 2 天 → 5 分钟，人工工时节省 60%

**场景B：异常处理的跨系统联动**

物流延误告警触发时，Agent 自动：
- 查询 Neptune 知识图谱（MCP）找可替代供应商
- 查询 SAP 库存判断紧迫性（MCP）
- 如紧急：在 Oracle 发起紧急采购申请（MCP）
- 在 WMS 标记受影响 SKU 为"监控中"（MCP）
- 向相关负责人发告警（MCP Notification）

**数据要求**：物流事件流、各系统 MCP Server
**预期产出**：自动化异常处理报告 + 行动日志 + 升级通知
**业务价值**：异常响应从 3 小时人工协调 → 15 分钟自动处理，防止缺货损失 5-20 万元/次

## ③ 代码模板

```python
import json
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    input_schema: Dict
    handler: Callable  # 实际执行函数

@dataclass
class MCPCallResult:
    """MCP 调用结果"""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

class MCPServer:
    """
    MCP Server 基类
    每个外部系统（SAP/Oracle/WMS）继承此类实现自己的工具集
    """
    
    def __init__(self, server_name: str, system_type: str):
        self.server_name = server_name
        self.system_type = system_type
        self.tools: Dict[str, MCPTool] = {}
        self.call_log: List[MCPCallResult] = []
    
    def register_tool(self, tool: MCPTool):
        self.tools[tool.name] = tool
    
    def call_tool(self, tool_name: str, arguments: Dict) -> MCPCallResult:
        """统一工具调用入口（带延迟测量和日志）"""
        if tool_name not in self.tools:
            result = MCPCallResult(tool_name, False, 
                                   error=f"工具 '{tool_name}' 不存在于 {self.server_name}")
            self.call_log.append(result)
            return result
        
        t0 = time.time()
        try:
            data = self.tools[tool_name].handler(arguments)
            result = MCPCallResult(tool_name, True, data=data,
                                   latency_ms=round((time.time() - t0) * 1000, 2))
        except Exception as e:
            result = MCPCallResult(tool_name, False, error=str(e),
                                   latency_ms=round((time.time() - t0) * 1000, 2))
        
        self.call_log.append(result)
        return result
    
    def list_tools(self) -> List[Dict]:
        return [{"name": t.name, "description": t.description} 
                for t in self.tools.values()]


class SAPMCPServer(MCPServer):
    """SAP ERP MCP Server（模拟）"""
    
    def __init__(self):
        super().__init__("SAP-S4HANA", "ERP")
        self._inventory_db = {}  # 模拟库存
        self._po_counter = 1000
        self._pos = {}
        self._setup_tools()
    
    def seed_inventory(self, sku: str, stock: int, in_transit: int):
        self._inventory_db[sku] = {"stock": stock, "in_transit": in_transit}
    
    def _setup_tools(self):
        self.register_tool(MCPTool(
            name="get_inventory",
            description="查询当前库存和在途数量",
            input_schema={"type": "object", "properties": {
                "sku": {"type": "string", "description": "物料编码"},
            }, "required": ["sku"]},
            handler=self._handle_get_inventory
        ))
        self.register_tool(MCPTool(
            name="create_purchase_order",
            description="创建采购订单",
            input_schema={"type": "object", "properties": {
                "supplier_id": {"type": "string"},
                "sku": {"type": "string"},
                "quantity": {"type": "integer"},
                "delivery_date": {"type": "string"},
                "po_type": {"type": "string", "enum": ["standard", "urgent"]}
            }, "required": ["supplier_id", "sku", "quantity", "delivery_date"]},
            handler=self._handle_create_po
        ))
    
    def _handle_get_inventory(self, args: Dict) -> Dict:
        sku = args["sku"]
        inv = self._inventory_db.get(sku, {"stock": 0, "in_transit": 0})
        return {"sku": sku, "stock": inv["stock"], "in_transit": inv["in_transit"],
                "total_available": inv["stock"] + inv["in_transit"],
                "last_updated": time.strftime("%Y-%m-%d")}
    
    def _handle_create_po(self, args: Dict) -> Dict:
        po_number = f"PO-{self._po_counter}"
        self._po_counter += 1
        self._pos[po_number] = {**args, "status": "created", "created_at": time.time()}
        return {"po_number": po_number, "status": "created", 
                "supplier_id": args["supplier_id"], "sku": args["sku"],
                "quantity": args["quantity"], "estimated_amount_usd": args["quantity"] * 18.5}


class WMSMCPServer(MCPServer):
    """WMS 仓储管理系统 MCP Server（模拟）"""
    
    def __init__(self):
        super().__init__("Manhattan-WMS", "WMS")
        self._slots = {}
        self._setup_tools()
    
    def _setup_tools(self):
        self.register_tool(MCPTool(
            name="reserve_inbound_slot",
            description="预约入库槽位",
            input_schema={"type": "object", "properties": {
                "sku": {"type": "string"},
                "expected_quantity": {"type": "integer"},
                "expected_arrival_date": {"type": "string"},
                "po_reference": {"type": "string"}
            }, "required": ["sku", "expected_quantity", "expected_arrival_date"]},
            handler=self._handle_reserve_slot
        ))
    
    def _handle_reserve_slot(self, args: Dict) -> Dict:
        slot_id = f"SLOT-{len(self._slots) + 1:04d}"
        self._slots[slot_id] = {**args, "status": "reserved"}
        cbm_estimate = args["expected_quantity"] * 0.002  # 估算体积
        return {"slot_id": slot_id, "warehouse": "FBA-PREP-US", 
                "status": "reserved", "cbm_reserved": round(cbm_estimate, 2)}


class SCMCPAgentOrchestrator:
    """
    供应链 Agent 编排器
    管理多个 MCP Server，编排跨系统业务流程
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.workflow_log: List[Dict] = []
    
    def register_server(self, server_id: str, server: MCPServer):
        self.servers[server_id] = server
    
    def call(self, server_id: str, tool_name: str, arguments: Dict) -> MCPCallResult:
        """统一调用入口"""
        server = self.servers.get(server_id)
        if not server:
            return MCPCallResult(tool_name, False, error=f"Server '{server_id}' 未注册")
        return server.call_tool(tool_name, arguments)
    
    def execute_replenishment_workflow(self, sku: str, trigger_qty: int,
                                        supplier_id: str, delivery_date: str) -> Dict:
        """
        补货工作流（跨 SAP + WMS）
        完整的事务性操作：库存检查 → 创建PO → 预约槽位
        """
        workflow_id = f"WF-{int(time.time())}"
        steps = []
        
        # Step 1: 查询当前库存（SAP）
        inv_result = self.call("SAP", "get_inventory", {"sku": sku})
        steps.append({"step": "check_inventory", "server": "SAP", "result": inv_result.success,
                      "data": inv_result.data})
        
        if not inv_result.success:
            return {"workflow_id": workflow_id, "status": "failed", "steps": steps,
                    "error": "库存查询失败"}
        
        # Step 2: 创建采购订单（SAP）
        po_result = self.call("SAP", "create_purchase_order", {
            "supplier_id": supplier_id,
            "sku": sku,
            "quantity": trigger_qty,
            "delivery_date": delivery_date,
            "po_type": "standard"
        })
        steps.append({"step": "create_po", "server": "SAP", "result": po_result.success,
                      "data": po_result.data})
        
        if not po_result.success:
            return {"workflow_id": workflow_id, "status": "failed", "steps": steps,
                    "error": "创建PO失败"}
        
        # Step 3: 预约入库槽位（WMS）
        wms_result = self.call("WMS", "reserve_inbound_slot", {
            "sku": sku,
            "expected_quantity": trigger_qty,
            "expected_arrival_date": delivery_date,
            "po_reference": po_result.data["po_number"]
        })
        steps.append({"step": "reserve_slot", "server": "WMS", "result": wms_result.success,
                      "data": wms_result.data})
        
        success = all(s["result"] for s in steps)
        
        return {
            "workflow_id": workflow_id,
            "status": "completed" if success else "partial_failure",
            "sku": sku,
            "po_number": po_result.data.get("po_number") if po_result.data else None,
            "slot_id": wms_result.data.get("slot_id") if wms_result.data else None,
            "steps_completed": sum(1 for s in steps if s["result"]),
            "total_steps": len(steps),
            "steps_detail": steps
        }
    
    def get_audit_log(self) -> List[Dict]:
        """获取完整审计日志（Palantir Action Audit Trail）"""
        log = []
        for server_id, server in self.servers.items():
            for entry in server.call_log:
                log.append({
                    "server": server_id,
                    "tool": entry.tool_name,
                    "success": entry.success,
                    "latency_ms": entry.latency_ms,
                    "timestamp": entry.timestamp
                })
        return sorted(log, key=lambda x: x["timestamp"])


# ===== 测试用例 =====
def run_test():
    # 初始化系统
    sap = SAPMCPServer()
    sap.seed_inventory("STERILIZER-PRO", stock=120, in_transit=500)
    
    wms = WMSMCPServer()
    
    orchestrator = SCMCPAgentOrchestrator()
    orchestrator.register_server("SAP", sap)
    orchestrator.register_server("WMS", wms)
    
    # Test 1: 库存查询
    inv = orchestrator.call("SAP", "get_inventory", {"sku": "STERILIZER-PRO"})
    assert inv.success, f"库存查询应成功: {inv.error}"
    assert inv.data["total_available"] == 620, f"总可用应为620，实际{inv.data['total_available']}"
    print(f"  库存查询: 在库{inv.data['stock']}件, 在途{inv.data['in_transit']}件")
    
    # Test 2: 完整补货工作流
    result = orchestrator.execute_replenishment_workflow(
        sku="STERILIZER-PRO",
        trigger_qty=2000,
        supplier_id="SUP-SZ-001",
        delivery_date="2026-08-01"
    )
    assert result["status"] == "completed", f"工作流应完成: {result}"
    assert result["po_number"] is not None, "应生成PO号"
    assert result["slot_id"] is not None, "应预约槽位"
    assert result["steps_completed"] == 3, f"应完成3步，实际{result['steps_completed']}"
    
    print(f"  补货工作流: {result['status']}")
    print(f"  PO号: {result['po_number']}, 槽位: {result['slot_id']}")
    print(f"  步骤完成: {result['steps_completed']}/{result['total_steps']}")
    
    # Test 3: 错误处理（调用不存在的工具）
    bad_result = orchestrator.call("SAP", "nonexistent_tool", {})
    assert not bad_result.success, "不存在的工具应返回失败"
    assert bad_result.error is not None, "应有错误信息"
    print(f"  错误处理: {bad_result.error[:50]}...")
    
    # Test 4: 审计日志
    log = orchestrator.get_audit_log()
    assert len(log) >= 3, f"应有至少3条审计记录，实际{len(log)}"
    assert all("server" in entry and "tool" in entry for entry in log), "审计日志格式正确"
    print(f"  审计日志: {len(log)} 条记录")
    
    # Test 5: 工具列表
    tools = sap.list_tools()
    assert len(tools) == 2, f"SAP应有2个工具，实际{len(tools)}"
    print(f"  SAP工具列表: {[t['name'] for t in tools]}")
    
    print("\n[✓] SC-Agent-MCP-ERP-Integration 测试通过 — MCP编排+跨系统工作流+审计日志就绪")

run_test()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supply-Chain-Ontology-Action-Trigger]] — Ontology Action是MCP写回的语义层抽象
- **前置（prerequisite）**：[[Skill-SCPA-Autonomous-SC-Planning-Agent]] — SCPA Agent 通过 MCP 工具调用 ERP
- **延伸（extends）**：[[Skill-Human-in-Loop-Approval-Gate-Tag]] — 高风险 MCP 写操作需要人工审批门控
- **延伸（extends）**：[[Skill-Decision-Audit-Trail-Ontology]] — MCP 调用日志是决策审计的核心数据源
- **可组合（combinable）**：[[Skill-LLM-SC-MultiAgent-Consensus-Replenishment]] — 多智能体共识后通过 MCP 写回 ERP 执行
- **可组合（combinable）**：[[Skill-PO-Exception-Handling-Workflow]] — PO 异常处理工作流通过 MCP 跨系统协调

## ⑤ 商业价值评估

- **ROI 预估**：AWS+Elixir Claw 案例：采购自动化率 30% → 80%（+50pp），集成延迟 <100ms，采购周期 2 天 → 5 分钟；年化节省采购人工成本约 20-50 万元
- **实施难度**：⭐⭐⭐⭐☆（MCP Server 开发标准化，但 ERP API 对接仍有工程挑战）
- **优先级**：⭐⭐⭐⭐⭐（Palantir OSDK 的开源替代方案，解锁 Agent → 实际执行 的最后一公里）
- **企业AI知识库依赖**：高 — MCP 工具注册表是企业 AI 知识库的"行动能力层"，决定 Agent 能做什么

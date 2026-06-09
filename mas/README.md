# MAS 部署与运维指南

> **最新更新**：2026-06-04
> Skill 卡片库当前扫描 **302** 个，MAS 工具注册 **112 个 / 14 个域**，MCP Server 层已落地（4 个 domain server）

母婴跨境电商多智能体系统 (MAS) - 阶段 0~8 MVP 已就绪。

## 集成现状

- **MAS 基础设施**：5 工作流 + 14 模块 + **61/61 测试全绿**（2026-06-04，含治理、增量 workflow 与 Darwin evolution 回归）
- **Skill 工具注册**：**112 个工具 / 14 个域**（SkillRegistry），含 selection 域（新）
- **MCP Server 层**：**4 个 domain server / 28 个工具**（阶段 8 已完成）
  - `supply_chain_server`：9 工具（demand_forecast / safety_stock 接真实实现）
  - `advertising_server`：7 工具（negative_keywords / dara / mmm 接真实实现）
  - `customer_service_server`：6 工具（全部接真实实现）
  - `selection_server`：6 工具（全部接真实实现）
- **工作流覆盖率**：WF-A 95% / WF-B 90% / WF-C 85% / WF-D 80% / WF-E 85%

## 一、架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                  HITL 审批层 (mas/hitl/)                       │
│   approval_api.py · ApprovalStore · feishu_webhook_notifier   │
└──────────┬────────────────────────────────────────────────────┘
           │ interrupt_fn callback
┌──────────▼────────────────────────────────────────────────────┐
│              Orchestrator (mas/agents/orchestrator.py)         │
│   orchestrator_route · human_approval_gate                     │
│   execute_approved_action · handle_rejection                   │
└──┬──────────┬──────────┬───────────┬──────────────┬───────────┘
   │          │          │           │              │
┌──▼──┐ ┌─────▼────┐ ┌───▼─────┐ ┌───▼──────┐ ┌─────▼─────┐
│ Sup │ │Marketing │ │Customer │ │Selection │ │   QA      │
│Chain│ │  Agent   │ │ Service │ │  Agent   │ │  Agent    │
│Agent│ │          │ │  Agent  │ │          │ │           │
└─────┘ └──────────┘ └─────────┘ └──────────┘ └───────────┘
   │          │          │           │              │
┌──▼──────────▼──────────▼───────────▼──────────────▼────────┐
│         Skill 工具层 (mas/skills/)                          │
│  supply_chain_tools / marketing_tools /                     │
│  customer_service_tools / selection_tools / registry        │
└────────────────────────────────────────────────────────────┘
   │
┌──▼─────────────────────────────────────────────────────────┐
│      基础设施 (mas/checkpointing/ + mas/observability/)     │
│  SQLiteCheckpointer (生产可换 PostgresSaver)                │
│  TraceCollector (生产可换 LangSmith)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、5 个端到端工作流

| 工作流 | workflow_type | Agent | 主要 Skill 链 | 典型 HITL 触发条件 |
|---|---|---|---|---|
| WF-A 智能补货 | `restock`/`large_restock`/`price_change` | SupplyChainAgent | 需求预测 → 反事实 GCF → 安全库存 → DRL 补货 | 单 PO > ¥1万 |
| WF-B 广告优化 | `ad_campaign`/`ad_budget_increase`/`promotion_launch` | MarketingAgent | 搜索词清洗 → 否定词 → Uplift 提价 → MMM → DARA 跨渠道 | 单次预算变动 > ¥1万 |
| WF-C 客服分诊 | `customer_ops` | CustomerServiceAgent | 意图分类 → 情感分析 → GraphRAG FAQ → 决策树 | 大额退款 (>¥500) |
| WF-D 选品扫描 | `product_selection` | SelectionAgent | 市场空间 → 毛利 → 合规 → KGQA → Uplift | 自动通过 (cost=0) |
| WF-E Review 监控 | `review_monitor` | CustomerServiceAgent (复用) | Review 主题聚类 → 健康度评分 | 评分 < 4 触发行动 |

---

## 三、本地运行

### 安装

```bash
# 标准库即可,无第三方依赖 (生产部署再装 langgraph/fastapi/etc.)
git clone <repo>
cd paper_to_skills
python3 --version  # 需要 3.12+
```

### 跑全部 53 个集成测试

```bash
python3 -m mas.tests.test_phase0_skeleton     # 12 项 MAS 骨架
python3 -m mas.tests.test_wfa_restock_e2e      # 5 项 WF-A 补货
python3 -m mas.tests.test_wfb_ad_campaign_e2e  # 6 项 WF-B 广告（含工具链断言）
python3 -m mas.tests.test_wfce_customer_ops_e2e # 5 项 WF-C/E 客服
python3 -m mas.tests.test_wfd_selection_e2e    # 5 项 WF-D 选品（含 registry 断言）
python3 -m mas.tests.test_phase6_integration   # 6 项 MAS 集成
python3 -m mas.tests.test_mcp_server_routing   # 8 项 MCP 路由（新）
python3 -m pytest mas/tests/test_governance_regressions.py -q # 6 项治理回归
```

### 入口 demo

```bash
python3 -m mas.main
```

### Python API 调用

```python
from mas.main import MAS
from mas.hitl.approval_api import make_blocking_interrupt

# 1. 同步审批模式 (适合自动化测试)
mas = MAS(interrupt_fn=lambda req: {"action": "approve", "note": "ok"})

# 2. 异步审批模式 (生产推荐,接飞书 webhook)
mas = MAS(interrupt_fn=make_blocking_interrupt())

result = mas.trigger(
    workflow_type="restock",
    operator_id="ops-shen",
    payload={
        "sku_id": "AB-FORMULA-S1",
        "history_daily_sales": [...],
        "current_stock": 800,
        "in_transit": 0,
        "lead_time_days": 35,
        "moq": 1000,
        "unit_cost_rmb": 80.0,
    },
    token_budget=20_000,
)

print(result["final_output"])
print(mas.trace_summary(result["workflow_id"]))
```

---

## 四、生产部署清单

### 必须替换的 stub

| 当前 stub | 生产替换 | 说明 |
|---|---|---|
| `mas/graphs/base_graph.py` 纯 Python StateGraph | `from langgraph.graph import StateGraph` | 接口已对齐,迁移仅需改 import |
| `mas/checkpointing/sqlite_saver.py` | `langgraph.checkpoint.postgres.PostgresSaver` | 改 db_path → POSTGRES_URL |
| `mas/observability/tracer.py` | LangSmith SDK | `langsmith.Client()` |
| `mas/hitl/approval_api.py` ApprovalStore | FastAPI + 飞书 OpenAPI | 配合 `interrupt()` Async |
| `mas/skills/registry.py` SkillTool stub | LangChain `@tool` 装饰器 / MCP Server | 阶段 2+ 切换 |
| `mas/agents/base.py` `_stub_llm` | `langchain_anthropic.ChatAnthropic("claude-opus-4-5")` 或同等 | LLM provider 替换 |

### 推荐 pip 依赖（生产）

```bash
pip install langgraph>=1.1.0 \
            langchain-anthropic \
            langsmith \
            psycopg2-binary \
            fastapi \
            uvicorn \
            requests
```

### 环境变量

```bash
export ANTHROPIC_API_KEY=...
export POSTGRES_URL=postgresql://user:pass@host:5432/mas
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=mas-baby-cross-border
export PAPER2SKILLS_FEISHU_WEBHOOK=https://open.feishu.cn/...
```

### 配置阈值

`mas/agents/orchestrator.py`:
- `HIGH_RISK_THRESHOLD = 10_000.0` 触发 HITL 的金额阈值（人民币）
- `RISKY_WORKFLOW_TYPES` 强制审批的工作流类型集合

`mas/state/schema.py`:
- `init_state()` 默认 `token_budget=50_000`

---

## 五、扩展路线

### 阶段 7：替换为真实 LangGraph 1.1+

1. 在 `requirements.txt` 添加 langgraph
2. 把 `mas/graphs/base_graph.py` 改为 `from langgraph.graph import StateGraph, START, END`
3. `interrupt()` 改用 `langgraph.types.interrupt`
4. 53 个集成测试零修改通过即视为迁移完成

### 阶段 8：MCP Server 化 Skill 工具 ✅ 已完成

`mas/skills/mcp_servers/` 已落地 4 个 domain server + `MultiServerMCPClient`：
- `supply_chain_server.py`（9 工具）/ `advertising_server.py`（7 工具）
- `customer_service_server.py`（6 工具）/ `selection_server.py`（6 工具）
- `client.py`：`MultiServerMCPClient`，工具名 → server 自动路由
- `SkillRegistry.get_mcp_client()` 可直接获取客户端实例

生产迁移：将 `BaseMCPServer` 改继承 `mcp.server.Server`，用 `@server.tool()` 注册。

### 阶段 9：多租户与成本控制

1. WorkflowContext 增加 `tenant_id`
2. SQLite → Postgres 多租户分库
3. tracer 增加 token / cost 看板

---

## 六、当前 MVP 状态

| 维度 | 状态 |
|---|---|
| **集成测试** | ✅ **61/61 全部通过**（骨架/工作流/MCP/工具链 + 治理、增量 workflow、Darwin evolution 回归） |
| **5 个工作流** | ✅ 全部端到端可跑 |
| **HITL 双路径** | ✅ 同步/异步 callback 都验证 |
| **Checkpointing** | ✅ SQLite 持久化跨进程加载 |
| **Tracer** | ✅ token / 事件聚合 |
| **MCP Server 层** | ✅ **4 个 domain server / 28 工具**（阶段 8 已落地） |
| **SkillRegistry** | ✅ **112 工具 / 14 个域**（含 selection 域新增） |
| **依赖** | ✅ 纯 Python 标准库（生产可一键迁 LangGraph） |
| **总代码量** | ~3500 行，涵盖 14 个核心模块 + 7 个 MCP 文件 |

### 已知限制

1. **运行模式默认显式显示为 `{'llm': 'stub', 'mcp': 'mcp_stub'}`** — 61/61 通过不等于真实 LLM/MCP 生产就绪
2. **LLM 仍是 stub** — 生产替换 `BaseAgent._stub_llm` 为真实 ChatAnthropic
3. **Skill 工具是规则版** — 当前是论文公式的简化实现,生产可接真实 Skill 卡片代码
4. **HITL 仅 callback 模式** — 异步审批仍需补 FastAPI + 飞书 OpenAPI
4. **没有并发隔离** — checkpointer SQLite 写入未做行锁,高并发需切 Postgres

---

## 七、文件清单 (14 个核心模块)

```
mas/
├── __init__.py
├── main.py                          # MAS 入口 + 5 工作流注册
├── state/
│   └── schema.py                    # WorkflowContext TypedDict
├── agents/
│   ├── base.py                      # BaseAgent 基类
│   ├── orchestrator.py              # 路由 + HITL gate
│   ├── supply_chain_agent.py        # WF-A
│   ├── marketing_agent.py           # WF-B
│   ├── customer_service_agent.py    # WF-C / WF-E
│   ├── selection_agent.py           # WF-D
│   └── qa_agent.py                  # 横切质检
├── skills/
│   ├── registry.py                  # SkillRegistry (112 工具 / 14 域)
│   ├── supply_chain_tools.py        # WF-A 业务工具
│   ├── marketing_tools.py           # WF-B 业务工具
│   ├── customer_service_tools.py    # WF-C/E 业务工具
│   ├── selection_tools.py           # WF-D 业务工具
│   └── mcp_servers/                 # 阶段 8 MCP Server 层（已落地）
│       ├── base.py                  # BaseMCPServer 统一接口
│       ├── client.py                # MultiServerMCPClient 路由
│       ├── supply_chain_server.py   # 9 工具
│       ├── advertising_server.py    # 7 工具
│       ├── customer_service_server.py # 6 工具
│       ├── selection_server.py      # 6 工具
│       └── README.md                # MCP 接口规范 + 生产迁移路径
├── graphs/
│   ├── base_graph.py                # StateGraph + RetryPolicy
│   ├── restock_graph.py             # WF-A
│   ├── ad_campaign_graph.py         # WF-B
│   ├── customer_ops_graph.py        # WF-C / WF-E
│   └── selection_graph.py           # WF-D
├── hitl/
│   └── approval_api.py              # 审批 stub
├── checkpointing/
│   └── sqlite_saver.py              # SQLite 持久化
├── observability/
│   └── tracer.py                    # in-memory tracer
└── tests/
    ├── test_phase0_skeleton.py
    ├── test_wfa_restock_e2e.py
    ├── test_wfb_ad_campaign_e2e.py
    ├── test_wfce_customer_ops_e2e.py
    ├── test_wfd_selection_e2e.py
    ├── test_phase6_integration.py
    └── test_mcp_server_routing.py   # 阶段 8 MCP 路由测试（新）
```
